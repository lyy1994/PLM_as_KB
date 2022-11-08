#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from itertools import chain

import torch
import torch.nn.functional as F
from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.agents.hugging_face.dialogpt import DialoGPTDecoder, DialoGPTModel, DialogptAgent
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE, TOKEN_NOCHOSEN
from parlai.utils.misc import warn_once
from parlai.core.metrics import AverageMetric
from parlai.core.torch_generator_agent import PPLMetric
from parlai.core.torch_agent import Batch
from parlai.utils.torch import neginf, padded_tensor

try:
    from transformers import GPT2Model
except ImportError:
    raise ImportError('Please run `pip install transformers`.')


############################################
## Modules
############################################


class Selector(torch.nn.Module):

    def __init__(self, opt, dict):
        super().__init__()
        self.model = self._get_model(opt, dict)
        self.config = self.model.transformer.config
        self.knowledge_topk = opt['knowledge_topk']
        self.hist_proj = torch.nn.Linear(self.config.n_embd, self.config.n_embd)
        self.know_proj = torch.nn.Linear(self.config.n_embd, self.config.n_embd)

    def _get_model(self, opt, dict):
        return DialoGPTDecoder(opt, dict)

    def forward(self, *xs):
        """
        xs:
            history: LongTensor[bsz, seqlen]
            knowledge: LongTensor[bsz, n_cands, seqlen]
        """
        history, knowledge = xs

        bsz, n_cands = knowledge.size()[:2]
        _knowledge = knowledge.view(bsz * n_cands, -1)
        output, _ = self.model(_knowledge, torch.full((bsz * n_cands, 1), self.model.NULL_IDX, dtype=torch.long, device=_knowledge.device))
        output = output.view(bsz, n_cands, output.size(1), output.size(2))
        # Since we are using GPT-2 instead of Plato-2, it does not have [CLS] label.
        # Here we use the last token instead.
        know_embed = output[:, :, -1, :]  # [bsz, n_cands, n_embd]

        output, _ = self.model(history, torch.full((bsz, 1), self.model.NULL_IDX, dtype=torch.long, device=history.device))
        hist_embed = output[:, -1, :]  # [bsz, n_embd]

        logit = torch.einsum('bh,bnh->bn', self.hist_proj(hist_embed), self.know_proj(know_embed))
        values, indices = torch.topk(logit, self.knowledge_topk, dim=-1)
        doc_probs = F.log_softmax(values, dim=-1)
        new_knowledge = torch.gather(knowledge, 1, indices.unsqueeze(-1).repeat(1, 1, knowledge.size(-1)))

        metrics = {
            'f_at_k_acc': (indices == 0).float().sum(dim=-1),
            # the last knowledge might be no_passage_used due to padding
            # but in this case we always use all knowledge, so it is fine
            'l_at_k_acc': (indices == (logit.size(1) - 1)).float().sum(dim=-1),
        }
        return history, new_knowledge, doc_probs, metrics


class Generator(DialoGPTDecoder):

    def forward(self, input, encoder_state, incr_state=None):
        """
        encoder_state: tuple
            history: LongTensor[bsz, seqlen]
            knowledge: LongTensor[bsz, n_docs, seqlen]
            doc_probs: FloatTensor[bsz, n_docs]
        """
        history, knowledge, doc_probs, _ = encoder_state
        new_encoder_state = torch.cat([
            history.unsqueeze(1).expand(-1, knowledge.size(1), -1),
            knowledge,
        ], dim=-1)
        bsz, n_cands, seqlen = new_encoder_state.size()
        new_encoder_state = new_encoder_state.view(bsz * n_cands, seqlen)
        new_input = input.unsqueeze(1).repeat(1, knowledge.size(1), 1).reshape(bsz * n_cands, -1)
        output, new_incr_state = super().forward(new_input, new_encoder_state, incr_state)
        output = output.view(bsz, n_cands, output.size(1), output.size(2))
        length = (input != self.NULL_IDX).float().sum(dim=-1)
        return (output, doc_probs, length), new_incr_state


class PlatoKagModel(DialoGPTModel):
    """
    Our implementation of Plato-Kag. It is based on DialoGPT, as their Plato-2
    is not available in Huggingface.

    PLATO-KAG: Unsupervised Knowledge-Grounded Conversation via Joint Modeling
    https://aclanthology.org/2021.nlp4convai-1.14/
    """

    def __init__(self, opt, dict):
        super().__init__(opt, dict)
        self.encoder = self._get_encoder(opt, dict)

    def _get_encoder(self, opt, dict):
        return Selector(opt, dict)

    def _get_decoder(self, opt, dict):
        return Generator(opt, dict)

    def output(self, tensor):
        """
        tensor: Optional[tuple, FloatTensor[bsz, n_docs, seqlen, n_embd]]
            hidden: FloatTensor[bsz, n_docs, seqlen, n_embd]
            doc_probs: FloatTensor[bsz, n_docs]
            length: FloatTensor[bsz]
        """
        if isinstance(tensor, tuple):
            # training (output is log probability)
            hidden, doc_probs, length = tensor
            token_probs = F.log_softmax(self.lm_head(hidden), dim=-1, dtype=torch.float32)
            # token_probs = token_probs / length.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).type_as(token_probs)
            # marginalize, similar to RAG-Sequence
            doc_probs = doc_probs.unsqueeze(-1).unsqueeze(-1)
            first_token_scores = token_probs[:, :, :1, :]
            remainder = token_probs[:, :, 1:, :]
            output = torch.cat([first_token_scores + doc_probs, remainder], dim=2)
        else:
            # inference (output is logit)
            output = self.lm_head(tensor)
        return output

    def reorder_encoder_states(self, encoder_states, indices):
        history, knowledge, doc_probs, metrics = encoder_states
        hist = torch.index_select(history, 0, indices)
        know = torch.index_select(knowledge, 0, indices)
        probs = torch.index_select(doc_probs, 0, indices)
        m = {}
        for k, v in metrics.items():
            m[k] = torch.index_select(v, 0, indices)
        return (hist, know, probs, m)

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        new_incr_state = []
        for layer_past in incremental_state:
            if not torch.is_tensor(layer_past):
                # newer versions of HF split up the intermediate outputs
                assert isinstance(layer_past, tuple)
                layer_past = torch.stack(layer_past, dim=0)
            # we combine n_docs with bsz for processing, so we need to handle them separately
            new_inds = inds.unsqueeze(1).repeat(1, layer_past.size(1) // inds.size(0)).reshape(-1)
            new_incr_state.append(torch.index_select(layer_past, 1, new_inds))

        return tuple(new_incr_state)


############################################
## Agent
############################################


class PlatoKagAgent(DialogptAgent):

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        agent = parser.add_argument_group('Plato-Kag Args')
        agent.add_argument(
            '--knowledge-topk',
            type=int,
            default=8,
            help='Number of knowledge to be reserved.',
        )
        agent.add_argument(
            '--max-knowledge',
            type=int,
            default=32,
            help='Number of knowledge to be selected.',
        )
        agent.add_argument(
            '--knowledge-truncate',
            type=int,
            default=128,
            help='Number of tokens per knowledge.',
        )
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        warn_once('WARNING: this model is in beta and the API is subject to change.')
        return agent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.max_knowledge = opt['max_knowledge']
        self.knowledge_truncate = opt['knowledge_truncate']
        self.knowledge_topk = opt['knowledge_topk']

    def _model_input(self, batch):
        return (batch.text_vec, batch.knowledge_vec,)

    def _encoder_input(self, batch):
        return self._model_input(batch)

    def build_model(self, states=None):
        return PlatoKagModel(self.opt, self.dict)

    def _parse_knowledge(self, obs):
        if 'knowledge_parsed' in obs:
            # make a copy of the list to prevent the future padding step from
            # being destructive
            return list(obs['knowledge_parsed'])

        if 'checked_sentence' not in obs:
            # interactive time. we're totally on our own
            obs_know = [
                k.strip() for k in obs.get('knowledge', 'no_passages_used').split('\n')
            ]
            obs_know = [k for k in obs_know if k]
            obs['knowledge_parsed'] = obs_know
            return obs['knowledge_parsed']

        checked_sentence = '{} {} {}'.format(
            obs['title'], TOKEN_KNOWLEDGE, obs['checked_sentence']
        )
        # grab all the nonempty knowledge
        obs_know = [
            k.strip() for k in obs.get('knowledge', 'no_passages_used').split('\n')
        ]
        obs_know = [k for k in obs_know if k]

        # we want the correct knowledge to always be in index 0
        try:
            i = obs_know.index(checked_sentence)
        except ValueError:
            # uh oh, couldn't find the sentence in the knowledge. This happens for
            # one or two examples in the training set. We can just artificially
            # put it back in
            i = 0
            obs_know[0] = checked_sentence
        obs_know[0], obs_know[i] = obs_know[i], obs_know[0]

        obs['knowledge_parsed'] = obs_know
        obs['checked_sentence_parsed'] = checked_sentence
        return obs['knowledge_parsed']

    def batchify(self, obs_batch, sort=True):
        """
        Add a new 'knowledge' field to Batch.
        """
        batch = super().batchify(obs_batch, sort=sort)
        reordered_observations = [obs_batch[i] for i in batch.valid_indices]
        is_training = 'labels' in reordered_observations[0]

        # first parse and compile all the knowledge together
        all_knowledges = []  # list-of-lists knowledge items for each observation
        knowledge_counts = []  # how much knowledge each observation gets
        for obs in reordered_observations:
            obs_know = self._parse_knowledge(obs)
            # downsample if desired
            if (
                    is_training
                    and self.max_knowledge
                    and len(obs_know) > self.max_knowledge
            ):
                # offset by one so that we don't choose 0
                keepers = 1 + np.random.choice(
                    len(obs_know) - 1, self.max_knowledge, False
                )
                # correct answer is always the first one
                keepers[0] = 0
                obs_know = [obs_know[i] for i in keepers]
            if 'response' in obs:
                obs_know.extend(obs.get('response'))
            all_knowledges.append(obs_know)
            knowledge_counts.append(len(obs_know))

        # now we want to actually pack this into a tensor, along with the mask
        N = len(reordered_observations)
        K = max(max(knowledge_counts), self.knowledge_topk)
        # round out the array so everything is equally sized
        for i in range(N):
            all_knowledges[i] += [TOKEN_NOCHOSEN] * (K - knowledge_counts[i])
        flattened_knowledge = list(chain(*all_knowledges))

        knowledge_vec = [
            self._vectorize_text(
                # the beginning of the sentence is more useful
                k,
                truncate=self.knowledge_truncate,
                add_end=True,
                truncate_left=False,
            )
            for k in flattened_knowledge
        ]
        knowledge_vec, _ = padded_tensor(
            knowledge_vec, pad_idx=self.NULL_IDX, left_padded=True
        )
        knowledge_vec[:, -1] = self.END_IDX
        T = knowledge_vec.size(-1)
        knowledge_vec = knowledge_vec.view(N, K, T)

        batch['knowledge_vec'] = knowledge_vec  # bsz, n_cands, seqlen
        return batch

    def compute_loss(self, batch, return_output=False):
        """
        Plato-Kag is trained with a loss similar to RAG-Sequence,
        so we adopt their implementation here (_rag_sequence_loss).
        """
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        model_output = self.model(*self._model_input(batch), ys=batch.label_vec)
        scores, preds, encoder_states = model_output

        # compute rag sequence loss
        n_docs = scores.size(1)
        target = batch.label_vec.unsqueeze(1).unsqueeze(-1).repeat(1, n_docs, 1, 1)
        ll = scores.gather(dim=-1, index=target)
        pad_mask = target.eq(self.NULL_IDX)
        if pad_mask.any():
            ll.masked_fill_(pad_mask, 0.0)
        ll = ll.squeeze(-1)
        ll = ll.sum(2)  # sum over tokens
        ll = ll.logsumexp(1)  # sum over docs
        nll_loss = -ll
        loss = nll_loss

        # save loss to metrics
        notnull = batch.label_vec.ne(self.NULL_IDX)
        target_tokens = notnull.long().sum(dim=-1)

        # cross entropy loss
        self.record_local_metric('loss', AverageMetric.many(loss.tolist(), target_tokens))
        # perplexity
        self.record_local_metric('ppl', PPLMetric.many(loss.tolist(), target_tokens))
        # knowledge accuracy (checked_sentence accuracy)
        metrics = encoder_states[3]
        f_at_k_acc = metrics['f_at_k_acc']
        l_at_k_acc = metrics['l_at_k_acc']
        self.record_local_metric('f_at_k_acc', AverageMetric.many(f_at_k_acc.tolist(), torch.ones_like(f_at_k_acc)))
        self.record_local_metric('l_at_k_acc', AverageMetric.many(l_at_k_acc.tolist(), torch.ones_like(l_at_k_acc)))
        # actually do backwards loss
        loss = loss.sum()
        loss /= target_tokens.sum()  # average loss per token
        if return_output:
            return (loss, model_output)
        else:
            return loss

    def _generate(
        self,
        batch: Batch,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
    ):
        """
        Most code follows the original _generate.
        """
        model = self.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        encoder_states = model.encoder(*self._encoder_input(batch))
        if batch.text_vec is not None:
            dev = batch.text_vec.device
        else:
            assert batch.label_vec is not None, "need label_vec for _generate"
            dev = batch.label_vec.device

        bsz = batch.batchsize
        if batch.text_vec is not None:
            batchsize = batch.batchsize
            batch_context_list = self._get_batch_context(batch).tolist()
            beams = [
                self._treesearch_factory(dev)
                .set_batch_context(batch_context_list, batch_idx)
                .set_block_list(self.beam_block_list)
                for batch_idx in range(batchsize)
            ]
        else:
            beams = [self._treesearch_factory(dev) for _ in range(bsz)]

        # repeat encoder outputs and decoder inputs
        decoder_input = self._get_initial_decoder_input(bsz, beam_size, dev)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).repeat(1, beam_size).view(-1)
        encoder_states = model.reorder_encoder_states(encoder_states, inds)
        incr_state = None

        for _ts in range(max_ts):
            if all((b.is_done() for b in beams)):
                # exit early if possible
                break

            ################################### Plato-Kag ###########################################
            # we do not know how Plato-Kag performs inference exactly,
            # we assume it does something like ensemble instead of reranking
            # like RAG-Sequence.
            (score, doc_probs, length), incr_state = model.decoder(decoder_input, encoder_states, incr_state)
            # only need the final hidden state to make the word prediction
            score = score[:, :, -1:, :]
            score = model.output(score)  # logit
            if self.temperature != 1.0:
                score.div_(self.temperature)
            # force to fp32 to avoid overflow issues during search calculations
            score = torch.softmax(score, dim=-1, dtype=torch.float32)
            score = (score * doc_probs.unsqueeze(-1).unsqueeze(-1).exp()).sum(dim=1)
            score = torch.log(score)  # bsz * beam_size, 1, vocab_size
            #########################################################################################

            # score contains softmax scores for bsz * beam_size samples
            score = score.view(bsz, beam_size, -1)
            if prefix_tokens is not None and _ts < prefix_tokens.size(1):
                # generate prefix_tokens for every timestep that they exist
                # achieve by setting score of all other tokens to be -inf
                prefix_toks = prefix_tokens[:, _ts]
                prefix_mask = torch.ones_like(score, dtype=torch.bool)
                prefix_mask[
                    :, :, prefix_toks
                ] = False  # everything except prefix toks should be neginf
                score[prefix_mask] = neginf(score.dtype)
            for i, b in enumerate(beams):
                if not b.is_done():
                    b.advance(score[i])
            incr_state_inds = torch.cat(
                [
                    beam_size * i + b.get_backtrack_from_current_step()
                    for i, b in enumerate(beams)
                ]
            )
            incr_state = model.reorder_decoder_incremental_state(
                incr_state, incr_state_inds
            )
            selection = torch.cat(
                [b.get_output_from_current_step() for b in beams]
            ).unsqueeze(-1)
            decoder_input = self._get_next_decoder_input(
                decoder_input, selection, incr_state_inds
            )

        # get all finalized candidates for each sample (and validate them)
        n_best_beam_preds_scores = [b.get_rescored_finished() for b in beams]

        if hasattr(self, '_rerank_beams'):
            n_best_beam_preds_scores = self._rerank_beams(  # type: ignore
                batch, n_best_beam_preds_scores
            )

        # get the top prediction for each beam (i.e. minibatch sample)
        beam_preds_scores = [n_best_list[0] for n_best_list in n_best_beam_preds_scores]

        return beam_preds_scores, beams
