#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.utils.misc import warn_once
from parlai.core.torch_agent import Batch
from parlai.utils.torch import neginf
from projects.plm_as_kb.plato_kag import Selector, Generator, PlatoKagModel, PlatoKagAgent


############################################
## Modules
############################################


class GumbelTopK(torch.nn.Module):
    """
    Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement
    https://arxiv.org/pdf/1903.06059.pdf
    """

    def __init__(self, tau: float = 1):
        super().__init__()
        self.tau = tau  # the lower the value is, the closer to top-k it is

    def forward(self, logits: torch.Tensor, k: int, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            gumbels = (
                -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
            )  # ~Gumbel(0,1)
            gumbels = (logits + gumbels) / self.tau  # ~Gumbel(logits,tau)
        else:
            gumbels = logits
        return torch.topk(gumbels, k, dim=dim)


class SelectorPlus(Selector):

    def __init__(self, opt, dict):
        super().__init__(opt, dict)
        self.pooler_type = opt['pooler_type']
        self.reweigh_type = opt['reweigh_type']
        self.reweigh_temperature = opt['reweigh_temperature']
        self.num_ref = 0 if 'model_response_names' not in opt.keys() else len(opt['model_response_names'].split(','))
        if opt['gumbel_topk']:
            self.topk = GumbelTopK()
        else:
            self.topk = torch.topk

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
        # Since we are using DialoGPT instead of Plato-2, it does not have [CLS] label.
        # Here we use the last token instead.
        if self.pooler_type == 'last':
            know_embed = output[:, :, -1, :]  # [bsz, n_cands, n_embd]
        elif self.pooler_type == 'max':
            know_embed = torch.max(output, 2)[0]
        else:
            raise NotImplementedError(f"--pooler-type='{self.pooler_type}'")

        output, _ = self.model(history, torch.full((bsz, 1), self.model.NULL_IDX, dtype=torch.long, device=history.device))
        hist_embed = output[:, -1, :]  # [bsz, n_embd]

        if not self.training and self.reweigh_type != 'none':
            ref_embed = know_embed[:, -self.num_ref:, :]
            know_embed = know_embed[:, :-self.num_ref, :]
        logit = torch.einsum('bh,bnh->bn', self.hist_proj(hist_embed), self.know_proj(know_embed))
        if not self.training and self.reweigh_type == 'additive':
            weight = torch.einsum('bmh,bnh->bmn', self.hist_proj(ref_embed), self.know_proj(know_embed))
            weight = weight.sum(1)
            logit = self.reweigh_temperature * logit + (1. - self.reweigh_temperature) * weight
        values, indices = self.topk(logit, self.knowledge_topk, dim=-1)
        doc_probs = F.log_softmax(values, dim=-1)
        if not self.training and self.reweigh_type == 'multiplicative':
            weight = torch.einsum('bmh,bnh->bmn', self.hist_proj(ref_embed), self.know_proj(know_embed))
            weight = torch.gather(weight, 2, indices.unsqueeze(1).repeat(1, self.num_ref, 1)) / self.reweigh_temperature
            doc_probs += F.log_softmax(weight, dim=-1).sum(dim=1)
        new_knowledge = torch.gather(knowledge, 1, indices.unsqueeze(-1).repeat(1, 1, knowledge.size(-1)))

        metrics = {
            'f_at_k_acc': (indices == 0).float().sum(dim=-1),
            # the last knowledge might be no_passage_used due to padding
            # but in this case we always use all knowledge, so it is fine
            'l_at_k_acc': (indices == (logit.size(1) - 1)).float().sum(dim=-1),
        }
        return history, new_knowledge, doc_probs, metrics


class GeneratorPlus(Generator):

    def __init__(self, opt, dict):
        super().__init__(opt, dict)


class PlatoKagPlusModel(PlatoKagModel):

    def _get_encoder(self, opt, dict):
        return SelectorPlus(opt, dict)

    def _get_decoder(self, opt, dict):
        return GeneratorPlus(opt, dict)


############################################
## Agent
############################################


class PlatoKagPlusAgent(PlatoKagAgent):

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        agent = parser.add_argument_group('Plato-Kag Plus Args')
        agent.add_argument(
            '--gumbel-topk',
            type='bool',
            default=False,
            help='Use gumbel top-k or standard top-k in knowledge selection.',
        )
        agent.add_argument(
            '--pooler-type',
            type=str,
            default='last',
            choices=['last', 'max'],
            help='The way to construct fixed length representation for knowledge.',
        )
        agent.add_argument(
            '--ensemble-type',
            type=str,
            default='prior',
            choices=['prior', 'posterior', 'uniform'],
            help='The way to weigh the models given different knowledge.',
        )
        agent.add_argument(
            '--likelihood-sharpness',
            type=float,
            default=1.,
            help='The sharpness of the likelihood if reweighing is applied',
        )
        agent.add_argument(
            '--posterior-sharpness',
            type=float,
            default=1.,
            help='The sharpness of the knowledge selection distribution.',
        )
        agent.add_argument(
            '--reweigh-type',
            type=str,
            default='none',
            choices=['none', 'additive', 'multiplicative'],
            help='The way to reweigh the knowledge selection using pseudo reference.',
        )
        agent.add_argument(
            '--reweigh-temperature',
            type=float,
            default=1.,
            help='The temperature for reweighing.',
        )
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        warn_once('WARNING: this model is in beta and the API is subject to change.')
        return agent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.ensemble_type = opt['ensemble_type']
        self.likelihood_sharpness = opt['likelihood_sharpness']
        self.posterior_sharpness = opt['posterior_sharpness']
        if opt['reweigh_type'] != "none":
            # a hack to ensure at least k knowledge exists if pseudo reference is used
            self.knowledge_topk += len(opt['model_response_names'].split(","))

    def build_model(self, states=None):
        return PlatoKagPlusModel(self.opt, self.dict)

    def _knowledge_log_likelihood(
        self,
        batch: Batch,
        encoder_states: Tuple,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
    ):
        from parlai.core.torch_generator_agent import GreedySearch

        model = self.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        if batch.text_vec is not None:
            dev = batch.text_vec.device
        else:
            assert batch.label_vec is not None, "need label_vec for _generate"
            dev = batch.label_vec.device

        bsz = batch.batchsize
        history, knowledge, doc_probs, metrics = encoder_states
        n_docs = knowledge.size(1)
        new_history = history.unsqueeze(1).repeat(1, n_docs, 1).view(bsz * n_docs, -1)
        new_knowledge = knowledge.view(bsz * n_docs, -1).unsqueeze(1)
        new_doc_probs = doc_probs.view(bsz * n_docs, -1)
        new_metrics = {}
        for k, v in metrics.items():
            new_metrics[k] = v.unsqueeze(1).repeat(1, n_docs).view(bsz * n_docs)
        new_encoder_states = (new_history, new_knowledge, new_doc_probs, new_metrics)
        bsz = bsz * n_docs

        _treesearch_factory = lambda device: GreedySearch(
                1,
                min_length=0,
                block_ngram=self.beam_block_ngram,
                context_block_ngram=self.beam_context_block_ngram,
                length_penalty=self.opt.get('beam_length_penalty', 0.65),
                padding_token=self.NULL_IDX,
                bos_token=self.START_IDX,
                eos_token=self.END_IDX,
                device=device,
            )
        if batch.text_vec is not None:
            batchsize = batch.batchsize * n_docs
            batch_context_list = self._get_batch_context(batch).unsqueeze(1).repeat(1, n_docs, 1).view(batchsize, -1).tolist()
            beams = [
                _treesearch_factory(dev)
                .set_batch_context(batch_context_list, batch_idx)
                .set_block_list(self.beam_block_list)
                for batch_idx in range(batchsize)
            ]
        else:
            beams = [_treesearch_factory(dev) for _ in range(bsz)]

        # repeat encoder outputs and decoder inputs
        decoder_input = self._get_initial_decoder_input(bsz, 1, dev)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).view(-1)
        new_encoder_states = model.reorder_encoder_states(new_encoder_states, inds)
        incr_state = None

        for _ts in range(max_ts):
            if all((b.is_done() for b in beams)):
                # exit early if possible
                break

            (score, doc_probs, length), incr_state = model.decoder(decoder_input, new_encoder_states, incr_state)
            # only need the final hidden state to make the word prediction
            score = score[:, 0, -1:, :]  # [bsz * n_docs, 1, vocab]
            score = model.output(score)
            if self.temperature != 1.0:
                score.div_(self.temperature)
            # force to fp32 to avoid overflow issues during search calculations
            score = F.log_softmax(score, dim=-1, dtype=torch.float32)  # type: ignore
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
                    1 * i + b.get_backtrack_from_current_step()
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

        preds, _ = zip(*beam_preds_scores)

        lengths = torch.tensor([(pred == self.END_IDX).nonzero(as_tuple=True)[0] + 1 for pred in preds]).to(dev)
        all_scores = [
            torch.tensor(b.all_scores[1:lengths[i]], dtype=torch.float64, device=dev) -
            torch.tensor(b.all_scores[:lengths[i] - 1], dtype=torch.float64, device=dev)
            for i, b in enumerate(beams)
        ]  # beam.all_scores are accumulated log probs for all time steps
        scores = torch.tensor([scores.exp().mean() for scores in all_scores]).to(dev).view(batch.batchsize, n_docs)
        if self.likelihood_sharpness != 1.:
            scores = scores ** self.likelihood_sharpness
            scores = scores / scores.sum(dim=1)
        scores = scores.log().float()

        return scores

    def _log_posterior(self, log_likelihood, log_prior):
        log_likelihood = log_likelihood.float()
        log_prior = log_prior.float()
        _unnormalized_log_posterior = log_likelihood + log_prior
        _log_posterior = _unnormalized_log_posterior - torch.logsumexp(_unnormalized_log_posterior, dim=1, keepdim=True)
        return _log_posterior

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
        if self.ensemble_type == 'prior':
            encoder_states = encoder_states
        elif self.ensemble_type == 'posterior':
            history, knowledge, doc_probs, metrics = encoder_states
            log_likelihood = self._knowledge_log_likelihood(batch, encoder_states, max_ts, prefix_tokens)
            post_doc_probs = self._log_posterior(log_likelihood, doc_probs) / self.posterior_sharpness
            encoder_states = (history, knowledge, post_doc_probs, metrics)
        elif self.ensemble_type == 'uniform':
            history, knowledge, doc_probs, metrics = encoder_states
            encoder_states = (history, knowledge, torch.ones_like(doc_probs) / doc_probs.shape[1], metrics)
        else:
            raise NotImplementedError(f"--ensemble-type {self.ensemble_type}")

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
