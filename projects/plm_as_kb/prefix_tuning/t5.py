#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Tuple, Any, Dict

import torch
import transformers

from parlai.core.opt import Opt
from parlai.core.torch_agent import Batch
from parlai.core.params import ParlaiParser
from parlai.agents.hugging_face.t5 import (
    set_device,
    ParlaiT5Model,
    T5Agent,
    TASK_CONFIGS,
)

from projects.plm_as_kb.prefix_tuning.utils.modeling_t5 import (
    T5ForConditionalGeneration,
    T5Stack,
)


HF_VERSION = (
    int(transformers.__version__.split('.')[0]),
    int(transformers.__version__.split('.')[1]),
)


def check_hf_version(v: Tuple[int, int]) -> bool:
    """
    Check that HF version is greater than 4.3.
    """
    main, sub = v
    return main > 4 or (main == 4 and sub >= 3)


def build_t5(opt: Opt) -> T5ForConditionalGeneration:
    if not check_hf_version(HF_VERSION):
        raise RuntimeError('Must use transformers package >= 4.3 to use t5')
    return T5ForConditionalGeneration.from_pretrained(
        opt['t5_model_arch'], dropout_rate=opt['t5_dropout']
    )


class PTT5Agent(T5Agent):

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        group = parser.add_argument_group('Prefix-Tuning T5 Args')
        group.add_argument(
            "--preseqlen",
            type=int,
            default=0,
            help="preseqlen for how many tokens of prefix should we include.",
        )
        group.add_argument(
            "--mid-dim",
            type=int,
            default=512,
            help="the mid dim.",
        )
        group.add_argument(
            "--prefix-dropout",
            type=float,
            default=0.0,
            help="dropout rate for the prefix tuning model.",
        )
        return parser

    def build_model(self) -> 'PTParlaiT5Model':
        """
        Build and return model.
        """
        model = PTParlaiT5Model(self.opt, self.dict)
        if self.opt['t5_model_parallel']:
            model.t5.parallelize()
        return model

    def _generate(
        self,
        batch: Batch,
        beam_size: int,
        max_ts: int,
        prefix_tokens: Optional[torch.LongTensor] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate an output with beam search.

        Use HF's built-in generation to perform beam search.
        """
        model = self.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        bad_words_ids = None
        if self.beam_block_list is not None:
            bad_words_ids = [
                gram for _, ngram in self.beam_block_list.items() for gram in ngram
            ]

        method = self.opt.get('inference', 'greedy')

        mask = batch.text_vec != self.NULL_IDX
        mask = torch.cat(
            [torch.ones(mask.size(0), self.opt['preseqlen'], dtype=mask.dtype, device=mask.device), mask], dim=1)

        generation_params = {
            'input_ids': batch.text_vec,
            'max_length': max_ts,
            'min_length': self.beam_min_length,
            'do_sample': self.opt['inference'] in ['topk', 'topp'],
            'early_stopping': None,
            'num_beams': beam_size,
            'temperature': self.temperature,
            'top_k': self.opt['topk'] if method in ['topk', 'delayedbeam'] else None,
            'top_p': self.opt['topp'] if method == 'nucleus' else None,
            'repetition_penalty': None,
            'bad_words_ids': bad_words_ids if bad_words_ids else None,
            'bos_token_id': self.START_IDX,
            'pad_token_id': self.NULL_IDX,
            'eos_token_id': self.END_IDX,
            'length_penalty': self.opt['beam_length_penalty'],
            'no_repeat_ngram_size': self.beam_block_ngram,
            'num_return_sequences': None,
            'attention_mask': mask,
            'decoder_start_token_id': self.NULL_IDX,
            'prefix_key_value': model.prefix_tuning_encoder(batch.text_vec.size(0)),
            'decoder_prefix_key_value': model.prefix_tuning_decoder(batch.text_vec.size(0) * beam_size),
        }

        if self.opt['t5_generation_config']:
            config = TASK_CONFIGS[self.opt['t5_generation_config']]
            config.pop('prefix', None)
            generation_params.update(config)
        if overrides:
            generation_params.update(overrides)

        outputs = model.t5.generate(**generation_params)
        outputs = [(outputs[i], 0) for i in range(outputs.size(0))]
        return outputs, []


##############
# T5 Modules #
##############


class PTParlaiT5Encoder(torch.nn.Module):
    def __init__(self, opt: Opt, encoder: T5Stack, padding_idx: Optional[int] = None):
        super().__init__()
        self.stack = encoder
        self.padding_idx = padding_idx
        self.paralleled = not opt[
            't5_model_parallel'
        ]  # need to parallel in forward; bug in HF

    @set_device
    def forward(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
        prefix_key_value: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param LongTensor[batch,seqlen] positions:
            Positions for input IDs
        :param LongTensor[batch,seqlen] segments:
            If provided, additionally adds ``segments`` as extra embedding features.
        :param Tuple[Tuple[LongTensor[batch,n_head,seqlen,head_dim] x kv] x n_layer] prefix_key_value:
        """
        if not self.paralleled:
            self.stack.parallelize()
        mask = input != self.padding_idx
        if prefix_key_value is not None:
            mask = torch.cat([torch.ones(mask.size(0), prefix_key_value[0][0].size(2), dtype=mask.dtype, device=mask.device), mask], dim=1)
        outputs = self.stack(input, attention_mask=mask, output_hidden_states=False, prefix_key_value=prefix_key_value)
        for k in outputs:
            if torch.is_tensor(outputs[k]):
                outputs[k] = outputs[k].to(input.device)
        if prefix_key_value is not None:
            mask = mask[:, prefix_key_value[0][0].size(2):]
        return outputs[0], mask


class PTParlaiT5Decoder(torch.nn.Module):
    def __init__(self, opt: Opt, decoder: T5Stack, padding_idx: Optional[int] = None):
        super().__init__()
        self.stack = decoder
        self.padding_idx = padding_idx
        self.paralleled = not opt[
            't5_model_parallel'
        ]  # need to parallel in forward; bug in HF

    @set_device
    def forward(
        self, input: torch.LongTensor, encoder_state: Tuple[Any], prefix_key_value=None, incr_state=None
    ):
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The decoder inputs (partial or full decoded token IDs).
        :param encoder_state:
            Output from the encoder module forward pass.
        :param prefix_key_value:
            Keys and values of the prefix.
        :param incr_state:
            The incremental state: a dictionary whose keys index the layers and whose
            values contain the incremental state for each layer.
        """
        if not self.paralleled:
            self.stack.parallelize()
        encoder_output, encoder_mask = encoder_state

        mask = input != self.padding_idx
        mask[:, 0] = True  # first token is pad

        if prefix_key_value is not None:
            mask = torch.cat([torch.ones(mask.size(0), prefix_key_value[0][0].size(2), dtype=mask.dtype, device=mask.device), mask], dim=1)
            encoder_mask = torch.cat([torch.ones(encoder_mask.size(0), prefix_key_value[0][2].size(2), dtype=encoder_mask.dtype, device=encoder_mask.device), encoder_mask], dim=1)

        outputs = self.stack(
            input_ids=input,
            attention_mask=mask,
            encoder_hidden_states=encoder_output.to(input.device),
            encoder_attention_mask=encoder_mask.to(input.device),
            decoder_prefix_key_value=prefix_key_value,
        )
        return outputs[0].to(input.device), incr_state


class PTParlaiT5Model(ParlaiT5Model):

    def __init__(self, opt, dictionary):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = self.pad_idx
        self.end_idx = dictionary[dictionary.end_token]
        super(ParlaiT5Model, self).__init__(self.pad_idx, self.start_idx, self.end_idx)
        self.t5 = build_t5(opt)
        self.encoder = PTParlaiT5Encoder(opt, self.t5.get_encoder(), self.pad_idx)
        self.decoder = PTParlaiT5Decoder(opt, self.t5.get_decoder(), self.pad_idx)
        self.paralleled = not opt['t5_model_parallel']
        self.prefix_tuning_init(opt)

    def prefix_tuning_init(self, opt):
        # Prefix-tuning
        # encoder self-attention
        dim = self.t5.config.d_kv * self.t5.config.num_heads
        self.encoder_prompt_token = torch.nn.Parameter(torch.Tensor(opt["preseqlen"], dim))
        torch.nn.init.normal_(self.encoder_prompt_token)
        self.encoder_control_trans = torch.nn.Sequential(
            torch.nn.Linear(dim, opt["mid_dim"]),
            torch.nn.Tanh(),
            torch.nn.Linear(opt["mid_dim"], opt["mid_dim"]),
            torch.nn.Tanh(),
            torch.nn.Linear(opt["mid_dim"], self.t5.config.num_layers * 2 * dim),
            torch.nn.Dropout(opt["prefix_dropout"]),
        )  # bottleneck to stabilize training
        # decoder self-attention
        self.self_prompt_token = torch.nn.Parameter(torch.Tensor(opt["preseqlen"], dim))
        torch.nn.init.normal_(self.self_prompt_token)
        self.self_control_trans = torch.nn.Sequential(
            torch.nn.Linear(dim, opt["mid_dim"]),
            torch.nn.Tanh(),
            torch.nn.Linear(opt["mid_dim"], opt["mid_dim"]),
            torch.nn.Tanh(),
            torch.nn.Linear(opt["mid_dim"], self.t5.config.num_decoder_layers * 2 * dim),
            torch.nn.Dropout(opt["prefix_dropout"]),
        )  # bottleneck to stabilize training
        # decoder cross-attention
        self.cross_prompt_token = torch.nn.Parameter(torch.Tensor(opt["preseqlen"], dim))
        torch.nn.init.normal_(self.cross_prompt_token)
        self.cross_control_trans = torch.nn.Sequential(
            torch.nn.Linear(dim, opt["mid_dim"]),
            torch.nn.Tanh(),
            torch.nn.Linear(opt["mid_dim"], opt["mid_dim"]),
            torch.nn.Tanh(),
            torch.nn.Linear(opt["mid_dim"], self.t5.config.num_decoder_layers * 2 * dim),
            torch.nn.Dropout(opt["prefix_dropout"]),
        )  # bottleneck to stabilize training
        for param in self.t5.parameters():
            param.requires_grad = False  # only tune prompt

    def prefix_tuning_encoder(self, bsz):
        # encoder
        encoder_past_key_values = self.encoder_control_trans(self.encoder_prompt_token.unsqueeze(0).expand(bsz, -1, -1))
        bsz, seqlen, _ = encoder_past_key_values.shape  # bsz, seqlen, layer * emb
        encoder_past_key_values = encoder_past_key_values.view(bsz, seqlen,
                                                               self.t5.config.num_layers * 2,
                                                               self.t5.config.num_heads,
                                                               self.t5.config.d_kv)
        encoder_past_key_values = encoder_past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return tuple((kv[0, ...], kv[1, ...]) for kv in encoder_past_key_values)

    def prefix_tuning_decoder(self, bsz):
        # decoder self-attention
        self_past_key_values = self.self_control_trans(self.self_prompt_token.unsqueeze(0).expand(bsz, -1, -1))
        bsz, seqlen, _ = self_past_key_values.shape  # bsz, seqlen, layer * emb
        self_past_key_values = self_past_key_values.view(bsz, seqlen,
                                                         self.t5.config.num_decoder_layers * 2,
                                                         self.t5.config.num_heads,
                                                         self.t5.config.d_kv)
        self_past_key_values = self_past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        # decoder cross-attention
        cross_past_key_values = self.cross_control_trans(self.cross_prompt_token.unsqueeze(0).expand(bsz, -1, -1))
        bsz, seqlen, _ = cross_past_key_values.shape  # bsz, seqlen, layer * emb
        cross_past_key_values = cross_past_key_values.view(bsz, seqlen,
                                                           self.t5.config.num_decoder_layers * 2,
                                                           self.t5.config.num_heads,
                                                           self.t5.config.d_kv)
        cross_past_key_values = cross_past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return tuple((s[0, ...], s[1, ...], c[0, ...], c[1, ...]) for s, c in zip(self_past_key_values, cross_past_key_values))

    def forward(self, *xs, ys=None, prev_enc=None, maxlen=None, bsz=None):
        assert ys is not None, "Greedy decoding in TGModel.forward no longer supported."
        # TODO: get rid of longest_label
        # keep track of longest label we've ever seen
        # we'll never produce longer ones than that during prediction
        self.longest_label = max(self.longest_label, ys.size(1))

        # use cached encoding if available
        encoder_states = prev_enc if prev_enc is not None else self.encoder(*xs, prefix_key_value=self.prefix_tuning_encoder(ys.size(0)))

        # use teacher forcing
        scores, preds = self.decode_forced(encoder_states, ys)
        return scores, preds, encoder_states

    def decode_forced(self, encoder_states, ys):
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        if (ys[:, 0] == self.START_IDX).any():
            raise AssertionError(
                "The Beginning of Sentence token is automatically added to the "
                "label in decode_forced, but you included it in the label. This means "
                "your model will have a double BOS token, which is probably not what "
                "you intended."
            )
        inputs = self._get_initial_forced_decoder_input(bsz, inputs)
        latent, _ = self.decoder(inputs, encoder_states, prefix_key_value=self.prefix_tuning_decoder(bsz))
        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds
