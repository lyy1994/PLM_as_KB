#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt

import torch
from parlai.utils.misc import warn_once

from parlai.agents.hugging_face.gpt2 import GPT2Decoder, HFGPT2Model, Gpt2Agent

try:
    from transformers import GPT2Model
except ImportError:
    raise ImportError('Please run `pip install transformers`.')


############################################
## Modules
############################################


class PTGPT2Decoder(GPT2Decoder):

    def __init__(self, opt, dict):
        super().__init__(opt, dict)
        # Prefix-tuning
        self.preseqlen = opt["preseqlen"]
        self.prompt_token = torch.nn.Parameter(torch.Tensor(opt["preseqlen"], self.transformer.config.n_embd))
        torch.nn.init.normal_(self.prompt_token)
        self.control_trans = torch.nn.Sequential(
            torch.nn.Linear(self.transformer.config.n_embd, opt["mid_dim"]),
            torch.nn.Tanh(),
            torch.nn.Linear(opt["mid_dim"], opt["mid_dim"]),
            torch.nn.Tanh(),
            torch.nn.Linear(opt["mid_dim"], self.transformer.config.n_layer * 2 * self.transformer.config.n_embd),
            torch.nn.Dropout(opt["prefix_dropout"]),
        )  # bottleneck to stabilize training
        # TODO: --add-special-token add additional parameters into embeddings, but we did not tune them
        for param in self.transformer.parameters():
            param.requires_grad = False  # only tune prompt

    def prefix_tuning(self, bsz):
        past_key_values = self.control_trans(self.prompt_token.unsqueeze(0).expand(bsz, -1, -1))
        bsz, seqlen, _ = past_key_values.shape  # bsz, seqlen, layer * emb
        past_key_values = past_key_values.view(bsz, seqlen,
                                               self.transformer.config.n_layer * 2,
                                               self.transformer.config.n_head,
                                               self.transformer.config.n_embd // self.transformer.config.n_head)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self, input, encoder_state, incr_state=None):
        attention_mask = None
        position_ids = None

        if incr_state is None:
            # add soft prompts
            incr_state = self.prefix_tuning(encoder_state.size(0))
            generate = False
        else:
            generate = True

        if not generate:
            # first step
            if (
                    not self.add_start_token
                    and input.size(1) == 1
                    and int(input[0][0]) == self.START_IDX
            ):
                # generating: ignore the start token
                # without deep copy, the padding_idx (-1) in encoder_state can be reset to 0 with clamp_ inplace operation
                model_input = encoder_state.clone()
            else:
                # forced decoding: concatenate the context
                # with the labels
                model_input = torch.cat([encoder_state, input], dim=-1)
            attention_mask = model_input != self.NULL_IDX
            position_ids = (
                    attention_mask.cumsum(dim=-1, dtype=torch.int64) - 1
            ).clamp_(min=0)
        else:
            if not self.add_start_token:
                input = input[:, 1:]
            # generating with continuation
            # get the position ids
            position_ids = (encoder_state != self.NULL_IDX).sum(
                -1, True, dtype=torch.int64
            ) - 1
            delta = ((input != self.NULL_IDX)).sum(-1, True, dtype=torch.int64)
            position_ids += delta
            # generation: get the last token input
            model_input = input[:, -1:]
            attention_mask = torch.cat([encoder_state, input], dim=-1) != self.NULL_IDX

        # add soft prompts mask
        attention_mask = torch.cat(
            [
                 torch.ones(attention_mask.size(0), self.preseqlen,
                            dtype=attention_mask.dtype,
                            layout=attention_mask.layout,
                            device=attention_mask.device),
                 attention_mask,
            ],
            dim=-1
        )

        model_input = model_input.clamp_(min=0)
        transformer_outputs = self.transformer(
            model_input,
            past_key_values=incr_state,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = transformer_outputs[0]
        new_incr_state = transformer_outputs[1]

        if not generate:
            # pull out only the hidden states for the label tokens
            output = hidden_states[:, -input.size(1) - 1 + int(self.add_start_token):]
            # hack: we need the last state of the encoder-side to be the first
            # element of the decoder-side
            lengths = (input != self.NULL_IDX).sum(dim=-1)
            for i in range(input.size(0)):
                output[i, input.size(1) - lengths[i]] = output[i, 0]

        else:
            # generation, we're only doing one token at a time. no need to
            # shove things back in
            output = hidden_states

        return output, new_incr_state


class PTGPT2Model(HFGPT2Model):

    def _get_decoder(self, opt, dict):
        return PTGPT2Decoder(opt, dict)


############################################
## Agent
############################################


class PTGpt2Agent(Gpt2Agent):

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        agent = parser.add_argument_group("Prefix-Tuning Gpt2 Args")
        agent.add_argument(
            "--preseqlen",
            type=int,
            default=0,
            help="preseqlen for how many tokens of prefix should we include.",
        )
        agent.add_argument(
            "--mid-dim",
            type=int,
            default=512,
            help="the mid dim.",
        )
        agent.add_argument(
            "--prefix-dropout",
            type=float,
            default=0.0,
            help="dropout rate for the prefix tuning model.",
        )
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        warn_once("WARNING: this model is in beta and the API is subject to change.")
        return agent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        return PTGPT2Model(self.opt, self.dict)
