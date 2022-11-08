#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt

from parlai.utils.misc import warn_once

from parlai.agents.hugging_face.dialogpt import DialogptAgent

from .gpt2 import PTGPT2Decoder, PTGPT2Model

try:
    from transformers import GPT2Model
except ImportError:
    raise ImportError('Please run `pip install transformers`.')


############################################
## DialoGPT
############################################


class PTDialoGPTDecoder(PTGPT2Decoder):

    def __init__(self, opt, dict):
        super().__init__(opt, dict)
        self.NULL_IDX, self.START_IDX, self.END_IDX = self._get_special_tokens(
            opt, dict
        )

    @staticmethod
    def _get_special_tokens(opt, dict):
        null_idx = dict.null_idx
        if (
            opt.get('batchsize', 1) == 1
            and not opt['add_special_tokens']
            and null_idx == dict.end_idx
        ):
            # get around the dual usage of end_idx that would otherwise mask endtoken during forward pass.
            null_idx = -1
        return null_idx, dict.start_idx, dict.end_idx

    def _init_from_pretrained(self, opt):
        # load model
        model_sz = opt['gpt2_size']
        fle_key = f'microsoft/DialoGPT-{model_sz}'
        return GPT2Model.from_pretrained(fle_key)


class PTDialoGPTModel(PTGPT2Model):

    def _get_special_tokens(self, opt, dict):
        # keep it consistent between DialoGPTModel and DialoGPTDecoder on start_idx, end_idx, null_idx
        return PTDialoGPTDecoder._get_special_tokens(opt, dict)

    def _get_decoder(self, opt, dict):
        return PTDialoGPTDecoder(opt, dict)


class PTDialogptAgent(DialogptAgent):

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        agent = parser.add_argument_group('Prefix-Tuning DialoGPT Args')
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
        warn_once('WARNING: this model is in beta and the API is subject to change.')
        return agent
