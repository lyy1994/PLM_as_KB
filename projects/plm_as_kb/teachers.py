#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt

from parlai.tasks.wizard_of_wikipedia.agents import GeneratorTeacher, TOKEN_NOCHOSEN, TOKEN_KNOWLEDGE


class KnowledgeGeneratorTeacher(GeneratorTeacher):
    """
    Teacher to train models that generate knowledge from context.
    """

    def get(self, episode_idx, entry_idx=0):
        a = super().get(episode_idx, entry_idx)
        d = self.data[episode_idx]
        wizard_first = 'Wizard' in d['dialog'][0]['speaker']
        idx = entry_idx * 2 if wizard_first else (entry_idx * 2) + 1

        wizard_entry = d['dialog'][idx]
        sentence_dict = wizard_entry['checked_sentence']
        if len(sentence_dict.values()) == 0:
            checked_sentence = TOKEN_NOCHOSEN
        else:
            checked_sentence = list(sentence_dict.values())[0]
        a['labels'] = [checked_sentence]
        return a


class KnowledgeGeneratorTestTeacher(GeneratorTeacher):
    """
    Teacher to test models that generate knowledge from context.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.model_response_name = opt.get('model_response_name', None)

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        agent = parser.add_argument_group('KnowledgeGeneratorTestTeacher Arguments')
        agent.add_argument(
            '--model-response-name',
            type=str,
            required=True,
            default=None,
            help='the name of model to be tested',
        )
        return parser

    def get(self, episode_idx, entry_idx=0):
        a = super().get(episode_idx, entry_idx)
        d = self.data[episode_idx]
        wizard_first = 'Wizard' in d['dialog'][0]['speaker']
        idx = entry_idx * 2 if wizard_first else (entry_idx * 2) + 1

        wizard_entry = d['dialog'][idx]
        # replace checked_sentence by the generated result
        assert self.model_response_name is not None, "please set --model-response-name"
        a['checked_sentence'] = (
                wizard_entry['responses'][self.model_response_name]
        )
        return a


class AugmentGeneratorTeacher(GeneratorTeacher):
    """
    Generator teacher with augmented knowledge from generated outputs of models.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.model_response_names = opt.get('model_response_names', None)

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        agent = parser.add_argument_group('AugmentGeneratorTeacher Arguments')
        agent.add_argument(
            '--model-response-names',
            type=str,
            required=True,
            default=None,
            help='the name of model',
        )
        return parser

    def get(self, episode_idx, entry_idx=0):
        a = super().get(episode_idx, entry_idx)
        d = self.data[episode_idx]
        wizard_first = 'Wizard' in d['dialog'][0]['speaker']
        idx = entry_idx * 2 if wizard_first else (entry_idx * 2) + 1

        wizard_entry = d['dialog'][idx]
        assert self.model_response_names is not None, "please set --model-response-name"
        # TODO: use chosen_topic may not be a good choice
        try:
            response = ['{} {} {}'.format(a['chosen_topic'], TOKEN_KNOWLEDGE, wizard_entry['responses'][name]) for name in self.model_response_names.split(",")]
        except KeyError:
            raise KeyError(f"Unknown model name, please choose from {list(wizard_entry['responses'].keys())}")
        a['response'] = response
        return a
