#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Evaluate model outputs stored in dataset.

## Examples

```shell
python eval_output.py --dataset-file INPUT_PATH
```
"""
import os
import json
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.core.metrics import F1Metric, Metrics
from parlai.tasks.wizard_of_wikipedia.agents import _get_chosen_title_and_sent


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, False, 'Evaluate output.')
    # Get command line arguments
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=10)
    parser.add_argument(
        '--dataset-file',
        type=str,
        required=True,
        help='The path of dataset file'
    )
    parser.add_argument(
        '--extract-name',
        type=str,
        required=True,
        help='The name of agent whose response will be extracted'
    )
    return parser


def evaluate(opt):
    if not os.path.exists(opt['dataset_file']):
        raise ValueError(f"{opt['dataset_file']} does not exist.")
    if not os.path.isfile(opt['dataset_file']):
        raise ValueError(f"{opt['dataset_file']} is not a file.")

    with open(opt['dataset_file'], 'r', encoding='utf-8') as reader:
        data = json.load(reader)

    metrics = Metrics()
    for d in data:
        wizard_first = 'Wizard' in d['dialog'][0]['speaker']
        for idx, utterance in enumerate(d["dialog"]):
            if "Wizard" in utterance["speaker"]:
                apprentice_ret_passages = wizard_ret_passages = {}

                if not wizard_first or idx != 0:
                    apprentice_entry = d['dialog'][idx - 1]
                    apprentice_ret_passages = apprentice_entry['retrieved_passages']
                if idx - 2 >= 0:
                    wizard_prev_entry = d['dialog'][idx - 2]
                    wizard_ret_passages = wizard_prev_entry['retrieved_passages']

                chosen_topic = d.get('chosen_topic', '')
                chosen_topic_passages = d['chosen_topic_passage']

                knowledge_dict = {chosen_topic: chosen_topic_passages}
                for ret_passes in [apprentice_ret_passages, wizard_ret_passages]:
                    for passage in ret_passes:
                        for k, v in passage.items():
                            if k not in knowledge_dict.keys():
                                knowledge_dict[k] = v

                title, sentence = _get_chosen_title_and_sent(utterance, knowledge_dict)
                checked_sentence = sentence

                metrics.add(
                    'f1',
                    F1Metric.compute(
                        utterance["responses"][opt["extract_name"]], [utterance['text']]
                    ),
                )

                metrics.add(
                    'knowledge_f1',
                    F1Metric.compute(
                        utterance["responses"][opt["extract_name"]], [checked_sentence]
                    ),
                )

                cands = []
                for title, passage in knowledge_dict.items():
                    for p in passage:
                        # exclude checked_sentence from knowledge
                        if p != checked_sentence:
                            cands.append(p)

                if len(cands) > 0:
                    # this will be empty when wizard is the first and we exclude the checked_sentence
                    # in this case we cannot calculate the below metrics
                    metrics.add(
                        'max_knowledge_f1',
                        F1Metric.compute(
                            utterance["responses"][opt["extract_name"]], cands
                        ),
                    )

                    metrics.add(
                        'max_baseline_f1',
                        F1Metric.compute(
                            checked_sentence, cands
                        ),
                    )

                    for cand in cands:
                        metrics.add(
                            'mean_baseline_f1',
                            F1Metric.compute(
                                cand, [checked_sentence]
                            ),
                        )
    print(metrics.report())


@register_script('eval_output', hidden=True)
class Evaluator(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return evaluate(self.opt)


if __name__ == '__main__':
    Evaluator.main()
