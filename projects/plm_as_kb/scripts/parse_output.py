#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Parse the output file to be a new dataset and save it.

## Examples

```shell
python parse_output.py --input-file INPUT_PATH --output-file OUTPUT_PATH
```
"""
import os
import json
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, False, 'Parse output file into dataset.')
    # Get command line arguments
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=10)
    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='The path of file to be parsed'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='The path of file to be outputted'
    )
    parser.add_argument(
        '--extract-name',
        type=str,
        required=True,
        help='The name of agent whose response will be extracted'
    )
    parser.add_argument(
        '--raw-file',
        type=str,
        required=True,
        help='The path of raw data file'
    )
    parser.add_argument(
        '--store-name',
        type=str,
        required=True,
        help='The name of response when stored'
    )
    return parser


def _clean(raw_sentence: str) -> str:
    sentence = raw_sentence.replace("no_passages_used", "")
    sentence = sentence.replace("<pad>", "")
    sentence = sentence.replace("no_images", "")
    sentence = sentence.replace("<|endoftext|>", " ")
    sentence = sentence.strip()
    return sentence


def parse(opt):
    if not os.path.exists(opt['input_file']):
        raise ValueError(f"{opt['input_file']} does not exist.")
    if not os.path.isfile(opt['input_file']):
        raise ValueError(f"{opt['input_file']} is not a file.")

    if not os.path.exists(opt['raw_file']):
        raise ValueError(f"{opt['raw_file']} does not exist.")
    if not os.path.isfile(opt['raw_file']):
        raise ValueError(f"{opt['raw_file']} is not a file.")

    with open(opt['input_file'], 'r', encoding='utf-8') as reader:
        data = reader.readlines()

    clean_data = []
    # truncate head printing
    for idx, line in enumerate(data):
        if line.startswith('['):
            clean_data = data[idx:]
            break
    # truncate tail reporting
    for idx, line in enumerate(reversed(clean_data)):
        if "Finished evaluating tasks" in line:
            clean_data = clean_data[:-(idx + 1)]
            break

    extracted_data = []
    clean_data_str = "".join(clean_data)
    for dialog in clean_data_str.split("- - - - - - - END OF EPISODE - - - - - - - - - -")[:-1]:
        # in one conversation
        extracted_dialog = []
        for turn in dialog.split("\n~~"):
            # in one turn
            if turn.strip() == "":
                continue
            response = turn.split(f"[{opt['extract_name']}]: ")[-1].strip()
            extracted_dialog.append(response)
        extracted_data.append(extracted_dialog)

    raw_data = json.load(open(opt["raw_file"]))
    assert len(raw_data) == len(extracted_data)
    for i in range(len(raw_data)):
        wizard = [utterance['text'] for utterance in raw_data[i]["dialog"] if 'Wizard' in utterance["speaker"]]
        assert len(wizard) == len(extracted_data[i]), f"{i} {len(wizard)} {len(extracted_data[i])}"

    # augment the raw data with the generated output
    # TODO: prepend the template string to the generated output
    for i, dialog in enumerate(raw_data):
        j = 0
        for utterance in dialog["dialog"]:
            if 'Wizard' in utterance["speaker"]:
                if "responses" not in utterance.keys():
                    utterance["responses"] = {}
                utterance["responses"][opt["store_name"]] = _clean(extracted_data[i][j])
                j += 1

    with open(opt["output_file"], 'w') as writer:
        json.dump(raw_data, writer)


@register_script('parse_output', hidden=True)
class Parser(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return parse(self.opt)


if __name__ == '__main__':
    Parser.main()
