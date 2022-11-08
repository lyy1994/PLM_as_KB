# Eliciting Knowledge from Large Pre-Trained Models for Unsupervised Knowledge-Grounded Conversation

This is the implementation of the paper [Eliciting Knowledge from Large Pre-Trained Models for Unsupervised Knowledge-Grounded Conversation](https://arxiv.org/abs/2211.01587).

We will show the step-by-step guide for how to elicit knowledge from pre-trained models and exploit the generated knowledge in a knowledge-grounded conversation system on the [Wizard of Wikipedia](https://parl.ai/projects/wizard_of_wikipedia/) dataset. All commands are trained on 16 A100 80G GPUs by default.

## 1. Knowledge Generation

In this section, we will show you how to fine-tune/prefix-tune a DialoGPT/T5 model such that it can learn to generate knowledge for a given dialogue history. Then we show how to generate knowledge from a tuned model using various decoding algorithms.

### 1.1 Training

We show how to perform fine-tuning and prefix-tuning in two separate subsections, as their hyperparameters are slightly different from each other, but remain the same for both DialoGPT and T5.

#### 1.1.1 Fine-Tuning

Below is the commandline to fine-tune DialoGPT-Large for knowledge generation:
```commandline
MODEL_FILE=path/to/your/model
parlai multiprocessing_train \
    -t projects.plm_as_kb.teachers:KnowledgeGeneratorTeacher \
    --add-missing-turns all \
    -m hugging_face/dialogpt \
    --add-special-tokens True \
    --add-start-token True \
    --gpt2-size large \
    --tr 512 \
    -lr 5e-05 \
    --optimizer adam \
    --lr-scheduler invsqrt \
    --warmup-updates 1000 \
    --max-train-steps 200000 \
    --betas 0.9,0.999 \
    --update-freq 2 \
    --skip-generation True \
    -vp 15 \
    -vstep 1000 \
    -bs 2 \
    -vmt ppl \
    -vmm min \
    --model-file ${MODEL_FILE}
```
In this command, we take advantage from the utility of ParlAI and you can take a look at the [documentation](https://parl.ai/docs/agent_refs/hugging_face.html) for a detailed explanation of each argument.

Different sized DialoGPT could be used via setting `--gpt2-size` to `small`, `medium` or `large`.

If you want to fine-tune T5-XL models, please set `-m hugging_face/t5`, add `--t5-model-arch t5-3b --t5-model-parallel False` and remove `--add-special-tokens`, `--add-start-token` and `--gpt2-size` these three arguments. Different sized T5 could be used via setting `--t5-model-arch` to `t5-small`, `t5-base`, `t5-large`, `t5-3b` or `t5-11b`.

**NOTE**: You must set `--fp16 False` to fine-tune or prefix-tune DialoGPT, otherwise you might face the overflowing issue. 

#### 1.1.2 Prefix-Tuning

Below is the commandline to prefix-tune DialoGPT-Large for knowledge generation:
```commandline
MODEL_FILE=path/to/your/model
parlai multiprocessing_train \
    -t projects.plm_as_kb.teachers:KnowledgeGeneratorTeacher \
    --add-missing-turns all \
    -m projects.plm_as_kb.prefix_tuning.dialogpt:PTDialogptAgent \
    --preseqlen 5 \
    --add-special-tokens True \
    --add-start-token True \
    --gpt2-size large \
    --tr 512 \
    -lr 5e-06 \
    --optimizer adam \
    --lr-scheduler reduceonplateau \
    --max-train-steps 200000 \
    --betas 0.9,0.999 \
    --update-freq 2 \
    --skip-generation True \
    -vp 15 \
    -vstep 1000 \
    -bs 2 \
    -vmt ppl \
    -vmm min \
    --model-file ${MODEL_FILE}
```
If you want to prefix-tune a T5 model, please set `-m projects.plm_as_kb.prefix_tuning.t5:PTT5Agent` and other arguments are set similarly as in fine-tuning.


### 1.2 Inference

After tuning a pre-trained model, we will show the commandline to generate knowledge from this tuned model. As DialoGPT and T5 work the best with distinct decoding algorithms, we show the respective commandlines in two separate subsections.

#### 1.2.1 T5

Below is the commandline to use a tuned T5 model to generate knowledge on the seen valid set via beam search:
```commandline
MODEL_FILE=path/to/your/model
parlai multiprocessing_eval \
    -t projects.plm_as_kb.teachers:KnowledgeGeneratorTeacher \
    --add-missing-turns all \
    -bs 1 \
    --skip-generation False \
    --display-examples True \
    -ltim inf \
    --inference beam \
    --beam-size 10 \
    --beam-context-block-ngram 3 \
    --beam-block-ngram 3 \
    --beam-min-length 25 \
    --fp16 True \
    --model-file ${MODEL_FILE}
```
To generate knowledge on unseen valid set, please set `--task projects.plm_as_kb.teachers:KnowledgeGeneratorTeacher:topic_split`.

To generate knowledge on test set, please add `-dt test`.

**NOTE**: Since we set `--display-examples True`, the generated knowledge will be printed. You can redirect the output to a file and later we will parse the generated knowledge from this file.

#### 1.2.2 DialoGPT

Below is the commandline to use a tuned DialoGPT model to generate knowledge on the seen valid set via top-K sampling:
```commandline
MODEL_FILE=path/to/your/model
parlai multiprocessing_eval \
-t projects.plm_as_kb.teachers:KnowledgeGeneratorTeacher \
--add-missing-turns all \
-bs 1 \
--skip-generation False \
--display-examples True \
-ltim inf \
--inference topk \
--topk 10 \
--beam-size 20 \
--fp16 True \
--model-file ${MODEL_FILE}
```
We can similarly use DialoGPT to generate knowledge on the unseen valid set by adding `:topic_split` to `--task/-t` and test set by adding `-dt test`.


### 1.3 Dataset Construction

Once we store the generated knowledge from a pre-trained model to a log file, we need to parse knowledge from this log file and put it into the original dataset for future usage.

Here we show the commandline to extract knowledge from the log file of seen valid set, produced by a prefix-tuned T5-XXL model: 
```commandline
LOG=path/to/your/log/file
python scripts/parse_output.py \
    --input-file ${LOG} \
    --raw-file ../../data/wizard_of_wikipedia/valid_random_split.json \
    --output-file ../../data/wizard_of_wikipedia/valid_random_split.json \
    --extract-name PTT5 \
    --store-name "T5-XXL"
```
* `--raw-file`: this is the dataset file we would like to stitch the parsed knowledge.
* `--output-file`: this is the path where we want to store the data, including the parsed knowledge. You could set it to be the same as `--raw-file` such that 
* `--extract-name`:
* `--store-name`:

You can use the blow commandline to test the F1 score of the extracted knowledge see check whether it match the score reported in the log file. If no, then it means the log file parsing goes wrong and you may need to check the arguments (e.g., `--extract-name` is set to a wrong name or `-ltim` is not set to `inf` during inference):
```commandline
python scripts/eval_output.py \
--dataset-file ../../data/wizard_of_wikipedia/valid_random_split.json \
--extract-name "T5-XXL"
```

## 2. Response Generation

### 2.1 Baseline
#### 2.1.1 Training
Below is the commandline to train [PLATO-KAG](https://aclanthology.org/2021.nlp4convai-1.14/) baseline:
```commandline
MODEL_FILE=path/to/your/model
parlai multiprocessing_train \
    -t wizard_of_wikipedia:generator \
    --add-missing-turns all \
    -m projects.plm_as_kb.plato_kag:PlatoKagAgent \
    --add-special-tokens True \
    --add-start-token True \
    --gpt2-size medium \
    --knowledge-topk 8 \
    --knowledge-truncate 64 \
    --max-knowledge 32 \
    --tr 256 \
    -lr 2e-05 \
    --fp16 False \
    --optimizer adamw \
    --weight-decay 0.01 \
    --lr-scheduler invsqrt \
    --warmup-updates 1000 \
    --max-train-steps 200000 \
    --betas 0.9,0.999 \
    --update-freq 4 \
    --skip-generation True \
    -vp 15 \
    -vstep 2000 \
    -bs 1 \
    -vmt ppl \
    -vmm min \
    --model-file ${MODEL_FILE}
```

#### 2.1.2 Inference
Below is the commandline to evaluate the trained model on the seen valid set:
```commandline
MODEL_FILE=path/to/your/model
parlai multiprocessing_eval \
    --task wizard_of_wikipedia:generator \
    --add-missing-turns all  \
    -bs 1 \
    --fp16 True \
    --skip-generation False \
    --display-examples True \
    -ltim inf \
    --model-file ${MODEL_FILE}
```
To evaluate on unseen valid set, please set `--task wizard_of_wikipedia:generator:topic_split`.

To evaluate on test set, please add `-dt test`.

### 2.2 Our Model
#### 2.2.1 Training
The commandline to train our PLATO-KAG+ model:
```commandline
MODEL_FILE=path/to/your/model
parlai multiprocessing_train \
    -t wizard_of_wikipedia:generator \
    --add-missing-turns all \
    -m projects.plm_as_kb.plato_kag_plus:PlatoKagPlusAgent \
    --gumbel-topk True \
    --add-special-tokens True \
    --add-start-token True \
    --gpt2-size medium \
    --knowledge-topk 8 \
    --knowledge-truncate 64 \
    --max-knowledge 32 \
    --tr 256 \
    -lr 2e-05 \
    --fp16 False \
    --optimizer adamw \
    --weight-decay 0.01 \
    --lr-scheduler invsqrt \
    --warmup-updates 1000 \
    --max-train-steps 200000 \
    --betas 0.9,0.999 \
    --update-freq 4 \
    --skip-generation True \
    -vp 15 \
    -vstep 2000 \
    -bs 1 \
    -vmt ppl \
    -vmm min \
    --model-file ${MODEL_FILE}
```

#### 2.2.2 Inference
The commandline to evaluate our model on the seen valid set:
```commandline
MODEL_FILE=path/to/your/model
DATA=path/to/your/data
parlai multiprocessing_eval \
    --reweigh-type multiplicative \
    --reweigh-temperature 5 \
    --ensemble-type posterior \
    --posterior-sharpness 0.4 \
    -t projects.plm_as_kb.teachers:AugmentGeneratorTeacher \
    --fp16 True \
    --datapath ${DATA} \
    --model-response-names 'DialoGPT-large' \
    --add-missing-turns all \
    -bs 1 \
    --skip-generation False \
    --display-examples True \
    -ltim inf \
    --model-file ${MODEL_FILE}
```
We can similarly evaluate the model on the unseen valid set by adding `:topic_split` to `--task/-t` and test set by adding `-dt test`.

**NOTE**: our model requires the generated knowledge during inference, therefore we should insert generated knowledge to the original data and provide the path to the new data via `--datapath`.

## Citation
Please cite our paper if you use our code in your work:
```
@inproceedings{gao2021making,
   title={Eliciting Knowledge from Large Pre-Trained Models for Unsupervised Knowledge-Grounded Conversation},
   author={Yanyang Li and Jianqiao Zhao and Michael R. Lyu and Liwei Wang},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2022}
}
```
