# EmbedLLM data checks (Stage 1)

Generated 2026-10-25 23:52 by `repro/check_data.py`.

- Dataset: `RZ412/EmbedLLM`, revision `266ae3e180ccb533c86afc9cfb2f5db99d171aea (recorded at download; HF cache since removed)`
- Raw CSVs: `locus_code_release/data/datasets/embedllm/raw`
- Concise parquets: `locus_code_release/data/datasets/embedllm/concise`

## Problems found

- none

## 1. Concise parquets

| split | prompts | unique prompt_id | label cols | NaN labels | positive rate | model acc min | model acc max | emb dims |
|---|---|---|---|---|---|---|---|---|
| train | 29,673 | 29,673 | 112 | 0 | 0.4064 | 0.2191 | 0.6013 | paraphrase_albert_small_v2=768, all_minilm_l6_v2=384, all_mpnet_base_v2=768, all_distilroberta_v1=768 |
| val | 3,000 | 3,000 | 112 | 0 | 0.4091 | 0.2253 | 0.5990 | paraphrase_albert_small_v2=768, all_minilm_l6_v2=384, all_mpnet_base_v2=768, all_distilroberta_v1=768 |
| test | 3,000 | 3,000 | 112 | 0 | 0.4058 | 0.1937 | 0.6050 | paraphrase_albert_small_v2=768, all_minilm_l6_v2=384, all_mpnet_base_v2=768, all_distilroberta_v1=768 |

## 2. Prompts per benchmark

| benchmark | paper column | train | val | test | paper test (implied) | match |
|---|---|---|---|---|---|---|
| mmlu | MMLU | 11,680 | 1,157 | 1,178 | 1,178 | yes |
| gpqa | GPQA | 4,720 | 446 | 500 | 500 | yes |
| medmcqa | MedQA | 3,454 | 375 | 354 | 354 | yes |
| mathqa | MathQA | 2,468 | 280 | 237 | 237 | yes |
| asdiv | ASDiv | 1,916 | 190 | 198 | 198 | yes |
| social_iqa | SoQA | 1,607 | 161 | 162 | 162 | yes |
| piqa | PIQA | 1,509 | 160 | 134 | 134 | yes |
| gsm8k | GSM8k | 1,095 | 112 | 112 | 112 | yes |
| truthfulqa_mc1 | TruthQA | 684 | 59 | 74 | 74 | yes |
| logiqa | LogiQA | 540 | 60 | 51 | 51 | yes |
| **total** |  | 29,673 | 3,000 | 3,000 | 3,000 |  |

Raw categories per split: {'train': 80, 'val': 80, 'test': 80}. `mmlu*` and `gpqa*` are merged as in `scripts/evaluation.py`.


## 3. Duplicate (prompt, model) rows in the raw CSVs

| split | raw rows | unique pairs | extra rows | duplicated pairs | pairs with conflicting labels | conflicting / all pairs | missing model_id |
|---|---|---|---|---|---|---|---|
| train | 3,814,272 | 3,323,376 | 490,896 | 459,984 | 64,781 | 0.0195 | 0 |
| val | 383,488 | 336,000 | 47,488 | 44,576 | 6,116 | 0.0182 | 0 |
| test | 388,528 | 336,000 | 52,528 | 49,056 | 6,684 | 0.0199 | 0 |

`make_concise_split.py` keeps the **last** occurrence of each (prompt_id, model_id) pair, so for conflicting pairs the label in the parquet is the last row's label.


## 4. Prompt overlap between splits

| splits | shared prompt_id | shared exact prompt text |
|---|---|---|
| train ∩ val | 0 | 0 |
| train ∩ test | 0 | 0 |
| val ∩ test | 0 | 0 |

## 5. Sentence-encoder truncation

| encoder | max_seq_length | split | median tokens | truncated | truncated by benchmark |
|---|---|---|---|---|---|
| all_mpnet_base_v2 | 384 | train | 74 | 0.0604 | gsm8k 100%, gpqa 8%, mmlu 3% |
| all_mpnet_base_v2 | 384 | val | 72 | 0.0640 | gsm8k 100%, gpqa 10%, mmlu 3% |
| all_mpnet_base_v2 | 384 | test | 76 | 0.0650 | gsm8k 100%, gpqa 11%, mmlu 2% |
| paraphrase_albert_small_v2 | 100 | train | 77 | 0.4035 | truthfulqa_mc1 100%, gsm8k 100%, logiqa 99%, gpqa 95%, mmlu 42%, medmcqa 4%, mathqa 3%, asdiv 0% |
| paraphrase_albert_small_v2 | 100 | val | 75 | 0.3893 | logiqa 100%, gsm8k 100%, truthfulqa_mc1 100%, gpqa 95%, mmlu 42%, medmcqa 6%, mathqa 2% |
| paraphrase_albert_small_v2 | 100 | test | 78 | 0.4090 | logiqa 100%, gsm8k 100%, truthfulqa_mc1 100%, gpqa 97%, mmlu 41%, medmcqa 4%, mathqa 4% |
| all_minilm_l6_v2 | 256 | train | 74 | 0.1222 | gsm8k 100%, gpqa 29%, mmlu 10%, logiqa 9% |
| all_minilm_l6_v2 | 256 | val | 72 | 0.1210 | gsm8k 100%, gpqa 29%, mmlu 10%, logiqa 7% |
| all_minilm_l6_v2 | 256 | test | 76 | 0.1300 | gsm8k 100%, gpqa 31%, logiqa 14%, mmlu 10% |
| all_distilroberta_v1 | 512 | train | 79 | 0.0450 | gsm8k 100%, gpqa 3%, mmlu 1% |
| all_distilroberta_v1 | 512 | val | 77 | 0.0480 | gsm8k 100%, gpqa 5%, mmlu 1% |
| all_distilroberta_v1 | 512 | test | 80 | 0.0480 | gsm8k 100%, gpqa 5%, mmlu 1% |

Truncated prompts are embedded from their first `max_seq_length` tokens only. GSM8K prompts are few-shot with the actual question last, so truncation removes the question itself.


## 6. Held-out models (Fig. 3)

| held-out model (paper Fig. 3) | model_id |
|---|---|
| FelixChao/llama2-13b-math1.2 | 96 |
| MaziyarPanahi/WizardLM-Math-70B-v0.1 | 97 |
| 01-ai/Yi-6B-200K | 98 |
| WizardLM/WizardLM-70B-V1.0 | 99 |
| bigscience/bloom-7b1 | 100 |
| sail/Sailor-7B | 101 |
| codellama/CodeLlama-13b-Instruct-hf | 102 |
| Writer/palmyra-med-20b | 103 |
| Qwen/Qwen1.5-0.5B-Chat | 104 |
| databricks/dolly-v2-12b | 105 |
| nomic-ai/gpt4all-13b-snoozy | 106 |
| stabilityai/stablelm-tuned-alpha-7b | 107 |
| AdaptLLM/medicine-chat | 108 |
| AdaptLLM/medicine-LLM | 109 |
| EleutherAI/pythia-12b | 110 |
| Q-bert/Optimus-7B | 111 |

All 16 held-out models are indices 96-111: **yes**. Total models in model_order.csv: 112.

