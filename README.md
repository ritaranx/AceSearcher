# AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play

<!-- [![Paper](https://img.shields.io/badge/Paper-PDF-red)](TODO)  -->
<!-- [![ArXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b)](TODO) -->

---
This is the code repo for the paper *AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play* (NeurIPS 2025 Spotlight).

---

## 📌 Overview
**AceSearcher** is a framework that unifies **reasoning** and **search** for large language models (LLMs) via **reinforced self-play**.  
Our method bootstraps LLMs’ ability to solve multi-hop reasoning tasks by jointly training decomposer and solver modules with supervised finetuning and reinforcement learning stage. 

---

## ⚙️ Installation
```bash
git clone https://github.com/ritaranx/AceSearcher.git
cd AceSearcher
pip install -r requirements.txt
```

## Data Generation 
Most of the data generation used in AceSearcher is in `rollout` folder. The description for files are listed as belows:
- `rs_mhqa.py` |  `rs_cot.py` | `rs_pot.py`: [Step 1] the rollout pipeline for multi-hop QA, chain-of-thought, and program-of-thought datasets.
- `create_training_pairs.py`: [Step 2] the process for filtering & selecting preference pairs in mDPO iterations. 
- `create_dpo_pairs.py`: [Step 3] the process of curating the final preference pairs for reinforcement finetuning

## Evaluation
- For QA / Fact Verification Datasets:
    - Use `decompose_vllm.py` to first decompose the data.
    - Use `main_qa.py` to generate the final answer.
- For Document-level Financial Reasoning Datasets:
    - Use `main_reasoning.py` for evaluation.


## Data Directories
- Put corpus and embeddings in `embeddings/{dataset}/`. We use the wikipedia dump for `hover`, `exfever` and `bamboogle` datasets while using the script in [IRCOT](https://github.com/StonyBrookNLP/ircot) repo for getting the corpus for `hotpotqa`, `2wikimhqa` and `musique`. 
- The training data should be in `train_data` folder as `f"train_data/{dataset}/train.jsonl"`.
- The processed data after rollout are in `processed_data/{dataset}/train.jsonl`. 
- The data used for mDPO finetuning will be put in `processed_data/mdpo/` folder.
- The evaluation data is put into `./eval_datasets` folder.


## Data Download 

| Resource        | Link |
|-----------------|------|
| SFT Data        | [AceSearcher/Search-SFT](https://huggingface.co/datasets/AceSearcher/Search-SFT) |
| RFT Data        | [AceSearcher/Search-RFT-Pairs](https://huggingface.co/datasets/AceSearcher/Search-RFT-Pairs) |
| RFT Prompts     | [AceSearcher/Search-RFT-Prompts](https://huggingface.co/datasets/AceSearcher/Search-RFT-Prompts) |
| Evaluation Data | [AceSearcher/evaluation_datasets](https://huggingface.co/datasets/AceSearcher/evaluation_datasets) |

## Model Usage
For question decomposition on QA tasks: 
```
prompt_plan_qa = """Please break down the question "{question}" into multiple specific sub-questions that address individual components of the original question. 
Mark each sub-question with ### at the beginning.  If you need to refer to answers from earlier sub-questions, use #1, #2, etc., to indicate the corresponding answers.
Decomposed Question:"""

prompt_qa = prompt_plan_qa.replace("{question}", question)

prompt = [
    {"role": "user", "content": prompt_qa.strip()}
] 

text = tokenizer.apply_chat_template(
    prompt,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

outputs = llm.generate([text], sampling_params)
generated_text = outputs[0].outputs[0].text
```

For question decomposition on fact verification tasks: 
```
prompt_plan_claim = """Please break down the claim "{claim}" into multiple smaller sub-claims that each focus on a specific component of the original statement, making it easier for a model to verify.
Begin each sub-claim with ###. If needed, refer to answers from earlier sub-claims using #1, #2, etc.
Decomposed claim:"""

prompt_plan_claim = prompt_plan_claim.replace("{question}", question)

prompt = [
    {"role": "user", "content": prompt_plan_claim.strip()}
] 

text = tokenizer.apply_chat_template(
    prompt,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

outputs = llm.generate([text], sampling_params)
generated_text = outputs[0].outputs[0].text
```

For question answering for subquestions:
```
prompt = f"""You have the following context passages:
{context_text}

Please answer the question '{sub_q}' with a short span using the context as reference.
If no answer is found in the context, use your own knowledge. Your answer needs to be as short as possible."""
```

For fact verification tasks for subquestions:
```
prompt = f"""You have the following context passages:
{context_text}

Please verify whether the claim '{sub_q}' is correct using the context as reference. 
If no answer is found in the context, use your own knowledge.
Please only output Yes or No and do not give any explanation."""
```

For question answering to generate the final answer:
```
prompt = f"""You have the following passages:
{passages}

You are also given some subquestions and their answers:
{sub_answer_text}

Please answer the question '{original_question}' with {final_prompt} using the documents and subquestions as reference.
Make sure your response is grounded in documents and provides clear reasoning followed by a concise conclusion. If no relevant information is found, use your own knowledge. 
Wrap your answer with <answer> and </answer> tags."""
```

For fact verification tasks to generate the final answer:
```
prompt = f"""You have the following passages:
{passages}

You are given some subquestions and their answers:
{sub_answer_text}

Please verify the correctness of the claim: '{original_question}' using the subquestions as reference. Please provide a concise and clear reasoning followed by a concise conclusion. Your answer should be Yes or No only. 
Wrap your answer with <answer> and </answer> tags."""
```

For Decomposition for document-level financial reasoning tasks:
```
decompose_prompt = """You have the following passages and table:\nPassages:\n{passage}\nPlease break down the question '{question}' into multiple specific sub-questions that address individual components of the original question, with the table and passages as the reference. Use ### to mark the start of each sub-question."""

qa_prompt = """You have the following passages and table:\nPassages:\n{passage}\nFor the question '{question}', here is a referenced breakdown:\n{decompose}.\n\nWrite a Python program to solve the question. Store the final result in the variable ans."""


question = "What would the change in furniture and fixtures between 2018 and 2019 be if furniture and fixtures were $5,000 thousand in 2018 instead? (in thousand)"

context_text = "\n|||December 31,||\n||Useful Life|2019|2018|\n|Computer equipment and software|3 \u2013 5 years|$57,474|$52,055|\n|Furniture and fixtures|7 years|6,096|4,367|\n|Leasehold improvements|2 \u2013 6 years|22,800|9,987|\n|Renovation in progress|n/a|8|1,984|\n|Build-to-suit property|25 years|\u2014|51,058|\n|Total property and equipment, gross||86,378|119,451|\n|Less: accumulated depreciation and amortization||(49,852)|(42,197)|\n|Total property and equipment, net||$36,526|$77,254|\n 7. OTHER BALANCE SHEET AMOUNTS The components of property and equipment, net is as follows (in thousands): Depreciation expense for the years ended December 31, 2019, 2018, and 2017 was $11.8 million, $10.2 million, and $10.3 million, respectively.\n"

decompose_prompt = decompose_prompt.replace("{passage}" , context_text)
decompose_prompt = decompose_prompt.replace("{question}", question)
message = [{"role": "user", "content": decompose_prompt.strip()}]
prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
generated_text = llm.generate(prompt, sampling_params)[0].outputs[0].text

qa_prompt = qa_prompt.replace("{passage}", context_text)
qa_prompt = qa_prompt.replace("{question}", question)
qa_prompt = qa_prompt.replace("{decompose}", generated_text)
message = [{"role": "user", "content": qa_prompt.strip()}]
prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
output = llm.generate(prompt, sampling_params)[0].outputs[0].text
```

## Training
We use [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory/) codebase for both SFT and RFT (mDPO) finetuning. Please see `config` folder for the example configs used.

## Reference
If you find this work useful, consider citing it. Thank you in advance:
```
@inproceedings{
xu2025acesearcher,
title={AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play},
author={Ran Xu and Yuchen Zhuang and Zihan Dong and Ruiyu Wang and Yue Yu and Joyce C. Ho and Linjun Zhang and Haoyu Wang and Wenqi Shi and Carl Yang},
booktitle={the 39th Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=jSgCM0uZn3}
}

```