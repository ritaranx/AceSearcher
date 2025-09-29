# creating DPO pairs
import json
import copy, os
import random

prompt_qa = """You have the following context passages:
{context}

Please answer the question '{question}' with a short span using the context as reference.
If no answer is found in the context, use your own knowledge. Your answer needs to be as short as possible."""

prompt_qd = """Please break down the question '{question}' into multiple specific sub-questions that address individual components of the original question. 
Mark each sub-question with ### at the beginning.  If you need to refer to answers from earlier sub-questions, use #1, #2, etc., to indicate the corresponding answers.
Decomposed Question:"""


prompt_qd_claim = """Please break down the claim "{question}" into multiple smaller sub-claims that each focus on a specific component of the original statement, making it easier for a model to verify.
Begin each sub-claim with ###. If needed, refer to answers from earlier sub-claims using #1, #2, etc.
Decomposed claim:"""

prompt_rag_cot = """You have the following passages:
{passages}

You are also given some subquestions and their answers:
{subquestions}

Please answer the question '{question}' with a short span using the documents and subquestions as reference.
Make sure your response is grounded in documents and provides clear reasoning followed by a concise conclusion. If no relevant information is found, use your own knowledge. 
Wrap your answer with <answer> and </answer> tags."""

prompt_rag_cot_claim = """You are given some subquestions and their answers:
{subquestions}

Please verify the correctness of the claim: '{question}' using the subquestions as reference. Please provide a concise and clear reasoning followed by a concise conclusion. Your answer should be Yes or No only. 
Wrap your answer with <answer> and </answer> tags."""

def load_mhqa_examples(model, dataset, round):
    # examples_qa = []
    # with open(f"mhqa-r{round}/{model}-{dataset}-mhqa-qa.json", "r") as f:

    examples_qd = []
    with open(f"processed_data/mhqa-r{round}/{model}-{dataset}-mhqa-qd.json", "r") as f:
        examples = json.load(f)
        for example in examples:
            question = example["question"]
            positives = []
            negatives = []
            for pos in example["positive"]:
                raw_subquestions = "\n".join([f"### {x['label']}: {x['text']}" for x in pos])
                positives.append(raw_subquestions)
            for neg in example["negative"]:
                raw_subquestions = "\n".join([f"### {x['label']}: {x['text']}" for x in neg])
                negatives.append(raw_subquestions)
            # dedup
            positives_new = [x for x in positives if x not in negatives]
            negatives_new = [x for x in negatives if x not in positives]
            positives, negatives = positives_new, negatives_new
            if dataset == "hover":
                prompt_tmp = copy.deepcopy(prompt_qd_claim)
            else:
                prompt_tmp = copy.deepcopy(prompt_qd)
            prompt_tmp = prompt_tmp.replace("{question}", question)
            if positives and negatives:
                if len(positives) == 1 and len(negatives) > 1:
                    positives = positives * len(negatives)
                elif len(positives) > 1 and len(negatives) == 1:
                    negatives = negatives * len(positives)
                for (x, y) in zip(positives, negatives):
                    if x is None or y is None or x == "" or y == "":
                        print(x, y)
                        continue
                    data = {
                        "conversations": [
                            {
                                "from": "human",
                                "value": prompt_tmp
                            }
                        ],
                        "chosen": {
                            "from": "gpt",
                            "value": x
                        },
                        "rejected": {
                            "from": "gpt",
                            "value": y
                        },
                        "gold": example["gold"]
                    }
                    examples_qd.append(data)

    examples_cot = []
    # cot
    with open(f"processed_data/mhqa-r{round}/{model}-{dataset}-mhqa-cot.json", "r") as f:
        examples = json.load(f)
        for example in examples:
            question = example["question"]
            subqa = []
            for i in range(len(example["subquestions"])):
                subquestion = example["subquestions"][i]["text"]
                if f"Q{i+1}" not in example["subanswers"]:
                    break
                answer = example["subanswers"][f"Q{i+1}"]
                subqa.append(
                    f"### Q{i+1}: {subquestion}, Answer for Q{i+1}: {answer}"
                )
            sub_answer_text = "\n".join(subqa)
            passages = example["passages"]
            if dataset == "hover":
                prompt_tmp = copy.deepcopy(prompt_rag_cot_claim)
            else:
                prompt_tmp = copy.deepcopy(prompt_rag_cot)
            prompt_tmp = prompt_tmp.replace("{subquestions}", sub_answer_text)
            prompt_tmp = prompt_tmp.replace("{passages}", passages)
            prompt_tmp = prompt_tmp.replace("{question}", question)
            if example["positive"] is None or example["negative"] is None or example["positive"] == "" or example["negative"] == "":
                print(example["positive"], example["negative"])
                continue
            data = {
                "conversations": [
                    {
                        "from": "human",
                        "value": prompt_tmp
                    }
                ],
                "chosen": {
                    "from": "gpt",
                    "value": example["positive"]
                },
                "rejected": {
                    "from": "gpt",
                    "value": example["negative"]
                },
                "gold": example["gold"]
            }
            examples_cot.append(data)
    print(len(examples_cot))
    print(len(examples_qd))
    return examples_cot, examples_qd
    

def load_cot_examples(model, dataset, round):
    # no template
    cot_examples = []
    with open(f"processed_data/cot-r{round}/{model}-{dataset}.json", "r") as f:
        examples = json.load(f)
        for example in examples:
            question = example["question"]
            positive = example["positive"]
            negative = example["negative"]
            if positive is None or negative is None or positive == "" or negative == "":
                print("cot", positive, negative)
                continue
            data = {
                "conversations": [
                    {
                        "from": "human",
                        "value": question
                    }
                ],
                "chosen": {
                    "from": "gpt",
                    "value": positive
                },
                "rejected": {
                    "from": "gpt",
                    "value": negative
                }
            }
            cot_examples.append(data)
    return cot_examples


prompt_plan = { 
    "gsm8k": """Please break down the question '{question}' into multiple specific sub-questions that address individual components of the original question. Use ### to mark the start for each sub-question.""",
    "TabMWP": """You have the following table:\n{table}\nFor the question '{question}', please break down the question into multiple specific sub-questions that address individual components of the original question, with the table as the reference. Use ### to mark the start of each sub-question.""",
    "convfinqa": """You have the following passages and table:\nPassages:\n{passage}\nTable:\n{table}\nPlease break down the question '{question}' into multiple specific sub-questions that address individual components of the original question, with the table and passages as the reference. Use ### to mark the start of each sub-question."""
}

prompt_plan_text = { 
    "gsm8k": """For the question '{question}', here is a referenced breakdown:\n{decompose}.\n\nWrite a Python program to solve the question. Store the final result in the variable ans.""",
    "TabMWP": """You have the following table:\n{table}\nFor the question: '{question}', here is a referenced breakdown:\n{decompose}.\n\nWrite a Python program to solve the question. Store the final result in the variable ans.""",
    "convfinqa": """You have the following passages and table:\nPassages:\n{passage}\nTable:\n{table}\nFor the question '{question}', here is a referenced breakdown:\n{decompose}.\n\nWrite a Python program to solve the question. Store the final result in the variable ans."""
}

prompt_text = { 
    "gsm8k": """For the question '{question}', write a Python program to solve the question. Store the final result in the variable ans.""",
    "TabMWP": """You have the following table:\n{table}\nFor the question '{question}', write a Python program to solve the question. Store the final result in the variable ans.""",
    "convfinqa": """You have the following passages and table:\nPassages:\n{passage}\nTable:\n{table}\nFor the question '{question}', write a Python program to solve the question. Store the final result in the variable ans."""
}


def load_code_examples(model, dataset, round):
    # code decomposition
    code_examples_qd = []
    with open(f"processed_data/code-r{round}/{model}-{dataset}-qd.json", "r") as f:
        examples = json.load(f)
        for example in examples:
            question = example.get("question", "")
            passage =  example.get("passage", "")
            table = example.get("table", "")
            decompose = example.get("decompose", "")
            prompt_tmp = copy.deepcopy(prompt_plan[dataset])
            prompt_tmp = prompt_tmp.replace("{passage}", passage)
            prompt_tmp = prompt_tmp.replace("{table}", table)
            prompt_tmp = prompt_tmp.replace("{question}", question)
            positives = example["positive"]
            negatives = example["negative"]
            if len(positives) == 1 and len(negatives) > 1:
                positives = positives * len(negatives)
            elif len(positives) > 1 and len(negatives) == 1:
                negatives = negatives * len(positives)
            for (positive, negative) in zip(positives, negatives):
                if positive is None or negative is None or positive == "" or negative == "":
                    print("code qd", positive, negative)
                    continue
                data = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": prompt_tmp
                        }
                    ],
                    "chosen": {
                        "from": "gpt",
                        "value": positive
                    },
                    "rejected": {
                        "from": "gpt",
                        "value": negative
                    }
                }
                code_examples_qd.append(data)

    # code generaiton
    code_examples_code = []
    code_examples_plan_code = []
    with open(f"processed_data/code-r{round}/{model}-{dataset}-pot.json", "r") as f:
        examples = json.load(f)
        for example in examples:
            question =  example.get("question", "")
            passage =  example.get("passage", "")
            table = example.get("table", "")
            decompose = example.get("decompose", "")
            prompt_tmp_plan_code = copy.deepcopy(prompt_plan_text[dataset])
            prompt_tmp_plan_code = prompt_tmp_plan_code.replace("{passage}", passage)
            prompt_tmp_plan_code = prompt_tmp_plan_code.replace("{decompose}", decompose)
            prompt_tmp_plan_code = prompt_tmp_plan_code.replace("{table}", table)
            prompt_tmp_plan_code = prompt_tmp_plan_code.replace("{question}", question)

            prompt_tmp = copy.deepcopy(prompt_text[dataset])
            prompt_tmp = prompt_tmp.replace("{passage}", passage)
            prompt_tmp = prompt_tmp.replace("{table}", table)
            prompt_tmp = prompt_tmp.replace("{question}", question)

            positives = example["positive"]
            negatives = example["negative"]
            if len(positives) == 1 and len(negatives) > 1:
                positives = positives * len(negatives)
            elif len(positives) > 1 and len(negatives) == 1:
                negatives = negatives * len(positives)
            for (positive, negative) in zip(positives, negatives):
                if positive is None or negative is None or positive == "" or negative == "":
                    print("code pot", positive, negative)
                    continue
                data_plan = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": prompt_tmp_plan_code
                        }
                    ],
                    "chosen": {
                        "from": "gpt",
                        "value": positive
                    },
                    "rejected": {
                        "from": "gpt",
                        "value": negative
                    }
                }
                code_examples_plan_code.append(data_plan)
                data = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": prompt_tmp
                        }
                    ],
                    "chosen": {
                        "from": "gpt",
                        "value": positive
                    },
                    "rejected": {
                        "from": "gpt",
                        "value": negative
                    }
                }
                code_examples_code.append(data)
    print(dataset, "code_examples_code", len(code_examples_code))
    print(dataset, "code_examples_plan_code", len(code_examples_plan_code))
    print(dataset, "code_examples_qd", len(code_examples_qd))
    return code_examples_qd, code_examples_plan_code, code_examples_code


if __name__ == "__main__":
    model = "qwen_32b"
    round = 2
    examples_cot =[ ]
    examples_qd = []
    dpo_examples = {}
    for dataset in ["hover", "hotpotqa", "2wikimultihopqa"]:
        example_cot, example_qd = load_mhqa_examples(model, dataset, round)
        examples_cot += example_cot
        examples_qd += example_qd
        dpo_examples[f"qd_{dataset}"] = example_qd
        dpo_examples[f"cot_{dataset}"] = example_cot
    print(len(examples_cot), len(examples_qd))

    cot_examples = []
    for dataset in ["ifqa-rag", "gsm8k-rag", "TabMWP-rag"]:
        cot_examples = load_cot_examples(model, dataset,  round)
        print(dataset, len(cot_examples))
        dpo_examples[f"cot_{dataset}"] = cot_examples

    code_examples = []
    code_plan_examples = [ ]
    code_examples_qd = []
    for dataset in ["convfinqa", "gsm8k", "TabMWP"]:
        # code_examples_qd, code_examples_plan_code, code_examples_code
        qd, code_plan, code = load_code_examples(model, dataset, round)
        code_examples += code
        code_examples_qd += qd
        code_plan_examples += code_plan
        print(dataset, len(code), len(qd), len(code_plan))
        dpo_examples[f"qd_{dataset}"] = qd
        dpo_examples[f"code_plan_{dataset}"] = code_plan
        dpo_examples[f"code_{dataset}"] = code
    print("===========")
    cnt = 0
    dpo_train_examples = []
    for x in dpo_examples:
        print(x, len(dpo_examples[x]))
        cnt += len(dpo_examples[x])
        dpo_train_examples += dpo_examples[x]
        print("------")
    print(cnt)
    random.shuffle(dpo_train_examples)
    os.makedirs("processed_data/mdpo/", exist_ok=True)
    with open(f"processed_data/mdpo/dpo_mix_{model}_r{round}.json", "w") as f:
        json.dump(dpo_train_examples, f, indent = 2)
