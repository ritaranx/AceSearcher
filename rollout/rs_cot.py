# rs_cot.py
from genericpath import exists
from tqdm import tqdm
import argparse
import json
from utils import read_jsonl, write_json, build_tokenizer, set_seed, LLMConfig, build_sampling_params, build_llm, call_llm
import random
import os
from functools import partial


# -------- dataset loaders (light wrappers around local file formats) --------
def load_gsm8k(path, n_examples=2000):
    records = []
    for ex in read_jsonl(path):
        if "####" not in ex["answer"]:
            continue
        cot, ans = ex["answer"].split("\n####", 1)
        if random.random() > 0.5:
            q = f"Please answer the following question using text: {ex['question']} Use <answer> and </answer> tags to mark your answer in the end of your response."
        else:
            q = ex["question"] + "\nPlease answer the above question in plain text. Use <answer> and </answer> tags to mark your answer in the end."
        records.append({
            "question": q.strip(),
            "cot": cot.strip() + f" <answer>{ans.strip()}</answer>",
            "answer": ans.strip(),
        })
    random.shuffle(records)
    return records[:n_examples]

def load_tabmwp(path, n_examples=2000):
    data = json.load(open(path))
    records = []
    for _, ex in data.items():
        soln = ex["solution"]
        ans = ex["answer"]
        question = ex["question"]
        table_title = ex.get("table_title", "")
        table = (table_title + "\n" if table_title else "") + ex["table"]
        if random.random() > 0.4:
            q = f"You have the following information:\nTable:\n{table}\nPlease answer the question '{question}' in plain text. Use <answer> and </answer> tags to mark your answer in the end."
        else:
            q = f"Table:\n{table}\nAnswer the following question using the information in the above table: {question} Please answer in plain text. Mark your answer by enclosing it within <answer> and </answer> tags."
        records.append({
            "question": q.strip(),
            "answer": ans.strip(),
            "cot": soln.strip() + f" <answer>{ans.strip()}</answer>",
        })
    random.shuffle(records)
    return records[:n_examples]

def load_ifqa(path, n_examples=2000):
    data = json.load(open(path))
    base_prompt = (
        "You are given some passages:\n{passages}\n\n"
        "Please answer the question '{question}' with a short span. Provides clear reasoning followed by a concise conclusion. "
        "If no relevant information is found, use your own knowledge. Wrap your answer with <answer> and </answer> tags."
    )
    records = []
    for ex in data:
        passages = ex["ctx"]
        random.shuffle(passages)
        prompt = base_prompt.format(passages="\n\n".join(passages), question=ex["question"])
        records.append({"question": prompt.strip(), "answer": ex["answer"]})
    random.shuffle(records)
    return records[:n_examples]

def main_rs_cot():
    parser = argparse.ArgumentParser("rs_cot")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--dataset", type=str, default="", required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.99)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--round", type=int, default=2)
    parser.add_argument("--N_call", type=int, default=3)
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = LLMConfig(
        model_path=args.model_path,
        tokenizer_name=args.tokenizer,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=1024,
        repetition_penalty=1.05,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_mem_util=0.9,
        cache_dir=args.cache_dir,
    )
    tokenizer = build_tokenizer(cfg)
    sp = build_sampling_params(cfg)
    llm = build_llm(cfg)

    datasets = {
        "ifqa": partial(load_ifqa, path = f"train_data/ifqa/train.jsonl"),
        "gsm8k": partial(load_gsm8k, path = f"train_data/gsm8k/train.jsonl"), 
        "TabMWP": partial(load_tabmwp, path = f"train_data/TabMWP/train.jsonl")
    }

    for name, loader in datasets.items():
        examples = loader(n_examples=2000)
        print(name, len(examples))
        new_records = []
        for idx, ex in enumerate(tqdm(examples)):
            q_text = ex["question"]
            ans = ex["answer"]
            cot = ex.get("cot", "")
            responses = [call_llm(q_text, llm, tokenizer, sp) for _ in range(args.N_call)]
            if idx % 100 == 0:
                print(f"prompts_cot_{args.expname}_r{args.round}", idx, q_text[:100], ans, responses[-1][:100])
            new_records.append({
                "prompt": q_text,
                "generation": responses,
                "answer": ans,
                "cot": cot,
            })
        os.makedirs(f"{args.save_dir}/cot", exist_ok = True)
        out_path = f"{args.save_dir}/cot/prompts_cot_{name}_{args.expname}_r{args.round}.json"
        write_json(out_path, new_records)


if __name__ == "__main__":
    main_rs_cot()