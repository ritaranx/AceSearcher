
# rs_pot.py
from tqdm import tqdm
import argparse
import json
from utils import read_jsonl, write_json, build_tokenizer, set_seed, LLMConfig, build_sampling_params, build_llm, call_llm, unique_take
import random
import os
from typing import Dict, List, Iterable, Any

def parse_table(table: List[List[str]]) -> str:
    lines = []
    for row in table:
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def load_convfinqa(path) -> List[Dict[str, Any]]:
    out = []
    for ex in read_jsonl(path):
        table = parse_table(ex.get("table", []))
        text = ex.get("annotation", {}).get("amt_post_text", "")
        qa = ex.get("qa", {})
        out.append({
            "question": qa.get("question", ""),
            "answer": [qa.get("exe_ans", ""), qa.get("answer", "")],
            "passage": text,
            "table": table,
        })
    return out

def load_gsm8k(path) -> List[Dict[str, Any]]:
    out = []
    for ex in read_jsonl(path):
        ans = ex.get("answer", "").split("####")[-1].strip()
        out.append({"question": ex.get("question", ""), "answer": ans})
    return out

def load_tabmwp(path) -> List[Dict[str, Any]]:
    data = json.load(open(path, "r"))
    out = []
    for _, ex in data.items():
        grade = ex.get("grade", 0)
        if grade <= 4: # filter out easy problems
            continue
        out.append({
            "question": ex.get("question", ""),
            "answer": ex.get("answer", ""),
            "table": ex.get("table", ""),
            "grade": grade,
        })
    return out

def main_rs_pot():
    parser = argparse.ArgumentParser("rs_pot")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.995)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--round", type=int, default=2)
    parser.add_argument("--n_decomp", type=int, default=4)
    parser.add_argument("--n_code", type=int, default=4)
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
        gpu_mem_util=0.89,
        cache_dir=args.cache_dir,
    )
    tokenizer = build_tokenizer(cfg)
    sp = build_sampling_params(cfg)
    model = build_llm(cfg)    

    paths = {
        "convfinqa": "train_data/convfinqa/train.jsonl",
        "gsm8k": "train_data/gsm8k/train.jsonl", 
        "TabMWP": "train_data/TabMWP/train.jsonl"
    }

   

    prompt_plan = {
        "gsm8k": "Please break down the question '{question}' into multiple specific sub-questions that address individual components of the original question. Use ### to mark the start for each sub-question.",
        "TabMWP": "You have the following table:\n{table}\nFor the question '{question}', please break down the question into multiple specific sub-questions that address individual components of the original question, with the table as the reference. Use ### to mark the start of each sub-question.",
        "convfinqa": "You have the following passages and table:\nPassages:\n{passage}\nTable:\n{table}\nPlease break down the question '{question}' into multiple specific sub-questions that address individual components of the original question, with the table and passages as the reference. Use ### to mark the start of each sub-question.",
    }

    prompt_plan_to_code = {
        "gsm8k": "For the question '{question}', here is a referenced breakdown:\n{decompose}.\n\nWrite a Python program to solve the question. Store the final result in the variable ans.",
        "TabMWP": "You have the following table:\n{table}\nFor the question: '{question}', here is a referenced breakdown:\n{decompose}.\n\nWrite a Python program to solve the question. Store the final result in the variable ans.",
        "convfinqa": "You have the following passages and table:\nPassages:\n{passage}\nTable:\n{table}\nFor the question '{question}', here is a referenced breakdown:\n{decompose}.\n\nWrite a Python program to solve the question. Store the final result in the variable ans.",
    }

    loaders = {
        "convfinqa": load_convfinqa,
        "gsm8k": load_gsm8k,
        "TabMWP": load_tabmwp,
    }

    for dataset, loader in loaders.items():
        examples = loader(paths[dataset])
        random.shuffle(examples)

        code_records: List[Dict[str, Any]] = []
        for ex in tqdm(examples):
            question = ex.get("question", "")
            answer = ex.get("answer", "")
            table = ex.get("table", "")
            passage = ex.get("passage", "")

            # Plan (decompose)
            plan_prompt = prompt_plan[dataset].format(question=question, table=table, passage=passage)
            decomp_candidates: List[str] = []
            for _ in range(args.n_decomp):
                decomp_candidates.append(call_llm(plan_prompt, model, tokenizer, sp))
            decomp_uniq = unique_take(decomp_candidates, 3)

            for di, decomp in enumerate(decomp_uniq):
                # Code
                code_prompt = prompt_plan_to_code[dataset].format(
                    question=question, table=table, passage=passage, decompose=decomp.strip()
                )
                code_candidates: List[str] = []
                for _ in range(args.n_code):
                    code_candidates.append(call_llm(code_prompt, model, tokenizer, sp))
                code_uniq = unique_take(code_candidates, 3)

                for ci, code in enumerate(code_uniq):
                    rec = {
                        "question": question,
                        "answer": answer,
                        "passage": passage,
                        "table": table,
                        "decompose_id": di,
                        "decompose": decomp,
                        "code_id": ci,
                        "code": code,
                    }
                    if di == 0 and ci == 0:
                        print(args.expname, args.round, rec)
                    code_records.append(rec)
        os.makedirs(f"{args.save_dir}/pot", exist_ok = True)
        out_path = f"{args.save_dir}/pot/prompts_pot_{dataset}_{args.expname}_r{args.round}.json"
        write_json(out_path, code_records)


if __name__ == "__main__":
    main_rs_pot()
