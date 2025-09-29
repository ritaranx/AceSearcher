# rs_mhqa.py
"""
Multi-Hop QA: question/claim decomposition rollouts.
Reads train_data/{dataset}/train.jsonl and produces processed_data/{dataset}/prompts_decompose_train_{exp}_r{round}/corpus_*.txt
"""
import argparse
import os

from tqdm import trange
from utils import read_jsonl, append_jsonl, build_tokenizer, set_seed, LLMConfig, build_sampling_params, build_llm, call_llm, unique_take, parse_decomposed_items, 
from typing import Dict, List, Iterable, Any

def main_rs_mhqa():
    parser = argparse.ArgumentParser("rs_mhqa")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.99)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--round", type=int, default=2)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10000)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--gpu_mem_util", type=float, default=0.9)
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
        gpu_mem_util=args.gpu_mem_util,
        cache_dir=args.cache_dir,
    )
    tokenizer = build_tokenizer(cfg)
    sp = build_sampling_params(cfg)
    llm = build_llm(cfg)

    data_path = f"train_data/{args.dataset}/train.jsonl"

    prompt_q = (
        "Please break down the question '{question}' into multiple specific sub-questions that "
        "address individual components of the original question. Mark each sub-question with ### "
        "at the beginning. If you need to refer to answers from earlier sub-questions, use #1, #2, etc., "
        "to indicate the corresponding answers.\nDecomposed Question:"
    )
    prompt_claim = (
        "Please break down the claim \"{question}\" into multiple smaller sub-claims that each focus on a specific "
        "component of the original statement, making it easier for a model to verify. Begin each sub-claim with ###. "
        "If needed, refer to answers from earlier sub-claims using #1, #2, etc.\nDecomposed claim:"
    )

    # Load all questions/answers
    questions: List[str] = []
    gold: List[List[str]] = []
    for ex in read_jsonl(data_path):
        if args.dataset == "hover":
            questions.append(ex["claim"]) 
            label = ex.get("label", "").upper()
            gold.append(["Yes"] if label in {"SUPPORT", "SUPPORTED"} else ["No"])
        else:
            answers = []
            for z in ex.get("answers_objects", []):
                answers += z.get("spans", [])
            questions.append(ex.get("question_text", ""))
            gold.append(answers)

    total = len(questions)
    s = max(0, args.start)
    e = min(args.end, total)

    out_dir = f"processed_data/{args.dataset}/prompts_decompose_train_{args.expname}_r{args.round}"
    os.makedirs(out_dir, exist_ok=True)

    examples_batch: List[Dict[str, Any]] = []
    for i in trange(s, e):
        q_text = questions[i]
        g_ans = gold[i]
        user_prompt = (prompt_claim if args.dataset == "hover" else prompt_q).format(question=q_text)

        # Sample up to 8, keep first 3 unique
        gens: List[str] = []
        for _ in range(8):
            gen = call_llm(user_prompt, llm, tokenizer, sp)
            gens.append(gen)
            if len(set(gens)) >= 3:
                break
        uniq = unique_take(gens, 3)
        if i % 100 == 0:
            print("[sample]", user_prompt[:120].replace("\n", " "), {k: 1 for k in uniq})
        if len(uniq) <= 2:
            continue

        for j, gen in enumerate(uniq):
            decomposed = parse_decomposed_items(gen)
            if not decomposed:
                continue
            ctx = {
                "question": q_text,
                "answer": g_ans,
                "question_id": i,
                "decompose_id": j,
                "decomposed": decomposed,
            }
            if j == 0:
                print("======\n", ctx, "\n", gen, "\n======")
            examples_batch.append(ctx)

        # Periodic shard save
        if (i - s + 1) % 1000 == 0:
            shard = i - (i % 1000)
            shard_path = os.path.join(out_dir, f"corpus_{shard}.jsonl")
            append_jsonl(shard_path, examples_batch)
            examples_batch = []

    # Final flush
    if examples_batch:
        shard = 1000 * ((e - s) // 1000)
        shard_path = os.path.join(out_dir, f"corpus_{shard}.jsonl")
        append_jsonl(shard_path, examples_batch)


if __name__ == "__main__":
    # Guarded to avoid accidental execution when importing this file as utils.
    main_rs_mhqa()

