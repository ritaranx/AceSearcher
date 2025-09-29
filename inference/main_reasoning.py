#!/usr/bin/env python3
"""
Cleaned script: generate a decomposition plan and Python code for QA using vLLM.

- Loads a dataset split from JSONL
- Builds a context from passages/tables (supports *_gold and complong_testmini variants)
- Uses a chat template to:
    1) Decompose the question into sub-questions
    2) Generate Python code to solve it (store final result in `ans`)
- Saves generated plan/code alongside the original example to JSONL
"""

import argparse
import copy
import json
import os
from typing import Dict, List, Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from utils import init_llm

# --------------------------- Prompts ---------------------------

DECOMPOSE_PROMPT = (
    "You have the following passages and table:\n"
    "Passages:\n{passage}\n"
    "Please break down the question '{question}' into multiple specific sub-questions "
    "that address individual components of the original question, with the table and "
    "passages as the reference. Use ### to mark the start of each sub-question."
)

QA_PROMPT = (
    "You have the following passages and table:\n"
    "Passages:\n{passage}\n"
    "For the question '{question}', here is a referenced breakdown:\n"
    "{decompose}.\n\n"
    "Write a Python program to solve the question. Store the final result in the variable ans."
)


# --------------------------- Helpers ---------------------------

def get_message(message: str) -> List[Dict[str, str]]:
    """Wrap user content in a chat message list."""
    return [{"role": "user", "content": message.strip()}]


def safe_truncate(text: str, max_chars: int) -> str:
    """Truncate long strings to avoid context overflow."""
    return text if len(text) <= max_chars else text[:max_chars]


def build_context_text(example: Dict[str, Any], k: int) -> str:
    """
    Build the context string depending on dataset variant.
    Supports:
      - *_gold variants: uses provided paragraph/table evidence indices
      - complong_testmini: uses filtered/ranked ctxs (top-k)
      - otherwise: uses the first passage
    """
    passages: List[str] = example.get("ctxs", []) or []
    subset_fallback = example.get("__subset_name__", "")  # optional, not required
    passage_to_idx = {p.strip(): i for i, p in enumerate(passages)}

    subset = example.get("__subset__", "")  # we will inject this later
    if not subset:
        # Heuristic: infer from file or flags upstream if needed; keep empty by default
        pass

    # *_gold branch
    # If the caller’s subset string ends with "_gold", use evidence indices
    if subset.endswith("_gold"):
        para_idx: List[int] = example.get("paragraph_evidence", []) or []
        table_idx: List[int] = example.get("table_evidence", []) or []
        n_ctxs = len(passages)

        chunks = []
        # paragraphs
        for x in para_idx:
            if 0 <= x < n_ctxs and passages[x].strip():
                chunks.append(passages[x].strip())
        # tables
        for x in table_idx:
            if 0 <= x < n_ctxs and passages[x].strip():
                chunks.append("Tables:\n" + passages[x].strip())

        return "\n\n".join(chunks) if chunks else (passages[0] if passages else "")

    # complong_testmini branch
    if subset == "complong_testmini":
        filtered_passages: List[str] = example.get("filter_ctxs", []) or []
        ranked_idx: List[int] = example.get("filter_sorted_idx", []) or []

        top_passages: List[str] = []
        for x in ranked_idx[:k]:
            fp = filtered_passages[x].strip()
            if fp.startswith("|") and fp in passage_to_idx:
                idx = passage_to_idx[fp]
                # prepend previous passage as text if available
                prev = passages[idx - 1] if 0 <= idx - 1 < len(passages) else ""
                top_passages.append((prev + "\n" + fp).strip() if prev else fp)
            else:
                top_passages.append(fp)
        return "\n\n".join(top_passages) if top_passages else (passages[0] if passages else "")

    # default branch: first passage only
    return passages[0] if passages else ""


def generate_with_retry(
    llm: LLM,
    tokenizer: AutoTokenizer,
    prompt_template: str,
    context_text: str,
    question: str,
    extra_replace: Dict[str, str] | None = None,
    sampling_params: SamplingParams | None = None,
    max_context_chars: int = 30000,
) -> str:
    """
    Fill template, apply chat formatting, and generate.
    Retries once with truncated context if needed.
    """
    sampling_params = sampling_params or SamplingParams(temperature=0, top_k=1, max_tokens=2048)

    def _run(ctx: str) -> str:
        tmp = prompt_template.replace("{passage}", ctx).replace("{question}", question)
        if extra_replace:
            for k, v in extra_replace.items():
                tmp = tmp.replace(k, v)
        message = get_message(tmp)
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        out = llm.generate(prompt, sampling_params)[0]
        return out.outputs[0].text

    # Try full context
    try:
        return _run(context_text)
    except Exception:
        # Retry with truncated context
        ctx_trunc = safe_truncate(context_text, max_context_chars)
        return _run(ctx_trunc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("DocMath QA codegen with vLLM")

    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--model_path",
        type=str,
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.99)
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--subset", type=str, default="complong_testmini", choices=[
        "simplong_testmini_gold", "compshort_testmini", "complong_testmini", "simpshort_testmini"
    ])
    parser.add_argument("--expname", type=str, default="rag_sft_v6_lr1e-6")
    parser.add_argument("--save_dir", type=str, default="eval_datasets/docmath")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--sentence_embedding_model",
        type=str,
        default="intfloat/e5-large-v2",
    )
    parser.add_argument(
        "--sentence_embedding_model_save_name",
        type=str,
        default="e5-large",
        choices=["coco_base", "coco_large", "dragon", "gte-base", "e5-large"],
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="Max model length for vLLM (prompt + generation).",
    )
    parser.add_argument(
        "--max_gen_tokens",
        type=int,
        default=2048,
        help="Max tokens to generate per call.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Map subsets to files
    data_dir: Dict[str, str] = {
        "simplong_testmini_gold": "eval_datasets/docmath/simplong_testmini.jsonl",
        "compshort_testmini": "eval_datasets/docmath/compshort_testmini.jsonl",
        "complong_testmini": "eval_datasets/docmath/complong_testmini.jsonl",
        "simpshort_testmini": "eval_datasets/docmath/simpshort_testmini.jsonl",
    }
    input_path = data_dir[args.subset]
    out_dir = os.path.join(
        args.save_dir,
        args.subset,
        f"prompts_decompose_test_t{args.temperature}_{args.expname}",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"test_{args.sentence_embedding_model_save_name}_k{args.k}.jsonl"
    )

    # Initialize models
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    llm = init_llm(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=1,
        max_tokens=args.max_gen_tokens,
    )

    # I/O loop
    with open(input_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            example = json.loads(line)
            # inject subset info for context builder
            example["__subset__"] = args.subset

            question: str = example.get("question", "")
            if not question:
                continue

            context_text = build_context_text(example, args.k)

            # 1) Decomposition
            generated_plan = generate_with_retry(
                llm=llm,
                tokenizer=tokenizer,
                prompt_template=DECOMPOSE_PROMPT,
                context_text=context_text,
                question=question,
                sampling_params=sampling_params,
            )

            # 2) Code generation
            generated_code = generate_with_retry(
                llm=llm,
                tokenizer=tokenizer,
                prompt_template=QA_PROMPT,
                context_text=context_text,
                question=question,
                extra_replace={"{decompose}": generated_plan},
                sampling_params=sampling_params,
            )

            # Save
            save_example = copy.deepcopy(example)
            save_example["generated_plan"] = generated_plan
            save_example["generated_code"] = generated_code

            fout.write(json.dumps(save_example, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
