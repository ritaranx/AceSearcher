# utils.py
"""
Shared utilities for rollout scripts (vLLM-based generation, prompt templating, I/O, and parsing).
"""
from __future__ import annotations
import os
import json
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Iterable, Any, Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ----------------------------
# Reproducibility
# ----------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)


# ----------------------------
# Model / Tokenizer setup
# ----------------------------

@dataclass
class LLMConfig:
    model_path: str
    tokenizer_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    temperature: float = 0.7
    top_p: float = 0.99
    max_tokens: int = 1024
    repetition_penalty: float = 1.05
    tensor_parallel_size: int = 1
    gpu_mem_util: float = 0.88
    trust_remote_code: bool = True
    cache_dir: Optional[str] = None


def build_tokenizer(cfg: LLMConfig):
    return AutoTokenizer.from_pretrained(cfg.tokenizer_name, cache_dir=cfg.cache_dir)


def build_sampling_params(cfg: LLMConfig) -> SamplingParams:
    return SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty,
        max_tokens=cfg.max_tokens,
    )


def build_llm(cfg: LLMConfig) -> LLM:
    return LLM(
        model=cfg.model_path,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_mem_util,
        trust_remote_code=cfg.trust_remote_code,
    )


# ----------------------------
# Prompting helpers
# ----------------------------

def apply_chat_template(user_text: str, tokenizer) -> str:
    chat = [{"role": "user", "content": user_text.strip()}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def call_llm(user_text: str, model: LLM, tokenizer, sampling_params: SamplingParams) -> str:
    text = apply_chat_template(user_text, tokenizer)
    outputs = model.generate([text], sampling_params)
    return outputs[0].outputs[0].text.strip()


# ----------------------------
# I/O utilities
# ----------------------------

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------
# Parsing helpers
# ----------------------------
_DECOMP_LINE = re.compile(r"^#+\s*(?:Q\s*\d+[:\-\.]|)(.*)$", re.IGNORECASE)


def parse_decomposed_items(text: str) -> List[Dict[str, Any]]:
    """Parse lines starting with ### into structured sub-questions/claims.
    Accepts variations like "### Q1: ..." or just "### ...".
    """
    items: List[Dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("###"):
            continue
        m = _DECOMP_LINE.match(line)
        if not m:
            # Fallback: drop leading hashes and treat remainder as text
            content = line.lstrip("# ").strip()
        else:
            content = m.group(1).strip() or line.lstrip("# ").strip()
        if content:
            items.append({"text": content, "needs_context": True})
    return items


# ----------------------------
# Small helpers
# ----------------------------
def unique_take(iterable: Iterable[str], k: int) -> List[str]:
    seen = {}
    out: List[str] = []
    for x in iterable:
        if x not in seen:
            seen[x] = 1
            out.append(x)
            if len(out) >= k:
                break
    return out
