import os
import json
from transformers import AutoTokenizer
from vllm import SamplingParams, LLM
import torch 
import torch.nn.functional as F


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def load_tokenizer(model_name, cache_dir=None):
    return AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)

def init_llm(model_path, tensor_parallel_size=1, gpu_memory_utilization=0.88):
    return LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True
    )

def call_llm(prompt: str, model, tokenizer, sampling_params) -> str:
    """
    Calls the LLM using vllm's generate function.
    Applies a chat template and returns the generated text.
    """
    # Prepare the chat-style input for the LLM
    prompt_input = [{"role": "user", "content": prompt.strip()}]
    # print("LLM prompt:", prompt, prompt_input)
    
    # Apply chat template to build full prompt text
    text = tokenizer.apply_chat_template(
        prompt_input,
        tokenize=False,
        add_generation_prompt=True
    )
    outputs = model.generate([text], sampling_params)
    # Retrieve the generated text from output
    generated_text = outputs[0].outputs[0].text
    return generated_text.strip()


def make_sampling_params(temperature=0.0, top_p=0.99, max_tokens=2048):
    return SamplingParams(temperature=temperature, top_p=top_p, repetition_penalty=1.05, max_tokens=max_tokens)

def apply_chat_template(tokenizer, messages, enable_thinking=False):
    # Handles special params for some models
    if "qwen3" in tokenizer.name_or_path:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


### utils for embedding models
def embed_text(text: str, tokenizer, model, embedding_model_name: str) -> torch.Tensor:
    """
    Converts text to an embedding vector using the specified embedding model.
    Adjusts tokenization based on model name.
    """
    if "e5" in embedding_model_name:
        encoded_input = tokenizer("query: " + text, max_length=128, padding=True, truncation=True, return_tensors='pt')
    else:
        encoded_input = tokenizer(text, max_length=100, padding=True, truncation=True, return_tensors='pt')
    # Move input tensors to GPU
    for key in encoded_input:
        encoded_input[key] = encoded_input[key].cuda()
    
    # Compute embeddings based on model type
    with torch.no_grad():
        if embedding_model_name in ["gte-base"]:
            model_output = model(**encoded_input)
            sentence_embeddings = model_output.last_hidden_state[:, 0].detach().cpu()
        elif "e5" in embedding_model_name:
            outputs = model(**encoded_input)
            embeddings = average_pool(outputs.last_hidden_state, encoded_input['attention_mask'])
            sentence_embeddings = embeddings.detach().cpu()
        elif embedding_model_name in ["dragon"]:
            embeddings = model(**encoded_input, output_hidden_states=True, return_dict=True).last_hidden_state[:, 0, :]
            sentence_embeddings = embeddings.detach().cpu()
    
    if "e5" in embedding_model_name or "gte" in embedding_model_name:
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1).squeeze(1)
    return sentence_embeddings

def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Performs average pooling on the token embeddings using the attention mask.
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def generate_final_answer(original_question: str, sub_questions: Dict[str, str], sub_answers: Dict[str, str],
                          model, tokenizer, sampling_params, dataset: str, passages: List[str] = None, add_passage: int = 1) -> str:
    """
    Generates a final answer for the original question by summarizing sub-question answers.
    """
    sub_answer_text = "\n".join([f"### {k}: {sub_questions[k]}, Answer for {k}: {v}" for k, v in sub_answers.items()])
    final_prompt = ("True or False only." 
                    if dataset in ["strategyqa"] 
                    else "a short span")