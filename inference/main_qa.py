import argparse
import copy
import csv
import json
import numpy as np
from typing import List, Dict
from tqdm import tqdm, trange
from utils import load_tokenizer, init_llm, make_sampling_params, load_jsonl, save_jsonl, call_llm, embed_text
import re
from transformers import AutoTokenizer, AutoModel
import faiss
import pickle as pkl


# Import other retrieval/embedding code as needed
def replace_placeholders(question_text: str, answers_so_far: Dict[str, str]) -> str:
    """
    Replaces placeholders like "#1", "#2", etc. in the question text with answers from previous sub-questions.
    """
    matches = re.findall(r"#(\d+)", question_text)
    for m in matches:
        placeholder = f"#{m}"
        q_key = f"Q{m}"
        if q_key in answers_so_far:
            question_text = question_text.replace(placeholder, answers_so_far[q_key])
    return question_text


def retrieve_context(query: str, cpu_index, corpus: List[str], embedding_tokenizer, embedding_model, embedding_model_name: str, top_k: int = 3) -> List[str]:
    """
    Retrieves the top-k context passages from the vector store for a given query.
    """
    query_embedding = embed_text(query, embedding_tokenizer, embedding_model, embedding_model_name)
    dev_D, dev_I = cpu_index.search(query_embedding, top_k)
    passages = [corpus[r] for r in dev_I[0]]
    return passages

###################################
# Data Loading and Index Building
###################################
def load_embedding(dataset: str, embedding_model: str):
    """
    Loads corpus passages and their pre-computed embeddings from disk.
    """
    sentences = []
    passage_embeddings = []
    
    # Load sentences from corpus file
    with open(f"embeddings/{dataset}/corpus.tsv", "r") as f:
        reader = csv.reader(f, delimiter='\t')
        for lines in tqdm(reader):
            if lines[0] == "id":
                continue
            text = lines[1]
            sentences.append(text)
    
    # Load embeddings in 4 shards and concatenate them
    for i in trange(4): 
        path = f"embeddings/{dataset}/{embedding_model}/embeddings-{i}-of-4.pkl"
        with open(path, "rb") as f:
            passage_embedding = pkl.load(f)
            passage_embeddings.append(passage_embedding)
    passage_embeddings = np.concatenate(passage_embeddings, axis=0)
    print("Passage Size:", passage_embeddings.shape)
    return sentences, passage_embeddings


def load_data(dataset: str, expname: str, save_dir: str) -> List[Dict]:
    """
    Loads questions from a JSONL file for the given dataset.
    """
    questions = []
    if "-" in dataset:
        dataset = dataset.split("-")[0]
    with open(f"{save_dir}/{dataset}/prompts_decompose_test_t0.0_{expname}/generate.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            questions.append(data)
    print("========")
    print(f"Loaded {len(questions)} examples from {dataset}!")
    print("========")
    return questions


def build_index(dataset: str, embedding_model_name: str):
    """
    Builds a FAISS index from pre-computed embeddings.
    """
    corpus, embeddings = load_embedding(dataset, embedding_model_name)
    dim = embeddings.shape[1]
    faiss.omp_set_num_threads(32)
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(embeddings)
    return corpus, embeddings, cpu_index

###################################
# Answering Functions
###################################
def zigzag_visit(lst: List) -> List:
    """
    Reorders the input list in a zigzag fashion.
    Example:
        Input: [1, 2, 3, 4, 5, 6, 7]
        Output: [1, 3, 5, 7, 6, 4, 2]
    """
    n = len(lst)
    result = [None] * n
    
    # Fill first half (odd indices)
    i, j = 0, 0
    while j < (n + 1) // 2:
        result[j] = lst[i]
        i += 2
        j += 1

    # Fill second half (even indices)
    i = 1
    j = n - 1
    while j >= (n + 1) // 2:
        result[j] = lst[i]
        i += 2
        j -= 1
    
    return result

def answer_sub_claim(sub_q: str, context_passages: List[str], model, tokenizer, sampling_params) -> str:
    """
    Uses the LLM to answer a sub-question given the retrieved context.
    The context passages are reordered in a zigzag manner before being concatenated.
    """
    reordered_passages = zigzag_visit(context_passages)
    context_text = "\n\n".join(reordered_passages)
    prompt = f"""You have the following context passages:
{context_text}

Please verify whether the claim '{sub_q}' is correct using the context as reference. 
If no answer is found in the context, use your own knowledge.
Please only output Yes or No and do not give any explanation."""

    response = call_llm(prompt, model=model, tokenizer=tokenizer, sampling_params=sampling_params)
    return response.strip()

def answer_sub_question(sub_q: str, context_passages: List[str], model, tokenizer, sampling_params) -> str:
    """
    Uses the LLM to answer a sub-question given the retrieved context.
    The context passages are reordered in a zigzag manner before being concatenated.
    """
    reordered_passages = zigzag_visit(context_passages)
    context_text = "\n\n".join(reordered_passages)
    prompt = f"""You have the following context passages:
{context_text}

Please answer the question '{sub_q}' with a short span using the context as reference.
If no answer is found in the context, use your own knowledge. Your answer needs to be as short as possible."""
    response = call_llm(prompt, model=model, tokenizer=tokenizer, sampling_params=sampling_params)
    return response.strip()


def multi_turn_qa(question: str, sub_questions: List[Dict], cpu_index, corpus,
                  embedding_tokenizer, embedding_model, embedding_model_name: str,
                  llm_model, llm_tokenizer, sampling_params, dataset: str, add_passage, topk: int):
    """
    Orchestrates the multi-turn QA process:
    1. Resolve any placeholder references in sub-questions.
    2. Retrieve context if needed.
    3. Answer each sub-question.
    4. Combine sub-answers into a final answer.
    """
    # Create dictionaries to hold resolved sub-questions and answers
    subquestions_dict = {subq_dict["label"]: subq_dict["text"] for subq_dict in sub_questions}
    answer_dict = {}
    passage_dict = {}
    all_passages = []
    # Process each sub-question
    for subq_dict in sub_questions:
        q_label = subq_dict["label"]
        q_text = subq_dict["text"]
        
        # Replace placeholders (e.g., #1, #2) with previous answers
        q_text_resolved = replace_placeholders(q_text, answer_dict)
        
        passages = []
        # Retrieve context if required
        passages = retrieve_context(q_text_resolved, cpu_index, corpus, embedding_tokenizer,
                                        embedding_model, embedding_model_name, top_k=topk)
        all_passages += passages[:5] if len(sub_questions) <= 3 else passages[:3]
        all_passages = list(set(all_passages))
        # Answer the sub-question
        sub_answer = answer_sub_question(q_text_resolved, passages, llm_model, llm_tokenizer, sampling_params)
        answer_dict[q_label] = sub_answer
        passage_dict[q_label] = passages
        subquestions_dict[q_label] = q_text_resolved

    # Generate final answer based on sub-answers
    final_answer = generate_final_answer(question, subquestions_dict, answer_dict,
                                         llm_model, llm_tokenizer, sampling_params, dataset, all_passages, add_passage)
    print("-------\nquestion:", question,
          "\nsub-questions:", sub_questions, 
          "\nanswers:", answer_dict, 
          "\nFinal Answer:", final_answer)
    return final_answer, answer_dict, passage_dict


def generate_final_answer(original_question: str, sub_questions: Dict[str, str], sub_answers: Dict[str, str], model, tokenizer, sampling_params, dataset: str, passages: List[str] = None, add_passage: int = 1) -> str:
    """
    Generates a final answer for the original question by summarizing sub-question answers.
    """
    sub_answer_text = "\n".join([f"### {k}: {sub_questions[k]}, Answer for {k}: {v}" for k, v in sub_answers.items()])
    final_prompt = "a short span"

    if dataset in ["hover", "exfever"]:
        prompt = f"""You are given some subquestions and their answers:
{sub_answer_text}

Please verify the correctness of the claim: '{original_question}' using the subquestions as reference. Please provide a concise and clear reasoning followed by a concise conclusion. Your answer should be Yes or No only. 
Wrap your answer with <answer> and </answer> tags."""

    else:
        if add_passage:
            passages = "\n\n".join(list(set(passages)))
            prompt = f"""You have the following passages:
{passages}

You are also given some subquestions and their answers:
{sub_answer_text}

Please answer the question '{original_question}' with {final_prompt} using the documents and subquestions as reference.
Make sure your response is grounded in documents and provides clear reasoning followed by a concise conclusion. If no relevant information is found, use your own knowledge. 
Wrap your answer with <answer> and </answer> tags."""
        else:
            prompt = f"""You are given some subquestions and their answers:
{sub_answer_text}

Please answer the question '{original_question}' with {final_prompt} using the subquestions as reference. Provides clear reasoning followed by a concise conclusion. If no relevant information is found, use your own knowledge. 
Wrap your answer with <answer> and </answer> tags."""

    final = call_llm(prompt, model=model, tokenizer=tokenizer, sampling_params=sampling_params)
    return final.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_model_path", type=str)
    parser.add_argument("--llm_tokenizer", type=str)
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--top_p", type=float, default=0.99)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--add_passage", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--sentence_embedding_model", type=str)
    parser.add_argument("--sentence_embedding_model_save_name", type=str)
    args = parser.parse_args()
    
    questions = load_jsonl(f"{args.save_dir}/{args.dataset}/prompts_decompose_test_t0.0_{args.expname}/generate.jsonl")

    # Build retrieval index as needed
    tokenizer = AutoTokenizer.from_pretrained(args.sentence_embedding_model, 
                                                trust_remote_code=True)
    embedding_model = AutoModel.from_pretrained(args.sentence_embedding_model, 
                                                trust_remote_code=True).cuda()
    embedding_model.eval()
    corpus, embeddings, cpu_index = build_index(args.dataset, args.sentence_embedding_model_save_name)
    
    
    llm_tokenizer = load_tokenizer(args.llm_tokenizer)
    sampling_params = make_sampling_params(args.temperature, args.top_p, max_tokens=512)
    llm = init_llm(args.llm_model_path, args.tensor_parallel_size)
    
    saved_examples = []
    for index, item in enumerate(tqdm(questions)):
        try:
            final_answer, intermediate_answers, intermediate_passages = multi_turn_qa(
                item["question"], item["decomposed"], cpu_index, corpus,
                tokenizer, embedding_model, args.sentence_embedding_model_save_name,
                llm, llm_tokenizer, sampling_params, args.dataset, args.add_passage, args.k
            )
            new_item = copy.deepcopy(item)
            new_item.update({
                "index": index,
                "final_answer": final_answer,
                "intermediate_answers": intermediate_answers,
                "intermediate_passages": intermediate_passages
            })
            saved_examples.append(new_item)
        except Exception as e:
            print(f"Error at {index}: {e}")
            continue
    output_path = f"{args.save_dir}/{args.dataset}/prompts_decompose_test_{args.expname}/test_{args.sentence_embedding_model_save_name}_k{args.k}_passage{args.add_passage}.jsonl"
    save_jsonl(saved_examples, output_path)
    print(f"Saved {len(saved_examples)} results to {output_path}")

if __name__ == "__main__":
    main()
