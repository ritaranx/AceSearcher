# creating DPO pairs
import re
import string
import json
from tqdm import trange, tqdm
from collections import defaultdict
import copy, os
from typing import  Dict
import math

word2number = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten"
}

def extract_code_block(text: str):
    """
    Extracts the last Python code block from a given string.
    
    Args:
        text (str): The input string containing potential code blocks.

    Returns:
        str or None: The last extracted Python code block, or None if no code block is found.
    """
    pattern = r"```python\s+([\s\S]*?)\s*```"
    matches = re.findall(pattern, text)

    return matches[-1] if matches else None

def clean_string(text):
    text = text.strip()
    return re.sub(r'[^\w\s]', '', text)

def normalize_answer(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_tokens = normalize_answer(a_gold).split()
    pred_tokens = normalize_answer(a_pred).split()
    common = set(gold_tokens) & set(pred_tokens)
    num_same = len(common)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def compute_acc(a_gold, a_pred):
    gold_tokens = normalize_answer(a_gold).lower()
    pred_tokens = normalize_answer(a_pred).lower()
    if pred_tokens in gold_tokens or gold_tokens in pred_tokens:
        return 1 
    else:
        return 0

def extract_answer(str):
    if "<answer>" in str and "</answer>" in str:
        return str.split("<answer>")[-1].split("</answer>")[0].strip()
    else:
        return ""

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

def get_reward_mhqa(dataset, path, model, round):
    files = os.listdir(path)
    examples = []
    for file in files:
        with open(f"{path}/{file}", "r") as f:
            for lines in f:
                examples.append(json.loads(lines))
    print(dataset, model, "num examples:", len(examples))
    # calculate both question and qd
    decompose_reward = {}
    id_to_question = {}
    id_to_decompose = {}
    id_to_subanswers = {}
    id_to_subpassages = {}
    id_to_answers = {}
    id_to_gold_answer = {}
    qd_pairs = []
    qa_pairs = []
    cot_pairs = []
    for example in examples:
        question_id = example["question_id"]
        decompose_id = example["decompose_id"]
        answer_id = example["answer_id"]
        prediction = example["final_answer"]
        intermediate_answers = example["intermediate_answers"]
        ground_truth = example["answer"]
        id_to_question[question_id] = example["question"]
        id_to_decompose[f"{question_id}-{decompose_id}"] = example["decomposed"]
        id_to_subanswers[f"{question_id}-{decompose_id}-{answer_id}"] = example["intermediate_answers"]
        id_to_subpassages[f"{question_id}-{decompose_id}-{answer_id}"] = example["intermediate_passages"]
        id_to_answers[f"{question_id}-{decompose_id}-{answer_id}"] = example["final_answer"]
        id_to_gold_answer[question_id] = ground_truth
        em_score = 0
        f1_score = 0
        acc_score = 0
        for gt in example["answer"]:
            prediction = clean_string(extract_answer(example["final_answer"])).lower()
            if prediction in word2number:
                prediction = word2number[prediction]
            f1_score = max(f1_score, compute_f1(gt, prediction))
            acc_score = max(acc_score, compute_acc(gt, prediction))
            if gt.lower() in ["yes", "no"]:
                em_score = acc_score
                f1_score = acc_score
            else:
                em_score = max(em_score, compute_exact(gt, prediction))
        if question_id not in decompose_reward:
            decompose_reward[question_id] = defaultdict(dict)
        decompose_reward[question_id][decompose_id][answer_id] = em_score + acc_score + f1_score
    i = 0
    for x in decompose_reward:
        question = id_to_question[x]
        # Compute average values
        averages = {k: sum(v.values()) / len(v) for k, v in decompose_reward[x].items()}
        if len(averages) == 1:
            continue
        # Find max and min average values
        max_avg = max(averages.values())
        min_avg = min(averages.values())
        # Find all keys with those values
        max_avg_keys = [k for k, v in averages.items() if v == max_avg]
        min_avg_keys = [k for k, v in averages.items() if v == min_avg]
        # if max_avg != min_avg:
        #     print(decompose_reward[x], averages, max_avg_keys, min_avg_keys)
        #     assert 0
        if max_avg - min_avg >= 1.33 and min_avg <= 1:
            print(dict(decompose_reward[x]), question)
            print("Max avg keys:")
            positive_qd = []
            negative_qd = []
            for k_pos in max_avg_keys:
                pos_qd_tmp = id_to_decompose[f'{x}-{k_pos}']
                no_subq = 1 
                for q in pos_qd_tmp:
                    if "subquestion" in q["text"].lower() and "#" not in q["text"].lower():
                        no_subq = 0
                        break
                if len(pos_qd_tmp) < 7 and no_subq == 1:
                    positive_qd.append(pos_qd_tmp)
                else:
                    negative_qd.append(pos_qd_tmp)
            for k_neg in min_avg_keys:
                neg_qd_tmp = id_to_decompose[f'{x}-{k_neg}']
                negative_qd.append(neg_qd_tmp) 
            if len(positive_qd) > 0 and len(negative_qd) > 0:
                qd_pairs.append({
                    "question": question,
                    "gold": id_to_gold_answer[x],
                    "positive": positive_qd,
                    "negative": negative_qd,
                    "positive_score": max_avg,
                    "negative_score": min_avg
                })
        visit_subq = {}
        for k, subdict in decompose_reward[x].items():
            max_val = max(subdict.values())
            min_val = min(subdict.values())
            if (max_val - min_val) <= 1.33 or min_val > 1.2:
                continue
            print("======")
            max_items = [i for i, v in subdict.items() if v == max_val][:2]
            min_items = [i for i, v in subdict.items() if v == min_val][:2]
            print(f"For key {k}: max items = {max_val}: {max_items}, min items = {min_val}: {min_items}")
            print(id_to_decompose[f"{x}-{k}"], id_to_gold_answer[x] )
            for y_max in max_items:
                subpassages = id_to_subpassages[f"{x}-{k}-{y_max}"]
                passages = []
                for subq in subpassages:
                    passages += subpassages[subq][:5] if len(subpassages) <= 3 else subpassages[subq][:3]
                all_passages =  "\n\n".join(list(set(passages))) #list(set(all_passages))
                visit = 0
                tmp_sim = {}
                tmp_sim_max = -1
                for y_min in min_items:
                    subanswers_max = id_to_subanswers[f"{x}-{k}-{y_max}"]
                    subanswers_min = id_to_subanswers[f"{x}-{k}-{y_min}"]
                    sim = 0
                    for k_ in subanswers_min:
                        if "." in k_:
                            continue
                        if subanswers_max[k_].lower() in word2number:
                            subanswers_max[k_] = word2number[subanswers_max[k_].lower()]
                        if subanswers_min[k_].lower() in word2number:
                            subanswers_min[k_] = word2number[subanswers_min[k_].lower()]
                        if k_ in subanswers_max and subanswers_max[k_].lower().strip() == subanswers_min[k_].lower().strip():
                            sim += 1
                        else:
                            if compute_f1(subanswers_max[k_], subanswers_min[k_]) == 0 and visit == 0:
                                subquestion = id_to_decompose[f'{x}-{k}'][int(k_.strip("Q")) - 1]["text"] 
                                passages = subpassages[k_]
                                positive_ans = subanswers_max[k_]
                                negative_ans =  subanswers_min[k_]
                                if subquestion not in visit_subq:
                                    subquestion_complete = replace_placeholders(subquestion, subanswers_max)
                                    qa_pairs.append({
                                        "question": subquestion_complete, 
                                        "passages": passages,
                                        "positive": positive_ans,
                                        "negative": negative_ans
                                    })
                                    visit_subq[subquestion] = 1
                                visit = 1
                    sim = sim / len(subanswers_min)
                    if sim >= tmp_sim_max:
                        tmp_sim = {
                            "question": question, 
                            "subquestions": id_to_decompose[f'{x}-{k}'],
                            "subanswers": subanswers_max,
                            "passages": all_passages,
                            "positive": id_to_answers[f"{x}-{k}-{y_max}"],
                            "negative": id_to_answers[f"{x}-{k}-{y_min}"],
                            "gold": id_to_gold_answer[x],
                            "positive_score": max_val,
                            "negative_score": min_val
                        }
                        tmp_sim_max = sim
                if tmp_sim_max >= 0.666:
                    cot_pairs.append(copy.deepcopy(tmp_sim))
        i += 1
    print(dataset, model, f'r-{round}', "COT:", len(cot_pairs), "QA:", len(qa_pairs), "QD:", len(qd_pairs))
    os.makedirs(f"processed_data/mhqa-r{round}", exist_ok=True) 
    with open(f"processed_data/mhqa-r{round}/{model}-{dataset}-mhqa-cot.json", "w") as f:
        json.dump(cot_pairs, f, indent = 2)
    
    with open(f"processed_data/mhqa-r{round}/{model}-{dataset}-mhqa-qa.json", "w") as f:
        json.dump(qa_pairs, f, indent = 2)

    with open(f"processed_data/mhqa-r{round}/{model}-{dataset}-mhqa-qd.json", "w") as f:
        json.dump(qd_pairs, f, indent = 2)


def get_rag_cot(path, dataset, model, round):
    cot_pairs = []
    with open(path, 'r') as f:
        examples = json.load(f)
        for example in examples:
            if example["cot"]:
                positive = [example["cot"]]
                negative = []
                for generation in example["generation"]:
                    pred = extract_answer(generation).replace(",", "")
                    gold = example["answer"].replace(",", "")
                    if pred.lower().strip() != gold.lower().strip():
                        negative.append(generation)
                    else:
                        positive.append(generation)
                for p, n in zip(positive, negative):
                    cot_pairs.append({
                        "question": example["prompt"],
                        "positive": p,
                        "negative": n
                    })
                    if len(cot_pairs) < 10:
                        print("positive:", p, "\nnegative:", n)
                        print("======")
                
    print("COT RAG pairs", dataset, len(cot_pairs))
    os.makedirs(f"processed_data/cot-r{round}", exist_ok=True) 
    with open(f"processed_data/cot-r{round}/{model}-{dataset}-rag.json", "w") as f:
        json.dump(cot_pairs, f, indent = 2)


def get_rag_ifqa(path, model, round):
    cot_pairs = []
    with open(path, 'r') as f:
        examples = json.load(f)
        for example in examples:
            gold = example["answer"]
            positive = []
            negative = []
            for generation in example["generation"]:
                pred = extract_answer(generation)
                correct = False
                for x in gold:
                    if pred.lower().strip() == x.lower().strip():
                        correct = True
                        positive.append(generation)
                    elif x.lower().strip() in pred.lower().strip():
                        correct = True
                if not correct:
                    negative.append(generation)
            if len(positive) > 0 and len(negative) > 0:
                cot_pairs.append({
                        "question": example["prompt"],
                        "positive": positive[0],
                        "negative": negative[0],
                        "answer": gold
                })
    print("COT ifqa examples:", len(cot_pairs))
    os.makedirs(f"processed_data/cot-r{round}", exist_ok=True) 
    with open(f"processed_data/cot-r{round}/{model}-ifqa-rag.json", "w") as f:
        json.dump(cot_pairs, f, indent = 2)
    

def get_rag_code(path, dataset, model, round):
    pot_pairs_pd = {}
    pot_pairs_code = {}
    idx_to_qd = {}
    idx_to_code = {}
    idx_to_context = {}
    qa_pairs = []
    qd_pairs = []
    with open(path, 'r') as f:
        examples = json.load(f)
        for example in tqdm(examples):
            question = example["question"]
            decompose_id = example["decompose_id"]
            decompose = example["decompose"]
            code_id = example["code_id"]
            code = extract_code_block(example["code"]) 
            if code is None:
                code = example["code"]
            answers = example["answer"]
            if question not in pot_pairs_pd:
                pot_pairs_pd[question] = defaultdict(list)
                pot_pairs_code[question] = defaultdict(dict)
            idx_to_qd[f"{question}-{decompose_id}"] = decompose
            idx_to_code[f"{question}-{decompose_id}-{code_id}"] = code
            idx_to_context[question] = example
            namespace = {}
            try:
                correct = 0
                if "while" in code or "exit()" in code:
                    continue
                exec(code, namespace)
                # Now you can get the value of 'ans'
                result = namespace['ans']
                if dataset in ["gsm8k", "TabMWP"]:
                    try:    
                        answers = [float(answers)]
                    except:
                        answers = [answers]
                for answer in answers:
                    if isinstance(answer, float) or isinstance(answer, int):
                        if math.isclose(answer, result, abs_tol=1e-2) or math.isclose(answer, result*100, abs_tol=1e-3) or math.isclose(answer, result/100, abs_tol=1e-3):
                            correct = 1
                            break
                    else:
                        if answer.strip().lower() == str(result).strip().lower():
                            correct = 1
                            break
            except Exception as e:
                print("exception", e, answer, type(answer))
            pot_pairs_pd[question][decompose_id].append(correct) 
            pot_pairs_code[question][decompose_id][code_id] = correct
    i = 0
    print(len(idx_to_context))
    for question in pot_pairs_pd:
        i += 1
        averages = {k: sum(v.values()) / len(v) for k, v in pot_pairs_code[question].items()}
        if not averages.values():
            continue
         # Find max and min average values
        max_avg = max(averages.values())
        min_avg = min(averages.values())
        # Find all keys with those values
        max_avg_keys = [k for k, v in averages.items() if v == max_avg]
        min_avg_keys = [k for k, v in averages.items() if v == min_avg]
        if max_avg - min_avg > 0.3:
            save_example = copy.deepcopy(idx_to_context[question])
            save_example["positive"] = [
               idx_to_qd[f"{question}-{x}"] for x in max_avg_keys
            ]
            save_example["negative"] = [
               idx_to_qd[f"{question}-{x}"] for x in min_avg_keys
            ]
            qd_pairs.append(save_example)
        for decompose_id in pot_pairs_code[question]:
            decompose = idx_to_qd[f"{question}-{decompose_id}"]

            averages_code = pot_pairs_code[question][decompose_id]
            
            positive = [code_id for code_id in averages_code if  averages_code[code_id] == 1]
            negative = [code_id for code_id in averages_code if  averages_code[code_id] == 0]

            save_example = copy.deepcopy(idx_to_context[question])
            save_example["decompose"] = idx_to_qd[f"{question}-{decompose_id}"]
            save_example["positive"] = [idx_to_code[f"{question}-{decompose_id}-{code_id}"] for code_id in positive]
            save_example["negative"] = [idx_to_code[f"{question}-{decompose_id}-{code_id}"] for code_id in negative]
            if positive and negative:
                qa_pairs.append(save_example)
    print(model, dataset, len(qa_pairs), len(qd_pairs))
    os.makedirs(f"processed_data/code-r{round}", exist_ok=True) 
    with open(f"processed_data/code-r{round}/{model}-{dataset}-pot.json", "w") as f:
        json.dump(qa_pairs, f, indent = 2)

    with open(f"processed_data/code-r{round}/{model}-{dataset}-qd.json", "w") as f:
        json.dump(qd_pairs, f, indent = 2)


model = "llama_8b"
round = 2
####### mhqa #######
for dataset in ["hover", "hotpotqa", "2wikimultihopqa", "musique"]:
    get_reward_mhqa(dataset, f"processed_data/{dataset}/prompts_qa_train_{model}_r{round}", model, round = round)
#####################

for dataset in ["TabMWP", "gsm8k"]:
    get_rag_cot(f"processed_data/cot/prompts_cot_{dataset}_{model}_r{round}.json", dataset, model, round = round)

get_rag_ifqa(f"processed_data/cot/prompts_cot_ifqa_{model}_r{round}.json", model, round = round)


for dataset in ["gsm8k", "convfinqa", "TabMWP"]:
    print(model, dataset)
    get_rag_code(f"processed_data/code/prompts_code_{dataset}_{model}_r{round}.json", dataset, model, round = round)
