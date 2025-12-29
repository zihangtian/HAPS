from agent import Agent
from joint_rl.utils import (
    set_global_seed,
    f1_score,
    em,
    read_jsonl,
    load_txt,
)
from joint_rl.rl_batch import run
import os
import json
import time
import argparse
from multiprocessing import Pool
import random
set_global_seed(43)

PRESETS = {
    "llama_qwen": {
        "cdp": {0: "Llama-3.1-8B-Instruct", 1: "Qwen2.5-7B-Instruct"},
        "short_name": "Llama_8B-Qwen_7B",
    },
    "mistral_qwen": {
        "cdp": {0: "Mistral-7B-Instruct-v0.3", 1: "Qwen2.5-7B-Instruct"},
        "short_name": "Mistral_7B-Qwen_7B",
    },
    "llama_mistral": {
        "cdp": {0: "Llama-3.1-8B-Instruct", 1: "Mistral-7B-Instruct-v0.3"},
        "short_name": "Llama_8B-Mistral_7B",
    },
    "llama8b_qwen3b": {
        "cdp": {0: "Llama-3.1-8B-Instruct", 1: "Qwen2.5-3B-Instruct"},
        "short_name": "Llama_8B-Qwen_3B",
    },
    "llama8b_qwen14b": {
        "cdp": {0: "Llama-3.1-8B-Instruct", 1: "Qwen2.5-14B-Instruct"},
        "short_name": "Llama_8B-Qwen_14B",
    }
}

ACTIONS_DICT = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
def process_question(
    qid: int, 
    qa_data: dict, 
    teacher: Agent, 
    student: Agent, 
    model_dict: dict, 
    id2agent: dict, 
    base_generation_path: str, 
    unique: str, 
    processed_pairs: set,
    budget: int = 3,
):
    output_list = []
    actions_list = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
    question = qa_data['question']
    answer = qa_data['answer']
    data = [''.join(sents) for _, sents in qa_data["context"]]
    rng = random.Random(qid)
    action_indices = list(ACTIONS_DICT.keys())
    rng.shuffle(action_indices)

    for i in action_indices:
        action = actions_list[i]
        if (question, i) in processed_pairs:
            print(f"Skipping already processed: Question: {question}, action_index: {i}")
            continue

        for aid, agent in id2agent.items():
            if isinstance(agent, Agent):
                agent.model = model_dict[action[aid]]

        print(f"Teacher Model: {teacher.model}, Student Model: {student.model}")
        agents_answer = run(question, teacher, student, data, budget)
        f1_val, precision, recall = f1_score(agents_answer, answer)
        em_score = em(agents_answer, answer)
        output_dict = {
            "qid": qid,
            "action_index": i,
            "question": question,
            "teacher": teacher.model,
            "student": student.model,
            "answer": agents_answer,
            "ground_truth": answer,
            "f1": f1_val,
            "precision": precision,
            "recall": recall,
            "em": em_score
        }

        with open(f"{base_generation_path}/{unique}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")

        output_list.append(output_dict)

    return output_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="train", help="Dataset type: train, dev, etc.")
    parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()), default="llama_qwen", help="Model preset")
    parser.add_argument("--workers", type=int, default=32, help="Number of multiprocessing workers")
    parser.add_argument("--budget", type=int, default=3, help="Search budget")
    args = parser.parse_args()

    selected_config = PRESETS[args.preset]
    model_dict = selected_config["cdp"]
    short_name = selected_config["short_name"]
    
    print(f"--- Configuration ---")
    print(f"Data Type: {args.data_type}")
    print(f"Preset: {args.preset} ({short_name})")
    print(f"Models: {model_dict}")
    print(f"Workers: {args.workers}")
    print(f"---------------------")
    dataset_path = f"dataset/{args.data_type}.jsonl"
    base_generation_path = f"dataset/{short_name}/{args.data_type}"
    retrieve_url = "http://127.0.0.1:2022/retrieve"
    unique = "baseline"
    os.makedirs(base_generation_path, exist_ok=True)
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    qa_list = read_jsonl(dataset_path)
    print(f"Loaded {len(qa_list)} questions.")
    teacher = Agent(0, "teacher", 
                    load_txt('prompt/teacher/prompt1.txt'),
                    load_txt('prompt/teacher/prompt2.txt'),
                    retrieve_url)

    student = Agent(1, "student", 
                    load_txt('prompt/student/prompt1.txt'),
                    load_txt('prompt/student/prompt2.txt'),
                    retrieve_url)

    id2agent = {0: teacher, 1: student}
    processed_pairs = set()
    processed_file_path = f"{base_generation_path}/{unique}.jsonl"
    if os.path.exists(processed_file_path):
        with open(processed_file_path, 'r', encoding='utf-8') as pf:
            for line in pf:
                if line.strip():
                    try:
                        rd = json.loads(line.strip())
                        q_question = rd['question']
                        a_index = rd['action_index']
                        processed_pairs.add((q_question, a_index))
                    except:
                        print(f"Error in reading: {line.strip()}")
                        breakpoint()

    print(f"Already processed items: {len(processed_pairs)}")

    with Pool(args.workers) as pool:
        results = []
        for qid, qa_data in enumerate(qa_list):
            result = pool.apply_async(
                process_question, 
                (
                    qid, 
                    qa_data, 
                    teacher, 
                    student, 
                    model_dict, 
                    id2agent, 
                    base_generation_path, 
                    unique, 
                    processed_pairs,
                    args.budget
                )
            )
            results.append(result)
        for i, res in enumerate(results):
            try:
                res.get()
                if i % 100 == 0:
                    print(f"Processed {i}/{len(results)} tasks...")
            except Exception as e:
                print(f"Worker Exception: {e}")

    print("All questions have been processed!")

if __name__ == '__main__':
    print("Start Time:", time.ctime())
    main()
    print("End Time:", time.ctime())
