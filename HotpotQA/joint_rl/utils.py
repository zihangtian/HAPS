from pathlib import Path
import json
from typing import Any, Dict, List
from typing import Tuple, Optional
from collections import Counter
import string
import re
import json
import torch
import numpy as np
import random
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM

def set_global_seed(seed: int):
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def em(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    reward = 0
    if normalized_ground_truth in ['yes', 'no'] and normalized_ground_truth in normalized_prediction[:3]:
        reward = 1
    elif normalized_ground_truth == normalized_prediction:
        reward = 1
    return reward


def qa_eval(metric, result_dir, q_num, trial_num):
    output_path = result_dir + 'results_' + metric + '.txt'
    results = []
    score_dir = result_dir + 'scores.json'
    with open(score_dir, 'r', encoding='utf-8') as f:
        q_scores = json.load(f)
    for qid in range(q_num):
        qid_scores = [q[1]['reward'] for q in q_scores[str(qid)].items()]
        qid_scores = qid_scores + [0] * (trial_num - len(qid_scores))

        if metric == 'f1':
            for i in range(trial_num):
                if qid_scores[i] > 0:
                    tmp_score = qid_scores[i]
                    while i < trial_num:
                        qid_scores[i] = tmp_score
                        i += 1
        elif metric == 'em':
            for i in range(trial_num):
                if qid_scores[i] == 1:
                    tmp_score = qid_scores[i]
                    while i < trial_num:
                        qid_scores[i] = tmp_score
                        i += 1

        results.append(qid_scores)

    for trial in range(trial_num):
        trial_success = 0
        for qid in range(q_num):
            if metric == 'f1':
                trial_success += (results[qid][trial] > 0)
            elif metric == 'em':
                trial_success += (results[qid][trial] == 1)
        with open(output_path, 'a+', encoding='utf-8') as f:
            f.write("Trial Number: " + str(trial) + '\n')
            f.write("Success Ratio: " + str(trial_success / q_num) + '\n\n')

def read_jsonl(path: str | Path) -> list[Any]:
    """Read a JSONL file and return a list of parsed records."""
    items: list[Any] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def load_txt(file_path:str):
    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()

    return file_content

def parse_router_output(txt: str) -> Tuple[int,int]:
    l, r = txt.rfind("{"), txt.rfind("}") + 1
    try:
        obj = json.loads(txt[l:r])
        return int(obj["teacher"]), int(obj["student"])
    except Exception:
        print("[ERROR!] parse_router_output failed:", txt)
        return 0, 0


def align_model_and_tokenizer(model:AutoModelForCausalLM, 
                              tok:AutoTokenizer
                            ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    gen_cfg = model.generation_config
    if getattr(gen_cfg, "pad_token_id", None) is None:
        gen_cfg.pad_token_id = tok.pad_token_id
    if getattr(gen_cfg, "eos_token_id", None) is None and tok.eos_token_id is not None:
        gen_cfg.eos_token_id = tok.eos_token_id
    return model, tok

def load_mNt(
    model_name: str,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map=device
    )
    model, tok = align_model_and_tokenizer(model, tok)
    return model, tok


def build_prompt(tokenizer: AutoTokenizer, sys_desc: str, user_prompt: str, *, add_gen: bool = True) -> str:
    messages: List[Dict[str, str]] = []
    if sys_desc and sys_desc.strip():
        messages.append({"role": "system", "content": sys_desc})
    messages.append({"role": "user", "content": user_prompt})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_gen
    )


def route(formatted_prompt:str, router_model:AutoModelForCausalLM, router_tokenizer: AutoTokenizer)->str:
    input_ids = router_tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True).to(router_model.device)
    input_len = len(input_ids.input_ids[0])
    max_new_tokens = 16

    with torch.no_grad():
        outputs = router_model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=router_tokenizer.pad_token_id,
            eos_token_id=router_tokenizer.eos_token_id,
        )
    
    generated_text = router_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return generated_text


def route_batch(formatted_prompts: list[str], router_model, router_tokenizer, max_new_tokens: int = 16) -> list[str]:
    inputs = router_tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(router_model.device)
    attn = inputs["attention_mask"]
    input_lens = attn.sum(dim=1)
    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16):
            outputs = router_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=router_tokenizer.pad_token_id,
                eos_token_id=router_tokenizer.eos_token_id,
            )
        outs = []
        for i in range(outputs.size(0)):
            gen_seq = outputs[i, input_lens[i]:]
            outs.append(router_tokenizer.decode(gen_seq, skip_special_tokens=True))
        return outs
