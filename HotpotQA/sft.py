import os
SEED = 43
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from joint_rl.utils import *
set_global_seed(SEED)


import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List
from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset, disable_caching
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
disable_caching()

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
}

def parse_args():
    p = argparse.ArgumentParser(description="SFT warmup router for HotpotQA (fixed presets)")

    # Choose one or many presets
    p.add_argument(
        "--pair",
        type=str,
        default=None,
        choices=list(PRESETS.keys()),
        help="Single preset to run (one of: llama_qwen, mistral_qwen, llama_mistral).",
    )
    p.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Run multiple presets sequentially, comma-separated (e.g., 'llama_qwen,mistral_qwen,llama_mistral').",
    )

    # Prompts
    p.add_argument("--description", type=Path, default=Path("prompt/router_template.txt"), help="User prompt template.")
    p.add_argument("--system_prompt", type=Path, default=Path("prompt/system_router.txt"), help="System prompt.")

    # Data paths (if omitted, they will be resolved from short_name = preset)
    p.add_argument("--sft_json", type=Path, default=None, help="SFT data JSON (overrides preset-derived path).")
    p.add_argument("--train_bsl_jsonl", type=Path, default=None, help="Train baseline table (overrides preset-derived path).")
    p.add_argument("--valid_bsl_jsonl", type=Path, default=None, help="Valid baseline table (overrides preset-derived path).")

    # Model and training
    p.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--bsz", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--eval_bs", type=int, default=64)

    # Output root; final output is {output_root}/{short_name}/
    p.add_argument("--output_root", type=Path, default=Path("sft_best_router"), help="Root output directory.")

    return p.parse_args()

from collections import Counter

def summarize_sft_labels_from_file(sft_json_path: Path, cdp: dict):
    data = json.loads(sft_json_path.read_text(encoding="utf-8"))
    total = len(data)

    # (t,s) -> action_index
    tuple2index = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}

    four_counts = Counter()
    t_counts = Counter()   # teacher: 0/1
    s_counts = Counter()   # student: 0/1
    bad = []

    for i, ex in enumerate(data):
        out = str(ex.get("output", ""))
        try:
            t, s = parse_router_output(out)
            if t in (0, 1) and s in (0, 1):
                four_counts[tuple2index[(t, s)]] += 1
                t_counts[t] += 1
                s_counts[s] += 1
            else:
                bad.append((i, out))
        except Exception:
            bad.append((i, out))

    # Pretty print
    def pct(n): 
        return f"{(100.0 * n / max(total, 1)):.2f}%"

    idx2desc = {
        0: f"T={cdp[0]} , S={cdp[0]}",
        1: f"T={cdp[0]} , S={cdp[1]}",
        2: f"T={cdp[1]} , S={cdp[0]}",
        3: f"T={cdp[1]} , S={cdp[1]}",
    }

    print("\n======== SFT label distribution (parsed from `output`) ========")
    print(f"File: {sft_json_path} | #samples = {total}")
    for k in range(4):
        n = four_counts.get(k, 0)
        print(f"  action_index={k:<1}  ({idx2desc[k]}) : {n:>6}  ({pct(n)})")
    print(f"  -- teacher=0({cdp[0]}): {t_counts.get(0,0):>6} ({pct(t_counts.get(0,0))})"
          f" | teacher=1({cdp[1]}): {t_counts.get(1,0):>6} ({pct(t_counts.get(1,0))})")
    print(f"  -- student=0({cdp[0]}): {s_counts.get(0,0):>6} ({pct(s_counts.get(0,0))})"
          f" | student=1({cdp[1]}): {s_counts.get(1,0):>6} ({pct(s_counts.get(1,0))})")
    print(f"  parse_failed / invalid : {len(bad)}\n")
    return {
        "total": total,
        "four_counts": dict(four_counts),
        "t_counts": dict(t_counts),
        "s_counts": dict(s_counts),
        "bad": bad,
    }


def resolve_runs(args) -> List[Dict]:
    run_ids = []
    if args.pairs:
        run_ids = [x.strip() for x in args.pairs.split(",") if x.strip()]
    elif args.pair:
        run_ids = [args.pair]
    else:
        # Default to single preset if nothing provided
        run_ids = ["llama_qwen"]

    runs = []
    for rid in run_ids:
        conf = deepcopy(PRESETS[rid])
        cdp = conf["cdp"]
        short_name = conf["short_name"]

        sft_json = args.sft_json or Path(f"dataset/{short_name}/train/sft_train_all.json")
        train_bsl_jsonl = args.train_bsl_jsonl or Path(f"dataset/{short_name}/train/baseline.jsonl")
        valid_bsl_jsonl = args.valid_bsl_jsonl or Path(f"dataset/{short_name}/valid/baseline.jsonl")
        output_dir = args.output_root / short_name

        runs.append(
            dict(
                cdp=cdp,
                short_name=short_name,
                sft_json=sft_json,
                train_bsl_jsonl=train_bsl_jsonl,
                valid_bsl_jsonl=valid_bsl_jsonl,
                output_dir=output_dir,
            )
        )
    return runs


def compute_uniform_baseline(metric_tbl_path: Path):
    m = load_metric_tables(metric_tbl_path)
    questions = list(dict.fromkeys(q for q, _ in m))
    totals = {a: 0.0 for a in range(4)}
    for a in range(4):
        totals[a] = sum(m.get((q, a), 0.0) for q in questions)
    best_action = max(totals, key=totals.get)
    return best_action, totals[best_action], totals, len(questions)


def build_dataset(train_data: List[Dict], sys_desc: str, usr_desc: str, cdp: dict, tok: AutoTokenizer):
    def encode_messages(question: str) -> List[int]:
        msgs = []
        if sys_desc and sys_desc.strip():
            msgs.append({"role": "system", "content": sys_desc})
        usr_txt = usr_desc.format(question=question, model_0=cdp[0], model_1=cdp[1])
        msgs.append({"role": "user", "content": usr_txt})
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True)
        return enc["input_ids"]

    def _tokenize(item: Dict):
        prompt_ids = encode_messages(item["question"])
        target_txt = str(item["output"]) + (tok.eos_token or "")
        target_ids = tok(target_txt, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + target_ids
        attn_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + target_ids
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "_keep": True,
        }

    ds = Dataset.from_list(train_data)
    ds = ds.map(_tokenize, remove_columns=ds.column_names, num_proc=min(os.cpu_count() or 1, 8))
    ds = ds.filter(lambda x: x["_keep"])
    ds = ds.remove_columns(["_keep"])
    return ds

class LeftPadCollator:
    def __init__(self, tok):
        self.pad_id = tok.pad_token_id

    def __call__(self, feats):
        to_t = lambda x: torch.tensor(x, dtype=torch.long)
        ids = [to_t(f["input_ids"]) for f in feats]
        attn = [to_t(f["attention_mask"]) for f in feats]
        labels = [to_t(f["labels"]) for f in feats]
        ids = self._lpad(ids, self.pad_id)
        attn = self._lpad(attn, 0)
        labels = self._lpad(labels, -100)
        return {"input_ids": ids, "attention_mask": attn, "labels": labels}

    @staticmethod
    def _lpad(seqs, pad_val):
        rev = [torch.flip(s, dims=[0]) for s in seqs]
        pad = pad_sequence(rev, batch_first=True, padding_value=pad_val)
        return torch.flip(pad, dims=[1])

tuple2index = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}

def load_metric_tables(p: Path) -> Dict[Tuple[str, int], float]:
    metric = {}
    with p.open() as f:
        for line in f:
            entry = json.loads(line)
            key = (entry["question"], entry["action_index"])
            metric[key] = entry["f1"]
    return metric

class EvaluationCallback(TrainerCallback):
    def __init__(self, tok, sys_desc: str, usr_desc: str, cdp: dict, train_tbl: Path, 
                 valid_tbl: Path, bs: int, run_tag: str = "", min_save_score: float = float("-inf")):
        super().__init__()
        self.tok = tok
        self.sys = sys_desc
        self.usr = usr_desc
        self.cdp = cdp
        self.tr_m = load_metric_tables(train_tbl)
        self.va_m = load_metric_tables(valid_tbl)
        self.q_tr = list(dict.fromkeys(q for q, _ in self.tr_m))
        self.q_va = list(dict.fromkeys(q for q, _ in self.va_m))
        self.bs = bs
        self.best = float("-inf")
        self.best_ckpt = None
        self.run_tag = run_tag
        self.min_save_score = min_save_score


    def _build_prompt(self, q: str) -> str:
        msgs = []
        if self.sys is not None and len(self.sys.strip()) > 0:
            msgs.append({"role": "system", "content": self.sys})
        msgs.append(
            {
                "role": "user",
                "content": self.usr.format(question=q, model_0=self.cdp[0], model_1=self.cdp[1]),
            }
        )
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def _score(self, model, questions, metric):
        model.eval()
        dev = model.device
        prompts = [self._build_prompt(q) for q in questions]
        total, idx = 0.0, 0
        while idx < len(prompts):
            batch_p = prompts[idx : idx + self.bs]
            batch = self.tok(batch_p, return_tensors="pt", padding=True, truncation=True).to(dev)
            with torch.autocast("cuda", torch.bfloat16):
                outs = model.generate(
                    **batch,
                    max_new_tokens=16,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self.tok.pad_token_id,
                    eos_token_id=self.tok.eos_token_id,
                )
            for inp, out, q in zip(batch["input_ids"], outs, questions[idx : idx + self.bs]):
                gen = out[len(inp) :]
                resp = self.tok.decode(gen, skip_special_tokens=True)
                t, s = parse_router_output(resp)
                aidx = tuple2index.get((t, s), 0)
                total += metric.get((q, aidx), 0.0)
            idx += self.bs
        return total
    

    @torch.inference_mode()
    def _score_and_counts(self, model, questions, metric):
        model.eval()
        dev = model.device
        prompts = [self._build_prompt(q) for q in questions]
        total, idx = 0.0, 0
        counts = Counter()
        while idx < len(prompts):
            batch_p = prompts[idx : idx + self.bs]
            batch = self.tok(batch_p, return_tensors="pt", padding=True, truncation=True).to(dev)
            with torch.autocast("cuda", torch.bfloat16):
                outs = model.generate(
                    **batch,
                    max_new_tokens=16,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self.tok.pad_token_id,
                    eos_token_id=self.tok.eos_token_id,
                )
            for inp, out, q in zip(batch["input_ids"], outs, questions[idx : idx + self.bs]):
                gen = out[len(inp) :]
                resp = self.tok.decode(gen, skip_special_tokens=True)
                t, s = parse_router_output(resp)
                aidx = tuple2index.get((t, s), 0)
                counts[aidx] += 1
                total += metric.get((q, aidx), 0.0)
            idx += self.bs
        dist = [counts.get(i, 0) for i in range(4)]
        return total, dist

    def on_epoch_end(self, args, state, control, **kw):
        model = kw["model"]
        tr_r, tr_dist = self._score_and_counts(model, self.q_tr, self.tr_m)
        va_r, va_dist = self._score_and_counts(model, self.q_va, self.va_m)
        ep = int(state.epoch)
        print(f"[{self.run_tag}] [Epoch {ep}] train_pred_dist (0,1,2,3) = {tr_dist} | valid_pred_dist (0,1,2,3) = {va_dist}")
        print(f"[{self.run_tag}] [Epoch {ep}] valid_reward={va_r:.4f} | train_reward={tr_r:.4f} | "
              f"save_threshold={self.min_save_score:.4f} | best_so_far={self.best:.4f}")
        
        should_save = (va_r > self.best) and (va_r > self.min_save_score)
        if should_save:
            self.best = va_r
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt = Path(args.output_dir) / f"best_{va_r:.4f}_{ts}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt)
            self.tok.save_pretrained(ckpt)
            self.best_ckpt = ckpt
            print(f"[{self.run_tag}] New best saved to {ckpt}")
        else:
            reason = []
            if not (va_r > self.best):
                reason.append("not higher than previous best")
            if not (va_r > self.min_save_score):
                reason.append("not higher than uniform-baseline threshold")
            print(f"[{self.run_tag}] Skip saving ({' & '.join(reason)})")

def train_once(
    model_name_or_path: str,
    cdp: Dict[int, str],
    short_name: str,
    description_path: Path,
    system_prompt_path: Path,
    sft_json: Path,
    train_bsl_jsonl: Path,
    valid_bsl_jsonl: Path,
    output_dir: Path,
    epochs: int,
    lr: float,
    warmup_ratio: float,
    bsz: int,
    grad_accum: int,
    eval_bs: int,
):
    print(f"\n===== RUN: {short_name} =====")
    print(f"cdp: {cdp}")
    print(f"sft_json: {sft_json}")
    print(f"train_bsl: {train_bsl_jsonl}")
    print(f"valid_bsl: {valid_bsl_jsonl}")
    print(f"output_dir: {output_dir}")

    summarize_sft_labels_from_file(sft_json, cdp)
    ua_tr_act, ua_tr_total, ua_tr_totals, n_tr = compute_uniform_baseline(train_bsl_jsonl)
    ua_va_act, ua_va_total, ua_va_totals, n_va = compute_uniform_baseline(valid_bsl_jsonl)

    print(f"[{short_name}] Uniform-action baseline (TRAIN, n={n_tr}): "
          f"best_action={ua_tr_act}, total={ua_tr_total:.4f}, totals={ua_tr_totals}")
    print(f"[{short_name}] Uniform-action baseline (VALID, n={n_va}): "
          f"best_action={ua_va_act}, total={ua_va_total:.4f}, totals={ua_va_totals}")
    print(f"[{short_name}] >>> Save threshold set to VALID total = {ua_va_total:.4f}")



    output_dir.mkdir(parents=True, exist_ok=True)

    usr_desc = description_path.read_text(encoding="utf-8")
    sys_desc = system_prompt_path.read_text(encoding="utf-8") if system_prompt_path.exists() else ""

    train_raw = json.loads(sft_json.read_text(encoding="utf-8"))
    print(f"Train samples : {len(train_raw):,}")

    model, tok = load_mNt(model_name_or_path, "auto")
    model, tok = align_model_and_tokenizer(model, tok)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        model.config.pad_token_id = tok.pad_token_id

    train_ds_full = build_dataset(train_raw, sys_desc, usr_desc, cdp, tok)
    train_ds_full = train_ds_full.shuffle(seed=SEED)
    collator = LeftPadCollator(tok)

    tr_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=bsz,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        seed=SEED,
        bf16=True,
        dataloader_pin_memory=True,
        report_to=[],
    )

    eval_cb = EvaluationCallback(
        tok,
        sys_desc,
        usr_desc,
        cdp,
        train_bsl_jsonl,
        valid_bsl_jsonl,
        eval_bs,
        run_tag=short_name,
        min_save_score=ua_va_total
    )

    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=train_ds_full,
        data_collator=collator,
        callbacks=[eval_cb],
    )
    trainer.train()

    print(f"[{short_name}] Training finished")
    print(f"[{short_name}] Best valid_reward = {eval_cb.best:.4f}")
    if eval_cb.best_ckpt:
        print(f"[{short_name}] Best checkpoint: {eval_cb.best_ckpt}")

if __name__ == "__main__":
    args = parse_args()
    runs = resolve_runs(args)

    for conf in runs:
        train_once(
            model_name_or_path=args.model_name_or_path,
            cdp=conf["cdp"],
            short_name=conf["short_name"],
            description_path=args.description,
            system_prompt_path=args.system_prompt,
            sft_json=conf["sft_json"],
            train_bsl_jsonl=conf["train_bsl_jsonl"],
            valid_bsl_jsonl=conf["valid_bsl_jsonl"],
            output_dir=conf["output_dir"],
            epochs=args.epochs,
            lr=args.lr,
            warmup_ratio=args.warmup_ratio,
            bsz=args.bsz,
            grad_accum=args.grad_accum,
            eval_bs=args.eval_bs,
        )
