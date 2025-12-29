import argparse
import json
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

INDEX2PAIR = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}

PRESETS = {
    "llama_qwen": {"Llama-3.1-8B-Instruct": 0, "Qwen2.5-7B-Instruct": 1},
    "mistral_qwen": {"Mistral-7B-Instruct-v0.3": 0, "Qwen2.5-7B-Instruct": 1},
    "llama_mistral": {"Llama-3.1-8B-Instruct": 0, "Mistral-7B-Instruct-v0.3": 1},
}

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def resolve_model_mapping(args) -> Optional[Dict[str, int]]:
    if args.model_name_to_id_json:
        d = json.loads(args.model_name_to_id_json)
        return {str(k): int(v) for k, v in d.items()}
    if args.pair:
        return PRESETS[args.pair]
    return None

def to_pair(entry: Dict[str, Any],
            model_name_to_id: Optional[Dict[str, int]] = None) -> Optional[Tuple[int, int]]:
    if "action_index" in entry:
        idx = entry["action_index"]
        if idx in INDEX2PAIR:
            return INDEX2PAIR[idx]
        return None
    if "teacher" in entry and "student" in entry:
        t = entry["teacher"]; s = entry["student"]
        def normalize(x):
            if isinstance(x, int): return x
            if isinstance(x, str):
                if model_name_to_id and x in model_name_to_id: return model_name_to_id[x]
                if x.isdigit(): return int(x)
            return None
        t = normalize(t); s = normalize(s)
        if t in (0, 1) and s in (0, 1):
            return (t, s)
    return None

def find_all_actions(rows: List[Dict[str, Any]],
                     model_name_to_id: Optional[Dict[str, int]] = None
                     ) -> Dict[str, List[Tuple[Tuple[int, int], float]]]:
    per_q = defaultdict(list)
    for r in rows:
        q = r["question"]
        f1 = float(r["f1"])
        pair = to_pair(r, model_name_to_id)
        if pair is None:
            continue
        per_q[q].append((pair, f1))
    return per_q

def choose_from_ties(cands: List[Tuple[Tuple[int, int], float]],
                     tie_policy: str,
                     rng: random.Random) -> List[Tuple[Tuple[int, int], float]]:
    if not cands:
        return []
    max_score = max(s for (_, s) in cands)
    eps = 1e-12
    best = [(p, s) for (p, s) in cands if abs(s - max_score) <= eps]
    if tie_policy == "all":
        return best
    elif tie_policy == "random_one":
        return [rng.choice(best)]
    elif tie_policy == "deterministic_one":
        best_sorted = sorted(best, key=lambda x: (x[0][0], x[0][1]))
        return [best_sorted[0]]
    else:
        raise ValueError(f"Unknown tie_policy: {tie_policy}")

def build_sft_dataset(per_q: Dict[str, List[Tuple[Tuple[int, int], float]]],
                      tie_policy: str,
                      min_max_score: float,
                      rng: random.Random) -> List[Dict[str, Any]]:
    sft = []
    for q, lst in per_q.items():
        if not lst:
            continue
        max_score = max(s for (_, s) in lst)
        if max_score < min_max_score:
            continue
        chosen = choose_from_ties(lst, tie_policy, rng)
        for (pair, s) in chosen:
            t, st = pair
            sft.append({
                "question": q,
                "reward": float(s),
                "output": json.dumps({"teacher": t, "student": st}, separators=(",", ": "))
            })
    return sft

def class_counts_from_sft(sft: List[Dict[str, Any]]) -> Counter:
    cnt = Counter()
    for row in sft:
        obj = json.loads(row["output"])
        pair = (int(obj["teacher"]), int(obj["student"]))
        for idx, p in INDEX2PAIR.items():
            if p == pair:
                cnt[idx] += 1
                break
    return cnt


def _round_robin_oversample(rows: List[Dict[str, Any]], target: int, rng: random.Random) -> List[Dict[str, Any]]:
    """
    Oversample to 'target' without cropping majority. Duplicate each sample roughly equally.
    """
    n = len(rows)
    if n == 0 or target <= n:
        return list(rows)  # do not crop here (majority untouched)
    reps = target // n
    rem = target - reps * n
    out = rows * reps + rng.sample(rows, rem)
    rng.shuffle(out)
    return out

def smart_oversample_only(sft: List[Dict[str, Any]],
                          rng: random.Random,
                          target_per_class: Optional[int] = None,
                          frac_of_max: Optional[float] = None,
                          minority_dup_factor: Optional[float] = None,
                          auto_skip_if_balanced_ratio: Optional[float] = None
                          ) -> List[Dict[str, Any]]:
    buckets = {0: [], 1: [], 2: [], 3: []}
    for row in sft:
        obj = json.loads(row["output"])
        pair = (int(obj["teacher"]), int(obj["student"]))
        idx = None
        for k, p in INDEX2PAIR.items():
            if p == pair:
                idx = k
                break
        if idx is not None:
            buckets[idx].append(row)

    counts = {k: len(v) for k, v in buckets.items()}
    if len([c for c in counts.values() if c > 0]) == 0:
        return sft

    max_c = max(counts.values()) if counts else 0
    min_c = min(counts.values()) if counts else 0

    # auto skip if roughly balanced
    if auto_skip_if_balanced_ratio is not None and min_c > 0:
        ratio = max_c / min_c
        if ratio <= float(auto_skip_if_balanced_ratio):
            # Already balanced enough.
            return sft

    # decide target for each class
    targets = {}
    if target_per_class is not None:
        tgt = int(target_per_class)
        for k, n in counts.items():
            targets[k] = max(n, tgt)  # only raise up to tgt; if n >= tgt, keep n
    elif frac_of_max is not None:
        t = int(max(1, int(max_c * float(frac_of_max))))
        for k, n in counts.items():
            targets[k] = max(n, t)
    elif minority_dup_factor is not None:
        for k, n in counts.items():
            # scale each class up to min(ceil(n*factor), max_c)
            scaled = int((n * float(minority_dup_factor) + 0.9999))  # ceil
            targets[k] = min(max_c, max(n, scaled))
    else:
        # default: do nothing if no smart parameters provided
        return sft

    new_rows: List[Dict[str, Any]] = []
    for k in [0, 1, 2, 3]:
        rows_k = buckets[k]
        n = len(rows_k)
        tgt_k = targets.get(k, n)
        if n == 0:
            continue
        if tgt_k <= n:
            # majority or already above target: keep all, no cropping
            new_rows.extend(rows_k)
        else:
            new_rows.extend(_round_robin_oversample(rows_k, tgt_k, rng))
    rng.shuffle(new_rows)
    return new_rows
# ---------- end new helpers ----------

def rebalance(sft: List[Dict[str, Any]],
              mode: str,
              target_per_class: Optional[int],
              rng: random.Random) -> List[Dict[str, Any]]:
    if mode == "none":
        return sft

    buckets = {0: [], 1: [], 2: [], 3: []}
    for row in sft:
        obj = json.loads(row["output"])
        pair = (int(obj["teacher"]), int(obj["student"]))
        idx = None
        for k, p in INDEX2PAIR.items():
            if p == pair:
                idx = k
                break
        if idx is None:
            continue
        buckets[idx].append(row)

    counts = {k: len(v) for k, v in buckets.items()}
    if target_per_class is None:
        if mode == "undersample":
            tgt = min(counts.values()) if counts else 0
        else:
            tgt = max(counts.values()) if counts else 0
    else:
        tgt = int(target_per_class)

    new_rows = []
    for k in [0, 1, 2, 3]:
        rows_k = buckets[k]
        n = len(rows_k)
        if n == 0:
            continue
        if mode == "undersample":
            if n <= tgt:
                new_rows.extend(rows_k)
            else:
                new_rows.extend(rng.sample(rows_k, tgt))
        else:  # classic oversample: both crop majority and duplicate minority to tgt
            if n >= tgt:
                new_rows.extend(rng.sample(rows_k, tgt))
            else:
                need = tgt - n
                new_rows.extend(rows_k)
                new_rows.extend(rng.choices(rows_k, k=need))
    rng.shuffle(new_rows)
    return new_rows

def print_stats(stage: str, per_q: Dict[str, List[Tuple[Tuple[int, int], float]]], sft: List[Dict[str, Any]]):
    num_q = len(per_q)
    multi_actions = sum(1 for q, lst in per_q.items() if len({p for (p, _) in lst}) > 1)
    zero_max = sum(1 for q, lst in per_q.items() if max(s for (_, s) in lst) <= 1e-12)
    cnt = class_counts_from_sft(sft)
    print(f"[{stage}] questions: {num_q}, with_multiple_actions: {multi_actions}, max_score_eq_0: {zero_max}")
    print(f"[{stage}] class_counts (idx:count): {dict(cnt)}; total_rows: {len(sft)}")

def save_json(data: List[Dict[str, Any]], path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=Path("baseline.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("sft_train_all.json"))
    ap.add_argument("--seed", type=int, default=43)

    ap.add_argument("--pair", type=str, default=None,
                    choices=list(PRESETS.keys()),
                    help="Preset mapping name for model-name -> id, used if teacher/student are strings.")

    ap.add_argument("--tie_policy", type=str, default="deterministic_one",
                    choices=["all", "random_one", "deterministic_one"],
                    help="How to handle ties among best actions for a question.")
    ap.add_argument("--min_max_score", type=float, default=1e-6,
                    help="Drop questions whose best score < this value.")
    ap.add_argument("--balance", type=str, default="none",
                    choices=["none", "undersample", "oversample", "smart_oversample"],
                    help="Balancing mode. 'smart_oversample' only increases minority classes; majority untouched.")
    ap.add_argument("--target_per_class", type=int, default=None,
                    help="[classic undersample/oversample] target rows per class.")
    ap.add_argument("--smart_target", type=int, default=None,
                    help="[smart_oversample] raise any class below this to this target; leave larger classes unchanged.")
    ap.add_argument("--smart_frac_of_max", type=float, default=None,
                    help="[smart_oversample] raise classes to floor(max_count * frac).")
    ap.add_argument("--smart_minority_dup_factor", type=float, default=None,
                    help="[smart_oversample] per-class target = min(ceil(n * factor), max_count).")
    ap.add_argument("--auto_skip_if_balanced_ratio", type=float, default=None,
                    help="[smart_oversample] if max/min <= this ratio, skip balancing altogether.")
    ap.add_argument("--model_name_to_id_json", type=str, default=None,
                    help='Optional JSON string like {"Llama-3.1-8B-Instruct":0,"Qwen2.5-7B-Instruct":1}')

    args = ap.parse_args()
    rng = random.Random(args.seed)

    model_name_to_id = resolve_model_mapping(args)
    rows = load_jsonl(args.baseline)
    per_q = find_all_actions(rows, model_name_to_id)
    print(args.tie_policy)
    pre_sft = build_sft_dataset(per_q, tie_policy="all", min_max_score=-1e9, rng=rng)
    print_stats("raw", per_q, pre_sft)

    sft = build_sft_dataset(per_q, tie_policy=args.tie_policy,
                            min_max_score=args.min_max_score, rng=rng)
    print_stats("selected_before_balance", per_q, sft)

    if args.balance == "smart_oversample":
        sft = smart_oversample_only(
            sft,
            rng,
            target_per_class=args.smart_target,
            frac_of_max=args.smart_frac_of_max,
            minority_dup_factor=args.smart_minority_dup_factor,
            auto_skip_if_balanced_ratio=args.auto_skip_if_balanced_ratio,
        )
        print_stats("selected_after_smart_oversample", per_q, sft)
    elif args.balance != "none":
        sft = rebalance(sft, mode=args.balance, target_per_class=args.target_per_class, rng=rng)
        print_stats("selected_after_balance", per_q, sft)

    save_json(sft, args.output)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
