import argparse
import os

from typing import Optional, Dict
import random

from utils import (
    set_global_seed, 
    load_mNt, 
    align_model_and_tokenizer, 
    read_jsonl, 
    f1_score, 
    em, 
    load_txt, 
    build_prompt, 
    route, 
    route_batch, 
    parse_router_output, 
)

SEED = 43
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = f"{SEED}"
set_global_seed(SEED)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"


from agent import Agent, run_batch, run
from lora_generator import (
    GeneratorConfig,
    create_llama_8b,
    create_qwen_7b,
    create_mistral_7b,
    create_generator_optimizer,
    get_linear_scheduler,
)

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from contextlib import contextmanager

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()


PRESETS: Dict[str, Dict] = {
    "llama_qwen": {
        "cdp": {0: "Llama-3.1-8B-Instruct", 1: "Qwen2.5-7B-Instruct"},
        "short_name": "Llama_8B-Qwen_7B",
        "base_router_path": "PATH/TO/THE/SFT/BEST/CHECKPOINT"
    },
    "mistral_qwen": {
        "cdp": {0: "Mistral-7B-Instruct-v0.3", 1: "Qwen2.5-7B-Instruct"},
        "short_name": "Mistral_7B-Qwen_7B",
        "base_router_path": "PATH/TO/THE/SFT/BEST/CHECKPOINT"
    },
    "llama_mistral": {
        "cdp": {0: "Llama-3.1-8B-Instruct", 1: "Mistral-7B-Instruct-v0.3"},
        "short_name": "Llama_8B-Mistral_7B",
        "base_router_path": "PATH/TO/THE/SFT/BEST/CHECKPOINT"
    },
}

model2path: Dict[str, str] = {
    "Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3"
}
model2gen = {
    "Llama-3.1-8B-Instruct": create_llama_8b,
    "Qwen2.5-7B-Instruct": create_qwen_7b,
    "Mistral-7B-Instruct-v0.3": create_mistral_7b,
}


def infer_backbone_hidden_dim(
    router_model: AutoModelForCausalLM,
    router_tokenizer: Optional[AutoTokenizer] = None
) -> int:
    cfg = getattr(router_model, "config", None)
    for k in ("hidden_size", "n_embd", "d_model", "hidden_dim"):
        if cfg is not None and hasattr(cfg, k):
            val = getattr(cfg, k)
            if isinstance(val, int) and val > 0:
                return val
    if router_tokenizer is None:
        raise ValueError("Cannot infer hidden size: need router_tokenizer for fallback forward.")
    inputs = router_tokenizer("hi", return_tensors="pt")
    device = next(router_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = router_model.model(
            **inputs, output_hidden_states=True, return_dict=True
        )
    return int(outputs.hidden_states[-1].shape[-1])


@contextmanager
def eval_mode_for_base_and_gens(
    base_router_model, generators, *,
    set_use_cache=True,
    force_disable_backbone_grad=True
):
    prev = {
        "base_training": base_router_model.training,
        "use_cache": getattr(base_router_model.config, "use_cache", None),
        "gens_training": {id(g): g.training for g in generators},
        "gens_allow_backbone": {id(g): getattr(g, "allow_backbone_grad", None) for g in generators},
    }

    try:
        base_router_model.eval()
        if set_use_cache and hasattr(base_router_model.config, "use_cache"):
            base_router_model.config.use_cache = True

        for g in generators:
            g.eval()
            if force_disable_backbone_grad and hasattr(g, "allow_backbone_grad"):
                g.allow_backbone_grad = False

        yield

    finally:
        base_router_model.train(prev["base_training"])
        if prev["use_cache"] is not None and hasattr(base_router_model.config, "use_cache"):
            base_router_model.config.use_cache = prev["use_cache"]

        for g in generators:
            g.train(prev["gens_training"][id(g)])
            old = prev["gens_allow_backbone"][id(g)]
            if old is not None:
                g.allow_backbone_grad = old



def candidate2gen(candidate_pair: dict, *, 
    base_router_model: AutoModelForCausalLM, 
    base_router_tokenizer: AutoModelForCausalLM,
    r_device: str, 
    total_steps: int, 
    lr_each: dict | None = None, 
    default_lr: float,
    mlp_num_layer: int,
    lora_r: int,
    weight_decay: float = 0.0, 
    start_factor: float = 1.0, 
    end_factor: float = 0.1,
):
    inferred_dim = infer_backbone_hidden_dim(base_router_model, base_router_tokenizer)
    cfg = GeneratorConfig(
        seed=SEED,
        lora_r=lora_r,
        mlp_in_dim=inferred_dim,
        mlp_hidden_dim=256,
        mlp_num_layers=mlp_num_layer,
        allow_backbone_grad=True,
    )
    
    gens, opts, scheds = {}, {}, {} 
    lr_each = lr_each or {}
    for _, model in candidate_pair.items():
        assert model in model2gen, f"Unkonwn model '{model}'. Please register create_* factory function in model2gen."
        create_fn = model2gen[model]
        gen = create_fn(
            base_model=base_router_model,
            base_tokenizer=base_router_tokenizer,
            device=r_device,
            config=cfg,
        )
        gens[model] = gen
        lr = lr_each.get(model, default_lr)
        opt = create_generator_optimizer(gen, lr=lr, weight_decay=weight_decay)
        sch = get_linear_scheduler(opt, total_steps, start_factor=start_factor, end_factor=end_factor)
        opts[model] = opt
        scheds[model] = sch

    return gens, opts, scheds


@torch.inference_mode()
def evaluate_HALO_bucketed(
    teacher: Agent, student: Agent, gens: dict,
    
    pro_temp: str, sys_desc: str,
    
    valid_data: list, s_device: str, t_device: str,
    
    base_router_model:AutoModelForCausalLM, 
    base_router_tokenizer:AutoTokenizer,

    candidate_pair: dict,
    
    alpha: float, 
    batch_route_bs: int,
    batch_ab_bs: int,
    dialogue_bs: int,
    layers: int,

    budget: int = 3,
    mode: str = "valid",
):
    gens_list = list(gens.values())
    with eval_mode_for_base_and_gens(
        base_router_model, gens_list,
        set_use_cache=True,
        force_disable_backbone_grad=True,
    ):
        with torch.inference_mode():
            print(f"[BucketEval] {mode} samples = {len(valid_data)}")
            formatted_prompts, questions, answers, datas = [], [], [], []
            for entry in valid_data:
                q = entry['question']; a = entry['answer']
                d = [''.join(s) for _, s in entry['context']]
                user_prompt = pro_temp.format(
                    question=q, model_0=candidate_pair[0], model_1=candidate_pair[1],
                )
                fp = build_prompt(
                    tokenizer=base_router_tokenizer,
                    sys_desc=sys_desc,
                    user_prompt=user_prompt,
                    add_gen=True,
                )
                formatted_prompts.append(fp)
                questions.append(q); answers.append(a); datas.append(d)
            print("[BucketEval] Built formatted prompts successfully.")
            routes = []
            for i in range(0, len(formatted_prompts), batch_route_bs):
                chunk = formatted_prompts[i:i+batch_route_bs]
                routes.extend(route_batch(chunk, base_router_model, base_router_tokenizer))
            pair_ids = [parse_router_output(x) for x in routes]
            t_models = [candidate_pair[t] for (t, s) in pair_ids]
            s_models = [candidate_pair[s] for (t, s) in pair_ids]
            print("[BucketEval] Completed routing for all samples.")

            buckets = defaultdict(list)
            for i, (tm, sm) in enumerate(zip(t_models, s_models)):
                buckets[(tm, sm)].append(i)
            print(f"[BucketEval] Created {len(buckets)} buckets: {list(buckets.keys())}")

            ab_cache = {m: {} for m in gens.keys()}
            for model_name in gens.keys():
                idxs = [i for i in range(len(valid_data))
                        if t_models[i] == model_name or s_models[i] == model_name]
                if not idxs:
                    continue
                for j in range(0, len(idxs), batch_ab_bs):
                    batch_ids = idxs[j:j+batch_ab_bs]
                    texts = [formatted_prompts[i] for i in batch_ids]
                    A, B = gens[model_name](texts)
                    for k, idx in enumerate(batch_ids):
                        ab_cache[model_name][idx] = (A[k].cpu(), B[k].cpu())
            print("[BucketEval] Generated A/B for all required models.")

            all_f1 = all_precision = all_recall = em_scores = 0.0

            for (tm, sm), idxs in buckets.items():
                print(f"[BucketEval] bucket=({tm} | {sm}), size={len(idxs)}")
                teacher.model, teacher.tokenizer = load_mNt(
                    model_name=model2path[tm], device=t_device, dtype=torch.bfloat16
                )
                student.model, student.tokenizer = load_mNt(
                    model_name=model2path[sm], device=s_device, dtype=torch.bfloat16
                )
                teacher.model.eval(); student.model.eval()
                teacher.reset_runtime_state(grad=False); student.reset_runtime_state(grad=False)

                teacher.init_model_with_AB_batch_runtime(alpha=alpha, num_layers_to_patch=layers)
                student.init_model_with_AB_batch_runtime(alpha=alpha, num_layers_to_patch=layers)

                try:
                    for st in tqdm(range(0, len(idxs), dialogue_bs), desc=f"bucket=({tm}|{sm})", leave=False):
                        sub = idxs[st:st+dialogue_bs]

                        q_sub = [questions[i] for i in sub]
                        d_sub = [datas[i] for i in sub]
                        a_sub = [answers[i] for i in sub]
                        t_As = [ab_cache[tm][i][0] for i in sub]
                        t_Bs = [ab_cache[tm][i][1] for i in sub]
                        s_As = [ab_cache[sm][i][0] for i in sub]
                        s_Bs = [ab_cache[sm][i][1] for i in sub]

                        pred_sub = run_batch(
                            q_sub, teacher, student, d_sub,
                            t_As=t_As, t_Bs=t_Bs, s_As=s_As, s_Bs=s_Bs,
                            alpha=alpha, budget=budget
                        )

                        for pred, gold in zip(pred_sub, a_sub):
                            f1_val, precision, recall = f1_score(pred, gold)
                            em_score = em(pred, gold)
                            all_f1 += f1_val
                            all_precision += precision
                            all_recall += recall
                            em_scores += em_score
                finally:
                    teacher.restore_batch_runtime_patch()
                    student.restore_batch_runtime_patch()
                    del teacher.model, teacher.tokenizer, student.model, student.tokenizer
                    teacher.model = teacher.tokenizer = None
                    student.model = student.tokenizer = None
                    torch.cuda.empty_cache()

            return all_f1, all_precision, all_recall, em_scores


def rl_train(
    *,
    train_data: list, valid_data: list, test_data: list,
    candidate_pair: dict, short_name: str,

    base_router_path: str,
    teacher: Agent, student: Agent,
    s_device: str, t_device: str, r_device: str,
    
    epochs: int,
    alpha: float,
    high_router_lr: float,
    low_router_lr: float,
    eval_interval: int,
    early_stop_patience_evals: int,
    eval_batch_route_bs: int,
    eval_batch_ab_bs: int,
    eval_dialogue_bs: int,

    layers: int,
    mlp_num_layer: int,
    lora_r: int,
):
    assert base_router_path, "base_router_path must be provided either in PRESETS or as an argument."
    best_valid_f1 = -1.0
    base_router_model, base_router_tokenizer = load_mNt(base_router_path, device=r_device)
    base_router_model, base_router_tokenizer = align_model_and_tokenizer(base_router_model, base_router_tokenizer)
    for param in base_router_model.parameters():
        param.requires_grad = True
    base_router_model.train()

    router_optimizer = torch.optim.AdamW(
        [{"params": filter(lambda p: p.requires_grad, base_router_model.parameters()), "lr": high_router_lr}]
    )
    total_steps = epochs * len(train_data)
    router_scheduler = torch.optim.lr_scheduler.LinearLR(router_optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)

    gens, opts, scheds = candidate2gen(
        candidate_pair=candidate_pair,
        base_router_model=base_router_model,
        base_router_tokenizer=base_router_tokenizer,
        r_device=r_device,
        total_steps=total_steps,
        default_lr=low_router_lr,
        mlp_num_layer=mlp_num_layer,
        lora_r=lora_r,
    )

    for g in gens.values():
        g.train()
    
    pro_temp = load_txt("../prompt/router_template.txt")
    sys_desc = load_txt("../prompt/system_router.txt")

    total_processed = 0
    no_improve_evals = 0
    should_stop = False

    for epoch in range(epochs):
        rng = random.Random(SEED + epoch)
        rng.shuffle(train_data)
        print(f"Starting epoch {epoch+1}/{epochs}...")
        for qa_data in tqdm(train_data, desc="Training"):
            question = qa_data['question']
            answer = qa_data['answer']
            data = [''.join(sents) for _, sents in qa_data["context"]]
            user_prompt = pro_temp.format(
                question=question,
                model_0=candidate_pair[0],
                model_1=candidate_pair[1],
            )
            formmatted_prompt = build_prompt(
                tokenizer=base_router_tokenizer, 
                sys_desc=sys_desc, 
                user_prompt=user_prompt, 
                add_gen=True
            )
            teacher.initial_self_para(grad=True)
            student.initial_self_para(grad=True)

            teacher_id, student_id = parse_router_output(route(
                formatted_prompt=formmatted_prompt, 
                router_model=base_router_model, 
                router_tokenizer=base_router_tokenizer)
            )
            teacher_model_name = candidate_pair[teacher_id]
            student_model_name = candidate_pair[student_id]
            teacher_opt, student_opt = opts[teacher_model_name], opts[student_model_name]
            teacher_sched, student_sched = scheds[teacher_model_name], scheds[student_model_name]

            teacher.A, teacher.B = gens[teacher_model_name](formmatted_prompt)
            student.A, student.B = gens[student_model_name](formmatted_prompt)
            teacher.model, teacher.tokenizer = load_mNt(
                model_name=model2path[teacher_model_name],
                device=t_device,
                dtype=torch.bfloat16
            )
            student.model, student.tokenizer = load_mNt(
                model_name=model2path[student_model_name],
                device=s_device,
                dtype=torch.bfloat16
            )
            teacher.init_model_with_AB(num_layers_to_patch=layers, alpha=alpha)
            student.init_model_with_AB(num_layers_to_patch=layers, alpha=alpha)

            agents_answer = run(question, teacher, student, data)
            f1, _, _ = f1_score(agents_answer, answer)
            reward = -1 if f1 == 0 else f1
            teacher_probs = [p.to(r_device) for p in teacher.probs]
            student_probs = [p.to(r_device) for p in student.probs]
            total_loss = -reward * (torch.stack(teacher_probs + student_probs).sum())

            if teacher_id == student_id:
                student_opt.zero_grad()
                router_optimizer.zero_grad()
                total_loss.backward()
                student_opt.step()
                router_optimizer.step()
                student_sched.step()
                router_scheduler.step()
            else:
                student_opt.zero_grad()
                teacher_opt.zero_grad()
                router_optimizer.zero_grad()
                total_loss.backward()
                student_opt.step()
                teacher_opt.step()
                router_optimizer.step()
                student_sched.step()
                teacher_sched.step()
                router_scheduler.step()

            del teacher.model, teacher.tokenizer, student.model, student.tokenizer
            del student.A, student.B, teacher.A, teacher.B
            del student_probs, teacher_probs
            del total_loss
            teacher.probs.clear()
            student.probs.clear()
            torch.cuda.empty_cache()
            total_processed += 1

            if total_processed % eval_interval == 0:
                print(f"Evaluating after [Epoch:{epoch}], processing {total_processed} questions...")
                valid_f1, _, _, _ = evaluate_HALO_bucketed(
                    teacher=teacher, student=student, gens=gens,
                    pro_temp=pro_temp, sys_desc=sys_desc,
                    valid_data=valid_data,
                    s_device=s_device, t_device=t_device,
                    base_router_model=base_router_model,
                    base_router_tokenizer=base_router_tokenizer,
                    candidate_pair=candidate_pair,
                    alpha=alpha,
                    batch_route_bs=eval_batch_route_bs,
                    batch_ab_bs=eval_batch_ab_bs,
                    dialogue_bs=eval_dialogue_bs,
                    layers=layers,
                    mode="valid"
                )
                print(f"Epoch {epoch}, Valid f1: {valid_f1:.4f}")
                improved = valid_f1 > best_valid_f1 + 1e-12
                if improved:
                    best_valid_f1 = valid_f1
                    no_improve_evals = 0

                    test_f1, _, _, _ = evaluate_HALO_bucketed(
                        teacher=teacher, student=student, gens=gens,
                        pro_temp=pro_temp, sys_desc=sys_desc,
                        valid_data=test_data,
                        s_device=s_device, t_device=t_device,
                        base_router_model=base_router_model,
                        base_router_tokenizer=base_router_tokenizer,
                        candidate_pair=candidate_pair,
                        alpha=alpha,
                        batch_route_bs=eval_batch_route_bs,
                        batch_ab_bs=eval_batch_ab_bs,
                        dialogue_bs=eval_dialogue_bs,
                        layers=layers,
                        mode="test"
                    )
                    print(f"[Saving!] Epoch {epoch}, Valid f1: {valid_f1:.4f}, Test f1: {test_f1:.4f}")
                    save_dir = f"saved_router/{short_name}/{total_processed}/high_router_model_V_{valid_f1:.2f}_T_{test_f1:.2f}"
                    os.makedirs(f"saved_router/{short_name}/{total_processed}", exist_ok=True)
                    base_router_model.save_pretrained(save_dir)
                    base_router_tokenizer.save_pretrained(save_dir)
                    gens[candidate_pair[0]].save_parameters(f"saved_router/{short_name}/{total_processed}/{candidate_pair[0]}.pth")
                    gens[candidate_pair[1]].save_parameters(f"saved_router/{short_name}/{total_processed}/{candidate_pair[1]}.pth")
                else:
                    no_improve_evals += 1
                    print(f"No improvement in valid f1 for {no_improve_evals} evaluations.")
                    if no_improve_evals >= early_stop_patience_evals:
                        print("Early stopping triggered.")
                        should_stop = True
                        break
        if should_stop:
            break

    print(f"[Done] best_valid_f1={best_valid_f1:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", type=str, default="llama_qwen",
                   choices=list(PRESETS.keys()), help="Choose model pair from PRESETS, e.g., llama_qwen. You can also add your own presets in the PRESETS dict.")
    p.add_argument("--base_router_path", type=str, default="",
                   help="Base router model path. If provided, it overrides the preset path.")

    p.add_argument("--s_device", type=str, default="cuda:0", help="Student device")
    p.add_argument("--t_device", type=str, default="cuda:1", help="Teacher device")
    p.add_argument("--r_device", type=str, default="cuda:2", help="Router device")

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--high_router_lr", type=float, default=5e-6)
    p.add_argument("--low_router_lr", type=float, default=1e-5)
    p.add_argument("--eval_interval", type=int, default=30)
    p.add_argument("--early_patience_evals", type=int, default=4,
                   help="Stop training if no improvement after this many evaluations.")

    p.add_argument("--layers", type=int, default=1,
                   help="number of layers to patch with LoRA to agents")
    p.add_argument("--mlp_num_layer", type=int, default=2,
                   help="layers of MLP hidden layers in LoRA generator")
    p.add_argument("--lora_r", type=int, default=8,
                   help="LoRA r parameter in LoRA generator")
    
    p.add_argument("--batch_route_bs", type=int, default=32,
                   help="evaluate_HALO_bucketed routing batch size")
    p.add_argument("--batch_ab_bs", type=int, default=32,
                   help="evaluate_HALO_bucketed A/B generation batch size")
    p.add_argument("--dialogue_bs", type=int, default=8,
                   help="evaluate_HALO_bucketed dialogue batch size")
    
    p.add_argument("--train_jsonl", type=str, default="rl_dataset/train.jsonl")
    p.add_argument("--valid_jsonl", type=str, default="rl_dataset/valid.jsonl")
    p.add_argument("--test_jsonl", type=str, default="../dataset/test.jsonl")
    return p.parse_args()


def main():
    args = parse_args()

    preset = PRESETS[args.preset]
    candidate_pair = preset["cdp"]
    short_name = preset["short_name"]

    base_router_path = args.base_router_path or preset.get("base_router_path", "")
    if not base_router_path:
        raise ValueError("base_router_path not provided in args or preset. Please specify it.")

    train_data = read_jsonl(args.train_jsonl)
    valid_data = read_jsonl(args.valid_jsonl)
    test_data  = read_jsonl(args.test_jsonl)


    retrieve_url = 'http://127.0.0.1:2022/retrieve'
    teacher = Agent(
        role="teacher",
        prompt_1=load_txt('../prompt/teacher/prompt1.txt'),
        prompt_2=load_txt('../prompt/teacher/prompt2.txt'),
        retrieve_url=retrieve_url,
        device=args.t_device
    )

    student = Agent(
        role="student",
        prompt_1=load_txt('../prompt/student/prompt1.txt'),
        prompt_2=load_txt('../prompt/student/prompt2.txt'),
        retrieve_url=retrieve_url,
        device=args.s_device
    )
    
    rl_train(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        candidate_pair=candidate_pair,
        short_name=short_name,
        base_router_path=base_router_path,
        teacher=teacher,
        student=student,
        s_device=args.s_device,
        t_device=args.t_device,
        r_device=args.r_device,
        epochs=args.epochs,
        alpha=args.alpha,
        high_router_lr=args.high_router_lr,
        low_router_lr=args.low_router_lr,
        eval_interval=args.eval_interval,
        early_stop_patience_evals=args.early_patience_evals,
        eval_batch_route_bs=args.batch_route_bs,
        eval_batch_ab_bs=args.batch_ab_bs,
        eval_dialogue_bs=args.dialogue_bs,
        layers=args.layers,
        mlp_num_layer=args.mlp_num_layer,
        lora_r=args.lora_r,
    )

if __name__ == "__main__":
    main()
