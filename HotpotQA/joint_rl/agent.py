import requests
import json
import torch
import torch.nn.functional as F
import re
from typing import List, Tuple
import contextlib


class Retriever:
    def __init__(self, url):
        self.url = url

    def send_request(self, payload, headers=None):
        if headers is None:
            headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(self.url, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                error_message = f"Error occurred with status code: {response.status_code}, response text: {response.text}"
                print(error_message)
                return response.text
        except Exception as e:
            print(f"An error occurred: {e}")
            return str(e)

    def retrieve(self, entity, data):
        request_config = {
            'entity': entity,
            'data': data
        }
        response = self.send_request(request_config)
        return response


class Agent():
    def __init__(
        self, 
        role: str, 
        prompt_1: str, 
        prompt_2: str, 
        retrieve_url: str, 
        device: str
    ) -> None:
        self.model = None
        self.tokenizer = None
        self.A = None
        self.B = None
        self.grad = True
        self.probs = []
        self.device = device
        self.role = role
        self.prompt_1 = prompt_1  # thought and action
        self.prompt_2 = prompt_2  # context summary
        self.temperature = 0
        self.tool = Retriever(retrieve_url)

    def init_model_with_AB_batch_runtime(self, alpha=0.001, num_layers_to_patch="all"):
        self.device = self.model.device
        for p in self.model.parameters():
            p.requires_grad = False

        layers = self._get_decoder_layers(self.model)
        total = len(layers)
        if isinstance(num_layers_to_patch, str):
            n = total if num_layers_to_patch.lower().strip() in ("all", "*") else None
            if n is None:
                raise ValueError(f"Unknown num_layers_to_patch={num_layers_to_patch}")
        else:
            n = min(int(num_layers_to_patch), total)
        start_idx = total - n

        self._patched_old_forwards = []
        for li, layer in enumerate(layers[start_idx:], start=start_idx):
            o_proj = layer.self_attn.o_proj
            base_w = o_proj.weight
            base_b = o_proj.bias
            old_forward = o_proj.forward

            def new_forward(x, base_w=base_w, base_b=base_b, agent_ref=self):
                out = F.linear(x, base_w.to(x.dtype), None if base_b is None else base_b.to(x.dtype))
                ab = getattr(agent_ref, "_ab_batch", None)
                if ab is None:
                    return out
                A, B, a = ab
                A = A.to(x.device, dtype=x.dtype)
                B = B.to(x.device, dtype=x.dtype)
                xBt = torch.matmul(x, B.transpose(1, 2))
                lora_out = torch.matmul(xBt, A.transpose(1, 2))
                return out + (a * lora_out)

            o_proj.forward = new_forward
            self._patched_old_forwards.append((o_proj, old_forward))


    def restore_batch_runtime_patch(self):
        if getattr(self, "_patched_old_forwards", None):
            for (o_proj, old_f) in self._patched_old_forwards:
                o_proj.forward = old_f
        self._patched_old_forwards = []
        self._ab_batch = None

    @contextlib.contextmanager
    def _set_ab_batch(self, A_batch, B_batch, alpha: float):
        if isinstance(A_batch, list):
            A_batch = torch.stack(A_batch, dim=0)  # [B,H,r]
        if isinstance(B_batch, list):
            B_batch = torch.stack(B_batch, dim=0)  # [B,r,H]
        self._ab_batch = (A_batch, B_batch, float(alpha))
        try:
            yield
        finally:
            self._ab_batch = None

    def initial_self_para(self, grad:bool):
        self.A = None
        self.B = None
        self.model = None
        self.tokenizer = None
        self.probs = []
        self.grad = grad

    def reset_runtime_state(self, grad: bool):
        self.A = None
        self.B = None
        self.probs = []
        self.grad = grad

        
    def init_model_with_AB(self, alpha=0.001, num_layers_to_patch="all"):
        self.device = self.model.device
        self.A = self.A.to(self.device)
        self.B = self.B.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        
        def lora_update_fn():
            return alpha * (self.A @ self.B).squeeze(0)
        
        self.monkey_patch_all_layers(self.model, lora_update_fn, num_layers_to_patch=num_layers_to_patch)

    def _tNaparse(self, output_str: str) -> Tuple[str, str]:
        t_match = re.search(
            r"\bThought:\s*(.*?)(?=\n\s*Action:|\Z)",
            output_str,
            flags=re.DOTALL | re.IGNORECASE,
        )
        a_match = re.search(
            r"\bAction:\s*(.*)",
            output_str,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not t_match or not a_match:
            raise ValueError("Failed to parse Thought or Action section.")
        thought = t_match.group(1).strip()
        action = a_match.group(1).strip()
        return thought, action



    def _sparse(self, output_str: str) -> str:
        s_match = re.search(
            r"^\s*Summary\s*:\s*(.*?)(?=^\s*\w+\s*:|\Z)",
            output_str,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        if not s_match:
            raise ValueError("Failed to parse Summary.")
        return s_match.group(1).strip()

    def _context(self, question, context, observation,
                 thought, action) -> str:
        prompt = self.prompt_2.format(
            question=question,
            context=context,
            observation=observation,
            thought=thought,
            action=action
        )

        new_context = self.LLM4generate(prompt).strip()
        try:
            new_context = self._sparse(new_context)
            if self.grad == False:
                print(f"[{self.role.upper()}]", new_context)
            return new_context
        except:
            print("[Warning] Context may be wrong!")
            print(new_context)
            print("New context generation failed in the last round. Here is the previous context: " + context)
            return "New context generation failed in the last round. Here is the previous context: " + context

    
    def _observe(self, time_step:int, student_msg_tube:list, teacher_msg_tube:list):
        new_observation = ''
        if self.role == 'student':
            if time_step == 0:
                new_observation = 'Start to answer the question.'
            else:
                student_msg = student_msg_tube[-1]['action'] + " " + student_msg_tube[-1]['message']
                teacher_msg = teacher_msg_tube[-1]['action'] + " " + teacher_msg_tube[-1]['message']
                new_observation = f"Search Result: {student_msg} Teacher's Advice: {teacher_msg}"
        elif self.role == 'teacher':
            student_msg = student_msg_tube[-1]['action'] + " " + student_msg_tube[-1]['message']
            if time_step == 0:
                new_observation = f"Student's Action: {student_msg}"
            else:
                teacher_msg = teacher_msg_tube[-1]['action'] + " " + teacher_msg_tube[-1]['message']
                new_observation = f"Previous Advice: {teacher_msg} Student's Action: {student_msg}"
        if self.grad == False:
            print(f"[{self.role.upper()}]", new_observation)
        return new_observation
    
    def _thinkNaction(self, stop:bool, context:str, question:str, observation:str):
        if self.role == 'student':
            if stop:
                budget = "You have no search budget. Now you must finish the answer according to current information."
            else:
                budget = "You can search at this step."
            prompt = self.prompt_1.format(context=context,
                                        question=question,
                                        observation=observation,
                                        budget=budget)

            thoughtNaction = self.LLM4generate(prompt=prompt).strip()
            
            try:
                thought, action = self._tNaparse(thoughtNaction)
                end_index = action.find("]")
                if end_index != -1:
                    action = action[:end_index + 1]
                else:
                    print("[Warning] Action may be not standard.")
                if self.grad == False:
                    print(f"[{self.role.upper()}]", thought)
                    print(f"[{self.role.upper()}]", action)
                # print(thought)
                # print(action)
                return thought, action
            except:
                print("[Error] Action or thought missing!")
                thought = "Invalid output format. No valid thought is generated."
                action = "Invalid output format. No valid action is generated."
                print(thought)
                print(action)
                return thought, action

        elif self.role == 'teacher':
            prompt = self.prompt_1.format(
                context=context,
                question=question,
                observation=observation,
            )
            thoughtNaction = self.LLM4generate(prompt=prompt).strip()
            # breakpoint()
            try:
                thought, action = self._tNaparse(thoughtNaction)
                end_index = action.find("]")
                if end_index != -1:
                    action = action[:end_index + 1]
                else:
                    print("[Warning] Action may be not standard.")
                if self.grad == False:
                    print(f"[{self.role.upper()}]", thought)
                    print(f"[{self.role.upper()}]", action)
                # print(thought)
                # print(action)
                return thought, action
            except:
                print("Wrong!!!")
                thought = "Invalid output format. No valid thought is generated."
                action = "Invalid output format. No valid action is generated."
                print(thought)
                print(action)
                return thought, action

    def _parse(self, thought:str, action:str, data:list, stop:bool):
        flg = False  # Indicate if the move finish
        answer = None
        message = None

        if self.role == 'student':
            if stop == True and action[:6] != 'Finish':
                message = "Can't parse action {}".format(action)
                return flg, answer, message
            if action[:6] == 'Search':
                entity = action[7:-1]
                message = self.tool.retrieve(entity, data)
                return flg, answer, message
            elif action[:6] == 'Finish':
                answer = action[7:-1]
                flg = True
                message = 'Finish the answer.'
                return flg, answer, message
            else:
                message = "Can't parse action {}".format(action)

        elif self.role == 'teacher':
            message = thought
        return flg, answer, message

    def monkey_patch_o_proj(self, o_proj_layer, base_weight, lora_update):
        old_forward = o_proj_layer.forward
        def new_forward(x):
            u = lora_update
            if u.dtype != base_weight.dtype:
                u = u.to(base_weight.dtype)
            if u.device != base_weight.device:
                u = u.to(base_weight.device)
            merged_w = base_weight + u
            if merged_w.dtype != x.dtype:
                merged_w = merged_w.to(x.dtype)

            bias = o_proj_layer.bias
            if bias is not None and bias.dtype != x.dtype:
                bias = bias.to(x.dtype)
            return F.linear(x, merged_w, bias)
        o_proj_layer.forward = new_forward
        return old_forward

    def _get_decoder_layers(self, model):
        core = getattr(model, "model", getattr(model, "transformer", model))
        layers = getattr(core, "layers", None)
        if layers is None:
            raise AttributeError("Cannot find model layers at model.model.layers or model.transformer.layers")
        return layers

    def monkey_patch_all_layers(self, model, lora_update_fn, num_layers_to_patch="all"):
        layers = self._get_decoder_layers(model)
        total = len(layers)
        if isinstance(num_layers_to_patch, str):
            flag = num_layers_to_patch.lower().strip()
            if flag in ("all", "*"):
                n = total
            else:
                raise ValueError(f"Unknown value for num_layers_to_patch: {num_layers_to_patch}")
        else:
            n = int(num_layers_to_patch)
            if n <= 0:
                return []
            n = min(n, total)

        start_idx = total - n
        olds = []
        for li, layer in enumerate(layers[start_idx:], start=start_idx):
            o_proj = layer.self_attn.o_proj
            base_weight = o_proj.weight
            lora_update = lora_update_fn()
            old_forward = self.monkey_patch_o_proj(o_proj, base_weight, lora_update)
            olds.append((o_proj, old_forward))
        # print(f"[LoRA] Patched layers: {n}/{total} (indices {list(range(start_idx, total))})")
        return olds


    def restore_all_layers(self, olds):
        for (o_proj, old_forward) in olds:
            o_proj.forward = old_forward


    def _sample_from_logits(self, logits, do_sample: bool, temperature: float=1.0,
                            top_k: int=0, top_p: float=1.0):
        logits = logits / max(temperature, 1e-8)
        probs = torch.softmax(logits, dim=-1)

        if not do_sample:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)  # [B,1]
            log_p = torch.log(probs.gather(-1, next_token))         # [B,1]
            return next_token.squeeze(-1), log_p.squeeze(-1)
        
        if top_k > 0:
            topk = torch.topk(probs, top_k, dim=-1)
            mask = torch.zeros_like(probs).scatter_(-1, topk.indices, 1.0)
            probs = probs * mask
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            mask = (cumprobs <= top_p)
            mask[..., 0] = True
            keep = torch.zeros_like(probs, dtype=torch.bool).scatter(-1, sorted_idx, mask)
            probs = torch.where(keep, probs, torch.zeros_like(probs))
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        next_token = torch.multinomial(probs, num_samples=1)         # [B,1]
        log_p = torch.log(probs.gather(-1, next_token))              # [B,1]
        return next_token.squeeze(-1), log_p.squeeze(-1)


    def _manual_decode_with_logprob(self, inputs, max_new_tokens, temperature, top_k, top_p):
        input_ids = inputs["input_ids"].to(self.model.device)
        generated_tokens = []
        past_key_values = None
        log_probs_list = []

        with torch.enable_grad():
            for _ in range(max_new_tokens):
                outputs = self.model.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                last_hidden = outputs.last_hidden_state[:, -1:, :]      # [B,1,H]
                logits = self.model.lm_head(last_hidden)[:, -1, :]      # [B,V]

                next_token, log_p = self._sample_from_logits(
                    logits,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                log_probs_list.append(log_p.squeeze(0))

                if next_token.item() == self.tokenizer.eos_token_id:
                    generated_tokens.append(next_token.item())
                    break

                generated_tokens.append(next_token.item())
                input_ids = next_token.unsqueeze(0)

        if len(log_probs_list) > 0:
            self.probs.append(torch.stack(log_probs_list).sum())

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def LLM4generate(self, prompt: str, max_new_tokens=256,
                     temperature: float=1.0, top_k: int=0, top_p: float=0.9) -> str:
        message = {"role": "user", "content": prompt}
        prompt_text = self.tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")

        if self.grad:
            return self._manual_decode_with_logprob(inputs, max_new_tokens, temperature, top_k, top_p)

        self.model.eval()
        with torch.inference_mode():
            out = self.model.generate(
                **inputs.to(self.model.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        new_tokens = out[0, inputs["input_ids"].size(1):]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


    @torch.inference_mode()
    def LLM4generate_batch(self, prompts: list[str], *,
        max_new_tokens=256,
        do_sample: bool=False,
        temperature: float=None,
        top_k: int=None,
        top_p: float=None,
        ab: tuple | None = None,
        alpha: float = 0.01
    ) -> list[str]:
    
        msgs = [[{"role":"user","content":p}] for p in prompts]
        prompt_txts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                    for m in msgs]
        batch = self.tokenizer(prompt_txts, return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(self.model.device) for k, v in batch.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=bool(do_sample),
            temperature=temperature if do_sample else None,
            top_k=top_k if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

        if ab is not None:
            A_batch, B_batch = ab
            with self._set_ab_batch(A_batch, B_batch, alpha):
                outs = self.model.generate(**batch, **gen_kwargs)
        else:
            outs = self.model.generate(**batch, **gen_kwargs)

        dec = []
        inp_lens = batch["input_ids"].shape[1]
        for i in range(outs.size(0)):
            new_tokens = outs[i, inp_lens:]
            dec.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return dec



def run(question:str, teacher:Agent, student:Agent, data:list, budget=3):
    teacher_observations = []
    student_observations = []
    student_context = "<Because it's the first step, there is no previous context>"
    teacher_context = "<Because it's the first step, there is no previous context>"
    time_step = 0
    teacher_msg_tube = []
    student_msg_tube = []

    while True:
        stop = not(budget > time_step)
        student_observation = student._observe(
            time_step, 
            student_msg_tube, 
            teacher_msg_tube
        )
        student_observations.append(student_observation)
        student_thought, student_action = student._thinkNaction(
            stop,
            student_context,
            question,
            student_observation
        )

        flg, answer, student_message = student._parse(
            student_thought, 
            student_action, 
            data,
            stop
        )

        if flg:
            if answer is None:
                return ""
            return answer
        
        if stop:
            if answer is None:
                return ""
            return answer


        student_msg_tube.append({"action": student_action, "message": student_message})
        student_context = student._context(
            question,
            student_context,
            student_observations[-1],
            student_thought,
            student_action
        )
        
        teacher_observation = teacher._observe(
            time_step, 
            student_msg_tube,
            teacher_msg_tube
        )
        teacher_observations.append(teacher_observation)
        teacher_thought, teacher_action = teacher._thinkNaction(
            False,
            teacher_context,
            question,
            teacher_observation
        )

        _, _, teacher_message = teacher._parse(
            teacher_thought,
            teacher_action,
            data,
            stop
        )
        teacher_msg_tube.append({"action": teacher_action, "message": teacher_message})
        teacher_context = teacher._context(
            question,
            teacher_context,
            teacher_observations[-1],
            teacher_thought,
            teacher_action
        )
        time_step += 1



def run_batch(
    questions: List[str],
    teacher:Agent, student:Agent,
    datas: List[List[str]],
    *,
    t_As: List[torch.Tensor],
    t_Bs: List[torch.Tensor], 
    s_As: List[torch.Tensor],
    s_Bs: List[torch.Tensor],
    alpha: float = 0.01,
    budget: int = 3,
    max_new_tokens: int = 256
) -> List[str]:

    B = len(questions)
    student_ctx = ["<Because it's the first step, there is no previous context>"] * B
    teacher_ctx = ["<Because it's the first step, there is no previous context>"] * B
    student_tube = [[] for _ in range(B)]
    teacher_tube = [[] for _ in range(B)]
    finished = [False] * B
    answers = [""] * B

    for t in range(budget + 1):
        # print("=== Step", t, "===")

        active_idx = [i for i in range(B) if not finished[i]]
        if not active_idx:
            break

        stud_obs = []
        for i in active_idx:
            if t == 0:
                stud_obs.append("Start to answer the question.")
            else:
                s_msg = student_tube[i][-1]['action'] + " " + student_tube[i][-1]['message']
                t_msg = teacher_tube[i][-1]['action'] + " " + teacher_tube[i][-1]['message']
                stud_obs.append(f"Search Result: {s_msg} Teacher's Advice: {t_msg}")
        # if stud_obs:
        #     print("Student Observations:", stud_obs[0])

        stop_flag = not (budget > t)
        stud_prompts = []
        for i, obs in zip(active_idx, stud_obs):
            budget_tip = ("You have no search budget. Now you must finish the answer according to current information."
                          if stop_flag else "You can search at this step.")
            stud_prompts.append(
                student.prompt_1.format(
                    context=student_ctx[i],
                    question=questions[i],
                    observation=obs,
                    budget=budget_tip
                )
            )

        sA_sub = [s_As[i] for i in active_idx]
        sB_sub = [s_Bs[i] for i in active_idx]

        stud_texts = student.LLM4generate_batch(
            stud_prompts, ab=(sA_sub, sB_sub), alpha=alpha, max_new_tokens=max_new_tokens
        )

        parsed = []
        for i, txt in zip(active_idx, stud_texts):
            try:
                thought, action = student._tNaparse(txt)
                end_ix = action.find("]")
                if end_ix != -1:
                    action = action[:end_ix+1]
            except Exception:
                thought = "Invalid output format. No valid thought is generated."
                action = "Invalid output format. No valid action is generated."

            flg, ans, msg = student._parse(thought, action, datas[i], stop_flag)
            parsed.append((i, thought, action, flg, ans, msg))

        # if parsed:
        #     print("Student Parsed Results:", parsed[0])

        sum_prompts, need_sum_idx = [], []
        for (i, thought, action, flg, ans, msg) in parsed:
            if flg:
                finished[i] = True
                answers[i] = ans or ""
                continue
            student_tube[i].append({"action": action, "message": msg})
            s_prompt = student.prompt_2.format(
                question=questions[i],
                context=student_ctx[i],
                observation=("Start to answer the question." if t == 0 else stud_obs[active_idx.index(i)]),
                thought=thought,
                action=action
            )
            sum_prompts.append(s_prompt)
            need_sum_idx.append(i)

        if stop_flag:
            for i in range(B):
                if not finished[i]:
                    finished[i] = True
                    if not answers[i]:
                        answers[i] = ""
            return answers

        if sum_prompts:
            sA_sum = [s_As[i] for i in need_sum_idx]
            sB_sum = [s_Bs[i] for i in need_sum_idx]
            sums = student.LLM4generate_batch(sum_prompts, ab=(sA_sum, sB_sum), alpha=alpha, max_new_tokens=max_new_tokens)
            for i, txt in zip(need_sum_idx, sums):
                try:
                    student_ctx[i] = student._sparse(txt)
                except Exception:
                    student_ctx[i] = "New context generation failed in the last round. Here is the previous context: " + student_ctx[i]

        if all(finished):
            break
        # print("Student Contexts Updated:", student_ctx[0])

        active_idx = [i for i in range(B) if not finished[i]]
        if not active_idx:
            break

        teach_obs = []
        for i in active_idx:
            s_msg = student_tube[i][-1]['action'] + " " + student_tube[i][-1]['message']
            if t == 0:
                teach_obs.append(f"Student's Action: {s_msg}")
            else:
                t_msg = teacher_tube[i][-1]['action'] + " " + teacher_tube[i][-1]['message']
                teach_obs.append(f"Previous Advice: {t_msg} Student's Action: {s_msg}")
        # if teach_obs:
        #     print("Teacher Observations:", teach_obs[0])

        teach_prompts = [
            teacher.prompt_1.format(context=teacher_ctx[i], question=questions[i], observation=obs)
            for i, obs in zip(active_idx, teach_obs)
        ]

        tA_sub = [t_As[i] for i in active_idx]
        tB_sub = [t_Bs[i] for i in active_idx]
        teach_texts = teacher.LLM4generate_batch(
            teach_prompts, ab=(tA_sub, tB_sub), alpha=alpha, max_new_tokens=max_new_tokens
        )

        parsed_t = []
        for i, txt in zip(active_idx, teach_texts):
            try:
                th, ac = teacher._tNaparse(txt)
                end_ix = ac.find("]")
                if end_ix != -1:
                    ac = ac[:end_ix+1]
            except Exception:
                th = "Invalid output format. No valid thought is generated."
                ac = "Invalid output format. No valid action is generated."
            _, _, msg = teacher._parse(th, ac, datas[i], stop=False)  # teacher 的 message=thought
            parsed_t.append((i, th, ac, msg))

        # if parsed_t:
        #     print("Teacher Parsed Results:", parsed_t[0])

        sum_prompts_t, need_sum_idx_t = [], []
        for (i, th, ac, msg) in parsed_t:
            teacher_tube[i].append({"action": ac, "message": msg})
            tp = teacher.prompt_2.format(
                question=questions[i], context=teacher_ctx[i],
                observation=teach_obs[active_idx.index(i)], thought=th, action=ac
            )
            sum_prompts_t.append(tp); need_sum_idx_t.append(i)

        if sum_prompts_t:
            tA_sum = [t_As[i] for i in need_sum_idx_t]
            tB_sum = [t_Bs[i] for i in need_sum_idx_t]
            sums_t = teacher.LLM4generate_batch(sum_prompts_t, ab=(tA_sum, tB_sum), alpha=alpha, max_new_tokens=max_new_tokens)
            for i, txt in zip(need_sum_idx_t, sums_t):
                try:
                    teacher_ctx[i] = teacher._sparse(txt)
                except Exception:
                    teacher_ctx[i] = "New context generation failed in the last round. Here is the previous context: " + teacher_ctx[i]

        # print("Teacher Contexts Updated:", teacher_ctx[0])

    return answers


def run_batch_no_lora(
    questions: List[str],
    teacher:Agent, student:Agent,
    datas: List[List[str]],
    *,
    budget: int = 3,
    max_new_tokens: int = 256
) -> List[str]:
    B = len(questions)
    student_ctx = ["<Because it's the first step, there is no previous context>"] * B
    teacher_ctx = ["<Because it's the first step, there is no previous context>"] * B
    student_tube = [[] for _ in range(B)]
    teacher_tube = [[] for _ in range(B)]
    finished = [False] * B
    answers = [""] * B

    for t in range(budget + 1):
        # ---- Student: Observe ----
        active_idx = [i for i in range(B) if not finished[i]]
        if not active_idx:
            break

        stud_obs = []
        for i in active_idx:
            if t == 0:
                stud_obs.append("Start to answer the question.")
            else:
                s_msg = student_tube[i][-1]['action'] + " " + student_tube[i][-1]['message']
                t_msg = teacher_tube[i][-1]['action'] + " " + teacher_tube[i][-1]['message']
                stud_obs.append(f"Search Result: {s_msg} Teacher's Advice: {t_msg}")

        stop_flag = not (budget > t)
        stud_prompts = []
        for i, obs in zip(active_idx, stud_obs):
            budget_tip = ("You have no search budget. Now you must finish the answer according to current information."
                          if stop_flag else "You can search at this step.")
            stud_prompts.append(
                student.prompt_1.format(
                    context=student_ctx[i],
                    question=questions[i],
                    observation=obs,
                    budget=budget_tip
                )
            )

        # ---- Student: Greedy generation (no LoRA) ----
        stud_texts = student.LLM4generate_batch(
            stud_prompts,
            ab=None,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

        # ---- Parse & Act ----
        parsed = []
        for i, txt in zip(active_idx, stud_texts):
            try:
                thought, action = student._tNaparse(txt)
                end_ix = action.find("]")
                if end_ix != -1:
                    action = action[:end_ix+1]
            except Exception:
                thought = "Invalid output format. No valid thought is generated."
                action = "Invalid output format. No valid action is generated."

            flg, ans, msg = student._parse(thought, action, datas[i], stop_flag)
            parsed.append((i, thought, action, flg, ans, msg))

        # ---- Finish or build summaries ----
        sum_prompts, need_sum_idx = [], []
        for (i, thought, action, flg, ans, msg) in parsed:
            if flg:
                finished[i] = True
                answers[i] = ans or ""
            else:
                student_tube[i].append({"action": action, "message": msg})
                s_prompt = student.prompt_2.format(
                    question=questions[i],
                    context=student_ctx[i],
                    observation=("Start to answer the question." if t == 0 else stud_obs[active_idx.index(i)]),
                    thought=thought,
                    action=action
                )
                sum_prompts.append(s_prompt)
                need_sum_idx.append(i)

        if stop_flag:
            for i in range(B):
                if not finished[i]:
                    finished[i] = True
                    answers[i] = answers[i] or ""
            return answers

        if sum_prompts:
            sums = student.LLM4generate_batch(
                sum_prompts,
                ab=None,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            for i, txt in zip(need_sum_idx, sums):
                try:
                    student_ctx[i] = student._sparse(txt)
                except Exception:
                    student_ctx[i] = "New context generation failed in the last round. Here is the previous context: " + student_ctx[i]

        if all(finished):
            break

        # ---- Teacher: Observe ----
        active_idx = [i for i in range(B) if not finished[i]]
        if not active_idx:
            break

        teach_obs = []
        for i in active_idx:
            s_msg = student_tube[i][-1]['action'] + " " + student_tube[i][-1]['message']
            if t == 0:
                teach_obs.append(f"Student's Action: {s_msg}")
            else:
                t_msg = teacher_tube[i][-1]['action'] + " " + teacher_tube[i][-1]['message']
                teach_obs.append(f"Previous Advice: {t_msg} Student's Action: {s_msg}")

        teach_prompts = [
            teacher.prompt_1.format(context=teacher_ctx[i], question=questions[i], observation=obs)
            for i, obs in zip(active_idx, teach_obs)
        ]

        # ---- Teacher: Greedy generation (no LoRA) ----
        teach_texts = teacher.LLM4generate_batch(
            teach_prompts,
            ab=None,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

        parsed_t = []
        for i, txt in zip(active_idx, teach_texts):
            try:
                th, ac = teacher._tNaparse(txt)
                end_ix = ac.find("]")
                if end_ix != -1:
                    ac = ac[:end_ix+1]
            except Exception:
                th = "Invalid output format. No valid thought is generated."
                ac = "Invalid output format. No valid action is generated."
            _, _, msg = teacher._parse(th, ac, datas[i], stop=False)
            parsed_t.append((i, th, ac, msg))

        sum_prompts_t, need_sum_idx_t = [], []
        for (i, th, ac, msg) in parsed_t:
            teacher_tube[i].append({"action": ac, "message": msg})
            tp = teacher.prompt_2.format(
                question=questions[i], context=teacher_ctx[i],
                observation=teach_obs[active_idx.index(i)], thought=th, action=ac
            )
            sum_prompts_t.append(tp); need_sum_idx_t.append(i)

        if sum_prompts_t:
            sums_t = teacher.LLM4generate_batch(
                sum_prompts_t,
                ab=None,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            for i, txt in zip(need_sum_idx_t, sums_t):
                try:
                    teacher_ctx[i] = teacher._sparse(txt)
                except Exception:
                    teacher_ctx[i] = "New context generation failed in the last round. Here is the previous context: " + teacher_ctx[i]

    return answers
