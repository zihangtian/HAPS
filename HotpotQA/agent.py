from retrive import Retriever
import requests
from typing import Tuple
import re

class Agent():
    def __init__(
        self, 
        id:int, 
        role:str, 
        prompt_1:str, 
        prompt_2:str, 
        retrieve_url:str
    ) -> None:
        
        self.model=None
        self.id=id
        self.role = role
        self.prompt_1 = prompt_1  # thought and action
        self.prompt_2 = prompt_2  # context summary
        self.temperature = 0
        self.tool = Retriever(retrieve_url)
        self.model2url = {
            "Llama-3.1-8B-Instruct": "http://127.0.0.1:2024",
            "Qwen2.5-7B-Instruct": "http://127.0.0.1:2025",
            "Mistral-7B-Instruct-v0.3": "http://127.0.0.1:2026"
        }
    
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
            print(new_context)
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
        print(self.role, ": ", new_observation, sep="")
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
                print(thought)
                print(action)
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
            try:
                thought, action = self._tNaparse(thoughtNaction)
                end_index = action.find("]")
                if end_index != -1:
                    action = action[:end_index + 1]
                else:
                    print("[Warning] Action may be not standard.")
                print(thought)
                print(action)
                return thought, action
            except:
                print("[Error] Action or thought missing!")
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


    def LLM4generate(self, prompt:str) -> str:
        messages = [{"role": "user", "content": prompt}]
        data = {"messages": messages}

        response = requests.post(self.model2url[self.model], json=data)
        if response.status_code == 200:
            result = response.json()
            generated_text = str(result["response"][0]).strip()
            return generated_text
        else:
            print("Error", response.status_code, response.text)
