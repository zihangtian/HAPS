from fastapi import FastAPI, Request
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, set_seed
import uvicorn
import datetime
import torch
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

set_seed(43)
app = FastAPI()

class VLLMModel:
    def __init__(self, model_name):
        print(f"Loading model: {model_name}")
        self.llm = LLM(model=model_name, tensor_parallel_size=2, dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def run(self, messages, config=None):
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sampling_params = SamplingParams(
            temperature=config.get("temperature", 0.0),
            top_p=config.get("top_p", 1.0),
            max_tokens=config.get("max_tokens", 256),
            n=config.get("n", 1),
            seed=43,
            stop=[f"{self.tokenizer.eos_token}"]
        )
        
        outputs = self.llm.generate(formatted_prompt, sampling_params)
        return [output.outputs[0].text for output in outputs]

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model = VLLMModel(model_name)


@app.post("/")
async def generate_response(request: Request):
    json_post = await request.json()
    messages = json_post.get("messages", "")
    config = json_post.get("config", {})

    response = model.run(messages, config)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "response": response,
        "status": 200,
        "time": time
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2026, workers=1)
