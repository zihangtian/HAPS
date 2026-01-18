<h1 align="center">HAPS: Hierarchical LLM Routing with Joint Architecture and Parameter Search</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2512.19682"><img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat-square&logo=arxiv" alt="Paper"></a>
  <a href="https://github.com/zihangtian/HAPS"><img src="https://img.shields.io/badge/GitHub-Code-blue?style=flat-square&logo=github" alt="Code"></a>
</p>

---

## 📌 Introduction

**HAPS** (Hierarchical LLM Routing with Joint Architecture and Parameter Search) introduces a novel approach to routing large language models (LLMs). Traditional methods typically focus on **discrete model architecture selection**, but they often overlook the impact of *model parameter configurations* on performance.

HAPS proposes a **hierarchical routing framework** where:

* The upper-level router selects the best architecture from a set of candidate models (discrete decision),
* The lower-level router optimizes the model's parameters (continuous space),
* Both levels share information through a parameter generation network, enhancing both architecture selection and parameter tuning.

This method effectively combines **discrete architecture search** and **continuous parameter optimization**, allowing the system to select the best model for a task and optimize its parameters for improved performance. The approach has been tested on benchmark tasks, demonstrating superior results compared to existing methods.

In summary, **HAPS bridges the gap between discrete architecture search and continuous parameter optimization to create a more intelligent LLM routing solution**.


## 🛠️ Environment Setup

We recommend using [Conda](https://docs.conda.io/en/latest/) to manage the environment.

```bash
# 1. Create a new conda environment with Python 3.10
conda create -n haps python=3.10 -y

# 2. Activate the environment
conda activate haps

# 3. Install dependencies
pip install -r requirements.txt

```

## 💾 Data & Models

### Datasets

Please place the HotpotQA dataset (e.g., `test.jsonl`) in the `dataset/` folder.

> **Note:** We have included sample `train.jsonl`, `valid.jsonl`, and `test.jsonl` files in the repository for quick reproduction.

### Pre-trained Models

The code is designed to automatically download models from Hugging Face by default:

* `meta-llama/Meta-Llama-3.1-8B-Instruct`
* `Qwen/Qwen2.5-7B-Instruct`
* `mistralai/Mistral-7B-Instruct-v0.3`
* `meta-llama/Llama-3.2-1B-Instruct`
* `princeton-nlp/unsup-simcse-roberta-base`

**Using Local Weights:**
If you have downloaded the weights locally, please update the model paths in the configuration dictionary (usually `model2path`) to your absolute local paths.

## 🚀 Quick Start

### Step 1: Baseline Generation & Sampling

Before running the SFT warm-up, you must generate baseline samples from the candidate model pairs.

**1. Launch Model Servers**
Start the vLLM servers and the retrieval server in separate terminals (using `Llama_Qwen` as an example):

```bash
# Terminal 1: Launch Llama 8B
cd HotpotQA
CUDA_VISIBLE_DEVICES=0 python llama_8b_vllm_2024.py

# Terminal 2: Launch Qwen 7B
CUDA_VISIBLE_DEVICES=1 python qwen_7b_vllm_2025.py

# Terminal 3: Launch Retrieval Server
CUDA_VISIBLE_DEVICES=2 python deploy_simcse.py

```

**2. Configure Ports**
Ensure that the ports defined in `agent.py` and `generate_baselines.py` match the ports launched in the step above.

**3. Run Sampling Script**

```bash
# Generate training baselines
python generate_baselines.py --data_type train --preset llama_qwen --budget 3 --workers 32

# Generate validation baselines
python generate_baselines.py --data_type valid --preset llama_qwen --budget 3 --workers 32

```

### Step 2: SFT Warm-up for High-Level Router

**1. Construct SFT Training Data**
Process the sampled data to create the SFT dataset:

```bash
cd dataset
python build_sft.py \
  --baseline Llama_8B-Qwen_7B/train/baseline.jsonl \
  --output Llama_8B-Qwen_7B/train/sft_train_all.json \
  --seed 43 \
  --pair llama_qwen \
  --tie_policy all \
  --balance none
cd ..

```

**2. Run SFT Training**
Train the router using the constructed data:

```bash
CUDA_VISIBLE_DEVICES=0 python sft.py \
  --pair llama_qwen \
  --epochs 10 \
  --lr 1e-5 \
  --warmup_ratio 0.03 \
  --bsz 8 \
  --grad_accum 8

```

### Step 3: Joint-RL Training

**1. Update Configuration**
Before starting RL training:

* Update the `base_router_path` in the `PRESETS` dictionary within `joint_rl/rl_batch.py` to point to the **best checkpoint path** obtained from Step 2.
* Ensure the `retrieve_url` variable matches your deployed retrieval server address.

**2. Start Training**
Run the Joint-RL training script. You can customize the hyperparameters below as needed:

```bash
cd joint_rl
python rl_batch.py \
  --preset llama_qwen \
  --s_device cuda:0 \
  --t_device cuda:1 \
  --r_device cuda:2 \
  --epochs 3 \
  --alpha 0.01 \
  --high_router_lr 1e-6 \
  --low_router_lr 1e-5 \
  --eval_interval 30 \
  --early_patience_evals 4 \
  --layers 1 \
  --mlp_num_layer 2 \
  --lora_r 8 \
  --batch_route_bs 32 \
  --batch_ab_bs 32 \
  --dialogue_bs 8

```

> **Note:** The script automatically evaluates the model on the validation set during training. If a better validation score is achieved, it will automatically trigger an evaluation on the test set.

## 📖 Citation

If you find HAPS useful for your research, please consider citing:

```bibtex
@article{tian2026haps,
  title={HAPS: Hierarchical LLM Routing with Joint Architecture and Parameter Search},
  author={Tian, Zihang and Li, Rui and Zhang, Jingsen and Bo, Xiaohe and Huo, Wei and Chen, Xu},
  journal={arXiv preprint arXiv:2601.05903},
  year={2026}
}
```
