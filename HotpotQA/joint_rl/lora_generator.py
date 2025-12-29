import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from transformers import set_seed

__all__ = [
    "GeneratorConfig",
    "LoRAParamGenerator",
    "create_llama_8b",
    "create_llama_3b",
    "create_qwen_7b",
    "create_qwen_3b",
    "create_mistral_7b",
    "create_generator_optimizer",
    "get_linear_scheduler",
]

@dataclass
class GeneratorConfig:
    seed: int = 43
    lora_r: int = 8
    qwen_7b_target_dim: int = 3584
    llama_8b_target_dim: int = 4096
    llama_3b_target_dim: int = 3072
    qwen_3b_target_dim: int = 2048
    mistral_7b_target_dim: int = 4096
    mlp_in_dim: int = 2048
    mlp_hidden_dim: int = 256
    mlp_num_layers: int = 1
    allow_backbone_grad: bool = False


def _init_linear_(layer: nn.Linear):
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _build_mlp(in_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mlp = nn.Sequential(*layers)
    for m in mlp:
        if isinstance(m, nn.Linear):
            _init_linear_(m)
    return mlp


class LoRAParamGenerator(nn.Module):
    def __init__(
        self,
        device: str,
        base_model,
        base_tokenizer,
        target_dim: int,
        lora_r: int = 8,
        mlp_in_dim: int = 3072,
        mlp_hidden_dim: int = 256,
        mlp_num_layers: int = 1,
        allow_backbone_grad: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if seed is not None:
            set_seed(seed)

        self.base_model = base_model.to(device)
        self.tokenizer = base_tokenizer
        self.device = device
        self.allow_backbone_grad = allow_backbone_grad

        self.lora_r = lora_r
        self.target_dim = target_dim

        self.mlp = _build_mlp(mlp_in_dim, mlp_hidden_dim, mlp_num_layers).to(torch.float32)
        self.out_A = nn.Linear(mlp_hidden_dim, target_dim * lora_r).to(torch.float32)
        self.out_B = nn.Linear(mlp_hidden_dim, lora_r * target_dim).to(torch.float32)
        _init_linear_(self.out_A)
        _init_linear_(self.out_B)

        for p in self.mlp.parameters():
            p.requires_grad = True
        for p in self.out_A.parameters():
            p.requires_grad = True
        for p in self.out_B.parameters():
            p.requires_grad = True

        self.to(device)

    def _backbone(self):
        return getattr(self.base_model, "model",
               getattr(self.base_model, "transformer", self.base_model))

    def _encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        ctx = torch.enable_grad() if self.allow_backbone_grad else torch.no_grad()
        with ctx:
            outputs = self._backbone()(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                return_dict=True
            )
            last_hidden = outputs.hidden_states[-1]  # [B, L, H]
            attn = inputs["attention_mask"].unsqueeze(-1)  # [B, L, 1]
            masked_hidden = last_hidden * attn
            lengths = attn.sum(dim=1).clamp(min=1)         # [B, 1]
            pooled = (masked_hidden.sum(dim=1) / lengths).to(torch.float32)  # [B, H]

        return pooled

    def forward(self, text_batch):
        pooled = self._encode(text_batch)                 # [B, mlp_in_dim]
        x = self.mlp(pooled)                              # [B, hidden]
        A = self.out_A(x).view(-1, self.target_dim, self.lora_r)  # [B, D, r]
        B = self.out_B(x).view(-1, self.lora_r, self.target_dim)  # [B, r, D]
        return A, B

    def save_parameters(self, save_path: str):
        checkpoint = {
            "mlp_state_dict": self.mlp.state_dict(),
            "out_A_state_dict": self.out_A.state_dict(),
            "out_B_state_dict": self.out_B.state_dict(),
        }
        torch.save(checkpoint, save_path)

    def load_parameters(self, load_path: str):
        checkpoint = torch.load(load_path, map_location=self.device)
        self.mlp.load_state_dict(checkpoint["mlp_state_dict"])
        self.out_A.load_state_dict(checkpoint["out_A_state_dict"])
        self.out_B.load_state_dict(checkpoint["out_B_state_dict"])

def _make_generator(
    target_dim: int,
    *,
    base_model,
    base_tokenizer,
    device: str = "cuda:0",
    config: Optional[GeneratorConfig] = None,
    **overrides,
) -> LoRAParamGenerator:
    cfg = (config or GeneratorConfig())
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    set_seed(cfg.seed)

    gen = LoRAParamGenerator(
        device=device,
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        target_dim=target_dim,
        lora_r=cfg.lora_r,
        mlp_in_dim=cfg.mlp_in_dim,
        mlp_hidden_dim=cfg.mlp_hidden_dim,
        mlp_num_layers=cfg.mlp_num_layers,
        allow_backbone_grad=cfg.allow_backbone_grad,
        seed=cfg.seed,
    )
    return gen


def create_llama_8b(
    *,
    base_model,
    base_tokenizer,
    device: str = "cuda:0",
    config: Optional[GeneratorConfig] = None,
    **overrides,
) -> LoRAParamGenerator:
    target_dim = (config.llama_8b_target_dim if config is not None else GeneratorConfig().llama_8b_target_dim)
    return _make_generator(target_dim, base_model=base_model, base_tokenizer=base_tokenizer,
                           device=device, config=config, **overrides)


def create_llama_3b(
    *,
    base_model,
    base_tokenizer,
    device: str = "cuda:0",
    config: Optional[GeneratorConfig] = None,
    **overrides,
) -> LoRAParamGenerator:
    target_dim = (config.llama_3b_target_dim if config is not None else GeneratorConfig().llama_3b_target_dim)
    return _make_generator(target_dim, base_model=base_model, base_tokenizer=base_tokenizer,
                           device=device, config=config, **overrides)


def create_qwen_7b(
    *,
    base_model,
    base_tokenizer,
    device: str = "cuda:0",
    config: Optional[GeneratorConfig] = None,
    **overrides,
) -> LoRAParamGenerator:
    target_dim = (config.qwen_7b_target_dim if config is not None else GeneratorConfig().qwen_7b_target_dim)
    return _make_generator(target_dim, base_model=base_model, base_tokenizer=base_tokenizer,
                           device=device, config=config, **overrides)

def create_qwen_3b(
    *,
    base_model,
    base_tokenizer,
    device: str = "cuda:0",
    config: Optional[GeneratorConfig] = None,
    **overrides,
) -> LoRAParamGenerator:
    target_dim = (config.qwen_3b_target_dim if config is not None else GeneratorConfig().qwen_3b_target_dim)
    return _make_generator(target_dim, base_model=base_model, base_tokenizer=base_tokenizer,
                           device=device, config=config, **overrides)

def create_mistral_7b(
    *,
    base_model,
    base_tokenizer,
    device: str = "cuda:0",
    config: Optional[GeneratorConfig] = None,
    **overrides,
) -> LoRAParamGenerator:
    target_dim = (config.mistral_7b_target_dim if config is not None else GeneratorConfig().mistral_7b_target_dim)
    return _make_generator(target_dim, base_model=base_model, base_tokenizer=base_tokenizer,
                           device=device, config=config, **overrides)


def create_generator_optimizer(
    generator: LoRAParamGenerator,
    lr: float = 1e-6,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:

    params = [
        {"params": filter(lambda p: p.requires_grad, generator.mlp.parameters()), "lr": lr},
        {"params": filter(lambda p: p.requires_grad, generator.out_A.parameters()), "lr": lr},
        {"params": filter(lambda p: p.requires_grad, generator.out_B.parameters()), "lr": lr},
    ]
    return torch.optim.AdamW(params, weight_decay=weight_decay)


def get_linear_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    start_factor: float = 1.0,
    end_factor: float = 0.1,
) -> torch.optim.lr_scheduler._LRScheduler:
    return torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_steps
    )
