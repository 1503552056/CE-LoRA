
"""(Triple-LoRA injection, train C only)

"""

import copy
from abc import ABC
from typing import Iterable, List

import torch
import torch.nn as nn
from transformers import AutoConfig

from utils import registry
from models.utils import PromptType 
from models.modules.lora_triple import LinearLoRATriple



def _name_match(name: str, target_keywords: Iterable[str]) -> bool:
    """字符串包含匹配：name 中只要包含任一 target_keywords 即认为命中。"""
    for kw in target_keywords:
        if kw in name:
            return True
    return False


def lora_injection_policy_triple(
    model: nn.Module,
    r: int,
    alpha: float,
    dropout: float,
    target_modules: List[str],
    train_c_only: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """
    遍历模型，将命中 target_modules 的 nn.Linear 替换为 LinearLoRATriple（A、C、B 三因子）。
    仅 C 可训练（train_c_only=True）。
    """
    replace_count = 0

    for module_name, module in list(model.named_modules()):
        # 只替换叶子 Linear（避免重复包裹）
        for child_name, child in list(module.named_children()):
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if isinstance(child, nn.Linear) and _name_match(full_name, target_modules):
                wrapped = LinearLoRATriple(
                    base_linear=child,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    train_c_only=train_c_only,
                )
                setattr(module, child_name, wrapped)
                replace_count += 1
                if verbose:
                    print(
                        f"[TripleLoRA] Replaced Linear -> LinearLoRATriple at: {full_name} "
                        f"(r={r}, alpha={alpha}, dropout={dropout}, train_c_only={train_c_only})"
                    )

    if verbose:
        print(f"[TripleLoRA] Total replaced Linear layers: {replace_count}")
    return model


class BaseModels(nn.Module, ABC):
    """
    抽象基类：
    - _add_base_model()：由子类实现，返回 task 对应的 backbone（如 AutoModelForSequenceClassification）。
    - _add_delta_model()：在 backbone 上注入三分解 LoRA（当 tuning_type == 'lora'）。
    - _add_permutate_layers()：可选的层置换功能（沿用原逻辑）。
    """

    def __init__(self, task_name):
        super().__init__()

        self.task_name = task_name

        config = registry.get("config")
        self.model_config = config.model_config
        self.rank = config.federated_config.rank
        self.logger = registry.get("logger")

    def _build_config(self, **kwargs):
        auto_config = AutoConfig.from_pretrained(
            self.model_config.config_name
            if self.model_config.config_name
            else self.model_config.model_name_or_path,
            finetuning_task=self.task_name if self.task_name else None,
            # cache_dir=self.model_config.cache_dir,
            revision=self.model_config.model_revision,
            use_auth_token=True if self.model_config.use_auth_token else None,
            **kwargs,
        )
        return auto_config

    def _build_model(self):
        backbone = self._add_base_model()

        # 可选：层置换（保持原功能）
        if getattr(self.model_config, "permutation_layers", None):
            backbone = self._add_permutate_layers(backbone)

        # 注入（仅当配置要求微调）
        if self.model_config.tuning_type:
            backbone = self._add_delta_model(backbone)

        return backbone

    def _add_base_model(self):
        """由子类实现，返回具体任务的 backbone 模型。"""
        raise NotImplementedError

    def _add_permutate_layers(self, backbone):
        """仅支持 BERT 系 NLU 任务的层置换（沿用原逻辑）"""
        bert_modules = self.get_bert_module(backbone)
        old_modules = bert_modules.encoder.layer
        scrambled_modules = torch.nn.ModuleList()

        if self.rank > 0:
            permutation = self.model_config.client_model_layers
        else:
            permutation = self.model_config.server_model_layers
        self.logger.debug(f"model's layer: {permutation}")
        for i in permutation:
            assert i <= 11, permutation
            scrambled_modules.append(old_modules[i])

        # 深拷贝 backbone 并替换 encoder.layer
        backbone_copy = copy.deepcopy(backbone)
        bert_modules_copy = self.get_bert_module(backbone_copy)
        bert_modules_copy.encoder.layer = scrambled_modules
        return backbone_copy

    def _add_delta_model(self, backbone):
        """
        注入三因子 LoRA（A、C、B），并仅训练 C（A、B 冻结）。
        不修改 tuning.py 的结构：继续读取 lora_r / lora_alpha / lora_dropout / target_modules。
        """
        # 由外层在构图时放入 registry 的 delta_config（来自 configs/tuning.get_delta_config）
        delta_args = registry.get("delta_config")

        # 读取 LoRA 超参（保持原字段名）
        r = int(delta_args.get("lora_r", 8))
        alpha = float(delta_args.get("lora_alpha", 16))
        dropout = float(delta_args.get("lora_dropout", 0.0))
        target_modules = delta_args.get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2", "dense", "attention.self.query", "attention.self.key", "attention.self.value"]
        )

        # 当选择 LoRA 时，执行“三分解 LoRA 注入（仅 C 训练）”
        tuning = str(getattr(self.model_config, "tuning_type", "lora")).lower()
        print(f"[TripleLoRA] tuning_type='{tuning}' (accepts 'lora' or starting with 'lora')")

        if tuning == "lora" or tuning.startswith("lora"):
            model = lora_injection_policy_triple(
                model=backbone,
                r=r,
                alpha=alpha,
                dropout=dropout,
                target_modules=target_modules,
                train_c_only=False,  # 只训 C
                verbose=True,
            )
        else:
            raise ValueError(
                f"Only 'lora*' is supported in this build, got tuning_type={self.model_config.tuning_type}"
            )


        TRAIN_ABC = True  

        for n, p in model.named_parameters():
            if n.endswith(".A") or ".A." in n:
                p.requires_grad_(TRAIN_ABC)
            elif n.endswith(".B") or ".B." in n:
                p.requires_grad_(TRAIN_ABC)
            elif n.endswith(".C") or ".C." in n:
                p.requires_grad_(True)  
            elif ("LayerNorm" in n) or (".layer_norm" in n):
                p.requires_grad_(True)  
            elif "classifier" in n:
                
                p.requires_grad_(True)  # 或 False，自行决定
            else:
                p.requires_grad_(False)  # 主干其他参数保持冻结

        # 打印可训练参数比例
        try:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"[TripleLoRA] Trainable params: {trainable}/{total} ({trainable/total*100:.2f}%)")
        except Exception:
            pass
        
        # print("[TripleLoRA] ===== Model Structure =====")
        # print(model)
        # print("[TripleLoRA] ===========================")
        # print("=== A/B/C params (trainable flags) ===")
        # for n, p in model.named_parameters():
        #     if n.endswith(".A") or n.endswith(".B") or n.endswith(".C"):
        #         print(n, p.shape, "trainable=", p.requires_grad)

        return model

    def forward(self, inputs):
        """由子类实现 forward。"""
        raise NotImplementedError

    def get_bert_module(self, backbone):
        """按 model_type 返回 bert-like 模块，供层置换使用。"""
        if self.model_config.model_type == "bert":
            return backbone.bert
        elif self.model_config.model_type == "roberta":
            return backbone.roberta
        elif self.model_config.model_type == "distilbert":
            return backbone.distilbert
        else:
            raise NotImplementedError(f"Unsupported model_type: {self.model_config.model_type}")
