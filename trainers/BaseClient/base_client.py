from abc import ABC
from typing import List, Dict, Tuple
from thop import profile
from thop import clever_format

import os
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from utils import registry
from utils import get_parameter_number
from fedlab.utils import MessageCode, SerializationTool
from fedlab.core.client.trainer import ClientTrainer
from fedlab.core.client.manager import PassiveClientManager
from fedlab.core.client.manager import ORDINARY_TRAINER, SERIAL_TRAINER

# --- sklearn 依赖（GMM / PCA） ---
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


class BaseClientTrainer(ClientTrainer, ABC):
    def __init__(self, model, train_dataset, valid_dataset):

        self._model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self._before_training()

    def _before_training(self):
        """before training function"""

        self.type = SERIAL_TRAINER  # represent serial trainer

        config = registry.get("config")
        self.model_config = config.M
        self.data_config = config.D
        self.training_config = config.T
        self.federated_config = config.F

        self.client_num = len(config.F.clients_id_list)
        self.device = config.training_config.device
        self.rank = config.federated_config.rank
        self.param_list = []
        self.logger = registry.get("logger")

        self._build_metric()
        self._build_eval()

        # key: client idx, value: valid metric
        self.loc_best_metric = {}
        # key: client idx, value: test metric
        self.loc_test_metric = {}
        # key: client idx, value: serialized params
        self.loc_best_params = {}
        # local patient times
        self.loc_patient_times = 0
        # local early stop
        self.stop_early = False

        self.metric_name = self.metric.metric_name
        self._model.to(self.device)

        # ====== 新增：在客户端启动时为本进程持有的 client_id 生成 GMM（只做一次，可跳过） ======
        try:
            if bool(getattr(self.federated_config, "data_sim_auto", True)):
                self._maybe_prepare_data_similarity_npz()
        except Exception as e:
            self.logger.warning(f"[DataSim][Client] prepare GMM NPZ failed: {e}")

        if self.federated_config.rank == -1:
            self._calculate_model_computation()

    # ----------------- 数据相似度-----------------

    def _data_sim_root(self) -> str:
        root = getattr(self.federated_config, "data_sim_root", "") or \
               os.path.join(self.training_config.checkpoint_dir, "datasim")
        os.makedirs(root, exist_ok=True)
        gmm_dir = os.path.join(root, "gmm")
        os.makedirs(gmm_dir, exist_ok=True)
        return root

    def _maybe_prepare_data_similarity_npz(self):
        """
        为本进程持有的数据（train_dataset 可以是 dict[client_id] 或 DataLoader）
        生成 per-client 的 GMM NPZ：<data_sim_root>/gmm/client_{cid}.npz

        注意：
        - 我们直接用“当前 backbone 的 [CLS] 向量”当作特征，避免重新写文本编码脚本；
        - 如果你想改为完全离线的 HF 编码，也可以替换 _extract_features_for_client。
        """
        # 仅当 train_dataset 是 dict 时，才有多个 client_id；否则认为是单客户端数据
        if isinstance(self.train_dataset, dict):
            client_ids = list(self.train_dataset.keys())
        else:
            # 单个客户端：尝试用 rank 作为 cid（也可以从 federated_config 提供一个 id）
            client_ids = [getattr(self.federated_config, "rank", 0)]

        root = self._data_sim_root()
        gmm_dir = os.path.join(root, "gmm")
        overwrite = bool(getattr(self.federated_config, "data_sim_overwrite", False))

        gmm_components = int(getattr(self.federated_config, "data_sim_gmm_components", 3))
        cov_type = str(getattr(self.federated_config, "data_sim_gmm_covariance_type", "diag"))
        reg_covar = float(getattr(self.federated_config, "data_sim_gmm_reg_covar", 1e-5))
        gmm_max_iter = int(getattr(self.federated_config, "data_sim_gmm_max_iter", 200))
        gmm_tol = float(getattr(self.federated_config, "data_sim_gmm_tol", 1e-3))
        pca_dim = int(getattr(self.federated_config, "data_sim_pca_dim", 0))
        pca_whiten = bool(getattr(self.federated_config, "data_sim_pca_whiten", False))
        seed = int(getattr(self.training_config, "seed", 42))

        self.logger.info(f"[DataSim][Client] preparing GMM NPZ to {gmm_dir} (clients={client_ids})")

        for cid in client_ids:
            out_npz = os.path.join(gmm_dir, f"client_{int(cid)}.npz")
            if os.path.exists(out_npz) and not overwrite:
                self.logger.info(f"[DataSim][Client] exists, skip: {out_npz}")
                continue

            # 取该 client 的 dataloader
            dl = self._get_dataloader(dataset=self.train_dataset, client_id=cid)

            # 抽特征（CLS）与标签
            X, y = self._extract_features_for_client(dl)
            if X.shape[0] == 0:
                self.logger.warning(f"[DataSim][Client] no data for client {cid}, skip")
                continue

            # 可选 PCA
            Xp, pca = self._maybe_pca(X, pca_dim=pca_dim, whiten=pca_whiten, seed=seed)

            # 按类别拟合 GMM
            label_models = self._fit_gmm_per_class(
                Xp, y,
                n_components=gmm_components,
                covariance_type=cov_type,
                reg_covar=reg_covar,
                max_iter=gmm_max_iter,
                tol=gmm_tol,
                random_state=seed
            )

            # 保存 NPZ
            meta = dict(
                client_id=int(cid),
                encoder="backbone_cls",
                model_type=str(self.model_config.model_type),
                pca_dim=(Xp.shape[1] if pca is not None else 0),
                covariance_type=cov_type,
                gmm_components=gmm_components,
                reg_covar=reg_covar, max_iter=gmm_max_iter, tol=gmm_tol, seed=seed
            )
            self._save_gmm_npz(out_npz, label_models, pca, meta)
            self.logger.info(f"[DataSim][Client] GMM saved: {out_npz}")

    @torch.no_grad()
    def _extract_features_for_client(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        从当前模型 backbone 提取 [CLS] 表征作为特征；
        """
        self._model.eval()
        feats, labels = [], []

        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, attn_mask, token_type_ids, y = batch[0], batch[1], batch[2], batch[3]

            # --- 获取 backbone & 模型类型 ---
            mt = self.model_config.model_type
            bb = self._model.backbone

            if mt == "roberta":
                out = bb.roberta(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
                cls = out.last_hidden_state[:, 0, :]  # [B, H]
            elif mt == "bert":
                out = bb.bert(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids, return_dict=True)
                cls = out.last_hidden_state[:, 0, :]
            elif mt == "distilbert":
                out = bb.distilbert(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
                cls = out.last_hidden_state[:, 0, :]
            else:
                # 尝试通用 fallback：如果 backbone 有 base model 并返回 last_hidden_state
                out = bb(**{
                    "input_ids": input_ids,
                    "attention_mask": attn_mask,
                    "token_type_ids": token_type_ids if token_type_ids is not None else None,
                    "return_dict": True
                })
                if hasattr(out, "last_hidden_state"):
                    cls = out.last_hidden_state[:, 0, :]
                else:
                    # 再退一步，直接用分类头输入前的 dense（不可用时抛错）
                    raise NotImplementedError(f"[DataSim] Unsupported model_type for feature extraction: {mt}")

            feats.append(cls.detach().cpu().numpy())
            y_np = y.detach().cpu().numpy()
            labels.append(y_np)

        X = np.concatenate(feats, axis=0) if feats else np.zeros((0, getattr(self._model.backbone.config, "hidden_size", 768)), dtype=np.float32)
        y = np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=np.int64)
        return X, y

    # ---- GMM / PCA utils（与我们前面脚本一致）----

    def _maybe_pca(self, X: np.ndarray, pca_dim: int, whiten: bool, seed: int):
        if pca_dim is None or pca_dim <= 0 or pca_dim >= X.shape[1]:
            return X, None
        pca = PCA(n_components=pca_dim, whiten=whiten, random_state=seed)
        Xp = pca.fit_transform(X)
        self.logger.info(f"[DataSim][Client] PCA: {X.shape[1]} -> {Xp.shape[1]} (whiten={whiten})")
        return Xp, pca

    def _extract_covariances(self, gmm: GaussianMixture) -> np.ndarray:
        ct = gmm.covariance_type
        if ct in ("full", "diag"):
            return gmm.covariances_.copy()
        elif ct == "tied":
            C = gmm.covariances_
            G = gmm.weights_.shape[0]
            return np.stack([C.copy() for _ in range(G)], axis=0)
        elif ct == "spherical":
            G = gmm.weights_.shape[0]
            D = gmm.means_.shape[1]
            sig2 = gmm.covariances_.reshape(G, 1, 1)
            return np.eye(D)[None, ...] * sig2
        else:
            raise ValueError(f"Unsupported covariance_type: {ct}")

    def _fit_gmm_per_class(
        self, X: np.ndarray, y: np.ndarray, n_components: int = 3,
        covariance_type: str = "diag", reg_covar: float = 1e-5,
        max_iter: int = 200, tol: float = 1e-3, random_state: int = 42
    ) -> Dict[int, Dict]:
        models: Dict[int, Dict] = {}
        classes = np.unique(y)
        for cls in classes:
            Xk = X[y == cls]
            n_k, D = Xk.shape
            if n_k <= 1:
                self.logger.info(f"[DataSim][Client] SKIP label {cls}: n={n_k}")
                continue
            n_comp = min(n_components, n_k)
            gmm = GaussianMixture(
                n_components=n_comp, covariance_type=covariance_type,
                reg_covar=reg_covar, max_iter=max_iter, tol=tol,
                random_state=random_state, init_params="kmeans"
            )
            gmm.fit(Xk)
            models[int(cls)] = {
                "weights": gmm.weights_.copy(),
                "means": gmm.means_.copy(),
                "covariances": self._extract_covariances(gmm),
                "covariance_type": covariance_type,
                "n_samples": int(n_k),
            }
            self.logger.info(f"[DataSim][Client] OK label {cls}: n={n_k}, G={gmm.weights_.shape[0]}, D={D}, cov='{covariance_type}'")
        return models

    def _save_gmm_npz(self, output_npz: str, label_models: Dict[int, Dict], pca: PCA, meta: Dict):
        out = {}
        for lbl, m in label_models.items():
            out[f"{lbl}_weights"] = m["weights"]
            out[f"{lbl}_means"] = m["means"]
            out[f"{lbl}_covariances"] = m["covariances"]
            out[f"{lbl}_n"] = np.array([m["n_samples"]], dtype=np.int64)
            out[f"{lbl}_covariance_type"] = np.array([m["covariance_type"]], dtype=object)
        out["labels_sorted"] = np.array(sorted(label_models.keys()), dtype=np.int64)
        for k, v in meta.items():
            out[f"meta_{k}"] = np.array([v], dtype=object)
        if pca is not None:
            out["pca_components_"] = pca.components_
            out["pca_mean_"] = pca.mean_
            out["pca_whiten"] = np.array([pca.whiten], dtype=object)
            if hasattr(pca, "explained_variance_"):
                out["pca_explained_variance_"] = pca.explained_variance_
            if hasattr(pca, "explained_variance_ratio_"):
                out["pca_explained_variance_ratio_"] = pca.explained_variance_ratio_
        os.makedirs(os.path.dirname(output_npz) or ".", exist_ok=True)
        np.savez(output_npz, **out)

    # ----------------- 原有逻辑保留 -----------------

    def _calculate_model_computation(self):

        dummy_idx = list(self.train_dataset.keys())[0]
        train_loader = self._get_dataloader(dataset=self.train_dataset, client_id=dummy_idx)
        for step, batch in enumerate(train_loader):
            self._model.train()
            batch = tuple(t.to(self.device) for t in batch)

            macs, params = profile(self._model.backbone, inputs=(batch[0],))
            flops, params = clever_format([macs, params], "%.3f")
            self.logger.debug(f"Model Type: {self.model_config.model_type}, "
                              f"Tuning Type: {self.training_config.tuning_type}, "
                              f"Parameters: {get_parameter_number(self._model.backbone)}, "
                              f"FLOPs: {flops}")
            break

    @property
    def uplink_package(self):
        return self.param_list

    def _train_alone(self, idx: int, model_parameters: torch.Tensor, *args, **kwargs):
        """local training for Client"""

        train_loader = self._get_dataloader(dataset=self.train_dataset, client_id=idx)
        if model_parameters is not None:
            SerializationTool.deserialize_model(self._model, model_parameters)

        # build optimizer,scheduler,loss
        optimizer, scheduler = self._build_optimizer(self._model, len(train_loader))
        self._model, optimizer = self._mixed_train_model(self._model, optimizer)
        self._build_loss()

        for epoch in range(0, int(self.training_config.num_train_epochs)):
            self._on_epoch_begin()
            self._on_epoch(train_loader, optimizer, scheduler)
            self._on_epoch_end(idx)
            if self.federated_config.pson and self.stop_early:
                self.logger.critical(f"local stop early in {epoch}")
                break

    def _get_dataloader(self, dataset, client_id: int):
        """Get :class:`DataLoader` for ``client_id``."""
        if isinstance(dataset, dict):
            data_loader = dataset[client_id]
        else:
            data_loader = dataset
        return data_loader

    def local_process(self, id_list: List, payload: List):
        """local process for Federated Learning"""
        model_parameters = payload[0]
        self.param_list = self.fed_train(model_parameters, id_list)
        return self.param_list

    def fed_train(self, model_parameters: torch.Tensor, id_list: List):
        param_list = []

        for idx in id_list:
            self._train_alone(
                idx=idx,
                model_parameters=model_parameters
            )
            param_list.append(self.model_parameters)

        return param_list

    def cen_train(self, *args):
        self._train_alone(
            idx=-1,
            model_parameters=None,
        )

    # Local Training Functions
    def _build_loss(self):
        self.criterion = registry.get_loss_class(self.training_config.loss_name)(
            config=self.training_config
        )

    def _build_optimizer(self, model, train_dl_len):
        if self.training_config.max_steps > 0:
            t_total = self.training_config.max_steps
            self.training_config.num_train_epochs = \
                self.training_config.max_steps // (train_dl_len // self.training_config.gradient_accumulation_steps) + 1
        else:
            t_total = \
                train_dl_len // self.training_config.gradient_accumulation_steps * self.training_config.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer_grouped_parameters = self.get_optimized_model_params(model)

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.training_config.learning_rate,
            eps=self.training_config.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=t_total
        )

        return optimizer, scheduler

    def get_optimized_model_params(self, model):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.backbone.named_parameters() if
                        not any(nd in n for nd in no_decay)], 'weight_decay': self.training_config.weight_decay},
            {'params': [p for n, p in model.backbone.named_parameters() if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        return optimizer_grouped_parameters

    def _mixed_train_model(self, model, optimizer):
        if self.training_config.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.training_config.fp16_opt_level)

        if self.training_config.n_gpu > 1:
            self.logger.warning("We haven't tested our model under multi-gpu. Please be aware!")
            model = torch.nn.DataParallel(model)

        return model, optimizer

    # Local Test Function
    def _build_metric(self):
        self.metric = registry.get_metric_class(self.training_config.metric_name)(
            self.data_config.task_name, self.training_config.is_decreased_valid_metric
        )

    def _build_eval(self):
        self.eval = registry.get_eval_class(self.training_config.metric_name)(
            self.device, self.metric
        )

    def test_on_client(self, test_dataloader):

        for idx in self.loc_best_params:
            loc_best_params = self.loc_best_params[idx]
            SerializationTool.deserialize_model(self._model, loc_best_params)
            result = self.eval.test_and_eval(
                model=self._model,
                valid_dl=test_dataloader,
                model_type=self.model_config.model_type,
                model_output_mode=self.model_config.model_output_mode
            )
            test_metric, test_loss = result[self.metric_name], result["eval_loss"]
            self.logger.critical(
                f"{self.data_config.task_name.upper()} Test, "
                f"Client:{idx}, Test loss:{test_loss:.3f}, "
                f"Test {self.metric_name}:{test_metric:.3f}"
            )
            self.loc_test_metric[idx] = test_metric

    # Local Epoch Function
    def _on_epoch_begin(self):
        self.global_step = 0
        self.tr_loss, self.logging_loss = 0.0, 0.0
        self.total, self.correct = 0, 0

    def _on_epoch(self, train_loader, optimizer, scheduler):
        for step, batch in enumerate(train_loader):
            self._model.train()
            batch = tuple(t.to(self.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]
                      }
            label = inputs['labels']
            if self.model_config.model_type != 'distilbert' or self.model_config.model_type != 'roberta':
                inputs['token_type_ids'] = batch[2] \
                    if self.model_config.model_type in ['bert', 'xlnet'] else None
            outputs = self._model(inputs)

            loss, logits = outputs[:2]
            _, predicted = torch.max(logits, 1)

            optimizer.zero_grad()
            if self.training_config.n_gpu > 1:
                loss = loss.mean()
            if self.training_config.gradient_accumulation_steps > 1:
                loss = loss / self.training_config.gradient_accumulation_steps

            if self.training_config.fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.tr_loss += loss.item()
            if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                if self.training_config.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.training_config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.training_config.max_grad_norm)

                optimizer.step()
                scheduler.step()

                self.global_step += 1

            self.total += label.size(0)
            if self.model_config.model_output_mode == "seq_classification":
                self.correct += (predicted == label).sum().item()

    def _on_epoch_end(self, idx):
        """on epoch end"""

        self.logger.info(f"{self.data_config.task_name.upper()} Train, "
                         f"Client:{idx}, Loss:{self.tr_loss/self.global_step:.3f}, "
                         f"Accuracy:{self.correct/self.total:.3f}")

        if not self.federated_config.pson:
            return

        valid_data = self._get_dataloader(dataset=self.valid_dataset, client_id=idx)

        result = self.eval.test_and_eval(
            model=self._model,
            valid_dl=valid_data,
            model_type=self.model_config.model_type,
            model_output_mode=self.model_config.model_output_mode
        )

        test_metric, test_loss = result[self.metric_name], result["eval_loss"]

        if not self.loc_best_metric.get(idx, None):
            self.loc_best_metric[idx] = float('-inf')
        if self.loc_best_metric[idx] < test_metric:
            self.loc_best_metric[idx] = test_metric
            self.loc_best_params[idx] = SerializationTool.serialize_model(self._model)
            self.loc_patient_times = 0
        else:
            self.loc_patient_times += 1

        self.logger.debug(f"{self.data_config.task_name.upper()} Eval, "
                          f"Client:{idx}, Loss:{test_loss:.3f}, "
                          f"Current {self.metric_name}:{test_metric:.3f}, "
                          f"Best {self.metric_name}:{self.loc_best_metric[idx]:.3f}")

        if self.loc_patient_times >= self.training_config.patient_times:
            self.stop_early = True


class BaseClientManager(PassiveClientManager, ABC):
    def __init__(self, network, trainer):
        self.logger = registry.get("logger")
        super().__init__(network, trainer, self.logger)

    def main_loop(self):
        while True:
            sender_rank, message_code, payload = self._network.recv(src=0)

            if message_code == MessageCode.Exit:
                if self._network.rank == self._network.world_size - 1:
                    self._network.send(message_code=MessageCode.Exit, dst=0)
                break

            elif message_code == MessageCode.ParameterUpdate:

                id_list, payload = payload[0].to(torch.int32).tolist(), payload[1:]

                if self._trainer.type == SERIAL_TRAINER:  # serial
                    self._trainer.local_process(
                        id_list=id_list,
                        payload=payload
                    )

                elif self._trainer.type == ORDINARY_TRAINER:  # ordinary
                    assert len(id_list) == 1
                    self._trainer.local_process(payload=payload)

                self.synchronize()

            else:
                raise ValueError(f"Invalid MessageCode {message_code}. Please check MessageCode list.")

    def synchronize(self):
        self.logger.info("Uploading information to server.")
        self._network.send(
            content=self._trainer.uplink_package,
            message_code=MessageCode.ParameterUpdate,
            dst=0
        )
