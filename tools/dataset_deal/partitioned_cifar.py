# # Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at

# #     http://www.apache.org/licenses/LICENSE-2.0

# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import os

# import torch
# from torch.utils.data import DataLoader
# import torchvision
# from torchvision import transforms

# from .basic_dataset import FedDataset, CIFARSubset
# from division import CIFAR10Partitioner, CIFAR100Partitioner, MNISTPartitioner


# class PartitionCIFAR(FedDataset):
#     """:class:`FedDataset` with partitioning preprocess. For detailed partitioning, please
#     check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.

    
#     Args:
#         root (str): Path to download raw dataset.
#         path (str): Path to save partitioned subdataset.
#         dataname (str): "cifar10" or "cifar100"
#         num_clients (int): Number of clients.
#         download (bool): Whether to download the raw dataset.
#         preprocess (bool): Whether to preprocess the dataset.
#         balance (bool, optional): Balanced partition over all clients or not. Default as ``True``.
#         partition (str, optional): Partition type, only ``"iid"``, ``shards``, ``"dirichlet"`` are supported. Default as ``"iid"``.
#         unbalance_sgm (float, optional): Log-normal distribution variance for unbalanced data partition over clients. Default as ``0`` for balanced partition.
#         num_shards (int, optional): Number of shards in non-iid ``"shards"`` partition. Only works if ``partition="shards"``. Default as ``None``.
#         dir_alpha (float, optional): Dirichlet distribution parameter for non-iid partition. Only works if ``partition="dirichlet"``. Default as ``None``.
#         verbose (bool, optional): Whether to print partition process. Default as ``True``.
#         seed (int, optional): Random seed. Default as ``None``.
#         transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
#         target_transform (callable, optional): A function/transform that takes in the target and transforms it.
#     """
#     def __init__(self,
#                  root,
#                  path,
#                  dataname,
#                  num_clients,
#                  download=True,
#                  preprocess=False,
#                  balance=True,
#                  partition="iid",
#                  unbalance_sgm=0,
#                  num_shards=None,
#                  dir_alpha=None,
#                  verbose=True,
#                  seed=None,
#                  transform=None,
#                  target_transform=None) -> None:
#         self.dataname = dataname
#         self.root = os.path.expanduser(root)
#         self.path = path
#         self.num_clients = num_clients
#         self.transform = transform
#         self.targt_transform = target_transform

#         if preprocess:
#             self.preprocess(balance=balance,
#                             partition=partition,
#                             unbalance_sgm=unbalance_sgm,
#                             num_shards=num_shards,
#                             dir_alpha=dir_alpha,
#                             verbose=verbose,
#                             seed=seed,
#                             download=download)

#     def preprocess(self,
#                    balance=True,
#                    partition="iid",
#                    unbalance_sgm=0,
#                    num_shards=None,
#                    dir_alpha=None,
#                    verbose=True,
#                    seed=None,
#                    download=True):
#         """Perform FL partition on the dataset, and save each subset for each client into ``data{cid}.pkl`` file.

#         For details of partition schemes, please check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.
#         """
#         self.download = download

#         if os.path.exists(self.path) is not True:
#             os.mkdir(self.path)
#             os.mkdir(os.path.join(self.path, "train"))
#             os.mkdir(os.path.join(self.path, "var"))
#             os.mkdir(os.path.join(self.path, "test"))
#         # train dataset partitioning
#         if self.dataname == 'cifar10':
#             trainset = torchvision.datasets.CIFAR10(root=self.root,
#                                                     train=True,
#                                                     download=self.download)
#             partitioner = CIFAR10Partitioner(trainset.targets,
#                                              self.num_clients,
#                                              balance=balance,
#                                              partition=partition,
#                                              unbalance_sgm=unbalance_sgm,
#                                              num_shards=num_shards,
#                                              dir_alpha=dir_alpha,
#                                              verbose=verbose,
#                                              seed=seed)
#         elif self.dataname == 'cifar100':
#             trainset = torchvision.datasets.CIFAR100(root=self.root,
#                                                      train=True,
#                                                      download=self.download)
#             partitioner = CIFAR100Partitioner(trainset.targets,
#                                               self.num_clients,
#                                               balance=balance,
#                                               partition=partition,
#                                               unbalance_sgm=unbalance_sgm,
#                                               num_shards=num_shards,
#                                               dir_alpha=dir_alpha,
#                                               verbose=verbose,
#                                               seed=seed)
#         else:
#             raise ValueError(
#                 f"'dataname'={self.dataname} currently is not supported. Only 'cifar10', and 'cifar100' are supported."
#             )

#         subsets = {
#             cid: CIFARSubset(trainset,
#                         partitioner.client_dict[cid],
#                         transform=self.transform,
#                         target_transform=self.targt_transform)
#             for cid in range(self.num_clients)
#         }
#         for cid in subsets:
#             torch.save(
#                 subsets[cid],
#                 os.path.join(self.path, "train", "data{}.pkl".format(cid)))

#     def get_dataset(self, cid, type="train"):
#         """Load subdataset for client with client ID ``cid`` from local file.

#         Args:
#              cid (int): client id
#              type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.

#         Returns:
#             Dataset
#         """
#         dataset = torch.load(
#             os.path.join(self.path, type, "data{}.pkl".format(cid)), weights_only=False)
#         return dataset

#     def get_dataloader(self, cid, batch_size=None, type="train"):
#         """Return dataload for client with client ID ``cid``.

#         Args:
#             cid (int): client id
#             batch_size (int, optional): batch size in DataLoader.
#             type (str, optional): Dataset type, can be ``"train"``, ``"val"`` or ``"test"``. Default as ``"train"``.
#         """
#         dataset = self.get_dataset(cid, type)
#         batch_size = len(dataset) if batch_size is None else batch_size
#         data_loader = DataLoader(dataset, batch_size=batch_size)
#         return data_loader


# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)
# Licensed under the Apache License, Version 2.0 (the "License");

import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision

from .basic_dataset import FedDataset, CIFARSubset
from division import CIFAR10Partitioner, CIFAR100Partitioner


def _stratified_split_by_labels(indices, labels, val_ratio=0.1, seed=0):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for idx in indices:
        by_class[int(labels[idx])].append(idx)
    train_idx, val_idx = [], []
    for cls, idxs in by_class.items():
        rng.shuffle(idxs)
        k = int(len(idxs) * val_ratio)
        val_idx.extend(idxs[:k])
        train_idx.extend(idxs[k:])
    return train_idx, val_idx


def _compute_client_class_counts(client_indices_dict, labels, num_classes):
    num_clients = len(client_indices_dict)
    counts = np.zeros((num_clients, num_classes), dtype=np.int64)
    for cid, idxs in client_indices_dict.items():
        for i in idxs:
            counts[cid, int(labels[i])] += 1
    return counts


def _proportional_partition_test_like_train(train_counts, test_labels, seed=0):
    rng = random.Random(seed)
    num_clients, num_classes = train_counts.shape
    row_sums = train_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    props = train_counts / row_sums  # [C, K]

    test_by_class = defaultdict(list)
    for idx, y in enumerate(test_labels):
        test_by_class[int(y)].append(idx)
    for cls in range(num_classes):
        rng.shuffle(test_by_class[cls])

    assign = {cid: [] for cid in range(num_clients)}
    for cls in range(num_classes):
        pool = test_by_class[cls]
        n = len(pool)
        if n == 0:
            continue
        target = (props[:, cls] * n).astype(int)
        deficit = n - int(target.sum())
        if deficit > 0:
            fractional = props[:, cls] * n - target
            order = np.argsort(-fractional)
            for k in range(deficit):
                target[order[k % num_clients]] += 1
        start = 0
        for cid in range(num_clients):
            m = int(target[cid])
            if m > 0:
                assign[cid].extend(pool[start:start + m])
                start += m
    return assign


class PartitionCIFAR(FedDataset):
    """
    CIFAR-10/100 联邦划分：支持
    - Dirichlet/IID 等训练划分
    - 每客户端分层抽样切出 val
    - test 按各客户端 train 类别占比分配（或中心化）
    - train/val/test 使用不同的 transform
    - 仅在 label_whitelist 内进行划分与保存（可选）
    - 保存布局可选：split-centric 或 train-only
    """

    def __init__(self,
                 root,
                 path,
                 dataname,
                 num_clients,
                 download=True,
                 preprocess=False,
                 balance=True,
                 partition="iid",
                 unbalance_sgm=0,
                 num_shards=None,
                 dir_alpha=None,
                 verbose=True,
                 seed=None,
                 # 三套 transform
                 train_transform=None,
                 val_transform=None,
                 test_transform=None,
                 # 其余
                 target_transform=None,
                 val_ratio=0.1,
                 test_like_train=True,
                 # 新增
                 label_whitelist=None,           # 仅保留这些类别（int 列表），None 表示不过滤
                 save_layout="split-centric"     # "split-centric" | "train-only"
                 ) -> None:

        self.dataname = dataname
        self.root = os.path.expanduser(root)
        self.path = path
        self.num_clients = num_clients

        # 三套图像变换
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        self.target_transform = target_transform
        self.targt_transform = target_transform  # 兼容旧拼写

        self.val_ratio = val_ratio
        self.test_like_train = test_like_train

        # 新增
        self.label_whitelist = None if label_whitelist is None else list(map(int, label_whitelist))
        assert save_layout in ("split-centric", "train-only")
        self.save_layout = save_layout

        if preprocess:
            self.preprocess(balance=balance,
                            partition=partition,
                            unbalance_sgm=unbalance_sgm,
                            num_shards=num_shards,
                            dir_alpha=dir_alpha,
                            verbose=verbose,
                            seed=seed,
                            download=download)

    # ---------------- 内部工具 ----------------
    @staticmethod
    def _to_list_labels(labels):
        return labels if isinstance(labels, list) else list(labels)

    def _filter_indices_by_whitelist(self, indices, all_labels):
        """根据 label_whitelist 过滤索引；若未设置则原样返回。"""
        if not self.label_whitelist:
            return list(indices)
        wl = set(self.label_whitelist)
        return [i for i in indices if int(all_labels[i]) in wl]

    def _ensure_dirs(self):
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(os.path.join(self.path, "train"), exist_ok=True)
        if self.save_layout == "split-centric":
            os.makedirs(os.path.join(self.path, "val"), exist_ok=True)
            os.makedirs(os.path.join(self.path, "test"), exist_ok=True)

    @staticmethod
    def _client_fname(cid):
        return f"client{cid}.pkl"   # 新命名

    @staticmethod
    def _legacy_fname(cid):
        return f"data{cid}.pkl"     # 兼容旧命名

    @staticmethod
    def _count_by_class(indices, labels, num_classes):
        from collections import Counter
        c = Counter(int(labels[i]) for i in indices)
        # 返回稀疏 dict，保留真实出现过的类别
        return dict(c)

    @staticmethod
    def _make_summary_text(task_name, label_name_map, count_dict, top_k=4):
        """生成简短文本摘要。label_name_map: {id->name or str(id)}"""
        total = sum(count_dict.values()) if count_dict else 0
        if total == 0:
            return "This client contains no samples under current label whitelist."
        items = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
        top = items[:top_k]
        top_labels = ", ".join([str(label_name_map.get(k, k)) for k, _ in top])
        dist_str = ", ".join([f"{label_name_map.get(k, k)} {int(v/total*100)}%" for k, v in top])
        if task_name is None:
            task_name = "CIFAR Subset"
        return (
            f"This client contains CIFAR images for the {task_name} task.\n"
            f"Main categories: {top_labels}.\n"
            f"Estimated label distribution: {dist_str}.\n"
            f"Images are 32x32 RGB natural photos."
        )

    # ---------------- 主流程 ----------------
    def preprocess(self,
                   balance=True,
                   partition="iid",
                   unbalance_sgm=0,
                   num_shards=None,
                   dir_alpha=None,
                   verbose=True,
                   seed=None,
                   download=True):
        self.download = download

        # 1) 载入原始集
        if self.dataname == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root=self.root, train=True, download=self.download)
            testset  = torchvision.datasets.CIFAR10(root=self.root, train=False, download=self.download)
            partitioner = CIFAR10Partitioner(trainset.targets, self.num_clients,
                                             balance=balance, partition=partition,
                                             unbalance_sgm=unbalance_sgm, num_shards=num_shards,
                                             dir_alpha=dir_alpha, verbose=verbose, seed=seed)
            num_classes = 10
            id2name = {i: name for i, name in enumerate(
                ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
            )}
        elif self.dataname == 'cifar100':
            trainset = torchvision.datasets.CIFAR100(root=self.root, train=True, download=self.download)
            testset  = torchvision.datasets.CIFAR100(root=self.root, train=False, download=self.download)
            partitioner = CIFAR100Partitioner(trainset.targets, self.num_clients,
                                              balance=balance, partition=partition,
                                              unbalance_sgm=unbalance_sgm, num_shards=num_shards,
                                              dir_alpha=dir_alpha, verbose=verbose, seed=seed)
            num_classes = 100
            # 若需要，可自定义 id->name；默认用 id 字符串
            id2name = {i: str(i) for i in range(100)}
        else:
            raise ValueError("Only 'cifar10' and 'cifar100' are supported.")

        train_labels = self._to_list_labels(trainset.targets)
        test_labels  = self._to_list_labels(testset.targets)

        # 2) 创建目录
        self._ensure_dirs()

        # 3) 每客户端：先拿到 partitioner 的索引，再应用 label_whitelist 过滤
        base_seed = seed or 0
        client_train_indices = {}
        client_val_indices   = {}
        meta = {}              # 统计与摘要
        for cid in range(self.num_clients):
            # 原始划分（全类）
            full_idx = list(partitioner.client_dict[cid])
            # 白名单过滤
            full_idx = self._filter_indices_by_whitelist(full_idx, train_labels)

            # 分层切分 train/val
            tr_idx, val_idx = _stratified_split_by_labels(
                full_idx, train_labels, val_ratio=self.val_ratio, seed=base_seed + cid
            )
            client_train_indices[cid] = tr_idx
            client_val_indices[cid]   = val_idx

            # 构造子集并保存（torch.save，保持你现有格式）
            train_subset = CIFARSubset(trainset, tr_idx,
                                       transform=self.train_transform,
                                       target_transform=self.target_transform)
            if self.save_layout in ("split-centric", "train-only"):
                torch.save(train_subset, os.path.join(self.path, "train", self._client_fname(cid)))

            if self.save_layout == "split-centric":
                val_subset = CIFARSubset(trainset, val_idx,
                                         transform=self.val_transform,
                                         target_transform=self.target_transform)
                torch.save(val_subset, os.path.join(self.path, "val", self._client_fname(cid)))

            # 统计（基于 train 部分）
            cnt = self._count_by_class(tr_idx, train_labels, num_classes=num_classes)
            summary = self._make_summary_text(
                task_name=None,  # 若你在外层有任务名，可在实例化时传入并存到 self 上
                label_name_map=id2name,
                count_dict=cnt,
                top_k=4
            )

            meta[cid] = {
                "sizes": {
                    "train": len(tr_idx),
                    **({"val": len(val_idx)} if self.save_layout == "split-centric" else {})
                },
                "label_count_train": {id2name.get(k, str(k)): int(v) for k, v in cnt.items()},
                "summary": summary
            }

        # 4) test：仅当 split-centric 才保存 test，各客户端仍按 train 分布比例，且受 whitelist 限制
        if self.save_layout == "split-centric":
            if self.test_like_train:
                train_counts = _compute_client_class_counts(client_train_indices, train_labels, num_classes=num_classes)
                client_test_assign = _proportional_partition_test_like_train(
                    train_counts, test_labels, seed=base_seed
                )
                # 白名单过滤
                for cid in range(self.num_clients):
                    assign = self._filter_indices_by_whitelist(client_test_assign[cid], test_labels)
                    test_subset = CIFARSubset(testset, assign,
                                              transform=self.test_transform,
                                              target_transform=self.target_transform)
                    torch.save(test_subset, os.path.join(self.path, "test", self._client_fname(cid)))
                    meta[cid]["sizes"]["test"] = len(assign)
            else:
                # 中心化 test（不分客户端），但仍受 whitelist 限制
                all_test_idx = list(range(len(testset)))
                all_test_idx = self._filter_indices_by_whitelist(all_test_idx, test_labels)
                test_subset = CIFARSubset(testset, all_test_idx,
                                          transform=self.test_transform,
                                          target_transform=self.target_transform)
                torch.save(test_subset, os.path.join(self.path, "test", "data.pkl"))
                meta["central_test"] = {"size": len(all_test_idx)}

        # 5) 在总目录下保存统计与文本描述
        import json
        with open(os.path.join(self.path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        descriptions = {f"client{cid}": meta[cid]["summary"] for cid in range(self.num_clients) if isinstance(cid, int)}
        with open(os.path.join(self.path, "descriptions.json"), "w") as f:
            json.dump(descriptions, f, indent=2, ensure_ascii=False)

    # ---------------- 运行期加载（向后兼容） ----------------
    def _fallback_transform(self, dataset, split):
        """若载入的 pkl 里 transform=None，则用类里对应的默认 transform 补上。"""
        if getattr(dataset, "transform", None) is not None:
            return dataset
        if split == "train" and self.train_transform is not None:
            dataset.transform = self.train_transform
        elif split == "val" and self.val_transform is not None:
            dataset.transform = self.val_transform
        elif split == "test" and self.test_transform is not None:
            dataset.transform = self.test_transform
        return dataset

    def _load_split_file(self, split, cid):
        """
        优先加载新命名 client{cid}.pkl；若不存在则回退到 data{cid}.pkl（兼容旧产物）。
        """
        newp = os.path.join(self.path, split, self._client_fname(cid))
        if os.path.exists(newp):
            return torch.load(newp, weights_only=False)
        oldp = os.path.join(self.path, split, self._legacy_fname(cid))
        return torch.load(oldp, weights_only=False)

    def get_dataset(self, cid, type="train"):
        """Load subdataset for client with client ID `cid` from local file."""
        if type == "test" and self.save_layout == "split-centric" and not self.test_like_train:
            dataset = torch.load(os.path.join(self.path, "test", "data.pkl"), weights_only=False)
            return self._fallback_transform(dataset, "test")

        if self.save_layout == "train-only" and type != "train":
            raise RuntimeError("当前保存布局为 'train-only'，仅支持加载 train。")

        dataset = self._load_split_file(type, cid)
        return self._fallback_transform(dataset, type)

    def get_dataloader(self, cid, batch_size=None, type="train"):
        dataset = self.get_dataset(cid, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader
