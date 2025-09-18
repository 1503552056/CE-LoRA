# file: run_partition_cifar_example.py
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset_deal import PartitionCIFAR  # 按你的包结构导入

# （可选）仅统计标签的 collate_fn：更快更省显存
def labels_only_collate(batch):
    # batch: list of (img, label)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return None, labels

def class_histogram(labels: torch.Tensor, num_classes: int):
    counts = torch.bincount(labels, minlength=num_classes)
    return counts.tolist()

if __name__ == "__main__":
    # ---------------- 参数 ----------------
    root_dir   = "./dataset/cifar_raw"          # 原始数据目录（torchvision 会在这找/存）
    save_dir   = "./dataset/cifar_partitioned"  # 划分后的保存目录
    dataname   = "cifar10"                      # "cifar10" or "cifar100"
    num_clients = 10
    partition   = "dirichlet"
    dir_alpha   = 0.5
    seed        = 123
    val_ratio   = 0.1
    test_like_train = True

    # ---------------- transforms ----------------
    if dataname == "cifar10":
        MEAN = (0.4914, 0.4822, 0.4465)
        STD  = (0.2023, 0.1994, 0.2010)
        num_classes = 10
    else:  # cifar100
        MEAN = (0.5071, 0.4867, 0.4408)
        STD  = (0.2675, 0.2565, 0.2761)
        num_classes = 100

    # 训练：随机增强 + 归一化
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    # 验证/测试：确定性预处理 + 归一化（不要随机增强）
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # ---------------- 划分与保存 ----------------
    ds = PartitionCIFAR(
        root=root_dir,
        path=save_dir,
        dataname=dataname,
        num_clients=num_clients,
        download=False,          # 本地已有原始数据就 False；否则 True
        preprocess=True,         # 立即划分并保存 pkl
        balance=True,
        partition=partition,
        dir_alpha=dir_alpha,
        seed=seed,
        # 三套 transform
        train_transform=train_transform,
        val_transform=eval_transform,
        test_transform=eval_transform,
        val_ratio=val_ratio,
        test_like_train=test_like_train
    )

    # ---------------- 选择一个客户端，加载 train/val/test ----------------
    client_id = 0

    train_ds = ds.get_dataset(cid=client_id, type="train")
    val_ds   = ds.get_dataset(cid=client_id, type="val")
    test_ds  = ds.get_dataset(cid=client_id, type="test")

    print(f"[Client {client_id}] sizes -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
