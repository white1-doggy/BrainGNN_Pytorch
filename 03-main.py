import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.cuda.amp
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from torch.utils.data import DistributedSampler, Subset
from torch_geometric.data import DataLoader

from imports.ABIDEDataset import ABIDEDataset
from imports.hcp7task_dataset import HCP7TaskDataset
from imports.utils import train_val_test_split
from net.braingnn import Network

torch.manual_seed(123)

EPS = 1e-10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="starting epoch")
    parser.add_argument(
        "--n_epochs", type=int, default=100, help="number of epochs of training"
    )
    parser.add_argument("--batchSize", type=int, default=100, help="size of the batches")
    parser.add_argument(
        "--dataroot",
        type=str,
        default="/home/azureuser/projects/BrainGNN/data/ABIDE_pcp/cpac/filt_noglobal",
        help="root directory of the dataset",
    )
    parser.add_argument("--fold", type=int, default=0, help="training which fold")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--stepsize", type=int, default=20, help="scheduler step size")
    parser.add_argument("--gamma", type=float, default=0.5, help="scheduler shrinking rate")
    parser.add_argument("--weightdecay", type=float, default=5e-3, help="regularization")
    parser.add_argument("--lamb0", type=float, default=1, help="classification loss weight")
    parser.add_argument("--lamb1", type=float, default=0, help="s1 unit regularization")
    parser.add_argument("--lamb2", type=float, default=0, help="s2 unit regularization")
    parser.add_argument("--lamb3", type=float, default=0.1, help="s1 entropy regularization")
    parser.add_argument("--lamb4", type=float, default=0.1, help="s2 entropy regularization")
    parser.add_argument("--lamb5", type=float, default=0.1, help="s1 consistence regularization")
    parser.add_argument("--layer", type=int, default=2, help="number of GNN layers")
    parser.add_argument("--ratio", type=float, default=0.5, help="pooling ratio")
    parser.add_argument("--indim", type=int, default=200, help="feature dim")
    parser.add_argument("--nroi", type=int, default=200, help="num of ROIs")
    parser.add_argument("--nclass", type=int, default=2, help="num of classes")
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument(
        "--optim", type=str, default="Adam", help="optimization method: SGD, Adam"
    )
    parser.add_argument("--save_path", type=str, default="./model/", help="path to save model")
    parser.add_argument(
        "--dataset", type=str, default="ABIDE", help="dataset name: ABIDE or HCP7Task"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="HCP7Task task name (e.g. WM, SOCIAL) for binary classification",
    )
    parser.add_argument(
        "--subject_list",
        type=str,
        default="",
        help="path to subject list file for HCP7Task",
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default="",
        help="JSON path for HCP7Task task config",
    )
    parser.add_argument(
        "--atlas_path",
        type=str,
        default="",
        help="path to atlas nifti for ROI extraction",
    )
    parser.add_argument(
        "--fc_root",
        type=str,
        default="",
        help="root directory of precomputed FC files for HCP7Task",
    )
    parser.add_argument(
        "--roi_ids", type=str, default="", help="comma-separated ROI ids (optional)"
    )
    parser.add_argument("--ddp", action="store_true", help="enable NPU DDP training")
    parser.add_argument(
        "--devices",
        type=int,
        default=8,
        help="number of GPUs to use when --ddp is enabled",
    )
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="29501")
    parser.add_argument("--amp", action="store_true", help="enable NPU AMP")
    return parser.parse_args()


def setup(rank, world_size, master_addr, master_port):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def build_datasets(opt):
    if opt.dataset == "HCP7Task":
        if not opt.subject_list:
            raise ValueError("HCP7Task requires --subject_list")
        if not opt.task_config:
            raise ValueError("HCP7Task requires --task_config")
        if not opt.fc_root:
            raise ValueError("HCP7Task requires --fc_root")
        if not opt.task:
            raise ValueError("HCP7Task requires --task for binary classification")

        with open(opt.subject_list, "r", encoding="utf-8") as f:
            subjects = [line.strip() for line in f.readlines() if line.strip()]

        roi_ids = None
        if opt.roi_ids:
            roi_ids = [int(val) for val in opt.roi_ids.split(",") if val.strip()]

        dataset = HCP7TaskDataset(
            root=opt.dataroot,
            subject_dict=subjects,
            task_config=opt.task_config,
            roi_ids=roi_ids,
            fc_root=opt.fc_root,
            precomputed=True,
            task_name=opt.task,
        )
        opt.nroi = int(dataset.num_rois)
        opt.indim = opt.nroi
        opt.nclass = 2
        split_ratio = [0.7, 0.1, 0.2]
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        train_end = int(len(indices) * split_ratio[0])
        val_end = train_end + int(len(indices) * split_ratio[1])
        train_dataset = Subset(dataset, indices[:train_end])
        val_dataset = Subset(dataset, indices[train_end:val_end])
        test_dataset = Subset(dataset, indices[val_end:])
        return train_dataset, val_dataset, test_dataset

    dataset = ABIDEDataset(opt.dataroot, "ABIDE")
    dataset.data.y = dataset.data.y.squeeze()
    dataset.data.x[dataset.data.x == float("inf")] = 0

    tr_index, val_index, te_index = train_val_test_split(fold=opt.fold)
    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[te_index]
    return train_dataset, val_dataset, test_dataset


def build_loaders(opt, train_dataset, val_dataset, test_dataset, rank, world_size):
    train_sampler = None
    if opt.ddp:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batchSize,
        shuffle=train_sampler is None,
        sampler=train_sampler,
    )
    val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)
    return train_loader, val_loader, test_loader, train_sampler


def build_model(opt, device, rank):
    model = Network(opt.indim, opt.ratio, opt.nclass, R=opt.nroi).to(device)
    if opt.ddp:
        device_ids = [rank] if device.type == "cuda" else None
        model = DDP(model, device_ids=device_ids, find_unused_parameters=True)
    return model


def build_optimizer(opt, model):
    if opt.optim == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=opt.lr, weight_decay=opt.weightdecay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt.lr,
            momentum=0.9,
            weight_decay=opt.weightdecay,
            nesterov=True,
        )
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=opt.stepsize, gamma=opt.gamma
    )
    return optimizer, scheduler


def topk_loss(s, ratio):
    if ratio > 0.5:
        ratio = 1 - ratio
    s = s.sort(dim=1).values
    res = -torch.log(s[:, -int(s.size(1) * ratio) :] + EPS).mean() - torch.log(
        1 - s[:, : int(s.size(1) * ratio)] + EPS
    ).mean()
    return res


def consist_loss(s, device):
    if len(s) == 0:
        return 0
    s = torch.sigmoid(s)
    weight = torch.ones(s.shape[0], s.shape[0], device=device)
    diag = torch.eye(s.shape[0], device=device) * torch.sum(weight, dim=1)
    laplacian = diag - weight
    res = torch.trace(torch.transpose(s, 0, 1) @ laplacian @ s) / (
        s.shape[0] * s.shape[0]
    )
    return res


def train_epoch(model, loader, optimizer, device, opt, scaler, writer, epoch, rank):
    model.train()
    s1_list = []
    s2_list = []
    loss_all = 0
    step = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=opt.amp):
            output, w1, w2, s1, s2 = model(
                data.x, data.edge_index, data.batch, data.edge_attr, data.pos
            )
            s1_list.append(s1.view(-1).detach().cpu().numpy())
            s2_list.append(s2.view(-1).detach().cpu().numpy())

            loss_c = F.nll_loss(output, data.y)
            loss_p1 = (torch.norm(w1, p=2) - 1) ** 2
            loss_p2 = (torch.norm(w2, p=2) - 1) ** 2
            loss_tpk1 = topk_loss(s1, opt.ratio)
            loss_tpk2 = topk_loss(s2, opt.ratio)
            loss_consist = 0
            for c in range(opt.nclass):
                loss_consist += consist_loss(s1[data.y == c], device)
            loss = (
                opt.lamb0 * loss_c
                + opt.lamb1 * loss_p1
                + opt.lamb2 * loss_p2
                + opt.lamb3 * loss_tpk1
                + opt.lamb4 * loss_tpk2
                + opt.lamb5 * loss_consist
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_all += loss.item() * data.num_graphs
        if writer is not None and rank == 0:
            writer.add_scalar(
                "train/classification_loss", loss_c, epoch * len(loader) + step
            )
            writer.add_scalar("train/unit_loss1", loss_p1, epoch * len(loader) + step)
            writer.add_scalar("train/unit_loss2", loss_p2, epoch * len(loader) + step)
            writer.add_scalar("train/TopK_loss1", loss_tpk1, epoch * len(loader) + step)
            writer.add_scalar("train/TopK_loss2", loss_tpk2, epoch * len(loader) + step)
            writer.add_scalar(
                "train/GCL_loss", loss_consist, epoch * len(loader) + step
            )
        step += 1

    s1_arr = np.hstack(s1_list) if s1_list else np.array([])
    s2_arr = np.hstack(s2_list) if s2_list else np.array([])
    return loss_all / len(loader.dataset), s1_arr, s2_arr, w1, w2


def test_acc(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        outputs = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        pred = outputs[0].max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


def test_loss(model, loader, device, opt):
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        output, w1, w2, s1, s2 = model(
            data.x, data.edge_index, data.batch, data.edge_attr, data.pos
        )
        loss_c = F.nll_loss(output, data.y)
        loss_p1 = (torch.norm(w1, p=2) - 1) ** 2
        loss_p2 = (torch.norm(w2, p=2) - 1) ** 2
        loss_tpk1 = topk_loss(s1, opt.ratio)
        loss_tpk2 = topk_loss(s2, opt.ratio)
        loss_consist = 0
        for c in range(opt.nclass):
            loss_consist += consist_loss(s1[data.y == c], device)
        loss = (
            opt.lamb0 * loss_c
            + opt.lamb1 * loss_p1
            + opt.lamb2 * loss_p2
            + opt.lamb3 * loss_tpk1
            + opt.lamb4 * loss_tpk2
            + opt.lamb5 * loss_consist
        )
        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)


def train_worker(rank, world_size, opt):
    if opt.ddp:
        setup(rank, world_size, opt.master_addr, opt.master_port)

    device = (
        torch.device(f"cuda:{rank}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if not os.path.exists(opt.save_path) and rank == 0:
        os.makedirs(opt.save_path)
    writer = SummaryWriter(os.path.join("./log", str(opt.fold))) if rank == 0 else None

    train_dataset, val_dataset, test_dataset = build_datasets(opt)
    train_loader, val_loader, test_loader, train_sampler = build_loaders(
        opt, train_dataset, val_dataset, test_dataset, rank, world_size
    )

    model = build_model(opt, device, rank)
    optimizer, scheduler = build_optimizer(opt, model)
    scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(opt.epoch, opt.n_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        tr_loss, s1_arr, s2_arr, w1, w2 = train_epoch(
            model, train_loader, optimizer, device, opt, scaler, writer, epoch, rank
        )
        scheduler.step()

        if rank == 0:
            tr_acc = test_acc(model, train_loader, device)
            val_acc = test_acc(model, val_loader, device)
            val_loss = test_loss(model, val_loader, device, opt)

            if writer is not None:
                writer.add_scalar("train/acc", tr_acc, epoch)
                writer.add_scalar("val/acc", val_acc, epoch)
                writer.add_scalar("val/loss", val_loss, epoch)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                if opt.save_model:
                    torch.save(best_model_wts, os.path.join(opt.save_path, f"{opt.fold}.pth"))

            print(
                f"Epoch: {epoch} | Train Loss: {tr_loss:.6f} | "
                f"Train Acc: {tr_acc:.6f} | Val Acc: {val_acc:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

    if rank == 0:
        model.load_state_dict(best_model_wts)
        test_accuracy = test_acc(model, test_loader, device)
        test_l = test_loss(model, test_loader, device, opt)
        print(f"Test Acc: {test_accuracy:.7f}, Test Loss: {test_l:.7f}")

    if writer is not None:
        writer.close()

    if opt.ddp:
        cleanup()


def main():
    opt = parse_args()
    world_size = opt.devices if opt.ddp else 1
    if opt.ddp:
        mp.spawn(train_worker, args=(world_size, opt), nprocs=world_size, join=True)
    else:
        train_worker(0, world_size, opt)


if __name__ == "__main__":
    main()
