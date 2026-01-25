import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce


TASK_NAME_LIST = [
    "WM",
    "SOCIAL",
    "EMOTION",
    "MOTOR",
    "LANGUAGE",
    "RELATIONAL",
    "GAMBLING",
]

TASK_NAME_TO_ID = {
    "WM": 0,
    "SOCIAL": 1,
    "EMOTION": 2,
    "MOTOR": 3,
    "LANGUAGE": 4,
    "RELATIONAL": 5,
    "GAMBLING": 6,
}


@dataclass(frozen=True)
class TaskSample:
    node_path: str
    edge_path: str
    label: int


class TaskGraphDataset(Dataset):
    def __init__(self, node_root: str, edge_root: str, task_name: str):
        self.node_root = node_root
        self.edge_root = edge_root
        self.task_name = task_name
        self.samples = self._collect_samples()

    def _collect_samples(self) -> List[TaskSample]:
        if self.task_name not in TASK_NAME_TO_ID:
            raise ValueError(f"Unknown task_name: {self.task_name}")

        subjects = sorted(
            entry for entry in os.listdir(self.node_root)
            if os.path.isdir(os.path.join(self.node_root, entry))
        )

        samples: List[TaskSample] = []
        for subject in subjects:
            node_task_dir = os.path.join(self.node_root, subject, self.task_name)
            edge_task_dir = os.path.join(self.edge_root, subject, self.task_name)
            if not os.path.isdir(node_task_dir) or not os.path.isdir(edge_task_dir):
                continue

            for label_name in ("0", "1"):
                node_label_dir = os.path.join(node_task_dir, label_name)
                edge_label_dir = os.path.join(edge_task_dir, label_name)
                if not os.path.isdir(node_label_dir) or not os.path.isdir(edge_label_dir):
                    continue

                node_files = sorted(
                    fname for fname in os.listdir(node_label_dir)
                    if fname.endswith(".pt")
                )
                for fname in node_files:
                    node_path = os.path.join(node_label_dir, fname)
                    edge_path = os.path.join(edge_label_dir, fname)
                    if not os.path.isfile(edge_path):
                        continue
                    samples.append(TaskSample(node_path=node_path, edge_path=edge_path, label=int(label_name)))

        if not samples:
            raise RuntimeError(
                "No samples found. Check node_root/edge_root/task_name layout and files."
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        sample = self.samples[idx]
        node_feat = torch.load(sample.node_path)
        edge_weight = torch.load(sample.edge_path)

        if isinstance(node_feat, torch.Tensor):
            x = node_feat.float()
        else:
            x = torch.tensor(node_feat, dtype=torch.float32)

        edge_index, edge_attr = self._edge_tensor_to_index(edge_weight)
        num_nodes = x.size(0)
        pos = torch.eye(num_nodes, dtype=torch.float32)
        y = torch.tensor(sample.label, dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y)

    @staticmethod
    def _edge_tensor_to_index(edge_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(edge_weight, torch.Tensor):
            edge_weight = torch.tensor(edge_weight)

        if edge_weight.is_sparse:
            edge_weight = edge_weight.coalesce()
            edge_index = edge_weight.indices()
            edge_attr = edge_weight.values()
        else:
            nonzero = edge_weight.nonzero(as_tuple=False)
            edge_index = nonzero.t().contiguous()
            edge_attr = edge_weight[nonzero[:, 0], nonzero[:, 1]]

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        num_nodes = edge_weight.size(0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)
        return edge_index.long(), edge_attr.float()
