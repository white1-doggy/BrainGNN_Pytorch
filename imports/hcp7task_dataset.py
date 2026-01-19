import json
import os

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm


class HCP7TaskDataset(Dataset):
    """
    HCP 7-task Dataset (EV-anchored, unified length + time_mask)

    每个样本：
      - label = task_id
      - 以 EV onset 作为起点截取固定长度序列
      - 在 __getitem__ 中计算 ROI 平均信号与 FC 图
    """

    def __init__(
        self,
        root,
        subject_dict,
        task_config,
        atlas_path=None,
        roi_ids=None,
        tr_sec=0.72,
        fc_root=None,
        precomputed=True,
        task_name=None,
    ):
        self.root = root
        self.subject_list = (
            list(subject_dict.keys())
            if isinstance(subject_dict, dict)
            else list(subject_dict)
        )
        self.tr_sec = tr_sec
        self.task_config = self._load_task_config(task_config)
        self.task_name_list = self.task_config["task_name_list"]
        self.task_fixed_tr = self.task_config["task_fixed_tr"]
        self.task_label_rules = self.task_config["task_label_rules"]
        self.global_fixed_tr = self.task_config["global_fixed_tr"]
        self.precomputed = precomputed
        self.fc_root = fc_root
        self.task_name = task_name

        if not self.precomputed:
            if atlas_path is None:
                raise ValueError("atlas_path is required when precomputed is False")
            self.template_data = self.get_atlas_data(atlas_path)
            self.roi_ids = self._resolve_roi_ids(roi_ids)
        else:
            self.template_data = None
            self.roi_ids = None

        if self.precomputed:
            if self.fc_root is None:
                raise ValueError("fc_root is required when precomputed is True")
            self.samples = self._build_fc_index()
        else:
            self.samples = self._build_index()
        self.num_rois = self._infer_num_rois()

    def _load_task_config(self, task_config):
        if isinstance(task_config, str):
            with open(task_config, "r", encoding="utf-8") as f:
                return json.load(f)
        return task_config

    def _resolve_roi_ids(self, roi_ids):
        if roi_ids is None:
            roi_values = torch.unique(self.template_data.long())
            roi_values = roi_values[roi_values > 0]
            return roi_values.sort().values
        if isinstance(roi_ids, (list, tuple, np.ndarray)):
            return torch.tensor(roi_ids, dtype=torch.long)
        return roi_ids.long()

    # --------------------------------------------------
    # 构建样本索引（只存 metadata，不读数据）
    # --------------------------------------------------
    def _build_index(self):
        samples = []

        for subject in tqdm(self.subject_list, desc="Building HCP-7task index"):
            for task in self.task_name_list:
                if self.task_name and task != self.task_name:
                    continue
                task_dir = os.path.join(self.root, subject, task)
                if not os.path.exists(task_dir):
                    continue

                fixed_tr = self.task_fixed_tr[task]
                valid_evs = set(self.task_label_rules[task]["evs"])

                for run in ["LR", "RL"]:
                    run_dir = os.path.join(task_dir, run)
                    split_dir = os.path.join(run_dir, "Split_data")
                    evs_dir = os.path.join(run_dir, "EVs")

                    if not (os.path.exists(split_dir) and os.path.exists(evs_dir)):
                        continue

                    frame_files = sorted(
                        f
                        for f in os.listdir(split_dir)
                        if f.startswith("frame_") and f.endswith(".pt")
                    )
                    total_tr = len(frame_files)
                    if total_tr < fixed_tr:
                        continue

                    events = self._load_evs(evs_dir, valid_evs)
                    if len(events) == 0:
                        continue

                    for onset, duration, label in events:
                        start_tr = int(round(onset / self.tr_sec))
                        if start_tr < 0 or start_tr + fixed_tr > total_tr:
                            continue

                        samples.append(
                            {
                                "subject_path": split_dir,
                                "start_tr": start_tr,
                                "task": task,
                                "task_tr": fixed_tr,
                                "label": label,
                            }
                        )

        return samples

    def _build_fc_index(self):
        samples = []
        subject_set = set(self.subject_list)
        for subject in os.listdir(self.fc_root):
            if subject_set and subject not in subject_set:
                continue
            subject_dir = os.path.join(self.fc_root, subject)
            if not os.path.isdir(subject_dir):
                continue
            for task in self.task_name_list:
                if self.task_name and task != self.task_name:
                    continue
                task_dir = os.path.join(subject_dir, task)
                if not os.path.isdir(task_dir):
                    continue
                label_dirs = [
                    entry
                    for entry in os.listdir(task_dir)
                    if os.path.isdir(os.path.join(task_dir, entry))
                ]
                label_set = set(self.task_label_rules[task]["label_map"].keys())
                task_has_label_dirs = any(label in label_set for label in label_dirs)
                if task_has_label_dirs:
                    for label in label_dirs:
                        if label not in label_set:
                            continue
                        label_dir = os.path.join(task_dir, label)
                        fc_files = sorted(
                            f for f in os.listdir(label_dir) if f.endswith(".pt")
                        )
                        for fc_name in fc_files:
                            samples.append(
                                {
                                    "task": task,
                                    "subject": subject,
                                    "run": None,
                                    "label": label,
                                    "fc_path": os.path.join(label_dir, fc_name),
                                }
                            )
                    continue
                for run in os.listdir(task_dir):
                    run_dir = os.path.join(task_dir, run)
                    if not os.path.isdir(run_dir):
                        continue
                    label_dirs = [
                        entry
                        for entry in os.listdir(run_dir)
                        if os.path.isdir(os.path.join(run_dir, entry))
                    ]
                    if label_dirs:
                        for label in label_dirs:
                            label_dir = os.path.join(run_dir, label)
                            fc_files = sorted(
                                f for f in os.listdir(label_dir) if f.endswith(".pt")
                            )
                            for fc_name in fc_files:
                                samples.append(
                                    {
                                        "task": task,
                                        "subject": subject,
                                        "run": run,
                                        "label": label,
                                        "fc_path": os.path.join(label_dir, fc_name),
                                    }
                                )
                    else:
                        fc_files = sorted(
                            f for f in os.listdir(run_dir) if f.endswith(".pt")
                        )
                        for fc_name in fc_files:
                            samples.append(
                                {
                                    "task": task,
                                    "subject": subject,
                                    "run": run,
                                    "label": None,
                                    "fc_path": os.path.join(run_dir, fc_name),
                                }
                            )
        return samples

    # --------------------------------------------------
    # 读取 EV 文件（仅用于 onset）
    # --------------------------------------------------
    def _load_evs(self, evs_dir, valid_evs):
        events = []
        for fname in os.listdir(evs_dir):
            if not fname.endswith(".txt"):
                continue

            label = fname.replace(".txt", "")
            if label not in valid_evs:
                continue

            try:
                arr = np.loadtxt(os.path.join(evs_dir, fname))
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)

                for onset, duration, *_ in arr:
                    events.append((onset, duration, label))
            except Exception:
                continue

        return events

    # --------------------------------------------------
    # 加载 fMRI 序列（保留原命名逻辑）
    # --------------------------------------------------
    def _load_sequence(self, subject_path, start_tr, length):
        frames = []
        for i in range(start_tr, start_tr + length):
            frame = torch.load(os.path.join(subject_path, f"frame_{i}.pt"))
            frames.append(frame.unsqueeze(-1))
        x = torch.cat(frames, dim=-1)  # [D, H, W, T]
        return x.unsqueeze(0)  # [1, D, H, W, T]

    def get_average_roi_signal(self, fmri_data, roi_num):
        """
        fmri_data: [B, 1, 96, 96, 96, T]
        roi_num: [B, N] 或 [N]，每个样本选择的 ROI 编号（atlas中的编号）
        return: [B, N, T]，每个样本每个 ROI 的平均激活（z-score标准化）
        """
        batch_size, _, depth, height, width, time_len = fmri_data.shape
        roi_num = roi_num.view(1, -1) if roi_num.dim() == 1 else roi_num
        num_rois = roi_num.shape[1]

        device = fmri_data.device
        roi_num = roi_num.to(device)
        fmri_data = fmri_data.float()
        fmri_flat = fmri_data.view(batch_size, depth * height * width, time_len)
        atlas = self.template_data.to(device).view(-1).long()

        output = torch.zeros(batch_size, num_rois, time_len, device=device)

        roi_unique = torch.unique(roi_num)
        mask = torch.zeros_like(atlas, dtype=torch.bool)
        for roi in roi_unique:
            mask |= (atlas == roi)
        selected_voxels_idx = mask.nonzero(as_tuple=True)[0]

        atlas_selected = atlas[selected_voxels_idx]
        fmri_selected = fmri_flat[:, selected_voxels_idx, :]

        for n in range(num_rois):
            roi_val = roi_num[0, n].item()
            roi_mask = atlas_selected == roi_val
            if roi_mask.sum() == 0:
                output[:, n, :] = 0.0
                continue
            roi_voxels = fmri_selected[:, roi_mask, :]
            roi_mean = roi_voxels.mean(dim=1)

            mean_t = roi_mean.mean(dim=-1, keepdim=True)
            std_t = roi_mean.std(dim=-1, keepdim=True)
            std_t = torch.clamp(std_t, min=1e-6)
            output[:, n, :] = (roi_mean - mean_t) / std_t

        return output

    def get_atlas_data(self, path):
        atlas_img = nib.load(path)
        atlas_data = atlas_img.get_fdata()
        atlas_data = torch.tensor(atlas_data, dtype=torch.float32)
        atlas_data = atlas_data[9:82, 10:100, 5:78]
        atlas_data = torch.nn.functional.pad(atlas_data, (11, 12, 3, 3, 11, 12), value=0)
        return atlas_data

    def _compute_fc(self, roi_signal):
        time_len = roi_signal.shape[-1]
        if time_len < 2:
            return torch.zeros(
                roi_signal.shape[0], roi_signal.shape[0], device=roi_signal.device
            )
        fc = roi_signal @ roi_signal.transpose(0, 1)
        fc = fc / (time_len - 1)
        return torch.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)

    # --------------------------------------------------
    # Dataset API
    # --------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        if self.precomputed:
            fc = torch.load(item["fc_path"])
        else:
            fmri = self._load_sequence(
                subject_path=item["subject_path"],
                start_tr=item["start_tr"],
                length=item["task_tr"],
            )
            fmri = fmri.unsqueeze(0)  # [B, 1, D, H, W, T]
            roi_signal = self.get_average_roi_signal(fmri, self.roi_ids)
            fc = self._compute_fc(roi_signal[0])

        node_features = fc
        edge_weights = fc.abs().clone()
        edge_weights.fill_diagonal_(0.0)
        edge_index, edge_attr = dense_to_sparse(edge_weights)
        edge_attr = edge_attr.view(-1, 1)

        pos = torch.eye(fc.size(0), device=fc.device)
        task = item["task"]
        label_name = item.get("label")
        label_map = self.task_label_rules[task]["label_map"]
        if label_name is None:
            raise ValueError(
                f"Missing label for task '{task}' at {item.get('fc_path')}. "
                "Ensure FC files are stored under label directories."
            )
        label = label_map.get(label_name)
        if label is None:
            raise ValueError(
                f"Unknown label '{label_name}' for task '{task}'. "
                f"Valid labels: {sorted(label_map.keys())}"
            )

        return Data(
            x=node_features.float(),
            edge_index=edge_index.long(),
            edge_attr=edge_attr.float(),
            y=torch.tensor(label, dtype=torch.long),
            pos=pos.float(),
            task=task,
            subject=item.get("subject"),
            run=item.get("run"),
            block_label=item.get("label"),
        )

    def _infer_num_rois(self):
        if not self.samples:
            return 0
        if self.precomputed:
            sample_fc = torch.load(self.samples[0]["fc_path"])
            return int(sample_fc.shape[0])
        return int(self.roi_ids.numel())
