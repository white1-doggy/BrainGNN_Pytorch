import argparse
import json
import os

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm


def load_task_config(task_config_path):
    with open(task_config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_atlas_data(path):
    atlas_img = nib.load(path)
    atlas_data = atlas_img.get_fdata()
    atlas_data = torch.tensor(atlas_data, dtype=torch.float32)
    atlas_data = atlas_data[9:82, 10:100, 5:78]
    atlas_data = torch.nn.functional.pad(atlas_data, (11, 12, 3, 3, 11, 12), value=0)
    return atlas_data


def get_average_roi_signal(fmri_data, roi_ids, atlas_data):
    batch_size, _, depth, height, width, time_len = fmri_data.shape
    roi_num = roi_ids.view(1, -1)
    num_rois = roi_num.shape[1]

    device = fmri_data.device
    roi_num = roi_num.to(device)
    fmri_data = fmri_data.float()
    fmri_flat = fmri_data.view(batch_size, depth * height * width, time_len)
    atlas = atlas_data.to(device).view(-1).long()

    output = torch.zeros(batch_size, num_rois, time_len, device=device)

    roi_unique = torch.unique(roi_num)
    mask = torch.zeros_like(atlas, dtype=torch.bool)
    for roi in roi_unique:
        mask |= atlas == roi
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


def compute_fc(roi_signal):
    time_len = roi_signal.shape[-1]
    if time_len < 2:
        return torch.zeros(
            roi_signal.shape[0], roi_signal.shape[0], device=roi_signal.device
        )
    fc = roi_signal @ roi_signal.transpose(0, 1)
    fc = fc / (time_len - 1)
    return torch.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)


def load_sequence(subject_path, start_tr, length):
    frames = []
    for i in range(start_tr, start_tr + length):
        frame = torch.load(os.path.join(subject_path, f"frame_{i}.pt"))
        frames.append(frame.unsqueeze(-1))
    x = torch.cat(frames, dim=-1)
    return x.unsqueeze(0)


def load_evs(evs_dir, valid_evs):
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


def preprocess_subject(
    root,
    subject,
    task_name_list,
    task_label_rules,
    task_fixed_tr,
    tr_sec,
    atlas_data,
    roi_ids,
    fc_root,
    block_len,
):
    for task in task_name_list:
        task_dir = os.path.join(root, subject, task)
        if not os.path.exists(task_dir):
            continue

        valid_evs = set(task_label_rules[task]["evs"])
        fixed_tr = task_fixed_tr[task]
        out_dir = os.path.join(fc_root, task, subject)
        os.makedirs(out_dir, exist_ok=True)

        block_idx = 0
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

            events = load_evs(evs_dir, valid_evs)
            if not events:
                continue

            for onset, duration, _ in events:
                start_tr = int(round(onset / tr_sec))
                if start_tr < 0 or start_tr + block_len > total_tr:
                    continue

                fmri = load_sequence(split_dir, start_tr, block_len)
                fmri = fmri.unsqueeze(0)
                roi_signal = get_average_roi_signal(fmri, roi_ids, atlas_data)
                fc = compute_fc(roi_signal[0]).cpu()

                block_idx += 1
                fc_path = os.path.join(out_dir, f"fc{block_idx}.pt")
                torch.save(fc, fc_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True, help="HCP task root directory")
    parser.add_argument("--subject_list", required=True, help="subject list file")
    parser.add_argument("--task_config", required=True, help="task config json")
    parser.add_argument("--atlas_path", required=True, help="atlas nifti path")
    parser.add_argument("--fc_root", required=True, help="output FC root directory")
    parser.add_argument("--tr_sec", type=float, default=0.72)
    parser.add_argument("--block_len", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    task_config = load_task_config(args.task_config)
    task_name_list = task_config["task_name_list"]
    task_label_rules = task_config["task_label_rules"]
    task_fixed_tr = task_config["task_fixed_tr"]

    with open(args.subject_list, "r", encoding="utf-8") as f:
        subjects = [line.strip() for line in f.readlines() if line.strip()]

    atlas_data = get_atlas_data(args.atlas_path)
    roi_ids = torch.unique(atlas_data.long())
    roi_ids = roi_ids[roi_ids > 0].sort().values

    device = torch.device(args.device)
    atlas_data = atlas_data.to(device)
    roi_ids = roi_ids.to(device)

    for subject in tqdm(subjects, desc="Preprocessing subjects"):
        preprocess_subject(
            root=args.dataroot,
            subject=subject,
            task_name_list=task_name_list,
            task_label_rules=task_label_rules,
            task_fixed_tr=task_fixed_tr,
            tr_sec=args.tr_sec,
            atlas_data=atlas_data,
            roi_ids=roi_ids,
            fc_root=args.fc_root,
            block_len=args.block_len,
        )


if __name__ == "__main__":
    main()
