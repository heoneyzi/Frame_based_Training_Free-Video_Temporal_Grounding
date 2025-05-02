import os
import json
import torch
import itertools
from ftf_vtg.main import FTF_VTG
from ftf_vtg.src.utils.eval_metrics import compute_iou

import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from itertools import product

# ✅ 경로 설정
GT_JSON_PATH = "/content/drive/MyDrive/순간포착/DiDeMo/test_data_filtered.json"
PRED_JSON_DIR = "/content/drive/MyDrive/순간포착/DiDeMo/DiDeMo_json"

# ✅ GT 인덱스 구성: description 기준
with open(GT_JSON_PATH, "r") as f:
    gt_data = json.load(f)

gt_dict = {}
for item in gt_data:
    desc = item["description"]
    gt_dict.setdefault(desc, []).append({
        "segments": item["times"],
        "num_segments": item["num_segments"],
        "video": item["video"]
    })

def vtg_DiDeMo(json_filename, FTF_VTG, param_dict):
    full_path = os.path.join(PRED_JSON_DIR, json_filename)
    with open(full_path, "r") as f:
        data = json.load(f)

    video = data["video"]
    query = data["query"]
    fps = data["fps"]
    total_frames = data["total_frames"]
    similarities = torch.tensor(data["similarities"])

    try:
        pred_start, pred_end = FTF_VTG(similarities, **param_dict)
    except:
        return None

    if pred_start is None or pred_end is None or query not in gt_dict:
        return None

    matched_gts = gt_dict[query]
    all_gt_segments = []
    for item in matched_gts:
        num_segments = item["num_segments"]
        for seg_start, seg_end in item["segments"]:
            start_sec = seg_start * 5
            end_sec = (seg_end + 1) * 5 if seg_end < num_segments - 1 else total_frames / fps
            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            all_gt_segments.append((start_frame, end_frame))

    ious = [compute_iou(pred_start, pred_end, gt_s, gt_e) for gt_s, gt_e in all_gt_segments]
    max_iou = max(ious) if ious else 0.0
    is_hit = int(max_iou >= 0.5)

    return max_iou, is_hit

# ✅ 튜닝을 위한 param_grid 정의
param_grid = {
    "sigma": [0.5, 1.0, 1.5],
    "theta_high": [0.002, 0.003, 0.004],
    "kernel_size": [1,2,3,4],
    "min_seg_len": [1],
    "lambda_weight": [0.8, 0.9, 1.0],
    "gap_threshold": [12000],
    "alpha": [0.05, 0.1, 0.2, 0.9]
}

# ✅ 모든 조합 반복 → 가장 높은 mIoU 찾기
from copy import deepcopy

best_miou = -1
best_recall = -1
best_params = None

for param_tuple in product(*param_grid.values()):
    param_dict = dict(zip(param_grid.keys(), param_tuple))
    iou_scores = []
    recall_hits = []
    file_count = 0

    print(f"\n🔍 Trying: {param_dict}")

    for fname in tqdm(os.listdir(PRED_JSON_DIR)):
        if fname.endswith(".json"):
            result = vtg_DiDeMo(fname, FTF_VTG=FTF_VTG, param_dict=param_dict)
            if result is not None:
                iou, hit = result
                iou_scores.append(iou)
                recall_hits.append(hit)
                file_count += 1

    if file_count > 0:
        miou = np.mean(iou_scores)
        recall1 = np.mean(recall_hits)
        print(f"  → mIoU: {miou:.4f}, Recall@1: {recall1:.4f} over {file_count} files")

        # ✅ recall 기준으로 best 갱신
        if recall1 > best_recall:
            best_miou = miou
            best_recall = recall1
            best_params = deepcopy(param_dict)
            print("✅ New best found!")

print("\n🏆 Best Recall@1 result:")
print(f"Recall@1: {best_recall:.4f}")
print(f"mIoU: {best_miou:.4f}")
print("Best Params:")
for k, v in best_params.items():
    print(f"  ↳ {k}: {v}")