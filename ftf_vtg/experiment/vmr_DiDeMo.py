import os
import json
import numpy as np
import torch
from tqdm import tqdm
from itertools import product
import os
import json
import torch
import itertools
from ftf_vtg.main import FTF_VTG
from ftf_vtg.src.utils.eval_metrics import compute_iou

# 📁 평가 대상 JSON 경로
PRED_JSON_DIR = "/content/drive/MyDrive/순간포착/DiDeMo/DiDeMo_json_with_gt"



# 📌 단일 파일 평가
def evaluate_file(json_path, FTF_VTG = FTF_VTG, **param_dict):
    with open(json_path, "r") as f:
        data = json.load(f)

    if "gt_times" not in data or "num_segments" not in data:
        return None

    fps = data["fps"]
    total_frames = data["total_frames"]
    similarities = torch.tensor(data["similarities"])
    num_segments = data["num_segments"]
    gt_seg_indices = [tuple(gt) for gt in data["gt_times"]]

    try:
        pred_start, pred_end = FTF_VTG(similarities, **param_dict)
        if pred_start is None or pred_end is None:
            return None
    except:
        return None

    # 📌 모든 세그먼트 구간에 대해 IoU 계산
    iou_dict = {}
    for length in range(1, num_segments + 1):
        for start_seg in range(num_segments - length + 1):
            end_seg = start_seg + length - 1
            start_frame = int(start_seg * 5 * fps)
            end_frame = int(min((end_seg + 1) * 5 * fps, total_frames))
            iou = compute_iou(pred_start, pred_end, start_frame, end_frame)
            iou_dict[(start_seg, end_seg)] = iou

    sorted_segments = sorted(iou_dict.items(), key=lambda x: x[1], reverse=True)
    top1_seg = sorted_segments[0][0]
    top5_segs = [seg[0] for seg in sorted_segments[:5]]

    is_hit_r1 = int(top1_seg in gt_seg_indices)
    is_hit_r5 = int(any(seg in gt_seg_indices for seg in top5_segs))

    return is_hit_r1, is_hit_r5

# 📌 전체 평가 함수
def run_recall_evaluation(FTF_VTG, **param_dict):
    hits_r1, hits_r5 = [], []
    files = [f for f in os.listdir(PRED_JSON_DIR) if f.endswith(".json")]

    for fname in tqdm(files):
        path = os.path.join(PRED_JSON_DIR, fname)
        result = evaluate_file(path, FTF_VTG, **param_dict)
        if result is not None:
            r1, r5 = result
            hits_r1.append(r1)
            hits_r5.append(r5)

    recall1 = np.mean(hits_r1)
    recall5 = np.mean(hits_r5)
    print(f"\n📊 Final Results over {len(hits_r1)} examples:")
    print(f"Recall@1: {recall1:.4f}")
    print(f"Recall@5: {recall5:.4f}")
    return recall1, recall5

# 📌 그리드 서치 함수
def vmr_DiDeMo(FTF_VTG, param_grid):
    keys = list(param_grid.keys())
    value_combinations = list(product(*param_grid.values()))

    best_r1 = -1
    best_r5 = -1
    best_params = None

    for values in value_combinations:
        current_params = dict(zip(keys, values))
        print(f"\n🔍 Trying: {current_params}")
        r1, r5 = run_recall_evaluation(FTF_VTG, **current_params)

        if r1 > best_r1:
            best_r1 = r1
            best_r5 = r5
            best_params = current_params
            print("✅ New best!")

    print("\n🏁 Best Result:")
    print(f"Recall@1: {best_r1:.4f}, Recall@5: {best_r5:.4f}")
    print("Best Params:")
    for k, v in best_params.items():
        print(f"  ↳ {k}: {v}")