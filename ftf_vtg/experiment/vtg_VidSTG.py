import os
import json
import torch
import itertools
from ftf_vtg.main import FTF_VTG
from ftf_vtg.src.utils.eval_metrics import compute_iou

def vtg_VidSTG(json_dir, param_grid, FTF_VTG=FTF_VTG):
    arr_a_list = param_grid.pop("arr_a", [0.0])
    arr_b_list = param_grid.pop("arr_b", [0.0])
    combinations = list(itertools.product(arr_a_list, arr_b_list))
    hyper_grid = list(itertools.product(*[param_grid[k] for k in param_grid]))
    hyper_keys = list(param_grid.keys())

    best_miou = -1.0
    best_config = None

    for arr_a, arr_b in combinations:
        for hyper_values in hyper_grid:
            hyperparams = dict(zip(hyper_keys, hyper_values))

            total_iou = 0.0
            valid_count = 0

            for filename in os.listdir(json_dir):
                if not filename.endswith(".json"):
                    continue
                path = os.path.join(json_dir, filename)
                with open(path, "r") as f:
                    data = json.load(f)

                used_begin = data["used_segment"]["begin_fid"]
                used_end = data["used_segment"]["end_fid"]
                ground_begin = data["ground"]["begin_fid"]
                ground_end = data["ground"]["end_fid"]
                gt_start = ground_begin - used_begin
                gt_end = ground_end - used_begin

                tq = data["text_queries"]
                arr1 = tq["description"]["similarities"][used_begin:used_end + 1]
                combined_tensor = torch.tensor(arr1, dtype=torch.float32).unsqueeze(0)

                pred_start, pred_end = FTF_VTG(combined_tensor, **hyperparams)
                iou = compute_iou(pred_start, pred_end, gt_start, gt_end)

                total_iou += iou
                valid_count += 1

            mean_miou = total_iou / valid_count if valid_count > 0 else 0.0
            print(f"📌 조합 테스트: arr_a={arr_a}, arr_b={arr_b}, {hyperparams}, mIoU={mean_miou:.4f}")

            if mean_miou > best_miou:
                best_miou = mean_miou
                best_config = {"arr_a": arr_a, "arr_b": arr_b, **hyperparams}
                print(f"✅ [갱신] 최고 mIoU: {best_miou:.4f} @ {best_config}")

    return best_miou, best_config