from .segment_detector import detect_event_segments_with_gradient
from .segment_merger import generate_all_merge_candidates_by_gap
from .segment_scorer import compute_segment_score
import torch
import numpy as np

def FTF_VTG(
    similarity_array,
    theta_high=0.1,
    lambda_weight=0.5
):
    segments, _, _ = detect_event_segments_with_gradient(
        similarity_array,
        kernel_size=3,
        sigma=1,
        theta_high=theta_high,
        theta_low_ratio=0.5,
        min_seg_len=1,
        alpha=0.5
    )
    merged_segments = generate_all_merge_candidates_by_gap(
        segments,
        gap_threshold=100
    )
    scored = []
    if isinstance(similarity_array, torch.Tensor):
        similarity_array = similarity_array.squeeze().cpu().numpy()
    for (s, e) in merged_segments:
        score = compute_segment_score(similarity_array, s, e, lambda_weight=lambda_weight)
        scored.append((s, e, score))

    if scored:
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[0][0], scored[0][1]
    else:
        return None, None