import numpy as np

def compute_segment_score(similarity_array, start, end, lambda_weight=0.5):
    all_scores = similarity_array
    seg_scores = all_scores[start:end+1]

    mask = np.ones_like(all_scores, dtype=bool)
    mask[start:end+1] = False
    out_scores = all_scores[mask]

    a = seg_scores.mean() if len(seg_scores) > 0 else 1e-6
    b = out_scores.mean() if len(out_scores) > 0 else 1e-6
    score1 = a / b

    c = all_scores.mean()
    d = (seg_scores >= c).sum()
    e = (out_scores >= c).sum()
    score2 = (d / e) if e > 0 else 1.0

    final_score = lambda_weight * score1 + (1 - lambda_weight) * score2
    return final_score