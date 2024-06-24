import torch
import numpy as np
import statistics

from pdvc.dp.exact_dp import drop_dtw
from pdvc.dp.dp_utils import compute_sim
import statistics
from sklearn.cluster import KMeans


config_eval_l2norm = True 
config_eval_keep_percentile = 0.48 # Calculated from the data
config_eval_fixed_drop_sim = -1 

def segment_video_into_steps(frame_features, step_features, unordered=False):
    if step_features.shape[0] == 0:
        return -np.ones(frame_features.shape[0])

    sim = compute_sim(step_features, frame_features, l2_norm=True).cpu()
    frame_features, step_features = frame_features.cpu(), step_features.cpu()

    k = max([1, int(torch.numel(sim) * config_eval_keep_percentile)])
    baseline_logit = torch.topk(sim.reshape([-1]), k).values[-1].detach()
    baseline_logits = baseline_logit.repeat([1, sim.shape[1]])[0]  # making it of shape [1, N]
    zx_costs, drop_costs = -sim, -baseline_logits
    zx_costs, drop_costs = [t.detach().cpu().numpy() for t in [zx_costs, drop_costs]]
    sim = sim.detach().cpu().numpy()

    if unordered:
        max_vals, optimal_assignment = np.max(sim, axis=0), np.argmax(sim, axis=0)
        optimal_assignment[max_vals < baseline_logit.item()] = -1
    else:
        optimal_assignment = drop_dtw(zx_costs, drop_costs, return_labels=True) - 1
    return optimal_assignment # [num_frames]

def get_index(alignment):
    start_idx, end_idx = [], []
    for i in range(len(alignment)):
        if alignment[i] == -1:
            if i != 0 and alignment[i-1] != -1:
                end_idx.append(i-1)
            continue
        if i == 0:
            start_idx.append(i)
        elif alignment[i] != alignment[i-1]:
            start_idx.append(i)
            if alignment[i-1] != -1:
                end_idx.append(i-1)
        if i == len(alignment) - 1:
            end_idx.append(i)
    assert len(start_idx) == len(end_idx)
    for s, e in zip(start_idx, end_idx):
        assert alignment[s] <= alignment[e]
    return start_idx, end_idx

def get_index_update(alignment):
    optimal_alignment = np.append(np.insert(alignment, 0, -1), -1)
    diff_optimal_alignment = np.diff(optimal_alignment)

    optimal_alignment_end = optimal_alignment.copy()
    optimal_alignment_end[optimal_alignment_end==-1] = max(optimal_alignment_end) + 1
    diff_optimal_alignment_end = np.diff(optimal_alignment_end)

    start_idx = np.where(diff_optimal_alignment>0)[0]
    end_idx = np.where(diff_optimal_alignment_end>0)[0] - 1
    return start_idx, end_idx

def alignment_to_boundary(alignment, video_frame_num):
    start_idx, end_idx = get_index(alignment)
    start_time = start_idx / video_frame_num
    end_time = end_idx / video_frame_num
    boundaries = list(zip(start_time, end_time))

    return np.float32(np.stack(boundaries, axis=0))


def to_center_duration(alignments):
    new_alignments = []
    for alignment in alignments:
        start, end = alignment[:, 0], alignment[:, 1]
        center = (start + end) / 2
        duration = end - start
        alignment[:, 0], alignment[:, 1] = center, duration
        new_alignments.append(alignment)
    return new_alignments


def remove_outliers(indices, threshold):
    # Calculate the mean and standard deviation of the indices
    median = statistics.median(indices)
    mean = sum(indices) / len(indices)
    std_dev = (sum((x - mean) ** 2 for x in indices) / len(indices)) ** 0.5

    # Calculate the threshold for identifying outliers
    threshold_value = threshold * std_dev

    # Filter out indices that are far from the mean
    filtered_indices = [i for i in indices if abs(i - median) <= threshold_value]

    return filtered_indices


def align_frame_into_steps(frame_features, step_features, unordered=False, k=15, threshold=0.5):
    if step_features.shape[0] == 0:
        return -np.ones(frame_features.shape[0])
    
    sim = compute_sim(step_features, frame_features, True).cpu()
    frame_features, step_features = frame_features.cpu(), step_features.cpu()

    top_values, top_indices = torch.topk(sim, k, dim=1, largest=True, sorted=True)
    bbox = []
    for i in range(top_indices.shape[0]):
        filtered_indices = remove_outliers(top_indices[i].tolist(), threshold)
        bbox.append([min(filtered_indices), max(filtered_indices)])
    return bbox

if __name__ == '__main__':
    # frame_features = torch.randn(100, 768)
    # text_features = torch.randn(8, 768)
    # alignment = segment_video_into_steps(frame_features, text_features)
    # breakpoint()
    arr = [-1,-1,0,1,2,2,2,-1,-1,3,4,4,-1,-1,5,5,5,-1,6,6,7,-1,-1, 8, 8, 9]
    start, end = get_index(arr)
    start_1, end_1 = get_index_update(arr)
    # start = [2, 3, 4, 8, 9, 13, 16, 18]
    # end = [2, 3, 5, 8, 10, 15, 17, 18]
    breakpoint()
