import torch
import numpy as np


from pdvc.dp.exact_dp import drop_dtw, double_drop_dtw
from pdvc.dp.dp_utils import compute_sim
import statistics
from sklearn.cluster import KMeans
from pdvc.util import find_center_value, compute_overlap
# from config import CONFIG

''' configs of original file '''
config_eval_l2norm = True 
config_eval_keep_percentile = 0.48
config_eval_fixed_drop_sim = -1


''' 
return value:
frame features: [num_frames, feature_dim] -> optimal_assignment: [num_steps], -1 means no match, otherwise means the index of the matched step/caption/query

'''
# filter_threshold = 0.5

def clip_array(arr, threshold):
    clipped_arr = np.where(arr > threshold, arr, threshold)
    return clipped_arr



def compute_filtered_indices(topk_indices, topk_values, threshold=0.5):
    center_index = find_center_value(np.array(topk_indices))
    std_index = (sum((topk_indices - center_index) ** 2 * topk_values) / (sum(topk_values) + 1e-5)) ** 0.5
    boundary_width = std_index * threshold
    filtered_indices = [i for i in topk_indices if abs(i - center_index) <= boundary_width]
    return filtered_indices 

def compute_bbox_loss(index_list, box, similarity_values):
    left, right = box
    distances = []

    for i, index in enumerate(index_list):
        if left <= index <= right:
            distance = -min(index - left, right - index)
        else:
            distance = max(left - index, index - right)
        
        weighted_distance = similarity_values[i] * distance
        distances.append(weighted_distance)

    return sum(distances)





def remove_outliers(indices, threshold, mode, w):
    # Calculate the mean and standard deviation of the indices
    if mode == 'median':
        median = statistics.median(indices)
    elif mode == 'mean':
        mean = sum(indices) / len(indices)
    elif mode == 'mode':
        count_dict = {}
        for p in range(min(indices), max(indices) + 1):
            # print(p)
            count = sum(1 for c in indices if p - w <= c <= p + w)
            count_dict[p] = count

        max_count = max(count_dict.values())
        best_p_values = [p for p, count in count_dict.items() if count == max_count]
        if len(best_p_values) % 2 == 0:
            best_p_values.pop()
        
        mode_value = statistics.median(best_p_values)
    std_dev = (sum((x - mean) ** 2 for x in indices) / len(indices)) ** 0.5

    threshold_value = threshold * std_dev

    # Filter out indices that are far from the mean
    # breakpoint()

    if mode == 'median':
        filtered_indices = [i for i in indices if abs(i - median) <= threshold_value]
    elif mode == 'mode':
        filtered_indices = [i for i in indices if abs(i - mode_value) <= threshold_value]
    return filtered_indices


def remove_outliers_v1(indices, threshold):
    pass 

def get_mode(indices, w):
    count_dict = {}
    for p in range(min(indices), max(indices) + 1):
        # print(p)
        count = sum(1 for c in indices if p - w <= c <= p + w)
        count_dict[p] = count

    max_count = max(count_dict.values())
    best_p_values = [p for p, count in count_dict.items() if count == max_count]
    if len(best_p_values) % 2 == 0:
        best_p_values.pop()
    
    mode_value = statistics.median(best_p_values)
    return mode_value

def get_mode_box(sim, topk, w, ratio): # topk选择20 ratio 1 
    ''' 注意这里算中心的时候使用前topk是因为更相信前topk的准确率 但是确定中心以后需要找边界 就需要使用全部的'''
    avg_caption_length = sim.shape[1] // sim.shape[0]
    sorted_idx = torch.argsort(-sim, dim=1)
    top_indices = sorted_idx[:, :topk]
    # top_values, top_indices = torch.topk(sim, topk, dim=1, largest=True, sorted=True)
    # top_indices_half = top_indices[:, :topk//2]
    top_cap_indices = sorted_idx[:, :avg_caption_length]
    # sorted_idx = torch.argsort(-sim, dim=1)
    width = int(ratio * avg_caption_length / 2) # ratio选择1
    
    bbox = []
    for i in range(top_indices.shape[0]):
        # index_list = top_indices[i].tolist()
        mode_value = get_mode(top_indices[i].tolist(), w)
        filtered_indices = [i for i in top_cap_indices[i].tolist() if abs(i - mode_value) <= width]

        # if len(filtered_indices) == 0:
        #     filtered_indices = remove_outliers(sim[i].tolist(), top_indices[i].tolist(), 0.5, mode='median', w=w)
        #     if len(filtered_indices) == 0:
        #         bbox.append([0, sim.shape[1] - 1])
        #         continue
        if len(filtered_indices) == 0:
            bbox.append([mode_value-width, mode_value+width])
        else:
            bbox.append([min(filtered_indices), max(filtered_indices)])
    return bbox

def compute_threshold(data, threshold):
    mean = sum(data) / len(data)
    std_dev = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
    threshold_value = threshold * std_dev
    return threshold_value



def uniform_box(frame_features, step_features, topk=15, threshold=0.5, w=2, mode='median'):
    if step_features.shape[0] == 0:
        return -np.ones(frame_features.shape[0])
    
    uniform_boxes = uniform_window(frame_features.shape[0], step_features.shape[0])
    return uniform_boxes
    



# based on original code but change the method to compute center and std
def align_frame_into_steps_op_order_v2(frame_features, step_features, topk=15, threshold=0.5, ratio=1, iteration=3):
    # breakpoint()
    if step_features.shape[0] == 0:
        return -np.ones(frame_features.shape[0])
    
    sim = compute_sim(step_features, frame_features, config_eval_l2norm).cpu()
    sorted_index = torch.argsort(-sim, dim=1)
    top_indices_list_global = [sorted_index[i][:topk] for i in range(sim.shape[0])]
    top_values_list_global = [sim[i][top_indices_list_global[i]] for i in range(sim.shape[0])]


    uniform_boxes = uniform_window(frame_features.shape[0], step_features.shape[0])

    iter_bbox_loss = {}
    for iter in range(iteration):
        if iter == 0:
            refined_uniform_boxes = expand_window(uniform_boxes, frame_features.shape[0], step_features.shape[0], ratio)
        else:
            refined_uniform_boxes = expand_window(bbox, frame_features.shape[0], step_features.shape[0], ratio) # last bbox


        # global: from all frames, local: from refined uniform boxes
    
        top_indices_list_local = [sorted_index[i][(sorted_index[i] >= refined_uniform_boxes[i][0]) & (sorted_index[i] <= refined_uniform_boxes[i][1])][:topk] for i in range(sim.shape[0])]
        top_values_list_local = [sim[i][top_indices_list_local[i]] for i in range(sim.shape[0])]

        size_local = [len(top_indices_list_local[i]) for i in range(sim.shape[0])]
        if sum(size_local) < (topk-2) * len(size_local):
            top_indices_list = top_indices_list_global
            top_values_list = top_values_list_global
        else:
            top_indices_list = top_indices_list_local
            top_values_list = top_values_list_local

        # top_indices_list = [top_indices_list_global[i] if len(top_indices_list_local[i]) < topk else top_indices_list_local[i] for i in range(sim.shape[0])]

        bbox = []
        for i in range(len(top_indices_list)):
            filtered_indices = compute_filtered_indices(top_indices_list[i].tolist(), top_values_list[i].tolist(), threshold)
            if len(filtered_indices) == 0:
                filtered_indices = compute_filtered_indices(top_indices_list_global[i].tolist(), top_indices_list_global[i].tolist(), threshold)
                if len(filtered_indices) == 0:
                    bbox.append(uniform_boxes[i])
                    continue
            bbox.append([min(filtered_indices), max(filtered_indices)])

        # compute bbox loss
        bbox_loss_list = [compute_bbox_loss(top_indices_list[i], bbox[i], top_values_list[i]) for i in range(len(top_indices_list))]
        bbox_loss = sum(bbox_loss_list)
        iter_bbox_loss[iter] = {'loss': bbox_loss, 'bbox': bbox}

    # select the minimum bbox loss and bbox as output
    min_loss_iter = min(iter_bbox_loss.keys(), key=lambda k: iter_bbox_loss[k]['loss'])
    min_loss = iter_bbox_loss[min_loss_iter]['loss']
    best_bbox = iter_bbox_loss[min_loss_iter]['bbox']
        

    return (best_bbox, min_loss)

def align_frame_into_steps_op_v2(frame_features, step_features, topk=15, threshold=0.5, ratio=1, iteration=3):
    # breakpoint()
    if step_features.shape[0] == 0:
        return -np.ones(frame_features.shape[0])
    
    sim = compute_sim(step_features, frame_features, config_eval_l2norm).cpu()
    sorted_index = torch.argsort(-sim, dim=1)
    top_indices_list_global = [sorted_index[i][:topk] for i in range(sim.shape[0])]
    top_values_list_global = [sim[i][top_indices_list_global[i]] for i in range(sim.shape[0])]


    uniform_boxes = uniform_window(frame_features.shape[0], step_features.shape[0])

    iter_bbox_loss = {}
    for iter in range(iteration):
        bbox = []
        for i in range(len(top_indices_list_global)):
            filtered_indices = compute_filtered_indices(top_indices_list_global[i].tolist(), top_values_list_global[i].tolist(), threshold)
            if len(filtered_indices) == 0:
                filtered_indices = compute_filtered_indices(top_indices_list_global[i].tolist(), top_indices_list_global[i].tolist(), threshold)
                if len(filtered_indices) == 0:
                    bbox.append(uniform_boxes[i])
                    continue
            bbox.append([min(filtered_indices), max(filtered_indices)])

        # compute bbox loss
        bbox_loss_list = [compute_bbox_loss(top_indices_list_global[i], bbox[i], top_values_list_global[i]) for i in range(len(top_indices_list_global))]
        bbox_loss = sum(bbox_loss_list)
        iter_bbox_loss[iter] = {'loss': bbox_loss, 'bbox': bbox}

    # select the minimum bbox loss and bbox as output
    min_loss_iter = min(iter_bbox_loss.keys(), key=lambda k: iter_bbox_loss[k]['loss'])
    min_loss = iter_bbox_loss[min_loss_iter]['loss']
    best_bbox = iter_bbox_loss[min_loss_iter]['bbox']
        

    return (best_bbox, min_loss)


def uniform_window(frame_num, step_num):
    uniform_timestamps = torch.linspace(0, frame_num, step_num + 1)
    uniform_timestamps = torch.round(uniform_timestamps).int().tolist()
    bbox = []
    for i in range(step_num):
        bbox.append([uniform_timestamps[i], uniform_timestamps[i+1] - 1])

    # window_size = frame_num // step_num
    # bbox = []
    # for i in range(step_num):
    #     bbox.append([i * window_size, (i + 1) * window_size - 1])
    # bbox[-1][1] = frame_num - 1
    return bbox 

def expand_window(uniform_bbox, frame_num, step_num, ratio=1):
    '''ratio: gt box相对uniform box的波动范围 超过这个范围视为不可能 ratio单位为一个caption的平均长度'''
    window_size = frame_num // step_num
    refined_bbox = []
    for bbox in uniform_bbox:
        start = max(0, bbox[0] - ratio * window_size)
        end = min(frame_num - 1, bbox[1] + ratio * window_size)
        refined_bbox.append([start, end])
    return refined_bbox



# pesudo box 0: based on dtw
def segment_video_into_steps(frame_features, step_features, unordered=False):
    if step_features.shape[0] == 0:
        return -np.ones(frame_features.shape[0])

    sim = compute_sim(step_features, frame_features, config_eval_l2norm).cpu()
    frame_features, step_features = frame_features.cpu(), step_features.cpu()

    k = max([1, int(torch.numel(sim) * config_eval_keep_percentile)])
    baseline_logit = torch.topk(sim.reshape([-1]), k).values[-1].detach()
    baseline_logits = baseline_logit.repeat([1, sim.shape[1]])[0]  # making it of shape [1, N]
    zx_costs, drop_costs = -sim, -baseline_logits # base其实是从相似度矩阵中选择了一个中间值作为drop cost 这个中间值就是你认为匹配也可以 drop也可以的那个值
    zx_costs, drop_costs = [t.detach().cpu().numpy() for t in [zx_costs, drop_costs]]
    sim = sim.detach().cpu().numpy()

    if unordered:
        max_vals, optimal_assignment = np.max(sim, axis=0), np.argmax(sim, axis=0)  # 直接找与每个step最匹配的frame 这样原则上是一对一匹配
        optimal_assignment[max_vals < baseline_logit.item()] = -1
    else:
        optimal_assignment = drop_dtw(zx_costs, drop_costs, return_labels=True) - 1 # 调节drop cost的大小 从而调节匹配的严格程度
    return optimal_assignment

def align_query_into_steps(query_features, step_features, unordered=False):
    if step_features.shape[0] == 0:
        return -np.ones(query_features.shape[0])

    sim = compute_sim(step_features, query_features, config_eval_l2norm).cpu()
    query_features, step_features = query_features.cpu(), step_features.cpu()

    k = max([1, int(torch.numel(sim) * config_eval_keep_percentile)])
    baseline_logit = torch.topk(sim.reshape([-1]), k).values[-1].detach()
    baseline_logits = baseline_logit.repeat([1, sim.shape[1]])[0]  # making it of shape [1, N]
    zx_costs, drop_costs = -sim, -baseline_logits # base其实是从相似度矩阵中选择了一个中间值作为drop cost 这个中间值就是你认为匹配也可以 drop也可以的那个值
    zx_costs, drop_costs = [t.detach().cpu().numpy() for t in [zx_costs, drop_costs]]
    sim = sim.detach().cpu().numpy()

    if unordered:
        max_vals, optimal_assignment = np.max(sim, axis=0), np.argmax(sim, axis=0)  # 直接找与每个step最匹配的frame 这样原则上是一对一匹配
        optimal_assignment[max_vals < baseline_logit.item()] = -1
    else:
        optimal_assignment = drop_dtw(zx_costs, drop_costs, one_to_one=True, return_labels=True) - 1 # 调节drop cost的大小 从而调节匹配的严格程度
    return optimal_assignment

# inference时 video和slots之间的匹配
def segment_video_into_slots(video_features, pred_steps):
    sim = compute_sim(pred_steps, video_features, l2_norm=config_eval_l2norm).detach()
    if config_eval_fixed_drop_sim == -1:
        k = max([1, int(torch.numel(sim) * config_eval_keep_percentile)])
        baseline_logit = torch.topk(sim.reshape([-1]), k).values[-1].detach()
    else:
        baseline_logit = torch.tensor(config_eval_fixed_drop_sim)
    baseline_logits = baseline_logit.repeat([1, sim.shape[1]])  # making it of shape [1, N]
    x_drop_costs = -baseline_logits.squeeze()
    zx_costs = -sim

    z_drop_costs = -baseline_logit.repeat([1, sim.shape[0]]).squeeze()
    zx_costs = zx_costs - z_drop_costs[0].reshape([1, 1])
    z_drop_costs = z_drop_costs - z_drop_costs[0]
    x_drop_costs = x_drop_costs - x_drop_costs[0]
    segmentation = double_drop_dtw(zx_costs.numpy(), x_drop_costs.numpy(), z_drop_costs.numpy(), return_labels=True) - 1
    return segmentation


# get_index and alignment_to_boundary are used for 'align' based manner
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