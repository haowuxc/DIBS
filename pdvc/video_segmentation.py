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


# def compute_filtered_indices(topk_indices_list, topk_values_list, scale=0.5):
#     # center_indices = []
#     # boundary_widths = []
#     filtered_indices_list = []
#     for topk_indices, topk_values in zip(topk_indices_list, topk_values_list):
#         center_index = find_center_value(topk_indices)
#         std_index = (sum((topk_indices - center_index) ** 2 * topk_values) / sum(topk_values)) ** 0.5
#         boundary_width = std_index * scale
#         filtered_indices = [i for i in topk_indices if abs(i - center_index) <= boundary_width]
#         filtered_indices_list.append(filtered_indices)
#         # center_indices.append(center_index)
#         # boundary_widths.append(boundary_width)

#     return filtered_indices_list

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

    # if mode == 'mode':
    #     '''get mode-similar statistics'''
    #     count_dict = {}
    #     for p in range(min(indices), max(indices) + 1):
    #         # print(p)
    #         count = sum(1 for c in indices if p - w <= c <= p + w)
    #         count_dict[p] = count

    #     max_count = max(count_dict.values())
    #     best_p_values = [p for p, count in count_dict.items() if count == max_count]
    #     if len(best_p_values) % 2 == 0:
    #         best_p_values.pop()
        
    #     mode_value = statistics.median(best_p_values)

    # Calculate the threshold for identifying outliers
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


# using similarity as weight to find center
''' find center globally, then find the boundary locally. 
    1. find center: use the similarity as weight to find the center
    2. find boundary: use the center to find the boundary. steps are '''
def step_retrieval_weight_sim(frame_features, step_features, topk=15, threshold=0.5, w=2):
    # breakpoint()
    if step_features.shape[0] == 0:
        return -np.ones(frame_features.shape[0])
    
    similarity_matrix = compute_sim(step_features, frame_features, config_eval_l2norm).cpu()
    # sim sum along a window
    window_sums = torch.nn.functional.conv1d(similarity_matrix.unsqueeze(1), torch.ones(1, 1, 2 * w + 1)).squeeze()

    if len(window_sums.shape) == 1:
        window_sums = window_sums.unsqueeze(0)
        flag = 1
    else:
        flag = 0 

    top_values, top_indices = torch.topk(window_sums, topk, dim=1, largest=True, sorted=True)
    # breakpoint()

    # Find the frame with the maximum sum in each step
    _, step_center_frames = window_sums.max(dim=1)
    step_center_frames = step_center_frames.squeeze()

    if flag == 1:
        step_center_frames = step_center_frames.unsqueeze(0).tolist()
    else:
        step_center_frames = step_center_frames.tolist()

    bbox = []
    for i in range(top_indices.shape[0]):
        threshold_value = compute_threshold(top_indices[i].tolist(), threshold)
        filtered_indices = [frame for frame in top_indices[i].tolist() if abs(frame - step_center_frames[i]) <= threshold_value]
        if len(filtered_indices) == 0:
            bbox.append([step_center_frames[i] - w, step_center_frames[i] + w])
        else:
            bbox.append([w + min(filtered_indices), w + max(filtered_indices)])
    
    return bbox

''' TODO: get the right weight using index'''
def step_retrieval_weight_index(frame_features, step_features, topk=15, threshold=0.5, w=2):
    # breakpoint()
    if step_features.shape[0] == 0:
        return -np.ones(frame_features.shape[0])
    
    similarity_matrix = compute_sim(step_features, frame_features, config_eval_l2norm).cpu()
    sorted_idx = torch.argsort(-similarity_matrix, dim=1)
    # sim sum along a window
    window_sums = torch.nn.functional.conv1d(similarity_matrix.unsqueeze(1), torch.ones(1, 1, 2 * w + 1)).squeeze()

    top_values, top_indices = torch.topk(window_sums, topk, dim=1, largest=True, sorted=True)
    # breakpoint()

    # Find the frame with the maximum sum in each step
    _, step_center_frames = window_sums.max(dim=1)
    step_center_frames = step_center_frames.squeeze().tolist()

    bbox = []
    for i in range(top_indices.shape[0]):
        threshold_value = compute_threshold(top_indices[i].tolist(), threshold)
        filtered_indices = [frame for frame in top_indices[i].tolist() if abs(frame - step_center_frames[i]) <= threshold_value]
        bbox.append([w + min(filtered_indices), w + max(filtered_indices)])
    
    return bbox

def uniform_box(frame_features, step_features, topk=15, threshold=0.5, w=2, mode='median'):
    if step_features.shape[0] == 0:
        return -np.ones(frame_features.shape[0])
    
    uniform_boxes = uniform_window(frame_features.shape[0], step_features.shape[0])
    return uniform_boxes
    

def align_frame_into_steps(frame_features, step_features, topk=15, threshold=0.5, w=2, mode='median'):
    # breakpoint()
    if step_features.shape[0] == 0:
        return -np.ones(frame_features.shape[0])
    
    sim = compute_sim(step_features, frame_features, config_eval_l2norm).cpu()
    average_width = int(sim.shape[1] // sim.shape[0] / 2)
    # frame_features, step_features = frame_features.cpu(), step_features.cpu()
    # bbox = get_mode_box(sim, topk, w, ratio)

    top_values, top_indices = torch.topk(sim, topk, dim=1, largest=True, sorted=True)
    bbox = []
    for i in range(top_indices.shape[0]):
        filtered_indices = remove_outliers(top_indices[i].tolist(), threshold, mode=mode, w=w)
        if len(filtered_indices) < 2:
            filtered_indices = remove_outliers(top_indices[i].tolist(), 2*threshold, mode=mode, w=w)
            if len(filtered_indices) == 0:
                bbox.append([top_indices[0] - average_width, top_indices[0] + average_width])
                continue
        bbox.append([min(filtered_indices), max(filtered_indices)])
    return bbox

# use optimization to compute pseudo boundary
def align_frame_into_steps_op(frame_features, step_features, topk=15, num_iterations=4, beta=1, order=False, scale=1):
    # frame_features:  torch.Size([200, 768])
    augment_ratio_list = np.arange(0.5, 2, 0.1)

    if step_features.shape[0] == 0:
        return -np.ones(frame_features.shape[0])
    
    # breakpoint()
    # [#step, #frame]
    similarity_matrix = compute_sim(step_features, frame_features, config_eval_l2norm).cpu().numpy()

    num_steps, num_frames = similarity_matrix.shape

    # Select top-k frames for each caption [#step, #topk]
    sorted_indices = np.argsort(similarity_matrix, axis=1)
    # top_indices = np.argsort(similarity_matrix, axis=1)[:, -topk:]
    # top_values = np.take_along_axis(similarity_matrix, top_indices, axis=1)

    # Compute center indexes [#step, 1]
    

    # Update boundary width 
    initial_boundary_width = num_frames / num_steps # 1
    # boundary_width = initial_boundary_width * np.ones(num_steps, 1, 1) # 1 
    # overlap = np.zeros(num_steps)

    for i in range(num_iterations):
        if i == 0 and not order:
            boundary_width_last = np.full(num_steps, initial_boundary_width).reshape(-1, 1, 1)
            topk_indices = [index[-topk:] for index in sorted_indices]
            topk_values = [similarity_matrix[i][index] for i, index in enumerate(topk_indices)]


            center_indexes = np.array([find_center_value(index) for index in topk_indices]) 
            previous_index_center = None
        #     # overlap_weight = 0
        else:
            if i == 0:
                segment_boundary = np.linspace(0, num_frames, num_steps + 1).round().astype(int)
                start_indices, end_indices = segment_boundary[:-1], segment_boundary[1:]
                start_indices = np.clip(start_indices - initial_boundary_width * scale, 0, num_frames)
                end_indices = np.clip(end_indices + initial_boundary_width * scale, 0, num_frames)
                boundary_width_last = (end_indices - start_indices).reshape(-1, 1, 1)

                filtered_indices = [sorted_indices[i][(sorted_indices[i] >= start_indices[i]) & (sorted_indices[i] <= end_indices[i])] for i in range(num_steps)]
                if sum(len(index) for index in filtered_indices) < topk * num_steps * 0.4:
                    boundary_width_last = np.full(num_steps, initial_boundary_width).reshape(-1, 1, 1)
                    topk_indices = [index[-topk:] for index in sorted_indices]
                    topk_values = [similarity_matrix[i][index] for i, index in enumerate(topk_indices)]


                    center_indexes = np.array([find_center_value(index) for index in topk_indices]) 
                    previous_index_center = None
            else:
                boundary_width_last = boundary_width.reshape(-1, 1, 1)
                start_indices = np.clip(center_indexes - boundary_width // 2 - initial_boundary_width * scale, 0, num_frames)
                end_indices = np.clip(center_indexes + boundary_width // 2 + initial_boundary_width * scale, 0, num_frames)

            topk_indices = []
            topk_values = []
            for j, (start, end) in enumerate(zip(start_indices, end_indices)):
                # breakpoint()
                filtered_indices = sorted_indices[j][(sorted_indices[j] >= start) & (sorted_indices[j] <= end)]
                topk_index = filtered_indices[-topk:]
                topk_indices.append(topk_index)
                topk_values.append(similarity_matrix[j][topk_index])
            previous_index_center = center_indexes.copy() if i > 0 else None
            center_indexes = np.array([find_center_value(index) for index in topk_indices]) 

            # top_indices = sorted_indices[:, ]
            # previous_index_center = center_indexes
            # # overlap_weight = 0.5 * np.sum(overlap)
        
        boundary_width_candidates = augment_ratio_list * boundary_width_last # [#steps, 1, #candidates]
        # breakpoint()

        index_distance = [np.abs(index - center_indexes[i] + 1e-3)[:, np.newaxis] for i, index in enumerate(topk_indices)] # [[topk, 1]]

        loss_candidates_list = [value[:, np.newaxis] * (np.abs(index_distance[i] - 0.5 * boundary_width_candidates[i])) for i, value in enumerate(topk_values)] # [[topk, candidates]]
        # loss_candidates_list = [value[:, np.newaxis] / index_distance[i] * (np.abs(index_distance[i] - 0.5 * boundary_width_candidates[i])) for i, value in enumerate(topk_values)] # [[topk, candidates]]


        # index_distance = np.abs(topk_indices - center_indexes)[:, :, np.newaxis] # [#step, #topk, 1]

        # loss_sim = np.sum(top_values[:, :, np.newaxis] / index_distance * (np.abs(index_distance - 0.5 * boundary_width_candidates)), axis=1) # [#step, #candidates]
        loss_sim = np.array([np.mean(loss, axis=0) for loss in loss_candidates_list]) # [#step, #candidates]

        if i == 0:            
            loss = loss_sim
            # print('loss shape:', loss_sim.shape, loss.shape)
        else:
            # measure the overlap between boundaries given center and boundary width
            overlap = compute_overlap(center_indexes, boundary_width_candidates, previous_index_center, boundary_width_last) # [#step, #candidates]
            # breakpoint()
            # print(loss_sim.shape, overlap.shape)
            loss = loss_sim + beta * overlap 
            # print("ratio of overlap:", np.sum(overlap) / np.sum(loss_sim))
            # print('loss shape:', loss_sim.shape, overlap.shape, loss.shape)
        # find the best boundary width
        # breakpoint()
        best_boundary_width_index = np.argmin(loss, axis=1) # [#step]

        # Use broadcasting to create row indices corresponding to each row
        # row_indices = np.arange(num_steps)[:, np.newaxis]
        # breakpoint()
        # print(loss.shape, best_boundary_width.shape, boundary_width_candidates.shape)
        boundary_width = [boundary_width_candidates[i, 0][best_boundary_width_index[i]] for i in range(num_steps)] # [#step]
        # boundary_width = boundary_width_candidates[:,0][row_indices, best_boundary_width_index] # [#step]
        boundary_width = np.array(boundary_width)
        # print(boundary_width.shape)

    bbox = []
    left_bound = np.clip(center_indexes - boundary_width // 2, 0, num_frames)
    right_bound = np.clip(center_indexes + boundary_width // 2, 0, num_frames)
    # breakpoint()
    bbox = np.stack([left_bound, right_bound], axis=1).round().astype(int)

    return bbox.tolist()

# use optimization to compute pseudo boundary
def align_frame_into_steps_op_v1(frame_features, step_features, topk=15, num_iterations=4, beta=1, order=False, scale=1):
    # frame_features:  torch.Size([200, 768])
    augment_ratio_list = np.arange(0.5, 2, 0.1)

    if step_features.shape[0] == 0:
        return -np.ones(frame_features.shape[0])
    
    # breakpoint()
    # [#step, #frame]
    similarity_matrix = compute_sim(step_features, frame_features, config_eval_l2norm).cpu().numpy()

    num_steps, num_frames = similarity_matrix.shape

    # Select top-k frames for each caption [#step, #topk]
    sorted_indices = np.argsort(similarity_matrix, axis=1)
    # top_indices = np.argsort(similarity_matrix, axis=1)[:, -topk:]
    # top_values = np.take_along_axis(similarity_matrix, top_indices, axis=1)

    # Compute center indexes [#step, 1]
    

    # Update boundary width 
    initial_boundary_width = num_frames / num_steps # 1
    # boundary_width = initial_boundary_width * np.ones(num_steps, 1, 1) # 1 
    # overlap = np.zeros(num_steps)

    for i in range(num_iterations):
        if i == 0 and not order:
            boundary_width_last = np.full(num_steps, initial_boundary_width).reshape(-1, 1, 1)
            topk_indices = [index[-topk:] for index in sorted_indices]
            topk_values = [similarity_matrix[i][index] for i, index in enumerate(topk_indices)]


            center_indexes = np.array([find_center_value(index) for index in topk_indices]) 
            previous_index_center = None
        #     # overlap_weight = 0
        else:
            if i == 0:
                segment_boundary = np.linspace(0, num_frames, num_steps + 1).round().astype(int)
                start_indices, end_indices = segment_boundary[:-1], segment_boundary[1:]
                start_indices = np.clip(start_indices - initial_boundary_width * scale, 0, num_frames)
                end_indices = np.clip(end_indices + initial_boundary_width * scale, 0, num_frames)
                boundary_width_last = (end_indices - start_indices).reshape(-1, 1, 1)

                filtered_indices = [sorted_indices[i][(sorted_indices[i] >= start_indices[i]) & (sorted_indices[i] <= end_indices[i])] for i in range(num_steps)]
                if sum(len(index) for index in filtered_indices) < topk * num_steps * 0.4:
                    boundary_width_last = np.full(num_steps, initial_boundary_width).reshape(-1, 1, 1)
                    topk_indices = [index[-topk:] for index in sorted_indices]
                    topk_values = [similarity_matrix[i][index] for i, index in enumerate(topk_indices)]


                    center_indexes = np.array([find_center_value(index) for index in topk_indices]) 
                    previous_index_center = None
            else:
                boundary_width_last = boundary_width.reshape(-1, 1, 1)
                start_indices = np.clip(center_indexes - boundary_width // 2 - initial_boundary_width * scale, 0, num_frames)
                end_indices = np.clip(center_indexes + boundary_width // 2 + initial_boundary_width * scale, 0, num_frames)

            topk_indices = []
            topk_values = []
            for j, (start, end) in enumerate(zip(start_indices, end_indices)):
                # breakpoint()
                filtered_indices = sorted_indices[j][(sorted_indices[j] >= start) & (sorted_indices[j] <= end)]
                topk_index = filtered_indices[-topk:]
                topk_indices.append(topk_index)
                topk_values.append(similarity_matrix[j][topk_index])
            previous_index_center = center_indexes.copy() if i > 0 else None
            center_indexes = np.array([find_center_value(index) for index in topk_indices]) 

            # top_indices = sorted_indices[:, ]
            # previous_index_center = center_indexes
            # # overlap_weight = 0.5 * np.sum(overlap)
        
        boundary_width_candidates = augment_ratio_list * boundary_width_last # [#steps, 1, #candidates]
        # breakpoint()

        index_distance = [np.abs(index - center_indexes[i] + 1e-3)[:, np.newaxis] for i, index in enumerate(topk_indices)] # [[topk, 1]]
        
        weight_distance = [clip_array(index_distance[i], 0.5 * boundary_width_candidates[i]) for i in range(len(topk_indices))] # [[topk, 1]]

        loss_candidates_list = [value[:, np.newaxis] / weight_distance[i] * (np.abs(index_distance[i] - 0.5 * boundary_width_candidates[i])) for i, value in enumerate(topk_values)] # [[topk, candidates]]
        # loss_candidates_list = [value[:, np.newaxis] / index_distance[i] * (np.abs(index_distance[i] - 0.5 * boundary_width_candidates[i])) for i, value in enumerate(topk_values)] # [[topk, candidates]]


        # index_distance = np.abs(topk_indices - center_indexes)[:, :, np.newaxis] # [#step, #topk, 1]

        # loss_sim = np.sum(top_values[:, :, np.newaxis] / index_distance * (np.abs(index_distance - 0.5 * boundary_width_candidates)), axis=1) # [#step, #candidates]
        loss_sim = np.array([np.mean(loss, axis=0) for loss in loss_candidates_list]) # [#step, #candidates]

        if i == 0:            
            loss = loss_sim
            # print('loss shape:', loss_sim.shape, loss.shape)
        else:
            # measure the overlap between boundaries given center and boundary width
            overlap = compute_overlap(center_indexes, boundary_width_candidates, previous_index_center, boundary_width_last) # [#step, #candidates]
            # breakpoint()
            # print(loss_sim.shape, overlap.shape)
            loss = loss_sim + beta * overlap 
            # print("ratio of overlap:", np.sum(overlap) / np.sum(loss_sim))
            # print('loss shape:', loss_sim.shape, overlap.shape, loss.shape)
        # find the best boundary width
        # breakpoint()
        best_boundary_width_index = np.argmin(loss, axis=1) # [#step]

        # Use broadcasting to create row indices corresponding to each row
        # row_indices = np.arange(num_steps)[:, np.newaxis]
        # breakpoint()
        # print(loss.shape, best_boundary_width.shape, boundary_width_candidates.shape)
        boundary_width = [boundary_width_candidates[i, 0][best_boundary_width_index[i]] for i in range(num_steps)] # [#step]
        # boundary_width = boundary_width_candidates[:,0][row_indices, best_boundary_width_index] # [#step]
        boundary_width = np.array(boundary_width)
        # print(boundary_width.shape)

    bbox = []
    left_bound = np.clip(center_indexes - boundary_width // 2, 0, num_frames)
    right_bound = np.clip(center_indexes + boundary_width // 2, 0, num_frames)
    # breakpoint()
    bbox = np.stack([left_bound, right_bound], axis=1).round().astype(int)

    return bbox.tolist()





# # use optimization to compute pseudo boundary
# def align_frame_into_steps_op_order(frame_features, step_features, topk=15, threshold=0.5, num_iterations=4, beta=1):
#     # frame_features:  torch.Size([200, 768])
#     augment_ratio_list = np.arange(0.5, 2, 0.1)

#     if step_features.shape[0] == 0:
#         return -np.ones(frame_features.shape[0])
    
#     # breakpoint()
#     # [#step, #frame]
#     similarity_matrix = compute_sim(step_features, frame_features, config_eval_l2norm).cpu().numpy()

#     num_steps, num_frames = similarity_matrix.shape

#     # Select top-k frames for each caption [#step, #topk]
#     top_indices = np.argsort(similarity_matrix, axis=1)[:, -topk:]
#     top_values = np.take_along_axis(similarity_matrix, top_indices, axis=1)

#     # Compute center indexes [#step, 1]
#     center_indexes = find_center_index(top_indices)[:, np.newaxis]

#     # Update boundary width 
#     initial_boundary_width = num_frames / num_steps # 1
#     # boundary_width = initial_boundary_width * np.ones(num_steps, 1, 1) # 1 
#     # overlap = np.zeros(num_steps)

#     for i in range(num_iterations):
#         if i == 0:
#             boundary_width_last = np.full(num_steps, initial_boundary_width).reshape(-1, 1, 1)
#         #     previous_index_center = None
#         #     # overlap_weight = 0
#         else:
#             boundary_width_last = boundary_width.reshape(-1, 1, 1)
#             previous_index_center = center_indexes
#             # overlap_weight = 0.5 * np.sum(overlap)
        
#         boundary_width_candidates = augment_ratio_list * boundary_width_last # [#steps, 1, #candidates]

#         index_distance = np.abs(top_indices - center_indexes)[:, :, np.newaxis] # [#step, #topk, 1]

#         loss_sim = np.sum(top_values[:, :, np.newaxis] / index_distance * (np.abs(index_distance - 0.5 * boundary_width_candidates)), axis=1) # [#step, #candidates]

#         if i == 0:
#             loss = loss_sim # # [#step, #candidates]
#             print('loss shape:', loss_sim.shape, loss.shape)
#         else:
#             # measure the overlap between boundaries given center and boundary width
#             overlap = compute_overlap(center_indexes, boundary_width_candidates, previous_index_center, boundary_width_last) # [#step, #candidates]
#             loss = loss_sim + beta * overlap 
#             print('loss shape:', loss_sim.shape, overlap.shape, loss.shape)
#         # find the best boundary width
#         # breakpoint()
#         best_boundary_width = np.argmin(loss, axis=1) # [#step]
#         # print(loss.shape, best_boundary_width.shape, boundary_width_candidates.shape)
#         boundary_width = boundary_width_candidates[:,0][np.arange(num_steps), best_boundary_width] # [#step]
#         # print(boundary_width.shape)

#     return center_indexes, boundary_width
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
        # if iter == 0:
        #     refined_uniform_boxes = expand_window(uniform_boxes, frame_features.shape[0], step_features.shape[0], ratio)
        # else:
        #     refined_uniform_boxes = expand_window(bbox, frame_features.shape[0], step_features.shape[0], ratio) # last bbox


        # global: from all frames, local: from refined uniform boxes
    
        # top_indices_list_local = [sorted_index[i][(sorted_index[i] >= refined_uniform_boxes[i][0]) & (sorted_index[i] <= refined_uniform_boxes[i][1])][:topk] for i in range(sim.shape[0])]
        # top_values_list_local = [sim[i][top_indices_list_local[i]] for i in range(sim.shape[0])]

        # size_local = [len(top_indices_list_local[i]) for i in range(sim.shape[0])]
        # if sum(size_local) < (topk-2) * len(size_local):
        #     top_indices_list = top_indices_list_global
        #     top_values_list = top_values_list_global
        # else:
        #     top_indices_list = top_indices_list_local
        #     top_values_list = top_values_list_local

        # top_indices_list = [top_indices_list_global[i] if len(top_indices_list_local[i]) < topk else top_indices_list_local[i] for i in range(sim.shape[0])]

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



# pesudo box 4: based on fixed window. the result is bad. give up
def align_frame_into_steps_mode(frame_features, step_features, topk=15, w=2, ratio=1):
    # breakpoint()
    if step_features.shape[0] == 0:
        return -np.ones(frame_features.shape[0])
    
    sim = compute_sim(step_features, frame_features, config_eval_l2norm).cpu()
    # frame_features, step_features = frame_features.cpu(), step_features.cpu()


    bbox = get_mode_box(sim, topk, w, ratio)
    return bbox

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

# pesudo box 3: based on sim, consider the order of steps
def align_frame_into_steps_order(frame_features, step_features, unordered=False, topk=15, threshold=2, w=2, mode='median', ratio=1):
    # breakpoint()
    if step_features.shape[0] == 0:
        return -np.ones(frame_features.shape[0])
    
    sim = compute_sim(step_features, frame_features, config_eval_l2norm).cpu()

    uniform_boxes = uniform_window(frame_features.shape[0], step_features.shape[0])
    refined_uniform_boxes = expand_window(uniform_boxes, frame_features.shape[0], step_features.shape[0], ratio)
    
    # old setting (index is wrong)
    # # frame_features, step_features = frame_features.cpu(), step_features.cpu()
    # index_sim_list = [sim[i][refined_uniform_boxes[i][0]: refined_uniform_boxes[i][1]] for i in range(sim.shape[0])]
    # top_indices_list = [torch.topk(index_sim, k, dim=0, largest=True, sorted=True)[1] for index_sim in index_sim_list]
    # # top_values, top_indices = torch.topk(sim, k, dim=1, largest=True, sorted=True)

    sorted_index = torch.argsort(-sim, dim=1)
    # global: from all frames, local: from refined uniform boxes
    top_indices_list_global = [sorted_index[i][:topk] for i in range(sim.shape[0])]
    top_indices_list_local = [sorted_index[i][(sorted_index[i] >= refined_uniform_boxes[i][0]) & (sorted_index[i] <= refined_uniform_boxes[i][1])][:topk] for i in range(sim.shape[0])]

    size_local = [len(top_indices_list_local[i]) for i in range(sim.shape[0])]
    if sum(size_local) < (topk-2) * len(size_local):
        top_indices_list = top_indices_list_global
    else:
        top_indices_list = top_indices_list_local

    # top_indices_list = [top_indices_list_global[i] if len(top_indices_list_local[i]) < topk else top_indices_list_local[i] for i in range(sim.shape[0])]

    bbox = []
    for i in range(len(top_indices_list)):
        filtered_indices = remove_outliers(top_indices_list[i].tolist(), threshold, mode=mode, w=w)
        if len(filtered_indices) == 0:
            filtered_indices = remove_outliers(top_indices_list_global[i].tolist(), 0.5, mode=mode, w=w)
            if len(filtered_indices) == 0:
                bbox.append(uniform_boxes[i])
                continue
        bbox.append([min(filtered_indices), max(filtered_indices)])

    return bbox




# based on pbox3, if ratio 1 has enough value, use it otherwise
def align_frame_into_steps_order_adapt(frame_features, step_features, unordered=False, topk=15, threshold=2, w=2, mode='median', ratio=1):
    # breakpoint()
    if step_features.shape[0] == 0:
        return -np.ones(frame_features.shape[0])
    
    sim = compute_sim(step_features, frame_features, config_eval_l2norm).cpu()

    uniform_boxes = uniform_window(frame_features.shape[0], step_features.shape[0])
    refined_uniform_boxes = expand_window(uniform_boxes, frame_features.shape[0], step_features.shape[0], ratio)
    
    # old setting (index is wrong)
    # # frame_features, step_features = frame_features.cpu(), step_features.cpu()
    # index_sim_list = [sim[i][refined_uniform_boxes[i][0]: refined_uniform_boxes[i][1]] for i in range(sim.shape[0])]
    # top_indices_list = [torch.topk(index_sim, k, dim=0, largest=True, sorted=True)[1] for index_sim in index_sim_list]
    # # top_values, top_indices = torch.topk(sim, k, dim=1, largest=True, sorted=True)

    sorted_index = torch.argsort(-sim, dim=1)
    # global: from all frames, local: from refined uniform boxes
    top_indices_list_global = [sorted_index[i][:topk] for i in range(sim.shape[0])]
    top_indices_list_local = [sorted_index[i][(sorted_index[i] >= refined_uniform_boxes[i][0]) & (sorted_index[i] <= refined_uniform_boxes[i][1])][:topk] for i in range(sim.shape[0])]

    size_local = [len(top_indices_list_local[i]) for i in range(sim.shape[0])]
    if sum(size_local) < (topk-1) * len(size_local):
        flag = 0
        for i in range(4):
            refined_uniform_boxes = expand_window(uniform_boxes, frame_features.shape[0], step_features.shape[0], ratio+i*0.5)
            top_indices_list_local = [sorted_index[i][(sorted_index[i] >= refined_uniform_boxes[i][0]) & (sorted_index[i] <= refined_uniform_boxes[i][1])][:topk] for i in range(sim.shape[0])]
            size_local = [len(top_indices_list_local[i]) for i in range(sim.shape[0])]
            if sum(size_local) >= (topk-1) * len(size_local):
                flag = 1
                break
        if flag == 0:
            top_indices_list = top_indices_list_global
        else:
            top_indices_list = top_indices_list_local

    else:
        top_indices_list = top_indices_list_local

    # top_indices_list = [top_indices_list_global[i] if len(top_indices_list_local[i]) < topk else top_indices_list_local[i] for i in range(sim.shape[0])]

    bbox = []
    for i in range(len(top_indices_list)):
        filtered_indices = remove_outliers(top_indices_list[i].tolist(), threshold, mode=mode, w=w)
        if len(filtered_indices) == 0:
            filtered_indices = remove_outliers(top_indices_list_global[i].tolist(), 0.5, mode=mode, w=w)
            if len(filtered_indices) == 0:
                bbox.append(uniform_boxes[i])
                continue
        bbox.append([min(filtered_indices), max(filtered_indices)])

    return bbox

def step_retrieval_weight_sim_order(frame_features, step_features, unordered=False, topk=15, threshold=2, w=2, ratio=1):
    # breakpoint()
    if step_features.shape[0] == 0:
        return -np.ones(frame_features.shape[0])
    
    sim = compute_sim(step_features, frame_features, config_eval_l2norm).cpu()
    # breakpoint()

    window_sums = torch.nn.functional.conv1d(sim.unsqueeze(1), torch.ones(1, 1, 2 * w + 1)).squeeze()
    if len(window_sums.shape) == 1:
        window_sums = window_sums.unsqueeze(0)


    sorted_index = torch.argsort(-window_sums, dim=1) + w
    
    

    uniform_boxes = uniform_window(frame_features.shape[0], step_features.shape[0])
    refined_uniform_boxes = expand_window(uniform_boxes, frame_features.shape[0], step_features.shape[0], ratio)

    top_indices_list_global = [sorted_index[i][:topk] for i in range(sim.shape[0])]
    top_indices_list_local = [sorted_index[i][(sorted_index[i] >= refined_uniform_boxes[i][0]) & (sorted_index[i] <= refined_uniform_boxes[i][1])][:topk] for i in range(sim.shape[0])]
    

    size_local = [len(top_indices_list_local[i]) for i in range(sim.shape[0])]
    if sum(size_local) < (topk-2) * len(size_local):
        top_indices_list = top_indices_list_global
    else:
        top_indices_list = top_indices_list_local

    # top_indices_list = [top_indices_list_global[i] if len(top_indices_list_local[i]) < topk else top_indices_list_local[i] for i in range(sim.shape[0])]

    bbox = []
    for i in range(len(top_indices_list)):
        threshold_value = compute_threshold(top_indices_list[i].tolist(), threshold)
        filtered_indices = [frame for frame in top_indices_list[i].tolist() if abs(frame - top_indices_list[i][0]) <= threshold_value]
        if len(filtered_indices) == 0:
            bbox.append([top_indices_list[i] - w, top_indices_list[i] + w])
        else:
            bbox.append([min(filtered_indices), max(filtered_indices)])

    return bbox

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