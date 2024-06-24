import torch
import numpy as np

# def find_center_index(array: np.ndarray) -> np.ndarray:
#     """
#     Given a array with shape [steps, topk], find the center index between topk indexes
#     which has the minimal average distance with other indexes.

#     Args:
#     - array: numpy array representing the input array with shape [steps, topk]

#     Returns:
#     - center_indexes: numpy array of center indexes for each step
#     """

#     distances = np.sum(np.abs(array[:, np.newaxis, :] - array[:, :, np.newaxis]), axis=2)
#     center_indexes = np.argmin(distances, axis=1)

#     return center_indexes

def find_center_value(arr):
    # Compute pairwise distances between all values
    distances = np.abs(arr[:, np.newaxis] - arr[np.newaxis, :])
    
    # Sum distances for each value
    sum_distances = np.sum(distances, axis=1)
    
    # Find the index of the value with the smallest sum distance
    center_index = np.argmin(sum_distances)
    
    # Get the center value
    center_value = arr[center_index]
    
    return center_value


def compute_overlap(center_t, boundary_t, center_t_minus_1, boundary_t_minus_1):
    """
    Compute the overlap of boundaries between time t and t-1 for each element in the arrays.

    Args:
    - center_t: numpy array representing the center at time t with shape [N,]
    - boundary_t: numpy array representing the boundary at time t with shape [N,1, candidates]
    - center_t_minus_1: numpy array representing the center at time t-1 with shape [N,]
    - boundary_t_minus_1: numpy array representing the boundary at time t-1 with shape [N,]

    Returns:
    - overlap: numpy array representing the overlap of boundaries with shape [N,]
    """

    boundary_t = boundary_t.squeeze(1)
    boundary_t_minus_1 = boundary_t_minus_1.squeeze(1)
    center_t = center_t[:, np.newaxis]
    # breakpoint()
    center_t_minus_1 = center_t_minus_1[:, np.newaxis]
    # boundary_t_minus_1 = boundary_t_minus_1[:, np.newaxis]


    # Calculate the start and end positions of the boundaries at time t and t-1
    start_t = center_t - 0.5 * boundary_t
    end_t = center_t + 0.5 * boundary_t
    start_t_minus_1 = center_t_minus_1 - 0.5 * boundary_t_minus_1
    end_t_minus_1 = center_t_minus_1 + 0.5 * boundary_t_minus_1

    # Calculate the intersection and union of the boundaries
    intersection = np.maximum(0, np.minimum(end_t, end_t_minus_1) - np.maximum(start_t, start_t_minus_1))
    union = boundary_t + boundary_t_minus_1 - intersection

    # Compute the overlap using the Intersection over Union (IoU) formula
    overlap = intersection / union

    return overlap