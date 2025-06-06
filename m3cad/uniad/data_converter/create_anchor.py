import os
import numpy as np
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from sklearn.cluster import KMeans
import pickle


def classify_label_type_to_id(label_type: str) -> int:
    """
    Classifies label type to id.
    
    Args:
        label_type (str): The category name of the object.

    Returns:
        int: The corresponding id for the label type.
    """
    if 'human' in label_type:
        return 2
    if 'movable_object' in label_type:
        return 3
    if ('vehicle.bicycle' in label_type) or ('vehicle.motorcycle' in label_type):
        return 1
    else:
        return 0


def k_means_anchors(k: int, future_traj_all: np.ndarray) -> np.ndarray:
    """
    Extracts anchors for multipath/covernet using k-means on train set
    trajectories.
    
    Args:
        k (int): The number of clusters for k-means algorithm.
        future_traj_all (np.ndarray): The array containing all future trajectories.

    Returns:
        np.ndarray: The k anchor trajectories.
    """
    prototype_traj = future_traj_all
    traj_len = prototype_traj.shape[1]
    traj_dim = prototype_traj.shape[2]
    ds_size = future_traj_all.shape[0]
    trajectories = future_traj_all
    clustering = KMeans(n_clusters=k).fit(trajectories.reshape((ds_size, -1)))
    anchors = np.zeros((k, traj_len, traj_dim))
    for i in range(k):
        anchors[i] = np.mean(trajectories[clustering.labels_ == i], axis=0)
    return anchors


def run(version: str = 'v1.0-mini',
        dataroot: str = 'data/openv2v_4_cams',
        output_pkl: str = 'data/others/cooper_uniad_motion_anchor_infos_mode6.pkl',
        num_modes: int = 6,
        predicted_traj_len: int = 12) -> None:
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    predict_helper = PredictHelper(nusc)
    all_sample_tokens = [sample['token'] for sample in nusc.sample]
    grouped_trajectories = [[] for _ in range(4)]
    for sample_token in all_sample_tokens:
        sd_rec = nusc.get('sample', sample_token)
        ann_tokens = sd_rec['anns']
        for ann_token in ann_tokens:
            ann_record = nusc.get('sample_annotation', ann_token)
            label_type = ann_record['category_name']
            type_id = classify_label_type_to_id(label_type)
            instance_token = ann_record['instance_token']
            fut_traj_local = predict_helper.get_future_for_agent(
                instance_token, sample_token, seconds=6, in_agent_frame=True)
            if fut_traj_local.shape[0] < predicted_traj_len:
                continue
            grouped_trajectories[type_id].append(fut_traj_local)
    kmeans_anchors = []
    for type_id in [0]:
        if len(grouped_trajectories[type_id]) == 0:
            continue
        grouped_trajectory = np.stack(grouped_trajectories[type_id])
        kmeans_anchors.append(k_means_anchors(num_modes, grouped_trajectory))
    res = {
        'K_mode': num_modes,
        'anchors_all': kmeans_anchors,
        'class_list': [[0]],
        'grouped_classes': [['car']],
    }
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump(res, f)


if __name__ == '__main__':
    run(
        version='v1.0-trainval',
        dataroot='data/m3cad_carla_ue5',
        output_pkl='data/m3cad_carla_ue5/others/m3cad_uniad_motion_anchor_infos_mode6.pkl'
    )
