#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
from typing import Any, Final, List

import matplotlib
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
import mmcv

DESCRIPTION = """
# Nuopenv2v Dataset Visualization with Multiple Scenes
"""

EXAMPLE_DIR: Final = pathlib.Path(__file__).parent.parent
DATASET_DIR: Final = EXAMPLE_DIR / "dataset"

cmap = matplotlib.colormaps["turbo_r"]
norm = matplotlib.colors.Normalize(
    vmin=3.0,
    vmax=75.0,
)


def parse_predictions_multitask_pkl(predroot: str, score_thresh=0.25):
    """
    Reference the _parse_predictions_multitask_pkl example, simplify and retain only 3D detection box related content.
    Returns a dictionary: {sample_token: [list_of_predicted_boxes]}.
    Each predicted_box is represented as (center, dim, yaw, score, label, track_id).
    """
    if predroot is None or not pathlib.Path(predroot).exists():
        print(f"[WARNING] predroot={predroot} does not exist, unable to load prediction file, returning empty prediction set.")
        return {}

    outputs = mmcv.load(predroot)
    if "bbox_results" not in outputs:
        raise ValueError("The prediction pkl does not contain the `bbox_results` field!")

    results = outputs["bbox_results"]
    prediction_dict = {}

    for item in results:
        token = item["token"]
        if "boxes_3d" not in item:
            continue

        bboxes = item["boxes_3d"]
        scores = item["scores_3d"]
        labels = item["labels_3d"]

        track_scores = scores.cpu().numpy()
        track_labels = labels.cpu().numpy()
        box_tensor = bboxes.tensor.cpu().detach().numpy()  # get the box tensor
        box_center = bboxes.gravity_center.cpu().detach().numpy() # get the gravity center of the box

        track_ids = None
        if "track_ids" in item:
            track_ids = item["track_ids"].cpu().numpy()

        predicted_boxes_list = []
        for i in range(len(track_scores)):
            score = float(track_scores[i])
            if score < score_thresh:
                continue

            # https://mmdetection3d.readthedocs.io/en/latest/_modules/mmdet3d/structures/bbox_3d/lidar_box3d.html#LiDARInstance3DBoxes
            cx, cy, cz = box_center[i]
            x_size = float(box_tensor[i, 3])
            y_size = float(box_tensor[i, 4])
            z_size = float(box_tensor[i, 5])

            yaw = float(box_tensor[i, 6])

            label = int(track_labels[i])
            tid = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else -1

            predicted_boxes_list.append(
                dict(
                    center=(cx, cy, cz),
                    size=(x_size, y_size, z_size),
                    yaw=yaw,
                    label=label,
                    score=score,
                    track_id=tid,
                )
            )

        prediction_dict[token] = predicted_boxes_list

    return prediction_dict


def yaw_to_quaternion_3d(yaw: float) -> Quaternion:
    """Convert yaw rotation around Z-axis to quaternion (default wxyz format)."""
    return Quaternion(axis=[0, 0, 1], radians=yaw)


def load_pcd(file_path: str) -> np.ndarray:
    """
    Read an ASCII PCD file and return a NumPy array of shape (num_points, 3).
    Ignore RGB information, only extract x, y, z coordinates.
    """
    points = []
    header_ended = False
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not header_ended:
                if line.startswith("DATA ascii"):
                    header_ended = True
                continue

            if header_ended and line:
                values = line.split()
                if len(values) < 3:
                    continue
                try:
                    x, y, z = float(values[0]), float(values[1]), float(values[2])
                except ValueError:
                    continue
                points.append([x, y, z])

    return np.array(points, dtype=np.float32)


def ensure_scene_available(nusc: NuScenes, scene_name: str) -> None:
    scene_names = [s["name"] for s in nusc.scene]
    if scene_name not in scene_names:
        raise ValueError(f"{scene_name=} not found in dataset")


def nuscene_sensor_names(nusc: NuScenes, scene_name: str) -> list[str]:
    sensor_names = set()

    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    first_sample = nusc.get("sample", scene["first_sample_token"])
    for sample_data_token in first_sample["data"].values():
        sample_data = nusc.get("sample_data", sample_data_token)
        if sample_data["sensor_modality"] == "camera":
            current_camera_token = sample_data_token
            while current_camera_token != "":
                sample_data = nusc.get("sample_data", current_camera_token)
                sensor_name = sample_data["channel"]
                sensor_names.add(sensor_name)
                current_camera_token = sample_data["next"]

    ordering = {
        "CAM_FRONT_LEFT": 0,
        "CAM_FRONT": 1,
        "CAM_FRONT_RIGHT": 2,
        "CAM_BACK_RIGHT": 3,
        "CAM_BACK": 4,
        "CAM_BACK_LEFT": 5,
    }
    return sorted(list(sensor_names), key=lambda sensor_name: ordering.get(sensor_name, float("inf")))


def log_nuscenes(
    nusc: NuScenes,
    scene_name: str,
    max_time_sec: float,
    pred_dict: dict[str, Any],
    prefix: str = "world",
    log_pred: bool = False
) -> None:
    """
    Extend the original log_nuscenes by adding logging of predicted boxes from pred_dict.
    The `prefix` parameter allows logging multiple scenes under different namespaces.
    """
    scene = next(s for s in nusc.scene if s["name"] == scene_name)

    rr.log(f"{prefix}", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    first_sample_token = scene["first_sample_token"]
    first_sample = nusc.get("sample", first_sample_token)

    first_lidar_token = ""
    first_radar_tokens = []
    first_camera_tokens = []
    for sample_data_token in first_sample["data"].values():
        sample_data = nusc.get("sample_data", sample_data_token)
        log_sensor_calibration(sample_data, nusc, prefix)

        if sample_data["sensor_modality"] == "lidar":
            first_lidar_token = sample_data_token
        elif sample_data["sensor_modality"] == "radar":
            first_radar_tokens.append(sample_data_token)
        elif sample_data["sensor_modality"] == "camera":
            first_camera_tokens.append(sample_data_token)

    if first_lidar_token == "":
        print(f"[WARNING] No LIDAR data found for scene {scene_name}. Skipping scene.")
        return

    first_timestamp_us = nusc.get("sample_data", first_lidar_token)["timestamp"]
    max_timestamp_us = first_timestamp_us + 1e6 * max_time_sec

    log_lidar_and_ego_pose(first_lidar_token, nusc, max_timestamp_us, prefix)

    if prefix == "world":  # Only log cameras for the ego vehicle
        log_cameras(first_camera_tokens, nusc, max_timestamp_us, prefix)

    log_radars(first_radar_tokens, nusc, max_timestamp_us, prefix)

    # Original GT annotation boxes
    log_annotations(first_sample_token, nusc, max_timestamp_us, prefix)

    if log_pred:
        # Log predicted boxes
        log_predictions(nusc, first_sample_token, max_timestamp_us, pred_dict, prefix)


def log_lidar_and_ego_pose(first_lidar_token: str, nusc: NuScenes, max_timestamp_us: float, prefix: str) -> None:
    current_lidar_token = first_lidar_token

    while current_lidar_token != "":
        sample_data = nusc.get("sample_data", current_lidar_token)
        sensor_name = sample_data["channel"]

        if max_timestamp_us < sample_data["timestamp"]:
            break

        rr.set_time_seconds("timestamp", sample_data["timestamp"] * 1e-6)

        ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        rotation_xyzw = np.roll(ego_pose["rotation"], shift=-1)  # from wxyz to xyzw

        rr.log(
            f"{prefix}/ego_vehicle",
            rr.Transform3D(
                translation=ego_pose["translation"],
                rotation=rr.Quaternion(xyzw=rotation_xyzw),
                axis_length=10.0,
                from_parent=False,
            ),
        )

        current_lidar_token = sample_data["next"]

        data_file_path = nusc.dataroot / sample_data["filename"]
        points = load_pcd(str(data_file_path))

        # exchange x and y axes because lidar is in openv2v lidar coordinate system
        # We need transform openv2v lidar coordinate system to nuopenv2v lidar coordinate system
        points[:, [0, 1]] = points[:, [1, 0]]

        point_distances = np.linalg.norm(points, axis=1)
        point_colors = cmap(norm(point_distances))

        rr.log(f"{prefix}/ego_vehicle/{sensor_name}/lidar_points", 
               rr.Points3D(points, colors=point_colors),
        )


def log_cameras(first_camera_tokens: list[str], nusc: NuScenes, max_timestamp_us: float, prefix: str) -> None:
    for first_camera_token in first_camera_tokens:
        current_camera_token = first_camera_token
        while current_camera_token != "":
            sample_data = nusc.get("sample_data", current_camera_token)
            if max_timestamp_us < sample_data["timestamp"]:
                break
            sensor_name = sample_data["channel"]
            rr.set_time_seconds("timestamp", sample_data["timestamp"] * 1e-6)
            data_file_path = nusc.dataroot / sample_data["filename"]
            rr.log(f"{prefix}/ego_vehicle/{sensor_name}", rr.EncodedImage(path=str(data_file_path)))
            current_camera_token = sample_data["next"]


def log_radars(first_radar_tokens: list[str], nusc: NuScenes, max_timestamp_us: float, prefix: str) -> None:
    for first_radar_token in first_radar_tokens:
        current_radar_token = first_radar_token
        while current_radar_token != "":
            sample_data = nusc.get("sample_data", current_radar_token)
            if max_timestamp_us < sample_data["timestamp"]:
                break
            sensor_name = sample_data["channel"]
            rr.set_time_seconds("timestamp", sample_data["timestamp"] * 1e-6)
            data_file_path = nusc.dataroot / sample_data["filename"]
            pointcloud = NuScenes.RadarPointCloud.from_file(str(data_file_path))
            points = pointcloud.points[:3].T
            point_distances = np.linalg.norm(points, axis=1)
            point_colors = cmap(norm(point_distances))
            rr.log(
                f"{prefix}/ego_vehicle/{sensor_name}",
                rr.Points3D(points, colors=point_colors),
            )
            current_radar_token = sample_data["next"]


def log_annotations(first_sample_token: str, nusc: NuScenes, max_timestamp_us: float, prefix: str) -> None:
    """
    Render original GT boxes.
    """
    label2id: dict[str, int] = {}
    current_sample_token = first_sample_token
    while current_sample_token != "":
        sample = nusc.get("sample", current_sample_token)
        if max_timestamp_us < sample["timestamp"]:
            break
        rr.set_time_seconds("timestamp", sample["timestamp"] * 1e-6)
        ann_tokens = sample["anns"]
        sizes = []
        centers = []
        quaternions = []
        class_ids = []
        for ann_token in ann_tokens:
            ann = nusc.get("sample_annotation", ann_token)

            rotation_xyzw = np.roll(ann["rotation"], shift=-1)  # from wxyz to xyzw
            width, length, height = ann["size"]
            sizes.append((length, width, height))
            centers.append(ann["translation"])
            quaternions.append(rr.Quaternion(xyzw=rotation_xyzw))
            if ann["category_name"] not in label2id:
                label2id[ann["category_name"]] = len(label2id)
            class_ids.append(label2id[ann["category_name"]])

        rr.log(
            f"{prefix}/anns",
            rr.Boxes3D(
                sizes=sizes,
                centers=centers,
                quaternions=quaternions,
                class_ids=class_ids,
                colors=[(0, 255, 0)] * len(sizes),
                labels=[ann_token.split('_')[-1] for ann_token in ann_tokens] * len(sizes)
            ),
        )
        current_sample_token = sample["next"]

    annotation_context = [(i, label) for label, i in label2id.items()]
    rr.log(f"{prefix}/anns", rr.AnnotationContext(annotation_context), static=True)


def log_predictions(
    nusc: NuScenes,
    first_sample_token: str,
    max_timestamp_us: float,
    pred_dict: dict[str, Any],
    prefix: str
):
    """
    Visualize predicted boxes from parse_predictions_multitask_pkl in Rerun.
    Like GT boxes, placed under "{prefix}/preds".
    """
    if not pred_dict:
        return

    label2id: dict[int, int] = {}

    current_sample_token = first_sample_token
    while current_sample_token != "":
        sample = nusc.get("sample", current_sample_token)
        if max_timestamp_us < sample["timestamp"]:
            break

        sample_token = sample["token"]
        rr.set_time_seconds("timestamp", sample["timestamp"] * 1e-6)

        if sample_token not in pred_dict:
            current_sample_token = sample["next"]
            continue

        preds_this_sample = pred_dict[sample_token]

        if len(preds_this_sample) == 0:
            current_sample_token = sample["next"]
            continue

        sizes = []
        centers = []
        quaternions = []
        class_ids = []
        for p in preds_this_sample:
            cx, cy, cz = p["center"]
            l, w, h = p["size"]
            yaw = p["yaw"]
            label = p["label"]
            score = p["score"]
            track_id = p["track_id"]

            q = yaw_to_quaternion_3d(yaw)
            rotation_xyzw = [q.x, q.y, q.z, q.w]

            sizes.append((l, w, h))
            centers.append((cx, cy, cz))
            quaternions.append(rr.Quaternion(xyzw=rotation_xyzw))

            if label not in label2id:
                label2id[label] = label
            class_ids.append(label2id[label])

        rr.log(
            f"{prefix}/ego_vehicle/LIDAR_TOP/preds",
            rr.Boxes3D(
                sizes=sizes,
                centers=centers,
                quaternions=quaternions,
                class_ids=class_ids,
                colors=[(255, 165, 0)] * len(sizes),
                labels=[f"pred"] * len(sizes),
            ),
        )

        current_sample_token = sample["next"]

    annotation_context = []
    for label_id in label2id:
        annotation_context.append((label2id[label_id], f"pred_label_{label_id}"))
    rr.log(f"{prefix}/preds", rr.AnnotationContext(annotation_context), static=True)


def log_sensor_calibration(sample_data: dict[str, Any], nusc: NuScenes, prefix: str) -> None:
    sensor_name = sample_data["channel"]
    calibrated_sensor_token = sample_data["calibrated_sensor_token"]
    calibrated_sensor = nusc.get("calibrated_sensor", calibrated_sensor_token)
    rotation_xyzw = np.roll(calibrated_sensor["rotation"], shift=-1)  # from wxyz to xyzw

    if sensor_name == "LIDAR_TOP":
        rr.log(
            f"{prefix}/ego_vehicle/{sensor_name}",
            rr.Transform3D(
                translation=calibrated_sensor["translation"],
                rotation=rr.Quaternion(xyzw=rotation_xyzw),
                from_parent=False,
                axis_length=3.0,
            ),
            static=True,
        )
    else:
        rr.log(
            f"{prefix}/ego_vehicle/{sensor_name}",
            rr.Transform3D(
                translation=calibrated_sensor["translation"],
                rotation=rr.Quaternion(xyzw=rotation_xyzw),
                from_parent=False,
            ),
            static=True,
        )
    if len(calibrated_sensor.get("camera_intrinsic", [])) != 0:
        rr.log(
            f"{prefix}/ego_vehicle/{sensor_name}",
            rr.Pinhole(
                image_from_camera=calibrated_sensor["camera_intrinsic"],
                width=sample_data["width"],
                height=sample_data["height"],
            ),
            static=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizes the nuScenes dataset + predicted 3D boxes using Rerun.")
    parser.add_argument(
        "--root-dir",
        type=pathlib.Path,
        default=DATASET_DIR,
        help="Root directory of nuScenes dataset",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        nargs='+',
        required=True,
        help="Scene names to visualize (e.g. '2021_08_23_22_31_01_939' '2021_08_23_22_31_01_948')",
    )
    parser.add_argument("--dataset-version", type=str, default="v1.0-mini", help="Which nuScenes dataset version to use")
    parser.add_argument(
        "--seconds",
        type=float,
        default=float("inf"),
        help="If specified, limits the number of seconds logged",
    )
    parser.add_argument(
        "--predroot",
        type=str,
        default=None,
        help="Path to your prediction pkl file (e.g. results.pkl)",
    )

    rr.script_add_args(parser)
    args = parser.parse_args()

    # Initialize nuScenes once
    nusc = NuScenes(version=args.dataset_version, dataroot=args.root_dir, verbose=True)

    # Ensure all scenes are available
    for scene_name in args.scene_name:
        ensure_scene_available(nusc, scene_name)

    # Parse predictions (assuming predictions are only for the ego vehicle)
    pred_dict = parse_predictions_multitask_pkl(args.predroot)

    # Prepare sensor views for the blueprint
    sensor_views = []
    blueprint_contents = []
    for idx, scene_name in enumerate(args.scene_name):
        if idx == 0:
            prefix = "world"
            # Ego vehicle sensor views
            sensor_names = nuscene_sensor_names(nusc, scene_name)
            sensor_views.extend([
                rrb.Spatial2DView(
                    name=sensor_name,
                    origin=f"{prefix}/ego_vehicle/{sensor_name}",
                    contents=["$origin/**", f"{prefix}/anns", f"{prefix}/ego_vehicle/LIDAR_TOP/preds"],
                )
                for sensor_name in sensor_names
            ])
        else:
            sender_prefix = f"world/sender_{idx - 1:02d}"
            # Senders only have LIDAR and annotations
            sensor_views.append(
                rrb.Spatial2DView(
                    name=f"LIDAR_TOP_sender_{idx - 1:02d}",
                    origin=f"{sender_prefix}/ego_vehicle/LIDAR_TOP",
                    contents=[f"{sender_prefix}/ego_vehicle/LIDAR_TOP/**", f"{sender_prefix}/anns"],
                    overrides={f"{sender_prefix}/anns": [rr.components.FillModeBatch("majorwireframe")]},
                )
            )

    # Define the blueprint
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="3D",
                    origin="world",
                    defaults=[rr.components.ImagePlaneDistance(5.0)],
                    overrides={
                        "world/anns": [rr.components.FillModeBatch("solid")],
                        "world/ego_vehicle/LIDAR_TOP/preds": [rr.components.FillModeBatch("solid")],
                    },
                ),
                rrb.Vertical(
                    rrb.TextDocumentView(origin="description", name="Description"),
                    row_shares=[1, 1],
                ),
                column_shares=[3, 1],
            ),
            rrb.Grid(*sensor_views),
            row_shares=[4, 2],
        ),
        rrb.TimePanel(state="collapsed"),
    )

    rr.script_setup(args, "rerun_example_nuscenes_pred_multiple_scenes", default_blueprint=blueprint)

    rr.log(
        "description",
        rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN),
        timeless=True,
    )

    # Log each scene
    for idx, scene_name in enumerate(args.scene_name):
        if idx == 0:
            prefix = "world"
            log_nuscenes(nusc, scene_name, max_time_sec=args.seconds, pred_dict=pred_dict, prefix=prefix, log_pred=True)
        else:
            prefix = f"world/sender_{idx - 1:02d}"
            log_nuscenes(nusc, scene_name, max_time_sec=args.seconds, pred_dict={}, prefix=prefix, log_pred=False)

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
