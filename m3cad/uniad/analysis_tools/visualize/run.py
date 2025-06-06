import cv2
import torch
import argparse
import os
import re
import glob
import sys
from tqdm import tqdm
import imageio
import numpy as np
import mmcv
import matplotlib
import matplotlib.pyplot as plt
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils import splits
from pyquaternion import Quaternion
from mmcv.datasets.nuscenes_e2e_dataset import obtain_map_info
from mmcv.datasets.eval_utils.map_api import NuScenesMap
from PIL import Image
from m3cad.uniad.analysis_tools.visualize.utils import color_mapping, AgentPredictionData
from m3cad.uniad.analysis_tools.visualize.render.bev_render import BEVRender
from m3cad.uniad.analysis_tools.visualize.render.cam_render import CameraRender


class Visualizer:
    """
    BaseRender class
    """

    def __init__(
            self,
            dataroot='/mnt/petrelfs/yangjiazhi/e2e_proj/data/nus_mini',
            version='v1.0-mini',
            predroot=None,
            with_occ_map=False,
            with_map=False,
            with_planning=False,
            with_pred_box=True,
            with_pred_traj=False,
            show_gt_boxes=False,
            show_lidar=False,
            show_command=False,
            show_hd_map=False,
            show_sdc_car=False,
            show_sdc_traj=False,
            show_legend=False):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.predict_helper = PredictHelper(self.nusc)
        self.with_occ_map = with_occ_map
        self.with_map = with_map
        self.with_planning = with_planning
        self.show_lidar = show_lidar
        self.show_command = show_command
        self.show_hd_map = show_hd_map
        self.show_sdc_car = show_sdc_car
        self.show_sdc_traj = show_sdc_traj
        self.show_legend = show_legend
        self.with_pred_traj = with_pred_traj
        self.with_pred_box = with_pred_box
        self.veh_id_list = [0, 1, 2, 3, 4, 6, 7]
        self.use_json = '.json' in predroot
        self.token_set = set()
        self.predictions = self._parse_predictions_multitask_pkl(predroot)
        self.bev_render = BEVRender(show_gt_boxes=show_gt_boxes)
        self.cam_render = CameraRender(show_gt_boxes=show_gt_boxes)

        if self.show_hd_map:
            self.nusc_maps = {location: NuScenesMap(dataroot=dataroot, map_name=location) for location in 
                            ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD']}

    def _parse_predictions_multitask_pkl(self, predroot):
        outputs = mmcv.fileio.io.load(predroot)
        outputs = outputs['bbox_results']
        prediction_dict = dict()
        for k in range(len(outputs)):
            token = outputs[k]['token']
            self.token_set.add(token)
            if self.show_sdc_traj:
                outputs[k]['boxes_3d'].tensor = torch.cat(
                    [outputs[k]['boxes_3d'].tensor, outputs[k]['sdc_boxes_3d'].tensor], dim=0)
                outputs[k]['scores_3d'] = torch.cat(
                    [outputs[k]['scores_3d'], outputs[k]['sdc_scores_3d']], dim=0)
                outputs[k]['labels_3d'] = torch.cat([outputs[k]['labels_3d'], torch.zeros(
                    (1,), device=outputs[k]['labels_3d'].device)], dim=0)
            # detection
            bboxes = outputs[k]['boxes_3d']
            scores = outputs[k]['scores_3d']
            labels = outputs[k]['labels_3d']

            track_scores = scores.cpu().detach().numpy()
            track_labels = labels.cpu().detach().numpy()
            track_boxes = bboxes.tensor.cpu().detach().numpy()

            track_centers = bboxes.gravity_center.cpu().detach().numpy()
            track_dims = bboxes.dims.cpu().detach().numpy()
            track_yaw = bboxes.yaw.cpu().detach().numpy()

            if 'track_ids' in outputs[k]:
                track_ids = outputs[k]['track_ids'].cpu().detach().numpy()
            else:
                track_ids = None

            # speed
            track_velocity = bboxes.tensor.cpu().detach().numpy()[:, -2:]

            # trajectories
            if 'traj' in outputs[k]:
                trajs = outputs[k]['traj'].numpy()
                traj_scores = outputs[k]['traj_scores'].numpy()
            else:
                # 如果没有 'traj'，创建占位符或跳过相关处理
                trajs = None
                traj_scores = None

            predicted_agent_list = []

            # occflow
            if self.with_occ_map:
                if 'topk_query_ins_segs' in outputs[k]['occ']:
                    occ_map = outputs[k]['occ']['topk_query_ins_segs'][0].cpu(
                    ).numpy()
                else:
                    occ_map = np.zeros((1, 5, 200, 200))
            else:
                occ_map = None

            occ_idx = 0
            for i in range(track_scores.shape[0]):
                if track_scores[i] < 0.25:
                    continue
                if occ_map is not None and track_labels[i] in self.veh_id_list:
                    occ_map_cur = occ_map[occ_idx, :, ::-1]
                    occ_idx += 1
                else:
                    occ_map_cur = None
                if track_ids is not None:
                    if i < len(track_ids):
                        track_id = track_ids[i]
                    else:
                        track_id = 0
                else:
                    track_id = None

                # 创建 AgentPredictionData 时，处理 trajs 和 traj_scores 可能为 None 的情况
                if trajs is not None and traj_scores is not None:
                    traj = trajs[i]
                    traj_score = traj_scores[i]
                else:
                    traj = None
                    traj_score = None

                predicted_agent_list.append(
                    AgentPredictionData(
                        track_scores[i],
                        track_labels[i],
                        track_centers[i],
                        track_dims[i],
                        track_yaw[i],
                        track_velocity[i],
                        traj,
                        traj_score,
                        pred_track_id=track_id,
                        pred_occ_map=occ_map_cur,
                        past_pred_traj=None
                    )
                )

            if self.with_map:
                map_thres = 0.7
                score_list = outputs[k]['pts_bbox']['score_list'].cpu().numpy().transpose([
                    1, 2, 0])
                predicted_map_seg = outputs[k]['pts_bbox']['lane_score'].cpu().numpy().transpose([
                    1, 2, 0])  # H, W, C
                predicted_map_seg[..., -1] = score_list[..., -1]
                predicted_map_seg = (predicted_map_seg > map_thres) * 1.0
                predicted_map_seg = predicted_map_seg[::-1, :, :]
            else:
                predicted_map_seg = None

            if self.with_planning:
                # detection
                bboxes = outputs[k]['sdc_boxes_3d']
                scores = outputs[k]['sdc_scores_3d']
                labels = 0

                track_scores = scores.cpu().detach().numpy()
                track_labels = labels
                track_boxes = bboxes.tensor.cpu().detach().numpy()

                track_centers = bboxes.gravity_center.cpu().detach().numpy()
                track_dims = bboxes.dims.cpu().detach().numpy()
                track_yaw = bboxes.yaw.cpu().detach().numpy()
                track_velocity = bboxes.tensor.cpu().detach().numpy()[:, -2:]

                if self.show_command and 'command' in outputs[k]:
                    command = outputs[k]['command'][0].cpu().detach().numpy()
                else:
                    command = None

                if 'planning_traj' in outputs[k]:
                    planning_traj = outputs[k]['planning_traj'][0].cpu().detach().numpy()
                else:
                    planning_traj = None

                planning_agent = AgentPredictionData(
                    track_scores[0],
                    track_labels,
                    track_centers[0],
                    track_dims[0],
                    track_yaw[0],
                    track_velocity[0],
                    planning_traj,
                    1,
                    pred_track_id=-1,
                    pred_occ_map=None,
                    past_pred_traj=None,
                    is_sdc=True,
                    command=command,
                )
                predicted_agent_list.append(planning_agent)
            else:
                planning_agent = None

            prediction_dict[token] = dict(predicted_agent_list=predicted_agent_list,
                                        predicted_map_seg=predicted_map_seg,
                                        predicted_planning=planning_agent)
        return prediction_dict

    def visualize_bev(self, sample_token, out_filename, t=None):
        self.bev_render.reset_canvas(dx=1, dy=1)
        self.bev_render.set_plot_cfg()

        if self.show_lidar:
            self.bev_render.show_lidar_data(sample_token, self.nusc)
        if self.bev_render.show_gt_boxes:
            self.bev_render.render_anno_data(
                sample_token, self.nusc, self.predict_helper)
        if self.with_pred_box:
            self.bev_render.render_pred_box_data(
                self.predictions[sample_token]['predicted_agent_list'])
        if self.with_pred_traj:
            self.bev_render.render_pred_traj(
                self.predictions[sample_token]['predicted_agent_list'])
        if self.with_map:
            self.bev_render.render_pred_map_data(
                self.predictions[sample_token]['predicted_map_seg'])
        if self.with_occ_map:
            self.bev_render.render_occ_map_data(
                self.predictions[sample_token]['predicted_agent_list'])
        if self.with_planning:
            self.bev_render.render_pred_box_data(
                [self.predictions[sample_token]['predicted_planning']])
            self.bev_render.render_planning_data(
                self.predictions[sample_token]['predicted_planning'], show_command=self.show_command)
        if self.show_hd_map:
            self.bev_render.render_hd_map(
                self.nusc, self.nusc_maps, sample_token)
        if self.show_sdc_car:
            self.bev_render.render_sdc_car()
        if self.show_legend:
            self.bev_render.render_legend()
        self.bev_render.save_fig(out_filename + '.jpg')

    def visualize_cam(self, sample_token, out_filename):
        self.cam_render.reset_canvas(dx=2, dy=3, tight_layout=True)
        self.cam_render.render_image_data(sample_token, self.nusc)
        self.cam_render.render_pred_track_bbox(
            self.predictions[sample_token]['predicted_agent_list'], sample_token, self.nusc)
        if self.with_pred_traj:
            self.cam_render.render_pred_traj(
                self.predictions[sample_token]['predicted_agent_list'], sample_token, self.nusc, render_sdc=self.with_planning)
        self.cam_render.save_fig(out_filename + '_cam.jpg')

    def combine(self, out_filename):
        # pass
        bev_image = cv2.imread(out_filename + '.jpg')
        cam_image = cv2.imread(out_filename + '_cam.jpg')
        merge_image = cv2.hconcat([cam_image, bev_image])
        cv2.imwrite(out_filename + '.jpg', merge_image)
        os.remove(out_filename + '_cam.jpg')

    def to_video(self, folder_path, output_path, fps=4, downsample=1):
        def extract_number(file_name):
            match = re.match(r'(\d+)', file_name)
            return int(match.group()) if match else float('inf')

        files = os.listdir(folder_path)
        imgs_path = [file for file in files if file.endswith(('.png', '.jpg', '.jpeg'))]
        imgs_path.sort(key=extract_number)

        if not imgs_path:
            print(f"No images found in {folder_path}. Cannot create video.")
            return

        first_img = cv2.imread(os.path.join(folder_path, imgs_path[0]))
        if first_img is None:
            print("Failed to read the first image. Cannot proceed.")
            return

        h, w = first_img.shape[:2]
        h //= downsample
        w //= downsample
        h = h if h % 2 == 0 else h - 1
        w = w if w % 2 == 0 else w - 1

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for img_name in tqdm(imgs_path, desc="Writing video"):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(img)

        writer.release()
        print(f"Video saved successfully: {output_path}")


def main(args):
    render_cfg = dict(
        with_occ_map=False, # Not working
        with_map=False, # mapformer output, make sure comment the pop function in uniad_e2e.py
        with_planning=False,
        with_pred_box=True,
        with_pred_traj=False,
        show_gt_boxes=True,
        show_lidar=False, 
        show_command=False, # ['TURN RIGHT', 'TURN LEFT', 'KEEP FORWARD']
        show_hd_map=True, # 
        show_sdc_car=True, # showing the car picture (png from source/)
        show_legend=False, # showing the legend (png from source/)
        show_sdc_traj=False # showing sdc bboxes 3d
    )

    viser = Visualizer(version=args.version, predroot=args.predroot, dataroot=args.dataroot, **render_cfg)

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    val_splits = splits.val

    scene_token_to_name = dict()
    for i in range(len(viser.nusc.scene)):
        scene_token_to_name[viser.nusc.scene[i]['token']] = viser.nusc.scene[i]['name']

    for i in range(len(viser.nusc.sample)):
        sample_token = viser.nusc.sample[i]['token']
        scene_token = viser.nusc.sample[i]['scene_token']

        if scene_token_to_name[scene_token] not in val_splits:
            continue

        if sample_token not in viser.token_set:
            print(i, sample_token, 'not in prediction pkl!')
            continue
        

        viser.visualize_bev(sample_token, os.path.join(args.out_folder, str(i).zfill(3)))

        if args.project_to_cam:
            viser.visualize_cam(sample_token, os.path.join(args.out_folder, str(i).zfill(3)))
            viser.combine(os.path.join(args.out_folder, str(i).zfill(3)))

    viser.to_video(args.out_folder, args.demo_video, fps=30, downsample=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='data/m3cad_carla_ue5', help='Path to the dataset root directory')
    parser.add_argument('--version', default='v1.0-trainval', help='Dataset version (e.g., v1.0-mini, v1.0-trainval, v1.0-test)')
    parser.add_argument('--predroot', default='output/results.pkl', help='Path to results.pkl')
    parser.add_argument('--out_folder', default='output_vis', help='Output folder path')
    parser.add_argument('--demo_video', default='mini_val_final.avi', help='Demo video name')
    parser.add_argument('--project_to_cam', action='store_true', help='Project to cam (default: False)')
    args = parser.parse_args()
    main(args)
