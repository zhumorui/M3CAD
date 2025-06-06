import numpy as np
import torch
from einops import rearrange
import matplotlib.pyplot as plt
from datetime import datetime
from .transform_utils import transform_coordinates  


def check_overlap(pos1, pos2, threshold=2.0):
    """Check for overlapping points between two sets of positions
    Args:
        pos1, pos2: position tensors [N, 3]
        threshold: overlap threshold in meters
    Returns:
        overlapping_pairs: list of (idx1, idx2) tuples
    """
    overlapping_pairs = []
    for i in range(len(pos1)):
        for j in range(len(pos2)):
            dist = torch.norm(pos1[i, :2] - pos2[j, :2])
            if dist < threshold:
                overlapping_pairs.append((i, j))
    return overlapping_pairs


def plot_combined_view(sender_pos, sender_yaw, ego_pos, ego_yaw, 
                      transformed_pos, transformed_yaw, bev_embed, backup_sender_bev, sender_bev_embed,
                      fusion_bev_embed, title="Combined View"
                      ):

    """Plot combined visualization"""
    fig, axes = plt.subplots(1, 4, figsize=(32, 8))
    arrow_length = 2.0
    
    #TODO: AXES[0]: plot sender view (original)
    sender_bev = rearrange(backup_sender_bev, '(h w) b c -> b h w c', h=200, w=200)
    sender_bev_0 = sender_bev[0, :, :, 0].detach().cpu()
    extent = [-51.2, 51.2, -51.2, 51.2]
    axes[0].imshow(sender_bev_0, extent=extent, cmap='viridis', origin='lower')

    x_coords = sender_pos[:, 0].cpu().numpy()
    y_coords = sender_pos[:, 1].cpu().numpy()
    z_coords = sender_pos[:, 2].cpu().numpy()
    yaws = sender_yaw.cpu().numpy()
    
    scatter = axes[0].scatter(x_coords, y_coords, c=z_coords, cmap='viridis',
                            marker='o', s=100, alpha=0.6, label='Boxes')
    
    # Add direction arrows for sender
    dx = arrow_length * np.cos(yaws)
    dy = arrow_length * np.sin(yaws)
    for i, (x, y, dx_i, dy_i) in enumerate(zip(x_coords, y_coords, dx, dy)):
        axes[0].arrow(x, y, dx_i, dy_i,
                    head_width=0.5, head_length=0.8, fc='red', ec='red', alpha=0.6)
        axes[0].annotate(f'Box {i+1}', (x, y), xytext=(5, 5), textcoords='offset points')


    #TODO: AXES[1]: plot sender view (transformed)
    sender_bev = rearrange(sender_bev_embed, '(h w) b c -> b h w c', h=200, w=200)
    sender_bev_0 = sender_bev[0, :, :, 0].detach().cpu()
    extent = [-51.2, 51.2, -51.2, 51.2]
    axes[1].imshow(sender_bev_0, extent=extent, cmap='viridis', origin='lower')

    x_coords = transformed_pos[:, 0].cpu().numpy()
    y_coords = transformed_pos[:, 1].cpu().numpy()
    z_coords = transformed_pos[:, 2].cpu().numpy()
    yaws = transformed_yaw.cpu().numpy()
    
    scatter = axes[1].scatter(x_coords, y_coords, c=z_coords, cmap='viridis',
                            marker='o', s=100, alpha=0.6, label='Boxes')
    
    # Add direction arrows for sender
    dx = arrow_length * np.cos(yaws)
    dy = arrow_length * np.sin(yaws)
    for i, (x, y, dx_i, dy_i) in enumerate(zip(x_coords, y_coords, dx, dy)):
        axes[1].arrow(x, y, dx_i, dy_i,
                    head_width=0.5, head_length=0.8, fc='red', ec='red', alpha=0.6)
        axes[1].annotate(f'Box {i+1}', (x, y), xytext=(5, 5), textcoords='offset points')


    #TODO: AXES[2]: plot ego view 
    ego_bev = rearrange(bev_embed, '(h w) b c -> b h w c', h=200, w=200)
    ego_bev_0 = ego_bev[0, :, :, 0].detach().cpu()
    extent = [-51.2, 51.2, -51.2, 51.2]
    axes[2].imshow(ego_bev_0, extent=extent, cmap='viridis', origin='lower')

    x_coords = ego_pos[:, 0].cpu().numpy()
    y_coords = ego_pos[:, 1].cpu().numpy()
    z_coords = ego_pos[:, 2].cpu().numpy()
    yaws = ego_yaw.cpu().numpy()

    scatter = axes[2].scatter(x_coords, y_coords, c=z_coords, cmap='viridis',
                            marker='o', s=100, alpha=0.6, label='Boxes')

    dx = arrow_length * np.cos(yaws)
    dy = arrow_length * np.sin(yaws)
    for i, (x, y, dx_i, dy_i) in enumerate(zip(x_coords, y_coords, dx, dy)):
        axes[2].arrow(x, y, dx_i, dy_i,
                    head_width=0.5, head_length=0.8, fc='red', ec='red', alpha=0.6)
        axes[2].annotate(f'Box {i+1}', (x, y), xytext=(5, 5), textcoords='offset points')

    
    #TODO: AXES[3]: plot combined view
    fusion_bev = rearrange(fusion_bev_embed, '(h w) b c -> b h w c', h=200, w=200)
    fusion_bev_0 = fusion_bev[0, :, :, 0].detach().cpu()
    extent = [-51.2, 51.2, -51.2, 51.2]
    axes[3].imshow(fusion_bev_0, extent=extent, cmap='viridis', origin='lower')

    x_coords_ego = ego_pos[:, 0].cpu().numpy()
    y_coords_ego = ego_pos[:, 1].cpu().numpy()
    z_coords_ego = ego_pos[:, 2].cpu().numpy()
    yaws_ego = ego_yaw.cpu().numpy()
    
    scatter_ego = axes[3].scatter(x_coords_ego, y_coords_ego, c=z_coords_ego, 
                                cmap='viridis', marker='o', s=100, alpha=0.6, 
                                label='Ego Boxes')
    
    # Add direction arrows for ego
    dx_ego = arrow_length * np.cos(yaws_ego)
    dy_ego = arrow_length * np.sin(yaws_ego)
    for x, y, dx_i, dy_i in zip(x_coords_ego, y_coords_ego, dx_ego, dy_ego):
        axes[3].arrow(x, y, dx_i, dy_i,
                    head_width=0.5, head_length=0.8, fc='red', ec='red', alpha=0.6)
    
    # Plot transformed sender boxes
    x_coords_trans = transformed_pos[:, 0].cpu().numpy()
    y_coords_trans = transformed_pos[:, 1].cpu().numpy()
    z_coords_trans = transformed_pos[:, 2].cpu().numpy()
    yaws_trans = transformed_yaw.cpu().numpy()
    
    scatter_trans = axes[3].scatter(x_coords_trans, y_coords_trans, c=z_coords_trans,
                                  cmap='plasma', marker='s', s=100, alpha=0.6,
                                  label='Transformed Sender Boxes')
    
    # Add direction arrows for transformed sender
    dx_trans = arrow_length * np.cos(yaws_trans)
    dy_trans = arrow_length * np.sin(yaws_trans)
    for x, y, dx_i, dy_i in zip(x_coords_trans, y_coords_trans, dx_trans, dy_trans):
        axes[3].arrow(x, y, dx_i, dy_i,
                    head_width=0.5, head_length=0.8, fc='blue', ec='blue', alpha=0.6)
    
    # Check for overlaps
    overlaps = check_overlap(ego_pos, transformed_pos)

    print("num of overlaps", len(overlaps))
    
    # Mark overlapping points
    for ego_idx, trans_idx in overlaps:
        circle = plt.Circle((x_coords_ego[ego_idx], y_coords_ego[ego_idx]), 
                          1.5, fill=False, color='red', linestyle='--',
                          label='Overlap' if ego_idx == overlaps[0][0] else "")
        axes[3].add_artist(circle)
        circle = plt.Circle((x_coords_trans[trans_idx], y_coords_trans[trans_idx]),
                          1.5, fill=False, color='red', linestyle='--')
        axes[3].add_artist(circle)
        
        # Connect overlapping pairs
        axes[3].plot([x_coords_ego[ego_idx], x_coords_trans[trans_idx]],
                    [y_coords_ego[ego_idx], y_coords_trans[trans_idx]],
                    'r--', alpha=0.5)
    
    # Set titles and labels for all subplots
    for idx, subtitle in enumerate(["Original Sender View (LiDAR frame)", 
                                    "Transformed Sender View (LiDAR frame)",
                                    "Ego View (LiDAR frame)", 
                                    "Combined View (LiDAR frame)"]
                                    ):
        
        ax = axes[idx]
        ax.set_xlim(-51.2, 51.2)  
        ax.set_ylim(-51.2, 51.2)
        ax.set_title(subtitle)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig('combined_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_bev(bev_embed, backup_sender_bev, sender_bev_embed=None, fusion_bev_embed=None, 
                 gt_bboxes_3d=None, sender_gt_bboxes_3d=None, sender_img_metas=None, 
                 save_path=None, vis_idx=0, debug=False):
    """
    Visualize BEV embeddings and ground truth boxes.
    
    Args:
        bev_embed (torch.Tensor): Ego vehicle BEV embedding
        sender_bev_embed (torch.Tensor, optional): Sender vehicle BEV embedding
        fusion_bev_embed (torch.Tensor, optional): Fused BEV embedding
        gt_bboxes_3d (list, optional): Ground truth 3D boxes for ego vehicle
        sender_gt_bboxes_3d (list, optional): Ground truth 3D boxes for sender vehicle
        sender_img_metas (list, optional): Metadata for sender vehicle including transformation
        save_path (str, optional): Path to save visualization
        vis_idx (int, optional): Visualization index for batch
        debug (bool, optional): Whether to show debug information
    """

    ego_gt_bboxes_3d_pos = gt_bboxes_3d[0][vis_idx].center
    ego_gt_bboxes_3d_yaw = gt_bboxes_3d[0][vis_idx].yaw

    sender_gt_bboxes_3d_pos = sender_gt_bboxes_3d[0][vis_idx].center
    sender_gt_bboxes_3d_yaw = sender_gt_bboxes_3d[0][vis_idx].yaw


    ego_x, ego_y, ego_yaw, sender_x, sender_y, sender_yaw = sender_img_metas[0]['can_bus'][:6]


    transformed_pos, transformed_yaw = transform_coordinates(
        sender_gt_bboxes_3d_pos,
        sender_gt_bboxes_3d_yaw,
        ego_x, ego_y, ego_yaw,
        sender_x, sender_y, sender_yaw
    )

    plot_combined_view(sender_gt_bboxes_3d_pos, sender_gt_bboxes_3d_yaw,
                  ego_gt_bboxes_3d_pos, ego_gt_bboxes_3d_yaw,
                  transformed_pos, transformed_yaw, bev_embed, backup_sender_bev, sender_bev_embed, fusion_bev_embed)