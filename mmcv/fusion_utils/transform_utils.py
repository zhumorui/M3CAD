import torch
import numpy as np


def transform_coordinates(
    pos: torch.Tensor, 
    yaw: torch.Tensor, 
    ego_x: float, 
    ego_y: float, 
    ego_yaw: float, 
    sender_x: float, 
    sender_y: float, 
    sender_yaw: float
):
    """
    Transforms 3D points and yaw angles from the sender vehicle's LiDAR frame to the ego vehicle's LiDAR frame.

    Args:
        pos (torch.Tensor): N x 3 tensor of 3D points in the sender vehicle's LiDAR frame.
        yaw (torch.Tensor): N x 1 tensor of yaw angles (in radians) for each point in the sender vehicle's LiDAR frame.
        ego_x (float): x position of the ego vehicle in the global coordinate frame.
        ego_y (float): y position of the ego vehicle in the global coordinate frame.
        ego_yaw (float): yaw angle (in degrees) of the ego vehicle in the global coordinate frame.
        sender_x (float): x position of the sender vehicle in the global coordinate frame.
        sender_y (float): y position of the sender vehicle in the global coordinate frame.
        sender_yaw (float): yaw angle (in degrees) of the sender vehicle in the global coordinate frame.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Transformed 3D points and yaw angles in the ego vehicle's LiDAR frame.
    """
    device = pos.device
    
    # Convert angles to radians
    sender_yaw_rad = torch.tensor(sender_yaw * np.pi / 180.0, device=device, dtype=torch.float32)
    ego_yaw_rad = torch.tensor(ego_yaw * np.pi / 180.0, device=device, dtype=torch.float32)

    # LiDAR to IMU transformation matrix and translation (sender)
    R_lidar_to_imu = torch.tensor([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ], device=device, dtype=torch.float32)
    
    t_lidar_to_imu = torch.tensor([
        [0],
        [0],
        [0]
    ], device=device, dtype=torch.float32)
    
    pos = pos.to(torch.float32)
    yaw = yaw.to(torch.float32)
    
    # Transform positions from LiDAR to IMU frame
    pos_imu = torch.matmul(pos, R_lidar_to_imu.T)
    pos_imu = pos_imu + t_lidar_to_imu.T
    
    # IMU to Global transformation
    cos_sender = torch.cos(sender_yaw_rad)
    sin_sender = torch.sin(sender_yaw_rad)
    
    R_imu_to_global = torch.tensor([
        [cos_sender, -sin_sender, 0],
        [sin_sender, cos_sender, 0],
        [0, 0, 1]
    ], device=device, dtype=torch.float32)
    
    pos_global = torch.matmul(pos_imu, R_imu_to_global.T)
    pos_global[:, 0] += sender_x
    pos_global[:, 1] += sender_y
    
    yaw_global = yaw + sender_yaw_rad
    
    # Global to ego IMU transformation
    pos_global[:, 0] -= ego_x
    pos_global[:, 1] -= ego_y
    
    cos_ego = torch.cos(-ego_yaw_rad)
    sin_ego = torch.sin(-ego_yaw_rad)
    
    R_global_to_ego_imu = torch.tensor([
        [cos_ego, -sin_ego, 0],
        [sin_ego, cos_ego, 0],
        [0, 0, 1]
    ], device=device, dtype=torch.float32)
    
    pos_ego_imu = torch.matmul(pos_global, R_global_to_ego_imu.T)
    
    # IMU to LiDAR transformation matrix (ego)
    R_imu_to_lidar = torch.tensor([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], device=device, dtype=torch.float32)
    
    t_imu_to_lidar = torch.tensor([
        [0],
        [0],
        [0]
    ], device=device, dtype=torch.float32)
    
    transformed_pos = torch.matmul(pos_ego_imu, R_imu_to_lidar.T)
    transformed_pos = transformed_pos + t_imu_to_lidar.T
    transformed_yaw = yaw_global - ego_yaw_rad

    return transformed_pos, transformed_yaw