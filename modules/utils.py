import cv2
import yaml
import torch
import numpy as np

def open_configs(configs):
    out_configs = []
    for config in configs:
        with open(f'configs/{config}.yaml', 'r') as f:
            out_configs.append(yaml.load(f.read(), Loader=yaml.FullLoader))
    return out_configs


def angle2matrix(angles, gradient='false'):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch. positive for looking down.
        y: yaw. positive for looking left. 
        z: roll. positive for tilting head right. 
        gradient(str): whether to compute gradient matrix: dR/d_x,y,z
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    x, y, z = torch.deg2rad(angles[0]), torch.deg2rad(angles[1]), torch.deg2rad(angles[2])
    # x
    Rx=torch.tensor([[1,      0,       0],
                     [0, torch.cos(x), -torch.sin(x)],
                     [0, torch.sin(x),  torch.cos(x)]])
    # y
    Ry=torch.tensor([[ torch.cos(y), 0, torch.sin(y)],
                    [      0, 1,      0],
                    [-torch.sin(y), 0, torch.cos(y)]])
    # z
    Rz=torch.tensor([[torch.cos(z), -torch.sin(z), 0],
                    [torch.sin(z),  torch.cos(z), 0],
                    [     0,       0, 1]])
    R=torch.mm(Rz, torch.mm(Ry, Rx))

    if gradient != 'true':
        return R
    elif gradient == 'true':
        # gradident matrix
        dRxdx = torch.tensor([[0,      0,       0],
                            [0, -torch.sin(x), -torch.cos(x)],
                            [0, torch.cos(x),  -torch.sin(x)]])
        dRdx = torch.mm(Rz, torch.mm(Ry, dRxdx)) * torch.pi/180
        dRydy = torch.tensor([[-torch.sin(y), 0,  torch.cos(y)],
                            [      0, 0,       0],
                            [-torch.cos(y), 0, -torch.sin(y)]])
        dRdy = torch.mm(torch.mm(dRydy, Rx)) * torch.pi/180
        dRzdz = torch.tensor([[-torch.sin(z), -torch.cos(z), 0],
                            [ torch.cos(z), -torch.sin(z), 0],
                            [     0,        0, 0]])
        dRdz = torch.mm(dRzdz, torch.mm(Ry, Rx)) * torch.pi/180
        return R, [dRdx, dRdy, dRdz]


def build_projection_matrix(rear_size, factor=np.sqrt(2)):
    rear_depth = 0
    front_size = front_depth = factor * rear_size

    projections = np.tensor([
        [-rear_size, -rear_size, rear_depth],
        [-rear_size, rear_size, rear_depth],
        [rear_size, rear_size, rear_depth],
        [rear_size, -rear_size, rear_depth],
        [-front_size, -front_size, front_depth],
        [-front_size, front_size, front_depth],
        [front_size, front_size, front_depth],
        [front_size, -front_size, front_depth],
    ], dtype=np.float32)

    return projections


def draw_projection(frame, R, T, landmarks, color, thickness=2):
    # build projection matrix
    landmarks = np.array(landmarks)
    radius = np.max(np.max(landmarks, axis=0) - np.min(landmarks, axis=0)) // 2
    projections = build_projection_matrix(radius)
    # refine rotate matrix
    rotate_matrix = R[:, :2]
    rotate_matrix[:, 1] *= -1

    # 3D -> 2D
    center = np.mean(landmarks[:27], axis=0)
    points = projections @ rotate_matrix + T.squeeze(0)
    points = points.astype(np.int32)

    # draw poly
    cv2.polylines(frame, np.take(points, [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [0, 4], [1, 5], [2, 6], [3, 7],
        [4, 5], [5, 6], [6, 7], [7, 4]
    ], axis=0), False, color, thickness, cv2.LINE_AA)