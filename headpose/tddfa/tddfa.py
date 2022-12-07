__author__ = 'cleardusk'

import yaml
import pathlib
import os.path as osp
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
from .utils.onnx import convert_to_onnx
from .utils.io import _load
from .utils.functions import (
    crop_img, parse_roi_box_from_bbox, parse_roi_box_from_landmark,
)
from .utils.tddfa_util import _parse_param, similar_transform
from .bfm.bfm import BFMModel
from .bfm.bfm_onnx import convert_bfm_to_onnx
import cv2
from math import cos, sin, atan2, asin, sqrt

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)
main_file = pathlib.Path(__file__).parent


class TDDFA_ONNX(object):
    """TDDFA_ONNX: the ONNX version of Three-D Dense Face Alignment (TDDFA)"""

    def __init__(self, **kvs):
        # torch.set_grad_enabled(False)

        # load onnx version of BFM
        bfm_fp = f'{main_file}/configs/bfm_noneck_v3.pkl'
        bfm_onnx_fp = bfm_fp.replace('.pkl', '.onnx')
        if not osp.exists(bfm_onnx_fp):
            convert_bfm_to_onnx(
                bfm_onnx_fp,
                shape_dim=kvs.get('shape_dim', 40),
                exp_dim=kvs.get('exp_dim', 10)
            )
        self.bfm_session = onnxruntime.InferenceSession(bfm_onnx_fp, None)

        # load for optimization
        bfm = BFMModel(bfm_fp, shape_dim=kvs.get('shape_dim', 40), exp_dim=kvs.get('exp_dim', 10))
        self.tri = bfm.tri
        self.u_base, self.w_shp_base, self.w_exp_base = bfm.u_base, bfm.w_shp_base, bfm.w_exp_base

        # config
        self.gpu_mode = kvs.get('gpu_mode', False)
        self.gpu_id = kvs.get('gpu_id', 0)
        self.size = kvs.get('size', 120)

        param_mean_std_fp = kvs.get(
            'param_mean_std_fp', f'{main_file}/configs/param_mean_std_62d_{self.size}x{self.size}.pkl'
        )

        onnx_fp = kvs.get('onnx_fp', kvs.get('checkpoint_fp').replace('.pth', '.onnx'))
        onnx_fp = f'{main_file}/{onnx_fp}'
        # convert to onnx online if not existed
        if onnx_fp is None or not osp.exists(onnx_fp):
            print(f'{onnx_fp} does not exist, try to convert the `.pth` version to `.onnx` online')
            onnx_fp = convert_to_onnx(**kvs)

        self.session = onnxruntime.InferenceSession(onnx_fp, None)

        # params normalization config
        r = _load(param_mean_std_fp)
        self.param_mean = r.get('mean')
        self.param_std = r.get('std')

    def __call__(self, img_ori, objs, **kvs):
        # Crop image, forward to get the param
        param_lst = []
        roi_box_lst = []

        crop_policy = kvs.get('crop_policy', 'box')
        for obj in objs:
            if crop_policy == 'box':
                # by face box
                roi_box = parse_roi_box_from_bbox(obj)
            elif crop_policy == 'landmark':
                # by landmarks
                roi_box = parse_roi_box_from_landmark(obj)
            else:
                raise ValueError(f'Unknown crop policy {crop_policy}')

            roi_box_lst.append(roi_box)
            img = crop_img(img_ori, roi_box)
            img = cv2.resize(img, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
            img = (img - 127.5) / 128.

            inp_dct = {'input': img}

            param = self.session.run(None, inp_dct)[0]
            param = param.flatten().astype(np.float32)
            param = param * self.param_std + self.param_mean  # re-scale
            param_lst.append(param)

        return param_lst, roi_box_lst

    def recon_vers(self, param_lst, roi_box_lst, **kvs):
        dense_flag = kvs.get('dense_flag', False)
        size = self.size

        ver_lst = []
        for param, roi_box in zip(param_lst, roi_box_lst):
            R, offset, alpha_shp, alpha_exp = _parse_param(param)
            if dense_flag:
                inp_dct = {
                    'R': R, 'offset': offset, 'alpha_shp': alpha_shp, 'alpha_exp': alpha_exp
                }
                pts3d = self.bfm_session.run(None, inp_dct)[0]
                pts3d = similar_transform(pts3d, roi_box, size)
            else:
                pts3d = R @ (self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @ alpha_exp). \
                    reshape(3, -1, order='F') + offset
                pts3d = similar_transform(pts3d, roi_box, size)

            ver_lst.append(pts3d)

        return ver_lst


def calc_hypotenuse(pts):
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    return llength / 3



def P2sRt(P):
    """ decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    """
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d


def matrix2angle(R):
    """ compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv
    todo: check and debug
     Args:
         R: (3,3). rotation matrix
     Returns:
         x: yaw
         y: pitch
         z: roll
     """
    if R[2, 0] > 0.998:
        z = 0
        x = np.pi / 2
        y = z + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        z = 0
        x = -np.pi / 2
        y = -z + atan2(R[0, 1], R[0, 2])
    else:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    return x, y, z


def calc_pose(param):
    P = param[:12].reshape(3, -1)  # camera matrix
    s, R, t3d = P2sRt(P)
    P = np.concatenate((R, t3d.reshape(3, -1)), axis=1)  # without scale
    pose = matrix2angle(R)
    pose = [p * 180 / np.pi for p in pose]

    return P, pose


def build_camera_box(rear_size=90):
    point_3d = []
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = int(4 / 3 * rear_size)
    front_depth = int(4 / 3 * rear_size)
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

    return point_3d


def plot_pose_box(img, P, ver, color=(40, 255, 0), line_width=2):
    """ Draw a 3D box as annotation of pose.
    Ref:https://github.com/yinguobing/head-pose-estimation/blob/master/pose_estimator.py
    Args:
        img: the input image
        P: (3, 4). Affine Camera Matrix.
        kpt: (2, 68) or (3, 68)
    """
    llength = calc_hypotenuse(ver)
    point_3d = build_camera_box(llength)
    # Map to 2d image points
    point_3d_homo = np.hstack((point_3d, np.ones([point_3d.shape[0], 1])))  # n x 4
    point_2d = point_3d_homo.dot(P.T)[:, :2]

    point_2d[:, 1] = - point_2d[:, 1]
    point_2d[:, :2] = point_2d[:, :2] - np.mean(point_2d[:4, :2], 0) + np.mean(ver[:2, :27], 1)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)

    return img

def plot_image(img):
    height, width = img.shape[:2]
    plt.figure(figsize=(12, height / width * 12))

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    plt.imshow(img[..., ::-1])
    plt.show()


def viz_pose(img, param_lst, ver_lst, show_flag=False, wfp=None):
    for param, ver in zip(param_lst, ver_lst):
        P, pose = calc_pose(param)
        print(P, pose)
        img = plot_pose_box(img, P, ver)

    if wfp is not None:
        cv2.imwrite(wfp, img)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plot_image(img)

    return img

def load_model():
    main_file = pathlib.Path(__file__).parent
    with open(f'{main_file}/configs/mb1_120x120.yml', 'r') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader) 
    return TDDFA_ONNX(**cfg)


