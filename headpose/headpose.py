import torch
from .tddfa.tddfa import P2sRt
from .tddfa.tddfa import load_model as load_tddfa_model
from .facebox.face_boxes import load_model as load_facebox_model

class HeadposeInference():
    def __init__(self):
        self.tddfa = load_tddfa_model()
        self.faceboxes = load_facebox_model()

    def inference(self, image):
        boxes = self.faceboxes(image)
        param_lst, roi_box_lst = self.tddfa(image, boxes)
        ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)

        Rs, Ts = [], []
        for param, ver in zip(param_lst, ver_lst):
            P = param[:12].reshape(3, -1)  # camera matrix
            s, R, t3d = P2sRt(P)
            Rs.append(R)
            Ts.append(t3d)
        return Rs, Ts

    def batch_inference(self, images):
        Rs, Ts = [], []
        for image in images:
            R, T = self.inference(image)
            Rs.append(R)
            Ts.append(T)
        return Rs, Ts

