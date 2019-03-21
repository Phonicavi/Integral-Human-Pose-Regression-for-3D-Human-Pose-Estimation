# ------------------------------------------------------------------------------
# QiuFeng@2019-03-22
# ------------------------------------------------------------------------------
from __future__ import absolute_import

import os.path as osp
import numpy as np
import h5py
import random
import cv2
from pyquaternion.quaternion import Quaternion
from up3d_cache_storage import up_pure2d_base
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

EPS = 1E-5


class UP3D(Dataset):
    def __init__(self, data_label, const_dict, normalize_dict, img_size=256, seg_size=256, start_id=0, end_id=None):
        self.data_dir = osp.join('/mnt/lustre/share/qiufeng/dataset', 'up')
        self.proc_dir = osp.join(self.data_dir, './pproc/img/')
        assert osp.exists(self.data_dir) and osp.exists(self.proc_dir)
        self.data_label = data_label
        self.const_dict = const_dict
        self.normalize_dict = normalize_dict
        self.transform = None
        if isinstance(self.const_dict, dict):
            self.K_mean = self.const_dict['K_mean']
            self.K_std = self.const_dict['K_std']
            self.dist_mean = self.const_dict['dist_mean']
            self.dist_std = self.const_dict['dist_std']
        if isinstance(self.normalize_dict, dict):
            self.pixel_mean = self.normalize_dict['pixel_mean']
            self.pixel_std = self.normalize_dict['pixel_std']
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.pixel_mean,
                                     std=self.pixel_std)])
        self.img_size = img_size
        self.seg_size = seg_size
        self.kernel_dataset = up_pure2d_base(data_dir=self.data_dir, data_label=data_label, start_id=start_id, end_id=end_id,
                                             refresh_cache=False, refresh_img=False, refresh_mat=False)
        self.start_id, self.end_id = self.kernel_dataset.start_id, self.kernel_dataset.end_id
        self.img_idx = self.kernel_dataset.img_idx
        self.l_gt = np.array([
            [134.05063],
            [238.46465],
            [255.34024],
            [116.07932],
            [148.83488],
            [115.00086],
            [449.2047 ],
            [445.58194],
            [158.6467 ],
            [279.76044],
            [249.52332],
            [118.74958]], dtype=np.float32)

        self.joint_name = (
            'Pelvis', 'Torso', 'Neck', 'Nose', 'Head',
            'L_Hip', 'L_Knee', 'L_Ankle', 'L_Foot',
            'R_Hip', 'R_Knee', 'R_Ankle', 'R_Foot',
            'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand',
            'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand',
        )
        self.joint_num = 17
        self.skeleton = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7),
            (0, 8), (8, 9), (9, 10),
            (2, 11), (11, 12), (12, 13),
            (2, 14), (14, 15), (15, 16),
        ]
        self.seg_num = 15  # entire silhouette included
        self.root_idx = 0

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, index):
        running_idx = self.img_idx[index]
        img_ret, ann_ret, s2d_ret, bbox_ret, is_valid = \
            self.kernel_dataset.load_np(running_index=running_idx, img_size=self.img_size, seg_size=self.seg_size)
        if self.transform is not None:
            img_ret = self.transform(img_ret.transpose((1, 2, 0)))
        # ann_ret = ann_ret.astype(np.float32)
        # s2d21 = np.zeros((self.joint_num, 2), dtype=np.float32)
        # s2d21[:3] = s2d_ret[:3]
        # s2d21[4:8] = s2d_ret[3:7]
        # s2d21[9:12] = s2d_ret[7:10]
        # s2d21[13:16] = s2d_ret[10:13]
        # s2d21[17:20] = s2d_ret[13:16]
        # s2d_vis = np.ones((self.joint_num, 1))
        # for j in range(ann_ret.shape[0]):
        #     s2d_vis[j] *= ((s2d21[j, 0] >= 0) &
        #                    (s2d21[j, 0] < self.img_size) &
        #                    (s2d21[j, 1] >= 0) &
        #                    (s2d21[j, 1] < self.img_size))
        # s2d_vis = (s2d_vis > 0).astype(np.float32)
        # s2d_vis[3], s2d_vis[8], s2d_vis[12], s2d_vis[16], s2d_vis[20] = 0., 0., 0., 0., 0.
        # K = self.const_dict['K_mean'].astype(np.float32)
        # dist = self.const_dict['dist_mean'].astype(np.float32)
        # # bbox: ellip24-style
        # bbox_ret = np.array([self.img_size / 2., self.img_size / 2., self.img_size], dtype=np.float32)
        # l12 = self.l_gt.copy().astype(np.float32)
        return img_ret#, ann_ret, s2d21, s2d_vis, K, dist, bbox_ret, l12, is_valid


def unified_cameras(h5_obj):
    cm = {}
    Ks, dists = list(), list()
    for subj_id in [1, 5, 6, 7, 8, 9, 11]:
        stag = 'subject%d' % subj_id
        if stag not in cm:
            cm[stag] = {}
        for cam_id in [1, 2, 3, 4]:
            ctag = 'camera%d' % cam_id
            if ctag not in cm[stag]:
                cm[stag][ctag] = {}
            for item in ['f', 'c', 'k', 'p']:
                cm[stag][ctag][item] = np.reshape(h5_obj[stag][ctag][item].value, (-1))
            f = cm[stag][ctag]['f']
            c = cm[stag][ctag]['c']
            dist_k = cm[stag][ctag]['k']
            dist_p = cm[stag][ctag]['p']
            K = np.array([[f[0], 0, c[0]], [0, f[1], c[1]], [0, 0, 1]], dtype=np.float32)
            dist_coeffs = np.array([dist_k[0], dist_k[1], dist_p[0], dist_p[1], dist_k[2]], dtype=np.float32)
            cm[stag][ctag]['K'] = K
            cm[stag][ctag]['dist'] = dist_coeffs
            Ks.append(K)
            dists.append(dist_coeffs)
    Ks = np.stack(Ks, axis=0)
    dists = np.stack(dists, axis=0)
    return Ks, dists


if __name__ == '__main__':
    import warnings

    warnings.simplefilter(action='ignore', category=FutureWarning)
    data_label = "trainval"
    pixel_mean = (0.485, 0.456, 0.406)
    pixel_std = (0.229, 0.224, 0.225)
    const_dict = {
        'K_mean': np.array([[1.1473444e+03, 0.0000000e+00, 5.1404352e+02],
                            [0.0000000e+00, 1.1462365e+03, 5.0670016e+02],
                            [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]], dtype=np.float32),
        'K_std': np.array([[2.0789747, 0.0000001, 3.9817653],
                           [0.0000001, 2.0353518, 5.6948705],
                           [0.0000001, 0.0000001, 0.0000001]], dtype=np.float32),
        'dist_mean': np.array([-0.20200874,  0.24049881, -0.00168468, -0.00042439, -0.00191587], dtype=np.float32),
        'dist_std': np.array([0.00591341, 0.01386873, 0.00071674, 0.00116201, 0.00564366], dtype=np.float32),
    }
    normalize_dict = {
        'pixel_mean': pixel_mean,
        'pixel_std': pixel_std,
    }
    dataset_ = UP3D(data_label=data_label, const_dict=const_dict,
                                normalize_dict=normalize_dict, img_size=256)
    dataloader = DataLoader(
        dataset=dataset_,
        batch_size=4,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    print('%s => %d' % (data_label, len(dataset_)))
    print('%s => %d' % (data_label, len(dataloader)))

    import time
    import sys

    last_iter_time = time.time()
    start_time = last_iter_time
    monitor_count = 10

    for i, (img_patch, seg_patch, joint_img, joint_vis, K, dist, bbox, l12, _) in enumerate(dataloader):
        if i >= len(dataloader):
            break
        if not (i % monitor_count):
            print(i, '->', time.time() - last_iter_time)
            sys.stdout.flush()
            last_iter_time = time.time()
        print('img_patch:', img_patch.shape)
        print('seg_patch:', seg_patch.shape)
        print('joint_img:', joint_img.shape, joint_img[0])
        print('bbox:', bbox.shape, bbox[0])
        print('l12:', l12.shape, l12[0])

        s_img = img_patch[0].detach().cpu().numpy().transpose((1, 2, 0))
        s_joint = joint_img[0].detach().cpu().numpy()
        print(s_img.shape)
        for j in range(3, 14):
            p1 = s_joint[2]
            p2 = s_joint[4]
            print(p1, p2)
            s_img = cv2.line(s_img, (p1[0], p1[1]), (p2[0], p2[1]), color=(255, 0, 0), thickness=5)
            break
        cv2.imshow("", s_img)
        cv2.waitKey()
