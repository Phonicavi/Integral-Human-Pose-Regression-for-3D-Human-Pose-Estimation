from __future__ import absolute_import

import numpy as np
import os
import h5py
import pickle
# import ceph
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from up3d_s31_process import robust_person_size, get_crop


class up_pproc:

    def data_up3d_py2to3(self, data_root="/data/dataset/up/up-3d/"):
        """
        only using this function under Python2
        :return: hdf5
        """
        import sys
        if sys.version_info.major >= 3:
            return None
        x_raw = set()
        for label in ["train", "test", "trainval", "val"]:
            with open(os.path.join(data_root, label + '.txt')) as f:
                x_new = set(f.read().splitlines())
                x_raw.update(x_new)
        name_list = list(x_raw)
        name_list.sort()

        image_file, render_light_file, dataset_info, crop_info, quality_info = list(), list(), list(), list(), list()
        rt, j2d, f, pose, betas, t, trans, joints = list(), list(), list(), list(), list(), list(), list(), list()
        for i, fn in enumerate(name_list):
            ids = fn[1:].split('_')[0]
            image_fn = ids + '_image.png'
            image_render_light_fn = ids + '_render_light.png'
            image_file.append(image_fn)
            render_light_file.append(image_render_light_fn)
            with open(os.path.join(data_root, ids + '_dataset_info.txt')) as di_f:
                line = di_f.read().splitlines()[0].split()
                dataset_info.append((line[0], int(line[1])))
            with open(os.path.join(data_root, ids + '_fit_crop_info.txt')) as fci_f:
                line = fci_f.read().splitlines()[0].split()
                crop_info.append([int(line[j]) for j in range(len(line))])
            with open(os.path.join(data_root, ids + '_quality_info.txt')) as qi_f:
                line = qi_f.read().splitlines()
                quality_info.append(line[0])
            pk = pickle.load(open(os.path.join(data_root, ids + '_body.pkl'), 'rb'))
            rt.append(pk['rt'])
            j2d.append(pk['j2d'])
            f.append(pk['f'])
            pose.append(pk['pose'])
            betas.append(pk['betas'])
            t.append(pk['t'])
            trans.append(pk['trans'])
            data = np.load(os.path.join(data_root, ids + '_joints.npy')).transpose()
            joints.append(data)

        hdf5 = h5py.File(os.path.join(data_root, 'annot.h5'), 'w')
        hdf5['images'] = image_file
        hdf5['render_light'] = render_light_file
        hdf5['dataset_info'] = dataset_info
        hdf5['crop_info'] = crop_info
        hdf5['quality_info'] = quality_info
        hdf5['rt'] = rt
        hdf5['j2d'] = j2d
        hdf5['f'] = f
        hdf5['pose'] = pose
        hdf5['betas'] = betas
        hdf5['t'] = t
        hdf5['trans'] = trans
        hdf5['joints'] = joints

    def data_ups31_pproc(self, data_root="/data/dataset/up/up-s31/s31/"):
        x_raw = set()
        for label in ["train", "test", "trainval", "val"]:
            with open(os.path.join(data_root, label + '_31_500_pkg_dorder.txt')) as f:
                x_new = set(f.read().splitlines())
                x_raw.update(x_new)
        name_list = list(x_raw)
        name_list.sort()

        image_file, ann_list, ann_vis_list, factor_list = list(), list(), list(), list()
        for i, line in enumerate(name_list):
            _1, _2, _3 = line.split()
            image_fn = _1.split('/')[-1]
            ann_fn = _2.split('/')[-1]
            ann_vis_fn = ann_fn.split('.')[0] + '_vis.' + ann_fn.split('.')[-1]
            scale = float(_3)
            image_file.append(np.string_(image_fn))
            ann_list.append(np.string_(ann_fn))
            ann_vis_list.append(np.string_(ann_vis_fn))
            factor_list.append(scale)

        hdf5 = h5py.File(os.path.join(data_root, 'annot.h5'), 'w')
        hdf5['images'] = image_file
        hdf5['ann'] = ann_list
        hdf5['ann_vis'] = ann_vis_list
        hdf5['factor'] = factor_list

    def merge_individual_annot(self, root_dir="/data/dataset/up/pproc/img-old/"):
        total_num = 8515
        s2ds = []
        cbxs = []
        info_list = []
        for idx in range(total_num):
            target_path = os.path.join(root_dir, '%05d' % idx)
            with h5py.File(os.path.join(target_path, 'annot.h5')) as f:
                ridx = int(f['index'].value)
                assert ridx == idx
                s2d = f['s2d'].value.astype(np.float32)
                bbox = f['crop_box'].value.astype(np.float32)
                info = f['info'].value
                s2ds.append(s2d)
                cbxs.append(bbox)
                info_list.append(info)
        with h5py.File(os.path.join(root_dir, '../annot.h5'), 'w') as h:
            h['s2d'] = np.array(s2ds, dtype=np.float32)
            h['crop_box'] = np.array(cbxs, dtype=np.float32)
            h['info'] = np.array(info_list)


class up3d_cache_base(Dataset):

    def __init__(self, data_dir, data_label, start_id=0, end_id=None):
        self.data_dir = data_dir
        self.data_label = data_label

        with h5py.File(os.path.join(self.data_dir, 'annot.h5')) as f:
            self.images_fn = f['images'].value.astype(str)
            self.render_light = f['render_light'].value.astype(str)
            self.dataset_info = f['dataset_info'].value
            self.crop_info = f['crop_info'].value
            self.quality_info = f['quality_info'].value
            self.rt = f['rt'].value
            self.j2d = f['j2d'].value
            self.ff = f['f'].value
            self.pose = f['pose'].value
            self.betas = f['betas'].value
            self.t = f['t'].value
            self.trans = f['trans'].value
            self.joints = f['joints'].value
        with open(os.path.join(self.data_dir, self.data_label + '.txt')) as item_list:
            self.item_list = item_list.read().splitlines()

        start_id = max(0, start_id)
        end_id = max(start_id, min(end_id, len(self.item_list))) if end_id else len(self.item_list)
        assert start_id < end_id
        self.start_id, self.end_id = start_id, end_id
        self.img_idx = [int(item[1:6]) for item in self.item_list[self.start_id: self.end_id]]

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, index):
        running_idx = self.img_idx[index]
        img = cv2.imread(os.path.join(self.data_dir, self.images_fn[running_idx]))
        irl = cv2.imread(os.path.join(self.data_dir, self.render_light[running_idx]))
        return img, irl, self.crop_info[running_idx]


class ups31_cache_base(Dataset):

    def __init__(self, data_dir, data_label, start_id=0, end_id=None):
        self.data_dir = data_dir
        self.data_label = data_label

        with h5py.File(os.path.join(self.data_dir, 'annot.h5')) as f:
            self.images_fn = f['images'].value.astype(str)
            self.ann = f['ann'].value.astype(str)
            self.ann_vis = f['ann_vis'].value.astype(str)
            self.factor = f['factor'].value.astype(np.float32)

        with open(os.path.join(self.data_dir, self.data_label + '_31_500_pkg_dorder.txt')) as item_list:
            self.item_list = item_list.read().splitlines()
        self.target_person_size = 500
        self.crop_size = 513

        start_id = max(0, start_id)
        end_id = max(start_id, min(end_id, len(self.item_list))) if end_id else len(self.item_list)
        assert start_id < end_id
        self.start_id = start_id
        self.end_id = end_id
        self.img_idx = [int(item.split()[0].split('/')[-1].split('_')[0]) for item in self.item_list[self.start_id:
                                                                                                     self.end_id]]

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, index):
        running_idx = self.img_idx[index]
        img = cv2.imread(os.path.join(self.data_dir, self.images_fn[running_idx]))
        ann = cv2.imread(os.path.join(self.data_dir, self.ann[running_idx]))
        ann_vis = cv2.imread(os.path.join(self.data_dir, self.ann_vis[running_idx]))
        return img, ann, ann_vis, self.factor[running_idx]


class up_compose_base(Dataset):
    """
    Align key-points (up-3d) & segmentation annot (up-s31)
    """
    def __init__(self, data_dir, data_label, start_id=0, end_id=None):
        self.data_dir = data_dir
        self.data_label = data_label
        self.ds_up3d = up3d_cache_base(data_dir=os.path.join(self.data_dir, './up-3d/'),
                                       data_label=self.data_label, start_id=start_id, end_id=end_id)
        self.ds_ups31 = ups31_cache_base(data_dir=os.path.join(self.data_dir, './up-s31/s31/'),
                                         data_label=self.data_label, start_id=start_id, end_id=end_id)
        self.target_person_size = self.ds_ups31.target_person_size
        self.crop_size = self.ds_ups31.crop_size
        assert self.ds_up3d.start_id == self.ds_ups31.start_id
        assert self.ds_up3d.end_id == self.ds_ups31.end_id
        self.start_id = self.ds_up3d.start_id
        self.end_id = self.ds_up3d.end_id
        assert len(self.ds_up3d) == len(self.ds_ups31)
        self.img_idx = self.ds_up3d.img_idx

        # legacy:
        # self.ellip_order = [[], [8, 21], [7, 20, 31], [27, 28, 29, 30],
        #                     [22, 23], [24, 25],
        #                     [9, 10], [11, 12],
        #                     [19], [17, 18], [15, 16, 14],
        #                     [6], [4, 5], [2, 3, 1]]
        # self.ellip_tags = ["ass", "abdomen", "chest", "head", "L-uleg", "L-lleg", "R-uleg", "R-lleg",
        #                    "L-shoulder", "L-uarm", "L-larm", "R-shoulder", "R-uarm", "R-larm"]
        # ellip24:
        self.ellip_order = [[6, 7, 8, 19, 20, 21, 31], [27, 28, 29, 30],
                            [22, 23], [24, 25], [26],
                            [9, 10], [11, 12], [13],
                            [17, 18], [15, 16], [14],
                            [4, 5], [2, 3], [1]]
        self.ellip_tags = ["body", "head",
                           "L-uleg", "L-lleg", "L-foot",
                           "R-uleg", "R-lleg", "R-foot",
                           "L-uarm", "L-larm", "L-hand",
                           "R-uarm", "R-larm", "R-hand"]

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, index):
        running_idx = self.img_idx[index]
        raw, _, cbx = self.ds_up3d[index]
        img, ann, _, factor = self.ds_ups31[index]
        j2d = self.ds_up3d.j2d[running_idx]
        j3d = self.ds_up3d.joints[running_idx]
        # joints: crop & resize
        orig_w, orig_h = raw.shape[:2]
        crop_w, crop_h = cbx[3] - cbx[2], cbx[5] - cbx[4]
        j2d_w, j2d_h = cbx[0], cbx[1]
        lambda_w, lambda_h = float(j2d_w / crop_w), float(j2d_h / crop_h)
        j2d[:, 0] = j2d[:, 0] / lambda_w + cbx[4]
        j2d[:, 1] = j2d[:, 1] / lambda_h + cbx[2]
        j2d = j2d * factor
        raw_nd = cv2.resize(raw, (int(np.ceil(factor * orig_h * 8.0) / 8.0), int(np.ceil(factor * orig_w * 8.0) / 8.0)),
                            interpolation=cv2.INTER_LINEAR)
        j3d_locates = j3d.transpose().copy()
        j3d_locates = np.vstack((j3d_locates, np.all(j3d_locates > 0, axis=0)[None, :]))
        person_center = np.mean(j3d_locates[:2, j3d_locates[2, :] == 1], axis=1) * factor
        crpu, crpv = get_crop(raw_nd, person_center, crop=self.crop_size)
        j2d[:, 0] = j2d[:, 0] - crpv[0]
        j2d[:, 1] = j2d[:, 1] - crpu[0]
        j3d[:, :2] = j3d[:, :2] * factor
        j3d[:, 0] = j3d[:, 0] - crpv[0]
        j3d[:, 1] = j3d[:, 1] - crpu[0]
        cbx_nd = cbx * factor
        cbx_nd[2:4] = cbx_nd[2:4] - crpu[0]
        cbx_nd[4:6] = cbx_nd[4:6] - crpv[0]
        bbox = cbx_nd[2:6]

        # segments annot
        ann_stack = []
        for i, tag in enumerate(self.ellip_tags):
            assem = self.ellip_order[i]
            ann_i = np.zeros(ann.shape[:2], dtype=np.uint8)
            for j in assem:
                ann_i[(ann[:, :, 0] >= j) & (ann[:, :, 0] < j + 1)] = 255
            ann_stack.append(ann_i)
        silhouette = np.zeros(ann.shape[:2], dtype=np.uint8)
        for seg_silh in ann_stack:
            silhouette = silhouette | seg_silh
        ann_stack.append(silhouette)

        # info
        ds_triv, quality_triv = self.ds_up3d.dataset_info[running_idx], self.ds_up3d.quality_info[running_idx]
        triv = (ds_triv[0], ds_triv[1], quality_triv)

        return img, ann_stack, j3d, j2d, bbox, triv


class up_pure2d_base(Dataset):
    """
    File cached concatenated single image and overall annot hdf5
    """
    def __init__(self, data_dir, data_label, start_id=0, end_id=None,
                 refresh_cache=False, refresh_img=False, refresh_mat=False):
        self.data_dir = data_dir
        self.proc_dir = os.path.join(self.data_dir, './pproc/img/')
        self.overall_hdf5 = os.path.join(self.data_dir, './pproc/annot.h5')
        if not os.path.exists(self.proc_dir):
            os.makedirs(self.proc_dir, exist_ok=True)
        self.data_label = data_label
        self.kernel_dataset = up_compose_base(data_dir=data_dir, data_label=data_label, start_id=start_id, end_id=end_id)
        self.start_id, self.end_id = self.kernel_dataset.start_id, self.kernel_dataset.end_id
        self.target_person_size = self.kernel_dataset.target_person_size
        self.crop_size = self.kernel_dataset.crop_size
        self.target_size = 256
        self.seg_num = 14
        self.img_idx = self.kernel_dataset.img_idx
        assert os.path.isfile(self.overall_hdf5)
        self.use_overall_annot = True
        with h5py.File(self.overall_hdf5, 'r') as f:
            self.s2ds = f['s2d'].value.astype(np.float32)
            self.cbxs = f['crop_box'].value.astype(np.float32)
            self.info_list = f['info'].value
        self.refresh_cache = refresh_cache
        self.refresh_img = refresh_img
        self.refresh_mat = refresh_mat

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, index):
        running_idx = self.img_idx[index]
        if self.refresh_cache:
            img, ann_stack, j3d, j2d, bbox, info = self.kernel_dataset[index]
            mpii_j2d = j3d[:, :2].copy()

            # resize: up to crop_size
            input_w, input_h = img.shape[:2]
            assert input_w <= self.crop_size and input_h <= self.crop_size
            if input_w < self.crop_size and input_h < self.crop_size:
                edge0 = max(input_w, input_h)
                lambda0 = float(self.crop_size / edge0)
                rsz_w, rsz_h = int(input_w * lambda0), int(input_h * lambda0)
                center_w, center_h = input_w / 2.0, input_h / 2.0
                imrsz = cv2.resize(src=img, dsize=(rsz_h, rsz_w), interpolation=cv2.INTER_LINEAR)
                center_w2, center_h2 = rsz_w / 2.0, rsz_h / 2.0
                rsz_stack = []
                for ann in ann_stack:
                    rsz_i = cv2.resize(src=ann, dsize=(rsz_h, rsz_w), interpolation=cv2.INTER_LINEAR)
                    rsz_ibit = np.zeros_like(rsz_i, dtype=np.uint8)
                    rsz_ibit[rsz_i >= 128] = 255
                    rsz_stack.append(rsz_ibit)
                mpii_j2d_rsz = np.zeros_like(mpii_j2d, dtype=np.float32)
                j2d_rsz = np.zeros_like(j2d, dtype=np.float32)
                mpii_j2d_rsz[:, 0] = (mpii_j2d[:, 0] - center_w) * lambda0 + center_w2
                mpii_j2d_rsz[:, 1] = (mpii_j2d[:, 1] - center_h) * lambda0 + center_h2
                j2d_rsz[:, 0] = (j2d[:, 0] - center_w) * lambda0 + center_w2
                j2d_rsz[:, 1] = (j2d[:, 1] - center_h) * lambda0 + center_h2
                bbox[0:2] = (bbox[0:2] - center_w) * lambda0 + center_w2
                bbox[2:4] = (bbox[2:4] - center_h) * lambda0 + center_h2
                img, ann_stack, mpii_j2d, j2d = imrsz, rsz_stack, mpii_j2d_rsz, j2d_rsz

            # fill blank
            input_w, input_h = img.shape[:2]
            assert input_w == self.crop_size or input_h == self.crop_size
            if input_w < self.crop_size:
                img_fl = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
                offset_wf = (self.crop_size - input_w) * 0.5
                offset_w = int(offset_wf)
                img_fl[offset_w: offset_w + input_w, :] = img
                fl_stack = []
                for ann in ann_stack:
                    fl_i = np.zeros((self.crop_size, self.crop_size), dtype=np.uint8)
                    fl_i[offset_w: offset_w + input_w, :] = ann
                    fl_stack.append(fl_i)
                mpii_j2d[:, 1] += offset_wf
                j2d[:, 1] += offset_wf
                bbox[0:2] += offset_wf
                img, ann_stack = img_fl, fl_stack
            if input_h < self.crop_size:
                img_fl = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
                offset_hf = (self.crop_size - input_h) * 0.5
                offset_h = int(offset_hf)
                img_fl[:, offset_h: offset_h + input_h] = img
                fl_stack = []
                for ann in ann_stack:
                    fl_i = np.zeros((self.crop_size, self.crop_size), dtype=np.uint8)
                    fl_i[:, offset_h: offset_h + input_h] = ann
                    fl_stack.append(fl_i)
                mpii_j2d[:, 0] += offset_hf
                j2d[:, 0] += offset_hf
                bbox[2:4] += offset_hf
                img, ann_stack = img_fl, fl_stack

            # resize: down to target_size
            input_w, input_h = img.shape[:2]
            assert input_w == self.crop_size and input_h == self.crop_size
            lambda_x = float(self.target_size / self.crop_size)
            img = cv2.resize(src=img, dsize=(self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            sf_stack = []
            for j, ann in enumerate(ann_stack):
                ann_j = cv2.resize(src=ann, dsize=(self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
                ann_jbit = np.zeros_like(ann_j, dtype=np.uint8)
                ann_jbit[ann_j >= 128] = 255
                sf_stack.append(ann_jbit)
            mpii_j2d *= lambda_x
            j2d *= lambda_x
            bbox *= lambda_x
            ann_stack = np.array(sf_stack)

            # joints: concat & supplement
            j3d[:, :2] = mpii_j2d
            reorder_mpii = [[2, 3], None, 12, 13, 3, 4, 5, 2, 1, 0, 9, 10, 11, 8, 7, 6]
            mpii2j25_map = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20, 12, 24]
            reorder_j2d25 = [None, [3, 6, 9], [12, 13, 14], 24, 1, 4, 7, 2, 5, 8, 16, 18, 20, 17, 19, 21]
            s2d = np.zeros((16, 2), dtype=np.float32)
            for k in range(16):
                idx_mpii = reorder_mpii[k]
                idx_j2d = reorder_j2d25[k]
                if type(idx_mpii) is list:
                    s2d[k] = np.mean([j3d[memb, :2] if j3d[memb, -1] > 0.5 else
                                          j2d[mpii2j25_map[memb]] for memb in idx_mpii], axis=0)
                elif idx_mpii is None or j3d[idx_mpii][-1] < 0.5:
                    s2d[k] = np.mean([j2d[memb] for memb in idx_j2d], axis=0) if type(idx_j2d) is list else j2d[idx_j2d]
                else:
                    s2d[k] = j3d[idx_mpii][:2]
            self.save_np(running_index=running_idx, arr=(img, ann_stack, s2d, bbox, info))
            is_valid = 1.0
        else:
            img, ann_stack, s2d, bbox, is_valid = self.load_np(running_index=running_idx)
        img_ret, ann_ret, s2d_ret, bbox_ret = \
            torch.from_numpy(img), torch.from_numpy(ann_stack), \
            torch.from_numpy(s2d), torch.from_numpy(bbox)
        return img_ret, ann_ret, s2d_ret, bbox_ret, is_valid

    def save_np(self, running_index, arr):
        img, ann_stack, s2d, bbox, info = arr
        target_path = os.path.join(self.proc_dir, '%05d/' % running_index)
        os.makedirs(target_path, exist_ok=True)
        if self.refresh_img:
            img_stack = np.stack([img[:, :, 0], img[:, :, 1], img[:, :, 2]])
            all_stack = np.concatenate([ann_stack, img_stack], axis=0)
            im_all = np.concatenate([all_stack[c, :, :] for c in range(all_stack.shape[0])], axis=1)
            cv2.imwrite(os.path.join(target_path, 'im%05d.png' % running_index), im_all.astype(np.uint8))
        if self.refresh_mat:
            with h5py.File(os.path.join(target_path, 'annot.h5'), 'w') as f:
                f['index'] = running_index
                f['s2d'] = s2d.astype(np.float32)
                f['crop_box'] = bbox.astype(np.float32)
                f['info'] = info

    def load_np(self, running_index, img_size=None, seg_size=None):
        img_size = self.target_size if img_size is None else img_size
        seg_size = self.target_size if seg_size is None else seg_size
        target_path = os.path.join(self.proc_dir, '%05d/' % running_index)
        ann_stack = []
        ann_count = self.seg_num + 1
        im_all = cv2.imread(os.path.join(target_path, 'im%05d.png' % running_index), cv2.IMREAD_GRAYSCALE)

        if im_all is None:
            img = np.zeros((3, img_size, img_size), dtype=np.float32)
            ann_stack = np.zeros([ann_count, seg_size, seg_size], dtype=np.float32)
            s2d = np.zeros([16, 2], dtype=np.float32)
            bbox = np.zeros([4], dtype=np.float32)
            is_valid = 0.0
            return img, ann_stack, s2d, bbox, is_valid
        else:
            assert im_all.shape[0] == img_size and im_all.shape[1] == (ann_count + 3) * img_size
            im_all = (im_all / 255).astype(np.float32)
            for k in range(ann_count):
                ann = im_all[:, k*img_size:(k+1)*img_size]
                ann_stack.append(cv2.resize(ann, (seg_size, seg_size), interpolation=cv2.INTER_CUBIC))
            ann_stack = np.array(ann_stack, dtype=np.float32)
            img = np.stack([im_all[:, -3*img_size:-2*img_size],
                            im_all[:, -2*img_size:-1*img_size],
                            im_all[:, -1*img_size:]], axis=0)
            is_valid = 1.0
        if not self.use_overall_annot:
            with h5py.File(os.path.join(target_path, 'annot.h5'), 'r') as f:
                s2d = f['s2d'].value.astype(np.float32)
                bbox = f['crop_box'].value.astype(np.float32)
        else:
            s2d = self.s2ds[running_index]
            bbox = self.cbxs[running_index]
        return img, ann_stack, s2d, bbox, is_valid

    def draw_bbox(self, img, bbox, display=False):
        cc = (np.random.rand(), np.random.rand(), np.random.rand())
        cv2.line(img, (int(bbox[2]), int(bbox[1])), (int(bbox[3]), int(bbox[1])), cc, thickness=2)
        cv2.line(img, (int(bbox[3]), int(bbox[0])), (int(bbox[3]), int(bbox[1])), cc, thickness=2)
        cv2.line(img, (int(bbox[3]), int(bbox[0])), (int(bbox[2]), int(bbox[0])), cc, thickness=2)
        cv2.line(img, (int(bbox[2]), int(bbox[1])), (int(bbox[2]), int(bbox[0])), cc, thickness=2)
        if display:
            cv2.imshow("pure2d, cbox", img)

    def is_valid_coord(self, pt):
        return 0 <= pt[0] < self.target_size and 0 <= pt[1] < self.target_size


class up_pure2d_processed(Dataset):
    """
    Augmentation will be added in the future
    """
    def __init__(self, data_dir, data_label, img_size=256, seg_size=64, start_id=0, end_id=None):
        self.data_dir = data_dir
        self.proc_dir = os.path.join(self.data_dir, './pproc/img/')
        assert os.path.exists(self.data_dir) and os.path.exists(self.proc_dir)
        self.data_label = data_label
        self.img_size = img_size
        self.seg_size = seg_size
        self.kernel_dataset = up_pure2d_base(data_dir=data_dir, data_label=data_label, start_id=start_id, end_id=end_id,
                                             refresh_cache=False, refresh_img=False, refresh_mat=False)
        self.start_id, self.end_id = self.kernel_dataset.start_id, self.kernel_dataset.end_id
        self.img_idx = self.kernel_dataset.img_idx

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, index):
        running_idx = self.img_idx[index]
        img_ret, ann_ret, s2d_ret, bbox_ret, is_valid = \
            self.kernel_dataset.load_np(running_index=running_idx, img_size=self.img_size, seg_size=self.seg_size)
        # # shall not convert into torch tensor
        # img_ret, ann_ret, s2d_ret, bbox_ret = \
        #     torch.from_numpy(img), torch.from_numpy(ann_ret), \
        #     torch.from_numpy(s2d_ret), torch.from_numpy(bbox_ret)
        return img_ret, ann_ret, s2d_ret, bbox_ret, is_valid


if __name__ == '__main__':
    # # DONE: pre-process
    # pproc = up_pproc()
    # pproc.data_up3d_py2to3()
    # pproc.data_ups31_pproc()
    # pproc.merge_individual_annot()

    DATA_DIR = "/data/dataset/up/"
    DATA_LABEL = "test"  # train/test/trainval/val => 5703/1389/7126/1423
    ds = up_pure2d_base(data_dir=DATA_DIR, data_label=DATA_LABEL, start_id=0, end_id=None,
                        refresh_cache=True, refresh_img=True, refresh_mat=False)
    print('total:', len(ds))

    def run_dataset(dataset, debug=False):
        pending_list = [490, 735, 633, 282, 777, 240, 755, 376, 1307]
        for i in pending_list if debug else range(len(dataset)):
            # j = i if debug else int(np.random.rand() * len(dataset))
            j = i
            # print("#################################")
            img, ann_stack, s2d, bbox, is_valid = dataset[j]
            print('img >>', j, img.shape, is_valid)
            # img = img.numpy().transpose(1, 2, 0).copy()
            # # # dataset.draw_bbox(img, bbox)
            # # silh_list = [list(), list()]
            # # for k, ann in enumerate(ann_stack):
            # #     if k <= 0:
            # #         continue
            # #     silh_list[(k - 1) // 7].append(ann)
            # # col_stack = []
            # # for row_stack in silh_list:
            # #     col_stack.append(np.concatenate(row_stack, axis=1))
            # # # cv2.imshow("Silhouettes", np.concatenate(col_stack, axis=0))
            # # for lmb in limb_conn:
            # #     cv2.line(img,
            # #              pt1=(int(s2d[lmb[0]][0]), int(s2d[lmb[0]][1])),
            # #              pt2=(int(s2d[lmb[1]][0]), int(s2d[lmb[1]][1])),
            # #              color=(np.random.rand(), np.random.rand(), np.random.rand()),
            # #              thickness=3)
            #
            # if debug:
            #     s2d_check = s2d.numpy()
            #     checkbox = np.zeros_like(s2d_check, dtype=np.uint8)
            #     checkbox[(s2d_check < 0) | (s2d_check > 256)] = 1
            #     if checkbox.sum() >= 3 or \
            #             not dataset.is_valid_coord(s2d[0]) or \
            #             not dataset.is_valid_coord(s2d[1]) or \
            #             not dataset.is_valid_coord(s2d[2]):
            #         print('img >>', j, s2d)
            #         cv2.imwrite('tmp/OUT-IMG_%05d.png' % j, img)
            # else:
            #     cv2.imshow("OUT-IMG", img)
            #     cv2.waitKey(0)
            #
            # print("#################################")
            # break
    #
    # # np.random.seed(0xffff)
    # limb_conn = [
    #     [0, 1], [1, 2], [2, 3],
    #     [0, 4], [4, 5], [5, 6],
    #     [0, 7], [7, 8], [8, 9],
    #     [2, 10], [10, 11], [11, 12],
    #     [2, 13], [13, 14], [14, 15]
    # ]
    #
    run_dataset(ds)
    #
    # h_batch_size = 4
    # test_loader = DataLoader(
    #     dataset=up_pure2d_processed(
    #         data_dir=DATA_DIR,
    #         data_label=DATA_LABEL,
    #         img_size=256,
    #         seg_size=64,
    #         start_id=0,
    #         end_id=None),
    #     batch_size=h_batch_size,
    #     shuffle=True,
    #     num_workers=4,
    #     pin_memory=True)
    # print('%s => %d' % (DATA_LABEL, len(test_loader)))
    #
    # import time
    # import sys
    #
    # last_iter_time = time.time()
    # start_time = last_iter_time
    # monitor_count = 10
    # for i, (img, ann, s2d, bbox, is_valid) in enumerate(test_loader):
    #     if is_valid.sum() < h_batch_size:
    #         print(i, 'missing...')
    #         continue
    #     if not (i % monitor_count):
    #         print(i, '->', time.time() - last_iter_time)
    #         sys.stdout.flush()
    #         last_iter_time = time.time()
    #
    # print('total:', time.time() - start_time)
