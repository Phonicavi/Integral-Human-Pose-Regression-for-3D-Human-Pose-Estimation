import os
import os.path as osp
import argparse
from tqdm import tqdm
import numpy as np
import cv2
from config import cfg
import torch
from base import Tester
from torch.nn.parallel.scatter_gather import gather
from nets.loss import soft_argmax
from utils.vis import vis_keypoints, vis_3d_skeleton
from utils.pose_utils import flip
import torch.backends.cudnn as cudnn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--output', type=str, dest='output_dir', default='')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        args.gpu_ids = str(np.argmin(mem_info()))
    if args.output_dir == '':
        args.output_dir = None

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    assert args.test_epoch, 'Test epoch is required.'
    return args


def main():
    args = parse_args()
    cfg.set_args(args.gpu_ids, continue_train=False, output_dir=args.output_dir)
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    cfg.testset = 'MPII'

    tester = Tester(cfg, args.test_epoch, default_data_split="train")  # force_convert=True
    tester._make_batch_generator()
    tester._make_model()

    preds = []
    os.makedirs(osp.join(cfg.vis_dir, '%d-mpii' % tester.test_epoch), exist_ok=True)

    with torch.no_grad():
        for itr, (input_img, joint_vis, _) in enumerate(tqdm(tester.batch_generator)):

            input_img = input_img.cuda()
            batch_size = input_img.size(0)

            # if batch_size < cfg.test_batch_size * cfg.num_gpus:
            #     continue

            # forward
            heatmap_out = tester.model(input_img)
            if cfg.num_gpus > 1:
                heatmap_out = gather(heatmap_out, 0)
            coord_out = soft_argmax(heatmap_out, tester.joint_num)

            vis = False
            if vis:
                filename = str(itr)
                tmpimg = input_img[0].cpu().numpy()
                tmpimg = tmpimg * np.array(cfg.pixel_std).reshape(3, 1, 1) + np.array(cfg.pixel_mean).reshape(3, 1, 1)
                tmpimg = tmpimg.astype(np.uint8)
                tmpimg = tmpimg[::-1, :, :]
                tmpimg = np.transpose(tmpimg, (1, 2, 0)).copy()
                tmpkps = np.zeros((3, tester.joint_num))
                tmpkps[:2, :] = coord_out.cpu()[0, :, :2].transpose(1, 0) / cfg.output_shape[0] * cfg.input_shape[0]
                tmpkps[2, :] = 1
                tmpimg = vis_keypoints(tmpimg, tmpkps, tester.skeleton)
                cv2.imwrite(osp.join(cfg.vis_dir, ('%d-mpii/' % tester.test_epoch) + filename + '_output.jpg'), tmpimg)
                vis_3d_skeleton(kpt_3d=coord_out.cpu().numpy()[0].copy(), kpt_3d_vis=joint_vis.cpu().numpy()[0].copy(),
                                kps_lines=tester.skeleton,
                                filename=osp.join(cfg.vis_dir, ('%d-mpii/' % tester.test_epoch) + filename + '_figure.jpg'))

            coord_out = coord_out.cpu().numpy()
            preds.append(coord_out)

    # evaluate
    preds = np.concatenate(preds, axis=0)

    import pickle
    pickle.dump(preds, open(osp.join(cfg.vis_dir, ('%d-mpii/' % tester.test_epoch) + 'mpii-train.pkl'), 'wb'))
    # todo: notice that mpii._evaluate has not been implemented
    # tester._evaluate(preds, cfg.result_dir)


def generate_mpii_batch():

    args = parse_args()
    cfg.set_args(args.gpu_ids, continue_train=False, output_dir=args.output_dir)
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    cfg.testset = 'MPII'

    tester = Tester(cfg, args.test_epoch, default_data_split="train")  # force_convert=True
    tester._make_batch_generator()
    tester._make_model()

    vis_all = []
    path_all = []
    os.makedirs(osp.join(cfg.vis_dir, '%d-mpii' % tester.test_epoch), exist_ok=True)

    with torch.no_grad():
        for itr, (input_img, joint_vis, img_path) in enumerate(tqdm(tester.batch_generator)):

            input_img = input_img.cuda()
            # batch_size = input_img.size(0)
            if cfg.num_gpus > 1:
                input_img = gather(input_img, 0)
                joint_vis = gather(joint_vis, 0)
                img_path = gather(img_path, 0)

            # vis_all += [joint_vis[i].cpu().numpy() for i in range(len(joint_vis))]
            # path_all += img_path
            vis_all.append(joint_vis)
            path_all.append(img_path)

    vis_all = np.concatenate(vis_all, axis=0)
    path_all = np.concatenate(path_all, axis=0)

    import pickle
    pickle.dump(vis_all, open(osp.join(cfg.vis_dir, ('%d-mpii/' % tester.test_epoch) + 'mpii-train_vis.pkl'), 'wb'))
    pickle.dump(path_all, open(osp.join(cfg.vis_dir, ('%d-mpii/' % tester.test_epoch) + 'mpii-train_path.pkl'), 'wb'))


def draw_3d_sample():
    import pickle
    pr = pickle.load(open('mpii-train.pkl', 'rb'))
    s3d = pr[0]
    ref_skeleton = ((0, 7), (7, 8), (8, 9), (9, 10),
                    (8, 11), (11, 12), (12, 13),
                    (8, 14), (14, 15), (15, 16),
                    (0, 1), (1, 2), (2, 3),
                    (0, 4), (4, 5), (5, 6))
    vis_dummy = np.ones((17, 1))

    vis_3d_skeleton(kpt_3d=s3d, kpt_3d_vis=vis_dummy, kps_lines=ref_skeleton)


if __name__ == "__main__":
    main()
    # generate_mpii_batch()
    # draw_3d_sample()
