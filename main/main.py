import argparse
import tensorboardX
import cv2
import os
import os.path as osp
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from config import cfg
from base import Trainer, Tester, SimpleBaselineTrainer, SimpleBaselineTester
from torch.nn.parallel.scatter_gather import gather
from nets.loss import soft_argmax
from utils.pose_utils import flip
from utils.vis import vis_keypoints
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--output', type=str, dest='output_dir', default='')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--baseline', dest='baseline', action='store_true')
    args = parser.parse_args()

    if not args.gpu_ids:
        args.gpu_ids = str(np.argmin(mem_info()))
    if args.output_dir == '':
        args.output_dir = None

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def embedded_test(tensorx, test_epoch):
    tester = Tester(cfg, test_epoch)
    tester._make_batch_generator()
    tester._make_model()
    preds = []

    with torch.no_grad():
        for itr, input_img in enumerate(tqdm(tester.batch_generator)):

            input_img = input_img.cuda()
            batch_size = input_img.size(0)
            if batch_size < cfg.test_batch_size * cfg.num_gpus:
                continue

            # forward
            heatmap_out = tester.model(input_img)
            if cfg.num_gpus > 1:
                heatmap_out = gather(heatmap_out, 0)
            coord_out = soft_argmax(heatmap_out, tester.joint_num)

            if cfg.flip_test:
                flipped_input_img = flip(input_img, dims=3)
                flipped_heatmap_out = tester.model(flipped_input_img)
                if cfg.num_gpus > 1:
                    flipped_heatmap_out = gather(flipped_heatmap_out, 0)
                flipped_coord_out = soft_argmax(flipped_heatmap_out, tester.joint_num)
                flipped_coord_out[:, :, 0] = cfg.output_shape[1] - flipped_coord_out[:, :, 0] - 1
                for pair in tester.flip_pairs:
                    flipped_coord_out[:, pair[0], :], flipped_coord_out[:, pair[1], :] = flipped_coord_out[:, pair[1],
                                                                                         :].clone(), flipped_coord_out[
                                                                                                     :, pair[0],
                                                                                                     :].clone()
                coord_out = (coord_out + flipped_coord_out) / 2.

            vis = True
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
                os.makedirs(osp.join(cfg.vis_dir, '%d' % test_epoch), exist_ok=True)
                cv2.imwrite(osp.join(cfg.vis_dir, ('%d/' % test_epoch) + filename + '_output.jpg'), tmpimg)

            coord_out = coord_out.cpu().numpy()
            preds.append(coord_out)

    # evaluate
    preds = np.concatenate(preds, axis=0)
    os.makedirs(osp.join(cfg.result_dir, '%d' % test_epoch), exist_ok=True)
    p1_error, p2_error = \
        tester._evaluate(preds, osp.join(cfg.result_dir, '%d' % test_epoch))
    tensorx.add_scalars('Test', {'p1_error(PA.MPJPE)': p1_error, 'p2_error(MPJPE)': p2_error,}, test_epoch)

def embedded_test_baseline(tensorx, test_epoch):
    tester = SimpleBaselineTester(cfg, test_epoch)
    tester._make_batch_generator()
    tester._make_model()
    preds = []

    with torch.no_grad():
        for itr, input_img in enumerate(tqdm(tester.batch_generator)):

            input_img = input_img.cuda()
            batch_size = input_img.size(0)
            if batch_size < cfg.test_batch_size * cfg.num_gpus:
                continue

            # forward
            # heatmap_out = tester.model(input_img)
            # if cfg.num_gpus > 1:
            #     heatmap_out = gather(heatmap_out, 0)
            # coord_out = soft_argmax(heatmap_out, tester.joint_num)
            coords_out = tester.model(input_img)
            if cfg.num_gpus > 1:
                coords_out = gather(coords_out, 0)

            num_joints = tester.joint_num
            coord_out = coords_out.reshape((batch_size, num_joints, -1))

            # if cfg.flip_test:
            #     flipped_input_img = flip(input_img, dims=3)
            #     flipped_heatmap_out = tester.model(flipped_input_img)
            #     if cfg.num_gpus > 1:
            #         flipped_heatmap_out = gather(flipped_heatmap_out, 0)
            #     flipped_coord_out = soft_argmax(flipped_heatmap_out, tester.joint_num)
            #     flipped_coord_out[:, :, 0] = cfg.output_shape[1] - flipped_coord_out[:, :, 0] - 1
            #     for pair in tester.flip_pairs:
            #         flipped_coord_out[:, pair[0], :], flipped_coord_out[:, pair[1], :] = flipped_coord_out[:, pair[1],
            #                                                                              :].clone(), flipped_coord_out[
            #                                                                                          :, pair[0],
            #                                                                                          :].clone()
            #     coord_out = (coord_out + flipped_coord_out) / 2.

            vis = True
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
                os.makedirs(osp.join(cfg.vis_dir, '%d' % test_epoch), exist_ok=True)
                cv2.imwrite(osp.join(cfg.vis_dir, ('%d/' % test_epoch) + filename + '_output.jpg'), tmpimg)

            coord_out = coord_out.cpu().numpy()
            preds.append(coord_out)

    # evaluate
    preds = np.concatenate(preds, axis=0)
    os.makedirs(osp.join(cfg.result_dir, '%d' % test_epoch), exist_ok=True)
    p1_error, p2_error = \
        tester._evaluate(preds, osp.join(cfg.result_dir, '%d' % test_epoch))
    tensorx.add_scalars('Test', {'p1_error(PA.MPJPE)': p1_error, 'p2_error(MPJPE)': p2_error,}, test_epoch)

def main():
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.continue_train, args.output_dir)
    if args.baseline:
        return
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    trainer = Trainer(cfg)
    trainer._make_batch_generator()
    trainer._make_model()

    # train
    tbx = tensorboardX.SummaryWriter(cfg.tbx_dir)
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.scheduler.step()
        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        for itr, (input_img, joint_img, joint_vis, joints_have_depth) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            trainer.optimizer.zero_grad()

            input_img = input_img.cuda()
            joint_img = joint_img.cuda()
            joint_vis = joint_vis.cuda()
            joints_have_depth = joints_have_depth.cuda()

            # forward
            heatmap_out = trainer.model(input_img)

            # backward
            JointLocationLoss = trainer.JointLocationLoss(heatmap_out, joint_img, joint_vis, joints_have_depth)

            loss = JointLocationLoss

            loss.backward()
            trainer.optimizer.step()

            trainer.gpu_timer.toc()

            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.scheduler.get_lr()[0]),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                '%s: %.4f' % ('loss_loc', JointLocationLoss.detach()),
            ]
            trainer.logger.info(' '.join(screen))
            tbx.add_scalars('Train', {'loss_loc': JointLocationLoss.detach()}, epoch * trainer.itr_per_epoch + itr)

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        trainer.save_model({
            'epoch': epoch,
            'network': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
            'scheduler': trainer.scheduler.state_dict(),
        }, epoch)

        if not (epoch % 2000) or epoch + 1 >= cfg.end_epoch:
            embedded_test(tbx, epoch)

def run_baseline():
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.continue_train, args.output_dir)
    if not args.baseline:
        return
    cfg.trainset = ['Human36M']
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    trainer = SimpleBaselineTrainer(cfg)
    trainer._make_batch_generator()
    trainer._make_model()

    # train
    tbx = tensorboardX.SummaryWriter(cfg.tbx_dir)
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.scheduler.step()
        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        for itr, (input_img, joint_img, joint_vis, joints_have_depth) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            trainer.optimizer.zero_grad()

            input_img = input_img.cuda()
            joint_img = joint_img.cuda()
            joint_vis = joint_vis.cuda()
            joints_have_depth = joints_have_depth.cuda()

            # forward
            heatmap_out = trainer.model(input_img)

            # backward
            JointMSELoss = trainer.JointMSELoss(heatmap_out, joint_img, joint_vis)

            loss = JointMSELoss

            loss.backward()
            trainer.optimizer.step()

            trainer.gpu_timer.toc()

            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.scheduler.get_lr()[0]),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                '%s: %.4f' % ('loss_mse', JointMSELoss.detach()),
            ]
            trainer.logger.info(' '.join(screen))
            tbx.add_scalars('Train', {
                'loss_mse': JointMSELoss.detach(),
            }, epoch * trainer.itr_per_epoch + itr)

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        trainer.save_model({
            'epoch': epoch,
            'network': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
            'scheduler': trainer.scheduler.state_dict(),
        }, epoch)

        if not (epoch % 10) or epoch + 1 >= cfg.end_epoch:
            embedded_test_baseline(tbx, epoch)


if __name__ == "__main__":
    main()
    run_baseline()
