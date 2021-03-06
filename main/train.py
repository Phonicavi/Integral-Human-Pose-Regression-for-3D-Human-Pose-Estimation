import argparse
from config import cfg
from base import Trainer
import torch.backends.cudnn as cudnn
import tensorboardX

from nets.loss import soft_argmax

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--output', type=str, dest='output_dir', default='')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
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

def main():
    
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.continue_train, args.output_dir)
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
            print('input_img', input_img.shape)
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


if __name__ == "__main__":
    main()
