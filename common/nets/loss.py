import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.pose_utils import pixel2cam, warp_coord_to_original
from config import cfg

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

def soft_argmax(heatmaps, joint_num):
    assert isinstance(heatmaps, torch.Tensor)

    heatmaps = heatmaps.reshape((-1, joint_num, cfg.depth_dim*cfg.output_shape[0]*cfg.output_shape[1]))
    heatmaps = F.softmax(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, joint_num, cfg.depth_dim, cfg.output_shape[0], cfg.output_shape[1]))

    accu_x = heatmaps.sum(dim=(2,3))
    accu_y = heatmaps.sum(dim=(2,4))
    accu_z = heatmaps.sum(dim=(3,4))

    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(1,cfg.output_shape[1]+1).type(torch.cuda.FloatTensor), devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(1,cfg.output_shape[0]+1).type(torch.cuda.FloatTensor), devices=[accu_y.device.index])[0]
    accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(1,cfg.depth_dim+1).type(torch.cuda.FloatTensor), devices=[accu_z.device.index])[0]

    accu_x = accu_x.sum(dim=2, keepdim=True) -1
    accu_y = accu_y.sum(dim=2, keepdim=True) -1
    accu_z = accu_z.sum(dim=2, keepdim=True) -1

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)

    return coord_out

class JointLocationLoss(nn.Module):
    def __init__(self):
        super(JointLocationLoss, self).__init__()

    def forward(self, heatmap_out, gt_coord, gt_vis, gt_have_depth):
        #print('[JointLocationLoss]: heatmap_out', heatmap_out.shape, 'gt_coord', gt_coord.shape, 'gt_vis', gt_vis.shape)
        
        joint_num = gt_coord.shape[1]
        coord_out = soft_argmax(heatmap_out, joint_num)

        _assert_no_grad(gt_coord)
        _assert_no_grad(gt_vis)
        _assert_no_grad(gt_have_depth)

        loss = torch.abs(coord_out - gt_coord) * gt_vis
        loss = (loss[:,:,0] + loss[:,:,1] + loss[:,:,2] * gt_have_depth)/3.

        return loss.mean()


class JointMSELoss(nn.Module):
    def __init__(self):
        super(JointMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = True

    def forward(self, output, target, target_weight):
        print('[JointMSELoss]: heatmap_out', output.shape, 'target', target.shape, 'target_weight', target_weight.shape)
        batch_size = target.size(0)
        num_joints = target.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        print('[JointMSE-forward] heatmaps_pred:', heatmaps_pred.shape, 'heatmaps_gt', heatmaps_gt.shape)
        print('[JointMSE-forward][after-squz] heatmaps_pred:', heatmaps_pred[0].squeeze().shape, 'heatmaps_gt', heatmaps_gt[0].squeeze().shape)

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints
