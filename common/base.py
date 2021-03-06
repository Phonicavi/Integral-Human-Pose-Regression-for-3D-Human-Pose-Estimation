import os
import os.path as osp
import math
import time
import glob
import abc
import numpy as np
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms

from UP3D import UP3D

from config import cfg
from dataset import DatasetLoader
from timer import Timer
from logger import colorlogger
from nets.balanced_parallel import DataParallelModel, DataParallelCriterion
from model import get_pose_net, get_pose_net_baseline
from nets import loss

# dynamic dataset import
for i in range(len(cfg.trainset)):
    exec('from ' + cfg.trainset[i] + ' import ' + cfg.trainset[i])
exec('from ' + cfg.testset + ' import ' + cfg.testset)

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, cfg, log_name='logs.txt'):
        
        self.cfg = cfg
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

    def save_model(self, state, epoch):
        file_path = osp.join(self.cfg.model_dir,'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer, scheduler):
        model_file_list = glob.glob(osp.join(self.cfg.model_dir,'*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        ckpt = torch.load(osp.join(self.cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')) 
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['network'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

        return start_epoch, model, optimizer, scheduler


class Trainer(Base):
    
    def __init__(self, cfg, default_data_split="train"):
        self.JointLocationLoss = DataParallelCriterion(loss.JointLocationLoss())
        self.default_data_split = default_data_split
        super(Trainer, self).__init__(cfg, log_name = 'train_logs.txt')

    def get_optimizer(self, optimizer_name, model):
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.wd) 
        else:
            print("Error! Unknown optimizer name: ", optimizer_name)
            assert 0

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.lr_dec_epoch, gamma=self.cfg.lr_dec_factor)
        return optimizer, scheduler
    
    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        trainset_list = []
        for i in range(len(self.cfg.trainset)):
            trainset_list.append(eval(self.cfg.trainset[i])(self.default_data_split))
        trainset_loader = DatasetLoader(trainset_list, True, transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]))
        batch_generator = DataLoader(dataset=trainset_loader, batch_size=self.cfg.num_gpus * self.cfg.batch_size,
                                     shuffle=True, num_workers=self.cfg.num_thread, pin_memory=True)
        
        self.joint_num = trainset_loader.joint_num[0]
        self.itr_per_epoch = math.ceil(trainset_loader.__len__() / cfg.num_gpus / cfg.batch_size)
        self.batch_generator = batch_generator
    
    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_pose_net(self.cfg, True, self.joint_num)
        model = DataParallelModel(model).cuda()
        optimizer, scheduler = self.get_optimizer(self.cfg.optimizer, model)
        if self.cfg.continue_train:
            start_epoch, model, optimizer, scheduler = self.load_model(model, optimizer, scheduler)
            # todo: modify optimizer.lr, scheduler.milestones, scheduler.gamma
            if self.cfg.optimizer == 'adam':
                init_lr = self.cfg.lr
                milestones = self.cfg.lr_dec_epoch
                gamma = self.cfg.lr_dec_factor
                scheduler.milestones = milestones
                scheduler.gamma = gamma
                if start_epoch < milestones[0]:
                    optimizer.lr = init_lr
                else:
                    for item in milestones:
                        if start_epoch >= item:
                            init_lr *= gamma
                        else:
                            break
                    optimizer.lr = init_lr
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

class Tester(Base):
    
    def __init__(self, cfg, test_epoch, default_data_split="test"):
        self.coord_out = loss.soft_argmax
        self.test_epoch = int(test_epoch)
        self.default_data_split = default_data_split
        super(Tester, self).__init__(cfg, log_name = 'test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset = eval(self.cfg.testset)(self.default_data_split)
        force_convert = False
        curr_jn = None  # mpii-joint_name
        ref_jn = None  # h36m-joint_name
        ref_sks = None
        if self.default_data_split == "train":
            # todo: i.e. this=mpii, target=h36m
            force_convert = True
            curr_jn = ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Thorax', 'Neck', 'Head', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist')
            ref_jn = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso',
                      'Neck', 'Nose', 'Head',
                      'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', )#'Thorax')

            ref_skeleton = ((0, 7), (7, 8), (8, 9), (9, 10),
                             (8, 11), (11, 12), (12, 13),
                             (8, 14), (14, 15), (15, 16),
                             (0, 1), (1, 2), (2, 3),
                             (0, 4), (4, 5), (5, 6))
            ref_lr_skeleton = (((8, 11), (8, 14)), ((11, 12), (14, 15)), ((12, 13), (15, 16)),
                                ((0, 1), (0, 4)), ((1, 2), (4, 5)), ((2, 3), (5, 6)))
            ref_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
            ref_sks = (ref_skeleton, ref_lr_skeleton, ref_flip_pairs)

        testset_loader = DatasetLoader(testset, False,
                                       transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]),
                                       force_convert=force_convert, curr_jn=curr_jn, ref_jn=ref_jn, ref_sks=ref_sks)
        batch_generator = DataLoader(dataset=testset_loader, batch_size=self.cfg.num_gpus * self.cfg.test_batch_size,
                                     shuffle=False, num_workers=self.cfg.num_thread, pin_memory=True)
        
        self.testset = testset
        self.joint_num = testset_loader.joint_num
        self.skeleton = testset_loader.skeleton
        self.flip_pairs = testset.flip_pairs
        self.tot_sample_num = testset_loader.__len__()
        self.batch_generator = batch_generator
    
    def _make_model(self):
        
        model_path = os.path.join(self.cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        
        # prepare network
        self.logger.info("Creating graph...")
        model = get_pose_net(self.cfg, False, self.joint_num)
        model = DataParallelModel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'])
        model.eval()

        self.model = model

    def _evaluate(self, preds, result_save_path):
        return self.testset.evaluate(preds, result_save_path)

class SimpleBaselineTrainer(Base):

    def __init__(self, cfg, default_data_split="train"):
        self.JointMSELoss = DataParallelCriterion(loss.JointMSELoss())
        self.default_data_split = default_data_split
        super(SimpleBaselineTrainer, self).__init__(cfg, log_name='train_logs.txt')

    def get_optimizer(self, optimizer_name, model):
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum,
                                        weight_decay=self.cfg.wd)
        else:
            print("Error! Unknown optimizer name: ", optimizer_name)
            assert 0

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.lr_dec_epoch,
                                                         gamma=self.cfg.lr_dec_factor)
        return optimizer, scheduler

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        trainset_list = []
        for i in range(len(self.cfg.trainset)):
            trainset_list.append(eval(self.cfg.trainset[i])(self.default_data_split))
        trainset_loader = DatasetLoader(trainset_list, True, transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]))
        batch_generator = DataLoader(dataset=trainset_loader, batch_size=self.cfg.num_gpus * self.cfg.batch_size,
                                     shuffle=True, num_workers=self.cfg.num_thread, pin_memory=True)

        self.joint_num = trainset_loader.joint_num[0]
        self.itr_per_epoch = math.ceil(trainset_loader.__len__() / cfg.num_gpus / cfg.batch_size)
        self.batch_generator = batch_generator

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_pose_net_baseline(self.cfg, True, self.joint_num)
        model = DataParallelModel(model).cuda()
        optimizer, scheduler = self.get_optimizer(self.cfg.optimizer, model)
        if self.cfg.continue_train:
            start_epoch, model, optimizer, scheduler = self.load_model(model, optimizer, scheduler)
            # todo: modify optimizer.lr, scheduler.milestones, scheduler.gamma
            if self.cfg.optimizer == 'adam':
                init_lr = self.cfg.lr
                milestones = self.cfg.lr_dec_epoch
                gamma = self.cfg.lr_dec_factor
                scheduler.milestones = milestones
                scheduler.gamma = gamma
                if start_epoch < milestones[0]:
                    optimizer.lr = init_lr
                else:
                    for item in milestones:
                        if start_epoch >= item:
                            init_lr *= gamma
                        else:
                            break
                    optimizer.lr = init_lr
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

class SimpleBaselineTester(Base):

    def __init__(self, cfg, test_epoch, default_data_split="test"):
        self.coord_out = loss.soft_argmax
        self.test_epoch = int(test_epoch)
        self.default_data_split = default_data_split
        super(SimpleBaselineTester, self).__init__(cfg, log_name='test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset = eval(self.cfg.testset)(self.default_data_split)
        testset_loader = DatasetLoader(testset, False, transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]))
        batch_generator = DataLoader(dataset=testset_loader, batch_size=self.cfg.num_gpus * self.cfg.test_batch_size,
                                     shuffle=False, num_workers=self.cfg.num_thread, pin_memory=True)

        self.testset = testset
        self.joint_num = testset_loader.joint_num
        self.skeleton = testset_loader.skeleton
        self.flip_pairs = testset.flip_pairs
        self.tot_sample_num = testset_loader.__len__()
        self.batch_generator = batch_generator

    def _make_model(self):
        model_path = os.path.join(self.cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))

        # prepare network
        self.logger.info("Creating graph...")
        model = get_pose_net_baseline(self.cfg, False, self.joint_num)
        model = DataParallelModel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'])
        model.eval()

        self.model = model

    def _evaluate(self, preds, result_save_path):
        return self.testset.evaluate(preds, result_save_path)

class UP3DTester(Base):

    def __init__(self, cfg, test_epoch, default_data_split="test"):
        self.coord_out = loss.soft_argmax
        self.test_epoch = int(test_epoch)
        self.default_data_split = default_data_split
        super(UP3DTester, self).__init__(cfg, log_name='test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset = eval(self.cfg.testset)(self.default_data_split)
        force_convert = False
        curr_jn = None  # mpii-joint_name
        ref_jn = None  # h36m-joint_name
        ref_sks = None
        if self.default_data_split == "train":
            # todo: i.e. this=mpii, target=h36m
            force_convert = True
            curr_jn = (
            'R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Thorax', 'Neck', 'Head', 'R_Wrist',
            'R_Elbow', 'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist')
            ref_jn = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso',
                      'Neck', 'Nose', 'Head',
                      'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist',)  # 'Thorax')

            ref_skeleton = ((0, 7), (7, 8), (8, 9), (9, 10),
                            (8, 11), (11, 12), (12, 13),
                            (8, 14), (14, 15), (15, 16),
                            (0, 1), (1, 2), (2, 3),
                            (0, 4), (4, 5), (5, 6))
            ref_lr_skeleton = (((8, 11), (8, 14)), ((11, 12), (14, 15)), ((12, 13), (15, 16)),
                               ((0, 1), (0, 4)), ((1, 2), (4, 5)), ((2, 3), (5, 6)))
            ref_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
            ref_sks = (ref_skeleton, ref_lr_skeleton, ref_flip_pairs)

        # testset_loader = DatasetLoader(testset, False,
        #                                transforms.Compose([
        #                                    transforms.ToTensor(),
        #                                    transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]),
        #                                force_convert=force_convert, curr_jn=curr_jn, ref_jn=ref_jn, ref_sks=ref_sks)

        testset_loader = UP3D(data_label="trainval",
                                          normalize_dict={
                                              'pixel_mean': (0.485, 0.456, 0.406),
                                              'pixel_std': (0.229, 0.224, 0.225),
                                          },
                                          const_dict={
                                              'K_mean': np.array([[1.1473444e+03, 0.0000000e+00, 5.1404352e+02],
                                                                  [0.0000000e+00, 1.1462365e+03, 5.0670016e+02],
                                                                  [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]],
                                                                 dtype=np.float32),
                                              'K_std': np.array([[2.0789747, 0.0000001, 3.9817653],
                                                                 [0.0000001, 2.0353518, 5.6948705],
                                                                 [0.0000001, 0.0000001, 0.0000001]], dtype=np.float32),
                                              'dist_mean': np.array(
                                                  [-0.20200874, 0.24049881, -0.00168468, -0.00042439, -0.00191587],
                                                  dtype=np.float32),
                                              'dist_std': np.array(
                                                  [0.00591341, 0.01386873, 0.00071674, 0.00116201, 0.00564366],
                                                  dtype=np.float32),
                                          }, img_size=256)

        batch_generator = DataLoader(dataset=testset_loader, batch_size=self.cfg.num_gpus * self.cfg.test_batch_size,
                                     shuffle=False, num_workers=self.cfg.num_thread, pin_memory=True)

        self.testset = testset
        self.joint_num = testset_loader.joint_num
        self.skeleton = testset_loader.skeleton
        self.flip_pairs = testset.flip_pairs
        self.tot_sample_num = testset_loader.__len__()
        self.batch_generator = batch_generator

    def _make_model(self):
        model_path = os.path.join(self.cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))

        # prepare network
        self.logger.info("Creating graph...")
        model = get_pose_net(self.cfg, False, self.joint_num)
        model = DataParallelModel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'])
        model.eval()

        self.model = model

    def _evaluate(self, preds, result_save_path):
        return self.testset.evaluate(preds, result_save_path)
