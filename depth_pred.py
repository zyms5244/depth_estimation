import os
import torch
import torchvision
import numpy as np
from models import FCRN, MADUCNet, DUCNet, ResDUCNet
from models.weights import load_weights

from torch.autograd import Variable
from data.nyu_dataset import NyuDepthMat, NYUDepthDataset
from config import SDFCNConfig
import matplotlib.pyplot as plt


dtype = torch.cuda.FloatTensor

resume_from_file = True

Threshold_1_25 = 0
Threshold_1_25_2 = 0
Threshold_1_25_3 = 0
RMSE_linear = 0.0
RMSE_log = 0.0
RMSE_log_scale_invariant = 0.0
ARD = 0.0
SRD = 0.0

cfg = SDFCNConfig()

model = ResDUCNet(torchvision.models.resnet50(False))
loss_fn = torch.nn.MSELoss()

resume_file = '/home/ans/PycharmProjects/SDFCN/checkpoint/depth_ResDuc_resblock_l1/checkpoint.pth.tar_depth_ResDuc_resblock_l1_epoch_18_L1'
test_norm = True
# test_norm = False
if resume_from_file:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # model.load_state_dict(checkpoint['model_state'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_file, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_file))
        model.load_state_dict(load_weights(model, cfg.weights_file, dtype))


if cfg.use_gpu:
    model = model.cuda()
    loss_fn = torch.nn.MSELoss().cuda()

test_set = NyuDepthMat(cfg.test_data_root, '/home/ans/PycharmProjects/SDFCN/data/testIdxs.txt')
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=cfg.test_batch_size,
                                          shuffle=False, drop_last=False)

# test_set = NYUDepthDataset(cfg.trainval_data_root, 'val', transform=True)
# test_loader = torch.utils.data.DataLoader(test_set,
#                                          batch_size=cfg.test_batch_size, shuffle=False,
#                                          num_workers=cfg.num_workers, drop_last=False)

model.eval()
idx = 0
num_samples = 0
rmse_linear = 0
with torch.no_grad():
    for i_batch, sample_batched in enumerate(test_loader):
        num_samples += 1
        input_var = Variable(sample_batched['rgb'].type(dtype))
        gt_var = Variable(sample_batched['depth'].type(dtype))
        #input_var = input_var.unsqueeze(0)

        output = model(input_var)

        print('predict complete.')

        # input_rgb_image = input_var[0].data.permute(1, 2, 0).cpu().numpy()
        # input_rgb_image = input_rgb_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        input_gt_depth_image = gt_var[0].data.squeeze().cpu().numpy().astype(np.float32)
        pred_depth_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)
        if test_norm:
            input_gt_depth_image = (input_gt_depth_image - np.min(input_gt_depth_image)) / (np.max(input_gt_depth_image) - np.min(input_gt_depth_image) )
            # input_gt_depth_image *= 10
            pred_depth_image = (pred_depth_image - np.min(pred_depth_image)) / (np.max(pred_depth_image) - np.min(pred_depth_image))
            # pred_depth_image *= 10.0
            # input_gt_depth_image /= 10.0
            # pred_depth_image /= 10.0

        # plt.imsave('./results/rgb/Test_input_rgb_{:05d}.png'.format(idx), input_rgb_image)
            plt.imsave('./results/gt/Test_gt_depth_{:05d}.png'.format(idx), input_gt_depth_image, cmap='jet')
            plt.imsave('./results/pred/Test_pred_depth_{:05d}.png'.format(idx), pred_depth_image, cmap='jet')
        # fig = plt.figure()
        # ii = plt.imshow(pred_depth_image, interpolation='nearest')
        # fig.colorbar(ii)
        # fig.savefig('./results/Test_pred_depth_{:05d}.png'.format(idx))

#         rmse = (gt_var - output) * (gt_var - output)
#         rmse = torch.mean(torch.mean(rmse, 2), 2)
#         rmse_linear += torch.mean(torch.sqrt(rmse))
#
#
#
        idx = idx + 1
        print('idx', idx, 'saved')
#
#         n = np.sum(input_gt_depth_image > 1e-2)
#
#         idxs = (input_gt_depth_image <= 1e-2)
#         pred_depth_image[idxs] = 1
#         input_gt_depth_image[idxs] = 1
#
#         # pred_depth_image_log_idx = pred_depth_image[]
#
#         pred_d_gt = pred_depth_image / input_gt_depth_image
#         pred_d_gt[idxs] = 100
#         gt_d_pred = input_gt_depth_image / pred_depth_image
#         gt_d_pred[idxs] = 100
#
#         Threshold_1_25 += np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25) / n
#         Threshold_1_25_2 += np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25) / n
#         Threshold_1_25_3 += np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25 * 1.25) / n
#
#         log_pred = np.log(pred_depth_image+0.001)
#         log_gt = np.log(input_gt_depth_image)
#
#         d_i = log_gt - log_pred
#
#         RMSE_linear += np.sqrt(np.sum((pred_depth_image - input_gt_depth_image) ** 2) / n)
#         RMSE_log += np.sqrt(np.sum((log_pred - log_gt) ** 2) / n)
#         RMSE_log_scale_invariant += np.sum(d_i ** 2) / n + (np.sum(d_i) ** 2) / (n ** 2)
#         ARD += np.sum(np.abs((pred_depth_image - input_gt_depth_image)) / input_gt_depth_image) / n
#         SRD += np.sum(((pred_depth_image - input_gt_depth_image) ** 2) / input_gt_depth_image) / n
#
#         # break
#
# Threshold_1_25 /= num_samples
# Threshold_1_25_2 /= num_samples
# Threshold_1_25_3 /= num_samples
# RMSE_linear /= num_samples
# RMSE_log /= num_samples
# RMSE_log_scale_invariant /= num_samples
# ARD /= num_samples
# SRD /= num_samples
# rmse_linear /=num_samples
#
# print('Threshold_1_25: {}'.format(Threshold_1_25))
# print('Threshold_1_25_2: {}'.format(Threshold_1_25_2))
# print('Threshold_1_25_3: {}'.format(Threshold_1_25_3))
# print('rmse: {}'.format(rmse_linear))
# print('RMSE_linear: {}'.format(RMSE_linear))
# print('RMSE_log: {}'.format(RMSE_log))
# print('RMSE_log_scale_invariant: {}'.format(RMSE_log_scale_invariant))
# print('ARD: {}'.format(ARD))
# print('SRD: {}'.format(SRD))
# print('result:\n{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n'.format(Threshold_1_25, Threshold_1_25_2, Threshold_1_25_3, RMSE_linear, ARD, SRD, RMSE_log))