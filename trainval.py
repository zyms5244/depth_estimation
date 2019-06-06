import os
import torch
import torchvision
from torch.optim import lr_scheduler
from models import DUCNet,MADUCNet,ResDUCNet
from models.weights import load_weights
from data import NYUDepthDataset, NyuDepthMat
from torch.autograd import Variable
from utils.visualizer import Visualizer
from config import SDFCNConfig
import datetime
import numpy as np


cfg = SDFCNConfig()
dtype = torch.cuda.FloatTensor

class relativeloss(torch.nn.Module):
    def __init__(self, centers):
        self.centers = centers

    def forward(self, x, gt):
        mask = torch.abs(gt)
        nr, nc = self.centers.size()
        rloss = 0
        item = 0
        for i in range(nr):
            for j in range(i+1, nr):
                z_A = x[self.centers[i, 0].item, self.centers[i, 1].item]
                z_B = x[self.centers[j, 0].item, self.centers[j, 1].item]
                rloss += mask[item] * torch.log(1 + torch.exp(-gt[item] * (z_A - z_B))) + (1 - mask[item]) * (z_A - z_B) * (z_A - z_B)
                item = item + 1
        return rloss/item


class berHu(torch.nn.Module):
    def __init__(self):
        super(berHu, self).__init__()
        # self.epsilon = epsilon

    def forward(self, pred, gt):
        # mask = (gt > 0).detach()
        absdif = torch.abs(pred - gt)
        c = 0.2 * torch.max(absdif).detach()
        idx1 = (absdif <= c).detach()
        idx2 = (absdif > c).detach()
        l1_loss = torch.mean(absdif[idx1])
        l2_loss = torch.mean((absdif[idx2] * absdif[idx2] + c * c) / (2 * c + 1e-3))
        #
        # l1_loss = torch.nn.L1Loss()(pred[idx1], gt[idx1])
        # l2_loss = torch.nn.MSELoss()(pred[idx2], gt[idx2])

        return (l1_loss + l2_loss)

def validate(loader, model, loss, vis = None):
    # validate
    # model.eval()
    num_samples = 0
    loss_local = 0
    rmse_linear = 0
    model.eval()

    with torch.no_grad():
        for i_batch, sample_batched in enumerate(loader):
            input_var = Variable(sample_batched['rgb'].type(dtype))
            gt_var = Variable(sample_batched['depth'].type(dtype))
            # print(sample_batched['rgb'].shape)
            output = model(input_var)
            loss_local += loss(output, gt_var)
            num_samples += 1

            if vis:
                vis.depth('pred_val', output)
                vis.depth('depth_val', sample_batched['depth'].type(dtype))

            rmse = (gt_var - output) * (gt_var - output)
            rmse = torch.mean(torch.mean(rmse, 2), 2)
            # print(torch.sqrt(rmse))
            # print(torch.mean(torch.sqrt(rmse)))
            rmse_linear += torch.mean(torch.sqrt(rmse))


            # input_gt_depth_image = gt_var[0].data.squeeze().cpu().numpy().astype(np.float32)
            # pred_depth_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)

            # RMSE_linear += np.sqrt(np.mean((pred_depth_image - input_gt_depth_image) ** 2))


            # torchvision.utils.save_image(output, 'outimg.jpg', normalize=True)
            # input_rgb_image = input[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            # input_gt_depth_image = depth[0][0].data.cpu().numpy().astype(np.float32)
            # pred_depth_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)
            #
            # input_gt_depth_image /= np.max(input_gt_depth_image)
            # pred_depth_image /= np.max(pred_depth_image)
            # plot.imsave('input_rgb_epoch_0.jpg', input_rgb_image)
            # plot.imsave('gt_depth_epoch_0.jpg', input_gt_depth_image, cmap="viridis")
            # plot.imsave('pred_depth_epoch_0.jpg', pred_depth_image, cmap="viridis")


        err = float(loss_local) / num_samples
        RMSE_linear = float(rmse_linear.item()) / num_samples
        print('val_error:{} RMSE:{}'.format(err, RMSE_linear))

    return err, RMSE_linear


def main():
    # load data
    train_loader = torch.utils.data.DataLoader(NYUDepthDataset(cfg.trainval_data_root, 'train',
                                                               sample_num = cfg.sample_num,
                                                               superpixel = False,
                                                               relative = False,
                                                               transform = True),
                                               batch_size=cfg.batch_size, shuffle=True,
                                               num_workers=cfg.num_workers, drop_last=True)
    print('Train Batches:', len(train_loader))

    # val_loader = torch.utils.data.DataLoader(NYUDepthDataset(cfg.trainval_data_root, 'val', transform=True),
    #                                          batch_size=cfg.batch_size, shuffle=True,
    #                                          num_workers=cfg.num_workers, drop_last=True)
    # print('Validation Batches:', len(val_loader))

    test_set = NyuDepthMat(cfg.test_data_root, '/home/ans/PycharmProjects/SDFCN/data/testIdxs.txt')
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=cfg.batch_size,
                                              shuffle=True, drop_last=True)

    # train_set = NyuDepthMat(cfg.test_data_root, '/home/ans/PycharmProjects/SDFCN/data/trainIdxs.txt')
    # train_loader = torch.utils.data.DataLoader(train_set,
    #                                           batch_size=cfg.batch_size,
    #                                           shuffle=True, drop_last=True)
    # train_loader = test_loader
    #
    val_loader = test_loader
    # load model and weight
    # model = FCRN(cfg.batch_size)
    model = ResDUCNet(model=torchvision.models.resnet50(pretrained=False))
    init_upsample = False
    # print(model)


    loss_fn = berHu()

    if cfg.use_gpu:
        print('Use CUDA')
        model = model.cuda()
        # loss_fn = berHu().cuda()
        # loss_fn = torch.nn.MSELoss().cuda()
        loss_fn = torch.nn.L1Loss().cuda()

    start_epoch = 0
    best_val_err = 10e3



    if cfg.resume_from_file:
        if os.path.isfile(cfg.resume_file):
            print("=> loading checkpoint '{}'".format(cfg.resume_file))
            checkpoint = torch.load(cfg.resume_file)
            # start_epoch = checkpoint['epoch']
            start_epoch = 0
            # model.load_state_dict(checkpoint['state_dict'])
            model.load_state_dict(checkpoint['model_state'])
            # print("=> loaded checkpoint '{}' (epoch {})"
            #       .format(cfg.resume_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(cfg.resume_file))
    # else:
    #     if init_upsample:
    #         print('Loading weights from ', cfg.weights_file)
    #         # bone_state_dict = load_weights(model, cfg.weights_file, dtype)
    #         model.load_state_dict(load_weights(model, cfg.weights_file, dtype))
    #     else:
    #         print('Loading weights from ', cfg.resnet50_file)
    #         pretrained_dict = torch.load(cfg.resnet50_file)
    #         model_dict = model.state_dict()
    #         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #         model_dict.update(pretrained_dict)
    #         model.load_state_dict(model_dict)
    #     print('Weights loaded.')

    # val_error, val_rmse = validate(val_loader, model, loss_fn)
    # print('before train: val_error %f, rmse: %f' % (val_error, val_rmse))

    vis = Visualizer(cfg.env)
    # 4.Optim
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    print("optimizer set.")
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.step, gamma= cfg.lr_decay)


    for epoch in range(cfg.num_epochs):

        scheduler.step()
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        print('Starting train epoch %d / %d, lr=%f' % (start_epoch + epoch + 1, cfg.num_epochs,
                                                       optimizer.state_dict()['param_groups'][0]['lr']))

        model.train()
        running_loss = 0
        count = 0
        epoch_loss = 0

        for i_batch, sample_batched in enumerate(train_loader):
            input_var = Variable(sample_batched['rgb'].type(dtype))
            depth_var = Variable(sample_batched['depth'].type(dtype))

            optimizer.zero_grad()
            output = model(input_var)
            loss = loss_fn(output, depth_var)

            if i_batch % cfg.print_freq == cfg.print_freq - 1:
                print('{0} batches, loss:{1}'.format(i_batch+1, loss.data.cpu().item()))
                vis.plot('loss', loss.data.cpu().item())

            if i_batch % (cfg.print_freq*10) == (cfg.print_freq*10) - 1:
                vis.depth('pred', output)
                # vis.imshow('img', sample_batched['rgb'].type(dtype))
                vis.depth('depth', sample_batched['depth'].type(dtype))

            count += 1
            running_loss += loss.data.cpu().numpy()


            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / count
        print('epoch loss:', epoch_loss)


        val_error, val_rmse = validate(val_loader, model, loss_fn, vis=vis)
        vis.plot('val_error', val_error)
        vis.plot('val_rmse', val_rmse)
        vis.log('epoch:{epoch},lr={lr},epoch_loss:{loss},val_error:{val_cm}'.format(epoch=start_epoch + epoch + 1,
                                                                                    loss=epoch_loss,
                                                                                    val_cm=val_error,
                                                                                    lr=optimizer.state_dict()['param_groups'][0]['lr']))

        if val_error < best_val_err:
            best_val_err = val_error
            if not os.path.exists(cfg.checkpoint_dir):
                os.mkdir(cfg.checkpoint_dir)

            torch.save({
                'epoch': start_epoch + epoch + 1,
                'state_dict': model.state_dict(),
                # 'optimitezer': optimizer.state_dict(),
            }, os.path.join(cfg.checkpoint_dir,'{}_{}_epoch_{}_{}'.format(cfg.checkpoint,
                                                                          cfg.env, start_epoch + epoch + 1,
                                                                          cfg.checkpoint_postfix)))


    torch.save({
        'epoch': start_epoch + epoch + 1,
        'state_dict': model.state_dict(),
        # 'optimitezer': optimizer.state_dict(),
    }, os.path.join(cfg.checkpoint_dir,'{}_{}_epoch_{}_{}'.format(cfg.checkpoint,
                                                                  cfg.env, start_epoch + epoch + 1,
                                                                  cfg.checkpoint_postfix)))




if __name__ == '__main__':
    main()

