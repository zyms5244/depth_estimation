# -*- coding: utf-8 -*-
"""
loading NYU Depth
"""

from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.rgbdshow import show_landmarks_batch
import scipy.ndimage
import h5py
import random

import numpy as np
from data import flow_transforms,rgbd_transform

try:
    import cPickle as pickle
except ImportError:
    import pickle



def get_superpixel_relation(dptsp, idxs):
    h, w = dptsp.shape
    # print(dptsp.shape)
    dpt_sp = dptsp.ravel()
    centers = np.zeros((len(idxs), 2), dtype=int)

    for i, idx in enumerate(idxs):
        if dpt_sp[idx] < 0.001:
            centers[i] = 0
        else:
            centers[i] = np.mean(np.where((dpt_sp == dpt_sp[idx]).reshape(h, w)), axis = 1, dtype = int)
            # print((dpt_sp == dpt_sp[idxs[i]]).reshape(h, w))
            # print(np.where((dpt_sp == dpt_sp[idxs[i]]).reshape(h, w)))

    ordinate = []

    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            # print(dptspview[idxs[i]] , dptspview[idxs[j]])
            if dptsp[centers[i][0], centers[i][1]]==0 or dptsp[centers[j][0], centers[j][1]]==0:
                ordinate.append(0)
            else:
                if dptsp[centers[i][0], centers[i][1]] > dptsp[centers[j][0], centers[j][1]]:
                    ordinate.append(1)
                elif dptsp[centers[i][0], centers[i][1]] == dptsp[centers[j][0], centers[j][1]]:
                    ordinate.append(0)
                else:
                    ordinate.append(-1)
    # print(torch.from_numpy(centers).shape, torch.tensor(ordinate).shape)
    return centers, torch.tensor(ordinate, dtype=torch.float)

def generate_relative_pos(centers):
    print(centers.shape)
    nb, nr, nc = centers.shape
    Ah = []
    Aw = []
    Bh = []
    Bw = []
    for i in range(nr):
        for j in range(i + 1, nr):
            Ah.append(centers[i, 0])
            Aw.append(centers[i, 1])
            Bh.append(centers[j, 0])
            Bw.append(centers[j, 1])

    return Ah, Aw, Bh, Bw


class NYUDepthDataset(Dataset):
    """docstring for NYUDepthDataset"""

    def __init__(self, root_dir, stage, sample_num = None, superpixel = False, transform = False, relative = False):
        super(NYUDepthDataset, self).__init__()
        self.stage = stage
        self.dir = os.path.join(root_dir, stage)
        self.dirlist = os.listdir(self.dir)
        self.filelist = []
        self.transform = transform
        self.sample_num = sample_num
        self.superpixel = superpixel
        self.relative = relative


        cur_dir, _ = os.path.split(os.path.abspath(__file__))
        list_file = os.path.join(cur_dir, self.stage + '.pkl')
        if os.path.exists(list_file):
            with open(list_file,'rb') as f:
                self.filelist = pickle.load(f)
                print('load nyu file lists from pkl...')
        else:
            for nyudir in self.dirlist:
                for imgfile in os.listdir(os.path.join(self.dir, nyudir)):
                    self.filelist.append(os.path.join(self.dir, nyudir, imgfile))
            with open(list_file, 'wb') as f:
                pickle.dump(self.filelist, f)
        print('Total file number:', len(self.filelist))

        self.input_transform = transforms.Compose([
            rgbd_transform.Scale(240),
            transforms.ToTensor(),
            flow_transforms.TensorCenterCrop([240, 320]),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])

        self.output_transform = transforms.Compose([
            rgbd_transform.Scale_Single(240),
            rgbd_transform.ArrayToTensor(),
            rgbd_transform.TensorCenterCrop([240, 320]),
        ])

        self.output_transform_half = transforms.Compose([

            rgbd_transform.Scale_Single(224),
            rgbd_transform.ArrayToTensor(),
            # rgbd_transform.TensorCenterCrop([112, 144]),
        ])

        self.sample_transform = transforms.Compose([
            rgbd_transform.RandomHorizontalFlipRGBD(),
            rgbd_transform.RandomCropRGBD([224, 288])

        ])

    def filelist(self):
        return self.filelist

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        rgbd_file = h5py.File(self.filelist[idx], 'r')
        if self.sample_num is None:
            simple = {'rgb': rgbd_file['rgb'][:].transpose(1,2,0), 'depth': rgbd_file['depth'][:]}
        else:
            if self.superpixel:
                simple = {'rgb': rgbd_file['rgb'][:].transpose(1, 2, 0),
                          'depth': rgbd_file['dpt_sp%d' % self.sample_num][:]}

            else:
                # sample_idxs = rgbd_file['idx%d' % self.sample_num][:]
                sample_idxs = random.sample(list(range(480 * 640)), self.sample_num)
                dpt = rgbd_file['depth'][:]
                h, w = dpt.shape
                points = np.zeros(h * w, dtype=np.float32)
                dpt = dpt.reshape(h * w)

                points[sample_idxs] = dpt[sample_idxs]
                points = points.reshape([h, w])
                simple = {'rgb': rgbd_file['rgb'][:].transpose(1, 2, 0),
                          'depth': points}

            if self.relative:
                centers, ord = get_superpixel_relation(rgbd_file['dpt_sp%d' % self.sample_num][:],
                                                       rgbd_file['idx%d' % self.sample_num][:])
                simple['center'] = centers
                simple['ord'] = ord


        simple['rgb'] = self.input_transform(simple['rgb'])
        simple['depth'] = self.output_transform(simple['depth'])
        # if self.transform:

            # if self.relative:
            #     simple['center'] = simple['center'] * (112.0/480)
            # simple = self.sample_transform(simple)

        # tmp = simple['depth'].numpy()[0]

        # tmp = scipy.ndimage.zoom(tmp, 0.5)

        # simple['depth'] = torch.from_numpy(tmp).float().unsqueeze(0)

        # if self.relative:
        #     Ah, Aw, Bh, Bw = generate_relative_pos(simple['center'])
        #     simple['Ah'] = Ah
        #     simple['Aw'] = Aw
        #     simple['Bh'] = Bh
        #     simple['Bw'] = Bw
        return simple


#Test dataset
class NyuDepthMat(Dataset):
    def __init__(self, data_path, split_file_path, transform=False):
        self.data_path = data_path

        self.nyu = h5py.File(self.data_path)

        self.imgs = self.nyu['images']
        self.dpts = self.nyu['depths']

        with open(split_file_path, 'r') as f:
            lines = f.readlines()
            self.testidx = [int(l)-1 for l in lines]

        self.input_transform = transforms.Compose([

            rgbd_transform.Scale(224),
            # flow_transforms.ArrayToTensor(),
            transforms.ToTensor(),
            flow_transforms.TensorCenterCrop([224, 288]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.output_transform = transforms.Compose([

            rgbd_transform.Scale_Single(224),
            rgbd_transform.ArrayToTensor(),
            rgbd_transform.TensorCenterCrop([224, 288]),
        ])

        self.output_transform_half = transforms.Compose([

            rgbd_transform.Scale_Single(112),
            rgbd_transform.ArrayToTensor(),
            rgbd_transform.TensorCenterCrop([112, 144]),
        ])


    def __len__(self):
        return len(self.testidx)

    def __getitem__(self, index):
        # img_idx = self.lists[index]
        # print(index)
        img_idx = self.testidx[index]

        img = self.imgs[img_idx].transpose(2, 1, 0)
        # img = self.imgs[img_idx]
        dpt = self.dpts[img_idx].transpose(1, 0)
        #dpt = self.dpts[img_idx]

        # print(img.shape)
        # print(dpt.shape)
        img = self.input_transform(img)
        dpt = self.output_transform(dpt)

        #image = Image.fromarray(np.uint8(img))
        #image.save('img2.png')

        return {'rgb': img, 'depth': dpt}

def test1():
    nyu_train_set = NYUDepthDataset('/media/ans/Share/nyudepthv2/',
                                    'train',
                                    sample_num=20000,
                                    transform=True,
                                    superpixel=False,
                                    relative=False)
    dataloader = DataLoader(nyu_train_set, batch_size=1,
                            shuffle=True, num_workers=1)


    # from config import SDFCNConfig

    # cfg = SDFCNConfig()

    # test_set = NyuDepthMat('nyu_depth_v2_labeled.mat', 'testIdxs.txt')
    # dataloader = DataLoader(test_set, batch_size=4,
    #                          shuffle=False, drop_last=False)

    print(len(dataloader))
    #
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['rgb'].size(),
              sample_batched['depth'].size())
        # print(sample_batched)
        # observe 4th batch and stop
        show_landmarks_batch(sample_batched)
        # print(sample_batched['ord'].size())
        break

class relativeloss(torch.nn.Module):
    def __init__(self ):
        super(relativeloss, self).__init__()

    def forward(self, zA, zB, gt):
        rloss = 0
        mask = torch.abs(gt)
        rloss = mask * torch.log(1 + torch.exp(-gt * (zA - zB))) + (1 - mask) * (zA - zB) * (zA - zB)
        return rloss

def test2():
    from config import SDFCNConfig
    cfg = SDFCNConfig()
    dtype = torch.cuda.FloatTensor
    dataloader = torch.utils.data.DataLoader(NYUDepthDataset(cfg.trainval_data_root, 'train',
                                                               sample_num = cfg.sample_num,
                                                               superpixel = False,
                                                               relative = True,
                                                               transform = True),
                                               batch_size=cfg.batch_size, shuffle=True,
                                               num_workers=cfg.num_workers, drop_last=True)



    # test_set = NyuDepthMat('nyu_depth_v2_labeled.mat', 'testIdxs.txt')
    # dataloader = DataLoader(test_set, batch_size=4,
    #                          shuffle=False, drop_last=False)

    print(len(dataloader))
    #
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['rgb'].size())
        gt_var = sample_batched['depth'].type(dtype)

        print()


if __name__ == '__main__':
    test2()



