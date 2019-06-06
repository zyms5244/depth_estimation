import h5py
import config
import random
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
import os
import math

import sys
sys.path.append('/home/ans/Downloads/opencv/lib/python3.6/site-packages/')
import cv2 as cv


def vis_sample_depth(file, sample_num = 300):
    plt.figure()
    f = h5py.File(file)
    dpt = f['depth'][:]
    points = f['dpt%d' % sample_num][:]
    dpt_sp = f['dpt_sp%d' % sample_num][:]
    vis = np.vstack((dpt, points, dpt_sp))
    f.close()
    plt.imshow(vis)
    plt.show()



def sample_depth(file, sample_num = 300):

    f = h5py.File(file)
    # print(list(f.keys()))
    dpt = f['depth'][:]
    h,w = dpt.shape
    # print(dpt.shape)

    # idxs = random.sample(list(range(h * w)), sample_num)
    # idxs = np.sort(idxs)
    idxs = f['idx%d' % sample_num][:]
    # print(idxs)


    # points = _point_depth(f, idxs)

    dpt_sp = _superpixel_depth(f, idxs, sample_num=sample_num)


    # try:
        # del f['idx%d' % 1000]
        # del f['dpt_sp%d' % 1000]
        # del f['dpt_sp%d' % 300]

    # except:
    #     print('%s has no keys' % file)


    # f.create_dataset('idx%d' % sample_num, data=idxs)
    # f.create_dataset('dpt%d' % sample_num, data=points)
    f.create_dataset('dpt_sp%d' % sample_num, data=dpt_sp)

    f.close()

    # f['idx%d' % num] = idxs
    # f['dpt%d' % num] = points

def _point_depth(h5file, idxs):

    dpt = h5file['depth'][:]
    h, w = dpt.shape
    points = np.zeros(h * w, dtype=np.float32)
    dpt = dpt.reshape(h * w)

    points[idxs] = dpt[idxs]
    points = points.reshape([h, w])
    # print(points.shape)
    return points

def _superpixel_depth(h5file, idx, sample_num = 300):

    dpt = h5file['depth'][:]
    img = h5file['rgb'][:].transpose(1, 2, 0)
    # dpt_point = h5file['dpt%d' % sample_num][:]
    # print(img.shape)
    img = img[..., ::-1]
    sp_num, lables = _generate_superpixel(img, sample_num=sample_num)
    h,w,c = img.shape
    # print(img.shape)
    # cv.imshow('rgb', img)
    # cv.waitKey()
    # plt.imshow(lables)
    lables_view = lables.view()
    lables_view.shape = lables_view.size
    dpt_view = dpt.view()
    dpt_view.shape = dpt.size
    # print('sp:',sp_num)
    sp_num = min(sp_num, sample_num)
    dpt_sp = np.zeros((h, w), dtype=np.float32)

    for i in range(sp_num):
        # f = idx[i]
        spl = lables_view[idx[i]]
        dpt_sp[lables==spl] = dpt_view[idx[i]]

    # plt.imshow(dpt_sp)
    # plt.show()

    return dpt_sp


def _generate_superpixel(img, sample_num = 300):

    img = cv.GaussianBlur(img, (3, 3), 0)
    img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    h, w, _ = img.shape
    sp_size = int(math.sqrt(h * w / sample_num))
    # print(img.shape)
    # print(sp_size)
    slic = cv.ximgproc.createSuperpixelSLIC(img, cv.ximgproc.SLIC, sp_size)
    slic.iterate(10)
    slic.enforceLabelConnectivity(sp_size)
    sp_num = slic.getNumberOfSuperpixels()
    lables = slic.getLabels()
    return (sp_num, lables)

def generate_sample_depth(root_dir, sample_num = 300):
    dir = os.path.join(root_dir, 'train')
    dirlist = os.listdir(dir)
    count = 0

    for nyudir in dirlist:
        for imgfile in os.listdir(os.path.join(dir, nyudir)):
            print(os.path.join(dir, nyudir, imgfile))
            count += 1
            sample_depth(os.path.join(dir, nyudir, imgfile), sample_num)
            print('[%5d]' % count, imgfile, 'generate depth done.')
            # break
        # break
    # print(count)

def get_superpixel_center(dptsp, idxs):
    h, w = dptsp.shape
    print(dptsp.shape)
    dpt_sp = dptsp.ravel()
    centers = np.zeros((len(idxs), 2), dtype=int)

    # def vis1(img, points):
    #     plt.figure()
    #     plt.imshow(img)
    #     plt.scatter(points[1], points[0], color='r')
    #     plt.show()
    # def vis(img, points):
    #     plt.figure()
    #     plt.imshow(img)
    #     plt.scatter(points[:, 1], points[:, 0], color='r')
    #     plt.show()

    for i in range(len(idxs)):
        if dpt_sp[idxs[i]] < 0.001:
            centers[i] = 0
        else:
            centers[i] = np.mean(np.where((dpt_sp == dpt_sp[idxs[i]]).reshape(h, w)), axis = 1, dtype = int)

    ordinate = []

    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            # print(dptspview[idxs[i]] , dptspview[idxs[j]])
            if dptsp[centers[i][0], centers[i][1]]==0 or dptsp[centers[j][0], centers[j][1]]==0:
                ordinate.append(0)
            else:
                if dpt_sp[centers[i][0], centers[i][1]] > dpt_sp[centers[j][0], centers[j][1]]:
                    ordinate.append(1)
                elif dpt_sp[centers[i][0], centers[i][1]] == dpt_sp[centers[j][0], centers[j][1]]:
                    ordinate.append(0)
                else:
                    ordinate.append(-1)

    return centers, ordinate




if __name__ == '__main__':
    cfg = config.SDFCNConfig
    # sample_depth('/home/ans/PycharmProjects/supixel-depth/00001.h5', sample_num = cfg.sample_num)
    # vis_sample_depth('/home/ans/PycharmProjects/supixel-depth/00001.h5', sample_num = cfg.sample_num)
    # generate_sample_depth(cfg.trainval_data_root, sample_num = cfg.sample_num)
    f = h5py.File('/home/ans/PycharmProjects/supixel-depth/00001.h5')
    print(list(f.keys()))
    dpt = f['depth'][:]
    dpt_sp = f['dpt_sp300'][:]
    idxs = f['idx300'][:]
    # ord = generate_relationship(dpt_sp, idxs)

    # print(ord)
    centers, ord = get_superpixel_center(dpt_sp, idxs)
    print(centers.shape)
    print(len(ord))
    # print(centers)
