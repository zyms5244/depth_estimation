
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np

def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    rgb_batch, depth_batch = sample_batched['rgb'], sample_batched['depth']
    batch_size = len(rgb_batch)
    im_size = rgb_batch.size(2)

    rgb_grid = utils.make_grid(rgb_batch)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(rgb_grid.numpy().transpose(1, 2, 0)
               * np.array([0.229, 0.224, 0.225])
               + np.array([0.485, 0.456, 0.406]))

    # print(depth_batch.shape)
    # depth_grid = utils.make_grid(depth_batch)
    # depth_batch = depth_batch.numpy()
    n,c,h,w = depth_batch.shape
    depth_grid = depth_batch.numpy().reshape(n,h,w).transpose(1,0,2)
    # print(depth_grid.shape)


    plt.subplot(2, 1, 2)
    plt.imshow(depth_grid.reshape(h,w*n))

    plt.title('Batch from dataloader')

    # plt.axis('off')
    # plt.ioff()
    plt.show()

def save_pred_depth():
    pass
import matplotlib.pyplot as plt
import numpy as np

def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    rgb_batch, depth_batch = sample_batched['rgb'], sample_batched['depth']
    batch_size = len(rgb_batch)
    im_size = rgb_batch.size(2)

    rgb_grid = utils.make_grid(rgb_batch)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(rgb_grid.numpy().transpose(1, 2, 0)
               * np.array([0.229, 0.224, 0.225])
               + np.array([0.485, 0.456, 0.406]))

    # print(depth_batch.shape)
    # depth_grid = utils.make_grid(depth_batch)
    # depth_batch = depth_batch.numpy()
    n,c,h,w = depth_batch.shape
    depth_grid = depth_batch.numpy().reshape(n,h,w).transpose(1,0,2)
    # print(depth_grid.shape)


    plt.subplot(2, 1, 2)
    plt.imshow(depth_grid.reshape(h,w*n), cmap='jet')

    plt.title('Batch from dataloader')

    # plt.axis('off')
    # plt.ioff()
    plt.show()

def save_pred_depth():
    pass