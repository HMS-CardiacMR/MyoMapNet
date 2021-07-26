import matplotlib.pyplot as plt
import torch
import numpy as np
from parameters import Parameters
params = Parameters()
from utils.cmplxBatchNorm import magnitude
from matplotlib.path import Path
from saveNet import *

def get_ROI_conts(epi_cont, endo_cont = None, s=0.05):
    epi_cont = (epi_cont - np.mean(epi_cont, 0)) * (1-s) + np.mean(epi_cont, 0)
    if endo_cont is None:
        return epi_cont
    endo_cont = (endo_cont - np.mean(endo_cont, 0)) * (1+s) + np.mean(endo_cont, 0)
    return epi_cont, endo_cont

def get_mask(out_cont, in_cont=None, grid_size=[200,200]):
    nx, ny = grid_size[0], grid_size[1]

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    path = Path(out_cont)
    grid = path.contains_points(points)
    mask_out = grid.reshape((ny, nx))

    if in_cont is None:
        return mask_out

    path = Path(in_cont)
    grid = path.contains_points(points)
    mask_in = grid.reshape((ny, nx))

    mask = mask_out - mask_in
    return mask

def get_6segment_masks(msk, cont):
    mc = np.mean(cont, 0)
    seg_msks = np.zeros((6,msk.shape[0],msk.shape[1]))
    for s in range(0,6):
        l = np.stack([np.linspace(0,msk.shape[0],msk.shape[0]), np.zeros(200)])
        ang = s * (2 * np.pi) / 6
        rot = np.array([[np.sin(ang), np.cos(ang)], [-np.cos(ang), np.sin(ang)]])
        rl1 = np.matmul(rot, l)
        ang = (s+1) * (2 * np.pi) / 6
        rot = np.array([[np.sin(ang), np.cos(ang)], [-np.cos(ang), np.sin(ang)]])
        rl2 = np.matmul(rot, l)
        rl = np.stack([np.flip(rl1,1), rl2],1).reshape((2, msk.shape[0]*2)) + np.expand_dims(mc,1).repeat(msk.shape[0]*2, 1)

        ang_msk = get_mask(rl.swapaxes(0,1), grid_size=[2*msk.shape[0], 2*msk.shape[1]])
        seg_msks[s, ] = ang_msk[:200,:200] * msk
        # plt.imshow(ang_msk[:200,:200] + msk)
        # plt.show()

    return seg_msks

def combine_coils_RSOS(data):
    return torch.sqrt(torch.sum(magnitude(data) ** 2, dim=1, keepdim=True))

def tensorshow(x, sl_dims=(0, 0), rng=[0, 0.001]):
    plt.figure()

    if type(x) is np.ndarray:
        if np.iscomplexobj(x):
            x = np.abs(x)[sl_dims[0], sl_dims[1], :, :]
        else:
            x = np.sqrt(x[sl_dims[0], sl_dims[1], :, :, 0] ** 2 +
                               x[sl_dims[0], sl_dims[1], :, :, 1] ** 2)
    else:
        x = magnitude(x).cpu().data.numpy()[sl_dims[0], sl_dims[1], :, :] 

    plt.imshow(x, cmap='gray', vmin=rng[0], vmax=rng[1])
    plt.show()

def ntensorshow(x, sl_dims=(0, 0), rng=[0, 0.001], titles=None, saveFigs=False,figname=None):

    n_slices = x[0].shape[0]
    if figname is None:
        figname = [str(i).zfill(3) for i in range(0,n_slices)]

    for sl in range(0,n_slices):
        fig, axs = plt.subplots(1, len(x))
        i = 0
        for ax in axs:
            if titles is not None:
                ax.set_title(titles[i])

            img = x[i]
            if img.shape[1] > 1:
                img = combine_coils_RSOS(img)[sl, sl_dims[1], :, :]
            if img.shape[-1] == 2:
                img = magnitude(img)
            img = img[sl, sl_dims[1], :, :]
            # mx = 1*torch.max(torch.reshape(img, [img.numel()])) ## could be improved
            # mn = torch.min(torch.reshape(img, [img.numel()]))
            img_mean = torch.mean(torch.reshape(img, [img.numel()]))
            img_std = torch.std(torch.reshape(img, [img.numel()]))

            saveTensorToMat(img, 'map', figname[sl]+titles[i]+'.mat', params.tensorboard_dir)

            # if i < 1:
            #     ax.imshow(img.cpu().data.numpy(), cmap='gray', vmin=img_mean-1*img_std, vmax=img_mean+3*img_std)
            # else:
            #     ax.imshow(img.cpu().data.numpy(), cmap='jet', vmin=0, vmax=2000)
            ax.imshow(img.cpu().data.numpy(), cmap='jet', vmin=0, vmax=2400)
            # ax.imshow(img.cpu().data.numpy(), cmap='gray', vmin=rng[0], vmax=rng[1])
            ax.axis('off')

            # ## save seperate figures
            # sep_fig, sep_axs = plt.subplots(1, 1)
            # sep_axs.imshow(img.cpu().data.numpy(), cmap='gray', vmin=img_mean-1*img_std, vmax=img_mean+3*img_std)
            # sep_axs.set_title(titles[i])
            # sep_axs.axis('off')
            # sep_fig.savefig(params.tensorboard_dir + titles[i]+'_'+ figname[sl] + '.png', dpi=300)
            # plt.close(sep_fig)

            i += 1

        if saveFigs:
            fig.savefig(params.tensorboard_dir + figname[sl] +'.png', dpi=300)
            plt.close(fig)
        else:
            fig.show()
            fig.canvas.flush_events()

    # fig.frame.Maximize(True)
    # fig.pause(0.01)