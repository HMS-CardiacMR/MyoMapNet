import numpy as np
from numpy.lib.stride_tricks import as_strided

from utils import magnitude
from math import exp
from utils.gaussian_fit import gauss_fit, kspacelines_gauss_fit, kspaceImg_gauss_fit
from utils.data_vis import get_mask, get_ROI_conts
from utils.polygon import intersection

def soft_thresh(u, lmda):
    """Soft-threshing operator for complex valued input"""
    Su = (abs(u) - lmda) / abs(u) * u
    Su[abs(u) < lmda] = 0
    return Su


def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def var_dens_mask(shape, ivar, sample_high_freq=True):
    """Variable Density Mask (2D undersampling)"""
    if len(shape) == 3:
        Nt, Nx, Ny = shape
    else:
        Nx, Ny = shape
        Nt = 1

    pdf_x = normal_pdf(Nx, ivar)
    pdf_y = normal_pdf(Ny, ivar)
    pdf = np.outer(pdf_x, pdf_y)

    size = pdf.itemsize
    strided_pdf = as_strided(pdf, (Nt, Nx, Ny), (0, Ny * size, size))
    # this must be false if undersampling rate is very low (around 90%~ish)
    if sample_high_freq:
        strided_pdf = strided_pdf / 1.25 + 0.02
    mask = np.random.binomial(1, strided_pdf)

    xc = Nx / 2
    yc = Ny / 2
    mask[:, xc - 4:xc + 5, yc - 4:yc + 5] = True

    if Nt == 1:
        return mask.reshape((Nx, Ny))

    return mask


def cartesian_mask(shape, acc, sample_n=10, centred=False):
    """
    Sampling density estimated from implementation of kt FOCUSS
    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..
    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = mymath.ifftshift(mask, axes=(-1, -2))

    return mask


def shear_grid_mask(shape, acceleration_rate, sample_low_freq=True,
                    centred=False, sample_n=10):
    '''
    Creates undersampling mask which samples in sheer grid
    Parameters
    ----------
    shape: (nt, nx, ny)
    acceleration_rate: int
    Returns
    -------
    array
    '''
    Nt, Nx, Ny = shape
    start = np.random.randint(0, acceleration_rate)
    mask = np.zeros((Nt, Nx))
    for t in xrange(Nt):
        mask[t, (start+t)%acceleration_rate::acceleration_rate] = 1

    xc = Nx / 2
    xl = sample_n / 2
    if sample_low_freq and centred:
        xh = xl
        if sample_n % 2 == 0:
            xh += 1
        mask[:, xc - xl:xc + xh+1] = 1

    elif sample_low_freq:
        xh = xl
        if sample_n % 2 == 1:
            xh -= 1

        if xl > 0:
            mask[:, :xl] = 1
        if xh > 0:
            mask[:, -xh:] = 1

    mask_rep = np.repeat(mask[..., np.newaxis], Ny, axis=-1)
    return mask_rep


def perturbed_shear_grid_mask(shape, acceleration_rate, sample_low_freq=True,
                              centred=False,
                              sample_n=10):
    Nt, Nx, Ny = shape
    start = np.random.randint(0, acceleration_rate)
    mask = np.zeros((Nt, Nx))
    for t in xrange(Nt):
        mask[t, (start+t)%acceleration_rate::acceleration_rate] = 1

    # brute force
    rand_code = np.random.randint(0, 3, size=Nt*Nx)
    shift = np.array([-1, 0, 1])[rand_code]
    new_mask = np.zeros_like(mask)
    for t in xrange(Nt):
        for x in xrange(Nx):
            if mask[t, x]:
                new_mask[t, (x + shift[t*x])%Nx] = 1

    xc = Nx / 2
    xl = sample_n / 2
    if sample_low_freq and centred:
        xh = xl
        if sample_n % 2 == 0:
            xh += 1
        new_mask[:, xc - xl:xc + xh+1] = 1

    elif sample_low_freq:
        xh = xl
        if sample_n % 2 == 1:
            xh -= 1

        new_mask[:, :xl] = 1
        new_mask[:, -xh:] = 1
    mask_rep = np.repeat(new_mask[..., np.newaxis], Ny, axis=-1)

    return mask_rep


def undersample(x, mask, centred=False, norm='ortho', noise=0):
    '''
    Undersample x. FFT2 will be applied to the last 2 axis
    Parameters
    ----------
    x: array_like
        data
    mask: array_like
        undersampling mask in fourier domain
    norm: 'ortho' or None
        if 'ortho', performs unitary transform, otherwise normal dft
    noise_power: float
        simulates acquisition noise, complex AWG noise.
        must be percentage of the peak signal
    Returns
    -------
    xu: array_like
        undersampled image in image domain. Note that it is complex valued
    x_fu: array_like
        undersampled data in k-space
    '''
    assert x.shape == mask.shape
    # zero mean complex Gaussian noise
    noise_power = noise
    nz = np.sqrt(.5)*(np.random.normal(0, 1, x.shape) + 1j * np.random.normal(0, 1, x.shape))
    nz = nz * np.sqrt(noise_power)

    if norm == 'ortho':
        # multiplicative factor
        nz = nz * np.sqrt(np.prod(mask.shape[-2:]))
    else:
        nz = nz * np.prod(mask.shape[-2:])

    if centred:
        x_f = mymath.fft2c(x, norm=norm)
        x_fu = mask * (x_f + nz)
        x_u = mymath.ifft2c(x_fu, norm=norm)
        return x_u, x_fu
    else:
        x_f = mymath.fft2(x, norm=norm)
        x_fu = mask * (x_f + nz)
        x_u = mymath.ifft2(x_fu, norm=norm)
        return x_u, x_fu





import torch
from torch.utils import data
from parameters import Parameters
from scipy.io import loadmat, savemat
import numpy as np
import os
from saveNet import *
#from utils.gridkspace import *
from utils.gaussian_fit import gauss_fit, kspacelines_gauss_fit, kspaceImg_gauss_fit
from scipy.optimize import curve_fit
from scipy import interpolate as interp
import scipy as sp
from matplotlib.path import Path
# params = Parameters()

def resizeImage(img, newSize, Interpolation=False):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)

    if Interpolation:
        return imresize(img, tuple(newSize), interp='bilinear')
    else:

        x1 = (img.shape[0] - newSize[0]) // 2
        x2 = img.shape[0] - newSize[0] - x1

        y1 = (img.shape[1] - newSize[1]) // 2
        y2 = img.shape[1] - newSize[1] - y1

        if img.ndim == 3:
            if x1 > 0:
                img = img[x1:-x2, :, :]
            elif x1 < 0:
                img = np.pad(img, ((-x1, -x2), (0, 0), (0, 0)), 'constant')  # ((top, bottom), (left, right))

            if y1 > 0:
                img = img[:, y1:-y2, :]
            elif y1 < 0:
                img = np.pad(img, ((0, 0), (-y1, -y2), (0, 0)), 'constant')  # ((top, bottom), (left, right))

        elif img.ndim == 4:
            if x1 > 0:
                img = img[x1:-x2, :, :, :]
            elif x1 < 0:
                img = np.pad(img, ((-x1, -x2), (0, 0), (0, 0), (0, 0)), 'constant')  # ((top, bottom), (left, right))

            if y1 > 0:
                img = img[:, y1:-y2, :, :]
            elif y1 < 0:
                img = np.pad(img, ((0, 0), (-y1, -y2), (0, 0), (0, 0)), 'constant')  # ((top, bottom), (left, right))
        return img.squeeze()


def resize3DVolume(data, newSize, Interpolation=False):
    ndim = data.ndim
    if ndim < 3:
        return None
    elif ndim == 3:
        data = np.expand_dims(data, 3)

    if Interpolation:
        return imresize(data, tuple(newSize), interp='bilinear')

    elif ndim == 4:
        x1 = (data.shape[0] - newSize[0]) // 2
        x2 = data.shape[0] - newSize[0] - x1

        y1 = (data.shape[1] - newSize[1]) // 2
        y2 = data.shape[1] - newSize[1] - y1

        z1 = (data.shape[2] - newSize[2]) // 2
        z2 = data.shape[2] - newSize[2] - z1

        if x1 > 0:
            data = data[x1:-x2, :, :, :]
        elif x1 < 0:
            data = np.pad(data, ((-x1, -x2), (0, 0), (0, 0), (0, 0)), 'constant')  # ((top, bottom), (left, right))

        if y1 > 0:
            data = data[:, y1:-y2, :, :]
        elif y1 < 0:
            data = np.pad(data, ((0, 0), (-y1, -y2), (0, 0), (0, 0)), 'constant')  # ((top, bottom), (left, right))

        if z1 > 0:
            data = data[:, :, z1:-z2, :]
        elif z1 < 0:
            data = np.pad(data, ((0, 0), (0, 0), (-z1, -z2), (0, 0)), 'constant')  # ((top, bottom), (left, right))

        return data.squeeze()


def getPatientSlicesURLs(patient_url):
    islices = list()
    oslices = list()
    for fs in os.listdir(patient_url + '/InputData/Input_realAndImag/'):
        islices.append(patient_url + '/InputData/Input_realAndImag/' + fs)

    for fs in os.listdir(patient_url + '/CSRecon/CSRecon_Data_small/'):
        oslices.append(patient_url + '/CSRecon/CSRecon_Data_small/' + fs)
    islices = sorted(islices, key=lambda x: int((x.rsplit(sep='/')[-1])[8:-4]))
    oslices = sorted(oslices, key=lambda x: int((x.rsplit(sep='/')[-1])[8:-4]))

    return (islices, oslices)


def getDatasetGenerators(params):
    params.num_slices_per_patient = []
    params.input_slices = []
    params.groundTruth_slices = []
    params.us_rates = []
    params.patients = []
    params.training_patients_index = []

    ds_url = params.dir[0] + 'T1Dataset_40_MOLLI_MIRT_NUFFT_GoldenAng_graded/'
    datasets_dirs = sorted(os.listdir(ds_url), key=lambda x: int(x.split('_')[-1][:-4]))
    for i, dst in enumerate(datasets_dirs):
        params.patients.append(dst[2:-5])
        params.groundTruth_slices.append(ds_url + dst)


    # for dir in params.dir:
    #     datasets_dirs = sorted(os.listdir(dir + 'T1Dataset/'), key=lambda x: int(x[:-4]))
    #     for i, dst in enumerate(datasets_dirs):
    #         params.patients.append(dst)
    #         kspaces = sort_files(os.listdir(dir + 'kspace/' + dst))
    #         params.num_slices_per_patient.append(len(kspaces))
    #
    #         for j, ksp in enumerate(kspaces):
    #             params.input_slices.append(dir + 'kspace/' + dst + '/' + ksp)
    #
    #         '''read all 16 coils from DAT file'''
    #         images = sort_files(os.listdir(dir + 'image/' + dst))
    #         for j, img in enumerate(images):
    #             params.groundTruth_slices.append(dir + 'image/' + dst + '/' + img)
    #
    #         '''read coil-combined 1-channel complex-valued data from .mat files'''
    # #             images = sort_files(os.listdir(dir + 'ref/' + dst))
    # #             for j, img in enumerate(images):
    # #                 params.groundTruth_slices.append(dir + 'ref/' + dst + '/' + img)

    print('-- Number of Datasets: ' + str(len(params.patients)))

    # params.groundTruth_slices = list()
    #
    # for i in range(1, 211):
    #     params.patients.append(params.dir[0] + 'T1Dataset/' + str(i)+'.mat')
    #     for sl in range(1, 6):
    #         params.groundTruth_slices.append(params.dir[0] + 'T1Dataset/' + str(i) + '_' + str(sl) + '.mat')


    ## in case of moving window
    # for i in range(1, 211):
    #     params.patients.append(params.dir[0] + 'T1Dataset/' + str(i)+'.mat')
    #     for sl in range(1, 6):
    #         for it in range(1, 12):
    #             params.groundTruth_slices.append(params.dir[0] + 'T1Dataset/' + str(i) + '_' + str(sl) + '_' + str(it) + '.mat')
    params.patients = params.patients[0::3]
    params.input_slices = params.groundTruth_slices
    training_ptns = int(params.training_percent * len(params.patients))

    training_end_indx = training_ptns * 3

    # training_end_indx = training_ptns * 5 * 11

    params.training_patients_index = range(0, training_ptns + 1)

    dim = params.img_size[:]
    dim.append(2)

    tr_samples = 1

    ## remove the corrupted T1 maps from training
    in_train_urls =  params.input_slices[:training_end_indx:tr_samples]
    grades = [int(x.split('/')[-1].split('_')[0]) for x in in_train_urls]
    in_train_urls = list(np.array(in_train_urls)[np.array(grades)==1])


    training_DS = DataGenerator(input_IDs=in_train_urls,
                                output_IDs=in_train_urls,
                                params=params
                                )

    validation_DS = DataGenerator(input_IDs=params.input_slices[training_end_indx:],
                                  output_IDs=params.groundTruth_slices[training_end_indx:],
                                  params=params
                                  )



    training_DL = data.DataLoader(training_DS, batch_size=params.batch_size, shuffle=True,
                                  num_workers=params.data_loders_num_workers)
    #     validation_DL = data.DataLoader(validation_DS, batch_size=params.batch_size, shuffle=False, num_workers=params.data_loders_num_workers)
    validation_DL = data.DataLoader(validation_DS, batch_size=params.batch_size, shuffle=False,
                                    num_workers=params.data_loders_num_workers)

    return training_DL, validation_DL, params


def get_gaussian_mask(window_size, sigma, center=None, circ_window_radius=None):

    if center is None:
        center = [window_size // 2, window_size // 2]

    gauss_2d = np.zeros((window_size, window_size))
    [Xm, Ym] = np.meshgrid(np.linspace(1, window_size, window_size), np.linspace(1, window_size, window_size))
    gauss_x = np.asarray([exp(-(x-center[0])**2/float(2*sigma[0]**2)) for x in range(window_size)])
    gauss_y = np.asarray([exp(-(x-center[1])**2/float(2*sigma[1]**2)) for x in range(window_size)])
    gauss2= np.dot(np.expand_dims(gauss_x, 1), np.expand_dims(gauss_y, 1).T)
    if circ_window_radius is not None:
        gauss2[((Xm-window_size//2)**2+(Ym-window_size//2)**2)**0.5 <= circ_window_radius[0]] = 1

    return gauss2


def magnitude(input):
    real, imag = torch.unbind(input, -1)
    return (real ** 2 + imag ** 2) ** 0.5

def t1fit_2p(MTi, *T1):
    return np.abs(T1[0]*(1-2*np.exp(-MTi/T1[1])))

def t1fit(MTi, T1):
    return MTi[0]*(1-2*np.exp(-MTi[1:]/T1))

def polyArea(p):
    x, y = p[:,0], p[:,1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def create_mesh(cont):
    endo, epi = cont[0,], cont[1,]
    if polyArea(endo) > polyArea(epi):
        tmp = endo; endo = epi; epi = tmp

    endo, epi = np.concatenate((endo, endo[0:1,:])), np.concatenate((epi, epi[0:1,:]))

    cs_endo = interp.CubicSpline(np.linspace(0, endo.shape[-2], endo.shape[-2]), endo, bc_type='periodic')
    cs_epi = interp.CubicSpline(np.linspace(0, epi.shape[-2], epi.shape[-2]), epi, bc_type='periodic')

    xs =np.linspace(0,61,800)
    endo, epi = cs_endo(xs), cs_epi(xs)

    mean_c = (np.mean(epi, axis=0) + np.mean(endo, axis=0)) / 2
    # l_endo = clip_line_by_polygon()

    l = np.zeros((2, 2)); l[0, :] = mean_c; l[1, 1] = mean_c[1];
    intrsct_endo = intersection(l[:,0],l[:,1], endo[:,0], endo[:,1])
    intrsct_epi = intersection(l[:, 0], l[:, 1], epi[:, 0], epi[:, 1])

    def start_cont(cont, p):
        d = cont - p
        d = np.sqrt(d[:,0]**2 + d[:,1]**2)
        cont_start = np.where(d == np.min(d))[0][0]
        if cont_start > 0:
            cont = np.concatenate((cont[cont_start:], cont[:cont_start]))
        return cont

    endo = start_cont(endo, (intrsct_endo[0][0], intrsct_endo[1][0]))
    epi = start_cont(epi, (intrsct_epi[0][0], intrsct_epi[1][0]))

    epi_s = epi - mean_c
    endo_s = endo - mean_c

    nm = int(np.pi * 200)
    mesh_1 = np.asarray([endo_s*i + mean_c for i in np.linspace(0,1,nm//3)])
    mesh_2 = np.asarray([endo_s*i + epi_s*(1-i) + mean_c for i in np.linspace(0,1,nm//3)])
    mesh_3 = np.asarray([epi_s*i + epi_s + mean_c for i in np.linspace(0,2,nm//3)])

    # plt.scatter(mesh_1[:, :, 0], mesh_1[:, :, 1], 0.1, 'b')
    # plt.scatter(mesh_2[:, :, 0], mesh_2[:, :, 1], 0.1, 'g')
    # plt.scatter(mesh_3[:, :, 0], mesh_3[:, :, 1], 0.1, 'r')
    # plt.show()

    mesh = np.concatenate((mesh_1, mesh_2, mesh_3), axis=0)
    return mesh


def motion_correction(imgs, conts):
    m0 = create_mesh(conts[:,0,:,:])
    m0_s = np.reshape(m0, (m0.shape[0] * m0.shape[1], 2))
    grid_x, grid_y = np.mgrid[0:imgs.shape[-2], 0:imgs.shape[-1]]
    moco_imgs = imgs.copy();
    for c in range(1, conts.shape[1]):
        m1 = create_mesh(conts[:,c,:,:])
        m1_s = np.reshape(m1, (m1.shape[0]*m1.shape[1], 2))
        # intrp_f = interp.interp2d(np.linspace(0,imgs.shape[-1],imgs.shape[-1]), np.linspace(0,imgs.shape[-2],imgs.shape[-2]),imgs[0,c, :, :], 'cubic')
        # vals = intrp_f(m1_s[:, 0], m1_s[:, 1])
        vals = sp.ndimage.map_coordinates(imgs[0, c, :, :], np.moveaxis(m1_s, 0, 1), order=3)
        i_img = imgs[0,c,:,:]
        m_img = interp.griddata(m0_s, vals, (grid_x, grid_y), 'cubic')
        m_img[np.isnan(m_img)] = i_img[np.isnan(m_img)]
        moco_imgs[:, c, :, :] = m_img
    return moco_imgs

def t1_map_recon(data, ti):
    T1 = np.zeros((data.shape[-2], data.shape[-1]))
    p0 = (1000, 1000)
    for ii in range(0, data.shape[-2]):
        for jj in range(0, data.shape[-1]):
            # tti[0] = np.abs(output[0, 0, ii, jj])
            plor =np.array([-1,1,1,1,1])
            t1, pcv = curve_fit(t1fit_2p, ti, data[:, ii, jj]*plor, p0, maxfev=5000)
            T1[ii, jj] = t1[1]
    T1[T1 > 2000] = 2000
    return T1

def get_moving_window(indx, num_sl, total_num_sl):
    if indx - num_sl // 2 < 1:
        return range(1, num_sl + 1)

    if indx + num_sl // 2 > total_num_sl:
        return range(total_num_sl - num_sl + 1, total_num_sl + 1)

    return range(indx - num_sl // 2, indx + num_sl // 2 + 1)

def create_T1_map(T1w, Ti):
    T1w = T1w.astype('float')
    Ti = Ti.astype('float')
    T1 = np.zeros((T1w.shape[-2], T1w.shape[-1]))
    p0 = (1000, 900)
    for ii in range(0, T1w.shape[-2]):
        for jj in range(0, T1w.shape[-1]):
            # tti[0] = np.abs(output[0, 0, ii, jj])
            t1, pcv = curve_fit(t1fit_2p, Ti, T1w[:, ii, jj], p0, maxfev=5000)
            T1[ii, jj] = t1[1]
    T1[T1 > 2600] = 2600
    T1[T1 < 0] = 0
    return T1

def combine_coils_RSOS_np(data):
    data_rsos = np.sqrt(np.sum(np.abs(data) ** 2, axis=0, keepdims=True))
    data2 = np.zeros((*data_rsos.shape,2))
    data2[:,:,:,:,0] = data_rsos
    return data2

def get_circle_mask(pnts, grid_size=[200,200]):
    nx, ny = grid_size[0], grid_size[1]

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T
    # c = [pnts[1,0], pnts[0,0]]
    c = pnts[0,:]

    r = pnts[1,:] - pnts[0,:]; r = (r[0]**2 + r[1]**2)**0.5
    r = r * 1.6
    n=200
    cont = np.asarray([(np.cos(x) * r, np.sin(x) * r) for x in np.linspace(-np.pi, np.pi, n)])
    cont = cont + c
    path = Path(cont.astype('float'))
    grid = path.contains_points(points)
    mask_out = grid.reshape((ny, nx))
    return mask_out

class DataGenerator(data.Dataset):
    'Generates data for Keras'

    def __init__(self, input_IDs, output_IDs, params=None, nums_slices=None, mode='training'):
        'Initialization'

        self.output_IDs = output_IDs
        self.input_IDs = input_IDs
        self.dim = params.img_size[:]
        self.dim.append(2)
        self.n_channels = params.n_channels
        self.n_spokes = params.n_spokes
        self.nums_slices = nums_slices
        self.complex_net = params.complex_net
        self.mode = mode
        self.params = params

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.input_IDs)

    def shuffel_cases(self):
        sh_idx = list(range(0, len(self.input_IDs), self.params.num_phases))
        np.random.shuffle(sh_idx)
        rnds = np.asarray([list(range(id, id+25)) for id in sh_idx])
        rnds = rnds.reshape((rnds.shape[0]*rnds.shape[1]))
        self.input_IDs = [self.input_IDs[pid] for pid in rnds]
        self.output_IDs = [self.output_IDs[pid] for pid in rnds]

    def __getitem__(self, index):
        'Generate one batch of data'
        X, y, T1, TI, inputT1, mask, LVmask, ROImask = self.generate_radial_t1_2p(self.input_IDs[index], self.output_IDs[index])
        return X, y, T1, TI, inputT1, mask, LVmask, ROImask,  self.input_IDs[index]

    def generate_radial_t1_2p(self, input_ID, output_ID):

        sl = int(input_ID[-5])
        # it = int(input_ID.split('/')[-1].split('_')[-1][:-4])
        f_name = input_ID.split('/')[-1]

        if params.rot_angle:
            v_url = input_ID[0:72] + '_{0}_{1}_GoldenAng_graded/'.format(self.n_spokes, self.params.gridding_method)
        else:
            v_url = input_ID[0:72] + '_{0}_{1}/'.format(self.n_spokes, self.params.gridding_method)

        Input_Data_Loaded = False
        if os.path.exists(v_url + f_name):
            try:
                data = loadmat(v_url + f_name)['data']
                input = data['input'][0][0]
                output = data['output'][0][0]
                save_data = False
                try:
                    LVPnts = data['LVPnts'][0][0]
                except:
                    print('LV points doesnot exist: ' + f_name)
                    LVPnts = np.asarray([[input.shape[-2]//2, input.shape[-1]//2], [input.shape[-2]//2-50, input.shape[-1]//2-50]])

                try:
                    LVConts = data['LVcontours'][0][0][0]
                except:
                    LVConts = [] #[np.zeros((61,2)), np.zeros((61,2))]

                try:
                    septalROI = data['sepatlROI'][0][0][0]
                except:
                    septalROI = [] #[np.zeros((61,2)), np.zeros((61,2))]


                try:
                    TI = data['TI'][0][0][0]
                except:
                    TI = data['Tinv'][0][0][0]

                try:
                    T1 = data['T1'][0][0]
                except:
                    save_data = True
                    print('T1 map fitting ' + f_name[:-4])
                    T1 = create_T1_map(np.abs(output), TI)

                try:
                    T1_5 = data['T1_5'][0][0]
                except:
                    save_data = True
                    print('T1_5 map fitting ' + f_name[:-4])
                    T1_5 = create_T1_map(np.abs(output[0:5, :, :]), TI[0:5])

                if save_data:
                    data = {'input': input, 'output': output, 'T1': T1, 'TI': TI, 'T1_5': T1_5, 'LVPnts': LVPnts}
                    saveArrayToMat(data, 'data', f_name[:-4], v_url)

                Input_Data_Loaded = True
            except:
                pass

        if False:#not Input_Data_Loaded:
            print('Preparing Dataset: {0}'.format(input_ID))

            ln_f = len(input_ID.split('/')[-1])
            ss = loadmat(input_ID[:-ln_f] + f_name.split('_')[0] + '.mat')
            img_s = ss['data'][0]['Img'][
                0]  # [('Img'), ('SliceLocation'), ('TriggerTime'), ('EpiContours'), ('EndoContours')])
            ti = ss['data'][0]['TriggerTime'][0];

            if (sl >= img_s.shape[-1]):
                sl = 0

            img_s = img_s[:, :, :, sl]

            epi_cont = ss['data'][0]['EpiContours'][0][:, :, :, sl]
            endo_cont = ss['data'][0]['EndoContours'][0][:, :, :, sl]

            mc = np.round(np.mean(epi_cont, 0)).astype(int)
            ## resize by interpolation
            # img = np.moveaxis(img,(0,1,2,3),(2,3,0,1))
            # img = torch.FloatTensor(img.astype(float))
            # img = torch.nn.functional.interpolate(img, (200, 200), mode='bilinear')
            # img = img.data.numpy()

            ## resize by cropping around the heart
            img = np.ndarray((200, 200, img_s.shape[-1]))
            mc[mc < 100] = 100
            mm = np.expand_dims(mc, 0).repeat(epi_cont.shape[0], 0)
            epi_cont = epi_cont - mm + 100
            endo_cont = endo_cont - mm + 100
            cont = np.moveaxis(np.stack((endo_cont, epi_cont), 0), 3, 1)

            for t1w in range(0, img_s.shape[-1]):
                img[:, :, t1w] = img_s[mc[1, t1w] - 100:mc[1, t1w] + 100, mc[0, t1w] - 100:mc[0, t1w] + 100, t1w]

            us_kspace = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
            us_kspace = us_kspace + 1j * us_kspace

            NUFFT = False
            if NUFFT:
                for t1w in range(0, img.shape[2]):
                    s_angle = ((np.pi / self.params.n_spokes) / (self.params.moving_window_size)) * t1w
                    us_img = get_radial_undersampled_image(img[:,:,t1w], self.params.n_spokes, rot_angle=s_angle, grid_method='BART')
                    us_kspace[:, :, t1w] = np.fft.fftshift(np.fft.fft2(us_img, axes=(0, 1)), axes=(0, 1))

            else:
                kspace = np.fft.fftshift(np.fft.fft2(img, axes=(0, 1)), axes=(0, 1))
                # kspace = kspace[:,:,:,sl]
                for t1w in range(0, img.shape[2]):
                    s_angle = ((np.pi / self.params.n_spokes) / (self.params.moving_window_size)) * t1w
                    traj = create_radial_trajectory(kspace.shape[0], self.params.n_spokes, s_angle)
                    traj_nn = np.round(
                        np.reshape(traj, (traj.shape[0] * traj.shape[1], traj.shape[2])))  # nearest neighbour
                    traj_nn = traj_nn + kspace.shape[0] // 2 - 1
                    zks = np.zeros((kspace.shape[0], kspace.shape[1]))
                    zks[traj_nn[:, 0].astype(int), traj_nn[:, 1].astype(int)] = 1
                    us_kspace[:, :, t1w] = kspace[:, :, t1w] * zks

            output = np.moveaxis(np.expand_dims(img, 0), (1, 2, 3), (2, 3, 1))
            input = np.moveaxis(np.expand_dims(us_kspace, 0), (1, 2, 3), (2, 3, 1))
            try:
                data = loadmat(v_url + f_name)['data']
                input = data['input'][0][0]
            except:
                pass
            ## Create T1 map
            TI = ti[:, sl].astype(float)
            T1 = create_T1_map(output, TI)

            ## Perform Motion Correction (MOCO)
            output = motion_correction(output, cont)
            T1_MOCO = create_T1_map(output, TI)

            ## save the data for future
            if not os.path.exists(v_url):
                os.makedirs(v_url)
            input = np.stack((input.real, input.imag), axis=4).astype(float)
            data = {'input': input, 'output': output, 'cont': cont, 'T1': T1, 'TI': TI, 'T1_MOCO':T1_MOCO}
            saveArrayToMat(data, 'data', f_name[:-4], v_url)


        # if not input.shape[-1] == 2:
        #     input = np.stack((input.real, input.imag), axis=4).astype(float)
        #
        if not output.shape[-1] == 2:
            output = np.stack((output.real, output.imag), axis=3).astype(float)


        output = np.expand_dims(output, 0)
        input = np.stack((input.real, input.imag), axis=3).astype(float)
        input = np.expand_dims(input, 0)
        output = output.astype(float)
        T1 = np.expand_dims(T1, 0).astype(float)
        TI = TI.astype(float)




        # if cont.shape[-1] == 1:
        #     cont = np.moveaxis(np.stack((cont['endo_cont'][0][0], cont['epi_cont'][0][0]), 0), 3, 1)

        ## ------- create weighting mask ---------
        # r = 0.5 * ((np.max(cont[1,0,:, 0], axis=0) - np.min(cont[1,0,:, 0], axis=0)) ** 2 + (
        #             np.max(cont[1,0,:, 1], axis=0) - np.min(cont[1,0,:, 1], axis=0)) ** 2) ** 0.5
        # mask = get_gaussian_mask(200, [r, r])
        # epi_cont = cont[1,0,:, :]
        # epi_cont = (epi_cont - np.mean(epi_cont, 0)) * (1.1) + np.mean(epi_cont, 0)
        # bin_msk = get_mask(epi_cont)
        # mask[bin_msk] = 1
        # mask = np.expand_dims(mask, 0)


        ## ------- create myocardial mask ------
        # epi_cont, endo_cont = get_ROI_conts(cont[1,0,:, :], cont[0,0,:, :], s=-0.1)
        # mask = get_mask(epi_cont, endo_cont)
        # mask = np.expand_dims(np.ones((T1.shape[-2], T1.shape[-1])) * mask, 0)

        # ------- create myocardial mask ------
        if LVConts == []:
            LVmask = np.zeros((1, T1.shape[-2], T1.shape[-1]))
        else:
            epi_cont, endo_cont = get_ROI_conts(LVConts[1], LVConts[0], s=0.05)
            mask = get_mask(epi_cont, endo_cont, grid_size=[T1.shape[-2], T1.shape[-1]])
            LVmask = np.expand_dims(np.ones((T1.shape[-2], T1.shape[-1])) * mask, 0)


        if septalROI == []:
            ROImask = np.zeros((1, T1.shape[-2], T1.shape[-1]))
        else:
            epi_cont = get_ROI_conts(septalROI[0], s=0.00)
            mask = get_mask(epi_cont, grid_size=[T1.shape[-2], T1.shape[-1]])
            ROImask = np.expand_dims(np.ones((T1.shape[-2], T1.shape[-1])) * mask, 0)


        ## ------- create epicardial (LV heart) mask -------
        # epi_cont, endo_cont = get_ROI_conts(cont[1, 0, :, :], cont[0, 0, :, :], s=-0.1)
        # mask = get_mask(epi_cont)
        # mask = np.expand_dims(np.ones((T1.shape[-2], T1.shape[-1])) * mask, 0)

        ## ------- create epicardial (LV heart) mask -------
        # epi_cont, endo_cont = get_ROI_conts(cont[1, 0, :, :], cont[0, 0, :, :], s=-0.1)
        # mask = get_mask(epi_cont)
        # mask = np.expand_dims(np.ones((T1.shape[-2], T1.shape[-1])) * mask, 0)

        if LVPnts.shape[0] < 2:
            LVPnts = np.concatenate((LVPnts, LVPnts+[15, 15]))

        mask = get_circle_mask(LVPnts, (input.shape[-3], input.shape[-2]))
        mask = np.expand_dims(np.ones((T1.shape[-2], T1.shape[-1])) * mask, 0)

        # ### blood mask or testing only
        # epi_cont, endo_cont = get_ROI_conts(LVConts[1], LVConts[0], s=0.00001)
        # endo_cont = get_ROI_conts(endo_cont, s=0.3)
        # mask = get_mask(endo_cont, grid_size=[T1.shape[-2], T1.shape[-1]])
        # mask = np.expand_dims(np.ones((T1.shape[-2], T1.shape[-1])) * mask, 0)

        # its = np.array(range(it - self.params.moving_window_size // 2, it + self.params.moving_window_size // 2 + 1))
        # its[its[:] > 11] = its[its[:] > 11] - 11
        # its[its[:]<1] = 11 - abs(its[its[:]<1])
        # its = list(its - 1)
        #
        # input = input[:,its,:,:,:]
        # output = output[:,it-1,:,:]

        return input, output, T1, TI, T1_5, mask, LVmask, ROImask

    def generate_radial_t1(self, input_ID, output_ID):

        sl = int(input_ID.split('/')[-1].split('_')[1][:-4])
        # it = int(input_ID.split('/')[-1].split('_')[-1][:-4])
        f_name = input_ID.split('/')[-1].split('_')[0] + '_' + str(sl) + '.mat'

        if params.rot_angle:
            v_url = input_ID[0:72] + '_{0}_{1}_RotAng/'.format(self.n_spokes, self.params.gridding_method)
        else:
            v_url = input_ID[0:72] + '_{0}_{1}/'.format(self.n_spokes, self.params.gridding_method)

        Input_Data_Loaded = False
        if os.path.exists(v_url + f_name):
            try:
                data = loadmat(v_url + f_name)['data']
                input = data['input'][0][0]
                output = data['output'][0][0]
                cont = data['cont'][0][0]
                T1 = data['T1'][0][0]
                TI = data['TI'][0][0][0]
                Input_Data_Loaded = True
            except:
                pass

        if not Input_Data_Loaded:
            print('Preparing Dataset: {0}'.format(input_ID))
            ln_f = len(input_ID.split('/')[-1])
            ss = loadmat(input_ID[:-ln_f] + f_name.split('_')[0] + '.mat')
            img_s = ss['data'][0]['Img'][0] #[('Img'), ('SliceLocation'), ('TriggerTime'), ('EpiContours'), ('EndoContours')])
            ti = ss['data'][0]['TriggerTime'][0];

            if(sl >= img_s.shape[-1]):
                sl = 0

            img_s = img_s[:,:,:,sl]

            epi_cont = ss['data'][0]['EpiContours'][0][:,:,:,sl]
            endo_cont = ss['data'][0]['EndoContours'][0][:, :, :, sl]

            mc = np.round(np.mean(epi_cont, 0)).astype(int)
            ## resize by interpolation
            # img = np.moveaxis(img,(0,1,2,3),(2,3,0,1))
            # img = torch.FloatTensor(img.astype(float))
            # img = torch.nn.functional.interpolate(img, (200, 200), mode='bilinear')
            # img = img.data.numpy()

            ## resize by cropping around the heart
            img = np.ndarray((200,200,img_s.shape[-1]))
            mc[mc < 100] = 100
            mm = np.expand_dims(mc, 0).repeat(epi_cont.shape[0], 0)
            epi_cont = epi_cont - mm + 100
            endo_cont = endo_cont - mm + 100
            cont = np.moveaxis(np.stack((endo_cont, epi_cont), 0), 3, 1)

            for t1w in range(0,img_s.shape[-1]):
                img[:,:,t1w] = img_s[mc[1,t1w]-100:mc[1,t1w]+100, mc[0,t1w]-100:mc[0,t1w]+100, t1w]

            kspace = np.fft.fftshift(np.fft.fft2(img, axes=(0,1)), axes=(0,1))
            # kspace = kspace[:,:,:,sl]

            us_kspace = np.zeros((kspace.shape[0], kspace.shape[1], kspace.shape[2]))
            us_kspace = us_kspace + 1j * us_kspace

            for t1w in range(0, img.shape[2]):
                s_angle = ((np.pi / self.params.n_spokes) / (self.params.moving_window_size)) * t1w
                traj = create_radial_trajectory(kspace.shape[0], self.params.n_spokes, s_angle)
                traj_nn = np.round(
                    np.reshape(traj, (traj.shape[0] * traj.shape[1], traj.shape[2])))  # nearest neighbour
                traj_nn = traj_nn + kspace.shape[0]//2 -1
                zks = np.zeros((kspace.shape[0], kspace.shape[1]))
                zks[traj_nn[:,0].astype(int), traj_nn[:,1].astype(int)] = 1
                us_kspace[:,:,t1w] = kspace[:, :, t1w] * zks

            output = np.moveaxis(np.expand_dims(img, 0), (1,2,3), (2,3,1))
            input = np.moveaxis(np.expand_dims(us_kspace, 0), (1,2,3), (2,3,1))

            output[0, 0:2, :, :] = output[0, 0:2, :, :] * -1
            tti = np.zeros(12)
            tti[1:] = ti[:, sl]

            T1 = np.zeros((200, 200))
            for ii in range(0, 200):
                for jj in range(0, 200):
                    tti[0] = np.abs(output[0,0,ii,jj])
                    t1, pcv = curve_fit(t1fit, tti , output[0,:,ii,jj] , p0=100, maxfev=5000)
                    T1[ii,jj] = t1
            T1[T1>2000] = 2000
            TI = ti[:, sl].astype(float)
            ## save the data for future
            if not os.path.exists(v_url):
                os.makedirs(v_url)
            input = np.stack((input.real, input.imag), axis=4).astype(float)
            data = {'input': input, 'output': output, 'cont': cont, 'T1': T1, 'TI':TI}
            saveArrayToMat(data, 'data', f_name[:-4], v_url)


        output = output.astype(float)
        T1 = np.expand_dims(T1, 0)

        if cont.shape[-1] == 1:
            cont = np.moveaxis(np.stack((cont['endo_cont'][0][0],cont['epi_cont'][0][0]), 0), 3, 1)

        # its = np.array(range(it - self.params.moving_window_size // 2, it + self.params.moving_window_size // 2 + 1))
        # its[its[:] > 11] = its[its[:] > 11] - 11
        # its[its[:]<1] = 11 - abs(its[its[:]<1])
        # its = list(its - 1)
        #
        # input = input[:,its,:,:,:]
        # output = output[:,it-1,:,:]

        return input, output, T1, TI, cont

    def generate_radial_cine_mvw(self, input_ID, output_ID):

        f_name0 = input_ID.split('/')[-1][:-4] + '.mat'

        f_args = parse_dat_filename(input_ID.split('/')[-1])
        mvw = np.asarray(range(f_args['phs'] - params.moving_window_size//2 -1, f_args['phs'] + params.moving_window_size//2 ))
        mvw[mvw < 0] = mvw[mvw < 0] + params.num_phases
        mvw[mvw > params.num_phases-1] = mvw[mvw > params.num_phases-1] - params.num_phases
        mvw = mvw + 1

        input_mv = np.zeros((params.moving_window_size, params.n_channels, params.img_size[0], params.img_size[1], 2))
        mv_idx = 0

        for mv in mvw.tolist():
            f_name = get_dat_filename(slc=f_args['slc'], phs=mv,lins=f_args['lins'],cols=f_args['cols'], cha=f_args['cha'])[:-4] + '.mat'

            if params.rot_angle:
                v_url = input_ID[0:69] + '_{0}_{1}_RotAng_{2}'.format(self.n_spokes, self.params.gridding_method,
                                                                      self.params.gradient_delays_method) + input_ID[
                                                                                                            69:-len(f_name0)]
            else:
                v_url = input_ID[0:69] + '_{0}_{1}_{2}'.format(self.n_spokes, self.params.gridding_method,
                                                               self.params.gradient_delays_method) + input_ID[
                                                                                                     69:-len(f_name0)]

            Input_Data_Loaded = False
            if os.path.exists(v_url + f_name):
                try:
                    data = loadmat(v_url + f_name)['data']
                    input = data['input'][0][0]
                    trajectory = data['trajectory'][0][0]
                    ks_val = data['ks_val'][0][0]
                    SNR_rank = data['SNR_rank'][0][0][0]
                    gauss_params = data['gauss_param'][0][0][0]
                    if input.shape[0] == self.n_channels:
                        Input_Data_Loaded = True
                except:
                    pass

            if not Input_Data_Loaded:
                print('Preparing Dataset: {0}'.format(input_ID))
                print('File didnot exist : {0}'.format(v_url + f_name))
                kspace_lines = read_dat_file(input_ID)
                ''' interpolate the n_cols of kspace to have same size as reference images'''
                # dt = interp.interp1d(np.linspace(0, kspace_lines.shape[0], kspace_lines.shape[0]), kspace_lines, axis=0)
                # kspace_lines = dt(np.linspace(0, kspace_lines.shape[0], kspace_lines.shape[0]//2)) #

                ## zero-pad the kspace lines is equivalent to interpolation in image domain
                dim_diff = self.dim[0] - kspace_lines.shape[0] // 2
                if dim_diff % 2:
                    raise Exception('dimension difference between raw kspace lines and input can not be odd!')
                else:
                    zp = int(dim_diff / 2)
                    if dim_diff > 0:
                        kspace_lines = np.pad(kspace_lines, ((zp, zp), (0, 0), (0, 0), (0, 0)), 'constant')
                    elif dim_diff < 0:
                        kspace_lines = kspace_lines[zp:-zp, ]

                phase = 0
                if params.rot_angle:
                    phase = parse_dat_filename(f_name)['phs'] - 1

                ## unify the number of coils per case
                kspace_lines, SNR_rank = stratify_kspace_channels(kspace_lines, self.n_channels)
                input, trajectory, ks_val = undersample_radial_kspace(kspace_lines, self.n_spokes,
                                                                      trajectory=None,
                                                                      gridding_method=self.params.gridding_method,
                                                                      gradient_delays_method=self.params.gradient_delays_method,
                                                                      k_neighbors=self.params.k_neighbors,
                                                                      per_phase_rot=2 * phase)

                ## calculate gaussian weighting function paramters
                # gauss_params = kspaceImg_gauss_fit(input)
                # gauss_params[2] *= 3
                ## from kspacelines --> the problem is that the kspaceimages have density correction weighting so it is better to calculate from the image not from the original kspace lines
                gauss_params = kspacelines_gauss_fit(kspace_lines)
                gauss_params[1] /= 2
                gauss_params[2] /= 2  # kspace_lines are oversampled
                gauss_params[2] *= 3  # increase the STD by factor of 3

                if self.params.gridding_method == self.params.g_methods[0]:  # 'neighbours_matrix'
                    input = np.moveaxis(input, [2, 3], [0, 1])
                    # reshape as follow: in[N_neighbors_N_ch,Height,Width,cmplx] --> out[N_neighbors*N_ch,Height,Width,cmplx] ordered by N_neighbors first
                    input = np.reshape(input,
                                       [input.shape[0] * input.shape[1], input.shape[2], input.shape[3], input.shape[4]],
                                       'F')
                else:
                    input = np.moveaxis(input, [2], [0])

                ## save the data for future
                if not os.path.exists(v_url):
                    os.makedirs(v_url)
                data = {'input': input, 'trajectory': trajectory, 'ks_val': ks_val,
                        'SNR_rank': SNR_rank, 'gauss_param': gauss_params}
                saveArrayToMat(data, 'data', f_name[:-4], v_url)

            if len(trajectory) == 0:
                trajectory = np.zeros((1, 1))
            if len(ks_val) == 0:
                ks_val = np.zeros((1, 1))

            input_mv[mv_idx, ] = input
            mv_idx += 1

        input_mv = np.moveaxis(input_mv, [0], [1])
        ##################################################################
        ## load refernece fullysampled images

        Coil_combined = False

        if Coil_combined:
            '''read coil-combined 1-channel complex-valued data from .mat files'''
            output = loadmat(output_ID)['data']

        else:
            '''read all 16 coils from DAT file'''
            output = read_dat_file(output_ID)

            ch_diff = output.shape[2] - self.n_channels

            if ch_diff == 0:
                output = output[:, :, SNR_rank, :]
            elif ch_diff > 0:
                out = output[:, :, SNR_rank[:-ch_diff], :]
                out[:, :, -1, :] = np.mean(output[:, :, SNR_rank[-ch_diff - 1:], :], axis=2, keepdims=False)
                output = out
            elif ch_diff < 0:
                output = output[:, :, SNR_rank, :]
                output = np.append(output, output[:, :, -abs(ch_diff):, :], axis=2)
            output = output[::-1, :, :, :].copy()
            # # combine coils using square-root of sum-of-squares
            # output = np.expand_dims(combine_channels_RSS(output), axis=0)

        orig_size = output.shape
        if self.dim[0] != output.shape[0] or self.dim[1] != output.shape[1]:
            dt2 = interp.interp2d(np.linspace(0, output.shape[0], output.shape[0]),
                                  np.linspace(0, output.shape[1], output.shape[1]), output)

            output = dt2(np.linspace(0, output.shape[0], self.dim[0]),
                         np.linspace(0, output.shape[1], self.dim[1]))

        output = np.moveaxis(output, [2], [0])

        return input_mv, output, trajectory, ks_val, orig_size, self.n_spokes / 198, gauss_params


    def generate_radial_cine(self, input_ID, output_ID):

        f_name = input_ID.split('/')[-1][:-4] + '.mat'

        if params.rot_angle:
            v_url = input_ID[0:69] + '_{0}_{1}_RotAng_{2}'.format(self.n_spokes, self.params.gridding_method,
                                                       self.params.gradient_delays_method) + input_ID[69:-len(f_name)]
        else:
            v_url = input_ID[0:69] + '_{0}_{1}_{2}'.format(self.n_spokes, self.params.gridding_method,
                                                       self.params.gradient_delays_method) + input_ID[69:-len(f_name)]

        Input_Data_Loaded = False
        if os.path.exists(v_url + f_name):
            try:
                data = loadmat(v_url + f_name)['data']
                input = data['input'][0][0]
                trajectory = data['trajectory'][0][0]
                ks_val = data['ks_val'][0][0]
                SNR_rank = data['SNR_rank'][0][0][0]
                gauss_params = data['gauss_param'][0][0][0]
                if input.shape[0] == self.n_channels:
                    Input_Data_Loaded = True
            except:
                pass

        if not Input_Data_Loaded:
            print('Preparing Dataset: {0}'.format(input_ID))
            kspace_lines = read_dat_file(input_ID)
            ''' interpolate the n_cols of kspace to have same size as reference images'''
            # dt = interp.interp1d(np.linspace(0, kspace_lines.shape[0], kspace_lines.shape[0]), kspace_lines, axis=0)
            # kspace_lines = dt(np.linspace(0, kspace_lines.shape[0], kspace_lines.shape[0]//2)) #

            ## zero-pad the kspace lines is equivalent to interpolation in image domain
            dim_diff = self.dim[0] - kspace_lines.shape[0] // 2
            if dim_diff % 2:
                raise Exception('dimension difference between raw kspace lines and input can not be odd!')
            else:
                zp = int(dim_diff / 2)
                if dim_diff > 0:
                    kspace_lines = np.pad(kspace_lines, ((zp, zp), (0, 0), (0, 0), (0, 0)), 'constant')
                elif dim_diff < 0:
                    kspace_lines = kspace_lines[zp:-zp, ]

            phase = 0
            if params.rot_angle:
                phase = parse_dat_filename(f_name)['phs'] - 1

            ## unify the number of coils per case
            kspace_lines, SNR_rank = stratify_kspace_channels(kspace_lines, self.n_channels)
            input, trajectory, ks_val = undersample_radial_kspace(kspace_lines, self.n_spokes,
                                                                  trajectory=None,
                                                                  gridding_method=self.params.gridding_method,
                                                                  gradient_delays_method=self.params.gradient_delays_method,
                                                                  k_neighbors=self.params.k_neighbors,
                                                                  per_phase_rot=phase)

            ## calculate gaussian weighting function paramters
            # gauss_params = kspaceImg_gauss_fit(input)
            # gauss_params[2] *= 3
            ## from kspacelines --> the problem is that the kspaceimages have density correction weighting so it is better to calculate from the image not from the original kspace lines
            gauss_params = kspacelines_gauss_fit(kspace_lines)
            gauss_params[1] /= 2
            gauss_params[2] /= 2  # kspace_lines are oversampled
            gauss_params[2] *= 3  # increase the STD by factor of 3

            if self.params.gridding_method == self.params.g_methods[0]: #'neighbours_matrix'
                input = np.moveaxis(input, [2, 3], [0, 1])
                #reshape as follow: in[N_neighbors_N_ch,Height,Width,cmplx] --> out[N_neighbors*N_ch,Height,Width,cmplx] ordered by N_neighbors first
                input = np.reshape(input, [input.shape[0]*input.shape[1], input.shape[2], input.shape[3], input.shape[4]], 'F')
            else:
                input = np.moveaxis(input, [2], [0])

            ## save the data for future
            if not os.path.exists(v_url):
                os.makedirs(v_url)
            data = {'input': input, 'trajectory': trajectory, 'ks_val': ks_val,
                    'SNR_rank': SNR_rank, 'gauss_param': gauss_params}
            saveArrayToMat(data, 'data', f_name[:-4], v_url)

        if len(trajectory) == 0:
            trajectory = np.zeros((1,1))
        if len(ks_val) == 0:
            ks_val = np.zeros((1,1))

        ##################################################################
        ## load refernece fullysampled images

        Coil_combined = False

        if Coil_combined:
            '''read coil-combined 1-channel complex-valued data from .mat files'''
            output = loadmat(output_ID)['data']

        else:
            '''read all 16 coils from DAT file'''
            output = read_dat_file(output_ID)

            ch_diff = output.shape[2] - self.n_channels

            if ch_diff == 0:
                output = output[:, :, SNR_rank, :]
            elif ch_diff > 0:
                out = output[:, :, SNR_rank[:-ch_diff], :]
                out[:, :, -1, :] = np.mean(output[:, :, SNR_rank[-ch_diff - 1:], :], axis=2, keepdims=False)
                output = out
            elif ch_diff < 0:
                output = output[:, :, SNR_rank, :]
                output = np.append(output, output[:, :, -abs(ch_diff):, :], axis=2)
            output = output[::-1, :, :, :].copy()
            # # combine coils using square-root of sum-of-squares
            # output = np.expand_dims(combine_channels_RSS(output), axis=0)

        orig_size = output.shape
        if self.dim[0] != output.shape[0] or self.dim[1] != output.shape[1]:
            dt2 = interp.interp2d(np.linspace(0, output.shape[0], output.shape[0]),
                                  np.linspace(0, output.shape[1], output.shape[1]), output)

            output = dt2(np.linspace(0, output.shape[0], self.dim[0]),
                         np.linspace(0, output.shape[1], self.dim[1]))

        output = np.moveaxis(output, [2], [0])

        return input, output, trajectory, ks_val, orig_size, self.n_spokes / 198, gauss_params

    def __data_generation(self, index, input_IDs_temp, output_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization

        if self.complex_net:
            if len(self.dim) == 3:
                return self.generate_data2D(index, input_IDs_temp, output_IDs_temp)
            elif len(self.dim) > 3 and self.mode == 'training':
                if params.num_slices_3D > 50:  # whole volume Feeding
                    return self.generate_data3D(index, input_IDs_temp, output_IDs_temp)
                else:  # moving window feeding
                    return self.generate_data3D_moving_window(index, input_IDs_temp, output_IDs_temp)
            elif len(self.dim) > 3 and self.mode == 'testing':
                return self.generate_data3D_testing(index, input_IDs_temp, output_IDs_temp)
        else:
            if len(self.dim) == 2:
                return self.generate_data2D(index, input_IDs_temp, output_IDs_temp)
            else:
                return self.generate_data3D(index, input_IDs_temp, output_IDs_temp)

    def generate_data2D(self, index, input_IDs_temp, output_IDs_temp):
        # Initialization
        X = np.zeros((self.n_channels, *self.dim))
        y = np.zeros((self.n_channels, *self.dim))

        # Generate data
        img = loadmat(input_IDs_temp)['Input_realAndImag']
        orig_size = [img.shape[0], img.shape[1]]
        #         for i, ID in enumerate(input_IDs_temp):
        X[0,] = resizeImage(img, [self.dim[0], self.dim[1]])

        #         for i, ID in enumerate(output_IDs_temp):
        y[0, :, :, 0] = resizeImage(loadmat(output_IDs_temp)['Data'], [self.dim[0], self.dim[1]])
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        return X, y, orig_size

    def generate_data3D(self, index, patients, out_patients):
        '''
        Read 3D volumes or stack of 2D slices
        '''
        Stack_2D = True

        if Stack_2D:
            slices = getPatientSlicesURLs(patients)
            X = np.zeros((1, self.dim[0], self.dim[1], self.dim[2], 2))
            y = np.zeros((1, self.dim[0], self.dim[1], self.dim[2], 2))

            z1 = (len(slices[0]) - self.dim[2]) // 2
            z2 = len(slices[0]) - self.dim[2] - z1

            sz = 0
            if z1 > 0:
                rng = range(z1, len(slices[0]) - z2)
                sz = -z1
            elif z1 < 0:
                rng = range(0, len(slices[0]))
                sz = z1
            elif z1 == 0:
                rng = range(0, self.dim[2])

            for sl in rng:
                img = loadmat(slices[0][sl])['Input_realAndImag']
                orig_size = [img.shape[0], img.shape[1]]
                try:
                    X[0, :, :, sl + sz, :] = resizeImage(img, [self.dim[0], self.dim[1]])

                    y[0, :, :, sl + sz, 0] = resizeImage(loadmat(slices[1][sl])['Data'], [self.dim[0], self.dim[1]])
                except:
                    stop = 1
            X = np.nan_to_num(X)
            y = np.nan_to_num(y)
            return X, y, orig_size
        else:
            pass

    def generate_data3D_moving_window(self, index, input_IDs_temp, output_IDs_temp):
        '''
            Moving window
        '''
        # Initialization
        X = np.zeros((self.n_channels, *self.dim))
        y = np.zeros((self.n_channels, *self.dim))

        sl_indx = int(input_IDs_temp.split('/')[-1][8:-4])

        rng = get_moving_window(sl_indx, self.dim[2], self.nums_slices[index])

        i = 0
        # Generate data
        #         print(input_IDs_temp)
        #         print('sl_indx->', sl_indx, '  nslices->',self.dim[2], 'max_slices', self.nums_slices[index] )
        for sl in rng:
            #             print(sl)
            in_sl_url = '/'.join(input_IDs_temp.split('/')[0:-1]) + '/Input_sl' + str(sl) + '.mat'
            out_sl_url = '/'.join(output_IDs_temp.split('/')[0:-1]) + '/Input_sl' + str(sl) + '.mat'
            try:
                img = loadmat(in_sl_url)['Input_realAndImag']
            except:
                print('Data Loading Error ..... !')
            orig_size = [img.shape[0], img.shape[1]]
            X[0, :, :, i, :] = resizeImage(img, [self.dim[0], self.dim[1]])
            y[0, :, :, i, 0] = resizeImage(loadmat(out_sl_url)['Data'], [self.dim[0], self.dim[1]])
            i += 1
        #         print('---------------------------------')
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        return X, y, orig_size

    def generate_data3D_testing(self, index, patients, out_patients):
        def ceildiv(a, b):
            return -(-a // b)

        slices = getPatientSlicesURLs(patients)
        X = np.zeros((1, self.dim[0], self.dim[1], len(slices[0]), 2))
        y = np.zeros((1, self.dim[0], self.dim[1], len(slices[0]), 2))

        for sl in range(0, len(slices[0])):
            img = loadmat(slices[0][sl])['Input_realAndImag']
            orig_size = [img.shape[0], img.shape[1]]
            X[0, :, :, sl, :] = resizeImage(img, [self.dim[0], self.dim[1]])

            y[0, :, :, sl, 0] = resizeImage(loadmat(slices[1][sl])['Data'], [self.dim[0], self.dim[1]])

        X = np.nan_to_num(X)
        y = np.nan_to_num(y)

        #         n_batchs = ceildiv(len(slices[0]), self.dim[2])
        #
        #         # Initialization
        #         X = np.zeros((n_batchs,1, *self.dim))
        #         y = np.zeros((n_batchs,1, *self.dim))
        #
        #         ds_sl = 0
        #         for bt in range(0, n_batchs):
        #             for sl in range(0, self.dim[2]):
        #                 if ds_sl >= len(slices[0]):
        #                     break
        # #                 print('ds_sl:',ds_sl, 'sl:',sl, 'bt:', bt)
        #                 img = loadmat(slices[0][ds_sl])['Input_realAndImag']
        #                 orig_size = [img.shape[0], img.shape[1]]
        #                 X[bt,0,:,:,sl,:] = resizeImage(img,[self.dim[0],self.dim[1]])
        #
        #                 y[bt,0,:,:,sl,0] = resizeImage(loadmat(slices[1][ds_sl])['Data'],[self.dim[0],self.dim[1]])
        #                 X = np.nan_to_num(X)
        #                 y = np.nan_to_num(y)
        #                 ds_sl += 1
        return X, y, orig_size

# class DataGenerator(data.Dataset):
#     'Generates data for Keras'
#     def __init__(self, input_IDs, output_IDs, undersampling_rates=None, dim=(256,256,2), n_channels=1,complex_net=True ,nums_slices=None):
#         'Initialization'
#         self.dim = dim
#         self.output_IDs = output_IDs
#         self.input_IDs = input_IDs
#         self.n_channels = n_channels
#         self.undersampling_rates = undersampling_rates
#         self.nums_slices = nums_slices
#         self.complex_net = complex_net
#
#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return len(self.input_IDs)
#
#     def __getitem__(self, index):
#         'Generate one batch of data'
#         if len(self.dim)==2 or (len(self.dim) ==3 and self.complex_net):
#             return self.getItem2D(index)
#         else:
#             return self.getItem3D(index)
#
#     def getItem2D(self, index):
#         # Generate data
#         X, y, orig_size = self.__data_generation(self.input_IDs[index], self.output_IDs[index])
#         if self.undersampling_rates is not None:
#             usr = self.undersampling_rates[index]
#         else:
#             usr = None
#
#         return X, y, self.input_IDs[index], orig_size, usr
#
#     def getItem3D(self, index):
#         # Generate data
#         X, y, orig_size = self.__data_generation(self.input_IDs[index], self.output_IDs[index])
#         if self.undersampling_rates is not None:
#             usr = self.undersampling_rates[index]
#         else:
#             usr = None
#
#         return X, y, self.input_IDs[index], orig_size, usr
#
#
#     def __data_generation(self, input_IDs_temp, output_IDs_temp):
#         'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#         # Initialization
#         X = np.zeros((self.n_channels, *self.dim))
#         y = np.zeros((self.n_channels, *self.dim))
#
#         # Generate data
#         img = loadmat(input_IDs_temp)['Input_realAndImag']
#         orig_size = [img.shape[0], img.shape[1]]
# #         for i, ID in enumerate(input_IDs_temp):
#         X[0,] = resizeImage(img,[self.dim[0],self.dim[1]])
#
# #         for i, ID in enumerate(output_IDs_temp):
#         y[0,:,:,0] = resizeImage(loadmat(output_IDs_temp)['Data'],[self.dim[0],self.dim[1]])
#         X = np.nan_to_num(X)
#         y = np.nan_to_num(y)
#         return X, y, orig_size










































#
#
#
#
#
#
#
#
#
#
# import torch
# from torch.utils import data
# from parameters import Parameters
# from scipy.io import loadmat, savemat
# import numpy as np
# import os
# from saveNet import *
# from utils.gridkspace import *
# from utils.gaussian_fit import gauss_fit, kspacelines_gauss_fit, kspaceImg_gauss_fit
# from math import exp
# from scipy.optimize import curve_fit
#
#
# # params = Parameters()
#
# def resizeImage(img, newSize, Interpolation=False):
#     if img.ndim == 2:
#         img = np.expand_dims(img, 2)
#
#     if Interpolation:
#         return imresize(img, tuple(newSize), interp='bilinear')
#     else:
#
#         x1 = (img.shape[0] - newSize[0]) // 2
#         x2 = img.shape[0] - newSize[0] - x1
#
#         y1 = (img.shape[1] - newSize[1]) // 2
#         y2 = img.shape[1] - newSize[1] - y1
#
#         if img.ndim == 3:
#             if x1 > 0:
#                 img = img[x1:-x2, :, :]
#             elif x1 < 0:
#                 img = np.pad(img, ((-x1, -x2), (0, 0), (0, 0)), 'constant')  # ((top, bottom), (left, right))
#
#             if y1 > 0:
#                 img = img[:, y1:-y2, :]
#             elif y1 < 0:
#                 img = np.pad(img, ((0, 0), (-y1, -y2), (0, 0)), 'constant')  # ((top, bottom), (left, right))
#
#         elif img.ndim == 4:
#             if x1 > 0:
#                 img = img[x1:-x2, :, :, :]
#             elif x1 < 0:
#                 img = np.pad(img, ((-x1, -x2), (0, 0), (0, 0), (0, 0)), 'constant')  # ((top, bottom), (left, right))
#
#             if y1 > 0:
#                 img = img[:, y1:-y2, :, :]
#             elif y1 < 0:
#                 img = np.pad(img, ((0, 0), (-y1, -y2), (0, 0), (0, 0)), 'constant')  # ((top, bottom), (left, right))
#         return img.squeeze()
#
#
# def resize3DVolume(data, newSize, Interpolation=False):
#     ndim = data.ndim
#     if ndim < 3:
#         return None
#     elif ndim == 3:
#         data = np.expand_dims(data, 3)
#
#     if Interpolation:
#         return imresize(data, tuple(newSize), interp='bilinear')
#
#     elif ndim == 4:
#         x1 = (data.shape[0] - newSize[0]) // 2
#         x2 = data.shape[0] - newSize[0] - x1
#
#         y1 = (data.shape[1] - newSize[1]) // 2
#         y2 = data.shape[1] - newSize[1] - y1
#
#         z1 = (data.shape[2] - newSize[2]) // 2
#         z2 = data.shape[2] - newSize[2] - z1
#
#         if x1 > 0:
#             data = data[x1:-x2, :, :, :]
#         elif x1 < 0:
#             data = np.pad(data, ((-x1, -x2), (0, 0), (0, 0), (0, 0)), 'constant')  # ((top, bottom), (left, right))
#
#         if y1 > 0:
#             data = data[:, y1:-y2, :, :]
#         elif y1 < 0:
#             data = np.pad(data, ((0, 0), (-y1, -y2), (0, 0), (0, 0)), 'constant')  # ((top, bottom), (left, right))
#
#         if z1 > 0:
#             data = data[:, :, z1:-z2, :]
#         elif z1 < 0:
#             data = np.pad(data, ((0, 0), (0, 0), (-z1, -z2), (0, 0)), 'constant')  # ((top, bottom), (left, right))
#
#         return data.squeeze()
#
#
# def get_gaussian_mask(window_size, sigma, center=None, circ_window_radius=None):
#
#     if center is None:
#         center = [window_size // 2, window_size // 2]
#
#     gauss_2d = np.zeros((window_size, window_size, len(sigma)))
#     [Xm, Ym] = np.meshgrid(np.linspace(1, window_size, window_size), np.linspace(1, window_size, window_size))
#     for g in range(0, len(sigma)):
#         gauss_x = np.asarray([exp(-(x-center[0])**2/float(2*sigma[g]**2)) for x in range(window_size)])
#         gauss_y = np.asarray([exp(-(x-center[1])**2/float(2*sigma[g]**2)) for x in range(window_size)])
#         gauss2= np.dot(np.expand_dims(gauss_x, 1), np.expand_dims(gauss_y, 1).T)
#         if circ_window_radius is not None:
#             gauss2[((Xm-window_size//2)**2+(Ym-window_size//2)**2)**0.5 <= circ_window_radius[0]] = 1
#         gauss_2d[:, :, g] = gauss2
#     return gauss_2d
#
#
# def getPatientSlicesURLs(patient_url):
#     islices = list()
#     oslices = list()
#     for fs in os.listdir(patient_url + '/InputData/Input_realAndImag/'):
#         islices.append(patient_url + '/InputData/Input_realAndImag/' + fs)
#
#     for fs in os.listdir(patient_url + '/CSRecon/CSRecon_Data_small/'):
#         oslices.append(patient_url + '/CSRecon/CSRecon_Data_small/' + fs)
#     islices = sorted(islices, key=lambda x: int((x.rsplit(sep='/')[-1])[8:-4]))
#     oslices = sorted(oslices, key=lambda x: int((x.rsplit(sep='/')[-1])[8:-4]))
#
#     return (islices, oslices)
#
#
# def getDatasetGenerators(params):
#     params.num_slices_per_patient = []
#     params.input_slices = []
#     params.groundTruth_slices = []
#     params.us_rates = []
#     params.patients = []
#     params.training_patients_index = []
#
#
#
#     # for dir in params.dir:
#     #     datasets_dirs = sorted(os.listdir(dir + 'T1Dataset/'), key=lambda x: int(x[:-4]))
#     #     for i, dst in enumerate(datasets_dirs):
#     #         params.patients.append(dst)
#     #         kspaces = sort_files(os.listdir(dir + 'kspace/' + dst))
#     #         params.num_slices_per_patient.append(len(kspaces))
#     #
#     #         for j, ksp in enumerate(kspaces):
#     #             params.input_slices.append(dir + 'kspace/' + dst + '/' + ksp)
#     #
#     #         '''read all 16 coils from DAT file'''
#     #         images = sort_files(os.listdir(dir + 'image/' + dst))
#     #         for j, img in enumerate(images):
#     #             params.groundTruth_slices.append(dir + 'image/' + dst + '/' + img)
#     #
#     #         '''read coil-combined 1-channel complex-valued data from .mat files'''
#     # #             images = sort_files(os.listdir(dir + 'ref/' + dst))
#     # #             for j, img in enumerate(images):
#     # #                 params.groundTruth_slices.append(dir + 'ref/' + dst + '/' + img)
#
#     print('-- Number of Datasets: ' + str(len(params.patients)))
#
#     params.groundTruth_slices = list()
#     for i in range(1, 211):
#         params.patients.append(params.dir[0] + 'T1Dataset/' + str(i)+'.mat')
#         for sl in range(1, 6):
#             # for it in range(1, 12):
#             params.groundTruth_slices.append(params.dir[0] + 'T1Dataset/' + str(i) + '_' + str(sl) + '_' + str(it) + '.mat')
#
#     params.input_slices = params.groundTruth_slices
#     training_ptns = int(params.training_percent * len(params.patients))
#
#     training_end_indx = training_ptns * 5 * 11
#
#     params.training_patients_index = range(0, training_ptns + 1)
#
#     dim = params.img_size[:]
#     dim.append(2)
#
#     tr_samples = 1
#
#     training_DS = DataGenerator(input_IDs=params.input_slices[:training_end_indx:tr_samples],
#                                 output_IDs=params.groundTruth_slices[:training_end_indx:tr_samples],
#                                 params=params
#                                 )
#
#     validation_DS = DataGenerator(input_IDs=params.input_slices[training_end_indx:],
#                                   output_IDs=params.groundTruth_slices[training_end_indx:],
#                                   params=params
#                                   )
#
#
#
#     training_DL = data.DataLoader(training_DS, batch_size=params.batch_size, shuffle=True,
#                                   num_workers=params.data_loders_num_workers)
#     #     validation_DL = data.DataLoader(validation_DS, batch_size=params.batch_size, shuffle=False, num_workers=params.data_loders_num_workers)
#     validation_DL = data.DataLoader(validation_DS, batch_size=params.batch_size, shuffle=False,
#                                     num_workers=params.data_loders_num_workers)
#
#     return training_DL, validation_DL, params
#
#
# def get_moving_window(indx, num_sl, total_num_sl):
#     if indx - num_sl // 2 < 1:
#         return range(1, num_sl + 1)
#
#     if indx + num_sl // 2 > total_num_sl:
#         return range(total_num_sl - num_sl + 1, total_num_sl + 1)
#
#     return range(indx - num_sl // 2, indx + num_sl // 2 + 1)
#
#
# def t1fit(MTi, T1):
#     return MTi[0]*(1-2*np.exp(-MTi[1:]/T1))
#
# class DataGenerator(data.Dataset):
#     'Generates data for Keras'
#
#     def __init__(self, input_IDs, output_IDs, params=None, nums_slices=None, mode='training'):
#         'Initialization'
#
#         self.output_IDs = output_IDs
#         self.input_IDs = input_IDs
#         self.dim = params.img_size[:]
#         self.dim.append(2)
#         self.n_channels = params.n_channels
#         self.n_spokes = params.n_spokes
#         self.nums_slices = nums_slices
#         self.complex_net = params.complex_net
#         self.mode = mode
#         self.params = params
#
#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return len(self.input_IDs)
#
#     def shuffel_cases(self):
#         sh_idx = list(range(0, len(self.input_IDs), self.params.num_phases))
#         np.random.shuffle(sh_idx)
#         rnds = np.asarray([list(range(id, id+25)) for id in sh_idx])
#         rnds = rnds.reshape((rnds.shape[0]*rnds.shape[1]))
#         self.input_IDs = [self.input_IDs[pid] for pid in rnds]
#         self.output_IDs = [self.output_IDs[pid] for pid in rnds]
#
#     def __getitem__(self, index):
#         'Generate one batch of data'
#         X, y, w_mask = self.generate_radial_t1(self.input_IDs[index], self.output_IDs[index])
#         return X, y, w_mask, self.input_IDs[index]
#
#     def generate_radial_t1(self, input_ID, output_ID):
#
#         sl = int(input_ID.split('/')[-1].split('_')[-2])
#         it = int(input_ID.split('/')[-1].split('_')[-1][:-4])
#         f_name = input_ID.split('/')[-1].split('_')[0] + '_' +input_ID.split('/')[-1].split('_')[1] + '.mat'
#
#         if params.rot_angle:
#             v_url = input_ID[0:72] + '_{0}_{1}_RotAng/'.format(self.n_spokes, self.params.gridding_method)
#         else:
#             v_url = input_ID[0:72] + '_{0}_{1}/'.format(self.n_spokes, self.params.gridding_method)
#
#         Input_Data_Loaded = False
#         if False and os.path.exists(v_url + f_name):
#             try:
#                 data = loadmat(v_url + f_name)['data']
#                 input = data['input'][0][0]
#                 output = data['output'][0][0]
#                 cont = data['cont'][0][0]
#                 T1 = data['T1'][0][0]
#                 Input_Data_Loaded = True
#             except:
#                 pass
#
#         if not Input_Data_Loaded:
#             print('Preparing Dataset: {0}'.format(input_ID))
#             ln_f = len(input_ID.split('/')[-1])
#             ss = loadmat(input_ID[:-ln_f] + f_name.split('_')[0] + '.mat')
#             img_s = ss['data'][0]['Img'][0] #[('Img'), ('SliceLocation'), ('TriggerTime'), ('EpiContours'), ('EndoContours')])
#
#             if(sl >= img_s.shape[-1]):
#                 sl = 0
#             ti = ss['data'][0]['TriggerTime'][0];
#             img_s = img_s[:,:,:,sl]
#
#             cont = ss['data'][0]['EpiContours'][0][:,:,:,sl]
#             mc = np.round(np.mean(cont, 0)).astype(int)
#
#             ## resize by interpolation
#             # img = np.moveaxis(img,(0,1,2,3),(2,3,0,1))
#             # img = torch.FloatTensor(img.astype(float))
#             # img = torch.nn.functional.interpolate(img, (200, 200), mode='bilinear')
#             # img = img.data.numpy()
#
#             ## resize by cropping around the heart
#             img = np.ndarray((200,200,img_s.shape[-1]))
#             mc[mc < 100] = 100
#             for t1w in range(0,img_s.shape[-1]):
#                 img[:,:,t1w] = img_s[mc[1,t1w]-100:mc[1,t1w]+100, mc[0,t1w]-100:mc[0,t1w]+100, t1w]
#
#             kspace = np.fft.fftshift(np.fft.fft2(img, axes=(0,1)), axes=(0,1))
#             # kspace = kspace[:,:,:,sl]
#
#             us_kspace = np.zeros((kspace.shape[0], kspace.shape[1], kspace.shape[2]))
#             us_kspace = us_kspace + 1j * us_kspace
#
#             for t1w in range(0, img.shape[2]):
#                 s_angle = ((np.pi / self.params.n_spokes) / (self.params.moving_window_size)) * t1w
#                 traj = create_radial_trajectory(kspace.shape[0], self.params.n_spokes, s_angle)
#                 traj_nn = np.round(
#                     np.reshape(traj, (traj.shape[0] * traj.shape[1], traj.shape[2])))  # nearest neighbour
#                 traj_nn = traj_nn + kspace.shape[0]//2 -1
#                 zks = np.zeros((kspace.shape[0], kspace.shape[1]))
#                 zks[traj_nn[:,0].astype(int), traj_nn[:,1].astype(int)] = 1
#                 us_kspace[:,:,t1w] = kspace[:, :, t1w] * zks
#
#
#             # tti = np.expand_dims(np.expand_dims(ti[:,sl],1), 2).repeat(200,1).repeat(200,2)
#             output = np.moveaxis(np.expand_dims(img, 0), (1,2,3), (2,3,1))
#             input = np.moveaxis(np.expand_dims(us_kspace, 0), (1,2,3), (2,3,1))
#
#             output[0, 0:2, :, :] = output[0, 0:2, :, :] * -1
#             tti = np.zeros(12)
#             tti[1:] = ti[:, sl]
#
#             T1 = np.zeros((200, 200))
#             for ii in range(0, 200):
#                 for jj in range(0, 200):
#                     tti[0] = np.abs(output[0,0,ii,jj])
#                     t1, pcv = curve_fit(t1fit, tti , output[0,:,ii,jj] , p0=100)
#                     T1[ii,jj] = t1
#             T1[T1>2000] = 2000
#             ## save the data for future
#             if not os.path.exists(v_url):
#                 os.makedirs(v_url)
#             data = {'input': input, 'output': output, 'cont': cont, 'T1': T1}
#             saveArrayToMat(data, 'data', f_name[:-4], v_url)
#
#         input = np.stack((input.real, input.imag), axis=4).astype(float)
#         output = output.astype(float)
#
#
#         # ## create weighting mask
#         # r = 0.5 * ((np.max(cont[:, 0, :], axis=0) - np.min(cont[:, 0, :], axis=0)) ** 2 + (
#         #             np.max(cont[:, 1, :], axis=0) - np.min(cont[:, 1, :], axis=0)) ** 2) ** 0.5
#         # mask = get_gaussian_mask(200, r * 2, circ_window_radius=r * 1.2)
#         # w_invT = np.ones((1, 1, 11)) / 10
#         # w_invT[:, :, 2] = w_invT[:, :, 3] = w_invT[:, :, 4] = 1
#         # mask = mask * w_invT
#         # mask = np.expand_dims(mask.swapaxes(2, 0), 0)
#
#         ## select window of inversion times
#         its = np.array(range(it - self.params.moving_window_size // 2, it + self.params.moving_window_size // 2 + 1))
#         its[its[:] > 11] = its[its[:] > 11] - 11
#         its[its[:]<1] = 11 - abs(its[its[:]<1])
#         its = list(its - 1)
#
#         input = input[:,its,:,:,:]
#         output = output[:,it-1,:,:]
#         # mask = mask[:,it-1,:,:]
#         T1 = np.expand_dims(T1, 0)
#
#         return input, output, T1#mask
#
#     def generate_radial_cine_mvw(self, input_ID, output_ID):
#
#         f_name0 = input_ID.split('/')[-1][:-4] + '.mat'
#
#         f_args = parse_dat_filename(input_ID.split('/')[-1])
#         mvw = np.asarray(range(f_args['phs'] - params.moving_window_size//2 -1, f_args['phs'] + params.moving_window_size//2 ))
#         mvw[mvw < 0] = mvw[mvw < 0] + params.num_phases
#         mvw[mvw > params.num_phases-1] = mvw[mvw > params.num_phases-1] - params.num_phases
#         mvw = mvw + 1
#
#         input_mv = np.zeros((params.moving_window_size, params.n_channels, params.img_size[0], params.img_size[1], 2))
#         mv_idx = 0
#
#         for mv in mvw.tolist():
#             f_name = get_dat_filename(slc=f_args['slc'], phs=mv,lins=f_args['lins'],cols=f_args['cols'], cha=f_args['cha'])[:-4] + '.mat'
#
#             if params.rot_angle:
#                 v_url = input_ID[0:69] + '_{0}_{1}_RotAng_{2}'.format(self.n_spokes, self.params.gridding_method,
#                                                                       self.params.gradient_delays_method) + input_ID[
#                                                                                                             69:-len(f_name0)]
#             else:
#                 v_url = input_ID[0:69] + '_{0}_{1}_{2}'.format(self.n_spokes, self.params.gridding_method,
#                                                                self.params.gradient_delays_method) + input_ID[
#                                                                                                      69:-len(f_name0)]
#
#             Input_Data_Loaded = False
#             if os.path.exists(v_url + f_name):
#                 try:
#                     data = loadmat(v_url + f_name)['data']
#                     input = data['input'][0][0]
#                     trajectory = data['trajectory'][0][0]
#                     ks_val = data['ks_val'][0][0]
#                     SNR_rank = data['SNR_rank'][0][0][0]
#                     gauss_params = data['gauss_param'][0][0][0]
#                     if input.shape[0] == self.n_channels:
#                         Input_Data_Loaded = True
#                 except:
#                     pass
#
#             if not Input_Data_Loaded:
#                 print('Preparing Dataset: {0}'.format(input_ID))
#                 print('File didnot exist : {0}'.format(v_url + f_name))
#                 kspace_lines = read_dat_file(input_ID)
#                 ''' interpolate the n_cols of kspace to have same size as reference images'''
#                 # dt = interp.interp1d(np.linspace(0, kspace_lines.shape[0], kspace_lines.shape[0]), kspace_lines, axis=0)
#                 # kspace_lines = dt(np.linspace(0, kspace_lines.shape[0], kspace_lines.shape[0]//2)) #
#
#                 ## zero-pad the kspace lines is equivalent to interpolation in image domain
#                 dim_diff = self.dim[0] - kspace_lines.shape[0] // 2
#                 if dim_diff % 2:
#                     raise Exception('dimension difference between raw kspace lines and input can not be odd!')
#                 else:
#                     zp = int(dim_diff / 2)
#                     if dim_diff > 0:
#                         kspace_lines = np.pad(kspace_lines, ((zp, zp), (0, 0), (0, 0), (0, 0)), 'constant')
#                     elif dim_diff < 0:
#                         kspace_lines = kspace_lines[zp:-zp, ]
#
#                 phase = 0
#                 if params.rot_angle:
#                     phase = parse_dat_filename(f_name)['phs'] - 1
#
#                 ## unify the number of coils per case
#                 kspace_lines, SNR_rank = stratify_kspace_channels(kspace_lines, self.n_channels)
#                 input, trajectory, ks_val = undersample_radial_kspace(kspace_lines, self.n_spokes,
#                                                                       trajectory=None,
#                                                                       gridding_method=self.params.gridding_method,
#                                                                       gradient_delays_method=self.params.gradient_delays_method,
#                                                                       k_neighbors=self.params.k_neighbors,
#                                                                       per_phase_rot=2 * phase)
#
#                 ## calculate gaussian weighting function paramters
#                 # gauss_params = kspaceImg_gauss_fit(input)
#                 # gauss_params[2] *= 3
#                 ## from kspacelines --> the problem is that the kspaceimages have density correction weighting so it is better to calculate from the image not from the original kspace lines
#                 gauss_params = kspacelines_gauss_fit(kspace_lines)
#                 gauss_params[1] /= 2
#                 gauss_params[2] /= 2  # kspace_lines are oversampled
#                 gauss_params[2] *= 3  # increase the STD by factor of 3
#
#                 if self.params.gridding_method == self.params.g_methods[0]:  # 'neighbours_matrix'
#                     input = np.moveaxis(input, [2, 3], [0, 1])
#                     # reshape as follow: in[N_neighbors_N_ch,Height,Width,cmplx] --> out[N_neighbors*N_ch,Height,Width,cmplx] ordered by N_neighbors first
#                     input = np.reshape(input,
#                                        [input.shape[0] * input.shape[1], input.shape[2], input.shape[3], input.shape[4]],
#                                        'F')
#                 else:
#                     input = np.moveaxis(input, [2], [0])
#
#                 ## save the data for future
#                 if not os.path.exists(v_url):
#                     os.makedirs(v_url)
#                 data = {'input': input, 'trajectory': trajectory, 'ks_val': ks_val,
#                         'SNR_rank': SNR_rank, 'gauss_param': gauss_params}
#                 saveArrayToMat(data, 'data', f_name[:-4], v_url)
#
#             if len(trajectory) == 0:
#                 trajectory = np.zeros((1, 1))
#             if len(ks_val) == 0:
#                 ks_val = np.zeros((1, 1))
#
#             input_mv[mv_idx, ] = input
#             mv_idx += 1
#
#         input_mv = np.moveaxis(input_mv, [0], [1])
#         ##################################################################
#         ## load refernece fullysampled images
#
#         Coil_combined = False
#
#         if Coil_combined:
#             '''read coil-combined 1-channel complex-valued data from .mat files'''
#             output = loadmat(output_ID)['data']
#
#         else:
#             '''read all 16 coils from DAT file'''
#             output = read_dat_file(output_ID)
#
#             ch_diff = output.shape[2] - self.n_channels
#
#             if ch_diff == 0:
#                 output = output[:, :, SNR_rank, :]
#             elif ch_diff > 0:
#                 out = output[:, :, SNR_rank[:-ch_diff], :]
#                 out[:, :, -1, :] = np.mean(output[:, :, SNR_rank[-ch_diff - 1:], :], axis=2, keepdims=False)
#                 output = out
#             elif ch_diff < 0:
#                 output = output[:, :, SNR_rank, :]
#                 output = np.append(output, output[:, :, -abs(ch_diff):, :], axis=2)
#             output = output[::-1, :, :, :].copy()
#             # # combine coils using square-root of sum-of-squares
#             # output = np.expand_dims(combine_channels_RSS(output), axis=0)
#
#         orig_size = output.shape
#         if self.dim[0] != output.shape[0] or self.dim[1] != output.shape[1]:
#             dt2 = interp.interp2d(np.linspace(0, output.shape[0], output.shape[0]),
#                                   np.linspace(0, output.shape[1], output.shape[1]), output)
#
#             output = dt2(np.linspace(0, output.shape[0], self.dim[0]),
#                          np.linspace(0, output.shape[1], self.dim[1]))
#
#         output = np.moveaxis(output, [2], [0])
#
#         return input_mv, output, trajectory, ks_val, orig_size, self.n_spokes / 198, gauss_params
#
#
#     def generate_radial_cine(self, input_ID, output_ID):
#
#         f_name = input_ID.split('/')[-1][:-4] + '.mat'
#
#         if params.rot_angle:
#             v_url = input_ID[0:69] + '_{0}_{1}_RotAng_{2}'.format(self.n_spokes, self.params.gridding_method,
#                                                        self.params.gradient_delays_method) + input_ID[69:-len(f_name)]
#         else:
#             v_url = input_ID[0:69] + '_{0}_{1}_{2}'.format(self.n_spokes, self.params.gridding_method,
#                                                        self.params.gradient_delays_method) + input_ID[69:-len(f_name)]
#
#         Input_Data_Loaded = False
#         if os.path.exists(v_url + f_name):
#             try:
#                 data = loadmat(v_url + f_name)['data']
#                 input = data['input'][0][0]
#                 trajectory = data['trajectory'][0][0]
#                 ks_val = data['ks_val'][0][0]
#                 SNR_rank = data['SNR_rank'][0][0][0]
#                 gauss_params = data['gauss_param'][0][0][0]
#                 if input.shape[0] == self.n_channels:
#                     Input_Data_Loaded = True
#             except:
#                 pass
#
#         if not Input_Data_Loaded:
#             print('Preparing Dataset: {0}'.format(input_ID))
#             kspace_lines = read_dat_file(input_ID)
#             ''' interpolate the n_cols of kspace to have same size as reference images'''
#             # dt = interp.interp1d(np.linspace(0, kspace_lines.shape[0], kspace_lines.shape[0]), kspace_lines, axis=0)
#             # kspace_lines = dt(np.linspace(0, kspace_lines.shape[0], kspace_lines.shape[0]//2)) #
#
#             ## zero-pad the kspace lines is equivalent to interpolation in image domain
#             dim_diff = self.dim[0] - kspace_lines.shape[0] // 2
#             if dim_diff % 2:
#                 raise Exception('dimension difference between raw kspace lines and input can not be odd!')
#             else:
#                 zp = int(dim_diff / 2)
#                 if dim_diff > 0:
#                     kspace_lines = np.pad(kspace_lines, ((zp, zp), (0, 0), (0, 0), (0, 0)), 'constant')
#                 elif dim_diff < 0:
#                     kspace_lines = kspace_lines[zp:-zp, ]
#
#             phase = 0
#             if params.rot_angle:
#                 phase = parse_dat_filename(f_name)['phs'] - 1
#
#             ## unify the number of coils per case
#             kspace_lines, SNR_rank = stratify_kspace_channels(kspace_lines, self.n_channels)
#             input, trajectory, ks_val = undersample_radial_kspace(kspace_lines, self.n_spokes,
#                                                                   trajectory=None,
#                                                                   gridding_method=self.params.gridding_method,
#                                                                   gradient_delays_method=self.params.gradient_delays_method,
#                                                                   k_neighbors=self.params.k_neighbors,
#                                                                   per_phase_rot=phase)
#
#             ## calculate gaussian weighting function paramters
#             # gauss_params = kspaceImg_gauss_fit(input)
#             # gauss_params[2] *= 3
#             ## from kspacelines --> the problem is that the kspaceimages have density correction weighting so it is better to calculate from the image not from the original kspace lines
#             gauss_params = kspacelines_gauss_fit(kspace_lines)
#             gauss_params[1] /= 2
#             gauss_params[2] /= 2  # kspace_lines are oversampled
#             gauss_params[2] *= 3  # increase the STD by factor of 3
#
#             if self.params.gridding_method == self.params.g_methods[0]: #'neighbours_matrix'
#                 input = np.moveaxis(input, [2, 3], [0, 1])
#                 #reshape as follow: in[N_neighbors_N_ch,Height,Width,cmplx] --> out[N_neighbors*N_ch,Height,Width,cmplx] ordered by N_neighbors first
#                 input = np.reshape(input, [input.shape[0]*input.shape[1], input.shape[2], input.shape[3], input.shape[4]], 'F')
#             else:
#                 input = np.moveaxis(input, [2], [0])
#
#             ## save the data for future
#             if not os.path.exists(v_url):
#                 os.makedirs(v_url)
#             data = {'input': input, 'trajectory': trajectory, 'ks_val': ks_val,
#                     'SNR_rank': SNR_rank, 'gauss_param': gauss_params}
#             saveArrayToMat(data, 'data', f_name[:-4], v_url)
#
#         if len(trajectory) == 0:
#             trajectory = np.zeros((1,1))
#         if len(ks_val) == 0:
#             ks_val = np.zeros((1,1))
#
#         ##################################################################
#         ## load refernece fullysampled images
#
#         Coil_combined = False
#
#         if Coil_combined:
#             '''read coil-combined 1-channel complex-valued data from .mat files'''
#             output = loadmat(output_ID)['data']
#
#         else:
#             '''read all 16 coils from DAT file'''
#             output = read_dat_file(output_ID)
#
#             ch_diff = output.shape[2] - self.n_channels
#
#             if ch_diff == 0:
#                 output = output[:, :, SNR_rank, :]
#             elif ch_diff > 0:
#                 out = output[:, :, SNR_rank[:-ch_diff], :]
#                 out[:, :, -1, :] = np.mean(output[:, :, SNR_rank[-ch_diff - 1:], :], axis=2, keepdims=False)
#                 output = out
#             elif ch_diff < 0:
#                 output = output[:, :, SNR_rank, :]
#                 output = np.append(output, output[:, :, -abs(ch_diff):, :], axis=2)
#             output = output[::-1, :, :, :].copy()
#             # # combine coils using square-root of sum-of-squares
#             # output = np.expand_dims(combine_channels_RSS(output), axis=0)
#
#         orig_size = output.shape
#         if self.dim[0] != output.shape[0] or self.dim[1] != output.shape[1]:
#             dt2 = interp.interp2d(np.linspace(0, output.shape[0], output.shape[0]),
#                                   np.linspace(0, output.shape[1], output.shape[1]), output)
#
#             output = dt2(np.linspace(0, output.shape[0], self.dim[0]),
#                          np.linspace(0, output.shape[1], self.dim[1]))
#
#         output = np.moveaxis(output, [2], [0])
#
#         return input, output, trajectory, ks_val, orig_size, self.n_spokes / 198, gauss_params
#
#     def __data_generation(self, index, input_IDs_temp, output_IDs_temp):
#         'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
#         # Initialization
#
#         if self.complex_net:
#             if len(self.dim) == 3:
#                 return self.generate_data2D(index, input_IDs_temp, output_IDs_temp)
#             elif len(self.dim) > 3 and self.mode == 'training':
#                 if params.num_slices_3D > 50:  # whole volume Feeding
#                     return self.generate_data3D(index, input_IDs_temp, output_IDs_temp)
#                 else:  # moving window feeding
#                     return self.generate_data3D_moving_window(index, input_IDs_temp, output_IDs_temp)
#             elif len(self.dim) > 3 and self.mode == 'testing':
#                 return self.generate_data3D_testing(index, input_IDs_temp, output_IDs_temp)
#         else:
#             if len(self.dim) == 2:
#                 return self.generate_data2D(index, input_IDs_temp, output_IDs_temp)
#             else:
#                 return self.generate_data3D(index, input_IDs_temp, output_IDs_temp)
#
#     def generate_data2D(self, index, input_IDs_temp, output_IDs_temp):
#         # Initialization
#         X = np.zeros((self.n_channels, *self.dim))
#         y = np.zeros((self.n_channels, *self.dim))
#
#         # Generate data
#         img = loadmat(input_IDs_temp)['Input_realAndImag']
#         orig_size = [img.shape[0], img.shape[1]]
#         #         for i, ID in enumerate(input_IDs_temp):
#         X[0,] = resizeImage(img, [self.dim[0], self.dim[1]])
#
#         #         for i, ID in enumerate(output_IDs_temp):
#         y[0, :, :, 0] = resizeImage(loadmat(output_IDs_temp)['Data'], [self.dim[0], self.dim[1]])
#         X = np.nan_to_num(X)
#         y = np.nan_to_num(y)
#         return X, y, orig_size
#
#     def generate_data3D(self, index, patients, out_patients):
#         '''
#         Read 3D volumes or stack of 2D slices
#         '''
#         Stack_2D = True
#
#         if Stack_2D:
#             slices = getPatientSlicesURLs(patients)
#             X = np.zeros((1, self.dim[0], self.dim[1], self.dim[2], 2))
#             y = np.zeros((1, self.dim[0], self.dim[1], self.dim[2], 2))
#
#             z1 = (len(slices[0]) - self.dim[2]) // 2
#             z2 = len(slices[0]) - self.dim[2] - z1
#
#             sz = 0
#             if z1 > 0:
#                 rng = range(z1, len(slices[0]) - z2)
#                 sz = -z1
#             elif z1 < 0:
#                 rng = range(0, len(slices[0]))
#                 sz = z1
#             elif z1 == 0:
#                 rng = range(0, self.dim[2])
#
#             for sl in rng:
#                 img = loadmat(slices[0][sl])['Input_realAndImag']
#                 orig_size = [img.shape[0], img.shape[1]]
#                 try:
#                     X[0, :, :, sl + sz, :] = resizeImage(img, [self.dim[0], self.dim[1]])
#
#                     y[0, :, :, sl + sz, 0] = resizeImage(loadmat(slices[1][sl])['Data'], [self.dim[0], self.dim[1]])
#                 except:
#                     stop = 1
#             X = np.nan_to_num(X)
#             y = np.nan_to_num(y)
#             return X, y, orig_size
#         else:
#             pass
#
#     def generate_data3D_moving_window(self, index, input_IDs_temp, output_IDs_temp):
#         '''
#             Moving window
#         '''
#         # Initialization
#         X = np.zeros((self.n_channels, *self.dim))
#         y = np.zeros((self.n_channels, *self.dim))
#
#         sl_indx = int(input_IDs_temp.split('/')[-1][8:-4])
#
#         rng = get_moving_window(sl_indx, self.dim[2], self.nums_slices[index])
#
#         i = 0
#         # Generate data
#         #         print(input_IDs_temp)
#         #         print('sl_indx->', sl_indx, '  nslices->',self.dim[2], 'max_slices', self.nums_slices[index] )
#         for sl in rng:
#             #             print(sl)
#             in_sl_url = '/'.join(input_IDs_temp.split('/')[0:-1]) + '/Input_sl' + str(sl) + '.mat'
#             out_sl_url = '/'.join(output_IDs_temp.split('/')[0:-1]) + '/Input_sl' + str(sl) + '.mat'
#             try:
#                 img = loadmat(in_sl_url)['Input_realAndImag']
#             except:
#                 print('Data Loading Error ..... !')
#             orig_size = [img.shape[0], img.shape[1]]
#             X[0, :, :, i, :] = resizeImage(img, [self.dim[0], self.dim[1]])
#             y[0, :, :, i, 0] = resizeImage(loadmat(out_sl_url)['Data'], [self.dim[0], self.dim[1]])
#             i += 1
#         #         print('---------------------------------')
#         X = np.nan_to_num(X)
#         y = np.nan_to_num(y)
#         return X, y, orig_size
#
#     def generate_data3D_testing(self, index, patients, out_patients):
#         def ceildiv(a, b):
#             return -(-a // b)
#
#         slices = getPatientSlicesURLs(patients)
#         X = np.zeros((1, self.dim[0], self.dim[1], len(slices[0]), 2))
#         y = np.zeros((1, self.dim[0], self.dim[1], len(slices[0]), 2))
#
#         for sl in range(0, len(slices[0])):
#             img = loadmat(slices[0][sl])['Input_realAndImag']
#             orig_size = [img.shape[0], img.shape[1]]
#             X[0, :, :, sl, :] = resizeImage(img, [self.dim[0], self.dim[1]])
#
#             y[0, :, :, sl, 0] = resizeImage(loadmat(slices[1][sl])['Data'], [self.dim[0], self.dim[1]])
#
#         X = np.nan_to_num(X)
#         y = np.nan_to_num(y)
#
#         #         n_batchs = ceildiv(len(slices[0]), self.dim[2])
#         #
#         #         # Initialization
#         #         X = np.zeros((n_batchs,1, *self.dim))
#         #         y = np.zeros((n_batchs,1, *self.dim))
#         #
#         #         ds_sl = 0
#         #         for bt in range(0, n_batchs):
#         #             for sl in range(0, self.dim[2]):
#         #                 if ds_sl >= len(slices[0]):
#         #                     break
#         # #                 print('ds_sl:',ds_sl, 'sl:',sl, 'bt:', bt)
#         #                 img = loadmat(slices[0][ds_sl])['Input_realAndImag']
#         #                 orig_size = [img.shape[0], img.shape[1]]
#         #                 X[bt,0,:,:,sl,:] = resizeImage(img,[self.dim[0],self.dim[1]])
#         #
#         #                 y[bt,0,:,:,sl,0] = resizeImage(loadmat(slices[1][ds_sl])['Data'],[self.dim[0],self.dim[1]])
#         #                 X = np.nan_to_num(X)
#         #                 y = np.nan_to_num(y)
#         #                 ds_sl += 1
#         return X, y, orig_size
#
# # class DataGenerator(data.Dataset):
# #     'Generates data for Keras'
# #     def __init__(self, input_IDs, output_IDs, undersampling_rates=None, dim=(256,256,2), n_channels=1,complex_net=True ,nums_slices=None):
# #         'Initialization'
# #         self.dim = dim
# #         self.output_IDs = output_IDs
# #         self.input_IDs = input_IDs
# #         self.n_channels = n_channels
# #         self.undersampling_rates = undersampling_rates
# #         self.nums_slices = nums_slices
# #         self.complex_net = complex_net
# #
# #     def __len__(self):
# #         'Denotes the number of batches per epoch'
# #         return len(self.input_IDs)
# #
# #     def __getitem__(self, index):
# #         'Generate one batch of data'
# #         if len(self.dim)==2 or (len(self.dim) ==3 and self.complex_net):
# #             return self.getItem2D(index)
# #         else:
# #             return self.getItem3D(index)
# #
# #     def getItem2D(self, index):
# #         # Generate data
# #         X, y, orig_size = self.__data_generation(self.input_IDs[index], self.output_IDs[index])
# #         if self.undersampling_rates is not None:
# #             usr = self.undersampling_rates[index]
# #         else:
# #             usr = None
# #
# #         return X, y, self.input_IDs[index], orig_size, usr
# #
# #     def getItem3D(self, index):
# #         # Generate data
# #         X, y, orig_size = self.__data_generation(self.input_IDs[index], self.output_IDs[index])
# #         if self.undersampling_rates is not None:
# #             usr = self.undersampling_rates[index]
# #         else:
# #             usr = None
# #
# #         return X, y, self.input_IDs[index], orig_size, usr
# #
# #
# #     def __data_generation(self, input_IDs_temp, output_IDs_temp):
# #         'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
# #         # Initialization
# #         X = np.zeros((self.n_channels, *self.dim))
# #         y = np.zeros((self.n_channels, *self.dim))
# #
# #         # Generate data
# #         img = loadmat(input_IDs_temp)['Input_realAndImag']
# #         orig_size = [img.shape[0], img.shape[1]]
# # #         for i, ID in enumerate(input_IDs_temp):
# #         X[0,] = resizeImage(img,[self.dim[0],self.dim[1]])
# #
# # #         for i, ID in enumerate(output_IDs_temp):
# #         y[0,:,:,0] = resizeImage(loadmat(output_IDs_temp)['Data'],[self.dim[0],self.dim[1]])
# #         X = np.nan_to_num(X)
# #         y = np.nan_to_num(y)
# #         return X, y, orig_size
#
#
#
