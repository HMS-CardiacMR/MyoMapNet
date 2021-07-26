from scipy import interpolate as interp
import numpy as np
import sys, os
from saveNet import saveTensorToMat, saveArrayToMat
import matplotlib.pyplot as plt
#from pynufft import NUFFT_cpu
from utils.data_vis import tensorshow, ntensorshow
import multiprocessing
import ctypes
from functools import partial
NUFFTobj = NUFFT_cpu()

sys.path.append('/home/helrewaidy/bart/python/')
from bart import bart, bart_nested
os.environ["TOOLBOX_PATH"] = '/home/helrewaidy/bart/'
import time
'''
[IMPORTANT] --> Bart docs (https://github.com/mikgroup/bart-workshop)
if error "Exception: Environment variable TOOLBOX_PATH is not set." appeared
in terminal:
export TOOLBOX_PATH=/home/helrewaidy/bart
export PATH=${TOOLBOX_PATH}:${PATH}
'''
from parameters import Parameters
params = Parameters()



def shared_array(shape):
    """
    Form a shared memory numpy array.
    http://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-python-multiprocessing
    """
    l = 1
    for i in shape:
        l *= i
    # print(np.prod(shape))
    shared_array_base = multiprocessing.Array(ctypes.c_float, l)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(*shape)
    return np.asarray(shared_array)
    

def sort_files(flist):
    return sorted(flist, key=lambda x: 1000 * parse_dat_filename(x)['slc'] + parse_dat_filename(x)['phs'])

def create_radial_trajectory(n_cols, n_views, s_angle=0):
    angles = np.linspace(0, np.pi-np.pi/n_views, n_views) + s_angle
    line = np.zeros([2, n_cols]); line[0, :] = np.linspace(-n_cols//2, n_cols//2, n_cols)
    traj = [np.matmul([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]], line) for ang in angles]
    return np.flip(np.moveaxis(traj, [2, 0], [0, 1]), 0) # [n_cols, n_views, 2]

def create_radial_trajectory2(n_cols, n_views, s_angle=0):
    angles = np.linspace(0, np.pi-np.pi/n_views, n_views) + s_angle
    line = np.zeros([2, n_cols]); line[0, :] = np.linspace(-n_cols//2, n_cols//2, n_cols)
    traj = [np.matmul([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]], line) for ang in angles]
    return np.flip(np.moveaxis(traj, [2, 0], [0, 1]), 0), angles # [n_cols, n_views, 2]

def get_radial_trajectory(kspace_lines, gradient_delays_method=None):
    n_cols = kspace_lines.shape[0]
    n_views = kspace_lines.shape[1]

    # if 'NONE' == gradient_delays_method: # use python not Not BART
    #     trajectory = create_radial_trajectory(n_cols, n_views)

    # else:
    trajectory = bart(1, 'traj -r -x{0} -y{1} -c'.format(n_cols, n_views))
    ckl = np.expand_dims(kspace_lines[:,:,:,0], 0) + 1j* np.expand_dims(kspace_lines[:,:,:,1],0);
    ckl = ckl.astype(np.complex64)

    if 'RING' == gradient_delays_method:
        trajectory = bart_nested(1, 2, 'estdelay -R -r1.5',
             'traj -x{0} -y{1} -r -c -q'.format(n_cols, n_views), trajectory ,ckl)

    elif 'AC-addaptive' == gradient_delays_method:
        trajectory = bart_nested(1, 2, 'estdelay',
             'traj -x{0} -y{1} -r -c -q'.format(n_cols, n_views), trajectory ,ckl)

    debug = False
    if debug:
        img_nufft = bart(1, 'nufft -d{0}:{1}:1 -i'.format(n_cols, n_cols), trajectory, ckl)
        gd_img_nufft = bart(1, 'nufft -d{0}:{1}:1 -i'.format(n_cols, n_cols), trajectory, ckl)
        rss_img = np.sqrt(np.sum(img_nufft.real**2 + img_nufft.imag**2, 3))
        gd_rss_img = np.sqrt(np.sum(gd_img_nufft.real**2 + gd_img_nufft.imag**2, 3))
        show_complex_image(rss_img, rng=(0, 0.005))
        show_complex_image(gd_rss_img, rng=(0, 0.005))

    trajectory = np.flip(np.moveaxis(trajectory[0:2,:,:].real, [0], [2]), 2)# [n_cols, n_views, 2]
    return trajectory

def show_complex_image(img, ch = 0,rng=(0, 0.1)):
    plt.figure()
    plt.imshow(np.abs(img[:,:,ch]), cmap='gray', vmin=rng[0], vmax=rng[1])
    plt.show()

def get_grid_neighbors(kspace_lines, grid_size=(416, 416), k_neighbors=50,  trajectory=None, gradient_delays_method='RING', neighbor_mat=True):
    n_cols = kspace_lines.shape[0]
    n_views = kspace_lines.shape[1]
    n_cha = kspace_lines.shape[2]

    if trajectory is None:
        trajectory = get_radial_trajectory(kspace_lines, gradient_delays_method=gradient_delays_method)

    if neighbor_mat:
        ngrid = np.zeros((grid_size[0], grid_size[0], k_neighbors, n_cha, 2))

    traj = np.reshape(trajectory, [n_cols * n_views, 2], order='F')
    traj = traj.real

    ksl = np.reshape(kspace_lines, [n_cols * n_views, n_cha, 2], order='F')

    Ksp_ri = np.zeros((grid_size[0], grid_size[1], k_neighbors, n_cha, 2))
    Loc_xy = 1e10 * np.ones((grid_size[0], grid_size[1], k_neighbors, 2))

    k = 0
    # st = time.time()
    for ic in range(-grid_size[0]//2, grid_size[0]-grid_size[0]//2):
        for jc in range(-grid_size[1]//2, grid_size[1]-grid_size[1]//2):
            i = ic + grid_size[0] // 2
            j = jc + grid_size[1] // 2

            gi = ic * n_cols / grid_size[0]
            gj = jc * n_cols / grid_size[1]

            # rd = 5
            # rd = 0.05 * (abs(ic) + abs(jc)) + 1
            # rd = (10/208) * (ic**2 + jc**2)**0.5 + 2
            rd = 10
            dist = ((traj[:, 0] - gi) ** 2 + (traj[:, 1] - gj) ** 2) ** 0.5
            idxs = np.argwhere(dist <= rd)[:, 0].tolist()
            idxs = np.array([x for _,x in sorted(zip(dist[idxs], idxs))])


            # tmp_n[i, j] = len(idxs)

            # idxs = sorted(dist)
            # tmp_n[i, j] = idxs[9]
            if neighbor_mat:
                if len(idxs) == 0:
                    continue

                rdist = np.round(dist[idxs] * (k_neighbors / rd), 0)
                #
                for n in range(0, k_neighbors - 1):
                    if any(rdist == n):
                        kidx = idxs[np.argwhere(rdist == n)[:, 0].tolist()]
                        ngrid[i, j, n, :, :] = np.mean(ksl[kidx, :, :], axis=0)
            else:
                s = len(idxs) if len(idxs)<=k_neighbors else k_neighbors
                Ksp_ri[i, j, 0:s, ] = ksl[idxs[:s].tolist()]
                Loc_xy[i, j, 0:s, 0], Loc_xy[i, j, 0:s, 1] = traj[idxs[:s].tolist(), 0] - gi,  traj[idxs[:s].tolist(), 1] - gj
                k += 1
                # Idx_list.append(idxs.tolist())
                # Dist.append(dist[idxs.tolist()].tolist())

    return ngrid if neighbor_mat else Ksp_ri, Loc_xy


def fill_row(rn, ksl, traj, grid_size, k_neighbors, n_batch, n_cols, n_cha):
    # print('start filling row')
    l_Ksp_ri = np.zeros((n_batch, rn[1], grid_size[1], k_neighbors, n_cha, 2))
    l_Loc_xy = 1e3 * np.ones((rn[1], grid_size[1], k_neighbors, 2))
    i = 0
    for ic in range(rn[0], rn[0]+rn[1]):
        # i = ic + grid_size[0] // 2
        gi = ic * n_cols / grid_size[0]
        for jc in range(-grid_size[1]//2, grid_size[1]-grid_size[1]//2):
            j = jc + grid_size[1] // 2
            gj = jc * n_cols / grid_size[1]

            rd = 10
            dist = ((traj[:, 0] - gi) ** 2 + (traj[:, 1] - gj) ** 2) ** 0.5
            idxs = np.argwhere(dist <= rd)[:, 0].tolist()
            idxs = np.array([x for _,x in sorted(zip(dist[idxs], idxs))])

            s = len(idxs) if len(idxs)<=k_neighbors else k_neighbors
            l_Ksp_ri[:, i, j, 0:s, ] = ksl[:, idxs[:s].tolist()]
            l_Loc_xy[i, j, 0:s, 0], l_Loc_xy[i, j, 0:s, 1] = traj[idxs[:s].tolist(), 0] - gi,  traj[idxs[:s].tolist(), 1] - gj
        i += 1
    
    # lock.acquire()

    s = rn[0] + grid_size[0] // 2
    t = rn[0] + rn[1] + grid_size[0] // 2

    # print(s, '  ', t)
    # saveArrayToMat(l_Ksp_ri, 'ksp_{0}_{1}'.format(s, t))
    # saveArrayToMat(l_Loc_xy, 'loc_{0}_{1}'.format(s, t))

    Ksp_ri[:, :, :, s:t, :, :] = np.float32(np.flip(np.moveaxis(l_Ksp_ri, [3, 4], [2, 1]), axis=4))
    # print(l_Ksp_ri.shape)
    Loc_xy[:, s:t, :, :] = np.float32(np.flip(np.moveaxis(l_Loc_xy, [2], [0]), axis=2))
    # print(l_Loc_xy.shape)

    # lock.release()

    # i = ic + grid_size[0] // 2
    # time.sleep(10)
    # gi = ic * n_cols / grid_size[0]
    # for jc in range(-grid_size[1] // 2, grid_size[1] - grid_size[1] // 2):
    #     j = jc + grid_size[1] // 2
    #     gj = jc * n_cols / grid_size[1]
    # 
    #     rd = 10
    #     dist = ((traj[:, 0] - gi) ** 2 + (traj[:, 1] - gj) ** 2) ** 0.5
    #     idxs = np.argwhere(dist <= rd)[:, 0].tolist()
    #     idxs = np.array([x for _, x in sorted(zip(dist[idxs], idxs))])
    # 
    #     s = len(idxs) if len(idxs) <= k_neighbors else k_neighbors
    #     Ksp_ri[:, j, 0:s, ] = ksl[:, idxs[:s].tolist()]
    #     Loc_xy[j, 0:s, 0], Loc_xy[j, 0:s, 1] = traj[idxs[:s].tolist(), 0] - gi, traj[idxs[:s].tolist(), 1] - gj
    # print('Finish filling row')
    # return Ksp_ri, Loc_xy


n_process = 16 ## could be only in {2, 4, 8, 13, 16, 26, ...etc} ## for input of size 416; the 16 cores will make everything easier, since is is dividable to 16
Ksp_ri = shared_array([params.batch_size, params.n_channels, params.k_neighbors, params.img_size[0], params.img_size[1], 2])
Loc_xy = shared_array([params.k_neighbors, params.img_size[0], params.img_size[0], 2])
pool = multiprocessing.Pool(processes=n_process)
# lock = multiprocessing.Lock()


def get_grid_neighbors_mp(kspace_lines, grid_size=(416, 416), k_neighbors=50,  trajectory=None, gradient_delays_method='RING', neighbor_mat=True):
    '''
        Multi-processing version with a batch dimension
    '''
    n_batch = kspace_lines.shape[0]
    n_cols = kspace_lines.shape[1]
    n_views = kspace_lines.shape[2]
    n_cha = kspace_lines.shape[3]

    if trajectory is None:
        trajectory = get_radial_trajectory(kspace_lines, gradient_delays_method=gradient_delays_method)

    # if neighbor_mat:
    #     ngrid = np.zeros((grid_size[0], grid_size[0], k_neighbors, n_cha, 2))

    traj = np.reshape(trajectory[0,], [n_cols * n_views, 2], order='F')
    # traj = traj.real

    ksl = np.reshape(kspace_lines, [n_batch, n_cols * n_views, n_cha, 2], order='F')

    # # Ksp_ri = np.zeros((grid_size[0], grid_size[1], k_neighbors, n_cha, 2))
    # # Loc_xy = 1e3 * np.ones((grid_size[0], grid_size[1], k_neighbors, 2))
    # print('In Grid Function')

    # st = time.time()

    rn = [(r, grid_size[0]//n_process) for r in range(-grid_size[0]//2, grid_size[0]-grid_size[0]//2, grid_size[0]//n_process)]
    

    fill_row_p = partial(fill_row, ksl=ksl, traj=traj, grid_size=grid_size, k_neighbors=k_neighbors, n_batch=n_batch, n_cols=n_cols, n_cha=n_cha)

    pool.map(fill_row_p, rn)

    # # st = time.time()
    # # ss = pool.map(fill_row_p, rn)
    # # print(time.time()-st)
    # # st = time.time()
    # Ksp_ri, Loc_xy = zip(*pool.map(fill_row_p, rn))
    # # print(time.time() - st)
    # Ksp_ri = np.moveaxis(np.asarray(Ksp_ri), [1, 5, 4, 0], [0, 1, 2, 4])
    # Ksp_ri = np.reshape(Ksp_ri, [n_batch, n_cha, k_neighbors, grid_size[0], grid_size[1], 2])
    # Ksp_ri  = np.flip(Ksp_ri, axis=4)
    # 
    # Loc_xy = np.flip(np.moveaxis(
    #             np.asarray(Loc_xy), [3], [0]).reshape([k_neighbors, grid_size[0], grid_size[1], 2]),
    #             axis=3)

    # # for ic in range(-grid_size[0]//2, grid_size[0]-grid_size[0]//2):
    # print('Finished Grid Function')
    return ngrid if neighbor_mat else Ksp_ri, Loc_xy


def get_radial_undersampled_image(img, n_views, rot_angle=0, grid_method='BART'):
    traj, angles = create_radial_trajectory2(2*img.shape[0], n_views, rot_angle)

    im = np.zeros((2*img.shape[0], 2*img.shape[1]))
    im[img.shape[0]//2:-img.shape[0]//2, img.shape[0]//2:-img.shape[0]//2] = img
    img = im.astype(np.float32).astype(np.complex64)
    DC = True
    if grid_method == 'BART':
        trajectory = np.moveaxis(traj, 2, 0)
        trajectory = np.concatenate((trajectory, np.zeros((1, img.shape[0], n_views))), 0)
        ksl = bart(1, 'nufft -d{0}:{1}:1 -t'.format(img.shape[0], img.shape[1]), trajectory, img)
        if DC:
            d = np.abs(np.linspace(-1, 1, img.shape[0]) + 1/img.shape[0])
            d = np.repeat(np.expand_dims(np.expand_dims(d, 0), 2), ksl.shape[2], 2)
            # for ii in range(0, ksl.shape[2]):
            #     d[:, :, ii] = d[:, :, ii] * np.exp(1j*angles[ii])
            # w = abs(d.real + 1j * d.imag) / max(abs(d.real + 1j * d.imag))
            ksl = ksl * d # * w
        output = bart(1, 'nufft -d{0}:{1}:1 -i -t'.format(img.shape[0], img.shape[1]), trajectory, ksl)
        output = output[img.shape[0]//4:-img.shape[0]//4, img.shape[0]//4:-img.shape[0]//4]

    elif grid_method == 'pyNUFFT':
        traj = np.reshape(traj, [img.shape[0] * n_views, 2], order='F')
        NUFFTobj.plan(om=traj, Nd=(img.shape[0], img.shape[1]), Kd=(2*img.shape[0], 2*img.shape[1]), Jd=(6, 6))
        ksll = NUFFTobj.forward(img)
        output = NUFFTobj.solve(ksll, 'dc', maxiter=30)

    return output

def grid_kspace(kspace_lines, trajectory=None, gridding_method='grid_kernels', gradient_delays_method='RING', k_neighbors=10):
    n_cols = kspace_lines.shape[0]
    n_views = kspace_lines.shape[1]
    n_cha = kspace_lines.shape[2]

    if trajectory is None:
        trajectory = get_radial_trajectory(kspace_lines, gradient_delays_method=gradient_delays_method)

    if gridding_method == 'grid_kernels':
        # return get_grid_neighbors(kspace_lines, grid_size=(416, 416), k_neighbors=k_neighbors, trajectory=trajectory, neighbor_mat=False)
        return kspace_lines, trajectory

    if gridding_method == 'neighbours_matrix':
        ngrid = get_grid_neighbors(kspace_lines, grid_size=(416, 416), k_neighbors=k_neighbors, trajectory=trajectory, neighbor_mat=True)
        output = np.flip(ngrid, 1)
        return output, [], []

    if 'pyNUFFT' == gridding_method:
        trajectory = np.reshape(trajectory, [n_cols * n_views, 2], order='F')
        kspace_lines = np.reshape(kspace_lines, [n_cols * n_views, n_cha, 2], order='F')
        trajectory = trajectory.real * np.pi / (n_cols // 2)
        NUFFTobj.plan(om=trajectory, Nd=(n_cols//2, n_cols//2), Kd=(n_cols, n_cols), Jd=(6, 6), batch=n_cha)
        kspace_lines = kspace_lines[:, :, 0] + 1j*kspace_lines[:, :, 1]
        # st = time.time()
        output = NUFFTobj.solve(kspace_lines, solver='cg', maxiter=10)
        # print(time.time()-st)
        output = np.fft.fftn(output, axes=(0, 1))
        ## output = NUFFTobj.y2k(kspace_lines)

    elif 'BART' == gridding_method:
        ckl = np.expand_dims(kspace_lines[:, :, :, 0], 0) + 1j * np.expand_dims(kspace_lines[:, :, :, 1], 0)
        ckl = ckl.astype(np.complex64)
        trajectory = np.moveaxis(trajectory, [2], [0])
        trajectory = np.concatenate((trajectory, np.zeros((1, n_cols, n_views))), 0)
        # st = time.time()
        output = bart(1, 'nufft -d{0}:{1}:1 -i'.format(n_cols, n_cols), trajectory, ckl)
        # print(time.time()-st)
        output = np.fft.fftn(np.squeeze(output,2)[n_cols//4:-n_cols//4, n_cols//4:-n_cols//4, :], axes=(0, 1))

    else:
        trajectory = np.reshape(trajectory, [n_cols * n_views, 2], order='F')
        kspace_lines = np.reshape(kspace_lines, [n_cols * n_views, n_cha, 2], order='F')
        grid_x, grid_y = np.mgrid[-n_cols//2:n_cols//2:2, -n_cols//2:n_cols//2:2]
        # st = time.time()
        output = interp.griddata(trajectory, kspace_lines, (grid_x, grid_y), method='linear')
        output[np.isnan(output)] = 0.0
        # print(time.time()-st)
        output = output[:, :, :, 0] + 1j * output[:, :, :, 1]
        # out = np.fft.fftshift(np.fft.fftshift(np.fft.ifft2(output, axes=(0, 1)), 0), 1)
        # show_complex_image(out, 0, (0, 6e-6))
    output = np.flip(output, 1)
    output = np.concatenate((np.expand_dims(output.real, 3), np.expand_dims(output.imag, 3)), 3)
    tj_idx = np.round(trajectory).astype(int)
    ksl_vals = output[tj_idx[:, 0], tj_idx[:, 1], :, :]

    return output, tj_idx, ksl_vals

def parse_dat_filename(filename):
    args = {}
    for s in filename[:-4].split('_'):
        if 'slc' in s:
            args['slc'] = int(s[3:])
        elif 'phs' in s:
            args['phs'] = int(s[3:])
        elif 'line' in s:
            args['line'] = int(s[4:])
        elif 'lins' in s:
            args['lins'] = int(s[4:])
        elif 'cols' in s:
            args['cols'] = int(s[4:])
        elif 'cha' in s:
            args['cha'] = int(s[3:])
    return args

def get_dat_filename(slc=1, phs=1, lins=198, cols=831, cha=15):
    return 'slc{0}_phs{1}_lins{2}_cols{3}_cha{4}.dat'.format(slc, phs, lins, cols, cha)


def read_dat_file(filename):
    args = parse_dat_filename(filename.split('/')[-1])
    f = open(filename, 'r')
    data = np.fromfile(f, np.float32, 2 * args['cols'] * args['lins'] * args['cha']) #float(f.read(4 * 2 * args['cols'] * args['lins'] * args['cha']))
    data = np.reshape(data, [2,  args['cols'], args['cha'], args['lins']], order='F')
    return np.moveaxis(data, [0, 1, 2, 3], [3, 0, 2, 1]) #[cols, lins/views, cha, 2]

def undersample_radial_kspace(kspace_lines, n_spokes , trajectory=None, gridding_method='pyNUFFT', gradient_delays_method='RING',k_neighbors=1, per_phase_rot=0):
    n_views = kspace_lines.shape[1]
    
    if trajectory is None:
        trajectory = get_radial_trajectory(kspace_lines, gradient_delays_method=gradient_delays_method)
            
    r = n_views//n_spokes

    slct_views = np.linspace(0, n_views-r, n_spokes).astype(int) + per_phase_rot
    slct_views[slct_views >= n_views] = slct_views[slct_views >= n_views] - n_views
    # trj = trajectory[::3, slct_views, :]; trj = np.reshape(trj, [trj.shape[0]*trj.shape[1],trj.shape[2]])
    # plt.figure(1)
    # plt.scatter(trj[:, 0], trj[:, 1], s=1, c=(1,0.9,0.1), marker='o')
    # plt.gca().set_facecolor((0.0, 0.0, 0.0))
    # plt.show()
    #
    # abs_ksl = abs(kspace_lines[:,:,:,0] + 1j*kspace_lines[:,:,:,0])
    # plt.figure(2)
    # plt.plot(abs_ksl[:,1,0])
    # plt.gca().set_facecolor((0.0, 0.0, 0.0))
    # plt.show()

    #output, traj, ks_lines = grid_kspace(kspace_lines[:, slct_views, :, :], trajectory[:, slct_views, :])

    return grid_kspace(kspace_lines[:, slct_views, :, :], trajectory[:, slct_views, :], gridding_method=gridding_method,
                       gradient_delays_method=gradient_delays_method, k_neighbors=k_neighbors)

def calculate_percoil_kspace_SNR(kspace_lines, window=50):
    half = kspace_lines.shape[0] // 2
    ksp_abs = abs(kspace_lines[:, :, :, 0] + 1j * kspace_lines[:, :, :, 1])
    # ksp_abs = np.reshape(abs(cmplx_ksp), [cmplx_ksp.shape[0]*cmplx_ksp.shape[1], cmplx_ksp.shape[2]])
    signal_power = np.sum(np.reshape(ksp_abs[half - window:half + window, ],
                          [2 * window * kspace_lines.shape[1], kspace_lines.shape[2]]), axis=0)
    ksp_abs[half - window:half + window] = 0
    noise_power = np.sum(np.reshape(ksp_abs, [kspace_lines.shape[0] * kspace_lines.shape[1], kspace_lines.shape[2]]),
                         axis=0)
    return 10 * np.log10(signal_power / noise_power)

def stratify_kspace_channels(kspace_lines, n_channels):
    SNR = calculate_percoil_kspace_SNR(kspace_lines)

    ch_diff = kspace_lines.shape[2] - n_channels
    SNR_rank = np.argsort(SNR)[::-1] #high SNR coils are first

    if ch_diff == 0:
        return kspace_lines[:, :, SNR_rank, :], SNR_rank
    elif ch_diff > 0:
        ksp = kspace_lines[:, :, SNR_rank[:-ch_diff], :]
        ksp[:, :, -1, :] = np.mean(kspace_lines[:, :, SNR_rank[-ch_diff-1:], :], axis=2, keepdims=False)
        return ksp, SNR_rank
    elif ch_diff < 0:
        ksp = kspace_lines[:, :, SNR_rank, :]
        return np.append(ksp, ksp[:, :, -abs(ch_diff):, :], axis=2), SNR_rank

def combine_channels_RSS(data):
    return np.sqrt(np.sum(data[:, :, :, 0] ** 2 + data[:, :, :, 1] ** 2, axis=2))

def load_dataset(path=None, kspace=False, visualization=False):
    dataset = []
    datasets_dirs = sorted(os.listdir(path), key=lambda x: int(x))
    for i, dst in enumerate(datasets_dirs):
        slices = sort_files(os.listdir(path + '/' + dst))
        p_data = []
        for j, slc in enumerate(slices):
            data = read_dat_file(path + '/' + dst + '/' + slc)
            print(j)
            if kspace:
                ''' interpolate the n_cols of kspace to have same size as reference images'''

                dt = interp.interp1d(np.linspace(0, data.shape[0], data.shape[0]), data, axis=0)
                kspace_lines = dt(np.linspace(0, data.shape[0], data.shape[0]//2))

                data = undersample_radial_kspace(kspace_lines, 196)
                visualization = True
                if visualization:
                    # ksp = np.sqrt(np.sum(np.sqrt(data[:, :, :, 0] ** 2 + data[:, :, :, 1] ** 2), axis=2))
                    # plt.imshow(ksp)

                    cmplx_ksp = data[:, :, :, 0] + 1j * data[:, :, :, 1]
                    window = 50
                    half = kspace_lines.shape[0]//2

                    ksp_abs = abs(kspace_lines[:,:,:,0] + 1j*kspace_lines[:,:,:,1])
                    # ksp_abs = np.reshape(abs(cmplx_ksp), [cmplx_ksp.shape[0]*cmplx_ksp.shape[1], cmplx_ksp.shape[2]])
                    signal_power = np.sum(np.reshape(ksp_abs[half-window:half+window, ],
                                         [2*window*kspace_lines.shape[1], kspace_lines.shape[2]]), axis=0)
                    ksp_abs[half-window:half+window] = 0
                    noise_power = np.sum(np.reshape(ksp_abs,
                                         [kspace_lines.shape[0]*kspace_lines.shape[1], kspace_lines.shape[2]]), axis=0)
                    SNR = 10*np.log10(signal_power/noise_power)

                    cmplx_img = np.fft.fftshift(np.fft.fft2(cmplx_ksp, axes=[0, 1]), axes=[0, 1])
                    for im in range(0,kspace_lines.shape[2]):
                        img = abs(cmplx_img[:, :, im])
                        plt.figure()
                        plt.imshow(img)
                        plt.title('SNR: {0}'.format(SNR[im]))
                        plt.show()
                        plt.pause(0.1)
                        plt.close()

                    # img = np.sqrt(np.sum(abs(cmplx_img), axis=2))
                    # plt.close(); plt.imshow(img)

                data = grid_kspace(data) # instead of interpolation [0::2, :, :, :]
                # saveArrayToMat(data, 'data')

            p_data.append(data)
        # saveArrayToMat(p_data, 'ds')
        # dataset.append(p_data)

# load_dataset('/data2/helrewaidy/cine_recon/ICE_recon_dat_files/ice_dat_files/kspace/', kspace=True)

