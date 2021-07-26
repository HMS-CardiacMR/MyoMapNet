import torch
from torch.utils import data
from parameters import Parameters
from scipy.io import loadmat, savemat
import numpy as np
import os
from saveNet import *
from utils.gridkspace import *
from utils.gaussian_fit import gauss_fit, kspacelines_gauss_fit, kspaceImg_gauss_fit


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

    for dir in params.dir:
        datasets_dirs = sorted(os.listdir(dir + 'image/'), key=lambda x: int(x))
        for i, dst in enumerate(datasets_dirs):
            params.patients.append(dst)
            kspaces = sort_files(os.listdir(dir + 'kspace/' + dst))
            params.num_slices_per_patient.append(len(kspaces))

            for j, ksp in enumerate(kspaces):
                params.input_slices.append(dir + 'kspace/' + dst + '/' + ksp)

            '''read all 16 coils from DAT file'''
            images = sort_files(os.listdir(dir + 'image/' + dst))
            for j, img in enumerate(images):
                params.groundTruth_slices.append(dir + 'image/' + dst + '/' + img)

            '''read coil-combined 1-channel complex-valued data from .mat files'''
    #             images = sort_files(os.listdir(dir + 'ref/' + dst))
    #             for j, img in enumerate(images):
    #                 params.groundTruth_slices.append(dir + 'ref/' + dst + '/' + img)

    print('-- Number of Datasets: ' + str(len(params.patients)))

    training_ptns = int(params.training_percent * len(params.patients))

    training_end_indx = sum(params.num_slices_per_patient[0:training_ptns + 1])

    params.training_patients_index = range(0, training_ptns + 1)

    dim = params.img_size[:]
    dim.append(2)

    tr_samples = 1

    training_DS = DataGenerator(input_IDs=params.input_slices[:training_end_indx:tr_samples],
                                output_IDs=params.groundTruth_slices[:training_end_indx:tr_samples],
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


def get_moving_window(indx, num_sl, total_num_sl):
    if indx - num_sl // 2 < 1:
        return range(1, num_sl + 1)

    if indx + num_sl // 2 > total_num_sl:
        return range(total_num_sl - num_sl + 1, total_num_sl + 1)

    return range(indx - num_sl // 2, indx + num_sl // 2 + 1)


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
        X, y, trajectory, ks_val, orig_size, usr, gauss_params = self.generate_radial_cine_mvw(self.input_IDs[index],
                                                                             self.output_IDs[index])
        return X, y, trajectory, ks_val, self.input_IDs[index], orig_size, usr, gauss_params

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



