
import shutil
import sys
import random
# import pyCompare
# import pingouin as pg
import scipy
from scipy.io import loadmat, savemat
from tensorboardX import SummaryWriter
from torch import optim
import matplotlib.pyplot as plt
from Loss import *
import torch.nn.modules.loss as Loss
# import torchvision.utils as vutils
from FCMyoMapNet import UNet
import traceback


import h5py
#----------------------------------------------------------#
#---------Initialization and Golbal setting----------------#
#----------LOAD LATEST (or SPECIFIC) MODEL-----------------#
#----------------------------------------------------------#
#scaling factor for input inverion time and output T1,
TimeScaling = 1000;
TimeScalingFactor =1/TimeScaling
T1sigNum = 4
T1sigAndTi = T1sigNum*2;
# set seed points
seed_num = 888

torch.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)
random.seed(seed_num)
np.random.seed(seed_num)
params = Parameters()   #load parameter

T1sigNum = params.inputLen
T1sigAndTi = T1sigNum*2
# Create Model
#MyoMapNet = UNet(10, 1)        #For MyoMapNet_5PreGd;
#MyoMapNet = UNet(8, 1)        # For MyoMapNet_4PostGd, MyoMapNet_4PreandPostGd, MyoMapNet_4PreGd
MyoMapNet = UNet(T1sigAndTi, 1)

LOSS = list()  #training loss
myoAvgLossAllEpochs = list()  #validation loss
bpAvgLossAllEpochs = list()     #validatino loss
itNum = 0

# Print the total parameters
def multiply_elems(x):
    m = 1
    for e in x:
        m *= e
    return m
num_params = 0
for parameters in MyoMapNet.parameters():
    num_params += multiply_elems(parameters.shape)
print('Total number of parameters: {0}'.format(num_params))
#loading trained model



MyoMapNet.to(torch.device('cpu'))

# if not os.path.exists(params.model_save_dir):
#     os.makedirs(params.model_save_dir)
#
# if not os.path.exists(params.tensorboard_dir):
#     os.makedirs(params.tensorboard_dir)
#
# if not os.path.exists(params.validation_dir):
#     os.makedirs(params.validation_dir)
writer = SummaryWriter(params.tensorboard_dir)


##  0: don't load any model; start from model #1
##  num: load model #num

optimizer = optim.SGD(MyoMapNet.parameters(), lr=params.args.lr, momentum=0.8)

w_mae = weighted_mae()
w_mse = weighted_mse()

models = os.listdir(params.model_save_dir)
models = [m for m in models if m.endswith('.pth')]
s_epoch = -1  ## -1: load latest model or start from 1 if there is no saved models. Currently, 4996 for 4HBsPreandPost, 2775 for 4HBsPost, 2961 for 4HBsPre, 2994 for 5HBsPre
print(len(models))

if s_epoch == -1:
    if len(models) == 0:
        s_epoch = 1
    else:
        #loading the latest model
        try:
            s_epoch = max([int(epo[11:-4]) for epo in models[:]])
            print('loading model at epoch ' + str(s_epoch))
            model = torch.load(params.model_save_dir + models[0][0:11] + str(s_epoch) + '.pth')
            MyoMapNet.load_state_dict(model['state_dict'])
            optimizer.load_state_dict(model['optimizer'])

            itNum = model['iteration']
            lossTmp = model['loss']
            # LOSS = lossTmp.tolist()
            print(LOSS)
            itNum = model['iteration']
            lossTmp = loadmat('{0}mse_R{1}_Trial{2}'.format(params.tensorboard_dir, str(params.Rate), params.trialNum))['mse']
            LOSS = lossTmp.tolist()

            myoAvgLossTmp = loadmat(
                '{0}avgLoss_Myo_allEpochs_R{1}_Trial{2}'.format(params.validation_dir, str(params.Rate), params.trialNum))[
                'myoAvgLossAllEpochs']  # load training loss
            myoAvgLossAllEpochs = myoAvgLossTmp.tolist()  # validation loss

            bpAvgLossTmp = loadmat(
                '{0}avgLoss_BP_allEpochs_R{1}_Trial{2}'.format(params.validation_dir, str(params.Rate), params.trialNum))[
                'bpAvgLossAllEpochs']  # load training loss
            bpAvgLossAllEpochs = bpAvgLossTmp.tolist()  # validatino loss

        except:
            print('Model {0} does not exist!'.format(s_epoch))
elif s_epoch == 0:
    s_epoch = 1   #creat a model
else:
    try:
        #load specific model
        # s_epoch = max([int(epo[11:-4]) for epo in models[:]])
        print('loading model at epoch ' + str(s_epoch))
        model = torch.load(params.model_save_dir + models[0][0:11] + str(s_epoch) + '.pth')
        # LOSS = model['loss']
        # print(LOSS)
        itNum = model['iteration']

        MyoMapNet.load_state_dict(model['state_dict'])
        optimizer.load_state_dict(model['optimizer'])
        lossTmp = loadmat('{0}mse_R{1}_Trial{2}'.format(params.tensorboard_dir, str(params.Rate), params.trialNum))['mse']    #load training loss
        LOSS = lossTmp.tolist()

        myoAvgLossTmp = loadmat(
            '{0}avgLoss_Myo_allEpochs_R{1}_Trial{2}'.format(params.validation_dir, str(params.Rate), params.trialNum))[
            'myoAvgLossAllEpochs']  # load training loss
        myoAvgLossAllEpochs = myoAvgLossTmp.tolist()  # validation loss

        bpAvgLossTmp = loadmat(
            '{0}avgLoss_BP_allEpochs_R{1}_Trial{2}'.format(params.validation_dir, str(params.Rate), params.trialNum))[
            'bpAvgLossAllEpochs']  # load training loss
        bpAvgLossAllEpochs = bpAvgLossTmp.tolist()  # validatino loss
    except:
        print('Model {0} does not exist!'.format(s_epoch))

## copy the code with the model saving directory
os.system("cp -r {0} {1}".format(os.getcwd(), params.model_save_dir))
print('Model copied!')

def convertndarrytochar(inputarr):
        return 0

#loading dataset
if params.Training_Only:

    # loading training dateset (simulation)
    if params.training_with_Sim:
        print('Start loading training dataset')
        #read mat file
        #data = loadmat(params.preMyoSimFile)['data']
        matFile = h5py.File(params.prepostallInvivoTrain)
        data = matFile['data']
        T1arr =  np.array(data['T1'])       #from signal without noise
        T1StarNoisy =  np.array(data['T1_star_noisy'])
        ANoisy =  np.array(data['A_noisy'])
        BNoisy =  np.array(data['B_noisy'])
        T1Noisy =  np.array(data['T1_noisy'])
        T15Noisy =  np.array(data['T15_noisy'])
        T1wNoisy =  np.array(data['T1w_noisy'])
        Tiarr =  np.array(data['Ti'])

        #80% samples are used in training
        totalSamples = T1arr.shape[0]
        nSampleForTraining = np.fix(totalSamples*0.8).astype(int)
        # The simulated signals are stored into a Matrix. The size of matrix is same with that of in-vivo image

        ## Image size is
        sx, sy = 64, 64
        tr_N = np.fix(nSampleForTraining/(sx*sy)).astype(int)

        tr_t1w_TI = np.zeros((tr_N, 1, T1sigAndTi, sx, sy))     # include signals and corrsponding TI times
        tr_T1 = np.zeros((tr_N, 1, sx, sy))
        tr_T1_5 = np.zeros((tr_N, 1, sx, sy))

        tr_A = np.zeros((tr_N, 1, sx, sy))
        tr_B = np.zeros((tr_N, 1, sx, sy))
        tr_T1_noNoisy = np.zeros((tr_N, 1, sx, sy))
        tr_T1star = np.zeros((tr_N, 1, sx, sy))

        tr_mask = np.ones((tr_N, 1, sx, sy))
        tr_LVmask = np.zeros((tr_N, 1, sx, sy))
        tr_ROImask = np.zeros((tr_N, 1, sx, sy))
        tr_sliceID = list()

        for ix in range (0,tr_N):

            ixRange = range(ix*sx*sy,(ix+1)*sx*sy)
            t1wSigsTmp = T1wNoisy[0:T1sigNum,ixRange]
            tiTimesTmp = Tiarr[0:T1sigNum,ixRange]*TimeScalingFactor
            t1NoisyTmp = T1Noisy[ixRange,0]*TimeScalingFactor
            #rest are not used
            t1Tmp = T1arr[ixRange,0]*TimeScalingFactor
            t1starTmp = T1StarNoisy[ixRange,0]*TimeScalingFactor
            ATmp = ANoisy[ixRange,0]
            BTmp = BNoisy[ixRange,0]
            t15fitNoisyTmp = T15Noisy[ixRange,0]*TimeScalingFactor

            t1wCTi = np.concatenate((t1wSigsTmp.transpose(),tiTimesTmp.transpose()),axis=1).reshape(sx,sy,T1sigAndTi)
            t1wCTi = np.transpose(t1wCTi,(2,0,1))

            #input is magnitude signals
            tr_t1w_TI[ix,0,:,:,:] = np.abs(t1wCTi)
            tr_T1[ix,0,:,:] = t1NoisyTmp.reshape(sx,sy)

            tr_A[ix,0,:,:] = ATmp.reshape(sx,sy)
            tr_B[ix, 0, :, :] = BTmp.reshape(sx, sy)
            tr_T1star[ix,0,:,:] = t15fitNoisyTmp.reshape(sx,sy)
            tr_T1_5[ix, 0, :, :] = t15fitNoisyTmp.reshape(sx, sy)

    elif params.training_with_Invivo:


    # loading in-vivo training data
        print('Start loading training dataset')

        if T1sigNum == 4:
            t1wtiIdx = [0,  1, 2,   3,  5,  6,  7, 8]
        else:
            t1wtiIdx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # read pre-contrast mat file
        data = h5py.File(params.TrainPrePostInvivo_5HBs)
        tr_T1_tmp = data['allT1Maps'][:]  #T1
        torch_data = torch.from_numpy(tr_T1_tmp)
        torch_data_per = torch_data.permute((2,1,0))
        tr_T1_3 = torch_data_per.numpy()
        tr_T1 = np.zeros((tr_T1_3.shape[0],1,tr_T1_3.shape[1],tr_T1_3.shape[2]))
        #tr_mask = np.ones(tr_T1.shape)
        for ix in range(0,tr_T1_3.shape[0]):
            tr_T1[ix,0,:,:] = tr_T1_3[ix,:,:]

        tr_mask_tmp = data['allMask'][:]  #T1
        torch_mask = torch.from_numpy(tr_mask_tmp)
        torch_mask_per = torch_mask.permute((2,1,0))
        torch_mask_3 = torch_mask_per.numpy()
        tr_mask = np.zeros((torch_mask_3.shape[0],1,torch_mask_3.shape[1],torch_mask_3.shape[2]))
        #tr_mask = np.ones(tr_T1.shape)
        for ix in range(0,torch_mask_3.shape[0]):
            tr_mask[ix,0,:,:] = torch_mask_3[ix,:,:]

        tr_t1w_TI_tmp = data['allT1wTIs'][:]  # T1w_Tis
        #torch_data = torch.from_numpy(tr_t1w_TI_tmp[:,:,[0,1,2,3,5,6,7,8],:,:])
        torch_data = torch.from_numpy(tr_t1w_TI_tmp[:, :, :, :, :])
        torch_data_per = torch_data.permute((4,3,2,1,0))
        tr_t1w_TI = torch_data_per.numpy()
        # tr_B = loadmat(params.preInvivoFile)['allBsigs']  #B
        # tr_A = loadmat(params.preInvivoFile)['allAsigs']  #A
        # tr_T1star = loadmat(params.preInvivoFile)['allT1stMaps']  #T1*
        tr_T1_5 = tr_T1
        tr_N = tr_T1.shape[0]


        print('End loading pre-contrast training dataset')


    # plt.imshow(tr_T1[1800,0,:,:], cmap='plasma', vmin=0, vmax=2)
    # plt.colorbar()
    # plt.xticks(())
    # plt.yticks(())
    # plt.title("T1 maps")
    # plt.show()
    #
    # plt.imshow(tr_t1w_TI[100,0,5,:,:], cmap='plasma', vmin=0, vmax=2)
    # plt.colorbar()
    # plt.xticks(())
    # plt.yticks(())
    # plt.title("T1 maps")
    # plt.show()


    ##loading validation data


    print('Start loading in-vivo validation dataset')
    # read pre-contrast mat file
    data = h5py.File(params.ValidationPrePostInvivo_5HBs)
    val_T1_tmp = data['allT1Maps'][:]  # T1
    torch_data = torch.from_numpy(val_T1_tmp)
    torch_data_per = torch_data.permute((2, 1, 0))
    val_T1_3 = torch_data_per.numpy()
    val_T1 = np.zeros((val_T1_3.shape[0], 1, val_T1_3.shape[1], val_T1_3.shape[2]))
    # tr_mask = np.ones(tr_T1.shape)
    for ix in range(0, val_T1_3.shape[0]):
        val_T1[ix, 0, :, :] =val_T1_3[ix, :, :]

    #load LV and BP Mask
    val_mask_tmp = data['myobpMask'][:]  # T1
    torch_mask = torch.from_numpy(val_mask_tmp)
    torch_mask_per = torch_mask.permute((2, 1, 0))
    torch_mask_3 = torch_mask_per.numpy()
    val_LVBpmask = np.zeros((torch_mask_3.shape[0], 1, torch_mask_3.shape[1], torch_mask_3.shape[2]))
    # tr_mask = np.ones(tr_T1.shape)
    for ix in range(0, torch_mask_3.shape[0]):
       val_LVBpmask[ix, 0, :, :] = torch_mask_3[ix, :, :]

    #load Myo Mask

    val_mask_tmp = data['myoMask'][:]  # T1
    torch_mask = torch.from_numpy(val_mask_tmp)
    torch_mask_per = torch_mask.permute((2, 1, 0))
    torch_mask_3 = torch_mask_per.numpy()
    val_Myomask = np.zeros((torch_mask_3.shape[0], 1, torch_mask_3.shape[1], torch_mask_3.shape[2]))
    #tr_mask = np.ones(tr_T1.shape)
    for ix in range(0, torch_mask_3.shape[0]):
        val_Myomask[ix, 0, :, :] = torch_mask_3[ix, :, :]
    #load BP Mask
    val_mask_tmp = data['bpMask'][:]  # T1
    torch_mask = torch.from_numpy(val_mask_tmp)
    torch_mask_per = torch_mask.permute((2, 1, 0))
    torch_mask_3 = torch_mask_per.numpy()
    val_BPmask = np.zeros((torch_mask_3.shape[0], 1, torch_mask_3.shape[1], torch_mask_3.shape[2]))
    # tr_mask = np.ones(tr_T1.shape)
    for ix in range(0, torch_mask_3.shape[0]):
        val_BPmask[ix, 0, :, :] = torch_mask_3[ix, :, :]

    val_t1w_TI_tmp = data['allT1wTIs'][:]  # T1w_Tis

    #torch_data = torch.from_numpy(val_t1w_TI_tmp[:, :, [0,1,2,3,5,6,7,8], :, :])
    torch_data = torch.from_numpy(val_t1w_TI_tmp[:, :, :, :, :])
    torch_data_per = torch_data.permute((4, 3, 2, 1, 0))
    val_t1w_TI = torch_data_per.numpy()

    val_sliceID_input = data['subjectsLists'][:]
    val_sliceID = list()
    for ix in range(0, val_sliceID_input.shape[1]):
        tmpstr = ''
        for ij in range(0, val_sliceID_input.shape[0]):
            tmpstr =tmpstr+ chr(val_sliceID_input[ij,ix])
        val_sliceID.append(tmpstr)

    # torch_slice = torch.from_numpy(tst_sliceID)
    # tst_sliceID = tst_sliceID.tostring().decode("ascii")
    # tst_sliceID
    # tr_B = loadmat(params.preInvivoFile)['allBsigs']  #B
    # tr_A = loadmat(params.preInvivoFile)['allAsigs']  #A
    # tr_T1star = loadmat(params.preInvivoFile)['allT1stMaps']  #T1*
    # tr_T1_5 = tr_T1
    # tr_N = tr_T1.shape[0]
    # plt.imshow(tst_T1[0,0,:,:], cmap='plasma', vmin=0, vmax=3)
    # plt.colorbar()
    # plt.xticks(())
    # plt.yticks(())
    # plt.title("T1 maps")
    # plt.show()
    print('End loading validation dataset')

def train(net):

    ##
    i = itNum  #
    # fig, axs = plt.subplots(5, 6)
    params.batch_size = 40  # Batch size is 40 for simulation and in-vivo
    tr_N = tr_T1.shape[0]
    tr_lst = list(range(0, tr_N))

    initialLoss = 1e10
    bestModelEpochIx = 0
    tmpMyoBloodLoss = 0


    DebugSave = False
    #epoch loop
    for epoch in range(s_epoch, params.epochs+1):
        print('epoch {}/{}...'.format(epoch, params.epochs))

        random.shuffle(tr_lst)  # This is only used for in vivo data

        try:
            ## start Training
            l = 0
            itt = 0
            TAG = 'Training'
            MAX = list()

            #batch loop
            for idx in range(0, tr_N, params.batch_size):
                try:
                    lst = tr_lst[idx:idx+params.batch_size]
                    X = Variable(torch.FloatTensor(tr_t1w_TI[lst,:,:,:,:])).to('cuda:0')
                    y = Variable(torch.FloatTensor(tr_T1[lst,:,:,:])).to('cuda:0')
                    T1_5 = Variable(torch.FloatTensor(tr_T1_5[lst,:,:,:])).to('cuda:0')
                    w_mask = Variable(torch.FloatTensor(tr_mask[lst,:,:,:])).to('cuda:0')
                    #sliceID = tr_sliceID[lst].tolist()


                    xs = X.shape
                    X = X.permute((0, 3, 4, 1, 2)).reshape((xs[0]*xs[3]*xs[4],xs[1],xs[2]))
                    y = y.permute((0, 2, 3, 1)).reshape((xs[0] * xs[3] * xs[4], xs[1])).unsqueeze(-1)

                    w_mask = w_mask.permute((0, 2, 3, 1)).reshape((xs[0] * xs[3] * xs[4], xs[1])).unsqueeze(-1)

                    #predicated by net
                    net.train()
                    y_pred = net(X.to('cuda:0')).to('cuda:0')

                except Exception as e:
                    traceback.print_exc()
                    print(e)
                    continue


                #for 4HBsPreLowLR and 4HBsPreandPost
                if epoch > 4000:
                    losstmp = w_mask * torch.abs(y_pred - y)
                    w_mask[losstmp>0.2] = 0
                if epoch > 4500:
                    losstmp = w_mask * torch.abs(y_pred - y)
                    w_mask[losstmp>0.1] = 0

                # #for 4HBsPre33
                # if epoch > 500:
                #     losstmp = w_mask * torch.abs(y_pred - y)
                #     w_mask[losstmp>0.2] = 0
                # if epoch > 1000:
                #     losstmp = w_mask * torch.abs(y_pred - y)
                #     w_mask[losstmp>0.1] = 0

                #for all four models
                # if epoch > 2000:
                #     losstmp = w_mask * torch.abs(y_pred - y)
                #     w_mask[losstmp>0.2] = 0
                # if epoch > 2500:
                #     losstmp = w_mask * torch.abs(y_pred - y)
                #     w_mask[losstmp>0.1] = 0

                # if epoch > 5000:
                #     losstmp = w_mask * torch.abs(y_pred - y)
                #     w_mask[losstmp>0.05] = 0
                # if epoch > 5500:
                #     losstmp = w_mask * torch.abs(y_pred - y)
                #     w_mask[losstmp>0.03] = 0
                loss = w_mae(y_pred, y, w_mask)

                if DebugSave:
                    predT1 = y_pred.reshape((xs[0], 1, xs[3], xs[4]))
                    refT1 = y.reshape((xs[0], 1, xs[3], xs[4]))
                    maskSave = w_mask.reshape((xs[0], 1, xs[3], xs[4]))
                    saveArrayToMat(predT1.cpu().data.numpy(), 'predT1',
                                   'predT1forDebug',
                                   params.tensorboard_dir)
                    saveArrayToMat(refT1.cpu().data.numpy(), 'refT1',
                                   'refT1forDebug',
                                   params.tensorboard_dir)
                    saveArrayToMat(maskSave.cpu().data.numpy(), 'mask',
                                   'maskforDebug',
                                   params.tensorboard_dir)

                LOSS.append(loss.cpu().data.numpy())

                l += loss.data

                optimizer.zero_grad()
                loss.backward()
                i += 1
                optimizer.step()
                # torch.nn.utils.clip_grad_norm_(net.parameters(),0.25)

                print('Epoch: {0} - {1:.3f}%'.format(epoch, 100 * (itt * params.batch_size) /tr_N)
                      + ' \tIter: ' + str(i)
                      + '\tLoss: {0:.6f}'.format(loss.data)
                      # + '\tInputLoss: {0:.6f}'.format(inloss.data[0])
                      )
                itt += 1
                is_best = 0

                #trained model is backed up every 50 iteration
                if i % 50 == 0:
                    save_checkpoint({'epoch': epoch, 'loss': LOSS, 'arch': 'recoNet_Model1', 'state_dict': net.state_dict(),
                                    'optimizer': optimizer.state_dict(), 'iteration': i,
                                    }, is_best, filename=params.model_save_dir + 'MODEL_EPOCH{}.pth'.format(epoch))

                # if True or params.tbVisualize:
                #     writer.add_scalar(TAG + '/' + 'avg_SME', l / itt, epoch)
                #     saveArrayToMat(LOSS, 'mse',
                #                    'mse_R{0}_Trial{1}'.format(str(params.Rate), params.trialNum), params.tensorboard_dir)

            avg_loss = params.batch_size * l / tr_N
            print('Total Loss : {0:.6f} \t Avg. Loss {1:.6f}'.format(l, avg_loss))

            save_checkpoint({'epoch': epoch, 'loss': LOSS, 'arch': 'recoNet_Model1', 'state_dict': net.state_dict(),
                             'optimizer': optimizer.state_dict(), 'iteration': i,
                             }, is_best, filename=params.model_save_dir + 'MODEL_EPOCH{}.pth'.format(epoch))

            writer.add_scalar(TAG + '/' + 'avg_SME', l / itt, epoch)
            saveArrayToMat(LOSS, 'mse',
                           'mse_R{0}_Trial{1}'.format(str(params.Rate), params.trialNum), params.tensorboard_dir)

            ##validation Code
            T1_5_avg = list()
            ref_T1_avg = list()
            pred_T1_5_avg = list()

            allMyoBPPixelsPredict = list()
            allMyoBPPixelsRef = list()

            meanT1Ref = list()
            meanT1Pre = list()

            meanMyoT1Ref = list()       #Reference
            meanMyoT1Pre = list()       #Predicted by Net
            meanBloodT1Ref = list()     #Reference
            meanBloodT1Pre = list()     #Predicted by Net
            val_N = val_t1w_TI.shape[0]
            val_lst = list(range(0, val_N))
            sl_id = list()
            bs = 10  ##
            save_PNG = False

            myoLossLst = list()
            bpLossLst = list()
            myLossTotal = 0
            bpLossTotal = 0
            TAG = 'Validation'

            with torch.no_grad():
                for idx in range(0, val_N, bs):
                    # for X, y, T1, TI, T1_5, mask, LVmask, sliceID in training_DG:
                    try:

                        X = Variable(torch.FloatTensor(val_t1w_TI[val_lst[idx:idx + bs]])).to('cuda:0')
                        y = Variable(torch.FloatTensor(val_T1[val_lst[idx:idx + bs]])).to('cuda:0')
                        T1_5 = Variable(torch.FloatTensor(val_T1[val_lst[idx:idx + bs]])).to('cuda:0')
                        # LVmask = Variable(torch.FloatTensor(tst_LVmask[val_lst[idx:idx + bs]])).to('cuda:0')
                        # ROImask = Variable(torch.FloatTensor(tst_ROImask[val_lst[idx:idx + bs]])).to('cuda:0')
                        # bloodmask = Variable(torch.FloatTensor(tst_mask[val_lst[idx:idx + bs]])).to('cuda:0')
                        # sliceID = tst_sliceID[np.array(val_lst[idx:idx + bs])].tolist()
                        LVBpmask = Variable(torch.FloatTensor(val_LVBpmask[val_lst[idx:idx + bs]])).to('cuda:0')

                        myomask = Variable(torch.FloatTensor(val_Myomask[val_lst[idx:idx + bs]])).to('cuda:0')
                        bpmask = Variable(torch.FloatTensor(val_BPmask[val_lst[idx:idx + bs]])).to('cuda:0')

                        # LVBpmask = tst_LVBpmask[val_lst[idx:idx + bs]]
                        # t_const = 1e6
                        # X = torch.cat(
                        #     (X[:, :, 0:5, :, :, 0] * t_const, X[:, :, 0:5, :, :, 1] * t_const, X[:, :, 5:, :, :, 0]), 2)
                        # X = torch.cat((magnitude(X[:, :, 0:5, :, :, :]) * t_const, X[:, :, 5:, :, :, 0]), 2)
                        xs = X.shape
                        X = X.permute((0, 3, 4, 1, 2)).reshape((xs[0] * xs[3] * xs[4], xs[1], xs[2]))
                        y = y.permute((0, 2, 3, 1)).reshape((xs[0] * xs[3] * xs[4], xs[1])).unsqueeze(-1)
                        # y = y.permute((0, 2, 3, 1)).reshape((xs[0] * xs[3] * xs[4], xs[1])).unsqueeze(-1)
                        # TI = TI.unsqueeze(1).permute((0,3,4,1,2)).reshape((xs[0]*xs[3]*xs[4],xs[1],xs[2]))
                        # X = torch.cat((X*t_const, TI), 2)
                        # w_mask = w_mask.permute((0, 2, 3, 1)).reshape((xs[0] * xs[3] * xs[4], xs[1])).unsqueeze(-1)
                        LVBpmask = LVBpmask.permute((0, 2, 3, 1)).reshape((xs[0] * xs[3] * xs[4], xs[1])).unsqueeze(-1)
                        myoMaskLoss = myomask.permute((0, 2, 3, 1)).reshape((xs[0] * xs[3] * xs[4], xs[1])).unsqueeze(-1)
                        bpMaskLoss = bpmask.permute((0, 2, 3, 1)).reshape((xs[0] * xs[3] * xs[4], xs[1])).unsqueeze(-1)

                        net.eval()
                        y_pred = net(X.to('cuda:0')).to('cuda:0')

                        pred_T1_5 = y_pred.reshape((xs[0], 1, xs[3], xs[4]))
                        ref_T1 = y.reshape((xs[0], 1, xs[3], xs[4]))
                        LVBpmask = LVBpmask.reshape((xs[0], 1, xs[3], xs[4]))
                        # allMyoBPPixelsPredict.append(y_pred[np.nonzero(LVBpmask.cpu().data.numpy())].cpu().data.numpy())
                        # allMyoBPPixelsRef.append(y[np.nonzero(LVBpmask.cpu().data.numpy())].cpu().data.numpy())




                        #loss regarding myocrdium and blood pool

                        if epoch > 4000:
                            losstmp = myoMaskLoss * torch.abs(y_pred - y)
                            myoMaskLoss[losstmp > 0.2] = 0
                            losstmp = bpMaskLoss * torch.abs(y_pred - y)
                            bpMaskLoss[losstmp > 0.2] = 0

                        if epoch > 4500:
                            losstmp = myoMaskLoss * torch.abs(y_pred - y)
                            myoMaskLoss[losstmp > 0.1] = 0
                            losstmp = bpMaskLoss * torch.abs(y_pred - y)
                            bpMaskLoss[losstmp > 0.1] = 0
                        # if epoch > 3000:
                        #     losstmp = myoMaskLoss * torch.abs(y_pred - y)
                        #     myoMaskLoss[losstmp > 0.2] = 0
                        #     losstmp = bpMaskLoss * torch.abs(y_pred - y)
                        #     bpMaskLoss[losstmp > 0.2] = 0

                        myoLossTmp = w_mae(y_pred, y, myoMaskLoss)
                        myoLossLst.append(myoLossTmp.cpu().data.numpy())
                        myLossTotal+=myoLossTmp.data*bs

                        bpLossTmp = w_mae(y_pred, y, bpMaskLoss)
                        bpLossLst.append(bpLossTmp.cpu().data.numpy())
                        bpLossTotal+=bpLossTmp.data*bs

                    except Exception as e:
                        traceback.print_exc()
                        continue
                avg_myo_loss = myLossTotal/val_N
                myoAvgLossAllEpochs.append(avg_myo_loss.cpu().data.numpy())
                print('Myo: Total Loss : {0:.6f} \t Avg. Loss {1:.6f}'.format(myLossTotal, avg_myo_loss))
                avg_bp_loss = bpLossTotal / val_N
                bpAvgLossAllEpochs.append(avg_bp_loss.cpu().data.numpy())
                print('Blood pool: Total Loss : {0:.6f} \t Avg. Loss {1:.6f}'.format(bpLossTotal, avg_bp_loss))

                tmpMyoBloodLoss =avg_myo_loss+avg_bp_loss
                if initialLoss > tmpMyoBloodLoss:
                    initialLoss = tmpMyoBloodLoss
                    bestModelEpochIx = epoch

                print('The best model is @ epoch: Total Loss : {0:.0f} \t with myo+blood Loss {1:.6f}'.format(bestModelEpochIx, initialLoss))

        except Exception as e:
            traceback.print_exc()
            # print(e)
            continue

        except KeyboardInterrupt:
            print('Interrupted')
            torch.save(MyoMapNet.state_dict(), 'MODEL_INTERRUPTED.pth')
            saveArrayToMat(np.array(myoAvgLossAllEpochs), 'myoAvgLossAllEpochs',
                           'avgLoss_Myo_allEpochs_R{0}_Trial{1}'.format(str(params.Rate), params.trialNum),
                           params.validation_dir)
            saveArrayToMat(np.array(bpAvgLossAllEpochs), 'bpAvgLossAllEpochs',
                           'avgLoss_BP_allEpochs_R{0}_Trial{1}'.format(str(params.Rate), params.trialNum),
                           params.validation_dir)
            writer.add_scalar(TAG + '/' + 'avg_SME', l / itt, epoch)
            saveArrayToMat(LOSS, 'mse',
                           'mse_R{0}_Trial{1}'.format(str(params.Rate), params.trialNum), params.tensorboard_dir)

            saveArrayToMat(bestModelEpochIx, 'BestEpochIx',
                           'BestEpochIx_R{0}_Trial{1}_BestModelAtEpoch_{2}'.format(str(params.Rate), params.trialNum,
                                                                                   bestModelEpochIx),
                           params.validation_dir)
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    saveArrayToMat(np.array(myoAvgLossAllEpochs), 'myoAvgLossAllEpochs','avgLoss_Myo_allEpochs_R{0}_Trial{1}'.format(str(params.Rate), params.trialNum),params.validation_dir)
    saveArrayToMat(np.array(bpAvgLossAllEpochs), 'bpAvgLossAllEpochs','avgLoss_BP_allEpochs_R{0}_Trial{1}'.format(str(params.Rate), params.trialNum),params.validation_dir)
    saveArrayToMat(LOSS, 'mse',
                   'mse_R{0}_Trial{1}'.format(str(params.Rate), params.trialNum), params.tensorboard_dir)
    saveArrayToMat(bestModelEpochIx, 'BestEpochIx',
                   'BestEpochIx_R{0}_Trial{1}_BestModelAtEpoch_{2}'.format(str(params.Rate), params.trialNum,bestModelEpochIx), params.validation_dir)

    writer.close()


mseCriterion = Loss.MSELoss()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print('Model Saved!')
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = params.args.lr * (0.1 ** (epoch // 1000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def mean_T1(x, mask):
    meant1 = list()
    for i in range(0, x.shape[0]):
        xs = x[i,]
        myo_T1 = xs[np.nonzero(mask[i,].cpu().data.numpy())].cpu().data.numpy()
        myo_T1 = myo_T1[myo_T1 > 1500]
        myo_T1 = myo_T1[myo_T1 < 2200]
        # meant1.append(myo_T1.std())
        meant1.append(myo_T1.mean())
    return meant1

def mean_std_ROI(x, mask):
    meanstdValsArr = [ [0 for i in range(x.shape[0])] for i in range(2)]
    for i in range(0, x.shape[0]):
        xs = x[i,]
        roiVals = xs[np.nonzero(mask[i,])].cpu().data.numpy()
        meanstdValsArr[0][i] = roiVals.mean()
        meanstdValsArr[1][i] = roiVals.std()
    return meanstdValsArr

def get_allPixels(x, mask):
    allPixles = list()
    for i in range(0, x.shape[0]):
        xs = x[i,]
        allpixlestmp = xs[np.nonzero(mask[i,].cpu().data.numpy())].cpu().data.numpy()
        # myo_T1 = myo_T1[myo_T1 > 1500]
        # myo_T1 = myo_T1[myo_T1 < 2200]
        # # meant1.append(myo_T1.std())
        allPixles.append(allpixlestmp)
    return allPixles

if __name__ == '__main__':
    try:
        train(MyoMapNet)
    except KeyboardInterrupt:
        print('Interrupted')
        torch.save(MyoMapNet.state_dict(), 'MODEL_INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
