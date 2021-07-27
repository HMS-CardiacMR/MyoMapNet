from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from FCMyoMapNet import UNet
import torch
from torch.autograd import Variable
import numpy as np


TimeScaling = 1000;
TimeScalingFactor =1/TimeScaling
T1sigNum = 4
T1sigAndTi = T1sigNum*2;

# Select one model
modelName = "MyoMapNet_4PreandPostGd" #MyoMapNet_4PostGd; MyoMapNet_4PreandPostGd; MyoMapNet_4PreGd; MyoMapNet_5PreGd;
if modelName=="MyoMapNet_5PreGd":
    T1sigNum = 5
else:
    T1sigNum = 4
T1sigAndTi = T1sigNum*2

# Construct Model
MyoMapNet = UNet(T1sigAndTi, 1)
MyoMapNet.to(torch.device('cpu'))

#loading trained model
try:
    model = torch.load( 'TrainedModels/' +modelName+'.pth', map_location=torch.device('cpu'))
    MyoMapNet = torch.nn.DataParallel(MyoMapNet)
    MyoMapNet.load_state_dict(model['state_dict'])
    print('Model loaded!')
except Exception as e:
    print('Can not load model!')
    print(e)

print('Start loading demo data')
if T1sigNum == 4:
    t1wtiIdx = [0, 1, 2, 3, 5, 6, 7, 8]     #For MyoMapNet 4PreGd, 4Pre+PostGd, 4PostGd
else:
    t1wtiIdx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  #Only for MyoMapNet 5PreGd

try:
    data = loadmat("Data/Demo/demo_Phantom.mat")
    Pre5HBsT1wTIs_in = data['MOLLIT1wTI']  #get T1 weighted signals and corrsponding inversion time
    Pre5HBsT1wTIs_double = Pre5HBsT1wTIs_in.astype(np.double)
    Pre5HBs_tst_t1w_TI = Pre5HBsT1wTIs_double[:,:,t1wtiIdx,:,:]
    PreMOLLIT1MapOffLine_in = data['MOLLIoffLineT1Map']
    PreMOLLIT1MapOffLine_double = PreMOLLIT1MapOffLine_in.astype(np.double)
    PreMOLLIT1MapOffLineT1 = np.zeros((PreMOLLIT1MapOffLine_double.shape[0], 1, PreMOLLIT1MapOffLine_double.shape[1],
                                       PreMOLLIT1MapOffLine_double.shape[2]))
    PreMOLLIT1MapOffLineT1[0, 0, :, :] = PreMOLLIT1MapOffLine_double = PreMOLLIT1MapOffLine_in.astype(np.double)[
                                                                            0, :, :]
except Exception as e:
    print(e)

#Construct input signals and output T1 maps
X = Variable(torch.FloatTensor(Pre5HBs_tst_t1w_TI))
xs = X.shape
X = X.permute((0, 3, 4, 1, 2)).reshape((xs[0] * xs[3] * xs[4], xs[1], xs[2]))
MyoMapNet.eval()
y = MyoMapNet(X)
MyoMapNetT1 = y.reshape((xs[0], 1, xs[3], xs[4]))

print('Displaying T1 maps')
fig, axs = plt.subplots(1, 3)
axs[0].set_title('MyoMapNet')
axs[0].imshow(MyoMapNetT1[0, 0, :, :].data.numpy() * TimeScaling, cmap='jet',
              vmin=0, vmax=1500)
axs[0].axis('off')

axs[1].set_title('MOLLI5(3)3')
axs[1].imshow(PreMOLLIT1MapOffLineT1[0, 0, :, :] * TimeScaling, cmap='jet', vmin=0,
              vmax=1500)
axs[1].axis('off')

axs[2].set_title('MyoMapNet-MOLLI5(3)3')
axs[2].imshow((MyoMapNetT1[0, 0, :, :].data.numpy()-PreMOLLIT1MapOffLineT1[0, 0, :, :]) * TimeScaling, cmap='jet', vmin=-50,
              vmax=50)
axs[2].axis('off')
fig.show()