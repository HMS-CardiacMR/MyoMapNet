import torch
import random
from torch.optim import Adam, SGD
from torch.nn.modules.loss import L1Loss, MSELoss
from torch.autograd import Variable
from DataLoader import *
from Architectures import DenseNN, save_models
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from sys import exit


nb_epochs = 100
batch_size = 64
early_stopping = 70
learning_rate = 0.001
name = "name_of_the_network"

seed_num = 888
torch.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)
random.seed(seed_num)
np.random.seed(seed_num)
cuda_available = torch.cuda.is_available()
net = DenseNN(8, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=0.0001)


# Loading training and validation dataset
print("Loading data for training")
x_train, y_train = load_training_data("path to training dataset")
x_train = shuffle(x_train, random_state=seed_num)

y_train = shuffle(y_train, random_state=seed_num)
y_train = y_train * 1000
y_train = np.where(y_train<0, 0, y_train)
y_train = np.where(y_train>2000, 2000, y_train)
print("x train shape: ", x_train.shape)
print("y train shape: ", y_train.shape)
print(y_train.min(), y_train.max())

print("Loading data for validation")
x_validation, y_validation, bpMask, myoMask, myobpMask = load_validation_data("path to validation dataset")
y_validation = y_validation * 1000
y_validation = np.where(y_validation<0, 0, y_validation)
y_validation = np.where(y_validation>2000, 2000, y_validation)
print("x validation shape: ", x_validation.shape)
print("y validation shape: ", y_validation.shape)

total_size_training_data = x_train.shape[0]
total_size_validation_data = x_validation.shape[0]


#loss_nn = MSELoss()
loss_nn = L1Loss()
best_training_loss = 0
no_loss_improvement = 0
global_training_loss_list = []
training_loss_list_myo = []
training_loss_list_bloodB = []


global_validation_loss_list = []
validation_loss_list_myo = []
validation_loss_list_bloodB = []
validation_loss_ranges_list_bloodB = []
validation_loss_ranges_list_myo = []

epoch_numbers = []

print("Start training")

for epoch in range(1, nb_epochs+1):
    epoch_numbers.append(epoch)

    training_loss = 0



    nb_batches = 0
    for index in tqdm(range(0, total_size_training_data, batch_size)):
        x = Variable(torch.FloatTensor(x_train[index: index+batch_size])).to("cuda:0")
        y = Variable(torch.FloatTensor(y_train[index: index + batch_size])).to("cuda:0")

        x = torch.reshape(x, (x.shape[0], 160, 160, 8))
        y = torch.reshape(y, (y.shape[0],  160, 160))
        optimizer.zero_grad()
        y_pred = net(x)


        y_pred = torch.reshape(y_pred, (y_pred.shape[0], 160, 160))
        loss = loss_nn(y_pred, y)

        loss.backward()
        optimizer.step()


        training_loss += loss.cpu().data

        y_pred = y_pred.cpu().data.numpy()
        y = y.cpu().data.numpy()

       # Blood pool for training
        y_pred_train_bp = np.where(y_pred<1500, 1500, y_pred)
        y_pred_train_bp = np.where(y_pred_train_bp > 2000, 2000, y_pred_train_bp)
        y_pred_train_bp = Variable(torch.FloatTensor(y_pred_train_bp))
        y_bp = np.where(y<1500, 1500, y)
        y_bp = np.where(y_bp > 2000, 2000, y_bp)
        y_bp = Variable(torch.FloatTensor(y_bp))
        loss_train_bp = loss_nn(y_pred_train_bp, y_bp)
        training_loss_bp += loss_train_bp.cpu().data


        #Myocardium for training
        y_pred_train_myo = np.where(y_pred<1000, 1000, y_pred)
        y_pred_train_myo = np.where(y_pred_train_myo > 1400, 1400, y_pred_train_myo)
        y_pred_train_myo = Variable(torch.FloatTensor(y_pred_train_myo))
        y_myo = np.where(y<1000, 1000, y)
        y_myo = np.where(y_myo > 1400, 1400, y_myo)
        y_myo = Variable(torch.FloatTensor(y_myo))
        loss_train_myo = loss_nn(y_pred_train_myo, y_myo)
        training_loss_myo += loss_train_myo.cpu().data

        nb_batches += 1

    # print("index for training: {}".format(nb_batches))
    avg_training_loss = training_loss / nb_batches
    no_loss_improvement += 1
    global_training_loss_list.append(np.sqrt(avg_training_loss))

    avg_training_loss_bp = training_loss_bp / nb_batches
    training_loss_list_bloodB.append(np.sqrt(avg_training_loss_bp))
    
    avg_training_loss_myo = training_loss_myo / nb_batches
    training_loss_list_myo.append(np.sqrt(avg_training_loss_myo))



    #Save the model based on validation results
    validation_loss = 0
    validation_loss_bp = 0
    validation_loss_myo = 0

    nb_batches = 0
    with torch.no_grad():

        for index in tqdm(range(0, x_validation.shape[0], batch_size)):
            x_val = Variable(torch.FloatTensor(x_validation[index: index + batch_size])).to("cuda:0")
            # Add a for loop if you want to visualize each image in the batch_size
            y_val = Variable(torch.FloatTensor(y_validation[index: index + batch_size])).to("cuda:0")
            y_val = torch.reshape(y_val, (y_val.shape[0], 160 * 160))

            y_pred_val = net(x_val)
            y_pred_val = torch.reshape(y_pred_val, (y_pred_val.shape[0], 160 * 160))
            loss_val = loss_nn(y_pred_val, y_val)
            validation_loss += loss_val.cpu().data

            y_val = torch.reshape(y_val, (y_val.shape[0], 160 , 160))
            y_pred_val = torch.reshape(y_pred_val, (y_pred_val.shape[0], 160 , 160))


            #Blood pool for validation
            y_pred_val_bp = Variable(torch.FloatTensor(y_pred_val.cpu().data.numpy() * bpMask[index: index + batch_size]))
            y_val_bp = Variable(torch.FloatTensor(y_val.cpu().data.numpy() * bpMask[index: index + batch_size]))
            loss_val_bp = loss_nn(y_pred_val_bp, y_val_bp)
            validation_loss_bp += loss_val_bp.cpu().data
            print("validation_loss_bp: {} ".format(validation_loss_bp))

            y_pred_val = y_pred_val.cpu().data.numpy()
            y_val = y_val.cpu().data.numpy()
            


            #Myocardium for validation
            y_pred_val_myo = Variable(torch.FloatTensor(y_pred_val.cpu().data.numpy() * myoMask[index: index + batch_size]))
            y_val_myo = Variable(torch.FloatTensor(y_val.cpu().data.numpy() * myoMask[index: index + batch_size]))
            loss_val_myo = loss_nn(y_val_myo, y_pred_val_myo)
            validation_loss_myo += loss_val_myo.cpu().data
            
            
            nb_batches += 1

    # print("index for validation: {}".format(nb_batches))
    avg_validation_loss = validation_loss / nb_batches


    print("Epoch {} - training loss: {} Validation loss: {}".format(epoch, avg_training_loss, avg_validation_loss))

    avg_validation_loss_bp = validation_loss_bp / nb_batches
    validation_loss_list_bloodB.append(np.sqrt(avg_validation_loss_bp))
    print("Average validation blood pool: {}".format(avg_validation_loss_bp))

    avg_validation_loss_myo = validation_loss_myo / nb_batches
    validation_loss_list_myo.append(np.sqrt(avg_validation_loss_myo))
    print(avg_validation_loss_myo)


 
    if (epoch == 1 or avg_validation_loss < best_validation_loss):
        save_models(net, name, epoch)
        best_validation_loss = avg_validation_loss
        no_loss_improvement = 0

    else:
        print("Loss did not improve from {}".format(best_validation_loss))
    if(no_loss_improvement == early_stopping):
        print("Early stopping - no improvement afer {} iterations of training".format(early_stopping))
        break

# # Curve for training and validation
plt.plot(epoch_numbers, global_training_loss_list)
plt.plot(epoch_numbers, global_validation_loss_list)
plt.xlabel("epoch")
plt.ylabel("loss (ms)")
plt.legend(["training", "validation"], loc="upper right")
plt.savefig("path to save")
plt.close()

# #Curve for training
plt.title("Training")
plt.plot(epoch_numbers, training_loss_list_myo, "r")
plt.plot(epoch_numbers, training_loss_list_bloodB, "g")
plt.xlabel("epoch")
plt.ylabel("loss (ms)")
plt.legend(["T1 ranges[1000-1400]", "T1 ranges[1500-2000]"], loc="upper right")
plt.savefig("path to save")
plt.close()

# # Curve for validation
plt.title("validation")
plt.plot(epoch_numbers, validation_loss_list_myo, "m")
plt.plot(epoch_numbers, validation_loss_list_bloodB, "k")
plt.xlabel("epoch")
plt.ylabel("loss (ms)")
plt.legend(["Myocardium", "Blood Pool"], loc="upper right")
plt.savefig("path to save")
plt.close()



