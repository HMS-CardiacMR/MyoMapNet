from Architectures import DenseNN
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat



def plot(img, title, vmin, vmax):
    """
    
    """
    plt.title(title)
    plt.imshow(img, cmap="jet", vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.colorbar()
    plt.savefig("Data visualization/" + title)
    plt.show()

def subplot(images, name):

    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 2
    j = 0

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i-1], cmap="gray")
        i += 1
        j += 1
    plt.savefig("Data visualization/"+name)
    plt.show()

print("Loading the model")
net = DenseNN(8, 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
net.load_state_dict(torch.load("model_save_dir/name_of_the_model.model"))

print("loading test data")
test_data = loadmat("path to test dataset")
x_test = np.array(test_data["Pre5HBsT1wTIs"])


subplot(x_test[0,:,:,:], "preTest4HBs.png")

y_test = np.array(test_data["PreMOLLIT1MapOnLine"])
y_test = y_test * 1000
plot(y_test[0,:,:], "PreMOLLIT1MapOffLine1", vmin=0, vmax=2000)
plot(y_test[1,:,:], "PreMOLLIT1MapOffLine2", vmin=0, vmax=2000)

x_test = x_test.transpose(0, 2, 3, 1)
y_test = y_test[0,:,:]
y_test = y_test.reshape(1,160,160)
print("x test shape: ", x_test.shape)
print("y test shape: ", y_test.shape)



y_test = np.where(y_test<0, 0, y_test)
y_test = np.where(y_test>2000, 2000, y_test)

batch_size = 32
with torch.no_grad():

    for index in range(0, y_test.shape[0], batch_size):
        x = Variable(torch.FloatTensor(x_test[index: index+batch_size])).to("cuda:0")
        # Add a for loop if you want to visualize each image in the batch_size
        y = y_test[index: index+batch_size]
        print("y truth: ", y.min(), y.max())
        print(x.shape)
        y_pred = net(x)
        y_pred = y_pred.cpu().data.numpy()
        y_pred = y_pred * 1000
        print("ypred: ", y_pred.min(), y_pred.max())
        print(y_pred.shape)
        y_pred = np.where(y_pred < 0, 0, y_pred)
        y_pred = np.where(y_pred > 2000, 2000, y_pred)
        plot(y[0, :, :], "ground truth", 0, 2000)

        show_img_pred = y_pred
        print(show_img_pred.min(), show_img_pred.max())
        show_img_pred = show_img_pred.reshape(show_img_pred.shape[0], 160, 160)
        plot(show_img_pred[0, :, :], "prediction", 0, 2000)

        difference = show_img_pred - y
        plot(difference[0, :, :], "difference", -150, 150)




