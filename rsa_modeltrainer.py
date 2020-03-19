"""
RSA Model Trainer

Author:  Bruce Rushing
Date: 3/6/2020

Builds and trains several models on a generated data set.
"""

import os
import datetime
import numpy as np
import pickle
import random
import torch
import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from torch.utils.data import Dataset
from torchvision import transforms

#Constants
MAXSTEPS = 30

# Data processing
class distributionDataSet(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data_item = self.data[idx]
        sample = data_item[0]
        target = data_item[1]
        redundant = data_item[2]

        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)

        return sample, target, redundant

class toOneHot(object):
    def __init__(self, num_classes):
        self.numClasses = num_classes

    def __call__(self, integer):
        y_onehot = torch.zeros(self.numClasses)
        y_onehot[integer]=1
        return y_onehot

def import_data_func(fileName):
    """
    Imports and preprocesses data to be used for the neural network.

    Returns a train dataset object, a test dataset object, a coarse grained label dict and a fine grained label dict.
    """

    with open(filename, 'rb') as f:
        featuresLabels = pickle.load(f)

    # Build translation dictionaries
    oldToNew_fine = dict()
    oldToNew_coarse = dict()
    newToOld_fine = dict()
    newToOld_coarse = dict()
    count1 = 0
    count2 = 0
    for f in featuresLabels:
        if f[1] not in newToOld_fine.values():
            newToOld_fine[count1] = f[1]
            oldToNew_fine[f[1]] = count1
            count1 += 1

        if f[2] not in newToOld_coarse.values():
            newToOld_coarse[count2] = f[2]
            oldToNew_coarse[f[2]] = count2
            count2 += 1

    # Convert the labels to the new labels
    for f in featuresLabels:
        f[1] = oldToNew_fine[f[1]]
        f[2] = oldToNew_coarse[f[2]]

    labelsCoarse = newToOld_coarse
    labelsFine = newToOld_fine

    random.shuffle(featuresLabels)
    testSize = int(.1 * float(len(featuresLabels)))
    testD = np.array(featuresLabels[:testSize])
    trainD = np.array(featuresLabels[testSize:])

    #trainDataset = distributionDataSet(data=trainD, transform=torch.from_numpy, target_transform=toOneHot(num_classes=nClasses))
    #testDataset = distributionDataSet(data=testD, transform=torch.from_numpy, target_transform=toOneHot(num_classes=nClasses))

    trainDataset = distributionDataSet(data=trainD, transform=torch.from_numpy, target_transform=None)
    testDataset = distributionDataSet(data=testD, transform=torch.from_numpy, target_transform=None)

    return trainDataset, testDataset, labelsCoarse, labelsFine

# Models
"""
Several different models can be trained.  I examine two general types:  densely connected
and convolutional networks.

The simple network has just one hidden layer.  The densely connected network has 3 hidden layers,
while the convolutional networks have 3 convolutional layers and 2 densely connected layers.
"""
class SmoothStep(torch.autograd.Function):
    '''
    Modified from: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    '''

    @staticmethod
    def forward(aux, x):
        aux.save_for_backward(x)
        return (x >=0).float()

    def backward(aux, grad_output):
        # grad_input = grad_output.clone()
        input, = aux.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= -.5] = 0
        grad_input[input > .5] = 0
        return grad_input

class simpleNet(torch.nn.Module):
    def __init__(self, n_hidden, n_classes, name):
        super(simpleNet, self).__init__()
        self.cutlayer = torch.nn.Linear(512, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_classes)
        self.activation = torch.nn.Sigmoid()
        self.name = name

    def forward(self, data):
        y1 = self.activation(self.cutlayer(data))
        y2 = self.layer2(y1)
        return y2

class simpleNetReLU(torch.nn.Module):
    def __init__(self, n_hidden, n_classes, name):
        super(simpleNetReLU, self).__init__()
        self.cutlayer = torch.nn.Linear(512, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_classes)
        self.activation = torch.nn.ReLU()
        self.name = name

    def forward(self, data):
        y1 = self.activation(self.cutlayer(data))
        y2 = self.layer2(y1)
        return y2

class simpleNetSmoothStep(torch.nn.Module):
    def __init__(self, n_hidden, n_classes, name):
        super(simpleNetSmoothStep, self).__init__()
        self.cutlayer = torch.nn.Linear(512, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_classes)
        self.smoothstep = SmoothStep().apply
        self.name = name

    def forward(self, data):
        y1 = self.smoothstep(self.cutlayer(data))
        y2 = self.layer2(y1)
        return y2

class denselyConnectedNet(torch.nn.Module):
    def __init__(self, n_l1, n_l2, n_l3, n_classes, name):
        super(denselyConnectedNet, self).__init__()
        self.layer1 = torch.nn.Linear(512, n_l1)
        self.layer2 = torch.nn.Linear(n_l1, n_l2)
        self.layer3 = torch.nn.Linear(n_l2, n_l3)
        self.layer4 = torch.nn.Linear(n_l3, n_classes)
        self.activation = torch.nn.ReLU()
        self.name = name

    def forward(self, data):
        y1 = self.activation(self.layer1(data))
        y2 = self.activation(self.layer2(y1))
        y3 = self.activation(self.layer3(y2))
        y4 = self.layer4(y3)
        return y4

class denselyConnectedNetSmooth(torch.nn.Module):
    def __init__(self, n_l1, n_l2, n_l3, n_classes, name):
        super(denselyConnectedNetSmooth, self).__init__()
        self.layer1 = torch.nn.Linear(512, n_l1)
        self.layer2 = torch.nn.Linear(n_l1, n_l2)
        self.layer3 = torch.nn.Linear(n_l2, n_l3)
        self.layer4 = torch.nn.Linear(n_l3, n_classes)
        self.smoothstep = SmoothStep().apply
        self.name = name

    def forward(self, data):
        y1 = self.smoothstep(self.layer1(data))
        y2 = self.smoothstep(self.layer2(y1))
        y3 = self.smoothstep(self.layer3(y2))
        y4 = self.layer4(y3)
        return y4

# Cut Network models
class cut_simpleNetSmoothStep(torch.nn.Module):
    def __init__(self, cutlayer):
        super(cut_simpleNetSmoothStep, self).__init__()
        self.cutlayer = cutlayer
        self.smoothstep = SmoothStep().apply

    def forward(self, data):
        y1 = self.smoothstep(self.cutlayer(data))
        return y1

class cut_denselyConnectedNetSmooth(torch.nn.Module):
    def __init__(self, layer1, layer2, layer3):
        super(cut_denselyConnectedNetSmooth, self).__init__()
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.smoothstep = SmoothStep().apply

    def forward(self, data):
        y1 = self.smoothstep(self.layer1(data))
        y2 = self.smoothstep(self.layer2(y1))
        y3 = self.smoothstep(self.layer3(y2))
        return y3

# Training Functions
def train_step(x, t, net, opt_fn, loss_fn):
    y = net(x.float().cuda())
    loss = loss_fn(y, t.long().cuda())
    loss.backward()
    opt_fn.step()
    opt_fn.zero_grad()
    return loss

def train_network(train_data, test_data, net, epochs, coarse_grain=False):

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True)

    loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    acc_hist_train = []
    acc_hist_test = []
    for epoch in range(epochs):
        acc_batch = []
        net.train()
        net_c = net.cuda()
        for x,t,r in tqdm.tqdm(iter(train_loader)):
            if coarse_grain:
                loss_ = train_step(x, r, net_c, opt, loss_func)
                y = net_c(x.float().cuda()).cpu()
                acc_batch.append(torch.mean((r == y.argmax(1)).float()))
            else:
                loss_ = train_step(x, t, net_c, opt, loss_func)
                y = net_c(x.float().cuda()).cpu()
                acc_batch.append(torch.mean((t == y.argmax(1)).float()))
        acc_hist_train.append(torch.mean(torch.FloatTensor(acc_batch)))
        print(loss_)

        net.eval()
        net_c = net.cuda()
        acc_batch = []
        for x,t,r in iter(test_loader):
            y = net_c(x.float().cuda()).cpu()
            if coarse_grain:
                acc_batch.append(torch.mean((r == y.argmax(1)).float()))
            else:
                acc_batch.append(torch.mean((t == y.argmax(1)).float()))
        acc_hist_test.append(torch.mean(torch.FloatTensor(acc_batch)))

    return acc_hist_train, acc_hist_test

#Network processing functions
def cut_network_func(cut_net, comp_net, data):
    """
    Runs a cut network and returns the values on a sample of a data set.
    """

    cut_net.eval()
    comp_net.eval()

    test_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

    target_hist = [[],[],[],[]]

    for x,t,r in iter(test_loader):
        if t.numpy() not in target_hist[1]:
            y = cut_net(x.float())
            y_prime = comp_net(x.float())
            representation = y.clone().detach().numpy()
            answer = y_prime.argmax(1).numpy()
            target_hist[0].append(representation)
            target_hist[1].append(t.numpy())
            target_hist[2].append(r.numpy())
            target_hist[3].append(answer)

    return target_hist

def construct_rdm(input, renorm=False, euclid=False, abs=False, spearman=False):
    """
    Returns a complete RDM from a data set.

    Args are the input, a renorm boolean variable that determines whether the input is renormalized, a
    boolean variable that determines whether a euclidean distance is measure, and a boolean variable that
    determines whether absolute difference is measure.
    """

    def scale(X, x_min, x_max):
        nom = (X-X.min())*(x_max-x_min)
        denom = X.max() - X.min()
        denom = denom + (denom is 0)
        return x_min + nom/denom

    if renorm:
        data = [scale(v,-1,1) for v in input]
    else:
        data = input

    rdm = np.zeros([len(data), len(data)])

    for i in range(len(data)):
        for j in range(len(data)):
            if euclid and not (abs or spearman):
                e = np.linalg.norm(data[i]-data[j])
                rdm[i][j] = e
            elif abs and not (euclid or spearman):
                a = np.sum(np.absolute(data[i]-data[j]))
                rdm[i][j] = a
            elif spearman and not (euclid or abs):
                r, p = stats.spearmanr(data[i][0], data[j][0])
                rdm[i][j] = 1.0 - r
            else:
                r = np.corrcoef(data[i], data[j])
                rdm[i][j] = 1.0 - r[0][1]

    return rdm

def sort_hist_func(target_hist, labels, primes, coarse_grain=False):
    """
    Sorts history lists.  Returns four sorted lists based on whether a coarse or fine grain sort is needed.

    Args are the target_hist list that must be a 4 place list, a dictionary of labels, a list of primes, and
    a coarse_grain flag set to false by default.

    The sorted lists returned are a sorted fine grain of the activation values, a sorted coarse grain of the
    average of the activation values, a sorted target list, and a sorted answer list provided by the full network.
    """
    x = target_hist[0]
    z = target_hist[3]

    if coarse_grain:
        y = target_hist[2]
    else:
        y = target_hist[1]

    y_code = [labels[v[0]] for v in y]
    z_code = [labels[v[0]] for v in z]

    x_sorted = []
    y_sorted = []
    z_sorted = []
    xavg_sorted = []

    #Provide converted values and sort them.
    for p in primes:
        p_values = []
        for i in range(len(y_code)):
            if (y_code[i] % p) == 0:
                x_sorted.append(x[i])
                p_values.append(x[i])
                y_sorted.append(y_code[i])
                z_sorted.append(z_code[i])

        x_avg = np.zeros([1,np.size(x[0])])
        for v in p_values:
            x_avg = x_avg + v

        x_avg = np.true_divide(x_avg, len(p_values))
        xavg_sorted.append(x_avg)

    return x_sorted, xavg_sorted, y_sorted, z_sorted

def run_experiments_func(train_d, test_d, labels, grain):
    """
    Runs the main body of experiments.

    The code is broken into three parts:  1) generating data storage paths and processing data into coarse grain
    labels, 2) constructing networks, 3) the main loop with running the networks through a training loop, then cutting them,
    generating rdms, and storing the data.
    """

    #Create the save file directory
    direct_path = os.path.dirname(os.path.realpath(__file__))
    now = datetime.datetime.now()
    if grain:
        dir_str = "RSA_Run_coarse" + now.strftime("%Y-%m-%d_%H-%M-%S")
    else:
        dir_str = "RSA_Run_fine" + now.strftime("%Y-%m-%d_%H-%M-%S")
    dir = os.path.join(direct_path, dir_str)
    if not os.path.exists(dir):
        os.mkdir(dir)
    os.chdir(dir)

    nClasses = len(labels)
    #Get the primes
    label_set = set([labels[k] for k in labels])
    primes = set([labels[k] for k in labels])
    for l in label_set:
        sub_set = set()
        for l_ in label_set:
            if (l_ % l) == 0 and l != l_:
                sub_set.add(l_)
        primes = primes - sub_set
    primes = list(primes)

    #Construct networks
    net1 = simpleNet(64, nClasses, "simple_net_sigmoid")
    net2 = simpleNetReLU(64, nClasses, "simple_net_relu")
    net3 = simpleNetSmoothStep(64, nClasses, "simple_net_smoothstep")
    net4 = denselyConnectedNet(256, 128, 64, nClasses, "dense_net_relu")
    net5 = denselyConnectedNetSmooth(256, 128, 64, nClasses, "dense_net_smoothstep")

    networks = [net1, net2, net3, net4, net5]
    #Main loop
    for net in networks:
        #Train the network
        histTrain, histTest = train_network(train_d, test_d, net, MAXSTEPS, coarse_grain=grain)
        hist_x_axis = [i + 1 for i in range(len(histTest))]
        plt.figure()
        plt.plot(hist_x_axis, histTest)
        plt.savefig('hist_test_' + net.name + '.png')
        hist_x_axis = [i + 1 for i in range(len(histTrain))]
        plt.figure()
        plt.plot(hist_x_axis, histTrain)
        plt.savefig('hist_train_' + net.name + '.png')

        #Create the cut down network and format the resulting data
        if "smoothstep" not in net.name:
            if "simple" in net.name:
                #For all non-smooth step, simple networks
                cut_network = torch.nn.Sequential(
                        net.cutlayer,
                        net.activation)
            else:
                #For any non-smooth step, dense network
                cut_network = torch.nn.Sequential(
                            net.layer1,
                            net.activation,
                            net.layer2,
                            net.activation,
                            net.layer3,
                            net.activation)
        else:
            if "simple" in net.name:
                #For simple, smooth step networks
                cut_network = cut_simpleNetSmoothStep(net.cutlayer)
            else:
                #For dense, smooth step networks
                cut_network = cut_denselyConnectedNetSmooth(net.layer1, net.layer2, net.layer3)

        net.cpu()
        target_hist = cut_network_func(cut_network, net, test_d)
        x_sorted, xavg_sorted, y_sorted, z_sorted = sort_hist_func(target_hist, labels, primes, coarse_grain=grain)

        #Construct the RDMs and save them
        x_data = [x_sorted, xavg_sorted]
        data_labels = ['_fine_grain', '_coarse_grain']
        for i in range(len(x_data)):
            rdm = construct_rdm(x_data[i])
            plt.figure()
            plt.imshow(rdm)
            plt.savefig(net.name + data_labels[i] + '_correlation.png')

            rdm = construct_rdm(x_data[i], euclid=True)
            plt.figure()
            plt.imshow(rdm)
            plt.savefig(net.name + data_labels[i] + '_euclid.png')

            rdm = construct_rdm(x_data[i], abs=True)
            plt.figure()
            plt.imshow(rdm)
            plt.savefig(net.name + data_labels[i] + '_absolute.png')

            rdm = construct_rdm(x_data[i], spearman=True)
            plt.figure()
            plt.imshow(rdm)
            plt.savefig(net.name + data_labels[i] + '_spearman.png')

            rdm = construct_rdm(x_data[i], renorm=True)
            plt.figure()
            plt.imshow(rdm)
            plt.savefig(net.name + data_labels[i] + '_correlation_renorm.png')

            rdm = construct_rdm(x_data[i], renorm=True, euclid=True)
            plt.figure()
            plt.imshow(rdm)
            plt.savefig(net.name + data_labels[i] + '_euclid_renorm.png')

            rdm = construct_rdm(x_data[i], renorm=True, abs=True)
            plt.figure()
            plt.imshow(rdm)
            plt.savefig(net.name + data_labels[i] + '_absolute_renorm.png')

            rdm = construct_rdm(x_data[i], renorm=True, spearman=True)
            plt.figure()
            plt.imshow(rdm)
            plt.savefig(net.name + data_labels[i] + '_spearman_renorm.png')

        #Save the model
        torch.save(net.state_dict(), net.name + '.torch')

if __name__ == "__main__":

    direct_path = os.path.dirname(os.path.realpath(__file__))

    #Process the data.
    filename = "distributionData29-02-2020-11-22-11.pickle"

    trainData, testData, labelsCoarse, labelsFine = import_data_func(filename)

    #Do Coarse Grain Training
    run_experiments_func(trainData, testData, labelsCoarse, True)

    #Do Fine Grain Training
    os.chdir(direct_path)
    run_experiments_func(trainData, testData, labelsFine, False)
