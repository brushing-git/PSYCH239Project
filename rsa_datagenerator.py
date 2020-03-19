"""
RSA Data Generator

Author:  Bruce Rushing
Date: 2/29/2020

This programs builds and saves a data set to be used for training and testing of several algorithms in the
Representational Similarity Analysis Project.

The data consists of numpy arrays and a corresponding dictionary for the labels of the numpy array.

The data is saved as a pickle.
"""

import random
import numpy as np
from datetime import datetime
import pickle

# Constants
SAMPLE_SIZE = 512
TOTAL_SAMPLES = 10000
OFFSET = 0.5

"""
Data Set

The goal of the network is to correctly identify the a set of random floats as belonging to a distribution.

Each distribution has a set of parameters and a functional form.  The following distributions are utilized:

Gauss
Betavariate
Gamma

Each distribution has parameters that are generated to produce 100 different examples of each distribution.

Each generated example has a label.  The labels are numbers that correspond to a dictionary entry that has
as a value a distribution object.
"""

class distribution:
    """
    This object is just for keeping track of a data label's properties.
    Those properties are the name and the parameters.
    """

    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters

def generate_nextPrime_func(n):
    """
    Returns the next prime after n.

    Limit up to 100000000.
    """

    for val in range(n, 100000000):
        for i in range(1, val + 1):
            if val % i == 0 and i != 1 and i < val:
                break
            elif i == val and i != 1 and i != n:
                return val


def generate_distribution_func(distribution, parameters, label, label_base):
    """
    Generates examples for training and testing.  Returns a list of data for the input distribution.

    The generated distributions are all offset by OFFSET.  This allows there to be negative numbers for beta and gamma
    distributions.

    The args are the distribution function, the parameters, and the label.

    Each example is a list with a label number.

    The examples have the data, along with two entries for the same label.  This is important for data retrieval.

    Returns a list.
    """

    data = []

    for i in range(TOTAL_SAMPLES):
        data.append(np.asarray([np.asarray([distribution(*parameters) - OFFSET for i in range(SAMPLE_SIZE)]), label, label_base]))

    return data

def build_data_set():
    """
    The data sets are generated with specific parameters, whose label is a power of a prime.  The prime indicates
    the underlying functional form.
    """

    def create_distribution_data_func(name, labelBase, distFunc, parFunc, nPar):

        distData = []
        distDict = dict()

        for p in range(1, nPar):
            label = labelBase ** p
            parameters = parFunc(p)
            distData += generate_distribution_func(distFunc, parameters, label, labelBase)
            distDict[label] = distribution(name, parameters)

        return distData, distDict

    gaussParFunc = lambda x: [1 / (float(x) + 1), 1 / ((float(x) + 2)**2)]
    betaParFunc = lambda x: [0.5 + float(x), 0.5 + (float(x) * 0.5)]
    gammaParFunc = lambda x: [1.0 + float(x), 0.1 * float(x)]

    nParameters = 5

    parFuncs = [gaussParFunc, betaParFunc, gammaParFunc]
    distFuncs = [random.gauss, random.betavariate, random.gammavariate]
    distNames = ["gauss", "beta", "gamma"]

    label = 1

    data = []
    dataDict = dict()

    for i in range(len(distFuncs)):
        label = generate_nextPrime_func(label)
        d, dDict = create_distribution_data_func(distNames[i], label, distFuncs[i], parFuncs[i], nParameters)
        data += d
        dataDict.update(dDict)

    return data, dataDict

if __name__ == "__main__":

    data, dataDict = build_data_set()

    now = datetime.now()
    filename1 = "distributionData" + now.strftime("%d-%m-%Y-%H-%M-%S") + ".pickle"
    filename2 = "distributionDict" + now.strftime("%d-%m-%Y-%H-%M-%S") + ".pickle"

    with open(filename1, 'wb') as f:
        pickle.dump(data, f)

    with open(filename2, 'wb') as f:
        pickle.dump(dataDict, f)
