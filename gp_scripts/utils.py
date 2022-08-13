from typing import ForwardRef
import fileinput
import os
import sys
from nbformat import read
import numpy as np
import sdf
from scipy import constants
from scipy.optimize import brentq
import glob
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
import csv
import json
import time
import GPy
from sklearn.model_selection import train_test_split
import seaborn as sns
from smt.sampling_methods import LHS
import pickle

plt.rcParams["figure.figsize"] = [14, 10]
plt.rcParams["figure.autolayout"] = True
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
# Constants



def scale_axis(array, scale_array, rescale = False):
    if rescale:
        array = array*(scale_array.max() -scale_array.min()) + scale_array.min()
        return array
    else:
        array = (array - scale_array.min())/(scale_array.max() - scale_array.min())
        return array


def read_json_file(fname):
    with open(fname, 'r') as f:
            data = json.load(f)
    return np.array(data)


def read_pickle_file(fname):
    with open(fname, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict
