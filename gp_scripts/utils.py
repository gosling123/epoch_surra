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


plt.rcParams["figure.figsize"] = (20,3)
plt.rcParams["figure.figsize"] = [15, 15]
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


def expav(a,t):
    f1 = np.exp(-1.0/t)
    f2 = 1.0 - f1
    
    res = np.zeros_like(a)
    res[0] = a[0]
    for i in range(1,len(a)):
        res[i] = f1*res[i-1] + f2*a[i]
    return res