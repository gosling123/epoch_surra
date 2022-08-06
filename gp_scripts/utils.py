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

c = constants.c
eps0 = constants.epsilon_0
me = constants.m_e
e = constants.e
kB = constants.k
keV_to_K = (e*1e3)/kB
mu0 = constants.mu_0
pi = np.pi
pico = 1e-12
micron = 1e-6
nano = 1e-9
J_tot_KeV = 6.242e+15









