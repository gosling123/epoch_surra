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


## read_input
#
# Read inputs in input.deck in chosen directory
# @param dir  Directory which holds input.deck file you want to read (str)
# @param param  Specific paramter to read from input file ('intensity', 'momentum', 'ppc' or 'ne_scale_len')
def read_input(dir, param):
    
    line = []
    with open(dir+'input.deck') as f:
            found = False
            while not found:
                line = f.readline()
                words = line.split()
                if len(words) < 1:
                    continue

                if param == 'intensity':
                    if words[0] == "intensity":
                        found = True
                        return float(words[2])/1e4

                elif param == 'momentum':
                    if words[0] == "range1":
                        found = True
                        return float(words[2][1:-1]), float(words[3][:-1])
            

                elif param == 'ppc':
                    if words[0] == "PPC":
                        found = True
                        return float(words[2])

                elif param == 'ne_scale_len':
                    if words[0] == "Ln":
                        found = True
                        return float(words[2])

                else:
                    print('Please set param to one of the following as a string \
                           intensity, momentum, ppc or ne_scale_len')
                    break




## loss_func
#
# Calculates loss function to compare fits
# @param fit  Fit data
# @param data  Sim data
def loss_func(fit, data):
    N = len(data)
    sum_ = 0
    for i in range(N):
        l = fit[i] - data[i]
        sum_ += l*l
    
    loss = sum_/N

    return loss


## moving_av
#
# Finds movng average of an array using scipys uniform_filter1d function
# @param Q  Data array 
# @param span  Length of data 
# @param period  Period to average over
def moving_av(Q, span, period = 10):
    return uniform_filter1d(Q, size = span // period)



## replace_line
#
# Function rewrite line in input.deck via python
# @param line_in  Original line in input.deck 
# @param line_out  Repacement of line_in in input.deck
def replace_line(line_in, line_out, fname):
  finput = fileinput.input(fname, inplace=1)
  for i, line in enumerate(finput):
    sys.stdout.write(line.replace(line_in, line_out))
  finput.close()


## append_list_as_row
#
# Append data to csv file (for appending I and I_SRS result)
# @param file_name  File name of csv file 
# @param list_of_elem  List to write to csv file 
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)