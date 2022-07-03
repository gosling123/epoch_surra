## @package utils
# Documentation for utils module
#
# The utils module houses functions which are used to either read data,
# run simulations or just mathematical definitions that aren't relate to 
# plasma physics.

import fileinput
import numpy as np
import sys
import os
from csv import writer
import sdf
from epoch_calculator import *

## expav
#
# Calculates exponential moving average (EMA) mu_i = exp(-1/T)mu_i-1 + (1-exp(-1/T))*x_i 
# @param a : Array
# @param t : Period/Lengthscale of EMA
def expav(a,t):
    f1 = np.exp(-1.0/t)
    f2 = 1.0 - f1
    
    res = np.zeros_like(a)
    res[0] = a[0]
    for i in range(1,len(a)):
        res[i] = f1*res[i-1] + f2*a[i]
    return res

## read_intensity
#
# Read intensity in W/cm2 from input.deck
# @param dir : Directory which holds input.deck file you want to read (str)
def read_intensity(dir):
    
    line = []
    with open(dir+'input.deck') as f:
        found = False
        while not found:
            line = f.readline()
            words = line.split()

            if len(words) < 1:
                continue

            if words[0] == "intensity_w_cm2":
                found = True
                return float(words[2])

## replace_line
#
# Function rewrite line in input.deck via python
# @param line_in : original line in input.deck 
# @param line_out : repacement of line_in in input.deck
def replace_line(line_in, line_out, fname):
  finput = fileinput.input(fname, inplace=1)
  for i, line in enumerate(finput):
    sys.stdout.write(line.replace(line_in, line_out))
  finput.close()

## run_epoch
#
# Runs epoch1d simulations for set intensity 
# @param intensity : Intensity to write in input.deck 
# @param data_dir : Directory to store epoch data to and where the input.deck file is
# @param output : Ouput to command line (True) or to run.log file (False)
# @param np : Number of processors to eun epoch1d on (MPI)
def run_epoch(intensity, data_dir = 'Data', output = False, np = 10):
    dir = os.getenv('EPOCH1D')
    try:
        os.mkdir(data_dir)
    except:
        os.system('rm '+str(data_dir)+'/*sdf')
    os.system(f'cp {dir}/input.deck ' + str(data_dir)+'/input.deck')
    replace_line('intensity_w_cm2 = 2.0e15', f'intensity_w_cm2 = {intensity}', fname = str(data_dir)+'/input.deck')
    if output:
        os.system(f'echo ' + str(data_dir) + ' | mpiexec -n ' + str(np) + ' ./bin/epoch1d')
    else:
        os.system(f'echo ' + str(data_dir) + ' | mpiexec -n ' + str(np) + ' ./bin/epoch1d > run.log')
        
## get_I_SRS_res
#
# Runs epoch1d simulations for changing intensity and ouptputs backsactter SRS intensity
# @param I_array : Intensity array (list of data to sim) 
# @param dir : Directory to store epoch data to and where the input.deck file is
# @param np : Number of processors to eun epoch1d on (MPI)        
def get_I_SRS_res(I_array, dir, np = 10):
    I_SRS_data = []
    fname = 'I_SRS_Data.npy'
    for I in I_array:
        print('########################')
        print('Strting I = ', I/1e15, ' 10^15 W/cm^2')
       
        run_epoch(I, data_dir = dir, output = False, np = np)
        epoch_fields = EM_fields(dir = dir)
        res = epoch_fields.get_flux_grid_av(ncells = 50, laser = False)
        
        I_SRS_data.append(res)
        
    print('Writing Data to file I_SRS_Data.npy')   
    np.save(fname, I_SRS_data)
       
        
## append_list_as_row
#
# Append data to csv file (for appending I and I_SRS result)
# @param file_name : file name of csv file 
# @param list_of_elem : list to write to csv file 
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

