#!/usr/bin/python3

import fileinput
import numpy as np
import sys
import os
from csv import writer

# exponential  moving average mu_i = exp(-1/T)mu_i-1 + (1-exp(-1/T))*x_i 
def expav(a,t):
    f1 = np.exp(-1.0/t)
    f2 = 1.0 - f1
    
    res = np.zeros_like(a)
    res[0] = a[0]
    for i in range(1,len(a)):
        res[i] = f1*res[i-1] + f2*a[i]
    return res

    # Read intensity in W/cm2 from input deck
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


#************************************************************************
#> replace_line
##
## Function rewrite line in input.deck via python 
##
##@param line_in : original line in input.deck 
##       line_out : repacement of line_in in input.deck
#************************************************************************
#

def replace_line(line_in, line_out, fname):
  finput = fileinput.input(fname, inplace=1)
  for i, line in enumerate(finput):
    sys.stdout.write(line.replace(line_in, line_out))
  finput.close()



def run_epoch(intensity, data_dir = 'Data', output = False):

    dir = os.getenv('EPOCH1D')

    try:
        os.mkdir(data_dir)
    except:
        os.system('rm '+str(data_dir)+'/*sdf')

    
    # os.system(f'touch {dir}/'+str(data_dir)+'/input.deck')
    os.system(f'cp {dir}/input.deck ' + str(data_dir)+'/input.deck')

    replace_line('intensity_w_cm2 = 2.0e15', f'intensity_w_cm2 = {intensity}', fname = str(data_dir)+'/input.deck')

    if output:
        os.system(f'echo ' + str(data_dir) + ' | mpiexec -n 4 ./bin/epoch1d')
    else:
        os.system(f'echo ' + str(data_dir) + ' | mpiexec -n 4 ./bin/epoch1d > run.log')



def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

# run_epoch(intensity=1e14, output=True)

