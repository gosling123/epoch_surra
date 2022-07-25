from utils import *
from plasma_calc import *
from fields_calc import *
from hot_elec_calc import *
import time

## run_epoch
#
# Runs epoch1d simulations for set intensity 
# @param intensity : Intensity to write in input.deck 
# @param data_dir : Directory to store epoch data to and where the input.deck file is
# @param output : Ouput to command line (True) or to run.log file (False)
# @param np : Number of processors to eun epoch1d on (MPI)    
def run_epoch(intensity, data_dir = 'Data', output = False, npro = 4):
    dir = os.getenv('EPOCH1D')
    try:
        os.mkdir(data_dir)
    except:
        os.system('rm '+str(data_dir)+'/*sdf')
    os.system(f'cp {dir}/input.deck ' + str(data_dir)+'/input.deck')
    replace_line('intensity =', f'intensity = {intensity}', fname = str(data_dir)+'/input.deck')
    if output:
        os.system(f'./epoch.sh ' + str(data_dir) + ' ' + str(npro) + ' log')
    else:
        os.system(f'./epoch.sh ' + str(data_dir) + ' ' + str(npro))

        
        

## get_metrics_res
#
# Runs epoch1d simulations for changing intensity and ouptputs backsactter SRS intensity and T_hot
# @param I_array : Intensity array (list of data to sim) 
# @param dir : Directory to store epoch data to and where the input.deck file is
# @param npro : Number of processors to eun epoch1d on (MPI)
# # @param fname : filename for csv writer for back up             
def get_metrics_res(I_array, dir, npro = 4, fname = 'run_epoch.csv'):

    try:
        os.mkdir('metric_results')
    except:
        print('results directory exists')

    try:
        os.system(r'touch ' + str(fname))
        print(str(fname) + ' created for back up')
    except:
        print(str(fname) + ' exists, using for back up')
        

    I_SRS_data = np.array([])
    T_hot_data = np.array([])
    E_frac_data = np.array([])
    I_L_data = np.array([])

    fname_srs = 'metric_results/I_SRS_Data3.npy'
    fname_T = 'metric_results/T_hot_Data3.npy'
    fname_E = 'metric_results/E_frac_data3.npy'
    fname_I = 'metric_results/I_L_data3.npy'
    for I in I_array:
        t1 = time.time()
        print('###########################')
        print('Strting I = ', I/1e15, ' 10^15 W/cm^2')
        run_epoch(I, data_dir = dir, output = True, npro = npro)
        epoch_data = Laser_Plasma_Params(dir = dir)
        Ln = epoch_data.Ln / micron # in microns
        epoch_fields = EM_fields(dir = dir)
        res_srs = epoch_fields.get_flux_grid_av(ncells = 30, laser = False)
        he = hot_electron(dir = dir)
        res_T = he.get_hot_e_temp(n = 5, av=True, smooth = False)
        res_E_frac = he.get_energy_frac(smooth = False)
        
        res = [I, Ln, res_srs, res_srs/I, res_T, res_E_frac]

        append_list_as_row(fname, res)
        
        I_SRS_data = np.append(I_SRS_data, res_srs)
        T_hot_data = np.append(T_hot_data, res_T)
        E_frac_data = np.append(E_frac_data, res_E_frac)
        
        print('I_SRS = ', res_srs/1e13, ' 10^13 W/cm^2')
        print('P = ', res_srs/I)
        print('T_hot = ', res_T, ' keV ')
        print('E_frac = ', res_E_frac*100, ' % ')
        t2 = time.time()
        print('---------------------------')
        print('Time Elapsed = ', (t2 - t1)/60 , ' Minutes')
        print('---------------------------')
    print('Writing Data to .npy files')   
    np.save(fname_srs, I_SRS_data)
    np.save(fname_T, T_hot_data)
    np.save(fname_E, E_frac_data)
    np.save(fname_I, I_L_data)