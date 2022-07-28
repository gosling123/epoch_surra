from utils import *
from plasma_calc import *
from fields_calc import *
from hot_elec_calc import *
from sim_setup import *

## run_epoch
#
# Runs epoch1d simulations for set intensity 
# @param inputs  1D array of length two, [intensity, density scale length] 
# @param dir  Directory to store epoch data to and where the input.deck file is
# @param input_file  Input file name in the input_decks directory
# @param output  Ouput to command line (True) or to run.log file (False)
# @param np  Number of processors to eun epoch1d on (MPI)    
def run_epoch(inputs, dir = 'Data', input_file = 'input_0.15nc_mid.deck', output = False, npro = 4):
    if np.size(inputs) != 2:
            raise ValueError('ERROR: inputs must have length == 2 [intensity, density scale length]')
    I = inputs[0]
    Ln = inputs[1]
    epoch_path = os.getenv('EPOCH_SURRA')
    create_dir(dir)
    input_deck(I, Ln, dir, input_file)
    # try:
    #     os.mkdir(dir)
    # except:
    #     os.system('rm '+str(dir)+'/*sdf')
    # try:
    #     os.system(f'cp {epoch_path}/input_decks/{input_file}' + str(dir)+'/input.deck')
    # except:
    #     return print('ERROR: Ensure the input_file name is correct as in the input_decks directory')
    # replace_line('intensity_w_cm2 =', f'intensity = {intensity}', fname = str(dir)+'/input.deck')
    # replace_line('Ln =', f'Ln = {Ln}', fname = str(dir)+'/input.deck')
    if output:
        os.system(f'{epoch_path}/epoch.sh ' + str(dir) + ' ' + str(npro) + ' log')
    else:
        os.system(f'{epoch_path}/epoch.sh ' + str(dir) + ' ' + str(npro))

        
        

## get_metrics_res
#
# Finds the backscattered intesnity, hot electron temperature and fraction of electrons with E>100 keV.
# Results are appended to json files for the inputs (I, L) and outputs (I_srs, T, E_frac)
# @param dir : Directory where the resulting sdf files are stored.        
def get_metrics_res(dir):
    print(f'Getting Metric Results for {dir} Directory')
    # Create folder to store training data files
    epoch_path = os.getenv('EPOCH_SURRA')
    fname_in = f'{epoch_path}/training_results/train_inputs.json'
    fname_out = f'{epoch_path}/training_results/train_outputs.json'
    try:
        os.mkdir(f'{epoch_path}/training_results')
    except:
        print('results directory exists')
    try:
        os.path.exists(fname_in)
        with open(fname_in, 'r') as f:
            train_inputs = json.load(f)
    except:
        train_inputs = []
        with open(fname_in, 'w') as f:
            json.dump(train_inputs, f, indent=1) 
    try:
        os.path.exists(fname_out)
        with open(fname_out, 'r') as f:
            train_outputs = json.load(f)
    except:
        train_outputs = []
        with open(fname_out, 'w') as f:
            json.dump(train_outputs, f, indent=1) 
        
    #Find the metric results and append to respective JSON files
    epoch_data = Laser_Plasma_Params(dir = dir)
    I = epoch_data.intensity
    Ln = epoch_data.Ln
    inputs = [I, Ln]
    train_inputs.append(inputs)
    with open(fname_in, 'w') as f:
        json.dump(train_inputs, f, indent=1)

    epoch_fields = EM_fields(dir = dir)
    hot_e_data = hot_electron(dir = dir)
    start = time.time()
    P = epoch_fields.get_flux_grid_av(ncells = 10, signal = 'bsrs', refelctivity = False)
    time_P = time.time()
    print(f'Got Backscatter Intesnisty In {time_P-start} seconds : I_srs = {P}')
    T = hot_e_data.get_hot_e_temp(n = 4, av = True)
    time_T = time.time()
    print(f'Got Temperature In {time_T-time_P} seconds : T = {T} keV')
    E_100_frac = hot_e_data.get_energy_frac_bound(bounds = [100, 999999999])
    time_E = time.time()
    print(f'Got E>100 keV Fraction In {time_E-time_T} seconds : E_frac = {E_100_frac}')
    outputs = [[P], [T], [E_100_frac]]
    train_outputs.append(outputs)
    with open(fname_out, 'w') as f:
        json.dump(train_outputs, f, indent=1)
    