from utils import *
from plasma_calc import *
from fields_calc import *
from hot_elec_calc import *
from sim_setup import *

## run_epoch
#
# Runs epoch1d simulations for set intensity 
# @param dir  Directory to store epoch data to and where the input.deck file is
# @param output  Ouput to command line (True) or to run.log file (False)
# @param npro  Number of processors to eun epoch1d on (MPI)    
def run_epoch(dir = 'Data', output = False, npro = 4):
    if not isinstance(npro,int) or (npro < 1):
            raise Exception("ERROR: npro argument must be an integer > 0")
    if not isinstance(dir,str):
            raise Exception("ERROR: dir argument must be a string (directory)")
    epoch_path = os.getenv('EPOCH_SURRA')
    
    path = f'{epoch_path}/{dir}'
    dir_exist = os.path.exists(path)

    if dir_exist == False:
        print(f'ERROR: Directory {dir} does not exist or is not in epoch_surra.')
        print(f'Please ensure the epoch_surra path is added to your .bashrc as EPOCH_SURRA and that the directory {dir} is within the epoch_surra directory.')
        return None
    
    input_path = f'{path}/input.deck'
    input_exist = os.path.exists(input_path)

    if input_exist == False:
        print(f'ERROR: Input file not in {dir}.')
        print(f'Please ensure an input file is conatined within {dir} and named input.deck')
        return None

    I = read_input(f'{dir}/', param = 'intensity')
    Ln = read_input(f'{dir}/', param = 'ne_scale_len')
    ppc = read_input(f'{dir}/', param = 'ppc')
    print(f'Output Directory exists at {dir}')
    print(f'running epoch1d for I = {I} W/cm^2 ; Ln = {Ln} m ; PPC = {ppc}')
    start = time.time()
    if output:
        os.system(f'{epoch_path}/epoch.sh ' + str(dir) + ' ' + str(npro) + ' log')
    else:
        os.system(f'{epoch_path}/epoch.sh ' + str(dir) + ' ' + str(npro))
    print(f'Simulation Complete in {(time.time() - start)/60} minutes')
        
        

## get_metrics_res
#
# Finds the backscattered intesnity, hot electron temperature and fraction of electrons with E>100 keV.
# Results are appended to json files for the inputs (I, L) and outputs (I_srs, T, E_frac)
# @param dir  Directory where the resulting sdf files are stored.        
def get_metrics_res(dir, input_fname = 'train_inputs.json', output_fname = 'train_outputs.json', log = True):
    if not isinstance(dir,str):
            raise Exception("ERROR: dir argument must be a string (directory)")
    print(f'Getting Metric Results for {dir} Directory')
    # Create folder to store training data files
    epoch_path = os.getenv('EPOCH_SURRA')
    fname_in = f'{epoch_path}/training_results/{input_fname}'
    fname_out = f'{epoch_path}/training_results/{output_fname}'
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
    print(f'I = {I*1e-15} e15 w/cm^2')
    print(f'Ln = {Ln*1e6} microns')

    epoch_fields = EM_fields(dir = dir)
    hot_e_data = hot_electron(dir = dir)
    start = time.time()
    P = epoch_fields.get_flux_grid_av(ncells = 10, signal = 'bsrs', refelctivity = True)
    time_P = time.time()
    print(f'Got Backscatter Intesnisty In {time_P-start} seconds : I_srs = {P}')
    T = hot_e_data.get_hot_e_temp(n = 6, av = True)
    time_T = time.time()
    print(f'Got Temperature In {time_T-time_P} seconds : T = {T} keV')
    E_frac = hot_e_data.get_energy_frac()
    time_E = time.time()
    print(f'Got E Fraction In {time_E-time_T} seconds : E_frac = {E_frac}')
    if log:
        outputs = [[np.log(P)], [T], [np.log(E_frac)]]
    else:
        outputs = [[P], [T], [E_frac]]
    train_outputs.append(outputs)
    with open(fname_out, 'w') as f:
        json.dump(train_outputs, f, indent=1)


## get_metrics_res_ensemble
#
# Finds the backscattered intesnity, hot electron temperature and fraction of electrons with E>100 keV.
# Results are appended to json files for the inputs (I, L) and outputs (I_srs, T, E_frac). This is done 
# for an ensemble of the same runs (different random particle set-ups).
# @param dir  Directory where the resulting sdf files are stored.        
def get_metrics_res_ensemble(dirs):
    if len(dirs) == 0:
        raise Exception("ERROR: dirs argument must be an array of directory names (ideally housing the same problem)")
    # Create folder to store training data files
    epoch_path = os.getenv('EPOCH_SURRA')
    fname_in = f'{epoch_path}/training_results/train_inputs_ensemble.json'
    fname_out = f'{epoch_path}/training_results/train_outputs_ensemble.json'
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
    I = np.array([]) ; Ln = np.array([])
    P = np.array([]) ; T = np.array([]); E_frac = np.array([])
    for d in dirs:
        #Find the metric results and append to respective JSON files
        epoch_data = Laser_Plasma_Params(dir = d)
        I = np.append(I,epoch_data.intensity)
        Ln = np.append(Ln, epoch_data.Ln)
        epoch_fields = EM_fields(dir = d)
        hot_e_data = hot_electron(dir = d)
        P = np.append(P, epoch_fields.get_flux_grid_av(ncells = 10, signal = 'bsrs', refelctivity = True))
        T = np.append(T, hot_e_data.get_hot_e_temp(n = 5, av = True))
        E_frac = np.append(E_frac, hot_e_data.get_energy_frac())

    inputs = [np.mean(I), np.mean(Ln)]
    train_inputs.append(inputs)
    with open(fname_in, 'w') as f:
        json.dump(train_inputs, f, indent=1)
    outputs = [[np.mean(P), np.std(P)], [np.mean(T), np.std(T)], [np.mean([E_frac]), np.std(E_frac)]]
    train_outputs.append(outputs)
    with open(fname_out, 'w') as f:
        json.dump(train_outputs, f, indent=1)
    