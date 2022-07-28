from utils import *

## create_dir
#
# Creates a directory inside epoch_surra to host an epoch sim
# @param dir  Directory name
def create_dir(dir):
    if not isinstance(dir,str):
            raise Exception("ERROR: dir argument must be a string (directory)")
    epoch_path = os.getenv('EPOCH_SURRA')
    try:
        os.mkdir(f'{epoch_path}/{dir}')
    except:
        print(f'$EPOCH_SURRA/{dir} directory already exists')

## input_deck
#
# Copies a preset input deck into chosen directory setting the values of
# laser intensity (I), density scale length (Ln) and particles per cell (ppc).
# @param I  Laser intensity (W/cm^2)  
# @param Ln  Density scale length (m)
# @param ppc  Paricles per cell
# @param dir  Directory name
# @param input_file  Input file base to copy from input_decks folder
def input_deck(I, Ln, ppc, dir, input_file = 'input_0.15nc_mid.deck'):
    if not isinstance(ppc,int) or (ppc < 1):
            raise Exception("ERROR: ppc argument must be an integer > 0")
    if not isinstance(dir,str):
            raise Exception("ERROR: dir argument must be a string (directory)")
    if not isinstance(input_file,str):
            raise Exception("ERROR: input_file argument must be a string (.deck file)")
    try:
        os.system(f'{epoch_path}/{dir}/touch input.deck')
    except:
        print(f'$EPOCH_SURRA/{dir}/input.deck already exists')
    try:
        epoch_path = os.getenv('EPOCH_SURRA')
        os.system(f'cp {epoch_path}/input_decks/{input_file} {epoch_path}/{dir}/input.deck')
    except:
        return print('ERROR: Ensure the input_file name is correct as in the input_decks directory')
    replace_line('intensity_w_cm2 =', f'intensity = {I}', fname = str(dir)+'/input.deck')
    replace_line('Ln =', f'Ln = {Ln}', fname = str(dir)+'/input.deck')
    replace_line('PPC =', f'PPC = {ppc}', fname = str(dir)+'/input.deck')

## avon
#
# Function to write job script for avon for epoch sim and places it in chosen directory.
# @param dir  Directory name
# @param time  Time string in the form of hours:minutes:seconds (hh:mm:ss)
# @param nodes  Number of computational nodes to request
def avon(dir, time = '24:00:00',nodes = 1):
    if not isinstance(nodes,int) or (nodes < 1):
            raise Exception("ERROR: nodes argument must be an integer > 0")
    if not isinstance(dir,str):
            raise Exception("ERROR: dir argument must be a string (directory)")
    if not isinstance(time,str):
            raise Exception("ERROR: time argument must be a string in the form of hours:minutes:seconds (hh:mm:ss)")
    f = open(os.path.join(dir,'epoch.sbatch'), 'w+')
    directory = os.path.abspath(dir)
    f.write('#!/bin/bash\n')
    # resource requests
    f.write('#SBATCH --time={:}\n'.format(time))
    f.write('#SBATCH --nodes={:}\n'.format(np.int(nodes)))
    f.write('#SBATCH --ntasks-per-node=48\n')
    f.write('#SBATCH --ntasks={:}\n'.format(int(nodes*48)))
    f.write('#SBATCH --mail-type=END\n')
    f.write('#SBATCH --mail-user=b.gosling@warwick.ac.uk\n')
    f.write('\n')
    f.write('module purge\n')
    f.write('module load GCC/8.3.0  OpenMPI/3.1.4 ScaLAPACK/2.0.2\n')
    f.write('\n')
    f.write('DIR="{:}"\n'.format(dir))
    f.write('echo $DIR | srun $EPOCH1D_EXE')
    f.write('\n')

## epoch_sim_dir
#
# Creates epoch sim directory and populates it with chosen input.deck format
# and job script if hpc is set to True.
# @param dir  Directory name
# @param input_file  Input file base to copy from input_decks folder
# @param I  Laser intensity (W/cm^2)  
# @param Ln  Density scale length (m)
# @param ppc  Paricles per cell
# @param time  Time string in the form of hours:minutes:seconds (hh:mm:ss)
# @param nodes  Number of computational nodes to request
# @param hpc  (Logical) Whether to add hpc (avon) hob script or not
def epoch_sim_dir(dir, input_file, I, Ln, ppc = 100, time = '24:00:00', nodes = 1, hpc = True):
    if not isinstance(ppc,int) or (ppc < 1):
            raise Exception("ERROR: ppc argument must be an integer > 0")
    if not isinstance(dir,str):
            raise Exception("ERROR: dir argument must be a string (directory)")
    if not isinstance(input_file,str):
            raise Exception("ERROR: input_file argument must be a string (.deck file)")
    if not isinstance(time,str):
            raise Exception("ERROR: time argument must be a string in the form of hours:minutes:seconds (hh:mm:ss)")
    if not isinstance(nodes,int) or (nodes < 1):
            raise Exception("ERROR: nodes argument must be an integer > 0")
    create_dir(dir)
    input_deck(I, Ln, ppc, dir, input_file)
    if  hpc:
        avon(dir, time, nodes)
        return print('created directory, input.deck and epoch.sbatch in $EPOCH_SURRA')
    else:
        return print('created directory and input.deck in $EPOCH_SURR')