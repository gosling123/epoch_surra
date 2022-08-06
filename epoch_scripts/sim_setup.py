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

## create_sub_dir
#
# Creates a directory and directory within inside epoch_surra to host an epoch sim
# @param dir  Directory name
# @param sub_dir  Sub-Directory name
def create_sub_dir(dir, sub_dir):
    if not isinstance(dir,str):
            raise Exception("ERROR: dir argument must be a string (directory)")
    if not isinstance(sub_dir,str):
            raise Exception("ERROR: sub_dir argument must be a string (sub-directory)")
    epoch_path = os.getenv('EPOCH_SURRA')
    try:
        os.mkdir(f'{epoch_path}/{dir}')
    except:
        print(f'$EPOCH_SURRA/{dir} directory already exists')
    try:
        os.mkdir(f'{epoch_path}/{dir}/{sub_dir}')
    except:
        print(f'$EPOCH_SURRA/{dir}/{sub_dir} directory already exists')

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
    replace_line('intensity_w_cm2 =', f'intensity_w_cm2 = {I}', fname = str(dir)+'/input.deck')
    replace_line('Ln =', f'Ln = {Ln}', fname = str(dir)+'/input.deck')
    replace_line('PPC =', f'PPC = {ppc}', fname = str(dir)+'/input.deck')

## avon
#
# Function to write job script for avon for epoch sim and places it in chosen directory.
# @param dir  Directory name
# @param time  Time string in the form of hours:minutes:seconds (hh:mm:ss)
# @param nodes  Number of computational nodes to request
def avon(dir, fname = 'epoch.sbatch', time = '2:00:00',nodes = 1):
    if not isinstance(nodes,int) or (nodes < 1):
            raise Exception("ERROR: nodes argument must be an integer > 0")
    if not isinstance(dir,str):
            raise Exception("ERROR: dir argument must be a string (directory)")
    if not isinstance(time,str):
            raise Exception("ERROR: time argument must be a string in the form of hours:minutes:seconds (hh:mm:ss)")
    if not isinstance(fname,str):
            raise Exception("ERROR: fname argument must be a string (sbatch job file)")
    f = open(os.path.join(dir,fname), 'w+')
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
    f.write(f'DIR=$EPOCH1D/{dir}')
    f.write('\n')
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
def epoch_sim_dir(dir, input_file, I, Ln, ppc = 100, time = '2:00:00', nodes = 1, hpc = True):
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

## epoch_sim_sub_dirs
#
# Creates epoch sim directories with sub directories and populates it with chosen input.deck format.
# @param dir  Directory name
# @param sub_dir  Sub-Directory name
# @param input_file  Input file base to copy from input_decks folder
# @param I  Laser intensity (W/cm^2)  
# @param Ln  Density scale length (m)
# @param ppc  Paricles per cell
# @param time  Time string in the form of hours:minutes:seconds (hh:mm:ss)
# @param nodes  Number of computational nodes to request
# @param hpc  (Logical) Whether to add hpc (avon) hob script or not
def epoch_sim_sub_dir(dir, sub_dir, input_file, I, Ln, ppc = 100):
    if not isinstance(ppc,int) or (ppc < 1):
            raise Exception("ERROR: ppc argument must be an integer > 0")
    if not isinstance(dir,str):
            raise Exception("ERROR: dir argument must be a string (directory)")
    if not isinstance(sub_dir,str):
            raise Exception("ERROR: sub_dir argument must be a string (sub-directory)")
    if not isinstance(input_file,str):
            raise Exception("ERROR: input_file argument must be a string (.deck file)")
    create_sub_dir(dir, sub_dir)
    dir = f'{dir}/{sub_dir}'
    input_deck(I, Ln, ppc, dir, input_file)
        
    return print('created directory and input.deck in $EPOCH_SURR')

## avon_sub_dirs
#
# Function to write job script for avon for epoch sim for a chosen subdirectory
# @param dir  Directory name
# @param dir  Sub-directory name
# @param time  Time string in the form of hours:minutes:seconds (hh:mm:ss)
# @param nodes  Number of computational nodes to request
def avon_sub_dirs(dir, sub_dir, fname = 'epoch.sbatch', time = '2:00:00',nodes = 1):
    if not isinstance(nodes,int) or (nodes < 1):
            raise Exception("ERROR: nodes argument must be an integer > 0")
    if not isinstance(dir,str):
            raise Exception("ERROR: dir argument must be a string (directory)")
    if not isinstance(sub_dir,str):
            raise Exception("ERROR: sub_dir argument must be a string (sub directory)")
    if not isinstance(fname,str):
            raise Exception("ERROR: fname argument must be a string (sbatch job file)")
    if not isinstance(time,str):
            raise Exception("ERROR: time argument must be a string in the form of hours:minutes:seconds (hh:mm:ss)")
    f = open(os.path.join(f'{dir}/run_scripts',fname), 'w+')
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
    f.write(f'DIR=$EPOCH1D/{dir}/{sub_dir}')
    f.write('\n')
    f.write('echo $DIR | srun $EPOCH1D_EXE')
    f.write('\n')

## run_all_sub_dirs
#
# Function to write shell script which runs all jobs in a directory (i.e all sub directories)
# @param dir  Directory name
# @param sub_dirs  Sub-directories
# @param time  Time string in the form of hours:minutes:seconds (hh:mm:ss)
# @param nodes  Number of computational nodes to request
def run_all_sub_dirs(dir, sub_dirs, time = '2:00:00', nodes = 1):
    if not isinstance(dir,str):
            raise Exception("ERROR: dir argument must be a string (directory)")
    if not isinstance(sub_dirs[0],str):
            raise Exception("ERROR: sub_dirs argument must be a string/array of strings (sub directory/directories)")
    if not isinstance(time,str):
            raise Exception("ERROR: time argument must be a string in the form of hours:minutes:seconds (hh:mm:ss)")
    if not isinstance(nodes,int) or (nodes < 1):
            raise Exception("ERROR: nodes argument must be an integer > 0")
    create_dir(f'{dir}/run_scripts')
    for i in range(0, len(sub_dirs)):
            avon_sub_dirs(dir = dir, sub_dir = sub_dirs[i], fname=f'run{i+1}.sbatch', time = time, nodes = nodes)

    f = open(os.path.join(f'{dir}/run_scripts','run_all.sh'), 'w+')
    f.write('#!/bin/bash\n')
    f.write('\n')
    for i in range(len(sub_dirs)):
            f.write(f'sbatch run{i+1}.sbatch')
            f.write('\n')

## epoch_sim_sub_dirs
#
# Creates epoch sim directories with sub directories and populates it with chosen input.deck format
# and job script folder.
# @param dir  Directory name
# @param sub_dirs Sub-Directories
# @param I_array  Laser intensity array (W/cm^2)  
# @param Ln  Density scale length array (m)
# @param ppc  Paricles per cell
# @param input_file  Input file base to copy from input_decks folder
# @param time  Time string in the form of hours:minutes:seconds (hh:mm:ss)
# @param nodes  Number of computational nodes to request
def hpc_run(dir, sub_dirs, I_array, Ln_array, ppc = 2048, input_file = 'input_0.15nc_mid.deck', time = '2:00:00', nodes = 1):
        if not isinstance(ppc,int) or (ppc < 1):
                raise Exception("ERROR: ppc argument must be an integer > 0")
        if not isinstance(dir,str):
                raise Exception("ERROR: dir argument must be a string (directory)")
        if not isinstance(sub_dirs[0],str):
                raise Exception("ERROR: sub_dirs argument must be a string/array of strings (sub directory/directories)")
        if not isinstance(input_file,str):
                raise Exception("ERROR: input_file argument must be a string (.deck file)")
        if not isinstance(time,str):
                raise Exception("ERROR: time argument must be a string in the form of hours:minutes:seconds (hh:mm:ss)")
        if not isinstance(nodes,int) or (nodes < 1):
                raise Exception("ERROR: nodes argument must be an integer > 0")
        for i in range(len(sub_dirs)):
                epoch_sim_sub_dir(dir, sub_dirs[i], input_file, I_array[i], Ln_array[i], ppc)
        run_all_sub_dirs(dir, sub_dirs, time, nodes)


