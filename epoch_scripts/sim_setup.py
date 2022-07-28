from utils import *

def create_dir(dir):
    epoch_path = os.getenv('EPOCH_SURRA')
    try:
        os.mkdir(f'{epoch_path}/{dir}')
    except:
        print(f'$EPOCH_SURRA/{dir} directory already exists')


def input_deck(I, Ln, dir, input_file = 'input_0.15nc_mid.deck'):
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

def avon(time,nodes,dir):
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

def epoch_sim_dir(dir, I, Ln, input_file, time = 24, nodes = 1, hpc = True):
    create_dir(dir)
    input_deck(I, Ln, dir, input_file)
    if avon:
        avon(time, nodes, dir)
        return print('created directory, input.deck and epoch.sbatch in $EPOCH_SURRA')
    else:
        return print('created directory and input.deck in $EPOCH_SURR')