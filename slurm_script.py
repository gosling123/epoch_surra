import os
import numpy as np

def avon(time,nodes,directory):
    f = open(os.path.join(directory,'epoch.sbatch'), 'w+')
    directory = os.path.abspath(directory)
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

    f.write('DIR="{:}"\n'.format(directory))
    f.write('echo $DIR | srun ../bin/epoch1d')

    f.write('\n')