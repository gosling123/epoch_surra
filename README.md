# epoch_surra

Repo for PX915 summer project

## How to use

Clone this repository inside the epoch1d director once you have a working copy of EPOCH on your machiene. The various python scripts contain classes and functions that aid in the calculation of plasma quantities and LPI metrics. Gaussian Process regression scripts are also contained within this repository for if you want to produce a surrogate for backscattered SRS intensity and hot-electron temperature.

To make full use of the scripts, please add the path to this directory to your .bashrc, with the name `EPOCH_SURRA`.


## Python package dependencies:
#### (NOTE: sdf package comes with EPOCH. Instructions to install package are given in the Using EPOCH section, please do not do pip3 install sdf as this is a seprate package.)

* sdf (Warwick's own scientific data format - comes with EPOCH)
* typing
* time
* fileinput
* os
* sys
* glob
* csv (1.0)
* GPy (1.10.0)
* nbformat (4.4.0)
* matplotlib (3.1.1)
* numpy (1.17.3)
* scipy (1.3.1)
* smt (1.2.0)
* sklearn (1.0.2)


## Acessing EPOCH
This project used EPOCH version 4.17.16, the latest releases can by found here https://github.com/Warwick-Plasma/epoch/releases.

#### Downloading
To download EPOCH using the tarball file, simply click the version you want to download it. Once you have moved it into your chosen directory, you can unpack the code using the command:
`tar xzf epoch-4.17.6.tar.gz`, creating a directory called epoch-4.17.16.

#### Cloning
Alternatively you can clone the EPOCH repository using the command `git clone --recursive https://github.com/Warwick-Plasma/epoch.git` inside a chosen directory of your choice. This will then create a new directory named epoch. The `--recursive` flag is required to have access to the SDF subdirectory.


## Using EPOCH
(More in depth information is found in the README of https://github.com/Warwick-Plasma/epoch and the epoch documentation https://epochpic.github.io/)
#### Compiling EPOCH
This project only uses the 1D version of the code, so this example will be for compiling epoch1d, however to do so for 2D and 3D is analagous to that of 1D.
To compile the 1D version of the code, you must first change to the correct directory (`epoch1d`) by typing `cd <your_name_for_epoch>/epoch1d`, where `<your_name_for_epoch>` could be `epoch-4.17.6` or `epoch` depending on if you downloaded or cloned epoch.
The code is compiled using make, however a compiler must be specified. EPOCH is written using Fortran so a common compiler used is `gfortran`.
* To compile the code you type in the command `make COMPILER=gfortran`.
* To compile the code in parallel (for example using 4 processors) which saves a bit of time, you can use the command `make COMPILER=gfortran -j4` instead.
* You can also save typing by editing your ~/.bashrc file and adding the line export COMPILER=gfortran at the top of the file. Then the command would just be `make -j4`.

#### Compiling SDF
To access the SDF python package used in this repo, type the command `make sdfutils` in the chosen version of epoch you have previously compiled/want to use.
To check everything is set up corrcetly after it has compiled, type in the following commands in the terminal:
* `$ python3`
* `>>> import sdf`
* `>>> sdf.__file__`
Hopefully this will print out a path to your site-packages directory with the filename being of the form `sdf.cpython-37m-x86_64-linux-gnu.so`.

#### Running EPOCH
Once you have built the version of EPOCH you want (e.g 1D), you can simply run it using the command `echo <directory> | mpiexec -n <npro> ./bin/epoch1d`, where you simply replace `<directory>` with the name of the directory that houses the required `input.deck` file and is where the outputted data will be, and `<npro>` to the number of processors you want it to run on.

This reepo houses a shell script which will do this for you named `epoch.sh` which in which the directory and number of processors to use are controlled by an arg parser, as well as an optional to print the result to the file `run.log` rather than the terminal. An exmple is shown below for how to use the shell script to run epoch:
* `$ chmod u+x epoch.sh` (only required the first time)
* `$ ./epoch.sh <directory> <npro>` (prints to terminal)
* `$ ./epoch.sh <directory> <npro> log` (terminal print goes to run.log file)

#### input.deck file
To run epoch, you must have a suitable input file that epoch can read. This must be saved as 'input.deck' and an example of such a file is given in this repo under the same name. This file must always be in the directory you specify in command line, for if the input.deck file is not found epoch won't run. To edit the given input.deck file or create a new one, please read the epoch user manual for gidence.
https://github.com/Warwick-Plasma/EPOCH_manuals/releases/
