#!/usr/bin/python3

from nbformat import read
import numpy as np
import sdf
import sdf_helper as sh
from scipy import constants
from scipy.optimize import brentq, bisect
import glob
from utils import *

# Constants

c = constants.c
eps0 = constants.epsilon_0
me = constants.m_e
e = constants.e
kB = constants.k
keV_to_K = (e*1e3)/kB
mu0 = constants.mu_0
pi = np.pi
pico = 1e-12
micron = 1e-6
nano = 1e-9




## Dispersion relations ##

# Electron-Plasma angular frequency
def plasmon(ne):
    return np.sqrt(ne*e**2 / (me*eps0))

# Stokes dispersion realtion for SRS
def dispersion_Stokes(k, k0, ne, omega0):
    omega_pe = plasmon(ne)
    omega_stk = omega0 - np.sqrt(omega_pe**2 + c**2 * (k-k0)**2)
    return omega_stk   

# Electron Plasma wave dispersion realtion - Bohm-Gross
def dispersion_EPW(k, ne, v_th):
    omega_pe = plasmon(ne)
    omega_epw = np.sqrt(omega_pe**2 + 3*k**2*v_th**2)
    return omega_epw

# EM wave in plasama dipersion
def dispersion_EM(k, ne):
    omega_pe = plasmon(ne)
    omega_em = np.sqrt(omega_pe**2 + c**2 * k**2)
    return omega_em


class Laser_Plasma_Params:
    
    def __init__(self, dir):
        self.directory = dir+'/'
        self.intensity = read_intensity(self.directory) ## W/cm^2
        self.wavelength = 351*nano
        # Number of files
        files = (glob.glob(self.directory+'*.sdf')) # lists files that cotaiin the name Ising_Data
        nfiles = len(files) # returns the number of files
        self.timesteps = nfiles
        self.frequency =  c * 2 * pi / self.wavelength ### Rad s^-1
        self.critical_density = self.frequency**2 * me * eps0 / e**2
        self.k0_vac = 2 * pi / self.wavelength
        
    def read_data(self):
        
        # read in data to calculate sim data
        self.data0 = sdf.read(self.directory+'0000.sdf', dict=True) # for initial set-up
        self.data1 = sdf.read(self.directory+'0001.sdf', dict=True) # for working out dt etc
        final_file = self.directory+str(self.timesteps-1).zfill(4)+'.sdf'
        self.data_final = sdf.read(final_file, dict=True) # for end simulation time (or set-cut off)

    def get_spatio_temporal(self, mic = True):
          
        # grid edges (field positions)
        self.grid = np.array((self.data0['Grid/Grid'].data)).reshape(-1) ### m
        
        # centre nodes (thermodymanic variable location)
        self.nodes = np.zeros(len(self.grid)-1)
        for i in range(1, len(self.grid)):
            delta = 0.5*(self.grid[i] - self.grid[i-1])
            self.nodes[i-1] = self.grid[i] - delta
        
        # grid resoloutions
        self.dx = self.grid[1]-self.grid[0]
        self.nx = len(self.nodes)
        self.Lx = self.grid[-1]
        
        # output in microns (better for plots)
        if mic:
            self.grid /= micron
            self.nodes /= micron 
            
        # time step and sim time 
        self.dt = self.data1['Header']['time'] - self.data0['Header']['time']## s
        self.t_end = self.data_final['Header']['time'] ## s
        
        # fourier space
        self.k_space = np.fft.fftshift(np.fft.fftfreq(self.nx, d = self.dx / 2 / pi)) / self.k0_vac
        self.omega_space = np.fft.fftshift(np.fft.fftfreq(self.timesteps, d = self.dt / 2 / pi)) / self.frequency
        
                    
    def get_plasma_param(self):
        
        # number of electrons
        self.npart = len(self.data0['Particles/Px/electrons'].data)
        
        # electron number density
        self.ne_data = np.array(self.data0['Derived/Number_Density/electrons'].data)
        self.ne = np.average(self.ne_data)
        self.ne_min = np.min(self.ne_data)
        self.ne_max = np.max(self.ne_data)
        
        # desnity scale length
        self.Ln = (self.Lx/(np.log(self.ne_max/self.ne_min)))/micron # in microns
        
        # plasmon frequency
        self.omega_pe_data = plasmon(self.ne_data)
        self.omega_pe = plasmon(self.ne)
        self.omega_pe_min = plasmon(self.ne_min)
        self.omega_pe_max = plasmon(self.ne_max)
        
        # initial wavenumber (EM wave in plasma dispersion)
        self.k0 = np.sqrt((self.frequency**2 - self.omega_pe**2) / (c**2))
        
        # temperature in KeV
        self.Te_data = np.array(self.data0['Derived/Temperature/electrons'].data)/keV_to_K
        self.Te = np.average(self.Te_data) 
        
        # thermal velocity
        self.v_th = np.sqrt(kB*self.Te*keV_to_K/me)
        
        # debeye length
        self.deb_len = self.v_th/self.omega_pe


class EM_fields:

    def __init__(self, dir):
        self.directory = dir+'/'
        files = (glob.glob(self.directory+'*.sdf')) # lists files that cotaiin the name Ising_Data
        nfiles = len(files) # returns the number of files
        self.timesteps = nfiles
        data0 = sdf.read(self.directory+'0000.sdf', dict=True) # for initial set-up
        self.nx = len(data0['Electric Field/Ex'].data)

        self.epoch_data = Laser_Plasma_Params(dir)
        self.epoch_data.read_data()
        self.epoch_data.get_spatio_temporal()
        self.epoch_data.get_plasma_param()
    
    # Get time and space data of Ex field
    def get_2D_Electric_Field_x(self):
        Ex = np.zeros((self.timesteps, self.nx))

        for i in range(0, self.timesteps):
            fname = self.directory+str(i).zfill(4)+'.sdf'
            data = sdf.read(fname, dict=True)
            Ex[i,:] = data['Electric Field/Ex'].data

        return Ex

    # Get time and space data of Ey field
    def get_2D_Electric_Field_y(self):
        Ey = np.zeros((self.timesteps, self.nx))

        for i in range(0, self.timesteps):
            fname = self.directory+str(i).zfill(4)+'.sdf'
            data = sdf.read(fname, dict=True)
            Ey[i,:] = data['Electric Field/Ey'].data

        return Ey

    # Get time and space data of Bz field
    def get_2D_Magnetic_Field_z(self):
        Bz = np.zeros((self.timesteps, self.nx))

        for i in range(0, self.timesteps):
            fname = self.directory+str(i).zfill(4)+'.sdf'
            data = sdf.read(fname, dict=True)
            Bz[i,:] = data['Magnetic Field/Bz'].data

        return Bz

    def get_2D_FFT(self, field, square_mod = True):
        if field == 'Ex':
            array = self.get_2D_Electric_Field_x()
        elif field == 'Ey':
            array = self.get_2D_Electric_Field_y()
        elif field == 'Bz':
            array = self.get_2D_Magnetic_Field_z()
        else:
            print('ERROR: Please set field to either Ex, Ey or Bz' )

        # 2D FFT
        tilde_array= np.fft.fft2(array)
        tilde_array = np.fft.fftshift(tilde_array)

        # Reverse second axis as we want right travelling wave to correspond to positive omega
        tilde_array = tilde_array[:, ::-1]

        if square_mod:
            square_mod_res = (np.abs(tilde_array))**2
            return square_mod_res
        else:
            return tilde_array

    def get_time_FFT(self, field, square_mod = True):

        if field == 'Ex':
            array = self.get_2D_Electric_Field_x().T
        elif field == 'Ey':
            array = self.get_2D_Electric_Field_y().T
        elif field == 'Bz':
            array = self.get_2D_Magnetic_Field_z().T
        else:
            print('ERROR: Please set field to either Ex, Ey or Bz' )

        # 1D FFT in time
        time_FFT = np.zeros((self.nx, self.timesteps), dtype = 'complex_')
        
        for i in range(0, self.nx):

            time_FFT[i,:] = np.fft.fftshift(np.fft.fft(array[i,:]))

        if square_mod:
            square_mod_res = (np.abs(time_FFT))**2
            return square_mod_res
        else:
            return time_FFT 

    def get_space_FFT(self, field, square_mod = True):

        if field == 'Ex':
            array = self.get_2D_Electric_Field_x()
        elif field == 'Ey':
            array = self.get_2D_Electric_Field_y()
        elif field == 'Bz':
            array = self.get_2D_Magnetic_Field_z()
        else:
            print('ERROR: Please set field to either Ex, Ey or Bz' )

        # 1D FFT in time
        space_FFT = np.zeros((self.timesteps, self.nx), dtype = 'complex_')
        
        for i in range(0, self.timesteps):

            space_FFT[i,:] = np.fft.fftshift(np.fft.fft(array[i,:]))

        if square_mod:
            square_mod_res = (np.abs(space_FFT))**2
            return square_mod_res
        else:
            return space_FFT

    def get_scatterd_signal(self):

        Ey_FFT_time = self.get_time_FFT(field = 'Ey', square_mod = False)
        Bz_FFT_time = self.get_time_FFT(field = 'Bz', square_mod = False)

        for i in range(self.nx):
            for j in range(self.timesteps):
                if 0.8<=self.epoch_data.omega_space[j]<=1.1:
                    Ey_FFT_time[i,j] = 0.0
                    Bz_FFT_time[i,j] = 0.0
                else:
                    continue
        
        Ey_filter = np.zeros((self.nx, self.timesteps), dtype = float)
        Bz_filter = np.zeros((self.nx, self.timesteps), dtype = float)

        for i in range(self.nx):
            Ey_filter[i] = np.abs(np.fft.ifft(np.fft.ifftshift(Ey_FFT_time[i,:])))
            Bz_filter[i] = np.abs(np.fft.ifft(np.fft.ifftshift(Bz_FFT_time[i,:])))

        # Transpose back to match original form of 2D field arrays
        Ey_filter = Ey_filter.T
        Bz_filter = Bz_filter.T

        return Ey_filter, Bz_filter

    def get_laser_signal(self):

        Ey_FFT_time = self.get_time_FFT(field = 'Ey', square_mod = False)
        Bz_FFT_time = self.get_time_FFT(field = 'Bz', square_mod = False)

        for i in range(self.nx):
            for j in range(self.timesteps):
                if 0.8<=self.epoch_data.omega_space[j]<=1.1:
                    continue                    
                else:
                    Ey_FFT_time[i,j] = 0.0
                    Bz_FFT_time[i,j] = 0.0
        
        Ey_filter = np.zeros((self.nx, self.timesteps), dtype = float)
        Bz_filter = np.zeros((self.nx, self.timesteps), dtype = float)

        for i in range(self.nx):
            Ey_filter[i] = np.abs(np.fft.ifft(np.fft.ifftshift(Ey_FFT_time[i,:])))
            Bz_filter[i] = np.abs(np.fft.ifft(np.fft.ifftshift(Bz_FFT_time[i,:])))

        # Transpose back to match original form of 2D field arrays
        Ey_filter = Ey_filter.T
        Bz_filter = Bz_filter.T

        return Ey_filter, Bz_filter

    # Averaged over cells close to LH boundary 
    def get_backscat_poynting(self, time_averaged = True, reflectivity = False):

        Ey, Bz = self.get_scatterd_signal()

        ncells = 10

        if time_averaged:
            S = 0
            for i in range(self.timesteps):
                for j in range(ncells):

                    S += Ey[i,j]*Bz[i,j]*self.epoch_data.dx*self.epoch_data.dt

            L = (1/ncells)*self.epoch_data.Lx
            T = self.timesteps*self.epoch_data.dt
            W_cm2 = 1e4
            factor = mu0*L*T*W_cm2

            S = (1/factor) * S

            if reflectivity:
                return S/self.epoch_data.intensity
            else:
                return S

        else:

            S = np.zeros(self.timesteps)

            for i in range(self.timesteps):
                for j in range(ncells):

                    S[i] += Ey[i,j]*Bz[i,j]*self.epoch_data.dx
            
                L = (1/ncells)*self.epoch_data.Lx
                W_cm2 = 1e4
                factor = mu0 * L*W_cm2
                S[i] = (1/factor) * S[i]

            if reflectivity:
                return S/self.epoch_data.intensity
            else:
                return S

    # Averaged over cells close to RH boundary 
    def get_transmitted_poynting(self, time_averaged = True, reflectivity = False):

        Ey, Bz = self.get_laser_signal()

        ncells = 10

        if time_averaged:
            S = 0
            for i in range(self.timesteps):
                for j in range(self.nx - ncells, self.nx):

                    S += Ey[i,j]*Bz[i,j]*self.epoch_data.dx*self.epoch_data.dt

            L = (1/ncells)*self.epoch_data.Lx
            T = self.timesteps*self.epoch_data.dt
            W_cm2 = 1e4
            factor = mu0*L*T*W_cm2

            S = (1/factor) * S

            if reflectivity:
                return S/self.epoch_data.intensity
            else:
                return S

        else:

            S = np.zeros(self.timesteps)

            for i in range(self.timesteps):
                for j in range(self.nx - ncells, self.nx):

                    S[i] += Ey[i,j]*Bz[i,j]*self.epoch_data.dx
            
                L = (1/ncells)*self.epoch_data.Lx
                W_cm2 = 1e4
                factor = mu0 * L * W_cm2
                S[i] = (1/factor) * S[i]

            
            if reflectivity:
                return S/self.epoch_data.intensity
            else:
                return S



def get_I_SRS_res(I_array, dir, output):

    for I in I_array:

        print('Strting I = ', I/1e15, ' 10^15 W/cm^2')

        run_epoch(I, data_dir = dir, output = False)

        epoch_fields = EM_fields(dir = dir)

        I_srs = epoch_fields.get_backscat_poynting(time_averaged = True)

        row_contents = [I, I_srs]

        # Append a list as new line to an old csv file
        append_list_as_row(output, row_contents)

        print('appended result - ', row_contents)













