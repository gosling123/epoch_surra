## @package epoch_calculator
# Documentation for epoch_calculator module
#
# The epoch_calaculator module reads the EPOCH data files
# and calculates several parameters.
# There are three classes, one which handles the calculations from
# grid quantities (Laser_Plasma_Params), field quantities (EM_fileds) 
# and the momentum distribution function (dist_f).

from typing import ForwardRef
import fileinput
import os
import sys
from nbformat import read
import numpy as np
import sdf
from scipy import constants
from scipy.optimize import brentq
import glob
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d

plt.rcParams["figure.figsize"] = (20,3)
plt.rcParams["figure.figsize"] = [15, 15]
plt.rcParams["figure.autolayout"] = True
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20

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
J_tot_KeV = 6.242e+15



## read_intensity
#
# Read intensity in W/cm2 from input.deck
# @param dir : Directory which holds input.deck file you want to read (str)
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
## read_momentum 
#
# Read momentum range of dist function in kg ms^-1 from input.deck
# @param dir : Directory which holds input.deck file you want to read (str)
def read_momentum(dir):
    
    line = []
    with open(dir+'input.deck') as f:
        found = False
        while not found:
            line = f.readline()
            words = line.split()

            if len(words) < 1:
                continue
                
            if words[0] == "range1":
                found = True
                return float(words[2][1:-1]), float(words[3][:-1])

## loss_func
#
# Calculates loss function to compare fits
# @param fit : fit data
# @param data : sim data
def loss_func(fit, data):
    N = len(data)
    sum_ = 0
    for i in range(N):
        l = fit[i] - data[i]
        sum_ += l*l
    
    loss = sum_/N

    return loss

## max_boltz
#
# Maxwell Boltzman distribution
# @param momenta : momentum array
# @param temp : temperature (kelvin)
# @param area  : average area of outputted f(p) from EPOCH (for normalisation to match)
def max_boltz(momenta, temp, area, flux = False):
    max_boltz_data = np.zeros(len(momenta))
    factor = area/np.sqrt(2*me*kB*temp*np.pi) # so integral equals 1
    for i in range(len(momenta)):
        
        exp_term = np.exp(-momenta[i]**2 / (2*me * kB * temp))
        max_boltz_data[i] = factor * exp_term
    if flux:
        vel = momenta/me
        max_boltz_flux = max_boltz_data * vel
        
        return max_boltz_flux
    else:    
        return max_boltz_data

def max_boltz_E(energy, temp):
    counts = np.zeros(len(energy))
    for i in range(len(energy)):
        exp_term = np.exp(-energy[i] / (kB * temp))
        counts[i] = exp_term
    density = counts / (np.sum(counts)*np.diff(energy)[0])
    return density

## winsincFIR
#
# Windowed sinc filter function (http://www.dspguide.com/ch16/2.htm (Equation 16-4))
# @param omega_c : cutoff frequency
# @param omega_s : sampling rate (sampling frequency)
# @param M  : length of the filter kernel (must be odd)
def winsincFIR(omega_c,omega_s,M):
    # cutoff frequency shoudl be a fraction of sampling frequency
    ker = np.sinc(2 * omega_c / omega_s * (np.arange(M) - (M - 1)/2))
    ker *= np.blackman(M)
    # unit gain at zero frequency 
    ker /= np.sum(ker) 
    
    return ker
    
## bandpass
#
# Create a band-pass filter by convolving a high-pass and a low-pass filter
# @param w0 : central frequency you want to filter around (fraction of omega0)
# @param bw : total bandwidth of your filter (fraction of omega0)
# @param M : half filter length (must be odd)
def bandpass(w0,bw,omega_s,M):
    # Angular frequency used for NIF Laser
    omega = 5.36652868179e+15
    w0 = w0 * omega
    bw = bw * omega
    # upper and lower bound frequencies of bandpass
    ub = w0 + (bw / 2)
    lb = w0 - (bw / 2)
    
    # create high-pass filter with cutoff at the lower-bound
    # inverse low-pass filter
    hhpf = -1 * winsincFIR(lb,omega_s,M) 
    hhpf[(M - 1) // 2] += 1
    
    # create low-pass filter with cutoff at the upper-bound
    hlpf = winsincFIR(ub,omega_s,M)
    
    # convolve the two into a band-pass filter
    h = np.convolve(hlpf, hhpf)
    
    return h

## moving_av
#
# Finds movng average of an array using scipys uniform_filter1d function
# @param Q : Data array 
# @param span : length of data 
# @param period : period to average over
def moving_av(Q, span, period = 10):
    return uniform_filter1d(Q, size = span // period)

# Dispersion Relations (fluid theory)

## plasmon
#
# Calculates electron-plasma frequency
# @param ne : Electron number density 
def plasmon(ne):
    return np.sqrt(ne*e**2 / (me*eps0))

## dispersion_Stokes
#
# Stokes dispersion curve (Stokes branch)
# (maximal SRS growth where this curve intersects EPW curve)
# @param k : Wavenumber in plasma
# @param k0 : Vacuum wavenumber (laser) 
# @param ne : Electron number density
# @param omega0 : Laser frequency
def dispersion_Stokes(k, k0, ne, omega0):
    omega_pe = plasmon(ne)
    omega_stk = omega0 - np.sqrt(omega_pe**2 + c**2 * (k-k0)**2)
    return omega_stk   

## dispersion_EPW
#
# Electron Plasma wave dispersion realtion - Bohm-Gross
# @param k : EPW wavenumber in plasma
# @param ne : Electron number density
# @param v_th : Electron thermal velocity
def dispersion_EPW(k, ne, v_th):
    omega_pe = plasmon(ne)
    omega_epw = np.sqrt(omega_pe**2 + 3*k**2*v_th**2)
    return omega_epw

## dispersion_EM
#
# EM wave in plasama dipersion relation
# @param k : EM wavenumber in plasma
# @param ne : Electron number density
def dispersion_EM(k, ne):
    omega_pe = plasmon(ne)
    omega_em = np.sqrt(omega_pe**2 + c**2 * k**2)
    return omega_em


## srs_matching
#
# SRS frequency matching condition (SRS when it returns zero)
# @param k : Wavenumber in plasma
# @param k0 : Vacuum wavenumber (laser) 
# @param ne : Electron number density
# @param v_th : Electron thermal velocity
# @param omega0 : Laser frequency
def srs_matching(k, k0, ne, v_th, omega0):
    
    omega_epw = dispersion_EPW(k, ne, v_th)
    
    ### SRS SCATTER WAVENUMBER AND ANGULAR FREQUENCY  
    k_s = k0 - k
    omega_s = dispersion_EM(k_s, ne)
    
    return omega0 - omega_epw - omega_s

## Laser_Plasma_Params Class.
#
# Class that reads and calculates plasama and grid quantities from grid_ output files. 
class Laser_Plasma_Params:
    
    ## __init__
    #
    # The constructor
    # @param self : The object pointer
    # @param dir : Directory where data is stored (str)
    def __init__(self, dir):
        self.directory = dir+'/' # Directory to look into 
        self.intensity = read_intensity(self.directory) # intensity W/cm^2
        self.wavelength = 351*nano # laser wavelength
        files =  glob.glob(self.directory+'fields*.sdf') # List of field_ .. .sdf data files
        self.timesteps = len(files) # Number of field timesteps
        self.omega0 =  c * 2 * pi / self.wavelength # Laser angular frequency Rad s^-1
        self.critical_density = self.omega0**2 * me * eps0 / e**2 # crtitical density (omega0 = omega_pe)
        self.k0_vac = 2 * pi / self.wavelength # Laser wavenumber in a vacuum

    ## read_data
    #
    # Reads the initial grid data and other files required to find sim data
    # @param self : The object pointer
    def read_data(self):
        self.grid_data = sdf.read(self.directory+'grid_data_0000.sdf', dict=True) # for initial set-up
        self.field_data_0 = sdf.read(self.directory+'fields_0000.sdf', dict=True) # for working out dt etc
        self.field_data_1 = sdf.read(self.directory+'fields_0001.sdf', dict=True) # for working out dt etc
        self.data_final = sdf.read(self.directory+'grid_data_0001.sdf', dict=True) # for end simulation time

    ## get_spatio_temporal
    #
    # Reads the initial grid data and other files required to find sim data
    # @param self : The object pointer
    # @param mic : (Logical) Output grid in microns
    def get_spatio_temporal(self, mic = False):
        self.grid = np.array((self.grid_data['Grid/Grid'].data)).reshape(-1) # grid edges in metres (field positions)
        self.nodes = np.zeros(len(self.grid)-1) # centre nodes (thermodymanic variable location)
        for i in range(1, len(self.grid)):
            delta = 0.5*(self.grid[i] - self.grid[i-1]) # grid centres
            self.nodes[i-1] = self.grid[i] - delta # fill nodal posistions
        
        self.dx = self.grid[1]-self.grid[0] # grid spacing
        self.Lx = self.grid[-1] # length of x domain in metres
        self.nx = len(self.field_data_0['Electric Field/Ex'].data) # grid resoloution (number of cells)
        
            
        self.dt = self.field_data_1['Header']['time'] - self.field_data_0['Header']['time'] # time step between field data dumps in seconds

        self.t_end = self.data_final['Header']['time'] # Total sim time period in seconds
        self.time = np.linspace(0, self.t_end, self.timesteps, endpoint=True) # Total time array for field data dumps in seconds
        
        # Normalised fourier space
        self.k_space = np.fft.fftshift(np.fft.fftfreq(self.nx, d = self.dx / 2 / pi)) / self.k0_vac # k-space 
        self.omega_space = np.fft.fftshift(np.fft.fftfreq(self.timesteps, d = self.dt / 2 / pi)) / self.omega0 # omega-space

        # output in microns (better for plots)
        if mic:
            self.grid /= micron
            self.nodes /= micron
            self.Lx /= micron
            self.dx /= micron  
        
    ## get_plasma_param
    #
    # Calculates plasma parameters/variables
    # @param self : The object pointer               
    def get_plasma_param(self):
        # electron number density
        self.ne_data = np.array(self.grid_data['Derived/Number_Density/electrons'].data) # Initial number density throughout domain in m^-3
        self.ne = np.average(self.ne_data) # Average initial number density in m^-3
        self.ne_min = np.min(self.ne_data) # Minimum initial number density in m^-3
        self.ne_max = np.max(self.ne_data) # Maximum initial number density in m^-3

        dn_dx_mid = (self.ne_data[self.nx//2 + 1] - self.ne_data[self.nx//2 - 1]) / (2.0*self.dx)
        self.Ln = self.ne_data[self.nx//2] / np.abs(dn_dx_mid) /micron # desnity scale length in microns

        self.omega_pe_data = plasmon(self.ne_data) # plasmon frequency throughout domain
        self.omega_pe = plasmon(self.ne) # average plasmon frequency
        self.omega_pe_min = plasmon(self.ne_min) # minimum plasmon frequency
        self.omega_pe_max = plasmon(self.ne_max) # maximum plasmon frequency
        
        self.k0 = np.sqrt((self.omega0**2 - self.omega_pe**2) / (c**2)) # initial wavenumber (EM wave in plasma dispersion)
        
        # Temperature in KeV
        self.Te_data = np.array(self.grid_data['Derived/Temperature/electrons'].data)/keV_to_K # Electron temperature in keV
        self.Te = np.average(self.Te_data) # Average electron temperature in keV
        self.Te_kelvin = self.Te * keV_to_K # Average electron temperature in kelvin
        
        self.v_th = np.sqrt(kB*self.Te*keV_to_K/me) # thermal velocity ms^-1
        
        self.deb_len = self.v_th/self.omega_pe # Debeye length m

    ## get_matching_conds
    #
    # Calculates SRS scattered wavenumber and frequency
    # @param self : The object pointer
    # # @param ne : Electron number density 
    def get_matching_conds(self, ne):

        # SRS matching conditions
        # omega_s = omega_0 - omega_EPW
        # k_s = k_0 - k_EPW

       # Back-scatter
       self.k_epw_bs = brentq(lambda x: srs_matching(x, self.k0, ne, self.v_th, self.omega0), self.k0, 2*self.k0) # EPW wavenumber
       self.omega_epw_bs = dispersion_EPW(self.k_epw_bs, ne, self.v_th)
       self.k_bs = self.k0 - self.k_epw_bs
       self.omega_bs = dispersion_EM(self.k0, ne) - self.omega_epw_bs

       # Normalised values
       self.omega_bs_norm = self.omega_bs / self.omega0
       self.k_bs_norm = self.k_bs / self.k0
       self.k_epw_bs_norm = self.k_epw_bs / self.k0

       # Forward-scatter
       self.k_epw_fs = brentq(lambda x: srs_matching(x, self.k0, ne, self.v_th, self.omega0), 0, self.k0)
       self.omega_epw_fs = dispersion_EPW(self.k_epw_fs, ne, self.v_th)
       self.k_fs = self.k0 - self.k_epw_fs
       self.omega_fs = dispersion_EM(self.k0, ne) - self.omega_epw_fs

       # Normalised values
       self.omega_fs_norm = self.omega_fs / self.omega0
       self.k_fs_norm = self.k_fs / self.k0
       self.k_epw_fs_norm = self.k_epw_fs / self.k0

    ## get_srs_phase_vel
    #
    # Calculates SRS (backscatter) phase velocity at n = ne
    # @param self : The object pointer
    # # @param ne : Electron number density      
    def get_srs_phase_vel(self, ne):

        self.get_matching_conds(ne = ne)

        self.v_phase = self.omega_bs/np.abs(self.k_bs)

        return self.v_phase

## EM_fields Class.
#
# Class that reads and calculates field quantities from fields_ output files.         
class EM_fields:

    ## __init__
    #
    # The constructor
    # @param self : The object pointer
    # @param dir : Directory where data is stored (str)
    def __init__(self, dir):
        self.directory = dir+'/' # Directory to look into 
        self.epoch_data = Laser_Plasma_Params(dir) # Laser_Plasma_Params class to get certain variables
        # call functions needed to get certain stored variables in above class
        self.epoch_data.read_data()
        self.epoch_data.get_spatio_temporal()
        self.epoch_data.get_plasma_param()
        self.timesteps = self.epoch_data.timesteps # Number of timesteps
        self.nx = self.epoch_data.nx # Number of grid cells
    
    ## get_2D_Electric_Field_x
    #
    # Get time and space data of Ex field i.e Ex(x,t)
    # @param self : The object pointer
    def get_2D_Electric_Field_x(self):
        Ex = np.zeros((self.timesteps, self.nx))
        for i in range(0, self.timesteps):
            fname = self.directory+'fields_'+str(i).zfill(4)+'.sdf'
            data = sdf.read(fname, dict=True)
            Ex[i,:] = data['Electric Field/Ex'].data
        return Ex

    ## get_2D_Electric_Field_y
    #
    # Get time and space data of Ey field i.e Ey(x,t)
    # @param self : The object pointer
    def get_2D_Electric_Field_y(self):
        Ey = np.zeros((self.timesteps, self.nx))
        for i in range(0, self.timesteps):
            fname = self.directory+'fields_'+str(i).zfill(4)+'.sdf'
            data = sdf.read(fname, dict=True)
            Ey[i,:] = data['Electric Field/Ey'].data
        return Ey

    ## get_2D_Magnetic_Field_z
    #
    # Get time and space data of Bz field i.e Bz(x,t)
    # @param self : The object pointer
    def get_2D_Magnetic_Field_z(self):
        Bz = np.zeros((self.timesteps, self.nx))
        for i in range(0, self.timesteps):
            fname = self.directory+'fields_'+str(i).zfill(4)+'.sdf'
            data = sdf.read(fname, dict=True)
            Bz[i,:] = data['Magnetic Field/Bz'].data
        return Bz

    ## get_2D_FFT
    #
    # Get 2D FFT (i.e space and time) for specific field
    # @param self : The object pointer
    # @param field : EM Field to FFT (inputs are either 'Ex', 'Ey', 'Bz')
    # @param square_mod : (Logical) outputs the sqaured modulus of the FFT 
    def get_2D_FFT(self, field, square_mod = True):
        if field == 'Ex':
            array = self.get_2D_Electric_Field_x() # Ex(x,t) field
        elif field == 'Ey':
            array = self.get_2D_Electric_Field_y() # Ey(x,t) field
        elif field == 'Bz':
            array = self.get_2D_Magnetic_Field_z() # Bz(x,t) field
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

    ## get_time_FFT
    #
    # Produces 1D time FFT for specific field
    # @param self : The object pointer
    # @param field : EM Field to FFT (inputs are either 'Ex', 'Ey', 'Bz')
    # @param square_mod : (Logical) outputs the sqaured modulus of the FFT
    def get_time_FFT(self, field, square_mod = True):
        if field == 'Ex':
            array = self.get_2D_Electric_Field_x().T # Ex(t,x) field
        elif field == 'Ey':
            array = self.get_2D_Electric_Field_y().T # Ey(t,x) field
        elif field == 'Bz':
            array = self.get_2D_Magnetic_Field_z().T # Bz(t,x) field
        else:
            print('ERROR: Please set field to either Ex, Ey or Bz' )

        # 1D FFT in time
        time_FFT = np.zeros((self.nx, self.timesteps), dtype = 'complex_')
        
        for i in range(0, self.nx):
            time_FFT[i,:] = fft(array[i,:])

        if square_mod:
            square_mod_res = (np.abs(time_FFT))**2
            return square_mod_res
        else:
            return time_FFT 

    ## get_space_FFT
    #
    # Produces 1D space FFT for specific field
    # @param self : The object pointer
    # @param field : EM Field to FFT (inputs are either 'Ex', 'Ey', 'Bz')
    # @param square_mod : (Logical) outputs the sqaured modulus of the FFT
    def get_space_FFT(self, field, square_mod = True):
        if field == 'Ex':
            array = self.get_2D_Electric_Field_x() # Ex(x,t) field
        elif field == 'Ey':
            array = self.get_2D_Electric_Field_y() # Ey(x,t) field
        elif field == 'Bz':
            array = self.get_2D_Magnetic_Field_z() # Bz(x,t) field
        else:
            print('ERROR: Please set field to either Ex, Ey or Bz' )

        # 1D FFT in space
        space_FFT = np.zeros((self.timesteps, self.nx), dtype = 'complex_')
        
        for i in range(0, self.timesteps):
            space_FFT[i,:] = fft(array[i,:])

        if square_mod:
            square_mod_res = (np.abs(space_FFT))**2
            return square_mod_res
        else:
            return space_FFT

    ## get_filtered_signals
    #
    # Finds filtered signals of Ey and Bz fields (either laser signal or SRS signal)
    # @param self : The object pointer
    # @param laser : (Logical) Whether to output laser sginal (true) or SRS signal (false)
    # @param plot_E : (Logical) Whether to plot the filter result at set grid point to test if it works (Ey field)
    # @param plot_B : (Logical) Whether to plot the filter result at set grid point to test if it works (Bz field)
    def get_filtered_signals(self, laser = False, plot_E = False, plot_B = False):

        Ey = self.get_2D_Electric_Field_y().T # Ey(t,x) field
        Bz = self.get_2D_Magnetic_Field_z().T # Bz(t,x) field

        n,m = Ey.shape # array size

        omega = 5.36652868179e+15 # laser frequency
        omega_0 = 1.0 # normalised laser frequency 
        omega_bw = 0.3 # bandswidth centred at laser frequency
        T_end = self.epoch_data.t_end # sim end time
        N = self.epoch_data.timesteps # number of time steps
        dt = T_end/N # time step
        omega_s = 2*np.pi*(1/dt) # sampling frequency 
        M = (N - 1)//2 + 1 # half length of the filter kernel (must be odd) 

        h = bandpass(omega_0,omega_bw,omega_s,M) #bandpass filter

        # Laser signals
        Ey_laser = np.zeros((n, m))
        Bz_laser = np.zeros((n, m))

        # SRS signals
        Ey_SRS = np.zeros((n, m))
        Bz_SRS = np.zeros((n, m))

        # Fill arrays with data
        for i in range(n):

            Ey_laser[i, :] = np.convolve(Ey[i,:],h,mode='same')
            Bz_laser[i, :] = np.convolve(Bz[i,:],h,mode='same')

            Ey_SRS[i, :] = Ey[i,:] - Ey_laser[i,:]
            Bz_SRS[i, :] = Bz[i,:] - Bz_laser[i,:]

        # plots
        if plot_E:
            i = 100
      
            fft_fre = np.fft.fftfreq(N,1/omega_s) / omega
            t = np.linspace(0,T_end,N)
            # plot the signal and its Fourier transform
            fig,ax = plt.subplots(3,2,figsize=(17,12))
            ax[0,0].plot(t*1e12, Ey[i]*1e-11)
            fftsig = fft(Ey[i])
            
            fft_plot = moving_av(np.abs(fftsig), span = len(fft_fre) , period = 100)
            
            ax[0,1].scatter(fft_fre, fft_plot)
            
            laser_sig = Ey_laser[i]
            fft_laser = moving_av(np.abs(fft(laser_sig)), span = len(fft_fre) , period = 100)
            ax[1,0].plot(t*1e12,laser_sig*1e-11)
            ax[1,1].scatter(fft_fre,fft_laser)
                
            scatter_sig = Ey_SRS[i]
            fft_srs = moving_av(np.abs(fft(scatter_sig)), span = len(fft_fre) , period = 100)
            ax[2,0].plot(t*1e12,scatter_sig*1e-11)
            ax[2,1].scatter(fft_fre,fft_srs)
       
            ax[2,0].set_xlabel(r'Time (ps)')
            ax[2,0].set_ylabel(r'$E_y (10^{11} V/m)$')
            ax[1,0].set_ylabel(r'$E_y (10^{11} V/m)$')
            ax[0,0].set_ylabel(r'$E_y (10^{11} V/m)$')
            
            ax[0,1].set_xlim(0, 1.2)
            ax[2,1].set_xlim(0, 1.2)
            ax[1,1].set_xlim(0, 1.2)
            
            ax[2,1].set_xlabel(r'$\omega/\omega_0$')
            ax[0,1].set_ylabel(r'$FFT\left|E_y(t)\right|$')
            ax[1,1].set_ylabel(r'$FFT\left|E_y(t)\right|$')
            ax[2,1].set_ylabel(r'$FFT\left|E_y(t)\right|$')
            ax[0,1].set_yscale('log')
            ax[1,1].set_yscale('log')
            ax[2,1].set_yscale('log')
            plt.show()

        if plot_B:
            i = 100

            fft_fre = np.fft.fftfreq(N,1/omega_s) / omega
            t = np.linspace(0,T_end,N)
            # plot the signal and its Fourier transform
            fig,ax = plt.subplots(3,2,figsize=(17,12))
            ax[0,0].plot(t, Bz[i])
            fftsig = fft(Bz[i])
            
            ax[0,1].plot(fft_fre,np.abs(fftsig))
       
            laser_sig = Bz_laser[i]
            ax[1,0].plot(t,laser_sig)
            ax[1,1].plot(fft_fre,np.abs(fft(laser_sig)))
    
            scatter_sig = Bz_SRS[i]
            ax[2,0].plot(t,scatter_sig)
            ax[2,1].plot(fft_fre,np.abs(fft(scatter_sig)))
   
            plt.show()

        if laser:    
            return Ey_laser, Bz_laser
        else:    
            return Ey_SRS, Bz_SRS

    ## get_flux
    #
    # Finds Poynting flux in x direction Sx = (EyBz-ByEz)/mu0
    # (SRS produces scattered light with same polarisation as the laser (i.e Ez and By are negliable) thus
    #  Sx = EyBz/mu0)
    # @param self : The object pointer
    # @param laser : (Logical) Whether to use laser sginal (true) or SRS signal (false)
    # @param plot_E : (Logical) Whether to output the Sx time series (true) or the time average (false)     
    def get_flux(self, laser = False, time_series = False):
        Ey, Bz = self.get_filtered_signals(laser = laser) # Filtered Ey and Bz fields
        Ey = Ey.T # transpose back to be Ey(x,t)
        Bz = Bz.T # transpose back to be Bz(x,t)
        W_cm2 = 1e4 # Convert to W_cm2
        factor = mu0*W_cm2 # Denominator of Sx
        period = self.timesteps*self.epoch_data.dt # end time
        S = Ey*Bz/factor # poynting flux
        
        # time average
        sum_ = np.zeros(self.nx)
        for i in range(self.timesteps):
            sum_ += S[i]*self.epoch_data.dt

        S_av = sum_/period
        
        if time_series:
            return S
        else:
            return S_av
        
    ## get_flux_grid_av
    #
    # Averages Poynting flux over ncells (near LH boundary for backscatter SRS and RH boundary for laser)
    # @param self : The object pointer
    # @param ncells : Number of cells to average over (default 50)
    # @param laser : (Logical) Whether to use laser sginal (true) or SRS signal (false)   
    def get_flux_grid_av(self, ncells = 50, laser = False):
        St_av = np.abs(self.get_flux(laser = laser))
        
        if laser:
            sum_ = 0
            for i in range(self.nx - ncells, self.nx):
                sum_ += St_av[i] 
        else:
            sum_ = 0
            for i in range(ncells):
                sum_ += St_av[i]
     
        S_av = sum_/ncells
       
        return S_av
        
## dist_f Class.
#
# Class that reads and ouptputs the electron momentum distrubution function from output dist_ .. .sdf files.  
class dist_f:
    
    ## __init__
    #
    # The constructor
    # @param self : The object pointer
    # @param dir : Directory where data is stored (str)
    def __init__(self, dir):
        self.directory = dir+'/' # Directory to look into 
        self.files =  glob.glob(self.directory+'dist*.sdf') # list of dist_ .. .sdf files
        self.nfiles = len(self.files) # number of dist_ .. .sdf files
        self.p_range = read_momentum(dir = self.directory)
        self.p_max = self.p_range[1] # momentum max 
        self.p_min = self.p_range[0] # momentum min
        data0 = sdf.read(self.directory+'dist_0000.sdf', dict=True) # read first file to get resoloution (i.e number of bins)
        self.res = len(data0['dist_fn/x_px/electrons'].data) # resoloution
        self.epoch_data = Laser_Plasma_Params(dir) # Laser_Plasma_Params class to get certain variables
        # call functions needed to get certain stored variables in above class
        self.epoch_data.read_data()
        self.epoch_data.get_spatio_temporal()
        self.epoch_data.get_plasma_param()
    
        self.v_th = self.epoch_data.v_th # electron thermal velocity 
        
        self.p_norm = me * self.v_th # momentum normalisation (thermal momentum )

    ## read_dist_data
    #
    # Read and store distrinution functions at output times
    # @param self : The object pointer
    def read_dist_data(self):
        
        self.times = []
        self.dist_funcs = np.zeros((self.nfiles, self.res))
        self.smooth_dist_funcs =  np.zeros((self.nfiles, self.res))
        
        for i in range(self.nfiles):
            
            fname = self.directory+'dist_'+str(i).zfill(4)+'.sdf'
            data = sdf.read(fname, dict=True)
            self.times.append(data['Header']['time'])
            
            self.dist_funcs[i] = data['dist_fn/x_px/electrons'].data
            self.smooth_dist_funcs[i] = moving_av(self.dist_funcs[i], self.res, period = self.res//40)
            
        self.dist_funcs_av = np.zeros(self.res)
        self.smooth_dist_funcs_av = np.zeros(self.res)
        
        for dist, smooth_dist in zip(self.dist_funcs, self.smooth_dist_funcs):
            self.dist_funcs_av += dist
            self.smooth_dist_funcs_av += smooth_dist
            
        self.dist_funcs_av /= self.nfiles
        self.smooth_dist_funcs_av /= self.nfiles
        self.momenta_bins = np.linspace(self.p_min, self.p_max, self.res, endpoint = True)
        self.energy_bins = self.momenta_bins**2 / (2*me)
        self.energy_bins_keV = self.energy_bins/(e * 1e3)

        self.area_av = 0

        for f in self.dist_funcs:
            self.area_av += np.trapz(f, x=self.momenta_bins)
        self.area_av /= len(self.dist_funcs)

        self.max_boltz_eq = max_boltz(self.momenta_bins, self.epoch_data.Te_kelvin, self.area_av)
        

    ## plot_p_dist_func
    #
    # Plots all distribution functions
    # @param self : The object pointer
    # @param smooth : (Logical) Smooths out function of random zeros
    # @param scaled_x : (Logical) Scales momentum using p_norm
    # @param plot_hot_e : (Logical) Plots hot electron tail
    def plot_p_dist_func(self, smooth = False, scaled_x = False, plot_hot_e = False):
        
        self.read_dist_data()
        
        if smooth:
            self.plot_dist_funcs = self.smooth_dist_funcs
            self.plot_dist_funcs_av = self.smooth_dist_funcs_av
        else:
            self.plot_dist_funcs = self.dist_funcs
            self.plot_dist_funcs_av = self.dist_funcs_av
        
        if scaled_x:
            for i in range(self.nfiles):
                plt.plot(self.momenta_bins / self.p_norm, self.plot_dist_funcs[i], label = str(np.round(self.times[i]*1e12, 3)) + 'ps', alpha = 0.7)
            plt.plot(self.momenta_bins / self.p_norm, self.plot_dist_funcs_av, label = 'Time Average', color = 'black')
            plt.plot(self.momenta_bins / self.p_norm, self.max_boltz_eq, label = 'Equilibrium MB', color = 'magenta')
            
        else:
            for i in range(self.nfiles):
                plt.plot(self.momenta_bins, self.plot_dist_funcs[i], label =  str(np.round(self.times[i]*1e12, 3)) + 'ps', alpha = 0.7)
            plt.plot(self.momenta_bins, self.plot_dist_funcs_av, label = 'Time Average', color = 'black')
            plt.plot(self.momenta_bins, self.max_boltz_eq, label = 'Equilibrium MB', color = 'magenta')
        
        
        self.y_max = 1.1*self.plot_dist_funcs.max()
        self.y_min = self.y_max * 1e-4
          
        plt.yscale('log')
        plt.ylim(self.y_min, self.y_max)
        if scaled_x:
           plt.xlim(self.p_min/self.p_norm, self.p_max/self.p_norm)
           plt.xlabel(r'$p / m_e v_{th}$)')
            
        else:
           plt.xlim(self.p_min, self.p_max)           
           plt.xlabel(r'$p (m s^{-1}$)')  
         
        if plot_hot_e:
            y_min = 5e16
            y_max = self.plot_dist_funcs[-1][2000]
    
            plt.ylim(y_min, y_max)
            if scaled_x:
                plt.xlabel(r'$p / m_e v_{th}$)')
                plt.xlim(self.momenta_bins[2000]/self.p_norm, self.momenta_bins[-1]/self.p_norm)
            else:
                plt.xlabel(r'$p (m s^{-1}$)')
                plt.xlim(self.momenta_bins[2000], self.momenta_bins[-1])
                    
        
        plt.ylabel(r'$f(p)$')
        plt.legend(fontsize = 16)
                    
        plt.gcf().set_size_inches(16,8)
                      
        plt.show()
        
   
class hot_electron:
    
    ## __init__
    #
    # The constructor
    # @param self : The object pointer
    # @param dir : Directory where data is stored (str)
    def __init__(self, dir):
        self.directory = dir+'/' # Directory to look into 
        self.epoch_data = Laser_Plasma_Params(dir) # Laser_Plasma_Params class to get certain variables
        # call functions needed to get certain stored variables in above class
        self.epoch_data.read_data()
        self.epoch_data.get_spatio_temporal()
        self.epoch_data.get_plasma_param()
        self.interp_len = 500
        self.nbins = 100

    
    

    def get_flux_dist(self, smooth = True, plot = False, log = False):

        files = glob.glob(self.directory+'probes_*.sdf')
        self.E_data = np.array([])
        for file in files:
            data = sdf.read(file)
            px = data.outgoing_e_probe__Px.data
            E = np.array(px**2/(2*me))
            self.E_data = np.concatenate((self.E_data, E), axis=None)
            
        
    
        self.E_dist, bins = np.histogram(self.E_data, bins = self.nbins, density=True)

        idx = np.where(self.E_dist/self.E_dist.max() < 1e-4)[0][0]

        self.E_dist = self.E_dist[:idx]
        bins = bins[:idx+1]

        self.E_bins = np.zeros(len(bins)-1)
        for i in range(1, len(bins)):
            delta = 0.5*(bins[i] - bins[i-1]) # grid centres
            self.E_bins[i-1] = bins[i] - delta # fill nodal posistions
  
  
        self.E_dist_smooth = moving_av(self.E_dist, len(self.E_dist), period = len(self.E_dist)//5)


        if plot:

            MB = max_boltz_E(self.E_bins, self.epoch_data.Te*keV_to_K)

            plt.plot(self.E_bins*J_tot_KeV, self.E_dist/self.E_dist.max(), label = 'Raw Data')
            plt.plot(self.E_bins*J_tot_KeV, self.E_dist_smooth/self.E_dist_smooth.max(), label = 'Smoothed')
            plt.plot(self.E_bins*J_tot_KeV, MB/MB.max(), label = 'MB', linestyle = '--', color = 'black')
            plt.legend()
            

            if log:
                plt.yscale('log')
            plt.xlabel(r'$E_e (keV)$')
            plt.ylabel(r'$f_e(E)$')
            # plt.ylim(1e-4, 1e0)
            plt.gcf().set_size_inches(10,6)

        if smooth:
            if plot:
                return None
            else:
                return self.E_bins, self.E_dist_smooth
        
        else:
            if plot:
                return None
            else:
                return self.E_bins, self.E_dist

    def split_dist(self, n, smooth = True, plot = False, log = False):

        E, flux = self.get_flux_dist(smooth = smooth)

        MB_eq = max_boltz_E(E, self.epoch_data.Te*keV_to_K)
        MB_eq_norm = MB_eq/MB_eq.max()
        flux_norm = flux/flux.max()

        error_per = np.abs((flux_norm -  MB_eq_norm) / MB_eq_norm)*100
        idx = np.where(error_per > 100)

        if len(idx[0])>0:
            idx = idx[0][0]
        else:
            idx = 0
        self.idx = idx
        flux_cut = flux[self.idx:]
        E_cut = E[self.idx:]
        
        flux_func = interp1d(E_cut, flux_cut)
        E_final = np.linspace(E_cut.min(), E_cut.max(), self.interp_len, endpoint=True)
        flux_final = flux_func(E_final)

        self.flux_parts = []
        self.E_parts = []

        len_parts = len(flux_final)//n

        if len_parts < 5:
            n = len(flux_final)//5
            print('Number of points chosen to fit maxewellian to deemed to low, \
                   setting to minimum value')
    
        for i in range(n):
            f = flux_final[i*len_parts:(i+1)*len_parts]
            e = E_final[i*len_parts:(i+1)*len_parts]
            self.flux_parts.append(f)
            self.E_parts.append(e)

        self.flux_parts = np.array(self.flux_parts)
        self.E_parts = np.array(self.E_parts)


        if plot:
            for e, f in zip(self.E_parts, self.flux_parts):
                plt.plot(e*J_tot_KeV, f)
            if log:
                plt.yscale('log')
            plt.xlabel(r'$E_e (keV)$')
            plt.ylabel(r'$f_e(E)$')
            plt.xlim(0, 100)
            plt.gcf().set_size_inches(10,6)

        if plot:
            return None
        else:
            return self.E_parts, self.flux_parts

    def fit_maxwellians(self, n, smooth = True, plot = False, log = False):

        E_parts, flux_parts = self.split_dist(n = n, smooth = smooth)

        temps = np.linspace(1, 100, 1000)
        n = len(flux_parts)
        loss = np.zeros((n, len(temps)))

        count = 0
        for e,f in zip(E_parts,flux_parts):
            res = []
            for i,t in enumerate(temps):
                fit = max_boltz_E(e, t*keV_to_K)
                res.append(loss_func(fit/fit.max(), f/f.max()))
            loss[count] = res
            count += 1

        self.T_vals = np.array([])
        for l in loss:
            idx = np.where(l == l.min())[0][0]
            self.T_vals = np.append(self.T_vals, temps[idx])

 
        count = 0
        self.scaled_fits = []
        self.MB_fits = []
        self.amplitudes = np.array([])
        self.scaled_fits_full = []
        for e, f in zip(E_parts,flux_parts):
            MB  = max_boltz_E(e, self.T_vals[count]*keV_to_K)
            MB_full  = max_boltz_E(self.E_bins, self.T_vals[count]*keV_to_K)
            amp = f.max()/MB.max()
            amp_full = f.max()/MB_full.max()
            self.amplitudes = np.append(self.amplitudes, amp)
            self.scaled_fits.append(MB*amp)
            self.MB_fits.append(MB/MB.max())
            self.scaled_fits_full.append(MB_full*amp_full)
            count += 1

        self.final_fit = np.array([])
        E_plot = np.array([])
        flux_plot = np.array([])
        for i, f in enumerate(self.scaled_fits):
            self.final_fit = np.concatenate((self.final_fit, f), axis = None)
            E_plot = np.concatenate((E_plot, E_parts[i]), axis = None)
            flux_plot = np.concatenate((flux_plot, flux_parts[i]), axis = None)

        
        if plot:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            for l in loss:
                ax1.plot(temps, l)
                ax1.set_xlabel(r'$T (keV)$')
                ax1.set_ylabel(r'Normed Loss Function')
            
            for i,f in enumerate(self.scaled_fits):
                ax2.plot(E_parts[i]*J_tot_KeV, f, label = r'$A$ = ' + str(np.round(self.amplitudes[i], 5)) + ', $T$ = ' + str(np.round(self.T_vals[i], 3)) + ' keV')
            ax2.scatter(E_plot*J_tot_KeV, flux_plot, color = 'black', label = 'Data')
            ax2.set_xlabel(r'$E (keV)$')
            ax2.set_ylabel(r'$f_e(E)$')
            ax2.legend()

            
            ax3.scatter(E_plot*J_tot_KeV, flux_plot, color = 'black', label = 'Data')
            ax3.plot(E_plot*J_tot_KeV, self.final_fit, color = 'red', label = 'Fit')
            ax3.set_xlabel(r'$E (keV)$')
            ax3.set_ylabel(r'$f_e(E)$')
            ax3.legend()
            if log:
                ax2.set_yscale('log')
                ax3.set_yscale('log')
            plt.gcf().set_size_inches(25,25)
        if plot:
            return None
        else:
            return self.T_vals, self.amplitudes, self.scaled_fits, self.scaled_fits_full

    def get_hot_e_temp(self, n = 40, smooth = True, av = True, plot = False):

        if av == False:
            T_vals, A, fits, fits_full = self.fit_maxwellians(n = n, smooth = smooth, plot = False)
            self.T_hot = np.average(T_vals, weights = A)
        
        # sum1 = 0; sum2 = 0
        # for T, f in zip(T_vals, fits):
        #     sum1 += np.sum(T*f)
        #     sum2 += np.sum(f)

        # self.T_hot_2 = sum1/sum2

        nfits = np.arange(1, self.interp_len//5)
        T_data = []
        for n in nfits:
            T_vals, A, fits, fits_full = self.fit_maxwellians(n = n, smooth = smooth, plot = False)
            T_est = np.average(T_vals, weights = A)
            T_data.append(T_est)
        self.T_hot_av = np.average(T_data)

        if plot:
            plt.plot(nfits, T_data, '-o', label = 'Data')
            plt.xlabel(r'$N$ Fits')
            plt.ylabel(r'$T_{hot}$')
            plt.axhline(self.T_hot_av, color ='red', ls = '--', label = 'Average')
            plt.gcf().set_size_inches(8,6)
            plt.legend()

        if av:
            if plot:
                return None
            else:
                return self.T_hot_av    
        else:
            if plot:
                return None
            else:
                return self.T_hot

    def get_energy_frac(self, smooth = True):
        E_h, f_h = self.split_dist(n = 1, smooth = smooth)
        E, f = self.get_flux_dist(smooth = smooth)

        if self.idx == 0:
            self.E_hot_frac = np.abs(1 - np.trapz(y=f_h, x =E_h) / np.trapz(y=f, x =E))
            
        else:
            self.E_hot_frac = np.trapz(y=f_h, x =E_h) / np.trapz(y=f, x =E)

        return self.E_hot_frac[0]



    
        




## replace_line
#
# Function rewrite line in input.deck via python
# @param line_in : original line in input.deck 
# @param line_out : repacement of line_in in input.deck
def replace_line(line_in, line_out, fname):
  finput = fileinput.input(fname, inplace=1)
  for i, line in enumerate(finput):
    sys.stdout.write(line.replace(line_in, line_out))
  finput.close()

## run_epoch
#
# Runs epoch1d simulations for set intensity 
# @param intensity : Intensity to write in input.deck 
# @param data_dir : Directory to store epoch data to and where the input.deck file is
# @param output : Ouput to command line (True) or to run.log file (False)
# @param np : Number of processors to eun epoch1d on (MPI)
def run_epoch(intensity, data_dir = 'Data', output = False, np = 10):
    dir = os.getenv('EPOCH1D')
    try:
        os.mkdir(data_dir)
    except:
        os.system('rm '+str(data_dir)+'/*sdf')
    os.system(f'cp {dir}/input.deck ' + str(data_dir)+'/input.deck')
    replace_line('intensity_w_cm2 =', f'intensity_w_cm2 = {intensity}', fname = str(data_dir)+'/input.deck')
    if output:
        os.system(f'echo ' + str(data_dir) + ' | mpiexec -n ' + str(np) + ' ./bin/epoch1d')
    else:
        os.system(f'echo ' + str(data_dir) + ' | mpiexec -n ' + str(np) + ' ./bin/epoch1d > run.log')
        

## get_metrics_res
#
# Runs epoch1d simulations for changing intensity and ouptputs backsactter SRS intensity and T_hot
# @param I_array : Intensity array (list of data to sim) 
# @param dir : Directory to store epoch data to and where the input.deck file is
# @param np : Number of processors to eun epoch1d on (MPI)        
def get_metrics_res(I_array, dir, np = 4):
    I_SRS_data = []
    T_hot_data = []
    E_frac_data = []
    I_L_data = []

    fname_srs = 'metric_results/I_SRS_Data.npy'
    fname_T = 'metric_results/T_hot_Data.npy'
    fname_E = 'metric_results/E_frac_data.npy'
    fname_I = 'metric_results/I_L_data.npy'
    for I in I_array:
        print('###########################')
        print('Strting I = ', I/1e15, ' 10^15 W/cm^2')
        run_epoch(I, data_dir = dir, output = False, np = np)
        epoch_fields = EM_fields(dir = dir)
        res_srs = epoch_fields.get_flux_grid_av(ncells = 10, laser = False)
        he = hot_electron(dir = dir)
        res_T = he.get_hot_e_temp(n = 40, smooth = True)
        res_E_frac = he.get_energy_frac(smooth = True)
        
        
        I_SRS_data.append(res_srs)
        T_hot_data.append(res_T)
        E_frac_data.append(res_E_frac)
        
        print('I_SRS = ', res_srs/1e15, ' 10^15 W/cm^2')
        print('P = ', res_srs/I)
        print('T_hot = ', res_T, ' keV ')
        print('E_frac = ', res_E_frac*100, ' % ')
        
    print('Writing Data to .npy files')   
    np.save(fname_srs, np.array(I_SRS_data))
    np.save(fname_T, np.array(T_hot_data))
    np.save(fname_E, np.array(E_frac_data))
    np.save(fname_I, np.array(I_L_data))
   

    
    





