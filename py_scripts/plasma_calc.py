from utils import *


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
        self.intensity = read_input(self.directory, param='intensity') # intensity W/cm^2
        self.wavelength = 351*nano # laser wavelength
        files =  glob.glob(self.directory+'fields*.sdf') # List of field_ .. .sdf data files
        self.timesteps = len(files) # Number of field timesteps
        self.omega0 =  c * 2 * pi / self.wavelength # Laser angular frequency Rad s^-1
        self.critical_density = self.omega0**2 * me * eps0 / e**2 # crtitical density (omega0 = omega_pe)
        self.k0_vac = 2 * pi / self.wavelength # Laser wavenumber in a vacuum
        self.ppc = read_input(self.directory, param='ppc')
        self.Ln = read_input(self.directory, param='Ln')


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