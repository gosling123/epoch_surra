## @package visualiser
# Documentation for visualiser module
#
# The visualiser module houses functions which are used to perform
# plotting routines on epoch data.

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
from utils import *
from epoch_calculator import *

plt.rcParams["figure.figsize"] = (20,3)
plt.rcParams["figure.figsize"] = [8, 8]
plt.rcParams["figure.autolayout"] = True
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14


## epoch_plotter Class.
#
# Class that contains plotting routines that are oten used. 
class epoch_plotter:

    ## __init__
    #
    # The constructor
    # @param self : The object pointer
    # @param dir : Directory where data is stored (str)
    def __init__(self, dir):
        self.directory = dir+'/'
        self.epoch_data = Laser_Plasma_Params(dir)
        self.epoch_fields = EM_fields(dir)
        self.epoch_data.read_data()
        self.epoch_data.get_spatio_temporal()
        self.epoch_data.get_plasma_param()
        self.epoch_data.get_matching_conds()

    ## density_plot
    #
    # Plots the number desnity over space
    # @param self : The object pointer
    def density_plot(self):
        x = self.epoch_data.nodes
        ne_min = self.epoch_data.ne_min/self.epoch_data.critical_density
        Ln = self.epoch_data.Ln
        nfunc = (ne_min)*np.exp(x/Ln)
        ne = self.epoch_data.ne_data/self.epoch_data.critical_density

        plt.plot(x, ne, label = 'Grid Data', linewidth = 1)
        plt.plot(x, nfunc, label = '$L_n = '+ str(np.round(Ln, decimals=0)) + ' \, \mu m $')
        plt.xlabel(r'x ($\mu$m)', fontsize = 20)
        plt.ylabel(r'$\frac{n_e}{n_c}$', fontsize = 20)
        plt.gcf().set_size_inches(16,8)
        plt.legend()
        plt.show()

    ## dispersion_2D_plot
    #
    # Plots 2D FFT of the fields
    # @param self : The object pointer
    # @param field : EM Field to FFT (inputs are either 'Ex', 'Ey', 'Bz')
    def dispersion_2D_plot(self, field):

        data = self.epoch_fields.get_2D_FFT(field = field, square_mod = True)

        if field == 'Ey':
            vmax = np.max(data)
            vmin = vmax*1e-8
        else:
            vmax = np.max(data)
            vmin = vmax*1e-8

        ### FFT PLOT
        fig = plt.figure()
        cmap = cm.inferno

        FFT = plt.pcolormesh(self.epoch_data.k_space, self.epoch_data.omega_space, data, cmap=cmap, norm = LogNorm(vmin=vmin, vmax=vmax), shading ='auto')
        ### CBAR TO GET CURRENT AXIS
        cbar = plt.colorbar(FFT, ax = plt.gca())

        if field == 'Ex':
            cbar.set_label(r'$\left|E_x\right|^2$', x = 0.0, y = 0.5, rotation=0, fontsize=20)
        elif field == 'Ey':
            cbar.set_label(r'$\left|E_y\right|^2$', x = 0.0, y = 0.5, rotation=0, fontsize=20)
        elif field == 'Bz':
            cbar.set_label(r'$\left|B_z\right|^2$', x = 0.0, y = 0.5, rotation=0, fontsize=20)
        else:
            cbar.set_label(r'$\left|2D FFT\right|^2$', x = 0.0, y = 0.5, rotation=90, fontsize=20)
 

        if field == 'Ex':

            plt.xlim(0,  2)
            plt.ylim(0.0, self.epoch_data.omega_space.max())
        else:
            plt.xlim(-1.2, 1.2)
            plt.ylim(0, self.epoch_data.omega_space.max())
        
        plt.xlabel(r'k/k0', fontsize = 20)
        plt.ylabel(r'$\omega/\omega_0$', fontsize = 20)


        if field == 'Ex':
            ### DISPERISON PLOT
            k_disp = np.linspace(-2, 2, self.epoch_data.nx)
            plt.plot(k_disp, dispersion_EPW(k_disp * self.epoch_data.k0_vac, np.average(self.epoch_data.ne_data), self.epoch_data.v_th) / self.epoch_data.omega0, 'white', linestyle='-', label = 'Bohm-Gross')
            plt.plot(k_disp, dispersion_Stokes(k_disp*self.epoch_data.k0_vac, self.epoch_data.k0_vac, np.average(self.epoch_data.ne_data), self.epoch_data.omega0) / self.epoch_data.omega0, 'blue', linestyle='-', label = 'Stokes')
            plt.legend(fontsize = 20)
            
        if field == 'Ey':
            plt.plot(self.epoch_data.k_bs_norm, self.epoch_data.omega_bs_norm, marker = 'x', markersize = 16, color = 'white', alpha = 0.85)
            plt.plot(self.epoch_data.k_fs_norm, self.epoch_data.omega_fs_norm, marker = 'x', markersize = 16, color = 'white', alpha = 0.85)
            plt.plot(self.epoch_data.k0/self.epoch_data.k0_vac, 1.0, marker = 'x', markersize = 16, color = 'white', alpha = 0.85)
        
        plt.title('Time passed = ' + str(np.round(self.epoch_data.t_end*1e12, decimals=2)) + 'ps', fontsize =20)
        plt.tick_params(labelsize=20)
        plt.gcf().set_size_inches(16,8)

        plt.show()















        
