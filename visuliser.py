#!/usr/bin/python3

from time import time
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



class epoch_plotter:

    def __init__(self, dir):
        self.directory = dir+'/'

        self.epoch_data = Laser_Plasma_Params(dir)
        self.epoch_fields = EM_fields(dir)
        self.epoch_data.read_data()
        self.epoch_data.get_spatio_temporal()
        self.epoch_data.get_plasma_param()

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

    def dispersion_2D_plot(self, data, case):

        vmax = np.max(data)
        vmin = vmax*1e-6

        ### FFT PLOT
        fig = plt.figure()
        cmap = cm.inferno

        FFT = plt.pcolormesh(self.epoch_data.k_space, self.epoch_data.omega_space, data, cmap=cmap, norm = LogNorm(vmin=vmin, vmax=vmax), shading ='auto')
        ### CBAR TO GET CURRENT AXIS
        cbar = plt.colorbar(FFT, ax = plt.gca())

        if case == 'Ex':
            cbar.set_label(r'$\left|E_x\right|^2$', x = 0.0, y = 0.5, rotation=0, fontsize=20)
        elif case == 'Ey':
            cbar.set_label(r'$\left|E_y\right|^2$', x = 0.0, y = 0.5, rotation=0, fontsize=20)
        elif case == 'Bz':
            cbar.set_label(r'$\left|B_z\right|^2$', x = 0.0, y = 0.5, rotation=0, fontsize=20)
        else:
            cbar.set_label(r'$\left|2D FFT\right|^2$', x = 0.0, y = 0.5, rotation=90, fontsize=20)
 


        plt.xlim(0,  self.epoch_data.k_space.max())
        plt.ylim(0.0, self.epoch_data.omega_space.max())
        plt.xlabel(r'k/k0', fontsize = 20)
        plt.ylabel(r'$\omega/\omega_0$', fontsize = 20)


        if case == 'Ex':
            ### DISPERISON PLOT
            k_disp = np.linspace(-2, 2, self.epoch_data.nx)
            plt.plot(k_disp, dispersion_EPW(k_disp * self.epoch_data.k0_vac, self.epoch_data.ne_data, self.epoch_data.v_th) / self.epoch_data.frequency, 'white', linestyle='-', label = 'Bohm-Gross')
            plt.plot(k_disp, (1/3)*np.linspace(0,1, self.epoch_data.nx), 'red', linestyle='-', label = 'Bohm-Gross (min)')
            plt.plot(k_disp, dispersion_EPW(k_disp * self.epoch_data.k0_vac, self.epoch_data.ne_max, self.epoch_data.v_th) / self.epoch_data.frequency, 'red', linestyle='-', label = 'Bohm-Gross (max)')
            plt.plot(k_disp, dispersion_Stokes(k_disp*self.epoch_data.k0_vac, self.epoch_data.k0_vac, self.epoch_data.ne_data, self.epoch_data.frequency) / self.epoch_data.frequency, 'blue', linestyle='-', label = 'Stokes')
            plt.legend(fontsize = 20)
        plt.title('Time passed = ' + str(np.round(self.epoch_data.t_end*1e12, decimals=2)) + 'ps', fontsize =20)
        plt.tick_params(labelsize=20)
        plt.gcf().set_size_inches(16,8)

        plt.show()

    def backscatter_flux_plot(self, reflectivity = False):

        time = np.linspace(0, self.epoch_data.t_end, self.epoch_data.timesteps) * 1e12

        if reflectivity:

            res = self.epoch_fields.get_backscat_poynting(time_averaged =False, reflectivity= True)
            
            fig, ax = plt.subplots(figsize=(15,15))
            ax.plot(time, res, label = 'Reflectivity')
            ax.set_xlabel(r'Time (ps)')
            ax.set_ylabel(r'$P0$')
            ax.legend()

            plt.show()

        else:

            res = self.epoch_fields.get_backscat_poynting(time_averaged =False, reflectivity= False)
            
            fig, ax = plt.subplots(figsize=(8,8))
            ax.plot(time, res, label = 'Backscatterd SRS')
            ax.set_xlabel(r'Time (ps)')
            ax.set_ylabel(r'$\langle I_{SRS} \rangle (W/cm^2)$')
            ax.legend()

            plt.show()













        