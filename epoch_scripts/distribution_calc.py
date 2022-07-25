from utils import *
from plasma_calc import *

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
        # Maxwell-Boltzmann factor
        exp_term = np.exp(-momenta[i]**2 / (2*me * kB * temp))
        max_boltz_data[i] = factor * exp_term
    # ouput maxwellian flux velocity * f(p)
    if flux:
        vel = momenta/me
        max_boltz_flux = max_boltz_data * vel
        return max_boltz_flux
    else:    
        return max_boltz_data


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
        self.p_range = read_input(dir = self.directory, param = 'momentum')
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
        
        self.times = [] # time 
        self.dist_funcs = np.zeros((self.nfiles, self.res)) # distribution functions at each time interval
        self.smooth_dist_funcs =  np.zeros((self.nfiles, self.res)) # smoothed distributions using uniform filter
        
        for i in range(self.nfiles):
            
            fname = self.directory+'dist_'+str(i).zfill(4)+'.sdf'
            data = sdf.read(fname, dict=True)
            self.times.append(data['Header']['time'])
            self.dist_funcs[i] = data['dist_fn/x_px/electrons'].data
            # uniform filter
            self.smooth_dist_funcs[i] = moving_av(self.dist_funcs[i], self.res, period = self.res//40)

        # time-averaged distribution function 
        self.dist_funcs_av = np.zeros(self.res)
        self.smooth_dist_funcs_av = np.zeros(self.res)
        
        for dist, smooth_dist in zip(self.dist_funcs, self.smooth_dist_funcs):
            self.dist_funcs_av += dist
            self.smooth_dist_funcs_av += smooth_dist
            
        self.dist_funcs_av /= self.nfiles
        self.smooth_dist_funcs_av /= self.nfiles
        # momenta and energy bins for plotting
        self.momenta_bins = np.linspace(self.p_min, self.p_max, self.res, endpoint = True)
        self.energy_bins = self.momenta_bins**2 / (2*me)
        self.energy_bins_keV = self.energy_bins/(e * 1e3)

        # average area under the distribution function curves. Reuired to match MB normalisation to EPOCH
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
        
        # get distribution functions
        self.read_dist_data()
        
        # whether to plot smoothed functions or raw data
        if smooth:
            self.plot_dist_funcs = self.smooth_dist_funcs
            self.plot_dist_funcs_av = self.smooth_dist_funcs_av
        else:
            self.plot_dist_funcs = self.dist_funcs
            self.plot_dist_funcs_av = self.dist_funcs_av
        
        # normalise moenta by the thermal momenta
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
        
        # plot details
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
         
        # plot hot electron energy tail
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