from utils import *
from plasma_calc import *

## max_boltz_E
#
# Maxwell Boltzman distribution in energy. 
# Normalised in the same way as np.histogram for integral to equal one.
# @param momenta : momentum array
# @param temp : temperature (kelvin)
def max_boltz_E(energy, temp):
    counts = np.zeros(len(energy))
    for i in range(len(energy)):
        exp_term = np.exp(-energy[i] / (kB * temp))
        counts[i] = energy[i]*exp_term
    density = counts / (np.sum(counts)*np.diff(energy)[0])
    return density

   

        
   
class hot_electron:
    
    ## __init__
    #
    # The constructor
    # @param self : The object pointer
    # @param dir : Directory where data is stored (str)
    def __init__(self, dir):
        # Enforce that dir is a string
        if not isinstance(dir,str):
            raise Exception("Error: dir argument must be a string (name of the directory housing the output data and input.deck")
        self.directory = dir+'/' # Directory to look into 
        self.epoch_data = Laser_Plasma_Params(dir) # Laser_Plasma_Params class to get certain variables
        # call functions needed to get certain stored variables in above class
        self.epoch_data.read_data()
        self.epoch_data.get_spatio_temporal()
        self.epoch_data.get_plasma_param()
        self.nbins = 1000 # Number of bins for histogram/distribution plot

    
    ## get_flux_dist
    #
    # Finds the distribution function in terms of kinetic energy.
    # The distribution function at each bin is weighted by the energy.
    # @param self : The object pointer
    # @param plot : (Logical) Plots the distribution function and equilibrium maxwellian normalised by their maxmium values.
    # @param log : (Logical) Plot y-axis set to log scale
    def get_flux_dist(self,  plot = False, log = False):
        files = glob.glob(self.directory+'probes_*.sdf') # provbe data files (probe near right-hand/exit boundary )
        self.E_data = np.array([]) # store energies of passing electrons
        for file in files:
            data = sdf.read(file)
            px = data.outgoing_e_probe__Px.data # Attain momenta in direction of travel
            gamma = np.sqrt(1 + (px/(me*c))**2) # Lorentz fator
            E = (gamma - 1)*me*c**2 # Convert to Kinetic energy (relativistic)
            self.E_data = np.concatenate((self.E_data, E), axis=None)

        # Get distribution using histogram, choose data to normalise such that the integral = 1   
        self.E_dist, bins = np.histogram(self.E_data, bins = self.nbins, density=True)

        # convert bin edges into bin centres
        self.E_bins = np.zeros(len(bins)-1)
        for i in range(1, len(bins)):
            delta = 0.5*(bins[i] - bins[i-1]) # grid centres
            self.E_bins[i-1] = bins[i] - delta # fill nodal posistions
        
        # Weight f(E) by E
        self.E_dist *= self.E_bins

        # plot                 
        if plot:
            # Maxwellian at intiliased/background plasma temperature
            MB = max_boltz_E(self.E_bins, self.epoch_data.Te*keV_to_K)
            plt.plot(self.E_bins*J_tot_KeV, self.E_dist/self.E_dist.max(), label = 'Raw Data')
            plt.plot(self.E_bins*J_tot_KeV, MB/MB.max(), label = 'MB', linestyle = '--', color = 'black')
            plt.legend()
            if log:
                plt.yscale('log')
            plt.xlabel(r'$E_e (keV)$')
            plt.ylabel(r'$f_e(E)$')
            plt.ylim(1e-4, 1e0)
            plt.gcf().set_size_inches(10,6)
            return None
        else:
            return self.E_bins, self.E_dist
    
    ## split_dist
    #
    # Finds the hot electron tail of the distribution, and splits it into n segments.
    # @param self : The object pointer
    # @param n : Number of segments.
    # @param plot : (Logical) Plots the split segemnts.
    # @param log : (Logical) Plot y-axis set to log scale
    def split_dist(self, n, plot = False, log = False):
        # Enforce that n is a positive integer
        if not isinstance(n,int) or (n < 1):
            raise Exception("ERROR: n argument must be an integer > 0")

        # get E and f(E)
        E, flux = self.get_flux_dist()
        # Maxwellian at intiliased/background plasma temperature
        MB_eq = max_boltz_E(E, self.epoch_data.Te*keV_to_K)
        # Normalised agaisnt maxima as we only care for the shape and not the magnitude.
        MB_eq_norm = MB_eq/MB_eq.max() 
        flux_norm = flux/flux.max()

        # Estimate where the probe distribution deviates significantly from the background maxwellian
        error_per = np.abs((flux_norm -  MB_eq_norm) / MB_eq_norm)*100
        idx = np.where(error_per > 500) # Some tolerance
        # Need at least 20 points for sensible fit (this can be changed it's just arbitrary)
        if len(idx[0])>self.nbins//20:
            idx = idx[0][0]
        else:
            idx = 0

        # Use this location to leave only the hot electron tail
        self.idx = idx
        flux_cut = flux[self.idx:]
        E_cut = E[self.idx:]
        # Smooth data to make it better to fit an analytic curve to (uniform1D filter)
        flux_final = moving_av(flux_cut, len(flux_cut), len(flux_cut)//200)
        # Don't want to fit maxwellian to bump on tail region
        df = np.diff(flux_final)
        tp = np.where(df == df.min())[0][0] # Estimate turning point from flat area to maxwellian like curve
        # Data with flat area mostly removed (chose to move tp slightly further back to not cut out any wanted data)
        flux_final = flux_final[int(0.75*tp):]
        E_final = E_cut[int(0.75*tp):]

        # Split distribution into n parts
        self.flux_parts = []
        self.E_parts = []
        len_parts = len(flux_final)//n
        # Want at least 20 points to a fit
        if len_parts < len(flux_final)//20:
            n = len(flux_final)//20
            print('Number of points chosen to fit maxewellian to deemed to low, \
                   setting to minimum value')
        for i in range(n):
            f = flux_final[i*len_parts:(i+1)*len_parts]
            e = E_final[i*len_parts:(i+1)*len_parts]
            self.flux_parts.append(f)
            self.E_parts.append(e)
        self.flux_parts = np.array(self.flux_parts)
        self.E_parts = np.array(self.E_parts)

        # Plot result
        if plot:
            for e, f in zip(self.E_parts, self.flux_parts):
                plt.plot(e*J_tot_KeV, f)
            if log:
                plt.yscale('log')
            plt.xlabel(r'$E_e (keV)$')
            plt.ylabel(r'$f_e(E)$')
            plt.gcf().set_size_inches(10,6)
            return None
        else:
            return self.E_parts, self.flux_parts

    ## fit_maxwellians
    #
    # Fit Maxwellian like distribution to each segemnt and estimate the best fitting
    # amplitude and temperature to the data (brute force).
    # @param self : The object pointer
    # @param n : Number of segments.
    # @param plot : (Logical) Plots the fits and loss functions.
    # @param log : (Logical) Plot y-axis set to log scale
    def fit_maxwellians(self, n = 5, plot = False, log = False):  
        # Enforce that n is a positive integer      
        if not isinstance(n,int) or (n < 1):
            raise Exception("ERROR: n argument must be an integer > 0")

        # Get E and f(E) hot electron parts
        E_parts, flux_parts = self.split_dist(n = n)
        # Range of temperatures to fit
        temps = np.linspace(self.epoch_data.Te, 500, 1000)
        n = len(flux_parts)
        # Loss function values
        loss = np.zeros((n, len(temps)))
        count = 0
        # Get loss function values
        for e,f in zip(E_parts,flux_parts):
            res = []
            for i,t in enumerate(temps):
                fit = max_boltz_E(e, t*keV_to_K)
                res.append(loss_func(fit/fit.max(), f/f.max()))
            loss[count] = res
            count += 1

        # Find best fit temperature vales from where losss is minimum
        self.T_vals = np.array([])
        for l in loss:
            idx = np.where(l == l.min())[0][0]
            self.T_vals = np.append(self.T_vals, temps[idx])

        # For the best fit temperatures, find the required scale factor to match magnitude of the
        # two distributions. This is the amplitude of the fit.
        count = 0
        self.scaled_fits = [] # New fits split into respective segments
        self.MB_fits = [] # Original MB fits
        self.amplitudes = np.array([])
        self.scaled_fits_full = [] # Concatenated new fit
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

        # Plot results
        if plot:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            for l in loss:
                ax1.plot(temps, l)
                ax1.set_xlabel(r'$T (keV)$')
                ax1.set_ylabel(r'Normed Loss Function')
            
            for i,f in enumerate(self.scaled_fits):
                ax2.plot(E_parts[i]*J_tot_KeV, f, label = r'$A$ = ' + str(np.round(self.amplitudes[i], 8)) + ', $T$ = ' + str(np.round(self.T_vals[i], 3)) + ' keV')
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

    ## get_hot_e_temp
    #
    # Get singular estimate of hot electron temperature from the Maxwellian fits
    # Found from weighted average of the found best fitting temperatures. The amplitudes
    # of the fits are used as the weights.
    # @param self : The object pointer
    # @param n : Number of plots.
    # @param av : (Logical) Finds the average value of temperature for all fits ranging from 1 to n.
    def get_hot_e_temp(self, n = 5, av = True, plot = False):
        if not isinstance(n,int) or (n < 1):
            raise Exception("ERROR: n argument must be an integer > 0")

        # Find average temperature for just one fitting  scheme of n segments
        if av == False:
            T_vals, A, fits, fits_full = self.fit_maxwellians(n = n, plot = False)
            self.T_hot = np.average(T_vals, weights = A)
        # Find average of the found singular tempeartures for 1 to n segment fits
        if av:
            nfits = np.linspace(1, n+1)
            T_data = []
            for i in range(1, n+1):
                T_vals, A, fits, fits_full = self.fit_maxwellians(n = int(i), plot = False)
                T_est = np.average(T_vals, weights = A)
                T_data.append(T_est)
            self.T_hot_av = np.average(T_data[:])
        # Plot T vs number of fits
        if plot:
            plt.plot(nfits, T_data, '-o', label = 'Data')
            plt.xlabel(r'$N$ Fits')
            plt.ylabel(r'$T_{hot}$')
            plt.xlim(1, n)
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

    ## get_energy_frac
    #
    # Estimate fraction of energy going to the hot electron population.
    # Estimated as the ratio of the area of the hot electron region to the whole region.
    # @param self : The object pointer
    def get_energy_frac(self):
        E_h, f_h = self.split_dist(n = 1) # Hot electron region
        E, f = self.get_flux_dist() # Whole domain
        if self.idx == 0:
            E_hot_frac = 0 # If there is said to be no hot electron region, set to zero
            return self.E_hot_frac
        else:
            E_hot_frac = np.trapz(y=f[self.idx:]*E[self.idx:], x =E[self.idx:]) / np.trapz(y=f*E, x =E)
            return E_hot_frac

    ## get_energy_frac_bound
    #
    # Estimate fraction of energy for elecron energies between set bounds.
    # @param self : The object pointer
    # @param bounds : One dimensional array of length two that specify lower and upper energy bound.
    def get_energy_frac_bound(self, bounds):
        if np.size(bounds) != 2:
            raise ValueError('ERROR: bounds must specify start and stop energies')
        if bounds[1] < bounds[0]:
            raise ValueError('ERROR: bounds must be in ascending order')
        if bounds[1] < 0 or bounds[0] < 0:
            raise ValueError('ERROR: bounds must be greater than 0')
        E, f = self.get_flux_dist()
        idx1 = np.where(bounds[0] <= E*J_tot_KeV)[0]
        if len(idx1) == 0:
            return 0.0
        else:
            idx1 = idx1[0]
        idx2 = np.where(E*J_tot_KeV <= bounds[1])[0][-1]
        res = np.trapz(y=f[idx1:idx2], x =E[idx1:idx2]) / np.trapz(y=f, x =E)
        return res