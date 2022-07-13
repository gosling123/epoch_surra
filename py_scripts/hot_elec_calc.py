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
        counts[i] = exp_term
    density = counts / (np.sum(counts)*np.diff(energy)[0])
    return density


        

        
   
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
        self.interp_len = 100
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
            plt.ylim(1e-4, 1e0)
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
        #look into changing that scaling
        idx = np.where(error_per > 100)
        # print(idx)
        if len(idx[0])>20:
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

        if len_parts < self.interp_len//20:
            n = self.interp_len//20
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

        temps = np.linspace(1, 200, 1000)
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
        nplots = 5
        if av:
            nfits = np.arange(1, nplots)
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