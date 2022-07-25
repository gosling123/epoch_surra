from utils import *
from plasma_calc import *

## winsincFIR
#
# Windowed sinc filter function (http://www.dspguide.com/ch16/2.htm (Equation 16-4))
# @param omega_c  Cutoff frequency
# @param omega_s  Sampling rate (sampling frequency)
# @param M   Length of the filter kernel (must be odd)
def winsincFIR(omega_c,omega_s,M):
    # cutoff frequency shoudl be a fraction of sampling frequency
    ker = np.sinc((omega_c / omega_s) * (np.arange(M) - (M - 1)/2))
    # Blackman window used for smooting filter
    ker *= np.blackman(M)
    # unit gain at zero frequency 
    ker /= np.sum(ker) 
    return ker
    
## bandpass
#
# Create a band-pass filter by convolving a high-pass and a low-pass filter
# @param w0  Central frequency you want to filter around (fraction of omega0)
# @param bw  Total bandwidth of your filter (fraction of omega0)
# @param M  Half filter length (must be odd)
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


## EM_fields Class.
#
# Class that reads and calculates field quantities from fields_ output files.         
class EM_fields:

    ## __init__
    #
    # The constructor
    # @param self  The object pointer
    # @param dir  Directory where data is stored (str)
    def __init__(self, dir):
        self.directory = dir+'/' # Directory to look into
        self.nfiles = len(glob.glob(self.directory+'fields_*.sdf'))
        initial_data = sdf.read(self.directory+'fields_0000.sdf')
        self.nx = len(initial_data.Electric_Field_Ey.data)
        self.timesteps = self.nfiles # Number of timesteps
        # Read in plasma data class
        self.epoch_data = Laser_Plasma_Params(dir = dir)
        self.epoch_data.get_plasma_param()
        self.epoch_data.get_spatio_temporal()
    
    ## get_2D_Electric_Field
    #
    # Get time and space data of E field i.e E(x,t)
    # @param self  The object pointer
    # @param ax  String to set direction you want (i.e x,y,z)
    def get_2D_Electric_Field(self, ax):
        if ax == 'x':
            field_str = 'Electric Field/Ex' # Ex(x,t)
        elif ax == 'y':
            field_str = 'Electric Field/Ey' # Ey(x,t)
        elif ax == 'z':
            field_str = 'Electric Field/Ez' # Ez(x,t)
        else:
            return print('ERROR: Please set ax to either x, y or z' )
        # create space-time electric field array
        E = np.zeros((self.nx, self.timesteps))
        for i in range(self.nfiles):
            fname = self.directory+'/fields_'+str(i).zfill(4)+'.sdf'
            data = sdf.read(fname, dict = True)
            field = data[field_str].data
            E[:, i] = field
        return E
    
    ## get_2D_Magnetic_Field
    #
    # Get time and space data of B field i.e B(x,t)
    # @param self  The object pointer
    # @param ax  String to set direction you want (i.e x,y,z)
    def get_2D_Magnetic_Field(self, ax):
        if ax == 'x':
            field_str = 'Magnetic Field/Bx' # Bx(x,t)
        elif ax == 'y':
            field_str = 'Magnetic Field/By' # By(x,t)
        elif ax == 'z':
            field_str = 'Magnetic Field/Bz' # Bz(x,t)
        else:
            return print('ERROR: Please set ax to either x, y or z' )
        # create space-time magnetic field array
        B = np.zeros((self.nx, self.timesteps))
        for i in range(self.nfiles):
            fname = self.directory+'/fields_'+str(i).zfill(4)+'.sdf'
            data = sdf.read(fname, dict = True)
            field = data[field_str].data
            B[:, i] = field
        return B

   
    ## get_2D_FFT
    #
    # Get 2D FFT (i.e space and time) for specific field
    # @param self  The object pointer
    # @param field  EM Field to FFT (inputs are either 'Ex', 'Ey', 'Bz')
    # @param square_mod  (Logical) outputs the sqaured modulus of the FFT 
    def get_2D_FFT(self, field, square_mod = True):
        if field == 'Ex':
            array = self.get_2D_Electric_Field(ax = 'x') # Ex(x,t) field
        elif field == 'Ey':
            array = self.get_2D_Electric_Field(ax = 'y') # Ey(x,t) field
        elif field == 'Ez':
            array = self.get_2D_Electric_Field(ax = 'z') # Ez(x,t) field
        if field == 'Bx':
            array = self.get_2D_Magnetic_Field(ax = 'x') # Bx(x,t) field
        elif field == 'By':
            array = self.get_2D_Magnetic_Field(ax = 'y') # By(x,t) field
        elif field == 'Bz':
            array = self.get_2D_Magnetic_Field(ax = 'z') # Bz(x,t) field
        else:
            print('ERROR: Please set field to either Ex, Ey, Ez, Bx, By or Bz' )

        # 2D FFT
        tilde_array= np.fft.fft2(array)
        tilde_array = np.fft.fftshift(tilde_array)

        # Reverse second axis as we want right travelling wave to correspond to positive omega
        tilde_array = tilde_array[:, ::-1]

        # return the squared modulus of the 2D FFT
        if square_mod:
            square_mod_res = (np.abs(tilde_array))**2
            return square_mod_res
        else:
            return tilde_array

    ## get_time_FFT
    #
    # Produces 1D time FFT for specific field
    # @param self  The object pointer
    # @param field  EM Field to FFT (inputs are either 'Ex', 'Ey', 'Bz')
    # @param square_mod  (Logical) outputs the sqaured modulus of the FFT
    def get_time_FFT(self, field, square_mod = True):
        if field == 'Ex':
            array = self.get_2D_Electric_Field(ax = 'x') # Ex(x,t) field
        elif field == 'Ey':
            array = self.get_2D_Electric_Field(ax = 'y') # Ey(x,t) field
        elif field == 'Ez':
            array = self.get_2D_Electric_Field(ax = 'z') # Ez(x,t) field
        if field == 'Bx':
            array = self.get_2D_Magnetic_Field(ax = 'x') # Bx(x,t) field
        elif field == 'By':
            array = self.get_2D_Magnetic_Field(ax = 'y') # By(x,t) field
        elif field == 'Bz':
            array = self.get_2D_Magnetic_Field(ax = 'z') # Bz(x,t) field
        else:
            print('ERROR: Please set field to either Ex, Ey, Ez, Bx, By or Bz' )

        # 1D FFT in time
        time_FFT = np.zeros((self.nx, self.timesteps), dtype = 'complex_')
        for i in range(0, self.nx):
            time_FFT[i,:] = fft(array[i,:])

        # return the squared modulus of the 1D FFT
        if square_mod:
            square_mod_res = (np.abs(time_FFT))**2
            return square_mod_res
        else:
            return time_FFT 

    ## get_space_FFT
    #
    # Produces 1D space FFT for specific field
    # @param self  The object pointer
    # @param field  EM Field to FFT (inputs are either 'Ex', 'Ey', 'Bz')
    # @param square_mod  (Logical) outputs the sqaured modulus of the FFT
    def get_space_FFT(self, field, square_mod = True):
        if field == 'Ex':
            array = self.get_2D_Electric_Field(ax = 'x') # Ex(x,t) field
        elif field == 'Ey':
            array = self.get_2D_Electric_Field(ax = 'y') # Ey(x,t) field
        elif field == 'Ez':
            array = self.get_2D_Electric_Field(ax = 'z') # Ez(x,t) field
        if field == 'Bx':
            array = self.get_2D_Magnetic_Field(ax = 'x') # Bx(x,t) field
        elif field == 'By':
            array = self.get_2D_Magnetic_Field(ax = 'y') # By(x,t) field
        elif field == 'Bz':
            array = self.get_2D_Magnetic_Field(ax = 'z') # Bz(x,t) field
        else:
            print('ERROR: Please set field to either Ex, Ey, Ez, Bx, By or Bz')

        # Transpose to get field in the form (t,x) rather than (x,t)
        array = array.T 
        # 1D FFT in space
        space_FFT = np.zeros((self.timesteps, self.nx), dtype = 'complex_')
        for i in range(0, self.timesteps):
            space_FFT[i,:] = fft(array[i,:])

        # return the squared modulus of the 1D FFT
        if square_mod:
            square_mod_res = (np.abs(space_FFT))**2
            return square_mod_res
        else:
            return space_FFT

    ## get_filtered_signals
    #
    # Finds filtered signals of Ey and Bz fields (either laser signal or SRS signal)
    # @param self  The object pointer
    # @param laser  (Logical) Whether to output laser sginal (true) or SRS signal (false)
    # @param plot_E  (Logical) Whether to plot the filter result at set grid point to test if it works (Ey field)
    # @param plot_B  (Logical) Whether to plot the filter result at set grid point to test if it works (Bz field)
    def get_filtered_signals(self, laser = False, plot_E = False, plot_B = False):
        # required fields
        Ey = self.get_2D_Electric_Field(ax = 'y') # Ey(x,t) field
        Bz = self.get_2D_Magnetic_Field(ax = 'z') # Bz(x,t) field

        n,m = Ey.shape # array size
        omega = 5.36652868179e+15 # laser frequency
        omega_0 = 1.0 # normalised laser frequency 
        omega_bw = 0.3 # bandswidth centred at laser frequency
        T_end = self.epoch_data.t_end # sim end time
        N = self.timesteps # number of time steps
        dt = T_end/N # time step
        omegaNyq = pi/dt
        omega_s = 2*pi/dt # sampling frequency 
        M = 1001 # half length of the filter kernel (must be odd) 

        h = bandpass(omega_0,omega_bw,omegaNyq,M) #bandpass filter

       
        # Laser signals
        Ey_laser = np.zeros((n, m))
        Bz_laser = np.zeros((n, m))

        # SRS signals
        Ey_SRS = np.zeros((n, m))
        Bz_SRS = np.zeros((n, m))

        # Fill arrays with data
        for i in range(n):
            # laser signals
            Ey_laser[i, :] = np.convolve(Ey[i,:],h,mode='same')
            Bz_laser[i, :] = np.convolve(Bz[i,:],h,mode='same')
            # SRS signals
            Ey_SRS[i, :] = Ey[i,:] - Ey_laser[i,:]
            Bz_SRS[i, :] = Bz[i,:] - Bz_laser[i,:]

        # plots
        if plot_E:
            i = 1
      
            fft_fre = np.fft.fftfreq(N,1/omega_s) / omega
            t = np.linspace(0,T_end,N)
            # plot the signal and its Fourier transform
            fig,ax = plt.subplots(3,2,figsize=(17,12))
            ax[0,0].plot(t*1e12, Ey[i]*1e-11)
            fftsig = fft(Ey[i])
            fft_plot = moving_av(np.abs(fftsig), span = len(fft_fre) , period = len(fft_fre)//1)
            ax[0,1].scatter(fft_fre, fft_plot)
            # plot laser signal and its fourier transform
            laser_sig = Ey_laser[i]
            fft_laser = moving_av(np.abs(fft(laser_sig)), span = len(fft_fre) , period = len(fft_fre)//1)
            ax[1,0].plot(t*1e12,laser_sig*1e-11)
            ax[1,1].scatter(fft_fre,fft_laser)
            # plot SRS signal and its fourier transform
            scatter_sig = Ey_SRS[i]
            fft_srs = moving_av(np.abs(fft(scatter_sig)), span = len(fft_fre) , period = len(fft_fre)//1)
            ax[2,0].plot(t*1e12,scatter_sig*1e-11)
            ax[2,1].scatter(fft_fre,fft_srs)
            # plot details
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
            i = 1

            fft_fre = np.fft.fftfreq(N,1/omega_s) / omega
            t = np.linspace(0,T_end,N)
            # plot the signal and its Fourier transform
            fig,ax = plt.subplots(3,2,figsize=(17,12))
            ax[0,0].plot(t*1e12, Bz[i])
            fftsig = fft(Bz[i])
            ax[0,1].plot(fft_fre,np.abs(fftsig))
            # plot laser signal and its fourier transform
            laser_sig = Bz_laser[i]
            ax[1,0].plot(t*1e12,laser_sig)
            ax[1,1].plot(fft_fre,np.abs(fft(laser_sig)))
            # plot SRS signal and its fourier transform
            scatter_sig = Bz_SRS[i]
            ax[2,0].plot(t*1e12,scatter_sig)
            ax[2,1].plot(fft_fre,np.abs(fft(scatter_sig)))
            # plot details
            ax[2,0].set_xlabel(r'Time (ps)')
            ax[2,0].set_ylabel(r'$B_z$')
            ax[1,0].set_ylabel(r'$B_z$')
            ax[0,0].set_ylabel(r'$B_z$')
            ax[0,1].set_xlim(0, 1.2)
            ax[2,1].set_xlim(0, 1.2)
            ax[1,1].set_xlim(0, 1.2)
            ax[2,1].set_xlabel(r'$\omega/\omega_0$')
            ax[0,1].set_ylabel(r'$FFT\left|B_z(t)\right|$')
            ax[1,1].set_ylabel(r'$FFT\left|B_z(t)\right|$')
            ax[2,1].set_ylabel(r'$FFT\left|B_z(t)\right|$')
            ax[0,1].set_yscale('log')
            ax[1,1].set_yscale('log')
            ax[2,1].set_yscale('log')
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
    # @param self  The object pointer
    # @param signal  String to set what signal is requires (either 'bsrs', 'fsrs' or 'laser')
    # @param plot_E  (Logical) Whether to output the Sx time series (true) or the time average (false)     
    def get_flux(self, signal = 'bsrs', time_series = False):
        # get required field signals
        if signal == 'bsrs' or signal == 'fsrs':
            Ey, Bz = self.get_filtered_signals(laser = False) # Filtered Ey and Bz fields
        elif signal == 'laser':
            Ey, Bz = self.get_filtered_signals(laser = True) # Filtered Ey and Bz fields
        else:
            return print('please set signal to either bsrs, fsrs or laser')
            
        W_cm2 = 1e4 # Convert to W_cm2
        factor = mu0*W_cm2 # Denominator of Sx
        S = Ey*Bz/factor # poynting flux
        # return full signal over all space and time
        if time_series:
            return S
        # integrate/average over time at each grid point
        else:
            if signal == 'laser':
                sum_ = np.zeros(self.nx)
                for i in range(self.timesteps):
                    sig = S[:,i]
                    indx = np.where(sig < 0) # only care for forawrd travelling flux
                    sig[indx] = 0
                    sum_ += sig
                S_av = np.abs(sum_)/self.timesteps
            elif signal == 'fsrs':
                sum_ = np.zeros(self.nx)
                for i in range(self.timesteps):
                    sig = S[:,i]
                    indx = np.where(sig < 0) # only care for forawrd travelling flux
                    sig[indx] = 0
                    sum_ += sig
                S_av = np.abs(sum_)/self.timesteps
            elif signal == 'bsrs':
                sum_ = np.zeros(self.nx)
                for i in range(self.timesteps):
                    sig = S[:,i]
                    indx = np.where(sig > 0) # only care for backward travelling flux
                    sig[indx] = 0
                    sum_ += sig
                S_av = np.abs(sum_)/self.timesteps
            return S_av
        
    ## get_flux_grid_av
    #
    # Averages Poynting flux over ncells (near LH boundary for backscatter SRS and RH boundary for laser)
    # @param self  The object pointer
    # @param ncells  Number of cells to average over (default 50)
    # @param signal  String to set what signal is requires (either 'bsrs', 'fsrs' or 'laser')
    def get_flux_grid_av(self, ncells = 10, signal = 'bsrs'):
        St_av = np.abs(self.get_flux(signal = signal, time_series = False))
        # for forward travelling signals, we want to average close to the right-hand boundary
        if signal == 'laser' or signal == 'fsrs':
            sum_ = 0
            for i in range(self.nx - ncells, self.nx):
                sum_ += St_av[i] 
        # for backward travelling signals, we want to average close to the left-hand boundary
        else:
            sum_ = 0
            for i in range(ncells):
                sum_ += St_av[i]
        S_av = sum_/ncells
        return S_av