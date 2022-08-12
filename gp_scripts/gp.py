from utils import *
from scipy.ndimage import gaussian_filter

class LPI_GP_1D:
    """Class implementing a Gaussian Process"""
    
    def __init__(self, input_file = None, input_type = None, output_type = None, output_file = None, var_file = None):
        
        self.input_file = input_file
        self.input_type = input_type
        self.output_file = output_file
        self.output_type = output_type
        self.var_file = var_file

    def get_input(self):
        os.path.exists(self.input_file)
        with open(self.input_file, 'r') as f:
            train_inputs = json.load(f)
        train_inputs = np.array(train_inputs)
        n = train_inputs.shape[0]
        input = np.zeros(n)
        if self.input_type == 'I':
            for i in range(n):
                input[i] = train_inputs[i][0]
        elif self.input_type == 'Ln':
            for i in range(n):
                input[i] = train_inputs[i][1]
        else:
            print('ERROR: Please set input type to either I or Ln (str)')
            return None
        return input

    def get_noise_var(self):
        os.path.exists(self.var_file)
        with open(self.var_file, 'r') as f:
            train_outputs = json.load(f)
        train_outputs = np.array(train_outputs)
        n = train_outputs.shape[0]
        noise_var = np.zeros(n)
        if self.output_type == 'P':
            for i in range(n):
                noise_var[i] = train_outputs[i][0]
        elif self.output_type == 'T':
            for i in range(n):
                noise_var[i] = train_outputs[i][1]
        elif self.output_type == 'E':
            for i in range(n):
                noise_var[i] = train_outputs[i][2]
        else:
            print('ERROR: Please set output type to either P, T or E (str)')
            return None
        return noise_var

    def get_output(self):
        os.path.exists(self.output_file)
        with open(self.output_file, 'r') as f:
            train_outputs = json.load(f)
        train_outputs = np.array(train_outputs)
        n = train_outputs.shape[0]
        output = np.zeros(n)
        if self.output_type == 'P':
            for i in range(n):
                output[i] = train_outputs[i][0]
        elif self.output_type == 'T':
            for i in range(n):
                output[i] = train_outputs[i][1]
        elif self.output_type == 'E':
            for i in range(n):
                output[i] = train_outputs[i][2]
        else:
            print('ERROR: Please set output type to either P, T or E (str)')
            return None
        return output
    
    def scale_input(self, X = None, rescale = False):
        input = self.get_input()
        if X is not None:
            res = scale_axis(X, input, rescale = rescale)
            return res
        else:
            res = scale_axis(input, input, rescale = rescale)
            return res
    
    def scale_output_var(self, Y = None, rescale = False):
        output_var = self.get_noise_var()
        if Y is not None:
            res = scale_axis(Y, output_var, rescale = rescale)
            return res
        else:
            res = scale_axis(output_var, output_var, rescale = rescale)
            return res
    
    def scale_output(self, Y = None, rescale = False):
        output = self.get_output()
        if Y is not None:
            res = scale_axis(Y, output, rescale = rescale)
            return res
        else:
            res = scale_axis(output, output, rescale = rescale)
            return res
    
    def set_noise_training_data(self):
        self.X_train_noise = self.scale_input()[:,None]
        self.Y_train_noise = self.scale_output_var()[:,None]

    def update_noise_GP_kern(self, l, var):
        # var = 1e-6
        self.kern_noise = GPy.kern.Exponential(input_dim=1, variance=var, lengthscale=l)
        self.K_noise = self.kern_noise.K(self.X_train_noise, self.X_train_noise)

    def update_noise_GP_weights(self, var_noise = 1e-10):
        self.L_noise = np.linalg.cholesky(self.K_noise + var_noise * np.eye(len(self.X_train_noise)))
        self.weights_noise = np.linalg.solve(self.L_noise.T, np.linalg.solve(self.L_noise, self.Y_train_noise))
    
    def update_noise_gp(self, l, var):
        self.set_noise_training_data()
        self.update_noise_GP_kern(l, var)
        self.update_noise_GP_weights()

    def get_noise_likelihood(self):
        
        y = self.Y_train_noise
        w = self.weights_noise
        K = self.K_noise
        K = np.array(K)

        sign, logdet = np.linalg.slogdet(K)

        n = len(K.diagonal())

        log_L = -0.5*np.dot(y.T, w) - 0.5*logdet - 0.5*n*np.log(2*np.pi)
    
        res = log_L
        
        return -1.0*res


    def optimise_noise_GP(self):
        ells = np.geomspace(1e-5, 1, 50)
        vars = np.geomspace(1e-5, 1, 50)
        self.log_L_noise = np.zeros((len(ells), len(vars)))
        for i, l in enumerate(ells):
            for j, v in enumerate(vars):
                self.update_noise_gp(l = l, var = v)
                self.log_L_noise[i,j] = self.get_noise_likelihood()

        idx = np.where(self.log_L_noise == np.array(self.log_L_noise).min())
        self.l_opt_noise = ells[idx[0][0]]
        self.var_opt_noise = vars[idx[1][0]]
        print('l = ' , self.l_opt_noise, 'var = ' , self.var_opt_noise)
        self.update_noise_gp(l = self.l_opt_noise, var = self.var_opt_noise)

    def noise_GP_predict(self, X_star, scaled = False, get_std = False):

        if scaled == False:
            X_star = self.scale_input(X = X_star)

        K_star_noise = self.kern_noise.K(X_star, X_star)
        k_star_noise = self.kern_noise.K(self.X_train_noise, X_star)

        f_star_noise = np.dot(k_star_noise.T, self.weights_noise)
        f_star_noise = self.scale_output_var(Y = f_star_noise, rescale = True)

        if get_std:
            v = np.linalg.solve(self.L_noise, k_star_noise)
            V_star_noise = K_star_noise - np.dot(v.T, v)
            std_epi = np.sqrt(self.scale_output_var(Y = np.diag(V_star_noise), rescale=True))
            return f_star_noise, std_epi
        else:
            return f_star_noise

    def set_training_data(self):
        self.X_train = self.get_input()[:,None]
        self.Y_train = self.get_output()[:,None]
        self.X_range = self.X_train.max()-self.X_train.min()
        self.Y_range = self.Y_train.max()-self.Y_train.min()

    def update_GP_kern(self, l, var):
        l *= self.X_range
        var *= self.Y_range
        self.noise_var = self.get_noise_var().flatten()
        self.noise_cov = np.diag(self.noise_var)

        self.kern = GPy.kern.RatQuad(input_dim=1, variance=var, lengthscale=l)
        self.K = self.kern.K(self.X_train, self.X_train)
        self.K += self.noise_cov

    def update_GP_weights(self):
        self.L = np.linalg.cholesky(self.K)
        self.weights = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.Y_train))

    def update_GP(self, l, var):
        self.set_training_data()
        self.update_GP_kern(l, var)
        self.update_GP_weights()

    def get_GP_likelihood(self):
      
        y = self.Y_train
        w = self.weights
        K = self.K
        K = np.array(K)

        sign, logdet = np.linalg.slogdet(K)

        n = len(K.diagonal())

        log_L = -0.5*np.dot(y.T, w) - 0.5*logdet - 0.5*n*np.log(2*np.pi)
    
        res = log_L
        
        return -1.0*res

   
    def optimise_GP(self):
        ells = np.geomspace(0.01, 100, 10)
        vars = np.geomspace(0.01, 100, 10)
        self.log_L = np.zeros((len(ells), len(vars)))
        for i, l in enumerate(ells):
            for j, v in enumerate(vars):
                self.update_GP(l = l, var = v)
                self.log_L[i,j] = self.get_GP_likelihood()

        idx = np.where(self.log_L == np.array(self.log_L).min())
        self.l_opt = ells[idx[0][0]]
        self.var_opt = vars[idx[1][0]]
        print('l = ', ells[idx[0][0]], 'var = ', vars[idx[1][0]])
        self.update_GP(l = self.l_opt, var = self.var_opt)
    
    def GP_predict(self, X_star, get_std = False):
        K_star = self.kern.K(X_star, X_star)
        k_star = self.kern.K(self.X_train, X_star)

        self.noise_var_star = self.noise_GP_predict(X_star, get_std=False)
        self.noise_cov_star = np.diag(self.noise_var_star.flatten())
        f_star = np.dot(k_star.T, self.weights)
        if get_std:
            v = np.linalg.solve(self.L, k_star)
            V_star_epi = K_star - np.dot(v.T, v)
            std_epi  = np.sqrt(np.diag(V_star_epi))
            V_star_noise = self.noise_cov_star
            std_noise  = np.sqrt(np.diag(V_star_noise))
            return f_star, std_epi, std_noise
        else:
            return f_star


def GP_1D_predict_all(X_star, input_file, input_type, output_file, var_file):
    output_types = ['P', 'T', 'E']
    label = ['Reflectivity', 'Hot-Elctron Temperature', 'Fraction E>50 keV']
    predicted_output = np.zeros((len(output_types), len(X_star)))
    epistemic_error = np.zeros((len(output_types), len(X_star)))
    noise_error = np.zeros((len(output_types), len(X_star)))
    for i, out in enumerate(output_types):
        start = time.time()
        print(f'Generating output = {label[i]} GP')
        gp = LPI_GP_1D(input_file = input_file, input_type = input_type,\
                            output_file = output_file, output_type = out,\
                             var_file = var_file)
        print('Optimiszing Noise GP')
        gp.optimise_noise_GP()
        print('Optimiszing Output GP')
        gp.optimise_GP()
        print('Making predictions for X_star')
        Y_star, sig_epi, sig_noise = gp.GP_predict(X_star, get_std=True)
        predicted_output[i] = Y_star
        epistemic_error[i] = 2.0*sig_epi
        noise_error[i] = 2.0*sig_noise
        print(f'Finished {out} GP regression in {(time.time() - start)/60} minutes')
        print('--------------------------------------------------------------------')