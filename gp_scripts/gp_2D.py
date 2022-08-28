from utils import *
from scipy.ndimage import gaussian_filter

class LPI_GP_2D:
    """Class implementing a Gaussian Process"""
    
    def __init__(self, input_file = None, output_type = None,\
                 output_file = None, var_file = None, train_frac = 0.4):
        
        self.input_file = input_file
        self.output_file = output_file
        self.var_file = var_file
        self.output_type = output_type
        self.train_frac = train_frac

    def get_input(self):
        os.path.exists(self.input_file)
        with open(self.input_file, 'r') as f:
            train_inputs = json.load(f)
        train_inputs = np.array(train_inputs)
        n = train_inputs.shape[0]
        input = []
        for i in range(n):
            input.append(train_inputs[i])
        return np.array(input)

    def get_noise_var(self, log = True):
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
                noise_var[i] = train_outputs[i][2] + 1e-12
        else:
            print('ERROR: Please set output type to either P, T or E (str)')
            return None
        if log:
            return np.log(noise_var)
        else:
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
                output[i] = train_outputs[i][2] + 1e-6
        else:
            print('ERROR: Please set output type to either P, T or E (str)')
            return None
        return output

    def set_training_data(self):
        X = self.get_input()
        Y = self.get_output()
        noise = self.get_noise_var()

        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = self.train_frac)

        idxs_train = []
        idxs_test = []
        for i in range(len(X_train)):
            mask = np.isin(X, X_train[i])
            mask = mask[:,0] & mask[:,1]
            indx = np.where(mask == True)[0][0]
            idxs_train.append(indx)
        for i in range(len(X_test)):
            mask = np.isin(X, X_test[i])
            mask = mask[:,0] & mask[:,1]
            idx = np.where(mask == True)[0][0]
            idxs_test.append(idx)

        noise_train = noise[idxs_train]
        noise_test = noise[idxs_test]

        self.X_train = X_train
        self.Y_train = Y_train[:,None]
        self.noise_train = noise_train
        self.X_test = X_test
        self.Y_test = Y_test[:,None]
        self.noise_test = noise_test

        self.X1_range = X[:,0].max()-X[:,0].min()
        self.X2_range = X[:,1].max()-X[:,1].min()
        self.Y_range = Y.max()-Y.min()
        self.noise_range = noise.max() - noise.min()    



    def update_noise_GP_kern(self, l1, l2, var):
        l1 *= self.X1_range
        l2 *= self.X2_range
        var *= self.Y_range**2
        self.kern_noise = GPy.kern.Exponential(input_dim=2, variance=var, lengthscale=[l1, l2], ARD=True)
        self.K_noise = self.kern_noise.K(self.X_train, self.X_train)

    def update_noise_GP_weights(self, var_noise = 1e-4):
        self.L_noise = np.linalg.cholesky(self.K_noise + var_noise * self.noise_range**2 * np.eye(len(self.X_train)))
        self.weights_noise = np.linalg.solve(self.L_noise.T, np.linalg.solve(self.L_noise, self.noise_train))
    
    def update_noise_gp(self, l1, l2, var):
        self.update_noise_GP_kern(l1, l2, var)
        self.update_noise_GP_weights()

    def get_noise_likelihood(self):
        
        y = self.noise_train
        w = self.weights_noise
        K = self.K_noise
        K = np.array(K)

        sign, logdet = np.linalg.slogdet(K)

        n = len(K.diagonal())

        log_L = -0.5*np.dot(y.T, w) - 0.5*logdet - 0.5*n*np.log(2*np.pi)
    
        res = log_L
        
        return -1.0*res


    def optimise_noise_GP(self):
        ells_1 = np.geomspace(1e-5, 1.0, 10)
        ells_2 = np.geomspace(1e-5, 1.0, 10)
        vars = np.geomspace(1e-5, 1.0, 10)
        self.log_L_noise = np.zeros((len(ells_1), len(ells_2) , len(vars)))
        for i, l1 in enumerate(ells_1):
            for j, l2 in enumerate(ells_2):
                for k, v in enumerate(vars):
                    self.update_noise_gp(l1 = l1, l2 = l2, var = v)
                    self.log_L_noise[i,j,k] = self.get_noise_likelihood()

        idx = np.where(self.log_L_noise == np.array(self.log_L_noise).min())
        self.l1_opt_noise = ells_1[idx[0][0]]
        self.l2_opt_noise = ells_2[idx[1][0]]
        self.var_opt_noise = vars[idx[2][0]]
        print('l1 = ' , self.l1_opt_noise, 'l2 = ' , self.l2_opt_noise, 'var = ' , self.var_opt_noise)
        self.update_noise_gp(l1 = self.l1_opt_noise, l2 = self.l2_opt_noise, var = self.var_opt_noise)

    def noise_GP_predict(self, X_star, get_err = False):
        K_star_noise = self.kern_noise.K(X_star, X_star)
        k_star_noise = self.kern_noise.K(self.X_train, X_star)
        f_star_noise = np.dot(k_star_noise.T, self.weights_noise)
        f_star_noise = np.exp(f_star_noise)
        if get_err:
            v_noise = np.linalg.solve(self.L_noise, k_star_noise)
            V_star_noise = K_star_noise - np.dot(v_noise.T, v_noise)
            V_noise = f_star_noise**2 * np.diag(V_star_noise)
            err = 2.0*np.sqrt(V_noise)
            return f_star_noise.flatten(), err.flatten()
        else:
            return f_star_noise.flatten()



    def update_GP_kern(self, l1, l2, var):
        l1 *= self.X1_range
        l2 *= self.X2_range
        var *= self.Y_range**2
        self.noise_var = self.noise_GP_predict(X_star=self.X_train)
        self.noise_cov = np.diag(self.noise_var)

        self.kern = GPy.kern.Exponential(input_dim=2, variance=var, lengthscale=[l1, l2], ARD=True)
        self.K = self.kern.K(self.X_train, self.X_train)
        self.K += self.noise_cov

    def update_GP_weights(self):
        self.L = np.linalg.cholesky(self.K)
        self.weights = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.Y_train))

    def update_GP(self, l1, l2, var):
        self.update_GP_kern(l1, l2, var)
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
        ells_1 = np.geomspace(0.01, 100, 10)
        ells_2 = np.geomspace(0.01, 100, 10)
        vars = np.geomspace(0.01, 5, 10)
        self.log_L = np.zeros((len(ells_1), len(ells_2), len(vars)))
        for i, l1 in enumerate(ells_1):
            for j, l2 in enumerate(ells_2):
                for k, v in enumerate(vars):
                    self.update_GP(l1 = l1, l2=l2, var = v)
                    self.log_L[i,j,k] = self.get_GP_likelihood()

        idx = np.where(self.log_L == np.array(self.log_L).min())
        self.l1_opt = ells_1[idx[0][0]]
        self.l2_opt = vars[idx[1][0]]
        self.var_opt = vars[idx[2][0]]
        print('l1 = ', ells_1[idx[0][0]], 'l2 = ', ells_2[idx[1][0]], 'var = ', vars[idx[2][0]])
        self.update_GP(l1 = self.l1_opt, l2 = self.l2_opt, var = self.var_opt)
    
    def GP_predict(self, X_star, get_var = False):
        X_star = np.log(X_star)
        K_star = self.kern.K(X_star, X_star)
        k_star = self.kern.K(self.X_train, X_star)

        self.noise_var_star = self.noise_GP_predict(X_star)
        self.noise_cov_star = np.diag(self.noise_var_star.flatten())

        f_star = np.dot(k_star.T, self.weights)
        if get_var:
            v = np.linalg.solve(self.L, k_star)
            V_star_epi = K_star - np.dot(v.T, v)
            V_star_noise = self.noise_cov_star
            if self.output_type == 'T':
                V_epi  = np.sqrt(np.diag(V_star_epi))
                V_noise  = np.sqrt(np.diag(V_star_noise))
                # V_noise = gaussian_filter(V_noise, sigma = 5)
                return f_star.flatten(), V_epi.flatten(), V_noise.flatten()
            else:
                f_star = np.exp(f_star.flatten())
                V_epi = f_star**2 * np.diag(V_star_epi)

                V_noise = f_star**2 * np.diag(V_star_noise)

                # V_noise = gaussian_filter(V_noise, sigma = 20)

                return f_star.flatten(), V_epi.flatten(), V_noise.flatten()
        else:
            if self.output_type == 'T':
                return f_star.flatten()
            else:
                return np.exp(f_star).flatten()
