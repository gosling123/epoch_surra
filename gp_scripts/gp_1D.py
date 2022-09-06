from utils import *
from scipy.ndimage import gaussian_filter

class LPI_GP_1D:
    """Class implementing a Gaussian Process"""
    
    def __init__(self, input_file = None, input_type = None, output_type = None,\
                 output_file = None, var_file = None, train_frac = 0.4):
        
        self.input_file = input_file
        self.input_type = input_type
        self.output_file = output_file
        self.output_type = output_type
        self.var_file = var_file
        self.train_frac = train_frac

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
        return np.log(noise_var)


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
    
    def set_training_data(self):
        X = self.get_input()
        Y = self.get_output()
        noise = self.get_noise_var()

        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = self.train_frac)

        indxs_train = []
        indxs_test = []
        for i in range(len(X_train)):
            mask = np.in1d(X, X_train[i])
            indx = np.where(mask == True)[0][0]
            indxs_train.append(indx)
        for i in range(len(X_test)):
            mask = np.in1d(X, X_test[i])
            indx = np.where(mask == True)[0][0]
            indxs_test.append(indx)
        
        noise_train = noise[indxs_train]
        noise_test = noise[indxs_test]

        self.X_train = X_train[:,None]
        self.Y_train = Y_train[:,None]
        self.noise_train = noise_train
        self.X_test = X_test[:,None]
        self.Y_test = Y_test[:,None]
        self.noise_test = noise_test

        self.X_range = X.max()-X.min()
        self.Y_range = Y.max()-Y.min()
        self.noise_range = noise.max() - noise.min()
    
    def update_noise_GP_kern(self, l, var):
        l *= self.X_range
        var *= self.noise_range
        self.kern_noise = GPy.kern.Exponential(input_dim=1, variance=var, lengthscale=l, ARD=True)
        self.K_noise = self.kern_noise.K(self.X_train, self.X_train)

    def update_noise_GP_weights(self, var_noise = 1e-6):
        self.L_noise = np.linalg.cholesky(self.K_noise + var_noise * self.noise_range* np.eye(len(self.X_train)))
        self.weights_noise = np.linalg.solve(self.L_noise.T, np.linalg.solve(self.L_noise, self.noise_train))
    
    def update_noise_gp(self, l, var):
        self.update_noise_GP_kern(l, var)
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
        ells = np.geomspace(0.1, 20, 50)
        vars = np.geomspace(0.1, 20, 50)
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
            return f_star_noise, err
        else:
            return f_star_noise

    def update_GP_kern(self, l, var):
        l *= self.X_range
        var *= self.Y_range
        self.noise_var = self.noise_GP_predict(X_star=self.X_train)
        self.noise_cov = np.diag(self.noise_var)
        self.kern = GPy.kern.RatQuad(input_dim=1, variance=var, lengthscale=l)
        self.K = self.kern.K(self.X_train, self.X_train)
        self.K += self.noise_cov

    def update_GP_weights(self):
        self.L = np.linalg.cholesky(self.K + 1e-6 * self.Y_range * np.eye(len(self.X_train)))
        self.weights = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.Y_train))

    def update_GP(self, l, var):
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
        ells = np.geomspace(0.05, 100, 20)
        vars = np.geomspace(0.05, 100, 20)
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
                V_noise = gaussian_filter(V_noise, sigma = 2)
                return f_star.flatten(), V_epi.flatten(), V_noise.flatten()
            else:
                f_star = np.exp(f_star.flatten())
                V_epi = f_star**2 * np.diag(V_star_epi)

                V_noise = f_star**2 * np.diag(V_star_noise)

                V_noise = gaussian_filter(V_noise, sigma = 2)

                return f_star.flatten(), V_epi.flatten(), V_noise.flatten()
        else:
            if self.output_type == 'T':
                return f_star.flatten()
            else:
                return np.exp(f_star).flatten()

    def test_train_plot(self):

        ### TARGET DATA
        if self.output_type == 'T':
            target_value = self.get_output()
            Y_train = self.Y_train.flatten()
            Y_test = self.Y_test.flatten()
        else:
            target_value = np.exp(self.get_output()).flatten()
            Y_train = np.exp(self.Y_train).flatten()
            Y_test = np.exp(self.Y_test).flatten()

        y_train_predict, var_train_epi, var_train_noise = self.GP_predict(X_star=np.exp(self.X_train), get_var = True)
        y_test_predict, var_test_epi, var_test_noise = self.GP_predict(X_star=np.exp(self.X_test), get_var = True)
        
        rmse_train = np.sqrt(np.mean((Y_train-y_train_predict)**2))
        rmse_test = np.sqrt(np.mean((Y_test-y_test_predict)**2)) 

        ### STANDARD DEVIATION  
        S_ptrain = np.sqrt(var_train_epi+var_train_noise)
        S_ptest = np.sqrt(var_test_epi+var_test_noise)

        #### ax1/2 --- train
        #### ax3/4 ---- test

        ### PLOT CORRELATION BETWEEN VALUES AND ERRORS
        fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
        ax3 = sns.kdeplot(target_value, label = 'Target Data', linestyle='dashdot', linewidth = 5, color = 'black')
        ax3 = sns.kdeplot(y_train_predict, label=f'Train', color = 'blue')
        ax3 = sns.kdeplot(y_test_predict, label=f'Test', color = 'orange')
        # ax3.legend()
        if self.output_type == 'P':
            ax3.set_xlabel(r'$\mathcal{P}$')
        elif self.output_type == 'T':
            ax3.set_xlabel(r'$T_{hot} \,\, keV$')
        elif self.output_type == 'E':
            ax3.set_xlabel(r'$\mathcal{E}$')

        ax1.scatter(Y_train, y_train_predict, label=f'Train (RSME = {np.round(rmse_train, 3)})', color = 'blue')
        ax1.plot([target_value.min(), target_value.max()], [target_value.min(), target_value.max()], 'k:', label = 'Target')

        if self.output_type == 'P':
            ax1.set_xlabel(r'True Value - $\mathcal{P}$')
            ax1.set_ylabel(r'Predicted Value - $\mathcal{P}$')
        elif self.output_type == 'T':
            ax1.set_xlabel(r'True Value - $T_{hot} \,\, keV$')
            ax1.set_ylabel(r'Predicted Value - $T_{hot} \,\, keV$')
        elif self.output_type == 'E':
            ax1.set_xlabel(r'True Value - $\mathcal{E}$')
            ax1.set_ylabel(r'Predicted Value - $\mathcal{E}$')



        ax2.plot(abs(y_train_predict - Y_train), S_ptrain, 'o', label='Train', color = 'blue')
        ax2.plot([0, S_ptrain.max()], [0, S_ptrain.max()], 'k:', label = 'Target')


        ax1.scatter(Y_test, y_test_predict, label=f'Test (RSME = {np.round(rmse_test, 3)})', color = 'orange')
        ax1.legend()

        ax2.plot(abs(y_test_predict - Y_test), S_ptest, 'o', label='Test', color='orange')
        ax2.plot([0, S_ptest.max()], [0, S_ptest.max()], 'k:', label = 'Target')
        # ax2.legend()
        if self.output_type == 'P':
            ax2.set_xlabel(r'True Error - $\mathcal{P}$')
            ax2.set_ylabel(r'Predicted Error - $\mathcal{P}$')
        elif self.output_type == 'T':
            ax2.set_xlabel(r'True Error - $T_{hot} \,\, keV$')
            ax2.set_ylabel(r'Predicted Error - $T_{hot} \,\, keV$')
        elif self.output_type == 'E':
            ax2.set_xlabel(r'True Error - $\mathcal{E}$')
            ax2.set_ylabel(r'Predicted Error - $\mathcal{E}$')

        plt.subplots_adjust(left=0.1,
                            bottom=0.1, 
                            right=0.9, 
                            top=0.9, 
                            wspace=0.5, 
                            hspace=0.4)

        plt.show()

def GP_1D_predict_all(X_star, input_file, input_type, output_file,\
                      var_file, fname = 'data_dict.pickle', save = False):
    output_types = ['P', 'T', 'E']
    label = ['Reflectivity', 'Hot Electron Temperature', 'Fraction E>50 keV']
    data_dict = {'input' : X_star.flatten(), 'output' : {}, 'error_epi' : {}, 'error_noise' : {}}
    start = time.time()
    for i, out in enumerate(output_types):
        t1 = time.time()
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
        data_dict['output'][label[i]] = Y_star.flatten()
        data_dict['error_epi'][label[i]] = 2.0*sig_epi.flatten()
        data_dict['error_noise'][label[i]] = 2.0*sig_noise.flatten()
        print(f'Finished {out} GP regression in {(time.time() - t1)/60} minutes')
        print('--------------------------------------------------------------------')
    print(f'All ouput GP predictions completed in {(time.time() - start)/60} minutes')
    if save:
        with open(fname, 'wb') as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved GP result dictioanry to {fname}')
    return data_dict
