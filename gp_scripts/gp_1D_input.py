from utils import *
from gp_regression import *

class GP_1D:

    def __init__(self, kernel, input_type, output_type, input_file, output_file, train_frac, restarts):

        self.kernel = kernel
        self.input_type = input_type
        self.output_type = output_type
        self.input_file = input_file
        self.output_file = output_file
        self.train_frac = train_frac
        self.restarts = restarts

    def get_input(self):
        os.path.exists(self.input_file)
        with open(self.input_file, 'r') as f:
            train_inputs = json.load(f)
        train_inputs = np.array(train_inputs)
        n,m = train_inputs.shape

        input = np.zeros(n)

        if (self.input_type == 'I'):
            for i in range(n):
                input[i] = train_inputs[i][0]
            return input
        elif (self.input_type == 'Ln'):
            for i in range(n):
                input[i] = train_inputs[i][1]
            return input
        else:
            print('ERROR: Please set input_type to either I (Intensity) or Ln (Density scale length) (str)')
            return None

    def get_output(self):
        os.path.exists(self.output_file)
        with open(self.output_file, 'r') as f:
            train_outputs = json.load(f)
        train_outputs = np.array(train_outputs)
        n,m,k = train_outputs.shape

        output = np.zeros(n)
        output_var = np.zeros(n)

        if (self.output_type == 'I_srs'):
            for i in range(n):
                output[i] = train_outputs[i][0][0]
                output_var[i] = train_outputs[i][0][1] ** 2
            return output
        elif (self.output_type == 'T_hot'):
            for i in range(n):
                output[i] = train_outputs[i][1][0]
                output_var[i] = train_outputs[i][1][1] ** 2
            return output
        elif (self.output_type == 'E_frac'):
            for i in range(n):
                output[i] = train_outputs[i][2][0]
                output_var[i] = train_outputs[i][2][1] ** 2
            return output
    
        else:
            print('ERROR: Please set output_type to either I_srs, T_hot or E_frac (str)')
            return None

    def sacle_data(self):
        input = self.get_input()
        output = self.get_output()
        
        in_min = input.min()
        in_max = input.max()
        out_min = output.min()
        out_max = output.max()

        scaled_input = (input - in_min) / (in_max - in_min)
        scaled_output = (output - out_min) / (out_max - out_min)

        return scaled_input, scaled_output


    def test_train(self):
        input, output = self.sacle_data()
        X_train, X_test, Y_train, Y_test = train_test_split(input, output, test_size = 1 - self.train_frac)

        self.X_train = X_train[:, None]
        self.X_test = X_test[:, None]

        self.Y_train = Y_train[:, None]
        self.Y_test = Y_test[:, None]

    def regression(self):

        self.test_train()

        if (self.kernel == 'RBF'):
            kern= GPy.kern.RBF(input_dim=1,variance=1,lengthscale=1)
        elif (self.kernel == 'Matern32'):
            kern = GPy.kern.Matern32(input_dim=1,variance=1,lengthscale=1)
        elif (self.kernel == 'Matern52'):
            kern = GPy.kern.Matern52(input_dim=1,variance=1,lengthscale=1)
        elif (self.kernel == 'Exponential'):
            kern = GPy.kern.Exponential(input_dim=1,variance=1,lengthscale=1)
        else:
            print('ERROR: Please set kernel to either RBF, Matern32, Matern52 or Exponential (str)')
            return None
        

        m = GPy.models.GPRegression(self.X_train, self.Y_train, kern, noise_var = 1)

        m.optimize_restarts(self.restarts, verbose=False)
        return m

    def GP_predict(self, model, X):
        input = self.get_input()
        output = self.get_output()

        in_min = input.min()
        in_max = input.max()
        out_min = output.min()
        out_max = output.max()


        scaled_X = (X - in_min) / (in_max - in_min)
       
        Y_p, V_p = model.predict(scaled_X, include_likelihood=False)

        rescaled_Y_P = Y_p * (out_max - out_min) + out_min
        rescaled_V_P = V_p * (out_max - out_min) + out_min

        return rescaled_Y_P, rescaled_V_P


    def test_train_plot(self, model):
        self.test_train()
        input, target_value = self.sacle_data()

        Y_ptrain, V_ptrain = model.predict(self.X_train)
        Y_ptest, V_ptest = model.predict(self.X_test)

        rmse_train = np.sqrt(np.mean((Y_ptrain[:,0] - self.Y_train[:,0])**2))
        rmse_test = np.sqrt(np.mean((Y_ptest[:,0] - self.Y_test[:,0])**2))


        ### DISTRIBUTION OF QUANTITY OF INTEREST
        fig = plt.figure()
        sns.kdeplot(target_value.reshape(len(target_value)), label = 'Target Data', linestyle='dashdot', linewidth = 5, color = 'black')
        sns.kdeplot(Y_ptrain.reshape(len(Y_ptrain)), label=f'Train, RMSE = '+str(rmse_train), color = 'blue')
        sns.kdeplot(Y_ptest.reshape(len(Y_ptest)), label=f'Test, RMSE = '+str(rmse_test), color = 'orange')
        plt.legend()
        if self.output_type == 'I_srs':
            plt.xlabel(r'$I_srs \,\, W/cm^2$' )
            plt.title(r'Backscatterd SRS Intensity')
        elif self.output_type == 'T_hot':
            plt.xlabel(r'$T_{hot} \,\, keV$')
            plt.title(r'Hot Electron Temperature')
        elif self.output_type == 'E_frac':
            plt.xlabel(r'$Fraction E_{hot} > 100 \,\, keV$')
            plt.title(r'Fraction of Hot Elcetrons with $E > 100 \,\, keV $')

        plt.ylabel(r'Density')
        plt.show()

        ### STANDARD DEVIATION 
        S_ptrain = np.sqrt(V_ptrain)
        S_ptest = np.sqrt(V_ptest)

        #### ax1/2 --- train
        #### ax3/4 ---- test

        ### PLOT CORRELATION BETWEEN VALUES AND ERRORS
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.scatter(self.Y_train, Y_ptrain, label=f'Train', color = 'blue', alpha = 0.6)
        ax1.plot([target_value.min(), target_value.max()], [target_value.min(), target_value.max()], 'k:', label = 'Target')
        ax1.set_xlabel('True Value')
        ax1.set_ylabel('Predicted Value')
        ax1.legend()

        ax2.plot(abs(Y_ptrain[:,0] - self.Y_train[:, 0]), S_ptrain[:, 0], 'o', label='Train', color = 'blue')
        ax2.set_xlabel('True Error')
        ax2.set_ylabel('Predicted Error')
        # ax2.plot([0, S_ptrain.max()], [0, S_ptrain.max()], 'k:', label = 'Target')
        ax2.legend()

        ax3.scatter(self.Y_test, Y_ptest, label=f'Test', alpha=0.6, color = 'orange')
        ax3.plot([target_value.min(), target_value.max()], [target_value.min(), target_value.max()], 'k:', label = 'Target')
        ax3.set_xlabel('True Value')
        ax3.set_ylabel('Predicted Value')
        ax3.legend()


        ax4.plot(abs(Y_ptest[:,0] - self.Y_test[:,0]), S_ptest, 'o', label='Test', color='orange')
        ax4.set_xlabel('True Error')
        ax4.set_ylabel('Predicted Error')
        # ax4.plot([0, S_ptest.max()], [0, S_ptest.max()], 'k:', label = 'Target')
        ax4.legend()




        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.5, 
                    hspace=0.4)
        plt.gcf().set_size_inches(20,10)

        plt.show()