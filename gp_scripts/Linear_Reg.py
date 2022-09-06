from utils import *

class LinearBasis:
    def __init__(self):
        self.num_basis = 2 # The number of basis functions
        
    def __call__(self, x):
        """
        ``x`` should be a 1D array of inputs
        """
        return [1., x[0]]

    
class PolynomialBasis:
    def __init__(self, degree):
        self.degree = degree
        self.num_basis = degree + 1
    def __call__(self, x):
        return np.array([x[0] ** i for i in range(self.degree + 1)])

def design_matrix(X, phi):
    num_observations = X.shape[0]
    num_basis = phi.num_basis
    Phi = np.zeros((num_observations, num_basis))
    for i in range(num_observations):
        Phi[i, :] = phi(X[i, :])
    return Phi

def Linear_Regression_fit(X, Y, basis, ax = None, degree = None, plot = True):

    if not isinstance(basis,str):
            raise Exception("ERROR: basis argument must be a string (linear or polynomial)")

    if basis == 'linear':
        phi = LinearBasis()
    elif basis == 'polynomial':
        phi = PolynomialBasis(degree)
    else:
        print('Please set basis argument to linear or polynomial (str)')
    
    if basis == 'polynomial' and degree == None:
        print('Please set degree argument to an intger value > 0')
        return None
    
    Phi = design_matrix(X[:,None], phi)
    w_MLE, res_MLE, _, _ = np.linalg.lstsq(Phi, Y, rcond=None)
    sigma_MLE = np.sqrt(res_MLE / X.shape[0])

    Phi_p = design_matrix(X[:,None], phi)
    Y_p = (Phi_p @ w_MLE).flatten()

    if plot:
        Y_l = Y_p - 2. * sigma_MLE # Lower predictive bound (95% confidence interval)
        Y_u = Y_p + 2. * sigma_MLE # Upper predictive bound (95% confidence interval)
        if ax == None:
            plt.plot(X, Y, 'x', label='Observations')
            plt.plot(X, Y_p, lw=2, label=f'LS prediction (Linear Basis)')
            plt.xlabel(r'$\mathrm{log}(L_n) $'); plt.ylabel(r'$\mathrm{log}(I_{thr})$'); plt.legend(loc='best');
            plt.fill_between(X.flatten(), Y_l, Y_u, color='C1', alpha=0.25)
            plt.text(x = -7.35, y = 35.03, s = r'$\sigma_{MLE}$ = '+ str(np.round(sigma_MLE[0], 3)), fontsize = 25)
            plt.text(x = -7.35, y = 35.07, s = r'$w_{MLE}$ = '+ str(np.round(w_MLE[0], 3)) + ' , ' +  str(np.round(w_MLE[1], 3)), fontsize = 25)
        else:
            ax.plot(X, Y, 'x', label='Observations')
            ax.plot(X, Y_p, lw=2, label=f'LS prediction (Linear Basis)')
            ax.set_xlabel(r'$\mathrm{log}(L_n) $'); ax.set_ylabel(r'$\mathrm{log}(I_{thr})$'); plt.legend(loc='best');
            ax.fill_between(X.flatten(), Y_l, Y_u, color='C1', alpha=0.25)
            ax.text(x = -7.35, y = 35.03, s = r'$\sigma_{MLE}$ = '+ str(np.round(sigma_MLE[0], 3)), fontsize = 25)
            ax.text(x = -7.35, y = 35.07, s = r'$w_{MLE}$ = '+ str(np.round(w_MLE[0], 3)) + ' , ' +  str(np.round(w_MLE[1], 3)), fontsize = 25)
        print('weights = ', w_MLE)
        print('sigma = ', sigma_MLE)
    else:
        return w_MLE, sigma_MLE