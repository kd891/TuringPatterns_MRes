'''
Run the Turing analysis
'''
from tqdm import tqdm
import time
import os
import pandas as pd
import numpy as np
from scipy import optimize
from sampling import *
from circuits import *
from dispersion_functions import *
from multiprocessing import Pool

'''
1. Import the Parameter file
'''
n_samples = 100000
gammas = [0,0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2]
par_df = pd.read_pickle('/Users/kushaal/Documents/Turing/Dispersion/current_method/param_files/df_entparams_100000parametersets_biological.pkl')
runs = len(par_df)


'''
2.  analysis function

28 Jul changes - removed the gammas loop and turned it into a uniform dist of
'''


def analysis(gammas, runs, par_df, n_samples):
    for gamma in gammas:
        par_df['GAMMA'] = gamma #GIVE GAMMA A VALUE
        for run in tqdm(range(runs)):
            k = extraction_seven(par_df, run)
            x0 = [1,1,1,1]

            try:
                ss = optimize.newton(ode_sys_Scale, x0, fprime=None, args=(k,), tol = 1e-7, maxiter=50)
                sum_ss = np.sum(ss)
                isnan_ss = np.isnan(sum_ss)
                if np.any(ss<0):
                    progress = 0
                elif isnan_ss==True:
                    progress = 0
                else:
                    progress = 1

            except RuntimeError:
                progress = 0
            except RuntimeWarning:
                progress = 0
            except Exception:
                progress = 0

            if progress == 0:
                turing = 0
            else:
                sted_jac = ss
                jac_mat = jacobian_Scale(sted_jac, k)
                wvnvals = np.linspace(0,10,200)
                maxeig_re = []
                maxeig_im = []

                i = 0
                for wvn in wvnvals:

                    eigval = DispRel_four(wvn, jac_mat, D_A=0.01, D_B=0.01, D_C=0.5, D_D=0.5)

                    realeig = (max(np.real(eigval[0])))
                    maxeig_re.append(realeig)

                    imval = np.imag(eigval[0])
                    maxeig_im.append(imval)


                    if wvn == wvnvals[0]:
                        wvn0eigval = realeig
                        highest_eigval = realeig
                        highest_wvn = wvn
                    if realeig > highest_eigval:
                        highest_wvn = wvnvals[i]
                        highest_wvn = wvn
                        highest_eigval = realeig
                    if (
                        highest_wvn == wvnvals[-1]
                        or highest_eigval == wvn0eigval
                        or highest_wvn < wvnvals[2]
                        or np.abs(wvn0eigval - highest_eigval) < 0.01
                        or wvn0eigval > 0
                        ):
                        peak_height = np.NaN
                        turing = 0
                    else:
                        peak_height = highest_eigval
                        turing = 1
                    i+=1


                if maxeig_re[-1] >= 0:
                    turing = 0
                elif max(maxeig_re)<=0:
                    turing = 0
                elif maxeig_re[0] >= 0:
                    turing = 0
                else:
                    turing = 1


            if turing == 1:
                par_df.iloc[[run], [9]] = 1
                par_df.iloc[[run], [10]] = peak_height
                par_df.iloc[[run], [11]] = highest_wvn

            else:
                par_df.iloc[[run], [9]] = 0


        filename = 'results_%s_params_gamma_%s' % (n_samples, gamma)
        D_C = 0.5
        diff_score = D_C*100


        par_df.to_csv(r'/Users/kushaal/Documents/Turing/Dispersion/current_method/23Aug_Scale_Diffx' + str(diff_score) + '_BiologicalParams_' + filename + '.csv')


analysis(gammas, runs, par_df, n_samples)
