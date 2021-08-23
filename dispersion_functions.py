'''
This file contains all the functions required to perform the linear stability analysis

1) Finding the steady state values
2) Harmonic diagonal perturbation of the steady state in the presence of diffusion

'''
import scipy
from scipy import optimize
import numpy as np


'''
Steady State: Powell's dogleg method
N.B. we have to call ss.x for the output and ss.result for the condition
'''


def steady_state(func, x0, k, jacobian):
    ss = optimize.root(func,x0, args=k, jac=jacobian)
    return ss


def ss_newton(func, x0, k):

    ss = scipy.optimize.newton(func, x0, fprime=None, args=(k,))

    return ss






'''
Harmonic perturbation
'''
def DispRel_four(wvn, jac, D_A=0.01, D_B=0.01, D_C=0.4, D_D=0.4):
    jac[0, 0] += -D_A*wvn**2
    jac[1, 1] += -D_B*wvn**2
    jac[2, 2] += -D_C*wvn**2
    jac[3, 3] += -D_D*wvn**2
    eigval = np.linalg.eig(jac)
    return eigval



def DispRel_two(wvn, jac, D_A=0.01, D_B=0.4):
    jac[0, 0] += -D_A*wvn**2
    jac[1, 1] += -D_B*wvn**2
    eigval = np.linalg.eig(jac)
    return eigval

def DispRel_three(wvn, jac, D_A=0.01, D_B=0.4, D_C=0.4):
    jac[0, 0] += -D_A*wvn**2
    jac[1, 1] += -D_B*wvn**2
    jac[2, 2] += -D_C*wvn**2
    eigval = np.linalg.eig(jac)
    return eigval
