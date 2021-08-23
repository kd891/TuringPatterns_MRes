

import os.path
import sys

import pickle5 as pickle
import pandas as pd
import numpy as np
from datetime import date

date = str(date.today())

#######################
#########CODE##########
#######################

#creates loguniform distribution
def loguniform(low=-3, high=3, size=None):
    return (10) ** (np.random.uniform(low, high, size))

#creates uniform distribution
def uniform(low=-3, high=3, size=None):
    return np.random.uniform(low, high, size)

#provided a dataset with a certain distribution and a number of samples, outputs dataset with specific distribution and n number of samples.
def lhs(data, nsample):
    m, nvar = data.shape
    ran = np.random.uniform(size=(nsample, nvar))
    s = np.zeros((nsample, nvar))
    for j in range(0, nvar):
        idx = np.random.permutation(nsample) + 1
        P = ((idx - ran[:, j]) / nsample) * 100
        s[:, j] = np.percentile(data[:, j], P)
    return s


def parameterfile_creator_function(numbercombinations):
    #create distribution to input in lhs function
    loguniformdist = loguniform(size=1000000)

    #These are different kinetic parameters of the model and the defined ranges where we define our parameter space
    Beta_range = (0.01, 50)
    mu_range = (0.001, 50)
    alpha_range = (0.01, 50)

    # - Split previously created distribution with the parameter ranges. it needs to be split so lhs function
    # understands where the ranges are.
    Beta_distribution = [x for x in loguniformdist if Beta_range[0] <= x <= Beta_range[1]]
    mu_distribution = [x for x in loguniformdist if mu_range[0] <= x <= mu_range[1]]
    alpha_distribution = [x for x in loguniformdist if alpha_range[0] <= x <= alpha_range[1]]

    #make all the distributions of the same size to stack them in a matrix.
    lenghtsdistributions = ( len(Beta_distribution), len(mu_distribution), len(alpha_distribution))
    minimumlenghtdistribution = np.amin(lenghtsdistributions)
    Beta_distribution = Beta_distribution[:minimumlenghtdistribution]
    mu_distribution = mu_distribution[:minimumlenghtdistribution]
    alpha_distribution = alpha_distribution[:minimumlenghtdistribution]

    # A general matrix is generated with the distributions for each parameter.
    # if you need 6Vm parameters (one for each molecular specie) you define it at this point.
    Beta_matrix = np.column_stack((Beta_distribution, Beta_distribution))
    mu_matrix = np.column_stack((mu_distribution, mu_distribution, mu_distribution, mu_distribution))
    alpha_matrix = np.column_stack((alpha_distribution, alpha_distribution))

    par_distribution_matrix = np.concatenate((alpha_matrix, Beta_matrix, mu_matrix), 1)

    #create lhs distribution from predefined loguniform distributions of each parameter in its ranges
    points = lhs(par_distribution_matrix, numbercombinations)

    #Defining constant parameters (these parameters are necessary for the model but wont be sampled in the parameter space)
    GAMMA = np.full((numbercombinations, 1), 0)
    Turing = np.full((numbercombinations, 1), 0)
    MaxEIG = np.full((numbercombinations, 1), 0)
    MaxWVN = np.full((numbercombinations, 1), 0)
    parameterindex = np.arange(1, numbercombinations + 1, dtype=np.int).reshape(numbercombinations, 1) #parID index

    #add constant parameters to matrix created in lhs (points)


    points = np.concatenate((parameterindex, points, GAMMA, Turing, MaxEIG, MaxWVN), 1)

    #define index of the columns of the dataframe
    parameternames = (
    'index','alpha_A1','alpha_A2', 'beta_A1', 'beta_A2', 'muA1', 'muA2', 'muB1', 'muB2', 'GAMMA', 'Turing', 'MaxEIG', 'MaxWVN')
    df = pd.DataFrame(data=points, columns=parameternames)
    df['index'] = df['index'].astype(int)
    df = df.set_index('index')

    return df

#number of parameter sets you want in your df.
n_parametersets = 100000
df = parameterfile_creator_function(n_parametersets)

outfile = open('df_entparams_%rparametersets_biological2.pkl'%n_parametersets, 'wb')
pickle.dump(df, outfile)
outfile.close()
