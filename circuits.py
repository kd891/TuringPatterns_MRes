'''
This file contains the ODE systems and the Jacobian matrices for the scale and the switch

'''

'''
SCALE
'''
import numpy as np

def ode_sys_Scale(x, k):

    A = x[0] #A1
    B = x[1] #A2
    C = x[2] #B1
    D = x[3] #B2


    ent = k[8]*A + (1-k[8])*B
    entA = ent*ent


    FA = k[2]*entA/C
    FB = k[3]*entA/D
    FC = k[2]*entA
    FD = k[3]*entA

    dAdt = k[0] + FA - k[4]*A
    dBdt = k[1] + FB - k[5]*B
    dCdt = FC - k[6]*C
    dDdt = FD - k[7]*D

    return np.array([dAdt, dBdt, dCdt, dDdt])


def jacobian_Scale(x, k):

    A1 = x[0] #A1
    A2 = x[1] #A2
    B1 = x[2] #B1
    B2 = x[3] #B2

    A1sq = A1*A1
    A2sq = A2*A2
    B1sq = B1*B1
    B2sq = B2*B2

    Ag = k[2]*k[8]
    Bg = k[3]*k[8]


    gneg = k[8]-1

    gA1 = k[8]*A1
    gA2 = k[8]*A2

    brac = (gA1 - gA2 + A2)
    bracsq = brac*brac

    #Jacobian terms

    dfdA = 2*Ag*brac/B1 - k[4]
    dfdB = (-2*k[2]*gneg*brac)/B1
    dfdC = (-k[2] * bracsq)/B1sq
    dfdD = 0

    dgdA = 2*Bg*brac/B2
    dgdB = (-2*k[3]*gneg*brac)/B2 - k[5]
    dgdC = 0
    dgdD = (-k[3]*bracsq)/B2sq

    dzdA = 2*Ag*brac
    dzdB = -2*k[2]*gneg*brac
    dzdC = -k[6]
    dzdD = 0

    dndA = 2*Bg*brac
    dndB = -2*k[3]*gneg*brac
    dndC = 0
    dndD = -k[7]


    jac = np.array(

                [[dfdA, dfdB, dfdC, dfdD],
                 [dgdA, dgdB, dgdC, dgdD],
                 [dzdA, dzdB, dzdC, dzdD],
                 [dndA, dndB, dndC, dndD]]
    )


    return jac



'''
SWITCH
'''




def ode_sys_Switch(x, k):

    A = x[0] #A1
    B = x[1] #A2
    C = x[2] #B1
    D = x[3] #B2


    entA1 = (k[8]*A + B)/(1+k[8])

    entA1sq = entA1 * entA1

    entA2 = (k[8]*B + A)/(1+k[8])

    entA2sq = entA2 * entA2

    FA1 = k[1]*entA1sq/C
    FA2 = k[2]*entA2sq/D
    FB1 = k[1]*entA1sq
    FB2 = k[2]*entA2sq


    dAdt = k[0] + FA1 - k[4]*A
    dBdt = k[0] + FA2 - k[5]*B
    dCdt = FB1 - k[6]*C
    dDdt = FB2 - k[7]*D

    return np.array([dAdt, dBdt, dCdt, dDdt])

def jacobian_Switch(x, k):

    A1 = x[0]
    A2 = x[1]
    B1 = x[2]
    B2 = x[3]

    A1sq = A1*A1
    A2sq = A2*A2
    B1sq = B1*B1
    B2sq = B2*B2

    Ag = k[2]*k[8]
    Bg = k[3]*k[8]

    gp1 = k[8] + 1 #g plus 1
    gp1sq = gp1*gp1

    gA1 = k[8]*A1
    gA2 = k[8]*A2

    #Jacobian terms

    dfdA = 2*Ag*(A1 + gA2)/(B1*gp1sq) - k[4]
    dfdB = 2*k[2]*(A1 + gA2)/(B1*gp1sq)
    dfdC = (-k[2]/B1sq) * ((A1+gA2)**2)/(B1sq*gp1sq)
    dfdD = 0

    dgdA = 2*k[3]*(A2 + gA1)/(B2*gp1sq)
    dgdB = 2*Bg*(A2 + gA1)/(B2*gp1sq) - k[5]
    dgdC = 0
    dgdD = (-k[3]/B2sq) * ((A2+gA1)**2)/(B2sq*gp1sq)

    dzdA = 2*k[2]*(A1 + gA2)/gp1sq
    dzdB = 2*Ag*(A1 + gA2)/gp1sq
    dzdC = -k[6]
    dzdD = 0

    dndA = 2*k[3]*(A2 + gA1)/gp1sq
    dndB = 2*Bg*(A2 + gA1)/gp1sq
    dndC = 0
    dndD = -k[7]

    jac = np.array(

                 [[dfdA, dfdB, dfdC, dfdD],

                  [dgdA, dgdB, dgdC, dgdD],

                  [dzdA, dzdB, dzdC, dzdD],

                  [dndA, dndB, dndC, dndD]]
    )


    return jac

'''
Classical GM Model, 2 Nodes
k0 = alpha
k1 = beta
k2 = muA
k3 = muB
'''

def ode_sys_GM(x, k):

    A = x[0] #A1/A2
    B = x[1] #B1/B2

    Asq = A*A

    FA = k[1]*Asq/B
    FB = k[1]*Asq


    dAdt = k[0] + FA - k[2]*A
    dBdt = FB - k[3]*B

    return np.array([dAdt, dBdt])



def jac_GM(x, k):

    A1 = x[0]
    B1 = x[1]

    Asq = A1*A1
    Bsq = B1*B1

    #Jacobian terms

    dfdA = (2*k[1]*A1/B1) - k[2]
    dfdB = -k[1]*Asq/Bsq

    dgdA = 2*k[1]*A1
    dgdB = -k[3]

    jac = np.array(

                 [[dfdA, dfdB],
                  [dgdA, dgdB]]
    )

    return jac

'''
3 Node circuit
'''

def ode_sys_three_A1(x,k) :

    A = x[0]
    B = x[1]
    C = x[2]

    Asq = A*A
    #BC = np.sqrt(B*C)

    FA = k[1]*Asq / B
    FB = k[1]*Asq
    FC = k[1]*Asq

    dAdt = k[0] + FA - k[2]*A
    dBdt = FB - k[3]*B
    dCdt = FC - k[4]*C

    return np.array([dAdt, dBdt, dCdt])

def ode_sys_three_A2(x,k) :

    A = x[0]
    B = x[1]
    C = x[2]

    Asq = A*A

    FA = k[1]*Asq / C
    FB = k[1]*Asq
    FC = k[1]*Asq

    dAdt = k[0] + FA - k[2]*A
    dBdt = FB - k[3]*B
    dCdt = FC - k[4]*C

    return np.array([dAdt, dBdt, dCdt])


def jac_three(x, k):

    A1 = x[0]
    B1 = x[1]
    B2 = x[2]

    Asq = A1*A1
    Bsq = B1*B1
    B2sq = B2*B2

    #Jacobian terms

    dfdA = (2*k[1]*A1/B2) - k[2]
    dfdB = -k[1]*Asq/Bsq
    dfdC = 0

    dgdA = 2*k[1]*A1
    dgdB = -k[3]
    dgdC = 0

    dzdA = 2*k[1]*A1
    dzdB = 0
    dzdC = -k[4]

    jac = np.array(

                 [[dfdA, dfdB, dfdC],
                  [dgdA, dgdB, dgdC],
                  [dzdA, dzdB, dzdC]]
    )

    return jac

'''
def jac_three_adj(x, k):

    A1 = x[0]
    B1 = x[1]
    B2 = x[2]

    Asq = A1*A1
    Bsq = B1*B1
    B2sq = B2*B2

    rB = np.sqrt(B1*B2)

    #Jacobian terms

    dfdA =
    dfdB =
    dfdC =

    dgdA = 2*k[1]*A1
    dgdB = -k[3]
    dgdC = 0

    dzdA = 2*k[1]*A1
    dzdB = 0
    dzdC = -k[4]

    jac = np.array(

                 [[dfdA, dfdB, dfdC],
                  [dgdA, dgdB, dgdC],
                  [dzdA, dzdB, dzdC]]
    )

    return jac
'''
