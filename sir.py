#!/usr/bin/env python
from numpy import array, arange
from scipy.optimize import fmin
from scipy.integrate import ode, odeint
import matplotlib.pyplot as plt


def dydt(y, t, *args):
    beta = args[0]
    gamma = args[1]

    S = y[0]
    I = y[1]
    R = y[2]

    dydt = (-beta * S * I,
            beta * S * I - gamma * I,
            gamma * I)

    return dydt


def discrepancy(x):
    y = odeint(dydt, y0, tspan, args=(x[0], x[1]))
    return(sum((y[:, 1] - data) ** 2))


if __name__ == '__main__':
    # resources:
    # http://www.samsi.info/sites/default/files/Shaby_sir_lab.pdf
    # http://www.samsi.info/workshop/2009-10-undergraduate-modeling-workshop
    # http://goo.gl/moUO83

    # data for model
    tspan = arange(0, 14)
    data = array([3, 6, 25, 73, 222, 294, 258, 237, 191, 125, 69, 27, 11, 4])

    # set guess for parameters
    p = (1e-3, 5e-4)

    # set initial conditions
    y0 = (760, data[0], 0)

    # find optimal beta and gamma parameters
    p_opt = fmin(discrepancy, p, disp=0)
    R0 = p_opt[0] / p_opt[1]

    # calculate values of SIR model with p_opt
    y = odeint(dydt, y0, tspan, args=(p_opt[0], p_opt[1]))

    plt.rc('text', usetex=True)
    plt.plot(tspan, data, '*')
    plt.plot(tspan, y[:, 1])
    plt.xlabel('Time Step')
    plt.ylabel('Infections')
    plt.title('SIR Model of Influenza')
    values = (y0[0], p_opt[0], p_opt[1], R0)
    text = 'Population: %d\nBeta: %.3f\nGamma: %.3f\nR0: %1.3f' % values 
    plt.annotate(text,
                 xy=(1, 1),
                 xycoords='axes fraction',
                 horizontalalignment='right',
                 verticalalignment='top')
    plt.savefig('sir.pdf')
