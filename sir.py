#!/usr/bin/env python
from numpy import array, arange
from scipy.optimize import minimize
from scipy.integrate import ode, odeint


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
    S0 = 760
    I0 = data[0]
    R0 = 0

    y0 = (S0, I0, R0)
    ode_results = odeint(dydt, y0, tspan, args=p)

    print ode_results
