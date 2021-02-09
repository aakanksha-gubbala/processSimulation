import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.integrate
import scipy.interpolate
import scipy.optimize
from matplotlib import style

style.use("classic")
# print(plt.style.available)

compH = 'Benzene'
compC = 'Ethylene glycol'

data = pd.read_csv("CpData.csv").replace(np.NaN, 0)
comp = data['Name']
MW = data['Mol. wt.']
Tmin = data['Tmin, K']
Tmax = data['Tmax, K']
C1 = data['C1']
C2 = data['C2']
C3 = data['C3']
C4 = data['C4']
C5 = data['C5']

iH = comp.index[comp == compH].tolist()[0]
iC = comp.index[comp == compC].tolist()[0]

# Defining the enthalpy-temperature functions from Perry's Handbook
# CpL = C1 + C2T + C3T2 + C4T3 + C5T4

Tref = 293.15  # K


def H_H(T):  # T in K
    return (C1[iH] * (T - Tref) + C2[iH] * (T ** 2 - Tref ** 2) / 2 +
            C3[iH] * (T ** 3 - Tref ** 3) / 3 + C4[iH] * (T ** 4 - Tref ** 4) / 4 +
            C5[iH] * (T ** 5 - Tref ** 5) / 5) / MW[iH]


def H_C(T):  # T in K
    return (C1[iC] * (T - Tref) + C2[iC] * (T ** 2 - Tref ** 2) / 2 +
            C3[iC] * (T ** 3 - Tref ** 3) / 3 + C4[iC] * (T ** 4 - Tref ** 4) / 4 +
            C5[iC] * (T ** 5 - Tref ** 5) / 5) / MW[iC]


# Using linear interpolation to find temperature at a given enthalpy value

T = np.linspace(max(Tmin[iH], Tmin[iC]), min(Tmax[iH], Tmax[iC]), 500)
T_H = scipy.interpolate.UnivariateSpline(H_H(T), T, k=1, s=0)
T_C = scipy.interpolate.UnivariateSpline(H_C(T), T, k=1, s=0)

# plt.figure(facecolor='white')
# plt.suptitle("Enthalpy-temperature plot")
# plt.title("Hot side: %s & Cold side: %s" % (compH, compC))
# plt.plot(T, H_H(T), color='red', linewidth=1.2, label=compH)
# plt.plot(T, H_C(T), color='blue', linewidth=1.2, label=compC)
# plt.xlabel(r'$T\, (K)$')
# plt.ylabel(r'$H\, (J/kg)$')
# plt.legend(loc='best', fontsize=8)


class CrossCurrentHX:
    def __init__(self):
        self.U = 350.0  # W/m2-K
        self.P = np.pi * 0.1  # m2/m
        self.L = 1.0  # m

        self.mH = 0.05  # kg/s
        self.mC = 0.10  # kg/s
        self.T_Hin = 323.16  # K
        self.T_Cin = 303.16  # K

    def model(self, H, z):
        [H_H, H_C] = H

        dH_Hdz = -self.U * self.P * (T_H(H_H) - T_C(H_C)) / self.mH
        dH_Cdz = -self.U * self.P * (T_H(H_H) - T_C(H_C)) / self.mC

        return [dH_Hdz, dH_Cdz]

    def initialize(self):
        T = np.linspace(self.T_Cin, self.T_Hin, 1000)
        self.T_H = scipy.interpolate.UnivariateSpline(H_H(T), T, k=1, s=0)
        self.T_C = scipy.interpolate.UnivariateSpline(H_C(T), T, k=1, s=0)

    def shoot(self, T_Cout):
        self.T_Cout = T_Cout
        H0 = [H_H(self.T_Hin), H_C(self.T_Cout)]
        z = [0, self.L]

        solution = scipy.integrate.odeint(self.model, H0, z)
        H_Cin = solution[-1, 1]
        T_Cin = T_C(H_Cin)

        error = [T_Cin - self.T_Cin]
        return error

    def solve(self, n=100):
        self.initialize()

        guess = [self.T_Cin + 0.0]

        lsq = scipy.optimize.least_squares(self.shoot, guess)

        H0 = [H_H(self.T_Hin), H_C(self.T_Cout)]
        z = np.linspace(0, self.L, n)

        sol = scipy.integrate.odeint(self.model, H0, z)
        H_Hsol = sol[:, 0]
        H_Csol = sol[:, 1]

        self.dfsol = pd.DataFrame({"z": z,
                                   "T_H": self.T_H(H_Hsol),
                                   "T_C": self.T_C(H_Csol)})


hx = CrossCurrentHX()
hx.solve()

T_Hout = hx.dfsol.T_H.iloc[-1]
T_Cout = hx.dfsol.T_C.iloc[0]

# plt.figure(facecolor='white', figsize=(7.5, 7.5))
# plt.grid()
# plt.suptitle("Enthalpy profile along the length of a cross-current heat exchanger")
# plt.title("Hot side: %s & Cold side: %s" % (compH, compC))
# plt.plot(hx.dfsol.z, hx.dfsol.T_H, color='red', linewidth=1.2, label=compH)
# plt.plot(hx.dfsol.z, hx.dfsol.T_C, color='blue', linewidth=1.2, label=compC)
# plt.annotate(r'$T_H^{in}$', xy=(0, hx.T_Hin))
# plt.annotate(r'$T_C^{in}$', xy=(hx.L, hx.T_Cin))
# plt.annotate(r'$T_H^{out}$', xy=(hx.L, T_Hout))
# plt.annotate(r'$T_C^{out}$', xy=(0, T_Cout))
# plt.xlabel(r'$L\, (m)$')
# plt.ylabel(r'$T\, (K)$')
# plt.legend(loc='best', fontsize=8);

# plt.show()
