# coding: utf-8

import numpy as np
from scipy import linalg
from scipy import optimize
from scipy.integrate import simps
from scipy.fftpack import fft, fftfreq
from scipy.misc import factorial
from scipy.special import eval_hermite
import matplotlib.pyplot as plt
import seaborn as sbn
from abc import abstractmethod, ABCMeta

# plot container
plot_real, plot_imag = [], []

# --*-- Constants' and variables' class --*--
class Variables():
    def __init__(self, N):
        # -*- Constants on real space -*-

        # Particle number
        self.N = N
        # Volume
        self.V = 1
        # Interaction constant
        self.g = 1e-3
        # Integrals for oder parameter and ajoint parameter
        self.A = self.g * self.N**2 / self.V
        self.B = self.g * self.N / self.V / 2
        self.C = self.g / self.V / 4
        self.D = self.g / self.V / self.N / 8
        self.E = self.g / self.V / self.N**2 / 16
        self.I = self.g / self.V

        # -*- Constant on Zeromode space -*-

        # Volume of Zeromode q-space
        self.L = 20 * (self.N * self.V)**(-1 / 3)
        # Step numbers of Zeromode q-space
        self.Nq = 600
        # Step size of Zeromode q-space
        self.dq = self.L / self.Nq
        # Zeromode q-space coordinate
        self.q = np.arange(self.Nq)
        


# --*-- Basis class --*--
class Procedures(metaclass=ABCMeta):

    # Make Zeromode hamiltonian(Lower triangle)
    def __MakeZeromodeHamiltonian(self, v, dmu):
        """debug: P**4 + Q**2 のみを残した子"""
        P_b = - 0.5j * (dmu + 4 * v.C) / v.dq
        P2_a, P2_b = (v.I - 4 * v.D) / v.dq**2, - 0.5 * (v.I - 4 * v.D) / v.dq**2
        P3_b, P3_g = 2.0j * v.D / v.dq**3, - 1j * v.D / v.dq**3
        P4_a, P4_b, P4_g = 3 * v.E / v.dq**4, -2 * v.E / v.dq**4, 0.5 * v.E / v.dq**4
        Q2_a = - 2 * v.dq**2 * v.B * (v.q - v.Nq / 2.0)**2
        Q4_a = 0.5 * v.dq**4 * v.A * (v.q - v.Nq / 2.0)**4
        QPQ_b = 1j * v.dq * v.B * (v.q - v.Nq / 2) * (v.q - v.Nq / 2 + 1)
        QP2Q_a, QP2Q_b = 2 * v.C * (v.q - v.Nq / 2.0)**2, -v.C * (v.q - v.Nq / 2) * (v.q - v.Nq / 2 + 1)
        
        #alpha = 3 * v.E / v.dq**4 + (v.I - 4 * v.D) / v.dq**2 + 2 * v.C * (v.q - v.Nq / 2.0)**2 - 2 * v.dq**2 * v.B * (v.q - v.Nq / 2.0)**2 + 0.5 * v.dq**4 * v.A * (v.q - v.Nq / 2.0)**4
        alpha = P2_a + Q4_a # debug
        
        #beta = -2 * v.E / v.dq**4 + 2.0j * v.D / v.dq**3 - 0.5 * (v.I - 4 * v.D) / v.dq**2 - 0.5j * (dmu + 4 * v.C) / v.dq - (v.C - 1j * v.dq * v.B) * (v.q - v.Nq / 2) * (v.q - v.Nq / 2 + 1)
        beta = P2_b + P_b + QPQ_b # debug
        
        #gamma = [0.5 * v.E / v.dq**4 - 1j * v.D / v.dq**3] * v.Nq
        gamma = [0] * v.Nq # debug
        
        return np.vstack((np.vstack((alpha, beta)), gamma))

    # Solve eigenvalue problem for Zeromode hamiltonian
    def ZeromodeEquation(self, v, dmu):
        Hqp = self.__MakeZeromodeHamiltonian(v, dmu)
        # Only ground state
        w, val = linalg.eig_banded(
            Hqp,
            lower=True,
            select="i",
            select_range=(0, 200),
            overwrite_a_band=True)
        
        #val = val.reshape(v.Nq)

        return w, val

    # Output expected value of Zeromode operator P
    def OutputP(self, v, dmu):
        _, val = self.ZeromodeEquation(v, dmu)
        val = val.T[0]
        # Check boundary value of Zeromode q-space
        if (np.abs(val[0]) > 0.01):
            print("warning!!!")

        return np.imag(np.dot(np.conj(val[:-1]), val[1:])) / v.dq

    # Find proper counter term dmu
    def SelfConsistent(self, v):
        dmu = optimize.bisect(lambda pro_dmu: self.OutputP(v, pro_dmu), -1e-1, 2e-1)

        return dmu

    def SetPlot(self,
                plot_x,
                plot_y,
                xlim=False,
                ylim=False,
                logscale_x=False,
                logscale_y=False,
                xlabel=False,
                ylabel=False,
                title=False):
        plt.plot(plot_x, plot_y)
        if (title):
            plt.title(title)
        if (xlabel):
            plt.xlabel(xlabel)
        if (ylabel):
            plt.ylabel(ylabel)
        if (xlim):
            plt.xlim(xlim[0], xlim[1])
        if (ylim):
            plt.ylim(ylim[0], ylim[1])
        if (logscale_x):
            plt.xscale("log")
        if (logscale_y):
            plt.yscale("log")

        plt.show()

    # Use this method with inheritance
    @abstractmethod
    def Procedure(self, N):
        pass


class OutputP_Mu(Procedures):

    # Set range of dmu
    def __init__(self, start=-0.5, end=0.5):
        self.start = start
        self.end = end

    def Procedure(self, N):
        v = Variables(N)
        ans = []
        ran = [self.start, self.end]
        for n, dmu in enumerate(np.linspace(ran[0], ran[1], 1000)):
            ans.append(self.OutputP(v, dmu))
            print(n, '/ 1000', '\r',  end="", flush=True)

        self.SetPlot(
            plot_x=np.linspace(ran[0], ran[1], 1000),
            plot_y=ans,
            xlim=ran,
            xlabel="dμ",
            ylabel="P",
            title="Expected value P for counter term dμ")


class OutputZeromodeGroundFunction(Procedures):

    # Set free pro_dmu
    def __init__(self, pro_dmu=False):
        self.pro_dmu = pro_dmu

    def Procedure(self, N):
        v = Variables(N)

        if (self.pro_dmu):
            dmu = self.pro_dmu
        else:
            dmu = self.SelfConsistent(v)
            
        w, val = self.ZeromodeEquation(v, dmu)
        igen_func = val.T
        val = val.T[0]
        

        
        #print(w[1]-w[0], '\r', end='')
        #f = open('p3_energy_N.txt', 'a')
        # for index, value in enumerate(np.diff(w)):
        #     print('{0}\t{1}'.format(index, value), file=f)
        #print('{0}\t{1}'.format(v.N, w[1] - w[0]), file=f)

        val /= v.dq**0.5
        v.q = (v.q - v.Nq/2) * v.dq
        P = -1j * v.g * simps(val.conjugate()*np.gradient(val, v.dq), v.q)
        A = v.g * v.N**2 / v.V
        B = v.g * v.N / v.V / 2
        I = v.g
        
        E_arr = []
        # for n, _ in enumerate(igen_func):
        #     if n == 0:
        #         alpha = np.sqrt(2) / 2 * v.N**(-1/3)
        #     else:
        #         numerator = np.sqrt(2) * (n + 1) * (4 * n**2 - 1) + np.sqrt(2 * (4 * n**2 - 1) * (36 * n**4 + 72 * n**3 + 59 * n**2 + 30 * n - 25))
        #         dedominator = 8 * (4 * n**2 - 1) * (2 * n + 3)
        #         alpha = (numerator/dedominator)**(1/3) * v.N**(-1/3)
        #         print(alpha)
            
        #     Q4 = A / 2 * (2 * n + 1) * (2*n + 3) * alpha**4
        #     P2 = I / 2 * (2 * n**3 + n**2 + 2 * n - 1) / (2 * (2 * n - 1) * alpha**2)
        #     QPQ = -np.sqrt(2) * B * (n + 1) * (2 * n + 1) * alpha
        #     E_arr.append(Q4 + P2 + QPQ)

        # print(E_arr)
        # plt.plot(np.diff(w), label='numerical')
        # plt.plot(np.diff(E_arr), '--', label='variational')
        # print(P, QPQ, P2, Q4)
        
        # self.SetPlot(
        #     plot_x=v.q,
        #     plot_y=np.abs(val)**2,
        #     xlabel="q",
        #     ylabel="Psi_0",
        #     title="Zeromode ground function for a dmu")

        # plt.plot(v.q, np.real(val)**2, label='N={0:d}'.format(v.N)

        ## modifiy phase
        theta = np.angle(val[int(v.Nq/2)])
        val = val / np.exp(1j * theta)

        ## psi
        plt.plot(v.q, np.real(val), label='real part of zeromode function')
        plt.plot(v.q, np.imag(val), label='iamginary part of zeromode function')

        
        # plt.plot(v.q, np.sqrt(np.real(val)**2 + np.imag(val)**2), label='QPQ')
        
        def psi(x, alpha):
            beta = 1 / (2 * np.sqrt(2) * alpha)
            gamma = -1 / (6 * np.sqrt(2) * alpha**3)
            return (1 / (2 * np.pi * alpha**2))**0.25 * np.exp(-x**2 / (4 * alpha**2) + 1j * (beta * x + gamma * x**3))

        def psi_general(x, n, alpha):
            beta = 1 / (2 * np.sqrt(2) * alpha)
            gamma = -1 / (6 * np.sqrt(2) * alpha**3)
            C = np.sqrt(2**n * factorial(n) / (np.sqrt(2 * np.pi) * factorial(2 * n) * alpha**(2 * n + 1)))
            return C * x**n * np.exp(-x**2 / (4 * alpha**2) + 1j * (beta * x + gamma * x**3))
            # return C * eval_hermite(n, x/alpha) * np.exp(-x**2 / (4 * alpha**2) + 1j * (beta * x + gamma * x**3))


            
        

        
        n = 0
        numerator = np.sqrt(2) * (n + 1) * (4 * n**2 - 1) + np.sqrt(2 * (4 * n**2 - 1) * (36 * n**4 + 72 * n**3 + 59 * n**2 + 30 * n - 25))
        dedominator = 8 * (4 * n**2 - 1) * (2 * n + 3)
        # alpha = (numerator/dedominator)**(1/3) * v.N**(-1/3)
        # alpha = 0.588 * (v.N)**(-1/3)
        # plt.plot(v.q, np.real(psi_general(v.q, n, alpha)), '--', label='real part(variational)')
        # plt.plot(v.q, np.imag(psi_general(v.q, n, alpha)), '--', label='imaginary part(variational)')
        # plt.plot(v.q, np.sqrt(psi(v.q, alpha) * np.conj(psi(v.q, alpha))), '--', label=r'$Q^4$')
        

        ## QPQ
        # plt.plot(v.q, np.real(-1j * v.q**2 * val.conjugate()*np.gradient(val, v.dq)), label='Integrand of <QPQ> for real-part')
        # plt.plot(v.q, np.imag(-1j * v.q**2 * val.conjugate()*np.gradient(val, v.dq)), label='Integrand of <QPQ> for imaginary-part')

        ## P
        # plt.plot(v.q, np.real(-1j * val.conjugate()*np.gradient(val, v.dq)), label='Integrand of <P> for real-part')
        # plt.plot(v.q, np.imag(-1j * val.conjugate()*np.gradient(val, v.dq)), label='Integrand of <P> for imaginary-part')

        # int_real = simps(np.real(-1j * val.conjugate()*np.gradient(val, v.dq))**2, v.q)
        # int_imag = simps(np.imag(-1j * val.conjugate()*np.gradient(val, v.dq))**2, v.q)
        # int_real = simps(np.real(val)**2, v.q)
        # int_imag = simps(np.imag(val)**2, v.q)
        # print(int_real/int_imag)
        # plt.plot(v.Nq, int_real/int_imag)

        # plt.plot(v.q, np.real(-1j * val.conjugate()*np.gradient(val, v.dq)))
        # plt.plot(v.q, np.imag(-1j * val.conjugate()*np.gradient(val, v.dq)))
        
        # plt.plot(v.q, np.imag(val), label='psi for imaginary-part')
        # plt.plot(v.q, np.imag(np.gradient(val, v.dq)), label='diff psi for imaginary-part')
        # plt.plot(v.q, np.imag(val.conjugate()*np.gradient(val, v.dq)), label='Integrand of <P> for imaginary-part')
        # plt.plot(v.q, v.q**2 * np.imag(val.conjugate()*np.gradient(val, v.dq)), label='Integrand of <QPQ> for imaginary-part')
        
        # plt.xlabel(r'$q$', fontsize=18)
        # plt.ylabel(r'$\langle q|\psi \rangle$', fontsize=18)
        # plt.xlim(-v.L/2, v.L/2)
    
        #plt.plot(v.q, np.sqrt(np.real(val)**2 + np.imag(val)**2), label='sum of real and imaginary(modified)')

        # plot_real.append(np.real(val[int(v.Nq/2)]))
        # plot_imag.append(np.imag(val[int(v.Nq/2)]))
        # print(v.N, np.imag(val[int(v.Nq/2)]), np.real(val[int(v.Nq/2)]), '\r', end='')

        
        # from scipy.optimize import curve_fit

        # def fitting_real(x, a, b, c, d):
        #     return c * np.cos(a * x**3 + d * x**5) * np.exp(-b*x**2)

        # def fitting_imag(x, a, b, c, d):
        #     return c * np.sin(a * x + d * x**3) * np.exp(-b*x**2)

        # x = np.linspace(-v.L/2, v.L/2, v.Nq)
        # param_real = curve_fit(fitting_real, x, np.real(val), maxfev=10000)[0]
        # #param_imag = curve_fit(fitting_imag, x, np.imag(val), maxfev=10000)[0]

        # print(param_real)
        # #print(param_imag)
        
        # plt.plot(x, fitting_real(x, param_real[0], param_real[1], param_real[2], param_real[3]))
        # #plt.plot(x, fitting_imag(x, param_imag[0], param_imag[1], param_imag[2], param_imag[3]))
        # plt.plot(x, np.real(val), label='real part of zeromode function')

        plt.xlim(-v.L/2, v.L/2)
        plt.xlabel(r'$q$', fontsize=18)
        plt.ylabel(r'$\Psi_q$', fontsize=18)
        plt.legend()
        plt.show()
        

class OutputQ2_N(Procedures):

    # Set range of N which is power of 10 (like 10^N)
    def __init__(self, N_start=2, N_end=4):
        self.N_start = N_start
        self.N_end = N_end

    def Procedure(self, N=None):
        arr_Q2 = []
        arr_N = np.logspace(self.N_start, self.N_end, num=100)
        for N in arr_N:
            v = Variables(N)
            dmu = self.SelfConsistent(v)
            _, val = self.ZeromodeEquation(v, dmu)
            val = val.T[0]
            print("N = {1}, dmu = {0}".format(dmu, N), '\r', end='')
            arr_Q2.append(
                np.sqrt(v.dq**2 * (np.sum(np.abs((v.q - v.Nq / 2) * val)**2))))

        self.SetPlot(
            plot_x=arr_N,
            plot_y=arr_Q2,
            xlim=(arr_N[0], arr_N[-1]),
            ylim=(0.01, 1),
            logscale_x=True,
            logscale_y=True,
            xlabel="N",
            ylabel="Q^2",
            title="Expected value Q^2 for N")


if __name__ == "__main__":

    Q2 = OutputQ2_N()
    Zero = OutputZeromodeGroundFunction()
    PMu = OutputP_Mu()
    # P-muグラフの出力
    # PMu.Procedure(N=1e4)
    # ゼロモード固有関数の出力
    # for N in range(5000, 50000, 100):
    #     Zero.Procedure(N)
    # f = open('variational_energy_N', 'w')
    # for N in range(5000, 50000, 1000):
    #     Zero.Procedure(N)
    
    # Zero.Procedure(N=1e4)
    # Zero.Procedure(N=1e5)
    Zero.Procedure(N=1e6)

    
    # plt.plot(range(5000, 50000, 1000), plot_imag, label='imaginary part')
    # plt.plot(range(5000, 50000, 1000), plot_real, label='real part')
    # plt.xlabel(r'$q$', fontsize=18)
    # plt.ylabel(r'${\Re}[\Psi]$', fontsize=18)
    # plt.xlabel(r'$N_0$', fontsize=18)
    # plt.ylabel(r'$\Psi_q$', fontsize=18)
    # plt.legend()
    # plt.show()
