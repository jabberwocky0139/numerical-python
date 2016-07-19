# coding: utf-8

import sys
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
from scipy.linalg import solve
from scipy.integrate import quad, simps


class Variable(object):

    def __init__(self):
        
        # --*-- Constants --*--
        
        # 全粒子数 N, 逆温度 β, 相互作用定数 g 
        self.N, self.BETA, self.G = 1e5, 1e-3, 1e-3
        self.GN = self.N*self.G
        # プランク定数 
        self.H_BAR = 1
        # 質量 m 
        self.M = 1
        # r空間サイズ, 分割数
        self.VR, self.NR = 10, 200
        # r空間微小幅
        self.DR = self.VR/self.NR
        # 3次元等方r空間
        self.R = np.linspace(self.DR, self.VR, self.NR)
        # q表示における系のサイズ L, ゼロモード空間の分割数 Nq 
        self.L, self.NQ = 1, 200
        # q表示においてNq分割 dq 
        self.DQ = self.L/self.NQ
        
        # --*-- Variables --*--
        
        # 化学ポテンシャル μ, 秩序変数 ξ, 共役モード η, 共役モードの規格化変数 I
        self.mu, self.xi, self.eta, self.I = [None]*4
        # 熱平均, 異常平均
        self.Vt, self.Va= [0]*2
        # Pの平均 <P>, Q^2の平均 <Q^2>, P^2の平均 <P^2> 
        self.P, self.Q2, self.P2 = [None]*3
        # 積分パラメータ A,B,C,D,E 
        self.A, self.B, self.C, self.D, self.E  = [None]*5
        # 励起エネルギー ω, 比熱・圧力用の全エネルギー 
        self.omega, self.U = [None]*2



class GrossPitaevskii():
    
    dt = 0.01
    
    # Set initial function
    @classmethod
    def __psi0(cls, v):
        return np.exp(-v.R**2)

    
    # Make matrix for Crank-Nicolson
    @classmethod
    def __make_crank_matrix(cls, v):
        
        a = np.diag(-2 + 2*v.DR**2*(v.mu - v.R**2/2 - v.GN*v.xi**2/(8*np.pi) + 2*v.Vt + v.Va - 2/cls.dt))
        c = np.diag(1 + 1/np.arange(0, v.NR))
        c = np.vstack((c[1:], np.array([0]*v.NR)))
        
        d = np.diag(1 - 1/np.arange(1, v.NR+1))
        d = np.vstack((d[1:], np.array([0]*v.NR))).T
        
        return a + c + d

    
    # Make vector for Crank-Nicolson
    @classmethod
    def __make_crank_vector(cls, v):
        
        a = (-2 + 2*v.DR**2*(v.mu - v.R**2/2 - v.GN*v.xi**2/(8*np.pi) + 2*v.Vt + v.Va + 2/cls.dt))*v.xi
        c = (1 + 1/np.arange(1, v.NR+1))*np.hstack((v.xi[1:], [0]))
        d = (1 - 1/np.arange(1, v.NR+1))*np.hstack(([0], v.xi[:v.NR-1]))
        
        return -(a + c + d)

    
    # Time evolution
    @classmethod
    def __time_evolution(cls, v):
        
        # Solve matrix equation
        v.xi = solve(cls.__make_crank_matrix(v), cls.__make_crank_vector(v))
        
        # Correction of chemical potential mu
        norm = simps(v.R**2*v.xi**2, v.R)
        v.mu += (1 - norm)/(2*cls.dt)
        
        # Normalize arr_Psi
        v.xi /= np.sqrt(norm)

    
    # Time evolution loop
    @classmethod
    def __solve_imaginarytime_gp(cls, v):
                
        print("\n--*-- GP --*--")
        old_mu = 0
        v.xi, v.mu = cls.__psi0(v), 10
        
        while(np.abs(v.mu - old_mu) > 1e-5):
            old_mu = v.mu
            cls.__time_evolution(v)
            sys.stdout.write("\r mu = {0}".format(v.mu))
            sys.stdout.flush()
        
        print("")

    
    # Plot xi
    @classmethod
    def __plot_procedure(cls, v):

        # --*-- Matplotlib configuration --*--
        #plt.plot(v.R, cls.__psi0(v), label="Initial")
        plt.plot(v.R, v.xi**2, label="gN = {0}".format(v.GN))
        plt.plot(v.R, v.R**2/2, label="Potential")
        plt.ylim(0, max(v.xi**2)*1.1)
        plt.xlim(0, v.R[-1])
        plt.xlabel("r-space")
        plt.ylabel("Order parameter")
        plt.title("Static vortex of Gross-Pitaevskii equation")
        plt.legend(loc = 'center right')
        plt.show()

    
    # Hundle
    @classmethod
    def procedure(cls, v):
        cls.__solve_imaginarytime_gp(v)
        cls.__plot_procedure(v)



class AjointModeEquation(object):

    # Make equation matrix
    @classmethod
    def __make_matrix(cls, v):
        
        dr = np.diag(1/v.DR**2 + v.R**2/2 - v.mu + v.GN/(4*np.pi)*(3*v.xi + 2*v.Vt - v.Va))
        
        eu = np.diag(-1/(2*v.DR**2)*(1 + 1/np.arange(0, v.NR)))
        eu = np.vstack((eu[1:], np.array([0]*v.NR)))
        
        ed = np.diag(-1/(2*v.DR**2)*(1 - 1/np.arange(1, v.NR+1)))
        ed = np.vstack((ed[1:], np.array([0]*v.NR))).T

        return dr + eu + ed

    
    # Obtain eta and I
    @classmethod
    def __solve_ajoint_equation(cls, v):
        
        print("--*-- Ajoint --*--")
        etatilde = solve(cls.__make_matrix(v), v.xi)

        v.I = 1/(2*v.N)*(simps(v.R**2 * etatilde * v.xi))**-1
        v.eta = v.I*etatilde
        print("I = {0}".format(v.I))

    
    # Plot eta
    @classmethod
    def __plot_procedure(cls, v):

        # --*-- Matplotlib configuration --*--
        plt.plot(v.R, v.eta, label="gN = {0}".format(v.GN))
        plt.xlim(0, v.R[-1])
        plt.xlabel("r-space")
        plt.ylabel("Ajoint parameter")
        plt.title("Solution of AjointMode equation")
        plt.legend(loc = 'center right')
        plt.show()

    
    # Hundle
    @classmethod
    def procedure(cls, v):
        
        cls.__solve_ajoint_equation(v)
        cls.__plot_procedure(v)



if(__name__ == "__main__"):
    
    var = Variable()
    GrossPitaevskii.procedure(v=var)
    AjointModeEquation.procedure(v=var)
