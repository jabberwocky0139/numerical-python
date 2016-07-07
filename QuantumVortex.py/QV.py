# coding: utf-8

import numpy as np
from scipy.integrate import simps
from scipy.linalg import solve
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt
import seaborn as sbn

class GrossPitaevskii():
    
    # --*-- Set constants and variables for initial condition --*--
    def __init__(self, gN=50):

        # Volume, Maximum x-step
        self.L, self.xN = 16, 300

        # Interaction constant, Chemical potential
        self.gN, self.mu, self.kappa = gN, 10, 2

        # Maximum time, Time step 
        self.tMax, self.tN = 2, 512

        # x-step, Time step
        self.h, self.dt = self.L/self.xN, self.tMax/self.tN

        # Set x-space
        self.x = np.linspace(0, self.L, self.xN)

        # Set time propagator of poptential term except non-linear term
        self.arr_Psi = self.__Psi_0(self.x)
        self.arr_Psi2D = None

    
    # --*-- Set functions --*--

    # Set initial function
    def __Psi_0(self, x):
        return x**(self.kappa)*np.exp(-x**2/2)

    # Set potential
    def __V(self, x):
        #return 25*np.sin(x/5)**2
        return x**2
    

    def __MakeCrankMatrix(self):
        
        a = np.diag(-2 - self.kappa**2/np.arange(1, self.xN+1)**2 + self.h**2*(self.mu - self.__V(self.x) - self.gN*self.arr_Psi**2 - 2/self.dt))
                
        c = np.diag(1 + 0.5/np.arange(0, self.xN))
        c = np.vstack((c[1:], np.array([0]*self.xN)))
                
        d = np.diag(1 - 0.5/np.arange(1, self.xN+1))
        d = np.vstack((d[1:], np.array([0]*self.xN))).T
        
        # Debug on non-interacting system
        #a = np.diag(-2 + self.h**2*(self.mu - self.__V(self.x) - self.gN*self.arr_Psi**2 - 2/self.dt))
        #c = np.vstack((np.eye(self.xN)[1:], np.array([0]*self.xN)))
        #d = c.T
        
        return a + c + d

    
    def __MakeCrankVector(self):
        
        a = (-2 - self.kappa**2/np.arange(1, self.xN+1)**2 + self.h**2*(self.mu - self.__V(self.x) - self.gN*self.arr_Psi**2 + 2/self.dt))*self.arr_Psi
                
        c = (1 + 0.5/np.arange(1, self.xN+1))*np.hstack((self.arr_Psi[1:], [0]))
                
        d = (1 - 0.5/np.arange(1, self.xN+1))*np.hstack(([0], self.arr_Psi[:self.xN-1]))

        # Debug on non-interacting system
        #a = (-2 + self.h**2*(self.mu - self.__V(self.x) - self.gN*self.arr_Psi**2 + 2/self.dt))*self.arr_Psi
        #c = np.hstack((self.arr_Psi[1:], [0]))
        #d = np.hstack(([0], self.arr_Psi[:self.xN-1]))
        
        return -a - c - d

    
    def __GaussSeidel(self):
        #--- 行列の用意 ---#
        a = self.__MakeCrankMatrix()
        #--- ベクトルの用意 ---#
        b = self.__MakeCrankVector()
        
        #--- 連立方程式の計算 ---#
        self.arr_Psi = solve(a, b)
        #--- μの補正 ---#
        norm2 = simps(self.arr_Psi**2, self.x)
        self.mu -= (norm2 - 1)/(2*self.dt)

        # 規格化 #
        self.arr_Psi = self.arr_Psi/np.sqrt(norm2)

        
    def GaussSeidelLoop(self):
        old_mu = 0
        while(np.abs(self.mu - old_mu) > 1e-5):
            old_mu = self.mu
            self.__GaussSeidel()
            #self.PrintProcedure()
            print(self.mu)

            
    def ExpandFor2D(self):
        self.arr_Psi2D = [[0 for i in range(self.xN)] for j in range(self.xN)]

        for i in range(self.xN):
            for j in range(self.xN):
                index = int(np.sqrt((i - self.xN/2)**2 + (j - self.xN/2)**2))
                theta = np.angle(i - self.xN/2 + 1j*(j - self.xN/2))
                self.arr_Psi2D[i][j] = self.arr_Psi[index]*np.exp(1j*self.kappa*theta)
                
            
    def PrintProcedureFor1D(self):

        # --*-- Matplotlib configuration --*--
        plt.plot(self.x, np.real(self.arr_Psi**2), label="gN = {0}".format(self.gN))
        plt.plot(self.x, self.__V(self.x)/50, label="Potential\n(provisional)")
        plt.ylim(0, 0.5)
        plt.xlim(0, 6)
        plt.xlabel("q-space")
        plt.ylabel("Order parameter")
        plt.title("Static soliton of Gross-Pitaevskii equation")
        plt.legend(loc = 'center right')
        plt.show()
        

    def PrintProcedureFor2D(self, type="abs"):
        
        x = np.arange(self.xN)
        y = np.arange(self.xN)
        X, Y = np.meshgrid(x, y)
        
        if(type == "abs"):
            plt.pcolor(X, Y, np.real(np.abs(self.arr_Psi2D)))
        else(type == "angle"):
            plt.pcolor(X, Y, np.real(np.angle(self.arr_Psi2D)))
            
        plt.axis("equal")
        plt.xlim(0, self.xN)
        plt.ylim(0, self.xN)
        plt.show()
        


#GrossPitaevskii.PrintProcedure()
hundle = GrossPitaevskii(gN=50)
#print(hundle.MakeCrankVector())
hundle.GaussSeidelLoop()
#hundle.PrintProcedureFor1D()
hundle.ExpandFor2D()
hundle.PrintProcedureFor2D()

