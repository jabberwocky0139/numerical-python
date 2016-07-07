# coding: utf-8

import numpy as np
from scipy.integrate import simps
from scipy.linalg import solve
from scipy.fftpack import fft2
from scipy.fftpack import ifft2
from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt
import seaborn as sbn

class GrossPitaevskii():
    
    # --*-- Set constants and variables for initial condition --*--
    def __init__(self, gN=50):

        # Volume, Maximum x-step
        self.L, self.xN = 12, 256

        # Interaction constant, Chemical potential
        self.gN, self.mu, self.kappa = gN, 10, 2

        # Maximum time, Time step 
        self.tMax, self.tN = 1, 512

        # x-step, Time step
        self.h, self.dt = self.L/self.xN, self.tMax/self.tN

        # Set x-space
        self.x = np.linspace(0, self.L, self.xN)

        # Set time propagator of poptential term except non-linear term
        self.arr_Psi = self.__Psi_0(self.x)
        

        # -*- For 2D-Symplectic -*-        
                
        #self.xy = self.h*np.sqrt(self.y2d**2+self.x2d**2)
        self.xy = [[0 for i in range(self.xN)] for j in range(self.xN)]
        
        for i in range(self.xN):
            for j in range(self.xN):
                self.xy[i][j] = self.h*np.sqrt((i - self.xN/2)**2 + (j - self.xN/2)**2)
        self.xy = np.array(self.xy)

        # Set kx, ky
        self.kx = 2*np.pi*fftfreq(self.xN, d=1/self.xN)/self.L
        self.ky = self.kx.reshape(self.xN, 1)

        # Set time propagator of poptential term except non-linear term
        self.arr_Psi2D = None
        #self.pre_expV = np.exp((self.mu - self.__V(self.xy))*self.dt)
        self.pre_expV = None
        #self.expK = np.exp(-(self.kx**2 + self.ky**2)*self.dt)
        self.expK = None
        #self.expV = np.exp((-self.gN*self.arr_Psi2D**2)*self.dt)*self.pre_expV
        self.expV = None
    
    # --*-- Set functions --*--

    # Set initial function
    def __Psi_0(self, x):
        return x**(self.kappa)*np.exp(-x**2/2)

    # Set potential
    def __V(self, x):
        #return 25*np.sin(x/5)**2
        return x**2


    def __VInst(self):
        pot = [[0 for i in range(self.xN)] for j in range(self.xN)]
        
        for i in range(self.xN):
            for j in range(self.xN):
                theta = np.angle(i - self.xN/2 + 1j*(j - self.xN/2))
                pot[i][j] = self.h**2*((i - self.xN/2)**2 + (j - self.xN/2)**2)*np.cos(self.kappa*theta)

        return np.array(pot)
    

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
        norm = simps(self.x*self.arr_Psi**2, self.x)/(2*np.pi)
        self.mu -= (norm - 1)/(2*self.dt)

        # 規格化 #
        self.arr_Psi = self.arr_Psi/np.sqrt(norm)

        
    def GaussSeidelLoop(self):
        old_mu = 0
        while(np.abs(self.mu - old_mu) > 1e-7):
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
        self.arr_Psi2D = np.array(self.arr_Psi2D)

    # Time evolution
    def Symplectic(self):

        self.pre_expV = np.exp(-1j*self.__V(self.xy)*self.dt)
        self.pre_expV_inst = np.exp(-1j*self.__VInst()*self.dt)
        self.expK = np.exp(-1j*(self.kx**2 + self.ky**2)*self.dt)
        self.expV = np.exp(-1j*(-self.mu + self.gN*self.arr_Psi2D**2)*self.dt)*self.pre_expV_inst
        
        for i in range(self.tN):
            print(i)
            # Time evolution
            self.arr_Psi2D = ifft2(fft2(self.arr_Psi2D*self.expV)*self.expK)
            
            # Correction of chemical potential mu
            # 矩形積分. 雑なので直したい
            norm = np.sum(np.abs(self.arr_Psi2D)**2)*self.h**2
            self.mu -= (norm - 1)/(2*self.dt)
            print(self.mu)

            # Normalization of order parameter arr_Psi
            self.arr_Psi2D /= np.sqrt(norm)

            # Reconfigure expV
            if(i > 30):
                self.expV = np.exp(-1j*(-self.mu + self.gN*self.arr_Psi2D**2)*self.dt)*self.pre_expV
            else:
                self.expV = np.exp(-1j*(-self.mu +self.gN*self.arr_Psi2D**2)*self.dt)*self.pre_expV_inst
        

                
            
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
            plt.pcolor(X, Y, np.abs(self.arr_Psi2D))
            plt.clim(0, 0.5)
            plt.colorbar()
        elif(type == "angle"):
            plt.pcolor(X, Y, np.angle(self.arr_Psi2D))
        
        plt.axis("equal")
        plt.hot()
        plt.xlim(0, self.xN)
        plt.ylim(0, self.xN)
        plt.show()
        



hundle = GrossPitaevskii(gN=5)

hundle.GaussSeidelLoop()

hundle.ExpandFor2D()
#hundle.PrintProcedureFor2D(type="abs")
#hundle.PrintProcedureFor2D(type="angle")
hundle.Symplectic()
hundle.PrintProcedureFor2D(type="abs")
hundle.PrintProcedureFor2D(type="angle")


