# coding: utf-8
 
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
from scipy.linalg import solve, eig, eigh, eig_banded
from scipy.linalg.lapack import dgeev
from scipy.integrate import quad, simps
from scipy import optimize, special
from abc import abstractmethod, ABCMeta
from tqdm import tqdm
from pprint import pprint
# Clean up some warnings we know about so as not to scare the users
import warnings
from matplotlib.cbook import MatplotlibDeprecationWarning
warnings.simplefilter('ignore', MatplotlibDeprecationWarning)
 
 
class Variable(object):
    def __init__(self):
 
        # --*-- Constants --*--
 
        # 凝縮粒子数 N, 逆温度 β, 相互作用定数 g
        self.N0, self.G = 1e3, 1e-3
        self.Temp = (self.N0/special.zeta(3, 1))**(1/3)*0.1
        self.BETA = 1/self.Temp
        
        self.GN = self.N0 * self.G
        self.G_TILDE = self.GN / (4 * np.pi)
        # プランク定数
        self.H_BAR = 1
        # 質量 m
        self.M = 1
        # r空間サイズ, 分割数
        self.VR, self.NR = 10, 300
        # r空間微小幅
        self.DR = self.VR / self.NR
        # 3次元等方r空間
        self.R = np.linspace(self.DR, self.VR, self.NR)
        # q表示における系のサイズ L, ゼロモード空間の分割数 Nq
        self.L, self.NQ = 30 * (self.N0 * self.VR)**(-1 / 3), 500
        self.NM = self.NQ*2/3
        # q表示においてNq分割 dq
        self.DQ = self.L / self.NQ
        self.Q = np.arange(self.NQ)
 
        # --*-- Variables --*--
 
        # 化学ポテンシャル μ, 秩序変数 ξ, 共役モード η, 共役モードの規格化変数 I
        self.mu, self.xi, self.eta, self.I = [0] * 4
        self.promu, self.proI = [1] * 2
        # 熱平均, 異常平均
        self.Vt = np.array([0 for _ in range(self.NR)])
        self.Va = np.array([0 for _ in range(self.NR)])
        
        # Pの平均 <P>, Q^2の平均 <Q^2>, P^2の平均 <P^2>
        self.P, self.Q2, self.P2 = [0] * 3
        self.proQ2 = 1
        # 積分パラメータ A,B,C,D,E (For Debug?)
        self.A, self.B, self.C, self.D, self.E, self.G = [None] * 6
        # Zeromode Energy, 励起エネルギー ω, 比熱・圧力用の全エネルギー
        self.E0, self.omegah, self.U = [[0]] * 3
        # Zeromode Eigenfunction
        self.Psi0 = None
        # Bogoliubov-de Gennesの固有関数
        self.l = 5
        self.Unlh = [np.array([None for _ in range(self.l + 1)])]
        self.Unl, self.Vnl = np.array([]), np.array([])
        self.omega = [None]*(self.l + 1)
        self.omegah = [None]*(self.l + 1)
        # Bogoliubov-de Gennes行列
        self.T = None
        
 
 
class PlotWrapper(metaclass=ABCMeta):
 
    # xcoor : iterable
    # ycoor : iterable or funcn
    # xrange, yrange : 2 element array-like object
    # xlabel, ylabel, title, linecolor : string
    # linewidth : integer
 
    legendloc = None
 
    @classmethod
    def plot_setter(cls,
                    xrange=None,
                    yrange=None,
                    xlabel=None,
                    ylabel=None,
                    title=None,
                    legendloc=None):
 
        if xrange is not None:
            # set xrange
            plt.xlim(xrange[0], xrange[1])
 
        if yrange is not None:
            # set yrange
            plt.ylim(yrange[0], yrange[1])
 
        if xlabel is not None:
            # set xlabel
            plt.xlabel(xlabel)
 
        if ylabel is not None:
            # set ylabel
            plt.ylabel(ylabel)
 
        if title is not None:
            # set title
            plt.title(title)
 
        if legendloc is not None:
            cls.legendloc = legendloc
 
    @classmethod
    def plot_getter(cls,
                    xcoor,
                    ycoor,
                    # linecolor="",
                    linewidth=2,
                    plotlabel=None,
                    showplot=True):
        # plot
        if callable(ycoor):
            plt.plot(
                xcoor,
                ycoor(xcoor),
                label=plotlabel,
                # color=linecolor,
                linewidth=linewidth)
        else:
            plt.plot(
                xcoor,
                ycoor,
                label=plotlabel,
                # color=linecolor,
                linewidth=linewidth)
 
        # show plot data
        if showplot:
            plt.legend(loc=cls.legendloc)
            #plt.pause(2.5)
            plt.show()
            plt.close("all")
 
    @abstractmethod
    def __plot_procedure(v):
        pass
 
 
class GrossPitaevskii(PlotWrapper):
 
    dt = 0.01
 
    # Set initial function
    @staticmethod
    def __psi0(v):
        return np.exp(-v.R**2)
 
    # Make matrix for Crank-Nicolson
    @classmethod
    def __make_crank_matrix(cls, v):
 
        a = np.diag(-2 + 2 * v.DR**2 * (v.mu - v.R**2 / 2 - v.GN * v.xi**2 / (
            8 * np.pi) + 2 * v.Vt + v.Va - 2 / cls.dt))
        c = np.diag(1 + 1 / np.arange(0, v.NR))
        c = np.vstack((c[1:], np.array([0] * v.NR)))
 
        d = np.diag(1 - 1 / np.arange(1, v.NR + 1))
        d = np.vstack((d[1:], np.array([0] * v.NR))).T
 
        return a + c + d
 
    # Make vector for Crank-Nicolson
    @classmethod
    def __make_crank_vector(cls, v):
 
        a = (-2 + 2 * v.DR**2 *
             (v.mu - v.R**2 / 2 - v.GN * v.xi**2 /
              (8 * np.pi) + 2 * v.Vt + v.Va + 2 / cls.dt)) * v.xi
        c = (1 + 1 / np.arange(1, v.NR + 1)) * np.hstack((v.xi[1:], [0]))
        d = (1 - 1 / np.arange(1, v.NR + 1)) * np.hstack(
            ([0], v.xi[:v.NR - 1]))
 
        return -(a + c + d)
 
    # Time evolution
    @classmethod
    def __time_evolution(cls, v):
 
        # Solve matrix equation
        v.xi = solve(cls.__make_crank_matrix(v), cls.__make_crank_vector(v))
 
        # Correction of chemical potential mu
        norm = simps(v.R**2 * v.xi**2, v.R)
        v.mu += (1 - norm) / (2 * cls.dt)
 
        # Normalize arr_Psi
        v.xi /= np.sqrt(norm)
 
    # Time evolution loop
    @classmethod
    def __solve_imaginarytime_gp(cls, v):
 
        print("\n--*-- GP --*--")
        oldmu = 0
        v.xi, v.mu = cls.__psi0(v), 10
 
        while (np.abs(v.mu - oldmu) > 1e-5):
            oldmu = v.mu
            cls.__time_evolution(v)
            sys.stdout.write("\rmu = {0}".format(v.mu))
            sys.stdout.flush()
 
        print("")
 
    # Plot xi
    @classmethod
    def __set_plot(cls, v):
 
        cls.plot_setter(
            yrange=(0, max(v.xi**2) * 1.1),
            xlabel="r-space",
            ylabel="Order parameter",
            title="Static solution of Gross-Pitaevskii equation",
            legendloc="center right")
 
        cls.plot_getter(
            v.R, v.xi**2, plotlabel="gN = {0}".format(v.GN), showplot=False)
        cls.plot_getter(v.R, v.R**2, plotlabel="Potential")
 
    # Hundle
    @classmethod
    def procedure(cls, v, showplot=True):
 
        cls.__solve_imaginarytime_gp(v)
        if (showplot):
            cls.__set_plot(v)
 
 
class AdjointMode(PlotWrapper):
 
    # Make equation matrix
    @staticmethod
    def __make_matrix(v):
 
        dr = np.diag(1 / v.DR**2 + v.R**2 / 2 - v.mu + v.GN / (4 * np.pi) * (
            3 * v.xi + 2 * v.Vt - v.Va))
 
        eu = np.diag(-1 / (2 * v.DR**2) * (1 + 1 / np.arange(0, v.NR)))
        eu = np.vstack((eu[1:], np.array([0] * v.NR)))
 
        el = np.diag(-1 / (2 * v.DR**2) * (1 - 1 / np.arange(1, v.NR + 1)))
        el = np.vstack((el[1:], np.array([0] * v.NR))).T
 
        return dr + eu + el
 
    # Obtain eta and I
    @classmethod
    def __solve_adjoint_equation(cls, v):
 
        print("--*-- Adjoint --*--")
        etatilde = solve(cls.__make_matrix(v), v.xi)
 
        v.I = 1 / (2 * v.N0) * (simps(v.R**2 * etatilde * v.xi))** -1
        v.eta = v.I * etatilde
        print("I = {0}".format(v.I))
 
    # Plot eta
    @classmethod
    def __set_plot(cls, v):
 
        cls.plot_setter(
            xlabel="r-space",
            ylabel="Adjoint parameter",
            title="Solution of AjointMode equation",
            legendloc="center right")
 
        cls.plot_getter(v.R, v.eta, plotlabel="gN = {0}".format(v.GN))
 
    # Hundle
    @classmethod
    def procedure(cls, v, showplot=True):
 
        cls.__solve_adjoint_equation(v)
        if (showplot):
            cls.__set_plot(v)
 
 
# unused as yet
class Bogoliubov(PlotWrapper):
 
    #realpositiveomega, realnegativeomega = [None] * 2
 
    # Make matrix for Bogoliubov-de Gennes equation
    @classmethod
    def __make_bogoliubov_matrix(cls, v, l):

        e1 = -0.5/v.DR**2
 
        Ld = np.diag(-2*e1  + l * (l + 1) / v.R**2 / 2 +
                     v.R**2 / 2 - v.mu + 2 * v.G_TILDE * (v.xi**2 + v.Vt))
 
        Lu = np.diag([e1]*v.NR)
        Lu = np.vstack((Lu[1:], np.array([0] * v.NR)))
 
        Ll = np.diag([e1]*v.NR)
        Ll = np.vstack((Ll[1:], np.array([0] * v.NR))).T
 
        L = Ld + Lu + Ll
 
        M = np.diag(v.G_TILDE * (v.xi**2 - v.Va))
 
        v.T = np.hstack((np.vstack((L, -M)), np.vstack((M, -L))))
 
    @classmethod
    def __update_bogoliubov_matrix(cls, v, l):
 
        Ld =  l / v.R**2
        v.T += np.diag(np.r_[Ld, -Ld])
 
    @classmethod
    def __solve_bogoliubov_equation(cls, v):

        # 初期化
        v.Vt, v.Va = 0, 0

        print("--*-- BdG --*--")
        cls.__make_bogoliubov_matrix(v, l=0)
        pbar = tqdm(range(v.l + 1))
        for l in pbar:
            wr, vr = eig(v.T)
            U, V = vr.T[:, :v.NR], vr.T[:, v.NR:]
            # 固有値の順にソート, 正の固有値のみ取り出す
            U, V = U[wr.argsort()][v.NR:], V[wr.argsort()][v.NR:]
            omega = np.array(sorted(np.real(wr))[v.NR:])
            
            # 固有値 omega が0.1以下のモードは捨てる
            for index, iter_omega in enumerate(omega):
                if(iter_omega < 0.1):
                    continue
                break
            
            omega = omega[index:]
            U, V = U[index:], V[index:]
            
            # 規格化係数
            norm2 = simps(np.array(U)**2 - np.array(V)**2, v.R)
            coo = (2*l + 1)/norm2/v.N0
            
            # debug
            """
            plt.plot(v.R, U[0]/np.sqrt(norm2[0]))
            plt.plot(v.R, V[0]/np.sqrt(norm2[0]))
            plt.show()
            """
            # debug
            
            ndist = (np.exp(v.BETA*omega) -1)**-1
            
            tmpVt = ((U**2 + V**2)*ndist.reshape(v.NR-index, 1) + V**2)*coo.reshape(v.NR-index, 1)
            tmpVt = tmpVt.T.sum(axis=1)
            v.Vt += tmpVt/v.R**2
            tmpVa = (2*U*V*ndist.reshape(v.NR-index, 1) + U*V)*coo.reshape(v.NR-index, 1)
            tmpVa = tmpVa.T.sum(axis=1)
            v.Va += tmpVa/v.R**2
            
            cls.__update_bogoliubov_matrix(v, l + 1)
            pbar.set_description("l = {0}".format(l))
 
    @classmethod
    def __set_plot(cls, v):
 
        cls.plot_setter(
            xlabel="r-space",
            ylabel="BdG eigenfunction",
            title="Solution of BdG equation",
            legendloc="center right")
 
        l, n = 2, 1
        for l in [2, 4]:
            cls.plot_getter(
                v.R,
                np.real(v.Unl[l][n]),
                plotlabel="Unl : omega={0}".format(v.omega[l][n]),
                showplot=False)

            cls.plot_getter(
                v.R,
                np.real(v.Vnl[l][n]), plotlabel="Vnl : omega={0}".format(v.omega[l][n]),
                showplot=False)

        plt.legend()
        plt.show()

 
    @classmethod
    def procedure(cls, v, showplot=True):
 
        cls.__solve_bogoliubov_equation(v)
        if (showplot):
            cls.__set_plot(v)
          


class ZeroMode(PlotWrapper):

    H0 = None

    @staticmethod
    def __set_zeromode_coefficient(v):
 
        v.A = v.G_TILDE * v.N0 * simps(v.R**2 * v.xi**4, v.R)
        v.B = v.G_TILDE * v.N0 * simps(v.R**2 * v.xi**3 * v.eta, v.R)
        v.C = v.G_TILDE * v.N0 * simps(v.R**2 * v.xi**2 * v.eta**2, v.R)
        v.D = v.G_TILDE * v.N0 * simps(v.R**2 * v.xi * v.eta**3, v.R)
        v.E = v.G_TILDE * v.N0 * simps(v.R**2 * v.eta**4, v.R)
        v.G = v.N0 * simps(v.R**2 * v.eta**2, v.R)
 
    # Make Zeromode hamiltonian(Lower triangle)
    @classmethod
    def __make_zeromode_hamiltonian(cls, v):
 
        dmu = 0
        cls.__set_zeromode_coefficient(v)
 
        alpha = 3 * v.E / v.DQ**4 + (v.I - 4 * v.D) / v.DQ**2 + 2 * v.C * (v.Q - v.NQ / 2.0)**2 - \
            2 * v.DQ**2 * v.B * (v.Q - v.NQ / 2.0)**2 + \
            0.5 * v.DQ**4 * v.A * (v.Q - v.NQ / 2.0)**4
        beta = -2 * v.E / v.DQ**4 + 2.0j * v.D / v.DQ**3 - 0.5 * (
            v.I - 4 * v.D) / v.DQ**2 - 0.5j * (dmu + 4 * v.C) / v.DQ - (
                v.C - 1j * v.DQ * v.B) * (v.Q - v.NQ / 2) * (v.Q - v.NQ / 2 + 1
                                                             )
        gamma = [0.5 * v.E / v.DQ**4 - 1j * v.D / v.DQ**3] * v.NQ
 
        cls.H0 = np.vstack((alpha, beta, gamma))
 
    # Solve eigenvalue problem for Zeromode hamiltonian
    @classmethod
    def __solve_zeromode_equation(cls, v, dmu, selecteig='a'):
 
        #print("E0_1 : {0}, select={1}, dmu={2}".format(v.E0, selecteig, dmu))
 
        # Only ground state
        v.E0, v.Psi0 = eig_banded(
            cls.H0 + np.vstack(
                ([0] * v.NQ, [-0.5j * dmu / v.DQ] * v.NQ, [0] * v.NQ)),
            lower=True,
            select=selecteig,
            select_range=(0, v.E0[0] + 4/v.BETA*np.log(10)),
            overwrite_a_band=True)
        #print("E0_2 : {0}".format(v.E0))
        v.Psi0 = v.Psi0.T
 
    @classmethod
    def __output_expected_value_p(cls, v, dmu):
 
        cls.__solve_zeromode_equation(v, dmu)
        #print("E0_3 : {0}".format(v.E0))
 
        dedominetor = np.exp(-v.BETA * (v.E0 - v.E0[0]))
        psi = np.imag(np.conj(v.Psi0[:, :v.NQ - 1]) * v.Psi0[:, 1:])
        psi = np.dot(psi.sum(axis=1), dedominetor)
        return psi / dedominetor.sum() / v.DQ
 
    @classmethod
 
    # Find proper counter term dmu
    def __zeromode_self_consistency(cls, v):
 
        cls.__make_zeromode_hamiltonian(v)
        #cls.__solve_zeromode_equation(v, dmu=0)
 
        print("--*-- ZeroMode --*--")
        dmu = optimize.bisect(
            lambda iterdmu: cls.__output_expected_value_p(v, iterdmu), -0.01, 0.01)
        print("dmu = {0}".format(dmu))
 
    @classmethod
    def __set_plot(cls, v):
 
        index = v.E0.size - 1
        print("index : {0}".format(index))
        cls.plot_setter(
            xlabel="ZeroMode-q-space",
            ylabel="ZeroMode eigenfunction",
            title="Solution of ZeroMode equation")
 
        cls.plot_getter(
            v.Q,
            np.abs(v.Psi0[0])**2,
            plotlabel="ZeroMode function : n = {0}".format(0))
 
    # For debug
    @classmethod
    def __output_dmu_p(cls, v):
        cls.__make_zeromode_hamiltonian(v)
        p = []
        iterdmu = np.linspace(-1, 1, 100)
        for dmu in iterdmu:
            p.append(cls.__output_expected_value_p(v, dmu))
 
        cls.plot_getter(iterdmu, p, plotlabel="expected value <P>")

    @classmethod
    def __set_zeromode_expected_value(cls, v):
    
        dedominator = np.exp(-v.BETA * v.E0)
 
        v.Q2 = (v.Q - v.NQ / 2)**2 * v.Psi0 * np.conj(v.Psi0)
        v.Q2 = np.dot(v.Q2.sum(axis=1), dedominator) / dedominator.sum() * v.DQ**2
        
        v.P2 = v.Psi0 * np.conj(v.Psi0) - np.real(np.conj(v.Psi0) * np.insert(v.Psi0, v.NQ, 0, axis=1)[:, 1:])
        v.P2 = 2 * np.dot(v.P2.sum(axis=1), dedominator) / dedominator.sum() / v.DQ**2

        v.Vt += np.real(v.xi**2*v.Q2 + v.eta**2*v.P2 - v.xi*v.eta)
        
        v.Va += np.real(-v.xi**2*v.Q2 + v.eta**2*v.P2)

        print("Q2 = {0:1.3e}, P2 = {1:1.3e}".format(v.Q2.real, v.P2.real))
 
 
    @classmethod
    def procedure(cls, v, showplot=True):
 
        cls.__zeromode_self_consistency(v)
        cls.__set_zeromode_expected_value(v)
        if (showplot):
            cls.__set_plot(v)
  
 
if (__name__ == "__main__"):

    var = Variable()
    i = 0
    while(abs(var.promu - var.mu) > 1e-5 and abs(var.I - var.proI) > 1e-10 and abs(var.Q2 - var.proQ2) > 1e-7):
        var.promu, var.proI, var.proQ2 = var.mu, var.I, var.Q2
        GrossPitaevskii.procedure(v=var, showplot=False)
        AdjointMode.procedure(v=var, showplot=False)
        Bogoliubov.procedure(v=var, showplot=False)
        ZeroMode.procedure(v=var, showplot=False)
        print("--*-- live score : {0} times --*--".format(i))
        print("Vt : {0}..., Va : {1}...".format(var.Vt[:5], var.Va[:5]))
        print("I : {0}".format(var.I))
        print("dmu : {0}, dI : {1}, dQ2 : {2}".format(abs(var.promu - var.mu), abs(var.I - var.proI), abs(var.Q2 - var.proQ2)))
        print("--*-- live score : {0} times --*--".format(i))
        i += 1
        
