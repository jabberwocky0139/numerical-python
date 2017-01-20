# coding: utf-8

import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
from scipy.linalg import solve, eig, eig_banded
from scipy.integrate import simps
from scipy import optimize, special
from abc import abstractmethod, ABCMeta
# Clean up some warnings we know about so as not to scare the users
import warnings
from matplotlib.cbook import MatplotlibDeprecationWarning
warnings.simplefilter('ignore', MatplotlibDeprecationWarning)


omega_thermo = []
omega_thermo_T = []
specific_thermo = []

class Variable(object):
    def __init__(self,
                 N0=1e3,
                 G=4 * np.pi * 1e-2,
                 TTc=0.1,
                 xi=0,
                 mu=10,
                 dmu=0.1,
                 index=0):

        # --*-- Constants --*--

        # 凝縮粒子数 N, 逆温度 β, 相互作用定数 g
        self.N0, self.G = N0, G
        self.TTc = TTc
        self.Temp = (self.N0 / special.zeta(3, 1))**(1 / 3) * TTc

        if (self.Temp == 0):
            self.BETA = np.inf
        else:
            self.BETA = 1 / self.Temp

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
        self.L, self.NQ = 20 * (self.N0)**(-1 / 3), 600
        self.NM = self.NQ * 1 / 3
        # q表示においてNq分割 dq
        self.DQ = self.L / self.NQ
        self.Q = np.arange(self.NQ)

        # --*-- Variables --*--

        # 化学ポテンシャル μ, 秩序変数 ξ, 共役モード η, 共役モードの規格化変数 I
        self.mu, self.xi, self.eta, self.I = mu, xi, 0, 0
        self.promu, self.proI = [1] * 2
        self.Nc, self.Ntot = 0, 0
        # 熱平均, 異常平均
        self.Vt = np.array([0 for _ in range(self.NR)])
        self.Va = np.array([0 for _ in range(self.NR)])

        # Pの平均 <P>, Q^2の平均 <Q^2>, P^2の平均 <P^2>
        self.P, self.Q2, self.P2 = [0] * 3
        self.proQ2, self.proP2 = 1, 1
        # 積分パラメータ A,B,C,D,E (For Debug?)
        self.A, self.B, self.C, self.D, self.E, self.G = [None] * 6
        # Zeromode Energy, 励起エネルギー ω, 比熱・圧力用の全エネルギー
        self.E0, self.omegah, self.U = [[0]] * 3
        # Zeromode Eigenfunction
        self.Psi0 = None
        # Bogoliubov-de Gennesの固有関数
        self.l = 14
        # 系の内部エネルギー, 比熱
        self.Energy, self.Specific = 0, 0
        # Bogoliubov-de Gennes行列
        # self.T = None
        # self.S = None
        self.dmu = dmu
        self.selecteig = "i"
        self.bdg_u, self.bdg_v, self.ndist = [0] * 15, [0] * 15, [0] * 15
        self.index = [0] * 15
        self.h_int = 0
        # self.index = index


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
    def plot_getter(
            cls,
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

    dt = 0.005

    # Set initial function
    @staticmethod
    def __psi0(v):
        return np.exp(-v.R**2)

# Make matrix for Crank-Nicolson

    @classmethod
    def __make_crank_matrix(cls, v):

        a = np.diag(-2 + 2 * v.DR**2 * (v.mu - 0.5 * v.R**2 - v.G_TILDE * (
            0.5 * v.xi**2 + 2 * v.Vt + v.Va) - 2 / cls.dt))
        c = np.diag(1 + 1 / np.arange(0, v.NR))
        c = np.vstack((c[1:], np.array([0] * v.NR)))

        d = np.diag(1 - 1 / np.arange(1, v.NR + 1))
        d = np.vstack((d[1:], np.array([0] * v.NR))).T

        return a + c + d

# Make vector for Crank-Nicolson

    @classmethod
    def __make_crank_vector(cls, v):

        a = (-2 + 2 * v.DR**2 *
             (v.mu - 0.5 * v.R**2 - v.G_TILDE *
              (0.5 * v.xi**2 + 2 * v.Vt + v.Va) + 2 / cls.dt)) * v.xi
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
    def __solve_imaginarytime_gp(cls, v, initial):

        print("\n--*-- GP --*--")
        oldmu = 0
        #v.xi, v.mu = cls.__psi0(v), 10
        if (initial):
            v.xi = np.exp(-v.R**2)
        i = 0
        while (np.abs(v.mu - oldmu) > 1e-8):
            #i += 1
            #if(i > 100):
            #    cls.dt *= 0.5
            oldmu = v.mu
            cls.__time_evolution(v)
            print("mu = {0}".format(v.mu), "\r", end='', flush=True)

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
    def procedure(cls, v, showplot=True, initial=False):

        cls.__solve_imaginarytime_gp(v, initial)
        if (showplot):
            cls.__set_plot(v)


class AdjointMode(PlotWrapper):

    # Make equation matrix
    @staticmethod
    def __make_matrix(v):

        dr = np.diag(1 / v.DR**2 + 0.5 * v.R**2 - v.mu + v.G_TILDE * (
            3 * v.xi**2 + 2 * v.Vt - v.Va))

        eu = np.diag(-0.5 / v.DR**2 * (1 + 1 / np.arange(1, v.NR + 1)))
        eu = np.vstack((eu[1:], np.array([0] * v.NR)))

        el = np.diag(-0.5 / v.DR**2 * (1 - 1 / np.arange(2, v.NR + 2)))
        el = np.vstack((el[1:], np.array([0] * v.NR))).T

        return dr + eu + el

    @classmethod
    def __make_array(cls, v):

        d = 1 / v.DR**2 + 0.5 * v.R**2 - v.mu + v.G_TILDE * (3 * v.xi**2 + 2 *
                                                             v.Vt - v.Va)
        du = -0.5 / v.DR**2 * (1 + 1 / np.arange(1, v.NR + 1))
        dl = -0.5 / v.DR**2 * (1 - 1 / np.arange(2, v.NR + 2))
        return np.vstack((du, d, dl))

# Obtain eta and I

    @classmethod
    def __solve_adjoint_equation(cls, v):

        print("--*-- Adjoint --*--")
        v.eta = solve(cls.__make_matrix(v), v.xi)
        v.I = 1 / (2 * v.N0) / simps(v.R**2 * v.eta * v.xi, v.R)
        v.eta *= v.I
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

# Handle

    @classmethod
    def procedure(cls, v, showplot=True):

        cls.__solve_adjoint_equation(v)
        if (showplot):
            cls.__set_plot(v)


class BogoliubovSMatrix(PlotWrapper):

    L, M = [None] * 2

    # Make matrix for Bogoliubov-de Gennes equation
    @classmethod
    def __make_bogoliubov_matrix(cls, v, l):

        e1 = -0.5 / v.DR**2

        Ld = np.diag(-2 * e1 + l * (l + 1) / v.R**2 / 2 + v.R**2 / 2 - v.mu + 2
                     * v.G_TILDE * (v.xi**2 + v.Vt))

        Lu = np.diag([e1] * v.NR)
        Lu = np.vstack((Lu[1:], np.array([0] * v.NR)))

        Ll = np.diag([e1] * v.NR)
        Ll = np.vstack((Ll[1:], np.array([0] * v.NR))).T

        cls.L = Ld + Lu + Ll

        cls.M = np.diag(v.G_TILDE * (v.xi**2 - v.Va))

        v.S = np.dot(cls.L - cls.M, cls.L + cls.M)

    @classmethod
    def __solve_bogoliubov_equation(cls, v):

        # 初期化
        v.Vt, v.Va, v.Energy, v.Specific = [0] * 4

        print("--*-- BdG (Smat) --*--")

        for l in range(v.l + 1):
            cls.__make_bogoliubov_matrix(v, l)

            wr, vl, vr = eig(v.S, left=True)

            Y, Z = vr.T[wr.argsort()], vl.T[wr.argsort()]
            omega2 = np.array(sorted(np.real(wr)))

            # 固有値 omega が0.1以下のモードは捨てる
            for index1, iter_omega in enumerate(omega2):
                if (np.sqrt(iter_omega) < 0.1):
                    continue
                break

            # 固有値 omega が200以上のモードは捨てる
            for index2, iter_omega in enumerate(omega2):
                if (np.sqrt(iter_omega) < 200):
                    continue
                break

            omega2 = omega2[index1:index2]
            omega = np.sqrt(omega2)
            Y, Z = Y[index1:index2], Z[index1:index2]

            c1 = []
            # c1の計算
            for z, y in zip(Z, Y):
                c = np.dot(cls.L - cls.M, z)
                c1.append(np.abs(np.dot(c, np.conj(y))))
            c1 = np.array(c1)

            v.index[l] = omega.shape[0]
            v.bdg_u[l], v.bdg_v[l] = Y * c1.reshape(
                v.index[l], 1) + Z * omega.reshape(
                    v.index[l], 1), Y * c1.reshape(
                        v.index[l], 1) - Z * omega.reshape(v.index[l], 1)

            # 規格化係数
            norm2 = simps(v.bdg_u[l]**2 - v.bdg_v[l]**2, v.R)
            coo = (2 * l + 1) / norm2 / v.N0

            v.ndist[l] = (np.exp(v.BETA * omega) - 1)**-1
            if (v.Temp < 1e-7):
                v.ndist[l] = np.array([0] * v.index[l])

            tmpVt = ((v.bdg_u[l]**2 + v.bdg_v[l]**2) * v.ndist[l].reshape(
                v.index[l], 1) + v.bdg_v[l]**2) * coo.reshape(v.index[l], 1)
            tmpVt = tmpVt.T.sum(axis=1)
            v.Vt += tmpVt / v.R**2

            tmpVa = (2 * v.bdg_u[l] * v.bdg_v[l] * v.ndist[l].reshape(
                v.index[l], 1) + v.bdg_u[l] * v.bdg_v[l]) * coo.reshape(
                    v.index[l], 1)
            tmpVa = tmpVa.T.sum(axis=1)
            v.Va += tmpVa / v.R**2

            v.Energy += (2 * l + 1) * np.dot(omega, v.ndist[l]) / v.N0
            v.Specific += (2 * l + 1)**2 * np.dot(omega**2, (
                np.exp(v.BETA * omega) + np.exp(-v.BETA * omega) - 2)**
                                                  -1) / v.Temp**2 / v.N0

            print("l = {0}".format(l), "\r", end='', flush=True)
        print(
            ", BdG_Va : {0:1.6f}, omega_low : {1:1.4f}, omega_high : {2:1.4f}, omega_len : {3}".
            format(v.Va[0], omega[0], omega[-1], omega.shape[0]))
        print("Energy : {0}, Cv : {1}".format(v.Energy, v.Specific))

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
                plotlabel="Unl : omega={0}".format(omega[l][n]),
                showplot=False)

            cls.plot_getter(
                v.R,
                np.real(v.Vnl[l][n]),
                plotlabel="Vnl : omega={0}".format(omega[l][n]),
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
    def __make_zeromode_band(cls, v, dmu):

        #dmu = 0
        cls.__set_zeromode_coefficient(v)

        alpha = 3 * v.E / v.DQ**4 + (v.I - 4 * v.D) / v.DQ**2 + 2 * v.C * (
            v.Q - v.NQ / 2.0)**2 - 2 * v.DQ**2 * v.B * (
                v.Q - v.NQ / 2.0)**2 + 0.5 * v.DQ**4 * v.A * (v.Q - v.NQ / 2.0
                                                              )**4
        beta = -2 * v.E / v.DQ**4 + 2.0j * v.D / v.DQ**3 - 0.5 * (
            v.I - 4 * v.D) / v.DQ**2 - 0.5j * (dmu + 4 * v.C) / v.DQ - (
                v.C - 1j * v.DQ * v.B) * (v.Q - v.NQ / 2) * (v.Q - v.NQ / 2 + 1
                                                             )
        gamma = [0.5 * v.E / v.DQ**4 - 1j * v.D / v.DQ**3] * v.NQ

        cls.H0 = np.vstack((alpha, beta, gamma))

    @classmethod
    def __make_zeromode_matrix(cls, v, dmu):

        #dmu = 0
        cls.__set_zeromode_coefficient(v)

        alpha = np.diag(3 * v.E / v.DQ**4 + (v.I - 4 * v.D) / v.DQ**2 + 2 * v.C
                        * (v.Q - v.NQ / 2.0)**2 - 2 * v.DQ**2 * v.B * (
                            v.Q - v.NQ / 2.0)**2 + 0.5 * v.DQ**4 * v.A * (
                                v.Q - v.NQ / 2.0)**4)
        beta = np.diag(-2 * v.E / v.DQ**4 + 2.0j * v.D / v.DQ**3 - 0.5 * (
            v.I - 4 * v.D) / v.DQ**2 - 0.5j * (dmu + 4 * v.C) / v.DQ - (
                v.C - 1j * v.DQ * v.B) * (v.Q - v.NQ / 2) * (v.Q - v.NQ / 2 + 1
                                                             ))
        beta = np.vstack((beta[1:], np.array([0] * v.NQ))).T
        gamma = np.diag([0.5 * v.E / v.DQ**4 - 1j * v.D / v.DQ**3] * v.NQ)
        gamma = np.vstack(
            (gamma[2:], np.array([0] * v.NQ), np.array([0] * v.NQ))).T

        cls.H0 = alpha + beta + gamma

# Solve eigenvalue problem for Zeromode hamiltonian

    @classmethod
    def __solve_zeromode_equation_v(cls, v):

        v.E0, v.Psi0 = eig_banded(
            #cls.H0 + np.vstack(
            #([0] * v.NQ, [-0.5j * dmu / v.DQ] * v.NQ, [0] * v.NQ)),
            cls.H0,
            lower=True,
            select=v.selecteig,
            select_range=(0, v.E0[0] + 9 / v.BETA * np.log(10)),
            overwrite_a_band=True)

        v.Psi0 = v.Psi0.T

    @classmethod
    def __solve_zeromode_equation_i(cls, v):
        v.E0, v.Psi0 = eig_banded(
            #cls.H0 + np.vstack(
            #([0] * v.NQ, [-0.5j * dmu / v.DQ] * v.NQ, [0] * v.NQ)),
            cls.H0,
            lower=True,
            select="i",
            select_range=(0, v.NM),
            overwrite_a_band=True)

        v.Psi0 = v.Psi0.T

    @classmethod
    def __output_expected_value_p(cls, v, dmu):

        cls.__make_zeromode_band(v, dmu)
        if (v.selecteig == "v"):
            cls.__solve_zeromode_equation_v(v)
        elif (v.selecteig == "i"):
            cls.__solve_zeromode_equation_i(v)

        dedominator = np.exp(-v.BETA * v.E0)
        # dedominator2 = np.exp(-v.BETA * (v.E0 - v.E0[0]))

        psi = np.imag(np.conj(v.Psi0[:, :v.NQ - 1]) * v.Psi0[:, 1:])

        if (v.Temp < 1e-6):
            P = psi[0].sum()
        else:
            psi = np.dot(psi.sum(axis=1), dedominator)
            P = psi / dedominator.sum() / v.DQ

        print("P = {0}".format(P), "\r", end='', flush=True)

        return P

    @classmethod
    # Find proper counter term dmu
    def __zeromode_self_consistency(cls, v):

        print("--*-- ZeroMode --*--")
        if (v.dmu > 0):
            v.dmu = optimize.brentq(
                lambda iterdmu: cls.__output_expected_value_p(v, iterdmu),
                -9 * abs(v.dmu),
                10 * abs(v.dmu),
                xtol=1e-15)
        else:
            v.dmu = optimize.brentq(
                lambda iterdmu: cls.__output_expected_value_p(v, iterdmu),
                -10 * abs(v.dmu),
                9 * abs(v.dmu),
                xtol=1e-15)

        print(", dmu = {0}, E0_len = {1}".format(v.dmu, v.E0.size))
        print("E0 = {0}..".format(v.E0[:3]))

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
            plotlabel="ZeroMode function : n = {0}".format(0),
            showplot=False)
        cls.plot_getter(
            v.Q,
            np.abs(v.Psi0[1])**2,
            plotlabel="ZeroMode function : n = {0}".format(1),
            showplot=False)
        cls.plot_getter(
            v.Q,
            np.abs(v.Psi0[v.NM - 1])**2,
            plotlabel="ZeroMode function : n = {0}".format(v.NM - 1))

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
        # dedominator2 = np.exp(-v.BETA * (v.E0 - v.E0[0]))

        v.Q2 = (v.Q - v.NQ / 2)**2 * v.Psi0 * np.conj(v.Psi0)
        if (v.Temp < 1e-6):
            v.Q2 = v.Q2[0].sum() * v.DQ**2
        else:
            v.Q2 = np.dot(v.Q2.sum(axis=1),
                          dedominator) / dedominator.sum() * v.DQ**2

        v.P2 = v.Psi0 * np.conj(v.Psi0) - np.real(
            np.conj(v.Psi0) * np.insert(
                v.Psi0, v.NQ, 0, axis=1)[:, 1:])
        if (v.Temp < 1e-6):
            v.P2 = 2 * v.P2[0].sum() / v.DQ**2
        else:
            v.P2 = 2 * np.dot(v.P2.sum(axis=1),
                              dedominator) / dedominator.sum() / v.DQ**2

        v.Vt += np.real(v.xi**2 * v.Q2.real + v.eta**2 * v.P2.real - v.xi *
                        v.eta)
        v.Va += np.real(-v.xi**2 * v.Q2.real + v.eta**2 * v.P2.real)

        psi2E0 = np.real(v.Psi0 * np.conj(v.Psi0) * v.E0.reshape(v.E0.size, 1))
        psi2E02 = np.real(v.Psi0 * np.conj(v.Psi0) * v.E0.reshape(v.E0.size, 1)
                          **2)

        #v.Energy += np.dot(v.E0, dedominator) / dedominator.sum()
        #v.Specific += (np.dot(v.E0**2, dedominator)/ dedominator.sum() - (np.dot(v.E0, dedominator) / dedominator.sum())**2)/v.Temp**2
        v.Energy += np.dot(psi2E0.sum(axis=1), dedominator) / dedominator.sum()
        v.Specific += (
            np.dot(psi2E02.sum(axis=1), dedominator) / dedominator.sum() -
            (np.dot(psi2E0.sum(axis=1), dedominator) / dedominator.sum())**2
        ) / v.Temp**2 / 100
        v.L = np.sqrt(v.Q2.real) * 25
        v.DQ = v.L / v.NQ

        print("Q2 = {0:1.3e}, P2 = {1:1.3e}".format(v.Q2.real, v.P2.real))

    @classmethod
    def procedure(cls, v, showplot=True):

        prodmu = v.dmu
        cls.__zeromode_self_consistency(v)
        while (v.dmu < 1e-20):
            prodmu = 0.1 * prodmu
            v.dmu = prodmu
            cls.__zeromode_self_consistency(v)

        cls.__set_zeromode_expected_value(v)

        if (showplot):
            cls.__set_plot(v)


class ParticleNumbers(object):
    @classmethod
    def __set_total_particle_number(cls, v, T):

        v.Nc = np.real(v.N0 * (1 + v.Q2) + v.G**2 * v.P2 - 0.5)
        v.Ntot = np.real(v.Nc + v.N0 * simps(v.R**2 * v.Vt, v.R))

        if (v.Temp < 1e-7):
            v.Temp = 0
            v.BETA = np.inf
        else:
            v.Temp = (v.Ntot / special.zeta(3, 1))**(1 / 3) * T
            v.BETA = 1 / v.Temp

    @classmethod
    def procedure(cls, v, T):

        cls.__set_total_particle_number(v, T)


class PerturbedSpecificheat(object):
    
    tmt1, tmt2, tmt3, tmt4, tmt5 = [0] * 5
    tmti1, tmti2, tmti3, tmti4 = [0] * 4
    print("--*-- PerturbedSpecificheat --*--")
    
    @classmethod
    def zero_bdg_coupling(cls, v):
        """Zeromode-BdGカップリングの計算"""
        # 念のため
        v.h_int = 0
        for l in range(v.l + 1):
            # ModifiedThermodynamicのZeromode交差項
            tva = (2 * v.bdg_u[l] * v.bdg_v[l] * v.ndist[l].reshape(
                v.index[l], 1) + v.bdg_u[l] * v.bdg_v[l])
            tva = tva.T.sum(axis=1)
            tvt = ((v.bdg_u[l]**2 + v.bdg_v[l]**2) * v.ndist[l].reshape(
                v.index[l], 1) + v.bdg_v[l]**2)
            tvt = tvt.T.sum(axis=1)
            cls.tmt1 += 4 * np.pi * v.G_TILDE * (2 * l + 1) * simps(
                v.R**2 * v.xi**2 * tva, v.R)
            cls.tmt2 += 4 * np.pi * v.G_TILDE * (2 * l + 1) * simps(
                v.R**2 * v.eta**2 * tva, v.R)
            cls.tmt3 += 8 * np.pi * v.G_TILDE * (2 * l + 1) * simps(
                v.R**2 * v.xi**2 * tvt, v.R)
            cls.tmt4 += 8 * np.pi * v.G_TILDE * (2 * l + 1) * simps(
                v.R**2 * v.eta**2 * tvt, v.R)
            cls.tmt5 += 8 * np.pi * v.G_TILDE * (2 * l + 1) * simps(
                v.R**2 * v.xi * v.eta * tvt, v.R)

        v.h_int += v.Q2.real * (-cls.tmt1 + cls.tmt3)
        v.h_int += v.P2.real * (cls.tmt2 + cls.tmt4) - cls.tmt5
        # 念のため
        cls.tmt1, cls.tmt2, cls.tmt3, cls.tmt4, cls.tmt5 = [0] * 5

    @classmethod
    def bdg_bdg_coupling(cls, v):
        """BdG-BdGカップリングの計算"""
        for l in range(v.l + 1):
            tmtii1 = (2 * l + 1) * v.bdg_u[l]**2 * v.ndist[l].reshape(v.index[l], 1)
            tmtii2 = (2 * l + 1) * (v.bdg_v[l]**2 * v.ndist[l].reshape(v.index[l], 1) + v.bdg_v[l]**2)
            tmtii3 = (2 * l + 1) * (2 * v.bdg_u[l] * v.bdg_v[l] * v.ndist[l].reshape(v.index[l], 1) + v.bdg_u[l] * v.bdg_v[l])
            cls.tmti1 += tmtii1.T.sum(axis=1)
            cls.tmti2 += tmtii2.T.sum(axis=1)
            cls.tmti3 += tmtii3.T.sum(axis=1)

        # この子を先に！
        cls.tmti4 = cls.tmti1 * cls.tmti2
        # 以下を後に！
        cls.tmti1 = (cls.tmti1)**2
        cls.tmti2 = (cls.tmti2)**2
        cls.tmti3 = (cls.tmti3)**2

        v.h_int +=  v.G_TILDE / (2 * v.N0) * simps(v.R**2 * (2 * cls.tmti1.real + 2 * cls.tmti2.real + cls.tmti3.real + 4 * cls.tmti4.real), v.R)
        # 念のため
        cls.tmti1, cls.tmti2, cls.tmti3 = 0, 0, 0
        print('v.h_int = {0}'.format(v.h_int))
        assert v.h_int < 1

    @classmethod
    def calc_specific(cls, v):
        cls.zero_bdg_coupling(v)
        cls.bdg_bdg_coupling(v)
        # 熱力学ポテンシャルの摂動項
        v.h_int = -v.BETA**-1 * np.log(1 - v.BETA * v.h_int)
        print('<H_int> = {0}'.format(v.h_int))

    

class IZMFSolver(object):

    TTc, a_s = [None] * 2

    @classmethod
    def __print_data(cls, var, i):
        print("|--*-- other parameters : {0} times --*--|".format(i))
        print("Vt : {0}..., Va : {1}...".format(var.Vt[:3], var.Va[:3]))
        print("I : {0}".format(var.I))
        print("N0 : {0}, Nc : {1}, Ntot : {2}".format(var.N0, var.Nc,
                                                      var.Ntot))
        print("Energy : {0}, Cv : {1}".format(var.Energy, var.Specific))
        print("dmu : {0}, dI : {1}, dQ2 : {2}, dP2 : {3}\n".format(
            abs(var.promu - var.mu),
            abs(var.I - var.proI),
            abs(var.Q2**0.5 - var.proQ2**0.5),
            abs(var.P2**0.5 - var.proP2**0.5)))

    @classmethod
    def procedure(cls, filename, iterable, TTc, a_s, which):

        with open(filename, "w") as f:

            cls.TTc, cls.a_s = TTc, a_s
            print(
                "# g\t T\t Q2\t P2\t Ntot\t Energy\t Cv\t beta\t Nc",
                file=f,
                flush=True)

            for index, physicalparameter in enumerate(iterable):

                if (which == "T"):
                    cls.TTc = physicalparameter
                elif (which == "g"):
                    cls.a_s = physicalparameter
                else:
                    print("invalid 'which' key !!")
                    raise KeyError

                print(
                    "\n\n |--------------------------------------------------------|"
                )
                print(
                    " |---*--- a_s = {0:1.3e}, TTc = {1:1.3e}, n = {2:2d} ---*---|".
                    format(cls.a_s, cls.TTc, index))
                print(
                    " |--------------------------------------------------------|\n"
                )

                if (index == 0):
                    var = Variable(
                        TTc=cls.TTc, G=4 * np.pi * cls.a_s, index=index)
                else:
                    var = Variable(
                        G=4 * np.pi * cls.a_s,
                        TTc=cls.TTc,
                        xi=var.xi,
                        mu=var.mu,
                        dmu=var.dmu,
                        index=index)

                i = 0
                while (abs(var.P2**0.5 - var.proP2**0.5) > 2e-2):

                    var.promu, var.proI, var.proQ2, var.proP2 = var.mu, var.I, var.Q2, var.P2

                    if (index == 0):
                        GrossPitaevskii.procedure(
                            v=var, showplot=False, initial=True)
                    else:
                        GrossPitaevskii.procedure(v=var, showplot=False)
                    AdjointMode.procedure(v=var, showplot=False)
                    BogoliubovSMatrix.procedure(v=var, showplot=False)
                    ZeroMode.procedure(v=var, showplot=False)
                    ParticleNumbers.procedure(v=var, T=cls.TTc)
                    PerturbedSpecificheat.calc_specific(v=var)
                    cls.__print_data(var, i)
                    i += 1

                # 比熱のための格納
                omega_thermo.append(var.h_int)
                omega_thermo_T.append(var.Temp)
                specific_thermo.append(var.Specific)
                    
                print(
                    "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}".format(
                        cls.a_s, cls.TTc, var.Q2.real, var.P2.real, var.Ntot,
                        var.Energy, var.Specific, var.BETA, var.Nc),
                    file=f,
                    flush=True)


if (__name__ == "__main__"):

    pure_T = np.logspace(-3, -1, num=100)
    for fn, g in zip(["1e-6"],
                     [1e-6]):
        #for fn, g in zip(["1e-4"], [1e-4]):
        IZMFSolver.procedure(
            filename="output_g{0}.txt".format(fn),
            iterable=pure_T,
            TTc=1e-3,
            a_s=g,
            which="T")

    # 比熱の摂動計算
    Cv = np.array(specific_thermo) - pure_T * np.gradient(np.gradient(omega_thermo, pure_T), pure_T)
    specific_f = open('specific_g{0}.txt'.format(1e-6), 'w')
    print('#{0}\t{1}\t{2}'.format('T', 'Cv', 'Unperturbed_Cv'), file=specific_f)
    for t, cv, unperturbed_cv in zip(pure_T, Cv, specific_thermo):
        print('{0}\t{1}\t{2}'.format(t, cv, unperturbed_cv), file=specific_f)
        

    

    # for fn, T in zip(["0", "1e-3", "1e-2", "5e-2", "1e-1"],
    #                  [1e-8, 1e-3, 1e-2, 5e-2, 1e-1]):
    #     #for fn, T in zip(["1e-3"], [1e-3]):
    #     IZMFSolver.procedure(
    #         filename="output_T{0}.txt".format(fn),
    #         iterable=np.logspace(
    #             -4, -1, num=20),
    #         TTc=T,
    #         a_s=1e-4,
    #         which="g")
