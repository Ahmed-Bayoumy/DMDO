"""
# ------------------------------------------------------------------------------------#
#  Distributed Multidisciplinary Design Optimization - DMDO                           #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU General Public License as published by the Free Software          #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the GNU General Public License for more details.          #
#                                                                                     #
#  You should have received a copy of the GNU General Public License along            #
#  with this program. If not, see <http://www.gnu.org/licenses/>.                     #
#                                                                                     #
#  You can find information on simple_mads at                                         #
#  https://github.com/Ahmed-Bayoumy/DMDO                                              #
# ------------------------------------------------------------------------------------#
# """

import numpy as np
import copy


class WingDesignAnalyzer:
    """
    A class to analyze and design aircraft wings based on structural and aerodynamic parameters.
    """

    def __init__(self, h=55000, Mach=1.4, tc=0.0641, ARw=2.5, LAMBDAw=70.0, Sref=667, Sht=99.7,
                 ARht=5.5, lambdatr=0.1, Lift=34300, t=[0.97, 0.52, 0.21, 4.00, 3.66, 0.91, 0.97, 0.52, 0.21],
                 ts=[2.17, 1.48, 0.76, 4.40, 4.02,1.01, 2.17, 1.48, 0.76]):
        # Inputs
        self.h = h
        self.Mach = Mach
        self.tc = tc
        self.ARw = ARw
        self.LAMBDAw = LAMBDAw
        self.Sref = Sref
        self.Sht = Sht
        self.ARht = ARht
        self.lambdatr = lambdatr
        self.Lift = Lift

        # Constants
        self.C = [500.0, 16000.0, 4.0, 4360.0, 0.01375, 1.0]
        self.G = 4000000 * 144
        self.E = 10600000 * 144
        self.nu = 0.3
        self.rho_alum = 0.1 * 144
        self.rho_core = 0.1 * 144 / 10
        self.rho_fuel = 6.5 * 7.4805
        self.Fw_at_t = 5
        self.k = 6.09375

        # Local variables
        self.Z = [self.tc, self.h, self.Mach, self.ARw, self.LAMBDAw, self.Sref, self.Sht, self.ARht]
        self.LAMBDA = self.lambdatr
        self.L = self.Lift

        # Panel parameters
        self.NP = 9  # number of panels per halfspan
        self.n = 90
        self.rn = int(self.n / self.NP)

        # Thickness values (converted from inches to feet)
        self.ti = t
        self.tsi = ts
        self.t = np.array([ti / 12.0 for ti in self.ti])
        self.ts = np.array([tsi / 12.0 for tsi in self.tsi])

        # Split thickness arrays
        self.t1 = self.t[:3]
        self.t2 = self.t[3:6]
        self.t3 = self.t[6:9]
        self.ts1 = self.ts[:3]
        self.ts2 = self.ts[3:6]
        self.ts3 = self.ts[6:9]

        # Beta factor
        self.beta = 0.9

        # Initialize results
        self.c = None
        self.c_box = None
        self.Sweep_40 = None
        self.D_mx = None
        self.b = None
        self.a = None
        self.P = None
        self.Mz = None
        self.Mx = None
        self.bend_twist = None
        self.Spanel = None
        self.Phi = None
        self.twist = None
        self.deltaL_divby_q = None
        self.Wtop_alum = None
        self.Wbottom_alum = None
        self.Wside_alum = None
        self.Wtop_core = None
        self.Wbottom_core = None
        self.Wside_core = None
        self.W_wingstruct = None
        self.W_fuel_wing = None
        self.Bh = None
        self.W_ht = None
        self.Wf = None
        self.Ws = None
        self.theta = None
        # self.G = None
        self.G1 = None

    def polyApprox(self, S, S_new, flag, S_bound):
        """Calculate polynomial approximation for structural parameters."""
        S_norm = []
        S_shifted = []
        Ai = []
        Aij = np.zeros((len(S), len(S)))

        for i in range(len(S)):
            S_norm.append(S_new[i] / S[i])
            if S_norm[i] > 1.25:
                S_norm[i] = 1.25
            elif S_norm[i] < 0.75:
                S_norm[i] = 0.75
            S_shifted.append(S_norm[i] - 1)
            a = 0.1
            b = a

            if flag[i] == 5:
                # Calculate polynomial coefficients (S-about origin)
                So = 0
                Sl = So - S_bound[i]
                Su = So + S_bound[i]
                Mtx_shifted = np.array([[1, Sl, Sl**2], [1, So, So**2], [1, Su, Su**2]])

                F_bound = np.array([1 + (.5 * a)**2, 1, 1 + (.5 * b)**2])
                A = np.linalg.solve(Mtx_shifted, F_bound)
                Ao = A[0]
                Ai.append(A[1])
                Aij[i, i] = A[2]
            else:
                if flag[i] == 0:
                    S_shifted[i] = 0  # MATLAB: S_shifted(i) = 0
                elif flag[i] == 3:
                    a *= -1.
                    b = copy.deepcopy(a)
                elif flag[i] == 2:
                    b = 2 * a
                elif flag[i] == 4:
                    a *= -1
                    b = 2 * a

                # Determine bounds on FF depending on slope-shape
                So = 0
                Sl = So - S_bound[i]
                Su = So + S_bound[i]
                Mtx_shifted = np.array([[1, Sl, Sl**2], [1, So, So**2], [1, Su, Su**2]])
                F_bound = np.array([1 - .5 * a, 1, 1 + .5 * b])
                A = np.linalg.solve(Mtx_shifted, F_bound)
                Ao = A[0]
                Ai.append(A[1])
                Aij[i, i] = A[2]

        # Create correlation matrix R
        R = np.array([[0.2736, 0.3970, 0.8152, 0.9230, 0.1108],
                      [0.4252, 0.4415, 0.6357, 0.7435, 0.1138],
                      [0.0329, 0.8856, 0.8390, 0.3657, 0.0019],
                      [0.0878, 0.7248, 0.1978, 0.0200, 0.0169],
                      [0.8955, 0.4568, 0.8075, 0.9239, 0.2525]])

        # Fill in correlation matrix
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                Aij[i, j] = Aij[i, i] * R[i, j]
                Aij[j, i] = Aij[i, j]

        S_shifted = np.array(S_shifted)

        # Calculate FF (structural efficiency factor)
        FF = Ao + np.dot(Ai, S_shifted.T) + 0.5 * np.dot(np.dot(S_shifted, Aij), S_shifted.T)

        return FF

    def Wing_Mod(self, Z, LAMBDA):
        """Calculate wing geometry parameters."""
        c = [0, 0, 0, 0]
        x = [0] * 8
        y = [0] * 8

        b = max(2, np.real(np.sqrt(Z[3] * Z[5])))
        c[0] = 2 * Z[5] / ((1 + LAMBDA) * b)
        c[3] = LAMBDA * c[0]
        x[0] = 0
        y[0] = 0
        x[1] = c[0]
        y[1] = 0
        x[6] = (b / 2) * np.tan(Z[4] * np.pi / 180)
        y[6] = b / 2
        x[7] = x[6] + c[3]
        y[7] = b / 2
        y[2] = b / 6
        x[2] = (x[6] / y[6]) * y[2]
        y[4] = b / 3
        x[4] = (x[6] / y[6]) * y[4]
        x[5] = x[7] + ((x[1] - x[7]) / y[7]) * (y[7] - y[4])
        y[5] = y[4]
        x[3] = x[7] + ((x[1] - x[7]) / y[7]) * (y[7] - y[2])
        y[3] = y[2]
        c[1] = x[3] - x[2]
        c[2] = x[5] - x[4]
        TE_sweep = (np.arctan((x[7] - x[1]) / y[7])) * 180 / np.pi
        Sweep_40 = (np.arctan(((x[7] - 0.6 * (x[7] - x[6])) - 0.4 * x[1]) / y[7])) * 180 / np.pi

        l = np.multiply([c[i] for i in range(3)], 0.4 * np.cos(Z[4] * np.pi / 180))  # noqa: E741
        k = np.multiply([c[i] for i in range(3)], 0.6 * np.sin((90 - TE_sweep) * np.pi / 180) /
                       np.sin((90 + TE_sweep - Z[4]) * np.pi / 180))
        c_box = np.add(l, k)
        D_mx = np.subtract(l, np.multiply(0.407, c_box))

        return c, c_box, Sweep_40, D_mx, b, l

    def loads(self, b, c, Sweep_40, D_mx, L, Izz, Z, E):
        """Calculate load distribution and structural response."""
        NP = self.NP
        n = self.n
        rn = self.rn

        h = (b / 2) / n
        x = np.linspace(0, b / 2 - h, n)
        x1 = np.linspace(h, b / 2, n)

        # Calculate wing loading
        l = np.linspace(0, (b / 2) - (b / 2) / NP, NP)  # noqa: E741
        c1mc4 = c[0] - c[3]
        f_all = np.multiply((3 * b / 10), np.sqrt(np.subtract(1, np.power(x, 2) /
                            np.power(np.divide(b, 2), 2))))
        f1_all = np.multiply((3 * b / 10), np.sqrt(np.subtract(1, np.power(x1, 2) /
                             np.power(np.divide(b, 2), 2))))
        C = c[3] + 2 * ((b / 2 - x) / b) * c1mc4
        C1 = c[3] + 2 * ((b / 2 - x1) / b) * c1mc4
        A_Tot = np.multiply((h / 4) * (C + C1), (np.add(f_all, f1_all)))
        Area = np.sum(A_Tot.reshape((rn, NP), order='F'), axis=0)
        idx1 = np.arange(0, n, rn)
        idx2 = np.arange(rn - 1, n, rn)
        Spanel = (h * rn / 2) * (C[idx1] + C[idx2])

        # Calculate sweep angles
        cosSweep = np.cos(Sweep_40 * np.pi / 180)
        cosInvSweep = 1 / cosSweep
        tanCos2Sweep = np.tan(Sweep_40 * np.pi / 180) * cosSweep * cosSweep

        # Calculate distributed loads
        p = L * Area / np.sum(Area)

        # Calculate shear force and bending moment
        Tcsp = np.cumsum(p)
        Tsp = Tcsp[-1]
        temp = [0] + [Tcsp[i] for i in range(len(Tcsp) - 1)]
        T = np.subtract(Tsp, temp)
        pl = np.multiply(p, l)
        Tcspl = np.cumsum(pl)
        Tspl = Tcspl[-1]
        Mb = np.multiply(np.subtract(np.subtract(Tspl, Tcspl),
                                     np.multiply(l, np.subtract(Tsp, Tcsp))), cosInvSweep)

        # Extract loads at specific points
        idx = np.arange(0, NP, NP // 3)
        P = np.array(T[idx])
        Mz = np.array(Mb[idx])
        Mx = P * D_mx
        # Mz = [Mb[int(i)] for i in np.arange(0, NP - 1, int(NP / 3))]

        # Calculate wing twist due to bending
        I = np.zeros((NP))  # noqa: E741
        chord = c[3] + (np.divide(2 * (b / 2 - l), b)) * c1mc4
        y = np.zeros((2, 9))
        y[0, :] = (l - 0.4 * chord * tanCos2Sweep) * cosInvSweep
        y[1, :] = (l + 0.6 * chord * tanCos2Sweep) * cosInvSweep
        y[1, 0] = 0

        I[0:int(NP / 3)] = np.sqrt((Izz[0]**2 + Izz[1]**2) / 2)
        I[int(NP / 3):int(2 * NP / 3)] = np.sqrt((Izz[1]**2 + Izz[2]**2) / 2)
        I[int(2 * NP / 3):int(NP)] = np.sqrt((Izz[2]**2) / 2)

        La = y[0, 1:NP] - y[0, 0:NP - 1]
        La = np.append(0, La)
        Lb = y[1, 1:NP] - y[1, 0:NP - 1]
        Lb = np.append(0, Lb)

        A = T * La**3 / (3 * E * I) + Mb * La**2 / (2 * E * I)
        B = T * Lb**3 / (3 * E * I) + Mb * Lb**2 / (2 * E * I)
        Slope_A = T * La**2 / (2 * E * I) + Mb * La / (E * I)
        Slope_B = T * Lb**2 / (2 * E * I) + Mb * Lb / (E * I)

        for i in range(NP - 1):
            Slope_A[i + 1] = Slope_A[i] + Slope_A[i + 1]
            Slope_B[i + 1] = Slope_B[i] + Slope_B[i + 1]
            A[i + 1] = A[i] + Slope_A[i] * La[i + 1] + A[i + 1]
            B[i + 1] = B[i] + Slope_B[i] * Lb[i + 1] + B[i + 1]

        bend_twist = ((B - A) / chord) * 180 / np.pi

        # Ensure twist is non-decreasing
        for i in range(1, len(bend_twist)):
            if bend_twist[i] < bend_twist[i - 1]:
                bend_twist[i] = bend_twist[i - 1]

        return P, Mz, Mx, bend_twist, Spanel

    def calculate_structural_response(self):
        """Calculate the complete structural response of the wing."""
        # Calculate wing geometry
        self.c, self.c_box, self.Sweep_40, self.D_mx, self.b, self.a = self.Wing_Mod(self.Z, self.LAMBDA)

        # Calculate moments of inertia (MATLAB SBJ_obj_weight.m: box width l = 0.6*c_box)
        l = 0.6 * np.asarray(self.c_box)  # noqa: E741
        h = (np.multiply([self.c[i] for i in range(3)], self.beta * float(self.Z[0]))) - \
            np.multiply(0.5, np.add(self.ts1, self.ts3))

        A_top = 0.5 * self.t1 * l + self.t2 * h / 6
        A_bottom = 0.5 * self.t3 * l + self.t2 * h / 6
        Y_bar = np.multiply(h, np.divide((2 * A_top), (2 * A_top + 2 * A_bottom)))
        self.Izz = np.multiply(2, np.multiply(A_top, np.power((h - Y_bar), 2))) + \
              np.multiply(2, np.multiply(A_bottom, np.power((-Y_bar), 2)))

        # Calculate loads and deflections
        self.P, self.Mz, self.Mx, self.bend_twist, self.Spanel = self.loads(
            self.b, self.c, self.Sweep_40, self.D_mx, self.L, self.Izz, self.Z, self.E)

        # Calculate torsional deformation
        Phi = (self.Mx / (4 * self.G * (l * h)**2)) * (l / self.t1 + 2 * h / self.t2 + l / self.t3)

        # Calculate total twist (float array: an integer array truncated the twist, so theta was 0 most of the time)
        aa = len(self.bend_twist)
        self.twist = np.zeros(aa)
        self.twist[0:int(aa/3)] = self.bend_twist[0:int(aa/3)] + Phi[0] * 180 / np.pi
        self.twist[int(aa/3):int(aa*2/3)] = self.bend_twist[int(aa/3):int(aa*2/3)] + Phi[1] * 180 / np.pi
        self.twist[int(aa*2/3):aa] = self.bend_twist[int(aa*2/3):aa] + Phi[2] * 180 / np.pi

        # Calculate total twist contribution
        self.deltaL_divby_q = np.sum(self.twist * self.Spanel * 0.1 * 2)

        # Calculate structural weights
        self.Wtop_alum = (self.b / 4) * (self.c[0] + self.c[3]) * np.mean(self.t1) * self.rho_alum
        self.Wbottom_alum = (self.b / 4) * (self.c[0] + self.c[3]) * np.mean(self.t3) * self.rho_alum
        self.Wside_alum = (self.b / 2) * np.mean(h) * np.mean(self.t2) * self.rho_alum
        self.Wtop_core = (self.b / 4) * (self.c[0] + self.c[3]) * np.mean(np.subtract(self.ts1, self.t1)) * self.rho_core
        self.Wbottom_core = (self.b / 4) * (self.c[0] + self.c[3]) * np.mean(np.subtract(self.ts3, self.t3)) * self.rho_core
        self.Wside_core = (self.b / 2) * np.mean(h) * np.mean(np.subtract(self.ts2, self.t2)) * self.rho_core
        self.W_wingstruct = (self.Wtop_alum + self.Wbottom_alum + self.Wside_alum +
                           self.Wtop_core + self.Wbottom_core + self.Wside_core)
        self.W_fuel_wing = np.mean(h * l) * (self.b / 3) * 2 * self.rho_fuel

        # Calculate horizontal tail weight
        self.Bh = np.sqrt(self.ARht * self.Sht)
        self.W_ht = 3.316 * ((1 + (self.Fw_at_t / self.Bh))**-2.0) * ((self.L * self.C[2] / 1000)**0.260) * (self.Sht**0.806)

        # Calculate total weights
        Wf = self.C[0] + self.W_fuel_wing
        Ws = self.C[1] + self.W_ht + 2 * self.W_wingstruct
        theta = self.deltaL_divby_q

        return [Ws, Wf, theta]

    def calculate_constraints(self):
        """MATLAB-equivalent implementation of SBJ_constraint_weight"""

        # ---- Inputs (MATCH MATLAB) ----
        # t = np.array(self.t).reshape(-1) / 12.0
        # ts = np.array(self.ts).reshape(-1) / 12.0

        t1 = self.t[0:3]
        t2 = self.t[3:6]
        t3 = self.t[6:9]

        ts1 = self.ts[0:3]
        ts2 = self.ts[3:6]
        ts3 = self.ts[6:9]

        G = np.zeros(72)

        # Constants
        self.E = 10600000 * 144
        self.nu = 0.3
        self.beta = 0.9
        self.k = 6.09375

        # ---- Geometry ----
        self.c, self.c_box, self.Sweep_40, self.D_mx, self.b, self.a = self.Wing_Mod(self.Z, self.LAMBDA)

        c = np.array(self.c)

        l = 0.6 * self.c_box  # noqa: E741  IMPORTANT: matches MATLAB

        # ---- Equivalent thickness ----
        teq1 = ((t1**3)/4 + (3*t1)*(ts1 - t1/2)**2)**(1/3)
        teq2 = ((t2**3)/4 + (3*t2)*(ts2 - t2/2)**2)**(1/3)
        teq3 = ((t3**3)/4 + (3*t3)*(ts3 - t3/2)**2)**(1/3)

        # ---- Section geometry ----
        h = self.Z[0] * (self.beta * c[0:3]) - 0.5 * (ts1 + ts3)

        A_top = 0.5 * (t1 * l) + (1/6) * (t2 * h)
        A_bottom = 0.5 * (t3 * l) + (1/6) * (t2 * h)

        Y_bar = h * (2 * A_top) / (2 * A_top + 2 * A_bottom)

        self.Izz = 2 * A_top * (h - Y_bar)**2 + 2 * A_bottom * (-Y_bar)**2

        # ---- Loads ----
        P, Mz, Mx, *_ = self.loads(self.b, c, self.Sweep_40, self.D_mx, self.L, self.Izz, self.Z, self.E)

        # Ensure arrays
        P = np.array(P)
        Mz = np.array(Mz)
        Mx = np.array(Mx)

        # ---- Stresses (FIXED: use Y_bar and element-wise Izz) ----
        sig_1 = Mz * (0.95 * h - Y_bar) / self.Izz
        sig_2 = Mz * (h - Y_bar) / self.Izz
        sig_3 = sig_1
        sig_4 = Mz * (0.05 * h - Y_bar) / self.Izz
        sig_5 = Mz * (-Y_bar) / self.Izz
        sig_6 = sig_4

        q = Mx / (2 * l * h)  # FIXED: uses l

        # ---- Point 1 ----
        T1 = P * (l - self.a) / l  # FIXED
        tau1_T = T1 / (h * t2)
        tau1 = q / t2 + tau1_T

        sig_eq1 = np.sqrt(sig_1**2 + 3 * tau1**2)

        sig_cr1 = (np.pi**2 * self.E * 4 / (12 * (1 - self.nu**2))) * (teq2 / (0.95 * h))**2
        tau_cr1 = (np.pi**2 * self.E * 5.5 / (12 * (1 - self.nu**2))) * (teq2 / (0.95 * h))**2

        G[0:3] = self.k * sig_eq1
        G[3:6] = self.k * ((sig_1 / sig_cr1) + (tau1 / tau_cr1)**2)
        G[6:9] = self.k * ((-sig_1 / sig_cr1) + (tau1 / tau_cr1)**2)

        # ---- Point 2 ----
        tau2 = q / t1
        sig_eq2 = np.sqrt(sig_2**2 + 3 * tau2**2)

        sig_cr2 = (np.pi**2 * self.E * 4 / (12 * (1 - self.nu**2))) * (teq1 / l)**2
        tau_cr2 = (np.pi**2 * self.E * 5.5 / (12 * (1 - self.nu**2))) * (teq1 / l)**2

        G[9:12] = self.k * sig_eq2
        G[12:15] = self.k * ((sig_2 / sig_cr2) + (tau2 / tau_cr2)**2)
        G[15:18] = self.k * ((-sig_2 / sig_cr2) + (tau2 / tau_cr2)**2)

        # ---- Point 3 ----
        T2 = P * self.a / l  # FIXED
        tau3_T = -T2 / (h * t2)
        tau3 = q / t2 + tau3_T

        sig_eq3 = np.sqrt(sig_3**2 + 3 * tau3**2)

        sig_cr3 = sig_cr1
        tau_cr3 = tau_cr1

        G[18:21] = self.k * sig_eq3
        G[21:24] = self.k * ((sig_3 / sig_cr3) + (tau3 / tau_cr3)**2)
        G[24:27] = self.k * ((-sig_3 / sig_cr3) + (tau3 / tau_cr3)**2)

        # ---- Point 4 ----
        tau4 = -q / t2 + tau1_T
        sig_eq4 = np.sqrt(sig_4**2 + 3 * tau4**2)

        G[27:30] = self.k * sig_eq4

        # ---- Point 5 ----
        tau5 = q / t3
        sig_eq5 = np.sqrt(sig_5**2 + 3 * tau5**2)

        sig_cr5 = (np.pi**2 * self.E * 4 / (12 * (1 - self.nu**2))) * (teq3 / l)**2
        tau_cr5 = (np.pi**2 * self.E * 5.5 / (12 * (1 - self.nu**2))) * (teq3 / l)**2

        G[30:33] = self.k * sig_eq5
        G[33:36] = self.k * ((sig_5 / sig_cr5) + (tau5 / tau_cr5)**2)
        G[36:39] = self.k * ((-sig_5 / sig_cr5) + (tau5 / tau_cr5)**2)

        # ---- Point 6 ----
        tau6 = -q / t2 + tau3_T
        sig_eq6 = np.sqrt(sig_6**2 + 3 * tau6**2)

        G[39:42] = self.k * sig_eq6

        # ---- Constraints ----
        Sig_C = 65000 * 144
        Sig_T = 65000 * 144

        G1 = np.zeros(72)

        G1[0:3] = (G[0:3] / Sig_C) - 1
        G1[54:57] = -(G[0:3] / Sig_C) - 1
        G1[3:9] = G[3:9] - 1

        G1[9:12] = (G[9:12] / Sig_C) - 1
        G1[57:60] = -(G[9:12] / Sig_C) - 1
        G1[12:18] = G[12:18] - 1

        G1[18:21] = (G[18:21] / Sig_C) - 1
        G1[60:63] = -(G[18:21] / Sig_C) - 1
        G1[21:27] = G[21:27] - 1

        G1[27:30] = (G[27:30] / Sig_T) - 1
        G1[63:66] = -(G[27:30] / Sig_T) - 1

        G1[30:33] = (G[30:33] / Sig_T) - 1
        G1[66:69] = -(G[30:33] / Sig_T) - 1
        G1[33:39] = G[33:39] - 1

        G1[39:42] = (G[39:42] / Sig_T) - 1
        G1[69:72] = -(G[39:42] / Sig_T) - 1

        # ---- Thickness constraints ----
        G1[42:45] = 0.5 * (ts1 + ts3) / h - 1
        G1[45:48] = t1 / (ts1 - 0.1 * t1) - 1
        G1[48:51] = t2 / (ts2 - 0.1 * t2) - 1
        G1[51:54] = t3 / (ts3 - 0.1 * t3) - 1

        return G1.tolist()

    def SBJ_wing_structural_design(self):
        """Run the complete wing design analysis."""
        # Calculate structural response
        Ws, Wf, theta = self.calculate_structural_response()

        # Calculate constraints
        G1 = self.calculate_constraints()

        # Return results
        return {
            'Lift': self.Lift,
            'Ws': Ws,
            'Wf': Wf,
            'theta': theta,
            'G1': G1,
            'twist': self.twist,
            'Spanel': self.Spanel,
            'bend_twist': self.bend_twist
        }

    def SBJ_structure_analysis(self):
        return self.calculate_structural_response()

    def SBJ_structure_opt(self):
        return [0, self.calculate_constraints()]

    def print_results(self):
        """Print the analysis results."""
        out_dict = self.SBJ_wing_structural_design()
        print("=== Wing Design Analysis Results ===")
        print(f"Lift = {out_dict['Lift']}")
        print(f"Theta = {out_dict['theta']}")
        print(f"Ws = {out_dict['Ws']}")
        print(f"Wf = {out_dict['Wf']}")
        print(f"Twist = {out_dict['twist']}")
        print(f"Spanel = {out_dict['Spanel']}")
        print(f"Bend Twist = {out_dict['bend_twist']}")
        print(f"Constraints (G1) = {out_dict['G1']}")
        CV = []
        for i, g in enumerate(out_dict['G1']):
          if g > 0:
              CV.append((i, g))
        print(f"Constraints Violation (G1) = {CV}")
        print("==================================")

if __name__ == "__main__":
    # twist =  [0 0 0 0 0 0 0 0 0]
    # Spanel =  [41.23931624 37.91547958 34.59164292 31.26780627 27.94396961 24.62013295
    #  21.2962963  17.97245964 14.64862298]
    # t =  [0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25]
    # ts =  [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]
    # Lift =  25000
    # Theta =  0.0
    # Ws = 20293.722985582557
    # Wf = 2158.127657838667
    wda = WingDesignAnalyzer()
    wda.print_results()
