import numpy as np
import copy

class SSBJAerodynamics:
    """
    Class for SSBJ (Supersonic Business Jet) aerodynamic calculations.
    This class implements the drag polar and constraint calculations
    for supersonic aircraft design.
    """

    def __init__(self, h=55000, Mach=1.4, tc=0.05, ARw=3.0, LAMBDAw=60.0, LAMBDAht=45.0,
                 Sref=500.0, Sht=100.0, ARht=5.5, Lw=0.15, Lht=1.5, Wt=25000, theta=10.0, ESFp=1.0):
        """
        Initialize the SSBJ aerodynamics class with given parameters.

        Args:
            h: Altitude in feet
            Mach: Mach number
            tc: Thickness-to-chord ratio
            ARw: Wing aspect ratio
            LAMBDAw: Wing sweep angle in degrees
            LAMBDAht: Horizontal tail sweep angle in degrees
            Sref: Wing reference area in ft²
            Sht: Horizontal tail area in ft²
            ARht: Horizontal tail aspect ratio
            Lw: Wing moment arm
            Lht: Horizontal tail moment arm
            Wt: Total aircraft weight in lb
            theta: Incidence angle in degrees
            ESFp: Equivalent skin friction coefficient
        """
        # Constants
        self.C = [500.0, 16000.0, 4.0, 4360.0, 0.01375, 1.0]  # CDminM = C[4]
        self.Nh = self.C[5]

        # Input parameters
        self.h = h
        self.Mach = Mach
        self.tc = tc
        self.ARw = ARw
        self.LAMBDAw = LAMBDAw
        self.LAMBDAht = LAMBDAht
        self.Sref = Sref
        self.Sht = Sht
        self.ARht = ARht
        self.Lw = Lw
        self.Lht = Lht
        self.Wt = Wt
        self.theta = theta
        self.ESFp = ESFp

        # Local variables
        self.Z = [self.tc, self.h, self.Mach, self.ARw, self.LAMBDAw, self.Sref, self.Sht, self.ARht]

        # Initialize results
        self.Lift = 0
        self.Drag = 0
        self.LDr = 0
        self.Pg = 0
        self.g1 = 0
        self.g2 = 0
        self.g3 = 0
        self.CLo = np.array([0.0, 0.0])
        self.DCL = np.array([0.0, 0.0])

    def poly_approx(self, S, S_new, flag, S_bound):
        """
        Polynomial approximation function for scaling factors.

        Args:
            S: Base values
            S_new: New values to scale
            flag: Flag for different calculation modes
            S_bound: Boundaries for scaling

        Returns:
            FF: Scaled factor
        """
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
                # CALCULATE POLYNOMIAL COEFFICIENTS (S-ABOUT ORIGIN)
                So = 0
                Sl = So - S_bound[i]
                Su = So + S_bound[i]
                Mtx_shifted = np.array([[1, Sl, Sl**2], [1, So, So**2], [1, Su, Su**2]])

                F_bound = np.array([1 + (.5*a)**2, 1, 1 + (.5*b)**2])
                A = np.linalg.solve(Mtx_shifted, F_bound)
                Ao = A[0]
                Ai.append(A[1])
                Aij[i, i] = A[2]

                # CALCULATE POLYNOMIAL COEFFICIENTS
            else:
                if flag[i] == 0:
                    S_shifted.append(0)
                elif flag[i] == 3:
                    a *= -1
                    b = copy.deepcopy(a)
                elif flag[i] == 2:
                    b = 2 * a
                elif flag[i] == 4:
                    a *= -1
                    b = 2*a
                # DETERMINE BOUNDS ON FF DEPENDING ON SLOPE-SHAPE
                # CALCULATE POLYNOMIAL COEFFICIENTS (S-ABOUT ORIGIN)
                So = 0
                Sl = So - S_bound[i]
                Su = So + S_bound[i]
                Mtx_shifted = np.array([[1, Sl, Sl**2], [1, So, So**2], [1, Su, Su**2]])
                F_bound = np.array([1 - .5*a, 1, 1 + .5*b])
                A = np.linalg.solve(Mtx_shifted, F_bound)
                Ao = A[0]
                Ai.append(A[1])
                Aij[i, i] = A[2]

                # CALCULATE POLYNOMIAL COEFFICIENTS

        # Correlation matrix
        R = np.array([[0.2736, 0.3970, 0.8152, 0.9230, 0.1108],
                      [0.4252, 0.4415, 0.6357, 0.7435, 0.1138],
                      [0.0329, 0.8856, 0.8390, 0.3657, 0.0019],
                      [0.0878, 0.7248, 0.1978, 0.0200, 0.0169],
                      [0.8955, 0.4568, 0.8075, 0.9239, 0.2525]])

        for i in range(len(S)):
            for j in range(i+1, len(S)):
                Aij[i, j] = Aij[i, i] * R[i, j]
                Aij[j, i] = Aij[i, j]

        S_shifted = np.array(S_shifted)
        FF = Ao + np.dot(Ai, np.transpose(S_shifted)) + (1/2) * np.dot(np.dot(S_shifted, Aij), np.transpose(S_shifted))
        return FF

    def calculate_drag_polar(self):
        """
        Calculate the drag polar and related parameters.
        """
        # Extract variables from Z
        ARht = self.Z[7]
        S_ht = self.Z[6]

        # Calculate velocity and density
        if self.Z[1] < 36089:
            V = self.Z[2] * (1116.39 * np.sqrt(1 - (6.875e-06 * self.Z[1])))
            rho = (2.377e-03) * (1 - (6.875e-06 * self.Z[1]))**4.2561
        else:
            V = self.Z[2] * 968.1
            rho = (2.377e-03) * (.2971) * np.exp(-(self.Z[1] - 36089) / 20806.7)

        q = 0.5 * rho * (V**2)

        # Scale coefficients for proper conditioning of matrix A
        a = q * self.Z[5] / 1e5
        b = self.Nh * q * S_ht / 1e5
        c = self.Lw
        d = self.Lht * self.Nh * (S_ht / self.Z[5])

        A = np.array([[a, b], [c, d]])

        # Scale coefficient Wt for proper conditioning of matrix A
        B = np.array([self.Wt / 1e5, 0])

        # Solve for CLo
        try:
            CLo = np.linalg.solve(A, B)
        except:  # noqa: E722
            CLo = np.array([-np.inf, np.inf])

        # Calculate delta_L
        delta_L = self.theta * q
        Lw1 = CLo[0] * q * self.Z[5] - delta_L
        CLw1 = Lw1 / (q * self.Z[5])
        CLht1 = -CLw1 * c / d

        # Scale first coefficient of D for proper conditioning of matrix A
        D = np.array([(self.Wt - CLw1 * a - CLht1 * b) / 1e5, -CLw1 * c - CLht1 * d])

        # Solve for DCL
        try:
            self.DCL = np.linalg.solve(A, D)
        except:  # noqa: E722
            self.DCL = np.array([np.nan, np.nan])

        # Calculate induced drag factors
        if self.Z[2] >= 1:
            kw = self.Z[3] * (self.Z[2]**2 - 1) * np.cos(self.Z[4] * np.pi / 180) / \
            (4 * self.Z[3] * np.sqrt(self.Z[2]**2 - 1) - 2)
            kht = ARht * (self.Z[2]**2 - 1) * np.cos(self.LAMBDAht * np.pi / 180) / \
            (4 * ARht * np.sqrt(self.Z[2]**2 - 1) - 2)
        else:
            kw = 1 / (np.pi * 0.8 * self.Z[3])
            kht = 1 / (np.pi * 0.8 * ARht)

        # Calculate Fo1
        S_initial1 = copy.deepcopy(self.ESFp)
        S1 = copy.deepcopy(self.ESFp)
        flag1 = 1
        bound1 = 0.25
        Fo1 = self.poly_approx(S_initial1 if isinstance(S_initial1, list) else [S_initial1],
                              S1 if isinstance(S1, list) else [S1],
                              flag1 if isinstance(flag1, list) else [flag1],
                              bound1 if isinstance(bound1, list) else [bound1])

        # Calculate minimum drag coefficient
        CDmin = self.C[4] * Fo1 + 3.05 * (self.Z[0]**(5/3)) * ((np.cos(self.Z[4] * np.pi / 180))**(3/2))

        # Calculate total drag coefficients
        CDw = CDmin + kw * (CLo[0]**2) + kw * (self.DCL[0]**2)
        CDht = kht * (CLo[1]**2) + kht * (self.DCL[1]**2)
        CDp = CDw + CDht
        CLp = CLo[0] + CLo[1]

        # Calculate lift and drag
        Lift = self.Wt
        Drag = q * CDw * self.Z[5] + q * CDht * self.Z[6]
        LDr = CLp / CDp

        # Calculate adverse pressure gradient (G2)
        S_initial2 = copy.deepcopy(self.tc)
        S2 = copy.deepcopy(self.Z[0])
        flag1 = [1]
        bound1 = [0.25]

        Pg = self.poly_approx(S_initial2 if isinstance(S_initial2, list) else [S_initial2],
                             S2 if isinstance(S2, list) else [S2],
                             flag1 if isinstance(flag1, list) else [flag1],
                             bound1 if isinstance(bound1, list) else [bound1])

        return [Lift, Drag, LDr, Pg, CLo[0], CLo[1]]

    def calculate_constraints(self, Pg, CLo):
        """
        Calculate the constraints for the aerodynamic analysis.
        """

        # Constraints
        Pg_uA = 1.1
        if CLo[0] > 0:
            g2 = (2 * CLo[1]) - CLo[0]
            g3 = (2 * (-CLo[1])) - CLo[0]
        else:
            g2 = (2 * (-CLo[1])) - CLo[0]
            g3 = (2 * CLo[1]) - CLo[0]

        g1 = Pg / Pg_uA - 1
        
        return [g1, g2, g3]

    def SBJ_aerodynamics_analysis(self):
        """
        Run the complete aerodynamic analysis.
        """
        return self.calculate_drag_polar()
    
    def SBJ_aerodynamics_opt(self, Pg, CLo):
        """
        Run the complete aerodynamic analysis.
        """
        
        return [0, self.calculate_constraints(Pg, CLo)]

    def print_results(self):
        """Print the analysis results."""
        Lift, Drag, LDr, Pg, CLo0, CLo1 = self.SBJ_aerodynamics_analysis()
        g1, g2, g3 = self.calculate_constraints(Pg, [CLo0, CLo1])
        print("=== Wing Design Analysis Results ===")
        print(f"Lift = {Lift}")
        print(f"Drag = {Drag}")
        print(f"LDr = {LDr}")
        print(f"Pg = {Pg}")
        print(f"Clo0 = {CLo0}")
        print(f"CLo1 = {CLo1}")
        print(f"g1 = {g1}")
        print(f"g2 = {g2}")
        print(f"g3 = {g3}")
        print("==================================")

# Example usage:
if __name__ == "__main__":
    # Create an instance with default parameters
    aerodynamics = SSBJAerodynamics()
    # Run the analysis
    print(aerodynamics.print_results())

# /SSBJ_Aerodynamics.py
# Drag =  4608.661215792792
# LDr =  2.7450900636447657
# Lift =  25000
# G2 =  1.0
# g1 =  -0.09090909090909094
# g2 =  -0.42509404401935696
# g3 =  -5.551115123125783e-17