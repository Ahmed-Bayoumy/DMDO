# SSBJ Propulsion Class

import copy

import numpy as np


class SSBJPropulsion:
    """Class for SSBJ propulsion system calculations."""

    def __init__(self, D=40000, T=0.6, h=55000, Mach=1.4):
        """Initialize constants and default values."""
        # Constants from original paper
        self.C = [500.0, 16000.0, 4.0, 4360.0, 0.01375, 1.0]
        self.Wbe = self.C[3]  # constant weight [lbs]
        
        # Polynomial coefficient matrix
        self.R = np.array([
            [0.2736, 0.3970, 0.8152, 0.9230, 0.1108],
            [0.4252, 0.4415, 0.6357, 0.7435, 0.1138],
            [0.0329, 0.8856, 0.8390, 0.3657, 0.0019],
            [0.0878, 0.7248, 0.1978, 0.0200, 0.0169],
            [0.8955, 0.4568, 0.8075, 0.9239, 0.2525]
        ])
        
        # Throttle scaling factor
        self.throttle_scale = 16168
        
        # Initial values
        self.h = 55000
        self.Mach = 1.4
        self.Drag = D
        self.Throttle = T
        
        # SFC coefficients
        self.s = [1.13238425638512, 1.53436586044561, -0.00003295564466, 
                 -0.00016378694115, -0.31623315541888, 0.00000410691343, 
                 -0.00005248000590, -0.00000000008574, 0.00000000190214, 
                 0.00000001059951]
        
        # Throttle coefficients
        self.p = [11483.7822254806, 10856.2163466548, -0.5080237941, 
                 3200.157926969, -0.1466251679, 0.0000068572]
        
        # Constraint values
        self.Temp_uA = 1.02
        self.Throttle_uA = 16168 * 0.6
        
    def poly_approx(self, S, S_new, flag, S_bound):
        """Calculate polynomial approximation for propulsion parameters."""
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
                Aij[i,i] = A[2]
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
                Aij[i,i] = A[2]
            
        # Calculate cross terms
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                Aij[i, j] = Aij[i,i] * self.R[i,j]
                Aij[j, i] = Aij[i, j]
        
        S_shifted = np.array(S_shifted)
        
        # Calculate FF (Performance Factor)
        FF = Ao + np.dot(Ai, (np.transpose(S_shifted))) + \
             (1/2) * np.dot(np.dot(S_shifted, Aij), np.transpose(S_shifted))
        
        return FF
    
    def calculate_sfc(self, Mach, h, Throttle):
        """Calculate Specific Fuel Consumption (SFC)."""
        return (self.s[0] + self.s[1]*Mach + self.s[2]*h + self.s[3]*Throttle + \
                self.s[4]*Mach**2 + 2*h*Mach*self.s[5] + 2*Throttle*Mach*self.s[6] + \
                self.s[7]*h**2 + 2*Throttle*h*self.s[8] + self.s[9]*Throttle**2)
    
    def calculate_esf(self, Thrust, Throttle):
        """Calculate Engine Specific Force (ESF)."""
        return (Thrust / 2) / Throttle
    
    def calculate_engine_weight(self, ESFp):
        """Calculate engine weight based on ESF."""
        return self.Wbe * (ESFp**1.05) * 2
    
    def calculate_tempe_throttleua(self):
        """ Calculate Temp_E and  Throttle_uA """
        Temp_E = self.poly_approx([self.Mach, self.h, self.Drag], [self.Mach, self.h, self.Throttle], [2, 4, 2],\
                                   [.25, .25, .25])
        
        Throttle_uA = self.p[0] + self.p[1]*self.Mach + self.p[2]*self.h + self.p[3]*self.Mach**2 + \
                     2*self.p[4]*self.Mach*self.h + self.p[5]*self.h**2
        return Temp_E, Throttle_uA
    
    def calculate_constraints(self, Temp_E, Throttle_uA):
        """Calculate constraint functions for the propulsion system."""
        Dim_Throttle = self.Throttle * self.throttle_scale
        # SFCp = self.calculate_sfc(self.Mach, self.h, Dim_Throttle)
        
        # # Calculate ESF
        # ESFp = self.calculate_esf(self.Drag, Dim_Throttle)
        # Temp_E = self.poly_approx([Mach, h, Throttle], [Mach, h, Throttle], [2, 4, 2], [.25, .25, .25])
        
        # Throttle_uA = self.p[0] + self.p[1]*Mach + self.p[2]*h + self.p[3]*Mach**2 + \
        #              2*self.p[4]*Mach*h + self.p[5]*h**2
        
        g1 = Temp_E / self.Temp_uA - 1
        g2 = Dim_Throttle / Throttle_uA - 1
        return [g1, g2]
            
    
    def SBJ_propulsion_analysis(self):
        """Run complete analysis with current parameters."""
        Dim_Throttle = self.Throttle * self.throttle_scale
        # Calculate SFC
        SFCp = self.calculate_sfc(self.Mach, self.h, Dim_Throttle)
        
        # Calculate ESF
        ESFp = self.calculate_esf(self.Drag, Dim_Throttle)
        
        # Calculate engine weight
        We = self.calculate_engine_weight(ESFp)
        
        # Print results
        # print(f'SFCp = {SFCp}')
        # print(f'ESFp = {ESFp}')
        # print(f'We = {We}')
        # print(f'g1 = {g1}')
        # print(f'g2 = {g2}')
        Temp_E, Throttle_uA = self.calculate_tempe_throttleua()
        return [SFCp, We, ESFp, Temp_E, Throttle_uA]
    
    def SBJ_propulsion_opt(self, Temp_E, Throttle_uA):
        return [0, self.calculate_constraints(Temp_E, Throttle_uA)]
    
    def print_results(self):
        SFCp, We, ESFp, Temp_E, Throttle_uA = self.SBJ_propulsion_analysis()
        G = self.calculate_constraints(Temp_E, Throttle_uA)
        print('SFCp = ', SFCp)
        print('ESFp = ', ESFp)
        print('Temp_E = ', Temp_E)
        print('Throttle_uA = ', Throttle_uA)
        print('Dim_Throttle = ', self.Throttle * self.throttle_scale)
        print('We = ', We)
        print('g1 = ', G[0])
        print('g2 = ', G[1])


    
if __name__ == "__main__":
    prop = SSBJPropulsion()
    prop.print_results()