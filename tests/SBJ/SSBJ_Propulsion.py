import numpy as np
import copy 

#Initial values
# from original text:
#       λ     x	    Cf	    T	  t/c	  h	      M	      AR	  Δ	  SREF
# vlb	0.1	  0.75	0.75	0.1	  0.01	30000	  1.4	  2.5	  40	500
# i0	0.25  1	    1	    0.5	  0.05	45000	  1.6	  5.5	  55	1000
# vub	0.4	  1.25	1.25	1	  0.09	60000	  1.8	  8.5	  70	1500

# used here, from Tosserams; baselines set to the middle of the range
# "sp_index": 2 , f"name": "SFC"        , "coupling_type": ff   , "link": 1     , "lb": 1     , "ub": 4       
# "sp_index": 2 , f"name": "We"         , "coupling_type": ff   , "link": 1     , "lb": 100   , "ub": 30000      
# "sp_index": 2 , f"name": "D"          , "coupling_type": fb   , "link": 3     , "lb": 1000  , "ub": 70000      
# "sp_index": 2 , f"name": "ESF"        , "coupling_type": ff   , "link": 3     , "lb": 0.5   , "ub": 1.5      
# "sp_index": 2 , f"name": "T"          , "coupling_type": un   , "link": None  , "lb": 0.1   , "ub": 1.0     

#######################
#Inputs
h = 55000
Mach = 1.4

Drag =  40000
Throttle = 0.6 # T

#Constants
C=[500.0, 16000.0, 4.0 , 4360.0,  0.01375,  1.0] # list of constants from original paper
Wbe = C[3] # constant weight [lbs]

#Locals

#Poly function
def polyApprox(S, S_new, flag, S_bound): #SP2, SP3
  S_norm = []
  S_shifted = []
  Ai = []
  Aij = np.zeros((len(S),len(S)))
  for i in range(len(S)):
    S_norm.append(S_new[i]/S[i])
    if S_norm[i]>1.25:
      S_norm[i]=1.25
    elif S_norm[i]<0.75:
        S_norm[i]=0.75
    S_shifted.append(S_norm[i] - 1)
    a = 0.1
    b = a

    if flag[i]==5:
      # CALCULATE POLYNOMIAL COEFFICIENTS (S-ABOUT ORIGIN)
      So=0
      Sl=So-S_bound[i]
      Su=So+S_bound[i]
      Mtx_shifted = np.array([[1, Sl, Sl**2], [1, So, So**2], [1, Su, Su**2]])

      F_bound = np.array([1+(.5*a)**2, 1, 1+(.5*b)**2])
      A = np.linalg.solve(Mtx_shifted, F_bound)
      Ao = A[0]
      Ai.append(A[1])
      Aij[i,i] = A[2]

      # CALCULATE POLYNOMIAL COEFFICIENTS
    else:
      if flag[i] == 0:
        S_shifted.append(0)
      elif flag[i]==3:
        a *= -1.
        b=copy.deepcopy(a)
      elif flag[i]==2:
        b = 2 * a
      elif flag[i] == 4:
        a *= -1
        b = 2*a
      # DETERMINE BOUNDS ON FF DEPENDING ON SLOPE-SHAPE
      #  CALCULATE POLYNOMIAL COEFFICIENTS (S-ABOUT ORIGIN)
      So=0
      Sl=So-S_bound[i]
      Su=So+S_bound[i]
      Mtx_shifted = np.array([[1, Sl, Sl**2], [1, So, So**2], [1, Su, Su**2]])
      F_bound = np.array([1-.5*a, 1, 1+.5*b])
      A = np.linalg.solve(Mtx_shifted, F_bound)
      Ao = A[0]
      Ai.append(A[1])
      Aij[i,i] = A[2]
    
      #  CALCULATE POLYNOMIAL COEFFICIENTS
  R = np.array([[0.2736,    0.3970,    0.8152,    0.9230,    0.1108], 
                [0.4252,    0.4415,    0.6357,    0.7435,    0.1138],
                [0.0329,    0.8856,    0.8390,    0.3657,    0.0019],
                [0.0878,    0.7248,    0.1978,    0.0200,    0.0169],
                [0.8955,    0.4568,    0.8075,    0.9239,    0.2525]])
  
  for i in range(len(S)):
    for j in range(i+1, len(S)):
      Aij[i, j] = Aij[i,i] * R[i,j]
      Aij[j, i] = Aij[i, j]

  S_shifted = np.array(S_shifted)
  
  FF = Ao + np.dot(Ai, (np.transpose(S_shifted))) + (1/2)*np.dot(np.dot(S_shifted, Aij), np.transpose(S_shifted))

  return FF

#Calculations
#  THIS SECTION COMPUTES SFC, ESF, AND ENGINE WEIGHT
Thrust = Drag
Dim_Throttle = Throttle * 16168 # nondimensional throttle setting

s=[1.13238425638512, 1.53436586044561, -0.00003295564466, -0.00016378694115, -0.31623315541888, 0.00000410691343, -0.00005248000590, -0.00000000008574, 0.00000000190214, 0.00000001059951]
SFCp = s[0]+s[1]*Mach+s[2]*h+s[3]*Dim_Throttle+s[4]*Mach**2+2*h*Mach*s[5]+2*Dim_Throttle*Mach*s[6]+s[7]*h**2+2*Dim_Throttle*h*s[8]+s[9]*Dim_Throttle**2

ESFp = (Thrust/2)/Dim_Throttle

We = Wbe*(ESFp**1.05)*2

S_initial1=[Mach,h,Drag]
S1=[Mach,h,Throttle]
flag1 = [2,4,2]
bound1 = [.25,.25,.25]
Temp_E = polyApprox(S_initial1, S1, flag1, bound1)

p=[11483.7822254806, 10856.2163466548, -0.5080237941, 3200.157926969, -0.1466251679, 0.0000068572]
Throttle_uA=p[0]+p[1]*Mach+p[2]*h+p[3]*Mach**2+2*p[4]*Mach*h+p[5]*h**2


#Constraints
# -----THIS SECTION COMPUTES POLYNOMIAL CONSTRAINT FUNCTIONS-----
Dim_Throttle = Throttle * 16168
Temp_uA = 1.02
g1 = Temp_E /Temp_uA - 1
g2 = Dim_Throttle/Throttle_uA - 1

#Printing
# print('SFCp = ', SFCp)
# print('ESFp = ', ESFp)
# print('We = ', We)
# print('g1 = ', g1)
# print('g2 = ', g2)
