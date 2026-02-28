import numpy as np
import copy 

#Initial values
# from original text:
#       λ     x	    Cf	    T	  t/c	  h	      M	      AR	  Δ	  SREF
# vlb	0.1	  0.75	0.75	0.1	  0.01	30000	  1.4	  2.5	  40	500
# i0	0.25  1	    1	    0.5	  0.05	45000	  1.6	  5.5	  55	1000
# vub	0.4	  1.25	1.25	1	  0.09	60000	  1.8	  8.5	  70	1500

#
# used here, from Tosserams; baselines set to the middle of the range
# "sp_index": 3 , f"name": "D"          , "coupling_type": ff   , "link": 2     , "lb": 1000  , "ub": 70000   
# "sp_index": 3 , f"name": "ESF"        , "coupling_type": fb   , "link": 2     , "lb": 0.5   , "ub": 1.5     
# "sp_index": 3 , f"name": "Wt"         , "coupling_type": fb   , "link": 1     , "lb": 5000  , "ub": 100000  
# "sp_index": 3 , f"name": "LD"         , "coupling_type": ff   , "link": 1     , "lb": 0.1   , "ub": 10      
# "sp_index": 3 , f"name": "theta"      , "coupling_type": fb   , "link": 4     , "lb": 0.2   , "ub": 50      
# "sp_index": 3 , f"name": "L"          , "coupling_type": ff   , "link": 4     , "lb": 5000  , "ub": 100000  

# "sp_index": 3 , f"name": "tc"         , "coupling_type": s    , "link": 4     , "lb": 0.01  , "ub": 0.1     
# "sp_index": 3 , f"name": "ARw"        , "coupling_type": s    , "link": 4     , "lb": 2.5   , "ub": 8.0     
# "sp_index": 3 , f"name": "LAMBDAw"    , "coupling_type": s    , "link": 4     , "lb": 40.   , "ub": 70.     
# "sp_index": 3 , f"name": "Sref"       , "coupling_type": s    , "link": 4     , "lb": 200.  , "ub": 800.    
# "sp_index": 3 , f"name": "Sht"        , "coupling_type": s    , "link": 4     , "lb": 50    , "ub": 148.9   
# "sp_index": 3 , f"name": "ARht"       , "coupling_type": s    , "link": 4     , "lb": 2.5   , "ub": 8.5     

# "sp_index": 3 , f"name": "LAMBDAht"   , "coupling_type": un   , "link": None  , "lb": 40.   , "ub": 70.     
# "sp_index": 3 , f"name": "Lw"         , "coupling_type": un   , "link": None  , "lb": 0.01  , "ub": 0.2     
# "sp_index": 3 , f"name": "Lht"        , "coupling_type": un   , "link": None  , "lb": 1     , "ub": 3.5     


#######################
#Inputs
h = 55000
Mach = 1.4

tc = 0.05 #Thickness/Chord 0.01<=0.05<=0.1 []
ARw = 3.0 #Wing aspect ratio 2.5<=3.0<=8.0 []; orig paper BL=5.5
LAMBDAw = 60. #Wing sweep angle 40<=60<=70 [deg]
LAMBDAht = 45. #Wing sweep angle 40<=60<=70 [deg]
Sref = 500.0 #Wing surface area 200<=500<=800 [ft2]; orig paper BL=1000
Sht = 100.0
ARht = 5.5
Lw = 0.15
Lht = 1.5
Wt = 25000 #Total weight [lb]
theta = 10.0
ESFp = 1.0

#Constants
C = [500.0, 16000.0,  4.0,  4360.0,  0.01375,  1.0]
# CDminM = C[4] #Coef of drag

#Locals
Z = [tc,h,Mach,ARw,LAMBDAw,Sref,Sht,ARht] # these are the shared variables

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
# SBJ_obj_dragpolar
# Z = [1,	55000,	1.40000000000000,	1,	1,	1,	1,	1]
ARht=Z[7]
S_ht=Z[6]    
Nh=C[5]

# %-----Drag computations----%
if Z[1]<36089:
   V = Z[2]*(1116.39*np.sqrt(1-(6.875e-06*Z[1])))
   rho = (2.377e-03)*(1-(6.875e-06*Z[1]))**4.2561
else:
   V = Z[2]*968.1
   rho = (2.377e-03)*(.2971)*np.exp(-(Z[1]-36089)/20806.7)
q=.5*rho*(V**2)

# ### Modified by S. Tosserams:
# # scale coefficients for proper conditioning of matrix A 
a=q*Z[5]/1e5
b=Nh*q*S_ht/1e5
# -------------------------------
c= Lw
d= Lht * Nh * (S_ht/Z[5])
A= np.array([[a, b], [c, d]])
# ---- Modified by S. Tosserams:
# ---- scale coefficient Wt for proper conditioning of matrix A 
B=np.array([Wt/1e5, 0])
# -----------------------
try:
  CLo=np.linalg.solve(A, B)
except:
  CLo = np.array([-np.inf, np.inf])
delta_L = theta * q
Lw1 = CLo[0]*q*Z[5]-delta_L
CLw1 = Lw1/(q*Z[5])
CLht1 = -CLw1*c/d
#  Modified by S. Tosserams:
#  scale first coefficient of D for proper conditioning of matrix A 
D=np.array([(Wt-CLw1*a-CLht1*b)/1e5, -CLw1*c-CLht1*d])
# -----------------
try:
  DCL = np.linalg.solve(A, D)
except:
  DCL = np.array([np.nan,np.nan])

if Z[2] >= 1:
  kw = Z[3] * (Z[2]**2-1) * np.cos(Z[4]*np.pi/180)/(4*Z[3]*np.sqrt(Z[2]**2-1)-2)
  kht = ARht * (Z[2]**2-1)*np.cos(LAMBDAht*np.pi/180)/(4*ARht*np.sqrt(Z[2]**2-1)-2)
else:
  kw = 1/(np.pi*0.8*Z[3])
  kht= 1/(np.pi*0.8*ARht)
  
S_initial1 = copy.deepcopy(ESFp)
S1 = copy.deepcopy(ESFp)
flag1 = 1
bound1 = 0.25
Fo1 = polyApprox(S_initial1 if isinstance(S_initial1, list) else [S_initial1], S1 if isinstance(S1, list) else [S1], flag1 if isinstance(flag1, list) else [flag1], bound1 if isinstance(bound1, list) else [bound1])

CDmin = C[4]*Fo1 + 3.05*(Z[0]**(5/3))*((np.cos(Z[4]*np.pi/180))**(3/2))

CDw=CDmin+kw*(CLo[0]**2)+kw*(DCL[0]**2)
CDht=kht*(CLo[1]**2)+kht*(DCL[1]**2)
CDp=CDw+CDht
CLp=CLo[0]+CLo[1]

# return [L, D, LD, Pg, CLo[0], CLo[1]] # these are responses
Lift = Wt
Drag = q*CDw*Z[5]+q*CDht*Z[6]
LDr = CLp/CDp

# SBJ_constraint_dragpolar
S_initial2 = copy.deepcopy(tc)
S2 = copy.deepcopy(Z[0])
flag1 = [1]
bound1 = [0.25]

#adverse pressure gradient also mentioned as G2  
Pg = polyApprox(S_initial2 if isinstance(S_initial2, list) else [S_initial2], S2 if isinstance(S2, list) else [S2], flag1 if isinstance(flag1, list) else [flag1], bound1 if isinstance(bound1, list) else [bound1])

#Constraints
Pg_uA=1.1
if CLo[0] > 0:
    g2=(2*(CLo[1]))-(CLo[0])
    g3=(2*(-CLo[1]))-(CLo[0])
else:
    g2=(2*(-CLo[1]))-(CLo[0])
    g3=(2*(CLo[1]))-(CLo[0])

g1=Pg/Pg_uA-1

#Printing
# print('Drag = ', Drag)
# print('LDr = ', LDr)
# print('Lift = ', Lift)
# print('G2 = ', Pg)
# print('g1 = ', g1)
# print('g2 = ', g2)
# print('g3 = ', g3)