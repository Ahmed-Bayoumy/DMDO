import math
import numpy as np
import copy
from typing import List, Dict, Any, Callable, Protocol, Optional

#Initial values
# from original text:
#     λ	    x	    Cf	  T	    t/c	  h	      M	    AR	  Δ	  SREF
# vlb	0.1	  0.75	0.75	0.1	  0.01	30000	  1.4	  2.5	  40	500
# i0	0.25	1	    1	    0.5	  0.05	45000	  1.6	  5.5	  55	1000
# vub	0.4	  1.25	1.25	1	    0.09	60000	  1.8	  8.5	  70	1500

#
# used here, from Tosserams; baselines set to the middle of the range
# "sp_index": 4 , f"name": "theta"      , "coupling_type": ff   , "link": 3     , "lb": 0.2   , "ub": 50      
# "sp_index": 4 , f"name": "L"          , "coupling_type": fb   , "link": 3     , "lb": 5000  , "ub": 100000  
# "sp_index": 4 , f"name": "Ws"         , "coupling_type": ff   , "link": 1     , "lb": 5000  , "ub": 100000  
# "sp_index": 4 , f"name": "Wf"         , "coupling_type": ff   , "link": 1     , "lb": 5000  , "ub": 100000  

# "sp_index": 4 , f"name": "tc"         , "coupling_type": s    , "link": 3     , "lb": 0.01  , "ub": 0.1     
# "sp_index": 4 , f"name": "ARw"        , "coupling_type": s    , "link": 3     , "lb": 2.5   , "ub": 8.0     
# "sp_index": 4 , f"name": "LAMBDAw"    , "coupling_type": s    , "link": 3     , "lb": 40.   , "ub": 70.     
# "sp_index": 4 , f"name": "Sref"       , "coupling_type": s    , "link": 3     , "lb": 200.  , "ub": 800.    
# "sp_index": 4 , f"name": "Sht"        , "coupling_type": s    , "link": 3     , "lb": 50    , "ub": 148.9   
# "sp_index": 4 , f"name": "ARht"       , "coupling_type": s    , "link": 3     , "lb": 2.5   , "ub": 8.5     

# "sp_index": 4 , f"name": "lambda"     , "coupling_type": un   , "link": None  , "lb": 0.1   , "ub": 0.4     
# "sp_index": 4 , f"name": "t"          , "coupling_type": un   , "link": None  , "lb": 0.1   , "ub": 4.0     
# "sp_index": 4 , f"name": "t"          , "coupling_type": un   , "link": None  , "lb": 0.1   , "ub": 4.0     
# "sp_index": 4 , f"name": "t"          , "coupling_type": un   , "link": None  , "lb": 0.1   , "ub": 4.0     
# "sp_index": 4 , f"name": "t"          , "coupling_type": un   , "link": None  , "lb": 0.1   , "ub": 4.0     
# "sp_index": 4 , f"name": "t"          , "coupling_type": un   , "link": None  , "lb": 0.1   , "ub": 4.0     
# "sp_index": 4 , f"name": "t"          , "coupling_type": un   , "link": None  , "lb": 0.1   , "ub": 4.0     
# "sp_index": 4 , f"name": "t"          , "coupling_type": un   , "link": None  , "lb": 0.1   , "ub": 4.0     
# "sp_index": 4 , f"name": "t"          , "coupling_type": un   , "link": None  , "lb": 0.1   , "ub": 4.0     
# "sp_index": 4 , f"name": "t"          , "coupling_type": un   , "link": None  , "lb": 0.1   , "ub": 4.0     
# "sp_index": 4 , f"name": "ts"         , "coupling_type": un   , "link": None  , "lb": 0.1   , "ub": 9.0     
# "sp_index": 4 , f"name": "ts"         , "coupling_type": un   , "link": None  , "lb": 0.1   , "ub": 9.0     
# "sp_index": 4 , f"name": "ts"         , "coupling_type": un   , "link": None  , "lb": 0.1   , "ub": 9.0     
# "sp_index": 4 , f"name": "ts"         , "coupling_type": un   , "link": None  , "lb": 0.1   , "ub": 9.0     
# "sp_index": 4 , f"name": "ts"         , "coupling_type": un   , "link": None  , "lb": 0.1   , "ub": 9.0     
# "sp_index": 4 , f"name": "ts"         , "coupling_type": un   , "link": None  , "lb": 0.1   , "ub": 9.0     
# "sp_index": 4 , f"name": "ts"         , "coupling_type": un   , "link": None  , "lb": 0.1   , "ub": 9.0     
# "sp_index": 4 , f"name": "ts"         , "coupling_type": un   , "link": None  , "lb": 0.1   , "ub": 9.0     
# "sp_index": 4 , f"name": "ts"         , "coupling_type": un   , "link": None  , "lb": 0.1   , "ub": 9.0     


#######################
#Inputs
h = 55000
Mach = 1.4

tc = 0.05 #Thickness/Chord 0.01<=0.05<=0.1 []
ARw = 3.0 #Wing aspect ratio 2.5<=3.0<=8.0 []; orig paper BL=5.5
LAMBDAw = 60. #Wing sweep angle 40<=60<=70 [deg]
Sref = 500 #Wing surface area 200<=500<=800 [ft2]; orig paper BL=1000
Sht = 100
ARht = 5.5
lambdatr = 0.3 #taper ratio lambda 0.1<=0.3<=0.4

Lift = 25000 #Lift

ti1  = 3
ti2  = 3
ti3  = 3
ti4  = 3
ti5  = 3
ti6  = 3
ti7  = 3
ti8  = 3
ti9  = 3
tsi1 = 6
tsi2 = 6
tsi3 = 6
tsi4 = 6
tsi5 = 6
tsi6 = 6
tsi7 = 6
tsi8 = 6
tsi9 = 6

#Constants
C = [500.0, 16000.0,  4.0,  4360.0,  0.01375,  1.0]
G=4000000*144
E=10600000*144
nu=0.3
rho_alum=0.1*144
rho_core=0.1*144/10
rho_fuel=6.5*7.4805
Fw_at_t=5
k=6.09375

#Locals
Z = [tc,h,Mach,ARw,LAMBDAw,Sref,Sht,ARht] # these are the shared variables
LAMBDA = lambdatr
L = Lift

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

def Wing_Mod(Z, LAMBDA):
  c = [0, 0, 0, 0]
  x = [0]*8
  y = [0] * 8
  b=max(2,np.real(np.sqrt(Z[3]*Z[5])))
  c[0]=2*Z[5]/((1+LAMBDA)*b)
  c[3]=LAMBDA*c[0]
  x[0]=0
  y[0]=0
  x[1]=c[0]
  y[1]=0
  x[6]=(b/2)*np.tan(Z[4]*np.pi/180)
  y[6]=b/2
  x[7]=x[6]+c[3]
  y[7]=b/2
  y[2]=b/6
  x[2]=(x[6]/y[6])*y[2]
  y[4]=b/3
  x[4]=(x[6]/y[6])*y[4]
  x[5]=x[7]+((x[1]-x[7])/y[7])*(y[7]-y[4])
  y[5]=y[4]
  x[3]=x[7]+((x[1]-x[7])/y[7])*(y[7]-y[2])
  y[3]=y[2]
  c[1]=x[3]-x[2]
  c[2]=x[5]-x[4]
  TE_sweep=(np.arctan((x[7]-x[1])/y[7]))*180/np.pi
  Sweep_40=(np.arctan(((x[7]-0.6*(x[7]-x[6]))-0.4*x[1])/y[7]))*180/np.pi

  l=np.multiply([c[i] for i in range(3)], .4*np.cos(Z[4]*np.pi/180))
  k=np.multiply([c[i] for i in range(3)], .6*np.sin((90-TE_sweep)*np.pi/180)/np.sin((90+TE_sweep-Z[4])*np.pi/180))
  c_box=np.add(l, k)
  D_mx=np.subtract(l, np.multiply(0.407, c_box))

  return c,c_box,Sweep_40,D_mx,b,l

def loads(b,c,Sweep_40,D_mx,L,Izz,Z,E):
  NP=9 #   %----number of panels per halfspan
  n=90
  rn = int(n/NP)
    
  h=(b/2)/n
  x=np.linspace(0,b/2-h, n)
  x1=np.linspace(h, b/2, n)

  #%----Calculate Mx, Mz, and P----%
  l=np.linspace(0, (b/2)-(b/2)/NP, NP)

  c1mc4 = c[0]-c[3]
  f_all  =np.multiply((3*b/10), np.sqrt(np.subtract(1, ( np.power(x, 2))/(np.power(np.divide(b,2),2)))))
  f1_all =np.multiply((3*b/10), np.sqrt(np.subtract(1, ( np.power(x1, 2))/(np.power(np.divide(b,2),2)))))
  C= c[3] + 2*( (b/2- x)/b )*c1mc4
  C1=c[3] + 2*( (b/2-x1)/b )*c1mc4
  A_Tot: np.ndarray =np.multiply((h/4)*(C+C1), (np.add(f_all, f1_all)))
  Area = np.sum(A_Tot.reshape((NP,rn)), axis=1)
  Spanel = np.multiply((h*(rn)/2), (np.add([C[int(i)] for i in np.linspace(0, n-10, 9)], [C[int(i)] for i in np.linspace(9, n-1, 9)])))

  # % cos, tan, and cos**-1 of Sweep
  cosSweep = np.cos(Sweep_40*np.pi/180)
  cosInvSweep = 1/cosSweep
  tanCos2Sweep = np.tan(Sweep_40*np.pi/180)*cosSweep*cosSweep
  
  p=np.divide(L*Area, sum(Area))

  # % Replace T by:
  Tcsp = np.cumsum(p)
  Tsp = Tcsp[-1]
  temp = [0]
  temp = temp +[Tcsp[i] for i in range(len(Tcsp)-1)]
  T = np.subtract(Tsp, temp)
  pl = np.multiply(p, l)
  Tcspl = np.cumsum(pl)
  Tspl = Tcspl[-1]
  Mb = np.multiply(np.subtract( np.subtract(Tspl, Tcspl), np.multiply(l, np.subtract(Tsp, Tcsp)) ),cosInvSweep)

  P=[T[int(i)] for i in np.arange(0,NP-1, int(NP/3))]
  Mx=np.multiply(P, D_mx)
  Mz=[Mb[int(i)] for i in np.arange(0,NP-1, int(NP/3))]

  # %----Calculate Wing Twist due to Bending----%
  I = np.zeros((NP))
  chord=c[3]+ (np.divide(2*(b/2-l), b))*c1mc4
  y = np.zeros((2,9))
  y[0,:]=(l-.4*chord*tanCos2Sweep)*cosInvSweep
  y[1,:]=(l+.6*chord*tanCos2Sweep)*cosInvSweep
  y[1,0]=0
  I[0:int(NP/3)]=np.sqrt((Izz[0]**2+Izz[1]**2)/2)
  I[int(NP/3):int(2*NP/3)]=np.sqrt((Izz[1]**2+Izz[2]**2)/2)
  I[int(2*NP/3):int(NP)]=np.sqrt((Izz[2]**2)/2)

  La=y[0,1:NP]-y[0,0:NP-1]
  La = np.append(0, La)
  Lb=y[1,1:NP]-y[1,0:NP-1]
  Lb=np.append(0, Lb)
  A=T*La**3/(3*E*I)+Mb*La**2./(2*E*I)
  B=T*Lb**3/(3*E*I)+Mb*Lb**2./(2*E*I)
  Slope_A=T*La**2./(2*E*I)+Mb*La/(E*I)
  Slope_B=T*Lb**2./(2*E*I)+Mb*Lb/(E*I)

  for i in range(NP-1):
    Slope_A[i+1]=Slope_A[i]+Slope_A[i+1]
    Slope_B[i+1]=Slope_B[i]+Slope_B[i+1]
    A[i+1]=A[i]+Slope_A[i]*La[i+1]+A[i+1]
    B[i+1]=B[i]+Slope_B[i]*Lb[i+1]+B[i+1]

  bend_twist=((B-A)/chord) *180/np.pi
  
  # print("bend_twist = ",bend_twist) #McH_here is something going wrong, why do we have such big bending?

  for i in range(1, len(bend_twist)):
    if bend_twist[i]<bend_twist[i-1]:
        bend_twist[i]=bend_twist[i-1]
  
  return P,Mz,Mx,bend_twist,Spanel


#Calculations
t= np.divide([ti1, ti2, ti3, ti4, ti5, ti6, ti7, ti8, ti9], 12.)#convert to feet
ts= np.divide([tsi1, tsi2, tsi3, tsi4, tsi5, tsi6, tsi7, tsi8, tsi9], 12.);#convert to feet

t1= [t[i] for i in range(3)] 
t2= [t[i] for i in range(3, 6)]  
t3= [t[i] for i in range(6, 9)]  
ts1=[ts[i] for i in range(3)] 
ts2=[ts[i] for i in range(3, 6)]  
ts3=[ts[i] for i in range(6, 9)] 

beta=0.9
# c,c_box,Sweep_40,D_mx,b,a = Wing_Mod(Z,LAMBDA)
[c,c_box,Sweep_40,D_mx,b,a]=Wing_Mod(Z,LAMBDA)

l=0.6*c_box
h=(np.multiply([c[i] for i in range(3)], beta*float(Z[0])))-np.multiply((0.5),np.add(ts1,ts3))
A_top=(np.multiply(t1, 0.5*l))+(np.multiply(t2, h/6))
A_bottom=(np.multiply(t3, 0.5*l))+(np.multiply(t2, h/6))
Y_bar=np.multiply(h, np.divide((2*A_top), (2*A_top+2*A_bottom)))
Izz=np.multiply(2, np.multiply(A_top, np.power((h-Y_bar), 2)))+np.multiply(2, np.multiply(A_bottom,np.power((-Y_bar), 2)))
[P,Mz,Mx,bend_twist,Spanel]=loads(b,c,Sweep_40,D_mx,L,Izz,Z,E)

Phi=(Mx/(4*G*(l*h)**2))*(l/t1+2*h/t2+l/t3)
aa=len(bend_twist)
twist = np.array([0] * aa)
twist[0:int(aa/3)]=bend_twist[0:int(aa/3)]+Phi[0]*180/np.pi
twist[int(aa/3):int(aa*2/3)]=bend_twist[int(aa/3):int(aa*2/3)]+Phi[1]*180/np.pi
twist[int(aa*2/3):aa]=bend_twist[int(aa*2/3):aa]+Phi[2]*180/np.pi
deltaL_divby_q=np.sum(twist*Spanel*0.1*2)
print("twist = ", twist) #twist is insanely huge
print("Spanel = ",Spanel)

#   # %-----THIS SECTION COMPUTES THE TOTAL WEIGHT OF A/C-----%
Wtop_alum=(b/4)*(c[0]+c[3])*np.mean(t1)*rho_alum
Wbottom_alum=(b/4)*(c[0]+c[3])*np.mean(t3)*rho_alum
Wside_alum=(b/2)*np.mean(h)*np.mean(t2)*rho_alum
Wtop_core=(b/4)*(c[0]+c[3])*np.mean(np.subtract(ts1, t1))*rho_core
Wbottom_core=(b/4)*(c[0]+c[3])*np.mean(np.subtract(ts3,t3))*rho_core
Wside_core=(b/2)*np.mean(h)*np.mean(np.subtract(ts2,t2))*rho_core
W_wingstruct=Wtop_alum+Wbottom_alum+Wside_alum+Wtop_core+Wbottom_core+Wside_core
W_fuel_wing=np.mean(h*l)*(b/3)*(2)*rho_fuel
Bh=np.sqrt(Z[7]*Z[6])
W_ht=3.316*((1+(Fw_at_t/Bh))**-2.0)*((L*C[2]/1000)**0.260)*(Z[6]**0.806)

#   return [Ws, Wf, theta] # these are responses
Wf = C[0] + W_fuel_wing
Ws = C[1] + W_ht + 2*W_wingstruct
theta = deltaL_divby_q

#%-----Constraints-----%
# [c,c_box,Sweep_40,D_mx,b,a]=Wing_Mod(Z,LAMBDA)

t1=t[0:3] 
t2=t[3:6]
t3=t[6:9]
ts1=ts[0:3]
ts2=ts[3:6]
ts3=ts[6:9]

teq1=((t1**3)/4+(3*t1)*(ts1-t1/2)**2)**(1/3)
teq2=((t2**3)/4+(3*t2)*(ts2-t2/2)**2)**(1/3)
teq3=((t3**3)/4+(3*t3)*(ts3-t3/2)**2)**(1/3)

sig_1=Mz*(0.95*h-Y_bar)/Izz
sig_2=Mz*(h-Y_bar)/Izz
sig_3=sig_1
sig_4=Mz*(0.05*h-Y_bar)/Izz
sig_5=Mz*(-Y_bar)/Izz
sig_6=sig_4
q=Mx/(2*l*h)

# %----Point 1----%
T1=P*(l-a)/l
tau1_T=T1/(h*t2)
tau1=q/t2+tau1_T
sig_eq1=np.sqrt(sig_1**2+3*tau1**2)
sig_cr1=((np.pi**2)*E*4/(12*(1-nu**2)))*(teq2/(0.95*h))**2
tau_cr1=((np.pi**2)*E*5.5/(12*(1-nu**2)))*(teq2/(0.95*h))**2
G = np.zeros((72))
G[0:3]=k*sig_eq1
G[3:6]=k*(((sig_1)/sig_cr1)+(tau1/tau_cr1)**2)
G[6:9]=k*((-(sig_1)/sig_cr1)+(tau1/tau_cr1)**2)


# %----Point 2----%
tau2=q/t1
sig_eq2=np.sqrt(sig_2**2+3*tau2**2)
sig_cr2=((np.pi**2)*E*4/(12*(1-nu**2)))*(teq1/l)**2
tau_cr2=((np.pi**2)*E*5.5/(12*(1-nu**2)))*(teq1/l)**2
G[9:12]=k*sig_eq2
G[12:15]=k*(((sig_2)/sig_cr2)+(tau2/tau_cr2)**2)
G[15:18]=k*((-(sig_2)/sig_cr2)+(tau2/tau_cr2)**2)

# %----Point 3----%
T2=P*a/l
tau3_T=-T2/(h*t2)
tau3=q/t2+tau3_T
sig_eq3=np.sqrt(sig_3**2+3*tau3**2)
sig_cr3=sig_cr1
tau_cr3=tau_cr1
G[18:21]=k*sig_eq3
G[21:24]=k*(((sig_3)/sig_cr3)+(tau3/tau_cr3)**2)
G[24:27]=k*(((-sig_3)/sig_cr3)+(tau3/tau_cr3)**2)

# %----Point 4----%
tau4=-q/t2+tau1_T
sig_eq4=np.sqrt(sig_4**2+3*tau4**2)
G[27:30]=k*sig_eq4

# %----Point 5----%
tau5=q/t3
sig_eq5=np.sqrt(sig_5**2+3*tau5**2)
sig_cr5=((np.pi**2)*E*4/(12*(1-nu**2)))*(teq3/l)**2
tau_cr5=((np.pi**2)*E*5.5/(12*(1-nu**2)))*(teq3/l)**2
G[30:33]=k*sig_eq5
G[33:36]=k*(((sig_5)/sig_cr5)+(tau5/tau_cr5)**2)
G[36:39]=k*(((-sig_5)/sig_cr5)+(tau5/tau_cr5)**2)

# %----Point 6----%
tau6=-q/t2+tau3_T
sig_eq6=np.sqrt(sig_6**2+3*tau6**2)
G[39:42]=k*sig_eq6

#Constraints
Sig_C=65000*144
Sig_T=65000*144

G1 = np.zeros((72))

G1[0:3]=((G[0:3])/Sig_C)-1
G1[54:57]=-(G[0:3])/Sig_C-1
G1[3:9]=G[3:9]-1

G1[9:12]=(G[9:12])/Sig_C-1
G1[57:60]=-(G[9:12])/Sig_C-1
G1[12:18]=G[12:18]-1

G1[18:21]=(G[18:21])/Sig_C-1
G1[60:63]=-(G[18:21])/Sig_C-1
G1[21:27]=G[21:27]-1

G1[27:30]=(G[27:30])/Sig_T-1
G1[63:66]=-(G[27:30])/Sig_T-1

G1[30:33]=(G[30:33])/Sig_T-1
G1[66:69]=-(G[30:33])/Sig_T-1
G1[33:39]=G[33:39]-1

G1[39:42]=(G[39:42])/Sig_T-1
G1[69:72]=-(G[39:42])/Sig_T-1

G1[42:45]=(1/2)*(ts1+ts3)/h-1
G1[45:48]=t1/(ts1-.1*t1)-1
G1[48:51]=t2/(ts2-.1*t2)-1
G1[51:54]=t3/(ts3-.1*t3)-1

# return[0, constraints] # you have 72 constraints!!

#Printing
print("t = ", t)
print("ts = ", ts)
# print("x = ", x)
# print("bhalf = ", bhalf)
# print("Ratio = ", Ratio)
print("Lift = ", Lift)
print("Theta = ", theta)
# print("Fol =", Fol)
print("Ws =", Ws)
# print("WFw =", WFw)
print("Wf =", Wf)
# print("Wt =", Wt)
# print("sig1 =", sig1)
# print("sig2 =", sig2)
# print("sig3 =", sig3)
# print("sig4 =", sig4)
# print("sig5 =", sig5)
# print("sig6 =", sig6)