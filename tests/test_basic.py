import os
from DMDO import *
import math
import numpy as np
from numpy import cos, exp, pi, prod, sin, sqrt, subtract, inf
import copy
from typing import List, Dict, Any, Callable, Protocol, Optional

user =USER

def termTest():
  print("Termination criterria work!")

def A1(x):
  LAMBDA = 0.0
  for i in range(len(x)):
    if x[i] == 0.:
      x[i] = 1e-12
  return math.log(x[0]+LAMBDA) + math.log(x[1]+LAMBDA) + math.log(x[2]+LAMBDA)

def A2(x):
  LAMBDA = 0.0
  for i in range(len(x)):
    if x[i] == 0.:
      x[i] = 1e-12
  return np.divide(1., (x[0]+LAMBDA)) + np.divide(1., (x[1]+LAMBDA)) + np.divide(1., (x[2]+LAMBDA))

def opt1(x, y):
  return [sum(x)+y[0], [0.]]

def opt2(x, y):
  return [0., [x[1]+y[0]-10.]]

def SR_A1(x):
  """ Speed reducer A1 """
  return (0.7854*x[0]*x[1]**2*(3.3333*x[2]*x[2] + 14.9335*x[2] - 43.0934))

def SR_A2(x):
  """ Speed reducer A2 """
  return (-1.5079*x[0]*x[4]**2) + (7.477 * x[4]**3) + 0.7854 * x[3] * x[4]**2

def SR_A3(x):
  """ Speed reducer A3 """
  return (-1.5079*x[0]*x[4]**2 + 7.477*x[4]**3 + 0.7854*x[3]*x[4]**2)

def SR_opt1(x, y):
  g5 = 27/(x[0]*x[1]**2*x[2]) -1
  g6 = 397.5/(x[0]*x[1]**2*x[2]**2) -1
  g9 = x[1]*x[2]/40 -1
  g10 = 5*x[1]/x[0] -1
  g11 = x[0]/(12*x[1]) -1
  return [y[0], [g5,g6,g9,g10,g11]]

def SR_opt2(x, y):
  g1 = sqrt( ((745*x[3])/(x[1]*x[2]))**2 + 1.69e+7)/(110*x[4]**3) -1
  g3 = (1.5*x[4] + 1.9)/x[3] -1
  g7 = 1.93*x[3]**3/(x[1]*x[2]*x[4]**4) -1
  return [y, [g1,g3,g7]]

def SR_opt3(x, y):
  g2 = sqrt( ((745*x[3])/(x[1]*x[2]))**2 + 1.575e+8)/(85*x[4]**3) -1
  g4 = (1.1*x[4] + 1.9)/x[3] -1
  g8 = (1.93*x[3]**3)/(x[1]*x[2]*x[4]**4) -1
  return [y, [g2, g4, g8]]

def GP_A1(z):
  z1 = sqrt(z[0]**2 + z[1]**-2 + z[2]**2)
  z2 = sqrt(z[2]**2 + z[3]**2  + z[4]**2)
  return z1**2 + z2**2

def GP_opt1(z,y):
  if isinstance(y, list) and len(y) > 0:
    return [y[0], [z[0]**-2 + z[1]**2 - z[2]**2, z[2]**2 + z[3]**-2  - z[4]**2]]
  else:
    return [y, [z[0]**-2 + z[1]**2 - z[2]**2, z[2]**2 + z[3]**-2  - z[4]**2]]

def GP_A2(z):
  z3 = sqrt(z[0]**2 + z[1]**-2 + z[2]**-2 + z[3]**2)
  return z3

def GP_opt2(z,y):
  return [0, [z[0]**2 + z[1]**2 - z[3]**2, z[0]**-2 + z[2]**2 - z[3]**2]]

def GP_A3(z):
  z6 = sqrt(z[0]**2 + z[1]**2 + z[2]**2 +z[3] **2)
  return z6

def GP_opt3(z, y):
  return [0, [z[0]**2 + z[1]**-2 - z[2]**2, z[0]**2 +z[1]**2 - z[3]**2]]

def SBJ_A1(x):
  # SBJ_obj_range
  return x[1]+x[3]+x[4]

def SBJ_opt1(x, y):
  # SBJ_constraint_range
  if user.h <36089:
    theta = 1-0.000006875*user.h
  else:
    theta = 0.7519
  r = user.M * x[2] * 661 * np.sqrt(theta/x[0])*math.log(y[0]/(y[0]-x[4]))
  G = -r/2000 +1

  return [y[0], [G]]

def SBJ_A2(x):
  # SBJ_obj_power
  #  THIS SECTION COMPUTES SFC, ESF, AND ENGINE WEIGHT
  C=[500.0, 16000.0, 4.0 , 4360.0,  0.01375,  1.0]
  Thrust = x[0]
  Dim_Throttle = x[1]*16168
  s=[1.13238425638512, 1.53436586044561, -0.00003295564466, -0.00016378694115, -0.31623315541888, 0.00000410691343, -0.00005248000590, -0.00000000008574, 0.00000000190214, 0.00000001059951]
  SFC = s[0]+s[1]*user.M+s[2]*user.h+s[3]*Dim_Throttle+s[4]*user.M**2+2*user.h*user.M*s[5]+2*Dim_Throttle*user.M*s[6]+s[7]*user.h**2+2*Dim_Throttle*user.h*s[8]+s[9]*Dim_Throttle**2

  ESF = (Thrust/2)/Dim_Throttle

  We = C[3]*(ESF**1.05)*2

  return [We,SFC, ESF]

def SBJ_opt2(x, y):
  # SBJ_constraint_power
  # -----THIS SECTION COMPUTES POLYNOMIAL CONSTRAINT FUNCTIONS-----
  Dim_Throttle = x[1]*16168
  S_initial1=[user.M,user.h,x[0]]
  S1=[user.M,user.h,x[1]]
  flag1 = [2,4,2]
  bound1 = [.25,.25,.25]
  Temp_uA=1.02
  g1 = polyApprox(S_initial1, S1, flag1, bound1)
  g1 = g1 /Temp_uA-1
  p=[11483.7822254806, 10856.2163466548, -0.5080237941, 3200.157926969, -0.1466251679, 0.0000068572]
  Throttle_uA=p[0]+p[1]*user.M+p[2]*user.h+p[3]*user.M**2+2*p[4]*user.M*user.h+p[5]*user.h**2

  return[0, [g1, Dim_Throttle/Throttle_uA-1]]

def SBJ_A3(x):
  # SBJ_obj_dragpolar
  # %----Inputs----%
  Z = [x[3],user.h,user.M,x[4],x[5],x[6],x[7],x[8]]
  C = [500.0, 16000.0,  4.0,  4360.0,  0.01375,  1.0]
  Z = [1,	55000,	1.40000000000000,	1,	1,	1,	1,	1]
  ARht=Z[7]
  S_ht=Z[6]    
  Nh=C[5]

  # %-----Drag computations----%

  if Z[1]<36089:
     V = Z[2]*(1116.39*sqrt(1-(6.875e-06*Z[1])))
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
  c= x[10] #Lw
  d=(x[11])*Nh*(S_ht/Z[5])
  A= np.array([[a, b], [c, d]])
  # ---- Modified by S. Tosserams:
  # ---- scale coefficient Wt for proper conditioning of matrix A 
  B=np.array([x[1]/1e5, 0])
  # -----------------------
  try:
    CLo=np.linalg.solve(A, B)
  except:
    CLo = np.array([-np.inf, np.inf])
  delta_L=x[2]*q
  Lw1=CLo[0]*q*Z[5]-delta_L
  CLw1=Lw1/(q*Z[5])
  CLht1=-CLw1*c/d
  #  Modified by S. Tosserams:
  #  scale first coefficient of D for proper conditioning of matrix A 
  D=np.array([(x[1]-CLw1*a-CLht1*b)/1e5, -CLw1*c-CLht1*d])
  # -----------------
  try:
    DCL = np.linalg.solve(A, D)
  except:
    DCL = np.array([np.nan,np.nan])

  if Z[2] >= 1:
    kw = Z[3] * (Z[2]**2-1) * np.cos(Z[4]*np.pi/180)/(4*Z[3]*np.sqrt(Z[2]**2-1)-2)
    kht = ARht * (Z[2]**2-1)*np.cos(x[9]*pi/180)/(4*ARht*np.sqrt(Z[2]**2-1)-2)
  else:
    kw = 1/(np.pi*0.8*Z[3])
    kht= 1/(np.pi*0.8*ARht)
  
  S_initial1 = copy.deepcopy(x[0])
  S1 = copy.deepcopy(x[0])
  flag1 = 1
  bound1 = 0.25
  Fo1 = polyApprox(S_initial1 if isinstance(S_initial1, list) else [S_initial1], S1 if isinstance(S1, list) else [S1], flag1 if isinstance(flag1, list) else [flag1], bound1 if isinstance(bound1, list) else [bound1])


  CDmin = C[4]*Fo1 + 3.05*(Z[0]**(5/3))*((cos(Z[4]*np.pi/180))**(3/2))

  CDw=CDmin+kw*(CLo[0]**2)+kw*(DCL[0]**2)
  CDht=kht*(CLo[1]**2)+kht*(DCL[1]**2)
  CD=CDw+CDht
  CL=CLo[0]+CLo[1]
  L = x[1]
  D = q*CDw*Z[5]+q*CDht*Z[6]
  LD = CL/CD

  return [D, LD, L]

def SBJ_opt3(x, y):
  # SBJ_constraint_dragpolar
  # %----Inputs----%
  Z = [x[3],user.h,user.M,x[4],x[5],x[6],x[7],x[9]]
  C = [500.0, 16000.0,  4.0,  4360.0,  0.01375,  1.0]
  S_ht=Z[6]
  Nh=C[5]

  # %-----Drag computations----%

  if Z[1]<36089:
     V = Z[2]*(1116.39*sqrt(1-(6.875e-06*Z[1])))
     rho = (2.377e-03)*(1-(6.875e-06*Z[1]))**4.2561
  else:
     V = Z[2]*968.1
     rho = (2.377e-03)*(.2971)*exp(-(Z[1]-36089)/20806.7)
  q=.5*rho*(V**2)

  # ### Modified by S. Tosserams:
  # # scale coefficients for proper conditioning of matrix A 
  a=q*Z[5]/1e5
  b=Nh*q*S_ht/1e5
  # -------------------------------
  c= x[10] #Lw
  d=(x[11])*Nh*(S_ht/Z[5])
  A= np.array([[a, b], [c, d]])
  # ---- Modified by S. Tosserams:
  # ---- scale coefficient Wt for proper conditioning of matrix A 
  B=[x[1]/1e5, 0]
  # -----------------------
  try:
    CLo=np.linalg.solve(A, B)
  except:
    CLo = np.array([-np.inf, np.inf])
  
  S_initial2 = copy.deepcopy(x[3])
  S2 = copy.deepcopy(Z[0])
  flag1 = [1]
  bound1 = [0.25]
  
  g1 = polyApprox(S_initial2 if isinstance(S_initial2, list) else [S_initial2], S2 if isinstance(S2, list) else [S2], flag1 if isinstance(flag1, list) else [flag1], bound1 if isinstance(bound1, list) else [bound1])


  # %------Constraints------%

  Pg_uA=1.1
  g1=g1/Pg_uA-1
  if CLo[0] > 0:
      g2=(2*(CLo[1]))-(CLo[0])
      g3=(2*(-CLo[1]))-(CLo[0])
  else:
      g2=(2*(-CLo[1]))-(CLo[0])
      g3=(2*(CLo[1]))-(CLo[0])

  return [0, [g1, g2, g3]]

def SBJ_A4(x):
  # SBJ_obj_weight
  Z= [x[1], user.h, user.M, x[2], x[3], x[4], x[5], x[6]]
  L = x[0]
  # %----Inputs----%
  t= np.divide(x[7:16], 12.)#convert to feet
  ts= np.divide(x[16:], 12.);#convert to feet
  LAMBDA = x[10]
  C=[500.0, 16000.0,  4.0,  4360.0,  0.01375,  1.0]

  t1= [t[i] for i in range(3)] 
  t2= [t[i] for i in range(3, 6)]  
  t3= [t[i] for i in range(6, 9)]  
  ts1=[ts[i] for i in range(3)] 
  ts2=[ts[i] for i in range(3, 6)]  
  ts3=[ts[i] for i in range(6, 9)]  
  G=4000000*144
  E=10600000*144
  rho_alum=0.1*144
  rho_core=0.1*144/10
  rho_fuel=6.5*7.4805
  Fw_at_t=5

  # %----Inputs----%
  beta=.9
  c,c_box,Sweep_40,D_mx,b,_ = Wing_Mod(Z,LAMBDA)
  l=0.6*c_box
  h=(np.multiply([c[i] for i in range(3)], beta*float(Z[0])))-np.multiply((0.5),np.add(ts1,ts3))
  A_top=(np.multiply(t1, 0.5*l))+(np.multiply(t2, h/6))
  A_bottom=(np.multiply(t3, 0.5*l))+(np.multiply(t2, h/6))
  Y_bar=np.multiply(h, np.divide((2*A_top), (2*A_top+2*A_bottom)))
  Izz=np.multiply(2, np.multiply(A_top, np.power((h-Y_bar), 2)))+np.multiply(2, np.multiply(A_bottom,np.power((-Y_bar), 2)))
  P,Mz,Mx,bend_twist,Spanel=loads(b,c,Sweep_40,D_mx,L,Izz,Z,E)

  Phi=(Mx/(4*G*(l*h)**2))*(l/t1+2*h/t2+l/t3)
  aa=len(bend_twist)
  twist = np.array([0] * aa)
  twist[0:int(aa/3)]=bend_twist[0:int(aa/3)]+Phi[0]*180/pi
  twist[int(aa/3):int(aa*2/3)]=bend_twist[int(aa/3):int(aa*2/3)]+Phi[1]*180/pi
  twist[int(aa*2/3):aa]=bend_twist[int(aa*2/3):aa]+Phi[2]*180/pi
  deltaL_divby_q=sum(twist*Spanel*0.1*2)


  # %-----THIS SECTION COMPUTES THE TOTAL WEIGHT OF A/C-----%

  Wtop_alum=(b/4)*(c[0]+c[3])*np.mean(t1)*rho_alum
  Wbottom_alum=(b/4)*(c[0]+c[3])*np.mean(t3)*rho_alum
  Wside_alum=(b/2)*np.mean(h)*np.mean(t2)*rho_alum
  Wtop_core=(b/4)*(c[0]+c[3])*np.mean(np.subtract(ts1, t1))*rho_core
  Wbottom_core=(b/4)*(c[0]+c[3])*np.mean(np.subtract(ts3,t3))*rho_core
  Wside_core=(b/2)*np.mean(h)*np.mean(np.subtract(ts2,t2))*rho_core
  W_wingstruct=Wtop_alum+Wbottom_alum+Wside_alum+Wtop_core+Wbottom_core+Wside_core
  W_fuel_wing=np.mean(h*l)*(b/3)*(2)*rho_fuel
  Bh=sqrt(Z[7]*Z[6])
  W_ht=3.316*((1+(Fw_at_t/Bh))**-2.0)*((L*C[2]/1000)**0.260)*(Z[6]**0.806)
  Wf = C[0] + W_fuel_wing
  Ws = C[1] + W_ht + 2*W_wingstruct
  theta = deltaL_divby_q

  return [Ws, Wf, theta]

def Wing_Mod(Z, LAMBDA):
  c = [0, 0, 0, 0]
  x = [0]*8
  y = [0] * 8
  b=max(2,np.real(sqrt(Z[3]*Z[5])))
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
  k=np.multiply([c[i] for i in range(3)], .6*np.sin((90-TE_sweep)*np.pi/180)/sin((90+TE_sweep-Z[4])*np.pi/180))
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
  cosSweep = np.cos(Sweep_40*pi/180)
  cosInvSweep = 1/cosSweep
  tanCos2Sweep = np.tan(Sweep_40*pi/180)*cosSweep*cosSweep



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
  I[0:int(NP/3)]=sqrt((Izz[0]**2+Izz[1]**2)/2)
  I[int(NP/3):int(2*NP/3)]=sqrt((Izz[1]**2+Izz[2]**2)/2)
  I[int(2*NP/3):int(NP)]=sqrt((Izz[2]**2)/2)

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

  bend_twist=((B-A)/chord)*180/pi
  for i in range(1, len(bend_twist)):
    if bend_twist[i]<bend_twist[i-1]:
        bend_twist[i]=bend_twist[i-1]

  return P,Mz,Mx,bend_twist,Spanel

def SBJ_opt4(x, y):
  # SBJ_constraint_weight
  # Z = [tc,h,M,ARw,LAMBDAw,Sref,Sht,ARht]
  Z= [x[1], user.h, user.M, x[2], x[3], x[4], x[5], x[6]]
  t= np.divide(x[7:16], 12.)#convert to feet
  ts= np.divide(x[16:], 12.);#convert to feet
  LAMBDA= x[10]
  L=x[0]

  # %----Inputs----%
  # if isinstance(t, np.ndarray):
  #   t=t/12 #convert to feet
  #   ts=ts/12 #convert to feet
  # else:
  #   t=np.array(t)/12 #convert to feet
  #   ts=np.array(ts)/12 #convert to feet

  t1=t[0:3] 
  t2=t[3:6]
  t3=t[6:9]
  ts1=ts[0:3]
  ts2=ts[3:6]
  ts3=ts[6:9]
  G=4000000*144
  E=10600000*144
  nu=0.3

  # %----Inputs----%

  beta=.9
  [c,c_box,Sweep_40,D_mx,b,a]=Wing_Mod(Z,LAMBDA)
  teq1=((t1**3)/4+(3*t1)*(ts1-t1/2)**2)**(1/3)
  teq2=((t2**3)/4+(3*t2)*(ts2-t2/2)**2)**(1/3)
  teq3=((t3**3)/4+(3*t3)*(ts3-t3/2)**2)**(1/3)
  l=0.6*c_box
  h=Z[0]*(beta*np.array(c[0:3]))-(0.5)*(ts1+ts3)
  A_top=(0.5)*(t1*l)+(1/6)*(t2*h)
  A_bottom=(0.5)*(t3*l)+(1/6)*(t2*h)
  Y_bar=h*(2*A_top)/(2*A_top+2*A_bottom)
  Izz=2*A_top*(h-Y_bar)**2+2*A_bottom*(-Y_bar)**2
  P,Mz,Mx,_,_ =loads(b,c,Sweep_40,D_mx,L,Izz,Z,E)

  sig_1=Mz*(0.95*h-Y_bar)/Izz
  sig_2=Mz*(h-Y_bar)/Izz
  sig_3=sig_1
  sig_4=Mz*(0.05*h-Y_bar)/Izz
  sig_5=Mz*(-Y_bar)/Izz
  sig_6=sig_4
  q=Mx/(2*l*h)

  # %-----THIS SECTION COMPUTES THE TOTAL WEIGHT OF A/C-----%

  k=6.09375


  # %----Point 1----%
  T1=P*(l-a)/l
  tau1_T=T1/(h*t2)
  tau1=q/t2+tau1_T
  sig_eq1=sqrt(sig_1**2+3*tau1**2)
  sig_cr1=((pi**2)*E*4/(12*(1-nu**2)))*(teq2/(0.95*h))**2
  tau_cr1=((pi**2)*E*5.5/(12*(1-nu**2)))*(teq2/(0.95*h))**2
  G = np.zeros((72))
  G[0:3]=k*sig_eq1
  G[3:6]=k*(((sig_1)/sig_cr1)+(tau1/tau_cr1)**2)
  G[6:9]=k*((-(sig_1)/sig_cr1)+(tau1/tau_cr1)**2)


  # %----Point 2----%

  tau2=q/t1
  sig_eq2=sqrt(sig_2**2+3*tau2**2)
  sig_cr2=((pi**2)*E*4/(12*(1-nu**2)))*(teq1/l)**2
  tau_cr2=((pi**2)*E*5.5/(12*(1-nu**2)))*(teq1/l)**2
  G[9:12]=k*sig_eq2
  G[12:15]=k*(((sig_2)/sig_cr2)+(tau2/tau_cr2)**2)
  G[15:18]=k*((-(sig_2)/sig_cr2)+(tau2/tau_cr2)**2)

  # %----Point 3----%

  T2=P*a/l
  tau3_T=-T2/(h*t2)
  tau3=q/t2+tau3_T
  sig_eq3=sqrt(sig_3**2+3*tau3**2)
  sig_cr3=sig_cr1
  tau_cr3=tau_cr1
  G[18:21]=k*sig_eq3
  G[21:24]=k*(((sig_3)/sig_cr3)+(tau3/tau_cr3)**2)
  G[24:27]=k*(((-sig_3)/sig_cr3)+(tau3/tau_cr3)**2)

  # %----Point 4----%

  tau4=-q/t2+tau1_T
  sig_eq4=sqrt(sig_4**2+3*tau4**2)
  G[27:30]=k*sig_eq4

  # %----Point 5----%

  tau5=q/t3
  sig_eq5=sqrt(sig_5**2+3*tau5**2)
  sig_cr5=((pi**2)*E*4/(12*(1-nu**2)))*(teq3/l)**2
  tau_cr5=((pi**2)*E*5.5/(12*(1-nu**2)))*(teq3/l)**2
  G[30:33]=k*sig_eq5
  G[33:36]=k*(((sig_5)/sig_cr5)+(tau5/tau_cr5)**2)
  G[36:39]=k*(((-sig_5)/sig_cr5)+(tau5/tau_cr5)**2)

  # %----Point 6----%

  tau6=-q/t2+tau3_T
  sig_eq6=sqrt(sig_6**2+3*tau6**2)
  G[39:42]=k*sig_eq6

  # %-----Constraints-----%

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


  return[0, [xx for xx in G1]]

def polyApprox(S, S_new, flag, S_bound):
  S_norm = []
  S_shifted = []
  Ai = []
  Aij = np.zeros((len(S),len(S)))
  for i in range(len(S)):
    S_norm.append(S_new[int(i/S[i])])
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

def Basic_MDO():
  #  Variables setup
  v = {}
  V: List[variableData] = []
  names = ["u", "v", "a", "b", "u", "w", "a", "b"]
  spi = [1,1,1,1,2,2,2,2]
  links = [2, None, 2, 2, 1, None, 1, 1]
  coupling_t = [COUPLING_TYPE.SHARED, COUPLING_TYPE.UNCOUPLED, COUPLING_TYPE.FEEDFORWARD,
  COUPLING_TYPE.FEEDBACK, COUPLING_TYPE.SHARED, COUPLING_TYPE.UNCOUPLED,
   COUPLING_TYPE.FEEDBACK, COUPLING_TYPE.FEEDFORWARD]
  lb = [0.]*8
  ub = [10.]*8
  bl = [1.]*8
  scaling = np.subtract(ub,lb)
  Qscaling = []
  # Variables dictionary with subproblems link
  for i in range(8):
    v[f"var{i+1}"] = {"index": i+1,
    "sp_index": spi[i],
    f"name": names[i],
    "dim": 1,
    "value": 0.,
    "coupling_type": coupling_t[i],
    "link": links[i],
    "baseline": bl[i],
    "scaling": scaling[i],
    "lb": lb[i],
    "value": bl[i],
    "ub": ub[i]}
    Qscaling.append(1/scaling[i] if 1/scaling[i] != np.inf and 1/scaling[i] != np.nan else 1.)

  for i in range(8):
    V.append(variableData(**v[f"var{i+1}"]))

  # Analyses setup; construct disciplinary analyses
  DA1: process = DA(inputs=[V[0], V[1], V[3]],
  outputs=[V[2]],
  blackbox=A1,
  links=2,
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  DA2: process = DA(inputs=[V[4], V[5], V[6]],
  outputs=[V[7]],
  blackbox=A2,
  links=1,
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  # MDA setup; construct subproblems MDA
  sp1_MDA: process = MDA(nAnalyses=1, analyses = [DA1], variables=[V[0], V[1], V[3]], responses=[V[2]])
  sp2_MDA: process = MDA(nAnalyses=1, analyses = [DA2], variables=[V[4], V[5], V[6]], responses=[V[7]])

  # Construct the coordinator
  coord = ADMM(beta = 1.3,
  nsp=2,
  budget = 50,
  index_of_master_SP=1,
  display = True,
  scaling = Qscaling,
  mode = "serial",
  M_update_scheme= w_scheme.MEDIAN,
  store_q_io=True)

  

  # Construct subproblems
  sp1 = SubProblem(nv = 3,
  index = 1,
  vars = [V[0], V[1], V[3]],
  resps = [V[2]],
  is_main=1,
  analysis= sp1_MDA,
  coordination=coord,
  opt=opt1,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  freal=2.625)

  sp2 = SubProblem(nv = 3,
  index = 2,
  vars = [V[4], V[5], V[6]],
  resps = [V[7]],
  is_main=0,
  analysis= sp2_MDA,
  coordination=coord,
  opt=opt2,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST
  )

  # Construct MDO workflow
  MDAO: MDO = MDO(
  Architecture = MDO_ARCHITECTURE.IDF,
  Coordinator = coord,
  subProblems = [sp1, sp2],
  variables = V,
  responses = [V[2], V[7]],
  fmin = np.inf,
  hmin = np.inf,
  display = True,
  inc_stop = 1E-9,
  stop = "Iteration budget exhausted",
  tab_inc = [],
  noprogress_stop = 100
  )

  # Run the MDO problem
  out = MDAO.run()

  print(f'------Run_Summary------')
  print(MDAO.stop)
  print(f'q = {MDAO.Coordinator.q}')
  for i in MDAO.Coordinator.master_vars:
    print(f'{i.name}_{i.sp_index} = {i.value}')

  fmin = 0
  hmax = -inf
  for j in range(len(MDAO.subProblems)):
    print(f'SP_{MDAO.subProblems[j].index}: fmin= {MDAO.subProblems[j].MDA_process.getOutputs()}, hmin= {MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , MDAO.subProblems[j].MDA_process.getOutputs())[1]}')
    fmin += sum(MDAO.subProblems[j].MDA_process.getOutputs())
    hmin= MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , MDAO.subProblems[j].MDA_process.getOutputs())[1]
    if max(hmin) > hmax: 
      hmax = max(hmin) 
  print(f'P_main: fmin= {fmin}, hmax= {hmax}')
  print(f'Final obj value of the main problem: \n {fmin}')

def speedReducer():
  #  Variables setup
  f1min = 722
  f1max = 5408
  f2min = 184
  f2max = 506
  f3min = 942
  f3max = 1369
  #
  v = {}
  V: List[variableData] = []
  s  = COUPLING_TYPE.SHARED
  ff = COUPLING_TYPE.FEEDFORWARD
  fb = COUPLING_TYPE.FEEDBACK
  un = COUPLING_TYPE.UNCOUPLED
  dum = COUPLING_TYPE.DUMMY


  names = ["x1", "x2", "x3", "f1",   "x1", "x2", "x3", "x4", "x6", "f2",   "x1", "x2", "x3", "x5", "x7", "f3"]
  spi =   [   1,    1,    1,		1,		  2,		2,		2,		2,		2,		2,      3,    3,		3,		3,    3,	  3]
  links = [[2,3],[2,3],[2,3],   None,  [1,3],[1,3],[1,3], None, None,    None,  [1,2],[1,2],[1,2], None, None,    None]
  lb =    [2.6 ,  0.7 ,  17., 722.,  2.6 ,  0.7,  17.,  7.3,  2.9, 184.,   2.6 ,  0.7,  17.,  7.3,   5.,942.]
  ub =    [3.6 ,  0.8 ,  28.,5408.,  3.6 ,  0.8,  28.,  8.3,  3.9, 506.,   3.6 ,  0.8 , 28.,  8.3,  5.5,1369.]
  bl =    np.divide(np.subtract(ub, lb), 2.)
  
  coupling_t = \
          [ s,      s,		s,		un,		s,		s,		s,		un,		un,	 un,   s,    s,    s,   un,    un,    un]
 
  scaling = [10.] * 16

  # Variables dictionary with subproblems link
  for i in range(16):
    v[f"var{i+1}"] = {"index": i+1,
    "sp_index": spi[i],
    f"name": names[i],
    "dim": 1,
    "value": 0.,
    "coupling_type": coupling_t[i],
    "link": links[i],
    "baseline": bl[i],
    "scaling": scaling[i],
    "lb": lb[i],
    "value": bl[i],
    "ub": ub[i]}

  for i in range(16):
    V.append(variableData(**v[f"var{i+1}"]))

  # Analyses setup; construct disciplinary analyses
  DA1: process = DA(inputs=[V[0], V[1], V[2]],
  outputs=[V[3]],
  blackbox=SR_A1,
  links=[4],
  coupling_type=COUPLING_TYPE.FEEDFORWARD)
  
  DA2: process = DA(inputs=[V[4], V[5], V[6], V[7], V[8]],
  outputs=[V[9]],
  blackbox=SR_A2,
  links=[4],
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  DA3: process = DA(inputs=[V[10], V[11], V[12], V[13], V[14]],
  outputs=[V[15]],
  blackbox=SR_A3,
  links=[4],
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  # MDA setup; construct subproblems MDA
  sp1_MDA: process = MDA(nAnalyses=1, analyses = [DA1], variables=[V[0], V[1], V[2]], responses=[V[3]])
  sp2_MDA: process = MDA(nAnalyses=1, analyses = [DA2], variables=[V[4], V[5], V[6], V[7], V[8]], responses=[V[9]])
  sp3_MDA: process = MDA(nAnalyses=1, analyses = [DA3], variables=[V[10], V[11], V[12], V[13], V[14]], responses=[V[15]])

  # Construct the coordinator
  coord = ADMM(beta = 1.3,
  nsp=4,
  budget = 50,
  index_of_master_SP=1,
  display = True,
  scaling = 0.1,
  mode = "serial",
  M_update_scheme= w_scheme.MEDIAN,
  store_q_io=True
  )

  # Construct subproblems
  sp1 = SubProblem(nv = 3,
  index = 1,
  vars = [V[0], V[1], V[2]],
  resps = [V[3]],
  is_main=1,
  analysis= sp1_MDA,
  coordination=coord,
  opt=SR_opt1,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  freal=2994.47)

  sp2 = SubProblem(nv = 5,
  index = 2,
  vars = [V[4], V[5], V[6], V[7], V[8]],
  resps = [V[9]],
  is_main=0,
  analysis= sp2_MDA,
  coordination=coord,
  opt=SR_opt2,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST
  )

  sp3 = SubProblem(nv = 5,
  index = 3,
  vars = [V[10], V[11], V[12], V[13], V[14]],
  resps = [V[15]],
  is_main=0,
  analysis= sp3_MDA,
  coordination=coord,
  opt=SR_opt3,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST)

# Construct MDO workflow
  MDAO: MDO = MDO(
  Architecture = MDO_ARCHITECTURE.IDF,
  Coordinator = coord,
  subProblems = [sp1, sp2, sp3],
  variables = V,
  responses = [V[3], V[9], V[15]],
  fmin = np.inf,
  hmin = np.inf,
  display = True,
  inc_stop = 1E-9,
  stop = "Iteration budget exhausted",
  tab_inc = [],
  noprogress_stop = 100
  )


# Run the MDO problem
  out = MDAO.run()

  print(f'------Run_Summary------')
  print(MDAO.stop)
  print(f'q = {MDAO.Coordinator.q}')
  for i in MDAO.Coordinator.master_vars:
    print(f'{i.name}_{i.sp_index} = {i.value}')
  fmin = 0
  hmax = -inf
  for j in range(len(MDAO.subProblems)):
    print(f'SP_{MDAO.subProblems[j].index}: fmin= {MDAO.subProblems[j].MDA_process.getOutputs()}, hmin= {MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , MDAO.subProblems[j].MDA_process.getOutputs())[1]}')
    fmin += sum(MDAO.subProblems[j].MDA_process.getOutputs())
    hmin= MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , MDAO.subProblems[j].MDA_process.getOutputs())[1]
    if max(hmin) > hmax: 
      hmax = max(hmin) 
  print(f'P_main: fmin= {fmin}, hmax= {hmax}')
  print(f'Final obj value of the main problem: \n {fmin}')

def geometric_programming():
  v = {}
  V: List[variableData] = []
  s  = COUPLING_TYPE.SHARED
  ff = COUPLING_TYPE.FEEDFORWARD
  fb = COUPLING_TYPE.FEEDBACK
  un = COUPLING_TYPE.UNCOUPLED
  dum = COUPLING_TYPE.DUMMY


  names = ["z3", "z4", "z5", "z6",   "z7", "z3", "z8", "z9", "z10", "z11",   "z6", "z11", "z12", "z13", "z14", "dd"]
  spi =   [   1,    1,    1,		1,		  1,		2,		2,		2,		2,		  2,      3,     3,		  3,		 3,     3, 1]
  links = [   2, None, None,    3,   None,    1, None, None, None,      3,      1,     2,  None,  None,  None, None]
  coupling_t = \
          [ fb,    un,	 un,	 fb,		 un,	 ff,	 un,	 un,	 un,	    s,     ff,     s,    un,    un,    un, ff]
 
  lb =    [1e-6]*15
  ub =    [1e6]*15
  bl =    [1.]*15
  scaling = [9.e5] * 15

  lb.append(15)
  ub.append(18)
  bl.append(GP_A1([1]*5))
  scaling.append(1)

  # Variables dictionary with subproblems link
  for i in range(16):
    v[f"var{i+1}"] = {"index": i+1,
    "sp_index": spi[i],
    f"name": names[i],
    "dim": 1,
    "coupling_type": coupling_t[i],
    "link": links[i],
    "baseline": bl[i],
    "scaling": scaling[i],
    "lb": lb[i],
    "value": bl[i],
    "ub": ub[i]}

  for i in range(16):
    V.append(variableData(**v[f"var{i+1}"]))
  
  # Analyses setup; construct disciplinary analyses
  DA1: process = DA(inputs=[V[0], V[1], V[2], V[3], V[4]],
  outputs=[V[15]],
  blackbox=GP_A1,
  links=[2, 3],
  coupling_type=COUPLING_TYPE.FEEDBACK)
  
  DA2: process = DA(inputs=[V[6], V[7], V[8], V[9]],
  outputs=[V[5]],
  blackbox=GP_A2,
  links=[1, 3],
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  DA3: process = DA(inputs=[V[11], V[12], V[13], V[14]],
  outputs=[V[10]],
  blackbox=GP_A3,
  links=[1, 2],
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  # MDA setup; construct subproblems MDA
  sp1_MDA: process = MDA(nAnalyses=1, analyses = [DA1], variables=[V[0], V[1], V[2], V[3], V[4]], responses=[V[15]])
  sp2_MDA: process = MDA(nAnalyses=1, analyses = [DA2], variables=[V[6], V[7], V[8], V[9]], responses=[V[5]])
  sp3_MDA: process = MDA(nAnalyses=1, analyses = [DA3], variables=[V[11], V[12], V[13], V[14]], responses=[V[10]])

  # Construct the coordinator
  coord = ADMM(beta = 1.3,
  nsp=3,
  budget = 200,
  index_of_master_SP=1,
  display = True,
  scaling = 1.,
  mode = "serial",
  M_update_scheme= w_scheme.MEDIAN,
  store_q_io=True
  )

  # Construct subproblems
  sp1 = SubProblem(nv = 5,
  index = 1,
  vars = [V[0], V[1], V[2], V[3], V[4]],
  resps = [V[15]],
  is_main=1,
  analysis= sp1_MDA,
  coordination=coord,
  opt=GP_opt1,
  fmin_nop=np.inf,
  budget=5,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  freal=15.)

  sp2 = SubProblem(nv = 4,
  index = 2,
  vars = [V[6], V[7], V[8], V[9]],
  resps = [V[5]],
  is_main=0,
  analysis= sp2_MDA,
  coordination=coord,
  opt=GP_opt2,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST
  )

  sp3 = SubProblem(nv = 4,
  index = 3,
  vars = [V[11], V[12], V[13], V[14]],
  resps = [V[10]],
  is_main=0,
  analysis= sp3_MDA,
  coordination=coord,
  opt=GP_opt3,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST)

# Construct MDO workflow
  MDAO: MDO = MDO(
  Architecture = MDO_ARCHITECTURE.IDF,
  Coordinator = coord,
  subProblems = [sp1, sp2, sp3],
  variables = V,
  responses = [V[5], V[10], V[15]],
  fmin = np.inf,
  hmin = np.inf,
  display = True,
  inc_stop = 1E-9,
  stop = "Iteration budget exhausted",
  tab_inc = [],
  noprogress_stop = 100
  )


# Run the MDO problem
  out = MDAO.run()
  print(f'------Run_Summary------')
  print(MDAO.stop)
  print(f'q = {MDAO.Coordinator.q}')
  xsp2 = []
  xsp3 = []
  index = np.argmax(MDAO.Coordinator.q)
  for i in MDAO.Coordinator.master_vars:
    if i.sp_index == MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].link:
      xsp2.append(i.value)
    if i.sp_index == MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].link:
      xsp3.append(i.value)
    print(f'{i.name}_{i.sp_index} = {i.value}')
  fmin = 0
  hmax = -inf
  for j in range(len(MDAO.subProblems)):
    print(f'SP_{MDAO.subProblems[j].index}: fmin= {MDAO.subProblems[j].MDA_process.getOutputs()}, hmin= {MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , [])[1]}')
    if MDAO.subProblems[j].is_main:
      fmin = MDAO.subProblems[j].MDA_process.getOutputs()
      hmin= MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , [])[1]
      if max(hmin) > hmax: 
        hmax = max(hmin) 
  print(f'P_main: fmin= {fmin}, hmax= {hmax}')
  print(f'Final obj value of the main problem: \n {fmin}')

  # Checking the impact of swapping z11_2 and z11_3 on the feasibility of SP2 and SP3, respectively
  
  temp = copy.deepcopy(xsp3[1])
  xsp3[1] = xsp2[-1]
  xsp2[-1] = temp
  print(f'For {MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].name}_'
    f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].link} to '
      f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].name}_'
    f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].link}, using {MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].name}_'
    f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].link}' 
    f' will make h_sp{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].link} = {GP_opt2(xsp2, 0)[1]}')
  print(f'For {MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].name}_'
    f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].link} to '
      f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].name}_'
    f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].link}, using {MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].name}_'
    f'{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[1,index]-1].link}' 
    f' will make h_sp{MDAO.Coordinator.master_vars[MDAO.Coordinator._linker[0,index]-1].link} = {GP_opt3(xsp3, 0)[1]}')

def SBJ():
  """ Supersonic Business Jet aircraft conceptual design"""
  # Variables definition
  v = {}
  V: List[variableData] = []
  s  = COUPLING_TYPE.SHARED
  ff = COUPLING_TYPE.FEEDFORWARD
  fb = COUPLING_TYPE.FEEDBACK
  un = COUPLING_TYPE.UNCOUPLED
  dum = COUPLING_TYPE.DUMMY


  names = ["SFC", "We", "LD", "Ws",   "Wf", "Wt",   "D", "T", "We", "SFC",   "ESF", "ESF", "Wt", "theta", "tc", "ARw", "LAMBDAw", "Sref", "Sht", "ARht", "LAMBDAht", "Lw", "Lht", "D", "LD", "L",   "L", "tc", "ARw", "LAMBDAw", "Sref", "Sht", "ARht", "Ws", "Wf", "theta", "lambda", "t", "ts"]
  spi =   [   1,    1,    1,		1,		  1,		1,		  2,	 2,		 2,		  2,       2,     3,		3,		   3,    3,     3,         3,      3,     3,      3,          3,    3,     3,   3,    3,   3,     4,    4,     4,         4,      4,     4,      4,    4,    4,       4,        4,   4,    4]
  links = [   2,    2,    3,    4,      4,    3,      3, None,   1,     1,       3,     2,    1,       4,    4,     4,         4,      4,     4,      4,       None, None,  None,   2,    1,   4,     3,    3,     3,         3,      3,     3,      3,    1,    1,       3,     None,None, None]
  coupling_t = \
          [ fb,    fb,	 fb,	 fb,		 fb,	 ff,	   fb,	 un,	 ff,	 ff,      ff,    fb,   fb,      fb,    s,     s,         s,      s,     s,      s,         un,   un,    un,  ff,   ff,  ff,    fb,   s,     s,         s,      s,     s,      s,   ff,   ff,      ff,       un,  un,   un]
  lb =    [1  ,   100,  0.1, 5000,   5000, 5000,   1000,  0.1,  100,    1,     0.5,   0.5, 5000,     0.2, 0.01,   2.5,        40.,    200.,    50,    2.5,         40, 0.01,     1,1000,  0.1,5000,  5000, 0.01,   2.5,        40,    200,    50,    2.5, 5000, 5000,     0.2,      0.1, 0.1,  0.1] 
  ub =    [4  , 30000,   10,100000,100000,100000, 70000,   1.,30000,    4,     1.5,   1.5,100000,     50,  0.1,    8.,        70.,    800., 148.9,    8.5,         70,  0.2,   3.5,70000,  10,100000,100000,0.1,     8,        70,    800, 148.9,    8.5,100000,100000,    50,      0.4, 4.0,  9.0]
  dim = [1]*39
  dim[37] = 9
  dim[38] = 9

  fstar = 33600
  frealistic = 30000
  altitude = 55000
  Mach = 1.4
  bl = [1.]*55
  scaling = np.subtract(ub,lb)

  Qscaling = []

  # Variables dictionary with subproblems link
  c = 0
  for i in range(len(names)):
    if dim[i] > 1:
      for j in range(dim[i]):
        v[f"var{i+c+1}"] = {"index": i+c+1,
        "sp_index": spi[i],
        f"name": names[i],
        "dim": 1,
        "value": 0.,
        "coupling_type": coupling_t[i],
        "link": links[i],
        "baseline": bl[i],
        "scaling": scaling[i],
        "lb": lb[i],
        "value": bl[i],
        "ub": ub[i]}
        c += 1
        Qscaling.append(.1/scaling[i] if .1/scaling[i] != np.inf and .1/scaling[i] != np.nan else 1.)
      c -=1
    else:
      v[f"var{i+c+1}"] = {"index": i+c+1,
      "sp_index": spi[i],
      f"name": names[i],
      "dim": 1,
      "value": 0.,
      "coupling_type": coupling_t[i],
      "link": links[i],
      "baseline": bl[i],
      "scaling": scaling[i],
      "lb": lb[i],
      "value": bl[i],
      "ub": ub[i]}
      Qscaling.append(.1/scaling[i] if .1/scaling[i] != np.inf and .1/scaling[i] != np.nan else 1.)

  for i in range(len(v)):
    V.append(variableData(**v[f"var{i+1}"]))

  # Analyses setup; construct disciplinary analyses
  V1 = copy.deepcopy([V[0], V[1], V[2], V[3], V[4]])
  Y1 = copy.deepcopy([V[5]])
  DA1: process = DA(inputs=V1,
  outputs=Y1,
  blackbox=SBJ_A1,
  links=[2,3,4],
  coupling_type=COUPLING_TYPE.FEEDFORWARD)
  
  V2 = copy.deepcopy([V[6], V[7]])
  Y2 = copy.deepcopy([V[8], V[9], V[10]])
  DA2: process = DA(inputs=V2,
  outputs=Y2,
  blackbox=SBJ_A2,
  links=[1,3],
  coupling_type=[COUPLING_TYPE.FEEDFORWARD, COUPLING_TYPE.FEEDFORWARD, COUPLING_TYPE.FEEDFORWARD]
  )
                    # "ESF", "Wt",  "theta", "tc", "ARw", "LAMBDAw", "Sref", "Sht", "ARht", "LAMBDAht", "Lw", "Lht", 
  V3 = copy.deepcopy([V[11], V[12], V[13], V[14], V[15], V[16],     V[17],  V[18], V[19],       V[20], V[21], V[22]])
  Y3 = copy.deepcopy([V[23], V[24], V[25]])
  DA3: process = DA(inputs=V3,
  outputs=Y3,
  blackbox=SBJ_A3,
  links=[1,2,4],
  coupling_type=[COUPLING_TYPE.FEEDFORWARD, COUPLING_TYPE.FEEDFORWARD, COUPLING_TYPE.FEEDFORWARD]
  )

  V4 = copy.deepcopy([V[26], V[27], V[28], V[29], V[30], V[31], V[32], V[36]]+V[37:len(V)])
  Y4 = copy.deepcopy([V[33], V[34], V[35]])
  DA4: process = DA(inputs=V4,
  outputs=Y4,
  blackbox=SBJ_A4,
  links=[1,3],
  coupling_type=[COUPLING_TYPE.FEEDFORWARD, COUPLING_TYPE.FEEDFORWARD, COUPLING_TYPE.FEEDFORWARD]
  )

# MDA setup; construct subproblems MDA
  sp1_MDA: process = MDA(nAnalyses=1, analyses = [DA1], variables=V1, responses=Y1)
  sp2_MDA: process = MDA(nAnalyses=1, analyses = [DA2], variables=V2, responses=Y2)
  sp3_MDA: process = MDA(nAnalyses=1, analyses = [DA3], variables=V3, responses=Y3)
  sp4_MDA: process = MDA(nAnalyses=1, analyses = [DA4], variables=V4, responses=Y4)

  # Construct the coordinator
  coord = ADMM(beta = 1.3,
  nsp=4,
  budget = 500,
  index_of_master_SP=1,
  display = True,
  scaling = Qscaling,
  mode = "serial",
  M_update_scheme= w_scheme.MEDIAN,
  store_q_io=True
  )

  # Construct subproblems
  sp1 = SubProblem(nv = len(V1),
  index = 1,
  vars = V1,
  resps = Y1,
  is_main=1,
  analysis= sp1_MDA,
  coordination=coord,
  opt=SBJ_opt1,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  freal=30000)

  sp2 = SubProblem(nv = len(V2),
  index = 2,
  vars = V2,
  resps = Y2,
  is_main=0,
  analysis= sp2_MDA,
  coordination=coord,
  opt=SBJ_opt2,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST
  )

  sp3 = SubProblem(nv = len(V3),
  index = 3,
  vars = V3,
  resps = Y3,
  is_main=0,
  analysis= sp3_MDA,
  coordination=coord,
  opt=SBJ_opt3,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST)

  sp4 = SubProblem(nv = len(V4),
  index = 4,
  vars = V4,
  resps = Y4,
  is_main=0,
  analysis= sp4_MDA,
  coordination=coord,
  opt=SBJ_opt4,
  fmin_nop=np.inf,
  budget=20,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST)

  # Construct MDO workflow
  MDAO: MDO = MDO(
  Architecture = MDO_ARCHITECTURE.IDF,
  Coordinator = coord,
  subProblems = [sp1, sp2, sp3, sp4],
  variables = V,
  responses = Y1+Y2+Y3+Y4,
  fmin = np.inf,
  hmin = np.inf,
  display = True,
  inc_stop = 1E-9,
  stop = "Iteration budget exhausted",
  tab_inc = [],
  noprogress_stop = 500
  )

  user = USER
  # Altitude
  user.h = 55000
  # Mach number
  user.M = 1.4
  
  out = MDAO.run()
  
def Sellar_A1(x):
  return x[0] + x[1]**2 + x[2] - 0.2*x[3]

def Sellar_A2(x):
  return x[0] + x[1] + np.sqrt(x[2]) 

def Sellar_opt1(x, y):
  return [x[0]**2 + x[2] + y[0] + np.exp(-x[3]), [3.16-y[0]]]

def Sellar_opt2(x, y):
  return [0., [y[0]-24.]]

def Sellar():
  #  Sellar - Two discipline problem with IDF

  #  Variables grouping and problem setup
  x = {}
  X: List[variableData] = []
  # Define variable names
  N = ["x", "z1", "z2", "y1", "y2", "z1", "z2", "y1", "y2"]
  nx: int = len(N)
  # Subproblem indices: Indices should be non-zero
  J = [1,1,1,1,1,2,2,2,2]
  # Subproblems links
  L = [None, 2, 2, 2, 2, 1, 1, 1, 1]
  # Coupling types
  Ct = [COUPLING_TYPE.UNCOUPLED, 
        COUPLING_TYPE.SHARED, 
        COUPLING_TYPE.SHARED,
        COUPLING_TYPE.FEEDFORWARD, 
        COUPLING_TYPE.FEEDBACK, 
        COUPLING_TYPE.SHARED,
        COUPLING_TYPE.SHARED, 
        COUPLING_TYPE.FEEDBACK,
        COUPLING_TYPE.FEEDFORWARD]
  # Realistic lower bounds
  lb = [0, -10, 0, 3.16, 1.77763888346, -10, 0, 3.16, 1.77763888346]
  # Realistic upper bounds
  ub = [10.,10.,10., 115.2, 24., 10.,10., 115.2, 24.]

  # # Artificial lower bounds
  # lb = [0, -10, 0, 2., 1.5, -10, 0, 2., 1.5]
  # # Artificial upper bounds
  # ub = [10.,10.,10., 50., 50, 10.,10., 50., 50]

  # Bad artificial lower bounds
  # lb = [0, -10, 0, 0., 0., -10, 0, 0., 0.]
  # Bad artificial upper bounds
  # ub = [10.]*9

  # Baseline
  x0 = [0., 5., 5., 8.43, 7.848, 5., 5., 8.43, 7.848]
  # Scaling
  scaling = np.subtract(ub,lb)
  Qscaling = []
  # Create a dictionary for each variable
  for i in range(nx):
    x[f"var{i+1}"] = {"index": i+1,
    "sp_index": J[i],
    f"name": N[i],
    "dim": 1,
    "value": 0.,
    "coupling_type": Ct[i],
    "link": L[i],
    "baseline": x0[i],
    "scaling": scaling[i],
    "lb": lb[i],
    "value": x0[i],
    "ub": ub[i]}
    Qscaling.append(10./scaling[i] if 10./scaling[i] != np.inf and 10./scaling[i] != np.nan else 1.)

  # Instantiate the variableData class for each variable using its according dictionary
  for i in range(nx):
    X.append(variableData(**x[f"var{i+1}"]))




  # Analyses setup; construct disciplinary analyses
  DA1: process = DA(inputs=[X[0], X[1], X[2], X[4]],
  outputs=[X[3]],
  blackbox=Sellar_A1,
  links=2,
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  DA2: process = DA(inputs=[X[5], X[6], X[7]],
  outputs=[X[8]],
  blackbox=Sellar_A2,
  links=1,
  coupling_type=COUPLING_TYPE.FEEDFORWARD
  )

  sp1_MDA: process = MDA(nAnalyses=1, analyses = [DA1], variables=[X[0], X[1], X[2], X[4]], responses=[X[3]])
  sp2_MDA: process = MDA(nAnalyses=1, analyses = [DA2], variables=[X[5], X[6], X[7]], responses=[X[8]])


  # Construct the coordinator
  coord = ADMM(beta = 1.3,
  nsp=2,
  budget = 500,
  index_of_master_SP=1,
  display = True,
  scaling = Qscaling,
  mode = "serial",
  M_update_scheme= w_scheme.MAX,
  store_q_io=True
  )

  # Construct subproblems
  sp1 = SubProblem(nv = 4,
  index = 1,
  vars = [X[0], X[1], X[2], X[4]],
  resps = [X[3]],
  is_main=1,
  analysis= sp1_MDA,
  coordination=coord,
  opt=Sellar_opt1,
  fmin_nop=np.inf,
  budget=10,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST,
  freal=3.160,
  solver="scipy")

  sp2 = SubProblem(nv = 3,
  index = 2,
  vars = [X[5], X[6], X[7]],
  resps = [X[8]],
  is_main=0,
  analysis= sp2_MDA,
  coordination=coord,
  opt=Sellar_opt2,
  fmin_nop=np.inf,
  budget=10,
  display=False,
  psize = 1.,
  pupdate=PSIZE_UPDATE.LAST
  )

  MDAO: MDO = MDO(
  Architecture = MDO_ARCHITECTURE.IDF,
  Coordinator = coord,
  subProblems = [sp1, sp2],
  variables = X,
  responses = [X[3], X[8]],
  fmin = np.inf,
  hmin = np.inf,
  display = True,
  inc_stop = 1E-9,
  stop = "Iteration budget exhausted",
  tab_inc = [],
  noprogress_stop = 1000
  )

  # Run the MDO problem
  out = MDAO.run()

  # Print summary output

  print(f'------Run_Summary------')
  print(MDAO.stop)
  print(f'q = {MDAO.Coordinator.q}')
  for i in MDAO.Coordinator.master_vars:
    print(f'{i.name}_{i.sp_index} = {i.value}')

  fmin = 0
  hmax = -np.inf
  for j in range(len(MDAO.subProblems)):
    print(f'SP_{MDAO.subProblems[j].index}: fmin= {MDAO.subProblems[j].MDA_process.getOutputs()}, hmin= {MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , MDAO.subProblems[j].MDA_process.getOutputs())[1]}')
    fmin += sum(MDAO.subProblems[j].MDA_process.getOutputs())
    hmin= MDAO.subProblems[j].opt([s.value for s in MDAO.subProblems[j].get_design_vars()] , MDAO.subProblems[j].MDA_process.getOutputs())[1]
    if max(hmin) > hmax: 
      hmax = max(hmin) 
  print(f'P_main: fmin= {fmin}, hmax= {hmax}')
  print(f'Final obj value of the main problem: \n {fmin}')

# def test_basic_quick():
#   Basic_MDO()

def test_auto_build():
  p_file: str = os.path.abspath("./tests/test_files/Basic_MDO.yaml")
  MDAO: MDO = main(p_file, "build")
  for i in range(len(MDAO.subProblems)):
    temp :MDA = MDAO.subProblems[i].MDA_process
    for j in range(len(temp.analyses)):
      MDAO.subProblems[i].MDA_process.analyses[j].blackbox = globals()[MDAO.subProblems[i].MDA_process.analyses[j].blackbox]
    MDAO.subProblems[i].opt = globals()[MDAO.subProblems[i].opt]
  
  MDAO.run()


# def test_auto_build_run():
#   p_file: str = os.path.abspath("/Users/ahmedb/apps/code/Bay_dev/DMDO/tests/test_files/Basic_MDO.yaml")
#   global A1, A2, opt1, opt2
#   MDAO: MDO = main(p_file, "run", A1, A2, opt1, opt2)

# test_auto_build_run()
def test_comperhensive():
  Sellar()
  speedReducer()
  geometric_programming()
  # SBJ()
