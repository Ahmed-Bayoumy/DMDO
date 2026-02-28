import numpy as np
import math

#Initial values
# from original text:
#       λ     x	    Cf	    T	  t/c	  h	      M	      AR	  Δ	  SREF
# vlb	0.1	  0.75	0.75	0.1	  0.01	30000	  1.4	  2.5	  40	500
# i0	0.25  1	    1	    0.5	  0.05	45000	  1.6	  5.5	  55	1000
# vub	0.4	  1.25	1.25	1	  0.09	60000	  1.8	  8.5	  70	1500

# used here, from Tosserams; baselines set to the middle of the range
#"sp_index": 1 , f"name": "SFC"        , "coupling_type": fb   , "link": 2     , "lb": 1     , "ub": 4          
#"sp_index": 1 , f"name": "We"         , "coupling_type": fb   , "link": 2     , "lb": 100   , "ub": 30000      
#"sp_index": 1 , f"name": "Wt"         , "coupling_type": ff   , "link": 3     , "lb": 5000  , "ub": 100000    
#"sp_index": 1 , f"name": "LD"         , "coupling_type": fb   , "link": 3     , "lb": 0.1   , "ub": 10         
#"sp_index": 1 , f"name": "Ws"         , "coupling_type": fb   , "link": 4     , "lb": 5000  , "ub": 100000    
#"sp_index": 1 , f"name": "Wf"         , "coupling_type": fb   , "link": 4     , "lb": 5000  , "ub": 100000   

#######################
#Inputs
h = 55000
Mach = 1.4

SFCp =  2
We = 15000 #Engnie weight [lb]
LDr =  5.0 #Lift/Drag ratio
Ws = 25000 #Structural weight
Wf = 25000 #Fuel weight

#Constants

#Locals

#Calculations
Wt = We + Wf + Ws # total weight

# SBJ_constraint_range
if h < 36089 :
  theta_r = 1-0.000006875*h
else :
   theta_r = 0.7519

range = Mach * LDr * 661.0 * np.sqrt(theta_r/SFCp)*math.log(Wt/(Wt-Wf))

#Constraints
g = -range/2000. + 1.

#Printing
# print("range = ", range)
# print("Wt = ", Wt)
# print("g = ", g)