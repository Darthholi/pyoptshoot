"""
todo:
-theano vector function to matrix
-integrovat musim i od pocatecnich hodnot, ale ty sou pevny.
-integracni zacatky JSOU pro stavovy promenny, ale NE pro stav objective...
"""

#general objects:
import Solv.py
import copy
from theano.compile.io import In  
import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def state( x, t, w ):
  dx=[2*t,3*t*t]
  return dx

def ModelFunkce(sim,control):
  ret['obj']=  -sim[0]
  ret['eqcon'] = []   #other things that need to be equal to zero, comes from problem formulation                                                                      #dod kolik eqs? secist nebo soustavu? soustavu!
  ret['incon'] = []

def main():

  x0 = [ 0, 0]; # Initial states+1 state for cost variable .. yes it is discretized too...
  xend=[None,None]  
  
  tspace = np.linspace(0,10,100)
  x=np.tile(tspace,(2,100))
  
  ParallelizSim = BuildTheanoModel(state,ModelFunkce)
  
  #Test n1 - funguje paralleliz sim?
  case['x'] = x
  case['u'] = None
  case['p'] = None
  case['k']=1000 #self.odeintsteps
  case['Tmax']= 1 #self.T 
  computedsim = ParallelizSim(case)    #Simres = ParallelizSim(x=x,u=0,k=1000,Tmax=1)
    
  plt.plot(tspace, computedsim['obj'][:,0], color='b')
  plt.plot(tspace, computedsim['obj'][:,1], color='r')
  
  """
  OptimSim = SimModel(x0,xend, 
                      stateMax=[np.inf,np.inf],
                      stateMin=[-np.inf,-np.inf],
                      controlMax=[],
                      controlMin=[],
                      #multipleshootingdim,     #dod rict ze stavy popisujeme dva ale ze funkce bude pouzivat dalsi 1 na ukladani ceny...                
                      fodestate=state,
                      ffinalobjcon=ModelFunkce,
                      fpathconstraints=None,              #others than min max
                      otherparamsMin=None,   #params to optimize, that are not states and not controls...
                      otherparamsMax=None)
                      
                      
                      
   OptimSim.GenForDiscretization(ndisc=32,maxoptimiters=1000,odeintsteps=1000):
   """
                      
   #res=OptimSim.RunOptim()
   #OptimSim.DrawResults(res)               

        
