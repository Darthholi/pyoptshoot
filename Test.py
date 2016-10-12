"""
todo:
-theano vector function to matrix
-integrovat musim i od pocatecnich hodnot, ale ty sou pevny.
-integracni zacatky JSOU pro stavovy promenny, ale NE pro stav objective...
"""

#general objects:
import Solv as solv
import copy
from theano.compile.io import In  
import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import theano.tensor as T
import theano

import copy
from theano.compile.io import In
from theano.compile.io import Out  
import numpy as np
import scipy as sp
import theano.tensor as T
import theano
from scipy.integrate import odeint
import matplotlib            


def state( x, t, w, p ):       #the order of arguments must be like for odeint...
  dx=[2.0*t,3.0*t]
  return dx

def ModelFunkce(sim,control,params):
  ret = {'obj': -sim[0]+control[0]*control[0], 'eqcon': None, 'incon': None}
  return ret
   #other things that need to be equal to zero, comes from problem formulation                                                                      #dod kolik eqs? secist nebo soustavu? soustavu!

def main():

  x0 = [ 0, 0]; # Initial states+1 state for cost variable .. yes it is discretized too...
  xend=[None,None]  
  
  tspace = np.linspace(0,10,1000)
  x=np.tile(tspace,(2,1)).astype('float32')   #-> to shape (2,100)
  print x.shape
  
  #theano.config.compute_test_value = 'warn' - musi bejt vyplneny - defvalue pro theano tensory
  
  ParallelizSim = solv.BuildTheanoModel(state,ModelFunkce)
  
  #Test n1 - funguje paralleliz sim?
  case = {}
  case['x'] = x
  case['u'] = None
  #case['p'] = None
  case['k']=10000 #self.odeintsteps
  case['Tmax']= 10.0 #self.T
  
  #theano.printing.debugprint(ParallelizSim)
  
  
  
  ####debug integratoru:   -------------------------------------------------------
  
  TryScan = False
  if (TryScan):
    x0 = [ 0.0]; # Initial states+1 state for cost variable .. yes it is discretized too...
    xend=[None]  
    
    tspace = np.linspace(0,10,100)
    x=np.tile(tspace,(1,1)).astype('float32')   #-> to shape (2,100)
    trysteps = 1000
    tmaxconst = 100.0
    def theano_inner_rk4_step(#accum:                               #i_step je integer...
                            i_step,accum,                         #accum je matice - pocet dimenzi vysledny funkce krat pocet bodu ve kterejch integrujeme po krivce
                            #menici se:                          
                            #pevny paramtery: 
                            Index):
      #integracni casy:    
      fshape = T.cast(Index.shape[0],'float32')   
      Tim_t = Index * tmaxconst / fshape #vektor (index)0,1,2,3,4,5,6,7,.....  -> vektor (tim_t)0,1/(n*Tmax) .... n/n * Tmax
      Tim_t = Tim_t+ tmaxconst*(i_step / (fshape*trysteps))                 #-posunuto na integracni step ted...
      #integracni krok:
      t_step =  (tmaxconst / fshape) / trysteps   
      
      #accum - states x (ndisc-1)
      
      def f( x, t ):
        dx=[200.0*t[0]]#[100.0*theano.tensor.ones_like(x[0])]#[200.0*x[0]]#,3*x[0]
        return theano.tensor.stack(*dx)    #vraci list[a,b,...] a my ho chceme dat primo jako parametr..
                                 
      k1 = f(accum,Tim_t)                                                   #y'=f(y,t) (vicedim fce...) #aplikuj funkci PO SLOUPCICH
      k2 = f(accum + t_step*0.5*k1,Tim_t+0.5*t_step)
      k3 = f(accum + t_step*0.5*k2,Tim_t+0.5*t_step)
      k4 = f(accum + t_step*k3,Tim_t+t_step)
      return i_step+1,T.cast(accum + t_step/6.0 *( k1 + 2*k2 + 2*k3 + k4),'float32')
    
    x_begs = T.matrix("x_begs")
    
    #try_result = theano_inner_rk4_step(0,x_begs,theano.tensor.arange(x_begs.shape[1]))
    #try_objective_call = theano.function(inputs=[ In(x_begs,name='xbegs')],
    #                          outputs=try_result,#updates=None,
    #                          on_unused_input='warn'          #u_function for example...
    #                          )
    
    pIndex = T.cast(theano.tensor.stack(theano.tensor.arange(x_begs.shape[1])).dimshuffle(1, 0),'float32')
    try_result, try_updates = theano.scan(fn=theano_inner_rk4_step,
                                outputs_info=[T.cast(0,'int32'),T.cast(x_begs,'float32')],
                                #sequences=
                                non_sequences=[pIndex],
                                n_steps=trysteps)
    #try_result[1] = try_result[1][-1]
    try_result_ret = try_result[1]
                                
    try_objective_call = theano.function(inputs=[ In(x_begs,name='xbegs')],
                              outputs=try_result_ret,#updates=None,
                              on_unused_input='warn'          #u_function for example...
                              )
     
    toprint = try_objective_call(x)
    print "clk"
    print toprint.shape
    #print toprint
    print " ......begs"
    print x.shape
    print x
    print " ................. [-1,0,:] naintegrovane konce"
    print toprint[-1,0,:].shape    
    print toprint[-1,0,:]
    #print " ................. [:,0,-1] jak se integrovalo"    
    #print toprint[:,0,-1]    
    plt.plot(x[0,:],toprint[-1,0,:],color = 'r')
    plt.plot(x[0,:],toprint[-1,0,:]-x[0,:],color = 'b')
    plt.show()                       
    #x_bags = 2,1000; "theano.tensor.arange(x_begs.shape[1])".shape[0]=10000, 
    ###innerrk4_step - Tim_t shape[0]=10000; 
    ###innerrk4step - k1 = list
    ###innerrk4step - accum.shape[0] = 2, accum.shape[1]=10000
    ###innerrk4step - k4 = [array(),array()]
    ###             - T.cast(i_step+1, 'int32') , ok
    ###             - ( k1 + 2*k2 + 2*k3 + k4) shape[0]] bylo 12!! to nema bejt... (shape[1] bylo 10000 ok)
    # zjistteni -> dx=[2*t,3*t*t] vraci list a ne matrix. na to kdyz pouziju + tak se dimenze scitaj a neni to elementwise...
    #reseni je bud T.cast() PRED operaci + NEBO theano.tensor.stack(*dx), dx=[a,b,c,...]
    #v outputs info musi bejt T.cast(0,'int32'),, jinak to z toho udelat int8 a pretece...
    #vysledek theano.scan je [1000,2] ... ?
    #theano scan na scitani tisici poli 2x10000 vrati pole 1000x2x10000 a my vezmeme tu posledni hodnotu (viz http://deeplearning.net/software/theano/library/scan.html)
                          
                                
  
  ####end debug integratoru  -------------------------------------------------------- 
  
  tryobj = False
  if (tryobj):
    #odkomentuj: 
    computedsim = ParallelizSim(**case)    #Simres = ParallelizSim(x=x,u=0,k=1000,Tmax=1)
    #print computedsim
    #print computedsim['obj'].shape
    print "diffs"
    print computedsim['obj']-x[0,:]
    print "acts"    
    print computedsim['obj']
    #print tspace.shape
      
  
    #print matplotlib.rcParams['backend']  
      
    #plt.plot(tspace, computedsim['obj'], color='b')
    #plt.savefig('D:\\Data\Projects\\optimalcontrol\\Theano\\testfig.png')
    #plt.plot(tspace,computedsim['obj']+tspace,color = 'b')
    #plt.plot(tspace,computedsim['obj'],color = 'b')
    #plt.show()
  
    
    #raw_input('press enter')
  
  
  tryodeint = False
  if (tryodeint):
    init = x0
    #Sim = {}
    #Sim['u'] = 
    #tspace= np.linspace(0,Tmax,100)
    def integstate(x,t):
      u=0 #Sim['u'][:,floor(t/Tmax)]
      return state(x,t,u)
      
    sol=odeint(integstate, init, tspace)
    plt.plot(tspace, sol[:,0], color='b')
    plt.plot(tspace, sol[:,1], color='r')
    plt.show() 
    
  tryclass=True
  if (tryclass):
    OptimSim = solv.SimModel(x0,xend, 
                        stateMax=[np.inf,np.inf],
                        stateMin=[-np.inf,-np.inf],
                        controlMax=[10],
                        controlMin=[-10],
                        #multipleshootingdim,     #dod rict ze stavy popisujeme dva ale ze funkce bude pouzivat dalsi 1 na ukladani ceny...                
                        fodestate=state,
                        ffinalobjcon=ModelFunkce,
                        fpathconstraints=None,              #others than min max
                        otherparamsMin=None,   #params to optimize, that are not states and not controls...
                        otherparamsMax=None)
                        
                        
                        
    OptimSim.GenForDiscretization(32,1000,1000) #ndisc=32,
   
                      
    res=OptimSim.RunOptim()
    OptimSim.DrawResults(res)               
   
        
if __name__ == '__main__':
    #main()
    import HydrogModel
    HydrogModel.main()