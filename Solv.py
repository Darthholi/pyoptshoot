"""
todo:
-sensitivita zatim neni pro self.fpathconstraints, self.fendstate, pro scipy optim minimiz zpusob, pro pocatecni podminky
-fpathconstraints neotestovano, fendstate neotestovano

-integrator pro kresleni grafu
-p jako parametry pro optimalizaci nejsou zaneseny dovnitr do funkce a finalnimodelfunkce

-shape je theano symboilicka variable pro size .... muzou bejt nekde pripady, kde recenim konkretni size (znamy ze zadani problemu) problem zrychlime? (tzn size neceho jinyho misto shape neceho?)

-integrovat musim i od pocatecnich hodnot, ale ty sou pevny.
-integracni zacatky JSOU pro stavovy promenny, ale NE pro stav objective...
"""

#general objects:
import copy
from theano.compile.io import In
from theano.compile.io import Out  
import numpy as np
import scipy as sp
import theano.tensor as T
import theano
import intheanogators as tint
from scipy.integrate import odeint
#from scikits.odes import ode
import scipy.optimize
import matplotlib.pyplot as plt
from pyOpt import Optimization
from pyOpt.pySLSQP import pySLSQP
from pyOpt.pyPSQP import pyPSQP
from pyOpt.pyALGENCAN import pyALGENCAN
#http://stackoverflow.com/questions/2883189/calling-matlab-functions-from-python
##https://www.python-forum.de/viewtopic.php?t=37839 pyopt64bit
#http://scicomp.stackexchange.com/questions/83/is-there-a-high-quality-nonlinear-programming-solver-for-python
#PSQP SLSQP scipy.optimize.minimize...
#fsqp nlpqlp commercial
#for debug:
#theano.config.optimizer = 'fast_compile' #'None' #'fast_run' #fast compile
#theano.config.exception_verbosity = 'high'

floatUse = T.config.floatX

class ModelMultiEval:     #a class that simplyfies the cases wen obj, constraint and gradients are computed together. IF THE Objective is the first one to call everytime! (#otherwise can check lastx....)
  def __init__(self,
               MasterFun=None,#funkce parametru a vraci: {obj: , objgrad: , eqcon: , eqcongrad: , incon: , incongrad:}
               ):
    self.MasterFun=MasterFun
    self.lastx = None
    self.result = None
    self.debugprint = True
    self.iter = 0
    self.integratorfunc = tint.theano_inner_rk4_step
    
  def FuncCall(self,x):
    if (self.lastx is None or np.any(x != self.lastx)):
      if (self.result is not None):
        self.oldobj = self.result['obj']
      self.result = copy.deepcopy(self.MasterFun(x))  # only thing needed if no dbugprint

      if (self.debugprint and self.lastx is not None):
        self.iter +=1
        print "res: "+str(self.result['obj'])+" at iter "+str(self.iter)+" xdiff: "+str(np.max(np.abs(x - self.lastx)))+" objdiff:"+str(self.result['obj']-self.oldobj)
    
      self.lastx = copy.deepcopy(x)  # only thing needed if no debugprint
    
  def ObjCall(self,x):
    self.FuncCall(x)
    return np.cast(self.result['obj']),floatUse)
  
  def Call(self,x,str):
    self.FuncCall(x)
    return np.cast(self.result[str]),floatUse)
  
  def ObjGradCall(self,x):
    self.FuncCall(x)
    return self.result['objgrad']
    
  def EqConCall(self,x):
    self.FuncCall(x)
    return self.result['eqcon']
    
  def EqConGradCall(self,x):
    self.FuncCall(x)
    return self.result['eqcongrad']
   
  def InConCall(self,x):
    self.FuncCall(x)
    if ('incon' in self.result):
      return self.result['incon']
    else:
      return None
   
  def InConGradCall(self,x):
    self.FuncCall(x)
    return self.result['incongrad'] 
  

class SimModel:
  def __init__(self,statebeg,stateend, stateMax,stateMin,controlMax,controlMin,
        #multipleshootingdim,                     
        fodestate, ffinalobjcon,
        fpathconstraints=None,
        constarrays = None,
        otherparamsMin=None,   #params to optimize, that are not states and not controls...
        otherparamsMax=None,
        laststatesum = False,#set true if fodestate returns more dimensions, than there are state equations
        T=1,
        gensens = 'ad'):
    self.gensens = gensens
    self.statebeg = np.array(statebeg)                  #None means IS not fixed...
    self.stateend = np.array(stateend)
    self.laststatesum = laststatesum
    for i in xrange(len(stateMax)):
      if (stateMax[i]>np.finfo(floatUse).max/20.0):
        stateMax[i] = np.finfo(floatUse).max/20.0
    for i in xrange(len(stateMin)):
      if (stateMin[i]<-1.0*np.finfo(floatUse).max/20.0):
        stateMin[i] = -1.0*np.finfo(floatUse).max/20.0
        
    self.stateMax=np.array(stateMax,dtype = floatUse) #scalary:
    self.stateMin=np.array(stateMin,dtype = floatUse)
    self.controlMax=np.array(controlMax)
    self.controlMin=np.array(controlMin)
    self.otherparamsMin=np.array(otherparamsMin)                               #asi ve forme array?? nebo povolime i list?
    self.otherparamsMax=np.array(otherparamsMax)
    #dod check that all theese things above have the same dimensions!
    
    self.multipleshootingdim = self.stateMax.shape[0] #multipleshootingdim #this number of dimensions (from states = statebeg, fodestate()) will be discretized to multipleshooting. The remaining (multipleshootingdim+1:-1) will be kept continuous
    self.controlfdim = self.controlMax.shape[0]
    self.fodestate = fodestate #state equation state(x,u,t,p)
    self.constarrays = constarrays
    self.ffinalobjcon = ffinalobjcon #objective function recieving result of ODE from state faftersim(simresult), returning constraints at the final point!
    self.fpathconstraints = fpathconstraints #funkce rikajici jestli stavova rovnice splnuje constrainty-rovnice a nerovnice (vraci jako list)     ...nastaveno na NONE znamena ze se nemusi volat. Musi li se volat, vola se podle logiky multipleshootingu na kazdej bod z diskretizovanejch
      #(x,u) at all times...
      
    self.T = T #fixed timespan [0,T] for ode solver.

#dod - volat simulaci s hodnotama z optimalizatoru, A pridat k nim pevny hodnoty pokud fixedbegin OK
 
  def GenForDiscretization(self,ndisc=32,maxoptimizers=1000,odeintsteps=1000):                  #ndisc INCLUDES the beginning and ending point regardless if we set them to be free or not!
    self.maxoptimizers=maxoptimizers
    self.odeintsteps=odeintsteps
    self.ndisc = ndisc #number of discretization points
    
    #how many variables will we be optimizing?
    self.startmaptopack = []
    self.PackForOptimSize = 0  
    for i in range(self.statebeg.shape[0]):
        if (self.statebeg[i] == None):  #if it is NONE then it IS a parameter to be optimized, so pack him
          self.PackForOptimSize+=1
          self.startmaptopack.append(i) 
    self.PackForOptimSize += (self.ndisc-2)*self.multipleshootingdim #self.statebeg.size[0] #each state has its own discretization points
    self.PackForOptimSize += (self.ndisc-1)*self.controlfdim  #each control has its own discretization points - also in the first point!
    if (len(self.otherparamsMin.shape)>0):
      self.PackForOptimSize += self.otherparamsMin.shape[0]     
    
    #so the packing is like this:
    #free starting values (if any)
    #discretized states (to ndisc-2) points (dimension after dimension) #only the states in self.multipleshootingdim first dimensions!
    #discretized control (to ndisc-1) points (dimension after dimension)
    #parameters - (not discretized)
    
    #so the bounds to the optim function will go like this
    #state constraints (if any free beginning states)
    #state constraints (to ndisc-2) points (dimension after dimension)
    #control constraints (to ndisc-1) points (dimension after dimension)
    #parameter constraints
    
    #
      
    def PackForOptim(xbeg,x,u,p):#uses ndisc    #some xbeg ARE not packed!!
      packed = np.empty(self.PackForOptimSize,floatUse)
      xoffset=0
      for i in range(self.statebeg.shape[0]):
        if (self.statebeg[i] == None):            #if it is NONE then it IS a parameter to be optimized, so pack him
          packed[xoffset]=xbeg[i]
          xoffset+=1     
      
      packed[xoffset:(xoffset+self.multipleshootingdim*(self.ndisc-2))] = np.reshape(x,self.multipleshootingdim*(self.ndisc-2))
      xoffset+=self.multipleshootingdim*(self.ndisc-2)
      if (self.controlfdim>0):
        packed[xoffset:(xoffset+self.controlfdim*(self.ndisc-1))] = np.reshape(u,self.controlfdim*(self.ndisc-1))
        xoffset+=self.controlfdim*(self.ndisc-1)
      if (p is not None):
        packed[xoffset:] = p[:]
      
      return packed

    """#the same packing function but using theano...
    def PackForOptimTheano(xbeg, x, u, p):  # uses ndisc    #some xbeg ARE not packed!!
      packed = T.empty(self.PackForOptimSize, floatUse)
      xoffset = 0
      for i in range(self.statebeg.shape[0]):
        if (self.statebeg[i] == None):  # if it is NONE then it IS a parameter to be optimized, so pack him
          packed[xoffset] = xbeg[i]
          xoffset += 1
  
      packed[xoffset:(xoffset + self.multipleshootingdim * (self.ndisc - 2))] = T.reshape(x,
                                                                                           self.multipleshootingdim * (
                                                                                           self.ndisc - 2))
      xoffset += self.multipleshootingdim * (self.ndisc - 2)
      if (self.controlfdim > 0):
        packed[xoffset:(xoffset + self.controlfdim * (self.ndisc - 1))] = T.reshape(u, self.controlfdim * (self.ndisc - 1))
        xoffset += self.controlfdim * (self.ndisc - 1)
      if (p is not None):
        packed[xoffset:] = p[:]
  
      return packed
      self.PackForOptimTheano = PackForOptimTheano
    """
          
    self.PackForOptim = PackForOptim
    
    def UnpackForOptim(Packed): #uses ndisc
      ret={} #scipy.optimize.minimize casts to float64.... we ned to cast back for theano
      #if we use f64 then np.empty is not needed for u,p
      ret['x']=np.empty([self.multipleshootingdim,(ndisc-2)+1],floatUse)
      ret['u']=np.empty([self.controlfdim,ndisc-1],floatUse)
      if (len(self.otherparamsMin.shape)>0):
        ret['p']=np.empty(self.otherparamsMin.shape[0],floatUse)
      else:
        ret['p'] = None
      xoffset=0
      for i in range(self.statebeg.shape[0]):
        if (self.statebeg[i] != None):
          ret['x'][i,0]=self.statebeg[i]
        else:
          ret['x'][i,0]=Packed[xoffset]          #if it is NONE then it IS a parameter to be optimized, so unpack him
          xoffset+=1
      ret['x'][:,1:]= np.reshape(Packed[xoffset:xoffset+(ndisc-2)*self.multipleshootingdim],(self.multipleshootingdim,ndisc-2))
      xoffset+=(ndisc-2)*self.multipleshootingdim
      ret['u'][:,:]=np.reshape(Packed[xoffset:xoffset+(ndisc-1)*self.controlfdim],(self.controlfdim,ndisc-1))
      xoffset+=(ndisc-1)*self.controlfdim
      if (len(self.otherparamsMin.shape)>0):
        ret['p'][:]=Packed[xoffset:]  #totally should be ... otherparamsMin.size[0]
     
      #print self.controlfdim
      #print ret['u'].shape
           
      return ret
      
    self.UnpackForOptim = UnpackForOptim

  def BuildTheanoModel(self, f, objcon, constarrays, gensens, sumlastaxis=False):
    u_function = T.matrix("u_function")  # parametrizovana ridici funkce
    p_params = T.vector("p_params")
  
    inputsarray = []
    theano_constarrays = None
    if constarrays is not None:
      if isinstance(constarrays, dict):
        theano_constarrays = {}
        for key in constarrays.keys():
          theano_constarrays[key] = TensorType(floatUse,
                                               (False,) * len(constarrays[key].shape))  # T.matrix("constarrays_" + key)
          inputsarray.append(In(theano_constarrays[key], name=key))
          # else:#error
  
    def finteg(accum,
               t):  # integrator inputs only accumulator vector and time scalar, so other parametrs we need to input NOW.
      listres = f(accum, t[0], u_function, p_params, theano_constarrays)
      return theano.tensor.stack(*listres)  # f does not need to use theano notation, can return [a,b]
  
    integratorvars = BuildTheanoIntegrator(finteg, sumlastaxis, self.integratorfunc)
  
    ##integratorvars['results_var'][:,-1] shoud be "all state variables; at their last point"
    # into this function the objcon goes for all computed path endings to get all possible custom path constraints
    # integratorresults = T.zeros([integratorvars['results_var'].shape[0]])
    integratorresults = integratorvars['results_var'][:, -1]  # should be the last value for each integrato result
    if (sumlastaxis):
      # integratorvars['results_var'][-1] = T.elemwise.Sum(integratorvars['results_var'][-1])
      integratorresults = T.set_subtensor(integratorresults[-1], T.sum(integratorvars['results_var'][-1, :]))

    # result = objcon(integratorvars['results_var'],u_function,p_params)                       #moznost dat pouze koncovy body uz tady
    ## #objective function must be evaluated & we do consider objective function at the last point. (multipleshooting says that the continuity is handled by the beginnings-endings of path)
    # result['obj']=result['obj'][-1]#if we put arrays into integratorresults
    origresult = objcon(integratorresults, u_function, p_params)  # moznost dat pouze koncovy body uz tady
    # objective function must be evaluated & we do consider objective function at the last point. (multipleshooting says that the continuity is handled by the beginnings-endings of path)
  
    begs = integratorvars['xbegs_var'][:, 1:]  # all but not first item
    ends = integratorvars['results_var'][0:begs.shape[0],0:-1]  # all but not last item ###############chci to rollovat po axis CASOVY - shape[1]] (ne po shape[0])
    endtrajectory = integratorvars['results_var'][0:begs.shape[0],-1] #only the last item
    result = {}
    result['obj'] = origresult['obj']

    # if the function returns list of matrices for example, we need to make one big vector out of it...
    # result['incon'] = origresult['incon']
    if (isinstance(origresult['incon'], list) and len(origresult['incon']) >= 1):
      if len(origresult['incon']) == 1:
        result['incon'] = T.flatten(origresult['incon'][0])
      else:
        tmpstack = []
        for il in origresult['incon']:
          tmpstack.append(T.flatten(il))
        result['incon'] = T.concatenate(tmpstack)

    #constraints-----------------------------------------------------------------------------------------------
    # apply multiple shooting algorithm constraints
    # way of less equations: -> sensitivities go into ONE line instead of matrix...
    multishootingeqcons = T.sum((ends - begs) * (ends - begs))  # should be a scalar .. dotproduct...
    #way    of    more    equations: T.flatten(ends-begs)
    if (origresult['eqcon'] == None or len(origresult['eqcon']) <= 0):
      result['eqcon'] = multishootingeqcons
    else:
      result['eqcon'] = T.concatenate(origresult['eqcon'], multishootingeqcons)

    if (self.stateend is None): #is stateend is none, we consider it constrained by normal path constraints. If it is not None, later we will add equality constraints
      # apply end state constraint - if theree are state-min and state-max, they are automatically applied to ALL beginnings of paths that means also to all endings, but not to the last one, so lets do it:
      multishootingincons = T.concatenate(T.flatten(endtrajectory - self.stateMin),
                                          T.flatten(self.stateMax - endtrajectory))  # needs to be >=0
      if (origresult['incon'] == None or len(origresult['incon']) <= 0):
        result['incon'] = multishootingeqcons
      else:
        result['incon'] = T.concatenate(origresult['incon'], multishootingincons)
      
    #todo: add fpathconstraints and stateend
    """
    # apply state-control-param path constraints:
    if (self.fpathconstraints != None):  # 2) apply state constraints can be included to theano...
      for i in range(self.ndisc - 1):
        thiscon = self.fpathconstraints(case['x'][:, i], case['u'][:, i], case['p'])
        if (thiscon['eqcon'] != None):
          computedsim['eqcon'].extend(thiscon['eqcon'])
        if (thiscon['incon'] != None):
          computedsim['incon'].extend(thiscon['incon'])
    """
    """
    # apply finalstate-param constraints:
    if (self.stateend is not None):
      if (self.stateend is callable):
        thiscon = self.stateend(computedsim['x'][:, -1], case['p'])
      else:  # is a vector of constants that we must target...
        thiscon = {}
        thiscon['eqcon'] = []
        for i in range(self.stateend.shape[0]):
          if (self.stateend[i] != None):
            thiscon['eqcon'].append(self.stateend[i] - computedsim['x'][i, -1])
      if (thiscon['eqcon'] != None and len(thiscon['eqcon']) > 0):
        computedsim['eqcon'] = np.append(computedsim['eqcon'],
                                         thiscon['eqcon'])  # computedsim['eqcon'].extend(thiscon['eqcon'])
      # if (thiscon['incon'] != None and len(thiscon['incon'])>0):
      #  computedsim['incon'] = np.append(computedsim['incon'],thiscon['eqcon']) #computedsim['incon'].extend(thiscon['incon'])
    """
  
    """
    #dod sensitivities for eqcon will be in the form of vectors
    begssens = sensvarsx['xbegs_var'][:, 1:]  # all but not first item
    endssens = sensvarsx['results_var'][0:begssens.shape[0], 0:-1]
    """
  
    if (gensens == 'ad'):
      result['objgrad_x'] = T.grad(result['obj'], integratorvars['xbegs_var'])  # vector
      result['objgrad_u'] = T.grad(result['obj'], u_function)  # vector
      result['objgrad_p'] = T.grad(result['obj'], p_params, disconnected_inputs='ignore')  # vector

      # matrices we want [i,j] di/dj
      if (result['eqcon'] is not None):
        result['eqcongrad_x'] = T.jacobian(result['eqcon'], integratorvars['xbegs_var'])
        result['eqcongrad_u'] = T.jacobian(result['eqcon'], u_function)
        result['eqcongrad_p'] = T.jacobian(result['eqcon'], p_params, disconnected_inputs='ignore')
    
      if (result['incon'] is not None):
        result['incongrad_x'] = T.jacobian(result['incon'], integratorvars['xbegs_var'])
        result['incongrad_u'] = T.jacobian(result['incon'], u_function)
        result['incongrad_p'] = T.jacobian(result['incon'], p_params, disconnected_inputs='ignore')
  
    """
    if (self.gensens == 'sens' ):
      # sensitivity here #dod overit ze d+1 jsou na spravnym miste...
      # we do not account for starting point being dependent on p or u...
      def fintegpartialx(accum,t):  # integrator inputs only accumulator vector and time scalar, so other parametrs we need to input NOW.
        listres =  T.gradient.jacobian(finteg,x) # = partial finteg / partial x ... matrix d x d+1 .... #f(accum,t[0],u_function,p_params,theano_constarrays)
        return theano.tensor.stack(*listres)  # f does not need to use theano notation, can return [a,b]

      def fintegpartialu(accum,t):  # integrator inputs only accumulator vector and time scalar, so other parametrs we need to input NOW.
        listres =  # = partial finteg / partial u #f(accum,t[0],u_function,p_params,theano_constarrays)
        return theano.tensor.stack(*listres)  # f does not need to use theano notation, can return [a,b]

      def fintegpartialp(accum,t):  # integrator inputs only accumulator vector and time scalar, so other parametrs we need to input NOW.
        listres =  # = partial finteg / partial p #f(accum,t[0],u_function,p_params,theano_constarrays)
        return theano.tensor.stack(*listres)  # f does not need to use theano notation, can return [a,b]

      #sens: (dim x dim+1 x ndisc) #+1 for 'additional state for cost'  # poradi v druhym nasobiteli? dod
      #if we set it like this, then it means that sensvar[i,j,n] is a sensitivity of state[j] against change in [i]
      #(if dim+1 is somewhere else than in the middle, change from now on)

      sensvarsx = BuildTheanoObjSensXIntegrator(fintegpartialx, integratorvars['Tmax'], integratorvars['k'],
                                                begs.shape[0], begs.shape[1])
      sensvarsu = BuildTheanoObjSensUIntegrator(fintegpartialu, integratorvars['Tmax'], integratorvars['k'],
                                                begs.shape[0], begs.shape[1])
      sensvarsu = BuildTheanoObjSensPIntegrator(fintegpartialp, integratorvars['Tmax'], integratorvars['k'],
                                                begs.shape[0], begs.shape[1])
      objpartialx = T.grad(objcon(integratorresults,u_function,p_params),integratorvars['results_var']) # = partial objcon / partial integratorvars['results_var'][:,-1]
      # objpartialu # = partial objcon / partial u_function
      # objpartialp # = partial objcon / partial p_params
      #chain rule - derivative at integrated point times derivative of parameters...
      obj_grad_xbegs = objpartialx(integratorresults, u_function, p_params) *  # matrix times vector
      obj_grad_u = objpartialu(integratorresults, u_function, p_params) *  # matrix times vector
      obj_grad_p = objpartialp(integratorresults, u_function, p_params) *  # matrix times vector
      # eqsensitivity:

      #derivative at integrated point times
      #matice (dim x ndisc) * ( dim x dim+1 x ndisc) #poradi v druhym nasobiteli? dod
      #for differentiating inner "ends":
      #for each ndisc we want to
      #(end-beg)[d]*sens[j,d] for each d sum against j
      #together it is (end-beg)[d,n]*sens[j,d,n] sum over d, vector over j x n (grads of jth component )
      #2* T.sum(T.tensordot(ends - begs,sensvarsx['results_var'][:,  0:begs.shape[0],0:-1], [[0,1],[1,2]] )) #poradi indexu? DOD
      #for differentiating inner "-begs":
      #begs do have a structure - it is sensvarsx['xbegs_var'] and they are identity matrixes x ndisc times repeated
      #dont forget that the begs are baing indexed the same way as 'begs' upper:
      #2* -T.sum(T.tensordot(ends - begs,sensvarsx['xbegs_var'][:,  1:,0:-1], [[0,1],[1,2]] )) #poradi indexu? DOD
      chaininnereqcon = sensvarsx['results_var'][:, 0:begs.shape[0], 0:-1] - sensvarsx['xbegs_var'][:, 1:, 0:-1]
      eqgrad = 2.0*(
        T.tensordot(ends - begs, chaininnereqcon, [[0], [1]]) )
      #this is a vector of gradients of the multipleshooting eq-constraint equation, should now have the length of ndisc-1

      #zbejva reagovat na result'incon'
    """
  
    theanoresult = {}
    for key in result:
      if (result[key] is not None):
        theanoresult[key] = result[key]  # Out(variable = result[key])#, name = key) variant #2
  
    inputsarray = [In(integratorvars['xbegs_var'], name='x'),
                   In(u_function, name='u'),
                   In(p_params, name='p'),
                   In(integratorvars['numsteps_var'], name='k'),
                   In(integratorvars['Tmax_var'], name='Tmax')] + inputsarray
  
    theano.config.on_unused_input = 'warn'
    objective_call = theano.function(inputs=inputsarray,
                                     outputs=theanoresult,  # updates=None,
                                     allow_input_downcast=True,  # todo try false and see where the downcast originates.
                                     on_unused_input='warn'  # u_function for example...
                                     )  # http://deeplearning.net/software/theano/library/compile/io.html
    # f a list of Variable or Out instances is given as argument, then the compiled function will return a list of their values.

    # def rtnf(x,u,p,k,Tmax):
    #  return objective_call(x=T.cast(x,floatUse),u=T.cast(u,floatUse),p=T.cast(p,floatUse),k=k,Tmax=T.cast(Tmax,floatUse))
  
    return objective_call
    
  def RunOptim(self):
  
    #broadcasts can be:, eg. np.random.uniform([0,1],[1,10],(5,2))
    #def random_vectored(low, high,):
    #  return [random.uniform(low[i], high[i]) for i in xrange(low.size[0])]
    
    #dod specifikovat zacatek, pripadne ho vzit z minulyho behu...
    
    print self.multipleshootingdim
    print self.ndisc-2
    
    cstartzeroes = True #set true to set zeroes, set false to set random
    
    controlfrng = None
    if (self.controlfdim>0):
      controlfrng = np.empty([self.controlfdim, self.ndisc - 1], floatUse)
      for i in xrange(self.controlfdim):
        controlfrng[i,:] = np.zeros((1,self.ndisc-1)) if cstartzeroes else np.random.uniform(low=self.controlMin[i], high=self.controlMax[i],size=(1,self.ndisc-1))
    randinitx = np.empty([self.multipleshootingdim,self.ndisc-2],floatUse)
    for i in xrange(self.multipleshootingdim):
      randinitx[i,:] = np.zeros((1,self.ndisc-2)) if cstartzeroes else np.random.uniform(low=self.stateMin[i], high=self.stateMax[i], size=(1,self.ndisc-2))
    otherparamsinit = None
    if (len(self.otherparamsMin.shape)>0):
      otherparamsinit = np.random.uniform(low=self.otherparamsMin,high=self.otherparamsMax)
      
    x0 = self.PackForOptim(np.zeros(self.stateMin.shape) if cstartzeroes else np.random.uniform(low=self.stateMin, high=self.stateMax),
                      randinitx,       #np.random.uniform(low=self.stateMin, high=self.stateMax, size=(self.multipleshootingdim,self.ndisc-2))
                      controlfrng,      #np.random.uniform(low=self.controlMin, high=self.controlMax,size=(self.controlfdim,self.ndisc-1)
                      otherparamsinit
                      )
    
    wlinitx = np.empty([self.multipleshootingdim,self.ndisc-2],floatUse)
    for i in xrange(self.multipleshootingdim):
      wlinitx[i,:] = self.stateMin[i]
    otherparamsminp = None
    if (len(self.otherparamsMin.shape)>0):
      otherparamsminp = self.otherparamsMin
    controlfwl = None
    if (self.controlfdim > 0):
      controlfwl = np.empty([self.controlfdim, self.ndisc - 1], floatUse)
      for i in xrange(self.controlfdim):
        controlfwl[i, :] = self.controlMin[i]
    wL = self.PackForOptim(self.stateMin,
                      wlinitx, #np.tile(self.stateMin,(self.multipleshootingdim,self.ndisc-2)),      
                      controlfwl,#np.tile(self.controlMin,(self.controlfdim,self.ndisc-1)),
                      otherparamsminp
                      )
    
    wuinitx = np.empty([self.multipleshootingdim,self.ndisc-2],floatUse)
    for i in xrange(self.multipleshootingdim):
      wuinitx[i,:] = self.stateMax[i]
    otherparamsmaxp = None   
    if (len(self.otherparamsMax.shape)>0):
      otherparamsmaxp = self.otherparamsMax
    controlfwu = None
    if (self.controlfdim > 0):
      controlfwu = np.empty([self.controlfdim, self.ndisc - 1], floatUse)
      for i in xrange(self.controlfdim):
        controlfwu[i, :] = self.controlMax[i]
    wU = self.PackForOptim(self.stateMax,
                      wuinitx, #np.tile(self.stateMax,(self.multipleshootingdim,self.ndisc-2)),
                      controlfwu, #np.tile(self.controlMax,(self.controlfdim,self.ndisc-1)),
                      otherparamsmaxp
                      )
    #print wL
    #print wU
    self.ParallelizSim = None
    def ObjFromPacked(Inp):
      case=self.UnpackForOptim(Inp)                                                    #1) unpack for optim can be included to theano
      case['k']=self.odeintsteps
      case['Tmax']=self.T
      if (self.constarrays is not None):
        case.update(self.constarrays) #we add the calling constant arrays .. each iteration they are the same...

      printdebug = False
      if (printdebug):
        print "x: "+str(case['x'])
        print "u: "+str(case['u'])
      
      if (self.ParallelizSim is None):
        self.ParallelizSim = self.BuildTheanoModel(self.fodestate, self.ffinalobjcon, self.constarrays, self.gensens,
                                         self.laststatesum)  # state,ModelFunkce)
        
      computedsim = self.ParallelizSim(**case)
    
    self.ObjFromPacked = ObjFromPacked
            
    #v kazdy iteraci optimalizatoru
    #calleval.ObjCall({'x': konkretni hodnota x, 'u': konretni hodnota u, 'k': pocet integracnich iteraci})
    calleval = ModelMultiEval(ObjFromPacked)     #object remembering all that we have computed to not call sim again for constraints...

    objx0 = calleval.ObjCall(x0)
    print "obj at x0 " + str(objx0) #important to have different obj at different points ....
    print "obj at wL "+str(calleval.ObjCall(wL))
    eqconx0 = calleval.EqConCall(x0)
    inconx0 = calleval.InConCall(x0)
    if (eqconx0 is not None):
      if (len(eqconx0.shape)<=0):
        eqconx0num = 1
      else:
        eqconx0num = len(eqconx0)
    else:
      eqconx0num = 0
     
    if (inconx0 is not None):
      if (len(inconx0.shape)<=0):
        inconx0num = 1
      else:
        inconx0num = len(inconx0)
    else:
        inconx0num = 0

    #paralleliz sim is a theano function. We reset it here, because when called from inside a library pyopt, python.exe crashes
    #if reset, it builds again (only once) and then does not crash
    #self.ParallelizSim = None # to compile again inside pyopt.
    #does not work, ipopt still crashes...
    
    """
    #sundials cvode:
    from scikits.odes import ode
    solution = ode('cvode', van_der_pol, old_api=False).solve(np.linspace(t0,500,200), y0)
    solution.
    """
    
    
    cusescipy =False
    if (cusescipy):
      #todo sensitivities
      if (eqconx0num>0 or eqconx0num>0):
        xconstr = []
        if (eqconx0num>0):
          xconstr.append({'type': 'eq', 'fun': calleval.EqConCall})
        if (inconx0num>0):
          xconstr.append({'type': 'ineq', 'fun': calleval.InConCall})  # inequality means that it is to be non-negative
      else:
        xconstr = ()  # empty iterable...
        
      res = scipy.optimize.minimize(fun = calleval.ObjCall,
          x0=x0, args=(), method='SLSQP', jac=False, hess=None, hessp=None, #SLSQP #BFGS #COBYLA
          bounds=zip(wL,wU),     #dod je zip spravne?
          constraints=xconstr,
          tol=1e-9, callback=None,
          options={'maxiter': self.maxoptimizers, 'disp':True, 'iprint':2})
          #http://stackoverflow.com/questions/23476152/dynamically-writing-the-objective-function-and-constraints-for-scipy-optimize-mihttp://stackoverflow.com/questions/23476152/dynamically-writing-the-objective-function-and-constraints-for-scipy-optimize-mi
      
      if (not res.success):
        print(res.message)
      print('Objective function at minimum: (computed by theano) ')
      print(res.fun)
      if (hasattr(res,'maxcv')):
        print('Constraint violation (computed by theano) ')
        print(res.maxcv)
      found = self.UnpackForOptim(res.x)
      return found
      #found.x - states x (ndisc-2) ....states    (incl. beginning)
      #found.u - controls x (ndisc-2) ....controls
      #found.p - params
    else:
      def pyoptobj(x):
        fail = 0 #ok
        g = []
        #in-this-order-we-do-present-the-constraints ... must be the same at each block labelled the same...
        if (eqconx0num>0 and inconx0num>0):
          g=np.concatenate([calleval.EqConCall(x),calleval.InConCall(x)])
        elif (eqconx0num>0):
          g = calleval.EqConCall(x)
        elif (inconx0num>0):
          g = calleval.InConCall(x)
        return calleval.ObjCall(x),g,fail

      if (self.gensens is not None):
        def pyoptgrad(x, f, g):
          #- x -> ARRAY: Design variables
          # - f -> ARRAY: Objective values
          # - g -> ARRAY: Constraint values
    
          g_obj = self.PackForOptim(
              calleval.Call(x,'objgrad_x')[:,0],#the gradient is solved against all beginning points of x. The pack for optim function takes first the initial state and then the discretized beginnings
              calleval.Call(x,'objgrad_x')[:,1:],
              calleval.Call(x,'objgrad_u'),
              calleval.Call(x,'objgrad_p')
                      )
          
          # in-this-order-we-do-present-the-constraints ... must be the same at each block labelled the same...
          if (eqconx0num>0 or eqconx0num>0):
            g_con = np.empty(eqconx0num+inconx0num, self.PackForOptimSize)
          for i in xrange(eqconx0num):
            g_con[i, :] = self.PackForOptim(
              calleval.Call(x, 'eqcongrad_x')[i, 0],
              calleval.Call(x, 'eqcongrad_x')[i, 1:],
              calleval.Call(x, 'eqcongrad_u')[i, :],
              calleval.Call(x, 'eqcongrad_p')[i, :]
            )
          for i in xrange(inconx0num):
            g_con[i + eqconx0num, :] = self.PackForOptim(
              calleval.Call(x, 'incongrad_x')[i, 0],
              calleval.Call(x, 'incongrad_x')[i, 1:],
              calleval.Call(x, 'incongrad_x')[i, :],
              calleval.Call(x, 'incongrad_u')[i, :],
              calleval.Call(x, 'incongrad_p')[i, :]
            )
          fail = 0
          return g_obj, g_con, fail
          #needs return g_obj vector
          #gcon - 2D array - [derivative of which con][over which x]
  
        sens_type = pyoptgrad
      else:
        sens_type = 'FD'
      
      opt_prob = Optimization('Multipleshooting', pyoptobj)
      for i in xrange(len(x0)):
        opt_prob.addVar('x'+str(i), 'c', lower = wL[i], upper = wU[i], value=x0[i])
      opt_prob.addObj('f')

      if (eqconx0num>0): #in-this-order-we-do-present-the-constraints ... must be the same at each block labelled the same...
        opt_prob.addConGroup('EqCons', eqconx0num, type='e', equal=0.0)
      if (inconx0num>0):
        opt_prob.addConGroup('InCons', inconx0num, type='i', lower=0.0)
      print opt_prob
      #opt = pySLSQP.SLSQP(options={'IPRINT': 0, 'MAXIT': self.maxoptimizers})
      opt = pyPSQP.PSQP(options={'IPRINT': 2, 'MIT': self.maxoptimizers })
      #opt = pyALGENCAN.ALGENCAN(options={'iprint': 12})
      #FSQP(options={'iprint': 3, 'miter': self.maxoptimizers })
      res = opt(opt_prob, sens_type=sens_type, disp_opts=True, sens_mode='',sens_step=1e-4)
      # Solve Problem with Optimizer Using Finite Difference
      print opt_prob.solution(0)
      print opt_prob.solution(0).opt_inform['text']
      found = self.UnpackForOptim(opt_prob.solution(0).parameters)
      return found
    
  def DrawResults(self,Sim):
    #-------------------------------------
    #for painting, compute the model and ODE's AGAIN numerically without gpu, plot them (and we can see if the numerical integration on gpu was precise or not...)
    #http://stackoverflow.com/questions/27820725/how-to-solve-diff-eq-using-scipy-integrate-odeint
    #def g(y, x):
    #    y0 = y[0]
    #    y1 = y[1]
    #    y2 = ((3*x+2)*y1 + (6*x-8)*y0)/(3*x-1)
    #    return y1, y2
    #jsou dve varianty jak integrovat - predpokladat spojitost, nebo nakreslit to co vyslo ev i s nespojitostma-zacinat forcyklovef 
    init = Sim['x'][:,0]
    Tmax = self.T
    tspace= np.linspace(0,Tmax,100)

    theano.config.on_unused_input = 'warn'
    sym_x = T.vector('x', floatUse)
    sym_t = T.scalar('t', floatUse)
    sym_p = T.matrix('p', floatUse)
    sym_w = T.vector('w', floatUse)
    sym_dataarrays = T.matrix('dataarrays', floatUse)
    onestateresult = self.fodestate(sym_x, sym_t, sym_w, sym_p, sym_dataarrays) #fodestate is defined as theano function ... so lets make python function from it
    state_call = theano.function(inputs=[In(sym_x, name='x'),
                                         In(sym_t, name='t'),
                                         In(sym_w, name='w'),
                                         In(sym_p, name='p'),
                                         In(sym_dataarrays, name='dataarrays')],
                                 outputs=onestateresult,  # updates=None,
                                 allow_input_downcast=True,
                                 on_unused_input='warn'
                                 )
    
    #trycall = state_call( init.astype(np.float32), np.array(0.0).astype(np.float32), Sim['u'][:,0].astype(np.float32), Sim['p'].astype(np.float32) if Sim['p'] is not None else None, self.constarrays.astype(np.float32) if self.constarrays is not None else None)
    
    def integstate(x,t):
      u=Sim['u'][:,int(np.floor(t/Tmax))]
      #return np.array(self.fodestate(x,t,u,Sim['p'],self.constarrays))
      return np.array(state_call( np.cast(x,floatUse), np.array(t,floatUse), np.cast(u,floatUse), np.cast(Sim['p'],floatUse) if Sim['p'] is not None else None, np.cast(self.constarrays,floatUse) if self.constarrays is not None else None))[0:init.shape[0]]
     
    #spojite: 
    sol=odeint(integstate, np.array(init), tspace)
    for i in range(init.shape[0]):
      plt.plot(tspace, sol[:,i], color='b')
      #plt.plot(tspace, sol[:,1], color='r')
      #plt.plot(tspace, sol[:,2], color='r')
    #plt.show()
    
    #nespojite
    for nd in range(self.ndisc-1):
      init = Sim['x'][:,nd]
      tspace= np.linspace((nd/self.ndisc)*Tmax,((nd+1.0)/self.ndisc)*Tmax,100)
      sol=odeint(integstate, np.array(init), tspace)
      for i in range(init.shape[0]):
        plt.plot(tspace, sol[:,i], color='gray')
    
    #objode = ModelFunkce(Sim[:,-1]) #objektivni funkce, tentokrat spocteno numericky bez theana a bez multipleshootingu
    #print('Objective function at minimum - without Multiple Shooting possible discontinuities: (computed by odeint ')
    #print(objode)
    
    plt.show()
    
        #dispopt( x0, ts, wopt,Data, optODE,-1 );
        #dlmwrite('controldata',wopt);
    

            
def BuildTheanoIntegrator(f,doaddstatezeros,integratorfunc):
  Tmax = T.scalar("Tmax")
  k = T.iscalar("k")                #pocet iteraci
  x_begs_input = T.matrix("x_begs",floatUse)       #zacatky vsech integracnich mist...  (vicedim funkce, proto matice) --------parametr
  #u_function = T.matrix("u_function")   #parametrizovana ridici funkce                                      --------parametr
  #f = definovanatheano expression
  
  if (doaddstatezeros):
    # new_zers = T.zeros((1,x_begs.shape[1]))
    # x_begs = T.unbroadcast(T.concatenate([x_begs,new_zers],axis = 1))
    new_x = T.zeros((x_begs_input.shape[0]+1,x_begs_input.shape[1]),floatUse)
    x_begs = T.set_subtensor(new_x[0:-1,:],x_begs_input)#all but not the last row
  else:
    x_begs = x_begs_input
  
  # Symbolic description of the result
  # f vstupuje pri kompilaci.... ostatni jako non_sequences... (DOD pridat f jako nonsequence?)
  pIndex = T.cast(theano.tensor.stack(theano.tensor.arange(x_begs.shape[1])).dimshuffle(1, 0),floatUse)
  result, updates = theano.scan(fn=lambda _i,_accum,_Tmax,_Index,_k :  integratorfunc(_i,_accum,_Tmax,_Index,_k,f),
                                outputs_info=[0,x_begs],#[T.zeros(1,1),x_begs],
                                #sequences=
                                non_sequences=[Tmax,pIndex,T.cast(k,floatUse)],
                                n_steps=k)
  
  # We only care about A**k, but scan has provided us with A**1 through A**k.
  # Discard the values that we don't care about. Scan is smart enough to
  # notice this and not waste memory saving them.
  #result [0] = k je pocet iteraci....
  #result [1] je to co chceme - pole vysledku...
  result = result[1][-1] #hodnoty poslednich vysledku
  #... z toho chceme to posledni, secteny. Theano si vsimne ze to predtim nepotrebujeme a nebude to ukladat...
  #ted ma origresult[1] tvar [#pocet iteraci scanu, dimenze funkce,pocet diskretizacnich bodu]
  #result = result[-1,0,-1] #hodnoty v posledni iteraci, prvni hodnote z objfunc (doufejme jediny), posledni diskretizacni bod


  xret = {'numsteps_var': k, 'xbegs_var': x_begs_input, 'results_var': result, 'Tmax_var': Tmax}
  
  return xret
  
  # compiled function that returns A**k
  #power = theano.function(inputs=[A,k], outputs=result, updates=updates)
  #print(power(range(10),2))
  #print(power(range(10),4))

"""
def BuildTheanoSensStateIntegrator(f, Tmax, k, ndim, ndisc):
  if (Tmax is None):
    Tmax = T.scalar("Tmax")
  if (k is None):
    k = T.iscalar("k")  # pocet iteraci
  if (ndim is None):
    ndim = T.iscalar("ndim")
  if (ndisc is None):
    ndisc = T.iscalar("ndisc")
  #input: we want theano to compute exaclty this: np.transpose(np.tile(np.eye(2, dtype=np.float32), (5, 1, 1)))
  #or, else, for the costfunction - np.transpose(np.tile(np.eye(3,2),(5,1,1)))
  #ndim .. +1 for cost function...
  #to understand - play in console with np.transpose(np.tile(np.eye(3,2),(5,1,1)))[0] ...
  sensbegs = T.transpose(T.tile(T.eye(ndim+1,ndim, floatUse), (ndisc, 1, 1)))

  
  # Symbolic description of the result
  # f vstupuje pri kompilaci.... ostatni jako non_sequences...
  # sensbegs.shape[2] = ndisc
  pIndex = T.cast(theano.tensor.stack(theano.tensor.arange(sensbegs.shape[2])).dimshuffle(1, 0), floatUse)       #is this true for sens?  # DOD
  result, updates = theano.scan(
    fn=lambda _i, _accum, _Tmax, _Index, _k: theano_inner_rk4_step(_i, _accum, _Tmax, _Index, _k, f),
    outputs_info=[0, sensbegs],  # [T.zeros(1,1),x_begs],
    # sequences=
    non_sequences=[Tmax, pIndex, T.cast(k, floatUse)],
    n_steps=k)

  # result [0] = k je pocet iteraci....
  # result [1] je to co chceme - pole vysledku...
  result = result[1][-1]  # hodnoty poslednich vysledku
  # ... z toho chceme to posledni, secteny. Theano si vsimne ze to predtim nepotrebujeme a nebude to ukladat...
  # ted ma origresult[1] tvar [#pocet iteraci scanu, dimenze funkce,pocet diskretizacnich bodu]
  # result = result[-1,0,-1] #hodnoty v posledni iteraci, prvni hodnote z objfunc (doufejme jediny), posledni diskretizacni bod
  
  
  xret = {'numsteps_var': k, 'xbegs_var': sensbegs, 'results_var': result, 'Tmax_var': Tmax, 'ndim_var': ndim, 'ndisc_var': ndisc}
  return xret
"""
"""
def BuildTheanoObjSensXIntegrator(f, Tmax, k, ndim, ndisc):
  if (Tmax is None):
    Tmax = T.scalar("Tmax")
  if (k is None):
    k = T.iscalar("k")  # pocet iteraci
  if (ndim is None):
    ndim = T.iscalar("ndim")
  if (ndisc is None):
    ndisc = T.iscalar("ndisc")
  # input: we want theano to compute exaclty this: np.transpose(np.tile(np.eye(2, dtype=np.float32), (5, 1, 1)))
  # or, else, for the costfunction - np.transpose(np.tile(np.eye(3,2),(5,1,1)))
  # ndim .. +1 for cost function...
  # to understand - play in console with np.transpose(np.tile(np.eye(3,2),(5,1,1)))[0] ...
  sensbegs = T.transpose(T.tile(T.eye(ndim + 1, ndim, floatUse), (ndisc, 1, 1)))
  
  # Symbolic description of the result
  # f vstupuje pri kompilaci.... ostatni jako non_sequences...
  # sensbegs.shape[2] = ndisc
  pIndex = T.cast(theano.tensor.stack(theano.tensor.arange(sensbegs.shape[2])).dimshuffle(1, 0),
                  floatUse)  # is this true for sens?  # DOD
  result, updates = theano.scan(
    fn=lambda _i, _accum, _Tmax, _Index, _k: theano_inner_rk4_step(_i, _accum, _Tmax, _Index, _k, f),
    outputs_info=[0, sensbegs],  # [T.zeros(1,1),x_begs],
    # sequences=
    non_sequences=[Tmax, pIndex, T.cast(k, floatUse)],
    n_steps=k)
  
  # result [0] = k je pocet iteraci....
  # result [1] je to co chceme - pole vysledku...
  result = result[1][-1]  # hodnoty poslednich vysledku
  # ... z toho chceme to posledni, secteny. Theano si vsimne ze to predtim nepotrebujeme a nebude to ukladat...
  # ted ma origresult[1] tvar [#pocet iteraci scanu, dimenze funkce,pocet diskretizacnich bodu]
  # result = result[-1,0,-1] #hodnoty v posledni iteraci, prvni hodnote z objfunc (doufejme jediny), posledni diskretizacni bod
  
  
  xret = {'numsteps_var': k, 'xbegs_var': sensbegs, 'results_var': result, 'Tmax_var': Tmax, 'ndim_var': ndim,
          'ndisc_var': ndisc}
  return xret
"""


"""
      how theano orders jacobians:
      In[3]: import theano
      In[4]: import theano.tensor as T
      In[5]: x = T.dvector('x')
      In[6]: y=T.concatenate([x,2.0*x])
      In[7]: J = T.jacobian(y,x)
      In[8]: f = theano.function([x],J)
      In[9]: f([4,5,6])
      Out[9]:
      array([[ 1.,  0.,  0.],
             [ 0.,  1.,  0.],
             [ 0.,  0.,  1.],
             [ 2.,  0.,  0.],
             [ 0.,  2.,  0.],
             [ 0.,  0.,  2.]])
      In[10]: f([4,5])
      Out[10]:
      array([[ 1.,  0.],
             [ 0.,  1.],
             [ 2.,  0.],
             [ 0.,  2.]])
      In[11]: f([4,5])[0,:]
      Out[11]: array([ 1.,  0.])
      In[12]: f([1])
      Out[12]:
      array([[ 1.],
             [ 2.]])
      In[13]: f([1])[0,:]
      Out[13]: array([ 1.])
      In[14]: f([1])[1,:]
      Out[14]: array([ 2.])
      """