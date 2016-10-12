"""
todo:
-theano vector function to matrix
-testy funkci predtim nez...
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
from scipy.integrate import odeint
import scipy.optimize
import matplotlib.pyplot as plt

class ModelMultiEval:     #a class that simplyfies the cases wen obj, constraint and gradients are computed together. IF THE Objective is the first one to call everytime! (#otherwise can check lastx....)
  def __init__(self,
               MasterFun=None,#funkce parametru a vraci: {obj: , objgrad: , eqcon: , eqcongrad: , incon: , incongrad:}
               ):
    self.MasterFun=MasterFun
    self.lastx = None
    self.result = None
    
  def ObjCall(self,x):
    if (self.lastx is None or np.all(x != self.lastx)):
      self.lastx = copy.deepcopy(x)
      self.result = self.MasterFun(x)
    return float(self.result['obj'])
    
  def ObjGradCall(self,x):
    if (self.lastx is None or np.all(x != self.lastx)):
      self.ObjCall(x)
    return self.result['objgrad']
    
  def EqConCall(self,x):
    if (self.lastx is None or np.all(x != self.lastx)):
      self.ObjCall(x)
    return self.result['eqcon']
    
  def EqConGradCall(self,x):
    if (self.lastx is None or np.all(x != self.lastx)):
      self.ObjCall(x)
    return self.result['eqcongrad']
   
  def InConCall(self,x):
    if (self.lastx is None or np.all(x != self.lastx)):
      self.ObjCall(x)
    if ('incon' in self.result):
      return self.result['incon']
    else:
      return None
   
  def InConGradCall(self,x):
    if (self.lastx == None or x != self.lastx):
      self.ObjCall(x)
    return self.result['incongrad'] 
    

class SimModel:
  def __init__(self,statebeg,stateend, stateMax,stateMin,controlMax,controlMin,
        #multipleshootingdim,                     
        fodestate, ffinalobjcon,
        fpathconstraints=None,
        constarrays = None,
        otherparamsMin=None,   #params to optimize, that are not states and not controls...
        otherparamsMax=None,
        T=1):
    self.statebeg = np.array(statebeg)                  #None means IS not fixed...
    self.stateend = np.array(stateend)
    for i in xrange(len(stateMax)):
      if (stateMax[i]>np.finfo('float32').max/20.0):
        stateMax[i] = np.finfo('float32').max/20.0
    for i in xrange(len(stateMin)):
      if (stateMin[i]<-1.0*np.finfo('float32').max/20.0):
        stateMin[i] = -1.0*np.finfo('float32').max/20.0
        
    self.stateMax=np.array(stateMax,dtype = 'float32') #scalary:
    self.stateMin=np.array(stateMin,dtype = 'float32')
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
          self.PackForOptimSize+=5
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
      packed = np.empty(self.PackForOptimSize,'float32')
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
          
    self.PackForOptim = PackForOptim
    
    def UnpackForOptim(Packed): #uses ndisc
      ret={} #scipy.optimize.minimize casts to float64.... we ned to cast back for theano
      #if we use f64 then np.empty is not needed for u,p
      ret['x']=np.empty([self.multipleshootingdim,(ndisc-2)+1],'float32')
      ret['u']=np.empty([self.controlfdim,ndisc-1],'float32')
      if (len(self.otherparamsMin.shape)>0):
        ret['p']=np.empty(self.otherparamsMin.shape[0],'float32')
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
     
      print self.controlfdim 
      print ret['u'].shape
           
      return ret
      
    self.UnpackForOptim = UnpackForOptim
    
  def RunOptim(self):
  
    #broadcasts can be:, eg. np.random.uniform([0,1],[1,10],(5,2))
    #def random_vectored(low, high,):
    #  return [random.uniform(low[i], high[i]) for i in xrange(low.size[0])]
    
    #dod specifikovat zacatek, pripadne ho vzit z minulyho behu...
    
    print self.multipleshootingdim
    print self.ndisc-2
    
    controlfrng = None
    if (self.controlfdim>0):
      controlfrng = np.empty([self.controlfdim, self.ndisc - 1], 'float32')
      for i in xrange(self.controlfdim):
        controlfrng[i,:] = np.random.uniform(low=self.controlMin[i], high=self.controlMax[i],size=(1,self.ndisc-1))
    randinitx = np.empty([self.multipleshootingdim,self.ndisc-2],'float32')  
    for i in xrange(self.multipleshootingdim):
      randinitx[i,:] = np.random.uniform(low=self.stateMin[i], high=self.stateMax[i], size=(1,self.ndisc-2))
    otherparamsinit = None
    if (len(self.otherparamsMin.shape)>0):
      otherparamsinit = np.random.uniform(low=self.otherparamsMin,high=self.otherparamsMax)
    x0 = self.PackForOptim(np.random.uniform(low=self.stateMin, high=self.stateMax),
                      randinitx,       #np.random.uniform(low=self.stateMin, high=self.stateMax, size=(self.multipleshootingdim,self.ndisc-2))
                      controlfrng,      #np.random.uniform(low=self.controlMin, high=self.controlMax,size=(self.controlfdim,self.ndisc-1)
                      otherparamsinit
                      )
    
    wlinitx = np.empty([self.multipleshootingdim,self.ndisc-2],'float32')
    for i in xrange(self.multipleshootingdim):
      wlinitx[i,:] = self.stateMin[i]
    otherparamsminp = None
    if (len(self.otherparamsMin.shape)>0):
      otherparamsminp = self.otherparamsMin
    controlfwl = None
    if (self.controlfdim > 0):
      controlfwl = np.empty([self.controlfdim, self.ndisc - 1], 'float32')
      for i in xrange(self.controlfdim):
        controlfwl[i, :] = self.controlMin[i]
    wL = self.PackForOptim(self.stateMin,
                      wlinitx, #np.tile(self.stateMin,(self.multipleshootingdim,self.ndisc-2)),      
                      controlfwl,#np.tile(self.controlMin,(self.controlfdim,self.ndisc-1)),
                      otherparamsminp
                      )
    
    wuinitx = np.empty([self.multipleshootingdim,self.ndisc-2],'float32')
    for i in xrange(self.multipleshootingdim):
      wuinitx[i,:] = self.stateMax[i]
    otherparamsmaxp = None   
    if (len(self.otherparamsMax.shape)>0):
      otherparamsmaxp = self.otherparamsMax
    controlfwu = None
    if (self.controlfdim > 0):
      controlfwu = np.empty([self.controlfdim, self.ndisc - 1], 'float32')
      for i in xrange(self.controlfdim):
        controlfwu[i, :] = self.controlMax[i]
    wU = self.PackForOptim(self.stateMax,
                      wuinitx, #np.tile(self.stateMax,(self.multipleshootingdim,self.ndisc-2)),
                      controlfwu, #np.tile(self.controlMax,(self.controlfdim,self.ndisc-1)),
                      otherparamsmaxp
                      )
    #print wL
    #print wU
    
    ParallelizSim = BuildTheanoModel(self.fodestate, self.ffinalobjcon, self.constarrays)#state,ModelFunkce)
    def ObjFromPacked(Inp):
      case=self.UnpackForOptim(Inp)                                                    #1) unpack for optim can be included to theano
      case['k']=self.odeintsteps
      case['Tmax']=self.T
      case.update(self.constarrays) #we add the calling constant arrays .. each iteration they are the same...
      computedsim = ParallelizSim(**case)
      #apply state-control-param path constraints:
      if (self.fpathconstraints != None):                                          #2) apply state constraints can be included to theano...
        for i in range(self.ndisc-1):
          thiscon = self.fpathconstraints(case['x'][:,i],case['u'][:,i],case['p'])
          if (thiscon['eqcon'] != None):
            computedsim['eqcon'].extend(thiscon['eqcon'])
          if (thiscon['incon'] != None):
            computedsim['incon'].extend(thiscon['incon'])
      #apply finalstate-param constraints:
      if (self.stateend is not None):
        if (self.stateend is callable):
          thiscon = self.stateend(computedsim['x'][:,-1],case['p'])
        else:    #is a vector of constants that we must target...
          thiscon = {}
          thiscon['eqcon']=[]
          for i in range(self.stateend.shape[0]):
            if (self.stateend[i] != None):
              thiscon['eqcon'].append(self.stateend[i]-computedsim['x'][i,-1])
        if (thiscon['eqcon'] != None and len(thiscon['eqcon'])>0):
          computedsim['eqcon'] = np.append(computedsim['eqcon'],thiscon['eqcon']) #computedsim['eqcon'].extend(thiscon['eqcon'])
        #if (thiscon['incon'] != None and len(thiscon['incon'])>0):
        #  computedsim['incon'] = np.append(computedsim['incon'],thiscon['eqcon']) #computedsim['incon'].extend(thiscon['incon'])
        return computedsim
      #apply multiple shooting algorithm constraints
      #already done, parallelized.... computedsim['eqcon'].extend()
            
    #v kazdy iteraci optimalizatoru
    #calleval.ObjCall({'x': konkretni hodnota x, 'u': konretni hodnota u, 'k': pocet integracnich iteraci})
    calleval = ModelMultiEval(ObjFromPacked)     #object remembering all that we have computed to not call sim again for constraints...

    haseqcon = (calleval.EqConCall(x0) is not None)
    hasincon = (calleval.InConCall(x0) is not None)
    if (haseqcon or hasincon):
      xconstr = []
      if (haseqcon):
        xconstr.append({'type': 'eq', 'fun': calleval.EqConCall})
      if (hasincon):
        xconstr.append( {'type': 'ineq', 'fun': calleval.InConCall})
    else:
      xconstr = () #empty iterable...

    res = scipy.optimize.minimize(fun = calleval.ObjCall,
        x0=x0, args=(), method='SLSQP', jac=False, hess=None, hessp=None,
        bounds=zip(wL,wU),     #dod je zip spravne?
        constraints=xconstr,
        tol=1e-9, callback=None,       
        options={'maxiter': self.maxoptimizers, 'disp':True})
        #http://stackoverflow.com/questions/23476152/dynamically-writing-the-objective-function-and-constraints-for-scipy-optimize-mihttp://stackoverflow.com/questions/23476152/dynamically-writing-the-objective-function-and-constraints-for-scipy-optimize-mi
    
    if (not res.success):
      print(res.message)
    print('Objective function at minimum: (computed by theano) ')
    print(res.fun)
    if (hasattr(res,'maxcv')):
      print('Constraint violation (computed by theano) ')
      print(res.maxcv)
    found = UnpackForOptim(res.x)
    return found
    #found.x - states x (ndisc-2) ....states    (incl. beginning)
    #found.u - controls x (ndisc-2) ....controls
    #found.p - params   
    
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
    def integstate(x,t):
      u=Sim['u'][:,floor(t/Tmax)]
      return state(x,t,u)
     
    #spojite: 
    sol=odeint(integstate, init, tspace)
    for i in range(init.shape[1]):
      plt.plot(tspace, sol[:,i], color='b')
      #plt.plot(tspace, sol[:,1], color='r')
      #plt.plot(tspace, sol[:,2], color='r')
    #plt.show()
    
    #nespojite
    for nd in range(self.ndisc):
      init = Sim['x'][:,nd]
      tspace= np.linspace((nd/self.ndisc)*Tmax,((nd+1)/self.ndisc)*Tmax,100)  
      sol=odeint(integstate, init, tspace)
      for i in range(init.shape[1]):
        plt.plot(tspace, sol[:,i], color='gray')
    
    objode = ModelFunkce(Sim[:,-1]) #objektivni funkce, tentokrat spocteno numericky bez theana a bez multipleshootingu
    print('Objective function at minimum - without Multiple Shooting possible discontinuities: (computed by odeint ')
    print(objode)  
    
    plt.show()
    
        #dispopt( x0, ts, wopt,Data, optODE,-1 );
        #dlmwrite('controldata',wopt);
    


"""
def rk4(f, xvinit, Tmax, N):
  T = np.linspace(0, Tmax, N+1)
  xv = np.zeros( (len(T), len(xvinit)) )
  xv[0] = xvinit
  h = Tmax / N
  for i in range(N):
      k1 = f(xv[i])
      k2 = f(xv[i] + h/2.0*k1)
      k3 = f(xv[i] + h/2.0*k2)
      k4 = f(xv[i] + h*k3)
      xv[i+1] = xv[i] + h/6.0 *( k1 + 2*k2 + 2*k3 + k4)
  return T, xv
  
...
slsqp...  

-beginning states can have bigger dimension than constraints and discretized states- that means that some states are not meant to be discretized

"""
def theano_inner_rk4_step(#accum:                               #i_step je integer...
                          i_step,accum,                         #accum je matice - pocet dimenzi vysledny funkce krat pocet bodu ve kterejch integrujeme po krivce
                          #menici se:                          
                          #pevny paramtery: 
                          Tmax, #scalar
                          Index,#vector
                          Int_steps_total,
                          f):
  #integracni casy: 
  fshape = T.cast(Index.shape[0],'float32')      
  Tim_t = Index * Tmax / fshape #vektor (index)0,1,2,3,4,5,6,7,.....  -> vektor (tim_t)0,1/(n*Tmax) .... n/n * Tmax
  Tim_t = Tim_t+ Tmax*(i_step / (fshape*Int_steps_total))                 #elemwise-posunuto na integracni step ted...
  #integracni krok:
  t_step =  (Tmax / fshape) / Int_steps_total        #scalar
  
  #accum - states x (ndisc-1)             tzn states x xbegs.shape[1]
  #Tim_t -                                tzn 1      x xbegs.shape[1]          z theano.tensor.arange(x_begs.shape[1]).dimshuffle(1, 0)                      
                             
  k1 = f(accum,Tim_t)                                                   #y'=f(y,t) (vicedim fce...) #aplikuj funkci PO SLOUPCICH
  k2 = f(accum + t_step*0.5*k1,Tim_t+0.5*t_step)
  k3 = f(accum + t_step*0.5*k2,Tim_t+0.5*t_step)
  k4 = f(accum + t_step*k3,Tim_t+t_step)
  return i_step+1,T.cast(accum + t_step/6.0 *( k1 + 2*k2 + 2*k3 + k4),'float32')
  #return i_step+1,T.cast(accum + t_step *f(accum,Tim_t),'float32')                       #euler
            
def BuildTheanoIntegrator(f):                         #dod  
  Tmax = T.scalar("Tmax")       
  k = T.iscalar("k")                #pocet iteraci
  x_begs = T.matrix("x_begs")       #zacatky vsech integracnich mist...  (vicedim funkce, proto matice) --------parametr
  #u_function = T.matrix("u_function")   #parametrizovana ridici funkce                                      --------parametr
  #f = definovanatheano expression
  
  # Symbolic description of the result
  # f vstupuje pri kompilaci.... ostatni jako non_sequences... (DOD pridat f jako nonsequence?)
  pIndex = T.cast(theano.tensor.stack(theano.tensor.arange(x_begs.shape[1])).dimshuffle(1, 0),'float32')
  result, updates = theano.scan(fn=lambda _i,_accum,_Tmax,_Index,_k :  theano_inner_rk4_step(_i,_accum,_Tmax,_Index,_k,f),
                                outputs_info=[0,x_begs],#[T.zeros(1,1),x_begs],
                                #sequences=
                                non_sequences=[Tmax,pIndex,T.cast(k,'float32')],
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


  xret = {'numsteps_var': k, 'xbegs_var': x_begs, 'results_var': result, 'Tmax_var': Tmax}
  
  return xret
  
  # compiled function that returns A**k
  #power = theano.function(inputs=[A,k], outputs=result, updates=updates)
  #print(power(range(10),2))
  #print(power(range(10),4))

"""
-f should be a state function
-objcon should be a function that gets the final integrated states and outputs:
['obj']
['eqcon']
['ineqcon']
"""
  
def BuildTheanoModel(f,objcon,constarrays):

  #if (self.controlfdim <= 0):
  #  def finteg(accum,t):            #integrator inputs only accumulator vector and time scalar, so other parametrs we need to input NOW.
  #    return f(accum,t)
  #  integratorvars = BuildTheanoIntegrator(finteg)
  #else:
  u_function = T.matrix("u_function")   #parametrizovana ridici funkce
  p_params = T.vector("p_params")
  
  def finteg(accum,t):            #integrator inputs only accumulator vector and time scalar, so other parametrs we need to input NOW.
    listres = f(accum,t[0],u_function,p_params)
    return theano.tensor.stack(*listres) #f does not need to use theano notation, can return [a,b]
  integratorvars = BuildTheanoIntegrator(finteg)

  #integratorvars['results_var'][:,-1] shoud be "all state variables; at their last point"
  #into this function the objcon goes for all computed path endings to get all possible custom path constraints
  result = objcon(integratorvars['results_var'],u_function,p_params)                       #moznost dat pouze koncovy body uz tady
  #objective function must be evaluated & we do consider objective function at the last point. (multipleshooting says that the continuity is handled by the beginnings-endings of path)
  result['obj']=result['obj'][-1]
  #smerujeme k  
  #return self.result['obj']
  #  return self.result['eqcon'] 
  #  return self.result['incon']                    #dod posun se o jedna
  #numpoints = integratorvars['xbegs_var'].shape[1]
  #if (result['eqcon'] == None):
  #  result['eqcon'] = []
  #result['eqcon'] = result['eqcon'].extend(
  ##dod tohle pro kazdou slozku vektorovy funkce az na tu posledni, ktera integruje cenu a scita se sama...
  ##je tu moznost z toho udelat vic eq constraints, nebo to vsechno sumovat do jedny... zatim sumujeme do jedny...
  #T.sub(integratorvars['xbegs_var'][1:numpoints],integratorvars['results_var'][0:numpoints-1])) #vezmeme ze je VIC rovnic
  #for ipoint in range(0,numpoints.eval()-1): #max is numpoints-2
  #  integratorvars['xbegs_var'][ipoint+1]-integratorvars['results_var'][ipoint]) #vezmeme ze je VIC rovnic
  
  #ends = integratorvars['results_var'][:,1:numpoints] ###############chci to rollovat po axis CASOVY - shape[1]] (ne po shape[0])
  #begs = integratorvars['xbegs_var'][:,0:numpoints-1]
  ends = integratorvars['results_var'][:,1:-1] ###############chci to rollovat po axis CASOVY - shape[1]] (ne po shape[0])
  begs = integratorvars['xbegs_var'][:,0:-2]
  if (result['eqcon'] == None):
    result['eqcon'] = T.flatten(ends-begs)
  else:
    result['eqcon'] = T.concatenate(result['eqcon'],T.flatten(ends-begs))
    
  theanoresult={}
  for key in result:
    if (result[key] is not None):
      theanoresult[key] = result[key] #Out(variable = result[key])#, name = key) variant #2

  inputsarray = [In(integratorvars['xbegs_var'], name='x'),
   In(u_function, name='u'),
   In(p_params, name='p'),
   In(integratorvars['numsteps_var'], name='k'),
   In(integratorvars['Tmax_var'], name='Tmax')]
  if constarray is not None:
    if isinstance(constarray,dict):
      theano_constarryas = []
      for key in constarrays.keys():
        theano_constarryas[key] = T.matrix("constarrays_" + key)
        inputsarray.add(In(theano_constarrays[key], name = key))
    #else:#error
    
      
  
  theano.config.on_unused_input = 'warn'
  objective_call = theano.function(inputs=inputsarray,
                              outputs=theanoresult,#updates=None,
                              allow_input_downcast = True,                                             #todo try false and see where the downcast originates.
                              on_unused_input='warn'          #u_function for example...
                              )                 #http://deeplearning.net/software/theano/library/compile/io.html
                   #f a list of Variable or Out instances is given as argument, then the compiled function will return a list of their values. 
  
  #def rtnf(x,u,p,k,Tmax):
  #  return objective_call(x=T.cast(x,'float32'),u=T.cast(u,'float32'),p=T.cast(p,'float32'),k=k,Tmax=T.cast(Tmax,'float32'))
  
  return objective_call  

#pouziti:
#def
#jednou
#calleval = ModelMultiEval(BuildTheanoModel(f,objcon))
#v kazdy iteraci optimalizatoru
#calleval.ObjCall({'x': konkretni hodnota x, 'u': konretni hodnota u, 'k': pocet integracnich iteraci})
#tohle je nasrel, konkretne je tam nesrovnalost mezi poctem promennejch
# - objcall bere jedno x a funkce z buildtheanomodel bere 3 [xbegs,ufunction,pocetinteg]
#xbegs je parametr kde se zacina, u func je jaka je tam kontrolni funkce a oboji se optimalizuje!