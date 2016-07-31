import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams  

#optimalizace:
# co optimalizuju - vektor nejakejch hodnot.
# pro volani potrebuju:
# hodnoty parametru; co z nasledujiciho seznamu OUT potrebuju: ------------------------------------------------------ IN
# hodnotu cost function (a grad vzhledem k parametrum), hodnoty constraints-eq (jakobian), hodnoty constraints-ineq(jakobian), ------------ OUT
#
# takze pro optcontrol potrebuju state function a pocatecni hodnoty.

#transformace z problemu na multiple shooting:
#- k optimalizacnim parametrum problemu pridam diskretizacni body
#- k constraints pridam ze dalsi iterval zacina tam kde predchozi konci
#- funkce musi umet evaluovat kde konci intervaly
#--paralelizuje se vyhodnoceni na intervalech
#--funkcicka co dostane cely vstupni data (theano funkci), id svyho intervalku a zavola 
#-musim umet cost function zapsat jako (theano fci) rozlozeni na poc. bod., parametr (a zdrojovy data)    

#
#
#

#aby mohlo byt DAE je potreba pridat nejakou algebraickou cast

#"""

class MultipleShooter(object):
    def __init__(
    #optimalizacni metaparametry: muze byt null=[]
    optimOriginParams=None, #theano parametry, co optimalizujeme jeste bez diskretizace                                      # U
    optimOriginParamsStart=None, #bod kde zacit optimalizaci pro tyhle parametry, konstanta, cislo.                          # U
    
    # ridici parametry - diskretizovana kontrolni funkce
    controlParamsDim=1  #dimenze kontrolni funkce                                                                       # U
    discretizeNumber=64,#pocet kontrolnich bodu na ktery to chci rozdelit                                               # U        - z tohohle zjistuju dimenzionalitu X a U na kterou paralelizuju
    discretizeOptimStart=None,#specifikovany pocatecni hodnoty techhle parametru mat[controlParamsDim,discretizeNumber]   # U
    
    #cas 
    stateStartTime=0,
    stateEndTime=0,
    
    #rovnice stavu:   -tahle se bude integrovat prez cas!
    stateEquation,    #theano funkce stavovy rovnice [stavovepromenne]=stateEquation(optimOriginParams,ridicipromenne,x,t)#X,U
    stateDimension,   #konstanta kolik mame stavovych promennych. Rovno dimenzionalite toho co vraci stateEquation      #X - z tohohle zjistuju dimenzionalitu X
    startPointState,  #startPointState(optimOriginParams) theano funkce vstupu (konst pokud neoptimalizujeme startovni bod)  #X  
    
    #cost
    costFunction, #jako funkce finalniho stavu a optimOriginParams costFunction(optimOriginParams,Y)
    
    #theano funkce pro linearni a nelinearni constrainty
    funConEq, # funConEq(vysledny zintegrovany stav,optimOriginParams) ma byt == 0
    funConInEq, # funConInEq(vysledny zintegrovany stav,optimOriginParams) ma byt <=0    
    
    #ridici parametry a stavy musi byt vetsi mensi nez tyhle vektory cisel (po slozkach)
    controlsMin,                                                                                                        #U>
    controlsMax,        #mmch pouze jen v diskretizovanych bodech                                                       #U<
    statesMin,                                                                                                          #X>
    statesMax,                                                                                                          #X<    
    ):
        #dod check jestli jsou parametry spravne dimenzionality atp...
        self.ktimes=np.linspace(stateStartTime,stateEndTime,discretizeNumber,endpoint=False) #prvni bod je startovni, posledni je tesne predtim
        self.stateStartTime=stateStartTime
        self.stateEndTime=stateEndTime
        self.discretizeNumber=discretizeNumber
        
        #optim params:
        self.optimOriginParams = optimOriginParams
        self.optimOriginParamsStart = optimOriginParamsStart
        
        #U:        
        self.controlsMin=controlsMin
        self.controlsMax=controlsMax     #dod if nejsou specifikovany, generovat pocatecni ne nahodne v range ale s nulou
        if discretizeOptimStart is None:                       #if dim(discretizeOptimStart)[] is not () dod
            self.discretizeOptimStart = numpy.asarray(
                rng.uniform(
                    low=controlsMin,
                    high=controlsMax,
                    size=(discretizeNumber, stateDimension)
                ),
                dtype=theano.config.floatX
            )
        else:
            self.discretizeOptimStart=discretizeOptimStart
        
        self.U = theano.shared(value=self.discretizeOptimStart, name='U', borrow=True) #matice velikosti discretizeNumber x controlParamsDim
        
        #X:   #matice velikosti discretizeNumber-1 x stateDimension
        self.statesMin=statesMin
        self.statesMax=statesMax #dod if nejsou specifikovany, generovat pocatecni ne nahodne v range ale s nulou
        self.X_inits = numpy.asarray(
                rng.uniform(
                    low=statesMin,
                    high=statesMax,
                    size=(discretizeNumber-1, stateDimension)
                ),
                dtype=theano.config.floatX
            )
        self.X = theano.shared(value=self.X_inits, name='X', borrow=True)
        
        #theanofunkce: pocatecni bod. Muze vracet konstantu ale je to funcke proste.
        self.X0 = startPointState(optimOriginParams)
        
        #
        self.stateEquation=stateEquation
        self.costFunction=costFunction
        self.funConEq=funConEq
        self.funConInEq=funConInEq

        #http://www.nehalemlabs.net/prototype/blog/2013/10/17/solving-stochastic-differential-equations-with-theano/
        # a ted integratory ODEcek

        #self.beg_params=[]#multipleshooting .. je jich o jeden min, protoze startovni policko je dany jinac
        self.ceq=np.zeroes(stateDimension)        #za multipleshooting, expression...
        self.Y = []                      #list vektoru
        # diskretizace a krasa multipleshootingu - vylepseni parametrickyho prostoru:
        for i in xrange(0,discretizeNumber)                           
            if (i<=0):
                self.Y.append(self.funInterval(ktimes[i],ktimes,self.X0,self.U[:,i])
            else:
                self.Y.append(self.funInterval(ktimes[i],ktimes,self.X[i-1,:],self.U[:,i])  #fun() .append(thano variable to optimize);
            if (i<discretizeNumber-1):          #posledni nema ceq constraint
                self.ceq += self.Y[i]-self.X[i] #zacatky nasledujicich musi by rovny koncum momentalnich               #varianta sem de pridat kvadraticka funkce aby se mu lip derivovalo?
        
          #konecnej vycet veci co pouzit v optimalizacnim algorimtu:
        self.cost=self.costFunction(self.optimOriginParams,self.Y[-1])
        self.ConEq=self.finConEq(self.optimOriginParams,self.Y[-1])
        self.ConInEq=self.finConInEq(self.optimOriginParams,self.Y[-1])
        #self.X0
        #self.ceq
        #self.controlsMin=controlsMin
        #self.controlsMax=controlsMax
        #statesMin
        #statesMax
        
        #co optimalizuju:
        #U (diskretizovana vec)
        #X (promenny z multipleshootingalgoritmu)
        #self.optimOriginParams metaparametry
        
        #monitoring:
        #cost
        #Y[-1]
        self.ceqsum=T.sum(seq)
        self.userceqsum=T.sum(ConEq)
        self.usercineqsum=T.sum(ConInEq)
        
            
    
    def funInterval(self,timebeg,timeend,xstart,controlparams):
        #theano funkce co se aplikuje na prostor o dimenzi discretizeNumber a pak zbyly parametry
        #http://www.sciencedirect.com/science/article/pii/S1877050914002683
        #http://deeplearning.net/software/theano/library/scan.html#conditional-ending-of-scan
        #optimOriginParams,ridicipromenne,x,t
        
        dt_val=0.001
        total_steps=abs(timebeg-timeend)/dt_val # = T/dt
        
        #dt = T.scalar("dt")
      	#t = T.scalar("t")
      	##x0 = T.vector("x0")
      	#fparams = T.vector("fparams")
      	#y = T.vector("y")
        y=xstart #theano.shared(0.5*np.ones(num_samples, dtype='float32'))
        dt=dt_val
        t=timebeg
        fparams = self.optimOriginParams
        
        def solvstep(y,t,fparams,controlparams,dt):
            def fun(y,t):
                return self.stateEquation(fparams,controlparams,y,t)
            return (self.rk4step(fun,x0,t,dt),t+dt)                        #pricitam t+dt, muze nakonec zkoncit v trosku jinym bode bacha na to
        
        #create loop
      	#first symbolic loop with everything
      	(cout, updates) = theano.scan(fn=solvstep,
      									outputs_info=[y,t], #output shape
      									non_sequences=[fparams,controlparams,dt], #fixed parameters n, optimOriginParams, l, dt
      									n_steps=total_steps)                                                          #predelanim na metodu s ukoncovanim pujde dat lib. delka kroku.
                        
        #compile it
      	#sim = theano.function(inputs=[fparams, x0, dt], 
      	#					outputs=cout[-1], #last element to reduce
      	#					givens={y:xstart}, 
      	#					updates=updates,
      	#					allow_input_downcast=True)
      	#print "running sim..."
      	#start = time.clock()
      	#cout = sim(n0, k0, l0, dt0)
        
        return cout[-1]  #dod nebude problem s tim, ze vracim jak t tak y?
        
    def rk4step(self, f, x0, t, dt ):
    """Fourth-order Runge-Kutta method to solve x' = f(x,t) with x(t[0]) = x0.

        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.
    """

    #n = len( t )
    #x = numpy.array( [ x0 ] * n )
    #for i in xrange( n - 1 ):
        #h = t[i+1] - t[i]
        h = dt
        k1 = h * f( x0, t ) #f vraci dimenzi tutez jako x
        k2 = h * f( x0 + 0.5 * k1, t + 0.5 * h )
        k3 = h * f( x0 + 0.5 * k2, t + 0.5 * h )
        k4 = h * f( x0 + k3, t )
        y = x0 + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6.0
    return T.cast(y,theano.config.floatX)
        
#"""        
        
        
    
if __name__ == '__main__':
    #theano test:
    
    theano.config.assert_no_cpu_op='warn'
    
    def vectop(v):
        return v[0]+2*v[1]
    
    x = T.vector('x')
    
    test = theano.function(
        inputs=[x],
        outputs=,
        allow_input_downcast=True,
        on_unused_input='ignore'
    )
    
    xtoinput=numpy.zeros((2,2), dtype=theano.config.floatX, order='C')
    
    result = test(xtoinput)
    
    print result