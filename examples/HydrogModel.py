import Solv as solv
import copy
from theano.compile.io import In  
import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import theano.tensor as T
from theano.ifelse import ifelse
import theano
from theano import shared

import copy
from theano.compile.io import In
from theano.compile.io import Out  
import numpy as np
import scipy as sp
import theano.tensor as T
import theano
from scipy.integrate import odeint
import matplotlib

def main():
    # ---------------specific setting:------------------------------------------------------------------
    # ---------------consts------------------------------------------------------------------

    timescales = 60.0 * 60.0
    powerscales = 0.00001

    ModelData = {
        'TimeScales': timescales,
        # reasons: not scaled time to hours it would take years to integrate AND ca significantly improve stability by right scaling
        'ControlScales': 100,
        # reasons: scaling can improve speed (cos algorithm step size is 1) AND can affect stability by the right scaling.
        'cSellPower': 100.0 *powerscales * timescales,
        'cBuyPower': 200.0 *powerscales * timescales,
        'cSellFinalPower': 150.0*powerscales,  # in order for algorithm not to try to sell all the power before final time.
        'cBattEfficiency': 0.9,  # now same for charge and discharge. TODO - change.
        'cH2Efficiency': 0.91,
        'cBattSigma': 0.001 * timescales,
        'cLogBegin': 8,  # 12,#2-logaritmus poctu stavu ve kterejch zacit
        'cLogEnd': 8,  # 12,#2-logaritmus poctu stavu ve kterejch zkoncit
        'numcontrols': 2,
        'numstates': 2,
        't_start ': 0.0,
        # 1.dataset
        # 't_end   ':  365*24*60*60/'TimeScales,#*60*1,#60*60*24,#seconds
        # 2.dataset
        # 't_end   ':  181*24*60*60/'TimeScales,#*60*1,#60*60*24,#seconds
        # 2. dataset one week
        't_end': 7 * 24 * 60 * 60 / timescales,  # *60*1,#60*60*24,#seconds
        'Wbattmax': 7920000,  # JOULES in battery                 #....2,2kWh (ujv)
        'Wfcmax': 0.0514131239,  # mol*n_e*faradayconstant of hydrogen    #ujv 10kg vodiku
        'Wbattinit': 0.0,  # 'Wbattmax/2.0,  #initial charge in battery
        # n_2 ':  2 electrons exchanged
        # F':  96485,3399 A*sec/mol
        # 10000 grams hydrogen ':  9921,22546977 moles
        # (ujv) 10kg hydrogen storage
        # 9921,22546977/2/96485,3399 ':  0.0514131239
        # usually batteries with 2,2kwh - https://library.e.abb.com/public/abf030c96ecac50d85257e1b00730610/REACT-3.6-4.6_BCD.00386_EN_RevB.pdf
        'Pbattplusmax': 3000 * timescales,  # joules per second maximal current of power to battery
        'Pbattminusmax': -3000 * timescales,  # joules per second maximal current of power FROM battery
        'Ph2plusmax': 0.00025910672 * timescales,
    # Ampers*FaradayConstant*n_e - maximal current to hydrogen                             #todo
        'Ph2minusmax': -0.00042493502 * timescales,  # Ampers*FaradayConstant*n_e - maximal current FROM hydrogen
        # ujv max fuel cell output 82A -> 0.00042493502 A*F*n_e (n_e ':  2 for hydrogen)
        # the goal is model for CURRENT
        # fumapem_fc_and electrolyz -> from that we do have the curves, so I will use maximal current input also - max 2A/cm2
        # 0.0025,#m^2, area of electrolyzer electrode
        # 2*0.0025*(10000) [Ampers] / (  96485,3399*2)  [FaradayConstant*n_e]   ': 0.00025910672
        'PowerConditionsTimeScale': 60 * 60,  # ':  hours
        # 'PowerConditions ':  dlmread(''txt'),#readtable(''txt','Delimiter',' ',     'Format','#f#f'),
        ##'PowerConditions ': powerconditions,
        # in fact it is NOW ':  PVinput - Power Demand
        'ElecNumCells': 2,
        'ElecVrev': 1.29,
        'ElecA': 0.0025,  # m^2, area of electrolyzer electrode
        'Elecr1': 0.08509,
        'Elecr2': 0.002397,
        'Elecs1': 3.357e-07,
        'Elecs2': 3.606e-07,
        'Elecs3': 3.986e-07,
        'Elect1': 0.0002209,
        'Elect2': 0.008437,
        'Elect3': 0.008486,
        'ElecTemp': 298.15,  # 25 celsius
        'ElecFefficiency': 0.85,  # faradays efficiency.
        'CellVrev': 1.29,
        'CellA': 0.0025,  # m^2, area of electrolyzer electrode
        'Celli0': np.exp(1.0),  # cant fit. anyway #CellA': 0.0025,#25 cm^2 area, at 65 Celsius, 1 cell,
        'CellB': 0.5172,  # Tafelparam
        'CellAt': 0.004908,  # tafel slope
        'Cellr': 0.0001521,  # resistance in Ohm Cm^2
        'Cellm': 0.09116,  # overvoltage parameters due to mass limit transports.
        'Celll': -0.0083,
        'CellFInvefficiency': 1.0 / 0.89,
        'CellNumCell': 2,
    }
    PowerConditions = np.loadtxt('examples\data2.txt', delimiter=' ')  # dlmread('data2.txt')
    PowerConditions[:, 0] = PowerConditions[:, 0] * (ModelData['PowerConditionsTimeScale'] / timescales)
    PowerConditions[:, 1] = PowerConditions[:, 1] * (-1.0)  # data are for electricity consumption. So make it electricity generation.
    ModelData['PowerConditions'] = PowerConditions

    # -----------------------------------------------------------------------state eq:
    # dod mozna rozmery
    # def state( t, x, w, ks, ndisc, ModelData ):
    # as accum,t,u
    def state(x, t, w, p,dataarrays):
        # dx=statesens(t,x,w,ks,ModelData); for later with sensitivity
        # dx=T.zeros(ModelData['numstates']+1,1);#column of state variables plus cost variable
        dx = [T.vector(),T.vector(),T.vector()]
        #bc = np.interp(t, ModelData['PowerConditions'][:, 0],ModelData['PowerConditions'][:, 1])  # dod zvladne tohle theano graf?
        #tl = np.clamp(np.floor(t),0,ModelData['PowerConditions'].shape[0]-1) #thanks to python indexing it will go around cyclically
        tl = T.cast(T.floor(t),'int32')  # thanks to python indexing it will go around cyclically
        #bc = dataarrays['PowerConditions'][tl] * (t-tl) + dataarrays['PowerConditions'][T.cast(tl+1,'int32')]* (1-t+tl)
        PowerCond = shared(ModelData['PowerConditions'][:,1])
        bc = (PowerCond[tl] * (t - tl) + PowerCond[T.cast(tl + 1, 'int32')] * (1 - t + tl))
        #xPWO = bc
        ubatt = (w[0] * ModelData['ControlScales'])
        #if (ubatt >= 0):  # into batt % w(ks,1) = ubatt
        #    xPWO = xPWO - ubatt
        #    dx[0] = ubatt * ModelData['cBattEfficiency'] - ModelData['cBattSigma'] * x[0]  # x1'  for battery
        #else:
        #    xPWO = xPWO - ubatt
        #    dx[0] = ubatt - ModelData['cBattSigma'] * x[0]
        #xPWO = xPWO - ubatt
        dx[0] = (T.switch(T.ge(ubatt,0.0),ubatt * ModelData['cBattEfficiency'] - ModelData['cBattSigma'] * x[0],  # x1'  for battery
          ubatt - ModelData['cBattSigma'] * x[0]))

        ufc = (w[1] * ModelData['ControlScales'])

        """if (ufc >= 0):  # into h2 w(ks,2) = ucell    "one empirical model" - electrolyzer
            xPWO = xPWO - ufc * (
            ModelData['ElecVrev'] + (ModelData['Elecr1'] + ModelData['Elecr2'] * ModelData['ElecTemp']) *
            (ufc / ModelData['ElecA']) +
            (ModelData['Elecs1'] + ModelData['Elecs2'] * ModelData['ElecTemp'] + ModelData['Elecs3'] * ModelData[
                'ElecTemp'] ** 2) *
            T.log(1 + (ufc / ModelData['ElecA']) * (
            ModelData['Elect1'] + ModelData['Elect2'] * ModelData['ElecTemp'] + ModelData['Elect3'] *
            ModelData['ElecTemp'] ** 2)))
            dx[1] = ModelData['ElecNumCells'] * ufc * ModelData['ElecFefficiency']
        else:  # - Fuelcell .. the parameter w(ks,2) is in this case the CURRENT Ampers*FaradayConstant*n_e
            # produced power:...............ufc is the same current through all
            # the cells
            areacurrent = (-ufc / ModelData['CellA'])  # inside the cell
            xPWO = xPWO - ModelData['CellNumCell'] * ufc * (
            ModelData['CellVrev'] + ModelData['CellB'] * T.log(ModelData['Celli0'])
            - ModelData['CellAt'] * T.log(areacurrent) - areacurrent * ModelData['Cellr'] +
            ModelData['Cellm'] * T.exp(areacurrent * ModelData['Celll']))
            # eaten hydrogen in units of [H2 times n_e times FaradayCOnstant]:
            dx[1] = ufc * ModelData['CellNumCell'] * ModelData['CellFInvefficiency']
        """
        electrolyzerpwo = (ufc * (
            ModelData['ElecVrev'] + (ModelData['Elecr1'] + ModelData['Elecr2'] * ModelData['ElecTemp']) *
            (ufc / ModelData['ElecA']) +
            (ModelData['Elecs1'] + ModelData['Elecs2'] * ModelData['ElecTemp'] + ModelData['Elecs3'] * ModelData[
                'ElecTemp'] ** 2) *
            T.log(1 + T.maximum(ufc / ModelData['ElecA'],0.0)) * (
            ModelData['Elect1'] + ModelData['Elect2'] * ModelData['ElecTemp'] + ModelData['Elect3'] *
            ModelData['ElecTemp'] ** 2)))
        areacurrent = (-ufc / ModelData['CellA'])  # inside the cell
        fuelcellpwo = (ModelData['CellNumCell'] * ufc * (
          ModelData['CellVrev'] + ModelData['CellB'] * T.log(ModelData['Celli0'])
          - ModelData['CellAt'] * T.log(T.maximum(areacurrent,0.0001)) - areacurrent * ModelData['Cellr'] +
          ModelData['Cellm'] * T.exp(areacurrent * ModelData['Celll'])))

        xPWO = bc - ubatt - T.switch(T.ge(ufc, 0.0), electrolyzerpwo, fuelcellpwo)
        
        dx[1] = T.switch(T.ge(ufc,0.0),ModelData['ElecNumCells'] * ufc * ModelData['ElecFefficiency'],
                       ufc * ModelData['CellNumCell'] * ModelData['CellFInvefficiency'])
        
        #if (xPWO >= 0):
        #    dx[2] = xPWO * ModelData['cSellPower']
        #else:
        #    dx[2] = xPWO * ModelData['cBuyPower']
        dx[2] = T.switch(T.ge(xPWO,0.0),xPWO * ModelData['cSellPower'],xPWO * ModelData['cBuyPower'])

        return dx

    def ModelFunkce(sim, control, params):
        # control is a matrix (controldim x (discretization points-1)), sim is a vector at the final state.

        # ret={obj: None, objgrad: None, eqcon: None, eqcongrad: None, incon: None, incongrad:None}
        ret = {}
        # sim = fun( x0, ts, ws)
        # simulation result using given controls ws and souhld return a function to minimize
        ret['obj'] = -sim[2] - ModelData['cSellFinalPower'] * (sim[0] + sim[1] * ModelData[
            'CellFInvefficiency'])  # power sold on our way plus what we managed to store in fuelcells and batteries
        # we want to maximize cost, so return minus cost to minimize

        # def ctr( x0, ts, ws,ModelData ):
        # ndisc=size(ts,2)-1#ts is row vector
        # [f,alldata] = fun( x0, ts, ws,ModelData )
        # sloupec vrqceno celkem ubatt,ucell,cost,...
        # maxubatt,maxucell,maxcost,minubatt,minucell,mincost .. maxima co sme potkali po ceste
        # single shooting: zadne constraints:
        # ceq = []# =0                     #radek .. toto ma byt rovno nule
        # multiple shooting:
        # for inumstates in range(1,ModelData['numstates']):
        #   xquaa=alldata(2:(ndisc),inumstates) #now it is a row ... alldata is from 1 to ndisc+1 ... at 1 it is x0 at ndisc+1 it is the value at the end of the last timeslot
        #   xquab=ws((ModelData['numcontrols']*ndisc+(inumstates-1)*(ndisc-1)+1)
        #       :(ModelData['numcontrols']*ndisc+(inumstates)*(ndisc-1)),1)'
        #   ceq = [ceq,(xquaa-xquab).*(xquaa-xquab)]#begin at the next time slot where you ended in the previous time slot

        ret['eqcon'] = []  # other things that need to be equal to zero, comes from problem formulation
        ret['incon'] = []  # other things that need to be greater than zero, comes from problem formulation
        # c = [f(4)-ModelData.Wbattmaxf(5)-ModelData.Wfcmax...
        #    -f(7)-f(8)]' #\leq 0                             #radek ... toto ma byt mensi nez nula

        #inequality constraints - taken care by stateMax, stateMin but not for the last node... ... finito, now taken care by object..
        #ret['incon'] = [-sim[0], -sim[1], sim[0] - ModelData['Wbattmax'], sim[1] - ModelData['Wfcmax']]
        return ret

    x0 = [ ModelData['Wbattinit'], 0]#,0 ]
    xend=[None,None]#,None]
    print "calling simmodel..."
    OptimSim = solv.SimModel(x0,xend,
                      stateMax=[ModelData['Wbattmax'],ModelData['Wfcmax']],#,np.inf],
                      stateMin=[0,0],#,-np.inf],
                      laststatesum = True, #last state returned by odesim will be a cost function to be summed up and NOT discretized... #dod rict ze stavy popisujeme dva ale ze funkce bude pouzivat dalsi 1 na ukladani ceny...
                      controlMax=[ModelData['Pbattplusmax']/ModelData['ControlScales'],ModelData['Ph2plusmax']/ModelData['ControlScales']],
                      controlMin=[ModelData['Pbattminusmax']/ModelData['ControlScales'],ModelData['Ph2minusmax']/ModelData['ControlScales']],
                      fodestate=state,
                      ffinalobjcon=ModelFunkce,
                      constarrays= None, #constarrays = {'PowerConditions': ModelData['PowerConditions'][:, 1]},
                      fpathconstraints=None,              #others than min max
                      otherparamsMin=None,   #params to optimize, that are not states and not controls...
                      otherparamsMax=None,
                      T=ModelData['t_end'])
    print "calling gen"
    OptimSim.GenForDiscretization(ndisc=32,maxoptimizers=1000,odeintsteps=1000)
    print "running optim"
    res=OptimSim.RunOptim()
    OptimSim.DrawResults(res)

if __name__ == '__main__':
    main()

        
"""
def main():
  #
  # MULTIPLE SHOOTING
  #

  # Options for NLP Solvers
  #optODE = odeset( 'RelTol', 1e-8, 'AbsTol', 1e-8 )'GradObj','on','DerivativeCheck','on','HessUpdate','bfgs','Diagnostic','on','LineSearchType','cubicpoly',
  #GradConstr on
  # Time Horizon and Initial State
  #Data.t_start % Initial time
  #Data.t_end % Final time

  x0 = [ Data.Wbattinit 0 0 ] # Initial states+1 state for cost variable

  ndisc = 2**(ModelData['cLogBegin']-1)
 
  w0 = rand(ModelData['numcontrols']*ndisc+ModelData['numstates']*(ndisc-1),1) #and each control has startvalues (one less than controls) - according to multiple shooting
  for _is in range(ModelData['cLogBegin']:ModelData['cLogEnd']):
    ndisc = 2*ndisc # Number of stages: ns = 2, 4, 8, 16, and 32
    ts = [ModelData['t_start']:(ModelData['t_end']-ModelData['t_start'])/ndisc:ModelData['t_end']] # Time stages (equipartition)
    
    # Initial Guess and Bounds for the Parameters
    wold=w0
    def PackParams(x,u):
      return np.concatenate(x,u)
      
    def UnpackParams(p,ndisc):
      return {'x': p[0:(ModelData['numcontrols']*ndisc)], 'u': p[ModelData['numcontrols']*ndisc+1:]}             #dod coz delat p i u vicedimenzionalni?
    
    #if (False):                               
    #  w0 = dlmread('controldataiter65')
    #else:
    w0 = zeros(ModelData['numcontrols']*ndisc+ModelData['numstates']*(ndisc-1),1)
    for inumcontrols in range(1,(ModelData['numcontrols'])):
       for jsi in range(1,ndisc):#use previous results as initial guess.
            w0(jsi+(inumcontrols-1)*ModelData['numcontrols'],1) = wold(1+floor((jsi-1)/2)+(inumcontrols-1)*(ndisc/2),1)
            
    for inumstates in range(1,(Data.numstates)):
       for jsi in range(2,(ndisc-1)): #ssame for initial points
            w0(ndisc*ModelData['numcontrols']+jsi+(ndisc-1)*(inumstates-1),1) = 
                wold(1+floor((jsi-1-1)/2)+(ndisc/2)*ModelData['numcontrols']+(ndisc/2-1)*(inumstates-1),1)
                       
       w0(ndisc*ModelData['numcontrols']+1+(ndisc-1)*(inumstates-1),1) = 0.0 #new appeared at the begining - set to initial value .. or zero it doesnot matter that much
  
    #bounds for control variables: (conrolScales are multiplied in the equation so here we must divide)
    wL = ModelData['Pbattminusmax']/ModelData['ControlScales']*ones(ModelData['numcontrols']*ndisc+ModelData['numstates']*(ndisc-1),1) #parameter (=control) bounds in each tie stage
    wU = ModelData['Pbattplusmax']/ModelData['ControlScales']*ones(ModelData['numcontrols']*ndisc+ModelData['numstates']*(ndisc-1),1)
    wL((ndisc+1):(2*ndisc),1)=ModelData['Ph2minusmax']/ModelData['ControlScales']*ones(2*ndisc-(ndisc+1)+1,1)
    wU((ndisc+1):(2*ndisc),1)=ModelData['Ph2plusmax']/ModelData['ControlScales']*ones(2*ndisc-(ndisc+1)+1,1) 
    ##bounds for states - on multipleshooting it equals bounds for control
    ##variables:
    ##bounds on the beginnings are just bounds on state variables:
    wL((2*ndisc+1):(2*ndisc+(ndisc-1)),1)=0*zeros((2*ndisc+(ndisc-1)-(2*ndisc+1)+1),1)
    wU((2*ndisc+1):(2*ndisc+(ndisc-1)),1)=ModelData['Wbattmax']*ones((2*ndisc+(ndisc-1)-(2*ndisc+1)+1),1)
    wL((2*ndisc+(ndisc-1)+1):(2*ndisc+2*(ndisc-1)),1)=0*zeros((2*ndisc+2*(ndisc-1))-(2*ndisc+(ndisc-1)+1)+1,1)
    wU((2*ndisc+(ndisc-1)+1):(2*ndisc+2*(ndisc-1)),1)=ModelData['Wfcmax']*ones((2*ndisc+2*(ndisc-1))-(2*ndisc+(ndisc-1)+1)+1,1)
    #lze predelat na wL = PackParams(...,...) wU je pack params...
   
   
    #optNLP = optimset('Algorithm','sqp', 'LargeScale', 'off', 'GradObj', 'off', 'GradConstr', 'off',...
    #'DerivativeCheck', 'off', 'Display', 'iter-detailed', 'TolX', 1e-11,...
    #'TolFun', 1e-9, 'TolCon', 1e-9, 'MaxFunEval', 8000000,...
    #'DiffMinChange', 1e-5,...
    #'OutputFcn',@(x,optimValues,state)nlpoutfun(x0,x,optimValues,state, ts, Data, optODE))#'PlotFcns'
  
    # Sequential Approach of Dynamic Optimization
    #[ wopt ] = fmincon( @(ws)obj(x0,ts,ws,Data), w0, [], [], [], [], wL, wU,...
    #@(ws)ctr(x0,ts,ws,Data), optNLP)
     
     #currently there are constraints only on control variables, and on control
     #variables representing beggining points of integration. And no other
     #constraints on states. So another implementation might be to also add
     #constraints on states. Or also while integrating taking care of max and
     #min values of states. Afraid it will damage precision, i did only the
     #constraints on the node points of discretization.
  
    calleval = ModelMultiEval(BuildTheanoModel(state,ModelFunkce))
    IntegratorSteps = 10000           #druhej parametr je ndisc...
    def ObjFromPacked(Inp):
      p=UnpackParams(Inp,ndisc)
      p['k']=IntegratorSteps 
      return calleval.ObjCall(p)
      #v kazdy iteraci optimalizatoru
      #calleval.ObjCall({'x': konkretni hodnota x, 'u': konretni hodnota u, 'k': pocet integracnich iteraci})
  
    res = scipy.optimize.minimize(fun = ObjFromPacked,
        x0=w0, args=(), method='SLSQP', jac=False, hess=None, hessp=None,
        bounds=zip(wL,wU),     #dod je zip spravne?
        constraints=({type: 'eq' fun: calleval.EqConCall},{ type: 'ineq', fun: calleval.InConCall}),
        tol=1e-9, callback=None,       
        options={'maxiter': 1000, 'disp':True})
        #http://stackoverflow.com/questions/23476152/dynamically-writing-the-objective-function-and-constraints-for-scipy-optimize-mihttp://stackoverflow.com/questions/23476152/dynamically-writing-the-objective-function-and-constraints-for-scipy-optimize-mi
    
    if (not res.success):
      print(res.message)
    print('Objective function at minimum: (computed by theano) ')
    print(res.fun)
    print('Constraint violation (computed by theano) ')
    print(res.maxcv)
    found = UnpackParams(res.x,ndisc)
    
    #-------------------------------------
    #for painting, compute the model and ODE's AGAIN numerically without gpu, plot them (and we can see if the numerical integration on gpu was precise or not...)
    #http://stackoverflow.com/questions/27820725/how-to-solve-diff-eq-using-scipy-integrate-odeint
    #def g(y, x):
    #    y0 = y[0]
    #    y1 = y[1]
    #    y2 = ((3*x+2)*y1 + (6*x-8)*y0)/(3*x-1)
    #    return y1, y2
    #jsou dve varianty jak integrovat - predpokladat spojitost, nebo nakreslit to co vyslo ev i s nespojitostma-zacinat forcyklovef 
    init = x0
    tspace= np.linspace(0,Tmax,100)
    def integstate(x,t)
      u=found['u'][floor(t/Tmax),:]
      return state(x,t,u)
      
    sol=odeint(integstate, init, tspace)
    plt.plot(tspace, sol[:,0], color='b')
    plt.plot(tspace, sol[:,1], color='r')
    plt.plot(tspace, sol[:,2], color='r')  
    
    objode = ModelFunkce(sol[-1,:]) #objektivni funkce, tentokrat spocteno numericky bez theana a bez multipleshootingu
    print('Objective function at minimum - without Multiple Shooting possible discontinuities: (computed by odeint ')
    print(objode)  
        #dispopt( x0, ts, wopt,Data, optODE,-1 )
        #dlmwrite('controldata',wopt)
"""        