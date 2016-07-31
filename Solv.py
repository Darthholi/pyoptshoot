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
"""
#---------------consts------------------------------------------------------------------
class ModelMultiEval:     #a class that simplyfies the cases wen obj, constraint and gradients are computed together. IF THE Objective is the first one to call everytime! (#otherwise can check lastx....)
  def __init__(self,
               MasterFun=None,#funkce parametru a vraci: {obj: , objgrad: , eqcon: , eqcongrad: , incon: , incongrad:}
               ):
  self.MasterFun=MasterFun
  
  def ObjCall(self,x):
    self.lastx=x
    self.result = MasterFun(x)
    return self.result['obj']
    
  def ObjGradCall(self,x):
    return self.result['objgrad']
    
  def EqConCall(self,x):
    return self.result['eqcon']
    
  def EqConGradCall(self,x):
    return self.result['eqcongrad']
   
  def InConCall(self,x):
    return self.result['incon']
   
  def InConGradCall(self,x):
    return self.result['incongrad'] 
        
  
#---------------consts------------------------------------------------------------------
timescales = 60*60

PowerConditions =  dlmread('data2.txt')
PowerConditions(:,1) = PowerConditions(:,1)*('PowerConditionsTimeScale/'TimeScales)
PowerConditions(:,2) = PowerConditions(:,1)*(-1),#data are for electricity consumption. So make it electricity generation.

ModelData = {
  'TimeScales': timescales,#reasons: not scaled time to hours it would take years to integrate AND ca significantly improve stability by right scaling
  'ControlScales': 100,#reasons: scaling can improve speed (cos algorithm step size is 1) AND can affect stability by the right scaling.
  'cSellPower': 100.0* timescales,
  'cBuyPower': 200.0* timescales,
  'cSellFinalPower': 150.0,#in order for algorithm not to try to sell all the power before final time.
  'cBattEfficiency': 0.9,     #now same for charge and discharge. TODO - change.
  'cH2Efficiency': 0.91,
  'cBattSigma': 0.001*timescales,
  'cLogBegin': 8, #12,#2-logaritmus poctu stavu ve kterejch zacit
  'cLogEnd': 8, #12,#2-logaritmus poctu stavu ve kterejch zkoncit
  'numcontrols': 2,
  'numstates': 2,
  't_start ':   0.0,
  #1.dataset
  #'t_end   ':  365*24*60*60/'TimeScales,#*60*1,#60*60*24,#seconds
  #2.dataset
  #'t_end   ':  181*24*60*60/'TimeScales,#*60*1,#60*60*24,#seconds
  #2. dataset one week
  't_end   ':  7*24*60*60/timescales, #*60*1,#60*60*24,#seconds
  'Wbattmax ':  7920000,#JOULES in battery                 #....2,2kWh (ujv)
  'Wfcmax ':  0.0514131239,#mol*n_e*faradayconstant of hydrogen    #ujv 10kg vodiku
  'Wbattinit ':  0.0,#'Wbattmax/2.0,  #initial charge in battery
  #n_2 ':  2 electrons exchanged
  #F':  96485,3399 A*sec/mol
  #10000 grams hydrogen ':  9921,22546977 moles
  #(ujv) 10kg hydrogen storage
  #9921,22546977/2/96485,3399 ':  0.0514131239
  #usually batteries with 2,2kwh - https://library.e.abb.com/public/abf030c96ecac50d85257e1b00730610/REACT-3.6-4.6_BCD.00386_EN_RevB.pdf
  'Pbattplusmax ':  3000*timescales, # joules per second maximal current of power to battery
  'Pbattminusmax ':  -3000*timescales, # joules per second maximal current of power FROM battery
  'Ph2plusmax ':  0.00025910672*timescales, # Ampers*FaradayConstant*n_e - maximal current to hydrogen                             #todo
  'Ph2minusmax ':  -0.00042493502*timescales, # Ampers*FaradayConstant*n_e - maximal current FROM hydrogen 
  #ujv max fuel cell output 82A -> 0.00042493502 A*F*n_e (n_e ':  2 for hydrogen)     
  #the goal is model for CURRENT   
  #fumapem_fc_and electrolyz -> from that we do have the curves, so I will use maximal current input also - max 2A/cm2
  #0.0025,#m^2, area of electrolyzer electrode
  #2*0.0025*(10000) [Ampers] / (  96485,3399*2)  [FaradayConstant*n_e]   ': 0.00025910672
  'PowerConditionsTimeScale': 60*60,# ':  hours
  #'PowerConditions ':  dlmread(''txt'),#readtable(''txt','Delimiter',' ',     'Format','#f#f'),
  'PowerConditions ':  powerconditions,
  #in fact it is NOW ':  PVinput - Power Demand
  'ElecNumCells': 2,
  'ElecVrev': 1.29,
  'ElecA': 0.0025,#m^2, area of electrolyzer electrode
  'Elecr1': 0.08509,
  'Elecr2': 0.002397,
  'Elecs1': 3.357e-07,
  'Elecs2': 3.606e-07,
  'Elecs3': 3.986e-07,
  'Elect1': 0.0002209,
  'Elect2': 0.008437,
  'Elect3': 0.008486,
  'ElecTemp': 298.15,# 25 celsius
  'ElecFefficiency': 0.85,#faradays efficiency.
  'CellVrev': 1.29,
  'CellA': 0.0025,#m^2, area of electrolyzer electrode
  'Celli0': exp(1),#cant fit. anyway #CellA': 0.0025,#25 cm^2 area, at 65 Celsius, 1 cell,
  'CellB': 0.5172,#Tafelparam
  'CellAt': 0.004908,#tafel slope
  'Cellr': 0.0001521,#resistance in Ohm Cm^2
  'Cellm': 0.09116,#overvoltage parameters due to mass limit transports.
  'Celll': -0.0083,
  'CellFInvefficiency': 1.0/0.89,
  'CellNumCell': 2,
  }


def theano_inner_rk4_step(#accum:                               #i_step je integer...
                          i_step,accum,                         #accum je matice - pocet dimenzi vysledny funkce krat pocet bodu ve kterejch integrujeme po krivce
                          #menici se:                          
                          #pevny paramtery: 
                          Tmax,Index,Int_steps_total,f):
  #integracni casy:       
  Tim_t = Index * Tmax / Index.size[0] #vektor 0,1,2,3,4,5,6,7,.....  -> vektor 0,1/(n*Tmax) .... n/n * Tmax
  Tim_t = Tim_t+ Tmax*(i_step / (Index.size[0]*Int_steps_total))
  #integracni krok:
  t_step =  (Tmax / Index.size[0]) / Int_steps_total   
                             
  k1 = f(accum,Tim_t)                                                   #y'=f(y,t) (vicedim fce...) #aplikuj funkci PO SLOUPCICH
  k2 = f(accum + t_step*0.5*k1,Tim_t+0.5*t_step)
  k3 = f(accum + t_step*0.5*k2,Tim_t+0.5*t_step)
  k4 = f(accum + t_step*k3,Tim_t+t_step)
  return i_step+1,accum + t_step/6.0 *( k1 + 2*k2 + 2*k3 + k4)
            
def BuildTheanoIntegrator(f):                         #dod         
  k = T.iscalar("k")                #pocet iteraci
  x_begs = T.matrix("x_begs")       #zacatky vsech integracnich mist...  (vicedim funkce, proto matice) --------parametr
  #u_function = T.matrix("u_function")   #parametrizovana ridici funkce                                      --------parametr
  #f = definovanatheano expression
  
  # Symbolic description of the result
  # f vstupuje pri kompilaci.... ostatni jako non_sequences... (DOD pridat f jako nonsequence?)
  result, updates = theano.scan(fn=lambda _i,_accum,_Tmax,_Index,_k :  theano_inner_rk4_step(_i,_accum,_Tmax,_Index,_k,f),
                                outputs_info=[T.zeros(1,1),x_begs],
                                #sequences=
                                non_sequences=[Tmax,theano.tensor.arange(x_begs.size[1]),k],
                                n_steps=k)
  
  # We only care about A**k, but scan has provided us with A**1 through A**k.
  # Discard the values that we don't care about. Scan is smart enough to
  # notice this and not waste memory saving them.
  #result [0] = k je pocet iteraci....
  #result [1] je to co chceme
  result = result[1]
  #result ma stejnej tvar jako x_begs, takze chceme
  #final_result = result               
  return {'numsteps_var:' k, 'xbegs_var': x_begs, 'integrated_results': result}
  # compiled function that returns A**k
  #power = theano.function(inputs=[A,k], outputs=result, updates=updates)
  #print(power(range(10),2))
  #print(power(range(10),4))
  
def BuildTheanoModel(f):                         #dod 

  def finteg(accum,t):            #integrator inputs only accumulator vector and time scalar, so other parametrs we need to input NOW.
    return f(accum,t,)
  integratorvars = BuildTheanoIntegrator(finteg)        
  u_function = T.matrix("u_function")   #parametrizovana ridici funkce  

#-----------------------------------------------------------------------state eq:

#def state( t, x, w, ks, ndisc, ModelData ):
#as accum,t,
def state( x, t, w, ks, ndisc, ModelData ):
  #dx=statesens(t,x,w,ks,ModelData); for later with sensitivity
  dx=T.zeros(ModelData['numstates']+1,1);#column of state variables plus cost variable
  bc = np.interp(t, ModelData['PowerConditions'][:,1],  ModelData['PowerConditions'][:,2])
  xPWO=bc;
  ubatt=w[ks,1]*ModelData['ControlScales'];
  if (ubatt>=0): #into batt; % w(ks,1) = ubatt
      xPWO=xPWO-ubatt;
      dx[1] = ubatt*ModelData['cBattEfficiency'] - ModelData['cBattSigma']*x[1];  #x1' ; for battery
  else:
      xPWO=xPWO-ubatt;
      dx[1] = ubatt - ModelData['cBattSigma']*x[1];
    
  ufc=w(ks+1*ndisc,1)*ModelData['ControlScales'];
    
  if (ufc>=0): #into h2; w(ks,2) = ucell    "one empirical model" - electrolyzer
      xPWO=xPWO-ufc*(ModelData['ElecVrev']+(ModelData['Elecr1']+ModelData['Elecr2']*ModelData['ElecTemp'])*
          (ufc/ModelData['ElecA'])+
          (ModelData['Elecs1']+ModelData['Elecs2']*ModelData['ElecTemp']+ModelData['Elecs3']*ModelData['ElecTemp']**2)*
          T.log(1+(ufc/ModelData['ElecA'])*(ModelData['Elect1']+ModelData['Elect2']*ModelData['ElecTemp']+ModelData['Elect3']*
          ModelData['ElecTemp']**2)));
      dx[2] = ModelData['ElecNumCells']*ufc*ModelData['ElecFefficiency'];
  else:      # - Fuelcell .. the parameter w(ks,2) is in this case the CURRENT Ampers*FaradayConstant*n_e
      #produced power:...............ufc is the same current through all
      #the cells
      areacurrent=(-ufc/ModelData['CellA']);#inside the cell
        xPWO=xPWO-ModelData['CellNumCell']*ufc*(ModelData['CellVrev']+ModelData['CellB']*T.log(ModelData['Celli0'])-...
            ModelData['CellAt']*T.log(areacurrent)-areacurrent*ModelData['Cellr']+ModelData['Cellm']*
            T.exp(areacurrent*ModelData['Celll']));
      #eaten hydrogen in units of [H2 times n_e times FaradayCOnstant]:
      dx[2] = ufc*ModelData['CellNumCell']*ModelData['CellFInvefficiency'];
    
      
  if (xPWO>=0):
      dx[3]=xPWO*ModelData['cSellPower'];
  else:
      dx[3]=xPWO*ModelData['cBuyPower'];

def fun( x0, ts, ws,ModelData )
ndisc=size(ts,2)-1;#ts is row vector

 # Options for ODE solver
 optODE = odeset( 'RelTol', 1e-8, 'AbsTol', 1e-8 );

 allvalues=zeros(ndisc,ModelData.numstates);
 
 # Forward state integration
 z0 = [ x0 ];#column vector
 allvalues(1,:)=z0(1:ModelData.numstates,1)';%results will be layered as rows to one big numstates-column matrix
 #checking of constraints...
 lastwarn('', '');
 for ks = 1:ndisc
    [tspan,zs] = ode15s( @(t,x)state(t,x,ws,ks,ndisc,ModelData), [ts(ks),ts(ks+1)], z0, optODE );
    if (lastwarn)
        dlmwrite('errored-controls.txt',ws);
        dlmwrite('errored-ks.txt',ks);
        error(lastwarn)
    end
    #for single shooting, we continue where we ended last time interval
    #z0 = zs(end,:)'; %vratilo sloupcovej vektor, do z0 udelame radek 
    #for multiple shooting, we have variable for the begining:
    if (ks<ndisc) #but not for the last stage, we set it only for
        for i=1:(ModelData.numstates)
            z0(i,1)=ws((ModelData.numcontrols*ndisc)+(i-1)*(ndisc-1)+ks,1); 
        end
        z0(ModelData.numstates+1,1)=zs(end,end);#we just add the cost, no multipleshooting for cost
    end
 
    allvalues(ks+1,:)=zs(end,1:ModelData.numstates);#ptze +1 je cost function a tu nemusime rikat. Also zs is returned as rows
    #if (ks==1)
    #   largest=max(zs)';
    #   smallest=min(zs)';
    #else
    #   largest=max([largest,zs']')';
    #   smallest=min([smallest,zs']')';
    #end
 end
 # Functions
 f = zs(end,:)';
 #f = [f,largest,smallest];

end


def main():
  #
  # MULTIPLE SHOOTING
  #

  # Options for NLP Solvers
  #optODE = odeset( 'RelTol', 1e-8, 'AbsTol', 1e-8 );'GradObj','on','DerivativeCheck','on','HessUpdate','bfgs','Diagnostic','on','LineSearchType','cubicpoly',
  #GradConstr on
  # Time Horizon and Initial State
  #Data.t_start; % Initial time
  #Data.t_end; % Final time

  #x0 = [ Data.Wbattinit; 0; 0 ]; % Initial states+1 state for cost variable

  ndisc = 2**(ModelData['cLogBegin']-1);
 
  w0 = rand(ModelData['numcontrols']*ndisc+ModelData['numstates']*(ndisc-1),1) #and each control has startvalues (one less than controls) - according to multiple shooting
  for _is in range(ModelData['cLogBegin']:ModelData['cLogEnd']):
    ndisc = 2*ndisc; # Number of stages: ns = 2, 4, 8, 16, and 32
    ts = [ModelData['t_start']:(ModelData['t_end']-ModelData['t_start'])/ndisc:ModelData['t_end']] # Time stages (equipartition)

  # Initial Guess and Bounds for the Parameters
  wold=w0;
  if (False):                               
    w0 = dlmread('controldataiter65')
  else:
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
  wL = ModelData['Pbattminusmax']/ModelData['ControlScales']*ones(ModelData['numcontrols']*ndisc+ModelData['numstates']*(ndisc-1),1); #parameter (=control) bounds in each tie stage
  wU = ModelData['Pbattplusmax']/ModelData['ControlScales']*ones(ModelData['numcontrols']*ndisc+ModelData['numstates']*(ndisc-1),1);
  wL((ndisc+1):(2*ndisc),1)=ModelData['Ph2minusmax']/ModelData['ControlScales']*ones(2*ndisc-(ndisc+1)+1,1);
  wU((ndisc+1):(2*ndisc),1)=ModelData['Ph2plusmax']/ModelData['ControlScales']*ones(2*ndisc-(ndisc+1)+1,1); 
  #bounds for states - on multipleshooting it equals bounds for control
  #variables:
  #bounds on the beginnings are just bounds on state variables:
  wL((2*ndisc+1):(2*ndisc+(ndisc-1)),1)=0*zeros((2*ndisc+(ndisc-1)-(2*ndisc+1)+1),1);
  wU((2*ndisc+1):(2*ndisc+(ndisc-1)),1)=ModelData['Wbattmax']*ones((2*ndisc+(ndisc-1)-(2*ndisc+1)+1),1);
  wL((2*ndisc+(ndisc-1)+1):(2*ndisc+2*(ndisc-1)),1)=0*zeros((2*ndisc+2*(ndisc-1))-(2*ndisc+(ndisc-1)+1)+1,1);
  wU((2*ndisc+(ndisc-1)+1):(2*ndisc+2*(ndisc-1)),1)=ModelData['Wfcmax']*ones((2*ndisc+2*(ndisc-1))-(2*ndisc+(ndisc-1)+1)+1,1);
 
 
  #optNLP = optimset('Algorithm','sqp', 'LargeScale', 'off', 'GradObj', 'off', 'GradConstr', 'off',...
  #'DerivativeCheck', 'off', 'Display', 'iter-detailed', 'TolX', 1e-11,...
  #'TolFun', 1e-9, 'TolCon', 1e-9, 'MaxFunEval', 8000000,...
  #'DiffMinChange', 1e-5,...
  #'OutputFcn',@(x,optimValues,state)nlpoutfun(x0,x,optimValues,state, ts, Data, optODE));#'PlotFcns'

  # Sequential Approach of Dynamic Optimization
  #[ wopt ] = fmincon( @(ws)obj(x0,ts,ws,Data), w0, [], [], [], [], wL, wU,...
  #@(ws)ctr(x0,ts,ws,Data), optNLP);
  
  def ModelFunkce(w):
    #ret={obj: None, objgrad: None, eqcon: None, eqcongrad: None, incon: None, incongrad:None}
    sim = fun( x0, ts, ws);
    #def obj( x0, ts, ws,ModelData ):
    #ndisc=size(ts,2)-1;%ts is row vector
      #simulates using given controls ws and souhld return a function to minimize
    ret['obj']=  -sim(3)-ModelData['cSellFinalPower']*(sim(1)+sim(2)*ModelData['CellFInvefficiency']);#power sold on our way plus what we managed to store in fuelcells and batteries
    #we want to maximize cost, so return minus cost to minimize
    
    #def ctr( x0, ts, ws,ModelData ):
    #ndisc=size(ts,2)-1;#ts is row vector
    #[f,alldata] = fun( x0, ts, ws,ModelData );                                   
    #sloupec vrqceno celkem ubatt,ucell,cost,...
    #maxubatt,maxucell,maxcost,minubatt,minucell,mincost .. maxima co sme potkali po ceste
    #single shooting: zadne constraints:
    #ceq = [];# =0                     #radek .. toto ma byt rovno nule
    ceq = [];
    #multiple shooting:
    for inumstates in range(1,ModelData['numstates']):
       xquaa=alldata(2:(ndisc),inumstates) #now it is a row ... alldata is from 1 to ndisc+1 ... at 1 it is x0 at ndisc+1 it is the value at the end of the last timeslot
       xquab=ws((ModelData['numcontrols']*ndisc+(inumstates-1)*(ndisc-1)+1)
           :(ModelData['numcontrols']*ndisc+(inumstates)*(ndisc-1)),1)';    
       ceq = [ceq,(xquaa-xquab).*(xquaa-xquab)];#begin at the next time slot where you ended in the previous time slot
    ret['eqcon'] = ceq   #other things that need to be equal to zero, comes from problem formulation
    #c = [f(4)-ModelData.Wbattmax;f(5)-ModelData.Wfcmax;...
    #    -f(7);-f(8)]'; #\leq 0                             #radek ... toto ma byt mensi nez nula
    ret['incon'] = [-sim(1),-sim(2),sim(1)-ModelData['Wbattmax'],sim(2)-ModelData['Wfcmax']];
   
   #currently there are constraints only on control variables, and on control
   #variables representing beggining points of integration. And no other
   #constraints on states. So another implementation might be to also add
   #constraints on states. Or also while integrating taking care of max and
   #min values of states. Afraid it will damage precision, i did only the
   #constraints on the node points of discretization.

  
  Ieval = ModelMultiEval(ModelFunkce) #dod
  
  scipy.optimize.minimize(fun = Ieval.ObjCall, x0=w0, args=(), method='SLSQP', jac=False, hess=None, hessp=None,
      bounds=zip(wL,wU),     #dod je zip spravne?
      constraints=({type: 'eq' fun: Ieval.EqConCall},{ type: 'ineq', fun: Ieval.InConCall}),
      tol=1e-9, callback=None,       
      options={'maxiter': 1000, 'disp':True})
      #http://stackoverflow.com/questions/23476152/dynamically-writing-the-objective-function-and-constraints-for-scipy-optimize-mihttp://stackoverflow.com/questions/23476152/dynamically-writing-the-objective-function-and-constraints-for-scipy-optimize-mi
  
  dispopt( x0, ts, wopt,Data, optODE,-1 );
  dlmwrite('controldata',wopt);