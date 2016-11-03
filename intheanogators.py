import theano.tensor as T
import theano

def theano_inner_rk4_step(  # accum:                               #i_step is integer...
    i_step, accum,  # accum je matice - pocet dimenzi vysledny funkce krat pocet bodu ve kterejch integrujeme po krivce
    # changing: (none)
    # constant parameters:
    Tmax,  # scalar
    Index,  # vector
    Int_steps_total,
    f):
  # times to integrate
  fshape = T.cast(Index.shape[0], T.config.floatX)
  Tim_t = Index * Tmax / fshape  # vektor (index)0,1,2,3,4,5,6,7,.....  -> vektor (tim_t)0,1/(n*Tmax) .... n/n * Tmax
  Tim_t = Tim_t + Tmax * (i_step / (fshape * Int_steps_total))  # elemwise-posunuto na integracni step ted...
  # integration step
  t_step = (Tmax / fshape) / Int_steps_total  # scalar

  # accum - states x (ndisc-1)             tzn states x xbegs.shape[1]
  # Tim_t -                                tzn 1      x xbegs.shape[1]          z theano.tensor.arange(x_begs.shape[1]).dimshuffle(1, 0)
  
  k1 = f(accum, Tim_t)  # y'=f(y,t) (vicedim fce...) #aplikuj funkci PO SLOUPCICH
  k2 = f(accum + t_step * 0.5 * k1, Tim_t + 0.5 * t_step)
  k3 = f(accum + t_step * 0.5 * k2, Tim_t + 0.5 * t_step)
  k4 = f(accum + t_step * k3, Tim_t + t_step)
  return i_step + 1, T.cast(accum + t_step / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), T.config.floatX)

def theano_inner_euler_step(  # accum:                               #i_step is integer...
    i_step, accum,
    # accum je matice - pocet dimenzi vysledny funkce krat pocet bodu ve kterejch integrujeme po krivce
    # changing: (none)
    # constant parameters:
    Tmax,  # scalar
    Index,  # vector
    Int_steps_total,
    f):
  # times to integrate
  fshape = T.cast(Index.shape[0], T.config.floatX)
  Tim_t = Index * Tmax / fshape  # vektor (index)0,1,2,3,4,5,6,7,.....  -> vektor (tim_t)0,1/(n*Tmax) .... n/n * Tmax
  Tim_t = Tim_t + Tmax * (i_step / (fshape * Int_steps_total))  # elemwise-posunuto na integracni step ted...
  # integration step
  t_step = (Tmax / fshape) / Int_steps_total  # scalar

  return i_step + 1, T.cast(accum + t_step * f(accum, Tim_t), T.config.floatX)  # euler
