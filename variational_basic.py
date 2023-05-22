import numpy as np

import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import optax
import corner
import matplotlib.pyplot as plt
import corner
import jax                                           
#jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
from functools import partial
import gwjax
import gwjax.imrphenom

import distrax
import haiku as hk      #neural network library for JAX 
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple
Array = jnp.ndarray
PRNGKey = Array
OptState = Any

# Set up detector
from gwjax.detector import antenna_pattern, detectors



def project_to_detector(params, htup, freqs, detector, gmst):     #simulated strain with injected signal
    """Compute the response of the detector to incoming strain

    Args:
        params (dict): signal parameters:
            ra, dec, psi
        htup (tuple): (h+, hx) in frequency domain
        freqs (array): frequencies of the strain data array
        invasd (array): 1/sqrt(PSD) - the metric for data space
        detector (namedtuple): detector structure (location, response)
        gmst (float): Greenwich Mean Sidereal Time

    Returns:
        htilde (array): measured strain in frequency domain
    """
    deltat = gwjax.time_delay_from_earth_center(detector, gmst,
                                                params['ra'], params['dec'])
    hp, hx = htup
    fp, fx = gwjax.antenna_pattern(detector, gmst,
                                    params['ra'], params['dec'], params['psi'])
    cexp = jnp.exp(-1j*2*jnp.pi*freqs*deltat)
    m = (fp*hp + fx*hx)*cexp
    #print([x.shape for x in (fp,hp,fx,hx,cexp,m)])
    return m

#@jax.jit
def waveform_projections(params, freqs, project_dict, gmst):
    htup = gwjax.imrphenom.IMRPhenomD(freqs, params)
    return {ifo: project(params, htup) for ifo, project in project_dict.items()}

# likelihood function
class LogL(object):
    true_params = dict(
        A=10,
        phi=1.5,
        dec=0.5,
        ra=1.5,
        psi=1.0
    )
    ifos = ['H1', 'L1']
    gps_time = 0.0
    gmst = gwjax.gmst(gps_time)
    def __init__(self, T = 1):
        self.detectors = {d: detectors[d] for d in self.ifos}
        true_params = self.true_params

        
        # simulate data
        T = 1 # duration of analysis segment
        srate = 20
        df = 1/T
        freqs = jnp.arange(1+srate*T, dtype=jnp.float32)*df
        f0 = 100
        Nbins = 1
        i0 =int(f0/df)
        self.freqs = jnp.atleast_2d(freqs[i0])

        self.beta = 1/T
        self.data = self.simulate_response(true_params['A']*jnp.cos(true_params['phi']),
                                            true_params['A']*jnp.sin(true_params['phi']),
                                        true_params)
        self.bounds = dict(
            A = [self.true_params['A']*jnp.exp(-1), self.true_params['A']*jnp.exp(1)],
            phi = [-jnp.pi, jnp.pi],
            dec = [-jnp.pi/2, jnp.pi/2],
            ra = [0, jnp.pi*2],
            psi = [-jnp.pi/2, jnp.pi/2]
        )

    def simulate_response(self, hp, hc, sky_params):
        r = {d: project_to_detector(sky_params,
                                    (hp, hc),
                                    self.freqs,
                                    self.detectors[d],
                                    self.gmst)
            for d in self.detectors.keys()}
        return r
    
    def __call__(self, params):
        hp = params['A']*jnp.cos(params['phi'])
        hc = params['A']*jnp.sin(params['phi'])
        r = self.simulate_response(hp, hc, params)
        residuals = jnp.array(
            [r[ifo] - self.data[ifo] for ifo in self.detectors.keys()]
        )
        #print(residuals.shape)
        return -self.beta*jnp.real(
            jnp.sum(
                residuals*jnp.conj(residuals),
                axis=(0,1)))/2

    @property
    def params(self):
        params = ['A','phi','dec','ra','psi']
        return params


    #@jax.jit
    def array_to_phys(self, x: Array) -> dict:
        p = dict()
        p['A'] = self.true_params['A'] * jnp.exp(x[:,0])
        p['phi'] = x[:,1]*jnp.pi
        # p['hp'], p['hc'] = true_params['hp']+x[:,0], true_params['hc']+x[:,1]
        p['dec'] = jnp.pi/2 - jnp.arccos(x[:,2])
        p['ra'] = (x[:,3]+1)*jnp.pi
        p['psi'] = 0.5*jnp.pi*(x[:,4])
        return p


def make_conditioner(
    event_shape: Sequence[int],
    hidden_sizes: Sequence[int],
    num_bijector_params: int
) -> hk.Sequential:
  """Creates an."""
  return hk.Sequential([
      hk.Flatten(preserve_dims=-len(event_shape)),
      hk.nets.MLP(hidden_sizes, activate_final=True),
      # We initialize this linear layer to zero so that the flow is initialized
      # to the identity function.
      hk.Linear(
          np.prod(event_shape) * num_bijector_params,
          w_init=jnp.zeros,
          b_init=jnp.zeros),
      hk.Reshape(tuple(event_shape) + (num_bijector_params,), preserve_dims=-1),
  ])

def make_flow_model(
    event_shape: Sequence[int],
    num_layers: int = 4,
    hidden_sizes: Sequence[int] = [250, 250],
    num_bins: int = 4,
) -> distrax.Transformed:
    """Creates the flow model."""
    # Alternating binary mask.
    mask = np.arange(0, np.prod(event_shape)) % 2
    mask = np.reshape(mask, event_shape)
    mask = mask.astype(bool)

    def bijector_fn(params: Array):
        return distrax.RationalQuadraticSpline(
            params, range_min=-1., range_max=1.
        )

    # Number of parameters for the rational-quadratic spline:
    # - `num_bins` bin widths
    # - `num_bins` bin heights
    # - `num_bins + 1` knot slopes
    # for a total of `3 * num_bins + 1` parameters.
    num_bijector_params = 3 * num_bins + 1

    layers = []
    for _ in range(num_layers):
        layer = distrax.MaskedCoupling(
            mask=mask,
            bijector=bijector_fn,
            conditioner=make_conditioner(event_shape, hidden_sizes,
                                        num_bijector_params))
        layers.append(layer)
        # Flip the mask after each layer.
        mask = jnp.logical_not(mask)

    # We invert the flow so that the `forward` method is called with `log_prob`.
    flow = distrax.Inverse(distrax.Chain(layers))
    base_distribution = distrax.Independent(
        distrax.Uniform(low=jnp.ones(event_shape)*-1, high=jnp.ones(event_shape)*1),
        #distrax.Normal(loc=jnp.zeros(event_shape), scale=jnp.ones(event_shape)),
        reinterpreted_batch_ndims=len(event_shape)
    )

    return distrax.Transformed(base_distribution, flow)


@hk.without_apply_rng
@hk.transform
def sample_and_log_prob(prng_key: PRNGKey, n: int) -> Tuple[Any, Array]:

    model = make_flow_model(
        event_shape=(n_params,),
        num_layers=flow_num_layers,
        hidden_sizes=[hidden_size] * mlp_num_layers,
        num_bins=num_bins
    )
    return model.sample_and_log_prob(seed=prng_key, sample_shape=(n,))   #internal to distrax
    # returns x (sample from the flow q), and model.log_prob(x) (array of log(q) of th sampled points)


def log_likelihood(x: Array) -> Array:
    p = log_l.array_to_phys(x)
    return log_l(p)

def loss_fn(params: hk.Params, prng_key: PRNGKey, n: int) -> Array:       #computes reverse KL-divergence for the sample x_flow between the flow and gw loglikelihood.

    x_flow, log_q = sample_and_log_prob.apply(params, prng_key, n)           #gets sample from the flow and computes log_q for the sampled points.
    log_p = log_likelihood(x_flow)                                          #gets gw loglikelihood for the sampled points.

    # int Q \log (Q / P)
    # x \sim Q
    # mean(log (Q / P))
    loss = jnp.mean(log_q - log_p)
    return loss

@jax.jit
def update(
    params: hk.Params,
    prng_key: PRNGKey,
    opt_state: OptState,
) -> Tuple[hk.Params, OptState]:
    """Single SGD update step."""
    grads = jax.grad(loss_fn)(params, prng_key, Nsamps)
    updates, new_opt_state = optimiser.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

if __name__ == '__main__':


    


    log_l = LogL(T=1)
    

    print('Simulated data: ',log_l.data)

    # Test likelihood
    l = log_l(log_l.true_params)
    print(f'log likelihood of true params = {l}')

    n_params = len(log_l.params)
    flow_num_layers = 4
    hidden_size = 16
    mlp_num_layers = 2
    num_bins = 4

    # perform variational inference
    epochs = 1000
    loss = dict(train=[], val=[])
    Nsamps = 1000

    learning_rate = 0.01
    optimiser = optax.adam(learning_rate)              #stochastic gradient descent 

    prng_seq = hk.PRNGSequence(42)
    key = next(prng_seq)
    params = sample_and_log_prob.init(key, prng_key=key, n=Nsamps)
    opt_state = optimiser.init(params)

    from tqdm import tqdm, trange
    T0 = 10
    t_decay = epochs/(1+jnp.log(T0))
    ldict = dict(loss = 0, T=T0)
    annealing_stop =0
    with trange(epochs) as tepochs:
        for epoch in tepochs:
            if epoch > annealing_stop:
                T=1
            else:
                T = max(T0*jnp.exp(-epoch/t_decay),1.0)
            log_l.beta = 1/T
            ldict['T'] = f'{T:.2f}'
            prng_key = next(prng_seq)
            loss = loss_fn(params,  prng_key, Nsamps)
            ldict['loss'] = f'{loss:.2f}'
            tepochs.set_postfix(ldict, refresh=True)
            params, opt_state = update(params, prng_key, opt_state)        #take a step in direction of stepest descent (negative gradient)
            if epoch%50 == 0:
                #print(f'Epoch {epoch}, loss {loss}')
                x_gen, log_prob_gen = sample_and_log_prob.apply(params, next(prng_seq), 10*Nsamps)
                x_gen = np.array(x_gen, copy=False)
                p_gen = np.vstack(list(log_l.array_to_phys(x_gen).values()))
                truths = [log_l.true_params[n] for n in ['A','phi','dec','ra','psi']]
                fig = corner.corner(p_gen.T, labels = log_l.params, truths = truths)
                plt.show()



    print("Done!")

    x_gen, log_prob_gen = sample_and_log_prob.apply(params, next(prng_seq), 100*Nsamps)

    x_gen = np.array(x_gen, copy=False)
    p_gen = np.vstack(list(log_l.array_to_phys(x_gen).values()))
    truths = [log_l.true_params[n] for n in ['A','phi','ra','dec','psi']]
    fig = corner.corner(p_gen.T, labels = log_l.params, truths = truths)
    plt.show()
    plt.savefig(f'posterior_${epochs}.png')



