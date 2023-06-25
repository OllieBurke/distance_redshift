import numpy as np
from scipy import integrate
from tqdm import tqdm
from astropy.cosmology import FlatLambdaCDM

def llike(data_dl, delta_dl, H0_prop, **kwargs):
    """
    Computes the (log) likelihood for a given set of data in the context of a distance-redshift relation.

    Parameters:
    data_dl (array-like): Array of observed luminosity distances.
    data_z (array-like): Array of corresponding redshift values.
    H0_prop (float): Proposed value of the Hubble constant (H0).
    delta_dl (array-like): Array of uncertainties (errors) in the luminosity distances.
    N_obs (int): Number of observations.
    c (float, optional): Speed of light in km/s. Default is 3e5.
    omega_m (float, optional): Density parameter for matter in the Universe. Default is 0.274.

    Returns:
    float: The (log) likelihood value.

    Note:
    This function assumes a distance-redshift relation given by the parameter omega_m and performs a likelihood analysis
    based on the observed luminosity distances and their uncertainties. It uses the proposed value of H0 (Hubble constant)
    and the speed of light to calculate the expected luminosity distances corresponding to the observed redshift values.
    The likelihood is computed as a combination of the first and second terms.

    The first term represents the normalization factor and does not depend on the model or data.
    The second term evaluates the squared difference between the observed and expected luminosity distances,
    weighted by the inverse of the squared uncertainties.

    The (log) likelihood is the sum of the first and second terms.

    """

    # We know omega_m and the redshift of the source
    Omega_m = kwargs['Omega_m']
    z_s = kwargs['z_s']

    # Generate new cosmology
    cosmo_prop = FlatLambdaCDM(H0=H0_prop, Om0 = Omega_m)

    # Estimate the distance with this cosmology
    estm_dl = cosmo_prop.luminosity_distance(z_s).value

    # Compute log-likelihood
    llike = -0.5*np.sum((delta_dl**-2) * (data_dl - estm_dl)**2)
    return(llike)

def lprior_uniform(param, param_low_val = 30, param_high_val = 150):
    """
    Set uniform priors on parameters with select ranges.
    """
    if param < param_low_val or param > param_high_val:
        return -np.inf
    else:
        return 0

def lpost(data_dl, delta_dl, H0_prop, **kwargs):
    """
    Compute the log posterior by combining the log prior and log likelihood.

    Parameters:
    data_dl (array-like): Array of observed luminosity distances.
    data_z (array-like): Array of corresponding redshift values.
    H0_prop (float): Proposed value of the H0 parameter.
    delta_dl (array-like): Array of uncertainties (errors) in the luminosity distances.
    N_obs (int): Number of observed data points.

    param1 (float): Value of parameter 1.
    param1_low_range (float, optional): Lower bound of the range for parameter 1. Default is 30.
    param1_high_range (float, optional): Upper bound of the range for parameter 1. Default is 100.

    Returns:
    float: Log posterior value.

    Note:
    This function computes the log posterior by adding the log prior and log likelihood.

    The log prior is calculated using a uniform prior distribution for parameter 1, defined by the `param1_low_range`
    and `param1_high_range` arguments.

    The log likelihood is computed using the `llike` function, which takes the observed data, proposed H0 value,
    uncertainty values, and number of observed data points as inputs.
    """
    return lprior_uniform(H0_prop) + llike(data_dl, delta_dl, H0_prop, **kwargs)


def accept_reject(lp_prop, lp_prev):
    """
    Compute the log acceptance probability and decide whether to accept or reject.

    Parameters:
    lp_prop (float): Log posterior value for the proposed point.
    lp_prev (float): Log posterior value for the previous point.

    Returns:
    int: 1 if the proposed point is accepted, 0 if it is rejected.

    Note:
    This function computes the log acceptance probability as the minimum of 0 and the difference between the log posterior
    values of the proposed point and the previous point.

    It then generates a random number from a uniform distribution in the range [0, 1] and compares its logarithm to the
    log acceptance probability. If the logarithm is less than the log acceptance probability, the proposed point is accepted
    (returning 1), otherwise it is rejected (returning 0).
    """

    u = np.random.uniform(size = 1)  # U[0, 1]
    logalpha = np.minimum(0, lp_prop - lp_prev)  # log acceptance probability
    if np.log(u) < logalpha:
        return(1)  # Accept
    else:
        return(0)  # Reject

def MCMC_run(data_dl, delta_dl, param_start,  printerval = 5000, H0_var_prop = 0.05,  Ntotal = 100000, burnin = 0, **kwargs):
    """
    Metropolis MCMC sampler for parameter estimation.

    Parameters:
    data_dl (array-like): Array of observed luminosity distances.
    data_z (array-like): Array of corresponding redshift values.
    delta_dl (array-like): Array of uncertainties (errors) in the luminosity distances.
    param_start (tuple): Tuple of starting parameter values, in this case, (H0_start,)
    printerval (int, optional): Interval for printing the accept/reject ratio. Default is 50000.
    H0_var_prop (float, optional): Variance of the proposal distribution for the H0 parameter. Default is 0.8.
    Ntotal (int, optional): Total number of iterations. Default is 300000.
    burnin (int, optional): Number of burn-in iterations to discard. Default is 10000.

    Returns:
    tuple: Tuple containing the chain of H0 values and the log posterior values.

    Note:
    This function implements a Metropolis MCMC sampler for parameter estimation. It samples the posterior distribution
    of the H0 parameter given the observed data.

    The function starts with the initial parameter values provided in the `param_start` tuple.
    It initializes the chain of H0 values and computes the log posterior for the initial value.

    The function then iterates for a total of `Ntotal` iterations. In each iteration, a new proposal value of H0 is
    generated from a normal distribution with mean equal to the previous H0 value and standard deviation equal to
    the square root of `H0_var_prop`. The log posterior is computed for the proposed value.

    If the proposed value is accepted according to the accept/reject criterion, it is added to the chain of H0 values.
    Otherwise, the previous value is retained.

    The function also tracks the accept/reject count and computes the log posterior for each accepted value.

    At the end of the iterations, the function returns the chain of H0 values after discarding the burn-in period,
    along with the corresponding log posterior values.
    """
    # Set starting values
    H0_chain = [param_start[0]]

    # Initial value for log posterior
    lp = []
    lp.append(lpost(data_dl, delta_dl, H0_chain[0], **kwargs)) # Append first value of log posterior 
    
    lp_store = lp[0]  # Create log posterior storage to be overwritten

    #####                                                  
    # Run MCMC
    #####
    accept_reject_count = [1]
    for i in tqdm(range(1, Ntotal)):
        
        if i % printerval == 0: # Print accept/reject ratio.
            accept_reject_ratio = sum(accept_reject_count)/len(accept_reject_count)
            tqdm.write("Iteration {0}, accept_reject = {1}".format(i,accept_reject_ratio))
            
        lp_prev = lp_store  # Call previous stored log posterior
        
        # Propose new points according to a normal proposal distribution of fixed variance 
        H0_prop = np.random.normal(H0_chain[i - 1], np.sqrt(H0_var_prop))
    
        # Compute log posterior
        lp_prop = lpost(data_dl, delta_dl, H0_prop, **kwargs)
        
        if accept_reject(lp_prop, lp_prev) == 1:  # Accept
            H0_chain.append(H0_prop)    # accept H0_{prop} as new sample
            accept_reject_count.append(1)
            lp_store = lp_prop  # Overwrite lp_store
            
        else:  # Reject, if this is the case we use previously accepted values
            H0_chain.append(H0_chain[i - 1])
            accept_reject_count.append(0)

        lp.append(lp_store)

    # Recast as np.arrays
    H0_chain = np.array(H0_chain)
    H0_chain = H0_chain[burnin:]
    
    return H0_chain, lp  # Return chains and log posterior.    