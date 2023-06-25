import numpy as np
import matplotlib.pyplot as plt
from MCMC_func import MCMC_run
from astropy.cosmology import FlatLambdaCDM

H0_true = 70.5   # True value of hubble constant 
Omega_m = 0.274
Omega_Lambda = 1 - Omega_m
z_s_weak = 1.26  # Weak EMRI
z_s_strong = 0.5

kwargs_weak = {'H0_true':H0_true, 'Omega_m':Omega_m, 
            'Omega_Lambda':Omega_Lambda, 'z_s':z_s_weak}
        
kwargs_strong = {'H0_true':H0_true, 'Omega_m':Omega_m, 
            'Omega_Lambda':Omega_Lambda, 'z_s':z_s_strong}

# Assign a cosmology
cosmo = FlatLambdaCDM(H0=H0_true, Om0=Omega_m)

# Calculate luminosity distance of source assuming a cosmology
data_dl_lensed_bright = cosmo.luminosity_distance(z_s_strong).value      # Measure in Mpc
data_dl_lensed_weak = cosmo.luminosity_distance(z_s_weak).value      # Measure in Mpc
# D_Mpc_non_lensed = 9.028820462969673 * 1e3 # redshift 1.26
# D_Mpc_lensed = 8.896044451209866 * 1e3  # redshift 1.26
# Give rough estimate of precision in parameters
delta_D_Mpc_lensed = 0.12121466818773752 * 1e3
delta_D_Mpc_lensed_bright = 0.010812092226434833 * 1e3
# Input data
delta_dl_lensed_bright = np.array([delta_D_Mpc_lensed_bright])
delta_dl_lensed_z1p26 = np.array([delta_D_Mpc_lensed])
# Start initial H0 
param_start =[H0_true]

# Run MCMC algorithm

H0_var_prop_bright = 0.2
H0_chain_z_0p5_lensed, lp = MCMC_run(data_dl_lensed_bright, delta_dl_lensed_bright, param_start, H0_var_prop = H0_var_prop_bright, **kwargs_strong)

H0_var_prop_lensed = 4
H0_chain_z1p26_lensed, lp = MCMC_run(data_dl_lensed_weak, delta_dl_lensed_z1p26, param_start, H0_var_prop = H0_var_prop_lensed, **kwargs_weak) 
# Output results
# ma = np.mean(H0_chain)
# sda = np.std(H0_chain)
# print("mean of H0 =", ma, '_', "standard deviation of H0 =", sda)
# Plot result

breakpoint()
import os
os.chdir('/Users/oburke/Documents/LISA_Science/Projects/EMRIs/Lensing/Distance_Redshift_Relation/new_directory/Ollie_Codes/data_file')
plt.hist(H0_chain_z1p26_lensed, bins = 30, color = 'red', alpha = 1, label = r'$z_{S} = 1.26$')
plt.hist(H0_chain_z_0p5_lensed, bins = 30, color = 'blue',  alpha = 0.8, label = r'$z_{S} = 0.5$')
plt.xlabel(r'$H_{0} \ [km \, sec^{-1} \, Mpc^{-1}]$',fontsize = 16)
plt.ylabel(r'$P(H_{0}|s)$',fontsize = 16)
plt.title(r'Constraint on Hubble Constant',fontsize = 20)
plt.axvline(x = H0_true, label = 'True value', c = 'black', linestyle = 'dashed')
plt.legend(fontsize = 12)
# plt.yticks([])
plt.tight_layout()
plt.savefig("Hubble_constraint.pdf")
plt.show() 

