import numpy as np
import os
import matplotlib.pyplot as plt
from MCMC_func import MCMC_run
import corner
from astropy.cosmology import FlatLambdaCDM

H0_true = 70.5   # True value of hubble constant 
Omega_m_true = 0.274
Omega_Lambda = 1 - Omega_m_true

# Weak source
z_weak = np.array([1.0, 1.26])
z_strong = np.array([0.3, 0.5])

kwargs_weak = {'H0_true':H0_true, 'Omega_m_true':Omega_m_true, 
            'Omega_Lambda':Omega_Lambda, 'z_s':z_weak}

kwargs_strong = {'H0_true':H0_true, 'Omega_m_true':Omega_m_true, 
            'Omega_Lambda':Omega_Lambda, 'z_s':z_strong}
# Assign a cosmology
cosmo = FlatLambdaCDM(H0=H0_true, Om0=Omega_m_true)
# Calculate luminosity distance of source assuming a cosmology
D_Mpc_weak = cosmo.luminosity_distance(z_weak).value      # Measure in Mpc
D_Mpc_strong = cosmo.luminosity_distance(z_strong).value      # Measure in Mpc
# Give rough estimate of precision in parameters
delta_D_Mpc_weak = np.array([0.17146203329740012 * 1e3 , 0.12121466818773752 * 1e3])
delta_D_Mpc_strong = np.array([0.03950497404117258 * 1e3, 0.010812092226434833 * 1e3])
# Input data
data_dl_weak  = np.array(D_Mpc_weak)
data_dl_strong = np.array(D_Mpc_strong) 
# Start initial H0 
param_start =[H0_true, Omega_m_true]

# Run MCMC algorithm
# H0_chain_weak, Omega_m_chain_weak, lp = MCMC_run(data_dl_weak, delta_D_Mpc_weak, param_start, H0_var_prop = 3, Omega_m_var_prop = 1e-3, **kwargs_weak)
H0_chain_strong, Omega_m_chain_strong, lp = MCMC_run(data_dl_strong, delta_D_Mpc_strong, param_start, H0_var_prop = 2, Omega_m_var_prop = 3e-5, **kwargs_strong)

# Output results

# median_H0_weak = np.median(H0_chain_weak)
median_H0_strong = np.median(H0_chain_strong)

# median_Omega_m_weak = np.median(Omega_m_chain_weak)
median_Omega_m_strong = np.median(Omega_m_chain_strong)

# print("Weak source - redshift = 1.26")
# print("median of H0 =", median_H0_weak)
# print("median of Omega_m =", median_Omega_m_weak)
print("")
print("Strong source - redshift = 0.5")
print("median of H0 = ", median_H0_strong)
print("median of Omega_m =", median_Omega_m_strong)
print("")
# samples_weak = np.column_stack([H0_chain_weak,Omega_m_chain_weak])
samples_strong = np.column_stack([H0_chain_strong,Omega_m_chain_strong])

params = [r"$H_0$", r"$\Omega_m$"]

# figure = corner.corner(samples_weak, bins = 30, color = 'purple', labels = params, 
#                 plot_datapoints = False, smooth1d = True, smooth = True, #quantiles=[0.16, 0.5, 0.84],
#                 show_titles = True, label_kwargs = {"fontsize":12}, title_fmt='.2f',title_kwargs={"fontsize": 12})

figure = corner.corner(samples_strong, bins = 30, color = 'red', labels = params, 
                plot_datapoints = False, smooth1d = True, smooth = True, #quantiles=[0.16, 0.5, 0.84],
                show_titles = True, label_kwargs = {"fontsize":12}, title_fmt='.2f',title_kwargs={"fontsize": 12})


axes = np.array(figure.axes).reshape((2, 2))
m = [H0_true, Omega_m_true]

for i in range(2):
    ax = axes[i, i]
    ax.axvline(m[i], color="g")

for yi in range(2):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axvline(m[xi], color="g")
        ax.axhline(m[yi], color="g")
        ax.plot(m[xi], m[yi], "sg")


os.chdir("plots/")
plt.savefig("constraint_omega_H0.pdf")
plt.show()

breakpoint()