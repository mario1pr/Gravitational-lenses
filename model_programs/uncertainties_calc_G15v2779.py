# Calculation of uncertainties

import numpy as np
import pickle
import gzip
import sys
import os
sys.path.append(os.path.abspath('..')) #points to ~/visilens-master/visilens
print("Current directory:", os.getcwd())
print("sys.path:", sys.path)
import visilens as vl
from astropy.cosmology import Planck18

output_file = "values_G15v2779_modelcont_3005003000.txt"
sys.stdout = open(output_file,"w")

filename = "chains_G15v2779_modelcont_3005003000.pzip"
n_last_steps = 5  #steps used for calculate uncertainties
tail_steps = 100  #steps used for detect outliers
k = 5             #MAD range

print("Loading chains...")
data = pickle.load(gzip.open(filename,"rb"))
chains_raw = data ["chains"]
print(type(chains_raw))
print(chains_raw.dtype)
print(chains_raw.dtype.names)

chains_values = np.array([list(c) for c in chains_raw])
nparams = chains_values.shape[1]
nwalkers = 300
if chains_values.shape[0] % nwalkers != 0:
	raise ValueError("Cannot infer nsteps: inconsistent chain lenght.")

nsteps = chains_values.shape[0] // nwalkers
chains = chains_values.reshape(nwalkers, nsteps, nparams)
print(f"Chains shape reconstructed: {chains.shape}")

if n_last_steps > nsteps:
	raise ValueError("You have asked for more steps than exist in the chain.")

chains_post = chains[:, -n_last_steps:,:]
print(f"Using last {n_last_steps} steps.")

samples = chains_post.reshape(-1,nparams)

if hasattr(chains_raw, "dtype") and chains_raw.dtype.names is not None:
    param_names = list(chains_raw.dtype.names)
    print("Using parameter names from recarray dtype.")
else:
    param_names = [f"param_{i}" for i in range(nparams)]
    print("WARNING: No parameter names found. Using generic labels.")

# Safety check
if len(param_names) != nparams:
    print(f"WARNING: mismatch names ({len(param_names)}) vs params ({nparams}). Fixing.")
    param_names = list(param_names)[:nparams]

# Statistical calculation
results = {}

for i, name in enumerate(param_names):
	print("\n===================================================")
	print(f"Parameter:{name}")
	print("===================================================")
        #Detection of outliers with MAD
	param_tail = chains[:,-tail_steps:,i]
	chain_medians = np.median(param_tail,axis=1)
	global_median = np.median(chain_medians)
	mad = np.median(np.abs(chain_medians-global_median))

	if mad == 0:
		mad = 1e-10
	
	thereshold = k*mad
	outliers = np.abs(chain_medians-global_median) > thereshold
	good_walkers = ~outliers
	n_outliers = np.sum(outliers)
	percent_outliers = 100 * n_outliers/nwalkers
	print(f"Outliers detected: {n_outliers}/{nwalkers} ({percent_outliers:.1f}%)")

	filtered_chains = chains_post[good_walkers,:,:]
	n_good_walkers = filtered_chains.shape[0]
	print (f"Good walkers used: {n_good_walkers}")
	samples = filtered_chains.reshape(-1,nparams)

	param_samples = samples[:,i]
	# percentiles
	p2_5,p16,p50,p84,p97_5 = np.percentile(param_samples,[2.5,16,50,84,97.5])

	#uncertainties
	err68_minus = p50-p16
	err68_plus = p84-p50

	err95_minus = p50-p2_5
	err95_plus = p97_5-p50
	
	# standard deviation, for reference only
	std = np.std(param_samples)
	results[name] = {
        "median": p50,
        "std": std,
        "68%": (p16, p84),
        "95%": (p2_5, p97_5),
        "err68": (err68_minus, err68_plus),
        "err95": (err95_minus, err95_plus),
	}
for name, res in results.items():
	print(f"\n--- {name} ---")
	print(f"Median = {res['median']:.5f}")
	
	print(f"68% CI = [{res['68%'][0]:.5f}, {res['68%'][1]:.5f}] "
	      f"(+{res['err68'][1]:.5f} / -{res['err68'][0]:.5f})")

	print(f"95% CI = [{res['95%'][0]:.5f}, {res['95%'][1]:.5f}] "
	      f"(+{res['err95'][1]:.5f} / -{res['err95'][0]:.5f})")

	print(f"Std = {res['std']:.5f}")
	# Gaussianity check
	iqr_sigma = (res["68%"][1] - res["68%"][0]) / 2
	if np.isclose(res["std"], iqr_sigma, rtol=0.1):
    		print("≈ Gaussian")
	else:
   		print("No gaussian")

# Half-light radius calculation

half_light_results = {}
# Detect all sources automatically
source_ids = []

for name in param_names:
        if name.startswith("majaxS"):
                sid = name.replace("majaxS","")
                source_ids.append(sid)
source_ids = sorted(source_ids)

for sid in source_ids:
        maj_name = f"majaxS{sid}"
        q_name = f"axisratioS{sid}"
        if maj_name not in results or q_name not in results:
                continue
        # Median values
        a = results[maj_name]["median"]
        q = results[q_name]["median"]
        # Ellipticity
        e = 1.0 - q
        # Half-light radius
        rhalf = a * np.sqrt(q)

        # Uncertainties

        # 68% CI
        sigma_a_68 = np.mean(results[maj_name]["err68"])
        sigma_q_68 = np.mean(results[q_name]["err68"])
        # 95% CI
        sigma_a_95 = np.mean(results[maj_name]["err95"])
        sigma_q_95 = np.mean(results[q_name]["err95"])
        # std
        sigma_a_std = results[maj_name]["std"]
        sigma_q_std = results[q_name]["std"]

        # Partial derivatives
        dr_da = np.sqrt(q)
        dr_dq = a / (2.0 * np.sqrt(q))
        # Error propagation
        sigma_r_68 = np.sqrt((dr_da*sigma_a_68)**2 + (dr_dq*sigma_q_68)**2)
        sigma_r_95 = np.sqrt((dr_da*sigma_a_95)**2 + (dr_dq*sigma_q_95)**2)
        sigma_r_std = np.sqrt((dr_da*sigma_a_std)**2 + (dr_dq*sigma_q_std)**2)

        # Ellipticity uncertainty
        # e = 1 - q -> sigma_e = sigma_q
        sigma_e_68 = sigma_q_68
        sigma_e_95 = sigma_q_95
        sigma_e_std = sigma_q_std

        # Conversion from arcsec to kpc for r_half

        # Source redshift
        z_source = 4.243
        # Angular diameter distance in kpc
        DA = Planck18.angular_diameter_distance(z_source).value * 1000.0

        conversion_factor = DA / 206265.0
        rhalf_kpc = rhalf * conversion_factor

        sigma_r_68_kpc = sigma_r_68 * conversion_factor
        sigma_r_95_kpc = sigma_r_95 * conversion_factor
        sigma_r_std_kpc = sigma_r_std * conversion_factor

        half_light_results[sid] = {
                "e": e,
                "rhalf": rhalf,
                "rhalf_kpc": rhalf_kpc,
                "sigma_e_68": sigma_e_68,
                "sigma_e_95": sigma_e_95,
                "sigma_e_std": sigma_e_std,
                "sigma_r_68": sigma_r_68,
                "sigma_r_95": sigma_r_95,
                "sigma_r_std": sigma_r_std,
                "sigma_r_68_kpc": sigma_r_68_kpc,
                "sigma_r_95_kpc": sigma_r_95_kpc,
                "sigma_r_std_kpc": sigma_r_std_kpc
        }


# Half-light radius output

print("\n")
print("===============================")
print("HALF-LIGHT RADIUS [ARCSEC]")
print("===============================")

for sid, res in half_light_results.items():
        print(f"\n--- Source {sid} ---")

        print(
                f"Ellipticity eS{sid} = "
                f"{res['e']:.5f} "
                f"± {res['sigma_e_68']:.5f} (68%) "
                f"± {res['sigma_e_95']:.5f} (95%) "
                f"± {res['sigma_e_std']:.5f} (std) "
        )

        print(
                f"r_half,S{sid} = "
                f"{res['rhalf']:.5f} arcsec"
                f"± {res['sigma_r_68']:.5f} (68%) "
                f"± {res['sigma_r_95']:.5f} (95%) "
                f"± {res['sigma_r_std']:.5f} (std) "
        )

print("\n")
print("===============================")
print("HALF-LIGHT RADIUS [kpc]")
print("===============================")

for sid, res in half_light_results.items():
        print(f"\n--- Source {sid} ---")

        print(
                f"r_half,S{sid} = "
                f"{res['rhalf_kpc']:.5f} kpc"
                f"± {res['sigma_r_68_kpc']:.5f} (68%) "
                f"± {res['sigma_r_95_kpc']:.5f} (95%) "
                f"± {res['sigma_r_std_kpc']:.5f} (std) "
        )

sys.stdout.close()
