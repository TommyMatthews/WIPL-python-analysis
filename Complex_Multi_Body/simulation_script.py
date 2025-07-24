import sys
sys.path.append('/Users/sstk4353/packages/.')

from ComplexMultiBody import ComplexMultiBody as cmb
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import truncnorm
import os

np.random.seed(42)  # For reproducibility

def discrete_truncated_normal_pmf(mean, spread, lower=0, upper=25):
    """
    Returns a DataFrame with pitch values from lower to upper, and the corresponding
    probabilities from a truncated normal distribution (discretized).

    The truncated normal is defined over a continuous range 


    Each discrete integer pitch is assigned a probability equal to the integral of the PDF between pitch - 0.5 and pitch + 0.5,

    The probabilities are renormalized to ensure they sum to 1.


    """
    # Define truncation in standard normal space
    a = (lower - mean) / spread
    b = (upper - mean) / spread
    # Create the truncated normal distribution
    dist = truncnorm(a=a, b=b, loc=mean, scale=spread)

    # Generate discrete pitch values
    pitch_vals = np.arange(lower, upper + 1)

    # Compute probabilities for each integer by integrating the PDF over integer bins
    pmf = []
    for pitch in pitch_vals:
        p = dist.cdf(pitch + 0.5) - dist.cdf(pitch - 0.5)
        pmf.append(p)

    # Normalize to ensure total probability is exactly 1 (safety)
    pmf = np.array(pmf)
    pmf /= pmf.sum()

    # Package into a DataFrame
    df = pd.DataFrame({
        'pitch': pitch_vals,
        'p(x)': pmf
    })

    return df

def generate_distribution_df(radar_params, biological_params, run_id, save=True):

    range_gate_separation = radar_params['range_gate_separation']
    radar_beam_width = radar_params['radar_beam_width']
    range_of_observation = radar_params['range_of_observation']
    beam_angle = radar_params['beam_angle']

    density = biological_params['density']
    sizes = biological_params['sizes']
    size_distribution = biological_params['size_distribution']
    mean_heading = biological_params['mean_heading']
    heading_spread = biological_params['heading_spread']
    mean_pitch = biological_params['mean_pitch']
    pitch_spread = biological_params['pitch_spread']

    pitch_probability_df = discrete_truncated_normal_pmf(mean_pitch, pitch_spread, lower=0, upper=25)

    radius = range_of_observation*np.deg2rad(radar_beam_width)/2 # meters
    volume = np.pi * radius**2 * range_gate_separation # cubic meters

    n_scatterers = int(density * volume) # number of scatterers in the cylinder
    spacing = ((1/density)**(1/3)) # spacing between scatterers in meters

    print(f"Radar beam radius: {radius} m")
    print(f"Radar volume size: {volume} m^3")
    print(f"Estimated number of scatterers: {n_scatterers}"
      f"\nSpacing between scatterers: {spacing} m")


    # Generate coordinates

    x_vals = np.arange(-range_gate_separation/2, range_gate_separation/2 + spacing, spacing)
    y_vals = np.arange(-radius, radius + spacing, spacing)
    z_vals = np.arange(-radius, radius + spacing, spacing)  

    data = []
    cos_t = np.cos(np.deg2rad(-beam_angle))
    sin_t = np.sin(np.deg2rad(-beam_angle))

    for ix, x in enumerate(x_vals):
        for iy, y in enumerate(y_vals):
            for iz, z in enumerate(z_vals):
                if y**2 + z**2 <= radius**2:
                    # Rotate around y-axis: tilt cylinder in x–z plane
                    x_rot = x * cos_t + z * sin_t
                    y_rot = y
                    z_rot = -x * sin_t + z * cos_t

                    distance = np.sqrt(x_rot**2 + y_rot**2 + z_rot**2)
                    index_str = f"{ix}_{iy}_{iz}"

                    data.append({
                        "index": index_str,
                        "x": x_rot,
                        "y": y_rot,
                        "z": z_rot,
                        "distance_to_center": distance
                    })

    df = pd.DataFrame(data)
    df.set_index("index", inplace=True)

    df.attrs = {
        "radar_beam_width": radar_beam_width,
        "range_of_observation": range_of_observation,
        "range_gate_separation": range_gate_separation,
        "beam_angle": beam_angle,
        "density": density,
    }

    df.attrs['mean_heading'] = mean_heading
    df.attrs['heading_spread'] = heading_spread
    df.attrs['mean_pitch'] = mean_pitch
    df.attrs['pitch_spread'] = pitch_spread


    # Add scatterer params

    string_list = []
    size_list = []
    heading_list = []
    pitch_list = []

    name = 'Xxanth' # for Xestia xanthographa
    name_list = [name] * len(df)

    for _ in range(len(df)):
        size = np.random.choice(sizes, p=size_distribution)
        
        heading = int(np.random.normal(mean_heading, heading_spread))

        #heading = heading_sample if heading_sample>0 else 360 + heading_sample  # Ensure heading is positive
        pitch = int(np.random.choice(pitch_probability_df['pitch'], p=pitch_probability_df['p(x)']))
        string_list.append(f"{name}_{size}_{heading}_{pitch}")
        size_list.append(size)
        heading_list.append(heading)
        pitch_list.append(pitch)

    df['scatterer_id'] = string_list
    df['size'] = size_list
    df['heading'] = heading_list
    df['pitch'] = pitch_list

    # Calculate projected distances
    beam_vector = np.array([np.cos(np.deg2rad(beam_angle)), 0, np.sin(np.deg2rad(beam_angle))])

    # Project each (x, y, z) coordinate from df onto beam_vector
    df['relative_distance_along_beam'] = df[['x', 'y', 'z']].values @ beam_vector

    if save:
        pickle.dump(df, open(f'./runs/{run_id}/distribution_df_{run_id}.pkl', 'wb'))

    return df

def run_simulation(df, scatterer_dataset, frequency=5.6, override=False, ignore_heading=False):
    """
    Run the simulation for the given distribution DataFrame.
    
    Parameters:
    - df: DataFrame containing distribution data
    - frequency: Frequency in GHz
    - scatterer_dataset: xarray Dataset containing scatterer profiles
    - override: If True, use a specific scatterer profile
    - ignore_heading: If True, ignore the heading of scatterers
    """
    
    CMB = cmb(
        scatterer_dataset=scatterer_dataset,
        frequency=frequency,
        distribution_df=df,
        override=override,
        ignore_heading=ignore_heading
    )

    CMB.calculate_resultant_voltages()
    CMB._dual_pol_calcs_on_recieved_voltages()

    return CMB

if __name__ == "__main__":

    radar_params = {
        'range_gate_separation': 600,  # meters
        'radar_beam_width': 1,  # degrees
        'range_of_observation': 150000,  # meters
        'beam_angle': 0.5,  # degrees
    }

    biological_params = {
        'density': 10e-5,  # scatterers per cubic meter
        'sizes': [9,17],  # size in mm
        'size_distribution': [0.7,0.3],  # not used in this example
        'mean_heading': 0,  # degrees
        'heading_spread': 20,  # degrees
        'mean_pitch': 11,  # degrees
        'pitch_spread': 10,  # degrees
    }

    run_id = "70_30_9_17"

    dir_path = f'./runs/{run_id}'
    plots_path = os.path.join(dir_path, 'plots')
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    df = generate_distribution_df(radar_params, biological_params, run_id)

    CMB = run_simulation(df, xr.open_dataset('bioscatterer_database_v0.001.nc'), frequency=5.6)

    # Plot 1: Differential Reflectivity
    plt.figure()
    plt.plot(CMB.differential_reflectivity)
    plt.title('Differential Reflectivity')
    plt.xlabel('Azimuth Angle (degrees)')
    plt.ylabel('Differential Reflectivity (dB)')
    plt.savefig(os.path.join(plots_path, 'differential_reflectivity.png'))
    plt.close()

    # Plot 2: Differential Phase
    plt.figure()
    plt.plot(CMB.differential_phase)
    plt.title('Differential Phase')
    plt.xlabel('Azimuth Angle (degrees)')
    plt.ylabel('Differential Phase (degrees)')
    plt.savefig(os.path.join(plots_path, 'differential_phase.png'))
    plt.close()

    # Plot 3: Horizontal Power
    plt.figure()
    plt.plot(CMB.horizontal_power)
    plt.title('Horizontal Power')
    plt.xlabel('Azimuth Angle (degrees)')
    plt.ylabel('Horizontal Power (arbitrary units)')
    plt.savefig(os.path.join(plots_path, 'horizontal_power.png'))
    plt.close()

    # Plot 4: Vertical Power
    plt.figure()
    plt.plot(CMB.vertical_power)
    plt.title('Vertical Power')
    plt.xlabel('Azimuth Angle (degrees)')         
    plt.ylabel('Vertical Power (arbitrary units)')
    plt.savefig(os.path.join(plots_path, 'vertical_power.png'))
    plt.close()

