from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import numpy as np
import xarray as xr
from datetime import datetime

import sys
sys.path.append('/Users/sstk4353/packages/WIPL_python_analysis/Complex_Multi_Body/')

from simulation_script import generate_distribution_df, run_simulation

MAX_WORKERS = 3 # This nearly maxes out the ram on my mac book pro, so be careful with this number

scatterer_dataset = xr.open_dataset('/Users/sstk4353/packages/WIPL_python_analysis/Complex_Multi_Body/bioscatterer_database_v0.001.nc')

def run_parallel_simulations(base_file_path):
    simulation_outputs = {}
    run_id_list = []

    # Use ProcessPoolExecutor for CPU-bound work
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(single_simulation_size_sweep_interface, counter): counter
            for counter in range(0, 11)
        }

        for future in as_completed(futures):
            run_id, cmb, results_df = future.result()
            simulation_outputs[run_id] = cmb
            run_id_list.append(run_id)
            results_df.to_csv(os.path.join(base_file_path, f'{run_id}.csv'))

    return simulation_outputs, run_id_list

def single_simulation_size_sweep_interface(counter):

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



    counter_complement = 10 - counter
    biological_params['size_distribution'] = [counter/10, counter_complement/10]

    run_id = f"{counter}_{counter_complement}_9_17"

    df = generate_distribution_df(radar_params, biological_params, run_id, save=False)
    cmb = run_simulation(df, scatterer_dataset.copy(), frequency=5.6)
    results_df = cmb.generate_df()

    results_df.attrs = {**biological_params, **radar_params}
    results_df.attrs['run_id'] = run_id
    
    return run_id, cmb, results_df

if __name__ == "__main__":
    base_file_path = './outputs/size_sweep/'
    os.makedirs(base_file_path, exist_ok=True)
    simulation_outputs, run_id_list = run_parallel_simulations(base_file_path)

