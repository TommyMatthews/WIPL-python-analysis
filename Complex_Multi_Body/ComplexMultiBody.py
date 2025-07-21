import pandas as pd
import numpy as np

class ComplexMultiBody:
    """
    Development of the MultiBody class to handle more complex multi-body scenarios.
    """


    def __init__(self, distribution_df, frequency, scatterer_dataset = None):

        self.distribution_df = distribution_df
        self.scatterer_dataset = scatterer_dataset
        self.frequency = frequency

    def _convert_relative_distances_to_phase_shifts(self):
        """
        Convert relative distances to phase shifts based on the frequency.
        """

        c = 3e8  # Speed of light in m/s
        wavelength = (c / (self.frequency * 1e9)) # Convert GHz to Hz, then calculate wavelength in meters
        self.distribution_df['phase_shift'] = (self.distribution_df['relative_distance_along_beam'] / wavelength) * 2 * np.pi  # Convert to radians

    def _extract_single_body_results(self):
        """
        Placeholder method to extract single body results 
        """
        self.single_scatterer_profiles = {}
        
        for key in self.distribution_df['scatterer_id'].unique():

            ## This will need to read in the base scatterer and then correct for azimuth (probably in a spearate method)
            single_body_results = np.zeros((180, 4), dtype=complex)
            
            single_body_results[:,0] = self.scatterer_dataset['H_H'].astype(complex).to_numpy()[0:-1]
            single_body_results[:,1] = self.scatterer_dataset['H_V'].astype(complex).to_numpy()[0:-1]
            single_body_results[:,2] = self.scatterer_dataset['V_H'].astype(complex).to_numpy()[0:-1]
            single_body_results[:,3] = self.scatterer_dataset['V_V'].astype(complex).to_numpy()[0:-1]
    

            self.single_scatterer_profiles[key] = single_body_results
            

    def _perform_phase_shifts(self):
        """
        Apply phase shifts to the single body results based on the relative distances.
        
        The distance to each body is the distance GREATER that the wave has to travel, so we delay the phase by multiplying by a positive phasor
        
        A factor of 2 is included to account for the round trip distance
        """

        self._convert_relative_distances_to_phase_shifts()
        self._extract_single_body_results()

        self.full_results_shifted = np.zeros((len(self.distribution_df), 4, 180), dtype=complex)

        row_counter = 0
        for scatterer_row_index, row in self.distribution_df.iterrows():
            scatterer_id = row['scatterer_id']
            phase_shift = row['phase_shift']
            
            single_body_results = self.single_scatterer_profiles[scatterer_id]


            for scattering_result_index in range(4):
                for azimuth in range(180):
                    self.full_results_shifted[row_counter, scattering_result_index, azimuth] = (
                        single_body_results[azimuth, scattering_result_index] * np.exp(2j * phase_shift)
                    )

            row_counter += 1

    def calculate_resultant_voltages(self):
        """
        Summing across all bodies to get the resultant voltages, then summing within each channel to get the final voltages.
        """

        self._perform_phase_shifts()

        self.resultant_voltage_phi = np.zeros(180, dtype=complex)  
        self.resultant_voltage_theta = np.zeros(180, dtype=complex)

        self.resultant_voltage_phi = np.sum(self.full_results_shifted[:, 0, :], axis=0) + np.sum(self.full_results_shifted[:, 2, :], axis=0)
        self.resultant_voltage_theta = np.sum(self.full_results_shifted[:, 1, :], axis=0) + np.sum(self.full_results_shifted[:, 3, :], axis=0)

    

    def _de_alias(self, aliased_data): #IMPLEMENT THIS!!!

        number_of_points = len(aliased_data)
        de_aliased_data = np.zeros(number_of_points)

        de_aliased_data[0] = aliased_data[0]

        for counter in range(1, number_of_points):

            difference = aliased_data[counter] - de_aliased_data[counter-1]
            if np.abs(difference) > 180:
                if difference > 0:
                    de_aliased_data[counter] = aliased_data[counter] - 360
                elif difference < 0:
                    de_aliased_data[counter] = aliased_data[counter] + 360
            else:
                de_aliased_data[counter] = aliased_data[counter]

        return de_aliased_data
    
    def _dual_pol_calcs_on_recieved_voltages(self):
        
        """
        Calculate the differential phase, differential reflectivity, and linear depolarization ratio
        """
        self.horizontal_power = np.abs(self.resultant_voltage_phi)**2
        self.vertical_power = np.abs(self.resultant_voltage_theta)**2

        self.differential_reflectivity = np.zeros(180)
        self.differential_phase = np.zeros(180)
        self.differential_phase_de_aliased = np.zeros(180)


        for phi in range(180):
            self.differential_reflectivity[phi] = 10 * np.log10(self.horizontal_power[phi] / self.vertical_power[phi])
            self.differential_phase[phi] = np.rad2deg(np.angle(self.resultant_voltage_phi[phi]) - np.angle(self.resultant_voltage_theta[phi]))
                
        self.differential_phase_de_aliased = self._de_alias(self.differential_phase)
