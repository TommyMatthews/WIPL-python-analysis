import pandas as pd
import numpy as np


class MultiBody:
    """
    Class to combine outputs from multiple bodies at different locations.
    """


    def __init__(self, single_body_horizontal_df, single_body_vertical_df, coordinate_file, frequency,slant = 0):

        self.single_body_horizontal_df = single_body_horizontal_df
        self.single_body_vertical_df = single_body_vertical_df
        self.coordinate_file = coordinate_file
        self.slant = slant

        self.frequency = frequency #GHz

        self.phi_array = np.array(self.single_body_horizontal_df['phi'])
        self.num_combos, num_columns = self.coordinate_file.shape
        self.num_bodies = num_columns // 3
        
    def _calculate_direction_unit_vectors(self):

        "Calculate directional unit vectors for each incident phi direction"

        self.direction_unit_vectors = np.zeros((len(self.phi_array), 3))
        for i in range(len(self.phi_array)):
            phi = self.phi_array[i]
            theta = -1*self.slant
            x = np.cos(np.radians(theta)) * np.cos(np.radians(phi))
            y = np.cos(np.radians(theta)) * np.sin(np.radians(phi))
            z = np.sin(np.radians(theta))
            self.direction_unit_vectors[i] = [x, y, z]

    def _calculate_body_distance_vectors(self):
        "Calculate the distance vectors from the origin to each body location"

        self.body_distance_vectors = np.zeros((self.num_combos, 3, self.num_bodies))
        coord_df = self.coordinate_file
        for i in range(self.num_combos):
            x2, y2, z2 = coord_df['x2'][i], coord_df['y2'][i], coord_df['z2'][i]
            x3, y3, z3 = coord_df['x3'][i], coord_df['y3'][i], coord_df['z3'][i]
            self.body_distance_vectors[i, :, 0] = x2, y2, z2
            self.body_distance_vectors[i, :, 1] = x3, y3, z3

    def _calculate_projection_of_distance_onto_direction_unit_vectors(self):
        """
        Calculate the projection of the distance vectors onto the direction unit vectors
        
        With this sign, this tells you how much GREATER the path to the body for the incident wave, relative to the path to the origin

        units of mm!
        """
  

        self.projected_distances = np.zeros((self.num_combos, self.num_bodies, len(self.phi_array)))
        for i in range(self.num_combos):
            for j in range(self.num_bodies):
                for k in range(len(self.phi_array)):
                    self.projected_distances[i, j, k] = np.dot(
                        self.body_distance_vectors[i, :, j],
                        self.direction_unit_vectors[k]
                    )

    def _convert_relative_distances_to_phase_shifts(self):
        """
        Convert the relative distances to phase shifts
        """
        
        c = 3e8  # Speed of light in m/s
        wavelength_mm = (c / (self.frequency * 1e9)) * 1e3  # Convert GHz to Hz, then m to mm
        self.phase_shifts =(self.projected_distances / wavelength_mm) * 2 * np.pi  # Convert to radians
        

    def _calculate_phase_shifts(self):
        "Calculate, for each body location, the phase shift to apply to the scattered WIPL results"

        self._calculate_direction_unit_vectors()
        self._calculate_body_distance_vectors()
        self._calculate_projection_of_distance_onto_direction_unit_vectors()
        self._convert_relative_distances_to_phase_shifts()


    def _extract_single_body_results(self):
        """
        Extract the results from the single body WIPL runs
        """

        self.single_body_results = np.zeros((len(self.phi_array), 4), dtype=complex)

        self.single_body_results[:,0] = self.single_body_horizontal_df['Ephi'].astype(complex).to_numpy()
        self.single_body_results[:,1] = self.single_body_horizontal_df['Etheta'].astype(complex).to_numpy()
        self.single_body_results[:,2] = self.single_body_vertical_df['Ephi'].astype(complex).to_numpy()
        self.single_body_results[:,3] = self.single_body_vertical_df['Etheta'].astype(complex).to_numpy()



    def _perform_phase_shifts(self):
        """
        The distance to each body is the distance GREATER that the wave has to travel, so we delay the phase by multiplying by a positive phasor
        
        A factor of 2 is included to account for the round trip distance
        """

        self._calculate_phase_shifts()
        self._extract_single_body_results()

        self.full_results_shifted = np.zeros((self.num_combos, self.num_bodies, len(self.phi_array), 4), dtype=complex)
        
        for i in range(self.num_combos):
            for j in range(self.num_bodies):
                for k in range(len(self.phi_array)):
                    self.full_results_shifted[i, j, k, :] = self.single_body_results[k, :] * np.exp(2j * self.phase_shifts[i, j, k])


    def _calculate_resultant_voltages(self):
        """
        Calculate the resultant voltages from the bodies for each combination. 

        Include the presence of one body at the origin, which is the single body results
        The resultant voltages are calculated by summing the voltages from each body, including the phase shift

        Aiming to reduce down to 4 voltages (as a function of incident azimuth) for each combination

        The resultant theta and phi voltages can then be calculated from each combination
        """

        # Reduce down to one set of voltages for each combination

        self.resultant_voltages = np.zeros((self.num_combos, len(self.phi_array), 4), dtype=complex)

        for i in range(self.num_combos):
            for j in range(len(self.phi_array)):
                self.resultant_voltages[i, j, :] = np.sum(self.full_results_shifted[i, :, j, :], axis=0) + self.single_body_results[j, :]
            
        # Combine into just theta and phi voltages
        self.resultant_voltages_theta = self.resultant_voltages[:, :, 1] + self.resultant_voltages[:, :, 3]
        self.resultant_voltages_phi = self.resultant_voltages[:, :, 0] + self.resultant_voltages[:, :, 2]

    def calculate_rho_hv(self):

        horizontal_phi = np.zeros((self.num_combos, 181))
        vertical_theta = np.zeros((self.num_combos, 181))

        horizontal_phi = self.resultant_voltages[:, :, 0]
        vertical_theta = self.resultant_voltages[:, :, 3]

        rho_hv = np.zeros((self.num_combos, 181))

        # for coordinate_row in range(self.num_combos):
        #     horizontal_phi[coordinate_row] = data_dict['H'][coordinate_row]['Ephi']
        #     vertical_theta[coordinate_row] = data_dict['V'][coordinate_row]['Etheta']

        coordinate_level_correlation = np.conj(horizontal_phi) * vertical_theta
        coordinate_level_hh_power = np.abs(horizontal_phi) ** 2
        coordinate_level_vv_power = np.abs(vertical_theta) ** 2

        mean_correlation = np.mean(coordinate_level_correlation, axis=0)
        mean_hh_power = np.mean(coordinate_level_hh_power, axis=0)
        mean_vv_power = np.mean(coordinate_level_vv_power, axis=0)
        rho_hv = np.abs(mean_correlation) / np.sqrt(mean_hh_power * mean_vv_power)
        return rho_hv

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
        self.horizontal_power = np.abs(self.resultant_voltages_phi)**2
        self.vertical_power = np.abs(self.resultant_voltages_theta)**2

        self.differential_reflectivity = np.zeros((self.num_combos, len(self.phi_array)))
        self.differential_phase = np.zeros((self.num_combos, len(self.phi_array)))
        self.differential_phase_de_aliased = np.zeros((self.num_combos, len(self.phi_array)))


        for i in range(self.num_combos):
            for phi in range(len(self.phi_array)):
                self.differential_reflectivity[i,phi] = 10 * np.log10(self.horizontal_power[i, phi] / self.vertical_power[i, phi])
                self.differential_phase[i,phi] = np.rad2deg(np.angle(self.resultant_voltages_phi[i, phi]) - np.angle(self.resultant_voltages_theta[i, phi]))
                
            self.differential_phase_de_aliased[i,:] = self._de_alias(self.differential_phase[i, :])


