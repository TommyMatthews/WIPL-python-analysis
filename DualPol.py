import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DualPol:
    def __init__(self, horizontal_csv_file, vertical_csv_file, hacky_fix = False, read_from_df = False):
        
        if read_from_df:
            temp = horizontal_csv_file
            self.vertical_data = vertical_csv_file
            

        else:
        
            temp = pd.read_csv(horizontal_csv_file)
            self.vertical_data = pd.read_csv(vertical_csv_file)

        ##########################################################
        # Remove this hacky fix
        if hacky_fix:
            self.horizontal_data = temp.iloc[::2]

        else:
            self.horizontal_data = temp

        self.resultant_fields_calculated = False
        self.differential_reflectivity_calculated = False
        self.differential_phase_calculated = False
        self.phase_calculated = False

    def _calculate_resultant_fields(self):

        horizontal_e_phi = self.horizontal_data['Ephi'].astype(complex).to_numpy()
        horizontal_e_theta = self.horizontal_data['Etheta'].astype(complex).to_numpy()
        vertical_e_phi = self.vertical_data['Ephi'].astype(complex).to_numpy()
        vertical_e_theta = self.vertical_data['Etheta'].astype(complex).to_numpy()


        self.resultant_theta_field = (
            horizontal_e_theta + vertical_e_theta#*np.exp(2j*np.pi/3)
        )#*np.exp(2j*np.pi/3)
        self.resultant_phi_field = (
            horizontal_e_phi + vertical_e_phi#*np.exp(2j*np.pi/3)
        )

        self.resultant_fields_calculated = True

    def _calculate_phases(self):

        if not self.resultant_fields_calculated:
            self._calculate_resultant_fields()

        self.resultant_theta_phase = np.angle(self.resultant_theta_field) * 180 / np.pi
        self.resultant_phi_phase = np.angle(self.resultant_phi_field) * 180 / np.pi

    def _de_alias(self, aliased_data):

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
    
    def _crude_de_alias(self, aliased_data):

        return np.where(aliased_data < -90, aliased_data + 360, aliased_data)

    def _resultant_fields_check(self):
        if not self.resultant_fields_calculated:
            self._calculate_resultant_fields()

    def _calculate_differential_reflectivity(self):

        self._resultant_fields_check()

        relative_theta_power = np.abs(self.resultant_theta_field)**2
        relative_phi_power = np.abs(self.resultant_phi_field)**2

        self.differential_reflectivity = 10 * np.log10(
            relative_phi_power / relative_theta_power
        )

        self.differential_reflectivity_calculated = True

    def _calculate_differential_phase(self, de_alias = False):

        self._resultant_fields_check()

        differential_phase_in_radians = np.angle(self.resultant_phi_field) - np.angle(self.resultant_theta_field)
        self.differential_phase = differential_phase_in_radians * 180 / np.pi

        if de_alias:
            self.differential_phase = self._de_alias(self.differential_phase)

        self.differential_phase_calculated = True    

    def plot_differential_reflectivity(self, title=None):

        if not self.differential_reflectivity_calculated:
            self._calculate_differential_reflectivity()

        phi = self.horizontal_data["phi"]
        plt.figure(figsize=(8, 6))
        plt.plot(phi, self.differential_reflectivity, label="Differential Reflectivity")
        plt.xlabel("Azimuth (degrees)")
        plt.ylabel("Differential Reflectivity (dB)")

        if title:
            plt.title(title)

        plt.legend()
        plt.show()

    def plot_differential_phase(self, title=None, de_alias = False):
        
        
        self._calculate_differential_phase(de_alias=de_alias)

        phi = self.horizontal_data["phi"]
        plt.figure(figsize=(8, 6))
        plt.plot(phi, self.differential_phase, label="Differential Phase")
        plt.xlabel("Azimuth (degrees)")
        plt.ylabel("Differential Phase (radians)")

        if title:
            plt.title(title)

        plt.legend()
        plt.show()