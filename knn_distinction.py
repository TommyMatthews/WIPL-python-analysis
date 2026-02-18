import numpy as np
from sklearn.neighbors import KDTree, KernelDensity
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

class KNNSeparator():
    """
    Class for quantifying separation of different scatterer groups.

    Parameters optimised on C-band, 5.6GHz, 0,10,20 pitches, and the following length combos:
    length_combos = [
        [2, 4, 6],
        [8, 10, 12],
        [14, 16, 18, 20 , 25],
        [35, 45, 50,75, 125, 200],
    ]

    """

    def __init__(self, results_dict, frequency, length_combos):
        
        self.results_dict = results_dict
        self.frequency = frequency
        self.length_combos = length_combos

        self.DATA_SELECTED = False
        self.BASE_GRID_GENERATED = False
        self.DENSITIES_CALCULATED = False

    def _calc_dp(self, ds, file_kwargs, phiT=0):
        
        scatterer_file = ds.sel(**file_kwargs)

        HH = scatterer_file['H_H_r'].to_numpy() + scatterer_file['H_H_i'].to_numpy() * 1j
        HV = scatterer_file['H_V_r'].to_numpy() + scatterer_file['H_V_i'].to_numpy() * 1j
        VH = scatterer_file['V_H_r'].to_numpy() + scatterer_file['V_H_i'].to_numpy() * 1j
        VV = scatterer_file['V_V_r'].to_numpy() + scatterer_file['V_V_i'].to_numpy() * 1j 



        phiT_rad = phiT*np.pi/180

        res_h = HH + VH*np.exp(1j*phiT_rad)
        res_v = (HV + VV*np.exp(1j*phiT_rad))*np.exp(1j*phiT_rad)

        horizontal_power = np.abs(res_h)**2
        vertical_power = np.abs(res_v)**2

        Zdr = 10*np.log10(horizontal_power/vertical_power)

        
        phiDP = np.angle((res_h * np.conj(res_v)), deg=True)
        phiDP = -phiDP

        return Zdr, phiDP
            


    def _generate_data_array(self, lengths, pitches, clipping=True, phiT=0, Zdr_max = 8, Zdr_min=-8):
        results_dict = self.results_dict
        frequency = self.frequency

        kwargs = {
            'slant' : 1, 
            'frequency' : frequency,
        }

        n_samples = len(lengths)*len(results_dict)*len(pitches)*72

        Zdr_array = np.zeros(n_samples)
        PhiDP_array = np.zeros(n_samples)

        length_list = []
        
        counter = 0

        for name in results_dict.keys():
            for pitch in pitches:

                kwargs['pitch'] = pitch

                for length in lengths:
                    default_length = length
                
                    ds = results_dict[name]
                    kwargs['length'] = default_length

                    Zdr, phiDP = self._calc_dp(ds, kwargs, phiT=phiT)

                    for Z, P in zip(Zdr, phiDP): 
                    
                        if (Z < Zdr_max) & (Z > Zdr_min):
                            Zdr_array[counter] = Z
                            PhiDP_array[counter] = P
                            # Zdr_list.append(Z)
                            # PhiDP_list.append(Z)
                            # length_list.append(length)
                        else:
                            Zdr_array[counter] = np.nan
                            PhiDP_array[counter] = np.nan
                        
                        # length_record[counter] = length

                        counter +=1       

        X = np.zeros((n_samples,2))

        if clipping:

            Zdr_array = np.clip(Zdr_array, -8,8)

        X[:,0] = Zdr_array
        X[:,1] = PhiDP_array


        return X, length_list

    @staticmethod
    def normalise(target, master):
        normalised = np.zeros(target.shape)
        for i in range(2):
            normalised[:,i] = (target[:,i]-master[:,i].min())/(master[:,i].max()-master[:,i].min())

        return normalised

    def _generate_array_list(self, calculation_kwargs, normalise_flag = True):

        array_list = []
        length_dist = []

        kwargs = calculation_kwargs.copy()
        pitches = kwargs.pop('pitches')

        if not isinstance(pitches,list):
            pitches = [pitches]

        for lengths in self.length_combos:
            X, ld = self._generate_data_array(lengths, pitches, **kwargs)
            array_list.append(X)
            length_dist.append(ld)

        array_list_nan_free = [x[~np.isnan(x[:,0]),:] for x in array_list]
        
        master = np.concatenate(array_list_nan_free)

        if normalise_flag:

            array_list_nan_free = [self.normalise(x, master) for x in array_list_nan_free]
            

        return array_list_nan_free, length_dist     


    def _calc_proportion_outside_overlap(self, inner_tree, outer_array, threshold, num_neighbours):
        indices = inner_tree.query_radius(outer_array, r=threshold)

        mask = np.array([len(neighbors) < num_neighbours for neighbors in indices])
        count =  mask.sum()

        matching_indices = np.where(mask)[0]

        print("Count and proportion of points outside distance:", count, count/len(outer_array))
        

        return matching_indices


    def _generate_selected_data(self, n_neighbours, threshold_1, threshold_2, calculation_kwargs, normalise_flag=True):

        """
        Example calculation_kwargs:
        kwargs = {
            'pitches' : pitches,
            'Zdr_max' : 8,
            'Zdr_min' :-8,
            'phiT' : 30
            }

        """


        arrays, _ = self._generate_array_list(calculation_kwargs, normalise_flag)

        tree_0 = KDTree(arrays[0])

        indices_l1 = self._calc_proportion_outside_overlap(tree_0, arrays[1], threshold_1, n_neighbours)

        cumulative_array = np.concatenate([
                arrays[1][indices_l1,:],
                arrays[0]
                ])

        updated_tree = KDTree(cumulative_array)

        indices_l2 = self._calc_proportion_outside_overlap(updated_tree, arrays[2], threshold_2, n_neighbours)

        cumulative_array = np.concatenate([
                arrays[2][indices_l2,:],
                arrays[1],
                arrays[0],
                ])

        updated_tree = KDTree(cumulative_array)

        indices_l3 = self._calc_proportion_outside_overlap(updated_tree, arrays[3], threshold_2, n_neighbours)


        l1_selected = arrays[1][indices_l1,:]
        l2_selected = arrays[2][indices_l2,:]
        l3_selected = arrays[3][indices_l3,:]

        selected_data = [arrays[0], l1_selected, l2_selected, l3_selected]

        return selected_data
    

    def quantify_category_distinction(self, n_neighbours, threshold_1, threshold_2, calculation_kwargs, normalise_flag=True, return_data = False):

        print('Delta Zdr and PhiDP required for separation #1:', threshold_1 * 16, 'dB,', threshold_1 *360, 'degrees')

        print('Delta Zdr and PhiDP required for separation #2:', threshold_2 * 16, 'dB,', threshold_2 *360, 'degrees')

        selected_data = self._generate_selected_data(n_neighbours, threshold_1, threshold_2, calculation_kwargs, normalise_flag)

        self.last_selected_data = selected_data

        self.DATA_SELECTED = True

        if return_data:
            return selected_data

    def scatter_plot(self, selected_data = None, indices_to_plot = [0,1,2,3],alpha=0.2):

        if not selected_data:

            if not self.DATA_SELECTED:
                print("RUN 'quantify_category_distinction()' method first!")
                raise Exception("You must run 'quantify_category_distinction' before trying to plot the results")

            selected_data = self.last_selected_data

        if not isinstance(indices_to_plot,list):
            indices_to_plot == list(indices_to_plot)

        colours = [
            'red',
            'blue',
            'green',
            'orange'
        ]

        for index in indices_to_plot[::-1]:
            plt.scatter(selected_data[index][:,0],selected_data[index][:,1], alpha = alpha, color=colours[index])

        plt.grid()
        plt.show()

        # densities, kdes, bandwidths, coords = full_NN_KDE_pipeline_to_densities(selected_data)
        # plot_KDEs_from_densities(densities, coords, 10e-6)

    #Now for KDES,less important, get above working first
    def _generate_base_KDE_grid(self, n_X = 200, n_Y = 200):
        "Currently assuming normalised"

        self.BASE_GRID_GENERATED = True

        x_grid = np.linspace(0, 1, 200)
        y_grid = np.linspace(0, 1, 200)

        X, Y = np.meshgrid(x_grid, y_grid)

        self.X = X
        self.Y = Y
        self.grid_samples = np.vstack([X.ravel(), Y.ravel()]).T


    def _KDE_pipeline_to_densities(self, array_list = None, return_data = None):
        
        if not array_list:
            array_list = self.last_selected_data

        if not self.BASE_GRID_GENERATED:
            self._generate_base_KDE_grid()

        print("Running grid search")

        bandwidths = []

        for index, array in enumerate(array_list):
            params = {'bandwidth': np.logspace(-3, 0, 30)}

            grid = GridSearchCV(
                KernelDensity(kernel='gaussian'),
                params,
                cv=5
            )

            grid.fit(array)

            print(f"Best bandwidth {index}:", grid.best_estimator_.bandwidth)
            bandwidths.append(grid.best_estimator_.bandwidth)

        fitted_kdes = []
        densities = []

        for index, array in enumerate(array_list):
            kde = KernelDensity(
            bandwidth=bandwidths[index],
            kernel='gaussian')

            kde.fit(array_list[index])

            fitted_kdes.append(kde)

            log_density = kde.score_samples(self.grid_samples)
            density = np.exp(log_density)

            density = density.reshape(self.X.shape)

            densities.append(density)

        print("Caching densities")
        self.last_densities = densities
        self.last_kdes = fitted_kdes
        self.last_bandwidths = bandwidths

        self.DENSITIES_CALCULATED = True

        if return_data:
            return densities, fitted_kdes, bandwidths
        

    def check_densities(self, densities):

        if not densities:
            
            if not self.DENSITIES_CALCULATED:
                print("Calculating probability densities")
                self._KDE_pipeline_to_densities()

            densities = self.last_densities

        return densities


    def plot_KDEs_separate(self, densities = None, lower_threshold = 10e-6, upper_threshold = None):

        densities = self.check_densities(densities)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout = 'constrained')

        axes = np.asarray(axes).ravel()

        for index, ax in enumerate(axes):
            density = densities[index]

            if lower_threshold:
                density = np.where(density>lower_threshold, density, np.nan)

            if upper_threshold:
                density = np.where(density<upper_threshold, density, np.nan)

            ax.contourf(self.X, self.Y, density, levels=40, cmap='viridis')

            ax.set_title(f"KDE of {index}")
            ax.set_xlabel("Normalised PhiDP")
            ax.set_ylabel("Normalised Zdr")
            ax.grid()
        
        plt.show()  

    
    def plot_KDEs_together(self, densities = None, lower_threshold = 1, upper_threshold = None, indices_to_plot = [0,1,2,3]):
        
        densities = self.check_densities(densities)

        fig, ax = plt.subplots(1, 1, figsize=(7, 7), layout = 'constrained')

        #axes = np.asarray(axes).ravel()

        maps=['Purples', 'Reds', 'Blues', 'Greens']

        for index in np.arange(4)[::-1]:
            density = densities[index]

            if lower_threshold:

                density = np.where(density>lower_threshold, density, np.nan)

            if upper_threshold:
                density = np.where(density<upper_threshold, density, np.nan)

            ax.contourf(self.X, self.Y, density, levels=40, cmap=maps[index])
        
            

            # overlay original points
            #
        ax.set_title(f"Combined KDE plot")
        ax.set_xlabel("Normalised Zdr")
        ax.set_ylabel("Normalised PhiDP")

        # plt.colorbar()
        plt.show()  



    def full_analysis(self,
                      n_neighbours = 2,
                      threshold_1 = 0.01,
                      threshold_2 = 0.02,
                      calculation_kwargs = {},
                      scatter_plot = True, 
                      separate_kde_plots = True, 
                      combined_kde_plot = True,
                      scatter_plot_kwargs = {},
                      separate_kde_plot_kwargs = {},
                      combined_kde_plot_kwargs = {},
                      ):
        
        # required_keys = ['pitches', 'Zdr_max', 'Zdr_min', 'phiT']

        # for key in required_keys:

        #     if key not in calculation_kwargs.keys():
        #         raise ValueError(f'{key} not present in calculation kwargs; calculation kwargs must contain {required_keys}')

        self.quantify_category_distinction(
            n_neighbours,
            threshold_1,
            threshold_2,
            calculation_kwargs = calculation_kwargs
            )
        
        if scatter_plot:
            self.scatter_plot(**scatter_plot_kwargs)

        if separate_kde_plots:
            self.plot_KDEs_separate(**separate_kde_plot_kwargs)

        if combined_kde_plot:
            self.plot_KDEs_together(**combined_kde_plot_kwargs)




