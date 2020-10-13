import numpy as np
import matplotlib.pyplot as plt
#import .functions
import os,sys
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
from functions.spectools import rotate,do_tilt, binData
from functions.fabio_add import find_nearest_index,multi_gaussian,all_file_names,load_dark,find_index_of_file,get_metadata,examine_current_trace
from PIL import Image
import xarray as xr
import pandas as pd
import copy
from scipy.optimize import curve_fit
from scipy import interpolate


merge=84
geom = pd.DataFrame(columns=['angle', 'tilt', 'binInterv', 'bin00a', 'bin00b'])
geom.loc[0] = [0.01759, np.nan, 29.202126, 53.647099, 53.647099]
geom['angle'][0]=0.0534850538519
geom['binInterv'] = 29.3051619433
geom['tilt'] = 0.0015142337977
geom['bin00a'] = 47.6592283252
geom['bin00b'] = 47.6592283252
waveLcoefs = np.ones((2, 3)) * np.nan
waveLcoefs[1] = [-1.39629946e-06, 1.09550955e-01, 3.49466935e+02]  # from examine_sensitivity.py
all_fits = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge) + '/merge' + str(merge) + '_all_fits.npy')
binnedSens = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/Sensitivity_2.npy')
binned_data = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge) + '/merge' + str(merge) + '_binned_data.npy')
inverted_profiles = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge) + '/inverted_profiles.npy')
merge_values = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge) + '/merge' + str(merge) + '_merge_tot.npz')['merge_values']
merge_time = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge) + '/merge' + str(merge) + '_merge_tot.npz')['merge_time']
merge_row = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge) + '/merge' + str(merge) + '_merge_tot.npz')['merge_row']
merge_Gain = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge) + '/merge' + str(merge) + '_merge_tot.npz')['merge_Gain']
merge_overexposed = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge) + '/merge' + str(merge) + '_merge_tot.npz')['merge_overexposed']


dx = 18 / 40 * (50.5 / 27.4) / 1e3
xx = np.arange(40) * dx  # m
xn = np.linspace(0, max(xx), 1000)
r = np.linspace(-max(xx) / 2, max(xx) / 2, 1000)
r=r[::10]

merge_time_window=[-10,10]
conventional_time_step = 0.05
new_timesteps = np.linspace(merge_time_window[0]+0.5, merge_time_window[1]-0.5,int((merge_time_window[1]-0.5-(merge_time_window[0]+0.5))/conventional_time_step+1))



grade_of_interpolation = 2  # this is the exponent used for the weights

rows_range_for_interp = 2
max_row_expansion = 2
time_range_for_interp = conventional_time_step / 2
max_time_expansion = time_range_for_interp
min_rows_for_interp = 4



composed_array = np.zeros((len(new_timesteps),1100,np.shape(merge_values)[1]))
interpolated_time=new_timesteps[0]
i=0
interpolated_row=200
j=200


selected_time = np.abs(merge_time - interpolated_time) <= time_range_for_interp
selected_row = np.abs(merge_row - interpolated_row) <= rows_range_for_interp
# selected_bin = np.logical_and(merge_row >= iBin[bin_bottom] ,  merge_row <= iBin[bin_top])
selected = selected_time*selected_row#*selected_bin
additional_time_range = 0
while (len(np.unique(merge_row[selected])) < min_rows_for_interp and additional_time_range<max_time_expansion):
# while (np.sum(selected) <= min_rows_for_interp and additional_time_range < max_time_expansion):
	additional_row_range = 0
	selected_time = np.abs(merge_time - interpolated_time) <= time_range_for_interp + additional_time_range
	selected_row = np.abs(merge_row - interpolated_row) <= rows_range_for_interp + additional_row_range
	selected = selected_time * selected_row# * selected_bin
	while (len(np.unique(merge_row[selected])) < min_rows_for_interp and additional_row_range<max_row_expansion):
	# while (np.sum(selected) <= min_rows_for_interp and additional_row_range < max_row_expansion):
		additional_row_range += 1
		print('interpolated_time ' + str(interpolated_time)+' row '+str(interpolated_row) + ' row increased of ' + str(additional_row_range) + str(np.unique(merge_row[selected])))
		selected_row = np.abs(merge_row - interpolated_row) <= rows_range_for_interp + additional_row_range
		selected = selected_time * selected_row# * selected_bin
		interpolation_borders_expanded[i, j] += 1

	if len(np.unique(merge_row[selected])) < min_rows_for_interp:
	# if np.sum(selected) <= min_rows_for_interp:
		additional_time_range += max_time_expansion / 10
		interpolation_borders_expanded[i, j] += 1
		print('interpolated_time ' + str(interpolated_time)+' row '+str(interpolated_row) + ' time increased of ' + str(additional_time_range) + str(np.unique(merge_row[selected])))



print('interpolated_time ' + str(interpolated_time) + ' row ' + str(interpolated_row) + ' num of rows ' + str(np.unique(merge_row[selected])))
weight_from_time_difference = 1/(np.abs(merge_time[selected] - interpolated_time)+np.max(np.abs(merge_time[selected] - interpolated_time))/100)
weight_from_time_difference = weight_from_time_difference/np.max(weight_from_time_difference)
weight_from_row_difference = 1/(np.abs(merge_row[selected] - interpolated_row)+np.max(np.abs(merge_row[selected] - interpolated_row))/100)
weight_from_row_difference = weight_from_row_difference/np.max(weight_from_row_difference)
# I want the penalty for time distance to be worst than row distance
weight_from_time_difference_multiplier = np.log(1.5*np.max(weight_from_row_difference)/np.min(weight_from_row_difference))/np.log(np.max(weight_from_time_difference)/np.min(weight_from_time_difference))
weight_from_time_difference = np.power(weight_from_time_difference,max(weight_from_time_difference_multiplier,1))
# weight_combined = weight_from_time_difference * weight_from_row_difference
weight_combined = np.sqrt(weight_from_time_difference**2 + (0.1*weight_from_row_difference)**2)
weight_combined = weight_combined/np.max(weight_combined)




def calc_stuff(z, i, j,grade_of_interpolation, values, weight_combined,merge_row_selected, composed_array):
	import numpy as np
	fit_coef_linear = np.polyfit(merge_row_selected, values, grade_of_interpolation, w=weight_combined)
	print([i, j,z])
	return np.polyval(fit_coef_linear, j)


from multiprocessing import Pool

pool = Pool(4)
composed_array[i, j] = zip(*pool.map(calc_stuff, range((1608)), np.ones((1608)) * i, np.ones((1608)) * j,np.ones((1608)) * grade_of_interpolation, merge_values[selected].T, np.ones((1608,1)) * weight_combined, np.ones((1608,1)) * merge_row[selected], composed_array[i].T))

gna = map(calc_stuff, range((1608)), np.ones((1608)) * i, np.ones((1608)) * j,np.ones((1608)) * grade_of_interpolation, merge_values[selected].T, np.ones((1608,1)) * weight_combined, np.ones((1608,1)) * merge_row[selected], composed_array[i].T)
gna2 = set(gna)