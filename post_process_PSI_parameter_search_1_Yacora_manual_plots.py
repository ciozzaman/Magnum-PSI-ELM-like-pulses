# 27/02/2020
# This is an extract from post_process_PSI_parameter_search_1_Yacora_plots.py
# only to make a manual plot for Bruce

import numpy as np
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())
# import matplotlib.pyplot as plt
#import .functions
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
from functions.fabio_add import find_index_of_file
import collections

from adas import read_adf15,read_adf11
import os,sys
from PIL import Image
import xarray as xr
import pandas as pd
import copy
from scipy.optimize import curve_fit,least_squares
from scipy import interpolate
from scipy.signal import find_peaks
import time as tm
from matplotlib import ticker
from multiprocessing import Pool,cpu_count
number_cpu_available = cpu_count()
print('Number of cores available: '+str(number_cpu_available))


# n = np.arange(10, 20)
# waveLengths = [656.45377,486.13615,434.0462,410.174,397.0072,388.9049,383.5384] + list((364.6*(n*n)/(n*n-4)))
waveLengths = [486.13615,434.0462,410.174,397.0072,388.9049,383.5384]
waveLengths_interp = interpolate.interp1d(waveLengths[:2], [1.39146712e+03,8.83761543e+02],fill_value='extrapolate')
#pathfiles = '/home/ff645/GoogleDrive/ff645/YPI - Fabio/Collaboratory/Experimental data/tests/2019-04-25/final_test/Untitled_1/Pos0'
# pathfiles = '/home/ff645/GoogleDrive/ff645/YPI - Fabio/Collaboratory/Experimental data/tests/temp - Desktop pc/Untitled_11/Pos0'
# pathfiles = '/home/ff645/GoogleDrive/ff645/YPI - Fabio/Collaboratory/Experimental data/tests/test_data_folder/2019-04-25/01/Untitled_1/Pos0'

fdir = '/home/ffederic/work/Collaboratory/test/experimental_data'
df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv', index_col=0)
df_settings = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_3.csv',index_col=0)

exec(open("/home/ffederic/work/Collaboratory/test/experimental_data/functions/MolRad_Yacora/Yacora_FF/import_PECs_FF_2.py").read())

pecfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/pec12#h_balmer#h0.dat'
scdfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/scd12_h.dat'
acdfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/acd12_h.dat'

path_where_to_save_plots = '/home/ffederic/work/Collaboratory/test/experimental_data/global_check_physics'
if not os.path.exists(path_where_to_save_plots):
	os.makedirs(path_where_to_save_plots)

figure_index = 0
# for max_n_neutrals_ne in [1,10, np.inf]:
for max_n_neutrals_ne in [1]:

	merge_ne_prof_multipulse_interp_crop_limited_multipulse = []
	merge_dne_prof_multipulse_interp_crop_limited_multipulse = []
	merge_Te_prof_multipulse_interp_crop_limited_multipulse = []
	merge_dTe_prof_multipulse_interp_crop_limited_multipulse = []

	nH_ne_all_multipulse = []
	nHp_ne_all_multipulse = []
	nHm_ne_all_multipulse = []
	nH2_ne_all_multipulse = []
	nH2p_ne_all_multipulse = []
	nH3p_ne_all_multipulse = []
	residuals_all_multipulse = []

	effective_ionisation_rates_multipulse = []
	effective_recombination_rates_multipulse = []
	ionization_length_multipulse = []

	rate_creation_Hm_multipulse = []
	rate_destruction_Hm_multipulse = []


	merge_ID_target_multipulse = [851,86,87,89,851,92, 93, 94]
	# merge_ID_target_multipulse = [851,86,87,89]


	inverted_profiles_crop_multipulse = []

	target_chamber_pressure_multipulse = []
	target_OES_distance_multipulse = []

	for merge_ID_target in merge_ID_target_multipulse:  # 88 excluded because I don't have a temperature profile
		if (merge_ID_target>=93 and merge_ID_target<100):
			merge_time_window = [-1,2]
		else:
			merge_time_window = [-10,10]

		for i in range(10):
			print('.')
		print('Starting to work on merge number' + str(merge_ID_target))
		for i in range(10):
			print('.')

		time_resolution_scan = False
		time_resolution_scan_improved = True
		time_resolution_extra_skip = 0



		if True:
			n_list = np.array([5, 6, 7,8, 9,10,11])
			n_list_1 = np.array([4])
			n_weights = [1, 1, 1, 1, 1, 1,3,3]
		else:
			n_list = np.array([6,7,8, 9])
			n_list_1 = np.array([4,5])
			n_weights = [1, 1, 1, 1,1, 3]



		max_nHp_ne = 1
		min_nHp_ne,atomic_restricted_high_n = [0.999,False]
		# max_n_neutrals_ne = np.inf

		mod_atomic_restricted_high_n = ''
		if atomic_restricted_high_n ==True:
			mod_atomic_restricted_high_n = '_atomic_restricted_high_n'
			print('calculus of nH and nH+ limited to lines >=8')

		mod = '/Yacora_Bayesian/absolute/lines_fitted'+str(len(n_list)+len(n_list_1))+'/min_nHp_ne' + str(min_nHp_ne) + mod_atomic_restricted_high_n+'/max_n_neutrals_ne' + str(
			max_n_neutrals_ne)

		path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) #+ '_back'


		spatial_factor = 1
		time_shift_factor = 0

		mod2 = mod + '/spatial_factor_' + str(spatial_factor) + '/time_shift_factor_' + str(time_shift_factor)



		min_multiplier = 1e-10
		# for max_nHm_ne,max_nH2_ne,max_nH2p_ne,max_nH3p_ne,mod3 in [[min_multiplier,min_multiplier,min_multiplier,min_multiplier,'only_H'],[1,min_multiplier,min_multiplier,min_multiplier,'only_Hm'],[min_multiplier,1,min_multiplier,min_multiplier,'only_H2'],[min_multiplier,min_multiplier,1,min_multiplier,'only_H2p'],[min_multiplier,min_multiplier,min_multiplier,1,'only_H3p'],[min_multiplier,1,1,min_multiplier,'only_H2_H2p'],[1,1,min_multiplier,min_multiplier,'only_Hm_H2'],[1,1,1,min_multiplier,'only_Hm_H2_H2p'],[1,1,1,1,'all']]:
		max_nHm_ne,max_nH2_ne,max_nH2p_ne,max_nH3p_ne,mod3 = [1,1,1,min_multiplier,'only_Hm_H2_H2p']
		mod4 = mod2 +'/' +mod3



		print('min_nHp/ne')
		print(min_nHp_ne)
		print('max_nH/ne')
		print(max_n_neutrals_ne)
		print('max_nHp/ne')
		print(max_nHp_ne)
		print('Starting '+mod3)



		boltzmann_constant_J = 1.380649e-23	# J/K
		eV_to_K = 8.617333262145e-5	# eV/K
		avogadro_number = 6.02214076e23
		all_j=find_index_of_file(merge_ID_target,df_settings,df_log,only_OES=True)
		target_chamber_pressure = []
		target_OES_distance = []
		for j in all_j:
			target_chamber_pressure.append(df_log.loc[j,['p_n [Pa]']])
			target_OES_distance.append(df_log.loc[j,['T_axial']])
		target_chamber_pressure = np.nanmean(target_chamber_pressure)	# Pa
		target_OES_distance = np.nanmean(target_OES_distance)	# Pa
		# Ideal gas law
		max_nH2_from_pressure = target_chamber_pressure/(boltzmann_constant_J*300)	# [#/m^3] I suppose ambient temp is ~ 300K


		print('lines')
		print(str(n_list_1.tolist()+n_list.tolist()))
		print('n_weights')
		print(n_weights)
		print('max_nH2 [#/m^3]')
		print(max_nH2_from_pressure)

		started = 0
		rows_range_for_interp = 25 / 3  # rows that I use for interpolation (box with twice this side length, not sphere)
		# if merge_ID_target>=66:
		# 	rows_range_for_interp = geom_store['binInterv'][0] / 6
		if time_resolution_scan:
			conventional_time_step = 0.01  # ms
		else:
			conventional_time_step = 0.05  # ms
		# interpolation_type = 'quadratic'	# 'linear' or 'quadratic'
		grade_of_interpolation = 3  # this is the exponent used for the weights of the interpolation for image resampling
		type_of_image = '12bit'  # '12bit' or '16bit'
		# if type_of_image=='12bit':
		row_shift = 2 * 10280 / 1000000  # ms
		# elif type_of_image=='16bit':
		# 	print('Row shift to be checked')
		# 	exit()
		# time_range_for_interp = rows_range_for_interp*row_shift
		# merge_time_window=[-1,4]
		overexposed_treshold = 3600

		new_timesteps = np.linspace(merge_time_window[0] + 0.5, merge_time_window[1] - 0.5, int(
			(merge_time_window[1] - 0.5 - (merge_time_window[0] + 0.5)) / conventional_time_step + 1))
		dt = np.nanmedian(np.diff(new_timesteps))

		# for spatial_factor in spatial_factor_all:
		#
		# 	mod = '/spatial_factor' + str(spatial_factor)
		# 	if not os.path.exists(path_where_to_save_everything + mod):
		# 		os.makedirs(path_where_to_save_everything + mod)

		# dx = (18 / 40 * (50.5 / 27.4) / 1e3) * spatial_factor
		# xx = np.arange(40) * dx  # m
		# xn = np.linspace(0, max(xx), 1000)
		# r = np.linspace(-max(xx) / 2, max(xx) / 2, 1000)
		# r=r[::10]
		# dr=np.median(np.diff(r))

		# first_time = np.min(new_timesteps)
		# last_time = np.max(new_timesteps)

		# Data from wikipedia
		energy_difference = np.array([1.89, 2.55, 2.86, 3.03, 3.13, 3.19, 3.23, 3.26, 3.29])  # eV
		# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
		statistical_weigth = np.array([32, 50, 72, 98, 128, 162, 200, 242, 288])  # gi-gk
		einstein_coeff = np.array(
			[8.4193e-2, 2.53044e-2, 9.7320e-3, 4.3889e-3, 2.2148e-3, 1.2156e-3, 7.1225e-4, 4.3972e-4,
			 2.8337e-4]) * 1e8  # 1/s
		J_to_eV = 6.242e18
		au_to_kg = 1.66053906660e-27	# kg/au
		# Used formula 2.3 in Rion Barrois thesys, 2017
		color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro',
				 'paleturquoise']

		inverted_profiles_original = 4 * np.pi * np.load(path_where_to_save_everything + '/inverted_profiles.npy')  # in W m^-3 sr^-1  *  4*pi sr  =  W m^-3
		all_fits = np.load(path_where_to_save_everything + '/merge' + str(merge_ID_target) + '_all_fits.npy')  # in W m^-2 sr^-1
		merge_Te_prof_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_Te_prof_multipulse']
		merge_dTe_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_dTe_multipulse']
		merge_ne_prof_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_ne_prof_multipulse']
		merge_dne_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_dne_multipulse']
		merge_time_original = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_time']

		# for time_shift_factor in time_shift_factor_all:
		#
		# 	mod = '/spatial_factor' + str(spatial_factor)+'/time_shift_factor' + str(time_shift_factor)
		# 	if not os.path.exists(path_where_to_save_everything + mod):
		# 		os.makedirs(path_where_to_save_everything + mod)



		# dx = 18 / 40 * (50.5 / 27.4) / 1e3
		dx = 1.06478505470992 / 1e3	# 10/02/2020 from	Calculate magnification_FF.xlsx
		xx = np.arange(40) * dx  # m
		xn = np.linspace(0, max(xx), 1000)
		r = np.linspace(-max(xx) / 2, max(xx) / 2, 1000)
		r = r[::10]
		dr = np.median(np.diff(r))

		merge_time = time_shift_factor + merge_time_original
		inverted_profiles = (1 / spatial_factor) * inverted_profiles_original  # in W m^-3 sr^-1  *  4*pi sr  =  W m^-3
		TS_dt = np.nanmedian(np.diff(merge_time))

		if np.max(merge_Te_prof_multipulse) <= 0:
			print('merge' + str(merge_ID_target) + " has no recorded temperature")
			continue

		TS_size = [-4.149230769230769056e+01, 4.416923076923076508e+01]
		TS_r = TS_size[0] + np.linspace(0, 1, 65) * (TS_size[1] - TS_size[0])
		TS_dr = np.median(np.diff(TS_r)) / 1000
		gauss = lambda x, A, sig, x0: A * np.exp(-((x - x0) / sig) ** 2)
		profile_centres = []
		profile_centres_score = []
		for index in range(np.shape(merge_Te_prof_multipulse)[0]):
			yy = merge_Te_prof_multipulse[index]
			p0 = [np.max(yy), 10, 0]
			bds = [[0, -40, np.min(TS_r)], [np.inf, 40, np.max(TS_r)]]
			fit = curve_fit(gauss, TS_r, yy, p0, maxfev=100000, bounds=bds)
			profile_centres.append(fit[0][-1])
			profile_centres_score.append(fit[1][-1, -1])
		# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)
		profile_centres = np.array(profile_centres)
		profile_centres_score = np.array(profile_centres_score)
		centre = np.nanmean(profile_centres[profile_centres_score < 1])
		TS_r_new = np.abs(TS_r - centre) / 1000
		# temp_r, temp_t = np.meshgrid(TS_r_new, merge_time)
		# plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse,vmin=0);plt.colorbar().set_label('Te [eV]');plt.pause(0.01)
		# plt.figure();plt.pcolor(temp_t,temp_r,merge_ne_prof_multipulse,vmin=0);plt.colorbar().set_label('ne [10^20 #/m^3]');plt.pause(0.01)

		temp1 = np.zeros_like(inverted_profiles[:, 0])
		temp2 = np.zeros_like(inverted_profiles[:, 0])
		temp3 = np.zeros_like(inverted_profiles[:, 0])
		temp4 = np.zeros_like(inverted_profiles[:, 0])
		interp_range_t = max(dt / 2, TS_dt) * 1
		interp_range_r = max(dx / 2, TS_dr) * 1
		for i_t, value_t in enumerate(new_timesteps):
			if np.sum(np.abs(merge_time - value_t) < interp_range_t) == 0:
				continue
			for i_r, value_r in enumerate(np.abs(r)):
				if np.sum(np.abs(TS_r_new - value_r) < interp_range_r) == 0:
					continue
				elif np.sum(np.logical_and(np.abs(merge_time - value_t) < interp_range_t,
										   np.sum(merge_Te_prof_multipulse, axis=1) > 0)) == 0:
					continue
				selected_values_t = np.logical_and(np.abs(merge_time - value_t) < interp_range_t,
												   np.sum(merge_Te_prof_multipulse, axis=1) > 0)
				selected_values_r = np.abs(TS_r_new - value_r) < interp_range_r
				selecte_values = (np.array([selected_values_t])).T * selected_values_r
				selecte_values[merge_Te_prof_multipulse == 0] = False
				if np.sum(selecte_values) == 0:
					continue
				# temp1[i_t,i_r] = np.mean(merge_Te_prof_multipulse[selected_values_t][:,selected_values_r])
				# temp2[i_t, i_r] = np.max(merge_dTe_multipulse[selected_values_t][:,selected_values_r]) / (np.sum(np.isfinite(merge_dTe_multipulse[selected_values_t][:,selected_values_r])) ** 0.5)
				temp1[i_t, i_r] = np.sum(merge_Te_prof_multipulse[selecte_values] / merge_dTe_multipulse[selecte_values]) / np.sum(1 / merge_dTe_multipulse[selecte_values])
				temp2[i_t, i_r] = (np.sum(selecte_values) / (np.sum(1 / merge_dTe_multipulse[selecte_values]) ** 2)) ** 0.5
				# temp3[i_t,i_r] = np.mean(merge_ne_prof_multipulse[selected_values_t][:,selected_values_r])
				# temp4[i_t, i_r] = np.max(merge_dne_multipulse[selected_values_t][:,selected_values_r]) / (np.sum(np.isfinite(merge_dne_multipulse[selected_values_t][:,selected_values_r])) ** 0.5)
				temp3[i_t, i_r] = np.sum(merge_ne_prof_multipulse[selecte_values] / merge_dne_multipulse[selecte_values]) / np.sum(1 / merge_dne_multipulse[selecte_values])
				temp4[i_t, i_r] = (np.sum(selecte_values) / (np.sum(1 / merge_dne_multipulse[selecte_values]) ** 2)) ** 0.5

		merge_Te_prof_multipulse_interp = np.array(temp1)
		merge_dTe_prof_multipulse_interp = np.array(temp2)
		merge_ne_prof_multipulse_interp = np.array(temp3)
		merge_dne_prof_multipulse_interp = np.array(temp4)
		temp_r, temp_t = np.meshgrid(r, new_timesteps)

		# I crop to the usefull stuff
		start_time = np.abs(new_timesteps - 0).argmin()
		end_time = np.abs(new_timesteps - 1.5).argmin() + 1
		time_crop = new_timesteps[start_time:end_time]
		start_r = np.abs(r - 0).argmin()
		end_r = np.abs(r - 5).argmin() + 1
		r_crop = r[start_r:end_r]
		temp_r, temp_t = np.meshgrid(r_crop, time_crop)
		merge_Te_prof_multipulse_interp_crop = merge_Te_prof_multipulse_interp[start_time:end_time,start_r:end_r]
		merge_dTe_prof_multipulse_interp_crop = merge_dTe_prof_multipulse_interp[start_time:end_time, start_r:end_r]
		merge_ne_prof_multipulse_interp_crop = merge_ne_prof_multipulse_interp[start_time:end_time,start_r:end_r]
		merge_dne_prof_multipulse_interp_crop = merge_dne_prof_multipulse_interp[start_time:end_time, start_r:end_r]
		inverted_profiles_crop = inverted_profiles[start_time:end_time, :, start_r:end_r]
		inverted_profiles_crop[np.isnan(inverted_profiles_crop)] = 0
		all_fits_crop = all_fits[start_time:end_time]
		# inverted_profiles_crop[inverted_profiles_crop<0] = 0

		# x_local = xx - spatial_factor * 17.4 / 1000
		dr_crop = np.median(np.diff(r_crop))

		merge_dTe_prof_multipulse_interp_crop_limited = cp.deepcopy(merge_dTe_prof_multipulse_interp_crop)
		merge_dTe_prof_multipulse_interp_crop_limited[merge_Te_prof_multipulse_interp_crop < 0.1] = 0
		merge_Te_prof_multipulse_interp_crop_limited = cp.deepcopy(merge_Te_prof_multipulse_interp_crop)
		merge_Te_prof_multipulse_interp_crop_limited[merge_Te_prof_multipulse_interp_crop < 0.1] = 0
		merge_dne_prof_multipulse_interp_crop_limited = cp.deepcopy(merge_dne_prof_multipulse_interp_crop)
		merge_dne_prof_multipulse_interp_crop_limited[merge_ne_prof_multipulse_interp_crop < 5e-07] = 0
		merge_ne_prof_multipulse_interp_crop_limited = cp.deepcopy(merge_ne_prof_multipulse_interp_crop)
		merge_ne_prof_multipulse_interp_crop_limited[merge_ne_prof_multipulse_interp_crop < 5e-07] = 0

		excitation = []
		for isel in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
			temp = read_adf15(pecfile, isel, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
			temp[np.isnan(temp)] = 0
			temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
			excitation.append(temp)
		excitation = np.array(excitation)  # in # photons cm^-3 s^-1
		excitation = (excitation.T * (10 ** -6) * (energy_difference / J_to_eV)).T  # in W m^-3 / (# / m^3)**2

		recombination = []
		for isel in [20, 21, 22, 23, 24, 25, 26, 27, 28]:
			temp = read_adf15(pecfile, isel, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
			temp[np.isnan(temp)] = 0
			temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
			recombination.append(temp)
		recombination = np.array(recombination)  # in # photons cm^-3 s^-1 / (# cm^-3)**2
		recombination = (recombination.T * (10 ** -6) * (energy_difference / J_to_eV)).T  # in W m^-3 / (# / m^3)**2

		arbitrary_H_temp = 5000	# K, It is the same used for the fittings
		thermal_velocity_H = ( (arbitrary_H_temp*boltzmann_constant_J)/ au_to_kg)**0.5
		temp = read_adf11(scdfile, 'scd', 1, 1, 1, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())
		temp[np.isnan(temp)] = 0
		effective_ionisation_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (10 ** -6)  # in ionisations m^-3 s-1 / (# / m^3)**2
		ionization_length = thermal_velocity_H/(effective_ionisation_rates * merge_ne_prof_multipulse_interp_crop_limited * 1e20 )
		ionization_length = np.where(np.isnan(ionization_length), 0, ionization_length)
		ionization_length = np.where(np.isinf(ionization_length), np.nan, ionization_length)
		ionization_length = np.where(np.isnan(ionization_length), np.nanmax(ionization_length[np.isfinite(ionization_length)]), ionization_length)
		ionization_length_multipulse.append(ionization_length)

		n_list_all = np.sort(np.concatenate((n_list, n_list_1)))

		# nH_ne_all = np.load(path_where_to_save_everything + mod4 +'/results_2.npz')['nH_ne_all']
		# nHp_ne_all = np.load(path_where_to_save_everything + mod4 +'/results_2.npz')['nHp_ne_all']
		# nHm_ne_all = np.load(path_where_to_save_everything + mod4 +'/results_2.npz')['nHm_ne_all']
		# nH2_ne_all = np.load(path_where_to_save_everything + mod4 +'/results_2.npz')['nH2_ne_all']
		# nH2p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results_2.npz')['nH2p_ne_all']
		# nH3p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results_2.npz')['nH3p_ne_all']
		# residuals_all = np.load(path_where_to_save_everything + mod4 +'/results_2.npz')['residuals_all']

		nH_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['nH_ne_all']
		nHp_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['nHp_ne_all']
		nHm_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['nHm_ne_all']
		nH2_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['nH2_ne_all']
		nH2p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['nH2p_ne_all']
		nH3p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['nH3p_ne_all']
		residuals_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['residuals_all']

		temp = read_adf11(scdfile, 'scd', 1, 1, 1, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())
		temp[np.isnan(temp)] = 0
		effective_ionisation_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (10 ** -6)  # in ionisations m^-3 s-1 / (# / m^3)**2
		effective_ionisation_rates = (effective_ionisation_rates * (merge_ne_prof_multipulse_interp_crop * (10 ** 20)) * (merge_ne_prof_multipulse_interp_crop_limited * (10 ** 20) * nH_ne_all) ).astype('float')

		temp = read_adf11(acdfile, 'acd', 1, 1, 1, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())
		temp[np.isnan(temp)] = 0
		effective_recombination_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (10 ** -6)  # in recombinations m^-3 s-1 / (# / m^3)**2
		effective_recombination_rates = (effective_recombination_rates *nHp_ne_all* (merge_ne_prof_multipulse_interp_crop * (10 ** 20)) ** 2).astype('float')


		nH_ne_all_multipulse.append(nH_ne_all)
		nHp_ne_all_multipulse.append(nHp_ne_all)
		nHm_ne_all_multipulse.append(nHm_ne_all)
		nH2_ne_all_multipulse.append(nH2_ne_all)
		nH2p_ne_all_multipulse.append(nH2p_ne_all)
		nH3p_ne_all_multipulse.append(nH3p_ne_all)
		residuals_all_multipulse.append(residuals_all)

		effective_ionisation_rates_multipulse.append(effective_ionisation_rates)
		effective_recombination_rates_multipulse.append(effective_recombination_rates)

		merge_ne_prof_multipulse_interp_crop_limited_multipulse.append(merge_ne_prof_multipulse_interp_crop_limited)
		merge_dne_prof_multipulse_interp_crop_limited_multipulse.append(merge_dne_prof_multipulse_interp_crop_limited)
		merge_Te_prof_multipulse_interp_crop_limited_multipulse.append(merge_Te_prof_multipulse_interp_crop_limited)
		merge_dTe_prof_multipulse_interp_crop_limited_multipulse.append(merge_dTe_prof_multipulse_interp_crop_limited)

		inverted_profiles_crop_multipulse.append(inverted_profiles_crop)

		target_chamber_pressure_multipulse.append(target_chamber_pressure)
		target_OES_distance_multipulse.append(target_OES_distance)

		rate_creation_Hm = np.load(path_where_to_save_everything + mod4 +'/molecular_results.npz')['rate_creation_Hm']
		rate_destruction_Hm = np.load(path_where_to_save_everything + mod4 +'/molecular_results.npz')['rate_destruction_Hm']
		rate_creation_Hm_multipulse.append(rate_creation_Hm)
		rate_destruction_Hm_multipulse.append(rate_destruction_Hm)

	nH_ne_all_multipulse=np.array(nH_ne_all_multipulse)
	nHp_ne_all_multipulse=np.array(nHp_ne_all_multipulse)
	nHm_ne_all_multipulse=np.array(nHm_ne_all_multipulse)
	nH2_ne_all_multipulse=np.array(nH2_ne_all_multipulse)
	nH2p_ne_all_multipulse=np.array(nH2p_ne_all_multipulse)
	nH3p_ne_all_multipulse=np.array(nH3p_ne_all_multipulse)
	residuals_all_multipulse=np.array(residuals_all_multipulse)
	effective_ionisation_rates_multipulse=np.array(effective_ionisation_rates_multipulse)
	merge_ne_prof_multipulse_interp_crop_limited_multipulse=np.array(merge_ne_prof_multipulse_interp_crop_limited_multipulse)
	merge_dne_prof_multipulse_interp_crop_limited_multipulse=np.array(merge_dne_prof_multipulse_interp_crop_limited_multipulse)
	merge_Te_prof_multipulse_interp_crop_limited_multipulse=np.array(merge_Te_prof_multipulse_interp_crop_limited_multipulse)
	merge_dTe_prof_multipulse_interp_crop_limited_multipulse=np.array(merge_dTe_prof_multipulse_interp_crop_limited_multipulse)
	inverted_profiles_crop_multipulse=np.array(inverted_profiles_crop_multipulse)
	target_chamber_pressure_multipulse=np.array(target_chamber_pressure_multipulse)
	target_chamber_pressure_multipulse=np.array(target_chamber_pressure_multipulse)

	rate_creation_Hm_multipulse=np.array(rate_creation_Hm_multipulse)
	rate_destruction_Hm_multipulse=np.array(rate_destruction_Hm_multipulse)

	fig, ax = plt.subplots(11, len(nH_ne_all_multipulse)+2,figsize=(30, 25), squeeze=True)
	fig.suptitle('min_nHp/ne=%.3g, max_nH/ne=%.3g ,' %(min_nHp_ne,max_n_neutrals_ne)+mod3+' lines '+str(n_list_1.tolist()+n_list.tolist())+' n_weights '+str(n_weights))
	for index in range(len(nH_ne_all_multipulse)):
		j=0

		im = ax[j,index].pcolor(temp_t.T,1000*temp_r.T,merge_Te_prof_multipulse_interp_crop_limited_multipulse[index].T,vmax=np.max(merge_Te_prof_multipulse_interp_crop_limited_multipulse),vmin=0,cmap='rainbow')
		# plt.set_sketch_params(scale=2)
		ax[j,index].set_aspect(1/40)
		ax[j,index].set_title('target/OES %.3gmm\ntarg. chamb. pres. %.3gPa\n(merge %.3g)\nTS Te [eV]' %(target_OES_distance_multipulse[index],target_chamber_pressure_multipulse[index],merge_ID_target_multipulse[index]))
		ax[j,index].set_xlabel('time [ms]')
		if index == 0:
			ax[j,index].set_ylabel('radial location [mm]')
		else:
			ax[j,index].set_yticks([])
		if index == (len(nH_ne_all_multipulse)-1):
			fig.colorbar(im, ax=ax[j,np.max(merge_Te_prof_multipulse_interp_crop_limited_multipulse,axis=(-1,-2)).argmax()])
		# plt.axes().set_aspect(0.1)

		j+=1

		im = ax[j,index].pcolor(temp_t.T,1000*temp_r.T,merge_ne_prof_multipulse_interp_crop_limited_multipulse[index].T,vmax=np.max(merge_ne_prof_multipulse_interp_crop_limited_multipulse),vmin=0,cmap='rainbow')
		# plt.set_sketch_params(scale=2)
		ax[j,index].set_aspect(1/40)
		ax[j,index].set_xlabel('time [ms]')
		if index == 0:
			ax[j,index].set_ylabel('radial location [mm]')
			ax[j,index].set_title('TS ne [10^20 # m^-3]')
		else:
			ax[j,index].set_yticks([])
			ax[j,index].set_title('TS ne')
		if index == (len(nH_ne_all_multipulse)-1):
			fig.colorbar(im, ax=ax[j,np.max(merge_ne_prof_multipulse_interp_crop_limited_multipulse,axis=(-1,-2)).argmax()])

		j+=1

		im = ax[j,index].pcolor(temp_t.T, 1000*temp_r.T, (merge_ne_prof_multipulse_interp_crop_limited_multipulse[index] * nH_ne_all_multipulse[index]).T,vmax=np.max(np.array(merge_ne_prof_multipulse_interp_crop_limited_multipulse) * np.array(nH_ne_all_multipulse)),vmin=max(np.max(np.array(merge_ne_prof_multipulse_interp_crop_limited_multipulse) * np.array(nH_ne_all_multipulse))/1e12,1e-3), cmap='rainbow', norm=LogNorm());
		# plt.set_sketch_params(scale=2)
		ax[j,index].set_aspect(1/40)
		ax[j,index].set_xlabel('time [ms]')
		if index == 0:
			ax[j,index].set_ylabel('radial location [mm]')
			ax[j,index].set_title('nH [10^20 # m^-3]')
		else:
			ax[j,index].set_yticks([])
			ax[j,index].set_title('nH')
		if index == (len(nH_ne_all_multipulse)-1):
			fig.colorbar(im, ax=ax[j,np.max(np.array(merge_ne_prof_multipulse_interp_crop_limited_multipulse) * np.array(nH_ne_all_multipulse),axis=(-1,-2)).argmax()])

		j+=1
		if np.max((merge_ne_prof_multipulse_interp_crop_limited_multipulse[index] * nHm_ne_all_multipulse[index]).T)>0:
			im = ax[j,index].pcolor(temp_t.T, 1000*temp_r.T, (merge_ne_prof_multipulse_interp_crop_limited_multipulse[index] * nHm_ne_all_multipulse[index]).T,vmax=np.max(np.array(merge_ne_prof_multipulse_interp_crop_limited_multipulse) * np.array(nHm_ne_all_multipulse)),vmin=max(np.max(np.array(merge_ne_prof_multipulse_interp_crop_limited_multipulse) * np.array(nHm_ne_all_multipulse))/1e6,1e-5), cmap='rainbow', norm=LogNorm());
		else:
			im = ax[j,index].pcolor(temp_t.T, 1000*temp_r.T, (merge_ne_prof_multipulse_interp_crop_limited_multipulse[index] * nHm_ne_all_multipulse[index]).T,vmax=np.max(np.array(merge_ne_prof_multipulse_interp_crop_limited_multipulse) * np.array(nHm_ne_all_multipulse)),vmin=max(np.max(np.array(merge_ne_prof_multipulse_interp_crop_limited_multipulse) * np.array(nHm_ne_all_multipulse))/1e6,1e-5), cmap='rainbow');
		# plt.set_sketch_params(scale=2)
		ax[j,index].set_aspect(1/40)
		ax[j,index].set_xlabel('time [ms]')
		if index == 0:
			ax[j,index].set_ylabel('radial location [mm]')
			ax[j,index].set_title('nH- [10^20 # m^-3]')
		else:
			ax[j,index].set_yticks([])
			ax[j,index].set_title('nH-')
		if index == (len(nH_ne_all_multipulse)-1):
			fig.colorbar(im, ax=ax[j,np.max(np.array(merge_ne_prof_multipulse_interp_crop_limited_multipulse) * np.array(nH_ne_all_multipulse),axis=(-1,-2)).argmax()])

		j+=1
		if np.max((merge_ne_prof_multipulse_interp_crop_limited_multipulse[index] * nH2_ne_all_multipulse[index]).T)>0:
			im = ax[j,index].pcolor(temp_t.T, 1000*temp_r.T, (merge_ne_prof_multipulse_interp_crop_limited_multipulse[index] * nH2_ne_all_multipulse[index]).T,vmax=np.max(np.array(merge_ne_prof_multipulse_interp_crop_limited_multipulse) * np.array(nH2_ne_all_multipulse)),vmin=max(np.max(np.array(merge_ne_prof_multipulse_interp_crop_limited_multipulse) * np.array(nH2_ne_all_multipulse))/1e6,1e-3), cmap='rainbow', norm=LogNorm());
		else:
			im = ax[j,index].pcolor(temp_t.T, 1000*temp_r.T, (merge_ne_prof_multipulse_interp_crop_limited_multipulse[index] * nH2_ne_all_multipulse[index]).T,vmax=np.max(np.array(merge_ne_prof_multipulse_interp_crop_limited_multipulse) * np.array(nH2_ne_all_multipulse)),vmin=max(np.max(np.array(merge_ne_prof_multipulse_interp_crop_limited_multipulse) * np.array(nH2_ne_all_multipulse))/1e6,1e-3), cmap='rainbow');
		# plt.set_sketch_params(scale=2)
		ax[j,index].set_aspect(1/40)
		ax[j,index].set_xlabel('time [ms]')
		if index == 0:
			ax[j,index].set_ylabel('radial location [mm]')
			ax[j,index].set_title('nH2 [10^20 # m^-3]')
		else:
			ax[j,index].set_yticks([])
			ax[j,index].set_title('nH2')
		if index == (len(nH_ne_all_multipulse)-1):
			fig.colorbar(im, ax=ax[j,np.max(np.array(merge_ne_prof_multipulse_interp_crop_limited_multipulse) * np.array(nH_ne_all_multipulse),axis=(-1,-2)).argmax()])

		j+=1
		if np.max((merge_ne_prof_multipulse_interp_crop_limited_multipulse[index] * nH2p_ne_all_multipulse[index]).T)>0:
			im = ax[j,index].pcolor(temp_t.T, 1000*temp_r.T, (merge_ne_prof_multipulse_interp_crop_limited_multipulse[index] * nH2p_ne_all_multipulse[index]).T,vmax=np.max(np.array(merge_ne_prof_multipulse_interp_crop_limited_multipulse) * np.array(nH2p_ne_all_multipulse)),vmin=max(np.max(np.array(merge_ne_prof_multipulse_interp_crop_limited_multipulse) * np.array(nH2p_ne_all_multipulse))/1e6,1e-3), cmap='rainbow', norm=LogNorm());
		else:
			im = ax[j,index].pcolor(temp_t.T, 1000*temp_r.T, (merge_ne_prof_multipulse_interp_crop_limited_multipulse[index] * nH2p_ne_all_multipulse[index]).T,vmax=np.max(np.array(merge_ne_prof_multipulse_interp_crop_limited_multipulse) * np.array(nH2p_ne_all_multipulse)),vmin=max(np.max(np.array(merge_ne_prof_multipulse_interp_crop_limited_multipulse) * np.array(nH2p_ne_all_multipulse))/1e6,1e-3), cmap='rainbow');
		# plt.set_sketch_params(scale=2)
		ax[j,index].set_aspect(1/40)
		ax[j,index].set_xlabel('time [ms]')
		if index == 0:
			ax[j,index].set_ylabel('radial location [mm]')
			ax[j,index].set_title('nH2+ [10^20 # m^-3]')
		else:
			ax[j,index].set_yticks([])
			ax[j,index].set_title('nH2+')
		if index == (len(nH_ne_all_multipulse)-1):
			fig.colorbar(im, ax=ax[j,np.max(np.array(merge_ne_prof_multipulse_interp_crop_limited_multipulse) * np.array(nH_ne_all_multipulse),axis=(-1,-2)).argmax()])

		j+=1

		im = ax[j,index].pcolor(temp_t.T,1000*temp_r.T,effective_ionisation_rates_multipulse[index].T,vmax=np.max(effective_ionisation_rates),cmap='rainbow',vmin=max(np.max(effective_ionisation_rates)/1e3,1e19), norm=LogNorm());
		# plt.set_sketch_params(scale=2)
		ax[j,index].set_aspect(1/40)
		ax[j,index].set_xlabel('time [ms]')
		if index==4:
			ax[j,index].set_title('eff. ionisation rates [# m^-3 s-1]')
		if index == 0:
			ax[j,index].set_ylabel('radial location [mm]')
		else:
			ax[j,index].set_yticks([])
		if index == (len(nH_ne_all_multipulse)-1):
			fig.colorbar(im, ax=ax[j,np.max(effective_ionisation_rates_multipulse,axis=(-1,-2)).argmax()])

		j+=1

		im = ax[j,index].pcolor(temp_t.T,1000*temp_r.T,effective_recombination_rates_multipulse[index].T,vmax=np.max(effective_recombination_rates_multipulse),cmap='rainbow',vmin=max(np.max(effective_recombination_rates_multipulse)/1e3,1e19), norm=LogNorm());
		# plt.set_sketch_params(scale=2)
		ax[j,index].set_aspect(1/40)
		ax[j,index].set_xlabel('time [ms]')
		if index==4:
			ax[j,index].set_title('eff. recombination rates [# m^-3 s-1]')
		if index == 0:
			ax[j,index].set_ylabel('radial location [mm]')
		else:
			ax[j,index].set_yticks([])
		if index == (len(nH_ne_all_multipulse)-1):
			fig.colorbar(im, ax=ax[j,np.max(effective_recombination_rates_multipulse,axis=(-1,-2)).argmax()])

		j+=1

		im = ax[j,index].pcolor(temp_t.T,1000*temp_r.T,inverted_profiles_crop_multipulse[index][:,0].T,vmax=np.max(inverted_profiles_crop_multipulse),vmin=0,cmap='rainbow')
		# plt.set_sketch_params(scale=2)
		ax[j,index].set_aspect(1/40)
		ax[j,index].set_xlabel('time [ms]')
		if index==4:
			ax[j,index].set_title('OES n=4>2 line [W m^-3]')
		if index == 0:
			ax[j,index].set_ylabel('radial location [mm]')
		else:
			ax[j,index].set_yticks([])
		if index == (len(nH_ne_all_multipulse)-1):
			fig.colorbar(im, ax=ax[j,np.max(inverted_profiles_crop_multipulse,axis=(-1,-2,-3)).argmax()],format='%.2g')

		j+=1

		im = ax[j,index].pcolor(temp_t.T,1000*temp_r.T,ionization_length_multipulse[index].T,vmax=1,cmap='rainbow', norm=LogNorm());
		# plt.set_sketch_params(scale=2)
		ax[j,index].set_aspect(1/40)
		# ax[j,index].set_xlabel('time [ms]')
		if index==4:
			ax[j,index].set_title('H vs e- ionisation length [m]')
		if index == 0:
			ax[j,index].set_ylabel('radial location [mm]')
		else:
			ax[j,index].set_yticks([])
		if index == (len(nH_ne_all_multipulse)-1):
			fig.colorbar(im, ax=ax[j,np.max(ionization_length_multipulse,axis=(-1,-2)).argmax()])

		j+=1

		threshold_radious = 0.010
		area = 2*np.pi*(r_crop[r_crop<=threshold_radious] + np.diff([0,*r_crop[r_crop<=threshold_radious]])/2) * np.diff([0,*r_crop[r_crop<=threshold_radious]])

		im = ax[j,index].plot(time_crop, np.sum(merge_Te_prof_multipulse_interp_crop_limited_multipulse[index][:,r_crop<threshold_radious]*area,axis=1)/np.sum(area)/np.max(np.sum(merge_Te_prof_multipulse_interp_crop_limited_multipulse[index][:,r_crop<threshold_radious]*area,axis=1)/np.sum(area)),'g--', label='Te TS\n(max='+'%.3g eV)' % np.max(np.sum(merge_Te_prof_multipulse_interp_crop_limited_multipulse[index][:,r_crop<threshold_radious]*area,axis=1)/np.sum(area)));
		im = ax[j,index].plot(time_crop, np.sum(merge_ne_prof_multipulse_interp_crop_limited_multipulse[index][:,r_crop<threshold_radious]*area,axis=1)/np.sum(area)/np.max(np.sum(merge_ne_prof_multipulse_interp_crop_limited_multipulse[index][:,r_crop<threshold_radious]*area,axis=1)/np.sum(area)),'r--', label='ne TS\n(max='+'%.3g 10^20 # m^-3)' % np.max(np.sum(merge_Te_prof_multipulse_interp_crop_limited_multipulse[index][:,r_crop<threshold_radious]*area,axis=1)/np.sum(area)));
		ax2 = ax[j,index].twinx()
		im = ax2.plot(time_crop, np.sum(effective_ionisation_rates_multipulse[index][:,r_crop<threshold_radious]*area,axis=1)/np.sum(area), label='ionisation rate');
		im = ax2.plot(time_crop, np.sum(effective_recombination_rates_multipulse[index][:,r_crop<threshold_radious]*area,axis=1)/np.sum(area), label='recombination rate');
		max_value_for_plot = max(np.max(np.sum(np.array(effective_ionisation_rates_multipulse)[:,:,r_crop<threshold_radious]*area,axis=-1)/np.sum(area)),np.max(np.sum(np.array(effective_recombination_rates_multipulse)[:,:,r_crop<threshold_radious]*area,axis=-1)/np.sum(area)))
		ax2.set_ylim(max_value_for_plot/1e4,max_value_for_plot)
		ax2.set_yscale('log')
		ax[j,index].grid()
		ax2.grid()
		# plt.legend(loc='best')
		# ax[j,index].set_aspect(1/40)
		ax[j,index].set_xlabel('time [ms]')
		# plt.set_sketch_params(scale=2)
		if index==4:
			ax[j,index].set_title('Time evolution of the radial average, 0>r>'+  '%.3g' % (1000*threshold_radious) +' mm weighted on the area')
		if index == 0:
			ax[j,index].set_ylabel('fraction of\nmax Te and ne')
		else:
			ax[j,index].set_yticks([])
		if index == (len(nH_ne_all_multipulse)-1):
			ax2.set_ylabel('ionisat/recomb rate[# m^-3 s-1]')
		else:
			ax2.set_yticks([])


	for index_2,merge_ID_target_at_the_target in enumerate([90,91]):

		all_j=find_index_of_file(merge_ID_target_at_the_target,df_settings,df_log,only_OES=True)
		target_chamber_pressure_at_the_target = []
		target_OES_distance_at_the_target = []
		for j in all_j:
			target_chamber_pressure_at_the_target.append(df_log.loc[j,['p_n [Pa]']])
			target_OES_distance_at_the_target.append(df_log.loc[j,['T_axial']])
		target_chamber_pressure_at_the_target = np.nanmean(target_chamber_pressure_at_the_target)	# Pa
		target_OES_distance_at_the_target = np.nanmean(target_OES_distance_at_the_target)	# Pa
		# Ideal gas law
		max_nH2_from_pressure_at_the_target = target_chamber_pressure_at_the_target/(boltzmann_constant_J*300)	# [#/m^3] I suppose ambient temp is ~ 300K

		path_where_to_save_everything_target = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target_at_the_target)
		merge_Te_prof_multipulse_target = np.load(path_where_to_save_everything_target + '/TS_data_merge_' + str(merge_ID_target_at_the_target) + '.npz')['merge_Te_prof_multipulse']
		merge_dTe_multipulse_target = np.load(path_where_to_save_everything_target + '/TS_data_merge_' + str(merge_ID_target_at_the_target) + '.npz')['merge_dTe_multipulse']
		merge_ne_prof_multipulse_target = np.load(path_where_to_save_everything_target + '/TS_data_merge_' + str(merge_ID_target_at_the_target) + '.npz')['merge_ne_prof_multipulse']
		merge_dne_multipulse_target = np.load(path_where_to_save_everything_target + '/TS_data_merge_' + str(merge_ID_target_at_the_target) + '.npz')['merge_dne_multipulse']
		merge_time_original_target = np.load(path_where_to_save_everything_target + '/TS_data_merge_' + str(merge_ID_target_at_the_target) + '.npz')['merge_time']
		merge_time_target = time_shift_factor + merge_time_original_target

		profile_centres_target = []
		profile_centres_score_target = []
		for index in range(np.shape(merge_Te_prof_multipulse_target)[0]):
			yy = merge_Te_prof_multipulse_target[index]
			p0 = [np.max(yy), 10, 0]
			bds = [[0, -40, np.min(TS_r)], [np.inf, 40, np.max(TS_r)]]
			fit = curve_fit(gauss, TS_r, yy, p0, maxfev=100000, bounds=bds)
			profile_centres_target.append(fit[0][-1])
			profile_centres_score_target.append(fit[1][-1, -1])
		# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)
		profile_centres_target = np.array(profile_centres_target)
		profile_centres_score_target = np.array(profile_centres_score_target)
		centre = np.nanmean(profile_centres_target[profile_centres_score_target < 1])
		TS_r_new = np.abs(TS_r - centre) / 1000
		# temp_r, temp_t = np.meshgrid(TS_r_new, merge_time)
		# plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse,vmin=0);plt.colorbar().set_label('Te [eV]');plt.pause(0.01)
		# plt.figure();plt.pcolor(temp_t,temp_r,merge_ne_prof_multipulse,vmin=0);plt.colorbar().set_label('ne [10^20 #/m^3]');plt.pause(0.01)

		temp1 = np.zeros_like(inverted_profiles[:, 0])
		temp2 = np.zeros_like(inverted_profiles[:, 0])
		temp3 = np.zeros_like(inverted_profiles[:, 0])
		temp4 = np.zeros_like(inverted_profiles[:, 0])
		interp_range_t = max(dt / 2, TS_dt) * 1
		interp_range_r = max(dx / 2, TS_dr) * 1
		for i_t, value_t in enumerate(new_timesteps):
			if np.sum(np.abs(merge_time_target - value_t) < interp_range_t) == 0:
				continue
			for i_r, value_r in enumerate(np.abs(r)):
				if np.sum(np.abs(TS_r_new - value_r) < interp_range_r) == 0:
					continue
				elif np.sum(np.logical_and(np.abs(merge_time_target - value_t) < interp_range_t,np.sum(merge_Te_prof_multipulse_target, axis=1) > 0)) == 0:
					continue
				selected_values_t = np.logical_and(np.abs(merge_time_target - value_t) < interp_range_t,np.sum(merge_Te_prof_multipulse_target, axis=1) > 0)
				selected_values_r = np.abs(TS_r_new - value_r) < interp_range_r
				selecte_values = (np.array([selected_values_t])).T * selected_values_r
				selecte_values[merge_Te_prof_multipulse_target == 0] = False
				if np.sum(selecte_values) == 0:
					continue
				# temp1[i_t,i_r] = np.mean(merge_Te_prof_multipulse[selected_values_t][:,selected_values_r])
				# temp2[i_t, i_r] = np.max(merge_dTe_multipulse[selected_values_t][:,selected_values_r]) / (np.sum(np.isfinite(merge_dTe_multipulse[selected_values_t][:,selected_values_r])) ** 0.5)
				temp1[i_t, i_r] = np.sum(
					merge_Te_prof_multipulse_target[selecte_values] / merge_dTe_multipulse_target[
						selecte_values]) / np.sum(1 / merge_dTe_multipulse_target[selecte_values])
				temp2[i_t, i_r] = (np.sum(selecte_values) / (
							np.sum(1 / merge_dTe_multipulse_target[selecte_values]) ** 2)) ** 0.5
				# temp3[i_t,i_r] = np.mean(merge_ne_prof_multipulse[selected_values_t][:,selected_values_r])
				# temp4[i_t, i_r] = np.max(merge_dne_multipulse[selected_values_t][:,selected_values_r]) / (np.sum(np.isfinite(merge_dne_multipulse[selected_values_t][:,selected_values_r])) ** 0.5)
				temp3[i_t, i_r] = np.sum(
					merge_ne_prof_multipulse_target[selecte_values] / merge_dne_multipulse_target[
						selecte_values]) / np.sum(1 / merge_dne_multipulse_target[selecte_values])
				temp4[i_t, i_r] = (np.sum(selecte_values) / (
							np.sum(1 / merge_dne_multipulse_target[selecte_values]) ** 2)) ** 0.5

		merge_Te_prof_multipulse_interp_target = np.array(temp1)
		merge_dTe_prof_multipulse_interp_target = np.array(temp2)
		merge_ne_prof_multipulse_interp_target = np.array(temp3)
		merge_dne_prof_multipulse_interp_target = np.array(temp4)
		temp_r, temp_t = np.meshgrid(r, new_timesteps)

		# I crop to the usefull stuff
		start_time = np.abs(new_timesteps - 0).argmin()
		end_time = np.abs(new_timesteps - 1.5).argmin() + 1
		time_crop = new_timesteps[start_time:end_time]
		start_r = np.abs(r - 0).argmin()
		end_r = np.abs(r - 5).argmin() + 1
		r_crop = r[start_r:end_r]
		temp_r, temp_t = np.meshgrid(r_crop, time_crop)
		merge_Te_prof_multipulse_interp_crop_target = merge_Te_prof_multipulse_interp_target[start_time:end_time,start_r:end_r]
		merge_dTe_prof_multipulse_interp_crop_target = merge_dTe_prof_multipulse_interp_target[start_time:end_time, start_r:end_r]
		merge_ne_prof_multipulse_interp_crop_target = merge_ne_prof_multipulse_interp_target[start_time:end_time,start_r:end_r]
		merge_dne_prof_multipulse_interp_crop_target = merge_dne_prof_multipulse_interp_target[start_time:end_time, start_r:end_r]



		# x_local = xx - spatial_factor * 17.4 / 1000
		# dr_crop = np.median(np.diff(r_crop))

		merge_dTe_prof_multipulse_interp_crop_limited_target = cp.deepcopy(merge_dTe_prof_multipulse_interp_crop_target)
		merge_dTe_prof_multipulse_interp_crop_limited_target[merge_Te_prof_multipulse_interp_crop_target < 0.1] = 0
		merge_Te_prof_multipulse_interp_crop_limited_target = cp.deepcopy(merge_Te_prof_multipulse_interp_crop_target)
		merge_Te_prof_multipulse_interp_crop_limited_target[merge_Te_prof_multipulse_interp_crop_target < 0.1] = 0
		merge_dne_prof_multipulse_interp_crop_limited_target = cp.deepcopy(merge_dne_prof_multipulse_interp_crop_target)
		merge_dne_prof_multipulse_interp_crop_limited_target[merge_ne_prof_multipulse_interp_crop_target < 5e-07] = 0
		merge_ne_prof_multipulse_interp_crop_limited_target = cp.deepcopy(merge_ne_prof_multipulse_interp_crop_target)
		merge_ne_prof_multipulse_interp_crop_limited_target[merge_ne_prof_multipulse_interp_crop_target < 5e-07] = 0

		j=0
		index = len(nH_ne_all_multipulse)+index_2
		im = ax[j,index].pcolor(temp_t.T,1000*temp_r.T,merge_Te_prof_multipulse_interp_crop_limited_target.T,vmax=np.max(merge_Te_prof_multipulse_interp_crop_limited_multipulse),vmin=0,cmap='rainbow')
		# plt.set_sketch_params(scale=2)
		ax[j,index].set_aspect(1/40)
		ax[j,index].set_title('target/OES %.3gmm\ntarg. chamb. pres. %.3gPa\n(merge %.3g)\nTS Te [eV]' %(target_OES_distance_at_the_target,target_chamber_pressure_at_the_target,merge_ID_target_at_the_target))
		ax[j,index].set_xlabel('time [ms]')
		ax[j,index].set_yticks([])
		# plt.axes().set_aspect(0.1)

		j=1

		im = ax[j,index].pcolor(temp_t.T,1000*temp_r.T,merge_ne_prof_multipulse_interp_crop_limited_target.T,vmax=np.max(merge_ne_prof_multipulse_interp_crop_limited_multipulse),vmin=0,cmap='rainbow')
		# plt.set_sketch_params(scale=2)
		ax[j,index].set_aspect(1/40)
		ax[j,index].set_xlabel('time [ms]')
		ax[j,index].set_yticks([])
		ax[j,index].set_title('TS ne')

		for j in [2,3,4,5,6,7,8,9,10]:
			ax[j,index].set_yticks([])
			ax[j,index].set_xticks([])


	figure_index += 1
	plt.savefig(path_where_to_save_plots+'/global_check_physics'+str(figure_index) + '.eps', bbox_inches='tight')
	plt.close()



	threshold_radious = 0.010
	fig, ax = plt.subplots( len(nH_ne_all_multipulse),1,figsize=(20, 25), squeeze=True)
	fig.suptitle('min_nHp/ne=%.3g, max_nH/ne=%.3g ,' %(min_nHp_ne,max_n_neutrals_ne)+mod3+' lines '+str(n_list_1.tolist()+n_list.tolist())+' n_weights '+str(n_weights)+'\nTime evolution of the radial average, 0>r>'+  '%.3g' % (1000*threshold_radious) +' mm weighted on the area')
	for index in range(len(nH_ne_all_multipulse)):

		area = 2*np.pi*(r_crop[r_crop<=threshold_radious] + np.diff([0,*r_crop[r_crop<=threshold_radious]])/2) * np.diff([0,*r_crop[r_crop<=threshold_radious]])

		im = ax[index].plot(time_crop, np.sum(merge_Te_prof_multipulse_interp_crop_limited_multipulse[index][:,r_crop<threshold_radious]*area,axis=1)/np.sum(area)/np.max(np.sum(merge_Te_prof_multipulse_interp_crop_limited_multipulse[index][:,r_crop<threshold_radious]*area,axis=1)/np.sum(area)),'g--', label='Te TS\n(max='+'%.3g eV)' % np.max(np.sum(merge_Te_prof_multipulse_interp_crop_limited_multipulse[index][:,r_crop<threshold_radious]*area,axis=1)/np.sum(area)));
		im = ax[index].plot(time_crop, np.sum(merge_ne_prof_multipulse_interp_crop_limited_multipulse[index][:,r_crop<threshold_radious]*area,axis=1)/np.sum(area)/np.max(np.sum(merge_ne_prof_multipulse_interp_crop_limited_multipulse[index][:,r_crop<threshold_radious]*area,axis=1)/np.sum(area)),'r--', label='ne TS\n(max='+'%.3g 10^20 # m^-3)' % np.max(np.sum(merge_ne_prof_multipulse_interp_crop_limited_multipulse[index][:,r_crop<threshold_radious]*area,axis=1)/np.sum(area)));
		ax[index].legend(loc=1)
		ax2 = ax[index].twinx()
		im = ax2.plot(time_crop, np.sum(effective_ionisation_rates_multipulse[index][:,r_crop<threshold_radious]*area,axis=1)/np.sum(area), label='ionisation rate');
		im = ax2.plot(time_crop, np.sum(effective_recombination_rates_multipulse[index][:,r_crop<threshold_radious]*area,axis=1)/np.sum(area), label='recombination rate');
		max_value_for_plot = max(np.max(np.sum(np.array(effective_ionisation_rates_multipulse)[:,:,r_crop<threshold_radious]*area,axis=-1)/np.sum(area)),np.max(np.sum(np.array(effective_recombination_rates_multipulse)[:,:,r_crop<threshold_radious]*area,axis=-1)/np.sum(area)))
		ax2.set_ylim(max_value_for_plot/1e7,max_value_for_plot)
		ax2.set_yscale('log')
		ax2.legend(loc=4)
		ax2.grid()
		ax[index].grid()

		# plt.legend(loc='best')
		# ax[j,index].set_aspect(1/40)
		# plt.set_sketch_params(scale=2)
		ax[index].set_title('target/OES %.3gmm targ. chamb. pres. %.3gPa (merge %.3g)' %(target_OES_distance_multipulse[index],target_chamber_pressure_multipulse[index],merge_ID_target_multipulse[index]))
		ax2.set_ylabel('ionisat/recomb rate\n[# m^-3 s-1]')
		ax[index].set_ylabel('fraction of\nmax Te and ne')
		if index == (len(nH_ne_all_multipulse)-1):
			ax[index].set_xlabel('time [ms]')
		else:
			ax[index].set_xticks([])
	figure_index += 1
	plt.savefig(path_where_to_save_plots+'/global_check_physics'+str(figure_index) + '.eps', bbox_inches='tight')
	plt.close()



	plt.figure()
	plt.title('min_nHp/ne=%.3g, max_nH/ne=%.3g ,' %(min_nHp_ne,max_n_neutrals_ne)+mod3+' lines '+str(n_list_1.tolist()+n_list.tolist())+' n_weights '+str(n_weights)+'\nioniastion rate, scan for target chamber pressure')
	area = 2*np.pi*(r_crop + np.diff([0,*r_crop])/2) * np.diff([0,*r_crop])
	levs = np.linspace(np.log10(np.max(np.sum(np.array(effective_ionisation_rates_multipulse[:4])*area,axis=-1)/np.sum(area))*1e-3),np.log10(np.max(np.sum(np.array(effective_ionisation_rates_multipulse[:4])*area,axis=-1)/np.sum(area))),num=15)
	levs = np.power(10, levs)
	# plt.contourf(temp_t[:,:4].T,(np.ones_like(temp_t[:,:4])*target_chamber_pressure_multipulse[:4]).T,np.sum(np.array(effective_ionisation_rates_multipulse[:4])*area,axis=-1)/np.sum(area),levs, cmap='rainbow',norm=LogNorm())#25,vmin=np.max(np.sum(np.array(effective_ionisation_rates_multipulse[:4])*area,axis=-1)/np.sum(area))*1e-2)#, norm=LogNorm());
	plt.contourf(temp_t[:,:4].T,(np.ones_like(temp_t[:,:4])*target_chamber_pressure_multipulse[:4]).T,np.sum(np.array(effective_ionisation_rates_multipulse[:4])*area,axis=-1)/np.sum(area),15, cmap='rainbow')#25,vmin=np.max(np.sum(np.array(effective_ionisation_rates_multipulse[:4])*area,axis=-1)/np.sum(area))*1e-2)#, norm=LogNorm());
	peak_time =(np.sum(np.array(effective_ionisation_rates_multipulse[:4])*area,axis=-1)/np.sum(area)).argmax(axis=1)
	plt.plot(temp_t[:,0][peak_time],target_chamber_pressure_multipulse[:4],'k--',label='peak location')
	plt.plot(temp_t[:,0][peak_time],target_chamber_pressure_multipulse[:4],'k+')
	plt.legend(loc=1)
	plt.xlabel('time [ms]')
	plt.ylabel('target chamber pressure [Pa]')
	cbar=plt.colorbar()
	cbar.set_label('eff. ionisation rates [# m^-3 s-1]')
	# cbar.set_ticks(levs[::3])
	# cbar.set_ticklabels(['%.3g' %(_) for _ in levs[::3]])
	figure_index += 1
	plt.savefig(path_where_to_save_plots+'/global_check_physics'+str(figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.title('min_nHp/ne=%.3g, max_nH/ne=%.3g ,' %(min_nHp_ne,max_n_neutrals_ne)+mod3+' lines '+str(n_list_1.tolist()+n_list.tolist())+' n_weights '+str(n_weights)+'\nrecombination rate, scan for target chamber pressure')
	area = 2*np.pi*(r_crop + np.diff([0,*r_crop])/2) * np.diff([0,*r_crop])
	levs = np.linspace(np.log10(np.max(np.sum(np.array(effective_recombination_rates_multipulse[:4])*area,axis=-1)/np.sum(area))*1e-3),np.log10(np.max(np.sum(np.array(effective_recombination_rates_multipulse[:4])*area,axis=-1)/np.sum(area))),num=15)
	levs = np.power(10, levs)
	# plt.contourf(temp_t[:,:4].T,(np.ones_like(temp_t[:,:4])*target_chamber_pressure_multipulse[:4]).T,np.sum(np.array(effective_recombination_rates_multipulse[:4])*area,axis=-1)/np.sum(area),levs, cmap='rainbow',norm=LogNorm())#25,vmin=np.max(np.sum(np.array(effective_ionisation_rates_multipulse[:4])*area,axis=-1)/np.sum(area))*1e-2)#, norm=LogNorm());
	plt.contourf(temp_t[:,:4].T,(np.ones_like(temp_t[:,:4])*target_chamber_pressure_multipulse[:4]).T,np.sum(np.array(effective_recombination_rates_multipulse[:4])*area,axis=-1)/np.sum(area),15, cmap='rainbow')#25,vmin=np.max(np.sum(np.array(effective_ionisation_rates_multipulse[:4])*area,axis=-1)/np.sum(area))*1e-2)#, norm=LogNorm());
	peak_time =(np.sum(np.array(effective_recombination_rates_multipulse[:4])*area,axis=-1)/np.sum(area)).argmax(axis=1)
	plt.plot(temp_t[:,0][peak_time],target_chamber_pressure_multipulse[:4],'k--',label='peak location')
	plt.plot(temp_t[:,0][peak_time],target_chamber_pressure_multipulse[:4],'k+')
	plt.legend(loc=1)
	plt.xlabel('time [ms]')
	plt.ylabel('target chamber pressure [Pa]')
	cbar=plt.colorbar()
	cbar.set_label('eff. recombination rates [# m^-3 s-1]')
	# cbar.set_ticks(levs[::3])
	# cbar.set_ticklabels(['%.3g' %(_) for _ in levs[::3]])
	figure_index += 1
	plt.savefig(path_where_to_save_plots+'/global_check_physics'+str(figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.title('min_nHp/ne=%.3g, max_nH/ne=%.3g ,' %(min_nHp_ne,max_n_neutrals_ne)+mod3+' lines '+str(n_list_1.tolist()+n_list.tolist())+' n_weights '+str(n_weights)+'\narea averaged ioniastion rate, scan for distance OES/TS to target')
	area = 2*np.pi*(r_crop + np.diff([0,*r_crop])/2) * np.diff([0,*r_crop])
	levs = np.linspace(np.log10(np.max(np.sum(np.array(effective_ionisation_rates_multipulse[4:])*area,axis=-1)/np.sum(area))*1e-3),np.log10(np.max(np.sum(np.array(effective_ionisation_rates_multipulse[4:])*area,axis=-1)/np.sum(area))),num=15)
	levs = np.power(10, levs)
	# plt.contourf(temp_t[:,:4].T,(np.ones_like(temp_t[:,:4])*target_OES_distance_multipulse[4:]).T,np.sum(np.array(effective_ionisation_rates_multipulse[4:])*area,axis=-1)/np.sum(area),levs, cmap='rainbow',norm=LogNorm())#25,vmin=np.max(np.sum(np.array(effective_ionisation_rates_multipulse[4:])*area,axis=-1)/np.sum(area))*1e-2)#, norm=LogNorm());
	plt.contourf(temp_t[:,:4].T,(np.ones_like(temp_t[:,:4])*target_OES_distance_multipulse[4:]).T,np.sum(np.array(effective_ionisation_rates_multipulse[4:])*area,axis=-1)/np.sum(area),15, cmap='rainbow')#25,vmin=np.max(np.sum(np.array(effective_ionisation_rates_multipulse[4:])*area,axis=-1)/np.sum(area))*1e-2)#, norm=LogNorm());
	peak_time =(np.sum(np.array(effective_ionisation_rates_multipulse[4:])*area,axis=-1)/np.sum(area)).argmax(axis=1)
	plt.plot(temp_t[:,0][peak_time],target_OES_distance_multipulse[4:],'k--',label='peak location')
	plt.plot(temp_t[:,0][peak_time],target_OES_distance_multipulse[4:],'k+')
	plt.legend(loc=1)
	plt.xlabel('time [ms]')
	plt.ylabel('distance OES/TS to target [mm]')
	cbar=plt.colorbar()
	cbar.set_label('eff. ionisation rates [# m^-3 s-1]')
	# cbar.set_ticks(levs[::3])
	# cbar.set_ticklabels(['%.3g' %(_) for _ in levs[::3]])
	figure_index += 1
	plt.savefig(path_where_to_save_plots+'/global_check_physics'+str(figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.title('min_nHp/ne=%.3g, max_nH/ne=%.3g ,' %(min_nHp_ne,max_n_neutrals_ne)+mod3+' lines '+str(n_list_1.tolist()+n_list.tolist())+' n_weights '+str(n_weights)+'\narea averaged recombination rate, scan for distance OES/TS to target')
	area = 2*np.pi*(r_crop + np.diff([0,*r_crop])/2) * np.diff([0,*r_crop])
	levs = np.linspace(np.log10(np.max(np.sum(np.array(effective_recombination_rates_multipulse[4:])*area,axis=-1)/np.sum(area))*1e-3),np.log10(np.max(np.sum(np.array(effective_recombination_rates_multipulse[4:])*area,axis=-1)/np.sum(area))),num=15)
	levs = np.power(10, levs)
	# plt.contourf(temp_t[:,:4].T,(np.ones_like(temp_t[:,:4])*target_OES_distance_multipulse[4:]).T,np.sum(np.array(effective_recombination_rates_multipulse[4:])*area,axis=-1)/np.sum(area),levs, cmap='rainbow',norm=LogNorm())#25,vmin=np.max(np.sum(np.array(effective_ionisation_rates_multipulse[4:])*area,axis=-1)/np.sum(area))*1e-2)#, norm=LogNorm());
	plt.contourf(temp_t[:,:4].T,(np.ones_like(temp_t[:,:4])*target_OES_distance_multipulse[4:]).T,np.sum(np.array(effective_recombination_rates_multipulse[4:])*area,axis=-1)/np.sum(area),15, cmap='rainbow')#25,vmin=np.max(np.sum(np.array(effective_ionisation_rates_multipulse[4:])*area,axis=-1)/np.sum(area))*1e-2)#, norm=LogNorm());
	peak_time =(np.sum(np.array(effective_recombination_rates_multipulse[4:])*area,axis=-1)/np.sum(area)).argmax(axis=1)
	plt.plot(temp_t[:,0][peak_time],target_OES_distance_multipulse[4:],'k--',label='peak location')
	plt.plot(temp_t[:,0][peak_time],target_OES_distance_multipulse[4:],'k+')
	plt.legend(loc=1)
	plt.xlabel('time [ms]')
	plt.ylabel('distance OES/TS to target [mm]')
	cbar=plt.colorbar()
	cbar.set_label('eff. recombination rates [# m^-3 s-1]')
	# cbar.set_ticks(levs[::3])
	# cbar.set_ticklabels(['%.3g' %(_) for _ in levs[::3]])
	figure_index += 1
	plt.savefig(path_where_to_save_plots+'/global_check_physics'+str(figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	only_average=True
	plasma_static_pressure = np.multiply(merge_Te_prof_multipulse_interp_crop_limited_multipulse,merge_Te_prof_multipulse_interp_crop_limited_multipulse)*boltzmann_constant_J*eV_to_K
	plt.figure(figsize=(10, 7))
	plt.plot(target_chamber_pressure_multipulse[:4],np.max(np.sum(merge_Te_prof_multipulse_interp_crop_limited_multipulse[:4]*area,axis=-1)/sum(area),axis=(-1))/np.max(np.sum(merge_Te_prof_multipulse_interp_crop_limited_multipulse[:4]*area,axis=-1)/sum(area)),'b',label='max area averaged Te',linewidth=0.4)
	plt.plot(target_chamber_pressure_multipulse[:4],np.max(np.sum(merge_ne_prof_multipulse_interp_crop_limited_multipulse[:4]*area,axis=-1)/sum(area),axis=(-1))/np.max(np.sum(merge_ne_prof_multipulse_interp_crop_limited_multipulse[:4]*area,axis=-1)/sum(area)),'y',label='max area averaged ne',linewidth=0.4)
	plt.plot(target_chamber_pressure_multipulse[:4],np.max(np.sum((merge_ne_prof_multipulse_interp_crop_limited_multipulse*nH_ne_all_multipulse)[:4]*area,axis=-1)/sum(area),axis=(-1))/np.max(np.sum((merge_ne_prof_multipulse_interp_crop_limited_multipulse*nH_ne_all_multipulse)[:4]*area,axis=-1)/sum(area)),'c',label='max area averaged nH',linewidth=0.4)
	plt.plot(target_chamber_pressure_multipulse[:4],np.max(np.sum(plasma_static_pressure[:4]*area,axis=-1)/sum(area),axis=(-1))/np.max(np.sum(plasma_static_pressure[:4]*area,axis=-1)/sum(area)),'m',label='max area averaged plasma static pressure',linewidth=0.4)
	plt.plot(target_chamber_pressure_multipulse[:4],np.max(np.sum(effective_ionisation_rates_multipulse[:4]*area,axis=-1)/sum(area),axis=(-1))/np.max(np.sum(effective_ionisation_rates_multipulse[:4]*area,axis=-1)/sum(area)),'g',label='max area averaged effective ionisation rate',linewidth=0.4)
	plt.plot(target_chamber_pressure_multipulse[:4],np.max(np.sum(effective_recombination_rates_multipulse[:4]*area,axis=-1)/sum(area),axis=(-1))/np.max(np.sum(effective_recombination_rates_multipulse[:4]*area,axis=-1)/sum(area)),'r',label='max area averaged effective recombination rate',linewidth=0.4)

	# plt.plot(target_chamber_pressure_multipulse[:4],np.max(np.sum((merge_ne_prof_multipulse_interp_crop_limited_multipulse*nHm_ne_all_multipulse)[:4]*area,axis=-1)/sum(area),axis=(-1))/np.max(np.sum((merge_ne_prof_multipulse_interp_crop_limited_multipulse*nHm_ne_all_multipulse)[:4]*area,axis=-1)/sum(area)),'brown',label='max area averaged nH-',linewidth=0.4)
	# plt.plot(target_chamber_pressure_multipulse[:4],np.max(np.sum(rate_creation_Hm_multipulse[:4]*area,axis=-1)/sum(area),axis=(-1))/np.max(np.sum(rate_creation_Hm_multipulse[:4]*area,axis=-1)/sum(area)),'orange',label='max area averaged H- creation rate',linewidth=0.4)
	# plt.plot(target_chamber_pressure_multipulse[:4],np.max(np.sum(rate_destruction_Hm_multipulse[:4]*area,axis=-1)/sum(area),axis=(-1))/np.max(np.sum(rate_destruction_Hm_multipulse[:4]*area,axis=-1)/sum(area)),'pink',label='max area averaged H- destruction rate',linewidth=0.4)
	if not only_average:
		plt.plot(target_chamber_pressure_multipulse[:4],np.max(merge_Te_prof_multipulse_interp_crop_limited_multipulse[:4],axis=(-1,-2))/np.max(merge_Te_prof_multipulse_interp_crop_limited_multipulse[:4]),'bo',label='max peak Te',linewidth=0.4)
		plt.plot(target_chamber_pressure_multipulse[:4],np.max(merge_ne_prof_multipulse_interp_crop_limited_multipulse[:4],axis=(-1,-2))/np.max(merge_ne_prof_multipulse_interp_crop_limited_multipulse[:4]),'yo',label='max peak ne',linewidth=0.4)
		plt.plot(target_chamber_pressure_multipulse[:4],np.max((merge_ne_prof_multipulse_interp_crop_limited_multipulse*nH_ne_all_multipulse)[:4],axis=(-1,-2))/np.max((merge_ne_prof_multipulse_interp_crop_limited_multipulse*nH_ne_all_multipulse)[:4]),'co',label='max peak nH',linewidth=0.4)
		plt.plot(target_chamber_pressure_multipulse[:4],np.max(plasma_static_pressure[:4],axis=(-1,-2))/np.max(plasma_static_pressure[:4]),'mo',label='max peak plasma static pressure',linewidth=0.4)
		plt.plot(target_chamber_pressure_multipulse[:4],np.max(effective_ionisation_rates_multipulse[:4],axis=(-1,-2))/np.max(effective_ionisation_rates_multipulse[:4]),'go',label='max peak effective ionisation rate',linewidth=0.4)
		plt.plot(target_chamber_pressure_multipulse[:4],np.max(effective_recombination_rates_multipulse[:4],axis=(-1,-2))/np.max(effective_recombination_rates_multipulse[:4]),'ro',label='max peak effective recombination rate',linewidth=0.4)

		# plt.plot(target_chamber_pressure_multipulse[:4],np.max((merge_ne_prof_multipulse_interp_crop_limited_multipulse*nHm_ne_all_multipulse)[:4],axis=(-1,-2))/np.max((merge_ne_prof_multipulse_interp_crop_limited_multipulse*nHm_ne_all_multipulse)[:4]),'o',color='brown',label='max peak nH',linewidth=0.4)
		# plt.plot(target_chamber_pressure_multipulse[:4],np.max(rate_creation_Hm_multipulse[:4],axis=(-1,-2))/np.max(rate_creation_Hm_multipulse[:4]),'o',color='orange',label='max peak H- creation rate',linewidth=0.4)
		# plt.plot(target_chamber_pressure_multipulse[:4],np.max(rate_destruction_Hm_multipulse[:4],axis=(-1,-2))/np.max(rate_destruction_Hm_multipulse[:4]),'o',color='pink',label='max peak H- destruction rate',linewidth=0.4)

	plt.legend(loc='best',fontsize='xx-small')
	plt.xlabel('target chamber pressure [Pa]')
	plt.ylabel('peak value respect to the maximum peak value [au]')
	plt.title('min_nHp/ne=%.3g, max_nH/ne=%.3g ,' %(min_nHp_ne,max_n_neutrals_ne)+mod3+' lines '+str(n_list_1.tolist()+n_list.tolist())+' n_weights '+str(n_weights)+'\nscan for target chamber pressure')
	plt.grid()
	figure_index += 1
	plt.savefig(path_where_to_save_plots+'/global_check_physics'+str(figure_index) + '.eps', bbox_inches='tight')
	plt.close()


	only_average=True
	plt.figure(figsize=(10, 7))
	plt.plot(target_OES_distance_multipulse[4:],np.max(np.sum(merge_Te_prof_multipulse_interp_crop_limited_multipulse[4:]*area,axis=-1)/sum(area),axis=(-1))/np.max(np.sum(merge_Te_prof_multipulse_interp_crop_limited_multipulse[4:]*area,axis=-1)/sum(area)),'b',label='max area averaged Te',linewidth=0.4)
	plt.plot(target_OES_distance_multipulse[4:],np.max(np.sum(merge_ne_prof_multipulse_interp_crop_limited_multipulse[4:]*area,axis=-1)/sum(area),axis=(-1))/np.max(np.sum(merge_ne_prof_multipulse_interp_crop_limited_multipulse[4:]*area,axis=-1)/sum(area)),'y',label='max area averaged ne',linewidth=0.4)
	plt.plot(target_OES_distance_multipulse[4:],np.max(np.sum((merge_ne_prof_multipulse_interp_crop_limited_multipulse*nH_ne_all_multipulse)[4:]*area,axis=-1)/sum(area),axis=(-1))/np.max(np.sum((merge_ne_prof_multipulse_interp_crop_limited_multipulse*nH_ne_all_multipulse)[4:]*area,axis=-1)/sum(area)),'c',label='max area averaged nH',linewidth=0.4)
	plt.plot(target_OES_distance_multipulse[4:],np.max(np.sum(plasma_static_pressure[4:]*area,axis=-1)/sum(area),axis=(-1))/np.max(np.sum(plasma_static_pressure[4:]*area,axis=-1)/sum(area)),'m',label='max area averaged plasma static pressure',linewidth=0.4)
	plt.plot(target_OES_distance_multipulse[4:],np.max(np.sum(effective_ionisation_rates_multipulse[4:]*area,axis=-1)/sum(area),axis=(-1))/np.max(np.sum(effective_ionisation_rates_multipulse[4:]*area,axis=-1)/sum(area)),'g',label='max area averaged effective ionisation rate',linewidth=0.4)
	plt.plot(target_OES_distance_multipulse[4:],np.max(np.sum(effective_recombination_rates_multipulse[4:]*area,axis=-1)/sum(area),axis=(-1))/np.max(np.sum(effective_recombination_rates_multipulse[4:]*area,axis=-1)/sum(area)),'r',label='max area averaged effective recombination rate',linewidth=0.4)

	# plt.plot(target_OES_distance_multipulse[4:],np.max(np.sum((merge_ne_prof_multipulse_interp_crop_limited_multipulse*nHm_ne_all_multipulse)[4:]*area,axis=-1)/sum(area),axis=(-1))/np.max(np.sum((merge_ne_prof_multipulse_interp_crop_limited_multipulse*nHm_ne_all_multipulse)[4:]*area,axis=-1)/sum(area)),'brown',label='max area averaged nH-',linewidth=0.4)
	# plt.plot(target_OES_distance_multipulse[4:],np.max(np.sum(rate_creation_Hm_multipulse[4:]*area,axis=-1)/sum(area),axis=(-1))/np.max(np.sum(rate_creation_Hm_multipulse[4:]*area,axis=-1)/sum(area)),'orange',label='max area averaged H- creation rate',linewidth=0.4)
	# plt.plot(target_OES_distance_multipulse[4:],np.max(np.sum(rate_destruction_Hm_multipulse[4:]*area,axis=-1)/sum(area),axis=(-1))/np.max(np.sum(rate_destruction_Hm_multipulse[4:]*area,axis=-1)/sum(area)),'pink',label='max area averaged H- destruction rate',linewidth=0.4)
	if not only_average:
		plt.plot(target_OES_distance_multipulse[4:],np.max(merge_Te_prof_multipulse_interp_crop_limited_multipulse[4:],axis=(-1,-2))/np.max(merge_Te_prof_multipulse_interp_crop_limited_multipulse[4:]),'bo',label='max peak Te',linewidth=0.4)
		plt.plot(target_OES_distance_multipulse[4:],np.max(merge_ne_prof_multipulse_interp_crop_limited_multipulse[4:],axis=(-1,-2))/np.max(merge_ne_prof_multipulse_interp_crop_limited_multipulse[4:]),'yo',label='max peak ne',linewidth=0.4)
		plt.plot(target_OES_distance_multipulse[4:],np.max((merge_ne_prof_multipulse_interp_crop_limited_multipulse*nH_ne_all_multipulse)[4:],axis=(-1,-2))/np.max((merge_ne_prof_multipulse_interp_crop_limited_multipulse*nH_ne_all_multipulse)[4:]),'co',label='max peak nH',linewidth=0.4)
		plt.plot(target_OES_distance_multipulse[4:],np.max(plasma_static_pressure[4:],axis=(-1,-2))/np.max(plasma_static_pressure[4:]),'mo',label='max peak plasma static pressure',linewidth=0.4)
		plt.plot(target_OES_distance_multipulse[4:],np.max(effective_ionisation_rates_multipulse[4:],axis=(-1,-2))/np.max(effective_ionisation_rates_multipulse[4:]),'go',label='max peak effective ionisation rate',linewidth=0.4)
		plt.plot(target_OES_distance_multipulse[4:],np.max(effective_recombination_rates_multipulse[4:],axis=(-1,-2))/np.max(effective_recombination_rates_multipulse[4:]),'ro',label='max peak effective recombination rate',linewidth=0.4)

		# plt.plot(target_OES_distance_multipulse[4:],np.max((merge_ne_prof_multipulse_interp_crop_limited_multipulse*nHm_ne_all_multipulse)[4:],axis=(-1,-2))/np.max((merge_ne_prof_multipulse_interp_crop_limited_multipulse*nHm_ne_all_multipulse)[4:]),'o',color='brown',label='max peak nH',linewidth=0.4)
		# plt.plot(target_OES_distance_multipulse[4:],np.max(rate_creation_Hm_multipulse[:4],axis=(-1,-2))/np.max(rate_creation_Hm_multipulse[4:]),'o',color='orange',label='max peak H- creation rate',linewidth=0.4)
		# plt.plot(target_OES_distance_multipulse[4:],np.max(rate_destruction_Hm_multipulse[:4],axis=(-1,-2))/np.max(rate_destruction_Hm_multipulse[4:]),'o',color='pink',label='max peak H- destruction rate',linewidth=0.4)
	plt.legend(loc='best',fontsize='x-small')
	plt.xlabel('distance OES/TS to target [mm]')
	plt.ylabel('peak value respect to the maximum peak value [au]')
	plt.title('min_nHp/ne=%.3g, max_nH/ne=%.3g ,' %(min_nHp_ne,max_n_neutrals_ne)+mod3+' lines '+str(n_list_1.tolist()+n_list.tolist())+' n_weights '+str(n_weights)+'\nscan for distance OES/TS to target')
	plt.grid()
	figure_index += 1
	plt.savefig(path_where_to_save_plots+'/global_check_physics'+str(figure_index) + '.eps', bbox_inches='tight')
	plt.close()




	threshold_radious = 0.015

	fig, ax = plt.subplots(figsize=(10, 7))
	fig.suptitle('Comparison between high and low pressure case, OES/target %.3gmm\naverage limited to 0>r>%.3gmm to avoid TS imprecisions at low ne' % (target_OES_distance_multipulse[0],(1000*threshold_radious)))
	ax2 = ax.twinx()
	pointer = 1
	ax.plot( temp_t[:,0],np.sum((merge_Te_prof_multipulse_interp_crop_limited_multipulse[pointer]*area)[:,r_crop<=threshold_radious],axis=(-1))/np.sum(area[r_crop<=threshold_radious]),'b',label='peak area averaged Te for pressure %.3gPa' %target_chamber_pressure_multipulse[pointer])
	ax.plot( temp_t[:,0],np.max(merge_Te_prof_multipulse_interp_crop_limited_multipulse[pointer],axis=(-1)),'b--',label='peak Te for pressure %.3gPa' %target_chamber_pressure_multipulse[pointer])
	ax2.plot( temp_t[:,0],np.sum((merge_ne_prof_multipulse_interp_crop_limited_multipulse[pointer]*area)[:,r_crop<=threshold_radious],axis=(-1))/np.sum(area[r_crop<=threshold_radious]),'r',label='peak area averaged ne for pressure %.3gPa' %target_chamber_pressure_multipulse[pointer])
	ax2.plot( temp_t[:,0],np.max(merge_ne_prof_multipulse_interp_crop_limited_multipulse[pointer],axis=(-1)),'r--',label='peak ne for pressure %.3gPa' %target_chamber_pressure_multipulse[pointer])
	pointer = 3
	ax.plot( temp_t[:,0],np.sum((merge_Te_prof_multipulse_interp_crop_limited_multipulse[pointer]*area)[:,r_crop<=threshold_radious],axis=(-1))/np.sum(area[r_crop<=threshold_radious]),'g',label='peak area averaged Te for pressure %.3gPa' %target_chamber_pressure_multipulse[pointer])
	ax.plot( temp_t[:,0],np.max(merge_Te_prof_multipulse_interp_crop_limited_multipulse[pointer],axis=(-1)),'g--',label='peak Te for pressure %.3gPa' %target_chamber_pressure_multipulse[pointer])
	ax2.plot( temp_t[:,0],np.sum((merge_ne_prof_multipulse_interp_crop_limited_multipulse[pointer]*area)[:,r_crop<=threshold_radious],axis=(-1))/np.sum(area[r_crop<=threshold_radious]),'y',label='peak area averaged ne for pressure %.3gPa' %target_chamber_pressure_multipulse[pointer])
	ax2.plot( temp_t[:,0],np.max(merge_ne_prof_multipulse_interp_crop_limited_multipulse[pointer],axis=(-1)),'y--',label='peak ne for pressure %.3gPa' %target_chamber_pressure_multipulse[pointer])
	ax.set_yscale('log')
	ax2.set_yscale('log')
	ax.set_ylim(bottom=0.1)
	ax2.set_ylim(bottom=1)
	ax2.grid()
	ax.grid()
	ax2.legend(loc=7,fontsize='small')
	ax.legend(loc=1,fontsize='small')
	ax2.set_ylabel('ne [# m^-3]')
	ax.set_ylabel('Te [eV]')
	ax.set_xlabel('time [ms]')
	figure_index += 1
	plt.savefig(path_where_to_save_plots+'/global_check_physics'+str(figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	fig, ax = plt.subplots(figsize=(10, 7))
	fig.suptitle('Comparison between close and far target case, pressure %.3gPa\naverage limited to 0>r>%.3gmm to avoid TS imprecisions at low ne' % (target_chamber_pressure_multipulse[4],(1000*threshold_radious)))
	ax2 = ax.twinx()
	pointer = 4
	ax.plot( temp_t[:,0],np.sum((merge_Te_prof_multipulse_interp_crop_limited_multipulse[pointer]*area)[:,r_crop<=threshold_radious],axis=(-1))/np.sum(area[r_crop<=threshold_radious]),'b',label='peak area averaged Te for OES/target %.3gmm' %target_OES_distance_multipulse[pointer])
	ax.plot( temp_t[:,0],np.max(merge_Te_prof_multipulse_interp_crop_limited_multipulse[pointer],axis=(-1)),'b--',label='peak Te for OES/target %.3gmm' %target_OES_distance_multipulse[pointer])
	ax2.plot( temp_t[:,0],np.sum((merge_ne_prof_multipulse_interp_crop_limited_multipulse[pointer]*area)[:,r_crop<=threshold_radious],axis=(-1))/np.sum(area[r_crop<=threshold_radious]),'r',label='peak area averaged ne for OES/target %.3gmm' %target_OES_distance_multipulse[pointer])
	ax2.plot( temp_t[:,0],np.max(merge_ne_prof_multipulse_interp_crop_limited_multipulse[pointer],axis=(-1)),'r--',label='peak ne for OES/target %.3gmm' %target_OES_distance_multipulse[pointer])
	pointer = 7
	ax.plot( temp_t[:,0],np.sum((merge_Te_prof_multipulse_interp_crop_limited_multipulse[pointer]*area)[:,r_crop<=threshold_radious],axis=(-1))/np.sum(area[r_crop<=threshold_radious]),'g',label='peak area averaged Te for OES/target %.3gmm' %target_OES_distance_multipulse[pointer])
	ax.plot( temp_t[:,0],np.max(merge_Te_prof_multipulse_interp_crop_limited_multipulse[pointer],axis=(-1)),'g--',label='peak Te for OES/target %.3gmm' %target_OES_distance_multipulse[pointer])
	ax2.plot( temp_t[:,0],np.sum((merge_ne_prof_multipulse_interp_crop_limited_multipulse[pointer]*area)[:,r_crop<=threshold_radious],axis=(-1))/np.sum(area[r_crop<=threshold_radious]),'y',label='peak area averaged ne for OES/target %.3gmm' %target_OES_distance_multipulse[pointer])
	ax2.plot( temp_t[:,0],np.max(merge_ne_prof_multipulse_interp_crop_limited_multipulse[pointer],axis=(-1)),'y--',label='peak ne for OES/target %.3gmm' %target_OES_distance_multipulse[pointer])
	ax.set_yscale('log')
	ax2.set_yscale('log')
	ax.set_ylim(bottom=0.1)
	ax2.set_ylim(bottom=1)
	ax2.grid()
	ax.grid()
	ax2.legend(loc=7,fontsize='small')
	ax.legend(loc=1,fontsize='small')
	ax2.set_ylabel('ne [# m^-3]')
	ax.set_ylabel('Te [eV]')
	ax.set_xlabel('time [ms]')
	figure_index += 1
	plt.savefig(path_where_to_save_plots+'/global_check_physics'+str(figure_index) + '.eps', bbox_inches='tight')
	plt.close()
