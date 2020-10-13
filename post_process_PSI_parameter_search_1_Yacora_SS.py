import numpy as np
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())
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
from scipy.signal import find_peaks, peak_prominences as get_proms
import time as tm
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
pecfile_2 = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/pec12#h_pju#h0.dat'
scdfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/scd12_h.dat'
acdfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/acd12_h.dat'

if True:  # copy of post_process_PSI_parameter_search_1_Yacora at 08/05/2020 just for SS

	# merge_ID_target_multipulse = [851,86,87,89,92, 93, 94,95,96,97,98,99]
	merge_ID_target_multipulse = [98,97,96]

	for merge_ID_target in merge_ID_target_multipulse:  # 88 excluded because I don't have a temperature profile
		if (merge_ID_target>=93 and merge_ID_target<100):
			merge_time_window = [-1,2]
		else:
			merge_time_window = [-10,10]


		# recorded_data_override = [False,False]
		recorded_data_override = [True,True]

		# merge_ID_target = 85

		# for min_nHp_ne,atomic_restricted_high_n in [[0.999,False],[0.8,False],[0.999,True],[0.3,False],[0,False]]:	#0.9999 is for the case when I relax this boundary if nH<0.001
		# for min_nHp_ne,atomic_restricted_high_n in [[0.01,True],[0.1,True],[0.5,True],[0.7,True],[0.999,True]]:
		for min_nHp_ne,atomic_restricted_high_n in [[0,True]]:

			# for max_n_neutrals_ne in [1,2,10,100]:
			for max_n_neutrals_ne in [np.inf]:

				figure_index = 0

				# calculate_geometry = False
				# merge_ID_target = 17	#	THIS IS GIVEN BY THE LAUNCHER
				for i in range(10):
					print('.')
				print('Starting to work on merge number' + str(merge_ID_target))
				for i in range(10):
					print('.')

				time_resolution_scan = False
				time_resolution_scan_improved = True
				time_resolution_extra_skip = 0

				if False:
					n_list = np.array([5, 6, 7,8, 9,10,11])
					n_list_1 = np.array([4])
					n_weights = [1, 1, 1, 1, 1, 1,3,3]
				elif True:
					n_list = np.array([5, 6, 7,8])
					n_list_1 = np.array([4])
					n_weights = [1, 1, 1, 1, 1]
				# for index in (n_list_1 - 4):
				# 	n_weights[index] = 4
				# n_weights[np.max(n_list_1) - 3] = 2
				# min_nH_ne = 0.6

				max_nHp_ne = 1
				min_nH_ne = 0

				mod_atomic_restricted_high_n = ''
				if atomic_restricted_high_n ==True:
					mod_atomic_restricted_high_n = '_atomic_restricted_high_n'
					print('calculus of nH and nH+ limited to lines >=8')

				mod = '/Yacora_Bayesian/absolute/lines_fitted'+str(len(n_list)+len(n_list_1))+'/min_nHp_ne' + str(min_nHp_ne) + mod_atomic_restricted_high_n+'/max_n_neutrals_ne' + str(
					max_n_neutrals_ne)

				boltzmann_constant_J = 1.380649e-23	# J/K
				eV_to_K = 8.617333262145e-5	# eV/K
				avogadro_number = 6.02214076e23
				hydrogen_mass = 1.008*1.660*1e-27	# kg
				electron_mass = 9.10938356* 1e-31	# kg
				all_j=find_index_of_file(merge_ID_target,df_settings,df_log,only_OES=True)
				target_chamber_pressure = []
				target_OES_distance = []
				feed_rate_SLM = []
				for j in all_j:
					target_chamber_pressure.append(df_log.loc[j,['p_n [Pa]']])
					target_OES_distance.append(df_log.loc[j,['T_axial']])
					feed_rate_SLM.append(df_log.loc[j,['Seed']])
				target_chamber_pressure = np.nanmean(target_chamber_pressure)	# Pa
				target_OES_distance = np.nanmean(target_OES_distance)	# Pa
				feed_rate_SLM = np.nanmean(feed_rate_SLM)	# SLM
				# Ideal gas law
				max_nH2_from_pressure = target_chamber_pressure/(boltzmann_constant_J*300)	# [#/m^3] I suppose ambient temp is ~ 300K

				print('lines')
				print(str(n_list_1.tolist()+n_list.tolist()))
				print('min_nHp/ne')
				print(min_nHp_ne)
				print('max_nH/ne')
				print(max_n_neutrals_ne)
				print('steady state nH2 [#/m^3]')
				print(max_nH2_from_pressure)
				if False:	# I cannot assume a constant temperature, so I use a simple model in the target chamber heating to estimate the max H2 density.
					max_nH2_from_pressure = 10*max_nH2_from_pressure
					print("considering that it's a transient phenomena the istantaneous max_nH2 is set 10 times higher to [#/m^3]")
					print(max_nH2_from_pressure)
				print('max_nHp/ne')
				print(max_nHp_ne)
				print('n_weights')
				print(n_weights)
				print('recorded data override is '+str(recorded_data_override))

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
				path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(
					merge_ID_target) #+ '_back'

				if not os.path.exists(path_where_to_save_everything + mod):
					os.makedirs(path_where_to_save_everything + mod)

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
				energy_difference = np.array([2.54970, 2.85567, 3.02187, 3.12208, 3.18716, 3.23175, 3.26365, 3.28725, 3.30520])  # eV
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

				try:
					inverted_profiles_original = np.load(path_where_to_save_everything + '/SS_inverted_profiles.npy')  # in W m^-3 sr^-1  *  4*pi sr  =  W m^-3
					inverted_profiles_sigma_original = np.load(path_where_to_save_everything + '/SS_inverted_profiles_sigma.npy')  # in W m^-3 sr^-1  *  4*pi sr  =  W m^-3
					all_fits = np.load(path_where_to_save_everything + '/merge' + str(merge_ID_target) + '_SS_all_fits.npy')  # in W m^-2 sr^-1
					merge_Te_prof_multipulse = np.load(path_where_to_save_everything + '/TS_SS_data_merge_' + str(merge_ID_target) + '.npz')['merge_Te_prof_multipulse']
					merge_dTe_multipulse = np.load(path_where_to_save_everything + '/TS_SS_data_merge_' + str(merge_ID_target) + '.npz')['merge_dTe_multipulse']
					merge_ne_prof_multipulse = np.load(path_where_to_save_everything + '/TS_SS_data_merge_' + str(merge_ID_target) + '.npz')['merge_ne_prof_multipulse']
					merge_dne_multipulse = np.load(path_where_to_save_everything + '/TS_SS_data_merge_' + str(merge_ID_target) + '.npz')['merge_dne_multipulse']
				except Exception as e:
					print('WARNING for merge '+str(merge_ID_target)+' data missing')
					print(e)
					continue

				# for time_shift_factor in time_shift_factor_all:
				#
				# 	mod = '/spatial_factor' + str(spatial_factor)+'/time_shift_factor' + str(time_shift_factor)
				# 	if not os.path.exists(path_where_to_save_everything + mod):
				# 		os.makedirs(path_where_to_save_everything + mod)

				# for spatial_factor in [1,1.1,1.2,1.3,1.4,1.5]:
				for spatial_factor in [1]:
					# for time_shift_factor in [-0.1,-0.05,0,0.05]:
					for time_shift_factor in [0]:
						mod2 = mod + '/spatial_factor_' + str(spatial_factor) + '/time_shift_factor_' + str(time_shift_factor)

						if not os.path.exists(path_where_to_save_everything + mod2):
							os.makedirs(path_where_to_save_everything + mod2)

						# dx = 18 / 40 * (50.5 / 27.4) / 1e3
						dx = 1.06478505470992 / 1e3	# 10/02/2020 from	Calculate magnification_FF.xlsx
						xx = np.arange(40) * dx  # m
						xn = np.linspace(0, max(xx), 1000)
						# r = np.linspace(-max(xx) / 2, max(xx) / 2, 1000)
						number_of_radial_divisions = np.shape(inverted_profiles_original)[-1]
						r = np.arange(number_of_radial_divisions)*dx
						dr = np.median(np.diff(r))

						# I don't need this because I use pure abel
						if False:	# modification for an iterative aproach and finding a self-consistent solution in the SS case
							reduction_in_radius = 5
							r_short = []
							dr = np.median(np.diff(r))
							inverted_profiles_original_short = []
							for i_r in range(int(len(r)/reduction_in_radius)):
								inverted_profiles_original_short.append(np.sum(inverted_profiles_original[:,i_r*reduction_in_radius:(i_r+1)*reduction_in_radius]*r[i_r*reduction_in_radius:(i_r+1)*reduction_in_radius],axis=-1)/np.sum(r[i_r*reduction_in_radius:(i_r+1)*reduction_in_radius]))	# the area of an anular ring of width dx depends linearly on r
								r_short.append(np.min(np.abs(r[i_r*reduction_in_radius:(i_r+1)*reduction_in_radius])))
							inverted_profiles_original_short = np.array(inverted_profiles_original_short).T
							r_short = np.array(r_short)
						inverted_profiles = (1 / spatial_factor) * inverted_profiles_original  # in W m^-3 sr^-1  *  4*pi sr  =  W m^-3
						inverted_profiles_sigma = cp.deepcopy(inverted_profiles_sigma_original)

						if np.max(merge_Te_prof_multipulse) <= 0:
							print('merge' + str(merge_ID_target) + " has no recorded temperature")
							continue

						TS_size = [-4.149230769230769056e+01, 4.416923076923076508e+01]
						TS_r = TS_size[0] + np.linspace(0, 1, 65) * (TS_size[1] - TS_size[0])
						TS_dr = np.median(np.diff(TS_r)) / 1000
						gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
						yy = merge_ne_prof_multipulse
						yy_sigma = merge_dne_multipulse
						yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
						yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
						p0 = [np.max(yy), 10, 0]
						bds = [[0, 0, np.min(TS_r)], [np.inf, TS_size[1], np.max(TS_r)]]
						fit = curve_fit(gauss, TS_r, yy, p0, sigma=yy_sigma, maxfev=100000, bounds=bds)
						profile_centres=[fit[0][-1]]
						profile_sigma=[fit[0][-2]]
						profile_centres_score=[fit[1][-1, -1]**0.5]
						# plt.figure();plt.plot(TS_r,merge_ne_prof_multipulse,label='ne')
						# plt.plot([fit[0][-1],fit[0][-1]],[np.max(merge_ne_prof_multipulse),np.min(merge_ne_prof_multipulse)],'--',label='ne')
						yy = merge_Te_prof_multipulse
						yy_sigma = merge_dTe_multipulse
						yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
						yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
						p0 = [np.max(yy), 10, 0]
						bds = [[0, 0, np.min(TS_r)], [np.inf, TS_size[1], np.max(TS_r)]]
						fit = curve_fit(gauss, TS_r, yy, p0, sigma=yy_sigma, maxfev=100000, bounds=bds)
						profile_centres = np.append(profile_centres,[fit[0][-1]],axis=0)
						profile_sigma = np.append(profile_sigma,[fit[0][-2]],axis=0)
						profile_centres_score = np.append(profile_centres_score,[fit[1][-1, -1]**0.5],axis=0)
						centre = np.nanmean(profile_centres)
						TS_r_new = (TS_r - centre) / 1000
						print('TS profile centre at %.3gmm compared to the theoretical centre' %centre)
						# plt.plot(TS_r,merge_Te_prof_multipulse,label='Te')
						# plt.plot([fit[0][-1],fit[0][-1]],[np.max(merge_Te_prof_multipulse),np.min(merge_Te_prof_multipulse)],'--',label='ne')
						# plt.plot(TS_r_new*1000,merge_ne_prof_multipulse,label='Te')
						# plt.plot(TS_r_new*1000,merge_Te_prof_multipulse,label='Te')
						# plt.pause(0.01)

						# This is the mean of Te and ne weighted in their own uncertainties.
						interp_range_r = max(dx, TS_dr) * 1
						# weights_r = TS_r_new/interp_range_r
						merge_Te_prof_multipulse_interp = np.zeros_like(inverted_profiles[ 0])
						merge_dTe_prof_multipulse_interp = np.zeros_like(inverted_profiles[ 0])
						merge_ne_prof_multipulse_interp = np.zeros_like(inverted_profiles[ 0])
						merge_dne_prof_multipulse_interp = np.zeros_like(inverted_profiles[ 0])
						for i_r, value_r in enumerate(np.abs(r)):
							if np.sum(np.abs(np.abs(TS_r_new) - value_r) < interp_range_r) == 0:
								continue
							selected_values = np.abs(np.abs(TS_r_new) - value_r) < interp_range_r
							selected_values[merge_Te_prof_multipulse == 0] = False
							# weights = 1/np.abs(weights_r[selected_values]+1e-5)
							# weights = np.ones_like(weights_r[selected_values])
							weights = np.ones((np.sum(selected_values)))
							if np.sum(selected_values) == 0:
								continue
							merge_Te_prof_multipulse_interp[i_r] = np.sum(merge_Te_prof_multipulse[selected_values]*weights / merge_dTe_multipulse[selected_values]) / np.sum(weights / merge_dTe_multipulse[selected_values])
							merge_ne_prof_multipulse_interp[i_r] = np.sum(merge_ne_prof_multipulse[selected_values]*weights / merge_dne_multipulse[selected_values]) / np.sum(weights / merge_dne_multipulse[selected_values])
							if False: 	# suggestion from Daljeet: use the "worst case scenario" in therms of uncertainty
								merge_dTe_prof_multipulse_interp[i_r] = 1/(np.sum(1 / merge_dTe_multipulse[selected_values]))*(np.sum( ((merge_Te_prof_multipulse_interp[i_r]-merge_Te_prof_multipulse[selected_values])/merge_dTe_multipulse[selected_values])**2 )**0.5)
								merge_dne_prof_multipulse_interp[i_r] = 1/(np.sum(1 / merge_dne_multipulse[selected_values]))*(np.sum( ((merge_ne_prof_multipulse_interp[i_r]-merge_ne_prof_multipulse[selected_values])/merge_dne_multipulse[selected_values])**2 )**0.5)
							else:
								merge_dTe_prof_multipulse_interp_temp = 1/(np.sum(1 / merge_dTe_multipulse[selected_values]))*(np.sum( ((merge_Te_prof_multipulse_interp[i_r]-merge_Te_prof_multipulse[selected_values])/merge_dTe_multipulse[selected_values])**2 )**0.5)
								merge_dne_prof_multipulse_interp_temp = 1/(np.sum(1 / merge_dne_multipulse[selected_values]))*(np.sum( ((merge_ne_prof_multipulse_interp[i_r]-merge_ne_prof_multipulse[selected_values])/merge_dne_multipulse[selected_values])**2 )**0.5)
								merge_dTe_prof_multipulse_interp[i_r] = max(merge_dTe_prof_multipulse_interp_temp,(np.max(merge_Te_prof_multipulse[selected_values])-np.min(merge_Te_prof_multipulse[selected_values]))/2 )
								merge_dne_prof_multipulse_interp[i_r] = max(merge_dne_prof_multipulse_interp_temp,(np.max(merge_ne_prof_multipulse[selected_values])-np.min(merge_ne_prof_multipulse[selected_values]))/2 )
						temp_r, temp_t = np.meshgrid(r, new_timesteps)
						# plt.plot(r*1000,merge_Te_prof_multipulse_interp,label='Te')
						# plt.plot(r*1000,merge_ne_prof_multipulse_interp,label='Te')
						# plt.pause(0.01)

						start_r = np.abs(r - 0).argmin()
						end_r = np.abs(r - 5).argmin() + 1
						r_crop = r[start_r:end_r]
						merge_Te_prof_multipulse_interp_crop = merge_Te_prof_multipulse_interp[start_r:end_r]
						merge_dTe_prof_multipulse_interp_crop = merge_dTe_prof_multipulse_interp[start_r:end_r]
						merge_ne_prof_multipulse_interp_crop = merge_ne_prof_multipulse_interp[start_r:end_r]
						merge_dne_prof_multipulse_interp_crop = merge_dne_prof_multipulse_interp[start_r:end_r]
						inverted_profiles_crop = inverted_profiles[:, start_r:end_r]
						inverted_profiles_crop[np.isnan(inverted_profiles_crop)] = 0
						inverted_profiles_sigma_crop = inverted_profiles_sigma[ :, start_r:end_r]
						inverted_profiles_sigma_crop[np.isnan(inverted_profiles_sigma_crop)] = 0
						all_fits_crop = all_fits
						# inverted_profiles_crop[inverted_profiles_crop<0] = 0

						gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
						yy = merge_ne_prof_multipulse_interp_crop
						yy_sigma = merge_dne_prof_multipulse_interp_crop
						yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
						yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
						p0 = [np.max(yy), np.max(r_crop)/2, np.min(r_crop)]
						bds = [[0, 0, np.min(r_crop)], [np.inf, np.max(r_crop), np.max(r_crop)]]
						fit = curve_fit(gauss, r_crop, yy, p0, sigma=yy_sigma, maxfev=100000, bounds=bds)
						averaged_profile_sigma=fit[0][-2]
						# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)

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
						temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (10 ** -6)  # in ionisations m^-3 s-1 / (# / m^3)**2
						ionization_length_H = thermal_velocity_H/(temp * merge_ne_prof_multipulse_interp_crop_limited * 1e20 )
						ionization_length_H = np.where(np.isnan(ionization_length_H), 0, ionization_length_H)
						ionization_length_H = np.where(np.isinf(ionization_length_H), np.nan, ionization_length_H)
						ionization_length_H = np.where(np.isnan(ionization_length_H), np.nanmax(ionization_length_H[np.isfinite(ionization_length_H)]), ionization_length_H)
						ionization_length_Hm = np.ones_like(ionization_length_H)
						ionization_length_H2 = np.ones_like(ionization_length_H)
						ionization_length_H2p = np.ones_like(ionization_length_H)

						min_multiplier = 1e-10
						# # for max_nHm_ne,max_nH2_ne,max_nH2p_ne,max_nH3p_ne,mod3 in [[min_multiplier,min_multiplier,min_multiplier,min_multiplier,'only_H'],[1,min_multiplier,min_multiplier,min_multiplier,'only_Hm'],[min_multiplier,1,min_multiplier,min_multiplier,'only_H2'],[min_multiplier,min_multiplier,1,min_multiplier,'only_H2p'],[min_multiplier,min_multiplier,min_multiplier,1,'only_H3p'],[min_multiplier,1,1,min_multiplier,'only_H2_H2p'],[1,1,min_multiplier,min_multiplier,'only_Hm_H2'],[1,1,1,min_multiplier,'only_Hm_H2_H2p'],[1,1,1,1,'all']]:
						# for max_nHm_ne,max_nH2_ne,max_nH2p_ne,max_nH3p_ne,mod3 in [[1,min_multiplier,min_multiplier,min_multiplier,'only_Hm'],[min_multiplier,1,1,min_multiplier,'only_H2_H2p'],[1,1,min_multiplier,min_multiplier,'only_Hm_H2'],[1,1,1,min_multiplier,'only_Hm_H2_H2p'],[1,1,1,1,'all']]:
						mod3 = 'only_H'
						mod4 = mod2 +'/' +mod3

						if not os.path.exists(path_where_to_save_everything + mod4):
							os.makedirs(path_where_to_save_everything + mod4)

						print('Starting '+mod3)

						n_list_all = np.sort(np.concatenate((n_list, n_list_1)))
						# OES_multiplier = 0.81414701
						# Te_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						# ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						nH_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						nHp_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						nHm_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						nH2_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						nH2p_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						nH3p_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)

						residuals_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						# guess = [0, 1, 0.01 * max_nHm_ne, 0.1 * max_nH2_ne, 0.0001 * max_nH2p_ne,0.0001 * max_nH3p_ne]
						guess = [0.5, 1, 0.0005,0.0005, 0.0001,0.0005]


						# this is to estimate better the background max H2 density
						# I assume the flow velocity from CTS (Jonathan) of 10km/s
						T_Hp = np.min([np.max([1000*np.ones_like(merge_Te_prof_multipulse_interp_crop_limited),merge_Te_prof_multipulse_interp_crop_limited/eV_to_K],axis=0),12000*np.ones_like(merge_Te_prof_multipulse_interp_crop_limited)],axis=0)	# K
						area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))	# m^2
						ionisation_potential = 13.6	# eV
						electron_charge = 1.60217662e-19	# coulombs
						upstream_electron_particle_flux = np.sum(area * merge_ne_prof_multipulse_interp_crop_limited * 10000,axis=-1)* 1e20	# #/s
						heat_inflow_upstream = np.sum(area * merge_ne_prof_multipulse_interp_crop_limited * 10000*(ionisation_potential+ T_Hp*eV_to_K +merge_Te_prof_multipulse_interp_crop_limited)/J_to_eV,axis=-1)* 1e20	# W
						sound_speed = (merge_Te_prof_multipulse_interp_crop_limited/eV_to_K*boltzmann_constant_J/(hydrogen_mass))**0.5	# m/s
						# sheat_transmission_factor = 7.7	# from T W Morgan et al 2014
						# heat_flux_target = np.sum(area *merge_ne_prof_multipulse_interp_crop_limited*1e20*sound_speed *(ionisation_potential/J_to_eV+sheat_transmission_factor*merge_Te_prof_multipulse_interp_crop_limited/eV_to_K*boltzmann_constant_J),axis=1)	# W
						target_electron_particle_flux = 0.5*merge_ne_prof_multipulse_interp_crop_limited*1e20*(boltzmann_constant_J*(1+5/3)*merge_Te_prof_multipulse_interp_crop_limited/(eV_to_K*hydrogen_mass))**0.5	# #/s
						heat_flux_target = np.sum(area*target_electron_particle_flux*( (5.03*merge_Te_prof_multipulse_interp_crop_limited+14.5)*boltzmann_constant_J/eV_to_K ),axis=-1)	# from Gijs paper 2020, W
						target_electron_particle_flux = np.sum(area*target_electron_particle_flux,axis=-1)
						neat_background_heating = heat_inflow_upstream - heat_flux_target
						if neat_background_heating<0:
							neat_background_heating=0
						volume = (0.351+target_OES_distance/1000)*np.pi*(0.4**2)	# m^2
						H2_mass = target_chamber_pressure*2*hydrogen_mass/(boltzmann_constant_J*300)*volume	# kg
						heat_capacity_H2 = 14.5	# kJ/(kg K)
						feed_H2_kg = feed_rate_SLM/1000*101325/(boltzmann_constant_J*300)*2*hydrogen_mass/60	# kg/s
						temp_increase = neat_background_heating/(heat_capacity_H2*(feed_H2_kg+1e-7))	# let's say that 0.1 SLM from the heating chamber with no seeding
						max_nH2_from_pressure_all = target_chamber_pressure/(boltzmann_constant_J*(273+20+temp_increase))
						max_nH2_from_pressure_all = 10*max_nH2_from_pressure_all	# as a safety factor
						max_nH2_from_pressure = np.ones_like(merge_Te_prof_multipulse_interp_crop_limited)
						max_nH2_from_pressure[merge_Te_prof_multipulse_interp_crop_limited>(273+20+temp_increase)*eV_to_K]=max_nH2_from_pressure_all*(273+20+temp_increase)*eV_to_K/merge_Te_prof_multipulse_interp_crop_limited[merge_Te_prof_multipulse_interp_crop_limited>(273+20+temp_increase)*eV_to_K]


						class calc_stuff_output:
							def __init__(self, my_time_pos,my_r_pos, results):
								self.my_time_pos = my_time_pos
								self.my_r_pos = my_r_pos
								self.results = results

						def calc_stuff_atomic_restricted_high_n(arg):
							index = arg[0]
							my_time_pos = arg[1]
							my_r_pos = arg[2]
							pass_index = arg[3]
							guess = arg[4]
							ionization_length_H = arg[5]
							ionization_length_Hm = arg[6]
							ionization_length_H2 = arg[7]
							ionization_length_H2p = arg[8]

							print(my_r_pos)

							inverted_profiles_crop_restrict = inverted_profiles_crop[n_list_all - 4, my_r_pos].flatten()
							inverted_profiles_crop_sigma_restrict = inverted_profiles_sigma_crop[n_list_all - 4, my_r_pos].flatten()
							inverted_profiles_crop_sigma_restrict[np.logical_not(np.isfinite(np.log(inverted_profiles_crop_restrict)))] = np.nanmax(inverted_profiles_crop_sigma_restrict)*10
							inverted_profiles_crop_restrict[np.logical_not(np.isfinite(np.log(inverted_profiles_crop_restrict)))]=np.nanmin(inverted_profiles_crop_restrict[np.isfinite(np.log(inverted_profiles_crop_restrict))])
							inverted_profiles_crop_sigma_restrict[np.isnan(inverted_profiles_crop_sigma_restrict)]=np.nanmax(inverted_profiles_crop_sigma_restrict)
							inverted_profiles_crop_sigma_restrict[inverted_profiles_crop_sigma_restrict==0]=np.nanmax(inverted_profiles_crop_sigma_restrict)
							merge_ne_prof_multipulse_interp_crop_limited_restrict = merge_ne_prof_multipulse_interp_crop_limited[ my_r_pos]
							merge_dne_prof_multipulse_interp_crop_limited_restrict = merge_dne_prof_multipulse_interp_crop_limited[my_r_pos]
							merge_Te_prof_multipulse_interp_crop_limited_restrict = merge_Te_prof_multipulse_interp_crop_limited[my_r_pos]
							merge_dTe_prof_multipulse_interp_crop_limited_restrict = merge_dTe_prof_multipulse_interp_crop_limited[my_r_pos]
							recombination_restrict = recombination[n_list_all - 4, my_r_pos].flatten()
							excitation_restrict = excitation[n_list_all - 4, my_r_pos].flatten()

							if (merge_ne_prof_multipulse_interp_crop_limited_restrict == 0 or merge_Te_prof_multipulse_interp_crop_limited_restrict == 0):
								results = [0, 0, 0, 0, 0, 0,0]
								output = calc_stuff_output(my_time_pos,my_r_pos, results)
								return output

							if False:
								T_Hp = 12000	# K
								T_Hm = 12000	# K
								T_H2p = 5000	# K
							else:
								T_Hp = min(max(1000,merge_Te_prof_multipulse_interp_crop_limited_restrict/eV_to_K),12000)	# K
								T_Hm = 5000	# K
								T_H2p = 5000	# K
							ne = merge_ne_prof_multipulse_interp_crop_limited_restrict * 1e20
							multiplicative_factor = energy_difference[n_list_all - 4] * einstein_coeff[n_list_all - 4] / J_to_eV

							def fit_Yacora_pop_coeff_prelim(merge_ne_prof_multipulse_interp_crop_limited_restrict,merge_Te_prof_multipulse_interp_crop_limited_restrict,T_Hp,T_Hm,T_H2p,ne,multiplicative_factor,excitation_restrict,recombination_restrict):
								def calculated_emission(n_list_all, nH_ne,nHp_ne):
									total = np.array([(excitation_restrict *  nH_ne /multiplicative_factor).astype('float')])
									total += np.array([(recombination_restrict * nHp_ne /multiplicative_factor).astype('float')])
									# total = nH_ne*From_H_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne,nHp_ne*ne,nH_ne*ne]],n_list_all)
									# total += nHp_ne*From_Hp_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne]],n_list_all)
									total = total[0]*(ne ** 2)*multiplicative_factor
									return np.log(total)
								return calculated_emission
							guess_prelim=[min_nH_ne,1]
							bds_prelim = [[min_nH_ne, min_nHp_ne],
								   [max_n_neutrals_ne, max_nHp_ne]]

							try:
								# from Verhaegh2019a: effect of molecules should not be relevant for Balmer n>=5
								n_weights_actual = np.array(np.log(inverted_profiles_crop_sigma_restrict))*n_weights
								fit = curve_fit(fit_Yacora_pop_coeff_prelim(merge_ne_prof_multipulse_interp_crop_limited_restrict,
									merge_Te_prof_multipulse_interp_crop_limited_restrict,T_Hp,T_Hm,T_H2p, ne, multiplicative_factor,excitation_restrict,recombination_restrict),
									n_list_all, np.log(inverted_profiles_crop_restrict), p0=guess_prelim, bounds=bds_prelim,sigma=n_weights_actual, maxfev=1000000)
							except Exception as e:
								print('fit at time index %.3g, radious index %.3g failed for reason %s' %(my_time_pos,my_r_pos,e))
								fit = [guess_prelim,np.ones((len(guess_prelim),len(guess_prelim)))]

							nH_ne, nHp_ne,nHm_ne, nH2_ne, nH2p_ne, nH3p_ne = [*fit[0],0,0,0,0]
							nH_ne_sigma, nHp_ne_sigma,nHm_ne_sigma, nH2_ne_sigma, nH2p_ne_sigma, nH3p_ne_sigma = [*np.sqrt(np.diag(fit[1])),1,1,1,1]
							residuals = np.sum(((np.log(inverted_profiles_crop_restrict) - fit_Yacora_pop_coeff_prelim(merge_ne_prof_multipulse_interp_crop_limited_restrict,merge_Te_prof_multipulse_interp_crop_limited_restrict,T_Hp,T_Hm,T_H2p,ne,multiplicative_factor,excitation_restrict,recombination_restrict)(n_list_all, *fit[0]))) ** 2)

							if True:
								sample_radious=[]
								for radious in [0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.009, 0.012]:	#radious in m that I want to take a better look at
									sample_radious.append(int((np.abs(r_crop - radious)).argmin()))
								sample_radious = np.unique(sample_radious)
								if my_r_pos in sample_radious:
									tm.sleep(np.random.random()*10)
									plt.figure()
									# nH_ne, nHp_ne, nH2p_ne = fit[0]
									# nHp_ne = 1 + nHm_ne - nH2p_ne
									plt.errorbar(n_list_all,inverted_profiles_crop_restrict,yerr=n_weights_actual,label='OES')
									plt.plot(n_list_all,(excitation_restrict *  nH_ne).astype('float')*(ne ** 2),label='direct excitation\nH(q) + e- → H(p>q) + e- (ADAS coeff)')
									plt.plot(n_list_all,(recombination_restrict * nHp_ne).astype('float')*(ne ** 2),label='recombination\nH+ + e- → H(p) + hν\nH+ + 2e- → H(p) + e- (ADAS coeff)')
									# plt.plot(n_list_all,nH_ne*From_H_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne,nHp_ne*ne,nH_ne*ne]],n_list_all)[0]*(ne ** 2)*multiplicative_factor,label='direct excitation\nH(q) + e- → H(p>q) + e-')
									# plt.plot(n_list_all,nHp_ne*From_Hp_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne]],n_list_all)[0]*(ne ** 2)*multiplicative_factor,label='recombination\nH+ + e- → H(p) + hν\nH+ + 2e- → H(p) + e-')
									plt.plot(n_list_all,nHm_ne*( From_Hn_with_Hp_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,T_Hp,T_Hm,ne,nHp_ne*ne]],n_list_all) )[0]*(ne ** 2)*multiplicative_factor,label='H+ mutual neutralisation\nH+ + H- → H(p) + H')
									plt.plot(n_list_all,nHm_ne*( From_Hn_with_H2p_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,T_H2p,T_Hm,ne,nH2p_ne*ne]],n_list_all) )[0]*(ne ** 2)*multiplicative_factor,label='H2+ mutual neutralisation\nH2+ + H- → H(p) + H2')
									plt.plot(n_list_all,nH2_ne*From_H2_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne]],n_list_all)[0]*(ne ** 2)*multiplicative_factor,label='H2 dissociation\nH2 + e- → H(p) + H(1) + e-')
									plt.plot(n_list_all,nH2p_ne*From_H2p_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne]],n_list_all)[0]*(ne ** 2)*multiplicative_factor,label='H2+ dissociation\nH2+ + e- → H(p) + H+ + e-')
									plt.plot(n_list_all,nH3p_ne*From_H3p_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne]],n_list_all)[0]*(ne ** 2)*multiplicative_factor,label='H3+ dissociation\nH3+ + e- → H(p) + H2')
									plt.plot(n_list_all,np.exp(fit_Yacora_pop_coeff_prelim(merge_ne_prof_multipulse_interp_crop_limited_restrict,merge_Te_prof_multipulse_interp_crop_limited_restrict,T_Hp,T_Hm,T_H2p,ne,multiplicative_factor,excitation_restrict,recombination_restrict)(n_list_all, *fit[0])),'--',label='excitation + recombination')
									# plt.plot(n_list_all,np.exp(fit_Yacora_pop_coeff_prelim(merge_ne_prof_multipulse_interp_crop_limited_restrict,merge_Te_prof_multipulse_interp_crop_limited_restrict,T_Hp,T_Hm,T_H2p,ne,multiplicative_factor,excitation_restrict,recombination_restrict)(n_list_all, nH_ne,nHp_ne, *np.zeros_like(fit[0][2:]))),'+',label='excitation + recombination')
									plt.semilogy()
									plt.legend(loc='best', fontsize='xx-small')
									plt.ylim(np.min(inverted_profiles_crop_restrict)/2,np.max(inverted_profiles_crop_restrict)*2)
									plt.title('SS lines '+str(n_list_all)+' weights '+str(n_weights)+'\nlocation [r]'+ ' [%.3g mm]' % (1000*r_crop[my_r_pos]) +' , [TS Te, TS ne] '+ ' [%.3g eV, ' % merge_Te_prof_multipulse_interp_crop_limited_restrict + '%.3g #10^20/m^3]' % merge_ne_prof_multipulse_interp_crop_limited_restrict +'\nfit [nH/ne, nH+/ne, nH-/ne, nH2/ne, nH2+/ne, nH3+/ne] , '+ '[%.3g, ' % nH_ne + '%.3g, ' % nHp_ne+ '%.3g, ' % nHm_ne+ '%.3g, ' % nH2_ne+ '%.3g, ' % nH2p_ne+ '%.3g]' % nH3p_ne +'\nH ionization length %.3gm' % ionization_length_H[my_r_pos] +'\nH2 ionization length %.3gm' % ionization_length_H2[my_r_pos])
									plt.xlabel('Exctited state n')
									plt.ylabel('Line emissivity n->2 [W m^-3]')
									# plt.pause(0.01)
									save_done = 0
									save_index=1
									try:	# All this trie are because of a known issues when saving lots of figure inside of multiprocessor
										plt.savefig(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_SS_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps', bbox_inches='tight')
									except Exception as e:
										print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_SS_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(1)+' failed. Reason %s' % e)
										while save_done==0 and save_index<100:
											try:
												tm.sleep(np.random.random()**2)
												plt.savefig(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_SS_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps', bbox_inches='tight')
												print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_SS_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(save_index+1)+' successfull')
												save_done=1
											except Exception as e:
												# print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_SS_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(save_index+1)+' failed. Reason %s' % e)
												save_index+=1
									plt.close('all')


							results = [nH_ne, nHp_ne, nHm_ne, nH2_ne, nH2p_ne, nH3p_ne,residuals]

							output = calc_stuff_output(my_time_pos,my_r_pos, results)
							return output


						def calc_stuff_atomic_restricted_high_n_mixed(arg):
							index = arg[0]
							my_time_pos = arg[1]
							my_r_pos = arg[2]
							pass_index = arg[3]
							guess = arg[4]
							ionization_length_H = arg[5]
							ionization_length_Hm = arg[6]
							ionization_length_H2 = arg[7]
							ionization_length_H2p = arg[8]

							print(my_r_pos)

							inverted_profiles_crop_restrict = inverted_profiles_crop[n_list_all - 4, my_r_pos].flatten()
							inverted_profiles_crop_sigma_restrict = inverted_profiles_sigma_crop[n_list_all - 4, my_r_pos].flatten()
							inverted_profiles_crop_sigma_restrict[np.logical_not(np.isfinite(np.log(inverted_profiles_crop_restrict)))] = np.nanmax(inverted_profiles_crop_sigma_restrict)*10
							inverted_profiles_crop_restrict[np.logical_not(np.isfinite(np.log(inverted_profiles_crop_restrict)))]=np.nanmin(inverted_profiles_crop_restrict[np.isfinite(np.log(inverted_profiles_crop_restrict))])
							inverted_profiles_crop_sigma_restrict[np.isnan(inverted_profiles_crop_sigma_restrict)]=np.nanmax(inverted_profiles_crop_sigma_restrict)
							inverted_profiles_crop_sigma_restrict[inverted_profiles_crop_sigma_restrict==0]=np.nanmax(inverted_profiles_crop_sigma_restrict)
							merge_ne_prof_multipulse_interp_crop_limited_restrict = merge_ne_prof_multipulse_interp_crop_limited[ my_r_pos]
							merge_dne_prof_multipulse_interp_crop_limited_restrict = merge_dne_prof_multipulse_interp_crop_limited[my_r_pos]
							merge_Te_prof_multipulse_interp_crop_limited_restrict = merge_Te_prof_multipulse_interp_crop_limited[my_r_pos]
							merge_dTe_prof_multipulse_interp_crop_limited_restrict = merge_dTe_prof_multipulse_interp_crop_limited[my_r_pos]
							recombination_restrict = recombination[n_list_all - 4, my_r_pos].flatten()
							excitation_restrict = excitation[n_list_all - 4, my_r_pos].flatten()

							if (merge_ne_prof_multipulse_interp_crop_limited_restrict == 0 or merge_Te_prof_multipulse_interp_crop_limited_restrict == 0):
								results = [0, 0, 0, 0, 0, 0,0]
								output = calc_stuff_output(my_time_pos,my_r_pos, results)
								return output

							if False:
								T_Hp = 12000	# K
								T_Hm = 12000	# K
								T_H2p = 5000	# K
							else:
								T_Hp = min(max(1000,merge_Te_prof_multipulse_interp_crop_limited_restrict/eV_to_K),12000)	# K
								T_Hm = 5000	# K
								T_H2p = 5000	# K
							ne = merge_ne_prof_multipulse_interp_crop_limited_restrict * 1e20
							multiplicative_factor = energy_difference[n_list_all - 4] * einstein_coeff[n_list_all - 4] / J_to_eV



							TS_steps = 5
							to_find_steps = 33
							# probability = np.ones((TS_steps,TS_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps))
							# guessed_values = np.ones((TS_steps,TS_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps))
							# calculated_emission = np.ones((TS_steps,TS_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,len(n_list_all)))
							Te_values = merge_Te_prof_multipulse_interp_crop_limited_restrict+np.linspace(-1,+1,TS_steps)*merge_dTe_prof_multipulse_interp_crop_limited_restrict
							Te_values = np.ones((TS_steps,TS_steps))*Te_values
							Te_values[Te_values<=0]=300*eV_to_K
							Te_probs = gauss(np.linspace(-1,+1,TS_steps),1/((2*np.pi)**0.5),1,0)
							Te_probs = Te_probs/np.sum(Te_probs)
							Te_probs = np.ones((TS_steps,TS_steps))*Te_probs
							ne_values = ne+np.linspace(-1,+1,TS_steps)*merge_dne_prof_multipulse_interp_crop_limited_restrict*1e20
							ne_values = (np.ones((TS_steps,TS_steps))*ne_values).T
							ne_values[ne_values<=0]=0.001*1e20
							ne_probs = gauss(np.linspace(-1,+1,TS_steps),1/((2*np.pi)**0.5),1,0)
							ne_probs = ne_probs/np.sum(ne_probs)
							ne_probs = (np.ones((TS_steps,TS_steps))*ne_probs).T

							excitation_internal = []
							for isel in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
								temp = read_adf15(pecfile, isel, Te_values.flatten(),(ne_values * 10 ** (- 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
								temp[np.isnan(temp)] = 0
								# temp = temp.reshape((np.shape(Te_values)))
								excitation_internal.append(temp)
							excitation_internal = np.array(excitation_internal)  # in # photons cm^-3 s^-1
							excitation_internal = (excitation_internal.T * (10 ** -6) * (energy_difference / J_to_eV))  # in W m^-3 / (# / m^3)**2
							excitation_internal = (excitation_internal[:,n_list_all - 4] /multiplicative_factor)  # in m^-3 / (# / m^3)**2

							recombination_internal = []
							for isel in [20, 21, 22, 23, 24, 25, 26, 27, 28]:
								temp = read_adf15(pecfile, isel, Te_values.flatten(),(ne_values * 10 ** (- 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
								temp[np.isnan(temp)] = 0
								# temp = temp.reshape((np.shape(Te_values)))
								recombination_internal.append(temp)
							recombination_internal = np.array(recombination_internal)  # in # photons cm^-3 s^-1 / (# cm^-3)**2
							recombination_internal = (recombination_internal.T * (10 ** -6) * (energy_difference / J_to_eV))  # in W m^-3 / (# / m^3)**2
							recombination_internal = (recombination_internal[:,n_list_all - 4] /multiplicative_factor)  # in m^-3 / (# / m^3)**2



							def fit_Yacora_pop_coeff_prelim(merge_ne_prof_multipulse_interp_crop_limited_restrict,merge_Te_prof_multipulse_interp_crop_limited_restrict,T_Hp,T_Hm,T_H2p,ne,multiplicative_factor,excitation_restrict,recombination_restrict):
								def calculated_emission(n_list_all, nH_ne,nHp_ne):
									total = (excitation_internal *  nH_ne ).astype('float')
									total += (recombination_internal * nHp_ne).astype('float')
									# total = nH_ne*From_H_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne,nHp_ne*ne,nH_ne*ne]],n_list_all)
									# total += nHp_ne*From_Hp_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne]],n_list_all)
									total = (total.T*(ne_values.flatten() ** 2)).T*multiplicative_factor
									total = np.sum(total.T * Te_probs.flatten() * ne_probs.flatten(),axis=-1)
									return np.log(total)
								return calculated_emission
							guess_prelim=[1,min(max_nHp_ne,10)]
							bds_prelim = [[min_nH_ne, min_nHp_ne],
								   [max_n_neutrals_ne, max_nHp_ne]]

							try:
								n_weights_actual = np.array(np.log(inverted_profiles_crop_sigma_restrict))*n_weights
								fit = curve_fit(fit_Yacora_pop_coeff_prelim(merge_ne_prof_multipulse_interp_crop_limited_restrict,merge_Te_prof_multipulse_interp_crop_limited_restrict,T_Hp,T_Hm,T_H2p,ne,multiplicative_factor,excitation_restrict,recombination_restrict),
									n_list_all, np.log(inverted_profiles_crop_restrict), p0=guess_prelim, bounds=bds_prelim,sigma=n_weights_actual, maxfev=1000000)
							except Exception as e:
								print('fit at time index %.3g, radious index %.3g failed for reason %s' %(my_time_pos,my_r_pos,e))
								fit = [guess_prelim,np.ones((len(guess_prelim),len(guess_prelim)))]

							nH_ne, nHp_ne,nHm_ne, nH2_ne, nH2p_ne, nH3p_ne = [*fit[0],0,0,0,0]
							nH_ne_sigma, nHp_ne_sigma,nHm_ne_sigma, nH2_ne_sigma, nH2p_ne_sigma, nH3p_ne_sigma = [*np.sqrt(np.diag(fit[1])),1,1,1,1]
							residuals = np.sum(((np.log(inverted_profiles_crop_restrict) - fit_Yacora_pop_coeff_prelim(merge_ne_prof_multipulse_interp_crop_limited_restrict,merge_Te_prof_multipulse_interp_crop_limited_restrict,T_Hp,T_Hm,T_H2p,ne,multiplicative_factor,excitation_restrict,recombination_restrict)(n_list_all, *fit[0]))) ** 2)

							if True:
								sample_radious=[]
								for radious in [0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.009, 0.012]:	#radious in m that I want to take a better look at
									sample_radious.append(int((np.abs(r_crop - radious)).argmin()))
								sample_radious = np.unique(sample_radious)
								if my_r_pos in sample_radious:
									tm.sleep(np.random.random()*10)
									plt.figure()
									# nH_ne, nHp_ne, nH2p_ne = fit[0]
									# nHp_ne = 1 + nHm_ne - nH2p_ne
									plt.errorbar(n_list_all,inverted_profiles_crop_restrict,yerr=n_weights_actual,label='OES')
									plt.plot(n_list_all,(excitation_restrict *  nH_ne).astype('float')*(ne ** 2),label='direct excitation\nH(q) + e- → H(p>q) + e- (ADAS coeff)')
									plt.plot(n_list_all,(recombination_restrict * nHp_ne).astype('float')*(ne ** 2),label='recombination\nH+ + e- → H(p) + hν\nH+ + 2e- → H(p) + e- (ADAS coeff)')
									# plt.plot(n_list_all,nH_ne*From_H_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne,nHp_ne*ne,nH_ne*ne]],n_list_all)[0]*(ne ** 2)*multiplicative_factor,label='direct excitation\nH(q) + e- → H(p>q) + e-')
									# plt.plot(n_list_all,nHp_ne*From_Hp_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne]],n_list_all)[0]*(ne ** 2)*multiplicative_factor,label='recombination\nH+ + e- → H(p) + hν\nH+ + 2e- → H(p) + e-')
									plt.plot(n_list_all,nHm_ne*( From_Hn_with_Hp_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,T_Hp,T_Hm,ne,nHp_ne*ne]],n_list_all) )[0]*(ne ** 2)*multiplicative_factor,label='H+ mutual neutralisation\nH+ + H- → H(p) + H')
									plt.plot(n_list_all,nHm_ne*( From_Hn_with_H2p_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,T_H2p,T_Hm,ne,nH2p_ne*ne]],n_list_all) )[0]*(ne ** 2)*multiplicative_factor,label='H2+ mutual neutralisation\nH2+ + H- → H(p) + H2')
									plt.plot(n_list_all,nH2_ne*From_H2_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne]],n_list_all)[0]*(ne ** 2)*multiplicative_factor,label='H2 dissociation\nH2 + e- → H(p) + H(1) + e-')
									plt.plot(n_list_all,nH2p_ne*From_H2p_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne]],n_list_all)[0]*(ne ** 2)*multiplicative_factor,label='H2+ dissociation\nH2+ + e- → H(p) + H+ + e-')
									plt.plot(n_list_all,nH3p_ne*From_H3p_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne]],n_list_all)[0]*(ne ** 2)*multiplicative_factor,label='H3+ dissociation\nH3+ + e- → H(p) + H2')
									plt.plot(n_list_all,np.exp(fit_Yacora_pop_coeff_prelim(merge_ne_prof_multipulse_interp_crop_limited_restrict,merge_Te_prof_multipulse_interp_crop_limited_restrict,T_Hp,T_Hm,T_H2p,ne,multiplicative_factor,excitation_restrict,recombination_restrict)(n_list_all, *fit[0])),'--',label='excitation + recombination')
									# plt.plot(n_list_all,np.exp(fit_Yacora_pop_coeff_prelim(merge_ne_prof_multipulse_interp_crop_limited_restrict,merge_Te_prof_multipulse_interp_crop_limited_restrict,T_Hp,T_Hm,T_H2p,ne,multiplicative_factor,excitation_restrict,recombination_restrict)(n_list_all, nH_ne,nHp_ne, *np.zeros_like(fit[0][2:]))),'+',label='excitation + recombination')
									plt.semilogy()
									plt.legend(loc='best', fontsize='xx-small')
									plt.ylim(np.min(inverted_profiles_crop_restrict)/2,np.max(inverted_profiles_crop_restrict)*2)
									plt.title('SS lines '+str(n_list_all)+' weights '+str(n_weights)+'\nlocation [r]'+ ' [%.3g mm]' % (1000*r_crop[my_r_pos]) +' , [TS Te, TS ne] '+ ' [%.3g eV, ' % merge_Te_prof_multipulse_interp_crop_limited_restrict + '%.3g #10^20/m^3]' % merge_ne_prof_multipulse_interp_crop_limited_restrict +'\nfit [nH/ne, nH+/ne, nH-/ne, nH2/ne, nH2+/ne, nH3+/ne] , '+ '[%.3g, ' % nH_ne + '%.3g, ' % nHp_ne+ '%.3g, ' % nHm_ne+ '%.3g, ' % nH2_ne+ '%.3g, ' % nH2p_ne+ '%.3g]' % nH3p_ne +'\nH ionization length %.3gm' % ionization_length_H[my_r_pos] +'\nH2 ionization length %.3gm' % ionization_length_H2[my_r_pos])
									plt.xlabel('Exctited state n')
									plt.ylabel('Line emissivity n->2 [W m^-3]')
									# plt.pause(0.01)
									save_done = 0
									save_index=1
									try:	# All this trie are because of a known issues when saving lots of figure inside of multiprocessor
										plt.savefig(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_SS_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps', bbox_inches='tight')
									except Exception as e:
										print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_SS_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(1)+' failed. Reason %s' % e)
										while save_done==0 and save_index<100:
											try:
												tm.sleep(np.random.random()**2)
												plt.savefig(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_SS_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps', bbox_inches='tight')
												print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_SS_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(save_index+1)+' successfull')
												save_done=1
											except Exception as e:
												# print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_SS_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(save_index+1)+' failed. Reason %s' % e)
												save_index+=1
									plt.close('all')


							results = [nH_ne, nHp_ne, nHm_ne, nH2_ne, nH2p_ne, nH3p_ne,residuals]

							output = calc_stuff_output(my_time_pos,my_r_pos, results)
							return output


						def calc_stuff(arg):
							index = arg[0]
							my_time_pos = arg[1]
							my_r_pos = arg[2]
							pass_index = arg[3]
							guess = arg[4]
							ionization_length_H = arg[5]
							ionization_length_Hm = arg[6]
							ionization_length_H2 = arg[7]
							ionization_length_H2p = arg[8]

							inverted_profiles_crop_restrict = inverted_profiles_crop[n_list_all - 4, my_r_pos].flatten()
							inverted_profiles_crop_sigma_restrict = inverted_profiles_sigma_crop[n_list_all - 4, my_r_pos].flatten()
							merge_ne_prof_multipulse_interp_crop_limited_restrict = merge_ne_prof_multipulse_interp_crop_limited[ my_r_pos]
							merge_dne_prof_multipulse_interp_crop_limited_restrict = merge_dne_prof_multipulse_interp_crop_limited[my_r_pos]
							merge_Te_prof_multipulse_interp_crop_limited_restrict = merge_Te_prof_multipulse_interp_crop_limited[my_r_pos]
							merge_dTe_prof_multipulse_interp_crop_limited_restrict = merge_dTe_prof_multipulse_interp_crop_limited[my_r_pos]
							recombination_restrict = recombination[n_list_all - 4, my_r_pos].flatten()
							excitation_restrict = excitation[n_list_all - 4, my_r_pos].flatten()


							if (merge_ne_prof_multipulse_interp_crop_limited_restrict == 0 or merge_Te_prof_multipulse_interp_crop_limited_restrict == 0):
								results = [0, 0, 0, 0, 0, 0,0]
								output = calc_stuff_output(my_time_pos,my_r_pos, results)
								return output

							# if np.sum(inverted_profiles_crop_restrict == 0)>2:
							# 	continue

							if False:
								T_Hp = 12000	# K
								T_Hm = 12000	# K
								T_H2p = 5000	# K
							else:
								T_Hp = min(max(1000,merge_Te_prof_multipulse_interp_crop_limited_restrict/eV_to_K),12000)	# K
								T_Hm = 5000	# K
								T_H2p = 5000	# K
							ne = merge_ne_prof_multipulse_interp_crop_limited_restrict * 1e20
							multiplicative_factor = energy_difference[n_list_all - 4] * einstein_coeff[n_list_all - 4] / J_to_eV

							max_nH2_from_pressure_int = max_nH2_from_pressure_all
							if ionization_length_H[my_r_pos]<=0.5:
								max_nmolecules_ne_low_ionisation_length = 0.05
								# max_nH2_from_pressure_int = ne/10
							else:
								max_nmolecules_ne_low_ionisation_length = 0.15
								# max_nH2_from_pressure_int = max_nH2_from_pressure*1

							# if pass_index>1:
							# 	if ionization_length_H2[my_r_pos]<1e-6:
							# 		max_nH2_from_pressure_int = ne/100000
							if merge_Te_prof_multipulse_interp_crop_limited_restrict>(273+20+temp_increase)*eV_to_K:
								max_nH2_from_pressure_int = max_nH2_from_pressure_int*(273+20+temp_increase)*eV_to_K/merge_Te_prof_multipulse_interp_crop_limited_restrict

							TS_steps = 5
							to_find_steps = 33
							# probability = np.ones((TS_steps,TS_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps))
							# guessed_values = np.ones((TS_steps,TS_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps))
							# calculated_emission = np.ones((TS_steps,TS_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,len(n_list_all)))
							Te_values = merge_Te_prof_multipulse_interp_crop_limited_restrict+np.linspace(-1,+1,TS_steps)*merge_dTe_prof_multipulse_interp_crop_limited_restrict
							Te_values = np.ones((TS_steps,TS_steps))*Te_values
							Te_values[Te_values<=0]=0.001
							Te_probs = gauss(np.linspace(-1,+1,TS_steps),1/((2*np.pi)**0.5),1,0)
							Te_probs = Te_probs/np.sum(Te_probs)
							Te_probs = np.ones((TS_steps,TS_steps))*Te_probs
							ne_values = ne+np.linspace(-1,+1,TS_steps)*merge_dne_prof_multipulse_interp_crop_limited_restrict*1e20
							ne_values = (np.ones((TS_steps,TS_steps))*ne_values).T
							ne_values[ne_values<=0]=0.001*1e20
							ne_probs = gauss(np.linspace(-1,+1,TS_steps),1/((2*np.pi)**0.5),1,0)
							ne_probs = ne_probs/np.sum(ne_probs)
							ne_probs = (np.ones((TS_steps,TS_steps))*ne_probs).T
							nHp_ne = np.logspace(np.log10(max_nHp_ne),np.log10(max(min_nHp_ne,1e-6)),num=to_find_steps)
							nH_ne = np.logspace(np.log10(min(max_n_neutrals_ne,1e5)),np.log10(max(min_nH_ne,1e-6)),num=to_find_steps)
							nHpolecule_ne = np.logspace(np.log10(max_nmolecules_ne_low_ionisation_length),np.log10(max_nmolecules_ne_low_ionisation_length/1000),num=to_find_steps)
							nH2_ne = np.logspace(np.log10(max_nH2_from_pressure_int/ne),np.log10(max_nH2_from_pressure_int/10000/ne),num=to_find_steps)

							excitation_internal = []
							for isel in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
								temp = read_adf15(pecfile, isel, Te_values.flatten(),(ne_values * 10 ** (- 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
								temp[np.isnan(temp)] = 0
								# temp = temp.reshape((np.shape(Te_values)))
								excitation_internal.append(temp)
							excitation_internal = np.array(excitation_internal)  # in # photons cm^-3 s^-1
							excitation_internal = (excitation_internal.T * (10 ** -6) * (energy_difference / J_to_eV))  # in W m^-3 / (# / m^3)**2
							excitation_internal = (excitation_internal[:,n_list_all - 4] /multiplicative_factor)  # in m^-3 / (# / m^3)**2

							recombination_internal = []
							for isel in [20, 21, 22, 23, 24, 25, 26, 27, 28]:
								temp = read_adf15(pecfile, isel, Te_values.flatten(),(ne_values * 10 ** (- 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
								temp[np.isnan(temp)] = 0
								# temp = temp.reshape((np.shape(Te_values)))
								recombination_internal.append(temp)
							recombination_internal = np.array(recombination_internal)  # in # photons cm^-3 s^-1 / (# cm^-3)**2
							recombination_internal = (recombination_internal.T * (10 ** -6) * (energy_difference / J_to_eV))  # in W m^-3 / (# / m^3)**2
							recombination_internal = (recombination_internal[:,n_list_all - 4] /multiplicative_factor)  # in m^-3 / (# / m^3)**2

							# calculated_emission = np.ones((TS_steps,TS_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,len(n_list_all)))
							# for i1,nH_ne_value in enumerate(nH_ne):
							# 	total_nH_ne_value = (((excitation_internal[n_list_all - 4] *  nH_ne_value*np.ones((TS_steps,TS_steps)) ).T/multiplicative_factor).T).astype('float').T.reshape((TS_steps*TS_steps,len(n_list_all)))
							# 	nHp_ne_value = 1
							# 	# for i2,nHp_ne_value in enumerate(nHp_ne):
							# 	total_nHp_ne_value = (((recombination_internal[n_list_all - 4] * nHp_ne_value*np.ones((TS_steps,TS_steps)) ).T/multiplicative_factor).T).astype('float').T.reshape((TS_steps*TS_steps,len(n_list_all)))
							# 	for i3,nHm_ne_value in enumerate(nHpolecule_ne):
							# 		for i4,nH2_ne_value in enumerate(nHpolecule_ne):
							# 			total_nH2_ne_value = nH2_ne_value*From_H2_pop_coeff_full(np.array([Te_values.flatten(),ne_values.flatten()]).T,n_list_all)
							# 			for i5,nH2p_ne_value in enumerate(nHpolecule_ne):
							# 				total_nHm_ne_value = nHm_ne_value*( From_Hn_with_Hp_pop_coeff_full(np.array([Te_values.flatten(),T_Hp*np.ones_like(Te_values.flatten()),T_Hm*np.ones_like(Te_values.flatten()),ne_values.flatten()*1e20,nHp_ne_value*ne_values.flatten()*1e20]).T ,n_list_all) + From_Hn_with_H2p_pop_coeff_full(np.array([Te_values.flatten(),T_H2p*np.ones_like(Te_values.flatten()),T_Hm*np.ones_like(Te_values.flatten()),ne_values.flatten()*1e20,nH2p_ne_value*ne_values.flatten()*1e20]).T,n_list_all) )
							# 				total_nH2p_ne_value = nH2p_ne_value*From_H2p_pop_coeff_full(np.array([Te_values.flatten(),ne_values.flatten()*1e20]).T,n_list_all)
							# 				# for i6,nH3p_ne_value in enumerate(nHpolecule_ne):
							# 				# 	total_nH3p_ne_value = nH3p_ne_value*From_H3p_pop_coeff_full(np.array([Te_values.flatten(),ne_values.flatten()*1e20]).T,n_list_all)
							# 				total = total_nH_ne_value+total_nHp_ne_value+total_nH2_ne_value+total_nHm_ne_value+total_nH2p_ne_value# +total_nH3p_ne_value
							# 				total = (total.T*(Te_values.flatten() ** 2)).T*multiplicative_factor * 1e40
							# 				calculated_emission[:,:,i1,i3,i4,i5,:]=total.reshape((5,5,8))


							# calculated_emission = np.ones((TS_steps,TS_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,len(n_list_all)))
							calculated_emission_error = np.ones((to_find_steps,to_find_steps))
							for i2,nHp_ne_value in enumerate(nHp_ne):
							# total_nHp_ne_value = (((recombination_internal[n_list_all - 4] * nHp_ne_value*np.ones((TS_steps,TS_steps)) ).T/multiplicative_factor).T).astype('float').T.reshape((TS_steps*TS_steps,len(n_list_all)))
								total_nHp_ne_value = recombination_internal * nHp_ne_value
								# coeff_1 = From_H2p_pop_coeff_full(np.array([Te_values.flatten(),ne_values.flatten()]).T,n_list_all)
								# coeff_2 = From_H2_pop_coeff_full(np.array([Te_values.flatten(),ne_values.flatten()]).T,n_list_all)
								# coeff_3 = From_Hn_with_Hp_pop_coeff_full(np.array([Te_values.flatten(),T_Hp*np.ones_like(Te_values.flatten()),T_Hm*np.ones_like(Te_values.flatten()),ne_values.flatten(),nHp_ne_value*ne_values.flatten()]).T ,n_list_all)
								# for i5,nH2p_ne_value in enumerate(nHpolecule_ne):
								# 	total_nH2p_ne_value = nH2p_ne_value * coeff_1
								# 	coeff_4 = From_Hn_with_H2p_pop_coeff_full(np.array([Te_values.flatten(),T_H2p*np.ones_like(Te_values.flatten()),T_Hm*np.ones_like(Te_values.flatten()),ne_values.flatten(),nH2p_ne_value*ne_values.flatten()]).T,n_list_all)
								for i1,nH_ne_value in enumerate(nH_ne):
									# total_nH_ne_value = (((excitation_internal[n_list_all - 4] *  nH_ne_value*np.ones((TS_steps,TS_steps)) ).T/multiplicative_factor).T).astype('float').T.reshape((TS_steps*TS_steps,len(n_list_all)))
									total_nH_ne_value = excitation_internal *  nH_ne_value
									# for i3,nHm_ne_value in enumerate(nHpolecule_ne):
									# 	total_nHm_ne_value = nHm_ne_value*( coeff_3 + coeff_4 )
										# for i4,nH2_ne_value in enumerate(nH2_ne):
										# 	total_nH2_ne_value = nH2_ne_value * coeff_2
											# total = total_nH_ne_value+total_nHp_ne_value+total_nH2_ne_value+total_nHm_ne_value+total_nH2p_ne_value# +total_nH3p_ne_value
									total = total_nH_ne_value+total_nHp_ne_value
									total = (total.T*(ne_values.flatten() ** 2)).T*multiplicative_factor
									calculated_emission = total.reshape((TS_steps*TS_steps,len(n_list_all)))
									temp = np.sum(((calculated_emission - inverted_profiles_crop_restrict)/inverted_profiles_crop_sigma_restrict)**2,axis=-1)
									temp = np.sum(temp * Te_probs.flatten() * ne_probs.flatten())
									# calculated_emission_error[i1,i3,i4,i5]= temp
									calculated_emission_error[i1,i2]= temp

							# calculated_emission_error = np.sum(((calculated_emission - inverted_profiles_crop_restrict)/inverted_profiles_crop_sigma_restrict)**2,axis=-1)
							# calculated_emission_error = np.sum(calculated_emission_error.T * Te_probs,axis=-1)
							# calculated_emission_error = np.sum(calculated_emission_error * ne_probs,axis=-1).T

							index_best_fit = calculated_emission_error.argmin()
							# best_nH2p_ne_index = index_best_fit%to_find_steps
							best_nHp_ne_index = index_best_fit%to_find_steps
							# best_nH2_ne_index = (index_best_fit-best_nH2p_ne_index)//to_find_steps%to_find_steps
							# best_nHm_ne_index = ((index_best_fit-best_nH2p_ne_index)//to_find_steps -best_nH2_ne_index)//to_find_steps%to_find_steps
							# best_nH_ne_index = (((index_best_fit-best_nH2p_ne_index)//to_find_steps -best_nH2_ne_index)//to_find_steps - best_nHm_ne_index )//to_find_steps%to_find_steps
							best_nH_ne_index = (index_best_fit-best_nHp_ne_index)//to_find_steps%to_find_steps

							# best_nH2p_ne_value = nHpolecule_ne[best_nH2p_ne_index]
							# best_nH2_ne_value = nH2_ne[best_nH2_ne_index]
							# best_nHm_ne_value = nHpolecule_ne[best_nHm_ne_index]
							best_nHp_ne_value = nHp_ne[best_nHp_ne_index]
							best_nH_ne_value = nH_ne[best_nH_ne_index]

							# print('best_nH_ne_value %.3g,best_nHm_ne_value %.3g,best_nH2_ne_value %.3g,best_nH2p_ne_value %.3g' %(best_nH_ne_value,best_nHm_ne_value,best_nH2_ne_value,best_nH2p_ne_value))

							to_find_steps_2 = 33
							nH_ne = np.logspace(np.log10(min(best_nH_ne_value*50,max_n_neutrals_ne)),np.log10(best_nH_ne_value/10),num=to_find_steps_2)
							nHp_ne = np.logspace(np.log10(min(best_nHp_ne_value*50,max_nHp_ne)),np.log10(max(best_nHp_ne_value/10,max(min_nHp_ne,1e-6))),num=to_find_steps_2)
							# nHm_ne = np.logspace(np.log10(min(best_nHm_ne_value*50,1)),np.log10(best_nHm_ne_value/10),num=to_find_steps_2)
							# nH2_ne = np.logspace(np.log10(min(best_nH2_ne_value*50,max_nH2_from_pressure_int/ne)),np.log10(best_nH2_ne_value/10),num=to_find_steps_2)
							# nH2p_ne = np.logspace(np.log10(min(best_nHm_ne_value*50,1)),np.log10(best_nHm_ne_value/10),num=to_find_steps_2)



							# calculated_emission = np.ones((TS_steps,TS_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,len(n_list_all)))
							calculated_emission_error = np.ones((to_find_steps,to_find_steps))
							for i2,nHp_ne_value in enumerate(nHp_ne):
							# total_nHp_ne_value = (((recombination_internal[n_list_all - 4] * nHp_ne_value*np.ones((TS_steps,TS_steps)) ).T/multiplicative_factor).T).astype('float').T.reshape((TS_steps*TS_steps,len(n_list_all)))
								total_nHp_ne_value = recombination_internal * nHp_ne_value
								# coeff_1 = From_H2p_pop_coeff_full(np.array([Te_values.flatten(),ne_values.flatten()]).T,n_list_all)
								# coeff_2 = From_H2_pop_coeff_full(np.array([Te_values.flatten(),ne_values.flatten()]).T,n_list_all)
								# coeff_3 = From_Hn_with_Hp_pop_coeff_full(np.array([Te_values.flatten(),T_Hp*np.ones_like(Te_values.flatten()),T_Hm*np.ones_like(Te_values.flatten()),ne_values.flatten(),nHp_ne_value*ne_values.flatten()]).T ,n_list_all)
								# for i5,nH2p_ne_value in enumerate(nHpolecule_ne):
								# 	total_nH2p_ne_value = nH2p_ne_value * coeff_1
								# 	coeff_4 = From_Hn_with_H2p_pop_coeff_full(np.array([Te_values.flatten(),T_H2p*np.ones_like(Te_values.flatten()),T_Hm*np.ones_like(Te_values.flatten()),ne_values.flatten(),nH2p_ne_value*ne_values.flatten()]).T,n_list_all)
								for i1,nH_ne_value in enumerate(nH_ne):
									# total_nH_ne_value = (((excitation_internal[n_list_all - 4] *  nH_ne_value*np.ones((TS_steps,TS_steps)) ).T/multiplicative_factor).T).astype('float').T.reshape((TS_steps*TS_steps,len(n_list_all)))
									total_nH_ne_value = excitation_internal *  nH_ne_value
									# for i3,nHm_ne_value in enumerate(nHpolecule_ne):
									# 	total_nHm_ne_value = nHm_ne_value*( coeff_3 + coeff_4 )
										# for i4,nH2_ne_value in enumerate(nH2_ne):
										# 	total_nH2_ne_value = nH2_ne_value * coeff_2
											# total = total_nH_ne_value+total_nHp_ne_value+total_nH2_ne_value+total_nHm_ne_value+total_nH2p_ne_value# +total_nH3p_ne_value
									total = total_nH_ne_value+total_nHp_ne_value
									total = (total.T*(ne_values.flatten() ** 2)).T*multiplicative_factor
									calculated_emission = total.reshape((TS_steps*TS_steps,len(n_list_all)))
									temp = np.sum(((calculated_emission - inverted_profiles_crop_restrict)/inverted_profiles_crop_sigma_restrict)**2,axis=-1)
									temp = np.sum(temp * Te_probs.flatten() * ne_probs.flatten())
									# calculated_emission_error[i1,i3,i4,i5]= temp
									calculated_emission_error[i1,i2]= temp



							# calculated_emission_error = np.sum(((calculated_emission - inverted_profiles_crop_restrict)/inverted_profiles_crop_sigma_restrict)**2,axis=-1)
							# calculated_emission_error = np.sum(calculated_emission_error.T * Te_probs,axis=-1)
							# calculated_emission_error = np.sum(calculated_emission_error * ne_probs,axis=-1).T

							index_best_fit = calculated_emission_error.argmin()
							# best_nH2p_ne_index = index_best_fit%to_find_steps_2
							best_nHp_ne_index = index_best_fit%to_find_steps_2
							# best_nH2_ne_index = (index_best_fit-best_nH2p_ne_index)//to_find_steps_2%to_find_steps_2
							# best_nHm_ne_index = ((index_best_fit-best_nH2p_ne_index)//to_find_steps_2 -best_nH2_ne_index)//to_find_steps_2%to_find_steps_2
							# best_nH_ne_index = (((index_best_fit-best_nH2p_ne_index)//to_find_steps_2 -best_nH2_ne_index)//to_find_steps_2 - best_nHm_ne_index )//to_find_steps_2%to_find_steps_2
							best_nH_ne_index = (index_best_fit-best_nHp_ne_index)//to_find_steps_2%to_find_steps_2

							# best_nH2p_ne_value = nHpolecule_ne[best_nH2p_ne_index]
							# best_nH2_ne_value = nH2_ne[best_nH2_ne_index]
							# best_nHm_ne_value = nHpolecule_ne[best_nHm_ne_index]
							best_nHp_ne_value = nHp_ne[best_nHp_ne_index]
							best_nH_ne_value = nH_ne[best_nH_ne_index]

							def fit_Yacora_pop_coeff(merge_ne_prof_multipulse_interp_crop_limited_restrict,merge_Te_prof_multipulse_interp_crop_limited_restrict,T_Hp,T_Hm,T_H2p,ne,multiplicative_factor,excitation_restrict,recombination_restrict):
								def calculated_emission(n_list_all, nH_ne,nHp_ne,nHm_ne,nH2_ne,nH2p_ne,nH3p_ne):
									# nHp_ne=1+nHm_ne-nH2p_ne
									total = np.array([(excitation_restrict *  nH_ne /multiplicative_factor).astype('float')])
									total += np.array([(recombination_restrict * nHp_ne /multiplicative_factor).astype('float')])
									# total = nH_ne*From_H_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne,nHp_ne*ne,nH_ne*ne]],n_list_all)
									# total += nHp_ne*From_Hp_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne]],n_list_all)
									# total += nHm_ne*( From_Hn_with_Hp_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,T_Hp,T_Hm,ne,nHp_ne*ne]],n_list_all) + From_Hn_with_H2p_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,T_H2p,T_Hm,ne,nH2p_ne*ne]],n_list_all) )
									# total += nH2_ne*From_H2_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne]],n_list_all)
									# total += nH2p_ne*From_H2p_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne]],n_list_all)
									# total += nH3p_ne*From_H3p_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne]],n_list_all)
									total = total[0]*(ne ** 2)*multiplicative_factor
									# print(total)
									return np.log(total)
								return calculated_emission


							if True:
								sample_radious=[]
								for radious in [0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.009, 0.012]:	#radious in m that I want to take a better look at
									sample_radious.append(int((np.abs(r_crop - radious)).argmin()))
								sample_radious = np.unique(sample_radious)
								if my_r_pos in sample_radious:
									# print('best_nH_ne_value %.3g,best_nHm_ne_value %.3g,best_nH2_ne_value %.3g,best_nH2p_ne_value %.3g' %(best_nH_ne_value,best_nHm_ne_value,best_nH2_ne_value,best_nH2p_ne_value))
									tm.sleep(np.random.random()*10)

									fig, ax = plt.subplots( 2,1,figsize=(10, 25), squeeze=True)
									fig.suptitle('SS best_nH_ne_value %.3g,best_nHp_ne_value %.3g' %(best_nH_ne_value*ne,best_nHp_ne_value*ne) +'\nlines '+str(n_list_all)+'\nlocation [r]'+ ' [%.3g mm]' % (1000*r_crop[my_r_pos]) +' , [TS Te, TS ne] '+ ' [%.3g eV, ' % merge_Te_prof_multipulse_interp_crop_limited_restrict + '%.3g #10^20/m^3]' % merge_ne_prof_multipulse_interp_crop_limited_restrict +'\nfit [nH/ne, nH+/ne] , '+ '[%.3g, ' % best_nH_ne_value + '%.3g] ' % best_nHp_ne_value +'\nH ionization length %.3gm' % ionization_length_H[my_r_pos] +'\nH2 ionization length %.3gm' % ionization_length_H2[my_r_pos])
									plot_index = 0
									im = ax[plot_index].plot(nH_ne*ne, calculated_emission_error[:,best_nHp_ne_index]);
									im = ax[plot_index].plot([best_nH_ne_value*ne,best_nH_ne_value*ne], np.sort(calculated_emission_error[:,best_nHp_ne_index])[[0,-1]],'k--');
									ax[plot_index].set_title('nH')
									ax[plot_index].set_ylabel('fitting error')
									ax[plot_index].set_xlabel('density [#/m^3]')
									ax[plot_index].set_xscale('log')
									ax[plot_index].set_yscale('log')
									plot_index += 1
									im = ax[plot_index].plot(nHp_ne*ne, calculated_emission_error[best_nH_ne_index,:]);
									im = ax[plot_index].plot([best_nHp_ne_value*ne,best_nHp_ne_value*ne], np.sort(calculated_emission_error[best_nH_ne_index,:])[[0,-1]],'k--');
									ax[plot_index].set_title('nH+')
									ax[plot_index].set_ylabel('fitting error')
									ax[plot_index].set_xlabel('density [#/m^3]')
									ax[plot_index].set_xscale('log')
									ax[plot_index].set_yscale('log')
									# plot_index += 1
									# im = ax[plot_index].plot(nH2_ne*ne, calculated_emission_error[best_nH_ne_index,best_nHm_ne_index,:,best_nH2p_ne_index]);
									# im = ax[plot_index].plot([best_nH2_ne_value*ne,best_nH2_ne_value*ne], np.sort(calculated_emission_error[best_nH_ne_index,best_nHm_ne_index,:,best_nH2p_ne_index])[[0,-1]],'k--');
									# ax[plot_index].set_title('nH2')
									# ax[plot_index].set_ylabel('fitting error')
									# ax[plot_index].set_xlabel('density [#/m^3]')
									# ax[plot_index].set_xscale('log')
									# ax[plot_index].set_yscale('log')
									# plot_index += 1
									# im = ax[plot_index].plot(nH2p_ne*ne, calculated_emission_error[best_nH_ne_index,best_nHm_ne_index,best_nH2_ne_index,:]);
									# im = ax[plot_index].plot([best_nH2p_ne_value*ne,best_nH2p_ne_value*ne], np.sort(calculated_emission_error[best_nH_ne_index,best_nHm_ne_index,best_nH2_ne_index,:])[[0,-1]],'k--');
									# ax[plot_index].set_title('nH2+')
									# ax[plot_index].set_ylabel('fitting error')
									# ax[plot_index].set_xlabel('density [#/m^3]')
									# ax[plot_index].set_xscale('log')
									# ax[plot_index].set_yscale('log')
									# plt.pause(0.01)
									save_done = 0
									save_index=1
									try:	# All this trie are because of a known issues when saving lots of figure inside of multiprocessor
										plt.savefig(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_SS_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps', bbox_inches='tight')
									except Exception as e:
										print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_SS_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(1)+' failed. Reason %s' % e)
										while save_done==0 and save_index<100:
											try:
												tm.sleep(np.random.random()**2)
												plt.savefig(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_SS_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps', bbox_inches='tight')
												print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_SS_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(save_index+1)+' successfull')
												save_done=1
											except Exception as e:
												# print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_SS_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(save_index+1)+' failed. Reason %s' % e)
												save_index+=1
									plt.close('all')


							nH_ne, nHp_ne, nHm_ne, nH2_ne, nH2p_ne, nH3p_ne = best_nH_ne_value,best_nHp_ne_value,0,0,0,0
							nH_ne_sigma, nHp_ne_sigma, nHm_ne_sigma, nH2_ne_sigma, nH2p_ne_sigma, nH3p_ne_sigma = 1,1,1,1,1,1
							# residuals = np.sum(((np.log(inverted_profiles_crop_restrict) - fit_Yacora_pop_coeff(merge_ne_prof_multipulse_interp_crop_limited_restrict,merge_Te_prof_multipulse_interp_crop_limited_restrict,T_Hp,T_Hm,T_H2p,ne,multiplicative_factor,excitation_restrict,recombination_restrict)(n_list_all, *fit[0]))) ** 2)
							residuals = np.sum(((inverted_profiles_crop_restrict - np.exp(fit_Yacora_pop_coeff(merge_ne_prof_multipulse_interp_crop_limited_restrict,merge_Te_prof_multipulse_interp_crop_limited_restrict,T_Hp,T_Hm,T_H2p,ne,multiplicative_factor,excitation_restrict,recombination_restrict)(n_list_all, nH_ne,nHp_ne,nHm_ne,nH2_ne,nH2p_ne,nH3p_ne)))/inverted_profiles_crop_sigma_restrict) ** 2)

							if True:
								sample_radious=[]
								for radious in [0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.009, 0.012]:	#radious in m that I want to take a better look at
									sample_radious.append(int((np.abs(r_crop - radious)).argmin()))
								sample_radious = np.unique(sample_radious)
								if my_r_pos in sample_radious:
									tm.sleep(np.random.random()*10)
									plt.figure(figsize=(20, 10))
									# nH_ne, nHp_ne, nH2p_ne = fit[0]
									# nHp_ne = 1 + nHm_ne - nH2p_ne
									plt.errorbar(n_list_all,inverted_profiles_crop_restrict,yerr=100*inverted_profiles_crop_sigma_restrict,label='OES')
									plt.plot(n_list_all,(excitation_restrict *  nH_ne).astype('float')*(ne ** 2),label='direct excitation\nH(q) + e- → H(p>q) + e- (ADAS coeff)')
									plt.plot(n_list_all,(recombination_restrict * nHp_ne).astype('float')*(ne ** 2),label='recombination\nH+ + e- → H(p) + hν\nH+ + 2e- → H(p) + e- (ADAS coeff)')
									# plt.plot(n_list_all,nH_ne*From_H_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne,nHp_ne*ne,nH_ne*ne]],n_list_all)[0]*(ne ** 2)*multiplicative_factor,label='direct excitation\nH(q) + e- → H(p>q) + e-')
									# plt.plot(n_list_all,nHp_ne*From_Hp_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne]],n_list_all)[0]*(ne ** 2)*multiplicative_factor,label='recombination\nH+ + e- → H(p) + hν\nH+ + 2e- → H(p) + e-')
									plt.plot(n_list_all,nHm_ne*( From_Hn_with_Hp_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,T_Hp,T_Hm,ne,nHp_ne*ne]],n_list_all) )[0]*(ne ** 2)*multiplicative_factor,label='H+ mutual neutralisation\nH+ + H- → H(p) + H')
									plt.plot(n_list_all,nHm_ne*( From_Hn_with_H2p_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,T_H2p,T_Hm,ne,nH2p_ne*ne]],n_list_all) )[0]*(ne ** 2)*multiplicative_factor,label='H2+ mutual neutralisation\nH2+ + H- → H(p) + H2')
									plt.plot(n_list_all,nH2_ne*From_H2_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne]],n_list_all)[0]*(ne ** 2)*multiplicative_factor,label='H2 dissociation\nH2 + e- → H(p) + H(1) + e-')
									plt.plot(n_list_all,nH2p_ne*From_H2p_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne]],n_list_all)[0]*(ne ** 2)*multiplicative_factor,label='H2+ dissociation\nH2+ + e- → H(p) + H+ + e-')
									plt.plot(n_list_all,nH3p_ne*From_H3p_pop_coeff_full([[merge_Te_prof_multipulse_interp_crop_limited_restrict,ne]],n_list_all)[0]*(ne ** 2)*multiplicative_factor,label='H3+ dissociation\nH3+ + e- → H(p) + H2')
									plt.plot(n_list_all,np.exp(fit_Yacora_pop_coeff(merge_ne_prof_multipulse_interp_crop_limited_restrict,merge_Te_prof_multipulse_interp_crop_limited_restrict,T_Hp,T_Hm,T_H2p,ne,multiplicative_factor,excitation_restrict,recombination_restrict)(n_list_all, nH_ne,nHp_ne,nHm_ne,nH2_ne,nH2p_ne,nH3p_ne)),'--',label='total fit')
									plt.plot(n_list_all,np.exp(fit_Yacora_pop_coeff(merge_ne_prof_multipulse_interp_crop_limited_restrict,merge_Te_prof_multipulse_interp_crop_limited_restrict,T_Hp,T_Hm,T_H2p,ne,multiplicative_factor,excitation_restrict,recombination_restrict)(n_list_all, nH_ne,nHp_ne,0,0,0,0)),'+',label='excitation + recombination')
									plt.semilogy()
									plt.legend(loc='best')
									plt.ylim(np.min(inverted_profiles_crop_restrict)/2,np.max(inverted_profiles_crop_restrict)*2)
									plt.title('SS lines '+str(n_list_all)+' weights '+str(n_weights)+'\nlocation [r]'+ ' [%.3g mm]' % (1000*r_crop[my_r_pos]) +' , [TS Te, TS ne] '+ ' [%.3g eV, ' % merge_Te_prof_multipulse_interp_crop_limited_restrict + '%.3g #10^20/m^3]' % merge_ne_prof_multipulse_interp_crop_limited_restrict +'\nfit [nH/ne, nH+/ne, nH-/ne, nH2/ne, nH2+/ne, nH3+/ne] , '+ '[%.3g, ' % nH_ne + '%.3g, ' % nHp_ne+ '%.3g, ' % nHm_ne+ '%.3g, ' % nH2_ne+ '%.3g, ' % nH2p_ne+ '%.3g]' % nH3p_ne +'\nH ionization length %.3gm' % ionization_length_H[my_r_pos] +'\nH2 ionization length %.3gm' % ionization_length_H2[my_r_pos])
									plt.xlabel('Exctited state n')
									plt.ylabel('Line emissivity n->2 [W m^-3]')
									# plt.pause(0.01)
									save_done = 0
									save_index=1
									try:	# All this trie are because of a known issues when saving lots of figure inside of multiprocessor
										plt.savefig(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_SS_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps', bbox_inches='tight')
									except Exception as e:
										print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_SS_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(1)+' failed. Reason %s' % e)
										while save_done==0 and save_index<100:
											try:
												tm.sleep(np.random.random()**2)
												plt.savefig(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_SS_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps', bbox_inches='tight')
												print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_SS_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(save_index+1)+' successfull')
												save_done=1
											except Exception as e:
												# print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_SS_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(save_index+1)+' failed. Reason %s' % e)
												save_index+=1
									plt.close('all')


							results = [nH_ne, nHp_ne, nHm_ne, nH2_ne, nH2p_ne, nH3p_ne,residuals]

							output = calc_stuff_output(my_time_pos,my_r_pos, results)
							return output


						if ( (not os.path.exists(path_where_to_save_everything + mod4 +'/SS_results.npz') ) or recorded_data_override[0] ):

							all_indexes = []
							global_index = 0
							my_time_pos = 0
							for my_r_pos in range(len(r_crop)):
								all_indexes.append([global_index,my_time_pos,my_r_pos,1,guess,ionization_length_H,ionization_length_Hm,ionization_length_H2,ionization_length_H2p])
								global_index+=1

							pool = Pool(number_cpu_available)
							# all_results = [*pool.map(calc_stuff, all_indexes)]
							all_results = [*pool.map(calc_stuff_atomic_restricted_high_n_mixed, all_indexes)]
							pool.close()
							pool.join()

							for i in range(len(all_results)):
								nH_ne_all[int(all_results[i].my_r_pos)] = (all_results[i].results)[0]
								nHp_ne_all[int(all_results[i].my_r_pos)] = (all_results[i].results)[1]
								nHm_ne_all[int(all_results[i].my_r_pos)] = (all_results[i].results)[2]
								nH2_ne_all[int(all_results[i].my_r_pos)] = (all_results[i].results)[3]
								nH2p_ne_all[int(all_results[i].my_r_pos)] = (all_results[i].results)[4]
								nH3p_ne_all[int(all_results[i].my_r_pos)] = (all_results[i].results)[5]
								residuals_all[int(all_results[i].my_r_pos)] = (all_results[i].results)[6]


							global_pass = 1
							exec(open("/home/ffederic/work/Collaboratory/test/experimental_data/post_process_PSI_parameter_search_1_Yacora_plots_SS.py").read())

				# 	break
				# break

						# 	np.savez_compressed(path_where_to_save_everything + mod4 +'/SS_results',nH_ne_all=nH_ne_all,nHp_ne_all=nHp_ne_all,nHm_ne_all=nHm_ne_all,nH2_ne_all=nH2_ne_all,nH2p_ne_all=nH2p_ne_all,nH3p_ne_all=nH3p_ne_all,residuals_all=residuals_all)
						# else:
						# 	nH_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['nH_ne_all']
						# 	nHp_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['nHp_ne_all']
						# 	nHm_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['nHm_ne_all']
						# 	nH2_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['nH2_ne_all']
						# 	nH2p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['nH2p_ne_all']
						# 	nH3p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['nH3p_ne_all']
						# 	residuals_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['residuals_all']
						#
						# global_pass = 1
						# exec(open("/home/ffederic/work/Collaboratory/test/experimental_data/post_process_PSI_parameter_search_1_Yacora_plots.py").read())
						#
						# if ( (not os.path.exists(path_where_to_save_everything + mod4 +'/results2.npz') ) or recorded_data_override[0] ):
						#
						#
						#
						# 	all_indexes = []
						# 	global_index = 0
						# 	for my_time_pos in range(len(time_crop)):
						# 		for my_r_pos in range(len(r_crop)):
						# 			all_indexes.append([global_index,my_time_pos,my_r_pos,2,guess,ionization_length_H,ionization_length_Hm,ionization_length_H2,ionization_length_H2p])
						# 			global_index+=1
						#
						# 	pool = Pool(number_cpu_available)
						# 	all_results = [*pool.map(calc_stuff, all_indexes)]
						# 	pool.close()
						# 	pool.join()
						#
						# 	for i in range(len(all_results)):
						# 		nH_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[0]
						# 		nHp_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[1]
						# 		nHm_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[2]
						# 		nH2_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[3]
						# 		nH2p_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[4]
						# 		nH3p_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[5]
						# 		residuals_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[6]
						#
						# 	np.savez_compressed(path_where_to_save_everything + mod4 +'/results2',nH_ne_all=nH_ne_all,nHp_ne_all=nHp_ne_all,nHm_ne_all=nHm_ne_all,nH2_ne_all=nH2_ne_all,nH2p_ne_all=nH2p_ne_all,nH3p_ne_all=nH3p_ne_all,residuals_all=residuals_all)
						# else:
						# 	nH_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['nH_ne_all']
						# 	nHp_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['nHp_ne_all']
						# 	nHm_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['nHm_ne_all']
						# 	nH2_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['nH2_ne_all']
						# 	nH2p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['nH2p_ne_all']
						# 	nH3p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['nH3p_ne_all']
						# 	residuals_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['residuals_all']
						#
						# global_pass = 2
						# exec(open("/home/ffederic/work/Collaboratory/test/experimental_data/post_process_PSI_parameter_search_1_Yacora_plots.py").read())


					# 	if ( (not os.path.exists(path_where_to_save_everything + mod4 +'/results_2.npz') ) or recorded_data_override[1] ):
					#
					# 		if False:
					# 			nH_ne_all = medfilt(nH_ne_all,[3,15])
					# 			nHp_ne_all = medfilt(nHp_ne_all,[3,7])
					# 			nHm_ne_all = medfilt(nHm_ne_all,[3,5])
					# 			nH2_ne_all = medfilt(nH2_ne_all,[3,7])
					# 			nH2p_ne_all = medfilt(nH2p_ne_all,[3,5])
					# 			nH3p_ne_all = medfilt(nH3p_ne_all,[3,5])
					# 		else:
					# 			nH_ne_all = convolve(nH_ne_all,np.ones((3,15))/np.sum(np.ones((3,15))),mode='constant', cval=0.0)
					# 			nHp_ne_all = convolve(nHp_ne_all,np.ones((3,15))/np.sum(np.ones((3,15))),mode='constant', cval=0.0)
					# 			nHm_ne_all = convolve(nHm_ne_all,np.ones((3,3))/np.sum(np.ones((3,3))),mode='constant', cval=0.0)
					# 			nH2_ne_all = medfilt(nH2_ne_all,[3,7])
					# 			nH2p_ne_all = convolve(nH2p_ne_all,np.ones((3,3))/np.sum(np.ones((3,3))),mode='constant', cval=0.0)
					# 			nH3p_ne_all = convolve(nH3p_ne_all,np.ones((3,3))/np.sum(np.ones((3,3))),mode='constant', cval=0.0)
					#
					# 		global_pass = 2
					# 		exec(open("/home/ffederic/work/Collaboratory/test/experimental_data/post_process_PSI_parameter_search_1_Yacora_plots.py").read())
					#
					#
					# 		all_indexes = []
					# 		global_index = 0
					# 		for my_time_pos in range(len(time_crop)):
					# 			for my_r_pos in range(len(r_crop)):
					# 				if max_nHm_ne == min_multiplier and max_nH2_ne == min_multiplier and max_nH2p_ne == min_multiplier and max_nH3p_ne == min_multiplier:
					# 					all_indexes.append([global_index,my_time_pos,my_r_pos,2,[nH_ne_all[my_time_pos,my_r_pos], nHp_ne_all[my_time_pos,my_r_pos]]])
					# 					global_index+=1
					# 				if max_nHm_ne == 1 and max_nH2_ne == min_multiplier and max_nH2p_ne == min_multiplier and max_nH3p_ne == min_multiplier:
					# 					all_indexes.append([global_index,my_time_pos,my_r_pos,2,[nH_ne_all[my_time_pos,my_r_pos], nHp_ne_all[my_time_pos,my_r_pos],nHm_ne_all[my_time_pos,my_r_pos]]])
					# 					global_index+=1
					# 				elif max_nHm_ne == min_multiplier and max_nH2_ne == 1 and max_nH2p_ne == min_multiplier and max_nH3p_ne == min_multiplier:
					# 					all_indexes.append([global_index,my_time_pos,my_r_pos,2,[nH_ne_all[my_time_pos,my_r_pos], nHp_ne_all[my_time_pos,my_r_pos],nH2_ne_all[my_time_pos,my_r_pos]]])
					# 					global_index+=1
					# 				elif max_nHm_ne == min_multiplier and max_nH2_ne == min_multiplier and max_nH2p_ne == 1 and max_nH3p_ne == min_multiplier:
					# 					all_indexes.append([global_index,my_time_pos,my_r_pos,2,[nH_ne_all[my_time_pos,my_r_pos], nHp_ne_all[my_time_pos,my_r_pos],nH2p_ne_all[my_time_pos,my_r_pos]]])
					# 					global_index+=1
					# 				elif max_nHm_ne == min_multiplier and max_nH2_ne == min_multiplier and max_nH2p_ne == min_multiplier and max_nH3p_ne == 1:
					# 					all_indexes.append([global_index,my_time_pos,my_r_pos,2,[nH_ne_all[my_time_pos,my_r_pos], nHp_ne_all[my_time_pos,my_r_pos],nH3p_ne_all[my_time_pos,my_r_pos]]])
					# 					global_index+=1
					# 				elif max_nHm_ne == 1 and max_nH2_ne == 1 and max_nH2p_ne == min_multiplier and max_nH3p_ne == min_multiplier:
					# 					all_indexes.append([global_index,my_time_pos,my_r_pos,2,[nH_ne_all[my_time_pos,my_r_pos], nHp_ne_all[my_time_pos,my_r_pos],nHm_ne_all[my_time_pos,my_r_pos],nH2_ne_all[my_time_pos,my_r_pos]]])
					# 					global_index+=1
					# 				elif max_nHm_ne == min_multiplier and max_nH2_ne == 1 and max_nH2p_ne == 1 and max_nH3p_ne == min_multiplier:
					# 					all_indexes.append([global_index,my_time_pos,my_r_pos,2,[nH_ne_all[my_time_pos,my_r_pos], nHp_ne_all[my_time_pos,my_r_pos],nH2_ne_all[my_time_pos,my_r_pos],nH2p_ne_all[my_time_pos,my_r_pos]]])
					# 					global_index+=1
					# 				elif max_nHm_ne == 1 and max_nH2_ne == 1 and max_nH2p_ne == 1 and max_nH3p_ne == min_multiplier:
					# 					all_indexes.append([global_index,my_time_pos,my_r_pos,2,[nH_ne_all[my_time_pos,my_r_pos], nHp_ne_all[my_time_pos,my_r_pos],nHm_ne_all[my_time_pos,my_r_pos],nH2_ne_all[my_time_pos,my_r_pos],nH2p_ne_all[my_time_pos,my_r_pos]]])
					# 					global_index+=1
					# 				elif max_nHm_ne == 1 and max_nH2_ne == 1 and max_nH2p_ne == 1 and max_nH3p_ne == 1:
					# 					all_indexes.append([global_index,my_time_pos,my_r_pos,2,[nH_ne_all[my_time_pos,my_r_pos], nHp_ne_all[my_time_pos,my_r_pos],nHm_ne_all[my_time_pos,my_r_pos],nH2_ne_all[my_time_pos,my_r_pos],nH2p_ne_all[my_time_pos,my_r_pos],nH3p_ne_all[my_time_pos,my_r_pos]]])
					# 					global_index+=1
					#
					# 		pool = Pool(number_cpu_available)
					# 		all_results = [*pool.map(calc_stuff, all_indexes)]
					# 		pool.close()
					# 		pool.join()
					#
					# 		for i in range(len(all_results)):
					# 			nH_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[0]
					# 			nHp_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[1]
					# 			nHm_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[2]
					# 			nH2_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[3]
					# 			nH2p_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[4]
					# 			nH3p_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[5]
					# 			residuals_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[6]
					#
					#
					# 		np.savez_compressed(path_where_to_save_everything + mod4 +'/results_2',nH_ne_all=nH_ne_all,nHp_ne_all=nHp_ne_all,nHm_ne_all=nHm_ne_all,nH2_ne_all=nH2_ne_all,nH2p_ne_all=nH2p_ne_all,nH3p_ne_all=nH3p_ne_all,residuals_all=residuals_all)
					# 	else:
					# 		nH_ne_all = np.load(path_where_to_save_everything + mod4 +'/results_2.npz')['nH_ne_all']
					# 		nHp_ne_all = np.load(path_where_to_save_everything + mod4 +'/results_2.npz')['nHp_ne_all']
					# 		nHm_ne_all = np.load(path_where_to_save_everything + mod4 +'/results_2.npz')['nHm_ne_all']
					# 		nH2_ne_all = np.load(path_where_to_save_everything + mod4 +'/results_2.npz')['nH2_ne_all']
					# 		nH2p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results_2.npz')['nH2p_ne_all']
					# 		nH3p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results_2.npz')['nH3p_ne_all']
					# 		residuals_all = np.load(path_where_to_save_everything + mod4 +'/results_2.npz')['residuals_all']
					#
					#
					# # plt.figure()
					# # plt.title('ne,Te,nH_ne,h_atomic_density,OES_multiplier\n'+str([*merge_ne_prof_multipulse_interp_crop_limited_restrict,*merge_Te_prof_multipulse_interp_crop_limited_restrict,*fit[0],OES_multiplier]))
					# # plt.plot(n_list,inverted_profiles_crop_restrict*OES_multiplier)
					# # plt.plot(n_list,fit_nH_ne_h_atomic_density(recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict)(n_list,*fit[0]))
					# # plt.pause(0.01)
					#
					# global_pass = 3
					# exec(open("/home/ffederic/work/Collaboratory/test/experimental_data/post_process_PSI_parameter_search_1_Yacora_plots.py").read())
