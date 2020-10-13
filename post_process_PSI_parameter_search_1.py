import numpy as np
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())
# import matplotlib.pyplot as plt
#import .functions
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
from functions.fabio_add import find_index_of_file

os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
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

pecfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/pec12#h_balmer#h0.dat'
scdfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/scd12_h.dat'
acdfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/acd12_h.dat'

if False:

	solutions=[]
	for merge_ID_target in [85,89,93]:
		for high_line in [7,8,9,10]:
			# high_line = 7
			low_line = 4
			time_shift_factor = -0.05
			spatial_factor = 1.35
			nH_ne = 1
			OES_multiplier = 1

			# merge_ID_target = 85


			figure_index = 0

			# calculate_geometry = False
			# merge_ID_target = 17	#	THIS IS GIVEN BY THE LAUNCHER
			for i in range(10):
				print('.')
			print('Starting to work on merge number'+str(merge_ID_target))
			for i in range(10):
				print('.')


			time_resolution_scan = False
			time_resolution_scan_improved = True
			time_resolution_extra_skip = 0




			started=0
			rows_range_for_interp = 25/3 # rows that I use for interpolation (box with twice this side length, not sphere)
			# if merge_ID_target>=66:
			# 	rows_range_for_interp = geom_store['binInterv'][0] / 6
			if time_resolution_scan:
				conventional_time_step = 0.01	# ms
			else:
				conventional_time_step = 0.05	# ms
			# interpolation_type = 'quadratic'	# 'linear' or 'quadratic'
			grade_of_interpolation = 3	#this is the exponent used for the weights of the interpolation for image resampling
			type_of_image = '12bit'	# '12bit' or '16bit'
			# if type_of_image=='12bit':
			row_shift=2*10280/1000000	# ms
			# elif type_of_image=='16bit':
			# 	print('Row shift to be checked')
			# 	exit()
			# time_range_for_interp = rows_range_for_interp*row_shift
			# merge_time_window=[-1,4]
			merge_time_window=[-10,10]
			overexposed_treshold = 3600
			path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)



			new_timesteps = np.linspace(merge_time_window[0]+0.5, merge_time_window[1]-0.5,int((merge_time_window[1]-0.5-(merge_time_window[0]+0.5))/conventional_time_step+1))
			dt=np.nanmedian(np.diff(new_timesteps))



			# first_time = np.min(new_timesteps)
			# last_time = np.max(new_timesteps)


			# Data from wikipedia
			energy_difference = np.array([1.89,2.55,2.86,3.03,3.13,3.19,3.23,3.26, 3.29 ])	#eV
			# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
			statistical_weigth = np.array([32,50,72,98,128,162,200,242,288])	#gi-gk
			einstein_coeff = np.array([8.4193e-2,2.53044e-2,9.7320e-3,4.3889e-3,2.2148e-3,1.2156e-3,7.1225e-4,4.3972e-4,2.8337e-4])*1e8	#1/s
			J_to_eV = 6.242e18
			# Used formula 2.3 in Rion Barrois thesys, 2017
			color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise']

			inverted_profiles_original = 4*np.pi*np.load(path_where_to_save_everything+'/inverted_profiles.npy')		# in W m^-3 sr^-1  *  4*pi sr  =  W m^-3
			all_fits = np.load(path_where_to_save_everything +'/merge'+str(merge_ID_target)+'_all_fits.npy')	# in W m^-2 sr^-1
			merge_Te_prof_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_Te_prof_multipulse']
			merge_dTe_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_dTe_multipulse']
			merge_ne_prof_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_ne_prof_multipulse']
			merge_dne_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_dne_multipulse']
			merge_time_original = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_time']
			TS_dt=np.nanmedian(np.diff(merge_time_original))


			TS_size=[-4.149230769230769056e+01,4.416923076923076508e+01]
			TS_r=TS_size[0] + np.linspace(0,1,65)*(TS_size[1]- TS_size[0])
			TS_dr = np.median(np.diff(TS_r))/1000
			gauss = lambda x, A, sig, x0: A * np.exp(-((x - x0) / sig) ** 2)
			profile_centres = []
			profile_centres_score = []
			for index in range(np.shape(merge_Te_prof_multipulse)[0]):
				yy = merge_Te_prof_multipulse[index]
				p0 = [np.max(yy),10,0]
				bds = [[0,-40,np.min(TS_r)],[np.inf,40,np.max(TS_r)]]
				fit = curve_fit(gauss, TS_r, yy, p0, maxfev=100000, bounds=bds)
				profile_centres.append(fit[0][-1])
				profile_centres_score.append(fit[1][-1,-1])
				# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)
			profile_centres = np.array(profile_centres)
			profile_centres_score = np.array(profile_centres_score)
			centre = np.nanmean(profile_centres[profile_centres_score<1])
			TS_r_new = np.abs(TS_r-centre)/1000


			def residual_ext(nH_ne,high_line,merge_time_original,merge_Te_prof_multipulse,inverted_profiles_original,merge_ne_prof_multipulse,new_timesteps,TS_dt,TS_dr,dt ):
				# def residual(input):
				def residual(input):
					import numpy as np
					spatial_factor, time_shift_factor, OES_multiplier = *input,1
					print('spatial_factor,time_shift_factor,OES_multiplier')
					print(spatial_factor,time_shift_factor,OES_multiplier)
					# spatial_factor = input[0]
					# time_shift_factor = input[1]
					# OES_multiplier = input[2]


					merge_time = time_shift_factor + merge_time_original
					inverted_profiles = (1 / spatial_factor) * inverted_profiles_original  # in W m^-3 sr^-1  *  4*pi sr  =  W m^-3

					dx = (18 / 40 * (50.5 / 27.4) / 1e3) * spatial_factor
					xx = np.arange(40) * dx  # m
					xn = np.linspace(0, max(xx), 1000)
					r = np.linspace(-max(xx) / 2, max(xx) / 2, 1000)
					r = r[::10]
					dr = np.median(np.diff(r))

					temp1 = np.zeros_like(inverted_profiles[:, 0])
					temp3 = np.zeros_like(inverted_profiles[:, 0])
					interp_range_t = max(dt / 2, TS_dt) * 1
					interp_range_r = max(dx / 2, TS_dr) * 1
					for i_t, value_t in enumerate(new_timesteps):
						if np.sum(np.abs(merge_time - value_t) < interp_range_t) == 0:
							continue
						for i_r, value_r in enumerate(np.abs(r)):
							if np.sum(np.abs(TS_r_new - value_r) < interp_range_r) == 0:
								continue
							temp1[i_t, i_r] = np.mean(merge_Te_prof_multipulse[np.abs(merge_time - value_t) < interp_range_t][:,np.abs(TS_r_new - value_r) < interp_range_r])
							temp3[i_t, i_r] = np.mean(merge_ne_prof_multipulse[np.abs(merge_time - value_t) < interp_range_t][:,np.abs(TS_r_new - value_r) < interp_range_r])

					merge_Te_prof_multipulse_interp = np.array(temp1)
					merge_ne_prof_multipulse_interp = np.array(temp3)

					# I crop to the usefull stuff
					start_time = np.abs(new_timesteps - 0).argmin()
					end_time = np.abs(new_timesteps - 1.5).argmin() + 1
					time_crop = new_timesteps[start_time:end_time]
					start_r = np.abs(r - 0).argmin()
					end_r = np.abs(r - 5).argmin() + 1
					r_crop = r[start_r:end_r]
					temp_r, temp_t = np.meshgrid(r_crop, time_crop)
					merge_Te_prof_multipulse_interp_crop = merge_Te_prof_multipulse_interp[start_time:end_time, start_r:end_r]
					merge_ne_prof_multipulse_interp_crop = merge_ne_prof_multipulse_interp[start_time:end_time, start_r:end_r]
					inverted_profiles_crop = inverted_profiles[start_time:end_time, :, start_r:end_r]
					inverted_profiles_crop[np.isnan(inverted_profiles_crop)] = 0
					# all_fits_crop = all_fits[start_time:end_time]

					merge_Te_prof_multipulse_interp_crop_limited = cp.deepcopy(merge_Te_prof_multipulse_interp_crop)
					merge_Te_prof_multipulse_interp_crop_limited[merge_Te_prof_multipulse_interp_crop_limited < 0.2] = 0
					merge_ne_prof_multipulse_interp_crop_limited = cp.deepcopy(merge_ne_prof_multipulse_interp_crop)
					merge_ne_prof_multipulse_interp_crop_limited[merge_ne_prof_multipulse_interp_crop_limited < 5e-07] = 0
					# excitation = []
					# for isel in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
					isel =high_line - 4 +2
					temp = read_adf15(pecfile, isel, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
					temp[np.isnan(temp)] = 0
					excitation = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
					# excitation.append(temp)
					excitation = np.array(excitation)  # in # photons cm^-3 s^-1
					excitation = (excitation.T * (10 ** -6) * (energy_difference[high_line - 4] / J_to_eV)).T  # in W m^-3 / (# / m^3)**2

					# recombination = []
					# for isel in [20, 21, 22, 23, 24, 25, 26, 27, 28]:
					isel =high_line - 4 + 20
					temp = read_adf15(pecfile, isel, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
					temp[np.isnan(temp)] = 0
					recombination = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
					# recombination.append(temp)
					recombination = np.array(recombination)  # in # photons cm^-3 s^-1 / (# cm^-3)**2
					recombination = (recombination.T * (10 ** -6) * (energy_difference[high_line - 4] / J_to_eV)).T  # in W m^-3 / (# / m^3)**2

					recombination_emissivity = (recombination * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited * (10 ** 20)) ** 2)).astype('float')
					difference = inverted_profiles_crop[:, high_line - 4] * OES_multiplier - recombination_emissivity
					h_atomic_density = difference / (excitation * (merge_ne_prof_multipulse_interp_crop * (10 ** 20))).astype('float')
					h_atomic_density[np.logical_not(np.isfinite(h_atomic_density))] = 0
					h_atomic_density[h_atomic_density < 0] = 0
					excitation_emissivity = (excitation * (merge_ne_prof_multipulse_interp_crop * (10 ** 20)) * (h_atomic_density)).astype('float')

					if True:
						isel =low_line - 4 +2
						temp = read_adf15(pecfile, isel, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
						temp[np.isnan(temp)] = 0
						excitation = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
						# excitation.append(temp)
						excitation = np.array(excitation)  # in # photons cm^-3 s^-1
						excitation = (excitation.T * (10 ** -6) * (energy_difference[low_line - 4] / J_to_eV)).T  # in W m^-3 / (# / m^3)**2
						excitation_emissivity = (excitation * (merge_ne_prof_multipulse_interp_crop * (10 ** 20)) * (h_atomic_density)).astype('float')

						# recombination = []
						# for isel in [20, 21, 22, 23, 24, 25, 26, 27, 28]:
						isel =low_line - 4 + 20
						temp = read_adf15(pecfile, isel, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
						temp[np.isnan(temp)] = 0
						recombination = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
						# recombination.append(temp)
						recombination = np.array(recombination)  # in # photons cm^-3 s^-1 / (# cm^-3)**2
						recombination = (recombination.T * (10 ** -6) * (energy_difference[low_line - 4] / J_to_eV)).T  # in W m^-3 / (# / m^3)**2
						recombination_emissivity = (recombination * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited * (10 ** 20)) ** 2)).astype('float')


						# residual_emission = []
						# for index in range(len(recombination_emissivity)):
						index=low_line - 4
						residual_emission=(inverted_profiles_crop[:, index] * OES_multiplier - recombination_emissivity -excitation_emissivity)
						residual_emission = np.array(residual_emission)
					elif False:
						index=high_line - 4
						residual_emission=(inverted_profiles_crop[:, index] * OES_multiplier - recombination_emissivity -excitation_emissivity)
						residual_emission = np.array(residual_emission)


					# print(np.sum(residual_emission[0][residual_emission[0]<0]))
					return residual_emission.flatten()
				return residual

			# # bds=[[1,-0.2,1],[2,0.1,4]]
			# # guess = [1,0,1]
			# # curve_fit(residual_ext(nH_ne,high_line,merge_time_original,merge_Te_prof_multipulse,inverted_profiles_original,merge_ne_prof_multipulse,new_timesteps,TS_dt,TS_dr,dt ),0,0,p0=guess,bounds=bds)
			# #
			# # guess = [spatial_factor,time_shift_factor,OES_multiplier]
			# # sol = newton_krylov(residual_ext(nH_ne,high_line,merge_time_original,merge_Te_prof_multipulse,inverted_profiles_original,merge_ne_prof_multipulse,new_timesteps,TS_dt,TS_dr,dt ), guess,method='lgmres', verbose=1,iter=30)
			# # print('Residual: %g' % abs(residual_ext(hx,hy,ht,P_left, P_right,P_top, P_bottom, P_init, conductivity,thickness,emissivity,diffusivity,sigmaSB,T_reference,Power_in,d2x,d2y,dt)(sol)).max())
			#
			# bds=[[1,-0.2,0.5],[2,0.1,4]]
			# guess = [1.3, -0.09      ,  1.9]
			# guess = np.array(guess).astype('float').flatten()
			# sol = least_squares(residual_ext(nH_ne,high_line,merge_time_original,merge_Te_prof_multipulse,inverted_profiles_original,merge_ne_prof_multipulse,new_timesteps,TS_dt,TS_dr,dt ), guess,bounds = bds,ftol=1-5,xtol=1e-16, max_nfev = 60,verbose=2,diff_step=[0.05,0.05,0.01],tr_solver='lsmr',x_scale='jac')

			bds=[[1,-0.2],[2,0.1]]
			guess = [1, 0 ]
			guess = np.array(guess).astype('float').flatten()
			sol = least_squares(residual_ext(nH_ne,high_line,merge_time_original,merge_Te_prof_multipulse,inverted_profiles_original,merge_ne_prof_multipulse,new_timesteps,TS_dt,TS_dr,dt ), guess,bounds = bds,ftol=1-5,xtol=1e-8, max_nfev = 6,verbose=2,diff_step=[-0.05,-0.06],tr_solver='lsmr',x_scale='jac')
			for index in range(5):
				sol = least_squares(residual_ext(nH_ne,high_line,merge_time_original,merge_Te_prof_multipulse,inverted_profiles_original,merge_ne_prof_multipulse,new_timesteps,TS_dt,TS_dr,dt ), sol.x,bounds = bds,ftol=1-5,xtol=1e-8, max_nfev = 10,verbose=2,diff_step=[-0.05,-0.06],tr_solver='lsmr',x_scale='jac')
			solutions.append(sol.x)

	solution=[]
	# high_line = 7
	solution .append( [1.26030977, -0.09      ,  1.88243027])
	# high_line = 8
	solution .append([1.26007684, -0.09      ,  1.9137136])
	# high_line = 9
	solution .append( [1.25725712, -0.08995191,  1.90669528])
	spatial_factor, time_shift_factor, OES_multiplier = np.mean(solution,axis=0)

	solutions = np.array([[ 1.48296984, -0.04946108],
		   [ 1.51064555, -0.04680084],
		   [ 1.32579228, -0.02394259],
		   [ 1.24151298, -0.00740485],
		   [ 1.61002045, -0.09527693],
		   [ 1.99950828, -0.02437629],
		   [ 1.55462026, -0.06259724],
		   [ 1.56731445, -0.02995821],
		   [ 1.2821108 , -0.01490611],
		   [ 1.39634897, -0.04119113],
		   [ 1.01181224, -0.00750937],
		   [ 1.01936835, -0.03175309]])

	np.median(solutions,axis=0)
	spatial_factor, time_shift_factor = np.median(solutions,axis=0)


elif False:

	make_plots = True



	for merge_ID_target in [85,86,87,88,89,92,93,94]:

		# merge_ID_target = 85


		figure_index = 0

		# calculate_geometry = False
		# merge_ID_target = 17	#	THIS IS GIVEN BY THE LAUNCHER
		for i in range(10):
			print('.')
		print('Starting to work on merge number'+str(merge_ID_target))
		for i in range(10):
			print('.')


		time_resolution_scan = False
		time_resolution_scan_improved = True
		time_resolution_extra_skip = 0


		n_list = np.array([5,6,7, 9, 10, 11, 12])
		n_list_1 = np.array([4])
		n_weights = [1,1,1, 1, 1, 1, 1, 1000]
		for index in (n_list_1-4):
			n_weights[index]=3
		n_weights[np.max(n_list_1)-3]=2
		min_nH_ne = 0.999
		max_nH_ne = 0.1

		mod = '/min_nH_ne'+str(min_nH_ne)+'/n_dummy_'+str(len(n_list_1))

		print('dummy')
		print(n_list_1)
		print('min_nH_ne')
		print(min_nH_ne)
		print('max_nH_ne')
		print(max_nH_ne)
		print('n_weights')
		print(n_weights)

		started=0
		rows_range_for_interp = 25/3 # rows that I use for interpolation (box with twice this side length, not sphere)
		# if merge_ID_target>=66:
		# 	rows_range_for_interp = geom_store['binInterv'][0] / 6
		if time_resolution_scan:
			conventional_time_step = 0.01	# ms
		else:
			conventional_time_step = 0.05	# ms
		# interpolation_type = 'quadratic'	# 'linear' or 'quadratic'
		grade_of_interpolation = 3	#this is the exponent used for the weights of the interpolation for image resampling
		type_of_image = '12bit'	# '12bit' or '16bit'
		# if type_of_image=='12bit':
		row_shift=2*10280/1000000	# ms
		# elif type_of_image=='16bit':
		# 	print('Row shift to be checked')
		# 	exit()
		# time_range_for_interp = rows_range_for_interp*row_shift
		# merge_time_window=[-1,4]
		merge_time_window=[-10,10]
		overexposed_treshold = 3600
		path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)

		if not os.path.exists(path_where_to_save_everything + mod):
			os.makedirs(path_where_to_save_everything + mod)


		new_timesteps = np.linspace(merge_time_window[0]+0.5, merge_time_window[1]-0.5,int((merge_time_window[1]-0.5-(merge_time_window[0]+0.5))/conventional_time_step+1))
		dt=np.nanmedian(np.diff(new_timesteps))

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
		energy_difference = np.array([1.89,2.55,2.86,3.03,3.13,3.19,3.23,3.26, 3.29 ])	#eV
		# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
		statistical_weigth = np.array([32,50,72,98,128,162,200,242,288])	#gi-gk
		einstein_coeff = np.array([8.4193e-2,2.53044e-2,9.7320e-3,4.3889e-3,2.2148e-3,1.2156e-3,7.1225e-4,4.3972e-4,2.8337e-4])*1e8	#1/s
		J_to_eV = 6.242e18
		# Used formula 2.3 in Rion Barrois thesys, 2017
		color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise']

		inverted_profiles_original = 4*np.pi*np.load(path_where_to_save_everything+'/inverted_profiles.npy')		# in W m^-3 sr^-1  *  4*pi sr  =  W m^-3
		all_fits = np.load(path_where_to_save_everything +'/merge'+str(merge_ID_target)+'_all_fits.npy')	# in W m^-2 sr^-1
		merge_Te_prof_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_Te_prof_multipulse']
		merge_dTe_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_dTe_multipulse']
		merge_ne_prof_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_ne_prof_multipulse']
		merge_dne_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_dne_multipulse']
		merge_time_original = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_time']

		# for time_shift_factor in time_shift_factor_all:
		#
		# 	mod = '/spatial_factor' + str(spatial_factor)+'/time_shift_factor' + str(time_shift_factor)
		# 	if not os.path.exists(path_where_to_save_everything + mod):
		# 		os.makedirs(path_where_to_save_everything + mod)

		def residual_ext(inverted_profiles_original,merge_Te_prof_multipulse,merge_ne_prof_multipulse,merge_time_original):
			def residuals(input):
				spatial_factor=input[0]
				time_shift_factor = input[1]
				print('spatial_factor,time_shift_factor  '+str([spatial_factor,time_shift_factor]))


				dx = (18 / 40 * (50.5 / 27.4) / 1e3) * spatial_factor
				xx = np.arange(40) * dx  # m
				xn = np.linspace(0, max(xx), 1000)
				r = np.linspace(-max(xx) / 2, max(xx) / 2, 1000)
				r=r[::10]
				dr=np.median(np.diff(r))


				merge_time = time_shift_factor + merge_time_original
				inverted_profiles = (1 / spatial_factor) * inverted_profiles_original  # in W m^-3 sr^-1  *  4*pi sr  =  W m^-3
				TS_dt=np.nanmedian(np.diff(merge_time))


				if np.max(merge_Te_prof_multipulse)<=0:
					print('merge'+str(merge_ID_target)+" has no recorded temperature")
					# continue

				TS_size=[-4.149230769230769056e+01,4.416923076923076508e+01]
				TS_r=TS_size[0] + np.linspace(0,1,65)*(TS_size[1]- TS_size[0])
				TS_dr = np.median(np.diff(TS_r))/1000
				gauss = lambda x, A, sig, x0: A * np.exp(-((x - x0) / sig) ** 2)
				profile_centres = []
				profile_centres_score = []
				for index in range(np.shape(merge_Te_prof_multipulse)[0]):
					yy = merge_Te_prof_multipulse[index]
					p0 = [np.max(yy),10,0]
					bds = [[0,-40,np.min(TS_r)],[np.inf,40,np.max(TS_r)]]
					fit = curve_fit(gauss, TS_r, yy, p0, maxfev=100000, bounds=bds)
					profile_centres.append(fit[0][-1])
					profile_centres_score.append(fit[1][-1,-1])
					# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)
				profile_centres = np.array(profile_centres)
				profile_centres_score = np.array(profile_centres_score)
				centre = np.nanmean(profile_centres[profile_centres_score<1])
				TS_r_new = np.abs(TS_r-centre)/1000
				# temp_r, temp_t = np.meshgrid(TS_r_new, merge_time)
				# plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse,vmin=0);plt.colorbar().set_label('Te [eV]');plt.pause(0.01)
				# plt.figure();plt.pcolor(temp_t,temp_r,merge_ne_prof_multipulse,vmin=0);plt.colorbar().set_label('ne [10^20 #/m^3]');plt.pause(0.01)


				temp1=np.zeros_like(inverted_profiles[:,0])
				temp3=np.zeros_like(inverted_profiles[:,0])
				interp_range_t = max(dt/2,TS_dt)*1
				interp_range_r = max(dx/2,TS_dr)*1
				for i_t,value_t in enumerate(new_timesteps):
					if np.sum(np.abs(merge_time-value_t) < interp_range_t) == 0:
						continue
					for i_r,value_r in enumerate(np.abs(r)):
						if np.sum(np.abs(TS_r_new - value_r) < interp_range_r) == 0:
							continue
						temp1[i_t,i_r] = np.mean(merge_Te_prof_multipulse[np.abs(merge_time-value_t)<interp_range_t][:,np.abs(TS_r_new-value_r)<interp_range_r])
						temp3[i_t,i_r] = np.mean(merge_ne_prof_multipulse[np.abs(merge_time-value_t)<interp_range_t][:,np.abs(TS_r_new-value_r)<interp_range_r])

				merge_Te_prof_multipulse_interp=np.array(temp1)
				merge_ne_prof_multipulse_interp=np.array(temp3)
				temp_r, temp_t = np.meshgrid(r, new_timesteps)


				# I crop to the usefull stuff
				start_time = np.abs(new_timesteps-0).argmin()
				end_time = np.abs(new_timesteps-1.5).argmin()+1
				time_crop = new_timesteps[start_time:end_time]
				start_r = np.abs(r-0).argmin()
				end_r = np.abs(r-5).argmin()+1
				r_crop = r[start_r:end_r]
				temp_r, temp_t = np.meshgrid(r_crop, time_crop)
				merge_Te_prof_multipulse_interp_crop = merge_Te_prof_multipulse_interp[start_time:end_time,start_r:end_r]
				merge_ne_prof_multipulse_interp_crop = merge_ne_prof_multipulse_interp[start_time:end_time,start_r:end_r]
				inverted_profiles_crop = inverted_profiles[start_time:end_time,:,start_r:end_r]
				inverted_profiles_crop[np.isnan(inverted_profiles_crop)] = 0
				all_fits_crop = all_fits[start_time:end_time]
				# inverted_profiles_crop[inverted_profiles_crop<0] = 0

				x_local = xx - spatial_factor * 17.4 / 1000
				dr_crop = np.median(np.diff(r_crop))

				merge_Te_prof_multipulse_interp_crop_limited = cp.deepcopy(merge_Te_prof_multipulse_interp_crop)
				merge_Te_prof_multipulse_interp_crop_limited[merge_Te_prof_multipulse_interp_crop_limited<0.2]=0
				merge_ne_prof_multipulse_interp_crop_limited = cp.deepcopy(merge_ne_prof_multipulse_interp_crop)
				merge_ne_prof_multipulse_interp_crop_limited[merge_ne_prof_multipulse_interp_crop_limited<5e-07]=0
				excitation = []
				for isel in [2,3,4,5,6,7,8,9,10]:
					temp = read_adf15(pecfile, isel, merge_Te_prof_multipulse_interp_crop_limited.flatten(), (merge_ne_prof_multipulse_interp_crop_limited*10**(20-6)).flatten())[0]	# ADAS database is in cm^3   # photons s^-1 cm^-3
					temp[np.isnan(temp)] = 0
					temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
					excitation.append(temp)
				excitation = np.array(excitation)	# in # photons cm^-3 s^-1
				excitation = (excitation.T*(10**-6)*(energy_difference/J_to_eV)).T	# in W m^-3 / (# / m^3)**2

				recombination = []
				for isel in [20,21,22,23,24,25,26,27,28]:
					temp = read_adf15(pecfile, isel, merge_Te_prof_multipulse_interp_crop_limited.flatten(), (merge_ne_prof_multipulse_interp_crop_limited*10**(20-6)).flatten())[0]	# ADAS database is in cm^3   # photons s^-1 cm^-3
					temp[np.isnan(temp)] = 0
					temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
					recombination.append(temp)
				recombination = np.array(recombination)	# in # photons cm^-3 s^-1 / (# cm^-3)**2
				recombination = (recombination.T*(10**-6)*(energy_difference/J_to_eV)).T	# in W m^-3 / (# / m^3)**2




				def residual_ext1(n_list,n_list_1,n_weights,time_crop,r_crop,inverted_profiles_crop,recombination,merge_ne_prof_multipulse_interp_crop_limited,merge_Te_prof_multipulse_interp_crop_limited,excitation):
					def residuals1(input):
						OES_multiplier=input
						print(OES_multiplier)

						n_list_all = np.sort(np.concatenate((n_list, n_list_1)))
						# OES_multiplier = 0.81414701
						# nH_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						# h_atomic_density_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						# dummy1_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						# dummy2_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						# dummy3_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						residuals_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						for my_time_pos in range(len(time_crop)):
							for my_r_pos in range(len(r_crop)):
								inverted_profiles_crop_restrict = inverted_profiles_crop[my_time_pos, n_list_all-4, my_r_pos].flatten()
								recombination_restrict=recombination[n_list_all-4,my_time_pos, my_r_pos].flatten()
								merge_ne_prof_multipulse_interp_crop_limited_restrict=merge_ne_prof_multipulse_interp_crop_limited[my_time_pos, my_r_pos].flatten()
								merge_Te_prof_multipulse_interp_crop_limited_restrict=merge_Te_prof_multipulse_interp_crop_limited[my_time_pos, my_r_pos].flatten()
								excitation_restrict=excitation[n_list_all-4,my_time_pos, my_r_pos].flatten()

								if (merge_ne_prof_multipulse_interp_crop_limited_restrict==0 and merge_Te_prof_multipulse_interp_crop_limited_restrict==0):
									continue

								dummy = np.zeros((len(n_list_1), len(n_list_all)))
								for value in n_list_1:
									dummy[value - 4][value - 4] = 1

								if len(n_list_1) == 3:
									def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
										def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1, dummy2,dummy3):
											recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
											excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
											dummy_emissivity = np.sum((dummy.T * [dummy1, dummy2, dummy3]).T, axis=0)
											return recombination_emissivity + excitation_emissivity + dummy_emissivity
										return calculated_emission

									bds = [[min_nH_ne, 0, 0, 0, 0], [max_nH_ne, 300, np.inf, np.inf, np.inf]]
									guess = [max_nH_ne, 10, 1000, 1000, 1000]
								elif len(n_list_1) == 2:
									def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
										def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1, dummy2):
											recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
											excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
											dummy_emissivity = np.sum((dummy.T * [dummy1, dummy2]).T, axis=0)
											return recombination_emissivity + excitation_emissivity + dummy_emissivity
										return calculated_emission

									bds = [[min_nH_ne, 0, 0, 0], [max_nH_ne, 300, np.inf, np.inf]]
									guess = [max_nH_ne, 10, 1000, 1000]
								elif len(n_list_1) == 1:
									def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
										def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1):
											recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
											excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
											dummy_emissivity = np.sum((dummy.T * [dummy1]).T, axis=0)
											return recombination_emissivity + excitation_emissivity + dummy_emissivity
										return calculated_emission

									bds = [[min_nH_ne, 0, 0], [max_nH_ne, 300, np.inf]]
									guess = [max_nH_ne, 10, 1000]

								fit = curve_fit(fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict), n_list,inverted_profiles_crop_restrict * OES_multiplier, p0=guess, bounds=bds,sigma=n_weights, maxfev=10000000)

								# nH_ne_all[my_time_pos, my_r_pos]=fit[0][0]
								# h_atomic_density_all[my_time_pos, my_r_pos]=fit[0][1]
								# dummy1_all[my_time_pos, my_r_pos]=fit[0][2]
								# dummy2_all[my_time_pos, my_r_pos]=fit[0][3]
								# dummy3_all[my_time_pos, my_r_pos]=fit[0][4]
								residuals_all[my_time_pos, my_r_pos] = np.sum(((inverted_profiles_crop_restrict * OES_multiplier - fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict, excitation_restrict)(n_list, *fit[0]))) ** 2)
						return residuals_all.flatten()
					return residuals1


				guess = [1]
				bds = [[0.4],[5]]
				sol = least_squares(residual_ext1(n_list,n_list_1,n_weights,time_crop,r_crop,inverted_profiles_crop,recombination,merge_ne_prof_multipulse_interp_crop_limited,merge_Te_prof_multipulse_interp_crop_limited,excitation), guess, bounds=bds, max_nfev=60,	verbose=2, gtol=1e-20, xtol=1e-5, ftol=1e-16, diff_step=0.01, x_scale='jac')
				OES_multiplier=sol.x
				return residual_ext1(n_list,n_list_1, n_weights, time_crop, r_crop, inverted_profiles_crop, recombination,merge_ne_prof_multipulse_interp_crop_limited, merge_Te_prof_multipulse_interp_crop_limited,excitation)(OES_multiplier)
			return residuals


		guess = [1,0]
		bds = [[1,-0.1], [2,0.1]]
		sol = least_squares(residual_ext(inverted_profiles_original, merge_Te_prof_multipulse, merge_ne_prof_multipulse, merge_time_original), guess, bounds=bds,max_nfev=60, verbose=2, gtol=1e-20, xtol=1e-12, ftol=1e-16, diff_step=[0.05,0.05], x_scale='jac')
		print(sol.x)

		spatial_factor = sol.x[0]
		time_shift_factor = sol.x[1]

		print('spatial_factor,time_shift_factor  ' + str([spatial_factor, time_shift_factor]))

		dx = (18 / 40 * (50.5 / 27.4) / 1e3) * spatial_factor
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
		# continue

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
		temp3 = np.zeros_like(inverted_profiles[:, 0])
		interp_range_t = max(dt / 2, TS_dt) * 1
		interp_range_r = max(dx / 2, TS_dr) * 1
		for i_t, value_t in enumerate(new_timesteps):
			if np.sum(np.abs(merge_time - value_t) < interp_range_t) == 0:
				continue
			for i_r, value_r in enumerate(np.abs(r)):
				if np.sum(np.abs(TS_r_new - value_r) < interp_range_r) == 0:
					continue
				temp1[i_t, i_r] = np.mean(merge_Te_prof_multipulse[np.abs(merge_time - value_t) < interp_range_t][:,np.abs(TS_r_new - value_r) < interp_range_r])
				temp3[i_t, i_r] = np.mean(merge_ne_prof_multipulse[np.abs(merge_time - value_t) < interp_range_t][:,np.abs(TS_r_new - value_r) < interp_range_r])

		merge_Te_prof_multipulse_interp = np.array(temp1)
		merge_ne_prof_multipulse_interp = np.array(temp3)
		temp_r, temp_t = np.meshgrid(r, new_timesteps)

		# I crop to the usefull stuff
		start_time = np.abs(new_timesteps - 0).argmin()
		end_time = np.abs(new_timesteps - 1.5).argmin() + 1
		time_crop = new_timesteps[start_time:end_time]
		start_r = np.abs(r - 0).argmin()
		end_r = np.abs(r - 5).argmin() + 1
		r_crop = r[start_r:end_r]
		temp_r, temp_t = np.meshgrid(r_crop, time_crop)
		merge_Te_prof_multipulse_interp_crop = merge_Te_prof_multipulse_interp[start_time:end_time, start_r:end_r]
		merge_ne_prof_multipulse_interp_crop = merge_ne_prof_multipulse_interp[start_time:end_time, start_r:end_r]
		inverted_profiles_crop = inverted_profiles[start_time:end_time, :, start_r:end_r]
		inverted_profiles_crop[np.isnan(inverted_profiles_crop)] = 0
		all_fits_crop = all_fits[start_time:end_time]
		# inverted_profiles_crop[inverted_profiles_crop<0] = 0

		x_local = xx - spatial_factor * 17.4 / 1000
		dr_crop = np.median(np.diff(r_crop))

		merge_Te_prof_multipulse_interp_crop_limited = cp.deepcopy(merge_Te_prof_multipulse_interp_crop)
		merge_Te_prof_multipulse_interp_crop_limited[merge_Te_prof_multipulse_interp_crop_limited < 0.2] = 0
		merge_ne_prof_multipulse_interp_crop_limited = cp.deepcopy(merge_ne_prof_multipulse_interp_crop)
		merge_ne_prof_multipulse_interp_crop_limited[merge_ne_prof_multipulse_interp_crop_limited < 5e-07] = 0
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



		def residual_ext1(n_list, n_list_1, n_weights, time_crop, r_crop, inverted_profiles_crop, recombination,merge_ne_prof_multipulse_interp_crop_limited, merge_Te_prof_multipulse_interp_crop_limited,excitation):
			def residuals1(input):
				OES_multiplier = input
				print(OES_multiplier)

				n_list_all = np.sort(np.concatenate((n_list, n_list_1)))
				# OES_multiplier = 0.81414701
				# nH_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
				# h_atomic_density_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
				# dummy1_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
				# dummy2_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
				# dummy3_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
				residuals_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
				for my_time_pos in range(len(time_crop)):
					for my_r_pos in range(len(r_crop)):
						inverted_profiles_crop_restrict = inverted_profiles_crop[my_time_pos, n_list_all - 4, my_r_pos].flatten()
						recombination_restrict = recombination[n_list_all - 4, my_time_pos, my_r_pos].flatten()
						merge_ne_prof_multipulse_interp_crop_limited_restrict = merge_ne_prof_multipulse_interp_crop_limited[my_time_pos, my_r_pos].flatten()
						merge_Te_prof_multipulse_interp_crop_limited_restrict = merge_Te_prof_multipulse_interp_crop_limited[my_time_pos, my_r_pos].flatten()
						excitation_restrict = excitation[n_list_all - 4, my_time_pos, my_r_pos].flatten()

						if (merge_ne_prof_multipulse_interp_crop_limited_restrict == 0 and merge_Te_prof_multipulse_interp_crop_limited_restrict == 0):
							continue

						dummy = np.zeros((len(n_list_1), len(n_list_all)))
						for value in n_list_1:
							dummy[value - 4][value - 4] = 1

						if len(n_list_1) == 3:
							def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
								def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1, dummy2, dummy3):
									recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
									excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
									dummy_emissivity = np.sum((dummy.T * [dummy1, dummy2, dummy3]).T, axis=0)
									return recombination_emissivity + excitation_emissivity + dummy_emissivity
								return calculated_emission

							bds = [[min_nH_ne, 0, 0, 0, 0], [max_nH_ne, 300, np.inf, np.inf, np.inf]]
							guess = [max_nH_ne, 10, 1000, 1000, 1000]
						elif len(n_list_1) == 2:
							def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
								def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1, dummy2):
									recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
									excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
									dummy_emissivity = np.sum((dummy.T * [dummy1, dummy2]).T, axis=0)
									return recombination_emissivity + excitation_emissivity + dummy_emissivity
								return calculated_emission

							bds = [[min_nH_ne, 0, 0, 0], [max_nH_ne, 300, np.inf, np.inf]]
							guess = [max_nH_ne, 10, 1000, 1000]
						elif len(n_list_1) == 1:
							def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
								def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1):
									recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
									excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
									dummy_emissivity = np.sum((dummy.T * [dummy1]).T, axis=0)
									return recombination_emissivity + excitation_emissivity + dummy_emissivity
								return calculated_emission

							bds = [[min_nH_ne, 0, 0], [max_nH_ne, 300, np.inf]]
							guess = [max_nH_ne, 10, 1000]

						fit = curve_fit(fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict), n_list,inverted_profiles_crop_restrict * OES_multiplier, p0=guess, bounds=bds,sigma=n_weights, maxfev=10000000)

						# nH_ne_all[my_time_pos, my_r_pos]=fit[0][0]
						# h_atomic_density_all[my_time_pos, my_r_pos]=fit[0][1]
						# dummy1_all[my_time_pos, my_r_pos]=fit[0][2]
						# dummy2_all[my_time_pos, my_r_pos]=fit[0][3]
						# dummy3_all[my_time_pos, my_r_pos]=fit[0][4]
						residuals_all[my_time_pos, my_r_pos] = np.sum(((inverted_profiles_crop_restrict * OES_multiplier - fit_nH_ne_h_atomic_density(dummy,recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict)(n_list, *fit[0]))) ** 2)
				return residuals_all.flatten()
			return residuals1


		guess = [1]
		bds = [[0.4], [5]]
		sol = least_squares(residual_ext1(n_list, n_list_1, n_weights, time_crop, r_crop, inverted_profiles_crop, recombination,merge_ne_prof_multipulse_interp_crop_limited, merge_Te_prof_multipulse_interp_crop_limited,excitation), guess, bounds=bds, max_nfev=60, verbose=2, gtol=1e-20, xtol=1e-5, ftol=1e-16,diff_step=0.01, x_scale='jac')
		OES_multiplier = sol.x

		print('OES_multiplier')
		print(OES_multiplier)


		n_list_all = np.sort(np.concatenate((n_list, n_list_1)))
		# OES_multiplier = 0.81414701
		# Te_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
		# ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
		nH_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
		h_atomic_density_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
		dummy1_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
		dummy2_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
		dummy3_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
		residuals_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
		for my_time_pos in range(len(time_crop)):
			for my_r_pos in range(len(r_crop)):
				inverted_profiles_crop_restrict = inverted_profiles_crop[my_time_pos, n_list_all - 4, my_r_pos].flatten()
				recombination_restrict = recombination[n_list_all - 4, my_time_pos, my_r_pos].flatten()
				merge_ne_prof_multipulse_interp_crop_limited_restrict = merge_ne_prof_multipulse_interp_crop_limited[my_time_pos, my_r_pos].flatten()
				merge_Te_prof_multipulse_interp_crop_limited_restrict = merge_Te_prof_multipulse_interp_crop_limited[my_time_pos, my_r_pos].flatten()
				excitation_restrict = excitation[n_list_all - 4, my_time_pos, my_r_pos].flatten()

				if (merge_ne_prof_multipulse_interp_crop_limited_restrict == 0 and merge_Te_prof_multipulse_interp_crop_limited_restrict == 0):
					continue
				# if np.sum(inverted_profiles_crop_restrict == 0)>2:
				# 	continue

				dummy = np.zeros((len(n_list_1), len(n_list_all)))
				for value in n_list_1:
					dummy[value - 4][value - 4] = 1

				if len(n_list_1)==3:
					def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
						def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1, dummy2, dummy3):
							recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
							excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
							dummy_emissivity = np.sum((dummy.T * [dummy1, dummy2, dummy3]).T, axis=0)
							return recombination_emissivity + excitation_emissivity + dummy_emissivity
						return calculated_emission

					bds = [[min_nH_ne, 0, 0,0,0], [max_nH_ne, 300, np.inf, np.inf, np.inf]]
					guess = [max_nH_ne, 10, 1000,1000,1000]
				elif len(n_list_1)==2:
					def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
						def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1, dummy2):
							recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
							excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
							dummy_emissivity = np.sum((dummy.T * [dummy1, dummy2]).T, axis=0)
							return recombination_emissivity + excitation_emissivity + dummy_emissivity
						return calculated_emission

					bds = [[min_nH_ne, 0, 0, 0], [max_nH_ne, 300, np.inf, np.inf]]
					guess = [max_nH_ne, 10, 1000, 1000]
				elif len(n_list_1)==1:
					def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
						def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1):
							recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
							excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
							dummy_emissivity = np.sum((dummy.T * [dummy1]).T, axis=0)
							return recombination_emissivity + excitation_emissivity + dummy_emissivity
						return calculated_emission

					bds = [[min_nH_ne, 0, 0], [max_nH_ne, 300, np.inf]]
					guess = [max_nH_ne, 10, 1000]

				fit = curve_fit(fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict), n_list,inverted_profiles_crop_restrict*OES_multiplier, p0=guess, bounds=bds, sigma=n_weights,maxfev=10000000)

				# plt.figure()
				# plt.plot(n_list_all,inverted_profiles_crop_restrict * OES_multiplier)
				# plt.plot(n_list_all,fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict)(n_list, *fit[0]))
				# # plt.plot(n_list_all,np.ones_like(inverted_profiles_crop_restrict) * OES_multiplier)
				# # plt.plot(n_list_all,fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict,inverted_profiles_crop_restrict)(n_list, *fit[0]))
				# plt.semilogy()
				# plt.title(str(fit[0])+'\n'+str(n_weights))
				# plt.pause(0.01)

				nH_ne_all[my_time_pos, my_r_pos]=fit[0][0]
				h_atomic_density_all[my_time_pos, my_r_pos]=fit[0][1]
				if len(n_list_1)>=1:
					dummy1_all[my_time_pos, my_r_pos]=fit[0][2]
				if len(n_list_1)>=2:
					dummy2_all[my_time_pos, my_r_pos]=fit[0][3]
				if len(n_list_1)>=3:
					dummy3_all[my_time_pos, my_r_pos]=fit[0][4]
				# Te_all[my_time_pos, my_r_pos]=fit[0][3]
				# ne_all[my_time_pos, my_r_pos]=fit[0][4]
				residuals_all[my_time_pos, my_r_pos] = np.nansum(((inverted_profiles_crop_restrict * OES_multiplier - fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict)(n_list, *fit[0]))/(n_weights*inverted_profiles_crop_restrict * OES_multiplier)) ** 2)

		# plt.figure()
		# plt.title('ne,Te,nH_ne,h_atomic_density,OES_multiplier\n'+str([*merge_ne_prof_multipulse_interp_crop_limited_restrict,*merge_Te_prof_multipulse_interp_crop_limited_restrict,*fit[0],OES_multiplier]))
		# plt.plot(n_list,inverted_profiles_crop_restrict*OES_multiplier)
		# plt.plot(n_list,fit_nH_ne_h_atomic_density(recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict)(n_list,*fit[0]))
		# plt.pause(0.01)

		temp_r, temp_t = np.meshgrid(r_crop, time_crop)
		plt.figure();
		plt.pcolor(temp_t, temp_r, nH_ne_all, cmap='rainbow');
		plt.colorbar().set_label('nH/ne [au]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]')
		plt.title('OES_multiplier,spatial_factor,time_shift_factor \n' + str([OES_multiplier,spatial_factor,time_shift_factor]))
		figure_index += 1
		plt.savefig(path_where_to_save_everything +mod+ '/post_process_mega_global_fit' + str(figure_index) + '.eps',bbox_inches='tight')
		plt.close()

		plt.figure();
		plt.pcolor(temp_t, temp_r, h_atomic_density_all, cmap='rainbow');
		plt.colorbar().set_label('neutral atomic hydrogen density [# m^-3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]')
		plt.title('OES_multiplier,spatial_factor,time_shift_factor \n' + str([OES_multiplier,spatial_factor,time_shift_factor]))
		figure_index += 1
		plt.savefig(path_where_to_save_everything +mod + '/post_process_mega_global_fit' + str(figure_index) + '.eps',bbox_inches='tight')
		plt.close()

		plt.figure();
		plt.pcolor(temp_t, temp_r, residuals_all, cmap='rainbow');
		plt.colorbar().set_label('relative residual unaccounted line emission [W m^-3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]')
		plt.title('OES_multiplier,spatial_factor,time_shift_factor \n' + str([OES_multiplier,spatial_factor,time_shift_factor]))
		figure_index += 1
		plt.savefig(path_where_to_save_everything +mod + '/post_process_mega_global_fit' + str(figure_index) + '.eps',bbox_inches='tight')
		plt.close()

		plt.figure();
		plt.pcolor(temp_t, temp_r, dummy1_all, cmap='rainbow');
		plt.colorbar().set_label('extra emissivity n=4 [W m^-3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]')
		plt.title('OES_multiplier,spatial_factor,time_shift_factor \n' + str([OES_multiplier,spatial_factor,time_shift_factor]))
		figure_index += 1
		plt.savefig(path_where_to_save_everything +mod + '/post_process_mega_global_fit' + str(figure_index) + '.eps',bbox_inches='tight')
		plt.close()

		plt.figure();
		plt.pcolor(temp_t, temp_r, dummy2_all, cmap='rainbow');
		plt.colorbar().set_label('extra emissivity n=5 [W m^-3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]')
		plt.title('OES_multiplier,spatial_factor,time_shift_factor \n' + str([OES_multiplier,spatial_factor,time_shift_factor]))
		figure_index += 1
		plt.savefig(path_where_to_save_everything +mod + '/post_process_mega_global_fit' + str(figure_index) + '.eps',bbox_inches='tight')
		plt.close()

		plt.figure();
		plt.pcolor(temp_t, temp_r, dummy3_all, cmap='rainbow');
		plt.colorbar().set_label('extra emissivity n=6 [W m^-3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]')
		plt.title('OES_multiplier,spatial_factor,time_shift_factor \n' + str([OES_multiplier,spatial_factor,time_shift_factor]))
		figure_index += 1
		plt.savefig(path_where_to_save_everything +mod + '/post_process_mega_global_fit' + str(figure_index) + '.eps',bbox_inches='tight')
		plt.close()

		# figure_index += 1
		# plt.savefig(path_where_to_save_everything + mod + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
		# plt.close()

		recombination_emissivity = (recombination * nH_ne_all * ((merge_ne_prof_multipulse_interp_crop_limited * (10 ** 20)) ** 2)).astype('float')
		excitation_emissivity = (excitation * (merge_ne_prof_multipulse_interp_crop_limited * (10 ** 20)) * (h_atomic_density_all * (10 ** 20))).astype('float')

		residual_emission = []
		for index in range(len(recombination_emissivity)):
			residual_emission.append(inverted_profiles_crop[:, index] * OES_multiplier - recombination_emissivity[index] - excitation_emissivity[index])
		residual_emission = np.array(residual_emission)


		plt.figure();
		plt.pcolor(temp_t, temp_r, residual_emission[0], cmap='rainbow', vmin=np.min(
			residual_emission[0][np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
		plt.colorbar().set_label('line emission [W m^-3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]')
		plt.title('residual emissivity, attribuited to MAR, line4\nnH_ne='  + ' , OES_multiplier=' + str(
			OES_multiplier))
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod + '/post_process_' + str(figure_index) + '.eps',
					bbox_inches='tight')
		plt.close()

		temp = residual_emission[0] / recombination_emissivity[0]
		temp[np.logical_not(np.isfinite(temp))] = 0
		plt.figure();
		plt.pcolor(temp_t, temp_r, temp, cmap='rainbow',
				   vmax=np.max(temp[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]),
				   vmin=np.min(temp[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
		plt.colorbar().set_label(' relative line emission [au]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]')
		plt.title(
			'relative residual emissivity, attribuited to MAR, line4\nnH_ne='  + ' , OES_multiplier=' + str(
				OES_multiplier))
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod + '/post_process_' + str(figure_index) + '.eps',
					bbox_inches='tight')
		plt.close()

		plt.figure();
		plt.pcolor(temp_t, temp_r, residual_emission[1], cmap='rainbow', vmin=np.min(
			residual_emission[1][np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
		plt.colorbar().set_label('line emission [W m^-3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]')
		plt.title('residual emissivity, attribuited to MAR, line5\nnH_ne='  + ' , OES_multiplier=' + str(
			OES_multiplier))
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod + '/post_process_' + str(figure_index) + '.eps',
					bbox_inches='tight')
		plt.close()

		temp = residual_emission[1] / recombination_emissivity[1]
		temp[np.logical_not(np.isfinite(temp))] = 0
		plt.figure();
		plt.pcolor(temp_t, temp_r, temp, cmap='rainbow',
				   vmax=np.max(temp[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]),
				   vmin=np.min(temp[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
		plt.colorbar().set_label(' relative line emission [au]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]')
		plt.title(
			'relative residual emissivity, attribuited to MAR, line5\nnH_ne='  + ' , OES_multiplier=' + str(
				OES_multiplier))
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod + '/post_process_' + str(figure_index) + '.eps',
					bbox_inches='tight')
		plt.close()

		plt.figure();
		plt.pcolor(temp_t, temp_r, residual_emission[2], cmap='rainbow', vmin=np.min(
			residual_emission[2][np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
		plt.colorbar().set_label('line emission [W m^-3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]')
		plt.title('residual emissivity, attribuited to MAR, line6\nnH_ne='  + ' , OES_multiplier=' + str(
			OES_multiplier))
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod + '/post_process_' + str(figure_index) + '.eps',
					bbox_inches='tight')
		plt.close()

		temp = residual_emission[2] / recombination_emissivity[2]
		temp[np.logical_not(np.isfinite(temp))] = 0
		plt.figure();
		plt.pcolor(temp_t, temp_r, temp, cmap='rainbow',
				   vmax=np.max(temp[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]),
				   vmin=np.min(temp[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
		plt.colorbar().set_label(' relative line emission [au]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]')
		plt.title(
			'relative residual emissivity, attribuited to MAR, line6\nnH_ne='  + ' , OES_multiplier=' + str(
				OES_multiplier))
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod + '/post_process_' + str(figure_index) + '.eps',
					bbox_inches='tight')
		plt.close()

		temp = read_adf11(scdfile, 'scd', 1, 1, 1, merge_Te_prof_multipulse_interp_crop_limited.flatten(),
						  (merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())
		temp[np.isnan(temp)] = 0
		effective_ionisation_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (
					10 ** -6)  # in ionisations m^-3 s-1 / (# / m^3)**2
		effective_ionisation_rates = (effective_ionisation_rates * (
					merge_ne_prof_multipulse_interp_crop * (10 ** 20)) * h_atomic_density).astype('float')
		plt.figure();
		plt.pcolor(temp_t, temp_r, effective_ionisation_rates, cmap='rainbow');
		plt.colorbar().set_label('effective_ionisation_rates [# m^-3 s-1]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]')
		plt.title('effective_ionisation_rates\nnH_ne='  + ' , OES_multiplier=' + str(OES_multiplier))
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod + '/post_process_' + str(figure_index) + '.eps',
					bbox_inches='tight')
		plt.close()

		temp = read_adf11(acdfile, 'acd', 1, 1, 1, merge_Te_prof_multipulse_interp_crop_limited.flatten(),
						  (merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())
		temp[np.isnan(temp)] = 0
		effective_recombination_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (
					10 ** -6)  # in recombinations m^-3 s-1 / (# / m^3)**2
		effective_recombination_rates = (
					effective_recombination_rates * (merge_ne_prof_multipulse_interp_crop * (10 ** 20)) ** 2).astype(
			'float')
		plt.figure();
		plt.pcolor(temp_t, temp_r, effective_recombination_rates, cmap='rainbow');
		plt.colorbar().set_label('effective_recombination_rates [# m^-3 s-1]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]')
		plt.title('effective_recombination_rates (three body plus radiative)\nnH_ne=' + ' , OES_multiplier=' + str(OES_multiplier))
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod + '/post_process_' + str(figure_index) + '.eps',
					bbox_inches='tight')
		plt.close()
elif False:	#relative intensity fit




	make_plots = True

	for merge_ID_target in [85,86,87,88,89,92,93,94]:

		# merge_ID_target = 85

		for min_nH_ne in [0.3,0.01,0.1,0.6,0.999]:


			figure_index = 0

			# calculate_geometry = False
			# merge_ID_target = 17	#	THIS IS GIVEN BY THE LAUNCHER
			for i in range(10):
				print('.')
			print('Starting to work on merge number'+str(merge_ID_target))
			for i in range(10):
				print('.')


			time_resolution_scan = False
			time_resolution_scan_improved = True
			time_resolution_extra_skip = 0


			n_list = np.array([5,6,7,8 , 9])#, 10, 11, 12])
			n_list_1 = np.array([4])
			n_weights = [1,1,1, 1,3, 1]#, 1, 1, 1000]
			for index in (n_list_1-4):
				n_weights[index]=4
			n_weights[np.max(n_list_1)-3]=2
			# min_nH_ne = 0.6
			max_nH_ne = 1

			mod = '/relative/min_nH_ne'+str(min_nH_ne)+'/n_dummy_'+str(len(n_list_1))

			print('dummy')
			print(n_list_1)
			print('min_nH_ne')
			print(min_nH_ne)
			print('max_nH_ne')
			print(max_nH_ne)
			print('n_weights')
			print(n_weights)

			started=0
			rows_range_for_interp = 25/3 # rows that I use for interpolation (box with twice this side length, not sphere)
			# if merge_ID_target>=66:
			# 	rows_range_for_interp = geom_store['binInterv'][0] / 6
			if time_resolution_scan:
				conventional_time_step = 0.01	# ms
			else:
				conventional_time_step = 0.05	# ms
			# interpolation_type = 'quadratic'	# 'linear' or 'quadratic'
			grade_of_interpolation = 3	#this is the exponent used for the weights of the interpolation for image resampling
			type_of_image = '12bit'	# '12bit' or '16bit'
			# if type_of_image=='12bit':
			row_shift=2*10280/1000000	# ms
			# elif type_of_image=='16bit':
			# 	print('Row shift to be checked')
			# 	exit()
			# time_range_for_interp = rows_range_for_interp*row_shift
			# merge_time_window=[-1,4]
			merge_time_window=[-10,10]
			overexposed_treshold = 3600
			path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)+'_back'

			if not os.path.exists(path_where_to_save_everything + mod):
				os.makedirs(path_where_to_save_everything + mod)


			new_timesteps = np.linspace(merge_time_window[0]+0.5, merge_time_window[1]-0.5,int((merge_time_window[1]-0.5-(merge_time_window[0]+0.5))/conventional_time_step+1))
			dt=np.nanmedian(np.diff(new_timesteps))

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
			energy_difference = np.array([1.89,2.55,2.86,3.03,3.13,3.19,3.23,3.26, 3.29 ])	#eV
			# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
			statistical_weigth = np.array([32,50,72,98,128,162,200,242,288])	#gi-gk
			einstein_coeff = np.array([8.4193e-2,2.53044e-2,9.7320e-3,4.3889e-3,2.2148e-3,1.2156e-3,7.1225e-4,4.3972e-4,2.8337e-4])*1e8	#1/s
			J_to_eV = 6.242e18
			# Used formula 2.3 in Rion Barrois thesys, 2017
			color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise']

			inverted_profiles_original = 4*np.pi*np.load(path_where_to_save_everything+'/inverted_profiles.npy')		# in W m^-3 sr^-1  *  4*pi sr  =  W m^-3
			all_fits = np.load(path_where_to_save_everything +'/merge'+str(merge_ID_target)+'_all_fits.npy')	# in W m^-2 sr^-1
			merge_Te_prof_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_Te_prof_multipulse']
			merge_dTe_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_dTe_multipulse']
			merge_ne_prof_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_ne_prof_multipulse']
			merge_dne_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_dne_multipulse']
			merge_time_original = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_time']

			# for time_shift_factor in time_shift_factor_all:
			#
			# 	mod = '/spatial_factor' + str(spatial_factor)+'/time_shift_factor' + str(time_shift_factor)
			# 	if not os.path.exists(path_where_to_save_everything + mod):
			# 		os.makedirs(path_where_to_save_everything + mod)

			if False:
				def residual_ext(inverted_profiles_original,merge_Te_prof_multipulse,merge_ne_prof_multipulse,merge_time_original):
					def residuals(input):
						spatial_factor=input[0]
						time_shift_factor = input[1]
						print('spatial_factor,time_shift_factor  '+str([spatial_factor,time_shift_factor]))


						dx = (18 / 40 * (50.5 / 27.4) / 1e3) * spatial_factor
						xx = np.arange(40) * dx  # m
						xn = np.linspace(0, max(xx), 1000)
						r = np.linspace(-max(xx) / 2, max(xx) / 2, 1000)
						r=r[::10]
						dr=np.median(np.diff(r))


						merge_time = time_shift_factor + merge_time_original
						inverted_profiles = (1 / spatial_factor) * inverted_profiles_original  # in W m^-3 sr^-1  *  4*pi sr  =  W m^-3
						TS_dt=np.nanmedian(np.diff(merge_time))


						if np.max(merge_Te_prof_multipulse)<=0:
							print('merge'+str(merge_ID_target)+" has no recorded temperature")
							# continue

						TS_size=[-4.149230769230769056e+01,4.416923076923076508e+01]
						TS_r=TS_size[0] + np.linspace(0,1,65)*(TS_size[1]- TS_size[0])
						TS_dr = np.median(np.diff(TS_r))/1000
						gauss = lambda x, A, sig, x0: A * np.exp(-((x - x0) / sig) ** 2)
						profile_centres = []
						profile_centres_score = []
						for index in range(np.shape(merge_Te_prof_multipulse)[0]):
							yy = merge_Te_prof_multipulse[index]
							p0 = [np.max(yy),10,0]
							bds = [[0,-40,np.min(TS_r)],[np.inf,40,np.max(TS_r)]]
							fit = curve_fit(gauss, TS_r, yy, p0, maxfev=100000, bounds=bds)
							profile_centres.append(fit[0][-1])
							profile_centres_score.append(fit[1][-1,-1])
							# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)
						profile_centres = np.array(profile_centres)
						profile_centres_score = np.array(profile_centres_score)
						centre = np.nanmean(profile_centres[profile_centres_score<1])
						TS_r_new = np.abs(TS_r-centre)/1000
						# temp_r, temp_t = np.meshgrid(TS_r_new, merge_time)
						# plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse,vmin=0);plt.colorbar().set_label('Te [eV]');plt.pause(0.01)
						# plt.figure();plt.pcolor(temp_t,temp_r,merge_ne_prof_multipulse,vmin=0);plt.colorbar().set_label('ne [10^20 #/m^3]');plt.pause(0.01)


						temp1=np.zeros_like(inverted_profiles[:,0])
						temp3=np.zeros_like(inverted_profiles[:,0])
						interp_range_t = max(dt/2,TS_dt)*1
						interp_range_r = max(dx/2,TS_dr)*1
						for i_t,value_t in enumerate(new_timesteps):
							if np.sum(np.abs(merge_time-value_t) < interp_range_t) == 0:
								continue
							for i_r,value_r in enumerate(np.abs(r)):
								if np.sum(np.abs(TS_r_new - value_r) < interp_range_r) == 0:
									continue
								temp1[i_t,i_r] = np.mean(merge_Te_prof_multipulse[np.abs(merge_time-value_t)<interp_range_t][:,np.abs(TS_r_new-value_r)<interp_range_r])
								temp3[i_t,i_r] = np.mean(merge_ne_prof_multipulse[np.abs(merge_time-value_t)<interp_range_t][:,np.abs(TS_r_new-value_r)<interp_range_r])

						merge_Te_prof_multipulse_interp=np.array(temp1)
						merge_ne_prof_multipulse_interp=np.array(temp3)
						temp_r, temp_t = np.meshgrid(r, new_timesteps)


						# I crop to the usefull stuff
						start_time = np.abs(new_timesteps-0).argmin()
						end_time = np.abs(new_timesteps-1.5).argmin()+1
						time_crop = new_timesteps[start_time:end_time]
						start_r = np.abs(r-0).argmin()
						end_r = np.abs(r-5).argmin()+1
						r_crop = r[start_r:end_r]
						temp_r, temp_t = np.meshgrid(r_crop, time_crop)
						merge_Te_prof_multipulse_interp_crop = merge_Te_prof_multipulse_interp[start_time:end_time,start_r:end_r]
						merge_ne_prof_multipulse_interp_crop = merge_ne_prof_multipulse_interp[start_time:end_time,start_r:end_r]
						inverted_profiles_crop = inverted_profiles[start_time:end_time,:,start_r:end_r]
						inverted_profiles_crop[np.isnan(inverted_profiles_crop)] = 0
						all_fits_crop = all_fits[start_time:end_time]
						# inverted_profiles_crop[inverted_profiles_crop<0] = 0

						x_local = xx - spatial_factor * 17.4 / 1000
						dr_crop = np.median(np.diff(r_crop))

						merge_Te_prof_multipulse_interp_crop_limited = cp.deepcopy(merge_Te_prof_multipulse_interp_crop)
						merge_Te_prof_multipulse_interp_crop_limited[merge_Te_prof_multipulse_interp_crop_limited<0.2]=0
						merge_ne_prof_multipulse_interp_crop_limited = cp.deepcopy(merge_ne_prof_multipulse_interp_crop)
						merge_ne_prof_multipulse_interp_crop_limited[merge_ne_prof_multipulse_interp_crop_limited<5e-07]=0
						excitation = []
						for isel in [2,3,4,5,6,7,8,9,10]:
							temp = read_adf15(pecfile, isel, merge_Te_prof_multipulse_interp_crop_limited.flatten(), (merge_ne_prof_multipulse_interp_crop_limited*10**(20-6)).flatten())[0]	# ADAS database is in cm^3   # photons s^-1 cm^-3
							temp[np.isnan(temp)] = 0
							temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
							excitation.append(temp)
						excitation = np.array(excitation)	# in # photons cm^-3 s^-1
						excitation = (excitation.T*(10**-6)*(energy_difference/J_to_eV)).T	# in W m^-3 / (# / m^3)**2

						recombination = []
						for isel in [20,21,22,23,24,25,26,27,28]:
							temp = read_adf15(pecfile, isel, merge_Te_prof_multipulse_interp_crop_limited.flatten(), (merge_ne_prof_multipulse_interp_crop_limited*10**(20-6)).flatten())[0]	# ADAS database is in cm^3   # photons s^-1 cm^-3
							temp[np.isnan(temp)] = 0
							temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
							recombination.append(temp)
						recombination = np.array(recombination)	# in # photons cm^-3 s^-1 / (# cm^-3)**2
						recombination = (recombination.T*(10**-6)*(energy_difference/J_to_eV)).T	# in W m^-3 / (# / m^3)**2




						# def residual_ext1(n_list,n_list_1,n_weights,time_crop,r_crop,inverted_profiles_crop,recombination,merge_ne_prof_multipulse_interp_crop_limited,merge_Te_prof_multipulse_interp_crop_limited,excitation):
						# 	def residuals1(input):
						# 		OES_multiplier=input
						# 		print(OES_multiplier)

						n_list_all = np.sort(np.concatenate((n_list, n_list_1)))
						# OES_multiplier = 0.81414701
						# nH_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						# h_atomic_density_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						# dummy1_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						# dummy2_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						# dummy3_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						residuals_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						for my_time_pos in range(len(time_crop)):
							for my_r_pos in range(len(r_crop)):
								inverted_profiles_crop_restrict = inverted_profiles_crop[my_time_pos, n_list_all-4, my_r_pos].flatten()
								recombination_restrict=recombination[n_list_all-4,my_time_pos, my_r_pos].flatten()
								merge_ne_prof_multipulse_interp_crop_limited_restrict=merge_ne_prof_multipulse_interp_crop_limited[my_time_pos, my_r_pos].flatten()
								merge_Te_prof_multipulse_interp_crop_limited_restrict=merge_Te_prof_multipulse_interp_crop_limited[my_time_pos, my_r_pos].flatten()
								excitation_restrict=excitation[n_list_all-4,my_time_pos, my_r_pos].flatten()

								if (merge_ne_prof_multipulse_interp_crop_limited_restrict==0 or merge_Te_prof_multipulse_interp_crop_limited_restrict==0):
									continue

								dummy = np.zeros((len(n_list_1), len(n_list_all)))
								for value in n_list_1:
									dummy[value - 4][value - 4] = 1

								if len(n_list_1) == 3:
									def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
										def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1, dummy2,dummy3):
											recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
											excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
											dummy_emissivity = np.sum((dummy.T * [dummy1, dummy2, dummy3]).T, axis=0)
											total = recombination_emissivity + excitation_emissivity + dummy_emissivity
											return total/(total[0])
										return calculated_emission

									bds = [[min_nH_ne, 0, 0, 0, 0], [max_nH_ne, 300, np.inf, np.inf, np.inf]]
									guess = [max_nH_ne, 10, 1000, 1000, 1000]
								elif len(n_list_1) == 2:
									def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
										def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1, dummy2):
											recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
											excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
											dummy_emissivity = np.sum((dummy.T * [dummy1, dummy2]).T, axis=0)
											total = recombination_emissivity + excitation_emissivity + dummy_emissivity
											return total/(total[0])
										return calculated_emission

									bds = [[min_nH_ne, 0, 0, 0], [max_nH_ne, 300, np.inf, np.inf]]
									guess = [max_nH_ne, 10, 1000, 1000]
								elif len(n_list_1) == 1:
									def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
										def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1):
											recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
											excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
											dummy_emissivity = np.sum((dummy.T * [dummy1]).T, axis=0)
											total = recombination_emissivity + excitation_emissivity + dummy_emissivity
											return total/(total[0])
										return calculated_emission

									bds = [[min_nH_ne, 0, 0], [max_nH_ne, 300, np.inf]]
									guess = [max_nH_ne, 10, 1000]

								fit = curve_fit(fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict), n_list,inverted_profiles_crop_restrict/(inverted_profiles_crop_restrict[0]), p0=guess, bounds=bds,sigma=n_weights, maxfev=10000000)

								# if np.sum(fit[1] == 0) > 0:
								# 	continue

								# plt.figure()
								# plt.plot(n_list_all,inverted_profiles_crop_restrict/(inverted_profiles_crop_restrict[0]))
								# plt.plot(n_list_all,fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict)(n_list, *fit[0]))
								# # plt.plot(n_list_all,np.ones_like(inverted_profiles_crop_restrict) * OES_multiplier)
								# # plt.plot(n_list_all,fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict,inverted_profiles_crop_restrict)(n_list, *fit[0]))
								# plt.semilogy()
								# plt.title(str(fit[0])+'\n'+str(n_weights))
								# plt.pause(0.01)

								# nH_ne_all[my_time_pos, my_r_pos]=fit[0][0]
								# h_atomic_density_all[my_time_pos, my_r_pos]=fit[0][1]
								# dummy1_all[my_time_pos, my_r_pos]=fit[0][2]
								# dummy2_all[my_time_pos, my_r_pos]=fit[0][3]
								# dummy3_all[my_time_pos, my_r_pos]=fit[0][4]
								if np.sum(fit[1]==0)>0:
									residuals_all[my_time_pos, my_r_pos] =np.sum(fit[1]==0)
								else:
									residuals_all[my_time_pos, my_r_pos] = np.sum(((inverted_profiles_crop_restrict/(inverted_profiles_crop_restrict[0]) - fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict, excitation_restrict)(n_list, *fit[0]))) ** 2)
								# if residuals_all[my_time_pos, my_r_pos]>1:
								# 	print([my_time_pos, my_r_pos])
							# 	return residuals_all.flatten()
							# return residuals1


						# guess = [1]
						# bds = [[0.4],[5]]
						# sol = least_squares(residual_ext1(n_list,n_list_1,n_weights,time_crop,r_crop,inverted_profiles_crop,recombination,merge_ne_prof_multipulse_interp_crop_limited,merge_Te_prof_multipulse_interp_crop_limited,excitation), guess, bounds=bds, max_nfev=60,	verbose=2, gtol=1e-20, xtol=1e-5, ftol=1e-16, diff_step=0.01, x_scale='jac')
						# OES_multiplier=sol.x
						return residuals_all.flatten()
					return residuals


				guess = [1.3,-0.05]
				bds = [[1,-0.1], [2,0.1]]
				sol = least_squares(residual_ext(inverted_profiles_original, merge_Te_prof_multipulse, merge_ne_prof_multipulse, merge_time_original), guess, bounds=bds,max_nfev=60, verbose=2, gtol=1e-20, xtol=1e-12, ftol=1e-16, diff_step=[0.05,0.05], x_scale='jac')

				spatial_factor = sol.x[0]
				time_shift_factor = sol.x[1]

				print('spatial_factor,time_shift_factor  ' + str([spatial_factor, time_shift_factor]))

			else:
				for spatial_factor in [1,1.1,1.2,1.3,1.4,1.5]:
					for time_shift_factor in [-0.1,-0.05,0,0.05]:
						mod2=mod + '/spatial_factor_'+str(spatial_factor)+'/time_shift_factor_'+str(time_shift_factor)

						if not os.path.exists(path_where_to_save_everything + mod2):
							os.makedirs(path_where_to_save_everything + mod2)

						dx = (18 / 40 * (50.5 / 27.4) / 1e3) * spatial_factor
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
						# continue

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
						temp3 = np.zeros_like(inverted_profiles[:, 0])
						interp_range_t = max(dt / 2, TS_dt) * 1
						interp_range_r = max(dx / 2, TS_dr) * 1
						for i_t, value_t in enumerate(new_timesteps):
							if np.sum(np.abs(merge_time - value_t) < interp_range_t) == 0:
								continue
							for i_r, value_r in enumerate(np.abs(r)):
								if np.sum(np.abs(TS_r_new - value_r) < interp_range_r) == 0:
									continue
								temp1[i_t, i_r] = np.mean(merge_Te_prof_multipulse[np.abs(merge_time - value_t) < interp_range_t][:,np.abs(TS_r_new - value_r) < interp_range_r])
								temp3[i_t, i_r] = np.mean(merge_ne_prof_multipulse[np.abs(merge_time - value_t) < interp_range_t][:,np.abs(TS_r_new - value_r) < interp_range_r])

						merge_Te_prof_multipulse_interp = np.array(temp1)
						merge_ne_prof_multipulse_interp = np.array(temp3)
						temp_r, temp_t = np.meshgrid(r, new_timesteps)

						# I crop to the usefull stuff
						start_time = np.abs(new_timesteps - 0).argmin()
						end_time = np.abs(new_timesteps - 1.5).argmin() + 1
						time_crop = new_timesteps[start_time:end_time]
						start_r = np.abs(r - 0).argmin()
						end_r = np.abs(r - 5).argmin() + 1
						r_crop = r[start_r:end_r]
						temp_r, temp_t = np.meshgrid(r_crop, time_crop)
						merge_Te_prof_multipulse_interp_crop = merge_Te_prof_multipulse_interp[start_time:end_time, start_r:end_r]
						merge_ne_prof_multipulse_interp_crop = merge_ne_prof_multipulse_interp[start_time:end_time, start_r:end_r]
						inverted_profiles_crop = inverted_profiles[start_time:end_time, :, start_r:end_r]
						inverted_profiles_crop[np.isnan(inverted_profiles_crop)] = 0
						all_fits_crop = all_fits[start_time:end_time]
						# inverted_profiles_crop[inverted_profiles_crop<0] = 0

						x_local = xx - spatial_factor * 17.4 / 1000
						dr_crop = np.median(np.diff(r_crop))

						merge_Te_prof_multipulse_interp_crop_limited = cp.deepcopy(merge_Te_prof_multipulse_interp_crop)
						merge_Te_prof_multipulse_interp_crop_limited[merge_Te_prof_multipulse_interp_crop_limited < 0.2] = 0
						merge_ne_prof_multipulse_interp_crop_limited = cp.deepcopy(merge_ne_prof_multipulse_interp_crop)
						merge_ne_prof_multipulse_interp_crop_limited[merge_ne_prof_multipulse_interp_crop_limited < 5e-07] = 0
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



						n_list_all = np.sort(np.concatenate((n_list, n_list_1)))
						# OES_multiplier = 0.81414701
						# Te_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						# ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						nH_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						h_atomic_density_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						dummy1_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						dummy2_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						dummy3_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						residuals_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
						for my_time_pos in range(len(time_crop)):
							for my_r_pos in range(len(r_crop)):
								inverted_profiles_crop_restrict = inverted_profiles_crop[my_time_pos, n_list_all - 4, my_r_pos].flatten()
								recombination_restrict = recombination[n_list_all - 4, my_time_pos, my_r_pos].flatten()
								merge_ne_prof_multipulse_interp_crop_limited_restrict = merge_ne_prof_multipulse_interp_crop_limited[my_time_pos, my_r_pos].flatten()
								merge_Te_prof_multipulse_interp_crop_limited_restrict = merge_Te_prof_multipulse_interp_crop_limited[my_time_pos, my_r_pos].flatten()
								excitation_restrict = excitation[n_list_all - 4, my_time_pos, my_r_pos].flatten()

								if (merge_ne_prof_multipulse_interp_crop_limited_restrict == 0 or merge_Te_prof_multipulse_interp_crop_limited_restrict == 0):
									continue
								# if np.sum(inverted_profiles_crop_restrict == 0)>2:
								# 	continue

								dummy = np.zeros((len(n_list_1), len(n_list_all)))
								for value in n_list_1:
									dummy[value - 4][value - 4] = 1

								if len(n_list_1)==3:
									def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
										def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1, dummy2, dummy3):
											recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
											excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
											dummy_emissivity = np.sum((dummy.T * [dummy1, dummy2, dummy3]).T, axis=0)
											total = recombination_emissivity + excitation_emissivity + dummy_emissivity
											return total / (total[0])
										return calculated_emission

									bds = [[min_nH_ne, 0, 0,0,0], [max_nH_ne, 300, np.inf, np.inf, np.inf]]
									guess = [max_nH_ne, 10, 1000,1000,1000]
								elif len(n_list_1)==2:
									def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
										def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1, dummy2):
											recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
											excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
											dummy_emissivity = np.sum((dummy.T * [dummy1, dummy2]).T, axis=0)
											total = recombination_emissivity + excitation_emissivity + dummy_emissivity
											return total / (total[0])
										return calculated_emission

									bds = [[min_nH_ne, 0, 0, 0], [max_nH_ne, 300, np.inf, np.inf]]
									guess = [max_nH_ne, 10, 1000, 1000]
								elif len(n_list_1)==1:
									def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
										def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1):
											recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
											excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
											dummy_emissivity = np.sum((dummy.T * [dummy1]).T, axis=0)
											total = recombination_emissivity + excitation_emissivity + dummy_emissivity
											return total / (total[0])
										return calculated_emission

									bds = [[min_nH_ne, 0, 0], [max_nH_ne, 300, np.inf]]
									guess = [max_nH_ne, 1, 0]

								fit = curve_fit(fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict), n_list,inverted_profiles_crop_restrict/(inverted_profiles_crop_restrict[0]), p0=guess, bounds=bds, sigma=n_weights,maxfev=10000000)

								# if np.sum(fit[1] == 0)>0:
								# 	continue

								# plt.figure()
								# plt.plot(n_list_all,inverted_profiles_crop_restrict/inverted_profiles_crop_restrict[0])
								# plt.plot(n_list_all,fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict)(n_list, *fit[0]))
								# # plt.plot(n_list_all,np.ones_like(inverted_profiles_crop_restrict) * OES_multiplier)
								# # plt.plot(n_list_all,fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict,inverted_profiles_crop_restrict)(n_list, *fit[0]))
								# plt.semilogy()
								# plt.title(str(fit[0])+'\n'+str(n_weights))
								# plt.pause(0.01)

								nH_ne_all[my_time_pos, my_r_pos]=fit[0][0]
								h_atomic_density_all[my_time_pos, my_r_pos]=fit[0][1]
								if len(n_list_1)>=1:
									dummy1_all[my_time_pos, my_r_pos]=fit[0][2]
								if len(n_list_1)>=2:
									dummy2_all[my_time_pos, my_r_pos]=fit[0][3]
								if len(n_list_1)>=3:
									dummy3_all[my_time_pos, my_r_pos]=fit[0][4]
								# Te_all[my_time_pos, my_r_pos]=fit[0][3]
								# ne_all[my_time_pos, my_r_pos]=fit[0][4]
								# if np.sum(fit[1] == 0) == len(fit[0]) ** 2:
								# 	residuals_all[my_time_pos, my_r_pos] = 0
								# else:
								residuals_all[my_time_pos, my_r_pos] = np.sum(((inverted_profiles_crop_restrict / (inverted_profiles_crop_restrict[0]) - fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict)(n_list, *fit[0]))) ** 2)

						# plt.figure()
						# plt.title('ne,Te,nH_ne,h_atomic_density,OES_multiplier\n'+str([*merge_ne_prof_multipulse_interp_crop_limited_restrict,*merge_Te_prof_multipulse_interp_crop_limited_restrict,*fit[0],OES_multiplier]))
						# plt.plot(n_list,inverted_profiles_crop_restrict*OES_multiplier)
						# plt.plot(n_list,fit_nH_ne_h_atomic_density(recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict)(n_list,*fit[0]))
						# plt.pause(0.01)

						figure_index = 0

						temp_r, temp_t = np.meshgrid(r_crop, time_crop)
						plt.figure();
						plt.pcolor(temp_t, temp_r, nH_ne_all, cmap='rainbow');
						plt.colorbar().set_label('nH/ne [au]')  # ;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('OES_multiplier,spatial_factor,time_shift_factor \n' + str([np.nan,spatial_factor,time_shift_factor]))
						figure_index += 1
						plt.savefig(path_where_to_save_everything +mod2+ '/post_process_mega_global_fit' + str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						plt.figure();
						plt.pcolor(temp_t, temp_r, h_atomic_density_all, cmap='rainbow');
						plt.colorbar().set_label('neutral atomic hydrogen density [# m^-3]')  # ;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('OES_multiplier,spatial_factor,time_shift_factor \n' + str([np.nan,spatial_factor,time_shift_factor]))
						figure_index += 1
						plt.savefig(path_where_to_save_everything +mod2 + '/post_process_mega_global_fit' + str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						plt.figure();
						plt.pcolor(temp_t, temp_r, residuals_all, cmap='rainbow',vmax=np.max(residuals_all[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
						plt.colorbar().set_label('relative residual unaccounted line emission [W m^-3]')  # ;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('OES_multiplier,spatial_factor,time_shift_factor \n' + str([np.nan,spatial_factor,time_shift_factor])+'\nsum = '+str(np.sum(residuals_all)))
						figure_index += 1
						plt.savefig(path_where_to_save_everything +mod2 + '/post_process_mega_global_fit' + str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						plt.figure();
						plt.pcolor(temp_t, temp_r, dummy1_all, cmap='rainbow',vmax=np.max(dummy1_all[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]),vmin=np.min(dummy1_all[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
						plt.colorbar().set_label('extra emissivity n=4 [W m^-3]')  # ;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('OES_multiplier,spatial_factor,time_shift_factor \n' + str([np.nan,spatial_factor,time_shift_factor]))
						figure_index += 1
						plt.savefig(path_where_to_save_everything +mod2 + '/post_process_mega_global_fit' + str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						plt.figure();
						plt.pcolor(temp_t, temp_r, dummy2_all, cmap='rainbow');
						plt.colorbar().set_label('extra emissivity n=5 [W m^-3]')  # ;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('OES_multiplier,spatial_factor,time_shift_factor \n' + str([np.nan,spatial_factor,time_shift_factor]))
						figure_index += 1
						plt.savefig(path_where_to_save_everything +mod2 + '/post_process_mega_global_fit' + str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						plt.figure();
						plt.pcolor(temp_t, temp_r, dummy3_all, cmap='rainbow');
						plt.colorbar().set_label('extra emissivity n=6 [W m^-3]')  # ;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('OES_multiplier,spatial_factor,time_shift_factor \n' + str([np.nan,spatial_factor,time_shift_factor]))
						figure_index += 1
						plt.savefig(path_where_to_save_everything +mod2 + '/post_process_mega_global_fit' + str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						# figure_index += 1
						# plt.savefig(path_where_to_save_everything + mod + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
						# plt.close()

						recombination_emissivity = (recombination * nH_ne_all * ((merge_ne_prof_multipulse_interp_crop_limited * (10 ** 20)) ** 2)).astype('float')
						excitation_emissivity = (excitation * (merge_ne_prof_multipulse_interp_crop_limited * (10 ** 20)) * (h_atomic_density_all * (10 ** 20))).astype('float')

						residual_emission = []
						for index in range(len(recombination_emissivity)):
							residual_emission.append(inverted_profiles_crop[:, index] - recombination_emissivity[index] - excitation_emissivity[index])
						residual_emission = np.array(residual_emission)

						to_print = inverted_profiles_crop[:, 0]*dummy1_all/(recombination_emissivity[0]+excitation_emissivity[0]+ dummy1_all)
						plt.figure();
						plt.pcolor(temp_t, temp_r, to_print, cmap='rainbow',vmax=np.max(to_print[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]),vmin=np.min(dummy1_all[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
						plt.colorbar().set_label('extra emissivity n=4 [W m^-3]')  # ;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('extra emissivity using only the ratio from the fitting\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([np.nan,spatial_factor,time_shift_factor]))
						figure_index += 1
						plt.savefig(path_where_to_save_everything +mod2 + '/post_process_mega_global_fit' + str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()
						figure_index -= 1


						plt.figure();
						plt.pcolor(temp_t, temp_r, residual_emission[0], cmap='rainbow', vmin=np.min(
							residual_emission[0][np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
						plt.colorbar().set_label('line emission [W m^-3]')  # ;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('residual emissivity, attribuited to MAR, line4\nOES_multiplier=' + str(
							np.nan))
						figure_index += 1
						plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',
									bbox_inches='tight')
						plt.close()

						temp = residual_emission[0] / recombination_emissivity[0]
						temp[np.logical_not(np.isfinite(temp))] = 0
						plt.figure();
						plt.pcolor(temp_t, temp_r, temp, cmap='rainbow',
								   vmax=np.max(temp[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]),
								   vmin=np.min(temp[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
						plt.colorbar().set_label(' relative line emission [au]')  # ;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title(
							'relative residual emissivity, attribuited to MAR, line4\nOES_multiplier=' + str(
								np.nan))
						figure_index += 1
						plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',
									bbox_inches='tight')
						plt.close()

						plt.figure();
						plt.pcolor(temp_t, temp_r, residual_emission[0]-dummy1_all, cmap='rainbow', vmin=np.min(
							residual_emission[0][np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
						plt.colorbar().set_label('line emission [W m^-3]')  # ;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('residual emissivity not included in the fitting, line4\nOES_multiplier=' + str(
							np.nan)+'\nsum = '+str(np.sum(np.abs(residual_emission[0]-dummy1_all))))
						figure_index += 1
						plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',
									bbox_inches='tight')
						plt.close()


						plt.figure();
						plt.pcolor(temp_t, temp_r, residual_emission[1], cmap='rainbow', vmin=np.min(
							residual_emission[1][np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
						plt.colorbar().set_label('line emission [W m^-3]')  # ;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('residual emissivity, attribuited to MAR, line5\nOES_multiplier=' + str(
							np.nan))
						figure_index += 1
						plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',
									bbox_inches='tight')
						plt.close()

						temp = residual_emission[1] / recombination_emissivity[1]
						temp[np.logical_not(np.isfinite(temp))] = 0
						plt.figure();
						plt.pcolor(temp_t, temp_r, temp, cmap='rainbow',
								   vmax=np.max(temp[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]),
								   vmin=np.min(temp[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
						plt.colorbar().set_label(' relative line emission [au]')  # ;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title(
							'relative residual emissivity, attribuited to MAR, line5\nOES_multiplier=' + str(
								np.nan))
						figure_index += 1
						plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',
									bbox_inches='tight')
						plt.close()

						if len(n_list_1)>1:
							plt.figure();
							plt.pcolor(temp_t, temp_r, residual_emission[1]-dummy2_all, cmap='rainbow', vmin=np.min(
								residual_emission[1][np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
							plt.colorbar().set_label('line emission [W m^-3]')  # ;plt.pause(0.01)
							plt.axes().set_aspect(20)
							plt.xlabel('time [ms]')
							plt.ylabel('radial location [m]')
							plt.title('residual emissivity not included in the fitting, line5\nOES_multiplier=' + str(
								np.nan)+'\nsum = '+str(np.sum(np.abs(residual_emission[1]-dummy1_all))))
							figure_index += 1
							plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',
										bbox_inches='tight')
							plt.close()


						plt.figure();
						plt.pcolor(temp_t, temp_r, residual_emission[2], cmap='rainbow', vmin=np.min(
							residual_emission[2][np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
						plt.colorbar().set_label('line emission [W m^-3]')  # ;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('residual emissivity, attribuited to MAR, line6\nOES_multiplier=' + str(
							np.nan))
						figure_index += 1
						plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',
									bbox_inches='tight')
						plt.close()

						temp = residual_emission[2] / recombination_emissivity[2]
						temp[np.logical_not(np.isfinite(temp))] = 0
						plt.figure();
						plt.pcolor(temp_t, temp_r, temp, cmap='rainbow',
								   vmax=np.max(temp[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]),
								   vmin=np.min(temp[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
						plt.colorbar().set_label(' relative line emission [au]')  # ;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title(
							'relative residual emissivity, attribuited to MAR, line6\nOES_multiplier=' + str(
								np.nan))
						figure_index += 1
						plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',
									bbox_inches='tight')
						plt.close()

						if len(n_list_1)>2:
							plt.figure();
							plt.pcolor(temp_t, temp_r, residual_emission[2]-dummy3_all, cmap='rainbow', vmin=np.min(
								residual_emission[2][np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
							plt.colorbar().set_label('line emission [W m^-3]')  # ;plt.pause(0.01)
							plt.axes().set_aspect(20)
							plt.xlabel('time [ms]')
							plt.ylabel('radial location [m]')
							plt.title('residual emissivity not included in the fitting, line6\nOES_multiplier=' + str(
								np.nan)+'\nsum = '+str(np.sum(np.abs(residual_emission[2]-dummy1_all))))
							figure_index += 1
							plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',
										bbox_inches='tight')
							plt.close()


						temp = read_adf11(scdfile, 'scd', 1, 1, 1, merge_Te_prof_multipulse_interp_crop_limited.flatten(),
										  (merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())
						temp[np.isnan(temp)] = 0
						effective_ionisation_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (
									10 ** -6)  # in ionisations m^-3 s-1 / (# / m^3)**2
						effective_ionisation_rates = (effective_ionisation_rates * (
									merge_ne_prof_multipulse_interp_crop * (10 ** 20)) * h_atomic_density_all).astype('float')
						plt.figure();
						plt.pcolor(temp_t, temp_r, effective_ionisation_rates, cmap='rainbow');
						plt.colorbar().set_label('effective_ionisation_rates [# m^-3 s-1]')  # ;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('effective_ionisation_rates\nOES_multiplier=' + str(np.nan))
						figure_index += 1
						plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',
									bbox_inches='tight')
						plt.close()

						temp = read_adf11(acdfile, 'acd', 1, 1, 1, merge_Te_prof_multipulse_interp_crop_limited.flatten(),
										  (merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())
						temp[np.isnan(temp)] = 0
						effective_recombination_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (
									10 ** -6)  # in recombinations m^-3 s-1 / (# / m^3)**2
						effective_recombination_rates = (
									effective_recombination_rates *nH_ne_all* (merge_ne_prof_multipulse_interp_crop * (10 ** 20)) ** 2).astype(
							'float')
						plt.figure();
						plt.pcolor(temp_t, temp_r, effective_recombination_rates, cmap='rainbow');
						plt.colorbar().set_label('effective_recombination_rates [# m^-3 s-1]')  # ;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('effective_recombination_rates (three body plus radiative)\nOES_multiplier=' + str(np.nan))
						figure_index += 1
						plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',
									bbox_inches='tight')
						plt.close()


						plt.figure();
						plt.pcolor(temp_t, temp_r, merge_ne_prof_multipulse_interp_crop_limited, cmap='rainbow');
						plt.colorbar().set_label('electron density [# m^-3]')  # ;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('Scaled electron density')
						figure_index += 1
						plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',
									bbox_inches='tight')
						plt.close()

						plt.figure();
						plt.pcolor(temp_t, temp_r, merge_Te_prof_multipulse_interp_crop_limited, cmap='rainbow');
						plt.colorbar().set_label('electron temperature [eV]')  # ;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('Scaled electron temperature')
						figure_index += 1
						plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',
									bbox_inches='tight')
						plt.close()

						plt.figure();
						plt.pcolor(temp_t, temp_r, inverted_profiles_crop[:, 0], cmap='rainbow');
						plt.colorbar().set_label('line emission n=4 [w m^-3]')  # ;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('Scaled OES line emissivity')
						figure_index += 1
						plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',
									bbox_inches='tight')
						plt.close()

						plt.figure();
						plt.pcolor(temp_t, temp_r, inverted_profiles_crop[:, 3], cmap='rainbow');
						plt.colorbar().set_label('line emission n=7 [w m^-3]')  # ;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('Scaled OES line emissivity')
						figure_index += 1
						plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',
									bbox_inches='tight')
						plt.close()
elif True:	#absolute intensity fit




	make_plots = True

	# for merge_ID_target in [851,86,87,89,92]:	# 88 excluded because I don't have a temperature profile
	# 	merge_time_window=[-10,10]
	# for merge_ID_target in np.flip([93, 94],axis=0):	# 88 excluded because I don't have a temperature profile
	# 	merge_time_window=[-1,2]
		# merge_ID_target = 85

	merge_ID_target_multipulse = [851,86,87,89,92, 93, 94]

	for merge_ID_target in merge_ID_target_multipulse:  # 88 excluded because I don't have a temperature profile
		if merge_ID_target>=93:
			merge_time_window = [-1,2]
		else:
			merge_time_window = [-10,10]


		recorded_data_override = False

		# for min_nH_ne in [0.3,0.01,0.1,0.6,0.999,0.9999]:	#0.9999 is for the case when I relax this boundary if nH<0.001
		for min_nHp_ne in [0,0.3,0.999]:
			# for max_h_atomic_density in [300,np.inf]:
			for max_nH_ne_atomic_density in [1,np.inf]:


				figure_index = 0

				# calculate_geometry = False
				# merge_ID_target = 17	#	THIS IS GIVEN BY THE LAUNCHER
				for i in range(10):
					print('.')
				print('Starting to work on merge number'+str(merge_ID_target))
				for i in range(10):
					print('.')


				time_resolution_scan = False
				time_resolution_scan_improved = True
				time_resolution_extra_skip = 0

				if False:
					complete_line_list = np.array([4,5,6,7,8, 9,10,11])
					n_weights = np.array([1, 1, 1, 1, 1, 1,3,3])
				else:
					complete_line_list = np.array([4,5,6,7,8, 9])
					n_weights = np.array([1, 1, 1, 1, 1, 3])

				for number_of_dummy in [1,2,3,4,5]:
					n_list = complete_line_list[number_of_dummy:]
					n_list_1 = complete_line_list[:number_of_dummy]

					# if False:
					# 	n_list = np.array([5,6,7, 9,10,11])
					# 	n_list_1 = np.array([4])
					# 	n_weights = [1, 1, 1, 1, 1,1,1]
					# else:
					# 	n_list = np.array([ 7,8, 9])
					# 	n_list_1 = np.array([4,5,6])
					# 	n_weights = [1, 1, 1, 1,10, 1]
					# for index in (n_list_1 - 4):
					# 	n_weights[index] = 4
					# n_weights[np.max(n_list_1) - 3] = 2
					# min_nH_ne = 0.6
					max_nHp_ne = 1

					mod = '/absolute/lines_fitted'+str(len(n_list)+len(n_list_1))+'/min_nHp_ne'+str(min_nHp_ne)+'/max_nH_ne_atomic_density'+str(max_nH_ne_atomic_density)+'/n_dummy_'+str(len(n_list_1))

					boltzmann_constant_eV = 8.617333262145e-5	# eV/K
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
					max_nH2_from_pressure = target_chamber_pressure*avogadro_number/(boltzmann_constant_eV*300)	# [#/m^3] I suppose ambient temp is ~ 300K


					print('dummy')
					print(n_list_1)
					print('min_nHp_ne')
					print(min_nHp_ne)
					print('max_nH_ne_atomic_density')
					print(max_nH_ne_atomic_density)
					print('max_nHp_ne')
					print(max_nHp_ne)
					print('n_weights')
					print(n_weights)
					print('recorded data override is '+str(recorded_data_override))

					started=0
					rows_range_for_interp = 25/3 # rows that I use for interpolation (box with twice this side length, not sphere)
					# if merge_ID_target>=66:
					# 	rows_range_for_interp = geom_store['binInterv'][0] / 6
					if time_resolution_scan:
						conventional_time_step = 0.01	# ms
					else:
						conventional_time_step = 0.05	# ms
					# interpolation_type = 'quadratic'	# 'linear' or 'quadratic'
					grade_of_interpolation = 3	#this is the exponent used for the weights of the interpolation for image resampling
					type_of_image = '12bit'	# '12bit' or '16bit'
					# if type_of_image=='12bit':
					row_shift=2*10280/1000000	# ms
					# elif type_of_image=='16bit':
					# 	print('Row shift to be checked')
					# 	exit()
					# time_range_for_interp = rows_range_for_interp*row_shift
					# merge_time_window=[-1,4]
					# merge_time_window=[-10,10]
					overexposed_treshold = 3600
					path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)

					if not os.path.exists(path_where_to_save_everything + mod):
						os.makedirs(path_where_to_save_everything + mod)


					new_timesteps = np.linspace(merge_time_window[0]+0.5, merge_time_window[1]-0.5,int((merge_time_window[1]-0.5-(merge_time_window[0]+0.5))/conventional_time_step+1))
					dt=np.nanmedian(np.diff(new_timesteps))

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
					energy_difference = np.array([1.89,2.55,2.86,3.03,3.13,3.19,3.23,3.26, 3.29 ])	#eV
					# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
					statistical_weigth = np.array([32,50,72,98,128,162,200,242,288])	#gi-gk
					einstein_coeff = np.array([8.4193e-2,2.53044e-2,9.7320e-3,4.3889e-3,2.2148e-3,1.2156e-3,7.1225e-4,4.3972e-4,2.8337e-4])*1e8	#1/s
					J_to_eV = 6.242e18
					au_to_kg = 1.66053906660e-27	# kg/au
					boltzmann_constant_J = 1.380649e-23	# J/K
					# Used formula 2.3 in Rion Barrois thesys, 2017
					color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise']

					inverted_profiles_original = 4*np.pi*np.load(path_where_to_save_everything+'/inverted_profiles.npy')		# in W m^-3 sr^-1  *  4*pi sr  =  W m^-3
					all_fits = np.load(path_where_to_save_everything +'/merge'+str(merge_ID_target)+'_all_fits.npy')	# in W m^-2 sr^-1
					merge_Te_prof_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_Te_prof_multipulse']
					merge_dTe_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_dTe_multipulse']
					merge_ne_prof_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_ne_prof_multipulse']
					merge_dne_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_dne_multipulse']
					merge_time_original = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_time']

					# for time_shift_factor in time_shift_factor_all:
					#
					# 	mod = '/spatial_factor' + str(spatial_factor)+'/time_shift_factor' + str(time_shift_factor)
					# 	if not os.path.exists(path_where_to_save_everything + mod):
					# 		os.makedirs(path_where_to_save_everything + mod)

					if False:
						def residual_ext(inverted_profiles_original,merge_Te_prof_multipulse,merge_ne_prof_multipulse,merge_time_original):
							def residuals(input):
								spatial_factor=input[0]
								time_shift_factor = input[1]
								print('spatial_factor,time_shift_factor  '+str([spatial_factor,time_shift_factor]))


								dx = (18 / 40 * (50.5 / 27.4) / 1e3) * spatial_factor
								xx = np.arange(40) * dx  # m
								xn = np.linspace(0, max(xx), 1000)
								r = np.linspace(-max(xx) / 2, max(xx) / 2, 1000)
								r=r[::10]
								dr=np.median(np.diff(r))


								merge_time = time_shift_factor + merge_time_original
								inverted_profiles = (1 / spatial_factor) * inverted_profiles_original  # in W m^-3 sr^-1  *  4*pi sr  =  W m^-3
								TS_dt=np.nanmedian(np.diff(merge_time))


								if np.max(merge_Te_prof_multipulse)<=0:
									print('merge'+str(merge_ID_target)+" has no recorded temperature")
									# continue

								TS_size=[-4.149230769230769056e+01,4.416923076923076508e+01]
								TS_r=TS_size[0] + np.linspace(0,1,65)*(TS_size[1]- TS_size[0])
								TS_dr = np.median(np.diff(TS_r))/1000
								gauss = lambda x, A, sig, x0: A * np.exp(-((x - x0) / sig) ** 2)
								profile_centres = []
								profile_centres_score = []
								for index in range(np.shape(merge_Te_prof_multipulse)[0]):
									yy = merge_Te_prof_multipulse[index]
									p0 = [np.max(yy),10,0]
									bds = [[0,-40,np.min(TS_r)],[np.inf,40,np.max(TS_r)]]
									fit = curve_fit(gauss, TS_r, yy, p0, maxfev=100000, bounds=bds)
									profile_centres.append(fit[0][-1])
									profile_centres_score.append(fit[1][-1,-1])
									# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)
								profile_centres = np.array(profile_centres)
								profile_centres_score = np.array(profile_centres_score)
								centre = np.nanmean(profile_centres[profile_centres_score<1])
								TS_r_new = np.abs(TS_r-centre)/1000
								# temp_r, temp_t = np.meshgrid(TS_r_new, merge_time)
								# plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse,vmin=0);plt.colorbar().set_label('Te [eV]');plt.pause(0.01)
								# plt.figure();plt.pcolor(temp_t,temp_r,merge_ne_prof_multipulse,vmin=0);plt.colorbar().set_label('ne [10^20 #/m^3]');plt.pause(0.01)


								temp1=np.zeros_like(inverted_profiles[:,0])
								temp3=np.zeros_like(inverted_profiles[:,0])
								interp_range_t = max(dt/2,TS_dt)*1
								interp_range_r = max(dx/2,TS_dr)*1
								for i_t,value_t in enumerate(new_timesteps):
									if np.sum(np.abs(merge_time-value_t) < interp_range_t) == 0:
										continue
									for i_r,value_r in enumerate(np.abs(r)):
										if np.sum(np.abs(TS_r_new - value_r) < interp_range_r) == 0:
											continue
										if np.sum(np.logical_and(np.abs(merge_time-value_t)<interp_range_t,np.sum(merge_Te_prof_multipulse,axis=1)>0))==0:
											continue
										temp1[i_t,i_r] = np.mean(merge_Te_prof_multipulse[np.logical_and(np.abs(merge_time-value_t)<interp_range_t,np.sum(merge_Te_prof_multipulse,axis=1)>0)][:,np.abs(TS_r_new-value_r)<interp_range_r])
										temp3[i_t,i_r] = np.mean(merge_ne_prof_multipulse[np.logical_and(np.abs(merge_time-value_t)<interp_range_t,np.sum(merge_Te_prof_multipulse,axis=1)>0)][:,np.abs(TS_r_new-value_r)<interp_range_r])

								merge_Te_prof_multipulse_interp=np.array(temp1)
								merge_ne_prof_multipulse_interp=np.array(temp3)
								temp_r, temp_t = np.meshgrid(r, new_timesteps)


								# I crop to the usefull stuff
								start_time = np.abs(new_timesteps-0).argmin()
								end_time = np.abs(new_timesteps-1.5).argmin()+1
								time_crop = new_timesteps[start_time:end_time]
								start_r = np.abs(r-0).argmin()
								end_r = np.abs(r-5).argmin()+1
								r_crop = r[start_r:end_r]
								temp_r, temp_t = np.meshgrid(r_crop, time_crop)
								merge_Te_prof_multipulse_interp_crop = merge_Te_prof_multipulse_interp[start_time:end_time,start_r:end_r]
								merge_ne_prof_multipulse_interp_crop = merge_ne_prof_multipulse_interp[start_time:end_time,start_r:end_r]
								inverted_profiles_crop = inverted_profiles[start_time:end_time,:,start_r:end_r]
								inverted_profiles_crop[np.isnan(inverted_profiles_crop)] = 0
								all_fits_crop = all_fits[start_time:end_time]
								# inverted_profiles_crop[inverted_profiles_crop<0] = 0

								x_local = xx - spatial_factor * 17.4 / 1000
								dr_crop = np.median(np.diff(r_crop))

								merge_Te_prof_multipulse_interp_crop_limited = cp.deepcopy(merge_Te_prof_multipulse_interp_crop)
								merge_Te_prof_multipulse_interp_crop_limited[merge_Te_prof_multipulse_interp_crop_limited<0.2]=0
								merge_ne_prof_multipulse_interp_crop_limited = cp.deepcopy(merge_ne_prof_multipulse_interp_crop)
								merge_ne_prof_multipulse_interp_crop_limited[merge_ne_prof_multipulse_interp_crop_limited<5e-07]=0
								excitation = []
								for isel in [2,3,4,5,6,7,8,9,10]:
									temp = read_adf15(pecfile, isel, merge_Te_prof_multipulse_interp_crop_limited.flatten(), (merge_ne_prof_multipulse_interp_crop_limited*10**(20-6)).flatten())[0]	# ADAS database is in cm^3   # photons s^-1 cm^-3
									temp[np.isnan(temp)] = 0
									temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
									excitation.append(temp)
								excitation = np.array(excitation)	# in # photons cm^-3 s^-1
								excitation = (excitation.T*(10**-6)*(energy_difference/J_to_eV)).T	# in W m^-3 / (# / m^3)**2

								recombination = []
								for isel in [20,21,22,23,24,25,26,27,28]:
									temp = read_adf15(pecfile, isel, merge_Te_prof_multipulse_interp_crop_limited.flatten(), (merge_ne_prof_multipulse_interp_crop_limited*10**(20-6)).flatten())[0]	# ADAS database is in cm^3   # photons s^-1 cm^-3
									temp[np.isnan(temp)] = 0
									temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
									recombination.append(temp)
								recombination = np.array(recombination)	# in # photons cm^-3 s^-1 / (# cm^-3)**2
								recombination = (recombination.T*(10**-6)*(energy_difference/J_to_eV)).T	# in W m^-3 / (# / m^3)**2




								# def residual_ext1(n_list,n_list_1,n_weights,time_crop,r_crop,inverted_profiles_crop,recombination,merge_ne_prof_multipulse_interp_crop_limited,merge_Te_prof_multipulse_interp_crop_limited,excitation):
								# 	def residuals1(input):
								# 		OES_multiplier=input
								# 		print(OES_multiplier)

								n_list_all = np.sort(np.concatenate((n_list, n_list_1)))
								# OES_multiplier = 0.81414701
								# nH_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
								# h_atomic_density_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
								# dummy1_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
								# dummy2_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
								# dummy3_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
								residuals_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
								for my_time_pos in range(len(time_crop)):
									for my_r_pos in range(len(r_crop)):
										inverted_profiles_crop_restrict = inverted_profiles_crop[my_time_pos, n_list_all-4, my_r_pos].flatten()
										recombination_restrict=recombination[n_list_all-4,my_time_pos, my_r_pos].flatten()
										merge_ne_prof_multipulse_interp_crop_limited_restrict=merge_ne_prof_multipulse_interp_crop_limited[my_time_pos, my_r_pos].flatten()
										merge_Te_prof_multipulse_interp_crop_limited_restrict=merge_Te_prof_multipulse_interp_crop_limited[my_time_pos, my_r_pos].flatten()
										excitation_restrict=excitation[n_list_all-4,my_time_pos, my_r_pos].flatten()

										if (merge_ne_prof_multipulse_interp_crop_limited_restrict==0 or merge_Te_prof_multipulse_interp_crop_limited_restrict==0):
											continue

										dummy = np.zeros((len(n_list_1), len(n_list_all)))
										for value in n_list_1:
											dummy[value - 4][value - 4] = 1

										if len(n_list_1) == 3:
											def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
												def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1, dummy2,dummy3):
													recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
													excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
													dummy_emissivity = np.sum((dummy.T * [dummy1, dummy2, dummy3]).T, axis=0)
													total = recombination_emissivity + excitation_emissivity + dummy_emissivity
													return total
												return calculated_emission

											bds = [[min_nH_ne, 0, 0, 0, 0], [max_nH_ne, 300, np.inf, np.inf, np.inf]]
											guess = [max_nH_ne, 10, 1000, 1000, 1000]
										elif len(n_list_1) == 2:
											def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
												def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1, dummy2):
													recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
													excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
													dummy_emissivity = np.sum((dummy.T * [dummy1, dummy2]).T, axis=0)
													total = recombination_emissivity + excitation_emissivity + dummy_emissivity
													return total
												return calculated_emission

											bds = [[min_nH_ne, 0, 0, 0], [max_nH_ne, 300, np.inf, np.inf]]
											guess = [max_nH_ne, 10, 1000, 1000]
										elif len(n_list_1) == 1:
											def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
												def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1):
													recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
													excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
													dummy_emissivity = np.sum((dummy.T * [dummy1]).T, axis=0)
													total = recombination_emissivity + excitation_emissivity + dummy_emissivity
													return total
												return calculated_emission

											bds = [[min_nH_ne, 0, 0], [max_nH_ne, 300, np.inf]]
											guess = [max_nH_ne, 10, 1000]

										fit = curve_fit(fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict), n_list,inverted_profiles_crop_restrict, p0=guess, bounds=bds,sigma=n_weights, maxfev=10000000)

										# if np.sum(fit[1] == 0) > 0:
										# 	continue

										# plt.figure()
										# plt.plot(n_list_all,inverted_profiles_crop_restrict/(inverted_profiles_crop_restrict[0]))
										# plt.plot(n_list_all,fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict)(n_list, *fit[0]))
										# # plt.plot(n_list_all,np.ones_like(inverted_profiles_crop_restrict) * OES_multiplier)
										# # plt.plot(n_list_all,fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict,inverted_profiles_crop_restrict)(n_list, *fit[0]))
										# plt.semilogy()
										# plt.title(str(fit[0])+'\n'+str(n_weights))
										# plt.pause(0.01)

										# nH_ne_all[my_time_pos, my_r_pos]=fit[0][0]
										# h_atomic_density_all[my_time_pos, my_r_pos]=fit[0][1]
										# dummy1_all[my_time_pos, my_r_pos]=fit[0][2]
										# dummy2_all[my_time_pos, my_r_pos]=fit[0][3]
										# dummy3_all[my_time_pos, my_r_pos]=fit[0][4]
										if np.sum(fit[1]==0)>0:
											residuals_all[my_time_pos, my_r_pos] =np.sum(fit[1]==0)
										else:
											residuals_all[my_time_pos, my_r_pos] = np.sum(((inverted_profiles_crop_restrict - fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict, excitation_restrict)(n_list, *fit[0]))) ** 2)
										# if residuals_all[my_time_pos, my_r_pos]>1:
										# 	print([my_time_pos, my_r_pos])
									# 	return residuals_all.flatten()
									# return residuals1


								# guess = [1]
								# bds = [[0.4],[5]]
								# sol = least_squares(residual_ext1(n_list,n_list_1,n_weights,time_crop,r_crop,inverted_profiles_crop,recombination,merge_ne_prof_multipulse_interp_crop_limited,merge_Te_prof_multipulse_interp_crop_limited,excitation), guess, bounds=bds, max_nfev=60,	verbose=2, gtol=1e-20, xtol=1e-5, ftol=1e-16, diff_step=0.01, x_scale='jac')
								# OES_multiplier=sol.x
								return residuals_all.flatten()
							return residuals


						guess = [1.3,-0.05]
						bds = [[1,-0.1], [2,0.1]]
						sol = least_squares(residual_ext(inverted_profiles_original, merge_Te_prof_multipulse, merge_ne_prof_multipulse, merge_time_original), guess, bounds=bds,max_nfev=60, verbose=2, gtol=1e-20, xtol=1e-12, ftol=1e-16, diff_step=[0.05,0.05], x_scale='jac')

						spatial_factor = sol.x[0]
						time_shift_factor = sol.x[1]

						print('spatial_factor,time_shift_factor  ' + str([spatial_factor, time_shift_factor]))

					else:
						# for spatial_factor in [1,1.1,1.2,1.3,1.4,1.5]:
						for spatial_factor in [1]:
							# for time_shift_factor in [-0.1,-0.05,0,0.05]:
							for time_shift_factor in [0]:
								mod2=mod + '/spatial_factor_'+str(spatial_factor)+'/time_shift_factor_'+str(time_shift_factor)

								if not os.path.exists(path_where_to_save_everything + mod2):
									os.makedirs(path_where_to_save_everything + mod2)

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
								# continue

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
								print('TS profile_centre [mm]')
								print(centre)
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
										elif np.sum(np.logical_and(np.abs(merge_time-value_t)<interp_range_t,np.sum(merge_Te_prof_multipulse,axis=1)>0))==0:
											continue
										selected_values_t = np.logical_and(np.abs(merge_time-value_t)<interp_range_t,np.sum(merge_Te_prof_multipulse,axis=1)>0)
										selected_values_r = np.abs(TS_r_new-value_r)<interp_range_r
										selecte_values = (np.array([selected_values_t])).T*selected_values_r
										selecte_values[merge_Te_prof_multipulse==0]=False
										if np.sum(selecte_values)==0:
											continue
										# temp1[i_t,i_r] = np.mean(merge_Te_prof_multipulse[selected_values_t][:,selected_values_r])
										# temp2[i_t, i_r] = np.max(merge_dTe_multipulse[selected_values_t][:,selected_values_r]) / (np.sum(np.isfinite(merge_dTe_multipulse[selected_values_t][:,selected_values_r])) ** 0.5)
										temp1[i_t,i_r] = np.sum(merge_Te_prof_multipulse[selecte_values]/merge_dTe_multipulse[selecte_values])/np.sum(1/merge_dTe_multipulse[selecte_values])
										temp2[i_t, i_r] = (np.sum(selecte_values)/(np.sum(1/merge_dTe_multipulse[selecte_values]) ** 2))**0.5
										# temp3[i_t,i_r] = np.mean(merge_ne_prof_multipulse[selected_values_t][:,selected_values_r])
										# temp4[i_t, i_r] = np.max(merge_dne_multipulse[selected_values_t][:,selected_values_r]) / (np.sum(np.isfinite(merge_dne_multipulse[selected_values_t][:,selected_values_r])) ** 0.5)
										temp3[i_t,i_r] = np.sum(merge_ne_prof_multipulse[selecte_values]/merge_dne_multipulse[selecte_values])/np.sum(1/merge_dne_multipulse[selecte_values])
										temp4[i_t, i_r] = (np.sum(selecte_values)/(np.sum(1/merge_dne_multipulse[selecte_values]) ** 2))**0.5

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
								merge_Te_prof_multipulse_interp_crop = merge_Te_prof_multipulse_interp[start_time:end_time, start_r:end_r]
								merge_dTe_prof_multipulse_interp_crop = merge_dTe_prof_multipulse_interp[start_time:end_time, start_r:end_r]
								merge_ne_prof_multipulse_interp_crop = merge_ne_prof_multipulse_interp[start_time:end_time, start_r:end_r]
								merge_dne_prof_multipulse_interp_crop = merge_dne_prof_multipulse_interp[start_time:end_time, start_r:end_r]
								inverted_profiles_crop = inverted_profiles[start_time:end_time, :, start_r:end_r]
								inverted_profiles_crop[np.isnan(inverted_profiles_crop)] = 0
								all_fits_crop = all_fits[start_time:end_time]
								# inverted_profiles_crop[inverted_profiles_crop<0] = 0

								x_local = xx - spatial_factor * 17.4 / 1000
								dr_crop = np.median(np.diff(r_crop))

								merge_dTe_prof_multipulse_interp_crop_limited = cp.deepcopy(merge_dTe_prof_multipulse_interp_crop)
								merge_dTe_prof_multipulse_interp_crop_limited[merge_Te_prof_multipulse_interp_crop < 0.2] = 0
								merge_Te_prof_multipulse_interp_crop_limited = cp.deepcopy(merge_Te_prof_multipulse_interp_crop)
								merge_Te_prof_multipulse_interp_crop_limited[merge_Te_prof_multipulse_interp_crop < 0.2] = 0
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



								n_list_all = np.sort(np.concatenate((n_list, n_list_1)))
								# OES_multiplier = 0.81414701
								# Te_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
								# ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
								nH_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
								h_atomic_density_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
								dummy1_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
								dummy2_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
								dummy3_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
								dummy4_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
								dummy5_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
								dummy6_all =  np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
								residuals_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)

								if ( (not os.path.exists(path_where_to_save_everything + mod2 + '/results.npz')) or recorded_data_override):

									if len(n_list_1)==6:
										guess = [min(1,max_nHp_ne), 10, 1000,1000,1000, 1000, 1000, 1000]
									elif len(n_list_1)==5:
										guess = [min(1,max_nHp_ne), 10, 1000, 1000, 1000, 1000, 1000]
									elif len(n_list_1)==4:
										guess = [min(1,max_nHp_ne), 10, 1000, 1000, 1000, 1000]
									elif len(n_list_1)==3:
										guess = [min(1,max_nHp_ne), 10, 1000, 1000, 1000]
									elif len(n_list_1)==2:
										guess = [min(1,max_nHp_ne), 10, 1000, 1000]
									elif len(n_list_1)==1:
										guess = [min(1,max_nHp_ne), 10, 1000]


									for my_time_pos in range(len(time_crop)):
										for my_r_pos in range(len(r_crop)):
											inverted_profiles_crop_restrict = inverted_profiles_crop[my_time_pos, n_list_all - 4, my_r_pos].flatten()
											recombination_restrict = recombination[n_list_all - 4, my_time_pos, my_r_pos].flatten()
											merge_ne_prof_multipulse_interp_crop_limited_restrict = merge_ne_prof_multipulse_interp_crop_limited[my_time_pos, my_r_pos].flatten()
											merge_Te_prof_multipulse_interp_crop_limited_restrict = merge_Te_prof_multipulse_interp_crop_limited[my_time_pos, my_r_pos].flatten()
											excitation_restrict = excitation[n_list_all - 4, my_time_pos, my_r_pos].flatten()

											if (merge_ne_prof_multipulse_interp_crop_limited_restrict == 0 or merge_Te_prof_multipulse_interp_crop_limited_restrict == 0):
												continue
											# if np.sum(inverted_profiles_crop_restrict == 0)>2:
											# 	continue

											dummy = np.zeros((len(n_list_1), len(n_list_all)))
											for value in n_list_1:
												dummy[value - 4][value - 4] = 1

											if len(n_list_1)==6:
												def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
													def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1, dummy2, dummy3, dummy4, dummy5, dummy6):
														recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
														excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
														dummy_emissivity = np.sum((dummy.T * [dummy1, dummy2, dummy3, dummy4, dummy5, dummy6]).T, axis=0)
														total = recombination_emissivity + excitation_emissivity + dummy_emissivity
														return total
													return calculated_emission

												bds = [[min_nHp_ne, 0, 0,0,0,0,0,0], [max_nHp_ne, merge_ne_prof_multipulse_interp_crop_limited_restrict*max_nH_ne_atomic_density, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]]
											if len(n_list_1)==5:
												def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
													def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1, dummy2, dummy3, dummy4, dummy5):
														recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
														excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
														dummy_emissivity = np.sum((dummy.T * [dummy1, dummy2, dummy3, dummy4, dummy5]).T, axis=0)
														total = recombination_emissivity + excitation_emissivity + dummy_emissivity
														return total
													return calculated_emission

												bds = [[min_nHp_ne, 0, 0,0,0,0,0], [max_nHp_ne, merge_ne_prof_multipulse_interp_crop_limited_restrict*max_nH_ne_atomic_density, np.inf, np.inf, np.inf, np.inf, np.inf]]
											if len(n_list_1)==4:
												def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
													def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1, dummy2, dummy3, dummy4):
														recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
														excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
														dummy_emissivity = np.sum((dummy.T * [dummy1, dummy2, dummy3, dummy4]).T, axis=0)
														total = recombination_emissivity + excitation_emissivity + dummy_emissivity
														return total
													return calculated_emission

												bds = [[min_nHp_ne, 0, 0,0,0,0], [max_nHp_ne, merge_ne_prof_multipulse_interp_crop_limited_restrict*max_nH_ne_atomic_density, np.inf, np.inf, np.inf, np.inf]]
											if len(n_list_1)==3:
												def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
													def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1, dummy2, dummy3):
														recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
														excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
														dummy_emissivity = np.sum((dummy.T * [dummy1, dummy2, dummy3]).T, axis=0)
														total = recombination_emissivity + excitation_emissivity + dummy_emissivity
														return total
													return calculated_emission

												bds = [[min_nHp_ne, 0, 0,0,0], [max_nHp_ne, merge_ne_prof_multipulse_interp_crop_limited_restrict*max_nH_ne_atomic_density, np.inf, np.inf, np.inf]]
											elif len(n_list_1)==2:
												def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
													def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1, dummy2):
														recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
														excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
														dummy_emissivity = np.sum((dummy.T * [dummy1, dummy2]).T, axis=0)
														total = recombination_emissivity + excitation_emissivity + dummy_emissivity
														return total
													return calculated_emission

												bds = [[min_nHp_ne, 0, 0, 0], [max_nHp_ne, merge_ne_prof_multipulse_interp_crop_limited_restrict*max_nH_ne_atomic_density, np.inf, np.inf]]
												# guess = [min(1,max_nHp_ne), merge_ne_prof_multipulse_interp_crop_limited_restrict, 1000, 1000]
											elif len(n_list_1)==1:
												def fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict):
													def calculated_emission(n_list, nH_ne, h_atomic_density, dummy1):
														recombination_emissivity = (recombination_restrict * nH_ne * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
														excitation_emissivity = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (h_atomic_density * (10 ** 20))).astype('float')
														dummy_emissivity = np.sum((dummy.T * [dummy1]).T, axis=0)
														total = recombination_emissivity + excitation_emissivity + dummy_emissivity
														return total
													return calculated_emission

												bds = [[min_nHp_ne, 0, 0], [max_nHp_ne, merge_ne_prof_multipulse_interp_crop_limited_restrict*max_nH_ne_atomic_density, np.inf]]

											for index in range(len(guess)):
												guess[index]=min(max(guess[index],bds[0][index]),bds[1][index])

											n_weights_actual = np.array(n_weights) * np.array(inverted_profiles_crop_restrict) / 10
											try:
												fit = curve_fit(fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict), n_list,inverted_profiles_crop_restrict, p0=guess, bounds=bds, sigma=n_weights_actual,maxfev=10000000,ftol=1e-5)
												if (fit[0][1] < 0.01 and min_nHp_ne==0.9999):
													bds[0][0]=0
													fit = curve_fit(fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict), n_list,inverted_profiles_crop_restrict, p0=guess, bounds=bds, sigma=n_weights_actual,maxfev=10000000,ftol=1e-5)
													guess=fit[0]
											except Exception as e:
												print('fit at time index %.3g, radious index %.3g failed for reason %s' %(my_time_pos,my_r_pos,e))
												fit = np.array([guess])




											# # if np.sum(fit[1] == 0)>0:
											# # 	continue
											if int(time_crop[my_time_pos]*100) in np.array([0.2,0.4,0.6,0.8])*100:
												if r_crop[my_r_pos] in r_crop[[0,10,20,30]]:
													plt.figure()
													# plt.plot(n_list_all,inverted_profiles_crop_restrict/inverted_profiles_crop_restrict[0])
													# plt.plot(n_list_all,fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict)(n_list, *fit[0]))
													recombination_emissivity_local = (recombination_restrict * fit[0][0] * ((merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) ** 2)).astype('float')
													excitation_emissivity_local = (excitation_restrict * (merge_ne_prof_multipulse_interp_crop_limited_restrict * (10 ** 20)) * (fit[0][1] * (10 ** 20))).astype('float')
													dummy_emissivity_local = np.sum((dummy.T * [fit[0][2:]]).T, axis=0)
													plt.plot(n_list_all,inverted_profiles_crop_restrict,label='OES')
													plt.plot(n_list_all,recombination_emissivity_local,label='recombination')
													plt.plot(n_list_all,excitation_emissivity_local,label='excitation')
													plt.plot(n_list_all,dummy_emissivity_local,'*',label='extra radiation')
													plt.plot(n_list_all,fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict)(n_list, *fit[0]),'--',label='total fit')
													plt.plot(n_list_all,fit_nH_ne_h_atomic_density(dummy*0, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict)(n_list, *fit[0]),'+',label='excitation + recombination')
													plt.semilogy()
													plt.legend(loc='best')
													plt.ylim(np.min(inverted_profiles_crop_restrict)/2,np.max(inverted_profiles_crop_restrict)*2)
													plt.title('lines '+str(n_list_all)+' weights '+str(n_weights)+'\nlocation [time, r]'+ ' [%.3g ms, ' % time_crop[my_time_pos] + ' %.3g mm]' % (1000*r_crop[my_r_pos]) +' , [TS Te, TS ne] '+ ' [%.3g eV, ' % merge_Te_prof_multipulse_interp_crop_limited_restrict + '%.3g #/m^3]' % merge_ne_prof_multipulse_interp_crop_limited_restrict +'\nfit [nH/ne, H_atomic_density, extra rad] , '+ '[%.3g, ' % fit[0][0] + '%.3g, ' % fit[0][1]+ str(np.around(fit[0][2:]))+']')# +'\nweights '+str(n_weights))
													plt.xlabel('Exctited state n')
													plt.ylabel('Line emissivity n->2 [W m^-3]')
													# plt.pause(0.01)
													plt.savefig(path_where_to_save_everything + mod2 + '/post_process_mega_global_fit' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps', bbox_inches='tight')
													plt.close()

											nH_ne_all[my_time_pos, my_r_pos]=fit[0][0]
											h_atomic_density_all[my_time_pos, my_r_pos]=fit[0][1]
											if len(n_list_1)>=1:
												dummy1_all[my_time_pos, my_r_pos]=fit[0][2]
											if len(n_list_1)>=2:
												dummy2_all[my_time_pos, my_r_pos]=fit[0][3]
											if len(n_list_1)>=3:
												dummy3_all[my_time_pos, my_r_pos]=fit[0][4]
											if len(n_list_1)>=4:
												dummy4_all[my_time_pos, my_r_pos]=fit[0][5]
											if len(n_list_1)>=5:
												dummy5_all[my_time_pos, my_r_pos]=fit[0][6]
											if len(n_list_1)>=6:
												dummy6_all[my_time_pos, my_r_pos]=fit[0][7]
											# Te_all[my_time_pos, my_r_pos]=fit[0][3]
											# ne_all[my_time_pos, my_r_pos]=fit[0][4]
											# if np.sum(fit[1] == 0) == len(fit[0]) ** 2:
											# 	residuals_all[my_time_pos, my_r_pos] = 0
											# else:
											residuals_all[my_time_pos, my_r_pos] = np.sum(((inverted_profiles_crop_restrict - fit_nH_ne_h_atomic_density(dummy, recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict)(n_list, *fit[0]))) ** 2)

									np.savez_compressed(path_where_to_save_everything + mod2 + '/results',nH_ne_all=nH_ne_all,h_atomic_density_all=h_atomic_density_all,dummy1_all=dummy1_all,dummy2_all=dummy2_all,dummy3_all=dummy3_all,dummy4_all=dummy4_all,dummy5_all=dummy5_all,dummy6_all=dummy6_all,residuals_all=residuals_all)
								else:
									nH_ne_all = np.load(path_where_to_save_everything + mod2 +'/results.npz')['nH_ne_all']
									h_atomic_density_all = np.load(path_where_to_save_everything + mod2 +'/results.npz')['h_atomic_density_all']
									dummy1_all = np.load(path_where_to_save_everything + mod2 +'/results.npz')['dummy1_all']
									dummy2_all = np.load(path_where_to_save_everything + mod2 +'/results.npz')['dummy2_all']
									dummy3_all = np.load(path_where_to_save_everything + mod2 +'/results.npz')['dummy3_all']
									dummy4_all = np.load(path_where_to_save_everything + mod2 +'/results.npz')['dummy4_all']
									dummy5_all = np.load(path_where_to_save_everything + mod2 +'/results.npz')['dummy5_all']
									dummy6_all = np.load(path_where_to_save_everything + mod2 +'/results.npz')['dummy6_all']
									residuals_all = np.load(path_where_to_save_everything + mod2 +'/results.npz')['residuals_all']


								# plt.figure()
								# plt.title('ne,Te,nH_ne,h_atomic_density,OES_multiplier\n'+str([*merge_ne_prof_multipulse_interp_crop_limited_restrict,*merge_Te_prof_multipulse_interp_crop_limited_restrict,*fit[0],OES_multiplier]))
								# plt.plot(n_list,inverted_profiles_crop_restrict*OES_multiplier)
								# plt.plot(n_list,fit_nH_ne_h_atomic_density(recombination_restrict,merge_ne_prof_multipulse_interp_crop_limited_restrict,excitation_restrict)(n_list,*fit[0]))
								# plt.pause(0.01)

								figure_index = 0

								temp_r, temp_t = np.meshgrid(r_crop, time_crop)
								plt.figure();
								plt.pcolor(temp_t, temp_r, nH_ne_all, cmap='rainbow');
								plt.colorbar().set_label('[au]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('hydrogen ion density / electron density\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor]))
								figure_index += 1
								plt.savefig(path_where_to_save_everything +mod2+ '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								plt.figure();
								plt.pcolor(temp_t, temp_r, merge_ne_prof_multipulse_interp_crop_limited*nH_ne_all, cmap='rainbow');
								plt.colorbar().set_label('[# m^-3]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('hydrogen ion density\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor]))
								figure_index += 1
								plt.savefig(path_where_to_save_everything +mod2+ '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								plt.figure();
								if np.mean(np.sort(h_atomic_density_all.flatten())[-100:-10])>0:
									plt.pcolor(temp_t, temp_r, h_atomic_density_all, cmap='rainbow',norm=LogNorm(),vmin=0.1,vmax=np.mean(np.sort(h_atomic_density_all.flatten())[-100:-10]));
								else:
									plt.pcolor(temp_t, temp_r, h_atomic_density_all, cmap='rainbow');
								plt.colorbar().set_label('[10^20 # m^-3]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('neutral atomic hydrogen density\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor]))
								figure_index += 1
								plt.savefig(path_where_to_save_everything +mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								plt.figure();
								plt.pcolor(temp_t, temp_r, residuals_all, cmap='rainbow',vmax=np.max(residuals_all[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
								plt.colorbar().set_label('[W m^-3]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('relative residual unaccounted line emission\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor])+'\nsum = '+str(np.sum(residuals_all)))
								figure_index += 1
								plt.savefig(path_where_to_save_everything +mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								plt.figure();
								plt.pcolor(temp_t, temp_r, dummy1_all, cmap='rainbow',vmax=np.max(dummy1_all[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]),vmin=np.min(dummy1_all[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
								plt.colorbar().set_label('[W m^-3]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('extra emissivity n=4\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor]))
								figure_index += 1
								plt.savefig(path_where_to_save_everything +mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								plt.figure();
								plt.pcolor(temp_t, temp_r, dummy2_all, cmap='rainbow');
								plt.colorbar().set_label('[W m^-3]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('extra emissivity n=5\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor]))
								figure_index += 1
								plt.savefig(path_where_to_save_everything +mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								plt.figure();
								plt.pcolor(temp_t, temp_r, dummy3_all, cmap='rainbow');
								plt.colorbar().set_label('[W m^-3]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('extra emissivity n=6\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor]))
								figure_index += 1
								plt.savefig(path_where_to_save_everything +mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								plt.figure();
								plt.pcolor(temp_t, temp_r, dummy4_all, cmap='rainbow');
								plt.colorbar().set_label('[W m^-3]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('extra emissivity n=7\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor]))
								figure_index += 1
								plt.savefig(path_where_to_save_everything +mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								plt.figure();
								plt.pcolor(temp_t, temp_r, dummy5_all, cmap='rainbow');
								plt.colorbar().set_label('[W m^-3]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('extra emissivity n=8\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor]))
								figure_index += 1
								plt.savefig(path_where_to_save_everything +mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								plt.figure();
								plt.pcolor(temp_t, temp_r, dummy6_all, cmap='rainbow');
								plt.colorbar().set_label('[W m^-3]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('extra emissivity n=7\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor]))
								figure_index += 1
								plt.savefig(path_where_to_save_everything +mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								# figure_index += 1
								# plt.savefig(path_where_to_save_everything + mod + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								# plt.close()

								recombination_emissivity = (recombination * nH_ne_all * ((merge_ne_prof_multipulse_interp_crop_limited * (10 ** 20)) ** 2)).astype('float')
								excitation_emissivity = (excitation * (merge_ne_prof_multipulse_interp_crop_limited * (10 ** 20)) * (h_atomic_density_all * (10 ** 20))).astype('float')

								residual_emission = []
								for index in range(len(recombination_emissivity)):
									residual_emission.append(inverted_profiles_crop[:, index] - recombination_emissivity[index] - excitation_emissivity[index])
								residual_emission = np.array(residual_emission)

								to_print = inverted_profiles_crop[:, 0]*dummy1_all/(recombination_emissivity[0]+excitation_emissivity[0]+ dummy1_all)
								plt.figure();
								plt.pcolor(temp_t, temp_r, to_print, cmap='rainbow',vmax=np.max(to_print[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]),vmin=np.min(dummy1_all[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
								plt.colorbar().set_label('[W m^-3]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('extra emissivity using only the ratio from the fitting on n=4\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor]))
								figure_index += 1
								plt.savefig(path_where_to_save_everything +mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()
								figure_index -= 1


								plt.figure();
								plt.pcolor(temp_t, temp_r, residual_emission[0], cmap='rainbow', vmin=np.min(residual_emission[0][np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
								plt.colorbar().set_label('line emission [W m^-3]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('residual emissivity, attribuited to MAR, line4\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor]))
								figure_index += 1
								plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								temp = residual_emission[0] / recombination_emissivity[0]
								temp[np.logical_not(np.isfinite(temp))] = 0
								plt.figure();
								plt.pcolor(temp_t, temp_r, temp, cmap='rainbow',vmax=np.max(temp[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]),vmin=np.min(temp[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
								plt.colorbar().set_label(' relative line emission [au]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('relative residual emissivity, attribuited to MAR, line4\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor]))
								figure_index += 1
								plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								plt.figure();
								plt.pcolor(temp_t, temp_r, residual_emission[0]-dummy1_all, cmap='rainbow', vmin=np.min(residual_emission[0][np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
								plt.colorbar().set_label('line emission [W m^-3]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('residual emissivity not included in the fitting, line4\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor])+'\nsum = '+str(np.sum(np.abs(residual_emission[0]-dummy1_all))))
								figure_index += 1
								plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()


								plt.figure();
								plt.pcolor(temp_t, temp_r, residual_emission[1], cmap='rainbow', vmin=np.min(residual_emission[1][np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
								plt.colorbar().set_label('line emission [W m^-3]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('residual emissivity, attribuited to MAR, line5\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor]))
								figure_index += 1
								plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								temp = residual_emission[1] / recombination_emissivity[1]
								temp[np.logical_not(np.isfinite(temp))] = 0
								plt.figure();
								plt.pcolor(temp_t, temp_r, temp, cmap='rainbow',vmax=np.max(temp[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]),vmin=np.min(temp[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
								plt.colorbar().set_label(' relative line emission [au]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('relative residual emissivity, attribuited to MAR, line5\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor]))
								figure_index += 1
								plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								if len(n_list_1)>1:
									plt.figure();
									plt.pcolor(temp_t, temp_r, residual_emission[1]-dummy2_all, cmap='rainbow', vmin=np.min(residual_emission[1][np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
									plt.colorbar().set_label('line emission [W m^-3]')  # ;plt.pause(0.01)
									plt.axes().set_aspect(20)
									plt.xlabel('time [ms]')
									plt.ylabel('radial location [m]')
									plt.title('residual emissivity not included in the fitting, line5\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor])+'\nsum = '+str(np.sum(np.abs(residual_emission[1]-dummy1_all))))
									figure_index += 1
									plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
									plt.close()

								plt.figure();
								plt.pcolor(temp_t, temp_r, residual_emission[2], cmap='rainbow', vmin=np.min(residual_emission[2][np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
								plt.colorbar().set_label('line emission [W m^-3]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('residual emissivity, attribuited to MAR, line6\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor]))
								figure_index += 1
								plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								temp = residual_emission[2] / recombination_emissivity[2]
								temp[np.logical_not(np.isfinite(temp))] = 0
								plt.figure();
								plt.pcolor(temp_t, temp_r, temp, cmap='rainbow',vmax=np.max(temp[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]),vmin=np.min(temp[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
								plt.colorbar().set_label(' relative line emission [au]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('relative residual emissivity, attribuited to MAR, line6\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor]))
								figure_index += 1
								plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								if len(n_list_1)>2:
									plt.figure();
									plt.pcolor(temp_t, temp_r, residual_emission[2]-dummy3_all, cmap='rainbow', vmin=np.min(residual_emission[2][np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
									plt.colorbar().set_label('line emission [W m^-3]')  # ;plt.pause(0.01)
									plt.axes().set_aspect(20)
									plt.xlabel('time [ms]')
									plt.ylabel('radial location [m]')
									plt.title('residual emissivity not included in the fitting, line6\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor])+'\nsum = '+str(np.sum(np.abs(residual_emission[2]-dummy1_all))))
									figure_index += 1
									plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
									plt.close()

								if len(n_list_1)>3:
									plt.figure();
									plt.pcolor(temp_t, temp_r, residual_emission[3]-dummy4_all, cmap='rainbow', vmin=np.min(residual_emission[1][np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
									plt.colorbar().set_label('line emission [W m^-3]')  # ;plt.pause(0.01)
									plt.axes().set_aspect(20)
									plt.xlabel('time [ms]')
									plt.ylabel('radial location [m]')
									plt.title('residual emissivity not included in the fitting, line7\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor])+'\nsum = '+str(np.sum(np.abs(residual_emission[1]-dummy1_all))))
									figure_index += 1
									plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
									plt.close()

								if len(n_list_1)>4:
									plt.figure();
									plt.pcolor(temp_t, temp_r, residual_emission[4]-dummy5_all, cmap='rainbow', vmin=np.min(residual_emission[1][np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
									plt.colorbar().set_label('line emission [W m^-3]')  # ;plt.pause(0.01)
									plt.axes().set_aspect(20)
									plt.xlabel('time [ms]')
									plt.ylabel('radial location [m]')
									plt.title('residual emissivity not included in the fitting, line8\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor])+'\nsum = '+str(np.sum(np.abs(residual_emission[1]-dummy1_all))))
									figure_index += 1
									plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
									plt.close()

								if len(n_list_1)>5:
									plt.figure();
									plt.pcolor(temp_t, temp_r, residual_emission[5]-dummy6_all, cmap='rainbow', vmin=np.min(residual_emission[1][np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
									plt.colorbar().set_label('line emission [W m^-3]')  # ;plt.pause(0.01)
									plt.axes().set_aspect(20)
									plt.xlabel('time [ms]')
									plt.ylabel('radial location [m]')
									plt.title('residual emissivity not included in the fitting, line9\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor])+'\nsum = '+str(np.sum(np.abs(residual_emission[1]-dummy1_all))))
									figure_index += 1
									plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
									plt.close()



								temp = read_adf11(scdfile, 'scd', 1, 1, 1, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())
								temp[np.isnan(temp)] = 0
								effective_ionisation_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (10 ** -6)  # in ionisations m^-3 s-1 / (# / m^3)**2
								effective_ionisation_rates = (effective_ionisation_rates * (merge_ne_prof_multipulse_interp_crop * (10 ** 20)) * (h_atomic_density_all*1e20)).astype('float')

								plt.figure();
								plt.pcolor(temp_t, temp_r, effective_ionisation_rates, cmap='rainbow');
								plt.colorbar().set_label('effective_ionisation_rates [# m^-3 s-1]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('effective_ionisation_rates\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor]))
								# plt.title('effective_ionisation_rates')
								figure_index += 1
								plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								temp = read_adf11(acdfile, 'acd', 1, 1, 1, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())
								temp[np.isnan(temp)] = 0
								effective_recombination_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (10 ** -6)  # in recombinations m^-3 s-1 / (# / m^3)**2
								effective_recombination_rates = (effective_recombination_rates *nH_ne_all* (merge_ne_prof_multipulse_interp_crop * (10 ** 20)) ** 2).astype('float')

								plt.figure();
								plt.pcolor(temp_t, temp_r, effective_recombination_rates, cmap='rainbow');
								plt.colorbar().set_label('effective_recombination_rates [# m^-3 s-1]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('effective_recombination_rates (three body plus radiative)\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1,spatial_factor,time_shift_factor]))
								figure_index += 1
								plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()


								plt.figure();
								plt.pcolor(temp_t, temp_r, merge_ne_prof_multipulse_interp_crop_limited, cmap='rainbow');
								plt.colorbar().set_label('electron density [10^20 # m^-3]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('Scaled electron density')
								figure_index += 1
								plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								temp = 100*merge_dne_prof_multipulse_interp_crop_limited / merge_ne_prof_multipulse_interp_crop_limited
								temp[np.logical_not(np.isfinite(temp))] = 0
								plt.figure();
								plt.pcolor(temp_t, temp_r, temp, cmap='rainbow',norm=LogNorm());
								plt.colorbar().set_label('uncertainty [%]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('Scaled electron density relative uncertainty')
								figure_index += 1
								plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								plt.figure();
								plt.pcolor(temp_t, temp_r, merge_Te_prof_multipulse_interp_crop_limited, cmap='rainbow');
								plt.colorbar().set_label('electron temperature [eV]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('Scaled electron temperature')
								figure_index += 1
								plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								temp = 100*merge_dTe_prof_multipulse_interp_crop_limited / merge_Te_prof_multipulse_interp_crop_limited
								temp[np.logical_not(np.isfinite(temp))] = 0
								plt.figure();
								plt.pcolor(temp_t, temp_r, temp, cmap='rainbow',norm=LogNorm());
								plt.colorbar().set_label('uncertainty [%]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('Scaled electron temperature relative uncertainty')
								figure_index += 1
								plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',
											bbox_inches='tight')
								plt.close()

								plt.figure();
								plt.pcolor(temp_t, temp_r, inverted_profiles_crop[:,0], cmap='rainbow');
								plt.colorbar().set_label('[W m^-3]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('OES emissivity n=4->2\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
								figure_index += 1
								plt.savefig(path_where_to_save_everything +mod2+ '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								plt.figure();
								plt.pcolor(temp_t, temp_r, inverted_profiles_crop[:,np.abs(n_list_all-9).argmin()], cmap='rainbow');
								plt.colorbar().set_label('[W m^-3]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('OES emissivity n=9->2\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
								figure_index += 1
								plt.savefig(path_where_to_save_everything +mod2+ '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								to_print = 100*dummy1_all/inverted_profiles_crop[:, 0]
								to_print[np.logical_not(np.isfinite(to_print))] = 0
								plt.figure();
								plt.pcolor(temp_t, temp_r, to_print, cmap='rainbow');
								plt.colorbar().set_label('emissivity ratio [%]')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('Ratio of emissivity attributed to MAR\nover total OEs measurement')
								figure_index += 1
								plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()


								fig = plt.figure()
								# fig=plt.figure(merge_ID_target * 100 + 2)
								ax = fig.add_subplot(1, 6, 1)
								im = ax.pcolor(1000*temp_r.T,temp_t.T,merge_Te_prof_multipulse_interp_crop_limited.T,cmap='rainbow')
								# plt.set_sketch_params(scale=2)
								ax.set_aspect(40)
								plt.title('TS\nelectron temperature')
								plt.xlabel('radial location [mm]')
								plt.ylabel('time [ms]')
								fig.colorbar(im, ax=ax).set_label('[eV]')
								# plt.axes().set_aspect(0.1)


								ax=fig.add_subplot(1, 6, 2)
								im=ax.pcolor(1000*temp_r.T,temp_t.T,merge_ne_prof_multipulse_interp_crop_limited.T,cmap='rainbow')
								ax.set_aspect(40)
								plt.title('TS\nelectron density')
								plt.xlabel('radial location [mm]')
								plt.yticks([])
								# plt.ylabel('time [ms]')
								fig.colorbar(im, ax=ax).set_label('[10^20 # m^-3]')

								ax=fig.add_subplot(1, 6, 3)
								im=ax.pcolor(1000*temp_r.T,temp_t.T,inverted_profiles_crop[:, 0].T,cmap='rainbow')
								ax.set_aspect(40)
								plt.title('OES\nn=4>2 line')
								plt.xlabel('radial location [mm]')
								plt.yticks([])
								# plt.ylabel('time [ms]')
								fig.colorbar(im, ax=ax).set_label('emissivity [W m^-3]')

								ax=fig.add_subplot(1, 6, 4)
								im = ax.pcolor(1000*temp_r.T,temp_t.T,effective_ionisation_rates.T,cmap='rainbow')
								ax.set_aspect(40)
								plt.title('effective ionisation\nrates')
								plt.xlabel('radial location [mm]')
								# plt.ylabel('time [ms]')
								plt.yticks([])
								fig.colorbar(im, ax=ax).set_label('effective ionisation rates [# m^-3 s-1]')

								ax=fig.add_subplot(1, 6, 5)
								im = ax.pcolor(1000*temp_r.T,temp_t.T,effective_recombination_rates.T, cmap='rainbow')
								ax.set_aspect(40)
								plt.title('effective recombination\nrates')
								plt.xlabel('radial location [mm]')
								# plt.ylabel('time [ms]')
								plt.yticks([])
								fig.colorbar(im, ax=ax).set_label('effective recombination rates [# m^-3 s-1]')

								ax=fig.add_subplot(1, 6, 6)
								im = ax.pcolor(1000*temp_r.T,temp_t.T,dummy1_all.T, cmap='rainbow')
								ax.set_aspect(40)
								plt.title('extra emissivity\nn=4')
								plt.xlabel('radial location [mm]')
								# plt.ylabel('time [ms]')
								plt.yticks([])
								fig.colorbar(im, ax=ax).set_label('emissivity [W m^-3]')
								figure_index += 1
								# plt.pause(0.001)
								plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',
											bbox_inches='tight')
								plt.close()


								fig = plt.figure()
								ax = fig.add_subplot(1, 3, 1)
								im = ax.pcolor(1000 * temp_r.T, temp_t.T, effective_ionisation_rates.T, cmap='rainbow')
								ax.set_aspect(40)
								plt.title('effective ionisation\nrates')
								plt.xlabel('radial location [mm]')
								plt.ylabel('time [ms]')
								# plt.yticks([])
								fig.colorbar(im, ax=ax).set_label('effective ionisation rates [# m^-3 s-1]')

								ax = fig.add_subplot(1, 3, 2)
								im = ax.pcolor(1000 * temp_r.T, temp_t.T, effective_recombination_rates.T, cmap='rainbow')
								ax.set_aspect(40)
								plt.title('effective recombination\nrates')
								plt.xlabel('radial location [mm]')
								# plt.ylabel('time [ms]')
								plt.yticks([])
								fig.colorbar(im, ax=ax).set_label('effective recombination rates [# m^-3 s-1]')

								ax = fig.add_subplot(1, 3, 3)
								im = ax.pcolor(1000 * temp_r.T, temp_t.T, dummy1_all.T, cmap='rainbow')
								ax.set_aspect(40)
								plt.title('extra emissivity\nn=4')
								plt.xlabel('radial location [mm]')
								# plt.ylabel('time [ms]')
								plt.yticks([])
								fig.colorbar(im, ax=ax).set_label('emissivity [W m^-3]')
								figure_index += 1
								# plt.pause(0.001)
								plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()


								plt.figure();
								threshold_radious = 0.005
								plt.plot(time_crop, np.mean(effective_ionisation_rates[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(effective_ionisation_rates[:,r_crop<threshold_radious],axis=1)), label='ionisation rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.mean(effective_ionisation_rates[:,r_crop<threshold_radious],axis=1))));
								plt.plot(time_crop, np.mean(effective_recombination_rates[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(effective_recombination_rates[:,r_crop<threshold_radious],axis=1)), label='recombination rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.mean(effective_recombination_rates[:,r_crop<threshold_radious],axis=1))));
								plt.plot(time_crop, np.mean(dummy1_all[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(dummy1_all[:,r_crop<threshold_radious],axis=1)), label='additional H-β emissivity\n(max='+'%.3g W m^-3)' % (np.max(np.mean(dummy1_all[:,r_crop<threshold_radious],axis=1))));
								if len(n_list_1)>1:
									plt.plot(time_crop, np.mean(dummy2_all[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(dummy2_all[:,r_crop<threshold_radious],axis=1)), label='additional H-β emissivity\n(max='+'%.3g W m^-3)' % (np.max(np.mean(dummy2_all[:,r_crop<threshold_radious],axis=1))));
								if len(n_list_1)>2:
									plt.plot(time_crop, np.mean(dummy3_all[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(dummy3_all[:,r_crop<threshold_radious],axis=1)), label='additional H-β emissivity\n(max='+'%.3g W m^-3)' % (np.max(np.mean(dummy3_all[:,r_crop<threshold_radious],axis=1))));
								if len(n_list_1)>3:
									plt.plot(time_crop, np.mean(dummy4_all[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(dummy4_all[:,r_crop<threshold_radious],axis=1)), label='additional H-β emissivity\n(max='+'%.3g W m^-3)' % (np.max(np.mean(dummy4_all[:,r_crop<threshold_radious],axis=1))));
								if len(n_list_1)>4:
									plt.plot(time_crop, np.mean(dummy5_all[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(dummy5_all[:,r_crop<threshold_radious],axis=1)), label='additional H-β emissivity\n(max='+'%.3g W m^-3)' % (np.max(np.mean(dummy5_all[:,r_crop<threshold_radious],axis=1))));
								if len(n_list_1)>5:
									plt.plot(time_crop, np.mean(dummy6_all[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(dummy6_all[:,r_crop<threshold_radious],axis=1)), label='additional H-β emissivity\n(max='+'%.3g W m^-3)' % (np.max(np.mean(dummy6_all[:,r_crop<threshold_radious],axis=1))));
								plt.plot(time_crop, np.mean(merge_Te_prof_multipulse_interp_crop_limited[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(merge_Te_prof_multipulse_interp_crop_limited[:,r_crop<threshold_radious],axis=1)),'--', label='Te TS\n(max='+'%.3g eV)' % (np.max(np.mean(merge_Te_prof_multipulse_interp_crop_limited[:,r_crop<threshold_radious],axis=1))));
								plt.plot(time_crop, np.mean(merge_ne_prof_multipulse_interp_crop_limited[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(merge_ne_prof_multipulse_interp_crop_limited[:,r_crop<threshold_radious],axis=1)),'--', label='ne TS\n(max='+'%.3g 10^20 # m^-3)' % (np.max(np.mean(merge_ne_prof_multipulse_interp_crop_limited[:,r_crop<threshold_radious],axis=1))));
								plt.legend(loc='best')
								plt.xlabel('time from beginning of pulse [ms]')
								plt.ylabel('relative radial average [au]')
								plt.title('Time evolution of the radial average, 0>r>'+  '%.3g' % (1000*threshold_radious) +' mm, '+str(target_OES_distance)+'mm from the target')
								figure_index += 1
								plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								plt.figure();
								threshold_radious = 0.015
								plt.plot(time_crop, np.mean(effective_ionisation_rates[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(effective_ionisation_rates[:,r_crop<threshold_radious],axis=1)), label='ionisation rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.mean(effective_ionisation_rates[:,r_crop<threshold_radious],axis=1))));
								plt.plot(time_crop, np.mean(effective_recombination_rates[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(effective_recombination_rates[:,r_crop<threshold_radious],axis=1)), label='recombination rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.mean(effective_recombination_rates[:,r_crop<threshold_radious],axis=1))));
								plt.plot(time_crop, np.mean(dummy1_all[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(dummy1_all[:,r_crop<threshold_radious],axis=1)), label='additional H-β emissivity\n(max='+'%.3g W m^-3)' % (np.max(np.mean(dummy1_all[:,r_crop<threshold_radious],axis=1))));
								if len(n_list_1)>1:
									plt.plot(time_crop, np.mean(dummy2_all[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(dummy2_all[:,r_crop<threshold_radious],axis=1)), label='additional H-β emissivity\n(max='+'%.3g W m^-3)' % (np.max(np.mean(dummy2_all[:,r_crop<threshold_radious],axis=1))));
								if len(n_list_1)>2:
									plt.plot(time_crop, np.mean(dummy3_all[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(dummy3_all[:,r_crop<threshold_radious],axis=1)), label='additional H-β emissivity\n(max='+'%.3g W m^-3)' % (np.max(np.mean(dummy3_all[:,r_crop<threshold_radious],axis=1))));
								if len(n_list_1)>3:
									plt.plot(time_crop, np.mean(dummy4_all[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(dummy4_all[:,r_crop<threshold_radious],axis=1)), label='additional H-β emissivity\n(max='+'%.3g W m^-3)' % (np.max(np.mean(dummy4_all[:,r_crop<threshold_radious],axis=1))));
								if len(n_list_1)>4:
									plt.plot(time_crop, np.mean(dummy5_all[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(dummy5_all[:,r_crop<threshold_radious],axis=1)), label='additional H-β emissivity\n(max='+'%.3g W m^-3)' % (np.max(np.mean(dummy5_all[:,r_crop<threshold_radious],axis=1))));
								if len(n_list_1)>5:
									plt.plot(time_crop, np.mean(dummy6_all[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(dummy6_all[:,r_crop<threshold_radious],axis=1)), label='additional H-β emissivity\n(max='+'%.3g W m^-3)' % (np.max(np.mean(dummy6_all[:,r_crop<threshold_radious],axis=1))));
								plt.plot(time_crop, np.mean(merge_Te_prof_multipulse_interp_crop_limited[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(merge_Te_prof_multipulse_interp_crop_limited[:,r_crop<threshold_radious],axis=1)),'--', label='Te TS\n(max='+'%.3g eV)' % (np.max(np.mean(merge_Te_prof_multipulse_interp_crop_limited[:,r_crop<threshold_radious],axis=1))));
								plt.plot(time_crop, np.mean(merge_ne_prof_multipulse_interp_crop_limited[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(merge_ne_prof_multipulse_interp_crop_limited[:,r_crop<threshold_radious],axis=1)),'--', label='ne TS\n(max='+'%.3g 10^20 # m^-3)' % (np.max(np.mean(merge_ne_prof_multipulse_interp_crop_limited[:,r_crop<threshold_radious],axis=1))));
								plt.legend(loc='best')
								plt.xlabel('time from beginning of pulse [ms]')
								plt.ylabel('relative radial average [au]')
								plt.title('Time evolution of the radial average, 0>r>'+  '%.3g' % (1000*threshold_radious) +' mm, '+str(target_OES_distance)+'mm from the target')
								figure_index += 1
								plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								volume = 2*np.pi*(r_crop + np.diff([0,*r_crop])/2) * np.diff([0,*r_crop]) * target_OES_distance/1000
								area = 2*np.pi*(r_crop + np.diff([0,*r_crop])/2) * np.diff([0,*r_crop])
								ionisation_source = np.sum(volume * effective_ionisation_rates * np.median(np.diff(time_crop))/1000,axis=1)
								recombination_source = np.sum(volume * effective_recombination_rates * np.median(np.diff(time_crop))/1000,axis=1)
								inflow_min = np.sum(area * merge_ne_prof_multipulse_interp_crop_limited * 1000* np.median(np.diff(time_crop))/1000,axis=1)* 1e20	#steady state flow from upstream ~1km/s ballpark from CTS (Jonathan)
								inflow_max = np.sum(area * merge_ne_prof_multipulse_interp_crop_limited * 10000* np.median(np.diff(time_crop))/1000,axis=1)* 1e20	#peak flow from upstream ~10km/s ballpark from CTS (Jonathan)
								total_e = np.sum(volume * merge_ne_prof_multipulse_interp_crop_limited,axis=1) * 1e20
								total_Hp = np.sum(volume * merge_ne_prof_multipulse_interp_crop_limited * nH_ne_all,axis=1) * 1e20
								total_H = np.sum(volume * h_atomic_density_all * nH_ne_all,axis=1) * 1e20
								total_dummy1_all = np.sum(volume * dummy1_all ,axis=1)
								total_dummy2_all = np.sum(volume * dummy2_all ,axis=1)
								total_dummy3_all = np.sum(volume * dummy3_all ,axis=1)
								total_dummy4_all = np.sum(volume * dummy4_all ,axis=1)
								total_dummy5_all = np.sum(volume * dummy5_all ,axis=1)
								total_dummy6_all = np.sum(volume * dummy6_all ,axis=1)
								au_kg = 1.66054e-27	#1au to xxxkg
								J_to_eV = 6.242e18	#1J = xxx eV
								bohm_velocity = (2*merge_Te_prof_multipulse_interp_crop_limited/(1*au_kg*J_to_eV))**0.5	# assuming Te=Ti, m/s
								ion_flux_target = np.sum(area *merge_ne_prof_multipulse_interp_crop_limited*bohm_velocity * 1e20 * np.median(np.diff(time_crop))/1000,axis=1)
								# plt.figure();
								fig, ax = plt.subplots(figsize=(20, 10))
								ax.fill_between(time_crop, inflow_min,inflow_max, inflow_max>=inflow_min,color='green',alpha=0.1,label='electron inflow from upstream\nflow vel 1-10km/s (CTS ballpark)')#, label='inflow, 1-10km/s (CTS)');
								ax.plot(time_crop, ion_flux_target, label='ion sink at the target');
								ax.plot(time_crop, ionisation_source, label='total ionisation source');
								ax.plot(time_crop, recombination_source, label='total recombination source');
								ax.plot(time_crop, ionisation_source - recombination_source,'kv', label='ionisation - recombination');
								ax.plot(time_crop, -(ionisation_source - recombination_source),'k^', label='recombination - ionisation');
								ax.plot(time_crop, total_e, label='total munber of electrons');
								ax.plot(time_crop, total_Hp, label='total munber of H+');
								ax.plot(time_crop, total_H, label='total munber of H');
								ax.plot(time_crop, total_dummy1_all*np.max(total_Hp)/np.max(total_dummy1_all), label='total extra radiation line 4->2\nscaled as its max=H+max');
								if np.max(total_dummy2_all)>0:
									ax.plot(time_crop, total_dummy2_all*np.max(total_Hp)/np.max(total_dummy2_all), label='total extra radiation line 5->2\nscaled as its max=H+max');
								if np.max(total_dummy3_all)>0:
									ax.plot(time_crop, total_dummy3_all*np.max(total_Hp)/np.max(total_dummy3_all), label='total extra radiation line 5->2\nscaled as its max=H+max');
								if np.max(total_dummy4_all)>0:
									ax.plot(time_crop, total_dummy4_all*np.max(total_Hp)/np.max(total_dummy4_all), label='total extra radiation line 5->2\nscaled as its max=H+max');
								if np.max(total_dummy5_all)>0:
									ax.plot(time_crop, total_dummy5_all*np.max(total_Hp)/np.max(total_dummy5_all), label='total extra radiation line 5->2\nscaled as its max=H+max');
								if np.max(total_dummy6_all)>0:
									ax.plot(time_crop, total_dummy6_all*np.max(total_Hp)/np.max(total_dummy6_all), label='total extra radiation line 5->2\nscaled as its max=H+max');
								ax.legend(loc='best')
								ax.set_xlabel('time from beginning of pulse [ms]')
								ax.set_ylabel('Particle balance [#]')
								ax.set_title('Particles destroyed and generated via recombination and excitation, 0>r>'+  '%.3g' % (1000*np.max(r_crop)) +' mm, from target to '+str(target_OES_distance)+'mm from it')
								ax.set_yscale('log')
								ax.set_ylim(np.max([ionisation_source,recombination_source,total_Hp,total_H,inflow_max])*1e-8,np.max([ionisation_source,recombination_source,total_Hp,total_H,inflow_max]))
								figure_index += 1
								plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()

								arbitrary_H_temp = 5000	# K, It is the same used for the fittings
								thermal_velocity_H = ( (arbitrary_H_temp*boltzmann_constant_J)/ au_to_kg)**0.5
								temp = read_adf11(scdfile, 'scd', 1, 1, 1, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())
								temp[np.isnan(temp)] = 0
								effective_ionisation_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (10 ** -6)  # in ionisations m^-3 s-1 / (# / m^3)**2
								ionization_length = thermal_velocity_H/(effective_ionisation_rates * merge_ne_prof_multipulse_interp_crop_limited * 1e20 )
								ionization_length = np.where(np.isnan(ionization_length), 0, ionization_length)
								ionization_length = np.where(np.isinf(ionization_length), np.nan, ionization_length)
								ionization_length = np.where(np.isnan(ionization_length), np.nanmax(ionization_length[np.isfinite(ionization_length)]), ionization_length)
								plt.figure();
								plt.pcolor(temp_t, temp_r, ionization_length,vmax=1, cmap='rainbow', norm=LogNorm());
								plt.colorbar().set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
								plt.axes().set_aspect(20)
								plt.xlabel('time [ms]')
								plt.ylabel('radial location [m]')
								plt.title('ionization length of neutral H from ADAS\nH temperature '+str(arbitrary_H_temp)+'K\nOES_multiplier,spatial_factor,time_shift_factor \n' + str(
									[1, spatial_factor, time_shift_factor]))
								figure_index += 1
								plt.savefig(path_where_to_save_everything + mod2 + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
								plt.close()
