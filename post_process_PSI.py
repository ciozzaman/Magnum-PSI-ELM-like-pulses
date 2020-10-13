import numpy as np
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())
# import matplotlib.pyplot as plt
#import .functions
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
import collections

from adas import read_adf15,read_adf11
import os,sys
from PIL import Image
import xarray as xr
import pandas as pd
import copy
from scipy.optimize import curve_fit
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

high_line = 9
# # OES_multiplier_all = [0.1,0.3,0.6,1,1.4,1.8,2.2,3,4,6]
# OES_multiplier_all = [1.6]
# # nH_ne_all = [0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]
# nH_ne_all = [1]
# make_plots = True
# time_shift_factor = -0.05
# spatial_factor = 1.45


OES_multiplier_all = [0.5,0.8,1,1.2,1.5,2,3,4]
nH_ne_all = [0.5,0.6,0.7,0.8,0.9,1]
make_plots = True
time_shift_factor_all = [-0.1,-0.05,-0.03,0,0.03,0.05,0.1]
spatial_factor_all = [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7]



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

	for spatial_factor in spatial_factor_all:

		mod = '/spatial_factor' + str(spatial_factor)
		if not os.path.exists(path_where_to_save_everything + mod):
			os.makedirs(path_where_to_save_everything + mod)

		dx = (18 / 40 * (50.5 / 27.4) / 1e3) * spatial_factor
		xx = np.arange(40) * dx  # m
		xn = np.linspace(0, max(xx), 1000)
		r = np.linspace(-max(xx) / 2, max(xx) / 2, 1000)
		r=r[::10]
		dr=np.median(np.diff(r))



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

		for time_shift_factor in time_shift_factor_all:

			mod = '/spatial_factor' + str(spatial_factor)+'/time_shift_factor' + str(time_shift_factor)
			if not os.path.exists(path_where_to_save_everything + mod):
				os.makedirs(path_where_to_save_everything + mod)


			merge_time = time_shift_factor + merge_time_original
			inverted_profiles = (1 / spatial_factor) * inverted_profiles_original  # in W m^-3 sr^-1  *  4*pi sr  =  W m^-3
			TS_dt=np.nanmedian(np.diff(merge_time))


			if np.max(merge_Te_prof_multipulse)<=0:
				print('merge'+str(merge_ID_target)+" has no recorded temperature")
				continue

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
			temp2=np.zeros_like(inverted_profiles[:,0])
			temp3=np.zeros_like(inverted_profiles[:,0])
			temp4=np.zeros_like(inverted_profiles[:,0])
			interp_range_t = max(dt/2,TS_dt)*1
			interp_range_r = max(dx/2,TS_dr)*1
			for i_t,value_t in enumerate(new_timesteps):
				if np.sum(np.abs(merge_time-value_t) < interp_range_t) == 0:
					continue
				for i_r,value_r in enumerate(np.abs(r)):
					if np.sum(np.abs(TS_r_new - value_r) < interp_range_r) == 0:
						continue
					temp1[i_t,i_r] = np.mean(merge_Te_prof_multipulse[np.abs(merge_time-value_t)<interp_range_t][:,np.abs(TS_r_new-value_r)<interp_range_r])
					temp2[i_t,i_r] = np.max(merge_dTe_multipulse[np.abs(merge_time-value_t)<interp_range_t][:,np.abs(TS_r_new-value_r)<interp_range_r])/(np.sum( np.isfinite(merge_dTe_multipulse[np.abs(merge_time-value_t)<interp_range_t][:,np.abs(TS_r_new-value_r)<interp_range_r]))**0.5)
					temp3[i_t,i_r] = np.mean(merge_ne_prof_multipulse[np.abs(merge_time-value_t)<interp_range_t][:,np.abs(TS_r_new-value_r)<interp_range_r])
					temp4[i_t,i_r] = np.max(merge_dne_multipulse[np.abs(merge_time-value_t)<interp_range_t][:,np.abs(TS_r_new-value_r)<interp_range_r])/(np.sum( np.isfinite(merge_dne_multipulse[np.abs(merge_time-value_t)<interp_range_t][:,np.abs(TS_r_new-value_r)<interp_range_r]))**0.5)

			merge_Te_prof_multipulse_interp=np.array(temp1)
			merge_dTe_prof_multipulse_interp=np.array(temp2)
			merge_ne_prof_multipulse_interp=np.array(temp3)
			merge_dne_prof_multipulse_interp=np.array(temp4)
			temp_r, temp_t = np.meshgrid(r, new_timesteps)
			# plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse_interp,cmap='rainbow');plt.colorbar().set_label('Te [eV]')#;plt.pause(0.01)
			# plt.axes().set_aspect(100)
			# plt.xlabel('time [ms]')
			# plt.ylabel('radial location [m]')
			# plt.title('Te axially averaged and smoothed to match OES resolution')
			# figure_index+=1
			# plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
			# plt.close()
			#
			# plt.figure();plt.pcolor(temp_t,temp_r,merge_ne_prof_multipulse_interp,cmap='rainbow');plt.colorbar().set_label('ne [10^20 #/m^3]')#;plt.pause(0.01)
			# plt.axes().set_aspect(100)
			# plt.xlabel('time [ms]')
			# plt.ylabel('radial location [m]')
			# plt.title('ne axially averaged and smoothed to match OES resolution')
			# figure_index+=1
			# plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
			# plt.close()


			# plt.figure();plt.pcolor(temp_t,temp_r,inverted_profiles[:,0]);plt.colorbar();plt.pause(0.01)




			# merge_Te_prof_multipulse_interp_fun = bisplrep((merge_time*np.ones((len(merge_time),len(TS_r_new))).T).T,TS_r_new*np.ones((len(merge_time),len(TS_r_new))), merge_Te_prof_multipulse,kx=3,ky=3,s=40)#len(merge_time)*len(TS_r_new)/10)
			# # d_interpolated = bisplev(pixels_centre_location[1], pixels_centre_location[0], foil_power_interpolator)
			# merge_Te_prof_multipulse_interp = bisplev(new_timesteps,np.sort(np.abs(r)), merge_Te_prof_multipulse_interp_fun)
			# temp_r, temp_t = np.meshgrid(np.sort(np.abs(r)), new_timesteps)
			# plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse_interp);plt.colorbar();plt.pause(0.01)
			#
			# merge_ne_prof_multipulse_interp_fun = bisplrep((merge_time*np.ones((len(merge_time),len(TS_r_new))).T).T,TS_r_new*np.ones((len(merge_time),len(TS_r_new))), merge_ne_prof_multipulse,kx=5,ky=5,s=100000)#len(merge_time)*len(TS_r_new)/10)
			# merge_ne_prof_multipulse_interp = bisplev(new_timesteps,np.sort(np.abs(r)), merge_ne_prof_multipulse_interp_fun)
			# temp_r, temp_t = np.meshgrid(np.sort(np.abs(r)), new_timesteps)
			# plt.figure();plt.pcolor(temp_t,temp_r,merge_ne_prof_multipulse_interp);plt.colorbar();plt.pause(0.01)
			#
			# # plt.figure();plt.imshow(merge_Te_prof_multipulse_interp,vmin=0);plt.colorbar();plt.pause(0.01)




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
			merge_dTe_prof_multipulse_interp_crop = merge_dTe_prof_multipulse_interp[start_time:end_time,start_r:end_r]
			merge_dne_prof_multipulse_interp_crop = merge_dne_prof_multipulse_interp[start_time:end_time,start_r:end_r]
			inverted_profiles_crop = inverted_profiles[start_time:end_time,:,start_r:end_r]
			inverted_profiles_crop[np.isnan(inverted_profiles_crop)] = 0
			all_fits_crop = all_fits[start_time:end_time]
			# inverted_profiles_crop[inverted_profiles_crop<0] = 0

			x_local = xx - spatial_factor * 17.4 / 1000
			dr_crop = np.median(np.diff(r_crop))
			all_dy = []
			for x_value in np.abs(x_local):
				y = np.zeros_like(r_crop)
				y_prime = np.zeros_like(r_crop)
				for i_r, r_value in enumerate(r_crop):
					# print(r_value)
					if (r_value + dr_crop / 2) < (x_value):
						# print(str([r_value+dr_crop/2,x_value-dx/2]))
						continue
					elif (x_value >= (r_value - dr_crop / 2) and x_value <= (r_value + dr_crop / 2)):
						y[i_r] = np.sqrt((r_value + dr_crop / 2) ** 2 - x_value ** 2)
					# print([r_value+dr_crop/2,x_value])
					# print(y[i_r])
					elif r_value - dr_crop > 0:
						y[i_r] = np.sqrt((r_value + dr_crop / 2) ** 2 - x_value ** 2)
						y_prime[i_r] = np.sqrt((r_value - dr_crop / 2) ** 2 - x_value ** 2)
					else:
						y[i_r] = np.sqrt((r_value + dr_crop / 2) ** 2)
				# print([r_value+dr_crop/2,x_value])
				# print(y[i_r])
				dy = 2 * (y - y_prime)
				all_dy.append(dy)
				if np.sum(np.isnan(dy)) > 0:
					print(str(x_value) + ' is bad')
					print(dy)
			# line_integrated_recombination.append(np.sum(recombination_emissivity*dy,axis=-1))
			all_dy = np.array(all_dy)


			if make_plots:
				plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse_interp_crop,cmap='rainbow');plt.colorbar().set_label('Te [eV]')#;plt.pause(0.01)
				plt.axes().set_aspect(20)
				plt.xlabel('time [ms]')
				plt.ylabel('radial location [m]')
				plt.title('Te axially averaged and smoothed to match OES resolution')
				figure_index+=1
				plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
				plt.close()

				plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse_interp_crop,cmap='rainbow',vmax=0.2);plt.colorbar().set_label('Te [eV]')#;plt.pause(0.01)
				plt.axes().set_aspect(20)
				plt.xlabel('time [ms]')
				plt.ylabel('radial location [m]')
				plt.title('Te axially averaged and smoothed to match OES resolution\nonly values<0.2eV where ADAS data become unreliable')
				figure_index+=1
				plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
				plt.close()

				plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse_interp_crop,cmap='rainbow',vmax=1);plt.colorbar().set_label('Te [eV]')#;plt.pause(0.01)
				plt.axes().set_aspect(20)
				plt.xlabel('time [ms]')
				plt.ylabel('radial location [m]')
				plt.title('Te axially averaged and smoothed to match OES resolution\nonly values<1eV where YACORA data become unreliable')
				figure_index+=1
				plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
				plt.close()


				temp = merge_dTe_prof_multipulse_interp_crop/merge_Te_prof_multipulse_interp_crop
				temp[np.isnan(temp)]=0
				plt.figure();plt.pcolor(temp_t,temp_r,temp,norm=LogNorm(),cmap='rainbow');plt.colorbar().set_label('relative error [au]')#;plt.pause(0.01)
				plt.axes().set_aspect(20)
				plt.xlabel('time [ms]')
				plt.ylabel('radial location [m]')
				plt.title('merge_Te_prof_multipulse_interp_crop uncertainty from TS')
				figure_index+=1
				plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
				plt.close()

				plt.figure();plt.pcolor(temp_t,temp_r,merge_ne_prof_multipulse_interp_crop,cmap='rainbow');plt.colorbar().set_label('ne [10^20 #/m^3]')#;plt.pause(0.01)
				plt.axes().set_aspect(20)
				plt.xlabel('time [ms]')
				plt.ylabel('radial location [m]')
				plt.title('ne axially averaged and smoothed to match OES resolution')
				figure_index+=1
				plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
				plt.close()

				temp = merge_dne_prof_multipulse_interp_crop/merge_ne_prof_multipulse_interp_crop
				temp[np.isnan(temp)]=0
				plt.figure();plt.pcolor(temp_t,temp_r,temp,norm=LogNorm(),cmap='rainbow');plt.colorbar().set_label('relative error [au]')#;plt.pause(0.01)
				plt.axes().set_aspect(20)
				plt.xlabel('time [ms]')
				plt.ylabel('radial location [m]')
				plt.title('merge_ne_prof_multipulse_interp_crop uncertainty from TS')
				figure_index+=1
				plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
				plt.close()



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

			eccess_OES_line4 = []
			lack_OES_line4 = []
			eccess_OES_line4_coord = []

			for nH_ne in nH_ne_all:
				temp_r, temp_t = np.meshgrid(r_crop, time_crop)

				mod = '/spatial_factor' + str(spatial_factor)+'/time_shift_factor' + str(time_shift_factor)+'/nH_ne_'+str(nH_ne)
				if not os.path.exists(path_where_to_save_everything+mod):
					os.makedirs(path_where_to_save_everything+mod)

				recombination_emissivity = (recombination * nH_ne*((merge_ne_prof_multipulse_interp_crop_limited*(10**20))**2)).astype('float')
				if make_plots:
					# plt.figure();plt.pcolor(temp_t,temp_r,recombination_emissivity[0],vmax=np.nanmax(inverted_profiles[:,0]));plt.colorbar();plt.pause(0.01)
					plt.figure();plt.pcolor(temp_t,temp_r,recombination_emissivity[high_line-4],cmap='rainbow');plt.colorbar().set_label('line emission [W m^-3]')#;plt.pause(0.01)
					plt.axes().set_aspect(20)
					plt.xlabel('time [ms]')
					plt.ylabel('radial location [m]')
					plt.title('recombination_emissivity line'+str(high_line)+' from ne,Te\nnH_ne='+str(nH_ne))
					figure_index+=1
					plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
					plt.close()

					plt.figure();plt.pcolor(temp_t,temp_r,recombination_emissivity[0],cmap='rainbow');plt.colorbar().set_label('line emission [W m^-3]')#;plt.pause(0.01)
					plt.axes().set_aspect(20)
					plt.xlabel('time [ms]')
					plt.ylabel('radial location [m]')
					plt.title('recombination_emissivity line4 from ne,Te\nnH_ne='+str(nH_ne))
					figure_index+=1
					plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
					plt.close()

					plt.figure();plt.pcolor(temp_t,temp_r,recombination_emissivity[high_line-4]/np.max(recombination_emissivity[high_line-4]),cmap='rainbow');plt.colorbar().set_label('relative line emission [au]')#;plt.pause(0.01)
					plt.axes().set_aspect(20)
					plt.xlabel('time [ms]')
					plt.ylabel('radial location [m]')
					plt.title('relative recombination_emissivity line'+str(high_line)+' from ne,Te\nnH_ne='+str(nH_ne))
					figure_index+=1
					plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
					plt.close()

					plt.figure();plt.pcolor(temp_t,temp_r,excitation[high_line-4]/np.max(excitation[high_line-4]),cmap='rainbow');plt.colorbar().set_label('relative line emission [au]')#;plt.pause(0.01)
					plt.axes().set_aspect(20)
					plt.xlabel('time [ms]')
					plt.ylabel('radial location [m]')
					plt.title('relative excitation line'+str(high_line)+' from ne,Te\nnH_ne='+str(nH_ne))
					figure_index+=1
					plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
					plt.close()

				for OES_multiplier in OES_multiplier_all:
					temp_r, temp_t = np.meshgrid(r_crop, time_crop)

					mod = '/spatial_factor' + str(spatial_factor)+'/time_shift_factor' + str(time_shift_factor) + '/nH_ne_' + str(nH_ne) + '/OES_multiplier_'+str(OES_multiplier)

					if not os.path.exists(path_where_to_save_everything + mod):
						os.makedirs(path_where_to_save_everything + mod)

					# the peaks of excitataion and density are apart, so I scale up the OES measurements to match the expected data where I have the maximum total radiation

					# integrated_recombination_emissitivy = np.sum(recombination_emissivity[high_line-4]*dr*dt*2*np.pi*r_crop,axis=-1)
					# integrated_measured_emissitivy = np.sum(inverted_profiles_crop[:,high_line-4]*dr*dt*2*np.pi*r_crop,axis=-1)
					#
					#
					# difference_all=[]
					# for line in range(4,13):
					# 	OES_multiplier = 1
					# 	difference_all_line = inverted_profiles_crop[:,line-4]*OES_multiplier - recombination_emissivity[line-4]
					# 	min_mean = 2*np.std(difference_all_line)
					# 	while np.mean(difference_all_line)<min_mean:
					# 		OES_multiplier+=1.2
					# 		difference_all_line = inverted_profiles_crop[:, line - 4] * OES_multiplier - recombination_emissivity[line - 4]
					# 	print('line ' + str(line))
					# 	print(OES_multiplier)
					# 	difference_all.append(difference_all_line)
					# 	plt.figure();
					# 	plt.pcolor(temp_t, temp_r, difference_all_line,vmax=0, cmap='rainbow');
					# 	plt.colorbar().set_label('line emission [W m^-3 sr^-1]')  # ;plt.pause(0.01)
					# 	plt.axes().set_aspect(20)
					# 	plt.xlabel('time [ms]')
					# 	plt.ylabel('radial location [m]')
					# 	plt.title('difference between inverted_profiles and recombination_emissivity\n and recombination_emissivity for line' + str(line) + '\nI attribute this to excitation')
					# 	plt.close()
					# difference_all = np.array(difference_all)
					#
					#
					#
					# difference_all=[]
					# for line in range(4,13):
					# 	OES_multiplier = 1
					# 	difference_all_line = inverted_profiles_crop[:,line-4]*OES_multiplier - recombination_emissivity[line-4]
					# 	min_mean = 2*np.std(difference_all_line)
					# 	while np.mean(difference_all_line)<0:
					# 		OES_multiplier+=1.2
					# 		difference_all_line = inverted_profiles_crop[:, line - 4] * OES_multiplier - recombination_emissivity[line - 4]
					# 	print('line ' + str(line))
					# 	print(OES_multiplier)
					# 	difference_all.append(difference_all_line)
					# 	plt.figure();
					# 	plt.pcolor(temp_t, temp_r, difference_all_line,vmax=0, cmap='rainbow');
					# 	plt.colorbar().set_label('line emission [W m^-3 sr^-1]')  # ;plt.pause(0.01)
					# 	plt.axes().set_aspect(20)
					# 	plt.xlabel('time [ms]')
					# 	plt.ylabel('radial location [m]')
					# 	plt.title('difference between inverted_profiles and recombination_emissivity\n and recombination_emissivity for line' + str(line) + '\nI attribute this to excitation')
					# 	plt.close()
					# difference_all = np.array(difference_all)


					# if False:
					# 	OES_multiplier = integrated_recombination_emissitivy.max()/integrated_measured_emissitivy[integrated_recombination_emissitivy.argmax()]
					# 	OES_multiplier = recombination_emissivity[high_line-4].max()/(inverted_profiles_crop[:,high_line-4].flatten()[recombination_emissivity[high_line-4].argmax()])
					# elif True:
					# 	OES_multiplier = 1
					difference = inverted_profiles_crop[:,high_line-4]*OES_multiplier - recombination_emissivity[high_line-4]
					if make_plots:
						plt.figure();plt.pcolor(temp_t,temp_r,difference,cmap='rainbow',vmin=max(np.min(difference),-np.max(difference)));plt.colorbar().set_label('line emission [W m^-3]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('difference between inverted_profiles\n and recombination_emissivity for line'+str(high_line)+'\nI attribute this to excitation\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()


					# I want to find what is the best line to calculate the neutral hydrogen atomic density
					# best means that it fits best with ower lines

					if False:
						score = []
						for line_index in [6,7,8,9,10,11,12]:
							difference = inverted_profiles_crop[:, line_index - 4] * OES_multiplier - recombination_emissivity[line_index - 4]
							h_atomic_density = difference/(excitation[line_index-4] * (merge_ne_prof_multipulse_interp_crop*(10**20))).astype('float')
							h_atomic_density[np.logical_not(np.isfinite(h_atomic_density))]=0
							h_atomic_density[h_atomic_density<0]=0

							excitation_emissivity = (excitation * (merge_ne_prof_multipulse_interp_crop*(10**20))*(h_atomic_density)).astype('float')

							residual_emission=[]
							for index in range(len(recombination_emissivity)):
								residual_emission.append(inverted_profiles_crop[:,index]*OES_multiplier-recombination_emissivity[index]-excitation_emissivity[index])
							residual_emission=np.array(residual_emission)

							score.append(np.min(residual_emission[0]))
						print('score')
						print(score)


					h_atomic_density = difference/(excitation[high_line-4] * (merge_ne_prof_multipulse_interp_crop_limited*(10**20))).astype('float')
					h_atomic_density[np.logical_not(np.isfinite(h_atomic_density))]=0
					h_atomic_density[h_atomic_density<0]=0
					if np.max(h_atomic_density[np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007])<=0:
						print('merge'+str(merge_ID_target)+' , nH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier)+" didn't allow for any neutral atomic hydrogen")
						continue
					if make_plots:
						plt.figure();plt.pcolor(temp_t,temp_r,h_atomic_density,norm=LogNorm(),cmap='rainbow',vmax=np.max(h_atomic_density[np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label('n0 [# m^-3]')#;plt.pause(0.01)
						# plt.figure();plt.pcolor(temp_t,temp_r,h_atomic_density,norm=LogNorm(),cmap='rainbow',vmax=np.);plt.colorbar().set_label('n0 [# m^-3]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('estimated hydrogen neutral atomic density from excitation\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()


					excitation_emissivity = (excitation * (merge_ne_prof_multipulse_interp_crop*(10**20))*(h_atomic_density)).astype('float')
					if make_plots:
						# plt.figure();plt.pcolor(temp_t,temp_r,excitation_emissivity[0],cmap='rainbow',vmax=np.max(excitation_emissivity[0][np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label('line emission [W m^-3 sr^-1]')#;plt.pause(0.01)
						plt.figure();plt.pcolor(temp_t,temp_r,excitation_emissivity[0],cmap='rainbow');plt.colorbar().set_label('line emission [W m^-3]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('excitation_emissivity line4\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						plt.figure();plt.pcolor(temp_t,temp_r,excitation_emissivity[high_line-4],cmap='rainbow',vmax=np.max(excitation_emissivity[high_line-4][np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label('line emission [W m^-3]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('excitation_emissivity line'+str(high_line)+'\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()


					residual_emission=[]
					for index in range(len(recombination_emissivity)):
						residual_emission.append(inverted_profiles_crop[:,index]*OES_multiplier-recombination_emissivity[index]-excitation_emissivity[index])
					residual_emission=np.array(residual_emission)
					eccess_OES_line4.append(np.min(residual_emission[0]))
					lack_OES_line4.append(np.max(residual_emission[0]))
					eccess_OES_line4_coord.append([nH_ne,OES_multiplier])
					if make_plots:
						# residual_emission = inverted_profiles_crop-recombination_emissivity-excitation_emissivity
						plt.figure();plt.pcolor(temp_t,temp_r,residual_emission[0],cmap='rainbow',vmin=np.min(residual_emission[0][np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label('line emission [W m^-3]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('residual emissivity, attribuited to MAR, line4\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						temp = residual_emission[0]/recombination_emissivity[0]
						temp[np.logical_not( np.isfinite(temp))] = 0
						plt.figure();plt.pcolor(temp_t,temp_r,temp,cmap='rainbow',vmax=np.max(temp[np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]),vmin=np.min(temp[np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label(' relative line emission [au]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('relative residual emissivity, attribuited to MAR, line4\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						plt.figure();plt.pcolor(temp_t,temp_r,residual_emission[1],cmap='rainbow',vmin=np.min(residual_emission[1][np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label('line emission [W m^-3]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('residual emissivity, attribuited to MAR, line5\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						temp = residual_emission[1]/recombination_emissivity[1]
						temp[np.logical_not( np.isfinite(temp))] = 0
						plt.figure();plt.pcolor(temp_t,temp_r,temp,cmap='rainbow',vmax=np.max(temp[np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]),vmin=np.min(temp[np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label(' relative line emission [au]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('relative residual emissivity, attribuited to MAR, line5\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						plt.figure();plt.pcolor(temp_t,temp_r,residual_emission[2],cmap='rainbow',vmin=np.min(residual_emission[2][np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label('line emission [W m^-3]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('residual emissivity, attribuited to MAR, line6\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						temp = residual_emission[2]/recombination_emissivity[2]
						temp[np.logical_not( np.isfinite(temp))] = 0
						plt.figure();plt.pcolor(temp_t,temp_r,temp,cmap='rainbow',vmax=np.max(temp[np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]),vmin=np.min(temp[np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label(' relative line emission [au]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('relative residual emissivity, attribuited to MAR, line6\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						plt.figure();plt.pcolor(temp_t,temp_r,residual_emission[high_line-4],cmap='rainbow',vmin=np.min(residual_emission[high_line-4][np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label('line emission [W m^-3]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('residual emissivity, attribuited to MAR, line'+str(high_line)+'\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						temp = residual_emission[high_line-4]/recombination_emissivity[high_line-4]
						temp[np.logical_not( np.isfinite(temp))] = 0
						plt.figure();plt.pcolor(temp_t,temp_r,temp,cmap='rainbow',vmax=np.max(temp[np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]),vmin=np.min(temp[np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label(' relative line emission [au]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('relative residual emissivity, attribuited to MAR, line'+str(high_line)+'\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()


						temp = read_adf11(scdfile, 'scd', 1,1,1,merge_Te_prof_multipulse_interp_crop_limited.flatten(), (merge_ne_prof_multipulse_interp_crop_limited*10**(20-6)).flatten())
						temp[np.isnan(temp)] = 0
						effective_ionisation_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop)))*(10**-6)	# in ionisations m^-3 s-1 / (# / m^3)**2
						effective_ionisation_rates = (effective_ionisation_rates * (merge_ne_prof_multipulse_interp_crop*(10**20))*h_atomic_density).astype('float')
						plt.figure();plt.pcolor(temp_t,temp_r,effective_ionisation_rates,cmap='rainbow');plt.colorbar().set_label('effective_ionisation_rates [# m^-3 s-1]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('effective_ionisation_rates\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()


						temp = read_adf11(acdfile, 'acd', 1,1,1,merge_Te_prof_multipulse_interp_crop_limited.flatten(), (merge_ne_prof_multipulse_interp_crop_limited*10**(20-6)).flatten())
						temp[np.isnan(temp)] = 0
						effective_recombination_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop)))*(10**-6)	# in recombinations m^-3 s-1 / (# / m^3)**2
						effective_recombination_rates = (effective_recombination_rates * (merge_ne_prof_multipulse_interp_crop*(10**20))**2).astype('float')
						plt.figure();plt.pcolor(temp_t,temp_r,effective_recombination_rates,cmap='rainbow');plt.colorbar().set_label('effective_recombination_rates [# m^-3 s-1]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('effective_recombination_rates (three body plus radiative)\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()



					line_integrated_recombination = []
					line_integrated_excitation = []
					line_integrated_emission = []
					for dy in all_dy:
						line_integrated_recombination.append(np.sum(recombination_emissivity*dy,axis=-1))
						line_integrated_excitation.append(np.sum(excitation_emissivity*dy,axis=-1))
						line_integrated_emission.append(np.sum((recombination_emissivity+excitation_emissivity)*dy,axis=-1))
					line_integrated_recombination = np.array(line_integrated_recombination)
					line_integrated_excitation = np.array(line_integrated_excitation)
					line_integrated_emission = np.array(line_integrated_emission)

					all_fits[np.isnan(all_fits)]=0
					if make_plots:
						plt.figure();plt.plot(np.nanmax(4*np.pi*all_fits,axis=(0,1)),label='from OES')
						plt.plot(np.max(line_integrated_recombination[:,:,np.logical_and(time_crop<0.9,time_crop>0.4)],axis=(0,-1)),label='expected')
						plt.legend(loc='best')
						plt.ylabel('line integrated emission peak value [W m^-2]')
						plt.xlabel('line index-4')
						plt.title('nH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()


					# plt.figure();plt.imshow(line_integrated_recombination[:,0],cmap='rainbow',vmax=400);plt.colorbar().set_label('effective_recombination_rates [# m^-3 s-1]')#;plt.pause(0.01)
					# plt.figure();
					# plt.imshow(line_integrated_recombination[:, 5], cmap='rainbow');
					# plt.colorbar().set_label('effective_recombination_rates [# m^-3 s-1]')  # ;plt.pause(0.01)

					temp_xx, temp_t = np.meshgrid(xx, time_crop)
					if make_plots:
						plt.figure();plt.pcolor(temp_t,temp_xx,line_integrated_recombination[:,0].T,cmap='rainbow');plt.colorbar().set_label('line integrated emission [W m^-2]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('expected line integrated recombination line'+str(4)+'\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						plt.figure();plt.pcolor(temp_t,temp_xx,line_integrated_excitation[:,0].T,cmap='rainbow');plt.colorbar().set_label('line integrated emission [W m^-2]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('expected line integrated excitation line'+str(4)+'\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						plt.figure();plt.pcolor(temp_t,temp_xx,line_integrated_emission[:,0].T,cmap='rainbow');plt.colorbar().set_label('line integrated emission [W m^-2]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('expected line integrated total emission line'+str(4)+'\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						plt.figure();plt.pcolor(temp_t,temp_xx,line_integrated_recombination[:,5-4].T,cmap='rainbow');plt.colorbar().set_label('line integrated emission [W m^-2]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('expected line integrated recombination line'+str(5)+'\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						plt.figure();plt.pcolor(temp_t,temp_xx,line_integrated_excitation[:,5-4].T,cmap='rainbow');plt.colorbar().set_label('line integrated emission [W m^-2]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('expected line integrated excitation line'+str(5)+'\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						plt.figure();plt.pcolor(temp_t,temp_xx,line_integrated_emission[:,5-4].T,cmap='rainbow');plt.colorbar().set_label('line integrated emission [W m^-2]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('expected line integrated total emission line'+str(5)+'\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						plt.figure();plt.pcolor(temp_t,temp_xx,line_integrated_recombination[:,6-4].T,cmap='rainbow');plt.colorbar().set_label('line integrated emission [W m^-2]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('expected line integrated recombination line'+str(6)+'\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						plt.figure();plt.pcolor(temp_t,temp_xx,line_integrated_excitation[:,6-4].T,cmap='rainbow');plt.colorbar().set_label('line integrated emission [W m^-2]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('expected line integrated excitation line'+str(6)+'\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						plt.figure();plt.pcolor(temp_t,temp_xx,line_integrated_emission[:,6-4].T,cmap='rainbow');plt.colorbar().set_label('line integrated emission [W m^-2]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('expected line integrated total emission line'+str(6)+'\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						plt.figure();plt.pcolor(temp_t,temp_xx,line_integrated_recombination[:,high_line-4].T,cmap='rainbow');plt.colorbar().set_label('line integrated emission [W m^-2]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('expected line integrated recombination line'+str(high_line)+'\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						plt.figure();plt.pcolor(temp_t,temp_xx,line_integrated_excitation[:,high_line-4].T,cmap='rainbow');plt.colorbar().set_label('line integrated emission [W m^-2]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('expected line integrated excitation line'+str(high_line)+'\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()

						plt.figure();plt.pcolor(temp_t,temp_xx,line_integrated_emission[:,high_line-4].T,cmap='rainbow');plt.colorbar().set_label('line integrated emission [W m^-2]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('expected line integrated total emission line'+str(high_line)+'\nnH_ne='+str(nH_ne)+' , OES_multiplier='+str(OES_multiplier))
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()


						temp_r, temp_t = np.meshgrid(r_crop, time_crop)
						plt.figure();plt.pcolor(temp_t,temp_r,OES_multiplier*inverted_profiles_crop[:,high_line-4],cmap='rainbow');plt.colorbar().set_label('line emission [W m^-3]')#;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('inverted_profiles_crop line'+str(high_line)+' from OES' +'\n OES_multiplier='+str(OES_multiplier) )
						figure_index+=1
						plt.savefig(path_where_to_save_everything+mod+ '/post_process_'+str(figure_index)+'.eps',bbox_inches='tight')
						plt.close()

						plt.figure();
						plt.pcolor(temp_t, temp_r, OES_multiplier*inverted_profiles_crop[:, 0], cmap='rainbow');
						plt.colorbar().set_label('line emission [W m^-3]')  # ;plt.pause(0.01)
						plt.axes().set_aspect(20)
						plt.xlabel('time [ms]')
						plt.ylabel('radial location [m]')
						plt.title('inverted_profiles_crop line4 from OES'+'\n OES_multiplier='+str(OES_multiplier))
						figure_index += 1
						plt.savefig(path_where_to_save_everything+mod + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
						plt.close()




			temp_xx, temp_t = np.meshgrid(xx, time_crop)
			plt.figure();
			plt.pcolor(temp_t, temp_xx, 4 * np.pi * all_fits_crop[:, :, 0], cmap='rainbow');
			plt.colorbar().set_label('line integrated emission [W m^-2]')  # ;plt.pause(0.01)
			plt.axes().set_aspect(20)
			plt.xlabel('time [ms]')
			plt.ylabel('radial location [m]')
			plt.title('line integrated emission line' + str(4))
			figure_index += 1
			plt.savefig(path_where_to_save_everything + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
			plt.close()

			plt.figure();
			plt.pcolor(temp_t, temp_xx, 4 * np.pi * all_fits_crop[:, :, 5 - 4], cmap='rainbow');
			plt.colorbar().set_label('line integrated emission [W m^-2]')  # ;plt.pause(0.01)
			plt.axes().set_aspect(20)
			plt.xlabel('time [ms]')
			plt.ylabel('radial location [m]')
			plt.title('line integrated emission line' + str(5))
			figure_index += 1
			plt.savefig(path_where_to_save_everything + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
			plt.close()

			plt.figure();
			plt.pcolor(temp_t, temp_xx, 4 * np.pi * all_fits_crop[:, :, 6 - 4], cmap='rainbow');
			plt.colorbar().set_label('line integrated emission [W m^-2]')  # ;plt.pause(0.01)
			plt.axes().set_aspect(20)
			plt.xlabel('time [ms]')
			plt.ylabel('radial location [m]')
			plt.title('line integrated emission line' + str(6))
			figure_index += 1
			plt.savefig(path_where_to_save_everything + '/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
			plt.close()

			plt.figure();
			plt.pcolor(temp_t, temp_xx, 4 * np.pi * all_fits_crop[:, :, high_line - 4], cmap='rainbow');
			plt.colorbar().set_label('line integrated emission [W m^-2]')  # ;plt.pause(0.01)
			plt.axes().set_aspect(20)
			plt.xlabel('time [ms]')
			plt.ylabel('radial location [m]')
			plt.title('line integrated emission line' + str(high_line))
			figure_index += 1
			plt.savefig(path_where_to_save_everything +'/post_process_' + str(figure_index) + '.eps',bbox_inches='tight')
			plt.close()



	eccess_OES_line4 = -np.array(eccess_OES_line4)
	lack_OES_line4 = np.array(lack_OES_line4)
	eccess_OES_line4_coord = np.array(eccess_OES_line4_coord)

	plt.figure()
	for nH_ne_coord in np.unique(eccess_OES_line4_coord[:,0]):
		OES_multiplier_coord = eccess_OES_line4_coord[eccess_OES_line4_coord[:,0]==nH_ne_coord][:,1]
		plt.plot(OES_multiplier_coord,eccess_OES_line4[eccess_OES_line4_coord[:,0]==nH_ne_coord],label='nH_ne='+str(nH_ne_coord))
	plt.xlabel('OES_multiplier [au]')
	plt.ylabel('eccess line radiation [W / m^3]')
	plt.title('minimum of the difference between measured and expected line 4 emission\n(it should be =0)')
	plt.legend(loc='best')
	plt.semilogy()
	figure_index+=1
	plt.savefig(path_where_to_save_everything+ '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()


	break








# # This is just to understand what are the Te/ne combinations I care about
# plt.figure()
# for merge_ID_target in [85,86,87,88,89,92,93,94,96,97]:
#
# 	path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)
# 	merge_Te_prof_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_Te_prof_multipulse']
# 	merge_Te_prof_multipulse[merge_Te_prof_multipulse<=0]=np.nan
# 	merge_ne_prof_multipulse = 10**20*np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_ne_prof_multipulse']
# 	merge_ne_prof_multipulse[merge_ne_prof_multipulse <= 0] = np.nan
# 	# plt.plot(np.log(merge_Te_prof_multipulse.flatten()), np.log(merge_ne_prof_multipulse.flatten().astype('float')), 'o',label='merge' + str(merge_ID_target))
# 	plt.plot(merge_Te_prof_multipulse.flatten(), merge_ne_prof_multipulse.flatten().astype('float'), '+',label='merge' + str(merge_ID_target))
#
# plt.semilogy()
# plt.semilogx()
# plt.xlabel('Te [eV]')
# plt.ylabel('ne [# / m^3]')
# plt.grid()
#
# y = lambda x,m,q : m*x + q
# y(np.linspace(0,3,10),2,42)
# plt.plot(np.exp(np.linspace(-1,3,10)),np.exp(y(np.linspace(-1,3,10),2,43)),'k--')
# plt.plot(np.exp(np.linspace(-0.5,-3.5,10)),np.exp(np.linspace(41,41,10)),'--',label='6.4e+17#/m^3')
#
# # plt.plot(np.exp(np.linspace(0,-3.5,10)),np.exp(y(np.linspace(0,-3.5,10),0.75,49.5)),'k--')
# # plt.plot(np.exp(np.linspace(0,3,10)),np.exp(np.linspace(50.3,50.3,10)),'k--')
# plt.plot(np.exp(np.linspace(0.5,-3.5,10)),np.exp(y(np.linspace(0.5,-3.5,10),0.9,49.9)),'k--')
# plt.plot(np.exp(np.linspace(0.5,3,10)),np.exp(np.linspace(49.9+0.45,49.9+0.45,10)),'--',label='7.36e+21#/m^3')
#
# plt.plot(np.linspace(0.1,0.1,10),np.linspace(10**18,10**21,10),'--',label='0.1eV')
# plt.plot(np.linspace(0.03,0.03,10),np.linspace(0.7*10**18,2*10**20,10),'--',label='0.03eV')
# plt.plot(np.linspace(1,1,10),np.linspace(1*10**18,5*10**21,10),'--',label='1eV')
#
# plt.plot(np.linspace(15,15,10),np.linspace(3*10**20,np.exp(49.9+0.45),10),'--',label='15eV')
# plt.title('Te/ne paramenter space')
# plt.legend(loc='best')
#
# plt.pause(0.01)




