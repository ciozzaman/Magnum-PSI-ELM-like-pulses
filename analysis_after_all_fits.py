import numpy as np
import matplotlib.pyplot as plt
#import .functions
import os,sys
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
from functions.spectools import rotate,do_tilt, binData
from functions.fabio_add import find_nearest_index,multi_gaussian,all_file_names,load_dark,find_index_of_file,get_metadata,movie_from_data,get_angle_no_lines,do_tilt_no_lines,four_point_transform,fix_minimum_signal,get_bin_and_interv_no_lines,examine_current_trace
from functions.GaussFitData import doLateralfit_time_tependent
from PIL import Image
import xarray as xr
import pandas as pd
import copy
from scipy.optimize import curve_fit
from scipy import interpolate

#waveLengths = [656.45377,486.13615,434.0462,410.174,397.0072,388.9049,383.5384] + list((364.6*(n*n)/(n*n-4)))
waveLengths = [486.13615,434.0462,410.174,397.0072,388.9049,383.5384]
waveLengths_interp = interpolate.interp1d(waveLengths[:2], [1.39146712e+03,8.83761543e+02],fill_value='extrapolate')
#pathfiles = '/home/ff645/GoogleDrive/ff645/YPI - Fabio/Collaboratory/Experimental data/tests/2019-04-25/final_test/Untitled_1/Pos0'
# pathfiles = '/home/ff645/GoogleDrive/ff645/YPI - Fabio/Collaboratory/Experimental data/tests/temp - Desktop pc/Untitled_11/Pos0'
# pathfiles = '/home/ff645/GoogleDrive/ff645/YPI - Fabio/Collaboratory/Experimental data/tests/test_data_folder/2019-04-25/01/Untitled_1/Pos0'


#rotation = 0 #deg
#tilt = 0 #deg
#read_out_time = 10280 #ns
#conventional_read_out_time = 10000 # ns
#extra_shifts_pulse_before = 300 #
#extra_shifts_pulse_after = 300 #
#row_skip = 5 #


calculate_geometry = False



#fdir = '/home/ff645/GoogleDrive/ff645/YPI - Fabio/Collaboratory/Experimental data/tests/test_data_folder'
#df_log = pd.read_csv('functions/Log/shots_2.csv',index_col=0)
#df_settings = pd.read_csv('functions/Log/settin	gs_2.csv',index_col=0)

fdir = '/home/ffederic/work/Collaboratory/test/experimental_data'
df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
df_settings = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_3.csv',index_col=0)


#waveLcoefs = functions.do_waveL_Calib(df_settings,df_log,geom,fdir=fdir)
#print('waveLcoefs= '+str(waveLcoefs))



# for i in range(17,32):
# 	all_fits = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(i) + '_all_fits.npy')
# 	doLateralfit_time_tependent(df_settings, all_fits, i)







merge=17
geom = pd.DataFrame(columns=['angle', 'tilt', 'binInterv', 'bin00a', 'bin00b'])
geom.loc[0] = [0.01759, np.nan, 29.202126, 53.647099, 53.647099]
geom['angle'][0]=0.0534850538519
geom['binInterv'] = 29.3051619433
geom['tilt'] = 0.0015142337977
geom['bin00a'] = 47.6592283252
geom['bin00b'] = 47.6592283252
waveLcoefs = np.ones((2, 3)) * np.nan
waveLcoefs[1] = [-2.21580743e-06 ,  1.07315743e-01  , 3.40712312e+02]  # from examine_sensitivity.py
all_fits = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge) + '/merge' + str(merge) + '_all_fits.npy')
binnedSens = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/Sensitivity_1.npy')
binned_data = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge) + '/merge' + str(merge) + '_binned_data.npy')
composed_array = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge)+'_composed_array.npy')
inverted_profiles = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge)+'/inverted_profiles.npy')


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
# composed_array = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge) + '/merge' + str(merge)+'_composed_array.npy')
inverted_profiles = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge) + '/inverted_profiles.npy')
merge_values = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge) + '/merge' + str(merge) + '_merge_tot.npz')['merge_values']
merge_time = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge) + '/merge' + str(merge) + '_merge_tot.npz')['merge_time']
merge_row = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge) + '/merge' + str(merge) + '_merge_tot.npz')['merge_row']
merge_Gain = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge) + '/merge' + str(merge) + '_merge_tot.npz')['merge_Gain']
merge_overexposed = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge) + '/merge' + str(merge) + '_merge_tot.npz')['merge_overexposed']


for merge in range(66,83+1,1):	# !! I already did merge84 !!
	merge_values = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge) + '/merge' + str(merge) + '_merge_tot.npz')['merge_values']
	merge_time = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge) + '/merge' + str(merge) + '_merge_tot.npz')['merge_time']
	merge_row = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge) + '/merge' + str(merge) + '_merge_tot.npz')['merge_row']
	merge_Gain = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge) + '/merge' + str(merge) + '_merge_tot.npz')['merge_Gain']
	merge_overexposed = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge) + '/merge' + str(merge) + '_merge_tot.npz')['merge_overexposed']
	merge_time += 0.19
	np.savez_compressed('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge) + '/merge' + str(merge) + '_merge_tot',
						merge_values=merge_values, merge_time=merge_time, merge_row=merge_row, merge_Gain=merge_Gain,
						merge_overexposed=merge_overexposed)

dx = 18 / 40 * (50.5 / 27.4) / 1e3
xx = np.arange(40) * dx  # m
xn = np.linspace(0, max(xx), 1000)
r = np.linspace(-max(xx) / 2, max(xx) / 2, 1000)
r=r[::10]

merge_time_window=[-10,10]
conventional_time_step = 0.05
new_timesteps = np.linspace(merge_time_window[0]+0.5, merge_time_window[1]-0.5,int((merge_time_window[1]-0.5-(merge_time_window[0]+0.5))/conventional_time_step+1))
row_steps = np.linspace(0, 1100 - 1, 1100)



all_merge = [17,18,19,20,21,22,23]
all_merge = range(83,66,-1)
# inverted_profiles = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(17)+'/inverted_profiles.npy')
# Data from wikipedia
energy_difference = np.array([1.89,2.55,2.86,3.03,3.13,3.19,3.23,3.26, 3.29 ])	#eV
# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
statistical_weigth = np.array([32,50,72,98,128,162,200,242,288])	#gi-gk
einstein_coeff = np.array([8.4193e-2,2.53044e-2,9.7320e-3,4.3889e-3,2.2148e-3,1.2156e-3,7.1225e-4,4.3972e-4,2.8337e-4])*1e8	#1/s
J_to_eV = 6.242e18
# Used formula 2.3 in Rion Barrois thesys, 2017

color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise']
for merge_ID_target in all_merge:
	all_fits = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) +'/merge' + str(merge_ID_target) + '_all_fits.npy')
	# doLateralfit_time_tependent(df_settings, all_fits, merge_ID_target,np.min(new_timesteps),np.max(new_timesteps),conventional_time_step,dx,xx,r)
	inverted_profiles = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/inverted_profiles.npy')
	# binned_data = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) +'/merge' + str(merge_ID_target) + '_binned_data.npy')
	sample_time_step = []
	for time in [-2, 0, 0.1, 0.4, 0.5, 0.7, 1, 2, 4]:  # times in ms that I want to take a better look at
		sample_time_step.append(int((np.abs(new_timesteps - time)).argmin()))
	sample_radious = []
	for radious in [0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.009,0.0012]:  # radious in m that I want to take a better look at
		sample_radious.append(int((np.abs(r - radious)).argmin()))
	try:
		for time_step in sample_time_step:
			# time_step=19
			plt.figure(figsize=(20, 10))
			to_plot_all = []
			for index, loc in enumerate(r[sample_radious]):
				if (loc > 0 and loc < 0.015):
					to_plot = np.divide(np.pi * 4 * inverted_profiles[time_step, :, index],statistical_weigth * einstein_coeff * energy_difference / J_to_eV)
					if np.sum(to_plot > 0) > 4:
						to_plot_all.append(to_plot > 0)
			# to_plot_all.append(to_plot)
			to_plot_all = np.array(to_plot_all)
			relative_index = int(np.max((np.sum(to_plot_all, axis=(0)) == np.max(np.sum(to_plot_all, axis=(0)))) * np.linspace(1, len(to_plot),len(to_plot)))) - 1
			for index, loc in enumerate(r[sample_radious]):
				if (loc > 0 and loc < 0.015):
					to_plot = np.divide(np.pi * 4 * inverted_profiles[time_step, :, index],statistical_weigth * einstein_coeff * energy_difference / J_to_eV)
					plt.plot(energy_difference, to_plot / np.min(to_plot[relative_index]),label='axial pos=' + str(np.around(loc, decimals=4)))
			plt.legend(loc='best')
			plt.title('Boltzmann plot at ' + str(np.around(new_timesteps[time_step], decimals=2)) + 'ms')
			plt.semilogy()
			plt.semilogx()
			plt.xlabel('trensition energy [eV]')
			plt.ylabel('relative population density scaled to the min value [au]')
			plt.savefig(path_where_to_save_everything + '/Bplot_relative_time_' + str(np.around(new_timesteps[time_step], decimals=2)) + '.eps', bbox_inches='tight')
			plt.close()

			plt.figure(figsize=(20, 10))
			for iR in range(np.shape(inverted_profiles)[1]):
				plt.plot(r*1000,np.divide(np.pi * 4 * inverted_profiles[time_step,iR],statistical_weigth[iR] * einstein_coeff[iR] * energy_difference[iR] / J_to_eV),color[iR],label='n='+str(iR+4))
			plt.legend(loc='best')
			plt.title('Excited states density plot at ' + str(np.around(new_timesteps[time_step], decimals=2)) + 'ms')
			# plt.semilogy()
			# plt.semilogx()
			plt.xlabel('radial location [mm]')
			plt.ylabel('excited state density [m^-3]')
			plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/Excited_dens_time_' + str(np.around(new_timesteps[time_step], decimals=2)) + '.eps', bbox_inches='tight')
			plt.close()


			plt.figure(figsize=(20, 10))
			to_plot_all = []
			for index, loc in enumerate(r[sample_radious]):
				if (loc > 0 and loc < 0.015):
					to_plot = np.divide(np.pi * 4 * inverted_profiles[time_step, :, index],
										statistical_weigth * einstein_coeff * energy_difference / J_to_eV)
					plt.plot(energy_difference, to_plot, label='axial pos=' + str(np.around(loc, decimals=4)))
			plt.legend(loc='best')
			plt.title('Boltzmann plot at ' + str(np.around(new_timesteps[time_step], decimals=2)) + 'ms')
			plt.semilogy()
			plt.semilogx()
			plt.xlabel('trensition energy [eV]')
			plt.ylabel('population density scaled to the min value [#/m^3]')
			plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/Bplot_absolute_time_' + str(np.around(new_timesteps[time_step], decimals=2)) + '.eps',bbox_inches='tight')
			plt.close()
	except:
		print('failed merge' + str(merge_ID_target))
		plt.close()

	for index, loc in enumerate(sample_radious):
		plt.figure(figsize=(20, 10))
		reference_point = inverted_profiles[:, 0, loc].argmax()
		for iR in range(np.shape(inverted_profiles)[1]):
			plt.plot(new_timesteps, inverted_profiles[:, iR, loc] / inverted_profiles[reference_point, iR, loc],
					 color[iR],
					 linewidth=0.3, label='line ' + str(iR + 4))
		plt.title('Relative line intensity at radious' + str(np.around(r[loc], decimals=3)) + 'mm')
		# plt.semilogy()
		plt.xlabel('time [ms]')
		plt.ylabel('relative intensity scaled to the max value [au]')
		plt.legend(loc='best')
		plt.ylim(bottom=0)
		plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(
			merge_ID_target) + '/rel_intens_r_' + str(np.around(r[loc], decimals=3)) + '.eps', bbox_inches='tight')
		plt.close()

	# for index, loc in enumerate([26, 28, 30, 32, 34, 36, 40, 45]):
	# 	plt.figure(figsize=(20, 10))
	# 	reference_point = inverted_profiles[:, 0, loc].argmax()
	# 	max_ratio = np.max(inverted_profiles[reference_point, :-2, loc] / inverted_profiles[reference_point, 1:-1, loc])
	# 	print(max_ratio)
	# 	for iR in range(np.shape(inverted_profiles)[1] - 1):
	# 		# plt.plot(new_timesteps, inverted_profiles[:, iR, loc] / inverted_profiles[:, iR + 1, loc], color[iR]+'+',label='line ratio ' + str(iR + 4) + '/' + str(iR + 4 + 1))
	# 		plt.plot(new_timesteps, inverted_profiles[:, iR, loc] / inverted_profiles[:, iR + 1, loc], color=color[iR],
	# 				 marker='+', linewidth=0.4, label='line ratio ' + str(iR + 4) + '/' + str(iR + 4 + 1))
	# 	plt.title('Line ratio at radious' + str(np.around(r[loc], decimals=3)) + 'mm')
	# 	# plt.semilogy()
	# 	plt.xlabel('time [ms]')
	# 	plt.ylabel('relative intensity [au]')
	# 	plt.legend(loc='best')
	# 	plt.ylim(0, np.max([max_ratio, 4]))
	# 	plt.savefig(
	# 		'/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(
	# 			merge_ID_target) + '/line_ratio_intens_r_' + str(
	# 			np.around(r[loc], decimals=3)) + '.eps', bbox_inches='tight')
	# 	plt.close()

	for index, loc in enumerate(sample_radious):
		plt.figure(figsize=(20, 10))
		reference_point = inverted_profiles[:, 0, loc].argmax()
		max_ratio = max(inverted_profiles[reference_point, :-2, loc] / inverted_profiles[reference_point, 1:-1, loc])
		print(max_ratio)
		for iR in range(np.shape(inverted_profiles)[1] - 1):
			# plt.plot(new_timesteps, inverted_profiles[:, iR, loc] / inverted_profiles[:, iR + 1, loc], color[iR]+'+',label='line ratio ' + str(iR + 4) + '/' + str(iR + 4 + 1))
			plt.plot(new_timesteps, inverted_profiles[:, iR + 1, loc] / inverted_profiles[:, iR, loc], color=color[iR],
					 marker='+', linewidth=0.4, label='line ratio ' + str(iR + 4 + 1) + '/' + str(iR + 4))
		plt.title('Line ratio at radious' + str(np.around(r[loc], decimals=3)) + 'mm')
		# plt.semilogy()
		plt.xlabel('time [ms]')
		plt.ylabel('relative intensity [au]')
		plt.legend(loc='best')
		plt.ylim(bottom=0)
		plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(
			merge_ID_target) + '/line_ratio_intens_r_' + str(np.around(r[loc], decimals=3)) + '.eps',
					bbox_inches='tight')
		plt.close()



	for iR in range(np.shape(inverted_profiles)[1] - 1):
		plt.figure(figsize=(20, 10))
		# plt.plot(new_timesteps, inverted_profiles[:, iR, loc] / inverted_profiles[:, iR + 1, loc], color[iR]+'+',label='line ratio ' + str(iR + 4) + '/' + str(iR + 4 + 1))
		plt.imshow(inverted_profiles[:, iR + 1] / inverted_profiles[:, iR], 'rainbow',vmin=0,vmax=1,extent=[min(xx),max(xx),np.max(new_timesteps),np.min(new_timesteps)],aspect=0.02)
		plt.title('Line ratio ' +str(iR + 4 + 1) +'/' + str(iR + 4))
		plt.xlabel('radial location [m]')
		plt.ylabel('time [ms]')
		plt.colorbar()
		plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(	merge_ID_target) + '/inverted_line_ratio_map_' +str(iR + 4 + 1) +'-' + str(iR + 4) + '.eps',bbox_inches='tight')
		plt.close()

	for iR in range(np.shape(inverted_profiles)[1] - 1):
		plt.figure(figsize=(20, 10))
		# plt.plot(new_timesteps, inverted_profiles[:, iR, loc] / inverted_profiles[:, iR + 1, loc], color[iR]+'+',label='line ratio ' + str(iR + 4) + '/' + str(iR + 4 + 1))
		plt.imshow(all_fits[:,:, iR + 1] / all_fits[:,:, iR], 'rainbow',vmin=0,vmax=1,extent=[min(xx),max(xx),np.max(new_timesteps),np.min(new_timesteps)],aspect=0.02)
		plt.title('Line ratio ' +str(iR + 4 + 1) +'/' + str(iR + 4))
		plt.xlabel('radial location [m]')
		plt.ylabel('time [ms]')
		plt.colorbar()
		plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(	merge_ID_target) + '/line_ratio_map_' +str(iR + 4 + 1) +'-' + str(iR + 4) + '.eps',bbox_inches='tight')
		plt.close()



	SS_inverted_profiles = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/SS_inverted_profiles.npy')
	time=int((np.abs(new_timesteps+0.3)).argmin())
	plt.figure(figsize=(20, 10))
	for iR in range(np.shape(inverted_profiles)[1]):
		plt.plot(r, SS_inverted_profiles[iR]/np.max(SS_inverted_profiles[iR]), color[iR],label='line ' + str(iR + 4))
	plt.title('Line profiles in steady state')
	plt.xlabel('radial location [m]')
	plt.ylabel('Intensity [au]')
	plt.legend(loc='best')
	plt.ylim(bottom=0)
	plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/inverted_line_intensity_at_time_' + str(new_timesteps[time]) +'ms' + '.eps',bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)

	time=int((np.abs(new_timesteps-0.6)).argmin())
	plt.figure(figsize=(20, 10))
	for iR in range(np.shape(inverted_profiles)[1]):
		plt.plot(r, inverted_profiles[time, iR]/np.max(inverted_profiles[time, iR]), color[iR],label='line ' + str(iR + 4))
	plt.title('Line profiles at time ' +str(new_timesteps[time]) +'ms')
	plt.xlabel('radial location [m]')
	plt.ylabel('Intensity [au]')
	plt.legend(loc='best')
	plt.ylim(bottom=0)
	plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/inverted_line_intensity_at_time_' + str(new_timesteps[time]) +'ms' + '.eps',bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)


	time=int((np.abs(new_timesteps-6)).argmin())
	plt.figure(figsize=(20, 10))
	for iR in range(np.shape(inverted_profiles)[1]):
		plt.plot(r, inverted_profiles[time, iR]/np.max(inverted_profiles[time, iR]), color[iR],label='line ' + str(iR + 4))
	plt.title('Line profiles at time ' +str(new_timesteps[time]) +'ms')
	plt.xlabel('radial location [m]')
	plt.ylabel('Intensity [au]')
	plt.legend(loc='best')
	plt.ylim(bottom=0)
	plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/inverted_line_intensity_at_time_' + str(new_timesteps[time]) +'ms' + '.eps',bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)


	path = '/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)
	time_start = int((np.abs(new_timesteps + 0.5)).argmin())
	time_end = int((np.abs(new_timesteps -1.5)).argmin())
	for iR in range(len(inverted_profiles[0])):
		plt.figure();
		plt.imshow(inverted_profiles[time_start:time_end,iR],'rainbow',vmin=0,extent=[min(r),max(r),new_timesteps[time_end],new_timesteps[time_start]],aspect=0.05)
		plt.title('Line '+str(iR+4))
		plt.xlabel('radius [m]')
		plt.ylabel('time [ms]')
		plt.colorbar()
		plt.savefig('%s/abel_inverted_profile%s.eps' % (path, iR), bbox_inches='tight')
		plt.close()



	time_step = int((np.abs(new_timesteps+1)).argmin())
	all_fits_ss = all_fits[:time_step]
	all_fits_ss[all_fits_ss==0]=np.nan
	all_fits_ss = np.nanmean(all_fits_ss,axis=0)
	doLateralfit_single(df_settings, all_fits_ss, merge_ID_target,dx,xx,r)













exit()
# extra stuff about finding what is the best profile in the LOS direction that describes observations


geom_null = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
geom_null.loc[0] = [0,0,0,0,0]

j=218
(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
type = '.tif'
filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)
type = '.txt'
filename_metadata = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)[0]
index=0
dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
data = []
for fname in filenames:
	fname = fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0/'+filenames[index]
	im = Image.open(fname)
	data.append(np.array(im)-dataDark)
data = np.mean(data,axis=0)

angle = 0.773281098404
tilt_4_points = np.array([[0., 1042.85026968],
						  [1608., 1051.36500268],
						  [0., 61.22662907],
						  [1608., 16.9049927]])
tilt = [2.45894737e+01, 6.54521132e-04, 2.35137447e+01]
waveLcoefs = np.ones((2, 3)) * np.nan
waveLcoefs[1] = [2.49416833e-06, 1.20808818e-01, 3.18155463e+02]


data = rotate(data, angle)
data = do_tilt_no_lines(data)
first_bin, bin_interv = get_bin_and_interv_no_lines(data)

geom = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
geom.loc[0] = [angle, tilt_4_points, bin_interv, first_bin, first_bin]
geom_store = copy.deepcopy(geom)


bottom_limit = np.ceil(geom['bin00a'][0]+range(40)*geom['binInterv'][0]).astype(int)
top_limit = np.floor(geom['bin00a'][0]+range(1,41)*geom['binInterv'][0]).astype(int)

gauss_bias2 = lambda x, A, sig, x0, m, q,exponent: A * np.exp(-np.power(np.abs((x - x0) / sig) , exponent)) + q + x * m

fits = np.zeros((40,np.shape(data)[-1]-200,6))
i=10
j=300
bin_sel = data[bottom_limit[i]:top_limit[i]+1]
fit = curve_fit(gauss_bias2, np.linspace(1, len(bin_sel), len(bin_sel)), bin_sel[:, j],
				[np.max(bin_sel[:, j]), 20, len(bin_sel) / 2, 0, 0, 2.7], maxfev=100000000,
				bounds=[[np.min(bin_sel[:, j]), 5, 0, -0.01, -np.inf, 1],
						[np.inf, np.inf, len(bin_sel), 0.01, 0.01, 5]])

for i in range(40):
	bin_sel = data[bottom_limit[i]:top_limit[i]+1]
	for indexj,j in enumerate(range(100,np.shape(bin_sel)[-1]-100)):
		fit = curve_fit(gauss_bias2, np.linspace(1,len(bin_sel),len(bin_sel)), bin_sel[:,j], [np.max(bin_sel[:,j]),20,len(bin_sel)/2,0,0,2.7], maxfev=100000000,bounds=[[np.min(bin_sel[:,j]),5,0,-0.01,-np.inf,1],[np.inf,np.inf,len(bin_sel),0.01,0.01,5]])
		fits[i,indexj]=fit[0]


# Nope, I cannot fit a profile to resample the data, because for low signal I don't really have a profile (wavelength axis<400) but becomes flat