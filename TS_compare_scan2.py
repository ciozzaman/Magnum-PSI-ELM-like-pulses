import numpy as np
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())
# import matplotlib.pyplot as plt
#import .functions
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
from functions.spectools import rotate,do_tilt, binData, get_angle, get_tilt,get_angle_2
from functions.Calibrate import do_waveL_Calib, do_Intensity_Calib
from functions.fabio_add import find_nearest_index,multi_gaussian,all_file_names,load_dark,find_index_of_file,get_metadata,movie_from_data,get_angle_no_lines,do_tilt_no_lines,four_point_transform,fix_minimum_signal,fix_minimum_signal2,get_bin_and_interv_no_lines,examine_current_trace
from functions.GetSpectrumGeometry import getGeom
from functions.SpectralFit import doSpecFit_single_frame
from functions.GaussFitData import doLateralfit_time_tependent
import collections

import os,sys
from PIL import Image
import xarray as xr
import pandas as pd
import copy
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.signal import find_peaks

os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data/functions/ADAS_opacity')
from adas import read_adf15
from idlbridge import export_procedure

fdir = '/home/ffederic/work/Collaboratory/test/experimental_data'
df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
df_settings = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_3.csv',index_col=0)

TS_size=[-4.149230769230769056e+01,4.416923076923076508e+01]
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro','paleturquoise']
line_style = ['-','--',':','-.']
marker = ['x','+']
boltzmann_constant_J = 1.380649e-23	# J/K
eV_to_K = 8.617333262145e-5	# eV/K

compare_peak_Te = []
compare_peak_dTe = []
compare_peak_ne = []
compare_peak_dne = []
distance_all=[]
pressure_all=[]

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel('Pressure [Pa]')
ax1.set_ylabel('max Te [eV]', color='tab:red')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('max ne [10^20 #/m3]', color='tab:blue')  # we already handled the x-label with ax1
ax3.set_ylabel('max average static pressure [Pa]', color='tab:green')  # we already handled the x-label with ax1
ax3.spines["right"].set_position(("axes", 1.1))
for i_to_scan,to_scan in enumerate([[99,98,96,97],[95,89,87,86,85]]):
	# to_scan = [99,98,96,97]
	# to_scan = [95,89,87,86,85]
	# to_scan = [91,90,85,92,93,94]
	# to_scan = [70,69,68,67,66]
	# for merge_ID_target in [91,90,85,92,93,94]:
	# for merge_ID_target in [95,89,88,87,86,85]:

	merge_Te_prof_multipulse_interp_crop_all = []
	merge_Te_SS_all = []
	merge_dTe_prof_multipulse_interp_crop_all = []
	merge_ne_prof_multipulse_interp_crop_all = []
	merge_ne_SS_all = []
	merge_dne_prof_multipulse_interp_crop_all = []
	target_chamber_pressure_all = []
	target_chamber_pressure_SS_all = []
	target_OES_distance_all = []
	feed_rate_SLM_all = []
	capacitor_voltage_all = []
	magnetic_field_all = []
	average_static_pressure_all = []

	for merge_ID_target in to_scan:
		merge_time_window = [-1,2]

		for i in range(2):
			print('.')
		print('Starting to work on merge number ' + str(merge_ID_target))
		for i in range(2):
			print('.')

		all_j=find_index_of_file(merge_ID_target,df_settings,df_log,only_OES=True)
		target_chamber_pressure = []
		target_OES_distance = []
		feed_rate_SLM = []
		capacitor_voltage = []
		magnetic_field = []
		for j in all_j:
			target_chamber_pressure.append(df_log.loc[j,['p_n [Pa]']])
			target_OES_distance.append(df_log.loc[j,['T_axial']])
			feed_rate_SLM.append(df_log.loc[j,['Seed']])
			capacitor_voltage.append(df_log.loc[j,['Vc']])
			magnetic_field.append(df_log.loc[j,['B']])
		target_chamber_pressure = np.nanmean(target_chamber_pressure)
		target_OES_distance = np.nanmean(target_OES_distance)
		feed_rate_SLM = np.nanmean(feed_rate_SLM)
		capacitor_voltage = np.nanmean(capacitor_voltage)
		magnetic_field = np.nanmean(magnetic_field)
		target_chamber_pressure_all.append(target_chamber_pressure)
		target_OES_distance_all.append(target_OES_distance)
		feed_rate_SLM_all.append(feed_rate_SLM)
		capacitor_voltage_all.append(capacitor_voltage)
		magnetic_field_all.append(magnetic_field)

		path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) #+ '_back'
		inverted_profiles_original = 4 * np.pi * np.load(path_where_to_save_everything + '/inverted_profiles.npy')  # in W m^-3 sr^-1  *  4*pi sr  =  W m^-3
		merge_Te_prof_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_Te_prof_multipulse']
		merge_dTe_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_dTe_multipulse']
		merge_ne_prof_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_ne_prof_multipulse']
		merge_dne_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_dne_multipulse']
		merge_time_original = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_time']
		# new_timesteps = np.load(path_where_to_save_everything + '/merge' + str(merge_ID_target) + '_new_timesteps.npy')
		new_timesteps = np.linspace(-0.5,1.5,num=41)
		dt = np.nanmedian(np.diff(new_timesteps))
		time_shift_factor=0
		spatial_factor=1


		# dx = 18 / 40 * (50.5 / 27.4) / 1e3
		dx = 1.06478505470992 / 1e3	# 10/02/2020 from	Calculate magnification_FF.xlsx
		xx = np.arange(40) * dx  # m
		xn = np.linspace(0, max(xx), 1000)
		# r = np.linspace(-max(xx) / 2, max(xx) / 2, 1000)
		# r = r[::10]
		# dr = np.median(np.diff(r))
		number_of_radial_divisions = np.shape(inverted_profiles_original)[-1]
		r = np.arange(number_of_radial_divisions)*dx
		dr = np.median(np.diff(r))

		merge_time = time_shift_factor + merge_time_original
		TS_dt = np.nanmedian(np.diff(merge_time))

		TS_size = [-4.149230769230769056e+01, 4.416923076923076508e+01]
		TS_r = TS_size[0] + np.linspace(0, 1, 65) * (TS_size[1] - TS_size[0])
		TS_dr = np.median(np.diff(TS_r)) / 1000
		gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
		profile_centres = []
		profile_sigma = []
		profile_centres_score = []
		for index in range(np.shape(merge_ne_prof_multipulse)[0]):
			yy = merge_ne_prof_multipulse[index]
			yy_sigma = merge_dne_multipulse[index]
			yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
			if np.sum(yy>0)<5:
				profile_centres.append(0)
				profile_sigma.append(10)
				profile_centres_score.append(np.max(TS_r))
				continue
			yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
			p0 = [np.max(yy), 10, 0]
			bds = [[0, 0, np.min(TS_r)], [np.inf, TS_size[1], np.max(TS_r)]]
			fit = curve_fit(gauss, TS_r, yy, p0, sigma=yy_sigma, maxfev=100000, bounds=bds)
			profile_centres.append(fit[0][-1])
			profile_sigma.append(fit[0][-2])
			profile_centres_score.append(fit[1][-1, -1])
		# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)
		profile_centres = np.array(profile_centres)
		profile_sigma = np.array(profile_sigma)
		profile_centres_score = np.array(profile_centres_score)
		# centre = np.nanmean(profile_centres[profile_centres_score < 1])
		centre = np.nansum(profile_centres/(profile_centres_score**1))/np.sum(1/profile_centres_score**1)
		TS_r_new = (TS_r - centre) / 1000
		print('TS profile centre at %.3gmm compared to the theoretical centre' %centre)
		# temp_r, temp_t = np.meshgrid(TS_r_new, merge_time)
		# plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse,vmin=0);plt.colorbar().set_label('Te [eV]');plt.pause(0.01)
		# plt.figure();plt.pcolor(temp_t,temp_r,merge_ne_prof_multipulse,vmin=0);plt.colorbar().set_label('ne [10^20 #/m^3]');plt.pause(0.01)

		if os.path.exists(path_where_to_save_everything + '/TS_SS_data_merge_' + str(merge_ID_target) + '.npz'):
			merge_Te_prof_multipulse_SS = np.load(path_where_to_save_everything + '/TS_SS_data_merge_' + str(merge_ID_target) + '.npz')['merge_Te_prof_multipulse']
			merge_dTe_multipulse_SS = np.load(path_where_to_save_everything + '/TS_SS_data_merge_' + str(merge_ID_target) + '.npz')['merge_dTe_multipulse']
			merge_ne_prof_multipulse_SS = np.load(path_where_to_save_everything + '/TS_SS_data_merge_' + str(merge_ID_target) + '.npz')['merge_ne_prof_multipulse']
			merge_dne_multipulse_SS = np.load(path_where_to_save_everything + '/TS_SS_data_merge_' + str(merge_ID_target) + '.npz')['merge_dne_multipulse']

			merge_Te_SS_all.append(merge_Te_prof_multipulse_SS)
			merge_ne_SS_all.append(merge_ne_prof_multipulse_SS)
			target_chamber_pressure_SS_all.append(target_chamber_pressure)

			if False:	# I don't think I need this part, I use the SS info to replace directly the original time dependent data
				gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
				yy = merge_ne_prof_multipulse_SS
				yy_sigma = merge_dne_multipulse_SS
				if np.sum(np.isfinite(yy_sigma))>0:
					yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
					yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
				else:
					yy_sigma = np.ones_like(yy)*np.nanmin([np.nanmax(yy),1])
				p0 = [np.max(yy), 10, 0]
				bds = [[0, 0, np.min(TS_r)], [np.inf, TS_size[1], np.max(TS_r)]]
				fit = curve_fit(gauss, TS_r, yy, p0, sigma=yy_sigma, maxfev=100000, bounds=bds)
				SS_profile_centres=[fit[0][-1]]
				SS_profile_sigma=[fit[0][-2]]
				SS_profile_centres_score=[fit[1][-1, -1]]
				# plt.figure();plt.plot(TS_r,merge_ne_prof_multipulse,label='ne')
				# plt.plot([fit[0][-1],fit[0][-1]],[np.max(merge_ne_prof_multipulse),np.min(merge_ne_prof_multipulse)],'--',label='ne')
				yy = merge_Te_prof_multipulse_SS
				yy_sigma = merge_dTe_multipulse_SS
				if np.sum(np.isfinite(yy_sigma))>0:
					yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
					yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
				else:
					yy_sigma = np.ones_like(yy)*np.nanmin([np.nanmax(yy),1])
				p0 = [np.max(yy), 10, 0]
				bds = [[0, 0, np.min(TS_r)], [np.inf, TS_size[1], np.max(TS_r)]]
				fit = curve_fit(gauss, TS_r, yy, p0, sigma=yy_sigma, maxfev=100000, bounds=bds)
				SS_profile_centres = np.append(SS_profile_centres,[fit[0][-1]],axis=0)
				SS_profile_sigma = np.append(SS_profile_sigma,[fit[0][-2]],axis=0)
				SS_profile_centres_score = np.append(SS_profile_centres_score,[fit[1][-1, -1]],axis=0)
				SS_centre = np.nanmean(SS_profile_centres)
				SS_TS_r_new = (TS_r - SS_centre) / 1000
				print('TS profile SS_centre at %.3gmm compared to the theoretical centre' %centre)

				# This is the mean of Te and ne weighted in their own uncertainties.
				interp_range_r = max(dx, TS_dr) * 1.5
				# weights_r = TS_r_new/interp_range_r
				weights_r = SS_TS_r_new
				merge_Te_prof_multipulse_SS_interp = np.zeros_like(merge_Te_prof_multipulse_interp[ 0])
				merge_dTe_prof_multipulse_SS_interp = np.zeros_like(merge_Te_prof_multipulse_interp[ 0])
				merge_ne_prof_multipulse_SS_interp = np.zeros_like(merge_Te_prof_multipulse_interp[ 0])
				merge_dne_prof_multipulse_SS_interp = np.zeros_like(merge_Te_prof_multipulse_interp[ 0])
				for i_r, value_r in enumerate(np.abs(r)):
					if np.sum(np.abs(np.abs(SS_TS_r_new) - value_r) < interp_range_r) == 0:
						continue
					selected_values = np.abs(np.abs(SS_TS_r_new) - value_r) < interp_range_r
					selected_values[merge_Te_prof_multipulse_SS == 0] = False
					# weights = 1/np.abs(weights_r[selected_values]+1e-5)
					weights = 1/((weights_r[selected_values]-value_r)/interp_range_r)**2
					# weights = np.ones((np.sum(selected_values)))
					if np.sum(selected_values) == 0:
						continue
					merge_Te_prof_multipulse_SS_interp[i_r] = np.sum(merge_Te_prof_multipulse_SS[selected_values]*weights / merge_dTe_multipulse_SS[selected_values]) / np.sum(weights / merge_dTe_multipulse_SS[selected_values])
					merge_ne_prof_multipulse_SS_interp[i_r] = np.sum(merge_ne_prof_multipulse_SS[selected_values]*weights / merge_dne_multipulse_SS[selected_values]) / np.sum(weights / merge_dne_multipulse_SS[selected_values])
					if False: 	# suggestion from Daljeet: use the "worst case scenario" in therms of uncertainty
						merge_dTe_prof_multipulse_SS_interp[i_r] = 1/(np.sum(1 / merge_dTe_multipulse_SS[selected_values]))*(np.sum( ((merge_Te_prof_multipulse_SS_interp[i_r]-merge_Te_prof_multipulse_SS[selected_values])/merge_dTe_multipulse_SS[selected_values])**2 )**0.5)
						merge_dne_prof_multipulse_SS_interp[i_r] = 1/(np.sum(1 / merge_dne_multipulse_SS[selected_values]))*(np.sum( ((merge_ne_prof_multipulse_SS_interp[i_r]-merge_ne_prof_multipulse_SS[selected_values])/merge_dne_multipulse_SS[selected_values])**2 )**0.5)
					else:
						merge_dTe_prof_multipulse_SS_interp_temp = 1/(np.sum(1 / merge_dTe_multipulse_SS[selected_values]))*(np.sum( ((merge_Te_prof_multipulse_SS_interp[i_r]-merge_Te_prof_multipulse_SS[selected_values])/merge_dTe_multipulse_SS[selected_values])**2 )**0.5)
						merge_dne_prof_multipulse_SS_interp_temp = 1/(np.sum(1 / merge_dne_multipulse_SS[selected_values]))*(np.sum( ((merge_ne_prof_multipulse_SS_interp[i_r]-merge_ne_prof_multipulse_SS[selected_values])/merge_dne_multipulse_SS[selected_values])**2 )**0.5)
						merge_dTe_prof_multipulse_SS_interp[i_r] = max(merge_dTe_prof_multipulse_SS_interp_temp,(np.max(merge_Te_prof_multipulse_SS[selected_values])-np.min(merge_Te_prof_multipulse_SS[selected_values]))/2/2 )	# I enlarged the integration range by 1.5, so I reduce the sigma in the second calculation mechanism to compensate for that and get reasonables uncertainties
						merge_dne_prof_multipulse_SS_interp[i_r] = max(merge_dne_prof_multipulse_SS_interp_temp,(np.max(merge_ne_prof_multipulse_SS[selected_values])-np.min(merge_ne_prof_multipulse_SS[selected_values]))/2/2 )	# I enlarged the integration range by 1.5, so I reduce the sigma in the second calculation mechanism to compensate for that and get reasonables uncertainties
				temp_r, temp_t = np.meshgrid(r, new_timesteps)

				start_r = np.abs(r - 0).argmin()
				end_r = np.abs(r - 5).argmin() + 1
				r_crop = r[start_r:end_r]
				merge_Te_prof_multipulse_SS_interp_crop = merge_Te_prof_multipulse_SS_interp[start_r:end_r]
				merge_dTe_prof_multipulse_SS_interp_crop = merge_dTe_prof_multipulse_SS_interp[start_r:end_r]
				merge_ne_prof_multipulse_SS_interp_crop = merge_ne_prof_multipulse_SS_interp[start_r:end_r]
				merge_dne_prof_multipulse_SS_interp_crop = merge_dne_prof_multipulse_SS_interp[start_r:end_r]

				gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
				yy = merge_ne_prof_multipulse_SS_interp_crop
				yy_sigma = merge_dne_prof_multipulse_SS_interp_crop
				yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
				yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
				p0 = [np.max(yy), np.max(r_crop)/2, np.min(r_crop)]
				bds = [[0, 0, np.min(r_crop)], [np.inf, np.max(r_crop), np.max(r_crop)]]
				fit = curve_fit(gauss, r_crop, yy, p0, maxfev=100000, bounds=bds, sigma=yy_sigma,absolute_sigma=True)
				SS_averaged_profile_sigma=fit[0][-2]
				# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)

				for i_t in range(len(time_crop)):	# the 2e20 limit in density comes from VanDerMeiden2012a
					merge_Te_prof_multipulse_interp_crop[i_t,merge_ne_prof_multipulse_interp_crop[i_t]<2] = np.max([merge_Te_prof_multipulse_SS_interp_crop[merge_ne_prof_multipulse_interp_crop[i_t]<2],merge_Te_prof_multipulse_interp_crop[i_t,merge_ne_prof_multipulse_interp_crop[i_t]<2]],axis=0)
					merge_dTe_prof_multipulse_interp_crop[i_t,merge_ne_prof_multipulse_interp_crop[i_t]<2] = np.max([merge_Te_prof_multipulse_SS_interp_crop[merge_ne_prof_multipulse_interp_crop[i_t]<2],merge_dTe_prof_multipulse_SS_interp_crop[merge_ne_prof_multipulse_interp_crop[i_t]<2]],axis=0)
					merge_ne_prof_multipulse_interp_crop[i_t,merge_ne_prof_multipulse_interp_crop[i_t]<2] = np.max([merge_ne_prof_multipulse_SS_interp_crop[merge_ne_prof_multipulse_interp_crop[i_t]<2],merge_ne_prof_multipulse_interp_crop[i_t,merge_ne_prof_multipulse_interp_crop[i_t]<2]],axis=0)
					merge_dne_prof_multipulse_interp_crop[i_t,merge_ne_prof_multipulse_interp_crop[i_t]<2] = np.max([merge_ne_prof_multipulse_SS_interp_crop[merge_ne_prof_multipulse_interp_crop[i_t]<2],merge_dne_prof_multipulse_SS_interp_crop[merge_ne_prof_multipulse_interp_crop[i_t]<2]],axis=0)

			else:
				temp = np.mean(merge_ne_prof_multipulse,axis=1)
				start = (temp>np.mean(merge_ne_prof_multipulse_SS)).argmax()
				end = (np.flip(temp,axis=0)>np.mean(merge_ne_prof_multipulse_SS)).argmax()
				merge_Te_prof_multipulse[:start] = merge_Te_prof_multipulse_SS
				merge_dTe_multipulse[:start] = 2*merge_Te_prof_multipulse_SS
				merge_ne_prof_multipulse[:start] = merge_ne_prof_multipulse_SS
				merge_dne_multipulse[:start] = merge_ne_prof_multipulse_SS
				merge_Te_prof_multipulse[-end:] = merge_Te_prof_multipulse_SS
				merge_dTe_multipulse[-end:] = 2*merge_Te_prof_multipulse_SS
				merge_ne_prof_multipulse[-end:] = merge_ne_prof_multipulse_SS
				merge_dne_multipulse[-end:] = merge_ne_prof_multipulse_SS


		# This is the mean of Te and ne weighted in their own uncertainties.
		temp1 = np.zeros_like(inverted_profiles_original[:, 0])
		temp2 = np.zeros_like(inverted_profiles_original[:, 0])
		temp3 = np.zeros_like(inverted_profiles_original[:, 0])
		temp4 = np.zeros_like(inverted_profiles_original[:, 0])
		interp_range_t = max(dt, TS_dt) * 1.5
		interp_range_r = max(dx, TS_dr) * 1.5
		weights_r = (np.zeros_like(merge_Te_prof_multipulse) + TS_r_new)
		weights_t = (((np.zeros_like(merge_Te_prof_multipulse)).T + merge_time).T)
		for i_t, value_t in enumerate(new_timesteps):
			if np.sum(np.abs(merge_time - value_t) < interp_range_t) == 0:
				continue
			for i_r, value_r in enumerate(np.abs(r)):
				if np.sum(np.abs(np.abs(TS_r_new) - value_r) < interp_range_r) == 0:
					continue
				elif np.sum(np.logical_and(np.abs(merge_time - value_t) < interp_range_t,np.sum(merge_Te_prof_multipulse, axis=1) > 0)) == 0:
					continue
				selected_values_t = np.logical_and(np.abs(merge_time - value_t) < interp_range_t,np.sum(merge_Te_prof_multipulse, axis=1) > 0)
				selected_values_r = np.abs(np.abs(TS_r_new) - value_r) < interp_range_r
				selected_values = (np.array([selected_values_t])).T * selected_values_r
				selected_values[merge_Te_prof_multipulse == 0] = False
				# weights = 1/(weights_r[selected_values]-value_r)**2 + 1/(weights_t[selected_values]-value_t)**2
				weights = 1/((weights_t[selected_values]-value_t)/interp_range_t)**2 + 1/((weights_r[selected_values]-value_r)/interp_range_r)**2
				if np.sum(selected_values) == 0:
					continue
				# temp1[i_t,i_r] = np.mean(merge_Te_prof_multipulse[selected_values_t][:,selected_values_r])
				# temp2[i_t, i_r] = np.max(merge_dTe_multipulse[selected_values_t][:,selected_values_r]) / (np.sum(np.isfinite(merge_dTe_multipulse[selected_values_t][:,selected_values_r])) ** 0.5)
				temp1[i_t, i_r] = np.sum(merge_Te_prof_multipulse[selected_values]*weights / merge_dTe_multipulse[selected_values]) / np.sum(weights / merge_dTe_multipulse[selected_values])
				# temp3[i_t,i_r] = np.mean(merge_ne_prof_multipulse[selected_values_t][:,selected_values_r])
				# temp4[i_t, i_r] = np.max(merge_dne_multipulse[selected_values_t][:,selected_values_r]) / (np.sum(np.isfinite(merge_dne_multipulse[selected_values_t][:,selected_values_r])) ** 0.5)
				temp3[i_t, i_r] = np.sum(merge_ne_prof_multipulse[selected_values]*weights / merge_dne_multipulse[selected_values]) / np.sum(weights / merge_dne_multipulse[selected_values])
				if False: 	# suggestion from Daljeet: use the "worst case scenario" in therms of uncertainty
					# temp2[i_t, i_r] = (np.sum(selected_values) / (np.sum(1 / merge_dTe_multipulse[selected_values]) ** 2)) ** 0.5
					# temp4[i_t, i_r] = (np.sum(selected_values) / (np.sum(1 / merge_dne_multipulse[selected_values]) ** 2)) ** 0.5
					temp2[i_t, i_r] = 1/(np.sum(1 / merge_dTe_multipulse[selected_values]))*(np.sum( ((temp1[i_t, i_r]-merge_Te_prof_multipulse[selected_values])/merge_dTe_multipulse[selected_values])**2 )**0.5)
					temp4[i_t, i_r] = 1/(np.sum(1 / merge_dne_multipulse[selected_values]))*(np.sum( ((temp3[i_t, i_r]-merge_ne_prof_multipulse[selected_values])/merge_dne_multipulse[selected_values])**2 )**0.5)
				else:
					# temp2_temp = 1/(np.sum(1 / merge_dTe_multipulse[selected_values]))*(np.sum( ((temp1[i_t, i_r]-merge_Te_prof_multipulse[selected_values])/merge_dTe_multipulse[selected_values])**2 )**0.5)
					# temp4_temp = 1/(np.sum(1 / merge_dne_multipulse[selected_values]))*(np.sum( ((temp3[i_t, i_r]-merge_ne_prof_multipulse[selected_values])/merge_dne_multipulse[selected_values])**2 )**0.5)
					temp2[i_t, i_r] = max(np.sqrt(np.sum(weights**2))/np.sum(weights / merge_dTe_multipulse[selected_values]),(np.max(merge_Te_prof_multipulse[selected_values])-np.min(merge_Te_prof_multipulse[selected_values]))/2/2 )	# I enlarged the integration range by 1.5, so I reduce the sigma in the second calculation mechanism to compensate for that and get reasonables uncertainties
					temp4[i_t, i_r] = max(np.sqrt(np.sum(weights**2))/np.sum(weights / merge_dne_multipulse[selected_values]),(np.max(merge_ne_prof_multipulse[selected_values])-np.min(merge_ne_prof_multipulse[selected_values]))/2/2 )	# I enlarged the integration range by 1.5, so I reduce the sigma in the second calculation mechanism to compensate for that and get reasonables uncertainties

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
		merge_Te_prof_multipulse_interp_crop[merge_Te_prof_multipulse_interp_crop<0]=0
		merge_dTe_prof_multipulse_interp_crop = merge_dTe_prof_multipulse_interp[start_time:end_time, start_r:end_r]
		merge_ne_prof_multipulse_interp_crop = merge_ne_prof_multipulse_interp[start_time:end_time,start_r:end_r]
		merge_ne_prof_multipulse_interp_crop[merge_ne_prof_multipulse_interp_crop<0]=0
		merge_dne_prof_multipulse_interp_crop = merge_dne_prof_multipulse_interp[start_time:end_time, start_r:end_r]

		merge_Te_prof_multipulse_interp_crop_all.append(merge_Te_prof_multipulse_interp_crop)
		merge_dTe_prof_multipulse_interp_crop_all.append(merge_dTe_prof_multipulse_interp_crop)
		merge_ne_prof_multipulse_interp_crop_all.append(merge_ne_prof_multipulse_interp_crop)
		merge_dne_prof_multipulse_interp_crop_all.append(merge_dne_prof_multipulse_interp_crop)

		area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
		average_static_pressure = merge_ne_prof_multipulse_interp_crop*1e20*( (1*merge_Te_prof_multipulse_interp_crop/eV_to_K*boltzmann_constant_J + merge_Te_prof_multipulse_interp_crop/eV_to_K*boltzmann_constant_J))
		average_static_pressure = np.sum(average_static_pressure*area,axis=-1)/sum(area)
		average_static_pressure_all.append(average_static_pressure)

	merge_Te_prof_multipulse_interp_crop_all = np.array(merge_Te_prof_multipulse_interp_crop_all)
	merge_dTe_prof_multipulse_interp_crop_all = np.array(merge_dTe_prof_multipulse_interp_crop_all)
	merge_dTe_prof_multipulse_interp_crop_all[merge_dTe_prof_multipulse_interp_crop_all==0] = np.max(merge_dTe_prof_multipulse_interp_crop_all[merge_dTe_prof_multipulse_interp_crop_all>0])
	merge_ne_prof_multipulse_interp_crop_all = np.array(merge_ne_prof_multipulse_interp_crop_all)
	merge_dne_prof_multipulse_interp_crop_all = np.array(merge_dne_prof_multipulse_interp_crop_all)
	merge_dne_prof_multipulse_interp_crop_all[merge_dne_prof_multipulse_interp_crop_all==0] = np.max(merge_dne_prof_multipulse_interp_crop_all[merge_dne_prof_multipulse_interp_crop_all>0])
	target_chamber_pressure_all = np.array(target_chamber_pressure_all)
	target_OES_distance_all = np.array(target_OES_distance_all)
	feed_rate_SLM_all = np.array(feed_rate_SLM_all)
	capacitor_voltage_all = np.array(capacitor_voltage_all)
	magnetic_field_all = np.array(magnetic_field_all)


	ax1.plot(target_chamber_pressure_all,np.max(merge_Te_prof_multipulse_interp_crop_all,axis=(1,2)),ls=line_style[i_to_scan],color='r',label='Te B=%.3gT' %(np.mean(magnetic_field_all)))
	ax2.plot(target_chamber_pressure_all,np.max(merge_ne_prof_multipulse_interp_crop_all,axis=(1,2)),ls=line_style[i_to_scan],color='b',label='ne B=%.3gT' %(np.mean(magnetic_field_all)))
	ax3.plot(target_chamber_pressure_all,np.max(average_static_pressure_all,axis=(1)),ls=line_style[i_to_scan],color='g',label='pressure B=%.3gT' %(np.mean(magnetic_field_all)))
	# ax1.plot(target_chamber_pressure_SS_all,np.max(merge_Te_SS_all,axis=(1)),ls=line_style[i_to_scan],color='r',label='SS Te B=%.3gT' %(np.mean(magnetic_field_all)))
	# ax2.plot(target_chamber_pressure_SS_all,np.max(merge_ne_SS_all,axis=(1)),ls=line_style[i_to_scan],color='c',label='SS ne B=%.3gT' %(np.mean(magnetic_field_all)))
ax1.tick_params(axis='y', labelcolor='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax3.tick_params(axis='y', labelcolor='tab:green')
ax1.legend(loc=1, fontsize='x-small')
ax2.legend(loc=2, fontsize='x-small')
ax3.legend(loc=4, fontsize='x-small')
ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)
ax3.set_ylim(bottom=0)
ax1.grid()
# ax2.grid()
plt.pause(0.01)
