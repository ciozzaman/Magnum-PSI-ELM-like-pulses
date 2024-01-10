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
results_summary = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/results_summary.csv',index_col=0)

TS_size=[-4.149230769230769056e+01,4.416923076923076508e+01]
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro','paleturquoise']
line_style = ['-','--',':','-.']
marker = ['x','+']
boltzmann_constant_J = 1.380649e-23	# J/K
eV_to_K = 8.617333262145e-5	# eV/K

plt.rcParams.update({'font.size': 20})
fig, ax1 = plt.subplots(figsize=(12, 8))

ax1.set_xlabel('Pressure [Pa]')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax3.spines["right"].set_position(("axes", 1.1))
for i_to_scan,to_scan in enumerate([[99,98,96,97],[95,89,87,86,85]]):
	magnetic_field_all = []
	average_static_pressure = []
	average_static_pressure_sigma = []
	average_Te = []
	average_Te_sigma = []
	average_ne = []
	average_ne_sigma = []
	target_chamber_pressure_2 = []

	for merge_ID_target in to_scan:
		input_data_dict = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'/input_data.npz')
		input_data_dict.allow_pickle = True
		merge_Te_prof_multipulse_interp_crop = input_data_dict['Te'].all()['original_TS']['full']['most_likely']
		merge_dTe_prof_multipulse_interp_crop = input_data_dict['Te'].all()['original_TS']['full']['most_likely_sigma']
		merge_ne_prof_multipulse_interp_crop = input_data_dict['ne'].all()['original_TS']['full']['most_likely']
		merge_dne_prof_multipulse_interp_crop = input_data_dict['ne'].all()['original_TS']['full']['most_likely_sigma']

		area_equivalent_to_downstream_peak_pressure = merge_ne_prof_multipulse_interp_crop*1e20*(2*merge_Te_prof_multipulse_interp_crop/eV_to_K*boltzmann_constant_J)
		area_equivalent_to_downstream_peak_pressure_sigma = ((merge_dne_prof_multipulse_interp_crop*1e20*(2*merge_Te_prof_multipulse_interp_crop/eV_to_K*boltzmann_constant_J))**2 + (merge_ne_prof_multipulse_interp_crop*1e20*(2*merge_dTe_prof_multipulse_interp_crop/eV_to_K*boltzmann_constant_J))**2)**0.5

		temp2 = np.max(area_equivalent_to_downstream_peak_pressure,axis=-1)
		temp2_sigma = np.max(area_equivalent_to_downstream_peak_pressure_sigma,axis=-1)
		temp3 = np.max(merge_Te_prof_multipulse_interp_crop,axis=-1)
		temp3_sigma = np.max(merge_dTe_prof_multipulse_interp_crop,axis=-1)
		temp4 = np.max(merge_ne_prof_multipulse_interp_crop,axis=-1)
		temp4_sigma = np.max(merge_dne_prof_multipulse_interp_crop,axis=-1)
		results_summary = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/results_summary.csv',index_col=0)
		fast_camera_record_duration = results_summary.loc[merge_ID_target,['fast_camera_record_duration_long']][0]	# [s]
		fast_camera_record_duration_dt = max(1,int(round(fast_camera_record_duration*1e3/0.05)))

		temp2 = generic_filter(temp2,np.mean,size=[fast_camera_record_duration_dt])
		temp2_sigma = generic_filter(temp2_sigma**2,np.sum,size=[fast_camera_record_duration_dt])**0.5 /(fast_camera_record_duration_dt)
		temp3 = generic_filter(temp3,np.mean,size=[fast_camera_record_duration_dt])
		temp3_sigma = generic_filter(temp3_sigma**2,np.sum,size=[fast_camera_record_duration_dt])**0.5 /(fast_camera_record_duration_dt)
		temp4 = generic_filter(temp4,np.mean,size=[fast_camera_record_duration_dt])
		temp4_sigma = generic_filter(temp4_sigma**2,np.sum,size=[fast_camera_record_duration_dt])**0.5 /(fast_camera_record_duration_dt)


		print('Looking at '+str(merge_ID_target))
		target_chamber_pressure_2.append(np.float(results_summary.loc[merge_ID_target,['p_n [Pa]']]))
		# average_static_pressure.append(np.float(results_summary.loc[merge_ID_target,['average_static_pressure']]))
		# average_Te.append(np.float(results_summary.loc[merge_ID_target,['average_Te']]))
		# average_ne.append(np.float(results_summary.loc[merge_ID_target,['average_ne']]))
		average_static_pressure.append(np.max(temp2))
		average_static_pressure_sigma.append(np.max(temp2_sigma))
		average_Te.append(np.max(temp3))
		average_Te_sigma.append(np.max(temp3_sigma))
		average_ne.append(np.max(temp4))
		average_ne_sigma.append(np.max(temp4_sigma))
		# average_static_pressure.append(np.float(results_summary.loc[merge_ID_target,['peak_static_pressure']]))
		# average_Te.append(np.float(results_summary.loc[merge_ID_target,['peak_Te']]))
		# average_ne.append(np.float(results_summary.loc[merge_ID_target,['peak_ne']]))
		magnetic_field_all.append(np.float(results_summary.loc[merge_ID_target,['B']]))


	a1 = ax1.errorbar(target_chamber_pressure_2,average_Te,yerr=average_Te_sigma,capsize=5,ls=line_style[i_to_scan],color='r',label='Te B=%.3gT' %(np.mean(magnetic_field_all)))
	# a1, = ax1.plot(target_chamber_pressure_2,average_Te,ls=line_style[i_to_scan],color='r',label='Te B=%.3gT' %(np.mean(magnetic_field_all)))
	# a1, = ax1.plot(target_chamber_pressure_2,average_Te,ls=line_style[i_to_scan],color='r',marker='x')
	a2 = ax2.errorbar(target_chamber_pressure_2,average_ne,yerr=average_ne_sigma,capsize=5,ls=line_style[i_to_scan],color='b',label='ne B=%.3gT' %(np.mean(magnetic_field_all)))
	# a2, = ax2.plot(target_chamber_pressure_2,average_ne,ls=line_style[i_to_scan],color='b',label='ne B=%.3gT' %(np.mean(magnetic_field_all)))
	# a2, = ax2.plot(target_chamber_pressure_2,average_ne,ls=line_style[i_to_scan],color='b',marker='x')
	a3 = ax3.errorbar(target_chamber_pressure_2,average_static_pressure,yerr=average_static_pressure_sigma,capsize=5,ls=line_style[i_to_scan],color='g',label='pressure B=%.3gT' %(np.mean(magnetic_field_all)))
	# a3, = ax3.plot(target_chamber_pressure_2,average_static_pressure,ls=line_style[i_to_scan],color='g',label='pressure B=%.3gT' %(np.mean(magnetic_field_all)))
	# a3, = ax3.plot(target_chamber_pressure_2,average_static_pressure,ls=line_style[i_to_scan],color='g',marker='x')
	# ax1.plot(target_chamber_pressure_SS_all,np.max(merge_Te_SS_all,axis=(1)),ls=line_style[i_to_scan],color='r',label='SS Te B=%.3gT' %(np.mean(magnetic_field_all)))
	# ax2.plot(target_chamber_pressure_SS_all,np.max(merge_ne_SS_all,axis=(1)),ls=line_style[i_to_scan],color='c',label='SS ne B=%.3gT' %(np.mean(magnetic_field_all)))

ax1.tick_params(axis='y', labelcolor=a1[0].get_color())
ax2.tick_params(axis='y', labelcolor=a2[0].get_color())
ax3.tick_params(axis='y', labelcolor=a3[0].get_color())
ax1.set_ylabel('smoothed peak Te [eV]', color=a1[0].get_color())
ax2.set_ylabel('smoothed peak ne [10^20 #/m3]', color=a2[0].get_color())  # we already handled the x-label with ax1
ax3.set_ylabel('smoothed peak static pressure [Pa]', color=a3[0].get_color())  # we already handled the x-label with ax1
# ax1.legend(loc=7, fontsize='x-small')
# ax2.legend(loc=6, fontsize='x-small')
# ax3.legend(loc=4, fontsize='x-small')
ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)
ax3.set_ylim(bottom=0)
ax1.grid()
# ax2.grid()
# plt.pause(0.01)
plt.savefig('/home/ffederic/work/Collaboratory/pure_TS_compare2'+ ''+ '.png', bbox_inches='tight')
