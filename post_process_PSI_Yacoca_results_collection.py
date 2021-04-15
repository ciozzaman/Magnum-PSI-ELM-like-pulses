# this script is to collect together results from the Bayesian search and IR camera.
# The outcome I want are the plots for the paper

import numpy as np
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())
# import matplotlib.pyplot as plt
#import .functions
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
from functions.fabio_add import find_index_of_file,energy_flow_from_TS_at_sound_speed,examine_current_trace
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


fdir = '/home/ffederic/work/Collaboratory/test/experimental_data'
df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv', index_col=0)
df_settings = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_3.csv',index_col=0)

results_summary = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/results_summary.csv',index_col=0)
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise', 'teal', 'olive']

# merge_ID_target_multipulse = [95,89,88,87,86]
merge_ID_target_multipulse = [95,89,88,87,86,85]
# merge_ID_target_multipulse = [99,98,96,97]
# merge_ID_target_multipulse = [66,67,68,69,70,74]


target_chamber_pressure = []
target_chamber_pressure_2 = []
target_OES_distance = []
feed_rate_SLM = []
CB_pulse_energy = []
delivered_pulse_energy = []
magnetic_field = []
area_equiv_max_static_pressure = []
T_pre_pulse = []
DT_pulse = []
DT_pulse_late = []
DT_pulse_time_scaled2 = []
pulse_en_semi_inf = []
pulse_en_semi_inf_sigma = []
pulse_en_semi_inf2 = []
pulse_en_semi_inf_sigma2 = []
pulse_en_SS = []
pulse_en_SS2 = []
area_of_interest_IR = []

net_power_removed_plasma_column = []
net_power_removed_plasma_column_sigma = []
tot_rad_power = []
tot_rad_power_sigma = []
power_rec_neutral = []
power_rec_neutral_sigma = []
power_rad_mol = []
power_rad_mol_sigma = []
power_rad_excit = []
power_rad_excit_sigma = []
power_rad_rec_bremm = []
power_rad_rec_bremm_sigma = []
max_CX_energy = []
average_static_pressure = []
average_Te = []
average_ne = []
power_rad_Hm = []
power_rad_Hm_H2p = []
power_rad_Hm_Hp = []
power_rad_H2 = []
power_rad_H2p = []
power_rad_Hm_sigma = []
power_rad_Hm_H2p_sigma = []
power_rad_Hm_Hp_sigma = []
power_rad_H2_sigma = []
power_rad_H2p_sigma = []
power_via_ionisation = []
power_via_ionisation_sigma = []
power_via_recombination = []
power_via_recombination_sigma = []
pulse_t0_semi_inf2 = []
pulse_t0_semi_inf2_sigma = []
net_recombination = []
net_recombination_sigma = []
TS_pulse_duration_at_max_power = []
TS_pulse_duration_at_max_power2 = []
pulse_energy_density = []
pulse_energy_density_sigma = []
pulse_energy_density2 = []
pulse_energy_density_sigma2 = []

j_specific_target_chamber_pressure = []
j_specific_target_OES_distance = []
j_specific_feed_rate_SLM = []
j_specific_CB_pulse_energy = []
j_specific_delivered_pulse_energy = []
j_specific_magnetic_field = []
j_specific_T_pre_pulse = []
j_specific_DT_pulse = []
j_specific_DT_pulse_sigma = []
j_specific_DT_pulse_late = []
j_specific_DT_pulse_late_sigma = []
j_specific_DT_pulse_time_scaled = []
j_specific_DT_pulse_time_scaled_sigma = []
j_specific_pulse_en_semi_inf = []
j_specific_pulse_en_semi_inf_sigma = []
j_specific_pulse_en_SS = []
j_specific_pulse_t0_semi_inf = []
j_specific_pulse_t0_semi_inf_sigma = []
j_specific_area_of_interest_IR = []
j_specific_area_of_interest_IR_sigma = []
j_specific_energy_density = []
j_specific_energy_density_sigma = []
j_specific_merge_ID_target = []

j_twoD_peak_evolution_time_averaged = []
j_twoD_peak_evolution_temp_averaged_delta = []
j_twoD_peak_evolution_temp_averaged = []
j_collect_shape_information = []
j_TS_pulse_duration_at_max_power = []
radial_average_brightness_OES_location_1ms_int_time = []
radial_average_brightness_1ms_int_time = []
radial_average_brightness_bayesian = []
radial_average_brightness_bayesian_long = []
TS_time = []
energy_flow = []
TS_time_interp = []
energy_flow_interp = []
time_source_power = []
power_pulse_shape = []

for merge_ID_target in merge_ID_target_multipulse:
	print('Looking at '+str(merge_ID_target))
	all_j=find_index_of_file(merge_ID_target,df_settings,df_log,only_OES=True)
	target_chamber_pressure_2.append(np.float(results_summary.loc[merge_ID_target,['p_n [Pa]']]))
	path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)
	full_saved_file_dict = np.load(path_where_to_save_everything+'/Yacora_Bayesian/absolute/lines_fitted5/fit_bounds_from_sims/spatial_factor_1/time_shift_factor_0/only_Hm_H2_H2p_mol_lim/bayesian_results3'+'.npz')
	# full_saved_file_dict = np.load(path_where_to_save_everything+'/Yacora_Bayesian/absolute/lines_fitted5/fit_bounds_from_sims/spatial_factor_1/time_shift_factor_0/only_Hm_H2p_mol_lim/bayesian_results3'+'.npz')

	# (merge_folder,sequence,fname_current_trace) = df_log.loc[all_j[0],['folder','sequence','current_trace_file']]
	# sequence = int(sequence)
	# bad_pulses,first_good_pulse,first_pulse,last_pulse,miss_pulses,double_pulses,good_pulses, time_of_pulses, energy_per_pulse,duration_per_pulse,median_energy_delivered_good_pulses,median_duration_good_pulses,mean_peak_shape,mean_peak_std,mean_steady_state_power,mean_steady_state_power_std,time_resolution,*trash = examine_current_trace(fdir+'/'+merge_folder+'/'+"{0:0=2d}".format(sequence)+'/', fname_current_trace, df_log.loc[all_j[0], ['number_of_pulses']][0],want_the_power_per_pulse=True,want_the_mean_power_profile=True)
	# # power_pulse_shape.append(mean_peak_shape)

	if merge_ID_target != 88:
		time_source_power.append(full_saved_file_dict['input_power'].all()['TS']['time_source_power'])
		power_pulse_shape.append(full_saved_file_dict['input_power'].all()['TS']['power_pulse_shape'])
		temp = energy_flow_from_TS_at_sound_speed(merge_ID_target,interpolated_TS=False)
		TS_time.append(temp[0])
		energy_flow.append(temp[1])
		temp = energy_flow_from_TS_at_sound_speed(merge_ID_target,interpolated_TS=True)
		TS_time_interp.append(temp[0])
		energy_flow_interp.append(temp[1])
		temp[1][temp[1]>0]-=np.mean(temp[1][temp[0]<=0])
		temp1=np.cumsum(temp[1])
		TS_pulse_duration_at_max_power.append(temp1.max()/np.max(temp[1])*np.median(np.diff(temp[0]))*1e-3)
		TS_pulse_duration_at_max_power2.append(temp1.max()/np.max(temp[1])*np.median(np.diff(temp[0]))*1e-3)
		target_chamber_pressure.append(np.float(results_summary.loc[merge_ID_target,['p_n [Pa]']]))
		target_OES_distance.append(np.float(results_summary.loc[merge_ID_target,['T_axial']]))
		feed_rate_SLM.append(np.float(results_summary.loc[merge_ID_target,['Seed']]))
		CB_pulse_energy.append(np.float(results_summary.loc[merge_ID_target,['CB energy [J]']]))
		delivered_pulse_energy.append(np.float(results_summary.loc[merge_ID_target,['Delivered energy [J]']]))
		magnetic_field.append(np.float(results_summary.loc[merge_ID_target,['B']]))
		area_equiv_max_static_pressure.append(np.float(results_summary.loc[merge_ID_target,['area_equiv_max_static_pressure']]))

		# net_power_removed_plasma_column.append(np.float(results_summary.loc[merge_ID_target,['net_power_removed_plasma_column']]))
		# net_power_removed_plasma_column_sigma.append(np.float(results_summary.loc[merge_ID_target,['net_power_removed_plasma_column_sigma']]))
		net_power_removed_plasma_column.append(full_saved_file_dict['net_power_removed_plasma_column'].all()['radial_time_sum']['most_likely'])
		net_power_removed_plasma_column_sigma.append(full_saved_file_dict['net_power_removed_plasma_column'].all()['radial_time_sum']['most_likely_sigma'])
		tot_rad_power.append(full_saved_file_dict['tot_rad_power'].all()['radial_time_sum']['most_likely'])
		# tot_rad_power_sigma.append(np.float(results_summary.loc[merge_ID_target,['tot_rad_power_sigma']]))
		tot_rad_power_sigma.append(full_saved_file_dict['tot_rad_power'].all()['radial_time_sum']['most_likely_sigma'])
		power_via_ionisation.append(full_saved_file_dict['power_via_ionisation'].all()['radial_time_sum']['most_likely'])
		# power_via_ionisation_sigma.append(np.float(results_summary.loc[merge_ID_target,['power_via_ionisation_sigma']]))
		power_via_ionisation_sigma.append(full_saved_file_dict['power_via_ionisation'].all()['radial_time_sum']['most_likely_sigma'])
		power_via_recombination.append(full_saved_file_dict['power_via_recombination'].all()['radial_time_sum']['most_likely'])
		# power_via_recombination_sigma.append(np.float(results_summary.loc[merge_ID_target,['power_via_recombination_sigma']]))
		power_via_recombination_sigma.append(full_saved_file_dict['power_via_recombination'].all()['radial_time_sum']['most_likely_sigma'])
		power_rec_neutral.append(full_saved_file_dict['power_rec_neutral'].all()['radial_time_sum']['most_likely'])
		# power_rec_neutral_sigma.append(np.float(results_summary.loc[merge_ID_target,['power_rec_neutral_sigma']]))
		power_rec_neutral_sigma.append(full_saved_file_dict['power_rec_neutral'].all()['radial_time_sum']['most_likely_sigma'])
		power_rad_mol.append(full_saved_file_dict['power_rad_mol'].all()['radial_time_sum']['most_likely'])
		# power_rad_mol_sigma.append(np.float(results_summary.loc[merge_ID_target,['power_rad_mol_sigma']]))
		power_rad_mol_sigma.append(full_saved_file_dict['power_rad_mol'].all()['radial_time_sum']['most_likely_sigma'])
		power_rad_excit.append(full_saved_file_dict['power_rad_excit'].all()['radial_time_sum']['most_likely'])
		# power_rad_excit_sigma.append(np.float(results_summary.loc[merge_ID_target,['power_rad_excit_sigma']]))
		power_rad_excit_sigma.append(full_saved_file_dict['power_rad_excit'].all()['radial_time_sum']['most_likely_sigma'])
		power_rad_rec_bremm.append(full_saved_file_dict['power_rad_rec_bremm'].all()['radial_time_sum']['most_likely'])
		# power_rad_rec_bremm_sigma.append(np.float(results_summary.loc[merge_ID_target,['power_rad_rec_bremm_sigma']]))
		power_rad_rec_bremm_sigma.append(full_saved_file_dict['power_rad_rec_bremm'].all()['radial_time_sum']['most_likely_sigma'])
		# max_CX_energy.append(np.float(results_summary.loc[merge_ID_target,['max_CX_energy']]))
		max_CX_energy.append(full_saved_file_dict['P_HCX_3'].all()['total_energy']['most_likely'])
		average_static_pressure.append(np.float(results_summary.loc[merge_ID_target,['average_static_pressure']]))
		average_Te.append(np.float(results_summary.loc[merge_ID_target,['average_Te']]))
		average_ne.append(np.float(results_summary.loc[merge_ID_target,['average_ne']]))
		power_rad_Hm.append(full_saved_file_dict['power_rad_Hm'].all()['radial_time_sum']['most_likely'])
		# power_rad_Hm_sigma.append(np.float(results_summary.loc[merge_ID_target,['power_rad_Hm_sigma']]))
		power_rad_Hm_sigma.append(full_saved_file_dict['power_rad_Hm'].all()['radial_time_sum']['most_likely_sigma'])
		power_rad_Hm_H2p.append(full_saved_file_dict['power_rad_Hm_H2p'].all()['radial_time_sum']['most_likely'])
		power_rad_Hm_H2p_sigma.append(full_saved_file_dict['power_rad_Hm_H2p'].all()['radial_time_sum']['most_likely_sigma'])
		power_rad_Hm_Hp.append(full_saved_file_dict['power_rad_Hm_Hp'].all()['radial_time_sum']['most_likely'])
		power_rad_Hm_Hp_sigma.append(full_saved_file_dict['power_rad_Hm_Hp'].all()['radial_time_sum']['most_likely_sigma'])
		power_rad_H2.append(full_saved_file_dict['power_rad_H2'].all()['radial_time_sum']['most_likely'])
		# power_rad_H2_sigma.append(np.float(results_summary.loc[merge_ID_target,['power_rad_H2_sigma']]))
		power_rad_H2_sigma.append(full_saved_file_dict['power_rad_H2'].all()['radial_time_sum']['most_likely_sigma'])
		power_rad_H2p.append(full_saved_file_dict['power_rad_H2p'].all()['radial_time_sum']['most_likely'])
		# power_rad_H2p_sigma.append(np.float(results_summary.loc[merge_ID_target,['power_rad_H2p_sigma']]))
		power_rad_H2p_sigma.append(full_saved_file_dict['power_rad_H2p'].all()['radial_time_sum']['most_likely_sigma'])
		radial_average_brightness_bayesian.append(full_saved_file_dict['total_removed_power_visible'].all()['average_brightness']['most_likely'])
		radial_average_brightness_bayesian_long.append(full_saved_file_dict['total_removed_power_visible'].all()['long_average_brightness']['most_likely'])
		net_recombination.append(full_saved_file_dict['power_rad_rec_bremm'].all()['radial_time_sum']['most_likely']-full_saved_file_dict['power_heating_rec'].all()['radial_time_sum']['most_likely'])
		net_recombination_sigma.append((full_saved_file_dict['power_rad_rec_bremm'].all()['radial_time_sum']['most_likely_sigma']**2 + full_saved_file_dict['power_heating_rec'].all()['radial_time_sum']['most_likely_sigma']**2)**0.5)
	else:
		TS_pulse_duration_at_max_power.append(0.4e-3)

	path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)
	full_saved_file_dict = np.load(path_where_to_save_everything +'/fast_camera_merge_'+str(merge_ID_target)+'.npz')
	radial_average_brightness_OES_location_1ms_int_time.append(full_saved_file_dict['radial_average_brightness_OES_location_1ms_int_time'])
	radial_average_brightness_1ms_int_time.append(full_saved_file_dict['radial_average_brightness_1ms_int_time'])

	temp1=[]
	temp2=[]
	temp3=[]
	temp4=[]
	temp5=[]
	temp6=[]
	temp7=[]
	temp8=[]
	temp9=[]
	temp10=[]
	temp11=[]
	for j in all_j:
		IR_trace, = df_log.loc[j,['IR_trace']]
		if isinstance(IR_trace,str):
			j_specific_merge_ID_target.append(merge_ID_target)
			j_specific_target_chamber_pressure.append(np.float(results_summary.loc[merge_ID_target,['p_n [Pa]']]))
			j_specific_target_OES_distance.append(np.float(results_summary.loc[merge_ID_target,['T_axial']]))
			j_specific_feed_rate_SLM.append(np.float(results_summary.loc[merge_ID_target,['Seed']]))
			j_specific_CB_pulse_energy.append(np.float(results_summary.loc[merge_ID_target,['CB energy [J]']]))
			j_specific_delivered_pulse_energy.append(np.float(results_summary.loc[merge_ID_target,['Delivered energy [J]']]))
			j_specific_magnetic_field.append(np.float(results_summary.loc[merge_ID_target,['B']]))

			path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)
			full_saved_file_dict = dict(np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz'))
			# j_specific_T_pre_pulse.append(df_log.loc[j,['T_pre_pulse']])
			# temp1.append(df_log.loc[j,['T_pre_pulse']])
			# j_specific_DT_pulse.append(df_log.loc[j,['DT_pulse']])
			# temp2.append(df_log.loc[j,['DT_pulse']])
			# j_specific_DT_pulse_late.append(df_log.loc[j,['DT_pulse_late']])
			# temp8.append(df_log.loc[j,['DT_pulse_late']])
			# j_specific_pulse_en_semi_inf.append(df_log.loc[j,['pulse_en_semi_inf [J]']])
			# temp3.append(df_log.loc[j,['pulse_en_semi_inf [J]']])
			# temp5.append(df_log.loc[j,['pulse_en_semi_inf_sigma [J]']])
			# j_specific_area_of_interest_IR.append(df_log.loc[j,['area_of_interest [m2]']])
			# j_specific_area_of_interest_IR.append(df_log.loc[j,['area_of_interest [m2]']])
			# temp4.append(df_log.loc[j,['area_of_interest [m2]']])
			# j_specific_pulse_t0_semi_inf.append(df_log.loc[j,['pulse_t0_semi_inf [ms]']])
			# temp6.append(df_log.loc[j,['pulse_t0_semi_inf [ms]']][0])
			# j_specific_DT_pulse_time_scaled.append(df_log.loc[j,['DT_pulse_time_scaled']])
			# temp7.append(df_log.loc[j,['DT_pulse_time_scaled']])
			# j_specific_pulse_en_SS.append(df_log.loc[j,['SS_Energy [J]']])
			# temp9.append(df_log.loc[j,['SS_Energy [J]']])
			j_specific_T_pre_pulse.append(full_saved_file_dict['average_pulse_fitted_data'].all()['temperature']['steady_state'])
			temp1.append(full_saved_file_dict['average_pulse_fitted_data'].all()['temperature']['steady_state'])
			j_specific_DT_pulse.append(full_saved_file_dict['average_pulse_fitted_data'].all()['temperature']['peak_plus_2_3'])
			j_specific_DT_pulse_sigma.append(full_saved_file_dict['average_pulse_fitted_data'].all()['temperature']['peak_plus_2_3_sigma'])
			temp2.append(full_saved_file_dict['average_pulse_fitted_data'].all()['temperature']['peak_plus_2_3'])
			j_specific_DT_pulse_late.append(full_saved_file_dict['average_pulse_fitted_data'].all()['temperature']['peak_plus_10_11'])
			j_specific_DT_pulse_late_sigma.append(full_saved_file_dict['average_pulse_fitted_data'].all()['temperature']['peak_plus_10_11_sigma'])
			temp8.append(full_saved_file_dict['average_pulse_fitted_data'].all()['temperature']['peak_plus_10_11'])
			j_specific_pulse_en_semi_inf.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_energy'])
			temp3.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_energy'])
			j_specific_pulse_en_semi_inf_sigma.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_energy_sigma'])
			temp5.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_energy_sigma'])
			j_specific_area_of_interest_IR.append(full_saved_file_dict['average_pulse_fitted_data'].all()['area_of_interest'])
			j_specific_area_of_interest_IR_sigma.append(full_saved_file_dict['average_pulse_fitted_data'].all()['area_of_interest_sigma'])
			j_specific_energy_density.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_energy_density'])
			j_specific_energy_density_sigma.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_energy_density_sigma'])
			temp10.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_energy_density'])
			temp11.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_energy_density_sigma'])
			temp4.append(full_saved_file_dict['average_pulse_fitted_data'].all()['area_of_interest'])
			j_specific_pulse_t0_semi_inf.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_t0_semi_inf'])
			j_specific_pulse_t0_semi_inf_sigma.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_t0_semi_inf_sigma'])
			temp6.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_t0_semi_inf'])
			j_specific_DT_pulse_time_scaled.append(full_saved_file_dict['average_pulse_peak_data'].all()['energy_fit_fix_duration']['DT_pulse_time_scaled'])
			j_specific_DT_pulse_time_scaled_sigma.append(full_saved_file_dict['average_pulse_peak_data'].all()['energy_fit_fix_duration']['DT_pulse_time_scaled_sigma'])
			temp7.append(full_saved_file_dict['average_pulse_peak_data'].all()['energy_fit_fix_duration']['DT_pulse_time_scaled'])
			j_specific_pulse_en_SS.append(full_saved_file_dict['average_pulse_fitted_data'].all()['SS_Energy'])
			temp9.append(full_saved_file_dict['average_pulse_fitted_data'].all()['SS_Energy'])
			j_TS_pulse_duration_at_max_power.append(TS_pulse_duration_at_max_power[-1])

		# if np.logical_not(isinstance(IR_trace,str)):
		# 	j_twoD_peak_evolution_time_averaged.append(np.ones_like(j_twoD_peak_evolution_time_averaged[-1])*np.nan)
		# 	# j_twoD_peak_evolution_temp_averaged_delta.append(np.ones_like(j_twoD_peak_evolution_temp_averaged_delta[-1])*np.nan)
		# 	# j_twoD_peak_evolution_temp_averaged.append(np.ones_like(j_twoD_peak_evolution_temp_averaged[-1])*np.nan)
		# 	j_collect_shape_information.append(np.ones_like(j_collect_shape_information[-1])*np.nan)
		# else:
			j_twoD_peak_evolution_time_averaged.append(full_saved_file_dict['twoD_peak_evolution_time_averaged'])
			# j_twoD_peak_evolution_temp_averaged_delta.append(full_saved_file_dict['twoD_peak_evolution_temp_averaged_delta'])
			# j_twoD_peak_evolution_temp_averaged.append(full_saved_file_dict['twoD_peak_evolution_temp_averaged'])
			j_collect_shape_information.append(full_saved_file_dict['collect_shape_information'])

	T_pre_pulse.append(np.nanmean(temp1))
	DT_pulse.append(np.nanmean(temp2))
	DT_pulse_late.append(np.nanmean(temp8))
	pulse_en_semi_inf2.append(np.nansum(np.divide(temp3,temp5))/np.nansum(np.divide(1,temp5)))
	pulse_en_semi_inf_sigma2.append((np.nansum(np.isfinite(np.array(temp5,dtype=float)))/(np.nansum(np.divide(1,temp5))**2))**0.5)
	pulse_en_SS2.append(np.nanmean(temp9))
	pulse_t0_semi_inf2.append(np.nanmean(temp6))
	DT_pulse_time_scaled2.append(np.nanmean(temp7))
	pulse_energy_density2.append(np.nanmean(temp10))
	pulse_energy_density_sigma2.append((np.nansum(np.isfinite(np.array(temp11,dtype=float)))/(np.nansum(np.divide(1,temp11))**2))**0.5)
	if merge_ID_target != 88:
		pulse_energy_density.append(np.nanmean(temp10))
		pulse_energy_density_sigma.append((np.nansum(np.isfinite(np.array(temp11,dtype=float)))/(np.nansum(np.divide(1,temp11))**2))**0.5)
		pulse_en_semi_inf.append(np.nansum(np.divide(temp3,temp5))/np.nansum(np.divide(1,temp5)))
		pulse_en_semi_inf_sigma.append((np.nansum(np.isfinite(np.array(temp5,dtype=float)))/(np.nansum(np.divide(1,temp5))**2))**0.5)
		pulse_en_SS.append(np.nanmean(temp9))
		area_of_interest_IR.append(np.nanmean(temp4))

# plt.figure(figsize=(10, 5))
# plt.plot(j_specific_target_chamber_pressure,j_specific_T_pre_pulse,'+b')
# plt.plot(target_chamber_pressure,T_pre_pulse,'b')
# plt.plot(j_specific_target_chamber_pressure,np.array(j_specific_DT_pulse),'+r')
# plt.plot(target_chamber_pressure,np.array(DT_pulse),'r')
# plt.grid()
# plt.pause(0.001)

if True:	#			PLOTS FOR THE PAPER
	fig, ax1 = plt.subplots(figsize=(12, 5))
	fig.subplots_adjust(right=0.77)
	ax1.set_title('Pressure scan magnetic_field %.3gT,target/OES distance %.3gmm,ELM pulse voltage %.3gV' %(magnetic_field[0],target_OES_distance[0],(2*CB_pulse_energy[0]/150e-6)**0.5)+'\nIR data analysis\n ')
	ax1.set_xlabel('Target chamber pressure [Pa]')
	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	# ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	# ax3.spines["right"].set_position(("axes", 1.1125))
	# ax3.spines["right"].set_visible(True)
	ax3.spines["right"].set_position(("axes", 1.1125))
	ax3.spines["right"].set_visible(True)
	ax1.plot(j_specific_target_chamber_pressure,j_specific_T_pre_pulse,'xb')
	a1, = ax1.plot(target_chamber_pressure_2,T_pre_pulse,'b')
	# ax2.plot(j_specific_target_chamber_pressure,np.array(j_specific_DT_pulse),'+r')
	# a2, = ax2.plot(target_chamber_pressure_2,np.array(DT_pulse),'--r',label='mean temperature peak + 2/3ms')
	# ax2.plot(j_specific_target_chamber_pressure,np.array(j_specific_DT_pulse_time_scaled),'+r')
	# ax2.plot(target_chamber_pressure_2,np.array(DT_pulse_time_scaled2),'r',label='mean temperature t0 + 2/3ms')
	ax2.errorbar(j_specific_target_chamber_pressure,j_specific_DT_pulse_late,yerr=np.array(j_specific_DT_pulse_late_sigma),fmt='o',color='r',capsize=5)
	a2, = ax2.plot(target_chamber_pressure_2,np.array(DT_pulse_late),'r',label='mean temperature peak + 10/11ms')
	# ax3.plot(j_specific_target_chamber_pressure,np.array(j_specific_pulse_t0_semi_inf),'+g')
	# a3, = ax3.plot(target_chamber_pressure_2,np.array(pulse_t0_semi_inf2),'g')
	ax3.errorbar(j_specific_target_chamber_pressure,np.array(j_specific_pulse_en_semi_inf)+np.array(j_specific_pulse_en_SS),yerr=j_specific_pulse_en_semi_inf_sigma,fmt='o',color='g',capsize=5)
	ax3.plot(j_specific_target_chamber_pressure,np.array(j_specific_pulse_en_SS),'xg')
	a3a, = ax3.plot(target_chamber_pressure_2,np.array(pulse_en_SS2),'--g',label='only steady state')
	a3, = ax3.plot(target_chamber_pressure_2,np.array(pulse_en_semi_inf2)+np.array(pulse_en_SS2),'g')
	# ax2.plot(j_specific_target_chamber_pressure,np.array(j_specific_DT_pulse_time_scaled),'or')
	ax1.set_ylabel('Temp before ELM-like pulse (T(t=0)) [°C]', color=a1.get_color())
	ax2.set_ylabel(r'$\Delta T$'+' after ELM [°C]', color=a2.get_color())  # we already handled the x-label with ax1
	# ax3.set_ylabel('ELM-like pulse start to IR peak (t0) [ms]', color=a3.get_color())  # we already handled the x-label with ax1
	ax3.set_ylabel('Energy reaching the target [J]', color=a3.get_color())  # we already handled the x-label with ax1
	ax1.tick_params(axis='y', labelcolor=a1.get_color())
	ax2.tick_params(axis='y', labelcolor=a2.get_color())
	# ax3.tick_params(axis='y', labelcolor=a3.get_color())
	ax3.tick_params(axis='y', labelcolor=a3.get_color())
	ax2.legend()
	handles, labels = ax2.get_legend_handles_labels()
	handles.append(a3a)
	labels.append(a3a.get_label())
	ax2.legend(handles=handles, labels=labels, loc='upper center', fontsize='x-small')
	ax1.set_ylim(bottom=0)
	ax2.set_ylim(bottom=0)
	ax3.set_ylim(bottom=0)
	# ax2.set_ylim(bottom=0,top=max(np.nanmax(j_specific_DT_pulse),np.max(DT_pulse_time_scaled2))*1.1)
	# ax3.set_ylim(bottom=0,top=np.max([j_specific_DT_pulse,j_specific_DT_pulse_time_scaled])*1.1)
	ax1.grid()
	# ax2.grid()
	plt.pause(0.01)

	fig, ax = plt.subplots( 2,1,figsize=(10, 9), squeeze=False, sharex=True)
	fig.suptitle('Pressure scan\nmagnetic_field %.3gT,target/OES distance %.3gmm,ELM energy %.3gJ\n ' %(magnetic_field[0],target_OES_distance[0],CB_pulse_energy[0]))
	ax[0,0].errorbar(j_specific_pulse_en_semi_inf,1e-6*np.array(j_specific_energy_density)/(np.array(j_TS_pulse_duration_at_max_power)**0.5),xerr=j_specific_pulse_en_semi_inf_sigma,yerr=1e-6*np.array(j_specific_energy_density_sigma)/(np.array(j_TS_pulse_duration_at_max_power)**0.5),fmt='+',color='b',capsize=5)
	temp = np.polyfit(j_specific_pulse_en_semi_inf,1e-6*np.array(j_specific_energy_density)/(np.array(j_TS_pulse_duration_at_max_power)**0.5),1)
	ax[0,0].plot(np.sort(j_specific_pulse_en_semi_inf),np.polyval(temp,np.sort(j_specific_pulse_en_semi_inf)),'--b')
	ax[1,0].errorbar(j_specific_pulse_en_semi_inf,j_specific_area_of_interest_IR,xerr=j_specific_pulse_en_semi_inf_sigma,yerr=j_specific_area_of_interest_IR_sigma,fmt='+',color='r',capsize=5)
	temp = np.polyfit(j_specific_pulse_en_semi_inf,j_specific_area_of_interest_IR,1)
	ax[1,0].plot(np.sort(j_specific_pulse_en_semi_inf),np.polyval(temp,np.sort(j_specific_pulse_en_semi_inf)),'--r')
	ax[0,0].grid()
	ax[1,0].grid()
	ax[0,0].set_ylabel('Heat flux factor [MJ s^-1/2]')
	ax[1,0].set_ylabel('Interested area found [mm2]')
	ax[1,0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	ax[1,0].set_xlabel('Pulse energy [J]')
	plt.pause(0.001)

else:
	fig, ax1 = plt.subplots(figsize=(12, 5))
	fig.subplots_adjust(right=0.8)
	ax1.set_title('Pressure scan magnetic_field %.3gT,target/OES distance %.3gmm,ELM pulse voltage %.3gV' %(magnetic_field[0],target_OES_distance[0],(2*CB_pulse_energy[0]/150e-6)**0.5)+'\nIR data analysis')
	ax1.set_xlabel('Target chamber pressure [Pa]')
	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	# ax4 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	ax3.spines["right"].set_position(("axes", 1.1))
	ax3.spines["right"].set_visible(True)
	# ax4.spines["right"].set_position(("axes", 1.35))
	# ax4.spines["right"].set_visible(True)
	ax1.plot(j_specific_target_chamber_pressure,j_specific_T_pre_pulse,'+b')
	a1, = ax1.plot(target_chamber_pressure_2,T_pre_pulse,'b')
	ax2.plot(j_specific_target_chamber_pressure,np.array(j_specific_DT_pulse),'+r')
	a2, = ax2.plot(target_chamber_pressure_2,np.array(DT_pulse),'--r',label='mean temperature peak + 2/3ms')
	a2, = ax2.plot(target_chamber_pressure_2,np.array(DT_pulse_late),':r',label='mean temperature peak + 10/11ms')
	ax3.plot(j_specific_target_chamber_pressure,np.array(j_specific_pulse_t0_semi_inf),'+g')
	a3, = ax3.plot(target_chamber_pressure_2,np.array(pulse_t0_semi_inf2),'g')
	# ax4.plot(j_specific_target_chamber_pressure,np.array(j_specific_DT_pulse_time_scaled),'+m')
	# a4, = ax4.plot(target_chamber_pressure_2,np.array(DT_pulse_time_scaled2),'m')
	ax2.plot(target_chamber_pressure_2,np.array(DT_pulse_time_scaled2),'r',label='mean temperature peak + t0 + 2/3ms')
	# ax2.plot(j_specific_target_chamber_pressure,np.array(j_specific_DT_pulse_time_scaled),'or')
	ax1.set_ylabel('Temp before ELM-like pulse (T(t=0)) [°C]', color=a1.get_color())
	ax2.set_ylabel(r'$\Delta T$'+' before - mean 2/3 ms after ELM [°C]', color=a2.get_color())  # we already handled the x-label with ax1
	ax3.set_ylabel('ELM-like pulse start to IR peak (t0) [ms]', color=a3.get_color())  # we already handled the x-label with ax1
	# ax4.set_ylabel(r'$\Delta T$'+' 2.5ms after ELM-like pulse start [°C]', color=a4.get_color())  # we already handled the x-label with ax1
	ax1.tick_params(axis='y', labelcolor=a1.get_color())
	ax2.tick_params(axis='y', labelcolor=a2.get_color())
	ax3.tick_params(axis='y', labelcolor=a3.get_color())
	# ax4.tick_params(axis='y', labelcolor=a4.get_color())
	ax2.legend(loc='center left', fontsize='x-small')
	ax1.set_ylim(bottom=0)
	ax2.set_ylim(bottom=0)
	# ax2.set_ylim(bottom=0,top=max(np.nanmax(j_specific_DT_pulse),np.max(DT_pulse_time_scaled2))*1.1)
	# ax4.set_ylim(bottom=0,top=np.max([j_specific_DT_pulse,j_specific_DT_pulse_time_scaled])*1.1)
	ax1.grid()
	# ax2.grid()
	plt.pause(0.01)


	fig, ax1 = plt.subplots(figsize=(12, 5))
	fig.subplots_adjust(right=0.8)
	ax1.set_title('Pressure scan magnetic_field %.3gT,target/OES distance %.3gmm,ELM pulse voltage %.3gV' %(magnetic_field[0],target_OES_distance[0],(2*CB_pulse_energy[0]/150e-6)**0.5)+'\nIR data analysis')
	ax1.set_xlabel('Target chamber pressure [Pa]')
	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	ax3.spines["right"].set_position(("axes", 1.1))
	ax3.spines["right"].set_visible(True)
	ax1.plot(j_specific_target_chamber_pressure,j_specific_T_pre_pulse,'+b')
	a1, = ax1.plot(target_chamber_pressure_2,T_pre_pulse,'b')
	ax2.plot(j_specific_target_chamber_pressure,np.array(j_specific_pulse_en_semi_inf),'+r')
	a2, = ax2.plot(target_chamber_pressure_2,np.array(pulse_en_semi_inf2),'r')
	ax3.plot(j_specific_target_chamber_pressure,np.array(j_specific_pulse_t0_semi_inf),'+g')
	a3, = ax3.plot(target_chamber_pressure_2,np.array(pulse_t0_semi_inf2),'g')
	ax1.set_ylabel('Temp before ELM-like pulse (T(t=0)) [°C]', color=a1.get_color())
	ax2.set_ylabel('Energy reaching the target [J]', color=a2.get_color())  # we already handled the x-label with ax1
	ax3.set_ylabel('ELM-like pulse start to IR peak (t0) [ms]', color=a3.get_color())  # we already handled the x-label with ax1
	ax1.tick_params(axis='y', labelcolor=a1.get_color())
	ax2.tick_params(axis='y', labelcolor=a2.get_color())
	ax3.tick_params(axis='y', labelcolor=a3.get_color())
	# ax1.legend(loc='best', fontsize='x-small')
	# ax2.legend(loc='best', fontsize='x-small')
	ax1.set_ylim(bottom=0)
	ax2.set_ylim(bottom=0)
	# ax2.set_ylim(bottom=0)
	ax1.grid()
	# ax2.grid()
	plt.pause(0.01)

	fig, ax1 = plt.subplots(figsize=(12, 5))
	fig.subplots_adjust(right=0.77)
	ax1.set_title('Pressure scan magnetic_field %.3gT,target/OES distance %.3gmm,ELM pulse voltage %.3gV' %(magnetic_field[0],target_OES_distance[0],(2*CB_pulse_energy[0]/150e-6)**0.5)+'\nIR data analysis\n ')
	ax1.set_xlabel('Pulse energy [J]')
	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	ax1.errorbar(j_specific_pulse_en_semi_inf,1e-6*np.array(j_specific_energy_density)/(np.array(j_TS_pulse_duration_at_max_power)**0.5),xerr=j_specific_pulse_en_semi_inf_sigma,yerr=1e-6*np.array(j_specific_energy_density_sigma)/(np.array(j_TS_pulse_duration_at_max_power)**0.5),fmt='+',color='b',capsize=5)
	temp = np.polyfit(j_specific_pulse_en_semi_inf,1e-6*np.array(j_specific_energy_density)/(np.array(j_TS_pulse_duration_at_max_power)**0.5),1)
	a1, = ax1.plot(np.sort(j_specific_pulse_en_semi_inf),np.polyval(temp,np.sort(j_specific_pulse_en_semi_inf)),'--b')
	ax2.errorbar(j_specific_pulse_en_semi_inf,j_specific_area_of_interest_IR,xerr=j_specific_pulse_en_semi_inf_sigma,yerr=j_specific_area_of_interest_IR_sigma,fmt='+',color='r',capsize=5)
	temp = np.polyfit(j_specific_pulse_en_semi_inf,j_specific_area_of_interest_IR,1)
	a2, = ax2.plot(np.sort(j_specific_pulse_en_semi_inf),np.polyval(temp,np.sort(j_specific_pulse_en_semi_inf)),'--r')
	ax1.set_ylabel('Heat flux factor [MJ s^-1/2]', color=a1.get_color())
	ax2.set_ylabel('Interested area found [mm2]', color=a2.get_color())  # we already handled the x-label with ax1
	ax1.tick_params(axis='y', labelcolor=a1.get_color())
	ax2.tick_params(axis='y', labelcolor=a2.get_color())
	ax1.set_ylim(bottom=0)
	ax2.set_ylim(bottom=0)
	ax1.grid()
	plt.pause(0.01)

	plt.figure(figsize=(12, 5))
	plt.plot(target_chamber_pressure_2,1e-6*np.array(pulse_energy_density2)*np.array(TS_pulse_duration_at_max_power)**0.5)
	plt.plot(j_specific_target_chamber_pressure,1e-6*np.array(j_specific_energy_density)*np.array(j_TS_pulse_duration_at_max_power)**0.5,'+')
	plt.ylabel('Heat flux factor [Mj s^-1/2]')
	plt.xlabel('Target chamber pressure [Pa]')
	plt.pause(0.01)

	plt.figure(figsize=(12, 5))
	# plt.plot(target_chamber_pressure_2,1e-6*np.array(pulse_energy_density)*np.array(TS_pulse_duration_at_max_power)**0.5)
	plt.errorbar(j_specific_pulse_en_semi_inf,1e-6*np.array(j_specific_energy_density)/(np.array(j_TS_pulse_duration_at_max_power)**0.5),xerr=j_specific_pulse_en_semi_inf_sigma,yerr=1e-6*np.array(j_specific_energy_density_sigma)/(np.array(j_TS_pulse_duration_at_max_power)**0.5),fmt='+')
	plt.ylabel('Heat flux factor [MJ s^-1/2]')
	plt.xlabel('Pulse delivered energy [J]')
	plt.pause(0.01)

	plt.figure(figsize=(12, 5))
	# plt.plot(target_chamber_pressure_2,1e-6*np.array(pulse_energy_density)*np.array(TS_pulse_duration_at_max_power)**0.5)
	plt.errorbar(1e-6*np.array(j_specific_energy_density)/(np.array(j_TS_pulse_duration_at_max_power)**0.5),j_specific_area_of_interest_IR,fmt='+')
	plt.ylabel('Heat flux factor [MJ s^-1/2]')
	plt.xlabel('Interested area found [mm2]')
	plt.pause(0.01)

	plt.figure(figsize=(12, 5))
	# plt.plot(target_chamber_pressure_2,1e-6*np.array(pulse_energy_density)*np.array(TS_pulse_duration_at_max_power)**0.5)
	plt.errorbar(1e-6*np.array(pulse_energy_density)/(np.array(TS_pulse_duration_at_max_power2)**0.5),net_power_removed_plasma_column,xerr=1e-6*np.array(pulse_energy_density_sigma)/(np.array(TS_pulse_duration_at_max_power2)**0.5),yerr=net_power_removed_plasma_column_sigma,fmt='+')
	plt.xlabel('Heat flux factor [MJ s^-1/2]')
	plt.ylabel('Energy removed from plasma column [J]')
	plt.pause(0.01)

	plt.figure(figsize=(12, 5))
	# plt.plot(target_chamber_pressure_2,1e-6*np.array(pulse_energy_density)*np.array(TS_pulse_duration_at_max_power)**0.5)
	plt.errorbar(pulse_en_semi_inf,net_power_removed_plasma_column,xerr=pulse_en_semi_inf_sigma,yerr=net_power_removed_plasma_column_sigma,fmt='+')
	plt.xlabel('Pulse delivered energy [J]')
	plt.ylabel('Energy removed from plasma column [J]')
	plt.pause(0.01)

	plt.figure(figsize=(12, 5))
	# plt.plot(target_chamber_pressure_2,1e-6*np.array(pulse_energy_density)*np.array(TS_pulse_duration_at_max_power)**0.5)
	plt.errorbar(pulse_en_semi_inf,np.array(area_equiv_max_static_pressure)**0.5*np.array(target_chamber_pressure)*np.array(TS_pulse_duration_at_max_power2),xerr=pulse_en_semi_inf_sigma,fmt='+')
	plt.xlabel('Pulse delivered energy [J]')
	plt.ylabel('Energy removed from plasma column [J]')
	plt.pause(0.01)


if True:	#			PLOTS FOR THE PAPER
	fig, ax = plt.subplots( 2,1,figsize=(10, 9), squeeze=False, sharex=True)
	fig.suptitle('Pressure scan\nmagnetic_field %.3gT,target/OES distance %.3gmm,ELM energy %.3gJ\n ' %(magnetic_field[0],target_OES_distance[0],CB_pulse_energy[0]))
	ax[0,0].plot(target_chamber_pressure,delivered_pulse_energy,linewidth=3,color=color[0],label=r'$E_{upstream}$')
	ax[0,0].errorbar(target_chamber_pressure,net_power_removed_plasma_column,yerr=np.array(net_power_removed_plasma_column_sigma),linewidth=3,color=color[1],capsize=5,label=r'$E_{removed \: from \: plasma}$')
	ax[0,0].plot(target_chamber_pressure,max_CX_energy,'--',linewidth=3,color=color[7],label=r'$E_{CX \: max}$')
	ax[0,0].errorbar(target_chamber_pressure,np.array(pulse_en_semi_inf)+np.array(pulse_en_SS),yerr=np.array(pulse_en_semi_inf_sigma),linewidth=3,color=color[8],capsize=5,label=r'$E_{target}$')
	ax[0,0].legend(loc=0, fontsize='small',framealpha=0.5)
	ax[0,0].grid()
	ax[0,0].set_ylabel('Energy [J]')
	ax[1,0].plot(target_chamber_pressure,delivered_pulse_energy,linewidth=3,color=color[0],label=r'$E_{upstream}$')
	ax[1,0].errorbar(target_chamber_pressure,tot_rad_power,yerr=np.array(tot_rad_power_sigma),capsize=5,color=color[2],label=r'$E_{radiated}$')
	# ax[1,0].errorbar(target_chamber_pressure,power_rec_neutral,yerr=np.array(power_rec_neutral_sigma),capsize=5,color=color[3],label=r'$E_{neutral \: from \: recombination}$')
	ax[1,0].errorbar(target_chamber_pressure,power_via_ionisation,yerr=np.array(power_via_ionisation_sigma),capsize=5,color=color[14],label=r'$E_{ionisation \: (only \: potential)}$')
	ax[1,0].errorbar(target_chamber_pressure,power_via_recombination,yerr=np.array(power_via_recombination_sigma),capsize=5,color=color[9],label=r'$E_{recombination \: (only \: potential)}$')
	ax[1,0].errorbar(target_chamber_pressure,net_recombination,yerr=np.array(net_recombination_sigma),capsize=5,color=color[4],label=r'$E_{recombination \: (radiated \: & \: bremm \: - \: heating)}$')
	ax[1,0].legend(loc=0, fontsize='small',framealpha=0.5)
	ax[1,0].grid()
	ax[1,0].set_ylabel('Energy [J]')
	ax[1,0].set_xlabel('Target chamber pressure [Pa]')
	plt.pause(0.001)


	power_rad_mol = np.array(power_rad_mol)
	power_rad_Hm = np.array(power_rad_Hm)
	power_rad_Hm_H2p = np.array(power_rad_Hm_H2p)
	power_rad_Hm_Hp = np.array(power_rad_Hm_Hp)
	power_rad_H2p = np.array(power_rad_H2p)
	power_rad_H2 = np.array(power_rad_H2)
	power_rad_excit = np.array(power_rad_excit)
	power_rad_rec_bremm = np.array(power_rad_rec_bremm)
	plt.figure(figsize=(10, 5));
	temp = power_rad_Hm_H2p + power_rad_Hm_Hp + power_rad_H2p + power_rad_H2 + power_rad_excit + power_rad_rec_bremm
	labels = [r'$E_{radiated \: \: H^- + {H_2}^+  \:→\: H(p) + H_2}$', r'$E_{radiated \: \: H^- + H^+  \:→\: H(p) + H(1)}$' , r'$E_{radiated \: \: {H_2}^+ + e \:→\: H(p) + H^+ + e \:, \:→ \: H(p) + H(1)}$' , r'$E_{radiated \: \: H_2 + e \:→\: H(p) + H(1) + e}$' , r'$E_{direct \: excitation}$' , r'$E_{recombination \: radiated \: \: + \: brems.}$' ]
	plt.stackplot(target_chamber_pressure,power_rad_Hm_H2p/temp,power_rad_Hm_Hp/temp,power_rad_H2p/temp,power_rad_H2/temp,power_rad_excit/temp,power_rad_rec_bremm/temp,labels=labels)
	plt.errorbar(target_chamber_pressure,power_rad_mol/temp,yerr=np.array(power_rad_mol_sigma)/temp,linestyle='--',capsize=5,color=color[6],label=r'$E_{radiated \: \: molecules}$')
	plt.legend(loc='best', fontsize='small')
	plt.ylim(bottom=0,top=1)
	plt.ylabel('fraction of the total radiated energy [au]')
	plt.xlabel('Target chamber pressure [Pa]')
	plt.grid()
	plt.title('Pressure scan\nmagnetic_field %.3gT,target/OES distance %.3gmm,ELM energy %.3gJ\n ' %(magnetic_field[0],target_OES_distance[0],CB_pulse_energy[0]))
	plt.pause(0.001)

else:	# plots not for the paper
	plt.figure(figsize=(10, 4.5))
	plt.plot(target_chamber_pressure,delivered_pulse_energy,linewidth=3,color=color[0],label=r'$E_{upstream}$')
	plt.errorbar(target_chamber_pressure,net_power_removed_plasma_column,yerr=np.array(net_power_removed_plasma_column_sigma),linewidth=3,color=color[1],capsize=5,label=r'$E_{removed \: from \: plasma}$')
	# plt.errorbar(target_chamber_pressure,power_rad_mol,yerr=np.array(power_rad_mol_sigma),linestyle='--',capsize=5,color=color[4],label=r'$E_{radiated \: molecules}$')
	# plt.errorbar(target_chamber_pressure,power_rad_excit,yerr=np.array(power_rad_excit_sigma),linestyle='--',capsize=5,color=color[5],label=r'$E_{excitation}$')
	# plt.errorbar(target_chamber_pressure,power_rad_rec_bremm,yerr=np.array(power_rad_rec_bremm_sigma),linestyle='--',capsize=5,color=color[6],label=r'$E_{radiated \: recombination + bremsstrahlung}$')
	plt.plot(target_chamber_pressure,max_CX_energy,'--',linewidth=3,color=color[7],label=r'$E_{CX \: max}$')
	# plt.plot(target_chamber_pressure,np.array(pulse_en_semi_inf)/np.array(area_of_interest_IR)*np.array(area_equiv_max_static_pressure)*6,linewidth=3,color=color[8],label=r'$E_{target}$')
	# plt.errorbar(target_chamber_pressure,np.array(pulse_en_semi_inf),yerr=np.array(pulse_en_semi_inf_sigma),color=color[8])
	plt.errorbar(target_chamber_pressure,np.array(pulse_en_semi_inf)+np.array(pulse_en_SS),yerr=np.array(pulse_en_semi_inf_sigma),linewidth=3,color=color[8],capsize=5,label=r'$E_{target}$')
	# plt.errorbar(j_specific_target_chamber_pressure,j_specific_pulse_en_semi_inf,yerr=j_specific_pulse_en_semi_inf_sigma,color=color[8],fmt='+')
	# plt.plot(target_chamber_pressure,np.array(net_power_removed_plasma_column) + np.array(max_CX_energy),linestyle='--',color=color[9],label=r'$E_{rem}+E_{CX \: max}$')
	# plt.plot(target_chamber_pressure,np.array(pulse_en_semi_inf)/np.array(area_of_interest_IR)*np.array(area_equiv_max_static_pressure)*6 + np.array(net_power_removed_plasma_column) + np.array(max_CX_energy),linewidth=3,color=color[9],label=r'$6 \cdot E_{target} +$'+'\n'+r'$ E_{rem}+E_{CX \: max}$')
	# plt.plot(target_chamber_pressure,np.array(pulse_en_semi_inf) + np.array(net_power_removed_plasma_column) + np.array(max_CX_energy),linewidth=3,color=color[9],label=r'$E_{target} +$'+'\n'+r'$ E_{rem}+E_{CX \: max}$')
	# plt.legend(loc=2, fontsize='x-small',ncol=2,handleheight=2, labelspacing=0.00005)
	plt.legend(loc=2, fontsize='small')
	plt.grid()
	plt.title('Pressure scan\nmagnetic_field %.3gT,target/OES distance %.3gmm,ELM energy %.3gJ\n ' %(magnetic_field[0],target_OES_distance[0],CB_pulse_energy[0]))
	plt.xlabel('Target chamber pressure [Pa]')
	plt.ylabel('Energy [J]')
	# if merge_ID_target==85:
	# 	plt.xlim(right=target_chamber_pressure[-2])
	plt.pause(0.001)

	plt.figure(figsize=(10, 4.5))
	plt.plot(target_chamber_pressure,delivered_pulse_energy,linewidth=3,color=color[0],label=r'$E_{upstream}$')
	plt.errorbar(target_chamber_pressure,tot_rad_power,yerr=np.array(tot_rad_power_sigma),capsize=5,color=color[2],label=r'$E_{radiated}$')
	plt.errorbar(target_chamber_pressure,power_rec_neutral,yerr=np.array(power_rec_neutral_sigma),capsize=5,color=color[3],label=r'$E_{neutral \: from \: recombination}$')
	plt.errorbar(target_chamber_pressure,power_via_ionisation,yerr=np.array(power_via_ionisation_sigma),capsize=5,color=color[14],label=r'$E_{ionisation \: (only \: potential)}$')
	plt.errorbar(target_chamber_pressure,power_via_recombination,yerr=np.array(power_via_recombination_sigma),capsize=5,color=color[9],label=r'$E_{recombination \: (only \: potential)}$')
	plt.legend(loc=2, fontsize='small')
	plt.grid()
	plt.title('Pressure scan\nmagnetic_field %.3gT,target/OES distance %.3gmm,ELM energy %.3gJ' %(magnetic_field[0],target_OES_distance[0],CB_pulse_energy[0]))
	plt.xlabel('Target chamber pressure [Pa]')
	plt.ylabel('Energy [J]')
	# if merge_ID_target==85:
	# 	plt.xlim(right=target_chamber_pressure[-2])
	plt.pause(0.001)


	plt.figure(figsize=(10, 8))
	plt.plot(target_chamber_pressure,delivered_pulse_energy,linewidth=3,color=color[0],label=r'$E_{upstream}$')
	plt.plot(target_chamber_pressure,max_CX_energy,linewidth=3,color=color[7],label=r'$E_{CX \: max}$')
	plt.errorbar(target_chamber_pressure,np.array(pulse_en_semi_inf)+np.array(pulse_en_SS),yerr=np.array(pulse_en_semi_inf_sigma),linewidth=3,color=color[8],label=r'$E_{target}$')
	plt.errorbar(target_chamber_pressure,net_power_removed_plasma_column,yerr=np.array(net_power_removed_plasma_column_sigma),linewidth=3,color=color[1],capsize=5,label=r'$E_{removed \: from \: plasma}$')
	plt.errorbar(target_chamber_pressure,tot_rad_power,yerr=np.array(tot_rad_power_sigma),capsize=5,color=color[2],label=r'$E_{radiated}$')
	plt.errorbar(target_chamber_pressure,power_rec_neutral,yerr=np.array(power_rec_neutral_sigma),capsize=5,color=color[3],label=r'$E_{neutral \: from \: recombination}$')
	plt.errorbar(target_chamber_pressure,power_rad_mol,yerr=np.array(power_rad_mol_sigma),linestyle='--',capsize=5,color=color[4],label=r'$E_{radiated \: molecules}$')
	plt.errorbar(target_chamber_pressure,power_rad_Hm,yerr=np.array(power_rad_Hm_sigma),linestyle='-.',capsize=5,color=color[9],label=r'$E_{radiated \: H^- + e \: H^- + H^+}$')
	plt.errorbar(target_chamber_pressure,power_rad_H2p,yerr=np.array(power_rad_H2p_sigma),linestyle='-.',capsize=5,color=color[10],label=r'$E_{radiated \: {H_2}^+ + e}$')
	plt.errorbar(target_chamber_pressure,power_rad_H2,yerr=np.array(power_rad_H2_sigma),linestyle='-.',capsize=5,color=color[11],label=r'$E_{radiated \: H_2 + e}$')
	# plt.errorbar(target_chamber_pressure,power_rad_Hm_H2p,yerr=np.array(power_rad_Hm_H2p_sigma),linestyle='-.',capsize=5,color=color[12],label=r'$E_{radiated \: from \: H^- + {H_2}^+}$')
	# plt.errorbar(target_chamber_pressure,power_rad_Hm_Hp,yerr=np.array(power_rad_Hm_Hp_sigma),linestyle='-.',capsize=5,color=color[13],label=r'$E_{radiated \: from \: $H^- + H^+}$')
	plt.errorbar(target_chamber_pressure,power_rad_excit,yerr=np.array(power_rad_excit_sigma),linestyle='--',capsize=5,color=color[5],label=r'$E_{excitation}$')
	plt.errorbar(target_chamber_pressure,power_rad_rec_bremm,yerr=np.array(power_rad_rec_bremm_sigma),linestyle='--',capsize=5,color=color[6],label=r'$E_{recomb. \: radiated \: + \: brems.}$')
	# plt.plot(target_chamber_pressure,np.array(pulse_en_semi_inf)/np.array(area_of_interest_IR)*np.array(area_equiv_max_static_pressure)*6,linewidth=3,color=color[8],label=r'$E_{target}$')
	# plt.errorbar(target_chamber_pressure,np.array(pulse_en_semi_inf),yerr=np.array(pulse_en_semi_inf_sigma),color=color[8])
	# plt.errorbar(j_specific_target_chamber_pressure,j_specific_pulse_en_semi_inf,yerr=j_specific_pulse_en_semi_inf_sigma,color=color[8],fmt='+')
	# plt.plot(target_chamber_pressure,np.array(net_power_removed_plasma_column) + np.array(max_CX_energy),linestyle='--',color=color[9],label=r'$E_{rem}+E_{CX \: max}$')
	# plt.plot(target_chamber_pressure,np.array(pulse_en_semi_inf)/np.array(area_of_interest_IR)*np.array(area_equiv_max_static_pressure)*6 + np.array(net_power_removed_plasma_column) + np.array(max_CX_energy),linewidth=3,color=color[9],label=r'$6 \cdot E_{target} +$'+'\n'+r'$ E_{rem}+E_{CX \: max}$')
	# plt.plot(target_chamber_pressure,np.array(pulse_en_semi_inf) + np.array(net_power_removed_plasma_column) + np.array(max_CX_energy),linewidth=3,color=color[9],label=r'$E_{target} +$'+'\n'+r'$ E_{rem}+E_{CX \: max}$')
	# plt.legend(loc=2, fontsize='x-small',ncol=2,handleheight=2, labelspacing=0.00005)
	plt.errorbar(target_chamber_pressure,power_via_ionisation,yerr=np.array(power_via_ionisation_sigma),capsize=5,color=color[14],label=r'$E_{ionisation \: (only \: potential)}$')
	plt.errorbar(target_chamber_pressure,power_via_recombination,yerr=np.array(power_via_recombination_sigma),capsize=5,color=color[3],label=r'$E_{recombination \: (only \: potential)}$')
	plt.legend(loc=2, fontsize='x-small')
	plt.grid()
	plt.title('Pressure scan\nmagnetic_field %.3gT,target/OES distance %.3gmm,ELM energy %.3gJ\n ' %(magnetic_field[0],target_OES_distance[0],CB_pulse_energy[0]))
	plt.xlabel('Target chamber pressure [Pa]')
	plt.ylabel('Energy [J]')
	# plt.semilogy()
	# plt.ylim(bottom=np.max(delivered_pulse_energy)/100)
	# if merge_ID_target==85:
	# 	plt.xlim(right=target_chamber_pressure[-2])
	plt.pause(0.001)

	plt.figure(figsize=(10, 8))
	plt.plot(target_chamber_pressure,delivered_pulse_energy,linewidth=3,color=color[0],label=r'$E_{upstream}$')
	plt.plot(target_chamber_pressure,max_CX_energy,linewidth=3,color=color[7],label=r'$E_{CX \: max}$')
	plt.errorbar(target_chamber_pressure,np.array(pulse_en_semi_inf)+np.array(pulse_en_SS),yerr=np.array(pulse_en_semi_inf_sigma),linewidth=3,color=color[8],label=r'$E_{target}$')
	plt.errorbar(target_chamber_pressure,net_power_removed_plasma_column,yerr=np.array(net_power_removed_plasma_column_sigma),linewidth=3,color=color[1],capsize=5,label=r'$E_{removed \: from \: plasma}$')
	plt.errorbar(target_chamber_pressure,tot_rad_power,yerr=np.array(tot_rad_power_sigma),capsize=5,color=color[2],label=r'$E_{radiated}$')
	plt.errorbar(target_chamber_pressure,power_rec_neutral,yerr=np.array(power_rec_neutral_sigma),capsize=5,color=color[3],label=r'$E_{neutral \: from \: recombination}$')
	# plt.errorbar(target_chamber_pressure,power_rad_mol,yerr=np.array(power_rad_mol_sigma),linestyle='--',capsize=5,color=color[4],label=r'$E_{radiated \: molecules}$')
	# plt.errorbar(target_chamber_pressure,power_rad_Hm,yerr=np.array(power_rad_Hm_sigma),linestyle='-.',capsize=5,color=color[9],label=r'$E_{radiated \: H^- + e \: H^- + H^+}$')
	# plt.errorbar(target_chamber_pressure,power_rad_H2p,yerr=np.array(power_rad_H2p_sigma),linestyle='-.',capsize=5,color=color[10],label=r'$E_{radiated \: {H_2}^+ + e}$')
	# plt.errorbar(target_chamber_pressure,power_rad_H2,yerr=np.array(power_rad_H2_sigma),linestyle='-.',capsize=5,color=color[11],label=r'$E_{radiated \: H_2 + e}$')
	# # plt.errorbar(target_chamber_pressure,power_rad_Hm_H2p,yerr=np.array(power_rad_Hm_H2p_sigma),linestyle='-.',capsize=5,color=color[12],label=r'$E_{radiated \: from \: H^- + {H_2}^+}$')
	# # plt.errorbar(target_chamber_pressure,power_rad_Hm_Hp,yerr=np.array(power_rad_Hm_Hp_sigma),linestyle='-.',capsize=5,color=color[13],label=r'$E_{radiated \: from \: $H^- + H^+}$')
	# plt.errorbar(target_chamber_pressure,power_rad_excit,yerr=np.array(power_rad_excit_sigma),linestyle='--',capsize=5,color=color[5],label=r'$E_{excitation}$')
	# plt.errorbar(target_chamber_pressure,power_rad_rec_bremm,yerr=np.array(power_rad_rec_bremm_sigma),linestyle='--',capsize=5,color=color[6],label=r'$E_{recomb. \: radiated \: + \: brems.}$')
	# plt.plot(target_chamber_pressure,np.array(pulse_en_semi_inf)/np.array(area_of_interest_IR)*np.array(area_equiv_max_static_pressure)*6,linewidth=3,color=color[8],label=r'$E_{target}$')
	# plt.errorbar(target_chamber_pressure,np.array(pulse_en_semi_inf),yerr=np.array(pulse_en_semi_inf_sigma),color=color[8])
	# plt.errorbar(j_specific_target_chamber_pressure,j_specific_pulse_en_semi_inf,yerr=j_specific_pulse_en_semi_inf_sigma,color=color[8],fmt='+')
	# plt.plot(target_chamber_pressure,np.array(net_power_removed_plasma_column) + np.array(max_CX_energy),linestyle='--',color=color[9],label=r'$E_{rem}+E_{CX \: max}$')
	# plt.plot(target_chamber_pressure,np.array(pulse_en_semi_inf)/np.array(area_of_interest_IR)*np.array(area_equiv_max_static_pressure)*6 + np.array(net_power_removed_plasma_column) + np.array(max_CX_energy),linewidth=3,color=color[9],label=r'$6 \cdot E_{target} +$'+'\n'+r'$ E_{rem}+E_{CX \: max}$')
	# plt.plot(target_chamber_pressure,np.array(pulse_en_semi_inf) + np.array(net_power_removed_plasma_column) + np.array(max_CX_energy),linewidth=3,color=color[9],label=r'$E_{target} +$'+'\n'+r'$ E_{rem}+E_{CX \: max}$')
	# plt.legend(loc=2, fontsize='x-small',ncol=2,handleheight=2, labelspacing=0.00005)
	plt.errorbar(target_chamber_pressure,power_via_ionisation,yerr=np.array(power_via_ionisation_sigma),capsize=5,color=color[14],label=r'$E_{ionisation \: (only \: potential)}$')
	plt.errorbar(target_chamber_pressure,power_via_recombination,yerr=np.array(power_via_recombination_sigma),capsize=5,color=color[9],label=r'$E_{recombination \: (only \: potential)}$')
	plt.legend(loc=2, fontsize='x-small')
	plt.grid()
	plt.title('Pressure scan\nmagnetic_field %.3gT,target/OES distance %.3gmm,ELM energy %.3gJ\n ' %(magnetic_field[0],target_OES_distance[0],CB_pulse_energy[0]))
	plt.xlabel('Target chamber pressure [Pa]')
	plt.ylabel('Energy [J]')
	# plt.semilogy()
	# plt.ylim(bottom=np.max(delivered_pulse_energy)/100)
	# if merge_ID_target==85:
	# 	plt.xlim(right=target_chamber_pressure[-2])
	plt.pause(0.001)



	plt.figure(figsize=(10, 8))
	# plt.plot(target_chamber_pressure,delivered_pulse_energy,linewidth=3,color=color[0],label=r'$E_{upstream}$')
	# plt.plot(target_chamber_pressure,max_CX_energy,linewidth=3,color=color[7],label=r'$E_{CX \: max}$')
	# plt.errorbar(target_chamber_pressure,np.array(pulse_en_semi_inf)+np.array(pulse_en_SS),yerr=np.array(pulse_en_semi_inf_sigma),linewidth=3,color=color[8],label=r'$E_{target}$')
	# plt.errorbar(target_chamber_pressure,net_power_removed_plasma_column,yerr=np.array(net_power_removed_plasma_column_sigma),linewidth=3,color=color[1],capsize=5,label=r'$E_{removed \: from \: plasma}$')
	# plt.errorbar(target_chamber_pressure,tot_rad_power,yerr=np.array(tot_rad_power_sigma),capsize=5,color=color[2],label=r'$E_{radiated}$')
	# plt.errorbar(target_chamber_pressure,power_rec_neutral,yerr=np.array(power_rec_neutral_sigma),capsize=5,color=color[3],label=r'$E_{neutral \: from \: recombination}$')
	plt.errorbar(target_chamber_pressure,power_rad_mol,yerr=np.array(power_rad_mol_sigma),linestyle='--',capsize=5,color=color[4],label=r'$E_{radiated \: molecules}$')
	plt.errorbar(target_chamber_pressure,power_rad_Hm,yerr=np.array(power_rad_Hm_sigma),linestyle='-.',capsize=5,color=color[9],label=r'$E_{radiated \: H^- + {H_2}^+ \:,\: H^- + H^+}$')
	plt.errorbar(target_chamber_pressure,power_rad_H2p,yerr=np.array(power_rad_H2p_sigma),linestyle='-.',capsize=5,color=color[10],label=r'$E_{radiated \: {H_2}^+ + e}$')
	plt.errorbar(target_chamber_pressure,power_rad_H2,yerr=np.array(power_rad_H2_sigma),linestyle='-.',capsize=5,color=color[11],label=r'$E_{radiated \: H_2 + e}$')
	# plt.errorbar(target_chamber_pressure,power_rad_Hm_H2p,yerr=np.array(power_rad_Hm_H2p_sigma),linestyle='-.',capsize=5,color=color[12],label=r'$E_{radiated \: from \: H^- + {H_2}^+}$')
	# plt.errorbar(target_chamber_pressure,power_rad_Hm_Hp,yerr=np.array(power_rad_Hm_Hp_sigma),linestyle='-.',capsize=5,color=color[13],label=r'$E_{radiated \: from \: $H^- + H^+}$')
	plt.errorbar(target_chamber_pressure,power_rad_excit,yerr=np.array(power_rad_excit_sigma),linestyle='--',capsize=5,color=color[5],label=r'$E_{excitation}$')
	plt.errorbar(target_chamber_pressure,power_rad_rec_bremm,yerr=np.array(power_rad_rec_bremm_sigma),linestyle='--',capsize=5,color=color[6],label=r'$E_{recomb. \: radiated \: + \: brems.}$')
	# plt.plot(target_chamber_pressure,np.array(pulse_en_semi_inf)/np.array(area_of_interest_IR)*np.array(area_equiv_max_static_pressure)*6,linewidth=3,color=color[8],label=r'$E_{target}$')
	# plt.errorbar(target_chamber_pressure,np.array(pulse_en_semi_inf),yerr=np.array(pulse_en_semi_inf_sigma),color=color[8])
	# plt.errorbar(j_specific_target_chamber_pressure,j_specific_pulse_en_semi_inf,yerr=j_specific_pulse_en_semi_inf_sigma,color=color[8],fmt='+')
	# plt.plot(target_chamber_pressure,np.array(net_power_removed_plasma_column) + np.array(max_CX_energy),linestyle='--',color=color[9],label=r'$E_{rem}+E_{CX \: max}$')
	# plt.plot(target_chamber_pressure,np.array(pulse_en_semi_inf)/np.array(area_of_interest_IR)*np.array(area_equiv_max_static_pressure)*6 + np.array(net_power_removed_plasma_column) + np.array(max_CX_energy),linewidth=3,color=color[9],label=r'$6 \cdot E_{target} +$'+'\n'+r'$ E_{rem}+E_{CX \: max}$')
	# plt.plot(target_chamber_pressure,np.array(pulse_en_semi_inf) + np.array(net_power_removed_plasma_column) + np.array(max_CX_energy),linewidth=3,color=color[9],label=r'$E_{target} +$'+'\n'+r'$ E_{rem}+E_{CX \: max}$')
	# plt.legend(loc=2, fontsize='x-small',ncol=2,handleheight=2, labelspacing=0.00005)
	# plt.errorbar(target_chamber_pressure,power_via_ionisation,yerr=np.array(power_via_ionisation_sigma),capsize=5,color=color[14],label=r'$E_{ionisation \: (only \: potential)}$')
	# plt.errorbar(target_chamber_pressure,power_via_recombination,yerr=np.array(power_via_recombination_sigma),capsize=5,color=color[3],label=r'$E_{recombination \: (only \: potential)}$')
	plt.legend(loc=2, fontsize='x-small')
	plt.grid()
	plt.title('Pressure scan\nmagnetic_field %.3gT,target/OES distance %.3gmm,ELM energy %.3gJ' %(magnetic_field[0],target_OES_distance[0],CB_pulse_energy[0]))
	plt.xlabel('Target chamber pressure [Pa]')
	plt.ylabel('Energy [J]')
	# plt.semilogy()
	# plt.ylim(bottom=np.max(delivered_pulse_energy)/100)
	# if merge_ID_target==85:
	# 	plt.xlim(right=target_chamber_pressure[-2])
	plt.pause(0.001)


	plt.figure(figsize=(10, 5));
	temp = (np.array(power_rad_mol) + np.array(power_rad_excit) + np.array(power_rad_rec_bremm))/100
	labels = ['H excitation from molecular reactions', 'H direct excitation', 'H excitation from recombination and bremsstrahlung']
	plt.stackplot(target_chamber_pressure,np.array(power_rad_mol)/temp,np.array(power_rad_excit)/temp,np.array(power_rad_rec_bremm)/temp,labels=labels)
	# plt.semilogy()
	# plt.ylim(bottom=1e0,top=1e6)
	plt.legend(loc=4, fontsize='small')
	# plt.ylim(bottom=0,top=max(np.max(most_likely_power_via_ionisation_r_up),np.max(power_pulse_shape_crop)))
	# plt.ylim(bottom=1e-1)
	plt.xlabel('Target chamber pressure [Pa]')
	plt.ylabel('Fraction of the radiated power [%]')
	plt.title('Pressure scan\nmagnetic_field %.3gT,target/OES distance %.3gmm,ELM energy %.3gJ' %(magnetic_field[0],target_OES_distance[0],CB_pulse_energy[0]))
	plt.grid(axis='y')
	if merge_ID_target==85:
		plt.xlim(right=target_chamber_pressure[-2])
	plt.pause(0.001)

start = np.nanmax([list[0] for list in j_twoD_peak_evolution_time_averaged])
stop = np.nanmin([list[-1] for list in j_twoD_peak_evolution_time_averaged])
if False:
	j_twoD_peak_evolution_temp_averaged_delta = [list2[np.abs(np.array(list1)-start).argmin():np.abs(np.array(list1)-stop).argmin()] for (list1,list2) in np.array([j_twoD_peak_evolution_time_averaged,j_twoD_peak_evolution_temp_averaged_delta]).T]
	j_twoD_peak_evolution_temp_averaged = [list2[np.abs(np.array(list1)-start).argmin():np.abs(np.array(list1)-stop).argmin()] for (list1,list2) in np.array([j_twoD_peak_evolution_time_averaged,j_twoD_peak_evolution_temp_averaged]).T]
	j_twoD_peak_evolution_time_averaged = [list1[np.abs(np.array(list1)-start).argmin():np.abs(np.array(list1)-stop).argmin()] for list1 in j_twoD_peak_evolution_time_averaged]
	j_twoD_peak_evolution_temp_averaged_delta = np.array([(list2 if len(list1)>0 else np.ones_like(j_twoD_peak_evolution_temp_averaged_delta[0])*np.nan) for (list1,list2) in np.array([j_twoD_peak_evolution_time_averaged,j_twoD_peak_evolution_temp_averaged_delta]).T])
	j_twoD_peak_evolution_temp_averaged = np.array([(list2 if len(list1)>0 else np.ones_like(j_twoD_peak_evolution_temp_averaged[0])*np.nan) for (list1,list2) in np.array([j_twoD_peak_evolution_time_averaged,j_twoD_peak_evolution_temp_averaged]).T])
	j_twoD_peak_evolution_time_averaged = np.array([(list1 if len(list1)>0 else np.ones_like(j_twoD_peak_evolution_time_averaged[0])*np.nan) for list1 in j_twoD_peak_evolution_time_averaged])

	plt.figure(figsize=(10, 5));
	for index,merge_ID_target in enumerate(merge_ID_target_multipulse):
		check = np.array(j_specific_merge_ID_target) == merge_ID_target
		temp_dt = np.nanmean(j_twoD_peak_evolution_temp_averaged_delta[check],axis=0)
		# temp_full = np.nanmean(j_twoD_peak_evolution_temp_averaged_delta[check],axis=0)
		temp_time = np.nanmean(j_twoD_peak_evolution_time_averaged[check],axis=0)
		total_number_pixels = temp_dt.shape[1]*temp_dt.shape[2]
		temp = np.sum(temp_dt.T>temp_dt.max(axis=(1,2))*0.2,axis=(0,1))
		# plt.plot(temp_time,temp/total_number_pixels,':',color=color[index])
		temp = np.sum(temp_dt.T>temp_dt.max(axis=(1,2))*0.5,axis=(0,1))
		plt.plot(temp_time,temp/total_number_pixels,'-',color=color[index],label='pressure %.3gPa' %(target_chamber_pressure_2[index]))
		temp = np.sum(temp_dt.T>temp_dt.max(axis=(1,2))*0.8,axis=(0,1))
		# plt.plot(temp_time,temp/total_number_pixels,'--',color=color[index])
		# plt.pause(1)
	plt.legend(loc='best', fontsize='small')
	plt.xlabel('time after the peak [s]')
	plt.ylabel('fraction of pixels (dt) [au]')
	plt.title('Pressure scan\nmagnetic_field %.3gT,target/OES distance %.3gmm,ELM energy %.3gJ' %(magnetic_field[0],target_OES_distance[0],CB_pulse_energy[0]))
	plt.grid()
	plt.pause(0.001)

	plt.figure(figsize=(10, 5));
	for index,merge_ID_target in enumerate(merge_ID_target_multipulse):
		check = np.array(j_specific_merge_ID_target) == merge_ID_target
		# temp_dt = np.nanmean(j_twoD_peak_evolution_temp_averaged_delta[check],axis=0)
		temp_full = np.nanmean(j_twoD_peak_evolution_temp_averaged[check],axis=0)
		temp_time = np.nanmean(j_twoD_peak_evolution_time_averaged[check],axis=0)
		total_number_pixels = temp_full.shape[1]*temp_full.shape[2]
		temp = np.sum(temp_full.T>temp_full.max(axis=(1,2))*0.2,axis=(0,1))
		# plt.plot(temp_time,temp/total_number_pixels,':',color=color[index])
		temp = np.sum(temp_full.T>temp_full.max(axis=(1,2))*0.5,axis=(0,1))
		# plt.plot(temp_time,temp/total_number_pixels,'-',color=color[index],label='pressure %.3gPa' %(target_chamber_pressure_2[index]))
		temp = np.sum(temp_full.T>temp_full.max(axis=(1,2))*0.8,axis=(0,1))
		plt.plot(temp_time,temp/total_number_pixels,'--',color=color[index])
		# plt.pause(1)
	plt.legend(loc='best', fontsize='small')
	plt.xlabel('time after the peak [s]')
	plt.ylabel('fraction of pixels (full) [au]')
	plt.title('Pressure scan\nmagnetic_field %.3gT,target/OES distance %.3gmm,ELM energy %.3gJ' %(magnetic_field[0],target_OES_distance[0],CB_pulse_energy[0]))
	plt.grid()
	plt.pause(0.001)

else:
	j_twoD_peak_evolution_temp_averaged_delta = [x[7] for x in j_collect_shape_information]
	j_twoD_peak_evolution_temp_averaged = [x[11] for x in j_collect_shape_information]
	j_twoD_peak_evolution_temp_averaged_delta = [list2[np.abs(np.array(list1)-start).argmin():np.abs(np.array(list1)-stop).argmin()] for (list1,list2) in np.array([j_twoD_peak_evolution_time_averaged,j_twoD_peak_evolution_temp_averaged_delta]).T]
	j_twoD_peak_evolution_temp_averaged = [list2[np.abs(np.array(list1)-start).argmin():np.abs(np.array(list1)-stop).argmin()] for (list1,list2) in np.array([j_twoD_peak_evolution_time_averaged,j_twoD_peak_evolution_temp_averaged]).T]
	j_twoD_peak_evolution_time_averaged = [list1[np.abs(np.array(list1)-start).argmin():np.abs(np.array(list1)-stop).argmin()] for list1 in j_twoD_peak_evolution_time_averaged]
	j_twoD_peak_evolution_temp_averaged_delta = np.array([(list2 if len(list1)>0 else np.ones_like(j_twoD_peak_evolution_temp_averaged_delta[0])*np.nan) for (list1,list2) in np.array([j_twoD_peak_evolution_time_averaged,j_twoD_peak_evolution_temp_averaged_delta]).T])
	j_twoD_peak_evolution_temp_averaged = np.array([(list2 if len(list1)>0 else np.ones_like(j_twoD_peak_evolution_temp_averaged[0])*np.nan) for (list1,list2) in np.array([j_twoD_peak_evolution_time_averaged,j_twoD_peak_evolution_temp_averaged]).T])
	j_twoD_peak_evolution_time_averaged = np.array([(list1 if len(list1)>0 else np.ones_like(j_twoD_peak_evolution_time_averaged[0])*np.nan) for list1 in j_twoD_peak_evolution_time_averaged])

	plt.figure(figsize=(10, 5));
	for index,merge_ID_target in enumerate(merge_ID_target_multipulse):
		check = np.arange(len(j_specific_merge_ID_target))[np.array(j_specific_merge_ID_target) == merge_ID_target]
		for index_3,index_2 in enumerate(check):
			plt.plot(j_twoD_peak_evolution_time_averaged[index_2],-j_twoD_peak_evolution_temp_averaged_delta[index_2],'--',color=color[index])
			if index_3==0:
				plt.plot(j_twoD_peak_evolution_time_averaged[index_2],j_twoD_peak_evolution_temp_averaged[index_2],'-',color=color[index],label='pressure %.3gPa' %(target_chamber_pressure_2[index]))
			else:
				plt.plot(j_twoD_peak_evolution_time_averaged[index_2],j_twoD_peak_evolution_temp_averaged[index_2],'-',color=color[index])
	plt.legend(loc='best', fontsize='small')
	plt.xlabel('time after the peak [s]')
	plt.ylabel('fraction of pixels [au]')
	plt.title('Pressure scan\nmagnetic_field %.3gT,target/OES distance %.3gmm,ELM energy %.3gJ' %(magnetic_field[0],target_OES_distance[0],CB_pulse_energy[0]) + '\nArea with '+r'$(T-T_{amb})>(T-T_{amb})_{max} * 0.8$'+' (-) and '+r'$T-T_{ss}>(T-T_{ss})_{max} * 0.5$'+' (--)')
	plt.grid()
	plt.pause(0.001)

	fig, ax = plt.subplots( 2,1,figsize=(20, 20), squeeze=False)
	# plt.figure(figsize=(10, 5));
	for index,merge_ID_target in enumerate(merge_ID_target_multipulse):
		check = np.array(j_specific_merge_ID_target) == merge_ID_target
		temp_dt = np.nanmean(j_twoD_peak_evolution_temp_averaged_delta[check],axis=0)
		temp_full = np.nanmean(j_twoD_peak_evolution_temp_averaged[check],axis=0)
		temp_time = np.nanmean(j_twoD_peak_evolution_time_averaged[check],axis=0)
		ax[0,0].plot(temp_time,temp_dt,'-',color=color[index])
		ax[1,0].plot(temp_time,temp_full,'-',color=color[index],label='pressure %.3gPa' %(target_chamber_pressure_2[index]))
	plt.legend(loc='best', fontsize='small')
	ax[1,0].set_xlabel('time after the peak [s]')
	ax[0,0].set_ylabel('fraction of pixels (dt) [au]')
	ax[1,0].set_ylabel('fraction of pixels (full) [au]')
	fig.suptitle('Pressure scan\nmagnetic_field %.3gT,target/OES distance %.3gmm,ELM energy %.3gJ' %(magnetic_field[0],target_OES_distance[0],CB_pulse_energy[0]) + '\nArea with '+r'$(T-T_{amb})>(T-T_{amb})_{max} * 0.8$'+' (-) and '+r'$T-T_{ss}>(T-T_{ss})_{max} * 0.5$'+' (--)')
	ax[0,0].grid()
	ax[1,0].grid()
	# ax[0,0].set_yscale('log')
	ax[1,0].set_yscale('log')
	plt.pause(0.001)


plt.figure(figsize=(10, 5))
for index in range(len(target_chamber_pressure)):
	# plt.plot(radial_average_brightness_OES_location_1ms_int_time[index]/np.max(radial_average_brightness_OES_location_1ms_int_time),color=color[index],label='%.3gPa' %(target_chamber_pressure_2[index]))
	plt.plot(radial_average_brightness_1ms_int_time[index]/np.max(radial_average_brightness_1ms_int_time),':',color=color[index])
	plt.plot(radial_average_brightness_bayesian[index]/np.max(radial_average_brightness_bayesian),'--',color=color[index])
	# plt.plot(radial_average_brightness_bayesian_long[index]/np.max(radial_average_brightness_bayesian_long),'-.',color=color[index])
plt.legend()
plt.legend(loc='best', fontsize='small')
plt.pause(0.001)

plt.figure(figsize=(10, 5))
for index in range(len(target_chamber_pressure_2)):
	# plt.plot(time_source_power[index],power_pulse_shape[index],color=color[index],label='%.3gPa' %(target_chamber_pressure_2[index]))
	plt.plot(np.arange(len(power_pulse_shape[index]))*time_resolution,power_pulse_shape[index],color=color[index],label='%.3gPa' %(target_chamber_pressure_2[index]))
plt.legend()
plt.legend(loc='best', fontsize='small')
plt.xlim(left=-0.1,right=1.2)
plt.ylabel('power-power SS [W]')
plt.xlabel('time from the beginning of the pulse [ms]')
plt.pause(0.001)

record=[]
record_interp=[]
plt.figure(figsize=(10, 5))
for index in range(len(target_chamber_pressure)):
	temp=cp.deepcopy(energy_flow[index])/np.median(np.diff(TS_time[index]))*1e3
	# temp[temp>0]-=np.mean(energy_flow[index][TS_time[index]<=0])/np.median(np.diff(TS_time[index]))*1e3
	temp[temp<1e-6]=0
	temp[(temp>0).argmax()-1]=0.1
	record.append(np.interp(10e3,temp[temp>0][:(np.diff(temp[temp>0])<0).argmax()],TS_time[index][temp>0][:(np.diff(temp[temp>0])<0).argmax()]))
	plt.plot(TS_time[index],temp,color=color[index],label='%.3gPa' %(target_chamber_pressure[index]))
	temp=cp.deepcopy(energy_flow_interp[index])/np.median(np.diff(TS_time[index]))*1e3
	# temp[temp>0]-=np.mean(energy_flow_interp[index][TS_time_interp[index]<=0])/np.median(np.diff(TS_time_interp[index]))*1e3
	temp[temp<1e-6]=0
	temp[(temp>0).argmax()-1]=0.1
	record_interp.append(np.interp(10e3,temp[temp>0][:(np.diff(temp[temp>0])<0).argmax()],TS_time_interp[index][temp>0][:(np.diff(temp[temp>0])<0).argmax()]))
	plt.plot(TS_time_interp[index],temp,'--',color=color[index])
plt.legend()
plt.legend(loc='best', fontsize='small')
plt.title('en=(1/2m*cs^2 + 5kTe +diss+ioniz)*ne*cs*1')
plt.ylabel('power-power SS [W]')
plt.xlabel('time from the beginning of the pulse [ms]')
plt.xlim(left=0,right=1.2)
plt.grid()
plt.pause(0.001)

plt.figure(figsize=(10, 5))
plt.plot(target_chamber_pressure,record)
plt.plot(target_chamber_pressure,record_interp,'--')
plt.title('-=untreated TS, --=interpolated TS')
plt.ylabel('time delay to reach 200W [ms]')
plt.xlabel('Target chamber pressure [Pa]')
plt.grid()
plt.pause(0.001)

record=[]
record_interp=[]
plt.figure(figsize=(10, 5))
for index in range(len(target_chamber_pressure)):
	temp=cp.deepcopy(energy_flow[index])
	temp[temp>0]-=np.mean(energy_flow[index][TS_time[index]<=0])
	temp=np.cumsum(temp)
	record.append(np.interp(temp.max()/50,temp,TS_time[index]))
	plt.plot(TS_time[index],temp,color=color[index],label='%.3gPa' %(target_chamber_pressure[index]))
	temp=cp.deepcopy(energy_flow_interp[index])
	temp[temp>0]-=np.mean(energy_flow_interp[index][TS_time_interp[index]<=0])
	temp=np.cumsum(temp)
	record_interp.append(np.interp(temp.max()/50,temp,TS_time_interp[index]))
	plt.plot(TS_time_interp[index],temp,'--',color=color[index])
	print(temp.max()/energy_flow_interp[index].max()*np.median(np.diff(TS_time_interp[index]))*1e-3)
plt.legend()
plt.legend(loc='best', fontsize='small')
plt.ylabel('cumulative\npower-power SS [J]')
plt.xlabel('time from the beginning of the pulse [ms]')
plt.xlim(left=0,right=1.2)
plt.grid()
plt.pause(0.001)

plt.figure(figsize=(10, 5))
plt.plot(target_chamber_pressure,record)
plt.plot(target_chamber_pressure,record_interp,'--')
plt.title('-=untreated TS, --=interpolated TS')
plt.ylabel('time delay to 1/50 of cumulative energy [ms]')
plt.xlabel('Target chamber pressure [Pa]')
plt.grid()
plt.pause(0.001)

# plt.figure()
# plt.plot(target_chamber_pressure,max_average_static_pressure)
# plt.title('Pressure scan magnetic_field %.3gT,target/OES distance %.3gmm,ELM pulse voltage %.3gV' %(magnetic_field[0],target_OES_distance[0],(2*CB_pulse_energy[0]/150e-6)**0.5))
# plt.grid()
# plt.xlabel('Target chamber pressure [Pa]')
# plt.ylabel('max averaged plasma static pressure [Pa]')
# plt.pause(0.001)


# Here I want to parametrise the ELMs to compare them with tokamaks
color = ['b', 'r', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise', 'teal', 'olive']

merge_ID_target_multipulse_all = [[99,98,96,97],[95,89,88,87,86,85,90,91,92],[75,76,77,78,79,80,81,82,83,101],[66,67,68,69,70,74]]
# merge_ID_target_multipulse_all = [[99,98,96,97],[95,89,88,87,86,85,90,91,92],[75,76,77,78,79,80,81,82,83,101]]
# merge_ID_target_multipulse = [99,98,96,97]
fig, ax = plt.subplots( 3,1,figsize=(10, 12), squeeze=False, sharex=True)
fig4, ax4 = plt.subplots( 2,1,figsize=(10, 9), squeeze=False, sharex=True)
fig2, ax2 = plt.subplots( 1,1,figsize=(10, 9), squeeze=False)
fig3, ax3 = plt.subplots( 1,1,figsize=(10, 9), squeeze=False)

for index,merge_ID_target_multipulse in enumerate(merge_ID_target_multipulse_all):
	j_specific_pulse_en_semi_inf = []
	j_specific_pulse_en_semi_inf_sigma = []
	j_specific_pulse_en_semi_inf2 = []
	j_specific_pulse_en_semi_inf_sigma2 = []
	j_specific_energy_density = []
	j_specific_energy_density_sigma = []
	j_TS_pulse_duration_at_max_power = []
	TS_pulse_duration_at_max_power = []
	TS_pulse_duration_at_max_power2 = []
	j_specific_area_of_interest_IR = []
	j_specific_area_of_interest_IR_sigma = []
	magnetic_field = []
	target_chamber_pressure_2 = []
	pulse_energy_density2 = []
	pulse_energy = []
	j_specific_target_chamber_pressure = []
	j_specific_area_ratio_IR = []
	j_specific_area_ratio_IR_sigma = []

	for merge_ID_target in merge_ID_target_multipulse:
		print('Looking at '+str(merge_ID_target))
		all_j=find_index_of_file(merge_ID_target,df_settings,df_log,only_OES=True)

		if merge_ID_target != 88:
			temp = energy_flow_from_TS_at_sound_speed(merge_ID_target,interpolated_TS=True)
			temp[1][temp[1]>0]-=np.mean(temp[1][temp[0]<=0])
			temp1=np.cumsum(temp[1])
			TS_pulse_duration_at_max_power.append(temp1.max()/np.max(temp[1])*np.median(np.diff(temp[0]))*1e-3)
		else:
			TS_pulse_duration_at_max_power.append(0.4e-3)

		temp1 = []
		temp2 = []
		temp3 = []
		temp4 = []

		for j in all_j:
			IR_trace, = df_log.loc[j,['IR_trace']]
			magnetic_field.append(df_log.loc[j,['B']][0])
			if isinstance(IR_trace,str):
				# print(j)
				path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)
				full_saved_file_dict = dict(np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz'))
				j_specific_pulse_en_semi_inf.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_energy'])
				temp4.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_energy'])
				j_specific_pulse_en_semi_inf_sigma.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_energy_sigma'])
				j_specific_energy_density.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_energy_density'])
				temp2.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_energy_density'])
				j_specific_energy_density_sigma.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_energy_density_sigma'])
				j_TS_pulse_duration_at_max_power.append(TS_pulse_duration_at_max_power[-1])
				temp3.append(TS_pulse_duration_at_max_power[-1])
				j_specific_area_of_interest_IR.append(full_saved_file_dict['average_pulse_fitted_data'].all()['area_of_interest'])
				j_specific_area_of_interest_IR_sigma.append(full_saved_file_dict['average_pulse_fitted_data'].all()['area_of_interest_sigma'])
				if not np.isnan(full_saved_file_dict['average_pulse_SS'].all()['area_of_interest']):
					if (full_saved_file_dict['average_pulse_fitted_data'].all()['area_of_interest'] / full_saved_file_dict['average_pulse_SS'].all()['area_of_interest'] >0.5):
						j_specific_area_ratio_IR.append(full_saved_file_dict['average_pulse_fitted_data'].all()['area_of_interest'] / full_saved_file_dict['average_pulse_SS'].all()['area_of_interest'])
						j_specific_area_ratio_IR_sigma.append( full_saved_file_dict['average_pulse_fitted_data'].all()['area_of_interest'] / full_saved_file_dict['average_pulse_SS'].all()['area_of_interest'] *( (full_saved_file_dict['average_pulse_fitted_data'].all()['area_of_interest_sigma']/full_saved_file_dict['average_pulse_fitted_data'].all()['area_of_interest'])**2 + (full_saved_file_dict['average_pulse_SS'].all()['area_of_interest_sigma']/full_saved_file_dict['average_pulse_SS'].all()['area_of_interest'])**2 )**0.5)
						j_specific_pulse_en_semi_inf2.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_energy'])
						j_specific_pulse_en_semi_inf_sigma2.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_energy_sigma'])
				j_specific_target_chamber_pressure.append(df_log.loc[j,['p_n [Pa]']][0])
				temp1.append(df_log.loc[j,['p_n [Pa]']][0])
		target_chamber_pressure_2.append(np.nanmedian(temp1))
		pulse_energy_density2.append(np.nanmedian(temp2))
		TS_pulse_duration_at_max_power2.append(np.nanmedian(temp3))
		pulse_energy.append(np.nanmedian(temp4))

	# fig.suptitle('Pressure scan\nmagnetic_field %.3gT,target/OES distance %.3gmm,ELM energy %.3gJ\n ' %(magnetic_field[0],target_OES_distance[0],CB_pulse_energy[0]))
	ax[0,0].errorbar(j_specific_pulse_en_semi_inf,1e-6*np.array(j_specific_energy_density)/(np.array(j_TS_pulse_duration_at_max_power)**0.5),xerr=j_specific_pulse_en_semi_inf_sigma,yerr=1e-6*np.array(j_specific_energy_density_sigma)/(np.array(j_TS_pulse_duration_at_max_power)**0.5),fmt='+',color=color[index],capsize=5)
	temp = np.polyfit(j_specific_pulse_en_semi_inf,1e-6*np.array(j_specific_energy_density)/(np.array(j_TS_pulse_duration_at_max_power)**0.5),1)
	ax[0,0].plot(np.sort(j_specific_pulse_en_semi_inf),np.polyval(temp,np.sort(j_specific_pulse_en_semi_inf)),'--',linewidth=4,color=color[index],label='B=%.3g' %(np.nanmedian(magnetic_field)))
	ax[1,0].errorbar(j_specific_pulse_en_semi_inf,1e6*np.array(j_specific_area_of_interest_IR),xerr=j_specific_pulse_en_semi_inf_sigma,yerr=1e6*np.array(j_specific_area_of_interest_IR_sigma),fmt='+',color=color[index],capsize=5)
	temp = np.polyfit(j_specific_pulse_en_semi_inf,j_specific_area_of_interest_IR,1)
	ax[1,0].plot(np.sort(j_specific_pulse_en_semi_inf),1e6*np.polyval(temp,np.sort(j_specific_pulse_en_semi_inf)),'--',linewidth=4,color=color[index])
	ax[2,0].errorbar(j_specific_pulse_en_semi_inf2,np.array(j_specific_area_ratio_IR),xerr=j_specific_pulse_en_semi_inf_sigma2,yerr=j_specific_area_ratio_IR_sigma,fmt='+',color=color[index],capsize=5)

	if index!=3:
		ax4[0,0].errorbar(j_specific_pulse_en_semi_inf,1e-6*np.array(j_specific_energy_density)/(np.array(j_TS_pulse_duration_at_max_power)**0.5),xerr=j_specific_pulse_en_semi_inf_sigma,yerr=1e-6*np.array(j_specific_energy_density_sigma)/(np.array(j_TS_pulse_duration_at_max_power)**0.5),fmt='+',color=color[index],capsize=5)
		temp = np.polyfit(j_specific_pulse_en_semi_inf,1e-6*np.array(j_specific_energy_density)/(np.array(j_TS_pulse_duration_at_max_power)**0.5),1)
		ax4[0,0].plot(np.sort(j_specific_pulse_en_semi_inf),np.polyval(temp,np.sort(j_specific_pulse_en_semi_inf)),'--',linewidth=4,color=color[index],label='B=%.3g' %(np.nanmedian(magnetic_field)))
		ax4[1,0].errorbar(j_specific_pulse_en_semi_inf,1e6*np.array(j_specific_area_of_interest_IR),xerr=j_specific_pulse_en_semi_inf_sigma,yerr=1e6*np.array(j_specific_area_of_interest_IR_sigma),fmt='+',color=color[index],capsize=5)
		temp = np.polyfit(j_specific_pulse_en_semi_inf,j_specific_area_of_interest_IR,1)
		ax4[1,0].plot(np.sort(j_specific_pulse_en_semi_inf),1e6*np.polyval(temp,np.sort(j_specific_pulse_en_semi_inf)),'--',linewidth=4,color=color[index])

	ax2[0,0].errorbar(j_specific_target_chamber_pressure,1e-6*np.array(j_specific_energy_density)/(np.array(j_TS_pulse_duration_at_max_power)**0.5),yerr=1e-6*np.array(j_specific_energy_density_sigma)/(np.array(j_TS_pulse_duration_at_max_power)**0.5),fmt='+',color=color[index],capsize=5)
	temp = np.array([y for _, y in sorted(zip(target_chamber_pressure_2, 1e-6*np.array(pulse_energy_density2)/np.array(TS_pulse_duration_at_max_power2)**0.5))])
	ax2[0,0].plot(np.sort(target_chamber_pressure_2),temp,color=color[index],label='B=%.3g' %(np.nanmedian(magnetic_field)))

	ax3[0,0].errorbar(j_specific_target_chamber_pressure,j_specific_pulse_en_semi_inf,yerr=j_specific_pulse_en_semi_inf_sigma,fmt='+',color=color[index],capsize=5)
	temp = np.array([y for _, y in sorted(zip(target_chamber_pressure_2, pulse_energy))])
	ax3[0,0].plot(np.sort(target_chamber_pressure_2),temp,color=color[index],label='B=%.3g' %(np.nanmedian(magnetic_field)))

ax[0,0].grid()
ax[0,0].legend(loc='best', fontsize='small')
ax[1,0].grid()
ax[2,0].grid()
ax[0,0].set_ylabel('Heat flux factor\n[MJ/m2 s^-1/2]')
ax[1,0].set_ylabel('Interested area\nfound [mm2]')
ax[2,0].set_ylabel('ELM/SS area ratio')
ax[1,0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax[2,0].set_xlabel('Pulse delivered energy [J]')
ax[0,0].set_xlim(left=-0.2,right=16)
# ax[1,0].set_xlim(left=0,right=15)
ax[0,0].set_ylim(bottom=-0.2,top=9)
ax[1,0].set_ylim(bottom=-10,top=700)
ax[2,0].set_ylim(bottom=-1,top=20)

ax4[0,0].grid()
ax4[0,0].legend(loc='best', fontsize='small')
ax4[1,0].grid()
ax4[0,0].set_ylabel('Heat flux factor\n[MJ/m2 s^-1/2]')
ax4[1,0].set_ylabel('Interested area\nfound [mm2]')
ax4[1,0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax4[0,0].set_xlim(left=-0.2,right=16)
# ax4[1,0].set_xlim(left=0,right=15)
ax4[0,0].set_ylim(bottom=-0.2,top=9)
ax4[1,0].set_ylim(bottom=-10,top=700)

ax2[0,0].set_ylabel('Heat flux factor [MJ/m2 s^-1/2]')
ax2[0,0].set_xlabel('Target chamber pressure [Pa]')
ax2[0,0].legend(loc='best', fontsize='small')
ax2[0,0].grid()
# plt.pause(0.01)

ax3[0,0].set_ylabel('Pulse delivered energy [J]')
ax3[0,0].set_xlabel('Target chamber pressure [Pa]')
ax3[0,0].legend(loc='best', fontsize='small')
ax3[0,0].grid()
plt.pause(0.01)


# I compare the particle flow to the target to search for a detachment marker
merge_ID_target_multipulse_all = [[99,98,96,97],[95,89,87,86,85],[101,75,76,77,78,79],[70, 69, 68, 67, 66]]
fig, ax = plt.subplots( len(merge_ID_target_multipulse_all),1,figsize=(6, 20), squeeze=True, sharex=True)
for i_merge_ID_target_multipulse,merge_ID_target_multipulse in enumerate(merge_ID_target_multipulse_all):
	for i_merge_ID_target,merge_ID_target in enumerate(merge_ID_target_multipulse):
		target_chamber_pressure = []
		integrated_Bohm_adiabatic_flow = []
		new_timesteps = []
		print('Looking at '+str(merge_ID_target))
		path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)
		try:
			full_saved_file_dict = np.load(path_where_to_save_everything +'/TS_data_merge_'+str(merge_ID_target) +'.npz')
			integrated_Bohm_adiabatic_flow = full_saved_file_dict['averaged_stats'].all()['integrated_Bohm_adiabatic_flow']
			new_timesteps = full_saved_file_dict['averaged_stats'].all()['new_timesteps']
			all_j=find_index_of_file(merge_ID_target,df_settings,df_log,only_OES=True)
			target_chamber_pressure = df_log.loc[all_j[0],['p_n [Pa]']][0]
			mgnetic_field = df_log.loc[all_j[0],['B']][0]
			CB_voltage = df_log.loc[all_j[0],['Vc']][0]
			ax[i_merge_ID_target_multipulse].plot(new_timesteps,integrated_Bohm_adiabatic_flow,color=color[i_merge_ID_target],label='%.3gPa' %(target_chamber_pressure))
		except:
			continue
	ax[i_merge_ID_target_multipulse].set_title('Magnetic_field %.3gT,ELM pulse voltage %.3gV\n ' %(mgnetic_field,CB_voltage))
	ax[i_merge_ID_target_multipulse].grid()
	ax[i_merge_ID_target_multipulse].set_xlim(left=0,right=1.3)
	ax[i_merge_ID_target_multipulse].set_ylabel('flow [#/s]')
	ax[i_merge_ID_target_multipulse].legend(loc='best', fontsize='xx-small')
	if i_merge_ID_target_multipulse==len(merge_ID_target_multipulse_all)-1:
		ax[i_merge_ID_target_multipulse].set_xlabel('time from beginning of pulse [ms]')
plt.pause(0.01)


# I'm not sure anymore what was this for
# merge_ID_target_multipulse_all = [99,98,96,97,95,89,87,86]
merge_ID_target_multipulse = [99,98,96,97,95,89,87,86]
j_specific_pulse_en_semi_inf = []
j_specific_pulse_en_semi_inf_sigma = []
j_specific_energy_density = []
j_specific_energy_density_sigma = []
j_TS_pulse_duration_at_max_power = []
j_specific_area_of_interest_IR = []
j_specific_area_of_interest_IR_sigma = []
j_magnetic_field = []
j_delivered_pulse_energy = []
net_power_removed_plasma_column = []
net_power_removed_plasma_column_sigma = []

for merge_ID_target in merge_ID_target_multipulse:
	print('Looking at '+str(merge_ID_target))
	all_j=find_index_of_file(merge_ID_target,df_settings,df_log,only_OES=True)


	for j in all_j:
		IR_trace, = df_log.loc[j,['IR_trace']]
		if isinstance(IR_trace,str):
			temp = energy_flow_from_TS_at_sound_speed(merge_ID_target,interpolated_TS=True)
			temp[1][temp[1]>0]-=np.mean(temp[1][temp[0]<=0])
			temp1=np.cumsum(temp[1])
			j_TS_pulse_duration_at_max_power.append(temp1.max()/np.max(temp[1])*np.median(np.diff(temp[0]))*1e-3)
			j_magnetic_field.append(df_log.loc[j,['B']][0])
			path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)
			full_saved_file_dict = dict(np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz'))
			j_specific_pulse_en_semi_inf.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_energy'])
			j_specific_pulse_en_semi_inf_sigma.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_energy_sigma'])
			j_specific_energy_density.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_energy_density'])
			j_specific_energy_density_sigma.append(full_saved_file_dict['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_energy_density_sigma'])
			j_specific_area_of_interest_IR.append(full_saved_file_dict['average_pulse_fitted_data'].all()['area_of_interest'])
			j_specific_area_of_interest_IR_sigma.append(full_saved_file_dict['average_pulse_fitted_data'].all()['area_of_interest_sigma'])
			j_delivered_pulse_energy.append(np.float(results_summary.loc[merge_ID_target,['Delivered energy [J]']]))

			path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)
			full_saved_file_dict = np.load(path_where_to_save_everything+'/Yacora_Bayesian/absolute/lines_fitted5/fit_bounds_from_sims/spatial_factor_1/time_shift_factor_0/only_Hm_H2_H2p_mol_lim/bayesian_results3'+'.npz')
			net_power_removed_plasma_column.append(full_saved_file_dict['net_power_removed_plasma_column'].all()['radial_time_sum']['most_likely'])
			net_power_removed_plasma_column_sigma.append(full_saved_file_dict['net_power_removed_plasma_column'].all()['radial_time_sum']['most_likely_sigma'])




######
