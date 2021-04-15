
# echo 'running batch'

import os,sys
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
import mkl
# mkl.set_num_threads(1)
import numpy as np
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())
# import matplotlib.pyplot as plt
#import .functions
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
from functions.fabio_add import find_index_of_file, examine_current_trace, shift_between_TS_and_power_source, load_TS, average_TS_around_axis
from generate_merge88_TS import generate_merge88_TS
import collections

from adas import read_adf15,read_adf11
from PIL import Image
import xarray as xr
import pandas as pd
import copy
from scipy.optimize import curve_fit,least_squares
from scipy import interpolate
from scipy.signal import find_peaks, peak_prominences as get_proms
from sklearn.linear_model import LinearRegression
import time as tm
import pickle
from multiprocessing import Pool,cpu_count,current_process,set_start_method,get_context,Semaphore
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import traceback
# set_start_method('spawn')
number_cpu_available = 8	#cpu_count()
# pool = get_context('spawn').Pool(number_cpu_available)
print('Number of cores available: '+str(number_cpu_available))
# number_cpu_available = 10
# print('Number of cores available: '+str(number_cpu_available))
# print('Number of cores available: '+str(len(os.sched_getaffinitymy_r_pos(0))))
# import psutil
# print('Number of cores available: '+str(psutil.cpu_count(logical = False)))



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
pltfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/plt12_h.dat'
prbfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/prb12_h.dat'
ccdfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/ccd96_h.dat'

# Data from wikipedia
energy_difference = np.array([2.54970, 2.85567, 3.02187, 3.12208, 3.18716, 3.23175, 3.26365, 3.28725, 3.30520])  # eV
energy_difference_full = np.array([10.1988,1.88867, 2.54970, 2.85567, 3.02187, 3.12208, 3.18716, 3.23175, 3.26365, 3.28725, 3.30520, 3.31917])  # eV
# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
statistical_weigth = np.array([32, 50, 72, 98, 128, 162, 200, 242, 288])  # gi-gk
einstein_coeff = np.array([8.4193e-2, 2.53044e-2, 9.7320e-3, 4.3889e-3, 2.2148e-3, 1.2156e-3, 7.1225e-4, 4.3972e-4, 2.8337e-4]) * 1e8  # 1/s
einstein_coeff_full = np.array([4.6986e+00, 4.4101e-01, 8.4193e-2, 2.53044e-2, 9.7320e-3, 4.3889e-3, 2.2148e-3, 1.2156e-3, 7.1225e-4, 4.3972e-4, 2.8337e-4, 1.8927e-04]) * 1e8  # 1/s
einstein_coeff_full_full = np.array([[4.6986,5.5751e-01,1.2785e-01,4.1250e-02,1.6440e-02,7.5684e-03,3.8694e-03,2.1425e-03,1.2631e-03,7.8340e-04,5.0659e-04,3.3927e-04],[0,0.44101,0.084193,0.0253044,0.009732,0.0043889,0.0022148,0.0012156,0.00071225,0.00043972,0.00028337,0.00018927],[0,0,8.9860e-02,2.2008e-02,7.7829e-03,3.3585e-03,1.6506e-03,8.9050e-04,5.1558e-04,3.1558e-04,2.0207e-04,1.3431e-04],[0,0,0,2.6993e-02,7.7110e-03,3.0415e-03,1.4242e-03,7.4593e-04,4.2347e-04,2.5565e-04,1.6205e-04,1.0689e-04],[0,0,0,0,1.0254e-02,3.2528e-03,1.3877e-03,6.9078e-04,3.7999e-04,2.2460e-04,1.4024e-04,9.1481e-05],[0,0,0,0,0,4.5608e-03,1.5609e-03,7.0652e-04,3.6881e-04,2.1096e-04,1.2884e-04,8.2716e-05],[0,0,0,0,0,0,2.2720e-03,8.2370e-04,3.9049e-04,2.1174e-04,1.2503e-04,7.8457e-05],[0,0,0,0,0,0,0,1.2328e-03,4.6762e-04,2.3007e-04,1.2870e-04,7.8037e-05],[0,0,0,0,0,0,0,0,7.1514e-04,2.8131e-04,1.4269e-04,8.1919e-05],[0,0,0,0,0,0,0,0,0,4.3766e-04,1.7740e-04,9.2309e-05],[0,0,0,0,0,0,0,0,0,0,2.7989e-04,1.1633e-04],[0,0,0,0,0,0,0,0,0,0,0,1.8569e-04]]) * 1e8  # 1/s
level_1 = (np.ones((13,13))*np.arange(1,14)).T
level_2 = (np.ones((13,13))*np.arange(1,14))
# Used formula 2.9 and 2.12 in Rion Barrois thesys, 2017
energy_difference_full_full = 1/level_1**2-1/level_2**2
energy_difference_full_full = 13.6*energy_difference_full_full[:-1,1:]
energy_difference_full_full[energy_difference_full_full<0]=0	# energy difference between energy levels [eV]
plank_constant_eV = 4.135667696e-15	# eV s
light_speed = 299792458	# m/s
photon_wavelength_full_full = plank_constant_eV * light_speed / energy_difference_full_full	# m
visible_light_flag_full_full = np.logical_and(photon_wavelength_full_full>=380*1e-9,photon_wavelength_full_full<=750*1e-9)
J_to_eV = 6.242e18
multiplicative_factor_full = energy_difference_full * einstein_coeff_full / J_to_eV
multiplicative_factor_full_full = np.sum(energy_difference_full_full * einstein_coeff_full_full / J_to_eV,axis=0)
multiplicative_factor_visible_light_full_full = np.sum(energy_difference_full_full * visible_light_flag_full_full * einstein_coeff_full_full / J_to_eV,axis=0)
au_to_kg = 1.66053906660e-27	# kg/au
# Used formula 2.3 in Rion Barrois thesys, 2017
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro','paleturquoise']
boltzmann_constant_J = 1.380649e-23	# J/K
eV_to_K = 8.617333262145e-5	# eV/K
avogadro_number = 6.02214076e23
hydrogen_mass = 1.008*1.660*1e-27	# kg
electron_mass = 9.10938356* 1e-31	# kg
ionisation_potential = 13.6	# eV
dissociation_potential = 2.2	# eV


# absolute intensity fit with Yacora coefficients, Bayesian aproach suggested by Kevin

# for merge_ID_target in [851,86,87,89,92]:	# 88 excluded because I don't have a temperature profile
# 	merge_time_window = [-10, 10]
# for merge_ID_target in [ 93, 94]:  # 88 excluded because I don't have a temperature profile
# 	merge_time_window = [-1,2]

# merge_ID_target_multipulse = np.flip([851,86,87,89,92, 93, 94],axis=0)
# merge_ID_target_multipulse = np.flip([73,75,76,77,78,79,85, 95, 86, 87, 88, 89, 92, 93, 94, 96, 97, 98, 99],axis=0)
# merge_ID_target_multipulse = [85, 95, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99]
# merge_ID_target_multipulse = [87, 88, 89]
# merge_ID_target_multipulse = [91, 92, 93, 94]
# merge_ID_target_multipulse = [73,75,76,77,78,79]
# merge_ID_target_multipulse = [85, 95, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99]
# merge_ID_target_multipulse = np.flip([85, 95, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99],axis=0)
# merge_ID_target_multipulse = np.flip([86, 87, 88, 89, 92, 93, 94, 96, 97, 98, 99],axis=0)
# merge_ID_target_multipulse = np.flip([95, 86, 87, 88, 89, 92, 93, 94, 96, 97, 98, 88],axis=0)
# merge_ID_target_multipulse = np.flip([95, 94, 93, 92, 89, 87, 86, 85],axis=0)
# merge_ID_target_multipulse = [73,77,78,79]
# merge_ID_target_multipulse = [86,95,85,87,89,88]
merge_ID_target_multipulse = [86]

merge_ID_target_multipulse = np.flip(merge_ID_target_multipulse,axis=0)

for merge_ID_target in merge_ID_target_multipulse:  # 88 excluded because I don't have a temperature profile
	merge_time_window = [-1,2]
	# if (merge_ID_target>=93 and merge_ID_target<100):
	# 	merge_time_window = [-1,2]
	# else:
	# 	merge_time_window = [-10,10]


	recorded_data_override = [True,True,True]
	# recorded_data_override = [False,False,False]
	# recorded_data_override = [True,True]
	completion_loop_done = True
	do_only_one_loop = False
	include_particles_limitation = True
	H2_suppression = False

	externally_provided_TS_Te_steps = 13
	externally_provided_TS_Te_steps_increase = 5
	externally_provided_TS_ne_steps = 13
	externally_provided_TS_ne_steps_increase = 6
	if H2_suppression:
		externally_provided_H2p_steps = 17
		externally_provided_Hn_steps = 17
		externally_provided_H_steps = 17
		externally_provided_H2_steps = 1
	else:
		externally_provided_H2p_steps = 13
		externally_provided_Hn_steps = 11
		externally_provided_H_steps = 11
		externally_provided_H2_steps = 11

	# merge_ID_target = 85

	# for min_nHp_ne,atomic_restricted_high_n in [[0.999,False],[0.8,False],[0.999,True],[0.3,False],[0,False]]:	#0.9999 is for the case when I relax this boundary if nH<0.001
	# for min_nHp_ne,atomic_restricted_high_n in [[0.999,False],[0.999,True],[0.01,True]]:
	for min_nHp_ne,atomic_restricted_high_n in [[0.01,True]]:

		# for max_n_neutrals_ne in [1,2,10,100]:
		for max_n_neutrals_ne in [1]:

			figure_index = 0

			# calculate_geometry = False
			# merge_ID_target = 17	#	THIS IS GIVEN BY THE LAUNCHER
			for i in range(10):
				print('.')
			print('Starting to work on merge number ' + str(merge_ID_target))
			print('of requested '+str(merge_ID_target_multipulse))
			for i in range(10):
				print('.')

			time_resolution_scan = False
			time_resolution_scan_improved = True
			time_resolution_extra_skip = 0

			if False:
				n_list = np.array([5, 6, 7,8, 9,10,11,12])
				n_list_1 = np.array([4])
				n_weights = [1, 1, 1, 1, 1, 1,3,1,1]
			elif False:
				n_list = np.array([5, 6, 7,8, 9,10])
				n_list_1 = np.array([4])
				n_weights = [1, 1, 1, 1, 1, 1,3]
			elif False:
				n_list = np.array([6,7,8,9])
				n_list_1 = np.array([4,5])
				n_weights = [1, 1, 1, 1,1, 3,3]
			else:
				n_list = np.array([6,7,8])
				n_list_1 = np.array([4,5])
				n_weights = [1, 1, 1, 1,1, 3]
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

			mod = '/Yacora_Bayesian/absolute/lines_fitted'+str(len(n_list)+len(n_list_1))+'/fit_bounds_from_sims'


			# here I determine the standard time shift between TS and power source
			if merge_ID_target in [851,85,86,87,88,89,90,91,92,93,94,95]:
				shift_TS_to_power_source = shift_between_TS_and_power_source(95)
				print('standard time shift between TS and power source of %.3gms from %.5g' %(shift_TS_to_power_source,95))
			elif merge_ID_target in [37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,54,96,97,98,99]:
				shift_TS_to_power_source = shift_between_TS_and_power_source(99)
				print('standard time shift between TS and power source of %.3gms from %.5g' %(shift_TS_to_power_source,99))
			elif merge_ID_target in [74,75,76,77,78,79,80,81,82,83,84,101]:
				shift_TS_to_power_source = shift_between_TS_and_power_source(101)
				print('standard time shift between TS and power source of %.3gms from %.5g' %(shift_TS_to_power_source,101))
			elif merge_ID_target in [66,67,68,69,71,72,73,70]:
				shift_TS_to_power_source = shift_between_TS_and_power_source(70)
				print('standard time shift between TS and power source of %.3gms from %.5g' %(shift_TS_to_power_source,70))
			else:
				print('There is an error, the reference shot for TS/power source shift was not given')
				exit()
			internal_shift_TS_to_power_source = shift_between_TS_and_power_source(merge_ID_target)

			all_j=find_index_of_file(merge_ID_target,df_settings,df_log,only_OES=True)
			target_chamber_pressure = []
			source_flow_rate = []
			target_OES_distance = []
			feed_rate_SLM = []
			capacitor_voltage = []
			magnetic_field = []
			power_pulse_shape_time_dependent = []
			power_pulse_shape_time_dependent_std = []
			steady_state_power = []
			steady_state_power_std = []
			energy_delivered_good_pulses = []
			mean_target_voltage_peak_shape = []
			mean_target_voltage_peak_std = []
			mean_steady_state_target_voltage = []
			mean_steady_state_target_voltage_std = []
			mean_voltage_peak_shape = []
			mean_voltage_peak_std = []
			mean_steady_state_voltage = []
			mean_steady_state_voltage_std = []
			for j in all_j:
				target_chamber_pressure.append(df_log.loc[j,['p_n [Pa]']])
				source_flow_rate.append(df_log.loc[j,['Flow']])
				target_OES_distance.append(df_log.loc[j,['T_axial']])
				feed_rate_SLM.append(df_log.loc[j,['Seed']])
				capacitor_voltage.append(df_log.loc[j,['Vc']])
				magnetic_field.append(df_log.loc[j,['B']])
				(merge_folder,sequence,fname_current_trace) = df_log.loc[j,['folder','sequence','current_trace_file']]
				sequence = int(sequence)
				bad_pulses,first_good_pulse,first_pulse,last_pulse,miss_pulses,double_pulses,good_pulses, time_of_pulses, energy_per_pulse,duration_per_pulse,median_energy_delivered_good_pulses,median_duration_good_pulses,mean_peak_shape,mean_peak_std,mean_steady_state_power,mean_steady_state_power_std,time_resolution,trash1,trash2,voltage_peak_shape,voltage_peak_std,target_voltage_peak_shape,target_voltage_peak_std,trash5,trash6,steady_state_voltage,steady_state_voltage_std,steady_state_target_voltage,steady_state_target_voltage_std = examine_current_trace(fdir+'/'+merge_folder+'/'+"{0:0=2d}".format(sequence)+'/', fname_current_trace, df_log.loc[j, ['number_of_pulses']][0],want_the_power_per_pulse=True,want_the_mean_power_profile=True)
				energy_delivered_good_pulses.append(median_energy_delivered_good_pulses)
				power_pulse_shape_time_dependent.append(mean_peak_shape)
				power_pulse_shape_time_dependent_std.append(mean_peak_std)
				steady_state_power.append(mean_steady_state_power)
				steady_state_power_std.append(mean_steady_state_power_std)
				target_material = df_log.loc[j,['Target']][0]
				mean_target_voltage_peak_shape.append(target_voltage_peak_shape)
				mean_target_voltage_peak_std.append(target_voltage_peak_std)
				mean_steady_state_target_voltage.append(steady_state_target_voltage)
				mean_steady_state_target_voltage_std.append(steady_state_target_voltage_std)
				mean_voltage_peak_shape.append(voltage_peak_shape)
				mean_voltage_peak_std.append(voltage_peak_std)
				mean_steady_state_voltage.append(steady_state_voltage)
				mean_steady_state_voltage_std.append(steady_state_voltage_std)
			target_chamber_pressure = np.nanmean(target_chamber_pressure)	# Pa
			source_flow_rate = np.nanmean(source_flow_rate)	# Pa
			target_OES_distance = np.nanmean(target_OES_distance)	# Pa
			feed_rate_SLM = np.nanmean(feed_rate_SLM)	# SLM
			capacitor_voltage = np.nanmean(capacitor_voltage)	# V
			magnetic_field = np.nanmean(magnetic_field)	# T
			energy_delivered_good_pulses = np.nanmean(energy_delivered_good_pulses)	# J
			# power_pulse_shape = np.sum(np.divide(power_pulse_shape,power_pulse_shape_std),axis=0)/np.sum(np.divide(1,power_pulse_shape_std),axis=0)
			power_pulse_shape_time_dependent = 0.92*np.mean(power_pulse_shape_time_dependent,axis=0)	# 0.92 is the efficiency of electricity to plasma heat conversion
			power_pulse_shape_time_dependent_std = 0.92*np.sum(0.25*np.array(power_pulse_shape_time_dependent_std)**2,axis=0)**0.5	# 0.92 is the efficiency of electricity to plasma heat conversion
			# steady_state_power = np.sum(np.divide(steady_state_power,steady_state_power_std),axis=0)/np.sum(np.divide(1,power_pulse_shape_std),axis=0)
			steady_state_power = np.mean(steady_state_power,axis=0)
			steady_state_power_std = np.sum(0.25*np.array(steady_state_power_std)**2,axis=0)**0.5
			power_pulse_shape =power_pulse_shape_time_dependent + steady_state_power
			power_pulse_shape_std = (power_pulse_shape_time_dependent_std**2+steady_state_power_std**2)**0.5
			# time_source_power = np.arange(len(power_pulse_shape))*time_resolution*1000 - 49.945	# 49.945 comes from TS-current trace comparison.py using merge 75 and 95
			time_source_power = np.arange(len(power_pulse_shape))*time_resolution*1000 - shift_TS_to_power_source	# this way because the time vary with magnetic field, probably because resulting in different sound speed
			interpolated_power_pulse_shape = interp1d(time_source_power,power_pulse_shape)
			interpolated_power_pulse_shape_std = interp1d(time_source_power,power_pulse_shape_std)
			# Ideal gas law
			nH2_from_pressure = target_chamber_pressure/(boltzmann_constant_J*300)	# [#/m^3] I suppose ambient temp is ~ 300K
			mean_target_voltage_peak_shape = np.nanmean(mean_target_voltage_peak_shape,axis=0)
			mean_target_voltage_peak_std = np.nanmean(mean_target_voltage_peak_std,axis=0)
			mean_steady_state_target_voltage = np.nanmean(mean_steady_state_target_voltage,axis=0)
			mean_steady_state_target_voltage_std = np.nanmean(mean_steady_state_target_voltage_std,axis=0)
			interpolated_target_voltage_shape = interp1d(time_source_power,mean_target_voltage_peak_shape-mean_steady_state_target_voltage)
			interpolated_target_voltage_shape_std = interp1d(time_source_power,(mean_target_voltage_peak_std**2+mean_steady_state_target_voltage_std**2)**0.5)
			mean_voltage_peak_shape = np.nanmean(mean_voltage_peak_shape,axis=0)
			mean_voltage_peak_std = np.nanmean(mean_voltage_peak_std,axis=0)
			mean_steady_state_voltage = np.nanmean(mean_steady_state_voltage,axis=0)
			mean_steady_state_voltage_std = np.nanmean(mean_steady_state_voltage_std,axis=0)
			interpolated_voltage_shape = interp1d(time_source_power,mean_voltage_peak_shape-mean_steady_state_voltage)
			interpolated_voltage_shape_std = interp1d(time_source_power,(mean_voltage_peak_std**2+mean_steady_state_voltage_std**2)**0.5)

			print('lines')
			print(str(n_list_1.tolist()+n_list.tolist()))
			print('min_nHp/ne')
			print(min_nHp_ne)
			print('max_nH/ne')
			print(max_n_neutrals_ne)
			print('steady state nH2 [#/m^3]')
			print(nH2_from_pressure)
			print("considering that it's a transient phenomena the istantaneous max_nH2 bound is *10/*0.1 times that")
			if False:	# I cannot assume a constant temperature, so I use a simple model in the target chamber heating to estimate the max H2 density.
				max_nH2_from_pressure = 10*max_nH2_from_pressure
				print("considering that it's a transient phenomena the istantaneous max_nH2 is set 10 times higher to [#/m^3]")
				print(max_nH2_from_pressure)
			print('max_nHp/ne')
			print(max_nHp_ne)
			print('n_weights')
			print(n_weights)
			print('recorded data override is '+str(recorded_data_override))
			print('will the very last loop be done on top of the third, just to redo points that failed? '+str(completion_loop_done))
			fixed_fractions_nHm_H2p = [1/4,1,4]
			max_frac_acceptable_deviation_ext = 0.08	#0.07
			print('the fixed values of nH-/nH2+ tested are '+str(fixed_fractions_nHm_H2p)+' +/-'+str(max_frac_acceptable_deviation_ext*100)+'%')
			min_mol_fraction_ext = 1e-6
			print('minimum molecular fraction to give significant effect on emission taken as '+str(min_mol_fraction_ext))
			collect_power_PDF = True
			print('Will power PDF be collected? '+str(collect_power_PDF))
			print('Will particles banace be included? '+str(include_particles_limitation))
			timeout_bayesian_search = 60*60	# time in seconds
			print('Timeout for bayesian search of a single point %.3g min' %(timeout_bayesian_search/60))
			power_molecular_precision,power_atomic_precision,power_budget_precision = 0.5,0.2,0.5
			print('precision (sigma) for molecular elements of power balance %.3g%%, for atomic elements %.3g%%, for the budget %.3g%%' %(power_molecular_precision*100,power_atomic_precision*100,power_budget_precision*100))
			particle_molecular_precision,particle_atomic_precision,particle_molecular_budget_precision,particle_atomic_budget_precision = 0.5,0.2,0.2,0.5
			if include_particles_limitation:
				print('precision (sigma) for molecular elements of particle balance %.3g%%, for atomic elements %.3g%%, for the molecular budget %.3g%%, for the atomic budget %.3g%%' %(particle_molecular_precision*100,particle_atomic_precision*100,particle_molecular_budget_precision*100,particle_atomic_budget_precision*100))
			print('externally_provided_H2p_steps = '+str(externally_provided_H2p_steps))
			print('externally_provided_Hn_steps = '+str(externally_provided_Hn_steps))
			print('externally_provided_H2_steps = '+str(externally_provided_H2_steps))
			print('externally_provided_H_steps = '+str(externally_provided_H_steps))
			print('externally_provided_TS_Te_steps = '+str(externally_provided_TS_Te_steps))
			print('externally_provided_TS_Te_steps_increase = '+str(externally_provided_TS_Te_steps_increase))
			print('externally_provided_TS_ne_steps = '+str(externally_provided_TS_ne_steps))
			print('externally_provided_TS_ne_steps_increase = '+str(externally_provided_TS_ne_steps_increase))

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
			path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) #+ '_back'

			if not os.path.exists(path_where_to_save_everything + mod):
				os.makedirs(path_where_to_save_everything + mod)

			# new_timesteps = np.linspace(merge_time_window[0] + 0.5, merge_time_window[1] - 0.5, int(
			# 	(merge_time_window[1] - 0.5 - (merge_time_window[0] + 0.5)) / conventional_time_step + 1))
			# dt = np.nanmedian(np.diff(new_timesteps))

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


			inverted_profiles_original = 4 * np.pi * np.load(path_where_to_save_everything + '/inverted_profiles.npy')  # in W m^-3 sr^-1  *  4*pi sr  =  W m^-3
			inverted_profiles_sigma_original = 4 * np.pi * np.load(path_where_to_save_everything + '/inverted_profiles_sigma.npy')  # in W m^-3 sr^-1  *  4*pi sr  =  W m^-3
			all_fits = np.load(path_where_to_save_everything + '/merge' + str(merge_ID_target) + '_all_fits.npy')  # in W m^-2 sr^-1
			# merge_Te_prof_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_Te_prof_multipulse']
			# merge_dTe_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_dTe_multipulse']
			# merge_ne_prof_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_ne_prof_multipulse']
			# merge_dne_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_dne_multipulse']
			# merge_time_original = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_time']
			# new_timesteps = np.load(path_where_to_save_everything + '/merge' + str(merge_ID_target) + '_new_timesteps.npy')
			new_timesteps = np.linspace(-0.5,1.5,num=len(inverted_profiles_original))
			dt = np.nanmedian(np.diff(new_timesteps))
			start_time = np.abs(new_timesteps - 0).argmin()
			end_time = np.abs(new_timesteps - 1.5).argmin() + 1
			number_of_radial_divisions = np.shape(inverted_profiles_original)[-1]

			# dx = 18 / 40 * (50.5 / 27.4) / 1e3
			dx = 1.06478505470992 / 1e3	# 10/02/2020 from	Calculate magnification_FF.xlsx
			xx = np.arange(40) * dx  # m
			xn = np.linspace(0, max(xx), 1000)
			# r = np.linspace(-max(xx) / 2, max(xx) / 2, 1000)
			# r = r[::10]
			# dr = np.median(np.diff(r))
			# number_of_radial_divisions = np.shape(inverted_profiles_original)[-1]
			r = np.arange(number_of_radial_divisions)*dx
			dr = np.median(np.diff(r))

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


					# merge_time = time_shift_factor + merge_time_original
					inverted_profiles = (1 / spatial_factor) * inverted_profiles_original  # in W m^-3 sr^-1  *  4*pi sr  =  W m^-3
					inverted_profiles_sigma = cp.deepcopy(inverted_profiles_sigma_original)
					# TS_dt = np.nanmedian(np.diff(merge_time))
					#
					# if (np.max(merge_Te_prof_multipulse) <= 0) and merge_ID_target!=88 :
					# 	print('merge' + str(merge_ID_target) + " has no recorded temperature")
					# 	continue
					#
					# TS_size = [-4.149230769230769056e+01, 4.416923076923076508e+01]
					# TS_r = TS_size[0] + np.linspace(0, 1, 65) * (TS_size[1] - TS_size[0])
					# TS_dr = np.median(np.diff(TS_r)) / 1000
					# gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
					# profile_centres = []
					# profile_sigma = []
					# profile_centres_score = []
					# for index in range(np.shape(merge_ne_prof_multipulse)[0]):
					# 	yy = merge_ne_prof_multipulse[index]
					# 	yy_sigma = merge_dne_multipulse[index]
					# 	yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
					# 	if np.sum(yy>0)<5:
					# 		profile_centres.append(0)
					# 		profile_sigma.append(50)
					# 		profile_centres_score.append(np.max(TS_r))
					# 		continue
					# 	yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
					# 	p0 = [np.max(yy), 10, 0]
					# 	bds = [[0, 0, np.min(TS_r)], [np.inf, TS_size[1], np.max(TS_r)]]
					# 	fit = curve_fit(gauss, TS_r, yy, p0, sigma=yy_sigma, maxfev=100000, bounds=bds)
					# 	profile_centres.append(fit[0][-1])
					# 	profile_sigma.append(fit[0][-2])
					# 	profile_centres_score.append(fit[1][-1, -1])
					# # plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)
					# profile_centres = np.array(profile_centres)
					# profile_sigma = np.array(profile_sigma)
					# profile_centres_score = np.array(profile_centres_score)
					# # centre = np.nanmean(profile_centres[profile_centres_score < 1])
					# centre = np.nansum(profile_centres/(profile_centres_score**1))/np.sum(1/profile_centres_score**1)
					# TS_r_new = (TS_r - centre) / 1000
					# print('TS profile centre at %.3gmm compared to the theoretical centre' %centre)
					# # temp_r, temp_t = np.meshgrid(TS_r_new, merge_time)
					# # plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse,vmin=0);plt.colorbar().set_label('Te [eV]');plt.pause(0.01)
					# # plt.figure();plt.pcolor(temp_t,temp_r,merge_ne_prof_multipulse,vmin=0);plt.colorbar().set_label('ne [10^20 #/m^3]');plt.pause(0.01)
					#
					# if os.path.exists(path_where_to_save_everything + '/TS_SS_data_merge_' + str(merge_ID_target) + '.npz'):
					# 	merge_Te_prof_multipulse_SS = np.load(path_where_to_save_everything + '/TS_SS_data_merge_' + str(merge_ID_target) + '.npz')['merge_Te_prof_multipulse']
					# 	merge_dTe_multipulse_SS = np.load(path_where_to_save_everything + '/TS_SS_data_merge_' + str(merge_ID_target) + '.npz')['merge_dTe_multipulse']
					# 	merge_ne_prof_multipulse_SS = np.load(path_where_to_save_everything + '/TS_SS_data_merge_' + str(merge_ID_target) + '.npz')['merge_ne_prof_multipulse']
					# 	merge_dne_multipulse_SS = np.load(path_where_to_save_everything + '/TS_SS_data_merge_' + str(merge_ID_target) + '.npz')['merge_dne_multipulse']
					#
					# 	if False:	# I don't think I need this part, I use the SS info to replace directly the original time dependent data
					# 		gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
					# 		yy = merge_ne_prof_multipulse_SS
					# 		yy_sigma = merge_dne_multipulse_SS
					# 		if np.sum(np.isfinite(yy_sigma))>0:
					# 			yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
					# 			yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
					# 		else:
					# 			yy_sigma = np.ones_like(yy)*np.nanmin([np.nanmax(yy),1])
					# 		p0 = [np.max(yy), 10, 0]
					# 		bds = [[0, 0, np.min(TS_r)], [np.inf, TS_size[1], np.max(TS_r)]]
					# 		fit = curve_fit(gauss, TS_r, yy, p0, sigma=yy_sigma, maxfev=100000, bounds=bds)
					# 		SS_profile_centres=[fit[0][-1]]
					# 		SS_profile_sigma=[fit[0][-2]]
					# 		SS_profile_centres_score=[fit[1][-1, -1]]
					# 		# plt.figure();plt.plot(TS_r,merge_ne_prof_multipulse,label='ne')
					# 		# plt.plot([fit[0][-1],fit[0][-1]],[np.max(merge_ne_prof_multipulse),np.min(merge_ne_prof_multipulse)],'--',label='ne')
					# 		yy = merge_Te_prof_multipulse_SS
					# 		yy_sigma = merge_dTe_multipulse_SS
					# 		if np.sum(np.isfinite(yy_sigma))>0:
					# 			yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
					# 			yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
					# 		else:
					# 			yy_sigma = np.ones_like(yy)*np.nanmin([np.nanmax(yy),1])
					# 		p0 = [np.max(yy), 10, 0]
					# 		bds = [[0, 0, np.min(TS_r)], [np.inf, TS_size[1], np.max(TS_r)]]
					# 		fit = curve_fit(gauss, TS_r, yy, p0, sigma=yy_sigma, maxfev=100000, bounds=bds)
					# 		SS_profile_centres = np.append(SS_profile_centres,[fit[0][-1]],axis=0)
					# 		SS_profile_sigma = np.append(SS_profile_sigma,[fit[0][-2]],axis=0)
					# 		SS_profile_centres_score = np.append(SS_profile_centres_score,[fit[1][-1, -1]],axis=0)
					# 		SS_centre = np.nanmean(SS_profile_centres)
					# 		SS_TS_r_new = (TS_r - SS_centre) / 1000
					# 		print('TS profile SS_centre at %.3gmm compared to the theoretical centre' %centre)
					#
					# 		# This is the mean of Te and ne weighted in their own uncertainties.
					# 		interp_range_r = max(dx, TS_dr) * 1.5
					# 		# weights_r = TS_r_new/interp_range_r
					# 		weights_r = SS_TS_r_new
					# 		merge_Te_prof_multipulse_SS_interp = np.zeros_like(merge_Te_prof_multipulse_interp[ 0])
					# 		merge_dTe_prof_multipulse_SS_interp = np.zeros_like(merge_Te_prof_multipulse_interp[ 0])
					# 		merge_ne_prof_multipulse_SS_interp = np.zeros_like(merge_Te_prof_multipulse_interp[ 0])
					# 		merge_dne_prof_multipulse_SS_interp = np.zeros_like(merge_Te_prof_multipulse_interp[ 0])
					# 		for i_r, value_r in enumerate(np.abs(r)):
					# 			if np.sum(np.abs(np.abs(SS_TS_r_new) - value_r) < interp_range_r) == 0:
					# 				continue
					# 			selected_values = np.abs(np.abs(SS_TS_r_new) - value_r) < interp_range_r
					# 			selected_values[merge_Te_prof_multipulse_SS == 0] = False
					# 			# weights = 1/np.abs(weights_r[selected_values]+1e-5)
					# 			weights = 1/((weights_r[selected_values]-value_r)/interp_range_r)**2
					# 			# weights = np.ones((np.sum(selected_values)))
					# 			if np.sum(selected_values) == 0:
					# 				continue
					# 			merge_Te_prof_multipulse_SS_interp[i_r] = np.sum(merge_Te_prof_multipulse_SS[selected_values]*weights / merge_dTe_multipulse_SS[selected_values]) / np.sum(weights / merge_dTe_multipulse_SS[selected_values])
					# 			merge_ne_prof_multipulse_SS_interp[i_r] = np.sum(merge_ne_prof_multipulse_SS[selected_values]*weights / merge_dne_multipulse_SS[selected_values]) / np.sum(weights / merge_dne_multipulse_SS[selected_values])
					# 			if False: 	# suggestion from Daljeet: use the "worst case scenario" in therms of uncertainty
					# 				merge_dTe_prof_multipulse_SS_interp[i_r] = 1/(np.sum(1 / merge_dTe_multipulse_SS[selected_values]))*(np.sum( ((merge_Te_prof_multipulse_SS_interp[i_r]-merge_Te_prof_multipulse_SS[selected_values])/merge_dTe_multipulse_SS[selected_values])**2 )**0.5)
					# 				merge_dne_prof_multipulse_SS_interp[i_r] = 1/(np.sum(1 / merge_dne_multipulse_SS[selected_values]))*(np.sum( ((merge_ne_prof_multipulse_SS_interp[i_r]-merge_ne_prof_multipulse_SS[selected_values])/merge_dne_multipulse_SS[selected_values])**2 )**0.5)
					# 			else:
					# 				merge_dTe_prof_multipulse_SS_interp_temp = 1/(np.sum(1 / merge_dTe_multipulse_SS[selected_values]))*(np.sum( ((merge_Te_prof_multipulse_SS_interp[i_r]-merge_Te_prof_multipulse_SS[selected_values])/merge_dTe_multipulse_SS[selected_values])**2 )**0.5)
					# 				merge_dne_prof_multipulse_SS_interp_temp = 1/(np.sum(1 / merge_dne_multipulse_SS[selected_values]))*(np.sum( ((merge_ne_prof_multipulse_SS_interp[i_r]-merge_ne_prof_multipulse_SS[selected_values])/merge_dne_multipulse_SS[selected_values])**2 )**0.5)
					# 				merge_dTe_prof_multipulse_SS_interp[i_r] = max(merge_dTe_prof_multipulse_SS_interp_temp,(np.max(merge_Te_prof_multipulse_SS[selected_values])-np.min(merge_Te_prof_multipulse_SS[selected_values]))/2/2 )	# I enlarged the integration range by 1.5, so I reduce the sigma in the second calculation mechanism to compensate for that and get reasonables uncertainties
					# 				merge_dne_prof_multipulse_SS_interp[i_r] = max(merge_dne_prof_multipulse_SS_interp_temp,(np.max(merge_ne_prof_multipulse_SS[selected_values])-np.min(merge_ne_prof_multipulse_SS[selected_values]))/2/2 )	# I enlarged the integration range by 1.5, so I reduce the sigma in the second calculation mechanism to compensate for that and get reasonables uncertainties
					# 		temp_r, temp_t = np.meshgrid(r, new_timesteps)
					#
					# 		start_r = np.abs(r - 0).argmin()
					# 		end_r = np.abs(r - 5).argmin() + 1
					# 		r_crop = r[start_r:end_r]
					# 		merge_Te_prof_multipulse_SS_interp_crop = merge_Te_prof_multipulse_SS_interp[start_r:end_r]
					# 		merge_dTe_prof_multipulse_SS_interp_crop = merge_dTe_prof_multipulse_SS_interp[start_r:end_r]
					# 		merge_ne_prof_multipulse_SS_interp_crop = merge_ne_prof_multipulse_SS_interp[start_r:end_r]
					# 		merge_dne_prof_multipulse_SS_interp_crop = merge_dne_prof_multipulse_SS_interp[start_r:end_r]
					#
					# 		gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
					# 		yy = merge_ne_prof_multipulse_SS_interp_crop
					# 		yy_sigma = merge_dne_prof_multipulse_SS_interp_crop
					# 		yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
					# 		yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
					# 		p0 = [np.max(yy), np.max(r_crop)/2, np.min(r_crop)]
					# 		bds = [[0, 0, np.min(r_crop)], [np.inf, np.max(r_crop), np.max(r_crop)]]
					# 		fit = curve_fit(gauss, r_crop, yy, p0, maxfev=100000, bounds=bds, sigma=yy_sigma,absolute_sigma=True)
					# 		SS_averaged_profile_sigma=fit[0][-2]
					# 		# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)
					#
					# 		for i_t in range(len(time_crop)):	# the 2e20 limit in density comes from VanDerMeiden2012a
					# 			merge_Te_prof_multipulse_interp_crop[i_t,merge_ne_prof_multipulse_interp_crop[i_t]<2] = np.max([merge_Te_prof_multipulse_SS_interp_crop[merge_ne_prof_multipulse_interp_crop[i_t]<2],merge_Te_prof_multipulse_interp_crop[i_t,merge_ne_prof_multipulse_interp_crop[i_t]<2]],axis=0)
					# 			merge_dTe_prof_multipulse_interp_crop[i_t,merge_ne_prof_multipulse_interp_crop[i_t]<2] = np.max([merge_Te_prof_multipulse_SS_interp_crop[merge_ne_prof_multipulse_interp_crop[i_t]<2],merge_dTe_prof_multipulse_SS_interp_crop[merge_ne_prof_multipulse_interp_crop[i_t]<2]],axis=0)
					# 			merge_ne_prof_multipulse_interp_crop[i_t,merge_ne_prof_multipulse_interp_crop[i_t]<2] = np.max([merge_ne_prof_multipulse_SS_interp_crop[merge_ne_prof_multipulse_interp_crop[i_t]<2],merge_ne_prof_multipulse_interp_crop[i_t,merge_ne_prof_multipulse_interp_crop[i_t]<2]],axis=0)
					# 			merge_dne_prof_multipulse_interp_crop[i_t,merge_ne_prof_multipulse_interp_crop[i_t]<2] = np.max([merge_ne_prof_multipulse_SS_interp_crop[merge_ne_prof_multipulse_interp_crop[i_t]<2],merge_dne_prof_multipulse_SS_interp_crop[merge_ne_prof_multipulse_interp_crop[i_t]<2]],axis=0)
					#
					# 	else:
					# 		temp = np.mean(merge_ne_prof_multipulse,axis=1)
					# 		start = (temp>np.mean(merge_ne_prof_multipulse_SS)).argmax()
					# 		end = (np.flip(temp,axis=0)>np.mean(merge_ne_prof_multipulse_SS)).argmax()
					# 		merge_Te_prof_multipulse[:start] = merge_Te_prof_multipulse_SS
					# 		merge_dTe_multipulse[:start] = 2*merge_Te_prof_multipulse_SS
					# 		merge_ne_prof_multipulse[:start] = merge_ne_prof_multipulse_SS
					# 		merge_dne_multipulse[:start] = merge_ne_prof_multipulse_SS
					# 		merge_Te_prof_multipulse[-end:] = merge_Te_prof_multipulse_SS
					# 		merge_dTe_multipulse[-end:] = 2*merge_Te_prof_multipulse_SS
					# 		merge_ne_prof_multipulse[-end:] = merge_ne_prof_multipulse_SS
					# 		merge_dne_multipulse[-end:] = merge_ne_prof_multipulse_SS
					#
					#
					# # This is the mean of Te and ne weighted in their own uncertainties.
					# temp1 = np.zeros_like(inverted_profiles[:, 0])
					# temp2 = np.zeros_like(inverted_profiles[:, 0])
					# temp3 = np.zeros_like(inverted_profiles[:, 0])
					# temp4 = np.zeros_like(inverted_profiles[:, 0])
					# interp_range_t = max(dt, TS_dt) * 1.5
					# interp_range_r = max(dx, TS_dr) * 1.5
					# weights_r = (np.zeros_like(merge_Te_prof_multipulse) + TS_r_new)
					# weights_t = (((np.zeros_like(merge_Te_prof_multipulse)).T + merge_time).T)
					# for i_t, value_t in enumerate(new_timesteps):
					# 	if np.sum(np.abs(merge_time - value_t) < interp_range_t) == 0:
					# 		continue
					# 	for i_r, value_r in enumerate(np.abs(r)):
					# 		if np.sum(np.abs(np.abs(TS_r_new) - value_r) < interp_range_r) == 0:
					# 			continue
					# 		elif np.sum(np.logical_and(np.abs(merge_time - value_t) < interp_range_t,np.sum(merge_Te_prof_multipulse, axis=1) > 0)) == 0:
					# 			continue
					# 		selected_values_t = np.logical_and(np.abs(merge_time - value_t) < interp_range_t,np.sum(merge_Te_prof_multipulse, axis=1) > 0)
					# 		selected_values_r = np.abs(np.abs(TS_r_new) - value_r) < interp_range_r
					# 		selected_values = (np.array([selected_values_t])).T * selected_values_r
					# 		selected_values[merge_Te_prof_multipulse == 0] = False
					# 		# weights = 1/(weights_r[selected_values]-value_r)**2 + 1/(weights_t[selected_values]-value_t)**2
					# 		weights = 1/((weights_t[selected_values]-value_t)/interp_range_t)**2 + 1/((weights_r[selected_values]-value_r)/interp_range_r)**2
					# 		if np.sum(selected_values) == 0:
					# 			continue
					# 		# temp1[i_t,i_r] = np.mean(merge_Te_prof_multipulse[selected_values_t][:,selected_values_r])
					# 		# temp2[i_t, i_r] = np.max(merge_dTe_multipulse[selected_values_t][:,selected_values_r]) / (np.sum(np.isfinite(merge_dTe_multipulse[selected_values_t][:,selected_values_r])) ** 0.5)
					# 		temp1[i_t, i_r] = np.sum(merge_Te_prof_multipulse[selected_values]*weights / merge_dTe_multipulse[selected_values]) / np.sum(weights / merge_dTe_multipulse[selected_values])
					# 		# temp3[i_t,i_r] = np.mean(merge_ne_prof_multipulse[selected_values_t][:,selected_values_r])
					# 		# temp4[i_t, i_r] = np.max(merge_dne_multipulse[selected_values_t][:,selected_values_r]) / (np.sum(np.isfinite(merge_dne_multipulse[selected_values_t][:,selected_values_r])) ** 0.5)
					# 		temp3[i_t, i_r] = np.sum(merge_ne_prof_multipulse[selected_values]*weights / merge_dne_multipulse[selected_values]) / np.sum(weights / merge_dne_multipulse[selected_values])
					# 		if False: 	# suggestion from Daljeet: use the "worst case scenario" in therms of uncertainty
					# 			# temp2[i_t, i_r] = (np.sum(selected_values) / (np.sum(1 / merge_dTe_multipulse[selected_values]) ** 2)) ** 0.5
					# 			# temp4[i_t, i_r] = (np.sum(selected_values) / (np.sum(1 / merge_dne_multipulse[selected_values]) ** 2)) ** 0.5
					# 			temp2[i_t, i_r] = 1/(np.sum(1 / merge_dTe_multipulse[selected_values]))*(np.sum( ((temp1[i_t, i_r]-merge_Te_prof_multipulse[selected_values])/merge_dTe_multipulse[selected_values])**2 )**0.5)
					# 			temp4[i_t, i_r] = 1/(np.sum(1 / merge_dne_multipulse[selected_values]))*(np.sum( ((temp3[i_t, i_r]-merge_ne_prof_multipulse[selected_values])/merge_dne_multipulse[selected_values])**2 )**0.5)
					# 		else:
					# 			# temp2_temp = 1/(np.sum(1 / merge_dTe_multipulse[selected_values]))*(np.sum( ((temp1[i_t, i_r]-merge_Te_prof_multipulse[selected_values])/merge_dTe_multipulse[selected_values])**2 )**0.5)
					# 			# temp4_temp = 1/(np.sum(1 / merge_dne_multipulse[selected_values]))*(np.sum( ((temp3[i_t, i_r]-merge_ne_prof_multipulse[selected_values])/merge_dne_multipulse[selected_values])**2 )**0.5)
					# 			temp2[i_t, i_r] = max(np.sqrt(np.sum(weights**2))/np.sum(weights / merge_dTe_multipulse[selected_values]),(np.max(merge_Te_prof_multipulse[selected_values])-np.min(merge_Te_prof_multipulse[selected_values]))/2/2 )	# I enlarged the integration range by 1.5, so I reduce the sigma in the second calculation mechanism to compensate for that and get reasonables uncertainties
					# 			temp4[i_t, i_r] = max(np.sqrt(np.sum(weights**2))/np.sum(weights / merge_dne_multipulse[selected_values]),(np.max(merge_ne_prof_multipulse[selected_values])-np.min(merge_ne_prof_multipulse[selected_values]))/2/2 )	# I enlarged the integration range by 1.5, so I reduce the sigma in the second calculation mechanism to compensate for that and get reasonables uncertainties
					#
					# merge_Te_prof_multipulse_interp = np.array(temp1)
					# merge_dTe_prof_multipulse_interp = np.array(temp2)
					# merge_ne_prof_multipulse_interp = np.array(temp3)
					# merge_dne_prof_multipulse_interp = np.array(temp4)

					# I crop to the usefull stuff
					# start_time = np.abs(new_timesteps - 0).argmin()
					# end_time = np.abs(new_timesteps - 1.5).argmin() + 1
					time_crop = new_timesteps[start_time:end_time]
					start_r = np.abs(r - 0).argmin()
					end_r = np.abs(r - 5).argmin() + 1
					r_crop = r[start_r:end_r]
					temp_r, temp_t = np.meshgrid(r_crop, time_crop)
					# merge_Te_prof_multipulse_interp_crop = merge_Te_prof_multipulse_interp[start_time:end_time,start_r:end_r]
					# merge_Te_prof_multipulse_interp_crop[merge_Te_prof_multipulse_interp_crop<0]=0
					# merge_dTe_prof_multipulse_interp_crop = merge_dTe_prof_multipulse_interp[start_time:end_time, start_r:end_r]
					# merge_ne_prof_multipulse_interp_crop = merge_ne_prof_multipulse_interp[start_time:end_time,start_r:end_r]
					# merge_ne_prof_multipulse_interp_crop[merge_ne_prof_multipulse_interp_crop<0]=0
					# merge_dne_prof_multipulse_interp_crop = merge_dne_prof_multipulse_interp[start_time:end_time, start_r:end_r]
					inverted_profiles_crop = inverted_profiles[start_time:end_time, :, start_r:end_r]
					inverted_profiles_crop[np.isnan(inverted_profiles_crop)] = 0
					inverted_profiles_sigma_crop = inverted_profiles_sigma[start_time:end_time, :, start_r:end_r]
					inverted_profiles_sigma_crop[np.isnan(inverted_profiles_sigma_crop)] = 0
					all_fits_crop = all_fits[start_time:end_time]
					# inverted_profiles_crop[inverted_profiles_crop<0] = 0

					merge_Te_prof_multipulse,merge_dTe_multipulse,merge_ne_prof_multipulse,merge_dne_multipulse,centre,profile_centres,profile_sigma,profile_centres_score,TS_r,dt,TS_dt,dx,TS_dr,TS_r_new,merge_time,number_of_radial_divisions,merge_time_original = load_TS(merge_ID_target,new_timesteps,r,spatial_factor=spatial_factor,time_shift_factor=time_shift_factor)
					merge_Te_prof_multipulse_interp_crop,merge_dTe_prof_multipulse_interp_crop,merge_ne_prof_multipulse_interp_crop,merge_dne_prof_multipulse_interp_crop,interp_range_r = average_TS_around_axis(merge_Te_prof_multipulse,merge_dTe_multipulse,merge_ne_prof_multipulse,merge_dne_multipulse,r,TS_r,TS_dt,TS_r_new,merge_time,new_timesteps,number_of_radial_divisions,start_time=start_time,end_time=end_time)

					if True:	# this doesn't work, needs to be replaced with others Te, ne. I could try with data from 76, that is same but at 1.5T
						if merge_ID_target == 88:	# merge88 does not have TS measurements
							merge_Te_prof_multipulse_interp_crop,merge_dTe_prof_multipulse_interp_crop,merge_ne_prof_multipulse_interp_crop,merge_dne_prof_multipulse_interp_crop = generate_merge88_TS()
							print(merge_Te_prof_multipulse_interp_crop)
							print(merge_dTe_prof_multipulse_interp_crop)
							print(merge_ne_prof_multipulse_interp_crop)
							print(merge_dne_prof_multipulse_interp_crop)

					gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
					averaged_profile_sigma = []
					averaged_profile_sigma_sigma = []
					for index in range(np.shape(merge_ne_prof_multipulse_interp_crop)[0]):
						yy = merge_ne_prof_multipulse_interp_crop[index]
						yy_sigma = merge_dne_prof_multipulse_interp_crop[index]
						yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
						if (np.sum(yy>0)==0 or np.sum(yy_sigma>0)==0):
							averaged_profile_sigma.append(0)
							averaged_profile_sigma_sigma.append(1)
							continue
						yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
						yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
						p0 = [np.max(yy), np.max(r_crop)/2, 0]
						bds = [[0, 0, -interp_range_r/1000], [np.inf, np.max(r_crop), interp_range_r/1000]]
						fit = curve_fit(gauss, r_crop, yy, p0, maxfev=100000, bounds=bds, sigma=yy_sigma,absolute_sigma=True)
						averaged_profile_sigma.append(fit[0][-2])
						averaged_profile_sigma_sigma.append(fit[1][-2,-2])
					# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)
					averaged_profile_sigma = np.array(averaged_profile_sigma)
					averaged_profile_sigma_sigma = np.array(averaged_profile_sigma_sigma)**0.5
					mean_plasma_radius = np.sum(averaged_profile_sigma/averaged_profile_sigma_sigma)/np.sum(1/averaged_profile_sigma_sigma)
					averaged_profile_sigma_2 = []
					averaged_profile_sigma_2_sigma = []
					for index in range(np.shape(merge_ne_prof_multipulse_interp_crop)[0]):
						yy = merge_Te_prof_multipulse_interp_crop[index]
						yy_sigma = merge_dTe_prof_multipulse_interp_crop[index]
						yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
						if (np.sum(yy>0)==0 or np.sum(yy_sigma>0)==0):
							averaged_profile_sigma_2.append(0)
							averaged_profile_sigma_2_sigma.append(1)
							continue
						yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
						yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
						p0 = [np.max(yy), mean_plasma_radius, 0]
						bds = [[0, 0, -interp_range_r/1000], [np.inf, np.max(r_crop), interp_range_r/1000]]
						fit = curve_fit(gauss, r_crop, yy, p0, maxfev=100000, bounds=bds, sigma=yy_sigma,absolute_sigma=True)
						averaged_profile_sigma_2.append(fit[0][-2])
						averaged_profile_sigma_2_sigma.append(fit[1][-2,-2])
					# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)
					averaged_profile_sigma_2 = np.array(averaged_profile_sigma_2)
					averaged_profile_sigma_2_sigma = np.array(averaged_profile_sigma_2_sigma)**0.5
					mean_plasma_radius_2 = np.sum(averaged_profile_sigma_2/averaged_profile_sigma_2_sigma)/np.sum(1/averaged_profile_sigma_2_sigma)
					source_power_spread = (mean_plasma_radius+mean_plasma_radius_2)*1000/4


					# x_local = xx - spatial_factor * 17.4 / 1000
					dr_crop = np.median(np.diff(r_crop))

					# This is kind of deprecated, but I don't remove it for fear of breaking the code
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
					if include_particles_limitation:
						if H2_suppression == True:
							mod3 = 'only_Hm_H2p_mol_lim'
						else:
							mod3 = 'only_Hm_H2_H2p_mol_lim'
					else:
						mod3 = 'only_Hm_H2_H2p'
					mod4 = mod2 +'/' +mod3

					if not os.path.exists(path_where_to_save_everything + mod4):
						os.makedirs(path_where_to_save_everything + mod4)

					print('Starting '+mod3)

					n_list_all = np.sort(np.concatenate((n_list, n_list_1)))
					# OES_multiplier = 0.81414701
					# Te_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
					# ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
					nH_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
					nHp_ne_all = np.ones_like(merge_ne_prof_multipulse_interp_crop_limited)
					nHm_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
					nH2_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
					nH2p_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
					nH3p_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
					Te_all = cp.deepcopy(merge_Te_prof_multipulse_interp_crop_limited)
					ne_all = cp.deepcopy(merge_ne_prof_multipulse_interp_crop_limited)
					sigma_Te_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
					sigma_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
					sigma_nH_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
					sigma_nHp_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
					sigma_nHm_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
					sigma_nH2_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
					sigma_nH2p_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
					sigma_nH3p_ne_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
					power_balance_data_dict = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited).flatten().tolist()

					residuals_all = np.zeros_like(merge_ne_prof_multipulse_interp_crop_limited)
					# guess = [0, 1, 0.01 * max_nHm_ne, 0.1 * max_nH2_ne, 0.0001 * max_nH2p_ne,0.0001 * max_nH3p_ne]
					guess = [0.5, 1, 0.0005,0.0005, 0.0001,0.0005]


					# this is to estimate better the background max H2 density
					# I assume the flow velocity from CTS (Jonathan) of 10km/s
					T_Hp = np.min([np.max([1000*np.ones_like(merge_Te_prof_multipulse_interp_crop_limited),merge_Te_prof_multipulse_interp_crop_limited/eV_to_K],axis=0),12000*np.ones_like(merge_Te_prof_multipulse_interp_crop_limited)],axis=0)	# K
					# area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))	# m^2
					temp = np.pi*((r_crop + np.median(np.diff(r_crop))/2)**2)
					area = np.array([temp[0]]+np.diff(temp).tolist())	# m^2
					ionisation_potential = 13.6	# eV
					heat_inflow_upstream = np.sum(area * merge_ne_prof_multipulse_interp_crop_limited * 10000*(ionisation_potential+ T_Hp*eV_to_K +merge_Te_prof_multipulse_interp_crop_limited)/J_to_eV,axis=1)* 1e20	# W
					sound_speed = (merge_Te_prof_multipulse_interp_crop_limited/eV_to_K*boltzmann_constant_J/(hydrogen_mass))**0.5	# m/s
					sheat_transmission_factor = 7.7	# from T W Morgan et al 2014
					heat_flux_target = np.sum(area *merge_ne_prof_multipulse_interp_crop_limited*1e20*sound_speed *(ionisation_potential/J_to_eV+sheat_transmission_factor*merge_Te_prof_multipulse_interp_crop_limited/eV_to_K*boltzmann_constant_J),axis=1)	# W
					neat_background_heating = heat_inflow_upstream - heat_flux_target
					neat_background_heating[neat_background_heating<0] = 0
					net_power_in = neat_background_heating*np.median(np.diff(time_crop))/1000	# J
					volume = (0.351+target_OES_distance/1000)*np.pi*(0.4**2)	# m^2
					H2_mass = target_chamber_pressure*2*hydrogen_mass/(boltzmann_constant_J*300)*volume	# kg
					heat_capacity_H2 = 14.5	# kJ/(kg K)
					temp_increase = net_power_in/(heat_capacity_H2*H2_mass)
					temp_H2 = np.ones_like(net_power_in)*300
					# feed_H2 = 19.8	# SLM
					feed_H2_kg = feed_rate_SLM/1000*101325/(boltzmann_constant_J*300)*2*hydrogen_mass/60	# kg/s
					for index in range(len(net_power_in)):
						if index==0:
							H2_mass = target_chamber_pressure*2*hydrogen_mass/(boltzmann_constant_J*300)*volume	# kg
						else:
							H2_mass = target_chamber_pressure*2*hydrogen_mass/(boltzmann_constant_J*temp_H2[index-1])*volume	# kg
						# flow_out = ( feed_H2_kg + np.sum(area * merge_ne_prof_multipulse_interp_crop_limited[index] * 10000)* 1e20*hydrogen_mass )*np.median(np.diff(time_crop))/1000	# kg
						flow_out = ( feed_rate_SLM /1000/60 + np.sum(area * 10000) )*np.median(np.diff(time_crop))/1000 *target_chamber_pressure/(boltzmann_constant_J*temp_H2[index-1])*2*hydrogen_mass	# kg
						power_balance = net_power_in[index] + feed_H2_kg*np.median(np.diff(time_crop))/1000*300*heat_capacity_H2 - flow_out*heat_capacity_H2*temp_H2[index-1]
						temp_H2[index]=min(max(temp_H2[index-1]+power_balance/(heat_capacity_H2*H2_mass),300),1/eV_to_K)
						# print([power_balance,flow_out,temp_H2[index]])
					max_nH2_from_pressure_all = target_chamber_pressure/(boltzmann_constant_J*temp_H2)
					max_nH2_from_pressure_all = 10*max_nH2_from_pressure_all	# as a safety factor

					initial_conditions = True
					global_pass = 1
					exec(open("/home/ffederic/work/Collaboratory/test/experimental_data/post_process_PSI_parameter_search_1_Yacora_plots_temp.py").read())
					initial_conditions = False

					class calc_stuff_output:
						def __init__(self, my_time_pos,my_r_pos, results):
							self.my_time_pos = my_time_pos
							self.my_r_pos = my_r_pos
							self.results = results

					sample_time_step = []
					for time in [0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 1, 1.4]:	#times in ms that I want to take a better look at
						sample_time_step.append(int((np.abs(time_crop - time)).argmin()))
					sample_time_step = np.unique(sample_time_step)
					sample_radious=[]
					for radious in [0, 0.004, 0.008, 0.012,0.018]:	#radious in m that I want to take a better look at
						sample_radious.append(int((np.abs(r_crop - radious)).argmin()))
					sample_radious = np.unique(sample_radious)

					def calc_power_balance_elements(Te_int,ne_int,nH_ne_int,nHm_ne_int,nH2p_ne_int,nH2_ne_int,T_Hp,T_Hm,T_H2p,T_Hp_prop_Te=True,T_Hm_prop_Te=True,T_H2p_prop_Te=True):
						# This is to calculate the power balance only

						if np.shape(Te_int)!=np.shape(ne_int):
							print('ERROR, shape of Te and ne must be same')
							exit()

						final_number_data = 1
						if not np.shape(Te_int)==():
							final_number_data = max(final_number_data,len(Te_int))
						# 	Te_int = Te_int * np.ones_like(nH_ne_int)
						# if np.shape(ne_int)==():
						# 	ne_int = ne_int * np.ones_like(nH_ne_int)
						# if np.shape(T_Hp)==():
						# 	T_Hp = T_Hp * np.ones_like(nH_ne_int)
						# if np.shape(T_Hm)==():
						# 	T_Hm = T_Hm * np.ones_like(nH_ne_int)
						# if np.shape(T_H2p)==():
						# 	T_H2p = T_H2p * np.ones_like(nH_ne_int)
						if not np.shape(nH_ne_int)==():
							final_number_data = max(final_number_data,len(nH_ne_int))
						if not np.shape(nHm_ne_int)==():
							final_number_data = max(final_number_data,len(nHm_ne_int))
						if not np.shape(nH2p_ne_int)==():
							final_number_data = max(final_number_data,len(nH2p_ne_int))
						if not np.shape(nH2_ne_int)==():
							final_number_data = max(final_number_data,len(nH2_ne_int))

						if np.shape(Te_int)==():
							excitation_full = []
							for isel in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
								if isel==0:
									temp = read_adf15(pecfile_2, 1, Te_int,ne_int * 10 ** (0 - 6))[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
								else:
									temp = read_adf15(pecfile, isel, Te_int,ne_int * 10 ** (0 - 6)) [0]  # ADAS database is in cm^3   # photons s^-1 cm^
								excitation_full.append(temp)
							excitation_full = np.array(excitation_full)  # in # photons cm^-3 s^-1
							excitation_full = (excitation_full.T * (10 ** -6) * (energy_difference_full / J_to_eV)).T  # in W m^-3 / (# / m^3)**2
							# excitation_full = (np.ones((len(nH_ne_int),len(excitation_full)))*excitation_full).T

							recombination_full = []
							for isel in [0, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
								if isel==0:
									temp = read_adf15(pecfile_2, 67, Te_int,ne_int * 10 ** (0 - 6))[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
								else:
									temp = read_adf15(pecfile, isel, Te_int,ne_int * 10 ** (0 - 6))[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
								recombination_full.append(temp)
							recombination_full = np.array(recombination_full)  # in # photons cm^-3 s^-1 / (# cm^-3)**2
							recombination_full = (recombination_full.T * (10 ** -6) * (energy_difference_full / J_to_eV)).T  # in W m^-3 / (# / m^3)**2
							# recombination_full = (np.ones((len(nH_ne_int),len(recombination_full)))*recombination_full).T
						else:
							unique_Te = np.unique(Te_int)
							unique_ne = np.unique(ne_int)
							sample_Te = (np.ones((len(unique_Te),len(unique_ne))).T*unique_Te).T
							sample_ne = np.ones((len(unique_Te),len(unique_ne)))*unique_ne

							excitation_full = []
							for isel in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
								if isel==0:
									temp = read_adf15(pecfile_2, 1, sample_Te.flatten(),sample_ne.flatten() * 10 ** (0 - 6))[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
								else:
									temp = read_adf15(pecfile, isel, sample_Te.flatten(),sample_ne.flatten() * 10 ** (0 - 6)) [0]  # ADAS database is in cm^3   # photons s^-1 cm^
								excitation_full.append(temp)
							excitation_full = np.array(excitation_full)  # in # photons cm^-3 s^-1
							excitation_full = (excitation_full.T * (10 ** -6) * (energy_difference_full / J_to_eV)).T  # in W m^-3 / (# / m^3)**2
							# excitation_full = (np.ones((len(nH_ne_int),len(excitation_full)))*excitation_full).T
							excitation_full = excitation_full.reshape((len(excitation_full),*np.shape(sample_Te)))

							temp = np.zeros((len(excitation_full),len(Te_int)))
							for i_Te,Te in enumerate(unique_Te):
								for i_ne,ne in enumerate(unique_ne):
									temp[:,np.logical_and(Te_int==Te,ne_int==ne)]=(np.ones_like(temp[:,np.logical_and(Te_int==Te,ne_int==ne)]).T*excitation_full[:,i_Te,i_ne]).T
							excitation_full = cp.deepcopy(temp)

							recombination_full = []
							for isel in [0, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
								if isel==0:
									temp = read_adf15(pecfile_2, 67, sample_Te.flatten(),sample_ne.flatten() * 10 ** (0 - 6))[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
								else:
									temp = read_adf15(pecfile, isel, sample_Te.flatten(),sample_ne.flatten() * 10 ** (0 - 6))[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
								recombination_full.append(temp)
							recombination_full = np.array(recombination_full)  # in # photons cm^-3 s^-1 / (# cm^-3)**2
							recombination_full = (recombination_full.T * (10 ** -6) * (energy_difference_full / J_to_eV)).T  # in W m^-3 / (# / m^3)**2
							# recombination_full = (np.ones((len(nH_ne_int),len(recombination_full)))*recombination_full).T
							recombination_full = recombination_full.reshape((len(recombination_full),*np.shape(sample_Te)))

							temp = np.zeros((len(recombination_full),len(Te_int)))
							for i_Te,Te in enumerate(unique_Te):
								for i_ne,ne in enumerate(unique_ne):
									temp[:,np.logical_and(Te_int==Te,ne_int==ne)]=(np.ones_like(temp[:,np.logical_and(Te_int==Te,ne_int==ne)]).T*recombination_full[:,i_Te,i_ne]).T
							recombination_full = cp.deepcopy(temp)


						nHp_ne_int = 1 + nHm_ne_int - nH2p_ne_int
						if (np.shape(Te_int)==()) and (not np.shape(nH_ne_int)==()):
							population_coefficients = (((np.ones((len(nH_ne_int),len(excitation_full)))*excitation_full.flatten()).T *  nH_ne_int).T /multiplicative_factor_full).T
						else:
							population_coefficients = ((excitation_full *  nH_ne_int).T /multiplicative_factor_full).T

						if (np.shape(Te_int)==()) and (not np.shape(nHp_ne_int)==()):
							temp = (((np.ones((len(nHp_ne_int),len(recombination_full)))*recombination_full.flatten()).T * nHp_ne_int).T /multiplicative_factor_full).T
						else:
							temp = ((recombination_full * nHp_ne_int).T /multiplicative_factor_full).T

						if np.shape(population_coefficients)[-1]>np.shape(temp)[-1]:
							population_coefficients += (np.ones_like(population_coefficients).T * (temp.flatten())).T
						elif np.shape(population_coefficients)[-1]<np.shape(temp)[-1]:
							population_coefficients = (np.ones_like(temp).T * (population_coefficients.flatten())).T + temp
						else:
							population_coefficients += temp

						if np.shape(population_coefficients)[-1]<final_number_data:
							population_coefficients = (np.ones((final_number_data,len(population_coefficients)))* (population_coefficients.flatten())).T
						population_states_atoms = population_coefficients* ne_int**2

						if np.shape(Te_int)==() and final_number_data>1:
							Te_int = Te_int * np.ones((final_number_data))
						if np.shape(ne_int)==() and final_number_data>1:
							ne_int = ne_int * np.ones((final_number_data))
						if np.shape(T_Hp)==() and final_number_data>1:
							T_Hp = T_Hp * np.ones((final_number_data))
						if np.shape(T_Hm)==() and final_number_data>1:
							T_Hm = T_Hm * np.ones((final_number_data))
						if np.shape(T_H2p)==() and final_number_data>1:
							T_H2p = T_H2p * np.ones((final_number_data))
						if np.shape(nH_ne_int)==() and final_number_data>1:
							nH_ne_int = nH_ne_int * np.ones((final_number_data))
						if np.shape(nHm_ne_int)==() and final_number_data>1:
							nHm_ne_int = nHm_ne_int * np.ones((final_number_data))
						if np.shape(nH2p_ne_int)==() and final_number_data>1:
							nH2p_ne_int = nH2p_ne_int * np.ones((final_number_data))
						if np.shape(nH2_ne_int)==() and final_number_data>1:
							nH2_ne_int = nH2_ne_int * np.ones((final_number_data))

						if final_number_data==1:
							temp1 = From_Hn_with_Hp_pop_coeff_full_extra(np.array([[Te_int,T_Hp,T_Hm,ne_int,nHp_ne_int*ne_int]]),np.unique(excited_states_From_Hn_with_Hp))
							temp2 = From_Hn_with_H2p_pop_coeff_full_extra(np.array([[Te_int,T_H2p,T_Hm,ne_int,nH2p_ne_int*ne_int]]),np.unique(excited_states_From_Hn_with_H2p))
							population_coefficients_Hm = (nHm_ne_int*( temp1 + temp2 ).T).reshape((np.shape(population_coefficients)))
							population_coefficients += (nHm_ne_int*( temp1 + temp2 ).T).reshape((np.shape(population_coefficients)))
							temp = From_H2_pop_coeff_full_extra(np.array([[Te_int,ne_int]]),np.unique(excited_states_From_H2))
							population_coefficients_H2 = (nH2_ne_int*temp.T).reshape((np.shape(population_coefficients)))
							population_coefficients += (nH2_ne_int*temp.T).reshape((np.shape(population_coefficients)))
							temp = From_H2p_pop_coeff_full_extra(np.array([[Te_int,ne_int]]),np.unique(excited_states_From_H2p))
							population_coefficients_H2p = (nH2p_ne_int*temp.T).reshape((np.shape(population_coefficients)))
							population_coefficients += (nH2p_ne_int*temp.T).reshape((np.shape(population_coefficients)))
							# temp = From_H3p_pop_coeff_full_extra(np.array([[Te_int,ne_int]]),np.unique(excited_states_From_H3p))
							# population_coefficients += (nH3p_ne_all.flatten()*temp.T).reshape((np.shape(population_coefficients)))
						else:
							unique_Te = np.unique(Te_int)
							unique_T_Hp = np.unique(T_Hp)
							unique_T_Hm = np.unique(T_Hm)
							unique_T_H2p = np.unique(T_H2p)
							unique_ne = np.unique(ne_int)
							unique_nHp_ne = np.unique(nHp_ne_int)
							unique_nH2p_ne = np.unique(nH2p_ne_int)
							if (T_Hp_prop_Te==True and T_Hm_prop_Te==True and T_H2p_prop_Te==True):
								sample = []
								for Te in unique_Te:
									select1 = Te_int==Te
									nHp_ne_int_mod = nHp_ne_int[select1]
									ne_int_mod = ne_int[select1]
									T_Hp_mod = T_Hp[Te_int==Te][0]
									T_Hm_mod = T_Hm[Te_int==Te][0]
									for ne in unique_ne:
										select2 = ne_int_mod==ne
										if np.sum(select2)==0:
											continue
										nHp_ne_int_mod_mod = nHp_ne_int_mod[select2]
										for nHp_ne in unique_nHp_ne:
											if np.sum(nHp_ne_int_mod_mod==nHp_ne)>0:
												sample.append([Te,T_Hp_mod,T_Hm_mod,ne,nHp_ne])
								sample_orig=np.array(sample)
								sample = cp.deepcopy(sample_orig)
								sample[:,-1] = sample[:,-1]*sample[:,-2]
								# sample_Te,sample_ne,sample_nHp_ne = np.meshgrid(unique_Te,unique_ne,unique_nHp_ne,indexing='ij')
								# sample_T_Hp = np.zeros_like(sample_Te)
								# sample_T_Hm = np.zeros_like(sample_Te)
								# for values in unique_Te:
								# 	sample_T_Hp[sample_Te==values]=T_Hp[Te_int==values][0]
								# 	sample_T_Hm[sample_Te==values]=T_Hm[Te_int==values][0]
								# temp1 = From_Hn_with_Hp_pop_coeff_full_extra(np.array([sample_Te.flatten(),sample_T_Hp.flatten(),sample_T_Hm.flatten(),sample_ne.flatten(),sample_nHp_ne.flatten()*sample_ne.flatten()]).T,np.unique(excited_states_From_Hn_with_Hp))
								temp1 = From_Hn_with_Hp_pop_coeff_full_extra(sample,np.unique(excited_states_From_Hn_with_Hp))
								temp1 = temp1.T
								# temp1 = temp1.reshape((len(temp1),*np.shape(sample_Te)))
								temp = np.zeros((len(temp1),len(Te_int)))
								Te_mod = 0
								ne_mod = 0
								for index in range(len(sample_orig)):
									if Te_mod!=sample_orig[index][0]:
										Te_mod = sample_orig[index][0]
										# print(Te_mod)
										select1 = Te_int==Te_mod
										# ne_int_mod = ne_int[select1]
									if ne_mod!=sample_orig[index][3]:
										ne_mod = sample_orig[index][3]
										# print(ne_mod)
										# select2 = cp.deepcopy(select1)
										# select2[select2] = ne_int_mod==ne_mod
										select2 = np.logical_and(select1,ne_int==ne_mod)
										# nHp_ne_int_mod = nHp_ne_int[select2]
									# select3 = cp.deepcopy(select2)
									# select3[select3] = nHp_ne_int_mod==sample_orig[index][4]
									# temp[:,select3] = (np.ones((len(temp1),np.sum(select3))).T*temp1[:,index]).T
									select3 = nHp_ne_int==sample_orig[index][4]
									select = np.logical_and(select3,select2)
									temp[:,select] = (np.ones((len(temp1),np.sum(select))).T*temp1[:,index]).T

								# temp = np.zeros((len(temp1),len(Te_int)))
								# for Te in unique_Te:
								# 	select1 = Te_int==Te
								# 	# nHp_ne_int_mod = nHp_ne_int[select1]
								# 	# ne_int_mod = ne_int[select1]
								# 	temp1_1 = temp1[:,sample_orig[:,0]==Te]
								# 	sample_orig_1 = sample_orig[sample_orig[:,0]==Te]
								# 	for ne in unique_ne:
								# 		select2 = np.logical_and(ne_int==ne,select1)
								# 		if np.sum(select2)>0:
								# 		# if np.sum(ne_int_mod==ne)>0:
								# 			# select2 = cp.deepcopy(select1)
								# 			# select2[select2] = select3 = ne_int_mod==ne
								# 			temp1_2 = temp1_1[:,sample_orig_1[:,3]==ne]
								# 			sample_orig_2 = sample_orig_1[sample_orig_1[:,3]==ne]
								# 			# nHp_ne_int_mod_mod = nHp_ne_int_mod[select3]
								# 			for nHp_ne in unique_nHp_ne:
								# 				select4 = np.logical_and(nHp_ne_int==nHp_ne,select2)
								# 				if np.sum(select4)>0:
								# 				# if np.sum(nHp_ne_int_mod_mod==nHp_ne)>0:
								# 					# select4 = cp.deepcopy(select2)
								# 					# select4[select4] = nHp_ne_int_mod_mod==nHp_ne
								# 					temp[:,select4] = (np.ones((len(temp1),np.sum(select4))).T*(temp1_2[:,sample_orig_2[:,4]==nHp_ne].flatten())).T
									#
									# for i_Te,Te in enumerate(unique_Te):
									# 	select1 = nHp_ne_int==value3
									# 	for i_ne,ne in enumerate(unique_ne):
									# 		# temp[:,np.logical_and(np.logical_and(Te_int==Te,ne_int==ne),nHp_ne_int==nHp_ne)]=(np.ones_like(temp[:,np.logical_and(np.logical_and(Te_int==Te,ne_int==ne),nHp_ne_int==nHp_ne)]).T*temp1[:,i_Te,i_ne,i_nHp_ne]).T
									# 		temp[:,np.logical_and(np.logical_and(Te_int==Te,ne_int==ne),nHp_ne_int==nHp_ne)]=(np.ones_like(temp[:,np.logical_and(np.logical_and(Te_int==Te,ne_int==ne),nHp_ne_int==nHp_ne)]).T*temp1[:,np.logical_and(np.logical_and(sample[:,0]==Te,sample[:,3]==ne),sample[:,4]==nHp_ne*ne)]).T
								population_coefficients += (nHm_ne_int*( temp )).reshape((np.shape(population_coefficients)))
								population_coefficients_Hm = (nHm_ne_int*( temp )).reshape((np.shape(population_coefficients)))

								sample = []
								for Te in unique_Te:
									select1 = Te_int==Te
									nH2p_ne_int_mod = nH2p_ne_int[select1]
									ne_int_mod = ne_int[select1]
									T_H2p_mod = T_H2p[Te_int==Te][0]
									T_Hm_mod = T_Hm[Te_int==Te][0]
									for ne in unique_ne:
										select2 = ne_int_mod==ne
										if np.sum(select2)==0:
											continue
										nH2p_ne_int_mod_mod = nH2p_ne_int_mod[select2]
										for nH2p_ne in unique_nH2p_ne:
											if np.sum(nH2p_ne_int_mod_mod==nH2p_ne)>0:
												sample.append([Te,T_H2p_mod,T_Hm_mod,ne,nH2p_ne])
								sample_orig=np.array(sample)
								sample = cp.deepcopy(sample_orig)
								sample[:,-1] = sample[:,-1]*sample[:,-2]
								# sample_Te,sample_ne,sample_nH2p_ne = np.meshgrid(unique_Te,unique_ne,unique_nH2p_ne,indexing='ij')
								# sample_T_Hm = np.zeros_like(sample_Te)
								# sample_T_H2p = np.zeros_like(sample_Te)
								# for values in unique_Te:
								# 	sample_T_Hm[sample_Te==values]=T_Hm[Te_int==values][0]
								# 	sample_T_H2p[sample_Te==values]=T_H2p[Te_int==values][0]
								# temp1 = From_Hn_with_H2p_pop_coeff_full_extra(np.array([sample_Te.flatten(),sample_T_H2p.flatten(),sample_T_Hm.flatten(),sample_ne.flatten(),sample_nH2p_ne.flatten()*sample_ne.flatten()]).T,np.unique(excited_states_From_Hn_with_H2p))
								temp1 = From_Hn_with_H2p_pop_coeff_full_extra(sample,np.unique(excited_states_From_Hn_with_H2p))
								temp1 = temp1.T
								# temp1 = temp1.reshape((len(temp1),*np.shape(sample_Te)))
								temp = np.zeros((len(temp1),len(Te_int)))
								Te_mod = 0
								ne_mod = 0
								for index in range(len(sample_orig)):
									if Te_mod!=sample_orig[index][0]:
										Te_mod = sample_orig[index][0]
										select1 = Te_int==Te_mod
										# ne_int_mod = ne_int[select1]
									if ne_mod!=sample_orig[index][3]:
										ne_mod = sample_orig[index][3]
										# select2 = cp.deepcopy(select1)
										# select2[select2] = ne_int_mod==ne_mod
										select2 = np.logical_and(select1,ne_int==ne_mod)
										# nHp_ne_int_mod = nHp_ne_int[select2]
									# select3 = cp.deepcopy(select2)
									# select3[select3] = nHp_ne_int_mod==sample_orig[index][4]
									# temp[:,select3] = (np.ones((len(temp1),np.sum(select3))).T*temp1[:,index]).T
									select3 = nH2p_ne_int==sample_orig[index][4]
									select = np.logical_and(select3,select2)
									temp[:,select] = (np.ones((len(temp1),np.sum(select))).T*temp1[:,index]).T

								# temp = np.zeros((len(temp1),len(Te_int)))
								# for Te in unique_Te:
								# 	select1 = Te_int==Te
								# 	nH2p_ne_int_mod = nH2p_ne_int[select1]
								# 	ne_int_mod = ne_int[select1]
								# 	temp1_1 = temp1[:,sample_orig[:,0]==Te]
								# 	sample_orig_1 = sample_orig[sample_orig[:,0]==Te]
								# 	for ne in unique_ne:
								# 		if np.sum(ne_int_mod==ne)>0:
								# 			select2 = cp.deepcopy(select1)
								# 			select2[select2] = select3 = ne_int_mod==ne
								# 			temp1_2 = temp1_1[:,sample_orig_1[:,3]==ne]
								# 			sample_orig_2 = sample_orig_1[sample_orig_1[:,3]==ne]
								# 			nH2p_ne_int_mod_mod = nH2p_ne_int_mod[select3]
								# 			for nH2p_ne in unique_nH2p_ne:
								# 				if np.sum(nH2p_ne_int_mod_mod==nH2p_ne)>0:
								# 					select4 = cp.deepcopy(select2)
								# 					select4[select4] = nH2p_ne_int_mod_mod==nH2p_ne
								# 					temp[:,select4] = (np.ones_like(temp[:,select4]).T*(temp1_2[:,sample_orig_2[:,4]==nH2p_ne].flatten())).T
								# for index,Te,trash1,trash2,ne,nH2p_ne in enumerate(sample_orig):
								# 	select1 = nH2p_ne_int==nH2p_ne
								# 	select2 = Te_int==Te
								# 	select3 = ne_int==ne
								# 	select = np.sum([select1,select2,select3],axis=0)
								# 	temp[:,select==3] = (np.ones_like(temp[:,select==3]).T*temp1[:,index]).T
								# for i_Te,Te in enumerate(unique_Te):
								# 	for i_ne,ne in enumerate(unique_ne):
								# 		for i_nH2p_ne,nH2p_ne in enumerate(unique_nH2p_ne):
								# 			temp[:,np.logical_and(np.logical_and(Te_int==Te,ne_int==ne),nH2p_ne_int==nH2p_ne)]=(np.ones_like(temp[:,np.logical_and(np.logical_and(Te_int==Te,ne_int==ne),nH2p_ne_int==nH2p_ne)]).T*temp1[:,i_Te,i_ne,i_nH2p_ne]).T
								population_coefficients += (nHm_ne_int*( temp )).reshape((np.shape(population_coefficients)))
								population_coefficients_Hm += (nHm_ne_int*( temp )).reshape((np.shape(population_coefficients)))
							else:
								print('calculating Hn_with_Hp and Hn_with_H2p for each input value, it can be extremely slow and simply not work')
								temp1 = From_Hn_with_Hp_pop_coeff_full_extra(np.array([Te_int,T_Hp,T_Hm,ne_int,nHp_ne_int*ne_int]).T,np.unique(excited_states_From_Hn_with_Hp))
								temp2 = From_Hn_with_H2p_pop_coeff_full_extra(np.array([Te_int,T_H2p,T_Hm,ne_int,nH2p_ne_int*ne_int]).T,np.unique(excited_states_From_Hn_with_H2p))
								population_coefficients += (nHm_ne_int*( temp1 + temp2 )).reshape((np.shape(population_coefficients)))
								population_coefficients_Hm = (nHm_ne_int*( temp1 + temp2 )).reshape((np.shape(population_coefficients)))

							unique_Te = np.unique(Te_int)
							unique_ne = np.unique(ne_int)
							sample_Te = (np.ones((len(unique_Te),len(unique_ne))).T*unique_Te).T
							sample_ne = np.ones((len(unique_Te),len(unique_ne)))*unique_ne

							temp1 = From_H2_pop_coeff_full_extra(np.array([sample_Te.flatten(),sample_ne.flatten()]).T,np.unique(excited_states_From_H2))
							temp1 = temp1.T
							temp1 = temp1.reshape((len(temp1),*np.shape(sample_Te)))
							temp = np.zeros((len(temp1),len(Te_int)))
							for i_Te,Te in enumerate(unique_Te):
								for i_ne,ne in enumerate(unique_ne):
									temp[:,np.logical_and(Te_int==Te,ne_int==ne)]=(np.ones_like(temp[:,np.logical_and(Te_int==Te,ne_int==ne)]).T*temp1[:,i_Te,i_ne]).T
							population_coefficients += (nH2_ne_int*temp).reshape((np.shape(population_coefficients)))
							population_coefficients_H2 = (nH2_ne_int*temp).reshape((np.shape(population_coefficients)))

							temp1 = From_H2p_pop_coeff_full_extra(np.array([sample_Te.flatten(),sample_ne.flatten()]).T,np.unique(excited_states_From_H2p))
							temp1 = temp1.T
							temp1 = temp1.reshape((len(temp1),*np.shape(sample_Te)))
							temp = np.zeros((len(temp1),len(Te_int)))
							for i_Te,Te in enumerate(unique_Te):
								for i_ne,ne in enumerate(unique_ne):
									temp[:,np.logical_and(Te_int==Te,ne_int==ne)]=(np.ones_like(temp[:,np.logical_and(Te_int==Te,ne_int==ne)]).T*temp1[:,i_Te,i_ne]).T
							population_coefficients += (nH2p_ne_int*temp).reshape((np.shape(population_coefficients)))
							population_coefficients_H2p = (nH2p_ne_int*temp).reshape((np.shape(population_coefficients)))

							# temp1 = From_H3p_pop_coeff_full_extra(np.array([sample_Te.flatten(),sample_ne.flatten()]).T,np.unique(excited_states_From_H3p))
							# temp1 = temp1.T
							# temp1 = temp1.reshape((len(temp1),*np.shape(sample_Te)))
							# temp = np.zeros((len(temp1),len(Te_int)))
							# for i_Te,Te in enumerate(unique_Te):
							# 	for i_ne,ne in enumerate(unique_ne):
							# 		temp[:,np.logical_and(Te_int==Te,ne_int==ne)]=(np.ones_like(temp[:,np.logical_and(Te_int==Te,ne_int==ne)]).T*temp1[:,i_Te,i_ne]).T
							# population_coefficients += (nH3p_ne_int*temp).reshape((np.shape(population_coefficients)))
						population_states = population_coefficients * (ne_int**2)
						population_states_molecules = population_states - population_states_atoms
						power_rad_mol = np.sum((population_states_molecules.T * multiplicative_factor_full_full).T,axis=0)
						temp = population_coefficients_Hm * (ne_int**2)
						power_rad_Hm = np.sum((temp.T * multiplicative_factor_full_full).T,axis=0)
						temp = population_coefficients_H2 * (ne_int**2)
						power_rad_H2 = np.sum((temp.T * multiplicative_factor_full_full).T,axis=0)
						temp = population_coefficients_H2p * (ne_int**2)
						power_rad_H2p = np.sum((temp.T * multiplicative_factor_full_full).T,axis=0)

						if np.shape(Te_int)==():
							temp = read_adf11(pltfile, 'plt', 1, 1, 1, Te_int,ne_int * 10 ** (0 - 6))
							power_rad_excit = temp * (ne_int**2) *nH_ne_int * (10 ** -6)
						else:
							unique_Te = np.unique(Te_int)
							unique_ne = np.unique(ne_int)
							sample_Te = (np.ones((len(unique_Te),len(unique_ne))).T*unique_Te).T
							sample_ne = np.ones((len(unique_Te),len(unique_ne)))*unique_ne
							temp1 = read_adf11(pltfile, 'plt', 1, 1, 1, sample_Te.flatten(),sample_ne.flatten() * 10 ** (0 - 6))
							temp1 = temp1.reshape(np.shape(sample_Te))

							temp = np.zeros((len(Te_int)))
							for i_Te,Te in enumerate(unique_Te):
								for i_ne,ne in enumerate(unique_ne):
									temp[np.logical_and(Te_int==Te,ne_int==ne)]=(np.ones_like(temp[np.logical_and(Te_int==Te,ne_int==ne)])*temp1[i_Te,i_ne]).T
							temp1 = cp.deepcopy(temp)
							power_rad_excit = temp1 * (ne_int**2) *nH_ne_int * (10 ** -6)

						if np.shape(Te_int)==():
							temp = read_adf11(prbfile, 'prb', 1, 1, 1, Te_int,ne_int * 10 ** (0 - 6))
							power_rad_rec_bremm = temp * (ne_int**2) *nHp_ne_int * (10 ** -6)
						else:
							unique_Te = np.unique(Te_int)
							unique_ne = np.unique(ne_int)
							sample_Te = (np.ones((len(unique_Te),len(unique_ne))).T*unique_Te).T
							sample_ne = np.ones((len(unique_Te),len(unique_ne)))*unique_ne
							temp1 = read_adf11(prbfile, 'prb', 1, 1, 1, sample_Te.flatten(),sample_ne.flatten() * 10 ** (0 - 6))
							temp1 = temp1.reshape(np.shape(sample_Te))

							temp = np.zeros((len(Te_int)))
							for i_Te,Te in enumerate(unique_Te):
								for i_ne,ne in enumerate(unique_ne):
									temp[np.logical_and(Te_int==Te,ne_int==ne)]=(np.ones_like(temp[np.logical_and(Te_int==Te,ne_int==ne)])*temp1[i_Te,i_ne]).T
							temp1 = cp.deepcopy(temp)
							power_rad_rec_bremm = temp1 * (ne_int**2) *nHp_ne_int * (10 ** -6)

						if np.shape(Te_int)==():
							temp = read_adf11(scdfile, 'scd', 1, 1, 1, Te_int,ne_int * 10 ** (0 - 6))
							effective_ionisation_rates = temp * (ne_int**2) * nH_ne_int  * (10 ** -6)
							power_via_ionisation = effective_ionisation_rates*13.6/J_to_eV
						else:
							unique_Te = np.unique(Te_int)
							unique_ne = np.unique(ne_int)
							sample_Te = (np.ones((len(unique_Te),len(unique_ne))).T*unique_Te).T
							sample_ne = np.ones((len(unique_Te),len(unique_ne)))*unique_ne
							temp1 = read_adf11(scdfile, 'scd', 1, 1, 1, sample_Te.flatten(),sample_ne.flatten() * 10 ** (0 - 6))
							temp1 = temp1.reshape(np.shape(sample_Te))

							temp = np.zeros((len(Te_int)))
							for i_Te,Te in enumerate(unique_Te):
								for i_ne,ne in enumerate(unique_ne):
									temp[np.logical_and(Te_int==Te,ne_int==ne)]=(np.ones_like(temp[np.logical_and(Te_int==Te,ne_int==ne)])*temp1[i_Te,i_ne]).T
							temp1 = cp.deepcopy(temp)
							effective_ionisation_rates = temp1 * (ne_int**2) * nH_ne_int  * (10 ** -6)
							power_via_ionisation = effective_ionisation_rates*13.6/J_to_eV

						if np.shape(Te_int)==():
							temp = read_adf11(acdfile, 'acd', 1, 1, 1, Te_int,ne_int * 10 ** (0 - 6))
							effective_recombination_rates = temp *nHp_ne_int* (ne_int**2)* (10 ** -6)
							power_via_recombination = effective_recombination_rates*13.6/J_to_eV
						else:
							unique_Te = np.unique(Te_int)
							unique_ne = np.unique(ne_int)
							sample_Te = (np.ones((len(unique_Te),len(unique_ne))).T*unique_Te).T
							sample_ne = np.ones((len(unique_Te),len(unique_ne)))*unique_ne
							temp1 = read_adf11(acdfile, 'acd', 1, 1, 1, sample_Te.flatten(),sample_ne.flatten() * 10 ** (0 - 6))
							temp1 = temp1.reshape(np.shape(sample_Te))

							temp = np.zeros((len(Te_int)))
							for i_Te,Te in enumerate(unique_Te):
								for i_ne,ne in enumerate(unique_ne):
									temp[np.logical_and(Te_int==Te,ne_int==ne)]=(np.ones_like(temp[np.logical_and(Te_int==Te,ne_int==ne)])*temp1[i_Te,i_ne]).T
							temp1 = cp.deepcopy(temp)
							effective_recombination_rates = temp1 *nHp_ne_int* (ne_int**2)* (10 ** -6)
							power_via_recombination = effective_recombination_rates*13.6/J_to_eV

						if final_number_data==1:
							return np.array([power_rad_excit,power_rad_rec_bremm,power_rad_mol,power_via_ionisation,power_via_recombination,power_rad_Hm,power_rad_H2,power_rad_H2p]).flatten()
						else:
							return power_rad_excit,power_rad_rec_bremm,power_rad_mol,power_via_ionisation,power_via_recombination,power_rad_Hm,power_rad_H2,power_rad_H2p

					def calc_power_balance_elements_simplified2(H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps,Te_values,ne_values,multiplicative_factor_full_full,multiplicative_factor_visible_light_full_full,record_nH_ne_values,record_nHm_ne_values,record_nH2_ne_values,record_nH2p_ne_values,record_nHp_ne_values,T_Hp_values,T_H2p_values,T_Hm_values,coeff_1_record,coeff_2_record,coeff_3_record,coeff_4_record):
						total_wavelengths = np.unique(excited_states_From_Hn_with_Hp)
						power_rad_H2p = np.zeros((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps))	# H, Hm, H2, H2p, ne, Te
						coeff_1 = coeff_1_record.reshape((TS_ne_steps,TS_Te_steps,len(total_wavelengths)))
						power_rad_H2p = np.zeros((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps))	# H, Hm, H2, H2p, ne, Te
						nH_ne_excited_states_mol = power_rad_H2p + np.sum((coeff_1),axis=-1) * (ne_values)
						nH_ne_excited_state_mol_2 = power_rad_H2p + coeff_1[:,:,0]* (ne_values)
						nH_ne_excited_state_mol_3 = power_rad_H2p + coeff_1[:,:,1]* (ne_values)
						nH_ne_excited_state_mol_4 = power_rad_H2p + coeff_1[:,:,2]* (ne_values)
						power_rad_H2p_visible = power_rad_H2p + np.sum((coeff_1 * multiplicative_factor_visible_light_full_full),axis=-1) * (ne_values**2)
						power_rad_H2p = power_rad_H2p + np.sum((coeff_1 * multiplicative_factor_full_full),axis=-1) * (ne_values**2)
						power_rad_H2p_visible = np.float32(power_rad_H2p_visible * (record_nH2p_ne_values.reshape(H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps)))	# H, Hm, H2, H2p, ne, Te
						power_rad_H2p = np.float32(power_rad_H2p * (record_nH2p_ne_values.reshape(H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps)))	# H, Hm, H2, H2p, ne, Te
						nH_ne_excited_states_mol = np.float32(nH_ne_excited_states_mol * (record_nH2p_ne_values.reshape(H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps)))	# H, Hm, H2, H2p, ne, Te
						nH_ne_excited_state_mol_2 = np.float32(nH_ne_excited_state_mol_2 * (record_nH2p_ne_values.reshape(H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps)))	# H, Hm, H2, H2p, ne, Te
						nH_ne_excited_state_mol_3 = np.float32(nH_ne_excited_state_mol_3 * (record_nH2p_ne_values.reshape(H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps)))	# H, Hm, H2, H2p, ne, Te
						nH_ne_excited_state_mol_4 = np.float32(nH_ne_excited_state_mol_4 * (record_nH2p_ne_values.reshape(H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps)))	# H, Hm, H2, H2p, ne, Te
						# coeff_1 = coeff_1.reshape((*np.shape(Te_values),len(np.unique(excited_states_From_H2p))))
						coeff_2 = coeff_2_record.reshape((TS_ne_steps,TS_Te_steps,len(total_wavelengths)))
						power_rad_H2 = np.zeros((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps))	# H, Hm, H2, H2p, ne, Te
						temp = power_rad_H2 + np.sum((coeff_2),axis=-1) * (ne_values)
						temp = np.float32(temp * np.transpose([(record_nH2_ne_values.reshape(H2_steps,TS_ne_steps,TS_Te_steps)).tolist()]*(H2p_steps), (1,0,2,3)))	# H, Hm, H2, H2p, ne, Te
						nH_ne_excited_states_mol += temp
						temp = power_rad_H2 + coeff_2[:,:,0] * (ne_values)
						temp = np.float32(temp * np.transpose([(record_nH2_ne_values.reshape(H2_steps,TS_ne_steps,TS_Te_steps)).tolist()]*(H2p_steps), (1,0,2,3)))	# H, Hm, H2, H2p, ne, Te
						nH_ne_excited_state_mol_2 += temp
						temp = power_rad_H2 + coeff_2[:,:,1] * (ne_values)
						temp = np.float32(temp * np.transpose([(record_nH2_ne_values.reshape(H2_steps,TS_ne_steps,TS_Te_steps)).tolist()]*(H2p_steps), (1,0,2,3)))	# H, Hm, H2, H2p, ne, Te
						nH_ne_excited_state_mol_3 += temp
						temp = power_rad_H2 + coeff_2[:,:,2] * (ne_values)
						temp = np.float32(temp * np.transpose([(record_nH2_ne_values.reshape(H2_steps,TS_ne_steps,TS_Te_steps)).tolist()]*(H2p_steps), (1,0,2,3)))	# H, Hm, H2, H2p, ne, Te
						nH_ne_excited_state_mol_4 += temp
						power_rad_H2_visible = power_rad_H2 + np.sum((coeff_2 * multiplicative_factor_visible_light_full_full),axis=-1) * (ne_values**2)
						power_rad_H2 = power_rad_H2 + np.sum((coeff_2 * multiplicative_factor_full_full),axis=-1) * (ne_values**2)
						power_rad_H2_visible = np.float32(power_rad_H2_visible * np.transpose([(record_nH2_ne_values.reshape(H2_steps,TS_ne_steps,TS_Te_steps)).tolist()]*(H2p_steps), (1,0,2,3)))	# H, Hm, H2, H2p, ne, Te
						power_rad_H2 = np.float32(power_rad_H2 * np.transpose([(record_nH2_ne_values.reshape(H2_steps,TS_ne_steps,TS_Te_steps)).tolist()]*(H2p_steps), (1,0,2,3)))	# H, Hm, H2, H2p, ne, Te
						power_rad_excit = np.zeros((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps))	# H, Hm, H2, H2p, ne, Te
						total_power_rad_excit = (read_adf11(pltfile, 'plt', 1, 1, 1, Te_values.flatten(),ne_values.flatten() * 10 ** (0 - 6)) * (10 ** -6) * (ne_values.flatten()**2)).reshape((TS_ne_steps,TS_Te_steps))
						power_rad_excit = power_rad_excit + total_power_rad_excit
						power_rad_excit = np.float32(np.transpose(power_rad_excit, (1,2,3,0,4,5)) * record_nH_ne_values.reshape((H_steps,TS_ne_steps,TS_Te_steps)))	# Hm, H2, H2p, H, ne, Te
						power_rad_excit = np.transpose(power_rad_excit, (3,0,1,2,4,5))	# H, Hm, H2, H2p, ne, Te
						power_via_ionisation = np.zeros((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps))	# H, Hm, H2, H2p, ne, Te
						total_power_via_ionisation = (read_adf11(scdfile, 'scd', 1, 1, 1, Te_values.flatten(),ne_values.flatten() * 10 ** (0 - 6)) * (10 ** -6) * (ne_values.flatten()**2) *13.6/J_to_eV).reshape((TS_ne_steps,TS_Te_steps))
						power_via_ionisation = power_via_ionisation + total_power_via_ionisation
						power_via_ionisation = np.float32(np.transpose(power_via_ionisation, (1,2,3,0,4,5)) * record_nH_ne_values.reshape((H_steps,TS_ne_steps,TS_Te_steps)))	# Hm, H2, H2p, H, ne, Te
						power_via_ionisation = np.transpose(power_via_ionisation, (3,0,1,2,4,5))	# H, Hm, H2, H2p, ne, Te
						total_power_rad_rec_bremm = (read_adf11(prbfile, 'prb', 1, 1, 1, Te_values.flatten(),ne_values.flatten() * 10 ** (0 - 6)) * (10 ** -6) * (ne_values.flatten()**2)).reshape((TS_ne_steps,TS_Te_steps))
						power_rad_rec_bremm = np.float32([(total_power_rad_rec_bremm * record_nHp_ne_values.reshape((Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps))).tolist()]*(H_steps))
						total_power_via_recombination = (read_adf11(acdfile, 'acd', 1, 1, 1, Te_values.flatten(),ne_values.flatten() * 10 ** (0 - 6)) * (10 ** -6) * (ne_values.flatten()**2) *13.6/J_to_eV).reshape((TS_ne_steps,TS_Te_steps))
						power_via_recombination = np.float32([(total_power_via_recombination * record_nHp_ne_values.reshape((Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps))).tolist()]*(H_steps))
						power_via_brem =  np.float32([(record_nHp_ne_values.reshape(Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps) * 5.35*1e-37 * (Te_values/1000)**0.5 * (ne_values**2)).tolist()]*H_steps)	# Wesson, Tokamaks
						power_rad_Hm_H2p = np.zeros((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps),dtype=np.float32)	# H, Hm, H2, H2p, ne, Te
						power_rad_Hm_Hp = np.zeros((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps),dtype=np.float32)	# H, Hm, H2, H2p, ne, Te
						power_rad_Hm_visible = np.zeros((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps),dtype=np.float32)	# H, Hm, H2, H2p, ne, Te
						power_rec_neutral = 3/2*(power_via_recombination/13.6*(T_Hp_values.reshape((TS_ne_steps,TS_Te_steps))*eV_to_K)).astype(np.float32)	# power removed from plasma column due to recombination by the kinetic energy of the neutral (assuming Te=TH+)

						for i4 in range(H2_steps):
							for i5 in range(H2p_steps):
								nH2p_ne_values = record_nH2p_ne_values[i4,i5]
								# coeff_4 = (From_Hn_with_H2p_pop_coeff_full_extra(np.array([Te_values.flatten(),T_H2p_values,T_Hm_values,ne_values.flatten(),nH2p_ne_values*ne_values.flatten()]).T,total_wavelengths)).reshape((TS_ne_steps,TS_Te_steps,len(total_wavelengths)))
								coeff_4 = coeff_4_record[i4*H2p_steps + i5].reshape((TS_ne_steps,TS_Te_steps,len(total_wavelengths)))		# From_Hn_with_H2p_pop_coeff_full_extra
								temp = np.sum((coeff_4),axis=-1)
								temp_2 = cp.deepcopy(coeff_4[:,:,0])	# I am forced to use cp.deepcopy because if not this lines would change coeff_4_record itself
								temp_3 = cp.deepcopy(coeff_4[:,:,1])
								temp_4 = cp.deepcopy(coeff_4[:,:,2])
								coeff_4_visible = np.sum((coeff_4 * multiplicative_factor_visible_light_full_full),axis=-1)*(ne_values**2)
								coeff_4 = np.sum((coeff_4 * multiplicative_factor_full_full),axis=-1)*(ne_values**2)
								for i3 in range(Hm_steps):
									nHm_ne_values = record_nHm_ne_values[i4,i3]
									nHp_ne_values = 1 - nH2p_ne_values + nHm_ne_values
									nHp_ne_good = nHp_ne_values>=0
									# nHp_ne_mid = nHp_ne_values==0
									# nHp_ne_bad = nHp_ne_values<0
									# nHp_ne_values[nHp_ne_mid]=1e-10
									# nHp_ne_values[nHp_ne_bad]=1e-10
									# temp = From_Hn_with_Hp_pop_coeff_full_extra(np.array([Te_values.flatten()[nHp_ne_good],T_Hp_values[nHp_ne_good],T_Hm_values[nHp_ne_good],ne_values.flatten()[nHp_ne_good],nHp_ne_values[nHp_ne_good]*(ne_values.flatten()[nHp_ne_good])]).T ,total_wavelengths)
									# coeff_3 = np.zeros((TS_ne_steps*TS_Te_steps,len(total_wavelengths)))
									# coeff_3[nHp_ne_good] = temp
									# coeff_3 = coeff_3.reshape((TS_ne_steps,TS_Te_steps,len(total_wavelengths)))
									# coeff_3 = (From_Hn_with_Hp_pop_coeff_full_extra(np.array([Te_values.flatten(),T_Hp_values,T_Hm_values,ne_values.flatten(),nHp_ne_values*(ne_values.flatten())]).T ,total_wavelengths)).reshape((TS_ne_steps,TS_Te_steps,len(total_wavelengths)))
									coeff_3 = coeff_3_record[i4*Hm_steps*H2p_steps + i5*Hm_steps + i3].reshape((TS_ne_steps,TS_Te_steps,len(total_wavelengths)))		# From_Hn_with_Hp_pop_coeff_full_extra
									temp += np.sum((coeff_3),axis=-1)
									temp_2 += coeff_3[:,:,0]
									temp_3 += coeff_3[:,:,1]
									temp_4 += coeff_3[:,:,2]
									coeff_3_visible = np.sum((coeff_3 * multiplicative_factor_visible_light_full_full),axis=-1) *(ne_values**2)
									coeff_3 = np.sum((coeff_3 * multiplicative_factor_full_full),axis=-1) *(ne_values**2)
									# coeff_3[(np.logical_and(nHp_ne_mid,nHp_ne_bad)).reshape((TS_ne_steps,TS_Te_steps))]=0
									# nHp_ne_good = np.logical_or(nHp_ne_good,nHp_ne_mid).reshape((TS_ne_steps,TS_Te_steps))
									nHp_ne_good = nHp_ne_good.reshape((TS_ne_steps,TS_Te_steps))
									power_rad_Hm_visible[:,i3,i4,i5,nHp_ne_good] = (nHm_ne_values.reshape((TS_ne_steps,TS_Te_steps))*( coeff_3_visible + coeff_4_visible )).astype(np.float32)[nHp_ne_good]
									power_rad_Hm_H2p[:,i3,i4,i5,nHp_ne_good] = (nHm_ne_values.reshape((TS_ne_steps,TS_Te_steps))*( coeff_4 )).astype(np.float32)[nHp_ne_good]
									power_rad_Hm_Hp[:,i3,i4,i5,nHp_ne_good] = (nHm_ne_values.reshape((TS_ne_steps,TS_Te_steps))*( coeff_3 )).astype(np.float32)[nHp_ne_good]
									nH_ne_excited_states_mol[:,i3,i4,i5] += (nHm_ne_values.reshape((TS_ne_steps,TS_Te_steps))*( temp * ne_values ))
									nH_ne_excited_state_mol_2[:,i3,i4,i5] += (nHm_ne_values.reshape((TS_ne_steps,TS_Te_steps))*( temp_2 * ne_values ))
									nH_ne_excited_state_mol_3[:,i3,i4,i5] += (nHm_ne_values.reshape((TS_ne_steps,TS_Te_steps))*( temp_3 * ne_values ))
									nH_ne_excited_state_mol_4[:,i3,i4,i5] += (nHm_ne_values.reshape((TS_ne_steps,TS_Te_steps))*( temp_4 * ne_values ))
						power_rad_mol = (power_rad_Hm_H2p + power_rad_Hm_Hp + power_rad_H2 + power_rad_H2p).astype(np.float32)
						power_rad_Hm = (power_rad_Hm_H2p + power_rad_Hm_Hp).astype(np.float32)
						power_rad_mol_visible = (power_rad_Hm_visible + power_rad_H2_visible + power_rad_H2p_visible).astype(np.float32)
						power_heating_rec = (power_via_recombination - power_rad_rec_bremm + power_via_brem).astype(np.float32)
						# power_heating_rec[power_heating_rec<0]=0
						tot_rad_power = (power_rad_excit+power_rad_rec_bremm+power_rad_mol).astype(np.float32)
						# total_removed_power_atomic = np.float32(power_via_ionisation + power_rad_excit + power_via_recombination + power_rec_neutral + power_via_brem)
						total_removed_power_atomic = np.float32(power_via_ionisation + power_rad_excit + power_rad_rec_bremm + power_rec_neutral)
						# total_removed_power = power_via_ionisation + power_rad_excit + power_via_recombination + power_rec_neutral + power_via_brem + power_rad_mol

						return power_rad_H2p,power_rad_H2,power_rad_excit,power_via_ionisation,power_rad_rec_bremm,power_via_recombination,power_via_brem,power_rec_neutral,power_rad_Hm_H2p,power_rad_Hm_Hp,power_rad_Hm,power_rad_mol,power_heating_rec,tot_rad_power,total_removed_power_atomic,nH_ne_excited_states_mol,nH_ne_excited_state_mol_2,nH_ne_excited_state_mol_3,nH_ne_excited_state_mol_4,power_rad_mol_visible

					def calc_stuff_2(arg):
						try:
							domain_index = arg[0]
							my_time_pos = arg[1]
							my_r_pos = arg[2]
							sgna = 0;print('worker '+str(current_process())+' marker '+str(sgna)+' '+str(domain_index)+' t index '+str(my_time_pos)+' r index '+str(my_r_pos))
							pass_index = arg[3]
							guess = arg[4]
							ionization_length_H = arg[5]
							# ionization_length_Hm = arg[6]
							ionization_length_H2 = arg[6]
							# ionization_length_H2p = arg[8]
							to_print = arg[7]

							inverted_profiles_crop_restrict = inverted_profiles_crop[my_time_pos, n_list_all - 4, my_r_pos].flatten()
							inverted_profiles_crop_sigma_restrict = inverted_profiles_sigma_crop[my_time_pos, n_list_all - 4, my_r_pos].flatten()
							merge_ne_prof_multipulse_interp_crop_limited_restrict = merge_ne_prof_multipulse_interp_crop[my_time_pos, my_r_pos]
							merge_dne_prof_multipulse_interp_crop_limited_restrict = max(merge_dne_prof_multipulse_interp_crop[my_time_pos, my_r_pos],0.1)
							merge_Te_prof_multipulse_interp_crop_limited_restrict = merge_Te_prof_multipulse_interp_crop[my_time_pos, my_r_pos]
							merge_dTe_prof_multipulse_interp_crop_limited_restrict = max(merge_dTe_prof_multipulse_interp_crop[my_time_pos, my_r_pos],0.1)
							recombination_restrict = recombination[n_list_all - 4, my_time_pos, my_r_pos].flatten()
							excitation_restrict = excitation[n_list_all - 4, my_time_pos, my_r_pos].flatten()
							max_frac_acceptable_deviation = max_frac_acceptable_deviation_ext
							total_wavelengths = np.unique(excited_states_From_Hn_with_Hp)

							if pass_index<=1:
								if (merge_ne_prof_multipulse_interp_crop_limited_restrict < 5e-07 or merge_Te_prof_multipulse_interp_crop_limited_restrict < 0.1):
									print('worker '+str(current_process())+' marker 999 '+str(domain_index)+' my_time_pos '+str(my_time_pos)+', my_r_pos '+str(my_r_pos)+' skipped ne %.3g, Te %.3g' %(merge_ne_prof_multipulse_interp_crop_limited_restrict,merge_Te_prof_multipulse_interp_crop_limited_restrict))
									results = [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,True]
									output = calc_stuff_output(my_time_pos,my_r_pos, results)
									return output
							else:
								if (merge_ne_prof_multipulse_interp_crop_limited_restrict == 0 or merge_Te_prof_multipulse_interp_crop_limited_restrict == 0):
									print('worker '+str(current_process())+' marker 999 '+str(domain_index)+' my_time_pos '+str(my_time_pos)+', my_r_pos '+str(my_r_pos)+' skipped ne %.3g, Te %.3g' %(merge_ne_prof_multipulse_interp_crop_limited_restrict,merge_Te_prof_multipulse_interp_crop_limited_restrict))
									results = [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,True]
									output = calc_stuff_output(my_time_pos,my_r_pos, results)
									return output

							start_time = tm.time()
							print('worker '+str(current_process())+' actually running '+str(domain_index))
							# if np.sum(inverted_profiles_crop_restrict == 0)>2:
							# 	continue

							if False:
								T_Hp = 12000	# K
								T_Hm = 12000	# K
								T_H2p = 5000	# K
							else:
								T_Hp = merge_Te_prof_multipulse_interp_crop_limited_restrict/eV_to_K	# K
								T_Hm = np.exp(TH2_fit_from_simulations(np.log(merge_Te_prof_multipulse_interp_crop_limited_restrict)))/eV_to_K	# K
								T_H2p = np.exp(TH2_fit_from_simulations(np.log(merge_Te_prof_multipulse_interp_crop_limited_restrict)))/eV_to_K	# K
							ne = merge_ne_prof_multipulse_interp_crop_limited_restrict * 1e20
							multiplicative_factor = energy_difference[n_list_all - 4] * einstein_coeff[n_list_all - 4] / J_to_eV
							if merge_ne_prof_multipulse_interp_crop_limited_restrict>2:
								min_mol_fraction = min_mol_fraction_ext/10
							else:
								min_mol_fraction = min_mol_fraction_ext

							# this is after analysis of Ray simulations with B2.5-Eunomia
							max_nH2_ne = np.max(nH2_ne_fit_from_simulations(merge_Te_prof_multipulse_interp_crop_limited_restrict))*20
							min_nH2_ne = np.min(nH2_ne_fit_from_simulations(merge_Te_prof_multipulse_interp_crop_limited_restrict))/5
							centre_nH2_ne = nH2_ne_fit_from_simulations(merge_Te_prof_multipulse_interp_crop_limited_restrict)
							# limit_H_H2_up = interpolate.interp1d([0.1,0.2,4],[100,20,20],fill_value='extrapolate')
							# limit_H_H2_down = interpolate.interp1d([0.1,2,4],[10,0.03,0.03],fill_value='extrapolate')
							# limit_H_H2_up = np.exp(limit_H_H2_up(np.log(min(merge_Te_prof_multipulse_interp_crop_limited_restrict-merge_dTe_prof_multipulse_interp_crop_limited_restrict,0.1))))
							# limit_H_H2_down = np.exp(limit_H_H2_down(np.log(merge_Te_prof_multipulse_interp_crop_limited_restrict+merge_dTe_prof_multipulse_interp_crop_limited_restrict)))
							# max_nH_ne = min(limit_H_H2_up-min_nH2_ne,20)
							# min_nH_ne = max(limit_H_H2_down-max_nH2_ne,3e-2)
							max_nH_ne = np.exp(limit_H_up(np.log(merge_Te_prof_multipulse_interp_crop_limited_restrict)))
							min_nH_ne = np.exp(limit_H_down(np.log(merge_Te_prof_multipulse_interp_crop_limited_restrict)))
							centre_nH_ne = nH_ne_fit_from_simulations(merge_Te_prof_multipulse_interp_crop_limited_restrict)


							# if pass_index>1:
							# 	if ionization_length_H2<1e-6:
							# 		max_nH2_from_pressure_int = ne/100000

							# TS_steps = 11
							# TS_Te_steps = 13	# 13, 15
							# TS_Te_steps_increase = 5
							# TS_ne_steps = 13	# 17, 19
							# TS_ne_steps_increase = 6
							TS_Te_steps = externally_provided_TS_Te_steps
							TS_Te_steps_increase = externally_provided_TS_Te_steps_increase
							TS_ne_steps = externally_provided_TS_ne_steps
							TS_ne_steps_increase = externally_provided_TS_ne_steps_increase
							if collect_power_PDF:
								Hm_to_find_steps = externally_provided_Hn_steps	# 25/21 no, it was 11
								H2p_to_find_steps = externally_provided_H2p_steps	# 25/21 no, it was 11
							else:
								Hm_to_find_steps = externally_provided_Hn_steps	# 25/21 no, it was 11
								H2p_to_find_steps = externally_provided_H2p_steps	# 25/21 no, it was 11
							H_steps = externally_provided_H_steps	# this MUST be odd		# 23/13, 15 no, it was 13
							H2_steps = externally_provided_H2_steps	# this MUST be odd		# 11/7 no, it was 13
							# probability = np.ones((TS_steps,TS_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps))
							# guessed_values = np.ones((TS_steps,TS_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps))
							# calculated_emission = np.ones((TS_steps,TS_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,len(n_list_all)))
							# Te_values = np.linspace(max(np.min(merge_Te_prof_multipulse_interp_crop[merge_Te_prof_multipulse_interp_crop>0])/2,merge_Te_prof_multipulse_interp_crop_limited_restrict-4*merge_dTe_prof_multipulse_interp_crop_limited_restrict),merge_Te_prof_multipulse_interp_crop_limited_restrict+4*merge_dTe_prof_multipulse_interp_crop_limited_restrict,TS_Te_steps)
							Te_values = np.linspace(max(merge_Te_prof_multipulse_interp_crop_limited_restrict-4*merge_dTe_prof_multipulse_interp_crop_limited_restrict,min(0.1,merge_Te_prof_multipulse_interp_crop_limited_restrict/2)),merge_Te_prof_multipulse_interp_crop_limited_restrict+4*merge_dTe_prof_multipulse_interp_crop_limited_restrict,TS_Te_steps)
							# if pass_index<=1:
							# 	Te_values[Te_values<0.1]=0.1
							# else:
							# Te_values[Te_values<np.min(merge_Te_prof_multipulse_interp_crop[merge_Te_prof_multipulse_interp_crop>0])/2]=np.min(merge_Te_prof_multipulse_interp_crop[merge_Te_prof_multipulse_interp_crop>0])/2
							Te_values_array = np.unique(Te_values)
							Te_values_array_initial = np.array([0])
							# gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
							# Te_probs = gauss(np.linspace(-1,+1,TS_steps),1/((2*np.pi)**0.5),1,0)
							# Te_probs = Te_probs/np.sum(Te_probs)
							# # Te_probs = np.ones((TS_steps,TS_steps))*Te_probs

							# ne_values = ne+np.linspace(-2,+6,TS_ne_steps)*merge_dne_prof_multipulse_interp_crop_limited_restrict*1e20
							# ne_values = np.logspace(np.log10(max(ne-2*merge_dne_prof_multipulse_interp_crop_limited_restrict*1e20,1e20*np.min(merge_ne_prof_multipulse_interp_crop[merge_ne_prof_multipulse_interp_crop>0])/10)),np.log10(ne+8*merge_dne_prof_multipulse_interp_crop_limited_restrict*1e20),num=TS_ne_steps)
							min_ne = max(ne-4*merge_dne_prof_multipulse_interp_crop_limited_restrict*1e20,min(0.1*1e20,ne/10))
							max_ne = ne+8*merge_dne_prof_multipulse_interp_crop_limited_restrict*1e20
							if False:	# if ne and min_ne are close it end up doing too few steps there, so I force it to be more regular
								steps_left = np.ceil(1.5*(TS_ne_steps-1)*(ne-min_ne)/(max_ne-min_ne))
								steps_right = np.ceil(1/1.5*(TS_ne_steps-1)*(max_ne-ne)/(max_ne-min_ne))
								ne_values = [*np.logspace(np.log10(min_ne),np.log10(ne),num=steps_left+1),*np.logspace(np.log10(ne),np.log10(max_ne),num=steps_right+1)]
							else:
								steps_left = np.ceil((TS_ne_steps-1)/2)
								steps_right = np.ceil(TS_ne_steps/2)	# the right range is always larger so 1 point more is fair
								ne_values = [*np.linspace(min_ne,ne,num=steps_left),*np.logspace(np.log10(ne),np.log10(max_ne),num=steps_right)]	# I try linear spacing on the left to not have too many points away from the centre of the gaussian
							# ne_values = np.logspace(np.log10(max(ne-2*merge_dne_prof_multipulse_interp_crop_limited_restrict*1e20,min(0.05*1e20,ne/2))),np.log10(ne+8*merge_dne_prof_multipulse_interp_crop_limited_restrict*1e20),num=TS_ne_steps)
							ne_values = np.unique(ne_values)
							ne_values_array_initial = np.array([0])
							# if pass_index<=1:
							# 	ne_values[ne_values<5e13]=5e13
							# else:
							# ne_values[ne_values<1e20*np.min(merge_ne_prof_multipulse_interp_crop[merge_ne_prof_multipulse_interp_crop>0])/4]=1e20*np.min(merge_ne_prof_multipulse_interp_crop[merge_ne_prof_multipulse_interp_crop>0])/4
							ne_values_array = np.unique(ne_values)
							# ne_probs = gauss(np.linspace(-1,+1,TS_steps),1/((2*np.pi)**0.5),1,0)
							# ne_probs = ne_probs/np.sum(ne_probs)
							# # ne_probs = (np.ones((TS_steps,TS_steps))*ne_probs).T
							if False:
								# nHp_ne = np.logspace(np.log10(max_nHp_ne),np.log10(max(min_nHp_ne,0.01)),num=to_find_steps)	# this is 1 - nH2p_ne + nHm_ne
								nH_ne = np.logspace(np.log10(max_n_neutrals_ne),np.log10(max(min_nH_ne,0.1)),num=to_find_steps)
								# nHpolecule_ne = np.logspace(np.log10(max_nmolecules_ne_low_ionisation_length),np.log10(max_nmolecules_ne_low_ionisation_length/1000),num=to_find_steps)
								# nH2_ne = np.logspace(np.log10(max_nH2_from_pressure_int/ne),np.log10(max_nH2_from_pressure_int/10000/ne),num=to_find_steps)
								nHpolecule_ne = np.logspace(np.log10(10*max_nmolecules_ne_low_ionisation_length/ne),np.log10(max_nmolecules_ne_low_ionisation_length/1000/ne),num=to_find_steps)
								nH2_ne = np.logspace(np.log10(10*max_nH2_from_pressure_int/ne),np.log10(1e-3*max_nH2_from_pressure_int/ne),num=to_find_steps_H2)
							if False:	# this is after analysis of Ray simulations with B2.5-Eunomia		2020/07/15 used functions in import_PECs_FF_2.py instead
								# nH2_ne_values = np.array([min_nH2_ne,(centre_nH2_ne+2*min_nH2_ne)/3,(2*centre_nH2_ne+min_nH2_ne)/3,centre_nH2_ne,(2*centre_nH2_ne+max_nH2_ne)/3,(centre_nH2_ne+2*max_nH2_ne)/3,max_nH2_ne])
								# # nH2_ne_probs = gauss(np.linspace(-2,+2,5),1/((2*np.pi)**0.5),1,0)
								# # nH2_ne_probs = nH2_ne_probs/np.sum(nH2_ne_probs)
								# nH2_ne_log_probs = -0.5*(np.linspace(-2,+2,7)**2)
								H2_left_interval = np.logspace(np.log10(min_nH2_ne),np.log10(centre_nH2_ne),num=(H2_steps+1)/2)[:-1]
								H2_right_interval = np.logspace(np.log10(centre_nH2_ne),np.log10(max_nH2_ne),num=(H2_steps+1)/2)[1:]
								nH2_ne_values = np.array(H2_left_interval.tolist() + [centre_nH2_ne] + H2_right_interval.tolist())
								H2_left_interval_log_probs = -0.5*((np.log10(H2_left_interval/centre_nH2_ne)/(np.log10(centre_nH2_ne/min_nH2_ne)/2))**2)	# centre_nH2_ne-min_nH2_ne is 2 sigma
								H2_right_interval_log_probs = -0.5*((np.log10(H2_right_interval/centre_nH2_ne)/(np.log10(centre_nH2_ne/max_nH2_ne)/2))**2)	# centre_nH2_ne-max_nH2_ne is 2 sigma
								nH2_ne_log_probs = np.array(H2_left_interval_log_probs.tolist() + [0] + H2_right_interval_log_probs.tolist())
								nH2_ne_log_probs = nH2_ne_log_probs -np.log(np.sum(np.exp(nH2_ne_log_probs)))	# normalisation for logarithmic probabilities
								# nH_ne_probs = gauss(np.linspace(-2,+2,5),1/((2*np.pi)**0.5),1,0)
								# nH_ne_probs = nH_ne_probs/np.sum(nH_ne_probs)

							# first 2 loops of evaluation, in order to improve the resolution in Te and ne
							PDF_matrix_shape = []
							for loop_index in [0,1]:
								TS_Te_steps = len(Te_values_array)
								TS_ne_steps = len(ne_values_array)

								Te_log_probs = -(0.5*(((Te_values_array-merge_Te_prof_multipulse_interp_crop_limited_restrict)/(merge_dTe_prof_multipulse_interp_crop_limited_restrict/2))**2))**1	# super gaussian order 1
								Te_log_probs = Te_log_probs -np.log(np.sum(np.exp(Te_log_probs)))	# normalisation for logarithmic probabilities
								ne_log_probs = -(0.5*(((ne_values_array-ne)/(merge_dne_prof_multipulse_interp_crop_limited_restrict*1e20/2))**2))**1	# super gaussian order 1
								ne_log_probs = ne_log_probs -np.log(np.sum(np.exp(ne_log_probs)))	# normalisation for logarithmic probabilities
								ne_values = (np.ones((TS_Te_steps,TS_ne_steps))*ne_values_array).T
								Te_values = np.ones((TS_ne_steps,TS_Te_steps))*Te_values_array

								# excitation_internal = []
								# for isel in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
								# 	temp = read_adf15(pecfile, isel, Te_values.flatten(),(ne_values * 10 ** (- 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
								# 	temp[np.isnan(temp)] = 0
								# 	# temp = temp.reshape((np.shape(Te_values)))
								# 	excitation_internal.append(temp)
								# excitation_internal = np.array(excitation_internal)  # in # photons cm^-3 s^-1
								# excitation_internal = (excitation_internal.T * (10 ** -6) * (energy_difference / J_to_eV))  # in W m^-3 / (# / m^3)**2
								# excitation_internal = (excitation_internal[:,n_list_all - 4] /multiplicative_factor)  # in m^-3 / (# / m^3)**2

								multiplicative_factor_full = energy_difference_full * einstein_coeff_full / J_to_eV
								excitation_full = []
								for isel in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
									if isel==0:
										temp = read_adf15(pecfile_2, 1, Te_values.flatten(),(ne_values * 10 ** (- 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
									else:
										temp = read_adf15(pecfile, isel, Te_values.flatten(),(ne_values * 10 ** (- 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
									temp[np.isnan(temp)] = 0
									temp[np.isinf(temp)] = 0
									# temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
									excitation_full.append(temp)
								del temp
								excitation_full = np.array(excitation_full)  # in # photons cm^-3 s^-1
								excitation_full = (excitation_full.T * (10 ** -6) * (energy_difference_full / J_to_eV))  # in W m^-3 / (# / m^3)**2
								excitation_full = (excitation_full /multiplicative_factor_full)  # in m^-3 / (# / m^3)**2
								excitation_internal = excitation_full[:,n_list_all - 2]

								# recombination_internal = []
								# for isel in [20, 21, 22, 23, 24, 25, 26, 27, 28]:
								# 	temp = read_adf15(pecfile, isel, Te_values.flatten(),(ne_values * 10 ** (- 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
								# 	temp[np.isnan(temp)] = 0
								# 	# temp = temp.reshape((np.shape(Te_values)))
								# 	recombination_internal.append(temp)
								# recombination_internal = np.array(recombination_internal)  # in # photons cm^-3 s^-1 / (# cm^-3)**2
								# recombination_internal = (recombination_internal.T * (10 ** -6) * (energy_difference / J_to_eV))  # in W m^-3 / (# / m^3)**2
								# recombination_internal = (recombination_internal[:,n_list_all - 4] /multiplicative_factor)  # in m^-3 / (# / m^3)**2

								recombination_full = []
								for isel in [0, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
									if isel==0:
										temp = read_adf15(pecfile_2, 67, Te_values.flatten(),(ne_values * 10 ** (- 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
									else:
										temp = read_adf15(pecfile, isel, Te_values.flatten(),(ne_values * 10 ** (- 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
									temp[np.isnan(temp)] = 0
									# temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
									recombination_full.append(temp)
								del temp
								recombination_full = np.array(recombination_full)  # in # photons cm^-3 s^-1 / (# cm^-3)**2
								recombination_full = (recombination_full.T * (10 ** -6) * (energy_difference_full / J_to_eV))  # in W m^-3 / (# / m^3)**2
								recombination_full = (recombination_full /multiplicative_factor_full)  # in # m^-3 / (# / m^3)**2
								recombination_internal = recombination_full[:,n_list_all - 2]

								total_wavelengths = np.unique(excited_states_From_Hn_with_Hp)
								# calculated_emission = np.ones((TS_steps,TS_steps,to_find_steps,to_find_steps,to_find_steps,to_find_steps,len(n_list_all)))
								calculated_emission_log_probs = -np.ones((H_steps,Hm_to_find_steps,H2_steps,H2p_to_find_steps,TS_ne_steps,TS_Te_steps),dtype=np.float32)*np.inf	# H, Hm, H2, H2p, ne, Te
								# calculated_emission_error = np.ones((to_find_steps,to_find_steps))	# Hm, H2p
								# nHp_ne_value = 1
								# total_nHp_ne_value = (((recombination_internal[n_list_all - 4] * nHp_ne_value*np.ones((TS_steps,TS_steps)) ).T/multiplicative_factor).T).astype('float').T.reshape((TS_steps*TS_steps,len(n_list_all)))
								# total_nHp_ne_value = recombination_internal * nHp_ne_value
								coeff_1_record = From_H2p_pop_coeff_full_extra(np.array([Te_values.flatten(),ne_values.flatten()]).T,total_wavelengths)
								coeff_1 = coeff_1_record[:,n_list_all-2]
								coeff_2_record = From_H2_pop_coeff_full_extra(np.array([Te_values.flatten(),ne_values.flatten()]).T,total_wavelengths)
								coeff_2 = coeff_2_record[:,n_list_all-2]
								T_Hm_values = np.exp(TH2_fit_from_simulations(np.log(Te_values.flatten())))/eV_to_K	# K
								T_Hm_values[T_Hm_values<300]=300
								T_H2p_values = np.exp(TH2_fit_from_simulations(np.log(Te_values.flatten())))/eV_to_K	# K
								T_H2p_values[T_H2p_values<300]=300
								T_Hp_values = Te_values.flatten()/eV_to_K	# K
								T_Hp_values[T_Hp_values<300]=300
								T_H2_values = np.exp(TH2_fit_from_simulations(np.log(Te_values.flatten())))/eV_to_K	# K
								T_H2_values[T_H2_values<300]=300
								T_H_values = np.exp(TH_fit_from_simulations(np.log(Te_values.flatten())))/eV_to_K	# K
								T_H_values[T_H_values<300]=300
								T_H_values_low = np.exp(TH_low_fit_from_simulations(np.log(Te_values.flatten())))/eV_to_K	# K
								T_H_values_low[T_H_values_low<300]=300
								record_nH_ne_values = []
								record_nH_ne_log_prob = []
								for i_Te_for_nH_ne,Te_for_nH_ne in enumerate(Te_values_array):
									nH_ne_values = nH_ne_values_Te(Te_for_nH_ne,H_steps)
									record_nH_ne_values.append(nH_ne_values)
									nH_ne_log_probs = nH_ne_log_probs_Te(Te_for_nH_ne,nH_ne_values)
									record_nH_ne_log_prob.append(nH_ne_log_probs)
								total_nH_ne_value = multiplicative_factor*np.array([[record_nH_ne_values]*TS_ne_steps]*len(n_list_all)).T	# H, Te, ne, lines
								total_nH_ne_value = np.transpose(total_nH_ne_value, (0,1,3,2))*(ne_values_array ** 2)	# H, Te, lines, ne
								total_nH_ne_value = excitation_internal*(np.transpose(total_nH_ne_value, (0,3,1,2))).reshape((H_steps,TS_ne_steps*TS_Te_steps,len(n_list_all)))	# H, ne*Te, lines
								total_nH_ne_value = np.float32(np.transpose(np.array([total_nH_ne_value.tolist()]*H2_steps),(1,0,2,3)))	# H, H2, ne*Te, lines
								record_nH_ne_values = (np.array([record_nH_ne_values]*TS_ne_steps).reshape((TS_ne_steps*TS_Te_steps,H_steps))).T
								record_nH_ne_log_prob = (np.array([record_nH_ne_log_prob]*TS_ne_steps).reshape((TS_ne_steps*TS_Te_steps,H_steps))).T
								nH_ne_excited_states_atomic = [[[np.float32(record_nH_ne_values*np.sum(excitation_full,axis=-1)*ne_values.flatten()).tolist()]*H2p_to_find_steps]*H2_steps]*Hm_to_find_steps
								nH_ne_excited_states_atomic = np.transpose(nH_ne_excited_states_atomic, (3,0,1,2,4))	# H, Hm, H2, H2p, ne, Te
								nH_ne_excited_state_atomic_2 = [[[np.float32(record_nH_ne_values*excitation_full[:,0]*ne_values.flatten()).tolist()]*H2p_to_find_steps]*H2_steps]*Hm_to_find_steps
								nH_ne_excited_state_atomic_2 = np.transpose(nH_ne_excited_state_atomic_2, (3,0,1,2,4))	# H, Hm, H2, H2p, ne, Te
								nH_ne_excited_state_atomic_3 = [[[np.float32(record_nH_ne_values*excitation_full[:,1]*ne_values.flatten()).tolist()]*H2p_to_find_steps]*H2_steps]*Hm_to_find_steps
								nH_ne_excited_state_atomic_3 = np.transpose(nH_ne_excited_state_atomic_3, (3,0,1,2,4))	# H, Hm, H2, H2p, ne, Te
								nH_ne_excited_state_atomic_4 = [[[np.float32(record_nH_ne_values*excitation_full[:,2]*ne_values.flatten()).tolist()]*H2p_to_find_steps]*H2_steps]*Hm_to_find_steps
								nH_ne_excited_state_atomic_4 = np.transpose(nH_ne_excited_state_atomic_4, (3,0,1,2,4))	# H, Hm, H2, H2p, ne, Te
								# total_nH_ne_value = []
								# for i1,nH_ne_value in enumerate(nH_ne_values):
								# 	# total_nH_ne_value.append(excitation_internal *  nH_ne_value)
								# 	total_nH_ne_value.append(np.ones((H2_steps,*np.shape(excitation_internal)))*excitation_internal *  nH_ne_value)
								# 	# total_nH_ne_value = excitation_internal *  nH_ne_value
								# total_nH_ne_value = np.array(total_nH_ne_value)
								# total_nH_ne_value = np.transpose(total_nH_ne_value*multiplicative_factor, (0,1,3,2))	# H, H2, lines, ne*Te
								# total_nH_ne_value = total_nH_ne_value * (ne_values.flatten() ** 2)
								# total_nH_ne_value = np.float32(np.transpose(total_nH_ne_value, (0,1,3,2)))	# H, H2, ne*Te, lines
								total_nH2_ne_value = []
								record_nH2_ne_values = []
								record_nH2_ne_log_prob = []
								for i_Te_for_nH2_ne,Te_for_nH2_ne in enumerate(Te_values_array):
									nH2_ne_values = nH2_ne_values_Te(Te_for_nH2_ne,H2_steps)
									record_nH2_ne_values.append(nH2_ne_values)
									nH2_ne_log_probs = nH2_ne_log_probs_Te(Te_for_nH2_ne,nH2_ne_values)
									record_nH2_ne_log_prob.append(nH2_ne_log_probs)
								total_nH2_ne_value = multiplicative_factor*np.array([[record_nH2_ne_values]*TS_ne_steps]*len(n_list_all)).T	# H2, Te, ne, lines
								total_nH2_ne_value = np.transpose(total_nH2_ne_value, (0,1,3,2))*(ne_values_array ** 2)	# H2, Te, lines, ne
								total_nH2_ne_value = np.float32(coeff_2*(np.transpose(total_nH2_ne_value, (0,3,1,2))).reshape((H2_steps,TS_ne_steps*TS_Te_steps,len(n_list_all))))	# H2, ne*Te, lines
								record_nH2_ne_values = (np.array([record_nH2_ne_values]*TS_ne_steps).reshape((TS_ne_steps*TS_Te_steps,H2_steps))).T
								record_nH2_ne_log_prob = (np.array([record_nH2_ne_log_prob]*TS_ne_steps).reshape((TS_ne_steps*TS_Te_steps,H2_steps))).T
								# total_nH2_ne_value = []
								# for i4,nH2_ne_value in enumerate(nH2_ne_values):
								# 	total_nH2_ne_value.append( coeff_2 *  nH2_ne_value )
								# 	# total_nH2_ne_value = nH2_ne_value * coeff_2
								# total_nH2_ne_value = np.array(total_nH2_ne_value)
								# total_nH2_ne_value = np.transpose(total_nH2_ne_value*multiplicative_factor, (0,2,1))	# H2, lines, ne*Te
								# total_nH2_ne_value = total_nH2_ne_value * (ne_values.flatten() ** 2)
								# total_nH2_ne_value = np.float32(np.transpose(total_nH2_ne_value, (0,2,1)))	# H2, ne*Te, lines
								total_nH_nH2_ne_values = total_nH_ne_value + total_nH2_ne_value
								record_nH2p_nH2_values = nH2p_nH2_values_Te_ne(Te_values.flatten(),ne_values.flatten(),H2p_to_find_steps,H2_suppression=H2_suppression)	# H2p, ne*Te
								record_nH2p_ne_values = np.transpose([record_nH2_ne_values.tolist()]*(H2p_to_find_steps) ,(1,0,2)) * record_nH2p_nH2_values
								# record_nH2p_ne_values = np.array([record_nH2_ne_values.tolist()]*(H2p_to_find_steps)) * record_nH2p_nH2_values	# Hm, H2p, ne*Te
								record_nHm_nH2_values = nHm_nH2_values_Te(Te_values.flatten(),Hm_to_find_steps,H2_suppression=H2_suppression)	# Hm, ne*Te
								record_nHm_ne_values = np.transpose([record_nH2_ne_values.tolist()]*(Hm_to_find_steps) ,(1,0,2)) * record_nHm_nH2_values
								record_nHp_ne_values = 1 - record_nH2p_ne_values + np.transpose([record_nHm_ne_values.tolist()]*(H2p_to_find_steps), (2,1,0,3))	# Hm, H2, H2p, ne*Te
								record_nHp_ne_values[record_nHp_ne_values<=0] = 1e-15	# H, Hm, H2, H2p, ne, Te
								if loop_index==1:
									record_nH2p_nH2_values_initial = cp.deepcopy(record_nH2p_nH2_values)
									record_nH2p_ne_values_initial = cp.deepcopy(record_nH2p_ne_values)
									record_nHm_nH2_values_initial = cp.deepcopy(record_nHm_nH2_values)
									record_nHm_ne_values_initial = cp.deepcopy(record_nHm_ne_values)
								recombination_full_sum = np.sum(recombination_full,axis=-1)
								coeff_3_record = []
								coeff_4_record = []
								for i4 in range(H2_steps):
									for i5 in range(H2p_to_find_steps):
										# nH2p_nH2_values = record_nH2p_nH2_values[i5]
										# nH2p_ne_values = record_nH2_ne_values[i4]*nH2p_nH2_values
										nH2p_ne_values = record_nH2p_ne_values[i4,i5]
										total_nH2p_ne_values = np.float32(nH2p_ne_values *(ne_values.flatten() ** 2) * (coeff_1*multiplicative_factor).T).T
										coeff_4 = From_Hn_with_H2p_pop_coeff_full_extra(np.array([Te_values.flatten(),T_H2p_values,T_Hm_values,ne_values.flatten(),nH2p_ne_values*ne_values.flatten()]).T,total_wavelengths)
										coeff_4_record.append(coeff_4)
										coeff_4 = cp.deepcopy(coeff_4[:,n_list_all-2])
										total_nH_nH2_nH2p_ne_values = total_nH_nH2_ne_values[:,i4] + total_nH2p_ne_values
									# total_nH2p_ne_value = nH2p_ne_value * coeff_1
									# total_nH2p_ne_value = (total_nH2p_ne_value*multiplicative_factor).T	# lines, ne*Te
									# total_nH2p_ne_value = total_nH2p_ne_value * (ne_values.flatten() ** 2)
									# total_nH2p_ne_value = np.float32(total_nH2p_ne_value.T)	# ne*Te, lines
									# total_nH_nH2_nH2p_ne_value = total_nH_nH2_ne_value + total_nH2p_ne_value
									# coeff_4 = From_Hn_with_H2p_pop_coeff_full_extra(np.array([Te_values.flatten(),T_H2p_values,T_Hm_values,ne_values.flatten(),nH2p_ne_value*ne_values.flatten()]).T,n_list_all)
									# coeff_4_record.append(coeff_4)
										for i3 in range(Hm_to_find_steps):
											# nHm_nH2_values = record_nHm_nH2_values[i3]
											# nHm_ne_values = record_nH2_ne_values[i4]*nHm_nH2_values
											nHm_ne_values = record_nHm_ne_values[i4,i3]
											nHp_ne_values = 1 - nH2p_ne_values + nHm_ne_values
											nHp_ne_good = nHp_ne_values>0
											# nHp_ne_mid = nHp_ne_values==0
											nHp_ne_bad = nHp_ne_values<0
											# nHp_ne_values[nHp_ne_mid]=1e-10
											nHp_ne_values[nHp_ne_bad]=0
											total_nHp_ne_values = (recombination_internal.T * nHp_ne_values).T
											nH_ne_excited_states_atomic[:,i3,i4,i5] += np.float32(recombination_full_sum * nHp_ne_values * ne_values.flatten())
											nH_ne_excited_state_atomic_2[:,i3,i4,i5] += np.float32(recombination_full[:,0] * nHp_ne_values * ne_values.flatten())
											nH_ne_excited_state_atomic_3[:,i3,i4,i5] += np.float32(recombination_full[:,1] * nHp_ne_values * ne_values.flatten())
											nH_ne_excited_state_atomic_4[:,i3,i4,i5] += np.float32(recombination_full[:,2] * nHp_ne_values * ne_values.flatten())
											# total_nHp_ne_values[nHp_ne_mid]=0
											coeff_3 = np.zeros((len(nHp_ne_values),len(total_wavelengths)))
											coeff_3[nHp_ne_good] = From_Hn_with_Hp_pop_coeff_full_extra(np.array([Te_values.flatten()[nHp_ne_good],T_Hp_values[nHp_ne_good],T_Hm_values[nHp_ne_good],ne_values.flatten()[nHp_ne_good],(nHp_ne_values*ne_values.flatten())[nHp_ne_good]]).T ,total_wavelengths)
											# coeff_3[np.logical_and(nHp_ne_mid,nHp_ne_bad)]=0
											coeff_3_record.append(coeff_3)
											coeff_3 = cp.deepcopy(coeff_3[:,n_list_all-2])
											total_nHm_ne_values = (nHm_ne_values*( coeff_3 + coeff_4 ).T).T
											total = total_nHp_ne_values+total_nHm_ne_values# +total_nH3p_ne_value	# H, H2, ne*Te, lines
											total = (total*multiplicative_factor).T	# lines, ne*Te
											total = np.float32(total * (ne_values.flatten() ** 2))
											calculated_emission = total_nH_nH2_nH2p_ne_values + total.T	# H, H2, ne*Te, lines
											temp = np.float32(-0.5*np.sum(((calculated_emission - inverted_profiles_crop_restrict)/inverted_profiles_crop_sigma_restrict)**2,axis=-1))
											nHp_ne_good = np.logical_not(nHp_ne_bad.reshape((TS_ne_steps,TS_Te_steps)))
											calculated_emission_log_probs[:,i3,i4,i5,nHp_ne_good]= temp.reshape((H_steps,TS_ne_steps,TS_Te_steps))[:,nHp_ne_good]
											del temp,calculated_emission
										# if nHp_ne_value<0:
										# 	nHp_ne_value=1e-8
										# 	coeff_3_record.append(0)
										# 	continue
										# elif nHp_ne_value==0:
										# 	nHp_ne_value=1e-8
										# 	total_nHp_ne_value = 0
										# else:
										# coeff_3 = From_Hn_with_Hp_pop_coeff_full_extra(np.array([Te_values.flatten(),T_Hp_values,T_Hm_values,ne_values.flatten(),nHp_ne_value*ne_values.flatten()]).T ,n_list_all)
										# coeff_3_record.append(coeff_3)
										# total_nHm_ne_value = nHm_ne_value*( coeff_3 + coeff_4 )
										# total = total_nHp_ne_value+total_nHm_ne_value# +total_nH3p_ne_value	# H, H2, ne*Te, lines
										# total = (total*multiplicative_factor).T	# lines, ne*Te
										# total = np.float32(total * (ne_values.flatten() ** 2))
										# calculated_emission = total_nH_nH2_nH2p_ne_value + total.T	# H, H2, ne*Te, lines
										# temp = np.float32(-0.5*np.sum(((calculated_emission - inverted_profiles_crop_restrict)/inverted_profiles_crop_sigma_restrict)**2,axis=-1))
										# calculated_emission_log_probs[:,i3,:,i5]= temp.reshape((len(nH_ne_values),H2_steps,TS_ne_steps,TS_Te_steps))
								calculated_emission_log_probs = calculated_emission_log_probs-np.max(calculated_emission_log_probs)
								calculated_emission_log_probs = calculated_emission_log_probs-np.log(np.sum(np.exp(calculated_emission_log_probs)))	# normalisation for logarithmic probabilities

								time_lapsed = tm.time()-start_time
								print('worker '+str(current_process())+' Initial loop nr '+str(loop_index)+'-3 done '+str(domain_index)+' in %.3gmin %.3gsec' %(int(time_lapsed/60),int(time_lapsed%60)) +' shape '+str(np.shape(calculated_emission_log_probs))+' - good=%.3g, nan=%.3g, -inf=%.3g' %( np.sum(np.isfinite(calculated_emission_log_probs)),np.sum(np.isnan(calculated_emission_log_probs)),np.sum(np.isinf(calculated_emission_log_probs)) ) )

								power_rad_H2p,power_rad_H2,power_rad_excit,power_via_ionisation,power_rad_rec_bremm,power_via_recombination,power_via_brem,power_rec_neutral,power_rad_Hm_H2p,power_rad_Hm_Hp,power_rad_Hm,power_rad_mol,power_heating_rec,tot_rad_power,total_removed_power_atomic,nH_ne_excited_states_mol,nH_ne_excited_state_mol_2,nH_ne_excited_state_mol_3,nH_ne_excited_state_mol_4,power_rad_mol_visible = calc_power_balance_elements_simplified2(H_steps,Hm_to_find_steps,H2_steps,H2p_to_find_steps,TS_ne_steps,TS_Te_steps,Te_values,ne_values,multiplicative_factor_full_full,multiplicative_factor_visible_light_full_full,record_nH_ne_values,record_nHm_ne_values,record_nH2_ne_values,record_nH2p_ne_values,record_nHp_ne_values,T_Hp_values,T_H2p_values,T_Hm_values,coeff_1_record,coeff_2_record,coeff_3_record,coeff_4_record)
								total_removed_power = total_removed_power_atomic + power_rad_mol

								# area = 2*np.pi*(r_crop[my_r_pos] + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
								temp = np.pi*((r_crop + np.median(np.diff(r_crop))/2)**2)
								area = np.array([temp[0]]+np.diff(temp).tolist())[my_r_pos] # m^2

								length = 0.351+target_OES_distance/1000	# mm distance skimmer to OES/TS + OES/TS to target
								total_removed_power_times_volume = total_removed_power * area*length
								# power_penalty = np.zeros_like((total_removed_power_times_volume),dtype=np.float32)
								if False:	# here I do not use any info on the shape of the plasma upstream
									select = total_removed_power_times_volume>interpolated_power_pulse_shape(time_crop[my_time_pos])/source_power_spread
									power_penalty[select] = np.float32(-0.5*((total_removed_power_times_volume[select] - interpolated_power_pulse_shape(time_crop[my_time_pos])/source_power_spread)/(interpolated_power_pulse_shape_std(time_crop[my_time_pos])))**2)
								else:	# here I do
									# max_local_flow_vel = max(1,homogeneous_mach_number[my_time_pos])*upstream_adiabatic_collisional_velocity[my_time_pos,my_r_pos]
									max_local_flow_vel = homogeneous_mach_number[my_time_pos]*upstream_adiabatic_collisional_velocity[my_time_pos,my_r_pos]
									total_removable_power_times_volume_SS = area*(0.5*hydrogen_mass*(max_local_flow_vel**2) + (5*Te_all_upstream[my_time_pos,my_r_pos] + ionisation_potential + dissociation_potential)/eV_to_K*boltzmann_constant_J)*max_local_flow_vel*ne_all_upstream[my_time_pos,my_r_pos]*1e20
									total_removable_power_times_volume_dynamic = area*length/dt*1000*(0.5*hydrogen_mass*(max_local_flow_vel**2) + (5*Te_values + ionisation_potential + dissociation_potential)/eV_to_K*boltzmann_constant_J)*ne_values
									total_removable_power_times_volume = total_removable_power_times_volume_SS + total_removable_power_times_volume_dynamic
									total_removable_power_times_volume = np.array([[[[total_removable_power_times_volume.tolist()]*H2p_to_find_steps]*H2_steps]*Hm_to_find_steps]*H_steps,dtype=np.float32)
									if False:
										select = total_removed_power_times_volume>total_removable_power_times_volume
										power_penalty[select] = np.float32(-0.5*((total_removed_power_times_volume[select] - total_removable_power_times_volume[select])/total_removable_power_times_volume[select])**2)
									elif True:	# real formula for the probability of an inequality
										# total_removed_power_times_volume_sigma = (0.2*total_removed_power_atomic + 0.5*power_rad_mol) * area*length
										total_removed_power_times_volume_sigma = (( (power_atomic_precision**2)*(power_via_ionisation**2 + power_rad_excit**2 + power_via_recombination**2 + power_rec_neutral**2 + power_via_brem**2) + (power_molecular_precision**2)*(power_rad_Hm**2 + power_rad_H2**2 + power_rad_H2p**2))**0.5) * area*length	#this properly adds in quadrature all the uncertanties
										total_removable_power_times_volume_sigma = (total_removable_power_times_volume*power_budget_precision)
										# power_penalty = np.log( 1 + np.erf( (total_removable_power_times_volume-total_removed_power_times_volume)/(2**0.5 * (total_removed_power_times_volume_sigma**2 + (total_removable_power_times_volume*0.5)**2)**0.5)) )
										power_penalty = np.float32(np.erf( total_removed_power_times_volume/( 2**0.5 * total_removed_power_times_volume_sigma ) ))
										power_penalty += np.float32(np.erf( (total_removable_power_times_volume-total_removed_power_times_volume)/(2**0.5 * (total_removed_power_times_volume_sigma**2 + total_removable_power_times_volume_sigma**2)**0.5)))
										power_penalty = power_penalty/np.float32(0.5 + np.erf( total_removed_power_times_volume/( 2**0.5 * total_removed_power_times_volume_sigma ) ))
										power_penalty = power_penalty/np.float32(0.5 + np.erf( total_removable_power_times_volume/( 2**0.5 * total_removable_power_times_volume_sigma ) ))
										time_lapsed = tm.time()-start_time
										print('worker '+str(current_process())+' Initial loop nr '+str(loop_index)+'-2 done '+str(domain_index)+' in %.3gmin %.3gsec' %(int(time_lapsed/60),int(time_lapsed%60)) +', power penalty - good=%.3g, nan=%.3g, -inf=%.3g, -<0 =%.3g, min=%.3g' %( np.sum(np.isfinite(power_penalty)),np.sum(np.isnan(power_penalty)),np.sum(np.isinf(power_penalty)),np.sum(power_penalty<0) ,np.nanmin(power_penalty)) )
										power_penalty[power_penalty<0]=0
										power_penalty = np.log(power_penalty)
										# power_penalty += -np.log( 1+ np.erf( total_removed_power_times_volume/( 2**0.5 * total_removed_power_times_volume_sigma ) ) ) - np.log( 1+ np.erf( total_removable_power_times_volume/( 2**0.5 * total_removable_power_times_volume*0.5 ) ) )
										power_penalty -= power_penalty.max()
										# del total_removed_power_times_volume_sigma

								nH_ne_excited_states = nH_ne_excited_states_atomic + nH_ne_excited_states_mol.reshape((H_steps,Hm_to_find_steps,H2_steps,H2p_to_find_steps,TS_ne_steps*TS_Te_steps))
								nH_ne_excited_state_2 = nH_ne_excited_state_atomic_2 + nH_ne_excited_state_mol_2.reshape((H_steps,Hm_to_find_steps,H2_steps,H2p_to_find_steps,TS_ne_steps*TS_Te_steps))
								nH_ne_excited_state_3 = nH_ne_excited_state_atomic_3 + nH_ne_excited_state_mol_3.reshape((H_steps,Hm_to_find_steps,H2_steps,H2p_to_find_steps,TS_ne_steps*TS_Te_steps))
								nH_ne_excited_state_4 = nH_ne_excited_state_atomic_4 + nH_ne_excited_state_mol_4.reshape((H_steps,Hm_to_find_steps,H2_steps,H2p_to_find_steps,TS_ne_steps*TS_Te_steps))
								nH_ne_ground_state = np.transpose(record_nH_ne_values - np.transpose(nH_ne_excited_states, (1,2,3,0,4)), (3,0,1,2,4))
								# temp = np.transpose(np.transpose(nH_ne_excited_states, (1,2,3,0,4)) > record_nH_ne_values, (3,0,1,2,4))
								nH_ne_penalty = np.zeros_like(nH_ne_excited_states,dtype=np.float16)
								# nH_ne_penalty[temp==True] = -np.inf
								nH_ne_penalty[nH_ne_ground_state<0] = -np.inf
								nH_ne_ground_state[nH_ne_ground_state<0] = 0
								# del temp
								nH_ne_penalty = nH_ne_penalty.reshape((H_steps,Hm_to_find_steps,H2_steps,H2p_to_find_steps,TS_ne_steps,TS_Te_steps))
								nH_ne_ground_state = nH_ne_ground_state.reshape((H_steps,Hm_to_find_steps,H2_steps,H2p_to_find_steps,TS_ne_steps,TS_Te_steps))
								nH_ne_excited_states = nH_ne_excited_states.reshape((H_steps,Hm_to_find_steps,H2_steps,H2p_to_find_steps,TS_ne_steps,TS_Te_steps))
								nH_ne_excited_state_2 = nH_ne_excited_state_2.reshape((H_steps,Hm_to_find_steps,H2_steps,H2p_to_find_steps,TS_ne_steps,TS_Te_steps))
								nH_ne_excited_state_3 = nH_ne_excited_state_3.reshape((H_steps,Hm_to_find_steps,H2_steps,H2p_to_find_steps,TS_ne_steps,TS_Te_steps))
								nH_ne_excited_state_4 = nH_ne_excited_state_4.reshape((H_steps,Hm_to_find_steps,H2_steps,H2p_to_find_steps,TS_ne_steps,TS_Te_steps))
								del nH_ne_excited_states_atomic,nH_ne_excited_state_atomic_2,nH_ne_excited_state_atomic_3,nH_ne_excited_state_atomic_4,nH_ne_excited_states_mol,nH_ne_excited_state_mol_2,nH_ne_excited_state_mol_3,nH_ne_excited_state_mol_4

								# likelihood_log_probs = np.zeros((len(nH_ne_values),len(nHm_ne_values),len(nH2_ne_values),len(nH2p_ne_values),TS_ne_steps,TS_Te_steps))	# H, Hm, H2, H2p, ne, Te
								# for i5,nH2p_ne_value in enumerate(nH2p_ne_values):
								# 	for i3,nHm_ne_value in enumerate(nHm_ne_values):
								# 		for i1,nH_ne_value in enumerate(nH_ne_values):
								# 			temp_i1 = nH_ne_log_probs[i1]
								# 			for i4,nH2_ne_value in enumerate(nH2_ne_values):
								# 				temp_i4 = nH2_ne_log_probs[i4]
								# 				for i6 in range(TS_ne_steps):
								# 					temp_i6 = ne_log_probs[i6]
								# 					for i7 in range(TS_Te_steps):
								# 						likelihood_log_probs[i1,i3,i4,i5,i6,i7]= calculated_emission_log_probs[i1,i3,i4,i5,i6,i7]+Te_log_probs[i7]+temp_i6+temp_i4+temp_i1
								# likelihood_log_probs = ((calculated_emission_log_probs + Te_log_probs + (np.ones((TS_Te_steps,TS_ne_steps))*ne_log_probs).T + (np.ones((TS_Te_steps,TS_ne_steps,len(nH2p_ne_values),len(nH2_ne_values)))*nH2_ne_log_probs).T ).T + nH_ne_log_probs).T
								likelihood_log_probs = calculated_emission_log_probs + np.float32(Te_log_probs) + (np.ones((TS_Te_steps,TS_ne_steps),dtype=np.float32)*np.float32(ne_log_probs)).T + np.float32(np.transpose([np.float32(record_nH2_ne_log_prob).reshape((H2_steps,TS_ne_steps,TS_Te_steps)).tolist()]*H2p_to_find_steps,(1,0,2,3)))
								likelihood_log_probs = np.transpose(np.transpose(likelihood_log_probs,(1,2,3,0,4,5)) + np.float32(record_nH_ne_log_prob.reshape((H_steps,TS_ne_steps,TS_Te_steps))) ,(3,0,1,2,4,5)) + power_penalty + nH_ne_penalty
								likelihood_log_probs -= np.max(likelihood_log_probs)
								likelihood_log_probs -= np.log(np.sum(np.exp(likelihood_log_probs)))	# normalisation for logarithmic probabilities
								# good_priors = likelihood_log_probs!=-np.inf

								time_lapsed = tm.time()-start_time
								print('worker '+str(current_process())+' Initial loop nr '+str(loop_index)+'-1 done '+str(domain_index)+' in %.3gmin %.3gsec' %(int(time_lapsed/60),int(time_lapsed%60)) +' - good=%.3g, nan=%.3g, -inf=%.3g' %( np.sum(np.isfinite(likelihood_log_probs)),np.sum(np.isnan(likelihood_log_probs)),np.sum(np.isinf(likelihood_log_probs)) ) )

								# H, Hm, H2, H2p, ne, Te
								all_nHm_ne_values = np.array([np.transpose(np.array([record_nHm_ne_values.reshape((H2_steps,Hm_to_find_steps,TS_ne_steps,TS_Te_steps)).tolist()]*H2p_to_find_steps), (2,1,0,3,4)).tolist()]*H_steps,dtype=np.float32)
								all_nH2p_ne_values = np.array([[record_nH2p_ne_values.reshape((H2_steps,H2p_to_find_steps,TS_ne_steps,TS_Te_steps)).tolist()]*Hm_to_find_steps]*H_steps,dtype=np.float32)
								all_nHm_nH2_values = np.transpose(np.array([[[record_nHm_nH2_values.reshape((Hm_to_find_steps,TS_ne_steps,TS_Te_steps)).tolist()]*H2p_to_find_steps]*H2_steps]*H_steps,dtype=np.float32), (0,3,1,2,4,5))
								all_nH2p_nH2_values = np.array([[[record_nH2p_nH2_values.reshape((H2p_to_find_steps,TS_ne_steps,TS_Te_steps)).tolist()]*H2_steps]*Hm_to_find_steps]*H_steps,dtype=np.float32)
								all_nH2_ne_values = np.array([[np.transpose([record_nH2_ne_values.reshape((H2_steps,TS_ne_steps,TS_Te_steps)).tolist()]*H2p_to_find_steps, (1,0,2,3)).tolist()]*Hm_to_find_steps]*H_steps, dtype=np.float32)
								all_nH_ne_values = np.array(np.transpose([[[record_nH_ne_values.reshape((H_steps,TS_ne_steps,TS_Te_steps)).tolist()]*H2p_to_find_steps]*H2_steps]*Hm_to_find_steps, (3,0,1,2,4,5)), dtype=np.float32)
								all_ne_values = np.array([[[[ne_values.tolist()]*H2p_to_find_steps]*H2_steps]*Hm_to_find_steps]*H_steps,dtype=np.float32)
								all_Te_values = np.array([[[[Te_values.tolist()]*H2p_to_find_steps]*H2_steps]*Hm_to_find_steps]*H_steps,dtype=np.float32)
								all_nHp_ne_values = np.array([record_nHp_ne_values.reshape((Hm_to_find_steps,H2_steps,H2p_to_find_steps,TS_ne_steps,TS_Te_steps)).tolist()]*H_steps,dtype=np.float32)

								if include_particles_limitation:
									particles_penalty = np.zeros_like((likelihood_log_probs),dtype=np.float32)
									all_net_Hp_destruction = np.zeros_like((likelihood_log_probs),dtype=np.float32)
									all_net_e_destruction = np.zeros_like((likelihood_log_probs),dtype=np.float32)
									all_net_Hm_destruction = np.zeros_like((likelihood_log_probs),dtype=np.float32)
									all_net_H2p_destruction = np.zeros_like((likelihood_log_probs),dtype=np.float32)
									all_net_Hp_destruction_sigma = np.ones_like((likelihood_log_probs),dtype=np.float32)
									all_net_e_destruction_sigma = np.ones_like((likelihood_log_probs),dtype=np.float32)
									all_net_Hm_destruction_sigma = np.ones_like((likelihood_log_probs),dtype=np.float32)
									all_net_H2p_destruction_sigma = np.ones_like((likelihood_log_probs),dtype=np.float32)
									if False:
										x_full =np.array([all_nH_ne_values[np.isfinite(likelihood_log_probs)],all_nHm_nH2_values[np.isfinite(likelihood_log_probs)],all_nH2_ne_values[np.isfinite(likelihood_log_probs)],all_nH2p_nH2_values[np.isfinite(likelihood_log_probs)],1e-20*all_ne_values[np.isfinite(likelihood_log_probs)],all_Te_values[np.isfinite(likelihood_log_probs)]]).T
										all_net_Hp_destruction[np.isfinite(likelihood_log_probs)] = area*length*dt/1000*(interpolate.interpn((np.log(samples_nH_ne_array),np.log(samples_nHm_nH2_array),np.log(samples_nH2_ne_array),np.log(samples_nH2p_nH2_array),np.log(samples_ne_array),np.log(samples_Te_array)),net_rate_Hp_destruction,np.log(x_full), bounds_error=False,fill_value=None))
										all_net_e_destruction[np.isfinite(likelihood_log_probs)] = area*length*dt/1000*(interpolate.interpn((np.log(samples_nH_ne_array),np.log(samples_nHm_nH2_array),np.log(samples_nH2_ne_array),np.log(samples_nH2p_nH2_array),np.log(samples_ne_array),np.log(samples_Te_array)),net_rate_e_destruction,np.log(x_full), bounds_error=False,fill_value=None))
										all_net_Hm_destruction[np.isfinite(likelihood_log_probs)] = area*length*dt/1000*(interpolate.interpn((np.log(samples_nH_ne_array),np.log(samples_nHm_nH2_array),np.log(samples_nH2_ne_array),np.log(samples_nH2p_nH2_array),np.log(samples_ne_array),np.log(samples_Te_array)),net_rate_Hm_destruction,np.log(x_full), bounds_error=False,fill_value=None))
										all_net_H2p_destruction[np.isfinite(likelihood_log_probs)] = area*length*dt/1000*(interpolate.interpn((np.log(samples_nH_ne_array),np.log(samples_nHm_nH2_array),np.log(samples_nH2_ne_array),np.log(samples_nH2p_nH2_array),np.log(samples_ne_array),np.log(samples_Te_array)),net_rate_H2p_destruction,np.log(x_full), bounds_error=False,fill_value=None))
										del x_full
									elif True:	# I optimised the real functions enough that it's about the same speed as the interpolated method
										temp_coord = np.ones_like(nH_ne_excited_states[:,:,:,:,0,0])
										for i6 in range(TS_Te_steps):
											fractional_population_states_H2 = calc_H2_fractional_population_states(T_H2p_values[i6])
											for i5 in range(TS_ne_steps):
												if ((Te_values_array[i6] in Te_values_array_initial) and (ne_values_array[i5] in ne_values_array_initial)) and loop_index>0:
													i5_old = np.abs(ne_values_array_initial-ne_values_array[i5]).argmin()
													i6_old = np.abs(Te_values_array_initial-Te_values_array[i6]).argmin()
													all_net_Hp_destruction[:,:,:,:,i5,i6] = record_all_net_Hp_destruction[:,:,:,:,i5_old,i6_old]
													all_net_e_destruction[:,:,:,:,i5,i6] = record_all_net_e_destruction[:,:,:,:,i5_old,i6_old]
													all_net_Hm_destruction[:,:,:,:,i5,i6] = record_all_net_Hm_destruction[:,:,:,:,i5_old,i6_old]
													all_net_H2p_destruction[:,:,:,:,i5,i6] = record_all_net_H2p_destruction[:,:,:,:,i5_old,i6_old]
												else:
													# local_good_priors = good_priors[:,:,:,:,i5,i6]
													# # local_good_priors = np.ones_like(good_priors[:,:,:,:,i5,i6],dtype=bool)
													# if np.sum(local_good_priors)==0:
													# 	continue
													# temp_nH_excited_states = nH_ne_excited_states[:,:,:,:,i5,i6]*ne_values_array[i5]*1e-20
													temp_nH_ne_excited_state_2 = nH_ne_excited_state_2[:,:,:,:,i5,i6]
													temp_nH_ne_excited_state_3 = nH_ne_excited_state_3[:,:,:,:,i5,i6]
													temp_nH_ne_excited_state_4 = nH_ne_excited_state_4[:,:,:,:,i5,i6]
													temp_nH_ne_ground_state = nH_ne_ground_state[:,:,:,:,i5,i6]
													arguments = (temp_coord*Te_values_array[i6],temp_coord*T_Hp_values[i6],temp_coord*T_H_values[i6],temp_coord*T_H2_values[i6],temp_coord*T_Hm_values[i6],temp_coord*T_H2p_values[i6],temp_coord*ne_values_array[i5]*1e-20,all_nHp_ne_values[:,:,:,:,i5,i6],all_nH_ne_values[:,:,:,:,i5,i6],all_nH2_ne_values[:,:,:,:,i5,i6],all_nHm_ne_values[:,:,:,:,i5,i6],all_nH2p_ne_values[:,:,:,:,i5,i6],fractional_population_states_H2,temp_nH_ne_ground_state,temp_nH_ne_excited_state_2,temp_nH_ne_excited_state_3,temp_nH_ne_excited_state_4)
													temp,temp_sigma = np.float32(RR_rate_destruction_Hp(*arguments,particle_molecular_precision,particle_atomic_precision))
													temp1,temp1_sigma =	np.float32(RR_rate_creation_Hp(*arguments,particle_molecular_precision,particle_atomic_precision))
													all_net_Hp_destruction[:,:,:,:,i5,i6] = np.float32(temp - temp1)
													all_net_Hp_destruction_sigma[:,:,:,:,i5,i6] = 0.5*(temp_sigma**2 + temp1_sigma**2)**0.5
													temp,temp_sigma = np.float32(RR_rate_destruction_e(*arguments,particle_molecular_precision,particle_atomic_precision))
													temp1,temp1_sigma =	np.float32(RR_rate_creation_e(*arguments,particle_molecular_precision,particle_atomic_precision))
													all_net_e_destruction[:,:,:,:,i5,i6] = np.float32(temp - temp1)
													all_net_e_destruction_sigma[:,:,:,:,i5,i6] = 0.5*(temp_sigma**2 + temp1_sigma**2)**0.5
													temp,temp_sigma = np.float32(RR_rate_destruction_Hm(*arguments,particle_molecular_precision,particle_atomic_precision))
													temp1,temp1_sigma =	np.float32(RR_rate_creation_Hm(*arguments,particle_molecular_precision,particle_atomic_precision))
													all_net_Hm_destruction[:,:,:,:,i5,i6] = np.float32(temp - temp1)
													all_net_Hm_destruction_sigma[:,:,:,:,i5,i6] = 0.5*(temp_sigma**2 + temp1_sigma**2)**0.5
													temp,temp_sigma = np.float32(RR_rate_destruction_H2p(*arguments,particle_molecular_precision,particle_atomic_precision))
													temp1,temp1_sigma =	np.float32(RR_rate_creation_H2p(*arguments,particle_molecular_precision,particle_atomic_precision))
													all_net_H2p_destruction[:,:,:,:,i5,i6] = np.float32(temp - temp1)
													all_net_H2p_destruction_sigma[:,:,:,:,i5,i6] = 0.5*(temp_sigma**2 + temp1_sigma**2)**0.5
										try:
											del temp_coord,temp,temp_sigma,temp1,temp1_sigma
										except:
											print('worker '+str(current_process())+' Initial loop nr '+str(loop_index)+'-1 done '+str(domain_index)+' in %.3gmin %.3gsec' %(int(time_lapsed/60),int(time_lapsed%60)) + 'for some reason del temp_coord,temp,temp_sigma,temp1,temp1_sigma failed' )
										time_lapsed = tm.time()-start_time
										print('worker '+str(current_process())+' Initial loop nr '+str(loop_index)+'-0.5 done '+str(domain_index)+' in %.3gmin %.3gsec' %(int(time_lapsed/60),int(time_lapsed%60)) +' - good=%.3g, nan=%.3g, -inf=%.3g' %( np.sum(np.isfinite(likelihood_log_probs)),np.sum(np.isnan(likelihood_log_probs)),np.sum(np.isinf(likelihood_log_probs)) ) )
										all_net_Hp_destruction *= area*length*dt/1000
										all_net_e_destruction *= area*length*dt/1000
										all_net_Hm_destruction *= area*length*dt/1000
										all_net_H2p_destruction *= area*length*dt/1000
										all_net_Hp_destruction_sigma *= area*length*dt/1000
										all_net_e_destruction_sigma *= area*length*dt/1000
										all_net_Hm_destruction_sigma *= area*length*dt/1000
										all_net_H2p_destruction_sigma *= area*length*dt/1000
									total_removable_Hp = np.float32(area*dt/1000*ne_all_upstream[my_time_pos,my_r_pos]*max_local_flow_vel + area*length*all_ne_values*all_nHp_ne_values*1e-20)	# I assume no molecules upstream >> nH+/ne~1
									total_removable_e = np.float32(area*dt/1000*ne_all_upstream[my_time_pos,my_r_pos]*max_local_flow_vel + area*length*all_ne_values*1e-20)
									if False:
										select = all_net_Hp_destruction>total_removable_Hp
										particles_penalty[select] += np.float32(-0.5*((all_net_Hp_destruction[select] - total_removable_Hp[select])/total_removable_Hp[select])**2)	# 100% uncertanty
										select = all_net_e_destruction>total_removable_e
										particles_penalty[select] += np.float32(-0.5*((all_net_e_destruction[select] - total_removable_e[select])/total_removable_e[select])**2)	# 100% uncertanty
									elif True:
										particles_penalty += np.log( 1 + np.erf( (total_removable_Hp-all_net_Hp_destruction)/(2**0.5 * (all_net_Hp_destruction_sigma**2 + (total_removable_Hp*particle_atomic_budget_precision)**2)**0.5)) )
										particles_penalty += np.log( 1 + np.erf( (total_removable_e-all_net_e_destruction)/(2**0.5 * (all_net_e_destruction_sigma**2 + (total_removable_e*particle_atomic_budget_precision)**2)**0.5)) )

									total_removable_Hm = np.float32(area*dt/1000*ne_all_upstream[my_time_pos,my_r_pos]*all_nHm_ne_values*max_local_flow_vel + area*length*all_ne_values*all_nHm_ne_values*1e-20)	# I assume molecules upstream ~ downstream
									total_removable_H2p = np.float32(area*dt/1000*ne_all_upstream[my_time_pos,my_r_pos]*all_nH2p_ne_values*max_local_flow_vel + area*length*all_ne_values*all_nH2p_ne_values*1e-20)	# I assume molecules upstream ~ downstream
									if False:
										select = all_net_Hm_destruction>total_removable_Hm
										particles_penalty[select] += np.float32(-0.5*((all_net_Hm_destruction[select] - total_removable_Hm[select])/(total_removable_Hm[select]*2))**2)	# 200% uncertanty
										select = all_net_H2p_destruction>total_removable_H2p
										particles_penalty[select] += np.float32(-0.5*((all_net_H2p_destruction[select] - total_removable_H2p[select])/(total_removable_H2p[select]*2))**2)	# 200% uncertanty
									elif True:
										particles_penalty += np.log( 1 + np.erf( (total_removable_Hm-all_net_Hm_destruction)/(2**0.5 * (all_net_Hm_destruction_sigma**2 + (total_removable_Hm*particle_molecular_budget_precision)**2)**0.5)) )
										particles_penalty += np.log( 1 + np.erf( (total_removable_H2p-all_net_H2p_destruction)/(2**0.5 * (all_net_H2p_destruction_sigma**2 + (total_removable_H2p*particle_molecular_budget_precision)**2)**0.5)) )
									particles_penalty -= np.nanmax(particles_penalty)
									# particles_penalty[np.isnan(particles_penalty)]=-np.inf


									likelihood_log_probs += particles_penalty
									likelihood_log_probs -= np.max(likelihood_log_probs)
									likelihood_log_probs -= np.log(np.sum(np.exp(likelihood_log_probs)))	# normalisation for logarithmic probabilities

									if loop_index==0:
										record_all_net_Hp_destruction = all_net_Hp_destruction/(area*length*dt/1000)
										record_all_net_e_destruction = all_net_e_destruction/(area*length*dt/1000)
										record_all_net_Hm_destruction = all_net_Hm_destruction/(area*length*dt/1000)
										record_all_net_H2p_destruction = all_net_H2p_destruction/(area*length*dt/1000)
										record_all_net_Hp_destruction_sigma = all_net_Hp_destruction_sigma/(area*length*dt/1000)
										record_all_net_e_destruction_sigma = all_net_e_destruction_sigma/(area*length*dt/1000)
										record_all_net_Hm_destruction_sigma = all_net_Hm_destruction_sigma/(area*length*dt/1000)
										record_all_net_H2p_destruction_sigma = all_net_H2p_destruction_sigma/(area*length*dt/1000)
									else:
										del record_all_net_Hp_destruction,record_all_net_e_destruction,record_all_net_Hm_destruction,record_all_net_H2p_destruction,record_all_net_Hp_destruction_sigma,record_all_net_e_destruction_sigma,record_all_net_Hm_destruction_sigma,record_all_net_H2p_destruction_sigma
									del all_nHm_nH2_values,all_nH2p_nH2_values,all_nH2_ne_values,all_nH_ne_values,all_ne_values,all_Te_values,all_nHp_ne_values,all_net_Hp_destruction,all_net_Hp_destruction_sigma,total_removable_Hp,all_net_e_destruction,all_net_e_destruction_sigma,total_removable_e,all_net_Hm_destruction,all_net_Hm_destruction_sigma,all_net_H2p_destruction,all_net_H2p_destruction_sigma,total_removable_Hm,total_removable_H2p,all_nHm_ne_values,all_nH2p_ne_values
									del particles_penalty,nH_ne_excited_states,temp_nH_ne_excited_state_2,temp_nH_ne_excited_state_3,temp_nH_ne_excited_state_4,nH_ne_penalty,power_penalty

								time_lapsed = tm.time()-start_time
								print('worker '+str(current_process())+' Initial loop nr '+str(loop_index)+' done '+str(domain_index)+' in %.3gmin %.3gsec' %(int(time_lapsed/60),int(time_lapsed%60)) +' - good=%.3g, nan=%.3g, -inf=%.3g' %( np.sum(np.isfinite(likelihood_log_probs)),np.sum(np.isnan(likelihood_log_probs)),np.sum(np.isinf(likelihood_log_probs)) ) )

								if loop_index==0:
									hypervolume_of_each_combination = np.ones((H_steps,Hm_to_find_steps,H2_steps,H2p_to_find_steps,TS_ne_steps*TS_Te_steps),dtype=np.float32)	# H, Hm, H2, H2p, ne*Te
									hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (0,1,3,4,2))	# H, Hm, H2p, ne*Te, H2
									for i6 in range(TS_ne_steps*TS_Te_steps):
										nH2_ne_values = record_nH2_ne_values[:,i6]
										if H2_steps==1:
											d_nH2_ne_values = np.array([1])
										else:
											d_nH2_ne_values = np.array([*np.diff(nH2_ne_values[[0,1]]),*(np.diff(nH2_ne_values)[:-1]/2+np.diff(nH2_ne_values)[1:]/2),*np.diff(nH2_ne_values[[-2,-1]])])
										hypervolume_of_each_combination[:,:,:,i6] *= d_nH2_ne_values/np.sum(d_nH2_ne_values)
									hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (0,4,2,3,1))	# H, H2, H2p, ne*Te, Hm
									for i6 in range(TS_ne_steps*TS_Te_steps):
										nHm_nH2_values = record_nHm_nH2_values[:,i6]
										d_nHm_nH2_values = np.array([*np.diff(nHm_nH2_values[[0,1]]),*(np.diff(nHm_nH2_values)[:-1]/2+np.diff(nHm_nH2_values)[1:]/2),*np.diff(nHm_nH2_values[[-2,-1]])])
										hypervolume_of_each_combination[:,:,:,i6] *= d_nHm_nH2_values/np.sum(d_nHm_nH2_values)
									hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (0,4,1,3,2))	# H, Hm, H2, ne*Te, H2p
									for i6 in range(TS_ne_steps*TS_Te_steps):
										nH2p_nH2_values = record_nH2p_nH2_values[:,i6]
										d_nH2p_nH2_values = np.array([*np.diff(nH2p_nH2_values[[0,1]]),*(np.diff(nH2p_nH2_values)[:-1]/2+np.diff(nH2p_nH2_values)[1:]/2),*np.diff(nH2p_nH2_values[[-2,-1]])])
										hypervolume_of_each_combination[:,:,:,i6] *= d_nH2p_nH2_values/np.sum(d_nH2p_nH2_values)
									hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (1,2,4,3,0))	# Hm, H2, H2p, ne*Te, H
									for i6 in range(TS_ne_steps*TS_Te_steps):
										nH_ne_values = record_nH_ne_values[:,i6]
										d_nH_ne_values = np.array([*np.diff(nH_ne_values[[0,1]]),*(np.diff(nH_ne_values)[:-1]/2+np.diff(nH_ne_values)[1:]/2),*np.diff(nH_ne_values[[-2,-1]])])
										hypervolume_of_each_combination[:,:,:,i6] *= d_nH_ne_values/np.sum(d_nH_ne_values)
									hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (4,0,1,2,3))	# H, Hm, H2, H2p, ne*Te
									hypervolume_of_each_combination = hypervolume_of_each_combination.reshape((H_steps,Hm_to_find_steps,H2_steps,H2p_to_find_steps,TS_ne_steps,TS_Te_steps))	# H, Hm, H2, H2p, ne, Te
									d_Te_values_array = np.ones_like(Te_values_array)
									hypervolume_of_each_combination *= d_Te_values_array/np.sum(d_Te_values_array)
									hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (0,1,2,3,5,4))	# H, Hm, H2, H2p, Te, ne
									d_ne_values_array = np.ones_like(ne_values_array)
									hypervolume_of_each_combination *= d_ne_values_array/np.sum(d_ne_values_array)
									hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (0,1,2,3,5,4))	# H, Hm, H2, H2p, ne, Te

									marginalised_log_prob_ne_Te = np.sum(np.exp(likelihood_log_probs)*hypervolume_of_each_combination,axis=(0,1,2,3))	# ne, Te
									del hypervolume_of_each_combination
									# print(marginalised_log_prob_ne_Te)
									index_most_likely_marginalised = marginalised_log_prob_ne_Te.argmax()
									# print(index_most_likely_marginalised)
									most_likely_Te_index = index_most_likely_marginalised%TS_Te_steps
									most_likely_ne_index = (index_most_likely_marginalised-most_likely_Te_index)//TS_Te_steps%TS_ne_steps
									marginalised_log_prob_ne = marginalised_log_prob_ne_Te[:,most_likely_Te_index]
									marginalised_log_prob_Te = marginalised_log_prob_ne_Te[most_likely_ne_index]
									# print(marginalised_log_prob_ne)
									# print(marginalised_log_prob_Te)
									# marginalised_log_prob_ne = np.log(np.sum(np.exp(likelihood_log_probs)*hypervolume_of_each_combination,axis=(0,1,2,3,5)))	# ne, Te
									# marginalised_log_prob_Te = np.log(np.sum(np.exp(likelihood_log_probs)*hypervolume_of_each_combination,axis=(0,1,2,3,4)))	# ne, Te

									ne_values_limits = np.array([ne_values_array for _, ne_values_array in sorted(zip(marginalised_log_prob_ne, ne_values_array))])[-4:]
									# ne_additional_values = np.linspace(np.min(ne_values_limits),np.max(ne_values_limits),12)
									ne_additional_values = np.logspace(np.log10(np.min(ne_values_limits)),np.log10(np.max(ne_values_limits)),num=TS_ne_steps_increase)
									ne_values_array_initial = cp.deepcopy(ne_values_array)
									ne_values_array = np.unique(np.concatenate((ne_values_array,ne_additional_values)))
									Te_values_limits = np.array([Te_values_array for _, Te_values_array in sorted(zip(marginalised_log_prob_Te, Te_values_array))])[-4:]
									Te_additional_values = np.linspace(np.min(Te_values_limits),np.max(Te_values_limits),TS_Te_steps_increase)
									Te_values_array_initial = cp.deepcopy(Te_values_array)
									Te_values_array = np.unique(np.concatenate((Te_values_array,Te_additional_values)))
								PDF_matrix_shape.append(np.shape(likelihood_log_probs))




							# total_nH2p_ne_value
							# nH2p_ne_value = np.array([[(np.ones((to_find_steps,to_find_steps,to_find_steps_H2,to_find_steps))*nHpolecule_ne).tolist()]*TS_steps*TS_steps]*len(n_list_all))
							# nHm_ne_value = np.array([[[(np.ones((to_find_steps,to_find_steps_H2,to_find_steps)).T*nHpolecule_ne).T.tolist()]*to_find_steps]*TS_steps*TS_steps]*len(n_list_all))
							# nHp_ne_value = 1 - nH2p_ne_value + nHm_ne_value

							# calculated_emission_error = np.sum(((calculated_emission - inverted_profiles_crop_restrict)/inverted_profiles_crop_sigma_restrict)**2,axis=-1)
							# calculated_emission_error = np.sum(calculated_emission_error.T * Te_probs,axis=-1)
							# calculated_emission_error = np.sum(calculated_emission_error * ne_probs,axis=-1).T

							index_best_fit = calculated_emission_log_probs.argmax()	# H, Hm, H2, H2p, ne, Te
							best_fit_nH_ne_index,best_fit_nHm_nH2_index,best_fit_nH2_ne_index,best_fit_nH2p_nH2_index,best_fit_ne_index,best_fit_Te_index = np.unravel_index(index_best_fit, calculated_emission_log_probs.shape)
							# best_fit_Te_index = index_best_fit%TS_Te_steps
							# best_fit_ne_index = (index_best_fit-best_fit_Te_index)//TS_Te_steps%TS_ne_steps
							# best_fit_nH2p_nH2_index = ((index_best_fit-best_fit_Te_index)//TS_Te_steps-best_fit_ne_index)//TS_ne_steps%H2p_to_find_steps
							# best_fit_nH2_ne_index = (((index_best_fit-best_fit_Te_index)//TS_Te_steps-best_fit_ne_index)//TS_ne_steps-best_fit_nH2p_nH2_index)//H2p_to_find_steps%H2_steps
							# best_fit_nHm_nH2_index = ((((index_best_fit-best_fit_Te_index)//TS_Te_steps-best_fit_ne_index)//TS_ne_steps-best_fit_nH2p_nH2_index)//H2p_to_find_steps - best_fit_nH2_ne_index)//H2_steps%Hm_to_find_steps
							# best_fit_nH_ne_index = (((((index_best_fit-best_fit_Te_index)//TS_Te_steps-best_fit_ne_index)//TS_ne_steps-best_fit_nH2p_nH2_index)//H2p_to_find_steps - best_fit_nH2_ne_index)//H2_steps - best_fit_nHm_nH2_index)//Hm_to_find_steps%H_steps
							del calculated_emission_log_probs

							best_fit_Te_value = Te_values[best_fit_ne_index,best_fit_Te_index]
							best_fit_ne_value = ne_values[best_fit_ne_index,best_fit_Te_index]
							best_fit_nH2p_ne_value = record_nH2p_nH2_values[best_fit_nH2p_nH2_index,best_fit_ne_index*best_fit_Te_index]*record_nH2_ne_values[best_fit_nH2_ne_index,best_fit_ne_index*best_fit_Te_index]
							best_fit_nH2_ne_value = record_nH2_ne_values[best_fit_nH2_ne_index,best_fit_ne_index*best_fit_Te_index]
							best_fit_nHm_ne_value = record_nHm_nH2_values[best_fit_nHm_nH2_index,best_fit_ne_index*best_fit_Te_index]*record_nH2_ne_values[best_fit_nH2_ne_index,best_fit_ne_index*best_fit_Te_index]
							best_fit_nH_ne_value = record_nH_ne_values[best_fit_nH_ne_index,best_fit_ne_index*best_fit_Te_index]


							index_most_likely = likelihood_log_probs.argmax()
							most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index = np.unravel_index(index_most_likely, likelihood_log_probs.shape)
							# most_likely_Te_index = index_most_likely%TS_Te_steps
							# most_likely_ne_index = (index_most_likely-most_likely_Te_index)//TS_Te_steps%TS_ne_steps
							# most_likely_nH2p_nH2_index = ((index_most_likely-most_likely_Te_index)//TS_Te_steps-most_likely_ne_index)//TS_ne_steps%H2p_to_find_steps
							# most_likely_nH2_ne_index = (((index_most_likely-most_likely_Te_index)//TS_Te_steps-most_likely_ne_index)//TS_ne_steps-most_likely_nH2p_nH2_index)//H2p_to_find_steps%H2_steps
							# most_likely_nHm_nH2_index = ((((index_most_likely-most_likely_Te_index)//TS_Te_steps-most_likely_ne_index)//TS_ne_steps-most_likely_nH2p_nH2_index)//H2p_to_find_steps - most_likely_nH2_ne_index)//H2_steps%Hm_to_find_steps
							# most_likely_nH_ne_index = (((((index_most_likely-most_likely_Te_index)//TS_Te_steps-most_likely_ne_index)//TS_ne_steps-most_likely_nH2p_nH2_index)//H2p_to_find_steps - most_likely_nH2_ne_index)//H2_steps - most_likely_nHm_nH2_index)//Hm_to_find_steps%H_steps
							max_likelihood_log_prob = likelihood_log_probs[most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index]

							most_likely_Te_value = Te_values[most_likely_ne_index,most_likely_Te_index]
							most_likely_ne_value = ne_values[most_likely_ne_index,most_likely_Te_index]
							most_likely_nH2p_ne_value = record_nH2p_nH2_values[most_likely_nH2p_nH2_index,most_likely_ne_index*TS_Te_steps+most_likely_Te_index] * record_nH2_ne_values[most_likely_nH2_ne_index,most_likely_ne_index*TS_Te_steps+most_likely_Te_index]
							most_likely_nH2_ne_value = record_nH2_ne_values[most_likely_nH2_ne_index,most_likely_ne_index*TS_Te_steps+most_likely_Te_index]
							most_likely_nHm_ne_value = record_nHm_nH2_values[most_likely_nHm_nH2_index,most_likely_ne_index*TS_Te_steps+most_likely_Te_index] * record_nH2_ne_values[most_likely_nH2_ne_index,most_likely_ne_index*TS_Te_steps+most_likely_Te_index]
							most_likely_nH_ne_value = record_nH_ne_values[most_likely_nH_ne_index,most_likely_ne_index*TS_Te_steps+most_likely_Te_index]

							# now I marginalise H2, Te, ne in order to find the most likely Hm, H2p for the next step
							hypervolume_of_each_combination = np.ones((H_steps,Hm_to_find_steps,H2_steps,H2p_to_find_steps,TS_ne_steps*TS_Te_steps),dtype=np.float32)	# H, Hm, H2, H2p, ne*Te
							hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (0,1,3,4,2))	# H, Hm, H2p, ne*Te, H2
							for i6 in range(TS_ne_steps*TS_Te_steps):
								nH2_ne_values = record_nH2_ne_values[:,i6]
								if H2_steps==1:
									d_nH2_ne_values = np.array([1])
								else:
									d_nH2_ne_values = np.array([*np.diff(nH2_ne_values[[0,1]]),*(np.diff(nH2_ne_values)[:-1]/2+np.diff(nH2_ne_values)[1:]/2),*np.diff(nH2_ne_values[[-2,-1]])])
								hypervolume_of_each_combination[:,:,:,i6] *= d_nH2_ne_values/np.sum(d_nH2_ne_values)
							hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (0,4,2,3,1))	# H, H2, H2p, ne*Te, Hm
							for i6 in range(TS_ne_steps*TS_Te_steps):
								nHm_nH2_values = record_nHm_nH2_values[:,i6]
								# d_nHm_nH2_values = np.array([*np.diff(nHm_nH2_values[[0,1]]),*(np.diff(nHm_nH2_values)[:-1]/2+np.diff(nHm_nH2_values)[1:]/2),*np.diff(nHm_nH2_values[[-2,-1]])])/(np.diff(nHm_nH2_values[[0,-1]])+np.diff(nHm_nH2_values[[0,1]])/2+np.diff(nHm_nH2_values[[-2,-1]])/2)
								d_nHm_nH2_values = np.ones_like(nHm_nH2_values)
								hypervolume_of_each_combination[:,:,:,i6] *= d_nHm_nH2_values/np.sum(d_nHm_nH2_values)
							hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (0,4,1,3,2))	# H, Hm, H2, ne*Te, H2p
							for i6 in range(TS_ne_steps*TS_Te_steps):
								nH2p_nH2_values = record_nH2p_nH2_values[:,i6]
								# d_nH2p_nH2_values = np.array([*np.diff(nH2p_nH2_values[[0,1]]),*(np.diff(nH2p_nH2_values)[:-1]/2+np.diff(nH2p_nH2_values)[1:]/2),*np.diff(nH2p_nH2_values[[-2,-1]])])/(np.diff(nH2p_nH2_values[[0,-1]])+np.diff(nH2p_nH2_values[[0,1]])/2+np.diff(nH2p_nH2_values[[-2,-1]])/2)
								d_nH2p_nH2_values = np.ones_like(nH2p_nH2_values)
								hypervolume_of_each_combination[:,:,:,i6] *= d_nH2p_nH2_values/np.sum(d_nH2p_nH2_values)
							hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (1,2,4,3,0))	# Hm, H2, H2p, ne*Te, H
							for i6 in range(TS_ne_steps*TS_Te_steps):
								nH_ne_values = record_nH_ne_values[:,i6]
								# d_nH_ne_values = np.array([*np.diff(nH_ne_values[[0,1]]),*(np.diff(nH_ne_values)[:-1]/2+np.diff(nH_ne_values)[1:]/2),*np.diff(nH_ne_values[[-2,-1]])])/(np.diff(nH_ne_values[[0,-1]])+np.diff(nH_ne_values[[0,1]])/2+np.diff(nH_ne_values[[-2,-1]])/2)
								d_nH_ne_values = np.ones_like(nH_ne_values)
								hypervolume_of_each_combination[:,:,:,i6] *= d_nH_ne_values/np.sum(d_nH_ne_values)
							hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (4,0,1,2,3))	# H, Hm, H2, H2p, ne*Te
							hypervolume_of_each_combination = hypervolume_of_each_combination.reshape((H_steps,Hm_to_find_steps,H2_steps,H2p_to_find_steps,TS_ne_steps,TS_Te_steps))	# H, Hm, H2, H2p, ne, Te
							d_Te_values_array = np.array([*np.diff(Te_values_array[[0,1]]),*(np.diff(Te_values_array)[:-1]/2+np.diff(Te_values_array)[1:]/2),*np.diff(Te_values_array[[-2,-1]])])
							hypervolume_of_each_combination *= d_Te_values_array/np.sum(d_Te_values_array)
							hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (0,1,2,3,5,4))	# H, Hm, H2, H2p, Te, ne
							d_ne_values_array = np.array([*np.diff(ne_values_array[[0,1]]),*(np.diff(ne_values_array)[:-1]/2+np.diff(ne_values_array)[1:]/2),*np.diff(ne_values_array[[-2,-1]])])
							hypervolume_of_each_combination *= d_ne_values_array/np.sum(d_ne_values_array)
							hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (0,1,2,3,5,4))	# H, Hm, H2, H2p, ne, Te

							marginalised_likelihood_log_probs = np.log(np.sum(np.exp(np.float64(likelihood_log_probs))*hypervolume_of_each_combination,axis=(2,4,5)))	# H, Hm, H2, H2p, ne, Te
							del hypervolume_of_each_combination
							marginalised_likelihood_log_probs -= np.max(marginalised_likelihood_log_probs)
							marginalised_likelihood_log_probs -= np.log(np.sum(np.exp(marginalised_likelihood_log_probs)))	# normalisation for logarithmic probabilities
							index_most_likely_marginalised = marginalised_likelihood_log_probs.argmax()
							most_likely_marginalised_nH_ne_index,most_likely_marginalised_nHm_nH2_index,most_likely_marginalised_nH2p_nH2_index = np.unravel_index(index_most_likely_marginalised, marginalised_likelihood_log_probs.shape)
							# most_likely_marginalised_nH2p_nH2_index = index_most_likely_marginalised%H2p_to_find_steps
							# most_likely_marginalised_nHm_nH2_index = (index_most_likely_marginalised-most_likely_marginalised_nH2p_nH2_index)//H2p_to_find_steps%Hm_to_find_steps
							# most_likely_marginalised_nH_ne_index = ((index_most_likely_marginalised-most_likely_marginalised_nH2p_nH2_index)//H2p_to_find_steps-most_likely_marginalised_nHm_nH2_index)//Hm_to_find_steps%H_steps

							# now I cannot do this, because I don't have fixed density intervals
							# most_likely_marginalised_nH2p_ne_value = nH2p_ne_values[most_likely_marginalised_nH2p_nH2_index]
							# most_likely_marginalised_nHm_ne_value = nHm_ne_values[most_likely_marginalised_nHm_nH2_index]
							# most_likely_marginalised_nH_ne_value = nH_ne_values[most_likely_marginalised_nH_ne_index]

							first_most_likely_Te_value = cp.deepcopy(most_likely_Te_value)
							first_most_likely_ne_value = cp.deepcopy(most_likely_ne_value)
							first_most_likely_nH2p_ne_value = cp.deepcopy(most_likely_nH2p_ne_value)
							first_most_likely_nH2_ne_value = cp.deepcopy(most_likely_nH2_ne_value)
							first_most_likely_nHm_ne_value = cp.deepcopy(most_likely_nHm_ne_value)
							first_most_likely_nH_ne_value = cp.deepcopy(most_likely_nH_ne_value)
							first_most_likely_Te_index = cp.deepcopy(most_likely_Te_index)
							first_most_likely_ne_index = cp.deepcopy(most_likely_ne_index)
							first_most_likely_nH2p_nH2_index = cp.deepcopy(most_likely_nH2p_nH2_index)
							first_most_likely_nH2_ne_index = cp.deepcopy(most_likely_nH2_ne_index)
							first_most_likely_nHm_nH2_index = cp.deepcopy(most_likely_nHm_nH2_index)
							first_most_likely_nH_ne_index = cp.deepcopy(most_likely_nH_ne_index)

							first_most_likely_marginalised_nH2p_nH2_index = cp.deepcopy(most_likely_marginalised_nH2p_nH2_index)
							first_most_likely_marginalised_nHm_nH2_index = cp.deepcopy(most_likely_marginalised_nHm_nH2_index)
							first_most_likely_marginalised_nH_ne_index = cp.deepcopy(most_likely_marginalised_nH_ne_index)

							if np.sum(np.isfinite(likelihood_log_probs[0,0,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]))==0:
								nH2p_nH2_index_where_expand = np.nan
							else:
								nH2p_nH2_index_where_expand = likelihood_log_probs[0,0,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index].argmax()

							if np.sum(np.isfinite(likelihood_log_probs[0,:,most_likely_nH2_ne_index,0,most_likely_ne_index,most_likely_Te_index]))==0:
								nHm_nH2_index_where_expand = np.nan
							else:
								nHm_nH2_index_where_expand = likelihood_log_probs[0,:,most_likely_nH2_ne_index,0,most_likely_ne_index,most_likely_Te_index].argmax()

							if np.sum(np.isfinite(likelihood_log_probs[:,0,most_likely_nH2_ne_index,0,most_likely_ne_index,most_likely_Te_index]))==0:
								nH_ne_index_where_expand = np.nan
							else:
								nH_ne_index_where_expand = likelihood_log_probs[:,0,most_likely_nH2_ne_index,0,most_likely_ne_index,most_likely_Te_index].argmax()

							if np.sum(np.isfinite(marginalised_likelihood_log_probs[0,0,:]))==0:
								marginalised_nH2p_nH2_index_where_expand = np.nan
							else:
								marginalised_nH2p_nH2_index_where_expand = marginalised_likelihood_log_probs[0,0,:].argmax()

							if np.sum(np.isfinite(marginalised_likelihood_log_probs[0,:,0]))==0:
								marginalised_nHm_nH2_index_where_expand = np.nan
							else:
								marginalised_nHm_nH2_index_where_expand = marginalised_likelihood_log_probs[0,:,0].argmax()

							if np.sum(np.isfinite(marginalised_likelihood_log_probs[:,0,0]))==0:
								marginalised_nH_ne_index_where_expand = np.nan
							else:
								marginalised_nH_ne_index_where_expand = marginalised_likelihood_log_probs[:,0,0].argmax()

							del likelihood_log_probs,marginalised_likelihood_log_probs
							additional_steps = 5

							if np.sum(np.isnan([nH2p_nH2_index_where_expand,marginalised_nH2p_nH2_index_where_expand]))==2:
								next_nH2p_nH2_index_up = H2p_to_find_steps-1
							else:
								next_nH2p_nH2_index_up = min(int(np.nanmax([nH2p_nH2_index_where_expand,marginalised_nH2p_nH2_index_where_expand])+2),H2p_to_find_steps-1)
							if np.sum(np.isnan([nH2p_nH2_index_where_expand,marginalised_nH2p_nH2_index_where_expand]))==2:
								next_nH2p_nH2_index_down = 0
							else:
								next_nH2p_nH2_index_down = max(int(np.nanmin([nH2p_nH2_index_where_expand,marginalised_nH2p_nH2_index_where_expand])-2),0)
							next_nH2p_nH2_index_expansions_within_interval = int(additional_steps/(next_nH2p_nH2_index_up-next_nH2p_nH2_index_down))

							if np.sum(np.isnan([nHm_nH2_index_where_expand,marginalised_nHm_nH2_index_where_expand]))==2:
								next_nHm_nH2_index_up = Hm_to_find_steps-1
							else:
								next_nHm_nH2_index_up = min(int(np.nanmax([nHm_nH2_index_where_expand,marginalised_nHm_nH2_index_where_expand])+2),Hm_to_find_steps-1)
							if np.sum(np.isnan([nHm_nH2_index_where_expand,marginalised_nHm_nH2_index_where_expand]))==2:
								next_nHm_nH2_index_down = 0
							else:
								next_nHm_nH2_index_down = max(int(np.nanmin([nHm_nH2_index_where_expand,marginalised_nHm_nH2_index_where_expand])-2),0)
							next_nHm_nH2_index_expansions_within_interval = int(additional_steps/(next_nHm_nH2_index_up-next_nHm_nH2_index_down))

							if np.sum(np.isnan([nH_ne_index_where_expand,marginalised_nH_ne_index_where_expand]))==2:
								next_nH_ne_index_up = H_steps-1
							else:
								next_nH_ne_index_up = min(int(np.nanmax([nH_ne_index_where_expand,marginalised_nH_ne_index_where_expand])+2),H_steps-1)
							if np.sum(np.isnan([nH_ne_index_where_expand,marginalised_nH_ne_index_where_expand]))==2:
								next_nH_ne_index_down = 0
							else:
								next_nH_ne_index_down = max(int(np.nanmin([nH_ne_index_where_expand,marginalised_nH_ne_index_where_expand])-2),0)
							next_nH_ne_index_expansions_within_interval = int(additional_steps/(next_nH_ne_index_up-next_nH_ne_index_down))


							next_nHm_nH2_index_up_2 = min(most_likely_marginalised_nHm_nH2_index+2,Hm_to_find_steps-1)
							next_nHm_nH2_index_down_2 = max(most_likely_marginalised_nHm_nH2_index-2,0)

							next_nHm_nH2_index_up_3 = min(most_likely_nHm_nH2_index+2,Hm_to_find_steps-1)
							next_nHm_nH2_index_down_3 = max(most_likely_nHm_nH2_index-2,0)

							temp_1 = np.arange(next_nHm_nH2_index_down,next_nHm_nH2_index_up)
							temp_2 = np.arange(next_nHm_nH2_index_down_2,next_nHm_nH2_index_up_2)
							temp_3 = np.arange(next_nHm_nH2_index_down_3,next_nHm_nH2_index_up_3)
							next_nHm_nH2_index_expansions_within_interval_2 = int(2*additional_steps/len([*temp_2,*temp_3]))
							output = []
							for i in range(Hm_to_find_steps-1):
								temp_5 = 0
								if i in temp_1:
									temp_5 += next_nHm_nH2_index_expansions_within_interval
								if i in temp_2:
									temp_5 += next_nHm_nH2_index_expansions_within_interval_2
								if i in temp_3:
									temp_5 += next_nHm_nH2_index_expansions_within_interval_2
								output.append(temp_5)
							how_expand_nHm_nH2_indexes = np.array(output)
							how_much_expand_nHm_nH2_indexes = np.sum(how_expand_nHm_nH2_indexes)
							additional_nHm_nH2_point = []
							for i in range(Hm_to_find_steps-1):
								additional_nHm_nH2_point.append(False)
								if how_expand_nHm_nH2_indexes[i]!=0:
									temp=how_expand_nHm_nH2_indexes[i]
									while temp>0:
										additional_nHm_nH2_point.append(True)
										temp-=1
							additional_nHm_nH2_point.append(False)


							next_nH_ne_index_up_2 = min(most_likely_marginalised_nH_ne_index+2,H_steps-1)
							next_nH_ne_index_down_2 = max(most_likely_marginalised_nH_ne_index-2,0)

							next_nH_ne_index_up_3 = min(most_likely_nH_ne_index+2,H_steps-1)
							next_nH_ne_index_down_3 = max(most_likely_nH_ne_index-2,0)

							temp_1 = np.arange(next_nH_ne_index_down,next_nH_ne_index_up)
							temp_2 = np.arange(next_nH_ne_index_down_2,next_nH_ne_index_up_2)
							temp_3 = np.arange(next_nH_ne_index_down_3,next_nH_ne_index_up_3)
							next_nH_ne_index_expansions_within_interval_2 = int(2*additional_steps/len([*temp_2,*temp_3]))
							output = []
							for i in range(H_steps-1):
								temp_5 = 0
								if i in temp_1:
									temp_5 += next_nH_ne_index_expansions_within_interval
								if i in temp_2:
									temp_5 += next_nH_ne_index_expansions_within_interval_2
								if i in temp_3:
									temp_5 += next_nH_ne_index_expansions_within_interval_2
								output.append(temp_5)
							how_expand_nH_ne_indexes = np.array(output)
							how_much_expand_nH_ne_indexes = np.sum(how_expand_nH_ne_indexes)
							additional_nH_ne_point = []
							for i in range(H_steps-1):
								additional_nH_ne_point.append(False)
								if how_expand_nH_ne_indexes[i]!=0:
									temp=how_expand_nH_ne_indexes[i]
									while temp>0:
										additional_nH_ne_point.append(True)
										temp-=1
							additional_nH_ne_point.append(False)
							existing_nH_ne_point = np.logical_not(additional_nH_ne_point)

							next_nH2p_nH2_index_up_2 = min(most_likely_marginalised_nH2p_nH2_index+2,H2p_to_find_steps-1)
							next_nH2p_nH2_index_down_2 = max(most_likely_marginalised_nH2p_nH2_index-2,0)

							next_nH2p_nH2_index_up_3 = min(most_likely_nH2p_nH2_index+2,H2p_to_find_steps-1)
							next_nH2p_nH2_index_down_3 = max(most_likely_nH2p_nH2_index-2,0)

							temp_1 = np.arange(next_nH2p_nH2_index_down,next_nH2p_nH2_index_up)
							temp_2 = np.arange(next_nH2p_nH2_index_down_2,next_nH2p_nH2_index_up_2)
							temp_3 = np.arange(next_nH2p_nH2_index_down_3,next_nH2p_nH2_index_up_3)
							next_nH2p_nH2_index_expansions_within_interval_2 = int(2*additional_steps/len([*temp_2,*temp_3]))
							output = []
							for i in range(H2p_to_find_steps-1):
								temp_5 = 0
								if i in temp_1:
									temp_5 += next_nH2p_nH2_index_expansions_within_interval
								if i in temp_2:
									temp_5 += next_nH2p_nH2_index_expansions_within_interval_2
								if i in temp_3:
									temp_5 += next_nH2p_nH2_index_expansions_within_interval_2
								output.append(temp_5)
							how_expand_nH2p_nH2_indexes = np.array(output)
							how_much_expand_nH2p_nH2_indexes = np.sum(how_expand_nH2p_nH2_indexes)
							additional_nH2p_nH2_point = []
							for i in range(H2p_to_find_steps-1):
								additional_nH2p_nH2_point.append(False)
								if how_expand_nH2p_nH2_indexes[i]!=0:
									temp=how_expand_nH2p_nH2_indexes[i]
									while temp>0:
										additional_nH2p_nH2_point.append(True)
										temp-=1
							additional_nH2p_nH2_point.append(False)

							first_most_likely_nH2p_nH2_index = first_most_likely_nH2p_nH2_index + np.sum(how_expand_nH2p_nH2_indexes[:first_most_likely_nH2p_nH2_index])
							first_most_likely_nHm_nH2_index = first_most_likely_nHm_nH2_index + np.sum(how_expand_nHm_nH2_indexes[:first_most_likely_nHm_nH2_index])
							first_most_likely_nH_ne_index = first_most_likely_nH_ne_index + np.sum(how_expand_nH_ne_indexes[:first_most_likely_nH_ne_index])

							first_most_likely_marginalised_nH2p_nH2_index = first_most_likely_marginalised_nH2p_nH2_index + np.sum(how_expand_nH2p_nH2_indexes[:first_most_likely_marginalised_nH2p_nH2_index])
							first_most_likely_marginalised_nHm_nH2_index = first_most_likely_marginalised_nHm_nH2_index + np.sum(how_expand_nHm_nH2_indexes[:first_most_likely_marginalised_nHm_nH2_index])
							first_most_likely_marginalised_nH_ne_index = first_most_likely_marginalised_nH_ne_index + np.sum(how_expand_nH_ne_indexes[:first_most_likely_marginalised_nH_ne_index])


							calculated_emission_log_probs_expanded = -np.ones((H_steps+how_much_expand_nH_ne_indexes,Hm_to_find_steps+how_much_expand_nHm_nH2_indexes,H2_steps,H2p_to_find_steps+how_much_expand_nH2p_nH2_indexes,TS_ne_steps,TS_Te_steps),dtype=np.float32)*np.inf	# H, Hm, H2, H2p, ne, Te
							# calculated_emission_error = np.ones((to_find_steps,to_find_steps))	# Hm, H2p
							# nHp_ne_value = 1
							# total_nHp_ne_value = (((recombination_internal[n_list_all - 4] * nHp_ne_value*np.ones((TS_steps,TS_steps)) ).T/multiplicative_factor).T).astype('float').T.reshape((TS_steps*TS_steps,len(n_list_all)))
							# total_nHp_ne_value = recombination_internal * nHp_ne_value
							coeff_3_record_2 = []
							coeff_4_record_2 = []
							record_nH_ne_values = []
							record_nH_ne_log_prob = []
							for i_Te_for_nH_ne,Te_for_nH_ne in enumerate(Te_values_array):
								nH_ne_values = nH_ne_values_Te_expanded(Te_for_nH_ne,H_steps,how_expand_nH_ne_indexes)
								record_nH_ne_values.append(nH_ne_values)
								nH_ne_log_probs = nH_ne_log_probs_Te(Te_for_nH_ne,nH_ne_values)
								record_nH_ne_log_prob.append(nH_ne_log_probs)
							total_nH_ne_value = multiplicative_factor*np.array([[record_nH_ne_values]*TS_ne_steps]*len(n_list_all)).T	# H, Te, ne, lines
							total_nH_ne_value = np.transpose(total_nH_ne_value, (0,1,3,2))*(ne_values_array ** 2)	# H, Te, lines, ne
							total_nH_ne_value = excitation_internal*(np.transpose(total_nH_ne_value, (0,3,1,2))).reshape((H_steps+how_much_expand_nH_ne_indexes,TS_ne_steps*TS_Te_steps,len(n_list_all)))	# H, ne*Te, lines
							total_nH_ne_value = np.float32(np.transpose(np.array([total_nH_ne_value.tolist()]*H2_steps),(1,0,2,3)))	# H, H2, ne*Te, lines
							record_nH_ne_values = (np.array([record_nH_ne_values]*TS_ne_steps).reshape((TS_ne_steps*TS_Te_steps,H_steps+how_much_expand_nH_ne_indexes))).T
							record_nH_ne_log_prob = (np.array([record_nH_ne_log_prob]*TS_ne_steps).reshape((TS_ne_steps*TS_Te_steps,H_steps+how_much_expand_nH_ne_indexes))).T
							nH_ne_excited_states_atomic = [[[np.float32(record_nH_ne_values*np.sum(excitation_full,axis=-1)*ne_values.flatten()).tolist()]*(H2p_to_find_steps+how_much_expand_nH2p_nH2_indexes)]*H2_steps]*(Hm_to_find_steps+how_much_expand_nHm_nH2_indexes)
							nH_ne_excited_states_atomic = np.transpose(nH_ne_excited_states_atomic, (3,0,1,2,4))
							nH_ne_excited_state_atomic_2 = [[[np.float32(record_nH_ne_values*excitation_full[:,0]*ne_values.flatten()).tolist()]*(H2p_to_find_steps+how_much_expand_nH2p_nH2_indexes)]*H2_steps]*(Hm_to_find_steps+how_much_expand_nHm_nH2_indexes)
							nH_ne_excited_state_atomic_2 = np.transpose(nH_ne_excited_state_atomic_2, (3,0,1,2,4))
							nH_ne_excited_state_atomic_3 = [[[np.float32(record_nH_ne_values*excitation_full[:,0]*ne_values.flatten()).tolist()]*(H2p_to_find_steps+how_much_expand_nH2p_nH2_indexes)]*H2_steps]*(Hm_to_find_steps+how_much_expand_nHm_nH2_indexes)
							nH_ne_excited_state_atomic_3 = np.transpose(nH_ne_excited_state_atomic_3, (3,0,1,2,4))
							nH_ne_excited_state_atomic_4 = [[[np.float32(record_nH_ne_values*excitation_full[:,0]*ne_values.flatten()).tolist()]*(H2p_to_find_steps+how_much_expand_nH2p_nH2_indexes)]*H2_steps]*(Hm_to_find_steps+how_much_expand_nHm_nH2_indexes)
							nH_ne_excited_state_atomic_4 = np.transpose(nH_ne_excited_state_atomic_4, (3,0,1,2,4))
							power_rad_atomic_visible = [[[np.float32(record_nH_ne_values*np.sum(excitation_full*multiplicative_factor_visible_light_full_full,axis=-1)*ne_values.flatten()).tolist()]*(H2p_to_find_steps+how_much_expand_nH2p_nH2_indexes)]*H2_steps]*(Hm_to_find_steps+how_much_expand_nHm_nH2_indexes)
							power_rad_atomic_visible = np.transpose(power_rad_atomic_visible, (3,0,1,2,4))
							# total_nH_ne_value = []
							# for i1,nH_ne_value in enumerate(nH_ne_values):
							# 	total_nH_ne_value.append(np.ones((len(nH2_ne_values),*np.shape(excitation_internal)))*excitation_internal *  nH_ne_value)
							# 	# total_nH_ne_value.append(excitation_internal *  nH_ne_value)
							# 	# total_nH_ne_value = excitation_internal *  nH_ne_value
							# total_nH_ne_value = np.array(total_nH_ne_value)
							# total_nH_ne_value = np.transpose(total_nH_ne_value*multiplicative_factor, (0,1,3,2))	# H, H2, lines, ne*Te
							# total_nH_ne_value = total_nH_ne_value * (ne_values.flatten() ** 2)
							# total_nH_ne_value = np.float32(np.transpose(total_nH_ne_value, (0,1,3,2)))	# H, H2, ne*Te, lines
							if False:	# redundant, done in the first loops
								total_nH2_ne_value = []
								record_nH2_ne_values = []
								record_nH2_ne_log_prob = []
								for i_Te_for_nH2_ne,Te_for_nH2_ne in enumerate(Te_values_array):
									nH2_ne_values = nH2_ne_values_Te(Te_for_nH2_ne,H2_steps)
									record_nH2_ne_values.append(nH2_ne_values)
									nH2_ne_log_probs = nH2_ne_log_probs_Te(Te_for_nH2_ne,nH2_ne_values)
									record_nH2_ne_log_prob.append(nH2_ne_log_probs)
								total_nH2_ne_value = multiplicative_factor*np.array([[record_nH2_ne_values]*TS_ne_steps]*len(n_list_all)).T	# H2, Te, ne, lines
								total_nH2_ne_value = np.transpose(total_nH2_ne_value, (0,1,3,2))*(ne_values_array ** 2)	# H2, Te, lines, ne
								total_nH2_ne_value = np.float32(coeff_2*(np.transpose(total_nH2_ne_value, (0,3,1,2))).reshape((H2_steps,TS_ne_steps*TS_Te_steps,len(n_list_all))))	# H2, ne*Te, lines
								record_nH2_ne_values = (np.array([record_nH2_ne_values]*TS_ne_steps).reshape((TS_ne_steps*TS_Te_steps,H2_steps))).T
								record_nH2_ne_log_prob = (np.array([record_nH2_ne_log_prob]*TS_ne_steps).reshape((TS_ne_steps*TS_Te_steps,H2_steps))).T
							# total_nH2_ne_value = []
							# for i4,nH2_ne_value in enumerate(nH2_ne_values):
							# 	total_nH2_ne_value.append( coeff_2 *  nH2_ne_value )
							# 	# total_nH2_ne_value = nH2_ne_value * coeff_2
							# total_nH2_ne_value = np.array(total_nH2_ne_value)
							# total_nH2_ne_value = np.transpose(total_nH2_ne_value*multiplicative_factor, (0,2,1))	# H2, lines, ne*Te
							# total_nH2_ne_value = total_nH2_ne_value * (ne_values.flatten() ** 2)
							# total_nH2_ne_value = np.float32(np.transpose(total_nH2_ne_value, (0,2,1)))	# H2, ne*Te, lines
							total_nH_nH2_ne_values = total_nH_ne_value + total_nH2_ne_value
							record_nH2p_nH2_values = nH2p_nH2_values_Te_ne_expanded(Te_values.flatten(),ne_values.flatten(),H2p_to_find_steps,how_expand_nH2p_nH2_indexes,H2_suppression=H2_suppression)
							# record_nH2p_nH2_values_initial = nH2p_nH2_values_Te_ne(Te_values.flatten(),ne_values.flatten(),H2p_to_find_steps,H2_suppression=H2_suppression)
							record_nH2p_ne_values = np.transpose([record_nH2_ne_values.tolist()]*(H2p_to_find_steps+how_much_expand_nH2p_nH2_indexes) ,(1,0,2)) * record_nH2p_nH2_values
							record_nHm_nH2_values = nHm_nH2_values_Te_expanded(Te_values.flatten(),Hm_to_find_steps,how_expand_nHm_nH2_indexes,H2_suppression=H2_suppression)
							# record_nHm_nH2_values_initial = nHm_nH2_values_Te(Te_values.flatten(),Hm_to_find_steps,H2_suppression=H2_suppression)
							record_nHm_ne_values = np.transpose([record_nH2_ne_values.tolist()]*(Hm_to_find_steps+how_much_expand_nHm_nH2_indexes) ,(1,0,2)) * record_nHm_nH2_values

							record_nHp_ne_values = 1 - record_nH2p_ne_values + np.transpose([record_nHm_ne_values.tolist()]*(H2p_to_find_steps+how_much_expand_nH2p_nH2_indexes), (2,1,0,3))	# H, Hm, H2, H2p, ne, Te
							record_nHp_ne_values[record_nHp_ne_values<=0] = 1e-15

							# total_nH2p_ne_values = np.array([record_nH2_ne_values.T.tolist()]*(H2p_to_find_steps+how_much_expand_nH2p_nH2_indexes)).T * record_nH2p_nH2_values.T
							# total_nH2p_ne_values = np.transpose(total_nH2p_ne_values, (0,2,1)) * (ne_values.flatten() ** 2)
							# total_nH2p_ne_values = np.float32(np.array([total_nH2p_ne_values.T]*len(n_list_all)).T * (coeff_1*multiplicative_factor))
							# total_nH_nH2_ne_values = np.transpose(np.float32([total_nH_nH2_ne_values.tolist()]*(H2p_to_find_steps+how_much_expand_nH2p_nH2_indexes)), (1,2,0,3,4))
							# total_nH_nH2_nH2p_ne_values = total_nH_nH2_ne_values + total_nH2p_ne_values
							recombination_full_sum = np.sum(recombination_full,axis=-1)
							recombination_full_sum_visible = np.sum(recombination_full*multiplicative_factor_visible_light_full_full,axis=-1)
							for i4 in range(H2_steps):
								# if include_particles_limitation and False:
								# 	selected_part_balance[:,:,np.arange(H2_steps)!=i4]=False
								for i5 in range(H2p_to_find_steps+how_much_expand_nH2p_nH2_indexes):
									nH2p_ne_values = record_nH2p_ne_values[i4,i5]
									total_nH2p_ne_values = np.float32(nH2p_ne_values *(ne_values.flatten() ** 2) * (coeff_1*multiplicative_factor).T).T
									if additional_nH2p_nH2_point[i5]:
										coeff_4 = From_Hn_with_H2p_pop_coeff_full_extra(np.array([Te_values.flatten(),T_H2p_values,T_Hm_values,ne_values.flatten(),nH2p_ne_values*ne_values.flatten()]).T,total_wavelengths)
									else:
										i5_old = i5-np.sum(additional_nH2p_nH2_point[:i5+1])
										coeff_4 = coeff_4_record[i4*H2p_to_find_steps + i5_old]
									coeff_4_record_2.append(coeff_4)
									coeff_4 = cp.deepcopy(coeff_4[:,n_list_all-2])
									total_nH_nH2_nH2p_ne_values = total_nH_nH2_ne_values[:,i4] + total_nH2p_ne_values
								# total_nH2p_ne_value = nH2p_ne_value * coeff_1
								# total_nH2p_ne_value = (total_nH2p_ne_value*multiplicative_factor).T	# lines, ne*Te
								# total_nH2p_ne_value = total_nH2p_ne_value * (ne_values.flatten() ** 2)
								# total_nH2p_ne_value = np.float32(total_nH2p_ne_value.T)	# ne*Te, lines
								# total_nH_nH2_nH2p_ne_value = total_nH_nH2_ne_value + total_nH2p_ne_value
								# coeff_4 = From_Hn_with_H2p_pop_coeff_full_extra(np.array([Te_values.flatten(),T_H2p_values,T_Hm_values,ne_values.flatten(),nH2p_ne_value*ne_values.flatten()]).T,n_list_all)
									for i3 in range(Hm_to_find_steps+how_much_expand_nHm_nH2_indexes):
										# if include_particles_limitation and False:
										# 	selected_part_balance[:,np.arange(Hm_to_find_steps+how_much_expand_nHm_nH2_indexes)!=i3]=False
										nHm_ne_values = record_nHm_ne_values[i4,i3]
										nHp_ne_values = 1 - nH2p_ne_values + nHm_ne_values
										nHp_ne_good = nHp_ne_values>0
										nHp_ne_bad = nHp_ne_values<0
										nHp_ne_values[nHp_ne_bad]=0
										total_nHp_ne_values = (recombination_internal.T * nHp_ne_values).T
										nH_ne_excited_states_atomic[:,i3,i4,i5] += np.float32(recombination_full_sum * nHp_ne_values * ne_values.flatten())
										nH_ne_excited_state_atomic_2[:,i3,i4,i5] += np.float32(recombination_full[:,0] * nHp_ne_values * ne_values.flatten())
										nH_ne_excited_state_atomic_3[:,i3,i4,i5] += np.float32(recombination_full[:,1] * nHp_ne_values * ne_values.flatten())
										nH_ne_excited_state_atomic_4[:,i3,i4,i5] += np.float32(recombination_full[:,2] * nHp_ne_values * ne_values.flatten())
										power_rad_atomic_visible[:,i3,i4,i5] += np.float32(recombination_full_sum_visible * nHp_ne_values * (ne_values.flatten()**2))
										if additional_nH2p_nH2_point[i5] or additional_nHm_nH2_point[i3]:
											coeff_3 = np.zeros((len(nHp_ne_values),len(total_wavelengths)))
											coeff_3[nHp_ne_good] = From_Hn_with_Hp_pop_coeff_full_extra(np.array([Te_values.flatten()[nHp_ne_good],T_Hp_values[nHp_ne_good],T_Hm_values[nHp_ne_good],ne_values.flatten()[nHp_ne_good],(nHp_ne_values*ne_values.flatten())[nHp_ne_good]]).T ,total_wavelengths)
										else:
											i3_old = i3-np.sum(additional_nHm_nH2_point[:i3+1])
											coeff_3 = coeff_3_record[i4*Hm_to_find_steps*H2p_to_find_steps + i5_old*Hm_to_find_steps + i3_old]
											# all_net_e_destruction[:,i3,i4,i5]=record_all_net_e_destruction[]
										coeff_3_record_2.append(coeff_3)
										coeff_3 = cp.deepcopy(coeff_3[:,n_list_all-2])
										# coeff_3[np.logical_and(nHp_ne_mid,nHp_ne_bad)]=0
										total_nHm_ne_values = (nHm_ne_values*( coeff_3 + coeff_4 ).T).T
										total = total_nHp_ne_values+total_nHm_ne_values# +total_nH3p_ne_value	# H, H2, ne*Te, lines
										total = (total*multiplicative_factor).T	# lines, ne*Te
										total = np.float32(total * (ne_values.flatten() ** 2))
										calculated_emission = total_nH_nH2_nH2p_ne_values + total.T	# H, H2, ne*Te, lines
										temp = np.float32(-0.5*np.sum(((calculated_emission - inverted_profiles_crop_restrict)/inverted_profiles_crop_sigma_restrict)**2,axis=-1))
										nHp_ne_good = np.logical_not(nHp_ne_bad).reshape((TS_ne_steps,TS_Te_steps))
										# nHp_ne_good = nHp_ne_good.reshape((TS_ne_steps,TS_Te_steps))
										calculated_emission_log_probs_expanded[:,i3,i4,i5,nHp_ne_good]= temp.reshape((H_steps+how_much_expand_nH_ne_indexes,TS_ne_steps,TS_Te_steps))[:,nHp_ne_good]
										del temp,calculated_emission
										# if include_particles_limitation and False:
										# 	if additional_nH2p_nH2_point[i5] or additional_nHm_nH2_point[i3]:
										# 		x_partial =np.array([record_nH_ne_values[:,nHp_ne_good.flatten()].flatten(),np.array([record_nHm_nH2_values[i3,nHp_ne_good.flatten()].tolist()]*(H_steps+how_much_expand_nH_ne_indexes)).flatten(),np.array([record_nH2_ne_values[i4,nHp_ne_good.flatten()].tolist()]*(H_steps+how_much_expand_nH_ne_indexes)).flatten(),np.array([record_nH2p_nH2_values[i5,nHp_ne_good.flatten()].tolist()]*(H_steps+how_much_expand_nH_ne_indexes)).flatten(),np.array([1e-20*ne_values.flatten()[nHp_ne_good.flatten()]]*(H_steps+how_much_expand_nH_ne_indexes)).flatten(),np.array([Te_values.flatten()[nHp_ne_good.flatten()]]*(H_steps+how_much_expand_nH_ne_indexes)).flatten()]).T
										# 		all_net_e_destruction[:,i3,i4,i5,nHp_ne_good]=(interpolate.interpn((np.log(samples_nH_ne_array),np.log(samples_nHm_nH2_array),np.log(samples_nH2_ne_array),np.log(samples_nH2p_nH2_array),np.log(samples_ne_array),np.log(samples_Te_array)),net_rate_e_destruction,np.log(x_partial), bounds_error=False,fill_value=None)).reshape((H_steps+how_much_expand_nH_ne_indexes,np.sum(nHp_ne_good)))
										# 		all_net_Hp_destruction[:,i3,i4,i5,nHp_ne_good]=(interpolate.interpn((np.log(samples_nH_ne_array),np.log(samples_nHm_nH2_array),np.log(samples_nH2_ne_array),np.log(samples_nH2p_nH2_array),np.log(samples_ne_array),np.log(samples_Te_array)),net_rate_Hp_destruction,np.log(x_partial), bounds_error=False,fill_value=None)).reshape((H_steps+how_much_expand_nH_ne_indexes,np.sum(nHp_ne_good)))
										# 		all_net_Hm_destruction[:,i3,i4,i5,nHp_ne_good]=(interpolate.interpn((np.log(samples_nH_ne_array),np.log(samples_nHm_nH2_array),np.log(samples_nH2_ne_array),np.log(samples_nH2p_nH2_array),np.log(samples_ne_array),np.log(samples_Te_array)),net_rate_Hm_destruction,np.log(x_partial), bounds_error=False,fill_value=None)).reshape((H_steps+how_much_expand_nH_ne_indexes,np.sum(nHp_ne_good)))
										# 		all_net_H2p_destruction[:,i3,i4,i5,nHp_ne_good]=(interpolate.interpn((np.log(samples_nH_ne_array),np.log(samples_nHm_nH2_array),np.log(samples_nH2_ne_array),np.log(samples_nH2p_nH2_array),np.log(samples_ne_array),np.log(samples_Te_array)),net_rate_H2p_destruction,np.log(x_partial), bounds_error=False,fill_value=None)).reshape((H_steps+how_much_expand_nH_ne_indexes,np.sum(nHp_ne_good)))
										# 	else:
										# 		x_partial =np.array([record_nH_ne_values[additional_nH_ne_point][:,nHp_ne_good.flatten()].flatten(),np.array([record_nHm_nH2_values[i3,nHp_ne_good.flatten()].tolist()]*(how_much_expand_nH_ne_indexes)).flatten(),np.array([record_nH2_ne_values[i4,nHp_ne_good.flatten()].tolist()]*(how_much_expand_nH_ne_indexes)).flatten(),np.array([record_nH2p_nH2_values[i5,nHp_ne_good.flatten()].tolist()]*(how_much_expand_nH_ne_indexes)).flatten(),np.array([1e-20*ne_values.flatten()[nHp_ne_good.flatten()]]*(how_much_expand_nH_ne_indexes)).flatten(),np.array([Te_values.flatten()[nHp_ne_good.flatten()]]*(how_much_expand_nH_ne_indexes)).flatten()]).T
										# 		selected_part_balance[existing_nH_ne_point]=False
										# 		selected_part_balance[:,:,:,np.arange(H2p_to_find_steps+how_much_expand_nH2p_nH2_indexes)!=i5]=False
										# 		selected_part_balance[additional_nH_ne_point,i3,i4,i5]=nHp_ne_good
										# 		all_net_e_destruction[selected_part_balance]=(interpolate.interpn((np.log(samples_nH_ne_array),np.log(samples_nHm_nH2_array),np.log(samples_nH2_ne_array),np.log(samples_nH2p_nH2_array),np.log(samples_ne_array),np.log(samples_Te_array)),net_rate_e_destruction,np.log(x_partial), bounds_error=False,fill_value=None))
										# 		all_net_Hp_destruction[selected_part_balance]=(interpolate.interpn((np.log(samples_nH_ne_array),np.log(samples_nHm_nH2_array),np.log(samples_nH2_ne_array),np.log(samples_nH2p_nH2_array),np.log(samples_ne_array),np.log(samples_Te_array)),net_rate_Hp_destruction,np.log(x_partial), bounds_error=False,fill_value=None))
										# 		all_net_Hm_destruction[selected_part_balance]=(interpolate.interpn((np.log(samples_nH_ne_array),np.log(samples_nHm_nH2_array),np.log(samples_nH2_ne_array),np.log(samples_nH2p_nH2_array),np.log(samples_ne_array),np.log(samples_Te_array)),net_rate_Hm_destruction,np.log(x_partial), bounds_error=False,fill_value=None))
										# 		all_net_H2p_destruction[selected_part_balance]=(interpolate.interpn((np.log(samples_nH_ne_array),np.log(samples_nHm_nH2_array),np.log(samples_nH2_ne_array),np.log(samples_nH2p_nH2_array),np.log(samples_ne_array),np.log(samples_Te_array)),net_rate_H2p_destruction,np.log(x_partial), bounds_error=False,fill_value=None))
										# 		selected_part_balance[additional_nH_ne_point]=False
										# 		selected_part_balance[existing_nH_ne_point,i3,i4,i5]=nHp_ne_good
										# 		all_net_e_destruction[selected_part_balance]=record_all_net_e_destruction[:,i3_old,i4,i5_old,nHp_ne_good].flatten()
										# 		all_net_Hp_destruction[selected_part_balance]=record_all_net_Hp_destruction[:,i3_old,i4,i5_old,nHp_ne_good].flatten()
										# 		all_net_Hm_destruction[selected_part_balance]=record_all_net_Hm_destruction[:,i3_old,i4,i5_old,nHp_ne_good].flatten()
										# 		all_net_H2p_destruction[selected_part_balance]=record_all_net_H2p_destruction[:,i3_old,i4,i5_old,nHp_ne_good].flatten()

							del coeff_3_record,coeff_4_record
							# if include_particles_limitation and False:
							# 	del record_all_net_e_destruction,record_all_net_Hp_destruction,record_all_net_Hm_destruction,record_all_net_H2p_destruction,selected_part_balance
							# 	all_net_e_destruction = all_net_e_destruction*area*length*dt/1000
							# 	all_net_Hp_destruction = all_net_Hp_destruction*area*length*dt/1000
							# 	all_net_Hm_destruction = all_net_Hm_destruction*area*length*dt/1000
							# 	all_net_H2p_destruction = all_net_H2p_destruction*area*length*dt/1000

							calculated_emission_log_probs_expanded -= np.max(calculated_emission_log_probs_expanded)
							calculated_emission_log_probs_expanded -= np.log(np.sum(np.exp(calculated_emission_log_probs_expanded)))	# normalisation for logarithmic probabilities

							H_steps = H_steps+how_much_expand_nH_ne_indexes
							Hm_steps = Hm_to_find_steps+how_much_expand_nHm_nH2_indexes
							H2p_steps = H2p_to_find_steps+how_much_expand_nH2p_nH2_indexes

							power_rad_H2p,power_rad_H2,power_rad_excit,power_via_ionisation,power_rad_rec_bremm,power_via_recombination,power_via_brem,power_rec_neutral,power_rad_Hm_H2p,power_rad_Hm_Hp,power_rad_Hm,power_rad_mol,power_heating_rec,tot_rad_power,total_removed_power_atomic,nH_ne_excited_states_mol,nH_ne_excited_state_mol_2,nH_ne_excited_state_mol_3,nH_ne_excited_state_mol_4,power_rad_mol_visible = calc_power_balance_elements_simplified2(H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps,Te_values,ne_values,multiplicative_factor_full_full,multiplicative_factor_visible_light_full_full,record_nH_ne_values,record_nHm_ne_values,record_nH2_ne_values,record_nH2p_ne_values,record_nHp_ne_values,T_Hp_values,T_H2p_values,T_Hm_values,coeff_1_record,coeff_2_record,coeff_3_record_2,coeff_4_record_2)
							total_removed_power = total_removed_power_atomic + power_rad_mol
							del coeff_1_record,coeff_2_record,coeff_3_record_2,coeff_4_record_2

							# thermal_velocity_H = ( (T_H_values*boltzmann_constant_J)/ hydrogen_mass)**0.5
							# temp = read_adf11(scdfile, 'scd', 1, 1, 1, Te_values.flatten(),(ne_values * 10 ** (0 - 6)).flatten())
							# temp[np.isnan(temp)] = 0
							# temp = temp.reshape((np.shape(Te_values))) * (10 ** -6)  # in ionisations m^-3 s-1 / (# / m^3)**2
							# ionization_length_H_CX = thermal_velocity_H.reshape((np.shape(Te_values)))/(temp * ne_values * 1e20 )
							# ionization_length_H_CX = np.where(np.isnan(ionization_length_H_CX), np.inf, ionization_length_H_CX)
							# # ionization_length_H_CX = np.where(np.isinf(ionization_length_H_CX), np.nan, ionization_length_H_CX)
							# # ionization_length_H_CX = np.where(np.isnan(ionization_length_H_CX), np.nanmax(ionization_length_H_CX[np.isfinite(ionization_length_H_CX)]), ionization_length_H_CX)

							temp = read_adf11(ccdfile, 'ccd', 1, 1, 1, Te_values.flatten(),(ne_values * 10 ** (0 - 6)).flatten())
							temp[np.isnan(temp)] = 0
							eff_CX_RR = temp.reshape((np.shape(Te_values))) * (10 ** -6)  # in CX m^-3 s-1 / (# / m^3)**2
							del temp
							eff_CX_RR_int = (eff_CX_RR * (ne_values**2) ).astype('float') * (record_nH_ne_values.reshape((H_steps,TS_ne_steps,TS_Te_steps)))
							eff_CX_RR_int = np.transpose([[[eff_CX_RR_int]*H2p_steps]*H2_steps]*Hm_steps, (3,0,1,2,4,5))
							eff_CX_RR_int = eff_CX_RR_int * np.array([(record_nHp_ne_values.reshape((Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps)))]*H_steps)

							# I separate the complete RR to the RR/nH I need for the CX length
							eff_CX_RR = (eff_CX_RR * (ne_values) ).astype('float32') * (np.ones_like(record_nH_ne_values.reshape((H_steps,TS_ne_steps,TS_Te_steps))))
							eff_CX_RR = np.transpose([[[eff_CX_RR]*H2p_steps]*H2_steps]*Hm_steps, (3,0,1,2,4,5))
							eff_CX_RR = eff_CX_RR * np.array([(record_nHp_ne_values.reshape((Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps)))]*H_steps)	# H, Hm, H2, H2p, ne, Te	# m^-3/s / nH

							# geometric_factor = (ionization_length_H_CX.T / np.abs(2*averaged_profile_sigma[my_time_pos]-r_crop[my_r_pos])).T	# I take diameter = 2 * FWHM
							# geometric_factor[geometric_factor>1] = 1
							delta_t = (T_Hp_values - T_H_values)
							delta_t[delta_t<0] = 0
							local_CX = np.float32(3/2* (delta_t.reshape((TS_ne_steps,TS_Te_steps)) * eV_to_K) * eff_CX_RR_int / J_to_eV)	# W / m^3
							del eff_CX_RR_int,delta_t

							# local_CX = P_RR_Hp_H1s__H1s_Hp(T_Hp_values,T_H_values,ne_values.flatten()*1e-20).reshape(np.shape(ne_values)) * (record_nH_ne_values.reshape((H_steps,TS_ne_steps,TS_Te_steps)))
							# local_CX = np.transpose([[[local_CX]*H2p_steps]*H2_steps]*Hm_steps , (3,0,1,2,4,5))	# H, Hm, H2, H2p, ne, Te

							# area = 2*np.pi*(r_crop[my_r_pos] + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
							temp = np.pi*((r_crop + np.median(np.diff(r_crop))/2)**2)
							area = np.array([temp[0]]+np.diff(temp).tolist())[my_r_pos]
							length = 0.351+target_OES_distance/1000	# mm distance skimmer to OES/TS + OES/TS to target
							total_removed_power_times_volume = total_removed_power * area*length
							# power_penalty = np.zeros_like((total_removed_power_times_volume),dtype=np.float32)
							if False:	# here I do not use any info on the shape of the plasma upstream
								select = total_removed_power_times_volume>interpolated_power_pulse_shape(time_crop[my_time_pos])/source_power_spread
								power_penalty[select] = np.float32(-0.5*((total_removed_power_times_volume[select] - interpolated_power_pulse_shape(time_crop[my_time_pos])/source_power_spread)/(interpolated_power_pulse_shape_std(time_crop[my_time_pos])))**2)
							else:	# H, Hm, H2, H2p, ne, Te
								# max_local_flow_vel = max(1,homogeneous_mach_number[my_time_pos])*upstream_adiabatic_collisional_velocity[my_time_pos,my_r_pos]
								max_local_flow_vel = homogeneous_mach_number[my_time_pos]*upstream_adiabatic_collisional_velocity[my_time_pos,my_r_pos]
								total_removable_power_times_volume_SS = area*(0.5*hydrogen_mass*(max_local_flow_vel**2) + (5*Te_all_upstream[my_time_pos,my_r_pos] + ionisation_potential + dissociation_potential)/eV_to_K*boltzmann_constant_J)*max_local_flow_vel*ne_all_upstream[my_time_pos,my_r_pos]*1e20
								total_removable_power_times_volume_dynamic = area*length/dt*1000*(0.5*hydrogen_mass*(max_local_flow_vel**2) + (5*Te_values + ionisation_potential + dissociation_potential)/eV_to_K*boltzmann_constant_J)*ne_values
								total_removable_power_times_volume = total_removable_power_times_volume_SS + total_removable_power_times_volume_dynamic
								total_removable_power_times_volume = np.array([[[[total_removable_power_times_volume.tolist()]*H2p_steps]*H2_steps]*Hm_steps]*H_steps,dtype=np.float32)
								if False:
									select = total_removed_power_times_volume>total_removable_power_times_volume
									power_penalty[select] = np.float32(-0.5*((total_removed_power_times_volume[select] - total_removable_power_times_volume[select])/total_removable_power_times_volume[select])**2)
								elif True:	# real formula for the probability of an inequality
									# total_removed_power_times_volume_sigma = (0.2*total_removed_power_atomic + 0.5*power_rad_mol) * area*length
									total_removed_power_times_volume_sigma = (( (power_atomic_precision**2)*(power_via_ionisation**2 + power_rad_excit**2 + power_via_recombination**2 + power_rec_neutral**2 + power_via_brem**2) + (power_molecular_precision**2)*(power_rad_Hm**2 + power_rad_H2**2 + power_rad_H2p**2))**0.5) * area*length	#this properly adds in quadrature all the uncertanties
									total_removable_power_times_volume_sigma = (total_removable_power_times_volume*power_budget_precision)
									# power_penalty = np.log( 1 + np.erf( (total_removable_power_times_volume-total_removed_power_times_volume)/(2**0.5 * (total_removed_power_times_volume_sigma**2 + (total_removable_power_times_volume*0.5)**2)**0.5)) )
									power_penalty = np.float32(np.erf( total_removed_power_times_volume/( 2**0.5 * total_removed_power_times_volume_sigma ) ))
									power_penalty += np.float32(np.erf( (total_removable_power_times_volume-total_removed_power_times_volume)/(2**0.5 * (total_removed_power_times_volume_sigma**2 + total_removable_power_times_volume_sigma**2)**0.5)))
									power_penalty = power_penalty/np.float32(0.5 + np.erf( total_removed_power_times_volume/( 2**0.5 * total_removed_power_times_volume_sigma ) ))
									power_penalty = power_penalty/np.float32(0.5 + np.erf( total_removable_power_times_volume/( 2**0.5 * total_removable_power_times_volume_sigma ) ))
									time_lapsed = tm.time()-start_time
									print('worker '+str(current_process())+' marker 1-1 '+str(domain_index)+' in %.3gmin %.3gsec' %(int(time_lapsed/60),int(time_lapsed%60)) +', power penalty - good=%.3g, nan=%.3g, -inf=%.3g, -<0 =%.3g, min=%.3g' %( np.sum(np.isfinite(power_penalty)),np.sum(np.isnan(power_penalty)),np.sum(np.isinf(power_penalty)),np.sum(power_penalty<0) ,np.nanmin(power_penalty)) )
									power_penalty[power_penalty<0]=0
									power_penalty = np.log(power_penalty)
									# power_penalty += -np.log( 1+ np.erf( total_removed_power_times_volume/( 2**0.5 * total_removed_power_times_volume_sigma ) ) ) - np.log( 1+ np.erf( total_removable_power_times_volume/( 2**0.5 * total_removable_power_times_volume*0.5 ) ) )
									del total_removed_power_times_volume_sigma
								power_penalty -= power_penalty.max()

							nH_ne_excited_states = nH_ne_excited_states_atomic + nH_ne_excited_states_mol.reshape((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps*TS_Te_steps))
							nH_ne_excited_state_2 = nH_ne_excited_state_atomic_2 + nH_ne_excited_state_mol_2.reshape((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps*TS_Te_steps))
							nH_ne_excited_state_3 = nH_ne_excited_state_atomic_3 + nH_ne_excited_state_mol_3.reshape((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps*TS_Te_steps))
							nH_ne_excited_state_4 = nH_ne_excited_state_atomic_4 + nH_ne_excited_state_mol_4.reshape((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps*TS_Te_steps))
							power_rad_atomic_visible = power_rad_atomic_visible.reshape((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps))
							total_removed_power_visible = power_rad_atomic_visible + power_rad_mol_visible
							nH_ne_ground_state = np.transpose(record_nH_ne_values - np.transpose(nH_ne_excited_states, (1,2,3,0,4)), (3,0,1,2,4))
							# temp = np.transpose(np.transpose(nH_ne_excited_states, (1,2,3,0,4)) > record_nH_ne_values, (3,0,1,2,4))
							nH_ne_penalty = np.zeros_like(nH_ne_excited_states,dtype=np.float16)
							# nH_ne_penalty[temp==True] = -np.inf
							nH_ne_penalty[nH_ne_ground_state<0] = -np.inf
							nH_ne_ground_state[nH_ne_ground_state<0] = 0
							del temp
							nH_ne_penalty = nH_ne_penalty.reshape((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps))
							nH_ne_excited_states = nH_ne_excited_states.reshape((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps))
							nH_ne_excited_state_2 = nH_ne_excited_state_2.reshape((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps))
							nH_ne_excited_state_3 = nH_ne_excited_state_3.reshape((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps))
							nH_ne_excited_state_4 = nH_ne_excited_state_4.reshape((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps))
							nH_ne_ground_state = nH_ne_ground_state.reshape((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps))
							del nH_ne_excited_states_atomic,nH_ne_excited_state_atomic_2,nH_ne_excited_state_atomic_3,nH_ne_excited_state_atomic_4,nH_ne_excited_states_mol,nH_ne_excited_state_mol_2,nH_ne_excited_state_mol_3,nH_ne_excited_state_mol_4

							likelihood_log_probs = calculated_emission_log_probs_expanded + np.float32(Te_log_probs) + (np.ones((TS_Te_steps,TS_ne_steps),dtype=np.float32)*np.float32(ne_log_probs)).T + np.float32(np.transpose([np.float32(record_nH2_ne_log_prob).reshape((H2_steps,TS_ne_steps,TS_Te_steps)).tolist()]*(H2p_steps),(1,0,2,3)))
							likelihood_log_probs = np.transpose(np.transpose(likelihood_log_probs,(1,2,3,0,4,5)) + np.float32(record_nH_ne_log_prob.reshape((H_steps,TS_ne_steps,TS_Te_steps))) ,(3,0,1,2,4,5)) + power_penalty + nH_ne_penalty
							likelihood_log_probs -= np.max(likelihood_log_probs)
							likelihood_log_probs -= np.log(np.sum(np.exp(likelihood_log_probs)))	# normalisation for logarithmic probabilities
							PDF_matrix_shape.append(np.shape(likelihood_log_probs))
							# PDF_matrix_shape = np.array(PDF_matrix_shape)
							# good_priors = likelihood_log_probs!=-np.inf

							# del power_penalty,nH_ne_penalty

							all_nHm_ne_values = np.array([np.transpose(np.array([record_nHm_ne_values.reshape((H2_steps,Hm_steps,TS_ne_steps,TS_Te_steps)).tolist()]*H2p_steps), (2,1,0,3,4)).tolist()]*H_steps,dtype=np.float32)
							all_nH2p_ne_values = np.array([[record_nH2p_ne_values.reshape((H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps)).tolist()]*Hm_steps]*H_steps,dtype=np.float32)
							all_nHm_nH2_values = np.transpose(np.array([[[record_nHm_nH2_values.reshape((Hm_steps,TS_ne_steps,TS_Te_steps)).tolist()]*H2p_steps]*H2_steps]*H_steps,dtype=np.float32), (0,3,1,2,4,5))
							all_nH2p_nH2_values = np.array([[[record_nH2p_nH2_values.reshape((H2p_steps,TS_ne_steps,TS_Te_steps)).tolist()]*H2_steps]*Hm_steps]*H_steps,dtype=np.float32)
							all_nH2_ne_values = np.array([[np.transpose([record_nH2_ne_values.reshape((H2_steps,TS_ne_steps,TS_Te_steps)).tolist()]*H2p_steps, (1,0,2,3)).tolist()]*Hm_steps]*H_steps, dtype=np.float32)
							all_nH_ne_values = np.array(np.transpose([[[record_nH_ne_values.reshape((H_steps,TS_ne_steps,TS_Te_steps)).tolist()]*H2p_steps]*H2_steps]*Hm_steps, (3,0,1,2,4,5)), dtype=np.float32)
							all_ne_values = np.array([[[[ne_values.tolist()]*H2p_steps]*H2_steps]*Hm_steps]*H_steps,dtype=np.float32)
							all_Te_values = np.array([[[[Te_values.tolist()]*H2p_steps]*H2_steps]*Hm_steps]*H_steps,dtype=np.float32)
							all_nHp_ne_values = np.array([record_nHp_ne_values.reshape((Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps)).tolist()]*H_steps,dtype=np.float32)

							if True:
								all_H_destruction_RR = np.zeros_like((calculated_emission_log_probs_expanded),dtype=np.float32)
								all_H_destruction_RR2 = np.zeros_like((calculated_emission_log_probs_expanded),dtype=np.float32)
								all_H_creation_RR = np.zeros_like((calculated_emission_log_probs_expanded),dtype=np.float32)
								all_H2_destruction_RR = np.zeros_like((calculated_emission_log_probs_expanded),dtype=np.float32)
								all_H2_destruction_RR2 = np.zeros_like((calculated_emission_log_probs_expanded),dtype=np.float32)
								all_H2_creation_RR = np.zeros_like((calculated_emission_log_probs_expanded),dtype=np.float32)
								all_net_H_destruction_RR = np.zeros_like((calculated_emission_log_probs_expanded),dtype=np.float32)
								all_net_H2_destruction_RR = np.zeros_like((calculated_emission_log_probs_expanded),dtype=np.float32)
								if include_particles_limitation:
									all_net_Hp_destruction = np.zeros_like((calculated_emission_log_probs_expanded),dtype=np.float32)
									all_net_e_destruction = np.zeros_like((calculated_emission_log_probs_expanded),dtype=np.float32)
									all_net_Hm_destruction = np.zeros_like((calculated_emission_log_probs_expanded),dtype=np.float32)
									all_net_H2p_destruction = np.zeros_like((calculated_emission_log_probs_expanded),dtype=np.float32)
									all_net_Hp_destruction_sigma = np.ones_like((calculated_emission_log_probs_expanded),dtype=np.float32)
									all_net_e_destruction_sigma = np.ones_like((calculated_emission_log_probs_expanded),dtype=np.float32)
									all_net_Hm_destruction_sigma = np.ones_like((calculated_emission_log_probs_expanded),dtype=np.float32)
									all_net_H2p_destruction_sigma = np.ones_like((calculated_emission_log_probs_expanded),dtype=np.float32)
									# selected_part_balance = np.zeros_like(calculated_emission_log_probs_expanded,dtype=bool)
								temp_coord = np.ones_like(nH_ne_excited_states[:,:,:,:,0,0])
								for i6 in range(TS_Te_steps):
									fractional_population_states_H2 = calc_H2_fractional_population_states(T_H2p_values[i6])
									for i5 in range(TS_ne_steps):
										# local_good_priors = good_priors[:,:,:,:,i5,i6]
										# if np.sum(local_good_priors)==0:
										# 	continue
										# temp_nH_excited_states = nH_ne_excited_states[:,:,:,:,i5,i6]*ne_values_array[i5]*1e-20	# # 10e20 /m^3
										temp_nH_ne_excited_state_2 = nH_ne_excited_state_2[:,:,:,:,i5,i6]
										temp_nH_ne_excited_state_3 = nH_ne_excited_state_3[:,:,:,:,i5,i6]
										temp_nH_ne_excited_state_4 = nH_ne_excited_state_4[:,:,:,:,i5,i6]
										temp_nH_ne_ground_state = nH_ne_ground_state[:,:,:,:,i5,i6]
										arguments = (temp_coord*Te_values_array[i6],temp_coord*T_Hp_values[i6],temp_coord*T_H_values[i6],temp_coord*T_H2_values[i6],temp_coord*T_Hm_values[i6],temp_coord*T_H2p_values[i6],temp_coord*ne_values_array[i5]*1e-20,all_nHp_ne_values[:,:,:,:,i5,i6],all_nH_ne_values[:,:,:,:,i5,i6],all_nH2_ne_values[:,:,:,:,i5,i6],all_nHm_ne_values[:,:,:,:,i5,i6],all_nH2p_ne_values[:,:,:,:,i5,i6],fractional_population_states_H2,temp_nH_ne_ground_state,temp_nH_ne_excited_state_2,temp_nH_ne_excited_state_3,temp_nH_ne_excited_state_4)
										temp,temp_sigma = np.float32(RR_rate_destruction_H(*arguments,particle_molecular_precision,particle_atomic_precision))	# m^-3/s *1e-20
										temp1,temp1_sigma =	np.float32(RR_rate_creation_H(*arguments,particle_molecular_precision,particle_atomic_precision))	# m^-3/s *1e-20
										all_H_destruction_RR[:,:,:,:,i5,i6] = np.float32(temp*1e20/(all_nH_ne_values[:,:,:,:,i5,i6]*ne_values_array[i5]))	# m^-3/s / nH
										all_H_destruction_RR2[:,:,:,:,i5,i6] = np.float32(temp)	# m^-3/s *1e-20
										all_H_creation_RR[:,:,:,:,i5,i6] = np.float32(temp1)	# m^-3/s *1e-20
										all_net_H_destruction_RR[:,:,:,:,i5,i6] = np.float32(temp - temp1)	# m^-3/s *1e-20
										temp,temp_sigma = np.float32(RR_rate_destruction_H2(*arguments,particle_molecular_precision,particle_atomic_precision))	# m^-3/s *1e-20
										temp1,temp1_sigma =	np.float32(RR_rate_creation_H2(*arguments,particle_molecular_precision,particle_atomic_precision))	# m^-3/s *1e-20
										all_H2_destruction_RR[:,:,:,:,i5,i6] = np.float32(temp*1e20/(all_nH2_ne_values[:,:,:,:,i5,i6]*ne_values_array[i5]))	# m^-3/s / nH2
										all_H2_destruction_RR2[:,:,:,:,i5,i6] = np.float32(temp)	# m^-3/s *1e-20
										all_H2_creation_RR[:,:,:,:,i5,i6] = np.float32(temp1)	# m^-3/s *1e-20
										all_net_H2_destruction_RR[:,:,:,:,i5,i6] = np.float32(temp - temp1)	# m^-3/s *1e-20
										if include_particles_limitation:
											temp,temp_sigma = np.float32(RR_rate_destruction_Hp(*arguments,particle_molecular_precision,particle_atomic_precision))	# m^-3/s *1e-20
											temp1,temp1_sigma =	np.float32(RR_rate_creation_Hp(*arguments,particle_molecular_precision,particle_atomic_precision))	# m^-3/s *1e-20
											all_net_Hp_destruction[:,:,:,:,i5,i6] = np.float32(temp - temp1)	# m^-3/s *1e-20
											all_net_Hp_destruction_sigma[:,:,:,:,i5,i6] = 0.5*(temp_sigma**2 + temp1_sigma**2)**0.5	# m^-3/s *1e-20
											temp,temp_sigma = np.float32(RR_rate_destruction_e(*arguments,particle_molecular_precision,particle_atomic_precision))	# m^-3/s *1e-20
											temp1,temp1_sigma =	np.float32(RR_rate_creation_e(*arguments,particle_molecular_precision,particle_atomic_precision))	# m^-3/s *1e-20
											all_net_e_destruction[:,:,:,:,i5,i6] = np.float32(temp - temp1)	# m^-3/s *1e-20
											all_net_e_destruction_sigma[:,:,:,:,i5,i6] = 0.5*(temp_sigma**2 + temp1_sigma**2)**0.5	# m^-3/s *1e-20
											temp,temp_sigma = np.float32(RR_rate_destruction_Hm(*arguments,particle_molecular_precision,particle_atomic_precision))	# m^-3/s *1e-20
											temp1,temp1_sigma =	np.float32(RR_rate_creation_Hm(*arguments,particle_molecular_precision,particle_atomic_precision))	# m^-3/s *1e-20
											all_net_Hm_destruction[:,:,:,:,i5,i6] = np.float32(temp - temp1)	# m^-3/s *1e-20
											all_net_Hm_destruction_sigma[:,:,:,:,i5,i6] = 0.5*(temp_sigma**2 + temp1_sigma**2)**0.5	# m^-3/s *1e-20
											temp,temp_sigma = np.float32(RR_rate_destruction_H2p(*arguments,particle_molecular_precision,particle_atomic_precision))	# m^-3/s *1e-20
											temp1,temp1_sigma =	np.float32(RR_rate_creation_H2p(*arguments,particle_molecular_precision,particle_atomic_precision))	# m^-3/s *1e-20
											all_net_H2p_destruction[:,:,:,:,i5,i6] = np.float32(temp - temp1)	# m^-3/s *1e-20
											all_net_H2p_destruction_sigma[:,:,:,:,i5,i6] = 0.5*(temp_sigma**2 + temp1_sigma**2)**0.5	# m^-3/s *1e-20
								try:
									del temp_coord,temp,temp_sigma,temp1,temp1_sigma
								except:
									print('worker '+str(current_process())+' marker 1-2 '+str(domain_index)+' in %.3gmin %.3gsec' %(int(time_lapsed/60),int(time_lapsed%60)) +' for some reason del temp_coord,temp,temp_sigma,temp1,temp1_sigma failed' )
							if include_particles_limitation:
								particles_penalty = np.zeros_like((likelihood_log_probs),dtype=np.float32)
								all_net_Hp_destruction *= area*length*dt/1000	# *1e-20
								all_net_e_destruction *= area*length*dt/1000	# *1e-20
								all_net_Hm_destruction *= area*length*dt/1000	# *1e-20
								all_net_H2p_destruction *= area*length*dt/1000	# *1e-20
								all_net_Hp_destruction_sigma *= area*length*dt/1000	# *1e-20
								all_net_e_destruction_sigma *= area*length*dt/1000	# *1e-20
								all_net_Hm_destruction_sigma *= area*length*dt/1000	# *1e-20
								all_net_H2p_destruction_sigma *= area*length*dt/1000	# *1e-20
								total_removable_Hp = np.float32(area*dt/1000*ne_all_upstream[my_time_pos,my_r_pos]*max_local_flow_vel + area*length*all_ne_values*all_nHp_ne_values*1e-20)	# I assume no molecules upstream >> nH+/ne~1	# *1e-20
								total_removable_e = np.float32(area*dt/1000*ne_all_upstream[my_time_pos,my_r_pos]*max_local_flow_vel + area*length*all_ne_values*1e-20)	# *1e-20
								if False:
									select = all_net_Hp_destruction>total_removable_Hp
									particles_penalty[select] += np.float32(-0.5*((all_net_Hp_destruction[select] - total_removable_Hp[select])/total_removable_Hp[select])**2)	# 100% uncertanty
									# particles_penalty[np.isnan(particles_penalty)]=-np.inf
									select = all_net_e_destruction>total_removable_e
									particles_penalty[select] += np.float32(-0.5*((all_net_e_destruction[select] - total_removable_e[select])/total_removable_e[select])**2)	# 100% uncertanty
									# particles_penalty[np.isnan(particles_penalty)]=-np.inf
								elif True:
									particles_penalty += np.log( 1 + np.erf( (total_removable_Hp-all_net_Hp_destruction)/(2**0.5 * (all_net_Hp_destruction_sigma**2 + (total_removable_Hp*particle_atomic_budget_precision)**2)**0.5)) )
									particles_penalty += np.log( 1 + np.erf( (total_removable_e-all_net_e_destruction)/(2**0.5 * (all_net_e_destruction_sigma**2 + (total_removable_e*particle_atomic_budget_precision)**2)**0.5)) )
									del all_net_Hp_destruction_sigma,all_net_e_destruction_sigma

								total_removable_Hm = np.float32(area*dt/1000*ne_all_upstream[my_time_pos,my_r_pos]*all_nHm_ne_values*max_local_flow_vel + area*length*all_nHm_ne_values*all_ne_values*1e-20)	# I assume molecules upstream ~ downstream
								total_removable_H2p = np.float32(area*dt/1000*ne_all_upstream[my_time_pos,my_r_pos]*all_nH2p_ne_values*max_local_flow_vel + area*length*all_nH2p_ne_values*all_ne_values*1e-20)	# I assume molecules upstream ~ downstream
								if False:
									select = all_net_Hm_destruction>total_removable_Hm
									particles_penalty[select] += np.float32(-0.5*((all_net_Hm_destruction[select] - total_removable_Hm[select])/(total_removable_Hm[select]*2))**2)	# 200% uncertanty
									# particles_penalty[np.isnan(particles_penalty)]=-np.inf
									select = all_net_H2p_destruction>total_removable_H2p
									particles_penalty[select] += np.float32(-0.5*((all_net_H2p_destruction[select] - total_removable_H2p[select])/(total_removable_H2p[select]*2))**2)	# 200% uncertanty
									particles_penalty[np.isnan(particles_penalty)]=-np.inf
								elif True:
									particles_penalty += np.log( 1 + np.erf( (total_removable_Hm-all_net_Hm_destruction)/(2**0.5 * (all_net_Hm_destruction_sigma**2 + (total_removable_Hm*particle_molecular_budget_precision)**2)**0.5)) )
									particles_penalty += np.log( 1 + np.erf( (total_removable_H2p-all_net_H2p_destruction)/(2**0.5 * (all_net_H2p_destruction_sigma**2 + (total_removable_H2p*particle_molecular_budget_precision)**2)**0.5)) )
									del all_net_Hm_destruction_sigma,all_net_H2p_destruction_sigma
								particles_penalty -= np.nanmax(particles_penalty)
								# particles_penalty[np.isnan(particles_penalty)] = -np.inf

								likelihood_log_probs += particles_penalty
								likelihood_log_probs -= np.max(likelihood_log_probs)
								likelihood_log_probs -= np.log(np.sum(np.exp(likelihood_log_probs)))	# normalisation for logarithmic probabilities

								del all_nHp_ne_values
								# del particles_penalty

							# new addition to calculate CX properly accounting for entering cold neutrals and exiting hot ones
							thermal_velocity_H = ( (T_H_values*boltzmann_constant_J)/ hydrogen_mass)**0.5
							CX_term_1_1 = np.float32(dr*(eff_CX_RR + all_H_destruction_RR)/(thermal_velocity_H.reshape(np.shape(Te_values))))	# m^-3 / nH
							CX_term_1_2 = np.float32(dr*(eff_CX_RR + all_H_destruction_RR))	# m^-3/s / nH * m
							delta_t = (T_Hp_values - T_H_values)
							delta_t[delta_t<0] = 0
							delta_t_low = (T_Hp_values - T_H_values_low)
							delta_t_low[delta_t_low<0] = 0
							CX_term_1_3 = np.float32(3/2* ((delta_t.reshape(np.shape(Te_values))) * boltzmann_constant_J)*(eff_CX_RR)/(thermal_velocity_H.reshape(np.shape(Te_values))))
							CX_term_1_11 = np.float32(3/2* ((delta_t_low.reshape(np.shape(Te_values))) * boltzmann_constant_J)*(eff_CX_RR)/(thermal_velocity_H.reshape(np.shape(Te_values))))
							CX_term_1_4 = np.float32(np.ones_like(eff_CX_RR)/(thermal_velocity_H.reshape(np.shape(Te_values))))
							thermal_velocity_H2 = ( (T_H2_values*boltzmann_constant_J)/ (2*hydrogen_mass))**0.5
							CX_term_1_5 = np.float32(dr*all_H2_destruction_RR/(thermal_velocity_H2.reshape(np.shape(Te_values))))
							CX_term_1_6 = np.float32(np.ones_like(eff_CX_RR)/(thermal_velocity_H2.reshape(np.shape(Te_values))))
							CX_term_1_7 = np.float32(3/2* ((T_Hp_values.reshape(np.shape(Te_values))) * boltzmann_constant_J)*(all_H_creation_RR*1e20))
							CX_term_1_8 =np.float32(dr*all_H2_destruction_RR/(thermal_velocity_H.reshape(np.shape(Te_values))))
							CX_term_1_9 =np.float32(np.ones_like(all_H2_destruction_RR)/(thermal_velocity_H.reshape(np.shape(Te_values))))
							# CX_term_1_10 =np.float32(np.ones_like(all_H2_destruction_RR)/(thermal_velocity_H2.reshape(np.shape(Te_values))))
							del thermal_velocity_H,thermal_velocity_H2

							index_best_fit = calculated_emission_log_probs_expanded.argmax()
							# del calculated_emission_log_probs_expanded	# I need as much memory as mpossible
							best_fit_nH_ne_index,best_fit_nHm_nH2_index,best_fit_nH2_ne_index,best_fit_nH2p_nH2_index,best_fit_ne_index,best_fit_Te_index = np.unravel_index(index_best_fit, calculated_emission_log_probs_expanded.shape)
							# best_fit_Te_index = index_best_fit%TS_Te_steps
							# best_fit_ne_index = (index_best_fit-best_fit_Te_index)//TS_Te_steps%TS_ne_steps
							# best_fit_nH2p_nH2_index = ((index_best_fit-best_fit_Te_index)//TS_Te_steps-best_fit_ne_index)//TS_ne_steps%(H2p_steps)
							# best_fit_nH2_ne_index = (((index_best_fit-best_fit_Te_index)//TS_Te_steps-best_fit_ne_index)//TS_ne_steps-best_fit_nH2p_nH2_index)//(H2p_steps)%H2_steps
							# best_fit_nHm_nH2_index = ((((index_best_fit-best_fit_Te_index)//TS_Te_steps-best_fit_ne_index)//TS_ne_steps-best_fit_nH2p_nH2_index)//(H2p_steps) - best_fit_nH2_ne_index)//H2_steps%(Hm_steps)
							# best_fit_nH_ne_index = (((((index_best_fit-best_fit_Te_index)//TS_Te_steps-best_fit_ne_index)//TS_ne_steps-best_fit_nH2p_nH2_index)//(H2p_steps) - best_fit_nH2_ne_index)//H2_steps - best_fit_nHm_nH2_index)//(Hm_steps)%(H_steps)

							best_fit_Te_value = Te_values[best_fit_ne_index,best_fit_Te_index]
							best_fit_ne_value = ne_values[best_fit_ne_index,best_fit_Te_index]
							best_fit_nH2p_ne_value = record_nH2p_nH2_values[best_fit_nH2p_nH2_index,best_fit_ne_index*best_fit_Te_index]*record_nH2_ne_values[best_fit_nH2_ne_index,best_fit_ne_index*best_fit_Te_index]
							best_fit_nH2_ne_value = record_nH2_ne_values[best_fit_nH2_ne_index,best_fit_ne_index*best_fit_Te_index]
							best_fit_nHm_ne_value = record_nHm_nH2_values[best_fit_nHm_nH2_index,best_fit_ne_index*best_fit_Te_index]*record_nH2_ne_values[best_fit_nH2_ne_index,best_fit_ne_index*best_fit_Te_index]
							best_fit_nH_ne_value = record_nH_ne_values[best_fit_nH_ne_index,best_fit_ne_index*best_fit_Te_index]


							index_most_likely = likelihood_log_probs.argmax()
							most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index = np.unravel_index(index_most_likely, likelihood_log_probs.shape)
							# most_likely_Te_index = index_most_likely%TS_Te_steps
							# most_likely_ne_index = (index_most_likely-most_likely_Te_index)//TS_Te_steps%TS_ne_steps
							# most_likely_nH2p_nH2_index = ((index_most_likely-most_likely_Te_index)//TS_Te_steps-most_likely_ne_index)//TS_ne_steps%(H2p_steps)
							# most_likely_nH2_ne_index = (((index_most_likely-most_likely_Te_index)//TS_Te_steps-most_likely_ne_index)//TS_ne_steps-most_likely_nH2p_nH2_index)//(H2p_steps)%H2_steps
							# most_likely_nHm_nH2_index = ((((index_most_likely-most_likely_Te_index)//TS_Te_steps-most_likely_ne_index)//TS_ne_steps-most_likely_nH2p_nH2_index)//(H2p_steps) - most_likely_nH2_ne_index)//H2_steps%(Hm_steps)
							# most_likely_nH_ne_index = (((((index_most_likely-most_likely_Te_index)//TS_Te_steps-most_likely_ne_index)//TS_ne_steps-most_likely_nH2p_nH2_index)//(H2p_steps) - most_likely_nH2_ne_index)//H2_steps - most_likely_nHm_nH2_index)//(Hm_steps)%(H_steps)
							max_likelihood_log_prob = likelihood_log_probs[most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index]

							most_likely_Te_value = Te_values[most_likely_ne_index,most_likely_Te_index]
							most_likely_ne_value = ne_values[most_likely_ne_index,most_likely_Te_index]
							most_likely_nH2p_ne_value = record_nH2p_nH2_values[most_likely_nH2p_nH2_index,most_likely_ne_index*TS_Te_steps+most_likely_Te_index] * record_nH2_ne_values[most_likely_nH2_ne_index,most_likely_ne_index*TS_Te_steps+most_likely_Te_index]
							most_likely_nH2_ne_value = record_nH2_ne_values[most_likely_nH2_ne_index,most_likely_ne_index*TS_Te_steps+most_likely_Te_index]
							most_likely_nHm_ne_value = record_nHm_nH2_values[most_likely_nHm_nH2_index,most_likely_ne_index*TS_Te_steps+most_likely_Te_index] * record_nH2_ne_values[most_likely_nH2_ne_index,most_likely_ne_index*TS_Te_steps+most_likely_Te_index]
							most_likely_nH_ne_value = record_nH_ne_values[most_likely_nH_ne_index,most_likely_ne_index*TS_Te_steps+most_likely_Te_index]
							most_likely_T_Hm_value = np.exp(TH2_fit_from_simulations(np.log(most_likely_Te_value)))/eV_to_K	# K
							most_likely_T_H2p_value = np.exp(TH2_fit_from_simulations(np.log(most_likely_Te_value)))/eV_to_K	# K
							most_likely_T_Hp_value = most_likely_Te_value/eV_to_K	# K
							most_likely_T_H_value = np.exp(TH_fit_from_simulations(np.log(most_likely_Te_value)))/eV_to_K	# K
							most_likely_power_rad_excit = power_rad_excit.flatten()[index_most_likely]
							most_likely_power_rad_rec_bremm = power_rad_rec_bremm.flatten()[index_most_likely]
							most_likely_power_rad_mol = power_rad_mol.flatten()[index_most_likely]
							most_likely_power_via_ionisation = power_via_ionisation.flatten()[index_most_likely]
							most_likely_power_via_recombination = power_via_recombination.flatten()[index_most_likely]
							most_likely_power_rad_Hm = power_rad_Hm .flatten()[index_most_likely]
							most_likely_power_rad_H2 = power_rad_H2.flatten()[index_most_likely]
							most_likely_power_rad_H2p = power_rad_H2p.flatten()[index_most_likely]

							most_likely_total_removable_power_times_volume = total_removable_power_times_volume[most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index]/area/length
							max_total_removable_power_times_volume = total_removable_power_times_volume.max()/area/length
							min_total_removable_power_times_volume = total_removable_power_times_volume.min()/area/length
							most_likely_local_CX = local_CX[most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index]
							if include_particles_limitation:
								most_likely_net_Hp_destruction = all_net_Hp_destruction[most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index]
								most_likely_total_removable_Hp = total_removable_Hp[most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index]
								most_likely_net_e_destruction = all_net_e_destruction[most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index]
								most_likely_total_removable_e = total_removable_e[most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index]
								most_likely_net_Hm_destruction = all_net_Hm_destruction[most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index]
								most_likely_total_removable_Hm = total_removable_Hm[most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index]
								most_likely_net_H2p_destruction = all_net_H2p_destruction[most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index]
								most_likely_total_removable_H2p = total_removable_H2p[most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index]

							# now I marginalise H2, Te, ne in order to find the most likely Hm, H2p for the next step
							hypervolume_of_each_combination = np.ones((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps*TS_Te_steps),dtype=np.float32)	# H, Hm, H2, H2p, ne*Te
							hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (0,1,3,4,2))	# H, Hm, H2p, ne*Te, H2
							for i6 in range(TS_ne_steps*TS_Te_steps):
								nH2_ne_values = record_nH2_ne_values[:,i6]
								if H2_steps==1:
									d_nH2_ne_values = np.array([1])
								else:
									d_nH2_ne_values = np.array([*np.diff(nH2_ne_values[[0,1]]),*(np.diff(nH2_ne_values)[:-1]/2+np.diff(nH2_ne_values)[1:]/2),*np.diff(nH2_ne_values[[-2,-1]])])
								hypervolume_of_each_combination[:,:,:,i6] *= d_nH2_ne_values/np.sum(d_nH2_ne_values)
							hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (0,4,2,3,1))	# H, H2, H2p, ne*Te, Hm
							for i6 in range(TS_ne_steps*TS_Te_steps):
								nHm_nH2_values = record_nHm_nH2_values[:,i6]
								# d_nHm_nH2_values = np.array([*np.diff(nHm_nH2_values[[0,1]]),*(np.diff(nHm_nH2_values)[:-1]/2+np.diff(nHm_nH2_values)[1:]/2),*np.diff(nHm_nH2_values[[-2,-1]])])/(np.diff(nHm_nH2_values[[0,-1]])+np.diff(nHm_nH2_values[[0,1]])/2+np.diff(nHm_nH2_values[[-2,-1]])/2)
								d_nHm_nH2_values = np.ones_like(nHm_nH2_values)
								hypervolume_of_each_combination[:,:,:,i6] *= d_nHm_nH2_values/np.sum(d_nHm_nH2_values)
							hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (0,4,1,3,2))	# H, Hm, H2, ne*Te, H2p
							for i6 in range(TS_ne_steps*TS_Te_steps):
								nH2p_nH2_values = record_nH2p_nH2_values[:,i6]
								# d_nH2p_nH2_values = np.array([*np.diff(nH2p_nH2_values[[0,1]]),*(np.diff(nH2p_nH2_values)[:-1]/2+np.diff(nH2p_nH2_values)[1:]/2),*np.diff(nH2p_nH2_values[[-2,-1]])])/(np.diff(nH2p_nH2_values[[0,-1]])+np.diff(nH2p_nH2_values[[0,1]])/2+np.diff(nH2p_nH2_values[[-2,-1]])/2)
								d_nH2p_nH2_values = np.ones_like(nH2p_nH2_values)
								hypervolume_of_each_combination[:,:,:,i6] *= d_nH2p_nH2_values/np.sum(d_nH2p_nH2_values)
							hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (1,2,4,3,0))	# Hm, H2, H2p, ne*Te, H
							for i6 in range(TS_ne_steps*TS_Te_steps):
								nH_ne_values = record_nH_ne_values[:,i6]
								# d_nH_ne_values = np.array([*np.diff(nH_ne_values[[0,1]]),*(np.diff(nH_ne_values)[:-1]/2+np.diff(nH_ne_values)[1:]/2),*np.diff(nH_ne_values[[-2,-1]])])/(np.diff(nH_ne_values[[0,-1]])+np.diff(nH_ne_values[[0,1]])/2+np.diff(nH_ne_values[[-2,-1]])/2)
								d_nH_ne_values = np.ones_like(nH_ne_values)
								hypervolume_of_each_combination[:,:,:,i6] *= d_nH_ne_values/np.sum(d_nH_ne_values)
							hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (4,0,1,2,3))	# H, Hm, H2, H2p, ne*Te
							hypervolume_of_each_combination = hypervolume_of_each_combination.reshape((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps))	# H, Hm, H2, H2p, ne, Te
							d_Te_values_array = np.array([*np.diff(Te_values_array[[0,1]]),*(np.diff(Te_values_array)[:-1]/2+np.diff(Te_values_array)[1:]/2),*np.diff(Te_values_array[[-2,-1]])])
							hypervolume_of_each_combination *= d_Te_values_array/np.sum(d_Te_values_array)
							hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (0,1,2,3,5,4))	# H, Hm, H2, H2p, Te, ne
							d_ne_values_array = np.array([*np.diff(ne_values_array[[0,1]]),*(np.diff(ne_values_array)[:-1]/2+np.diff(ne_values_array)[1:]/2),*np.diff(ne_values_array[[-2,-1]])])
							hypervolume_of_each_combination *= d_ne_values_array/np.sum(d_ne_values_array)
							hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (0,1,2,3,5,4))	# H, Hm, H2, H2p, ne, Te

							marginalised_likelihood_log_probs = np.log(np.sum(np.exp(likelihood_log_probs)*np.float32(hypervolume_of_each_combination),axis=(2,4,5)))	# H, Hm, H2, H2p, ne, Te
							marginalised_likelihood_log_probs = marginalised_likelihood_log_probs-np.max(marginalised_likelihood_log_probs)
							marginalised_likelihood_log_probs = marginalised_likelihood_log_probs-np.log(np.sum(np.exp(marginalised_likelihood_log_probs)))	# normalisation for logarithmic probabilities
							index_most_likely_marginalised = marginalised_likelihood_log_probs.argmax()
							most_likely_marginalised_nH_ne_index,most_likely_marginalised_nHm_nH2_index,most_likely_marginalised_nH2p_nH2_index = np.unravel_index(index_most_likely_marginalised, marginalised_likelihood_log_probs.shape)
							# most_likely_marginalised_nH2p_nH2_index = index_most_likely_marginalised%(H2p_steps)
							# most_likely_marginalised_nHm_nH2_index = (index_most_likely_marginalised-most_likely_marginalised_nH2p_nH2_index)//(H2p_steps)%(Hm_steps)
							# most_likely_marginalised_nH_ne_index = ((index_most_likely_marginalised-most_likely_marginalised_nH2p_nH2_index)//(H2p_steps)-most_likely_marginalised_nHm_nH2_index)//(Hm_steps)%(H_steps)

							marginalised_calculated_emission_log_probs_expanded = np.log(np.sum(np.exp(calculated_emission_log_probs_expanded)*np.float32(hypervolume_of_each_combination),axis=(2,4,5)))	# H, Hm, H2, H2p, ne, Te
							marginalised_calculated_emission_log_probs_expanded = marginalised_calculated_emission_log_probs_expanded-np.max(marginalised_calculated_emission_log_probs_expanded)
							marginalised_calculated_emission_log_probs_expanded = marginalised_calculated_emission_log_probs_expanded-np.log(np.sum(np.exp(marginalised_calculated_emission_log_probs_expanded)))	# normalisation for logarithmic probabilities

							marginalised_power_penalty = np.log(np.sum(np.exp(power_penalty)*np.float32(hypervolume_of_each_combination),axis=(2,4,5)))	# H, Hm, H2, H2p, ne, Te
							marginalised_power_penalty = marginalised_power_penalty-np.max(marginalised_power_penalty)
							marginalised_power_penalty = marginalised_power_penalty-np.log(np.sum(np.exp(marginalised_power_penalty)))	# normalisation for logarithmic probabilities

							marginalised_nH_ne_penalty = np.log(np.sum(np.exp(nH_ne_penalty)*np.float32(hypervolume_of_each_combination),axis=(2,4,5)))	# H, Hm, H2, H2p, ne, Te
							marginalised_nH_ne_penalty = marginalised_nH_ne_penalty-np.max(marginalised_nH_ne_penalty)
							marginalised_nH_ne_penalty = marginalised_nH_ne_penalty-np.log(np.sum(np.exp(marginalised_nH_ne_penalty)))	# normalisation for logarithmic probabilities

							marginalised_particles_penalty = np.log(np.sum(np.exp(particles_penalty)*np.float32(hypervolume_of_each_combination),axis=(2,4,5)))	# H, Hm, H2, H2p, ne, Te
							marginalised_particles_penalty = marginalised_particles_penalty-np.max(marginalised_particles_penalty)
							marginalised_particles_penalty = marginalised_particles_penalty-np.log(np.sum(np.exp(marginalised_particles_penalty)))	# normalisation for logarithmic probabilities

							# now I cannot do this, because I don't have fixed density intervals
							# most_likely_marginalised_nH2p_ne_value = nH2p_ne_values[most_likely_marginalised_nH2p_ne_index]
							# most_likely_marginalised_nHm_ne_value = nHm_ne_values[most_likely_marginalised_nHm_ne_index]
							# most_likely_marginalised_nH_ne_value = nH_ne_values[most_likely_marginalised_nH_ne_index]


							# print('best_fit_nH_ne_index %.3g,best_fit_nHm_ne_value %.3g,best_fit_nH2_ne_value %.3g,best_fit_nH2p_ne_value %.3g' %(best_fit_nH_ne_index,best_fit_nHm_ne_value,best_fit_nH2_ne_value,best_fit_nH2p_ne_value))
							time_lapsed = tm.time()-start_time
							print('worker '+str(current_process())+' marker 1 '+str(domain_index)+' in %.3gmin %.3gsec' %(int(time_lapsed/60),int(time_lapsed%60)) + str(np.shape(likelihood_log_probs)) +' - good=%.3g, nan=%.3g, -inf=%.3g' %(np.sum(np.isfinite(likelihood_log_probs)),np.sum(np.isnan(likelihood_log_probs)),np.sum(np.isinf(likelihood_log_probs))) )

							# Here I try to obtain the Probability Distribution Function (PDF) for the radiated power, ionisation and recombination rate
							if (collect_power_PDF or to_print):
								# Approach suggested by Christopher Bowman. integral of the full probability distribution given intervals of given power
								hypervolume_of_each_combination = np.ones((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps*TS_Te_steps),dtype=np.float32)	# H, Hm, H2, H2p, ne*Te
								hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (0,1,3,4,2))	# H, Hm, H2p, ne*Te, H2
								for i6 in range(TS_ne_steps*TS_Te_steps):
									nH2_ne_values = record_nH2_ne_values[:,i6]
									if H2_steps==1:
										d_nH2_ne_values = np.array([1])
									else:
										d_nH2_ne_values = np.array([*np.diff(nH2_ne_values[[0,1]]),*(np.diff(nH2_ne_values)[:-1]/2+np.diff(nH2_ne_values)[1:]/2),*np.diff(nH2_ne_values[[-2,-1]])])
									hypervolume_of_each_combination[:,:,:,i6] *= d_nH2_ne_values/np.sum(d_nH2_ne_values)
								hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (0,4,2,3,1))	# H, H2, H2p, ne*Te, Hm
								for i6 in range(TS_ne_steps*TS_Te_steps):
									nHm_nH2_values = record_nHm_nH2_values[:,i6]
									d_nHm_nH2_values = np.array([*np.diff(nHm_nH2_values[[0,1]]),*(np.diff(nHm_nH2_values)[:-1]/2+np.diff(nHm_nH2_values)[1:]/2),*np.diff(nHm_nH2_values[[-2,-1]])])
									# d_nHm_nH2_values = np.ones_like(nHm_nH2_values)
									hypervolume_of_each_combination[:,:,:,i6] *= d_nHm_nH2_values/np.sum(d_nHm_nH2_values)
								hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (0,4,1,3,2))	# H, Hm, H2, ne*Te, H2p
								for i6 in range(TS_ne_steps*TS_Te_steps):
									nH2p_nH2_values = record_nH2p_nH2_values[:,i6]
									d_nH2p_nH2_values = np.array([*np.diff(nH2p_nH2_values[[0,1]]),*(np.diff(nH2p_nH2_values)[:-1]/2+np.diff(nH2p_nH2_values)[1:]/2),*np.diff(nH2p_nH2_values[[-2,-1]])])
									# d_nH2p_nH2_values = np.ones_like(nH2p_nH2_values)
									hypervolume_of_each_combination[:,:,:,i6] *= d_nH2p_nH2_values/np.sum(d_nH2p_nH2_values)
								hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (1,2,4,3,0))	# Hm, H2, H2p, ne*Te, H
								for i6 in range(TS_ne_steps*TS_Te_steps):
									nH_ne_values = record_nH_ne_values[:,i6]
									d_nH_ne_values = np.array([*np.diff(nH_ne_values[[0,1]]),*(np.diff(nH_ne_values)[:-1]/2+np.diff(nH_ne_values)[1:]/2),*np.diff(nH_ne_values[[-2,-1]])])
									# d_nH_ne_values = np.ones_like(nH_ne_values)
									hypervolume_of_each_combination[:,:,:,i6] *= d_nH_ne_values/np.sum(d_nH_ne_values)
								hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (4,0,1,2,3))	# H, Hm, H2, H2p, ne*Te
								hypervolume_of_each_combination = hypervolume_of_each_combination.reshape((H_steps,Hm_steps,H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps))	# H, Hm, H2, H2p, ne, Te
								d_Te_values_array = np.array([*np.diff(Te_values_array[[0,1]]),*(np.diff(Te_values_array)[:-1]/2+np.diff(Te_values_array)[1:]/2),*np.diff(Te_values_array[[-2,-1]])])
								hypervolume_of_each_combination *= d_Te_values_array/np.sum(d_Te_values_array)
								hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (0,1,2,3,5,4))	# H, Hm, H2, H2p, Te, ne
								d_ne_values_array = np.array([*np.diff(ne_values_array[[0,1]]),*(np.diff(ne_values_array)[:-1]/2+np.diff(ne_values_array)[1:]/2),*np.diff(ne_values_array[[-2,-1]])])
								hypervolume_of_each_combination *= d_ne_values_array/np.sum(d_ne_values_array)
								hypervolume_of_each_combination = np.transpose(hypervolume_of_each_combination, (0,1,2,3,5,4))	# H, Hm, H2, H2p, ne, Te

								likelihood_probs_times_volume = np.float32(np.exp(likelihood_log_probs)*hypervolume_of_each_combination)
								likelihood_probs_times_volume = likelihood_probs_times_volume/np.sum(likelihood_probs_times_volume)
								temp = np.flip(np.sort(likelihood_probs_times_volume.flatten()),axis=0)
								likelihood_probs_times_volume_short_lim = temp[np.abs(np.cumsum(temp)-(1-1e-6)).argmin()]
								del temp
								likelihood_probs_times_volume_short_lim_select = likelihood_probs_times_volume>likelihood_probs_times_volume_short_lim
								likelihood_probs_times_volume_short = (likelihood_probs_times_volume[likelihood_probs_times_volume_short_lim_select].flatten()).astype(np.float32)
								likelihood_probs_times_volume_short = (likelihood_probs_times_volume_short/np.sum(likelihood_probs_times_volume_short)).astype(np.float32)

								min_fraction_of_prob_considered = 0
								non_zero_prob_select = likelihood_probs_times_volume>1*min_fraction_of_prob_considered
								likelihood_probs_times_volume_usefull = likelihood_probs_times_volume[non_zero_prob_select]
								# power_rad_excit,power_rad_rec_bremm,power_rad_mol,power_via_ionisation,power_via_recombination,power_rad_Hm,power_rad_H2,power_rad_H2p = calc_power_balance_elements_simplified(nH_ne_values,nHm_ne_values,nH2_ne_values,nH2p_ne_values,ne_values_array,Te_values_array,Te_values,ne_values,T_Hm_values,T_H2p_values,T_Hp_values)


								time_lapsed = tm.time()-start_time
								print('worker '+str(current_process())+' marker 2 '+str(domain_index)+' in %.3gmin %.3gsec' %(int(time_lapsed/60),int(time_lapsed%60)))

								PDF_intervals = 20

								def build_log_PDF(power,use_all=True,treshold_ratio=1.3,require_dict=False,PDF_intervals=PDF_intervals):
									start_time = tm.time()
									# d_power = min(np.diff(np.log10(np.sort(power[power>0].flatten())[[0,-1]]))/PDF_intervals,0.5)
									if use_all:
										# num_samples = len(power.flatten())
										# power_values = np.sort(power.flatten())[(np.arange(PDF_intervals)*num_samples/PDF_intervals).astype(int)]
										power_short = (power[likelihood_probs_times_volume_short_lim_select].flatten()).astype(np.float32)
										if len(np.unique(power_short))>10000:
											sorted_prob = (np.array([to_sort for _, to_sort in sorted(zip(power_short, likelihood_probs_times_volume_short))])).astype(np.float32)
											sorted_prob = (sorted_prob/np.sum(sorted_prob)).astype(np.float32)
											# print('1')
											power_values = (np.sort(power_short.flatten())).astype(np.float32)
											treshold = 1/10000
											temp = (np.cumsum(sorted_prob)).astype(np.float32)
											# print('2')
											temp1=[]
											temp2 = []
											prev_loc=0
											for i in range(0,10000):
												loc = np.abs(temp-(treshold*i+treshold/2)).argmin()
												temp1.append(power_values[loc])
												if i==0:
													temp2.append(temp[loc])
												else:
													temp2.append(temp[loc]-temp[prev_loc])
												prev_loc=loc
											# temp1.append(power_values[-1])
											# temp2.append(temp[-1]-np.sum(temp2))
											power_values = np.unique(temp1)
											temp1 = np.array(temp1)
											temp2 = np.array(temp2)
											sorted_prob = []
											for value in power_values:
												sorted_prob.append(np.sum(temp2[temp1==value]))
											sorted_prob = np.array(sorted_prob)/np.sum(sorted_prob)
										else:
											power_values = np.unique(power_short)
											sorted_prob = []
											for value in power_values:
												sorted_prob.append(np.sum(likelihood_probs_times_volume_short[power_short==value]))
											sorted_prob = np.array(sorted_prob)/np.sum(sorted_prob)
									else:
										num_samples = np.sum(non_zero_prob_select)
										power_values = np.sort(power[non_zero_prob_select])[(np.arange(PDF_intervals)*num_samples/PDF_intervals).astype(int)]
									# power_values = power_values[power_values>0]
									# print('3')
									if False:
										done=0
										limit = len(power_values)*2
										while (done==0 and limit>0):
											limit-=1
											if limit==1:
												print('failed PDF building in t=%.3g, r=%.3g' %(my_time_pos,my_r_pos))
											for i in range(len(power_values)-1):
												# print(power_values)
												# if (power_values[i+1]/power_values[i]<1.1 and len(power_values)>3):
												if power_values[i]>=0 and power_values[i+1]>=0:
													if power_values[i]!=0 and power_values[i+1]!=0:
														if ((power_values[i+1]/power_values[i]<treshold_ratio or sorted_prob[i]<1e-30) and len(power_values)>PDF_intervals):
															power_values = np.concatenate((power_values[:i+1],power_values[i+2:]))
															sorted_prob = np.concatenate((sorted_prob[:i],[np.sum(sorted_prob[i:i+2])],sorted_prob[i+2:]))
															break
												elif power_values[i]<0 and power_values[i+1]<0:
													if power_values[i]!=0 and power_values[i+1]!=0:
														if ((power_values[i]/power_values[i+1]<treshold_ratio or sorted_prob[i]<1e-30) and len(power_values)>PDF_intervals):
															power_values = np.concatenate((power_values[:i+1],power_values[i+2:]))
															sorted_prob = np.concatenate((sorted_prob[:i],[np.sum(sorted_prob[i:i+2])],sorted_prob[i+2:]))
															break
												if i==len(power_values)-2:
													done=1
									else:
										prob_0 = np.sum(sorted_prob[power_values==0])
										power_values_pos = power_values[power_values>0]
										sorted_prob_pos = sorted_prob[power_values>0]
										# plt.figure()
										# plt.plot(power_values_pos,sorted_prob_pos)
										if power_values.max()>0 :
											while len(power_values_pos)>PDF_intervals*7:
												i = (sorted_prob_pos[1:]+sorted_prob_pos[:-1]).argmin()
												power_values_pos = np.concatenate((power_values_pos[:i],[np.mean(power_values_pos[i:i+2])],power_values_pos[i+2:]))
												sorted_prob_pos = np.concatenate((sorted_prob_pos[:i],[np.sum(sorted_prob_pos[i:i+2])],sorted_prob_pos[i+2:]))
											# plt.plot(power_values_pos,sorted_prob_pos)
											while len(power_values_pos)>PDF_intervals:
												i = (power_values_pos[1:]/power_values_pos[:-1]).argmin()
												power_values_pos = np.concatenate((power_values_pos[:i],[np.mean(power_values_pos[i:i+2])],power_values_pos[i+2:]))
												# sorted_prob_pos = np.concatenate((sorted_prob_pos[:i],[np.sum(sorted_prob_pos[i:i+2])],sorted_prob_pos[i+2:]))
											# plt.plot(power_values_pos,sorted_prob_pos)
											# plt.xscale('log')
											# plt.pause(0.01)
										if power_values.min()<0 :
											power_values_neg = power_values[power_values<0]
											sorted_prob_neg = sorted_prob[power_values<0]
											while len(power_values_neg)>PDF_intervals*7:
												i = (sorted_prob_neg[1:]+sorted_prob_neg[:-1]).argmin()
												power_values_neg = np.concatenate((power_values_neg[:i],[np.mean(power_values_neg[i:i+2])],power_values_neg[i+2:]))
												sorted_prob_neg = np.concatenate((sorted_prob_neg[:i],[np.sum(sorted_prob_neg[i:i+2])],sorted_prob_neg[i+2:]))
											while len(power_values_neg)>PDF_intervals:
												i = (power_values_neg[:-1]/power_values_neg[1:]).argmin()
												power_values_neg = np.concatenate((power_values_neg[:i],[np.mean(power_values_neg[i:i+2])],power_values_neg[i+2:]))
												# sorted_prob_neg = np.concatenate((sorted_prob_neg[:i],[np.sum(sorted_prob_neg[i:i+2])],sorted_prob_neg[i+2:]))
										power_usefull = power[non_zero_prob_select]
										if power_values.min()>=0:
											power_values = np.unique(np.concatenate(([power_usefull.min()],power_values_pos,[power_values.max()])))
										elif power_values.max()<=0:
											power_values = np.unique(np.concatenate(([power_values.min()],power_values_neg,[power_usefull.max()])))
										else:
											power_values = np.unique(np.concatenate(([power_usefull.min()],power_values_neg,[0],power_values_pos,[power_usefull.max()])))

									# power_values = [*power_values[:-1][power_values[1:]/power_values[:-1]>1.05],power_values[-1]]
									# log_power = np.log10(power)
									# low_point = np.max(log_power)-d_power*PDF_intervals
									# log_power_values = np.array([-np.inf,*np.linspace(low_point,np.max(log_power),num=PDF_intervals)])
									if False:
										power_usefull = power[non_zero_prob_select]
										temp = power_usefull[power_usefull>0]
										if len(temp)==0:
											power_values = np.array([0,np.min(power_usefull),*power_values,np.max(power_usefull)])
										else:
											power_values = np.array([0,np.min(power_usefull),np.min(power_usefull[power_usefull>0]),*power_values,np.max(power_usefull)])
										power_values = np.unique(power_values)
									prob_power = []
									actual_values_power = []
									for index in range(len(power_values)-1):
										if index==0:
											select = np.logical_and(power_usefull>=power_values[index],power_usefull<=power_values[index+1])
										else:
											select = np.logical_and(power_usefull>power_values[index],power_usefull<=power_values[index+1])
										if np.sum(select)==0:
											prob_power.append(0)
											actual_values_power.append(np.mean([power_values[index],power_values[index+1]]))
										else:
											temp1 = power_usefull[select]
											temp2 = likelihood_probs_times_volume_usefull[select]
											prob_power.append(np.sum(temp2))
											temp = np.sum(temp2*temp1)/np.sum(prob_power[-1])
											if np.isnan(temp):
												print('why?')
												actual_values_power.append(np.nanmean(temp1))
											else:
												actual_values_power.append(temp)
									prob_power = np.array(prob_power)/np.sum(prob_power)
									actual_values_power = np.array(actual_values_power)
									time_lapsed = tm.time()-start_time
									# print('PDF built in %.3gsec' %(time_lapsed))
									if require_dict==False:
										return power_values,prob_power,actual_values_power
									else:
										return dict([('intervals',power_values),('prob',prob_power),('actual_values',actual_values_power)])


								power_balance_data_dict = dict([])
								power_balance_data_dict['PDF_matrix_shape'] = PDF_matrix_shape
								intervals_power_rad_excit,prob_power_rad_excit,actual_values_power_rad_excit = build_log_PDF(power_rad_excit)
								power_balance_data_dict['power_rad_excit'] = dict([('intervals',intervals_power_rad_excit),('prob',prob_power_rad_excit),('actual_values',actual_values_power_rad_excit)])
								intervals_power_rad_rec_bremm,prob_power_rad_rec_bremm,actual_values_power_rad_rec_bremm = build_log_PDF(power_rad_rec_bremm)
								power_balance_data_dict['power_rad_rec_bremm'] = dict([('intervals',intervals_power_rad_rec_bremm),('prob',prob_power_rad_rec_bremm),('actual_values',actual_values_power_rad_rec_bremm)])
								intervals_power_rad_mol,prob_power_rad_mol,actual_values_power_rad_mol = build_log_PDF(power_rad_mol)
								power_balance_data_dict['power_rad_mol'] = dict([('intervals',intervals_power_rad_mol),('prob',prob_power_rad_mol),('actual_values',actual_values_power_rad_mol)])
								intervals_power_via_ionisation,prob_power_via_ionisation,actual_values_power_via_ionisation = build_log_PDF(power_via_ionisation)
								power_balance_data_dict['power_via_ionisation'] = dict([('intervals',intervals_power_via_ionisation),('prob',prob_power_via_ionisation),('actual_values',actual_values_power_via_ionisation)])
								intervals_power_via_recombination,prob_power_via_recombination,actual_values_power_via_recombination = build_log_PDF(power_via_recombination)
								power_balance_data_dict['power_via_recombination'] = dict([('intervals',intervals_power_via_recombination),('prob',prob_power_via_recombination),('actual_values',actual_values_power_via_recombination)])
								intervals_tot_rad_power,prob_tot_rad_power,actual_values_tot_rad_power = build_log_PDF(tot_rad_power)
								power_balance_data_dict['tot_rad_power'] = dict([('intervals',intervals_tot_rad_power),('prob',prob_tot_rad_power),('actual_values',actual_values_tot_rad_power)])
								intervals_power_rad_Hm,prob_power_rad_Hm,actual_values_power_rad_Hm = build_log_PDF(power_rad_Hm)
								power_balance_data_dict['power_rad_Hm'] = dict([('intervals',intervals_power_rad_Hm),('prob',prob_power_rad_Hm),('actual_values',actual_values_power_rad_Hm)])
								intervals_power_rad_Hm_H2p,prob_power_rad_Hm_H2p,actual_values_power_rad_Hm_H2p = build_log_PDF(power_rad_Hm_H2p)
								power_balance_data_dict['power_rad_Hm_H2p'] = dict([('intervals',intervals_power_rad_Hm_H2p),('prob',prob_power_rad_Hm_H2p),('actual_values',actual_values_power_rad_Hm_H2p)])
								intervals_power_rad_Hm_Hp,prob_power_rad_Hm_Hp,actual_values_power_rad_Hm_Hp = build_log_PDF(power_rad_Hm_Hp)
								power_balance_data_dict['power_rad_Hm_Hp'] = dict([('intervals',intervals_power_rad_Hm_Hp),('prob',prob_power_rad_Hm_Hp),('actual_values',actual_values_power_rad_Hm_Hp)])
								intervals_power_rad_H2,prob_power_rad_H2,actual_values_power_rad_H2 = build_log_PDF(power_rad_H2)
								power_balance_data_dict['power_rad_H2'] = dict([('intervals',intervals_power_rad_H2),('prob',prob_power_rad_H2),('actual_values',actual_values_power_rad_H2)])
								intervals_power_rad_H2p,prob_power_rad_H2p,actual_values_power_rad_H2p = build_log_PDF(power_rad_H2p)
								power_balance_data_dict['power_rad_H2p'] = dict([('intervals',intervals_power_rad_H2p),('prob',prob_power_rad_H2p),('actual_values',actual_values_power_rad_H2p)])
								intervals_power_heating_rec,prob_power_heating_rec,actual_values_power_heating_rec = build_log_PDF(power_heating_rec)
								power_balance_data_dict['power_heating_rec'] = dict([('intervals',intervals_power_heating_rec),('prob',prob_power_heating_rec),('actual_values',actual_values_power_heating_rec)])
								intervals_power_rec_neutral,prob_power_rec_neutral,actual_values_power_rec_neutral = build_log_PDF(power_rec_neutral)
								power_balance_data_dict['power_rec_neutral'] = dict([('intervals',intervals_power_rec_neutral),('prob',prob_power_rec_neutral),('actual_values',actual_values_power_rec_neutral)])
								intervals_power_via_brem,prob_power_via_brem,actual_values_power_via_brem = build_log_PDF(power_via_brem)
								power_balance_data_dict['power_via_brem'] = dict([('intervals',intervals_power_via_brem),('prob',prob_power_via_brem),('actual_values',actual_values_power_via_brem)])
								intervals_total_removed_power,prob_total_removed_power,actual_values_total_removed_power = build_log_PDF(total_removed_power)
								power_balance_data_dict['total_removed_power'] = dict([('intervals',intervals_total_removed_power),('prob',prob_total_removed_power),('actual_values',actual_values_total_removed_power)])
								if not(my_time_pos in sample_time_step):
									if not(my_r_pos in sample_radious):
										del power_rad_excit,power_rad_rec_bremm,power_rad_mol,power_via_ionisation,power_via_recombination,power_rad_Hm,power_rad_H2,power_rad_H2p,tot_rad_power,power_via_brem,power_heating_rec,power_rec_neutral,total_removed_power
								intervals_nH_ne_excited_states,prob_nH_ne_excited_states,actual_values_nH_ne_excited_states = build_log_PDF(nH_ne_excited_states)
								power_balance_data_dict['nH_ne_excited_states'] = dict([('intervals',intervals_nH_ne_excited_states),('prob',prob_nH_ne_excited_states),('actual_values',actual_values_nH_ne_excited_states)])
								intervals_ne_values,prob_ne_values,actual_values_ne_values = build_log_PDF(all_ne_values,treshold_ratio=1.05)
								power_balance_data_dict['ne_values'] = dict([('intervals',intervals_ne_values),('prob',prob_ne_values),('actual_values',actual_values_ne_values)])
								intervals_Te_values,prob_Te_values,actual_values_Te_values = build_log_PDF(all_Te_values,treshold_ratio=1.05)
								power_balance_data_dict['Te_values'] = dict([('intervals',intervals_Te_values),('prob',prob_Te_values),('actual_values',actual_values_Te_values)])
								# all_nHm_ne_values = np.array([np.transpose(np.array([record_nHm_ne_values.reshape((H2_steps,Hm_steps,TS_ne_steps,TS_Te_steps)).tolist()]*H2p_steps), (2,1,0,3,4)).tolist()]*H_steps,dtype=np.float32)
								intervals_nHm_ne_values,prob_nHm_ne_values,actual_values_nHm_ne_values = build_log_PDF(all_nHm_ne_values)
								power_balance_data_dict['nHm_ne_values'] = dict([('intervals',intervals_nHm_ne_values),('prob',prob_nHm_ne_values),('actual_values',actual_values_nHm_ne_values)])
								power_balance_data_dict['nHm_nH2_values'] = build_log_PDF(all_nHm_nH2_values,require_dict=True)
								all_nHm_values = all_nHm_ne_values*ne_values
								intervals_nHm_values,prob_nHm_values,actual_values_nHm_values = build_log_PDF(all_nHm_values)
								power_balance_data_dict['nHm_values'] = dict([('intervals',intervals_nHm_values),('prob',prob_nHm_values),('actual_values',actual_values_nHm_values)])
								del all_nHm_ne_values,all_nHm_values
								# all_nH2p_ne_values = np.array([[record_nH2p_ne_values.reshape((H2_steps,H2p_steps,TS_ne_steps,TS_Te_steps)).tolist()]*Hm_steps]*H_steps,dtype=np.float32)
								intervals_nH2p_ne_values,prob_nH2p_ne_values,actual_values_nH2p_ne_values = build_log_PDF(all_nH2p_ne_values)
								power_balance_data_dict['nH2p_ne_values'] = dict([('intervals',intervals_nH2p_ne_values),('prob',prob_nH2p_ne_values),('actual_values',actual_values_nH2p_ne_values)])
								power_balance_data_dict['nH2p_nH2_values'] = build_log_PDF(all_nH2p_nH2_values,require_dict=True)
								all_nH2p_values = all_nH2p_ne_values*ne_values
								intervals_nH2p_values,prob_nH2p_values,actual_values_nH2p_values = build_log_PDF(all_nH2p_values)
								power_balance_data_dict['nH2p_values'] = dict([('intervals',intervals_nH2p_values),('prob',prob_nH2p_values),('actual_values',actual_values_nH2p_values)])
								del all_nH2p_ne_values,all_nH2p_values
								intervals_nH2_ne_values,prob_nH2_ne_values,actual_values_nH2_ne_values = build_log_PDF(all_nH2_ne_values)
								power_balance_data_dict['nH2_ne_values'] = dict([('intervals',intervals_nH2_ne_values),('prob',prob_nH2_ne_values),('actual_values',actual_values_nH2_ne_values)])
								all_nH2_values = all_nH2_ne_values*ne_values
								intervals_nH2_values,prob_nH2_values,actual_values_nH2_values = build_log_PDF(all_nH2_values)
								power_balance_data_dict['nH2_values'] = dict([('intervals',intervals_nH2_values),('prob',prob_nH2_values),('actual_values',actual_values_nH2_values)])
								del all_nH2_values
								intervals_nH_ne_values,prob_nH_ne_values,actual_values_nH_ne_values = build_log_PDF(all_nH_ne_values)
								power_balance_data_dict['nH_ne_values'] = dict([('intervals',intervals_nH_ne_values),('prob',prob_nH_ne_values),('actual_values',actual_values_nH_ne_values)])
								all_nH_values = all_nH_ne_values*ne_values
								intervals_nH_values,prob_nH_values,actual_values_nH_values = build_log_PDF(all_nH_values)
								power_balance_data_dict['nH_values'] = dict([('intervals',intervals_nH_values),('prob',prob_nH_values),('actual_values',actual_values_nH_values)])
								del all_nH_values
								if include_particles_limitation:
									intervals_net_e_destruction,prob_net_e_destruction,actual_values_net_e_destruction = build_log_PDF(all_net_e_destruction)
									power_balance_data_dict['net_e_destruction'] = dict([('intervals',intervals_net_e_destruction),('prob',prob_net_e_destruction),('actual_values',actual_values_net_e_destruction)])
									intervals_net_Hp_destruction,prob_net_Hp_destruction,actual_values_net_Hp_destruction = build_log_PDF(all_net_Hp_destruction)
									power_balance_data_dict['net_Hp_destruction'] = dict([('intervals',intervals_net_Hp_destruction),('prob',prob_net_Hp_destruction),('actual_values',actual_values_net_Hp_destruction)])
									intervals_net_Hm_destruction,prob_net_Hm_destruction,actual_values_net_Hm_destruction = build_log_PDF(all_net_Hm_destruction)
									power_balance_data_dict['net_Hm_destruction'] = dict([('intervals',intervals_net_Hm_destruction),('prob',prob_net_Hm_destruction),('actual_values',actual_values_net_Hm_destruction)])
									intervals_net_H2p_destruction,prob_net_H2p_destruction,actual_values_net_H2p_destruction = build_log_PDF(all_net_H2p_destruction)
									power_balance_data_dict['net_H2p_destruction'] = dict([('intervals',intervals_net_H2p_destruction),('prob',prob_net_H2p_destruction),('actual_values',actual_values_net_H2p_destruction)])
								else:
									# intervals_net_e_destruction,prob_net_e_destruction,actual_values_net_e_destruction = [[0],[0],[0]]
									# intervals_net_Hp_destruction,prob_net_Hp_destruction,actual_values_net_Hp_destruction = [[0],[0],[0]]
									# intervals_net_Hm_destruction,prob_net_Hm_destruction,actual_values_net_Hm_destruction = [[0],[0],[0]]
									# intervals_net_H2p_destruction,prob_net_H2p_destruction,actual_values_net_H2p_destruction = [[0],[0],[0]]
									power_balance_data_dict['net_e_destruction'] = dict([('intervals',[0]),('prob',[0]),('actual_values',[0])])
									power_balance_data_dict['net_Hp_destruction'] = dict([('intervals',[0]),('prob',[0]),('actual_values',[0])])
									power_balance_data_dict['net_Hm_destruction'] = dict([('intervals',[0]),('prob',[0]),('actual_values',[0])])
									power_balance_data_dict['net_H2p_destruction'] = dict([('intervals',[0]),('prob',[0]),('actual_values',[0])])
								# intervals_H_destruction_RR,prob_H_destruction_RR,actual_values_H_destruction_RR = build_log_PDF(all_H_destruction_RR)
								power_balance_data_dict['H_destruction_RR'] = build_log_PDF(all_H_destruction_RR,require_dict=True)
								power_balance_data_dict['H_destruction_RR2'] = build_log_PDF(all_H_destruction_RR2,require_dict=True)
								power_balance_data_dict['net_H_destruction_RR'] = build_log_PDF(all_net_H_destruction_RR,require_dict=True)
								del all_H_destruction_RR,all_H_destruction_RR2,all_net_H_destruction_RR
								# intervals_H_creation_RR,prob_H_creation_RR,actual_values_H_creation_RR = build_log_PDF(np.float64(all_H_creation_RR)*1e20)
								power_balance_data_dict['H_creation_RR'] = build_log_PDF(all_H_creation_RR,require_dict=True)
								del all_H_creation_RR
								# intervals_H2_destruction_RR,prob_H2_destruction_RR,actual_values_H2_destruction_RR = build_log_PDF(all_H2_destruction_RR)
								power_balance_data_dict['H2_destruction_RR'] = build_log_PDF(all_H2_destruction_RR,require_dict=True)
								power_balance_data_dict['H2_destruction_RR2'] = build_log_PDF(all_H2_destruction_RR2,require_dict=True)
								power_balance_data_dict['net_H2_destruction_RR'] = build_log_PDF(all_net_H2_destruction_RR,require_dict=True)
								del all_H2_destruction_RR,all_H2_destruction_RR2,all_net_H2_destruction_RR
								power_balance_data_dict['H2_creation_RR'] = build_log_PDF(all_H2_creation_RR,require_dict=True)
								del all_H2_creation_RR
								intervals_local_CX,prob_local_CX,actual_values_local_CX = build_log_PDF(local_CX)
								power_balance_data_dict['local_CX'] = dict([('intervals',intervals_local_CX),('prob',prob_local_CX),('actual_values',actual_values_local_CX)])
								# intervals_eff_CX_RR,prob_eff_CX_RR,actual_values_eff_CX_RR = build_log_PDF(eff_CX_RR)
								power_balance_data_dict['eff_CX_RR'] = build_log_PDF(eff_CX_RR,require_dict=True)
								del eff_CX_RR
								# intervals_CX_term_1_1,prob_CX_term_1_1,actual_values_CX_term_1_1 = build_log_PDF(CX_term_1_1)
								power_balance_data_dict['CX_term_1_1'] = build_log_PDF(CX_term_1_1,require_dict=True)
								del CX_term_1_1
								# intervals_CX_term_1_2,prob_CX_term_1_2,actual_values_CX_term_1_2 = build_log_PDF(CX_term_1_2)
								power_balance_data_dict['CX_term_1_2'] = build_log_PDF(CX_term_1_2,require_dict=True)
								del CX_term_1_2
								# intervals_CX_term_1_3,prob_CX_term_1_3,actual_values_CX_term_1_3 = build_log_PDF(CX_term_1_3)
								power_balance_data_dict['CX_term_1_3'] = build_log_PDF(CX_term_1_3,require_dict=True)
								del CX_term_1_3
								# intervals_CX_term_1_4,prob_CX_term_1_4,actual_values_CX_term_1_4 = build_log_PDF(CX_term_1_4)
								power_balance_data_dict['CX_term_1_4'] = build_log_PDF(CX_term_1_4,require_dict=True)
								del CX_term_1_4
								# intervals_CX_term_1_5,prob_CX_term_1_5,actual_values_CX_term_1_5 = build_log_PDF(CX_term_1_5)
								power_balance_data_dict['CX_term_1_5'] = build_log_PDF(CX_term_1_5,require_dict=True)
								del CX_term_1_5
								# intervals_CX_term_1_6,prob_CX_term_1_6,actual_values_CX_term_1_6 = build_log_PDF(CX_term_1_6)
								power_balance_data_dict['CX_term_1_6'] = build_log_PDF(CX_term_1_6,require_dict=True)
								del CX_term_1_6
								# intervals_CX_term_1_7,prob_CX_term_1_7,actual_values_CX_term_1_7 = build_log_PDF(CX_term_1_7)
								power_balance_data_dict['CX_term_1_7'] = build_log_PDF(CX_term_1_7,require_dict=True)
								del CX_term_1_7
								power_balance_data_dict['CX_term_1_8'] = build_log_PDF(CX_term_1_8,require_dict=True)
								del CX_term_1_8
								power_balance_data_dict['CX_term_1_9'] = build_log_PDF(CX_term_1_9,require_dict=True)
								del CX_term_1_9
								# power_balance_data_dict['CX_term_1_10'] = build_log_PDF(CX_term_1_10,require_dict=True)
								# del CX_term_1_10
								power_balance_data_dict['CX_term_1_11'] = build_log_PDF(CX_term_1_11,require_dict=True)
								del CX_term_1_11
								# intervals_total_removed_power_visible,prob_total_removed_power_visible,actual_values_total_removed_power_visible = build_log_PDF(total_removed_power_visible)
								power_balance_data_dict['total_removed_power_visible'] = build_log_PDF(total_removed_power_visible,require_dict=True)
								del total_removed_power_visible
								# intervals_power_rad_atomic_visible,prob_power_rad_atomic_visible,actual_values_power_rad_atomic_visible = build_log_PDF(power_rad_atomic_visible)
								power_balance_data_dict['power_rad_atomic_visible'] = build_log_PDF(power_rad_atomic_visible,require_dict=True)
								del power_rad_atomic_visible
								# intervals_power_rad_mol_visible,prob_power_rad_mol_visible,actual_values_power_rad_mol_visible = build_log_PDF(power_rad_mol_visible)
								power_balance_data_dict['power_rad_mol_visible'] = build_log_PDF(power_rad_mol_visible,require_dict=True)
								del power_rad_mol_visible

								# power_balance_data = [intervals_power_rad_excit,prob_power_rad_excit,actual_values_power_rad_excit, intervals_power_rad_rec_bremm,prob_power_rad_rec_bremm,actual_values_power_rad_rec_bremm, intervals_power_rad_mol,prob_power_rad_mol,actual_values_power_rad_mol, intervals_power_via_ionisation,prob_power_via_ionisation,actual_values_power_via_ionisation, intervals_power_via_recombination,prob_power_via_recombination,actual_values_power_via_recombination, intervals_tot_rad_power,prob_tot_rad_power,actual_values_tot_rad_power,intervals_power_rad_Hm,prob_power_rad_Hm,actual_values_power_rad_Hm,intervals_power_rad_H2,prob_power_rad_H2,actual_values_power_rad_H2,intervals_power_rad_H2p,prob_power_rad_H2p,actual_values_power_rad_H2p,intervals_power_heating_rec,prob_power_heating_rec,actual_values_power_heating_rec,intervals_power_rec_neutral,prob_power_rec_neutral,actual_values_power_rec_neutral,intervals_power_via_brem,prob_power_via_brem,actual_values_power_via_brem,intervals_total_removed_power,prob_total_removed_power,actual_values_total_removed_power]
								# real_prob_ne = np.sum(likelihood_probs_times_volume,axis=(0,1,2,3,5))	# H, Hm, H2, H2p, ne, Te
								# real_prob_Te = np.sum(likelihood_probs_times_volume,axis=(0,1,2,3,4))	# H, Hm, H2, H2p, ne, Te
								# real_prob_nH_ne = np.sum(likelihood_probs_times_volume,axis=(1,2,3,4,5))	# H, Hm, H2, H2p, ne, Te
								# real_prob_nH2_ne = np.sum(likelihood_probs_times_volume,axis=(0,1,3,4,5))	# H, Hm, H2, H2p, ne, Te
							else:
								power_balance_data_dict = 0
								# intervals_power_rad_excit,prob_power_rad_excit,actual_values_power_rad_excit = [[0],[0],[0]]
								# intervals_power_rad_rec_bremm,prob_power_rad_rec_bremm,actual_values_power_rad_rec_bremm = [[0],[0],[0]]
								# intervals_power_rad_mol,prob_power_rad_mol,actual_values_power_rad_mol = [[0],[0],[0]]
								# intervals_power_via_ionisation,prob_power_via_ionisation,actual_values_power_via_ionisation = [[0],[0],[0]]
								# intervals_power_via_recombination,prob_power_via_recombination,actual_values_power_via_recombination = [[0],[0],[0]]
								# intervals_tot_rad_power,prob_tot_rad_power,actual_values_tot_rad_power = [[0],[0],[0]]
								# intervals_power_rad_Hm,prob_power_rad_Hm,actual_values_power_rad_Hm = [[0],[0],[0]]
								# intervals_power_rad_H2,prob_power_rad_H2,actual_values_power_rad_H2 = [[0],[0],[0]]
								# intervals_power_rad_H2p,prob_power_rad_H2p,actual_values_power_rad_H2p = [[0],[0],[0]]
								# intervals_power_heating_rec,prob_power_heating_rec,actual_values_power_heating_rec = [[0],[0],[0]]
								# intervals_power_rec_neutral,prob_power_rec_neutral,actual_values_power_rec_neutral = [[0],[0],[0]]
								# intervals_power_via_brem,prob_power_via_brem,actual_values_power_via_brem = [[0],[0],[0]]
								# intervals_total_removed_power,prob_total_removed_power,actual_values_total_removed_power = [[0],[0],[0]]
								# intervals_nH_ne_excited_states,prob_nH_ne_excited_states,actual_values_nH_ne_excited_states = [[0],[0],[0]]
								# intervals_ne_values,prob_ne_values,actual_values_ne_values = [[0],[0],[0]]
								# intervals_Te_values,prob_Te_values,actual_values_Te_values = [[0],[0],[0]]
								# intervals_nHm_ne_values,prob_nHm_ne_values,actual_values_nHm_ne_values = [[0],[0],[0]]
								# intervals_nH2p_ne_values,prob_nH2p_ne_values,actual_values_nH2p_ne_values = [[0],[0],[0]]
								# intervals_nH2_ne_values,prob_nH2_ne_values,actual_values_nH2_ne_values = [[0],[0],[0]]
								# intervals_nH_ne_values,prob_nH_ne_values,actual_values_nH_ne_values = [[0],[0],[0]]
								# intervals_net_e_destruction,prob_net_e_destruction,actual_values_net_e_destruction = [[0],[0],[0]]
								# intervals_net_Hp_destruction,prob_net_Hp_destruction,actual_values_net_Hp_destruction = [[0],[0],[0]]
								# intervals_net_Hm_destruction,prob_net_Hm_destruction,actual_values_net_Hm_destruction = [[0],[0],[0]]
								# intervals_net_H2p_destruction,prob_net_H2p_destruction,actual_values_net_H2p_destruction = [[0],[0],[0]]
								# intervals_H_destruction_RR,prob_H_destruction_RR,actual_values_H_destruction_RR = [[0],[0],[0]]
								# intervals_H2_destruction_RR,prob_H2_destruction_RR,actual_values_H2_destruction_RR = [[0],[0],[0]]
								# intervals_local_CX,prob_local_CX,actual_values_local_CX = [[0],[0],[0]]
								# intervals_eff_CX_RR,prob_eff_CX_RR,actual_values_eff_CX_RR = [[0],[0],[0]]
								# intervals_CX_term_1_1,prob_CX_term_1_1,actual_values_CX_term_1_1 = [[0],[0],[0]]
								# intervals_CX_term_1_2,prob_CX_term_1_2,actual_values_CX_term_1_2 = [[0],[0],[0]]
								# intervals_CX_term_1_3,prob_CX_term_1_3,actual_values_CX_term_1_3 = [[0],[0],[0]]
								# intervals_CX_term_1_4,prob_CX_term_1_4,actual_values_CX_term_1_4 = [[0],[0],[0]]
								# intervals_CX_term_1_5,prob_CX_term_1_5,actual_values_CX_term_1_5 = [[0],[0],[0]]
								# intervals_CX_term_1_6,prob_CX_term_1_6,actual_values_CX_term_1_6 = [[0],[0],[0]]
								# intervals_CX_term_1_7,prob_CX_term_1_7,actual_values_CX_term_1_7 = [[0],[0],[0]]
								# intervals_H_creation_RR,prob_H_creation_RR,actual_values_H_creation_RR = [[0],[0],[0]]
								# intervals_total_removed_power_visible,prob_total_removed_power_visible,actual_values_total_removed_power_visible = [[0],[0],[0]]
								# intervals_power_rad_atomic_visible,prob_power_rad_atomic_visible,actual_values_power_rad_atomic_visible = [[0],[0],[0]]
								# intervals_power_rad_mol_visible,prob_power_rad_mol_visible,actual_values_power_rad_mol_visible = [[0],[0],[0]]


							# power_balance_data = [intervals_power_rad_excit,prob_power_rad_excit,actual_values_power_rad_excit, intervals_power_rad_rec_bremm,prob_power_rad_rec_bremm,actual_values_power_rad_rec_bremm, intervals_power_rad_mol,prob_power_rad_mol,actual_values_power_rad_mol, intervals_power_via_ionisation,prob_power_via_ionisation,actual_values_power_via_ionisation, intervals_power_via_recombination,prob_power_via_recombination,actual_values_power_via_recombination, intervals_tot_rad_power,prob_tot_rad_power,actual_values_tot_rad_power,intervals_power_rad_Hm,prob_power_rad_Hm,actual_values_power_rad_Hm,intervals_power_rad_H2,prob_power_rad_H2,actual_values_power_rad_H2,intervals_power_rad_H2p,prob_power_rad_H2p,actual_values_power_rad_H2p,intervals_power_heating_rec,prob_power_heating_rec,actual_values_power_heating_rec,intervals_power_rec_neutral,prob_power_rec_neutral,actual_values_power_rec_neutral,intervals_power_via_brem,prob_power_via_brem,actual_values_power_via_brem,intervals_total_removed_power,prob_total_removed_power,actual_values_total_removed_power,intervals_nH_ne_excited_states,prob_nH_ne_excited_states,actual_values_nH_ne_excited_states,intervals_ne_values,prob_ne_values,actual_values_ne_values,intervals_Te_values,prob_Te_values,actual_values_Te_values,intervals_nHm_ne_values,prob_nHm_ne_values,actual_values_nHm_ne_values,intervals_nH2p_ne_values,prob_nH2p_ne_values,actual_values_nH2p_ne_values,intervals_nH2_ne_values,prob_nH2_ne_values,actual_values_nH2_ne_values,intervals_nH_ne_values,prob_nH_ne_values,actual_values_nH_ne_values,intervals_net_e_destruction,prob_net_e_destruction,actual_values_net_e_destruction,intervals_net_Hp_destruction,prob_net_Hp_destruction,actual_values_net_Hp_destruction,intervals_net_Hm_destruction,prob_net_Hm_destruction,actual_values_net_Hm_destruction,intervals_net_H2p_destruction,prob_net_H2p_destruction,actual_values_net_H2p_destruction,intervals_local_CX,prob_local_CX,actual_values_local_CX,intervals_H_destruction_RR,prob_H_destruction_RR,actual_values_H_destruction_RR,intervals_eff_CX_RR,prob_eff_CX_RR,actual_values_eff_CX_RR,intervals_H2_destruction_RR,prob_H2_destruction_RR,actual_values_H2_destruction_RR,intervals_CX_term_1_1,prob_CX_term_1_1,actual_values_CX_term_1_1,intervals_CX_term_1_2,prob_CX_term_1_2,actual_values_CX_term_1_2,intervals_CX_term_1_3,prob_CX_term_1_3,actual_values_CX_term_1_3,intervals_CX_term_1_4,prob_CX_term_1_4,actual_values_CX_term_1_4,intervals_CX_term_1_5,prob_CX_term_1_5,actual_values_CX_term_1_5,intervals_CX_term_1_6,prob_CX_term_1_6,actual_values_CX_term_1_6,intervals_CX_term_1_7,prob_CX_term_1_7,actual_values_CX_term_1_7,intervals_H_creation_RR,prob_H_creation_RR,actual_values_H_creation_RR,intervals_total_removed_power_visible,prob_total_removed_power_visible,actual_values_total_removed_power_visible,intervals_power_rad_atomic_visible,prob_power_rad_atomic_visible,actual_values_power_rad_atomic_visible,intervals_power_rad_mol_visible,prob_power_rad_mol_visible,actual_values_power_rad_mol_visible]



							if my_time_pos in sample_time_step:
								if my_r_pos in sample_radious:
									if to_print:
										# here I find the plame that contains the most of the high probability points
										nH2p_ne_values_full,nHm_ne_values_full = np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])
										temp = np.exp(likelihood_log_probs[most_likely_nH_ne_index,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index])
										temp=temp.flatten()
										fit1 = curve_fit(line,nH2p_ne_values_full.flatten()[temp>0],nHm_ne_values_full.flatten()[temp>0],p0=[1,1],sigma=(1/(temp[temp>0])).flatten())[0]

										nH_ne_values_full,nHm_ne_values_full,nH2p_ne_values_full = np.meshgrid(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],indexing='ij')
										temp = np.exp(likelihood_log_probs[:,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index])
										temp=temp.flatten()
										def find_correlation(params):
											calc = line_3D(nH_ne_values_full,nH2p_ne_values_full,*params)
											diff = (np.sum((((nHm_ne_values_full-calc).flatten()*temp)**2)[temp>0])/np.sum(temp[temp>0]**2))/(np.sum((((nHm_ne_values_full-np.mean(nHm_ne_values_full)).flatten()*temp)**2)[temp>0])/np.sum(temp[temp>0]**2))
											return diff
										x_scale = np.abs(np.array([1e-3,fit1[0],fit1[1]]))
										x_scale[x_scale==0]=1e-6
										sol_line = least_squares(find_correlation,[0,fit1[0],fit1[1]],max_nfev=10000,x_scale=x_scale)
										R2_sol_line = 1-find_correlation(sol_line.x)

										def find_correlation(params):
											calc = parabola_3D(nH_ne_values_full,nH2p_ne_values_full,*params)
											diff = (np.sum((((nHm_ne_values_full-calc).flatten()*temp)**2)[temp>0])/np.sum(temp[temp>0]**2))/(np.sum((((nHm_ne_values_full-np.mean(nHm_ne_values_full)).flatten()*temp)**2)[temp>0])/np.sum(temp[temp>0]**2))
											return diff
										x_scale = np.abs(np.array([1e-3,sol_line.x[0],1e-3,sol_line.x[1],sol_line.x[2]]))
										x_scale[x_scale==0]=1e-6
										try:
											sol_parab = least_squares(find_correlation,[0,sol_line.x[0],0,sol_line.x[1],sol_line.x[2]],max_nfev=10000,x_scale=x_scale)
											R2_sol_parab = 1-find_correlation(sol_parab.x)
										except:
											R2_sol_parab = 0

										nH2p_ne_values_full,nHm_ne_values_full = np.meshgrid(np.arange(H2p_steps),np.arange(Hm_steps))
										temp = np.exp(marginalised_likelihood_log_probs[most_likely_marginalised_nH_ne_index,:,:])
										temp=temp.flatten()
										fit1 = curve_fit(line,nH2p_ne_values_full.flatten()[temp>0],nHm_ne_values_full.flatten()[temp>0],p0=[1,1],sigma=(1/(temp[temp>0])).flatten())[0]

										nH_ne_values_full,nHm_ne_values_full,nH2p_ne_values_full = np.meshgrid(np.arange(H_steps),np.arange(Hm_steps),np.arange(H2p_steps),indexing='ij')
										temp = np.exp(marginalised_likelihood_log_probs[:,:,:])
										temp=temp.flatten()
										def find_correlation(params):
											calc = line_3D(nH_ne_values_full,nH2p_ne_values_full,*params)
											diff = (np.sum((((nHm_ne_values_full-calc).flatten()*temp)**2)[temp>0])/np.sum(temp[temp>0]**2))/(np.sum((((nHm_ne_values_full-np.mean(nHm_ne_values_full)).flatten()*temp)**2)[temp>0])/np.sum(temp[temp>0]**2))
											return diff
										x_scale = np.abs(np.array([1e-3,fit1[0],fit1[1]]))
										x_scale[x_scale==0]=1e-6
										marginalised_sol_line = least_squares(find_correlation,[0,fit1[0],fit1[1]],max_nfev=10000,x_scale=x_scale)
										R2_marginalised_sol_line = 1-find_correlation(marginalised_sol_line.x)

										def find_correlation(params):
											calc = parabola_3D(nH_ne_values_full,nH2p_ne_values_full,*params)
											diff = (np.sum((((nHm_ne_values_full-calc).flatten()*temp)**2)[temp>0])/np.sum(temp[temp>0]**2))/(np.sum((((nHm_ne_values_full-np.mean(nHm_ne_values_full)).flatten()*temp)**2)[temp>0])/np.sum(temp[temp>0]**2))
											return diff
										x_scale = np.abs(np.array([1e-3,sol_line.x[0],1e-3,sol_line.x[1],sol_line.x[2]]))
										x_scale[x_scale==0]=1e-6
										marginalised_sol_parab = least_squares(find_correlation,[0,marginalised_sol_line.x[0],0,marginalised_sol_line.x[1],marginalised_sol_line.x[2]],max_nfev=10000,x_scale=x_scale)
										R2_marginalised_sol_parab = 1-find_correlation(marginalised_sol_parab.x)


							if my_time_pos in sample_time_step:
								if my_r_pos in sample_radious:
									if to_print:
										print('worker '+str(current_process())+' '+str(domain_index)+' first scan best_fit_nH_ne_value %.3g,best_fit_nHm_ne_value %.3g,best_fit_nH2_ne_value %.3g,best_fit_nH2p_ne_value %.3g' %(best_fit_nH_ne_value,best_fit_nHm_ne_value,best_fit_nH2_ne_value,best_fit_nH2p_ne_value))
										# tm.sleep(np.random.random()*10)
										print('First scan\nmost_likely_nH_ne_index %.3g,most_likely_nHm_ne_value %.3g,most_likely_nH2_ne_value %.3g,most_likely_nH2p_ne_value %.3g,most_likely_Te_value %.3g,most_likely_ne_value %.3g' %(most_likely_nH_ne_value,most_likely_nHm_ne_value,most_likely_nH2_ne_value,most_likely_nH2p_ne_value,most_likely_Te_value,most_likely_ne_value))
										fig, ax = plt.subplots( 4,2,figsize=(20, 45), squeeze=False)
										fig.suptitle('first scan\nmost_likely_nH_ne_value %.3g,most_likely_nHm_ne_value %.3g,most_likely_nH2_ne_value %.3g,most_likely_nH2p_ne_value %.3g' %(most_likely_nH_ne_value*most_likely_ne_value,most_likely_nHm_ne_value*most_likely_ne_value,most_likely_nH2_ne_value*most_likely_ne_value,most_likely_nH2p_ne_value*most_likely_ne_value) +'\nlines '+str(n_list_all)+'\nlocation [time, r]'+ ' [%.3g ms, ' % time_crop[my_time_pos] + ' %.3g mm]' % (1000*r_crop[my_r_pos]) +' , [TS Te, TS ne] '+ ' [%.3g(ML %.3g) eV, ' %(merge_Te_prof_multipulse_interp_crop_limited_restrict,most_likely_Te_value) + '%.3g(ML %.3g) #10^20/m^3]' %(merge_ne_prof_multipulse_interp_crop_limited_restrict,most_likely_ne_value*1e-20) +'\nfit [nH/ne, nH+/ne, nH-/ne, nH2/ne, nH2+/ne, nH3+/ne] , '+ '[%.3g, ' % most_likely_nH_ne_value + '%.3g, ' %(1 - most_likely_nH2p_ne_value + most_likely_nHm_ne_value)+ '%.3g, ' % most_likely_nHm_ne_value+ '%.3g, ' % most_likely_nH2_ne_value+ '%.3g, ' % most_likely_nH2p_ne_value+ '%.3g]' % 0+'\nBest index: [nH/ne,nH-/ne,nH2/ne,nH2+/ne,ne,Te] [%.3g,%.3g,%.3g,%.3g,%.3g,%.3g]' %(most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index)+'\nMarginalised best index: [nH/ne,nH-/ne,nH2+/ne] [%.3g,%.3g,%.3g]' %(most_likely_marginalised_nH_ne_index,most_likely_marginalised_nHm_nH2_index,most_likely_marginalised_nH2p_nH2_index)+'\nPDF_matrix_shape = '+str(PDF_matrix_shape))
										plot_index = 0
										im = ax[plot_index,0].plot(range(TS_Te_steps)-most_likely_Te_index,np.exp(likelihood_log_probs[most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,:]),label='Te');
										im = ax[plot_index,0].plot(range(TS_ne_steps)-most_likely_ne_index,np.exp(likelihood_log_probs[most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,:,most_likely_Te_index]),label='ne');
										im = ax[plot_index,0].plot(range(H2p_steps)-most_likely_nH2p_nH2_index,np.exp(likelihood_log_probs[most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]),label='nH2+/ne');
										im = ax[plot_index,0].plot(range(H2_steps)-most_likely_nH2_ne_index,np.exp(likelihood_log_probs[most_likely_nH_ne_index,most_likely_nHm_nH2_index,:,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index]),label='nH2/ne');
										im = ax[plot_index,0].plot(range(Hm_steps)-most_likely_nHm_nH2_index,np.exp(likelihood_log_probs[most_likely_nH_ne_index,:,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index]),label='nH-/ne');
										im = ax[plot_index,0].plot(range(H_steps)-most_likely_nH_ne_index,np.exp(likelihood_log_probs[:,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index]),label='nH/ne');
										ax[plot_index,0].set_title('Likelihood around the best')
										ax[plot_index,0].set_ylabel('normalised likelihood')
										ax[plot_index,0].set_xlabel('range used for search')
										ax[plot_index,0].legend(loc='best')
										ax[plot_index,0].grid()
										# ax[plot_index].set_xscale('log')
										# ax[plot_index,0].set_yscale('log')
										# ax[plot_index,0].set_ylim(bottom=np.exp(max_likelihood_log_prob)*1e-2,top=np.exp(max_likelihood_log_prob)*1.1)

										im = ax[plot_index,1].plot(range(H2p_steps)-most_likely_marginalised_nH2p_nH2_index,np.exp(marginalised_likelihood_log_probs[most_likely_marginalised_nH_ne_index,most_likely_marginalised_nHm_nH2_index,:]),label='nH2+/ne');
										im = ax[plot_index,1].plot(range(Hm_steps)-most_likely_marginalised_nHm_nH2_index,np.exp(marginalised_likelihood_log_probs[most_likely_marginalised_nH_ne_index,:,most_likely_marginalised_nH2p_nH2_index]),label='nH-/ne');
										im = ax[plot_index,1].plot(range(H_steps)-most_likely_marginalised_nH_ne_index,np.exp(marginalised_likelihood_log_probs[:,most_likely_marginalised_nHm_nH2_index,most_likely_marginalised_nH2p_nH2_index]),label='nH/ne');
										ax[plot_index,1].set_title('Marginalised likelihood around the best')
										ax[plot_index,1].set_ylabel('normalised likelihood')
										ax[plot_index,1].set_xlabel('range used for search')
										ax[plot_index,1].legend(loc='best')
										ax[plot_index,1].grid()
										# ax[plot_index].set_xscale('log')
										# ax[plot_index,1].set_yscale('log')
										# ax[plot_index,1].set_ylim(bottom=np.exp(np.max(marginalised_likelihood_log_probs))*1e-2,top=np.exp(np.max(marginalised_likelihood_log_probs))*1.1)

										plot_index += 1
										im = ax[plot_index,0].contourf(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(likelihood_log_probs[most_likely_nH_ne_index,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,0])
										im = ax[plot_index,0].contour(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(likelihood_log_probs[most_likely_nH_ne_index,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]),levels=10);
										# im = ax[plot_index,0].plot(most_likely_marginalised_nH2p_ne_value,most_likely_marginalised_nHm_ne_value,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,0].plot(most_likely_nH2p_ne_value,most_likely_nHm_ne_value,color='k',marker='$X$',markersize=30);
										# im = ax[plot_index,0].plot(first_most_likely_marginalised_nH2p_ne_value,first_most_likely_marginalised_nHm_ne_value,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,0].plot(first_most_likely_nH2p_ne_value,first_most_likely_nHm_ne_value,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,0].plot([next_nH2p_ne_value_up,next_nH2p_ne_value_up,next_nH2p_ne_value_down,next_nH2p_ne_value_down,next_nH2p_ne_value_up],[next_nHm_ne_value_down,next_nHm_ne_value_up,next_nHm_ne_value_up,next_nHm_ne_value_down,next_nHm_ne_value_down],'k--');
										im = ax[plot_index,0].plot(np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[0],np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[1],'k,')
										nH2p_ne_values_full,nHm_ne_values_full = np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])
										temp = np.exp(likelihood_log_probs[most_likely_nH_ne_index,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index])
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH2p_ne_values_full.flatten(),nHm_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nHm_ne_values_full.flatten(),nH2p_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH2p_ne_values_full = nH2p_ne_values_full[temp>np.max(temp)*start]
										nHm_ne_values_full = nHm_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH2p_ne_values_full,nHm_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH2p_ne_values_full)))**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.sort(nH2p_ne_values_full),np.polyval(fit1,np.sort(nH2p_ne_values_full)),'b')
											fit2 = np.polyfit(nHm_ne_values_full,nH2p_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nHm_ne_values_full)))**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.polyval(fit2,np.sort(nHm_ne_values_full)),np.sort(nHm_ne_values_full),'b')
											fit_average = [(fit1[0]+1/fit2[0])/2,(fit1[1]-fit2[1]/fit2[0])/2]
											reverse_fit_average = [1/fit_average[0],-fit_average[1]/fit_average[0]]
											im = ax[plot_index,0].plot(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],np.polyval(fit_average,record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index]),'k--')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H\nM=marginalised best, X=unmarginalised best, --=samples\nblue=fit H-/H2+ (%.3g,%.3g)R2 %.3g\nfit H2+/H- (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H\nM=marginalised best, X=unmarginalised best, --=samples')
										ax[plot_index,0].set_ylabel('nH-/ne')
										ax[plot_index,0].set_xlabel('nH2+/ne')
										ax[plot_index,0].set_xscale('log')
										ax[plot_index,0].set_yscale('log')

										im = ax[plot_index,1].contourf(range(H2p_steps),range(Hm_steps), np.exp(marginalised_likelihood_log_probs[most_likely_marginalised_nH_ne_index]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,1])
										im = ax[plot_index,1].contour(range(H2p_steps),range(Hm_steps), np.exp(marginalised_likelihood_log_probs[most_likely_marginalised_nH_ne_index,:,:]),levels=10);
										im = ax[plot_index,1].plot(most_likely_marginalised_nH2p_nH2_index,most_likely_marginalised_nHm_nH2_index,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,1].plot(most_likely_nH2p_nH2_index,most_likely_nHm_nH2_index,color='k',marker='$X$',markersize=30);
										im = ax[plot_index,1].plot(first_most_likely_marginalised_nH2p_nH2_index,first_most_likely_marginalised_nHm_nH2_index,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,1].plot(first_most_likely_nH2p_nH2_index,first_most_likely_nHm_nH2_index,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,1].plot([next_nH2p_ne_value_up,next_nH2p_ne_value_up,next_nH2p_ne_value_down,next_nH2p_ne_value_down,next_nH2p_ne_value_up],[next_nHm_ne_value_down,next_nHm_ne_value_up,next_nHm_ne_value_up,next_nHm_ne_value_down,next_nHm_ne_value_down],'k--');
										im = ax[plot_index,1].plot(np.meshgrid(range(H2p_steps), range(Hm_steps))[0],np.meshgrid(range(H2p_steps), range(Hm_steps))[1],'k,')
										nH2p_ne_values_full,nHm_ne_values_full = np.meshgrid(range(H2p_steps),range(Hm_steps))
										temp = np.exp(marginalised_likelihood_log_probs[most_likely_marginalised_nH_ne_index])
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH2p_ne_values_full.flatten(),nHm_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nHm_ne_values_full.flatten(),nH2p_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH2p_ne_values_full = nH2p_ne_values_full[temp>np.max(temp)*start]
										nHm_ne_values_full = nHm_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH2p_ne_values_full,nHm_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH2p_ne_values_full)))**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.sort(nH2p_ne_values_full),np.polyval(fit1,np.sort(nH2p_ne_values_full)),'b')
											fit2 = np.polyfit(nHm_ne_values_full,nH2p_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nHm_ne_values_full)))**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.polyval(fit2,np.sort(nHm_ne_values_full)),np.sort(nHm_ne_values_full),'b')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H\nblue=fit H-/H2+ (%.3g,%.3g)R2 %.3g\nfit H2+/H- (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H')
										ax[plot_index,1].set_ylabel('nH-/nH2 index')
										ax[plot_index,1].set_xlabel('nH2+/nH2 index')
										# ax[plot_index,1].set_xscale('log')
										# ax[plot_index,1].set_yscale('log')

										plot_index += 1
										im = ax[plot_index,0].contourf(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(likelihood_log_probs[:,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,0])
										im = ax[plot_index,0].contour(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(likelihood_log_probs[:,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]),levels=10);
										# im = ax[plot_index,0].plot(most_likely_marginalised_nH2p_ne_value,most_likely_marginalised_nH_ne_value,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,0].plot(most_likely_nH2p_ne_value,most_likely_nH_ne_value,color='k',marker='$X$',markersize=30);
										# im = ax[plot_index,0].plot(first_most_likely_marginalised_nH2p_ne_value,first_most_likely_marginalised_nH_ne_value,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,0].plot(first_most_likely_nH2p_ne_value,first_most_likely_nH_ne_value,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,0].plot([next_nH2p_ne_value_up,next_nH2p_ne_value_up,next_nH2p_ne_value_down,next_nH2p_ne_value_down,next_nH2p_ne_value_up],[next_nH_ne_value_down,next_nH_ne_value_up,next_nH_ne_value_up,next_nH_ne_value_down,next_nH_ne_value_down],'k--');
										im = ax[plot_index,0].plot(np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[0],np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[1],'k,')
										nH2p_ne_values_full,nH_ne_values_full = np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])
										temp = np.exp(likelihood_log_probs[:,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index])
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH2p_ne_values_full.flatten(),nH_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit1,nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nH_ne_values_full.flatten(),nH2p_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH2p_ne_values_full = nH2p_ne_values_full[temp>np.max(temp)*start]
										nH_ne_values_full = nH_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH2p_ne_values_full,nH_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit1,nH2p_ne_values_full)))**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.sort(nH2p_ne_values_full),np.polyval(fit1,np.sort(nH2p_ne_values_full)),'b')
											fit2 = np.polyfit(nH_ne_values_full,nH2p_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nH_ne_values_full)))**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.polyval(fit2,np.sort(nH_ne_values_full)),np.sort(nH_ne_values_full),'b')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H-\nM=marginalised best, X=unmarginalised best\nblue=fit H/H2+ (%.3g,%.3g)R2 %.3g\nfit H2+/H (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H-\nM=marginalised best, X=unmarginalised best')
										ax[plot_index,0].set_ylabel('nH/ne')
										ax[plot_index,0].set_xlabel('nH2+/ne')
										ax[plot_index,0].set_xscale('log')
										ax[plot_index,0].set_yscale('log')

										im = ax[plot_index,1].contourf(range(H2p_steps),range(H_steps), np.exp(marginalised_likelihood_log_probs[:,most_likely_marginalised_nHm_nH2_index,:]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,1])
										im = ax[plot_index,1].contour(range(H2p_steps),range(H_steps), np.exp(marginalised_likelihood_log_probs[:,most_likely_marginalised_nHm_nH2_index,:]),levels=10);
										im = ax[plot_index,1].plot(most_likely_marginalised_nH2p_nH2_index,most_likely_marginalised_nH_ne_index,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,1].plot(most_likely_nH2p_nH2_index,most_likely_nH_ne_index,color='k',marker='$X$',markersize=30);
										im = ax[plot_index,1].plot(first_most_likely_marginalised_nH2p_nH2_index,first_most_likely_marginalised_nH_ne_index,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,1].plot(first_most_likely_nH2p_nH2_index,first_most_likely_nH_ne_index,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,1].plot([next_nH2p_ne_value_up,next_nH2p_ne_value_up,next_nH2p_ne_value_down,next_nH2p_ne_value_down,next_nH2p_ne_value_up],[next_nH_ne_value_down,next_nH_ne_value_up,next_nH_ne_value_up,next_nH_ne_value_down,next_nH_ne_value_down],'k--');
										im = ax[plot_index,1].plot(np.meshgrid(range(H2p_steps), range(H_steps))[0],np.meshgrid(range(H2p_steps), range(H_steps))[1],'k,')
										nH2p_ne_values_full,nH_ne_values_full = np.meshgrid(range(H2p_steps),range(H_steps))
										temp = np.exp(marginalised_likelihood_log_probs[:,most_likely_marginalised_nHm_nH2_index,:])
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH2p_ne_values_full.flatten(),nH_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit1,nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nH_ne_values_full.flatten(),nH2p_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH2p_ne_values_full = nH2p_ne_values_full[temp>np.max(temp)*start]
										nH_ne_values_full = nH_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH2p_ne_values_full,nH_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit1,nH2p_ne_values_full)))**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.sort(nH2p_ne_values_full),np.polyval(fit1,np.sort(nH2p_ne_values_full)),'b')
											fit2 = np.polyfit(nH_ne_values_full,nH2p_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nH_ne_values_full)))**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.polyval(fit2,np.sort(nH_ne_values_full)),np.sort(nH_ne_values_full),'b')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H-\nblue=fit H/H2+ (%.3g,%.3g)R2 %.3g\nfit H2+/H (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H-')
										ax[plot_index,1].set_ylabel('nH/ne index')
										ax[plot_index,1].set_xlabel('nH2+/nH2 index')
										# ax[plot_index,1].set_xscale('log')
										# ax[plot_index,1].set_yscale('log')

										plot_index += 1
										im = ax[plot_index,0].contourf(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(likelihood_log_probs[:,:,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index].T),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,0])
										im = ax[plot_index,0].contour(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(likelihood_log_probs[:,:,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index].T),levels=10);
										# im = ax[plot_index,0].plot(most_likely_marginalised_nH_ne_value,most_likely_marginalised_nHm_ne_value,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,0].plot(most_likely_nH_ne_value,most_likely_nHm_ne_value,color='k',marker='$X$',markersize=30);
										# im = ax[plot_index,0].plot(first_most_likely_marginalised_nH_ne_value,first_most_likely_marginalised_nHm_ne_value,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,0].plot(first_most_likely_nH_ne_value,first_most_likely_nHm_ne_value,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,0].plot([next_nH_ne_value_up,next_nH_ne_value_up,next_nH_ne_value_down,next_nH_ne_value_down,next_nH_ne_value_up],[next_nHm_ne_value_down,next_nHm_ne_value_up,next_nHm_ne_value_up,next_nHm_ne_value_down,next_nHm_ne_value_down],'k--');
										im = ax[plot_index,0].plot(np.meshgrid(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[0],np.meshgrid(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[1],'k,')
										nH_ne_values_full,nHm_ne_values_full = np.meshgrid(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])
										temp = np.exp(likelihood_log_probs[:,:,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index].T)
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH_ne_values_full.flatten(),nHm_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nHm_ne_values_full.flatten(),nH_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit2,nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH_ne_values_full = nH_ne_values_full[temp>np.max(temp)*start]
										nHm_ne_values_full = nHm_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH_ne_values_full,nHm_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH_ne_values_full)))**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.sort(nH_ne_values_full),np.polyval(fit1,np.sort(nH_ne_values_full)),'b')
											fit2 = np.polyfit(nHm_ne_values_full,nH_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit2,nHm_ne_values_full)))**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.polyval(fit2,np.sort(nHm_ne_values_full)),np.sort(nHm_ne_values_full),'b')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H2+\nM=marginalised best, X=unmarginalised best\nblue=fit H-/H (%.3g,%.3g)R2 %.3g\nfit H/H- (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H2+\nM=marginalised best, X=unmarginalised best')
										ax[plot_index,0].set_ylabel('nH-/ne')
										ax[plot_index,0].set_xlabel('nH/ne')
										ax[plot_index,0].set_xscale('log')
										ax[plot_index,0].set_yscale('log')

										im = ax[plot_index,1].contourf(range(H_steps),range(Hm_steps), np.exp(marginalised_likelihood_log_probs[:,:,most_likely_marginalised_nH2p_nH2_index].T),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,1])
										im = ax[plot_index,1].contour(range(H_steps),range(Hm_steps), np.exp(marginalised_likelihood_log_probs[:,:,most_likely_marginalised_nH2p_nH2_index].T),levels=10);
										im = ax[plot_index,1].plot(most_likely_marginalised_nH_ne_index,most_likely_marginalised_nHm_nH2_index,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,1].plot(most_likely_nH_ne_index,most_likely_nHm_nH2_index,color='k',marker='$X$',markersize=30);
										im = ax[plot_index,1].plot(first_most_likely_marginalised_nH_ne_index,first_most_likely_marginalised_nHm_nH2_index,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,1].plot(first_most_likely_nH_ne_index,first_most_likely_nHm_nH2_index,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,1].plot([next_nH_ne_value_up,next_nH_ne_value_up,next_nH_ne_value_down,next_nH_ne_value_down,next_nH_ne_value_up],[next_nHm_ne_value_down,next_nHm_ne_value_up,next_nHm_ne_value_up,next_nHm_ne_value_down,next_nHm_ne_value_down],'k--');
										im = ax[plot_index,1].plot(np.meshgrid(range(H_steps), range(Hm_steps))[0],np.meshgrid(range(H_steps), range(Hm_steps))[1],'k,')
										nH_ne_values_full,nHm_ne_values_full = np.meshgrid(range(H_steps),range(Hm_steps))
										temp = np.exp(marginalised_likelihood_log_probs[:,:,most_likely_marginalised_nH2p_nH2_index].T)
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH_ne_values_full.flatten(),nHm_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nHm_ne_values_full.flatten(),nH_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit2,nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH_ne_values_full = nH_ne_values_full[temp>np.max(temp)*start]
										nHm_ne_values_full = nHm_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH_ne_values_full,nHm_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH_ne_values_full)))**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.sort(nH_ne_values_full),np.polyval(fit1,np.sort(nH_ne_values_full)),'b')
											fit2 = np.polyfit(nHm_ne_values_full,nH_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit2,nHm_ne_values_full)))**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.polyval(fit2,np.sort(nHm_ne_values_full)),np.sort(nHm_ne_values_full),'b')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H2+\nblue=fit H-/H (%.3g,%.3g)R2 %.3g\nfit H/H- (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H2+')
										ax[plot_index,1].set_ylabel('nH-/nH2 index')
										ax[plot_index,1].set_xlabel('nH/ne index')
										# ax[plot_index,1].set_xscale('log')
										# ax[plot_index,1].set_yscale('log')
										save_done = 0
										save_index=1
										try:	# All this trie are because of a known issues when saving lots of figure inside of multiprocessor
											plt.savefig(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps', bbox_inches='tight')
										except Exception as e:
											print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps save try number '+str(1)+' failed. Reason %s' % e)
											while save_done==0 and save_index<100:
												try:
													plt.savefig(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps', bbox_inches='tight')
													print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps save try number '+str(save_index+1)+' successfull')
													save_done=1
												except Exception as e:
													tm.sleep(np.random.random()**2)
													# print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(save_index+1)+' failed. Reason %s' % e)
													save_index+=1
										plt.close('all')

										print('First scan +1')
										fig, ax = plt.subplots( 3,2,figsize=(20, 35), squeeze=False)
										fig.suptitle('first scan + 1\nmost_likely_nH_ne_value %.3g,most_likely_nHm_ne_value %.3g,most_likely_nH2_ne_value %.3g,most_likely_nH2p_ne_value %.3g' %(most_likely_nH_ne_value*most_likely_ne_value,most_likely_nHm_ne_value*most_likely_ne_value,most_likely_nH2_ne_value*most_likely_ne_value,most_likely_nH2p_ne_value*most_likely_ne_value) +'\nlines '+str(n_list_all)+'\nlocation [time, r]'+ ' [%.3g ms, ' % time_crop[my_time_pos] + ' %.3g mm]' % (1000*r_crop[my_r_pos]) +' , [TS Te, TS ne] '+ ' [%.3g(ML %.3g) eV, ' %(merge_Te_prof_multipulse_interp_crop_limited_restrict,most_likely_Te_value) + '%.3g(ML %.3g) #10^20/m^3]' %(merge_ne_prof_multipulse_interp_crop_limited_restrict,most_likely_ne_value*1e-20) +'\nfit [nH/ne, nH+/ne, nH-/ne, nH2/ne, nH2+/ne, nH3+/ne] , '+ '[%.3g, ' % most_likely_nH_ne_value + '%.3g, ' %(1 - most_likely_nH2p_ne_value + most_likely_nHm_ne_value)+ '%.3g, ' % most_likely_nHm_ne_value+ '%.3g, ' % most_likely_nH2_ne_value+ '%.3g, ' % most_likely_nH2p_ne_value+ '%.3g]' % 0+'\nBest index: [nH/ne,nH-/ne,nH2/ne,nH2+/ne,ne,Te] [%.3g,%.3g,%.3g,%.3g,%.3g,%.3g]' %(most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index)+'\nMarginalised best index: [nH/ne,nH-/ne,nH2+/ne] [%.3g,%.3g,%.3g]' %(most_likely_marginalised_nH_ne_index,most_likely_marginalised_nHm_nH2_index,most_likely_marginalised_nH2p_nH2_index)+'\nPDF_matrix_shape = '+str(PDF_matrix_shape) + '\n EMISSIVITY PENALTY')
										plot_index = 0
										im = ax[plot_index,0].contourf(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(calculated_emission_log_probs_expanded[most_likely_nH_ne_index,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,0])
										im = ax[plot_index,0].contour(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(calculated_emission_log_probs_expanded[most_likely_nH_ne_index,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]),levels=10);
										# im = ax[plot_index,0].plot(most_likely_marginalised_nH2p_ne_value,most_likely_marginalised_nHm_ne_value,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,0].plot(most_likely_nH2p_ne_value,most_likely_nHm_ne_value,color='k',marker='$X$',markersize=30);
										# im = ax[plot_index,0].plot(first_most_likely_marginalised_nH2p_ne_value,first_most_likely_marginalised_nHm_ne_value,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,0].plot(first_most_likely_nH2p_ne_value,first_most_likely_nHm_ne_value,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,0].plot([next_nH2p_ne_value_up,next_nH2p_ne_value_up,next_nH2p_ne_value_down,next_nH2p_ne_value_down,next_nH2p_ne_value_up],[next_nHm_ne_value_down,next_nHm_ne_value_up,next_nHm_ne_value_up,next_nHm_ne_value_down,next_nHm_ne_value_down],'k--');
										im = ax[plot_index,0].plot(np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[0],np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[1],'k,')
										nH2p_ne_values_full,nHm_ne_values_full = np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])
										temp = np.exp(calculated_emission_log_probs_expanded[most_likely_nH_ne_index,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index])
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH2p_ne_values_full.flatten(),nHm_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nHm_ne_values_full.flatten(),nH2p_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH2p_ne_values_full = nH2p_ne_values_full[temp>np.max(temp)*start]
										nHm_ne_values_full = nHm_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH2p_ne_values_full,nHm_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH2p_ne_values_full)))**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.sort(nH2p_ne_values_full),np.polyval(fit1,np.sort(nH2p_ne_values_full)),'b')
											fit2 = np.polyfit(nHm_ne_values_full,nH2p_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nHm_ne_values_full)))**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.polyval(fit2,np.sort(nHm_ne_values_full)),np.sort(nHm_ne_values_full),'b')
											fit_average = [(fit1[0]+1/fit2[0])/2,(fit1[1]-fit2[1]/fit2[0])/2]
											reverse_fit_average = [1/fit_average[0],-fit_average[1]/fit_average[0]]
											im = ax[plot_index,0].plot(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],np.polyval(fit_average,record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index]),'k--')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H\nM=marginalised best, X=unmarginalised best, --=samples\nblue=fit H-/H2+ (%.3g,%.3g)R2 %.3g\nfit H2+/H- (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H\nM=marginalised best, X=unmarginalised best, --=samples')
										ax[plot_index,0].set_ylabel('nH-/ne')
										ax[plot_index,0].set_xlabel('nH2+/ne')
										ax[plot_index,0].set_xscale('log')
										ax[plot_index,0].set_yscale('log')

										im = ax[plot_index,1].contourf(range(H2p_steps),range(Hm_steps), np.exp(marginalised_calculated_emission_log_probs_expanded[most_likely_marginalised_nH_ne_index]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,1])
										im = ax[plot_index,1].contour(range(H2p_steps),range(Hm_steps), np.exp(marginalised_calculated_emission_log_probs_expanded[most_likely_marginalised_nH_ne_index,:,:]),levels=10);
										im = ax[plot_index,1].plot(most_likely_marginalised_nH2p_nH2_index,most_likely_marginalised_nHm_nH2_index,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,1].plot(most_likely_nH2p_nH2_index,most_likely_nHm_nH2_index,color='k',marker='$X$',markersize=30);
										im = ax[plot_index,1].plot(first_most_likely_marginalised_nH2p_nH2_index,first_most_likely_marginalised_nHm_nH2_index,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,1].plot(first_most_likely_nH2p_nH2_index,first_most_likely_nHm_nH2_index,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,1].plot([next_nH2p_ne_value_up,next_nH2p_ne_value_up,next_nH2p_ne_value_down,next_nH2p_ne_value_down,next_nH2p_ne_value_up],[next_nHm_ne_value_down,next_nHm_ne_value_up,next_nHm_ne_value_up,next_nHm_ne_value_down,next_nHm_ne_value_down],'k--');
										im = ax[plot_index,1].plot(np.meshgrid(range(H2p_steps), range(Hm_steps))[0],np.meshgrid(range(H2p_steps), range(Hm_steps))[1],'k,')
										nH2p_ne_values_full,nHm_ne_values_full = np.meshgrid(range(H2p_steps),range(Hm_steps))
										temp = np.exp(marginalised_calculated_emission_log_probs_expanded[most_likely_marginalised_nH_ne_index])
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH2p_ne_values_full.flatten(),nHm_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nHm_ne_values_full.flatten(),nH2p_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH2p_ne_values_full = nH2p_ne_values_full[temp>np.max(temp)*start]
										nHm_ne_values_full = nHm_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH2p_ne_values_full,nHm_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH2p_ne_values_full)))**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.sort(nH2p_ne_values_full),np.polyval(fit1,np.sort(nH2p_ne_values_full)),'b')
											fit2 = np.polyfit(nHm_ne_values_full,nH2p_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nHm_ne_values_full)))**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.polyval(fit2,np.sort(nHm_ne_values_full)),np.sort(nHm_ne_values_full),'b')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H\nblue=fit H-/H2+ (%.3g,%.3g)R2 %.3g\nfit H2+/H- (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H')
										ax[plot_index,1].set_ylabel('nH-/nH2 index')
										ax[plot_index,1].set_xlabel('nH2+/nH2 index')
										# ax[plot_index,1].set_xscale('log')
										# ax[plot_index,1].set_yscale('log')

										plot_index += 1
										im = ax[plot_index,0].contourf(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(calculated_emission_log_probs_expanded[:,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,0])
										im = ax[plot_index,0].contour(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(calculated_emission_log_probs_expanded[:,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]),levels=10);
										# im = ax[plot_index,0].plot(most_likely_marginalised_nH2p_ne_value,most_likely_marginalised_nH_ne_value,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,0].plot(most_likely_nH2p_ne_value,most_likely_nH_ne_value,color='k',marker='$X$',markersize=30);
										# im = ax[plot_index,0].plot(first_most_likely_marginalised_nH2p_ne_value,first_most_likely_marginalised_nH_ne_value,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,0].plot(first_most_likely_nH2p_ne_value,first_most_likely_nH_ne_value,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,0].plot([next_nH2p_ne_value_up,next_nH2p_ne_value_up,next_nH2p_ne_value_down,next_nH2p_ne_value_down,next_nH2p_ne_value_up],[next_nH_ne_value_down,next_nH_ne_value_up,next_nH_ne_value_up,next_nH_ne_value_down,next_nH_ne_value_down],'k--');
										im = ax[plot_index,0].plot(np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[0],np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[1],'k,')
										nH2p_ne_values_full,nH_ne_values_full = np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])
										temp = np.exp(calculated_emission_log_probs_expanded[:,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index])
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH2p_ne_values_full.flatten(),nH_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit1,nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nH_ne_values_full.flatten(),nH2p_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH2p_ne_values_full = nH2p_ne_values_full[temp>np.max(temp)*start]
										nH_ne_values_full = nH_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH2p_ne_values_full,nH_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit1,nH2p_ne_values_full)))**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.sort(nH2p_ne_values_full),np.polyval(fit1,np.sort(nH2p_ne_values_full)),'b')
											fit2 = np.polyfit(nH_ne_values_full,nH2p_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nH_ne_values_full)))**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.polyval(fit2,np.sort(nH_ne_values_full)),np.sort(nH_ne_values_full),'b')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H-\nM=marginalised best, X=unmarginalised best\nblue=fit H/H2+ (%.3g,%.3g)R2 %.3g\nfit H2+/H (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H-\nM=marginalised best, X=unmarginalised best')
										ax[plot_index,0].set_ylabel('nH/ne')
										ax[plot_index,0].set_xlabel('nH2+/ne')
										ax[plot_index,0].set_xscale('log')
										ax[plot_index,0].set_yscale('log')

										im = ax[plot_index,1].contourf(range(H2p_steps),range(H_steps), np.exp(marginalised_calculated_emission_log_probs_expanded[:,most_likely_marginalised_nHm_nH2_index,:]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,1])
										im = ax[plot_index,1].contour(range(H2p_steps),range(H_steps), np.exp(marginalised_calculated_emission_log_probs_expanded[:,most_likely_marginalised_nHm_nH2_index,:]),levels=10);
										im = ax[plot_index,1].plot(most_likely_marginalised_nH2p_nH2_index,most_likely_marginalised_nH_ne_index,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,1].plot(most_likely_nH2p_nH2_index,most_likely_nH_ne_index,color='k',marker='$X$',markersize=30);
										im = ax[plot_index,1].plot(first_most_likely_marginalised_nH2p_nH2_index,first_most_likely_marginalised_nH_ne_index,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,1].plot(first_most_likely_nH2p_nH2_index,first_most_likely_nH_ne_index,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,1].plot([next_nH2p_ne_value_up,next_nH2p_ne_value_up,next_nH2p_ne_value_down,next_nH2p_ne_value_down,next_nH2p_ne_value_up],[next_nH_ne_value_down,next_nH_ne_value_up,next_nH_ne_value_up,next_nH_ne_value_down,next_nH_ne_value_down],'k--');
										im = ax[plot_index,1].plot(np.meshgrid(range(H2p_steps), range(H_steps))[0],np.meshgrid(range(H2p_steps), range(H_steps))[1],'k,')
										nH2p_ne_values_full,nH_ne_values_full = np.meshgrid(range(H2p_steps),range(H_steps))
										temp = np.exp(marginalised_calculated_emission_log_probs_expanded[:,most_likely_marginalised_nHm_nH2_index,:])
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH2p_ne_values_full.flatten(),nH_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit1,nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nH_ne_values_full.flatten(),nH2p_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH2p_ne_values_full = nH2p_ne_values_full[temp>np.max(temp)*start]
										nH_ne_values_full = nH_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH2p_ne_values_full,nH_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit1,nH2p_ne_values_full)))**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.sort(nH2p_ne_values_full),np.polyval(fit1,np.sort(nH2p_ne_values_full)),'b')
											fit2 = np.polyfit(nH_ne_values_full,nH2p_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nH_ne_values_full)))**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.polyval(fit2,np.sort(nH_ne_values_full)),np.sort(nH_ne_values_full),'b')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H-\nblue=fit H/H2+ (%.3g,%.3g)R2 %.3g\nfit H2+/H (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H-')
										ax[plot_index,1].set_ylabel('nH/ne index')
										ax[plot_index,1].set_xlabel('nH2+/nH2 index')
										# ax[plot_index,1].set_xscale('log')
										# ax[plot_index,1].set_yscale('log')

										plot_index += 1
										im = ax[plot_index,0].contourf(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(calculated_emission_log_probs_expanded[:,:,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index].T),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,0])
										im = ax[plot_index,0].contour(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(calculated_emission_log_probs_expanded[:,:,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index].T),levels=10);
										# im = ax[plot_index,0].plot(most_likely_marginalised_nH_ne_value,most_likely_marginalised_nHm_ne_value,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,0].plot(most_likely_nH_ne_value,most_likely_nHm_ne_value,color='k',marker='$X$',markersize=30);
										# im = ax[plot_index,0].plot(first_most_likely_marginalised_nH_ne_value,first_most_likely_marginalised_nHm_ne_value,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,0].plot(first_most_likely_nH_ne_value,first_most_likely_nHm_ne_value,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,0].plot([next_nH_ne_value_up,next_nH_ne_value_up,next_nH_ne_value_down,next_nH_ne_value_down,next_nH_ne_value_up],[next_nHm_ne_value_down,next_nHm_ne_value_up,next_nHm_ne_value_up,next_nHm_ne_value_down,next_nHm_ne_value_down],'k--');
										im = ax[plot_index,0].plot(np.meshgrid(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[0],np.meshgrid(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[1],'k,')
										nH_ne_values_full,nHm_ne_values_full = np.meshgrid(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])
										temp = np.exp(calculated_emission_log_probs_expanded[:,:,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index].T)
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH_ne_values_full.flatten(),nHm_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nHm_ne_values_full.flatten(),nH_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit2,nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH_ne_values_full = nH_ne_values_full[temp>np.max(temp)*start]
										nHm_ne_values_full = nHm_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH_ne_values_full,nHm_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH_ne_values_full)))**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.sort(nH_ne_values_full),np.polyval(fit1,np.sort(nH_ne_values_full)),'b')
											fit2 = np.polyfit(nHm_ne_values_full,nH_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit2,nHm_ne_values_full)))**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.polyval(fit2,np.sort(nHm_ne_values_full)),np.sort(nHm_ne_values_full),'b')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H2+\nM=marginalised best, X=unmarginalised best\nblue=fit H-/H (%.3g,%.3g)R2 %.3g\nfit H/H- (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H2+\nM=marginalised best, X=unmarginalised best')
										ax[plot_index,0].set_ylabel('nH-/ne')
										ax[plot_index,0].set_xlabel('nH/ne')
										ax[plot_index,0].set_xscale('log')
										ax[plot_index,0].set_yscale('log')

										im = ax[plot_index,1].contourf(range(H_steps),range(Hm_steps), np.exp(marginalised_calculated_emission_log_probs_expanded[:,:,most_likely_marginalised_nH2p_nH2_index].T),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,1])
										im = ax[plot_index,1].contour(range(H_steps),range(Hm_steps), np.exp(marginalised_calculated_emission_log_probs_expanded[:,:,most_likely_marginalised_nH2p_nH2_index].T),levels=10);
										im = ax[plot_index,1].plot(most_likely_marginalised_nH_ne_index,most_likely_marginalised_nHm_nH2_index,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,1].plot(most_likely_nH_ne_index,most_likely_nHm_nH2_index,color='k',marker='$X$',markersize=30);
										im = ax[plot_index,1].plot(first_most_likely_marginalised_nH_ne_index,first_most_likely_marginalised_nHm_nH2_index,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,1].plot(first_most_likely_nH_ne_index,first_most_likely_nHm_nH2_index,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,1].plot([next_nH_ne_value_up,next_nH_ne_value_up,next_nH_ne_value_down,next_nH_ne_value_down,next_nH_ne_value_up],[next_nHm_ne_value_down,next_nHm_ne_value_up,next_nHm_ne_value_up,next_nHm_ne_value_down,next_nHm_ne_value_down],'k--');
										im = ax[plot_index,1].plot(np.meshgrid(range(H_steps), range(Hm_steps))[0],np.meshgrid(range(H_steps), range(Hm_steps))[1],'k,')
										nH_ne_values_full,nHm_ne_values_full = np.meshgrid(range(H_steps),range(Hm_steps))
										temp = np.exp(marginalised_calculated_emission_log_probs_expanded[:,:,most_likely_marginalised_nH2p_nH2_index].T)
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH_ne_values_full.flatten(),nHm_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nHm_ne_values_full.flatten(),nH_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit2,nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH_ne_values_full = nH_ne_values_full[temp>np.max(temp)*start]
										nHm_ne_values_full = nHm_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH_ne_values_full,nHm_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH_ne_values_full)))**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.sort(nH_ne_values_full),np.polyval(fit1,np.sort(nH_ne_values_full)),'b')
											fit2 = np.polyfit(nHm_ne_values_full,nH_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit2,nHm_ne_values_full)))**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.polyval(fit2,np.sort(nHm_ne_values_full)),np.sort(nHm_ne_values_full),'b')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H2+\nblue=fit H-/H (%.3g,%.3g)R2 %.3g\nfit H/H- (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first_1.eps'+'fits failed')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H2+')
										ax[plot_index,1].set_ylabel('nH-/nH2 index')
										ax[plot_index,1].set_xlabel('nH/ne index')
										# ax[plot_index,1].set_xscale('log')
										# ax[plot_index,1].set_yscale('log')
										save_done = 0
										save_index=1
										try:	# All this trie are because of a known issues when saving lots of figure inside of multiprocessor
											plt.savefig(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first_1.eps', bbox_inches='tight')
										except Exception as e:
											print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first_1.eps save try number '+str(1)+' failed. Reason %s' % e)
											while save_done==0 and save_index<100:
												try:
													plt.savefig(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first_1.eps', bbox_inches='tight')
													print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first_1.eps save try number '+str(save_index+1)+' successfull')
													save_done=1
												except Exception as e:
													tm.sleep(np.random.random()**2)
													# print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(save_index+1)+' failed. Reason %s' % e)
													save_index+=1
										plt.close('all')

										print('First scan +2')
										fig, ax = plt.subplots( 3,2,figsize=(20, 35), squeeze=False)
										fig.suptitle('first scan +2\nmost_likely_nH_ne_value %.3g,most_likely_nHm_ne_value %.3g,most_likely_nH2_ne_value %.3g,most_likely_nH2p_ne_value %.3g' %(most_likely_nH_ne_value*most_likely_ne_value,most_likely_nHm_ne_value*most_likely_ne_value,most_likely_nH2_ne_value*most_likely_ne_value,most_likely_nH2p_ne_value*most_likely_ne_value) +'\nlines '+str(n_list_all)+'\nlocation [time, r]'+ ' [%.3g ms, ' % time_crop[my_time_pos] + ' %.3g mm]' % (1000*r_crop[my_r_pos]) +' , [TS Te, TS ne] '+ ' [%.3g(ML %.3g) eV, ' %(merge_Te_prof_multipulse_interp_crop_limited_restrict,most_likely_Te_value) + '%.3g(ML %.3g) #10^20/m^3]' %(merge_ne_prof_multipulse_interp_crop_limited_restrict,most_likely_ne_value*1e-20) +'\nfit [nH/ne, nH+/ne, nH-/ne, nH2/ne, nH2+/ne, nH3+/ne] , '+ '[%.3g, ' % most_likely_nH_ne_value + '%.3g, ' %(1 - most_likely_nH2p_ne_value + most_likely_nHm_ne_value)+ '%.3g, ' % most_likely_nHm_ne_value+ '%.3g, ' % most_likely_nH2_ne_value+ '%.3g, ' % most_likely_nH2p_ne_value+ '%.3g]' % 0+'\nBest index: [nH/ne,nH-/ne,nH2/ne,nH2+/ne,ne,Te] [%.3g,%.3g,%.3g,%.3g,%.3g,%.3g]' %(most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index)+'\nMarginalised best index: [nH/ne,nH-/ne,nH2+/ne] [%.3g,%.3g,%.3g]' %(most_likely_marginalised_nH_ne_index,most_likely_marginalised_nHm_nH2_index,most_likely_marginalised_nH2p_nH2_index)+'\nPDF_matrix_shape = '+str(PDF_matrix_shape) + '\n POWER BALANCE PENALTY')
										plot_index = 0
										im = ax[plot_index,0].contourf(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(power_penalty[most_likely_nH_ne_index,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,0])
										im = ax[plot_index,0].contour(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(power_penalty[most_likely_nH_ne_index,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]),levels=10);
										# im = ax[plot_index,0].plot(most_likely_marginalised_nH2p_ne_value,most_likely_marginalised_nHm_ne_value,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,0].plot(most_likely_nH2p_ne_value,most_likely_nHm_ne_value,color='k',marker='$X$',markersize=30);
										# im = ax[plot_index,0].plot(first_most_likely_marginalised_nH2p_ne_value,first_most_likely_marginalised_nHm_ne_value,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,0].plot(first_most_likely_nH2p_ne_value,first_most_likely_nHm_ne_value,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,0].plot([next_nH2p_ne_value_up,next_nH2p_ne_value_up,next_nH2p_ne_value_down,next_nH2p_ne_value_down,next_nH2p_ne_value_up],[next_nHm_ne_value_down,next_nHm_ne_value_up,next_nHm_ne_value_up,next_nHm_ne_value_down,next_nHm_ne_value_down],'k--');
										im = ax[plot_index,0].plot(np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[0],np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[1],'k,')
										nH2p_ne_values_full,nHm_ne_values_full = np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])
										temp = np.exp(power_penalty[most_likely_nH_ne_index,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index])
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH2p_ne_values_full.flatten(),nHm_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nHm_ne_values_full.flatten(),nH2p_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH2p_ne_values_full = nH2p_ne_values_full[temp>np.max(temp)*start]
										nHm_ne_values_full = nHm_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH2p_ne_values_full,nHm_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH2p_ne_values_full)))**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.sort(nH2p_ne_values_full),np.polyval(fit1,np.sort(nH2p_ne_values_full)),'b')
											fit2 = np.polyfit(nHm_ne_values_full,nH2p_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nHm_ne_values_full)))**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.polyval(fit2,np.sort(nHm_ne_values_full)),np.sort(nHm_ne_values_full),'b')
											fit_average = [(fit1[0]+1/fit2[0])/2,(fit1[1]-fit2[1]/fit2[0])/2]
											reverse_fit_average = [1/fit_average[0],-fit_average[1]/fit_average[0]]
											im = ax[plot_index,0].plot(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],np.polyval(fit_average,record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index]),'k--')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H\nM=marginalised best, X=unmarginalised best, --=samples\nblue=fit H-/H2+ (%.3g,%.3g)R2 %.3g\nfit H2+/H- (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H\nM=marginalised best, X=unmarginalised best, --=samples')
										ax[plot_index,0].set_ylabel('nH-/ne')
										ax[plot_index,0].set_xlabel('nH2+/ne')
										ax[plot_index,0].set_xscale('log')
										ax[plot_index,0].set_yscale('log')

										im = ax[plot_index,1].contourf(range(H2p_steps),range(Hm_steps), np.exp(marginalised_power_penalty[most_likely_marginalised_nH_ne_index]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,1])
										im = ax[plot_index,1].contour(range(H2p_steps),range(Hm_steps), np.exp(marginalised_power_penalty[most_likely_marginalised_nH_ne_index,:,:]),levels=10);
										im = ax[plot_index,1].plot(most_likely_marginalised_nH2p_nH2_index,most_likely_marginalised_nHm_nH2_index,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,1].plot(most_likely_nH2p_nH2_index,most_likely_nHm_nH2_index,color='k',marker='$X$',markersize=30);
										im = ax[plot_index,1].plot(first_most_likely_marginalised_nH2p_nH2_index,first_most_likely_marginalised_nHm_nH2_index,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,1].plot(first_most_likely_nH2p_nH2_index,first_most_likely_nHm_nH2_index,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,1].plot([next_nH2p_ne_value_up,next_nH2p_ne_value_up,next_nH2p_ne_value_down,next_nH2p_ne_value_down,next_nH2p_ne_value_up],[next_nHm_ne_value_down,next_nHm_ne_value_up,next_nHm_ne_value_up,next_nHm_ne_value_down,next_nHm_ne_value_down],'k--');
										im = ax[plot_index,1].plot(np.meshgrid(range(H2p_steps), range(Hm_steps))[0],np.meshgrid(range(H2p_steps), range(Hm_steps))[1],'k,')
										nH2p_ne_values_full,nHm_ne_values_full = np.meshgrid(range(H2p_steps),range(Hm_steps))
										temp = np.exp(marginalised_power_penalty[most_likely_marginalised_nH_ne_index])
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH2p_ne_values_full.flatten(),nHm_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nHm_ne_values_full.flatten(),nH2p_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH2p_ne_values_full = nH2p_ne_values_full[temp>np.max(temp)*start]
										nHm_ne_values_full = nHm_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH2p_ne_values_full,nHm_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH2p_ne_values_full)))**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.sort(nH2p_ne_values_full),np.polyval(fit1,np.sort(nH2p_ne_values_full)),'b')
											fit2 = np.polyfit(nHm_ne_values_full,nH2p_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nHm_ne_values_full)))**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.polyval(fit2,np.sort(nHm_ne_values_full)),np.sort(nHm_ne_values_full),'b')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H\nblue=fit H-/H2+ (%.3g,%.3g)R2 %.3g\nfit H2+/H- (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H')
										ax[plot_index,1].set_ylabel('nH-/nH2 index')
										ax[plot_index,1].set_xlabel('nH2+/nH2 index')
										# ax[plot_index,1].set_xscale('log')
										# ax[plot_index,1].set_yscale('log')

										plot_index += 1
										im = ax[plot_index,0].contourf(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(power_penalty[:,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,0])
										im = ax[plot_index,0].contour(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(power_penalty[:,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]),levels=10);
										# im = ax[plot_index,0].plot(most_likely_marginalised_nH2p_ne_value,most_likely_marginalised_nH_ne_value,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,0].plot(most_likely_nH2p_ne_value,most_likely_nH_ne_value,color='k',marker='$X$',markersize=30);
										# im = ax[plot_index,0].plot(first_most_likely_marginalised_nH2p_ne_value,first_most_likely_marginalised_nH_ne_value,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,0].plot(first_most_likely_nH2p_ne_value,first_most_likely_nH_ne_value,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,0].plot([next_nH2p_ne_value_up,next_nH2p_ne_value_up,next_nH2p_ne_value_down,next_nH2p_ne_value_down,next_nH2p_ne_value_up],[next_nH_ne_value_down,next_nH_ne_value_up,next_nH_ne_value_up,next_nH_ne_value_down,next_nH_ne_value_down],'k--');
										im = ax[plot_index,0].plot(np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[0],np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[1],'k,')
										nH2p_ne_values_full,nH_ne_values_full = np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])
										temp = np.exp(power_penalty[:,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index])
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH2p_ne_values_full.flatten(),nH_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit1,nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nH_ne_values_full.flatten(),nH2p_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH2p_ne_values_full = nH2p_ne_values_full[temp>np.max(temp)*start]
										nH_ne_values_full = nH_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH2p_ne_values_full,nH_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit1,nH2p_ne_values_full)))**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.sort(nH2p_ne_values_full),np.polyval(fit1,np.sort(nH2p_ne_values_full)),'b')
											fit2 = np.polyfit(nH_ne_values_full,nH2p_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nH_ne_values_full)))**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.polyval(fit2,np.sort(nH_ne_values_full)),np.sort(nH_ne_values_full),'b')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H-\nM=marginalised best, X=unmarginalised best\nblue=fit H/H2+ (%.3g,%.3g)R2 %.3g\nfit H2+/H (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H-\nM=marginalised best, X=unmarginalised best')
										ax[plot_index,0].set_ylabel('nH/ne')
										ax[plot_index,0].set_xlabel('nH2+/ne')
										ax[plot_index,0].set_xscale('log')
										ax[plot_index,0].set_yscale('log')

										im = ax[plot_index,1].contourf(range(H2p_steps),range(H_steps), np.exp(marginalised_power_penalty[:,most_likely_marginalised_nHm_nH2_index,:]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,1])
										im = ax[plot_index,1].contour(range(H2p_steps),range(H_steps), np.exp(marginalised_power_penalty[:,most_likely_marginalised_nHm_nH2_index,:]),levels=10);
										im = ax[plot_index,1].plot(most_likely_marginalised_nH2p_nH2_index,most_likely_marginalised_nH_ne_index,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,1].plot(most_likely_nH2p_nH2_index,most_likely_nH_ne_index,color='k',marker='$X$',markersize=30);
										im = ax[plot_index,1].plot(first_most_likely_marginalised_nH2p_nH2_index,first_most_likely_marginalised_nH_ne_index,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,1].plot(first_most_likely_nH2p_nH2_index,first_most_likely_nH_ne_index,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,1].plot([next_nH2p_ne_value_up,next_nH2p_ne_value_up,next_nH2p_ne_value_down,next_nH2p_ne_value_down,next_nH2p_ne_value_up],[next_nH_ne_value_down,next_nH_ne_value_up,next_nH_ne_value_up,next_nH_ne_value_down,next_nH_ne_value_down],'k--');
										im = ax[plot_index,1].plot(np.meshgrid(range(H2p_steps), range(H_steps))[0],np.meshgrid(range(H2p_steps), range(H_steps))[1],'k,')
										nH2p_ne_values_full,nH_ne_values_full = np.meshgrid(range(H2p_steps),range(H_steps))
										temp = np.exp(marginalised_power_penalty[:,most_likely_marginalised_nHm_nH2_index,:])
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH2p_ne_values_full.flatten(),nH_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit1,nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nH_ne_values_full.flatten(),nH2p_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH2p_ne_values_full = nH2p_ne_values_full[temp>np.max(temp)*start]
										nH_ne_values_full = nH_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH2p_ne_values_full,nH_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit1,nH2p_ne_values_full)))**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.sort(nH2p_ne_values_full),np.polyval(fit1,np.sort(nH2p_ne_values_full)),'b')
											fit2 = np.polyfit(nH_ne_values_full,nH2p_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nH_ne_values_full)))**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.polyval(fit2,np.sort(nH_ne_values_full)),np.sort(nH_ne_values_full),'b')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H-\nblue=fit H/H2+ (%.3g,%.3g)R2 %.3g\nfit H2+/H (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H-')
										ax[plot_index,1].set_ylabel('nH/ne index')
										ax[plot_index,1].set_xlabel('nH2+/nH2 index')
										# ax[plot_index,1].set_xscale('log')
										# ax[plot_index,1].set_yscale('log')

										plot_index += 1
										im = ax[plot_index,0].contourf(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(power_penalty[:,:,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index].T),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,0])
										im = ax[plot_index,0].contour(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(power_penalty[:,:,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index].T),levels=10);
										# im = ax[plot_index,0].plot(most_likely_marginalised_nH_ne_value,most_likely_marginalised_nHm_ne_value,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,0].plot(most_likely_nH_ne_value,most_likely_nHm_ne_value,color='k',marker='$X$',markersize=30);
										# im = ax[plot_index,0].plot(first_most_likely_marginalised_nH_ne_value,first_most_likely_marginalised_nHm_ne_value,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,0].plot(first_most_likely_nH_ne_value,first_most_likely_nHm_ne_value,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,0].plot([next_nH_ne_value_up,next_nH_ne_value_up,next_nH_ne_value_down,next_nH_ne_value_down,next_nH_ne_value_up],[next_nHm_ne_value_down,next_nHm_ne_value_up,next_nHm_ne_value_up,next_nHm_ne_value_down,next_nHm_ne_value_down],'k--');
										im = ax[plot_index,0].plot(np.meshgrid(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[0],np.meshgrid(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[1],'k,')
										nH_ne_values_full,nHm_ne_values_full = np.meshgrid(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])
										temp = np.exp(power_penalty[:,:,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index].T)
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH_ne_values_full.flatten(),nHm_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nHm_ne_values_full.flatten(),nH_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit2,nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH_ne_values_full = nH_ne_values_full[temp>np.max(temp)*start]
										nHm_ne_values_full = nHm_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH_ne_values_full,nHm_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH_ne_values_full)))**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.sort(nH_ne_values_full),np.polyval(fit1,np.sort(nH_ne_values_full)),'b')
											fit2 = np.polyfit(nHm_ne_values_full,nH_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit2,nHm_ne_values_full)))**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.polyval(fit2,np.sort(nHm_ne_values_full)),np.sort(nHm_ne_values_full),'b')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H2+\nM=marginalised best, X=unmarginalised best\nblue=fit H-/H (%.3g,%.3g)R2 %.3g\nfit H/H- (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H2+\nM=marginalised best, X=unmarginalised best')
										ax[plot_index,0].set_ylabel('nH-/ne')
										ax[plot_index,0].set_xlabel('nH/ne')
										ax[plot_index,0].set_xscale('log')
										ax[plot_index,0].set_yscale('log')

										im = ax[plot_index,1].contourf(range(H_steps),range(Hm_steps), np.exp(marginalised_power_penalty[:,:,most_likely_marginalised_nH2p_nH2_index].T),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,1])
										im = ax[plot_index,1].contour(range(H_steps),range(Hm_steps), np.exp(marginalised_power_penalty[:,:,most_likely_marginalised_nH2p_nH2_index].T),levels=10);
										im = ax[plot_index,1].plot(most_likely_marginalised_nH_ne_index,most_likely_marginalised_nHm_nH2_index,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,1].plot(most_likely_nH_ne_index,most_likely_nHm_nH2_index,color='k',marker='$X$',markersize=30);
										im = ax[plot_index,1].plot(first_most_likely_marginalised_nH_ne_index,first_most_likely_marginalised_nHm_nH2_index,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,1].plot(first_most_likely_nH_ne_index,first_most_likely_nHm_nH2_index,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,1].plot([next_nH_ne_value_up,next_nH_ne_value_up,next_nH_ne_value_down,next_nH_ne_value_down,next_nH_ne_value_up],[next_nHm_ne_value_down,next_nHm_ne_value_up,next_nHm_ne_value_up,next_nHm_ne_value_down,next_nHm_ne_value_down],'k--');
										im = ax[plot_index,1].plot(np.meshgrid(range(H_steps), range(Hm_steps))[0],np.meshgrid(range(H_steps), range(Hm_steps))[1],'k,')
										nH_ne_values_full,nHm_ne_values_full = np.meshgrid(range(H_steps),range(Hm_steps))
										temp = np.exp(marginalised_power_penalty[:,:,most_likely_marginalised_nH2p_nH2_index].T)
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH_ne_values_full.flatten(),nHm_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nHm_ne_values_full.flatten(),nH_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit2,nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH_ne_values_full = nH_ne_values_full[temp>np.max(temp)*start]
										nHm_ne_values_full = nHm_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH_ne_values_full,nHm_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH_ne_values_full)))**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.sort(nH_ne_values_full),np.polyval(fit1,np.sort(nH_ne_values_full)),'b')
											fit2 = np.polyfit(nHm_ne_values_full,nH_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit2,nHm_ne_values_full)))**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.polyval(fit2,np.sort(nHm_ne_values_full)),np.sort(nHm_ne_values_full),'b')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H2+\nblue=fit H-/H (%.3g,%.3g)R2 %.3g\nfit H/H- (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first_2.eps'+'fits failed')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H2+')
										ax[plot_index,1].set_ylabel('nH-/nH2 index')
										ax[plot_index,1].set_xlabel('nH/ne index')
										# ax[plot_index,1].set_xscale('log')
										# ax[plot_index,1].set_yscale('log')
										save_done = 0
										save_index=1
										try:	# All this trie are because of a known issues when saving lots of figure inside of multiprocessor
											plt.savefig(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first_2.eps', bbox_inches='tight')
										except Exception as e:
											print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first_2.eps save try number '+str(1)+' failed. Reason %s' % e)
											while save_done==0 and save_index<100:
												try:
													plt.savefig(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first_2.eps', bbox_inches='tight')
													print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first_2.eps save try number '+str(save_index+1)+' successfull')
													save_done=1
												except Exception as e:
													tm.sleep(np.random.random()**2)
													# print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(save_index+1)+' failed. Reason %s' % e)
													save_index+=1
										plt.close('all')

										print('First scan +3')
										fig, ax = plt.subplots( 3,2,figsize=(20, 35), squeeze=False)
										fig.suptitle('first scan +3\nmost_likely_nH_ne_value %.3g,most_likely_nHm_ne_value %.3g,most_likely_nH2_ne_value %.3g,most_likely_nH2p_ne_value %.3g' %(most_likely_nH_ne_value*most_likely_ne_value,most_likely_nHm_ne_value*most_likely_ne_value,most_likely_nH2_ne_value*most_likely_ne_value,most_likely_nH2p_ne_value*most_likely_ne_value) +'\nlines '+str(n_list_all)+'\nlocation [time, r]'+ ' [%.3g ms, ' % time_crop[my_time_pos] + ' %.3g mm]' % (1000*r_crop[my_r_pos]) +' , [TS Te, TS ne] '+ ' [%.3g(ML %.3g) eV, ' %(merge_Te_prof_multipulse_interp_crop_limited_restrict,most_likely_Te_value) + '%.3g(ML %.3g) #10^20/m^3]' %(merge_ne_prof_multipulse_interp_crop_limited_restrict,most_likely_ne_value*1e-20) +'\nfit [nH/ne, nH+/ne, nH-/ne, nH2/ne, nH2+/ne, nH3+/ne] , '+ '[%.3g, ' % most_likely_nH_ne_value + '%.3g, ' %(1 - most_likely_nH2p_ne_value + most_likely_nHm_ne_value)+ '%.3g, ' % most_likely_nHm_ne_value+ '%.3g, ' % most_likely_nH2_ne_value+ '%.3g, ' % most_likely_nH2p_ne_value+ '%.3g]' % 0+'\nBest index: [nH/ne,nH-/ne,nH2/ne,nH2+/ne,ne,Te] [%.3g,%.3g,%.3g,%.3g,%.3g,%.3g]' %(most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index)+'\nMarginalised best index: [nH/ne,nH-/ne,nH2+/ne] [%.3g,%.3g,%.3g]' %(most_likely_marginalised_nH_ne_index,most_likely_marginalised_nHm_nH2_index,most_likely_marginalised_nH2p_nH2_index)+'\nPDF_matrix_shape = '+str(PDF_matrix_shape) + '\n nHexcited<nH PENALTY')
										plot_index = 0
										im = ax[plot_index,0].contourf(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(nH_ne_penalty[most_likely_nH_ne_index,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,0])
										im = ax[plot_index,0].contour(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(nH_ne_penalty[most_likely_nH_ne_index,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]),levels=10);
										# im = ax[plot_index,0].plot(most_likely_marginalised_nH2p_ne_value,most_likely_marginalised_nHm_ne_value,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,0].plot(most_likely_nH2p_ne_value,most_likely_nHm_ne_value,color='k',marker='$X$',markersize=30);
										# im = ax[plot_index,0].plot(first_most_likely_marginalised_nH2p_ne_value,first_most_likely_marginalised_nHm_ne_value,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,0].plot(first_most_likely_nH2p_ne_value,first_most_likely_nHm_ne_value,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,0].plot([next_nH2p_ne_value_up,next_nH2p_ne_value_up,next_nH2p_ne_value_down,next_nH2p_ne_value_down,next_nH2p_ne_value_up],[next_nHm_ne_value_down,next_nHm_ne_value_up,next_nHm_ne_value_up,next_nHm_ne_value_down,next_nHm_ne_value_down],'k--');
										im = ax[plot_index,0].plot(np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[0],np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[1],'k,')
										nH2p_ne_values_full,nHm_ne_values_full = np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])
										temp = np.exp(nH_ne_penalty[most_likely_nH_ne_index,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index])
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH2p_ne_values_full.flatten(),nHm_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nHm_ne_values_full.flatten(),nH2p_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH2p_ne_values_full = nH2p_ne_values_full[temp>np.max(temp)*start]
										nHm_ne_values_full = nHm_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH2p_ne_values_full,nHm_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH2p_ne_values_full)))**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.sort(nH2p_ne_values_full),np.polyval(fit1,np.sort(nH2p_ne_values_full)),'b')
											fit2 = np.polyfit(nHm_ne_values_full,nH2p_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nHm_ne_values_full)))**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.polyval(fit2,np.sort(nHm_ne_values_full)),np.sort(nHm_ne_values_full),'b')
											fit_average = [(fit1[0]+1/fit2[0])/2,(fit1[1]-fit2[1]/fit2[0])/2]
											reverse_fit_average = [1/fit_average[0],-fit_average[1]/fit_average[0]]
											im = ax[plot_index,0].plot(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],np.polyval(fit_average,record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index]),'k--')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H\nM=marginalised best, X=unmarginalised best, --=samples\nblue=fit H-/H2+ (%.3g,%.3g)R2 %.3g\nfit H2+/H- (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H\nM=marginalised best, X=unmarginalised best, --=samples')
										ax[plot_index,0].set_ylabel('nH-/ne')
										ax[plot_index,0].set_xlabel('nH2+/ne')
										ax[plot_index,0].set_xscale('log')
										ax[plot_index,0].set_yscale('log')

										im = ax[plot_index,1].contourf(range(H2p_steps),range(Hm_steps), np.exp(marginalised_nH_ne_penalty[most_likely_marginalised_nH_ne_index]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,1])
										im = ax[plot_index,1].contour(range(H2p_steps),range(Hm_steps), np.exp(marginalised_nH_ne_penalty[most_likely_marginalised_nH_ne_index,:,:]),levels=10);
										im = ax[plot_index,1].plot(most_likely_marginalised_nH2p_nH2_index,most_likely_marginalised_nHm_nH2_index,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,1].plot(most_likely_nH2p_nH2_index,most_likely_nHm_nH2_index,color='k',marker='$X$',markersize=30);
										im = ax[plot_index,1].plot(first_most_likely_marginalised_nH2p_nH2_index,first_most_likely_marginalised_nHm_nH2_index,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,1].plot(first_most_likely_nH2p_nH2_index,first_most_likely_nHm_nH2_index,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,1].plot([next_nH2p_ne_value_up,next_nH2p_ne_value_up,next_nH2p_ne_value_down,next_nH2p_ne_value_down,next_nH2p_ne_value_up],[next_nHm_ne_value_down,next_nHm_ne_value_up,next_nHm_ne_value_up,next_nHm_ne_value_down,next_nHm_ne_value_down],'k--');
										im = ax[plot_index,1].plot(np.meshgrid(range(H2p_steps), range(Hm_steps))[0],np.meshgrid(range(H2p_steps), range(Hm_steps))[1],'k,')
										nH2p_ne_values_full,nHm_ne_values_full = np.meshgrid(range(H2p_steps),range(Hm_steps))
										temp = np.exp(marginalised_nH_ne_penalty[most_likely_marginalised_nH_ne_index])
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH2p_ne_values_full.flatten(),nHm_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nHm_ne_values_full.flatten(),nH2p_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH2p_ne_values_full = nH2p_ne_values_full[temp>np.max(temp)*start]
										nHm_ne_values_full = nHm_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH2p_ne_values_full,nHm_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH2p_ne_values_full)))**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.sort(nH2p_ne_values_full),np.polyval(fit1,np.sort(nH2p_ne_values_full)),'b')
											fit2 = np.polyfit(nHm_ne_values_full,nH2p_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nHm_ne_values_full)))**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.polyval(fit2,np.sort(nHm_ne_values_full)),np.sort(nHm_ne_values_full),'b')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H\nblue=fit H-/H2+ (%.3g,%.3g)R2 %.3g\nfit H2+/H- (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H')
										ax[plot_index,1].set_ylabel('nH-/nH2 index')
										ax[plot_index,1].set_xlabel('nH2+/nH2 index')
										# ax[plot_index,1].set_xscale('log')
										# ax[plot_index,1].set_yscale('log')

										plot_index += 1
										im = ax[plot_index,0].contourf(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(nH_ne_penalty[:,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,0])
										im = ax[plot_index,0].contour(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(nH_ne_penalty[:,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]),levels=10);
										# im = ax[plot_index,0].plot(most_likely_marginalised_nH2p_ne_value,most_likely_marginalised_nH_ne_value,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,0].plot(most_likely_nH2p_ne_value,most_likely_nH_ne_value,color='k',marker='$X$',markersize=30);
										# im = ax[plot_index,0].plot(first_most_likely_marginalised_nH2p_ne_value,first_most_likely_marginalised_nH_ne_value,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,0].plot(first_most_likely_nH2p_ne_value,first_most_likely_nH_ne_value,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,0].plot([next_nH2p_ne_value_up,next_nH2p_ne_value_up,next_nH2p_ne_value_down,next_nH2p_ne_value_down,next_nH2p_ne_value_up],[next_nH_ne_value_down,next_nH_ne_value_up,next_nH_ne_value_up,next_nH_ne_value_down,next_nH_ne_value_down],'k--');
										im = ax[plot_index,0].plot(np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[0],np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[1],'k,')
										nH2p_ne_values_full,nH_ne_values_full = np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])
										temp = np.exp(nH_ne_penalty[:,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index])
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH2p_ne_values_full.flatten(),nH_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit1,nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nH_ne_values_full.flatten(),nH2p_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH2p_ne_values_full = nH2p_ne_values_full[temp>np.max(temp)*start]
										nH_ne_values_full = nH_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH2p_ne_values_full,nH_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit1,nH2p_ne_values_full)))**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.sort(nH2p_ne_values_full),np.polyval(fit1,np.sort(nH2p_ne_values_full)),'b')
											fit2 = np.polyfit(nH_ne_values_full,nH2p_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nH_ne_values_full)))**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.polyval(fit2,np.sort(nH_ne_values_full)),np.sort(nH_ne_values_full),'b')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H-\nM=marginalised best, X=unmarginalised best\nblue=fit H/H2+ (%.3g,%.3g)R2 %.3g\nfit H2+/H (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H-\nM=marginalised best, X=unmarginalised best')
										ax[plot_index,0].set_ylabel('nH/ne')
										ax[plot_index,0].set_xlabel('nH2+/ne')
										ax[plot_index,0].set_xscale('log')
										ax[plot_index,0].set_yscale('log')

										im = ax[plot_index,1].contourf(range(H2p_steps),range(H_steps), np.exp(marginalised_nH_ne_penalty[:,most_likely_marginalised_nHm_nH2_index,:]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,1])
										im = ax[plot_index,1].contour(range(H2p_steps),range(H_steps), np.exp(marginalised_nH_ne_penalty[:,most_likely_marginalised_nHm_nH2_index,:]),levels=10);
										im = ax[plot_index,1].plot(most_likely_marginalised_nH2p_nH2_index,most_likely_marginalised_nH_ne_index,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,1].plot(most_likely_nH2p_nH2_index,most_likely_nH_ne_index,color='k',marker='$X$',markersize=30);
										im = ax[plot_index,1].plot(first_most_likely_marginalised_nH2p_nH2_index,first_most_likely_marginalised_nH_ne_index,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,1].plot(first_most_likely_nH2p_nH2_index,first_most_likely_nH_ne_index,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,1].plot([next_nH2p_ne_value_up,next_nH2p_ne_value_up,next_nH2p_ne_value_down,next_nH2p_ne_value_down,next_nH2p_ne_value_up],[next_nH_ne_value_down,next_nH_ne_value_up,next_nH_ne_value_up,next_nH_ne_value_down,next_nH_ne_value_down],'k--');
										im = ax[plot_index,1].plot(np.meshgrid(range(H2p_steps), range(H_steps))[0],np.meshgrid(range(H2p_steps), range(H_steps))[1],'k,')
										nH2p_ne_values_full,nH_ne_values_full = np.meshgrid(range(H2p_steps),range(H_steps))
										temp = np.exp(marginalised_nH_ne_penalty[:,most_likely_marginalised_nHm_nH2_index,:])
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH2p_ne_values_full.flatten(),nH_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit1,nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nH_ne_values_full.flatten(),nH2p_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH2p_ne_values_full = nH2p_ne_values_full[temp>np.max(temp)*start]
										nH_ne_values_full = nH_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH2p_ne_values_full,nH_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit1,nH2p_ne_values_full)))**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.sort(nH2p_ne_values_full),np.polyval(fit1,np.sort(nH2p_ne_values_full)),'b')
											fit2 = np.polyfit(nH_ne_values_full,nH2p_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nH_ne_values_full)))**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.polyval(fit2,np.sort(nH_ne_values_full)),np.sort(nH_ne_values_full),'b')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H-\nblue=fit H/H2+ (%.3g,%.3g)R2 %.3g\nfit H2+/H (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H-')
										ax[plot_index,1].set_ylabel('nH/ne index')
										ax[plot_index,1].set_xlabel('nH2+/nH2 index')
										# ax[plot_index,1].set_xscale('log')
										# ax[plot_index,1].set_yscale('log')

										plot_index += 1
										im = ax[plot_index,0].contourf(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(nH_ne_penalty[:,:,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index].T),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,0])
										im = ax[plot_index,0].contour(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(nH_ne_penalty[:,:,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index].T),levels=10);
										# im = ax[plot_index,0].plot(most_likely_marginalised_nH_ne_value,most_likely_marginalised_nHm_ne_value,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,0].plot(most_likely_nH_ne_value,most_likely_nHm_ne_value,color='k',marker='$X$',markersize=30);
										# im = ax[plot_index,0].plot(first_most_likely_marginalised_nH_ne_value,first_most_likely_marginalised_nHm_ne_value,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,0].plot(first_most_likely_nH_ne_value,first_most_likely_nHm_ne_value,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,0].plot([next_nH_ne_value_up,next_nH_ne_value_up,next_nH_ne_value_down,next_nH_ne_value_down,next_nH_ne_value_up],[next_nHm_ne_value_down,next_nHm_ne_value_up,next_nHm_ne_value_up,next_nHm_ne_value_down,next_nHm_ne_value_down],'k--');
										im = ax[plot_index,0].plot(np.meshgrid(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[0],np.meshgrid(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[1],'k,')
										nH_ne_values_full,nHm_ne_values_full = np.meshgrid(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])
										temp = np.exp(nH_ne_penalty[:,:,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index].T)
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH_ne_values_full.flatten(),nHm_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nHm_ne_values_full.flatten(),nH_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit2,nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH_ne_values_full = nH_ne_values_full[temp>np.max(temp)*start]
										nHm_ne_values_full = nHm_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH_ne_values_full,nHm_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH_ne_values_full)))**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.sort(nH_ne_values_full),np.polyval(fit1,np.sort(nH_ne_values_full)),'b')
											fit2 = np.polyfit(nHm_ne_values_full,nH_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit2,nHm_ne_values_full)))**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.polyval(fit2,np.sort(nHm_ne_values_full)),np.sort(nHm_ne_values_full),'b')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H2+\nM=marginalised best, X=unmarginalised best\nblue=fit H-/H (%.3g,%.3g)R2 %.3g\nfit H/H- (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H2+\nM=marginalised best, X=unmarginalised best')
										ax[plot_index,0].set_ylabel('nH-/ne')
										ax[plot_index,0].set_xlabel('nH/ne')
										ax[plot_index,0].set_xscale('log')
										ax[plot_index,0].set_yscale('log')

										im = ax[plot_index,1].contourf(range(H_steps),range(Hm_steps), np.exp(marginalised_nH_ne_penalty[:,:,most_likely_marginalised_nH2p_nH2_index].T),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,1])
										im = ax[plot_index,1].contour(range(H_steps),range(Hm_steps), np.exp(marginalised_nH_ne_penalty[:,:,most_likely_marginalised_nH2p_nH2_index].T),levels=10);
										im = ax[plot_index,1].plot(most_likely_marginalised_nH_ne_index,most_likely_marginalised_nHm_nH2_index,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,1].plot(most_likely_nH_ne_index,most_likely_nHm_nH2_index,color='k',marker='$X$',markersize=30);
										im = ax[plot_index,1].plot(first_most_likely_marginalised_nH_ne_index,first_most_likely_marginalised_nHm_nH2_index,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,1].plot(first_most_likely_nH_ne_index,first_most_likely_nHm_nH2_index,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,1].plot([next_nH_ne_value_up,next_nH_ne_value_up,next_nH_ne_value_down,next_nH_ne_value_down,next_nH_ne_value_up],[next_nHm_ne_value_down,next_nHm_ne_value_up,next_nHm_ne_value_up,next_nHm_ne_value_down,next_nHm_ne_value_down],'k--');
										im = ax[plot_index,1].plot(np.meshgrid(range(H_steps), range(Hm_steps))[0],np.meshgrid(range(H_steps), range(Hm_steps))[1],'k,')
										nH_ne_values_full,nHm_ne_values_full = np.meshgrid(range(H_steps),range(Hm_steps))
										temp = np.exp(marginalised_nH_ne_penalty[:,:,most_likely_marginalised_nH2p_nH2_index].T)
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH_ne_values_full.flatten(),nHm_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nHm_ne_values_full.flatten(),nH_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit2,nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH_ne_values_full = nH_ne_values_full[temp>np.max(temp)*start]
										nHm_ne_values_full = nHm_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH_ne_values_full,nHm_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH_ne_values_full)))**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.sort(nH_ne_values_full),np.polyval(fit1,np.sort(nH_ne_values_full)),'b')
											fit2 = np.polyfit(nHm_ne_values_full,nH_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit2,nHm_ne_values_full)))**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.polyval(fit2,np.sort(nHm_ne_values_full)),np.sort(nHm_ne_values_full),'b')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H2+\nblue=fit H-/H (%.3g,%.3g)R2 %.3g\nfit H/H- (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first_3.eps'+'fits failed')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H2+')
										ax[plot_index,1].set_ylabel('nH-/nH2 index')
										ax[plot_index,1].set_xlabel('nH/ne index')
										# ax[plot_index,1].set_xscale('log')
										# ax[plot_index,1].set_yscale('log')
										save_done = 0
										save_index=1
										try:	# All this trie are because of a known issues when saving lots of figure inside of multiprocessor
											plt.savefig(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first_3.eps', bbox_inches='tight')
										except Exception as e:
											print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first_3.eps save try number '+str(1)+' failed. Reason %s' % e)
											while save_done==0 and save_index<100:
												try:
													plt.savefig(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first_3.eps', bbox_inches='tight')
													print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first_3.eps save try number '+str(save_index+1)+' successfull')
													save_done=1
												except Exception as e:
													tm.sleep(np.random.random()**2)
													# print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(save_index+1)+' failed. Reason %s' % e)
													save_index+=1
										plt.close('all')

										print('First scan +4')
										fig, ax = plt.subplots( 3,2,figsize=(20, 35), squeeze=False)
										fig.suptitle('first scan + 4\nmost_likely_nH_ne_value %.3g,most_likely_nHm_ne_value %.3g,most_likely_nH2_ne_value %.3g,most_likely_nH2p_ne_value %.3g' %(most_likely_nH_ne_value*most_likely_ne_value,most_likely_nHm_ne_value*most_likely_ne_value,most_likely_nH2_ne_value*most_likely_ne_value,most_likely_nH2p_ne_value*most_likely_ne_value) +'\nlines '+str(n_list_all)+'\nlocation [time, r]'+ ' [%.3g ms, ' % time_crop[my_time_pos] + ' %.3g mm]' % (1000*r_crop[my_r_pos]) +' , [TS Te, TS ne] '+ ' [%.3g(ML %.3g) eV, ' %(merge_Te_prof_multipulse_interp_crop_limited_restrict,most_likely_Te_value) + '%.3g(ML %.3g) #10^20/m^3]' %(merge_ne_prof_multipulse_interp_crop_limited_restrict,most_likely_ne_value*1e-20) +'\nfit [nH/ne, nH+/ne, nH-/ne, nH2/ne, nH2+/ne, nH3+/ne] , '+ '[%.3g, ' % most_likely_nH_ne_value + '%.3g, ' %(1 - most_likely_nH2p_ne_value + most_likely_nHm_ne_value)+ '%.3g, ' % most_likely_nHm_ne_value+ '%.3g, ' % most_likely_nH2_ne_value+ '%.3g, ' % most_likely_nH2p_ne_value+ '%.3g]' % 0+'\nBest index: [nH/ne,nH-/ne,nH2/ne,nH2+/ne,ne,Te] [%.3g,%.3g,%.3g,%.3g,%.3g,%.3g]' %(most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index)+'\nMarginalised best index: [nH/ne,nH-/ne,nH2+/ne] [%.3g,%.3g,%.3g]' %(most_likely_marginalised_nH_ne_index,most_likely_marginalised_nHm_nH2_index,most_likely_marginalised_nH2p_nH2_index)+'\nPDF_matrix_shape = '+str(PDF_matrix_shape) + '\n PARTICLE BALANCE PENALTY')
										plot_index = 0
										im = ax[plot_index,0].contourf(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(particles_penalty[most_likely_nH_ne_index,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,0])
										im = ax[plot_index,0].contour(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(particles_penalty[most_likely_nH_ne_index,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]),levels=10);
										# im = ax[plot_index,0].plot(most_likely_marginalised_nH2p_ne_value,most_likely_marginalised_nHm_ne_value,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,0].plot(most_likely_nH2p_ne_value,most_likely_nHm_ne_value,color='k',marker='$X$',markersize=30);
										# im = ax[plot_index,0].plot(first_most_likely_marginalised_nH2p_ne_value,first_most_likely_marginalised_nHm_ne_value,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,0].plot(first_most_likely_nH2p_ne_value,first_most_likely_nHm_ne_value,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,0].plot([next_nH2p_ne_value_up,next_nH2p_ne_value_up,next_nH2p_ne_value_down,next_nH2p_ne_value_down,next_nH2p_ne_value_up],[next_nHm_ne_value_down,next_nHm_ne_value_up,next_nHm_ne_value_up,next_nHm_ne_value_down,next_nHm_ne_value_down],'k--');
										im = ax[plot_index,0].plot(np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[0],np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[1],'k,')
										nH2p_ne_values_full,nHm_ne_values_full = np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])
										temp = np.exp(particles_penalty[most_likely_nH_ne_index,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index])
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH2p_ne_values_full.flatten(),nHm_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nHm_ne_values_full.flatten(),nH2p_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH2p_ne_values_full = nH2p_ne_values_full[temp>np.max(temp)*start]
										nHm_ne_values_full = nHm_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH2p_ne_values_full,nHm_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH2p_ne_values_full)))**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.sort(nH2p_ne_values_full),np.polyval(fit1,np.sort(nH2p_ne_values_full)),'b')
											fit2 = np.polyfit(nHm_ne_values_full,nH2p_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nHm_ne_values_full)))**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.polyval(fit2,np.sort(nHm_ne_values_full)),np.sort(nHm_ne_values_full),'b')
											fit_average = [(fit1[0]+1/fit2[0])/2,(fit1[1]-fit2[1]/fit2[0])/2]
											reverse_fit_average = [1/fit_average[0],-fit_average[1]/fit_average[0]]
											im = ax[plot_index,0].plot(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],np.polyval(fit_average,record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index]),'k--')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H\nM=marginalised best, X=unmarginalised best, --=samples\nblue=fit H-/H2+ (%.3g,%.3g)R2 %.3g\nfit H2+/H- (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H\nM=marginalised best, X=unmarginalised best, --=samples')
										ax[plot_index,0].set_ylabel('nH-/ne')
										ax[plot_index,0].set_xlabel('nH2+/ne')
										ax[plot_index,0].set_xscale('log')
										ax[plot_index,0].set_yscale('log')

										im = ax[plot_index,1].contourf(range(H2p_steps),range(Hm_steps), np.exp(marginalised_particles_penalty[most_likely_marginalised_nH_ne_index]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,1])
										im = ax[plot_index,1].contour(range(H2p_steps),range(Hm_steps), np.exp(marginalised_particles_penalty[most_likely_marginalised_nH_ne_index,:,:]),levels=10);
										im = ax[plot_index,1].plot(most_likely_marginalised_nH2p_nH2_index,most_likely_marginalised_nHm_nH2_index,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,1].plot(most_likely_nH2p_nH2_index,most_likely_nHm_nH2_index,color='k',marker='$X$',markersize=30);
										im = ax[plot_index,1].plot(first_most_likely_marginalised_nH2p_nH2_index,first_most_likely_marginalised_nHm_nH2_index,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,1].plot(first_most_likely_nH2p_nH2_index,first_most_likely_nHm_nH2_index,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,1].plot([next_nH2p_ne_value_up,next_nH2p_ne_value_up,next_nH2p_ne_value_down,next_nH2p_ne_value_down,next_nH2p_ne_value_up],[next_nHm_ne_value_down,next_nHm_ne_value_up,next_nHm_ne_value_up,next_nHm_ne_value_down,next_nHm_ne_value_down],'k--');
										im = ax[plot_index,1].plot(np.meshgrid(range(H2p_steps), range(Hm_steps))[0],np.meshgrid(range(H2p_steps), range(Hm_steps))[1],'k,')
										nH2p_ne_values_full,nHm_ne_values_full = np.meshgrid(range(H2p_steps),range(Hm_steps))
										temp = np.exp(marginalised_particles_penalty[most_likely_marginalised_nH_ne_index])
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH2p_ne_values_full.flatten(),nHm_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nHm_ne_values_full.flatten(),nH2p_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH2p_ne_values_full = nH2p_ne_values_full[temp>np.max(temp)*start]
										nHm_ne_values_full = nHm_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH2p_ne_values_full,nHm_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH2p_ne_values_full)))**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.sort(nH2p_ne_values_full),np.polyval(fit1,np.sort(nH2p_ne_values_full)),'b')
											fit2 = np.polyfit(nHm_ne_values_full,nH2p_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nHm_ne_values_full)))**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.polyval(fit2,np.sort(nHm_ne_values_full)),np.sort(nHm_ne_values_full),'b')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H\nblue=fit H-/H2+ (%.3g,%.3g)R2 %.3g\nfit H2+/H- (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H')
										ax[plot_index,1].set_ylabel('nH-/nH2 index')
										ax[plot_index,1].set_xlabel('nH2+/nH2 index')
										# ax[plot_index,1].set_xscale('log')
										# ax[plot_index,1].set_yscale('log')

										plot_index += 1
										im = ax[plot_index,0].contourf(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(particles_penalty[:,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,0])
										im = ax[plot_index,0].contour(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(particles_penalty[:,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]),levels=10);
										# im = ax[plot_index,0].plot(most_likely_marginalised_nH2p_ne_value,most_likely_marginalised_nH_ne_value,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,0].plot(most_likely_nH2p_ne_value,most_likely_nH_ne_value,color='k',marker='$X$',markersize=30);
										# im = ax[plot_index,0].plot(first_most_likely_marginalised_nH2p_ne_value,first_most_likely_marginalised_nH_ne_value,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,0].plot(first_most_likely_nH2p_ne_value,first_most_likely_nH_ne_value,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,0].plot([next_nH2p_ne_value_up,next_nH2p_ne_value_up,next_nH2p_ne_value_down,next_nH2p_ne_value_down,next_nH2p_ne_value_up],[next_nH_ne_value_down,next_nH_ne_value_up,next_nH_ne_value_up,next_nH_ne_value_down,next_nH_ne_value_down],'k--');
										im = ax[plot_index,0].plot(np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[0],np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[1],'k,')
										nH2p_ne_values_full,nH_ne_values_full = np.meshgrid(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])
										temp = np.exp(particles_penalty[:,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index])
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH2p_ne_values_full.flatten(),nH_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit1,nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nH_ne_values_full.flatten(),nH2p_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH2p_ne_values_full = nH2p_ne_values_full[temp>np.max(temp)*start]
										nH_ne_values_full = nH_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH2p_ne_values_full,nH_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit1,nH2p_ne_values_full)))**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.sort(nH2p_ne_values_full),np.polyval(fit1,np.sort(nH2p_ne_values_full)),'b')
											fit2 = np.polyfit(nH_ne_values_full,nH2p_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nH_ne_values_full)))**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.polyval(fit2,np.sort(nH_ne_values_full)),np.sort(nH_ne_values_full),'b')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H-\nM=marginalised best, X=unmarginalised best\nblue=fit H/H2+ (%.3g,%.3g)R2 %.3g\nfit H2+/H (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H-\nM=marginalised best, X=unmarginalised best')
										ax[plot_index,0].set_ylabel('nH/ne')
										ax[plot_index,0].set_xlabel('nH2+/ne')
										ax[plot_index,0].set_xscale('log')
										ax[plot_index,0].set_yscale('log')

										im = ax[plot_index,1].contourf(range(H2p_steps),range(H_steps), np.exp(marginalised_particles_penalty[:,most_likely_marginalised_nHm_nH2_index,:]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,1])
										im = ax[plot_index,1].contour(range(H2p_steps),range(H_steps), np.exp(marginalised_particles_penalty[:,most_likely_marginalised_nHm_nH2_index,:]),levels=10);
										im = ax[plot_index,1].plot(most_likely_marginalised_nH2p_nH2_index,most_likely_marginalised_nH_ne_index,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,1].plot(most_likely_nH2p_nH2_index,most_likely_nH_ne_index,color='k',marker='$X$',markersize=30);
										im = ax[plot_index,1].plot(first_most_likely_marginalised_nH2p_nH2_index,first_most_likely_marginalised_nH_ne_index,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,1].plot(first_most_likely_nH2p_nH2_index,first_most_likely_nH_ne_index,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,1].plot([next_nH2p_ne_value_up,next_nH2p_ne_value_up,next_nH2p_ne_value_down,next_nH2p_ne_value_down,next_nH2p_ne_value_up],[next_nH_ne_value_down,next_nH_ne_value_up,next_nH_ne_value_up,next_nH_ne_value_down,next_nH_ne_value_down],'k--');
										im = ax[plot_index,1].plot(np.meshgrid(range(H2p_steps), range(H_steps))[0],np.meshgrid(range(H2p_steps), range(H_steps))[1],'k,')
										nH2p_ne_values_full,nH_ne_values_full = np.meshgrid(range(H2p_steps),range(H_steps))
										temp = np.exp(marginalised_particles_penalty[:,most_likely_marginalised_nHm_nH2_index,:])
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH2p_ne_values_full.flatten(),nH_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit1,nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nH_ne_values_full.flatten(),nH2p_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH2p_ne_values_full = nH2p_ne_values_full[temp>np.max(temp)*start]
										nH_ne_values_full = nH_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH2p_ne_values_full,nH_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit1,nH2p_ne_values_full)))**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.sort(nH2p_ne_values_full),np.polyval(fit1,np.sort(nH2p_ne_values_full)),'b')
											fit2 = np.polyfit(nH_ne_values_full,nH2p_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH2p_ne_values_full-np.polyval(fit2,nH_ne_values_full)))**2))/(np.sum(((nH2p_ne_values_full-np.mean(nH2p_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.polyval(fit2,np.sort(nH_ne_values_full)),np.sort(nH_ne_values_full),'b')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H-\nblue=fit H/H2+ (%.3g,%.3g)R2 %.3g\nfit H2+/H (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H-')
										ax[plot_index,1].set_ylabel('nH/ne index')
										ax[plot_index,1].set_xlabel('nH2+/nH2 index')
										# ax[plot_index,1].set_xscale('log')
										# ax[plot_index,1].set_yscale('log')

										plot_index += 1
										im = ax[plot_index,0].contourf(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(particles_penalty[:,:,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index].T),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,0])
										im = ax[plot_index,0].contour(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], np.exp(particles_penalty[:,:,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index].T),levels=10);
										# im = ax[plot_index,0].plot(most_likely_marginalised_nH_ne_value,most_likely_marginalised_nHm_ne_value,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,0].plot(most_likely_nH_ne_value,most_likely_nHm_ne_value,color='k',marker='$X$',markersize=30);
										# im = ax[plot_index,0].plot(first_most_likely_marginalised_nH_ne_value,first_most_likely_marginalised_nHm_ne_value,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,0].plot(first_most_likely_nH_ne_value,first_most_likely_nHm_ne_value,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,0].plot([next_nH_ne_value_up,next_nH_ne_value_up,next_nH_ne_value_down,next_nH_ne_value_down,next_nH_ne_value_up],[next_nHm_ne_value_down,next_nHm_ne_value_up,next_nHm_ne_value_up,next_nHm_ne_value_down,next_nHm_ne_value_down],'k--');
										im = ax[plot_index,0].plot(np.meshgrid(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[0],np.meshgrid(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index], record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[1],'k,')
										nH_ne_values_full,nHm_ne_values_full = np.meshgrid(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])
										temp = np.exp(particles_penalty[:,:,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index].T)
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH_ne_values_full.flatten(),nHm_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nHm_ne_values_full.flatten(),nH_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit2,nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH_ne_values_full = nH_ne_values_full[temp>np.max(temp)*start]
										nHm_ne_values_full = nHm_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH_ne_values_full,nHm_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH_ne_values_full)))**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.sort(nH_ne_values_full),np.polyval(fit1,np.sort(nH_ne_values_full)),'b')
											fit2 = np.polyfit(nHm_ne_values_full,nH_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit2,nHm_ne_values_full)))**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full)))**2))
											im = ax[plot_index,0].plot(np.polyval(fit2,np.sort(nHm_ne_values_full)),np.sort(nHm_ne_values_full),'b')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H2+\nM=marginalised best, X=unmarginalised best\nblue=fit H-/H (%.3g,%.3g)R2 %.3g\nfit H/H- (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/post_process_mega_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first.eps'+'fits failed')
											ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, H2+\nM=marginalised best, X=unmarginalised best')
										ax[plot_index,0].set_ylabel('nH-/ne')
										ax[plot_index,0].set_xlabel('nH/ne')
										ax[plot_index,0].set_xscale('log')
										ax[plot_index,0].set_yscale('log')

										im = ax[plot_index,1].contourf(range(H_steps),range(Hm_steps), np.exp(marginalised_particles_penalty[:,:,most_likely_marginalised_nH2p_nH2_index].T),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,1])
										im = ax[plot_index,1].contour(range(H_steps),range(Hm_steps), np.exp(marginalised_particles_penalty[:,:,most_likely_marginalised_nH2p_nH2_index].T),levels=10);
										im = ax[plot_index,1].plot(most_likely_marginalised_nH_ne_index,most_likely_marginalised_nHm_nH2_index,color='k',marker='$M$',markersize=30);
										im = ax[plot_index,1].plot(most_likely_nH_ne_index,most_likely_nHm_nH2_index,color='k',marker='$X$',markersize=30);
										im = ax[plot_index,1].plot(first_most_likely_marginalised_nH_ne_index,first_most_likely_marginalised_nHm_nH2_index,color='y',marker='$M$',markersize=15);
										im = ax[plot_index,1].plot(first_most_likely_nH_ne_index,first_most_likely_nHm_nH2_index,color='y',marker='$X$',markersize=15);
										# if scan_after_first:
										# 	im = ax[plot_index,1].plot([next_nH_ne_value_up,next_nH_ne_value_up,next_nH_ne_value_down,next_nH_ne_value_down,next_nH_ne_value_up],[next_nHm_ne_value_down,next_nHm_ne_value_up,next_nHm_ne_value_up,next_nHm_ne_value_down,next_nHm_ne_value_down],'k--');
										im = ax[plot_index,1].plot(np.meshgrid(range(H_steps), range(Hm_steps))[0],np.meshgrid(range(H_steps), range(Hm_steps))[1],'k,')
										nH_ne_values_full,nHm_ne_values_full = np.meshgrid(range(H_steps),range(Hm_steps))
										temp = np.exp(marginalised_particles_penalty[:,:,most_likely_marginalised_nH2p_nH2_index].T)
										temp[temp<1e-100]=1e-100
										# fit1 = curve_fit(line,nH_ne_values_full.flatten(),nHm_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										# fit2 = curve_fit(line,nHm_ne_values_full.flatten(),nH_ne_values_full.flatten(),p0=[1,1],sigma=(1/temp).flatten())[0]
										# R2_fit2 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit2,nHm_ne_values_full))*temp)**2)/np.sum(1*temp**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full))*temp)**2)/np.sum(1*temp**2))
										start = 0.8
										while (np.sum(temp>np.max(temp)*start)<5 and start>0):
											start/=1.1
										nH_ne_values_full = nH_ne_values_full[temp>np.max(temp)*start]
										nHm_ne_values_full = nHm_ne_values_full[temp>np.max(temp)*start]
										try:
											fit1 = np.polyfit(nH_ne_values_full,nHm_ne_values_full,1)
											R2_fit1 = 1-(np.sum(((nHm_ne_values_full-np.polyval(fit1,nH_ne_values_full)))**2))/(np.sum(((nHm_ne_values_full-np.mean(nHm_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.sort(nH_ne_values_full),np.polyval(fit1,np.sort(nH_ne_values_full)),'b')
											fit2 = np.polyfit(nHm_ne_values_full,nH_ne_values_full,1)
											R2_fit2 = 1-(np.sum(((nH_ne_values_full-np.polyval(fit2,nHm_ne_values_full)))**2))/(np.sum(((nH_ne_values_full-np.mean(nH_ne_values_full)))**2))
											im = ax[plot_index,1].plot(np.polyval(fit2,np.sort(nHm_ne_values_full)),np.sort(nHm_ne_values_full),'b')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H2+\nblue=fit H-/H (%.3g,%.3g)R2 %.3g\nfit H/H- (%.3g,%.3g)R2 %.3g' %(*fit1,R2_fit1,*fit2,R2_fit2))
										except:
											print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first_4.eps'+'fits failed')
											ax[plot_index,1].set_title('PDF marginalised on H2, Te, ne, most likely H2+')
										ax[plot_index,1].set_ylabel('nH-/nH2 index')
										ax[plot_index,1].set_xlabel('nH/ne index')
										# ax[plot_index,1].set_xscale('log')
										# ax[plot_index,1].set_yscale('log')
										save_done = 0
										save_index=1
										try:	# All this trie are because of a known issues when saving lots of figure inside of multiprocessor
											plt.savefig(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first_4.eps', bbox_inches='tight')
										except Exception as e:
											print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first_4.eps save try number '+str(1)+' failed. Reason %s' % e)
											while save_done==0 and save_index<100:
												try:
													plt.savefig(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first_4.eps', bbox_inches='tight')
													print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_first_4.eps save try number '+str(save_index+1)+' successfull')
													save_done=1
												except Exception as e:
													tm.sleep(np.random.random()**2)
													# print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(save_index+1)+' failed. Reason %s' % e)
													save_index+=1
										plt.close('all')


							if my_time_pos in sample_time_step:
								if my_r_pos in sample_radious:
									if to_print:
										# tm.sleep(np.random.random()*10)
										fig, ax = plt.subplots( 4,2,figsize=(22, 30), squeeze=False)
										fig.suptitle('first scan\nmost_likely_nH_ne_value %.3g,most_likely_nHm_ne_value %.3g,most_likely_nH2_ne_value %.3g,most_likely_nH2p_ne_value %.3g' %(most_likely_nH_ne_value*most_likely_ne_value,most_likely_nHm_ne_value*most_likely_ne_value,most_likely_nH2_ne_value*most_likely_ne_value,most_likely_nH2p_ne_value*most_likely_ne_value) +'\nlines '+str(n_list_all)+'\nlocation [time, r]'+ ' [%.3g ms, ' % time_crop[my_time_pos] + ' %.3g mm]' % (1000*r_crop[my_r_pos]) +' , [TS Te, TS ne] '+ ' [%.3g(ML %.3g) eV, ' %(merge_Te_prof_multipulse_interp_crop_limited_restrict,most_likely_Te_value) + '%.3g(ML %.3g) #10^20/m^3]' %(merge_ne_prof_multipulse_interp_crop_limited_restrict,most_likely_ne_value*1e-20) +'\nfit [nH/ne, nH+/ne, nH-/ne, nH2/ne, nH2+/ne, nH3+/ne] , '+ '[%.3g, ' % most_likely_nH_ne_value + '%.3g, ' %(1 - most_likely_nH2p_ne_value + most_likely_nHm_ne_value)+ '%.3g, ' % most_likely_nHm_ne_value+ '%.3g, ' % most_likely_nH2_ne_value+ '%.3g, ' % most_likely_nH2p_ne_value+ '%.3g]' % 0+'\nBest index: [nH/ne,nH-/ne,nH2/ne,nH2+/ne,ne,Te] [%.3g,%.3g,%.3g,%.3g,%.3g,%.3g]' %(most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index)+'\nMarginalised best index: [nH/ne,nH-/ne,nH2+/ne] [%.3g,%.3g,%.3g]' %(most_likely_marginalised_nH_ne_index,most_likely_marginalised_nHm_nH2_index,most_likely_marginalised_nH2p_nH2_index))
										plot_index = 0
										index = likelihood_log_probs[:,0,:,:,:,:].argmax()	# H, Hm, H2, H2p, ne, Te
										Te_index = index%TS_Te_steps
										ne_index = (index-Te_index)//TS_Te_steps%TS_ne_steps
										nH2p_nH2_index = ((index-Te_index)//TS_Te_steps-ne_index)//TS_ne_steps%H2p_steps
										nH2_ne_index = (((index-Te_index)//TS_Te_steps-ne_index)//TS_ne_steps-nH2p_nH2_index)//H2p_steps%H2_steps
										nH_ne_index = ((((index-Te_index)//TS_Te_steps-ne_index)//TS_ne_steps-nH2p_nH2_index)//H2p_steps - nH2_ne_index)//H2_steps%H_steps
										im = ax[plot_index,0].contourf(record_nH2p_ne_values[nH2_ne_index,:,ne_index*TS_Te_steps + Te_index],record_nH_ne_values[:,ne_index*TS_Te_steps + Te_index], np.exp(likelihood_log_probs[:,0,nH2_ne_index,:,ne_index,Te_index]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,0])
										im = ax[plot_index,0].contour(record_nH2p_ne_values[nH2_ne_index,:,ne_index*TS_Te_steps + Te_index],record_nH_ne_values[:,ne_index*TS_Te_steps + Te_index], np.exp(likelihood_log_probs[:,0,nH2_ne_index,:,ne_index,Te_index]),levels=10);
										# im = ax[plot_index,0].plot(most_likely_marginalised_nH2p_ne_value,most_likely_marginalised_nH_ne_value,color='k',marker='$M$',markersize=10);
										im = ax[plot_index,0].plot(most_likely_nH2p_ne_value,most_likely_nH_ne_value,color='k',marker='$X$',markersize=10);
										im = ax[plot_index,0].plot(record_nH2p_ne_values[nH2_ne_index,:,ne_index*TS_Te_steps + Te_index][nH2p_nH2_index],record_nH_ne_values[:,ne_index*TS_Te_steps + Te_index][nH_ne_index],color='k',marker='$X$',markersize=30);
										im = ax[plot_index,0].plot(np.meshgrid(record_nH2p_ne_values[nH2_ne_index,:,ne_index*TS_Te_steps + Te_index], record_nH_ne_values[:,ne_index*TS_Te_steps + Te_index])[0],np.meshgrid(record_nH2p_ne_values[nH2_ne_index,:,ne_index*TS_Te_steps + Te_index], record_nH_ne_values[:,ne_index*TS_Te_steps + Te_index])[1],'k,')
										ax[plot_index,0].set_title('PDF at most likely H2, Te, ne, nH-=%.3g\nM=marginalised best, X=unmarginalised best' %(np.min(record_nHm_ne_values[nH2_ne_index,:,ne_index*TS_Te_steps + Te_index])))
										ax[plot_index,0].set_ylabel('nH/ne')
										ax[plot_index,0].set_xlabel('nH2+/ne')
										ax[plot_index,0].set_xscale('log')
										ax[plot_index,0].set_yscale('log')

										index = likelihood_log_probs[:,:,:,0,:,:].argmax()	# H, Hm, H2, H2p, ne, Te
										Te_index = index%TS_Te_steps
										ne_index = (index-Te_index)//TS_Te_steps%TS_ne_steps
										nH2_ne_index = ((index-Te_index)//TS_Te_steps-ne_index)//TS_ne_steps%H2_steps
										nHm_ne_index = (((index-Te_index)//TS_Te_steps-ne_index)//TS_ne_steps-nH2_ne_index)//H2_steps%Hm_steps
										nH_ne_index = ((((index-Te_index)//TS_Te_steps-ne_index)//TS_ne_steps-nH2_ne_index)//H2_steps - nHm_ne_index)//Hm_steps%H_steps
										im = ax[plot_index,1].contourf(record_nHm_ne_values[nH2_ne_index,:,ne_index*TS_Te_steps + Te_index],record_nH_ne_values[:,ne_index*TS_Te_steps + Te_index], np.exp(likelihood_log_probs[:,:,nH2_ne_index,0,ne_index,Te_index]),levels=10, cmap='rainbow');	# H, Hm, H2, H2p, ne, Te
										plt.colorbar(im, ax=ax[plot_index,1])
										im = ax[plot_index,1].contour(record_nHm_ne_values[nH2_ne_index,:,ne_index*TS_Te_steps + Te_index],record_nH_ne_values[:,ne_index*TS_Te_steps + Te_index], np.exp(likelihood_log_probs[:,:,nH2_ne_index,0,ne_index,Te_index]),levels=10);
										# im = ax[plot_index,1].plot(most_likely_marginalised_nHm_ne_value,most_likely_marginalised_nH_ne_value,color='k',marker='$M$',markersize=10);
										im = ax[plot_index,1].plot(most_likely_nHm_ne_value,most_likely_nH_ne_value,color='k',marker='$X$',markersize=10);
										im = ax[plot_index,1].plot(record_nHm_ne_values[nH2_ne_index,:,ne_index*TS_Te_steps + Te_index][nHm_ne_index],record_nH_ne_values[:,ne_index*TS_Te_steps + Te_index][nH_ne_index],color='k',marker='$X$',markersize=30);
										im = ax[plot_index,1].plot(np.meshgrid(record_nHm_ne_values[nH2_ne_index,:,ne_index*TS_Te_steps + Te_index], record_nH_ne_values[:,ne_index*TS_Te_steps + Te_index])[0],np.meshgrid(record_nHm_ne_values[nH2_ne_index,:,ne_index*TS_Te_steps + Te_index], record_nH_ne_values[:,ne_index*TS_Te_steps + Te_index])[1],'k,')
										ax[plot_index,1].set_title('PDF at most likely H2, Te, ne, nH2+=%.3g' %(np.min(record_nH2p_ne_values[nH2_ne_index,:,ne_index*TS_Te_steps + Te_index])))
										ax[plot_index,1].set_ylabel('nH/ne')
										ax[plot_index,1].set_xlabel('nH-/ne')
										ax[plot_index,1].set_xscale('log')
										ax[plot_index,1].set_yscale('log')

										plot_index += 1
										ax[plot_index,0].remove()
										ax[plot_index,0]=fig.add_subplot(4,2,3,projection='3d')
										# nH_ne_sample = np.logspace(np.log10(np.min(nH_ne_values)),np.log10(np.max(nH_ne_values)),num=1000)
										# nH2p_ne_sample = np.logspace(np.log10(np.min(nH2p_ne_values)),np.log10(np.max(nH2p_ne_values)),num=10000)
										nH_ne_values_full,nH2p_ne_values_full = np.meshgrid(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],indexing='ij')
										nHm_ne_values_full = np.zeros_like(nH_ne_values_full)
										# specific_prob = np.zeros_like(nH_ne_values_full)
										temp = np.exp(likelihood_log_probs[:,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index])
										temp[temp<1e-100]=1e-100
										for iH,nH_ne in enumerate(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index]):
											for i2p,nH2p_ne in enumerate(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index]):
												calc = line_3D(nH_ne,nH2p_ne,*sol_line.x)
												im = np.abs(record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index]-calc).argmin()
												nHm_ne_values_full[iH,i2p] = record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][im]
										# 		nH_ne_values_full[iH,i2p] = nH_ne_values[np.abs(nH_ne_values-nH_ne).argmin()]
										# 		nH2p_ne_values_full[iH,i2p] = nH2p_ne_values[np.abs(nH2p_ne_values-nH2p_ne).argmin()]
										# 		specific_prob[iH,i2p]=temp[np.abs(nH_ne_values-nH_ne).argmin(),im,np.abs(nH2p_ne_values-nH2p_ne).argmin()]
										select = temp>=np.exp(max_likelihood_log_prob)*0.9
										# sample=np.array([nH_ne_values_full.flatten(),nHm_ne_values_full.flatten(),nH2p_ne_values_full.flatten(),specific_prob.flatten()]).T
										# reduced_sample = sample[sample[:,3]>=np.exp(max_likelihood_log_prob)*0.9]
										# temp_reduced_sample=np.array([[0,0,0,0]])
										# for i in range(1,len(reduced_sample)-1):
										# 	if np.sum(np.logical_and(np.logical_and(temp_reduced_sample[:,0]==reduced_sample[i,0],temp_reduced_sample[:,1]==reduced_sample[i,1]),temp_reduced_sample[:,2]==reduced_sample[i,2]))==0:
										# 		temp_reduced_sample = np.concatenate((temp_reduced_sample,[reduced_sample[i]]))
										# reduced_sample = temp_reduced_sample[1:]
										# specific_prob = reduced_sample[3]
										# my_col = cm.rainbow(specific_prob/np.max(specific_prob)*255,alpha=0.5)
										# scatter = ax[plot_index,0].scatter(np.log10(reduced_sample[:,2]),np.log10(reduced_sample[:,1]),np.log10(reduced_sample[:,0]), c=reduced_sample[:,3], cmap='rainbow', zorder = 0.6)
										# surf = ax[plot_index,0].plot_surface(np.log10(nH2p_ne_values_full),np.log10(nHm_ne_values_full),np.log10(nH_ne_values_full), facecolors=my_col,linewidth=1,rstride=1,cstride=1, antialiased=False,alpha=0.5, zorder = 0.1)
										ax[plot_index,0].plot_wireframe(np.log10(nH2p_ne_values_full),np.log10(nHm_ne_values_full),np.log10(nH_ne_values_full),linewidth=0.5,rcount=15,ccount=15,color='y', zorder = 0.2)
										# ax[plot_index,0].plot(np.log10(reduced_sample[:,2]),np.log10(reduced_sample[:,1]),np.log10(reduced_sample[:,0]),'go', zorder = 0.6)
										nH_ne_values_full_temp,nHm_ne_values_full_temp,nH2p_ne_values_full_temp = np.meshgrid(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],indexing='ij')
										scatter = ax[plot_index,0].scatter(np.log10(nH2p_ne_values_full_temp)[select],np.log10(nHm_ne_values_full_temp)[select],np.log10(nH_ne_values_full_temp)[select], c=temp[select], cmap='rainbow', zorder = 0.6)
										ax[plot_index,0].plot(np.log10(nH2p_ne_values_full_temp)[temp>=np.exp(max_likelihood_log_prob)*0.8],np.log10(nHm_ne_values_full_temp)[temp>=np.exp(max_likelihood_log_prob)*0.8],np.log10(nH_ne_values_full_temp)[temp>=np.exp(max_likelihood_log_prob)*0.8],'k+', zorder = 0.5)
										# ax[plot_index,0].plot([np.log10(most_likely_marginalised_nH2p_ne_value)],[np.log10(most_likely_marginalised_nHm_ne_value)],[np.log10(most_likely_marginalised_nH_ne_value)],color='r',marker='$M$',markersize=30, zorder = 0.7)
										ax[plot_index,0].plot([np.log10(most_likely_nH2p_ne_value)],[np.log10(most_likely_nHm_ne_value)],[np.log10(most_likely_nH_ne_value)],color='r',marker='$X$',markersize=30, zorder = 0.8)
										# ax[plot_index,0].plot([np.log10(first_most_likely_marginalised_nH2p_ne_value)],[np.log10(first_most_likely_marginalised_nHm_ne_value)],[np.log10(first_most_likely_marginalised_nH_ne_value)],color='y',marker='$M$',markersize=15,zorder = 0.7)
										ax[plot_index,0].plot([np.log10(first_most_likely_nH2p_ne_value)],[np.log10(first_most_likely_nHm_ne_value)],[np.log10(first_most_likely_nH_ne_value)],color='y',marker='$X$',markersize=15, zorder = 0.8)
										ax[plot_index,0].view_init(elev=60,azim=45)
										ax[plot_index,0].set_title('Best 20 and 10'+'%'+' of likelihood at most likely H2, Te, ne\nnH=%.3gnH2+ +%.3gnH- +%.3g, R2=%.3g' %(sol_line.x[0],sol_line.x[1],sol_line.x[2],R2_sol_line))
										ax[plot_index,0].set_xlabel('Log10 nH2+_ne')
										ax[plot_index,0].set_ylabel('Log10 nH-_ne')
										ax[plot_index,0].set_zlabel('Log10 nH_ne')

										ax[plot_index,1].remove()
										ax[plot_index,1]=fig.add_subplot(4,2,4,projection='3d')
										ax[plot_index,1].plot_wireframe(np.log10(nH2p_ne_values_full),np.log10(nHm_ne_values_full),np.log10(nH_ne_values_full),linewidth=0.5,rcount=15,ccount=15,color='y', zorder = 0.2)
										# scatter = ax[plot_index,1].scatter(np.log10(reduced_sample[:,2]),np.log10(reduced_sample[:,1]),np.log10(reduced_sample[:,0]), c=power_rad_excit, cmap='rainbow', zorder = 0.4)
										scatter = ax[plot_index,1].scatter(np.log10(nH2p_ne_values_full_temp[select]),np.log10(nHm_ne_values_full_temp[select]),np.log10(nH_ne_values_full_temp[select]), c=power_rad_excit[:,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index][select], cmap='rainbow', zorder = 0.4)
										plt.colorbar(scatter, ax=ax[plot_index,1]).set_label('Power [W]')
										# ax[plot_index,1].plot([np.log10(most_likely_marginalised_nH2p_ne_value)],[np.log10(most_likely_marginalised_nHm_ne_value)],[np.log10(most_likely_marginalised_nH_ne_value)],color='r',marker='$M$',markersize=30, zorder = 0.7)
										ax[plot_index,1].plot([np.log10(most_likely_nH2p_ne_value)],[np.log10(most_likely_nHm_ne_value)],[np.log10(most_likely_nH_ne_value)],color='r',marker='$X$',markersize=30, zorder = 0.8)
										# ax[plot_index,1].plot([np.log10(first_most_likely_marginalised_nH2p_ne_value)],[np.log10(first_most_likely_marginalised_nHm_ne_value)],[np.log10(first_most_likely_marginalised_nH_ne_value)],color='y',marker='$M$',markersize=15,zorder = 0.7)
										ax[plot_index,1].plot([np.log10(first_most_likely_nH2p_ne_value)],[np.log10(first_most_likely_nHm_ne_value)],[np.log10(first_most_likely_nH_ne_value)],color='y',marker='$X$',markersize=15, zorder = 0.8)
										ax[plot_index,1].view_init(elev=60,azim=45)
										ax[plot_index,1].set_title('Radiative power sink due to H excitation')
										ax[plot_index,1].set_xlabel('Log10 nH2+_ne')
										ax[plot_index,1].set_ylabel('Log10 nH-_ne')
										ax[plot_index,1].set_zlabel('Log10 nH_ne')

										plot_index += 1
										ax[plot_index,0].remove()
										ax[plot_index,0]=fig.add_subplot(4,2,5,projection='3d')
										ax[plot_index,0].plot_wireframe(np.log10(nH2p_ne_values_full),np.log10(nHm_ne_values_full),np.log10(nH_ne_values_full),linewidth=0.5,rcount=15,ccount=15,color='y', zorder = 0.2)
										scatter = ax[plot_index,0].scatter(np.log10(nH2p_ne_values_full_temp[select]),np.log10(nHm_ne_values_full_temp[select]),np.log10(nH_ne_values_full_temp[select]), c=power_rad_rec_bremm[:,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index][select], cmap='rainbow', zorder = 0.4)
										plt.colorbar(scatter, ax=ax[plot_index,0]).set_label('Power [W]')
										# ax[plot_index,0].plot([np.log10(most_likely_marginalised_nH2p_ne_value)],[np.log10(most_likely_marginalised_nHm_ne_value)],[np.log10(most_likely_marginalised_nH_ne_value)],color='r',marker='$M$',markersize=30, zorder = 0.7)
										ax[plot_index,0].plot([np.log10(most_likely_nH2p_ne_value)],[np.log10(most_likely_nHm_ne_value)],[np.log10(most_likely_nH_ne_value)],color='r',marker='$X$',markersize=30, zorder = 0.8)
										# ax[plot_index,0].plot([np.log10(first_most_likely_marginalised_nH2p_ne_value)],[np.log10(first_most_likely_marginalised_nHm_ne_value)],[np.log10(first_most_likely_marginalised_nH_ne_value)],color='y',marker='$M$',markersize=15,zorder = 0.7)
										ax[plot_index,0].plot([np.log10(first_most_likely_nH2p_ne_value)],[np.log10(first_most_likely_nHm_ne_value)],[np.log10(first_most_likely_nH_ne_value)],color='y',marker='$X$',markersize=15, zorder = 0.8)
										ax[plot_index,0].view_init(elev=60,azim=45)
										ax[plot_index,0].set_title('Radiative power sink due to recombination and Bremsstrahlung')
										ax[plot_index,0].set_xlabel('Log10 nH2+_ne')
										ax[plot_index,0].set_ylabel('Log10 nH-_ne')
										ax[plot_index,0].set_zlabel('Log10 nH_ne')

										ax[plot_index,1].remove()
										ax[plot_index,1]=fig.add_subplot(4,2,6,projection='3d')
										ax[plot_index,1].plot_wireframe(np.log10(nH2p_ne_values_full),np.log10(nHm_ne_values_full),np.log10(nH_ne_values_full),linewidth=0.5,rcount=15,ccount=15,color='y', zorder = 0.2)
										scatter = ax[plot_index,1].scatter(np.log10(nH2p_ne_values_full_temp[select]),np.log10(nHm_ne_values_full_temp[select]),np.log10(nH_ne_values_full_temp[select]), c=power_rad_mol[:,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index][select], cmap='rainbow', zorder = 0.4)
										plt.colorbar(scatter, ax=ax[plot_index,1]).set_label('Power [W]')
										# ax[plot_index,1].plot([np.log10(most_likely_marginalised_nH2p_ne_value)],[np.log10(most_likely_marginalised_nHm_ne_value)],[np.log10(most_likely_marginalised_nH_ne_value)],color='r',marker='$M$',markersize=30, zorder = 0.7)
										ax[plot_index,1].plot([np.log10(most_likely_nH2p_ne_value)],[np.log10(most_likely_nHm_ne_value)],[np.log10(most_likely_nH_ne_value)],color='r',marker='$X$',markersize=30, zorder = 0.8)
										# ax[plot_index,1].plot([np.log10(first_most_likely_marginalised_nH2p_ne_value)],[np.log10(first_most_likely_marginalised_nHm_ne_value)],[np.log10(first_most_likely_marginalised_nH_ne_value)],color='y',marker='$M$',markersize=15,zorder = 0.7)
										ax[plot_index,1].plot([np.log10(first_most_likely_nH2p_ne_value)],[np.log10(first_most_likely_nHm_ne_value)],[np.log10(first_most_likely_nH_ne_value)],color='y',marker='$X$',markersize=15, zorder = 0.8)
										ax[plot_index,1].view_init(elev=60,azim=45)
										ax[plot_index,1].set_title('Radiative power sink due molecule induced H de-excitation')
										ax[plot_index,1].set_xlabel('Log10 nH2+_ne')
										ax[plot_index,1].set_ylabel('Log10 nH-_ne')
										ax[plot_index,1].set_zlabel('Log10 nH_ne')

										plot_index += 1
										ax[plot_index,0].remove()
										ax[plot_index,0]=fig.add_subplot(4,2,7,projection='3d')
										ax[plot_index,0].plot_wireframe(np.log10(nH2p_ne_values_full),np.log10(nHm_ne_values_full),np.log10(nH_ne_values_full),linewidth=0.5,rcount=15,ccount=15,color='y', zorder = 0.2)
										scatter = ax[plot_index,0].scatter(np.log10(nH2p_ne_values_full_temp[select]),np.log10(nHm_ne_values_full_temp[select]),np.log10(nH_ne_values_full_temp[select]), c=power_via_ionisation[:,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index][select], cmap='rainbow', zorder = 0.4)
										plt.colorbar(scatter, ax=ax[plot_index,0]).set_label('Power [W]')
										# ax[plot_index,0].plot([np.log10(most_likely_marginalised_nH2p_ne_value)],[np.log10(most_likely_marginalised_nHm_ne_value)],[np.log10(most_likely_marginalised_nH_ne_value)],color='r',marker='$M$',markersize=30, zorder = 0.7)
										ax[plot_index,0].plot([np.log10(most_likely_nH2p_ne_value)],[np.log10(most_likely_nHm_ne_value)],[np.log10(most_likely_nH_ne_value)],color='r',marker='$X$',markersize=30, zorder = 0.8)
										# ax[plot_index,0].plot([np.log10(first_most_likely_marginalised_nH2p_ne_value)],[np.log10(first_most_likely_marginalised_nHm_ne_value)],[np.log10(first_most_likely_marginalised_nH_ne_value)],color='y',marker='$M$',markersize=15,zorder = 0.7)
										ax[plot_index,0].plot([np.log10(first_most_likely_nH2p_ne_value)],[np.log10(first_most_likely_nHm_ne_value)],[np.log10(first_most_likely_nH_ne_value)],color='y',marker='$X$',markersize=15, zorder = 0.8)
										ax[plot_index,0].view_init(elev=60,azim=45)
										ax[plot_index,0].set_title('Power sink due to atomic ionisation')
										ax[plot_index,0].set_xlabel('Log10 nH2+_ne')
										ax[plot_index,0].set_ylabel('Log10 nH-_ne')
										ax[plot_index,0].set_zlabel('Log10 nH_ne')

										ax[plot_index,1].remove()
										ax[plot_index,1]=fig.add_subplot(4,2,8,projection='3d')
										ax[plot_index,1].plot_wireframe(np.log10(nH2p_ne_values_full),np.log10(nHm_ne_values_full),np.log10(nH_ne_values_full),linewidth=0.5,rcount=15,ccount=15,color='y', zorder = 0.2)
										scatter = ax[plot_index,1].scatter(np.log10(nH2p_ne_values_full_temp[select]),np.log10(nHm_ne_values_full_temp[select]),np.log10(nH_ne_values_full_temp[select]), c=power_via_recombination[:,:,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index][select], cmap='rainbow', zorder = 0.4)
										plt.colorbar(scatter, ax=ax[plot_index,1]).set_label('Power [W]')
										# ax[plot_index,1].plot([np.log10(most_likely_marginalised_nH2p_ne_value)],[np.log10(most_likely_marginalised_nHm_ne_value)],[np.log10(most_likely_marginalised_nH_ne_value)],color='r',marker='$M$',markersize=30, zorder = 0.7)
										ax[plot_index,1].plot([np.log10(most_likely_nH2p_ne_value)],[np.log10(most_likely_nHm_ne_value)],[np.log10(most_likely_nH_ne_value)],color='r',marker='$X$',markersize=30, zorder = 0.8)
										# ax[plot_index,1].plot([np.log10(first_most_likely_marginalised_nH2p_ne_value)],[np.log10(first_most_likely_marginalised_nHm_ne_value)],[np.log10(first_most_likely_marginalised_nH_ne_value)],color='y',marker='$M$',markersize=15,zorder = 0.7)
										ax[plot_index,1].plot([np.log10(first_most_likely_nH2p_ne_value)],[np.log10(first_most_likely_nHm_ne_value)],[np.log10(first_most_likely_nH_ne_value)],color='y',marker='$X$',markersize=15, zorder = 0.8)
										ax[plot_index,1].view_init(elev=60,azim=45)
										ax[plot_index,1].set_title('Power sink due to atomic recombination')
										ax[plot_index,1].set_xlabel('Log10 nH2+_ne')
										ax[plot_index,1].set_ylabel('Log10 nH-_ne')
										ax[plot_index,1].set_zlabel('Log10 nH_ne')


										save_done = 0
										save_index=1
										try:	# All this trie are because of a known issues when saving lots of figure inside of multiprocessor
											plt.savefig(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_second.eps', bbox_inches='tight')
										except Exception as e:
											print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_second.eps save try number '+str(1)+' failed. Reason %s' % e)
											while save_done==0 and save_index<100:
												try:
													plt.savefig(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_second.eps', bbox_inches='tight')
													print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_adaptable_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_second.eps save try number '+str(save_index+1)+' successfull')
													save_done=1
												except Exception as e:
													tm.sleep(np.random.random()**2)
													# print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(save_index+1)+' failed. Reason %s' % e)
													save_index+=1
										plt.close('all')


							if my_time_pos in sample_time_step:
								if my_r_pos in sample_radious:
									if to_print:
										# tm.sleep(np.random.random()*10)
										fig, ax = plt.subplots( 4,2,figsize=(25, 40), squeeze=False)
										fig.suptitle('first scan\nmost_likely_nH_ne_value %.3g,most_likely_nHm_ne_value %.3g,most_likely_nH2_ne_value %.3g,most_likely_nH2p_ne_value %.3g' %(most_likely_nH_ne_value*most_likely_ne_value,most_likely_nHm_ne_value*most_likely_ne_value,most_likely_nH2_ne_value*most_likely_ne_value,most_likely_nH2p_ne_value*most_likely_ne_value) +'\nlines '+str(n_list_all)+'\nlocation [time, r]'+ ' [%.3g ms, ' % time_crop[my_time_pos] + ' %.3g mm]' % (1000*r_crop[my_r_pos]) +' , [TS Te, TS ne] '+ ' [%.3g(ML %.3g) eV, ' %(merge_Te_prof_multipulse_interp_crop_limited_restrict,most_likely_Te_value) + '%.3g(ML %.3g) #10^20/m^3]' %(merge_ne_prof_multipulse_interp_crop_limited_restrict,most_likely_ne_value*1e-20) +'\nfit [nH/ne, nH+/ne, nH-/ne, nH2/ne, nH2+/ne, nH3+/ne] , '+ '[%.3g, ' % most_likely_nH_ne_value + '%.3g, ' %(1 - most_likely_nH2p_ne_value + most_likely_nHm_ne_value)+ '%.3g, ' % most_likely_nHm_ne_value+ '%.3g, ' % most_likely_nH2_ne_value+ '%.3g, ' % most_likely_nH2p_ne_value+ '%.3g]' % 0+'\nBest index: [nH/ne,nH-/ne,nH2/ne,nH2+/ne,ne,Te] [%.3g,%.3g,%.3g,%.3g,%.3g,%.3g]' %(most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index)+'\nMarginalised best index: [nH/ne,nH-/ne,nH2+/ne] [%.3g,%.3g,%.3g]' %(most_likely_marginalised_nH_ne_index,most_likely_marginalised_nHm_nH2_index,most_likely_marginalised_nH2p_nH2_index))
										plot_index = 0
										nH_ne_values_full,nHm_ne_values_full,nH2_ne_values_full,nH2p_ne_values_full,ne_values_full,Te_values_full = np.meshgrid(np.arange(H_steps),np.arange(Hm_steps),np.arange(H2_steps),np.arange(H2p_steps),ne_values_array,Te_values_array,indexing='ij')	# H, Hm, H2, H2p, ne, Te
										fraction_of_max_prob_selected = 50000
										select = likelihood_probs_times_volume>likelihood_probs_times_volume.max()/fraction_of_max_prob_selected
										while np.sum(select)/10>50000:
											fraction_of_max_prob_selected*=0.9
											select = likelihood_probs_times_volume>likelihood_probs_times_volume.max()/fraction_of_max_prob_selected
										select = np.logical_and(likelihood_probs_times_volume>likelihood_probs_times_volume.max()/fraction_of_max_prob_selected,np.random.random(np.shape(select))<0.1)
										temp_prob = likelihood_probs_times_volume[select]
										max_temp_prob = np.max(temp_prob)
										nH_ne_values_full = nH_ne_values_full[select]
										nHm_ne_values_full = nHm_ne_values_full[select]
										nH2_ne_values_full = nH2_ne_values_full[select]
										nH2p_ne_values_full = nH2p_ne_values_full[select]
										Te_values_full = Te_values_full[select]
										ne_values_full = ne_values_full[select]
										if include_particles_limitation:
											all_net_Hp_destruction_full = all_net_Hp_destruction[select]
											all_net_e_destruction_full = all_net_e_destruction[select]
											all_net_Hm_destruction_full = all_net_Hm_destruction[select]
											all_net_H2p_destruction_full = all_net_H2p_destruction[select]
										T_Hm_values_full = np.exp(TH2_fit_from_simulations(np.log(Te_values_full.flatten())))/eV_to_K	# K
										T_H2p_values_full = np.exp(TH2_fit_from_simulations(np.log(Te_values_full.flatten())))/eV_to_K	# K
										T_Hp_values_full = Te_values_full.flatten()/eV_to_K	# K
										# power_rad_excit,power_rad_rec_bremm,power_rad_mol,power_via_ionisation,power_via_recombination = calc_power_balance_elements(Te_values_full,ne_values_full,nH_ne_values_full,nHm_ne_values_full,nH2p_ne_values_full,nH2_ne_values_full,T_Hp_values_full,T_Hm_values_full,T_H2p_values_full)
										# most_likely_power_rad_excit,most_likely_power_rad_rec_bremm,most_likely_power_rad_mol,most_likely_power_via_ionisation,most_likely_power_via_recombination,most_likely_power_rad_Hm,most_likely_power_rad_H2,most_likely_power_rad_H2p = calc_power_balance_elements(most_likely_Te_value,most_likely_ne_value,most_likely_nH_ne_value,most_likely_nHm_ne_value,most_likely_nH2p_ne_value,most_likely_nH2_ne_value,most_likely_T_Hp_value,most_likely_T_Hm_value,most_likely_T_H2p_value)
										most_likely_power_rad_excit = power_rad_excit.flatten()[index_most_likely]
										most_likely_power_rad_rec_bremm = power_rad_rec_bremm.flatten()[index_most_likely]
										most_likely_power_rad_mol = power_rad_mol.flatten()[index_most_likely]
										most_likely_power_via_ionisation = power_via_ionisation.flatten()[index_most_likely]
										most_likely_power_via_recombination = power_via_recombination.flatten()[index_most_likely]
										most_likely_power_rad_Hm = power_rad_Hm.flatten()[index_most_likely]
										most_likely_power_rad_H2 = power_rad_H2.flatten()[index_most_likely]
										most_likely_power_rad_H2p = power_rad_H2p.flatten()[index_most_likely]
										most_likely_tot_rad_power = tot_rad_power.flatten()[index_most_likely]
										most_likely_power_via_brem = power_via_brem.flatten()[index_most_likely]
										most_likely_power_heating_rec = power_heating_rec.flatten()[index_most_likely]
										most_likely_power_rec_neutral = power_rec_neutral.flatten()[index_most_likely]
										most_likely_total_removed_power = total_removed_power.flatten()[index_most_likely]
										# most_likely_tot_rad_power = most_likely_power_rad_excit+most_likely_power_rad_rec_bremm+most_likely_power_rad_mol
										# most_likely_power_via_brem =  5.35*1e-37 * ( most_likely_ne_value**2 ) * (1 - most_likely_nH2p_ne_value + most_likely_nHm_ne_value) * (most_likely_Te_value/1000)**0.5
										# most_likely_power_heating_rec = most_likely_power_via_recombination-most_likely_power_rad_rec_bremm + most_likely_power_via_brem
										# most_likely_power_rec_neutral = most_likely_power_rad_rec_bremm + 3/2*most_likely_power_via_recombination/13.6*most_likely_Te_value
										# most_likely_total_removed_power = most_likely_power_via_ionisation + most_likely_power_rad_mol + most_likely_power_rad_excit + most_likely_power_via_recombination + most_likely_power_via_brem + most_likely_power_rec_neutral
										color = ['b', 'r', 'm', 'y', 'g', 'c', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise', 'teal', 'olive']
										im = ax[plot_index,0].plot(power_rad_excit[select],100*temp_prob/max_temp_prob,color[0]+'+',label='power_rad_excit');
										im = ax[plot_index,0].plot(most_likely_power_rad_excit,100,color[0]+'o',markersize=30,fillstyle='none');
										im = ax[plot_index,0].plot(power_rad_rec_bremm[select],100*temp_prob/max_temp_prob,color[1]+'+',label='power_rad_rec_bremm');
										im = ax[plot_index,0].plot(most_likely_power_rad_rec_bremm,100,color[1]+'o',markersize=30,fillstyle='none');
										im = ax[plot_index,0].plot(power_rad_mol[select],100*temp_prob/max_temp_prob,color[2]+'+',label='power_rad_mol');
										im = ax[plot_index,0].plot(most_likely_power_rad_mol,100,color[2]+'o',markersize=30,fillstyle='none');
										im = ax[plot_index,0].plot(power_via_ionisation[select],100*temp_prob/max_temp_prob,color[3]+'+',label='power_via_ionisation');
										im = ax[plot_index,0].plot(most_likely_power_via_ionisation,100,color[3]+'o',markersize=30,fillstyle='none');
										im = ax[plot_index,0].plot(power_via_recombination[select],100*temp_prob/max_temp_prob,color[4]+'+',label='power_via_recombination');
										im = ax[plot_index,0].plot(most_likely_power_via_recombination,100,color[4]+'o',markersize=30,fillstyle='none');
										im = ax[plot_index,0].plot(tot_rad_power[select],100*temp_prob/max_temp_prob,color[5]+'+',label='tot_rad_power');
										im = ax[plot_index,0].plot(most_likely_tot_rad_power,100,color[5]+'o',markersize=30,fillstyle='none');
										im = ax[plot_index,0].plot(power_rad_Hm[select],100*temp_prob/max_temp_prob,'+',color=color[6],label='power_rad_Hm');
										im = ax[plot_index,0].plot(most_likely_power_rad_Hm,100,'o',color=color[6],markersize=30,fillstyle='none');
										im = ax[plot_index,0].plot(power_rad_H2[select],100*temp_prob/max_temp_prob,'+',color=color[7],label='power_rad_H2');
										im = ax[plot_index,0].plot(most_likely_power_rad_H2,100,'o',color=color[7],markersize=30,fillstyle='none');
										im = ax[plot_index,0].plot(power_rad_H2p[select],100*temp_prob/max_temp_prob,'+',color=color[8],label='power_rad_H2p');
										im = ax[plot_index,0].plot(most_likely_power_rad_H2p,100,'o',color=color[8],markersize=30,fillstyle='none');
										im = ax[plot_index,0].plot(power_heating_rec[select],100*temp_prob/max_temp_prob,'+',color=color[9],label='power_heating_rec');
										im = ax[plot_index,0].plot(most_likely_power_heating_rec,100,'o',color=color[9],markersize=30,fillstyle='none');
										im = ax[plot_index,0].plot(power_rec_neutral[select],100*temp_prob/max_temp_prob,'+',color=color[10],label='power_rec_neutral');
										im = ax[plot_index,0].plot(most_likely_power_rec_neutral,100,'o',color=color[10],markersize=30,fillstyle='none');
										im = ax[plot_index,0].plot(power_via_brem[select],100*temp_prob/max_temp_prob,'+',color=color[11],label='power_via_brem');
										im = ax[plot_index,0].plot(most_likely_power_via_brem,100,'o',color=color[11],markersize=30,fillstyle='none');
										im = ax[plot_index,0].plot(total_removed_power[select],100*temp_prob/max_temp_prob,'+',color=color[12],label='total_removed_power');
										im = ax[plot_index,0].plot(most_likely_total_removed_power,100,'o',color=color[12],markersize=30,fillstyle='none');
										# im = ax[plot_index,0].plot([interpolated_power_pulse_shape(time_crop[my_time_pos])/source_power_spread/area/length]*2,[0,100],'k--');
										im = ax[plot_index,0].plot(local_CX[select],100*temp_prob/max_temp_prob,'+',color=color[13],label='local_CX');
										im = ax[plot_index,0].plot(most_likely_local_CX,100,'o',color=color[13],markersize=30,fillstyle='none');
										im = ax[plot_index,0].plot([most_likely_total_removable_power_times_volume]*2,[0,100],'k--');
										im = ax[plot_index,0].plot([max_total_removable_power_times_volume]*2,[0,100],'--',color='gray');
										im = ax[plot_index,0].plot([min_total_removable_power_times_volume]*2,[0,100],'--',color='gray');
										ax[plot_index,0].set_title('Power balance for likelyhood above %.3g' %(100/fraction_of_max_prob_selected) +'%'+' of of the PDF peak, O=ML' + '\nsigma: molecular %.3g%%, atomic %.3g%%, budget %.3g%%' %(power_molecular_precision*100,power_atomic_precision*100,power_budget_precision*100))
										ax[plot_index,0].set_ylabel('normalised likelihood')
										ax[plot_index,0].set_xlabel('Power [W/m3]')
										ax[plot_index,0].set_xscale('log')
										ax[plot_index,0].set_xlim(left=max(1,1e-1*np.min([most_likely_power_rad_excit,most_likely_power_rad_rec_bremm,most_likely_power_rad_mol,most_likely_power_via_ionisation,most_likely_power_via_recombination,most_likely_tot_rad_power,most_likely_power_heating_rec,most_likely_power_via_brem,most_likely_total_removed_power,most_likely_power_rec_neutral])))
										ax[plot_index,0].legend(loc='best', fontsize='x-small')
										ax[plot_index,0].grid()

										im = ax[plot_index,1].plot(Te_values_full,100*temp_prob/max_temp_prob,'+');
										im = ax[plot_index,1].plot(most_likely_Te_value,100,'o',markersize=30,fillstyle='none');
										im = ax[plot_index,1].plot(merge_Te_prof_multipulse_interp_crop_limited_restrict,100,'x',markersize=30);
										ax[plot_index,1].set_title('Te for likelyhood above %.3g' %(100/fraction_of_max_prob_selected) +'%'+' of of the PDF peak, O=ML, X=TS')
										ax[plot_index,1].set_ylabel('normalised likelihood')
										ax[plot_index,1].set_xlabel('temperature [eV]')
										# ax[plot_index,0].set_xscale('log')
										# ax[plot_index,1].legend(loc='best', fontsize='x-small')
										ax[plot_index,1].grid()

										plot_index += 1
										im = ax[plot_index,0].plot(nH_ne_values_full,100*temp_prob/max_temp_prob,color[0]+'+',label='nH_ne');
										im = ax[plot_index,0].plot(record_nH_ne_log_prob[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index].argmax(),100,color[0]+'x',markersize=30);
										im = ax[plot_index,0].plot(most_likely_nH_ne_index,100,color[0]+'o',markersize=30,fillstyle='none');
										im = ax[plot_index,0].plot(nHm_ne_values_full,100*temp_prob/max_temp_prob,color[1]+'+',label='nHm_nH2');
										im = ax[plot_index,0].plot(most_likely_nHm_nH2_index,100,color[1]+'o',markersize=30,fillstyle='none');
										im = ax[plot_index,0].plot(nH2_ne_values_full,100*temp_prob/max_temp_prob,color[2]+'+',label='nH2_ne');
										im = ax[plot_index,0].plot(record_nH_ne_log_prob[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index].argmax(),100,color[2]+'x',markersize=30);
										im = ax[plot_index,0].plot(most_likely_nH2_ne_index,100,color[2]+'o',markersize=30,fillstyle='none');
										im = ax[plot_index,0].plot(nH2p_ne_values_full,100*temp_prob/max_temp_prob,color[3]+'+',label='nH2p_nH2');
										im = ax[plot_index,0].plot(most_likely_nH2p_nH2_index,100,color[3]+'o',markersize=30,fillstyle='none');
										ax[plot_index,0].set_title('Relative densities for likelyhood above %.3g' %(100/fraction_of_max_prob_selected) +'%'+' of of the PDF peak, O=ML, X=central')
										ax[plot_index,0].set_ylabel('normalised likelihood')
										ax[plot_index,0].set_xlabel('index')
										# ax[plot_index,0].set_xscale('log')
										ax[plot_index,0].legend(loc='best', fontsize='x-small')
										ax[plot_index,0].grid()

										im = ax[plot_index,1].plot(ne_values_full,100*temp_prob/max_temp_prob,'+');
										im = ax[plot_index,1].plot(most_likely_ne_value,100,'o',markersize=30,fillstyle='none');
										im = ax[plot_index,1].plot(merge_ne_prof_multipulse_interp_crop_limited_restrict*1e20,100,'x',markersize=30);
										ax[plot_index,1].set_title('ne for likelyhood above %.3g' %(100/fraction_of_max_prob_selected) +'%'+' of of the PDF peak, O=ML, X=TS')
										ax[plot_index,1].set_ylabel('normalised likelihood')
										ax[plot_index,1].set_xlabel('density [#/m3]')
										# ax[plot_index,0].set_yscale('log')
										# ax[plot_index,1].legend(loc='best', fontsize='x-small')
										ax[plot_index,1].grid()

										if include_particles_limitation:
											plot_index += 1
											im = ax[plot_index,0].plot(all_net_e_destruction_full,100*temp_prob/max_temp_prob,color[0]+'+');
											im = ax[plot_index,0].plot(most_likely_net_e_destruction,100,color[0]+'o',markersize=30,fillstyle='none');
											im = ax[plot_index,0].plot([most_likely_total_removable_e]*2,[0,100],'k--');
											im = ax[plot_index,0].plot([total_removable_e[non_zero_prob_select].max()]*2,[0,100],'--',color='gray');
											im = ax[plot_index,0].plot([total_removable_e[non_zero_prob_select].min()]*2,[0,100],'--',color='gray');
											ax[plot_index,0].set_title('electrons consumed within time and volume step')
											ax[plot_index,0].set_ylabel('normalised likelihood')
											ax[plot_index,0].set_xlabel('particles [#*10^20]')
											# ax[plot_index,0].set_xscale('log')
											# ax[plot_index,0].legend(loc='best', fontsize='x-small')
											ax[plot_index,0].grid()

											im = ax[plot_index,1].plot(all_net_Hp_destruction_full,100*temp_prob/max_temp_prob,color[0]+'+');
											im = ax[plot_index,1].plot(most_likely_net_Hp_destruction,100,color[0]+'o',markersize=30,fillstyle='none');
											im = ax[plot_index,1].plot([most_likely_total_removable_Hp]*2,[0,100],'k--');
											im = ax[plot_index,1].plot([total_removable_Hp[non_zero_prob_select].max()]*2,[0,100],'--',color='gray');
											im = ax[plot_index,1].plot([total_removable_Hp[non_zero_prob_select].min()]*2,[0,100],'--',color='gray');
											ax[plot_index,1].set_title('H+ consumed within time and volume step')
											ax[plot_index,1].set_ylabel('normalised likelihood')
											ax[plot_index,1].set_xlabel('particles [#*10^20]')
											ax[plot_index,1].grid()

											plot_index += 1
											im = ax[plot_index,0].plot(all_net_Hm_destruction_full,100*temp_prob/max_temp_prob,color[0]+'+');
											im = ax[plot_index,0].plot(most_likely_net_Hm_destruction,100,color[0]+'o',markersize=30,fillstyle='none');
											im = ax[plot_index,0].plot([most_likely_total_removable_Hm]*2,[0,100],'k--');
											im = ax[plot_index,0].plot([total_removable_Hm[non_zero_prob_select].max()]*2,[0,100],'--',color='gray');
											im = ax[plot_index,0].plot([total_removable_Hm[non_zero_prob_select].min()]*2,[0,100],'--',color='gray');
											ax[plot_index,0].set_title('H- consumed within time and volume step')
											ax[plot_index,0].set_ylabel('normalised likelihood')
											ax[plot_index,0].set_xlabel('particles [#*10^20]')
											ax[plot_index,0].grid()

											im = ax[plot_index,1].plot(all_net_H2p_destruction_full,100*temp_prob/max_temp_prob,color[0]+'+');
											im = ax[plot_index,1].plot(most_likely_net_H2p_destruction,100,color[0]+'o',markersize=30,fillstyle='none');
											im = ax[plot_index,1].plot([most_likely_total_removable_H2p]*2,[0,100],'k--');
											im = ax[plot_index,1].plot([total_removable_H2p[non_zero_prob_select].max()]*2,[0,100],'--',color='gray');
											im = ax[plot_index,1].plot([total_removable_H2p[non_zero_prob_select].min()]*2,[0,100],'--',color='gray');
											ax[plot_index,1].set_title('H2+ consumed within time and volume step')
											ax[plot_index,1].set_ylabel('normalised likelihood')
											ax[plot_index,1].set_xlabel('particles [#*10^20]')
											ax[plot_index,1].grid()

										save_done = 0
										save_index=1
										try:	# All this trie are because of a known issues when saving lots of figure inside of multiprocessor
											plt.savefig(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_third.eps', bbox_inches='tight')
										except Exception as e:
											print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_third.eps save try number '+str(1)+' failed. Reason %s' % e)
											while save_done==0 and save_index<100:
												try:
													plt.savefig(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_third.eps', bbox_inches='tight')
													print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_third.eps save try number '+str(save_index+1)+' successfull')
													save_done=1
												except Exception as e:
													tm.sleep(np.random.random()**2)
													# print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(save_index+1)+' failed. Reason %s' % e)
													save_index+=1
										plt.close('all')


							# Here I find an approximate value for the measurement precision
							def find_sigma(x,sigma):
								return x/sigma

							# to calculate the integral of the probability I need their width
							if False:	# logarithmic interval scaling
								d_nH_ne_values = np.array([*np.diff(np.log10(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,1]])),*(np.diff(np.log10(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index]))[:-1]/2+np.diff(np.log10(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index]))[1:]/2),*np.diff(np.log10(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[-2,-1]]))])/(np.diff(np.log10(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,-1]]))+np.diff(np.log10(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,1]]))/2+np.diff(np.log10(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[-2,-1]]))/2)
								d_nHm_ne_values = np.array([*np.diff(np.log10(record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,1]])),*(np.diff(np.log10(record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index]))[:-1]/2+np.diff(np.log10(record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index]))[1:]/2),*np.diff(np.log10(record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[-2,-1]]))])/(np.diff(np.log10(record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,-1]]))+np.diff(np.log10(record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,1]]))/2+np.diff(np.log10(record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[-2,-1]]))/2)
								d_nH2_ne_values = np.array([*np.diff(np.log10(record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,1]])),*(np.diff(np.log10(record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index]))[:-1]/2+np.diff(np.log10(record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index]))[1:]/2),*np.diff(np.log10(record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[-2,-1]]))])/(np.diff(np.log10(record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,-1]]))+np.diff(np.log10(record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,1]]))/2+np.diff(np.log10(record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[-2,-1]]))/2)
								d_nH2p_ne_values = np.array([*np.diff(np.log10(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,1]])),*(np.diff(np.log10(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index]))[:-1]/2+np.diff(np.log10(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index]))[1:]/2),*np.diff(np.log10(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[-2,-1]]))])/(np.diff(np.log10(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,-1]]))+np.diff(np.log10(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,1]]))/2+np.diff(np.log10(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[-2,-1]]))/2)
							elif True:	# linear interval scaling
								d_nH_ne_values = np.array([*np.diff(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,1]]),*(np.diff(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[:-1]/2+np.diff(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[1:]/2),*np.diff(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[-2,-1]])])/(np.diff(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,-1]])+np.diff(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,1]])/2+np.diff(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[-2,-1]])/2)
								d_nHm_ne_values = np.array([*np.diff(record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,1]]),*(np.diff(record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[:-1]/2+np.diff(record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[1:]/2),*np.diff(record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[-2,-1]])])/(np.diff(record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,-1]])+np.diff(record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,1]])/2+np.diff(record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[-2,-1]])/2)
								if H2_steps!=1:
									d_nH2_ne_values = np.array([*np.diff(record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,1]]),*(np.diff(record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[:-1]/2+np.diff(record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[1:]/2),*np.diff(record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[-2,-1]])])/(np.diff(record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,-1]])+np.diff(record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,1]])/2+np.diff(record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[-2,-1]])/2)
								d_nH2p_ne_values = np.array([*np.diff(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,1]]),*(np.diff(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[:-1]/2+np.diff(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])[1:]/2),*np.diff(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[-2,-1]])])/(np.diff(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,-1]])+np.diff(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[0,1]])/2+np.diff(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][[-2,-1]])/2)
							d_ne_values_array = np.array([*np.diff(ne_values_array[[0,1]]),*(np.diff(ne_values_array)[:-1]/2+np.diff(ne_values_array)[1:]/2),*np.diff(ne_values_array[[-2,-1]])])/(np.diff(ne_values_array[[0,-1]])+np.diff(ne_values_array[[0,1]])/2+np.diff(ne_values_array[[-2,-1]])/2)
							d_Te_values_array = np.array([*np.diff(Te_values_array[[0,1]]),*(np.diff(Te_values_array)[:-1]/2+np.diff(Te_values_array)[1:]/2),*np.diff(Te_values_array[[-2,-1]])])/(np.diff(Te_values_array[[0,-1]])+np.diff(Te_values_array[[0,1]])/2+np.diff(Te_values_array[[-2,-1]])/2)

							# temp = np.sqrt(-2*(likelihood_log_probs[most_likely_nH_ne_index,most_likely_nHm_ne_index,most_likely_nH2_ne_index,most_likely_nH2p_ne_index,most_likely_ne_index,:]-max_likelihood_log_prob))
							temp_2 = likelihood_log_probs[most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,:]
							temp_2 = np.exp(temp_2+np.log(d_Te_values_array)-np.log(np.sum(np.exp(temp_2)*d_Te_values_array)))
							if False:
								try:
									# sigma_Te_right = curve_fit(find_sigma,np.abs(Te_values[0,most_likely_Te_index:]-most_likely_Te_value),temp[most_likely_Te_index:],p0=[1])[0][0]#,sigma=np.abs(Te_values[0,most_likely_Te_index:]-most_likely_Te_value)/most_likely_Te_value+0.1)[0]
									sigma_Te_right = temp_2[most_likely_Te_index:]
									if np.sum(sigma_Te_right)<0.34:
										sigma_Te_right=0
									else:
										sigma_Te_right = np.abs(np.array([sum(sigma_Te_right[:_+1]) for _ in range(len(sigma_Te_right)) ])-0.34)
										sigma_Te_right = sigma_Te_right[1:].argmin()
										sigma_Te_right = -Te_values_array[most_likely_Te_index]+Te_values_array[most_likely_Te_index+1+sigma_Te_right]
								except:
									sigma_Te_right=0
								try:
									# sigma_Te_left = curve_fit(find_sigma,np.abs(Te_values[0,:most_likely_Te_index+1]-most_likely_Te_value),temp[:most_likely_Te_index+1],p0=[1])[0][0]#,sigma=np.abs(Te_values[0,:most_likely_Te_index+1]-most_likely_Te_value)/most_likely_Te_value+0.1)[0]
									sigma_Te_left = np.flip(temp_2[:most_likely_Te_index+1],axis=0)
									if np.sum(sigma_Te_left)<0.34:
										sigma_Te_left=0
									else:
										sigma_Te_left = np.abs(np.array([sum(sigma_Te_left[:_+1]) for _ in range(len(sigma_Te_left)) ])-0.34)
										sigma_Te_left = sigma_Te_left[1:].argmin()
										sigma_Te_left = Te_values_array[most_likely_Te_index]-Te_values_array[most_likely_Te_index-1-sigma_Te_left]
								except:
									sigma_Te_left=0
							else:
								temp_2 = np.cumsum(temp_2)
								sigma = np.array([Te_values_array[most_likely_Te_index]-Te_values_array[np.abs(temp_2-0.159).argmin()+1],Te_values_array[np.abs(temp_2-1+0.159).argmin()]-Te_values_array[most_likely_Te_index]])
								sigma[sigma<0] = np.array([Te_values_array[most_likely_Te_index]-Te_values_array.min(),Te_values_array.max()-Te_values_array[most_likely_Te_index]])[sigma<0]
								sigma_Te_left,sigma_Te_right = sigma
							sigma_Te = max(sigma_Te_right,sigma_Te_left)
							# temp = np.sqrt(-2*(likelihood_log_probs[most_likely_nH_ne_index,most_likely_nHm_ne_index,most_likely_nH2_ne_index,most_likely_nH2p_ne_index,:,most_likely_Te_index]-max_likelihood_log_prob))
							temp_2 = likelihood_log_probs[most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,:,most_likely_Te_index]
							temp_2 = np.exp(temp_2+np.log(d_ne_values_array)-np.log(np.sum(np.exp(temp_2)*d_ne_values_array)))
							if False:
								try:
									# sigma_ne_right = curve_fit(find_sigma,np.abs(ne_values[most_likely_ne_index:,0]-most_likely_ne_value),temp[most_likely_ne_index:],p0=[1])[0][0]#,sigma=np.abs(ne_values[most_likely_ne_index:,0]-most_likely_ne_value)/most_likely_ne_value+0.1)[0]
									sigma_ne_right = temp_2[most_likely_ne_index:]
									if np.sum(sigma_ne_right)<0.34:
										sigma_ne_right=0
									else:
										sigma_ne_right = np.abs(np.array([sum(sigma_ne_right[:_+1]) for _ in range(len(sigma_ne_right)) ])-0.34)
										sigma_ne_right = sigma_ne_right[1:].argmin()
										sigma_ne_right = -ne_values_array[most_likely_ne_index]+ne_values_array[most_likely_ne_index+1+sigma_ne_right]
								except:
									sigma_ne_right=0
								try:
									# sigma_ne_left = curve_fit(find_sigma,np.abs(ne_values[:most_likely_ne_index+1,0]-most_likely_ne_value),temp[:most_likely_ne_index+1],p0=[1])[0][0]#,sigma=np.abs(ne_values[:most_likely_ne_index+1,0]-most_likely_ne_value)/most_likely_ne_value+0.1)[0]
									sigma_ne_left = np.flip(temp_2[:most_likely_ne_index+1],axis=0)
									if np.sum(sigma_ne_left)<0.34:
										sigma_ne_left=0
									else:
										sigma_ne_left = np.abs(np.array([sum(sigma_ne_left[:_+1]) for _ in range(len(sigma_ne_left)) ])-0.34)
										sigma_ne_left = sigma_ne_left[1:].argmin()
										sigma_ne_left = ne_values_array[most_likely_ne_index]-ne_values_array[most_likely_ne_index-1-sigma_ne_left]
								except:
									sigma_ne_left=0
							else:
								temp_2 = np.cumsum(temp_2)
								sigma = np.array([ne_values_array[most_likely_ne_index]-ne_values_array[np.abs(temp_2-0.159).argmin()+1],ne_values_array[np.abs(temp_2-1+0.159).argmin()]-ne_values_array[most_likely_ne_index]])
								sigma[sigma<0] = np.array([ne_values_array[most_likely_ne_index]-ne_values_array.min(),ne_values_array.max()-ne_values_array[most_likely_ne_index]])[sigma<0]
								sigma_ne_left,sigma_ne_right = sigma
							sigma_ne = max(sigma_ne_right,sigma_ne_left)
							# temp = np.sqrt(-2*(likelihood_log_probs[most_likely_nH_ne_index,most_likely_nHm_ne_index,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]-max_likelihood_log_prob))
							temp_2 = likelihood_log_probs[most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]
							temp_2 = np.exp(temp_2+np.log(d_nH2p_ne_values)-np.log(np.sum(np.exp(temp_2)*d_nH2p_ne_values)))
							if False:
								try:
									# sigma_nH2p_ne_right = curve_fit(find_sigma,np.abs(nH2p_ne_values[most_likely_nH2p_ne_index:]-most_likely_nH2p_ne_value),temp[most_likely_nH2p_ne_index:],p0=[1])[0][0]#,sigma=np.abs(nH2p_ne_values[most_likely_nH2p_ne_index:]-most_likely_nH2p_ne_value)/most_likely_nH2p_ne_value+0.1)[0]
									sigma_nH2p_ne_right = temp_2[most_likely_nH2p_nH2_index:]
									if np.sum(sigma_nH2p_ne_right)<0.34:
										sigma_nH2p_ne_right=0
									else:
										sigma_nH2p_ne_right = np.abs(np.array([sum(sigma_nH2p_ne_right[:_+1]) for _ in range(len(sigma_nH2p_ne_right)) ])-0.34)
										sigma_nH2p_ne_right = sigma_nH2p_ne_right[1:].argmin()
										sigma_nH2p_ne_right = -record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][most_likely_nH2p_nH2_index]+record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][most_likely_nH2p_nH2_index+1+sigma_nH2p_ne_right]
								except:
									sigma_nH2p_ne_right=0
								try:
									# sigma_nH2p_ne_left = curve_fit(find_sigma,np.abs(nH2p_ne_values[:most_likely_nH2p_ne_index+1]-most_likely_nH2p_ne_value),temp[:most_likely_nH2p_ne_index+1],p0=[1])[0][0]#,sigma=np.abs(nH2p_ne_values[:most_likely_nH2p_ne_index+1]-most_likely_nH2p_ne_value)/most_likely_nH2p_ne_value+0.1)[0]
									sigma_nH2p_ne_left = np.flip(temp_2[:most_likely_nH2p_nH2_index+1],axis=0)
									if np.sum(sigma_nH2p_ne_left)<0.34:
										sigma_nH2p_ne_left=0
									else:
										sigma_nH2p_ne_left = np.abs(np.array([sum(sigma_nH2p_ne_left[:_+1]) for _ in range(len(sigma_nH2p_ne_left)) ])-0.34)
										sigma_nH2p_ne_left = sigma_nH2p_ne_left[1:].argmin()
										sigma_nH2p_ne_left = record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][most_likely_nH2p_nH2_index]-record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][most_likely_nH2p_nH2_index-1-sigma_nH2p_ne_left]
								except:
									sigma_nH2p_ne_left=0
							else:
								temp_2 = np.cumsum(temp_2)
								sigma = np.array([most_likely_nH2p_ne_value-record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][np.abs(temp_2-0.159).argmin()+1],record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][np.abs(temp_2-1+0.159).argmin()]-most_likely_nH2p_ne_value])
								sigma[sigma<0] = np.array([most_likely_nH2p_ne_value-record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index].min(),record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index].max()-most_likely_nH2p_ne_value])[sigma<0]
								sigma_nH2p_ne_left,sigma_nH2p_ne_right = sigma
							sigma_nH2p_ne = max(sigma_nH2p_ne_right,sigma_nH2p_ne_left)
							# temp = np.sqrt(-2*(likelihood_log_probs[most_likely_nH_ne_index,most_likely_nHm_ne_index,:,most_likely_nH2p_ne_index,most_likely_ne_index,most_likely_Te_index]-max_likelihood_log_prob))
							if H2_steps!=1:
								temp_2 = likelihood_log_probs[most_likely_nH_ne_index,most_likely_nHm_nH2_index,:,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index]
								temp_2 = np.exp(temp_2+np.log(d_nH2_ne_values)-np.log(np.sum(np.exp(temp_2)*d_nH2_ne_values)))
								if False:
									try:
										# sigma_nH2_ne_right = curve_fit(find_sigma,np.abs(nH2_ne_values[most_likely_nH2_ne_index:]-most_likely_nH2_ne_value),temp[most_likely_nH2_ne_index:],p0=[1])[0][0]#,sigma=np.abs(nH2_ne_values[most_likely_nH2_ne_index:]-most_likely_nH2_ne_value)/most_likely_nH2_ne_value+0.1)[0]
										sigma_nH2_ne_right = temp_2[most_likely_nH2_ne_index:]
										if np.sum(sigma_nH2_ne_right)<0.34:
											sigma_nH2_ne_right=0
										else:
											sigma_nH2_ne_right = np.abs(np.array([sum(sigma_nH2_ne_right[:_+1]) for _ in range(len(sigma_nH2_ne_right)) ])-0.34)
											sigma_nH2_ne_right = sigma_nH2_ne_right[1:].argmin()
											sigma_nH2_ne_right = -record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][most_likely_nH2_ne_index]+record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][most_likely_nH2_ne_index+1+sigma_nH2_ne_right]
									except:
										sigma_nH2_ne_right=0
									try:
										# sigma_nH2_ne_left = curve_fit(find_sigma,np.abs(nH2_ne_values[:most_likely_nH2_ne_index+1]-most_likely_nH2_ne_value),temp[:most_likely_nH2_ne_index+1],p0=[1])[0][0]#,sigma=np.abs(nH2_ne_values[:most_likely_nH2_ne_index+1]-most_likely_nH2_ne_value)/most_likely_nH2_ne_value+0.1)[0]
										sigma_nH2_ne_left = np.flip(temp_2[:most_likely_nH2_ne_index+1],axis=0)
										if np.sum(sigma_nH2_ne_left)<0.34:
											sigma_nH2_ne_left=0
										else:
											sigma_nH2_ne_left = np.abs(np.array([sum(sigma_nH2_ne_left[:_+1]) for _ in range(len(sigma_nH2_ne_left)) ])-0.34)
											sigma_nH2_ne_left = sigma_nH2_ne_left[1:].argmin()
											sigma_nH2_ne_left = record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][most_likely_nH2_ne_index]-record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][most_likely_nH2_ne_index-1-sigma_nH2_ne_left]
									except:
										sigma_nH2_ne_left=0
								else:
									temp_2 = np.cumsum(temp_2)
									sigma = np.array([most_likely_nH2_ne_value-record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][np.abs(temp_2-0.159).argmin()+1],record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][np.abs(temp_2-1+0.159).argmin()]-most_likely_nH2_ne_value])
									sigma[sigma<0] = np.array([most_likely_nH2_ne_value-record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index].min(),record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index].max()-most_likely_nH2_ne_value])[sigma<0]
									sigma_nH2_ne_left,sigma_nH2_ne_right = sigma
								sigma_nH2_ne = max(sigma_nH2_ne_right,sigma_nH2_ne_left)
							else:
								sigma_nH2_ne_right = 0
								sigma_nH2_ne_left = 0
								sigma_nH2_ne = 0
							# temp = np.sqrt(-2*(likelihood_log_probs[most_likely_nH_ne_index,:,most_likely_nH2_ne_index,most_likely_nH2p_ne_index,most_likely_ne_index,most_likely_Te_index]-max_likelihood_log_prob))
							temp_2 = likelihood_log_probs[most_likely_nH_ne_index,:,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index]
							temp_2 = np.exp(temp_2+np.log(d_nHm_ne_values)-np.log(np.sum(np.exp(temp_2)*d_nHm_ne_values)))
							if False:
								try:
									# sigma_nHm_ne_right = curve_fit(find_sigma,np.abs(nHm_ne_values[most_likely_nHm_ne_index:]-most_likely_nHm_ne_value),temp[most_likely_nHm_ne_index:],p0=[1])[0][0]#,sigma=np.abs(nHm_ne_values[most_likely_nHm_ne_index:]-most_likely_nHm_ne_value)/most_likely_nHm_ne_value+0.1)[0]
									sigma_nHm_ne_right = temp_2[most_likely_nHm_nH2_index:]
									if np.sum(sigma_nHm_ne_right)<0.34:
										sigma_nHm_ne_right=0
									else:
										sigma_nHm_ne_right = np.abs(np.array([sum(sigma_nHm_ne_right[:_+1]) for _ in range(len(sigma_nHm_ne_right)) ])-0.34)
										sigma_nHm_ne_right = sigma_nHm_ne_right[1:].argmin()
										sigma_nHm_ne_right = -record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][most_likely_nHm_nH2_index]+record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][most_likely_nHm_nH2_index+1+sigma_nHm_ne_right]
								except:
									sigma_nHm_ne_right=0
								try:
									# sigma_nHm_ne_left = curve_fit(find_sigma,np.abs(nHm_ne_values[:most_likely_nHm_ne_index+1]-most_likely_nHm_ne_value),temp[:most_likely_nHm_ne_index+1],p0=[1])[0][0]#,sigma=np.abs(nHm_ne_values[:most_likely_nHm_ne_index+1]-most_likely_nHm_ne_value)/most_likely_nHm_ne_value+0.1)[0]
									sigma_nHm_ne_left = np.flip(temp_2[:most_likely_nHm_nH2_index+1],axis=0)
									if np.sum(sigma_nHm_ne_left)<0.34:
										sigma_nHm_ne_left=0
									else:
										sigma_nHm_ne_left = np.abs(np.array([sum(sigma_nHm_ne_left[:_+1]) for _ in range(len(sigma_nHm_ne_left)) ])-0.34)
										sigma_nHm_ne_left = sigma_nHm_ne_left[1:].argmin()
										sigma_nHm_ne_left = record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][most_likely_nHm_nH2_index]-record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][most_likely_nHm_nH2_index-1-sigma_nHm_ne_left]
								except:
									sigma_nHm_ne_left=0
							else:
								temp_2 = np.cumsum(temp_2)
								sigma = np.array([most_likely_nHm_ne_value-record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][np.abs(temp_2-0.159).argmin()+1],record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][np.abs(temp_2-1+0.159).argmin()]-most_likely_nHm_ne_value])
								sigma[sigma<0] = np.array([most_likely_nHm_ne_value-record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index].min(),record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index].max()-most_likely_nHm_ne_value])[sigma<0]
								sigma_nHm_ne_left,sigma_nHm_ne_right = sigma
							sigma_nHm_ne = max(sigma_nHm_ne_right,sigma_nHm_ne_left)
							# temp = np.sqrt(-2*(likelihood_log_probs[:,most_likely_nHm_ne_index,most_likely_nH2_ne_index,most_likely_nH2p_ne_index,most_likely_ne_index,most_likely_Te_index]-max_likelihood_log_prob))
							temp_2 = likelihood_log_probs[:,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index]
							temp_2 = np.exp(temp_2+np.log(d_nH_ne_values)-np.log(np.sum(np.exp(temp_2)*d_nH_ne_values)))
							if False:
								try:
									# sigma_nH_ne_right = curve_fit(find_sigma,np.abs(nH_ne_values[most_likely_nH_ne_index:]-most_likely_nH_ne_value),temp[most_likely_nH_ne_index:],p0=[1])[0][0]#,sigma=np.abs(nH_ne_values[most_likely_nH_ne_index:]-most_likely_nH_ne_value)/most_likely_nH_ne_value+0.1)[0]
									sigma_nH_ne_right = temp_2[most_likely_nH_ne_index:]
									if np.sum(sigma_nH_ne_right)<0.34:
										sigma_nH_ne_right=0
									else:
										sigma_nH_ne_right = np.abs(np.array([sum(sigma_nH_ne_right[:_+1]) for _ in range(len(sigma_nH_ne_right)) ])-0.34)
										sigma_nH_ne_right = sigma_nH_ne_right[1:].argmin()
										sigma_nH_ne_right = -record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][most_likely_nH_ne_index]+record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][most_likely_nH_ne_index+1+sigma_nH_ne_right]
								except:
									sigma_nH_ne_right=0
								try:
									# sigma_nH_ne_left = curve_fit(find_sigma,np.abs(nH_ne_values[:most_likely_nH_ne_index+1]-most_likely_nH_ne_value),temp[:most_likely_nH_ne_index+1],p0=[1])[0][0]#,sigma=np.abs(nH_ne_values[:most_likely_nH_ne_index+1]-most_likely_nH_ne_value)/most_likely_nH_ne_value+0.1)[0][0]
									sigma_nH_ne_left = np.flip(temp_2[:most_likely_nH_ne_index+1],axis=0)
									if np.sum(sigma_nH_ne_left)<0.34:
										sigma_nH_ne_left=0
									else:
										sigma_nH_ne_left = np.abs(np.array([sum(sigma_nH_ne_left[:_+1]) for _ in range(len(sigma_nH_ne_left)) ])-0.34)
										sigma_nH_ne_left = sigma_nH_ne_left[1:].argmin()
										sigma_nH_ne_left = record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][most_likely_nH_ne_index]-record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][most_likely_nH_ne_index-1-sigma_nH_ne_left]
								except:
									sigma_nH_ne_left=0
							else:
								temp_2 = np.cumsum(temp_2)
								sigma = np.array([most_likely_nH_ne_value-record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][np.abs(temp_2-0.159).argmin()+1],record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][np.abs(temp_2-1+0.159).argmin()]-most_likely_nH_ne_value])
								sigma[sigma<0] = np.array([most_likely_nH_ne_value-record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index].min(),record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index].max()-most_likely_nH_ne_value])[sigma<0]
								sigma_nH_ne_left,sigma_nH_ne_right = sigma
							sigma_nH_ne = max(sigma_nH_ne_right,sigma_nH_ne_left)
							sigma_nHp_ne = (1+most_likely_nHm_ne_value-most_likely_nH2p_ne_value-0)*((sigma_nHm_ne)**2+(sigma_nH2p_ne)**2)**0.5
							sigma_nH3p_ne = 0

							if my_time_pos in sample_time_step:
								if my_r_pos in sample_radious:
									if to_print:
										# tm.sleep(np.random.random()*10)
										fig, ax = plt.subplots( 4,2,figsize=(25, 40), squeeze=False)
										fig.suptitle('first scan\nmost_likely_nH_ne_value %.3g,most_likely_nHm_ne_value %.3g,most_likely_nH2_ne_value %.3g,most_likely_nH2p_ne_value %.3g' %(most_likely_nH_ne_value*most_likely_ne_value,most_likely_nHm_ne_value*most_likely_ne_value,most_likely_nH2_ne_value*most_likely_ne_value,most_likely_nH2p_ne_value*most_likely_ne_value) +'\nlines '+str(n_list_all)+'\nlocation [time, r]'+ ' [%.3g ms, ' % time_crop[my_time_pos] + ' %.3g mm]' % (1000*r_crop[my_r_pos]) +' , [TS Te, TS ne] '+ ' [%.3g(ML %.3g) eV, ' %(merge_Te_prof_multipulse_interp_crop_limited_restrict,most_likely_Te_value) + '%.3g(ML %.3g) #10^20/m^3]' %(merge_ne_prof_multipulse_interp_crop_limited_restrict,most_likely_ne_value*1e-20) +'\nfit [nH/ne, nH+/ne, nH-/ne, nH2/ne, nH2+/ne, nH3+/ne] , '+ '[%.3g, ' % most_likely_nH_ne_value + '%.3g, ' %(1 - most_likely_nH2p_ne_value + most_likely_nHm_ne_value)+ '%.3g, ' % most_likely_nHm_ne_value+ '%.3g, ' % most_likely_nH2_ne_value+ '%.3g, ' % most_likely_nH2p_ne_value+ '%.3g]' % 0+'\nBest index: [nH/ne,nH-/ne,nH2/ne,nH2+/ne,ne,Te] [%.3g,%.3g,%.3g,%.3g,%.3g,%.3g]' %(most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index)+'\nMarginalised best index: [nH/ne,nH-/ne,nH2+/ne] [%.3g,%.3g,%.3g]' %(most_likely_marginalised_nH_ne_index,most_likely_marginalised_nHm_nH2_index,most_likely_marginalised_nH2p_nH2_index))
										plot_index = 0
										color = ['b', 'r', 'm', 'y', 'g', 'c', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise', 'teal', 'olive']
										im = ax[plot_index,0].plot(np.sort(intervals_power_rad_excit.tolist()*2)[1:-1],100*np.array([prob_power_rad_excit.tolist()]*2).T.flatten(),color=color[0],label='power_rad_excit');
										im = ax[plot_index,0].plot(actual_values_power_rad_excit,100*prob_power_rad_excit,'+',color=color[0],markersize=5);
										# im = ax[plot_index,0].plot(most_likely_power_rad_excit,100,color[0]+'o',markersize=10,fillstyle='none');
										im = ax[plot_index,0].axvline(x=most_likely_power_rad_excit,linestyle='--',color=color[0])
										im = ax[plot_index,0].plot(np.sort(intervals_power_rad_rec_bremm.tolist()*2)[1:-1],100*np.array([prob_power_rad_rec_bremm.tolist()]*2).T.flatten(),color=color[1],label='power_rad_rec_bremm');
										im = ax[plot_index,0].plot(actual_values_power_rad_rec_bremm,100*prob_power_rad_rec_bremm,'+',color=color[1],markersize=5);
										# im = ax[plot_index,0].plot(most_likely_power_rad_rec_bremm,100,color[1]+'o',markersize=10,fillstyle='none');
										im = ax[plot_index,0].axvline(x=most_likely_power_rad_rec_bremm,linestyle='--',color=color[1])
										im = ax[plot_index,0].plot(np.sort(intervals_power_rad_mol.tolist()*2)[1:-1],100*np.array([prob_power_rad_mol.tolist()]*2).T.flatten(),color=color[2],label='power_rad_mol');
										im = ax[plot_index,0].plot(actual_values_power_rad_mol,100*prob_power_rad_mol,'+',color=color[2],markersize=5);
										# im = ax[plot_index,0].plot(most_likely_power_rad_mol,100,color[2]+'o',markersize=10,fillstyle='none');
										im = ax[plot_index,0].axvline(x=most_likely_power_rad_mol,linestyle='--',color=color[2])
										im = ax[plot_index,0].plot(np.sort(intervals_power_via_ionisation.tolist()*2)[1:-1],100*np.array([prob_power_via_ionisation.tolist()]*2).T.flatten(),color=color[3],label='power_via_ionisation');
										im = ax[plot_index,0].plot(actual_values_power_via_ionisation,100*prob_power_via_ionisation,'+',color=color[3],markersize=5);
										# im = ax[plot_index,0].plot(most_likely_power_via_ionisation,100,color[3]+'o',markersize=10,fillstyle='none');
										im = ax[plot_index,0].axvline(x=most_likely_power_via_ionisation,linestyle='--',color=color[3])
										im = ax[plot_index,0].plot(np.sort(intervals_power_via_recombination.tolist()*2)[1:-1],100*np.array([prob_power_via_recombination.tolist()]*2).T.flatten(),color=color[4],label='power_via_recombination');
										im = ax[plot_index,0].plot(actual_values_power_via_recombination,100*prob_power_via_recombination,'+',color=color[4],markersize=5);
										# im = ax[plot_index,0].plot(most_likely_power_via_recombination,100,color[4]+'o',markersize=10,fillstyle='none');
										im = ax[plot_index,0].axvline(x=most_likely_power_via_recombination,linestyle='--',color=color[4])
										im = ax[plot_index,0].plot(np.sort(intervals_tot_rad_power.tolist()*2)[1:-1],100*np.array([prob_tot_rad_power.tolist()]*2).T.flatten(),color=color[5],label='tot_rad_power');
										im = ax[plot_index,0].plot(actual_values_tot_rad_power,100*prob_tot_rad_power,'+',color=color[5],markersize=5);
										# im = ax[plot_index,0].plot(most_likely_tot_rad_power,100,color[5]+'o',markersize=10,fillstyle='none');
										im = ax[plot_index,0].axvline(x=most_likely_tot_rad_power,linestyle='--',color=color[5])
										im = ax[plot_index,0].plot(np.sort(intervals_power_rad_Hm.tolist()*2)[1:-1],100*np.array([prob_power_rad_Hm.tolist()]*2).T.flatten(),color=color[6],label='power_rad_Hm');
										im = ax[plot_index,0].plot(actual_values_power_rad_Hm,100*prob_power_rad_Hm,'+',color=color[6],markersize=5);
										# im = ax[plot_index,0].plot(most_likely_power_rad_Hm,100,'o',color=color[6],markersize=10,fillstyle='none');
										im = ax[plot_index,0].axvline(x=most_likely_power_rad_Hm,linestyle='--',color=color[6])
										im = ax[plot_index,0].plot(np.sort(intervals_power_rad_H2.tolist()*2)[1:-1],100*np.array([prob_power_rad_H2.tolist()]*2).T.flatten(),color=color[7],label='power_rad_H2');
										im = ax[plot_index,0].plot(actual_values_power_rad_H2,100*prob_power_rad_H2,'+',color=color[7],markersize=5);
										# im = ax[plot_index,0].plot(most_likely_power_rad_H2,100,'o',color=color[7],markersize=10,fillstyle='none');
										im = ax[plot_index,0].axvline(x=most_likely_power_rad_H2,linestyle='--',color=color[7])
										im = ax[plot_index,0].plot(np.sort(intervals_power_rad_H2p.tolist()*2)[1:-1],100*np.array([prob_power_rad_H2p.tolist()]*2).T.flatten(),color=color[8],label='power_rad_H2p');
										im = ax[plot_index,0].plot(actual_values_power_rad_H2p,100*prob_power_rad_H2p,'+',color=color[8],markersize=5);
										# im = ax[plot_index,0].plot(most_likely_power_rad_H2p,100,'o',color=color[8],markersize=10,fillstyle='none');
										im = ax[plot_index,0].axvline(x=most_likely_power_rad_H2p,linestyle='--',color=color[8])
										im = ax[plot_index,0].plot(np.sort(intervals_power_heating_rec.tolist()*2)[1:-1],100*np.array([prob_power_heating_rec.tolist()]*2).T.flatten(),color=color[9],label='power_heating_rec');
										im = ax[plot_index,0].plot(actual_values_power_heating_rec,100*prob_power_heating_rec,'+',color=color[9],markersize=5);
										# im = ax[plot_index,0].plot(most_likely_power_heating_rec,100,'o',color=color[9],markersize=10,fillstyle='none');
										im = ax[plot_index,0].axvline(x=most_likely_power_heating_rec,linestyle='--',color=color[9])
										im = ax[plot_index,0].plot(np.sort(intervals_power_rec_neutral.tolist()*2)[1:-1],100*np.array([prob_power_rec_neutral.tolist()]*2).T.flatten(),color=color[10],label='power_rec_neutral');
										im = ax[plot_index,0].plot(actual_values_power_rec_neutral,100*prob_power_rec_neutral,'+',color=color[10],markersize=5);
										# im = ax[plot_index,0].plot(most_likely_power_rec_neutral,100,'o',color=color[10],markersize=10,fillstyle='none');
										im = ax[plot_index,0].axvline(x=most_likely_power_rec_neutral,linestyle='--',color=color[10])
										im = ax[plot_index,0].plot(np.sort(intervals_power_via_brem.tolist()*2)[1:-1],100*np.array([prob_power_via_brem.tolist()]*2).T.flatten(),color=color[11],label='power_via_brem');
										im = ax[plot_index,0].plot(actual_values_power_via_brem,100*prob_power_via_brem,'+',color=color[11],markersize=5);
										# im = ax[plot_index,0].plot(most_likely_power_via_brem,100,'o',color=color[11],markersize=10,fillstyle='none');
										im = ax[plot_index,0].axvline(x=most_likely_power_via_brem,linestyle='--',color=color[11])
										im = ax[plot_index,0].plot(np.sort(intervals_total_removed_power.tolist()*2)[1:-1],100*np.array([prob_total_removed_power.tolist()]*2).T.flatten(),color=color[12],label='total_removed_power');
										im = ax[plot_index,0].plot(actual_values_total_removed_power,100*prob_total_removed_power,'+',color=color[12],markersize=5);
										# im = ax[plot_index,0].plot(most_likely_total_removed_power,100,'o',color=color[12],markersize=10,fillstyle='none');
										im = ax[plot_index,0].axvline(x=most_likely_total_removed_power,linestyle='--',color=color[12])
										im = ax[plot_index,0].plot(np.sort(intervals_local_CX.tolist()*2)[1:-1],100*np.array([prob_local_CX.tolist()]*2).T.flatten(),color=color[13],label='local_CX');
										im = ax[plot_index,0].plot(actual_values_local_CX,100*prob_local_CX,'+',color=color[13],markersize=5);
										# im = ax[plot_index,0].plot(most_likely_local_CX,100,'o',color=color[13],markersize=10,fillstyle='none');
										im = ax[plot_index,0].axvline(x=most_likely_local_CX,linestyle='--',color=color[13])
										# im = ax[plot_index,0].plot([interpolated_power_pulse_shape(time_crop[my_time_pos])/source_power_spread/area/length]*2,[0,100],'k--');
										im = ax[plot_index,0].axvline(x=most_likely_total_removable_power_times_volume,linestyle='--',color='k');
										im = ax[plot_index,0].axvline(x=max_total_removable_power_times_volume,linestyle='--',color='gray');
										im = ax[plot_index,0].axvline(x=min_total_removable_power_times_volume,linestyle='--',color='gray');
										ax[plot_index,0].set_title('Power balance PDF, prob times volume below %.3g of peak neglected, "--"=ML' %(min_fraction_of_prob_considered) + '\nsigma: molecular %.3g%%, atomic %.3g%%, budget %.3g%%' %(power_molecular_precision*100,power_atomic_precision*100,power_budget_precision*100))
										ax[plot_index,0].set_ylabel('normalised likelihood')
										ax[plot_index,0].set_xlabel('Power [W/m3]')
										ax[plot_index,0].set_xscale('log')
										ax[plot_index,0].legend(loc='best', fontsize='x-small')
										# ax[plot_index,0].set_xlim(left=1e-1*np.min([most_likely_power_rad_excit,most_likely_power_rad_rec_bremm,most_likely_power_rad_mol,most_likely_power_via_ionisation,most_likely_power_via_recombination,most_likely_tot_rad_power]))
										ax[plot_index,0].set_xlim(left=max(1,1e-1*np.min([most_likely_power_rad_excit,most_likely_power_rad_rec_bremm,most_likely_power_rad_mol,most_likely_power_via_ionisation,most_likely_power_via_recombination,most_likely_tot_rad_power,most_likely_power_heating_rec,most_likely_power_rec_neutral,most_likely_power_via_brem,most_likely_total_removed_power])),right=1e2*np.max([most_likely_power_rad_excit,most_likely_power_rad_rec_bremm,most_likely_power_rad_mol,most_likely_power_via_ionisation,most_likely_power_via_recombination,most_likely_tot_rad_power]))
										ax[plot_index,0].grid()

										temp_2 = likelihood_log_probs[most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,:]
										temp_2 = np.exp(temp_2-np.log(np.sum(np.exp(temp_2))))
										im = ax[plot_index,1].plot(Te_values_array,100*temp_2,'b');
										im = ax[plot_index,1].plot(Te_values_array,100*temp_2,'xb');
										# im = ax[plot_index,1].plot(most_likely_Te_value,100,'ko',markersize=30,fillstyle='none');
										# im = ax[plot_index,1].plot(merge_Te_prof_multipulse_interp_crop_limited_restrict,100,'kx',markersize=30);
										im = ax[plot_index,1].axvline(x=most_likely_Te_value,linestyle='--',color='k')
										im = ax[plot_index,1].axvline(x=merge_Te_prof_multipulse_interp_crop_limited_restrict,linestyle='-.',color='k')
										im = ax[plot_index,1].plot(Te_values_array,100*np.exp(Te_log_probs)/np.sum(np.exp(Te_log_probs)),':');
										im = ax[plot_index,1].axvline(x=most_likely_Te_value-sigma_Te_left,linestyle='--',color='k')
										im = ax[plot_index,1].axvline(x=most_likely_Te_value+sigma_Te_right,linestyle='--',color='k')
										im = ax[plot_index,1].axvline(x=np.min(Te_values_limits),linestyle='--',color='y')
										im = ax[plot_index,1].axvline(x=np.max(Te_values_limits),linestyle='--',color='y')
										im = ax[plot_index,1].plot(np.sort(intervals_Te_values.tolist()*2)[1:-1],100*np.array([prob_Te_values.tolist()]*2).T.flatten(),color='m');
										im = ax[plot_index,1].plot(actual_values_Te_values,100*prob_Te_values,'+',color='m',markersize=5);
										ax[plot_index,1].set_title('Te at highest likelihood with precision\n :=prior, "--"=ML+sigma, "-."=TS')
										ax[plot_index,1].set_ylabel('normalised likelihood')
										ax[plot_index,1].set_xlabel('temperature [eV]')
										# ax[plot_index,0].set_xscale('log')
										# ax[plot_index,1].legend(loc='best', fontsize='x-small')
										ax[plot_index,1].grid()

										plot_index += 1
										temp_2 = likelihood_log_probs[:,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index]
										temp_2 = np.exp(temp_2-np.log(np.sum(np.exp(temp_2))))
										im = ax[plot_index,0].plot(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],100*temp_2,color[0],label='nH_ne');
										# im = ax[plot_index,0].plot(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][record_nH_ne_log_prob[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index].argmax()],100,color[0]+'x',markersize=30);
										# im = ax[plot_index,0].plot(most_likely_nH_ne_value,100,color[0]+'o',markersize=30,fillstyle='none');
										im = ax[plot_index,0].axvline(x=record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][record_nH_ne_log_prob[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index].argmax()],linestyle='-.',color=color[0])
										im = ax[plot_index,0].axvline(x=most_likely_nH_ne_value,linestyle='--',color=color[0])
										im = ax[plot_index,0].axvline(x=most_likely_nH_ne_value-sigma_nH_ne_left,linestyle='--',color=color[0])
										im = ax[plot_index,0].axvline(x=most_likely_nH_ne_value+sigma_nH_ne_right,linestyle='--',color=color[0])
										im = ax[plot_index,0].plot(record_nH_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],100*np.exp(record_nH_ne_log_prob[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])/np.sum(np.exp(record_nH_ne_log_prob[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])),color[0]+':');
										im = ax[plot_index,0].plot(np.sort(intervals_nH_ne_values.tolist()*2)[1:-1],100*np.array([prob_nH_ne_values.tolist()]*2).T.flatten(),color=color[0]);
										im = ax[plot_index,0].plot(actual_values_nH_ne_values,100*prob_nH_ne_values,'+',color=color[0],markersize=5);
										im = ax[plot_index,0].plot(np.sort(intervals_nH_ne_excited_states.tolist()*2)[1:-1],100*np.array([prob_nH_ne_excited_states.tolist()]*2).T.flatten(),'--',color=color[0]);
										im = ax[plot_index,0].plot(actual_values_nH_ne_excited_states,100*prob_nH_ne_excited_states,'+',color=color[0],markersize=5);
										temp_2 = likelihood_log_probs[most_likely_nH_ne_index,:,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index]
										temp_2 = np.exp(temp_2-np.log(np.sum(np.exp(temp_2))))
										im = ax[plot_index,0].plot(record_nHm_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],100*temp_2,color[1],label='nHm_ne');
										# im = ax[plot_index,0].plot(most_likely_nHm_ne_value,100,color[1]+'o',markersize=30,fillstyle='none');
										im = ax[plot_index,0].axvline(x=most_likely_nHm_ne_value,linestyle='--',color=color[1])
										im = ax[plot_index,0].axvline(x=most_likely_nHm_ne_value-sigma_nHm_ne_left,linestyle='--',color=color[1])
										im = ax[plot_index,0].axvline(x=most_likely_nHm_ne_value+sigma_nHm_ne_right,linestyle='--',color=color[1])
										im = ax[plot_index,0].plot(np.sort(intervals_nHm_ne_values.tolist()*2)[1:-1],100*np.array([prob_nHm_ne_values.tolist()]*2).T.flatten(),color=color[1]);
										im = ax[plot_index,0].plot(actual_values_nHm_ne_values,100*prob_nHm_ne_values,'+',color=color[1],markersize=5);
										temp_2 = likelihood_log_probs[most_likely_nH_ne_index,most_likely_nHm_nH2_index,:,most_likely_nH2p_nH2_index,most_likely_ne_index,most_likely_Te_index]
										temp_2 = np.exp(temp_2-np.log(np.sum(np.exp(temp_2))))
										im = ax[plot_index,0].plot(record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],100*temp_2,color[2],label='nH2_ne');
										# im = ax[plot_index,0].plot(record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][record_nH2_ne_log_prob[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index].argmax()],100,color[2]+'x',markersize=30);
										# im = ax[plot_index,0].plot(most_likely_nH2_ne_value,100,color[2]+'o',markersize=30,fillstyle='none');
										im = ax[plot_index,0].axvline(x=most_likely_nH2_ne_value,linestyle='--',color=color[2])
										im = ax[plot_index,0].axvline(x=record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index][record_nH2_ne_log_prob[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index].argmax()],linestyle='-.',color=color[2])
										im = ax[plot_index,0].axvline(x=most_likely_nH2_ne_value-sigma_nH2_ne_left,linestyle='--',color=color[2])
										im = ax[plot_index,0].axvline(x=most_likely_nH2_ne_value+sigma_nH2_ne_right,linestyle='--',color=color[2])
										im = ax[plot_index,0].plot(record_nH2_ne_values[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],100*np.exp(record_nH2_ne_log_prob[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])/np.sum(np.exp(record_nH2_ne_log_prob[:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index])),color[2]+':');
										im = ax[plot_index,0].plot(np.sort(intervals_nH2_ne_values.tolist()*2)[1:-1],100*np.array([prob_nH2_ne_values.tolist()]*2).T.flatten(),color=color[2]);
										im = ax[plot_index,0].plot(actual_values_nH2_ne_values,100*prob_nH2_ne_values,'+',color=color[2],markersize=5);
										temp_2 = likelihood_log_probs[most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,:,most_likely_ne_index,most_likely_Te_index]
										temp_2 = np.exp(temp_2-np.log(np.sum(np.exp(temp_2))))
										im = ax[plot_index,0].plot(record_nH2p_ne_values[most_likely_nH2_ne_index,:,most_likely_ne_index*TS_Te_steps + most_likely_Te_index],100*temp_2,color[3],label='nH2p_ne');
										# im = ax[plot_index,0].plot(most_likely_nH2p_ne_value,100,color[3]+'o',markersize=30,fillstyle='none');
										im = ax[plot_index,0].axvline(x=most_likely_nH2p_ne_value,linestyle='--',color=color[3])
										im = ax[plot_index,0].axvline(x=most_likely_nH2p_ne_value-sigma_nH2p_ne_left,linestyle='--',color=color[3])
										im = ax[plot_index,0].axvline(x=most_likely_nH2p_ne_value+sigma_nH2p_ne_right,linestyle='--',color=color[3])
										im = ax[plot_index,0].plot(np.sort(intervals_nH2p_ne_values.tolist()*2)[1:-1],100*np.array([prob_nH2p_ne_values.tolist()]*2).T.flatten(),color=color[3]);
										im = ax[plot_index,0].plot(actual_values_nH2p_ne_values,100*prob_nH2p_ne_values,'+',color=color[3],markersize=5);
										ax[plot_index,0].set_title('Relative densities at highest likelihood with precision\n :=prior, "--"=ML+sigma, "-."=central')
										ax[plot_index,0].set_ylabel('normalised likelihood')
										ax[plot_index,0].set_xlabel('[au]')
										ax[plot_index,0].set_xscale('log')
										ax[plot_index,0].legend(loc='best', fontsize='x-small')
										ax[plot_index,0].grid()

										temp_2 = likelihood_log_probs[most_likely_nH_ne_index,most_likely_nHm_nH2_index,most_likely_nH2_ne_index,most_likely_nH2p_nH2_index,:,most_likely_Te_index]
										temp_2 = np.exp(temp_2-np.log(np.sum(np.exp(temp_2))))
										im = ax[plot_index,1].plot(ne_values_array,100*temp_2,'b');
										im = ax[plot_index,1].plot(ne_values_array,100*temp_2,'xb');
										im = ax[plot_index,1].axvline(x=most_likely_ne_value,linestyle='--',color='k')
										im = ax[plot_index,1].axvline(x=merge_ne_prof_multipulse_interp_crop_limited_restrict*1e20,linestyle='-.',color='k')
										im = ax[plot_index,1].plot(ne_values_array,100*np.exp(ne_log_probs)/np.sum(np.exp(ne_log_probs)),':');
										im = ax[plot_index,1].axvline(x=most_likely_ne_value-sigma_ne_left,linestyle='--',color='k');
										im = ax[plot_index,1].axvline(x=most_likely_ne_value+sigma_ne_right,linestyle='--',color='k');
										im = ax[plot_index,1].axvline(x=np.min(ne_values_limits),linestyle='--',color='y');
										im = ax[plot_index,1].axvline(x=np.max(ne_values_limits),linestyle='--',color='y');
										im = ax[plot_index,1].plot(np.sort(intervals_ne_values.tolist()*2)[1:-1],100*np.array([prob_ne_values.tolist()]*2).T.flatten(),color='m');
										im = ax[plot_index,1].plot(actual_values_ne_values,100*prob_ne_values,'+',color='m',markersize=5);
										ax[plot_index,1].set_title('ne at highest likelihood with precision\n :=prior, "--"=ML+sigma, "-."=TS')
										ax[plot_index,1].set_ylabel('normalised likelihood')
										ax[plot_index,1].set_xlabel('density [#/m3]')
										ax[plot_index,1].set_xscale('log')
										# ax[plot_index,1].legend(loc='best', fontsize='x-small')
										ax[plot_index,1].grid()

										if include_particles_limitation:
											plot_index += 1
											if most_likely_net_e_destruction<=0:
												im = ax[plot_index,0].axvline(x=-most_likely_net_e_destruction,linestyle='--',color='g')
											else:
												im = ax[plot_index,0].axvline(x=most_likely_net_e_destruction,linestyle='--',color='r')
											# im = ax[plot_index,0].plot(np.sort(intervals_net_e_destruction.tolist()*2)[1:-1],100*np.array([prob_net_e_destruction.tolist()]*2).T.flatten(),color='m');
											# im = ax[plot_index,0].plot(actual_values_net_e_destruction,100*prob_net_e_destruction,'+',color='m',markersize=5);
											im = ax[plot_index,0].plot(np.sort(intervals_net_e_destruction[intervals_net_e_destruction>=0].tolist()*2)[1:-1],100*np.array([prob_net_e_destruction[(intervals_net_e_destruction>=0)[:-1]].tolist()]*2).T.flatten(),color='r');
											im = ax[plot_index,0].plot(-np.sort(intervals_net_e_destruction[intervals_net_e_destruction<=0].tolist()*2)[1:-1],100*np.array([prob_net_e_destruction[(intervals_net_e_destruction<=0)[1:]].tolist()]*2).T.flatten(),color='g');
											im = ax[plot_index,0].plot(-actual_values_net_e_destruction[actual_values_net_e_destruction<=0],100*prob_net_e_destruction[actual_values_net_e_destruction<=0],'+',color='g',markersize=5);
											im = ax[plot_index,0].plot(actual_values_net_e_destruction[actual_values_net_e_destruction>=0],100*prob_net_e_destruction[actual_values_net_e_destruction>=0],'+',color='r',markersize=5);
											im = ax[plot_index,0].axvline(x=most_likely_total_removable_e,linestyle='--',color='k');
											im = ax[plot_index,0].axvline(x=total_removable_e[non_zero_prob_select].max(),linestyle='--',color='gray');
											im = ax[plot_index,0].axvline(x=total_removable_e[non_zero_prob_select].min(),linestyle='--',color='gray');
											ax[plot_index,0].set_title('electrons balance within time and volume step\ngreen=creation, red=destruction\nsigma: molecular %.3g%%, atomic %.3g%%, budget %.3g%%' %(particle_molecular_precision*100,particle_atomic_precision*100,particle_atomic_budget_precision*100))
											ax[plot_index,0].set_ylabel('normalised likelihood')
											ax[plot_index,0].set_xlabel('particles [#*10^20]')
											ax[plot_index,0].set_xscale('log')
											# ax[plot_index,0].legend(loc='best', fontsize='x-small')
											ax[plot_index,0].grid()

											if most_likely_net_Hp_destruction<=0:
												im = ax[plot_index,1].axvline(x=-most_likely_net_Hp_destruction,linestyle='--',color='g')
											else:
												im = ax[plot_index,1].axvline(x=most_likely_net_Hp_destruction,linestyle='--',color='r')
											# im = ax[plot_index,1].plot(np.sort(intervals_net_Hp_destruction.tolist()*2)[1:-1],100*np.array([prob_net_Hp_destruction.tolist()]*2).T.flatten(),color='m');
											# im = ax[plot_index,1].plot(actual_values_net_Hp_destruction,100*prob_net_Hp_destruction,'+',color='m',markersize=5);
											im = ax[plot_index,1].plot(np.sort(intervals_net_Hp_destruction[intervals_net_Hp_destruction>=0].tolist()*2)[1:-1],100*np.array([prob_net_Hp_destruction[(intervals_net_Hp_destruction>=0)[:-1]].tolist()]*2).T.flatten(),color='r');
											im = ax[plot_index,1].plot(-np.sort(intervals_net_Hp_destruction[intervals_net_Hp_destruction<=0].tolist()*2)[1:-1],100*np.array([prob_net_Hp_destruction[(intervals_net_Hp_destruction<=0)[1:]].tolist()]*2).T.flatten(),color='g');
											im = ax[plot_index,1].plot(-actual_values_net_Hp_destruction[actual_values_net_Hp_destruction<=0],100*prob_net_Hp_destruction[actual_values_net_Hp_destruction<=0],'+',color='g',markersize=5);
											im = ax[plot_index,1].plot(actual_values_net_Hp_destruction[actual_values_net_Hp_destruction>=0],100*prob_net_Hp_destruction[actual_values_net_Hp_destruction>=0],'+',color='r',markersize=5);
											im = ax[plot_index,1].axvline(x=most_likely_total_removable_Hp,linestyle='--',color='k');
											im = ax[plot_index,1].axvline(x=total_removable_Hp[non_zero_prob_select].max(),linestyle='--',color='gray');
											im = ax[plot_index,1].axvline(x=total_removable_Hp[non_zero_prob_select].min(),linestyle='--',color='gray');
											ax[plot_index,1].set_title('H+ balance within time and volume step\ngreen=creation, red=destruction\nlarge = H+, small = H\nsigma: molecular %.3g%%, atomic %.3g%%, budget %.3g%%' %(particle_molecular_precision*100,particle_atomic_precision*100,particle_atomic_budget_precision*100))
											ax[plot_index,1].set_ylabel('normalised likelihood')
											ax[plot_index,1].set_xlabel('particles [#*10^20]')
											ax[plot_index,1].set_xscale('log')
											# ax[plot_index,0].legend(loc='best', fontsize='x-small')
											ax[plot_index,1].grid()

											plot_index += 1
											if most_likely_net_Hm_destruction<=0:
												im = ax[plot_index,0].axvline(x=-most_likely_net_Hm_destruction,linestyle='--',color='g')
											else:
												im = ax[plot_index,0].axvline(x=most_likely_net_Hm_destruction,linestyle='--',color='r')
											# im = ax[plot_index,0].plot(np.sort(intervals_net_Hm_destruction.tolist()*2)[1:-1],100*np.array([prob_net_Hm_destruction.tolist()]*2).T.flatten(),color='m');
											# im = ax[plot_index,0].plot(actual_values_net_Hm_destruction,100*prob_net_Hm_destruction,'+',color='m',markersize=5);
											im = ax[plot_index,0].plot(np.sort(intervals_net_Hm_destruction[intervals_net_Hm_destruction>=0].tolist()*2)[1:-1],100*np.array([prob_net_Hm_destruction[(intervals_net_Hm_destruction>=0)[:-1]].tolist()]*2).T.flatten(),color='r');
											im = ax[plot_index,0].plot(-np.sort(intervals_net_Hm_destruction[intervals_net_Hm_destruction<=0].tolist()*2)[1:-1],100*np.array([prob_net_Hm_destruction[(intervals_net_Hm_destruction<=0)[1:]].tolist()]*2).T.flatten(),color='g');
											im = ax[plot_index,0].plot(-actual_values_net_Hm_destruction[actual_values_net_Hm_destruction<=0],100*prob_net_Hm_destruction[actual_values_net_Hm_destruction<=0],'+',color='g',markersize=5);
											im = ax[plot_index,0].plot(actual_values_net_Hm_destruction[actual_values_net_Hm_destruction>=0],100*prob_net_Hm_destruction[actual_values_net_Hm_destruction>=0],'+',color='r',markersize=5);
											im = ax[plot_index,0].axvline(x=most_likely_total_removable_Hm,linestyle='--',color='k');
											im = ax[plot_index,0].axvline(x=total_removable_Hm[non_zero_prob_select].max(),linestyle='--',color='gray');
											im = ax[plot_index,0].axvline(x=total_removable_Hm[non_zero_prob_select].min(),linestyle='--',color='gray');
											ax[plot_index,0].set_title('H- balance within time and volume step\ngreen=creation, red=destruction\nsigma: molecular %.3g%%, atomic %.3g%%, budget %.3g%%' %(particle_molecular_precision*100,particle_atomic_precision*100,particle_molecular_budget_precision*100))
											ax[plot_index,0].set_ylabel('normalised likelihood')
											ax[plot_index,0].set_xlabel('particles [#*10^20]')
											ax[plot_index,0].set_xscale('log')
											# ax[plot_index,0].legend(loc='best', fontsize='x-small')
											ax[plot_index,0].grid()

											if most_likely_net_H2p_destruction<=0:
												im = ax[plot_index,1].axvline(x=-most_likely_net_H2p_destruction,linestyle='--',color='g')
											else:
												im = ax[plot_index,1].axvline(x=most_likely_net_H2p_destruction,linestyle='--',color='r')
											# im = ax[plot_index,1].plot(np.sort(intervals_net_H2p_destruction.tolist()*2)[1:-1],100*np.array([prob_net_H2p_destruction.tolist()]*2).T.flatten(),color='m');
											# im = ax[plot_index,1].plot(actual_values_net_H2p_destruction,100*prob_net_H2p_destruction,'+',color='m',markersize=5);
											im = ax[plot_index,1].plot(np.sort(intervals_net_H2p_destruction[intervals_net_H2p_destruction>=0].tolist()*2)[1:-1],100*np.array([prob_net_H2p_destruction[(intervals_net_H2p_destruction>=0)[:-1]].tolist()]*2).T.flatten(),color='r');
											im = ax[plot_index,1].plot(-np.sort(intervals_net_H2p_destruction[intervals_net_H2p_destruction<=0].tolist()*2)[1:-1],100*np.array([prob_net_H2p_destruction[(intervals_net_H2p_destruction<=0)[1:]].tolist()]*2).T.flatten(),color='g');
											im = ax[plot_index,1].plot(-actual_values_net_H2p_destruction[actual_values_net_H2p_destruction<=0],100*prob_net_H2p_destruction[actual_values_net_H2p_destruction<=0],'+',color='g',markersize=5);
											im = ax[plot_index,1].plot(actual_values_net_H2p_destruction[actual_values_net_H2p_destruction>=0],100*prob_net_H2p_destruction[actual_values_net_H2p_destruction>=0],'+',color='r',markersize=5);
											im = ax[plot_index,1].axvline(x=most_likely_total_removable_H2p,linestyle='--',color='k');
											im = ax[plot_index,1].axvline(x=total_removable_H2p[non_zero_prob_select].max(),linestyle='--',color='gray');
											im = ax[plot_index,1].axvline(x=total_removable_H2p[non_zero_prob_select].min(),linestyle='--',color='gray');
											ax[plot_index,1].set_title('H2+ balance within time and volume step\ngreen=creation, red=destruction\nsigma: molecular %.3g%%, atomic %.3g%%, budget %.3g%%' %(particle_molecular_precision*100,particle_atomic_precision*100,particle_molecular_budget_precision*100))
											ax[plot_index,1].set_ylabel('normalised likelihood')
											ax[plot_index,1].set_xlabel('particles [#*10^20]')
											ax[plot_index,1].set_xscale('log')
											# ax[plot_index,0].legend(loc='best', fontsize='x-small')
											ax[plot_index,1].grid()

										save_done = 0
										save_index=1
										try:	# All this trie are because of a known issues when saving lots of figure inside of multiprocessor
											plt.savefig(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_fourth.eps', bbox_inches='tight')
										except Exception as e:
											print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_fourth.eps save try number '+str(1)+' failed. Reason %s' % e)
											while save_done==0 and save_index<100:
												try:
													plt.savefig(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_fourth.eps', bbox_inches='tight')
													print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_pass'+str(pass_index)+'_Bayesian_search_' + str(my_time_pos)+'_'+str(my_r_pos)+ '_fourth.eps save try number '+str(save_index+1)+' successfull')
													save_done=1
												except Exception as e:
													tm.sleep(np.random.random()**2)
													# print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(save_index+1)+' failed. Reason %s' % e)
													save_index+=1
										plt.close('all')

							# I add this because I want to see how Halpha works
							n_list_all_full = np.array([3]+np.array(n_list_all).tolist())
							multiplicative_factor_full = energy_difference_full[n_list_all_full - 2] * einstein_coeff_full[n_list_all_full - 2] / J_to_eV


							excitation_full = []
							for isel in (np.array(n_list_all_full)-2):
								temp = read_adf15(pecfile, isel, most_likely_Te_value,most_likely_ne_value * 10 ** (0 - 6)) [0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
								temp[np.isinf(temp)] = 0
								excitation_full.append(temp)
							excitation_full = np.array(excitation_full).flatten()  # in # photons cm^-3 s^-1
							excitation_full = (excitation_full * (10 ** -6) * (energy_difference_full[n_list_all_full - 2] / J_to_eV)).T  # in W m^-3 / (# / m^3)**2

							recombination_full = []
							for isel in (np.array(n_list_all_full)+16):
								temp = read_adf15(pecfile, isel, most_likely_Te_value,most_likely_ne_value * 10 ** (0 - 6)) [0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
								recombination_full.append(temp)
							recombination_full = np.array(recombination_full).flatten()  # in # photons cm^-3 s^-1 / (# cm^-3)**2
							recombination_full = (recombination_full * (10 ** -6) * (energy_difference_full[n_list_all_full - 2] / J_to_eV)).T  # in W m^-3 / (# / m^3)**2


							def fit_Yacora_pop_coeff(ne,Te,T_Hp,T_Hm,T_H2p,multiplicative_factor,excitation_restrict,recombination_restrict):
								def calculated_emission(n_list_all, nH_ne,nHp_ne,nHm_ne,nH2_ne,nH2p_ne,nH3p_ne):
									# I add this because I want to see how Halpha works
									# nHp_ne=1+nHm_ne-nH2p_ne-nH3p_ne
									multiplicative_factor = energy_difference_full[n_list_all - 2] * einstein_coeff_full[n_list_all - 2] / J_to_eV
									total = np.array([(excitation_full[n_list_all - 3] *  nH_ne /multiplicative_factor).astype('float')])
									total += np.array([(recombination_full[n_list_all - 3] * nHp_ne /multiplicative_factor).astype('float')])
									# total = nH_ne*From_H_pop_coeff_full([[Te,ne,nHp_ne*ne,nH_ne*ne]],n_list_all)
									# total += nHp_ne*From_Hp_pop_coeff_full([[Te,ne]],n_list_all)
									total += nHm_ne*( From_Hn_with_Hp_pop_coeff_full_extra([[Te,T_Hp,T_Hm,ne,nHp_ne*ne]],n_list_all) + From_Hn_with_H2p_pop_coeff_full_extra([[Te,T_H2p,T_Hm,ne,nH2p_ne*ne]],n_list_all) )
									total += nH2_ne*From_H2_pop_coeff_full_extra([[Te,ne]],n_list_all)
									total += nH2p_ne*From_H2p_pop_coeff_full_extra([[Te,ne]],n_list_all)
									total += nH3p_ne*From_H3p_pop_coeff_full_extra([[Te,ne]],n_list_all)
									total = total[0]*(ne ** 2)*multiplicative_factor
									# print(total)
									return np.log(total)
								return calculated_emission

							nH_ne, nHp_ne, nHm_ne, nH2_ne, nH2p_ne, nH3p_ne = most_likely_nH_ne_value,1+most_likely_nHm_ne_value-most_likely_nH2p_ne_value-0,most_likely_nHm_ne_value,most_likely_nH2_ne_value,most_likely_nH2p_ne_value,0
							# nH_ne_sigma, nHp_ne_sigma, nHm_ne_sigma, nH2_ne_sigma, nH2p_ne_sigma, nH3p_ne_sigma = 1,1,1,1,1,1
							# residuals = np.sum(((np.log(inverted_profiles_crop_restrict) - fit_Yacora_pop_coeff(merge_ne_prof_multipulse_interp_crop_limited_restrict,merge_Te_prof_multipulse_interp_crop_limited_restrict,T_Hp,T_Hm,T_H2p,ne,multiplicative_factor,excitation_restrict,recombination_restrict)(n_list_all, *fit[0]))) ** 2)
							residuals = np.sum(((inverted_profiles_crop_restrict - np.exp(fit_Yacora_pop_coeff(most_likely_ne_value,most_likely_Te_value,T_Hp,T_Hm,T_H2p,multiplicative_factor,excitation_restrict,recombination_restrict)(n_list_all, nH_ne,nHp_ne,nHm_ne,nH2_ne,nH2p_ne,nH3p_ne)))/inverted_profiles_crop_sigma_restrict) ** 2)

							if my_time_pos in sample_time_step:
								if my_r_pos in sample_radious:
									if to_print:
										# tm.sleep(np.random.random()*10)
										plt.figure(figsize=(20, 10))
										# nH_ne, nHp_ne, nH2p_ne = fit[0]
										# nHp_ne = 1 + nHm_ne - nH2p_ne
										plt.errorbar(n_list_all,inverted_profiles_crop_restrict,yerr=inverted_profiles_crop_sigma_restrict,label='OES')
										plt.plot(n_list_all_full,(excitation_full *  nH_ne).astype('float')*(most_likely_ne_value ** 2),label='direct excitation (ADAS)\n'+r'$H(q) + e^- → H(p>q) + e^-$')
										plt.plot(n_list_all_full,(recombination_full * nHp_ne).astype('float')*(most_likely_ne_value ** 2),label='recombination (ADAS)\n'+r'$H^+ + e^- → H(p) + hν$'+'\n'+r'$H^+ + 2e^- → H(p) + e^-$')
										# plt.plot(n_list_all,nH_ne*From_H_pop_coeff_full([[most_likely_Te_value,ne,nHp_ne*ne,nH_ne*ne]],n_list_all)[0]*(ne ** 2)*multiplicative_factor,label='direct excitation\nH(q) + e- → H(p>q) + e-')
										# plt.plot(n_list_all,nHp_ne*From_Hp_pop_coeff_full([[most_likely_Te_value,ne]],n_list_all)[0]*(ne ** 2)*multiplicative_factor,label='recombination\nH+ + e- → H(p) + hν\nH+ + 2e- → H(p) + e-')
										plt.plot(n_list_all_full,nHm_ne*( From_Hn_with_Hp_pop_coeff_full_extra([[most_likely_Te_value,T_Hp,T_Hm,most_likely_ne_value,nHp_ne*most_likely_ne_value]],n_list_all_full) )[0]*(most_likely_ne_value ** 2)*multiplicative_factor_full,label='H+ mutual neutralisation (Yacora)\n'+r'$H^+ + H^- → H(p) + H(1)$')
										plt.plot(n_list_all_full,nHm_ne*( From_Hn_with_H2p_pop_coeff_full_extra([[most_likely_Te_value,T_H2p,T_Hm,most_likely_ne_value,nH2p_ne*most_likely_ne_value]],n_list_all_full) )[0]*(most_likely_ne_value ** 2)*multiplicative_factor_full,label='H2+ mutual neutralisation (Yacora)\n'+r'${H_2}^+ + H^- → H(p) + H_2$')
										plt.plot(n_list_all_full,nH2_ne*From_H2_pop_coeff_full_extra([[most_likely_Te_value,most_likely_ne_value]],n_list_all_full)[0]*(most_likely_ne_value ** 2)*multiplicative_factor_full,label='H2 dissociation (Yacora)\n'+r'$H_2 + e^- → H(p) + H(1) + e^-$')
										plt.plot(n_list_all_full,nH2p_ne*From_H2p_pop_coeff_full_extra([[most_likely_Te_value,most_likely_ne_value]],n_list_all_full)[0]*(most_likely_ne_value ** 2)*multiplicative_factor_full,label='H2+ dissociation (Yacora)\n'+r'${H_2}^+ + e^- → H(p) + H^+ + e^-$' +'\n'+r'${H_2}^+ + e^- → H(p) + H(1)$')
										plt.plot(n_list_all_full,nH3p_ne*From_H3p_pop_coeff_full_extra([[most_likely_Te_value,most_likely_ne_value]],n_list_all_full)[0]*(most_likely_ne_value ** 2)*multiplicative_factor_full,label='H3+ dissociation (Yacora)\n'+r'${H_3}^+ + e^- → H(p) + H_2$')
										temp = np.exp(fit_Yacora_pop_coeff(most_likely_ne_value,most_likely_Te_value,T_Hp,T_Hm,T_H2p,multiplicative_factor_full,excitation_restrict,recombination_restrict)(n_list_all_full, nH_ne,nHp_ne,nHm_ne,nH2_ne,nH2p_ne,nH3p_ne))
										plt.plot(n_list_all_full,temp,'--',label='total fit')
										plt.plot(n_list_all_full,((excitation_full *  nH_ne).astype('float')*(most_likely_ne_value ** 2) + (recombination_full * nHp_ne).astype('float')*(most_likely_ne_value ** 2)),'+',label='excitation + recombination')
										plt.semilogy()
										plt.legend(loc='best', fontsize='x-small')
										plt.ylim(np.min(inverted_profiles_crop_restrict)/2,max(np.max(inverted_profiles_crop_restrict)*2,np.max(temp)))
										plt.title('lines '+str(n_list_all)+' weights '+str(n_weights)+'\nlocation [time, r]'+ ' [%.3g ms, ' % time_crop[my_time_pos] + ' %.3g mm]' % (1000*r_crop[my_r_pos]) +' , [TS Te, TS ne] '+ ' [%.3g(ML %.3g+/-%.3g) eV, ' %(merge_Te_prof_multipulse_interp_crop_limited_restrict,most_likely_Te_value,sigma_Te) + '%.3g(ML %.3g+/-%.3g) #10^20/m^3]' %(merge_ne_prof_multipulse_interp_crop_limited_restrict,most_likely_ne_value*1e-20,sigma_ne*1e-20) +'\nfit [nH/ne, nH+/ne, nH-/ne, nH2/ne, nH2+/ne, nH3+/ne] , '+ '[%.3g+/-%.3g, ' %(nH_ne,sigma_nH_ne) + '%.3g+/-%.3g, ' %(nHp_ne,sigma_nHp_ne)+ '%.3g+/-%.3g, ' %(nHm_ne,sigma_nHm_ne)+ '%.3g+/-%.3g, ' %(nH2_ne,sigma_nH2_ne)+ '%.3g+/-%.3g, ' %(nH2p_ne,sigma_nH2p_ne)+ '%.3g+/-%.3g]' %(nH3p_ne,sigma_nH3p_ne) +'\nH ionization length %.3gm' % ionization_length_H +'\nH2 ionization length %.3gm' % ionization_length_H2)
										plt.xlabel('Exctited state n')
										plt.ylabel('Line emissivity n->2 [W m^-3]')
										# plt.pause(0.01)
										save_done = 0
										save_index=1
										try:	# All this trie are because of a known issues when saving lots of figure inside of multiprocessor
											plt.savefig(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps', bbox_inches='tight')
										except Exception as e:
											print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(1)+' failed. Reason %s' % e)
											while save_done==0 and save_index<100:
												try:
													plt.savefig(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps', bbox_inches='tight')
													print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(save_index+1)+' successfull')
													save_done=1
												except Exception as e:
													tm.sleep(np.random.random()**2)
													# print(path_where_to_save_everything + mod4 + '/merge'+str(merge_ID_target)+'_global_fit_pass'+str(pass_index)+'_' + str(my_time_pos)+'_'+str(my_r_pos)+ '.eps save try number '+str(save_index+1)+' failed. Reason %s' % e)
													save_index+=1
										plt.close('all')

							end_time = tm.time()
							time_spent = end_time-start_time

							sgna =999 ;print('worker '+str(current_process())+' marker '+str(sgna)+' '+str(domain_index)+' in %.3gmin %.3gsec' %(int(time_spent/60),int(time_spent%60)))
							results = [nH_ne, nHp_ne, nHm_ne, nH2_ne, nH2p_ne, nH3p_ne,residuals,most_likely_Te_value,most_likely_ne_value*1e-20,sigma_Te,sigma_ne,sigma_nH_ne,sigma_nHp_ne,sigma_nHm_ne,sigma_nH2_ne,sigma_nH2p_ne,sigma_nH3p_ne,False,power_balance_data_dict]
							output = calc_stuff_output(my_time_pos,my_r_pos, results)
							try:
								del build_log_PDF,find_correlation,find_sigma,fit_Yacora_pop_coeff
							except:
								try:
									del build_log_PDF,find_sigma,fit_Yacora_pop_coeff
								except:
									del find_sigma,fit_Yacora_pop_coeff
						except Exception as e:
							results = [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,True]
							output = calc_stuff_output(my_time_pos,my_r_pos, results)
							print(traceback.format_exc())
							print('child multiprocessor terminated with error '+str(e)+'worker '+str(current_process())+' '+str(domain_index)+' t=%.3g, r=%.3g' %(my_time_pos,my_r_pos))
						return output	# here I use flexible priors for H2, H2+, H-

					def abortable_worker(func, *arg):
						timeout = timeout_bayesian_search
						p = ThreadPool(1)
						res = p.apply_async(func, args=[arg])
						print('External starting of ' + str(arg[0]))
						start = tm.time()
						try:
							out = res.get(timeout)  # Wait timeout seconds for func to complete.
							print("Succesful end of " +str(arg[0])+ " in %.3g min" %((tm.time() - start)/60))
							return out
						# except multiprocessing.TimeoutError:
						except Exception as e:
							print("External aborting due to timeout " +str(arg[0])+ " in %.3g min" %((tm.time() - start)/60))
							results = [0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,True]
							my_time_pos = arg[1]
							my_r_pos = arg[2]
							output = calc_stuff_output(my_time_pos,my_r_pos, results)
							return output

					if do_only_one_loop==False:
						if ( (not os.path.exists(path_where_to_save_everything + mod4 +'/results.npz') ) or recorded_data_override[0] ):

							all_indexes = []
							global_index = 0
							to_print=collect_power_PDF
							for my_time_pos in range(len(time_crop)):
							# for my_time_pos in sample_time_step:
								for my_r_pos in range(len(r_crop)):
								# for my_r_pos in sample_radious:
									if (merge_Te_prof_multipulse_interp_crop_limited[my_time_pos,my_r_pos]> 0.1 and merge_ne_prof_multipulse_interp_crop_limited[my_time_pos,my_r_pos]> 5e-07):
										all_indexes.append([global_index,my_time_pos,my_r_pos,1,guess,ionization_length_H[my_time_pos, my_r_pos],ionization_length_H2[my_time_pos, my_r_pos],to_print])
										# calc_stuff([global_index,my_time_pos,my_r_pos,1,guess,ionization_length_H,ionization_length_Hm,ionization_length_H2,ionization_length_H2p,to_print])
									global_index+=1

							# if __name__ == '__main__':
							if False:	# old loop that doesn't have a timeout
								try:
									# with Pool(number_cpu_available,maxtasksperchild=1) as pool:
										# pool = Pool(number_cpu_available)
									all_results = []
									for fraction in range(int(np.ceil(len(all_indexes)/(number_cpu_available*4)))):
										with Pool(number_cpu_available,maxtasksperchild=1) as pool:
											print('starting set '+str(fraction)+' of '+str(int(np.ceil(len(all_indexes)/(number_cpu_available*4))-1)) + ', index ' + str(all_indexes[fraction*(number_cpu_available*4)][0]) + ' to ' + str(all_indexes[(fraction+1)*(number_cpu_available*4)-1][0]))
											temp = pool.map(calc_stuff_2, all_indexes[fraction*(number_cpu_available*4):(fraction+1)*(number_cpu_available*4)])
											pool.close()
											pool.join()
											pool.terminate()
											del pool
											all_results.extend(temp)
											print('set '+str(fraction)+' of '+str(int(np.ceil(len(all_indexes)/(number_cpu_available*4))-1))+' done')
										#all_results = [*pool.map(calc_stuff, all_indexes)]
								except Exception as e:
									print('parent multiprocessor terminated with error '+str(e))
									pool.close()
								# pool.close()
									pool.join()
									pool.terminate()
									del pool
							elif True:	# new loop with timeout
								try:
									# with Pool(number_cpu_available,maxtasksperchild=1) as pool:
										# pool = Pool(number_cpu_available)
									all_results = []
									for fraction in range(int(np.ceil(len(all_indexes)/(number_cpu_available*4)))):
										with Pool(number_cpu_available,maxtasksperchild=1) as pool:
											print('starting set '+str(fraction)+' of '+str(int(np.ceil(len(all_indexes)/(number_cpu_available*4))-1)) + ', index ' + str(all_indexes[fraction*(number_cpu_available*4)][0]) + ' to ' + str(all_indexes[(fraction+1)*(number_cpu_available*4)-1][0]))
											temp1 = []
											for f in all_indexes[fraction*(number_cpu_available*4):(fraction+1)*(number_cpu_available*4)]:
												abortable_func = partial(abortable_worker, calc_stuff_2)
												temp1.append(pool.apply_async(abortable_func, args=f))
											pool.close()
											pool.join()
											pool.terminate()
											temp = []
											for value in temp1:
												temp.append(value.get())
											del pool
											all_results.extend(temp)
											print('set '+str(fraction)+' of '+str(int(np.ceil(len(all_indexes)/(number_cpu_available*4))-1))+' done')
										#all_results = [*pool.map(calc_stuff, all_indexes)]
								except Exception as e:
									print('parent multiprocessor terminated with error '+str(e))
									pool.close()
								# pool.close()
									pool.join()
									pool.terminate()
									del pool

							for i in range(len(all_results)):
								if (all_results[i].results)[17]==False:
									nH_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[0]
									nHp_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[1]
									nHm_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[2]
									nH2_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[3]
									nH2p_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[4]
									nH3p_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[5]
									residuals_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[6]
									Te_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[7]
									ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[8]
									sigma_Te_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[9]
									sigma_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[10]
									sigma_nH_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[11]
									sigma_nHp_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[12]
									sigma_nHm_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[13]
									sigma_nH2_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[14]
									sigma_nH2p_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[15]
									sigma_nH3p_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[16]
									# power_balance_data[int(all_results[i].my_time_pos)*np.shape(nH_ne_all)[1] + int(all_results[i].my_r_pos)] = (all_results[i].results)[18]
									power_balance_data_dict[int(all_results[i].my_time_pos)*np.shape(nH_ne_all)[1] + int(all_results[i].my_r_pos)] = (all_results[i].results)[18]


							# all_indexes = []
							# global_index = 0
							# to_print=True
							# for my_time_pos in range(len(time_crop)):
							# 	if my_time_pos in sample_time_step:
							# 		for my_r_pos in range(len(r_crop)):
							# 			if my_r_pos in sample_radious:
							# 				calc_stuff([global_index,my_time_pos,my_r_pos,1,guess,ionization_length_H,ionization_length_Hm,ionization_length_H2,ionization_length_H2p,to_print])


							np.savez_compressed(path_where_to_save_everything + mod4 +'/results',nH_ne_all=nH_ne_all,nHp_ne_all=nHp_ne_all,nHm_ne_all=nHm_ne_all,nH2_ne_all=nH2_ne_all,nH2p_ne_all=nH2p_ne_all,nH3p_ne_all=nH3p_ne_all,residuals_all=residuals_all,Te_all=Te_all,ne_all=ne_all,sigma_Te_all=sigma_Te_all,sigma_ne_all=sigma_ne_all,sigma_nH_ne_all=sigma_nH_ne_all,sigma_nHp_ne_all=sigma_nHp_ne_all,sigma_nHm_ne_all=sigma_nHm_ne_all,sigma_nH2_ne_all=sigma_nH2_ne_all,sigma_nH2p_ne_all=sigma_nH2p_ne_all,sigma_nH3p_ne_all=sigma_nH3p_ne_all)
							if collect_power_PDF:
								with open(path_where_to_save_everything + mod4 +'/power_balance.data', 'wb') as filehandle:
								    pickle.dump(power_balance_data_dict, filehandle)
							else:
								with open(path_where_to_save_everything + mod4 +'/power_balance_no_PDF.data', 'wb') as filehandle:
								    pickle.dump(power_balance_data_dict, filehandle)

						else:
							nH_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['nH_ne_all']
							nHp_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['nHp_ne_all']
							nHm_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['nHm_ne_all']
							nH2_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['nH2_ne_all']
							nH2p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['nH2p_ne_all']
							nH3p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['nH3p_ne_all']
							residuals_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['residuals_all']
							try:
								Te_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['Te_all']
								ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['ne_all']
							except:
								print('Bayesian Te, ne not present')
								Te_all = cp.deepcopy(merge_Te_prof_multipulse_interp_crop_limited)
								ne_all = cp.deepcopy(merge_ne_prof_multipulse_interp_crop_limited)
							try:
								sigma_Te_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['sigma_Te_all']
								sigma_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['sigma_ne_all']
								sigma_nH_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['sigma_nH_ne_all']
								sigma_nHp_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['sigma_nHp_ne_all']
								sigma_nHm_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['sigma_nHm_ne_all']
								sigma_nH2_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['sigma_nH2_ne_all']
								sigma_nH2p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['sigma_nH2p_ne_all']
								sigma_nH3p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results.npz')['sigma_nH3p_ne_all']
							except:
								print('sigma not present')
							try:
								with open(path_where_to_save_everything + mod4 +'/power_balance.data', 'rb') as filehandle:
								    power_balance_data_dict = pickle.load(filehandle)
							except:
								print('power_balance_data_dict not present')

						global_pass = 2
						exec(open("/home/ffederic/work/Collaboratory/test/experimental_data/post_process_PSI_parameter_search_1_Yacora_plots_temp.py").read())

					# if True: # I don't think now, given the indetermination in how emission profiles fit the measurements, that second passes are useful. 04/06/2020 I do it for the locations with low Te, ne

					if ( (not os.path.exists(path_where_to_save_everything + mod4 +'/results2.npz') ) or recorded_data_override[1] ):


						all_indexes = []
						global_index = 0
						to_print=collect_power_PDF
						for my_time_pos in range(len(time_crop)):
							for my_r_pos in range(len(r_crop)):
								if (merge_Te_prof_multipulse_interp_crop[my_time_pos,my_r_pos]> 0 and merge_ne_prof_multipulse_interp_crop[my_time_pos,my_r_pos]> 0):
									if nHm_ne_all[my_time_pos,my_r_pos]==0:
										all_indexes.append([global_index,my_time_pos,my_r_pos,2,guess,ionization_length_H[my_time_pos, my_r_pos],ionization_length_H2[my_time_pos, my_r_pos],to_print])
								global_index+=1

						# if __name__ == '__main__':
						try:
							# with Pool(number_cpu_available,maxtasksperchild=1) as pool:
								# pool = Pool(number_cpu_available)
							all_results = []
							for fraction in range(int(np.ceil(len(all_indexes)/(number_cpu_available*4)))):
								with Pool(number_cpu_available,maxtasksperchild=1) as pool:
									print('starting set '+str(fraction)+' of '+str(int(np.ceil(len(all_indexes)/(number_cpu_available*4))-1)))
									temp = pool.map(calc_stuff_2, all_indexes[fraction*(number_cpu_available*4):(fraction+1)*(number_cpu_available*4)])
									pool.close()
									pool.join()
									pool.terminate()
									del pool
									all_results.extend(temp)
									print('set '+str(fraction)+' of '+str(int(np.ceil(len(all_indexes)/(number_cpu_available*4))-1))+' done')
								#all_results = [*pool.map(calc_stuff, all_indexes)]
						except Exception as e:
							print('parent multiprocessor terminated with error '+str(e))
							pool.close()
						# pool.close()
							pool.join()
							pool.terminate()
							del pool

						for i in range(len(all_results)):
							if (all_results[i].results)[17]==False:
								nH_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[0]
								nHp_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[1]
								nHm_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[2]
								nH2_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[3]
								nH2p_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[4]
								nH3p_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[5]
								residuals_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[6]
								Te_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[7]
								ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[8]
								sigma_Te_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[9]
								sigma_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[10]
								sigma_nH_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[11]
								sigma_nHp_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[12]
								sigma_nHm_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[13]
								sigma_nH2_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[14]
								sigma_nH2p_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[15]
								sigma_nH3p_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[16]
								# power_balance_data[int(all_results[i].my_time_pos)*np.shape(nH_ne_all)[1] + int(all_results[i].my_r_pos)] = (all_results[i].results)[18]
								power_balance_data_dict[int(all_results[i].my_time_pos)*np.shape(nH_ne_all)[1] + int(all_results[i].my_r_pos)] = (all_results[i].results)[18]

						np.savez_compressed(path_where_to_save_everything + mod4 +'/results2',nH_ne_all=nH_ne_all,nHp_ne_all=nHp_ne_all,nHm_ne_all=nHm_ne_all,nH2_ne_all=nH2_ne_all,nH2p_ne_all=nH2p_ne_all,nH3p_ne_all=nH3p_ne_all,residuals_all=residuals_all,Te_all=Te_all,ne_all=ne_all,sigma_Te_all=sigma_Te_all,sigma_ne_all=sigma_ne_all,sigma_nH_ne_all=sigma_nH_ne_all,sigma_nHp_ne_all=sigma_nHp_ne_all,sigma_nHm_ne_all=sigma_nHm_ne_all,sigma_nH2_ne_all=sigma_nH2_ne_all,sigma_nH2p_ne_all=sigma_nH2p_ne_all,sigma_nH3p_ne_all=sigma_nH3p_ne_all)
						if collect_power_PDF:
							with open(path_where_to_save_everything + mod4 +'/power_balance2.data', 'wb') as filehandle:
							    pickle.dump(power_balance_data_dict, filehandle)
						else:
							with open(path_where_to_save_everything + mod4 +'/power_balance2_no_PDF.data', 'wb') as filehandle:
							    pickle.dump(power_balance_data_dict, filehandle)

					else:
						nH_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['nH_ne_all']
						nHp_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['nHp_ne_all']
						nHm_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['nHm_ne_all']
						nH2_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['nH2_ne_all']
						nH2p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['nH2p_ne_all']
						nH3p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['nH3p_ne_all']
						residuals_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['residuals_all']
						try:
							Te_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['Te_all']
							ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['ne_all']
						except:
							print('Bayesian Te, ne not present')
							Te_all = cp.deepcopy(merge_Te_prof_multipulse_interp_crop_limited)
							ne_all = cp.deepcopy(merge_ne_prof_multipulse_interp_crop_limited)
						try:
							sigma_Te_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['sigma_Te_all']
							sigma_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['sigma_ne_all']
							sigma_nH_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['sigma_nH_ne_all']
							sigma_nHp_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['sigma_nHp_ne_all']
							sigma_nHm_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['sigma_nHm_ne_all']
							sigma_nH2_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['sigma_nH2_ne_all']
							sigma_nH2p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['sigma_nH2p_ne_all']
							sigma_nH3p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['sigma_nH3p_ne_all']
						except:
							print('sigma not present')
						try:
							with open(path_where_to_save_everything + mod4 +'/power_balance2.data', 'rb') as filehandle:
							    power_balance_data_dict = pickle.load(filehandle)
						except:
							print('power_balance_data_dict not present')

					# this will be done later, no need to do it here
					# global_pass = 3
					# exec(open("/home/ffederic/work/Collaboratory/test/experimental_data/post_process_PSI_parameter_search_1_Yacora_plots_temp.py").read())


					# cycle always done, just have the chance to run points that failed before, maybe because of memory errors.
					if completion_loop_done:

						all_indexes = []
						global_index = 0
						to_print=collect_power_PDF
						for my_time_pos in range(len(time_crop)):
							for my_r_pos in range(len(r_crop)):
								if (merge_Te_prof_multipulse_interp_crop[my_time_pos,my_r_pos]> 0 and merge_ne_prof_multipulse_interp_crop[my_time_pos,my_r_pos]> 0):
									if nHm_ne_all[my_time_pos,my_r_pos]==0:
										all_indexes.append([global_index,my_time_pos,my_r_pos,2,guess,ionization_length_H[my_time_pos, my_r_pos],ionization_length_H2[my_time_pos, my_r_pos],to_print])
								global_index+=1

						# if __name__ == '__main__':
						try:
							# with Pool(number_cpu_available,maxtasksperchild=1) as pool:
								# pool = Pool(number_cpu_available)
							all_results = []
							for fraction in range(int(np.ceil(len(all_indexes)/(number_cpu_available*4)))):
								with Pool(number_cpu_available,maxtasksperchild=1) as pool:
									print('starting set '+str(fraction)+' of '+str(int(np.ceil(len(all_indexes)/(number_cpu_available*4))-1)))
									temp = pool.map(calc_stuff_2, all_indexes[fraction*(number_cpu_available*4):(fraction+1)*(number_cpu_available*4)])
									pool.close()
									pool.join()
									pool.terminate()
									del pool
									all_results.extend(temp)
									print('set '+str(fraction)+' of '+str(int(np.ceil(len(all_indexes)/(number_cpu_available*4))-1))+' done')
								#all_results = [*pool.map(calc_stuff, all_indexes)]
						except Exception as e:
							print('parent multiprocessor terminated with error '+str(e))
							pool.close()
						# pool.close()
							pool.join()
							pool.terminate()
							del pool

						for i in range(len(all_results)):
							if (all_results[i].results)[17]==False:
								nH_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[0]
								nHp_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[1]
								nHm_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[2]
								nH2_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[3]
								nH2p_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[4]
								nH3p_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[5]
								residuals_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[6]
								Te_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[7]
								ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[8]
								sigma_Te_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[9]
								sigma_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[10]
								sigma_nH_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[11]
								sigma_nHp_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[12]
								sigma_nHm_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[13]
								sigma_nH2_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[14]
								sigma_nH2p_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[15]
								sigma_nH3p_ne_all[int(all_results[i].my_time_pos), int(all_results[i].my_r_pos)] = (all_results[i].results)[16]
								# power_balance_data[int(all_results[i].my_time_pos)*np.shape(nH_ne_all)[1] + int(all_results[i].my_r_pos)] = (all_results[i].results)[18]
								power_balance_data_dict[int(all_results[i].my_time_pos)*np.shape(nH_ne_all)[1] + int(all_results[i].my_r_pos)] = (all_results[i].results)[18]

						np.savez_compressed(path_where_to_save_everything + mod4 +'/results2',nH_ne_all=nH_ne_all,nHp_ne_all=nHp_ne_all,nHm_ne_all=nHm_ne_all,nH2_ne_all=nH2_ne_all,nH2p_ne_all=nH2p_ne_all,nH3p_ne_all=nH3p_ne_all,residuals_all=residuals_all,Te_all=Te_all,ne_all=ne_all,sigma_Te_all=sigma_Te_all,sigma_ne_all=sigma_ne_all,sigma_nH_ne_all=sigma_nH_ne_all,sigma_nHp_ne_all=sigma_nHp_ne_all,sigma_nHm_ne_all=sigma_nHm_ne_all,sigma_nH2_ne_all=sigma_nH2_ne_all,sigma_nH2p_ne_all=sigma_nH2p_ne_all,sigma_nH3p_ne_all=sigma_nH3p_ne_all)
						if collect_power_PDF:
							with open(path_where_to_save_everything + mod4 +'/power_balance2.data', 'wb') as filehandle:
							    pickle.dump(power_balance_data_dict, filehandle)
						else:
							with open(path_where_to_save_everything + mod4 +'/power_balance2_no_PDF.data', 'wb') as filehandle:
							    pickle.dump(power_balance_data_dict, filehandle)

					else:
						nH_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['nH_ne_all']
						nHp_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['nHp_ne_all']
						nHm_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['nHm_ne_all']
						nH2_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['nH2_ne_all']
						nH2p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['nH2p_ne_all']
						nH3p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['nH3p_ne_all']
						residuals_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['residuals_all']
						try:
							Te_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['Te_all']
							ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['ne_all']
						except:
							print('Bayesian Te, ne not present')
							Te_all = cp.deepcopy(merge_Te_prof_multipulse_interp_crop_limited)
							ne_all = cp.deepcopy(merge_ne_prof_multipulse_interp_crop_limited)
						try:
							sigma_Te_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['sigma_Te_all']
							sigma_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['sigma_ne_all']
							sigma_nH_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['sigma_nH_ne_all']
							sigma_nHp_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['sigma_nHp_ne_all']
							sigma_nHm_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['sigma_nHm_ne_all']
							sigma_nH2_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['sigma_nH2_ne_all']
							sigma_nH2p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['sigma_nH2p_ne_all']
							sigma_nH3p_ne_all = np.load(path_where_to_save_everything + mod4 +'/results2.npz')['sigma_nH3p_ne_all']
						except:
							print('sigma not present')
						try:
							with open(path_where_to_save_everything + mod4 +'/power_balance2.data', 'rb') as filehandle:
							    power_balance_data_dict = pickle.load(filehandle)
						except:
							print('power_balance_data_dict not present')

					global_pass = 3
					exec(open("/home/ffederic/work/Collaboratory/test/experimental_data/post_process_PSI_parameter_search_1_Yacora_plots_temp.py").read())
