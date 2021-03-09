import numpy as np
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())
# import matplotlib.pyplot as plt
#import .functions
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
from functions.spectools import rotate,do_tilt, binData, get_angle, get_tilt,get_angle_2
from functions.Calibrate import do_waveL_Calib, do_Intensity_Calib
from functions.fabio_add import find_nearest_index,multi_gaussian,all_file_names,load_dark,find_index_of_file,get_metadata,movie_from_data,get_angle_no_lines,do_tilt_no_lines,four_point_transform,fix_minimum_signal,fix_minimum_signal2,get_bin_and_interv_no_lines,examine_current_trace,shift_between_TS_and_power_source
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
from scipy.signal import find_peaks, peak_prominences as get_proms
from multiprocessing import Pool,cpu_count
number_cpu_available = cpu_count()
print('Number of cores available: '+str(number_cpu_available))


os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data/functions')
print(os.path.abspath(os.getcwd()))
import pyradi.ryptw as ryptw

fdir = '/home/ffederic/work/Collaboratory/test/experimental_data'
df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
df_settings = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_3.csv',index_col=0)

try:
	figure_index+=3
except:
	figure_index=0
	plt.close('all')
merge_ID_target = 87
all_j=find_index_of_file(merge_ID_target,df_settings,df_log,only_OES=True)
path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) #+ '_back'

merge_time_window = [-1,2]
power_pulse_shape_time_dependent = []
power_pulse_shape_time_dependent_std = []
steady_state_power = []
steady_state_power_std = []
for j in all_j:
	(merge_folder,sequence,fname_current_trace,SS_current) = df_log.loc[j,['folder','sequence','current_trace_file','I']]
	bad_pulses,first_good_pulse,first_pulse,last_pulse,miss_pulses,double_pulses,good_pulses, time_of_pulses, energy_per_pulse,duration_per_pulse,median_energy_good_pulses,median_duration_good_pulses,mean_peak_shape,mean_peak_std,mean_steady_state_power,mean_steady_state_power_std,time_resolution = examine_current_trace(fdir+'/'+merge_folder+'/'+"{0:0=2d}".format(sequence)+'/', fname_current_trace, df_log.loc[j, ['number_of_pulses']][0],want_the_power_per_pulse=True,want_the_mean_power_profile=True,SS_current=SS_current)
	# current_traces = pd.read_csv(fdir+'/'+merge_folder+'/'+"{0:0=2d}".format(sequence)+'/'+ fname_current_trace+'.tsf',index_col=False, delimiter='\t')
	# current_traces_time = current_traces['Time [s]']
	# current_traces_total = current_traces['I_Src_AC [A]']	# this is the current of the Rogowski coil, and it is only the current on top of the SS. the SS component here have to be removed.
	# voltage_traces_total = current_traces['U_Src_DC [V]']
	# plt.figure()
	# plt.plot(current_traces_time,-current_traces_total*voltage_traces_total/np.max(-current_traces_total*voltage_traces_total),label='power')
	# plt.plot(current_traces_time,current_traces_total/np.max(current_traces_total),label='current')
	# plt.plot(current_traces_time,-voltage_traces_total/np.max(-voltage_traces_total),label='voltage')
	# plt.legend(loc='best')
	# plt.grid()
	# plt.pause(0.001)
	power_pulse_shape_time_dependent.append(mean_peak_shape)
	power_pulse_shape_time_dependent_std.append(mean_peak_std)
	steady_state_power.append(mean_steady_state_power)
	steady_state_power_std.append(mean_steady_state_power_std)
power_pulse_shape_time_dependent = np.mean(power_pulse_shape_time_dependent,axis=0)
power_pulse_shape_time_dependent_std = np.sum(0.25*np.array(power_pulse_shape_time_dependent_std)**2,axis=0)**0.5
steady_state_power = np.mean(steady_state_power,axis=0)
steady_state_power_std = np.sum(0.25*np.array(steady_state_power_std)**2,axis=0)**0.5
power_pulse_shape =power_pulse_shape_time_dependent + steady_state_power
power_pulse_shape_std = (power_pulse_shape_time_dependent_std**2+steady_state_power_std**2)**0.5

path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) #+ '_back'
merge_Te_prof_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_Te_prof_multipulse']
merge_dTe_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_dTe_multipulse']
merge_ne_prof_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_ne_prof_multipulse']
merge_dne_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_dne_multipulse']
merge_time_original = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_time']
# new_timesteps = np.load(path_where_to_save_everything + '/merge' + str(merge_ID_target) + '_new_timesteps.npy')
new_timesteps = np.linspace(-0.5,1.5,num=41)
dt = np.nanmedian(np.diff(new_timesteps))


spatial_factor = 1
time_shift_factor = 0

dx = 1.06478505470992 / 1e3	# 10/02/2020 from	Calculate magnification_FF.xlsx
xx = np.arange(40) * dx  # m
xn = np.linspace(0, max(xx), 1000)
number_of_radial_divisions = 21
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
	if np.sum(yy>0)==0:
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




left = power_pulse_shape_time_dependent.argmax() - int(2e-3/time_resolution)
right = power_pulse_shape_time_dependent.argmax() + int(4e-3/time_resolution)

interpolation = interp1d((np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000)[left:right],np.cumsum(power_pulse_shape_time_dependent[left:right]))
t_temp = np.linspace(np.min((np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000)[left:right]),np.max((np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000)[left:right]),num=1000000)
current_trace_start = t_temp[np.abs(interpolation(t_temp) - np.max(np.cumsum(power_pulse_shape_time_dependent[left:right]))/50).argmin()]
# current_trace_start = (np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000)[current_trace_start]

plt.figure()
# plt.plot(np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000,power_pulse_shape_time_dependent)
plt.plot((np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000)[left:right],np.cumsum(power_pulse_shape_time_dependent[left:right]))
plt.plot((np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000)[left:right][[0,-1]],[np.max(np.cumsum(power_pulse_shape_time_dependent[left:right]))/50]*2,'--',label='1/50 of peak')
plt.xlabel('arbitrary time [ms]')
plt.ylabel('cumulative power transmitted by the power source')
plt.legend(loc='best')
plt.grid()
plt.pause(0.01)


start_r = np.abs(r - 0).argmin()
end_r = np.abs(r - 5).argmin() + 1
r_crop = r[start_r:end_r]
area = 2*np.pi*(np.abs(TS_r_new) + np.median(np.diff(TS_r_new))/2) * np.median(np.diff(TS_r_new))/2	# the last /2 is due to the fact that I didn't do the left/right average so I'm double counting the area
# energy_flow = np.sum(merge_ne_prof_multipulse*(13.6 + merge_Te_prof_multipulse) * area,axis=-1)
hydrogen_mass = 1.008*1.660*1e-27	# kg
boltzmann_constant_J = 1.380649e-23	# J/K
eV_to_K = 8.617333262145e-5	# eV/K
adiabatic_collisional_velocity = ((merge_Te_prof_multipulse + 5/3 *merge_Te_prof_multipulse*eV_to_K)/eV_to_K*boltzmann_constant_J/hydrogen_mass)**0.5
homogeneous_mach_number = 1	# as an approximate assumption I assume sonic flow
energy_flow = np.sum(homogeneous_mach_number*adiabatic_collisional_velocity*merge_ne_prof_multipulse*(0.5*hydrogen_mass*(homogeneous_mach_number*adiabatic_collisional_velocity)**2 + (5*merge_Te_prof_multipulse+13.6 + 2.2)/eV_to_K*boltzmann_constant_J) * area,axis=-1)
plt.figure()
# plt.plot(merge_time_original,energy_flow)
# plt.plot(merge_time_original[[0,-1]],[np.median(energy_flow[merge_time_original<0])]*2,'--')
# plt.pause(0.001)
plt.plot(merge_time_original,np.cumsum(energy_flow-np.mean(energy_flow[merge_time_original<0])))
plt.plot(merge_time_original[[0,-1]],[np.cumsum(energy_flow-np.mean(energy_flow[merge_time_original<0])).max()/50]*2,'--',label='1/50 of peak')
plt.xlabel('arbitrary time [ms]')
plt.ylabel('cumulative energy transported by plasma')
plt.legend(loc='best')
plt.grid()
plt.pause(0.01)

interpolation = interp1d(merge_time_original,np.cumsum(energy_flow-np.mean(energy_flow[merge_time_original<0])))
t_temp = np.linspace(np.min(merge_time_original),np.max(merge_time_original),num=1000000)
TS_start = t_temp[np.abs(interpolation(t_temp) - np.cumsum(energy_flow-np.mean(energy_flow[merge_time_original<0])).max()/50).argmin()]

offset_current_trace = current_trace_start-TS_start



plt.figure()
# plt.plot(np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000,power_pulse_shape_time_dependent)
plt.plot((np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000)[left:right]-current_trace_start,np.cumsum(power_pulse_shape_time_dependent[left:right])/np.max(np.cumsum(power_pulse_shape_time_dependent[left:right])),label='energy generated from power source')
plt.plot((np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000)[left:right][[0,-1]]-current_trace_start,[1/50]*2,'--')

plt.plot(merge_time_original-TS_start,np.cumsum(energy_flow-np.mean(energy_flow[merge_time_original<0]))/np.max(np.cumsum(energy_flow-np.mean(energy_flow[merge_time_original<0]))),label='energy transported by plasma (TS)')
plt.plot(merge_time_original[[0,-1]]-TS_start,[1/50]*2,'--')
plt.xlabel('Time from 1/50 of maximum energy transferred [s]')
plt.ylabel('cumulative energy transferred [au]')
plt.grid()
plt.title('merge '+str(merge_ID_target)+'\nshift to remove from current trace to match TS = %.5gms' %(offset_current_trace))
plt.legend(loc='best')
plt.pause(0.01)

plt.figure()
# plt.plot(np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000,power_pulse_shape_time_dependent)
plt.plot((np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000)[left:right]-current_trace_start,power_pulse_shape_time_dependent[left:right]/np.max((power_pulse_shape_time_dependent[left:right])),label='energy generated from power source')
plt.plot((np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000)[left:right][[0,-1]]-current_trace_start,[1/50]*2,'--')

plt.plot(merge_time_original-TS_start,(energy_flow-np.mean(energy_flow[merge_time_original<0]))/np.max((energy_flow-np.mean(energy_flow[merge_time_original<0]))),label='energy transported by plasma (TS)')
plt.plot(merge_time_original[[0,-1]]-TS_start,[1/50]*2,'--')
plt.xlabel('Time from 1/50 of maximum energy transferred [s]')
plt.ylabel('relative energy transferred [au]')
plt.grid()
plt.title('merge '+str(merge_ID_target)+'\nshift to remove from current trace to match TS = %.5gms' %(offset_current_trace))
plt.legend(loc='best')
plt.pause(0.01)


target_chamber_pressure_all = []
power_pulse_shape_time_dependent_all = []
power_pulse_shape_time_dependent_std_all = []
steady_state_power_all = []
steady_state_power_std_all = []
# merge_ID_target_all = [95,89,88,87,86,85]
# merge_ID_target_all = [99,98,96,97]
merge_ID_target_all = [101,75,76,77,78,79]
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro','paleturquoise']
plt.figure()
for i_merge_ID_target,merge_ID_target in enumerate(merge_ID_target_all):
	print('merge '+str(merge_ID_target))
	all_j=find_index_of_file(merge_ID_target,df_settings,df_log,only_OES=True)
	target_chamber_pressure = []
	target_OES_distance = []
	magnetic_field = []
	feed_rate_SLM = []
	power_pulse_shape_time_dependent = []
	power_pulse_shape_time_dependent_std = []
	steady_state_power = []
	steady_state_power_std = []
	for j in all_j:
		target_chamber_pressure.append(df_log.loc[j,['p_n [Pa]']])
		target_OES_distance.append(df_log.loc[j,['T_axial']])
		feed_rate_SLM.append(df_log.loc[j,['Seed']])
		magnetic_field.append(df_log.loc[j,['B']])
		(merge_folder,sequence,fname_current_trace) = df_log.loc[j,['folder','sequence','current_trace_file']]
		bad_pulses,first_good_pulse,first_pulse,last_pulse,miss_pulses,double_pulses,good_pulses, time_of_pulses, energy_per_pulse,duration_per_pulse,median_energy_good_pulses,median_duration_good_pulses,mean_peak_shape,mean_peak_std,mean_steady_state_power,mean_steady_state_power_std,time_resolution = examine_current_trace(fdir+'/'+merge_folder+'/'+"{0:0=2d}".format(sequence)+'/', fname_current_trace, df_log.loc[j, ['number_of_pulses']][0],want_the_power_per_pulse=True,want_the_mean_power_profile=True)
		power_pulse_shape_time_dependent.append(mean_peak_shape)
		power_pulse_shape_time_dependent_std.append(mean_peak_std)
		steady_state_power.append(mean_steady_state_power)
		steady_state_power_std.append(mean_steady_state_power_std)
		time_source_power = np.arange(len(mean_peak_shape))*time_resolution*1000
		# time_source_power = time_source_power[np.abs(time_source_power-49.5).argmin():np.abs(time_source_power-51.5).argmin()]
		plt.errorbar(time_source_power[np.abs(time_source_power-49.5).argmin():np.abs(time_source_power-51.5).argmin()], 0.92*(mean_peak_shape+mean_steady_state_power)[np.abs(time_source_power-49.5).argmin():np.abs(time_source_power-51.5).argmin()],yerr=0.92*((mean_peak_std**2+mean_steady_state_power_std**2)**0.5)[np.abs(time_source_power-49.5).argmin():np.abs(time_source_power-51.5).argmin()],linestyle='--',color=color[i_merge_ID_target],label='ID %.3g, Pressure %.3g, tot energy %.3g' %(j,df_log.loc[j,['p_n [Pa]']],np.sum((mean_peak_shape[np.abs(time_source_power-49.5).argmin():np.abs(time_source_power-51.5).argmin()])*0.92)*time_resolution));
	target_chamber_pressure = np.nanmean(target_chamber_pressure)	# Pa
	target_OES_distance = np.nanmean(target_OES_distance)	# Pa
	feed_rate_SLM = np.nanmean(feed_rate_SLM)	# SLM
	magnetic_field = np.nanmean(magnetic_field)	# T
	# power_pulse_shape = np.sum(np.divide(power_pulse_shape,power_pulse_shape_std),axis=0)/np.sum(np.divide(1,power_pulse_shape_std),axis=0)
	power_pulse_shape_time_dependent = np.mean(power_pulse_shape_time_dependent,axis=0)
	power_pulse_shape_time_dependent_std = np.sum(0.25*np.array(power_pulse_shape_time_dependent_std)**2,axis=0)**0.5
	# steady_state_power = np.sum(np.divide(steady_state_power,steady_state_power_std),axis=0)/np.sum(np.divide(1,power_pulse_shape_std),axis=0)
	steady_state_power = np.mean(steady_state_power,axis=0)
	steady_state_power_std = np.sum(0.25*np.array(steady_state_power_std)**2,axis=0)**0.5
	power_pulse_shape =power_pulse_shape_time_dependent + steady_state_power
	power_pulse_shape_std = (power_pulse_shape_time_dependent_std**2+steady_state_power_std**2)**0.5
	power_pulse_shape *=0.92	# 92% efficiency from Morgan 2014
	power_pulse_shape_std *=0.92	# 92% efficiency from Morgan 2014
	steady_state_power *= 0.92	# 92% efficiency from Morgan 2014
	steady_state_power_std *= 0.92	# 92% efficiency from Morgan 2014
	time_source_power = np.arange(len(power_pulse_shape))*time_resolution*1000
	power_pulse_shape = power_pulse_shape[np.abs(time_source_power-49.5).argmin():np.abs(time_source_power-51.5).argmin()]	# I cut around the pulse not to look at what happens before and after
	power_pulse_shape_std = power_pulse_shape_std[np.abs(time_source_power-49.5).argmin():np.abs(time_source_power-51.5).argmin()]
	time_source_power = time_source_power[np.abs(time_source_power-49.5).argmin():np.abs(time_source_power-51.5).argmin()]
	plt.errorbar(time_source_power, power_pulse_shape,yerr=power_pulse_shape_std,color=color[i_merge_ID_target],label='Pressure %.3g, tot energy %.3g' %(target_chamber_pressure,np.sum(power_pulse_shape)*time_resolution-steady_state_power*len(power_pulse_shape)*time_resolution));
plt.grid()
plt.xlabel('arbitrary time [ms]')
plt.ylabel('power [W]')
plt.title('Comparison of power source profile for pressure scan at %.3gT' %(magnetic_field))
plt.legend(loc='best', fontsize='xx-small')
plt.pause(0.01)
