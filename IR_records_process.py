import numpy as np
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())
# import matplotlib.pyplot as pl#import .functions
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
from functions.spectools import rotate,do_tilt, binData, get_angle, get_tilt,get_angle_2
from functions.Calibrate import do_waveL_Calib, do_Intensity_Calib
from functions.fabio_add import find_nearest_index,multi_gaussian,all_file_names,load_dark,find_index_of_file,get_metadata,movie_from_data,get_angle_no_lines,do_tilt_no_lines,four_point_transform,fix_minimum_signal,fix_minimum_signal2,get_bin_and_interv_no_lines,examine_current_trace,energy_flow_from_TS_at_sound_speed
from functions.GetSpectrumGeometry import getGeom
from functions.SpectralFit import doSpecFit_single_frame
from functions.GaussFitData import doLateralfit_time_tependent
import collections

import os,sys
from PIL import Image
import xarray as xr
import pandas as pd
import copy
from uncertainties.unumpy import exp,nominal_values,std_devs,erf
from uncertainties import ufloat,unumpy,correlated_values
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.signal import find_peaks, peak_prominences as get_proms
from multiprocessing import Pool,cpu_count
number_cpu_available = cpu_count()
print('Number of cores available: '+str(number_cpu_available))
import datetime

os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data/functions')
print(os.path.abspath(os.getcwd()))
import pyradi.ryptw as ryptw

# to calculate the Hessian and consequently the covariance matrix from a fit with scipy.optimize.fmin_l_bfgs_b
# see https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/DAVIES1/rd_bhatt_cvonline/node9.html
import numdifftools as nd

# to show the line where it fails
import sys, traceback, logging
logging.basicConfig(level=logging.ERROR)

fdir = '/home/ffederic/work/Collaboratory/test/experimental_data'
df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
df_settings = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_3.csv',index_col=0)
figure_index=0

exec(open("/home/ffederic/work/Collaboratory/test/experimental_data/IR_records_process_preamble.py").read())


temp=[]
# merge_id_all = [17,18,19,20,21,24,32,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85, 95, 86, 87, 89, 90, 91, 92, 93, 94, 96, 97, 98, 88, 99,101,103,104]
# merge_id_all = [66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85, 95, 86, 87, 89, 90, 91, 92, 93, 94, 96, 97, 98, 88, 99,101,103]
# merge_id_all = [66,67,68,69,70,71,72,73,74,75,76]
# merge_id_all = [85, 86, 87, 89, 90, 91, 92, 93, 94, 96, 97, 98, 88, 99, 95]
# merge_id_all = [85, 95, 104]
# merge_id_all = [75,104]
merge_id_all = [80]
merge_id_all = [21]
for merge_id in merge_id_all:
	temp.extend(find_index_of_file(merge_id,df_settings,df_log,only_OES=False))
all_j=[]
for i in range(len(df_log)-1):
	IR_trace,IR_reference = df_log.loc[i,['IR_trace','IR_reference']]
	# if (isinstance(IR_trace, str) and np.isnan(df_log.loc[i,['DT_pulse']][0])):
	if (isinstance(IR_trace, str) and (i in temp)):
		# if i<=265:
		all_j.append(i)


# all_j = [402]
# # all_j = [227,228]
# all_j = [159,163,166,174,186]
# all_j = [104,105,106,107,108]
# all_j = [396,397,398,399,400,401,402,403]
# all_j = [227,232,247,250]
# all_j = [*np.arange(267,270+1),*np.arange(272,275+1),*np.arange(277,286+1),393,394,*np.arange(396,403+1)]

# all_j = np.flip(all_j,axis=0)

# for j in np.flip(all_j,axis=0):
# for j in all_j:
for j in all_j:
# for j in np.flip(all_j[1::2],axis=0):
# def calc_stuff(arg):
# 	j = arg[1]
	print('analysing item n '+str(j)+' of '+str(all_j))

	df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
	(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
	(IR_trace,IR_reference,IR_shape,magnetic_field,target_OES_distance,target_chamber_pressure,capacitor_voltage,SS_current) = df_log.loc[j,['IR_trace','IR_reference','IR_shape','B','T_axial','p_n [Pa]','Vc','I']]
	(CB_to_OES_initial_delay,incremental_step,number_of_pulses) = df_log.loc[j,['CB_to_OES_initial_delay','incremental_step','number_of_pulses']]
	number_of_pulses = int(number_of_pulses)
	(folder,date,sequence,untitled,target_material,fname_current_trace,first_pulse_at_this_frame,bad_pulses_indexes,bad_pulses_indexes_extra) = df_log.loc[j,['folder','date','sequence','untitled','Target','current_trace_file','first_pulse_at_this_frame','bad_pulses_indexes','bad_pulses_indexes_extra']]
	sequence = int(sequence)
	merge_found = 0
	for i in range(len(df_settings)):
		a,b = df_settings.loc[i,['Hb','merge_ID']]
		if a==j:
			merge_ID=b
			merge_found=1
			print('Equivalent to merge '+str(merge_ID))
			break

	pre_title = 'merge %.3g, B=%.3gT, pos=%.3gmm, P=%.3gPa, ELMen=%.3gJ, target: %.100s\n' %(merge_ID,magnetic_field,target_OES_distance,target_chamber_pressure,0.5*(capacitor_voltage**2)*150e-6,target_material)
	if merge_found:
		path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID)
	else:
		path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/'+"{0:0=2g}".format(sequence)+'/IR_camera'
	figure_index = 0

	if not os.path.exists(path_where_to_save_everything):
		os.makedirs(path_where_to_save_everything)

	if isinstance(fname_current_trace,str):
		if os.path.exists(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/'+fname_current_trace+'.tsf'):
			try:
				bad_pulses,first_good_pulse,first_pulse,last_pulse,miss_pulses,double_pulses,good_pulses, time_of_pulses, energy_per_pulse,duration_per_pulse,median_energy_good_pulses,median_duration_good_pulses,mean_peak_shape,mean_peak_std,mean_steady_state_power,mean_steady_state_power_std,ADC_time_resolution,mean_current_peak_shape,mean_current_peak_std,mean_voltage_peak_shape,mean_voltage_peak_std,mean_target_voltage_peak_shape,mean_target_voltage_peak_std,mean_steady_state_current,mean_steady_state_current_std,mean_steady_state_voltage,mean_steady_state_voltage_std,mean_steady_state_target_voltage,mean_steady_state_target_voltage_std = examine_current_trace(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/', fname_current_trace, df_log.loc[j, ['number_of_pulses']][0],want_the_power_per_pulse=True,want_the_mean_power_profile=True,SS_current=SS_current)
				ADC_time_resolution = ADC_time_resolution*1e3	# ms
				if False:
					current_traces = pd.read_csv(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/' + fname_current_trace+'.tsf',index_col=False, delimiter='\t')
					current_traces_time = current_traces['Time [s]']
					current_traces_total = current_traces['I_Src_AC [A]']
					voltage_traces_total = current_traces['U_Src_DC [V]']
					target_voltage_traces_total = current_traces['U_Tar_DC [V]']
					plt.figure()
					plt.plot(current_traces_time,-current_traces_total*voltage_traces_total)
					plt.xlabel('Time [s]')
					plt.ylabel('Power [W]')
					plt.grid()
					plt.title(pre_title+'file '+fname_current_trace)
					plt.pause(0.01)

					plt.figure()
					plt.plot(current_traces_time,-current_traces_total*voltage_traces_total)
					plt.xlabel('Time [s]')
					plt.plot(current_traces_time,current_traces_total,label='current[a]')
					plt.plot(current_traces_time,voltage_traces_total,label='source voltage[V]')
					plt.plot(current_traces_time,target_voltage_traces_total,label='target voltage[V]')
					plt.legend(loc='best')
					plt.grid()
					plt.title(pre_title+'file '+fname_current_trace)
					plt.pause(0.01)
				peak_size = int(round(median_duration_good_pulses/ADC_time_resolution*1e3))
				peak_loc = generic_filter(mean_peak_shape,np.mean,size=[peak_size]).argmax()
				fig, ax1 = plt.subplots(figsize=(12, 5))
				fig.subplots_adjust(right=0.77)
				ax1.set_title(pre_title+'Average pulse shape at plasma source\n'+fname_current_trace+'\nenergy during pulse=%.3gJ, pulse duration=%.3gms, SS during pulse=%.3gJ, --=SS' %(median_energy_good_pulses,median_duration_good_pulses*1e3,mean_steady_state_power*median_duration_good_pulses))
				ax1.set_xlabel('time (arbitrary) [ms]')
				ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
				ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
				ax4 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
				ax3.spines["right"].set_position(("axes", 1.1125))
				ax3.spines["right"].set_visible(True)
				ax4.spines["right"].set_position(("axes", 1.25))
				ax4.spines["right"].set_visible(True)
				a1, = ax1.plot(np.arange(len(mean_peak_shape))*ADC_time_resolution,mean_peak_shape+mean_steady_state_power,'b')
				ax1.plot(np.arange(len(mean_peak_shape))*ADC_time_resolution,mean_steady_state_power*np.ones_like(mean_peak_shape),'--b')
				ax1.fill_between(np.arange(len(mean_peak_shape))*ADC_time_resolution, mean_peak_shape - mean_peak_std+mean_steady_state_power-mean_steady_state_power_std, mean_peak_shape + mean_peak_std+mean_steady_state_power+mean_steady_state_power_std,color='royalblue', alpha=0.4)
				ax1.axvline(x=(peak_loc-peak_size/2)*ADC_time_resolution,color='k',linestyle='--')
				ax1.axvline(x=(peak_loc+peak_size/2)*ADC_time_resolution,color='k',linestyle='--')
				# ax1.plot([(peak_loc-peak_size/2)*ADC_time_resolution]*2,[mean_peak_shape.min()+mean_steady_state_power,mean_peak_shape.max()+mean_steady_state_power],'--k')
				# ax1.plot([(peak_loc+peak_size/2)*ADC_time_resolution]*2,[mean_peak_shape.min()+mean_steady_state_power,mean_peak_shape.max()+mean_steady_state_power],'--k')
				a2, = ax2.plot(np.arange(len(mean_peak_shape))*ADC_time_resolution,mean_current_peak_shape,'r')
				ax2.plot(np.arange(len(mean_peak_shape))*ADC_time_resolution,mean_steady_state_current*np.ones_like(mean_peak_shape),'--r')
				ax2.fill_between(np.arange(len(mean_peak_shape))*ADC_time_resolution, mean_current_peak_shape - mean_current_peak_std, mean_current_peak_shape + mean_current_peak_std,color='lightcoral', alpha=0.4)
				a3, = ax3.plot(np.arange(len(mean_peak_shape))*ADC_time_resolution,mean_voltage_peak_shape,'g')
				ax3.plot(np.arange(len(mean_peak_shape))*ADC_time_resolution,mean_steady_state_voltage*np.ones_like(mean_peak_shape),'--g')
				ax3.fill_between(np.arange(len(mean_peak_shape))*ADC_time_resolution, mean_voltage_peak_shape - mean_voltage_peak_std, mean_voltage_peak_shape + mean_voltage_peak_std,color='palegreen', alpha=0.4)
				a4, = ax4.plot(np.arange(len(mean_peak_shape))*ADC_time_resolution,mean_target_voltage_peak_shape,'y')
				ax4.plot(np.arange(len(mean_peak_shape))*ADC_time_resolution,mean_steady_state_target_voltage*np.ones_like(mean_peak_shape),'--y')
				ax4.fill_between(np.arange(len(mean_peak_shape))*ADC_time_resolution, mean_target_voltage_peak_shape - mean_target_voltage_peak_std, mean_target_voltage_peak_shape + mean_target_voltage_peak_std,color='khaki', alpha=0.4)
				# ax2.plot(j_specific_target_chamber_pressure,np.array(j_specific_DT_pulse_time_scaled),'or')
				ax1.set_ylabel('power [W]', color=a1.get_color())
				ax2.set_ylabel('current [A]', color=a2.get_color())  # we already handled the x-label with ax1
				ax3.set_ylabel('voltage [V]', color=a3.get_color())  # we already handled the x-label with ax1
				ax4.set_ylabel('target voltage [V]', color=a4.get_color())  # we already handled the x-label with ax1
				ax1.tick_params(axis='y', labelcolor=a1.get_color())
				ax2.tick_params(axis='y', labelcolor=a2.get_color())
				ax3.tick_params(axis='y', labelcolor=a3.get_color())
				ax4.tick_params(axis='y', labelcolor=a4.get_color())
				ax1.set_xlim(left=49.5,right=52)
				ax2.set_xlim(left=49.5,right=52)
				ax3.set_xlim(left=49.5,right=52)
				ax4.set_xlim(left=49.5,right=52)
				ax1.grid()
				# plt.pause(0.01)
				plt.savefig(path_where_to_save_everything +'/ADC_'+fname_current_trace+'.png', bbox_inches='tight')
				plt.close()
				power_source_dict = dict([])
				power_source_dict['time_resolution'] = ADC_time_resolution/1e3	# s
				power_source_dict['median_energy_good_pulses'] = median_energy_good_pulses
				power_source_dict['median_duration_good_pulses'] = median_duration_good_pulses
				power_source_dict['power'] = dict([])
				power_source_dict['power']['average'] = mean_peak_shape+mean_steady_state_power
				power_source_dict['power']['std'] = mean_peak_std + mean_steady_state_power_std
				power_source_dict['power']['SS'] = mean_steady_state_power
				power_source_dict['power']['SS_std'] = mean_steady_state_power_std
				power_source_dict['current'] = dict([])
				power_source_dict['current']['average'] = mean_current_peak_shape
				power_source_dict['current']['std'] = mean_current_peak_std
				power_source_dict['current']['SS'] = mean_steady_state_current
				power_source_dict['current']['SS_std'] = mean_steady_state_current_std
				power_source_dict['voltage'] = dict([])
				power_source_dict['voltage']['average'] = mean_voltage_peak_shape
				power_source_dict['voltage']['std'] = mean_voltage_peak_std
				power_source_dict['voltage']['SS'] = mean_steady_state_voltage
				power_source_dict['voltage']['SS_std'] = mean_steady_state_voltage_std
				power_source_dict['target_voltage'] = dict([])
				power_source_dict['target_voltage']['average'] = mean_target_voltage_peak_shape
				power_source_dict['target_voltage']['std'] = mean_target_voltage_peak_std
				power_source_dict['target_voltage']['SS'] = mean_steady_state_target_voltage
				power_source_dict['target_voltage']['SS_std'] = mean_steady_state_target_voltage_std
				np.savez_compressed(path_where_to_save_everything +'/ADC_'+fname_current_trace,**power_source_dict)
			except:
				print('ADC trace invalid')
				current_traces = pd.read_csv(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/' + fname_current_trace+'.tsf',index_col=False, delimiter='\t')
				current_traces_time = current_traces['Time [s]']
				current_traces_total = current_traces['I_Src_AC [A]']
				voltage_traces_total = current_traces['U_Src_DC [V]']
				target_voltage_traces_total = current_traces['U_Tar_DC [V]']
				plt.figure()
				plt.plot(current_traces_time,-current_traces_total*voltage_traces_total)
				plt.xlabel('Time [s]')
				plt.ylabel('Power [W]')
				plt.grid()
				plt.title(pre_title+'file '+fname_current_trace+'\nfull invalid ADC trace')
				plt.savefig(path_where_to_save_everything +'/ADC_'+fname_current_trace+'.png', bbox_inches='tight')
				median_duration_good_pulses = 0.7e-3	# s
	else:
		median_duration_good_pulses = 0.7e-3	# s

	TS_found = ' TS missing'
	try:
		TS_time_interp,energy_flow_interp = energy_flow_from_TS_at_sound_speed(merge_ID,interpolated_TS=True)
		energy_flow_interp[energy_flow_interp>0]-=np.mean(energy_flow_interp[TS_time_interp<=0])
		temp=np.cumsum(energy_flow_interp)
		TS_pulse_duration_at_max_power = temp.max()/np.max(energy_flow_interp)*np.median(np.diff(TS_time_interp))*1e-3	# s
		TS_found = ' TS found'
	except:
		TS_pulse_duration_at_max_power = 0.4e-3	# s
	if np.isnan(TS_pulse_duration_at_max_power):
		TS_pulse_duration_at_max_power = 0.4e-3	# s
		TS_found = ' TS missing'
	pre_title = pre_title[:-1]+TS_found + '\n'

	if bad_pulses_indexes=='':
		bad_pulses_indexes=[0]
	elif (isinstance(bad_pulses_indexes, float) or isinstance(bad_pulses_indexes, int)):
		bad_pulses_indexes=[bad_pulses_indexes]
	else:
		bad_pulses_indexes = bad_pulses_indexes.replace(' ', '').split(',')
		bad_pulses_indexes = list(map(int, bad_pulses_indexes))

	if bad_pulses_indexes_extra=='':
		bad_pulses_indexes_extra=[]
	elif (isinstance(bad_pulses_indexes_extra, float) or isinstance(bad_pulses_indexes_extra, int)):
		bad_pulses_indexes_extra=[bad_pulses_indexes_extra]
	else:
		bad_pulses_indexes_extra = bad_pulses_indexes_extra.replace(' ', '').split(',')
		bad_pulses_indexes_extra = list(map(int, bad_pulses_indexes_extra))

	# if -100 in bad_pulses_indexes:	# "keyword" in case I do not want to consider any of the pulses
	# 	bad_pulses_indexes = np.linspace(1,number_of_pulses,number_of_pulses).astype('int')

	# if bad_pulses_indexes==np.nan:
	# 	bad_pulses_indexes=[]
	# else:
	# 	bad_pulses_indexes = bad_pulses_indexes.replace(' ', '').split(',')
	# 	bad_pulses_indexes = list(map(int, bad_pulses_indexes))
	incremental_step = 0.1	# I do this because what I mean really is the time between pulses, that is always 0.1s

	IR_shape = np.array((IR_shape.replace(' ', '').split(','))).astype(int)

	if IR_reference[:3]=='Cap':
		header = ryptw.readPTWHeader('/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/'+"{0:0=2g}".format(sequence)+'/IR_camera/'+IR_reference+'.ptw')
	else:
		(folder_reference,date_reference,sequence_reference,untitled_reference) = df_log.loc[int(IR_reference),['folder','date','sequence','untitled']]
		(IR_trace_reference) = df_log.loc[int(IR_reference),['IR_trace']][0]
		header = ryptw.readPTWHeader('/home/ffederic/work/Collaboratory/test/experimental_data/'+folder_reference+'/'+"{0:0=2g}".format(sequence_reference)+'/IR_camera/'+IR_trace_reference+'.ptw')
		print('reference is item '+str(IR_trace_reference))
	# ryptw.showHeader(header)
	frequency = 1/header.h_CEDIPAquisitionPeriod # Hz
	int_time = round(header.h_CEDIPIntegrationTime,9) # s
	p2d_h = 25/77	# 25 mm == 77 pixels without sample tilting
	p2d_v = 25/92	# 25 mm == 92 pixels
	h_coordinates,v_coordinates = np.meshgrid(np.arange(header.h_PixelsPerLine+1)*p2d_h,np.arange(header.h_LinesPerField+1)*p2d_v)
	aspect_ratio = header.h_PixelsPerLine/header.h_LinesPerField/(p2d_h/p2d_v)
	IR_shape_in_mm = np.array([IR_shape[0]*p2d_v,IR_shape[1]*p2d_h,IR_shape[2]*(p2d_h+p2d_v)/2,IR_shape[3]*(p2d_h+p2d_v)/2])

	if int_time == 1e-4:
		background_interval = int(6*frequency)
	elif int_time == 4e-4:
		background_interval = int(5*frequency)
	elif int_time == 1e-3:
		background_interval = int(4*frequency)
	else:
		print('error, only 1, 0.1 and 0.4ms int time now')
		continue

	print('mark7')

	try:
		test = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace +'.npz')
		test.allow_pickle = True
		test = dict(test)
		dark = list(test['dark'].all()['dark_frames'])
		dark_bad_pixels_marker = list(test['dark'].all()['dark_bad_pixels_marker'])
	except:
		print('dark info generated')

		# frames = frames.flatten().reshape((np.shape(frames)[1],np.shape(frames)[2],np.shape(frames)[0])).T
		# frames = [[],[]]	# this does not work because during the recording the sequence of the digitizers can and does change, so I cannot assume that it is regular
		frames = []
		for i in range(1,background_interval):
			frame = ryptw.getPTWFrames(header, [i])[0][0]
			# if i%2:
			# 	frames[0].append(frame.flatten().reshape(np.shape(frame.T)).T)
			# else:
			# 	frames[1].append(frame.flatten().reshape(np.shape(frame.T)).T)
			frames.append(frame.flatten().reshape(np.shape(frame.T)).T)
		print('mark6')

		# dark = np.mean(frames,axis=0)
		bad_pixels_marker_2 = []
		dark_bad_pixels_marker_str = []
		for i in range(len(frames)):
			bad_pixels_flag = coleval.find_dead_pixels([frames[i]],treshold_for_bad_low_std=-1,treshold_for_bad_std=100,treshold_for_bad_difference=30,verbose=1).flatten()
			bad_pixels_marker_2.append(np.arange(len(bad_pixels_flag))[bad_pixels_flag>0])
			dark_bad_pixels_marker_str.append(str(bad_pixels_marker_2[-1]))
		counter = collections.Counter(dark_bad_pixels_marker_str)
		most_common_str = [counter.most_common()[0][0],counter.most_common()[1][0]]
		dark_bad_pixels_marker = [bad_pixels_marker_2[(np.array(dark_bad_pixels_marker_str)==most_common_str[0]).argmax()],bad_pixels_marker_2[(np.array(dark_bad_pixels_marker_str)==most_common_str[1]).argmax()]]
		temp=0
		while len(dark_bad_pixels_marker[0])*0.8 < np.sum([value in dark_bad_pixels_marker[1] for value in dark_bad_pixels_marker[0]]):	# safety clause in case by mistake it is used twice the same pattern of dewad pixels for both digitizers
			most_common_str = [counter.most_common()[0][0],counter.most_common()[temp+1+1][0]]
			dark_bad_pixels_marker = [bad_pixels_marker_2[(np.array(dark_bad_pixels_marker_str)==most_common_str[0]).argmax()],bad_pixels_marker_2[(np.array(dark_bad_pixels_marker_str)==most_common_str[1]).argmax()]]
			temp+=1
			print('pattern of dead pixels shifted of '+str(temp))

		# most_common_str = [counter.most_common()[0][0],counter.most_common()[1][0]]
		# most_common = [dark_bad_pixels_marker[(np.array(dark_bad_pixels_marker_str)==most_common_str[0]).argmax()],dark_bad_pixels_marker[(np.array(dark_bad_pixels_marker_str)==most_common_str[1]).argmax()]]
		dark0 = []
		dark1 = []
		for i in range(len(frames)):
			if np.sum([value in bad_pixels_marker_2[0] for value in bad_pixels_marker_2[i]]) > np.sum([value in bad_pixels_marker_2[1] for value in bad_pixels_marker_2[i]]):
				dark0.append(frames[i])
			elif np.sum([value in bad_pixels_marker_2[1] for value in bad_pixels_marker_2[i]]) > np.sum([value in bad_pixels_marker_2[0] for value in bad_pixels_marker_2[i]]):
				dark1.append(frames[i])
			else:
				print('dark frame n'+str(i)+' discarded')
		dark = [np.mean(dark0,axis=0),np.mean(dark1,axis=0)]
		# dark_bad_pixels_marker = most_common
	print('dark_bad_pixels_marker '+str(dark_bad_pixels_marker))
		# plt.figure();plt.imshow(dark),plt.colorbar();plt.pause(0.01)
	for i in range(len(dark)):
		plt.figure(figsize=(20, 10))
		# plt.imshow(dark,'rainbow')
		# plt.plot(IR_shape[1],IR_shape[0],'k+')
		# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--',label='externally supplied')
		# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
		# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
		# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
		# plt.xlabel('horizontal coord [pixels]')
		# plt.ylabel('vertical coord [pixels]')
		plt.pcolor(h_coordinates,v_coordinates,dark[i],cmap='rainbow')
		plt.plot(IR_shape[1]*p2d_h,IR_shape[0]*p2d_v,'k+')
		plt.plot((IR_shape_in_mm[1] + np.arange(-IR_shape_in_mm[2],+IR_shape_in_mm[2]+IR_shape_in_mm[2]/10,IR_shape_in_mm[2]/10)),(IR_shape_in_mm[0] + np.abs(IR_shape_in_mm[2]**2-np.arange(-IR_shape_in_mm[2],+IR_shape_in_mm[2]+IR_shape_in_mm[2]/10,IR_shape_in_mm[2]/10)**2)**0.5),'k--',label='externally supplied')
		plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--',label='externally supplied')
		plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--')
		plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
		plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
		plt.xlabel('horizontal coord [mm]')
		plt.ylabel('vertical coord [mm]')
		plt.colorbar().set_label('digital level [au]')
		# plt.axes().set_aspect('equal')
		plt.axes().set_aspect(aspect_ratio)
		plt.title(pre_title+'Dark background in '+str(j)+', IR trace '+IR_reference+'\n frequency %.3gHz, int time %.3gms\ndigit%.3g dead pixel marker ' %(frequency,int_time*1000,i) +str(dark_bad_pixels_marker[i]))
		figure_index+=1
		plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
		plt.close()

	header = ryptw.readPTWHeader('/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/'+"{0:0=2g}".format(sequence)+'/IR_camera/'+IR_trace+'.ptw')

	# check what operation is required controlling the .npz file
	try:
		selected_1_mean_counts = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace +'.npz')['selected_1_mean_counts']
		original_timestamp_ms = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace +'.npz')['original_timestamp_ms']
		successfull_reading = True
		try:
			corrected_frames_temp_restrict = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['twoD_peak_evolution_temp_averaged_std']
			corrected_frames_generation_required = False
		except:
			print('generation of corrected_frames required')
			corrected_frames_generation_required = True
	except:
		print('generation of the space averaged temperatures required')
		successfull_reading = False
		corrected_frames_generation_required = True
	# if ( not os.path.exists(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '.npz') ):



	# !! TEMPORARY !! need to reset everything
	successfull_reading = False
	# corrected_frames_generation_required = True
	# !! TEMPORARY !! need to reset everything
	corrected_frames_generation_required = True
	print('generation of corrected_frames required')



	if corrected_frames_generation_required:
		frames = [[],[]]
		frames_ordering = [[],[]]
		discarded_frames = []
		yyyy = []
		mm = []
		dd = []
		hh = []
		minutes = []
		ss_sss = []
		yyyy_mm_dd_in_ss = []
		for i in range(header.h_firstframe,header.h_lastframe):
			frame = ryptw.getPTWFrames(header, [i])[0][0]
			frame = frame.flatten().reshape(np.shape(frame.T)).T
			if True:	# seems a much better way to search for it
				shape = np.shape(frame)
				median = median_filter(frame,footprint=[[1,1,1],[1,0,1],[1,1,1]])
				bad_pixels_marker = [[] , []]
				for i_ in range(len(dark_bad_pixels_marker)):
					for i__ in dark_bad_pixels_marker[i_]:
						i__ = np.unravel_index(i__,shape)
						bad_pixels_marker[i_].append(np.abs(frame[i__]-median[i__]))
				for i_ in range(len(dark_bad_pixels_marker)):
					bad_pixels_marker[i_] = np.mean(bad_pixels_marker[i_])
				if bad_pixels_marker[0]>bad_pixels_marker[1]:
					frames[0].append(frame)
					frames_ordering[0].append(i)
					discarded_frames.append(False)
					if len(frames_ordering[0])<2:
						print('frame '+str(i)+' dig 0')
					elif i!=frames_ordering[0][-2]+2:
						print('frame '+str(i)+' dig 0')
				elif bad_pixels_marker[1]>bad_pixels_marker[0]:
					frames[1].append(frame)
					frames_ordering[1].append(i)
					discarded_frames.append(False)
					if len(frames_ordering[1])<2:
						print('frame '+str(i)+' dig 1')
					elif i!=frames_ordering[1][-2]+2:
						print('frame '+str(i)+' dig 1')
				else:
					print('frame n'+str(i)+' discarded')
					discarded_frames.append(True)
			if False:	# this approach fails to work when  there is a naturally strong variation in the image
				bad_pixels_flag = coleval.find_dead_pixels([frame],treshold_for_bad_low_std=-1,treshold_for_bad_std=100,treshold_for_bad_difference=10,verbose=1).flatten()
				bad_pixels_marker = np.arange(len(bad_pixels_flag))[bad_pixels_flag>0]
				if np.sum([value in dark_bad_pixels_marker[0] for value in bad_pixels_marker]) > np.sum([value in dark_bad_pixels_marker[1] for value in bad_pixels_marker]):
					frames[0].append(frame)
					frames_ordering[0].append(i)
					discarded_frames.append(False)
					if len(frames_ordering[0])<2:
						print('frame '+str(i)+' dig 0')
					elif i!=frames_ordering[0][-2]+2:
						print('frame '+str(i)+' dig 0')
				elif np.sum([value in dark_bad_pixels_marker[1] for value in bad_pixels_marker]) > np.sum([value in dark_bad_pixels_marker[0] for value in bad_pixels_marker]):
					frames[1].append(frame)
					frames_ordering[1].append(i)
					discarded_frames.append(False)
					if len(frames_ordering[1])<2:
						print('frame '+str(i)+' dig 1')
					elif i!=frames_ordering[1][-2]+2:
						print('frame '+str(i)+' dig 1')
				else:
					print('frame n'+str(i)+' discarded')
					discarded_frames.append(True)
			temp = ryptw.getPTWFrames(header, [i])[1][0]
			yyyy.append(temp.h_FileSaveYear)
			mm.append(temp.h_FileSaveMonth)
			dd.append(temp.h_FileSaveDay)
			hh.append(temp.h_frameHour)
			minutes.append(temp.h_frameMinute)
			ss_sss.append(temp.h_frameSecond)
			yyyy_mm_dd_in_ss.append(datetime.datetime(temp.h_FileSaveYear-30,temp.h_FileSaveMonth,temp.h_FileSaveDay).timestamp())	# I follow the same convention as the OES camera. t=0 is at the start of year 2000
		print('reversals at: dig 0 '+str(np.array(frames_ordering[0])[:-1][np.diff(frames_ordering[0])!=2])+' and dig 1 '+str(np.array(frames_ordering[1])[:-1][np.diff(frames_ordering[1])!=2]))
		yyyy = np.array(yyyy)
		mm = np.array(mm)
		dd = np.array(dd)
		hh = np.array(hh)
		minutes = np.array(minutes)
		ss_sss = np.array(ss_sss)
		yyyy_mm_dd_in_ss = np.array(yyyy_mm_dd_in_ss)
		day_ss = (hh*60+minutes)*60+ss_sss
		original_timestamp_ms = (yyyy_mm_dd_in_ss+day_ss)*1000
		# effective_dt = np.mean(np.diff(day_ss))
		effective_dt = header.h_CEDIPAquisitionPeriod
		temp = np.mean(day_ss-np.arange(len(day_ss))*effective_dt)
		day_ss = np.arange(len(day_ss))*effective_dt + temp
		corrected_timestamp_ms = (yyyy_mm_dd_in_ss + day_ss)*1000
		original_timestamp_ms = original_timestamp_ms[np.logical_not(discarded_frames)]
		corrected_timestamp_ms = corrected_timestamp_ms[np.logical_not(discarded_frames)]
		if os.path.exists(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz'):
			try:
				temp = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')
				temp.allow_pickle=True
				full_saved_file_dict = dict(temp)
			except:
				full_saved_file_dict = dict([])
		else:
			full_saved_file_dict = dict([])
		full_saved_file_dict['original_timestamp_ms'] = original_timestamp_ms
		full_saved_file_dict['corrected_timestamp_ms'] = corrected_timestamp_ms
	else:
		temp = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')
		temp.allow_pickle=True
		full_saved_file_dict = dict(temp)
		original_timestamp_ms = full_saved_file_dict['original_timestamp_ms']
		corrected_timestamp_ms = full_saved_file_dict['corrected_timestamp_ms']
	full_saved_file_dict['dark'] = dict([])
	full_saved_file_dict['dark']['dark_frames'] = dark
	full_saved_file_dict['dark']['dark_bad_pixels_marker'] = dark_bad_pixels_marker
	# np.savez_compressed(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace,**full_saved_file_dict)

	print('mark3')

	if corrected_frames_generation_required:
		spectra = np.fft.fft(np.array(frames[0]).reshape((np.shape(frames[0])[0],np.shape(frames[0])[1]*np.shape(frames[0])[2]))[:,2000:5000],axis=0)
		magnitude = 2 * np.abs(spectra) / len(spectra)
		freq = np.fft.fftfreq(len(magnitude), d=2*1 / frequency)	# 2* comes from separating digitizers

		# plt.figure();plt.plot(freq,magnitude[:,40,24]),plt.semilogy();plt.pause(0.01)
		# plt.figure();plt.imshow(magnitude[freq.argmax()]),plt.colorbar();plt.pause(0.01)
		# plt.figure();plt.imshow(magnitude[freq.argmin()]),plt.colorbar();plt.pause(0.01)

		# magnitude_reshape = magnitude.reshape((magnitude.shape[0],magnitude.shape[1]*magnitude.shape[2]))
		magnitude_reshape = magnitude[np.abs(freq)>freq.max()//10*10-1]
		magnitude_reshape = magnitude_reshape[:magnitude_reshape.shape[0]//2,:]
		temp = freq[np.abs(freq)>freq.max()//10*10-1]
		freq_2,location = np.meshgrid(temp[:temp.shape[0]//2],np.arange(magnitude.shape[1])+2000)
		fig = plt.figure(figsize=(20, 10))
		ax = fig.gca(projection='3d')
		surf = ax.plot_surface(freq_2.T, location.T, np.log(magnitude_reshape),rstride=10,cstride=10,cmap=cm.rainbow,linewidth=0, antialiased=False)
		fig.colorbar(surf, shrink=0.5, aspect=5)
		ax.set_xlabel('frequency [Hz]')
		ax.set_ylabel('pixel index')
		ax.set_zlabel('amplitude [au]')
		plt.title(pre_title+'Original FT for rows interested by oscillation in '+str(j)+', IR trace '+IR_reference+'\n frequency %.3gHz, int time %.3gms digitizer 0' %(frequency,int_time*1000))
		figure_index+=1
		plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
		plt.close()
	else:
		figure_index+=1

	# I definietly need some filtering because I have massive oscillations in especially 2 bands horizontaly at ~30 and35 pixels
	# BUT if I do two different filtering the situation gets worst, so I do only one
	if False:
		# # spectra[np.abs(freq)>(np.sort(freq)[-1]-np.sort(freq)[-1]/20000)] = 0
		spectra[np.abs(freq)>freq.max()//10*10+3] = 0
		# spectra[freq.argmin()] = 0
		# spectra[freq.argmax()] = 0
		# corrected[freq.argmin()] = 0
		print('mark4')
		corrected_frames = np.abs(np.fft.ifft(spectra,axis=0))
		corrected_spectra = np.fft.fft(corrected_frames,axis=0)
		corrected_magnitude = 2 * np.abs(corrected_spectra) / len(corrected_spectra)

		magnitude_reshape = corrected_magnitude.reshape((magnitude.shape[0],magnitude.shape[1]*magnitude.shape[2]))
		magnitude_reshape = magnitude_reshape[np.abs(freq)>freq.max()//10*10-1]
		magnitude_reshape = magnitude_reshape[:magnitude_reshape.shape[0]//2,:][:,2000:5000]
		temp = freq[np.abs(freq)>freq.max()//10*10-1]
		freq_2,location = np.meshgrid(temp[:temp.shape[0]//2],np.arange(magnitude.shape[1]*magnitude.shape[2])[2000:5000])
		fig = plt.figure(figsize=(20, 10))
		ax = fig.gca(projection='3d')
		surf = ax.plot_surface(freq_2.T, location.T, np.log(magnitude_reshape),rstride=10,cstride=10,cmap=cm.rainbow,linewidth=0, antialiased=False)
		fig.colorbar(surf, shrink=0.5, aspect=5)
		ax.set_xlabel('frequency [Hz]')
		ax.set_ylabel('pixel index')
		ax.set_zlabel('amplitude [au]')
		plt.title(pre_title+'Intermediate FT for rows interested by oscillation in '+str(j)+', IR trace '+IR_reference+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
		figure_index+=1
		plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
		plt.close()
		print('mark5')

	# ani = coleval.movie_from_data(np.array([frames[-10:]]), frequency, int_time/1000,'horizontal coord [pixels]','vertical coord [pixels]','Intersity [au]')
	# ani = coleval.movie_from_data(np.array([corrected_frames[-10:]]), frequency, int_time/1000,'horizontal coord [pixels]','vertical coord [pixels]','Intersity [au]')
	# ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_original.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
	# # ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_original.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
	# plt.close()

	# plt.figure(figsize=(20, 10))
	# plt.plot(freq[freq.argmin():freq.argmin()+50],magnitude[freq.argmin():freq.argmin()+50,int(np.shape(frames[0])[0]/2),int(np.shape(frames[0])[1]/2)],label='original')
	# if False:	# I don't think this is actually necessary
	# 	plt.plot(freq[freq.argmin():freq.argmin()+50],corrected_magnitude[freq.argmin():freq.argmin()+50,int(np.shape(frames[0])[0]/2),int(np.shape(frames[0])[1]/2)],label='corrected')
	# # plt.plot([(np.sort(freq)[-1]-np.sort(freq)[-1]/20000),(np.sort(freq)[-1]-np.sort(freq)[-1]/20000)],[np.min(magnitude[:,int(np.shape(frames[0])[0]/2),int(np.shape(frames[0])[1]/2)]),np.max(magnitude[:,int(np.shape(frames[0])[0]/2),int(np.shape(frames[0])[1]/2)])],'k--')
	# plt.plot([(np.sort(freq)[0]+np.sort(freq)[-1]/20000),(np.sort(freq)[0]+np.sort(freq)[-1]/20000)],[np.min(magnitude[:,int(np.shape(frames[0])[0]/2),int(np.shape(frames[0])[1]/2)]),np.max(magnitude[:,int(np.shape(frames[0])[0]/2),int(np.shape(frames[0])[1]/2)])],'k--')
	# plt.xlabel('frequency [Hz]')
	# plt.ylabel('amplitude [au]')
	# plt.title(pre_title+'Example for elimination of the oscillation in '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	# plt.semilogy()
	# plt.legend(loc='best')
	# figure_index+=1
	# plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	# plt.close()
	#
	# plt.figure(figsize=(20, 10))
	# plt.plot(freq[0:np.abs(freq-50).argmin()],magnitude[0:np.abs(freq-50).argmin(),int(np.shape(frames[0])[0]/2),int(np.shape(frames[0])[1]/2)],label='original')
	# if False:	# I don't think this is actually necessary
	# 	plt.plot(freq[0:np.abs(freq-50).argmin()],corrected_magnitude[0:np.abs(freq-50).argmin(),int(np.shape(frames[0])[0]/2),int(np.shape(frames[0])[1]/2)],label='corrected')
	# plt.xlabel('frequency [Hz]')
	# plt.ylabel('amplitude [au]')
	# plt.title(pre_title+'Check for classic ~25Hz oscillation in '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	# plt.semilogy()
	# plt.legend(loc='best')
	# figure_index+=1
	# plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	# plt.close()


	# plt.figure();plt.imshow(corrected_frames[0]),plt.colorbar();plt.pause(0.01)

	if corrected_frames_generation_required:
		if False:	# done at the stage of collecting the frames alrady
			# addition to manage dead pixels
			bad_pixels_marker = []
			for i in range(len(frames)):
				bad_pixels_flag = coleval.find_dead_pixels([frames[i][:background_interval]],treshold_for_bad_std=100,treshold_for_bad_difference=15,verbose=1).flatten()
				bad_pixels_marker.append(np.arange(len(bad_pixels_flag))[bad_pixels_flag>0])

			score = np.zeros((len(frames),len(frames)))
			for i in range(len(frames)):
				for ii in range(len(frames)):
					score[i,ii] = np.sum([value in dark_bad_pixels_marker[ii] for value in bad_pixels_marker[i]])
			if score[0,1]+score[1,0]<score[0,0]+score[1,1]:
				dark = np.flip(dark,axis=0)
		corrected_frames = []
		digitizer_ID = []
		for i in range(len(frames)):
			corrected_frames.append(frames[i]-dark[i])
			digitizer_ID.append(np.ones_like(frames_ordering[i])*i)
		temp = coleval.flatten(frames_ordering)
		corrected_frames = coleval.flatten(corrected_frames)
		digitizer_ID = coleval.flatten(digitizer_ID)
		corrected_frames = np.array([y for _, y in sorted(zip(temp, corrected_frames))])
		digitizer_ID = np.array([y for _, y in sorted(zip(temp, digitizer_ID))])

		# corrected_frames = np.array(frames)
		# corrected_frames = corrected_frames-dark
		# # corrected_frames = medfilt(corrected_frames,[1,5,5])
		corrected_frames_no_min_zero = cp.deepcopy(corrected_frames)

		plt.figure(figsize=(20, 20))
		# plt.imshow(peak_image,'rainbow',vmax=np.median(np.sort(peak_image.flatten())[-10:]))
		# plt.plot(IR_shape[1],IR_shape[0],'k+')
		# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--',label='externally supplied')
		# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
		# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
		# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
		# plt.xlabel('horizontal coord [pixels]')
		# plt.ylabel('vertical coord [pixels]')
		plt.pcolor(h_coordinates,v_coordinates,np.nanmean(corrected_frames_no_min_zero,axis=0),cmap='rainbow')
		plt.plot(IR_shape[1]*p2d_h,IR_shape[0]*p2d_v,'k+')
		# plt.plot((IR_shape_in_mm[1] + np.arange(-IR_shape_in_mm[2],+IR_shape_in_mm[2]+IR_shape_in_mm[2]/10,IR_shape_in_mm[2]/10)),(IR_shape_in_mm[0] + np.abs(IR_shape_in_mm[2]**2-np.arange(-IR_shape_in_mm[2],+IR_shape_in_mm[2]+IR_shape_in_mm[2]/10,IR_shape_in_mm[2]/10)**2)**0.5),'k--',label='externally supplied')
		# plt.plot((IR_shape_in_mm[1] + np.arange(-IR_shape_in_mm[2],+IR_shape_in_mm[2]+IR_shape_in_mm[2]/10,IR_shape_in_mm[2]/10)),(IR_shape_in_mm[0] - np.abs(IR_shape_in_mm[2]**2-np.arange(-IR_shape_in_mm[2],+IR_shape_in_mm[2]+IR_shape_in_mm[2]/10,IR_shape_in_mm[2]/10)**2)**0.5),'k--')
		# plt.plot((IR_shape_in_mm[1] + np.arange(-IR_shape_in_mm[3],+IR_shape_in_mm[3]+IR_shape_in_mm[3]/10,IR_shape_in_mm[3]/10)),(IR_shape_in_mm[0] + np.abs(IR_shape_in_mm[3]**2-np.arange(-IR_shape_in_mm[3],+IR_shape_in_mm[3]+IR_shape_in_mm[3]/10,IR_shape_in_mm[3]/10)**2)**0.5),'k--')
		# plt.plot((IR_shape_in_mm[1] + np.arange(-IR_shape_in_mm[3],+IR_shape_in_mm[3]+IR_shape_in_mm[3]/10,IR_shape_in_mm[3]/10)),(IR_shape_in_mm[0] - np.abs(IR_shape_in_mm[3]**2-np.arange(-IR_shape_in_mm[3],+IR_shape_in_mm[3]+IR_shape_in_mm[3]/10,IR_shape_in_mm[3]/10)**2)**0.5),'k--')
		plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--',label='externally supplied')
		plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--')
		plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
		plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
		plt.xlabel('horizontal coord [mm]')
		plt.ylabel('vertical coord [mm]')
		plt.colorbar().set_label('Counts [au]')
		# plt.axes().set_aspect('equal')
		plt.axes().set_aspect(aspect_ratio)
		plt.legend(loc='best', fontsize='x-small')
		plt.title(pre_title+'Full average of counts in IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
		figure_index+=1
		plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
		plt.close()

		corrected_frames = median_filter(corrected_frames,size=[2,7,7])
		corrected_frames[corrected_frames<0] = 0

		spectra = np.fft.fft(np.array(corrected_frames).reshape((np.shape(corrected_frames)[0],np.shape(corrected_frames)[1]*np.shape(corrected_frames)[2]))[:,2000:5000],axis=0)
		magnitude = 2 * np.abs(spectra) / len(spectra)
		freq = np.fft.fftfreq(len(magnitude), d=1 / frequency)

		# magnitude_reshape = magnitude.reshape((magnitude.shape[0],magnitude.shape[1]*magnitude.shape[2]))
		magnitude_reshape = magnitude[np.abs(freq)>freq.max()//10*10-1]
		magnitude_reshape = magnitude_reshape[:magnitude_reshape.shape[0]//2,:]
		temp = freq[np.abs(freq)>freq.max()//10*10-1]
		freq_2,location = np.meshgrid(temp[:temp.shape[0]//2],np.arange(magnitude.shape[1])+2000)
		fig = plt.figure(figsize=(20, 10))
		ax = fig.gca(projection='3d')
		surf = ax.plot_surface(freq_2.T, location.T, np.log(magnitude_reshape),rstride=10,cstride=10,cmap=cm.rainbow,linewidth=0, antialiased=False)
		fig.colorbar(surf, shrink=0.5, aspect=5)
		ax.set_xlabel('frequency [Hz]')
		ax.set_ylabel('pixel index')
		ax.set_zlabel('amplitude [au]')
		plt.title(pre_title+'Corrected FT for rows interested by oscillation in '+str(j)+', IR trace '+IR_reference+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
		figure_index+=1
		plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
		plt.close()
	else:
		figure_index+=1
		figure_index+=1

	de=0
	if True:	# temperature dependent material properties. I leave the density unchanged to avoid losses of mass and because the variation is negligible
		tau = 0.145
		if target_material in ['molibdenum']:
			thermal_conductivity = Moly_thermal_conductivity_interp	# W/mK
			heat_capacity = Moly_heat_capacity_interp	# J/kg k
			density = Moly_density_interp(20)	# kg/m3
			de= 0.3#0.15
			target_thickness = 9e-3	# m
		elif target_material in ['TZM thin','TZM thick']:
			thermal_conductivity = TZM_thermal_conductivity_interp	# W/mK
			heat_capacity = TZM_heat_capacity_interp	# J/kg k
			density = TZM_density_interp(20)	# kg/m3
			de= 0.3#0.15
			if target_material == 'TZM thin':
				target_thickness = 9e-3	# m
			else:
				target_thickness = 14e-3	# m
		elif target_material in ['tungsten dummy','tungsten']:
			thermal_conductivity = W_thermal_conductivity_interp	# W/mK
			heat_capacity = W_heat_capacity_interp	# J/kg k
			density = W_density_interp(20)	# kg/m3
			de=0.035
			target_thickness = 1e-3	# m
		else:
			print('Error, target material not specified. add it to the shot characteristics')
			exit()
		# interpolator DL to ln(°C)
		counts_to_temperature,counts_to_emissivity = dl2temp_generator(de,0,tau,int_time*1e6,target_material,out_emissivity=True)
	elif False:	# counts temperature conversion from Yu Li
		tau = 0.145
		if target_material in ['molibdenum']:
			thermal_conductivity = 126	# W/mK
			heat_capacity = 305	# J/kg k
			density = 10220	# kg/m3
			de=0.15
		elif target_material in ['TZM thin','TZM thick']:
			thermal_conductivity = 126	# W/mK
			heat_capacity = 305	# J/kg k
			density = 10220	# kg/m3
			de=0.15
		elif target_material in ['tungsten dummy','tungsten']:
			thermal_conductivity = 175	# W/mK
			heat_capacity = 134	# J/kg k
			density = 19250	# kg/m3
			de=0.035
		else:
			print('Error, target material not specified. add it to the shot characteristics')
			exit()
		counts_to_temperature,counts_to_emissivity = dl2temp_generator(de,0,tau,int_time*1e6,target_material,out_emissivity=True)
	elif False:	# deprecated
		if int_time == 1e-4:
			if target_material in ['TZM thin','TZM thick','molibdenum']:
				counts_to_emissivity = emissivity_interpolator_100mus_inttime_molibdenum
				counts_to_temperature = interpolator_100mus_inttime_molibdenum
				thermal_conductivity = 126	# W/mK
				heat_capacity = 305	# J/kg k
				density = 370	# kg/m3
			elif target_material in ['tungsten dummy','tungsten']:
				counts_to_emissivity = emissivity_interpolator_100mus_inttime_tungsten
				counts_to_temperature = interpolator_100mus_inttime_tungsten
				thermal_conductivity = 175	# W/mK
				heat_capacity = 134	# J/kg k
				density = 19250	# kg/m3
			else:
				print('Error, target material not specified. add it to the shot characteristics')
				exit()
		elif int_time == 4e-4:
			if target_material in ['TZM thin','TZM thick','molibdenum']:
				counts_to_emissivity = emissivity_interpolator_400mus_inttime_molibdenum
				counts_to_temperature = interpolator_400mus_inttime_molibdenum
				thermal_conductivity = 126	# W/mK
				heat_capacity = 305	# J/kg k
				density = 370	# kg/m3
			elif target_material in ['tungsten dummy','tungsten']:
				counts_to_emissivity = emissivity_interpolator_400mus_inttime_tungsten
				counts_to_temperature = interpolator_400mus_inttime_tungsten
				thermal_conductivity = 175	# W/mK
				heat_capacity = 134	# J/kg k
				density = 19250	# kg/m3
			else:
				print('Error, target material not specified. add it to the shot characteristics')
				exit()
			# corrected_frames = np.exp(interpolator_400mus_inttime(corrected_frames))+20
		elif int_time == 1e-3:
			if target_material in ['TZM thin','TZM thick','molibdenum']:
				counts_to_emissivity = emissivity_interpolator_1000mus_inttime_molibdenum
				counts_to_temperature = interpolator_1000mus_inttime_molibdenum
				thermal_conductivity = 126	# W/mK
				heat_capacity = 305	# J/kg k
				density = 370	# kg/m3
			elif target_material in ['tungsten dummy','tungsten']:
				counts_to_emissivity = emissivity_interpolator_1000mus_inttime_tungsten
				counts_to_temperature = interpolator_1000mus_inttime_tungsten
				thermal_conductivity = 175	# W/mK
				heat_capacity = 134	# J/kg k
				density = 19250	# kg/m3
			else:
				print('Error, target material not specified. add it to the shot characteristics')
				exit()
			# corrected_frames = np.exp(interpolator_1000mus_inttime(corrected_frames))+20
		else:
			print('error, only 0.1, 0.4, 1ms int time now')
			continue
			# return 'fail '+str(j)



	if not successfull_reading:

		selected_x = np.zeros_like(corrected_frames[0])+np.arange(np.shape(corrected_frames)[-1])
		selected_y = (np.zeros_like(corrected_frames[0]).T+np.arange(np.shape(corrected_frames)[-2])).T
		selected_1 = ((selected_x-IR_shape[1])**2 + (selected_y-IR_shape[0])**2)<IR_shape[2]**2
		selected_2 = ((selected_x-IR_shape[1])**2 + (selected_y-IR_shape[0])**2)<IR_shape[3]**2

		selected_1_mean_counts = np.mean(corrected_frames[:,selected_1],axis=-1)
		selected_1_max_counts = np.max(corrected_frames[:,selected_1],axis=-1)
		selected_2_mean_counts = np.mean(corrected_frames[:,selected_2],axis=-1)
		selected_2_max_counts = np.max(corrected_frames[:,selected_2],axis=-1)

		# peak_image_counts = corrected_frames[selected_2_max_counts.argmax()]
		# peak_after_image_counts = corrected_frames[selected_2_max_counts.argmax()+round(1.5/1000*frequency)]

		# plt.figure();plt.imshow(corrected_frames[3]),plt.colorbar();plt.pause(0.01)
		# plt.figure();plt.plot(freq,corrected_magnitude[:,40,24]),plt.semilogy();plt.pause(0.01)
		emissivity = np.zeros_like(corrected_frames)
		corrected_frames_temp = np.zeros_like(corrected_frames)
		print('mark1')
		for i in range(len(corrected_frames)):
			# emissivity[i] = np.exp(counts_to_emissivity(corrected_frames[i]))
			corrected_frames_temp[i] = np.exp(counts_to_temperature(corrected_frames[i]))
		print('mark2')

		selected_1_mean = np.mean(corrected_frames_temp[:,selected_1],axis=-1)
		selected_1_max = np.max(corrected_frames_temp[:,selected_1],axis=-1)
		selected_2_mean = np.mean(corrected_frames_temp[:,selected_2],axis=-1)
		selected_2_max = np.max(corrected_frames_temp[:,selected_2],axis=-1)
		del corrected_frames_temp

		peaks = find_peaks(selected_2_mean,distance=incremental_step*frequency*0.9)[0]
		proms = get_proms(selected_2_mean,peaks)[0]

		# ani = coleval.movie_from_data(np.array([frames]), frequency, int_time/1000,'horizontal coord [pixels]','vertical coord [pixels]','Intersity [au]')
		# ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_original.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
		# # ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_original.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
		# plt.close()
		#
		# ani = coleval.movie_from_data(np.array([corrected_frames_temp]), frequency, int_time/1000,'horizontal coord [pixels]','vertical coord [pixels]','Intersity [au]')
		# ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_original.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
		# # ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_original.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
		# plt.close()

		# peak_image = corrected_frames_temp[peaks[proms.argmax()]]
		# peak_after_image = corrected_frames_temp[peaks[proms.argmax()+round(1.5/1000*frequency)]]
		# peak_before_image = np.mean(corrected_frames_temp[peaks[proms.argmax()]-10:peaks[proms.argmax()]-5],axis=0)

		# try:
		# 	full_saved_file_dict = dict(np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz'))
		# except:
		# 	full_saved_file_dict = dict([])
		full_saved_file_dict['selected_1_mean'] = selected_1_mean
		full_saved_file_dict['selected_1_max'] = selected_1_max
		full_saved_file_dict['selected_2_mean'] = selected_2_mean
		full_saved_file_dict['selected_2_max'] = selected_2_max
		full_saved_file_dict['selected_1_mean_counts'] = selected_1_mean_counts
		full_saved_file_dict['selected_1_max_counts'] = selected_1_max_counts
		full_saved_file_dict['selected_2_mean_counts'] = selected_2_mean_counts
		full_saved_file_dict['selected_2_max_counts'] = selected_2_max_counts
		# np.savez_compressed(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace,**full_saved_file_dict)
	else:
		# full_saved_file_dict = dict(np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz'))
		selected_1_mean = full_saved_file_dict['selected_1_mean']
		selected_1_max = full_saved_file_dict['selected_1_max']
		selected_2_mean = full_saved_file_dict['selected_2_mean']
		selected_2_max = full_saved_file_dict['selected_2_max']
		selected_1_mean_counts = full_saved_file_dict['selected_1_mean_counts']
		selected_1_max_counts = full_saved_file_dict['selected_1_max_counts']
		selected_2_mean_counts = full_saved_file_dict['selected_2_mean_counts']
		selected_2_max_counts = full_saved_file_dict['selected_2_max_counts']
		# peak_image_counts = full_saved_file_dict['peak_image_counts']
		# peak_image = full_saved_file_dict['peak_image']
		# peak_after_image_counts = full_saved_file_dict['peak_after_image_counts']
		# peak_after_image = full_saved_file_dict['peak_after_image']
		# peak_before_image = full_saved_file_dict['peak_before_image']

		peaks = find_peaks(selected_2_mean,distance=incremental_step*frequency*0.9)[0]
		proms = get_proms(selected_2_mean,peaks)[0]
		# figure_index+=8

	plt.figure(figsize=(20, 10))
	plt.plot((np.arange(len(selected_1_mean))/frequency),selected_1_mean,label='selected_1_mean')
	plt.plot((np.arange(len(selected_1_max))/frequency),selected_1_max,label='selected_1_max')
	plt.plot((np.arange(len(selected_2_mean))/frequency),selected_2_mean,label='selected_2_mean')
	plt.plot((np.arange(len(selected_2_max))/frequency),selected_2_max,label='selected_2_max')
	plt.xlabel('time [s]')
	plt.ylabel('Temperature [°C]')
	plt.legend(loc='best')
	plt.grid()
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close('all')

	# plt.figure(figsize=(20, 10));plt.plot(selected_1_mean);plt.plot(selected_1_max);plt.plot(selected_2_mean);plt.plot(selected_2_max);plt.pause(0.01)

	if np.max(proms[:-int(number_of_pulses/3)])<20:
		if np.sum((proms[1:]/proms[:-1])>50)==0:
			if np.sum((proms[1:]/proms[:-1])>10)==0:
				minimum_time=0
			else:
				minimum_time = (peaks[:-1][(proms[1:]/proms[:-1])>10])[0]-1
		else:
			minimum_time = (peaks[:-1][(proms[1:]/proms[:-1])>50])[0]-1
	else:
		if len(bad_pulses_indexes)>1:
			treshold =np.sort(np.diff(np.sort(proms)))[-2]
		else:
			treshold =np.sort(np.diff(np.sort(proms)))[-1]
		minimum_time = (peaks[1:][np.diff(proms)>max(treshold,20)])[0]-int(incremental_step*frequency*0.2)
	minimum_time = max(minimum_time,background_interval)
	# minimum_time = cp.deepcopy(background_interval)
	peaks_initial = find_peaks(selected_2_max,distance=incremental_step*frequency*0.85)[0]
	proms_initial = get_proms(selected_2_max,peaks_initial)[0]
	peaks = peaks_initial[peaks_initial>minimum_time]
	peaks = peaks[:number_of_pulses]
	proms = get_proms(selected_2_max,peaks)[0]

	# guess_for_good_pulse = max(np.median(proms),np.mean(proms))
	if frequency>400:
		if np.mean(proms)>2*np.median(proms):
			guess_for_good_pulse = np.mean(proms)
			limit_up = guess_for_good_pulse*2.5
			limit_down = guess_for_good_pulse*0.2
		else:
			guess_for_good_pulse = np.median(proms)
			limit_up = guess_for_good_pulse*1.2
			limit_down = guess_for_good_pulse*0.5
		limit_down_down = cp.deepcopy(limit_down)
	else:
		guess_for_good_pulse = np.max(proms)
		limit_down = min(np.mean(proms),np.sum(np.sort(proms)[-2:])/2/2)
		limit_down_down = np.mean(proms)*0.2
		limit_up = np.max(proms)*1.1

	bad_pulses_from_IR = proms_initial[peaks_initial>background_interval]<limit_down_down
	bad_pulses_from_IR = np.arange(1,1+len(bad_pulses_from_IR))[bad_pulses_from_IR]
	# bad_pulses_from_IR = bad_pulses_from_IR[bad_pulses_from_IR<=number_of_pulses]
	if len(bad_pulses_from_IR>0):
		if np.sum(np.diff(bad_pulses_from_IR)>1)>2:
			bad_pulses_from_IR = bad_pulses_from_IR[(np.diff(bad_pulses_from_IR)>1).argmax() : len(bad_pulses_from_IR)-(np.flip(np.diff(bad_pulses_from_IR)>1,axis=0)).argmax()]

	if (len(bad_pulses_indexes)>1 and frequency>400):

		plt.figure(figsize=(20, 10))
		plt.plot((np.arange(len(selected_1_mean))/frequency)[peaks_initial],proms_initial,label='prominence of pulse')
		plt.plot((np.arange(len(selected_1_mean))/frequency)[peaks],proms,'o',label='used pulses for limits')
		plt.plot([0,len(selected_1_mean)/frequency],[np.median(proms),np.median(proms)],'--',label='median')
		plt.plot([0,len(selected_1_mean)/frequency],[np.mean(proms),np.mean(proms)],'--',label='mean')
		plt.plot([0,len(selected_1_mean)/frequency],[limit_down,limit_down],'k--',label='limit of good pulse')
		plt.plot([0,len(selected_1_mean)/frequency],[limit_up,limit_up],'k--')
		plt.plot([(np.arange(len(selected_1_mean))/frequency)[minimum_time]]*2,[proms_initial.max(),proms_initial.min()],'b--',label='minimum time')

		score = []
		for shift in range(np.max(bad_pulses_from_IR)):
			score.append(np.sum([value+shift in bad_pulses_from_IR for value in bad_pulses_indexes]) )
		# peaks = np.array(peaks_initial[peaks_initial>background_interval][np.array(score).argmax():][:number_of_pulses])[np.array([not(value in bad_pulses_indexes) for value in np.arange(1,number_of_pulses+1)])]
		score = np.array(score)
		score[score>np.floor((len(bad_pulses_indexes)-1)/1.5)+1]=0
		if j==304:
			score[9]=999
		if score[score.argmax()]==score[score.argmax()+1]:
			score[score.argmax()+1] +=1
		all_bad_pulses_indexes = bad_pulses_indexes + bad_pulses_indexes_extra
		select = np.array([not((value) in all_bad_pulses_indexes) for value in np.arange(1,number_of_pulses+1)])
		peaks = np.array(peaks_initial[peaks_initial>background_interval][score.argmax():][:number_of_pulses])
		peaks = peaks[select[:len(peaks)]]
		proms = get_proms(selected_2_max,peaks)[0]

		peaks = peaks[proms>np.median(proms)/10]
		proms = get_proms(selected_2_max,peaks)[0]

		plt.plot((np.arange(len(selected_1_mean))/frequency)[peaks],proms,'x',markersize=10,label='good pulses from OES/TS + extras')
		plt.grid()
		if len(peaks)>0:
			plt.title(pre_title+'Pulse comparison for '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000)+'\nfirst pulse at %.5gs' %((np.arange(len(selected_1_mean))/frequency)[peaks[0]]))
		else:
			plt.title(pre_title+'Pulse comparison for '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))

		# full_saved_file_dict = dict(np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz'))
		full_saved_file_dict['index_good_pulses'] = peaks
		# np.savez_compressed(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace,**full_saved_file_dict)
		# del full_saved_file_dict

		very_good_pulses = []
		interval_between_pulses = np.min(np.diff(peaks))
		for i_peak_pos,peak_pos in enumerate(peaks):
			if i_peak_pos < len(peaks)/2:
				continue
			right = peak_pos+int(interval_between_pulses*1/3)
			if right>=len(selected_2_max_counts):
				continue
			very_good_pulses.append(peak_pos)
		strongest_very_good_pulse = very_good_pulses[selected_2_max[very_good_pulses].argmax()]
		very_good_pulses = np.array(very_good_pulses)
		# strongest_very_good_pulse = very_good_pulses[get_proms(selected_2_max,very_good_pulses)[0].argmax()]
		plt.plot((np.arange(len(selected_1_mean))/frequency)[strongest_very_good_pulse],proms[peaks==strongest_very_good_pulse],'y*',markersize=20,label='sample pulse for area check')

		plt.xlabel('time [s]')
		plt.ylabel('Temperature peak prominence [°C]')
		plt.legend(loc='best')
		figure_index+=1
		plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
		plt.close('all')


	elif frequency>400:
		plt.figure(figsize=(20, 10))
		plt.plot((np.arange(len(selected_1_mean))/frequency)[peaks_initial],proms_initial,label='prominence of pulse')
		plt.plot((np.arange(len(selected_1_mean))/frequency)[peaks],proms,'o',label='used pulses for limits')
		plt.plot([0,len(selected_1_mean)/frequency],[np.median(proms),np.median(proms)],'--',label='median')
		plt.plot([0,len(selected_1_mean)/frequency],[np.mean(proms),np.mean(proms)],'--',label='mean')
		plt.plot([0,len(selected_1_mean)/frequency],[limit_down,limit_down],'k--',label='limit of good pulse')
		plt.plot([0,len(selected_1_mean)/frequency],[limit_up,limit_up],'k--')
		plt.plot([0,len(selected_1_mean)/frequency],[limit_down_down,limit_down_down],'--',label='limit of down down')

		if len(bad_pulses_from_IR)>0:
			score = []
			for shift in range(np.max(bad_pulses_from_IR)):
				score.append(np.sum([value+shift in bad_pulses_from_IR for value in bad_pulses_indexes]) )
			if score[np.array(score).argmax()]==score[np.array(score).argmax()+1]:
				if j==279:
					score[np.array(score).argmax()] +=1
				else:
					score[np.array(score).argmax()+1] +=1
			all_bad_pulses_indexes = bad_pulses_indexes + bad_pulses_indexes_extra
			select = np.array([not((value-1) in all_bad_pulses_indexes) for value in np.arange(1,number_of_pulses+1)])
			peaks_temp = np.array(peaks_initial[peaks_initial>background_interval][np.array(score).argmax():][:number_of_pulses])
			peaks_temp = peaks_temp[select[:len(peaks_temp)]]
			proms_temp = get_proms(selected_2_max,peaks_temp)[0]
			plt.plot((np.arange(len(selected_1_mean))/frequency)[peaks_temp],proms_temp,'x',markersize=10,label='good pulses from OES/TS + extras')

		peaks = peaks[np.logical_and(proms>limit_down,proms<limit_up)]
		proms = proms[np.logical_and(proms>limit_down,proms<limit_up)]

		# full_saved_file_dict = dict(np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz'))
		full_saved_file_dict['index_good_pulses'] = peaks
		# np.savez_compressed(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace,**full_saved_file_dict)
		# del full_saved_file_dict

		very_good_pulses = []
		interval_between_pulses = np.min(np.diff(peaks))
		for i_peak_pos,peak_pos in enumerate(peaks):
			if i_peak_pos < len(peaks)/2:
				continue
			right = peak_pos+int(interval_between_pulses*1/3)
			if right>=len(selected_2_max_counts):
				continue
			very_good_pulses.append(peak_pos)
		strongest_very_good_pulse = very_good_pulses[selected_2_max[very_good_pulses].argmax()]
		very_good_pulses = np.array(very_good_pulses)
		# strongest_very_good_pulse = very_good_pulses[get_proms(selected_2_max,very_good_pulses)[0].argmax()]
		plt.plot((np.arange(len(selected_1_mean))/frequency)[strongest_very_good_pulse],proms[peaks==strongest_very_good_pulse],'y*',markersize=20,label='sample pulse for area check')

		plt.grid()
		plt.title(pre_title+'Pulse comparison for '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000)+'\nfirst pulse at %.5gs' %((np.arange(len(selected_1_mean))/frequency)[peaks][0]))
		plt.xlabel('time [s]')
		plt.ylabel('Temperature peak prominence [°C]')
		plt.legend(loc='best')
		figure_index+=1
		plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
		plt.close('all')
	else:
		plt.figure(figsize=(20, 10))
		plt.plot((np.arange(len(selected_1_mean))/frequency)[peaks_initial],proms_initial,label='prominence of pulse')
		plt.plot((np.arange(len(selected_1_mean))/frequency)[peaks],proms,'o',label='used pulses for limits')
		plt.plot([0,len(selected_1_mean)/frequency],[np.median(proms),np.median(proms)],'--',label='median')
		plt.plot([0,len(selected_1_mean)/frequency],[np.mean(proms),np.mean(proms)],'--',label='mean')
		plt.plot([0,len(selected_1_mean)/frequency],[limit_down,limit_down],'k--',label='limit of good pulse')
		plt.plot([0,len(selected_1_mean)/frequency],[limit_up,limit_up],'k--')
		plt.plot([0,len(selected_1_mean)/frequency],[limit_down_down,limit_down_down],'--',label='limit of down down')

		if len(bad_pulses_from_IR>0):
			score = []
			for shift in range(np.max(bad_pulses_from_IR)):
				score.append(np.sum([value+shift in bad_pulses_from_IR for value in bad_pulses_indexes]) )
			if score[np.array(score).argmax()]==score[np.array(score).argmax()+1]:
				score[np.array(score).argmax()+1] +=1
			all_bad_pulses_indexes = bad_pulses_indexes + bad_pulses_indexes_extra
			select = np.array([not((value-1) in all_bad_pulses_indexes) for value in np.arange(1,number_of_pulses+1)])
			peaks_temp = np.array(peaks_initial[peaks_initial>background_interval][np.array(score).argmax():][:number_of_pulses])
			peaks_temp = peaks_temp[select[:len(peaks_temp)]]
			proms_temp = get_proms(selected_2_max,peaks_temp)[0]
			plt.plot((np.arange(len(selected_1_mean))/frequency)[peaks_temp],proms_temp,'x',markersize=10,label='good pulses from OES/TS + extras')

		peaks = peaks[np.logical_and(proms>limit_down,proms<limit_up)]
		proms = proms[np.logical_and(proms>limit_down,proms<limit_up)]

		# full_saved_file_dict = dict(np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz'))
		full_saved_file_dict['index_good_pulses'] = peaks
		# np.savez_compressed(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace,**full_saved_file_dict)
		# del full_saved_file_dict

		very_good_pulses = cp.deepcopy(peaks)
		interval_between_pulses = min(np.min(np.diff(peaks)),int(incremental_step*frequency))
		strongest_very_good_pulse = very_good_pulses[selected_2_max[very_good_pulses].argmax()]
		very_good_pulses = np.array(very_good_pulses)
		# strongest_very_good_pulse = very_good_pulses[get_proms(selected_2_max,very_good_pulses)[0].argmax()]
		plt.plot((np.arange(len(selected_1_mean))/frequency)[strongest_very_good_pulse],proms[peaks==strongest_very_good_pulse],'y*',markersize=20,label='sample pulse for area check')

		plt.grid()
		plt.title(pre_title+'Pulse comparison for '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000)+'\nfirst pulse at %.5gs' %((np.arange(len(selected_1_mean))/frequency)[peaks][0]))
		plt.xlabel('time [s]')
		plt.ylabel('Temperature peak prominence [°C]')
		plt.legend(loc='best')
		figure_index+=1
		plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
		plt.close('all')


	# sometimes the last pulse is too close to the end of the record, so I exclude it
	peaks = peaks[peaks+round(1.5/1000*frequency)<len(selected_2_max_counts)]
	proms = proms[peaks+round(1.5/1000*frequency)<len(selected_2_max_counts)]

	# print(np.shape(selected_2_max))
	# print(peaks)
	temp = []
	temp2 = []
	temp3 = []
	temp4 = []
	temp5 = []
	for index in range(10):
		# temp.append(selected_1_mean[peaks-index-5])
		temp4.append(selected_1_mean[very_good_pulses-index-5])
		temp5.append(selected_2_max[very_good_pulses-index-5])
		temp2.append(selected_2_max_counts[very_good_pulses-index-5])
		temp3.append(selected_2_mean_counts[very_good_pulses-index-5])
	ELM_temp_jump = np.median(np.max([selected_1_mean[very_good_pulses-1],selected_1_mean[very_good_pulses],selected_1_mean[very_good_pulses-1]],axis=0) - np.mean(temp4,axis=0))
	ELM_temp_jump_large_area_max = np.mean(np.max([selected_2_max[very_good_pulses-1],selected_2_max[very_good_pulses],selected_2_max[very_good_pulses-1]],axis=0) - np.mean(temp5,axis=0))
	ELM_temp_jump_after = np.median(selected_1_mean[very_good_pulses + round(1.5/1000*frequency)] - np.mean(temp4,axis=0))
	ELM_temp_jump_after_large_area_max = np.mean(selected_2_max[very_good_pulses + round(1.5/1000*frequency)] - np.mean(temp5,axis=0))
	# print(np.shape(temp))
	# print(np.shape(np.mean(temp,axis=0)))
	max_temp_before_peaks = np.max(np.mean(temp4,axis=0))
	mean_temp_before_peaks_large_area_max = np.mean(np.mean(temp5,axis=0))
	max_counts_before_peaks = np.max(np.mean(temp2,axis=0))
	mean_counts_before_peaks = np.max(np.mean(temp3,axis=0))
	temp_before_peaks = np.mean(temp4,axis=0)
	temp_before_peaks_large_area_max = np.mean(temp5,axis=0)

	if corrected_frames_generation_required:
		ani = coleval.movie_from_data(np.array([corrected_frames[strongest_very_good_pulse-round(1/1000*frequency):strongest_very_good_pulse+round(10/1000*frequency)]]), frequency, integration=int_time/1000,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Net counts [au]',time_offset=-1/1000,prelude=pre_title+'Counts around the strongest peak (%.5gs) \n' %(strongest_very_good_pulse/frequency))
		figure_index+=1
		ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.mp4', fps=3, writer='ffmpeg',codec='mpeg4')
		plt.close()
		corrected_frames_temp_restrict = np.zeros_like(corrected_frames[strongest_very_good_pulse-round(1/1000*frequency):strongest_very_good_pulse+round(10/1000*frequency)])
		for i in range(strongest_very_good_pulse-round(1/1000*frequency),strongest_very_good_pulse+round(10/1000*frequency)):
			# emissivity[i] = np.exp(counts_to_emissivity(corrected_frames[i]))
			corrected_frames_temp_restrict[i-(strongest_very_good_pulse-round(1/1000*frequency))] = np.exp(counts_to_temperature(corrected_frames[i]))
		ani = coleval.movie_from_data(np.array([corrected_frames_temp_restrict]), frequency, integration=int_time/1000,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Temperature [°C]',time_offset=-1/1000,prelude=pre_title+'Temperature around the strongest peak (%.5gs) \n' %(strongest_very_good_pulse/frequency))
		figure_index+=1
		ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.mp4', fps=3, writer='ffmpeg',codec='mpeg4')
		plt.close()
		# full_saved_file_dict = dict(np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz'))
		full_saved_file_dict['corrected_frames_temp_restrict'] = np.float32(corrected_frames_temp_restrict)
		# np.savez_compressed(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace,**full_saved_file_dict)
	else:
		figure_index+=1
		figure_index+=1
		# corrected_frames_temp_restrict = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['corrected_frames_temp_restrict']
		corrected_frames_temp_restrict = full_saved_file_dict['corrected_frames_temp_restrict']



	if frequency>400:
		# all_ELM_jumps = np.max([selected_1_mean[peaks-1],selected_1_mean[peaks],selected_1_mean[peaks-1]],axis=0) - temp_before_peaks
		all_ELM_jumps = selected_1_mean[very_good_pulses + round(1.5/1000*frequency)] - np.mean(temp4,axis=0)
		low_end_interval_ELM_jumps = np.sort(all_ELM_jumps)[int(len(all_ELM_jumps)*0.1)]
		up_end_interval_ELM_jumps = np.sort(all_ELM_jumps)[int(len(all_ELM_jumps)*0.9)]
		all_ELM_jumps_large_area_max = selected_2_max[very_good_pulses + round(1.5/1000*frequency)] - np.mean(temp5,axis=0)
		low_end_interval_ELM_jumps_large_area_max = np.sort(all_ELM_jumps_large_area_max)[int(len(all_ELM_jumps_large_area_max)*0.1)]
		up_end_interval_ELM_jumps_large_area_max = np.sort(all_ELM_jumps_large_area_max)[int(len(all_ELM_jumps_large_area_max)*0.9)]
	else:
		low_end_interval_ELM_jumps = 0
		up_end_interval_ELM_jumps = 0
		low_end_interval_ELM_jumps_large_area_max = 0
		up_end_interval_ELM_jumps_large_area_max = 0

	if corrected_frames_generation_required:
		peak_image_counts = corrected_frames[strongest_very_good_pulse]
		peak_after_image_counts = corrected_frames[strongest_very_good_pulse+round(1.5/1000*frequency)]
		peak_before_image_counts = np.mean(corrected_frames[strongest_very_good_pulse-15:strongest_very_good_pulse-5],axis=0)
		# full_saved_file_dict = dict(np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz'))
		full_saved_file_dict['peak_image_counts'] = np.float32(peak_image_counts)
		full_saved_file_dict['peak_after_image_counts'] = np.float32(peak_after_image_counts)
		full_saved_file_dict['peak_before_image_counts'] = np.float32(peak_before_image_counts)
		# np.savez_compressed(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace,**full_saved_file_dict)
	else:
		# peak_image_counts = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['peak_image_counts']
		# peak_after_image_counts = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['peak_after_image_counts']
		# peak_before_image_counts = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['peak_before_image_counts']
		peak_image_counts = full_saved_file_dict['peak_image_counts']
		peak_after_image_counts = full_saved_file_dict['peak_after_image_counts']
		peak_before_image_counts = full_saved_file_dict['peak_before_image_counts']
	peak_image = np.exp(counts_to_temperature(peak_image_counts))
	peak_after_image = np.exp(counts_to_temperature(peak_after_image_counts))
	peak_before_image = np.exp(counts_to_temperature(peak_before_image_counts))
	peak_delta_image = peak_after_image-peak_before_image
	peak_emissivity = np.exp(counts_to_emissivity(peak_image_counts))
	peak_after_emissivity = np.exp(counts_to_emissivity(peak_after_image_counts))
	peak_before_emissivity = np.exp(counts_to_emissivity(peak_before_image_counts))

	ani = coleval.movie_from_data(np.array([(corrected_frames_temp_restrict-peak_before_image)]), frequency, integration=int_time/1000,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Temperature increase [°C]',time_offset=-1/1000,extvmax=np.max(selected_1_max[strongest_very_good_pulse+round(1.5/1000*frequency)]-selected_1_max[strongest_very_good_pulse-15:strongest_very_good_pulse-5]),extvmin=0,prelude=pre_title+'Temperature increase around the strongest peak (%.5gs) - before\n' %(strongest_very_good_pulse/frequency))
	figure_index+=1
	ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.mp4', fps=3, writer='ffmpeg',codec='mpeg4')
	plt.close()

	# p2d_h = 25/77	# 25 mm == 77 pixels without sample tilting
	# p2d_v = 25/92	# 25 mm == 92 pixels
	max_diameter_to_fit = 24	# mm
	def gaussian_2D_fitting(spatial_coord,*args):
		x = spatial_coord[0]	# horizontal
		y = spatial_coord[1]	# vertical
		out = args[0]*np.exp(- (((x-args[2])*p2d_v/p2d_h)**2 + (y-args[1])**2)/(2*(args[3]**2)) )
		select = ((x-args[2])*p2d_h)**2+((y-args[1])*p2d_v)**2>(max_diameter_to_fit/4)**2
		out[select]=peak_delta_image[select]
		return (out-peak_delta_image).flatten()

	# guess = [max(0,peak_delta_image.min()),peak_delta_image.max()-max(0,peak_delta_image.min()),IR_shape[0],IR_shape[1],IR_shape[2]]
	# bds=[[0,0,max(0,IR_shape[0]-IR_shape[3]),max(0,IR_shape[1]-IR_shape[3]),min(IR_shape[2],10)],[np.inf,np.inf,min(peak_after_image.shape[0],IR_shape[0]+IR_shape[3]),min(peak_after_image.shape[1],IR_shape[1]+IR_shape[3]),np.max(peak_after_image.shape)/2]]
	guess = [peak_delta_image.max()-max(0,peak_delta_image.min()),IR_shape[0],IR_shape[1],IR_shape[2]]
	bds=[[0,max(0,IR_shape[0]-IR_shape[3]),max(0,IR_shape[1]-IR_shape[3]),min(IR_shape[2],10)],[np.inf,min(peak_after_image.shape[0],IR_shape[0]+IR_shape[3]),min(peak_after_image.shape[1],IR_shape[1]+IR_shape[3]),np.max(peak_after_image.shape)/2]]
	spatial_coord=np.meshgrid(np.arange(np.shape(peak_after_image)[1]),np.arange(np.shape(peak_after_image)[0]))
	fit1 = curve_fit(gaussian_2D_fitting, spatial_coord, np.zeros_like(peak_delta_image.flatten()), guess,absolute_sigma=True,bounds=bds,maxfev=int(1e4))
	x = spatial_coord[0]	# horizontal
	y = spatial_coord[1]	# vertical
	# temp = max(1,np.median(np.sort(peak_delta_image[((x-fit1[0][2])*p2d_h)**2+((y-fit1[0][3])*p2d_v)**2<(max_diameter_to_fit/2)**2])[:10]))
	# guess = [temp,max(0,peak_delta_image.max()-temp),fit1[0][2],fit1[0][3],fit1[0][4]]
	# bds=[[0,0,fit1[0][2]-1,fit1[0][3]-1,min(IR_shape[2],10)],[temp,np.inf,fit1[0][2]+1,fit1[0][3]+1,np.max(peak_after_image.shape)/2]]
	guess = [max(0,peak_delta_image.max()),fit1[0][1],fit1[0][2],fit1[0][3]]
	bds=[[0,fit1[0][1]-1,fit1[0][2]-1,min(IR_shape[2],10)],[np.inf,fit1[0][1]+1,fit1[0][2]+1,np.max(peak_after_image.shape)/2]]
	spatial_coord=np.meshgrid(np.arange(np.shape(peak_after_image)[1]),np.arange(np.shape(peak_after_image)[0]))
	try:
		fit2 = curve_fit(gaussian_2D_fitting, spatial_coord, np.zeros_like(peak_delta_image.flatten()), guess,absolute_sigma=True,bounds=bds,maxfev=int(1e4))
	except:
		print('fit2 row ~ 1220 failed')
		fit2 = fit1
	area_of_interest_sigma = np.pi*(2**0.5 * (correlated_values(fit2[0],fit2[1])[3])*p2d_v/1000)**2	# Area equivalent to all the overtemperature profile collapsed in a homogeneous circular spot
	area_of_interest_sigma = area_of_interest_sigma + ufloat(0,0.1*area_of_interest_sigma.nominal_value)	# this to add the uncertainly coming from the method itself
	area_of_interest_radius = 2**0.5 *fit2[0][3]

	plt.figure(figsize=(20, 20))
	# plt.imshow(peak_image,'rainbow',vmax=np.median(np.sort(peak_image.flatten())[-10:]))
	# plt.plot(IR_shape[1],IR_shape[0],'k+')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--',label='externally supplied')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.xlabel('horizontal coord [pixels]')
	# plt.ylabel('vertical coord [pixels]')
	plt.pcolor(h_coordinates,v_coordinates,peak_image,cmap='rainbow',vmax=np.median(np.sort(peak_image.flatten())[-10:]))
	plt.plot(IR_shape[1]*p2d_h,IR_shape[0]*p2d_v,'k+')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--',label='externally supplied')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.xlabel('horizontal coord [mm]')
	plt.ylabel('vertical coord [mm]')
	plt.colorbar().set_label('Temperature [°C]')
	# plt.errorbar(fit1[0][2],fit1[0][1],xerr=fit1[1][2,2]**0.5,yerr=fit1[1][1,1]**0.5,color='c')
	# plt.plot(fit1[0][2] + np.arange(-fit1[0][3],+fit1[0][3]+fit1[0][3]/10,fit1[0][3]/10)*p2d_v/p2d_h,fit1[0][1] + np.abs(fit1[0][3]**2-np.arange(-fit1[0][3],+fit1[0][3]+fit1[0][3]/10,fit1[0][3]/10)**2)**0.5,'c--',label='fitted\nTmin=%.3g°C, dt=%.3g°C' %(fit1[0][0],fit1[0][1]))
	# plt.plot(fit1[0][2] + np.arange(-fit1[0][3],+fit1[0][3]+fit1[0][3]/10,fit1[0][3]/10)*p2d_v/p2d_h,fit1[0][1] - np.abs(fit1[0][3]**2-np.arange(-fit1[0][3],+fit1[0][3]+fit1[0][3]/10,fit1[0][3]/10)**2)**0.5,'c--')
	plt.plot((fit1[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1[0][1] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g--',label='target diameter')
	plt.plot((fit1[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1[0][1] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g--')
	plt.errorbar(fit2[0][2]*p2d_h,fit2[0][1]*p2d_v,xerr=(fit2[1][2,2]**0.5)*p2d_h,yerr=(fit2[1][1,1]**0.5)*p2d_v,color='b')
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] + np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--',label='fitted')
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] - np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--')
	# plt.plot(fit2[0][1] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][2] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--',label='target diameter')
	# plt.plot(fit2[0][1] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][2] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--')
	plt.axes().set_aspect(aspect_ratio)
	plt.legend(loc='best', fontsize='x-small')
	plt.title(pre_title+'Temperature distribution for the strongest peak (%.5gs) in ' %(strongest_very_good_pulse/frequency)+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	# plt.imshow(peak_before_image,'rainbow',vmax=np.median(np.sort(peak_before_image.flatten())[-10:]))
	# plt.plot(IR_shape[1],IR_shape[0],'k+')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--',label='externally supplied')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.xlabel('horizontal coord [pixels]')
	# plt.ylabel('vertical coord [pixels]')
	plt.pcolor(h_coordinates,v_coordinates,peak_before_image,cmap='rainbow',vmax=np.median(np.sort(peak_before_image.flatten())[-10:]))
	plt.plot(IR_shape[1]*p2d_h,IR_shape[0]*p2d_v,'k+')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--',label='externally supplied')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.xlabel('horizontal coord [mm]')
	plt.ylabel('vertical coord [mm]')
	plt.colorbar().set_label('Temperature [°C]')
	# plt.errorbar(fit1[0][2],fit1[0][1],xerr=fit1[1][2,2]**0.5,yerr=fit1[1][1,1]**0.5,color='c')
	# plt.plot(fit1[0][2] + np.arange(-fit1[0][3],+fit1[0][3]+fit1[0][3]/10,fit1[0][3]/10)*p2d_v/p2d_h,fit1[0][1] + np.abs(fit1[0][3]**2-np.arange(-fit1[0][3],+fit1[0][3]+fit1[0][3]/10,fit1[0][3]/10)**2)**0.5,'c--',label='fitted\nTmin=%.3g°C, dt=%.3g°C' %(fit1[0][0],fit1[0][1]))
	# plt.plot(fit1[0][2] + np.arange(-fit1[0][3],+fit1[0][3]+fit1[0][3]/10,fit1[0][3]/10)*p2d_v/p2d_h,fit1[0][1] - np.abs(fit1[0][3]**2-np.arange(-fit1[0][3],+fit1[0][3]+fit1[0][3]/10,fit1[0][3]/10)**2)**0.5,'c--')
	plt.plot((fit1[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1[0][1] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g--',label='target diameter')
	plt.plot((fit1[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1[0][1] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g--')
	plt.errorbar(fit2[0][2]*p2d_h,fit2[0][1]*p2d_v,xerr=(fit2[1][2,2]**0.5)*p2d_h,yerr=(fit2[1][1,1]**0.5)*p2d_v,color='b')
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] + np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--',label='fitted')
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] - np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--')
	plt.axes().set_aspect(aspect_ratio)
	plt.legend(loc='best', fontsize='x-small')
	plt.title(pre_title+'Temperature distribution before the strongest peak (average of %.3gms to %.3gms) in ' %(-15/frequency*1000,-6/frequency*1000)+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	# plt.imshow(peak_after_image,'rainbow',vmax=np.median(np.sort(peak_after_image.flatten())[-10:]))
	# plt.plot(IR_shape[1],IR_shape[0],'k+')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--',label='externally supplied')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.xlabel('horizontal coord [pixels]')
	# plt.ylabel('vertical coord [pixels]')
	plt.pcolor(h_coordinates,v_coordinates,peak_after_image,cmap='rainbow',vmax=np.median(np.sort(peak_after_image.flatten())[-10:]))
	plt.plot(IR_shape[1]*p2d_h,IR_shape[0]*p2d_v,'k+')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--',label='externally supplied')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.xlabel('horizontal coord [mm]')
	plt.ylabel('vertical coord [mm]')
	plt.colorbar().set_label('Temperature [°C]')
	# plt.errorbar(fit1[0][2],fit1[0][1],xerr=fit1[1][2,2]**0.5,yerr=fit1[1][1,1]**0.5,color='c')
	# plt.plot(fit1[0][2] + np.arange(-fit1[0][3],+fit1[0][3]+fit1[0][3]/10,fit1[0][3]/10)*p2d_v/p2d_h,fit1[0][1] + np.abs(fit1[0][3]**2-np.arange(-fit1[0][3],+fit1[0][3]+fit1[0][3]/10,fit1[0][3]/10)**2)**0.5,'c--',label='fitted\nTmin=%.3g°C, dt=%.3g°C' %(fit1[0][0],fit1[0][1]))
	# plt.plot(fit1[0][2] + np.arange(-fit1[0][3],+fit1[0][3]+fit1[0][3]/10,fit1[0][3]/10)*p2d_v/p2d_h,fit1[0][1] - np.abs(fit1[0][3]**2-np.arange(-fit1[0][3],+fit1[0][3]+fit1[0][3]/10,fit1[0][3]/10)**2)**0.5,'c--')
	plt.plot((fit1[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1[0][1] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g--',label='target diameter')
	plt.plot((fit1[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1[0][1] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g--')
	plt.errorbar(fit2[0][2]*p2d_h,fit2[0][1]*p2d_v,xerr=(fit2[1][2,2]**0.5)*p2d_h,yerr=(fit2[1][1,1]**0.5)*p2d_v,color='b')
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] + np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--',label='fitted')
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] - np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--')
	# plt.plot(fit2[0][1] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][2] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--',label='target diameter')
	# plt.plot(fit2[0][1] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][2] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--')
	plt.axes().set_aspect(aspect_ratio)
	plt.legend(loc='best', fontsize='x-small')
	plt.title(pre_title+'Temperature distribution 1.5ms after the strongest peak in '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	# plt.imshow(peak_delta_image,'rainbow',vmin=0,vmax=np.mean(np.sort(peak_delta_image[((x-fit2[0][2])*p2d_h)**2+((y-fit2[0][1])*p2d_v)**2<(max_diameter_to_fit/2)**2])[-20:]))
	# plt.plot(IR_shape[1],IR_shape[0],'k+')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--',label='externally supplied')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.xlabel('horizontal coord [pixels]')
	# plt.ylabel('vertical coord [pixels]')
	plt.pcolor(h_coordinates,v_coordinates,peak_delta_image,cmap='rainbow',vmin=0,vmax=np.mean(np.sort(peak_delta_image[((x-fit2[0][2])*p2d_h)**2+((y-fit2[0][1])*p2d_v)**2<(max_diameter_to_fit/2)**2])[-20:]))
	plt.plot(IR_shape[1]*p2d_h,IR_shape[0]*p2d_v,'k+')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--',label='externally supplied')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.xlabel('horizontal coord [mm]')
	plt.ylabel('vertical coord [mm]')
	plt.colorbar().set_label('Temperature [°C] limited to 0')
	plt.errorbar(fit1[0][2]*p2d_h,fit1[0][1]*p2d_v,xerr=(fit1[1][2,2]**0.5)*p2d_h,yerr=(fit1[1][1,1]**0.5)*p2d_v,color='c',label='fitted\ndt=%.3g°C' %(fit1[0][0]))
	# plt.plot(fit1[0][2] + np.arange(-fit1[0][3],+fit1[0][3]+fit1[0][3]/10,fit1[0][3]/10)*p2d_v/p2d_h,fit1[0][1] + np.abs(fit1[0][3]**2-np.arange(-fit1[0][3],+fit1[0][3]+fit1[0][3]/10,fit1[0][3]/10)**2)**0.5,'c--',label='fitted\nTmin=%.3g°C, dt=%.3g°C' %(fit1[0][0],fit1[0][1]))
	# plt.plot(fit1[0][2] + np.arange(-fit1[0][3],+fit1[0][3]+fit1[0][3]/10,fit1[0][3]/10)*p2d_v/p2d_h,fit1[0][1] - np.abs(fit1[0][3]**2-np.arange(-fit1[0][3],+fit1[0][3]+fit1[0][3]/10,fit1[0][3]/10)**2)**0.5,'c--')
	plt.plot((fit1[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1[0][1] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g--',label='target diameter')
	plt.plot((fit1[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1[0][1] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g--')
	plt.errorbar(fit2[0][2]*p2d_h,fit2[0][1]*p2d_v,xerr=(fit2[1][2,2]**0.5)*p2d_h,yerr=(fit2[1][1,1]**0.5)*p2d_v,color='b')
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] + np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--',label='fitted\ndt=%.3g°C\narea=%.3g+/-%.3gm2' %(fit2[0][0],area_of_interest_sigma.nominal_value,area_of_interest_sigma.std_dev))
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] - np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--')
	# plt.plot(fit2[0][1] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][2] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--',label='target diameter')
	# plt.plot(fit2[0][1] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][2] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--')
	plt.axes().set_aspect(aspect_ratio)
	plt.legend(loc='best', fontsize='x-small')
	plt.title(pre_title+'Temperature increase 1.5ms - average of %.3gms to %.3gms around the strongest peak in ' %(-15/frequency*1000,-6/frequency*1000)+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	# plt.imshow(peak_emissivity,'rainbow')
	# plt.plot(IR_shape[1],IR_shape[0],'k+')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--',label='externally supplied')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.xlabel('horizontal coord [pixels]')
	# plt.ylabel('vertical coord [pixels]')
	plt.pcolor(h_coordinates,v_coordinates,peak_emissivity,cmap='rainbow')
	plt.plot(IR_shape[1]*p2d_h,IR_shape[0]*p2d_v,'k+')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--',label='externally supplied')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.xlabel('horizontal coord [mm]')
	plt.ylabel('vertical coord [mm]')
	plt.axes().set_aspect(aspect_ratio)
	plt.colorbar().set_label('Emissivity [au]')
	plt.title(pre_title+'Emissivity distribution for the strongest peak (%.5gs) in ' %(strongest_very_good_pulse/frequency)+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	# plt.imshow(peak_before_emissivity,'rainbow')
	# plt.plot(IR_shape[1],IR_shape[0],'k+')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--',label='externally supplied')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.xlabel('horizontal coord [pixels]')
	# plt.ylabel('vertical coord [pixels]')
	plt.pcolor(h_coordinates,v_coordinates,peak_before_emissivity,cmap='rainbow')
	plt.plot(IR_shape[1]*p2d_h,IR_shape[0]*p2d_v,'k+')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--',label='externally supplied')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.xlabel('horizontal coord [mm]')
	plt.ylabel('vertical coord [mm]')
	plt.axes().set_aspect(aspect_ratio)
	plt.colorbar().set_label('Emissivity [au]')
	plt.title(pre_title+'Emissivity distribution before for the strongest peak (average of %.3gms to %.3gms) in ' %(-10/frequency,-6/frequency)+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	index_good_pulses_refined = []
	index_good_pulses_refined_plus = []
	interval_to_find_peak = int(0.3e-3*frequency*10)
	for i_peak_pos,peak_pos in enumerate(peaks):
		left = max(0,peak_pos-int(3*frequency/1000))
		right = min(len(selected_2_max_counts),peak_pos+int(interval_between_pulses*1/3))
		# if right>=len(selected_1_max_counts):
		# 	continue
		traces_corrected = selected_2_max_counts[left:right]
		traces_corrected_mean = selected_2_mean_counts[left:right]
		# plt.figure(),plt.plot(np.arange(len(traces_corrected)),traces_corrected-traces_corrected[0]),plt.plot(np.arange(len(traces_corrected_mean)),traces_corrected_mean-traces_corrected_mean[0]),plt.pause(0.01)
		# tck = interpolate.splrep(np.arange(right-left)/frequency, traces_corrected,s=20,k=2)
		# interpolated = interpolate.splev(np.arange((right-left)*10)/10/frequency, tck)
		tck = interpolate.interp1d(np.arange(right-left)/frequency, traces_corrected,bounds_error=False,fill_value='extrapolate')
		interpolated = tck(np.arange((right-left)*10)/10/frequency)
		# plt.plot(np.arange((right-left)*10)/10,interpolated-interpolated[0],'+'),plt.pause(0.01)
		interpolated = np.convolve(interpolated, np.ones((interval_to_find_peak))/interval_to_find_peak , mode='same')
		# plt.plot(np.arange((right-left)*10)/10,interpolated-interpolated[0],'+'),plt.plot((np.arange((right-left)*10)/10)[interpolated.argmax()],interpolated.max()-interpolated[0],'o'),plt.pause(0.01)
		# real_peak_loc = (np.arange((right-left)*10)/10)[traces_corrected.argmax()*10-20+(interpolated[traces_corrected.argmax()*10-20:traces_corrected.argmax()*10+20].argmax())]
		real_peak_loc = interpolated.argmax()/10
		index_good_pulses_refined.append(left+real_peak_loc)
		fit = np.polyfit((np.arange((right-left)*10)/10)[interpolated.argmax()-int(0.5*interval_to_find_peak)+1:interpolated.argmax()+int(0.5*interval_to_find_peak)],interpolated[interpolated.argmax()-int(0.5*interval_to_find_peak)+1:interpolated.argmax()+int(0.5*interval_to_find_peak)],2)
		# plt.plot((np.arange((right-left)*10)/10)[interpolated.argmax()-int(0.5*interval_to_find_peak)+1:interpolated.argmax()+int(0.5*interval_to_find_peak)],-interpolated[0]+np.polyval(fit,(np.arange((right-left)*10)/10)[interpolated.argmax()-int(0.5*interval_to_find_peak)+1:interpolated.argmax()+int(0.5*interval_to_find_peak)]))
		# plt.plot(-fit[1]/(2*fit[0]),interpolated.max()-interpolated[0],'o'),plt.pause(0.001)
		index_good_pulses_refined_plus.append(left-fit[1]/(2*fit[0]))
	index_good_pulses_refined = np.array(index_good_pulses_refined)
	index_good_pulses_refined_plus = np.array(index_good_pulses_refined_plus)
	# full_saved_file_dict = dict(np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz'))
	full_saved_file_dict['index_good_pulses_refined'] = index_good_pulses_refined
	full_saved_file_dict['index_good_pulses_refined_plus'] = index_good_pulses_refined_plus
	# np.savez_compressed(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace,**full_saved_file_dict)

	# plt.figure()
	peak_shape = []
	peak_shape_time = []
	peak_mean_shape = []
	interval_to_find_peak = int(0.3e-3*frequency*10)
	for i_peak_pos,peak_pos in enumerate(very_good_pulses):
		# if i_peak_pos < len(peaks)/2:
		# 	continue
		left = peak_pos-int(3*frequency/1000)
		right = peak_pos+int(interval_between_pulses*1/3)
		left -= int(17*frequency/1000)
		# if right>=len(selected_1_max_counts):
		# 	continue
		traces_corrected = selected_2_max_counts[left:right]
		traces_corrected_mean = selected_2_mean_counts[left:right]
		# tck = interpolate.splrep(np.arange(right-left)/frequency, traces_corrected,s=20,k=2)
		# interpolated = interpolate.splev(np.arange((right-left)*10)/10/frequency, tck)
		tck = interpolate.interp1d(np.arange(right-left)/frequency, traces_corrected,bounds_error=False,fill_value='extrapolate')
		interpolated = tck(np.arange((right-left)*10)/10/frequency)
		interpolated = np.convolve(interpolated, np.ones((interval_to_find_peak))/interval_to_find_peak , mode='same')
		real_peak_loc = (np.arange((right-left)*10)/10)[traces_corrected.argmax()*10-20+(interpolated[traces_corrected.argmax()*10-20:traces_corrected.argmax()*10+20].argmax())]
		# plt.plot(np.arange((right-left)*10)/10 - real_peak_loc,interpolated,'--')
		# plt.plot(np.arange(right-left) - real_peak_loc,traces_corrected)
		peak_shape_time.append((np.arange((right-left)*10)/10 - real_peak_loc)/frequency)
		peak_shape.append(np.interp(np.arange((right-left)*10)/10,np.arange(right-left),traces_corrected-np.mean(traces_corrected[(np.arange(right-left) - real_peak_loc)/frequency<-1e-3])))
		peak_mean_shape.append(np.interp(np.arange((right-left)*10)/10,np.arange(right-left),traces_corrected_mean-np.mean(traces_corrected_mean[(np.arange(right-left) - real_peak_loc)/frequency<-1e-3])))

	fig, ax = plt.subplots( 2,4,figsize=(34, 18), squeeze=False)
	fig2, ax2 = plt.subplots( 2,2,figsize=(18, 18), squeeze=False)
	fig.suptitle(pre_title+'Shape of the max temperature increase for '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms, de=%.3g' %(frequency,int_time*1000,de))
	fig2.suptitle(pre_title+'Shape of the max temperature increase for '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms, de=%.3g' %(frequency,int_time*1000,de))
	ax[0,0].set_title('large area max counts')
	ax[0,0].plot(1000*np.array(peak_shape_time).T,np.array(peak_shape).T,linewidth=0.5)
	ax[1,0].set_title('large area mean counts')
	ax[1,0].plot(1000*np.array(peak_shape_time).T,np.array(peak_mean_shape).T,linewidth=0.5)

	ax2[0,0].set_title('large area max counts')
	ax2[1,0].set_title('large area mean counts')

	ax[0,1].set_title('large area max counts')
	ax[0,1].plot(1000*np.array(peak_shape_time).T,np.array(peak_shape).T,linewidth=0.5)
	ax[1,1].set_title('large area mean counts')
	ax[1,1].plot(1000*np.array(peak_shape_time).T,np.array(peak_mean_shape).T,linewidth=0.5)
	peak_shape = np.array(peak_shape).flatten()
	peak_mean_shape = np.array(peak_mean_shape).flatten()
	peak_shape_time = np.array(peak_shape_time).flatten()
	ensamble_peak_shape = []
	ensamble_peak_mean_shape = []
	ensamble_peak_time = np.unique(peak_shape_time)
	ensamble_peak_time = ensamble_peak_time[:-1][np.diff(ensamble_peak_time)>1/frequency/100]
	for time in ensamble_peak_time:
		ensamble_peak_shape.append(np.median(peak_shape[np.logical_and(peak_shape_time>time-1/frequency/100,peak_shape_time<time+1/frequency/100)]))
		ensamble_peak_mean_shape.append(np.median(peak_mean_shape[np.logical_and(peak_shape_time>time-1/frequency/100,peak_shape_time<time+1/frequency/100)]))
	all_time_points = len(peak_shape_time)
	peak_shape_time = ensamble_peak_time
	# ax[0,0].plot(1000*peak_shape_time,ensamble_peak_shape,'--m',linewidth=2)
	# ax[1,0].plot(1000*peak_shape_time,ensamble_peak_mean_shape,'--m',linewidth=2)
	#
	# ax[0,1].plot(1000*peak_shape_time,ensamble_peak_shape,'--m',linewidth=2)
	# ax[1,1].plot(1000*peak_shape_time,ensamble_peak_mean_shape,'--m',linewidth=2)
	window_size = 5
	ensamble_peak_shape = scipy.signal.savgol_filter(ensamble_peak_shape, window_size, 2)
	ensamble_peak_mean_shape = scipy.signal.savgol_filter(ensamble_peak_mean_shape, window_size, 2)
	ax[0,0].plot(1000*peak_shape_time,ensamble_peak_shape,'k',linewidth=2,label='mean profile')
	ax[0,0].plot([1.5]*2,[0,1e6],'--g')
	ax[1,0].plot(1000*peak_shape_time,ensamble_peak_mean_shape,'k',linewidth=2,label='mean profile')
	ax[1,0].plot([1.5]*2,[0,1e6],'--g')
	# if all_time_points>2*len(peak_shape_time):
	# 	ensamble_peak_shape = np.convolve(ensamble_peak_shape,np.ones(20)/20,'same')
	# ax[plot_index,0].plot(1000*peak_shape_time,ensamble_peak_shape,'r',linewidth=3)
	ax[0,0].set_xlabel('time [ms]')
	ax[0,0].set_ylabel('counts [au]')
	ax[1,0].set_xlabel('time [ms]')
	ax[1,0].set_ylabel('counts [au]')

	ax2[0,0].plot(1000*peak_shape_time,ensamble_peak_shape,'k',linewidth=2,label='mean profile')
	ax2[0,0].plot([1.5]*2,[0,1e6],'--g')
	ax2[1,0].plot(1000*peak_shape_time,ensamble_peak_mean_shape,'k',linewidth=2,label='mean profile')
	ax2[1,0].plot([1.5]*2,[0,1e6],'--g')
	ax2[0,0].set_xlabel('time [ms]')
	ax2[0,0].set_ylabel('counts [au]')
	ax2[1,0].set_xlabel('time [ms]')
	ax2[1,0].set_ylabel('counts [au]')

	ax[0,1].plot(1000*peak_shape_time,ensamble_peak_shape,'k',linewidth=2,label='mean profile')
	ax[0,1].plot([1.5]*2,[ensamble_peak_shape.min(),ensamble_peak_shape.max()],'--g')
	ax[1,1].plot(1000*peak_shape_time,ensamble_peak_mean_shape,'k',linewidth=2,label='mean profile')
	ax[1,1].plot([1.5]*2,[ensamble_peak_mean_shape.min(),ensamble_peak_mean_shape.max()],'--g')
	# if all_time_points>2*len(peak_shape_time):
	# 	ensamble_peak_shape = np.convolve(ensamble_peak_shape,np.ones(20)/20,'same')
	# ax[plot_index,0].plot(1000*peak_shape_time,ensamble_peak_shape,'r',linewidth=3)
	ax[0,1].set_xlabel('time [ms]')
	ax[0,1].set_ylabel('counts [au]')
	ax[1,1].set_xlabel('time [ms]')
	ax[1,1].set_ylabel('counts [au]')

	unsuccessful_fit = True
	try:
		if False:	# Here I wanted to find the slope of the temperature rise and find it in the temperature decrease.
			start_start_pulse = (ensamble_peak_shape>ensamble_peak_shape.max()*0.01).argmax()
			end_start_pulse = (ensamble_peak_shape>ensamble_peak_shape.max()*0.3).argmax()
			end_end_pulse = -(np.flip(ensamble_peak_shape,axis=0)>ensamble_peak_shape.max()*0.1).argmax()
			start_end_pulse = -(np.flip(ensamble_peak_shape,axis=0)>ensamble_peak_shape.max()*0.5).argmax()
			tau_up = np.polyfit(peak_shape_time[start_start_pulse:end_start_pulse],np.log(ensamble_peak_shape[start_start_pulse:end_start_pulse]),1)
			# plt.figure()
			# plt.plot(peak_shape_time[start_start_pulse:end_start_pulse],np.log(ensamble_peak_shape[start_start_pulse:end_start_pulse]))
			plt.plot(1000*peak_shape_time[start_start_pulse:end_start_pulse],np.exp(np.polyval(tau_up,peak_shape_time[start_start_pulse:end_start_pulse])),'--b')
			def to_fit(x,constant):
				out = np.exp(np.polyval([-tau_up[0],constant],x))
				return out
			guess=[1]
			fit = curve_fit(to_fit, np.float64(peak_shape_time[start_end_pulse:end_end_pulse]), np.float64(ensamble_peak_shape[start_end_pulse:end_end_pulse]), guess,maxfev=int(1e4))
			# plt.plot(peak_shape_time[start_end_pulse:end_end_pulse],np.log(ensamble_peak_shape[start_end_pulse:end_end_pulse]))
			plt.plot(1000*peak_shape_time[ensamble_peak_shape.argmax():end_end_pulse],np.exp(np.polyval([-tau_up[0],fit[0]],peak_shape_time[ensamble_peak_shape.argmax():end_end_pulse])),'--b',linewidth=3,label='time constant=%.3g' %(tau_up[0]))
			# plt.pause(0.01)
		elif True:	# we now believe that most of the peak is actually due to prompt emission of infrared line emission, so not due to an actual temperature increase
			start_start_pulse = (ensamble_peak_shape>ensamble_peak_shape.max()*0.6).argmax()
			start_pulse = (ensamble_peak_shape>ensamble_peak_shape.max()*0.2).argmax()
			end_end_pulse = -(np.flip(ensamble_peak_shape,axis=0)>ensamble_peak_shape.max()*0.6).argmax()
			pre_pulse = np.abs(peak_shape_time*1000+2).argmin()
			post_pulse = np.abs(peak_shape_time*1000-2).argmin()
			peak_pulse = ensamble_peak_shape.argmax()
			# gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
			# guess=[peak_shape_time.max(),(peak_shape_time[peak_pulse]-peak_shape_time[start_start_pulse])/2,peak_shape_time[peak_pulse]]
			gauss = lambda x, sig, x0: ensamble_peak_shape.max() * np.exp(-(((x - x0) / sig) ** 2)/2) + ensamble_peak_shape[pre_pulse]
			guess=[(peak_shape_time[peak_pulse]-peak_shape_time[start_start_pulse])/2,peak_shape_time[peak_pulse]]
			# peak_pulse = np.abs(peak_shape_time - (peak_shape_time[peak_pulse]+2e-4)).argmin()
			fit = curve_fit(gauss, np.float64(peak_shape_time[start_start_pulse:end_end_pulse]), np.float64(ensamble_peak_shape[start_start_pulse:end_end_pulse]), guess,absolute_sigma=True,maxfev=int(1e4))
			ax[0,0].plot(1000*peak_shape_time[start_start_pulse:end_end_pulse],gauss(peak_shape_time[start_start_pulse:end_end_pulse],*fit[0]),'--b',linewidth=2,label='peak fit')
			ax[0,0].plot(1000*peak_shape_time[pre_pulse:start_start_pulse],gauss(peak_shape_time[pre_pulse:start_start_pulse],*fit[0]),':b',label='raise extension')
			ax[0,1].plot(1000*peak_shape_time[start_start_pulse:end_end_pulse],gauss(peak_shape_time[start_start_pulse:end_end_pulse],*fit[0]),'--b',linewidth=2,label='peak fit')
			ax[0,1].plot(1000*peak_shape_time[pre_pulse:start_start_pulse],gauss(peak_shape_time[pre_pulse:start_start_pulse],*fit[0]),':b',label='raise extension')
			peak_pulse = np.abs(peak_shape_time-fit[0][-1]).argmin()
			end_end_pulse = np.abs(peak_shape_time-fit[0][-1]-fit[0][-2]*4).argmin()
			ax[0,0].plot([1000*fit[0][-1]]*2,[0,1e6],'--y',label='prompt peak')
			ax[1,0].plot([1000*fit[0][-1]]*2,[0,1e6],'--y',label='prompt peak')

			ax2[0,0].plot([1000*fit[0][-1]]*2,[0,1e6],'--y',label='prompt peak')
			ax2[1,0].plot([1000*fit[0][-1]]*2,[0,1e6],'--y',label='prompt peak')

			ax[0,1].plot([1000*fit[0][-1]]*2,[ensamble_peak_shape.min(),np.max(peak_shape)],'--y',label='prompt peak')
			ax[1,1].plot([1000*fit[0][-1]]*2,[ensamble_peak_mean_shape.min(),np.max(peak_mean_shape)],'--y',label='prompt peak')
			post_pulse = np.abs(peak_shape_time-(2*peak_shape_time[peak_pulse] - peak_shape_time[pre_pulse])).argmin()
			ax[0,0].plot(1000*peak_shape_time[peak_pulse:post_pulse],ensamble_peak_shape[peak_pulse:post_pulse] - gauss(peak_shape_time[peak_pulse:post_pulse],*fit[0]),'-b',linewidth=2,label='decrease-fit')
			ax[0,1].plot(1000*peak_shape_time[peak_pulse:post_pulse],ensamble_peak_shape[peak_pulse:post_pulse] - gauss(peak_shape_time[peak_pulse:post_pulse],*fit[0]),'-b',linewidth=2,label='decrease-fit')
			interpolator = interp1d(peak_shape_time[pre_pulse:peak_pulse+1]-fit[0][-1],ensamble_peak_shape[pre_pulse:peak_pulse+1],bounds_error=False,fill_value=(ensamble_peak_shape[pre_pulse],ensamble_peak_shape[peak_pulse]))
			interpolator_mean = interp1d(peak_shape_time[pre_pulse:peak_pulse+1]-fit[0][-1],ensamble_peak_mean_shape[pre_pulse:peak_pulse+1],bounds_error=False,fill_value=(ensamble_peak_mean_shape[pre_pulse],ensamble_peak_mean_shape[peak_pulse]))
			ax[0,0].plot(1000*peak_shape_time[peak_pulse:post_pulse],interpolator(fit[0][-1]-peak_shape_time[peak_pulse:post_pulse]),'--r',linewidth=1,label='mirror of raise')
			ax[1,0].plot(1000*peak_shape_time[peak_pulse:post_pulse],interpolator_mean(fit[0][-1]-peak_shape_time[peak_pulse:post_pulse]),'--r',linewidth=1,label='mirror of raise')

			ax2[0,0].plot(1000*peak_shape_time[peak_pulse:post_pulse],interpolator(fit[0][-1]-peak_shape_time[peak_pulse:post_pulse]),'--r',linewidth=1,label='mirror of raise')
			ax2[1,0].plot(1000*peak_shape_time[peak_pulse:post_pulse],interpolator_mean(fit[0][-1]-peak_shape_time[peak_pulse:post_pulse]),'--r',linewidth=1,label='mirror of raise')

			ax[0,1].plot(1000*peak_shape_time[peak_pulse:post_pulse],interpolator(fit[0][-1]-peak_shape_time[peak_pulse:post_pulse]),'--r',linewidth=1,label='mirror of raise')
			ax[1,1].plot(1000*peak_shape_time[peak_pulse:post_pulse],interpolator_mean(fit[0][-1]-peak_shape_time[peak_pulse:post_pulse]),'--r',linewidth=1,label='mirror of raise')

			ax[0,0].set_xlim(left=-2,right=2)
			ax[0,0].set_ylim(bottom=0,top=1.5*ensamble_peak_shape.max())
			ax[0,0].grid()

			ax2[0,0].set_xlim(left=-2,right=2)
			ax2[0,0].set_ylim(bottom=0,top=1.5*ensamble_peak_shape.max())
			ax2[0,0].grid()

			ax[0,1].set_xlim(left=0,right=15)
			ax[0,1].set_ylim(bottom=-1,top=max(3,ensamble_peak_shape[np.abs(peak_shape_time-1e-3).argmin()]))
			ax[0,1].grid()
			ensamble_peak_shape_full = ensamble_peak_shape
			SS_ensamble = np.mean(ensamble_peak_shape[np.abs(peak_shape_time+2e-3).argmin():np.abs(peak_shape_time+1e-3).argmin()])
			ensamble_peak_shape = ensamble_peak_shape[peak_pulse:]-interpolator(fit[0][-1]-peak_shape_time[peak_pulse:])
			ax[0,0].plot(1000*peak_shape_time[peak_pulse:],ensamble_peak_shape,'-r',linewidth=2,label='decrease-mirror')
			ax[0,1].plot(1000*peak_shape_time[peak_pulse:],ensamble_peak_shape,'-r',linewidth=2,label='decrease-mirror')

			ax2[0,0].plot(1000*peak_shape_time[peak_pulse:],ensamble_peak_shape,'-r',linewidth=2,label='decrease-mirror')

			ax[1,0].set_xlim(left=-2,right=2)
			ax[1,0].set_ylim(bottom=0,top=1.5*ensamble_peak_mean_shape.max())
			ax[1,0].grid()

			ax2[1,0].set_xlim(left=-2,right=2)
			ax2[1,0].set_ylim(bottom=0,top=1.5*ensamble_peak_mean_shape.max())
			ax2[1,0].grid()

			ax[1,1].set_xlim(left=0,right=15)
			ax[1,1].set_ylim(bottom=-1,top=max(3,ensamble_peak_mean_shape[np.abs(peak_shape_time-1e-3).argmin()]))
			ax[1,1].grid()
			ensamble_peak_mean_shape_full = ensamble_peak_mean_shape
			SS_ensamble_mean = np.mean(ensamble_peak_mean_shape_full[np.abs(peak_shape_time+2e-3).argmin():np.abs(peak_shape_time+1e-3).argmin()])
			ensamble_peak_mean_shape = ensamble_peak_mean_shape[peak_pulse:]-interpolator_mean(fit[0][-1]-peak_shape_time[peak_pulse:])
			ax[1,0].plot(1000*peak_shape_time[peak_pulse:],ensamble_peak_mean_shape,'-r',linewidth=2,label='decrease-mirror')
			ax[1,1].plot(1000*peak_shape_time[peak_pulse:],ensamble_peak_mean_shape,'-r',linewidth=2,label='decrease-mirror')

			ax2[1,0].plot(1000*peak_shape_time[peak_pulse:],ensamble_peak_mean_shape,'-r',linewidth=2,label='decrease-mirror')

			ax[0,0].legend(loc='best', fontsize='x-small')
			ax[1,0].legend(loc='best', fontsize='x-small')
			ax[0,1].legend(loc='best', fontsize='x-small')
			ax[1,1].legend(loc='best', fontsize='x-small')

			ax2[0,0].legend(loc='best', fontsize='small')
			ax2[1,0].legend(loc='best', fontsize='small')

			ensamble_peak_shape[ensamble_peak_shape<0]=0
			ensamble_peak_shape += max_counts_before_peaks
			ensamble_peak_shape_full += max_counts_before_peaks
			SS_ensamble += max_counts_before_peaks
			ensamble_peak_mean_shape[ensamble_peak_mean_shape<0]=0
			ensamble_peak_mean_shape += mean_counts_before_peaks
			ensamble_peak_mean_shape_full += mean_counts_before_peaks
			SS_ensamble_mean += mean_counts_before_peaks

			ensamble_peak_shape = np.exp(counts_to_temperature(ensamble_peak_shape))
			ensamble_peak_mean_shape = np.exp(counts_to_temperature(ensamble_peak_mean_shape))
			ensamble_peak_shape_full = np.exp(counts_to_temperature(ensamble_peak_shape_full))
			ensamble_peak_mean_shape_full = np.exp(counts_to_temperature(ensamble_peak_mean_shape_full))
			SS_ensamble = np.exp(counts_to_temperature(max(SS_ensamble,0)))
			emissivity_ensamble_peak_shape = np.exp(counts_to_emissivity(ensamble_peak_shape))
			emissivity_ensamble_peak_mean_shape = np.exp(counts_to_emissivity(ensamble_peak_mean_shape))
			emissivity_ensamble_peak_mean_shape_full = np.exp(counts_to_emissivity(ensamble_peak_mean_shape_full))
			SS_ensamble_mean = np.exp(counts_to_temperature(max(SS_ensamble_mean,0)))

			temp_time = peak_shape_time[peak_pulse:]
			ax[0,2].plot(1000*temp_time,ensamble_peak_shape,'-k',linewidth=2)
			ax[0,2].plot([1000*temp_time[ensamble_peak_shape.argmax()]]*2,[ensamble_peak_shape.min(),ensamble_peak_shape.max()],'--',label='dT=%.3g°C' %(ensamble_peak_shape[ensamble_peak_shape.argmax()]-SS_ensamble))
			ax[0,2].plot([1000*temp_time[np.abs(temp_time-1e-3).argmin()]]*2,[ensamble_peak_shape.min(),ensamble_peak_shape.max()],'--',label='dT=%.3g°C' %(ensamble_peak_shape[np.abs(temp_time-1e-3).argmin()]-SS_ensamble))
			ax[0,2].plot([1000*temp_time[np.abs(temp_time-1.5e-3).argmin()]]*2,[ensamble_peak_shape.min(),ensamble_peak_shape.max()],'--g',label='dT=%.3g°C' %(ensamble_peak_shape[np.abs(temp_time-1.5e-3).argmin()]-SS_ensamble))
			ax[0,3].plot([1000*temp_time[np.abs(temp_time-1.5e-3).argmin()]]*2,[ensamble_peak_shape_full.min(),ensamble_peak_shape_full.max()],'--g',label='dT=%.3g°C' %(ensamble_peak_shape[np.abs(temp_time-1.5e-3).argmin()]-SS_ensamble))
			ax[0,3].plot([1000*temp_time[np.abs(temp_time-2e-3).argmin()]]*2,[ensamble_peak_shape_full.min(),ensamble_peak_shape_full.max()],'--b',label='dT(mean peak + 2-3ms)=%.3g°C' %(np.mean(ensamble_peak_shape[np.abs(temp_time-2e-3).argmin():np.abs(temp_time-3e-3).argmin()])-SS_ensamble))
			ax[0,3].plot([1000*temp_time[np.abs(temp_time-3e-3).argmin()]]*2,[ensamble_peak_shape_full.min(),ensamble_peak_shape_full.max()],'--b')
			ax[0,3].plot(1000*peak_shape_time,ensamble_peak_shape_full,'k',linewidth=2)
			ax[0,3].set_ylim(top=min(ensamble_peak_shape_full.max()+40,ensamble_peak_shape_full.max()*1.07))

			# ax2[0,1].plot([1000*temp_time[np.abs(temp_time-1.5e-3).argmin()]]*2,[ensamble_peak_shape_full.min(),ensamble_peak_shape_full.max()],'--g',label='dT=%.3g°C' %(ensamble_peak_shape[np.abs(temp_time-1.5e-3).argmin()]-SS_ensamble))
			ax2[0,1].plot([1000*temp_time[np.abs(temp_time-1.5e-3).argmin()]]*2,[ensamble_peak_shape_full.min(),ensamble_peak_shape_full.max()],'--g')
			ax2[0,1].plot([1000*temp_time[np.abs(temp_time-2e-3).argmin()]]*2,[ensamble_peak_shape_full.min(),ensamble_peak_shape_full.max()],'--b',label='dT(mean peak + 2-3ms)=%.3g°C' %(np.mean(ensamble_peak_shape[np.abs(temp_time-5e-3).argmin():np.abs(temp_time-5e-3).argmin()])-SS_ensamble))
			ax2[0,1].plot([1000*temp_time[np.abs(temp_time-3e-3).argmin()]]*2,[ensamble_peak_shape_full.min(),ensamble_peak_shape_full.max()],'--b')
			ax2[0,1].plot(1000*peak_shape_time,ensamble_peak_shape_full,'k',linewidth=2)
			ax2[0,1].set_ylim(top=min(ensamble_peak_shape_full.max()+40,ensamble_peak_shape_full.max()*1.07))

			ax[1,2].plot(1000*temp_time,ensamble_peak_mean_shape,'-k',linewidth=2)
			ax[1,2].plot([1000*temp_time[ensamble_peak_mean_shape.argmax()]]*2,[ensamble_peak_mean_shape.min(),ensamble_peak_mean_shape.max()],'--',label='dT=%.3g°C' %(ensamble_peak_mean_shape[ensamble_peak_mean_shape.argmax()]-SS_ensamble_mean))
			ax[1,2].plot([1000*temp_time[np.abs(temp_time-1e-3).argmin()]]*2,[ensamble_peak_mean_shape.min(),ensamble_peak_mean_shape.max()],'--',label='dT=%.3g°C' %(ensamble_peak_mean_shape[np.abs(temp_time-1e-3).argmin()]-SS_ensamble_mean))
			ax[1,2].plot([1000*temp_time[np.abs(temp_time-1.5e-3).argmin()]]*2,[ensamble_peak_mean_shape.min(),ensamble_peak_mean_shape.max()],'--g',label='dT=%.3g°C' %(ensamble_peak_mean_shape[np.abs(temp_time-1.5e-3).argmin()]-SS_ensamble_mean))
			ax[1,3].plot([1000*temp_time[np.abs(temp_time-1.5e-3).argmin()]]*2,[ensamble_peak_mean_shape_full.min(),ensamble_peak_mean_shape_full.max()],'--g',label='dT=%.3g°C' %(ensamble_peak_mean_shape[np.abs(temp_time-1.5e-3).argmin()]-SS_ensamble_mean))
			ax[1,3].plot([1000*temp_time[np.abs(temp_time-2e-3).argmin()]]*2,[ensamble_peak_mean_shape_full.min(),ensamble_peak_mean_shape_full.max()],'--b',label='dT(mean peak + 2-3ms)=%.3g°C' %(np.mean(ensamble_peak_mean_shape[np.abs(temp_time-2e-3).argmin():np.abs(temp_time-3e-3).argmin()])-SS_ensamble_mean))
			ax[1,3].plot([1000*temp_time[np.abs(temp_time-3e-3).argmin()]]*2,[ensamble_peak_mean_shape_full.min(),ensamble_peak_mean_shape_full.max()],'--b')
			ax[1,3].plot(1000*peak_shape_time,ensamble_peak_mean_shape_full,'k',linewidth=2)
			ax[1,3].set_ylim(top=min(ensamble_peak_mean_shape_full.max()+40,ensamble_peak_mean_shape_full.max()*1.07))

			# ax2[1,1].plot([1000*temp_time[np.abs(temp_time-1.5e-3).argmin()]]*2,[ensamble_peak_mean_shape_full.min(),ensamble_peak_mean_shape_full.max()],'--g',label='dT=%.3g°C' %(ensamble_peak_mean_shape[np.abs(temp_time-1.5e-3).argmin()]-SS_ensamble_mean))
			ax2[1,1].plot([1000*temp_time[np.abs(temp_time-1.5e-3).argmin()]]*2,[ensamble_peak_mean_shape_full.min(),ensamble_peak_mean_shape_full.max()],'--g')
			ax2[1,1].plot([1000*temp_time[np.abs(temp_time-2e-3).argmin()]]*2,[ensamble_peak_mean_shape_full.min(),ensamble_peak_mean_shape_full.max()],'--b',label='dT(mean peak + 2-3ms)=%.3g°C' %(np.mean(ensamble_peak_mean_shape[np.abs(temp_time-2e-3).argmin():np.abs(temp_time-3e-3).argmin()])-SS_ensamble_mean))
			ax2[1,1].plot([1000*temp_time[np.abs(temp_time-3e-3).argmin()]]*2,[ensamble_peak_mean_shape_full.min(),ensamble_peak_mean_shape_full.max()],'--b')
			ax2[1,1].plot(1000*peak_shape_time,ensamble_peak_mean_shape_full,'k',linewidth=2)
			ax2[1,1].set_ylim(top=min(ensamble_peak_mean_shape_full.max()+40,ensamble_peak_mean_shape_full.max()*1.07))

			end_end_pulse = np.abs(temp_time*1000-1.5).argmin()
			post_pulse = np.abs(temp_time*1000-30).argmin()
			SS_time = int(1e-3*frequency*10)
			# temp = np.log(ensamble_peak_shape[end_end_pulse:post_pulse]-ensamble_peak_shape[-SS_time:].mean())
			# tau_down = np.polyfit(temp_time[end_end_pulse:post_pulse][np.isfinite(temp)],temp[np.isfinite(temp)],1)
			# min_target_temperature = np.exp(np.polyval((tau_down),temp_time))+ensamble_peak_shape[-SS_time:].mean()

			sigmaSB=5.6704e-08 #[W/(m2 K4)]
			if False:	# the exponential decay is an idea I had based on heat transport, but it is not representative of heat transport in time!
				def exponential_decay(time,*args):
					out = args[1]*np.exp(-(time-args[3])/args[2]) + args[0]
					return out

				guess = [ensamble_peak_shape[-SS_time:].mean(),ensamble_peak_shape[np.abs(temp_time-1e-3).argmin()],3e-3]
				bds=[[20,0,1e-3],[np.inf,np.inf,1e-1]]
				fit = curve_fit(exponential_decay, np.float64(temp_time[end_end_pulse:post_pulse]), np.float64(ensamble_peak_shape[end_end_pulse:post_pulse]), guess,absolute_sigma=True,bounds=bds,maxfev=int(1e4))
				min_target_temperature = exponential_decay(temp_time,*fit[0])
				# # area_of_interest = 2*np.pi*IR_shape[3]**2 * (0.3e-3)**2
				if False:	# fixed area
					area_of_interest = 2*np.pi*(0.026/2)**2	# 26mm is the diameter of the standard magnum target area_of_interest_sigma
				sigmaSB=5.6704e-08 #[W/(m2 K4)]
				# target_effected_thickness = 0.2e-3
				target_effected_thickness = 0.2e-3	# 1mm thickness of the target (approx.)
				# min_power = thermal_conductivity*(np.exp(np.polyval((tau_down),temp_time)))/target_effected_thickness + 1*sigmaSB*((min_target_temperature+273.15)**4 - 300**4)
				# min_power = area_of_interest*np.sum(min_power)/(10*frequency)
				# max_power = heat_capacity*density*area_of_interest*target_effected_thickness*np.exp(np.polyval((tau_down),0))
				min_power = thermal_conductivity*(min_target_temperature-fit[0][0])/target_effected_thickness + emissivity_ensamble_peak_shape*sigmaSB*((ensamble_peak_shape_full[peak_pulse:]+273.15)**4 - 300**4)
				min_power = area_of_interest*np.sum(min_power)/(10*frequency)
				# max_power = thermal_conductivity*(ensamble_peak_shape_full-ensamble_peak_shape_full.min())/target_effected_thickness + emissivity_ensamble_peak_shape_full*sigmaSB*((ensamble_peak_shape_full+273.15)**4 - 300**4)
				# max_power = area_of_interest*np.sum(max_power)/frequency
				max_power = heat_capacity*density*area_of_interest*target_effected_thickness*(fit[0][1])
				ax[0,2].plot(1000*temp_time,min_target_temperature,'--r',linewidth=2,label='T SS=%.3g°C\ndT=%.3g°C\ntau=%.3gms\nEnergy\nfrom area=%.3gJ\nfrom Tmax=%.3gJ' %(fit[0][0],fit[0][1],fit[0][2]*1000,min_power,max_power))
				ax[0,3].plot(1000*temp_time,min_target_temperature,'--r',linewidth=2)
				ax[0,2].set_ylim(bottom=fit[0][0]-2,top=10+min_target_temperature.max())

			# this aproach comes from a reference given me by Bruce: EVAPORATION FOR HEAT PULSES ON Ni, MO, W AND ATJ GRAPHITE AS FIRST WALL MATERIALS
			def semi_infinite_sink_decrease(time,*args):
				out = args[1]/args[2]*2/((np.pi*thermal_conductivity(args[0])*heat_capacity(args[0])*density)**0.5) *(time**0.5 - (time-args[2])**0.5) + args[0]
				# print(out)
				return out

			def semi_infinite_sink_increase(time,*args):
				out = args[1]/args[2]*2/((np.pi*thermal_conductivity(args[0])*heat_capacity(args[0])*density)**0.5) *(time**0.5) + args[0]
				# print(out)
				return out

			# def semi_infinite_sink_full_decrease(time,*args):
			# 	out = args[1]/args[2]*2/((np.pi*thermal_conductivity(args[0])*heat_capacity(args[0])*density)**0.5) *((time-args[3])**0.5 - (time-args[3]-args[2])**0.5) + args[0]
			# 	# print(out)
			# 	return out
			def semi_infinite_sink_full_decrease(max_total_time):
				def function(time,*args):
					out = (args[1]/((max_total_time-args[3])*args[2]))*2/((np.pi*thermal_conductivity(args[0])*heat_capacity(args[0])*density)**0.5) *((time-args[3])**0.5 - (time-args[3]-(max_total_time-args[3])*args[2])**0.5) + args[0]
					# print(out)
					return np.nanmax([np.ones_like(out)*args[0],out],axis=0)
				return function


			# def semi_infinite_sink_full_increase(time,*args):
			# 	out = args[1]/args[2]*2/((np.pi*thermal_conductivity(args[0])*heat_capacity(args[0])*density)**0.5) *((time-args[3])**0.5) + args[0]
			# 	# print(out)
			# 	return out
			def semi_infinite_sink_full_increase(max_total_time):
				def function(time,*args):
					out = (args[1]/((max_total_time-args[3])*args[2]))*2/((np.pi*thermal_conductivity(args[0])*heat_capacity(args[0])*density)**0.5) *((time-args[3])**0.5) + args[0]
					# print(out)
					return np.nanmax([np.ones_like(out)*args[0],out],axis=0)
				return function

			guess = [ensamble_peak_shape[-SS_time:].mean(),1e5,1e-3]
			bds=[[20,1e-6,1e-4],[np.inf,np.inf,temp_time[end_end_pulse]]]
			fit = curve_fit(semi_infinite_sink_decrease, np.float64(temp_time[end_end_pulse:post_pulse]), np.float64(ensamble_peak_shape[end_end_pulse:post_pulse]), guess,absolute_sigma=True,bounds=bds,maxfev=int(1e4))
			valid = np.logical_not(np.isnan(semi_infinite_sink_decrease(temp_time[end_end_pulse:post_pulse],*fit[0])))
			R2 = 1-np.sum(((ensamble_peak_shape[end_end_pulse:post_pulse]-semi_infinite_sink_decrease(temp_time[end_end_pulse:post_pulse],*fit[0]))**2)[valid])/np.sum(((ensamble_peak_shape[end_end_pulse:post_pulse]-np.mean(ensamble_peak_shape[end_end_pulse:post_pulse]))**2)[valid])
			fit_wit_errors = correlated_values(fit[0],fit[1])
			pulse_energy_density = (fit_wit_errors[1]+ufloat(0,0.1*fit_wit_errors[1].nominal_value))	# this to add the uncertainly coming from the method itself
			pulse_energy = pulse_energy_density*area_of_interest_sigma
			# ax[0,2].plot(1000*peak_shape_time,semi_infinite_sink_decrease(peak_shape_time,*fit[0]),'--m',linewidth=2,label='semi-infinite solid\nT SS=%.3g°C\nEdens=%.3gJ/m2\ntau=%.3gms\nEnergy=%.3gJ' %(fit[0][0],fit[0][1]*fit[0][2],fit[0][2]*1000,fit[0][1]*fit[0][2]*area_of_interest))
			# ax[0,3].plot(1000*peak_shape_time,semi_infinite_sink_decrease(peak_shape_time,*fit[0]),'--m',linewidth=2,label='semi-infinite solid\nT SS=%.3g°C\nEdens=%.3gJ/m2\ntau=%.3gms\nEnergy=%.3gJ' %(fit[0][0],fit[0][1]*fit[0][2],fit[0][2]*1000,fit[0][1]*fit[0][2]*area_of_interest))
			ax[0,2].plot(1000*peak_shape_time,semi_infinite_sink_decrease(peak_shape_time,*fit[0]),'--m',linewidth=1,label='semi-infinite solid t0=peak fixed\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,fit_wit_errors[2].nominal_value*1e3,fit_wit_errors[2].std_dev*1000,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
			ax[0,3].plot(1000*peak_shape_time,semi_infinite_sink_decrease(peak_shape_time,*fit[0]),'--m',linewidth=1,label='semi-infinite solid t0=peak fixed\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,fit_wit_errors[2].nominal_value*1e3,fit_wit_errors[2].std_dev*1000,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
			ax[0,2].plot(1000*peak_shape_time[peak_shape_time<fit[0][2]],semi_infinite_sink_increase(peak_shape_time[peak_shape_time<fit[0][2]],*fit[0]),'--m',linewidth=1)
			ax[0,3].plot(1000*peak_shape_time[peak_shape_time<fit[0][2]],semi_infinite_sink_increase(peak_shape_time[peak_shape_time<fit[0][2]],*fit[0]),'--m',linewidth=1)
			ax[0,2].set_ylim(bottom=fit[0][0]-2,top=np.nanmax(10+semi_infinite_sink_decrease(np.array([1.5,1.7,2])*1e-3,*fit[0])))

			# guess = [fit[0][0],fit[0][1],max(3e-4,fit[0][2]),peak_shape_time[start_pulse]]
			guess = [fit[0][0],fit[0][1],0.5,peak_shape_time[start_pulse]]
			# guess = fit[0]
			# bds=[[20,1e-6,1e-4,peak_shape_time[start_pulse]*1.5],[np.inf,np.inf,temp_time[end_end_pulse],temp_time[end_end_pulse]-peak_shape_time[start_pulse]*1.5]]
			bds=[[20,1e-6,1e-8,peak_shape_time[start_pulse]*1.5],[np.inf,np.inf,1,temp_time[end_end_pulse]-peak_shape_time[start_pulse]*1.5]]
			fit = curve_fit(semi_infinite_sink_full_decrease(temp_time[end_end_pulse]), np.float64(temp_time[end_end_pulse:post_pulse]), np.float64(ensamble_peak_shape[end_end_pulse:post_pulse]), guess,absolute_sigma=True,bounds=bds,maxfev=int(1e4))
			valid = np.logical_not(np.isnan(semi_infinite_sink_full_decrease(temp_time[end_end_pulse])(temp_time[end_end_pulse:post_pulse],*fit[0])))
			R2 = 1-np.sum(((ensamble_peak_shape[end_end_pulse:post_pulse]-semi_infinite_sink_full_decrease(temp_time[end_end_pulse])(temp_time[end_end_pulse:post_pulse],*fit[0]))**2)[valid])/np.sum(((ensamble_peak_shape[end_end_pulse:post_pulse]-np.mean(ensamble_peak_shape[end_end_pulse:post_pulse]))**2)[valid])
			# fit_wit_errors = fit[0]
			# fit_wit_errors[2] += -peak_shape_time[start_pulse]
			fit_wit_errors = correlated_values(fit[0],fit[1])
			pulse_energy_density = (fit_wit_errors[1]+ufloat(0,0.1*fit_wit_errors[1].nominal_value))	# this to add the uncertainly coming from the method itself
			pulse_energy = pulse_energy_density*area_of_interest_sigma
			pulse_duration_ms = 1e3*(temp_time[end_end_pulse]-fit_wit_errors[3])*fit_wit_errors[2]
			ax[0,2].plot(1000*(peak_shape_time[peak_shape_time-fit[0][3]>pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_decrease(temp_time[end_end_pulse])(peak_shape_time[peak_shape_time-fit[0][3]>pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2,label='semi-infinite solid\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_duration_ms.nominal_value,pulse_duration_ms.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
			ax[0,3].plot(1000*(peak_shape_time[peak_shape_time-fit[0][3]>pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_decrease(temp_time[end_end_pulse])(peak_shape_time[peak_shape_time-fit[0][3]>pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2,label='semi-infinite solid\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_duration_ms.nominal_value,pulse_duration_ms.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
			ax[0,2].plot(1000*(peak_shape_time[peak_shape_time-fit[0][3]<=pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_increase(temp_time[end_end_pulse])(peak_shape_time[peak_shape_time-fit[0][3]<=pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2)
			ax[0,3].plot(1000*(peak_shape_time[peak_shape_time-fit[0][3]<=pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_increase(temp_time[end_end_pulse])(peak_shape_time[peak_shape_time-fit[0][3]<=pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2)
			ax[0,3].plot([1000*fit[0][3]]*2,[ensamble_peak_shape_full.min(),ensamble_peak_shape_full.max()],'--c')
			ax[0,0].plot([1000*fit[0][3]]*2,[0,1e6],'--c')
			ax[0,2].plot([-5,20],[fit[0][0]]*2,'--r')
			ax[0,3].plot([-5,20],[fit[0][0]]*2,'--r')

			ax2[0,1].plot(1000*(peak_shape_time[peak_shape_time-fit[0][3]>pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_decrease(temp_time[end_end_pulse])(peak_shape_time[peak_shape_time-fit[0][3]>pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2,label='semi-infinite solid\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_duration_ms.nominal_value,pulse_duration_ms.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
			ax2[0,1].plot(1000*(peak_shape_time[peak_shape_time-fit[0][3]<=pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_increase(temp_time[end_end_pulse])(peak_shape_time[peak_shape_time-fit[0][3]<=pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2)
			ax2[0,1].plot([1000*fit[0][3]]*2,[ensamble_peak_shape_full.min(),ensamble_peak_shape_full.max()],'--c')
			ax2[0,0].plot([1000*fit[0][3]]*2,[0,1e6],'--c')

			# df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
			# # df_log.loc[j,['pulse_en [J]']] = min_power
			# df_log.loc[j,['pulse_en_semi_inf [J]']] = temp.nominal_value
			# df_log.loc[j,['pulse_en_semi_inf_sigma [J]']] = temp.std_dev
			# df_log.loc[j,['area_of_interest [m2]']] = area_of_interest_sigma.nominal_value
			# df_log.loc[j,['DT_pulse']] = (ensamble_peak_shape[np.abs(temp_time-1.5e-3).argmin()]-SS_ensamble)
			# df_log.to_csv(path_or_buf='/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv')
			# unsuccessful_fit = False

			if False:
				guess = [ensamble_peak_mean_shape[-SS_time:].mean(),ensamble_peak_mean_shape[np.abs(temp_time-1e-3).argmin()],3e-3]
				bds=[[20,0,1e-3],[np.inf,np.inf,1e-1]]
				fit = curve_fit(exponential_decay, temp_time[end_end_pulse:post_pulse], ensamble_peak_mean_shape[end_end_pulse:post_pulse], guess,bounds=bds,maxfev=int(1e4))
				min_target_temperature = exponential_decay(temp_time,*fit[0])
				if False:	# fixed area
					area_of_interest = 2*np.pi*(0.026/2)**2	# 26mm is the diameter of the standard magnum target
				sigmaSB=5.6704e-08 #[W/(m2 K4)]
				# target_effected_thickness = 0.7e-3	# 1mm thickness of the target (approx.)
				min_power = thermal_conductivity*(min_target_temperature-fit[0][0])/target_effected_thickness + emissivity_ensamble_peak_shape*sigmaSB*((ensamble_peak_mean_shape_full[peak_pulse:]+273.15)**4 - 300**4)
				min_power = area_of_interest*np.sum(min_power)/(10*frequency)
				max_power = heat_capacity*density*area_of_interest*target_effected_thickness*(fit[0][1])
				ax[1,2].plot(1000*temp_time,min_target_temperature,'--r',linewidth=2,label='T SS=%.3g°C\ndT=%.3g°C\ntau=%.3gms\nEnergy\nfrom area=%.3gJ\nfrom Tmax=%.3gJ' %(fit[0][0],fit[0][1],fit[0][2]*1000,min_power,max_power))
				ax[1,3].plot(1000*temp_time,min_target_temperature,'--r',linewidth=2)
				ax[1,2].set_ylim(bottom=fit[0][0]-2,top=10+min_target_temperature.max())

			guess = [ensamble_peak_mean_shape[-SS_time:].mean(),1e5,1e-3]
			bds=[[20,1e-6,1e-4],[np.inf,np.inf,temp_time[end_end_pulse]]]
			fit = curve_fit(semi_infinite_sink_decrease, np.float64(temp_time[end_end_pulse:post_pulse]), np.float64(ensamble_peak_mean_shape[end_end_pulse:post_pulse]), guess,absolute_sigma=True,bounds=bds,maxfev=int(1e4))
			valid = np.logical_not(np.isnan(semi_infinite_sink_decrease(temp_time[end_end_pulse:post_pulse],*fit[0])))
			R2 = 1-np.sum(((ensamble_peak_mean_shape[end_end_pulse:post_pulse]-semi_infinite_sink_decrease(temp_time[end_end_pulse:post_pulse],*fit[0]))**2)[valid])/np.sum(((ensamble_peak_mean_shape[end_end_pulse:post_pulse]-np.mean(ensamble_peak_mean_shape[end_end_pulse:post_pulse]))**2)[valid])
			fit_wit_errors = correlated_values(fit[0],fit[1])
			pulse_energy_density = (fit_wit_errors[1]+ufloat(0,0.1*fit_wit_errors[1].nominal_value))	# this to add the uncertainly coming from the method itself
			pulse_energy = pulse_energy_density*area_of_interest_sigma#*1/1.08	# 1/1.08 correction from the forward modeling check # 10/02/2021 now it doesn't seem to be needed
			ax[1,2].plot(1000*peak_shape_time,semi_infinite_sink_decrease(peak_shape_time,*fit[0]),'--m',linewidth=1,label='semi-infinite solid t0=peak fixed\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,fit_wit_errors[2].nominal_value*1e3,fit_wit_errors[2].std_dev*1000,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
			ax[1,3].plot(1000*peak_shape_time,semi_infinite_sink_decrease(peak_shape_time,*fit[0]),'--m',linewidth=1,label='semi-infinite solid t0=peak fixed\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,fit_wit_errors[2].nominal_value*1e3,fit_wit_errors[2].std_dev*1000,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
			ax[1,2].plot(1000*peak_shape_time[peak_shape_time<fit[0][2]],semi_infinite_sink_increase(peak_shape_time[peak_shape_time<fit[0][2]],*fit[0]),'--m',linewidth=1)
			ax[1,3].plot(1000*peak_shape_time[peak_shape_time<fit[0][2]],semi_infinite_sink_increase(peak_shape_time[peak_shape_time<fit[0][2]],*fit[0]),'--m',linewidth=1)
			ax[1,2].set_ylim(bottom=fit[0][0]-2,top=np.nanmax(10+semi_infinite_sink_decrease(np.array([1.5,1.7,2])*1e-3,*fit[0])))

			guess = [fit[0][0],fit[0][1],0.5,peak_shape_time[start_pulse]]
			# guess = fit[0]
			bds=[[20,1e-6,1e-8,peak_shape_time[start_pulse]*1.5],[np.inf,np.inf,1,temp_time[end_end_pulse]-peak_shape_time[start_pulse]*1.5]]
			fit = curve_fit(semi_infinite_sink_full_decrease(temp_time[end_end_pulse]), np.float64(temp_time[end_end_pulse:post_pulse]), np.float64(ensamble_peak_mean_shape[end_end_pulse:post_pulse]), guess,absolute_sigma=True,bounds=bds,maxfev=int(1e4))
			valid = np.logical_not(np.isnan(semi_infinite_sink_full_decrease(temp_time[end_end_pulse])(temp_time[end_end_pulse:post_pulse],*fit[0])))
			R2 = 1-np.sum(((ensamble_peak_mean_shape[end_end_pulse:post_pulse]-semi_infinite_sink_full_decrease(temp_time[end_end_pulse])(temp_time[end_end_pulse:post_pulse],*fit[0]))**2)[valid])/np.sum(((ensamble_peak_mean_shape[end_end_pulse:post_pulse]-np.mean(ensamble_peak_mean_shape[end_end_pulse:post_pulse]))**2)[valid])
			fit_wit_errors = correlated_values(fit[0],fit[1])
			pulse_energy_density = (fit_wit_errors[1]+ufloat(0,0.1*fit_wit_errors[1].nominal_value))	# this to add the uncertainly coming from the method itself
			pulse_energy = pulse_energy_density*area_of_interest_sigma#*1/1.08	# 1/1.08 correction from the forward modeling check # 10/02/2021 now it doesn't seem to be needed
			pulse_duration_ms = 1e3*(temp_time[end_end_pulse]-fit_wit_errors[3])*fit_wit_errors[2]
			ax[1,2].plot(1000*(peak_shape_time[peak_shape_time-fit[0][3]>pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_decrease(temp_time[end_end_pulse])(peak_shape_time[peak_shape_time-fit[0][3]>pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2,label='semi-infinite solid\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_duration_ms.nominal_value,pulse_duration_ms.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
			ax[1,3].plot(1000*(peak_shape_time[peak_shape_time-fit[0][3]>pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_decrease(temp_time[end_end_pulse])(peak_shape_time[peak_shape_time-fit[0][3]>pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2,label='semi-infinite solid\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_duration_ms.nominal_value,pulse_duration_ms.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
			ax[1,2].plot(1000*(peak_shape_time[peak_shape_time-fit[0][3]<=pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_increase(temp_time[end_end_pulse])(peak_shape_time[peak_shape_time-fit[0][3]<=pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2)
			ax[1,3].plot(1000*(peak_shape_time[peak_shape_time-fit[0][3]<=pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_increase(temp_time[end_end_pulse])(peak_shape_time[peak_shape_time-fit[0][3]<=pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2)
			ax[1,3].plot([1000*fit[0][3]]*2,[ensamble_peak_mean_shape_full.min(),ensamble_peak_mean_shape_full.max()],'--c')
			ax[1,0].plot([1000*fit[0][3]]*2,[0,1e6],'--c')
			ax[1,2].plot([-5,20],[fit[0][0]]*2,'--r')
			ax[1,3].plot([-5,20],[fit[0][0]]*2,'--r')

			ax2[1,1].plot(1000*(peak_shape_time[peak_shape_time-fit[0][3]>pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_decrease(temp_time[end_end_pulse])(peak_shape_time[peak_shape_time-fit[0][3]>pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2,label='semi-infinite solid\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_duration_ms.nominal_value,pulse_duration_ms.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
			ax2[1,1].plot(1000*(peak_shape_time[peak_shape_time-fit[0][3]<=pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_increase(temp_time[end_end_pulse])(peak_shape_time[peak_shape_time-fit[0][3]<=pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2)
			ax2[1,1].plot([1000*fit[0][3]]*2,[ensamble_peak_mean_shape_full.min(),ensamble_peak_mean_shape_full.max()],'--c')
			ax2[1,0].plot([1000*fit[0][3]]*2,[0,1e6],'--c')

	except:
		print('fit of the strongest pulse shape failed')

	# if unsuccessful_fit:
	# 	df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
	# 	# df_log.loc[j,['pulse_en [J]']] = min_power
	# 	df_log.loc[j,['pulse_en_semi_inf [J]']] = 0
	# 	df_log.loc[j,['pulse_en_semi_inf_sigma [J]']] = 0
	# 	df_log.loc[j,['area_of_interest [m2]']] = area_of_interest_sigma.nominal_value
	# 	df_log.loc[j,['DT_pulse']] = 0
	# 	df_log.to_csv(path_or_buf='/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv')


	ax[0,2].set_xlim(left=0,right=15)
	# ax[plot_index,0].set_ylim(bottom=ensamble_peak_shape.min(),top=1.5*ensamble_peak_shape.max())
	ax[0,2].legend(loc='best', fontsize='x-small')
	ax[0,2].grid()
	ax[0,2].set_xlabel('time [ms]')
	ax[0,2].set_ylabel('temperature [°C]')
	ax[0,2].set_title('large area max full temp')
	ax[1,2].set_xlim(left=0,right=15)
	# ax[plot_index,0].set_ylim(bottom=ensamble_peak_shape.min(),top=1.5*ensamble_peak_shape.max())
	ax[1,2].legend(loc='best', fontsize='x-small')
	ax[1,2].grid()
	ax[1,2].set_xlabel('time [ms]')
	ax[1,2].set_ylabel('temperature [°C]')
	# ax[plot_index,1].set_ylim(top=2*min_target_temperature.max())
	# ax[plot_index,1].set_yscale('log')
	ax[1,2].set_title('large area mean temp')

	ax[0,3].set_xlim(left=-2,right=15)
	ax[0,3].legend(loc='best', fontsize='x-small')
	ax[0,3].grid()
	ax[0,3].set_xlabel('time [ms]')
	ax[0,3].set_ylabel('temperature [°C]')
	ax[0,3].set_title('large area max temp\narea=%.3g+/-%.3gm2' %(area_of_interest_sigma.nominal_value,area_of_interest_sigma.std_dev))
	ax[1,3].set_xlim(left=-2,right=15)
	ax[1,3].legend(loc='best', fontsize='x-small')
	ax[1,3].grid()
	ax[1,3].set_xlabel('time [ms]')
	ax[1,3].set_ylabel('temperature [°C]')
	ax[1,3].set_title('large area mean full temp')
	figure_index+=1
	fig.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	# fig.close()
	ax2[0,1].set_xlim(left=-2,right=15)
	ax2[0,1].legend(loc='best', fontsize='small')
	ax2[0,1].grid()
	ax2[0,1].set_xlabel('time [ms]')
	ax2[0,1].set_ylabel('temperature [°C]')
	ax2[0,1].set_title('large area max full temp\narea=%.3g+/-%.3gm2' %(area_of_interest_sigma.nominal_value,area_of_interest_sigma.std_dev))
	ax2[1,1].set_xlim(left=-2,right=15)
	ax2[1,1].legend(loc='best', fontsize='small')
	ax2[1,1].grid()
	ax2[1,1].set_xlabel('time [ms]')
	ax2[1,1].set_ylabel('temperature [°C]')
	ax2[1,1].set_title('large area max mean temp')
	figure_index+=1
	fig2.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close('all')

	if corrected_frames_generation_required:
		twoD_peak_evolution_counts = []
		twoD_peak_evolution_counts_delta = []
		twoD_peak_evolution_counts_before = []
		twoD_peak_evolution_time = []
		interval_to_find_peak = int(0.3e-3*frequency*10)
		for i_peak_pos,peak_pos in enumerate(very_good_pulses):
			# if i_peak_pos < len(peaks)/2:
			# 	continue
			left = max(0,peak_pos-int(3*frequency/1000))
			right = min(len(selected_2_max_counts),peak_pos+int(interval_between_pulses*1/3))
			# if right>=len(selected_1_max_counts):
			# 	continue
			traces_corrected = selected_2_max_counts[left:right]
			traces_corrected_mean = selected_2_mean_counts[left:right]
			# tck = interpolate.splrep(np.arange(right-left)/frequency, traces_corrected,s=20,k=2)
			# interpolated = interpolate.splev(np.arange((right-left)*10)/10/frequency, tck)
			tck = interpolate.interp1d(np.arange(right-left)/frequency, traces_corrected,bounds_error=False,fill_value='extrapolate')
			interpolated = tck(np.arange((right-left)*10)/10/frequency)
			interpolated = np.convolve(interpolated, np.ones((interval_to_find_peak))/interval_to_find_peak , mode='same')
			real_peak_loc = (np.arange((right-left)*10)/10)[traces_corrected.argmax()*10-20+(interpolated[traces_corrected.argmax()*10-20:traces_corrected.argmax()*10+20].argmax())]
			# plt.plot(np.arange((right-left)*10)/10 - real_peak_loc,interpolated,'--')
			# plt.plot(np.arange(right-left) - real_peak_loc,traces_corrected)
			left -= int(17*frequency/1000)
			twoD_peak_evolution_counts.append(corrected_frames_no_min_zero[left:right])
			twoD_peak_evolution_time.append(np.arange(right-left)/frequency - (real_peak_loc + int(17*frequency/1000))/frequency)
			twoD_peak_evolution_counts_before.append(np.mean(twoD_peak_evolution_counts[-1][twoD_peak_evolution_time[-1]<-1e-3],axis=0))
			# cannot do it because of the dark subtraction is not good enough
			# twoD_peak_evolution_counts_delta.append(twoD_peak_evolution_counts[-1]-twoD_peak_evolution_counts_before[-1])
			temp_before = []
			temp_delta = []
			for i in np.unique(digitizer_ID):
				temp_before.append( np.mean(twoD_peak_evolution_counts[-1][twoD_peak_evolution_time[-1]<-1e-3][digitizer_ID[left:right][twoD_peak_evolution_time[-1]<-1e-3] == i],axis=0) )
				temp_delta.append( twoD_peak_evolution_counts[-1][digitizer_ID[left:right] == i] - temp_before[-1] )
			temp = []
			for i in range(len(twoD_peak_evolution_counts[-1])):
				if digitizer_ID[left:right][i] == np.unique(digitizer_ID)[0]:
					temp.append(temp_delta[int(np.unique(digitizer_ID)[0])][0])
					temp_delta[int(np.unique(digitizer_ID)[0])] = temp_delta[int(np.unique(digitizer_ID)[0])][1:]
				else:
					temp.append(temp_delta[int(np.unique(digitizer_ID)[1])][0])
					temp_delta[int(np.unique(digitizer_ID)[1])] = temp_delta[int(np.unique(digitizer_ID)[1])][1:]
			twoD_peak_evolution_counts_delta.append(np.array(temp))

		twoD_peak_evolution_time = np.array(twoD_peak_evolution_time)
		twoD_peak_evolution_counts = np.array(twoD_peak_evolution_counts)
		twoD_peak_evolution_counts_before = np.median(twoD_peak_evolution_counts_before,axis=0)
		twoD_peak_evolution_counts_before_std = np.std(twoD_peak_evolution_counts_before,axis=0)
		twoD_peak_evolution_counts_delta = np.array(twoD_peak_evolution_counts_delta)
		twoD_first_time = np.median(twoD_peak_evolution_time[:,0])
		twoD_peak_evolution_time_averaged = []
		twoD_peak_evolution_counts_averaged = []
		twoD_peak_evolution_counts_std = []
		twoD_peak_evolution_counts_delta_averaged = []
		twoD_peak_evolution_counts_delta_std = []
		# if (np.sum(np.logical_and(twoD_peak_evolution_time[:,0]>=twoD_first_time-1/frequency/10/2,twoD_peak_evolution_time[:,0]<twoD_first_time+1/frequency/10/2))<len(twoD_peak_evolution_time)*3/4):
		flag_time_missing = []
		for i_t in range(np.shape(twoD_peak_evolution_time)[1]*10):
			select = np.logical_and(twoD_peak_evolution_time<twoD_first_time+(i_t+1/2)/frequency/10,twoD_peak_evolution_time>=twoD_first_time+(i_t-1/2)/frequency/10)
			if np.sum(select)<len(twoD_peak_evolution_time)/10/3:
				twoD_peak_evolution_time_averaged.append(twoD_first_time+i_t/frequency/10)
				twoD_peak_evolution_counts_averaged.append(np.zeros_like(twoD_peak_evolution_counts[0][0]))
				twoD_peak_evolution_counts_std.append(np.ones_like(twoD_peak_evolution_counts[0][0]))
				twoD_peak_evolution_counts_delta_averaged.append(np.zeros_like(twoD_peak_evolution_counts_delta[0][0]))
				twoD_peak_evolution_counts_delta_std.append(np.ones_like(twoD_peak_evolution_counts_delta[0][0]))
				flag_time_missing.append(i_t)
				continue
			twoD_peak_evolution_time_averaged.append(np.mean(twoD_peak_evolution_time[select]))
			twoD_peak_evolution_counts_averaged.append(np.median(twoD_peak_evolution_counts[select],axis=0))
			twoD_peak_evolution_counts_std.append(np.std(twoD_peak_evolution_counts[select],axis=0))
			twoD_peak_evolution_counts_delta_averaged.append(np.median(twoD_peak_evolution_counts_delta[select],axis=0))
			twoD_peak_evolution_counts_delta_std.append(np.std(twoD_peak_evolution_counts_delta[select],axis=0))
		twoD_peak_evolution_counts_averaged_2 = []
		twoD_peak_evolution_counts_std_2 = []
		twoD_peak_evolution_counts_delta_averaged_2 = []
		twoD_peak_evolution_counts_delta_std_2 = []
		twoD_peak_evolution_time_averaged_2 = []
		flag_bad_data = np.max(twoD_peak_evolution_counts_averaged,axis=(1,2))==0
		for i_t in range(np.shape(twoD_peak_evolution_time)[1]*10):
			if i_t in flag_time_missing:
				difference_down = np.abs(twoD_peak_evolution_time_averaged-(twoD_first_time+i_t/frequency/10))
				difference_down[np.logical_or(twoD_peak_evolution_time_averaged>twoD_first_time+(i_t-1/2)/frequency/10,flag_bad_data)]=1e6
				if np.min(difference_down)==1e6:
					continue
				difference_down = difference_down.argmin()
				difference_up = np.abs(twoD_peak_evolution_time_averaged-(twoD_first_time+i_t/frequency/10))
				difference_up[np.logical_or(twoD_peak_evolution_time_averaged<twoD_first_time+(i_t+1/2)/frequency/10,flag_bad_data)]=1e6
				if np.min(difference_up)==1e6:
					continue
				difference_up = difference_up.argmin()
				twoD_peak_evolution_counts_averaged_2.append( ((difference_up-i_t)*twoD_peak_evolution_counts_averaged[difference_down]+(i_t-difference_down)*twoD_peak_evolution_counts_averaged[difference_up] )/(difference_up-difference_down) )
				twoD_peak_evolution_counts_std_2.append( np.max([twoD_peak_evolution_counts_std[difference_down],twoD_peak_evolution_counts_std[difference_up]],axis=0) )
				twoD_peak_evolution_counts_delta_averaged_2.append( ((difference_up-i_t)*twoD_peak_evolution_counts_delta_averaged[difference_down]+(i_t-difference_down)*twoD_peak_evolution_counts_delta_averaged[difference_up] )/(difference_up-difference_down) )
				twoD_peak_evolution_counts_delta_std_2.append( np.max([twoD_peak_evolution_counts_delta_std[difference_down],twoD_peak_evolution_counts_delta_std[difference_up]],axis=0) )
			else:
				twoD_peak_evolution_counts_averaged_2.append( twoD_peak_evolution_counts_averaged[i_t] )
				twoD_peak_evolution_counts_std_2.append( twoD_peak_evolution_counts_std[i_t] )
				twoD_peak_evolution_counts_delta_averaged_2.append( twoD_peak_evolution_counts_delta_averaged[i_t] )
				twoD_peak_evolution_counts_delta_std_2.append( twoD_peak_evolution_counts_delta_std[i_t] )
			twoD_peak_evolution_time_averaged_2.append(twoD_first_time+i_t/frequency/10)
		twoD_peak_evolution_time_averaged = np.array(twoD_peak_evolution_time_averaged_2)
		twoD_peak_evolution_counts_averaged = np.array(twoD_peak_evolution_counts_averaged_2)
		twoD_peak_evolution_counts_std = np.array(twoD_peak_evolution_counts_std_2)
		twoD_peak_evolution_counts_delta_averaged = np.array(twoD_peak_evolution_counts_delta_averaged_2)
		twoD_peak_evolution_counts_delta_std = np.array(twoD_peak_evolution_counts_delta_std_2)

		temporal_mean_filter_footprint = [20,1,1]
		spatial_mean_filter_footprint = [1,7,7]
		full_mean_filter_footprint = (np.array(temporal_mean_filter_footprint) * np.array(spatial_mean_filter_footprint)).tolist()
		# temp = cp.deepcopy(twoD_peak_evolution_counts_std)
		temp = cp.deepcopy((twoD_peak_evolution_counts_delta_std**2 + twoD_peak_evolution_counts_before_std**2)**0.5)
		temp[np.isnan(temp)] = np.nanmin(temp)
		temp[temp<=0] = np.nanmax(temp)
		temp = 1/temp**2
		twoD_peak_evolution_counts_std_full_mean_filter = (full_mean_filter_footprint[0]*full_mean_filter_footprint[1]*full_mean_filter_footprint[2])**0.5 * 1/(generic_filter(temp,np.sum,size=full_mean_filter_footprint)**0.5)
		twoD_peak_evolution_counts_std = (temporal_mean_filter_footprint[0]*temporal_mean_filter_footprint[1]*temporal_mean_filter_footprint[2])**0.5 * 1/(generic_filter(temp,np.sum,size=temporal_mean_filter_footprint)**0.5)
		temp = cp.deepcopy(twoD_peak_evolution_counts_delta_std)
		temp[np.isnan(temp)] = np.nanmin(temp)
		temp[temp<=0] = np.nanmax(temp)
		temp = 1/temp**2
		twoD_peak_evolution_counts_delta_std_full_mean_filter = (full_mean_filter_footprint[0]*full_mean_filter_footprint[1]*full_mean_filter_footprint[2])**0.5 * 1/(generic_filter(temp,np.sum,size=full_mean_filter_footprint)**0.5)
		twoD_peak_evolution_counts_delta_std = (temporal_mean_filter_footprint[0]*temporal_mean_filter_footprint[1]*temporal_mean_filter_footprint[2])**0.5 * 1/(generic_filter(temp,np.sum,size=temporal_mean_filter_footprint)**0.5)

		# twoD_peak_evolution_counts_averaged = median_filter(twoD_peak_evolution_counts_averaged,size=mean_filter_footprint)
		# twoD_peak_evolution_counts_averaged_full_mean_filter = generic_filter(twoD_peak_evolution_counts_averaged,np.median,size=full_mean_filter_footprint)
		# twoD_peak_evolution_counts_averaged = generic_filter(twoD_peak_evolution_counts_averaged,np.median,size=temporal_mean_filter_footprint)
		# twoD_peak_evolution_counts_averaged_before_full_mean_filter = np.mean(twoD_peak_evolution_counts_averaged_full_mean_filter[:np.abs(twoD_peak_evolution_time_averaged+5/1000).argmin()],axis=0)
		# twoD_peak_evolution_counts_averaged_before = np.mean(twoD_peak_evolution_counts_averaged[:np.abs(twoD_peak_evolution_time_averaged+5/1000).argmin()],axis=0)
		# twoD_peak_evolution_counts_averaged_delta = twoD_peak_evolution_counts_averaged-twoD_peak_evolution_counts_averaged_before
		twoD_peak_evolution_counts_averaged_before = cp.deepcopy(twoD_peak_evolution_counts_before)
		twoD_peak_evolution_counts_averaged_delta = cp.deepcopy(twoD_peak_evolution_counts_delta_averaged)
		twoD_peak_evolution_counts_averaged = twoD_peak_evolution_counts_averaged_before + twoD_peak_evolution_counts_averaged_delta

		ani = coleval.movie_from_data(np.array([twoD_peak_evolution_counts_averaged[::10]]), 1/(np.median(np.diff(twoD_peak_evolution_time_averaged)))/10, integration=int_time/1000,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Counts [au]',time_offset=np.min(twoD_peak_evolution_time_averaged),prelude=pre_title+'Counts averaged among all second half of peaks no mean filter\n')
		figure_index+=1
		ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.mp4', fps=3, writer='ffmpeg',codec='mpeg4')
		plt.close()

		ani = coleval.movie_from_data(np.array([twoD_peak_evolution_counts_delta_averaged[::10]]), 1/(np.median(np.diff(twoD_peak_evolution_time_averaged)))/10, integration=int_time/1000,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Count increment [au]',time_offset=np.min(twoD_peak_evolution_time_averaged),prelude=pre_title+'Counts averaged among all second half of peaks no mean filter\n')
		figure_index+=1
		ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.mp4', fps=3, writer='ffmpeg',codec='mpeg4')
		plt.close()

		twoD_peak_evolution_counts_averaged_full_mean_filter = generic_filter(twoD_peak_evolution_counts_averaged,np.median,size=full_mean_filter_footprint)
		twoD_peak_evolution_counts_averaged = generic_filter(twoD_peak_evolution_counts_averaged,np.median,size=temporal_mean_filter_footprint)
		twoD_peak_evolution_counts_averaged_delta = generic_filter(twoD_peak_evolution_counts_averaged_delta,np.median,size=temporal_mean_filter_footprint)
		twoD_peak_evolution_counts_averaged_before_full_mean_filter = np.mean(twoD_peak_evolution_counts_averaged_full_mean_filter[:np.abs(twoD_peak_evolution_time_averaged+5/1000).argmin()],axis=0)


		# else:
		# 	for i_t in range(np.shape(twoD_peak_evolution_time)[1]):
		# 		select = np.logical_and(twoD_peak_evolution_time<twoD_first_time+(i_t+1/2)/frequency/10,twoD_peak_evolution_time>=twoD_first_time+(i_t-1/2)/frequency/10)
		# 		twoD_peak_evolution_time_averaged.append(np.mean(twoD_peak_evolution_time[select]))
		# 		twoD_peak_evolution_counts_averaged.append(np.mean(twoD_peak_evolution_counts[select],axis=0))
		# 	twoD_peak_evolution_time_averaged = np.array(twoD_peak_evolution_time_averaged)
		# 	twoD_peak_evolution_counts_averaged = np.array(twoD_peak_evolution_counts_averaged)

		ani = coleval.movie_from_data(np.array([twoD_peak_evolution_counts_averaged_full_mean_filter[::10]]), 1/(np.median(np.diff(twoD_peak_evolution_time_averaged)))/10, integration=int_time/1000,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Counts [au]',time_offset=np.min(twoD_peak_evolution_time_averaged),prelude=pre_title+'Counts averaged among all second half of peaks mean filter '+str(full_mean_filter_footprint)+'\n')
		figure_index+=1
		ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.mp4', fps=3, writer='ffmpeg',codec='mpeg4')
		plt.close()

		ani = coleval.movie_from_data(np.array([(twoD_peak_evolution_counts_averaged_full_mean_filter/twoD_peak_evolution_counts_std_full_mean_filter)[::10]]), 1/(np.median(np.diff(twoD_peak_evolution_time_averaged)))/10, integration=int_time/1000,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='SNR [au]',time_offset=np.min(twoD_peak_evolution_time_averaged),prelude=pre_title+'SNR averaged among all second half of peaks mean filter '+str(full_mean_filter_footprint)+'\n')
		figure_index+=1
		ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.mp4', fps=3, writer='ffmpeg',codec='mpeg4')
		plt.close()


		twoD_peak_evolution_counts_averaged[twoD_peak_evolution_counts_averaged<0] = 0
		twoD_peak_evolution_counts_averaged_full_mean_filter[twoD_peak_evolution_counts_averaged_full_mean_filter<0] = 0
		twoD_peak_evolution_temp_averaged = np.zeros_like(twoD_peak_evolution_counts_averaged)
		twoD_peak_evolution_temp_averaged_std_up = np.zeros_like(twoD_peak_evolution_counts_averaged)
		twoD_peak_evolution_temp_averaged_std_down = np.zeros_like(twoD_peak_evolution_counts_averaged)
		for i in range(len(twoD_peak_evolution_time_averaged)):
			# emissivity[i] = np.exp(counts_to_emissivity(corrected_frames[i]))
			if False:
				twoD_peak_evolution_temp_averaged[i] = np.exp(counts_to_temperature(twoD_peak_evolution_counts_averaged[i]))
				twoD_peak_evolution_temp_averaged_std_up[i] = np.exp(counts_to_temperature(twoD_peak_evolution_counts_averaged[i]+twoD_peak_evolution_counts_std[i]))
				twoD_peak_evolution_temp_averaged_std_down[i] = np.exp(counts_to_temperature(np.max([np.zeros_like(twoD_peak_evolution_counts_std[i]),twoD_peak_evolution_counts_averaged[i]-twoD_peak_evolution_counts_std[i]],axis=0)))
			elif True:	# process calculating the delta and the before separately. important for high temperature
				twoD_peak_evolution_temp_averaged[i] = np.exp(counts_to_temperature(twoD_peak_evolution_counts_averaged[i]))
				twoD_peak_evolution_temp_averaged_std_up[i] = np.exp(counts_to_temperature(twoD_peak_evolution_counts_averaged[i]+twoD_peak_evolution_counts_delta_std[i]))
				twoD_peak_evolution_temp_averaged_std_down[i] = np.exp(counts_to_temperature(np.max([np.zeros_like(twoD_peak_evolution_counts_delta_std[i]),twoD_peak_evolution_counts_averaged[i]-twoD_peak_evolution_counts_delta_std[i]],axis=0)))
		twoD_peak_evolution_temp_averaged_before = np.mean(twoD_peak_evolution_temp_averaged[:np.abs(twoD_peak_evolution_time_averaged+5/1000).argmin()],axis=0)
		twoD_peak_evolution_temp_averaged_delta = twoD_peak_evolution_temp_averaged-twoD_peak_evolution_temp_averaged_before
		twoD_peak_evolution_temp_averaged_std = (twoD_peak_evolution_temp_averaged_std_up-twoD_peak_evolution_temp_averaged_std_down)/2
		twoD_peak_evolution_temp_averaged_full_mean_filter = generic_filter(twoD_peak_evolution_temp_averaged,np.mean,size=spatial_mean_filter_footprint)
		twoD_peak_evolution_temp_averaged_delta_full_mean_filter = generic_filter(twoD_peak_evolution_temp_averaged_delta,np.mean,size=spatial_mean_filter_footprint)
		twoD_peak_evolution_temp_averaged_before_full_mean_filter = np.mean(twoD_peak_evolution_temp_averaged_full_mean_filter[:np.abs(twoD_peak_evolution_time_averaged+5/1000).argmin()],axis=0)

		# full_saved_file_dict = dict(np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz'))
		full_saved_file_dict['twoD_peak_evolution_temp_averaged_before'] = np.float32(twoD_peak_evolution_temp_averaged_before)
		full_saved_file_dict['twoD_peak_evolution_temp_averaged_delta'] = np.float32(twoD_peak_evolution_temp_averaged_delta)
		full_saved_file_dict['twoD_peak_evolution_temp_averaged'] = np.float32(twoD_peak_evolution_temp_averaged)
		full_saved_file_dict['twoD_peak_evolution_temp_averaged_std'] = np.float32(twoD_peak_evolution_temp_averaged_std)
		full_saved_file_dict['twoD_peak_evolution_time_averaged'] = np.float32(twoD_peak_evolution_time_averaged)
		full_saved_file_dict['twoD_peak_evolution_counts_averaged'] = np.float32(twoD_peak_evolution_counts_averaged)
		full_saved_file_dict['twoD_peak_evolution_counts_std_full_mean_filter'] = np.float32(twoD_peak_evolution_counts_std_full_mean_filter)
		full_saved_file_dict['twoD_peak_evolution_counts_averaged_full_mean_filter'] = np.float32(twoD_peak_evolution_counts_averaged_full_mean_filter)
		full_saved_file_dict['twoD_peak_evolution_temp_averaged_full_mean_filter'] = np.float32(twoD_peak_evolution_temp_averaged_full_mean_filter)
		full_saved_file_dict['twoD_peak_evolution_temp_averaged_delta_full_mean_filter'] = np.float32(twoD_peak_evolution_temp_averaged_delta_full_mean_filter)
		full_saved_file_dict['twoD_peak_evolution_counts_averaged_before'] = np.float32(twoD_peak_evolution_counts_averaged_before)
		full_saved_file_dict['twoD_peak_evolution_counts_averaged_delta'] = np.float32(twoD_peak_evolution_counts_averaged_delta)
		full_saved_file_dict['twoD_peak_evolution_counts_averaged_before_full_mean_filter'] = np.float32(twoD_peak_evolution_counts_averaged_before_full_mean_filter)
		full_saved_file_dict['twoD_peak_evolution_temp_averaged_before_full_mean_filter'] = np.float32(twoD_peak_evolution_temp_averaged_before_full_mean_filter)
		full_saved_file_dict['temporal_mean_filter_footprint'] = temporal_mean_filter_footprint
		full_saved_file_dict['spatial_mean_filter_footprint'] = spatial_mean_filter_footprint
		# np.savez_compressed(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace,**full_saved_file_dict)
		#
		# continue

	else:
		# twoD_peak_evolution_temp_averaged_before = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['twoD_peak_evolution_temp_averaged_before']
		# twoD_peak_evolution_temp_averaged_delta = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['twoD_peak_evolution_temp_averaged_delta']
		# twoD_peak_evolution_temp_averaged = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['twoD_peak_evolution_temp_averaged']
		# twoD_peak_evolution_temp_averaged_std = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['twoD_peak_evolution_temp_averaged_std']
		# twoD_peak_evolution_time_averaged = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['twoD_peak_evolution_time_averaged']
		# twoD_peak_evolution_counts_averaged = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['twoD_peak_evolution_counts_averaged']
		# twoD_peak_evolution_counts_std_full_mean_filter = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['twoD_peak_evolution_counts_std_full_mean_filter']
		# twoD_peak_evolution_counts_averaged_full_mean_filter = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['twoD_peak_evolution_counts_averaged_full_mean_filter']
		# twoD_peak_evolution_temp_averaged_full_mean_filter = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['twoD_peak_evolution_temp_averaged_full_mean_filter']
		# twoD_peak_evolution_temp_averaged_delta_full_mean_filter = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['twoD_peak_evolution_temp_averaged_delta_full_mean_filter']
		# twoD_peak_evolution_counts_averaged_before = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['twoD_peak_evolution_counts_averaged_before']
		# twoD_peak_evolution_counts_averaged_delta = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['twoD_peak_evolution_counts_averaged_delta']
		# twoD_peak_evolution_counts_averaged_before_full_mean_filter = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['twoD_peak_evolution_counts_averaged_before_full_mean_filter']
		# twoD_peak_evolution_temp_averaged_before_full_mean_filter = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['twoD_peak_evolution_temp_averaged_before_full_mean_filter']
		# temporal_mean_filter_footprint = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['temporal_mean_filter_footprint']
		# spatial_mean_filter_footprint = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['spatial_mean_filter_footprint']
		twoD_peak_evolution_temp_averaged_before = full_saved_file_dict['twoD_peak_evolution_temp_averaged_before']
		twoD_peak_evolution_temp_averaged_delta = full_saved_file_dict['twoD_peak_evolution_temp_averaged_delta']
		twoD_peak_evolution_temp_averaged = full_saved_file_dict['twoD_peak_evolution_temp_averaged']
		twoD_peak_evolution_temp_averaged_std = full_saved_file_dict['twoD_peak_evolution_temp_averaged_std']
		twoD_peak_evolution_time_averaged = full_saved_file_dict['twoD_peak_evolution_time_averaged']
		twoD_peak_evolution_counts_averaged = full_saved_file_dict['twoD_peak_evolution_counts_averaged']
		twoD_peak_evolution_counts_std_full_mean_filter = full_saved_file_dict['twoD_peak_evolution_counts_std_full_mean_filter']
		twoD_peak_evolution_counts_averaged_full_mean_filter = full_saved_file_dict['twoD_peak_evolution_counts_averaged_full_mean_filter']
		twoD_peak_evolution_temp_averaged_full_mean_filter = full_saved_file_dict['twoD_peak_evolution_temp_averaged_full_mean_filter']
		twoD_peak_evolution_temp_averaged_delta_full_mean_filter = full_saved_file_dict['twoD_peak_evolution_temp_averaged_delta_full_mean_filter']
		twoD_peak_evolution_counts_averaged_before = full_saved_file_dict['twoD_peak_evolution_counts_averaged_before']
		twoD_peak_evolution_counts_averaged_delta = full_saved_file_dict['twoD_peak_evolution_counts_averaged_delta']
		twoD_peak_evolution_counts_averaged_before_full_mean_filter = full_saved_file_dict['twoD_peak_evolution_counts_averaged_before_full_mean_filter']
		twoD_peak_evolution_temp_averaged_before_full_mean_filter = full_saved_file_dict['twoD_peak_evolution_temp_averaged_before_full_mean_filter']
		temporal_mean_filter_footprint = full_saved_file_dict['temporal_mean_filter_footprint']
		spatial_mean_filter_footprint = full_saved_file_dict['spatial_mean_filter_footprint']
		figure_index+=1
		figure_index+=1
		figure_index+=1

	after_long_start_time = 1.4	# ms
	after_long_end_time = 4	# ms
	after_short_start_time = 1	# ms
	after_short_end_time = 1.6	# ms
	full_mean_filter_footprint = (np.array(temporal_mean_filter_footprint) * np.array(spatial_mean_filter_footprint)).tolist()
	twoD_peak_evolution_temp_averaged_delta_after_long = np.mean(twoD_peak_evolution_temp_averaged_delta[np.abs(twoD_peak_evolution_time_averaged-after_long_start_time/1000).argmin():np.abs(twoD_peak_evolution_time_averaged-after_long_end_time/1000).argmin()],axis=0)
	# twoD_peak_evolution_temp_averaged_delta_2_4ms = np.mean(twoD_peak_evolution_temp_averaged_delta[np.abs(twoD_peak_evolution_time_averaged-(2)/1000).argmin():np.abs(twoD_peak_evolution_time_averaged-(4)/1000).argmin()],axis=0)
	twoD_peak_evolution_temp_averaged_delta_after_short = np.mean(twoD_peak_evolution_temp_averaged_delta[np.abs(twoD_peak_evolution_time_averaged-after_short_start_time/1000).argmin():np.abs(twoD_peak_evolution_time_averaged-after_short_end_time/1000).argmin()],axis=0)
	twoD_peak_evolution_temp_averaged_after_short = np.mean(twoD_peak_evolution_temp_averaged[np.abs(twoD_peak_evolution_time_averaged-after_short_start_time/1000).argmin():np.abs(twoD_peak_evolution_time_averaged-after_short_end_time/1000).argmin()],axis=0)
	twoD_peak_evolution_temp_averaged_after_long = np.mean(twoD_peak_evolution_temp_averaged[np.abs(twoD_peak_evolution_time_averaged-after_long_start_time/1000).argmin():np.abs(twoD_peak_evolution_time_averaged-after_long_end_time/1000).argmin()],axis=0)

	ani = coleval.movie_from_data(np.array([twoD_peak_evolution_temp_averaged_full_mean_filter[::10]]), 1/(np.median(np.diff(twoD_peak_evolution_time_averaged)))/10, integration=int_time/1000,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Temperature [°C]',time_offset=np.min(twoD_peak_evolution_time_averaged),prelude=pre_title+'Temperature averaged among all second half of peaks mean filter '+str(full_mean_filter_footprint)+'\n')
	figure_index+=1
	ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.mp4', fps=3, writer='ffmpeg',codec='mpeg4')
	plt.close()

	p2d_h = 25/77	# 25 mm == 77 pixels without sample tilting
	p2d_v = 25/92	# 25 mm == 92 pixels
	max_diameter_to_fit = 24	# mm
	def gaussian_2D_fitting(spatial_coord,*args):
		x = spatial_coord[0]	# horizontal
		y = spatial_coord[1]	# vertical
		# out = args[1]*np.exp(- (((x-args[3])*p2d_v/p2d_h)**2 + (y-args[2])**2)/(2*(args[4]**2)) ) + args[0]
		out = args[0]*np.exp(- (((x-args[2])*p2d_v/p2d_h)**2 + (y-args[1])**2)/(2*(args[3]**2)) )
		select = ((x-args[2])*p2d_h)**2+((y-args[1])*p2d_v)**2>(max_diameter_to_fit/2/2)**2
		out[select]=twoD_peak_evolution_temp_averaged_delta_after_short[select]
		return (out-twoD_peak_evolution_temp_averaged_delta_after_short).flatten()


	spatial_coord=np.meshgrid(np.arange(np.shape(twoD_peak_evolution_temp_averaged_delta_after_short)[1]),np.arange(np.shape(twoD_peak_evolution_temp_averaged_delta_after_short)[0]))
	def select_circulra_region(args,sigma=max_diameter_to_fit/2/1.5/1.4):
		x = spatial_coord[0]	# horizontal
		y = spatial_coord[1]	# vertical
		diameter_2 = ((x-args[1])*p2d_h)**2+((y-args[0])*p2d_v)**2
		# sigma = max_diameter_to_fit/2/1.5/1.4	# the 1.4 factor is required
		select = np.exp(-(diameter_2 / (2*((sigma)**2)))**40)	# super gaussian that cuts at max_diameter_to_fit/2/2, the 1.4 factor is required
		return select

	def gaussian_2D_fitting_plus_gradient(full_output,data):
		def internal(args):
			x = spatial_coord[0]	# horizontal
			y = spatial_coord[1]	# vertical
			diameter_2 = ((x-args[2])*p2d_h)**2+((y-args[1])*p2d_v)**2
			out = args[0]*np.exp(- diameter_2/(2*(args[3]**2)) )
			sigma = max_diameter_to_fit/2/2/1.4	# the 1.4 factor is required

			select = np.exp(-(diameter_2 / (2*((sigma)**2)))**40)	# super gaussian that cuts at max_diameter_to_fit/2/2, the 1.4 factor is required
			full_out = out-data
			grad_a0 = np.sum( 2*full_out*np.exp(- diameter_2/(2*(args[3]**2)) )*select )
			grad_a1 = -np.sum( 2*full_out*out*(-1/(2*(args[3]**2)))*2*(p2d_v**2)*(y-args[1])*select + (full_out**2)*select*(-40*(diameter_2 / (2*((sigma)**2)))**39)*(1/(2*((sigma)**2)))*2*(p2d_v**2)*(y-args[1]) )
			grad_a2 = -np.sum( 2*full_out*out*(-1/(2*(args[3]**2)))*2*(p2d_h**2)*(x-args[2])*select + (full_out**2)*select*(-40*(diameter_2 / (2*((sigma)**2)))**39)*(1/(2*((sigma)**2)))*2*(p2d_h**2)*(x-args[2]) )
			grad_a3 = np.sum( 2*full_out*out*( diameter_2/(args[3]**3))*select )
			out = (full_out**2)*select
			if full_output==True:
				return np.sum(out),np.array([grad_a0,grad_a1,grad_a2,grad_a3])
			else:
				return np.sum(out)
		return internal

	def make_standatd_fit_output(function,x_optimal):
		hessian = nd.Hessian(function)
		hessian = hessian(x_optimal)
		covariance = np.linalg.inv(hessian)
		for i in range(len(covariance)):
			covariance[i,i] = np.abs(covariance[i,i])
		fit = [x_optimal,covariance]
		return fit

	if False:	# only for testinf the prob_and_gradient function
		target = 1
		scale = 1e-4
		# guess[target] = 1e5
		temp1 = gaussian_2D_fitting_plus_gradient(True,twoD_peak_evolution_temp_averaged_delta_after_short)(guess)
		guess[target] +=scale
		temp2 = gaussian_2D_fitting_plus_gradient(True,twoD_peak_evolution_temp_averaged_delta_after_short)(guess)
		guess[target] += -2*scale
		temp3 = gaussian_2D_fitting_plus_gradient(True,twoD_peak_evolution_temp_averaged_delta_after_short)(guess)
		guess[target] += scale
		print('calculated derivated of %.7g vs true of %.7g' %(temp1[1][target],((temp2[0]-temp3[0])/(2*scale))))

	if False:
		guess = [twoD_peak_evolution_temp_averaged_delta_after_short.max()-max(0,twoD_peak_evolution_temp_averaged_delta_after_short.min()),IR_shape[0],IR_shape[1],IR_shape[3]]
		bds=[[0,max(0,IR_shape[0]-IR_shape[3]),max(0,IR_shape[1]-IR_shape[3]),min(IR_shape[2],10)],[np.inf,min(np.shape(twoD_peak_evolution_temp_averaged_delta_after_short)[0],IR_shape[0]+IR_shape[3]),min(np.shape(twoD_peak_evolution_temp_averaged_delta_after_short)[1],IR_shape[1]+IR_shape[3]),np.max(twoD_peak_evolution_temp_averaged_delta_after_short.shape)/2]]
		spatial_coord=np.meshgrid(np.arange(np.shape(twoD_peak_evolution_temp_averaged_delta_after_short)[1]),np.arange(np.shape(twoD_peak_evolution_temp_averaged_delta_after_short)[0]))
		fit1 = curve_fit(gaussian_2D_fitting, spatial_coord, np.zeros_like(twoD_peak_evolution_temp_averaged_delta_after_short.flatten()), guess,absolute_sigma=True,bounds=bds,maxfev=int(1e4))
	else:
		guess = np.array([(twoD_peak_evolution_temp_averaged_delta_after_short*select_circulra_region([IR_shape[0],IR_shape[1]])).max()-max(0,(twoD_peak_evolution_temp_averaged_delta_after_short*select_circulra_region([IR_shape[0],IR_shape[1]])).min()),IR_shape[0],IR_shape[1],IR_shape[3]*p2d_v])
		bds=[[0,np.inf],[max(0,IR_shape[0]-IR_shape[3]),min(np.shape(twoD_peak_evolution_temp_averaged_delta_after_short)[0],IR_shape[0]+IR_shape[3])],[max(0,IR_shape[1]-IR_shape[3]),min(np.shape(twoD_peak_evolution_temp_averaged_delta_after_short)[1],IR_shape[1]+IR_shape[3])],[min(IR_shape[2],10)*p2d_v,np.max(twoD_peak_evolution_temp_averaged_delta_after_short.shape)/2*p2d_v]]
		x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(gaussian_2D_fitting_plus_gradient(True,twoD_peak_evolution_temp_averaged_delta_after_short), x0=guess, iprint=0, factr=1e0, pgtol=1e-6,bounds=bds)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
		fit1 = make_standatd_fit_output(gaussian_2D_fitting_plus_gradient(False,twoD_peak_evolution_temp_averaged_delta_after_short),x_optimal)
		fit1[0][-1] /= p2d_v
		fit1[1][-1,:] /= p2d_v
		fit1[1][:,-1] /= p2d_v


	# area calculation from delta Temperature
	if False:
		p2d_h = 25/77	# 25 mm == 77 pixels without sample tilting
		p2d_v = 25/92	# 25 mm == 92 pixels
		max_diameter_to_fit = 24	# mm
		def gaussian_2D_fitting(spatial_coord,*args):
			x = spatial_coord[0]	# horizontal
			y = spatial_coord[1]	# vertical
			# out = args[1]*np.exp(- (((x-args[3])*p2d_v/p2d_h)**2 + (y-args[2])**2)/(2*(args[4]**2)) ) + args[0]
			out = args[0]*np.exp(- (((x-args[2])*p2d_v/p2d_h)**2 + (y-args[1])**2)/(2*(args[3]**2)) )
			select = ((x-args[2])*p2d_h)**2+((y-args[1])*p2d_v)**2>(max_diameter_to_fit/2)**2
			out[select]=twoD_peak_evolution_temp_averaged_delta_after_long[select]
			return (out-twoD_peak_evolution_temp_averaged_delta_after_long).flatten()

		# guess = [0,twoD_peak_evolution_temp_averaged_delta_after_long.max()-max(0,twoD_peak_evolution_temp_averaged_delta_after_long.min()),IR_shape[0],IR_shape[1],IR_shape[3]]
		# bds=[[0,0,max(0,IR_shape[0]-IR_shape[3]),max(0,IR_shape[1]-IR_shape[3]),min(IR_shape[2],10)],[np.inf,np.inf,min(np.shape(twoD_peak_evolution_temp_averaged_delta_after_long)[0],IR_shape[0]+IR_shape[3]),min(np.shape(twoD_peak_evolution_temp_averaged_delta_after_long)[1],IR_shape[1]+IR_shape[3]),np.max(twoD_peak_evolution_temp_averaged_delta_after_long.shape)/2]]
		# spatial_coord=np.meshgrid(np.arange(np.shape(twoD_peak_evolution_temp_averaged_delta_after_long)[1]),np.arange(np.shape(twoD_peak_evolution_temp_averaged_delta_after_long)[0]))
		# fit1 = curve_fit(gaussian_2D_fitting, spatial_coord, np.zeros_like(twoD_peak_evolution_temp_averaged_delta_after_long.flatten()), guess,bounds=bds,maxfev=int(1e4))
		x = spatial_coord[0]	# horizontal
		y = spatial_coord[1]	# vertical
		temp = max(1,np.median(np.sort(twoD_peak_evolution_temp_averaged_delta_after_long[((x-fit1[0][1])*p2d_h)**2+((y-fit1[0][2])*p2d_v)**2<(max_diameter_to_fit/2)**2])[:10]))
		guess = [max(0,twoD_peak_evolution_temp_averaged_delta_after_long.max()-temp),fit1[0][1],fit1[0][2],fit1[0][3]]
		bds=[[0,fit1[0][1]-0.1,fit1[0][2]-0.1,min(IR_shape[2],10)],[np.inf,fit1[0][1]+0.1,fit1[0][2]+0.1,np.max(twoD_peak_evolution_temp_averaged_delta_after_long.shape)/2]]
		spatial_coord=np.meshgrid(np.arange(np.shape(twoD_peak_evolution_temp_averaged_delta_after_long)[1]),np.arange(np.shape(twoD_peak_evolution_temp_averaged_delta_after_long)[0]))
		print(guess);print(bds)
		fit2 = curve_fit(gaussian_2D_fitting, spatial_coord, np.zeros_like(twoD_peak_evolution_temp_averaged_delta_after_long.flatten()), guess,absolute_sigma=True,bounds=bds,maxfev=int(1e5))
		# area_of_interest_sigma = np.pi*( 2**0.5 *(correlated_values(fit2[0],fit2[1])[3])*p2d_v/1000)**2	# Area equivalent to all the overtemperature profile collapsed in a homogeneous circular spot
		# area_of_interest_sigma = area_of_interest_sigma + ufloat(0,0.1*area_of_interest_sigma.nominal_value)	# this to add the uncertainly coming from the method itself
		# area_of_interest_radius = 2**0.5 *fit2[0][3]	# pixels
	else:
		temp = max(1,np.median(np.sort(twoD_peak_evolution_temp_averaged_delta_after_long*select_circulra_region([IR_shape[0],IR_shape[1]]))[:10]))
		guess = [max(0,twoD_peak_evolution_temp_averaged_delta_after_long.max()-temp),fit1[0][1],fit1[0][2],fit1[0][3]]
		bds=[[0,np.inf],[fit1[0][1]-0.1,fit1[0][1]+0.1],[fit1[0][2]-0.1,fit1[0][2]+0.1],[min(IR_shape[2],10)*p2d_v,np.max(twoD_peak_evolution_temp_averaged_delta_after_long.shape)/2*p2d_v]]
		x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(gaussian_2D_fitting_plus_gradient(True,twoD_peak_evolution_temp_averaged_delta_after_long), x0=guess, iprint=0, factr=1e0, pgtol=1e-6,bounds=bds)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
		fit2 = make_standatd_fit_output(gaussian_2D_fitting_plus_gradient(False,twoD_peak_evolution_temp_averaged_delta_after_long),x_optimal)
		fit2[0][-1] /= p2d_v
		fit2[1][-1,:] /= p2d_v
		fit2[1][:,-1] /= p2d_v


	# here I refine the area from measuring it at each time ang getting a weighted average
	if False:
		spatial_coord=np.meshgrid(np.arange(np.shape(twoD_peak_evolution_temp_averaged_delta)[2]),np.arange(np.shape(twoD_peak_evolution_temp_averaged_delta)[1]))
		x = spatial_coord[0]	# horizontal
		y = spatial_coord[1]	# vertical
		select = ((x-fit2[0][2])*p2d_h)**2+((y-fit2[0][1])*p2d_v)**2>(max_diameter_to_fit/2)**2
		def gaussian_2D_fitting_reduced_master(temperature_distridution_to_fit):
			def gaussian_2D_fitting_reduced(spatial_coord,*args):
				out = args[0]*np.exp(- (((x-fit2[0][2])*p2d_v/p2d_h)**2 + (y-fit2[0][1])**2)/(2*(args[1]**2)) )# + args[0]
				out[select]=temperature_distridution_to_fit[select]
				return (out-temperature_distridution_to_fit).flatten()
			return gaussian_2D_fitting_reduced
		bds=[[0,fit2[0][3]/2],[twoD_peak_evolution_counts_averaged_delta.max()*2,fit2[0][3]*2]]
		guess = [twoD_peak_evolution_counts_averaged_delta.mean(),fit2[0][3]]
	else:
		guess = [max(0,twoD_peak_evolution_counts_averaged_delta.mean()),fit2[0][1],fit2[0][2],fit2[0][3]]
		bds = [[0,twoD_peak_evolution_counts_averaged_delta.max()*2],[fit2[0][1]*0.99,fit2[0][1]*1.01],[fit2[0][2]*0.99,fit2[0][2]*1.01],[fit2[0][3]/2*p2d_v,fit2[0][3]*2*p2d_v]]
	ensamble_peak_shape_counts_fitted = []
	ensamble_peak_shape_counts_fitted_std = []
	for temperature_distridution_to_fit in twoD_peak_evolution_counts_averaged_delta:
		print(len(ensamble_peak_shape_counts_fitted))
		if False:
			fit = curve_fit(gaussian_2D_fitting_reduced_master(temperature_distridution_to_fit), spatial_coord, np.zeros_like(temperature_distridution_to_fit.flatten()), guess,absolute_sigma=True,bounds=bds,maxfev=int(1e4))
			guess = fit[0]
		else:
			x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(gaussian_2D_fitting_plus_gradient(True,temperature_distridution_to_fit), x0=guess, iprint=0, factr=1e0, pgtol=1e-6,bounds=bds)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
			guess = cp.deepcopy(x_optimal)
			fit = make_standatd_fit_output(gaussian_2D_fitting_plus_gradient(False,temperature_distridution_to_fit),x_optimal)
			fit[0][-1] /= p2d_v
			fit[1][-1,:] /= p2d_v
			fit[1][:,-1] /= p2d_v
			fit[0] = np.array([fit[0][0],fit[0][-1]])
			fit[1] = np.array([[fit[1][0,0],fit[1][0,-1]],[fit[1][-1,0],fit[1][-1,-1]]])
		ensamble_peak_shape_counts_fitted.append(fit[0])
		ensamble_peak_shape_counts_fitted_std.append(np.abs(np.diag(fit[1]))**0.5)
		# print(guess)
	ensamble_peak_shape_counts_fitted = np.array(ensamble_peak_shape_counts_fitted)
	ensamble_peak_shape_counts_fitted_std = np.array(ensamble_peak_shape_counts_fitted_std)
	full_saved_file_dict['average_pulse_fitted_data'] = dict([])
	full_saved_file_dict['average_pulse_fitted_data']['counts'] = dict([])
	full_saved_file_dict['average_pulse_fitted_data']['counts']['delta'] = ensamble_peak_shape_counts_fitted[:,0]
	full_saved_file_dict['average_pulse_fitted_data']['counts']['delta_std'] = ensamble_peak_shape_counts_fitted_std[:,0]
	full_saved_file_dict['average_pulse_fitted_data']['counts']['sigma'] = ensamble_peak_shape_counts_fitted[:,1]
	full_saved_file_dict['average_pulse_fitted_data']['counts']['sigma_std'] = ensamble_peak_shape_counts_fitted_std[:,1]
	# select = ((x-fit2_absolute[0][3])*p2d_h)**2+((y-fit2_absolute[0][2])*p2d_v)**2>(max_diameter_to_fit/4)**2
	# def gaussian_2D_fitting_reduced_master(temperature_distridution_to_fit):
	# 	def gaussian_2D_fitting_reduced(spatial_coord,*args):
	# 		out = args[0]*np.exp(- (((x-fit2_absolute[0][3])*p2d_v/p2d_h)**2 + (y-fit2_absolute[0][2])**2)/(2*(args[1]**2)) ) + args[2]
	# 		out[select]=temperature_distridution_to_fit[select]
	# 		return (out-temperature_distridution_to_fit).flatten()
	# 	return gaussian_2D_fitting_reduced
	# ensamble_peak_shape_temp_fitted = []
	# guess = [twoD_peak_evolution_temp_averaged.mean(),fit2_absolute[0][4],20]
	# bds=[[0,min(fit2[0][3]/2,fit2_absolute[0][4]/3),20],[twoD_peak_evolution_temp_averaged.max()*1.5,max(fit2[0][3]*2,fit2_absolute[0][4]*2),twoD_peak_evolution_temp_averaged_before.max()]]
	# for temperature_distridution_to_fit in twoD_peak_evolution_temp_averaged:
	# 	fit = curve_fit(gaussian_2D_fitting_reduced_master(temperature_distridution_to_fit), spatial_coord, np.zeros_like(temperature_distridution_to_fit.flatten()), guess,bounds=bds,maxfev=int(1e4))
	# 	ensamble_peak_shape_temp_fitted.append(fit[0])
	# 	guess = fit[0]
	# 	# print(guess)
	# ensamble_peak_shape_temp_fitted = np.array(ensamble_peak_shape_temp_fitted)
	if False:
		guess = [twoD_peak_evolution_temp_averaged_delta.mean(),fit2[0][3]]
		bds=[[0,fit2[0][3]/2],[twoD_peak_evolution_temp_averaged_delta.max()*2,fit2[0][3]*2]]
	else:
		guess = [max(0,twoD_peak_evolution_temp_averaged_delta.mean()),fit2[0][1],fit2[0][2],fit2[0][3]]
		bds=[[0,twoD_peak_evolution_temp_averaged_delta.max()*2],[fit2[0][1]*0.99,fit2[0][1]*1.01],[fit2[0][2]*0.99,fit2[0][2]*1.01],[fit2[0][3]/2*p2d_v,fit2[0][3]*2*p2d_v]]
	ensamble_peak_shape_temp_fitted = []
	ensamble_peak_shape_temp_fitted_std = []
	for temperature_distridution_to_fit in twoD_peak_evolution_temp_averaged_delta:
		if False:
			fit = curve_fit(gaussian_2D_fitting_reduced_master(temperature_distridution_to_fit), spatial_coord, np.zeros_like(temperature_distridution_to_fit.flatten()), guess,absolute_sigma=True,bounds=bds,maxfev=int(1e4))
			guess = fit[0]
		else:
			x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(gaussian_2D_fitting_plus_gradient(True,temperature_distridution_to_fit), x0=guess, iprint=0, factr=1e0, pgtol=1e-6,bounds=bds)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
			guess = cp.deepcopy(x_optimal)
			fit = make_standatd_fit_output(gaussian_2D_fitting_plus_gradient(False,temperature_distridution_to_fit),x_optimal)
			fit[0][-1] /= p2d_v
			fit[1][-1,:] /= p2d_v
			fit[1][:,-1] /= p2d_v
			fit[0] = np.array([fit[0][0],fit[0][-1]])
			fit[1] = np.array([[fit[1][0,0],fit[1][0,-1]],[fit[1][-1,0],fit[1][-1,-1]]])
		ensamble_peak_shape_temp_fitted.append(fit[0])
		ensamble_peak_shape_temp_fitted_std.append(np.diag(fit[1])**0.5)
		# print(guess)
	ensamble_peak_shape_temp_fitted = np.array(ensamble_peak_shape_temp_fitted)
	ensamble_peak_shape_temp_fitted_std = np.array(ensamble_peak_shape_temp_fitted_std)
	full_saved_file_dict['average_pulse_peak_data']=dict([])
	full_saved_file_dict['average_pulse_fitted_data']['temperature'] = dict([])
	full_saved_file_dict['average_pulse_fitted_data']['temperature']['delta'] = ensamble_peak_shape_temp_fitted[:,0]
	full_saved_file_dict['average_pulse_fitted_data']['temperature']['delta_std'] = ensamble_peak_shape_temp_fitted_std[:,0]
	full_saved_file_dict['average_pulse_fitted_data']['temperature']['sigma'] = ensamble_peak_shape_temp_fitted[:,1]
	full_saved_file_dict['average_pulse_fitted_data']['temperature']['sigma_std'] = ensamble_peak_shape_temp_fitted_std[:,1]

	selected_area_average = np.logical_and(np.logical_and(np.logical_and(np.logical_and(twoD_peak_evolution_time_averaged>1.5e-3,twoD_peak_evolution_time_averaged<30e-3),np.isfinite(ensamble_peak_shape_temp_fitted[:,1])),np.isfinite(ensamble_peak_shape_temp_fitted_std[:,1])),ensamble_peak_shape_temp_fitted_std[:,1]>0)
	selected_area_average = np.logical_and(np.logical_and(np.logical_and(selected_area_average,np.abs(ensamble_peak_shape_temp_fitted[:,1]-fit2[0][3]*2)>1e-3),np.abs(ensamble_peak_shape_temp_fitted[:,1]-fit2[0][3]/2)>1e-3),ensamble_peak_shape_temp_fitted_std[:,1]>1e-5)
	averaged_sigma = np.nansum(ensamble_peak_shape_temp_fitted[:,1][selected_area_average]*(1/ensamble_peak_shape_temp_fitted_std[:,1][selected_area_average]**2))/np.nansum(1/ensamble_peak_shape_temp_fitted_std[:,1][selected_area_average]**2)
	averaged_sigma_sigma = (np.nanstd(ensamble_peak_shape_temp_fitted[:,1][selected_area_average])**2 + (np.nansum((1/ensamble_peak_shape_temp_fitted_std[:,1][selected_area_average])**2)**0.5/(np.nansum(1/ensamble_peak_shape_temp_fitted_std[:,1][selected_area_average]**2)))**2)**0.5
	# area_of_interest_sigma = np.pi*( 2**0.5 *ufloat(averaged_sigma,averaged_sigma_sigma)*p2d_v/1000)**2	# Area equivalent to all the overtemperature profile collapsed in a homogeneous circular spot
	# area_of_interest_sigma = area_of_interest_sigma + ufloat(0,0.1*area_of_interest_sigma.nominal_value)	# this to add the uncertainly coming from the method itself
	# area_of_interest_radius = 2**0.5 *fit2[0][3]	# pixels

	# full_saved_file_dict = dict(np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz'))
	fit_of_plasma_centre_and_area = cp.deepcopy(fit2[0])
	fit_of_plasma_centre_and_area_sigma = cp.deepcopy(fit2[1])
	full_saved_file_dict['fit_of_plasma_centre_and_area'] = fit_of_plasma_centre_and_area
	full_saved_file_dict['fit_of_plasma_centre_and_area_sigma'] = fit_of_plasma_centre_and_area_sigma
	full_saved_file_dict['average_of_the_area_fits'] = cp.deepcopy(averaged_sigma)
	full_saved_file_dict['average_of_the_area_fits_sigma'] = cp.deepcopy(averaged_sigma_sigma)
	# np.savez_compressed(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace,**full_saved_file_dict)

	if np.isfinite(averaged_sigma):
		fit2[0][3] = averaged_sigma
		fit2[1][3,3] = averaged_sigma_sigma**2

	area_of_interest_sigma = np.pi*( 2**0.5 *(correlated_values(fit2[0],fit2[1])[3])*p2d_v/1000)**2	# Area equivalent to all the overtemperature profile collapsed in a homogeneous circular spot
	area_of_interest_sigma = area_of_interest_sigma + ufloat(0,0.1*area_of_interest_sigma.nominal_value)	# this to add the uncertainly coming from the method itself
	area_of_interest_radius = 2**0.5 *fit2[0][3]	# pixels

	# calculation of area in SS
	SS_area_fit_ok = False
	try:	# I try because I'm not sure I'll be able to always find it
		# temph=np.arange(-max_diameter_to_fit/2/3,max_diameter_to_fit/2/3+p2d_h,p2d_h)
		# tempv=np.arange(-max_diameter_to_fit/2/3,max_diameter_to_fit/2/3+p2d_v,p2d_v)
		# temph,tempv=np.meshgrid(temph,tempv)
		# footprint = (temph**2+tempv**2)**0.5<max_diameter_to_fit/2/3
		# a=generic_filter(twoD_peak_evolution_temp_averaged_before,np.sum,footprint=footprint)
		if False:
			def gaussian_2D_fitting(spatial_coord,*args):
				x = spatial_coord[0]	# horizontal
				y = spatial_coord[1]	# vertical
				out = args[1]*np.exp(- (((x-args[3])*p2d_v/p2d_h)**2 + (y-args[2])**2)/(2*(args[4]**2)) ) + args[0]
				select = ((x-args[3])*p2d_h)**2+((y-args[2])*p2d_v)**2>(max_diameter_to_fit/2/2)**2
				out[select]=twoD_peak_evolution_temp_averaged_before[select]
				return (out-twoD_peak_evolution_temp_averaged_before).flatten()

			guess = [20,(twoD_peak_evolution_temp_averaged_before*select_circulra_region([IR_shape[0],IR_shape[1]])).max()-max(0,(twoD_peak_evolution_temp_averaged_before*select_circulra_region([IR_shape[0],IR_shape[1]])).min()),fit_of_plasma_centre_and_area[1],fit_of_plasma_centre_and_area[2],fit_of_plasma_centre_and_area[3]]
			bds=[[20,0,max(0,IR_shape[0]-IR_shape[3]),max(0,IR_shape[1]-IR_shape[3]),min(IR_shape[2],10)],[np.inf,np.inf,min(np.shape(twoD_peak_evolution_temp_averaged_before)[0],IR_shape[0]+IR_shape[3]),min(np.shape(twoD_peak_evolution_temp_averaged_before)[1],IR_shape[1]+IR_shape[3]),np.max(twoD_peak_evolution_temp_averaged_before.shape)/2]]
			spatial_coord=np.meshgrid(np.arange(np.shape(twoD_peak_evolution_temp_averaged_before)[1]),np.arange(np.shape(twoD_peak_evolution_temp_averaged_before)[0]))
			fit1_SS = curve_fit(gaussian_2D_fitting, spatial_coord, np.zeros_like(twoD_peak_evolution_temp_averaged_before.flatten()), guess,absolute_sigma=True,bounds=bds,maxfev=int(1e4))

			def gaussian_2D_fitting(spatial_coord,*args):
				x = spatial_coord[0]	# horizontal
				y = spatial_coord[1]	# vertical
				out = args[1]*np.exp(- (((x-args[3])*p2d_v/p2d_h)**2 + (y-args[2])**2)/(2*(args[4]**2)) ) + args[0]
				select = ((x-args[3])*p2d_h)**2+((y-args[2])*p2d_v)**2>(max_diameter_to_fit/2*0.7)**2
				out[select]=twoD_peak_evolution_temp_averaged_before[select]
				return (out-twoD_peak_evolution_temp_averaged_before).flatten()

			guess = [20,(twoD_peak_evolution_temp_averaged_before*select_circulra_region([IR_shape[0],IR_shape[1]])).max()-max(0,(twoD_peak_evolution_temp_averaged_before*select_circulra_region([IR_shape[0],IR_shape[1]])).min()),fit1_SS[0][2],fit1_SS[0][3],IR_shape[3]]
			bds=[[20,0,fit1_SS[0][2]-0.1,fit1_SS[0][3]-0.1,min(IR_shape[2],10)],[np.inf,np.inf,fit1_SS[0][2]+0.1,fit1_SS[0][3]+0.1,np.max(twoD_peak_evolution_temp_averaged_before.shape)/2]]
			spatial_coord=np.meshgrid(np.arange(np.shape(twoD_peak_evolution_temp_averaged_before)[1]),np.arange(np.shape(twoD_peak_evolution_temp_averaged_before)[0]))
			fit2_SS = curve_fit(gaussian_2D_fitting, spatial_coord, np.zeros_like(twoD_peak_evolution_temp_averaged_before.flatten()), guess,absolute_sigma=True,bounds=bds,maxfev=int(1e4))
		else:
			def gaussian_2D_fitting_plus_gradient_SS(full_output,data):
				def internal(args):
					x = spatial_coord[0]	# horizontal
					y = spatial_coord[1]	# vertical
					diameter_2 = ((x-args[3])*p2d_h)**2+((y-args[2])*p2d_v)**2
					out = args[1]*np.exp(- diameter_2/(2*(args[4]**2)) ) + args[0]
					sigma = max_diameter_to_fit/2/1.5/1.4	# the 1.4 factor is required


					select = np.exp(-(diameter_2 / (2*((sigma)**2)))**40)	# super gaussian that cuts at max_diameter_to_fit/2/2, the 1.4 factor is required
					full_out = out-data
					grad_a0 = np.sum( 2*full_out*select )
					grad_a1 = np.sum( 2*full_out*np.exp(- diameter_2/(2*(args[4]**2)) )*select )
					grad_a2 = -np.sum( 2*full_out*out*(-1/(2*(args[4]**2)))*2*(p2d_v**2)*(y-args[2])*select + (full_out**2)*select*(-40*(diameter_2 / (2*((sigma)**2)))**39)*(1/(2*((sigma)**2)))*2*(p2d_v**2)*(y-args[2]) )
					grad_a3 = -np.sum( 2*full_out*out*(-1/(2*(args[4]**2)))*2*(p2d_h**2)*(x-args[3])*select + (full_out**2)*select*(-40*(diameter_2 / (2*((sigma)**2)))**39)*(1/(2*((sigma)**2)))*2*(p2d_h**2)*(x-args[3]) )
					grad_a4 = np.sum( 2*full_out*out*( diameter_2/(args[4]**3))*select )
					out = (full_out**2)*select
					if full_output==True:
						return np.sum(out),np.array([grad_a0,grad_a1,grad_a2,grad_a3,grad_a4])
					else:
						return np.sum(out)
				return internal

			if False:	# only for testinf the prob_and_gradient function
				target = 0
				scale = 1e-4
				# guess[target] = 1e5
				temp1 = gaussian_2D_fitting_plus_gradient_SS(True,twoD_peak_evolution_temp_averaged_before)(guess)
				guess[target] +=scale
				temp2 = gaussian_2D_fitting_plus_gradient_SS(True,twoD_peak_evolution_temp_averaged_before)(guess)
				guess[target] += -2*scale
				temp3 = gaussian_2D_fitting_plus_gradient_SS(True,twoD_peak_evolution_temp_averaged_before)(guess)
				guess[target] += scale
				print('calculated derivated of %.7g vs true of %.7g' %(temp1[1][target],((temp2[0]-temp3[0])/(2*scale))))

			guess = [20,(twoD_peak_evolution_temp_averaged_before*select_circulra_region([IR_shape[0],IR_shape[1]])).max()-max(0,(twoD_peak_evolution_temp_averaged_before*select_circulra_region([IR_shape[0],IR_shape[1]])).min()),fit_of_plasma_centre_and_area[1],fit_of_plasma_centre_and_area[2],fit_of_plasma_centre_and_area[3]*p2d_v]
			bds=[[20,np.inf],[0,np.inf],[max(0,IR_shape[0]-IR_shape[3]),min(np.shape(twoD_peak_evolution_temp_averaged_before)[0],IR_shape[0]+IR_shape[3])],[max(0,IR_shape[1]-IR_shape[3]),min(np.shape(twoD_peak_evolution_temp_averaged_before)[1],IR_shape[1]+IR_shape[3])],[min(IR_shape[2],10)*p2d_v,np.max(twoD_peak_evolution_temp_averaged_before.shape)/2*p2d_v]]
			guess[-1] *= p2d_v
			x_optimal, y_opt, opt_info = scipy.optimize.fmin_l_bfgs_b(gaussian_2D_fitting_plus_gradient_SS(True,twoD_peak_evolution_temp_averaged_before), x0=guess, iprint=0, factr=1e0, pgtol=1e-6,bounds=bds)#,m=1000, maxls=1000, pgtol=1e-10, factr=1e0)#,approx_grad = True)
			fit2_SS = make_standatd_fit_output(gaussian_2D_fitting_plus_gradient_SS(False,twoD_peak_evolution_temp_averaged_before),x_optimal)
			fit2_SS[0][-1] /= p2d_v
			fit2_SS[1][-1,:] /= p2d_v
			fit2_SS[1][:,-1] /= p2d_v

			fit1_SS = fit2_SS



		area_of_interest_sigma_SS = 1/1.2*np.pi*( 2**0.5 *(correlated_values(fit2_SS[0],fit2_SS[1])[4])*p2d_v/1000)**2	# Area equivalent to all the overtemperature profile collapsed in a homogeneous circular spot
		area_of_interest_sigma_SS = area_of_interest_sigma_SS + ufloat(0,0.1*area_of_interest_sigma_SS.nominal_value)	# this to add the uncertainly coming from the method itself
		area_of_interest_radius_SS = 2**0.5 *fit2_SS[0][4]	# pixels
		full_saved_file_dict['SS_fit_of_plasma_centre_and_area'] = fit2_SS[0]
		full_saved_file_dict['SS_fit_of_plasma_centre_and_area_sigma'] = fit2_SS[1]
		full_saved_file_dict['average_pulse_SS'] = dict([])
		full_saved_file_dict['average_pulse_SS']['area_of_interest'] = area_of_interest_sigma_SS.nominal_value
		full_saved_file_dict['average_pulse_SS']['area_of_interest_sigma'] = area_of_interest_sigma_SS.std_dev
		SS_area_fit_ok = True
	except Exception as e:
		print('FAILED SS area fit')
		logging.exception('with error: ' + str(e))
		full_saved_file_dict['SS_fit_of_plasma_centre_and_area'] = np.ones_like(fit1[0])*np.nan
		full_saved_file_dict['SS_fit_of_plasma_centre_and_area_sigma'] = np.ones_like(fit1[1])*np.nan
		full_saved_file_dict['average_pulse_SS'] = dict([])
		full_saved_file_dict['average_pulse_SS']['area_of_interest'] = np.nan
		full_saved_file_dict['average_pulse_SS']['area_of_interest_sigma'] = np.nan


	ani = coleval.movie_from_data(np.array([twoD_peak_evolution_temp_averaged_delta_full_mean_filter[::10]]), 1/(np.median(np.diff(twoD_peak_evolution_time_averaged)))/10, integration=int_time/1000,xlabel='horizontal coord [pixels]',ylabel='vertical coord [pixels]',barlabel='Temperature increase [°C]',time_offset=np.min(twoD_peak_evolution_time_averaged),extvmin=0,prelude=pre_title+'Temperature averaged among all second half of peaks - before mean filter '+str(full_mean_filter_footprint)+'\n')
	figure_index+=1
	ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.mp4', fps=3, writer='ffmpeg',codec='mpeg4')
	plt.close()

	plt.figure(figsize=(20, 10))
	# plt.imshow(twoD_peak_evolution_temp_averaged_before,'rainbow',vmax=np.median(np.sort(twoD_peak_evolution_temp_averaged_before.flatten())[-10:]))
	# plt.plot(IR_shape[1],IR_shape[0],'k+')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--',label='externally supplied')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.xlabel('horizontal coord [pixels]')
	# plt.ylabel('vertical coord [pixels]')
	plt.pcolor(h_coordinates,v_coordinates,twoD_peak_evolution_temp_averaged_before,cmap='rainbow',vmax=np.median(np.sort(twoD_peak_evolution_temp_averaged_before.flatten())[-10:]))
	plt.plot(IR_shape[1]*p2d_h,IR_shape[0]*p2d_v,'k+')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--',label='externally supplied')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.xlabel('horizontal coord [mm]')
	plt.ylabel('vertical coord [mm]')
	plt.colorbar().set_label('Temperature [°C]')
	# plt.errorbar(fit1[0][3],fit1[0][2],xerr=fit1[1][3,3]**0.5,yerr=fit1[1][2,2]**0.5,color='c')
	# plt.plot(fit1[0][3] + np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)*p2d_v/p2d_h,fit1[0][2] + np.abs(fit1[0][4]**2-np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)**2)**0.5,'c--',label='fitted\nTmin=%.3g°C, dt=%.3g°C' %(fit1[0][0],fit1[0][1]))
	# plt.plot(fit1[0][3] + np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)*p2d_v/p2d_h,fit1[0][2] - np.abs(fit1[0][4]**2-np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)**2)**0.5,'c--')
	plt.plot((fit2_SS[0][3] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit2_SS[0][2] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g--',label='target diameter')
	plt.plot((fit2_SS[0][3] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit2_SS[0][2] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g--')
	plt.errorbar(fit1[0][2]*p2d_h,fit1[0][1]*p2d_v,xerr=(fit1[1][2,2]**0.5)*p2d_h,yerr=(fit1[1][1,1]**0.5)*p2d_v,color='b')
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] + np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--',label='fitted temp increase')
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] - np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--')
	if SS_area_fit_ok:
		plt.errorbar(fit2_SS[0][3]*p2d_h,fit2_SS[0][2]*p2d_v,xerr=max(1,(fit2_SS[1][3,3]**0.5)*p2d_h),yerr=max(1,(fit2_SS[1][2,2]**0.5)*p2d_v),color='C7')
		plt.plot((fit2_SS[0][3] + np.arange(-area_of_interest_radius_SS,+area_of_interest_radius_SS+area_of_interest_radius_SS/10,area_of_interest_radius_SS/10)*p2d_v/p2d_h)*p2d_h,(fit2_SS[0][2] + np.abs(area_of_interest_radius_SS**2-np.arange(-area_of_interest_radius_SS,+area_of_interest_radius_SS+area_of_interest_radius_SS/10,area_of_interest_radius_SS/10)**2)**0.5)*p2d_v,'--C7',label='SS fitted(*1/1.2)\nTmin=%.3g°C, dT=%.3g°C' %(fit2_SS[0][0],fit2_SS[0][1]))
		plt.plot((fit2_SS[0][3] + np.arange(-area_of_interest_radius_SS,+area_of_interest_radius_SS+area_of_interest_radius_SS/10,area_of_interest_radius_SS/10)*p2d_v/p2d_h)*p2d_h,(fit2_SS[0][2] - np.abs(area_of_interest_radius_SS**2-np.arange(-area_of_interest_radius_SS,+area_of_interest_radius_SS+area_of_interest_radius_SS/10,area_of_interest_radius_SS/10)**2)**0.5)*p2d_v,'--C7')
	plt.axes().set_aspect(aspect_ratio)
	plt.legend(loc='best', fontsize='x-small')
	plt.xlim(left=-5,right=35)
	plt.ylim(bottom=-5,top=22)
	plt.title(pre_title+'Temperature distribution before the averaged peaks (average of %.3gms to %.3gms) in ' %(np.min(twoD_peak_evolution_time_averaged),-5/frequency*1000)+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	# plt.imshow(twoD_peak_evolution_temp_averaged_after_short,'rainbow',vmax=np.median(np.sort(twoD_peak_evolution_temp_averaged_after_short.flatten())[-10:]))
	# plt.plot(IR_shape[1],IR_shape[0],'k+')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--',label='externally supplied')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.xlabel('horizontal coord [pixels]')
	# plt.ylabel('vertical coord [pixels]')
	plt.pcolor(h_coordinates,v_coordinates,twoD_peak_evolution_temp_averaged[np.abs(twoD_peak_evolution_time_averaged).argmin()],cmap='rainbow',vmax=np.median(np.sort(twoD_peak_evolution_temp_averaged[np.abs(twoD_peak_evolution_time_averaged).argmin()].flatten())[-10:]))
	plt.plot(IR_shape[1]*p2d_h,IR_shape[0]*p2d_v,'k+')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--',label='externally supplied')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.xlabel('horizontal coord [mm]')
	plt.ylabel('vertical coord [mm]')
	plt.colorbar().set_label('Temperature [°C]')
	# plt.errorbar(fit1[0][3],fit1[0][2],xerr=fit1[1][3,3]**0.5,yerr=fit1[1][2,2]**0.5,color='c')
	# plt.plot(fit1[0][3] + np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)*p2d_v/p2d_h,fit1[0][2] + np.abs(fit1[0][4]**2-np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)**2)**0.5,'c--',label='fitted\nTmin=%.3g°C, dt=%.3g°C' %(fit1[0][0],fit1[0][1]))
	# plt.plot(fit1[0][3] + np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)*p2d_v/p2d_h,fit1[0][2] - np.abs(fit1[0][4]**2-np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)**2)**0.5,'c--')
	plt.plot((fit1[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1[0][1] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g--',label='target diameter')
	plt.plot((fit1[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1[0][1] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g--')
	plt.errorbar(fit1[0][2]*p2d_h,fit1[0][1]*p2d_v,xerr=(fit1[1][2,2]**0.5)*p2d_h,yerr=(fit1[1][1,1]**0.5)*p2d_v,color='b')
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] + np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--',label='fitted temp increase')
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] - np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--')
	# plt.plot(fit2[0][1] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][2] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--',label='target diameter')
	# plt.plot(fit2[0][1] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][2] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--')
	plt.axes().set_aspect(aspect_ratio)
	plt.legend(loc='best', fontsize='x-small')
	plt.xlim(left=-5,right=35)
	plt.ylim(bottom=-5,top=22)
	plt.title(pre_title+'Temperature distribution at the averaged peaks in '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	def gaussian_2D_fitting(spatial_coord,*args):
		x = spatial_coord[0]	# horizontal
		y = spatial_coord[1]	# vertical
		out = args[1]*np.exp(- (((x-args[3])*p2d_v/p2d_h)**2 + (y-args[2])**2)/(2*(args[4]**2)) ) + args[0]
		select = ((x-args[3])*p2d_h)**2+((y-args[2])*p2d_v)**2>(max_diameter_to_fit/2/2)**2
		out[select]=twoD_peak_evolution_temp_averaged_after_short[select]
		return (out-twoD_peak_evolution_temp_averaged_after_short).flatten()

	guess = [20,twoD_peak_evolution_temp_averaged_after_short.max()-max(0,twoD_peak_evolution_temp_averaged_after_short.min()),IR_shape[0],IR_shape[1],IR_shape[3]]
	bds=[[20,0,max(0,IR_shape[0]-IR_shape[3]),max(0,IR_shape[1]-IR_shape[3]),min(IR_shape[2],10)],[np.inf,np.inf,min(np.shape(twoD_peak_evolution_temp_averaged_after_short)[0],IR_shape[0]+IR_shape[3]),min(np.shape(twoD_peak_evolution_temp_averaged_after_short)[1],IR_shape[1]+IR_shape[3]),np.max(twoD_peak_evolution_temp_averaged_after_short.shape)/2]]
	spatial_coord=np.meshgrid(np.arange(np.shape(twoD_peak_evolution_temp_averaged_after_short)[1]),np.arange(np.shape(twoD_peak_evolution_temp_averaged_after_short)[0]))
	fit1_absolute = curve_fit(gaussian_2D_fitting, spatial_coord, np.zeros_like(twoD_peak_evolution_temp_averaged_after_short.flatten()), guess,absolute_sigma=True,bounds=bds,maxfev=int(1e4))
	x = spatial_coord[0]	# horizontal
	y = spatial_coord[1]	# vertical

	if np.sum(np.isnan(fit1_absolute[1]))>0:
		fit1_absolute[0][1] = fit2[0][0]
		fit1_absolute[0][2] = fit2[0][1]
		fit1_absolute[0][3] = fit2[0][2]

	def gaussian_2D_fitting(spatial_coord,*args):
		x = spatial_coord[0]	# horizontal
		y = spatial_coord[1]	# vertical
		out = args[1]*np.exp(- (((x-args[3])*p2d_v/p2d_h)**2 + (y-args[2])**2)/(2*(args[4]**2)) ) + args[0]
		select = ((x-args[3])*p2d_h)**2+((y-args[2])*p2d_v)**2>(max_diameter_to_fit/2*0.7)**2
		out[select]=twoD_peak_evolution_temp_averaged_after_long[select]
		return (out-twoD_peak_evolution_temp_averaged_after_long).flatten()

	temp1 = ((x-fit1_absolute[0][2])*p2d_h)**2+((y-fit1_absolute[0][3])*p2d_v)**2<(max_diameter_to_fit/2)**2
	temp = max(1,np.median(np.sort(twoD_peak_evolution_temp_averaged_after_long[((x-fit1_absolute[0][2])*p2d_h)**2+((y-fit1_absolute[0][3])*p2d_v)**2<(max_diameter_to_fit/2)**2])[:int(np.sum(temp1)/4)]))
	guess = [fit1_absolute[0][0],max(0,twoD_peak_evolution_temp_averaged_after_long.max()-temp),fit1_absolute[0][2],fit1_absolute[0][3],fit1_absolute[0][4]]
	bds=[[20,0,fit1_absolute[0][2]-0.1,fit1_absolute[0][3]-0.1,min(IR_shape[2],10)],[max(temp,fit1_absolute[0][0]),np.inf,fit1_absolute[0][2]+0.1,fit1_absolute[0][3]+0.1,np.max(twoD_peak_evolution_temp_averaged_after_long.shape)/2]]
	spatial_coord=np.meshgrid(np.arange(np.shape(twoD_peak_evolution_temp_averaged_after_long)[1]),np.arange(np.shape(twoD_peak_evolution_temp_averaged_after_long)[0]))
	print(guess);print(bds)
	fit2_absolute = curve_fit(gaussian_2D_fitting, spatial_coord, np.zeros_like(twoD_peak_evolution_temp_averaged_after_long.flatten()), guess,absolute_sigma=True,bounds=bds,maxfev=int(1e4))
	area_of_interest_sigma_absolute = 1/1.14*np.pi*( 2**0.5 *(correlated_values(fit2_absolute[0],fit2_absolute[1])[4])*p2d_v/1000)**2	# Area equivalent to all the overtemperature profile collapsed in a homogeneous circular spot
	area_of_interest_sigma_absolute = area_of_interest_sigma_absolute + ufloat(0,0.1*area_of_interest_sigma_absolute.nominal_value)	# this to add the uncertainly coming from the method itself
	area_of_interest_radius_absolute = 2**0.5 *fit2_absolute[0][4]	# pixels

	if not SS_area_fit_ok:
		fit2_SS = cp.deepcopy(fit2_absolute)


	plt.figure(figsize=(20, 10))
	# plt.imshow(twoD_peak_evolution_temp_averaged_after_long,'rainbow',vmax=np.median(np.sort(twoD_peak_evolution_temp_averaged_after_long.flatten())[-10:]))
	# plt.plot(IR_shape[1],IR_shape[0],'k+')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--',label='externally supplied')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.xlabel('horizontal coord [pixels]')
	# plt.ylabel('vertical coord [pixels]')
	plt.pcolor(h_coordinates,v_coordinates,twoD_peak_evolution_temp_averaged_after_long,cmap='rainbow',vmax=np.median(np.sort(twoD_peak_evolution_temp_averaged_after_long.flatten())[-10:]))
	plt.plot(IR_shape[1]*p2d_h,IR_shape[0]*p2d_v,'k+')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--',label='externally supplied')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.xlabel('horizontal coord [mm]')
	plt.ylabel('vertical coord [mm]')
	plt.colorbar().set_label('Temperature [°C]')
	# plt.errorbar(fit1[0][3],fit1[0][2],xerr=fit1[1][3,3]**0.5,yerr=fit1[1][2,2]**0.5,color='c')
	# plt.plot(fit1[0][3] + np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)*p2d_v/p2d_h,fit1[0][2] + np.abs(fit1[0][4]**2-np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)**2)**0.5,'c--',label='fitted\nTmin=%.3g°C, dt=%.3g°C' %(fit1[0][0],fit1[0][1]))
	# plt.plot(fit1[0][3] + np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)*p2d_v/p2d_h,fit1[0][2] - np.abs(fit1[0][4]**2-np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)**2)**0.5,'c--')
	plt.plot((fit1_absolute[0][3] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1_absolute[0][2] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g--',label='target diameter')
	plt.plot((fit1_absolute[0][3] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1_absolute[0][2] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g--')
	plt.errorbar(fit1[0][2]*p2d_h,fit1[0][1]*p2d_v,xerr=(fit1[1][2,2]**0.5)*p2d_h,yerr=(fit1[1][1,1]**0.5)*p2d_v,color='b')
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] + np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--',label='fitted temp increase')
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] - np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--')
	plt.plot((fit1_absolute[0][3] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1_absolute[0][2] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g:',label='target diameter from absolute temp(*1/1.14)\nTmin=%.3g°C, dT=%.3g°C' %(fit2_absolute[0][0],fit2_absolute[0][1]))
	plt.plot((fit1_absolute[0][3] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1_absolute[0][2] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g:')
	plt.errorbar(fit1_absolute[0][3]*p2d_h,fit1_absolute[0][2]*p2d_v,xerr=(fit1_absolute[1][3,3]**0.5)*p2d_h,yerr=(fit1_absolute[1][2,2]**0.5)*p2d_v,color='b')
	plt.plot((fit2_absolute[0][3] + np.arange(-area_of_interest_radius_absolute,+area_of_interest_radius_absolute+area_of_interest_radius_absolute/10,area_of_interest_radius_absolute/10)*p2d_v/p2d_h)*p2d_h,(fit2_absolute[0][2] + np.abs(area_of_interest_radius_absolute**2-np.arange(-area_of_interest_radius_absolute,+area_of_interest_radius_absolute+area_of_interest_radius_absolute/10,area_of_interest_radius_absolute/10)**2)**0.5)*p2d_v,'b:',label='fitted absolute temp(*1/1.14)')
	plt.plot((fit2_absolute[0][3] + np.arange(-area_of_interest_radius_absolute,+area_of_interest_radius_absolute+area_of_interest_radius_absolute/10,area_of_interest_radius_absolute/10)*p2d_v/p2d_h)*p2d_h,(fit2_absolute[0][2] - np.abs(area_of_interest_radius_absolute**2-np.arange(-area_of_interest_radius_absolute,+area_of_interest_radius_absolute+area_of_interest_radius_absolute/10,area_of_interest_radius_absolute/10)**2)**0.5)*p2d_v,'b:')
	# plt.plot(fit2[0][1] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][2] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--',label='target diameter')
	# plt.plot(fit2[0][1] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][2] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--')
	plt.axes().set_aspect(aspect_ratio)
	plt.legend(loc='best', fontsize='x-small')
	plt.xlim(left=-5,right=35)
	plt.ylim(bottom=-5,top=22)
	plt.title(pre_title+'Temperature distribution, average %.2g to %.2gms after the averaged peaks in ' %(after_long_start_time,after_long_end_time)+'IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	# plt.imshow(twoD_peak_evolution_temp_averaged_after_long,'rainbow',vmax=np.median(np.sort(twoD_peak_evolution_temp_averaged_after_long.flatten())[-10:]))
	# plt.plot(IR_shape[1],IR_shape[0],'k+')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--',label='externally supplied')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.xlabel('horizontal coord [pixels]')
	# plt.ylabel('vertical coord [pixels]')
	plt.pcolor(h_coordinates,v_coordinates,twoD_peak_evolution_temp_averaged_delta[np.abs(twoD_peak_evolution_time_averaged).argmin()],cmap='rainbow',vmax=np.median(np.sort(twoD_peak_evolution_temp_averaged_delta[np.abs(twoD_peak_evolution_time_averaged).argmin()].flatten())[-10:]))
	plt.plot(IR_shape[1]*p2d_h,IR_shape[0]*p2d_v,'k+')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--',label='externally supplied')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.xlabel('horizontal coord [mm]')
	plt.ylabel('vertical coord [mm]')
	plt.colorbar().set_label('Temperature [°C]')
	# plt.errorbar(fit1[0][3],fit1[0][2],xerr=fit1[1][3,3]**0.5,yerr=fit1[1][2,2]**0.5,color='c')
	# plt.plot(fit1[0][3] + np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)*p2d_v/p2d_h,fit1[0][2] + np.abs(fit1[0][4]**2-np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)**2)**0.5,'c--',label='fitted\nTmin=%.3g°C, dt=%.3g°C' %(fit1[0][0],fit1[0][1]))
	# plt.plot(fit1[0][3] + np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)*p2d_v/p2d_h,fit1[0][2] - np.abs(fit1[0][4]**2-np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)**2)**0.5,'c--')
	plt.plot((fit1[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1[0][1] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g--',label='target diameter')
	plt.plot((fit1[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1[0][1] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g--')
	plt.errorbar(fit1[0][2]*p2d_h,fit1[0][1]*p2d_v,xerr=(fit1[1][2,2]**0.5)*p2d_h,yerr=(fit1[1][1,1]**0.5)*p2d_v,color='b')
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] + np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--',label='fitted')
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] - np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--')
	# plt.plot(fit2[0][1] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][2] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--',label='target diameter')
	# plt.plot(fit2[0][1] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][2] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--')
	plt.axes().set_aspect(aspect_ratio)
	plt.legend(loc='best', fontsize='x-small')
	plt.xlim(left=-5,right=35)
	plt.ylim(bottom=-5,top=22)
	plt.title(pre_title+'Temperature increase at the averaged peaks in '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.pcolor(h_coordinates,v_coordinates,twoD_peak_evolution_temp_averaged_delta[np.abs(twoD_peak_evolution_time_averaged).argmin()],cmap='rainbow',vmax=np.median(np.sort(twoD_peak_evolution_temp_averaged_delta[np.abs(twoD_peak_evolution_time_averaged).argmin()].flatten())[-10:]))
	plt.xlabel('horizontal coord [mm]')
	plt.ylabel('vertical coord [mm]')
	plt.colorbar().set_label('Temperature [°C]')
	plt.errorbar(fit1[0][2]*p2d_h,fit1[0][1]*p2d_v,xerr=(fit1[1][2,2]**0.5)*p2d_h,yerr=(fit1[1][1,1]**0.5)*p2d_v,color='b')
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] + np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--')
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] - np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--')
	plt.axes().set_aspect(aspect_ratio)
	plt.legend(loc='best', fontsize='x-small')
	plt.xlim(left=-5,right=35)
	plt.ylim(bottom=-5,top=22)
	plt.title(pre_title+'Temperature increase at the averaged peaks in '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'b.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	# plt.imshow(twoD_peak_evolution_temp_averaged_delta_after_short,'rainbow',vmin=0,vmax=np.mean(np.sort(twoD_peak_evolution_temp_averaged_delta_after_short[((x-fit2[0][2])*p2d_h)**2+((y-fit2[0][1])*p2d_v)**2<(IR_shape[3]*p2d_v)**2])[-20:]))
	# plt.plot(IR_shape[1],IR_shape[0],'k+')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--',label='externally supplied')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.xlabel('horizontal coord [pixels]')
	# plt.ylabel('vertical coord [pixels]')
	plt.pcolor(h_coordinates,v_coordinates,twoD_peak_evolution_temp_averaged_delta_after_short,cmap='rainbow',vmin=0,vmax=np.mean(np.sort(twoD_peak_evolution_temp_averaged_delta_after_short[((x-fit2[0][2])*p2d_h)**2+((y-fit2[0][1])*p2d_v)**2<(IR_shape[3]*p2d_v/2)**2])[-20:]))
	plt.plot(IR_shape[1]*p2d_h,IR_shape[0]*p2d_v,'k+')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--',label='externally supplied')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.xlabel('horizontal coord [mm]')
	plt.ylabel('vertical coord [mm]')
	plt.colorbar().set_label('Temperature [°C] limited to 0')
	plt.errorbar(fit1[0][2]*p2d_h,fit1[0][1]*p2d_v,xerr=(fit1[1][2,2]**0.5)*p2d_h,yerr=(fit1[1][1,1]**0.5)*p2d_v,color='c',label='fitted from temp increase\ndt=%.3g°C' %(fit1[0][0]))
	# plt.plot(fit1[0][3] + np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)*p2d_v/p2d_h,fit1[0][2] + np.abs(fit1[0][4]**2-np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)**2)**0.5,'c--',label='fitted\nTmin=%.3g°C, dt=%.3g°C' %(fit1[0][0],fit1[0][1]))
	# plt.plot(fit1[0][3] + np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)*p2d_v/p2d_h,fit1[0][2] - np.abs(fit1[0][4]**2-np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)**2)**0.5,'c--')
	plt.plot((fit1[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1[0][1] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g--',label='target diameter from temp increase')
	plt.plot((fit1[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1[0][1] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g--')
	plt.errorbar(fit1[0][2]*p2d_h,fit1[0][1]*p2d_v,xerr=(fit1[1][2,2]**0.5)*p2d_h,yerr=(fit1[1][1,1]**0.5)*p2d_v,color='b')
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] + np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--',label='fitted from temp increase, dT=%.3g°C\narea=%.3g+/-%.3gm2' %(fit2[0][0],area_of_interest_sigma.nominal_value,area_of_interest_sigma.std_dev))
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] - np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--')
	plt.plot((fit1_absolute[0][3] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1_absolute[0][2] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g:',label='target diameter from absolute temp(*1/1.14)')
	plt.plot((fit1_absolute[0][3] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1_absolute[0][2] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g:')
	plt.errorbar(fit1_absolute[0][3]*p2d_h,fit1_absolute[0][2]*p2d_v,xerr=(fit1_absolute[1][3,3]**0.5)*p2d_h,yerr=(fit1_absolute[1][2,2]**0.5)*p2d_v,color='b')
	plt.plot((fit2_absolute[0][3] + np.arange(-area_of_interest_radius_absolute,+area_of_interest_radius_absolute+area_of_interest_radius_absolute/10,area_of_interest_radius_absolute/10)*p2d_v/p2d_h)*p2d_h,(fit2_absolute[0][2] + np.abs(area_of_interest_radius_absolute**2-np.arange(-area_of_interest_radius_absolute,+area_of_interest_radius_absolute+area_of_interest_radius_absolute/10,area_of_interest_radius_absolute/10)**2)**0.5)*p2d_v,'b:',label='fitted absolute temp(*1/1.14)')
	plt.plot((fit2_absolute[0][3] + np.arange(-area_of_interest_radius_absolute,+area_of_interest_radius_absolute+area_of_interest_radius_absolute/10,area_of_interest_radius_absolute/10)*p2d_v/p2d_h)*p2d_h,(fit2_absolute[0][2] - np.abs(area_of_interest_radius_absolute**2-np.arange(-area_of_interest_radius_absolute,+area_of_interest_radius_absolute+area_of_interest_radius_absolute/10,area_of_interest_radius_absolute/10)**2)**0.5)*p2d_v,'b:')
	# plt.plot(fit2[0][1] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][2] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--',label='target diameter')
	# plt.plot(fit2[0][1] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][2] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--')
	plt.axes().set_aspect(aspect_ratio)
	plt.legend(loc='best', fontsize='x-small')
	plt.xlim(left=-5,right=35)
	plt.ylim(bottom=-5,top=22)
	plt.title(pre_title+'Temperature increase, average %.2g to %.2gms after the averaged peaks in ' %(after_short_start_time,after_short_end_time)+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.pcolor(h_coordinates,v_coordinates,twoD_peak_evolution_temp_averaged_delta_after_short,cmap='rainbow',vmin=0,vmax=np.mean(np.sort(twoD_peak_evolution_temp_averaged_delta_after_short[((x-fit2[0][2])*p2d_h)**2+((y-fit2[0][1])*p2d_v)**2<(IR_shape[3]*p2d_v/2)**2])[-20:]))
	plt.xlabel('horizontal coord [mm]')
	plt.ylabel('vertical coord [mm]')
	plt.colorbar().set_label('Temperature [°C] limited to 0')
	plt.errorbar(fit1[0][2]*p2d_h,fit1[0][1]*p2d_v,xerr=(fit1[1][2,2]**0.5)*p2d_h,yerr=(fit1[1][1,1]**0.5)*p2d_v,color='b')
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] + np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--')
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] - np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--')
	plt.axes().set_aspect(aspect_ratio)
	plt.legend(loc='best', fontsize='x-small')
	plt.xlim(left=-5,right=35)
	plt.ylim(bottom=-5,top=22)
	plt.title(pre_title+'Temperature increase, average %.2g to %.2gms after the averaged peaks in ' %(after_short_start_time,after_short_end_time)+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'b.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	# plt.imshow(twoD_peak_evolution_temp_averaged_delta_after_short,'rainbow',vmin=0,vmax=np.mean(np.sort(twoD_peak_evolution_temp_averaged_delta_after_short[((x-fit2[0][2])*p2d_h)**2+((y-fit2[0][1])*p2d_v)**2<(IR_shape[3]*p2d_v)**2])[-20:]))
	# plt.plot(IR_shape[1],IR_shape[0],'k+')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--',label='externally supplied')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.xlabel('horizontal coord [pixels]')
	# plt.ylabel('vertical coord [pixels]')
	plt.pcolor(h_coordinates,v_coordinates,twoD_peak_evolution_temp_averaged_delta_after_long,cmap='rainbow',vmin=0,vmax=np.mean(np.sort(twoD_peak_evolution_temp_averaged_delta_after_long[((x-fit2[0][2])*p2d_h)**2+((y-fit2[0][1])*p2d_v)**2<(IR_shape[3]*p2d_v/2)**2])[-20:]))
	plt.plot(IR_shape[1]*p2d_h,IR_shape[0]*p2d_v,'k+')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--',label='externally supplied')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.plot((IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10))*p2d_h,(IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5)*p2d_v,'k--')
	plt.xlabel('horizontal coord [mm]')
	plt.ylabel('vertical coord [mm]')
	plt.colorbar().set_label('Temperature [°C] limited to 0')
	plt.errorbar(fit1[0][2]*p2d_h,fit1[0][1]*p2d_v,xerr=(fit1[1][2,2]**0.5)*p2d_h,yerr=(fit1[1][1,1]**0.5)*p2d_v,color='c',label='fitted from temp increase\ndt=%.3g°C' %(fit1[0][0]))
	# plt.plot(fit1[0][3] + np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)*p2d_v/p2d_h,fit1[0][2] + np.abs(fit1[0][4]**2-np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)**2)**0.5,'c--',label='fitted\nTmin=%.3g°C, dt=%.3g°C' %(fit1[0][0],fit1[0][1]))
	# plt.plot(fit1[0][3] + np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)*p2d_v/p2d_h,fit1[0][2] - np.abs(fit1[0][4]**2-np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)**2)**0.5,'c--')
	plt.plot((fit1[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1[0][1] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g--',label='target diameter from temp increase')
	plt.plot((fit1[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1[0][1] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g--')
	plt.errorbar(fit1[0][2]*p2d_h,fit1[0][1]*p2d_v,xerr=(fit1[1][2,2]**0.5)*p2d_h,yerr=(fit1[1][1,1]**0.5)*p2d_v,color='b')
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] + np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--',label='fitted from temp increase, dT=%.3g°C\narea=%.3g+/-%.3gm2' %(fit2[0][0],area_of_interest_sigma.nominal_value,area_of_interest_sigma.std_dev))
	plt.plot((fit2[0][2] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h)*p2d_h,(fit2[0][1] - np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5)*p2d_v,'b--')
	plt.plot((fit1_absolute[0][3] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1_absolute[0][2] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g:',label='target diameter from absolute temp')
	plt.plot((fit1_absolute[0][3] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2)*p2d_h,(fit1_absolute[0][2] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v)*p2d_v,'g:')
	plt.errorbar(fit1_absolute[0][3]*p2d_h,fit1_absolute[0][2]*p2d_v,xerr=(fit1_absolute[1][3,3]**0.5)*p2d_h,yerr=(fit1_absolute[1][2,2]**0.5)*p2d_v,color='b')
	plt.plot((fit2_absolute[0][3] + np.arange(-area_of_interest_radius_absolute,+area_of_interest_radius_absolute+area_of_interest_radius_absolute/10,area_of_interest_radius_absolute/10)*p2d_v/p2d_h)*p2d_h,(fit2_absolute[0][2] + np.abs(area_of_interest_radius_absolute**2-np.arange(-area_of_interest_radius_absolute,+area_of_interest_radius_absolute+area_of_interest_radius_absolute/10,area_of_interest_radius_absolute/10)**2)**0.5)*p2d_v,'b:',label='fitted absolute temp(*1/1.14)')
	plt.plot((fit2_absolute[0][3] + np.arange(-area_of_interest_radius_absolute,+area_of_interest_radius_absolute+area_of_interest_radius_absolute/10,area_of_interest_radius_absolute/10)*p2d_v/p2d_h)*p2d_h,(fit2_absolute[0][2] - np.abs(area_of_interest_radius_absolute**2-np.arange(-area_of_interest_radius_absolute,+area_of_interest_radius_absolute+area_of_interest_radius_absolute/10,area_of_interest_radius_absolute/10)**2)**0.5)*p2d_v,'b:')
	# plt.plot(fit2[0][1] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][2] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--',label='target diameter')
	# plt.plot(fit2[0][1] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][2] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--')
	plt.axes().set_aspect(aspect_ratio)
	plt.legend(loc='best', fontsize='x-small')
	plt.xlim(left=-5,right=35)
	plt.ylim(bottom=-5,top=22)
	plt.title(pre_title+'Temperature increase, average %.2g to %.2gms after the averaged peaks in ' %(after_long_start_time,after_long_end_time)+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	collect_shape_information = []
	plt.figure(figsize=(20, 10))
	selected_area = (twoD_peak_evolution_temp_averaged_delta.shape[1]*twoD_peak_evolution_temp_averaged_delta.shape[2])
	temp = np.sum(twoD_peak_evolution_temp_averaged_delta.T>twoD_peak_evolution_temp_averaged_delta.max(axis=(1,2))*0.2,axis=(0,1))
	collect_shape_information.append(temp/selected_area)
	plt.plot([0]*2,[0,temp.max()/selected_area],'--k')
	plt.plot(twoD_peak_evolution_time_averaged,temp/selected_area,':r',label='dt, max x0.2')
	temp = np.sum(twoD_peak_evolution_temp_averaged_delta.T>twoD_peak_evolution_temp_averaged_delta.max(axis=(1,2))*0.5,axis=(0,1))
	collect_shape_information.append(temp/selected_area)
	plt.plot(twoD_peak_evolution_time_averaged,temp/selected_area,'-r',label='dt, max x0.5')
	temp = np.sum(twoD_peak_evolution_temp_averaged_delta.T>twoD_peak_evolution_temp_averaged_delta.max(axis=(1,2))*0.8,axis=(0,1))
	collect_shape_information.append(temp/selected_area)
	plt.plot(twoD_peak_evolution_time_averaged,temp/selected_area,'--r',label='dt, max x0.8')
	# from IR_records_process3_temp_dep_properties.py I see that I have to use target temperature - ambient temperature
	temp = np.sum((twoD_peak_evolution_temp_averaged-20).T>(twoD_peak_evolution_temp_averaged-20).max(axis=(1,2))*0.2,axis=(0,1))
	collect_shape_information.append(temp/selected_area)
	plt.plot([0]*2,[0,temp.max()/selected_area],'--k')
	plt.plot(twoD_peak_evolution_time_averaged,temp/selected_area,':b',label='full T, max x0.2')
	temp = np.sum((twoD_peak_evolution_temp_averaged-20).T>(twoD_peak_evolution_temp_averaged-20).max(axis=(1,2))*0.5,axis=(0,1))
	collect_shape_information.append(temp/selected_area)
	plt.plot(twoD_peak_evolution_time_averaged,temp/selected_area,'-b',label='full T, max x0.5')
	temp = np.sum((twoD_peak_evolution_temp_averaged-20).T>(twoD_peak_evolution_temp_averaged-20).max(axis=(1,2))*0.8,axis=(0,1))
	collect_shape_information.append(temp/selected_area)
	plt.plot(twoD_peak_evolution_time_averaged,temp/selected_area,'--b',label='full T, max x0.8')
	plt.legend(loc='best', fontsize='x-small')
	plt.xlabel('time after the peak [s]')
	plt.ylabel('fraction of pixels [au]')
	plt.title(pre_title+'Temperature increase, fraction high temperature area (full sensor) '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000) +'\nArea with '+r'$(T-T_{amb})>(T-T_{amb})_{max}* x $'+' (full) and '+r'$T-T_{ss}>(T-T_{ss})_{max}* x $'+' (dt)')
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	h_coordinates2,v_coordinates2 = np.meshgrid(np.arange(header.h_PixelsPerLine)*p2d_h,np.arange(header.h_LinesPerField)*p2d_v)
	selected_area = (h_coordinates2-fit2[0][2]*p2d_h)**2 + (v_coordinates2-fit2[0][1]*p2d_v)**2 < (max_diameter_to_fit/2)**2
	plt.figure(figsize=(20, 10))
	temp_selected = twoD_peak_evolution_temp_averaged_delta[:,selected_area]
	temp = np.sum(temp_selected.T>temp_selected.max(axis=(1))*0.2,axis=(0))
	collect_shape_information.append(temp/np.sum(selected_area))
	plt.plot([0]*2,[0,temp.max()/np.sum(selected_area)],'--k')
	plt.plot(twoD_peak_evolution_time_averaged,temp/np.sum(selected_area),':r',label='dt, max x0.2')
	temp = np.sum(temp_selected.T>temp_selected.max(axis=(1))*0.5,axis=(0))
	collect_shape_information.append(temp/np.sum(selected_area))
	plt.plot(twoD_peak_evolution_time_averaged,temp/np.sum(selected_area),'-r',label='dt, max x0.5')
	temp = np.sum(temp_selected.T>temp_selected.max(axis=(1))*0.8,axis=(0))
	collect_shape_information.append(temp/np.sum(selected_area))
	plt.plot(twoD_peak_evolution_time_averaged,temp/np.sum(selected_area),'--r',label='dt, max x0.8')
	temp_selected = twoD_peak_evolution_temp_averaged[:,selected_area]
	# from IR_records_process3_temp_dep_properties.py I see that I have to use target temperature - ambient temperature
	temp_selected -= 20
	temp = np.sum(temp_selected.T>temp_selected.max(axis=(1))*0.2,axis=(0))
	collect_shape_information.append(temp/np.sum(selected_area))
	plt.plot([0]*2,[0,temp.max()/np.sum(selected_area)],'--k')
	plt.plot(twoD_peak_evolution_time_averaged,temp/np.sum(selected_area),':b',label='full T, max x0.2')
	temp = np.sum(temp_selected.T>temp_selected.max(axis=(1))*0.5,axis=(0))
	collect_shape_information.append(temp/np.sum(selected_area))
	plt.plot(twoD_peak_evolution_time_averaged,temp/np.sum(selected_area),'-b',label='full T, max x0.5')
	temp = np.sum(temp_selected.T>temp_selected.max(axis=(1))*0.8,axis=(0))
	collect_shape_information.append(temp/np.sum(selected_area))
	plt.plot(twoD_peak_evolution_time_averaged,temp/np.sum(selected_area),'--b',label='full T, max x0.8')
	plt.legend(loc='best', fontsize='x-small')
	plt.xlabel('time after the peak [s]')
	plt.ylabel('fraction of pixels [au]')
	plt.title(pre_title+'Temperature increase, fraction high temperature area (only on target) '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000) +'\nArea with '+r'$(T-T_{amb})>(T-T_{amb})_{max} * 0.8$'+' (-) and '+r'$T-T_{ss}>(T-T_{ss})_{max} * 0.5$'+' (--)')
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()


	# full_saved_file_dict = dict(np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz'))
	full_saved_file_dict['target_diameter'] = max_diameter_to_fit
	full_saved_file_dict['T_pre_pulse'] = mean_temp_before_peaks_large_area_max
	full_saved_file_dict['collect_shape_information'] = collect_shape_information
	# np.savez_compressed(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace,**full_saved_file_dict)

	# New addition. I need to calculate and include also the energy of the steady state
	radious = ((h_coordinates2-fit2_SS[0][3]*p2d_h)**2 + (v_coordinates2-fit2_SS[0][2]*p2d_v)**2)**0.5
	SS_temperature_profile = np.array([np.mean(twoD_peak_evolution_temp_averaged_before[np.logical_and(radious>=i*max_diameter_to_fit/2/20,radious<(i+1)*max_diameter_to_fit/2/20)]) for i in np.arange(20)])
	SS_temperature_profile_std = np.array([np.std(twoD_peak_evolution_temp_averaged_before[np.logical_and(radious>=i*max_diameter_to_fit/2/20,radious<(i+1)*max_diameter_to_fit/2/20)]) for i in np.arange(20)])
	SS_radious_profile = np.array([np.mean(radious[np.logical_and(radious>=i*max_diameter_to_fit/2/20,radious<(i+1)*max_diameter_to_fit/2/20)]) for i in np.arange(20)])
	temp = np.zeros_like(SS_radious_profile)
	temp[:-1]+=np.diff(SS_radious_profile)/2
	temp[1:]+=np.diff(SS_radious_profile)/2
	temp[0]+=SS_radious_profile[0]
	temp[-1]+=max_diameter_to_fit/2-SS_radious_profile[-1]
	temp1=SS_temperature_profile-25
	temp1[temp1<0]=0
	SS_Power = np.nansum(np.mean([thermal_conductivity(SS_temperature_profile),thermal_conductivity(np.ones_like(SS_temperature_profile)*25)],axis=0) / target_thickness * temp1 * 2*np.pi*SS_radious_profile * temp * 1e-6)
	SS_Energy = SS_Power * median_duration_good_pulses
	plt.figure(figsize=(20, 10))
	plt.errorbar(SS_radious_profile,SS_temperature_profile,yerr=SS_temperature_profile_std)
	plt.plot(SS_radious_profile,np.ones_like(SS_radious_profile)*25,'--')
	plt.xlabel('radius [mm]')
	plt.ylabel('temperature [°C]')
	plt.legend(loc='best', fontsize='x-small')
	plt.grid()
	plt.title(pre_title+'Steady state temperature profile in ' +str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000) + '\nPower=%.3gW, Energy=%.3gJ' %(SS_Power,SS_Energy))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	full_saved_file_dict['average_pulse_fitted_data']['SS_Energy'] = SS_Energy
	full_saved_file_dict['average_pulse_fitted_data']['SS_Power'] = SS_Power
	full_saved_file_dict['average_pulse_fitted_data']['median_duration_good_pulses'] = median_duration_good_pulses
	full_saved_file_dict['average_pulse_fitted_data']['area_of_interest_absolute'] = area_of_interest_sigma_absolute.nominal_value
	full_saved_file_dict['average_pulse_fitted_data']['area_of_interest_absolute_sigma'] = area_of_interest_sigma_absolute.std_dev
	full_saved_file_dict['average_pulse_fitted_data']['area_of_interest'] = area_of_interest_sigma.nominal_value
	full_saved_file_dict['average_pulse_fitted_data']['area_of_interest_sigma'] = area_of_interest_sigma.std_dev


	fig, ax1 = plt.subplots(figsize=(12, 5))
	fig.subplots_adjust(right=0.8)
	ax1.set_title(pre_title+'fit of 2D averaged peak profile in ' +str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000) + '\nPower=%.3gW, Energy=%.3gJ' %(SS_Power,SS_Energy))
	ax1.set_xlabel('time from peak [s]')
	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	# ax4 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	ax3.spines["right"].set_position(("axes", 1.1))
	ax3.spines["right"].set_visible(True)
	# ax4.spines["right"].set_position(("axes", 1.2))
	# ax4.spines["right"].set_visible(True)
	ax1.fill_between(twoD_peak_evolution_time_averaged, ensamble_peak_shape_counts_fitted[:,0] - ensamble_peak_shape_counts_fitted_std[:,0], ensamble_peak_shape_counts_fitted[:,0] + ensamble_peak_shape_counts_fitted_std[:,0],color='b', alpha=0.4)
	a1, = ax1.plot(twoD_peak_evolution_time_averaged,ensamble_peak_shape_counts_fitted[:,0],'b')
	ax2.fill_between(twoD_peak_evolution_time_averaged, ensamble_peak_shape_temp_fitted[:,0] - ensamble_peak_shape_temp_fitted_std[:,0], ensamble_peak_shape_temp_fitted[:,0] + ensamble_peak_shape_temp_fitted_std[:,0],color='r', alpha=0.4)
	a2, = ax2.plot(twoD_peak_evolution_time_averaged,ensamble_peak_shape_temp_fitted[:,0],'r')
	ax3.fill_between(twoD_peak_evolution_time_averaged, ensamble_peak_shape_counts_fitted[:,1] - ensamble_peak_shape_counts_fitted_std[:,1], ensamble_peak_shape_counts_fitted[:,1] + ensamble_peak_shape_counts_fitted_std[:,1],color='g', alpha=0.3)
	ax3.plot(twoD_peak_evolution_time_averaged,ensamble_peak_shape_counts_fitted[:,1],'--g',label='counts')
	ax3.fill_between(twoD_peak_evolution_time_averaged, ensamble_peak_shape_temp_fitted[:,1] - ensamble_peak_shape_temp_fitted_std[:,1], ensamble_peak_shape_temp_fitted[:,1] + ensamble_peak_shape_temp_fitted_std[:,1],color='g', alpha=0.4)
	a3, = ax3.plot(twoD_peak_evolution_time_averaged,ensamble_peak_shape_temp_fitted[:,1],'g',label='temperature')
	ax3.plot(twoD_peak_evolution_time_averaged[selected_area_average],ensamble_peak_shape_temp_fitted[:,1][selected_area_average],'xg')
	ax3.axhline(y=fit_of_plasma_centre_and_area[3],linestyle=':',color='g',linewidth=1,label='1 fit')
	ax3.axhline(y=fit2[0][3],linestyle=':',color='g',linewidth=2,label='average fit')
	ax3.axvline(x=1.5*1e-3,linestyle='--',color='k',linewidth=1)
	ax3.axvline(x=30*1e-3,linestyle='--',color='k',linewidth=1)
	# a4, = ax4.plot(twoD_peak_evolution_time_averaged,ensamble_peak_shape_temp_fitted[:,2],'y')
	ax1.set_ylabel('fitted peak '+r'$\Delta counts$'+' [au]', color=a1.get_color())
	ax2.set_ylabel('fitted peak '+r'$\Delta temperature$'+' [°C]', color=a2.get_color())  # we already handled the x-label with ax1
	ax3.set_ylabel('fitted gaussian width [pixels]', color=a3.get_color())  # we already handled the x-label with ax1
	# ax4.set_ylabel('fitted temperature T0'+' [°C]', color=a4.get_color())  # we already handled the x-label with ax1
	ax1.tick_params(axis='y', labelcolor=a1.get_color())
	ax2.tick_params(axis='y', labelcolor=a2.get_color())
	ax3.tick_params(axis='y', labelcolor=a3.get_color())
	# ax4.tick_params(axis='y', labelcolor=a4.get_color())
	ax3.legend(loc='upper right', fontsize='x-small')
	ax1.set_ylim(bottom=0)
	ax2.set_ylim(bottom=0)
	ax3.set_ylim(bottom=0,top=fit_of_plasma_centre_and_area[3]*2.2)
	# ax4.set_ylim(bottom=0)
	ax1.grid()
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.png', bbox_inches='tight')
	plt.close()


	fig, ax = plt.subplots( 2,4,figsize=(34, 18), squeeze=False)
	fig2, ax2 = plt.subplots( 2,2,figsize=(18, 18), squeeze=False)
	fig.suptitle(pre_title+'Shape of the max temperature increase for '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms, de=%.3g' %(frequency,int_time*1000,de))
	fig2.suptitle(pre_title+'Shape of the max temperature increase for '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms, de=%.3g' %(frequency,int_time*1000,de))
	ax[0,0].set_title('large area max counts')
	ax[1,0].set_title('fitted peak counts')

	ax2[0,0].set_title('large area max counts')
	ax2[1,0].set_title('fitted peak counts')

	ax[0,1].set_title('large area max counts')
	ax[1,1].set_title('fitted peak counts')

	ensamble_peak_shape = np.max(twoD_peak_evolution_counts_averaged_full_mean_filter[:,((x-fit2[0][2])*p2d_h)**2+((y-fit2[0][1])*p2d_v)**2<(IR_shape[3]*p2d_v/2)**2],axis=1)
	peak_shape_time = np.float64(twoD_peak_evolution_time_averaged)
	max_counts_before_peaks = np.mean(ensamble_peak_shape[:np.abs(peak_shape_time+1.5/1000).argmin()])
	ensamble_peak_shape -= max_counts_before_peaks
	full_saved_file_dict['average_pulse_peak_data']['time'] = peak_shape_time
	full_saved_file_dict['average_pulse_peak_data']['counts']=dict([])
	full_saved_file_dict['average_pulse_peak_data']['counts']['delta'] = ensamble_peak_shape
	full_saved_file_dict['average_pulse_peak_data']['counts']['steady_state'] = max_counts_before_peaks
	# ensamble_peak_mean_shape = np.mean(twoD_peak_evolution_counts_averaged[:,((x-fit2[0][2])*p2d_h)**2+((y-fit2[0][1])*p2d_v)**2<(IR_shape[3]*p2d_v/2)**2],axis=1)
	# mean_counts_before_peaks = np.mean(ensamble_peak_mean_shape[:np.abs(peak_shape_time+1.5/1000).argmin()])
	# ensamble_peak_mean_shape -= mean_counts_before_peaks
	mean_counts_before_peaks = np.nanmax(twoD_peak_evolution_counts_averaged_before_full_mean_filter[((x-fit2[0][2])*p2d_h)**2+((y-fit2[0][1])*p2d_v)**2<(IR_shape[3]*p2d_v/2)**2])
	ensamble_peak_mean_shape = ensamble_peak_shape_counts_fitted[:,0]
	full_saved_file_dict['average_pulse_fitted_data']['counts']['steady_state'] = mean_counts_before_peaks
	full_saved_file_dict['average_pulse_fitted_data']['time'] = peak_shape_time
	ax[0,0].plot(1000*peak_shape_time,ensamble_peak_shape,'k',linewidth=2,label='mean profile')
	ax[0,0].axvline(x=1.5,color='g',linestyle='--')#plot([1.5]*2,[0,1e6],'--g')
	ax[1,0].plot(1000*peak_shape_time,ensamble_peak_mean_shape,'k',linewidth=2,label='mean profile')
	ax[1,0].axvline(x=1.5,color='g',linestyle='--')
	# if all_time_points>2*len(peak_shape_time):
	# 	ensamble_peak_shape = np.convolve(ensamble_peak_shape,np.ones(20)/20,'same')
	# ax[plot_index,0].plot(1000*peak_shape_time,ensamble_peak_shape,'r',linewidth=3)
	ax[0,0].set_xlabel('time [ms]')
	ax[0,0].set_ylabel('counts [au]')
	ax[1,0].set_xlabel('time [ms]')
	ax[1,0].set_ylabel('counts [au]')

	ax2[0,0].plot(1000*peak_shape_time,ensamble_peak_shape,'k',linewidth=2,label='mean profile')
	ax2[0,0].axvline(x=1.5,color='g',linestyle='--')
	ax2[1,0].plot(1000*peak_shape_time,ensamble_peak_mean_shape,'k',linewidth=2,label='mean profile')
	ax2[1,0].axvline(x=1.5,color='g',linestyle='--')
	ax2[0,0].set_xlabel('time [ms]')
	ax2[0,0].set_ylabel('counts [au]')
	ax2[1,0].set_xlabel('time [ms]')
	ax2[1,0].set_ylabel('counts [au]')

	ax[0,1].plot(1000*peak_shape_time,ensamble_peak_shape,'k',linewidth=2,label='mean profile')
	ax[0,1].axvline(x=1.5,color='g',linestyle='--')
	ax[1,1].plot(1000*peak_shape_time,ensamble_peak_mean_shape,'k',linewidth=2,label='mean profile')
	ax[1,1].axvline(x=1.5,color='g',linestyle='--')
	# if all_time_points>2*len(peak_shape_time):
	# 	ensamble_peak_shape = np.convolve(ensamble_peak_shape,np.ones(20)/20,'same')
	# ax[plot_index,0].plot(1000*peak_shape_time,ensamble_peak_shape,'r',linewidth=3)
	ax[0,1].set_xlabel('time [ms]')
	ax[0,1].set_ylabel('counts [au]')
	ax[1,1].set_xlabel('time [ms]')
	ax[1,1].set_ylabel('counts [au]')

	unsuccessful_fit = True
	# try:
	if False:	# Here I wanted to find the slope of the temperature rise and find it in the temperature decrease.
		start_start_pulse = (ensamble_peak_shape>ensamble_peak_shape.max()*0.01).argmax()
		end_start_pulse = (ensamble_peak_shape>ensamble_peak_shape.max()*0.3).argmax()
		end_end_pulse = -(np.flip(ensamble_peak_shape,axis=0)>ensamble_peak_shape.max()*0.1).argmax()
		start_end_pulse = -(np.flip(ensamble_peak_shape,axis=0)>ensamble_peak_shape.max()*0.5).argmax()
		tau_up = np.polyfit(peak_shape_time[start_start_pulse:end_start_pulse],np.log(ensamble_peak_shape[start_start_pulse:end_start_pulse]),1)
		# plt.figure()
		# plt.plot(peak_shape_time[start_start_pulse:end_start_pulse],np.log(ensamble_peak_shape[start_start_pulse:end_start_pulse]))
		plt.plot(1000*peak_shape_time[start_start_pulse:end_start_pulse],np.exp(np.polyval(tau_up,peak_shape_time[start_start_pulse:end_start_pulse])),'--b')
		def to_fit(x,constant):
			out = np.exp(np.polyval([-tau_up[0],constant],x))
			return out
		guess=[1]
		fit = curve_fit(to_fit, peak_shape_time[start_end_pulse:end_end_pulse], ensamble_peak_shape[start_end_pulse:end_end_pulse], guess,maxfev=int(1e4))
		# plt.plot(peak_shape_time[start_end_pulse:end_end_pulse],np.log(ensamble_peak_shape[start_end_pulse:end_end_pulse]))
		plt.plot(1000*peak_shape_time[ensamble_peak_shape.argmax():end_end_pulse],np.exp(np.polyval([-tau_up[0],fit[0]],peak_shape_time[ensamble_peak_shape.argmax():end_end_pulse])),'--b',linewidth=3,label='time constant=%.3g' %(tau_up[0]))
		# plt.pause(0.01)
	elif True:	# we now believe that most of the peak is actually due to prompt emission of infrared line emission, so not due to an actual temperature increase
		start_start_pulse = (ensamble_peak_shape>ensamble_peak_shape.max()*0.6).argmax()
		start_pulse = (ensamble_peak_shape>ensamble_peak_shape.max()*0.2).argmax()
		end_end_pulse = -(np.flip(ensamble_peak_shape,axis=0)>ensamble_peak_shape.max()*0.6).argmax()
		pre_pulse = np.abs(peak_shape_time*1000+2).argmin()
		post_pulse = np.abs(peak_shape_time*1000-2).argmin()
		peak_pulse = ensamble_peak_shape.argmax()
		# gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
		# guess=[peak_shape_time.max(),(peak_shape_time[peak_pulse]-peak_shape_time[start_start_pulse])/2,peak_shape_time[peak_pulse]]
		gauss = lambda x, sig, x0: ensamble_peak_shape.max() * np.exp(-(((x - x0) / sig) ** 2)/2) + ensamble_peak_shape[pre_pulse]
		guess=[(peak_shape_time[peak_pulse]-peak_shape_time[start_start_pulse])/2,peak_shape_time[peak_pulse]]
		# peak_pulse = np.abs(peak_shape_time - (peak_shape_time[peak_pulse]+2e-4)).argmin()
		fit = curve_fit(gauss, np.float64(peak_shape_time[start_start_pulse:end_end_pulse]), np.float64(ensamble_peak_shape[start_start_pulse:end_end_pulse]), guess,maxfev=int(1e4))
		ax[0,0].plot(1000*peak_shape_time[start_start_pulse:end_end_pulse],gauss(peak_shape_time[start_start_pulse:end_end_pulse],*fit[0]),'--b',linewidth=2,label='peak fit')
		ax[0,0].plot(1000*peak_shape_time[pre_pulse:start_start_pulse],gauss(peak_shape_time[pre_pulse:start_start_pulse],*fit[0]),':b',label='raise extension')
		ax[0,1].plot(1000*peak_shape_time[start_start_pulse:end_end_pulse],gauss(peak_shape_time[start_start_pulse:end_end_pulse],*fit[0]),'--b',linewidth=2,label='peak fit')
		ax[0,1].plot(1000*peak_shape_time[pre_pulse:start_start_pulse],gauss(peak_shape_time[pre_pulse:start_start_pulse],*fit[0]),':b',label='raise extension')
		peak_pulse = np.abs(peak_shape_time-fit[0][-1]).argmin()
		end_end_pulse = np.abs(peak_shape_time-fit[0][-1]-fit[0][-2]*4).argmin()
		ax[0,0].axvline(x=1e3*fit[0][-1],color='y',linestyle='--',label='prompt peak')#plot([1000*fit[0][-1]]*2,[0,1e6],'--y',label='prompt peak')
		ax[1,0].axvline(x=1e3*fit[0][-1],color='y',linestyle='--',label='prompt peak')

		ax2[0,0].axvline(x=1e3*fit[0][-1],color='y',linestyle='--',label='prompt peak')
		ax2[1,0].axvline(x=1e3*fit[0][-1],color='y',linestyle='--',label='prompt peak')

		ax[0,1].axvline(x=1e3*fit[0][-1],color='y',linestyle='--',label='prompt peak')
		ax[1,1].axvline(x=1e3*fit[0][-1],color='y',linestyle='--',label='prompt peak')
		post_pulse = np.abs(peak_shape_time-(2*peak_shape_time[peak_pulse] - peak_shape_time[pre_pulse])).argmin()
		ax[0,0].plot(1000*peak_shape_time[peak_pulse:post_pulse],ensamble_peak_shape[peak_pulse:post_pulse] - gauss(peak_shape_time[peak_pulse:post_pulse],*fit[0]),'-b',linewidth=2,label='decrease-fit')
		ax[0,1].plot(1000*peak_shape_time[peak_pulse:post_pulse],ensamble_peak_shape[peak_pulse:post_pulse] - gauss(peak_shape_time[peak_pulse:post_pulse],*fit[0]),'-b',linewidth=2,label='decrease-fit')
		interpolator = interp1d(peak_shape_time[pre_pulse:peak_pulse+1]-fit[0][-1],ensamble_peak_shape[pre_pulse:peak_pulse+1],bounds_error=False,fill_value=(ensamble_peak_shape[pre_pulse],ensamble_peak_shape[peak_pulse]))
		interpolator_mean = interp1d(peak_shape_time[pre_pulse:peak_pulse+1]-fit[0][-1],ensamble_peak_mean_shape[pre_pulse:peak_pulse+1],bounds_error=False,fill_value=(ensamble_peak_mean_shape[pre_pulse],ensamble_peak_mean_shape[peak_pulse]))
		ax[0,0].plot(1000*peak_shape_time[peak_pulse:post_pulse],interpolator(fit[0][-1]-peak_shape_time[peak_pulse:post_pulse]),'--r',linewidth=1,label='mirror of raise')
		ax[1,0].plot(1000*peak_shape_time[peak_pulse:post_pulse],interpolator_mean(fit[0][-1]-peak_shape_time[peak_pulse:post_pulse]),'--r',linewidth=1,label='mirror of raise')

		ax2[0,0].plot(1000*peak_shape_time[peak_pulse:post_pulse],interpolator(fit[0][-1]-peak_shape_time[peak_pulse:post_pulse]),'--r',linewidth=1,label='mirror of raise')
		ax2[1,0].plot(1000*peak_shape_time[peak_pulse:post_pulse],interpolator_mean(fit[0][-1]-peak_shape_time[peak_pulse:post_pulse]),'--r',linewidth=1,label='mirror of raise')

		ax[0,1].plot(1000*peak_shape_time[peak_pulse:post_pulse],interpolator(fit[0][-1]-peak_shape_time[peak_pulse:post_pulse]),'--r',linewidth=1,label='mirror of raise')
		ax[1,1].plot(1000*peak_shape_time[peak_pulse:post_pulse],interpolator_mean(fit[0][-1]-peak_shape_time[peak_pulse:post_pulse]),'--r',linewidth=1,label='mirror of raise')

		ax[0,0].set_xlim(left=-2,right=2)
		ax[0,0].set_ylim(bottom=0,top=1.5*ensamble_peak_shape.max())
		ax[0,0].grid()

		ax2[0,0].set_xlim(left=-2,right=2)
		ax2[0,0].set_ylim(bottom=0,top=1.5*ensamble_peak_shape.max())
		ax2[0,0].grid()

		ax[0,1].set_xlim(left=0,right=15)
		ax[0,1].set_ylim(bottom=-1,top=max(3,ensamble_peak_shape[np.abs(peak_shape_time-1e-3).argmin()]))
		ax[0,1].grid()
		ensamble_peak_shape_full = ensamble_peak_shape
		SS_ensamble = np.mean(ensamble_peak_shape[np.abs(peak_shape_time+2e-3).argmin():np.abs(peak_shape_time+1e-3).argmin()])
		ensamble_peak_shape = ensamble_peak_shape[peak_pulse:]-interpolator(fit[0][-1]-peak_shape_time[peak_pulse:])
		ax[0,0].plot(1000*peak_shape_time[peak_pulse:],ensamble_peak_shape,'-r',linewidth=2,label='decrease-mirror')
		ax[0,1].plot(1000*peak_shape_time[peak_pulse:],ensamble_peak_shape,'-r',linewidth=2,label='decrease-mirror')

		ax2[0,0].plot(1000*peak_shape_time[peak_pulse:],ensamble_peak_shape,'-r',linewidth=2,label='decrease-mirror')

		ax[1,0].set_xlim(left=-2,right=2)
		ax[1,0].set_ylim(bottom=0,top=1.5*ensamble_peak_mean_shape.max())
		ax[1,0].grid()

		ax2[1,0].set_xlim(left=-2,right=2)
		ax2[1,0].set_ylim(bottom=0,top=1.5*ensamble_peak_mean_shape.max())
		ax2[1,0].grid()

		ax[1,1].set_xlim(left=0,right=15)
		ax[1,1].set_ylim(bottom=-1,top=max(3,ensamble_peak_mean_shape[np.abs(peak_shape_time-1e-3).argmin()]))
		ax[1,1].grid()
		ensamble_peak_mean_shape_full = ensamble_peak_mean_shape
		SS_ensamble_mean = np.mean(ensamble_peak_mean_shape_full[np.abs(peak_shape_time+2e-3).argmin():np.abs(peak_shape_time+1e-3).argmin()])
		ensamble_peak_mean_shape = ensamble_peak_mean_shape[peak_pulse:]-interpolator_mean(fit[0][-1]-peak_shape_time[peak_pulse:])
		ax[1,0].plot(1000*peak_shape_time[peak_pulse:],ensamble_peak_mean_shape,'-r',linewidth=2,label='decrease-mirror')
		ax[1,1].plot(1000*peak_shape_time[peak_pulse:],ensamble_peak_mean_shape,'-r',linewidth=2,label='decrease-mirror')

		ax2[1,0].plot(1000*peak_shape_time[peak_pulse:],ensamble_peak_mean_shape,'-r',linewidth=2,label='decrease-mirror')

		ax[0,0].legend(loc='best', fontsize='x-small')
		ax[1,0].legend(loc='best', fontsize='x-small')
		ax[0,1].legend(loc='best', fontsize='x-small')
		ax[1,1].legend(loc='best', fontsize='x-small')

		ax2[0,0].legend(loc='best', fontsize='small')
		ax2[1,0].legend(loc='best', fontsize='small')

		ensamble_peak_shape[ensamble_peak_shape<0]=0
		ensamble_peak_shape += max_counts_before_peaks
		ensamble_peak_shape_full += max_counts_before_peaks
		SS_ensamble += max_counts_before_peaks
		ensamble_peak_mean_shape += mean_counts_before_peaks
		ensamble_peak_mean_shape[ensamble_peak_mean_shape<0]=0
		ensamble_peak_mean_shape_full += mean_counts_before_peaks
		ensamble_peak_mean_shape_full[ensamble_peak_mean_shape_full<0]=0
		SS_ensamble_mean += mean_counts_before_peaks

		emissivity_ensamble_peak_shape = np.exp(counts_to_emissivity(ensamble_peak_shape))
		ensamble_peak_shape = np.exp(counts_to_temperature(ensamble_peak_shape))
		ensamble_peak_shape_full = np.exp(counts_to_temperature(ensamble_peak_shape_full))
		SS_ensamble = np.exp(counts_to_temperature(max(SS_ensamble,0)))
		full_saved_file_dict['average_pulse_peak_data']['temperature'] = dict([])
		full_saved_file_dict['average_pulse_peak_data']['temperature']['full'] = ensamble_peak_shape_full
		full_saved_file_dict['average_pulse_peak_data']['temperature']['steady_state'] = SS_ensamble
		# emissivity_ensamble_peak_mean_shape = np.exp(counts_to_emissivity(ensamble_peak_mean_shape))
		# emissivity_ensamble_peak_mean_shape_full = np.exp(counts_to_emissivity(ensamble_peak_mean_shape_full))
		# ensamble_peak_mean_shape = np.exp(counts_to_temperature(ensamble_peak_mean_shape))
		# ensamble_peak_mean_shape_full = np.exp(counts_to_temperature(ensamble_peak_mean_shape_full))
		# SS_ensamble_mean = np.exp(counts_to_temperature(max(SS_ensamble_mean,0)))
		emissivity_ensamble_peak_mean_shape = np.exp(counts_to_emissivity(ensamble_peak_mean_shape))[peak_pulse:]
		emissivity_ensamble_peak_mean_shape_full = np.exp(counts_to_emissivity(ensamble_peak_mean_shape_full))[peak_pulse:]
		SS_ensamble_mean = np.mean(np.nanmax(twoD_peak_evolution_temp_averaged_before_full_mean_filter[((x-fit2[0][2])*p2d_h)**2+((y-fit2[0][1])*p2d_v)**2<(IR_shape[3]*p2d_v/2)**2]) + ensamble_peak_shape_temp_fitted[:,0][:np.abs(peak_shape_time+1.5/1000).argmin()])
		# SS_ensamble_mean = np.mean((ensamble_peak_shape_temp_fitted[:,0]+ensamble_peak_shape_temp_fitted[:,2])[:np.abs(peak_shape_time+1.5/1000).argmin()])
		ensamble_peak_mean_shape = SS_ensamble_mean + ensamble_peak_shape_temp_fitted[:,0][peak_pulse:]
		ensamble_peak_mean_shape_full = SS_ensamble_mean + ensamble_peak_shape_temp_fitted[:,0]
		ensamble_peak_mean_shape_std = (np.std(ensamble_peak_shape_temp_fitted[:,0][:np.abs(peak_shape_time+1.5/1000).argmin()])**2 + ensamble_peak_shape_temp_fitted_std[:,0][peak_pulse:]**2 )**0.5
		ensamble_peak_mean_shape_full_std = (np.std(ensamble_peak_shape_temp_fitted[:,0][:np.abs(peak_shape_time+1.5/1000).argmin()])**2 + SS_ensamble_mean + ensamble_peak_shape_temp_fitted_std[:,0]**2 )**0.5
		full_saved_file_dict['average_pulse_fitted_data']['temperature']['steady_state'] = SS_ensamble_mean

		temp_time = peak_shape_time[peak_pulse:]
		ax[0,2].plot(1000*temp_time,ensamble_peak_shape,'-k',linewidth=2)
		# ax[0,2].plot([1e3*temp_time[ensamble_peak_shape.argmax()]]*2,[ensamble_peak_shape.min(),ensamble_peak_shape.max()],'--',label='dT=%.3g°C' %(ensamble_peak_shape[ensamble_peak_shape.argmax()]-SS_ensamble))
		ax[0,2].axvline(x=1e3*temp_time[ensamble_peak_shape.argmax()],linestyle='--',label='dT=%.3g°C' %(ensamble_peak_shape[ensamble_peak_shape.argmax()]-SS_ensamble))
		# ax[0,2].plot([1000*temp_time[np.abs(temp_time-1e-3).argmin()]]*2,[ensamble_peak_shape.min(),ensamble_peak_shape.max()],'--',label='dT=%.3g°C' %(ensamble_peak_shape[np.abs(temp_time-1e-3).argmin()]-SS_ensamble))
		ax[0,2].axvline(x=1e3*temp_time[np.abs(temp_time-1e-3).argmin()],linestyle='--',label='dT=%.3g°C' %(ensamble_peak_shape[np.abs(temp_time-1e-3).argmin()]-SS_ensamble))
		ax[0,2].plot([1000*temp_time[np.abs(temp_time-1.5e-3).argmin()]]*2,[ensamble_peak_shape.min(),ensamble_peak_shape.max()],'--g',label='dT=%.3g°C' %(ensamble_peak_shape[np.abs(temp_time-1.5e-3).argmin()]-SS_ensamble))
		ax[0,2].axvline(x=1e3*temp_time[np.abs(temp_time-1.5e-3).argmin()],color='g',linestyle='--',label='dT=%.3g°C' %(ensamble_peak_shape[np.abs(temp_time-1.5e-3).argmin()]-SS_ensamble))
		ax[0,3].axvline(x=1000*temp_time[np.abs(temp_time-1.5e-3).argmin()],color='g',linestyle='--',label='dT=%.3g°C' %(ensamble_peak_shape[np.abs(temp_time-1.5e-3).argmin()]-SS_ensamble))
		ax[0,3].axvline(x=1000*temp_time[np.abs(temp_time-2e-3).argmin()],color='b',linestyle='--',label='dT(mean peak + 2-3ms)=%.3g°C' %(np.mean(ensamble_peak_shape[np.abs(temp_time-2e-3).argmin():np.abs(temp_time-3e-3).argmin()])-SS_ensamble))
		ax[0,3].axvline(x=1000*temp_time[np.abs(temp_time-3e-3).argmin()],color='b',linestyle='--')
		ax[0,3].plot(1000*peak_shape_time,ensamble_peak_shape_full,'k',linewidth=2)
		ax[0,3].set_ylim(top=min(ensamble_peak_shape_full.max()+40,ensamble_peak_shape_full.max()*1.07))

		# ax2[0,1].plot([1000*temp_time[np.abs(temp_time-1.5e-3).argmin()]]*2,[ensamble_peak_shape_full.min(),ensamble_peak_shape_full.max()],'--g',label='dT=%.3g°C' %(ensamble_peak_shape[np.abs(temp_time-1.5e-3).argmin()]-SS_ensamble))
		ax2[0,1].axvline(x=1000*temp_time[np.abs(temp_time-1.5e-3).argmin()],color='g',linestyle='--')
		ax2[0,1].axvline(x=1000*temp_time[np.abs(temp_time-10e-3).argmin()],color='b',linestyle='--',label='dT(mean peak + 10-11ms)=%.3g°C' %(np.mean(ensamble_peak_shape[np.abs(temp_time-10e-3).argmin():np.abs(temp_time-11e-3).argmin()])-SS_ensamble))
		ax2[0,1].axvline(x=1000*temp_time[np.abs(temp_time-11e-3).argmin()],color='b',linestyle='--')
		ax2[0,1].plot(1000*peak_shape_time,ensamble_peak_shape_full,'k',linewidth=2)
		ax2[0,1].set_ylim(top=min(ensamble_peak_shape_full.max()+40,ensamble_peak_shape_full.max()*1.07))

		ax[1,2].plot(1000*temp_time,ensamble_peak_mean_shape,'-k',linewidth=2)
		ax[1,2].axvline(x=1000*temp_time[ensamble_peak_mean_shape.argmax()],linestyle='--',label='dT=%.3g°C' %(ensamble_peak_mean_shape[ensamble_peak_mean_shape.argmax()]-SS_ensamble_mean))
		ax[1,2].axvline(x=1000*temp_time[np.abs(temp_time-1e-3).argmin()],linestyle='--',label='dT=%.3g°C' %(ensamble_peak_mean_shape[np.abs(temp_time-1e-3).argmin()]-SS_ensamble_mean))
		ax[1,2].axvline(x=1000*temp_time[np.abs(temp_time-1.5e-3).argmin()],color='g',linestyle='--',label='dT=%.3g°C' %(ensamble_peak_mean_shape[np.abs(temp_time-1.5e-3).argmin()]-SS_ensamble_mean))
		ax[1,3].axvline(x=1000*temp_time[np.abs(temp_time-1.5e-3).argmin()],color='g',linestyle='--',label='dT=%.3g°C' %(ensamble_peak_mean_shape[np.abs(temp_time-1.5e-3).argmin()]-SS_ensamble_mean))
		ax[1,3].axvline(x=1000*temp_time[np.abs(temp_time-2e-3).argmin()],color='b',linestyle='--',label='dT(mean peak + 2-3ms)=%.3g°C' %(np.mean(ensamble_peak_mean_shape[np.abs(temp_time-2e-3).argmin():np.abs(temp_time-3e-3).argmin()])-SS_ensamble_mean))
		ax[1,3].axvline(x=1000*temp_time[np.abs(temp_time-3e-3).argmin()],color='b',linestyle='--')
		ax[1,3].plot(1000*peak_shape_time,ensamble_peak_mean_shape_full,'k',linewidth=2)
		ax[1,3].set_ylim(top=min(ensamble_peak_mean_shape_full.max()+40,ensamble_peak_mean_shape_full.max()*1.07))

		# ax2[1,1].plot([1000*temp_time[np.abs(temp_time-1.5e-3).argmin()]]*2,[ensamble_peak_mean_shape_full.min(),ensamble_peak_mean_shape_full.max()],'--g',label='dT=%.3g°C' %(ensamble_peak_mean_shape[np.abs(temp_time-1.5e-3).argmin()]-SS_ensamble_mean))
		ax2[1,1].axvline(x=1000*temp_time[np.abs(temp_time-1.5e-3).argmin()],color='g',linestyle='--')
		ax2[1,1].axvline(x=1000*temp_time[np.abs(temp_time-10e-3).argmin()],color='b',linestyle='--',label='dT(mean peak + 10-11ms)=%.3g°C' %(np.mean(ensamble_peak_mean_shape[np.abs(temp_time-10e-3).argmin():np.abs(temp_time-11e-3).argmin()])-SS_ensamble_mean))
		ax2[1,1].axvline(x=1000*temp_time[np.abs(temp_time-11e-3).argmin()],color='b',linestyle='--')
		ax2[1,1].plot(1000*peak_shape_time,ensamble_peak_mean_shape_full,'k',linewidth=2)
		ax2[1,1].set_ylim(top=min(ensamble_peak_mean_shape_full.max()+40,ensamble_peak_mean_shape_full.max()*1.07))
		full_saved_file_dict['average_pulse_peak_data']['temperature']['peak'] = ensamble_peak_shape[ensamble_peak_shape.argmax()]-SS_ensamble
		full_saved_file_dict['average_pulse_peak_data']['temperature']['peak_plus_1'] = ensamble_peak_shape[np.abs(temp_time-1e-3).argmin()]-SS_ensamble
		full_saved_file_dict['average_pulse_peak_data']['temperature']['peak_plus_1.5'] = ensamble_peak_shape[np.abs(temp_time-1.5e-3).argmin()]-SS_ensamble
		full_saved_file_dict['average_pulse_peak_data']['temperature']['peak_plus_2_3'] = np.mean(ensamble_peak_shape[np.abs(temp_time-2e-3).argmin():np.abs(temp_time-3e-3).argmin()])-SS_ensamble
		full_saved_file_dict['average_pulse_peak_data']['temperature']['peak_plus_2_3_sigma'] = np.std(ensamble_peak_shape[np.abs(temp_time-2e-3).argmin():np.abs(temp_time-3e-3).argmin()])
		full_saved_file_dict['average_pulse_peak_data']['temperature']['peak_plus_10_11'] = np.mean(ensamble_peak_shape[np.abs(temp_time-10e-3).argmin():np.abs(temp_time-11e-3).argmin()])-SS_ensamble
		full_saved_file_dict['average_pulse_peak_data']['temperature']['peak_plus_10_11_sigma'] = np.std(ensamble_peak_shape[np.abs(temp_time-10e-3).argmin():np.abs(temp_time-11e-3).argmin()])
		full_saved_file_dict['average_pulse_fitted_data']['temperature']['peak'] = ensamble_peak_mean_shape[ensamble_peak_mean_shape.argmax()]-SS_ensamble_mean
		full_saved_file_dict['average_pulse_fitted_data']['temperature']['peak_plus_1'] = ensamble_peak_mean_shape[np.abs(temp_time-1e-3).argmin()]-SS_ensamble_mean
		full_saved_file_dict['average_pulse_fitted_data']['temperature']['peak_plus_1.5'] = ensamble_peak_mean_shape[np.abs(temp_time-1.5e-3).argmin()]-SS_ensamble_mean
		full_saved_file_dict['average_pulse_fitted_data']['temperature']['peak_plus_2_3'] = np.mean(ensamble_peak_mean_shape[np.abs(temp_time-2e-3).argmin():np.abs(temp_time-3e-3).argmin()])-SS_ensamble_mean
		full_saved_file_dict['average_pulse_fitted_data']['temperature']['peak_plus_2_3_sigma'] = np.std(ensamble_peak_mean_shape[np.abs(temp_time-2e-3).argmin():np.abs(temp_time-3e-3).argmin()])
		full_saved_file_dict['average_pulse_fitted_data']['temperature']['peak_plus_10_11'] = np.mean(ensamble_peak_mean_shape[np.abs(temp_time-10e-3).argmin():np.abs(temp_time-11e-3).argmin()])-SS_ensamble_mean
		full_saved_file_dict['average_pulse_fitted_data']['temperature']['peak_plus_10_11_sigma'] = np.std(ensamble_peak_mean_shape[np.abs(temp_time-10e-3).argmin():np.abs(temp_time-11e-3).argmin()])

		end_end_pulse = np.abs(temp_time*1000-1.5).argmin()	# start fit time 1.5ms
		post_pulse = np.abs(temp_time*1000-30).argmin()	# end fit time 30ms
		SS_time = int(1e-3*frequency*10)
		# temp = np.log(ensamble_peak_shape[end_end_pulse:post_pulse]-ensamble_peak_shape[-SS_time:].mean())
		# tau_down = np.polyfit(temp_time[end_end_pulse:post_pulse][np.isfinite(temp)],temp[np.isfinite(temp)],1)
		# min_target_temperature = np.exp(np.polyval((tau_down),temp_time))+ensamble_peak_shape[-SS_time:].mean()

		sigmaSB=5.6704e-08 #[W/(m2 K4)]
		if False:	# the exponential decay is an idea I had based on heat transport, but it is not representative of heat transport in time!
			def exponential_decay(time,*args):
				out = args[1]*np.exp(-time/args[2]) + args[0]
				return out

			guess = [ensamble_peak_shape[-SS_time:].mean(),ensamble_peak_shape[np.abs(temp_time-1e-3).argmin()],3e-3]
			bds=[[20,0,1e-3],[np.inf,np.inf,1e-1]]
			fit = curve_fit(exponential_decay, temp_time[end_end_pulse:post_pulse], ensamble_peak_shape[end_end_pulse:post_pulse], guess,bounds=bds,maxfev=int(1e4))
			min_target_temperature = exponential_decay(temp_time,*fit[0])
			# # area_of_interest = 2*np.pi*IR_shape[3]**2 * (0.3e-3)**2
			if False:	# fixed area
				area_of_interest = 2*np.pi*(0.026/2)**2	# 26mm is the diameter of the standard magnum target area_of_interest_sigma
			sigmaSB=5.6704e-08 #[W/(m2 K4)]
			# target_effected_thickness = 0.2e-3
			target_effected_thickness = 0.2e-3	# 1mm thickness of the target (approx.)
			# min_power = thermal_conductivity*(np.exp(np.polyval((tau_down),temp_time)))/target_effected_thickness + 1*sigmaSB*((min_target_temperature+273.15)**4 - 300**4)
			# min_power = area_of_interest*np.sum(min_power)/(10*frequency)
			# max_power = heat_capacity*density*area_of_interest*target_effected_thickness*np.exp(np.polyval((tau_down),0))
			min_power = thermal_conductivity*(min_target_temperature-fit[0][0])/target_effected_thickness + emissivity_ensamble_peak_shape*sigmaSB*((ensamble_peak_shape_full[peak_pulse:]+273.15)**4 - 300**4)
			min_power = area_of_interest*np.sum(min_power)/(10*frequency)
			# max_power = thermal_conductivity*(ensamble_peak_shape_full-ensamble_peak_shape_full.min())/target_effected_thickness + emissivity_ensamble_peak_shape_full*sigmaSB*((ensamble_peak_shape_full+273.15)**4 - 300**4)
			# max_power = area_of_interest*np.sum(max_power)/frequency
			max_power = heat_capacity*density*area_of_interest*target_effected_thickness*(fit[0][1])
			ax[0,2].plot(1000*temp_time,min_target_temperature,'--r',linewidth=2,label='T SS=%.3g°C\ndT=%.3g°C\ntau=%.3gms\nEnergy\nfrom area=%.3gJ\nfrom Tmax=%.3gJ' %(fit[0][0],fit[0][1],fit[0][2]*1000,min_power,max_power))
			ax[0,3].plot(1000*temp_time,min_target_temperature,'--r',linewidth=2)
			ax[0,2].set_ylim(bottom=fit[0][0]-2,top=10+min_target_temperature.max())

		# this aproach comes from a reference given me by Bruce: EVAPORATION FOR HEAT PULSES ON Ni, MO, W AND ATJ GRAPHITE AS FIRST WALL MATERIALS
		def semi_infinite_sink_decrease(time,*args):
			out = args[1]/args[2]*2/((np.pi*thermal_conductivity(args[0])*heat_capacity(args[0])*density)**0.5) *(time**0.5 - (time-args[2])**0.5) + args[0]
			# print(out)
			return out

		def semi_infinite_sink_increase(time,*args):
			out = args[1]/args[2]*2/((np.pi*thermal_conductivity(args[0])*heat_capacity(args[0])*density)**0.5) *(time**0.5) + args[0]
			# print(out)
			return out

		# def semi_infinite_sink_full_decrease(time,*args):
		# 	out = args[1]/args[2]*2/((np.pi*thermal_conductivity(args[0])*heat_capacity(args[0])*density)**0.5) *((time-args[3])**0.5 - (time-args[3]-args[2])**0.5) + args[0]
		# 	# print(out)
		# 	return out
		def semi_infinite_sink_full_decrease(max_total_time):
			def function(time,*args):
				out = (args[1]/((max_total_time-args[3])*args[2]))*2/((np.pi*thermal_conductivity(args[0])*heat_capacity(args[0])*density)**0.5) *((time-args[3])**0.5 - (time-args[3]-(max_total_time-args[3])*args[2])**0.5) + args[0]
				# print(out)
				# return np.nanmax([np.ones_like(out)*args[0],out],axis=0)
				return out
			return function

		def semi_infinite_sink_full_decrease_forced_T_SS(max_total_time,T_SS):
			def function(time,*args):
				out = (args[0]/((max_total_time-args[2])*args[1]))*2/((np.pi*thermal_conductivity(T_SS)*heat_capacity(T_SS)*density)**0.5) *((time-args[2])**0.5 - (time-args[2]-(max_total_time-args[2])*args[1])**0.5) + T_SS
				# print(out)
				# return np.nanmax([np.ones_like(out)*args[0],out],axis=0)
				return out
			return function

		def semi_infinite_sink_full_decrease_forced_duration(max_total_time,duration):
			def function(time,*args):
				out = (args[1]/duration)*2/((np.pi*thermal_conductivity(args[0])*heat_capacity(args[0])*density)**0.5) *((time-args[2])**0.5 - (time-args[2]-duration)**0.5)
				# print(out)
				# return np.nanmax([np.zeros_like(out),out],axis=0)+T_SS
				return out+args[0]
			return function


		# def semi_infinite_sink_full_increase(time,*args):
		# 	out = args[1]/args[2]*2/((np.pi*thermal_conductivity(args[0])*heat_capacity(args[0])*density)**0.5) *((time-args[3])**0.5) + args[0]
		# 	# print(out)
		# 	return out
		def semi_infinite_sink_full_increase(max_total_time):
			def function(time,*args):
				out = (args[1]/((max_total_time-args[3])*args[2]))*2/((np.pi*thermal_conductivity(args[0])*heat_capacity(args[0])*density)**0.5) *((time-args[3])**0.5) + args[0]
				# print(out)
				return np.nanmax([np.ones_like(out)*args[0],out],axis=0)
			return function

		def semi_infinite_sink_full_increase_forced_T_SS(max_total_time,T_SS):
			def function(time,*args):
				out = (args[0]/((max_total_time-args[2])*args[1]))*2/((np.pi*thermal_conductivity(T_SS)*heat_capacity(T_SS)*density)**0.5) *((time-args[2])**0.5) + T_SS
				# print(out)
				return np.nanmax([np.ones_like(out)*T_SS,out],axis=0)
			return function

		def semi_infinite_sink_full_increase_forced_duration(max_total_time,duration):
			def function(time,*args):
				out = (args[1]/duration)*2/((np.pi*thermal_conductivity(args[0])*heat_capacity(args[0])*density)**0.5) *((time-args[2])**0.5)
				# print(out)
				return np.nanmax([np.zeros_like(out),out],axis=0)+args[0]
				# return out+T_SS
			return function

		T0 = ensamble_peak_shape_full[:np.abs(peak_shape_time+2e-3).argmin()].mean()
		T0_std = max(2,ensamble_peak_shape_full[:np.abs(peak_shape_time+2e-3).argmin()].std())
		# guess = [T0,1e5,1e-3]
		# bds=[[T0-2*T0_std,1e-6,1e-4],[T0+2*T0_std,np.inf,temp_time[end_end_pulse]]]
		# guess = [SS_ensamble,1e5,1e-3]
		# bds=[[SS_ensamble-1e-3,1e-6,1e-4],[SS_ensamble+1e-3,np.inf,temp_time[end_end_pulse]]]
		guess = [SS_ensamble,1e5,0.5,peak_shape_time[start_pulse]]
		bds=[[SS_ensamble-2*T0_std,1e-6,1e-2,peak_shape_time[start_pulse]*1.5],[SS_ensamble+2*T0_std,np.inf,1,temp_time[end_end_pulse]]]
		fit = curve_fit(semi_infinite_sink_full_decrease(temp_time[end_end_pulse]), np.float64(temp_time[end_end_pulse:post_pulse]), np.float64(ensamble_peak_shape[end_end_pulse:post_pulse]), guess,x_scale=[100,1e5,1e-3,1e-4],bounds=bds,absolute_sigma=True,maxfev=int(1e5),ftol=1e-15,xtol=1e-15)
		valid = np.logical_not(np.isnan(semi_infinite_sink_full_decrease(temp_time[end_end_pulse])(temp_time[end_end_pulse:post_pulse],*fit[0])))
		R2 = 1-np.sum(((ensamble_peak_shape[end_end_pulse:post_pulse]-semi_infinite_sink_full_decrease(temp_time[end_end_pulse])(temp_time[end_end_pulse:post_pulse],*fit[0]))**2)[valid])/np.sum(((ensamble_peak_shape[end_end_pulse:post_pulse]-np.mean(ensamble_peak_shape[end_end_pulse:post_pulse]))**2)[valid])
		fit_wit_errors = correlated_values(fit[0],fit[1])
		pulse_energy_density = (fit_wit_errors[1]+ufloat(0,0.1*fit_wit_errors[1].nominal_value))	# this to add the uncertainly coming from the method itself
		pulse_energy = pulse_energy_density*area_of_interest_sigma#*1/1.08	# 1/1.08 correction from the forward modeling check # 10/02/2021 now it doesn't seem to be needed
		pulse_duration_ms = 1e3*(temp_time[end_end_pulse]-fit_wit_errors[3])*fit_wit_errors[2]
		# ax[0,2].plot(1000*peak_shape_time,semi_infinite_sink_decrease(peak_shape_time,*fit[0]),'--m',linewidth=2,label='semi-infinite solid\nT SS=%.3g°C\nEdens=%.3gJ/m2\ntau=%.3gms\nEnergy=%.3gJ' %(fit[0][0],fit[0][1]*fit[0][2],fit[0][2]*1000,fit[0][1]*fit[0][2]*area_of_interest))
		# ax[0,3].plot(1000*peak_shape_time,semi_infinite_sink_decrease(peak_shape_time,*fit[0]),'--m',linewidth=2,label='semi-infinite solid\nT SS=%.3g°C\nEdens=%.3gJ/m2\ntau=%.3gms\nEnergy=%.3gJ' %(fit[0][0],fit[0][1]*fit[0][2],fit[0][2]*1000,fit[0][1]*fit[0][2]*area_of_interest))
		# ax[0,2].plot(1000*peak_shape_time,semi_infinite_sink_decrease(peak_shape_time,*fit[0]),'--m',linewidth=1,label='semi-infinite solid t0=peak fixed\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,fit_wit_errors[2].nominal_value*1e3,fit_wit_errors[2].std_dev*1000,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
		# ax[0,3].plot(1000*peak_shape_time,semi_infinite_sink_decrease(peak_shape_time,*fit[0]),'--m',linewidth=1,label='semi-infinite solid t0=peak fixed\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,fit_wit_errors[2].nominal_value*1e3,fit_wit_errors[2].std_dev*1000,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
		# ax[0,2].plot(1000*peak_shape_time[peak_shape_time<fit[0][2]],semi_infinite_sink_increase(peak_shape_time[peak_shape_time<fit[0][2]],*fit[0]),'--m',linewidth=1)
		# ax[0,3].plot(1000*peak_shape_time[peak_shape_time<fit[0][2]],semi_infinite_sink_increase(peak_shape_time[peak_shape_time<fit[0][2]],*fit[0]),'--m',linewidth=1)
		ax[0,2].plot(1000*(peak_shape_time[peak_shape_time-fit[0][3]>pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_decrease(temp_time[end_end_pulse])(peak_shape_time[peak_shape_time-fit[0][3]>pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--m',linewidth=1,label='semi-infinite solid\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_duration_ms.nominal_value,pulse_duration_ms.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
		ax[0,3].plot(1000*(peak_shape_time[peak_shape_time-fit[0][3]>pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_decrease(temp_time[end_end_pulse])(peak_shape_time[peak_shape_time-fit[0][3]>pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--m',linewidth=1,label='semi-infinite solid\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_duration_ms.nominal_value,pulse_duration_ms.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
		ax[0,2].plot(1000*(peak_shape_time[peak_shape_time-fit[0][3]<=pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_increase(temp_time[end_end_pulse])(peak_shape_time[peak_shape_time-fit[0][3]<=pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--m',linewidth=1)
		ax[0,3].plot(1000*(peak_shape_time[peak_shape_time-fit[0][3]<=pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_increase(temp_time[end_end_pulse])(peak_shape_time[peak_shape_time-fit[0][3]<=pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--m',linewidth=1)
		# ax[0,2].set_ylim(bottom=fit[0][0]-2,top=np.nanmax(10+semi_infinite_sink_full_decrease(temp_time[end_end_pulse])(np.array([1.5,1.7,2])*1e-3,*fit[0])))
		ax[0,2].set_ylim(bottom=fit[0][0]-2,top=np.max(ensamble_peak_shape[end_end_pulse:post_pulse])+10)
		ax[0,2].axhline(y=fit[0][0],linestyle='--',color='m',linewidth=1)#.plot([-5,20],[fit[0][0]]*2,'--m',linewidth=1)
		ax[0,3].axhline(y=fit[0][0],linestyle='--',color='m',linewidth=1)#.plot([-5,20],[fit[0][0]]*2,'--m',linewidth=1)

		full_saved_file_dict['average_pulse_peak_data']['energy_fit'] = dict([])
		full_saved_file_dict['average_pulse_peak_data']['energy_fit']['pulse_energy_density'] = pulse_energy_density.nominal_value
		full_saved_file_dict['average_pulse_peak_data']['energy_fit']['pulse_energy_density_sigma'] =pulse_energy_density.std_dev
		full_saved_file_dict['average_pulse_peak_data']['energy_fit']['pulse_duration_ms'] = pulse_duration_ms.nominal_value
		full_saved_file_dict['average_pulse_peak_data']['energy_fit']['pulse_duration_ms_sigma'] =pulse_duration_ms.std_dev
		full_saved_file_dict['average_pulse_peak_data']['energy_fit']['pulse_energy'] = pulse_energy.nominal_value
		full_saved_file_dict['average_pulse_peak_data']['energy_fit']['pulse_energy_sigma'] =pulse_energy.std_dev
		full_saved_file_dict['average_pulse_peak_data']['energy_fit']['pulse_t0_semi_inf'] = 1000*fit_wit_errors[3].nominal_value
		full_saved_file_dict['average_pulse_peak_data']['energy_fit']['pulse_t0_semi_inf_sigma'] =1000*fit_wit_errors[3].std_dev
		full_saved_file_dict['average_pulse_peak_data']['energy_fit']['T_SS'] = fit_wit_errors[0].nominal_value
		full_saved_file_dict['average_pulse_peak_data']['energy_fit']['T_SS_sigma'] = fit_wit_errors[0].std_dev

		# T0 = ensamble_peak_shape_full[:np.abs(peak_shape_time+2e-3).argmin()].mean()
		# T0_std = ensamble_peak_shape_full[:np.abs(peak_shape_time+2e-3).argmin()].std()
		# guess = [fit[0][0],fit[0][1],max(3e-4,fit[0][2]),peak_shape_time[start_pulse]]
		guess = [fit[0][1],0.5,peak_shape_time[start_pulse]]
		# guess = fit[0]
		# bds=[[T0-2*T0_std,1e-6,1e-4,peak_shape_time[start_pulse]*1.5],[T0+2*T0_std,np.inf,temp_time[end_end_pulse],temp_time[end_end_pulse]]]
		bds=[[1e-6,1e-2,peak_shape_time[start_pulse]*1.5],[np.inf,1,temp_time[end_end_pulse]]]
		# bds=[[T0-2*T0_std,1e-6,1e-4,-np.inf],[T0+2*T0_std,np.inf,np.inf,np.inf]]
		fit = curve_fit(semi_infinite_sink_full_decrease_forced_T_SS(temp_time[end_end_pulse],SS_ensamble), np.float64(temp_time[end_end_pulse:post_pulse]), np.float64(ensamble_peak_shape[end_end_pulse:post_pulse]), guess,x_scale=[1e5,1e-3,1e-4],absolute_sigma=True,bounds=bds,maxfev=int(1e5),ftol=1e-15,xtol=1e-15)
		valid = np.logical_not(np.isnan(semi_infinite_sink_full_decrease_forced_T_SS(temp_time[end_end_pulse],SS_ensamble)(temp_time[end_end_pulse:post_pulse],*fit[0])))
		R2 = 1-np.sum(((ensamble_peak_shape[end_end_pulse:post_pulse]-semi_infinite_sink_full_decrease_forced_T_SS(temp_time[end_end_pulse],SS_ensamble)(temp_time[end_end_pulse:post_pulse],*fit[0]))**2)[valid])/np.sum(((ensamble_peak_shape[end_end_pulse:post_pulse]-np.mean(ensamble_peak_shape[end_end_pulse:post_pulse]))**2)[valid])
		# if R2<0.5:	# I already verified that in order to accurately measure the energy the baseline temperature must be free, so this step is unnecessary, I can do it from the start
		# 	bds=[[T0-10*T0_std,1e-6,1e-4,peak_shape_time[start_pulse]*1.5],[T0+10*T0_std,np.inf,temp_time[end_end_pulse],temp_time[end_end_pulse]]]
		# 	fit = curve_fit(semi_infinite_sink_full_decrease, temp_time[end_end_pulse:post_pulse], ensamble_peak_shape[end_end_pulse:post_pulse], guess,x_scale=np.abs(guess),bounds=bds,maxfev=int(1e4),xtol=1e-15,ftol=1e-14)
		# 	valid = np.logical_not(np.isnan(semi_infinite_sink_full_decrease(temp_time[end_end_pulse:post_pulse],*fit[0])))
		# 	R2 = 1-np.sum(((ensamble_peak_shape[end_end_pulse:post_pulse]-semi_infinite_sink_full_decrease(temp_time[end_end_pulse:post_pulse],*fit[0]))**2)[valid])/np.sum(((ensamble_peak_shape[end_end_pulse:post_pulse]-np.mean(ensamble_peak_shape[end_end_pulse:post_pulse]))**2)[valid])
		# fit_wit_errors = fit[0]
		# fit_wit_errors[2] += -peak_shape_time[start_pulse]
		fit_wit_errors = correlated_values(fit[0],fit[1])
		pulse_energy_density = (fit_wit_errors[0]+ufloat(0,0.1*fit_wit_errors[0].nominal_value))	# this to add the uncertainly coming from the method itself
		pulse_energy = pulse_energy_density*area_of_interest_sigma#*1/1.08	# 1/1.08 correction from the forward modeling check # 10/02/2021 now it doesn't seem to be needed
		pulse_duration_ms = 1e3*(temp_time[end_end_pulse]-fit_wit_errors[2])*fit_wit_errors[1]
		ax[0,2].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]>pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_decrease_forced_T_SS(temp_time[end_end_pulse],SS_ensamble)(peak_shape_time[peak_shape_time-fit[0][2]>pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2,label='semi-infinite solid\nfixed T SS=%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(SS_ensamble,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_duration_ms.nominal_value,pulse_duration_ms.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
		ax[0,3].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]>pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_decrease_forced_T_SS(temp_time[end_end_pulse],SS_ensamble)(peak_shape_time[peak_shape_time-fit[0][2]>pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2,label='semi-infinite solid\nfixed T SS=%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(SS_ensamble,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_duration_ms.nominal_value,pulse_duration_ms.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
		ax[0,2].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]<=pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_increase_forced_T_SS(temp_time[end_end_pulse],SS_ensamble)(peak_shape_time[peak_shape_time-fit[0][2]<=pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2)
		ax[0,3].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]<=pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_increase_forced_T_SS(temp_time[end_end_pulse],SS_ensamble)(peak_shape_time[peak_shape_time-fit[0][2]<=pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2)
		ax[0,3].axvline(x=1000*fit[0][2],linestyle='--',color='c',linewidth=1)
		ax[0,0].axvline(x=1000*fit[0][2],linestyle='--',color='c',linewidth=1)
		ax[0,2].axhline(y=SS_ensamble,linestyle='--',color='r',linewidth=1)#.plot([-5,20],[SS_ensamble]*2,'--r',linewidth=1)
		ax[0,3].axhline(y=SS_ensamble,linestyle='--',color='r',linewidth=1)#.plot([-5,20],[SS_ensamble]*2,'--r',linewidth=1)

		# ax2[0,1].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]>pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_decrease_forced_T_SS(temp_time[end_end_pulse],SS_ensamble)(peak_shape_time[peak_shape_time-fit[0][2]>pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2,label='semi-infinite solid\nfixed T SS=%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(SS_ensamble,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_duration_ms.nominal_value,pulse_duration_ms.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
		# ax2[0,1].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]<=pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_increase_forced_T_SS(temp_time[end_end_pulse],SS_ensamble)(peak_shape_time[peak_shape_time-fit[0][2]<=pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2)
		# ax2[0,1].axvline(x=1000*fit[0][2],linestyle='--',color='c',linewidth=1)
		# ax2[0,0].axvline(x=1000*fit[0][2],linestyle='--',color='c',linewidth=1)

		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_T_SS'] = dict([])
		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_T_SS']['pulse_energy_density'] = pulse_energy_density.nominal_value
		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_T_SS']['pulse_energy_density_sigma'] =pulse_energy_density.std_dev
		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_T_SS']['pulse_duration_ms'] = pulse_duration_ms.nominal_value
		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_T_SS']['pulse_duration_ms_sigma'] =pulse_duration_ms.std_dev
		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_T_SS']['pulse_energy'] = pulse_energy.nominal_value
		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_T_SS']['pulse_energy_sigma'] =pulse_energy.std_dev
		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_T_SS']['pulse_t0_semi_inf'] = 1000*fit_wit_errors[2].nominal_value
		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_T_SS']['pulse_t0_semi_inf_sigma'] =1000*fit_wit_errors[2].std_dev
		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_T_SS']['DT_pulse_time_scaled'] = np.mean(ensamble_peak_shape[np.abs(temp_time - fit[0][2] - 2e-3).argmin():np.abs(temp_time - fit[0][2] - 3e-3).argmin()])-SS_ensamble
		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_T_SS']['DT_pulse_time_scaled_sigma'] = np.std(ensamble_peak_shape[np.abs(temp_time - fit[0][2] - 2e-3).argmin():np.abs(temp_time - fit[0][2] - 3e-3).argmin()])

		# df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
		# # df_log.loc[j,['pulse_en [J]']] = min_power
		# df_log.loc[j,['pulse_en_semi_inf [J]']] = pulse_energy.nominal_value
		# df_log.loc[j,['pulse_en_semi_inf_sigma [J]']] = pulse_energy.std_dev
		# df_log.loc[j,['area_of_interest [m2]']] = area_of_interest_sigma.nominal_value
		# # df_log.loc[j,['DT_pulse']] = (ensamble_peak_shape[np.abs(temp_time-1.5e-3).argmin()]-SS_ensamble)
		# df_log.loc[j,['DT_pulse']] = (np.mean(ensamble_peak_shape[np.abs(temp_time-2e-3).argmin():np.abs(temp_time-3e-3).argmin()])-SS_ensamble)
		# df_log.loc[j,['DT_pulse_late']] = (np.mean(ensamble_peak_shape[np.abs(temp_time-10e-3).argmin():np.abs(temp_time-11e-3).argmin()])-SS_ensamble)
		# df_log.loc[j,['SS_Energy [J]']] = SS_Energy
		# df_log.to_csv(path_or_buf='/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv')

		guess = [SS_ensamble,fit[0][0],peak_shape_time[start_pulse]]
		bds=[[SS_ensamble-2*T0_std,1e-6,peak_shape_time[start_pulse]*1.5],[SS_ensamble+2*T0_std,np.inf,temp_time[end_end_pulse]-TS_pulse_duration_at_max_power]]
		fit = curve_fit(semi_infinite_sink_full_decrease_forced_duration(temp_time[end_end_pulse],TS_pulse_duration_at_max_power), np.float64(temp_time[end_end_pulse:post_pulse]), np.float64(ensamble_peak_shape[end_end_pulse:post_pulse]),guess,bounds=bds,x_scale=[100,1e5,1e-4],absolute_sigma=True,maxfev=int(1e4),ftol=1e-15,xtol=1e-15)
		valid = np.logical_not(np.isnan(semi_infinite_sink_full_decrease_forced_duration(temp_time[end_end_pulse],TS_pulse_duration_at_max_power)(temp_time[end_end_pulse:post_pulse],*fit[0])))
		R2 = 1-np.sum(((ensamble_peak_shape[end_end_pulse:post_pulse]-semi_infinite_sink_full_decrease_forced_duration(temp_time[end_end_pulse],TS_pulse_duration_at_max_power)(temp_time[end_end_pulse:post_pulse],*fit[0]))**2)[valid])/np.sum(((ensamble_peak_shape[end_end_pulse:post_pulse]-np.mean(ensamble_peak_shape[end_end_pulse:post_pulse]))**2)[valid])
		fit_wit_errors = correlated_values(fit[0],fit[1])
		pulse_energy_density = (fit_wit_errors[1]+ufloat(0,0.1*fit_wit_errors[1].nominal_value))	# this to add the uncertainly coming from the method itself
		pulse_energy = pulse_energy_density*area_of_interest_sigma#*1/1.08	# 1/1.08 correction from the forward modeling check # 10/02/2021 now it doesn't seem to be needed
		# pulse_duration_ms = 1e3*(temp_time[end_end_pulse]-fit_wit_errors[3])*fit_wit_errors[2]
		ax[0,2].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]>TS_pulse_duration_at_max_power]),semi_infinite_sink_full_decrease_forced_duration(temp_time[end_end_pulse],TS_pulse_duration_at_max_power)(peak_shape_time[peak_shape_time-fit[0][2]>TS_pulse_duration_at_max_power],*fit[0]),'--y',label='semi-infinite solid duration fixed %.3gms\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(TS_pulse_duration_at_max_power*1e3,fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
		ax[0,3].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]>TS_pulse_duration_at_max_power]),semi_infinite_sink_full_decrease_forced_duration(temp_time[end_end_pulse],TS_pulse_duration_at_max_power)(peak_shape_time[peak_shape_time-fit[0][2]>TS_pulse_duration_at_max_power],*fit[0]),'--y',label='semi-infinite solid duration fixed %.3gms\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(TS_pulse_duration_at_max_power*1e3,fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
		ax[0,2].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]<=TS_pulse_duration_at_max_power]),semi_infinite_sink_full_increase_forced_duration(temp_time[end_end_pulse],TS_pulse_duration_at_max_power)(peak_shape_time[peak_shape_time-fit[0][2]<=TS_pulse_duration_at_max_power],*fit[0]),'--y')
		ax[0,3].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]<=TS_pulse_duration_at_max_power]),semi_infinite_sink_full_increase_forced_duration(temp_time[end_end_pulse],TS_pulse_duration_at_max_power)(peak_shape_time[peak_shape_time-fit[0][2]<=TS_pulse_duration_at_max_power],*fit[0]),'--y')
		# ax[0,3].plot([1000*fit[0][2]]*2,[ensamble_peak_shape_full.min(),ensamble_peak_shape_full.max()],'--y')
		# ax[0,0].plot([1000*fit[0][2]]*2,[0,1e6],'--y')
		ax[0,3].axvline(x=1000*temp_time[np.abs(temp_time - fit[0][2] - 2e-3).argmin()],linestyle='--',color='m',label='dT(mean peak + t0 + 2-3ms)=%.3g°C' %(np.mean(ensamble_peak_shape[np.abs(temp_time - fit[0][2] - 2e-3).argmin():np.abs(temp_time - fit[0][2] - 3e-3).argmin()])-SS_ensamble))
		ax[0,3].axvline(x=1000*temp_time[np.abs(temp_time - fit[0][2] - 3e-3).argmin()],linestyle='--',color='m')

		ax2[0,1].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]>TS_pulse_duration_at_max_power]),semi_infinite_sink_full_decrease_forced_duration(temp_time[end_end_pulse],TS_pulse_duration_at_max_power)(peak_shape_time[peak_shape_time-fit[0][2]>TS_pulse_duration_at_max_power],*fit[0]),'--y',label='semi-infinite solid duration fixed %.3gms\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(TS_pulse_duration_at_max_power*1e3,fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
		ax2[0,1].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]<=TS_pulse_duration_at_max_power]),semi_infinite_sink_full_increase_forced_duration(temp_time[end_end_pulse],TS_pulse_duration_at_max_power)(peak_shape_time[peak_shape_time-fit[0][2]<=TS_pulse_duration_at_max_power],*fit[0]),'--y')
		ax2[0,1].axvline(x=1000*fit[0][2],linestyle='--',color='c',linewidth=1)
		ax2[0,0].axvline(x=1000*fit[0][2],linestyle='--',color='c',linewidth=1)

		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_duration'] = dict([])
		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_duration']['pulse_energy_density'] = pulse_energy_density.nominal_value
		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_duration']['pulse_energy_density_sigma'] =pulse_energy_density.std_dev
		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_duration']['pulse_t0_semi_inf'] = 1000*fit_wit_errors[2].nominal_value
		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_duration']['pulse_t0_semi_inf_sigma'] =1000*fit_wit_errors[2].std_dev
		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_duration']['pulse_energy'] = pulse_energy.nominal_value
		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_duration']['pulse_energy_sigma'] =pulse_energy.std_dev
		# full_saved_file_dict['average_pulse_peak_data']['energy_fit']['pulse_tau_semi_inf'] = pulse_duration_ms.nominal_value
		# full_saved_file_dict['average_pulse_peak_data']['energy_fit']['pulse_tau_semi_inf_sigma'] =pulse_duration_ms.std_dev
		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_duration']['DT_pulse_time_scaled'] = np.mean(ensamble_peak_shape[np.abs(temp_time - fit[0][2] - 2e-3).argmin():np.abs(temp_time - fit[0][2] - 3e-3).argmin()])-SS_ensamble
		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_duration']['DT_pulse_time_scaled_sigma'] = np.std(ensamble_peak_shape[np.abs(temp_time - fit[0][2] - 2e-3).argmin():np.abs(temp_time - fit[0][2] - 3e-3).argmin()])
		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_duration']['T_SS'] = fit_wit_errors[0].nominal_value
		full_saved_file_dict['average_pulse_peak_data']['energy_fit_fix_duration']['T_SS_sigma'] = fit_wit_errors[0].std_dev

		# df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
		# df_log.loc[j,['pulse_t0_semi_inf [ms]']] = 1000*fit[0][3]
		# df_log.loc[j,['pulse_tau_semi_inf [ms]']] = pulse_duration_ms.nominal_value
		# df_log.loc[j,['DT_pulse_time_scaled']] = (np.mean(ensamble_peak_shape[np.abs(temp_time - fit[0][3] - 2e-3).argmin():np.abs(temp_time - fit[0][3] - 3e-3).argmin()])-SS_ensamble)
		# df_log.to_csv(path_or_buf='/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv')
		# unsuccessful_fit = False


		if False:
			guess = [ensamble_peak_mean_shape[-SS_time:].mean(),ensamble_peak_mean_shape[np.abs(temp_time-1e-3).argmin()],3e-3]
			bds=[[20,0,1e-3],[np.inf,np.inf,1e-1]]
			fit = curve_fit(exponential_decay, temp_time[end_end_pulse:post_pulse], ensamble_peak_mean_shape[end_end_pulse:post_pulse], guess,bounds=bds,maxfev=int(1e4),xtol=1e-14,ftol=1e-14)
			min_target_temperature = exponential_decay(temp_time,*fit[0])
			if False:	# fixed area
				area_of_interest = 2*np.pi*(0.026/2)**2	# 26mm is the diameter of the standard magnum target
			sigmaSB=5.6704e-08 #[W/(m2 K4)]
			# target_effected_thickness = 0.7e-3	# 1mm thickness of the target (approx.)
			min_power = thermal_conductivity*(min_target_temperature-fit[0][0])/target_effected_thickness + emissivity_ensamble_peak_shape*sigmaSB*((ensamble_peak_mean_shape_full[peak_pulse:]+273.15)**4 - 300**4)
			min_power = area_of_interest*np.sum(min_power)/(10*frequency)
			max_power = heat_capacity*density*area_of_interest*target_effected_thickness*(fit[0][1])
			ax[1,2].plot(1000*temp_time,min_target_temperature,'--r',linewidth=2,label='T SS=%.3g°C\ndT=%.3g°C\ntau=%.3gms\nEnergy\nfrom area=%.3gJ\nfrom Tmax=%.3gJ' %(fit[0][0],fit[0][1],fit[0][2]*1000,min_power,max_power))
			ax[1,3].plot(1000*temp_time,min_target_temperature,'--r',linewidth=2)
			ax[1,2].set_ylim(bottom=fit[0][0]-2,top=10+min_target_temperature.max())

		T0 = ensamble_peak_mean_shape_full[:np.abs(peak_shape_time+2e-3).argmin()].mean()
		T0_std = max(2,ensamble_peak_mean_shape_full[:np.abs(peak_shape_time+2e-3).argmin()].std())
		# guess = [max(20,T0),1e5,1e-3]
		# bds=[[max(20,T0-2*T0_std),1e-6,1e-4],[max(25,T0+2*T0_std),np.inf,temp_time[end_end_pulse]]]
		# guess = [SS_ensamble_mean,1e5,1e-3]
		# bds=[[SS_ensamble_mean-1e-3,1e-6,1e-4],[SS_ensamble_mean+1e-3,np.inf,temp_time[end_end_pulse]]]
		guess = [SS_ensamble_mean,1e5,0.5,peak_shape_time[start_pulse]]
		bds=[[SS_ensamble_mean-2*T0_std,1e-6,1e-2,peak_shape_time[start_pulse]*1.5],[SS_ensamble_mean+2*T0_std,np.inf,1,temp_time[end_end_pulse]]]
		fit = curve_fit(semi_infinite_sink_full_decrease(temp_time[end_end_pulse]), np.float64(temp_time[end_end_pulse:post_pulse]), np.float64(ensamble_peak_mean_shape[end_end_pulse:post_pulse]), guess,sigma=np.float64(ensamble_peak_mean_shape_std[end_end_pulse:post_pulse]),x_scale=[100,1e5,1e-3,1e-4],absolute_sigma=True,bounds=bds,maxfev=int(1e5))
		valid = np.logical_not(np.isnan(semi_infinite_sink_full_decrease(temp_time[end_end_pulse])(temp_time[end_end_pulse:post_pulse],*fit[0])))
		R2 = 1-np.sum(((ensamble_peak_mean_shape[end_end_pulse:post_pulse]-semi_infinite_sink_full_decrease(temp_time[end_end_pulse])(temp_time[end_end_pulse:post_pulse],*fit[0]))**2)[valid])/np.sum(((ensamble_peak_mean_shape[end_end_pulse:post_pulse]-np.mean(ensamble_peak_mean_shape[end_end_pulse:post_pulse]))**2)[valid])
		fit_wit_errors = correlated_values(fit[0],fit[1])
		pulse_energy_density = (fit_wit_errors[1]+ufloat(0,0.1*fit_wit_errors[1].nominal_value))	# this to add the uncertainly coming from the method itself
		pulse_energy = pulse_energy_density*area_of_interest_sigma#*1/1.08	# 1/1.08 correction from the forward modeling check # 10/02/2021 now it doesn't seem to be needed
		pulse_duration_ms = 1e3*(temp_time[end_end_pulse]-fit_wit_errors[3])*fit_wit_errors[2]
		# ax[1,2].plot(1000*peak_shape_time,semi_infinite_sink_decrease(peak_shape_time,*fit[0]),'--m',linewidth=1,label='semi-infinite solid t0=peak fixed\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,fit_wit_errors[2].nominal_value*1e3,fit_wit_errors[2].std_dev*1000,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
		# ax[1,3].plot(1000*peak_shape_time,semi_infinite_sink_decrease(peak_shape_time,*fit[0]),'--m',linewidth=1,label='semi-infinite solid t0=peak fixed\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,fit_wit_errors[2].nominal_value*1e3,fit_wit_errors[2].std_dev*1000,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
		# ax[1,2].plot(1000*peak_shape_time[peak_shape_time<fit[0][2]],semi_infinite_sink_increase(peak_shape_time[peak_shape_time<fit[0][2]],*fit[0]),'--m',linewidth=1)
		# ax[1,3].plot(1000*peak_shape_time[peak_shape_time<fit[0][2]],semi_infinite_sink_increase(peak_shape_time[peak_shape_time<fit[0][2]],*fit[0]),'--m',linewidth=1)
		ax[1,2].plot(1000*(peak_shape_time[peak_shape_time-fit[0][3]>pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_decrease(temp_time[end_end_pulse])(peak_shape_time[peak_shape_time-fit[0][3]>pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--m',linewidth=1,label='semi-infinite solid\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_duration_ms.nominal_value,pulse_duration_ms.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
		ax[1,3].plot(1000*(peak_shape_time[peak_shape_time-fit[0][3]>pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_decrease(temp_time[end_end_pulse])(peak_shape_time[peak_shape_time-fit[0][3]>pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--m',linewidth=1,label='semi-infinite solid\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_duration_ms.nominal_value,pulse_duration_ms.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
		ax[1,2].plot(1000*(peak_shape_time[peak_shape_time-fit[0][3]<=pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_increase(temp_time[end_end_pulse])(peak_shape_time[peak_shape_time-fit[0][3]<=pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--m',linewidth=1)
		ax[1,3].plot(1000*(peak_shape_time[peak_shape_time-fit[0][3]<=pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_increase(temp_time[end_end_pulse])(peak_shape_time[peak_shape_time-fit[0][3]<=pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--m',linewidth=1)
		# ax[1,2].set_ylim(bottom=fit[0][0]-2,top=np.nanmax(10+semi_infinite_sink_full_increase(temp_time[end_end_pulse])(np.array([1.5,1.7,2])*1e-3,*fit[0])))
		ax[1,2].set_ylim(bottom=fit[0][0]-2,top=np.max(ensamble_peak_mean_shape[end_end_pulse:post_pulse])+10)
		ax[1,2].axhline(y=fit[0][0],linestyle='--',color='m',linewidth=1)#plot([-5,20],[fit[0][0]]*2,'--m',linewidth=1)
		ax[1,3].axhline(y=fit[0][0],linestyle='--',color='m',linewidth=1)#.plot([-5,20],[fit[0][0]]*2,'--m',linewidth=1)

		full_saved_file_dict['average_pulse_fitted_data']['energy_fit'] = dict([])
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit']['pulse_energy_density'] = pulse_energy_density.nominal_value
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit']['pulse_energy_density_sigma'] =pulse_energy_density.std_dev
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit']['pulse_duration_ms'] = pulse_duration_ms.nominal_value
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit']['pulse_duration_ms_sigma'] =pulse_duration_ms.std_dev
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit']['pulse_energy'] = pulse_energy.nominal_value
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit']['pulse_energy_sigma'] =pulse_energy.std_dev
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit']['pulse_t0_semi_inf'] = 1000*fit_wit_errors[3].nominal_value
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit']['pulse_t0_semi_inf_sigma'] =1000*fit_wit_errors[3].std_dev
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit']['T_SS'] = fit_wit_errors[0].nominal_value
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit']['T_SS_sigma'] = fit_wit_errors[0].std_dev

		# T0 = ensamble_peak_mean_shape_full[:np.abs(peak_shape_time+2e-3).argmin()].mean()
		# T0_std = ensamble_peak_mean_shape_full[:np.abs(peak_shape_time+2e-3).argmin()].std()
		guess = [fit[0][1],0.5,peak_shape_time[start_pulse]]
		# guess = fit[0]
		# bds=[[T0-2*T0_std,1e-6,1e-4,peak_shape_time[start_pulse]*1.5],[T0+2*T0_std,np.inf,temp_time[end_end_pulse],temp_time[end_end_pulse]]]
		bds=[[1e-6,1e-2,peak_shape_time[start_pulse]*1.5],[np.inf,1,temp_time[end_end_pulse]]]
		fit = curve_fit(semi_infinite_sink_full_decrease_forced_T_SS(temp_time[end_end_pulse],SS_ensamble_mean), np.float64(temp_time[end_end_pulse:post_pulse]), np.float64(ensamble_peak_mean_shape[end_end_pulse:post_pulse]), guess,sigma=np.float64(ensamble_peak_mean_shape_std[end_end_pulse:post_pulse]),x_scale=[1e5,1e-3,1e-4],absolute_sigma=True,bounds=bds,maxfev=int(1e4))
		valid = np.logical_not(np.isnan(semi_infinite_sink_full_decrease_forced_T_SS(temp_time[end_end_pulse],SS_ensamble_mean)(temp_time[end_end_pulse:post_pulse],*fit[0])))
		R2 = 1-np.sum(((ensamble_peak_mean_shape[end_end_pulse:post_pulse]-semi_infinite_sink_full_decrease_forced_T_SS(temp_time[end_end_pulse],SS_ensamble_mean)(temp_time[end_end_pulse:post_pulse],*fit[0]))**2)[valid])/np.sum(((ensamble_peak_mean_shape[end_end_pulse:post_pulse]-np.mean(ensamble_peak_mean_shape[end_end_pulse:post_pulse]))**2)[valid])
		fit_wit_errors = correlated_values(fit[0],fit[1])
		pulse_energy_density = (fit_wit_errors[0]+ufloat(0,0.1*fit_wit_errors[0].nominal_value))	# this to add the uncertainly coming from the method itself
		pulse_energy = pulse_energy_density*area_of_interest_sigma#*1/1.08	# 1/1.08 correction from the forward modeling check # 10/02/2021 now it doesn't seem to be needed
		pulse_duration_ms = 1e3*(temp_time[end_end_pulse]-fit_wit_errors[2])*fit_wit_errors[1]
		ax[1,2].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]>pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_decrease_forced_T_SS(temp_time[end_end_pulse],SS_ensamble_mean)(peak_shape_time[peak_shape_time-fit[0][2]>pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2,label='semi-infinite solid\nfixed T SS=%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(SS_ensamble_mean,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_duration_ms.nominal_value,pulse_duration_ms.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
		ax[1,3].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]>pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_decrease_forced_T_SS(temp_time[end_end_pulse],SS_ensamble_mean)(peak_shape_time[peak_shape_time-fit[0][2]>pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2,label='semi-infinite solid\nfixed T SS=%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(SS_ensamble_mean,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_duration_ms.nominal_value,pulse_duration_ms.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
		ax[1,2].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]<=pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_increase_forced_T_SS(temp_time[end_end_pulse],SS_ensamble_mean)(peak_shape_time[peak_shape_time-fit[0][2]<=pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2)
		ax[1,3].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]<=pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_increase_forced_T_SS(temp_time[end_end_pulse],SS_ensamble_mean)(peak_shape_time[peak_shape_time-fit[0][2]<=pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2)
		ax[1,3].axvline(x=1000*fit[0][2],linestyle='--',color='c',linewidth=1)
		ax[1,0].axvline(x=1000*fit[0][2],linestyle='--',color='c',linewidth=1)
		ax[1,2].axhline(y=SS_ensamble_mean,linestyle='--',color='r',linewidth=1)#.plot([-5,20],[SS_ensamble_mean]*2,'--r',linewidth=1)
		ax[1,3].axhline(y=SS_ensamble_mean,linestyle='--',color='r',linewidth=1)#.plot([-5,20],[SS_ensamble_mean]*2,'--r',linewidth=1)

		# ax2[1,1].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]>pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_decrease_forced_T_SS(temp_time[end_end_pulse],SS_ensamble_mean)(peak_shape_time[peak_shape_time-fit[0][2]>pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2,label='semi-infinite solid\nfixed T SS=%.3g°C\nEdens=%.3g+/-%.3gJ/m2\ntau=%.3g+/-%.3gms\npulse Energy=%.3g+/-%.3gJ\nR2=%.3g\nSS Energy=%.3gJ' %(SS_ensamble_mean,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_duration_ms.nominal_value,pulse_duration_ms.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2,SS_Energy))
		# ax2[1,1].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]<=pulse_duration_ms.nominal_value*1e-3]),semi_infinite_sink_full_increase_forced_T_SS(temp_time[end_end_pulse],SS_ensamble_mean)(peak_shape_time[peak_shape_time-fit[0][2]<=pulse_duration_ms.nominal_value*1e-3],*fit[0]),'--r',linewidth=2)
		# ax2[1,1].axvline(x=1000*fit[0][2],linestyle='--',color='c',linewidth=1)
		# ax2[1,0].axvline(x=1000*fit[0][2],linestyle='--',color='c',linewidth=1)

		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_T_SS'] = dict([])
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_T_SS']['pulse_energy_density'] = pulse_energy_density.nominal_value
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_T_SS']['pulse_energy_density_sigma'] =pulse_energy_density.std_dev
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_T_SS']['pulse_duration_ms'] = pulse_duration_ms.nominal_value
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_T_SS']['pulse_duration_ms_sigma'] =pulse_duration_ms.std_dev
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_T_SS']['pulse_energy'] = pulse_energy.nominal_value
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_T_SS']['pulse_energy_sigma'] =pulse_energy.std_dev
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_T_SS']['pulse_t0_semi_inf'] = 1000*fit_wit_errors[2].nominal_value
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_T_SS']['pulse_t0_semi_inf_sigma'] =1000*fit_wit_errors[2].std_dev
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_T_SS']['DT_pulse_time_scaled'] = np.mean(ensamble_peak_shape[np.abs(temp_time - fit[0][2] - 2e-3).argmin():np.abs(temp_time - fit[0][2] - 3e-3).argmin()])-SS_ensamble
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_T_SS']['DT_pulse_time_scaled_sigma'] = np.std(ensamble_peak_shape[np.abs(temp_time - fit[0][2] - 2e-3).argmin():np.abs(temp_time - fit[0][2] - 3e-3).argmin()])

		df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
		# df_log.loc[j,['pulse_en [J]']] = min_power
		df_log.loc[j,['pulse_en_semi_inf [J]']] = pulse_energy.nominal_value
		df_log.loc[j,['pulse_en_semi_inf_sigma [J]']] = pulse_energy.std_dev
		df_log.loc[j,['area_of_interest [m2]']] = area_of_interest_sigma.nominal_value
		# df_log.loc[j,['DT_pulse']] = (ensamble_peak_mean_shape[np.abs(temp_time-1.5e-3).argmin()]-SS_ensamble_mean)
		df_log.loc[j,['DT_pulse']] = (np.mean(ensamble_peak_mean_shape[np.abs(temp_time-2e-3).argmin():np.abs(temp_time-3e-3).argmin()])-SS_ensamble_mean)
		df_log.loc[j,['DT_pulse_late']] = (np.mean(ensamble_peak_mean_shape[np.abs(temp_time-10e-3).argmin():np.abs(temp_time-11e-3).argmin()])-SS_ensamble_mean)
		df_log.loc[j,['SS_Energy [J]']] = SS_Energy
		df_log.loc[j,['pulse_t0_semi_inf [ms]']] = 1000*fit[0][2]
		df_log.loc[j,['pulse_tau_semi_inf [ms]']] = pulse_duration_ms.nominal_value
		df_log.loc[j,['DT_pulse_time_scaled']] = (np.mean(ensamble_peak_mean_shape[np.abs(temp_time - fit[0][2] - 2e-3).argmin():np.abs(temp_time - fit[0][2] - 3e-3).argmin()])-SS_ensamble_mean)
		df_log.to_csv(path_or_buf='/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv')

		guess = [SS_ensamble_mean,fit[0][0],peak_shape_time[start_pulse]]
		bds=[[SS_ensamble_mean-2*T0_std,1e-6,peak_shape_time[start_pulse]*1.5],[SS_ensamble_mean+2*T0_std,np.inf,temp_time[end_end_pulse]-TS_pulse_duration_at_max_power]]
		fit = curve_fit(semi_infinite_sink_full_decrease_forced_duration(temp_time[end_end_pulse],TS_pulse_duration_at_max_power), np.float64(temp_time[end_end_pulse:post_pulse]), np.float64(ensamble_peak_mean_shape[end_end_pulse:post_pulse]),guess,sigma=np.float64(ensamble_peak_mean_shape_std[end_end_pulse:post_pulse]),x_scale=[100,1e5,1e-4],absolute_sigma=True,bounds=bds,maxfev=int(1e4))
		valid = np.logical_not(np.isnan(semi_infinite_sink_full_decrease_forced_duration(temp_time[end_end_pulse],TS_pulse_duration_at_max_power)(temp_time[end_end_pulse:post_pulse],*fit[0])))
		R2 = 1-np.sum(((ensamble_peak_mean_shape[end_end_pulse:post_pulse]-semi_infinite_sink_full_decrease_forced_duration(temp_time[end_end_pulse],TS_pulse_duration_at_max_power)(temp_time[end_end_pulse:post_pulse],*fit[0]))**2)[valid])/np.sum(((ensamble_peak_mean_shape[end_end_pulse:post_pulse]-np.mean(ensamble_peak_mean_shape[end_end_pulse:post_pulse]))**2)[valid])
		fit_wit_errors = correlated_values(fit[0],fit[1])
		pulse_energy_density = (fit_wit_errors[1]+ufloat(0,0.1*fit_wit_errors[1].nominal_value))	# this to add the uncertainly coming from the method itself
		pulse_energy = pulse_energy_density*area_of_interest_sigma#*1/1.08	# 1/1.08 correction from the forward modeling check # 10/02/2021 now it doesn't seem to be needed
		# pulse_duration_ms = 1e3*(temp_time[end_end_pulse]-fit_wit_errors[3])*fit_wit_errors[2]
		ax[1,2].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]>TS_pulse_duration_at_max_power]),semi_infinite_sink_full_decrease_forced_duration(temp_time[end_end_pulse],TS_pulse_duration_at_max_power)(peak_shape_time[peak_shape_time-fit[0][2]>TS_pulse_duration_at_max_power],*fit[0]),'--y',label='semi-infinite solid duration fixed %.3gms\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(TS_pulse_duration_at_max_power*1e3,fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
		ax[1,3].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]>TS_pulse_duration_at_max_power]),semi_infinite_sink_full_decrease_forced_duration(temp_time[end_end_pulse],TS_pulse_duration_at_max_power)(peak_shape_time[peak_shape_time-fit[0][2]>TS_pulse_duration_at_max_power],*fit[0]),'--y',label='semi-infinite solid duration fixed %.3gms\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(TS_pulse_duration_at_max_power*1e3,fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
		ax[1,2].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]<=TS_pulse_duration_at_max_power]),semi_infinite_sink_full_increase_forced_duration(temp_time[end_end_pulse],TS_pulse_duration_at_max_power)(peak_shape_time[peak_shape_time-fit[0][2]<=TS_pulse_duration_at_max_power],*fit[0]),'--y')
		ax[1,3].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]<=TS_pulse_duration_at_max_power]),semi_infinite_sink_full_increase_forced_duration(temp_time[end_end_pulse],TS_pulse_duration_at_max_power)(peak_shape_time[peak_shape_time-fit[0][2]<=TS_pulse_duration_at_max_power],*fit[0]),'--y')
		# ax[1,3].plot([1000*fit[0][2]]*2,[ensamble_peak_mean_shape_full.min(),ensamble_peak_mean_shape_full.max()],'--c')
		# ax[1,0].plot([1000*fit[0][2]]*2,[0,1e6],'--c')
		ax[1,3].axvline(x=1000*temp_time[np.abs(temp_time - fit[0][2] - 2e-3).argmin()],linestyle='--',color='m',label='dT(mean peak + t0 + 2-3ms)=%.3g°C' %(np.mean(ensamble_peak_mean_shape[np.abs(temp_time - fit[0][2] - 2e-3).argmin():np.abs(temp_time - fit[0][2] - 3e-3).argmin()])-SS_ensamble_mean))
		ax[1,3].axvline(x=1000*temp_time[np.abs(temp_time - fit[0][2] - 3e-3).argmin()],linestyle='--',color='m')

		ax2[1,1].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]>TS_pulse_duration_at_max_power]),semi_infinite_sink_full_decrease_forced_duration(temp_time[end_end_pulse],TS_pulse_duration_at_max_power)(peak_shape_time[peak_shape_time-fit[0][2]>TS_pulse_duration_at_max_power],*fit[0]),'--y',label='semi-infinite solid duration fixed %.3gms\nT SS=%.3g+/-%.3g°C\nEdens=%.3g+/-%.3gJ/m2\nEnergy=%.3g+/-%.3gJ\nR2=%.3g' %(TS_pulse_duration_at_max_power*1e3,fit_wit_errors[0].nominal_value,fit_wit_errors[0].std_dev,pulse_energy_density.nominal_value,pulse_energy_density.std_dev,pulse_energy.nominal_value,pulse_energy.std_dev,R2))
		ax2[1,1].plot(1000*(peak_shape_time[peak_shape_time-fit[0][2]<=TS_pulse_duration_at_max_power]),semi_infinite_sink_full_increase_forced_duration(temp_time[end_end_pulse],TS_pulse_duration_at_max_power)(peak_shape_time[peak_shape_time-fit[0][2]<=TS_pulse_duration_at_max_power],*fit[0]),'--y')
		ax2[1,1].axvline(x=1000*fit[0][2],linestyle='--',color='c',linewidth=1)
		ax2[1,0].axvline(x=1000*fit[0][2],linestyle='--',color='c',linewidth=1)

		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_duration'] = dict([])
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_duration']['pulse_energy_density'] = pulse_energy_density.nominal_value
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_duration']['pulse_energy_density_sigma'] =pulse_energy_density.std_dev
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_duration']['pulse_t0_semi_inf'] = 1000*fit_wit_errors[2].nominal_value
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_duration']['pulse_t0_semi_inf_sigma'] =1000*fit_wit_errors[2].std_dev
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_duration']['pulse_energy'] = pulse_energy.nominal_value
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_duration']['pulse_energy_sigma'] =pulse_energy.std_dev
		# full_saved_file_dict['average_pulse_fitted_data']['energy_fit']['pulse_tau_semi_inf'] = pulse_duration_ms.nominal_value
		# full_saved_file_dict['average_pulse_fitted_data']['energy_fit']['pulse_tau_semi_inf_sigma'] =pulse_duration_ms.std_dev
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_duration']['DT_pulse_time_scaled'] = np.mean(ensamble_peak_shape[np.abs(temp_time - fit[0][2] - 2e-3).argmin():np.abs(temp_time - fit[0][2] - 3e-3).argmin()])-SS_ensamble
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_duration']['DT_pulse_time_scaled_sigma'] = np.std(ensamble_peak_shape[np.abs(temp_time - fit[0][2] - 2e-3).argmin():np.abs(temp_time - fit[0][2] - 3e-3).argmin()])
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_duration']['T_SS'] = fit_wit_errors[0].nominal_value
		full_saved_file_dict['average_pulse_fitted_data']['energy_fit_fix_duration']['T_SS_sigma'] = fit_wit_errors[0].std_dev

		# df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
		# df_log.loc[j,['pulse_t0_semi_inf [ms]']] = 1000*fit[0][3]
		# df_log.loc[j,['pulse_tau_semi_inf [ms]']] = pulse_duration_ms.nominal_value
		# df_log.loc[j,['DT_pulse_time_scaled']] = (np.mean(ensamble_peak_mean_shape[np.abs(temp_time - fit[0][3] - 2e-3).argmin():np.abs(temp_time - fit[0][3] - 3e-3).argmin()])-SS_ensamble_mean)
		# df_log.to_csv(path_or_buf='/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv')
		unsuccessful_fit = False

	# except:
	# 	print('fit of the average pulse shape failed')

	# if unsuccessful_fit:
	# 	df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
	# 	# df_log.loc[j,['pulse_en [J]']] = min_power
	# 	df_log.loc[j,['pulse_en_semi_inf [J]']] = 0
	# 	df_log.loc[j,['pulse_en_semi_inf_sigma [J]']] = 0
	# 	df_log.loc[j,['area_of_interest [m2]']] = area_of_interest_sigma.nominal_value
	# 	df_log.loc[j,['DT_pulse']] = 0
	# 	df_log.to_csv(path_or_buf='/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv')


	ax[0,2].set_xlim(left=0,right=15)
	# ax[plot_index,0].set_ylim(bottom=ensamble_peak_shape.min(),top=1.5*ensamble_peak_shape.max())
	ax[0,2].legend(loc='best', fontsize='x-small')
	ax[0,2].grid()
	ax[0,2].set_xlabel('time [ms]')
	ax[0,2].set_ylabel('temperature [°C]')
	ax[0,2].set_title('large area max full temp')
	ax[1,2].set_xlim(left=0,right=15)
	# ax[plot_index,0].set_ylim(bottom=ensamble_peak_shape.min(),top=1.5*ensamble_peak_shape.max())
	ax[1,2].legend(loc='best', fontsize='x-small')
	ax[1,2].grid()
	ax[1,2].set_xlabel('time [ms]')
	ax[1,2].set_ylabel('temperature [°C]')
	# ax[plot_index,1].set_ylim(top=2*min_target_temperature.max())
	# ax[plot_index,1].set_yscale('log')
	ax[1,2].set_title('fitted peak temp')

	ax[0,3].set_xlim(left=-2,right=15)
	ax[0,3].legend(loc='best', fontsize='x-small')
	ax[0,3].grid()
	ax[0,3].set_xlabel('time [ms]')
	ax[0,3].set_ylabel('temperature [°C]')
	ax[0,3].set_title('large area max temp\narea=%.3g+/-%.3gm2' %(area_of_interest_sigma.nominal_value,area_of_interest_sigma.std_dev))
	ax[1,3].set_xlim(left=-2,right=15)
	ax[1,3].legend(loc='best', fontsize='x-small')
	ax[1,3].grid()
	ax[1,3].set_xlabel('time [ms]')
	ax[1,3].set_ylabel('temperature [°C]')
	ax[1,3].set_title('fitted peak full temp')
	figure_index+=1
	fig.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	# fig.close()
	ax2[0,1].set_xlim(left=-2,right=15)
	ax2[0,1].legend(loc='best', fontsize='small')
	ax2[0,1].grid()
	ax2[0,1].set_xlabel('time [ms]')
	ax2[0,1].set_ylabel('temperature [°C]')
	ax2[0,1].set_title('large area max full temp\narea=%.3g+/-%.3gm2' %(area_of_interest_sigma.nominal_value,area_of_interest_sigma.std_dev))
	ax2[1,1].set_xlim(left=-2,right=15)
	ax2[1,1].legend(loc='best', fontsize='small')
	ax2[1,1].grid()
	ax2[1,1].set_xlabel('time [ms]')
	ax2[1,1].set_ylabel('temperature [°C]')
	ax2[1,1].set_title('large area max fitted peak temp\narea=%.3g+/-%.3gm2' %(area_of_interest_sigma.nominal_value,area_of_interest_sigma.std_dev))
	figure_index+=1
	fig2.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close('all')

	color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise', 'teal', 'olive']
	plt.figure(figsize=(20, 10))
	plt.plot(np.arange(len(selected_1_mean))/frequency,selected_1_mean,color=color[0],label='centre '+str([IR_shape[0],IR_shape[1]])+' r='+str(IR_shape[2]))
	# plt.plot(np.arange(len(selected_1_mean))/frequency,selected_2_mean,label='mean, centre '+str([IR_shape[0],IR_shape[1]])+' r='+str(IR_shape[3]))
	# plt.plot(np.arange(len(selected_1_mean))/frequency,selected_2_max,'+',label='max, centre '+str([IR_shape[0],IR_shape[1]])+' r='+str(IR_shape[3]))
	plt.plot([0,len(selected_1_mean)/frequency],[max_temp_before_peaks,max_temp_before_peaks],'--',color=color[1],label='max temp before ELM pulse (%.3g°C)' %(max_temp_before_peaks))
	plt.plot([0,len(selected_1_mean)/frequency],[max_temp_before_peaks+ELM_temp_jump,max_temp_before_peaks+ELM_temp_jump],'--',color=color[2],label='max temp before ELM pulse + median ELM jump (%.3g°C)' %(ELM_temp_jump))
	plt.plot([0,len(selected_1_mean)/frequency],[max_temp_before_peaks+ELM_temp_jump_after,max_temp_before_peaks+ELM_temp_jump_after],'--',color=color[3],label='max temp before ELM pulse + median ELM jump after pulse(%.3g°C)' %(ELM_temp_jump_after))
	plt.plot([0,len(selected_1_mean)/frequency],[max_temp_before_peaks+low_end_interval_ELM_jumps,max_temp_before_peaks+low_end_interval_ELM_jumps],'-.',color=color[4],label='max temp before ELM pulse + ELM jump after pulse 80% interval')
	plt.plot([0,len(selected_1_mean)/frequency],[max_temp_before_peaks+up_end_interval_ELM_jumps,max_temp_before_peaks+up_end_interval_ELM_jumps],'-.',color=color[4])
	plt.plot((np.arange(len(selected_1_mean))/frequency)[very_good_pulses],temp_before_peaks,'^',color=color[5],label='temp before pulse')
	plt.plot((np.arange(len(selected_1_mean))/frequency)[very_good_pulses],np.max([selected_1_mean[very_good_pulses-1],selected_1_mean[very_good_pulses],selected_1_mean[very_good_pulses-1]],axis=0),'v',color=color[6],label='temp at pulse')
	plt.plot((np.arange(len(selected_1_mean))/frequency)[very_good_pulses],selected_1_mean[very_good_pulses+round(1.5/1000*frequency)],'o',color=color[7],label='temp 1.5ms after pulse')
	plt.plot((np.arange(len(selected_1_mean))/frequency)[strongest_very_good_pulse],np.max([selected_1_mean[strongest_very_good_pulse-1],selected_1_mean[strongest_very_good_pulse],selected_1_mean[strongest_very_good_pulse-1]]),'*',color=color[8],markersize=20,label='sample pulse for area check')
	# plt.plot([background_interval/frequency,background_interval/frequency],[np.min(selected_2_mean),np.max(selected_2_max)],'k--',label='limit for background in IR trace '+str(IR_reference))
	plt.grid()
	plt.title(pre_title+'Small area mean summary for '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	plt.xlabel('time [s]')
	plt.ylabel('Temperature [°C]')
	plt.legend(loc='best')
	figure_index+=1
	# plt.show()
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.plot(np.arange(len(selected_2_max))/frequency,selected_2_max,color=color[0],label='centre '+str([IR_shape[0],IR_shape[1]])+' r='+str(IR_shape[3]))
	# plt.plot(np.arange(len(selected_1_mean))/frequency,selected_2_mean,label='mean, centre '+str([IR_shape[0],IR_shape[1]])+' r='+str(IR_shape[3]))
	# plt.plot(np.arange(len(selected_1_mean))/frequency,selected_2_max,'+',label='max, centre '+str([IR_shape[0],IR_shape[1]])+' r='+str(IR_shape[3]))
	plt.plot([0,len(selected_2_max)/frequency],[mean_temp_before_peaks_large_area_max,mean_temp_before_peaks_large_area_max],'--',color=color[1],label='mean temp before ELM pulse (%.3g°C)' %(mean_temp_before_peaks_large_area_max))
	plt.plot([0,len(selected_2_max)/frequency],[mean_temp_before_peaks_large_area_max+ELM_temp_jump_large_area_max,mean_temp_before_peaks_large_area_max+ELM_temp_jump_large_area_max],'--',color=color[2],label='mean temp before ELM pulse + mean ELM jump (%.3g°C)' %(ELM_temp_jump_large_area_max))
	plt.plot([0,len(selected_2_max)/frequency],[mean_temp_before_peaks_large_area_max+ELM_temp_jump_after_large_area_max,mean_temp_before_peaks_large_area_max+ELM_temp_jump_after_large_area_max],'--',color=color[3],label='mean temp before ELM pulse + mean ELM jump after pulse (%.3g°C)' %(ELM_temp_jump_after_large_area_max))
	plt.plot([0,len(selected_2_max)/frequency],[mean_temp_before_peaks_large_area_max+low_end_interval_ELM_jumps_large_area_max,mean_temp_before_peaks_large_area_max+low_end_interval_ELM_jumps_large_area_max],'-.',color=color[4],label='mean temp before ELM pulse + ELM jump after pulse 80% interval')
	plt.plot([0,len(selected_2_max)/frequency],[mean_temp_before_peaks_large_area_max+up_end_interval_ELM_jumps_large_area_max,mean_temp_before_peaks_large_area_max+up_end_interval_ELM_jumps_large_area_max],'-.',color=color[4])
	plt.plot((np.arange(len(selected_2_max))/frequency)[very_good_pulses],temp_before_peaks_large_area_max,'^',color=color[5],label='temp before pulse')
	plt.plot((np.arange(len(selected_2_max))/frequency)[very_good_pulses],np.max([selected_2_max[very_good_pulses-1],selected_2_max[very_good_pulses],selected_2_max[very_good_pulses-1]],axis=0),'v',color=color[6],label='temp at pulse')
	plt.plot((np.arange(len(selected_2_max))/frequency)[very_good_pulses],selected_2_max[very_good_pulses+round(1.5/1000*frequency)],'o',color=color[7],label='temp 1.5ms after pulse')
	plt.plot((np.arange(len(selected_2_max))/frequency)[strongest_very_good_pulse],np.max([selected_2_max[strongest_very_good_pulse-1],selected_2_max[strongest_very_good_pulse],selected_2_max[strongest_very_good_pulse-1]]),'*',color=color[8],markersize=20,label='sample pulse for area check')
	# plt.plot([background_interval/frequency,background_interval/frequency],[np.min(selected_2_mean),np.max(selected_2_max)],'k--',label='limit for background in IR trace '+str(IR_reference))
	plt.grid()
	plt.title(pre_title+'Large area max summary for '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	plt.xlabel('time [s]')
	plt.ylabel('Temperature [°C]')
	plt.legend(loc='best')
	figure_index+=1
	# plt.show()
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	if isinstance(fname_current_trace,str):
		if os.path.exists(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/'+fname_current_trace+'.tsf'):

			if (len(bad_pulses_indexes)>2 and not bad_pulses_indexes[0]==-100):
				energy_per_pulse_good = []
				for i_energy_per_pulse_good in range(len(energy_per_pulse)):
					if not (i_energy_per_pulse_good+1 in bad_pulses_indexes):
						energy_per_pulse_good.append(energy_per_pulse[i_energy_per_pulse_good])

				select = np.array([not((value) in miss_pulses) for value in np.arange(1,number_of_pulses+1)])
				peaks = np.array(peaks_initial[peaks_initial>background_interval][np.array(score).argmax():][:number_of_pulses])
				peaks = peaks[select[:len(peaks)]]
				proms = get_proms(selected_2_max,peaks)[0]

				plt.figure(figsize=(20, 10))
				plt.plot(energy_per_pulse[select][:len(proms)], proms,'+b',label='prominence of the temperature peak')
				plt.plot([np.mean(energy_per_pulse_good)]*2,[np.min(proms),np.max(proms)],'k')
				plt.plot([np.mean(energy_per_pulse_good)+np.std(energy_per_pulse_good)]*2,[np.min(proms),np.max(proms)],'--k')
				plt.plot([np.mean(energy_per_pulse_good)-np.std(energy_per_pulse_good)]*2,[np.min(proms),np.max(proms)],'--k')
				y = np.array([y for _, y in sorted(zip(energy_per_pulse[select][:len(proms)], proms))])
				x = np.sort(energy_per_pulse[select][:len(proms)])
				if len(y)%10!=0:
					x = x[:(len(y)//10)*10]
					y = y[:(len(y)//10)*10]
				y = y.reshape((len(y)//10,10))
				x = x.reshape((len(x)//10,10))
				plt.plot(np.mean(x,axis=-1), np.mean(y,axis=-1),'--b')
				fit = scipy.stats.linregress(energy_per_pulse[select][:len(proms)],proms)
				plt.plot(np.sort(energy_per_pulse[select]), np.polyval(fit[:2],np.sort(energy_per_pulse[select])),':b',label='linear fit, coeffs=%.3g,%.3g, R2=%.3g' %(fit[0],fit[1],fit[2]**2))

				temp = []
				for index in range(10):
					temp.append(selected_2_mean[peaks-index-5])
				temp = np.mean(temp,axis=0)
				temp = np.max([selected_2_mean[np.array(peaks)-1],selected_2_mean[np.array(peaks)],selected_2_mean[np.array(peaks)+1]],axis=0) - temp
				plt.plot(energy_per_pulse[select][:len(temp)], temp,'or',label='maximum hight of the temperature peak-before')
				plt.plot([np.mean(energy_per_pulse_good)]*2,[np.min(temp),np.max(temp)],'k')
				plt.plot([np.mean(energy_per_pulse_good)+np.std(energy_per_pulse_good)]*2,[np.min(temp),np.max(temp)],'--k')
				plt.plot([np.mean(energy_per_pulse_good)-np.std(energy_per_pulse_good)]*2,[np.min(temp),np.max(temp)],'--k')
				y = np.array([y for _, y in sorted(zip(energy_per_pulse[select][:len(temp)], temp))])
				x = np.sort(energy_per_pulse[select][:len(proms)])
				if len(y)%10!=0:
					x = x[:(len(y)//10)*10]
					y = y[:(len(y)//10)*10]
				y = y.reshape((len(y)//10,10))
				x = x.reshape((len(x)//10,10))
				plt.plot(np.mean(x,axis=-1), np.mean(y,axis=-1),'--r')
				fit = scipy.stats.linregress(energy_per_pulse[select][:len(temp)],temp)
				plt.plot(np.sort(energy_per_pulse[select]), np.polyval(fit[:2],np.sort(energy_per_pulse[select])),':r',label='linear fit, coeffs=%.3g,%.3g, R2=%.3g' %(fit[0],fit[1],fit[2]**2))

				if peaks[-1]+round(1.5/1000*frequency)>len(selected_2_mean):
					peaks = peaks[:-1]
					select[-1] = False
				temp = []
				for index in range(10):
					temp.append(selected_2_mean[peaks-index-5])
				temp = np.mean(temp,axis=0)
				temp = selected_2_mean[peaks+round(1.5/1000*frequency)] - temp
				plt.plot(energy_per_pulse[select][:len(peaks)], temp,'xg',label='temperature after 1.5ms of peak-before')
				plt.plot([np.mean(energy_per_pulse_good)]*2,[np.min(temp),np.max(temp)],'k')
				plt.plot([np.mean(energy_per_pulse_good)+np.std(energy_per_pulse_good)]*2,[np.min(temp),np.max(temp)],'--k')
				plt.plot([np.mean(energy_per_pulse_good)-np.std(energy_per_pulse_good)]*2,[np.min(temp),np.max(temp)],'--k')
				y = np.array([y for _, y in sorted(zip(energy_per_pulse[select][:len(temp)], temp))])
				x = np.sort(energy_per_pulse[select][:len(proms)])
				if len(y)%10!=0:
					x = x[:(len(y)//10)*10]
					y = y[:(len(y)//10)*10]
				y = y.reshape((len(y)//10,10))
				x = x.reshape((len(x)//10,10))
				plt.plot(np.mean(x,axis=-1), np.mean(y,axis=-1),'--g')
				fit = scipy.stats.linregress(energy_per_pulse[select][:len(temp)],temp)
				plt.plot(np.sort(energy_per_pulse[select]), np.polyval(fit[:2],np.sort(energy_per_pulse[select])),':g',label='linear fit, coeffs=%.3g,%.3g, R2=%.3g' %(fit[0],fit[1],fit[2]**2))

				plt.legend(loc='best')
				plt.grid()
				plt.title(pre_title+'Comparison of effect on the target for different power upstream (using large area mean) for '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms\n good pulses energy %.3g std %.3g' %(frequency,int_time*1000,np.mean(energy_per_pulse_good),np.std(energy_per_pulse_good)))
				plt.xlabel('Power from the source [J]')
				plt.ylabel('Temperature increase at target [°C]')
				figure_index+=1
				plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
				plt.close()

			else:
				energy_per_pulse_good = []
				for i_energy_per_pulse_good in range(len(energy_per_pulse)):
					if not (i_energy_per_pulse_good+1 in bad_pulses_indexes):
						energy_per_pulse_good.append(energy_per_pulse[i_energy_per_pulse_good])

				select = np.array([not((value) in miss_pulses) for value in np.arange(1,number_of_pulses+1)])

			peaks = np.array(peaks_initial[peaks_initial>background_interval][np.array(score).argmax():][:number_of_pulses])
			peaks = peaks[select[:len(peaks)]]
			proms = get_proms(selected_2_max,peaks)[0]

			how_much_to_shift = (np.abs(energy_per_pulse-np.max(energy_per_pulse[[np.logical_and(value not in np.array(bad_pulses_indexes)-1,value<28*1.5) for value in np.arange(len(energy_per_pulse))]])).argmin())%28-5
			# peaks_initial = find_peaks(selected_2_mean,distance=incremental_step*frequency*0.9)[0]
			peaks = np.array(peaks_initial[peaks_initial>background_interval][np.array(score).argmax():][:number_of_pulses])
			# proms = get_proms(selected_2_mean,peaks)[0]
			if False:	# this was only the peak temperature, modified to be temp after - temp before
				temp = np.max([selected_2_mean[np.array(peaks)-1],selected_2_mean[np.array(peaks)],selected_2_mean[np.array(peaks)+1]],axis=0)
			else:
				if peaks[-1]+round(1.5/1000*frequency)>len(selected_2_mean):
					peaks = peaks[:-1]
					select[-1] = False
				temp = []
				for index in range(10):
					temp.append(selected_2_mean[peaks-index-5])
				temp = np.mean(temp,axis=0)
				temp = selected_2_mean[peaks+round(1.5/1000*frequency)] - temp

			energy_per_pulse_shifted = energy_per_pulse[how_much_to_shift:].tolist() + energy_per_pulse[:how_much_to_shift].tolist()
			temp_shifted = temp[how_much_to_shift:].tolist() + temp[:how_much_to_shift].tolist()
			time_shifted = (np.arange(len(selected_2_mean))/frequency)[peaks][how_much_to_shift:].tolist() + (np.arange(len(selected_2_mean))/frequency)[peaks][:how_much_to_shift].tolist()
			bad_pulses_indexes_shifted =np.array(bad_pulses_indexes)
			group_being_analysed = 0
			while group_being_analysed<100:
				select = np.arange(len(energy_per_pulse_shifted))/29
				select = np.logical_and(select<(group_being_analysed+1),select>=(group_being_analysed))
				if np.sum(select)==0:
					break
				bad_pulses_in_the_interval = [value in bad_pulses_indexes_shifted for value in np.arange(29*group_being_analysed,28*(group_being_analysed+1))]
				if np.sum(bad_pulses_in_the_interval)%2==0:
					energy_per_pulse_shifted = energy_per_pulse_shifted[:(group_being_analysed+1)*29] + [0] + energy_per_pulse_shifted[(group_being_analysed+1)*29:]
					temp_shifted = temp_shifted[:(group_being_analysed+1)*29] + [0] + temp_shifted[(group_being_analysed+1)*29:]
					time_shifted = time_shifted[:(group_being_analysed+1)*29] + [0] + time_shifted[(group_being_analysed+1)*29:]
					bad_pulses_indexes_shifted[bad_pulses_indexes_shifted>((group_being_analysed+1)*28)]+=1
				group_being_analysed+=1

			plt.figure(figsize=(20, 10))
			cmap = plt.cm.rainbow
			norm=plt.Normalize(vmin=0, vmax=29-1)
			for index in range(29):
				plt.plot(energy_per_pulse_shifted[index::29][:min(len(energy_per_pulse_shifted[index::29]),len(time_shifted[index::29]))], temp_shifted[index::29][:min(len(energy_per_pulse_shifted[index::29]),len(time_shifted[index::29]))],'o',color=cmap(norm(index)))
			# plt.plot(energy_per_pulse,proms,'+')
			plt.grid()
			plt.title(pre_title+'Comparison of effect on the target for different power upstream\nsame point color should be the same capacitor (using large area mean)\nfor '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms\n good pulses energy %.3g std %.3g' %(frequency,int_time*1000,np.mean(energy_per_pulse_good),np.std(energy_per_pulse_good)))
			plt.xlabel('Pulse energy from the source [J]')
			plt.ylabel('Temperature increase at target [°C]')
			plt.ylim(bottom=np.min(np.array(temp_shifted)[np.array(temp_shifted)>0]))
			figure_index+=1
			plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
			plt.close()

			plt.figure(figsize=(20, 10))
			cmap = plt.cm.rainbow
			norm=plt.Normalize(vmin=0, vmax=29-1)
			for index in range(29):
				plt.plot(time_shifted[index::29][:min(len(energy_per_pulse_shifted[index::29]),len(time_shifted[index::29]))],energy_per_pulse_shifted[index::29][:min(len(energy_per_pulse_shifted[index::29]),len(time_shifted[index::29]))],'o',color=cmap(norm(index)))
			# plt.plot(energy_per_pulse,proms,'+')
			plt.grid()
			plt.plot([0,np.max(time_shifted)],[np.mean(energy_per_pulse_good)]*2,'--k')
			plt.plot([0,np.max(time_shifted)],[np.mean(energy_per_pulse_good)+np.std(energy_per_pulse_good)]*2,'--y')
			plt.plot([0,np.max(time_shifted)],[np.mean(energy_per_pulse_good)-np.std(energy_per_pulse_good)]*2,'--y')
			plt.title(pre_title+'Energy released at the source\nsame point color should be the same capacitor\nfor '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms\n good pulses energy %.3g std %.3g' %(frequency,int_time*1000,np.mean(energy_per_pulse_good),np.std(energy_per_pulse_good)))
			plt.xlabel('time [s]')
			plt.ylabel('Pulse energy from the source [J]')
			figure_index+=1
			plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
			plt.close()



	df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
	# df_log.loc[j,['DT_pulse','T_pre_pulse']] = (ELM_temp_jump_after,max_temp_before_peaks)
	df_log.loc[j,['T_pre_pulse']] = mean_temp_before_peaks_large_area_max
	df_log.loc[j,['DT_pulse_low','DT_pulse_high']] = (low_end_interval_ELM_jumps_large_area_max,up_end_interval_ELM_jumps_large_area_max)
	df_log.to_csv(path_or_buf='/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv')
	np.savez_compressed(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace,**full_saved_file_dict)

	print('DONE analysing item n '+str(j))
	# return 'success '+str(j)

	# del spectra,magnitude,freq,corrected_frames,corrected_spectra,corrected_magnitude,selected_x,selected_y,selected_1,selected_2,peaks,proms,


# pool = Pool(number_cpu_available)
# composed_array = [*pool.map(calc_stuff, enumerate(np.flip(all_j,axis=0)))]
# pool.close()
# pool.join()
# print(composed_array)
