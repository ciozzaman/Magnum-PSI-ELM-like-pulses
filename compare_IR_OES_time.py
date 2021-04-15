import numpy as np
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())
# import matplotlib.pyplot as plt
#import .functions
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
from functions.spectools import rotate,do_tilt, binData, get_angle, get_tilt,get_angle_2
from functions.Calibrate import do_waveL_Calib, do_Intensity_Calib
from functions.fabio_add import find_nearest_index,multi_gaussian,all_file_names,load_dark,find_index_of_file,get_metadata,movie_from_data,get_angle_no_lines,do_tilt_no_lines,four_point_transform,fix_minimum_signal,fix_minimum_signal2,get_bin_and_interv_no_lines,examine_current_trace,DOM52sec
from functions.GetSpectrumGeometry import getGeom
from functions.SpectralFit import doSpecFit_single_frame
from functions.GaussFitData import doLateralfit_time_tependent
import collections

import os,sys
from PIL import Image
import xarray as xr
import pandas as pd
import copy
import datetime as dt
from uncertainties.unumpy import exp,nominal_values,std_devs,erf
from uncertainties import ufloat,unumpy,correlated_values
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
figure_index=0


file_full_path = '/home/ffederic/work/Collaboratory/test/experimental_data/2019-07-02/20190702_3.csv'
multi_1 = pd.read_csv(file_full_path,index_col=False,sep='|')
index1 = list(multi_1.head(0))
for index in range(len(index1)//3):
	index1[index*3]=index1[index*3+1]
	index1[index*3+2]=index1[index*3+1]
multi_1 = pd.read_csv(file_full_path,index_col=False,header=1,sep='|')
index2 = list(multi_1.head(0))
index = [_+' '+__ for _,__ in zip(index1,index2)]
for i_text,text in enumerate(index):
	if (text.find('.'))!=-1:
		index[i_text] = text[:text.find('.')]
multi_1 = pd.read_csv(file_full_path,index_col=False,header=1,sep='|')
multi_1_array = np.array(multi_1).T
CB_busy = dict([('CapStSourceBusy_time_orig',multi_1_array[1*3].astype(np.float32))])
CB_busy['CapStSourceBusy_status']=multi_1_array[1*3+2].astype(np.bool)
CB_busy['CapStSourceBusy_status'] = CB_busy['CapStSourceBusy_status'][np.logical_not(np.isnan(CB_busy['CapStSourceBusy_time_orig']))]
CB_busy['CapStSourceBusy_time_orig'] = multi_1_array[1*3][np.isfinite(CB_busy['CapStSourceBusy_time_orig'])].astype(np.int)
CB_busy['CapStSourceBusy_time_orig'] = CB_busy['CapStSourceBusy_time_orig'][np.logical_not(np.isnan(CB_busy['CapStSourceBusy_time_orig']))]
CB_busy['CapStSourceBusy_time'] = np.array([DOM52sec(time) for time in CB_busy['CapStSourceBusy_time_orig']])*1e3 - (dt.datetime(2000, 1, 1)-dt.datetime(1970, 1, 1)).total_seconds()*1e3
CB_busy['CapStPlasmaEnable_time_orig']=multi_1_array[2*3].astype(np.float32)
CB_busy['CapStPlasmaEnable_status']=multi_1_array[2*3+2].astype(np.bool)
CB_busy['CapStPlasmaEnable_status'] = CB_busy['CapStPlasmaEnable_status'][np.logical_not(np.isnan(CB_busy['CapStPlasmaEnable_time_orig']))]
CB_busy['CapStPlasmaEnable_time_orig'] = multi_1_array[2*3][np.isfinite(CB_busy['CapStPlasmaEnable_time_orig'])].astype(np.int)
CB_busy['CapStPlasmaEnable_time_orig'] = CB_busy['CapStPlasmaEnable_time_orig'][np.logical_not(np.isnan(CB_busy['CapStPlasmaEnable_time_orig']))]
CB_busy['CapStPlasmaEnable_time'] = np.array([DOM52sec(time) for time in CB_busy['CapStPlasmaEnable_time_orig']])*1e3 - (dt.datetime(2000, 1, 1)-dt.datetime(1970, 1, 1)).total_seconds()*1e3


global_dictionary = dict([])
print_1 = False
print_2 = True
print_3 = True

merge_id_all = [85,90,91,92,86,87,88,89,95]
all_reference_time = []
all_J_ext = []
all_time_ref = []
global_dictionary['merge_ID_target']=[]
global_dictionary['j']=[]
global_dictionary['pressure']=[]
global_dictionary['fit_OES_IR_R2']=[]
global_dictionary['fit_OES_IR']=[]
global_dictionary['elapsed_time_offset']=[]
global_dictionary['elapsed_time_offset2']=[]
global_dictionary['PVCAM_TimeStampBOF_offset']=[]
global_dictionary['PVCAM_TimeStampBOF_offset2']=[]
global_dictionary['time_of_pulses_DCsource_0']=[]
global_dictionary['time_of_pulses_IR_0']=[]
global_dictionary['IR_CB_time_difference1_record'] = []
global_dictionary['IR_CB_time_difference1_pulse'] = []
global_dictionary['IR_CB_time_difference2_record'] = []
global_dictionary['IR_CB_time_difference2_pulse'] = []
global_dictionary['timestamp_OES_0']=[]
global_dictionary['peaks_timestamp_plus_IR_0']=[]
global_dictionary['time_difference_0'] = []
global_dictionary['fit_SOURCE_IR_absolute'] = []
global_dictionary['fit_SOURCE_IR_absolute_R2'] = []
global_dictionary['fit_SOURCE_IR_relative'] = []
global_dictionary['fit_SOURCE_IR_relative_R2'] = []
global_dictionary['IR_pulse_start'] = []
# merge_ID_target=95
for merge_ID_target in merge_id_all:
	all_j=find_index_of_file(merge_ID_target,df_settings,df_log,only_OES=True)
	for j in all_j:
		# j=all_j[0]
		(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
		(CB_to_OES_initial_delay,incremental_step,first_pulse_at_this_frame,bad_pulses_indexes,number_of_pulses,fname_current_trace) = df_log.loc[j,['CB_to_OES_initial_delay','incremental_step','first_pulse_at_this_frame','bad_pulses_indexes','number_of_pulses','current_trace_file']]
		(IR_trace,IR_reference,IR_shape,magnetic_field,target_OES_distance,target_chamber_pressure,capacitor_voltage) = df_log.loc[j,['IR_trace','IR_reference','IR_shape','B','T_axial','p_n [Pa]','Vc']]
		(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
		if not(isinstance(IR_reference,str)):
			continue
		global_dictionary['merge_ID_target'].append(merge_ID_target)
		global_dictionary['j'].append(j)
		global_dictionary['pressure'].append(target_chamber_pressure)
		sequence = int(sequence)
		first_pulse_at_this_frame = int(first_pulse_at_this_frame)
		number_of_pulses = int(number_of_pulses)
		if bad_pulses_indexes=='':
			bad_pulses_indexes=[0]
		elif (isinstance(bad_pulses_indexes, float) or isinstance(bad_pulses_indexes, int)):
			bad_pulses_indexes=[bad_pulses_indexes]
		else:
			bad_pulses_indexes = bad_pulses_indexes.replace(' ', '').split(',')
			bad_pulses_indexes = list(map(int, bad_pulses_indexes))

		if -100 in bad_pulses_indexes:	# "keyword" in case I do not want to consider any of the pulses
			bad_pulses_indexes = np.linspace(1,number_of_pulses,number_of_pulses).astype('int')

		type = '.txt'
		filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2g}".format(int(sequence)) + '/Untitled_' + str(int(untitled)) + '/Pos0', type)[0]
		(bof, eof, roi_lb, roi_tr, time_info, real_exposure_time, PixelType, Gain,Noise) = get_metadata(fdir, folder, sequence,untitled,filename_metadata)
		# all_timestamp_OES = time_info['bof_corrected_ms']
		# if merge_ID_target==89:
		# 	number_of_pulses=249
		all_timestamp_OES = cp.deepcopy(time_info['PVCAM_TimeStampBOF_ms']) + time_info['PM_Cam_StartTime_ms']
		# all_timestamp_OES = bof*1e-6
		# all_timestamp_OES = cp.deepcopy(time_info['TimeStampMsec'])
		all_elapsed_time_OES = cp.deepcopy(time_info['elapsed_time'])
		elapsed_time_offset = np.median(all_timestamp_OES-all_elapsed_time_OES)
		all_elapsed_time_OES += time_info['PM_Cam_StartTime_ms']
		elapsed_time_offset2 = np.median(all_timestamp_OES-all_elapsed_time_OES)
		all_PVCAM_TimeStampBOF_ms_OES = cp.deepcopy(time_info['PVCAM_TimeStampBOF_ms'])
		PVCAM_TimeStampBOF_offset = np.median(all_timestamp_OES-all_PVCAM_TimeStampBOF_ms_OES)
		all_PVCAM_TimeStampBOF_ms_OES += time_info['PM_Cam_StartTime_ms']
		PVCAM_TimeStampBOF_offset2 = np.median(all_timestamp_OES-all_PVCAM_TimeStampBOF_ms_OES)
		# plt.figure();plt.plot(all_timestamp_OES,all_elapsed_time_OES);plt.pause(0.01)

		# all_timestamp_OES = cp.deepcopy(time_info['PVCAM_TimeStampBOF_ms'])-(time_info['PM_Cam_StartTime_ms']-time_info['PVCAM_TimeStampBOF_ms'][1])
		# all_timestamp_OES = cp.deepcopy(time_info['PVCAM_TimeStampBOF_ms'])
		guess_for_dt_OES = np.median(np.diff(all_timestamp_OES))
		temp = np.diff(all_timestamp_OES)
		temp = temp[temp<1.5*guess_for_dt_OES]
		guess_for_dt_OES = np.mean(temp[1:])
		extra_bad_pulses = np.arange(1,len(all_timestamp_OES))[np.diff(all_timestamp_OES)>1.5*guess_for_dt_OES]
		all_timestamp_OES = np.sort(np.concatenate((all_timestamp_OES,all_timestamp_OES[extra_bad_pulses-1]+guess_for_dt_OES)))
		all_elapsed_time_OES = np.sort(np.concatenate((all_elapsed_time_OES,all_elapsed_time_OES[extra_bad_pulses-1]+guess_for_dt_OES)))
		all_PVCAM_TimeStampBOF_ms_OES = np.sort(np.concatenate((all_PVCAM_TimeStampBOF_ms_OES,all_PVCAM_TimeStampBOF_ms_OES[extra_bad_pulses-1]+guess_for_dt_OES)))
		extra_bad_pulses = extra_bad_pulses-first_pulse_at_this_frame+2
		# bad_pulses_indexes = np.concatenate((bad_pulses_indexes,extra_bad_pulses))

		# guess_for_dt_OES = np.mean(np.diff(all_timestamp_OES[1:]))

		bad_pulses,first_good_pulse,first_pulse,last_pulse,miss_pulses,double_pulses,good_pulses, time_of_pulses, energy_per_pulse,duration_per_pulse,median_energy_good_pulses,median_duration_good_pulses = examine_current_trace(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/', fname_current_trace, df_log.loc[j, ['number_of_pulses']][0],want_the_power_per_pulse=True)
		all_time_of_pulses_DCsource = time_of_pulses['peak_of_the_peak']*1e3
		global_dictionary['time_of_pulses_DCsource_0'].append(all_time_of_pulses_DCsource[0])
		# guess_for_dt_DCsource = np.mean(np.diff(time_of_pulses))*1e3

		header = ryptw.readPTWHeader('/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/'+"{0:0=2g}".format(sequence)+'/IR_camera/'+IR_reference+'.ptw')
		frequency = 1/header.h_CEDIPAquisitionPeriod # Hz
		int_time = round(header.h_CEDIPIntegrationTime,9) # s
		timestamp_IR = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['corrected_timestamp_ms']
		index_good_pulses_IR = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['index_good_pulses']
		index_good_pulses_refined_IR = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['index_good_pulses_refined']
		index_good_pulses_refined_plus_IR = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['index_good_pulses_refined_plus']
		global_dictionary['IR_pulse_start'].append(np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['average_pulse_fitted_data'].all()['energy_fit_fix_duration']['pulse_t0_semi_inf'])
		interpolator = interp1d(np.arange(len(timestamp_IR)),timestamp_IR)
		peaks_timestamp_IR = interpolator(index_good_pulses_refined_IR)
		peaks_timestamp_plus_IR = interpolator(index_good_pulses_refined_plus_IR)
		peaks_relative_time_plus_IR = index_good_pulses_refined_plus_IR/frequency*1e3
		# index_good_pulses = index_good_pulses[:number_of_pulses]
		# np.diff(peaks_timestamp_IR)[np.diff(peaks_timestamp_IR)<np.median(np.diff(peaks_timestamp_IR))*1.5]
		# temp = np.diff(peaks_timestamp_plus_IR)
		# temp = temp[temp<1.5*guess_for_dt_OES]
		# guess_for_dt_IR = np.mean(temp)
		# measured_incremental_step = guess_for_dt_IR-guess_for_dt_OES

		all_timestamp_OES = all_timestamp_OES[first_pulse_at_this_frame:first_pulse_at_this_frame+number_of_pulses]
		all_elapsed_time_OES = all_elapsed_time_OES[first_pulse_at_this_frame:first_pulse_at_this_frame+number_of_pulses]
		# all_elapsed_time_OES = all_elapsed_time_OES[1:]

		number_of_pulses = min(number_of_pulses,len(all_timestamp_OES))	# this to account for the cases in which I have less frames than all the pulses done
		# all_timestamp_CB = all_timestamp_OES+CB_shift
		# timestamp_CB = []
		timestamp_OES = []
		index_good_pulses = []
		elapsed_time_OES = []
		time_of_pulses_DCsource = []
		for index in range(len(all_timestamp_OES)):
			if (index+1) in bad_pulses_indexes:
				continue
			timestamp_OES.append(all_timestamp_OES[index])
			elapsed_time_OES.append(all_elapsed_time_OES[index])
			# timestamp_CB.append(all_timestamp_CB[index])
			index_good_pulses.append(index+1)
			time_of_pulses_DCsource.append(all_time_of_pulses_DCsource[index])
		timestamp_OES = np.array(timestamp_OES)
		elapsed_time_OES = np.array(elapsed_time_OES)
		# timestamp_CB = np.array(timestamp_CB)
		index_good_pulses = np.array(index_good_pulses)
		time_of_pulses_DCsource = np.array(time_of_pulses_DCsource)
		if j in [280,275,393,394]:
			time_of_pulses_DCsource = time_of_pulses_DCsource[:-1]
			timestamp_OES = timestamp_OES[:-1]
			elapsed_time_OES = elapsed_time_OES[:-1]
			index_good_pulses = index_good_pulses[:-1]
		# plt.plot(timestamp_OES,elapsed_time_OES,'+');plt.pause(0.01)

		number_of_pulses = min(number_of_pulses,len(index_good_pulses))	# this to account for the cases in which I have less frames than all the pulses done

		peaks_timestamp_IR = peaks_timestamp_IR[:number_of_pulses]
		peaks_timestamp_plus_IR = peaks_timestamp_plus_IR[:number_of_pulses]
		peaks_relative_time_plus_IR = peaks_relative_time_plus_IR[:number_of_pulses]
		time_of_pulses_DCsource = time_of_pulses_DCsource[:number_of_pulses]

		# if len(extra_bad_pulses)>0:
		check = [not(value in extra_bad_pulses) for value in index_good_pulses]
		timestamp_OES = timestamp_OES[check]
		elapsed_time_OES = elapsed_time_OES[check]
		# timestamp_CB = timestamp_CB[check]
		index_good_pulses = index_good_pulses[check]
		peaks_timestamp_IR = peaks_timestamp_IR[check]
		peaks_timestamp_plus_IR = peaks_timestamp_plus_IR[check]
		peaks_relative_time_plus_IR = peaks_relative_time_plus_IR[check]
		time_of_pulses_DCsource = time_of_pulses_DCsource[check]

		# temp = np.diff(timestamp_OES)
		# temp = temp[temp<1.5*guess_for_dt_OES]
		# guess_for_dt_OES = np.mean(temp[1:])
		temp = np.diff(peaks_timestamp_plus_IR)
		temp = temp[temp<1.5*guess_for_dt_OES]
		guess_for_dt_IR = np.mean(temp)
		measured_incremental_step = guess_for_dt_IR-guess_for_dt_OES
		# CB_shift = CB_to_OES_initial_delay+(index_good_pulses-1)*measured_incremental_step
		CB_shift = CB_to_OES_initial_delay+(index_good_pulses-1)*incremental_step
		timestamp_CB = timestamp_OES + CB_shift

		# here I search for the time of the real start of the shot
		best_match = np.abs(CB_busy['CapStSourceBusy_time'][CB_busy['CapStSourceBusy_status']==True] - timestamp_CB[0] + 2*60*60*1e3).argmin()	# for some reason there is 2 hour difference
		CB_busy_start = CB_busy['CapStSourceBusy_time'][CB_busy['CapStSourceBusy_status']==True][best_match]
		# peaks_relative_time_plus_IR += CB_busy_start-timestamp_IR[0] + 1*60*60*1000
		global_dictionary['IR_CB_time_difference1_record'].append(CB_busy_start-timestamp_IR[0])
		global_dictionary['IR_CB_time_difference1_pulse'].append(peaks_timestamp_plus_IR[0]-CB_busy_start)
		best_match = np.abs(CB_busy['CapStPlasmaEnable_time'][CB_busy['CapStPlasmaEnable_status']==True] - timestamp_CB[0] + 2*60*60*1e3).argmin()	# for some reason there is 2 hour difference
		CB_system_ready = CB_busy['CapStPlasmaEnable_time'][CB_busy['CapStPlasmaEnable_status']==True][best_match]
		global_dictionary['IR_CB_time_difference2_record'].append(timestamp_IR[0]-CB_system_ready)
		global_dictionary['IR_CB_time_difference2_pulse'].append(peaks_timestamp_plus_IR[0]-CB_system_ready)

		# plt.figure(figsize=(10, 8))
		# plt.plot(index_good_pulses,timestamp_CB - peaks_timestamp_IR-60*60*1000)
		# fit = np.polyfit(index_good_pulses,timestamp_CB - peaks_timestamp_IR-60*60*1000,1)
		# # all_reference_time.append(fit[1])
		# plt.plot(index_good_pulses,np.polyval(fit,index_good_pulses),'--')
		# temp = timestamp_CB - peaks_timestamp_plus_IR-60*60*1000
		# plt.plot(index_good_pulses,temp)
		# fit = np.polyfit(index_good_pulses,temp,1)
		# R2 = 1-np.sum((temp-np.polyval(fit,index_good_pulses))**2)/np.sum((temp-np.mean(temp))**2)
		# plt.plot(index_good_pulses,np.polyval(fit,index_good_pulses),'--')
		# # plt.plot(index_good_pulses,temp -6e-4*index_good_pulses)
		# # fit = np.polyfit(index_good_pulses,temp -6e-4*index_good_pulses,1)
		# # plt.plot(index_good_pulses,np.polyval(fit,index_good_pulses),'--')
		# all_reference_time.append(np.concatenate((fit,[R2])))
		# all_J_ext.append(j)
		# all_time_ref.append([timestamp_CB[0],peaks_timestamp_plus_IR[0]-60*60*1000])
		# # fit = np.polyfit(np.arange(len(all_elapsed_time_OES)),all_elapsed_time_OES,1)
		# plt.title('merge='+str(merge_ID_target)+' j='+str(j)+'\nelapsed time offset=%.6g\nPVCAM TimeStampBOF offset=%.6g' %(elapsed_time_offset,PVCAM_TimeStampBOF_offset) +'\nfit '+str(fit))
		# plt.pause(0.01)

		temp = peaks_timestamp_plus_IR-60*60*1000 - time_of_pulses_DCsource
		fit = np.polyfit(index_good_pulses,temp,1)
		fit2 = np.polyfit(time_of_pulses_DCsource-time_of_pulses_DCsource[0],temp,1)
		R2 = 1-np.sum((temp-np.polyval(fit,index_good_pulses))**2)/np.sum((temp-np.mean(temp))**2)
		if print_3:
			plt.figure(figsize=(10, 5))
			plt.grid()
			plt.plot(index_good_pulses,temp)
			plt.plot(index_good_pulses,np.polyval(fit,index_good_pulses),'--')
			plt.title('merge='+str(merge_ID_target)+' j='+str(j)+ ' P=%.3gPa' %(target_chamber_pressure)+'\ncomparison of SOURCE and IR time\nfit '+str(fit)+'\nfit2 '+str(fit2))
			plt.xlabel('pulse index [au]')
			plt.ylabel('time difference [ms]')
			plt.pause(0.01)

		global_dictionary['fit_SOURCE_IR_absolute'].append(fit2)
		global_dictionary['fit_SOURCE_IR_absolute_R2'].append(R2)

		temp = timestamp_CB - peaks_timestamp_plus_IR-60*60*1000
		fit = np.polyfit(index_good_pulses,temp,1)
		fit2 = np.polyfit(timestamp_CB-timestamp_CB[0],temp,1)
		R2 = 1-np.sum((temp-np.polyval(fit,index_good_pulses))**2)/np.sum((temp-np.mean(temp))**2)
		if print_1:
			plt.figure(figsize=(10, 5))
			plt.grid()
			plt.plot(index_good_pulses,temp)
			plt.plot(index_good_pulses,np.polyval(fit,index_good_pulses),'--')
			plt.title('merge='+str(merge_ID_target)+' j='+str(j)+ ' P=%.3gPa' %(target_chamber_pressure) +'\ncomparison of OES and IR time\nfit '+str(fit)+'\nfit2 '+str(fit2))
			plt.pause(0.01)
		global_dictionary['fit_OES_IR_R2'].append(R2)
		global_dictionary['fit_OES_IR'].append(fit2)
		global_dictionary['time_difference_0'].append(temp[0])
		global_dictionary['elapsed_time_offset'].append(elapsed_time_offset)
		global_dictionary['elapsed_time_offset2'].append(elapsed_time_offset2)
		global_dictionary['PVCAM_TimeStampBOF_offset'].append(PVCAM_TimeStampBOF_offset)
		global_dictionary['PVCAM_TimeStampBOF_offset2'].append(PVCAM_TimeStampBOF_offset2)
		global_dictionary['time_of_pulses_IR_0'].append(peaks_relative_time_plus_IR[0])
		global_dictionary['timestamp_OES_0'].append(timestamp_OES[0])
		global_dictionary['peaks_timestamp_plus_IR_0'].append(peaks_timestamp_plus_IR[0]-60*60*1000)

		temp = time_of_pulses_DCsource-time_of_pulses_DCsource[0]-peaks_relative_time_plus_IR
		fit = np.polyfit(index_good_pulses,temp,1)
		fit2 = np.polyfit(time_of_pulses_DCsource*1e3-time_of_pulses_DCsource[0],temp,1)
		R2 = 1-np.sum((temp-np.polyval(fit,index_good_pulses))**2)/np.sum((temp-np.mean(temp))**2)
		if print_1:
			plt.figure(figsize=(10, 5))
			plt.grid()
			plt.plot(index_good_pulses,temp)
			plt.plot(index_good_pulses,np.polyval(fit,index_good_pulses),'--')
			plt.title('merge='+str(merge_ID_target)+' j='+str(j)+ ' P=%.3gPa' %(target_chamber_pressure) +'\ncomparison of SOURCE and IR time\nfit '+str(fit)+'\nfit2 '+str(fit2))
			plt.pause(0.01)
		global_dictionary['fit_SOURCE_IR_relative'].append(fit2)
		global_dictionary['fit_SOURCE_IR_relative_R2'].append(R2)


		# all_reference_time.append(np.concatenate((fit,[R2,temp[0],elapsed_time_offset,elapsed_time_offset2,PVCAM_TimeStampBOF_offset,PVCAM_TimeStampBOF_offset2,time_of_pulses[0],index_good_pulses_refined_plus_IR[0]])))
		# all_J_ext.append(j)
		# all_time_ref.append([timestamp_OES[0],peaks_timestamp_plus_IR[0]-60*60*1000])


# all_reference_time = np.array(all_reference_time).T
# all_J_ext = np.array(all_J_ext)
# all_time_ref = np.array(all_time_ref)
global_dictionary['fit_OES_IR'] = np.array(global_dictionary['fit_OES_IR']).T
global_dictionary['fit_SOURCE_IR_relative'] = np.array(global_dictionary['fit_SOURCE_IR_relative']).T
if print_2:
	plt.figure(figsize=(15, 7))
	plt.title('Absolute time comparison IR vs ADC')
	plt.plot(global_dictionary['pressure'],np.array(global_dictionary['fit_SOURCE_IR_absolute'])[:,1],'+')
	plt.xlabel('pressure [Pa]')
	plt.ylabel('time difference [ms]')
	plt.grid()
	plt.pause(0.01)
	# plt.figure()
	# plt.plot(global_dictionary['j'],global_dictionary['fit_OES_IR'][1]-global_dictionary['fit_OES_IR'][1,0])
	# plt.pause(0.01)
	plt.figure(figsize=(15, 7))
	plt.title('Absolute time comparison')
	plt.plot(np.array(global_dictionary['timestamp_OES_0'])-global_dictionary['timestamp_OES_0'][0],global_dictionary['fit_OES_IR'][1]-global_dictionary['fit_OES_IR'][1,0],label='fit')
	plt.plot(np.array(global_dictionary['timestamp_OES_0'])-global_dictionary['timestamp_OES_0'][0],global_dictionary['fit_OES_IR'][1]-global_dictionary['fit_OES_IR'][1,0],'+')
	plt.plot(np.array(global_dictionary['timestamp_OES_0'])-global_dictionary['timestamp_OES_0'][0],global_dictionary['time_difference_0']-global_dictionary['time_difference_0'][0],label='time_difference_0')
	plt.plot(np.array(global_dictionary['timestamp_OES_0'])-global_dictionary['timestamp_OES_0'][0],(np.array(global_dictionary['time_of_pulses_DCsource_0'])-global_dictionary['time_of_pulses_DCsource_0'][0])*1e3,label='relative_time_of_pulses_DCsource_0')
	plt.plot(np.array(global_dictionary['timestamp_OES_0'])-global_dictionary['timestamp_OES_0'][0],(np.array(global_dictionary['time_of_pulses_IR_0'])-global_dictionary['time_of_pulses_IR_0'][0]),label='relative_time_of_pulses_IR_0')
	plt.plot(np.array(global_dictionary['timestamp_OES_0'])-global_dictionary['timestamp_OES_0'][0],global_dictionary['fit_OES_IR'][1]-global_dictionary['fit_OES_IR'][1,0]-(np.array(global_dictionary['time_of_pulses_IR_0'])-global_dictionary['time_of_pulses_IR_0'][0]),label='fit-relative_time_of_pulses_DCsource_0')
	plt.legend(loc='best')
	plt.pause(0.01)

	plt.figure(figsize=(15, 7))
	plt.title('Absolute time comparison')
	plt.plot(global_dictionary['pressure'],global_dictionary['fit_OES_IR'][1]-global_dictionary['fit_OES_IR'][1,0],label='fit')
	plt.plot(global_dictionary['pressure'],global_dictionary['fit_OES_IR'][1]-global_dictionary['fit_OES_IR'][1,0],'+')
	plt.plot(global_dictionary['pressure'],(np.array(global_dictionary['time_of_pulses_DCsource_0'])-global_dictionary['time_of_pulses_DCsource_0'][0])*1e3,label='relative_time_of_pulses_DCsource_0')
	plt.plot(global_dictionary['pressure'],(np.array(global_dictionary['time_of_pulses_IR_0'])-global_dictionary['time_of_pulses_IR_0'][0]),label='relative_time_of_pulses_IR_0')
	plt.plot(global_dictionary['pressure'],global_dictionary['fit_OES_IR'][1]-global_dictionary['fit_OES_IR'][1,0]-(np.array(global_dictionary['time_of_pulses_IR_0'])-global_dictionary['time_of_pulses_IR_0'][0]),label='fit-relative_time_of_pulses_DCsource_0')
	plt.legend(loc='best')
	plt.pause(0.01)

	plt.figure(figsize=(15, 7))
	plt.title('Relative time comparison')
	plt.plot(np.array(global_dictionary['timestamp_OES_0'])-global_dictionary['timestamp_OES_0'][0],(np.array(global_dictionary['time_of_pulses_DCsource_0'])-global_dictionary['time_of_pulses_DCsource_0'][0])*1e3-(np.array(global_dictionary['time_of_pulses_IR_0'])-global_dictionary['time_of_pulses_IR_0'][0]),label='time_of_pulses_DCsource_0-time_of_pulses_IR_0')
	plt.plot(np.array(global_dictionary['timestamp_OES_0'])-global_dictionary['timestamp_OES_0'][0],np.array(global_dictionary['fit_SOURCE_IR'][1])-global_dictionary['fit_SOURCE_IR'][1,0],label='fit_SOURCE_IR')
	plt.plot(np.array(global_dictionary['timestamp_OES_0'])-global_dictionary['timestamp_OES_0'][0],np.array(global_dictionary['IR_CB_time_difference1_record'])-global_dictionary['IR_CB_time_difference1_record'][0],label='IR_CB_time_difference1_record')
	plt.plot(np.array(global_dictionary['timestamp_OES_0'])-global_dictionary['timestamp_OES_0'][0],np.array(global_dictionary['IR_CB_time_difference2_record'])-global_dictionary['IR_CB_time_difference2_record'][0],label='IR_CB_time_difference2_record')
	plt.plot(np.array(global_dictionary['timestamp_OES_0'])-global_dictionary['timestamp_OES_0'][0],np.array(global_dictionary['IR_CB_time_difference1_pulse'])-global_dictionary['IR_CB_time_difference1_pulse'][0],label='IR_CB_time_difference1_pulse')
	plt.plot(np.array(global_dictionary['timestamp_OES_0'])-global_dictionary['timestamp_OES_0'][0],np.array(global_dictionary['IR_CB_time_difference2_pulse'])-global_dictionary['IR_CB_time_difference2_pulse'][0],label='IR_CB_time_difference2_pulse')
	plt.legend(loc='best')
	plt.pause(0.01)

	plt.figure(figsize=(15, 7))
	plt.title('Relative time comparison')
	plt.plot(global_dictionary['pressure'],(np.array(global_dictionary['time_of_pulses_DCsource_0'])*1e3-np.array(global_dictionary['time_of_pulses_IR_0'])) - (np.array(global_dictionary['time_of_pulses_DCsource_0'])*1e3-np.array(global_dictionary['time_of_pulses_IR_0']))[0],'+')
	plt.plot(global_dictionary['pressure'],(np.array(global_dictionary['time_of_pulses_DCsource_0'])*1e3-np.array(global_dictionary['time_of_pulses_IR_0'])) - (np.array(global_dictionary['time_of_pulses_DCsource_0'])*1e3-np.array(global_dictionary['time_of_pulses_IR_0']))[0],label='time_of_pulses_DCsource_0-time_of_pulses_IR_0')
	plt.plot(global_dictionary['pressure'],np.array(global_dictionary['fit_SOURCE_IR'][1])-global_dictionary['fit_SOURCE_IR'][1,0],label='fit_SOURCE_IR')
	plt.plot(global_dictionary['pressure'],np.array(global_dictionary['IR_CB_time_difference1_record'])-global_dictionary['IR_CB_time_difference1_record'][0],label='IR_CB_time_difference1_record')
	plt.plot(global_dictionary['pressure'],np.array(global_dictionary['IR_CB_time_difference2_record'])-global_dictionary['IR_CB_time_difference2_record'][0],label='IR_CB_time_difference2_record')
	plt.plot(global_dictionary['pressure'],np.array(global_dictionary['IR_CB_time_difference1_pulse'])-global_dictionary['IR_CB_time_difference1_pulse'][0],label='IR_CB_time_difference1_pulse')
	plt.plot(global_dictionary['pressure'],np.array(global_dictionary['IR_CB_time_difference2_pulse'])-global_dictionary['IR_CB_time_difference2_pulse'][0],label='IR_CB_time_difference2_pulse')
	plt.legend(loc='best')
	plt.pause(0.01)

	plt.figure(figsize=(15, 7))
	plt.plot(global_dictionary['pressure'],(np.array(global_dictionary['time_of_pulses_DCsource_0'])*1e3-np.array(global_dictionary['time_of_pulses_IR_0'])) - (np.array(global_dictionary['time_of_pulses_DCsource_0'])*1e3-np.array(global_dictionary['time_of_pulses_IR_0']))[0],'+',label='time_of_pulses_DCsource_0-time_of_pulses_IR_0')
	plt.plot(global_dictionary['pressure'],np.array(global_dictionary['fit_SOURCE_IR'][1])-global_dictionary['fit_SOURCE_IR'][1,0],label='fit_SOURCE_IR')
	plt.legend(loc='best')
	plt.xlabel('pressure [Pa]')
	plt.ylabel('time difference [ms]')
	plt.title('Relative time difference between first plasma source peak and IR peak')
	plt.pause(0.01)


#   S T O P !
# This path is abanconed because I don't have a good enough precision. in IR, OES and source measurements time.
# The best I have is the difference between relative source and IR times (last plot), that is close within ~5-10 ms, but I think this is way too much.


'''
global_multiplicative_time_factor = np.mean(global_dictionary['fit_OES_IR'][0])
# reference_starting_point = all_time_ref[:,0].min()

global_dictionary['pressure'] = []
global_dictionary['fit_OES_IR_R2'] = []
global_dictionary['fit_OES_IR_corrected'] = []
global_dictionary['time_difference_0'] = []
global_dictionary['elapsed_time_offset'] = []
global_dictionary['elapsed_time_offset2'] = []
global_dictionary['PVCAM_TimeStampBOF_offset'] = []
global_dictionary['PVCAM_TimeStampBOF_offset2'] = []
global_dictionary['time_of_pulses_IR_0'] = []
global_dictionary['timestamp_OES_0'] = []
global_dictionary['peaks_timestamp_plus_IR_0'] = []

for merge_ID_target in merge_id_all:
	all_j=find_index_of_file(merge_ID_target,df_settings,df_log,only_OES=True)
	for j in all_j:
		# j=all_j[0]
		(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
		(CB_to_OES_initial_delay,incremental_step,first_pulse_at_this_frame,bad_pulses_indexes,number_of_pulses,fname_current_trace) = df_log.loc[j,['CB_to_OES_initial_delay','incremental_step','first_pulse_at_this_frame','bad_pulses_indexes','number_of_pulses','current_trace_file']]
		(IR_trace,IR_reference,IR_shape,magnetic_field,target_OES_distance,target_chamber_pressure,capacitor_voltage) = df_log.loc[j,['IR_trace','IR_reference','IR_shape','B','T_axial','p_n [Pa]','Vc']]
		(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
		if not(isinstance(IR_reference,str)):
			continue
		global_dictionary['merge_ID_target'].append(merge_ID_target)
		global_dictionary['j'].append(j)
		global_dictionary['pressure'].append(target_chamber_pressure)
		sequence = int(sequence)
		first_pulse_at_this_frame = int(first_pulse_at_this_frame)
		number_of_pulses = int(number_of_pulses)
		if bad_pulses_indexes=='':
			bad_pulses_indexes=[0]
		elif (isinstance(bad_pulses_indexes, float) or isinstance(bad_pulses_indexes, int)):
			bad_pulses_indexes=[bad_pulses_indexes]
		else:
			bad_pulses_indexes = bad_pulses_indexes.replace(' ', '').split(',')
			bad_pulses_indexes = list(map(int, bad_pulses_indexes))

		if -100 in bad_pulses_indexes:	# "keyword" in case I do not want to consider any of the pulses
			bad_pulses_indexes = np.linspace(1,number_of_pulses,number_of_pulses).astype('int')

		type = '.txt'
		filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2g}".format(int(sequence)) + '/Untitled_' + str(int(untitled)) + '/Pos0', type)[0]
		(bof, eof, roi_lb, roi_tr, time_info, real_exposure_time, PixelType, Gain,Noise) = get_metadata(fdir, folder, sequence,untitled,filename_metadata)
		# all_timestamp_OES = time_info['bof_corrected_ms']
		# if merge_ID_target==89:
		# 	number_of_pulses=249
		all_timestamp_OES = cp.deepcopy(time_info['PVCAM_TimeStampBOF_ms']) + time_info['PM_Cam_StartTime_ms']
		# all_timestamp_OES = bof*1e-6
		# all_timestamp_OES = cp.deepcopy(time_info['TimeStampMsec'])
		all_elapsed_time_OES = cp.deepcopy(time_info['elapsed_time'])
		elapsed_time_offset = np.median(all_timestamp_OES-all_elapsed_time_OES)
		all_elapsed_time_OES += time_info['PM_Cam_StartTime_ms']
		elapsed_time_offset2 = np.median(all_timestamp_OES-all_elapsed_time_OES)
		all_PVCAM_TimeStampBOF_ms_OES = cp.deepcopy(time_info['PVCAM_TimeStampBOF_ms'])
		PVCAM_TimeStampBOF_offset = np.median(all_timestamp_OES-all_PVCAM_TimeStampBOF_ms_OES)
		all_PVCAM_TimeStampBOF_ms_OES += time_info['PM_Cam_StartTime_ms']
		PVCAM_TimeStampBOF_offset2 = np.median(all_timestamp_OES-all_PVCAM_TimeStampBOF_ms_OES)
		# plt.figure();plt.plot(all_timestamp_OES,all_elapsed_time_OES);plt.pause(0.01)

		# all_timestamp_OES = cp.deepcopy(time_info['PVCAM_TimeStampBOF_ms'])-(time_info['PM_Cam_StartTime_ms']-time_info['PVCAM_TimeStampBOF_ms'][1])
		# all_timestamp_OES = cp.deepcopy(time_info['PVCAM_TimeStampBOF_ms'])
		guess_for_dt_OES = np.median(np.diff(all_timestamp_OES))
		temp = np.diff(all_timestamp_OES)
		temp = temp[temp<1.5*guess_for_dt_OES]
		guess_for_dt_OES = np.mean(temp[1:])
		extra_bad_pulses = np.arange(1,len(all_timestamp_OES))[np.diff(all_timestamp_OES)>1.5*guess_for_dt_OES]
		all_timestamp_OES = np.sort(np.concatenate((all_timestamp_OES,all_timestamp_OES[extra_bad_pulses-1]+guess_for_dt_OES)))
		all_elapsed_time_OES = np.sort(np.concatenate((all_elapsed_time_OES,all_elapsed_time_OES[extra_bad_pulses-1]+guess_for_dt_OES)))
		all_PVCAM_TimeStampBOF_ms_OES = np.sort(np.concatenate((all_PVCAM_TimeStampBOF_ms_OES,all_PVCAM_TimeStampBOF_ms_OES[extra_bad_pulses-1]+guess_for_dt_OES)))
		extra_bad_pulses = extra_bad_pulses-first_pulse_at_this_frame+2
		# bad_pulses_indexes = np.concatenate((bad_pulses_indexes,extra_bad_pulses))

		# guess_for_dt_OES = np.mean(np.diff(all_timestamp_OES[1:]))

		# bad_pulses,first_good_pulse,first_pulse,last_pulse,miss_pulses,double_pulses,good_pulses, time_of_pulses, energy_per_pulse,duration_per_pulse,median_energy_good_pulses,median_duration_good_pulses = examine_current_trace(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/', fname_current_trace, df_log.loc[j, ['number_of_pulses']][0],want_the_power_per_pulse=True)
		# all_time_of_pulses_DCsource = time_of_pulses['peak_of_the_peak']
		# global_dictionary['time_of_pulses_DCsource_0'].append(all_time_of_pulses_DCsource[0])
		# # guess_for_dt_DCsource = np.mean(np.diff(time_of_pulses))*1e3

		header = ryptw.readPTWHeader('/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/'+"{0:0=2g}".format(sequence)+'/IR_camera/'+IR_reference+'.ptw')
		frequency = 1/header.h_CEDIPAquisitionPeriod # Hz
		int_time = round(header.h_CEDIPIntegrationTime,9) # s
		timestamp_IR = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['corrected_timestamp_ms']
		index_good_pulses_IR = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['index_good_pulses']
		index_good_pulses_refined_IR = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['index_good_pulses_refined']
		index_good_pulses_refined_plus_IR = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) +'/file_index_' + str(j) +'_IR_trace_'+IR_trace+'.npz')['index_good_pulses_refined_plus']
		interpolator = interp1d(np.arange(len(timestamp_IR)),timestamp_IR)
		peaks_timestamp_IR = interpolator(index_good_pulses_refined_IR)
		peaks_timestamp_plus_IR = interpolator(index_good_pulses_refined_plus_IR)
		peaks_relative_time_plus_IR = index_good_pulses_refined_plus_IR/frequency*1e3
		# index_good_pulses = index_good_pulses[:number_of_pulses]
		# np.diff(peaks_timestamp_IR)[np.diff(peaks_timestamp_IR)<np.median(np.diff(peaks_timestamp_IR))*1.5]
		# temp = np.diff(peaks_timestamp_plus_IR)
		# temp = temp[temp<1.5*guess_for_dt_OES]
		# guess_for_dt_IR = np.mean(temp)
		# measured_incremental_step = guess_for_dt_IR-guess_for_dt_OES

		all_timestamp_OES = all_timestamp_OES[first_pulse_at_this_frame:first_pulse_at_this_frame+number_of_pulses]
		all_elapsed_time_OES = all_elapsed_time_OES[first_pulse_at_this_frame:first_pulse_at_this_frame+number_of_pulses]
		# all_elapsed_time_OES = all_elapsed_time_OES[1:]

		number_of_pulses = min(number_of_pulses,len(all_timestamp_OES))	# this to account for the cases in which I have less frames than all the pulses done
		# all_timestamp_CB = all_timestamp_OES+CB_shift
		# timestamp_CB = []
		timestamp_OES = []
		index_good_pulses = []
		elapsed_time_OES = []
		# time_of_pulses_DCsource = []
		for index in range(len(all_timestamp_OES)):
			if (index+1) in bad_pulses_indexes:
				continue
			timestamp_OES.append(all_timestamp_OES[index])
			elapsed_time_OES.append(all_elapsed_time_OES[index])
			# timestamp_CB.append(all_timestamp_CB[index])
			index_good_pulses.append(index+1)
			# time_of_pulses_DCsource.append(all_time_of_pulses_DCsource[index])
		timestamp_OES = np.array(timestamp_OES)
		elapsed_time_OES = np.array(elapsed_time_OES)
		# timestamp_CB = np.array(timestamp_CB)
		index_good_pulses = np.array(index_good_pulses)
		# time_of_pulses_DCsource = np.array(time_of_pulses_DCsource)
		# plt.plot(timestamp_OES,elapsed_time_OES,'+');plt.pause(0.01)

		number_of_pulses = min(number_of_pulses,len(index_good_pulses))	# this to account for the cases in which I have less frames than all the pulses done

		peaks_timestamp_IR = peaks_timestamp_IR[:number_of_pulses]
		peaks_timestamp_plus_IR = peaks_timestamp_plus_IR[:number_of_pulses]
		peaks_relative_time_plus_IR = peaks_relative_time_plus_IR[:number_of_pulses]
		# time_of_pulses_DCsource = time_of_pulses_DCsource[:number_of_pulses]

		# if len(extra_bad_pulses)>0:
		check = [not(value in extra_bad_pulses) for value in index_good_pulses]
		timestamp_OES = timestamp_OES[check]
		elapsed_time_OES = elapsed_time_OES[check]
		# timestamp_CB = timestamp_CB[check]
		index_good_pulses = index_good_pulses[check]
		peaks_timestamp_IR = peaks_timestamp_IR[check]
		peaks_timestamp_plus_IR = peaks_timestamp_plus_IR[check]
		peaks_relative_time_plus_IR = peaks_relative_time_plus_IR[check]
		# time_of_pulses_DCsource = time_of_pulses_DCsource[check]

		# temp = np.diff(timestamp_OES)
		# temp = temp[temp<1.5*guess_for_dt_OES]
		# guess_for_dt_OES = np.mean(temp[1:])
		temp = np.diff(peaks_timestamp_plus_IR)
		temp = temp[temp<1.5*guess_for_dt_OES]
		guess_for_dt_IR = np.mean(temp)
		measured_incremental_step = guess_for_dt_IR-guess_for_dt_OES
		# CB_shift = CB_to_OES_initial_delay+(index_good_pulses-1)*measured_incremental_step
		CB_shift = CB_to_OES_initial_delay+(index_good_pulses-1)*incremental_step
		timestamp_CB = timestamp_OES + CB_shift

		# plt.figure(figsize=(10, 8))
		# plt.plot(index_good_pulses,timestamp_CB - peaks_timestamp_IR-60*60*1000)
		# fit = np.polyfit(index_good_pulses,timestamp_CB - peaks_timestamp_IR-60*60*1000,1)
		# # all_reference_time.append(fit[1])
		# plt.plot(index_good_pulses,np.polyval(fit,index_good_pulses),'--')
		# temp = timestamp_CB - peaks_timestamp_plus_IR-60*60*1000
		# plt.plot(index_good_pulses,temp)
		# fit = np.polyfit(index_good_pulses,temp,1)
		# R2 = 1-np.sum((temp-np.polyval(fit,index_good_pulses))**2)/np.sum((temp-np.mean(temp))**2)
		# plt.plot(index_good_pulses,np.polyval(fit,index_good_pulses),'--')
		# # plt.plot(index_good_pulses,temp -6e-4*index_good_pulses)
		# # fit = np.polyfit(index_good_pulses,temp -6e-4*index_good_pulses,1)
		# # plt.plot(index_good_pulses,np.polyval(fit,index_good_pulses),'--')
		# all_reference_time.append(np.concatenate((fit,[R2])))
		# all_J_ext.append(j)
		# all_time_ref.append([timestamp_CB[0],peaks_timestamp_plus_IR[0]-60*60*1000])
		# # fit = np.polyfit(np.arange(len(all_elapsed_time_OES)),all_elapsed_time_OES,1)
		# plt.title('merge='+str(merge_ID_target)+' j='+str(j)+'\nelapsed time offset=%.6g\nPVCAM TimeStampBOF offset=%.6g' %(elapsed_time_offset,PVCAM_TimeStampBOF_offset) +'\nfit '+str(fit))
		# plt.pause(0.01)

		peaks_timestamp_plus_IR = peaks_timestamp_plus_IR + np.polyval([global_multiplicative_time_factor,0],timestamp_CB)

		temp = timestamp_CB - peaks_timestamp_plus_IR-60*60*1000
		fit = np.polyfit(index_good_pulses,temp,1)
		fit2 = np.polyfit(timestamp_CB-timestamp_CB[0],temp,1)
		R2 = 1-np.sum((temp-np.polyval(fit,index_good_pulses))**2)/np.sum((temp-np.mean(temp))**2)
		if print_1:
			plt.figure(figsize=(10, 8))
			plt.plot(index_good_pulses,temp)
			plt.plot(index_good_pulses,np.polyval(fit,index_good_pulses),'--')
			plt.title('merge='+str(merge_ID_target)+' j='+str(j) +'\ncomparison of OES and IR(corrected) time\nfit '+str(fit)+'\nfit2 '+str(fit2))
			plt.pause(0.01)
		global_dictionary['fit_OES_IR_R2'].append(R2)
		global_dictionary['fit_OES_IR_corrected'].append(fit2)
		global_dictionary['time_difference_0'].append(temp[0])
		global_dictionary['elapsed_time_offset'].append(elapsed_time_offset)
		global_dictionary['elapsed_time_offset2'].append(elapsed_time_offset2)
		global_dictionary['PVCAM_TimeStampBOF_offset'].append(PVCAM_TimeStampBOF_offset)
		global_dictionary['PVCAM_TimeStampBOF_offset2'].append(PVCAM_TimeStampBOF_offset2)
		global_dictionary['time_of_pulses_IR_0'].append(index_good_pulses_refined_plus_IR[0]/frequency)
		global_dictionary['timestamp_OES_0'].append(timestamp_OES[0])
		global_dictionary['peaks_timestamp_plus_IR_0'].append(peaks_timestamp_plus_IR[0]-60*60*1000)

		# temp = time_of_pulses_DCsource*1e3-peaks_relative_time_plus_IR
		# plt.figure(figsize=(10, 8))
		# plt.plot(index_good_pulses,temp)
		# fit = np.polyfit(index_good_pulses,temp,1)
		# fit2 = np.polyfit(time_of_pulses_DCsource*1e3-time_of_pulses_DCsource[0]*1e3,temp,1)
		# R2 = 1-np.sum((temp-np.polyval(fit,index_good_pulses))**2)/np.sum((temp-np.mean(temp))**2)
		# plt.plot(index_good_pulses,np.polyval(fit,index_good_pulses),'--')
		# plt.title('merge='+str(merge_ID_target)+' j='+str(j) +'\ncomparison of SOURCE and IR time\nfit '+str(fit)+'\nfit2 '+str(fit2))
		# plt.pause(0.01)
		# global_dictionary['fit_SOURCE_IR'].append(fit2)
		# global_dictionary['fit_SOURCE_IR_R2'].append(R2)


global_dictionary['fit_OES_IR_corrected'] = np.array(global_dictionary['fit_OES_IR_corrected']).T
if print_2:
	# plt.figure()
	# plt.plot(global_dictionary['j'],global_dictionary['fit_OES_IR_corrected'][1]-global_dictionary['fit_OES_IR_corrected'][1,0])
	# plt.pause(0.01)
	plt.figure()
	plt.plot(np.array(global_dictionary['timestamp_OES_0'])-global_dictionary['timestamp_OES_0'][0],global_dictionary['fit_OES_IR_corrected'][1]-global_dictionary['fit_OES_IR_corrected'][1,0],label='fit')
	plt.plot(np.array(global_dictionary['timestamp_OES_0'])-global_dictionary['timestamp_OES_0'][0],global_dictionary['fit_OES_IR_corrected'][1]-global_dictionary['fit_OES_IR_corrected'][1,0],'+')
	plt.plot(np.array(global_dictionary['timestamp_OES_0'])-global_dictionary['timestamp_OES_0'][0],global_dictionary['time_difference_0']-global_dictionary['time_difference_0'][0],label='time_difference_0')
	plt.plot(np.array(global_dictionary['timestamp_OES_0'])-global_dictionary['timestamp_OES_0'][0],(np.array(global_dictionary['time_of_pulses_DCsource_0'])-global_dictionary['time_of_pulses_DCsource_0'][0])*1e3,label='time_of_pulses_DCsource_0')
	plt.plot(np.array(global_dictionary['timestamp_OES_0'])-global_dictionary['timestamp_OES_0'][0],(np.array(global_dictionary['time_of_pulses_IR_0'])-global_dictionary['time_of_pulses_IR_0'][0])*1e3,label='time_of_pulses_IR_0')
	plt.plot(np.array(global_dictionary['timestamp_OES_0'])-global_dictionary['timestamp_OES_0'][0],global_dictionary['fit_OES_IR_corrected'][1]-global_dictionary['fit_OES_IR_corrected'][1,0]-(np.array(global_dictionary['time_of_pulses_IR_0'])-global_dictionary['time_of_pulses_IR_0'][0])*1e3,label='fit-time_of_pulses_IR_0')
	plt.legend(loc='best')
	plt.pause(0.01)

	plt.figure()
	plt.plot(global_dictionary['pressure'],global_dictionary['fit_OES_IR_corrected'][1]-global_dictionary['fit_OES_IR_corrected'][1,0],label='fit')
	plt.plot(global_dictionary['pressure'],global_dictionary['fit_OES_IR_corrected'][1]-global_dictionary['fit_OES_IR_corrected'][1,0],'+')
	plt.plot(global_dictionary['pressure'],(np.array(global_dictionary['time_of_pulses_DCsource_0'])-global_dictionary['time_of_pulses_DCsource_0'][0])*1e3,label='time_of_pulses_DCsource_0')
	plt.plot(global_dictionary['pressure'],(np.array(global_dictionary['time_of_pulses_IR_0'])-global_dictionary['time_of_pulses_IR_0'][0])*1e3,label='time_of_pulses_IR_0')
	plt.plot(global_dictionary['pressure'],global_dictionary['fit_OES_IR_corrected'][1]-global_dictionary['fit_OES_IR_corrected'][1,0]-(np.array(global_dictionary['time_of_pulses_IR_0'])-global_dictionary['time_of_pulses_IR_0'][0])*1e3,label='fit-time_of_pulses_IR_0')
	plt.legend(loc='best')
	plt.pause(0.01)







# merge_id_all=[85]
# all_reference_time = []
# all_J_ext = []
# all_time_ref = []
# all_target_chamber_pressure = []
# # merge_ID_target=95
# for merge_ID_target in merge_id_all:
# 	all_j=find_index_of_file(merge_ID_target,df_settings,df_log,only_OES=True)
# 	for j in all_j:
# 		# j=all_j[0]
# 		(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
# 		(CB_to_OES_initial_delay,incremental_step,first_pulse_at_this_frame,bad_pulses_indexes,number_of_pulses) = df_log.loc[j,['CB_to_OES_initial_delay','incremental_step','first_pulse_at_this_frame','bad_pulses_indexes','number_of_pulses']]
# 		first_pulse_at_this_frame = int(first_pulse_at_this_frame)
# 		number_of_pulses = int(number_of_pulses)
# 		if bad_pulses_indexes=='':
# 			bad_pulses_indexes=[0]
# 		elif (isinstance(bad_pulses_indexes, float) or isinstance(bad_pulses_indexes, int)):
# 			bad_pulses_indexes=[bad_pulses_indexes]
# 		else:
# 			bad_pulses_indexes = bad_pulses_indexes.replace(' ', '').split(',')
# 			bad_pulses_indexes = list(map(int, bad_pulses_indexes))
#
# 		if -100 in bad_pulses_indexes:	# "keyword" in case I do not want to consider any of the pulses
# 			bad_pulses_indexes = np.linspace(1,number_of_pulses,number_of_pulses).astype('int')
#
# 		type = '.txt'
# 		filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2g}".format(int(sequence)) + '/Untitled_' + str(int(untitled)) + '/Pos0', type)[0]
# 		(bof, eof, roi_lb, roi_tr, time_info, real_exposure_time, PixelType, Gain,Noise) = get_metadata(fdir, folder, sequence,untitled,filename_metadata)
# 		# all_timestamp_OES = time_info['bof_corrected_ms']
# 		# if merge_ID_target==89:
# 		# 	number_of_pulses=249
# 		all_timestamp_OES = bof*1e-6
# 		print(all_timestamp_OES[0])
# 		all_elapsed_time_OES = cp.deepcopy(time_info['elapsed_time'])
# 		elapsed_time_offset = np.median(all_timestamp_OES-all_elapsed_time_OES)
# 		all_elapsed_time_OES += time_info['PM_Cam_StartTime_ms']
# 		print(all_elapsed_time_OES[0])
# 		elapsed_time_offset2 = np.median(all_timestamp_OES-all_elapsed_time_OES)
# 		all_PVCAM_TimeStampBOF_ms_OES = cp.deepcopy(time_info['PVCAM_TimeStampBOF_ms'])
# 		PVCAM_TimeStampBOF_offset = time_info['PVCAM_TimeStampBOF_ms'][0]
# 		all_PVCAM_TimeStampBOF_ms_OES += time_info['PM_Cam_StartTime_ms']
# 		PVCAM_TimeStampBOF_offset2 = np.median(all_timestamp_OES-all_PVCAM_TimeStampBOF_ms_OES)

'''
