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
from scipy.signal import find_peaks, peak_prominences as get_proms

# n = np.arange(10, 20)
# waveLengths = [656.45377,486.13615,434.0462,410.174,397.0072,388.9049,383.5384] + list((364.6*(n*n)/(n*n-4)))
waveLengths = [486.13615,434.0462,410.174,397.0072,388.9049,383.5384]
waveLengths_interp = interpolate.interp1d(waveLengths[:2], [1.39146712e+03,8.83761543e+02],fill_value='extrapolate')


fdir = '/home/ffederic/work/Collaboratory/test/experimental_data'
df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
df_settings = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_3.csv',index_col=0)

geom_null = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
geom_null.loc[0] = [0,0,0,0,0]

# Here I put multiple pictures together to form a full image in time

merge_ID_target = int(input('insert the merge_ID as per file /home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_1.csv'))
# merge_ID_target =66
time_resolution_scan = False
time_resolution_scan_improved = True
time_resolution_extra_skip = 0

started=0
merge_time_window=[-1,4]
overexposed_treshold = 3600
data_sum=0
merge_values = []
merge_time = []
merge_row = []
merge_Gain = []
merge_overexposed = []
merge_wavelengths = []
# wavel_1 = 700
# wavel_2 = 1300
time_first_row_all_all = []
data_all_1_all = []
data_all_2_all = []
# if ((not time_resolution_scan and not os.path.exists('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_merge_tot.npz')) or (time_resolution_scan and not os.path.exists('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.npz'))):
if True:
	all_j = find_index_of_file(merge_ID_target, df_settings, df_log, only_OES=True)
	temp_gain = []
	for j in all_j:
		(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
		type = '.txt'
		filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)[0]
		# filename_metadata = functions.all_file_names(pathfiles, type)[0]
		(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir, folder, sequence,untitled,filename_metadata)
		temp_gain.append(Gain[0])
	all_j = np.array([all_j for _, all_j in sorted(zip(temp_gain, all_j))])
	for j in all_j:
		dataDark = load_dark(j,df_settings,df_log,fdir,geom_null)
		(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
		type = '.tif'
		filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)
		#filenames = functions.all_file_names(pathfiles, type)
		type = '.txt'
		filename_metadata = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)[0]
		#filename_metadata = functions.all_file_names(pathfiles, type)[0]
		(bof,eof,roi_lb,roi_tr,elapsed_time,real_exposure_time,PixelType,Gain) = get_metadata(fdir,folder,sequence,untitled,filename_metadata)
		(CB_to_OES_initial_delay,incremental_step,first_pulse_at_this_frame,bad_pulses_indexes,number_of_pulses) = df_log.loc[j,['CB_to_OES_initial_delay','incremental_step','first_pulse_at_this_frame','bad_pulses_indexes','number_of_pulses']]
		first_pulse_at_this_frame = int(first_pulse_at_this_frame)
		if bad_pulses_indexes=='':
			bad_pulses_indexes=[0]
		elif (isinstance(bad_pulses_indexes, float) or isinstance(bad_pulses_indexes, int)):
			bad_pulses_indexes=[bad_pulses_indexes]
		else:
			bad_pulses_indexes = bad_pulses_indexes.replace(' ', '').split(',')
			bad_pulses_indexes = list(map(int, bad_pulses_indexes))

		time_first_row_all = []
		data_all_1 = []
		data_all_2 = []

		data_all = []
		for index,filename in enumerate(filenames):
			fname = fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0/'+filename
			im = Image.open(fname)
			data = np.array(im)
			data_all.append(data)
		data=np.mean(data_all,axis=(0,1))
		binPeaks = find_peaks(data, distance=25)[0]
		binProms = get_proms(data, binPeaks)[0]
		binPeaks = np.array([binPeaks for _, binPeaks in sorted(zip(binProms, binPeaks))])
		wavel_1 = binPeaks[-1]
		wavel_2 = binPeaks[-2]

		flag_good_pluse_read = []
		for index,filename in enumerate(filenames):
			fname = fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0/'+filename
			im = Image.open(fname)
			data = np.array(im)
			data = data- dataDark
			data = fix_minimum_signal2(data)
			data_all_1.append(np.sum(data[:, wavel_1-15:wavel_1+15]))
			data_all_2.append(np.sum(data[:, wavel_2 - 15:wavel_2 + 15]))
			time_first_row = -CB_to_OES_initial_delay - (index - first_pulse_at_this_frame + 1) * incremental_step
			time_first_row_all.append(time_first_row)
			if (index<first_pulse_at_this_frame-1 or index>=(first_pulse_at_this_frame+number_of_pulses-1)):
				print(str(index)+' is out of pulses')
				continue
			elif (index-first_pulse_at_this_frame+2) in bad_pulses_indexes:
				print(str(index)+' is a bad pulse')
				continue
			if time_resolution_scan:
				if index%(1+time_resolution_extra_skip)!=0:
					continue	# The purpose of this is to test how things change for different time skippings
			print(filename)
			flag_good_pluse_read.append(index)
		flag_good_pluse_read = np.array(flag_good_pluse_read)

		data_all_1 = np.array(data_all_1)
		data_sum_max_1 = np.max(data_all_1)
		data_all_1 = data_all_1 / data_sum_max_1
		data_all_2 = np.array(data_all_2)
		data_sum_max_2 = np.max(data_all_2)
		data_all_2 = data_all_2 / data_sum_max_2

		plt.figure(j)
		plt.plot(flag_good_pluse_read,data_all_1[flag_good_pluse_read],'xk', markersize=20)
		plt.plot(flag_good_pluse_read, data_all_2[flag_good_pluse_read], 'xk', markersize=20)
		# plt.plot(index + 2 - first_pulse_at_this_frame, np.sum(data[:, 700:900] / data_sum_max, axis=(-1, -2)), 'xk',markersize=20)
		# plt.pause(0.001)

		# data_sum = np.sum(data_all[:, :, 1300:1500], axis=(-1, -2))
		# data_sum = np.sum(data_all[:, :, 700:900], axis=(-1, -2))
		# data_sum = data_sum / np.max(data_sum)
		plt.figure(j)
		plt.plot(range(0,len(data_all_1)),data_all_1)
		plt.plot(range(0,len(data_all_1)),data_all_1, 'o')
		# data_sum = np.sum(data_all[:, :, 1300:1500], axis=(-1, -2))
		# data_sum = np.sum(data_all[:, :, 700:900], axis=(-1, -2))
		# data_sum_max = np.max(data_sum)
		# data_sum = data_sum / np.max(data_sum)
		plt.figure(j)
		plt.plot(range(0,len(data_all_2)),data_all_2)
		plt.plot(range(0,len(data_all_2)),data_all_2, 'o')

		for i in range(1,1+len(data_all_2)):
			plt.plot([i, i], [np.min(data_all_2), np.max(data_all_2)])
		#plt.pause(0.001)

		fname_current_trace = df_log.loc[j, ['current_trace_file']][0]
		if isinstance(df_log.loc[j,['current_trace_file']][0],str):
			current_traces = pd.read_csv(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/' + fname_current_trace + '.tsf',index_col=False, delimiter='\t')
			current_traces_time = current_traces['Time [s]']
			current_traces_total = current_traces['I_Src_AC [A]']
			# plt.figure();plt.plot(current_traces_time,current_traces_total);plt.pause(0.01)

			bad_pulses, first_good_pulse, first_pulse, last_pulse, miss_pulses, double_pulses, good_pulses, time_of_pulses,prominences = examine_current_trace(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/', fname_current_trace,number_of_pulses,want_the_prominences=True)
			print('number of pulses is '+str(len(prominences)))
			print([bad_pulses])
			temp=np.ones_like(data_all_1)*np.max(prominences)
			temp[first_pulse_at_this_frame-1:first_pulse_at_this_frame-1+int(number_of_pulses)]=prominences[:min(int(number_of_pulses),len(prominences))]
			prominences = copy.deepcopy(temp)
			prominences = prominences/np.max(prominences)

			time_first_row_all_all.append(time_first_row_all)
			data_all_1_all.append(data_all_1/prominences)
			data_all_2_all.append(data_all_2/prominences)
		else:
			time_first_row_all_all.append(time_first_row_all)
			data_all_1_all.append(data_all_1)
			data_all_2_all.append(data_all_2)

plt.figure(merge_ID_target)
# time_first_row_all_all = (np.array(time_first_row_all_all)).flatten()
# data_all_1_all = (np.array(data_all_1_all)).flatten()
# data_all_2_all = (np.array(data_all_2_all)).flatten()
# data_all_1_all = np.array([data_all_1_all for _, data_all_1_all in sorted(zip(time_first_row_all_all, data_all_1_all))])
# data_all_2_all = np.array([data_all_2_all for _, data_all_2_all in sorted(zip(time_first_row_all_all, data_all_2_all))])
# time_first_row_all_all = np.sort(time_first_row_all_all)
markers = ['o','+','v','s','*','X']
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise']
for index in range(len(time_first_row_all_all)):
	plt.plot(time_first_row_all_all[index],data_all_1_all[index],color=color[0],marker=markers[index],label='record '+str(all_j[index]))
	plt.plot(time_first_row_all_all[index],data_all_2_all[index], color=color[1], marker=markers[index])
# plt.plot(data_all_1_all)
# plt.plot(data_all_2_all)

plt.legend(loc='best')
plt.pause(0.001)


