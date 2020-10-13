import numpy as np
import matplotlib.pyplot as plt
#import .functions
import os,sys

from numpy.core.multiarray import ndarray

os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
from functions.spectools import get_angle,rotate,get_tilt,do_tilt,getFirstBin, binData, get_angle_2
from functions.fabio_add import find_nearest_index,multi_gaussian,all_file_names,load_dark,find_index_of_file,get_metadata,examine_current_trace,movie_from_data,get_bin_and_interv_no_lines, four_point_transform, fix_minimum_signal, do_tilt_no_lines, fix_minimum_signal2
from functions.Calibrate import do_waveL_Calib
from PIL import Image
import xarray as xr
import pandas as pd
import copy
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.signal import gaussian
from skimage.transform import resize

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



#fdir = '/home/ff645/GoogleDrive/ff645/YPI - Fabio/Collaboratory/Experimental data/tests/test_data_folder'
#df_log = pd.read_csv('functions/Log/shots_2.csv',index_col=0)
#df_settings = pd.read_csv('functions/Log/settin	gs_2.csv',index_col=0)

fdir = '/home/ffederic/work/Collaboratory/test/experimental_data'
df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
df_settings = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_3.csv',index_col=0)

merge_ID=3
# merge_ID = 20


logdir = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/'
calfname = 'calLog.xlsx'

df_calLog = pd.read_excel(logdir + calfname)
df_calLog.columns = ['folder', 'date', 'time', 'gratPos', 't_exp', 'frames', 'ln/mm', 'gratTilt', 'I_det', 'type',
					 'exper', 'Dark']
df_calLog['folder'] = df_calLog['folder'].astype(str)

spherefname = r'Labsphere_Radiance.xls'
df_sphere = pd.read_excel(logdir + spherefname)
I_nom = df_sphere.columns[1].split()[-3:-1]  # Nominal calibration current as string in A
I_nom = float(I_nom[0]) * 10 ** (int(I_nom[1][-2:]) + 6)  # float, \muA
df_sphere.columns = ['waveL', 'RadRaw']
df_sphere.units = ['nm', 'W cm-2 sr-1 nm-1', 'W m-2 sr-1 nm-1 \muA-1']
df_sphere['Rad'] = df_sphere.RadRaw / I_nom * 1e4	# I think this is just to go from cm-2 to m-2


# binnedSens = do_Intensity_Calib(merge_ID,df_settings,df_log,df_calLog,df_sphere,geom,waveLcoefs,waves_to_calibrate=['Hb'],fdir=fdir)
#
# binnedSens = np.ones(np.shape(binnedSens))


angle = 0.0534850538519
tilt = [29.3051619433, 0.0015142337977, 47.6592283252]
waveLcoefs = np.ones((2, 3)) * np.nan
waveLcoefs[1] = [-2.21580743e-06, 1.07315743e-01, 3.40712312e+02]
geom = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
geom.loc[0] = [angle, tilt[1], tilt[0], tilt[2], tilt[2]]
geom_null = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
geom_null.loc[0] = [0,0,geom['binInterv'],geom['bin00a'],geom['bin00b']]



# int_time = [5, 10, 20, 50, 100, 200, 500]	# at 500ms int time is saturated
# calib_ID=[133,135,137,139,141,143,145]

int_time = [5, 10, 20, 50, 100, 200]
calib_ID=[133,135,137,139,141,143]
calib_ID = np.array([calib_ID for _, calib_ID in sorted(zip(int_time, calib_ID))])
int_time = np.sort(int_time)

wavel_range = 25
# wavelength_780=1266
# wavelength_750=1556

wavelength_750=1556
target_up = np.polyval(waveLcoefs[1], wavelength_750+wavel_range)
target_down = np.polyval(waveLcoefs[1], wavelength_750-wavel_range)
wavelength_780=wavelength_750-290

bin_interv_high_int_time = []
data_all = []
binned_all = []
peak_sum = []
longCal_all=[]
for iFile, j in enumerate(calib_ID):
	(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]

	dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)

	type = '.tif'
	filenames = all_file_names(
		fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
	type = '.txt'
	filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0',type)[0]
	(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir,folder,sequence,untitled,filename_metadata)

	data_sum = 0
	temp = []



	for index, filename in enumerate(filenames):
		fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(
			untitled) + '/Pos0/' + filename
		# print(fname)
		im = Image.open(fname)
		data = np.array(im)
		data = (data - dataDark)*Gain[index]
		# data = rotate(data, geom['angle'])
		# data = do_tilt(data, geom['tilt'])
		data = fix_minimum_signal2(data)


		data_sum += data
		temp.append(data)
	temp = np.mean(temp, axis=0)

	# if (df_log.loc[j, ['t_exp']][0]==0.01):
	temp = fix_minimum_signal(temp)
		# noise = temp[:20].tolist() + temp[-20:].tolist()
		# noise_std = np.std(noise, axis=(0, 1))
		# noise = np.mean(noise, axis=(0, 1))
		# test_std = 10000
		# factor_std = 0
		# std_record = []
		# while factor_std <= 2:
		# 	reconstruct = []
		# 	for irow, row in enumerate(temp.tolist()):
		# 		minimum = np.mean(np.sort(row)[:int(len(row) / 40)]) + factor_std * np.std(
		# 			np.sort(row)[:int(len(row) / 40)])
		# 		# minimum = np.min(row)
		# 		reconstruct.append(np.array(row) + noise - minimum)
		# 	reconstruct = np.array(reconstruct)
		# 	test_std = np.std(reconstruct[:, :40])
		# 	std_record.append(test_std)
		# 	print('factor_std ' + str(factor_std))
		# 	print('test_std ' + str(test_std))
		# 	factor_std += 0.1
		# factor_std = np.array(std_record).argmin()*0.1
		# reconstruct = []
		# for irow, row in enumerate(temp.tolist()):
		# 	minimum = np.mean(np.sort(row)[:int(len(row) / 40)]) + factor_std * np.std(
		# 		np.sort(row)[:int(len(row) / 40)])
		# 	# minimum = np.min(row)
		# 	reconstruct.append(np.array(row) + noise - minimum)
		# temp=reconstruct

	# temp = temp - dataDark
	# angle = get_angle_no_lines(temp)	# I cant use this because it's impossible to decouple the angle error from the tilt.
	temp = rotate(temp, geom['angle'])
	temp = do_tilt_no_lines(temp)

	first_bin, bin_interv = get_bin_and_interv_no_lines(temp)
	print('bin interval of '+str(bin_interv))
	bin_interv_high_int_time.append(bin_interv)
	binned_data = binData(temp, first_bin, bin_interv)/(df_log.loc[j, ['Calib_ID']][0]*1000)
	longCal_all.append( binData(temp, first_bin - bin_interv, bin_interv, nCh=42)/(df_log.loc[j, ['Calib_ID']][0]*1000))
	# binned_data = binned_data - min(np.min(binned_data), 0)
	# angle=get_angle(temp,no_line_present=True)
	# temp = rotate(temp, angle)
	# tilt = get_tilt(temp)
	# temp = do_tilt(temp, tilt)

	# temp = temp - dataDark
	# angle=get_angle(temp)
	# temp = rotate(temp, angle)
	# tilt = get_tilt(temp)
	# temp = do_tilt(temp, tilt)
	data_all.append(temp)
	binned_all.append(binned_data)

	if (df_log.loc[j, ['gratPos']][0] == 780):
		wavel = wavelength_780
	elif (df_log.loc[j, ['gratPos']][0] == 750):
		wavel = wavelength_750
	peak_sum.append(np.sum(binned_data[:,wavel - wavel_range:wavel + wavel_range],axis=(-1,-2)))
peak_sum=np.array(peak_sum)


max_coordinate = 0
for index in range(len(data_all)):
	max_coordinate = max(max_coordinate,np.shape(data_all[index])[0])
for index in range(len(data_all)):
	while np.shape(data_all[index])[0]<max_coordinate:
		temp_1=data_all[index].tolist()
		temp_1.append(np.zeros(np.shape(temp_1[0])))
		data_all[index] = np.array(temp_1)
data_all_long_int_time=np.array(data_all)
binned_all=np.array(binned_all)
bin_interv_high_int_time = np.nanmean(bin_interv_high_int_time)


'''
def line_throug_zero(x, *params):

	m = params[0]
	# q = params[1]
	return m*x


fit_1_L,temp2=curve_fit(line_throug_zero,int_time, peak_sum, p0=(peak_sum[-1]-peak_sum[0])/int_time[-1], maxfev=100000000)

constant=0
for i in range(10):
	fit_1 = np.polyfit(int_time, peak_sum, 1)

fit_1 = np.polyfit(int_time, peak_sum, 1)
fit_2 = np.polyfit(int_time, peak_sum, 2)
fit_3 = np.polyfit(int_time, peak_sum, 3)
# iFit = np.polyval(fit, 0.01)
plt.figure();
plt.plot(int_time, peak_sum, 'o')
plt.plot(int_time, np.polyval(fit_3, int_time), label='cubic fit')
plt.plot(int_time, np.polyval(fit_2, int_time),'r', label='quadratic fit')
plt.plot(int_time, np.polyval(fit_1, int_time),'g', label='linear fit')
plt.plot(int_time, np.polyval([fit_1_L,0], int_time),'k', label='lenear fit through zero')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best')
plt.title('Change in sensitivity with integration time from ' + str(np.around(target_down, decimals=1)) + ' to ' + str(
	np.around(target_up, decimals=1)) + ' nm')
plt.xlabel('integration time [ms]')
plt.ylabel('sum on all chords [au]')
plt.pause(0.001)


# plt.figure()
# for index in range(len(data_all)):
# 	plt.plot(data_all[index,:,800]);plt.pause(0.01)
#
# plt.figure()
# for index in range(len(data_all)):
# 	plt.plot(binned_all[index,20]/np.max(binned_all[index,20]),label='int time '+str(int_time[index])+'ms');plt.pause(0.01)
# plt.title('Scaled binned sensitivity for increasing integration time (channel 20)')
# plt.legend(loc='best');plt.pause(0.01)


if False:		#this bit is to extrapolate the intensity at 0.01ms integration time
	if (df_log.loc[j, ['gratPos']][0] == 780):
		wavel = wavelength_780
	elif (df_log.loc[j, ['gratPos']][0] == 750):
		wavel = wavelength_750
	# profiles = np.sum(data_all[:,:,wavel - wavel_range:wavel + wavel_range],axis=(-1))
	#
	# from scipy.signal import find_peaks, peak_prominences as get_proms
	# peak_sum=[]
	# for index in range(len(data_all)):
	# 	prominence=500
	# 	num_peaks=0
	# 	while num_peaks!=40:
	# 		peaks = find_peaks(profiles[index], prominence=prominence,distance=25)[0]
	# 		temp = []
	# 		for value in peaks:
	# 			if (value > 5 and value < len(profiles[index]) - 5):
	# 				temp.append(value)
	# 		peaks = np.array(temp)
	# 		num_peaks = len(peaks)
	# 		print(np.max(profiles[index][peaks]))
	# 		print(num_peaks)
	# 		if num_peaks<40:
	# 			prominence-=10
	# 		elif num_peaks>40:
	# 			prominence+=10
	#
	# 	seak_sum_single=[]
	# 	skip=-10
	# 	while skip<11:
	# 		seak_sum_single+=np.sum(profiles[index,peaks+skip])
	# 		skip+=1
	# 	peak_sum.append(np.sum(profiles[index,peaks]))

	peak_sum = np.sum(binned_all[:,:,wavel - wavel_range:wavel + wavel_range],axis=(-1,-2))

	peak_sum=np.array(peak_sum)
	# plt.figure();plt.plot(int_time,peak_sum);plt.pause(0.001)

	fit_2 = np.polyfit(int_time, peak_sum, 2)
	# iFit = np.polyval(fit, 0.01)
	plt.figure();plt.plot(int_time,peak_sum,'o')
	plt.plot(int_time,np.polyval(fit_2, int_time),label='quadratic fit')
	fit_1 = np.polyfit(int_time, peak_sum, 1)
	plt.plot(int_time,np.polyval(fit_1, int_time),label='lenear fit')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend(loc='best')
	plt.title('Change in sensitivity with integration time from '+str(np.around(target_down,decimals=1))+' to '+str(np.around(target_up,decimals=1))+' nm')
	plt.xlabel('integration time [ms]')
	plt.ylabel('sum on all chords [au]')
	plt.pause(0.001)

elif False:			# this is to check consistency between different integration times at 16bit
	profiles = data_all[:,820]
	plt.figure()
	for index in range(len(profiles)):
		plt.plot((profiles[index]-max(profiles[index]))/int_time[index],label='int time '+str(int_time[index]))
	# plt.xscale('log')
	# plt.yscale('log')
	plt.legend(loc='best')
	# plt.title('Change in sensitivity with integration time from '+str(np.around(target_down,decimals=1))+' to '+str(np.around(target_up,decimals=1))+' nm')
	plt.xlabel('pixel [au]')
	plt.ylabel('intensity [au]')
	plt.pause(0.001)


plt.plot(0.01, np.polyval(fit_2, 0.01),'+r', label='quadratic fit')
plt.plot(0.01, np.polyval(fit_1, 0.01),'+g', label='lenear fit')
plt.plot(0.01, np.polyval([fit_1_L,0], 0.01),'+k', label='lenear fit through zero')
plt.legend(loc='best')
plt.pause(0.001)

'''

int_time = [0.01]
data_all = []
calib_ID=[129,130]
for iFile, j in enumerate(calib_ID):
	(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]

	dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)

	type = '.tif'
	filenames = all_file_names(
		fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
	type = '.txt'
	filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0',type)[0]
	(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir,folder,sequence,untitled,filename_metadata)

	data_sum = 0
	temp = []
	for index, filename in enumerate(filenames):
		fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(
			untitled) + '/Pos0/' + filename
		# print(fname)
		im = Image.open(fname)
		data = np.array(im)
		data = (data - dataDark)*Gain[index]

		# data = rotate(data, geom['angle'])
		# data = do_tilt(data, geom['tilt'])
		data = fix_minimum_signal2(data)

		data_sum += data
		temp.append(data)
	temp = np.mean(temp, axis=0)
	# temp = fix_minimum_signal(temp)

	# if (df_log.loc[j, ['t_exp']][0]==0.01):	REMOVED BECAUSE I REALISED THAT THE SHIFTING OF THE MINUMIM HAPPENS FOR ALL CONDITIONS

		# noise = temp[:20].tolist() + temp[-20:].tolist()
		# noise_std = np.std(noise, axis=(0, 1))
		# noise = np.mean(noise, axis=(0, 1))
		# factor_std = 0
		# std_record = []
		# while factor_std <= 2:
		# 	reconstruct = []
		# 	for irow, row in enumerate(temp.tolist()):
		# 		minimum = np.mean(np.sort(row)[:int(len(row) / 40)]) + factor_std * np.std(
		# 			np.sort(row)[:int(len(row) / 40)])
		# 		# minimum = np.min(row)
		# 		reconstruct.append(np.array(row) + noise - minimum)
		# 	reconstruct = np.array(reconstruct)
		# 	test_std = np.std(reconstruct[:, :40])
		# 	std_record.append(test_std)
		# 	print('factor_std ' + str(factor_std))
		# 	print('test_std ' + str(test_std))
		# 	factor_std += 0.1
		# factor_std = np.array(std_record).argmin()*0.1
		# reconstruct = []
		# for irow, row in enumerate(temp.tolist()):
		# 	minimum = np.mean(np.sort(row)[:int(len(row) / 40)]) + factor_std * np.std(
		# 		np.sort(row)[:int(len(row) / 40)])
		# 	# minimum = np.min(row)
		# 	reconstruct.append(np.array(row) + noise - minimum)
		# temp=reconstruct

	# temp = temp - dataDark
	# angle = get_angle_no_lines(temp)	# I cant use this because it's impossible to decouple the angle error from the tilt.
	# temp = rotate(temp, geom['angle'])
	# temp = do_tilt_no_lines(temp)

	if (df_log.loc[j, ['gratPos']][0] == 780):
		wavel = wavelength_780
	elif (df_log.loc[j, ['gratPos']][0] == 750):
		wavel = wavelength_750
	# profiles_for_peaks = np.sum(temp[:, -100:-1], axis=(-1))
	# profiles = np.sum(temp[:, wavel-wavel_range:wavel+wavel_range], axis=(-1))
	# profiles_for_peaks=profiles
	#
	# from scipy.signal import find_peaks, peak_prominences as get_proms
	#
	# prominence = 20
	# num_peaks = 0
	# while num_peaks != 40:
	# 	peaks = find_peaks(profiles_for_peaks, prominence=prominence,distance=25)[0]
	# 	temp = []
	# 	for value in peaks:
	# 		if (value > 5 and value < len(profiles_for_peaks) - 5):
	# 			temp.append(value)
	# 	peaks = np.array(temp)
	# 	num_peaks = len(peaks)
	# 	print(np.max(profiles_for_peaks[peaks]))
	# 	print(num_peaks)
	# 	if num_peaks < 40:
	# 		prominence -= 0.1
	# 	elif num_peaks > 40:
	# 		prominence += 0.1
	#
	# seak_sum_single=0
	# skip=-10
	# while skip<11:
	# 	seak_sum_single+=np.sum(profiles[peaks+skip])
	# 	skip+=1

	data = copy.deepcopy(temp)
	data = data/(df_log.loc[j, ['Calib_ID']][0]*1000)
	data_all.append(data)

data_all = np.array(data_all)
data=np.mean(data_all,axis=(0))
data = rotate(data, geom['angle'])
data = do_tilt_no_lines(data)
# data = data - min(np.min(data),0)

# max_coordinate = 0
# for index in range(len(data_all)):
# 	max_coordinate = max(max_coordinate,np.shape(data_all[index])[0])
# for index in range(len(data_all)):
# 	while np.shape(data_all[index])[0]<max_coordinate:
# 		temp_1=data_all[index].tolist()
# 		temp_1.append(np.zeros(np.shape(temp_1[0])))
# 		data_all[index] = np.array(temp_1)
# data_all=np.array(data_all)
# data=np.mean(data_all,axis=(0))

data_no_mask = copy.deepcopy(data)
# data_no_mask = data_no_mask - min(np.min(data_no_mask),0)
first_bin, bin_interv = get_bin_and_interv_no_lines(data)
print('bin interval of ' + str(bin_interv))
bin_interv_no_mask = bin_interv
binned_data_no_mask = binData(data, first_bin, bin_interv)
# binned_data_no_mask = binned_data_no_mask - min(np.min(binned_data_no_mask),0)
longCal_no_mask = binData(data,first_bin-bin_interv,bin_interv,nCh = 42)
# longCal_no_mask = longCal_no_mask - min(np.min(longCal_no_mask),0)


peak_sum_single = np.sum(binned_data_no_mask[:, wavel - wavel_range:wavel + wavel_range], axis=(-1, -2))  # type: ndarray



int_time = [5, 10, 20, 50, 100, 200]
calib_ID=[133,135,137,139,141,143]
calib_ID = np.array([calib_ID for _, calib_ID in sorted(zip(int_time, calib_ID))])
int_time = np.sort(int_time)


plt.figure()
scaled_down_binned_all=[]
for i in range(len(binned_all)):
	scaled_down_binned_all.append(binned_all[i]*peak_sum_single/peak_sum[i])
	plt.plot(np.mean(binned_all[i]*peak_sum_single/peak_sum[i],axis=0),label=str(int_time[i]))
scaled_down_binned_all=np.array(scaled_down_binned_all)
plt.legend(loc='best')
plt.pause(0.01)

averaged_scaled_down_binned_all=np.mean(scaled_down_binned_all,axis=0)

plt.figure()
plt.imshow(averaged_scaled_down_binned_all, 'rainbow',origin='lower',aspect=10)
plt.colorbar()
plt.pause(0.01)



plt.figure()
scaled_down_calBinned=[]
for i in range(len(binned_all)):
	topDark = longCal_all[i][0]
	botDark = longCal_all[i][-1]
	wtDark = np.arange(1, 41) / 41
	calBinned = longCal_all[i][1:41] - (np.outer(wtDark, topDark) + np.outer(wtDark[::-1], botDark)) + (topDark + botDark) / 2  # MODIFIED TO AVOID NEGATIVE VALUES
	scaled_down_calBinned.append(calBinned*peak_sum_single/peak_sum[i])
	plt.plot(np.mean(binned_all[i]*peak_sum_single/peak_sum[i],axis=0),label=str(int_time[i]))
scaled_down_calBinned=np.array(scaled_down_calBinned)
plt.legend(loc='best')
plt.pause(0.01)

averaged_scaled_down_calBinned=np.mean(scaled_down_calBinned,axis=0)

plt.figure()
plt.imshow(averaged_scaled_down_calBinned, 'rainbow',origin='lower',aspect=10)
plt.colorbar()
plt.pause(0.01)



# plt.plot(0.01, peak_sum_single, 'o');
# plt.pause(0.001)


# # I'll use the quadratic fit because seems to fit better
# iFit = np.polyval(fit_2, 0.01)
# scaling = iFit / peak_sum_single
# data = data * scaling

'''

longCal = binData(data,first_bin-bin_interv,bin_interv,nCh = 42)
topDark = longCal[0]
botDark = longCal[-1]
wtDark = np.arange(1,41)/41
calBinned = longCal[1:41] -(np.outer(wtDark,topDark)+np.outer(wtDark[::-1],botDark)) +(topDark+botDark)/2	#	MODIFIED TO AVOID NEGATIVE VALUES


# Due to the low signal I smooth the curve for a better numerical stability

averaging = 20
extrapolation = 100
averaging = int(averaging/2)*2
for index,chord in enumerate(calBinned):
	temp = np.convolve(chord, np.ones(averaging)/averaging , mode='same')
	fit = np.polyfit(range(-extrapolation,0,1), chord[-extrapolation:], 2)
	fit[-1]=temp[-averaging//2]
	temp[-averaging//2:] = np.polyval(fit, range(-averaging//2,0,1))
	fit = np.polyfit(range(0,extrapolation,1), chord[:extrapolation], 2)
	fit[-1]=temp[averaging//2]
	temp[:averaging//2] = np.polyval(fit, range(0,averaging//2,1))
	calBinned[index] = temp

averaging = 3
extrapolation = 5
averaging = int(averaging/2)*2
for index,chord in enumerate(calBinned.T):
	# temp = np.convolve(chord, np.ones(averaging)/averaging , mode='same')
	temp = np.convolve(chord, [1/4,2/4,1/4], mode='same')
	fit = np.polyfit(range(-extrapolation,0,1), chord[-extrapolation:], 2)
	fit[-1]=temp[-averaging//2]
	temp[-averaging//2:] = np.polyval(fit, range(-averaging//2,0,1))
	fit = np.polyfit(range(0,extrapolation,1), chord[:extrapolation], 2)
	fit[-1]=temp[averaging//2]
	temp[:averaging//2] = np.polyval(fit, range(0,averaging//2,1))
	calBinned[:,index] = temp
calBinned = calBinned - min(calBinned.min(),0)+1
'''

spline=interpolate.interp1d(df_sphere.waveL,df_sphere.Rad,kind='cubic') #Spline interp of sphere radiance

#	THIS IS NOT NECESSARY ANYMORE BECAUSE i AVOID NEGATIVE VALUES
# calBinned_min= np.max(calBinned)/1000
# for irow, row in enumerate(calBinned):
# 	for icol, value in enumerate(row):
# 		if (value < calBinned_min):
# 			calBinned[irow, icol] =  calBinned_min

to_save = np.zeros((3,)+np.shape(averaged_scaled_down_calBinned))



to_save[1] = averaged_scaled_down_calBinned / spline(np.polyval(waveLcoefs[1],np.arange(data.shape[1])))

np.save('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/Sensitivity_1',to_save)



# to compare sensitivities with and without the hand made filter to avoid saturation I must compare the two in an area that stays the same. this is about before pixel=500




wavel_range = 100
wavelength_750=400
target_up = np.polyval(waveLcoefs[1], wavelength_750+wavel_range)
target_down = np.polyval(waveLcoefs[1], wavelength_750-wavel_range)
wavelength_780=wavelength_750-290
if (df_log.loc[j, ['gratPos']][0] == 780):
	wavel = wavelength_780
elif (df_log.loc[j, ['gratPos']][0] == 750):
	wavel = wavelength_750

peak_sum = np.sum(binned_data_no_mask[:, wavel - wavel_range:wavel + wavel_range], axis=(-1, -2))



int_time = [0.01]
data_all = []
calib_ID=[153, 154, 155]
for iFile, j in enumerate(calib_ID):
	(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]

	dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)

	type = '.tif'
	filenames = all_file_names(
		fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
	type = '.txt'
	filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0',type)[0]
	(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir,folder,sequence,untitled,filename_metadata)

	data_sum = 0
	temp = []
	for index, filename in enumerate(filenames):
		fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(
			untitled) + '/Pos0/' + filename
		# print(fname)
		im = Image.open(fname)
		data = np.array(im)
		data = (data - dataDark)*Gain[index]
		# data = rotate(data, geom['angle'])
		# data = do_tilt(data, geom['tilt'])
		data = fix_minimum_signal2(data)


		data_sum += data
		temp.append(data)
	temp = np.mean(temp, axis=0)

	# if (df_log.loc[j, ['t_exp']][0]==0.01):
	# temp = fix_minimum_signal(temp)
		# noise = temp[:20].tolist() + temp[-20:].tolist()
		# noise_std = np.std(noise, axis=(0, 1))
		# noise = np.mean(noise, axis=(0, 1))
		# factor_std = 0
		# std_record = []
		# while factor_std <= 2:
		# 	reconstruct = []
		# 	for irow, row in enumerate(temp.tolist()):
		# 		minimum = np.mean(np.sort(row)[:int(len(row) / 40)]) + factor_std * np.std(
		# 			np.sort(row)[:int(len(row) / 40)])
		# 		# minimum = np.min(row)
		# 		reconstruct.append(np.array(row) + noise - minimum)
		# 	reconstruct = np.array(reconstruct)
		# 	test_std = np.std(reconstruct[:, :40])
		# 	std_record.append(test_std)
		# 	print('factor_std ' + str(factor_std))
		# 	print('test_std ' + str(test_std))
		# 	factor_std += 0.1
		# factor_std = np.array(std_record).argmin()*0.1
		# reconstruct = []
		# for irow, row in enumerate(temp.tolist()):
		# 	minimum = np.mean(np.sort(row)[:int(len(row) / 40)]) + factor_std * np.std(
		# 		np.sort(row)[:int(len(row) / 40)])
		# 	# minimum = np.min(row)
		# 	reconstruct.append(np.array(row) + noise - minimum)
		# temp=reconstruct

	# temp = temp - dataDark
	# angle = get_angle_no_lines(temp)	# I cant use this because it's impossible to decouple the angle error from the tilt.
	# temp = rotate(temp, geom['angle'])
	# temp = do_tilt_no_lines(temp)


	if (df_log.loc[j, ['gratPos']][0] == 780):
		wavel = wavelength_780
	elif (df_log.loc[j, ['gratPos']][0] == 750):
		wavel = wavelength_750
	# profiles_for_peaks = np.sum(temp[:, -100:-1], axis=(-1))
	# profiles = np.sum(temp[:, wavel-wavel_range:wavel+wavel_range], axis=(-1))
	# profiles_for_peaks=profiles
	#
	# from scipy.signal import find_peaks, peak_prominences as get_proms
	#
	# prominence = 20
	# num_peaks = 0
	# while num_peaks != 40:
	# 	peaks = find_peaks(profiles_for_peaks, prominence=prominence,distance=25)[0]
	# 	temp = []
	# 	for value in peaks:
	# 		if (value > 5 and value < len(profiles_for_peaks) - 5):
	# 			temp.append(value)
	# 	peaks = np.array(temp)
	# 	num_peaks = len(peaks)
	# 	print(np.max(profiles_for_peaks[peaks]))
	# 	print(num_peaks)
	# 	if num_peaks < 40:
	# 		prominence -= 0.1
	# 	elif num_peaks > 40:
	# 		prominence += 0.1
	#
	# seak_sum_single=0
	# skip=-10
	# while skip<11:
	# 	seak_sum_single+=np.sum(profiles[peaks+skip])
	# 	skip+=1

	data = copy.deepcopy(temp)
	data = data/df_log.loc[j, ['Calib_ID']][0]
	data_all.append(data)

data_all = np.array(data_all)
data=np.mean(data_all,axis=(0))
data = rotate(data, geom['angle'])
data = do_tilt_no_lines(data)

# max_coordinate = 0
# for index in range(len(data_all)):
# 	max_coordinate = max(max_coordinate,np.shape(data_all[index])[0])
# for index in range(len(data_all)):
# 	while np.shape(data_all[index])[0]<max_coordinate:
# 		temp_1=data_all[index].tolist()
# 		temp_1.append(np.zeros(np.shape(temp_1[0])))
# 		data_all[index] = np.array(temp_1)
# data_all=np.array(data_all)
# data=np.mean(data_all,axis=(0))

first_bin, bin_interv = get_bin_and_interv_no_lines(data)
print('bin interval of ' + str(bin_interv))
bin_interv_mask = bin_interv
binned_data_mask = binData(data, first_bin, bin_interv)
# binned_data_mask = binned_data_mask - min(np.min(binned_data_mask),0)



#	NON POSSO USARE QUESTA STRATEGIA PERCHE' TRA LE DUE MISURAZIONI C'E' UNO SHIFT IN LUNGHEZZA D'ONDA
# peak_sum_single = np.sum(binned_data_mask[:, wavel - wavel_range:wavel + wavel_range], axis=(-1, -2))  # type: ndarray
#
# pixels_number = np.shape(binned_data_mask[:, wavel - wavel_range:wavel + wavel_range])[0]*np.shape(binned_data_mask[:, wavel - wavel_range:wavel + wavel_range])[1]
# scaling_1 = (peak_sum - peak_sum_single)/(pixels_number)
#
# binned_data_mask += scaling_1

to_be_fitted = np.sum(binned_data_no_mask,axis=(0))
to_fit = np.sum(binned_data_mask,axis=(0))
plt.figure();plt.plot(to_be_fitted,label='no balancing filter');plt.pause(0.001)
plt.plot(to_fit,label='balancing filter');plt.pause(0.001)
plt.legend(loc='best')
plt.title('Sum of the intensity across all chords from calibration')
plt.xlabel('wavelength axis [au]')
plt.ylabel('intensity [au]')
plt.pause(0.001)

averaging=30
to_be_fitted = np.convolve(to_be_fitted, np.ones(averaging)/averaging , mode='valid')
to_fit = np.convolve(to_fit, np.ones(averaging)/averaging , mode='valid')

#	I MUST FIT FROM 0-470 PIXELS OF THE INTENSITY WITH BALANCING FILTER

range_to_fit = 500-2*averaging
to_fit = to_fit[0:range_to_fit]
score = []
for shift in range(len(to_be_fitted)-range_to_fit):
	profile_to_be_fitted = copy.deepcopy(to_be_fitted[shift:shift+range_to_fit])
	profile_to_be_fitted -= profile_to_be_fitted[0]- to_fit[0]
	score.append(np.sum(((profile_to_be_fitted - to_fit)/profile_to_be_fitted)**2))
score = np.array(score)
best_wavelength_shift = score.argmin()+averaging//2
best_intensity_shift = to_be_fitted[best_wavelength_shift]-to_fit[0]
plt.figure();plt.plot(score);plt.pause(0.001)

#
# to_be_fitted = np.sum(binned_data_no_mask,axis=(0))
# to_fit = np.sum(binned_data_mask,axis=(0))
# plt.figure();plt.plot(range(len(to_be_fitted)),to_be_fitted,label='no balancing filter');plt.pause(0.001)
# plt.plot(range(best_wavelength_shift,best_wavelength_shift+len(to_be_fitted)),to_fit+best_intensity_shift,label='balancing filter');plt.pause(0.001)
# plt.legend(loc='best')
# plt.title('Sum of the intensity across all chords from calibration \n balancing filter fitted to match the other in ['+str(0+averaging)+':'+str(range_to_fit+averaging)+']')
# plt.xlabel('wavelength axis [au]')
# plt.ylabel('intensity [au]')
# plt.pause(0.001)
#
# longCal_mask = binData(data,first_bin-bin_interv,bin_interv,nCh = 42)
# longCal_mask = longCal_mask - min(np.min(longCal_mask),0)
# # # 16/06/19 add: I don't think I need to shift in wavelength, but only in hight!
# # longCal = longCal+best_intensity_shift/40
# longCal_new = np.zeros_like(longCal)
# longCal_new[:,:best_wavelength_shift] = longCal_no_mask[:,:best_wavelength_shift]
# longCal_new[:,best_wavelength_shift:] =longCal_mask [:,:-best_wavelength_shift]+best_intensity_shift/40
# longCal = copy.deepcopy(longCal_new)
#
# # data_new = np.zeros_like(data)
# # data_new[:,best_wavelength_shift:] = data[:,:-best_wavelength_shift]+best_intensity_shift/(len(binned_data_no_mask)*bin_interv*)
# # data_new[:,:best_wavelength_shift] = data_no_mask[:,:best_wavelength_shift]
#
# plt.figure();plt.plot(longCal_no_mask[21],label='no balancing filter');plt.pause(0.001)
# plt.plot(longCal[21],label='balancing filter');plt.pause(0.001)
# plt.legend(loc='best')
# plt.title('intensity on the chord 20 with and wothout balancing filter')
# plt.xlabel('wavelength axis [au]')
# plt.ylabel('intensity [au]')
# plt.pause(0.001)
#
#
#
# # longCal = binData(data,first_bin-bin_interv,bin_interv,nCh = 42)
# topDark = longCal[0]
# botDark = longCal[-1]
# wtDark = np.arange(1,41)/41
# calBinned = longCal[1:41] -(np.outer(wtDark,topDark)+np.outer(wtDark[::-1],botDark)) +(topDark+botDark)/2	#	MODIFIED TO AVOID NEGATIVE VALUES
#
#
# # Due to the low signal I smooth the curve for a better numerical stability
#
# averaging = 10
# extrapolation = 100
# averaging = int(averaging/2)*2
# for index,chord in enumerate(calBinned):
# 	temp = np.convolve(chord, np.ones(averaging)/averaging , mode='same')
# 	fit = np.polyfit(range(-extrapolation,0,1), chord[-extrapolation:], 2)
# 	fit[-1]=temp[-averaging//2]
# 	temp[-averaging//2:] = np.polyval(fit, range(-averaging//2,0,1))
# 	fit = np.polyfit(range(0,extrapolation,1), chord[:extrapolation], 2)
# 	fit[-1]=temp[averaging//2]
# 	temp[:averaging//2] = np.polyval(fit, range(0,averaging//2,1))
# 	calBinned[index] = temp
#
#
# averaging = 3
# extrapolation = 5
# averaging = int(averaging/2)*2
# for index,chord in enumerate(calBinned.T):
# 	temp = np.convolve(chord, [1/4,2/4,1/4] , mode='same')
# 	fit = np.polyfit(range(-extrapolation,0,1), chord[-extrapolation:], 2)
# 	fit[-1]=temp[-averaging//2]
# 	temp[-averaging//2:] = np.polyval(fit, range(-averaging//2,0,1))
# 	fit = np.polyfit(range(0,extrapolation,1), chord[:extrapolation], 2)
# 	fit[-1]=temp[averaging//2]
# 	temp[:averaging//2] = np.polyval(fit, range(0,averaging//2,1))
# 	calBinned[:,index] = temp
#
#
# spline=interpolate.interp1d(df_sphere.waveL,df_sphere.Rad,kind='cubic') #Spline interp of sphere radiance
#
# #	THIS IS NOT NECESSARY ANYMORE BECAUSE i AVOID NEGATIVE VALUES
# # calBinned_min= np.max(calBinned)/1000
# # for irow, row in enumerate(calBinned):
# # 	for icol, value in enumerate(row):
# # 		if (value < calBinned_min):
# # 			calBinned[irow, icol] =  calBinned_min
#
# to_save = np.zeros((3,)+np.shape(binned_data_no_mask))
#
#
#
# to_save[1] = calBinned / spline(np.polyval(waveLcoefs[1],np.arange(data.shape[1])))
#
# np.save('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/Sensitivity_2',to_save)
#
# plt.figure();plt.plot(to_save[1][20]);plt.pause(0.001)
# plt.xlabel('wavelength axis [au]')
# plt.ylabel('intensity [au]')
# plt.pause(0.001)
#
#
#
# data_all = []
# for merge_ID_target in range(40,41):
# 	all_j=find_index_of_file(merge_ID_target,df_settings,df_log)
# 	for j in all_j:
# 		dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
# 		(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
# 		type = '.tif'
# 		filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)
# 		type = '.txt'
# 		filename_metadata = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)[0]
#
# 		# plt.figure()
# 		# data_all = []
# 		for index, filename in enumerate(filenames):
# 			fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + \
# 					filename
# 			im = Image.open(fname)
# 			data = np.array(im)
# 			# data = fix_minimum_signal(data)
# 			data = data - dataDark
# 			# data = fix_minimum_signal(data)
# 			data_all.append(data)
# 		data_all = np.array(data_all)
# data_mean = np.mean(data_all,axis=0)
# tilt = np.array([ 28.08337189,          np.nan,  46.13475998]) # FROM analysis_filtered_data.py
# binned_data = binData(data_mean, tilt[-1], tilt[0])
#
# plt.figure();plt.plot(np.mean(binned_data/to_save[1],axis=0));plt.pause(0.01)
# plt.figure();plt.plot(np.mean(binned_data,axis=0));plt.pause(0.01)
# plt.figure();plt.plot(np.mean(to_save[1],axis=0));plt.pause(0.01)
#
#
#
#
# # THERE IS A PROBLEM! THE SENSITIVITY PROFILE DOES NOT LINE UP NICELY WITH THE SPECTR, SOME LINES ARE IN THE WRONG POSITION AND GET MULTIPLIED BY WRONG VALUE
# # i HAVE TO QUANTIFY THE MISMATCH AND ADJUST FFOR THAT
#
# waveLengths = [486.13615,434.0462,410.174,397.0072,388.9049,383.5384]
#
# peaks_pixels = []
# p=np.poly1d(waveLcoefs[1])
# for i, peak in enumerate(waveLengths):
# 	roots = (p-peak).roots
# 	root = roots[np.logical_and(roots<1608,roots>0)]
# 	peaks_pixels.append(root)
# peaks_pixels_first_set = np.array(peaks_pixels)
#
# peaks_pixels_first_set = np.array([ 1394.42983955,   884.88411048,   654.64440536,   528.50806905, 451.18562213,   400.09487505])
#
# # for merge_ID_target in range(40,53):
# # 	all_j=find_index_of_file(merge_ID_target,df_settings,df_log)
# # 	for j in all_j:
# # 		(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
# # 		type = '.tif'
# # 		filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)
# # 		type = '.txt'
# # 		filename_metadata = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)[0]
# #
# # 		# plt.figure()
# # 		data_all = []
# # 		for index, filename in enumerate(filenames):
# # 			fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + \
# # 					filename
# # 			im = Image.open(fname)
# # 			data = np.array(im)
# # 			# data = fix_minimum_signal(data)
# # 			data_all.append(data)
# # 		data_all = np.array(data_all)
# # data_mean = np.mean(data_all,axis=0)
#
# peaks_new = np.array([1268, 780, 558, 436, 362])
# peaks_pixels_first_set = peaks_pixels_first_set[:len(peaks_new)]
# waveLcoefs_new = np.polyfit(peaks_new,waveLengths[:len(peaks_new)],2)
# waveLcoefs_new = np.array([ -1.39629946e-06,   1.09550955e-01,   3.49466935e+02])
#
# # it's not a linear deformation, but I hane no time do deal with non univorm deformations. I will consider a constant shift
# shift = int(np.mean(peaks_new-peaks_pixels_first_set))
#
#
#
# # longCal = longCal+best_intensity_shift/40
# longCal_new = np.zeros_like(longCal)
# longCal_new[:,:best_wavelength_shift+shift] = longCal_no_mask[:,:best_wavelength_shift+shift]
# longCal_new[:,best_wavelength_shift+shift:] =longCal_mask [:,:-best_wavelength_shift-shift]+best_intensity_shift/40
# longCal = copy.deepcopy(longCal_new)
#
# # data_new = np.zeros_like(data)
# # data_new[:,best_wavelength_shift:] = data[:,:-best_wavelength_shift]+best_intensity_shift/(len(binned_data_no_mask)*bin_interv*)
# # data_new[:,:best_wavelength_shift] = data_no_mask[:,:best_wavelength_shift]
#
# plt.figure();plt.plot(longCal_no_mask[21],label='no balancing filter');plt.pause(0.001)
# plt.plot(longCal[21],label='balancing filter');plt.pause(0.001)
# plt.legend(loc='best')
# plt.title('intensity on the chord 20 with and wothout balancing filter')
# plt.xlabel('wavelength axis [au]')
# plt.ylabel('intensity [au]')
# plt.pause(0.001)
#
#
#
# # longCal = binData(data,first_bin-bin_interv,bin_interv,nCh = 42)
# topDark = longCal[0]
# botDark = longCal[-1]
# wtDark = np.arange(1,41)/41
# calBinned = longCal[1:41] -(np.outer(wtDark,topDark)+np.outer(wtDark[::-1],botDark)) +(topDark+botDark)/2	#	MODIFIED TO AVOID NEGATIVE VALUES
#
#
# # Due to the low signal I smooth the curve for a better numerical stability
#
# averaging = 10
# extrapolation = 100
# averaging = int(averaging/2)*2
# for index,chord in enumerate(calBinned):
# 	temp = np.convolve(chord, np.ones(averaging)/averaging , mode='same')
# 	fit = np.polyfit(range(-extrapolation,0,1), chord[-extrapolation:], 2)
# 	fit[-1]=temp[-averaging//2]
# 	temp[-averaging//2:] = np.polyval(fit, range(-averaging//2,0,1))
# 	fit = np.polyfit(range(0,extrapolation,1), chord[:extrapolation], 2)
# 	fit[-1]=temp[averaging//2]
# 	temp[:averaging//2] = np.polyval(fit, range(0,averaging//2,1))
# 	calBinned[index] = temp
#
#
# averaging = 3
# extrapolation = 5
# averaging = int(averaging/2)*2
# for index,chord in enumerate(calBinned.T):
# 	temp = np.convolve(chord, [1/4,2/4,1/4] , mode='same')
# 	fit = np.polyfit(range(-extrapolation,0,1), chord[-extrapolation:], 2)
# 	fit[-1]=temp[-averaging//2]
# 	temp[-averaging//2:] = np.polyval(fit, range(-averaging//2,0,1))
# 	fit = np.polyfit(range(0,extrapolation,1), chord[:extrapolation], 2)
# 	fit[-1]=temp[averaging//2]
# 	temp[:averaging//2] = np.polyval(fit, range(0,averaging//2,1))
# 	calBinned[:,index] = temp
#
#
# spline=interpolate.interp1d(df_sphere.waveL,df_sphere.Rad,kind='cubic') #Spline interp of sphere radiance
#
# #	THIS IS NOT NECESSARY ANYMORE BECAUSE i AVOID NEGATIVE VALUES
# # calBinned_min= np.max(calBinned)/1000
# # for irow, row in enumerate(calBinned):
# # 	for icol, value in enumerate(row):
# # 		if (value < calBinned_min):
# # 			calBinned[irow, icol] =  calBinned_min
#
# to_save = np.zeros((3,)+np.shape(binned_data_no_mask))
#
#
#
# to_save[1] = calBinned / spline(np.polyval(waveLcoefs[1],np.arange(data.shape[1])))
#
# # np.save('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/Sensitivity_2',to_save)
#
# plt.figure();plt.plot(to_save[1][20]);plt.pause(0.001)
# plt.xlabel('wavelength axis [au]')
# plt.ylabel('intensity [au]')
# plt.pause(0.001)
#
#
# data_all = []
# for merge_ID_target in range(40,41):
# 	all_j=find_index_of_file(merge_ID_target,df_settings,df_log)
# 	for j in all_j:
# 		dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
# 		(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
# 		type = '.tif'
# 		filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)
# 		type = '.txt'
# 		filename_metadata = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)[0]
#
# 		# plt.figure()
# 		# data_all = []
# 		for index, filename in enumerate(filenames):
# 			fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + \
# 					filename
# 			im = Image.open(fname)
# 			data = np.array(im)
# 			# data = fix_minimum_signal(data)
# 			data = data - dataDark
# 			# data = fix_minimum_signal(data)
# 			data_all.append(data)
# 		data_all = np.array(data_all)
# data_mean = np.mean(data_all,axis=0)
# tilt = np.array([ 28.08337189,          np.nan,  46.13475998]) # FROM analysis_filtered_data.py
# binned_data = binData(data_mean, tilt[-1], tilt[0])
#
# plt.figure();plt.plot(np.mean(binned_data/to_save[1],axis=0));plt.pause(0.01)
# plt.figure();plt.plot(np.mean(binned_data,axis=0));plt.pause(0.01)
# plt.figure();plt.plot(np.mean(to_save[1],axis=0));plt.pause(0.01)
#
#
# np.save('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/Sensitivity_2',to_save)
#
#
#







##### 23/06/2019 ANOTHER VERSION OF THE PREVIOUS FUNCTION, TO GET THE SENSITIVITY FOR THE SECOND SET OF MEASUREMENTS.
##### I WILL TRY TO ACCOUNT FOR SHIFTS OF WAVELENGTH

if False:
	data_all = []
	all_j=[149]
	for j in all_j:
		dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
		(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
		type = '.tif'
		filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)
		type = '.txt'
		filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0',type)[0]
		(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir,folder,sequence,untitled,filename_metadata)

		# plt.figure()
		# data_all = []
		for index, filename in enumerate(filenames[:len(filenames)//2]):
			fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + \
					filename
			im = Image.open(fname)
			data = np.array(im)
			# data = fix_minimum_signal(data)
			data = (data - dataDark)*Gain[index]
			data = fix_minimum_signal2(data)
			data_all.append(data)
	data_all = np.array(data_all)
	data_mean = np.mean(data_all,axis=0)
	# data_mean = fix_minimum_signal(data_mean)

	# result = 0
	# nLines = 8
	# while (result == 0 and nLines > 0):
	# 	try:
	# 		angle = get_angle(data_mean, nLines=nLines)
	# 		result = 1
	# 		angle = np.nansum(np.multiply(angle[0], np.divide(1, np.power(angle[1], 2)))) / np.nansum(
	# 			np.divide(1, np.power(angle[1], 2)))
	# 	except:
	# 		nLines -= 1
	# print(angle)
	# data_mean = rotate(data_mean, angle)
	#
	#result = 0
	#nLines = 8
	#tilt = (np.nan, np.nan, np.nan)
	#while ((result == 0 or np.isnan(tilt[1]) or np.isnan(tilt[2])) and nLines > 0):
	#	try:
	#		tilt = get_tilt(data_mean, nLines=nLines)
	#		print(tilt)
	#		result = 1
	#		geom['binInterv'] = tilt[0]
	#		geom['tilt'] = tilt[1]
	#		geom['bin00a'] = tilt[2]
	#		geom['bin00b'] = tilt[2]
	#		nLines -= 1
	#	except:
	#		nLines -= 1
	#print(tilt)
	#
	#data_mean = do_tilt(data_mean,geom['tilt'])
	plt.figure();plt.plot(np.sum(data_mean,axis=0));plt.semilogy(),plt.pause(0.01)

#### REALLY BAD: 	the hydrogen lamp measurement does not gives me a reliabe wavelength calibration.
#### 				the only information beside the location of the balmer beta line is the interval of bins, that I can use to get the magnification

balmer_beta_calib = 1194
binInterv_calib = 28.172469635627522





peaks_new = np.array([1268, 780, 558, 436, 362])
waveLcoefs_new = np.polyfit(peaks_new,waveLengths[:len(peaks_new)],2)
waveLcoefs_new = np.array([ -1.39629946e-06,   1.09550955e-01,   3.49466935e+02])
balmer_beta_new = peaks_new[0]
binInterv_new = 28.122847503373812


#### THE BIN INTERVAL IS ABOUT THE SAME!
#### I NEED JUST TO SHIFT IT

shift = peaks_new[0]-balmer_beta_calib



longCal_new = np.zeros_like(longCal)
longCal_new[:,:shift] = longCal_no_mask[:,:shift]
longCal_new[:,shift:] =longCal_mask [:,:-shift]+best_intensity_shift/40
longCal = copy.deepcopy(longCal_new)




topDark = longCal[0]
botDark = longCal[-1]
wtDark = np.arange(1,41)/41
calBinned = longCal[1:41] -(np.outer(wtDark,topDark)+np.outer(wtDark[::-1],botDark)) +(topDark+botDark)/2	#	MODIFIED TO AVOID NEGATIVE VALUES


# Due to the low signal I smooth the curve for a better numerical stability

averaging = 10
extrapolation = 100
averaging = int(averaging/2)*2
for index,chord in enumerate(calBinned):
	temp = np.convolve(chord, np.ones(averaging)/averaging , mode='same')
	fit = np.polyfit(range(-extrapolation,0,1), chord[-extrapolation:], 2)
	fit[-1]=temp[-averaging//2]
	temp[-averaging//2:] = np.polyval(fit, range(-averaging//2,0,1))
	fit = np.polyfit(range(0,extrapolation,1), chord[:extrapolation], 2)
	fit[-1]=temp[averaging//2]
	temp[:averaging//2] = np.polyval(fit, range(0,averaging//2,1))
	calBinned[index] = temp


averaging = 3
extrapolation = 5
averaging = int(averaging/2)*2
for index,chord in enumerate(calBinned.T):
	temp = np.convolve(chord, [1/4,2/4,1/4] , mode='same')
	fit = np.polyfit(range(-extrapolation,0,1), chord[-extrapolation:], 2)
	fit[-1]=temp[-averaging//2]
	temp[-averaging//2:] = np.polyval(fit, range(-averaging//2,0,1))
	fit = np.polyfit(range(0,extrapolation,1), chord[:extrapolation], 2)
	fit[-1]=temp[averaging//2]
	temp[:averaging//2] = np.polyval(fit, range(0,averaging//2,1))
	calBinned[:,index] = temp
calBinned = calBinned - min(calBinned.min(),0)+1



spline=interpolate.interp1d(df_sphere.waveL,df_sphere.Rad,kind='cubic') #Spline interp of sphere radiance

#	THIS IS NOT NECESSARY ANYMORE BECAUSE i AVOID NEGATIVE VALUES
# calBinned_min= np.max(calBinned)/1000
# for irow, row in enumerate(calBinned):
# 	for icol, value in enumerate(row):
# 		if (value < calBinned_min):
# 			calBinned[irow, icol] =  calBinned_min

to_save = np.zeros((3,)+np.shape(binned_data_no_mask))



to_save[1] = calBinned / spline(np.polyval(waveLcoefs_new,np.arange(data.shape[1])))

# np.save('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/Sensitivity_2',to_save)

plt.figure();plt.plot(to_save[1][20]);plt.pause(0.001)
plt.xlabel('wavelength axis [au]')
plt.ylabel('intensity [au]')
plt.pause(0.001)


data_all = []
for merge_ID_target in range(40,41):
	all_j=find_index_of_file(merge_ID_target,df_settings,df_log)
	for j in all_j:
		dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
		(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
		type = '.tif'
		filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)
		type = '.txt'
		filename_metadata = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)[0]

		# plt.figure()
		# data_all = []
		for index, filename in enumerate(filenames):
			fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + \
					filename
			im = Image.open(fname)
			data = np.array(im)
			# data = fix_minimum_signal(data)
			data = data - dataDark
			# data = fix_minimum_signal(data)
			data_all.append(data)
		data_all = np.array(data_all)
data_mean = np.mean(data_all,axis=0)
tilt = np.array([ 28.08337189,          np.nan,  46.13475998]) # FROM analysis_filtered_data.py
binned_data = binData(data_mean, tilt[-1], tilt[0])

plt.figure();plt.plot(np.mean(binned_data/to_save[1],axis=0));plt.pause(0.01)
plt.figure();plt.plot(np.mean(binned_data,axis=0));plt.pause(0.01)
plt.figure();plt.plot(np.mean(to_save[1],axis=0));plt.pause(0.01)


np.save('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/Sensitivity_2',to_save)







#####	THIS BIT IS FOR THE EXPERIMENTS 32 TO 39 AND 54


balmer_beta_calib = 1392
binInterv_calib = 29.3628

waveLcoefs_calib = [3.06707869e-07  , 1.01705766e-01  , 3.43412817e+02]



peaks_new = np.array([1268, 780, 558, 436, 362])
waveLcoefs_new = np.polyfit(peaks_new,waveLengths[:len(peaks_new)],2)
waveLcoefs_new = np.array([ -1.39629946e-06,   1.09550955e-01,   3.49466935e+02])
balmer_beta_new = peaks_new[0]
binInterv_new = 28.122847503373812




longCal_no_mask_2 = resize(longCal_no_mask, ( np.shape(longCal_no_mask)[0], int(np.shape(longCal_no_mask)[1]*binInterv_new/binInterv_calib)), order=1)



balmer_beta_calib = int(balmer_beta_calib*binInterv_new/binInterv_calib)


shift = peaks_new[0]-balmer_beta_calib


longCal_new = np.zeros_like(longCal)
if shift >=0:
	longCal_new[:,:shift] = longCal_no_mask[:,:shift]
	longCal_new[:,shift:] =longCal_mask [:,:-shift]+best_intensity_shift/40
	longCal = copy.deepcopy(longCal_new)
else:
	longCal_new[:,:shift-(len(longCal[0])-len(longCal_no_mask_2[0]))] = longCal_no_mask_2[:,-shift:]
	extrapolation = 300
	for index, chord in enumerate(longCal_no_mask_2):
		fit = np.polyfit(range( -extrapolation, 0, 1), chord[-extrapolation:], 2)
		longCal_new[index,shift-(len(longCal[0])-len(longCal_no_mask_2[0])):] = np.polyval(fit, range(0, np.abs(shift-(len(longCal[0])-len(longCal_no_mask_2[0]))), 1))


plt.figure();plt.plot(longCal_no_mask[20]),plt.pause(0.01)
plt.plot(longCal_no_mask_2[20]),plt.pause(0.01)
plt.plot(longCal_new[20]),plt.pause(0.01)

topDark = longCal[0]
botDark = longCal[-1]
wtDark = np.arange(1,41)/41
calBinned = longCal[1:41] -(np.outer(wtDark,topDark)+np.outer(wtDark[::-1],botDark)) +(topDark+botDark)/2	#	MODIFIED TO AVOID NEGATIVE VALUES


# Due to the low signal I smooth the curve for a better numerical stability

averaging = 10
extrapolation = 100
averaging = int(averaging/2)*2
for index,chord in enumerate(calBinned):
	temp = np.convolve(chord, np.ones(averaging)/averaging , mode='same')
	fit = np.polyfit(range(-extrapolation,0,1), chord[-extrapolation:], 2)
	fit[-1]=temp[-averaging//2]
	temp[-averaging//2:] = np.polyval(fit, range(-averaging//2,0,1))
	fit = np.polyfit(range(0,extrapolation,1), chord[:extrapolation], 2)
	fit[-1]=temp[averaging//2]
	temp[:averaging//2] = np.polyval(fit, range(0,averaging//2,1))
	calBinned[index] = temp


averaging = 3
extrapolation = 5
averaging = int(averaging/2)*2
for index,chord in enumerate(calBinned.T):
	temp = np.convolve(chord, [1/4,2/4,1/4] , mode='same')
	fit = np.polyfit(range(-extrapolation,0,1), chord[-extrapolation:], 2)
	fit[-1]=temp[-averaging//2]
	temp[-averaging//2:] = np.polyval(fit, range(-averaging//2,0,1))
	fit = np.polyfit(range(0,extrapolation,1), chord[:extrapolation], 2)
	fit[-1]=temp[averaging//2]
	temp[:averaging//2] = np.polyval(fit, range(0,averaging//2,1))
	calBinned[:,index] = temp
calBinned = calBinned - min(calBinned.min(),0)+1


spline=interpolate.interp1d(df_sphere.waveL,df_sphere.Rad,kind='cubic') #Spline interp of sphere radiance

#	THIS IS NOT NECESSARY ANYMORE BECAUSE i AVOID NEGATIVE VALUES
# calBinned_min= np.max(calBinned)/1000
# for irow, row in enumerate(calBinned):
# 	for icol, value in enumerate(row):
# 		if (value < calBinned_min):
# 			calBinned[irow, icol] =  calBinned_min

to_save = np.zeros((3,)+np.shape(binned_data_no_mask))



to_save[1] = calBinned / spline(np.polyval(waveLcoefs_new,np.arange(data.shape[1])))

np.save('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/Sensitivity_3',to_save)






#####	THIS BIT IS FOR THE EXPERIMENTS 66 onwards

limits_angle = [66, 84]
limits_tilt = [66, 84]
limits_wave = [66, 84]
figure_number = 0
where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_5/Sensitivity_4_plots_'

# geom_null = pd.DataFrame(columns=['angle', 'tilt', 'binInterv', 'bin00a', 'bin00b'])
# geom_null.loc[0] = [0, 0, 0, 0, 0]
#
# if np.sum(limits_angle) != 0:
# 	angle = []
# 	for merge in range(limits_angle[0], limits_angle[1] + 1):
# 		all_j = find_index_of_file(merge, df_settings, df_log)
# 		for j in all_j:
# 			dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
# 			(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
# 			type = '.tif'
# 			filenames = all_file_names(
# 				fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
# 			type = '.txt'
# 			filename_metadata = all_file_names(
# 				fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)[
# 				0]
# 			(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir, folder,
# 																										 sequence,
# 																										 untitled,
# 																										 filename_metadata)
#
# 			# plt.figure()
# 			data_all = []
# 			for index, filename in enumerate(filenames):
# 				fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(
# 					untitled) + '/Pos0/' + \
# 						filename
# 				im = Image.open(fname)
# 				data = np.array(im)
# 				data = (data - dataDark) * Gain[index]
# 				data = fix_minimum_signal2(data)
# 				data_all.append(data)
# 			data_all = np.array(data_all)
# 			data_mean = np.mean(data_all, axis=0)
# 			try:
# 				angle.append(get_angle_2(data_mean, nLines=2))
# 				print(angle)
# 			except:
# 				print('FAILED')
# 	angle = np.array(angle)
# 	print([angle])
# 	angle = np.nansum(angle[:, 0] / (angle[:, 1] ** 2)) / np.nansum(1 / angle[:, 1] ** 2)
#
# if np.sum(limits_tilt) != 0:
# 	tilt = []
# 	for merge in range(limits_tilt[0], limits_tilt[1] + 1):
# 		all_j = find_index_of_file(merge, df_settings, df_log)
# 		for j in all_j:
# 			dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
# 			(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
# 			type = '.tif'
# 			filenames = all_file_names(
# 				fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
# 			type = '.txt'
# 			filename_metadata = all_file_names(
# 				fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)[
# 				0]
# 			(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir, folder,
# 																										 sequence,
# 																										 untitled,
# 																										 filename_metadata)
#
# 			# plt.figure()
# 			data_all = []
# 			for index, filename in enumerate(filenames):
# 				fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(
# 					untitled) + '/Pos0/' + \
# 						filename
# 				im = Image.open(fname)
# 				data = np.array(im)
# 				data = (data - dataDark) * Gain[index]
# 				data = fix_minimum_signal2(data)
# 				data_all.append(data)
# 			data_all = np.array(data_all)
# 			data_mean = np.mean(data_all, axis=0)
# 			data_mean = rotate(data_mean, angle)
# 			try:
# 				tilt.append(get_tilt(data_mean, nLines=3))
# 				print(tilt)
# 			except:
# 				print('FAILED')
# 	tilt = np.array(tilt)
# 	print([tilt])
# 	tilt = np.nanmedian(tilt, axis=0)
#
# geom = pd.DataFrame(columns=['angle', 'tilt', 'binInterv', 'bin00a', 'bin00b'])
# geom.loc[0] = [angle, tilt[1], tilt[0], tilt[2], tilt[2]]
# geom_store = copy.deepcopy(geom)
# print(geom)
#
# if np.sum(limits_wave) != 0:
# 	waveLcoefs = []
# 	for merge in range(limits_wave[0], limits_wave[1] + 1):
# 		try:
# 			waveLcoefs.append(do_waveL_Calib(merge, df_settings, df_log, geom, fdir=fdir))
# 		except:
# 			print('FAILED')
# 	waveLcoefs = np.array(waveLcoefs)
# 	print([waveLcoefs])
# 	waveLcoefs = np.nanmedian(waveLcoefs, axis=0)


geom = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
geom.loc[0] = [ 0.808817942408, np.nan, np.nan, np.nan, np.nan]
waveLcoefs = np.ones((2, 3)) * np.nan
waveLcoefs[1] = [ 2.84709039e-06 ,  1.19975728e-01 ,  3.18673899e+02]
geom_null = pd.DataFrame(columns=['angle', 'tilt', 'binInterv', 'bin00a', 'bin00b'])
geom_null.loc[0] = [0, 0, 0, 0, 0]

# int_time = [0.01, 0.1, 1, 10, 100]
# calib_ID=[214, 215, 216, 217, 218]
int_time_long = [0.1, 1, 10, 100]
calib_ID_long=[215, 216, 217, 218]
calib_ID_long = np.array([calib_ID_long for _, calib_ID_long in sorted(zip(int_time_long, calib_ID_long))])
int_time_long = np.sort(int_time_long)
int_time_long_new = np.flip(int_time_long,axis=0)
calib_ID_long_new = np.flip(calib_ID_long,axis=0)

int_time = [0.01]
data_all = []
calib_ID=[214]
for iFile, j in enumerate(calib_ID):
	(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
	dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
	type = '.tif'
	filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
	type = '.txt'
	temp = []
	for index, filename in enumerate(filenames):
		fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
		im = Image.open(fname)
		data = np.array(im)
		data = (data - dataDark)
		data = fix_minimum_signal2(data)

		temp.append(data)
	temp = np.mean(temp, axis=0)
	data = copy.deepcopy(temp)
	data = data/(df_log.loc[j, ['Calib_ID']][0]*1000000)
	data_all.append(data)

data_all = np.array(data_all)
data=np.mean(data_all,axis=(0))
data = rotate(data, geom['angle'])
data = do_tilt_no_lines(data)
data=np.sum(data,axis=0)
data=np.convolve(data, np.ones(200)/200 , mode='valid')
wavel=data.argmax()+100


wavel_range = 100
# wavelength_780=1266
# wavelength_750=1556

# wavel = 1338
wavelength_950=wavel
target_up = np.polyval(waveLcoefs[1], wavelength_950+wavel_range)
target_down = np.polyval(waveLcoefs[1], wavelength_950-wavel_range)


bin_interv_high_int_time = []
data_all = []
binned_all = []
peak_sum = []
longCal_all=[]
for iFile, j in enumerate(calib_ID_long):
	(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]

	dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)

	type = '.tif'
	filenames = all_file_names(
		fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
	type = '.txt'
	filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0',type)[0]
	(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir,folder,sequence,untitled,filename_metadata)

	data_sum = 0
	temp = []



	for index, filename in enumerate(filenames):
		fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(
			untitled) + '/Pos0/' + filename
		# print(fname)
		im = Image.open(fname)
		data = np.array(im)
		data = (data - dataDark)*Gain[index]
		# data = rotate(data, geom['angle'])
		# data = do_tilt(data, geom['tilt'])
		data,fix_done_marker = fix_minimum_signal2(data,fix_done_marker=True)

		if fix_done_marker == True:
			continue

		data_sum += data
		temp.append(data)

	if np.max(data_sum) == 0:
		int_time_long_new = int_time_long_new[:-1]
		calib_ID_long_new = calib_ID_long_new[:-1]
		print('The integration time ' + str(int_time_long[iFile]) + 'ms will not be used because it requires minimum value corrrection to be used, and therefore it is deemed as unreliable')
		continue

	temp = np.mean(temp, axis=0)

	# if (df_log.loc[j, ['t_exp']][0]==0.01):
	# temp = fix_minimum_signal(temp)
		# noise = temp[:20].tolist() + temp[-20:].tolist()
		# noise_std = np.std(noise, axis=(0, 1))
		# noise = np.mean(noise, axis=(0, 1))
		# test_std = 10000
		# factor_std = 0
		# std_record = []
		# while factor_std <= 2:
		# 	reconstruct = []
		# 	for irow, row in enumerate(temp.tolist()):
		# 		minimum = np.mean(np.sort(row)[:int(len(row) / 40)]) + factor_std * np.std(
		# 			np.sort(row)[:int(len(row) / 40)])
		# 		# minimum = np.min(row)
		# 		reconstruct.append(np.array(row) + noise - minimum)
		# 	reconstruct = np.array(reconstruct)
		# 	test_std = np.std(reconstruct[:, :40])
		# 	std_record.append(test_std)
		# 	print('factor_std ' + str(factor_std))
		# 	print('test_std ' + str(test_std))
		# 	factor_std += 0.1
		# factor_std = np.array(std_record).argmin()*0.1
		# reconstruct = []
		# for irow, row in enumerate(temp.tolist()):
		# 	minimum = np.mean(np.sort(row)[:int(len(row) / 40)]) + factor_std * np.std(
		# 		np.sort(row)[:int(len(row) / 40)])
		# 	# minimum = np.min(row)
		# 	reconstruct.append(np.array(row) + noise - minimum)
		# temp=reconstruct

	# temp = temp - dataDark
	# angle = get_angle_no_lines(temp)	# I cant use this because it's impossible to decouple the angle error from the tilt.
	temp = rotate(temp, geom['angle'])
	temp = do_tilt_no_lines(temp)

	first_bin, bin_interv = get_bin_and_interv_no_lines(temp)
	print('bin interval of ' + str(bin_interv) + ' , first_bin of ' + str(first_bin))
	bin_interv_high_int_time.append(bin_interv)
	binned_data = binData(temp, first_bin, bin_interv)/(df_log.loc[j, ['Calib_ID']][0]*1000000)
	longCal_all.append( binData(temp, first_bin - bin_interv, bin_interv, nCh=42)/(df_log.loc[j, ['Calib_ID']][0]*1000000))
	# binned_data = binned_data - min(np.min(binned_data), 0)
	# angle=get_angle(temp,no_line_present=True)
	# temp = rotate(temp, angle)
	# tilt = get_tilt(temp)
	# temp = do_tilt(temp, tilt)

	# temp = temp - dataDark
	# angle=get_angle(temp)
	# temp = rotate(temp, angle)
	# tilt = get_tilt(temp)
	# temp = do_tilt(temp, tilt)
	data_all.append(temp)
	binned_all.append(binned_data)

	# peak_sum.append(np.sum(binned_data[:,wavel - wavel_range:wavel + wavel_range],axis=(-1,-2)))
	peak_sum.append(np.sum(binned_data[:, wavel - wavel_range:wavel + wavel_range], axis=(-1)))
peak_sum=np.array(peak_sum)

int_time_long = np.flip(int_time_long_new,axis=0)
calib_ID_long = np.flip(calib_ID_long_new,axis=0)

max_coordinate = 0
for index in range(len(data_all)):
	max_coordinate = max(max_coordinate,np.shape(data_all[index])[0])
for index in range(len(data_all)):
	while np.shape(data_all[index])[0]<max_coordinate:
		temp_1=data_all[index].tolist()
		temp_1.append(np.zeros(np.shape(temp_1[0])))
		data_all[index] = np.array(temp_1)
	plt.figure()
	plt.imshow(data_all[index], 'rainbow', origin='lower')
	plt.title('Exp ' + str(limits_angle) + '\n '+str(int_time_long[index])+'int time sensitivity')
	plt.xlabel('wavelength axis [pixels]')
	plt.ylabel('LOS axis [pixels]')
	figure_number += 1
	plt.savefig(where_to_save_everything + str(figure_number) + '.eps', bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)
data_all_long_int_time=np.array(data_all)
binned_all=np.array(binned_all)
bin_interv_high_int_time = np.nanmean(bin_interv_high_int_time)

calBinned_peak_sum=[]
calBinned_all=[]
longCal_all=np.array(longCal_all)
for i in range(len(binned_all)):
	topDark = longCal_all[i][0]
	botDark = longCal_all[i][-1]
	wtDark = np.arange(1, 41) / 41
	calBinned = longCal_all[i][1:41] - (np.outer(wtDark, topDark) + np.outer(wtDark[::-1], botDark)) + (topDark + botDark) / 2  # MODIFIED TO AVOID NEGATIVE VALUES
	# calBinned_peak_sum.append(np.sum(calBinned[:,wavel - wavel_range:wavel + wavel_range],axis=(-1,-2)))
	calBinned_peak_sum.append(np.sum(calBinned[:, wavel - wavel_range:wavel + wavel_range], axis=(-1)))
	calBinned_all.append(calBinned)
calBinned_peak_sum = np.array(calBinned_peak_sum)
calBinned_all=np.array(calBinned_all)


int_time = [0.01]
data_all = []
calib_ID=[214]
for iFile, j in enumerate(calib_ID):
	(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]

	dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)

	type = '.tif'
	filenames = all_file_names(
		fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
	type = '.txt'
	filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0',type)[0]
	(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir,folder,sequence,untitled,filename_metadata)

	data_sum = 0
	temp = []
	for index, filename in enumerate(filenames):
		fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
		# print(fname)
		im = Image.open(fname)
		data = np.array(im)
		data = (data - dataDark)*Gain[index]

		# data = rotate(data, geom['angle'])
		# data = do_tilt(data, geom['tilt'])
		data = fix_minimum_signal2(data)

		data_sum += data
		temp.append(data)
	temp = np.mean(temp, axis=0)
	# temp = fix_minimum_signal(temp)

	# if (df_log.loc[j, ['t_exp']][0]==0.01):	REMOVED BECAUSE I REALISED THAT THE SHIFTING OF THE MINUMIM HAPPENS FOR ALL CONDITIONS

		# noise = temp[:20].tolist() + temp[-20:].tolist()
		# noise_std = np.std(noise, axis=(0, 1))
		# noise = np.mean(noise, axis=(0, 1))
		# factor_std = 0
		# std_record = []
		# while factor_std <= 2:
		# 	reconstruct = []
		# 	for irow, row in enumerate(temp.tolist()):
		# 		minimum = np.mean(np.sort(row)[:int(len(row) / 40)]) + factor_std * np.std(
		# 			np.sort(row)[:int(len(row) / 40)])
		# 		# minimum = np.min(row)
		# 		reconstruct.append(np.array(row) + noise - minimum)
		# 	reconstruct = np.array(reconstruct)
		# 	test_std = np.std(reconstruct[:, :40])
		# 	std_record.append(test_std)
		# 	print('factor_std ' + str(factor_std))
		# 	print('test_std ' + str(test_std))
		# 	factor_std += 0.1
		# factor_std = np.array(std_record).argmin()*0.1
		# reconstruct = []
		# for irow, row in enumerate(temp.tolist()):
		# 	minimum = np.mean(np.sort(row)[:int(len(row) / 40)]) + factor_std * np.std(
		# 		np.sort(row)[:int(len(row) / 40)])
		# 	# minimum = np.min(row)
		# 	reconstruct.append(np.array(row) + noise - minimum)
		# temp=reconstruct

	# temp = temp - dataDark
	# angle = get_angle_no_lines(temp)	# I cant use this because it's impossible to decouple the angle error from the tilt.
	# temp = rotate(temp, geom['angle'])
	# temp = do_tilt_no_lines(temp)

	# profiles_for_peaks = np.sum(temp[:, -100:-1], axis=(-1))
	# profiles = np.sum(temp[:, wavel-wavel_range:wavel+wavel_range], axis=(-1))
	# profiles_for_peaks=profiles
	#
	# from scipy.signal import find_peaks, peak_prominences as get_proms
	#
	# prominence = 20
	# num_peaks = 0
	# while num_peaks != 40:
	# 	peaks = find_peaks(profiles_for_peaks, prominence=prominence,distance=25)[0]
	# 	temp = []
	# 	for value in peaks:
	# 		if (value > 5 and value < len(profiles_for_peaks) - 5):
	# 			temp.append(value)
	# 	peaks = np.array(temp)
	# 	num_peaks = len(peaks)
	# 	print(np.max(profiles_for_peaks[peaks]))
	# 	print(num_peaks)
	# 	if num_peaks < 40:
	# 		prominence -= 0.1
	# 	elif num_peaks > 40:
	# 		prominence += 0.1
	#
	# seak_sum_single=0
	# skip=-10
	# while skip<11:
	# 	seak_sum_single+=np.sum(profiles[peaks+skip])
	# 	skip+=1

	data = copy.deepcopy(temp)
	data = data/(df_log.loc[j, ['Calib_ID']][0]*1000000)
	data_all.append(data)

data_all = np.array(data_all)
data=np.mean(data_all,axis=(0))
data = rotate(data, geom['angle'])
data = do_tilt_no_lines(data)

plt.figure()
plt.imshow(data,'rainbow',origin='lower')
plt.title('Exp '+str(limits_angle)+'\n Low int time sensitivity')
plt.xlabel('wavelength axis [pixels]')
plt.ylabel('LOS axis [pixels]')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)

# data = data - min(np.min(data),0)

# max_coordinate = 0
# for index in range(len(data_all)):
# 	max_coordinate = max(max_coordinate,np.shape(data_all[index])[0])
# for index in range(len(data_all)):
# 	while np.shape(data_all[index])[0]<max_coordinate:
# 		temp_1=data_all[index].tolist()
# 		temp_1.append(np.zeros(np.shape(temp_1[0])))
# 		data_all[index] = np.array(temp_1)
# data_all=np.array(data_all)
# data=np.mean(data_all,axis=(0))

data_no_mask = copy.deepcopy(data)
# data_no_mask = data_no_mask - min(np.min(data_no_mask),0)
first_bin, bin_interv = get_bin_and_interv_no_lines(data)
print('bin interval of ' + str(bin_interv)+' , first_bin of '+str(first_bin))
bin_interv_no_mask = bin_interv
binned_data_no_mask = binData(data, first_bin, bin_interv)
# binned_data_no_mask = binned_data_no_mask - min(np.min(binned_data_no_mask),0)
longCal_no_mask = binData(data,first_bin-bin_interv,bin_interv,nCh = 42)
topDark_no_mask = longCal_no_mask[0]
botDark_no_mask = longCal_no_mask[-1]
wtDark_no_mask = np.arange(1, 41) / 41
calBinned_no_mask = longCal_no_mask[1:41] - (np.outer(wtDark_no_mask, topDark_no_mask) + np.outer(wtDark_no_mask[::-1], botDark_no_mask)) + (topDark_no_mask + botDark_no_mask) / 2  # MODIFIED TO AVOID NEGATIVE VALUES

# peak_sum_single = np.sum(binned_data_no_mask[:, wavel - wavel_range:wavel + wavel_range], axis=(-1, -2))  # type: ndarray
# calBinned_peak_sum_single = np.sum(calBinned_no_mask[:, wavel - wavel_range:wavel + wavel_range], axis=(-1, -2))  # type: ndarray
peak_sum_single = np.sum(binned_data_no_mask[:, wavel - wavel_range:wavel + wavel_range], axis=(-1))  # type: ndarray
calBinned_peak_sum_single = np.sum(calBinned_no_mask[:, wavel - wavel_range:wavel + wavel_range], axis=(-1))  # type: ndarray



max_coordinate=0
plt.figure()
scaled_down_binned_all=[]
for i in range(len(binned_all)):
	# scaled_down_binned_all.append(binned_all[i]*peak_sum_single/peak_sum[i])
	scaled_down_binned_all.append((binned_all[i].T *( peak_sum_single / peak_sum[i])).T)
	plt.plot(np.mean(scaled_down_binned_all[-1],axis=0),label=str(int_time_long[i]))
	max_coordinate = max(max_coordinate, np.max(np.mean(scaled_down_binned_all[-1],axis=0)))
scaled_down_binned_all=np.array(scaled_down_binned_all)

plt.plot(np.mean(binned_data_no_mask,axis=0),label='0.01')
max_coordinate = max(max_coordinate, np.max(np.mean(binned_data_no_mask,axis=0)))
plt.title('Exp '+str(limits_angle)+'\n Comparison of scaled_down_binned_all averaged over all bins and scaled around column '+str(wavel)+' for different integration times')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.plot([wavel,wavel],[0,max_coordinate],'--k')


averaged_scaled_down_binned_all=np.mean(scaled_down_binned_all,axis=0)
plt.plot(np.mean(averaged_scaled_down_binned_all,axis=0),label='Averaged')


plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)



# plt.figure()
# for i in range(len(binned_all)):
# 	plt.plot(scaled_down_binned_all[i,30],label=str(int_time_long[i]))
# plt.plot(binned_data_no_mask[30],label='0.01')
# plt.xlabel('wavelength axis [au]')
# plt.ylabel('sensitivity [au]')
# plt.title('scaled_down_binned_all comparison at bin 30')
# plt.plot(averaged_scaled_down_binned_all[30],label='Averaged')
# plt.legend(loc='best')
# plt.pause(0.01)
#
#
# plt.figure()
# for i in range(len(binned_all)):
# 	plt.plot(scaled_down_binned_all[i,20],label=str(int_time_long[i]))
# plt.plot(binned_data_no_mask[20],label='0.01')
# plt.xlabel('wavelength axis [au]')
# plt.ylabel('sensitivity [au]')
# plt.title('scaled_down_binned_all comparison at bin 20')
# plt.plot(averaged_scaled_down_binned_all[20],label='Averaged')
# plt.legend(loc='best')
# plt.pause(0.01)
#
#
# plt.figure()
# for i in range(len(binned_all)):
# 	plt.plot(np.sum(scaled_down_binned_all[i,:,1000-100:1000+100],axis=(-1)),label=str(int_time_long[i]))
# plt.plot(np.sum(binned_data_no_mask[:,1000-100:1000+100],axis=(-1)),label='0.01')
# plt.xlabel('wavelength axis [au]')
# plt.ylabel('sensitivity [au]')
# plt.title('scaled_down_binned_all comparison at 1000 +/-100')
# plt.plot(np.sum(averaged_scaled_down_binned_all[:,1000-100:1000+100],axis=(-1)),label='Averaged')
# plt.legend(loc='best')
# plt.pause(0.01)
#
#
#
# plt.figure()
# plt.imshow(averaged_scaled_down_binned_all, 'rainbow',origin='lower',aspect=10)
# plt.title('scaled_down_binned_all')
# plt.colorbar()
# plt.pause(0.01)



max_coordinate=0
plt.figure()
scaled_down_calBinned=[]
for i in range(len(binned_all)):
	# scaled_down_binned_all.append(binned_all[i]*peak_sum_single/peak_sum[i])
	scaled_down_calBinned.append((calBinned_all[i].T * (calBinned_peak_sum_single / calBinned_peak_sum[i])).T)
	plt.plot(np.mean(scaled_down_calBinned[-1],axis=0),label=str(int_time_long[i]))
	max_coordinate = max(max_coordinate, np.max(np.mean(scaled_down_calBinned[-1],axis=0)))
scaled_down_calBinned=np.array(scaled_down_calBinned)

plt.plot(np.mean(calBinned_no_mask,axis=0),label='0.01')
max_coordinate = max(max_coordinate, np.max(np.mean(calBinned_no_mask,axis=0)))
plt.title('Exp '+str(limits_angle)+'\n Comparison of scaled_down_calBinned averaged over all bins and scaled around column '+str(wavel)+' for different integration times')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.plot([wavel,wavel],[0,max_coordinate],'--k')


averaged_scaled_down_calBinned=np.mean(scaled_down_calBinned,axis=0)
averaged_scaled_down_calBinned[averaged_scaled_down_calBinned<=np.min(np.mean(averaged_scaled_down_calBinned,axis=0))]=np.min(np.mean(averaged_scaled_down_calBinned,axis=0))	#this is to erase negative or too low values at the borders
plt.plot(np.mean(averaged_scaled_down_calBinned,axis=0),label='Averaged')


plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)

# max_coordinate=0
# plt.figure()
# scaled_down_calBinned=[]
# for i in range(len(binned_all)):
# 	topDark = longCal_all[i][0]
# 	botDark = longCal_all[i][-1]
# 	wtDark = np.arange(1, 41) / 41
# 	calBinned = longCal_all[i][1:41] - (np.outer(wtDark, topDark) + np.outer(wtDark[::-1], botDark)) + (topDark + botDark) / 2  # MODIFIED TO AVOID NEGATIVE VALUES
# 	scaled_down_calBinned.append(calBinned*peak_sum_single/peak_sum[i])
# 	plt.plot(np.mean(calBinned*calBinned_peak_sum_single/calBinned_peak_sum[i],axis=0),label=str(int_time_long[i]))
# 	max_coordinate = max(max_coordinate, np.max(np.mean(calBinned*calBinned_peak_sum_single/calBinned_peak_sum[i],axis=0)))
# scaled_down_calBinned=np.array(scaled_down_calBinned)
#
#
# averaged_scaled_down_calBinned=np.mean(scaled_down_calBinned,axis=0)
# plt.plot(np.mean(calBinned_no_mask,axis=0),label='0.01')
# max_coordinate = max(max_coordinate, np.max(np.mean(calBinned_no_mask,axis=0)))
# plt.title('Comparison of the sensitivity averaged over all bins and scaled around column '+str(wavel)+' for different integration times')
# plt.xlabel('wavelength axis [au]')
# plt.ylabel('sensitivity [au]')
# plt.plot([wavel,wavel],[0,max_coordinate],'--k')
#
#
# averaged_scaled_down_calBinned=np.mean(scaled_down_calBinned,axis=0)
# averaged_scaled_down_calBinned[averaged_scaled_down_calBinned<=np.min(np.mean(averaged_scaled_down_calBinned,axis=0))]=np.min(np.mean(averaged_scaled_down_calBinned,axis=0))	#this is to erase negative or too low values at the borders
# plt.plot(np.mean(averaged_scaled_down_calBinned,axis=0),label='Averaged')
#
#
# plt.legend(loc='best')
# plt.pause(0.01)


plt.figure()
for i in range(len(binned_all)):
	plt.plot(np.sum(scaled_down_calBinned[i,:,1000-100:1000+100],axis=(-1))/np.max(np.sum(scaled_down_calBinned[i,:,1000-100:1000+100],axis=(-1))),label=str(int_time_long[i]))
plt.plot(np.sum(calBinned_no_mask[:,1000-100:1000+100],axis=(-1))/np.max(np.sum(calBinned_no_mask[:,1000-100:1000+100],axis=(-1))),label='0.01')
plt.title('Exp '+str(limits_angle)+'\n scaled_down_calBinned comparison at 1000 +/-100')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.plot(np.sum(averaged_scaled_down_calBinned[:,1000-100:1000+100],axis=(-1))/np.max(np.sum(averaged_scaled_down_calBinned[:,1000-100:1000+100],axis=(-1))),label='Averaged')
plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)


plt.figure()
for i in range(len(binned_all)):
	plt.plot(scaled_down_calBinned[i,30],label=str(int_time_long[i]))
plt.plot(calBinned_no_mask[30],label='0.01')
plt.title('Exp '+str(limits_angle)+'\n scaled_down_calBinned comparison for bin 30')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.plot(averaged_scaled_down_calBinned[30],label='Averaged')
plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)


plt.figure()
for i in range(len(binned_all)):
	plt.plot(scaled_down_calBinned[i,20],label=str(int_time_long[i]))
plt.plot(calBinned_no_mask[20],label='0.01')
plt.title('Exp '+str(limits_angle)+'\n scaled_down_calBinned comparison for bin 20')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.plot(averaged_scaled_down_calBinned[20],label='Averaged')
plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)



plt.figure()
plt.imshow(averaged_scaled_down_calBinned, 'rainbow',origin='lower',aspect=10)
plt.title('Exp '+str(limits_angle)+'\n scaled_down_calBinned')
plt.colorbar()
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)

plt.figure()
for i in range(len(binned_all)):
	plt.plot(np.sum(calBinned_all[i,:,1400-100:1400+100],axis=(-1))/np.max(np.sum(calBinned_all[i,:,1400-100:1400+100],axis=(-1))),label=str(int_time_long[i]))
plt.plot(np.sum(calBinned_no_mask[:,1400-100:1400+100],axis=(-1))/np.max(np.sum(calBinned_no_mask[:,1400-100:1400+100],axis=(-1))),label='0.01')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.title('Exp '+str(limits_angle)+'\n calBinned_all comparison for at 1400 +/-100')
plt.plot(np.sum(averaged_scaled_down_calBinned[:,1400-100:1400+100],axis=(-1))/np.max(np.sum(averaged_scaled_down_calBinned[:,1400-100:1400+100],axis=(-1))),label='Averaged')
plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)

plt.figure()
for i in range(len(binned_all)):
	plt.plot(np.sum(calBinned_all[i,:,1000-100:1000+100],axis=(-1))/np.max(np.sum(calBinned_all[i,:,1000-100:1000+100],axis=(-1))),label=str(int_time_long[i]))
plt.plot(np.sum(calBinned_no_mask[:,1000-100:1000+100],axis=(-1))/np.max(np.sum(calBinned_no_mask[:,1000-100:1000+100],axis=(-1))),label='0.01')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.title('Exp '+str(limits_angle)+'\n calBinned_all comparison for at 1000 +/-100')
plt.plot(np.sum(averaged_scaled_down_calBinned[:,1000-100:1000+100],axis=(-1))/np.max(np.sum(averaged_scaled_down_calBinned[:,1000-100:1000+100],axis=(-1))),label='Averaged')
plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)

plt.figure()
test_wavel = 1000
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise']
for i in range(len(binned_all)):
	plt.plot(calBinned_all[i,:,test_wavel]/np.max(calBinned_all[i,:,test_wavel]),color=color[i],linestyle='-',label=str(int_time_long[i]))
	plt.plot(calBinned_all[i, :, test_wavel+1] / np.max(calBinned_all[i, :, test_wavel+1]), color=color[i],linestyle='--', label=str(int_time_long[i]))
	plt.plot(calBinned_all[i, :, test_wavel+2] / np.max(calBinned_all[i, :, test_wavel+2]), color=color[i],linestyle='-.', label=str(int_time_long[i]))
plt.plot(calBinned_no_mask[:,test_wavel]/np.max(calBinned_no_mask[:,test_wavel]),color=color[i+1],linestyle='-',label='0.01')
plt.plot(calBinned_no_mask[:,test_wavel+1]/np.max(calBinned_no_mask[:,test_wavel+1]),color=color[i+1],linestyle='--',label='0.01')
plt.plot(calBinned_no_mask[:,test_wavel+2]/np.max(calBinned_no_mask[:,test_wavel+2]),color=color[i+1],linestyle='-.',label='0.01')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.title('Exp '+str(limits_angle)+'\n calBinned_all comparison at '+str(test_wavel)+', '+str(test_wavel+1)+', '+str(test_wavel+2))
plt.plot(averaged_scaled_down_calBinned[:,test_wavel]/np.max(averaged_scaled_down_calBinned[:,test_wavel]),color=color[i+2],linestyle='-',label='Averaged')
plt.plot(averaged_scaled_down_calBinned[:,test_wavel+1]/np.max(averaged_scaled_down_calBinned[:,test_wavel+1]),color=color[i+2],linestyle='--',label='Averaged')
plt.plot(averaged_scaled_down_calBinned[:,test_wavel+2]/np.max(averaged_scaled_down_calBinned[:,test_wavel+2]),color=color[i+2],linestyle='-.',label='Averaged')
plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)


plt.figure()
test_wavel = 1400
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise']
for i in range(len(binned_all)):
	plt.plot(calBinned_all[i,:,test_wavel]/np.max(calBinned_all[i,:,test_wavel]),color=color[i],linestyle='-',label=str(int_time_long[i]))
	plt.plot(calBinned_all[i, :, test_wavel+1] / np.max(calBinned_all[i, :, test_wavel+1]), color=color[i],linestyle='--', label=str(int_time_long[i]))
	plt.plot(calBinned_all[i, :, test_wavel+2] / np.max(calBinned_all[i, :, test_wavel+2]), color=color[i],linestyle='-.', label=str(int_time_long[i]))
plt.plot(calBinned_no_mask[:,test_wavel]/np.max(calBinned_no_mask[:,test_wavel]),color=color[i+1],linestyle='-',label='0.01')
plt.plot(calBinned_no_mask[:,test_wavel+1]/np.max(calBinned_no_mask[:,test_wavel+1]),color=color[i+1],linestyle='--',label='0.01')
plt.plot(calBinned_no_mask[:,test_wavel+2]/np.max(calBinned_no_mask[:,test_wavel+2]),color=color[i+1],linestyle='-.',label='0.01')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.title('Exp '+str(limits_angle)+'\n calBinned_all comparison at '+str(test_wavel)+', '+str(test_wavel+1)+', '+str(test_wavel+2))
plt.plot(averaged_scaled_down_calBinned[:,test_wavel]/np.max(averaged_scaled_down_calBinned[:,test_wavel]),color=color[i+2],linestyle='-',label='Averaged')
plt.plot(averaged_scaled_down_calBinned[:,test_wavel+1]/np.max(averaged_scaled_down_calBinned[:,test_wavel+1]),color=color[i+2],linestyle='--',label='Averaged')
plt.plot(averaged_scaled_down_calBinned[:,test_wavel+2]/np.max(averaged_scaled_down_calBinned[:,test_wavel+2]),color=color[i+2],linestyle='-.',label='Averaged')
plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)

'''
# # I'll use the quadratic fit because seems to fit better
# iFit = np.polyval(fit_2, 0.01)
# scaling = iFit / peak_sum_single
# data = data * scaling


longCal = binData(data,first_bin-bin_interv,bin_interv,nCh = 42)
topDark = longCal[0]
botDark = longCal[-1]
wtDark = np.arange(1,41)/41
calBinned = longCal[1:41] -(np.outer(wtDark,topDark)+np.outer(wtDark[::-1],botDark)) +(topDark+botDark)/2	#	MODIFIED TO AVOID NEGATIVE VALUES


# Due to the low signal I smooth the curve for a better numerical stability

averaging = 50
extrapolation = 100
averaging = int(averaging/2)*2
for index,chord in enumerate(calBinned):
	temp = np.convolve(chord, np.ones(averaging)/averaging , mode='same')
	fit = np.polyfit(range(-extrapolation,0,1), chord[-extrapolation:], 2)
	fit[-1]=temp[-averaging//2]
	temp[-averaging//2:] = np.polyval(fit, range(-averaging//2,0,1))
	fit = np.polyfit(range(0,extrapolation,1), chord[:extrapolation], 2)
	fit[-1]=temp[averaging//2]
	temp[:averaging//2] = np.polyval(fit, range(0,averaging//2,1))
	calBinned[index] = temp

averaging = 3
extrapolation = 4
# averaging = int(averaging/2)*2
for index,chord in enumerate(calBinned.T):
	# temp = np.convolve(chord, np.ones(averaging)/averaging , mode='same')
	temp = np.convolve(chord, [1/4,2/4,1/4], mode='same')
	fit = np.polyfit(range(-extrapolation+2,2,1), chord[-extrapolation:], 2)
	# plt.figure()
	# plt.plot(range(-extrapolation+2,2,1),chord[-extrapolation:])
	# plt.plot(range(-extrapolation+2,2,1),np.polyval(fit, range(-extrapolation+2,2,1)))
	# plt.pause(0.01)
	fit[-1]=temp[-2]
	temp[-1] = np.polyval(fit, 1)
	fit = np.polyfit(range(0,extrapolation,1), chord[:extrapolation], 2)
	fit[-1]=temp[np.ceil(averaging/2).astype('int')]
	# fit[-1]=temp[averaging//2]
	temp[0] = np.polyval(fit, -1)
	calBinned[:,index] = temp
calBinned = calBinned - min(calBinned.min(),0)+1
'''

spline=interpolate.interp1d(df_sphere.waveL,df_sphere.Rad,kind='cubic') #Spline interp of sphere radiance

#	THIS IS NOT NECESSARY ANYMORE BECAUSE i AVOID NEGATIVE VALUES
# calBinned_min= np.max(calBinned)/1000
# for irow, row in enumerate(calBinned):
# 	for icol, value in enumerate(row):
# 		if (value < calBinned_min):
# 			calBinned[irow, icol] =  calBinned_min




# averaged_scaled_down_calBinned[averaged_scaled_down_calBinned==0]=np.min(averaged_scaled_down_calBinned[averaged_scaled_down_calBinned!=0])
# spline=interpolate.interp1d(df_sphere.waveL,df_sphere.Rad,kind='cubic') #Spline interp of sphere radiance
# calibration = calBinned_no_mask / spline(np.polyval(waveLcoefs[1],np.arange(calBinned_no_mask.shape[1])))
calibration = copy.deepcopy(averaged_scaled_down_calBinned)

plt.figure()
plt.imshow(calibration, 'rainbow',origin='lower',aspect=10)
plt.title('Exp '+str(limits_angle)+'\n scaled_down_calBinned pre-smoothed')
plt.colorbar()
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)

calibration = calibration / spline(np.polyval(waveLcoefs[1],np.arange(calibration.shape[1])))


FWHM = 20	#this is about the width of a peak
for index,chord in enumerate(calibration):
	calibration[index] = np.convolve(chord, gaussian(100,FWHM/2.335)/np.sum(gaussian(100,FWHM/2.335)) , mode='same')	# I convolute it with a gaussian shape because peaks are about gaussian

# if np.sum(calibration<=0)>0:
# 	calibration = calibration + max(-np.min(calibration),0)+0.001	# 0.001 it's arbitrary just to avoid 1/0
calibration[calibration<=0] = np.min(calibration[calibration>0])



# calibration = calibration / spline(np.polyval(waveLcoefs[1],np.arange(calibration.shape[1])))


plt.figure()
plt.imshow(calibration, 'rainbow',origin='lower',aspect=10)
plt.title('Exp '+str(limits_angle)+'\n Actual calibration')
plt.colorbar()
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)

to_save = np.zeros((3,)+np.shape(calibration))

to_save[1] = calibration
#
#
# spline=interpolate.interp1d(df_sphere.waveL,df_sphere.Rad,kind='cubic') #Spline interp of sphere radiance
#
#
# to_save = np.zeros((3,)+np.shape(averaged_scaled_down_calBinned))
#
#
#
# to_save[1] = averaged_scaled_down_calBinned / spline(np.polyval(waveLcoefs[1],np.arange(averaged_scaled_down_calBinned.shape[1])))

np.save('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_5/Sensitivity_4',to_save)









# NOW I GET AGAIN THE SENSITIVITY FOR THE EXPERIMENTS IN WHICH I HAD THE anti overexposure FILTER

limits_angle = [40,52]
limits_tilt = [40,52]
limits_wave = [40,52]
figure_number = 0
where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_5/Sensitivity_2_plots_'

# geom_null = pd.DataFrame(columns=['angle', 'tilt', 'binInterv', 'bin00a', 'bin00b'])
# geom_null.loc[0] = [0, 0, 0, 0, 0]
#
# if np.sum(limits_angle) != 0:
# 	angle = []
# 	for merge in range(limits_angle[0], limits_angle[1] + 1):
# 		all_j = find_index_of_file(merge, df_settings, df_log)
# 		for j in all_j:
# 			dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
# 			(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
# 			type = '.tif'
# 			filenames = all_file_names(
# 				fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
# 			type = '.txt'
# 			filename_metadata = all_file_names(
# 				fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)[
# 				0]
# 			(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir, folder,
# 																										 sequence,
# 																										 untitled,
# 																										 filename_metadata)
#
# 			# plt.figure()
# 			data_all = []
# 			for index, filename in enumerate(filenames):
# 				fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(
# 					untitled) + '/Pos0/' + \
# 						filename
# 				im = Image.open(fname)
# 				data = np.array(im)
# 				data = (data - dataDark) * Gain[index]
# 				data = fix_minimum_signal2(data)
# 				data_all.append(data)
# 			data_all = np.array(data_all)
# 			data_mean = np.mean(data_all, axis=0)
# 			try:
# 				angle.append(get_angle_2(data_mean, nLines=2))
# 				print(angle)
# 			except:
# 				print('FAILED')
# 	angle = np.array(angle)
# 	print([angle])
# 	angle = np.nansum(angle[:, 0] / (angle[:, 1] ** 2)) / np.nansum(1 / angle[:, 1] ** 2)
#
# if np.sum(limits_tilt) != 0:
# 	tilt = []
# 	for merge in range(limits_tilt[0], limits_tilt[1] + 1):
# 		all_j = find_index_of_file(merge, df_settings, df_log)
# 		for j in all_j:
# 			dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
# 			(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
# 			type = '.tif'
# 			filenames = all_file_names(
# 				fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
# 			type = '.txt'
# 			filename_metadata = all_file_names(
# 				fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)[
# 				0]
# 			(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir, folder,
# 																										 sequence,
# 																										 untitled,
# 																										 filename_metadata)
#
# 			# plt.figure()
# 			data_all = []
# 			for index, filename in enumerate(filenames):
# 				fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(
# 					untitled) + '/Pos0/' + \
# 						filename
# 				im = Image.open(fname)
# 				data = np.array(im)
# 				data = (data - dataDark) * Gain[index]
# 				data = fix_minimum_signal2(data)
# 				data_all.append(data)
# 			data_all = np.array(data_all)
# 			data_mean = np.mean(data_all, axis=0)
# 			data_mean = rotate(data_mean, angle)
# 			try:
# 				tilt.append(get_tilt(data_mean, nLines=3))
# 				print(tilt)
# 			except:
# 				print('FAILED')
# 	tilt = np.array(tilt)
# 	print([tilt])
# 	tilt = np.nanmean(tilt, axis=0)
#
# geom = pd.DataFrame(columns=['angle', 'tilt', 'binInterv', 'bin00a', 'bin00b'])
# geom.loc[0] = [angle, tilt[1], tilt[0], tilt[2], tilt[2]]
# geom_store = copy.deepcopy(geom)
# print(geom)
#
# if np.sum(limits_wave) != 0:
# 	waveLcoefs = []
# 	for merge in range(limits_wave[0], limits_wave[1] + 1):
# 		try:
# 			waveLcoefs.append(do_waveL_Calib(merge, df_settings, df_log, geom, fdir=fdir))
# 		except:
# 			print('FAILED')
# 	waveLcoefs = np.array(waveLcoefs)
# 	print([waveLcoefs])
# 	waveLcoefs = np.nanmean(waveLcoefs, axis=0)


geom = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
geom.loc[0] = [ -0.114920732767, np.nan, np.nan, np.nan, np.nan]
waveLcoefs = np.ones((2, 3)) * np.nan
waveLcoefs[1] = [ -1.11822268e-06 ,  1.09054956e-01  , 3.48591929e+02]
geom_null = pd.DataFrame(columns=['angle', 'tilt', 'binInterv', 'bin00a', 'bin00b'])
geom_null.loc[0] = [0, 0, 0, 0, 0]

int_time_long = [0.1, 1, 10, 100]
calib_ID_long=[289,290,291,292]
calib_ID_long = np.array([calib_ID_long for _, calib_ID_long in sorted(zip(int_time_long, calib_ID_long))])
int_time_long = np.sort(int_time_long)
int_time_long_new = np.flip(int_time_long,axis=0)
calib_ID_long_new = np.flip(calib_ID_long,axis=0)

wavel_range = 100


# int_time = [0.01]
# data_all = []
# calib_ID=[288]
# for iFile, j in enumerate(calib_ID):
# 	(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
# 	dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
# 	type = '.tif'
# 	filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
# 	type = '.txt'
# 	temp = []
# 	for index, filename in enumerate(filenames):
# 		fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
# 		im = Image.open(fname)
# 		data = np.array(im)
# 		data = (data - dataDark)
# 		data = fix_minimum_signal2(data)
#
# 		temp.append(data)
# 	temp = np.mean(temp, axis=0)
# 	data = copy.deepcopy(temp)
# 	data = data/(df_log.loc[j, ['Calib_ID']][0]*1000000)
# 	data_all.append(data)
#
# data_all = np.array(data_all)
# data=np.mean(data_all,axis=(0))
# data = rotate(data, geom['angle'])
# data = do_tilt_no_lines(data)
# data=np.sum(data,axis=0)
# data=np.convolve(data, np.ones(200)/200 , mode='valid')
# wavel=data.argmax()+100



wavel = 800
wavelength_950=wavel
target_up = np.polyval(waveLcoefs[1], wavelength_950+wavel_range)
target_down = np.polyval(waveLcoefs[1], wavelength_950-wavel_range)


bin_interv_high_int_time = []
data_all = []
binned_all = []
peak_sum = []
longCal_all=[]
for iFile, j in enumerate(calib_ID_long):
	(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]

	dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)

	type = '.tif'
	filenames = all_file_names(
		fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
	type = '.txt'
	filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0',type)[0]
	(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir,folder,sequence,untitled,filename_metadata)

	data_sum = 0
	temp = []



	for index, filename in enumerate(filenames):
		fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(
			untitled) + '/Pos0/' + filename
		# print(fname)
		im = Image.open(fname)
		data = np.array(im)
		data = (data - dataDark)*Gain[index]
		data, fix_done_marker = fix_minimum_signal2(data, fix_done_marker=True)

		if fix_done_marker == True:
			continue

		data_sum += data
		temp.append(data)

	if np.max(data_sum) == 0:
		int_time_long_new = int_time_long_new[:-1]
		calib_ID_long_new = calib_ID_long_new[:-1]
		print('The integration time ' + str(int_time_long[iFile]) + 'ms will not be used because it requires minimum value corrrection to be used, and therefore it is deemed as unreliable')
		continue

	temp = np.mean(temp, axis=0)

	temp = rotate(temp, geom['angle'])
	temp = do_tilt_no_lines(temp)

	first_bin, bin_interv = get_bin_and_interv_no_lines(temp)
	print('bin interval of ' + str(bin_interv) + ' , first_bin of ' + str(first_bin))
	bin_interv_high_int_time.append(bin_interv)
	binned_data = binData(temp, first_bin, bin_interv)/(df_log.loc[j, ['Calib_ID']][0]*1000000)
	longCal_all.append( binData(temp, first_bin - bin_interv, bin_interv, nCh=42)/(df_log.loc[j, ['Calib_ID']][0]*1000000))


	data_all.append(temp)
	binned_all.append(binned_data)


	# peak_sum.append(np.sum(binned_data[:,wavel - wavel_range:wavel + wavel_range],axis=(-1,-2)))
	peak_sum.append(np.sum(binned_data[:, wavel - wavel_range:wavel + wavel_range], axis=(-1)))
peak_sum=np.array(peak_sum)

int_time_long = np.flip(int_time_long_new,axis=0)
calib_ID_long = np.flip(calib_ID_long_new,axis=0)

max_coordinate = 0
for index in range(len(data_all)):
	max_coordinate = max(max_coordinate,np.shape(data_all[index])[0])
for index in range(len(data_all)):
	while np.shape(data_all[index])[0]<max_coordinate:
		temp_1=data_all[index].tolist()
		temp_1.append(np.zeros(np.shape(temp_1[0])))
		data_all[index] = np.array(temp_1)
	plt.figure()
	plt.imshow(data_all[index], 'rainbow', origin='lower')
	plt.title('Exp ' + str(limits_angle) + '\n ' + str(int_time_long[index]) + 'int time sensitivity')
	plt.xlabel('wavelength axis [pixels]')
	plt.ylabel('LOS axis [pixels]')
	figure_number += 1
	plt.savefig(where_to_save_everything + str(figure_number) + '.eps', bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)
data_all_long_int_time=np.array(data_all)
binned_all=np.array(binned_all)
bin_interv_high_int_time = np.nanmean(bin_interv_high_int_time)

calBinned_peak_sum=[]
calBinned_all=[]
longCal_all=np.array(longCal_all)
for i in range(len(binned_all)):
	topDark = longCal_all[i][0]
	botDark = longCal_all[i][-1]
	wtDark = np.arange(1, 41) / 41
	calBinned = longCal_all[i][1:41] - (np.outer(wtDark, topDark) + np.outer(wtDark[::-1], botDark)) + (topDark + botDark) / 2  # MODIFIED TO AVOID NEGATIVE VALUES
	# calBinned_peak_sum.append(np.sum(calBinned[:,wavel - wavel_range:wavel + wavel_range],axis=(-1,-2)))
	calBinned_peak_sum.append(np.sum(calBinned[:, wavel - wavel_range:wavel + wavel_range], axis=(-1)))
	calBinned_all.append(calBinned)
calBinned_peak_sum = np.array(calBinned_peak_sum)
calBinned_all=np.array(calBinned_all)


int_time = [0.01]
data_all = []
calib_ID=[288]
for iFile, j in enumerate(calib_ID):
	(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]

	dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)

	type = '.tif'
	filenames = all_file_names(
		fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
	type = '.txt'
	filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0',type)[0]
	(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir,folder,sequence,untitled,filename_metadata)

	data_sum = 0
	temp = []
	for index, filename in enumerate(filenames):
		fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
		# print(fname)
		im = Image.open(fname)
		data = np.array(im)
		data = (data - dataDark)*Gain[index]

		# data = rotate(data, geom['angle'])
		# data = do_tilt(data, geom['tilt'])
		data = fix_minimum_signal2(data)

		data_sum += data
		temp.append(data)
	temp = np.mean(temp, axis=0)


	data = copy.deepcopy(temp)
	data = data/(df_log.loc[j, ['Calib_ID']][0]*1000000)
	data_all.append(data)

data_all = np.array(data_all)
data=np.mean(data_all,axis=(0))
data = rotate(data, geom['angle'])
data = do_tilt_no_lines(data)

plt.figure()
plt.imshow(data,'rainbow',origin='lower')
plt.title('Exp '+str(limits_angle)+'\n Low int time sensitivity')
plt.xlabel('wavelength axis [pixels]')
plt.ylabel('LOS axis [pixels]')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)

# data = data - min(np.min(data),0)

# max_coordinate = 0
# for index in range(len(data_all)):
# 	max_coordinate = max(max_coordinate,np.shape(data_all[index])[0])
# for index in range(len(data_all)):
# 	while np.shape(data_all[index])[0]<max_coordinate:
# 		temp_1=data_all[index].tolist()
# 		temp_1.append(np.zeros(np.shape(temp_1[0])))
# 		data_all[index] = np.array(temp_1)
# data_all=np.array(data_all)
# data=np.mean(data_all,axis=(0))

data_no_mask = copy.deepcopy(data)
# data_no_mask = data_no_mask - min(np.min(data_no_mask),0)
first_bin, bin_interv = get_bin_and_interv_no_lines(data)
print('bin interval of ' + str(bin_interv)+' , first_bin of '+str(first_bin))
bin_interv_no_mask = bin_interv
binned_data_no_mask = binData(data, first_bin, bin_interv)
# binned_data_no_mask = binned_data_no_mask - min(np.min(binned_data_no_mask),0)
longCal_no_mask = binData(data,first_bin-bin_interv,bin_interv,nCh = 42)
topDark_no_mask = longCal_no_mask[0]
botDark_no_mask = longCal_no_mask[-1]
wtDark_no_mask = np.arange(1, 41) / 41
calBinned_no_mask = longCal_no_mask[1:41] - (np.outer(wtDark_no_mask, topDark_no_mask) + np.outer(wtDark_no_mask[::-1], botDark_no_mask)) + (topDark_no_mask + botDark_no_mask) / 2  # MODIFIED TO AVOID NEGATIVE VALUES

# peak_sum_single = np.sum(binned_data_no_mask[:, wavel - wavel_range:wavel + wavel_range], axis=(-1, -2))  # type: ndarray
# calBinned_peak_sum_single = np.sum(calBinned_no_mask[:, wavel - wavel_range:wavel + wavel_range], axis=(-1, -2))  # type: ndarray
peak_sum_single = np.sum(binned_data_no_mask[:, wavel - wavel_range:wavel + wavel_range], axis=(-1))  # type: ndarray
calBinned_peak_sum_single = np.sum(calBinned_no_mask[:, wavel - wavel_range:wavel + wavel_range], axis=(-1))  # type: ndarray



max_coordinate=0
plt.figure()
scaled_down_binned_all=[]
for i in range(len(binned_all)):
	# scaled_down_binned_all.append(binned_all[i]*peak_sum_single/peak_sum[i])
	scaled_down_binned_all.append((binned_all[i].T *( peak_sum_single / peak_sum[i])).T)
	plt.plot(np.mean(scaled_down_binned_all[-1],axis=0),label=str(int_time_long[i]))
	max_coordinate = max(max_coordinate, np.max(np.mean(scaled_down_binned_all[-1],axis=0)))
scaled_down_binned_all=np.array(scaled_down_binned_all)

plt.plot(np.mean(binned_data_no_mask,axis=0),label='0.01')
max_coordinate = max(max_coordinate, np.max(np.mean(binned_data_no_mask,axis=0)))
plt.title('Exp '+str(limits_angle)+'\n Comparison of scaled_down_binned_all averaged over all bins and scaled around column '+str(wavel)+' for different integration times')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.plot([wavel,wavel],[0,max_coordinate],'--k')


averaged_scaled_down_binned_all=np.mean(scaled_down_binned_all,axis=0)
plt.plot(np.mean(averaged_scaled_down_binned_all,axis=0),label='Averaged')


plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)



# plt.figure()
# for i in range(len(binned_all)):
# 	plt.plot(scaled_down_binned_all[i,30],label=str(int_time_long[i]))
# plt.plot(binned_data_no_mask[30],label='0.01')
# plt.xlabel('wavelength axis [au]')
# plt.ylabel('sensitivity [au]')
# plt.title('Exp '+str(limits_angle)+'\n scaled_down_binned_all comparison at bin 30')
# plt.plot(averaged_scaled_down_binned_all[30],label='Averaged')
# plt.legend(loc='best')
# plt.pause(0.01)
#
#
# plt.figure()
# for i in range(len(binned_all)):
# 	plt.plot(scaled_down_binned_all[i,20],label=str(int_time_long[i]))
# plt.plot(binned_data_no_mask[20],label='0.01')
# plt.xlabel('wavelength axis [au]')
# plt.ylabel('sensitivity [au]')
# plt.title('Exp '+str(limits_angle)+'\n scaled_down_binned_all comparison at bin 20')
# plt.plot(averaged_scaled_down_binned_all[20],label='Averaged')
# plt.legend(loc='best')
# plt.pause(0.01)
#
#
# plt.figure()
# for i in range(len(binned_all)):
# 	plt.plot(np.sum(scaled_down_binned_all[i,:,1000-100:1000+100],axis=(-1)),label=str(int_time_long[i]))
# plt.plot(np.sum(binned_data_no_mask[:,1000-100:1000+100],axis=(-1)),label='0.01')
# plt.xlabel('wavelength axis [au]')
# plt.ylabel('sensitivity [au]')
# plt.title('Exp '+str(limits_angle)+'\n scaled_down_binned_all comparison at 1000 +/-100')
# plt.plot(np.sum(averaged_scaled_down_binned_all[:,1000-100:1000+100],axis=(-1)),label='Averaged')
# plt.legend(loc='best')
# plt.pause(0.01)
#
#
#
# plt.figure()
# plt.imshow(averaged_scaled_down_binned_all, 'rainbow',origin='lower',aspect=10)
# plt.title('Exp '+str(limits_angle)+'\n scaled_down_binned_all')
# plt.colorbar()
# plt.pause(0.01)



max_coordinate=0
plt.figure()
scaled_down_calBinned=[]
for i in range(len(binned_all)):
	# scaled_down_binned_all.append(binned_all[i]*peak_sum_single/peak_sum[i])
	scaled_down_calBinned.append((calBinned_all[i].T * (calBinned_peak_sum_single / calBinned_peak_sum[i])).T)
	plt.plot(np.mean(scaled_down_calBinned[-1],axis=0),label=str(int_time_long[i]))
	max_coordinate = max(max_coordinate, np.max(np.mean(scaled_down_calBinned[-1],axis=0)))
scaled_down_calBinned=np.array(scaled_down_calBinned)

plt.plot(np.mean(calBinned_no_mask,axis=0),label='0.01')
max_coordinate = max(max_coordinate, np.max(np.mean(calBinned_no_mask,axis=0)))
plt.title('Exp '+str(limits_angle)+'\n Comparison of scaled_down_calBinned averaged over all bins and scaled around column '+str(wavel)+' for different integration times')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.plot([wavel,wavel],[0,max_coordinate],'--k')


averaged_scaled_down_calBinned=np.mean(scaled_down_calBinned,axis=0)
averaged_scaled_down_calBinned[averaged_scaled_down_calBinned<=np.min(np.mean(averaged_scaled_down_calBinned,axis=0))]=np.min(np.mean(averaged_scaled_down_calBinned,axis=0))	#this is to erase negative or too low values at the borders
plt.plot(np.mean(averaged_scaled_down_calBinned,axis=0),label='Averaged')


plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)

# max_coordinate=0
# plt.figure()
# scaled_down_calBinned=[]
# for i in range(len(binned_all)):
# 	topDark = longCal_all[i][0]
# 	botDark = longCal_all[i][-1]
# 	wtDark = np.arange(1, 41) / 41
# 	calBinned = longCal_all[i][1:41] - (np.outer(wtDark, topDark) + np.outer(wtDark[::-1], botDark)) + (topDark + botDark) / 2  # MODIFIED TO AVOID NEGATIVE VALUES
# 	scaled_down_calBinned.append(calBinned*peak_sum_single/peak_sum[i])
# 	plt.plot(np.mean(calBinned*calBinned_peak_sum_single/calBinned_peak_sum[i],axis=0),label=str(int_time_long[i]))
# 	max_coordinate = max(max_coordinate, np.max(np.mean(calBinned*calBinned_peak_sum_single/calBinned_peak_sum[i],axis=0)))
# scaled_down_calBinned=np.array(scaled_down_calBinned)
#
#
# averaged_scaled_down_calBinned=np.mean(scaled_down_calBinned,axis=0)
# plt.plot(np.mean(calBinned_no_mask,axis=0),label='0.01')
# max_coordinate = max(max_coordinate, np.max(np.mean(calBinned_no_mask,axis=0)))
# plt.title('Exp '+str(limits_angle)+'\n Comparison of the sensitivity averaged over all bins and scaled around column '+str(wavel)+' for different integration times')
# plt.xlabel('wavelength axis [au]')
# plt.ylabel('sensitivity [au]')
# plt.plot([wavel,wavel],[0,max_coordinate],'--k')
#
#
# averaged_scaled_down_calBinned=np.mean(scaled_down_calBinned,axis=0)
# averaged_scaled_down_calBinned[averaged_scaled_down_calBinned<=np.min(np.mean(averaged_scaled_down_calBinned,axis=0))]=np.min(np.mean(averaged_scaled_down_calBinned,axis=0))	#this is to erase negative or too low values at the borders
# plt.plot(np.mean(averaged_scaled_down_calBinned,axis=0),label='Averaged')
#
#
# plt.legend(loc='best')
# plt.pause(0.01)


plt.figure()
for i in range(len(binned_all)):
	plt.plot(np.sum(scaled_down_calBinned[i,:,1000-100:1000+100],axis=(-1))/np.max(np.sum(scaled_down_calBinned[i,:,1000-100:1000+100],axis=(-1))),label=str(int_time_long[i]))
plt.plot(np.sum(calBinned_no_mask[:,1000-100:1000+100],axis=(-1))/np.max(np.sum(calBinned_no_mask[:,1000-100:1000+100],axis=(-1))),label='0.01')
plt.title('Exp '+str(limits_angle)+'\n scaled_down_calBinned comparison at 1000 +/-100')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.plot(np.sum(averaged_scaled_down_calBinned[:,1000-100:1000+100],axis=(-1))/np.max(np.sum(averaged_scaled_down_calBinned[:,1000-100:1000+100],axis=(-1))),label='Averaged')
plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)


plt.figure()
for i in range(len(binned_all)):
	plt.plot(scaled_down_calBinned[i,30],label=str(int_time_long[i]))
plt.plot(calBinned_no_mask[30],label='0.01')
plt.title('Exp '+str(limits_angle)+'\n scaled_down_calBinned comparison for bin 30')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.plot(averaged_scaled_down_calBinned[30],label='Averaged')
plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)


plt.figure()
for i in range(len(binned_all)):
	plt.plot(scaled_down_calBinned[i,20],label=str(int_time_long[i]))
plt.plot(calBinned_no_mask[20],label='0.01')
plt.title('Exp '+str(limits_angle)+'\n scaled_down_calBinned comparison for bin 20')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.plot(averaged_scaled_down_calBinned[20],label='Averaged')
plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)



plt.figure()
plt.imshow(averaged_scaled_down_calBinned, 'rainbow',origin='lower',aspect=10)
plt.title('Exp '+str(limits_angle)+'\n scaled_down_calBinned')
plt.colorbar()
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)

plt.figure()
for i in range(len(binned_all)):
	plt.plot(np.sum(calBinned_all[i,:,1400-100:1400+100],axis=(-1))/np.max(np.sum(calBinned_all[i,:,1400-100:1400+100],axis=(-1))),label=str(int_time_long[i]))
plt.plot(np.sum(calBinned_no_mask[:,1400-100:1400+100],axis=(-1))/np.max(np.sum(calBinned_no_mask[:,1400-100:1400+100],axis=(-1))),label='0.01')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.title('Exp '+str(limits_angle)+'\n calBinned_all comparison for at 1400 +/-100')
plt.plot(np.sum(averaged_scaled_down_calBinned[:,1400-100:1400+100],axis=(-1))/np.max(np.sum(averaged_scaled_down_calBinned[:,1400-100:1400+100],axis=(-1))),label='Averaged')
plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)

plt.figure()
for i in range(len(binned_all)):
	plt.plot(np.sum(calBinned_all[i,:,1000-100:1000+100],axis=(-1))/np.max(np.sum(calBinned_all[i,:,1000-100:1000+100],axis=(-1))),label=str(int_time_long[i]))
plt.plot(np.sum(calBinned_no_mask[:,1000-100:1000+100],axis=(-1))/np.max(np.sum(calBinned_no_mask[:,1000-100:1000+100],axis=(-1))),label='0.01')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.title('Exp '+str(limits_angle)+'\n calBinned_all comparison for at 1000 +/-100')
plt.plot(np.sum(averaged_scaled_down_calBinned[:,1000-100:1000+100],axis=(-1))/np.max(np.sum(averaged_scaled_down_calBinned[:,1000-100:1000+100],axis=(-1))),label='Averaged')
plt.legend(loc='best')

plt.figure()
test_wavel = 1000
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise']
for i in range(len(binned_all)):
	plt.plot(calBinned_all[i,:,test_wavel]/np.max(calBinned_all[i,:,test_wavel]),color=color[i],linestyle='-',label=str(int_time_long[i]))
	plt.plot(calBinned_all[i, :, test_wavel+1] / np.max(calBinned_all[i, :, test_wavel+1]), color=color[i],linestyle='--', label=str(int_time_long[i]))
	plt.plot(calBinned_all[i, :, test_wavel+2] / np.max(calBinned_all[i, :, test_wavel+2]), color=color[i],linestyle='-.', label=str(int_time_long[i]))
plt.plot(calBinned_no_mask[:,test_wavel]/np.max(calBinned_no_mask[:,test_wavel]),color=color[i+1],linestyle='-',label='0.01')
plt.plot(calBinned_no_mask[:,test_wavel+1]/np.max(calBinned_no_mask[:,test_wavel+1]),color=color[i+1],linestyle='--',label='0.01')
plt.plot(calBinned_no_mask[:,test_wavel+2]/np.max(calBinned_no_mask[:,test_wavel+2]),color=color[i+1],linestyle='-.',label='0.01')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.title('Exp '+str(limits_angle)+'\n calBinned_all comparison at '+str(test_wavel)+', '+str(test_wavel+1)+', '+str(test_wavel+2))
plt.plot(averaged_scaled_down_calBinned[:,test_wavel]/np.max(averaged_scaled_down_calBinned[:,test_wavel]),color=color[i+2],linestyle='-',label='Averaged')
plt.plot(averaged_scaled_down_calBinned[:,test_wavel+1]/np.max(averaged_scaled_down_calBinned[:,test_wavel+1]),color=color[i+2],linestyle='--',label='Averaged')
plt.plot(averaged_scaled_down_calBinned[:,test_wavel+2]/np.max(averaged_scaled_down_calBinned[:,test_wavel+2]),color=color[i+2],linestyle='-.',label='Averaged')
plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)


plt.figure()
test_wavel = 1400
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise']
for i in range(len(binned_all)):
	plt.plot(calBinned_all[i,:,test_wavel]/np.max(calBinned_all[i,:,test_wavel]),color=color[i],linestyle='-',label=str(int_time_long[i]))
	plt.plot(calBinned_all[i, :, test_wavel+1] / np.max(calBinned_all[i, :, test_wavel+1]), color=color[i],linestyle='--', label=str(int_time_long[i]))
	plt.plot(calBinned_all[i, :, test_wavel+2] / np.max(calBinned_all[i, :, test_wavel+2]), color=color[i],linestyle='-.', label=str(int_time_long[i]))
plt.plot(calBinned_no_mask[:,test_wavel]/np.max(calBinned_no_mask[:,test_wavel]),color=color[i+1],linestyle='-',label='0.01')
plt.plot(calBinned_no_mask[:,test_wavel+1]/np.max(calBinned_no_mask[:,test_wavel+1]),color=color[i+1],linestyle='--',label='0.01')
plt.plot(calBinned_no_mask[:,test_wavel+2]/np.max(calBinned_no_mask[:,test_wavel+2]),color=color[i+1],linestyle='-.',label='0.01')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.title('Exp '+str(limits_angle)+'\n calBinned_all comparison at '+str(test_wavel)+', '+str(test_wavel+1)+', '+str(test_wavel+2))
plt.plot(averaged_scaled_down_calBinned[:,test_wavel]/np.max(averaged_scaled_down_calBinned[:,test_wavel]),color=color[i+2],linestyle='-',label='Averaged')
plt.plot(averaged_scaled_down_calBinned[:,test_wavel+1]/np.max(averaged_scaled_down_calBinned[:,test_wavel+1]),color=color[i+2],linestyle='--',label='Averaged')
plt.plot(averaged_scaled_down_calBinned[:,test_wavel+2]/np.max(averaged_scaled_down_calBinned[:,test_wavel+2]),color=color[i+2],linestyle='-.',label='Averaged')
plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)

spline=interpolate.interp1d(df_sphere.waveL,df_sphere.Rad,kind='cubic') #Spline interp of sphere radiance

#	THIS IS NOT NECESSARY ANYMORE BECAUSE i AVOID NEGATIVE VALUES
# calBinned_min= np.max(calBinned)/1000
# for irow, row in enumerate(calBinned):
# 	for icol, value in enumerate(row):
# 		if (value < calBinned_min):
# 			calBinned[irow, icol] =  calBinned_min




# averaged_scaled_down_calBinned[averaged_scaled_down_calBinned==0]=np.min(averaged_scaled_down_calBinned[averaged_scaled_down_calBinned!=0])
# spline=interpolate.interp1d(df_sphere.waveL,df_sphere.Rad,kind='cubic') #Spline interp of sphere radiance
# calibration = calBinned_no_mask / spline(np.polyval(waveLcoefs[1],np.arange(calBinned_no_mask.shape[1])))
calibration = copy.deepcopy(averaged_scaled_down_calBinned)

plt.figure()
plt.imshow(calibration, 'rainbow',origin='lower',aspect=10)
plt.title('Exp '+str(limits_angle)+'\n scaled_down_calBinned pre-smoothed')
plt.colorbar()
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)

calibration = calibration / spline(np.polyval(waveLcoefs[1],np.arange(calibration.shape[1])))


FWHM = 20	#this is about the width of a peak
for index,chord in enumerate(calibration):
	calibration[index] = np.convolve(chord, gaussian(100,FWHM/2.335)/np.sum(gaussian(100,FWHM/2.335)) , mode='same')	# I convolute it with a gaussian shape because peaks are about gaussian

# if np.sum(calibration<=0)>0:
# 	calibration = calibration + max(-np.min(calibration),0)+0.001	# 0.001 it's arbitrary just to avoid 1/0
calibration[calibration<=0] = np.min(calibration[calibration>0])



# calibration = calibration / spline(np.polyval(waveLcoefs[1],np.arange(calibration.shape[1])))


plt.figure()
plt.imshow(calibration, 'rainbow',origin='lower',aspect=10)
plt.title('Exp '+str(limits_angle)+'\n Actual calibration')
plt.colorbar()
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)

to_save = np.zeros((3,)+np.shape(calibration))

to_save[1] = calibration
#
#
# spline=interpolate.interp1d(df_sphere.waveL,df_sphere.Rad,kind='cubic') #Spline interp of sphere radiance
#
#
# to_save = np.zeros((3,)+np.shape(averaged_scaled_down_calBinned))
#
#
#
# to_save[1] = averaged_scaled_down_calBinned / spline(np.polyval(waveLcoefs[1],np.arange(averaged_scaled_down_calBinned.shape[1])))

np.save('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_5/Sensitivity_2',to_save)









# NOW I GET AGAIN THE SENSITIVITY FOR THE FIRST ROUND OF EXPERIMENTS (merge<26)

limits_angle = [17,26]
limits_tilt = [17,26]
limits_wave = [17,26]
figure_number = 0
where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_5/Sensitivity_1_plots_'

# geom_null = pd.DataFrame(columns=['angle', 'tilt', 'binInterv', 'bin00a', 'bin00b'])
# geom_null.loc[0] = [0, 0, 0, 0, 0]
#
# if np.sum(limits_angle) != 0:
# 	angle = []
# 	for merge in range(limits_angle[0], limits_angle[1] + 1):
# 		all_j = find_index_of_file(merge, df_settings, df_log)
# 		for j in all_j:
# 			dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
# 			(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
# 			type = '.tif'
# 			filenames = all_file_names(
# 				fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
# 			type = '.txt'
# 			filename_metadata = all_file_names(
# 				fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)[
# 				0]
# 			(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir, folder,
# 																										 sequence,
# 																										 untitled,
# 																										 filename_metadata)
#
# 			# plt.figure()
# 			data_all = []
# 			for index, filename in enumerate(filenames):
# 				fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(
# 					untitled) + '/Pos0/' + \
# 						filename
# 				im = Image.open(fname)
# 				data = np.array(im)
# 				data = (data - dataDark) * Gain[index]
# 				data = fix_minimum_signal2(data)
# 				data_all.append(data)
# 			data_all = np.array(data_all)
# 			data_mean = np.mean(data_all, axis=0)
# 			try:
# 				angle.append(get_angle_2(data_mean, nLines=2))
# 				print(angle)
# 			except:
# 				print('FAILED')
# 	angle = np.array(angle)
# 	print([angle])
# 	angle = np.nansum(angle[:, 0] / (angle[:, 1] ** 2)) / np.nansum(1 / angle[:, 1] ** 2)
#
# if np.sum(limits_tilt) != 0:
# 	tilt = []
# 	for merge in range(limits_tilt[0], limits_tilt[1] + 1):
# 		all_j = find_index_of_file(merge, df_settings, df_log)
# 		for j in all_j:
# 			dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
# 			(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
# 			type = '.tif'
# 			filenames = all_file_names(
# 				fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
# 			type = '.txt'
# 			filename_metadata = all_file_names(
# 				fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)[
# 				0]
# 			(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir, folder,
# 																										 sequence,
# 																										 untitled,
# 																										 filename_metadata)
#
# 			# plt.figure()
# 			data_all = []
# 			for index, filename in enumerate(filenames):
# 				fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(
# 					untitled) + '/Pos0/' + \
# 						filename
# 				im = Image.open(fname)
# 				data = np.array(im)
# 				data = (data - dataDark) * Gain[index]
# 				data = fix_minimum_signal2(data)
# 				data_all.append(data)
# 			data_all = np.array(data_all)
# 			data_mean = np.mean(data_all, axis=0)
# 			data_mean = rotate(data_mean, angle)
# 			try:
# 				tilt.append(get_tilt(data_mean, nLines=3))
# 				print(tilt)
# 			except:
# 				print('FAILED')
# 	tilt = np.array(tilt)
# 	print([tilt])
# 	tilt = np.nanmean(tilt, axis=0)
#
# geom = pd.DataFrame(columns=['angle', 'tilt', 'binInterv', 'bin00a', 'bin00b'])
# geom.loc[0] = [angle, tilt[1], tilt[0], tilt[2], tilt[2]]
# geom_store = copy.deepcopy(geom)
# print(geom)
#
#
# if np.sum(limits_wave) != 0:
# 	waveLcoefs = []
# 	for merge in range(limits_wave[0], limits_wave[1] + 1):
# 		try:
# 			waveLcoefs.append(do_waveL_Calib(merge, df_settings, df_log, geom, fdir=fdir))
# 		except:
# 			print('FAILED')
# 	waveLcoefs = np.array(waveLcoefs)
# 	print([waveLcoefs])
# 	waveLcoefs = np.nanmean(waveLcoefs, axis=0)


geom = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
geom.loc[0] = [ 0.0592948797259, np.nan, np.nan, np.nan, np.nan]
waveLcoefs = np.ones((2, 3)) * np.nan
waveLcoefs[1] = [ 9.53889868e-07 ,  1.01412139e-01 ,  3.41962722e+02]
geom_null = pd.DataFrame(columns=['angle', 'tilt', 'binInterv', 'bin00a', 'bin00b'])
geom_null.loc[0] = [0, 0, 0, 0, 0]


# int_time_long = [0.1,0.1, 1,1, 10,10]
# calib_ID_long=[194,195,196,197,198,199]
int_time_long = [1,1, 10,10]		# I neglect 0.1ms to improve the regularity
calib_ID_long=[196,197,198,199]
calib_ID_long = np.array([calib_ID_long for _, calib_ID_long in sorted(zip(int_time_long, calib_ID_long))])
int_time_long = np.sort(int_time_long)
int_time_long_new = np.flip(int_time_long,axis=0)
calib_ID_long_new = np.flip(calib_ID_long,axis=0)

wavel_range = 100


int_time = [0.01,0.01]
data_all = []
calib_ID=[129,130]
for iFile, j in enumerate(calib_ID):
	(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
	dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
	type = '.tif'
	filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
	type = '.txt'
	temp = []
	for index, filename in enumerate(filenames):
		fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
		im = Image.open(fname)
		data = np.array(im)
		data = (data - dataDark)
		data = fix_minimum_signal2(data)

		temp.append(data)
	temp = np.mean(temp, axis=0)
	data = copy.deepcopy(temp)
	data = data/(df_log.loc[j, ['Calib_ID']][0]*1000000)
	data_all.append(data)

data_all = np.array(data_all)
data=np.mean(data_all,axis=(0))
data = rotate(data, geom['angle'])
data = do_tilt_no_lines(data)
data=np.sum(data,axis=0)
data=np.convolve(data, np.ones(200)/200 , mode='valid')
wavel=data.argmax()+100



# wavel = 1438
wavelength_950=wavel
target_up = np.polyval(waveLcoefs[1], wavelength_950+wavel_range)
target_down = np.polyval(waveLcoefs[1], wavelength_950-wavel_range)


bin_interv_high_int_time = []
data_all = []
binned_all = []
peak_sum = []
longCal_all=[]
for iFile, j in enumerate(calib_ID_long):
	(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]

	dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)

	type = '.tif'
	filenames = all_file_names(
		fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
	type = '.txt'
	filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0',type)[0]
	(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir,folder,sequence,untitled,filename_metadata)

	data_sum = 0
	temp = []



	for index, filename in enumerate(filenames):
		fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(
			untitled) + '/Pos0/' + filename
		# print(fname)
		im = Image.open(fname)
		data = np.array(im)
		data = (data - dataDark)*Gain[index]
		data, fix_done_marker = fix_minimum_signal2(data, fix_done_marker=True)

		# if fix_done_marker == True:	# check aborted because the data are too bad for that
		# 	continue

		data_sum += data
		temp.append(data)

	# if np.max(data_sum) == 0:		# check aborted because the data are too bad for that
	# 	int_time_long_new = int_time_long_new[:-1]
	# 	calib_ID_long_new = calib_ID_long_new[:-1]
	# 	print('The integration time ' + str(int_time_long[iFile]) + 'ms will not be used because it requires minimum value corrrection to be used, and therefore it is deemed as unreliable')
	# 	continue

	temp = np.mean(temp, axis=0)

	temp = rotate(temp, geom['angle'])
	temp = do_tilt_no_lines(temp)

	first_bin, bin_interv = get_bin_and_interv_no_lines(temp)
	print('bin interval of ' + str(bin_interv) + ' , first_bin of ' + str(first_bin))
	bin_interv_high_int_time.append(bin_interv)
	binned_data = binData(temp, first_bin, bin_interv)/(df_log.loc[j, ['Calib_ID']][0]*1000000)
	longCal_all.append( binData(temp, first_bin - bin_interv, bin_interv, nCh=42)/(df_log.loc[j, ['Calib_ID']][0]*1000000))


	data_all.append(temp)
	binned_all.append(binned_data)

	# peak_sum.append(np.sum(binned_data[:,wavel - wavel_range:wavel + wavel_range],axis=(-1,-2)))
	peak_sum.append(np.sum(binned_data[:, wavel - wavel_range:wavel + wavel_range], axis=(-1)))
peak_sum=np.array(peak_sum)

int_time_long = np.flip(int_time_long_new,axis=0)
calib_ID_long = np.flip(calib_ID_long_new,axis=0)

max_coordinate = 0
for index in range(len(data_all)):
	max_coordinate = max(max_coordinate,np.shape(data_all[index])[0])
for index in range(len(data_all)):
	while np.shape(data_all[index])[0]<max_coordinate:
		temp_1=data_all[index].tolist()
		temp_1.append(np.zeros(np.shape(temp_1[0])))
		data_all[index] = np.array(temp_1)
	plt.figure()
	plt.imshow(data_all[index], 'rainbow', origin='lower')
	plt.title('Exp ' + str(limits_angle) + '\n '+str(int_time_long[index])+'int time sensitivity')
	plt.xlabel('wavelength axis [pixels]')
	plt.ylabel('LOS axis [pixels]')
	figure_number += 1
	plt.savefig(where_to_save_everything + str(figure_number) + '.eps', bbox_inches='tight')
	plt.close()
	# plt.pause(0.01)
data_all_long_int_time=np.array(data_all)
binned_all=np.array(binned_all)
bin_interv_high_int_time = np.nanmean(bin_interv_high_int_time)

calBinned_peak_sum=[]
calBinned_all=[]
longCal_all=np.array(longCal_all)
for i in range(len(binned_all)):
	topDark = longCal_all[i][0]
	botDark = longCal_all[i][-1]
	wtDark = np.arange(1, 41) / 41
	calBinned = longCal_all[i][1:41] - (np.outer(wtDark, topDark) + np.outer(wtDark[::-1], botDark)) + (topDark + botDark) / 2  # MODIFIED TO AVOID NEGATIVE VALUES
	# calBinned_peak_sum.append(np.sum(calBinned[:,wavel - wavel_range:wavel + wavel_range],axis=(-1,-2)))
	calBinned_peak_sum.append(np.sum(calBinned[:, wavel - wavel_range:wavel + wavel_range], axis=(-1)))
	calBinned_all.append(calBinned)
calBinned_peak_sum = np.array(calBinned_peak_sum)
calBinned_all=np.array(calBinned_all)




int_time = [0.01,0.01]
data_all = []
calib_ID=[129,130]
for iFile, j in enumerate(calib_ID):
	(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]

	dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)

	type = '.tif'
	filenames = all_file_names(
		fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
	type = '.txt'
	filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0',type)[0]
	(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir,folder,sequence,untitled,filename_metadata)

	data_sum = 0
	temp = []
	for index, filename in enumerate(filenames):
		fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
		# print(fname)
		im = Image.open(fname)
		data = np.array(im)
		data = (data - dataDark)*Gain[index]

		# data = rotate(data, geom['angle'])
		# data = do_tilt(data, geom['tilt'])
		data = fix_minimum_signal2(data)

		data_sum += data
		temp.append(data)
	temp = np.mean(temp, axis=0)


	data = copy.deepcopy(temp)
	data = data/(df_log.loc[j, ['Calib_ID']][0]*1000000)
	data_all.append(data)

data_all = np.array(data_all)
data=np.mean(data_all,axis=(0))
data = rotate(data, geom['angle'])
data = do_tilt_no_lines(data)

plt.figure()
plt.imshow(data,'rainbow',origin='lower')
plt.title('Exp '+str(limits_angle)+'\n Low int time sensitivity')
plt.xlabel('wavelength axis [pixels]')
plt.ylabel('LOS axis [pixels]')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)

# data = data - min(np.min(data),0)

# max_coordinate = 0
# for index in range(len(data_all)):
# 	max_coordinate = max(max_coordinate,np.shape(data_all[index])[0])
# for index in range(len(data_all)):
# 	while np.shape(data_all[index])[0]<max_coordinate:
# 		temp_1=data_all[index].tolist()
# 		temp_1.append(np.zeros(np.shape(temp_1[0])))
# 		data_all[index] = np.array(temp_1)
# data_all=np.array(data_all)
# data=np.mean(data_all,axis=(0))

data_no_mask = copy.deepcopy(data)
# data_no_mask = data_no_mask - min(np.min(data_no_mask),0)
first_bin, bin_interv = get_bin_and_interv_no_lines(data)
print('bin interval of ' + str(bin_interv)+' , first_bin of '+str(first_bin))
bin_interv_no_mask = bin_interv
binned_data_no_mask = binData(data, first_bin, bin_interv)
# binned_data_no_mask = binned_data_no_mask - min(np.min(binned_data_no_mask),0)
longCal_no_mask = binData(data,first_bin-bin_interv,bin_interv,nCh = 42)
topDark_no_mask = longCal_no_mask[0]
botDark_no_mask = longCal_no_mask[-1]
wtDark_no_mask = np.arange(1, 41) / 41
calBinned_no_mask = longCal_no_mask[1:41] - (np.outer(wtDark_no_mask, topDark_no_mask) + np.outer(wtDark_no_mask[::-1], botDark_no_mask)) + (topDark_no_mask + botDark_no_mask) / 2  # MODIFIED TO AVOID NEGATIVE VALUES

# peak_sum_single = np.sum(binned_data_no_mask[:, wavel - wavel_range:wavel + wavel_range], axis=(-1, -2))  # type: ndarray
# calBinned_peak_sum_single = np.sum(calBinned_no_mask[:, wavel - wavel_range:wavel + wavel_range], axis=(-1, -2))  # type: ndarray
peak_sum_single = np.sum(binned_data_no_mask[:, wavel - wavel_range:wavel + wavel_range], axis=(-1))  # type: ndarray
calBinned_peak_sum_single = np.sum(calBinned_no_mask[:, wavel - wavel_range:wavel + wavel_range], axis=(-1))  # type: ndarray




max_coordinate=0
plt.figure()
scaled_down_binned_all=[]
for i in range(len(binned_all)):
	# scaled_down_binned_all.append(binned_all[i]*peak_sum_single/peak_sum[i])
	scaled_down_binned_all.append((binned_all[i].T *( peak_sum_single / peak_sum[i])).T)
	plt.plot(np.mean(scaled_down_binned_all[-1],axis=0),label=str(int_time_long[i]))
	max_coordinate = max(max_coordinate, np.max(np.mean(scaled_down_binned_all[-1],axis=0)))
scaled_down_binned_all=np.array(scaled_down_binned_all)

plt.plot(np.mean(binned_data_no_mask,axis=0),label='0.01')
max_coordinate = max(max_coordinate, np.max(np.mean(binned_data_no_mask,axis=0)))
plt.title('Exp '+str(limits_angle)+'\n Comparison of scaled_down_binned_all averaged over all bins and scaled around column '+str(wavel)+' for different integration times')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.plot([wavel,wavel],[0,max_coordinate],'--k')


averaged_scaled_down_binned_all=np.mean(scaled_down_binned_all,axis=0)
plt.plot(np.mean(averaged_scaled_down_binned_all,axis=0),label='Averaged')


plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)



# plt.figure()
# for i in range(len(binned_all)):
# 	plt.plot(scaled_down_binned_all[i,30],label=str(int_time_long[i]))
# plt.plot(binned_data_no_mask[30],label='0.01')
# plt.xlabel('wavelength axis [au]')
# plt.ylabel('sensitivity [au]')
# plt.title('scaled_down_binned_all comparison at bin 30')
# plt.plot(averaged_scaled_down_binned_all[30],label='Averaged')
# plt.legend(loc='best')
# plt.pause(0.01)
#
#
# plt.figure()
# for i in range(len(binned_all)):
# 	plt.plot(scaled_down_binned_all[i,20],label=str(int_time_long[i]))
# plt.plot(binned_data_no_mask[20],label='0.01')
# plt.xlabel('wavelength axis [au]')
# plt.ylabel('sensitivity [au]')
# plt.title('scaled_down_binned_all comparison at bin 20')
# plt.plot(averaged_scaled_down_binned_all[20],label='Averaged')
# plt.legend(loc='best')
# plt.pause(0.01)
#
#
# plt.figure()
# for i in range(len(binned_all)):
# 	plt.plot(np.sum(scaled_down_binned_all[i,:,1000-100:1000+100],axis=(-1)),label=str(int_time_long[i]))
# plt.plot(np.sum(binned_data_no_mask[:,1000-100:1000+100],axis=(-1)),label='0.01')
# plt.xlabel('wavelength axis [au]')
# plt.ylabel('sensitivity [au]')
# plt.title('scaled_down_binned_all comparison at 1000 +/-100')
# plt.plot(np.sum(averaged_scaled_down_binned_all[:,1000-100:1000+100],axis=(-1)),label='Averaged')
# plt.legend(loc='best')
# plt.pause(0.01)
#
#
#
# plt.figure()
# plt.imshow(averaged_scaled_down_binned_all, 'rainbow',origin='lower',aspect=10)
# plt.title('scaled_down_binned_all')
# plt.colorbar()
# plt.pause(0.01)



max_coordinate=0
plt.figure()
scaled_down_calBinned=[]
for i in range(len(binned_all)):
	# scaled_down_binned_all.append(binned_all[i]*peak_sum_single/peak_sum[i])
	scaled_down_calBinned.append((calBinned_all[i].T * (calBinned_peak_sum_single / calBinned_peak_sum[i])).T)
	plt.plot(np.mean(scaled_down_calBinned[-1],axis=0),label=str(int_time_long[i]))
	max_coordinate = max(max_coordinate, np.max(np.mean(scaled_down_calBinned[-1],axis=0)))
scaled_down_calBinned=np.array(scaled_down_calBinned)

plt.plot(np.mean(calBinned_no_mask,axis=0),label='0.01')
max_coordinate = max(max_coordinate, np.max(np.mean(calBinned_no_mask,axis=0)))
plt.title('Exp '+str(limits_angle)+'\n Comparison of scaled_down_calBinned averaged over all bins and scaled around column '+str(wavel)+' for different integration times')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.plot([wavel,wavel],[0,max_coordinate],'--k')


averaged_scaled_down_calBinned=np.mean(scaled_down_calBinned,axis=0)
averaged_scaled_down_calBinned[averaged_scaled_down_calBinned<=np.min(np.mean(averaged_scaled_down_calBinned,axis=0))]=np.min(np.mean(averaged_scaled_down_calBinned,axis=0))	#this is to erase negative or too low values at the borders
plt.plot(np.mean(averaged_scaled_down_calBinned,axis=0),label='Averaged')


plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)

# max_coordinate=0
# plt.figure()
# scaled_down_calBinned=[]
# for i in range(len(binned_all)):
# 	topDark = longCal_all[i][0]
# 	botDark = longCal_all[i][-1]
# 	wtDark = np.arange(1, 41) / 41
# 	calBinned = longCal_all[i][1:41] - (np.outer(wtDark, topDark) + np.outer(wtDark[::-1], botDark)) + (topDark + botDark) / 2  # MODIFIED TO AVOID NEGATIVE VALUES
# 	scaled_down_calBinned.append(calBinned*peak_sum_single/peak_sum[i])
# 	plt.plot(np.mean(calBinned*calBinned_peak_sum_single/calBinned_peak_sum[i],axis=0),label=str(int_time_long[i]))
# 	max_coordinate = max(max_coordinate, np.max(np.mean(calBinned*calBinned_peak_sum_single/calBinned_peak_sum[i],axis=0)))
# scaled_down_calBinned=np.array(scaled_down_calBinned)
#
#
# averaged_scaled_down_calBinned=np.mean(scaled_down_calBinned,axis=0)
# plt.plot(np.mean(calBinned_no_mask,axis=0),label='0.01')
# max_coordinate = max(max_coordinate, np.max(np.mean(calBinned_no_mask,axis=0)))
# plt.title('Comparison of the sensitivity averaged over all bins and scaled around column '+str(wavel)+' for different integration times')
# plt.xlabel('wavelength axis [au]')
# plt.ylabel('sensitivity [au]')
# plt.plot([wavel,wavel],[0,max_coordinate],'--k')
#
#
# averaged_scaled_down_calBinned=np.mean(scaled_down_calBinned,axis=0)
# averaged_scaled_down_calBinned[averaged_scaled_down_calBinned<=np.min(np.mean(averaged_scaled_down_calBinned,axis=0))]=np.min(np.mean(averaged_scaled_down_calBinned,axis=0))	#this is to erase negative or too low values at the borders
# plt.plot(np.mean(averaged_scaled_down_calBinned,axis=0),label='Averaged')
#
#
# plt.legend(loc='best')
# plt.pause(0.01)


plt.figure()
for i in range(len(binned_all)):
	plt.plot(np.sum(scaled_down_calBinned[i,:,1000-100:1000+100],axis=(-1))/np.max(np.sum(scaled_down_calBinned[i,:,1000-100:1000+100],axis=(-1))),label=str(int_time_long[i]))
plt.plot(np.sum(calBinned_no_mask[:,1000-100:1000+100],axis=(-1))/np.max(np.sum(calBinned_no_mask[:,1000-100:1000+100],axis=(-1))),label='0.01')
plt.title('Exp '+str(limits_angle)+'\n scaled_down_calBinned comparison at 1000 +/-100')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.plot(np.sum(averaged_scaled_down_calBinned[:,1000-100:1000+100],axis=(-1))/np.max(np.sum(averaged_scaled_down_calBinned[:,1000-100:1000+100],axis=(-1))),label='Averaged')
plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)


plt.figure()
for i in range(len(binned_all)):
	plt.plot(scaled_down_calBinned[i,30],label=str(int_time_long[i]))
plt.plot(calBinned_no_mask[30],label='0.01')
plt.title('Exp '+str(limits_angle)+'\n scaled_down_calBinned comparison for bin 30')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.plot(averaged_scaled_down_calBinned[30],label='Averaged')
plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)


plt.figure()
for i in range(len(binned_all)):
	plt.plot(scaled_down_calBinned[i,20],label=str(int_time_long[i]))
plt.plot(calBinned_no_mask[20],label='0.01')
plt.title('Exp '+str(limits_angle)+'\n scaled_down_calBinned comparison for bin 20')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.plot(averaged_scaled_down_calBinned[20],label='Averaged')
plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)



plt.figure()
plt.imshow(averaged_scaled_down_calBinned, 'rainbow',origin='lower',aspect=10)
plt.title('Exp '+str(limits_angle)+'\n scaled_down_calBinned')
plt.colorbar()
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)

plt.figure()
for i in range(len(binned_all)):
	plt.plot(np.sum(calBinned_all[i,:,1400-100:1400+100],axis=(-1))/np.max(np.sum(calBinned_all[i,:,1400-100:1400+100],axis=(-1))),label=str(int_time_long[i]))
plt.plot(np.sum(calBinned_no_mask[:,1400-100:1400+100],axis=(-1))/np.max(np.sum(calBinned_no_mask[:,1400-100:1400+100],axis=(-1))),label='0.01')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.title('Exp '+str(limits_angle)+'\n calBinned_all comparison for at 1400 +/-100')
plt.plot(np.sum(averaged_scaled_down_calBinned[:,1400-100:1400+100],axis=(-1))/np.max(np.sum(averaged_scaled_down_calBinned[:,1400-100:1400+100],axis=(-1))),label='Averaged')
plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)

plt.figure()
for i in range(len(binned_all)):
	plt.plot(np.sum(calBinned_all[i,:,1000-100:1000+100],axis=(-1))/np.max(np.sum(calBinned_all[i,:,1000-100:1000+100],axis=(-1))),label=str(int_time_long[i]))
plt.plot(np.sum(calBinned_no_mask[:,1000-100:1000+100],axis=(-1))/np.max(np.sum(calBinned_no_mask[:,1000-100:1000+100],axis=(-1))),label='0.01')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.title('Exp '+str(limits_angle)+'\n calBinned_all comparison for at 1000 +/-100')
plt.plot(np.sum(averaged_scaled_down_calBinned[:,1000-100:1000+100],axis=(-1))/np.max(np.sum(averaged_scaled_down_calBinned[:,1000-100:1000+100],axis=(-1))),label='Averaged')
plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)


plt.figure()
test_wavel = 1000
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise']
for i in range(len(binned_all)):
	plt.plot(calBinned_all[i,:,test_wavel]/np.max(calBinned_all[i,:,test_wavel]),color=color[i],linestyle='-',label=str(int_time_long[i]))
	plt.plot(calBinned_all[i, :, test_wavel+1] / np.max(calBinned_all[i, :, test_wavel+1]), color=color[i],linestyle='--', label=str(int_time_long[i]))
	plt.plot(calBinned_all[i, :, test_wavel+2] / np.max(calBinned_all[i, :, test_wavel+2]), color=color[i],linestyle='-.', label=str(int_time_long[i]))
plt.plot(calBinned_no_mask[:,test_wavel]/np.max(calBinned_no_mask[:,test_wavel]),color=color[i+1],linestyle='-',label='0.01')
plt.plot(calBinned_no_mask[:,test_wavel+1]/np.max(calBinned_no_mask[:,test_wavel+1]),color=color[i+1],linestyle='--',label='0.01')
plt.plot(calBinned_no_mask[:,test_wavel+2]/np.max(calBinned_no_mask[:,test_wavel+2]),color=color[i+1],linestyle='-.',label='0.01')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.title('Exp '+str(limits_angle)+'\n calBinned_all comparison at '+str(test_wavel)+', '+str(test_wavel+1)+', '+str(test_wavel+2))
plt.plot(averaged_scaled_down_calBinned[:,test_wavel]/np.max(averaged_scaled_down_calBinned[:,test_wavel]),color=color[i+2],linestyle='-',label='Averaged')
plt.plot(averaged_scaled_down_calBinned[:,test_wavel+1]/np.max(averaged_scaled_down_calBinned[:,test_wavel+1]),color=color[i+2],linestyle='--',label='Averaged')
plt.plot(averaged_scaled_down_calBinned[:,test_wavel+2]/np.max(averaged_scaled_down_calBinned[:,test_wavel+2]),color=color[i+2],linestyle='-.',label='Averaged')
plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)


plt.figure()
test_wavel = 1400
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise']
for i in range(len(binned_all)):
	plt.plot(calBinned_all[i,:,test_wavel]/np.max(calBinned_all[i,:,test_wavel]),color=color[i],linestyle='-',label=str(int_time_long[i]))
	plt.plot(calBinned_all[i, :, test_wavel+1] / np.max(calBinned_all[i, :, test_wavel+1]), color=color[i],linestyle='--', label=str(int_time_long[i]))
	plt.plot(calBinned_all[i, :, test_wavel+2] / np.max(calBinned_all[i, :, test_wavel+2]), color=color[i],linestyle='-.', label=str(int_time_long[i]))
plt.plot(calBinned_no_mask[:,test_wavel]/np.max(calBinned_no_mask[:,test_wavel]),color=color[i+1],linestyle='-',label='0.01')
plt.plot(calBinned_no_mask[:,test_wavel+1]/np.max(calBinned_no_mask[:,test_wavel+1]),color=color[i+1],linestyle='--',label='0.01')
plt.plot(calBinned_no_mask[:,test_wavel+2]/np.max(calBinned_no_mask[:,test_wavel+2]),color=color[i+1],linestyle='-.',label='0.01')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.title('Exp '+str(limits_angle)+'\n calBinned_all comparison at '+str(test_wavel)+', '+str(test_wavel+1)+', '+str(test_wavel+2))
plt.plot(averaged_scaled_down_calBinned[:,test_wavel]/np.max(averaged_scaled_down_calBinned[:,test_wavel]),color=color[i+2],linestyle='-',label='Averaged')
plt.plot(averaged_scaled_down_calBinned[:,test_wavel+1]/np.max(averaged_scaled_down_calBinned[:,test_wavel+1]),color=color[i+2],linestyle='--',label='Averaged')
plt.plot(averaged_scaled_down_calBinned[:,test_wavel+2]/np.max(averaged_scaled_down_calBinned[:,test_wavel+2]),color=color[i+2],linestyle='-.',label='Averaged')
plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)


spline=interpolate.interp1d(df_sphere.waveL,df_sphere.Rad,kind='cubic') #Spline interp of sphere radiance

#	THIS IS NOT NECESSARY ANYMORE BECAUSE i AVOID NEGATIVE VALUES
# calBinned_min= np.max(calBinned)/1000
# for irow, row in enumerate(calBinned):
# 	for icol, value in enumerate(row):
# 		if (value < calBinned_min):
# 			calBinned[irow, icol] =  calBinned_min




# averaged_scaled_down_calBinned[averaged_scaled_down_calBinned==0]=np.min(averaged_scaled_down_calBinned[averaged_scaled_down_calBinned!=0])
# spline=interpolate.interp1d(df_sphere.waveL,df_sphere.Rad,kind='cubic') #Spline interp of sphere radiance
# calibration = calBinned_no_mask / spline(np.polyval(waveLcoefs[1],np.arange(calBinned_no_mask.shape[1])))
calibration = copy.deepcopy(averaged_scaled_down_calBinned)

plt.figure()
plt.imshow(calibration, 'rainbow',origin='lower',aspect=10)
plt.title('Exp '+str(limits_angle)+'\n scaled_down_calBinned pre-smoothed')
plt.colorbar()
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)

calibration = calibration / spline(np.polyval(waveLcoefs[1],np.arange(calibration.shape[1])))


FWHM = 20	#this is about the width of a peak
for index,chord in enumerate(calibration):
	calibration[index] = np.convolve(chord, gaussian(100,FWHM/2.335)/np.sum(gaussian(100,FWHM/2.335)) , mode='same')	# I convolute it with a gaussian shape because peaks are about gaussian

# if np.sum(calibration<=0)>0:
# 	calibration = calibration + max(-np.min(calibration),0)+0.001	# 0.001 it's arbitrary just to avoid 1/0
calibration[calibration<=0] = np.min(calibration[calibration>0])



# calibration = calibration / spline(np.polyval(waveLcoefs[1],np.arange(calibration.shape[1])))


plt.figure()
plt.imshow(calibration, 'rainbow',origin='lower',aspect=10)
plt.title('Exp '+str(limits_angle)+'\n Actual calibration')
plt.colorbar()
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)

to_save = np.zeros((3,)+np.shape(calibration))

to_save[1] = calibration
#
#
# spline=interpolate.interp1d(df_sphere.waveL,df_sphere.Rad,kind='cubic') #Spline interp of sphere radiance
#
#
# to_save = np.zeros((3,)+np.shape(averaged_scaled_down_calBinned))
#
#
#
# to_save[1] = averaged_scaled_down_calBinned / spline(np.polyval(waveLcoefs[1],np.arange(averaged_scaled_down_calBinned.shape[1])))

np.save('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_5/Sensitivity_1',to_save)

