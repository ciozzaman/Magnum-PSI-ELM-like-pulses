import os,sys
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import mkl
mkl.set_num_threads(1)
number_cpu_available = 8	#cpu_count()
import numpy as np

# overwrite_everything = [False, False, False, False, False]
overwrite_everything = [False, True, True, True, True]
# this is for, in order:
# ... merge_tot.npz'
# ... composed_array.npy' and ...new_timesteps
# ... binned_data.npy'
# ... all_fits.npy'
# ... gain_scaled_composed_array.npy'
# if nothing is specified [False, False, False, False, False] is used


type_of_sensitivity = 12
# type of imputs are: 4, 5, 7, 8, 9:
# 4 = high int time scaling down with no LOS discretization
# 5 = high int time scaling down with LOS discretization
# 7 = low int time scaling up with no LOS discretization and profile smoothing
# 8 = low int time profile smoothing
# 9 = like 8 but with a new function to correct for the minimum signal.
# 11 = like 9 but with the new functions to correct minimum signal.
# if nothing is specified 5 is used

perform_convolution = True
# This is to select if you want to do a convolution of the sensitivity with a constant (20 pixels) gaussian width, or a width based on the line shape fitted

# merge_ID_target_all = np.flip([86,87,88,90,91,92,93,94,95,96,97,98,99,89,85],axis=0)
# merge_ID_target_all = [76,75,77,78,79]
# merge_ID_target_all = [80,81,83,84,101,103]
# for merge_ID_target in np.flip([17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,41],axis=0):#, 40,42,43,44,45,46,47,48,49,50,51,52,54]:#,84]:
# for merge_ID_target in np.flip([66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84],axis=0):
# for merge_ID_target in np.flip([36,37,38,39,41,42,43,44,45,46,47,48,49,50,51,52],axis=0):
# for merge_ID_target in np.flip([40,41,42,43,44,45,46,47,48,49,50,51,52,54],axis=0):
# for merge_ID_target in [86,87,88,89,90,91,92, 93, 94, 95, 96, 97, 98]:
# for merge_ID_target in np.flip([851,86,87,88,89,90,91,92,93,94,95],axis=0):
# for merge_ID_target in np.flip([99,85,97,89,92, 93,95,94],axis=0):
# for merge_ID_target in np.flip([90,91,89,99,85,97],axis=0):
merge_ID_target_all = [188]

merge_ID_target = merge_ID_target_all[0]

print('All that will be done: ' + str(merge_ID_target_all))
merge_time_window=[-1,2]



import numpy as np
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())
# import matplotlib.pyplot as plt
#import .functions
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
from functions.spectools import rotate,do_tilt, binData, get_angle, get_tilt,get_angle_2,get_4_points,get_line_position,binData_with_sigma
from functions.Calibrate import do_waveL_Calib, do_Intensity_Calib, do_waveL_Calib_simplified
from functions.fabio_add import find_nearest_index,multi_gaussian,all_file_names,load_dark,find_index_of_file,get_metadata,movie_from_data,get_angle_no_lines,do_tilt_no_lines,four_point_transform,fix_minimum_signal,fix_minimum_signal2,fix_minimum_signal3,get_bin_and_interv_no_lines,examine_current_trace,apply_proportionality_calibration,fix_minimum_signal_experiment,get_bin_and_interv_specific_wavelength
from functions.GetSpectrumGeometry import getGeom
from functions.SpectralFit import doSpecFit_single_frame,doSpecFit_single_frame_with_sigma
from functions.GaussFitData import doLateralfit_time_tependent,doLateralfit_single,find_plasma_axis,doLateralfit_time_tependent_with_sigma,doLateralfit_single_with_sigma,doLateralfit_single_with_sigma_pure_Abel
import collections

# to show the line where it fails
import sys, traceback, logging
logging.basicConfig(level=logging.ERROR)


import os,sys
from PIL import Image
import xarray as xr
import pandas as pd
import copy
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.signal import find_peaks
from multiprocessing import Pool,cpu_count
# number_cpu_available = 14	#cpu_count()
number_cpu_available = 8
print('Number of cores available: '+str(number_cpu_available))
# print("Number of cores available doesn't work, so I take 10: "+str(number_cpu_available))
import shutil
mkl.set_num_threads(number_cpu_available)


# n = np.arange(10, 20)
# waveLengths = [656.45377,486.13615,434.0462,410.174,397.0072,388.9049,383.5384] + list((364.6*(n*n)/(n*n-4)))
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

fdir = '/home/ffederic/work/Collaboratory/test/experimental_data'
df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
df_settings = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_3.csv',index_col=0)

# type_of_sensitivity = 11;perform_convolution = True;merge_ID_target=94;merge_time_window=[-1,2]

# calculate_geometry = False
# merge_ID_target = 17	#	THIS IS GIVEN BY THE LAUNCHER
for i in range(10):
	print('.')
print('Starting to work on merge number '+str(merge_ID_target))
for i in range(10):
	print('.')


try:
	print('last_merge_done '+str(last_merge_done))
except:
	last_merge_done = 100000000000000


try:
	# print('Order to over write all data is '+str(overwrite_everything))
	print('Over write ' + 'merge' + str(merge_ID_target) + '_merge_tot.npz' + ' is ' + str(overwrite_everything[0]))
	print('Over write ' + 'merge'+str(merge_ID_target)+'_composed_array.npy' + ' is ' + str(overwrite_everything[1]))
	print('Over write ' + 'merge'+str(merge_ID_target)+'_binned_data.npy' + ' is ' + str(overwrite_everything[2]))
	print('Over write ' + 'merge'+str(merge_ID_target)+'_all_fits.npy' + ' is ' + str(overwrite_everything[3]))
	print('Over write ' + 'merge'+str(merge_ID_target)+'_gain_scaled_composed_array.npy' + ' is ' + str(overwrite_everything[4]))
except:
	overwrite_everything = [False, False, False, False, False]
	# overwrite_everything = False
	# print('Order to over write all data is '+str(overwrite_everything))
	print('Over write ' + 'merge' + str(merge_ID_target) + '_merge_tot.npz' + ' is ' + str(overwrite_everything[0]))
	print('Over write ' + 'merge'+str(merge_ID_target)+'_composed_array.npy' + ' is ' + str(overwrite_everything[1]))
	print('Over write ' + 'merge'+str(merge_ID_target)+'_binned_data.npy' + ' is ' + str(overwrite_everything[2]))
	print('Over write ' + 'merge'+str(merge_ID_target)+'_gain_scaled_composed_array.npy' + ' is ' + str(overwrite_everything[4]))

try:
	if type_of_sensitivity==4:
		print('The type of sensitivity used is obtained from scaling down the signal at high integration time. \nThe smoothed peak of the signal at high int time is made to match the level of the low int time, at the same wavelength location. \nThe peak signal is averaged across LOS, non discretized.')
	elif type_of_sensitivity == 5:
		print('The type of sensitivity used is obtained from scaling down the signal at high integration time. \nThe smoothed peak of the signal at high int time is made to match the level of the low int time, at the same wavelength location. \nThe peak signal is not averaged across LOS, every LOS is scaled indipendently, though at the same wavelength.')
	elif type_of_sensitivity == 6:
		print('Not used sensitivity setting, error.')
		exit()
	elif type_of_sensitivity == 7:
		print('The type of sensitivity used is obtained from scaling up the signal at low integration time. \nThe smoothed peak of the signal at high int time is made to match the level of the low int time, at the same wavelength location. \nThen the smoothed lowest signal at high int time is calculated and the signal at low int time is made to match that. \nThe peak/bottom signal is not averaged across LOS, every LOS is scaled indipendently, though at the same wavelength. \nA smoothing function based on Savitzky-Golay filter is applied.')
	elif type_of_sensitivity == 8:
		print('The type of sensitivity used is obtained from smoothing the low integration time. The smoothing function is based on Savitzky-Golay filter.')
	elif type_of_sensitivity == 9:
		print('The type of sensitivity used is obtained from smoothing the low integration time. The smoothing function is based on Savitzky-Golay filter.\nUsed new minimum signal correction function fix_minimum_signal3.')
	elif type_of_sensitivity == 11:
		print('The type of sensitivity used is obtained from smoothing the low integration time. The smoothing function is based on Savitzky-Golay filter. New functions used to correct for negative signal and proportionality')
	elif type_of_sensitivity == 12:
		print('The type of sensitivity used is obtained from smoothing the low integration time. The smoothing function is based on Savitzky-Golay filter. New functions used to correct for negative signal and proportionality applied to averages, not frame by frame')
	else:
		print('Type of sensitivity not specified, error.')
		exit()
	print('The obtained sensitivity is convoluted with a gaussian of the averaged size it is measured with the OES')
except:
	type_of_sensitivity = 5
	print('The type of sensitivity used is obtained from scaling down the signal at high integration time. \nThe smoothed peak of the signal at high int time is made to match the level of the low int time, at the same wavelength location. \nThe peak signal is not averaged across LOS, every LOS is scaled indipendently, though at the same wavelength.')
	print('The obtained sensitivity is convoluted with a gaussian of the averaged size it is measured with the OES')

try:
	if perform_convolution==True:
		print('Convolution of the sensitivity with a Gaussian width based on the detected line shape')
		mod_convolution = '_non_convoluted'
	else:
		print('Convolution of the sensitivity with a Gaussian of fixed (20 pixels) FWHM')
		mod_convolution = ''
except:
	perform_convolution==False
	print('Convolution of the sensitivity with a Gaussian of fixed (20 pixels) FWHM')
	mod_convolution = ''

no_calculate_presets = 0
if (merge_ID_target<=26 or merge_ID_target==53):
	binnedSens = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_5/Sensitivity_1.npy')
	# limits_angle = [17, 26]
	# limits_tilt = [17, 26]
	# limits_wave = [17, 26]
	limits_angle = [0,0]
	limits_tilt = [0,0]
	limits_wave = [0,0]
	angle =	0.0592948797259
	tilt_4_points = np.array([[    0.    ,      1238.48061214],
	 [ 1608.    ,      1257.4354255 ],
	 [    0.     ,       33.24139596],
	 [ 1608.      ,      14.41433151]])
	tilt = [2.95701923e+01 ,  1.74358974e-03 ,  2.76814226e+01]
	waveLcoefs = np.ones((2, 3)) * np.nan
	waveLcoefs[1] =  [9.53889868e-07 ,  1.01412139e-01 ,  3.41962722e+02]
	if (last_merge_done <= 26 or last_merge_done == 53):
		no_calculate_presets = 1
elif (merge_ID_target<=31):
	binnedSens = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_'+str(type_of_sensitivity)+'/Sensitivity_1.npy')
	# limits_angle = [27, 31]
	# limits_tilt = [27, 31]
	# limits_wave = [27, 31]
	limits_angle = [0,0]
	limits_tilt = [0,0]
	limits_wave = [0,0]
	angle =	0.281032868172
	tilt_4_points = np.array([[   0, 1272],
 [1608, 1272],
 [   0 ,   0],
 [1608,    0]])
	tilt = [29.42186235    ,      np.nan , 44.43454791]
	waveLcoefs = np.ones((2, 3)) * np.nan
	waveLcoefs[1] =  [-2.52224584e-06 ,  1.08349203e-01  , 3.39974701e+02]
	if ( (last_merge_done <=31) and not (last_merge_done<=26 or last_merge_done==53) ):
		no_calculate_presets = 1
elif (merge_ID_target<=39):
	binnedSens = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/Sensitivity_3.npy')
	limits_angle = [32,39]
	limits_tilt = [32,39]
	limits_wave = [32,39]
	if ( (last_merge_done<=39) and not (last_merge_done<=31) ):
		no_calculate_presets = 1
elif (merge_ID_target<=52 or merge_ID_target==54):
	binnedSens = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_'+str(type_of_sensitivity)+'/Sensitivity_2.npy')
	# limits_angle = [40,52]
	# limits_tilt = [40,52]
	# limits_wave = [40,52]
	limits_angle = [0, 0]
	limits_tilt = [0, 0]
	limits_wave = [0, 0]
	angle = -0.114920732767
	tilt_4_points = np.array([[    0.    ,      1187.97607656],
	 [ 1608.    ,      1211.05741627],
	 [    0.     ,       20.40119617],
	 [ 1608.      ,      19.24712919]])
	tilt = [2.83611336e+01 , -2.97883382e-03  , 2.83526705e+01]
	waveLcoefs = np.ones((2, 3)) * np.nan
	waveLcoefs[1] = [-1.11822268e-06 ,  1.09054956e-01  , 3.48591929e+02]
	if ( (last_merge_done<=52 or last_merge_done==54) and not (merge_ID_target<=39) ):
		no_calculate_presets = 1
elif (merge_ID_target <= 84):
	binnedSens = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_'+str(type_of_sensitivity)+'/Sensitivity_4'+mod_convolution+'_smoothed.npy')
	binnedSens_sigma = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_'+str(type_of_sensitivity)+'/Sensitivity_4'+mod_convolution+'_sigma.npy')
	binnedSens_waveLcoefs = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_'+str(type_of_sensitivity)+'/calibration_waveLcoefs.npy')
	# limits_angle = [66,84]
	# limits_tilt = [66,84]
	# limits_wave = [66,84]
	limits_angle = [0,0]
	limits_tilt = [0,0]
	limits_wave = [0,0]
	angle = 0.808817942408
	tilt_4_points = np.array([[    0.    ,      1044.57932421],
	 [ 1608.    ,      1050.89097744],
	 [    0.     ,       62.26541353],
	 [ 1608.      ,      17.73308271]])
	tilt = [2.45694163e+01 ,  5.17285743e-04 ,  2.36518312e+01]
	waveLcoefs = np.ones((2, 3)) * np.nan
	waveLcoefs[1] =  [  2.84709039e-06 ,  1.19975728e-01 ,  3.18673899e+02]
	dx_to_etrapolate_to = 140
	if (last_merge_done<=84 and last_merge_done>=66):
		no_calculate_presets = 1
elif (merge_ID_target <= 1000):
	binnedSens = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_'+str(type_of_sensitivity)+'/Sensitivity_4'+mod_convolution+'_smoothed.npy')
	binnedSens_sigma = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_'+str(type_of_sensitivity)+'/Sensitivity_4'+mod_convolution+'_sigma.npy')
	binnedSens_waveLcoefs = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_'+str(type_of_sensitivity)+'/calibration_waveLcoefs.npy')
	# limits_angle = [85,98]
	# limits_tilt = [85,98]
	# limits_wave = [85,98]
	limits_angle = [0,0]
	limits_tilt = [0,0]
	limits_wave = [0,0]
	angle = 0.773281098404
	tilt_4_points = np.array([[    0.    ,      1042.85026968],
	 [ 1608.    ,      1051.36500268],
	 [    0.     ,       61.22662907],
	 [ 1608.      ,      16.9049927 ]])
	tilt = [2.45894737e+01 ,  6.54521132e-04 ,  2.35137447e+01]
	waveLcoefs = np.ones((2, 3)) * np.nan
	waveLcoefs[1] =  [  2.49416833e-06 ,  1.20808818e-01 ,  3.18155463e+02]
	dx_to_etrapolate_to = 140
	if (last_merge_done<=1000 and last_merge_done>=85):
		no_calculate_presets = 1


geom_null = pd.DataFrame([['angle','tilt','binInterv','bin00a','bin00b']],columns = ['angle','tilt','binInterv','bin00a','bin00b'])
geom_null.loc[0] = [0,0,0,0,0]


if no_calculate_presets==0:
	if np.sum(limits_angle)!=0:
		angle=[]
		for merge in range(limits_angle[0],limits_angle[1]+1):
			print('merge ' + str(merge))
			all_j=find_index_of_file(merge,df_settings,df_log,only_OES=True)
			data_sum=0
			for j in all_j:
				dataDark = load_dark(j,df_settings,df_log,fdir,geom_null)
				(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
				type = '.tif'
				filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(int(sequence))+'/Untitled_'+str(int(untitled))+'/Pos0', type)
				type = '.txt'
				filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(int(sequence)) + '/Untitled_' + str(int(untitled)) + '/Pos0',type)[0]
				(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain,Noise) = get_metadata(fdir, folder,sequence, untitled,filename_metadata)

				# plt.figure()
				data_all = []
				for index, filename in enumerate(filenames):
					fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(int(sequence)) + '/Untitled_' + str(int(untitled)) + '/Pos0/' + filename
					im = Image.open(fname)
					data = np.array(im)
					data = (data - dataDark) * Gain[index]
					data = fix_minimum_signal3(data)
					data_sum+=data
			# 		data_all.append(data)
			# data_all = np.array(data_all)
			# data_mean = np.mean(data_all,axis=0)
			try:
				# angle.append(get_angle_2(data_mean,nLines=3))
				angle.append(get_angle_2(data_sum, nLines=3))
				print(angle)
			except:
				print('FAILED')
		angle = np.array(angle)
		print([angle])
		angle = np.nansum(angle[:,0]/ (angle[:,1]**2))/np.nansum(1/angle[:,1]**2)
		print('angle')
		print(angle)


	if np.sum(limits_tilt)!=0:
		tilt_4_points=[]
		for merge in range(limits_tilt[0],limits_tilt[1]+1):
			print('merge ' + str(merge))
			all_j=find_index_of_file(merge,df_settings,df_log,only_OES=True)
			data_sum=0
			for j in all_j:
				dataDark = load_dark(j,df_settings,df_log,fdir,geom_null)
				(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
				type = '.tif'
				filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(int(sequence))+'/Untitled_'+str(int(untitled))+'/Pos0', type)
				type = '.txt'
				filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(int(sequence)) + '/Untitled_' + str(int(untitled)) + '/Pos0',type)[0]
				(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain,Noise) = get_metadata(fdir, folder,sequence, untitled,filename_metadata)

				# plt.figure()
				data_all = []
				for index, filename in enumerate(filenames):
					fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(int(sequence)) + '/Untitled_' + str(int(untitled)) + '/Pos0/' + \
							filename
					im = Image.open(fname)
					data = np.array(im)
					data = (data - dataDark) * Gain[index]
					data = fix_minimum_signal3(data)
					data_sum += data
			# 		data_all.append(data)
			# data_all = np.array(data_all)
			# data_mean = np.mean(data_all,axis=0)
			# data_mean = rotate(data_mean, angle)
			data_sum = rotate(data_sum, angle)
			try:
				# tilt_4_points.append(get_4_points(data_mean, nLines=5))
				tilt_4_points.append(get_4_points(data_sum, nLines=5))
				print(tilt_4_points)
			except:
				print('FAILED')
		tilt_4_points = np.array(tilt_4_points)
		print([tilt_4_points])
		tilt_4_points = np.nanmedian(tilt_4_points,axis=0)
		if np.sum(np.isnan(tilt_4_points))>0:
			tilt_4_points = np.array([[0,np.shape(data)[0]],[np.shape(data)[1],np.shape(data)[0]],[0,0],[np.shape(data)[1],0]])
		print('tilt_4_points')
		print(tilt_4_points)


	if np.sum(limits_tilt)!=0:
		tilt=[]
		for merge in range(limits_tilt[0],limits_tilt[1]+1):
			print('merge ' + str(merge))
			all_j=find_index_of_file(merge,df_settings,df_log,only_OES=True)
			data_sum=0
			for j in all_j:
				dataDark = load_dark(j,df_settings,df_log,fdir,geom_null)
				(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
				type = '.tif'
				filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(int(sequence))+'/Untitled_'+str(int(untitled))+'/Pos0', type)
				type = '.txt'
				filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(int(sequence)) + '/Untitled_' + str(int(untitled)) + '/Pos0',type)[0]
				(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain,Noise) = get_metadata(fdir, folder,sequence, untitled,filename_metadata)

				# plt.figure()
				data_all = []
				for index, filename in enumerate(filenames):
					fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(int(sequence)) + '/Untitled_' + str(int(untitled)) + '/Pos0/' + \
							filename
					im = Image.open(fname)
					data = np.array(im)
					data = (data - dataDark) * Gain[index]
					data = fix_minimum_signal3(data)
					data_sum += data
			# 		data_all.append(data)
			# data_all = np.array(data_all)
			# data_mean = np.mean(data_all,axis=0)
			# data_mean = rotate(data_mean, angle)
			# data_mean = four_point_transform(data_mean, tilt_4_points)
			data_sum = rotate(data_sum, angle)
			data_sum = four_point_transform(data_sum, tilt_4_points)
			try:
				# tilt.append(get_tilt(data_mean, nLines=3))
				tilt.append(get_tilt(data_sum, nLines=3))
				print(tilt)
			except:
				print('FAILED')
		tilt = np.array(tilt)
		print([tilt])
		tilt = np.nanmedian(tilt,axis=0)
		print('tilt')
		print(tilt)


	geom = pd.DataFrame([['angle','tilt','binInterv','bin00a','bin00b']],columns = ['angle','tilt','binInterv','bin00a','bin00b'])
	geom.loc[0] = [angle, tilt_4_points, tilt[0], tilt[2], tilt[2]]
	geom_store = copy.deepcopy(geom)
	print(geom)

	if np.sum(limits_wave)!=0:
		waveLcoefs = []
		for merge in range(limits_wave[0], limits_wave[1]+1):
			print('merge '+str(merge))
			all_j=find_index_of_file(merge,df_settings,df_log,only_OES=True)
			data_sum=0
			for j in all_j:
				dataDark = load_dark(j,df_settings,df_log,fdir,geom_null)
				(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
				type = '.tif'
				filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(int(sequence))+'/Untitled_'+str(int(untitled))+'/Pos0', type)
				type = '.txt'
				filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(int(sequence)) + '/Untitled_' + str(int(untitled)) + '/Pos0',type)[0]
				(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain,Noise) = get_metadata(fdir, folder,sequence, untitled,filename_metadata)

				# plt.figure()
				data_all = []
				for index, filename in enumerate(filenames):
					fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(int(sequence)) + '/Untitled_' + str(int(untitled)) + '/Pos0/' + \
							filename
					im = Image.open(fname)
					data = np.array(im)
					data = (data - dataDark) * Gain[index]
					data = fix_minimum_signal3(data)
					data_sum += data
			# 		data_all.append(data)
			# data_all = np.array(data_all)
			# data_mean = np.mean(data_all,axis=0)
			# data_mean = rotate(data_mean, angle)
			# data_mean = four_point_transform(data_mean, tilt_4_points)
			data_sum = rotate(data_sum, angle)
			data_sum = four_point_transform(data_sum, tilt_4_points)
			binnedData,trash = binData_with_sigma(data_sum_tilted,np.ones_like(data_sum_tilted),geom['bin00b'],geom['binInterv'],check_overExp=False)
			# try:
			# waveLcoefs.append(do_waveL_Calib(merge, df_settings, df_log, geom, fdir=fdir))
			waveLcoefs.append(do_waveL_Calib_simplified(binnedData))
			# except:
			# 	print('FAILED')
		waveLcoefs = np.array(waveLcoefs)
		print([waveLcoefs])
		waveLcoefs = np.nanmedian(waveLcoefs,axis=0)
else:
	geom = copy.deepcopy(geom_store)
	print('Averaged geometry data taken from last merge analysed')
	print(geom)
print('waveLcoefs')
print(waveLcoefs)


logdir = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/'
# calfname = 'calLog.xlsx'
#
# df_calLog = pd.read_excel(logdir + calfname)
# df_calLog.columns = ['folder', 'date', 'time', 'gratPos', 't_exp', 'frames', 'ln/mm', 'gratTilt', 'I_det', 'type',
# 					 'exper', 'Dark']
# df_calLog['folder'] = df_calLog['folder'].astype(str)

spherefname = r'Labsphere_Radiance.xls'
df_sphere = pd.read_excel(logdir + spherefname)
I_nom = df_sphere.columns[1].split()[-3:-1]  # Nominal calibration current as string in A
I_nom = float(I_nom[0]) * 10 ** (int(I_nom[1][-2:]) + 6)  # float, \muA
df_sphere.columns = ['waveL', 'RadRaw']
df_sphere.units = ['nm', 'W cm-2 sr-1 nm-1', 'W m-2 sr-1 nm-1 \muA-1']
df_sphere['Rad'] = df_sphere.RadRaw / I_nom * 1e4


# coordinates for the counts to signal calibration
x_calibration = np.array([  0.        ,   0.        ,   0.        ,   0.        ,
     0.        ,   0.        ,   0.        ,   0.        ,
     0.        ,   0.        ,   0.        ,   0.        ,
     0.        ,   0.        ,   0.        ,   0.        ,
     0.        ,   0.        ,   0.        ,   0.        ,
     0.        ,   0.        ,   0.        ,   0.        ,
     0.        ,   0.        ,   0.        ,   0.        ,
     0.        ,   0.        ,   0.        ,   0.        ,
     0.        ,   0.        ,   0.        ,   0.        ,
     0.        ,   0.        ,   0.        ,   0.        ,
     0.        ,   0.        ,   0.        ,   0.        ,
     0.        ,   0.        ,   0.        ,   0.        ,
     0.        ,   0.        , 0.0 ,  0.01719774,   0.05635798,
     0.10596626,   0.15878511,   0.2206661 ,   0.29043031,
     0.37521531,   0.45810353,   0.56199616,   0.6741604 ,
     0.79951203,   0.93815505,   1.07811153,   1.24027326,
     1.4061498 ,   1.59427947,   1.79589524,   2.01043855,
     2.24370014,   2.48498354,   2.73161621,   2.99880212,
     3.2814409 ,   3.57592368,   3.89605302,   4.29156186,
     4.70707071,   5.11111111,   5.51515152,   5.91919192,
     6.32323232,   6.72727273,   7.13131313,   7.53535354,
     7.93939394,   8.34343434,   8.74747475,   9.15151515,
     9.55555556,   9.95959596,  10.36363636,  10.76767677,
    11.17171717,  11.57575758,  11.97979798,  12.38383838,
    12.78787879,  13.19191919,  13.5959596 ,  14.        ])
# print([y_calibration])
y_calibration=np.array([  80.        ,   80.4040404 ,   80.80808081,   81.21212121,
     81.61616162,   82.02020202,   82.42424242,   82.82828283,
     83.23232323,   83.63636364,   84.04040404,   84.44444444,
     84.84848485,   85.25252525,   85.65656566,   86.06060606,
     86.46464646,   86.86868687,   87.27272727,   87.67676768,
     88.08080808,   88.48484848,   88.88888889,   89.29292929,
     89.6969697 ,   90.1010101 ,   90.50505051,   90.90909091,
     91.31313131,   91.71717172,   92.12121212,   92.52525253,
     92.92929293,   93.33333333,   93.73737374,   94.14141414,
     94.54545455,   94.94949495,   95.35353535,   95.75757576,
     96.16161616,   96.56565657,   96.96969697,   97.37373737,
     97.77777778,   98.18181818,   98.58585859,   98.98989899,
     99.39393939,   99.7979798 , 100.0 ,100.2020202 ,  100.60606061,
    101.01010101,  101.41414141,  101.81818182,  102.22222222,
    102.62626263,  103.03030303,  103.43434343,  103.83838384,
    104.24242424,  104.64646465,  105.05050505,  105.45454545,
    105.85858586,  106.26262626,  106.66666667,  107.07070707,
    107.47474747,  107.87878788,  108.28282828,  108.68686869,
    109.09090909,  109.49494949,  109.8989899 ,  110.3030303 ,
    110.70707071,  111.11111111,  111.51515152,  111.91919192,
    112.32323232,  112.72727273,  113.13131313,  113.53535354,
    113.93939394,  114.34343434,  114.74747475,  115.15151515,
    115.55555556,  115.95959596,  116.36363636,  116.76767677,
    117.17171717,  117.57575758,  117.97979798,  118.38383838,
    118.78787879,  119.19191919,  119.5959596 ,  120.        ])
# y_calibration = y_calibration - 100

# correlation found with examine_sensitivity_12_only_for_noise_vs_signal.py
std_vs_counts = lambda x, a,b,x0: np.maximum(2,a*(np.maximum(0,x-x0)**b))
std_vs_counts_fit = np.array([ 1.77615007,  0.45271138, 94.47173805])

# Here I put multiple pictures together to form a full image in time

# merge_ID_target = 53
time_resolution_scan = False
time_resolution_scan_improved = True
time_resolution_extra_skip = 0


# if (merge_ID_target<=31 or merge_ID_target==53):
# 	binnedSens = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/Sensitivity_1.npy')
# elif merge_ID_target<=52:
# 	binnedSens = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/Sensitivity_2.npy')
# 	waveLcoefs = np.ones_like(waveLcoefs)*np.nan
# 	waveLcoefs[1] = [ -1.39629946e-06,   1.09550955e-01,   3.49466935e+02]	#	from examine_sensitivity.py
# 	geom.loc[0] = [-0.098421420944948226,4.35085277e-03,2.80792602e+01,4.57449437e+01,4.57449437e+01]	# from analysis_filtered_data.py
# 	geom_store = copy.deepcopy(geom)


started=0
rows_range_for_interp = geom_store['binInterv'][0]/3 # rows that I use for interpolation (box with twice this side length, not sphere)
# if merge_ID_target>=66:
# 	rows_range_for_interp = geom_store['binInterv'][0] / 6
if time_resolution_scan:
	conventional_time_step = 0.01	# ms
else:
	conventional_time_step = 0.05	# ms
# interpolation_type = 'quadratic'	# 'linear' or 'quadratic'
grade_of_interpolation = 3	#this is the exponent used for the weights of the interpolation for image resampling
type_of_image = '12bit'	# '12bit' or '16bit'
# if type_of_image=='12bit':
row_shift=2*10280/1000000	# ms
# elif type_of_image=='16bit':
# 	print('Row shift to be checked')
# 	exit()
# time_range_for_interp = rows_range_for_interp*row_shift
# merge_time_window=[-1,4]
# merge_time_window=[-10,10]	# this is now in the input file
# merge_time_window=[0,1]
overexposed_treshold = 3600
data_sum=0
merge_values = []
merge_values_sigma = []
merge_time = []
merge_row = []
merge_Gain = []
merge_Noise = []
merge_overexposed = []
merge_wavelengths = []
# step to merge together all the time depemdent camera images
path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)
if True:
# if True:
	merge_ID_target = 95
	all_j=find_index_of_file(merge_ID_target,df_settings,df_log,only_OES=True)
	temp_gain = []
	for j in all_j:
		(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
		type = '.txt'
		filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2g}".format(int(sequence)) + '/Untitled_' + str(int(untitled)) + '/Pos0', type)[0]
		# filename_metadata = functions.all_file_names(pathfiles, type)[0]
		(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain,Noise) = get_metadata(fdir, folder, sequence,untitled,filename_metadata)
		temp_gain.append(Gain[0])
	all_j = np.array([all_j for _, all_j in sorted(zip(temp_gain, all_j))])

	# first loop only to find bininterv and first bin for fix_minimum_signal_experiment
	data_sum=0
	for j in all_j:
		(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
		type = '.tif'
		filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2g}".format(int(sequence))+'/Untitled_'+str(int(untitled))+'/Pos0', type)
		(CB_to_OES_initial_delay,incremental_step,first_pulse_at_this_frame,bad_pulses_indexes,number_of_pulses) = df_log.loc[j,['CB_to_OES_initial_delay','incremental_step','first_pulse_at_this_frame','bad_pulses_indexes','number_of_pulses']]
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

		data_all=[]
		for index,filename in enumerate(filenames):
			if (index<first_pulse_at_this_frame-1 or index>=(first_pulse_at_this_frame+number_of_pulses-1)):
				continue
			elif (index-first_pulse_at_this_frame+2) in bad_pulses_indexes:
				continue
			if time_resolution_scan:
				if index%(1+time_resolution_extra_skip)!=0:
					continue	# The purpose of this is to test how things change for different time skippings
			print(filename)
			fname = fdir+'/'+folder+'/'+"{0:0=2d}".format(int(sequence))+'/Untitled_'+str(int(untitled))+'/Pos0/'+filename
			im = Image.open(fname)
			data = np.array(im)
			data_all.append(data)
		data_sum += np.mean(data_all,axis=0)
	data_sum = medfilt(data_sum,[3,3])
	wavelength_1,wavelength_2,intermediate_wavelength,last_wavelength = get_line_position(data_sum,4)
	tilt_column_1 = get_bin_and_interv_specific_wavelength(fix_minimum_signal2(data_sum),wavelength_1)
	tilt_column_2 = get_bin_and_interv_specific_wavelength(fix_minimum_signal2(data_sum),wavelength_2)
	tilt_intermediate_column = get_bin_and_interv_specific_wavelength(fix_minimum_signal2(data_sum),intermediate_wavelength)
	tilt_last_column = get_bin_and_interv_specific_wavelength(fix_minimum_signal2(data_sum),last_wavelength)

	# second loop in which I check for binning parameters
	data_sum=0
	dataDark_all = []
	for j in all_j:
		dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
		dataDark_all.append(dataDark)
		(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
		type = '.tif'
		filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2g}".format(int(sequence)) + '/Untitled_' + str(int(untitled)) + '/Pos0', type)
		# filenames = functions.all_file_names(pathfiles, type)
		type = '.txt'
		filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2g}".format(int(sequence)) + '/Untitled_' + str(int(untitled)) + '/Pos0', type)[0]
		# filename_metadata = functions.all_file_names(pathfiles, type)[0]
		(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain,Noise) = get_metadata(fdir, folder,sequence,untitled,filename_metadata)
		if PixelType[-1] == 12:
			time_range_for_interp = rows_range_for_interp * row_shift / 2
		elif PixelType[-1] == 16:
			time_range_for_interp = rows_range_for_interp * row_shift
		(CB_to_OES_initial_delay, incremental_step, first_pulse_at_this_frame, bad_pulses_indexes,
		 number_of_pulses) = df_log.loc[
			j, ['CB_to_OES_initial_delay', 'incremental_step', 'first_pulse_at_this_frame', 'bad_pulses_indexes',
				'number_of_pulses']]
		if bad_pulses_indexes == '':
			bad_pulses_indexes = [0]
		elif (isinstance(bad_pulses_indexes, float) or isinstance(bad_pulses_indexes, int)):
			bad_pulses_indexes = [bad_pulses_indexes]
		else:
			bad_pulses_indexes = bad_pulses_indexes.replace(' ', '').split(',')
			bad_pulses_indexes = list(map(int, bad_pulses_indexes))
		data_all=[]
		for index, filename in enumerate(filenames):
			if (index < first_pulse_at_this_frame - 1 or index > (
					first_pulse_at_this_frame + number_of_pulses - 1)):
				data_all.append(np.zeros_like(dataDark))
				continue
			elif (index - first_pulse_at_this_frame + 2) in bad_pulses_indexes:
				data_all.append(np.zeros_like(dataDark))
				continue
			if time_resolution_scan:
				if index % (1 + time_resolution_extra_skip) != 0:
					continue  # The purpose of this is to test how things change for different time skippings
			print(filename)
			fname = fdir + '/' + folder + '/' + "{0:0=2g}".format(int(sequence)) + '/Untitled_' + str(int(untitled)) + '/Pos0/' + filename
			im = Image.open(fname)
			data = np.array(im)
			# data = data - dataDark  # I checked that for 16 or 12 bit the dark is always around 100 counts
			# data = data * Gain[index]
			# data = fix_minimum_signal3(data)

			data_sigma=np.sqrt(data)
			additive_factor,additive_factor_sigma = fix_minimum_signal_experiment(data,intermediate_wavelength,last_wavelength,tilt_intermediate_column,tilt_last_column,counts_treshold_fixed_increase=106,dx_to_etrapolate_to=dx_to_etrapolate_to)
			data = (data.T + additive_factor).T
			data_sigma = np.sqrt((data_sigma**2).T + (additive_factor_sigma**2)).T

			data_all.append(data)
			# merge_Noise.extend(np.ones((len(data)))*Noise[index])

		data_sum += np.mean(data_all,axis=0)

	dataDark = np.mean(dataDark_all,axis=0)
	path_filename = path_where_to_save_everything+'/merge' + str(merge_ID_target) + '_stats.csv'
	# if not os.path.exists(path_filename):
	file = open(path_filename, 'w')
	writer = csv.writer(file)
	writer.writerow(['Stats about merge ' + str(merge_ID_target)])
	# file.close()

	data_sum = medfilt(data_sum,[3,3])
	intermediate_wavelength,last_wavelength = get_line_position(data_sum,2)
	tilt_intermediate_column = get_bin_and_interv_specific_wavelength(fix_minimum_signal2(data_sum),intermediate_wavelength)
	writer.writerow(['LOS start and bin interval at '+str(intermediate_wavelength)+' wavelength axis', str(tilt_intermediate_column)])
	tilt_last_column = get_bin_and_interv_specific_wavelength(fix_minimum_signal2(data_sum),last_wavelength)
	writer.writerow(['LOS start and bin interval at '+str(last_wavelength)+' wavelength axis', str(tilt_last_column)])

	result = 0
	nLines = 8
	while (result == 0 and nLines > 0):
		try:
			angle = get_angle_2(data_sum, nLines=nLines)
			print(angle)
			result = 1
			geom['angle'] = np.nansum(np.multiply(angle[0], np.divide(1, np.power(angle[1], 2)))) / np.nansum(np.divide(1, np.power(angle[1], 2)))
			print(geom['angle'])
			writer.writerow(['Specific angle of ', str(geom['angle'][0]), ' found with nLines=', str(nLines)])
		except:
			nLines -= 1
	# if np.abs(geom['angle'][0])>2*np.abs(geom_store['angle'][0]):
	if np.abs(geom['angle'][0]-geom_store['angle'][0]) > np.abs(geom_store['angle'][0]):
		geom['angle'][0] = geom_store['angle'][0]
		writer.writerow(['No specific angle found. Used standard of ', str(geom['angle'][0])])
	data_sum = rotate(data_sum, geom['angle'][0])
	# file = open(path_filename, 'r')
	result = 0
	nLines = 8
	tilt_4_points = np.nan
	tilt = (np.nan,np.nan,np.nan)
	# while ((result == 0 or np.isnan(tilt[1]) or np.isnan(tilt[2])) and nLines > 0 ):
	while ((result == 0 or np.isnan(tilt[0]) or np.isnan(tilt[2])) and nLines > 0):
		try:
			tilt_4_points = get_4_points(data_sum, nLines=nLines)
			if np.sum(np.isnan(tilt_4_points)) > 0:
				tilt_4_points = np.array([[0, np.shape(data_sum)[0]],
										  [np.shape(data_sum)[1], np.shape(data_sum)[0]],
										  [0, 0],
										  [np.shape(data_sum)[1], 0]])
			data_sum_tilted = four_point_transform(data_sum, tilt_4_points)
			tilt = get_tilt(data_sum_tilted, nLines=nLines)
			print(tilt_4_points)
			print(tilt)
			result = 1
			geom['binInterv'] = tilt[0]
			geom['tilt'][0] = np.array(tilt_4_points)
			geom['bin00a'] = tilt[2]
			geom['bin00b'] = tilt[2]
			nLines -= 1
			writer.writerow(['Specific tilt of ', str(geom['tilt'][0]), ' found with nLines=', str(nLines), 'binInterv',str(geom['binInterv'][0]), 'first bin', str(geom['bin00a'][0])])
		except:
			nLines -= 1
	if (np.isnan(tilt[0]) or np.isnan(tilt[2])):
		geom['binInterv'] = geom_store['binInterv']
		geom['tilt'] = geom_store['tilt']
		geom['bin00a'] = geom_store['bin00a']
		geom['bin00b'] = geom_store['bin00b']
		writer.writerow(['No specific tilt found. Used standard of ', str(geom['tilt'][0]), 'binInterv',
					 str(geom['binInterv'][0]), 'first bin', str(geom['bin00a'][0])])

	try:
		data_sum_tilted = four_point_transform(data_sum, geom['tilt'][0])
		binnedData,trash = binData_with_sigma(data_sum_tilted,np.ones_like(data_sum_tilted),geom['bin00b'],geom['binInterv'],check_overExp=False)
		waveLcoefs = do_waveL_Calib_simplified(binnedData)
		print('waveLcoefs')
		print(waveLcoefs)
		writer.writerow(['Specific wavelength coefficients found' ,str(waveLcoefs)])
	except:
		writer.writerow(['No specific wavelength coefficients found. Used standard of' ,str(waveLcoefs)])


	merge_ID_target = 188
	all_j=find_index_of_file(merge_ID_target,df_settings,df_log,only_OES=True)
	temp_gain = []
	for j in all_j:
		(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
		type = '.txt'
		filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2g}".format(int(sequence)) + '/Untitled_' + str(int(untitled)) + '/Pos0', type)[0]
		# filename_metadata = functions.all_file_names(pathfiles, type)[0]
		(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain,Noise) = get_metadata(fdir, folder, sequence,untitled,filename_metadata)
		temp_gain.append(Gain[0])
	all_j = np.array([all_j for _, all_j in sorted(zip(temp_gain, all_j))])

	# second loop in which I check for binning parameters
	data_sum=0
	data_sum_sigma=0
	dataDark_all = []
	row_steps = np.linspace(0, roi_tr[1] - 1, roi_tr[1])
	wavelengths = np.linspace(0, roi_tr[0] - 1, roi_tr[0])
	for j in all_j:
		dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
		dataDark_all.append(dataDark)
		(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
		type = '.tif'
		filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2g}".format(int(sequence)) + '/Untitled_' + str(int(untitled)) + '/Pos0', type)
		# filenames = functions.all_file_names(pathfiles, type)
		type = '.txt'
		filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2g}".format(int(sequence)) + '/Untitled_' + str(int(untitled)) + '/Pos0', type)[0]
		# filename_metadata = functions.all_file_names(pathfiles, type)[0]
		(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain,Noise) = get_metadata(fdir, folder,sequence,untitled,filename_metadata)
		if PixelType[-1] == 12:
			time_range_for_interp = rows_range_for_interp * row_shift / 2
		elif PixelType[-1] == 16:
			time_range_for_interp = rows_range_for_interp * row_shift
		(CB_to_OES_initial_delay, incremental_step, first_pulse_at_this_frame, bad_pulses_indexes,
		 number_of_pulses) = df_log.loc[
			j, ['CB_to_OES_initial_delay', 'incremental_step', 'first_pulse_at_this_frame', 'bad_pulses_indexes',
				'number_of_pulses']]
		if bad_pulses_indexes == '':
			bad_pulses_indexes = [0]
		elif (isinstance(bad_pulses_indexes, float) or isinstance(bad_pulses_indexes, int)):
			bad_pulses_indexes = [bad_pulses_indexes]
		else:
			bad_pulses_indexes = bad_pulses_indexes.replace(' ', '').split(',')
			bad_pulses_indexes = list(map(int, bad_pulses_indexes))
		data_all=[]
		data_all_sigma = []
		for index, filename in enumerate(filenames):
			if (index < first_pulse_at_this_frame - 1 or index > (
					first_pulse_at_this_frame + number_of_pulses - 1)):
				data_all.append(np.zeros_like(dataDark))
				continue
			elif (index - first_pulse_at_this_frame + 2) in bad_pulses_indexes:
				data_all.append(np.zeros_like(dataDark))
				continue
			if time_resolution_scan:
				if index % (1 + time_resolution_extra_skip) != 0:
					continue  # The purpose of this is to test how things change for different time skippings
			print(filename)
			fname = fdir + '/' + folder + '/' + "{0:0=2g}".format(int(sequence)) + '/Untitled_' + str(int(untitled)) + '/Pos0/' + filename
			im = Image.open(fname)
			if np.shape(im)[0]==1100:
				data = np.array(im)
			else:
				data = np.array(im)[290:290+1100]	# done so it has the same 1100 nomber of rows as merge85
			# data = data - dataDark  # I checked that for 16 or 12 bit the dark is always around 100 counts
			# data = data * Gain[index]
			# data = fix_minimum_signal3(data)

			data_sigma=np.sqrt(data - min(data.min(),100) + 0.00001)
			# additive_factor,additive_factor_sigma = fix_minimum_signal_experiment(data,intermediate_wavelength,last_wavelength,tilt_intermediate_column,tilt_last_column,counts_treshold_fixed_increase=106,dx_to_etrapolate_to=dx_to_etrapolate_to)
			# data = (data.T + additive_factor).T
			# data_sigma = np.sqrt((data_sigma**2).T + (additive_factor_sigma**2)).T

			data_all.append(data)
			data_all_sigma.append(data_sigma)

			merge_Gain = np.ones((len(data_all)))*Gain[-1]
			merge_Noise = np.ones((len(data_all)))*Noise[-1]
			merge_overexposed = np.max(data_all,axis=2)>overexposed_treshold/Gain[-1]
			merge_wavelengths = np.ones((len(data_all),1))*wavelengths

		temp = medfilt(np.mean(data_all,axis=0),[3,3])
		additive_factor,additive_factor_sigma = fix_minimum_signal_experiment(temp,intermediate_wavelength,last_wavelength,tilt_intermediate_column,tilt_last_column,counts_treshold_fixed_increase=106,dx_to_etrapolate_to=dx_to_etrapolate_to)
		temp = (temp.T + additive_factor).T
		data_sum += temp
		data_sum_sigma += ((np.std(data_all,axis=0)**2 + ((np.sum(np.array(data_all_sigma)**2,axis=0)**0.5)/len(data_all))**2).T + (additive_factor_sigma**2)).T

	data_sum_sigma = data_sum_sigma**0.5
	dataDark = np.mean(dataDark_all,axis=0)
	# this isn't really necessary because the problem is not present in the dark data
	# additive_factor,additive_factor_sigma = fix_minimum_signal_experiment(dataDark,intermediate_wavelength,last_wavelength,tilt_intermediate_column,tilt_last_column,counts_treshold_fixed_increase=106,dx_to_etrapolate_to=dx_to_etrapolate_to)
	# dataDark = (dataDark.T + additive_factor).T

	np.savez_compressed(path_where_to_save_everything+'/merge' + str(merge_ID_target) + '_merge_tot',data_sum = data_sum , data_sum_sigma = data_sum_sigma , merge_Gain = merge_Gain , merge_Noise = merge_Noise , merge_wavelengths = merge_wavelengths , merge_overexposed = merge_overexposed)

		# merge_tot.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_merge_tot.nc')


# mask = np.zeros_like(data_sum)
data_sum_for_plot = rotate(data_sum, geom['angle'])
data_sum_for_plot = four_point_transform(data_sum_for_plot,geom['tilt'][0])
waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(data_sum)[1]+1))
temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(data_sum_for_plot)[0]+1))
plt.figure(figsize=(20, 10))
plt.pcolor(temp_w,temp_LOS,data_sum_for_plot,cmap='rainbow',rasterized=True,vmin=100)
# plt.imshow(data_sum,'rainbow',origin='lower',vmax=np.max(data_sum[:,:800]))
plt.colorbar()
for i in range(41):
	plt.plot([np.min(waveaxis),np.max(waveaxis)],[geom['bin00b']+i*geom['binInterv'],geom['bin00b']+i*geom['binInterv']],'--k',linewidth=0.5)
	# mask[int(geom['bin00b']+i*geom['binInterv'])-1:int(geom['bin00b']+i*geom['binInterv'])+2]=1
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(data_sum_for_plot)[0]-1],'--k',linewidth=0.5)
plt.title('example of the binning that will be done \nangle, tilt, binInterv, bin00a, bin00b\n%.5g, ' %(geom['angle']) +str(np.around(geom['tilt'][0],decimals=3))+', %.5g, %.5g, %.5g' %(geom['binInterv'],geom['bin00a'],geom['bin00b']))
plt.xlabel('wavelength [nm]')
plt.ylabel('LOS axis [pixels]')
plt.savefig(path_where_to_save_everything + '/' + 'binning_example.eps', bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 5))
plt.pcolor(temp_w,temp_LOS,data_sum_for_plot,cmap='rainbow',rasterized=True,vmin=100)
# plt.imshow(data_sum,'rainbow',origin='lower',vmax=np.max(data_sum[:,:800]))
plt.colorbar()
for i in range(41):
	plt.plot([np.min(waveaxis),np.max(waveaxis)],[geom['bin00b']+i*geom['binInterv'],geom['bin00b']+i*geom['binInterv']],'--k',linewidth=0.5)
	# mask[int(geom['bin00b']+i*geom['binInterv'])-1:int(geom['bin00b']+i*geom['binInterv'])+2]=1
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(data_sum_for_plot)[0]-1],'--k',linewidth=0.5)
plt.title('example of the binning that will be done \nangle, tilt, binInterv, bin00a, bin00b\n%.5g, ' %(geom['angle']) +str(np.around(geom['tilt'][0],decimals=3))+', %.5g, %.5g, %.5g' %(geom['binInterv'],geom['bin00a'],geom['bin00b']))
plt.xlabel('wavelength [nm]')
plt.ylabel('LOS axis [pixels]')
plt.savefig(path_where_to_save_everything + '/' + 'binning_example2.eps', bbox_inches='tight')
plt.close()


# writer.writerow(['the final result will be from', str(np.min(new_timesteps)), ' to ', str(np.max(new_timesteps)),' ms from the beginning of the discharge'])
# writer.writerow(['with a time resolution of ' + str(np.median(np.diff(new_timesteps))), ' ms'])
file.close()

mkl.set_num_threads(1)

# what composite array is, in practice, is a median filter on the column, and smoothed data on the otuer 2 dimentions. given I have only row and column here, a 2d mefian filter will do!
composed_array = np.array([data_sum])
composed_array_sigma = generic_filter([data_sum_sigma],np.max,size=[1,3,3])


# step to correct the gain to compensate for non linearity in the sensitivity
scaled_composed_array=np.zeros_like(composed_array)
scaled_composed_array_sigma=np.zeros_like(composed_array)
neg_sig_corr_composed_array=np.zeros_like(composed_array)
neg_sig_corr_composed_array_sigma=np.zeros_like(composed_array)
additive_factor_list = np.zeros(([*(np.shape(composed_array)[:2])]))

dataDark_calib = apply_proportionality_calibration(dataDark,x_calibration,y_calibration)

Gain = np.unique(merge_Gain)[0]
Noise = np.unique(merge_Noise)[0]
print('Noise '+str(Noise))

data = composed_array[0]
data_sigma = composed_array_sigma[0]
data_sigma_up = data+data_sigma
data_sigma_down = data-data_sigma
data = apply_proportionality_calibration(data,x_calibration,y_calibration)
data_sigma_up = apply_proportionality_calibration(data_sigma_up,x_calibration,y_calibration)
data_sigma_down = apply_proportionality_calibration(data_sigma_down,x_calibration,y_calibration)
data = data - dataDark_calib
data[data<0]=0
data_sigma = (data_sigma_up-data_sigma_down)/2
scaled_composed_array[0]=(data)
scaled_composed_array_sigma[0]=(data_sigma)


gain_scaled_composed_array = scaled_composed_array*Gain
gain_scaled_composed_array_sigma = np.sqrt((scaled_composed_array_sigma*Gain)**2 + Noise**2)

np.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_gain_scaled_composed_array', gain_scaled_composed_array)
np.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_gain_scaled_composed_array_sigma', gain_scaled_composed_array_sigma)
np.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_additive_factor_list', additive_factor_list)

data_sum_for_plot = rotate(gain_scaled_composed_array[0], geom['angle'])
data_sum_for_plot = four_point_transform(data_sum_for_plot,geom['tilt'][0])
waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(data_sum)[1]+1))
temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(data_sum_for_plot)[0]+1))

plt.figure(figsize=(20, 10))
plt.pcolor(temp_w,temp_LOS,data_sum_for_plot,cmap='rainbow',rasterized=True)
# plt.imshow(data_sum,'rainbow',origin='lower',vmax=np.max(data_sum[:,:800]))
plt.colorbar()
for i in range(41):
	plt.plot([np.min(waveaxis),np.max(waveaxis)],[geom['bin00b']+i*geom['binInterv'],geom['bin00b']+i*geom['binInterv']],'--k',linewidth=0.5)
	# mask[int(geom['bin00b']+i*geom['binInterv'])-1:int(geom['bin00b']+i*geom['binInterv'])+2]=1
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(data_sum_for_plot)[0]-1],'--k',linewidth=0.5)
plt.title('example of the binning that will be done \nangle, tilt, binInterv, bin00a, bin00b\n%.5g, ' %(geom['angle']) +str(np.around(geom['tilt'][0],decimals=3))+', %.5g, %.5g, %.5g' %(geom['binInterv'],geom['bin00a'],geom['bin00b']))
plt.xlabel('wavelength [nm]')
plt.ylabel('LOS axis [pixels]')
plt.savefig(path_where_to_save_everything + '/' + 'gain_scaled_composed_array.eps', bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 5))
plt.pcolor(temp_w,temp_LOS,data_sum_for_plot,cmap='rainbow',rasterized=True)
# plt.imshow(data_sum,'rainbow',origin='lower',vmax=np.max(data_sum[:,:800]))
plt.colorbar()
for i in range(41):
	plt.plot([np.min(waveaxis),np.max(waveaxis)],[geom['bin00b']+i*geom['binInterv'],geom['bin00b']+i*geom['binInterv']],'--k',linewidth=0.5)
	# mask[int(geom['bin00b']+i*geom['binInterv'])-1:int(geom['bin00b']+i*geom['binInterv'])+2]=1
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(data_sum_for_plot)[0]-1],'--k',linewidth=0.5)
plt.title('example of the binning that will be done \nangle, tilt, binInterv, bin00a, bin00b\n%.5g, ' %(geom['angle']) +str(np.around(geom['tilt'][0],decimals=3))+', %.5g, %.5g, %.5g' %(geom['binInterv'],geom['bin00a'],geom['bin00b']))
plt.xlabel('wavelength [nm]')
plt.ylabel('LOS axis [pixels]')
plt.savefig(path_where_to_save_everything + '/' + 'gain_scaled_composed_array2.eps', bbox_inches='tight')
plt.close()


# step to do the LOS binning

frame=rotate(gain_scaled_composed_array[0],geom['angle'])
frame=four_point_transform(frame,geom['tilt'][0])
wavelengths = np.linspace(0, np.shape(frame)[1] - 1, np.shape(frame)[1])

# bin_steps = np.linspace(0,39,40)
# binned_data=xr.DataArray(np.zeros((len(new_timesteps),40,np.shape(frame)[1])), dims=['time', 'bin', 'wavelength_axis'],coords={'time': new_timesteps,'bin': bin_steps,'wavelength_axis': wavelengths},name='interpolated array')


class calc_stuff_output:
	def __init__(self, time, binnedData,binnedData_sigma):
		self.time = time
		self.binnedData = binnedData
		self.binnedData_sigma = binnedData_sigma

def calc_stuff(arg,gain_scaled_composed_array=gain_scaled_composed_array,gain_scaled_composed_array_sigma=gain_scaled_composed_array_sigma,geom=geom,calc_stuff_output=calc_stuff_output):
	import numpy as np
	from functions.spectools import rotate
	from functions.fabio_add import four_point_transform
	from functions.spectools import binData,binData_with_sigma
	index = arg[0]
	time = arg[1]
	frame = gain_scaled_composed_array[index]
	frame = np.nan_to_num(frame)
	frame=rotate(frame,geom['angle'])
	frame=four_point_transform(frame,geom['tilt'][0])
	frame_sigma = gain_scaled_composed_array_sigma[index]
	frame_sigma = np.nan_to_num(frame_sigma)
	frame_sigma=rotate(frame_sigma,geom['angle'])
	frame_sigma=four_point_transform(frame_sigma,geom['tilt'][0])
	binnedData,binnedData_sigma = binData_with_sigma(frame,frame_sigma,geom['bin00b'],geom['binInterv'],check_overExp=False)

	# class calc_stuff_output:
	# 	def __init__(self,time,binnedData):
	# 		self.time = time
	# 		self.binnedData = binnedData

	output = calc_stuff_output(time,binnedData,binnedData_sigma)
	return output

pool = Pool(number_cpu_available,maxtasksperchild=1)
binned_data = [*pool.map(calc_stuff, enumerate([0]))]
pool.close()
pool.join()
pool.terminate()
del pool
# binned_data = set(binned_data)

binned_data = list(binned_data)
time_indexes = []
for i in range(len(binned_data)):
	time_indexes.append(binned_data[0].time)
binned_data = np.array([peaks for _, peaks in sorted(zip(time_indexes, binned_data))])
temp = []
for i in range(len(binned_data)):
	temp.append(binned_data[i].binnedData_sigma)
binned_data_sigma = np.array(temp)
temp = []
for i in range(len(binned_data)):
	temp.append(binned_data[i].binnedData)
binned_data = np.array(temp)

np.save(path_where_to_save_everything +'/merge'+str(merge_ID_target)+'_binned_data', binned_data)
np.save(path_where_to_save_everything +'/merge'+str(merge_ID_target)+'_binned_data_sigma', binned_data_sigma)

waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(binned_data[0])[1]+1))
temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(binned_data[0])[0]+1))
plt.figure(figsize=(20, 10))
plt.pcolor(temp_w,temp_LOS,binned_data[0],cmap='rainbow',rasterized=True)
# plt.imshow(data_sum,'rainbow',origin='lower',vmax=np.max(data_sum[:,:800]))
plt.colorbar()
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(binned_data[0])[0]-1],'--k',linewidth=0.5)
plt.title('example of the binning that will be done \nangle, tilt, binInterv, bin00a, bin00b\n%.5g, ' %(geom['angle']) +str(np.around(geom['tilt'][0],decimals=3))+', %.5g, %.5g, %.5g' %(geom['binInterv'],geom['bin00a'],geom['bin00b']))
plt.xlabel('wavelength [nm]')
plt.ylabel('LOS axis [pixels]')
plt.savefig(path_where_to_save_everything + '/' + 'binned_data.eps', bbox_inches='tight')
plt.close()


# step to do the line integration (wavelength)
if not os.path.exists(path_where_to_save_everything+'/spectral_fit_example'):
	os.makedirs(path_where_to_save_everything+'/spectral_fit_example')

class calc_stuff_output:
	def __init__(self, time, fit,fit_sigma):
		self.time = time
		self.fit = fit
		self.fit_sigma = fit_sigma

def calc_stuff(arg,binned_data=binned_data,binned_data_sigma=binned_data_sigma,df_settings=df_settings,df_log=df_log,geom=geom,waveLcoefs=waveLcoefs,binnedSens=binnedSens,binnedSens_sigma=binnedSens_sigma,calc_stuff_output=calc_stuff_output,perform_convolution=perform_convolution):
	import numpy as np
	from functions.SpectralFit import doSpecFit_single_frame,doSpecFit_single_frame_with_sigma
	index = arg[0]
	time = arg[1]
	to_print = False
	to_print = True
	print('fit of ' + str(time) + 'ms')
	try:
		fit,fit_sigma = doSpecFit_single_frame_with_sigma(np.array(binned_data[index]),np.array(binned_data_sigma[index]),df_settings, df_log, geom, waveLcoefs, binnedSens,binnedSens_sigma,path_where_to_save_everything,time,to_print,perform_convolution=perform_convolution,binnedSens_waveLcoefs=binnedSens_waveLcoefs)
		print('ok')
	except:
		print(str(index) + ' fitting failed')
		fit=np.zeros((40,9))*np.nan
		fit_sigma = np.ones_like(fit)

	# class calc_stuff_output:
	# 	def __init__(self, time, fit):
	# 		self.time = time
	# 		self.fit = fit

	output = calc_stuff_output(time, fit,fit_sigma)
	return output

pool = Pool(number_cpu_available,maxtasksperchild=1)
all_fits = [*pool.map(calc_stuff, enumerate([0]))]
pool.close()
pool.join()
pool.terminate()
del pool

all_fits = list(all_fits)
time_indexes = []
for i in range(len(all_fits)):
	time_indexes.append(all_fits[i].time)
all_fits = np.array([peaks for _, peaks in sorted(zip(time_indexes, all_fits))])
temp = []
for i in range(len(all_fits)):
	# print(np.shape(all_fits[i].fit))
	temp.append(np.array(all_fits[i].fit_sigma))
all_fits_sigma = np.array(temp)
all_fits_sigma=np.array(all_fits_sigma).astype(float)
temp = []
for i in range(len(all_fits)):
	# print(np.shape(all_fits[i].fit))
	temp.append(np.array(all_fits[i].fit))
all_fits = np.array(temp)
all_fits=np.array(all_fits).astype(float)



np.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_all_fits',all_fits)
np.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_all_fits_sigma',all_fits_sigma)
# ani = coleval.movie_from_data(np.array([all_fits]), 1000/conventional_time_step, row_shift, 'Transition from Hb' , 'Bin [au]' , 'Intersity [au]',extvmin=0,extvmax=100000)
# ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_all_fits' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

exit()







all_fits = np.load(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_all_fits.npy')
all_fits_sigma = np.load(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_all_fits_sigma.npy')
plt.figure()
plt.errorbar(np.arange(len(all_fits[0,:,0])),all_fits[0,:,0],yerr=all_fits_sigma[0,:,0])
fit = np.polyfit(np.arange(40)[0:5].tolist()+np.arange(40)[-3:].tolist(),all_fits[0,:,0][0:5].tolist()+all_fits[0,:,0][-3:].tolist(),1,w=(1/all_fits_sigma[0,:,0][0:5]).tolist()+(1/all_fits_sigma[0,:,0][-3:]).tolist())
plt.plot(np.arange(40),np.polyval(fit,np.arange(40)),'--')
plt.errorbar(np.arange(len(all_fits[0,:,0])),all_fits[0,:,0]/(np.polyval(fit,np.arange(40))/(np.polyval(fit,np.arange(40)).max())),yerr=all_fits_sigma[0,:,0])



print('mark1')

# dx = 18 / 40 * (50.5 / 27.4) / 1e3
dx = 1.06478505470992 / 1e3	# 10/02/2020 from	Calculate magnification_FF.xlsx
xx = np.arange(40).astype(int) * dx  # m
xn = np.linspace(0, max(xx), 1000)
# r = np.linspace(-max(xx) / 2, max(xx) / 2, 1000)
# r=r[::10]
number_of_radial_divisions=21
r = np.arange(number_of_radial_divisions)*dx

mkl.set_num_threads(number_cpu_available)

# first_time = np.min(new_timesteps)
# last_time = np.max(new_timesteps)

force_glogal_center = np.mean(find_plasma_axis(df_settings,all_fits, merge_ID_target, new_timesteps,dx,xx,r) )
print('center of the plasma located at %.6gmm' %(force_glogal_center*1000))

ss_image = []
ss_image_sigma = []
# time_step = int((np.abs(new_timesteps+0.05)).argmin())	#I select the time 0.05 ms before the pulse
for i_row in range(roi_tr[1]):
	select = np.logical_and(merge_row==i_row,merge_time<-0.05)	#I select the time 0.05 ms before the pulse
	ss_image.append(np.mean(merge_values[select],axis=0))
	ss_image_sigma.append(np.std(merge_values[select],axis=0))
ss_image=np.array(ss_image)
ss_image = ss_image
ss_image_sigma=np.array(ss_image_sigma)
# this is now done to build the merge_values
# additive_factor,additive_factor_sigma = fix_minimum_signal_experiment(ss_image,intermediate_wavelength,last_wavelength,tilt_intermediate_column,tilt_last_column,counts_treshold_fixed_increase=106,dx_to_etrapolate_to=dx_to_etrapolate_to)
# ss_image = (ss_image.T + additive_factor).T
# ss_image_sigma = np.sqrt((ss_image_sigma**2).T + (additive_factor_sigma**2)).T

plt.figure(figsize=(20, 10))
plt.imshow(ss_image,'rainbow',origin='lower')
plt.colorbar()
plt.title('SS image negative corrected')
plt.savefig(path_where_to_save_everything + '/' + 'SS_image_negative_corrected.eps', bbox_inches='tight')
plt.close()

plt.figure(figsize=(20, 10))
plt.imshow(ss_image_sigma,'rainbow',origin='lower')
plt.colorbar()
plt.title('SS image negative corrected sigma')
plt.savefig(path_where_to_save_everything + '/' + 'SS_image_negative_corrected_sigma.eps', bbox_inches='tight')
plt.close()

ss_image_sigma_up = ss_image+ss_image_sigma
ss_image_sigma_down = ss_image-ss_image_sigma
ss_image = apply_proportionality_calibration(ss_image,x_calibration,y_calibration)
ss_image_sigma_up = apply_proportionality_calibration(ss_image_sigma_up,x_calibration,y_calibration)
ss_image_sigma_down = apply_proportionality_calibration(ss_image_sigma_down,x_calibration,y_calibration)
ss_image_sigma = (ss_image_sigma_up-ss_image_sigma_down)/2
dataDark_calib = apply_proportionality_calibration(dataDark,x_calibration,y_calibration)
ss_image = ss_image - dataDark_calib
ss_image[ss_image<0]=0

plt.figure(figsize=(20, 10))
plt.imshow(ss_image,'rainbow',origin='lower')
plt.colorbar()
plt.title('SS image proportionality corrected')
plt.savefig(path_where_to_save_everything + '/' + 'SS_image_prop_corrected.eps', bbox_inches='tight')
plt.close()

plt.figure(figsize=(20, 10))
plt.imshow(ss_image_sigma,'rainbow',origin='lower')
plt.colorbar()
plt.title('SS image proportionality corrected sigma')
plt.savefig(path_where_to_save_everything + '/' + 'SS_image_prop_corrected_sigma.eps', bbox_inches='tight')
plt.close()

Gain = np.unique(merge_Gain)[0]
Noise = np.unique(merge_Noise)[0]
gain_scaled_ss_image = ss_image*Gain
gain_scaled_ss_image_sigma = np.sqrt((ss_image_sigma*Gain)**2 + Noise**2)

frame=rotate(gain_scaled_ss_image,geom['angle'])
frame=four_point_transform(frame,geom['tilt'][0])
frame_sigma=rotate(gain_scaled_ss_image_sigma,geom['angle'])
frame_sigma=four_point_transform(frame_sigma,geom['tilt'][0])

plt.figure(figsize=(20, 10))
plt.imshow(frame,'rainbow',origin='lower')
plt.colorbar()
for i in range(41):
	plt.plot([0,np.shape(data_sum)[-1]],[geom['bin00b']+i*geom['binInterv'],geom['bin00b']+i*geom['binInterv']],'--k',linewidth=0.5)
plt.title('SS image final')
plt.savefig(path_where_to_save_everything + '/' + 'SS_image.eps', bbox_inches='tight')
plt.close()

plt.figure(figsize=(20, 10))
plt.imshow(frame_sigma,'rainbow',origin='lower')
plt.colorbar()
for i in range(41):
	plt.plot([0,np.shape(data_sum)[-1]],[geom['bin00b']+i*geom['binInterv'],geom['bin00b']+i*geom['binInterv']],'--k',linewidth=0.5)
plt.title('SS image final sigma')
plt.savefig(path_where_to_save_everything + '/' + 'SS_image_sigma.eps', bbox_inches='tight')
plt.close()


binnedss_image,binnedss_image_sigma = binData_with_sigma(frame,frame_sigma,geom['bin00b'],geom['binInterv'],check_overExp=False)

fit,fit_sigma = doSpecFit_single_frame_with_sigma(binnedss_image,binnedss_image_sigma,df_settings, df_log, geom, waveLcoefs, binnedSens,binnedSens_sigma,path_where_to_save_everything,999,True,perform_convolution=perform_convolution)
all_fits_ss = np.array(fit).astype(float)
all_fits_ss_sigma = np.array(fit_sigma).astype(float)


np.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_SS_all_fits',all_fits_ss)
np.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_SS_all_fits_sigma',all_fits_ss_sigma)
all_fits_ss = np.load(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_SS_all_fits.npy')
all_fits_ss_sigma = np.load(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_SS_all_fits_sigma.npy')

# all_fits_ss = all_fits[:time_step]
# all_fits_ss[all_fits_ss==0]=np.nan
# all_fits_ss = np.nanmean(all_fits_ss,axis=0)
profile_centre_to_trash = doLateralfit_single_with_sigma(df_settings, all_fits_ss,all_fits_ss_sigma, merge_ID_target,dx,xx,r,same_centre_every_line=True,force_glogal_center=force_glogal_center)
doLateralfit_single_with_sigma_pure_Abel(df_settings, all_fits_ss,all_fits_ss_sigma, merge_ID_target,new_timesteps,dx,xx,r,same_centre_every_line=True,force_glogal_center=force_glogal_center)

mkl.set_num_threads(1)
if not os.path.exists(path_where_to_save_everything+'/abel_inversion_example'):
	os.makedirs(path_where_to_save_everything+'/abel_inversion_example')
exec(open("/home/ffederic/work/Collaboratory/test/experimental_data/functions/doLateralfit_time_tependent_with_sigma.py").read())
mkl.set_num_threads(number_cpu_available)

# if not os.path.exists(path_where_to_save_everything+'/abel_inversion_example'):
# 	os.makedirs(path_where_to_save_everything+'/abel_inversion_example')
# 	doLateralfit_time_tependent_with_sigma(df_settings, all_fits,all_fits_sigma,new_timesteps, merge_ID_target,new_timesteps,dx,xx,r,force_glogal_center=force_glogal_center)#17.5/1000)
# # else:
# # 	shutil.rmtree(path_where_to_save_everything+'/abel_inversion_example')
# # 	os.makedirs(path_where_to_save_everything+'/abel_inversion_example')


# r_actual_new = doLateralfit_time_tependent_LOS_overlapping(df_settings,all_fits,all_fits_sigma, merge_ID_target, new_timesteps,dx,xx,r,N,force_glogal_center)

print('mark2')
inverted_profiles = np.load(path_where_to_save_everything+'/inverted_profiles.npy')
inverted_profiles_sigma = np.load(path_where_to_save_everything+'/inverted_profiles_sigma.npy')



if os.path.exists(path_where_to_save_everything+'/TS_data_merge_'+str(merge_ID_target)+'.npz'):
	merge_Te_prof_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_Te_prof_multipulse']
	merge_dTe_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_dTe_multipulse']
	merge_ne_prof_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_ne_prof_multipulse']
	merge_dne_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_dne_multipulse']
	merge_time = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_time']

	dt = np.nanmedian(np.diff(new_timesteps))
	TS_dt = np.nanmedian(np.diff(merge_time))

	TS_size = [-4.149230769230769056e+01, 4.416923076923076508e+01]
	TS_r = TS_size[0] + np.linspace(0, 1, 65) * (TS_size[1] - TS_size[0])
	TS_dr = np.median(np.diff(TS_r)) / 1000
	gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
	profile_centres = []
	profile_centres_score = []
	for index in range(np.shape(merge_ne_prof_multipulse)[0]):
		yy = merge_ne_prof_multipulse[index]
		if np.sum(yy>0)<3:
			continue
		yy_sigma = merge_dne_multipulse[index]
		# print(yy_sigma)
		if np.sum(np.isfinite(yy_sigma))>0:
			yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
			# print(yy_sigma)
			yy_sigma[yy_sigma==0]=np.nanmax([1e-6,*yy_sigma[yy_sigma!=0]])
			# print(yy_sigma)
		else:
			yy_sigma = np.ones_like(yy)*np.nanmin([np.nanmax(yy),1])
		p0 = [np.max(yy), 10, 0]
		bds = [[0, 0, np.min(TS_r)], [np.inf, TS_size[1], np.max(TS_r)]]
		fit = curve_fit(gauss, TS_r, yy, p0, sigma=yy_sigma, maxfev=100000, bounds=bds)
		profile_centres.append(fit[0][-1])
		profile_centres_score.append(fit[1][-1, -1])
	# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)
	profile_centres = np.array(profile_centres)
	profile_centres_score = np.array(profile_centres_score)
	# centre = np.nanmean(profile_centres[profile_centres_score < 1])
	centre = np.nansum(profile_centres/(profile_centres_score**1))/np.sum(1/profile_centres_score**1)
	TS_r_new = (TS_r - centre) / 1000
	print('TS profile centre at %.3gmm compared to the theoretical centre' %centre)
	# temp_r, temp_t = np.meshgrid(TS_r_new, merge_time)
	# plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse,vmin=0);plt.colorbar().set_label('Te [eV]');plt.pause(0.01)
	# plt.figure();plt.pcolor(temp_t,temp_r,merge_ne_prof_multipulse,vmin=0);plt.colorbar().set_label('ne [10^20 #/m^3]');plt.pause(0.01)

	# This is the mean of Te and ne weighted in their own uncertainties.
	temp1 = np.zeros_like(inverted_profiles[:, 0])
	temp2 = np.zeros_like(inverted_profiles[:, 0])
	temp3 = np.zeros_like(inverted_profiles[:, 0])
	temp4 = np.zeros_like(inverted_profiles[:, 0])
	interp_range_t = max(dt, TS_dt) * 1
	interp_range_r = max(dx, TS_dr) * 1
	# weights_r = (np.zeros_like(merge_Te_prof_multipulse) + TS_r_new)/interp_range_r
	weights_t = (((np.zeros_like(merge_Te_prof_multipulse)).T + merge_time).T)/interp_range_t
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
			weights = 1/(weights_t[selected_values]-value_t)**2
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
				temp2_temp = 1/(np.sum(1 / merge_dTe_multipulse[selected_values]))*(np.sum( ((temp1[i_t, i_r]-merge_Te_prof_multipulse[selected_values])/merge_dTe_multipulse[selected_values])**2 )**0.5)
				temp4_temp = 1/(np.sum(1 / merge_dne_multipulse[selected_values]))*(np.sum( ((temp3[i_t, i_r]-merge_ne_prof_multipulse[selected_values])/merge_dne_multipulse[selected_values])**2 )**0.5)
				temp2[i_t, i_r] = max(temp2_temp,(np.max(merge_Te_prof_multipulse[selected_values])-np.min(merge_Te_prof_multipulse[selected_values]))/2 )
				temp4[i_t, i_r] = max(temp4_temp,(np.max(merge_ne_prof_multipulse[selected_values])-np.min(merge_ne_prof_multipulse[selected_values]))/2 )

	merge_Te_prof_multipulse_interp = np.array(temp1)
	merge_dTe_prof_multipulse_interp = np.array(temp2)
	merge_ne_prof_multipulse_interp = np.array(temp3)
	merge_dne_prof_multipulse_interp = np.array(temp4)
	temp_r, temp_t = np.meshgrid(r, new_timesteps)

	# I crop to the usefull stuff
	start_time = new_timesteps.argmin()
	end_time = new_timesteps.argmax() + 1
	time_crop = new_timesteps[start_time:end_time]
	start_r = r.argmin()
	end_r = r.argmax() + 1
	r_crop = r[start_r:end_r]
	temp_r, temp_t = np.meshgrid(r_crop, time_crop)
	merge_Te_prof_multipulse_interp_crop = merge_Te_prof_multipulse_interp[start_time:end_time,start_r:end_r]
	merge_dTe_prof_multipulse_interp_crop = merge_dTe_prof_multipulse_interp[start_time:end_time, start_r:end_r]
	merge_ne_prof_multipulse_interp_crop = merge_ne_prof_multipulse_interp[start_time:end_time,start_r:end_r]
	merge_dne_prof_multipulse_interp_crop = merge_dne_prof_multipulse_interp[start_time:end_time, start_r:end_r]


# Data from wikipedia
energy_difference = np.array([2.54970, 2.85567, 3.02187, 3.12208, 3.18716, 3.23175, 3.26365, 3.28725, 3.30520])  # eV
# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
statistical_weigth = np.array([32,50,72,98,128,162,200,242,288])	#gi-gk
einstein_coeff = np.array([8.4193e-2,2.53044e-2,9.7320e-3,4.3889e-3,2.2148e-3,1.2156e-3,7.1225e-4,4.3972e-4,2.8337e-4])*1e8	#1/s
J_to_eV = 6.242e18
energy_n_2 = 10.19884# eV
energy_levels = energy_difference + energy_n_2
# Used formula 2.3 in Rion Barrois thesys, 2017
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise']

sample_time_step = []
for time in [-4,-2, 0, 0.1,0.15, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.5, 2, 2.5,3, 3.5, 4, 4.5, 5, 6, 7, 8]:	#times in ms that I want to take a better look at
	sample_time_step.append(int((np.abs(new_timesteps - time)).argmin()))
sample_time_step = np.unique(sample_time_step)
sample_radious=[]
for radious in [0, 0.001, 0.002, 0.004, 0.006, 0.009, 0.012,0.015,0.02]:	#radious in m that I want to take a better look at
	sample_radious.append(int((np.abs(r - radious)).argmin()))
sample_radious = np.unique(sample_radious)
try:
	for time_step in sample_time_step:
		# time_step=19
		max_value = 0
		plt.figure(figsize=(20, 10))
		to_plot_all = []
		for index, loc in enumerate(r[sample_radious]):
			to_plot = np.divide(np.pi * 4 * np.mean(inverted_profiles[time_step-1:time_step+2, :, sample_radious[index]],axis=0),statistical_weigth * einstein_coeff * energy_difference / J_to_eV)
			if np.sum(to_plot > 0) > 4:
				to_plot_all.append(to_plot > 0)
		# to_plot_all.append(to_plot)
		to_plot_all = np.array(to_plot_all)
		relative_index = int(np.max((np.sum(to_plot_all, axis=(0)) == np.max(np.sum(to_plot_all, axis=(0)))) * np.linspace(1, len(to_plot),len(to_plot)))) - 1
		for index, loc in enumerate(r[sample_radious]):
			to_plot = np.divide(np.pi * 4 * np.mean(inverted_profiles[max(time_step-1,0):time_step+2, :, sample_radious[index]],axis=0),statistical_weigth * einstein_coeff * energy_difference / J_to_eV)
			plt.plot(energy_levels, to_plot / np.min(to_plot[relative_index]),color[index],label='axial pos=' + str(np.around(1000*loc, decimals=1)))
			max_value = max(np.max(to_plot / np.min(to_plot[relative_index])),max_value)
			fit = np.polyfit(energy_difference[4:],np.log(to_plot[4:]),1)
			plt.plot(energy_levels,np.exp(np.polyval(fit,energy_difference)),':',color=color[index],label='est. temp=%.3geV' %(-1/fit[0]))
			if os.path.exists(path_where_to_save_everything+'/TS_data_merge_'+str(merge_ID_target)+'.npz'):
				to_plot = np.exp(np.polyval(fit,energy_difference[-3])) * np.exp(-(energy_difference-energy_difference[-3])/merge_Te_prof_multipulse_interp_crop[time_step,sample_radious[index]])
				plt.plot(energy_levels, to_plot / np.min(to_plot[relative_index]),'--',color=color[index],label='axial pos=' + str(np.around(1000*loc, decimals=1))+'mm, Te=%.3g+/-%.3g' %(merge_Te_prof_multipulse_interp_crop[time_step,sample_radious[index]],merge_dTe_prof_multipulse_interp_crop[time_step,sample_radious[index]]) + 'eV, ne='+ str(np.around(merge_ne_prof_multipulse_interp_crop[time_step,sample_radious[index]], decimals=2))+'*10^20#/m^3')
		plt.legend(loc='best',fontsize='x-small')
		plt.title('Boltzmann plot at ' + str(np.around(new_timesteps[time_step], decimals=2)) + '+/-'+str(conventional_time_step)+' ms')
		plt.semilogy()
		# plt.semilogx()
		plt.xlabel('state energy [eV]')
		plt.ylabel('relative population density/statistical weight scaled to the min value [au]')
		plt.ylim(top=max_value*1.1)
		plt.savefig(path_where_to_save_everything+'/Bplot_relative_time_' + str(np.around(new_timesteps[time_step], decimals=2)) + '.eps',bbox_inches='tight')
		plt.close()

		plt.figure(figsize=(20, 10))
		to_plot_all = []
		max_value = 0
		for index, loc in enumerate(r[sample_radious]):
			to_plot = np.divide(np.pi * 4 * np.mean(inverted_profiles[max(time_step-1,0):time_step+2, :, sample_radious[index]],axis=0),statistical_weigth * einstein_coeff * energy_difference / J_to_eV)
			plt.plot(energy_levels, to_plot,color=color[index], label='axial pos=' + str(np.around(1000*loc, decimals=1)))
			max_value = max(np.max(to_plot),max_value)
			fit = np.polyfit(energy_difference[4:],np.log(to_plot[4:]),1)
			plt.plot(energy_levels,np.exp(np.polyval(fit,energy_difference)),':',color=color[index],label='est. temp=%.3geV' %(-1/fit[0]))
			if os.path.exists(path_where_to_save_everything+'/TS_data_merge_'+str(merge_ID_target)+'.npz'):
				to_plot = np.exp(np.polyval(fit,energy_difference[-3])) * np.exp(-(energy_difference-energy_difference[-3])/merge_Te_prof_multipulse_interp_crop[time_step,sample_radious[index]])
				plt.plot(energy_levels, to_plot,'--',color=color[index],label='axial pos=' + str(np.around(1000*loc, decimals=1))+'mm, Te=%.3g+/-%.3g' %(merge_Te_prof_multipulse_interp_crop[time_step,sample_radious[index]],merge_dTe_prof_multipulse_interp_crop[time_step,sample_radious[index]]) + 'eV, ne='+ str(np.around(merge_ne_prof_multipulse_interp_crop[time_step,sample_radious[index]], decimals=2))+'*10^20#/m^3')

		plt.legend(loc='best',fontsize='x-small')
		plt.title('Boltzmann plot at ' + str(np.around(new_timesteps[time_step], decimals=2)) + '+/-'+str(conventional_time_step)+' ms')
		plt.semilogy()
		# plt.semilogx()
		plt.xlabel('state energy [eV]')
		plt.ylabel('population density/statistical weight [#/m^3]')
		plt.ylim(top=max_value*1.1)
		plt.savefig(path_where_to_save_everything+'/Bplot_absolute_time_' + str(np.around(new_timesteps[time_step], decimals=2)) + '.eps',bbox_inches='tight')
		plt.close()

		plt.figure(figsize=(20, 10))
		for iR in range(np.shape(inverted_profiles)[1]):
			plt.plot(r * 1000, np.divide(np.pi * 4 * np.mean(inverted_profiles[max(time_step-1,0):time_step+2, iR],axis=0), einstein_coeff[iR] * energy_difference[iR] / J_to_eV),color[iR], label='n=' + str(iR + 4))
		plt.legend(loc='best')
		plt.title('Excited states density plot at ' + str(np.around(new_timesteps[time_step], decimals=2)) + '+/-'+str(conventional_time_step)+' ms')
		# plt.semilogy()
		# plt.semilogx()
		plt.ylim(bottom=0)
		plt.xlabel('radial location [mm]')
		plt.ylabel('excited state density [m^-3]')
		plt.savefig(path_where_to_save_everything + '/Excited_dens_time_' + str(np.around(new_timesteps[time_step], decimals=2)) + '.eps',bbox_inches='tight')
		plt.close()

except:
	print('failed merge' + str(merge_ID_target))
	plt.close()

for index, loc in enumerate(sample_radious):
	plt.figure(figsize=(20, 10))
	# reference_time = inverted_profiles[:, 0, loc].argmax()
	reference_time = (np.max(inverted_profiles[:, :, loc],axis=1)).argmax()
	for iR in range(np.shape(inverted_profiles)[1]):
		plt.plot(new_timesteps, inverted_profiles[:, iR, loc] / inverted_profiles[reference_time, iR, loc], color[iR],linewidth=0.3, label='line ' + str(iR + 4))
	plt.title('Relative line intensity at radious ' + str(np.around(1000*r[loc], decimals=1)) + 'mm')
	# plt.semilogy()
	plt.xlabel('time [ms]')
	plt.ylabel('relative intensity scaled to the max value [au]')
	plt.legend(loc='best')
	plt.ylim(0,1)
	plt.savefig(path_where_to_save_everything+'/rel_intens_r_' + str(np.around(1000*r[loc], decimals=1)) + '.eps', bbox_inches='tight')
	plt.close()

# for index, loc in enumerate([26, 28, 30, 32, 34, 36, 40, 45]):
# 	plt.figure(figsize=(20, 10))
# 	reference_point = inverted_profiles[:, 0, loc].argmax()
# 	max_ratio = np.max(inverted_profiles[reference_point, :-2, loc] / inverted_profiles[reference_point, 1:-1, loc])
# 	print(max_ratio)
# 	for iR in range(np.shape(inverted_profiles)[1] - 1):
# 		# plt.plot(new_timesteps, inverted_profiles[:, iR, loc] / inverted_profiles[:, iR + 1, loc], color[iR]+'+',label='line ratio ' + str(iR + 4) + '/' + str(iR + 4 + 1))
# 		plt.plot(new_timesteps, inverted_profiles[:, iR, loc] / inverted_profiles[:, iR + 1, loc],color=color[iR], marker='+', linewidth=0.4,label='line ratio ' + str(iR + 4) + '/' + str(iR + 4 + 1))
# 	plt.title('Line ratio at radious' + str(np.around(r[loc], decimals=3)) + 'mm')
# 	# plt.semilogy()
# 	plt.xlabel('time [ms]')
# 	plt.ylabel('relative intensity [au]')
# 	plt.legend(loc='best')
# 	plt.ylim(0, np.max([max_ratio,4]))
# 	plt.savefig(
# 		'/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/line_ratio_intens_r_' + str(
# 			np.around(r[loc], decimals=3)) + '.eps', bbox_inches='tight')
# 	plt.close()

for index, loc in enumerate(sample_radious):
	plt.figure(figsize=(20, 10))
	reference_time = (np.max(inverted_profiles[:, :, loc],axis=1)).argmax()
	max_ratio = min(10,np.max(inverted_profiles[reference_time, :-2, loc] / inverted_profiles[reference_time, 1:-1, loc]))
	print(max_ratio)
	for iR in range(np.shape(inverted_profiles)[1] - 1):
		# plt.plot(new_timesteps, inverted_profiles[:, iR, loc] / inverted_profiles[:, iR + 1, loc], color[iR]+'+',label='line ratio ' + str(iR + 4) + '/' + str(iR + 4 + 1))
		# plt.plot(new_timesteps, inverted_profiles[:, iR + 1, loc] / inverted_profiles[:, iR, loc], color=color[iR],marker='+', linewidth=0.4, label='line ratio ' + str(iR + 4 + 1) + '/' + str(iR + 4))
		plt.plot(new_timesteps, inverted_profiles[:, iR + 1, loc] / inverted_profiles[:, iR, loc], color=color[iR],marker='+', label='line ratio ' + str(iR + 4 + 1) + '/' + str(iR + 4))
	plt.title('Line ratio at radious ' + str(np.around(1000*r[loc], decimals=1)) + 'mm')
	# plt.semilogy()
	plt.xlabel('time [ms]')
	plt.ylabel('relative intensity [au]')
	plt.legend(loc='best')
	plt.ylim(0,max_ratio)
	plt.savefig(path_where_to_save_everything+'/line_ratio_intens_r_' + str(np.around(1000*r[loc], decimals=1)) + '.eps',bbox_inches='tight')
	plt.close()




for iR in range(np.shape(inverted_profiles)[1] - 1):
	plt.figure(figsize=(20, 10))
	# plt.plot(new_timesteps, inverted_profiles[:, iR, loc] / inverted_profiles[:, iR + 1, loc], color[iR]+'+',label='line ratio ' + str(iR + 4) + '/' + str(iR + 4 + 1))
	plt.imshow(inverted_profiles[:, iR + 1] / inverted_profiles[:, iR], 'rainbow',vmin=0,vmax=1,extent=[min(r)*1000,max(r)*1000,np.max(new_timesteps),np.min(new_timesteps)],aspect=20)
	plt.title('Line ratio ' +str(iR + 4 + 1) +'/' + str(iR + 4))
	plt.xlabel('radial location [mm]')
	plt.ylabel('time [ms]')
	plt.colorbar().set_label('abel inverted line intensity ratio [au]')
	plt.savefig(path_where_to_save_everything+'/inverted_line_ratio_map_' +str(iR + 4 + 1) +'-' + str(iR + 4) + '.eps',bbox_inches='tight')
	plt.close()

for iR in range(np.shape(inverted_profiles)[1] - 1):
	plt.figure(figsize=(20, 10))
	# plt.plot(new_timesteps, inverted_profiles[:, iR, loc] / inverted_profiles[:, iR + 1, loc], color[iR]+'+',label='line ratio ' + str(iR + 4) + '/' + str(iR + 4 + 1))
	plt.imshow(all_fits[:,:, iR + 1] / all_fits[:,:, iR], 'rainbow',vmin=0,vmax=1,extent=[min(xx)*1000,max(xx)*1000,np.max(new_timesteps),np.min(new_timesteps)],aspect=20)
	plt.title('Line ratio ' +str(iR + 4 + 1) +'/' + str(iR + 4))
	plt.xlabel('LOS location [mm]')
	plt.ylabel('time [ms]')
	plt.colorbar().set_label('LOS integrated line intensity ratio [au]')
	plt.savefig(path_where_to_save_everything+'/line_ratio_map_' +str(iR + 4 + 1) +'-' + str(iR + 4) + '.eps',bbox_inches='tight')
	plt.close()

for iR in range(np.shape(inverted_profiles)[1] - 1):
	plt.figure(figsize=(20, 10))
	# plt.plot(new_timesteps, inverted_profiles[:, iR, loc] / inverted_profiles[:, iR + 1, loc], color[iR]+'+',label='line ratio ' + str(iR + 4) + '/' + str(iR + 4 + 1))
	plt.imshow((inverted_profiles[:, iR + 1] / inverted_profiles[:, iR])*(einstein_coeff[iR] * energy_difference[iR])/( einstein_coeff[iR+1] * energy_difference[iR+1]), 'rainbow',vmin=0,vmax=np.nanmin([1,np.nanmax((inverted_profiles[:, iR + 1] / inverted_profiles[:, iR])*(einstein_coeff[iR] * energy_difference[iR])/( einstein_coeff[iR+1] * energy_difference[iR+1]))]),extent=[min(r)*1000,max(r)*1000,np.max(new_timesteps),np.min(new_timesteps)],aspect=20)
	plt.title('Excited population ratio ' +str(iR + 4 + 1) +'/' + str(iR + 4))
	plt.xlabel('radial location [mm]')
	plt.ylabel('time [ms]')
	plt.colorbar().set_label('excited state density ratio [au]')
	plt.savefig(path_where_to_save_everything+'/Excited_dens_ratio_map_' +str(iR + 4 + 1) +'-' + str(iR + 4) + '.eps',bbox_inches='tight')
	plt.close()


for iR in range(np.shape(inverted_profiles)[1]):
	plt.figure(figsize=(20, 10))
	# plt.plot(new_timesteps, inverted_profiles[:, iR, loc] / inverted_profiles[:, iR + 1, loc], color[iR]+'+',label='line ratio ' + str(iR + 4) + '/' + str(iR + 4 + 1))
	plt.imshow(np.pi * 4 *inverted_profiles[:, iR]/( einstein_coeff[iR] * energy_difference[iR] / J_to_eV), 'rainbow',vmin=0,extent=[min(r)*1000,max(r)*1000,np.max(new_timesteps),np.min(new_timesteps)],aspect=20)
	plt.title('Excited state ' +str(iR + 4) +' population')
	plt.xlabel('radial location [mm]')
	plt.ylabel('time [ms]')
	plt.colorbar().set_label('excited state density [m^-3]')
	plt.savefig(path_where_to_save_everything+'/Excited_dens_map_' +str(iR + 4)+ '.eps',bbox_inches='tight')
	plt.close()


# time_step = int((np.abs(new_timesteps+1)).argmin())	#I select the time 1 ms before the pulse
# all_fits_ss = all_fits[:time_step]
# all_fits_ss[all_fits_ss==0]=np.nan
# all_fits_ss = np.nanmean(all_fits_ss,axis=0)
# doLateralfit_single(df_settings, all_fits_ss, merge_ID_target,dx,xx,r,same_centre_every_line=True)

SS_inverted_profiles = np.load(path_where_to_save_everything+'/SS_inverted_profiles.npy')
SS_inverted_profiles_sigma = np.load(path_where_to_save_everything+'/SS_inverted_profiles.npy')

number_of_radial_divisions = np.shape(SS_inverted_profiles)[-1]
r = np.arange(number_of_radial_divisions)*dx
dr = np.median(np.diff(r))

sample_radious=[]
for radious in [0, 0.001, 0.002, 0.004, 0.006, 0.009, 0.012,0.015,0.02]:	#radious in m that I want to take a better look at
	sample_radious.append(int((np.abs(r - radious)).argmin()))
sample_radious = np.unique(sample_radious)

if os.path.exists(path_where_to_save_everything+'/TS_SS_data_merge_'+str(merge_ID_target)+'.npz'):
	merge_Te_prof_multipulse_SS = np.load(path_where_to_save_everything + '/TS_SS_data_merge_' + str(merge_ID_target) + '.npz')['merge_Te_prof_multipulse']
	merge_dTe_multipulse_SS = np.load(path_where_to_save_everything + '/TS_SS_data_merge_' + str(merge_ID_target) + '.npz')['merge_dTe_multipulse']
	merge_ne_prof_multipulse_SS = np.load(path_where_to_save_everything + '/TS_SS_data_merge_' + str(merge_ID_target) + '.npz')['merge_ne_prof_multipulse']
	merge_dne_multipulse_SS = np.load(path_where_to_save_everything + '/TS_SS_data_merge_' + str(merge_ID_target) + '.npz')['merge_dne_multipulse']

	TS_size = [-4.149230769230769056e+01, 4.416923076923076508e+01]
	TS_r = TS_size[0] + np.linspace(0, 1, 65) * (TS_size[1] - TS_size[0])
	TS_dr = np.median(np.diff(TS_r)) / 1000
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
	profile_centres=[fit[0][-1]]
	profile_sigma=[fit[0][-2]]
	profile_centres_score=[fit[1][-1, -1]]
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
	profile_centres = np.append(profile_centres,[fit[0][-1]],axis=0)
	profile_sigma = np.append(profile_sigma,[fit[0][-2]],axis=0)
	profile_centres_score = np.append(profile_centres_score,[fit[1][-1, -1]],axis=0)
	centre = np.nanmean(profile_centres)
	TS_r_new = (TS_r - centre) / 1000
	print('TS profile centre at %.3gmm compared to the theoretical centre' %centre)

	# This is the mean of Te and ne weighted in their own uncertainties.
	interp_range_r = max(dx, TS_dr) * 1.5
	# weights_r = TS_r_new/interp_range_r
	weights_r = TS_r_new
	merge_Te_prof_multipulse_SS_interp = np.zeros_like(SS_inverted_profiles[ 0])
	merge_dTe_prof_multipulse_SS_interp = np.zeros_like(SS_inverted_profiles[ 0])
	merge_ne_prof_multipulse_SS_interp = np.zeros_like(SS_inverted_profiles[ 0])
	merge_dne_prof_multipulse_SS_interp = np.zeros_like(SS_inverted_profiles[ 0])
	for i_r, value_r in enumerate(np.abs(r)):
		if np.sum(np.abs(np.abs(TS_r_new) - value_r) < interp_range_r) == 0:
			continue
		selected_values = np.abs(np.abs(TS_r_new) - value_r) < interp_range_r
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
	fit = curve_fit(gauss, r_crop, yy, p0, maxfev=100000, bounds=bds)#, sigma=yy_sigma)
	averaged_profile_sigma=fit[0][-2]
	# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)

	# x_local = xx - spatial_factor * 17.4 / 1000
	dr_crop = np.median(np.diff(r_crop))

	merge_dTe_prof_multipulse_SS_interp_crop_limited = cp.deepcopy(merge_dTe_prof_multipulse_SS_interp_crop)
	merge_dTe_prof_multipulse_SS_interp_crop_limited[merge_Te_prof_multipulse_SS_interp_crop < 0.1] = 0
	merge_Te_prof_multipulse_SS_interp_crop_limited = cp.deepcopy(merge_Te_prof_multipulse_SS_interp_crop)
	merge_Te_prof_multipulse_SS_interp_crop_limited[merge_Te_prof_multipulse_SS_interp_crop < 0.1] = 0
	merge_dne_prof_multipulse_SS_interp_crop_limited = cp.deepcopy(merge_dne_prof_multipulse_SS_interp_crop)
	merge_dne_prof_multipulse_SS_interp_crop_limited[merge_ne_prof_multipulse_SS_interp_crop < 5e-07] = 0
	merge_ne_prof_multipulse_SS_interp_crop_limited = cp.deepcopy(merge_ne_prof_multipulse_SS_interp_crop)
	merge_ne_prof_multipulse_SS_interp_crop_limited[merge_ne_prof_multipulse_SS_interp_crop < 5e-07] = 0

	fig = plt.figure(figsize=(20, 10))
	fig.add_subplot(1, 2, 1)
	plt.errorbar(TS_r/1000, merge_Te_prof_multipulse_SS,yerr=merge_dTe_multipulse_SS)
	plt.errorbar([profile_centres[1]/1000,profile_centres[1]/1000],np.sort(merge_Te_prof_multipulse_SS)[[0,-1]],xerr=[profile_centres_score[1]/1000,profile_centres_score[1]/1000],color='b',label='found centre')
	plt.plot(np.ones((2))*(profile_centres[1]+profile_sigma[1]*2.355/2)/1000,np.sort(merge_Te_prof_multipulse_SS)[[0,-1]],'--',color='grey',label='FWHM')
	plt.plot(np.ones((2))*(profile_centres[1]-profile_sigma[1]*2.355/2)/1000,np.sort(merge_Te_prof_multipulse_SS)[[0,-1]],'--',color='grey')
	plt.plot([centre/1000,centre/1000],np.sort(merge_Te_prof_multipulse_SS)[[0,-1]],'k--',label='centre')
	plt.errorbar(r_crop+centre/1000,merge_Te_prof_multipulse_SS_interp_crop_limited,yerr=merge_dTe_prof_multipulse_SS_interp_crop_limited,color='r',label='averaged')
	plt.plot(np.ones((2))*(averaged_profile_sigma*2.355/2+centre/1000),np.sort(merge_Te_prof_multipulse_SS)[[0,-1]],'r--',label='density FWHM')
	plt.legend(loc='best',fontsize='x-small')
	plt.title('SS Te for merge ' + str(merge_ID_target))
	plt.xlabel('radial location [mm]')
	plt.ylabel('temperature [eV]')
	fig.add_subplot(1, 2, 2)
	plt.errorbar(TS_r/1000, merge_ne_prof_multipulse_SS,yerr=merge_dne_multipulse_SS)
	plt.errorbar([profile_centres[0]/1000,profile_centres[0]/1000],np.sort(merge_ne_prof_multipulse_SS)[[0,-1]],xerr=[profile_centres_score[0]/1000,profile_centres_score[0]/1000],color='b',label='found centre')
	plt.plot(np.ones((2))*(profile_centres[0]+profile_sigma[0]*2.355/2)/1000,np.sort(merge_ne_prof_multipulse_SS)[[0,-1]],'--',color='grey',label='FWHM')
	plt.plot(np.ones((2))*(profile_centres[0]-profile_sigma[0]*2.355/2)/1000,np.sort(merge_ne_prof_multipulse_SS)[[0,-1]],'--',color='grey')
	plt.plot([centre/1000,centre/1000],np.sort(merge_ne_prof_multipulse_SS)[[0,-1]],'k--',label='centre')
	plt.errorbar(r_crop+centre/1000,merge_ne_prof_multipulse_SS_interp_crop_limited,yerr=merge_dne_prof_multipulse_SS_interp_crop_limited,color='r',label='averaged')
	plt.plot(np.ones((2))*(averaged_profile_sigma*2.355/2+centre/1000),np.sort(merge_ne_prof_multipulse_SS)[[0,-1]],'r--',label='density FWHM')
	plt.legend(loc='best',fontsize='x-small')
	plt.title('SS ne for merge ' + str(merge_ID_target))
	plt.xlabel('radial location [mm]')
	plt.ylabel('electron density [10^20 #/m^3]')
	plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/SS_Te_ne_interpolated_merge' + str(merge_ID_target) + '.eps', bbox_inches='tight')
	plt.close()


plt.figure(figsize=(20, 10))
for iR in range(np.shape(SS_inverted_profiles)[0]):
	if np.max(SS_inverted_profiles_sigma[iR])>10*np.max(SS_inverted_profiles[iR]):
		plt.plot(r * 1000, np.divide(np.pi * 4 * SS_inverted_profiles[iR], einstein_coeff[iR] * energy_difference[iR] / J_to_eV),color=color[iR], label='n=' + str(iR + 4))
	else:
		plt.errorbar(r * 1000, np.divide(np.pi * 4 * SS_inverted_profiles[iR], einstein_coeff[iR] * energy_difference[iR] / J_to_eV),yerr =np.divide(np.pi * 4 * SS_inverted_profiles_sigma[iR], einstein_coeff[iR] * energy_difference[iR] / J_to_eV) ,color=color[iR], label='n=' + str(iR + 4))
plt.legend(loc='best')
plt.title('Excited states density plot in steady state')
# plt.semilogy()
# plt.semilogx()
plt.xlabel('radial location [mm]')
plt.ylabel('excited state density [m^-3]')
plt.savefig(path_where_to_save_everything + '/SS_Excited_dens.eps',bbox_inches='tight')
plt.close()

max_value = 0
plt.figure(figsize=(20, 10))
to_plot_all = []
for index, loc in enumerate(r[sample_radious]):
	to_plot = np.divide(np.pi * 4 * SS_inverted_profiles[ :, sample_radious[index]],statistical_weigth * einstein_coeff * energy_difference / J_to_eV)
	if np.sum(to_plot > 0) > 4:
		to_plot_all.append(to_plot > 0)
# to_plot_all.append(to_plot)
to_plot_all = np.array(to_plot_all)
relative_index = int(np.max((np.sum(to_plot_all, axis=(0)) == np.max(np.sum(to_plot_all, axis=(0)))) * np.linspace(1, len(to_plot),len(to_plot)))) - 1
for index, loc in enumerate(r[sample_radious]):
	to_plot = np.divide(np.pi * 4 * SS_inverted_profiles[ :, sample_radious[index]],statistical_weigth * einstein_coeff * energy_difference / J_to_eV)
	plt.plot(energy_levels, to_plot / np.min(to_plot[relative_index]),color[index],label='axial pos=' + str(np.around(1000*loc, decimals=1)))
	max_value = max(np.max(to_plot / np.min(to_plot[relative_index])),max_value)
	fit = np.polyfit(energy_difference[4:],np.log(to_plot[4:]),1)
	plt.plot(energy_levels[3:],np.exp(np.polyval(fit,energy_difference[3:])),':',color=color[index],label='est. temp=%.3geV' %(-1/fit[0]))
	if os.path.exists(path_where_to_save_everything+'/TS_SS_data_merge_'+str(merge_ID_target)+'.npz'):
		to_plot = np.exp(np.polyval(fit,energy_difference[-3])) * np.exp(-(energy_difference-energy_difference[-3])/merge_Te_prof_multipulse_SS_interp_crop_limited[sample_radious[index]])
		to_plot_sigma = np.exp(np.polyval(fit,energy_difference[-3])) * np.exp(-(energy_difference-energy_difference[-3])/merge_dTe_prof_multipulse_SS_interp_crop_limited[sample_radious[index]])
		plt.errorbar(energy_levels, to_plot / np.min(to_plot[relative_index]),yerr=to_plot_sigma,linestyle='--',color=color[index],label='axial pos=' + str(np.around(1000*loc, decimals=1))+'mm, Te=%.3g+/-%.3g' %(merge_Te_prof_multipulse_SS_interp_crop_limited[sample_radious[index]],merge_dTe_prof_multipulse_SS_interp_crop_limited[sample_radious[index]]) + 'eV, ne='+ str(np.around(merge_ne_prof_multipulse_SS_interp_crop_limited[sample_radious[index]], decimals=2))+'*10^20#/m^3')
plt.legend(loc='best',fontsize='x-small')
plt.title('Steady state Boltzmann plot')
plt.semilogy()
# plt.semilogx()
plt.xlabel('state energy [eV]')
plt.ylabel('relative population density/statistical weight scaled to the min value [au]')
plt.ylim(top=max_value*1.1)
plt.savefig(path_where_to_save_everything+'/SS_Bplot_relative.eps',bbox_inches='tight')
plt.close()

plt.figure(figsize=(20, 10))
to_plot_all = []
max_value = 0
for index, loc in enumerate(r[sample_radious]):
	to_plot = np.divide(np.pi * 4 * SS_inverted_profiles[:, sample_radious[index]],statistical_weigth * einstein_coeff * energy_difference / J_to_eV)
	plt.plot(energy_levels, to_plot,color=color[index], label='axial pos=' + str(np.around(1000*loc, decimals=1)))
	max_value = max(np.max(to_plot),max_value)
	fit = np.polyfit(energy_difference[4:],np.log(to_plot[4:]),1)
	plt.plot(energy_levels,np.exp(np.polyval(fit,energy_difference)),':',color=color[index],label='est. temp=%.3geV' %(-1/fit[0]))
	if os.path.exists(path_where_to_save_everything+'/TS_SS_data_merge_'+str(merge_ID_target)+'.npz'):
		to_plot_2 = np.exp(np.polyval(fit,energy_difference[-3])) * np.exp(-(energy_difference-energy_difference[-3])/merge_Te_prof_multipulse_SS_interp_crop_limited[sample_radious[index]])
		to_plot_sigma = np.exp(np.polyval(fit,energy_difference[-3])) * np.exp(-(energy_difference-energy_difference[-3])/merge_dTe_prof_multipulse_SS_interp_crop_limited[sample_radious[index]])
		plt.plot(energy_levels, to_plot_2,'--',color=color[index],label='axial pos=' + str(np.around(1000*loc, decimals=1))+'mm, Te=%.3g+/-%.3g' %(merge_Te_prof_multipulse_SS_interp_crop_limited[sample_radious[index]],merge_dTe_prof_multipulse_SS_interp_crop_limited[sample_radious[index]]) + 'eV, ne='+ str(np.around(merge_ne_prof_multipulse_SS_interp_crop_limited[sample_radious[index]], decimals=2))+'*10^20#/m^3')
plt.legend(loc='best',fontsize='x-small')
plt.title('Steady state Boltzmann plot')
plt.semilogy()
# plt.semilogx()
plt.xlabel('state energy [eV]')
plt.ylabel('population density/statistical weight [#/m^3]')
plt.ylim(top=max_value*1.1)
plt.savefig(path_where_to_save_everything+'/SS_Bplot_absolute.eps',bbox_inches='tight')
plt.close()


last_merge_done = merge_ID_target

# for i in range(17,32):
# 	all_fits = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(i) + '_all_fits.npy')
# 	doLateralfit_time_tependent(df_settings, all_fits, i)
# #
#
# inverted_profiles = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(17)+'/inverted_profiles.npy')
#
# # doLateralfit_time_tependent(df_settings,all_fits, merge_ID_target)
#
# energy_difference = [1.89,2.55,2.86,3.03,3.13,3.19,3.23,3.26, 3.29 ]	#eV
# statistical_weigth = [18,32,50,72,98,128,162,200,242]
#
# plt.figure();plt.plot(energy_difference,np.divide(inverted_profiles[20,:,17],statistical_weigth)); plt.semilogy();plt.pause(0.01)




# number_pulses=80
# minimum_pulse = 0.0002
#
# type = '.tsf'
# filenames = all_file_names('/home/ffederic/work/Collaboratory/test/experimental_data/2019-05-01', type)
#
#
# all_peaks_missing=[]
# all_peaks_good=[]
# all_peaks_double=[]
# for fname in filenames[9:]:
# 	current_traces = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/2019-05-01/' + fname, index_col=False, delimiter='\t')
# 	current_traces_time = current_traces['Time [s]']
# 	current_traces_total = current_traces['I_Src_AC [A]']
# 	time_resolution = np.mean(np.diff(current_traces_time))
# 	peaks = \
# 	find_peaks(current_traces_total, width=int(minimum_pulse / time_resolution), height=current_traces_total.max() / 3)[0]
# 	difference = np.diff(np.sort(peaks))
# 	temp = []
# 	skip = 0
# 	for index, value in enumerate(peaks):
# 		if skip == 1:
# 			skip = 0
# 			continue
# 		if index == len(peaks) - 1:
# 			temp.append(value)
# 			continue
# 		if (difference[index] < np.mean(difference) * 0.5):
# 			if current_traces_total[value] < current_traces_total[peaks[index + 1]]:
# 				continue
# 			else:
# 				skip = 1
# 		temp.append(value)
#
# 	peaks = np.array(temp)
# 	difference = np.diff(np.sort(peaks))
#
# 	peaks = np.array([peaks for _, peaks in sorted(zip(current_traces_total[peaks], peaks))])
#
# 	if (current_traces_total[peaks]).max() < 1.25 * np.mean(current_traces_total[peaks]):
# 		peaks_good = np.sort(peaks)
# 		peaks_double = np.array([])
# 	else:
# 		peaks_good = np.sort(peaks[:np.diff(current_traces_total[peaks]).argmax() + 1])
# 		peaks_double = np.sort(peaks[np.diff(current_traces_total[peaks]).argmax() + 1:])
#
# 	peaks = np.sort(peaks)
# 	peaks_missing = []
# 	for index, value in enumerate(peaks):
# 		if index == len(peaks) - 1:
# 			continue
# 		if difference[index] > np.mean(difference) * 1.25:
# 			peaks_missing.append(int((peaks[index + 1] + peaks[index]) / 2))
# 	peaks_missing = np.array(peaks_missing)
#
# 	all_peaks = np.sort(peaks_good.tolist() + peaks_double.tolist() + peaks_missing.tolist())
# 	if len(peaks_double) == 0:
# 		while current_traces_time[all_peaks[0]] % 1 > 0.213:
# 			peaks_missing = np.concatenate(([int(min(all_peaks) - np.mean(np.diff(all_peaks)))], peaks_missing))
# 			all_peaks = np.concatenate(([int(min(all_peaks) - np.mean(np.diff(all_peaks)))], all_peaks))
# 	else:
# 		while (min(all_peaks) == min(peaks_double) or current_traces_time[all_peaks[0]] % 1 > 0.213):
# 			peaks_missing = np.concatenate(([int(min(all_peaks) - np.mean(np.diff(all_peaks)))], peaks_missing))
# 			all_peaks = np.concatenate(([int(min(all_peaks) - np.mean(np.diff(all_peaks)))], all_peaks))
#
# 	# peaks_missing = np.concatenate(([int(min(all_peaks) - np.mean(np.diff(all_peaks)))], peaks_missing))
# 	# all_peaks = np.concatenate(([int(min(all_peaks)-np.mean(np.diff(all_peaks)))],all_peaks))
#
# 	while len(all_peaks) < number_pulses:
# 		peaks_missing = np.concatenate((peaks_missing, [int(max(all_peaks) + np.mean(np.diff(all_peaks)))]))
# 		all_peaks = np.concatenate((all_peaks, [int(max(all_peaks) + np.mean(np.diff(all_peaks)))]))
# 	while len(all_peaks) > number_pulses:
# 		if all_peaks[-1] in peaks_good:
# 			peaks_good = peaks_good[:-1]
# 		all_peaks = all_peaks[:-1]
#
# 	double = []
# 	good = []
# 	missing = []
# 	for index, value in enumerate(all_peaks):
# 		if (value in peaks_double):
# 			double.append(index + 1)
# 		elif (value in peaks_good):
# 			good.append(index + 1)
# 		elif (value in peaks_missing):
# 			missing.append(index + 1)
# 	double = np.array(double)
# 	good = np.array(good)
# 	missing = np.array(missing)
#
#
# 	all_peaks_missing.append(missing)
# 	all_peaks_good.append(good)
# 	all_peaks_double.append(double)
#
#
# base_peaks_double=all_peaks_missing[1]
# shift=base_peaks_double
# for peaks_double in all_peaks_double:
# 	addition=0
# 	peaks_double = np.array(peaks_double)
# 	check=0
# 	while check==0:
# 		peaks = peaks_double + addition
# 		check=1
# 		for peak in peaks:
# 			if peak >= max(base_peaks_double):
# 				continue
# 			if not (peak in base_peaks_double):
# 				check=0
# 		addition+=1
# 	if addition-1<50:
# 		shift.extend(peaks)
# 	else:
# 		shift.append(np.nan)
#
# search_for = [5,7,16] # this is the common and regular number pulses in between of missing pulses





'''



fdir = '/home/ffederic/work/Collaboratory/test/experimental_data/2019-05-01/03/'
fname = '2019-05-01 17h 18m 40s TT_06686078285962697360.tsf'
fname = '2019-05-01 17h 21m 30s TT_06686079013835380998.tsf'
fname = '2019-05-01 16h 30m 20s TT_06686065742217175720.tsf'
fname = '2019-05-01 17h 44m 57s TT_06686085060543396175.tsf'
fname = '2019-05-01 18h 00m 55s TT_06686089175148122573.tsf'
fname = '2019-05-01 18h 17m 21s TT_06686093409282579396.tsf'

current_traces = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/2019-05-01/03/'+fname,index_col=False,delimiter='\t')
current_traces_time = current_traces['Time [s]']
current_traces_total = current_traces['I_Src_AC [A]']
from scipy.signal import find_peaks, peak_prominences as get_proms
time_resolution = np.mean(np.diff(current_traces_time))
minimum_pulse = 0.0002	#s
peaks = find_peaks(current_traces_total,width=int(minimum_pulse/time_resolution),height=current_traces_total.max()/3)[0]
difference = np.diff(np.sort(peaks))
temp=[]
skip=0
for index,value in enumerate(peaks):
	if skip==1:
		skip=0
		continue
	if index==len(peaks)-1:
		temp.append(value)
		continue
	if (difference[index]<np.mean(difference)*0.5):
		if current_traces_total[value]<current_traces_total[peaks[index+1]]:
			continue
		else:
			skip=1
	temp.append(value)

peaks = np.array(temp)
difference = np.diff(np.sort(peaks))
proms = get_proms(current_traces_total,peaks)[0]

first_pulse = peaks[0]
peaks=np.array([peaks for _,peaks in sorted(zip(current_traces_total[peaks],peaks))])

number_pulses = len(peaks)
if (current_traces_total[peaks]).max()<1.25*np.mean(current_traces_total[peaks]):
	peaks_good = np.sort(peaks)
	peaks_double = np.array([])
else:
	peaks_good = np.sort(peaks[:np.diff(current_traces_total[peaks]).argmax()+1])
	peaks_double = np.sort(peaks[np.diff(current_traces_total[peaks]).argmax()+1:])

# TS_laser_frequency = 10	#Hz
peaks = np.sort(peaks)
peaks_missing = []
for index,value in enumerate(peaks):
	if index==len(peaks)-1:
		continue
	if difference[index]>np.mean(difference)*1.25:
		peaks_missing.append(int((peaks[index+1]+peaks[index])/2))
peaks_missing = np.array(peaks_missing)

all_peaks = np.sort(peaks_good.tolist() + peaks_double.tolist() + peaks_missing.tolist())
if len(peaks_double)==0:
	if current_traces_time[first_pulse]%1>0.213:
		peaks_missing = np.concatenate(([int(min(all_peaks) - np.mean(np.diff(all_peaks)))], peaks_missing))
		all_peaks = np.concatenate(([int(min(all_peaks) - np.mean(np.diff(all_peaks)))], all_peaks))
elif (min(peaks)==min(peaks_double) or current_traces_time[first_pulse]%1>0.213):
	peaks_missing = np.concatenate(([int(min(all_peaks) - np.mean(np.diff(all_peaks)))], peaks_missing))
	all_peaks = np.concatenate(([int(min(all_peaks)-np.mean(np.diff(all_peaks)))],all_peaks))

# peaks_missing = np.concatenate(([int(min(all_peaks) - np.mean(np.diff(all_peaks)))], peaks_missing))
# all_peaks = np.concatenate(([int(min(all_peaks)-np.mean(np.diff(all_peaks)))],all_peaks))


while len(all_peaks)<80:
	peaks_missing = np.concatenate((peaks_missing,[int(max(all_peaks)+np.mean(np.diff(all_peaks)))]))
	all_peaks = np.concatenate((all_peaks,[int(max(all_peaks)+np.mean(np.diff(all_peaks)))]))

bad = []
for index,value in enumerate(all_peaks):
	if not ( value in peaks_good ):
		bad.append(index+1)
bad = np.array(bad)





# first_pulse_all=[]
# type = '.tsf'
# filenames = all_file_names('/home/ffederic/work/Collaboratory/test/experimental_data/2019-05-01/03', type)
# for fname in filenames:
# 	current_traces = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/2019-05-01/03/' + fname,
# 								 index_col=False, delimiter='\t')
# 	current_traces_time = current_traces['Time [s]']
# 	current_traces_total = current_traces['I_Src_AC [A]']
# 	# plt.plot(current_traces_time,current_traces_total,label=fname);plt.legend(loc='best');plt.pause(0.001)
# 	time_resolution = np.mean(np.diff(current_traces_time))
# 	minimum_pulse = 0.0003  # s
# 	peaks = find_peaks(current_traces_total, width=int(minimum_pulse / time_resolution), height=current_traces_total.max() / 3)[0]
# 	proms = get_proms(current_traces_total, peaks)[0]
#
# 	first_pulse_all.append(current_traces_time[peaks[0]])
#
#
#
#
# plt.plot(current_traces_time,current_traces_total);plt.pause(0.001)






bof=[]
eof=[]
metadata= open(os.path.join(fdir+'/'+folder+'/'+"{0:0=2d}".format(int(sequence))+'/Untitled_'+str(int(untitled))+'/Pos0', filename_metadata), 'r')
flag=0
roi_lb=[0,0]
roi_tr=[0,0]
for row in metadata:
	#print(row)
	#print('gna')
	if row.find('PVCAM-FMD-TimestampBofNs') !=-1:
		#print('found bof')
		start=0
		end=len(row)
		for index,value in enumerate(row):
			if (value.isdigit() and start==0):
				start=int(index)
			elif (not value.isdigit() and start!=0 and end==len(row)):
				end=int(index)
		#print(row[start:end])
		bof.append(int(row[start:end]))
#plt.figure()
#plt.imshow(data[0], origin='lower')
#plt.colorbar()
#plt.plot([0,data.shape[1]],[208+30*10]*2)
#plt.plot([392,958,1192],[200,193,196])
#plt.plot([392,958,1192],[1362,1366,1368])
#plt.plot([392,958,1192],[809,809,809])
#plt.pause(0.001)
	if row.find('PVCAM-FMD-TimestampEofNs') !=-1:
		#print('found eof')
		start=0
		end=len(row)
		for index,value in enumerate(row):
			if (value.isdigit() and start==0):
				start=int(index)
			elif (not value.isdigit() and start!=0 and end==len(row)):
				end=int(index)
		#print(row[start:end])
		eof.append(int(row[start:end]))
	if row.find('PVCAM-FMD-ExposureTimeNs') !=-1:
		#print('found eof')
		start=0
		end=len(row)
		for index,value in enumerate(row):
			if (value.isdigit() and start==0):
				start=int(index)
			elif (not value.isdigit() and start!=0 and end==len(row)):
				end=int(index)
		#print(row[start:end])
		real_exposure_time=int(row[start:end])*0.000001

	if row.find('"ROI": [') !=-1:
		flag=1
	elif flag==1:
		start=0
		end=len(row)
		for index,value in enumerate(row):
			if (value.isdigit() and start==0):
				start=int(index)
			elif (not value.isdigit() and start!=0 and end==len(row)):
				end=int(index)
		roi_lb[0]=int(row[start:end])
		flag=2
	elif flag==2:
		start=0
		end=len(row)
		for index,value in enumerate(row):
			if (value.isdigit() and start==0):
				start=int(index)
			elif (not value.isdigit() and start!=0 and end==len(row)):
				end=int(index)
		roi_lb[1]=int(row[start:end])
		flag=3
	elif flag==3:
		start=0
		end=len(row)
		for index,value in enumerate(row):
			if (value.isdigit() and start==0):
				start=int(index)
			elif (not value.isdigit() and start!=0 and end==len(row)):
				end=int(index)
		roi_tr[0]=int(row[start:end])
		flag=4
	elif flag==4:
		start=0
		end=len(row)
		for index,value in enumerate(row):
			if (value.isdigit() and start==0):
				start=int(index)
			elif (not value.isdigit() and start!=0 and end==len(row)):
				end=int(index)
		roi_tr[1]=int(row[start:end])
		flag=0
bof=np.array(bof)
eof=np.array(eof)
metadata.close()





plt.figure()
plt.imshow(data[0], origin='lower')
plt.colorbar()
plt.plot([0,data.shape[1]],[208+30*10]*2)
plt.plot([392,958,1192],[200,193,196])
plt.plot([392,958,1192],[1362,1366,1368])
plt.plot([392,958,1192],[809,809,809])
plt.pause(0.001)

plt.figure()
plt.plot(np.mean(data[0,:,1192-17:1192+17],axis=-1),label='right')
1368-196
plt.plot(np.mean(data[0,:,958-17:958+17],axis=-1),label='centre')
1366-193
plt.plot(np.mean(data[0,:,392-17:392+17],axis=-1),label='left')
1362-200
plt.semilogy()
plt.legend(loc='best')
plt.pause(0.001)

plt.figure()
plt.plot(np.mean(data[0,:,1403-17:1403+17],axis=-1))
plt.plot(np.mean(data[0,:,892-17:892+17],axis=-1))
plt.plot(np.mean(data[0,:,668-17:668+17],axis=-1))
plt.semilogy()
plt.pause(0.001)


bin00 = 196
binInterv  = (1387-bin00)/40
binnedData = functions.binData(data[0],bin00,binInterv)
plt.figure()
plt.imshow(binnedData, origin='lower')
plt.colorbar()
plt.pause(0.001)

# let's say that for now the image deformation before binning is fine

number_wavelength = roi_tr[0]
number_rows = roi_tr[1]
if ((eof[0]-bof[0])/number_rows!=read_out_time):
	read_out_time = (eof[0]-bof[0])/number_rows
	print('Replacing readout time based on timestamps to '+str(np.around(read_out_time,decimals=2))+'ns')


# times = np.linspace(-conventional_read_out_time*extra_shifts_pulse_before,conventional_read_out_time*(number_rows+extra_shifts_pulse_after-1),number_rows+extra_shifts_pulse_before+extra_shifts_pulse_after)
time_start = -conventional_read_out_time*extra_shifts_pulse_before
num_time_steps = int(extra_shifts_pulse_before/row_skip+np.ceil((number_rows+extra_shifts_pulse_after)/row_skip))
time_end = conventional_read_out_time*(np.ceil((number_rows+extra_shifts_pulse_after)/row_skip)*row_skip-1)
times = np.linspace(time_start,time_end,num_time_steps)
rows = np.linspace(0,number_rows-1,number_rows)
wavelengths = np.linspace(0,number_wavelength-1,number_wavelength)

da = xr.DataArray(-1*np.ones((number_rows+extra_shifts_pulse_before+extra_shifts_pulse_after,number_rows)),[('time', times),('space', rows),('wavelength', wavelengths)])
for i in range(np.shape(data)[0]):
	for j in range(np.shape(data)[1]):
		da[i,j]

'''










print('JOB DONE!!')
