import numpy as np
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())
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
	binnedSens = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_'+str(type_of_sensitivity)+'/Sensitivity_4'+mod_convolution+'.npy')
	binnedSens_sigma = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_'+str(type_of_sensitivity)+'/Sensitivity_4'+mod_convolution+'_sigma.npy')
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
	binnedSens = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_'+str(type_of_sensitivity)+'/Sensitivity_4'+mod_convolution+'.npy')
	binnedSens_sigma = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_'+str(type_of_sensitivity)+'/Sensitivity_4'+mod_convolution+'_sigma.npy')
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
path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)
if (((not time_resolution_scan and not os.path.exists(path_where_to_save_everything + '/merge'+str(merge_ID_target)+'_merge_tot.npz')) or (time_resolution_scan and not os.path.exists(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.npz'))) or overwrite_everything[0]):
# if True:
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
	intermediate_wavelength,last_wavelength = get_line_position(data_sum,2)
	tilt_intermediate_column = get_bin_and_interv_specific_wavelength(fix_minimum_signal2(data_sum),intermediate_wavelength)
	tilt_last_column = get_bin_and_interv_specific_wavelength(fix_minimum_signal2(data_sum),last_wavelength)

	# second loop in which I actually build up the merge of all images
	data_sum=0
	dataDark_all = []
	for j in all_j:
		dataDark = load_dark(j,df_settings,df_log,fdir,geom_null)
		dataDark_all.append(dataDark)
		(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
		type = '.tif'
		filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2g}".format(int(sequence))+'/Untitled_'+str(int(untitled))+'/Pos0', type)
		#filenames = functions.all_file_names(pathfiles, type)
		type = '.txt'
		filename_metadata = all_file_names(fdir+'/'+folder+'/'+"{0:0=2g}".format(int(sequence))+'/Untitled_'+str(int(untitled))+'/Pos0', type)[0]
		#filename_metadata = functions.all_file_names(pathfiles, type)[0]
		(bof,eof,roi_lb,roi_tr,elapsed_time,real_exposure_time,PixelType,Gain,Noise) = get_metadata(fdir,folder,sequence,untitled,filename_metadata)
		if PixelType[-1]==12:
			time_range_for_interp = rows_range_for_interp * row_shift/2
		elif PixelType[-1]==16:
			time_range_for_interp = rows_range_for_interp * row_shift
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

		time_of_pulses_present = 0
		if not os.path.exists(path_where_to_save_everything):
			os.makedirs(path_where_to_save_everything)
		if isinstance(df_log.loc[j,['current_trace_file']][0],str):
			fname_current_trace,SS_current = df_log.loc[j,['current_trace_file','I']]
			trash1, trash2, trash3, trash4, trash5, trash6, good_pulses, time_of_pulses = examine_current_trace(fdir + '/' + folder + '/' + "{0:0=2g}".format(int(sequence)) + '/', fname_current_trace,number_of_pulses,SS_current=SS_current)
			time_of_pulses = np.sort(time_of_pulses['peak_of_the_peak']) - np.min(time_of_pulses['peak_of_the_peak'])
			current_traces = pd.read_csv(fdir+'/'+folder+'/'+"{0:0=2g}".format(int(sequence))+'/' + fname_current_trace+'.tsf',index_col=False, delimiter='\t')
			current_traces_time = current_traces['Time [s]']
			current_traces_total = current_traces['I_Src_AC [A]']
			plt.figure(figsize=(20, 10))
			plt.plot(current_traces_time,current_traces_total)
			plt.grid()
			plt.title('Current trace for '+str(j))
			plt.xlabel('time [s]')
			plt.ylabel('Rogowski coil current [A]')
			plt.savefig(path_where_to_save_everything + '/file_index_' + str(j) + 'time_profile1.eps', bbox_inches='tight')
			plt.close()

			# temp = bof[int(first_pulse_at_this_frame-1):int(number_of_pulses+first_pulse_at_this_frame-1)]/1000000000
			# temp -= temp[0]
			# time_of_pulses -= temp

			# if ((first_pulse_at_this_frame>=0) and (first_pulse_at_this_frame + number_of_pulses <= len(bof))):
			time_of_beginning_of_frame = (bof[first_pulse_at_this_frame-1:first_pulse_at_this_frame+number_of_pulses-1]-bof[first_pulse_at_this_frame-1])/1e9

			if len(time_of_beginning_of_frame) < len(time_of_pulses):
				if first_pulse_at_this_frame < 1:
					time_of_pulses = time_of_pulses[-(first_pulse_at_this_frame-1):-(first_pulse_at_this_frame-1)+len(time_of_beginning_of_frame)+1]
				else:
					time_of_pulses = time_of_pulses[:len(time_of_beginning_of_frame)]
				good_pulses = good_pulses[np.logical_and(good_pulses>=-(first_pulse_at_this_frame-2),good_pulses<=len(time_of_pulses))]
			print('time_of_pulses')
			print(time_of_pulses)
			print('time_of_beginning_of_frame')
			print(time_of_beginning_of_frame)
			time_shift_betwee_pulses = (time_of_pulses - time_of_beginning_of_frame )*1000	# ms

			plt.figure(figsize=(20, 10))
			plt.plot(good_pulses-1,(time_of_pulses)[good_pulses-1],'c+')
			plt.plot((time_of_pulses),'c',label='time when the pulse starts (current), mean increment =%.7gs'%((time_of_pulses)[-1]/(len(time_of_pulses)-1)))
			plt.plot(good_pulses-1, (time_of_beginning_of_frame)[good_pulses-1],'rx')
			plt.plot( (time_of_beginning_of_frame),'r',label='time when record starts (OES camera), mean increment =%.7gs'%((time_of_beginning_of_frame)[-1]/(len(time_of_pulses)-1)))
			plt.grid()
			plt.title('Time of the pulses in '+str(j))
			plt.xlabel('pulse index')
			plt.ylabel('time shift [s]')
			plt.legend(loc='best')
			plt.savefig(path_where_to_save_everything + '/file_index_' + str(j) + 'time_profile2.eps', bbox_inches='tight')
			plt.close()

			plt.figure(figsize=(20, 10))
			plt.plot(good_pulses-1,1000*(time_of_pulses-np.linspace(0,(len(time_of_pulses)-1),(len(time_of_pulses)))*0.1)[good_pulses-1],'c+')
			plt.plot(1000*(time_of_pulses-np.linspace(0,(len(time_of_pulses)-1),(len(time_of_pulses)))*0.1),'c',label='time when the pulse starts (current), mean defferential increment =%.5gms' %(1000*(time_of_pulses-np.linspace(0,(len(time_of_pulses)-1),(len(time_of_pulses)))*0.1)[-1]/(len(time_of_pulses)-1)))
			plt.plot(good_pulses-1, 1000*(time_of_beginning_of_frame -np.linspace(0,(len(time_of_pulses)-1),(len(time_of_pulses)))*0.1)[good_pulses-1],'rx')
			plt.plot( 1000*(time_of_beginning_of_frame -np.linspace(0,(len(time_of_pulses)-1),(len(time_of_pulses)))*0.1),'r',label='time when record starts (OES camera), mean defferential increment =%.5gms' %(1000*(time_of_beginning_of_frame -np.linspace(0,(len(time_of_pulses)-1),(len(time_of_pulses)))*0.1)[-1]/(len(time_of_pulses)-1)))
			plt.grid()
			plt.title('Time difference respect a theoretical 10Hz pulse rate in '+str(j))
			plt.xlabel('pulse index')
			plt.ylabel('time shift [ms]')
			plt.legend(loc='best')
			plt.savefig(path_where_to_save_everything + '/file_index_' + str(j) + 'time_profile3.eps', bbox_inches='tight')
			plt.close()

			plt.figure(figsize=(20, 10))
			plt.plot(time_shift_betwee_pulses)
			plt.plot(good_pulses-1,time_shift_betwee_pulses[good_pulses-1],'+')
			plt.grid()
			plt.title('Time shift between camera and pulse current in '+str(j)+'\n mean inctement ='+str(np.around( time_shift_betwee_pulses[-1]/(len(time_of_pulses-time_of_beginning_of_frame)-1), decimals=5))+'ms instead of requested %.3gms' %(incremental_step))
			plt.xlabel('pulse index')
			plt.ylabel('time shift [ms]')
			# plt.legend(loc='best')
			plt.savefig(path_where_to_save_everything + '/file_index_' + str(j) + 'time_profile4.eps', bbox_inches='tight')
			plt.close()

			plt.figure(figsize=(20, 10))
			plt.plot(good_pulses-1,(time_shift_betwee_pulses - np.linspace(0,(len(time_of_pulses)-1),(len(time_of_pulses)))* incremental_step )[good_pulses-1])
			plt.plot(good_pulses-1,(time_shift_betwee_pulses - np.linspace(0,(len(time_of_pulses)-1),(len(time_of_pulses)))* incremental_step )[good_pulses-1],'+')
			plt.grid()
			plt.title('Difference between requested and measured increments in '+str(j))
			plt.xlabel('pulse index')
			plt.ylabel('time shift [ms]')
			# plt.legend(loc='best')
			plt.savefig(path_where_to_save_everything + '/file_index_' + str(j) + 'time_profile5.eps', bbox_inches='tight')
			plt.close()

			# time_of_pulses -= np.linspace(0,(len(time_of_pulses)-1),(len(time_of_pulses)))*0.1#np.mean(np.diff(bof)/1000000000)	#	I cannot use bof or eof because are too much imprecise
			# time_of_pulses = time_of_pulses*1000
			# time_of_pulses_present = 1
			# plt.figure(figsize=(20, 10))
			# plt.plot(good_pulses,time_of_pulses[good_pulses-1])
			# plt.plot(good_pulses, time_of_pulses[good_pulses-1],'o',label='original data')
			# fit = np.polyfit(good_pulses, time_of_pulses[good_pulses-1], 1)
			# plt.plot(good_pulses,np.polyval(fit, good_pulses),'--',label='fit parameters = '+str(fit))
			# fit_correction = incremental_step - fit[0]
			# time_of_pulses = time_of_pulses + np.linspace(0,(len(time_of_pulses)-1),(len(time_of_pulses)))*fit_correction
			# plt.plot(good_pulses, time_of_pulses[good_pulses-1],'o',label='corrected data')
			# fit = np.polyfit(good_pulses, time_of_pulses[good_pulses-1], 1)
			# plt.plot(good_pulses,np.polyval(fit, good_pulses),'--',label='fit parameters = '+str(fit))
			# plt.legend(loc='best')
			# plt.title('Time interval between good pulses for '+str(j))
			# plt.xlabel('pulse index')
			# plt.ylabel('time shift from first pulse [ms]')
			# # plt.pause(0.001)
			# plt.savefig(path_where_to_save_everything + '/file_index_' + str(j) + 'time_profile.eps', bbox_inches='tight')
			# plt.close()
			# print('time interval from camera '+str(np.mean(np.diff(bof)/1000000000))+'ms')
			# print('time profile for j='+str(j)+' found')
			# print('fitting coefficients found '+str(fit))
			# print('time interval of '+str(fit[0]*1000000)+' microseconds')

		data_all=[]
		for index,filename in enumerate(filenames):
			if (index<first_pulse_at_this_frame-1 or index>=(first_pulse_at_this_frame+number_of_pulses-1)):
				data_all.append(np.zeros_like(dataDark))
				continue
			elif (index-first_pulse_at_this_frame+2) in bad_pulses_indexes:
				data_all.append(np.zeros_like(dataDark))
				continue
			if time_resolution_scan:
				if index%(1+time_resolution_extra_skip)!=0:
					continue	# The purpose of this is to test how things change for different time skippings
			print(filename)
			fname = fdir+'/'+folder+'/'+"{0:0=2g}".format(int(sequence))+'/Untitled_'+str(int(untitled))+'/Pos0/'+filename
			im = Image.open(fname)
			data = np.array(im)

			data_sigma=np.sqrt(data)
			additive_factor,additive_factor_sigma = fix_minimum_signal_experiment(data,intermediate_wavelength,last_wavelength,tilt_intermediate_column,tilt_last_column,counts_treshold_fixed_increase=106,dx_to_etrapolate_to=dx_to_etrapolate_to)
			data = (data.T + additive_factor).T
			data_sigma = np.sqrt((data_sigma**2).T + (additive_factor_sigma**2)).T


			# data =data - dataDark	# I checked that for 16 or 12 bit the dark is always around 100 counts
			# data = data*Gain[index]
			# data = fix_minimum_signal3(data)
			# data_sum+=data
			data_all.append(data)
			if time_of_pulses_present==1:
				time_first_row = -CB_to_OES_initial_delay-time_shift_betwee_pulses[index-int(first_pulse_at_this_frame)+1]
			else:
				time_first_row = -CB_to_OES_initial_delay-(index-first_pulse_at_this_frame+1)*incremental_step
			if PixelType[index]==12:
				timesteps = np.linspace(time_first_row, time_first_row + (roi_tr[1] - 1+2) * row_shift, roi_tr[1]+2)

				timesteps = np.sort(timesteps.tolist()+timesteps.tolist())[:roi_tr[1]]
			elif PixelType[index] == 16:
				# timesteps = np.linspace(time_first_row/conventional_time_step, (time_first_row+(roi_tr[1]-1)*row_shift)/conventional_time_step,roi_tr[1])  # timesteps are normalised with the conventional time step in a way to easy the later interpolation
				timesteps = np.linspace(time_first_row,time_first_row + (roi_tr[1] - 1) * row_shift, roi_tr[1])

			# timesteps +=0.19	# ms, this comes from the observation that the increase in light is always before 0ms, and also in TS there is this time correction
			# UPDATE 13/09/2019 this seems to actually be not necessary

			if started == 0:
				row_steps = np.linspace(0, roi_tr[1] - 1, roi_tr[1])
				wavelengths = np.linspace(0, roi_tr[0] - 1, roi_tr[0])

			#array_to_add = np.ones((roi_tr[1],roi_tr[1],roi_tr[0]))*np.nan
			# started1 = 0
			# for i in range(roi_tr[1]):
			# 	data_to_record=data[i]
			# 	# for index2, value in enumerate(data_to_record):
			# 	# 	if value>=4095:	# THIS IS TO AVOID OVEREXPOSURE
			# 	# 		data_to_record[index2]=np.nan		# TO BE FINISHED!!!!!!!!!!!!!
			#
			# 	if ( timesteps[i]<merge_time_window[0] or timesteps[i]>merge_time_window[1] ):
			# 		continue
			# 	# array_to_add[i,i]=data[:,i]
			# 	# print(timesteps[i])
			overexposed=np.max(data,axis=1)>overexposed_treshold/Gain[index]
				# if np.max(data_to_record)>overexposed_treshold*Gain[index]:	#	it should be 4096 but comparing spectra with different gain you see it goes bad around this value
				# 	overexposed=1
				# # if started1==0:
				# # 	merge=xr.DataArray(data_to_record, dims=[ 'wavelength_axis'],coords={'time_row_to_start_pulse': timesteps[i],'row': row_steps[i],'wavelength_axis': wavelengths,'overexposed':overexposed, 'Gain':Gain[index]},name='first entry')
				# 	started1+=1
				# else:
				# 	addition_anticipated = False
			if len(merge_values)==0:
				merge_values=np.array(data)
				merge_values_sigma=np.array(data_sigma)
				merge_time = np.array(timesteps)
				merge_row = np.array(row_steps)
				merge_Gain = np.ones((len(data)))*Gain[index]
				merge_Noise = np.ones((len(data)))*Noise[index]
				merge_overexposed = np.array(overexposed)
				merge_wavelengths = np.ones((len(data),1))*wavelengths
				started=1

					# if np.shape(merge_tot.Gain.values)==():
					# 	merge = xr.concat([merge_tot, xr.DataArray(data_to_record, dims=['wavelength_axis'],
					# 										   coords={'time_row_to_start_pulse': timesteps[i],
					# 												   'row': row_steps[i],
					# 												   'wavelength_axis': wavelengths,
					# 												   'overexposed': overexposed,
					# 												   'Gain': Gain[index]},
					# 										   name='addition' + str([j, index, i]))],
					# 					  dim='time_row_to_start_pulse')
					# 	addition_anticipated = True
					# 	merge = xr.concat([merge,xr.DataArray(data_to_record, dims=[ 'wavelength_axis'],coords={'time_row_to_start_pulse': timesteps[i],'row': row_steps[i],'wavelength_axis': wavelengths,'overexposed':overexposed, 'Gain':Gain[index]},name='addition'+str([j,index,i]))], dim='time_row_to_start_pulse')
				# else:
				# 	normal=0
			else:
				for i in range(len(data)):
					data_to_record=data[i]
					data_to_record_sigma=data_sigma[i]

					same_time = merge_time == timesteps[i]
					if np.sum(same_time)>0:
						# sample = merge_tot[same_time]
						same_row = merge_row == row_steps[i]
						if np.sum(np.logical_and(same_time,same_row)) > 0:
							same_Gain = merge_Gain < Gain[index]
							if np.sum(np.logical_and(np.logical_and(same_time, same_row),same_Gain)) > 0:
								same_overexposed = merge_overexposed == 1
								rows_to_fix = np.logical_and(np.logical_and(np.logical_and(same_time, same_row),same_Gain),same_overexposed)
								if np.sum(rows_to_fix) > 0:
									for z,to_be_fixed in enumerate(rows_to_fix):
										if to_be_fixed==True:
											print('fixing time '+str(timesteps[i])+', row '+str(row_steps[i])+'/row num'+str(z))
											sample =  merge_values[z]
											sample_sigma =  merge_values_sigma[z]
											for z1,value in enumerate(sample):
												if value/merge_Gain[z]>overexposed_treshold:
													sample[z1]=data_to_record[z1]
													sample_sigma[z1]=data_to_record_sigma[z1]
													print('fixed at wavelength axis pixel '+str(z1))
											merge_values[z] = sample
											merge_values[z] = sample_sigma
											merge_Gain[z] = Gain[index]
											merge_Noise[z] = Noise[index]
											merge_overexposed[z] = overexposed[i]

				merge_values = np.append(merge_values,data,axis=0)
				merge_values_sigma = np.append(merge_values_sigma,data_sigma,axis=0)
				merge_time = np.append(merge_time,timesteps,axis=0)
				merge_row = np.append(merge_row,row_steps,axis=0)
				merge_Gain = np.append(merge_Gain,np.ones((len(data)))*Gain[index],axis=0)
				merge_Noise = np.append(merge_Noise,np.ones((len(data)))*Noise[index],axis=0)
				merge_overexposed = np.append(merge_overexposed,overexposed,axis=0)
				merge_wavelengths = np.append(merge_wavelengths,np.ones((len(data),1))*wavelengths,axis=0)

					# 			else:
						# 				normal=1
						# 		else:
						# 			normal = 1
						# 	else:
						# 		normal=1
						# else:
						# 	normal = 1
						# if normal==1:
	# 				if not addition_anticipated:
	# 					merge = xr.concat([merge, xr.DataArray(data_to_record, dims=['wavelength_axis'],
	# 																	   coords={'time_row_to_start_pulse': timesteps[i],
	# 																			   'row': row_steps[i],
	# 																			   'wavelength_axis': wavelengths,
	# 																			   'overexposed': overexposed,
	# 																			   'Gain': Gain[index]},
	# 																	   name='addition' + str([j, index, i]))],
	# 												  dim='time_row_to_start_pulse')
	# 		if started == 0:
	# 			merge_tot = merge
	# 			started += 1
	# 		else:
	# 			merge_tot = xr.concat([merge_tot, merge],dim='time_row_to_start_pulse')
	#
		data_sum += np.mean(data_all,axis=0)
		if not os.path.exists(path_where_to_save_everything):
			os.makedirs(path_where_to_save_everything)
		ani = coleval.movie_from_data(np.array([np.array(data_all)]), 1000/incremental_step, row_shift, 'Wavelength axis [pixles]', 'Row axis [pixles]','Intersity [au]',extvmin=90,extvmax=np.min([np.max(np.array(data_all)[:30]),np.max(np.array(data_all)[-70:])]))
		# plt.show()
		ani.save(path_where_to_save_everything + '/merge'+str(merge_ID_target)+'_scan_n'+str(j)+'_original_data' + '.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
		# ani.save(path_where_to_save_everything + '/merge'+str(merge_ID_target)+'_scan_n'+str(j)+'_original_data' + '.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
		plt.close()

	dataDark = np.mean(dataDark_all,axis=0)
	if time_resolution_scan:
		np.savez_compressed(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot',merge_values=merge_values,merge_values_sigma=merge_values_sigma,merge_time=merge_time,merge_row=merge_row,merge_Gain=merge_Gain,merge_overexposed=merge_overexposed)

		# np.save(.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.nc')
	else:
		np.savez_compressed(path_where_to_save_everything+'/merge' + str(merge_ID_target) + '_merge_tot',merge_values=merge_values,merge_values_sigma=merge_values_sigma,merge_time=merge_time,merge_row=merge_row,merge_Gain=merge_Gain,merge_overexposed=merge_overexposed)

		# merge_tot.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_merge_tot.nc')

	path_filename = path_where_to_save_everything+'/merge' + str(merge_ID_target) + '_stats.csv'
	# if not os.path.exists(path_filename):
	file = open(path_filename, 'w')
	writer = csv.writer(file)
	writer.writerow(['Stats about merge ' + str(merge_ID_target)])
		# file.close()

	intermediate_wavelength,last_wavelength = get_line_position(data_sum,2)
	tilt_intermediate_column = get_bin_and_interv_specific_wavelength(fix_minimum_signal2(data_sum),intermediate_wavelength)
	writer.writerow(['LOS start and bin interval at '+str(intermediate_wavelength)+' wavelength axis', str(tilt_intermediate_column)])
	tilt_last_column = get_bin_and_interv_specific_wavelength(fix_minimum_signal2(data_sum),last_wavelength)
	writer.writerow(['LOS start and bin interval at '+str(last_wavelength)+' wavelength axis', str(tilt_last_column)])

	result = 0
	nLines = 8
	while (result == 0 and nLines>0):
		try:
			angle = get_angle_2(data_sum,nLines=nLines)
			print(angle)
			result = 1
			geom['angle'] = np.nansum(np.multiply(angle[0], np.divide(1, np.power(angle[1], 2)))) / np.nansum(np.divide(1, np.power(angle[1], 2)))
			writer.writerow(['Specific angle of ',str(geom['angle'][0]),' found with nLines=',str(nLines)])
		except:
			nLines-=1
	# if np.abs(geom['angle'][0])>2*np.abs(geom_store['angle'][0]):
	if np.abs(geom['angle'][0]-geom_store['angle'][0]) > np.abs(geom_store['angle'][0]):
		geom['angle'][0]=geom_store['angle'][0]
		writer.writerow(['No specific angle found. Used standard of ', str(geom['angle'][0])])
	data_sum = rotate(data_sum, geom['angle'])
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
	# else:
	# 	writer.writerow(
	# 		['Specific tilt of ', str(geom['tilt'][0]), ' found with nLines=', str(nLines), 'binInterv',
	# 		 str(geom['binInterv'][0]), 'first bin', str(geom['bin00a'][0])])
	# writer.writerow(['wavelength coefficients of' ,str(waveLcoefs)])
	# writer.writerow(['the final result will be from' ,str(merge_time_window[0]),' to ',str(merge_time_window[1]),' ms from the beginning of the discharge'])
	# writer.writerow(['with a time resolution of '+str(conventional_time_step),' ms'])
	# file.close()

else:
	all_j=find_index_of_file(merge_ID_target,df_settings,df_log,only_OES=True)
	j = all_j[-1]
	(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
	type = '.txt'
	filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2g}".format(int(sequence)) + '/Untitled_' + str(int(untitled)) + '/Pos0',type)[0]
	(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain,Noise) = get_metadata(fdir,folder,sequence,untitled,filename_metadata)
	if PixelType[-1] == 12:
		time_range_for_interp = rows_range_for_interp * row_shift / 2
	elif PixelType[-1] == 16:
		time_range_for_interp = rows_range_for_interp * row_shift
	(CB_to_OES_initial_delay,incremental_step,first_pulse_at_this_frame,bad_pulses_indexes,number_of_pulses) = df_log.loc[j,['CB_to_OES_initial_delay','incremental_step','first_pulse_at_this_frame','bad_pulses_indexes','number_of_pulses']]
	first_pulse_at_this_frame = int(first_pulse_at_this_frame)
	wavelengths = np.linspace(0, roi_tr[0] - 1, roi_tr[0])
	row_steps = np.linspace(0, roi_tr[1] - 1, roi_tr[1])
	data_sum = 0

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
			fname = fdir+'/'+folder+'/'+"{0:0=2g}".format(int(sequence))+'/Untitled_'+str(int(untitled))+'/Pos0/'+filename
			im = Image.open(fname)
			data = np.array(im)
			data_all.append(data)
		data_sum += np.mean(data_all,axis=0)
	intermediate_wavelength,last_wavelength = get_line_position(data_sum,2)
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
			merge_Noise.extend(np.ones((len(data)))*Noise[index])

		data_sum += np.mean(data_all,axis=0)
		if not os.path.exists(path_where_to_save_everything):
			os.makedirs(path_where_to_save_everything)
		ani = coleval.movie_from_data(np.array([np.array(data_all)]), 1000/incremental_step, row_shift, 'Wavelength axis [pixles]', 'Row axis [pixles]','Intersity [au]',extvmin=90,extvmax=np.min([np.max(np.array(data_all)[:30]),np.max(np.array(data_all)[-70:])]))
		# plt.show()
		# ani.save(path_where_to_save_everything + '/merge'+str(merge_ID_target)+'_scan_n'+str(j)+'_original_data' + '.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
		ani.save(path_where_to_save_everything + '/merge'+str(merge_ID_target)+'_scan_n'+str(j)+'_original_data' + '.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
		plt.close()

		ani = coleval.movie_from_data(np.array([np.array(data_all)[::5]]), 1000/incremental_step*5, row_shift, 'Wavelength axis [pixles]', 'Row axis [pixles]','Intersity [au]',extvmin=90,extvmax=np.min([np.max(np.array(data_all)[:30]),np.max(np.array(data_all)[-70:])]))
		# plt.show()
		# ani.save(path_where_to_save_everything + '/merge'+str(merge_ID_target)+'_scan_n'+str(j)+'_original_data' + '.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
		ani.save(path_where_to_save_everything + '/merge'+str(merge_ID_target)+'_scan_n'+str(j)+'_original_data' + '_2.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
		plt.close()

	dataDark = np.mean(dataDark_all,axis=0)
	path_filename = path_where_to_save_everything+'/merge' + str(merge_ID_target) + '_stats.csv'
	# if not os.path.exists(path_filename):
	file = open(path_filename, 'w')
	writer = csv.writer(file)
	writer.writerow(['Stats about merge ' + str(merge_ID_target)])
	# file.close()

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
	# file.close()

	if time_resolution_scan:
		# merge_tot = xr.open_dataarray('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.nc')
		merge_values = np.load(path_where_to_save_everything+'/merge' + str(merge_ID_target) +'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.npz')['merge_values']
		merge_values_sigma = np.load(path_where_to_save_everything+'/merge' + str(merge_ID_target) +'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.npz')['merge_values_sigma']
		merge_time = np.load(path_where_to_save_everything+'/merge' + str(merge_ID_target) +'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.npz')['merge_time']
		merge_row = np.load(path_where_to_save_everything+'/merge' + str(merge_ID_target) +'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.npz')['merge_row']
		merge_Gain = np.load(path_where_to_save_everything+'/merge' + str(merge_ID_target) +'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.npz')['merge_Gain']
		merge_overexposed = np.load(path_where_to_save_everything+'/merge' + str(merge_ID_target) +'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.npz')['merge_overexposed']
	else:
		merge_values = np.load(path_where_to_save_everything+'/merge' + str(merge_ID_target) + '_merge_tot.npz')['merge_values']
		try:
			merge_values_sigma = np.load(path_where_to_save_everything+'/merge' + str(merge_ID_target) + '_merge_tot.npz')['merge_values_sigma']
		except Exception as e:
			print('WARNING for merge '+str(merge_ID_target)+' merge_values_sigma absent')
			print(e)
			merge_values_sigma = np.ones_like(merge_values)*np.min(merge_values[merge_values>0])
		merge_time = np.load(path_where_to_save_everything+'/merge' + str(merge_ID_target) + '_merge_tot.npz')['merge_time']
		merge_row = np.load(path_where_to_save_everything+'/merge' + str(merge_ID_target) + '_merge_tot.npz')['merge_row']
		merge_Gain = np.load(path_where_to_save_everything+'/merge' + str(merge_ID_target) + '_merge_tot.npz')['merge_Gain']
		merge_overexposed = np.load(path_where_to_save_everything+'/merge' + str(merge_ID_target) + '_merge_tot.npz')['merge_overexposed']
	merge_Noise=np.array(merge_Noise)

		# merge_tot = xr.open_dataarray('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_merge_tot.nc')

	# ani = coleval.movie_from_data(np.array([composed_array.values]), 1000/conventional_time_step, row_shift, 'Wavelength axis [pixles]', 'Row axis [pixles]','Intersity [au]')
	# # plt.show()
	# ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_composed_array' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# I will reconstruct a full scan of 10ns steps from 0 to rows*10ns and all rows

# mask = np.zeros_like(data_sum)
data_sum = four_point_transform(data_sum,geom['tilt'][0])
waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(data_sum)[1]+1))
temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(data_sum)[0]+1))
plt.figure(figsize=(20, 10))
plt.pcolor(temp_w,temp_LOS,data_sum,cmap='rainbow',vmax=np.max(data_sum[:,:800]),rasterized=True)
# plt.imshow(data_sum,'rainbow',origin='lower',vmax=np.max(data_sum[:,:800]))
plt.colorbar()
for i in range(41):
	plt.plot([np.min(waveaxis),np.max(waveaxis)],[geom['bin00b']+i*geom['binInterv'],geom['bin00b']+i*geom['binInterv']],'--k',linewidth=0.5)
	# mask[int(geom['bin00b']+i*geom['binInterv'])-1:int(geom['bin00b']+i*geom['binInterv'])+2]=1
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(data_sum)[0]-1],'--k',linewidth=0.5)
plt.title('example of the binning that will be done \nangle, tilt, binInterv, bin00a, bin00b\n%.5g, ' %(geom['angle']) +str(np.around(geom['tilt'][0],decimals=3))+', %.5g, %.5g, %.5g' %(geom['binInterv'],geom['bin00a'],geom['bin00b']))
plt.xlabel('wavelength [nm]')
plt.ylabel('LOS axis [pixels]')
plt.savefig(path_where_to_save_everything + '/' + 'binning_example.eps', bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 5))
plt.pcolor(temp_w,temp_LOS,data_sum,cmap='rainbow',rasterized=True)
# plt.imshow(data_sum,'rainbow',origin='lower',vmax=np.max(data_sum[:,:800]))
plt.colorbar()
for i in range(41):
	plt.plot([np.min(waveaxis),np.max(waveaxis)],[geom['bin00b']+i*geom['binInterv'],geom['bin00b']+i*geom['binInterv']],'--k',linewidth=0.5)
	# mask[int(geom['bin00b']+i*geom['binInterv'])-1:int(geom['bin00b']+i*geom['binInterv'])+2]=1
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(data_sum)[0]-1],'--k',linewidth=0.5)
plt.title('example of the binning that will be done \nangle, tilt, binInterv, bin00a, bin00b\n%.5g, ' %(geom['angle']) +str(np.around(geom['tilt'][0],decimals=3))+', %.5g, %.5g, %.5g' %(geom['binInterv'],geom['bin00a'],geom['bin00b']))
plt.xlabel('wavelength [nm]')
plt.ylabel('LOS axis [pixels]')
plt.savefig(path_where_to_save_everything + '/' + 'binning_example2.eps', bbox_inches='tight')
plt.close()

new_timesteps = np.linspace(merge_time_window[0]+0.5, merge_time_window[1]-0.5,int((merge_time_window[1]-0.5-(merge_time_window[0]+0.5))/conventional_time_step+1))

if overwrite_everything[1]:
	np.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_new_timesteps', new_timesteps)
	timesteps_cropping_needed = False
else:
	old_timesteps = np.load(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_new_timesteps.npy')
	timesteps_cropping_needed = False
	if len(old_timesteps)!=len(new_timesteps):
		timesteps_cropping_needed = True
		timesteps_limit_down = np.abs(old_timesteps-np.min(new_timesteps)).argmin()
		timesteps_limit_up = np.abs(old_timesteps-np.max(new_timesteps)).argmin() +1


writer.writerow(['the final result will be from', str(np.min(new_timesteps)), ' to ', str(np.max(new_timesteps)),' ms from the beginning of the discharge'])
writer.writerow(['with a time resolution of ' + str(np.median(np.diff(new_timesteps))), ' ms'])
file.close()

mkl.set_num_threads(1)


# composed_array=xr.DataArray(np.zeros((len(new_timesteps),roi_tr[1],roi_tr[0])), dims=['time', 'row', 'wavelength_axis'],coords={'time': new_timesteps,'row': row_steps,'wavelength_axis': wavelengths},name='interpolated array')
# composed_array = merge_tot.interp_like(composed_array,kwargs={'fill_value': 0.0})
# this seems not to work so I do it manually
composed_array = np.zeros((len(new_timesteps),roi_tr[1],np.shape(merge_values)[1]))
time_range_for_interp = min(time_range_for_interp,conventional_time_step)

interpolation_borders_expanded_row = np.zeros((len(new_timesteps),roi_tr[1]))
interpolation_borders_expanded_time = np.zeros((len(new_timesteps),roi_tr[1]))
number_of_points_for_interpolation = np.zeros((len(new_timesteps),roi_tr[1]))
if (((not time_resolution_scan and not os.path.exists(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_composed_array.npy')) or (time_resolution_scan and not os.path.exists(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_composed_array.npy'))) or overwrite_everything[1]):


	# # if merge_ID_target >= 66:
	# # 	grade_of_interpolation = 4
	#
	# # merge_tot = merge_tot.sortby('time_row_to_start_pulse')
	# for i,interpolated_time in enumerate(new_timesteps):
	# 	additional_time_range = 0
	# 	selected = np.abs(merge_time - interpolated_time) < time_range_for_interp + additional_time_range
	# 	if (not time_resolution_scan or time_resolution_scan_improved):
	# 		while (np.sum(selected) < 3*roi_tr[1] and additional_time_range<time_range_for_interp):	# range max doubled
	# 			additional_time_range += time_range_for_interp/10
	# 			selected = np.abs(merge_time - interpolated_time) < time_range_for_interp + additional_time_range
	# 			# sample = merge_tot.sel(time_row_to_start_pulse=slice(interpolated_time - time_range_for_interp-additional_time_range,
	# 			# 													 interpolated_time + time_range_for_interp+additional_time_range))
	# 			print('interpolated_time '+str(interpolated_time)+' increased of '+str(additional_time_range))
	# 			interpolation_borders_expanded[i]+=np.ones_like(interpolation_borders_expanded[i])
	# 	if  np.sum(selected)==0:
	# 		continue
	# 	sample_values_1 = merge_values[selected]
	# 	sample_time_1 = merge_time[selected]
	# 	sample_row_1 = merge_row[selected]
	#
	# 	# sample = sample.sortby('row')
	# 	for j, interpolated_row in enumerate(row_steps):
	# 		# weight = 1 / np.sqrt(0.00001+((sample_time - interpolated_time)/ row_shift ) ** 2 + (sample_row - interpolated_row) ** 2)
	# 		# row_selection = np.abs(sample_row - interpolated_row) < rows_range_for_interp + additional_row_range
	# 		# row_selection = (sample.row>float(interpolated_row-rows_range_for_interp)).astype('int') * (sample.row<float(interpolated_row+rows_range_for_interp)).astype('int')
	# 		additional_row_range = 0
	# 		selected = np.abs(sample_row_1 - interpolated_row) < rows_range_for_interp + additional_row_range
	# 		if (not time_resolution_scan or time_resolution_scan_improved):
	# 			while (np.sum(selected) < 2 and additional_row_range<rows_range_for_interp):	# range max doubled
	# 				additional_row_range +=rows_range_for_interp/10
	# 				selected = np.abs(sample_row_1 - interpolated_row) < rows_range_for_interp + additional_row_range
	# 				# row_selection = (sample.row > float(interpolated_row - rows_range_for_interp-additional_row_range)).astype('int') * (
	# 				# 			sample.row < float(interpolated_row + rows_range_for_interp+additional_row_range)).astype('int')
	# 				# print(sample.row)
	# 				# print(sample)
	# 				print('interpolated_time '+str(interpolated_time)+', interpolated_row '+str(interpolated_row)+' increased of ' + str(additional_row_range))
	# 				print('total number of rows '+str(len(sample_time_1)))
	# 				interpolation_borders_expanded[i,j] += 1
	# 		if np.sum(selected) < 1:
	# 			continue
	# 		# elif np.sum(row_selection) > 10:
	# 		# 	weight_row = row_selection*(weight)
	# 		# else:
	# 		# 	weight_row = weight
	# 		sample_values_2 = sample_values_1[selected]
	# 		sample_time_2 = sample_time_1[selected]
	# 		sample_row_2 = sample_row_1[selected]
	# 		if (np.sum(sample_time_2 - interpolated_time >= 0) == 0 or np.sum(sample_time_2 - interpolated_time <= 0) == 0):
	# 			print('time '+str(interpolated_time)+' row '+str(interpolated_row)+' too asymmetric in time')
	# 			interpolation_borders_expanded[i, j]=0
	# 			continue
	# 		if (np.sum(sample_row_2 - interpolated_row >= 0) == 0 or np.sum(sample_row_2 - interpolated_row <= 0) == 0):
	# 			print('time ' + str(interpolated_time) + ' row ' + str(interpolated_row) + ' too asymmetric in row')
	# 			interpolation_borders_expanded[i, j]=0
	# 			continue
	# 		weight_row = 1 / np.sqrt(0.00001 + (np.abs(sample_time_2 - interpolated_time) / row_shift) ** grade_of_interpolation + np.abs(sample_row_2 - interpolated_row) ** grade_of_interpolation)
	# 		# weight_row = row_selection * weight
	# 		# weight_row_filter=(weight_row.values).tolist()
	# 		# for index,value in enumerate(weight_row_filter):
	# 		# 	if value==np.nan:
	# 		# 		weight_row_filter.pop(index)
	# 		# weight_row=np.array(weight_row)
	# 		# sample_weighted = sample_values_2*weight_row
	# 		# weight_row = weight_row.reshape((len(weight_row), 1))
	# 		weight_row = np.array([weight_row]).T
	# 		sample_weighted = np.nansum(sample_values_2*weight_row,axis=0)/np.nansum(weight_row)
	# 		composed_array[i,j]=sample_weighted



	grade_of_interpolation = 2	#this is the exponent used for the weights

	rows_range_for_interp = 2
	max_row_expansion = 2
	time_range_for_interp = conventional_time_step/2
	max_time_expansion = time_range_for_interp
	min_rows_for_interp = 5
	time_vs_row_weighting = 1


	# def calc_stuff(arg):
	# 	import numpy as np
	# 	z = arg[0]
	# 	i = arg[1]
	# 	j = arg[2]
	# 	grade_of_interpolation = arg[3]
	# 	num_arg_to_split = len(arg[4:]) // 3
	# 	values = arg[4:4 + num_arg_to_split]
	# 	weight_combined = arg[4 + num_arg_to_split:4 + 2 * num_arg_to_split]
	# 	merge_row_selected = arg[4 + 2 * num_arg_to_split:4 + 3 * num_arg_to_split]
	#
	# 	fit_coef_linear = np.polyfit(merge_row_selected, values, grade_of_interpolation, w=weight_combined)
	# 	# print([i, j, z])
	# 	# # # print(fit_coef_linear)
	# 	# print([z,np.polyval(fit_coef_linear, j)])
	# 	return int(z), np.polyval(fit_coef_linear, j)

	if False:	# I noticed that the counts are smooth within the same row so I tried to fit only for different times but it is still very irregular between rows
		for j, interpolated_row in enumerate(row_steps):
			selected_row = merge_row == interpolated_row
			merge_time_selected = merge_time[selected_row]
			values=merge_values[selected_row]
			values = np.array([to_sort for _, to_sort in sorted(zip(merge_time_selected, values))])
			merge_time_selected = np.sort(merge_time_selected)
			for z, interpolated_wave in enumerate(range(1608)):
				interpolator = splrep(merge_time_selected,values.T[z],k=3,s=10*len(merge_time_selected))
				selected_time = np.logical_and(new_timesteps>=np.min(merge_time_selected),new_timesteps<=np.max(merge_time_selected))
				interpolated = splev(new_timesteps[selected_time],interpolator)
				# print('done row '+str(j)+' point '+str(z))
				composed_array[selected_time,j,z] = interpolated
	elif False:	# Here I do a bidimensional spline, but it is still not very homogeneus, afterall it looks only to a limited number of points
		for i, interpolated_time in enumerate(new_timesteps):
			selected_time = np.abs(merge_time - interpolated_time) <=time_range_for_interp*3
			if len(np.unique(merge_time[selected_time]))<4:
				continue
			for j, interpolated_row in enumerate(row_steps):
				selected_row = np.abs(merge_row - interpolated_row) <=rows_range_for_interp*2
				selected = selected_time*selected_row
				if np.sum(selected)<17:
					continue
				for z, interpolated_wave in enumerate(range(1608)):
					values = merge_values[selected,z]
					interpolator = bisplrep(merge_time[selected],merge_row[selected],values,s=1000*np.sum(selected),kx=2,ky=2)
					interpolated = bisplev(interpolated_row,interpolated_time,interpolator)
					composed_array[i,j,z] = interpolated
				print(interpolated_row)
			print(interpolated_time)
	elif False:	# here a 2D polinomial fit fith weights on the distance, and a median filter beforehand

		time_range_for_interp = conventional_time_step*1.3
		max_time_expansion = conventional_time_step
		rows_range_for_interp = 3
		max_row_expansion = 2
		min_points_for_interp = 15

		merge_values_medianfiltered = medfilt(merge_values,[1,3])

		from scipy.optimize import least_squares
		def polynomial_2D_residuals(coord):
			def internal_func(coeff):
				residuals = (np.array(coord[0])**2)*coeff[0] + (np.array(coord[0]))*coeff[1] + (np.array(coord[1])**2)*coeff[2] + (np.array(coord[1]))*coeff[3] + coeff[4] - np.array(coord[2])
				return residuals
			return internal_func
		def polynomial_2D(coord,coeff):
			z = (np.array(coord[0])**2)*coeff[0] + (np.array(coord[0]))*coeff[1] + (np.array(coord[1])**2)*coeff[2] + (np.array(coord[1]))*coeff[3] + coeff[4]
			return z

		from scipy.optimize import least_squares
		def polynomial_2D_residuals_1(coord):
			def internal_func(coeff):
				residuals = (np.array(coord[0])**2)*coeff[0] + (np.array(coord[0]))*coeff[1] + (np.array(coord[1])**2)*coeff[2] + (np.array(coord[1]))*coeff[3] + coeff[4] - np.array(coord[2])
				residuals = residuals * (1/np.abs(np.abs(np.array(coord[0])-coord[3])+0.01))
				return residuals
			return internal_func

		def polynomial_2D_residuals_2(coord):
			def internal_func(coeff):
				residuals = (np.array(coord[0])**3)*coeff[0] + (np.array(coord[0])**2)*coeff[1] + (np.array(coord[0]))*coeff[2] + (np.array(coord[1])**2)*coeff[3] + (np.array(coord[1]))*coeff[4] + coeff[5] - np.array(coord[2])
				residuals = residuals * (1/(np.abs(np.array(coord[0])-coord[3])+0.01)**2)
				return residuals
			return internal_func
		def polynomial_2D_2(coord,coeff):
			z = (np.array(coord[0])**3)*coeff[0] +(np.array(coord[0])**2)*coeff[1] + (np.array(coord[0]))*coeff[2] + (np.array(coord[1])**2)*coeff[3] + (np.array(coord[1]))*coeff[4] + coeff[5]
			return z

		class calc_stuff_output:
			def __init__(self, interpolated_time, composed_array, interpolation_borders_expanded_row,interpolation_borders_expanded_time, number_of_points_for_interpolation):
				self.interpolated_time = interpolated_time
				self.composed_array = composed_array
				self.interpolation_borders_expanded_row = interpolation_borders_expanded_row
				self.interpolation_borders_expanded_time = interpolation_borders_expanded_time
				self.number_of_points_for_interpolation = number_of_points_for_interpolation

		def calc_stuff(arg,row_steps=row_steps,merge_time=merge_time,merge_row=merge_row,merge_values_medianfiltered=merge_values_medianfiltered,time_range_for_interp=time_range_for_interp,rows_range_for_interp=rows_range_for_interp,calc_stuff_output=calc_stuff_output):
			i = arg[0]
			interpolated_time = arg[1]
			print(interpolated_time)
			interpolation_borders_expanded_row = np.zeros_like(row_steps)
			interpolation_borders_expanded_time = np.zeros_like(row_steps)
			number_of_points_for_interpolation = np.zeros_like(row_steps)
			composed_array = np.zeros((len(row_steps),1608))

			# for i, interpolated_time in enumerate(new_timesteps):
			selected_time = np.abs(merge_time - interpolated_time) <=time_range_for_interp
			if len(np.unique(merge_time[selected_time]))<4:
				output = calc_stuff_output(interpolated_time,composed_array,interpolation_borders_expanded_row,interpolation_borders_expanded_time,number_of_points_for_interpolation)
				return output
			for j, interpolated_row in enumerate(row_steps):
				selected_row = np.abs(merge_row - interpolated_row) <=rows_range_for_interp
				selected = selected_time*selected_row
				if np.sum(selected)<min_points_for_interp:
					print('time ' + str(interpolated_time) + ' , row ' + str(j)+' skipped, only '+str(np.sum(selected))+ ' points found')
					continue
				# if j<600 or j>650:
				# 	continue
				merge_time_selected = merge_time[selected]
				merge_row_selected = merge_row[selected]
				# interpolation_borders_expanded[i,j] = np.sum(selected)
				number_of_points_for_interpolation[j] = np.sum(selected)
				guess = [1,1,1,1,1]
				for z, interpolated_wave in enumerate(range(1608)):
					# values = merge_values[selected,z]
					# values = medfilt(merge_values[selected],[1,3])[:,z]	# modified 07/02/2020 to try to eliminate dead pixels, at least in a very rough way
					values = merge_values_medianfiltered[selected,z]	# modified 29/03/2020 to do the filtering only once
					if True:
						fit = least_squares(polynomial_2D_residuals_1([merge_time_selected,merge_row_selected,values,interpolated_time,interpolated_row]),guess)
						composed_array[j,z] = polynomial_2D([interpolated_time,interpolated_row],fit.x)
						guess = fit.x
					else:			#	VERY BAD APPROXIMATION!!! DONE ONLY TO SPEED UP THE CALCULATIONS!!!
						composed_array[j,z] = np.median(values)
					# print(fit.x)
					# plt.figure();plt.tricontourf(merge_time_selected,merge_row_selected,values);plt.colorbar();plt.plot(merge_time_selected,merge_row_selected,'+k');index+=1;plt.savefig('/home/ffederic/work/Collaboratory/image'+str(index)+ '.eps',bbox_inches='tight');plt.close()
					# plt.figure();plt.tricontourf(merge_time_selected,merge_row_selected,polynomial_2D([merge_time_selected,merge_row_selected],fit.x));plt.colorbar();plt.plot(merge_time_selected,merge_row_selected,'+k');index+=1;plt.savefig('/home/ffederic/work/Collaboratory/image'+str(index)+ '.eps',bbox_inches='tight');plt.close()
					# # plt.pause(0.01)
					# composed_array[i,j,z] = polynomial_2D([interpolated_time,interpolated_row],fit.x)
				print('time '+str(interpolated_time)+' , row '+str(j) + ' , '+str(number_of_points_for_interpolation[j]) + ' points')

				# # this takes about the same time so it's not usefull
				# def calc_fit(z,merge_values=merge_values,selected=selected,interpolated_time=interpolated_time,interpolated_row=interpolated_row):
				# 	values = merge_values[selected,z]
				# 	guess = [1,1,1,1,1]
				# 	fit = least_squares(polynomial_2D_residuals([merge_time_selected,merge_row_selected,values]),guess)
				# 	return z,polynomial_2D([interpolated_time,interpolated_row],fit.x)
				# composed_row = map(calc_fit, range(1608))
				# composed_row = set(composed_row)
				# composed_row = np.array(list(composed_row)).T
				# composed_array[j] = np.array([peaks for _, peaks in sorted(zip(composed_row[0], composed_row[1]))])

			if np.min(number_of_points_for_interpolation) == 0:
				for j, interpolated_row in enumerate(row_steps):
					if (j<=0 or j>=len(row_steps)-1):
						continue
					else:
						if ( np.max(number_of_points_for_interpolation[:j]) == 0 or  np.max(number_of_points_for_interpolation[j+1:]) == 0 ):
							continue
						elif number_of_points_for_interpolation[j] == 0:
							if not os.path.exists(path_where_to_save_everything + '/extended_fit_record'):
								os.makedirs(path_where_to_save_everything + '/extended_fit_record')
							selected_time = np.abs(merge_time - interpolated_time) <= time_range_for_interp
							selected_row = np.abs(merge_row - interpolated_row) <= rows_range_for_interp
							selected_original = selected_time * selected_row
							additional_row_range = 0
							additional_time_range = 0
							selected = selected_time * selected_row
							while np.sum(selected)<np.sum(selected_original)+4:
								additional_time_range += max_time_expansion / 10
								selected_time = np.abs(merge_time - interpolated_time) <= time_range_for_interp  + additional_time_range
								selected_row = np.abs(merge_row - interpolated_row) <= rows_range_for_interp
								selected = selected_time * selected_row
							if np.sum(selected)<min_points_for_interp:
								selected_original = selected_time * selected_row
								while np.sum(selected)<np.sum(selected_original)+4:
									additional_row_range += 1
									selected_row = np.abs(merge_row - interpolated_row) <= rows_range_for_interp  + additional_row_range
									selected = selected_time * selected_row
								if np.sum(selected) < min_points_for_interp:
									selected_original = selected_time * selected_row
									while np.sum(selected) < np.sum(selected_original) + 4:
										additional_time_range += max_time_expansion / 10
										selected_time = np.abs(merge_time - interpolated_time) <= time_range_for_interp + additional_time_range
										selected_row = np.abs(merge_row - interpolated_row) <= rows_range_for_interp  + additional_row_range
										selected = selected_time * selected_row
							merge_time_selected = merge_time[selected]
							merge_row_selected = merge_row[selected]
							# interpolation_borders_expanded[i,j] = np.sum(selected)
							number_of_points_for_interpolation[j] = np.sum(selected)
							interpolation_borders_expanded_row[j] = additional_row_range
							interpolation_borders_expanded_time[j] = additional_time_range
							guess = [1, 1, 1, 1, 1]
							max_value_index = (np.max(merge_values[selected],axis=0)).argmax()
							for z, interpolated_wave in enumerate(range(1608)):
								# values = medfilt(merge_values[selected],[1,3])[:,z]	# modified 07/02/2020 to try to eliminate dead pixels, at least in a very rough way
								values = merge_values_medianfiltered[selected,z]	# modified 29/03/2020 to do the filtering only once
								fit = least_squares(polynomial_2D_residuals_1([merge_time_selected, merge_row_selected, values,interpolated_time,interpolated_row]), guess)
								# print(fit.x)
								# plt.figure();plt.tricontourf(merge_time_selected,merge_row_selected,values);plt.colorbar();plt.plot(merge_time_selected,merge_row_selected,'+k');index+=1;plt.savefig('/home/ffederic/work/Collaboratory/image'+str(index)+ '.eps',bbox_inches='tight');plt.close()
								# plt.figure();plt.tricontourf(merge_time_selected,merge_row_selected,polynomial_2D([merge_time_selected,merge_row_selected],fit.x));plt.colorbar();plt.plot(merge_time_selected,merge_row_selected,'+k');index+=1;plt.savefig('/home/ffederic/work/Collaboratory/image'+str(index)+ '.eps',bbox_inches='tight');plt.close()
								# # plt.pause(0.01)
								# composed_array[i,j,z] = polynomial_2D([interpolated_time,interpolated_row],fit.x)
								composed_array[j, z] = polynomial_2D([interpolated_time, interpolated_row], fit.x)
								guess = fit.x
								if ( (z == max_value_index) and (j%10==0) ):
									plt.figure(figsize=(10, 10))
									merge_time_selected = np.flip(np.array([x for _, x in sorted(zip(values, merge_time_selected))]),axis=0)
									merge_row_selected = np.flip(np.array([x for _, x in sorted(zip(values, merge_row_selected))]),axis=0)
									values = np.flip(np.sort(values),axis=0)
									# plt.tricontourf(merge_time_selected, merge_row_selected, values,15,cmap='rainbow')
									# plt.plot(merge_time_selected, merge_row_selected, '+k')
									plt.scatter(merge_time_selected, merge_row_selected,s=values,c=values, cmap='rainbow')
									plt.plot(interpolated_time, interpolated_row, '+r', markersize=15)
									plt.colorbar().set_label('counts [au]')
									plt.title('Original data for fitting of the row with expanded range\ntime '+str(interpolated_time)+' ms row '+str(interpolated_row)+'\n fitting params '+str(fit.x)+'\n value found '+str(polynomial_2D([interpolated_time, interpolated_row], fit.x)))
									# plt.title('Record of the row and times that required the interpolation borders expanded for good statistics')
									plt.ylabel('Index of the row')
									plt.xlabel('Time from pulse [ms]')
									# plt.pause(0.001)
									try:
										plt.savefig(path_where_to_save_everything + '/extended_fit_record' + '/' + 'fitting_expanded_example_time'+str(i)+'_row'+str(j)+'_wavel_pos'+str(z)+'_data.eps',bbox_inches='tight')
									except:
										print('there was some error in plotting fitting_expanded_example_time'+str(i)+'_row'+str(j)+'_wavel_pos'+str(z)+'.eps')
									plt.close()

									xv,yv = np.meshgrid(np.linspace(np.min(merge_time_selected),np.max(merge_time_selected),10),np.linspace(np.min(merge_row_selected),np.max(merge_row_selected),10))
									plt.figure(figsize=(10, 10))
									plt.tricontourf(xv.flatten(), yv.flatten(), polynomial_2D([xv.flatten(), yv.flatten()], fit.x),15,cmap='rainbow')
									plt.plot(merge_time_selected, merge_row_selected, '+k')
									plt.plot(interpolated_time, interpolated_row, '+r', markersize=15)
									plt.colorbar().set_label('counts [au]')
									plt.title('Result of fitting of the row with expanded range\ntime '+str(interpolated_time)+' ms row '+str(interpolated_row)+'\n fitting params '+str(fit.x)+'\n value found '+str(polynomial_2D([interpolated_time, interpolated_row], fit.x)))
									# plt.title('Record of the row and times that required the interpolation borders expanded for good statistics')
									plt.ylabel('Index of the row')
									plt.xlabel('Time from pulse [ms]')
									try:
										plt.savefig(path_where_to_save_everything + '/extended_fit_record' + '/' + 'fitting_expanded_example_time'+str(i)+'_row'+str(j)+'_wavel_pos'+str(z)+'_fit.eps',bbox_inches='tight')
									except:
										print('there was some error in plotting fitting_expanded_example_time'+str(i)+'_row'+str(j)+'_wavel_pos'+str(z)+'.eps')
									plt.close()



							print('time ' + str(interpolated_time) + ' , row ' + str(j)+ ' recovered expanding the time range from ' + str(time_range_for_interp ) + ' to ' +str(time_range_for_interp + additional_time_range) + ' ms and the row range from ' + str(rows_range_for_interp ) + ' to ' + str(rows_range_for_interp + additional_row_range) + ' rows')

			output = calc_stuff_output(interpolated_time,composed_array,interpolation_borders_expanded_row,interpolation_borders_expanded_time,number_of_points_for_interpolation)

			return output

		pool = Pool(number_cpu_available,maxtasksperchild=1)
		composed_array = [*pool.map(calc_stuff, enumerate(new_timesteps))]
		print('np.shape(composed_array)'+str(np.shape(composed_array)))

		pool.close()
		pool.join()
		pool.terminate()
		del pool

		composed_array = list(composed_array)
		time_indexes = []
		for i in range(len(composed_array)):
			time_indexes.append(composed_array[i].interpolated_time)
		composed_array = np.array([peaks for _, peaks in sorted(zip(time_indexes, composed_array))])

		temp=[]
		for i in range(len(composed_array)):
			temp.append(composed_array[i].interpolation_borders_expanded_row)
		interpolation_borders_expanded_row = np.array(temp)
		temp=[]
		temp=[]
		for i in range(len(composed_array)):
			temp.append(composed_array[i].interpolation_borders_expanded_time)
		interpolation_borders_expanded_time = np.array(temp)
		temp=[]
		temp=[]
		for i in range(len(composed_array)):
			temp.append(composed_array[i].number_of_points_for_interpolation)
		number_of_points_for_interpolation = np.array(temp)
		temp=[]
		for i in range(len(composed_array)):
			temp.append(composed_array[i].composed_array)
		composed_array = np.array(temp)
	elif True:	# here a 2D polinomial fit fith weights on the distance, and a median filter beforehand. Here I include also the sigma of single pixels

		time_range_for_interp = conventional_time_step*1.5
		rows_range_for_interp = 4
		# time_range_for_interp = conventional_time_step*1.7
		# rows_range_for_interp = 4
		max_time_expansion = conventional_time_step
		max_row_expansion = 2
		min_points_for_interp = 17

		merge_values_medianfiltered = medfilt(merge_values,[1,3])
		# merge_values_sigma_medianfiltered = np.zeros((3,*np.shape(merge_values_sigma)))
		# merge_values_sigma_medianfiltered[1] = merge_values_sigma
		# merge_values_sigma_medianfiltered[0,:,:-1] = merge_values_sigma[:,:-1]
		# merge_values_sigma_medianfiltered[2,:,1:] = merge_values_sigma[:,1:]
		# merge_values_sigma_medianfiltered = np.max(merge_values_sigma_medianfiltered,axis=0)
		merge_values_sigma_medianfiltered = generic_filter(merge_values_sigma,np.max,size=[1,3])

		from scipy.optimize import least_squares
		def polynomial_2D_residuals(coord):
			def internal_func(coeff):
				residuals = (np.array(coord[0])**2)*coeff[0] + (np.array(coord[0]))*coeff[1] + (np.array(coord[1])**2)*coeff[2] + (np.array(coord[1]))*coeff[3] + coeff[4] - np.array(coord[2])
				return residuals
			return internal_func
		def polynomial_2D(coord,coeff):
			z = (np.array(coord[0])**2)*coeff[0] + (np.array(coord[0]))*coeff[1] + (np.array(coord[1])**2)*coeff[2] + (np.array(coord[1]))*coeff[3] + coeff[4]
			return z

		def polynomial_2D_1(coord,coeff):
			z = ((np.array(coord[0])-coord[2])**2)*coeff[0] + (np.array(coord[0])-coord[2])*coeff[1] + coeff[2]*( exp(coeff[3]*(np.array(coord[0])-coord[2])) -1) + ((np.array(coord[1])-coord[3])**2)*coeff[4] + (np.array(coord[1])-coord[3])*coeff[5] + coeff[6]
			return z

		def polynomial_2D_residuals_1(coord):
			def internal_func(coeff):
				residuals = ((np.array(coord[0]))**2)*coeff[0] + (np.array(coord[0]))*coeff[1] + ((np.array(coord[1]))**2)*coeff[2] + (np.array(coord[1]))*coeff[3] + coeff[4] - np.array(coord[2])
				residuals = (residuals)**2 / (np.array(coord[3])**2)
				residuals = residuals * (1/np.abs(np.abs(np.array(coord[0]))+0.01))
				return residuals
			return internal_func

		def polynomial_2D_residuals_11(coord):
			# coord = [merge_time_selected,merge_row_selected,values,values_sigma,interpolated_time,interpolated_row]
			def internal_func(coeff):
				residuals = ((np.array(coord[0])-coord[4])**2)*coeff[0] + (np.array(coord[0])-coord[4])*coeff[1] + coeff[2]*( np.exp(coeff[3]*(np.array(coord[0])-coord[4])) -1) + ((np.array(coord[1])-coord[5])**2)*coeff[4] + (np.array(coord[1])-coord[5])*coeff[5] + coeff[6] - np.array(coord[2])
				residuals = (residuals)**2 / (np.array(coord[3])**2)
				residuals = residuals * (1/np.abs(np.abs(np.array(coord[0])-coord[4])+0.01))
				return residuals
			return internal_func

		def polynomial_2D_11(coord):
			# coord = [merge_time_selected,merge_row_selected,values,values_sigma,interpolated_time,interpolated_row]
			def internal_func(trash,*coeff):
				# print(trash)
				# print(coeff)
				residuals = ((np.array(coord[0])-coord[4])**2)*coeff[0] + (np.array(coord[0])-coord[4])*coeff[1] + coeff[2]*( np.exp(coeff[3]*(np.array(coord[0])-coord[4])) -1) + ((np.array(coord[1])-coord[5])**2)*coeff[4] + (np.array(coord[1])-coord[5])*coeff[5] + coeff[6] - np.array(coord[2])
				# residuals = (residuals)**2 / (np.array(coord[3])**2)
				residuals = residuals**2 /np.abs(np.abs(np.array(coord[0])-coord[4])+0.01)
				return residuals
			return internal_func

		def polynomial_2D_residuals_2(coord):
			def internal_func(coeff):
				residuals = (np.array(coord[0])**3)*coeff[0] + (np.array(coord[0])**2)*coeff[1] + (np.array(coord[0]))*coeff[2] + (np.array(coord[1])**2)*coeff[3] + (np.array(coord[1]))*coeff[4] + coeff[5] - np.array(coord[2])
				residuals = residuals * (1/(np.abs(np.array(coord[0])-coord[3])+0.01)**2)
				return residuals
			return internal_func

		def polynomial_2D_2(coord,coeff):
			z = (np.array(coord[0])**3)*coeff[0] +(np.array(coord[0])**2)*coeff[1] + (np.array(coord[0]))*coeff[2] + (np.array(coord[1])**2)*coeff[3] + (np.array(coord[1]))*coeff[4] + coeff[5]
			return z

		class calc_stuff_output:
			def __init__(self, interpolated_time, composed_array, composed_array_sigma, interpolation_borders_expanded_row,interpolation_borders_expanded_time, number_of_points_for_interpolation):
				self.interpolated_time = interpolated_time
				self.composed_array = composed_array
				self.composed_array_sigma = composed_array_sigma
				self.interpolation_borders_expanded_row = interpolation_borders_expanded_row
				self.interpolation_borders_expanded_time = interpolation_borders_expanded_time
				self.number_of_points_for_interpolation = number_of_points_for_interpolation

		def calc_stuff(arg,row_steps=row_steps,merge_time=merge_time,merge_row=merge_row,merge_values_medianfiltered=merge_values_medianfiltered,merge_values_sigma_medianfiltered=merge_values_sigma_medianfiltered,time_range_for_interp=time_range_for_interp,rows_range_for_interp=rows_range_for_interp,calc_stuff_output=calc_stuff_output):
		# def calc_stuff(arg,row_steps=row_steps,merge_time=merge_time,merge_row=merge_row,merge_values_medianfiltered=merge_values,merge_values_sigma_medianfiltered=merge_values_sigma,time_range_for_interp=time_range_for_interp,rows_range_for_interp=rows_range_for_interp,calc_stuff_output=calc_stuff_output):
			i = arg[0]
			interpolated_time = arg[1]
			print(interpolated_time)
			interpolation_borders_expanded_row = np.zeros_like(row_steps)
			interpolation_borders_expanded_time = np.zeros_like(row_steps)
			number_of_points_for_interpolation = np.zeros_like(row_steps)
			composed_array = np.zeros((len(row_steps),1608))
			composed_array_sigma = np.zeros((len(row_steps),1608))

			# bds=[[-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]]
			bds=[[-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,0],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]]

			# for i, interpolated_time in enumerate(new_timesteps):
			selected_time = np.abs(merge_time - interpolated_time) <=time_range_for_interp
			if len(np.unique(merge_time[selected_time]))<4:
				output = calc_stuff_output(interpolated_time,composed_array,composed_array_sigma,interpolation_borders_expanded_row,interpolation_borders_expanded_time,number_of_points_for_interpolation)
				return output
			record_previous_sol = 0

			class output:
				def __init__(self, out):
					self.out = out
			for j, interpolated_row in enumerate(row_steps):
				selected_row = np.abs(merge_row - interpolated_row) <=rows_range_for_interp
				selected = selected_time*selected_row
				if np.sum(selected)<min_points_for_interp:
					print('time ' + str(interpolated_time) + ' , row ' + str(j)+' skipped, only '+str(np.sum(selected))+ ' points found')
					continue
				# if j<600 or j>650:
				# 	continue
				merge_time_selected = merge_time[selected]
				merge_row_selected = merge_row[selected]
				# interpolation_borders_expanded[i,j] = np.sum(selected)
				number_of_points_for_interpolation[j] = np.sum(selected)
				guess = [0,0,0,0,0,0,100]
				# for z, interpolated_wave in enumerate(range(1608)):
				def calc_fit(z,record_previous_sol=record_previous_sol,j=j,merge_values_medianfiltered_int=merge_values_medianfiltered[selected],merge_values_sigma_medianfiltered_int=merge_values_sigma_medianfiltered[selected],merge_time_selected=merge_time_selected,merge_row_selected=merge_row_selected,interpolated_time=interpolated_time,interpolated_row=interpolated_row,bds=bds):
					# values = merge_values[selected,z]
					values = merge_values_medianfiltered_int[:,z]	# modified 29/03/2020 to do the filtering only once
					values_sigma = merge_values_sigma_medianfiltered_int[:,z]
					# start = tm.time()
					if record_previous_sol==0:
						# guess = [1,1,1,1,1,1,1]
						guess = [0,0,0,0,0,0,0]
					else:
						guess = record_previous_sol[z]
					guess[-1]=np.nanmean(values)
					time_scale = (merge_time_selected.max()-merge_time_selected.min())/np.std(values)
					row_scale = (merge_row_selected.max()-merge_row_selected.min())/np.std(values)
					x_scale=[time_scale**2,time_scale,time_scale,1,row_scale**2,row_scale,1]
					coord = [merge_time_selected,merge_row_selected,values,values_sigma,interpolated_time,interpolated_row]
					try:
						if (z<1+np.interp(j,geom['tilt'][0][:2,0],geom['tilt'][0][:2,1])) and (z>-1+np.interp(j,geom['tilt'][0][2:,0],geom['tilt'][0][2:,1])):
							fit = curve_fit(polynomial_2D_11(coord),np.zeros_like(values),np.zeros_like(values),p0=guess,sigma=values_sigma,absolute_sigma=True,x_scale=x_scale,bounds=bds,ftol=1e-7,maxfev=1e5)
							# verified 2022/01/17. it could be sped up with scipy.optimize.fmin_l_bfgs_b but it actually works fine, so I won't
						else:
							fit = curve_fit(polynomial_2D_11(coord),np.zeros_like(values),np.zeros_like(values),p0=guess,sigma=values_sigma,absolute_sigma=True,x_scale=x_scale,bounds=bds,ftol=1e-5,maxfev=1e5)
						return z,fit[0][-1],fit[1][-1,-1]**0.5,output(fit[0])
					except:
						print('main fast fit failed, z='+str(z)+', j='+str(j))
						if (z<1+np.interp(j,geom['tilt'][0][:2,0],geom['tilt'][0][:2,1])) and (z>-1+np.interp(j,geom['tilt'][0][2:,0],geom['tilt'][0][2:,1])):
							fit = least_squares(polynomial_2D_residuals_11(coord),guess,bounds=bds,max_nfev=125,ftol=1e-7)
						else:
							fit = least_squares(polynomial_2D_residuals_11(coord),guess,bounds=bds,max_nfev=75,ftol=1e-5)
						# print(tm.time()-start)
						_, s, VT = svd(fit.jac, full_matrices=False)	# this method is from https://stackoverflow.com/questions/40187517/getting-covariance-matrix-of-fitted-parameters-from-scipy-optimize-least-squares
						threshold = np.finfo(float).eps * max(fit.jac.shape) * s[0]
						s = s[s > threshold]
						VT = VT[:s.size]
						pcov = np.dot(VT.T / s**2, VT)
						# print(tm.time()-start)
						# result = polynomial_2D_1([interpolated_time,interpolated_row,interpolated_time,interpolated_row],correlated_values(fit.x,pcov))
						# composed_array[j,z] = fit.x[-1]
						# composed_array_sigma[j,z] = pcov[-1,-1]**0.5
						# guess = fit.x
						# print(fit.x)
						# plt.figure();plt.tricontourf(merge_time_selected,merge_row_selected,values);plt.colorbar();plt.plot(merge_time_selected,merge_row_selected,'+k');index+=1;plt.savefig('/home/ffederic/work/Collaboratory/image'+str(index)+ '.eps',bbox_inches='tight');plt.close()
						# plt.figure();plt.tricontourf(merge_time_selected,merge_row_selected,polynomial_2D_1([merge_time_selected,merge_row_selected,interpolated_time,interpolated_row],fit[0]).astype(float));plt.colorbar();plt.plot(merge_time_selected,merge_row_selected,'+k');index+=1;plt.savefig('/home/ffederic/work/Collaboratory/image'+str(index)+ '.eps',bbox_inches='tight');plt.close()
						# # plt.pause(0.01)
						# composed_array[i,j,z] = polynomial_2D([interpolated_time,interpolated_row],fit.x)
						return z,fit.x[-1],pcov[-1,-1]**0.5,output(fit.x)


				composed_row = map(calc_fit, range(1608))
				composed_row = set(composed_row)
				composed_row = np.array(list(composed_row)).T
				composed_array[j] = np.array([peaks for _, peaks in sorted(zip(composed_row[0], composed_row[1]))])
				composed_array_sigma[j] = np.array([peaks for _, peaks in sorted(zip(composed_row[0], composed_row[2]))])
				record_previous_sol = [peaks.out for _, peaks in sorted(zip(composed_row[0], composed_row[3]))]

				print('time '+str(interpolated_time)+' , row '+str(j) + ' , '+str(number_of_points_for_interpolation[j]) + ' points')

				# # this takes about the same time so it's not usefull
				# def calc_fit(z,merge_values=merge_values,selected=selected,interpolated_time=interpolated_time,interpolated_row=interpolated_row):
				# 	values = merge_values[selected,z]
				# 	guess = [1,1,1,1,1]
				# 	fit = least_squares(polynomial_2D_residuals([merge_time_selected,merge_row_selected,values]),guess)
				# 	return z,polynomial_2D([interpolated_time,interpolated_row],fit.x)
				# composed_row = map(calc_fit, range(1608))
				# composed_row = set(composed_row)
				# composed_row = np.array(list(composed_row)).T
				# composed_array[j] = np.array([peaks for _, peaks in sorted(zip(composed_row[0], composed_row[1]))])

			if np.min(number_of_points_for_interpolation) == 0:
				for j, interpolated_row in enumerate(row_steps):
					guess = [0,0,0,0,0,0,100]
					if (j<=0 or j>=len(row_steps)-1):
						continue
					else:
						if ( np.max(number_of_points_for_interpolation[:j]) == 0 or  np.max(number_of_points_for_interpolation[j+1:]) == 0 ):
							continue
						elif number_of_points_for_interpolation[j] == 0:
							if not os.path.exists(path_where_to_save_everything + '/extended_fit_record'):
								os.makedirs(path_where_to_save_everything + '/extended_fit_record')
							selected_time = np.abs(merge_time - interpolated_time) <= time_range_for_interp
							selected_row = np.abs(merge_row - interpolated_row) <= rows_range_for_interp
							selected_original = selected_time * selected_row
							additional_row_range = 0
							additional_time_range = 0
							selected = selected_time * selected_row
							while np.sum(selected)<np.sum(selected_original)+4:
								additional_time_range += max_time_expansion / 10
								selected_time = np.abs(merge_time - interpolated_time) <= time_range_for_interp  + additional_time_range
								selected_row = np.abs(merge_row - interpolated_row) <= rows_range_for_interp
								selected = selected_time * selected_row
							if np.sum(selected)<min_points_for_interp:
								selected_original = selected_time * selected_row
								while np.sum(selected)<np.sum(selected_original)+4:
									additional_row_range += 1
									selected_row = np.abs(merge_row - interpolated_row) <= rows_range_for_interp  + additional_row_range
									selected = selected_time * selected_row
								if np.sum(selected) < min_points_for_interp:
									selected_original = selected_time * selected_row
									while np.sum(selected) < np.sum(selected_original) + 4:
										additional_time_range += max_time_expansion / 10
										selected_time = np.abs(merge_time - interpolated_time) <= time_range_for_interp + additional_time_range
										selected_row = np.abs(merge_row - interpolated_row) <= rows_range_for_interp  + additional_row_range
										selected = selected_time * selected_row
							merge_time_selected = merge_time[selected]
							merge_row_selected = merge_row[selected]
							# interpolation_borders_expanded[i,j] = np.sum(selected)
							number_of_points_for_interpolation[j] = np.sum(selected)
							interpolation_borders_expanded_row[j] = additional_row_range
							interpolation_borders_expanded_time[j] = additional_time_range
							max_value_index = (np.max(merge_values_medianfiltered[selected],axis=0)).argmax()
							for z, interpolated_wave in enumerate(range(1608)):
								values = merge_values_medianfiltered[selected,z]	# modified 29/03/2020 to do the filtering only once
								values_sigma = merge_values_sigma_medianfiltered[selected,z]
								guess = record_previous_sol[z]
								guess[-1]=np.nanmean(values)
								time_scale = (merge_time_selected.max()-merge_time_selected.min())/np.std(values)
								row_scale = (merge_row_selected.max()-merge_row_selected.min())/np.std(values)
								x_scale=[time_scale**2,time_scale,time_scale,1,row_scale**2,row_scale,1]
								coord = [merge_time_selected,merge_row_selected,values,values_sigma,interpolated_time,interpolated_row]
								try:
									if (z<1+np.interp(j,geom['tilt'][0][:2,0],geom['tilt'][0][:2,1])) and (z>-1+np.interp(j,geom['tilt'][0][2:,0],geom['tilt'][0][2:,1])):
										fit = curve_fit(polynomial_2D_11(coord),np.zeros_like(values),np.zeros_like(values),p0=guess,sigma=values_sigma,absolute_sigma=True,x_scale=x_scale,bounds=bds,ftol=1e-7,maxfev=1e5)
									else:
										fit = curve_fit(polynomial_2D_11(coord),np.zeros_like(values),np.zeros_like(values),p0=guess,sigma=values_sigma,absolute_sigma=True,x_scale=x_scale,bounds=bds,ftol=1e-5,maxfev=1e5)
									composed_array[j,z] = fit[0][-1]
									composed_array_sigma[j,z] = fit[0][-1,-1]**0.5
								except:
									print('main fast fit failed, z='+str(z)+', j='+str(j))
									if (z<1+np.interp(j,geom['tilt'][0][:2,0],geom['tilt'][0][:2,1])) and (z>-1+np.interp(j,geom['tilt'][0][2:,0],geom['tilt'][0][2:,1])):
										fit = least_squares(polynomial_2D_residuals_11(coord),guess,bounds=bds,max_nfev=500,ftol=1e-7)
									else:
										fit = least_squares(polynomial_2D_residuals_11(coord),guess,bounds=bds,max_nfev=100,ftol=1e-5)
									_, s, VT = svd(fit.jac, full_matrices=False)	# this method is from https://stackoverflow.com/questions/40187517/getting-covariance-matrix-of-fitted-parameters-from-scipy-optimize-least-squares
									threshold = np.finfo(float).eps * max(fit.jac.shape) * s[0]
									s = s[s > threshold]
									VT = VT[:s.size]
									pcov = np.dot(VT.T / s**2, VT)
									composed_array[j,z] = fit.x[-1]
									composed_array_sigma[j,z] = pcov[-1,-1]**0.5
									# guess = fit.x
								if False:	# I stop doing the plots because it slows down too much the process 25/05/2020
									if ( (z == max_value_index) and (j%140==0) ):
										result = polynomial_2D_1([interpolated_time,interpolated_row,interpolated_time,interpolated_row],correlated_values(fit.x,pcov))
										merge_time_selected = np.flip(np.array([x for _, x in sorted(zip(values, merge_time_selected))]),axis=0)
										merge_row_selected = np.flip(np.array([x for _, x in sorted(zip(values, merge_row_selected))]),axis=0)
										values = np.flip(np.sort(values),axis=0)
										plt.figure(figsize=(10, 10))
										# plt.tricontourf(merge_time_selected, merge_row_selected, values,15,cmap='rainbow')
										# plt.plot(merge_time_selected, merge_row_selected, '+k')
										plt.scatter(merge_time_selected, merge_row_selected,s=values,c=merge_values[selected,z], cmap='rainbow')
										plt.plot(interpolated_time, interpolated_row, '+r', markersize=15)
										plt.colorbar().set_label('counts [au]')
										plt.title('Original data for fitting of the row with expanded range\ntime '+str(interpolated_time)+' ms row '+str(interpolated_row)+'\n fitting params '+str(fit.x)+'\n value found '+str(result)+'\ncolor=values not median filtered, size=values median filtered')
										# plt.title('Record of the row and times that required the interpolation borders expanded for good statistics')
										plt.ylabel('Index of the row')
										plt.xlabel('Time from pulse [ms]')
										# plt.pause(0.001)
										try:
											plt.savefig(path_where_to_save_everything + '/extended_fit_record' + '/' + 'fitting_expanded_example_time'+str(i)+'_row'+str(j)+'_wavel_pos'+str(z)+'_data.eps',bbox_inches='tight')
										except:
											print('there was some error in plotting fitting_expanded_example_time'+str(i)+'_row'+str(j)+'_wavel_pos'+str(z)+'.eps')
										plt.close('all')

										xv,yv = np.meshgrid(np.linspace(np.min(merge_time_selected),np.max(merge_time_selected),10),np.linspace(np.min(merge_row_selected),np.max(merge_row_selected),10))
										plt.figure(figsize=(10, 10))
										plt.tricontourf(xv.flatten(), yv.flatten(), polynomial_2D_1([xv.flatten(), yv.flatten(),interpolated_time,interpolated_row], fit.x).astype(float),15,cmap='rainbow')
										plt.plot(merge_time_selected, merge_row_selected, '+k')
										plt.plot(interpolated_time, interpolated_row, '+r', markersize=15)
										plt.colorbar().set_label('counts [au]')
										plt.title('Result of fitting of the row with expanded range\ntime '+str(interpolated_time)+' ms row '+str(interpolated_row)+'\n fitting params '+str(fit.x)+'\n value found '+str(result))
										# plt.title('Record of the row and times that required the interpolation borders expanded for good statistics')
										plt.ylabel('Index of the row')
										plt.xlabel('Time from pulse [ms]')
										try:
											plt.savefig(path_where_to_save_everything + '/extended_fit_record' + '/' + 'fitting_expanded_example_time'+str(i)+'_row'+str(j)+'_wavel_pos'+str(z)+'_fit.eps',bbox_inches='tight')
										except:
											print('there was some error in plotting fitting_expanded_example_time'+str(i)+'_row'+str(j)+'_wavel_pos'+str(z)+'.eps')
										plt.close('all')



							print('time ' + str(interpolated_time) + ' , row ' + str(j)+ ' recovered expanding the time range from ' + str(time_range_for_interp ) + ' to ' +str(time_range_for_interp + additional_time_range) + ' ms and the row range from ' + str(rows_range_for_interp ) + ' to ' + str(rows_range_for_interp + additional_row_range) + ' rows')

			output = calc_stuff_output(interpolated_time,composed_array,composed_array_sigma,interpolation_borders_expanded_row,interpolation_borders_expanded_time,number_of_points_for_interpolation)

			return output

		pool = Pool(number_cpu_available,maxtasksperchild=1)
		composed_array = [*pool.map(calc_stuff, enumerate(new_timesteps))]
		print('np.shape(composed_array)'+str(np.shape(composed_array)))

		pool.close()
		pool.join()
		pool.terminate()
		del pool

		composed_array = list(composed_array)
		time_indexes = []
		for i in range(len(composed_array)):
			time_indexes.append(composed_array[i].interpolated_time)
		composed_array = np.array([peaks for _, peaks in sorted(zip(time_indexes, composed_array))])

		temp=[]
		for i in range(len(composed_array)):
			temp.append(composed_array[i].interpolation_borders_expanded_row)
		interpolation_borders_expanded_row = np.array(temp)
		temp=[]
		for i in range(len(composed_array)):
			temp.append(composed_array[i].interpolation_borders_expanded_time)
		interpolation_borders_expanded_time = np.array(temp)
		temp=[]
		for i in range(len(composed_array)):
			temp.append(composed_array[i].number_of_points_for_interpolation)
		number_of_points_for_interpolation = np.array(temp)
		temp=[]
		for i in range(len(composed_array)):
			temp.append(composed_array[i].composed_array_sigma)
		composed_array_sigma = np.array(temp)
		temp=[]
		for i in range(len(composed_array)):
			temp.append(composed_array[i].composed_array)
		composed_array = np.array(temp)	#


	elif False:


		class calc_stuff_output:
			def __init__(self, interpolated_time, composed_array, interpolation_borders_expanded):
				self.interpolated_time = interpolated_time
				self.composed_array = composed_array
				self.interpolation_borders_expanded = interpolation_borders_expanded


		def calc_stuff(arg,row_steps=row_steps,merge_time=merge_time,merge_row=merge_row,merge_values=merge_values,time_range_for_interp=time_range_for_interp,rows_range_for_interp=rows_range_for_interp,min_rows_for_interp=min_rows_for_interp,max_time_expansion=max_time_expansion,grade_of_interpolation=grade_of_interpolation,path_where_to_save_everything=path_where_to_save_everything,calc_stuff_output=calc_stuff_output,time_vs_row_weighting=time_vs_row_weighting):
			import numpy as np
			import matplotlib.pyplot as plt

			i = arg[0]
			interpolated_time = arg[1]
			interpolation_borders_expanded = np.zeros_like(row_steps)
			composed_array = np.zeros((len(row_steps),1608))
			for j, interpolated_row in enumerate(row_steps):
				# if interpolated_row <= geom['bin00b'][0]:
				# 	bin_bottom,bin_top = 0,1
				# elif interpolated_row >= geom['bin00b'][0]+40*geom['binInterv'][0]:
				# 	bin_bottom, bin_top = -2,-1
				# else:
				# 	bin_bottom,bin_top = (np.linspace(0,len(iBin)-1,len(iBin))[((np.abs(iBin-interpolated_row)-geom['binInterv'][0])<0)]).astype('int')
				selected_time = np.abs(merge_time - interpolated_time) <= time_range_for_interp
				selected_row = np.abs(merge_row - interpolated_row) <= rows_range_for_interp
				# selected_bin = np.logical_and(merge_row >= iBin[bin_bottom] ,  merge_row <= iBin[bin_top])
				selected = selected_time*selected_row#*selected_bin
				additional_time_range = 0
				while (len(np.unique(merge_row[selected])) < min_rows_for_interp and additional_time_range<max_time_expansion):
				# while (np.sum(selected) <= min_rows_for_interp and additional_time_range < max_time_expansion):
					additional_row_range = 0
					selected_time = np.abs(merge_time - interpolated_time) <= time_range_for_interp + additional_time_range
					selected_row = np.abs(merge_row - interpolated_row) <= rows_range_for_interp + additional_row_range
					selected = selected_time * selected_row# * selected_bin
					while (len(np.unique(merge_row[selected])) < min_rows_for_interp and additional_row_range<max_row_expansion):
					# while (np.sum(selected) <= min_rows_for_interp and additional_row_range < max_row_expansion):
						additional_row_range += 1
						print('interpolated_time ' + str(interpolated_time)+' row '+str(interpolated_row) + ' row window increased of ' + str(additional_row_range) + str(np.unique(merge_row[selected])))
						selected_row = np.abs(merge_row - interpolated_row) <= rows_range_for_interp + additional_row_range
						selected = selected_time * selected_row# * selected_bin
						interpolation_borders_expanded[j] += 1

					if len(np.unique(merge_row[selected])) < min_rows_for_interp:
					# if np.sum(selected) <= min_rows_for_interp:
						additional_time_range += max_time_expansion / 10
						interpolation_borders_expanded[j] += 1
						print('interpolated_time ' + str(interpolated_time)+' row '+str(interpolated_row) + ' time window increased of ' + str(additional_time_range) + str(np.unique(merge_row[selected])))

				if len(np.unique(merge_row[selected])) < min(min_rows_for_interp,grade_of_interpolation+1):
				# if np.sum(selected) < min_rows_for_interp:
					continue
				if (not(interpolated_row in merge_row[selected]) and (np.sum(merge_row[selected]-interpolated_row>0) in [0,len(merge_row[selected])]) and (np.min(np.abs(merge_row[selected]-interpolated_row))>1) ):
					# if np.sum(merge_row[selected]-interpolated_row>0) in [0,len(merge_row[selected])]:
					# 	if np.min(np.abs(merge_row[selected]-interpolated_row))>1:
					# print('skipped')
					continue
				# print('not skipped')


				print('interpolated_time ' + str(interpolated_time) + ' row ' + str(interpolated_row) + ' rows ' + str(np.unique(merge_row[selected])))
				weight_from_time_difference = 1/(np.abs(merge_time[selected] - interpolated_time)+np.max(np.abs(merge_time[selected] - interpolated_time))/100)
				weight_from_time_difference = weight_from_time_difference/np.max(weight_from_time_difference)
				weight_from_row_difference = 1/(np.abs(merge_row[selected] - interpolated_row)+np.max(np.abs(merge_row[selected] - interpolated_row))/100)
				weight_from_row_difference = weight_from_row_difference/np.max(weight_from_row_difference)
				# I want the penalty for time distance to be worst than row distance
				weight_from_time_difference_multiplier = np.log(time_vs_row_weighting*np.max(weight_from_row_difference)/np.min(weight_from_row_difference))/np.log(np.max(weight_from_time_difference)/np.min(weight_from_time_difference))
				weight_from_time_difference = np.power(weight_from_time_difference,max(weight_from_time_difference_multiplier,1))
				# weight_combined = weight_from_time_difference * weight_from_row_difference
				weight_combined = np.sqrt(weight_from_time_difference**2 + (0.7*weight_from_row_difference)**2)
				weight_combined = weight_combined/np.max(weight_combined)

				if False:
					# composed_array[j] = np.sum((merge_values[selected].T)*weight_combined,axis=1)/np.sum(weight_combined)
					composed_array[j] = np.mean(merge_values[selected].T,axis=1)
				else:
					# grade_of_interpolation_temp = min(grade_of_interpolation, len(np.unique(merge_row[selected]))-1)

					def calc_fit(z,j=j,grade_of_interpolation=grade_of_interpolation,values=merge_values[selected].T,weight_combined=weight_combined,merge_row_selected=merge_row[selected],merge_time_selected=merge_time[selected]):
						import numpy as np

						values = values[z]
						# print(np.shape(merge_row_selected))
						# print(np.shape(values))
						# print(np.shape(grade_of_interpolation))
						# print(np.shape(weight_combined))
						fit_coef_linear = np.polyfit(merge_row_selected, values, grade_of_interpolation, w=weight_combined)
						# print([i, j, z])
						# # print(fit_coef_linear)
						# print(np.polyval(fit_coef_linear, j))
						return z,np.polyval(fit_coef_linear, j)


					# composed_row = zip(*pool.map(calc_stuff, np.concatenate((np.array([np.linspace(0,1607,1608)]).T, np.ones((1608,1)) * i, np.ones((1608,1)) * j,np.ones((1608,1)) * grade_of_interpolation, merge_values[selected].T,np.ones((1608, 1)) * weight_combined, np.ones((1608, 1)) * merge_row[selected]) ,axis=1)))
					# print('mark1')
					composed_row = map(calc_fit, range(1608))

					composed_row = set(composed_row)
					# print('mark2')
					composed_row = np.array(list(composed_row)).T
					# print('mark3')
					# print(np.shape(composed_row))

					# for index,string in enumerate(composed_row):
					# 	if not (string[0] in range(1608)):
					# 		composed_row[index] = np.flip(string,axis=0)
					# composed_row = composed_row.T


					# print(np.shape(composed_row[0]))
					# print(np.shape(composed_row[1]))
					composed_array[j] = np.array([peaks for _, peaks in sorted(zip(composed_row[0], composed_row[1]))])
			# plt.figure(i)
			# plt.imshow(composed_array, 'rainbow', origin='lower', vmin=0)
			# plt.colorbar()
			# plt.title('frame' + str(i))
			# plt.savefig(path_where_to_save_everything + '/' + 'composed_array_frame_' + str(i) + '.eps',bbox_inches='tight')
			# plt.close()

			# class calc_stuff_output:
			# 	def __init__(self,interpolated_time,composed_array,interpolation_borders_expanded):
			# 		self.interpolated_time = interpolated_time
			# 		self.composed_array = composed_array
			# 		self.interpolation_borders_expanded = interpolation_borders_expanded

			output = calc_stuff_output(interpolated_time,composed_array,interpolation_borders_expanded)

			return output
			# return interpolated_time,composed_array,interpolation_borders_expanded


		# from multiprocessing import Pool
		pool = Pool(number_cpu_available,maxtasksperchild=1)


		# iBin = np.array([0,*(geom['bin00b'][0] + range(41)*geom['binInterv'][0]),np.shape(composed_array)[-1]])
		# for i,interpolated_time in enumerate(new_timesteps):
			# for j, interpolated_row in enumerate(row_steps):
			# 	# if interpolated_row <= geom['bin00b'][0]:
			# 	# 	bin_bottom,bin_top = 0,1
			# 	# elif interpolated_row >= geom['bin00b'][0]+40*geom['binInterv'][0]:
			# 	# 	bin_bottom, bin_top = -2,-1
			# 	# else:
			# 	# 	bin_bottom,bin_top = (np.linspace(0,len(iBin)-1,len(iBin))[((np.abs(iBin-interpolated_row)-geom['binInterv'][0])<0)]).astype('int')
			# 	selected_time = np.abs(merge_time - interpolated_time) <= time_range_for_interp
			# 	selected_row = np.abs(merge_row - interpolated_row) <= rows_range_for_interp
			# 	# selected_bin = np.logical_and(merge_row >= iBin[bin_bottom] ,  merge_row <= iBin[bin_top])
			# 	selected = selected_time*selected_row#*selected_bin
			# 	additional_time_range = 0
			# 	while (len(np.unique(merge_row[selected])) < min_rows_for_interp and additional_time_range<max_time_expansion):
			# 	# while (np.sum(selected) <= min_rows_for_interp and additional_time_range < max_time_expansion):
			# 		additional_row_range = 0
			# 		selected_time = np.abs(merge_time - interpolated_time) <= time_range_for_interp + additional_time_range
			# 		selected_row = np.abs(merge_row - interpolated_row) <= rows_range_for_interp + additional_row_range
			# 		selected = selected_time * selected_row# * selected_bin
			# 		while (len(np.unique(merge_row[selected])) < min_rows_for_interp and additional_row_range<max_row_expansion):
			# 		# while (np.sum(selected) <= min_rows_for_interp and additional_row_range < max_row_expansion):
			# 			additional_row_range += 1
			# 			print('interpolated_time ' + str(interpolated_time)+' row '+str(interpolated_row) + ' row increased of ' + str(additional_row_range) + str(np.unique(merge_row[selected])))
			# 			selected_row = np.abs(merge_row - interpolated_row) <= rows_range_for_interp + additional_row_range
			# 			selected = selected_time * selected_row# * selected_bin
			# 			interpolation_borders_expanded[i, j] += 1
			#
			# 		if len(np.unique(merge_row[selected])) < min_rows_for_interp:
			# 		# if np.sum(selected) <= min_rows_for_interp:
			# 			additional_time_range += max_time_expansion / 10
			# 			interpolation_borders_expanded[i, j] += 1
			# 			print('interpolated_time ' + str(interpolated_time)+' row '+str(interpolated_row) + ' time increased of ' + str(additional_time_range) + str(np.unique(merge_row[selected])))
			#
			# 	if len(np.unique(merge_row[selected])) < min(min_rows_for_interp,grade_of_interpolation+1):
			# 	# if np.sum(selected) < min_rows_for_interp:
			# 		continue
			# 	if (not(interpolated_row in merge_row[selected]) and (np.sum(merge_row[selected]-interpolated_row>0) in [0,len(merge_row[selected])]) and (np.min(np.abs(merge_row[selected]-interpolated_row))>1) ):
			# 		# if np.sum(merge_row[selected]-interpolated_row>0) in [0,len(merge_row[selected])]:
			# 		# 	if np.min(np.abs(merge_row[selected]-interpolated_row))>1:
			# 		# print('skipped')
			# 		continue
			# 	# print('not skipped')
			#
			#
			# 	print('interpolated_time ' + str(interpolated_time) + ' row ' + str(interpolated_row) + ' rows ' + str(np.unique(merge_row[selected])))
			# 	weight_from_time_difference = 1/(np.abs(merge_time[selected] - interpolated_time)+np.max(np.abs(merge_time[selected] - interpolated_time))/100)
			# 	weight_from_time_difference = weight_from_time_difference/np.max(weight_from_time_difference)
			# 	weight_from_row_difference = 1/(np.abs(merge_row[selected] - interpolated_row)+np.max(np.abs(merge_row[selected] - interpolated_row))/100)
			# 	weight_from_row_difference = weight_from_row_difference/np.max(weight_from_row_difference)
			# 	# I want the penalty for time distance to be worst than row distance
			# 	weight_from_time_difference_multiplier = np.log(1.5*np.max(weight_from_row_difference)/np.min(weight_from_row_difference))/np.log(np.max(weight_from_time_difference)/np.min(weight_from_time_difference))
			# 	weight_from_time_difference = np.power(weight_from_time_difference,max(weight_from_time_difference_multiplier,1))
			# 	# weight_combined = weight_from_time_difference * weight_from_row_difference
			# 	weight_combined = np.sqrt(weight_from_time_difference**2 + (0.5*weight_from_row_difference)**2)
			# 	weight_combined = weight_combined/np.max(weight_combined)
			# 	# grade_of_interpolation_temp = min(grade_of_interpolation, len(np.unique(merge_row[selected]))-1)
			#
			# 	def calc_fit(z,i,j,grade_of_interpolation,num_arg_to_split,values,weight_combined,merge_row_selected):
			# 		import numpy as np
			# 		fit_coef_linear = np.polyfit(merge_row_selected, values, grade_of_interpolation, w=weight_combined)
			# 		# print([i, j, z])
			# 		# # print(fit_coef_linear)
			# 		# print(np.polyval(fit_coef_linear, j))
			# 		return z,np.polyval(fit_coef_linear, j)
			#
			#
			# 	# composed_row = zip(*pool.map(calc_stuff, np.concatenate((np.array([np.linspace(0,1607,1608)]).T, np.ones((1608,1)) * i, np.ones((1608,1)) * j,np.ones((1608,1)) * grade_of_interpolation, merge_values[selected].T,np.ones((1608, 1)) * weight_combined, np.ones((1608, 1)) * merge_row[selected]) ,axis=1)))
			# 	# print('mark1')
			# 	composed_row = map(calc_fit, range((1608)), np.ones((1608)) * i, np.ones((1608)) * j,
			# 			  np.ones((1608)) * grade_of_interpolation, merge_values[selected].T,
			# 			  np.ones((1608, 1)) * weight_combined, np.ones((1608, 1)) * merge_row[selected])
			#
			# 	composed_row = set(composed_row)
			# 	# print('mark2')
			# 	composed_row = np.array(list(composed_row)).T
			# 	# print('mark3')
			# 	# print(np.shape(composed_row))
			#
			# 	# for index,string in enumerate(composed_row):
			# 	# 	if not (string[0] in range(1608)):
			# 	# 		composed_row[index] = np.flip(string,axis=0)
			# 	# composed_row = composed_row.T
			#
			#
			# 	print(composed_row[0])
			# 	print(composed_row[1])
			# 	composed_array[i, j] = np.array([peaks for _, peaks in sorted(zip(composed_row[0], composed_row[1]))])
			#
			#
			# 	# from multiprocessing import Pool
			# 	# pool = Pool(4)
			# 	# composed_array[i, j] = zip(*pool.map(calc_stuff, range(np.shape(composed_array)[-1]), np.ones((np.shape(composed_array)[-1]))*i, np.ones((np.shape(composed_array)[-1]))*j ,merge_values[selected],np.ones((np.shape(composed_array)[-1]))*weight_combined , np.ones((np.shape(composed_array)[-1]))*merge_row[selected] ,composed_array))
			#
			# 	# for z in range(np.shape(composed_array)[-1]):
			# 	# 	# print(str([i,j,z]))
			# 	# 	fit_coef_linear = np.polyfit(merge_row[selected],merge_values[selected,z],grade_of_interpolation,w=weight_combined)
			# 	# 	# plt.figure(),plt.errorbar(merge_row[selected],merge_values[selected,z],yerr=weight_combined),plt.plot(np.sort(merge_row[selected]),np.polyval(fit_coef_linear,np.sort(merge_row[selected]))),plt.pause(0.01)
			# 	# 	composed_array[i,j,z] = np.polyval(fit_coef_linear,interpolated_row)
			# plt.figure();plt.imshow(composed_array[i],'rainbow',origin='lower',vmin=0);plt.colorbar();plt.title('frame'+str(i));plt.savefig(path_where_to_save_everything+ '/' + 'composed_array_frame_'+str(i)+'.eps', bbox_inches='tight');plt.close()
		# # np.save(path_where_to_save_everything + '/merge' + str(merge_ID_target) + '_temp3', composed_array[i])


		# composed_array = map(calc_stuff, enumerate(new_timesteps))
		composed_array = [*pool.map(calc_stuff, enumerate(new_timesteps))]
		# composed_array = set(composed_array)
		print('np.shape(composed_array)'+str(np.shape(composed_array)))

		pool.close()
		pool.join()
		pool.terminate()
		del pool

		composed_array = list(composed_array)
		time_indexes = []
		for i in range(len(composed_array)):
			time_indexes.append(composed_array[i].interpolated_time)
		composed_array = np.array([peaks for _, peaks in sorted(zip(time_indexes, composed_array))])

		temp=[]
		for i in range(len(composed_array)):
			temp.append(composed_array[i].interpolation_borders_expanded)
		interpolation_borders_expanded = np.array(temp)
		temp=[]
		for i in range(len(composed_array)):
			temp.append(composed_array[i].composed_array)
		composed_array = np.array(temp)


	plt.figure(figsize=(20, 10))
	plt.imshow(interpolation_borders_expanded_row,'rainbow', origin='lower',extent=[0,np.shape(interpolation_borders_expanded_row)[1]-1,np.min(new_timesteps),np.max(new_timesteps)],aspect=100)
	plt.colorbar().set_label('number of rows added (it is this x2) [au]')
	plt.title('Record of the increase in row range required to find 10 points to perform the profile fitting')
	# plt.title('Record of the row and times that required the interpolation borders expanded for good statistics')
	plt.xlabel('Index of the row')
	plt.ylabel('Time from pulse [ms]')
	# plt.pause(0.001)
	plt.savefig(path_where_to_save_everything+ '/' + 'interpolation_borders_expanded_row.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.imshow(interpolation_borders_expanded_time,'rainbow', origin='lower',extent=[0,np.shape(interpolation_borders_expanded_time)[1]-1,np.min(new_timesteps),np.max(new_timesteps)],aspect=100)
	plt.colorbar().set_label('time added (it is this x2) [au]')
	plt.title('Record of the increase in time range required to find 10 points to perform the profile fitting')
	# plt.title('Record of the row and times that required the interpolation borders expanded for good statistics')
	plt.xlabel('Index of the row')
	plt.ylabel('Time from pulse [ms]')
	# plt.pause(0.001)
	plt.savefig(path_where_to_save_everything+ '/' + 'interpolation_borders_expanded_time.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.imshow(number_of_points_for_interpolation,'rainbow', origin='lower',extent=[0,np.shape(number_of_points_for_interpolation)[1]-1,np.min(new_timesteps),np.max(new_timesteps)],aspect=100)
	plt.colorbar().set_label('number of points used [au]')
	plt.title('Record of the number of points find within the range used to perform the profile fitting')
	# plt.title('Record of the row and times that required the interpolation borders expanded for good statistics')
	plt.xlabel('Index of the row')
	plt.ylabel('Time from pulse [ms]')
	# plt.pause(0.001)
	plt.savefig(path_where_to_save_everything+ '/' + 'number_of_points_for_interpolation.eps', bbox_inches='tight')
	plt.close()


	if time_resolution_scan:
		if time_resolution_scan_improved:
			np.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'improved_row_composed_array',composed_array)
			# composed_array.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'improved_row_composed_array.nc')
		else:
			np.save(path_where_to_save_everything+'/merge'+str(merge_ID_target) + '_skip_of_' + str(time_resolution_extra_skip + 1) + 'row_composed_array',composed_array)
			#
			# composed_array.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_skip_of_' + str(time_resolution_extra_skip + 1) + 'row_composed_array.nc')
		ani = coleval.movie_from_data(np.array([composed_array.values]), 1 / conventional_time_step, row_shift,'Wavelength axis [pixles]', 'Row axis [pixles]', 'Intersity [au]')
		# ani.save(path_where_to_save_everything+'/merge' + str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_composed_array' + '.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
		ani.save(path_where_to_save_everything+'/merge' + str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_composed_array' + '.mp4', fps=5, writer='ffmpeg',codec='mpeg4')

	else:
		np.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_composed_array', composed_array)
		np.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_composed_array_sigma', composed_array_sigma)
		np.savez_compressed(path_where_to_save_everything+'/merge'+str(merge_ID_target)+ '_interpolation_borders_info',interpolation_borders_expanded_row=interpolation_borders_expanded_row,interpolation_borders_expanded_time=interpolation_borders_expanded_time,number_of_points_for_interpolation=number_of_points_for_interpolation)
		# np.save(path_where_to_save_everything+'/merge' + str(merge_ID_target) + '_interpolation_borders_expanded', interpolation_borders_expanded)

		# composed_array.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_composed_array.nc')
		# ani = coleval.movie_from_data(np.array([composed_array.values]), 1 / conventional_time_step, row_shift,'Wavelength axis [pixles]', 'Row axis [pixles]', 'Intersity [au]')
		# ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_composed_array' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

else:
	if time_resolution_scan:
		composed_array = np.load(path_where_to_save_everything+'/merge' + str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_composed_array.npy')
	else:
		composed_array = np.load(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_composed_array.npy')
		try:
			composed_array_sigma = np.load(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_composed_array_sigma.npy')
		except Exception as e:
			print('WARNING for merge '+str(merge_ID_target)+' composed_array_sigma absent')
			print(e)
			composed_array_sigma = np.ones_like(composed_array)*np.min(composed_array[composed_array>0])
	if timesteps_cropping_needed and (len(composed_array)>len(new_timesteps)):
		composed_array = composed_array[timesteps_limit_down:timesteps_limit_up]
		composed_array_sigma = composed_array_sigma[timesteps_limit_down:timesteps_limit_up]

ani = coleval.movie_from_data(np.array([composed_array]), 1000/conventional_time_step, row_shift, 'Wavelength axis [pixles]', 'Row axis [pixles]','Intersity [au]',extvmin=0)#,mask=mask)
# plt.show()
# ani.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_composed_array' + '.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
ani.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_composed_array' + '.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
plt.close()

ani = coleval.movie_from_data(np.array([composed_array/composed_array_sigma]), 1000/conventional_time_step, row_shift, 'Wavelength axis [pixles]', 'Row axis [pixles]','SNR [au]',extvmin=0)#,mask=mask)
# plt.show()
# ani.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_composed_array_SNR' + '.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
ani.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_composed_array_SNR' + '.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
plt.close()


# else:
# 	all_j=find_index_of_file(merge_ID_target,df_settings,df_log)
# 	j = all_j[-1]
# 	(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
# 	type = '.txt'
# 	filename_metadata = \
# 	all_file_names(fdir + '/' + folder + '/' + "{0:0=2g}".format(int(sequence)) + '/Untitled_' + str(int(untitled)) + '/Pos0',type)[0]
# 	(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir,folder,sequence,untitled,filename_metadata)
# 	(CB_to_OES_initial_delay,incremental_step,first_pulse_at_this_frame,bad_pulses_indexes,number_of_pulses) = df_log.loc[j,['CB_to_OES_initial_delay','incremental_step','first_pulse_at_this_frame','bad_pulses_indexes','number_of_pulses']]
# 	new_timesteps = np.linspace(merge_time_window[0]+0.5, merge_time_window[1]-0.5,int((merge_time_window[1]-0.5-(merge_time_window[0]+0.5))/conventional_time_step+1))
# 	wavelengths = np.linspace(0, roi_tr[0] - 1, roi_tr[0])
# 	row_steps = np.linspace(0, roi_tr[1] - 1, roi_tr[1])
# 	composed_array = xr.open_dataarray('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_composed_array.nc')
# 	ani = coleval.movie_from_data(np.array([composed_array.values]), 1000/conventional_time_step, row_shift, 'Wavelength axis [pixles]', 'Row axis [pixles]','Intersity [au]')
# 	# plt.show()
# 	ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_composed_array' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])


if (((not time_resolution_scan and not os.path.exists(path_where_to_save_everything +'/merge'+str(merge_ID_target)+'_gain_scaled_composed_array.npy')) or (time_resolution_scan and not os.path.exists(path_where_to_save_everything +'/merge'+str(merge_ID_target)+'_gain_scaled_composed_array.npy'))) or overwrite_everything[4]):
	if len(np.unique(merge_Gain))>1:
		print('there must be an error, multiple valies of gain detected')
		print(np.unique(merge_Gain))
		exit()
	scaled_composed_array=np.zeros_like(composed_array)
	scaled_composed_array_sigma=np.zeros_like(composed_array)
	neg_sig_corr_composed_array=np.zeros_like(composed_array)
	neg_sig_corr_composed_array_sigma=np.zeros_like(composed_array)
	additive_factor_list = np.zeros(([*(np.shape(composed_array)[:2])]))

	dataDark_calib = apply_proportionality_calibration(dataDark,x_calibration,y_calibration)

	Gain = np.unique(merge_Gain)[0]
	Noise = np.unique(merge_Noise)[0]
	print('Noise '+str(Noise))

	for index in range(len(composed_array)):
		print('frame '+str(index))
		if False:	# additive factor is now added at the merge_tot stage
			additive_factor,additive_factor_sigma = fix_minimum_signal_experiment(composed_array[index],intermediate_wavelength,last_wavelength,tilt_intermediate_column,tilt_last_column,counts_treshold_fixed_increase=108,dx_to_etrapolate_to=dx_to_etrapolate_to)
			additive_factor_list[index] = additive_factor
			if np.sum(np.isnan(additive_factor))>0:
				print('Correction and use aborted of frame '+str(index))
				continue
			data = (np.array(composed_array[index]).T + additive_factor).T
			data_sigma=np.sqrt(np.max([data-100,np.ones_like(data)],axis=0))
			# data_sigma[data_sigma<1]=1
			data_sigma = np.sqrt((data_sigma**2).T + (additive_factor_sigma**2)).T
			neg_sig_corr_composed_array[index]=(data)
			neg_sig_corr_composed_array_sigma[index]=(data_sigma)
		else:
			data = composed_array[index]
			data_sigma = composed_array_sigma[index]
		data_sigma_up = data+data_sigma
		data_sigma_down = data-data_sigma
		data = apply_proportionality_calibration(data,x_calibration,y_calibration)
		data_sigma_up = apply_proportionality_calibration(data_sigma_up,x_calibration,y_calibration)
		data_sigma_down = apply_proportionality_calibration(data_sigma_down,x_calibration,y_calibration)
		data = data - dataDark_calib
		data[data<0]=0
		data_sigma = (data_sigma_up-data_sigma_down)/2
		scaled_composed_array[index]=(data)
		scaled_composed_array_sigma[index]=(data_sigma)


	gain_scaled_composed_array = scaled_composed_array*Gain
	gain_scaled_composed_array_sigma = np.sqrt((scaled_composed_array_sigma*Gain)**2 + Noise**2)

	np.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_gain_scaled_composed_array', gain_scaled_composed_array)
	np.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_gain_scaled_composed_array_sigma', gain_scaled_composed_array_sigma)
	np.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_additive_factor_list', additive_factor_list)

	ani = coleval.movie_from_data(np.array([neg_sig_corr_composed_array]), 1000/conventional_time_step, row_shift, 'Wavelength axis [pixles]', 'Row axis [pixles]','Intersity [au]',extvmin=90,extvmax=130)#,mask=mask)
	# ani.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_neg_sig_corr_composed_array' + '.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
	ani.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_neg_sig_corr_composed_array' + '.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
	plt.close()
	ani = coleval.movie_from_data(np.array([scaled_composed_array]), 1000/conventional_time_step, row_shift, 'Wavelength axis [pixles]', 'Row axis [pixles]','Intersity [au]')#,mask=mask)
	# ani.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_scaled_composed_array' + '.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
	ani.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_scaled_composed_array' + '.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
	plt.close()


else:
	gain_scaled_composed_array = np.load(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_gain_scaled_composed_array.npy')
	try:
		gain_scaled_composed_array_sigma = np.load(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_gain_scaled_composed_array_sigma.npy')
	except Exception as e:
		print('WARNING for merge '+str(merge_ID_target)+' gain_scaled_composed_array_sigma absent')
		print(e)
		gain_scaled_composed_array_sigma = np.ones_like(gain_scaled_composed_array)*np.min(gain_scaled_composed_array[gain_scaled_composed_array>0])
	additive_factor_list = np.load(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_additive_factor_list.npy')
	if timesteps_cropping_needed and (len(gain_scaled_composed_array)>len(new_timesteps)):
		gain_scaled_composed_array = gain_scaled_composed_array[timesteps_limit_down:timesteps_limit_up]
		gain_scaled_composed_array_sigma = gain_scaled_composed_array_sigma[timesteps_limit_down:timesteps_limit_up]


plt.figure(figsize=(20, 10))
plt.imshow(additive_factor_list,'rainbow', origin='lower',extent=[0,roi_tr[1]-1,np.min(new_timesteps),np.max(new_timesteps)],aspect=100)
plt.colorbar().set_label('counts each row is shifted [au]')
plt.title('Record of the number of counts that each row is shifted')
# plt.title('Record of the row and times that required the interpolation borders expanded for good statistics')
plt.xlabel('Index of the row')
plt.ylabel('Time from pulse [ms]')
# plt.pause(0.001)
plt.savefig(path_where_to_save_everything+ '/' + 'additive_factor_list.eps', bbox_inches='tight')
plt.close()


ani = coleval.movie_from_data(np.array([gain_scaled_composed_array]), 1000/conventional_time_step, row_shift, 'Wavelength axis [pixles]', 'Row axis [pixles]','Intersity [au]',extvmin=0)#,mask=mask)
# ani.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_gain_scaled_composed_array' + '.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
ani.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_gain_scaled_composed_array' + '.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
plt.close()

ani = coleval.movie_from_data(np.array([gain_scaled_composed_array/gain_scaled_composed_array_sigma]), 1000/conventional_time_step, row_shift, 'Wavelength axis [pixles]', 'Row axis [pixles]','Signal/Noise [au]',extvmin=0)#,mask=mask)
# ani.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_SNR' + '.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
ani.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_SNR' + '.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
plt.close()

ani = coleval.movie_from_data(np.array([gain_scaled_composed_array_sigma]), 1000/conventional_time_step, row_shift, 'Wavelength axis [pixles]', 'Row axis [pixles]','Signal/Noise [au]',extvmin=0)#,mask=mask)
# ani.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_noise' + '.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
ani.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_noise' + '.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
plt.close()



if (((not time_resolution_scan and not os.path.exists(path_where_to_save_everything +'/merge'+str(merge_ID_target)+'_binned_data.npy')) or (time_resolution_scan and not os.path.exists(path_where_to_save_everything +'/merge'+str(merge_ID_target)+'_binned_data.npy'))) or overwrite_everything[2]):
# if True:
#	all_angles = []
#	for index in range(len(composed_array)):
#		result = 0
#		nLines = 8
#		while (result == 0 and nLines > 0):
#			try:
#				angle = get_angle(composed_array[index], nLines=nLines)
#				print(str(index)+' '+str(angle))
#				result = 1
#				all_angles.append( np.nansum(np.multiply(angle[0], np.divide(1, np.power(angle[1], 2)))) / np.nansum(np.divide(1, np.power(angle[1], 2))))
#			except:
#				nLines -= 1
#

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
	if timesteps_cropping_needed and (len(gain_scaled_composed_array_sigma)>len(new_timesteps)):
		binned_data = [*pool.map(calc_stuff, enumerate(old_timesteps))]
	else:
		binned_data = [*pool.map(calc_stuff, enumerate(new_timesteps))]
	pool.close()
	pool.join()
	pool.terminate()
	del pool
	# binned_data = set(binned_data)

	binned_data = list(binned_data)
	time_indexes = []
	for i in range(len(binned_data)):
		time_indexes.append(binned_data[i].time)
	binned_data = np.array([peaks for _, peaks in sorted(zip(time_indexes, binned_data))])
	temp = []
	for i in range(len(binned_data)):
		temp.append(binned_data[i].binnedData_sigma)
	binned_data_sigma = np.array(temp)
	temp = []
	for i in range(len(binned_data)):
		temp.append(binned_data[i].binnedData)
	binned_data = np.array(temp)

	if timesteps_cropping_needed and (len(binned_data)>len(new_timesteps)):
		binned_data = binned_data[timesteps_limit_down:timesteps_limit_up]
		binned_data_sigma = binned_data_sigma[timesteps_limit_down:timesteps_limit_up]

	#
	# binned_data = np.zeros((len(new_timesteps), 40, np.shape(frame)[1]))
	# for index,time in enumerate(new_timesteps):
	# 	frame = composed_array[index]
	# 	frame = np.nan_to_num(frame)
	# 	frame=rotate(frame,geom['angle'])
	# 	frame=four_point_transform(frame,geom['tilt'][0])
	# 	binnedData = binData(frame,geom['bin00b'],geom['binInterv'],check_overExp=False)
	# 	# binnedData = binnedData - min(np.min(binnedData), 0)
	# 	binned_data[index]=binnedData

	if time_resolution_scan:
		if time_resolution_scan_improved:
			np.save(path_where_to_save_everything+'/merge' + str(merge_ID_target) + '_skip_of_' + str(time_resolution_extra_skip + 1) + 'improved_row_binned_data',binned_data)
			# binned_data.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_skip_of_' + str(time_resolution_extra_skip + 1) + 'improved_row_binned_data.nc')
		else:
			np.save(path_where_to_save_everything+'/merge' + str(merge_ID_target) + '_skip_of_' + str(time_resolution_extra_skip + 1) + 'row_binned_data',binned_data)
			# binned_data.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_binned_data.nc')
		# ani = coleval.movie_from_data(np.array([binned_data.values]), 1000/conventional_time_step, row_shift, 'Wavelength axis [pixles]', 'Bin [au]','Intersity [au]')
		# ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_binned_data' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
	else:
		np.save(path_where_to_save_everything +'/merge'+str(merge_ID_target)+'_binned_data', binned_data)
		np.save(path_where_to_save_everything +'/merge'+str(merge_ID_target)+'_binned_data_sigma', binned_data_sigma)

		# binned_data.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_binned_data.nc')
		# ani = coleval.movie_from_data(np.array([binned_data.values]), 1000/conventional_time_step, row_shift, 'Wavelength axis [pixles]', 'Bin [au]','Intersity [au]')
		# ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_binned_data' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
else:
	if time_resolution_scan:
		binned_data = xr.open_dataarray(path_where_to_save_everything+'/merge' + str(merge_ID_target) + '_skip_of_' + str(time_resolution_extra_skip + 1) + 'row_binned_data.npy')
	else:
		binned_data = np.load(path_where_to_save_everything +'/merge'+str(merge_ID_target)+'_binned_data.npy')
		try:
			binned_data_sigma = np.load(path_where_to_save_everything +'/merge'+str(merge_ID_target)+'_binned_data_sigma.npy')
		except Exception as e:
			print('WARNING for merge '+str(merge_ID_target)+' binned_data_sigma absent')
			print(e)
			binned_data_sigma = np.ones_like(binned_data)*np.min(binned_data[binned_data>0])
	if timesteps_cropping_needed and (len(binned_data)>len(new_timesteps)):
		binned_data = binned_data[timesteps_limit_down:timesteps_limit_up]
		binned_data_sigma = binned_data_sigma[timesteps_limit_down:timesteps_limit_up]

#composed_array = np.zeros((roi_tr[1],roi_tr[1],roi_tr[0]))
#for i in range(np.shape(composed_array)[0]):
#	for j in range(np.shape(composed_array)[1]):
#		data_to_interpolate = merge.sel(time_row_to_start_pulse=slice(i*conventional_time_step-time_range_for_interp,i*conventional_time_step-time_range_for_interp),row=slice(i-rows_range_for_interp,i-rows_range_for_interp))

if (((not time_resolution_scan and not os.path.exists(path_where_to_save_everything +'/merge'+str(merge_ID_target)+'_all_fits.npy')) or (time_resolution_scan and not os.path.exists(path_where_to_save_everything +'/merge'+str(merge_ID_target)+'_all_fits.npy'))) or overwrite_everything[3]):

	# fit = doSpecFit_single_frame(np.array(binned_data[0]), df_settings, df_log, geom, waveLcoefs, binnedSens)

	if not os.path.exists(path_where_to_save_everything+'/spectral_fit_example'):
		os.makedirs(path_where_to_save_everything+'/spectral_fit_example')
	# else:
	# 	shutil.rmtree(path_where_to_save_everything+'/spectral_fit_example')
	# 	os.makedirs(path_where_to_save_everything+'/spectral_fit_example')

	sample_time_step = []
	for time in [-0.2,0.05,0.15,0.2,0.25,0.5,0.55,0.6,1,*(new_timesteps[np.max(binned_data,axis=(-1,-2)).argmax()]+np.array([-0.05,0,0.05]))]:	#times in ms that I want to take a better look at
		sample_time_step.append(int((np.abs(new_timesteps - time)).argmin()))
	sample_time_step = np.unique(sample_time_step)

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
		if index in sample_time_step:
			to_print = True
		print('fit of ' + str(time) + 'ms')
		try:
			fit,fit_sigma = doSpecFit_single_frame_with_sigma(np.array(binned_data[index]),np.array(binned_data_sigma[index]),df_settings, df_log, geom, waveLcoefs, binnedSens,binnedSens_sigma,path_where_to_save_everything,time,to_print,perform_convolution=perform_convolution)
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
	if timesteps_cropping_needed and (len(binned_data)>len(new_timesteps)):
		all_fits = [*pool.map(calc_stuff, enumerate(old_timesteps))]
	else:
		all_fits = [*pool.map(calc_stuff, enumerate(new_timesteps))]
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

	if timesteps_cropping_needed and (len(all_fits)>len(new_timesteps)):
		all_fits = all_fits[timesteps_limit_down:timesteps_limit_up]
		all_fits_sigma = all_fits_sigma[timesteps_limit_down:timesteps_limit_up]

	# all_fits = []
	# for index,time in enumerate(new_timesteps):
	# 	print('fit of '+str(time)+'ms')
	# 	try:
	# 		fit = doSpecFit_single_frame(np.array(binned_data[index]),df_settings, df_log, geom, waveLcoefs, binnedSens)
	# 		print('ok')
	# 	except:
	# 		fit=np.zeros((40,9))*np.nan
	# 	all_fits.append(np.array(fit))
	# all_fits=np.array(all_fits).astype(float)

	if time_resolution_scan:
		if time_resolution_scan_improved:
			np.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'improved_row_all_fits',all_fits)
		else:
			np.save(path_where_to_save_everything+'/merge' + str(merge_ID_target) + '_skip_of_' + str(time_resolution_extra_skip + 1) + 'row_all_fits', all_fits)
		ani = coleval.movie_from_data(np.array([all_fits]), 1000/conventional_time_step, row_shift, 'Transition from Hb' , 'Bin [au]' , 'Intersity [au]',extvmin=0,extvmax=100000)
		# ani.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_all_fits' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
		ani.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_all_fits' + '.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
	else:
		np.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_all_fits',all_fits)
		np.save(path_where_to_save_everything+'/merge'+str(merge_ID_target)+'_all_fits_sigma',all_fits_sigma)
		# ani = coleval.movie_from_data(np.array([all_fits]), 1000/conventional_time_step, row_shift, 'Transition from Hb' , 'Bin [au]' , 'Intersity [au]',extvmin=0,extvmax=100000)
		# ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_all_fits' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])


else:
	if time_resolution_scan:
		all_fits = xr.open_dataarray(path_where_to_save_everything+'/merge' + str(merge_ID_target) + '_skip_of_' + str(time_resolution_extra_skip + 1) + 'row_all_fits.npy')
	else:
		all_fits = np.load(path_where_to_save_everything +'/merge'+str(merge_ID_target)+'_all_fits.npy')
		try:
			all_fits_sigma = np.load(path_where_to_save_everything +'/merge'+str(merge_ID_target)+'_all_fits_sigma.npy')
		except Exception as e:
			print('WARNING for merge '+str(merge_ID_target)+' all_fits_sigma absent')
			print(e)
			all_fits_sigma = np.ones_like(all_fits)*np.min(all_fits[all_fits>0])
	if timesteps_cropping_needed and (len(all_fits)>len(new_timesteps)):
		all_fits = all_fits[timesteps_limit_down:timesteps_limit_up]
		all_fits_sigma = all_fits_sigma[timesteps_limit_down:timesteps_limit_up]

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
