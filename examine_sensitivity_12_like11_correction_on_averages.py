import numpy as np
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())
#import .functions
import os,sys

from numpy.core.multiarray import ndarray

os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
from functions.spectools import get_angle,rotate,get_tilt,do_tilt,getFirstBin, binData, get_angle_2,binData_with_sigma,get_line_position,get_4_points
from functions.fabio_add import find_nearest_index,multi_gaussian,all_file_names,load_dark,find_index_of_file,get_metadata,examine_current_trace,movie_from_data,get_bin_and_interv_no_lines, four_point_transform, fix_minimum_signal, do_tilt_no_lines, fix_minimum_signal2,apply_proportionality_calibration,fix_minimum_signal_calibration,get_bin_and_interv_specific_wavelength,fix_minimum_signal_experiment
from functions.Calibrate import do_waveL_Calib,do_waveL_Calib_simplified
from PIL import Image
import xarray as xr
import pandas as pd
import copy
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.signal import gaussian
from skimage.transform import resize

exec(open("/home/ffederic/work/analysis_scripts/scripts/profile_smoothing.py").read())

n = np.arange(10,20)
waveLengths = [486.13615,434.0462,410.174,397.0072,388.9049,383.5384] + list((364.6*(n*n)/(n*n-4)))
# waveLengths = [486.13615,434.0462,410.174,397.0072,388.9049,383.5384]
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


# proportionality calibration from examine_sensitivity_10

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
     0.        ,   0.        ,   0.01719774,   0.05635798,
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
     99.39393939,   99.7979798 ,  100.2020202 ,  100.60606061,
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





#####	THIS BIT IS FOR THE EXPERIMENTS 66 onwards

limits_angle = [66, 70]
limits_tilt = [66, 70]
limits_wave = [66, 70]
figure_number = 0
where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_12/Sensitivity_4_plots_'

geom_null = pd.DataFrame(columns=['angle', 'tilt', 'binInterv', 'bin00a', 'bin00b'])
geom_null.loc[0] = [0, 0, 0, 0, 0]

full_list = [limits_angle,limits_tilt,limits_wave]
full_list = np.array(full_list).flatten()
full_list = full_list[full_list>0]
data_sum=0
for merge in range(int(np.min(full_list)),int(np.max(full_list)) + 1):
	all_j = find_index_of_file(merge, df_settings, df_log)
	for j in all_j:
		(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
		type = '.tif'
		filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)

		# plt.figure()
		data_all = []
		for index, filename in enumerate(filenames):
			fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(
				untitled) + '/Pos0/' + \
					filename
			im = Image.open(fname)
			data = np.array(im)
			data_all.append(data)
		# data_all = np.array(data_all)
		data_sum += np.mean(data_all,axis=0)
intermediate_wavelength,last_wavelength = get_line_position(data_sum,2)
tilt_intermediate_column = get_bin_and_interv_specific_wavelength(fix_minimum_signal2(data_sum),intermediate_wavelength)
tilt_last_column = get_bin_and_interv_specific_wavelength(fix_minimum_signal2(data_sum),last_wavelength)
dx_to_etrapolate_to=140


if np.sum(limits_angle) != 0:
	angles = []
	angles_scores = []
	for merge in range(limits_angle[0], limits_angle[1] + 1):
		all_j = find_index_of_file(merge, df_settings, df_log)
		for j in all_j:
			# dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
			(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
			type = '.tif'
			filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
			# type = '.txt'
			# filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)[0]
			# (bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain,Noise) = get_metadata(fdir, folder,sequence,untitled,filename_metadata)

			# plt.figure()
			data_all = []
			for index, filename in enumerate(filenames):
				fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
				im = Image.open(fname)
				data = np.array(im)
				# data_sigma=np.sqrt(data)
				additive_factor,additive_factor_sigma = fix_minimum_signal_experiment(data,intermediate_wavelength,last_wavelength,tilt_intermediate_column,tilt_last_column,counts_treshold_fixed_increase=106,dx_to_etrapolate_to=dx_to_etrapolate_to)
				data = (data.T + additive_factor).T
				# data_sigma = np.sqrt((data_sigma**2).T + (additive_factor_sigma**2)).T
				# data = fix_minimum_signal2(data)
				# data = (data - dataDark) * Gain[index]
				data_all.append(data)
			# data_all = np.array(data_all)
			data_mean = np.mean(data_all, axis=0)
			nLines=9
			done=0
			while done==0 and nLines>=2:
				try:
					temp = get_angle_2(data_mean, nLines=nLines)
					angles.append(np.array(temp[0]))
					angles_scores.append(np.array(temp[1]))
					print(temp)
					done=1
				except:
					nLines-=1
					print('FAILED')
	angles = np.concatenate(angles)
	angles_scores = np.concatenate(angles_scores)
	print('angles')
	print(angles)
	print('angles_scores')
	print(angles_scores)
	angle = np.nansum(angles / (angles_scores ** 2)) / np.nansum(1 / angles_scores ** 2)

# if np.sum(limits_tilt) != 0:
# 	tilt_4_points = []
# 	for merge in range(limits_tilt[0], limits_tilt[1] + 1):
# 		all_j = find_index_of_file(merge, df_settings, df_log)
# 		for j in all_j:
# 			# dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
# 			(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
# 			type = '.tif'
# 			filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
# 			# type = '.txt'
# 			# filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)[0]
# 			# (bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain,Noise) = get_metadata(fdir, folder,sequence,untitled,filename_metadata)
#
# 			# plt.figure()
# 			data_all = []
# 			for index, filename in enumerate(filenames):
# 				fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(
# 					untitled) + '/Pos0/' + \
# 						filename
# 				im = Image.open(fname)
# 				data = np.array(im)
# 				# data = (data - dataDark) * Gain[index]
# 				# data_sigma=np.sqrt(data)
# 				additive_factor,additive_factor_sigma = fix_minimum_signal_experiment(data,intermediate_wavelength,last_wavelength,tilt_intermediate_column,tilt_last_column,counts_treshold_fixed_increase=106,dx_to_etrapolate_to=dx_to_etrapolate_to)
# 				data = (data.T + additive_factor).T
# 				# data_sigma = np.sqrt((data_sigma**2).T + (additive_factor_sigma**2)).T
# 				# data = fix_minimum_signal2(data)
# 				data_all.append(data)
# 			# data_all = np.array(data_all)
# 			data_mean = np.mean(data_all, axis=0)
# 			data_mean = rotate(data_mean, angle)
# 			nLines=9
# 			done=0
# 			while done==0 and nLines>=2:
# 				try:
# 					temp = get_4_points(data_mean, nLines=nLines)
# 					tilt_4_points.append(np.array(temp))
# 					print(temp)
# 					done=1
# 				except:
# 					nLines-=1
# 					print('FAILED')
# 	tilt_4_points = np.array(tilt_4_points)
# 	print(tilt_4_points)
# 	tilt_4_points = np.nanmedian(tilt_4_points, axis=0)
#
# if np.sum(limits_tilt) != 0:
# 	tilt = []
# 	for merge in range(limits_tilt[0], limits_tilt[1] + 1):
# 		all_j = find_index_of_file(merge, df_settings, df_log)
# 		for j in all_j:
# 			# dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
# 			(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
# 			type = '.tif'
# 			filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
# 			# type = '.txt'
# 			# filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)[0]
# 			# (bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain,Noise) = get_metadata(fdir, folder,sequence,untitled,filename_metadata)
#
# 			# plt.figure()
# 			data_all = []
# 			for index, filename in enumerate(filenames):
# 				fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(
# 					untitled) + '/Pos0/' + \
# 						filename
# 				im = Image.open(fname)
# 				data = np.array(im)
# 				# data = (data - dataDark) * Gain[index]
# 				# data_sigma=np.sqrt(data)
# 				additive_factor,additive_factor_sigma = fix_minimum_signal_experiment(data,intermediate_wavelength,last_wavelength,tilt_intermediate_column,tilt_last_column,counts_treshold_fixed_increase=106,dx_to_etrapolate_to=dx_to_etrapolate_to)
# 				data = (data.T + additive_factor).T
# 				# data_sigma = np.sqrt((data_sigma**2).T + (additive_factor_sigma**2)).T
# 				# data = fix_minimum_signal2(data)
# 				data_all.append(data)
# 			# data_all = np.array(data_all)
# 			data_mean = np.mean(data_all, axis=0)
# 			data_mean = rotate(data_mean, angle)
# 			data_mean = four_point_transform(data_mean, tilt_4_points)
# 			nLines=9
# 			done=0
# 			while done==0 and nLines>=2:
# 				try:
# 					temp = get_tilt(data_mean, nLines=nLines)
# 					tilt.append(np.array(temp))
# 					print(temp)
# 					done=1
# 				except:
# 					nLines-=1
# 					print('FAILED')
# 	tilt = np.array(tilt)
# 	print(tilt)
# 	tilt = np.nanmedian(tilt, axis=0)

geom = pd.DataFrame(columns=['angle', 'tilt', 'binInterv', 'bin00a', 'bin00b'])
# geom.loc[0] = [angle, np.array(tilt_4_points), tilt[0], tilt[2], tilt[2]]
geom.loc[0] = [angle, 0, 0, 0, 0]
geom_store = copy.deepcopy(geom)
print(geom)

if np.sum(limits_wave) != 0:
	waveLcoefs = []
	for merge in range(limits_wave[0], limits_wave[1] + 1):
		all_j = find_index_of_file(merge, df_settings, df_log)
		for j in all_j:
			# dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
			(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
			type = '.tif'
			filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
			# type = '.txt'
			# filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)[0]
			# (bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain,Noise) = get_metadata(fdir, folder,sequence,untitled,filename_metadata)

			# plt.figure()
			data_all = []
			for index, filename in enumerate(filenames):
				fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(
					untitled) + '/Pos0/' + \
						filename
				im = Image.open(fname)
				data = np.array(im)
				# data = (data - dataDark) * Gain[index]
				# data_sigma=np.sqrt(data)
				additive_factor,additive_factor_sigma = fix_minimum_signal_experiment(data,intermediate_wavelength,last_wavelength,tilt_intermediate_column,tilt_last_column,counts_treshold_fixed_increase=106,dx_to_etrapolate_to=dx_to_etrapolate_to)
				data = (data.T + additive_factor).T
				# data_sigma = np.sqrt((data_sigma**2).T + (additive_factor_sigma**2)).T
				# data = fix_minimum_signal2(data)
				data_all.append(data)
			# data_all = np.array(data_all)
			data_mean = np.mean(data_all, axis=0)
			data_mean = rotate(data_mean, angle)
			nLines=9
			done=0
			while done==0 and nLines>=2:
				try:
					temp = get_4_points(data_mean, nLines=nLines)
					tilt_4_points=np.array(temp)
					print(temp)
					data_mean = four_point_transform(data_mean, tilt_4_points)
					nLines2=9
					done2=0
					while done2==0 and nLines2>=2:
						try:
							temp = get_tilt(data_mean, nLines=nLines2)
							tilt=np.array(temp)
							print(temp)
							if not np.isnan(temp[0]):
								done2=1
							else:
								nLines2-=1
								print('FAILED')
						except:
							nLines2-=1
							print('FAILED')
					binnedData,trash = binData_with_sigma(data_mean,np.ones_like(data_mean),tilt[2],tilt[0],check_overExp=False)
					waveLcoefs.append(np.array(do_waveL_Calib_simplified(binnedData)))
					done=1
				except:
					nLines-=1
					print('FAILED')
	waveLcoefs = np.array(waveLcoefs)
	print(waveLcoefs)
	waveLcoefs = np.nanmedian(waveLcoefs, axis=0)
	print('waveLcoefs')
	print(waveLcoefs)


# geom = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
# geom.loc[0] = [ 0.808817942408, np.nan, np.nan, np.nan, np.nan]
# waveLcoefs = np.ones((2, 3)) * np.nan
# waveLcoefs[1] = [ 2.84709039e-06 ,  1.19975728e-01 ,  3.18673899e+02]
# geom_null = pd.DataFrame(columns=['angle', 'tilt', 'binInterv', 'bin00a', 'bin00b'])
# geom_null.loc[0] = [0, 0, 0, 0, 0]

int_time_long = [0.1, 1, 10, 100]
calib_ID_long=[215, 216, 217, 218]
# int_time_long = [0.1, 1, 10, 100, 200, 500, 100]
# calib_ID_long=[215, 216, 217, 413, 414, 415, 218]	# I can't really compare 2xx and 4xx because I don't have a wavelength axis for 4xx
calib_ID_long = np.array([calib_ID_long for _, calib_ID_long in sorted(zip(int_time_long, calib_ID_long))])
int_time_long = np.sort(int_time_long)
int_time_long_new = np.flip(int_time_long,axis=0)
calib_ID_long_new = np.flip(calib_ID_long,axis=0)

intermediate_wavelength=1200
last_wavelength=1608
dx_to_etrapolate_to=140

int_time = [100]
data_all = []
calib_ID=[218]
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
		# data = (data - dataDark)
		# data = fix_minimum_signal2(data)

		temp.append(data)
	temp = np.mean(temp, axis=0)
	data = copy.deepcopy(temp)
	data = data/(df_log.loc[j, ['Calib_ID']][0]*1000000)
	data_all.append(data)

data_all = np.array(data_all)
data=np.mean(data_all,axis=(0))
data = fix_minimum_signal2(data)

tilt_intermediate_column = get_bin_and_interv_specific_wavelength(fix_minimum_signal2(data),intermediate_wavelength)
tilt_last_column = get_bin_and_interv_specific_wavelength(fix_minimum_signal2(data),last_wavelength)

data = rotate(data, geom['angle'])
data = do_tilt_no_lines(data)
data=np.sum(data,axis=0)
data=np.convolve(data, np.ones(200)/200 , mode='valid')
wavel=data.argmax()+100

calculate_sigma_from_std=True

wavel_range = 100
# wavelength_780=1266
# wavelength_750=1556

# wavel = 1338
wavelength_950=wavel
target_up = np.polyval(waveLcoefs[1], wavelength_950+wavel_range)
target_down = np.polyval(waveLcoefs[1], wavelength_950-wavel_range)


bin_interv_high_int_time = []
first_bin_high_int_time = []
tilt_4_points_high_int_time = []
data_all = []
binned_all = []
peak_sum = []
longCal_all=[]
data_sigma_all = []
binned_sigma_all = []
longCal_sigma_all=[]
for iFile, j in enumerate(calib_ID_long):
	(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]

	dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
	additive_factor,additive_factor_sigma = fix_minimum_signal_calibration(dataDark,counts_treshold_fixed_increase=106.5,intermediate_wavelength=intermediate_wavelength,last_wavelength=last_wavelength,tilt_intermediate_column=tilt_intermediate_column,tilt_last_column=tilt_last_column,dx_to_etrapolate_to=dx_to_etrapolate_to)
	dataDark = (np.array(dataDark).T + additive_factor).T
	dataDark = apply_proportionality_calibration(dataDark,x_calibration,y_calibration)
	plt.figure()
	plt.imshow(dataDark,'rainbow',origin='lower')
	plt.title('Exp '+str(limits_angle)+' , '+str(j)+'int time sensitivity\nDark measurement used, negative and proportionality corrected')
	plt.xlabel('wavelength axis [pixels]')
	plt.ylabel('LOS axis [pixels]')
	plt.colorbar().set_label('counts [au]')
	figure_number+=1
	plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
	plt.close()

	type = '.tif'
	filenames = all_file_names(
		fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
	type = '.txt'
	filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0',type)[0]
	(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain,Noise) = get_metadata(fdir,folder,sequence,untitled,filename_metadata)

	temp = []
	temp_sigma = []

	for index, filename in enumerate(filenames):
		fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(
			untitled) + '/Pos0/' + filename
		# print(fname)
		im = Image.open(fname)
		data = np.array(im)
		# print(filename)
		if index==0:
			additive_factor,additive_factor_sigma = fix_minimum_signal_calibration(data,counts_treshold_fixed_increase=106.5,intermediate_wavelength=intermediate_wavelength,last_wavelength=last_wavelength,tilt_intermediate_column=tilt_intermediate_column,tilt_last_column=tilt_last_column,dx_to_etrapolate_to=dx_to_etrapolate_to)
			plt.figure(figsize=(20, 10))
			plt.fill_between(np.arange(len(additive_factor)), additive_factor-additive_factor_sigma,additive_factor+additive_factor_sigma,additive_factor-additive_factor_sigma<additive_factor+additive_factor_sigma,color='y',alpha=0.1)
			plt.plot(additive_factor,'r',label='additive factor')
			plt.plot(np.mean(data[:, :100],axis=-1)-np.min(np.mean(data[:, :100],axis=-1)),'b',label='proxy for minimum signal in the row, not corrected')
			for i in range(41):
				plt.plot([tilt_last_column[0]+i*tilt_last_column[1],tilt_last_column[0]+i*tilt_last_column[1]],[0,7],'--k',linestyle=(0, (5, 4, 1, 4)),linewidth=1)
			plt.grid()
			plt.title('Exp '+str(limits_angle)+' , '+str(j)+'int time sensitivity\n'+'Check to find the right dx_to_etrapolate_to to use in fix_minimum_signal_calibration\nhere '+str(dx_to_etrapolate_to)+' is used\n single frame')
			plt.xlabel('row axis')
			plt.ylabel('counts [au]')
			plt.legend(loc='best')
			plt.ylim(0,7)
			plt.xlim(tilt_last_column[0]-tilt_last_column[1],tilt_last_column[0]+tilt_last_column[1]*41)
			figure_number+=1
			plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
			plt.close()
		# if np.sum(np.isnan(additive_factor))>0:
		# 	print(filename+' correction and use aborted')
		# 	continue
		# data = (np.array(data).T + additive_factor).T
		# data_sigma=np.sqrt(np.max([data-100,np.ones_like(data)],axis=0))
		# data_sigma = np.sqrt((data_sigma**2).T + (additive_factor_sigma**2)).T
		# data_sigma_up = data+data_sigma
		# data_sigma_down = data-data_sigma
		# data_sigma_up = apply_proportionality_calibration(data_sigma_up,x_calibration,y_calibration)
		# data_sigma_down = apply_proportionality_calibration(data_sigma_down,x_calibration,y_calibration)
		# data_sigma = (data_sigma_up-data_sigma_down)/2
		# data_sigma = np.sqrt((data_sigma*Gain[index])**2 + Noise[index]**2)
		#
		# data = apply_proportionality_calibration(data,x_calibration,y_calibration)
		# data = (data-dataDark)*Gain[index]

		temp.append(data)
		# temp_sigma.append(data_sigma)

	temp_sigma = np.std(temp,axis=0)
	temp = np.mean(temp, axis=0)

	additive_factor,additive_factor_sigma = fix_minimum_signal_calibration(temp,counts_treshold_fixed_increase=106.5,intermediate_wavelength=intermediate_wavelength,last_wavelength=last_wavelength,tilt_intermediate_column=tilt_intermediate_column,tilt_last_column=tilt_last_column,dx_to_etrapolate_to=dx_to_etrapolate_to)

	plt.figure(figsize=(20, 10))
	plt.fill_between(np.arange(len(additive_factor)), additive_factor-additive_factor_sigma,additive_factor+additive_factor_sigma,additive_factor-additive_factor_sigma<additive_factor+additive_factor_sigma,color='y',alpha=0.1)
	plt.plot(additive_factor,'r',label='additive factor')
	plt.plot(np.mean(temp[:, :100],axis=-1)-np.min(np.mean(temp[:, :100],axis=-1)),'b',label='proxy for minimum signal in the row, not corrected')
	for i in range(41):
		plt.plot([tilt_last_column[0]+i*tilt_last_column[1],tilt_last_column[0]+i*tilt_last_column[1]],[0,7],'--k',linestyle=(0, (5, 4, 1, 4)),linewidth=1)
	plt.grid()
	plt.title('Exp '+str(limits_angle)+' , '+str(j)+'int time sensitivity\n'+'Check to find the right dx_to_etrapolate_to to use in fix_minimum_signal_calibration\nhere '+str(dx_to_etrapolate_to)+' is used\n averaged frames')
	plt.xlabel('row axis')
	plt.ylabel('counts [au]')
	plt.legend(loc='best')
	plt.ylim(0,7)
	plt.xlim(tilt_last_column[0]-tilt_last_column[1],tilt_last_column[0]+tilt_last_column[1]*41)
	figure_number+=1
	plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
	plt.close()

	temp = (np.array(temp).T + additive_factor).T
	# temp_sigma=np.sqrt(np.max([temp-100,np.ones_like(temp)],axis=0))
	temp_sigma = np.sqrt((temp_sigma**2).T + (additive_factor_sigma**2)).T
	temp_sigma_up = temp+temp_sigma
	temp_sigma_down = temp-temp_sigma
	temp_sigma_up = apply_proportionality_calibration(temp_sigma_up,x_calibration,y_calibration)
	temp_sigma_down = apply_proportionality_calibration(temp_sigma_down,x_calibration,y_calibration)
	temp_sigma = (temp_sigma_up-temp_sigma_down)/2
	temp_sigma = np.sqrt((temp_sigma*Gain[0])**2 + Noise[0]**2)

	temp = apply_proportionality_calibration(temp,x_calibration,y_calibration)
	temp = (temp-dataDark)*Gain[0]

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
	tilt_4_points = do_tilt_no_lines(temp,return_4_points=True)
	temp = do_tilt_no_lines(temp)
	tilt_4_points_high_int_time.append(tilt_4_points)
	temp_sigma= rotate(temp_sigma, geom['angle'])
	temp_sigma = four_point_transform(temp_sigma, tilt_4_points)

	first_bin, bin_interv = get_bin_and_interv_no_lines(temp)
	print('bin interval of ' + str(bin_interv) + ' , first_bin of ' + str(first_bin))
	first_bin_high_int_time.append(first_bin)
	bin_interv_high_int_time.append(bin_interv)
	binned_data,binned_data_sigma = binData_with_sigma(temp,temp_sigma, first_bin, bin_interv)/(df_log.loc[j, ['Calib_ID']][0]*1000000)
	temp1,temp2 = binData_with_sigma(temp, temp_sigma, first_bin - bin_interv, bin_interv, nCh=42)/(df_log.loc[j, ['Calib_ID']][0]*1000000)
	longCal_all.append(temp1)
	longCal_sigma_all.append(temp2)
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
	data_sigma_all.append(temp_sigma)
	binned_all.append(binned_data)
	binned_sigma_all.append(binned_data_sigma)

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
		temp_1=data_sigma_all[index].tolist()
		temp_1.append(np.zeros(np.shape(temp_1[0])))
		data_sigma_all[index] = np.array(temp_1)

	first_bin, bin_interv = get_bin_and_interv_no_lines(data_all[index])
	waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(data_all[index])[1]+1))
	temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(data_all[index])[0]+1))
	plt.figure(figsize=(20, 10))
	plt.pcolor(temp_w,temp_LOS,data_all[index],cmap='rainbow',vmax=np.max(data_all[index]),rasterized=True)
	plt.colorbar()
	for i in range(len(waveLengths)):
		plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(data_all[index])[0]-1],'--k',linewidth=0.5)
	for i in range(41):
		plt.plot([np.min(waveaxis),np.max(waveaxis)],[first_bin+i*bin_interv,first_bin+i*bin_interv],'--k',linewidth=0.5)
	plt.title('Exp ' + str(limits_angle) + '\n '+str(calib_ID_long[index])+'int time sensitivity')
	plt.xlabel('wavelength [nm]')
	plt.ylabel('LOS axis [pixels]')
	figure_number+=1
	plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.pcolor(temp_w,temp_LOS,data_all[index],cmap='rainbow',vmax=np.max(data_all[index][:,600]),rasterized=True)
	plt.colorbar()
	for i in range(len(waveLengths)):
		plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(data_all[index])[0]-1],'--k',linewidth=0.5)
	for i in range(41):
		plt.plot([np.min(waveaxis),np.max(waveaxis)],[first_bin+i*bin_interv,first_bin+i*bin_interv],'--k',linewidth=0.5)
	plt.title('Exp ' + str(limits_angle) + '\n '+str(calib_ID_long[index])+'int time sensitivity low wavelengths')
	plt.xlabel('wavelength [nm]')
	plt.ylabel('LOS axis [pixels]')
	figure_number+=1
	plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
	plt.close()


	# plt.figure()
	# plt.imshow(data_all[index], 'rainbow', origin='lower')
	# first_bin, bin_interv = get_bin_and_interv_no_lines(data_all[index])
	# for i in range(41):
	# 	plt.plot([0,np.shape(data_all[index])[-1]],[first_bin+i*bin_interv,first_bin+i*bin_interv],'--k',linestyle=(0, (5, 4, 1, 4)),linewidth=0.5)
	# plt.title('Exp ' + str(limits_angle) + '\n '+str(calib_ID_long[index])+'int time sensitivity')
	# plt.xlabel('wavelength axis [pixels]')
	# plt.ylabel('LOS axis [pixels]')
	# plt.colorbar().set_label('counts [au]')
	# figure_number += 1
	# plt.savefig(where_to_save_everything + str(figure_number) + '.eps', bbox_inches='tight')
	# plt.close()

	waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(data_sigma_all[index])[1]+1))
	temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(data_sigma_all[index])[0]+1))
	plt.figure(figsize=(20, 10))
	plt.pcolor(temp_w,temp_LOS,data_sigma_all[index],cmap='rainbow',vmax=np.max(data_sigma_all[index]),rasterized=True)
	plt.colorbar()
	for i in range(len(waveLengths)):
		plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(data_sigma_all[index])[0]-1],'--k',linewidth=0.5)
	for i in range(41):
		plt.plot([np.min(waveaxis),np.max(waveaxis)],[first_bin+i*bin_interv,first_bin+i*bin_interv],'--k',linewidth=0.5)
	plt.title('Exp ' + str(limits_angle) + '\n '+str(calib_ID_long[index])+'int time sensitivity error')
	plt.xlabel('wavelength [nm]')
	plt.ylabel('LOS axis [pixels]')
	figure_number+=1
	plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
	plt.close()


	# plt.figure()
	# plt.imshow(data_sigma_all[index], 'rainbow', origin='lower')
	# for i in range(41):
	# 	plt.plot([0,np.shape(data_sigma_all[index])[-1]],[first_bin+i*bin_interv,first_bin+i*bin_interv],'--k',linestyle=(0, (5, 4, 1, 4)),linewidth=0.5)
	# plt.title('Exp ' + str(limits_angle) + '\n '+str(calib_ID_long[index])+'int time sensitivity error')
	# plt.xlabel('wavelength axis [pixels]')
	# plt.ylabel('LOS axis [pixels]')
	# plt.colorbar().set_label('counts [au]')
	# figure_number += 1
	# plt.savefig(where_to_save_everything + str(figure_number) + '.eps', bbox_inches='tight')
	# plt.close()
# plt.pause(0.01)
data_all_long_int_time=np.array(data_all)
binned_all=np.array(binned_all)
if False:	# the higher int time should give best results
	bin_interv_high_int_time = np.nanmean(bin_interv_high_int_time)
	first_bin_high_int_time = np.nanmean(first_bin_high_int_time)
	tilt_4_points_high_int_time = np.nanmean(tilt_4_points_high_int_time,axis=0)
else:
	bin_interv_high_int_time = bin_interv_high_int_time[-1]
	first_bin_high_int_time = first_bin_high_int_time[-1]
	tilt_4_points_high_int_time = tilt_4_points_high_int_time[-1]
data_sigma_all_long_int_time=np.array(data_sigma_all)
binned_sigma_all=np.array(binned_sigma_all)


calBinned_peak_sum=[]
calBinned_all=[]
calBinned_sigma_all=[]
longCal_all=np.array(longCal_all)
longCal_sigma_all=np.array(longCal_sigma_all)
for i in range(len(binned_all)):
	topDark = longCal_all[i][0]
	botDark = longCal_all[i][-1]
	wtDark = np.arange(1, 41) / 41
	calBinned = longCal_all[i][1:41] - (np.outer(wtDark, topDark) + np.outer(wtDark[::-1], botDark))	# + (topDark + botDark) / 2  # MODIFIED TO AVOID NEGATIVE VALUES
	# I do this to eliminate the negative values and relpace them with neighbouring ones
	for iLine in range(len(calBinned)):
		while np.sum(calBinned[iLine]<=0)>0:
			select = calBinned[iLine]<=0
			positive_values = calBinned[iLine][calBinned[iLine]>0]
			positive_indexes = np.arange(np.shape(calBinned)[1])[calBinned[iLine]>0]
			for iwave in (np.arange(np.shape(calBinned)[1])[select]):
				calBinned[iLine,iwave] = positive_values[np.abs(positive_indexes-iwave).argmin()]
	topDark_sigma = longCal_sigma_all[i][0]
	botDark_sigma = longCal_sigma_all[i][-1]
	calBinned_sigma = np.sqrt((longCal_sigma_all[i][1:41])**2 + (np.outer(wtDark, topDark_sigma))**2 + (np.outer(wtDark[::-1], botDark_sigma))**2 + (topDark_sigma**2 + botDark_sigma**2) / 4 ) # MODIFIED TO AVOID NEGATIVE VALUES
	# calBinned_peak_sum.append(np.sum(calBinned[:,wavel - wavel_range:wavel + wavel_range],axis=(-1,-2)))
	calBinned_peak_sum.append(np.sum(calBinned[:, wavel - wavel_range:wavel + wavel_range], axis=(-1)))
	calBinned_all.append(calBinned)
	calBinned_sigma_all.append(calBinned_sigma)
calBinned_peak_sum = np.array(calBinned_peak_sum)
calBinned_all=np.array(calBinned_all)
calBinned_sigma_all=np.array(calBinned_sigma_all)


int_time = [0.01]
data_all = []
data_sigma_all = []
data_all_for_sigma = []
data_all_non_proportional = []
calib_ID=[214]
for iFile, j in enumerate(calib_ID):
	(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]

	dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
	additive_factor,additive_factor_sigma = fix_minimum_signal_calibration(dataDark,counts_treshold_fixed_increase=106.5,intermediate_wavelength=intermediate_wavelength,last_wavelength=last_wavelength,tilt_intermediate_column=tilt_intermediate_column,tilt_last_column=tilt_last_column,dx_to_etrapolate_to=dx_to_etrapolate_to)
	dataDark_non_proportional = (np.array(dataDark).T + additive_factor).T
	dataDark = apply_proportionality_calibration(dataDark,x_calibration,y_calibration)
	plt.figure()
	plt.imshow(dataDark,'rainbow',origin='lower')
	plt.title('Exp '+str(limits_angle)+' , '+str(j)+'int time sensitivity\nDark measurement used, negative and proportionality corrected')
	plt.xlabel('wavelength axis [pixels]')
	plt.ylabel('LOS axis [pixels]')
	plt.colorbar().set_label('counts [au]')
	figure_number+=1
	plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
	plt.close()

	type = '.tif'
	filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
	type = '.txt'
	filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0',type)[0]
	(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain,Noise) = get_metadata(fdir,folder,sequence,untitled,filename_metadata)

	temp = []
	temp1 = []
	temp_sigma = []
	temp1_sigma = []
	for index, filename in enumerate(filenames):
		fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
		# print(fname)
		im = Image.open(fname)
		data = np.array(im)
		# print(filename)
		if index==0:
			additive_factor,additive_factor_sigma = fix_minimum_signal_calibration(data,counts_treshold_fixed_increase=106.5,intermediate_wavelength=intermediate_wavelength,last_wavelength=last_wavelength,tilt_intermediate_column=tilt_intermediate_column,tilt_last_column=tilt_last_column,dx_to_etrapolate_to=dx_to_etrapolate_to)
			plt.figure(figsize=(20, 10))
			plt.fill_between(np.arange(len(additive_factor)), additive_factor-additive_factor_sigma,additive_factor+additive_factor_sigma,additive_factor-additive_factor_sigma<additive_factor+additive_factor_sigma,color='y',alpha=0.1)
			plt.plot(additive_factor,'r',label='additive factor')
			plt.plot(np.mean(data[:, :100],axis=-1)-np.min(np.mean(data[:, :100],axis=-1)),'b',label='proxy for minimum signal in the row, not corrected')
			for i in range(41):
				plt.plot([tilt_last_column[0]+i*tilt_last_column[1],tilt_last_column[0]+i*tilt_last_column[1]],[0,7],'--k',linestyle=(0, (5, 4, 1, 4)),linewidth=1)
			plt.grid()
			plt.title('Exp '+str(limits_angle)+' , '+str(j)+'int time sensitivity\n'+'Check to find the right dx_to_etrapolate_to to use in fix_minimum_signal_calibration\nhere '+str(dx_to_etrapolate_to)+' is used\n single frame')
			plt.xlabel('row axis')
			plt.ylabel('counts [au]')
			plt.legend(loc='best')
			plt.ylim(0,7)
			plt.xlim(tilt_last_column[0]-tilt_last_column[1],tilt_last_column[0]+tilt_last_column[1]*41)
			figure_number+=1
			plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
			plt.close()
		if np.sum(np.isnan(additive_factor))>0:
			print(filename+' correction and use aborted')
			continue
		# data = (np.array(data).T + additive_factor).T
		# data_sigma=np.sqrt(np.max([data-100,np.ones_like(data)],axis=0))
		# data_sigma = np.sqrt((data_sigma**2).T + (additive_factor_sigma**2)).T
		# temp1_sigma.append(np.sqrt((data_sigma*Gain[index])**2 + Noise[index]**2))
		# data_sigma_up = data+data_sigma
		# data_sigma_down = data-data_sigma
		# data_sigma_up = apply_proportionality_calibration(data_sigma_up,x_calibration,y_calibration)
		# data_sigma_down = apply_proportionality_calibration(data_sigma_down,x_calibration,y_calibration)
		# data_sigma = (data_sigma_up-data_sigma_down)/2
		# data_sigma = np.sqrt((data_sigma*Gain[index])**2 + Noise[index]**2)
		#
		# temp1.append((data-dataDark_non_proportional)*Gain[index])
		# data = apply_proportionality_calibration(data,x_calibration,y_calibration)
		#
		# data = (data-dataDark)*Gain[index]
		#
		temp.append(data)
		# data_all_for_sigma.append(data/(df_log.loc[j, ['Calib_ID']][0]*1000000))
		# temp_sigma.append(data_sigma)
	data = np.mean(temp, axis=0)
	data_no_add = cp.deepcopy(data)
	data_sigma = np.std(temp, axis=0)

	additive_factor,additive_factor_sigma = fix_minimum_signal_calibration(data,counts_treshold_fixed_increase=106.5,intermediate_wavelength=intermediate_wavelength,last_wavelength=last_wavelength,tilt_intermediate_column=tilt_intermediate_column,tilt_last_column=tilt_last_column,dx_to_etrapolate_to=dx_to_etrapolate_to)

	plt.figure(figsize=(20, 10))
	plt.fill_between(np.arange(len(additive_factor)), additive_factor-additive_factor_sigma,additive_factor+additive_factor_sigma,additive_factor-additive_factor_sigma<additive_factor+additive_factor_sigma,color='y',alpha=0.1)
	plt.plot(additive_factor,'r',label='additive factor')
	plt.plot(np.mean(data[:, :100],axis=-1)-np.min(np.mean(data[:, :100],axis=-1)),'b',label='proxy for minimum signal in the row, not corrected')
	for i in range(41):
		plt.plot([tilt_last_column[0]+i*tilt_last_column[1],tilt_last_column[0]+i*tilt_last_column[1]],[0,7],'--k',linestyle=(0, (5, 4, 1, 4)),linewidth=1)
	plt.grid()
	plt.title('Exp '+str(limits_angle)+' , '+str(j)+'int time sensitivity\n'+'Check to find the right dx_to_etrapolate_to to use in fix_minimum_signal_calibration\nhere '+str(dx_to_etrapolate_to)+' is used\n averaged frames')
	plt.xlabel('row axis')
	plt.ylabel('counts [au]')
	plt.legend(loc='best')
	plt.ylim(0,7)
	plt.xlim(tilt_last_column[0]-tilt_last_column[1],tilt_last_column[0]+tilt_last_column[1]*41)
	figure_number+=1
	plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
	plt.close()

	data = (np.array(data).T + additive_factor).T
	# data_sigma=np.sqrt(np.max([data-100,np.ones_like(data)],axis=0))
	data_sigma = np.sqrt((data_sigma**2).T + (additive_factor_sigma**2)).T
	temp1_sigma = np.sqrt((data_sigma*Gain[0])**2 + Noise[index]**2)
	data_sigma_up = data+data_sigma
	data_sigma_down = data-data_sigma
	data_sigma_up = apply_proportionality_calibration(data_sigma_up,x_calibration,y_calibration)
	data_sigma_down = apply_proportionality_calibration(data_sigma_down,x_calibration,y_calibration)
	data_sigma = (data_sigma_up-data_sigma_down)/2
	data_sigma = np.sqrt((data_sigma*Gain[0])**2 + Noise[index]**2)

	temp1 = (data-dataDark_non_proportional)*Gain[0]
	data = apply_proportionality_calibration(data,x_calibration,y_calibration)

	data = (data-dataDark)*Gain[0]

	# temp.append(data)
	# data_all_for_sigma.append(data/(df_log.loc[j, ['Calib_ID']][0]*1000000))
	# temp_sigma.append(data_sigma)

	# data1 = np.mean(temp1, axis=0)


	# temp_sigma = np.sqrt(np.sum(np.array(temp_sigma)**2, axis=0))/len(temp_sigma)

	data = data/(df_log.loc[j, ['Calib_ID']][0]*1000000)
	data1 = temp1/(df_log.loc[j, ['Calib_ID']][0]*1000000)
	temp_sigma = data_sigma/(df_log.loc[j, ['Calib_ID']][0]*1000000)
	data_all.append(data)
	data_all_non_proportional.append(data1)
	data_sigma_all.append(temp_sigma)


data_all = np.array(data_all)
data_all_non_proportional = np.array(data_all_non_proportional)
data=np.mean(data_all,axis=(0))
data1=np.mean(data_all_non_proportional,axis=(0))
data = rotate(data, geom['angle'])
data1 = rotate(data1, geom['angle'])
data_no_add = rotate(data_no_add, geom['angle'])
data = four_point_transform(data, tilt_4_points_high_int_time)
data1 = four_point_transform(data1, tilt_4_points_high_int_time)
data_no_add = four_point_transform(data_no_add, tilt_4_points_high_int_time)
# data = do_tilt_no_lines(data)
# data1 = do_tilt_no_lines(data1)

data_sigma_all = np.array(data_sigma_all)
data_sigma = np.mean(data_sigma_all,axis=(0))
data_sigma = rotate(data_sigma, geom['angle'])
data_sigma = four_point_transform(data_sigma, tilt_4_points_high_int_time)
# data_sigma = do_tilt_no_lines(data_sigma)

# first_bin, bin_interv = get_bin_and_interv_no_lines(data)

to_plot_pcolor = cp.deepcopy(data_no_add)
waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(to_plot_pcolor)[1]+1))
temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(to_plot_pcolor)[0]+1))
plt.figure(figsize=(20, 10))
plt.pcolor(temp_w,temp_LOS,to_plot_pcolor,cmap='rainbow',vmax=np.max(to_plot_pcolor[:,600]),vmin=90,rasterized=True)
plt.colorbar()
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(to_plot_pcolor)[0]-1],'--k',linewidth=0.5)
for i in range(41):
	plt.plot([np.min(waveaxis),np.max(waveaxis)],[first_bin_high_int_time+i*bin_interv_high_int_time,first_bin_high_int_time+i*bin_interv_high_int_time],'--k',linewidth=0.5)
plt.title('Exp '+str(limits_angle)+'\n Low int time sensitivity\nwithout negative signal correction (zoom left)')
plt.xlabel('wavelength [nm]')
plt.ylabel('LOS axis [pixels]')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()

to_plot_pcolor = cp.deepcopy(data1)
waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(to_plot_pcolor)[1]+1))
temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(to_plot_pcolor)[0]+1))
plt.figure(figsize=(20, 10))
plt.pcolor(temp_w,temp_LOS,to_plot_pcolor,cmap='rainbow',vmax=np.max(to_plot_pcolor),rasterized=True)
plt.colorbar()
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(to_plot_pcolor)[0]-1],'--k',linewidth=0.5)
for i in range(41):
	plt.plot([np.min(waveaxis),np.max(waveaxis)],[first_bin_high_int_time+i*bin_interv_high_int_time,first_bin_high_int_time+i*bin_interv_high_int_time],'--k',linewidth=0.5)
plt.title('Exp '+str(limits_angle)+'\n Low int time sensitivity\nwithout proportionality correction')
plt.xlabel('wavelength [nm]')
plt.ylabel('LOS axis [pixels]')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()

to_plot_pcolor = cp.deepcopy(data)
waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(to_plot_pcolor)[1]+1))
temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(to_plot_pcolor)[0]+1))
plt.figure(figsize=(20, 10))
plt.pcolor(temp_w,temp_LOS,to_plot_pcolor,cmap='rainbow',vmax=np.max(to_plot_pcolor),rasterized=True)
plt.colorbar()
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(to_plot_pcolor)[0]-1],'--k',linewidth=0.5)
for i in range(41):
	plt.plot([np.min(waveaxis),np.max(waveaxis)],[first_bin_high_int_time+i*bin_interv_high_int_time,first_bin_high_int_time+i*bin_interv_high_int_time],'--k',linewidth=0.5)
plt.title('Exp '+str(limits_angle)+'\n Low int time sensitivity\nwith proportionality correction')
plt.xlabel('wavelength [nm]')
plt.ylabel('LOS axis [pixels]')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()

plt.figure(figsize=(20, 10))
plt.pcolor(temp_w,temp_LOS,to_plot_pcolor,cmap='rainbow',vmax=np.max(to_plot_pcolor[:,600]),rasterized=True)
plt.colorbar()
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(to_plot_pcolor)[0]-1],'--k',linewidth=0.5)
for i in range(41):
	plt.plot([np.min(waveaxis),np.max(waveaxis)],[first_bin_high_int_time+i*bin_interv_high_int_time,first_bin_high_int_time+i*bin_interv_high_int_time],'--k',linewidth=0.5)
plt.title('Exp '+str(limits_angle)+'\n Low int time sensitivity\nwith proportionality correction')
plt.xlabel('wavelength [nm]')
plt.ylabel('LOS axis [pixels]')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()

# plt.figure()
# plt.imshow(data,'rainbow',origin='lower')
# for i in range(41):
# 	plt.plot([0,np.shape(data)[-1]],[first_bin_high_int_time+i*bin_interv_high_int_time,first_bin_high_int_time+i*bin_interv_high_int_time],'--k',linestyle=(0, (5, 4, 1, 4)),linewidth=0.5)
# plt.title('Exp '+str(limits_angle)+'\n Low int time sensitivity\nwith proportionality correction')
# plt.xlabel('wavelength axis [pixels]')
# plt.ylabel('LOS axis [pixels]')
# plt.colorbar().set_label('counts [au]')
# figure_number+=1
# plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
# plt.close()
to_plot_pcolor = cp.deepcopy(data_sigma)
waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(to_plot_pcolor)[1]+1))
temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(to_plot_pcolor)[0]+1))
plt.figure(figsize=(20, 10))
plt.pcolor(temp_w,temp_LOS,to_plot_pcolor,cmap='rainbow',vmax=np.max(to_plot_pcolor),rasterized=True)
plt.colorbar()
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(to_plot_pcolor)[0]-1],'--k',linewidth=0.5)
for i in range(41):
	plt.plot([np.min(waveaxis),np.max(waveaxis)],[first_bin_high_int_time+i*bin_interv_high_int_time,first_bin_high_int_time+i*bin_interv_high_int_time],'--k',linewidth=0.5)
plt.title('Exp '+str(limits_angle)+'\n Low int time sensitivity error\nwith proportionality correction')
plt.xlabel('wavelength [nm]')
plt.ylabel('LOS axis [pixels]')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.figure()
# plt.imshow(data_sigma,'rainbow',origin='lower')
# for i in range(41):
# 	plt.plot([0,np.shape(data)[-1]],[first_bin_high_int_time+i*bin_interv_high_int_time,first_bin_high_int_time+i*bin_interv_high_int_time],'--k',linestyle=(0, (5, 4, 1, 4)),linewidth=0.5)
# plt.title('Exp '+str(limits_angle)+'\n Low int time sensitivity error\nwith proportionality correction')
# plt.xlabel('wavelength axis [pixels]')
# plt.ylabel('LOS axis [pixels]')
# plt.colorbar().set_label('counts [au]')
# figure_number+=1
# plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
# plt.close()

# plt.figure()
# plt.imshow(data1,'rainbow',origin='lower')
# for i in range(41):
# 	plt.plot([0,np.shape(data1)[-1]],[first_bin_high_int_time+i*bin_interv_high_int_time,first_bin_high_int_time+i*bin_interv_high_int_time],'--k',linestyle=(0, (5, 4, 1, 4)),linewidth=0.5)
# plt.title('Exp '+str(limits_angle)+'\n Low int time sensitivity\nwithout proportionality correction')
# plt.xlabel('wavelength axis [pixels]')
# plt.ylabel('LOS axis [pixels]')
# plt.colorbar().set_label('counts [au]')
# figure_number+=1
# plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
# plt.close()
# plt.pause(0.01)

for kernel in [[1,3],[1,5],[1,7]]:	#added 12/02/2020
	data2 = medfilt(np.mean(data_all,axis=(0)),kernel)
	data3 = medfilt(np.mean(data_all_non_proportional,axis=(0)),kernel)
	data2 = rotate(data2, geom['angle'])
	data3 = rotate(data3, geom['angle'])
	data2 = four_point_transform(data2, tilt_4_points_high_int_time)
	data3 = four_point_transform(data3, tilt_4_points_high_int_time)
	# data2 = do_tilt_no_lines(data2)
	# data3 = do_tilt_no_lines(data3)

	data_sigma = np.mean(data_sigma_all,axis=(0))
	data_sigma = convolve(data_sigma, np.ones((kernel[0],kernel[1]))/(kernel[0]*kernel[1]))	# I think this is convervative because more points I average the smaller the error, but here I do a simlpe mean of it
	data_sigma = rotate(data_sigma, geom['angle'])
	data_sigma = four_point_transform(data_sigma, tilt_4_points_high_int_time)
	# data_sigma = do_tilt_no_lines(data_sigma)

	to_plot_pcolor = cp.deepcopy(data2)
	waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(to_plot_pcolor)[1]+1))
	temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(to_plot_pcolor)[0]+1))
	plt.figure(figsize=(20, 10))
	plt.pcolor(temp_w,temp_LOS,to_plot_pcolor,cmap='rainbow',vmax=np.max(to_plot_pcolor),rasterized=True)
	plt.colorbar()
	for i in range(len(waveLengths)):
		plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(to_plot_pcolor)[0]-1],'--k',linewidth=0.5)
	for i in range(41):
		plt.plot([np.min(waveaxis),np.max(waveaxis)],[first_bin_high_int_time+i*bin_interv_high_int_time,first_bin_high_int_time+i*bin_interv_high_int_time],'--k',linewidth=0.5)
	plt.title('Exp '+str(limits_angle)+'\n Low int time sensitivity\nwith proportionality correction and median filter, kernel '+str(kernel)+'\nI pick [1,5]')
	plt.xlabel('wavelength [nm]')
	plt.ylabel('LOS axis [pixels]')
	figure_number+=1
	plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
	plt.close()
	# plt.figure()
	# plt.imshow(data2,'rainbow',origin='lower')
	# for i in range(41):
	# 	plt.plot([0,np.shape(data)[-1]],[first_bin_high_int_time+i*bin_interv_high_int_time,first_bin_high_int_time+i*bin_interv_high_int_time],'--k',linestyle=(0, (5, 4, 1, 4)),linewidth=0.5)
	# plt.title('Exp '+str(limits_angle)+'\n Low int time sensitivity\nwith proportionality correction and median filter, kernel '+str(kernel)+'\nI pick [1,5]')
	# plt.xlabel('wavelength axis [pixels]')
	# plt.ylabel('LOS axis [pixels]')
	# plt.colorbar().set_label('counts [au]')
	# figure_number+=1
	# plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
	# plt.close()
	to_plot_pcolor = cp.deepcopy(data_sigma)
	waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(to_plot_pcolor)[1]+1))
	temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(to_plot_pcolor)[0]+1))
	plt.figure(figsize=(20, 10))
	plt.pcolor(temp_w,temp_LOS,to_plot_pcolor,cmap='rainbow',vmax=np.max(to_plot_pcolor),rasterized=True)
	plt.colorbar()
	for i in range(len(waveLengths)):
		plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(to_plot_pcolor)[0]-1],'--k',linewidth=0.5)
	for i in range(41):
		plt.plot([np.min(waveaxis),np.max(waveaxis)],[first_bin_high_int_time+i*bin_interv_high_int_time,first_bin_high_int_time+i*bin_interv_high_int_time],'--k',linewidth=0.5)
	plt.title('Exp '+str(limits_angle)+'\n Low int time sensitivity\nwith proportionality correction error, averaged over kernel '+str(kernel)+'\nI pick [1,5]')
	plt.xlabel('wavelength [nm]')
	plt.ylabel('LOS axis [pixels]')
	figure_number+=1
	plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
	plt.close()
	# plt.figure()
	# plt.imshow(data_sigma,'rainbow',origin='lower')
	# for i in range(41):
	# 	plt.plot([0,np.shape(data)[-1]],[first_bin_high_int_time+i*bin_interv_high_int_time,first_bin_high_int_time+i*bin_interv_high_int_time],'--k',linestyle=(0, (5, 4, 1, 4)),linewidth=0.5)
	# plt.title('Exp '+str(limits_angle)+'\n Low int time sensitivity\nwith proportionality correction error, averaged over kernel '+str(kernel)+'\nI pick [1,5]')
	# plt.xlabel('wavelength axis [pixels]')
	# plt.ylabel('LOS axis [pixels]')
	# plt.colorbar().set_label('counts [au]')
	# figure_number+=1
	# plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
	# plt.close()
	to_plot_pcolor = cp.deepcopy(data3)
	waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(to_plot_pcolor)[1]+1))
	temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(to_plot_pcolor)[0]+1))
	plt.figure(figsize=(20, 10))
	plt.pcolor(temp_w,temp_LOS,to_plot_pcolor,cmap='rainbow',vmax=np.max(to_plot_pcolor),rasterized=True)
	plt.colorbar()
	for i in range(len(waveLengths)):
		plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(to_plot_pcolor)[0]-1],'--k',linewidth=0.5)
	for i in range(41):
		plt.plot([np.min(waveaxis),np.max(waveaxis)],[first_bin_high_int_time+i*bin_interv_high_int_time,first_bin_high_int_time+i*bin_interv_high_int_time],'--k',linewidth=0.5)
	plt.title('Exp '+str(limits_angle)+'\n Low int time sensitivity\nwithout proportionality correction, kernel '+str(kernel)+'\nI pick [1,5]')
	plt.xlabel('wavelength [nm]')
	plt.ylabel('LOS axis [pixels]')
	figure_number+=1
	plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
	plt.close()
	# plt.figure()
	# plt.imshow(data3,'rainbow',origin='lower')
	# for i in range(41):
	# 	plt.plot([0,np.shape(data1)[-1]],[first_bin_high_int_time+i*bin_interv_high_int_time,first_bin_high_int_time+i*bin_interv_high_int_time],'--k',linestyle=(0, (5, 4, 1, 4)),linewidth=0.5)
	# plt.title('Exp '+str(limits_angle)+'\n Low int time sensitivity\nwithout proportionality correction, kernel '+str(kernel)+'\nI pick [1,5]')
	# plt.xlabel('wavelength axis [pixels]')
	# plt.ylabel('LOS axis [pixels]')
	# plt.colorbar().set_label('counts [au]')
	# figure_number+=1
	# plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
	# plt.close()

kernel = [1,5]
data2 = medfilt(np.mean(data_all,axis=(0)),kernel)
data3 = medfilt(np.mean(data_all_non_proportional,axis=(0)),kernel)
data2 = rotate(data2, geom['angle'])
data3 = rotate(data3, geom['angle'])
data2 = four_point_transform(data2, tilt_4_points_high_int_time)
data3 = four_point_transform(data3, tilt_4_points_high_int_time)
# data2 = do_tilt_no_lines(data2)
# data3 = do_tilt_no_lines(data3)

data_sigma = np.mean(data_sigma_all,axis=(0))
data_sigma = convolve(data_sigma, np.ones((kernel[0],kernel[1]))/(kernel[0]*kernel[1]))	# I think this is convervative because more points I average the smaller the error, but here I do a simlpe mean of it
data_sigma = rotate(data_sigma, geom['angle'])
data_sigma = four_point_transform(data_sigma, tilt_4_points_high_int_time)
# data_sigma = do_tilt_no_lines(data_sigma)

if True:	# This enables the use of the median filter, modified 12/02/2020
	data = copy.deepcopy(data2)
data_no_mask = copy.deepcopy(data)
# data_no_mask = data_no_mask - min(np.min(data_no_mask),0)
# first_bin, bin_interv = get_bin_and_interv_no_lines(data)
print('bin interval of ' + str(bin_interv_high_int_time)+' , first_bin of '+str(first_bin_high_int_time))
# bin_interv_no_mask = bin_interv
binned_data_no_mask,binned_data_no_mask_sigma = binData_with_sigma(data,data_sigma, first_bin_high_int_time, bin_interv_high_int_time)
# binned_data_no_mask = binned_data_no_mask - min(np.min(binned_data_no_mask),0)
longCal_no_mask,longCal_no_mask_sigma = binData_with_sigma(data,data_sigma,first_bin_high_int_time-bin_interv_high_int_time,bin_interv_high_int_time,nCh = 42)
topDark_no_mask = longCal_no_mask[0]
botDark_no_mask = longCal_no_mask[-1]
wtDark_no_mask = np.arange(1, 41) / 41
calBinned_no_mask = longCal_no_mask[1:41] - (np.outer(wtDark_no_mask, topDark_no_mask) + np.outer(wtDark_no_mask[::-1], botDark_no_mask))#  + (topDark_no_mask + botDark_no_mask) / 2  # MODIFIED TO AVOID NEGATIVE VALUES
# I do this to eliminate the negative values and relpace them with neighbouring ones
for iLine in range(len(calBinned_no_mask)):
	while np.sum(calBinned_no_mask[iLine]<=0)>0:
		select = calBinned_no_mask[iLine]<=0
		positive_values = calBinned_no_mask[iLine][calBinned_no_mask[iLine]>0]
		positive_indexes = np.arange(np.shape(calBinned_no_mask)[1])[calBinned_no_mask[iLine]>0]
		for iwave in (np.arange(np.shape(calBinned_no_mask)[1])[select]):
			calBinned_no_mask[iLine,iwave] = positive_values[np.abs(positive_indexes-iwave).argmin()]
topDark_no_mask = longCal_no_mask_sigma[0]
botDark_no_mask = longCal_no_mask_sigma[-1]
wtDark_no_mask = np.arange(1, 41) / 41
calBinned_no_mask_sigma = np.sqrt((longCal_no_mask_sigma[1:41])**2 + (np.outer(wtDark_no_mask, topDark_no_mask))**2 + (np.outer(wtDark_no_mask[::-1], botDark_no_mask))**2 + (topDark_no_mask**2 + botDark_no_mask**2) / 4 )  # MODIFIED TO AVOID NEGATIVE VALUES

peak_sum_single = np.sum(binned_data_no_mask[:, wavel - wavel_range:wavel + wavel_range], axis=(-1))  # type: ndarray
calBinned_peak_sum_single = np.sum(calBinned_no_mask[:, wavel - wavel_range:wavel + wavel_range], axis=(-1))  # type: ndarray


to_plot_pcolor = cp.deepcopy(longCal_no_mask)
waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(to_plot_pcolor)[1]+1))
temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(to_plot_pcolor)[0]+1))
plt.figure(figsize=(20, 10))
plt.pcolor(temp_w,temp_LOS,to_plot_pcolor,cmap='rainbow',vmax=np.max(to_plot_pcolor),rasterized=True)
plt.colorbar()
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(to_plot_pcolor)[0]-1],'--k',linewidth=0.5)
plt.title('Exp '+str(limits_angle)+'\n binned data before upper and lower bin stray substraction')
plt.xlabel('wavelength [nm]')
plt.ylabel('LOS axis [pixels]')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()

to_plot_pcolor = cp.deepcopy(longCal_no_mask)
waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(to_plot_pcolor)[1]+1))
temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(to_plot_pcolor)[0]+1))
plt.figure(figsize=(20, 10))
plt.pcolor(temp_w,temp_LOS,to_plot_pcolor,cmap='rainbow',vmax=np.max(to_plot_pcolor[:,600]),rasterized=True)
plt.colorbar()
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(to_plot_pcolor)[0]-1],'--k',linewidth=0.5)
plt.title('Exp '+str(limits_angle)+'\n binned data before upper and lower bin stray substraction low waves')
plt.xlabel('wavelength [nm]')
plt.ylabel('LOS axis [pixels]')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()

to_plot_pcolor = cp.deepcopy(calBinned_no_mask)
waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(to_plot_pcolor)[1]+1))
temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(to_plot_pcolor)[0]+1))
plt.figure(figsize=(20, 10))
plt.pcolor(temp_w,temp_LOS,to_plot_pcolor,cmap='rainbow',vmax=np.max(to_plot_pcolor),rasterized=True)
plt.colorbar()
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(to_plot_pcolor)[0]-1],'--k',linewidth=0.5)
plt.title('Exp '+str(limits_angle)+'\n binned data after upper and lower bin stray substraction')
plt.xlabel('wavelength [nm]')
plt.ylabel('LOS axis [pixels]')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()

to_plot_pcolor = cp.deepcopy(calBinned_no_mask)
waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(to_plot_pcolor)[1]+1))
temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(to_plot_pcolor)[0]+1))
plt.figure(figsize=(20, 10))
plt.pcolor(temp_w,temp_LOS,to_plot_pcolor,cmap='rainbow',vmax=np.max(longCal_no_mask[:,600]),rasterized=True)
plt.colorbar()
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(to_plot_pcolor)[0]-1],'--k',linewidth=0.5)
plt.title('Exp '+str(limits_angle)+'\n binned data after upper and lower bin stray substraction')
plt.xlabel('wavelength [nm]')
plt.ylabel('LOS axis [pixels]')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()


max_coordinate=0
plt.figure()
waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(np.mean(calBinned_no_mask,axis=0))[0]))
scaled_down_calBinned=[]
for i in range(len(binned_all)):
	scaled_down_calBinned.append((calBinned_all[i].T * (calBinned_peak_sum_single / calBinned_peak_sum[i])).T)
	plt.plot(waveaxis,np.mean(scaled_down_calBinned[-1],axis=0),label=str(int_time_long[i]))
	max_coordinate = max(max_coordinate, np.max(np.mean(scaled_down_calBinned[-1],axis=0)))
scaled_down_calBinned=np.array(scaled_down_calBinned)

plt.plot(waveaxis,np.mean(calBinned_no_mask,axis=0),label='0.02')
max_coordinate = max(max_coordinate, np.max(np.mean(calBinned_no_mask,axis=0)))
plt.title('Exp '+str(limits_angle)+'\n Comparison of scaled_down_calBinned averaged over all bins\n and scaled around column '+str(wavel)+' for different integration times')
plt.xlabel('wavelength [nm]')
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,max_coordinate],'--k',linewidth=0.5)
plt.ylabel('sensitivity [au]')


averaged_scaled_down_calBinned=np.mean(scaled_down_calBinned,axis=0)
wavel_bottom = np.convolve(np.mean(averaged_scaled_down_calBinned,axis=0), np.ones(200)/200 , mode='valid')
wavel_bottom = wavel_bottom.argmin()+100
bottom_level = np.sum(averaged_scaled_down_calBinned[:, wavel_bottom - wavel_range:wavel_bottom + wavel_range], axis=(-1))  # type: ndarray
bottom_level_single = np.sum(calBinned_no_mask[:, wavel_bottom - wavel_range:wavel_bottom + wavel_range], axis=(-1))  # type: ndarray
averaged_scaled_down_calBinned[averaged_scaled_down_calBinned<=np.min(np.mean(averaged_scaled_down_calBinned,axis=0))]=np.min(np.mean(averaged_scaled_down_calBinned,axis=0))	#this is to erase negative or too low values at the borders
plt.plot(waveaxis,np.mean(averaged_scaled_down_calBinned,axis=0),'--',label='Averaged')

waveaxis = np.polyval(waveLcoefs[1],[wavel,wavel])
plt.plot(waveaxis,[0,max_coordinate],'--k')

plt.grid()
plt.legend(loc='best', fontsize='small')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()

calBinned_no_mask_smoothed = np.zeros_like(calBinned_no_mask)
for index in range(len(calBinned_no_mask)):
	calBinned_no_mask_smoothed[index] = smoothing_function(calBinned_no_mask[index],300)



scaled_up_calBinned_single = calBinned_no_mask.T-calBinned_peak_sum_single/(wavel_range*2)
scaled_up_calBinned_single = scaled_up_calBinned_single * (calBinned_peak_sum_single-bottom_level)/(calBinned_peak_sum_single-bottom_level_single)
scaled_up_calBinned_single = (scaled_up_calBinned_single+calBinned_peak_sum_single/(wavel_range*2)).T

color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro','paleturquoise']
plt.figure()
waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(np.mean(calBinned_no_mask,axis=0))[0]))
plt.xlabel('wavelength [nm]')
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.max(np.mean(calBinned_no_mask_smoothed,axis=0))],'--k',linewidth=0.5)
for iLine,Line in enumerate([5,20,25,30]):
	plt.plot(waveaxis,calBinned_no_mask[iLine],'--',color=color[iLine],linewidth=0.5,label=str(Line))
	plt.plot(waveaxis,calBinned_no_mask_smoothed[Line],'-',color=color[iLine],linewidth=0.5)
	plt.plot(waveaxis,scaled_up_calBinned_single[Line],':',color=color[iLine],linewidth=0.5)
plt.title('Exp '+str(limits_angle)+'\n Average calibration at smoothing\n--=before, -=after (not used), :=scaled up (not used)')
plt.ylabel('sensitivity [au]')
# plt.plot([wavel,wavel],[np.min(calBinned_no_mask),np.max(calBinned_no_mask)],'--k',label='peak location')
# plt.plot([wavel_bottom,wavel_bottom],[np.min(calBinned_no_mask),np.max(calBinned_no_mask)],'--r',label='bottom location')
plt.grid()
plt.legend(loc='best', fontsize='x-small')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)


spline=interpolate.interp1d(df_sphere.waveL,df_sphere.Rad,kind='cubic') #Spline interp of sphere radiance

# for index in range(len(scaled_up_calBinned_single)):
# 	scaled_up_calBinned_single[index] = smoothing_function(scaled_up_calBinned_single[index],300)




# scaled_up_calBinned_single[scaled_up_calBinned_single<0]=np.min(scaled_up_calBinned_single[scaled_up_calBinned_single>0])
# spline=interpolate.interp1d(df_sphere.waveL,df_sphere.Rad,kind='cubic') #Spline interp of sphere radiance
# calibration = calBinned_no_mask / spline(np.polyval(waveLcoefs[1],np.arange(calBinned_no_mask.shape[1])))
# calibration = copy.deepcopy(calBinned_no_mask_smoothed)
calibration = copy.deepcopy(calBinned_no_mask)	# modified 12/02/2012
calibration_sigma = copy.deepcopy(calBinned_no_mask_sigma)	# modified 12/02/2012

to_plot_pcolor = cp.deepcopy(calibration)
waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(to_plot_pcolor)[1]+1))
temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(to_plot_pcolor)[0]+1))
plt.figure(figsize=(20, 10))
plt.pcolor(temp_w,temp_LOS,to_plot_pcolor,cmap='rainbow',vmax=np.max(to_plot_pcolor),rasterized=True)
plt.colorbar()
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(to_plot_pcolor)[0]-1],'--k',linewidth=0.5)
plt.title('Exp '+str(limits_angle)+'\n calBinned_no_mask before Gaussian convolution')
plt.xlabel('wavelength [nm]')
plt.ylabel('LOS axis [pixels]')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.figure()
# plt.imshow(calibration, 'rainbow',origin='lower',aspect=10)
# plt.title('Exp '+str(limits_angle)+'\n calBinned_no_mask_smoothed before Gaussian convolution')
# plt.colorbar()
# figure_number+=1
# plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
# plt.close()
to_plot_pcolor = cp.deepcopy(calibration_sigma)
waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(to_plot_pcolor)[1]+1))
temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(to_plot_pcolor)[0]+1))
plt.figure(figsize=(20, 10))
plt.pcolor(temp_w,temp_LOS,to_plot_pcolor,cmap='rainbow',vmax=np.max(to_plot_pcolor),rasterized=True)
plt.colorbar()
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(to_plot_pcolor)[0]-1],'--k',linewidth=0.5)
plt.title('Exp '+str(limits_angle)+'\n calBinned_no_mask before Gaussian convolution error')
plt.xlabel('wavelength [nm]')
plt.ylabel('LOS axis [pixels]')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.figure()
# plt.imshow(calibration_sigma, 'rainbow',origin='lower',aspect=10)
# plt.title('Exp '+str(limits_angle)+'\n calBinned_no_mask_smoothed before Gaussian convolution error')
# plt.colorbar()
# figure_number+=1
# plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
# plt.close()
# plt.pause(0.01)

calibration = calibration / spline(np.polyval(waveLcoefs[1],np.arange(calibration.shape[1])))
calibration_sigma = calibration_sigma / spline(np.polyval(waveLcoefs[1],np.arange(calibration.shape[1])))

calibration_convoluted = cp.deepcopy(calibration)
FWHM = 20	#this is about the width of a peak
for index,chord in enumerate(calibration_convoluted):
	calibration_convoluted[index] = np.convolve(chord, gaussian(100,FWHM/2.335)/np.sum(gaussian(100,FWHM/2.335)) , mode='same')	# I convolute it with a gaussian shape because peaks are about gaussian

# if np.sum(calibration<=0)>0:
# 	calibration = calibration + max(-np.min(calibration),0)+0.001	# 0.001 it's arbitrary just to avoid 1/0
calibration[calibration<=0] = np.min(calibration[calibration>0])
calibration_convoluted[calibration_convoluted<=0] = np.min(calibration_convoluted[calibration_convoluted>0])



# calibration = calibration / spline(np.polyval(waveLcoefs[1],np.arange(calibration.shape[1])))

to_plot_pcolor = cp.deepcopy(calibration_convoluted)
waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(to_plot_pcolor)[1]+1))
temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(to_plot_pcolor)[0]+1))
plt.figure(figsize=(20, 10))
plt.pcolor(temp_w,temp_LOS,to_plot_pcolor,cmap='rainbow',vmax=np.max(to_plot_pcolor),rasterized=True)
plt.colorbar()
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(to_plot_pcolor)[0]-1],'--k',linewidth=0.5)
plt.title('Exp '+str(limits_angle)+'\n Convoluted calibration')
plt.xlabel('wavelength [nm]')
plt.ylabel('LOS axis [pixels]')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.figure()
# plt.imshow(calibration_convoluted, 'rainbow',origin='lower',aspect=10)
# plt.title('Exp '+str(limits_angle)+'\n Convoluted calibration')
# plt.colorbar()
# figure_number+=1
# plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
# plt.close()

to_plot_pcolor = cp.deepcopy(calibration)
waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(to_plot_pcolor)[1]+1))
temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(to_plot_pcolor)[0]+1))
plt.figure(figsize=(20, 10))
plt.pcolor(temp_w,temp_LOS,to_plot_pcolor,cmap='rainbow',vmax=np.max(to_plot_pcolor),rasterized=True)
plt.colorbar()
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(to_plot_pcolor)[0]-1],'--k',linewidth=0.5)
plt.title('Exp '+str(limits_angle)+'\n Non convoluted calibration')
plt.xlabel('wavelength [nm]')
plt.ylabel('LOS axis [pixels]')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.figure()
# plt.imshow(calibration, 'rainbow',origin='lower',aspect=10)
# plt.title('Exp '+str(limits_angle)+'\n Non convoluted calibration')
# plt.colorbar()
# figure_number+=1
# plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
# plt.close()
to_plot_pcolor = cp.deepcopy(calibration_sigma)
waveaxis = np.polyval(waveLcoefs[1],np.arange(np.shape(to_plot_pcolor)[1]+1))
temp_w,temp_LOS = np.meshgrid(waveaxis,np.arange(np.shape(to_plot_pcolor)[0]+1))
plt.figure(figsize=(20, 10))
plt.pcolor(temp_w,temp_LOS,to_plot_pcolor,cmap='rainbow',vmax=np.max(to_plot_pcolor),rasterized=True)
plt.colorbar()
for i in range(len(waveLengths)):
	plt.plot([waveLengths[i],waveLengths[i]],[0,np.shape(to_plot_pcolor)[0]-1],'--k',linewidth=0.5)
plt.title('Exp '+str(limits_angle)+'\n Non convoluted calibration error')
plt.xlabel('wavelength [nm]')
plt.ylabel('LOS axis [pixels]')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.figure()
# plt.imshow(calibration_sigma, 'rainbow',origin='lower',aspect=10)
# plt.title('Exp '+str(limits_angle)+'\n Non convoluted calibration error')
# plt.colorbar()
# figure_number+=1
# plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
# plt.close()
# plt.pause(0.01)

to_save = np.zeros((3,)+np.shape(calibration))
to_save[1] = calibration
np.save('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_12/Sensitivity_4_non_convoluted',to_save)

to_save = np.zeros((3,)+np.shape(calibration_convoluted))
to_save[1] = calibration_convoluted
np.save('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_11/Sensitivity_4',to_save)
to_save = np.zeros((3,)+np.shape(calibration_convoluted))
to_save[1] = calibration_sigma
np.save('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_12/Sensitivity_4_non_convoluted_sigma',to_save)




print("I only calculate Sensitivity_4 now")
exit()



# NOW I GET AGAIN THE SENSITIVITY FOR THE EXPERIMENTS IN WHICH I HAD THE anti overexposure FILTER

limits_angle = [40,52]
limits_tilt = [40,52]
limits_wave = [40,52]
figure_number = 0
where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_7/Sensitivity_2_plots_'

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
	plt.title('Exp ' + str(limits_angle) + '\n ' + str(calib_ID_long[index]) + 'int time sensitivity')
	plt.xlabel('wavelength axis [pixels]')
	plt.ylabel('LOS axis [pixels]')
	figure_number += 1
	plt.savefig(where_to_save_everything + str(figure_number) + '.eps', bbox_inches='tight')
	plt.close()
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
scaled_down_calBinned=[]
for i in range(len(binned_all)):
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
wavel_bottom = np.convolve(np.mean(averaged_scaled_down_calBinned,axis=0), np.ones(200)/200 , mode='valid')
wavel_bottom = wavel_bottom.argmin()+100
bottom_level = np.sum(averaged_scaled_down_calBinned[:, wavel_bottom - wavel_range:wavel_bottom + wavel_range], axis=(-1))  # type: ndarray
bottom_level_single = np.sum(calBinned_no_mask[:, wavel_bottom - wavel_range:wavel_bottom + wavel_range], axis=(-1))  # type: ndarray
averaged_scaled_down_calBinned[averaged_scaled_down_calBinned<=np.min(np.mean(averaged_scaled_down_calBinned,axis=0))]=np.min(np.mean(averaged_scaled_down_calBinned,axis=0))	#this is to erase negative or too low values at the borders
plt.plot(np.mean(averaged_scaled_down_calBinned,axis=0),label='Averaged')


plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()



scaled_up_calBinned_single = calBinned_no_mask.T-calBinned_peak_sum_single/200
scaled_up_calBinned_single = scaled_up_calBinned_single * (calBinned_peak_sum_single-bottom_level)/(calBinned_peak_sum_single-bottom_level_single)
scaled_up_calBinned_single = (scaled_up_calBinned_single+calBinned_peak_sum_single/200).T


plt.figure()
plt.plot(np.mean(calBinned_no_mask,axis=0),label='before')
plt.plot(np.mean(scaled_up_calBinned_single,axis=0),label='after')
plt.title('Exp '+str(limits_angle)+'\n Average calibration at scaling up')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.plot([wavel,wavel],[np.min(calBinned_no_mask),np.max(calBinned_no_mask)],'--k',label='peak location')
plt.plot([wavel_bottom,wavel_bottom],[np.min(calBinned_no_mask),np.max(calBinned_no_mask)],'--r',label='bottom location')

plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)


spline=interpolate.interp1d(df_sphere.waveL,df_sphere.Rad,kind='cubic') #Spline interp of sphere radiance

for index in range(len(scaled_up_calBinned_single)):
	scaled_up_calBinned_single[index] = smoothing_function(scaled_up_calBinned_single[index],300)




# scaled_up_calBinned_single[scaled_up_calBinned_single<0]=np.min(scaled_up_calBinned_single[scaled_up_calBinned_single>0])
# spline=interpolate.interp1d(df_sphere.waveL,df_sphere.Rad,kind='cubic') #Spline interp of sphere radiance
# calibration = calBinned_no_mask / spline(np.polyval(waveLcoefs[1],np.arange(calBinned_no_mask.shape[1])))
calibration = copy.deepcopy(scaled_up_calBinned_single)

plt.figure()
plt.imshow(calibration, 'rainbow',origin='lower',aspect=10)
plt.title('Exp '+str(limits_angle)+'\n scaled_up_calBinned_single pre-smoothed')
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

np.save('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_7/Sensitivity_2',to_save)









# NOW I GET AGAIN THE SENSITIVITY FOR THE FIRST ROUND OF EXPERIMENTS (merge<26)

limits_angle = [17,26]
limits_tilt = [17,26]
limits_wave = [17,26]
figure_number = 0
where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_7/Sensitivity_1_plots_'

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
	plt.title('Exp ' + str(limits_angle) + '\n ' + str(calib_ID_long[index]) + 'int time sensitivity')
	plt.xlabel('wavelength axis [pixels]')
	plt.ylabel('LOS axis [pixels]')
	figure_number += 1
	plt.savefig(where_to_save_everything + str(figure_number) + '.eps', bbox_inches='tight')
	plt.close()
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
scaled_down_calBinned=[]
for i in range(len(binned_all)):
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
wavel_bottom = np.convolve(np.mean(averaged_scaled_down_calBinned,axis=0), np.ones(200)/200 , mode='valid')
wavel_bottom = wavel_bottom.argmin()+100
bottom_level = np.sum(averaged_scaled_down_calBinned[:, wavel_bottom - wavel_range:wavel_bottom + wavel_range], axis=(-1))  # type: ndarray
bottom_level_single = np.sum(calBinned_no_mask[:, wavel_bottom - wavel_range:wavel_bottom + wavel_range], axis=(-1))  # type: ndarray
averaged_scaled_down_calBinned[averaged_scaled_down_calBinned<=np.min(np.mean(averaged_scaled_down_calBinned,axis=0))]=np.min(np.mean(averaged_scaled_down_calBinned,axis=0))	#this is to erase negative or too low values at the borders
plt.plot(np.mean(averaged_scaled_down_calBinned,axis=0),label='Averaged')


plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()



scaled_up_calBinned_single = calBinned_no_mask.T-calBinned_peak_sum_single/200
scaled_up_calBinned_single = scaled_up_calBinned_single * (calBinned_peak_sum_single-bottom_level)/(calBinned_peak_sum_single-bottom_level_single)
scaled_up_calBinned_single = (scaled_up_calBinned_single+calBinned_peak_sum_single/200).T


plt.figure()
plt.plot(np.mean(calBinned_no_mask,axis=0),label='before')
plt.plot(np.mean(scaled_up_calBinned_single,axis=0),label='after')
plt.title('Exp '+str(limits_angle)+'\n Average calibration at scaling up')
plt.xlabel('wavelength axis [au]')
plt.ylabel('sensitivity [au]')
plt.plot([wavel,wavel],[np.min(calBinned_no_mask),np.max(calBinned_no_mask)],'--k',label='peak location')
plt.plot([wavel_bottom,wavel_bottom],[np.min(calBinned_no_mask),np.max(calBinned_no_mask)],'--r',label='bottom location')

plt.legend(loc='best')
figure_number+=1
plt.savefig(where_to_save_everything+str(figure_number)+'.eps', bbox_inches='tight')
plt.close()
# plt.pause(0.01)


spline=interpolate.interp1d(df_sphere.waveL,df_sphere.Rad,kind='cubic') #Spline interp of sphere radiance

for index in range(len(scaled_up_calBinned_single)):
	scaled_up_calBinned_single[index] = smoothing_function(scaled_up_calBinned_single[index],300)




# scaled_up_calBinned_single[scaled_up_calBinned_single<0]=np.min(scaled_up_calBinned_single[scaled_up_calBinned_single>0])
# spline=interpolate.interp1d(df_sphere.waveL,df_sphere.Rad,kind='cubic') #Spline interp of sphere radiance
# calibration = calBinned_no_mask / spline(np.polyval(waveLcoefs[1],np.arange(calBinned_no_mask.shape[1])))
calibration = copy.deepcopy(scaled_up_calBinned_single)

plt.figure()
plt.imshow(calibration, 'rainbow',origin='lower',aspect=10)
plt.title('Exp '+str(limits_angle)+'\n scaled_up_calBinned_single pre-smoothed')
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

np.save('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/sensitivity_7/Sensitivity_1',to_save)
