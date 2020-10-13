import numpy as np
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())
# import matplotlib.pyplot as plt
#import .functions
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
from functions.spectools import rotate,do_tilt, binData, get_angle, get_tilt,get_angle_2
from functions.Calibrate import do_waveL_Calib, do_Intensity_Calib
from functions.fabio_add import find_nearest_index,multi_gaussian,all_file_names,load_dark,find_index_of_file,get_metadata,movie_from_data,get_angle_no_lines,do_tilt_no_lines,four_point_transform,fix_minimum_signal,fix_minimum_signal2,get_bin_and_interv_no_lines, examine_current_trace
from functions.GetSpectrumGeometry import getGeom
from functions.SpectralFit import doSpecFit_single_frame
from functions.GaussFitData import doLateralfit_time_tependent,doLateralfit_single
import collections

import os,sys
from PIL import Image
import xarray as xr
import pandas as pd
import copy
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.signal import find_peaks

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

# calculate_geometry = False
merge_ID_target = 18
for i in range(10):
	print('.')
print('Starting to work on merge number'+str(merge_ID_target))
for i in range(10):
	print('.')


if (merge_ID_target<=26 or merge_ID_target==53):
	binnedSens = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/Sensitivity_1.npy')
	# limits_angle = [17,26]
	# limits_tilt = [17, 26]
	# limits_wave = [17, 26]
	limits_angle = [0,0]
	limits_tilt = [0,0]
	limits_wave = [0,0]
	angle = 0.0534850538519
	tilt = [29.3051619433, 0.0015142337977, 47.6592283252]
	waveLcoefs = np.ones((2, 3)) * np.nan
	waveLcoefs[1] = [-2.21580743e-06, 1.07315743e-01, 3.40712312e+02]
elif (merge_ID_target<=31):
	binnedSens = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/Sensitivity_1.npy')
	limits_angle = [0,0]
	limits_tilt = [0,0]
	angle = 0.053485
	tilt = [29.391335 , 0.001987 , 46.810327]
	limits_wave = [0,0]
	waveLcoefs = np.ones((2, 3)) * np.nan
	waveLcoefs[1] = [-2.21580743e-06  , 1.07315743e-01  , 3.40712312e+02]
elif (merge_ID_target<=39):
	binnedSens = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/Sensitivity_3.npy')
	limits_angle = [32,39]
	limits_tilt = [32,39]
	limits_wave = [0,0]
	waveLcoefs = np.ones((2, 3)) * np.nan
	waveLcoefs[1] = [-1.39629946e-06, 1.09550955e-01, 3.49466935e+02]  # from examine_sensitivity.py
elif (merge_ID_target<=52 or merge_ID_target==54):
	binnedSens = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/Sensitivity_2.npy')
	limits_angle = [40,52]
	limits_tilt = [40,52]
	limits_wave = [32,39]
	waveLcoefs = np.ones((2, 3)) * np.nan
	waveLcoefs[1] = [-1.39629946e-06, 1.09550955e-01, 3.49466935e+02]  # from examine_sensitivity.py
	# geom.loc[0] = [-0.098421420944948226,4.35085277e-03,2.80792602e+01,4.57449437e+01,4.57449437e+01]	# from analysis_filtered_data.py
	# geom_store = copy.deepcopy(geom)
elif (merge_ID_target<=64 and merge_ID_target>=58):
	binnedSens = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/Sensitivity_2.npy')
	limits_angle = [40,52]
	limits_tilt = [40,52]
	limits_wave = [0,0]
	waveLcoefs = np.ones((2, 3)) * np.nan
	waveLcoefs[1] = [-2.42966723e-06 ,  1.11437517e-01 ,  3.48984758e+02]
	# waveLcoefs[1] = [-1.39629946e-06, 1.09550955e-01, 3.49466935e+02]  # from examine_sensitivity.py
	# geom.loc[0] = [-0.098421420944948226,4.35085277e-03,2.80792602e+01,4.57449437e+01,4.57449437e+01]	# from analysis_filtered_data.py
	# geom_store = copy.deepcopy(geom)
elif (merge_ID_target <= 100):
	binnedSens = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/Sensitivity_4.npy')
	# limits_angle = [66,84]
	# limits_tilt = [66,84]
	# limits_wave = [66,84]
	limits_angle = [0,0]
	limits_tilt = [0,0]
	limits_wave = [0,0]
	angle = 0.818308
	tilt = [24.158665, -0.011909, 68.200605]
	waveLcoefs = np.ones((2, 3)) * np.nan
	waveLcoefs[1] = [-2.68742585e-06,   1.30055740e-01,   3.16687350e+02]



geom_null = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
geom_null.loc[0] = [0,0,0,0,0]

if np.sum(limits_angle)!=0:
	angle=[]
	for merge in range(limits_angle[0],limits_angle[1]+1):
		all_j=find_index_of_file(merge,df_settings,df_log)
		for j in all_j:
			dataDark = load_dark(j,df_settings,df_log,fdir,geom_null)
			(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
			type = '.tif'
			filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)
			type = '.txt'
			filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0',type)[0]
			(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir, folder,sequence, untitled,filename_metadata)

			# plt.figure()
			data_all = []
			for index, filename in enumerate(filenames):
				fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + \
						filename
				im = Image.open(fname)
				data = np.array(im)
				data = (data - dataDark) * Gain[index]
				data = fix_minimum_signal2(data)
				data_all.append(data)
			data_all = np.array(data_all)
			data_mean = np.mean(data_all,axis=0)
			try:
				angle.append(get_angle_2(data_mean,nLines=2))
				print(angle)
			except:
				print('FAILED')
	angle = np.array(angle)
	print(angle)
	angle = np.nansum(angle[:,0]/ (angle[:,1]**2))/np.nansum(1/angle[:,1]**2)

if np.sum(limits_tilt)!=0:
	tilt=[]
	for merge in range(limits_tilt[0],limits_tilt[1]+1):
		all_j=find_index_of_file(merge,df_settings,df_log)
		for j in all_j:
			dataDark = load_dark(j,df_settings,df_log,fdir,geom_null)
			(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
			type = '.tif'
			filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)
			type = '.txt'
			filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0',type)[0]
			(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir, folder,sequence, untitled,filename_metadata)
	
			# plt.figure()
			data_all = []
			for index, filename in enumerate(filenames):
				fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + \
						filename
				im = Image.open(fname)
				data = np.array(im)
				data = (data - dataDark) * Gain[index]
				data = fix_minimum_signal2(data)
				data_all.append(data)
			data_all = np.array(data_all)
			data_mean = np.mean(data_all,axis=0)
			data_mean = rotate(data_mean, angle)
			try:
				tilt.append(get_tilt(data_mean, nLines=3))
				print(tilt)
			except:
				print('FAILED')
	tilt = np.array(tilt)
	print(tilt)
	tilt = np.nanmean(tilt,axis=0)

geom = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
geom.loc[0] = [angle, tilt[1], tilt[0], tilt[2], tilt[2]]
geom_store = copy.deepcopy(geom)
print(geom)

if np.sum(limits_wave)!=0:
	waveLcoefs = []
	for merge in range(limits_wave[0], limits_wave[1]+1):
		all_j=find_index_of_file(merge,df_settings,df_log)
		try:
			waveLcoefs.append(do_waveL_Calib(merge, df_settings, df_log, geom, fdir=fdir))
		except:
			print('FAILED')
	waveLcoefs = np.array(waveLcoefs)
	print(waveLcoefs)
	waveLcoefs = np.nanmean(waveLcoefs,axis=0)
print('waveLcoefs')
print(waveLcoefs)

#fdir = '/home/ff645/GoogleDrive/ff645/YPI - Fabio/Collaboratory/Experimental data/tests/test_data_folder'
#df_log = pd.read_csv('functions/Log/shots_2.csv',index_col=0)
#df_settings = pd.read_csv('functions/Log/settin	gs_2.csv',index_col=0)


# merge_ID=3
# merge_ID=35
# merge_ID = 20

# if calculate_geometry:
# 	geom=getGeom(merge_ID,df_settings,df_log,fdir=fdir)
# 	print(geom)
# else:
# 	geom = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
# 	geom.loc[0] = [0.01759,np.nan,29.202126,53.647099,53.647099]	#	from analysis_filtered_data.py
# geom_store = copy.deepcopy(geom)
# geom_store.loc[0] = [0.030500193392356705,2.59472927e-03,2.93804812e+01,4.69007154e+01,4.69007154e+01]
# geom_null = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
# geom_null.loc[0] = [0,0,geom['binInterv'],geom['bin00a'],geom['bin00b']]


# waveLcoefs = do_waveL_Calib(merge_ID,df_settings,df_log,geom,fdir=fdir)

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


# binnedSens = do_Intensity_Calib(merge_ID,df_settings,df_log,2,df_sphere,geom,waveLcoefs,waves_to_calibrate=['Hb'],fdir=fdir)
# if merge_ID<=31:
# 	binnedSens = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/Sensitivity_1.npy')

# binnedSens = np.ones(np.shape(binnedSens))


# j=find_index_of_file(merge_ID,df_settings,df_log)[0]
# (folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
# dataDark = load_dark(j,df_settings,df_log,fdir,geom)
# (folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
# type = '.tif'
# filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
# filename = filenames[50]
# fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
# im = Image.open(fname)
# data = np.array(im)
# data = data - dataDark
# data = rotate(data, geom['angle'])
# data = do_tilt(data, geom['tilt'])
# fit = doSpecFit_single_frame(data,df_settings, df_log, geom, waveLcoefs, binnedSens)

#
#
# # Lines to experiment on accumulation
#
# merge_ID_target = 1
# j=find_index_of_file(merge_ID_target,df_settings,df_log)[0]
# dataDark = load_dark(j,df_settings,df_log,fdir,geom)
# (folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
# pathfiles = '/home/ff645/Documents/Collaboratory/experimental data/test_data_folder/2019-05-01/03/Untitled_15/Pos0'
# type = '.tif'
# filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)
# #filenames = functions.all_file_names(pathfiles, type)
# type = '.txt'
# filename_metadata = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)[0]
# #filename_metadata = functions.all_file_names(pathfiles, type)[0]
# data_sum=0
# nr_images=0
# n=7
# guess=np.array([ 1.39146712e+03,  9.06377678e+00,  7.78942766e+06,
# 	8.83761543e+02,8.23732909e+00,  1.56981342e+06,
# 	6.53138557e+02,  8.16446300e+00,5.71018551e+05,
# 	5.28321275e+02, 8.79800451e+00,  2.31018551e+05,
# 	4.51308821e+02,  7.62315739e+00, 7.71018551e+04,
# 	3.99308821e+02,  7.62315739e+00, 7.71018551e+04,
# 	3.60308821e+02,  7.62315739e+00, 7.71018551e+04,
# 	-1.11721643e+01])
#
# #8.82062049e+02,8.45499096e+00,  1.26082681e+05,
# #	6.51799897e+02,  8.07745821e+00, 3.19871373e+04,
# #	5.26476885e+02, -7.04148448e+00,  7.38037064e+03,
# #	4.47476885e+02, -7.04148448e+00,  7.38037064e+03,
# 	#4.00476885e+02,  4.38037064e+00,  3.65476885e+01,
# 	#3.65148448e+02, -4.04148448e+00,  3.65476885e+01,
# 	#-4.47476885e+02, -700.04148448e+00,  500,
# #        2.90769043e+03])
#
#
# plt.figure()
# fit_all = []
# for filename in filenames:
# 	im = Image.open(os.path.join(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0/', filename))
# 	imarray = np.array(im)
# 	imarray=rotate(imarray,geom['angle'])
# 	imarray=do_tilt(imarray,geom['tilt'])
# 	if nr_images==0:
# 		data_sum=copy.deepcopy(imarray)-dataDark
# 	else:
# 		data_sum+=imarray-dataDark
# 	binnedData,oExpLim = binData(data_sum/(nr_images+1),geom['bin00b'],geom['binInterv'],check_overExp=True)
# 	plt.plot(binnedData.T[:,20]+1000*nr_images+1000,label='accumulation '+str(nr_images+1))
# 	temp1,temp2=curve_fit(multi_gaussian(n), np.linspace(0,np.shape(data_sum)[1]-1,np.shape(data_sum)[1]), binnedData.T[:,20], p0=guess, maxfev=100000)
# 	fit = multi_gaussian(n)(np.linspace(0,np.shape(data_sum)[1]-1,np.shape(data_sum)[1]),*temp1)
# 	plt.plot(fit+1000*nr_images+1000,label='fit, accumulation '+str(nr_images+1))
# 	nr_images+=1
# plt.title('from '+str(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0')+'\n bin 20 of 40, fit of '+str(n)+' gaussians + constant, Dark subtracted')
# plt.legend(loc='best')
# plt.semilogy()
# plt.show()
#
# data=np.divide(data_sum,nr_images)
# data=rotate(data,geom['angle'])
# data=do_tilt(data,geom['tilt'])
# plt.figure()
# plt.title('from '+str(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0')+'\n accumulation of '+str(nr_images)+' measurements, Dark subtracted')
# plt.imshow(data,'rainbow', origin='lower')
# plt.colorbar().set_label('Counts [au]')
# for i in range(40):
# 	plt.plot([0,data.shape[1]],[geom['bin00b']+geom['binInterv']*i]*2,'k')
# plt.show()
# plt.figure();plt.plot(binnedData.T);plt.plot();
#
#
#
#
#
# # Lines to understand at which point the position of the target
# # ceases to have an influence on the steady sate plasma
#
# merge_ID_target = 6
# j=find_index_of_file(merge_ID_target,df_settings,df_log)[0]
# dataDark = load_dark(j,df_settings,df_log,fdir,geom)
# (folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
# pathfiles = '/home/ff645/Documents/Collaboratory/experimental data/test_data_folder/2019-05-01/03/Untitled_15/Pos0'
# type = '.tif'
# filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)
# type = '.txt'
# filename_metadata = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)[0]
# data_sum=[]
# nr_images=0
# n=7
# guess=np.array([ 1.39146712e+03,  9.06377678e+00,  7.78942766e+06,
# 	8.83761543e+02,8.23732909e+00,  1.56981342e+06,
# 	6.53138557e+02,  8.16446300e+00,5.71018551e+05,
# 	5.28321275e+02, 8.79800451e+00,  2.31018551e+05,
# 	4.51308821e+02,  7.62315739e+00, 7.71018551e+04,
# 	3.99308821e+02,  7.62315739e+00, 7.71018551e+04,
# 	3.60308821e+02,  7.62315739e+00, 7.71018551e+04,
# 	-1.11721643e+01])
#
# #8.82062049e+02,8.45499096e+00,  1.26082681e+05,
# #	6.51799897e+02,  8.07745821e+00, 3.19871373e+04,
# #	5.26476885e+02, -7.04148448e+00,  7.38037064e+03,
# #	4.47476885e+02, -7.04148448e+00,  7.38037064e+03,
# 	#4.00476885e+02,  4.38037064e+00,  3.65476885e+01,
# 	#3.65148448e+02, -4.04148448e+00,  3.65476885e+01,
# 	#-4.47476885e+02, -700.04148448e+00,  500,
# #        2.90769043e+03])
#
# #good_ones = np.array([1,2,3,4,5,6,9])
# #good_ones=good_ones+2+1
# plt.figure()
# fit_all = []
# for filename in filenames:
# 	im = Image.open(os.path.join(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0/', filename))
# 	imarray = np.array(im)
# 	imarray=rotate(imarray,geom['angle'])
# 	imarray=do_tilt(imarray,geom['tilt'])
# 	binnedData,oExpLim = binData(imarray,geom['bin00b'],geom['binInterv'],check_overExp=True)
# 	data_sum.append(binnedData.T)
#
# data_sum=np.array(data_sum)
# plt.plot(np.sum(data_sum[:,:,20],axis=(1)),label='accumulation '+str(nr_images+1))
# plt.pause(0.001)
#
#




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
if time_resolution_scan:
	conventional_time_step = 0.01	# ms
else:
	conventional_time_step = 0.05	# ms
interpolation_type = 'quadratic'	# 'linear' or 'quadratic'
type_of_image = '12bit'	# '12bit' or '16bit'
# if type_of_image=='12bit':
row_shift=2*10280/1000000	# ms
# elif type_of_image=='16bit':
# 	print('Row shift to be checked')
# 	exit()
# time_range_for_interp = rows_range_for_interp*row_shift
# merge_time_window=[-1,4]
merge_time_window=[-10,10]
overexposed_treshold = 3600
data_sum=0
merge_values = []
merge_time = []
merge_row = []
merge_Gain = []
merge_overexposed = []
merge_wavelengths = []
if ((not time_resolution_scan and not os.path.exists('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_merge_tot.npz')) or (time_resolution_scan and not os.path.exists('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.npz'))):
# if True:
	all_j=find_index_of_file(merge_ID_target,df_settings,df_log)
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
		if PixelType[-1]==12:
			time_range_for_interp = rows_range_for_interp * row_shift/2
		elif PixelType[-1]==16:
			time_range_for_interp = rows_range_for_interp * row_shift
		(CB_to_OES_initial_delay,incremental_step,first_pulse_at_this_frame,bad_pulses_indexes,number_of_pulses) = df_log.loc[j,['CB_to_OES_initial_delay','incremental_step','first_pulse_at_this_frame','bad_pulses_indexes','number_of_pulses']]
		if bad_pulses_indexes=='':
			bad_pulses_indexes=[0]
		elif (isinstance(bad_pulses_indexes, float) or isinstance(bad_pulses_indexes, int)):
			bad_pulses_indexes=[bad_pulses_indexes]
		else:
			bad_pulses_indexes = bad_pulses_indexes.replace(' ', '').split(',')
			bad_pulses_indexes = list(map(int, bad_pulses_indexes))

		time_of_pulses_present = 0
		if not os.path.exists('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)):
			os.makedirs('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target))
		if isinstance(df_log.loc[j,['current_trace_file']][0],str):
			fname_current_trace = df_log.loc[j,['current_trace_file']][0]
			trash1, trash2, trash3, trash4, trash5, trash6, good_pulses, time_of_pulses = examine_current_trace(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/', fname_current_trace,number_of_pulses)
			# temp = bof[int(first_pulse_at_this_frame-1):int(number_of_pulses+first_pulse_at_this_frame-1)]/1000000000
			# temp -= temp[0]
			# time_of_pulses -= temp
			time_of_pulses -= np.linspace(0,(len(time_of_pulses)-1),(len(time_of_pulses)))*0.1#np.mean(np.diff(bof)/1000000000)	#	I cannot use bof or eof because are too much imprecise
			time_of_pulses = time_of_pulses*1000
			time_of_pulses_present = 1
			plt.figure(figsize=(20, 10))
			plt.plot(good_pulses,time_of_pulses[good_pulses-1])
			plt.plot(good_pulses, time_of_pulses[good_pulses-1],'o',label='original data')
			fit = np.polyfit(good_pulses, time_of_pulses[good_pulses-1], 1)
			plt.plot(good_pulses,np.polyval(fit, good_pulses),'--',label='fit parameters = '+str(fit))
			fit_correction = incremental_step - fit[0]
			time_of_pulses = time_of_pulses + np.linspace(0,(len(time_of_pulses)-1),(len(time_of_pulses)))*fit_correction
			plt.plot(good_pulses, time_of_pulses[good_pulses-1],'o',label='corrected data')
			fit = np.polyfit(good_pulses, time_of_pulses[good_pulses-1], 1)
			plt.plot(good_pulses,np.polyval(fit, good_pulses),'--',label='fit parameters = '+str(fit))
			plt.legend(loc='best')
			plt.title('Time interval between good pulses for '+str(j))
			plt.xlabel('pulse index')
			plt.ylabel('time shift from first pulse [ms]')
			# plt.pause(0.001)
			plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/file_index_' + str(j) + 'time_profile.eps', bbox_inches='tight')
			plt.close()
			print('time interval from camera '+str(np.mean(np.diff(bof)/1000000000))+'ms')
			print('time profile for j='+str(j)+' found')
			print('fitting coefficients found '+str(fit))
			print('time interval of '+str(fit[0]*1000000)+' microseconds')

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
			fname = fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0/'+filename
			im = Image.open(fname)
			data = np.array(im)
			# data=rotate(data,geom_null['angle'])
			# data=do_tilt(data,geom_null['tilt'])
			data =data - dataDark	# I checked that for 16 or 12 bit the dark is always around 100 counts
			data = data*Gain[index]
			# if Gain[index]==1:	#Case of gain in full well
			# 	data = data*8
			# elif Gain[index]==2:
			# 	data = data * 4
			# 09/06/19 section added to fix camera artifacts
			data = fix_minimum_signal2(data)
			data_sum+=data
			data_all.append(data)
			if time_of_pulses_present==1:
				time_first_row = -CB_to_OES_initial_delay-time_of_pulses[index-int(first_pulse_at_this_frame)+1]
			else:
				time_first_row = -CB_to_OES_initial_delay-(index-first_pulse_at_this_frame+1)*incremental_step
			if PixelType[index]==12:
				timesteps = np.linspace(time_first_row, time_first_row + (roi_tr[1] - 1+2) * row_shift, roi_tr[1]+2)
				timesteps = np.sort(timesteps.tolist()+timesteps.tolist())[:roi_tr[1]]
			elif PixelType[index] == 16:
				# timesteps = np.linspace(time_first_row/conventional_time_step, (time_first_row+(roi_tr[1]-1)*row_shift)/conventional_time_step,roi_tr[1])  # timesteps are normalised with the conventional time step in a way to easy the later interpolation
				timesteps = np.linspace(time_first_row,time_first_row + (roi_tr[1] - 1) * row_shift, roi_tr[1])

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
				merge_time = np.array(timesteps)
				merge_row = np.array(row_steps)
				merge_Gain = np.ones((len(data)))*Gain[index]
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
											for z1,value in enumerate(sample):
												if value/merge_Gain[z]>overexposed_treshold:
													sample[z1]=data_to_record[z1]
													print('fixed at wavelength axis pixel '+str(z1))
											merge_values[z] = sample
											merge_Gain[z] = Gain[index]
											merge_overexposed[z] = overexposed[i]

				merge_values = np.append(merge_values,data,axis=0)
				merge_time = np.append(merge_time,timesteps,axis=0)
				merge_row = np.append(merge_row,row_steps,axis=0)
				merge_Gain = np.append(merge_Gain,np.ones((len(data)))*Gain[index],axis=0)
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
		if not os.path.exists('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)):
			os.makedirs('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target))
		ani = coleval.movie_from_data(np.array([np.array(data_all)]), 1000/incremental_step, row_shift, 'Wavelength axis [pixles]', 'Row axis [pixles]','Intersity [au]',extvmin=0,extvmax=np.max(np.array(data_all)[-70:]))
		# plt.show()
		ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/merge'+str(merge_ID_target)+'_scan_n'+str(j)+'_original_data' + '.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
		plt.close()

	if time_resolution_scan:
		np.savez_compressed('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot',merge_values=merge_values,merge_time=merge_time,merge_row=merge_row,merge_Gain=merge_Gain,merge_overexposed=merge_overexposed)

		# np.save(.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.nc')
	else:
		np.savez_compressed('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_merge_tot',merge_values=merge_values,merge_time=merge_time,merge_row=merge_row,merge_Gain=merge_Gain,merge_overexposed=merge_overexposed)

		# merge_tot.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_merge_tot.nc')

	path_filename = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_stats.csv'
	# if not os.path.exists(path_filename):
	file = open(path_filename, 'w')
	writer = csv.writer(file)
	writer.writerow(['Stats about merge ' + str(merge_ID_target)])
		# file.close()

	result = 0
	nLines = 8
	while (result == 0 and nLines > 0):
		try:
			angle = get_angle(data_sum, nLines=nLines)
			print(angle)
			result = 1
			geom['angle'] = np.nansum(np.multiply(angle[0], np.divide(1, np.power(angle[1], 2)))) / np.nansum(np.divide(1, np.power(angle[1], 2)))
			writer.writerow(['Specific angle of ', str(geom['angle'][0]), ' found with nLines=', str(nLines)])
		except:
			nLines -= 1
	# if np.abs(geom['angle'][0])>2*np.abs(geom_store['angle'][0]):
	if np.abs(geom['angle'][0]-geom_store['angle'][0]) > np.abs(geom_store['angle'][0]):
		geom['angle'][0] = geom_store['angle'][0]
		result = 0
		nLines = 8
	if result == 0:
		writer.writerow(['No specific angle found. Used standard of ', str(geom['angle'][0])])
	data_sum = rotate(data_sum, geom['angle'])
	file = open(path_filename, 'r')
	result = 0
	nLines = 8
	tilt = (np.nan,np.nan,np.nan)
	# while ((result == 0 or np.isnan(tilt[1]) or np.isnan(tilt[2])) and nLines > 0 ):
	while ((result == 0 or np.isnan(tilt[0]) or np.isnan(tilt[2])) and nLines > 0):
		try:
			tilt = get_tilt(data_sum, nLines=nLines)
			print(tilt)
			result = 1
			geom['binInterv'] = tilt[0]
			geom['tilt'] = tilt[1]
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
	# else:
	# 	writer.writerow(
	# 		['Specific tilt of ', str(geom['tilt'][0]), ' found with nLines=', str(nLines), 'binInterv',
	# 		 str(geom['binInterv'][0]), 'first bin', str(geom['bin00a'][0])])
	writer.writerow(['wavelength coefficients of' ,str(waveLcoefs)])
	writer.writerow(['the final result will be from' ,str(merge_time_window[0]),' to ',str(merge_time_window[1]),' ms from the beginning of the discharge'])
	writer.writerow(['with a time resolution of '+str(conventional_time_step),' ms'])
	file.close()

else:
	all_j=find_index_of_file(merge_ID_target,df_settings,df_log)
	j = all_j[-1]
	(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
	type = '.txt'
	filename_metadata = \
	all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0',type)[0]
	(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir,folder,sequence,untitled,filename_metadata)
	if PixelType[-1] == 12:
		time_range_for_interp = rows_range_for_interp * row_shift / 2
	elif PixelType[-1] == 16:
		time_range_for_interp = rows_range_for_interp * row_shift
	(CB_to_OES_initial_delay,incremental_step,first_pulse_at_this_frame,bad_pulses_indexes,number_of_pulses) = df_log.loc[j,['CB_to_OES_initial_delay','incremental_step','first_pulse_at_this_frame','bad_pulses_indexes','number_of_pulses']]
	wavelengths = np.linspace(0, roi_tr[0] - 1, roi_tr[0])
	row_steps = np.linspace(0, roi_tr[1] - 1, roi_tr[1])
	data_sum = 0


	all_j = find_index_of_file(merge_ID_target, df_settings, df_log)
	for j in all_j:
		dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
		(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
		type = '.tif'
		filenames = all_file_names(
			fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
		# filenames = functions.all_file_names(pathfiles, type)
		type = '.txt'
		filename_metadata = all_file_names(
			fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)[
			0]
		# filename_metadata = functions.all_file_names(pathfiles, type)[0]
		(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir, folder,
																										 sequence,
																										 untitled,
																										 filename_metadata)
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
			fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(
				untitled) + '/Pos0/' + filename
			im = Image.open(fname)
			data = np.array(im)
			# data=rotate(data,geom_null['angle'])
			# data=do_tilt(data,geom_null['tilt'])
			data = data - dataDark  # I checked that for 16 or 12 bit the dark is always around 100 counts
			data = data * Gain[index]
			# if Gain[index]==1:	#Case of gain in full well
			# 	data = data*8
			# elif Gain[index]==2:
			# 	data = data * 4
			# 09/06/19 section added to fix camera artifacts
			data = fix_minimum_signal2(data)
			data_sum += data
			data_all.append(data)

		if not os.path.exists('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)):
			os.makedirs('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target))
		ani = coleval.movie_from_data(np.array([np.array(data_all)]), 1000/incremental_step, row_shift, 'Wavelength axis [pixles]', 'Row axis [pixles]','Intersity [au]',extvmin=0,extvmax=np.max(np.array(data_all)[-70:]))
		# plt.show()
		ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/merge'+str(merge_ID_target)+'_scan_n'+str(j)+'_original_data' + '.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
		plt.close()

	path_filename = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(
		merge_ID_target) + '_stats.csv'
	# if not os.path.exists(path_filename):
	file = open(path_filename, 'w')
	writer = csv.writer(file)
	writer.writerow(['Stats about merge ' + str(merge_ID_target)])
	# file.close()

	result = 0
	nLines = 8
	while (result == 0 and nLines > 0):
		try:
			angle = get_angle(data_sum, nLines=nLines)
			print(angle)
			result = 1
			geom['angle'] = np.nansum(np.multiply(angle[0], np.divide(1, np.power(angle[1], 2)))) / np.nansum(np.divide(1, np.power(angle[1], 2)))
			writer.writerow(['Specific angle of ', str(geom['angle'][0]), ' found with nLines=', str(nLines)])
		except:
			nLines -= 1
	# if np.abs(geom['angle'][0])>2*np.abs(geom_store['angle'][0]):
	if np.abs(geom['angle'][0]-geom_store['angle'][0]) > np.abs(geom_store['angle'][0]):
		geom['angle'][0] = geom_store['angle'][0]
		result = 0
	if result == 0:
		writer.writerow(['No specific angle found. Used standard of ', str(geom['angle'][0])])
	data_sum = rotate(data_sum, geom['angle'])
	file = open(path_filename, 'r')
	result = 0
	nLines = 8
	tilt = (np.nan,np.nan,np.nan)
	# while ((result == 0 or np.isnan(tilt[1]) or np.isnan(tilt[2])) and nLines > 0 ):
	while ((result == 0 or np.isnan(tilt[0]) or np.isnan(tilt[2])) and nLines > 0):
		try:
			tilt = get_tilt(data_sum, nLines=nLines)
			print(tilt)
			result = 1
			geom['binInterv'] = tilt[0]
			geom['tilt'] = tilt[1]
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
	# else:
	# 	writer.writerow(
	# 		['Specific tilt of ', str(geom['tilt'][0]), ' found with nLines=', str(nLines), 'binInterv',
	# 		 str(geom['binInterv'][0]), 'first bin', str(geom['bin00a'][0])])
	writer.writerow(['wavelength coefficients of' ,str(waveLcoefs)])
	writer.writerow(
		['the final result will be from', str(merge_time_window[0]), ' to ', str(merge_time_window[1]),
		 ' ms from the beginning of the discharge'])
	writer.writerow(['with a time resolution of ' + str(conventional_time_step), ' ms'])
	file.close()

	if time_resolution_scan:
		# merge_tot = xr.open_dataarray('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.nc')
		merge_values = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) +'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.npz')['merge_values']
		merge_time = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) +'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.npz')['merge_time']
		merge_row = np.load(
			'/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) +'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.npz')[
			'merge_row']
		merge_Gain = np.load(
			'/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) +'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.npz')[
			'merge_Gain']
		merge_overexposed = np.load(
			'/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) +'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.npz')[
			'merge_overexposed']
	else:
		merge_values = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_merge_tot.npz')['merge_values']
		merge_time = np.load(
			'/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_merge_tot.npz')[
			'merge_time']
		merge_row = np.load(
			'/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_merge_tot.npz')[
			'merge_row']
		merge_Gain = np.load(
			'/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_merge_tot.npz')[
			'merge_Gain']
		merge_overexposed = np.load(
			'/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_merge_tot.npz')[
			'merge_overexposed']


		# merge_tot = xr.open_dataarray('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_merge_tot.nc')

	# ani = coleval.movie_from_data(np.array([composed_array.values]), 1000/conventional_time_step, row_shift, 'Wavelength axis [pixles]', 'Row axis [pixles]','Intersity [au]')
	# # plt.show()
	# ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_composed_array' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# I will reconstruct a full scan of 10ns steps from 0 to rows*10ns and all rows

new_timesteps = np.linspace(merge_time_window[0]+0.5, merge_time_window[1]-0.5,int((merge_time_window[1]-0.5-(merge_time_window[0]+0.5))/conventional_time_step+1))
# composed_array=xr.DataArray(np.zeros((len(new_timesteps),roi_tr[1],roi_tr[0])), dims=['time', 'row', 'wavelength_axis'],coords={'time': new_timesteps,'row': row_steps,'wavelength_axis': wavelengths},name='interpolated array')
# composed_array = merge_tot.interp_like(composed_array,kwargs={'fill_value': 0.0})
# this seems not to work so I do it manually
composed_array = np.zeros((len(new_timesteps),roi_tr[1],np.shape(merge_values)[1]))

interpolation_borders_expanded = np.zeros((len(new_timesteps),roi_tr[1]))
if ((not time_resolution_scan and not os.path.exists('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_composed_array.npy')) or (time_resolution_scan and not os.path.exists('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_composed_array.npy'))):

	grade_of_interpolation = 3	#this is the exponent used for the weights

	# merge_tot = merge_tot.sortby('time_row_to_start_pulse')
	for i,interpolated_time in enumerate(new_timesteps):
		additional_time_range = 0
		if (not time_resolution_scan or time_resolution_scan_improved):
			while (np.sum(np.abs(merge_time-interpolated_time)<time_range_for_interp+additional_time_range) < 4*roi_tr[1] and additional_time_range<time_range_for_interp):	# range max doubled
				additional_time_range += time_range_for_interp/10
				# sample = merge_tot.sel(time_row_to_start_pulse=slice(interpolated_time - time_range_for_interp-additional_time_range,
				# 													 interpolated_time + time_range_for_interp+additional_time_range))
				print('interpolated_time '+str(interpolated_time)+' increased of '+str(additional_time_range))
				interpolation_borders_expanded[i]+=np.ones_like(interpolation_borders_expanded[i])
		if  np.sum(np.abs(merge_time-interpolated_time)<time_range_for_interp+additional_time_range)==0:
			continue
		sample_values_1 = merge_values[np.abs(merge_time-interpolated_time)<time_range_for_interp+additional_time_range]
		sample_time_1 = merge_time[np.abs(merge_time-interpolated_time)<time_range_for_interp+additional_time_range]
		sample_row_1 = merge_row[np.abs(merge_time - interpolated_time) < time_range_for_interp+additional_time_range]

		# sample = sample.sortby('row')
		for j, interpolated_row in enumerate(row_steps):
			# weight = 1 / np.sqrt(0.00001+((sample_time - interpolated_time)/ row_shift ) ** 2 + (sample_row - interpolated_row) ** 2)
			# row_selection = np.abs(sample_row - interpolated_row) < rows_range_for_interp + additional_row_range
			# row_selection = (sample.row>float(interpolated_row-rows_range_for_interp)).astype('int') * (sample.row<float(interpolated_row+rows_range_for_interp)).astype('int')
			additional_row_range = 0
			if (not time_resolution_scan or time_resolution_scan_improved):
				while (np.sum(np.abs(sample_row_1 - interpolated_row) < rows_range_for_interp + additional_row_range) < 2 and additional_row_range<rows_range_for_interp):	# range max doubled
					additional_row_range +=rows_range_for_interp/10
					# row_selection = (sample.row > float(interpolated_row - rows_range_for_interp-additional_row_range)).astype('int') * (
					# 			sample.row < float(interpolated_row + rows_range_for_interp+additional_row_range)).astype('int')
					# print(sample.row)
					# print(sample)
					print('interpolated_time '+str(interpolated_time)+', interpolated_row '+str(interpolated_row)+' increased of ' + str(additional_row_range))
					print('total number of rows '+str(len(sample_time_1)))
					interpolation_borders_expanded[i,j] += 1
			if np.sum(np.abs(sample_row_1 - interpolated_row) < rows_range_for_interp + additional_row_range) < 1:
				continue
			# elif np.sum(row_selection) > 10:
			# 	weight_row = row_selection*(weight)
			# else:
			# 	weight_row = weight
			sample_values_2 = sample_values_1[np.abs(sample_row_1 - interpolated_row) < rows_range_for_interp + additional_row_range]
			sample_time_2 = sample_time_1[np.abs(sample_row_1 - interpolated_row) < rows_range_for_interp + additional_row_range]
			sample_row_2 = sample_row_1[np.abs(sample_row_1 - interpolated_row) < rows_range_for_interp + additional_row_range]
			weight_row = 1 / np.sqrt(0.00001 + (np.abs(sample_time_2 - interpolated_time) / row_shift) ** grade_of_interpolation + np.abs(sample_row_2 - interpolated_row) ** grade_of_interpolation)
			# weight_row = row_selection * weight
			# weight_row_filter=(weight_row.values).tolist()
			# for index,value in enumerate(weight_row_filter):
			# 	if value==np.nan:
			# 		weight_row_filter.pop(index)
			# weight_row=np.array(weight_row)
			# sample_weighted = sample_values_2*weight_row
			weight_row = weight_row.reshape((len(weight_row), 1))
			sample_weighted = np.nansum(sample_values_2*weight_row,axis=0)/np.nansum(weight_row)
			composed_array[i,j]=sample_weighted

	plt.figure(figsize=(20, 10))
	plt.imshow(interpolation_borders_expanded, origin='lower',extent=[0,np.shape(interpolation_borders_expanded)[1]-1,np.min(new_timesteps),np.max(new_timesteps)],aspect=100)
	plt.colorbar()
	plt.title('Record of the row and times that required the interpolation borders expanded for good statistics')
	plt.xlabel('Index of the row')
	plt.ylabel('Time from pulse [ms]')
	# plt.pause(0.001)
	plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/' + 'interpolation_borders_expanded.eps', bbox_inches='tight')
	plt.close()


	if time_resolution_scan:
		if time_resolution_scan_improved:
			np.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'improved_row_composed_array',composed_array)
			# composed_array.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'improved_row_composed_array.nc')
		else:
			np.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target) + '_skip_of_' + str(time_resolution_extra_skip + 1) + 'row_composed_array',composed_array)
			#
			# composed_array.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_skip_of_' + str(time_resolution_extra_skip + 1) + 'row_composed_array.nc')
		ani = coleval.movie_from_data(np.array([composed_array.values]), 1 / conventional_time_step, row_shift,'Wavelength axis [pixles]', 'Row axis [pixles]', 'Intersity [au]')
		ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_composed_array' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

	else:
		np.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_composed_array', composed_array)
		np.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_interpolation_borders_expanded',interpolation_borders_expanded)

		# composed_array.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_composed_array.nc')
		# ani = coleval.movie_from_data(np.array([composed_array.values]), 1 / conventional_time_step, row_shift,'Wavelength axis [pixles]', 'Row axis [pixles]', 'Intersity [au]')
		# ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_composed_array' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

else:
	if time_resolution_scan:
		composed_array = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_composed_array.npy')
	else:
		composed_array = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_composed_array.npy')

ani = coleval.movie_from_data(np.array([composed_array]), 1000/conventional_time_step, row_shift, 'Wavelength axis [pixles]', 'Row axis [pixles]','Intersity [au]',extvmin=0)
# plt.show()
ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/merge'+str(merge_ID_target)+'_composed_array' + '.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
plt.close()


# else:
# 	all_j=find_index_of_file(merge_ID_target,df_settings,df_log)
# 	j = all_j[-1]
# 	(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
# 	type = '.txt'
# 	filename_metadata = \
# 	all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0',type)[0]
# 	(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir,folder,sequence,untitled,filename_metadata)
# 	(CB_to_OES_initial_delay,incremental_step,first_pulse_at_this_frame,bad_pulses_indexes,number_of_pulses) = df_log.loc[j,['CB_to_OES_initial_delay','incremental_step','first_pulse_at_this_frame','bad_pulses_indexes','number_of_pulses']]
# 	new_timesteps = np.linspace(merge_time_window[0]+0.5, merge_time_window[1]-0.5,int((merge_time_window[1]-0.5-(merge_time_window[0]+0.5))/conventional_time_step+1))
# 	wavelengths = np.linspace(0, roi_tr[0] - 1, roi_tr[0])
# 	row_steps = np.linspace(0, roi_tr[1] - 1, roi_tr[1])
# 	composed_array = xr.open_dataarray('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_composed_array.nc')
# 	ani = coleval.movie_from_data(np.array([composed_array.values]), 1000/conventional_time_step, row_shift, 'Wavelength axis [pixles]', 'Row axis [pixles]','Intersity [au]')
# 	# plt.show()
# 	ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_composed_array' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])


# if ((not time_resolution_scan and not os.path.exists('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_binned_data.nc')) or (time_resolution_scan and not os.path.exists('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_binned_data.nc'))):
if True:
	frame=rotate(composed_array[0],geom['angle'])
	frame=do_tilt(frame,geom['tilt'])
	wavelengths = np.linspace(0, np.shape(frame)[1] - 1, np.shape(frame)[1])

	bin_steps = np.linspace(0,39,40)
	# binned_data=xr.DataArray(np.zeros((len(new_timesteps),40,np.shape(frame)[1])), dims=['time', 'bin', 'wavelength_axis'],coords={'time': new_timesteps,'bin': bin_steps,'wavelength_axis': wavelengths},name='interpolated array')
	binned_data = np.zeros((len(new_timesteps), 40, np.shape(frame)[1]))


	for index,time in enumerate(new_timesteps):
		frame = composed_array[index]
		frame = np.nan_to_num(frame)
		frame=rotate(frame,geom['angle'])
		frame=do_tilt(frame,geom['tilt'])
		binnedData = binData(frame,geom['bin00b'],geom['binInterv'],check_overExp=False)
		# binnedData = binnedData - min(np.min(binnedData), 0)
		binned_data[index]=binnedData

	if time_resolution_scan:
		if time_resolution_scan_improved:
			np.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_skip_of_' + str(time_resolution_extra_skip + 1) + 'improved_row_binned_data',binned_data)
			# binned_data.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_skip_of_' + str(time_resolution_extra_skip + 1) + 'improved_row_binned_data.nc')
		else:
			np.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(
				merge_ID_target) + '_skip_of_' + str(time_resolution_extra_skip + 1) + 'row_binned_data',
					binned_data)
			# binned_data.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_binned_data.nc')
		# ani = coleval.movie_from_data(np.array([binned_data.values]), 1000/conventional_time_step, row_shift, 'Wavelength axis [pixles]', 'Bin [au]','Intersity [au]')
		# ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_binned_data' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
	else:
		np.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_binned_data', binned_data)

		# binned_data.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_binned_data.nc')
		# ani = coleval.movie_from_data(np.array([binned_data.values]), 1000/conventional_time_step, row_shift, 'Wavelength axis [pixles]', 'Bin [au]','Intersity [au]')
		# ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_binned_data' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
else:
	if time_resolution_scan:
		binned_data = xr.open_dataarray('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_skip_of_' + str(time_resolution_extra_skip + 1) + 'row_binned_data.npy')
	else:
		binned_data = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_binned_data.npy')

#composed_array = np.zeros((roi_tr[1],roi_tr[1],roi_tr[0]))
#for i in range(np.shape(composed_array)[0]):
#	for j in range(np.shape(composed_array)[1]):
#		data_to_interpolate = merge.sel(time_row_to_start_pulse=slice(i*conventional_time_step-time_range_for_interp,i*conventional_time_step-time_range_for_interp),row=slice(i-rows_range_for_interp,i-rows_range_for_interp))
fit = doSpecFit_single_frame(np.array(binned_data[0]), df_settings, df_log, geom, waveLcoefs, binnedSens)

all_fits=[]
for index,time in enumerate(new_timesteps):
	print('fit of '+str(time)+'ms')
	try:
		fit = doSpecFit_single_frame(np.array(binned_data[index]),df_settings, df_log, geom, waveLcoefs, binnedSens)
		print('ok')
	except:
		fit=np.zeros((40,9))*np.nan
	all_fits.append(np.array(fit))
all_fits=np.array(all_fits).astype(float)

if time_resolution_scan:
	if time_resolution_scan_improved:
		np.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'improved_row_all_fits',all_fits)
	else:
		np.save(
			'/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_skip_of_' + str(time_resolution_extra_skip + 1) + 'row_all_fits', all_fits)
	ani = coleval.movie_from_data(np.array([all_fits]), 1000/conventional_time_step, row_shift, 'Transition from Hb' , 'Bin [au]' , 'Intersity [au]',extvmin=0,extvmax=100000)
	ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_all_fits' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
else:
	np.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_all_fits',all_fits)
	# ani = coleval.movie_from_data(np.array([all_fits]), 1000/conventional_time_step, row_shift, 'Transition from Hb' , 'Bin [au]' , 'Intersity [au]',extvmin=0,extvmax=100000)
	# ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_all_fits' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])



dx = 18 / 40 * (50.5 / 27.4) / 1e3
xx = np.arange(40) * dx  # m
xn = np.linspace(0, max(xx), 1000)
r = np.linspace(-max(xx) / 2, max(xx) / 2, 1000)
r=r[::10]



doLateralfit_time_tependent(df_settings, all_fits, merge_ID_target,np.min(new_timesteps),np.max(new_timesteps),conventional_time_step,dx,xx,r)



# Data from wikipedia
energy_difference = np.array([1.89,2.55,2.86,3.03,3.13,3.19,3.23,3.26, 3.29 ])	#eV
# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
statistical_weigth = np.array([32,50,72,98,128,162,200,242,288])	#gi-gk
einstein_coeff = np.array([8.4193e-2,2.53044e-2,9.7320e-3,4.3889e-3,2.2148e-3,1.2156e-3,7.1225e-4,4.3972e-4,2.8337e-4])*1e8	#1/s
J_to_eV = 6.242e18
# Used formula 2.3 in Rion Barrois thesys, 2017
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise']

inverted_profiles = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/inverted_profiles.npy')
sample_time_step = []
for time in [-2, 0, 0.1, 0.4, 0.5, 0.7, 1, 2, 4]:	#times in ms that I want to take a better look at
	sample_time_step.append(int((np.abs(new_timesteps - time)).argmin()))
sample_radious=[]
for radious in [0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.009, 0.0012]:	#radious in m that I want to take a better look at
	sample_radious.append(int((np.abs(r - radious)).argmin()))
try:
	for time_step in sample_time_step:
		# time_step=19
		plt.figure(figsize=(20, 10))
		to_plot_all = []
		for index, loc in enumerate(r[sample_radious]):
			if (loc > 0 and loc < 0.015):
				to_plot = np.divide(np.pi * 4 * inverted_profiles[time_step, :, index],
									statistical_weigth * einstein_coeff * energy_difference / J_to_eV)
				if np.sum(to_plot > 0) > 4:
					to_plot_all.append(to_plot > 0)
		# to_plot_all.append(to_plot)
		to_plot_all = np.array(to_plot_all)
		relative_index = int(np.max(
			(np.sum(to_plot_all, axis=(0)) == np.max(np.sum(to_plot_all, axis=(0)))) * np.linspace(1, len(to_plot),
																								   len(to_plot)))) - 1
		for index, loc in enumerate(r[sample_radious]):
			if (loc > 0 and loc < 0.015):
				to_plot = np.divide(np.pi * 4 * inverted_profiles[time_step, :, index],statistical_weigth * einstein_coeff * energy_difference / J_to_eV)
				plt.plot(energy_difference, to_plot / np.min(to_plot[relative_index]),label='axial pos=' + str(np.around(loc, decimals=4)))
		plt.legend(loc='best')
		plt.title('Boltzmann plot at ' + str(np.around(new_timesteps[time_step], decimals=2)) + 'ms')
		plt.semilogy()
		plt.semilogx()
		plt.xlabel('trensition energy [eV]')
		plt.ylabel('relative population density scaled to the min value [au]')
		plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/Bplot_relative_time_' + str(np.around(new_timesteps[time_step], decimals=2)) + '.eps',bbox_inches='tight')
		plt.close()

		plt.figure(figsize=(20, 10))
		to_plot_all = []
		for index, loc in enumerate(r[sample_radious]):
			if (loc > 0 and loc < 0.015):
				to_plot = np.divide(np.pi * 4 * inverted_profiles[time_step, :, index],
									statistical_weigth * einstein_coeff * energy_difference / J_to_eV)
				plt.plot(energy_difference, to_plot, label='axial pos=' + str(np.around(loc, decimals=4)))
		plt.legend(loc='best')
		plt.title('Boltzmann plot at ' + str(np.around(new_timesteps[time_step], decimals=2)) + 'ms')
		plt.semilogy()
		plt.semilogx()
		plt.xlabel('trensition energy [eV]')
		plt.ylabel('population density scaled to the min value [#/m^3]')
		plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(
			merge_ID_target) + '/Bplot_absolute_time_' + str(np.around(new_timesteps[time_step], decimals=2)) + '.eps',
					bbox_inches='tight')
		plt.close()
except:
	print('failed merge' + str(merge_ID_target))
	plt.close()

for index, loc in enumerate(sample_radious):
	plt.figure(figsize=(20, 10))
	reference_point = inverted_profiles[:, 0, loc].argmax()
	for iR in range(np.shape(inverted_profiles)[1]):
		plt.plot(new_timesteps, inverted_profiles[:, iR, loc] / inverted_profiles[reference_point, iR, loc], color[iR],linewidth=0.3, label='line ' + str(iR + 4))
	plt.title('Relative line intensity at radious' + str(np.around(r[loc], decimals=3)) + 'mm')
	# plt.semilogy()
	plt.xlabel('time [ms]')
	plt.ylabel('relative intensity scaled to the max value [au]')
	plt.legend(loc='best')
	plt.ylim(bottom=0)
	plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/rel_intens_r_' + str(np.around(r[loc], decimals=3)) + '.eps', bbox_inches='tight')
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
# 	plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/line_ratio_intens_r_' + str(np.around(r[loc], decimals=3)) + '.eps', bbox_inches='tight')
# 	plt.close()

for index, loc in enumerate(sample_radious):
	plt.figure(figsize=(20, 10))
	reference_point = inverted_profiles[:, 0, loc].argmax()
	max_ratio = np.max(inverted_profiles[reference_point, :-2, loc] / inverted_profiles[reference_point, 1:-1, loc])
	print(max_ratio)
	for iR in range(np.shape(inverted_profiles)[1] - 1):
		# plt.plot(new_timesteps, inverted_profiles[:, iR, loc] / inverted_profiles[:, iR + 1, loc], color[iR]+'+',label='line ratio ' + str(iR + 4) + '/' + str(iR + 4 + 1))
		# plt.plot(new_timesteps, inverted_profiles[:, iR + 1, loc] / inverted_profiles[:, iR, loc], color=color[iR],marker='+', linewidth=0.4, label='line ratio ' + str(iR + 4 + 1) + '/' + str(iR + 4))
		plt.plot(new_timesteps, inverted_profiles[:, iR + 1, loc] / inverted_profiles[:, iR, loc], color=color[iR],marker='+', label='line ratio ' + str(iR + 4 + 1) + '/' + str(iR + 4))
	plt.title('Line ratio at radious' + str(np.around(r[loc], decimals=3)) + 'mm')
	# plt.semilogy()
	plt.xlabel('time [ms]')
	plt.ylabel('relative intensity [au]')
	plt.legend(loc='best')
	plt.ylim(bottom=0)
	plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/line_ratio_intens_r_' + str(np.around(r[loc], decimals=3)) + '.eps',bbox_inches='tight')
	plt.close()

for iR in range(np.shape(inverted_profiles)[1] - 1):
	plt.figure(figsize=(20, 10))
	# plt.plot(new_timesteps, inverted_profiles[:, iR, loc] / inverted_profiles[:, iR + 1, loc], color[iR]+'+',label='line ratio ' + str(iR + 4) + '/' + str(iR + 4 + 1))
	plt.imshow(inverted_profiles[:, iR + 1] / inverted_profiles[:, iR], 'rainbow',vmin=0,vmax=1,extent=[min(xx),max(xx),np.max(new_timesteps),np.min(new_timesteps)],aspect=0.02)
	plt.title('Line ratio ' +str(iR + 4 + 1) +'/' + str(iR + 4))
	plt.xlabel('radial location [m]')
	plt.ylabel('time [ms]')
	plt.colorbar()
	plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(	merge_ID_target) + '/inverted_line_ratio_map_' +str(iR + 4 + 1) +'-' + str(iR + 4) + '.eps',bbox_inches='tight')
	plt.close()

for iR in range(np.shape(inverted_profiles)[1] - 1):
	plt.figure(figsize=(20, 10))
	# plt.plot(new_timesteps, inverted_profiles[:, iR, loc] / inverted_profiles[:, iR + 1, loc], color[iR]+'+',label='line ratio ' + str(iR + 4) + '/' + str(iR + 4 + 1))
	plt.imshow(all_fits[:,:, iR + 1] / all_fits[:,:, iR], 'rainbow',vmin=0,vmax=1,extent=[min(xx),max(xx),np.max(new_timesteps),np.min(new_timesteps)],aspect=0.02)
	plt.title('Line ratio ' +str(iR + 4 + 1) +'/' + str(iR + 4))
	plt.xlabel('radial location [m]')
	plt.ylabel('time [ms]')
	plt.colorbar()
	plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(	merge_ID_target) + '/line_ratio_map_' +str(iR + 4 + 1) +'-' + str(iR + 4) + '.eps',bbox_inches='tight')
	plt.close()


time_step = int((np.abs(new_timesteps+1)).argmin())
all_fits_ss = all_fits[:time_step]
all_fits_ss[all_fits_ss==0]=np.nan
all_fits_ss = np.nanmean(all_fits_ss,axis=0)
doLateralfit_single(df_settings, all_fits_ss, merge_ID_target,dx,xx,r)




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
metadata= open(os.path.join(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', filename_metadata), 'r')
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

























