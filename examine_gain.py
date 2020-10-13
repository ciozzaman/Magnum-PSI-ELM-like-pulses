import numpy as np
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())
# import matplotlib.pyplot as plt
#import .functions
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
from functions.spectools import rotate,do_tilt, binData, get_angle, get_tilt
from functions.Calibrate import do_waveL_Calib, do_Intensity_Calib
from functions.fabio_add import find_nearest_index,multi_gaussian,all_file_names,load_dark,find_index_of_file,get_metadata,movie_from_data,get_angle_no_lines,do_tilt_no_lines,four_point_transform,fix_minimum_signal,get_bin_and_interv_no_lines
from functions.GetSpectrumGeometry import getGeom
from functions.SpectralFit import doSpecFit_single_frame
import os,sys
from PIL import Image
import xarray as xr
import pandas as pd
import copy
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.signal import find_peaks

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


calculate_geometry = True



#fdir = '/home/ff645/GoogleDrive/ff645/YPI - Fabio/Collaboratory/Experimental data/tests/test_data_folder'
#df_log = pd.read_csv('functions/Log/shots_2.csv',index_col=0)
#df_settings = pd.read_csv('functions/Log/settin	gs_2.csv',index_col=0)

fdir = '/home/ffederic/work/Collaboratory/test/experimental_data'
df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_1.csv',index_col=0)
df_settings = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_1.csv',index_col=0)

# merge_ID=3
merge_ID = 20

if calculate_geometry:
	geom=getGeom(merge_ID,df_settings,df_log,fdir=fdir)
	print(geom)
else:
	geom = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
	geom.loc[0] = [0.01759,np.nan,29.202126,53.647099,53.647099]
geom_null = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
geom_null.loc[0] = [0,0,geom['binInterv'],geom['bin00a'],geom['bin00b']]


waveLcoefs = do_waveL_Calib(merge_ID,df_settings,df_log,geom,fdir=fdir)

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


# binnedSens = do_Intensity_Calib(merge_ID,df_settings,df_log,df_calLog,df_sphere,geom,waveLcoefs,waves_to_calibrate=['Hb'],fdir=fdir)
#
# binnedSens = np.ones(np.shape(binnedSens))


if merge_ID<=31:
	binnedSens = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/Sensitivity_1.npy')




# Here I put multiple pictures together to form a full image in time

merge_ID_target = 3
time_resolution_scan = False
time_resolution_scan_improved = True
time_resolution_extra_skip = 0

started=0
rows_range_for_interp = geom['binInterv'][0]/3 # rows that I use for interpolation (box with twice this side length, not sphere)
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
merge_time_window=[-1,4]
data_sum=0
# if ((not time_resolution_scan and not os.path.exists('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_merge_tot.nc')) or (time_resolution_scan and not os.path.exists('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.nc'))):

plt.figure()
all_j=[75,76]
for j in all_j:
	data_all=[]

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
	for index,filename in enumerate(filenames):
		if (index<first_pulse_at_this_frame-1 or index>(first_pulse_at_this_frame+number_of_pulses-1)):
			continue
		elif (index-first_pulse_at_this_frame+2) in bad_pulses_indexes:
			continue
		if time_resolution_scan:
			if index%(1+time_resolution_extra_skip)!=0:
				continue	# The purpose of this is to test how things change for different time skippings
		print(filename)
		fname = fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0/'+filename
		im = Image.open(fname)
		data = np.array(im)

		# data = data * Gain[index]

		data_all.append(data)

	data_mean = np.mean(data_all,axis=(0))
	noise = str(np.around(np.mean(data_mean[:,1000:1300], axis=(0,1)),decimals=3))
	plt.plot(np.mean(data_mean,axis=(1)),label='Multiplier from Gain='+str(Gain[0])+', noise='+str(noise));plt.pause(0.001)
plt.legend(loc='best')
plt.title('Averaged counts for different gain exposure');plt.pause(0.001)





data_all=[]
time_all=[]
plt.figure()
all_j=[128,157,73]
for i,j in enumerate(all_j):
	data_all.append([])
	time_all.append([])

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
	for index,filename in enumerate(filenames):
		if (index<first_pulse_at_this_frame-1 or index>(first_pulse_at_this_frame+number_of_pulses-1)):
			continue
		elif (index-first_pulse_at_this_frame+2) in bad_pulses_indexes:
			continue
		if time_resolution_scan:
			if index%(1+time_resolution_extra_skip)!=0:
				continue	# The purpose of this is to test how things change for different time skippings
		print(filename)
		fname = fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0/'+filename
		im = Image.open(fname)
		data = np.array(im)
		data = data - dataDark
		# data = data * Gain[index]

		time_first_row = -CB_to_OES_initial_delay - (index - first_pulse_at_this_frame + 1) * incremental_step

		data_all[i].append(data)
		time_all[i].append(time_first_row)

data_t_equal_times = [[],[],[]]
for index_0,time in enumerate(time_all[0]):
	if ( (time in time_all[1]) and (time in time_all[2]) ):
		data_t_equal_times[0].append(data_all[0][index_0])
		index_1 = (np.array(time==time_all[1])).argmax()
		data_t_equal_times[1].append(data_all[1][index_1])
		index_2 = (np.array(time==time_all[2])).argmax()
		data_t_equal_times[2].append(data_all[2][index_1])
data_t_equal_times = np.array(data_t_equal_times)
plt.figure();plt.imshow(data_t_equal_times[0,0],'rainbow');plt.colorbar();plt.pause(0.001)
plt.figure();plt.imshow(data_t_equal_times[1,0],'rainbow');plt.colorbar();plt.pause(0.001)
plt.figure();plt.plot(data_t_equal_times[0,0,:,1393],label='Gain=1-Sensitivity');plt.pause(0.001)
plt.plot(data_t_equal_times[1,0,:,1393]*2,label='Gain=2-Balance x2');plt.pause(0.001)
plt.plot(data_t_equal_times[2,0,:,1393]*4,label='Gain=3-Full well x4');plt.pause(0.001)
plt.legend(loc='best')
plt.title('Copmarison of same plasma for different gain');plt.pause(0.001)


plt.figure();plt.plot(data_t_equal_times[0,0,508],label='Gain=1-Sensitivity');plt.pause(0.001)
plt.plot(data_t_equal_times[1,0,508]*2,label='Gain=2-Balance x2');plt.pause(0.001)
plt.plot(data_t_equal_times[2,0,508]*4,label='Gain=3-Full well x4');plt.pause(0.001)
plt.legend(loc='best')
plt.title('Copmarison of same spectra for different gain');plt.pause(0.001)

