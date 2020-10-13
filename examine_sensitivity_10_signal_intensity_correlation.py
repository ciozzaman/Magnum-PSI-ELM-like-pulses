import numpy as np
import matplotlib.pyplot as plt
#import .functions
import os,sys
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

from numpy.core.multiarray import ndarray

os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
from functions.spectools import get_angle,rotate,get_tilt,do_tilt,getFirstBin, binData, get_angle_2
from functions.fabio_add import find_nearest_index,multi_gaussian,all_file_names,load_dark,find_index_of_file,get_metadata,examine_current_trace,movie_from_data,get_bin_and_interv_no_lines, four_point_transform, fix_minimum_signal, do_tilt_no_lines, fix_minimum_signal2, fix_minimum_signal3,get_angle_no_lines,do_tilt_no_lines,get_bin_and_interv_no_lines,get_bin_and_interv_specific_wavelength,apply_proportionality_calibration,fix_minimum_signal_calibration
from functions.Calibrate import do_waveL_Calib
from PIL import Image
import xarray as xr
import pandas as pd
import copy
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.signal import gaussian
from skimage.transform import resize
from scipy.signal import medfilt
from scipy.ndimage import median_filter


exec(open("/home/ffederic/work/analysis_scripts/scripts/profile_smoothing.py").read())


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
# binnedSens = np.ones(np.shape(binnedSens))


where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/signal_vs_intensity/Plot_'

geom_null = pd.DataFrame(columns=['angle', 'tilt', 'binInterv', 'bin00a', 'bin00b'])
geom_null.loc[0] = [0, 0, 0, 0, 0]



if False:		# EXPLORATORY STUFF
	angle = []
	# all_j = [319,320,321,322,323,324,325,326,327]
	# current = [1.99E-05,1.50E-05,1.00E-05,5.00E-06,2.50E-06,1.00E-06,5.00E-07,2.50E-07,9.99E-08,1.06E-09]
	all_j = [353,354,355,356]

	data_mean_all = []
	data_std_all = []
	for j in all_j:
		# dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
		(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
		type = '.tif'
		filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
		type = '.txt'
		filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)[0]
		(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir, folder,sequence,untitled,filename_metadata)

		# plt.figure()
		data_all = []
		for index, filename in enumerate(filenames):
			fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
			im = Image.open(fname)
			data = np.array(im)
			# data = (data - dataDark) * Gain[index]
			# data = fix_minimum_signal2(data)
			data_all.append(data)
		# data_all = (np.array(data_all)- dataDark) * Gain[index]
		data_mean = np.mean(data_all, axis=0)
		data_std = np.std(data_all, axis=0)
		data_mean_all.append(data_mean)
		data_std_all.append(data_std)
	data_mean_all = np.array(data_mean_all)
	data_std_all = np.array(data_std_all)

	flag=coleval.find_dead_pixels(np.array([data_all]),treshold_for_bad_std=7,treshold_for_bad_difference=100)
	positive_flag = np.logical_not((flag/np.max(flag)).astype('bool'))
	plt.figure();
	plt.imshow(flag, 'rainbow',norm=LogNorm());
	plt.colorbar();
	plt.pause(0.01)

	plt.figure();plt.plot(np.array(current)/current[0],np.sum(data_mean_all[:,400:1300],axis=2));plt.semilogx();plt.semilogy();plt.pause(0.01)
	plt.figure();plt.plot(np.array(current)/current[0],data_mean_all[:,1000]);plt.semilogx();plt.semilogy();plt.pause(0.01)

	pos0,pos1=[1236,546]
	counter = collections.Counter((data_all[:, pos0, pos1]).astype(int) - (np.mean(data_all[:,pos0, pos1])))
	x = list(counter.keys())
	y = np.array(list(counter.values())) / len(data_all[:, pos0, pos1])
	y = np.array([y for _, y in sorted(zip(x, y))])
	x = np.sort(x)
	plt.figure();
	plt.plot(x,y);
	plt.pause(0.01)


	for index in range(len(data_mean_all)):
		plt.figure();
		plt.title('Labsphere current '+str(current[index])+'A')
		plt.imshow(data_mean_all[index], 'rainbow')
		plt.colorbar().set_label('mean [au]')
		plt.pause(0.01)

	for index in range(len(data_mean_all)-1):
		plt.figure();
		plt.title('Labsphere current '+str(current[index])+'A - '+str(current[index+1])+' A')
		plt.imshow(data_std_all[index]-data_std_all[index+1], 'rainbow')
		plt.colorbar().set_label('std [au]')
		plt.pause(0.01)



	row_sum=np.mean(data_mean_all[:,345:1330],axis=1)
	plt.figure();
	plt.plot(row_sum[0],row_sum[0]*(current[1]/current[0]),'C1')
	plt.plot(row_sum[0],row_sum[1],'+C1')
	# plt.plot(row_sum[1],row_sum[2]*(current[2]/current[1]),'C2')
	plt.plot(row_sum[1],row_sum[2],'+C2')
	# plt.plot(row_sum[2],row_sum[3]*(current[3]/current[2]),'C3')
	plt.plot(row_sum[2],row_sum[3],'+C3')
	# plt.plot(row_sum[3],row_sum[4]*(current[4]/current[3]),'C4')
	plt.plot(row_sum[3],row_sum[4],'+C4')
	plt.plot(row_sum[4],row_sum[5],'+C5')
	plt.plot(row_sum[5],row_sum[6],'+C6')
	plt.plot(row_sum[6],row_sum[7],'+C7')
	plt.pause(0.01)


	plt.figure();
	index=6
	plt.imshow(medfilt(data_mean_all[index],3), 'rainbow');
	plt.colorbar();
	plt.title(index)
	plt.pause(0.01)

	plt.figure();
	plt.imshow(medfilt(data_mean_all[0],3)/medfilt(data_mean_all[1],3), 'rainbow');
	plt.colorbar();
	plt.pause(0.01)


	temp = data_mean_all[:,345:1330].reshape((len(data_mean_all),np.shape(data_mean_all[0,345:1330])[-1]*np.shape(data_mean_all[0,345:1330])[-2])).T
	# max_value = np.sort(temp[positive_flag[345:1330].flatten(),0])
	# min_value = np.array([x for _, x in sorted(zip(temp[positive_flag[345:1330].flatten(),0], temp[positive_flag[345:1330].flatten(),1]))])
	max_value = np.sort(temp[:,0])
	min_value = np.array([x for _, x in sorted(zip(temp[:,0], temp[:,1]))])
	plt.figure()
	plt.plot(max_value,min_value,'+')
	plt.pause(0.01)
	max_value = np.sort(temp[positive_flag[345:1330].flatten(),1])
	min_value = np.array([x for _, x in sorted(zip(temp[positive_flag[345:1330].flatten(),1], temp[positive_flag[345:1330].flatten(),2]))])
	# plt.figure()
	plt.plot(max_value,min_value,'+')
	plt.pause(0.01)
	max_value = np.sort(temp[positive_flag[345:1330].flatten(),2])
	min_value = np.array([x for _, x in sorted(zip(temp[positive_flag[345:1330].flatten(),2], temp[positive_flag[345:1330].flatten(),3]))])
	# plt.figure()
	plt.plot(max_value,min_value,'+')
	plt.pause(0.01)



	max_value = np.sort(medfilt(data_mean_all[0,345:1330],3).flatten())
	min_value = np.array([x for _, x in sorted(zip(medfilt(data_mean_all[0,345:1330],3).flatten(), medfilt(data_mean_all[1,345:1330],3).flatten()))])
	plt.figure()
	plt.plot(max_value,min_value,'+')
	plt.pause(0.01)
	max_value = np.sort(medfilt(data_mean_all[1,345:1330],3).flatten())
	min_value = np.array([x for _, x in sorted(zip(medfilt(data_mean_all[1,345:1330],3).flatten(), medfilt(data_mean_all[2,345:1330],3).flatten()))])
	# plt.figure()
	plt.plot(max_value,min_value,'+')
	plt.pause(0.01)
	max_value = np.sort(medfilt(data_mean_all[2,345:1330],3).flatten())
	min_value = np.array([x for _, x in sorted(zip(medfilt(data_mean_all[2,345:1330],3).flatten(), medfilt(data_mean_all[3,345:1330],3).flatten()))])
	# plt.figure()
	plt.plot(max_value,min_value,'+')
	plt.pause(0.01)
	max_value = np.sort(medfilt(data_mean_all[-2,345:1330],3).flatten())
	min_value = np.array([x for _, x in sorted(zip(medfilt(data_mean_all[-2,345:1330],3).flatten(), medfilt(data_mean_all[-1,345:1330],3).flatten()))])
	# plt.figure()
	plt.plot(max_value,min_value,'+')
	plt.pause(0.01)
	max_value = np.sort(medfilt(data_mean_all[-4,345:1330],3).flatten())
	min_value = np.array([x for _, x in sorted(zip(medfilt(data_mean_all[-4,345:1330],3).flatten(), medfilt(data_mean_all[-3,345:1330],3).flatten()))])
	# plt.figure()
	plt.plot(max_value,min_value,'+')
	plt.pause(0.01)




	temp = data_mean_all[:,345:1330].reshape((len(data_mean_all),np.shape(data_mean_all[:,345:1330])[-1]*np.shape(data_mean_all[:,345:1330])[-2])).T
	max_value = np.sort(temp[:,-1])
	min_value = np.array([x for _, x in sorted(zip(temp[:,-1], temp[:,-2]))])
	smoothed_temp = []
	smoothed_temp_coord = []
	sigma = []
	resolution = 20
	current_start = np.min(max_value)
	while current_start<np.max(max_value):
		current_end = current_start + resolution
		if np.sum(np.logical_and(max_value<=current_end,max_value>=current_start))>1:
			smoothed_temp_coord.append(current_start + resolution / 2)
			smoothed_temp.append(np.mean(min_value[np.logical_and(max_value<=current_end,max_value>=current_start)]))
			sigma.append(1 / (np.sum(np.logical_and(max_value <= current_end, max_value >= current_start)) ** 0.5))
		current_start+=resolution
	smoothed_temp_coord = np.array(smoothed_temp_coord)
	smoothed_temp = np.array(smoothed_temp)
	sigma=np.array(sigma)
	# plt.figure()
	plt.errorbar(smoothed_temp_coord,smoothed_temp,yerr=100*sigma)
	plt.grid()
	plt.pause(0.01)


	plt.figure();
	plt.plot(data_mean_all[0,:,-1],np.min(medfilt(data_mean_all[0],[1,3]),axis=1));
	plt.pause(0.01)


	plt.figure();
	row=668
	for index in range(len(data_mean_all)):
		plt.plot(data_mean_all[index,row,:]-np.min(medfilt(data_mean_all[index,row,:],5)),label=str(index));
	plt.title('row '+str(row))
	plt.legend(loc='best')
	plt.pause(0.01)

	plt.figure();
	row=668
	for index in range(len(data_mean_all)):
		plt.plot(data_mean_all[index,row,:],label=str(index));
	plt.title('row '+str(row))
	plt.legend(loc='best')
	plt.pause(0.01)


	plt.figure();
	row=657
	plt.plot(data_mean_all[:,row,-1],data_mean_all[:,row,-1]-np.min(medfilt(data_mean_all,[1,1,5])[:,row],axis=1));
	plt.title('row '+str(row))
	plt.pause(0.01)

	plt.figure();
	row=716
	plt.plot(data_mean_all[:,row,-1],np.min(medfilt(data_mean_all,[1,1,5])[:,row],axis=1));
	plt.title('row '+str(row))
	plt.pause(0.01)

	plt.figure();
	index = 1
	plt.plot(medfilt(data_mean_all[index,400:1300],[1,5])[:,-3],np.min(medfilt(data_mean_all[index,400:1300,:-100],[1,5]),axis=1),'+');
	plt.title('obscuration step '+str(index+1))
	plt.pause(0.01)



	plt.figure();
	value=103
	for row in range(330,800):
		if (np.abs(medfilt(data_mean_all[0,row],5)[-3]-value)<1 and np.min(medfilt(data_mean_all[-1,row],5))>99):
		# if np.abs(medfilt(data_mean_all[0,row],5)[-3]-110)<2:
			plt.plot(medfilt(data_mean_all[:,row],[1,5])[:,-3],np.min(medfilt(data_mean_all[:,row],[1,5]),axis=1),label=row)
	plt.legend(loc='best')
	plt.pause(0.01)

	plt.figure();
	for row in range(330,800):
		if np.min(medfilt(data_mean_all[-1,row],5))>99.7:
		# if np.abs(medfilt(data_mean_all[0,row],5)[-3]-110)<2:
		# 	plt.plot(medfilt(data_mean_all[0,row],5)[-3],np.mean(data_mean_all[-1,row,:200])-np.mean(data_mean_all[0,row,:200]),'+')
		# 	plt.plot(medfilt(data_mean_all[1,row],5)[-3],np.mean(data_mean_all[-1,row,:200])-np.mean(data_mean_all[1,row,:200]),'+')
			plt.plot(medfilt(data_mean_all[2,row],5)[-3],np.mean(data_mean_all[-1,row,:200])-np.mean(data_mean_all[2,row,:200]),'+')
	# plt.legend(loc='best')
	plt.pause(0.01)



elif False:		# important plots: difference with the lens on the optical axis and not

	all_j = [353, 354, 355, 356]

	data_mean_all = []
	data_std_all = []
	for j in all_j:
		# dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
		(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
		type = '.tif'
		filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
		type = '.txt'
		filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0',type)[0]
		(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir, folder,sequence, untitled,filename_metadata)

		# plt.figure()
		data_all = []
		for index, filename in enumerate(filenames):
			fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
			im = Image.open(fname)
			data = np.array(im)
			# data = (data - dataDark) * Gain[index]
			# data = fix_minimum_signal2(data)
			data_all.append(data)
		# data_all = (np.array(data_all)- dataDark) * Gain[index]
		data_mean = np.mean(data_all, axis=0)
		data_std = np.std(data_all, axis=0)
		data_mean_all.append(data_mean)
		data_std_all.append(data_std)
	data_mean_all = np.array(data_mean_all)
	data_std_all = np.array(data_std_all)

	plt.figure();
	plt.imshow(data_mean_all[1], 'rainbow',vmax=1000);
	plt.colorbar();
	plt.pause(0.01)

	print((np.sum(data_mean_all[2, 330:1335]) - 100)/(np.sum(data_mean_all[0,330:1335])-100))			#0.860049947863
	print((np.sum(data_mean_all[2, 335:1330,950:]) - 100)/(np.sum(data_mean_all[0,335:1330,950:])-100))	#0.857111769667

	print((np.sum(data_mean_all[3, 332:1335]) - 100)/(np.sum(data_mean_all[1,332:1335])-100))			#0.865329331694
	print((np.sum(data_mean_all[3, 335:1330,500:]) - 100)/(np.sum(data_mean_all[1,335:1330,500:])-100)) #0.864498478873
	print((np.sum(data_mean_all[3, 335:1330,950:]) - 100)/(np.sum(data_mean_all[1,335:1330,950:])-100))	#0.863530515117


elif True:		# important plots: difference with and without the filter on the right of the image

	# all_j = [357,358,359,360,361,362,363]
	all_j = [357,363]
	# all_j = [367,366,369,368]	# in time 0.02 ms
	# all_j = [375,374,377,376]	# in time 0.08 ms
	# all_j = [381,380,383,382]	# in time 0.32 ms


	data_mean_all = []
	data_std_all = []
	for j in all_j:
		# dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
		(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
		type = '.tif'
		filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
		type = '.txt'
		filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)[0]
		(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir, folder,sequence,untitled,filename_metadata)

		# plt.figure()
		data_all = []
		for index, filename in enumerate(filenames):
			fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
			im = Image.open(fname)
			data = np.array(im)
			# data = (data - dataDark) * Gain[index]
			# data = fix_minimum_signal2(data)
			data_all.append(data)
		# data_all = (np.array(data_all)- dataDark) * Gain[index]
		data_mean = np.mean(data_all, axis=0)
		data_std = np.std(data_all, axis=0)
		data_mean_all.append(data_mean)
		data_std_all.append(data_std)
	data_mean_all = np.array(data_mean_all)
	data_std_all = np.array(data_std_all)
	data_mean_all_median = medfilt(data_mean_all,[1,1,3])


	plt.figure();
	plt.imshow(data_mean_all[1], 'rainbow');
	plt.colorbar();
	plt.pause(0.01)

	if True:
		plt.figure();
		plt.plot(np.mean(data_mean_all_median[0,815:825],axis=0),label='out');
		plt.plot(np.mean(data_mean_all_median[1,815:825],axis=0),label='1');
		plt.plot(np.mean(data_mean_all_median[2,815:825],axis=0),label='2');
		plt.plot(np.mean(data_mean_all_median[3,815:825],axis=0),label='3');
		plt.plot(np.mean(data_mean_all_median[4,815:825],axis=0),label='4');
		plt.plot(np.mean(data_mean_all_median[5,815:825],axis=0),label='5');
		plt.plot(np.mean(data_mean_all_median[6,815:825],axis=0),label='in');
		plt.legend(loc='best');
		plt.title('Calibration data with right section progressively being obscured')
		plt.xlabel('wavelength axis')
		plt.ylabel('average counts over lit area [au]')
		plt.pause(0.01)
	elif False:
		plt.figure();
		plt.plot(data_mean_all_median[0,820],label='high intensity, shade out');
		plt.plot(data_mean_all_median[1,820],label='high intensity, shade in\ndelta [820,0:200]='+str(np.mean(data_mean_all_median[1,820,:200])-np.mean(data_mean_all_median[0,820,:200])));
		plt.plot(data_mean_all_median[2,820],label='low intensity, shade out');
		plt.plot(data_mean_all_median[3,820],label='low intensity, shade in\ndelta [820,0:200]='+str(np.mean(data_mean_all_median[3,820,:200])-np.mean(data_mean_all_median[2,820,:200])));
		plt.legend(loc='best');
		plt.title('Calibration data with right section progressively being obscured')
		plt.xlabel('wavelength axis')
		plt.ylabel('average counts over lit area [au]')
		plt.pause(0.01)


	plt.figure();
	plt.plot(data_mean_all[-1,500],label='last');
	plt.plot(data_mean_all[-1,900],label='mid');
	plt.plot(data_mean_all[-1,1200],label='first');
	plt.plot(np.mean(data_mean_all[-1],axis=0),label='mean');
	plt.legend(loc='best');
	plt.title('Calibration data with right section obscured\nnot lit section from column ~1450')
	plt.pause(0.01)


	plt.figure();
	plt.plot(data_mean_all[-1,:,1600],label='last');
	plt.plot(data_mean_all[-1,:,1550],label='mid');
	plt.plot(data_mean_all[-1,:,1500],label='first');
	plt.plot(np.mean(data_mean_all[-1,:,1500:],axis=1),label='mean from 1500');
	plt.legend(loc='best');
	plt.pause(0.01)

	plt.figure();
	to_plot = np.min(medfilt(data_mean_all[0],[1,21])[:,20:-20],axis=-1)
	plt.plot(to_plot);
	# to_plot[325:1338]+=6
	# plt.plot(to_plot);
	to_plot2 = np.mean(data_mean_all_median[-1,:,:200],axis=-1)-np.mean(data_mean_all_median[0,:,:200],axis=-1)
	plt.plot(to_plot2)
	plt.pause(0.01)
	plt.figure();
	plt.plot(to_plot,to_plot2,'+');
	plt.plot([95,95.4,99.9],[5.9,5.9,1])
	plt.pause(0.01)
	interpolator = interp1d([90,95.4,99.9,100,110],[5.9,5.9,1,0,0])
	# plt.figure();
	image_corrected = []
	for index in range(1608):
		image_corrected.append(data_mean_all[0,index]+interpolator(to_plot[index]))
	# plt.imshow(image_corrected)
	# plt.colorbar()
	# plt.pause(0.01)
	# plt.figure();
	# plt.imshow(data_mean_all[-1])
	# plt.colorbar()
	# plt.pause(0.01)
	plt.figure();
	plt.imshow(data_mean_all[-1]-image_corrected,vmin=-4,vmax=4)
	plt.colorbar()
	plt.pause(0.01)
	to_plot3 = np.mean((data_mean_all[-1]-image_corrected)[:,1100:1300],axis=-1)
	plt.figure();
	plt.plot(to_plot);
	plt.plot(to_plot2)
	plt.plot(to_plot3)
	plt.plot(medfilt(data_mean_all[0],[1,21])[:,-22])
	plt.pause(0.01)

	plt.figure();
	plt.plot(medfilt(data_mean_all[0],[1,21])[:,-1],to_plot2)
	plt.pause(0.01)

	plt.figure();
	plt.plot(np.mean(data_mean_all_median[0,:,-100:],axis=-1)-np.min(medfilt(data_mean_all[0],[1,21])[:,22:-22],axis=-1),label='10')
	plt.plot(np.mean(data_mean_all_median[0,:,-10:],axis=-1)-np.min(medfilt(data_mean_all[0],[1,21])[:,22:-22],axis=-1),label='11')
	plt.plot(np.mean(data_mean_all_median[0,:,-2:],axis=-1)-np.min(medfilt(data_mean_all[0],[1,21])[:,22:-22],axis=-1),label='12')
	plt.plot(np.mean(data_mean_all[0,:,-100:-50],axis=-1)-np.min(medfilt(data_mean_all[0],[1,21])[:,22:-22],axis=-1),label='13')
	plt.plot(np.mean(data_mean_all[0,:,-10:-5],axis=-1)-np.min(medfilt(data_mean_all[0],[1,21])[:,22:-22],axis=-1),label='14')
	plt.plot(np.mean(data_mean_all[0,:,-10:-8],axis=-1)-np.min(medfilt(data_mean_all[0],[1,21])[:,22:-22],axis=-1),label='141')
	plt.plot(np.mean(data_mean_all[0,:,-5:-2],axis=-1)-np.min(medfilt(data_mean_all[0],[1,21])[:,22:-22],axis=-1),label='15')
	plt.plot(np.min(data_mean_all_median[-1,:,:1000],axis=-1)-100,label='2')
	plt.plot(np.min(data_mean_all_median[0,:,:1000],axis=-1)-100,label='3')
	plt.plot(np.mean(data_mean_all_median[-1,:,:200],axis=-1)-np.mean(data_mean_all_median[0,:,:200],axis=-1),label='4')
	plt.legend(loc='best');
	plt.pause(0.01)

	plt.figure();
	plt.plot(np.mean(data_mean_all_median[-1,:,:200],axis=-1)-np.mean(data_mean_all_median[0,:,:200],axis=-1),np.mean(data_mean_all_median[0,:,-100:],axis=-1)-np.min(medfilt(data_mean_all[0],[1,21])[:,22:-22],axis=-1),'+')
	# plt.plot(np.mean(data_mean_all_median[-1,:,:200],axis=-1)-np.mean(data_mean_all_median[0,:,:200],axis=-1),np.mean(data_mean_all_median[0,:,-50:],axis=-1)-np.min(medfilt(data_mean_all[0],[1,21])[:,22:-22],axis=-1),'+')
	# plt.plot(np.mean(data_mean_all_median[-1,:,:200],axis=-1)-np.mean(data_mean_all_median[0,:,:200],axis=-1),np.mean(data_mean_all_median[0,:,-10:],axis=-1)-np.min(medfilt(data_mean_all[0],[1,21])[:,22:-22],axis=-1),'o')
	# plt.plot(np.mean(data_mean_all_median[-1,:,:200],axis=-1)-np.mean(data_mean_all_median[0,:,:200],axis=-1),np.mean(data_mean_all_median[0,:,-5:],axis=-1)-np.min(medfilt(data_mean_all[0],[1,21])[:,22:-22],axis=-1),'v')
	plt.pause(0.01)

	plt.figure();
	plt.plot(np.mean(data_mean_all_median[-1,:,:200],axis=-1)-np.mean(data_mean_all_median[0,:,:200],axis=-1),data_mean_all[0,:,-1]-np.min(medfilt(data_mean_all[0],[1,21])[:,22:-22],axis=-1),'+')
	plt.pause(0.01)

	plt.figure();
	plt.plot(np.mean(data_mean_all_median[-1,:,:200],axis=-1)-np.mean(data_mean_all_median[0,:,:200],axis=-1),np.mean(data_all[0,:,-20:],axis=-1)-np.min(medfilt(data_all[0],[1,21])[:,22:-22],axis=-1),'+')
	plt.pause(0.01)



	if False:
		plt.figure();
		to_plot = data_mean_all[0]
		# to_plot[325:1338]+=6
		plt.imshow(to_plot);
		plt.colorbar()
		plt.pause(0.01)


		plt.figure()
		for average_from in [0]:
			averaging_all=[]
			minimum_all = []
			for averaging in range(1,500):
				averaging_all.append(averaging)
				minimum_all.append(np.min(np.median(data_all[1][:,-averaging:],axis=-1)))
			plt.plot(averaging_all,minimum_all,label='average_from '+str(average_from))
		plt.legend(loc='best')
		plt.pause(0.01)

		tilt = get_bin_and_interv_no_lines(data_mean_all[-1],bininterv_est=15)
		to_plot = np.array([np.linspace(tilt[0],tilt[0]+tilt[1]*40,41).tolist(),np.linspace(tilt[0],tilt[0]+tilt[1]*40,41).tolist()])


		plt.figure();
		for row in range(1608):
			# if np.min(medfilt(data_mean_all[-1,row],5))>99.7:
			# # if np.abs(medfilt(data_mean_all[0,row],5)[-3]-110)<2:
			# 	plt.plot(medfilt(data_mean_all[0,row],5)[-3],np.mean(data_mean_all[-1,row,:200])-np.mean(data_mean_all[0,row,:200]),'+')
			# 	plt.plot(medfilt(data_mean_all[1,row],5)[-3],np.mean(data_mean_all[-1,row,:200])-np.mean(data_mean_all[1,row,:200]),'+')
			plt.plot(medfilt(data_mean_all[0,row],5)[-3],np.mean(data_mean_all[1,row,:200])-np.mean(data_mean_all[0,row,:200]),'+')
		# plt.legend(loc='best')
		plt.pause(0.01)

		plt.figure();
		# plt.tricontourf(np.linspace(1,1608,1608), medfilt(data_mean_all[0,:,:20],[1,5])[:,-3], np.mean(data_mean_all[1,:,:200],axis=1)-np.mean(data_mean_all[0,:,:200],axis=1), 15, cmap='rainbow')
		plt.scatter(np.linspace(1, 1608, 1608), medfilt(data_mean_all[0, :, -20:], [1, 9])[:, -1],c=np.mean(data_mean_all[1, :, :200], axis=1) - np.mean(data_mean_all[0, :, :200], axis=1), cmap='rainbow')
		# plt.plot(np.linspace(1,1608,1608),medfilt(data_mean_all[0,:,:20],[1,5])[:,-3],'+')
		plt.colorbar()
		plt.pause(0.01)

		angle = get_angle_no_lines(data_mean_all[1],bininterv_est=15)
		frame=rotate(data_mean_all[1],angle)
		tilt_4_points = do_tilt_no_lines(frame,return_4_points=True)
		data_mean_all2 = []
		frame=rotate(data_mean_all[0],angle)
		data_mean_all2.append(four_point_transform(frame,tilt_4_points))
		frame=rotate(data_mean_all[1],angle)
		data_mean_all2.append(four_point_transform(frame, tilt_4_points))
		data_mean_all2 = np.array(data_mean_all2)

		tilt = get_bin_and_interv_no_lines(data_mean_all2[0],bininterv_est=15)
		to_plot = np.array([np.linspace(tilt[0],tilt[0]+tilt[1]*40,41).tolist(),np.linspace(tilt[0],tilt[0]+tilt[1]*40,41).tolist()])

		plt.figure();
		# plt.tricontourf(np.linspace(1,1608,1608), medfilt(data_mean_all[0,:,:20],[1,5])[:,-3], np.mean(data_mean_all[1,:,:200],axis=1)-np.mean(data_mean_all[0,:,:200],axis=1), 15, cmap='rainbow')
		plt.scatter(np.linspace(1, np.shape(data_mean_all2)[1], np.shape(data_mean_all2)[1]), medfilt(data_mean_all2[0, :, -20:], [1, 9])[:, -3],c=np.mean(data_mean_all2[1, :, :200], axis=1) - np.mean(data_mean_all2[0, :, :200], axis=1), cmap='rainbow')
		# plt.scatter(np.linspace(1, np.shape(data_mean_all2)[1], np.shape(data_mean_all2)[1]), np.mean(data_mean_all2[0, :, :200], axis=1),c=np.mean(data_mean_all2[1, :, :200], axis=1) - np.mean(data_mean_all2[0, :, :200], axis=1), cmap='rainbow')
		# plt.plot(to_plot,np.ones_like(to_plot)*[[np.max(np.mean(data_mean_all2[0, :, :200], axis=1))],[np.min(np.mean(data_mean_all2[0, :, :200], axis=1))]])
		# plt.plot(np.linspace(1,1608,1608),medfilt(data_mean_all[0,:,:20],[1,5])[:,-3],'+')
		plt.colorbar()
		plt.pause(0.01)

		tilt_last_column = get_bin_and_interv_specific_wavelength(fix_minimum_signal2(data_mean_all_median[0]),1608)
		tilt_intermediate_column = get_bin_and_interv_specific_wavelength(fix_minimum_signal2(data_mean_all_median[0]),1200)
		window_of_signal = [np.floor(tilt_last_column[0]).astype('int')-100,np.ceil(tilt_last_column[0]+tilt_last_column[1]*40).astype('int')+100]
		# interpolator = interp1d(np.linspace(0, 1607, 1608), medfilt(data_mean_all[0, :, -30:], [9, 3])[:, -3])
		# for column_averaging in [5,10,15,20,25,30,35,40,45,50,55,60]:

		for column_averaging in [30,50,80,100]:
			# plt.figure()
			for range_rows in [0]:
				temp = []
				for row_averaging in range(-range_rows,range_rows+1):
					# temp.append(np.mean(data_mean_all_median[0, window_of_signal[0]-row_averaging-10:window_of_signal[1]-row_averaging+10, -column_averaging:],axis=-1))
					# temp.append(np.mean(data_mean_all[0, window_of_signal[0]-row_averaging-10:window_of_signal[1]-row_averaging+10, -column_averaging:],axis=-1) - np.mean(data_mean_all[0, window_of_signal[0]-row_averaging-10:window_of_signal[1]-row_averaging+10, :100],axis=-1)+np.mean(data_mean_all[1, window_of_signal[0]-row_averaging-10:window_of_signal[1]-row_averaging+10, :200],axis=-1))
					# temp.append(np.mean(data_mean_all[0, window_of_signal[0]-row_averaging-10:window_of_signal[1]-row_averaging+10, -column_averaging:],axis=-1) - np.mean(data_mean_all[0, window_of_signal[0]-row_averaging-10:window_of_signal[1]-row_averaging+10, :100],axis=-1)+100)
					temp.append(np.mean(data_mean_all[0, window_of_signal[0]-row_averaging-10:window_of_signal[1]-row_averaging+10, -column_averaging:],axis=-1) - np.min(median_filter(data_mean_all[0, window_of_signal[0]-row_averaging-10:window_of_signal[1]-row_averaging+10],[1,101])[:,100:-100],axis=-1)+ np.min(median_filter(data_mean_all[-1, window_of_signal[0]-row_averaging-10:window_of_signal[1]-row_averaging+10,:1200],[1,101])[:,100:-100],axis=-1))
				interpolator = interp1d(np.linspace(window_of_signal[0]-10, window_of_signal[1]-1+10, window_of_signal[1] - window_of_signal[0]+20), np.mean(temp,axis=0))
				# for dx_to_etrapolate_to in [0,100,150,200,250,300,332,350,400]:
				for dx_to_etrapolate_to in [200]:
					# dx_to_etrapolate_to = 100	# number of columns to extrapolate to
					tilt_extapolated_column = [tilt_last_column[0] + (tilt_last_column[0]-tilt_intermediate_column[0])/(1608-1200)*dx_to_etrapolate_to,tilt_last_column[1] + (tilt_last_column[1]-tilt_intermediate_column[1])/(1608-1200)*dx_to_etrapolate_to]
					# equivalent_positions = []
					equivalent_counts = []
					for row in range(window_of_signal[0],window_of_signal[1]):
						# test_row = 1200
						row_distance = np.abs(tilt_extapolated_column[0] + tilt_extapolated_column[1]*np.linspace(-10,50,61) - row)
						interested_LOS = np.array([to_sort for _, to_sort in sorted(zip(row_distance, np.linspace(-10,50,61)))])[:2]
						B = row - (tilt_extapolated_column[0] + min(interested_LOS) *tilt_extapolated_column[1])
						b = B * tilt_last_column[1] / tilt_extapolated_column[1]
						# row_corrisponding_to_test_row = np.around((b + tilt_last_column[0] + min(interested_LOS) *tilt_last_column[1]) + 1e-10)
						row_corrisponding_to_test_row = (b + tilt_last_column[0] + min(interested_LOS) *tilt_last_column[1])
						if row_corrisponding_to_test_row<0:
							row_corrisponding_to_test_row = 0
						if row_corrisponding_to_test_row>1608:
							row_corrisponding_to_test_row = 1608
						# equivalent_positions.append(int(row_corrisponding_to_test_row))
						equivalent_counts.append(interpolator(row_corrisponding_to_test_row))
					equivalent_counts = np.array(equivalent_counts)
			# 		plt.plot(range(window_of_signal[0],window_of_signal[1]),np.array(equivalent_counts)-100,label='equivalent extr dist '+str(dx_to_etrapolate_to))
			#
			# plt.plot(np.mean(data_mean_all[-1,:,:200],axis=-1)-np.mean(data_mean_all[0,:,:200],axis=-1),label='4')
			# plt.legend(loc='best')
			# plt.grid()
			# plt.title('column_averaging '+str(column_averaging))
			# plt.pause(0.01)

					# plt.figure();
					# # plt.scatter(np.linspace(1, 1608, 1608), medfilt(data_mean_all[0, :, -20:], [1, 9])[:, -1][equivalent_positions],c=np.mean(data_mean_all[1, :, :200], axis=1) - np.mean(data_mean_all[0, :, :200], axis=1), cmap='rainbow')
					# # plt.scatter(np.linspace(1, 1608, 1608), equivalent_counts,c=np.mean(data_mean_all[1, :, :200], axis=1) - np.mean(data_mean_all[0, :, :200], axis=1), cmap='rainbow')
					# plt.scatter(np.linspace(window_of_signal[0], window_of_signal[1]-1, window_of_signal[1] - window_of_signal[0]), np.mean(data_mean_all[1, window_of_signal[0]:window_of_signal[1], :200], axis=1) - np.mean(data_mean_all[0, window_of_signal[0]:window_of_signal[1], :200], axis=1),c=equivalent_counts, cmap='rainbow')
					# plt.colorbar()
					# to_plot = np.array([np.linspace(tilt_last_column[0], tilt_last_column[0] + tilt_last_column[1] * 40, 41).tolist(),np.linspace(tilt_last_column[0], tilt_last_column[0] + tilt_last_column[1] * 40, 41).tolist()])
					# # plt.plot(to_plot,np.ones_like(to_plot)*[[np.max(equivalent_counts)],[np.min(equivalent_counts)]])
					# plt.plot(to_plot,np.ones_like(to_plot)*[[np.max(np.mean(data_mean_all[1, :, :200], axis=1) - np.mean(data_mean_all[0, :, :200], axis=1))],[np.min(np.mean(data_mean_all[1, :, :200], axis=1) - np.mean(data_mean_all[0, :, :200], axis=1))]])
					# plt.title('extrapolation distance '+str(dx_to_etrapolate_to))
					# plt.pause(0.01)

					plt.figure();
					for row in range(window_of_signal[0],window_of_signal[1]):
						# if np.min(medfilt(data_mean_all[-1,row],5))>99.7:
						# # if np.abs(medfilt(data_mean_all[0,row],5)[-3]-110)<2:
						# 	plt.plot(medfilt(data_mean_all[0,row],5)[-3],np.mean(data_mean_all[-1,row,:200])-np.mean(data_mean_all[0,row,:200]),'+')
						# 	plt.plot(medfilt(data_mean_all[1,row],5)[-3],np.mean(data_mean_all[-1,row,:200])-np.mean(data_mean_all[1,row,:200]),'+')
						plt.plot(equivalent_counts[row-window_of_signal[0]],np.mean(data_mean_all_median[-1,row,:100])-np.mean(data_mean_all_median[0,row,:100]),'+')

					# plt.legend(loc='best')
					plt.plot([100,106,112],[0,6,6])
					plt.title('extrapolation distance ' + str(dx_to_etrapolate_to)+'\nrange_rows '+str(range_rows)+'\ncolumn_averaging '+str(column_averaging))
					# plt.plot([108,101,100],[np.mean((np.mean(data_mean_all[1,:,:200],axis=1)-np.mean(data_mean_all[0,:,:200],axis=1))[equivalent_counts>104]),np.mean((np.mean(data_mean_all[1,:,:200],axis=1)-np.mean(data_mean_all[0,:,:200],axis=1))[equivalent_counts>104]),0])
					plt.pause(0.01)

		additive_factor = fix_minimum_signal_calibration(data_mean_all[0])

		plt.figure()
		plt.plot(np.mean(data_mean_all_median[-1,:,:100],axis=-1)-np.mean(data_mean_all_median[0,:,:100],axis=-1),label='row shift shaded/not shaded')
		plt.plot(additive_factor,label='additive factor')
		plt.plot(additive_factor-(np.mean(data_mean_all_median[-1,:,:100],axis=-1)-np.mean(data_mean_all_median[0,:,:100],axis=-1)),label='difference')
		plt.xlabel('rows')
		plt.ylabel('counts')
		plt.grid()
		plt.legend(loc='best')
		plt.pause(0.01)

		plt.figure()
		plt.imshow((data_mean_all[0].T+additive_factor).T,'rainbow')
		plt.colorbar()
		plt.pause(0.01)

		additive_factor_1 = fix_minimum_signal_calibration(data,counts_treshold_fixed_increase=106.5)	# I take the firts image without shade
		plt.figure()
		plt.imshow(data,'rainbow',vmax=110)
		plt.colorbar()
		plt.title('original image')
		plt.figure()
		plt.imshow((data.T+additive_factor_1).T,'rainbow',vmax=116)
		plt.colorbar()
		plt.title('negative signal corrected image')
		plt.figure()
		plt.plot(additive_factor,label='additive factor from 1000 frames mean')
		plt.plot(np.mean(data_mean_all_median[-1,:,:100],axis=-1)-np.mean(data_mean_all_median[0,:,:100],axis=-1),label='row shift shaded/not shaded from 1000 frames mean')
		plt.plot(additive_factor-(np.mean(data_mean_all_median[-1,:,:100],axis=-1)-np.mean(data_mean_all_median[0,:,:100],axis=-1)),label='difference from 1000 frames')
		plt.plot(additive_factor_1,label='additive factor from single frame')
		plt.plot(additive_factor_1-(np.mean(data_mean_all_median[-1,:,:100],axis=-1)-np.mean(data_mean_all_median[0,:,:100],axis=-1)),label='difference from single frame')
		plt.grid()
		plt.legend(loc='best')
		plt.pause(0.01)

		plt.figure()
		plt.imshow(apply_proportionality_calibration((data.T+additive_factor_1).T,x_calibration,y_calibration),'rainbow',vmax=5)
		plt.colorbar()
		plt.title('negative signal and proportionality corrected image')
		plt.pause(0.01)



		dx_to_etrapolate_to = 250		# should be the better one
		tilt_last_column = get_bin_and_interv_specific_wavelength(fix_minimum_signal2(data_mean_all[0]),1608)
		tilt_intermediate_column = get_bin_and_interv_specific_wavelength(fix_minimum_signal2(data_mean_all[0]),1200)
		interpolator = interp1d(np.linspace(0, 1607, 1608), medfilt(data_mean_all[0, :, -30:], [9, 3])[:, -3])
		equivalent_counts = []
		for row in range(1608):
			# test_row = 1200
			tilt_extapolated_column = [tilt_last_column[0] + (tilt_last_column[0] - tilt_intermediate_column[0]) / (
						1608 - 1200) * dx_to_etrapolate_to,
									   tilt_last_column[1] + (tilt_last_column[1] - tilt_intermediate_column[1]) / (
												   1608 - 1200) * dx_to_etrapolate_to]
			row_distance = np.abs(
				tilt_extapolated_column[0] + tilt_extapolated_column[1] * np.linspace(-10, 50, 61).astype('int') - row)
			interested_LOS = np.array(
				[to_sort for _, to_sort in sorted(zip(row_distance, np.linspace(-10, 50, 61).astype('int')))])[:2]
			B = row - (tilt_extapolated_column[0] + min(interested_LOS) * tilt_extapolated_column[1])
			b = B * tilt_last_column[1] / tilt_extapolated_column[1]
			row_corrisponding_to_test_row = np.around(
				(b + tilt_last_column[0] + min(interested_LOS) * tilt_last_column[1]) + 1e-10)
			if row_corrisponding_to_test_row < 0:
				row_corrisponding_to_test_row = 0
			if row_corrisponding_to_test_row > 1608:
				row_corrisponding_to_test_row = 1608
			# equivalent_positions.append(int(row_corrisponding_to_test_row))
			equivalent_counts.append(interpolator(row_corrisponding_to_test_row))
		equivalent_counts = np.array(equivalent_counts)


		divisions = 50
		y=np.mean(data_mean_all[1,:,:200],axis=1)-np.mean(data_mean_all[0,:,:200],axis=1)
		x = equivalent_counts[:]
		y = np.array([y for _, y in sorted(zip(x, y))])
		x = np.sort(x)
		dx = (np.max(x)-np.min(x))/divisions
		X=[]
		Y=[]
		for index in range(divisions-1):
			X.append(np.mean(x[np.logical_and(x>=np.min(x)+dx*index,x<np.min(x)+dx*(index+1))]))
			Y.append(np.mean(y[np.logical_and(x>=np.min(x)+dx*index,x<np.min(x)+dx*(index+1))]))

		plt.figure();
		for row in range(1608):
			plt.plot(equivalent_counts[row], np.mean(data_mean_all[1, row, :200]) - np.mean(data_mean_all[0, row, :200]),'+')
		plt.title('extrapolation distance ' + str(dx_to_etrapolate_to))
		plt.plot(X, Y)
		plt.pause(0.01)

		equivalent_counts, np.mean(data_mean_all[1,:,:200],axis=1)-np.mean(data_mean_all[0,:,:200],axis=1)

		np.mean((np.mean(data_mean_all[1,:,:200],axis=1)-np.mean(data_mean_all[0,:,:200],axis=1))[equivalent_counts>104])


elif True:		# EXPLORATORY STUFF: let's look first at the linearity counts/signal with the barrier in

	# all_j = [357,363]#,364,365]
	# all_j = [366,370,371,368,372,373]	#integration time 0.02
	all_j = [374,389,390,376,391,392]	#integration time 0.08
	signal_level = [2.00E-05,1.00E-05,5.00E-06,1.25E-06,3.13E-07,7.80E-08]

	data_mean_all = []
	data_std_all = []
	for j in all_j:
		# dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
		(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
		type = '.tif'
		filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
		type = '.txt'
		filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)[0]
		(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir, folder,sequence,untitled,filename_metadata)

		# plt.figure()
		data_all = []
		for index, filename in enumerate(filenames):
			fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
			im = Image.open(fname)
			data = np.array(im)
			# data = (data - dataDark) * Gain[index]
			# data = fix_minimum_signal2(data)
			data_all.append(data)
		# data_all = (np.array(data_all)- dataDark) * Gain[index]
		data_mean = np.mean(data_all, axis=0)
		data_std = np.std(data_all, axis=0)
		data_mean_all.append(data_mean)
		data_std_all.append(data_std)
	data_mean_all = np.array(data_mean_all)
	data_std_all = np.array(data_std_all)
	data_mean_all_median = medfilt(data_mean_all,[1,3,3])
	dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)


	columns_to_look_at = np.ones((1608))
	for index in range(1608):
		if np.abs(index-921) < 40:
			columns_to_look_at[index]=0
		if np.abs(index-1360) < 100:
			columns_to_look_at[index]=0


	plt.figure()
	plt.plot(signal_level,np.mean((data_mean_all-dataDark)[:,767,1000:1200],axis=-1),'+')
	interpolator = interp1d(np.mean((data_mean_all - dataDark)[:, 767, 1000:1200], axis=-1),signal_level)
	# plt.figure()
	plt.plot(np.array(signal_level)*interpolator(np.mean((data_mean_all-dataDark)[:,500,300:500],axis=-1)[0])/signal_level[0],np.mean((data_mean_all-dataDark)[:,500,300:500],axis=-1),'+')
	plt.plot(np.array(signal_level)*interpolator(np.mean((data_mean_all-dataDark)[:,500,1000:1200],axis=-1)[0])/signal_level[0],np.mean((data_mean_all-dataDark)[:,500,1000:1200],axis=-1),'+')
	plt.plot(np.array(signal_level)*interpolator(np.mean((data_mean_all-dataDark)[:,1000,300:500],axis=-1)[0])/signal_level[0],np.mean((data_mean_all-dataDark)[:,1000,300:500],axis=-1),'+')
	plt.plot(np.array(signal_level)*interpolator(np.mean((data_mean_all-dataDark)[:,1000,1000:1200],axis=-1)[0])/signal_level[0],np.mean((data_mean_all-dataDark)[:,1000,1000:1200],axis=-1),'+')
	plt.pause(0.01)

	plt.figure()
	plt.plot(signal_level,(data_mean_all)[:,670,1371])
	interpolator = interp1d((data_mean_all)[:, 670, 1371],signal_level,fill_value='extrapolate',kind='quadratic')
	for column in range(0,1608,100):
		if columns_to_look_at[column]:
			for row in range(0,1608,50):
				if (data_mean_all)[0, row, column]>102:
					plt.plot(np.array(signal_level) * interpolator((data_mean_all)[0, row, column]) / signal_level[0],(data_mean_all)[:, row, column], '+')
				else:
					print('column '+str(column)+' , row '+str(row)+' skipped')
	plt.pause(0.01)

	plt.figure();
	target = 112
	for column in range(0,1608):
		if columns_to_look_at[column]:
			for row in range(0,1608):
				if data_mean_all_median[0, row, column]//1==target:
					plt.plot(signal_level,data_mean_all_median[:, row, column])
	plt.title('counts for decreasing absolute input signal starting from '+str(target)+' to '+str(target+1))
	plt.xlabel('input signal level')
	plt.ylabel('counts [au]')
	plt.pause(0.01)

	grouping = np.array([[[0]*(len(data_mean_all_median)+1)]]*len(np.unique(data_mean_all_median[0]//1))).astype('float')
	interpolator = interp1d(np.unique(data_mean_all_median[0]//1),np.linspace(0,len(np.unique(data_mean_all_median[0]//1))-1,len(np.unique(data_mean_all_median[0]//1))))
	for column in range(0,1608):
		if columns_to_look_at[column]:
			for row in range(0,1608):
				grouping[int(interpolator(data_mean_all_median[0, row, column]//1))]+=[1.,*data_mean_all_median[:, row, column]]



	grouping_counts = []
	# grouping_max_counts = []
	for index,values in enumerate(grouping):
		if values[0,0]>0:
			grouping_counts.append(values[0,1:]/(values[0,0]))
	grouping_counts = np.array(grouping_counts)

	plt.figure()
	plt.plot(signal_level,grouping_counts[-1])
	interpolator = interp1d(grouping_counts[-1],signal_level,fill_value='extrapolate',kind='quadratic')
	for index,values in enumerate(grouping_counts[:-1]):
		plt.plot(np.array(signal_level) * interpolator(values[0]) / signal_level[0],values, '+')
	plt.title('counts for decreasing absolute input signal\nsignal for lower counts scaled')
	plt.xlabel('input signal level')
	plt.ylabel('counts [au]')
	interpolator2 = interp1d(signal_level[:2],grouping_counts[-1][:2],fill_value='extrapolate',kind='linear')
	plt.plot([*signal_level,0], interpolator2([*signal_level,0]), '--')
	plt.pause(0.01)

	x = cp.deepcopy([signal_level])
	y = [grouping_counts[-1].tolist()]
	for index,values in enumerate(grouping_counts[1:-1]):
		x.append(np.array(signal_level) * interpolator(values[0]) / signal_level[0])
		y.append(values)
	x = np.array(x).flatten()
	y = np.array(y).flatten()
	y = np.array([y for _, y in sorted(zip(x, y))])
	x = np.sort(x)
	x_scaled = x/np.max(x)
	# y_scaled = (y-min(y))/(np.max(y)-np.min(y))

	plt.figure()
	plt.plot(x_scaled,y,'+')
	plt.plot(scipy.signal.savgol_filter(x_scaled,61,2),y_scaled)
	plt.pause(0.01)

	counter = collections.Counter(np.rint(y/0.1)/10)
	y_steps = np.array(list(counter.keys()))
	y_repeats = np.array(list(counter.values()))
	x_collect = np.zeros_like(y_repeats).astype(float)
	y_collect = np.zeros_like(y_repeats).astype(float)
	for i_value,value in enumerate(np.rint(y/0.1)/10):
		# print([i_value,value])
		index = (np.abs(value-y_steps)).argmin()
		# print([index,x_scaled[i_value]])
		x_collect[index] += x_scaled[i_value]
		y_collect[index] += y[i_value]
	x_collect = x_collect /y_repeats
	y_collect = y_collect /y_repeats

	x_collect = np.array([y for _, y in sorted(zip(y_collect, x_collect))])
	y_collect = np.sort(y_collect)
	x_new = scipy.signal.savgol_filter(x_collect,13,3)
	x_new[x_new<=0] = 0
	y_new = scipy.signal.savgol_filter(y_collect,13,3)

	plt.figure()
	plt.plot(x_scaled,y,'+')
	plt.plot(x_collect,y_collect,'o')
	plt.plot(x_collect,y_new)
	plt.plot(x_new,y_collect)
	plt.plot(x_new,y_new)
	interpolator = interp1d(x_new[-4:],y_new[-4:],fill_value='extrapolate')
	plt.plot(x_new,interpolator(x_new),'--')
	plt.pause(0.01)

	x_new = x_new*(np.max(y_new)-100-6)


	# print([x_new])
	x_new = np.array([  0.00000000e+00,   3.46083510e-03,   9.56684309e-03,
		1.70594684e-02,   2.57761828e-02,   3.55544581e-02,
		4.62317660e-02,   5.59213278e-02,   6.69807268e-02,
        8.17019440e-02,   9.48800157e-02,   1.08867572e-01,
        1.26539937e-01,   1.47862928e-01,   1.70641781e-01,
        1.96680259e-01,   2.29750062e-01,   2.61597424e-01,
        2.97144279e-01,   3.35818456e-01,   3.76064626e-01,
        4.13983187e-01,   4.48509988e-01,   5.09560876e-01,
        5.80558303e-01,   6.63453805e-01,   7.62418912e-01,
        8.75887994e-01,   1.00302949e+00,   1.11743190e+00,
        1.25844530e+00,   1.39053127e+00,   1.53356473e+00,
        1.69247606e+00,   1.85230021e+00,   2.03927503e+00,
        2.28270301e+00,   2.52788093e+00,   2.80427804e+00,
        3.12755221e+00,   3.51991834e+00,   3.98092993e+00,
        4.55129171e+00,   5.25340339e+00,   6.10966467e+00,
        7.14247526e+00,   8.37423486e+00])
	# print([np.array([*y_new[:-4],*interpolator(x_new[-4:])])])
	y_new=np.array([  99.92669124,  100.01111895,  100.1028623 ,  100.20040497,
        100.30223065,  100.40682299,  100.51266568,  100.60216576,
        100.7008122 ,  100.81057189,  100.91625673,  101.0346604 ,
        101.17533297,  101.33280789,  101.50243377,  101.67514415,
        101.87235328,  102.07232941,  102.25712598,  102.4330159 ,
        102.63042864,  102.81394914,  102.99309262,  103.2298904 ,
        103.50672001,  103.80246223,  104.13449922,  104.46464549,
        104.83589732,  105.16289484,  105.49768961,  105.82446923,
        106.13690695,  106.46595755,  106.7761337 ,  107.12442244,
        107.54158702,  107.95029766,  108.40142474,  108.87579612,
        109.42426425,  110.00611442,  110.66401282,  111.41286759,
        112.26758685,  113.24307874,  114.35425138])

	plt.figure()
	# plt.plot(x_scaled,y,'+')
	# plt.plot(x_collect,y_collect,'o')
	# plt.plot(x_collect,y_new)
	# plt.plot(x_new,y_collect)
	plt.plot(y_new,x_new,'v',label='oroginal data')
	interpolator = interp1d(x_new[-4:],y_new[-4:],fill_value='extrapolate')
	plt.plot(interpolator(x_new),x_new,'--',label='extrapolated linearity')
	interpolator_1 = interp1d([0,6],[106,112],fill_value='extrapolate')
	plt.plot(np.linspace(80,120,30),interp1d([90,95,*y_new[:-3],*interpolator_1(x_new[-3:])],[0,0,*x_new],fill_value='extrapolate')(np.linspace(80,120,30)),'+',label='extrapolated range with imposed linearities')
	plt.xlabel('original camera counts')
	plt.ylabel('counts based on real signal')
	plt.legend(loc='best')
	plt.pause(0.01)


	# Final set of points for the counts signal conversion of the camera

	y_calibration = np.linspace(80,120,100)
	x_calibration = interp1d([90,95,*y_new[:-5],*interpolator_1(x_new[-5:])],[0,0,*x_new],fill_value='extrapolate')(np.linspace(80,120,100))

	plt.plot(y_calibration,x_calibration,'r',label='final points for calibration')
	plt.legend(loc='best')
	plt.grid()
	plt.pause(0.01)



	# print([x_calibration])
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




	j = 357
	# dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
	(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
	type = '.tif'
	filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
	type = '.txt'
	filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)[0]
	(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir, folder,sequence,untitled,filename_metadata)

	# plt.figure()
	data_all = []
	data_all_non_proportional = []
	for index, filename in enumerate(filenames):
		print(filename)
		fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
		im = Image.open(fname)
		data = np.array(im)
		# data = (data - dataDark) * Gain[index]
		additive_factor = fix_minimum_signal_calibration(data,counts_treshold_fixed_increase=106.5)
		if np.sum(np.isnan(additive_factor))>0:
			print(filename+' correction and use aborted')
			continue
		data = (np.array(data).T + additive_factor).T
		data_all_non_proportional.append(data)
		data = apply_proportionality_calibration(data,x_calibration,y_calibration)
		data_all.append(data)
	# data_all = (np.array(data_all)- dataDark) * Gain[index]
	data_all_non_proportional_mean = np.mean(data_all_non_proportional, axis=0)
	data_all_non_proportional_std = np.std(data_all_non_proportional, axis=0)
	data_mean = np.mean(data_all, axis=0)
	data_std = np.std(data_all, axis=0)



elif False:		# EXPLORATORY STUFF: let's look first at the linearity counts/signal without barrier

	all_j = [367,369]
	signal_level = [2.00E-05,1.25E-06]

	data_mean_all = []
	data_std_all = []
	for j in all_j:
		# dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
		(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
		type = '.tif'
		filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
		type = '.txt'
		filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)[0]
		(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir, folder,sequence,untitled,filename_metadata)

		# plt.figure()
		data_all = []
		for index, filename in enumerate(filenames):
			fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
			im = Image.open(fname)
			data = np.array(im)
			# data = (data - dataDark) * Gain[index]
			# data = fix_minimum_signal2(data)
			data_all.append(data)
		# data_all = (np.array(data_all)- dataDark) * Gain[index]
		data_mean = np.mean(data_all, axis=0)
		data_std = np.std(data_all, axis=0)
		data_mean_all.append(data_mean)
		data_std_all.append(data_std)
	data_mean_all = np.array(data_mean_all)
	data_std_all = np.array(data_std_all)

	dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
	data_mean_all_net = data_mean_all - dataDark

	columns_to_look_at = np.ones((1608))
	for index in range(1608):
		if np.abs(index-921) < 40:
			columns_to_look_at[index]=0
		if np.abs(index-1360) < 100:
			columns_to_look_at[index]=0


	plt.figure()
	plt.plot(signal_level,np.mean((data_mean_all-dataDark)[:,767,1000:1200],axis=-1),'+')
	interpolator = interp1d(np.mean((data_mean_all - dataDark)[:, 767, 1000:1200], axis=-1),signal_level)
	# plt.figure()
	plt.plot(np.array(signal_level)*interpolator(np.mean((data_mean_all-dataDark)[:,500,300:500],axis=-1)[0])/signal_level[0],np.mean((data_mean_all-dataDark)[:,500,300:500],axis=-1),'+')
	plt.plot(np.array(signal_level)*interpolator(np.mean((data_mean_all-dataDark)[:,500,1000:1200],axis=-1)[0])/signal_level[0],np.mean((data_mean_all-dataDark)[:,500,1000:1200],axis=-1),'+')
	plt.plot(np.array(signal_level)*interpolator(np.mean((data_mean_all-dataDark)[:,1000,300:500],axis=-1)[0])/signal_level[0],np.mean((data_mean_all-dataDark)[:,1000,300:500],axis=-1),'+')
	plt.plot(np.array(signal_level)*interpolator(np.mean((data_mean_all-dataDark)[:,1000,1000:1200],axis=-1)[0])/signal_level[0],np.mean((data_mean_all-dataDark)[:,1000,1000:1200],axis=-1),'+')
	plt.pause(0.01)

	plt.figure()
	plt.plot(signal_level,(data_mean_all_net)[:,767,1000])
	interpolator = interp1d((data_mean_all_net)[:, 767, 1000],signal_level,fill_value='extrapolate')
	for column in range(0,1608,100):
		if columns_to_look_at[column]:
			for row in range(1608):
				if (data_mean_all_net)[0, row, column]>2:
					plt.plot(np.array(signal_level) * interpolator((data_mean_all_net)[0, row, column]) / signal_level[0],(data_mean_all_net)[:, row, column], '+')
				else:
					print('column '+str(column)+' , row '+str(row)+' skipped')
	plt.pause(0.01)



if False:	# I check for crosstalk

	all_j = [386,387,384,388]

	data_mean_all = []
	data_std_all = []
	for j in all_j:
		# dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
		(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
		type = '.tif'
		filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
		type = '.txt'
		filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)[0]
		(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir, folder,sequence,untitled,filename_metadata)

		# plt.figure()
		data_all = []
		for index, filename in enumerate(filenames):
			fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
			im = Image.open(fname)
			data = np.array(im)
			# data = (data - dataDark) * Gain[index]
			# data = fix_minimum_signal2(data)
			data_all.append(data)
		# data_all = (np.array(data_all)- dataDark) * Gain[index]
		data_mean = np.mean(data_all, axis=0)
		data_std = np.std(data_all, axis=0)
		data_mean_all.append(data_mean)
		data_std_all.append(data_std)
	data_mean_all = np.array(data_mean_all)
	data_std_all = np.array(data_std_all)

	plt.figure()
	plt.imshow(data_mean_all[0],'rainbow',vmax=110)
	plt.colorbar()
	plt.figure()
	plt.imshow(data_mean_all[1],'rainbow',vmax=110)
	plt.colorbar()
	plt.figure()
	plt.imshow(data_mean_all[2],'rainbow')
	plt.colorbar()
	plt.figure()
	plt.imshow(data_mean_all[3],'rainbow')
	plt.colorbar()
	plt.pause(0.01)

	# outcome: not relevant

if False:	# I check how the correction routine works for data with plasma

	all_j = [268]

	data_mean_all = []
	data_std_all = []
	for j in all_j:
		# dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
		(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
		type = '.tif'
		filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
		type = '.txt'
		filename_metadata = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)[0]
		(bof, eof, roi_lb, roi_tr, elapsed_time, real_exposure_time, PixelType, Gain) = get_metadata(fdir, folder,sequence,untitled,filename_metadata)

		# plt.figure()
		data_all = []
		for index, filename in enumerate(filenames):
			fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
			im = Image.open(fname)
			data = np.array(im)
			# data = (data - dataDark) * Gain[index]
			# data = fix_minimum_signal2(data)
			data_all.append(data)
		# data_all = (np.array(data_all)- dataDark) * Gain[index]
		data_mean = np.mean(data_all, axis=0)
		data_std = np.std(data_all, axis=0)
		data_mean_all.append(data_mean)
		data_std_all.append(data_std)
	data_mean_all = np.array(data_mean_all)
	data_std_all = np.array(data_std_all)

	plt.figure()
	plt.imshow(data_all[150],'rainbow')
	plt.colorbar()
	plt.pause(0.01)
	plt.figure()
	plt.imshow(data_mean_all[0],'rainbow')
	plt.colorbar()
	plt.pause(0.01)


	from scipy.signal import find_peaks, peak_prominences as get_proms

	Spectrum = np.sum(data_mean_all[0],axis=0)  # Chord-integrated spectrum
	Spectrum-=min(Spectrum)
	#Spectrum.dtype = 'int'
	iNoise = Spectrum[20:-20].argmin()+20
	Noise = data[20:-20,iNoise];Noise = max(Noise)-min(Noise)
	iBright = Spectrum[20:-20].argmax()+20
	HM = (Spectrum[iNoise]+ Spectrum[iBright])/4+Spectrum[iNoise]
	#HM = (Spectrum[iNoise]+ Spectrum[iBright])/2
	RHM = list((Spectrum[iBright:] < HM)).index(True)
	LHM = list((Spectrum[iBright::-1] < HM)).index(True)

	peaks = find_peaks(Spectrum[2:])[0]    # Approximate horizontal postitions of spectral lines
	proms = get_proms(Spectrum[2:],peaks)[0]

	nLines=2
	intermediate_wavelength,last_wavelength = sorted(peaks[np.argpartition(-proms,nLines-1)[:nLines]]+2)
	column_averaging=30;range_rows=0;dx_to_etrapolate_to=180;nCh=40;counts_treshold_fixed_increase=106

	tilt_intermediate_column,tilt_last_column = get_bin_and_interv_specific_wavelength(data_mean_all[0],intermediate_wavelength),get_bin_and_interv_specific_wavelength(data_mean_all[0],last_wavelength)
	fix_minimum_signal_experiment(data_all[150],intermediate_wavelength,last_wavelength,tilt_intermediate_column,tilt_last_column)

	np.min(np.convolve(data[range_rows+row_averaging:len(data)-range_rows+row_averaging],np.ones((100))/100,mode='valid'),axis=-1)
