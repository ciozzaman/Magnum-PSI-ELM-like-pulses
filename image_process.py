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
from functions.GaussFitData import doLateralfit_time_tependent

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

# merge_ID=35
merge_ID = 3

if calculate_geometry:
	geom=getGeom(merge_ID,df_settings,df_log,fdir=fdir)
	print(geom)
else:
	geom = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
	geom.loc[0] = [0.01759,np.nan,29.202126,53.647099,53.647099]
geom_store = copy.deepcopy(geom)
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


# binnedSens = do_Intensity_Calib(merge_ID,df_settings,df_log,2,df_sphere,geom,waveLcoefs,waves_to_calibrate=['Hb'],fdir=fdir)
if merge_ID<=31:
	binnedSens = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/Sensitivity_1.npy')
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

merge_ID_target = 33
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
overexposed_treshold = 3600
data_sum=0
if ((not time_resolution_scan and not os.path.exists('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_merge_tot.nc')) or (time_resolution_scan and not os.path.exists('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.nc'))):
# if True:
	all_j=find_index_of_file(merge_ID_target,df_settings,df_log)
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
			# data=rotate(data,geom_null['angle'])
			# data=do_tilt(data,geom_null['tilt'])
			data =data - dataDark	# I checked that for 16 or 12 bit the dark is always around 100 counts
			data = data*Gain[index]
			# if Gain[index]==1:	#Case of gain in full well
			# 	data = data*8
			# elif Gain[index]==2:
			# 	data = data * 4
			# 09/06/19 section added to fix camera artifacts
			data = fix_minimum_signal(data)
			data_sum+=data
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
			started1 = 0
			for i in range(roi_tr[1]):
				data_to_record=data[i]
				# for index2, value in enumerate(data_to_record):
				# 	if value>=4095:	# THIS IS TO AVOID OVEREXPOSURE
				# 		data_to_record[index2]=np.nan		# TO BE FINISHED!!!!!!!!!!!!!

				if ( timesteps[i]<merge_time_window[0] or timesteps[i]>merge_time_window[1] ):
					continue
				# array_to_add[i,i]=data[:,i]
				# print(timesteps[i])
				overexposed=0
				if np.max(data_to_record)/Gain[index]>overexposed_treshold/Gain[index]:	#	it should be 4096 but comparing spectra with different gain you see it goes bad around this value
					overexposed=1
				if started1==0:
					merge=xr.DataArray(data_to_record, dims=[ 'wavelength_axis'],coords={'time_row_to_start_pulse': timesteps[i],'row': row_steps[i],'wavelength_axis': wavelengths,'overexposed':overexposed, 'Gain':Gain[index]},name='first entry')
					started1+=1
				else:
					addition_anticipated = False
					if started!=0:
						if np.shape(merge_tot.Gain.values)==():
							merge = xr.concat([merge_tot, xr.DataArray(data_to_record, dims=['wavelength_axis'],
																   coords={'time_row_to_start_pulse': timesteps[i],
																		   'row': row_steps[i],
																		   'wavelength_axis': wavelengths,
																		   'overexposed': overexposed,
																		   'Gain': Gain[index]},
																   name='addition' + str([j, index, i]))],
											  dim='time_row_to_start_pulse')
							addition_anticipated = True
						# 	merge = xr.concat([merge,xr.DataArray(data_to_record, dims=[ 'wavelength_axis'],coords={'time_row_to_start_pulse': timesteps[i],'row': row_steps[i],'wavelength_axis': wavelengths,'overexposed':overexposed, 'Gain':Gain[index]},name='addition'+str([j,index,i]))], dim='time_row_to_start_pulse')
					# else:
					# 	normal=0
						same_time = merge_tot.time_row_to_start_pulse.values == timesteps[i]
						if np.sum(same_time)>0:
							# sample = merge_tot[same_time]
							same_row = merge_tot.row.values == row_steps[i]
							if np.sum(np.logical_and(same_time,same_row)) > 0:
								same_Gain = merge_tot.Gain.values < Gain[index]
								if np.sum(np.logical_and(np.logical_and(same_time, same_row),same_Gain)) > 0:
									same_overexposed = merge_tot.overexposed.values == 1
									rows_to_fix = np.logical_and(np.logical_and(np.logical_and(same_time, same_row),same_Gain),same_overexposed)
									if np.sum(rows_to_fix) > 0:
										for z,to_be_fixed in enumerate(rows_to_fix):
											if to_be_fixed==True:
												print('fixing time '+str(timesteps[i])+', row '+str(row_steps[i])+'/row num'+str(z))
												sample =  merge_tot[z]
												for z1,value in enumerate(sample.values):
													if value/sample.Gain.values>overexposed_treshold:
														sample.values[z1]=data_to_record[z1]
														print('fixed at wavelength axis pixel '+str(z1))
												merge_tot[z].values = sample.values
												merge_tot.Gain[z] = Gain[index]
												merge_tot.overexposed[z] = overexposed
						# 			else:
						# 				normal=1
						# 		else:
						# 			normal = 1
						# 	else:
						# 		normal=1
						# else:
						# 	normal = 1
						# if normal==1:
					if not addition_anticipated:
						merge = xr.concat([merge, xr.DataArray(data_to_record, dims=['wavelength_axis'],
																		   coords={'time_row_to_start_pulse': timesteps[i],
																				   'row': row_steps[i],
																				   'wavelength_axis': wavelengths,
																				   'overexposed': overexposed,
																				   'Gain': Gain[index]},
																		   name='addition' + str([j, index, i]))],
													  dim='time_row_to_start_pulse')
			if started == 0:
				merge_tot = merge
				started += 1
			else:
				merge_tot = xr.concat([merge_tot, merge],dim='time_row_to_start_pulse')

	if time_resolution_scan:
		merge_tot.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.nc')
	else:
		merge_tot.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_merge_tot.nc')

	path_filename = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_stats.csv'
	# if not os.path.exists(path_filename):
	file = open(path_filename, 'w')
	writer = csv.writer(file)
	writer.writerow(['Stats about merge ' + str(merge_ID_target)])
		# file.close()

	result = 0
	nLines = 8
	while (result == 0 and nLines>0):
		try:
			angle = get_angle(data_sum,nLines=nLines)
			result = 1
			geom['angle'] = np.nansum(np.multiply(angle[0], np.divide(1, np.power(angle[1], 2)))) / np.nansum(np.divide(1, np.power(angle[1], 2)))
			writer.writerow(['Specific angle of ',str(geom['angle'][0]),' found with nLines=',str(nLines)])
		except:
			nLines-=1
	if np.abs(geom['angle'][0])>2*np.abs(geom_store['angle'][0]):
		geom['angle'][0]=geom_store['angle'][0]
		result=0
		nLines = 8
	if result == 0:
		writer.writerow(['No specific angle found. Used standard of ', str(geom['angle'][0])])
	data_sum = rotate(data_sum, geom['angle'])
	file = open(path_filename, 'r')
	result = 0
	while (result == 0 and nLines>0):
		try:
			tilt = get_tilt(data_sum,nLines=nLines)
			result = 1
			geom['binInterv'] = tilt[0]
			geom['tilt'] =tilt[1]
			geom['bin00a'] =tilt[2]
			geom['bin00b'] =tilt[2]
			writer.writerow(['Specific tilt of ', str(geom['tilt'][0]), ' found with nLines=', str(nLines), 'binInterv', str(geom['binInterv'][0]),'first bin',str(geom['bin00a'][0])])
		except:
			nLines-=1
	if result == 0:
		writer.writerow(['No specific tilt found. Used standard of ', str(geom['tilt'][0]),'binInterv',str(geom['binInterv'][0]), 'first bin', str(geom['bin00a'][0])])
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
		for index, filename in enumerate(filenames):
			if (index < first_pulse_at_this_frame - 1 or index > (
					first_pulse_at_this_frame + number_of_pulses - 1)):
				continue
			elif (index - first_pulse_at_this_frame + 2) in bad_pulses_indexes:
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
			# data = fix_minimum_signal(data)
			data_sum += data
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
			result = 1
			geom['angle'] = np.nansum(np.multiply(angle[0], np.divide(1, np.power(angle[1], 2)))) / np.nansum(
				np.divide(1, np.power(angle[1], 2)))
			writer.writerow(['Specific angle of ', str(geom['angle'][0]), ' found with nLines=', str(nLines)])
		except:
			nLines -= 1
	if np.abs(geom['angle'][0])>2*np.abs(geom_store['angle'][0]):
		geom['angle'][0] = geom_store['angle'][0]
		result = 0
	if result == 0:
		writer.writerow(['No specific angle found. Used standard of ', str(geom['angle'][0])])
	data_sum = rotate(data_sum, geom['angle'])
	file = open(path_filename, 'r')
	result = 0
	nLines = 8
	while (result == 0 and nLines > 0):
		try:
			tilt = get_tilt(data_sum, nLines=nLines)
			result = 1
			geom['binInterv'] = tilt[0]
			geom['tilt'] = tilt[1]
			geom['bin00a'] = tilt[2]
			geom['bin00b'] = tilt[2]
			writer.writerow(
				['Specific tilt of ', str(geom['tilt'][0]), ' found with nLines=', str(nLines), 'binInterv',
				 str(geom['binInterv'][0]), 'first bin', str(geom['bin00a'][0])])
		except:
			nLines -= 1
	if result == 0:
		writer.writerow(['No specific tilt found. Used standard of ', str(geom['tilt'][0]), 'binInterv',
						 str(geom['binInterv'][0]), 'first bin', str(geom['bin00a'][0])])
	writer.writerow(
		['the final result will be from', str(merge_time_window[0]), ' to ', str(merge_time_window[1]),
		 ' ms from the beginning of the discharge'])
	writer.writerow(['with a time resolution of ' + str(conventional_time_step), ' ms'])
	file.close()

	if time_resolution_scan:
		merge_tot = xr.open_dataarray('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_merge_tot.nc')
	else:
		merge_tot = xr.open_dataarray('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_merge_tot.nc')

	# ani = coleval.movie_from_data(np.array([composed_array.values]), 1000/conventional_time_step, row_shift, 'Wavelength axis [pixles]', 'Row axis [pixles]','Intersity [au]')
	# # plt.show()
	# ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_composed_array' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# I will reconstruct a full scan of 10ns steps from 0 to rows*10ns and all rows

new_timesteps = np.linspace(merge_time_window[0]+0.5, merge_time_window[1]-0.5,int((merge_time_window[1]-0.5-(merge_time_window[0]+0.5))/conventional_time_step+1))
composed_array=xr.DataArray(np.zeros((len(new_timesteps),roi_tr[1],roi_tr[0])), dims=['time', 'row', 'wavelength_axis'],coords={'time': new_timesteps,'row': row_steps,'wavelength_axis': wavelengths},name='interpolated array')
# composed_array = merge_tot.interp_like(composed_array,kwargs={'fill_value': 0.0})
# this seems not to work so I do it manually


if ((not time_resolution_scan and not os.path.exists('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_composed_array.nc')) or (time_resolution_scan and not os.path.exists('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_composed_array.nc'))):

	merge_tot = merge_tot.sortby('time_row_to_start_pulse')
	for i,interpolated_time in enumerate(new_timesteps):
		sample = merge_tot.sel(time_row_to_start_pulse=slice(interpolated_time - time_range_for_interp,interpolated_time + time_range_for_interp))
		additional_time_range = 0
		sample_original = copy.deepcopy(sample)
		if (not time_resolution_scan or time_resolution_scan_improved):
			while (len(sample) < 3*roi_tr[1] and additional_time_range<time_range_for_interp):	# range max doubled
				additional_time_range += time_range_for_interp/10
				sample = merge_tot.sel(time_row_to_start_pulse=slice(interpolated_time - time_range_for_interp-additional_time_range,
																	 interpolated_time + time_range_for_interp+additional_time_range))
				print(str(interpolated_time)+' increased of '+str(additional_time_range))
		if  len(sample)==0:
			continue

		# sample = sample.sortby('row')
		for j, interpolated_row in enumerate(row_steps):
			weight = 1 / np.sqrt(0.00001+((sample.time_row_to_start_pulse - interpolated_time)/ row_shift ) ** 2 + (sample.row - interpolated_row) ** 2)
			row_selection = (sample.row>float(interpolated_row-rows_range_for_interp)).astype('int') * (sample.row<float(interpolated_row+rows_range_for_interp)).astype('int')
			additional_row_range = 0
			if (not time_resolution_scan or time_resolution_scan_improved):
				while (np.sum(row_selection) < 2 and additional_row_range<rows_range_for_interp):	# range max doubled
					additional_row_range +=rows_range_for_interp/10
					row_selection = (sample.row > float(interpolated_row - rows_range_for_interp-additional_row_range)).astype('int') * (
								sample.row < float(interpolated_row + rows_range_for_interp+additional_row_range)).astype('int')
					# print(sample.row)
					# print(sample)
					print(str(interpolated_time)+' '+str(interpolated_row)+' increased of ' + str(additional_row_range))
					print('total number of rows '+str(len(weight)))
			if np.sum(row_selection) < 1:
				continue
			# elif np.sum(row_selection) > 10:
			# 	weight_row = row_selection*(weight)
			# else:
			# 	weight_row = weight
			weight_row = row_selection * weight
			# weight_row_filter=(weight_row.values).tolist()
			# for index,value in enumerate(weight_row_filter):
			# 	if value==np.nan:
			# 		weight_row_filter.pop(index)
			# weight_row=np.array(weight_row)
			sample_weighted = sample*weight_row
			sample_weighted = np.nansum(sample_weighted,axis = 0)/np.nansum(weight_row)
			composed_array[i,j]=sample_weighted


	if time_resolution_scan:
		if time_resolution_scan_improved:
			composed_array.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'improved_row_composed_array.nc')
		else:
			composed_array.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_skip_of_' + str(time_resolution_extra_skip + 1) + 'row_composed_array.nc')
		ani = coleval.movie_from_data(np.array([composed_array.values]), 1 / conventional_time_step, row_shift,'Wavelength axis [pixles]', 'Row axis [pixles]', 'Intersity [au]')
		ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_composed_array' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

	else:
		composed_array.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_composed_array.nc')
		# ani = coleval.movie_from_data(np.array([composed_array.values]), 1 / conventional_time_step, row_shift,'Wavelength axis [pixles]', 'Row axis [pixles]', 'Intersity [au]')
		# ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_composed_array' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

else:
	if time_resolution_scan:
		composed_array = xr.open_dataarray('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_composed_array.nc')
	else:
		composed_array = xr.open_dataarray('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_composed_array.nc')


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
	binned_data=xr.DataArray(np.zeros((len(new_timesteps),40,np.shape(frame)[1])), dims=['time', 'bin', 'wavelength_axis'],coords={'time': new_timesteps,'bin': bin_steps,'wavelength_axis': wavelengths},name='interpolated array')



	for index,time in enumerate(new_timesteps):
		frame = composed_array[index]
		frame = np.nan_to_num(frame)
		frame=rotate(frame,geom['angle'])
		frame=do_tilt(frame,geom['tilt'])
		binnedData = binData(frame,geom['bin00b'],geom['binInterv'],check_overExp=False)
		binned_data[index]=binnedData

	if time_resolution_scan:
		if time_resolution_scan_improved:
			binned_data.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_skip_of_' + str(time_resolution_extra_skip + 1) + 'improved_row_binned_data.nc')
		else:
			binned_data.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_binned_data.nc')
		# ani = coleval.movie_from_data(np.array([binned_data.values]), 1000/conventional_time_step, row_shift, 'Wavelength axis [pixles]', 'Bin [au]','Intersity [au]')
		# ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(time_resolution_extra_skip+1)+'row_binned_data' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
	else:
		binned_data.to_netcdf(path='/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_binned_data.nc')
		# ani = coleval.movie_from_data(np.array([binned_data.values]), 1000/conventional_time_step, row_shift, 'Wavelength axis [pixles]', 'Bin [au]','Intersity [au]')
		# ani.save('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_binned_data' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
else:
	if time_resolution_scan:
		binned_data = xr.open_dataarray(
			'/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '_skip_of_' + str(
				time_resolution_extra_skip + 1) + 'row_binned_data.nc')
	else:
		binned_data = xr.open_dataarray('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(
			merge_ID_target) + '_binned_data.nc')

#composed_array = np.zeros((roi_tr[1],roi_tr[1],roi_tr[0]))
#for i in range(np.shape(composed_array)[0]):
#	for j in range(np.shape(composed_array)[1]):
#		data_to_interpolate = merge.sel(time_row_to_start_pulse=slice(i*conventional_time_step-time_range_for_interp,i*conventional_time_step-time_range_for_interp),row=slice(i-rows_range_for_interp,i-rows_range_for_interp))
fit = doSpecFit_single_frame(np.array(composed_array[0]), df_settings, df_log, geom, waveLcoefs, binnedSens)

all_fits=[]
for index,time in enumerate(new_timesteps):
	print('fit of '+str(time)+'ms')
	try:
		fit = doSpecFit_single_frame(np.array(composed_array[index]),df_settings, df_log, geom, waveLcoefs, binnedSens)
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


all_fits = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_all_fits.npy')

doLateralfit_time_tependent(df_settings,all_fits, merge_ID_target)









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

























