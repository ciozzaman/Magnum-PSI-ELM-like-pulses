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

exec(open("/home/ffederic/work/Collaboratory/test/experimental_data/IR_records_process_preamble.py").read())


temp=[]
merge_id_all = [66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85, 95, 86, 87, 89, 92, 93, 94, 96, 97, 98, 88, 99]
# merge_id_all = [97]
for merge_id in merge_id_all:
	temp.extend(find_index_of_file(merge_id,df_settings,df_log,only_OES=False))
all_j=[]
for i in range(len(df_log)-1):
	IR_trace,IR_reference = df_log.loc[i,['IR_trace','IR_reference']]
	# if (isinstance(IR_trace, str) and np.isnan(df_log.loc[i,['DT_pulse']][0])):
	if (isinstance(IR_trace, str) and (i in temp)):
		# if i<=265:
		all_j.append(i)

# all_j = [402]
# # all_j = [227,228]
# all_j = [159,163,166,174,186]
# all_j = [104,105,106,107,108]
# all_j = [396,397,398,399,400,401,402,403]
# all_j = [227,232,247,250]
# all_j = [*np.arange(267,270+1),*np.arange(272,275+1),*np.arange(277,286+1),393,394,*np.arange(396,403+1)]

# for j in np.flip(all_j,axis=0):
# for j in np.flip(all_j[1::2],axis=0):
for j in all_j:
# def calc_stuff(arg):
# 	j = arg[1]
	print('analysing item n '+str(j))

	df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
	(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
	(IR_trace,IR_reference,IR_shape,magnetic_field,target_OES_distance,target_chamber_pressure,capacitor_voltage) = df_log.loc[j,['IR_trace','IR_reference','IR_shape','B','T_axial','p_n [Pa]','Vc']]
	(CB_to_OES_initial_delay,incremental_step,number_of_pulses) = df_log.loc[j,['CB_to_OES_initial_delay','incremental_step','number_of_pulses']]
	number_of_pulses = int(number_of_pulses)
	(folder,date,sequence,untitled,target_material,fname_current_trace,first_pulse_at_this_frame,bad_pulses_indexes) = df_log.loc[j,['folder','date','sequence','untitled','Target','current_trace_file','first_pulse_at_this_frame','bad_pulses_indexes']]
	sequence = int(sequence)
	merge_found = 0
	for i in range(len(df_settings)):
		a,b = df_settings.loc[i,['Hb','merge_ID']]
		if a==j:
			merge_ID=b
			merge_found=1
			print('Equivalent to merge '+str(merge_ID))
			break

	pre_title = 'merge %.3g, B=%.3gT, pos=%.3gmm, P=%.3gPa, ELMen=%.3gJ \n' %(merge_ID,magnetic_field,target_OES_distance,target_chamber_pressure,0.5*(capacitor_voltage**2)*150e-6)

	if isinstance(fname_current_trace,str):
		if os.path.exists(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/'+fname_current_trace+'.tsf'):
			bad_pulses,first_good_pulse,first_pulse,last_pulse,miss_pulses,double_pulses,good_pulses, time_of_pulses, energy_per_pulse = examine_current_trace(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/', fname_current_trace, df_log.loc[j, ['number_of_pulses']][0],want_the_power_per_pulse=True)
			if False:
				current_traces = pd.read_csv(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/' + fname_current_trace+'.tsf',index_col=False, delimiter='\t')
				current_traces_time = current_traces['Time [s]']
				current_traces_total = current_traces['I_Src_AC [A]']
				voltage_traces_total = current_traces['U_Src_DC [V]']
				target_voltage_traces_total = current_traces['U_Tar_DC [V]']
				plt.figure()
				plt.plot(current_traces_time,-current_traces_total*voltage_traces_total)
				plt.xlabel('Time [s]')
				plt.ylabel('Power [W]')
				plt.grid()
				plt.title(pre_title+'file '+fname_current_trace)
				plt.pause(0.01)

				plt.figure()
				plt.plot(current_traces_time,-current_traces_total*voltage_traces_total)
				plt.xlabel('Time [s]')
				plt.plot(current_traces_time,current_traces_total,label='current[a]')
				plt.plot(current_traces_time,voltage_traces_total,label='source voltage[V]')
				plt.plot(current_traces_time,target_voltage_traces_total,label='target voltage[V]')
				plt.legend(loc='best')
				plt.grid()
				plt.title(pre_title+'file '+fname_current_trace)
				plt.pause(0.01)


	if bad_pulses_indexes=='':
		bad_pulses_indexes=[0]
	elif (isinstance(bad_pulses_indexes, float) or isinstance(bad_pulses_indexes, int)):
		bad_pulses_indexes=[bad_pulses_indexes]
	else:
		bad_pulses_indexes = bad_pulses_indexes.replace(' ', '').split(',')
		bad_pulses_indexes = list(map(int, bad_pulses_indexes))

	# if -100 in bad_pulses_indexes:	# "keyword" in case I do not want to consider any of the pulses
	# 	bad_pulses_indexes = np.linspace(1,number_of_pulses,number_of_pulses).astype('int')

	# if bad_pulses_indexes==np.nan:
	# 	bad_pulses_indexes=[]
	# else:
	# 	bad_pulses_indexes = bad_pulses_indexes.replace(' ', '').split(',')
	# 	bad_pulses_indexes = list(map(int, bad_pulses_indexes))
	incremental_step = 0.1	# I do this because what I mean really is the time between pulses, that is always 0.1s

	IR_shape = np.array((IR_shape.replace(' ', '').split(','))).astype(int)

	if merge_found:
		path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID)
	else:
		path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/'+"{0:0=2g}".format(sequence)+'/IR_camera'
	figure_index = 0

	header = ryptw.readPTWHeader('/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/'+"{0:0=2g}".format(sequence)+'/IR_camera/'+IR_reference+'.ptw')
	# ryptw.showHeader(header)
	frequency = 1/header.h_CEDIPAquisitionPeriod # Hz
	int_time = round(header.h_CEDIPIntegrationTime,9) # s

	if int_time == 1e-4:
		background_interval = int(6*frequency)
	elif int_time == 4e-4:
		background_interval = int(5*frequency)
	elif int_time == 1e-3:
		background_interval = int(4*frequency)
	else:
		print('error, only 1, 0.1 and 0.4ms int time now')
		continue

	print('mark7')

	# frames = frames.flatten().reshape((np.shape(frames)[1],np.shape(frames)[2],np.shape(frames)[0])).T
	frames = []
	for i in range(1,background_interval):
		frame = ryptw.getPTWFrames(header, [i])[0][0]
		frames.append(frame.flatten().reshape(np.shape(frame.T)).T)
	print('mark6')

	dark = np.mean(frames,axis=0)
	# plt.figure();plt.imshow(dark),plt.colorbar();plt.pause(0.01)

	plt.figure(figsize=(20, 10))
	plt.imshow(dark,'rainbow')
	plt.colorbar().set_label('digital level [au]')
	plt.plot(IR_shape[1],IR_shape[0],'k+')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	plt.xlabel('horizontal coord [pixels]')
	plt.ylabel('vertical coord [pixels]')
	plt.title(pre_title+'Dark background in '+str(j)+', IR trace '+IR_reference+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	header = ryptw.readPTWHeader('/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/'+"{0:0=2g}".format(sequence)+'/IR_camera/'+IR_trace+'.ptw')

	frames = []
	for i in range(header.h_firstframe,header.h_lastframe):
		frame = ryptw.getPTWFrames(header, [i])[0][0]
		frames.append(frame.flatten().reshape(np.shape(frame.T)).T)
	print('mark3')

	spectra = np.fft.fft(frames,axis=0)
	magnitude = 2 * np.abs(spectra) / len(spectra)
	freq = np.fft.fftfreq(len(magnitude), d=1 / frequency)

	# plt.figure();plt.plot(freq,magnitude[:,40,24]),plt.semilogy();plt.pause(0.01)
	# plt.figure();plt.imshow(magnitude[freq.argmax()]),plt.colorbar();plt.pause(0.01)
	# plt.figure();plt.imshow(magnitude[freq.argmin()]),plt.colorbar();plt.pause(0.01)

	magnitude_reshape = magnitude.reshape((magnitude.shape[0],magnitude.shape[1]*magnitude.shape[2]))
	magnitude_reshape = magnitude_reshape[np.abs(freq)>freq.max()//10*10-1]
	magnitude_reshape = magnitude_reshape[:magnitude_reshape.shape[0]//2,:][:,2000:5000]
	temp = freq[np.abs(freq)>freq.max()//10*10-1]
	freq_2,location = np.meshgrid(temp[:temp.shape[0]//2],np.arange(magnitude.shape[1]*magnitude.shape[2])[2000:5000])
	fig = plt.figure(figsize=(20, 10))
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(freq_2.T, location.T, np.log(magnitude_reshape),rstride=10,cstride=10,cmap=cm.rainbow,linewidth=0, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	ax.set_xlabel('frequency [Hz]')
	ax.set_ylabel('pixel index')
	ax.set_zlabel('amplitude [au]')
	plt.title(pre_title+'Original FT for rows interested by oscillation in '+str(j)+', IR trace '+IR_reference+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	# I definietly need some filtering because I have massive oscillations in especially 2 bands horizontaly at ~30 and35 pixels
	# BUT if I do two different filtering the situation gets worst, so I do only one
	if False:
		# # spectra[np.abs(freq)>(np.sort(freq)[-1]-np.sort(freq)[-1]/20000)] = 0
		spectra[np.abs(freq)>freq.max()//10*10+3] = 0
		# spectra[freq.argmin()] = 0
		# spectra[freq.argmax()] = 0
		# corrected[freq.argmin()] = 0
		print('mark4')
		corrected_frames = np.abs(np.fft.ifft(spectra,axis=0))
		corrected_spectra = np.fft.fft(corrected_frames,axis=0)
		corrected_magnitude = 2 * np.abs(corrected_spectra) / len(corrected_spectra)

		magnitude_reshape = corrected_magnitude.reshape((magnitude.shape[0],magnitude.shape[1]*magnitude.shape[2]))
		magnitude_reshape = magnitude_reshape[np.abs(freq)>freq.max()//10*10-1]
		magnitude_reshape = magnitude_reshape[:magnitude_reshape.shape[0]//2,:][:,2000:5000]
		temp = freq[np.abs(freq)>freq.max()//10*10-1]
		freq_2,location = np.meshgrid(temp[:temp.shape[0]//2],np.arange(magnitude.shape[1]*magnitude.shape[2])[2000:5000])
		fig = plt.figure(figsize=(20, 10))
		ax = fig.gca(projection='3d')
		surf = ax.plot_surface(freq_2.T, location.T, np.log(magnitude_reshape),rstride=10,cstride=10,cmap=cm.rainbow,linewidth=0, antialiased=False)
		fig.colorbar(surf, shrink=0.5, aspect=5)
		ax.set_xlabel('frequency [Hz]')
		ax.set_ylabel('pixel index')
		ax.set_zlabel('amplitude [au]')
		plt.title(pre_title+'Intermediate FT for rows interested by oscillation in '+str(j)+', IR trace '+IR_reference+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
		figure_index+=1
		plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
		plt.close()
		print('mark5')

	# ani = coleval.movie_from_data(np.array([frames[-10:]]), frequency, int_time/1000,'horizontal coord [pixels]','vertical coord [pixels]','Intersity [au]')
	# ani = coleval.movie_from_data(np.array([corrected_frames[-10:]]), frequency, int_time/1000,'horizontal coord [pixels]','vertical coord [pixels]','Intersity [au]')
	# ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_original.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
	# # ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_original.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
	# plt.close()

	# plt.figure(figsize=(20, 10))
	# plt.plot(freq[freq.argmin():freq.argmin()+50],magnitude[freq.argmin():freq.argmin()+50,int(np.shape(frames[0])[0]/2),int(np.shape(frames[0])[1]/2)],label='original')
	# if False:	# I don't think this is actually necessary
	# 	plt.plot(freq[freq.argmin():freq.argmin()+50],corrected_magnitude[freq.argmin():freq.argmin()+50,int(np.shape(frames[0])[0]/2),int(np.shape(frames[0])[1]/2)],label='corrected')
	# # plt.plot([(np.sort(freq)[-1]-np.sort(freq)[-1]/20000),(np.sort(freq)[-1]-np.sort(freq)[-1]/20000)],[np.min(magnitude[:,int(np.shape(frames[0])[0]/2),int(np.shape(frames[0])[1]/2)]),np.max(magnitude[:,int(np.shape(frames[0])[0]/2),int(np.shape(frames[0])[1]/2)])],'k--')
	# plt.plot([(np.sort(freq)[0]+np.sort(freq)[-1]/20000),(np.sort(freq)[0]+np.sort(freq)[-1]/20000)],[np.min(magnitude[:,int(np.shape(frames[0])[0]/2),int(np.shape(frames[0])[1]/2)]),np.max(magnitude[:,int(np.shape(frames[0])[0]/2),int(np.shape(frames[0])[1]/2)])],'k--')
	# plt.xlabel('frequency [Hz]')
	# plt.ylabel('amplitude [au]')
	# plt.title(pre_title+'Example for elimination of the oscillation in '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	# plt.semilogy()
	# plt.legend(loc='best')
	# figure_index+=1
	# plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	# plt.close()
	#
	# plt.figure(figsize=(20, 10))
	# plt.plot(freq[0:np.abs(freq-50).argmin()],magnitude[0:np.abs(freq-50).argmin(),int(np.shape(frames[0])[0]/2),int(np.shape(frames[0])[1]/2)],label='original')
	# if False:	# I don't think this is actually necessary
	# 	plt.plot(freq[0:np.abs(freq-50).argmin()],corrected_magnitude[0:np.abs(freq-50).argmin(),int(np.shape(frames[0])[0]/2),int(np.shape(frames[0])[1]/2)],label='corrected')
	# plt.xlabel('frequency [Hz]')
	# plt.ylabel('amplitude [au]')
	# plt.title(pre_title+'Check for classic ~25Hz oscillation in '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	# plt.semilogy()
	# plt.legend(loc='best')
	# figure_index+=1
	# plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	# plt.close()


	# plt.figure();plt.imshow(corrected_frames[0]),plt.colorbar();plt.pause(0.01)

	corrected_frames = np.array(frames)
	corrected_frames = corrected_frames-dark
	# corrected_frames = medfilt(corrected_frames,[1,5,5])
	corrected_frames = median_filter(corrected_frames,size=[2,7,7])
	corrected_frames[corrected_frames<0] = 0

	spectra = np.fft.fft(corrected_frames,axis=0)
	magnitude = 2 * np.abs(spectra) / len(spectra)
	freq = np.fft.fftfreq(len(magnitude), d=1 / frequency)

	magnitude_reshape = magnitude.reshape((magnitude.shape[0],magnitude.shape[1]*magnitude.shape[2]))
	magnitude_reshape = magnitude_reshape[np.abs(freq)>freq.max()//10*10-1]
	magnitude_reshape = magnitude_reshape[:magnitude_reshape.shape[0]//2,:][:,2000:5000]
	temp = freq[np.abs(freq)>freq.max()//10*10-1]
	freq_2,location = np.meshgrid(temp[:temp.shape[0]//2],np.arange(magnitude.shape[1]*magnitude.shape[2])[2000:5000])
	fig = plt.figure(figsize=(20, 10))
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(freq_2.T, location.T, np.log(magnitude_reshape),rstride=10,cstride=10,cmap=cm.rainbow,linewidth=0, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	ax.set_xlabel('frequency [Hz]')
	ax.set_ylabel('pixel index')
	ax.set_zlabel('amplitude [au]')
	plt.title(pre_title+'Corrected FT for rows interested by oscillation in '+str(j)+', IR trace '+IR_reference+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	if True:	# counts temperature conversion from Yu Li
		tau = 0.145
		counts_to_temperature,counts_to_emissivity = dl2temp_generator(0,0,tau,int_time*1e6,target_material,out_emissivity=True)
		if target_material in ['TZM thin','TZM thick','molibdenum']:
			thermal_conductivity = 126	# W/mK
			heat_capacity = 305	# J/kg k
			density = 370	# kg/m3
		elif target_material in ['tungsten dummy','tungsten']:
			thermal_conductivity = 175	# W/mK
			heat_capacity = 134	# J/kg k
			density = 19250	# kg/m3
		else:
			print('Error, target material not specified. add it to the shot characteristics')
			exit()
	else:
		if int_time == 1e-4:
			if target_material in ['TZM thin','TZM thick','molibdenum']:
				counts_to_emissivity = emissivity_interpolator_100mus_inttime_molibdenum
				counts_to_temperature = interpolator_100mus_inttime_molibdenum
				thermal_conductivity = 126	# W/mK
				heat_capacity = 305	# J/kg k
				density = 370	# kg/m3
			elif target_material in ['tungsten dummy','tungsten']:
				counts_to_emissivity = emissivity_interpolator_100mus_inttime_tungsten
				counts_to_temperature = interpolator_100mus_inttime_tungsten
				thermal_conductivity = 175	# W/mK
				heat_capacity = 134	# J/kg k
				density = 19250	# kg/m3
			else:
				print('Error, target material not specified. add it to the shot characteristics')
				exit()
		elif int_time == 4e-4:
			if target_material in ['TZM thin','TZM thick','molibdenum']:
				counts_to_emissivity = emissivity_interpolator_400mus_inttime_molibdenum
				counts_to_temperature = interpolator_400mus_inttime_molibdenum
				thermal_conductivity = 126	# W/mK
				heat_capacity = 305	# J/kg k
				density = 370	# kg/m3
			elif target_material in ['tungsten dummy','tungsten']:
				counts_to_emissivity = emissivity_interpolator_400mus_inttime_tungsten
				counts_to_temperature = interpolator_400mus_inttime_tungsten
				thermal_conductivity = 175	# W/mK
				heat_capacity = 134	# J/kg k
				density = 19250	# kg/m3
			else:
				print('Error, target material not specified. add it to the shot characteristics')
				exit()
			# corrected_frames = np.exp(interpolator_400mus_inttime(corrected_frames))+20
		elif int_time == 1e-3:
			if target_material in ['TZM thin','TZM thick','molibdenum']:
				counts_to_emissivity = emissivity_interpolator_1000mus_inttime_molibdenum
				counts_to_temperature = interpolator_1000mus_inttime_molibdenum
				thermal_conductivity = 126	# W/mK
				heat_capacity = 305	# J/kg k
				density = 370	# kg/m3
			elif target_material in ['tungsten dummy','tungsten']:
				counts_to_emissivity = emissivity_interpolator_1000mus_inttime_tungsten
				counts_to_temperature = interpolator_1000mus_inttime_tungsten
				thermal_conductivity = 175	# W/mK
				heat_capacity = 134	# J/kg k
				density = 19250	# kg/m3
			else:
				print('Error, target material not specified. add it to the shot characteristics')
				exit()
			# corrected_frames = np.exp(interpolator_1000mus_inttime(corrected_frames))+20
		else:
			print('error, only 0.1, 0.4, 1ms int time now')
			continue
			# return 'fail '+str(j)


	try:
		selected_1_mean_counts = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace +'.npz')['selected_2_max']
		successfull_reading = True
	except:
		successfull_reading = False
	# if ( not os.path.exists(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '.npz') ):

	# !! TEMPORARY !! need to reset everything
	successfull_reading = False

	if not successfull_reading:

		selected_x = np.zeros_like(corrected_frames[0])+np.arange(np.shape(corrected_frames)[-1])
		selected_y = (np.zeros_like(corrected_frames[0]).T+np.arange(np.shape(corrected_frames)[-2])).T
		selected_1 = ((selected_x-IR_shape[1])**2 + (selected_y-IR_shape[0])**2)<IR_shape[2]**2
		selected_2 = ((selected_x-IR_shape[1])**2 + (selected_y-IR_shape[0])**2)<IR_shape[3]**2

		selected_1_mean_counts = np.mean(corrected_frames[:,selected_1],axis=-1)
		selected_1_max_counts = np.max(corrected_frames[:,selected_1],axis=-1)
		selected_2_mean_counts = np.mean(corrected_frames[:,selected_2],axis=-1)
		selected_2_max_counts = np.max(corrected_frames[:,selected_2],axis=-1)

		# peak_image_counts = corrected_frames[selected_2_max_counts.argmax()]
		# peak_after_image_counts = corrected_frames[selected_2_max_counts.argmax()+round(1.5/1000*frequency)]

		# plt.figure();plt.imshow(corrected_frames[3]),plt.colorbar();plt.pause(0.01)
		# plt.figure();plt.plot(freq,corrected_magnitude[:,40,24]),plt.semilogy();plt.pause(0.01)
		emissivity = np.zeros_like(corrected_frames)
		corrected_frames_temp = np.zeros_like(corrected_frames)
		print('mark1')
		for i in range(len(corrected_frames)):
			# emissivity[i] = np.exp(counts_to_emissivity(corrected_frames[i]))
			corrected_frames_temp[i] = np.exp(counts_to_temperature(corrected_frames[i]))+20
		print('mark2')

		selected_1_mean = np.mean(corrected_frames_temp[:,selected_1],axis=-1)
		selected_1_max = np.max(corrected_frames_temp[:,selected_1],axis=-1)
		selected_2_mean = np.mean(corrected_frames_temp[:,selected_2],axis=-1)
		selected_2_max = np.max(corrected_frames_temp[:,selected_2],axis=-1)

		peaks = find_peaks(selected_2_mean,distance=incremental_step*frequency*0.9)[0]
		proms = get_proms(selected_2_mean,peaks)[0]

		# ani = coleval.movie_from_data(np.array([frames]), frequency, int_time/1000,'horizontal coord [pixels]','vertical coord [pixels]','Intersity [au]')
		# ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_original.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
		# # ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_original.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
		# plt.close()
		#
		# ani = coleval.movie_from_data(np.array([corrected_frames_temp]), frequency, int_time/1000,'horizontal coord [pixels]','vertical coord [pixels]','Intersity [au]')
		# ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_original.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
		# # ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_original.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
		# plt.close()

		# peak_image = corrected_frames_temp[peaks[proms.argmax()]]
		# peak_after_image = corrected_frames_temp[peaks[proms.argmax()+round(1.5/1000*frequency)]]
		# peak_before_image = np.mean(corrected_frames_temp[peaks[proms.argmax()]-10:peaks[proms.argmax()]-5],axis=0)

		np.savez_compressed(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace,selected_1_mean=selected_1_mean,selected_1_max=selected_1_max,selected_2_mean=selected_2_mean,selected_2_max=selected_2_max,selected_1_mean_counts=selected_1_mean_counts,selected_1_max_counts=selected_1_max_counts,selected_2_mean_counts=selected_2_mean_counts,selected_2_max_counts=selected_2_max_counts)
	else:
		selected_1_mean = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace +'.npz')['selected_1_mean']
		selected_1_max = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace +'.npz')['selected_1_max']
		selected_2_mean = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace +'.npz')['selected_2_mean']
		selected_2_max = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace +'.npz')['selected_2_max']
		selected_1_mean_counts = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace +'.npz')['selected_1_mean_counts']
		selected_1_max_counts = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace +'.npz')['selected_1_max_counts']
		selected_2_mean_counts = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace +'.npz')['selected_2_mean_counts']
		selected_2_max_counts = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace +'.npz')['selected_2_max_counts']
		# peak_image_counts = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace +'.npz')['peak_image_counts']
		# peak_image = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace +'.npz')['peak_image']
		# peak_after_image_counts = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace +'.npz')['peak_after_image_counts']
		# peak_after_image = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace +'.npz')['peak_after_image']
		# peak_before_image = np.load(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace +'.npz')['peak_before_image']

		peaks = find_peaks(selected_2_mean,distance=incremental_step*frequency*0.9)[0]
		proms = get_proms(selected_2_mean,peaks)[0]
		# figure_index+=8

	# plt.figure(figsize=(20, 10));plt.plot(selected_1_mean);plt.plot(selected_1_max);plt.plot(selected_2_mean);plt.plot(selected_2_max);plt.pause(0.01)


	if np.max(proms[:-int(number_of_pulses/3)])<20:
		if np.sum((proms[1:]/proms[:-1])>50)==0:
			if np.sum((proms[1:]/proms[:-1])>10)==0:
				minimum_time=0
			else:
				minimum_time = (peaks[:-1][(proms[1:]/proms[:-1])>10])[0]-1
		else:
			minimum_time = (peaks[:-1][(proms[1:]/proms[:-1])>50])[0]-1
	else:
		minimum_time = (peaks[1:][np.diff(proms)>20])[0]-int(incremental_step*frequency*0.2)
	minimum_time = max(minimum_time,background_interval)
	peaks_initial = find_peaks(selected_2_max,distance=incremental_step*frequency*0.85)[0]
	proms_initial = get_proms(selected_2_max,peaks_initial)[0]
	peaks = peaks_initial[peaks_initial>minimum_time]
	peaks = peaks[:number_of_pulses]
	proms = get_proms(selected_2_max,peaks)[0]

	# guess_for_good_pulse = max(np.median(proms),np.mean(proms))
	if frequency>400:
		if np.mean(proms)>2*np.median(proms):
			guess_for_good_pulse = np.mean(proms)
			limit_up = guess_for_good_pulse*2.5
			limit_down = guess_for_good_pulse*0.2
		else:
			guess_for_good_pulse = np.median(proms)
			limit_up = guess_for_good_pulse*1.2
			limit_down = guess_for_good_pulse*0.5
		limit_down_down = cp.deepcopy(limit_down)
	else:
		guess_for_good_pulse = np.max(proms)
		limit_down = np.sum(np.sort(proms)[-2:])/2
		limit_down_down = np.mean(proms)*0.2
		limit_up = np.max(proms)*1.1

	bad_pulses_from_IR = proms_initial[peaks_initial>background_interval]<limit_down_down
	bad_pulses_from_IR = np.arange(1,1+len(bad_pulses_from_IR))[bad_pulses_from_IR]
	# bad_pulses_from_IR = bad_pulses_from_IR[bad_pulses_from_IR<=number_of_pulses]
	if len(bad_pulses_from_IR>0):
		if np.sum(np.diff(bad_pulses_from_IR)>1)>2:
			bad_pulses_from_IR = bad_pulses_from_IR[(np.diff(bad_pulses_from_IR)>1).argmax() : len(bad_pulses_from_IR)-(np.flip(np.diff(bad_pulses_from_IR)>1,axis=0)).argmax()]

	if (len(bad_pulses_indexes)>2 and frequency>400):

		plt.figure(figsize=(20, 10))
		plt.plot((np.arange(len(selected_1_mean))/frequency)[peaks_initial],proms_initial,label='prominence of pulse')
		plt.plot((np.arange(len(selected_1_mean))/frequency)[peaks],proms,'o',label='used pulses for limits')
		plt.plot([0,len(selected_1_mean)/frequency],[np.median(proms),np.median(proms)],'--',label='median')
		plt.plot([0,len(selected_1_mean)/frequency],[np.mean(proms),np.mean(proms)],'--',label='mean')
		plt.plot([0,len(selected_1_mean)/frequency],[limit_down,limit_down],'k--',label='limit of good pulse')
		plt.plot([0,len(selected_1_mean)/frequency],[limit_up,limit_up],'k--')
		plt.plot([(np.arange(len(selected_1_mean))/frequency)[minimum_time]]*2,[proms_initial.max(),proms_initial.min()],'b--',label='minimum time')

		score = []
		for shift in range(np.max(bad_pulses_from_IR)):
			score.append(np.sum([value+shift in bad_pulses_from_IR for value in bad_pulses_indexes]) )
		# peaks = np.array(peaks_initial[peaks_initial>background_interval][np.array(score).argmax():][:number_of_pulses])[np.array([not(value in bad_pulses_indexes) for value in np.arange(1,number_of_pulses+1)])]
		score = np.array(score)
		score[score>np.floor((len(bad_pulses_indexes)-1)/1.5)+1]=0
		if score[score.argmax()]==score[score.argmax()+1]:
			score[score.argmax()+1] +=1
		select = np.array([not((value) in bad_pulses_indexes) for value in np.arange(1,number_of_pulses+1)])
		peaks = np.array(peaks_initial[peaks_initial>background_interval][score.argmax():][:number_of_pulses])
		peaks = peaks[select[:len(peaks)]]
		proms = get_proms(selected_2_max,peaks)[0]

		peaks = peaks[proms>np.median(proms)/10]
		proms = get_proms(selected_2_max,peaks)[0]

		plt.plot((np.arange(len(selected_1_mean))/frequency)[peaks],proms,'x',markersize=20,label='good pulses from OES/TS')
		plt.grid()
		if len(peaks)>0:
			plt.title(pre_title+'Pulse comparison for '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000)+'\nfirst pulse at %.5gs' %((np.arange(len(selected_1_mean))/frequency)[peaks[0]]))
		else:
			plt.title(pre_title+'Pulse comparison for '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))

		very_good_pulses = []
		interval_between_pulses = np.min(np.diff(peaks))
		for i_peak_pos,peak_pos in enumerate(peaks):
			if i_peak_pos < len(peaks)/2:
				continue
			right = peak_pos+int(interval_between_pulses*1/3)
			if right>=len(selected_1_max_counts):
				continue
			very_good_pulses.append(peak_pos)
		strongest_very_good_pulse = very_good_pulses[selected_2_max[very_good_pulses].argmax()]
		very_good_pulses = np.array(very_good_pulses)
		# strongest_very_good_pulse = very_good_pulses[get_proms(selected_2_max,very_good_pulses)[0].argmax()]
		plt.plot((np.arange(len(selected_1_mean))/frequency)[strongest_very_good_pulse],proms[peaks==strongest_very_good_pulse],'y*',markersize=20,label='sample pulse for area check')

		plt.xlabel('time [s]')
		plt.ylabel('Temperature peak prominence [°C]')
		plt.legend(loc='best')
		figure_index+=1
		plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
		plt.close()


	else:
		plt.figure(figsize=(20, 10))
		plt.plot((np.arange(len(selected_1_mean))/frequency)[peaks_initial],proms_initial,label='prominence of pulse')
		plt.plot((np.arange(len(selected_1_mean))/frequency)[peaks],proms,'o',label='used pulses for limits')
		plt.plot([0,len(selected_1_mean)/frequency],[np.median(proms),np.median(proms)],'--',label='median')
		plt.plot([0,len(selected_1_mean)/frequency],[np.mean(proms),np.mean(proms)],'--',label='mean')
		plt.plot([0,len(selected_1_mean)/frequency],[limit_down,limit_down],'k--',label='limit of good pulse')
		plt.plot([0,len(selected_1_mean)/frequency],[limit_up,limit_up],'k--')
		plt.plot([0,len(selected_1_mean)/frequency],[limit_down_down,limit_down_down],'--',label='limit of down down')

		if len(bad_pulses_from_IR>0):
			score = []
			for shift in range(np.max(bad_pulses_from_IR)):
				score.append(np.sum([value+shift in bad_pulses_from_IR for value in bad_pulses_indexes]) )
			if score[np.array(score).argmax()]==score[np.array(score).argmax()+1]:
				score[np.array(score).argmax()+1] +=1
			select = np.array([not((value-1) in bad_pulses_indexes) for value in np.arange(1,number_of_pulses+1)])
			peaks_temp = np.array(peaks_initial[peaks_initial>background_interval][np.array(score).argmax():][:number_of_pulses])
			peaks_temp = peaks_temp[select[:len(peaks_temp)]]
			proms_temp = get_proms(selected_2_max,peaks_temp)[0]
			plt.plot((np.arange(len(selected_1_mean))/frequency)[peaks_temp],proms_temp,'x',markersize=20,label='good pulses from OES/TS')

		peaks = peaks[np.logical_and(proms>limit_down,proms<limit_up)]
		proms = proms[np.logical_and(proms>limit_down,proms<limit_up)]

		very_good_pulses = []
		interval_between_pulses = np.min(np.diff(peaks))
		for i_peak_pos,peak_pos in enumerate(peaks):
			if i_peak_pos < len(peaks)/2:
				continue
			right = peak_pos+int(interval_between_pulses*1/3)
			if right>=len(selected_1_max_counts):
				continue
			very_good_pulses.append(peak_pos)
		strongest_very_good_pulse = very_good_pulses[selected_2_max[very_good_pulses].argmax()]
		very_good_pulses = np.array(very_good_pulses)
		# strongest_very_good_pulse = very_good_pulses[get_proms(selected_2_max,very_good_pulses)[0].argmax()]
		plt.plot((np.arange(len(selected_1_mean))/frequency)[strongest_very_good_pulse],proms[peaks==strongest_very_good_pulse],'y*',markersize=20,label='sample pulse for area check')

		plt.grid()
		plt.title(pre_title+'Pulse comparison for '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000)+'\nfirst pulse at %.5gs' %((np.arange(len(selected_1_mean))/frequency)[peaks][0]))
		plt.xlabel('time [s]')
		plt.ylabel('Temperature peak prominence [°C]')
		plt.legend(loc='best')
		figure_index+=1
		plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
		plt.close('all')

	# sometimes the last pulse is too close to the end of the record, so I exclude it
	peaks = peaks[peaks+round(1.5/1000*frequency)<len(selected_1_max_counts)]
	proms = proms[peaks+round(1.5/1000*frequency)<len(selected_1_max_counts)]

	# print(np.shape(selected_2_max))
	# print(peaks)
	temp = []
	temp2 = []
	temp3 = []
	temp4 = []
	for index in range(10):
		temp.append(selected_1_mean[peaks-index-5])
		temp4.append(selected_1_mean[very_good_pulses-index-5])
		temp2.append(selected_1_max_counts[peaks-index-5])
		temp3.append(selected_2_mean_counts[peaks-index-5])
	ELM_temp_jump = np.median(np.max([selected_1_mean[very_good_pulses-1],selected_1_mean[very_good_pulses],selected_1_mean[very_good_pulses-1]],axis=0) - np.mean(temp4,axis=0))
	ELM_temp_jump_after = np.median(selected_1_mean[very_good_pulses + round(1.5/1000*frequency)] - np.mean(temp4,axis=0))
	# print(np.shape(temp))
	# print(np.shape(np.mean(temp,axis=0)))
	max_temp_before_peaks = np.max(np.mean(temp,axis=0))
	max_counts_before_peaks = np.max(np.mean(temp2,axis=0))
	mean_counts_before_peaks = np.max(np.mean(temp3,axis=0))
	temp_before_peaks = np.mean(temp,axis=0)

	if frequency>400:
		# all_ELM_jumps = np.max([selected_1_mean[peaks-1],selected_1_mean[peaks],selected_1_mean[peaks-1]],axis=0) - temp_before_peaks
		all_ELM_jumps = np.median(selected_1_mean[very_good_pulses + round(1.5/1000*frequency)] - np.mean(temp4,axis=0))
		low_end_interval_ELM_jumps = np.sort(all_ELM_jumps)[int(len(all_ELM_jumps)*0.1)]
		up_end_interval_ELM_jumps = np.sort(all_ELM_jumps)[int(len(all_ELM_jumps)*0.9)]
	else:
		low_end_interval_ELM_jumps = 0
		up_end_interval_ELM_jumps = 0


	peak_image_counts = corrected_frames[strongest_very_good_pulse]
	peak_after_image_counts = corrected_frames[strongest_very_good_pulse+round(1.5/1000*frequency)]
	peak_before_image_counts = np.mean(corrected_frames[strongest_very_good_pulse-15:strongest_very_good_pulse-5],axis=0)
	peak_image = np.exp(counts_to_temperature(peak_image_counts))+20
	peak_after_image = np.exp(counts_to_temperature(peak_after_image_counts))+20
	peak_before_image = np.exp(counts_to_temperature(peak_before_image_counts))+20
	peak_delta_image = peak_after_image-peak_before_image
	peak_emissivity = np.exp(counts_to_emissivity(peak_image_counts))
	peak_after_emissivity = np.exp(counts_to_emissivity(peak_after_image_counts))
	peak_before_emissivity = np.exp(counts_to_emissivity(peak_before_image_counts))

	p2d_h = 25/77	# 25 mm == 77 pixels without sample tilting
	p2d_v = 25/92	# 25 mm == 92 pixels
	max_diameter_to_fit = 24	# mm
	def gaussian_2D_fitting(spatial_coord,*args):
		x = spatial_coord[0]	# horizontal
		y = spatial_coord[1]	# vertical
		out = args[1]*np.exp(- (((x-args[3])*p2d_v/p2d_h)**2 + (y-args[2])**2)/(2*(args[4]**2)) ) + args[0]
		select = ((x-args[3])*p2d_h)**2+((y-args[2])*p2d_v)**2>(max_diameter_to_fit/2)**2
		out[select]=peak_delta_image[select]
		return (out-peak_delta_image).flatten()

	guess = [max(0,peak_delta_image.min()),peak_delta_image.max()-max(0,peak_delta_image.min()),IR_shape[0],IR_shape[1],IR_shape[2]]
	bds=[[0,0,0,0,min(IR_shape[2],10)],[np.inf,np.inf,peak_after_image.shape[0],peak_after_image.shape[1],np.inf]]
	spatial_coord=np.meshgrid(np.arange(np.shape(peak_after_image)[1]),np.arange(np.shape(peak_after_image)[0]))
	fit1 = curve_fit(gaussian_2D_fitting, spatial_coord, np.zeros_like(peak_delta_image.flatten()), guess,bounds=bds)
	x = spatial_coord[0]	# horizontal
	y = spatial_coord[1]	# vertical
	temp = max(1,np.median(np.sort(peak_delta_image[((x-fit1[0][2])*p2d_h)**2+((y-fit1[0][3])*p2d_v)**2<(max_diameter_to_fit/2)**2])[:10]))
	guess = [temp,peak_delta_image.max()-temp,fit1[0][2],fit1[0][3],fit1[0][4]]
	bds=[[0,0,fit1[0][2]-1,fit1[0][3]-1,min(IR_shape[2],10)],[temp,np.inf,fit1[0][2]+1,fit1[0][3]+1,np.inf]]
	spatial_coord=np.meshgrid(np.arange(np.shape(peak_after_image)[1]),np.arange(np.shape(peak_after_image)[0]))
	fit2 = curve_fit(gaussian_2D_fitting, spatial_coord, np.zeros_like(peak_delta_image.flatten()), guess,bounds=bds)
	area_of_interest = 2*np.pi*(fit2[0][4]*p2d_v/1000)**2	# Area equivalent to all the overtemperature profile collapsed in a homogeneous circular spot
	area_of_interest_radius = 2**0.5 *fit2[0][4]
	area_of_interest_sigma = 2*np.pi*fit2[1][4,4]*(p2d_v/1000)**2

	plt.figure(figsize=(20, 10))
	plt.imshow(peak_image,'rainbow',vmax=np.median(np.sort(peak_image.flatten())[-10:]))
	plt.colorbar().set_label('Temperature [°C]')
	plt.plot(IR_shape[1],IR_shape[0],'k+')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--',label='externally supplied')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.errorbar(fit1[0][3],fit1[0][2],xerr=fit1[1][3,3]**0.5,yerr=fit1[1][2,2]**0.5,color='c')
	# plt.plot(fit1[0][3] + np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)*p2d_v/p2d_h,fit1[0][2] + np.abs(fit1[0][4]**2-np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)**2)**0.5,'c--',label='fitted\nTmin=%.3g°C, dt=%.3g°C' %(fit1[0][0],fit1[0][1]))
	# plt.plot(fit1[0][3] + np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)*p2d_v/p2d_h,fit1[0][2] - np.abs(fit1[0][4]**2-np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)**2)**0.5,'c--')
	plt.plot(fit1[0][3] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit1[0][2] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--',label='target diameter')
	plt.plot(fit1[0][3] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit1[0][2] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--')
	plt.errorbar(fit2[0][3],fit2[0][2],xerr=fit2[1][3,3]**0.5,yerr=fit2[1][2,2]**0.5,color='b')
	plt.plot(fit2[0][3] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h,fit2[0][2] + np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5,'b--',label='fitted')
	plt.plot(fit2[0][3] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h,fit2[0][2] - np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5,'b--')
	# plt.plot(fit2[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][3] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--',label='target diameter')
	# plt.plot(fit2[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][3] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--')
	plt.legend(loc='best', fontsize='x-small')
	plt.xlabel('horizontal coord [pixels]')
	plt.ylabel('vertical coord [pixels]')
	plt.title(pre_title+'Temperature distribution for the strongest peak (%.5gs) in ' %(strongest_very_good_pulse/frequency)+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.imshow(peak_after_image,'rainbow',vmax=np.median(np.sort(peak_after_image.flatten())[-10:]))
	plt.colorbar().set_label('Temperature [°C]')
	plt.plot(IR_shape[1],IR_shape[0],'k+')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--',label='externally supplied')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.errorbar(fit1[0][3],fit1[0][2],xerr=fit1[1][3,3]**0.5,yerr=fit1[1][2,2]**0.5,color='c')
	# plt.plot(fit1[0][3] + np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)*p2d_v/p2d_h,fit1[0][2] + np.abs(fit1[0][4]**2-np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)**2)**0.5,'c--',label='fitted\nTmin=%.3g°C, dt=%.3g°C' %(fit1[0][0],fit1[0][1]))
	# plt.plot(fit1[0][3] + np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)*p2d_v/p2d_h,fit1[0][2] - np.abs(fit1[0][4]**2-np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)**2)**0.5,'c--')
	plt.plot(fit1[0][3] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit1[0][2] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--',label='target diameter')
	plt.plot(fit1[0][3] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit1[0][2] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--')
	plt.errorbar(fit2[0][3],fit2[0][2],xerr=fit2[1][3,3]**0.5,yerr=fit2[1][2,2]**0.5,color='b')
	plt.plot(fit2[0][3] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h,fit2[0][2] + np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5,'b--',label='fitted')
	plt.plot(fit2[0][3] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h,fit2[0][2] - np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5,'b--')
	# plt.plot(fit2[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][3] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--',label='target diameter')
	# plt.plot(fit2[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][3] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--')
	plt.legend(loc='best', fontsize='x-small')
	plt.xlabel('horizontal coord [pixels]')
	plt.ylabel('vertical coord [pixels]')
	plt.title(pre_title+'Temperature distribution 1.5ms after the strongest peak in '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.imshow(peak_before_image,'rainbow',vmax=np.median(np.sort(peak_before_image.flatten())[-10:]))
	plt.colorbar().set_label('Temperature [°C]')
	plt.plot(IR_shape[1],IR_shape[0],'k+')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	# plt.errorbar(fit1[0][3],fit1[0][2],xerr=fit1[1][3,3]**0.5,yerr=fit1[1][2,2]**0.5,color='c')
	# plt.plot(fit1[0][3] + np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)*p2d_v/p2d_h,fit1[0][2] + np.abs(fit1[0][4]**2-np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)**2)**0.5,'c--',label='fitted\nTmin=%.3g°C, dt=%.3g°C' %(fit1[0][0],fit1[0][1]))
	# plt.plot(fit1[0][3] + np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)*p2d_v/p2d_h,fit1[0][2] - np.abs(fit1[0][4]**2-np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)**2)**0.5,'c--')
	plt.plot(fit1[0][3] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit1[0][2] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--',label='target diameter')
	plt.plot(fit1[0][3] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit1[0][2] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--')
	plt.errorbar(fit2[0][3],fit2[0][2],xerr=fit2[1][3,3]**0.5,yerr=fit2[1][2,2]**0.5,color='b')
	plt.plot(fit2[0][3] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h,fit2[0][2] + np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5,'b--',label='fitted')
	plt.plot(fit2[0][3] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h,fit2[0][2] - np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5,'b--')
	plt.legend(loc='best', fontsize='x-small')
	plt.xlabel('horizontal coord [pixels]')
	plt.ylabel('vertical coord [pixels]')
	plt.title(pre_title+'Temperature distribution before the strongest peak (average of %.3gms to %.3gms) in ' %(-15/frequency*1000,-6/frequency*1000)+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.imshow(peak_delta_image,'rainbow',vmin=0,vmax=np.mean(np.sort(peak_delta_image[((x-fit2[0][2])*p2d_h)**2+((y-fit2[0][3])*p2d_v)**2<(max_diameter_to_fit/2)**2])[-20:]))
	plt.colorbar().set_label('Temperature [°C] limited to 0')
	plt.plot(IR_shape[1],IR_shape[0],'k+')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--',label='externally supplied')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	plt.errorbar(fit1[0][3],fit1[0][2],xerr=fit1[1][3,3]**0.5,yerr=fit1[1][2,2]**0.5,color='c',label='fitted\nTmin=%.3g°C, dt=%.3g°C' %(fit1[0][0],fit1[0][1]))
	# plt.plot(fit1[0][3] + np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)*p2d_v/p2d_h,fit1[0][2] + np.abs(fit1[0][4]**2-np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)**2)**0.5,'c--',label='fitted\nTmin=%.3g°C, dt=%.3g°C' %(fit1[0][0],fit1[0][1]))
	# plt.plot(fit1[0][3] + np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)*p2d_v/p2d_h,fit1[0][2] - np.abs(fit1[0][4]**2-np.arange(-fit1[0][4],+fit1[0][4]+fit1[0][4]/10,fit1[0][4]/10)**2)**0.5,'c--')
	plt.plot(fit1[0][3] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit1[0][2] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--',label='target diameter')
	plt.plot(fit1[0][3] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit1[0][2] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--')
	plt.errorbar(fit2[0][3],fit2[0][2],xerr=fit2[1][3,3]**0.5,yerr=fit2[1][2,2]**0.5,color='b')
	plt.plot(fit2[0][3] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h,fit2[0][2] + np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5,'b--',label='fitted\nTmin=%.3g°C, dt=%.3g°C\narea=%.3g+/-%.3gm2' %(fit2[0][0],fit2[0][1],area_of_interest,area_of_interest_sigma))
	plt.plot(fit2[0][3] + np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)*p2d_v/p2d_h,fit2[0][2] - np.abs(area_of_interest_radius**2-np.arange(-area_of_interest_radius,+area_of_interest_radius+area_of_interest_radius/10,area_of_interest_radius/10)**2)**0.5,'b--')
	# plt.plot(fit2[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][3] + (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--',label='target diameter')
	# plt.plot(fit2[0][2] + np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/p2d_h/2,fit2[0][3] - (np.abs((max_diameter_to_fit/2)**2-(np.arange(-max_diameter_to_fit,+max_diameter_to_fit+max_diameter_to_fit/10,max_diameter_to_fit/10)/2)**2)**0.5)/p2d_v,'g--')
	plt.legend(loc='best', fontsize='x-small')
	plt.xlabel('horizontal coord [pixels]')
	plt.ylabel('vertical coord [pixels]')
	plt.title(pre_title+'Temperature increase 1.5ms - average of %.3gms to %.3gms around the strongest peak in ' %(-15/frequency*1000,-6/frequency*1000)+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.imshow(peak_emissivity,'rainbow')
	plt.colorbar().set_label('Emissivity [au]')
	plt.plot(IR_shape[1],IR_shape[0],'k+')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	plt.xlabel('horizontal coord [pixels]')
	plt.ylabel('vertical coord [pixels]')
	plt.title(pre_title+'Emissivity distribution for the strongest peak (%.5gs) in ' %(strongest_very_good_pulse/frequency)+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	plt.imshow(peak_before_emissivity,'rainbow')
	plt.colorbar().set_label('Emissivity [au]')
	plt.plot(IR_shape[1],IR_shape[0],'k+')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] + np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10),IR_shape[0] - np.abs(IR_shape[2]**2-np.arange(-IR_shape[2],+IR_shape[2]+IR_shape[2]/10,IR_shape[2]/10)**2)**0.5,'k--')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] + np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	plt.plot(IR_shape[1] + np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10),IR_shape[0] - np.abs(IR_shape[3]**2-np.arange(-IR_shape[3],+IR_shape[3]+IR_shape[3]/10,IR_shape[3]/10)**2)**0.5,'k--')
	plt.xlabel('horizontal coord [pixels]')
	plt.ylabel('vertical coord [pixels]')
	plt.title(pre_title+'Emissivity distribution before for the strongest peak (average of %.3gms to %.3gms) in ' %(-10/frequency,-6/frequency)+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()


	# plt.figure()
	peak_shape = []
	peak_shape_time = []
	peak_mean_shape = []
	interval_to_find_peak = int(0.3e-3*frequency*10)
	for i_peak_pos,peak_pos in enumerate(very_good_pulses):
		# if i_peak_pos < len(peaks)/2:
		# 	continue
		left = peak_pos-int(3*frequency/1000)
		right = peak_pos+int(interval_between_pulses*1/3)
		# if right>=len(selected_1_max_counts):
		# 	continue
		traces_corrected = selected_1_max_counts[left:right]
		traces_corrected_mean = selected_2_mean_counts[left:right]
		# tck = interpolate.splrep(np.arange(right-left)/frequency, traces_corrected,s=20,k=2)
		# interpolated = interpolate.splev(np.arange((right-left)*10)/10/frequency, tck)
		tck = interpolate.interp1d(np.arange(right-left)/frequency, traces_corrected,bounds_error=False,fill_value='extrapolate')
		interpolated = tck(np.arange((right-left)*10)/10/frequency)
		interpolated = np.convolve(interpolated, np.ones((interval_to_find_peak))/interval_to_find_peak , mode='same')
		real_peak_loc = (np.arange((right-left)*10)/10)[traces_corrected.argmax()*10-20+(interpolated[traces_corrected.argmax()*10-20:traces_corrected.argmax()*10+20].argmax())]
		# plt.plot(np.arange((right-left)*10)/10 - real_peak_loc,interpolated,'--')
		# plt.plot(np.arange(right-left) - real_peak_loc,traces_corrected)
		peak_shape.append(np.interp(np.arange((right-left)*10)/10,np.arange(right-left),traces_corrected-np.mean(selected_1_max_counts[peak_pos-15:peak_pos-5])))
		peak_mean_shape.append(np.interp(np.arange((right-left)*10)/10,np.arange(right-left),traces_corrected_mean-np.mean(selected_2_mean_counts[peak_pos-15:peak_pos-5])))
		peak_shape_time.append((np.arange((right-left)*10)/10 - real_peak_loc)/frequency)
	fig, ax = plt.subplots( 2,4,figsize=(28, 14), squeeze=False)
	fig.suptitle(pre_title+'Shape of the max temperature increase for '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	plot_index = 0
	ax[0,0].set_title('small area max counts')
	im = ax[0,0].plot(1000*np.array(peak_shape_time).T,np.array(peak_shape).T,linewidth=0.5)
	ax[1,0].set_title('large area mean counts')
	im = ax[1,0].plot(1000*np.array(peak_shape_time).T,np.array(peak_mean_shape).T,linewidth=0.5)

	ax[0,1].set_title('small area max counts')
	im = ax[0,1].plot(1000*np.array(peak_shape_time).T,np.array(peak_shape).T,linewidth=0.5)
	ax[1,1].set_title('large area mean counts')
	im = ax[1,1].plot(1000*np.array(peak_shape_time).T,np.array(peak_mean_shape).T,linewidth=0.5)
	peak_shape = np.array(peak_shape).flatten()
	peak_mean_shape = np.array(peak_mean_shape).flatten()
	peak_shape_time = np.array(peak_shape_time).flatten()
	ensamble_peak_shape = []
	ensamble_peak_mean_shape = []
	ensamble_peak_time = np.unique(peak_shape_time)
	ensamble_peak_time = ensamble_peak_time[:-1][np.diff(ensamble_peak_time)>1/frequency/100]
	for time in ensamble_peak_time:
		ensamble_peak_shape.append(np.median(peak_shape[np.logical_and(peak_shape_time>time-1/frequency/100,peak_shape_time<time+1/frequency/100)]))
		ensamble_peak_mean_shape.append(np.median(peak_mean_shape[np.logical_and(peak_shape_time>time-1/frequency/100,peak_shape_time<time+1/frequency/100)]))
	all_time_points = len(peak_shape_time)
	peak_shape_time = ensamble_peak_time
	im = ax[0,0].plot(1000*peak_shape_time,ensamble_peak_shape,'--m',linewidth=2)
	im = ax[1,0].plot(1000*peak_shape_time,ensamble_peak_mean_shape,'--m',linewidth=2)

	im = ax[0,1].plot(1000*peak_shape_time,ensamble_peak_shape,'--m',linewidth=2)
	im = ax[1,1].plot(1000*peak_shape_time,ensamble_peak_mean_shape,'--m',linewidth=2)
	window_size = 5
	ensamble_peak_shape = scipy.signal.savgol_filter(ensamble_peak_shape, window_size, 2)
	ensamble_peak_mean_shape = scipy.signal.savgol_filter(ensamble_peak_mean_shape, window_size, 2)
	im = ax[0,0].plot(1000*peak_shape_time,ensamble_peak_shape,'r',linewidth=2,label='mean profile')
	im = ax[0,0].plot([1.5]*2,[ensamble_peak_shape.min(),ensamble_peak_shape.max()],'--g')
	im = ax[1,0].plot(1000*peak_shape_time,ensamble_peak_mean_shape,'r',linewidth=2,label='mean profile')
	im = ax[1,0].plot([1.5]*2,[ensamble_peak_mean_shape.min(),ensamble_peak_mean_shape.max()],'--g')
	# if all_time_points>2*len(peak_shape_time):
	# 	ensamble_peak_shape = np.convolve(ensamble_peak_shape,np.ones(20)/20,'same')
	# im = ax[plot_index,0].plot(1000*peak_shape_time,ensamble_peak_shape,'r',linewidth=3)
	ax[0,0].set_xlabel('time [ms]')
	ax[0,0].set_ylabel('counts [au]')
	ax[1,0].set_xlabel('time [ms]')
	ax[1,0].set_ylabel('counts [au]')

	im = ax[0,1].plot(1000*peak_shape_time,ensamble_peak_shape,'r',linewidth=2,label='mean profile')
	im = ax[0,1].plot([1.5]*2,[ensamble_peak_shape.min(),ensamble_peak_shape.max()],'--g')
	im = ax[1,1].plot(1000*peak_shape_time,ensamble_peak_mean_shape,'r',linewidth=2,label='mean profile')
	im = ax[1,1].plot([1.5]*2,[ensamble_peak_mean_shape.min(),ensamble_peak_mean_shape.max()],'--g')
	# if all_time_points>2*len(peak_shape_time):
	# 	ensamble_peak_shape = np.convolve(ensamble_peak_shape,np.ones(20)/20,'same')
	# im = ax[plot_index,0].plot(1000*peak_shape_time,ensamble_peak_shape,'r',linewidth=3)
	ax[0,1].set_xlabel('time [ms]')
	ax[0,1].set_ylabel('counts [au]')
	ax[1,1].set_xlabel('time [ms]')
	ax[1,1].set_ylabel('counts [au]')

	try:
		if False:	# Here I wanted to find the slope of the temperature rise and find it in the temperature decrease.
			start_start_pulse = (ensamble_peak_shape>ensamble_peak_shape.max()*0.01).argmax()
			end_start_pulse = (ensamble_peak_shape>ensamble_peak_shape.max()*0.3).argmax()
			end_end_pulse = -(np.flip(ensamble_peak_shape,axis=0)>ensamble_peak_shape.max()*0.1).argmax()
			start_end_pulse = -(np.flip(ensamble_peak_shape,axis=0)>ensamble_peak_shape.max()*0.5).argmax()
			tau_up = np.polyfit(peak_shape_time[start_start_pulse:end_start_pulse],np.log(ensamble_peak_shape[start_start_pulse:end_start_pulse]),1)
			# plt.figure()
			# plt.plot(peak_shape_time[start_start_pulse:end_start_pulse],np.log(ensamble_peak_shape[start_start_pulse:end_start_pulse]))
			plt.plot(1000*peak_shape_time[start_start_pulse:end_start_pulse],np.exp(np.polyval(tau_up,peak_shape_time[start_start_pulse:end_start_pulse])),'--b')
			def to_fit(x,constant):
				out = np.exp(np.polyval([-tau_up[0],constant],x))
				return out
			guess=[1]
			fit = curve_fit(to_fit, peak_shape_time[start_end_pulse:end_end_pulse], ensamble_peak_shape[start_end_pulse:end_end_pulse], guess)
			# plt.plot(peak_shape_time[start_end_pulse:end_end_pulse],np.log(ensamble_peak_shape[start_end_pulse:end_end_pulse]))
			plt.plot(1000*peak_shape_time[ensamble_peak_shape.argmax():end_end_pulse],np.exp(np.polyval([-tau_up[0],fit[0]],peak_shape_time[ensamble_peak_shape.argmax():end_end_pulse])),'--b',linewidth=3,label='time constant=%.3g' %(tau_up[0]))
			# plt.pause(0.01)
		elif True:	# we now believe that most of the peak is actually due to prompt emission of infrared line emission, so not due to an actual temperature increase
			start_start_pulse = (ensamble_peak_shape>ensamble_peak_shape.max()*0.6).argmax()
			end_end_pulse = -(np.flip(ensamble_peak_shape,axis=0)>ensamble_peak_shape.max()*0.6).argmax()
			pre_pulse = np.abs(peak_shape_time*1000+2).argmin()
			post_pulse = np.abs(peak_shape_time*1000-2).argmin()
			peak_pulse = ensamble_peak_shape.argmax()
			# gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
			# guess=[peak_shape_time.max(),(peak_shape_time[peak_pulse]-peak_shape_time[start_start_pulse])/2,peak_shape_time[peak_pulse]]
			gauss = lambda x, sig, x0: ensamble_peak_shape.max() * np.exp(-(((x - x0) / sig) ** 2)/2) + ensamble_peak_shape[pre_pulse]
			guess=[(peak_shape_time[peak_pulse]-peak_shape_time[start_start_pulse])/2,peak_shape_time[peak_pulse]]
			# peak_pulse = np.abs(peak_shape_time - (peak_shape_time[peak_pulse]+2e-4)).argmin()
			fit = curve_fit(gauss, peak_shape_time[start_start_pulse:end_end_pulse], ensamble_peak_shape[start_start_pulse:end_end_pulse], guess,maxfev=int(1e4))
			im = ax[0,0].plot(1000*peak_shape_time[start_start_pulse:end_end_pulse],gauss(peak_shape_time[start_start_pulse:end_end_pulse],*fit[0]),'--b',linewidth=2,label='peak fit')
			im = ax[0,0].plot(1000*peak_shape_time[pre_pulse:start_start_pulse],gauss(peak_shape_time[pre_pulse:start_start_pulse],*fit[0]),':b',label='raise extension')
			im = ax[0,1].plot(1000*peak_shape_time[start_start_pulse:end_end_pulse],gauss(peak_shape_time[start_start_pulse:end_end_pulse],*fit[0]),'--b',linewidth=2,label='peak fit')
			im = ax[0,1].plot(1000*peak_shape_time[pre_pulse:start_start_pulse],gauss(peak_shape_time[pre_pulse:start_start_pulse],*fit[0]),':b',label='raise extension')
			peak_pulse = np.abs(peak_shape_time-fit[0][-1]).argmin()
			end_end_pulse = np.abs(peak_shape_time-fit[0][-1]-fit[0][-2]*4).argmin()
			im = ax[0,0].plot([1000*fit[0][-1]]*2,[ensamble_peak_shape.min(),np.max(peak_shape)],'--y',label='prompt peak')
			im = ax[1,0].plot([1000*fit[0][-1]]*2,[ensamble_peak_mean_shape.min(),np.max(peak_mean_shape)],'--y',label='prompt peak')
			im = ax[0,1].plot([1000*fit[0][-1]]*2,[ensamble_peak_shape.min(),np.max(peak_shape)],'--y',label='prompt peak')
			im = ax[1,1].plot([1000*fit[0][-1]]*2,[ensamble_peak_mean_shape.min(),np.max(peak_mean_shape)],'--y',label='prompt peak')
			post_pulse = np.abs(peak_shape_time-(2*peak_shape_time[peak_pulse] - peak_shape_time[pre_pulse])).argmin()
			im = ax[0,0].plot(1000*peak_shape_time[peak_pulse:post_pulse],ensamble_peak_shape[peak_pulse:post_pulse] - gauss(peak_shape_time[peak_pulse:post_pulse],*fit[0]),'-b',linewidth=2,label='decrease-fit')
			im = ax[0,1].plot(1000*peak_shape_time[peak_pulse:post_pulse],ensamble_peak_shape[peak_pulse:post_pulse] - gauss(peak_shape_time[peak_pulse:post_pulse],*fit[0]),'-b',linewidth=2,label='decrease-fit')
			interpolator = interp1d(peak_shape_time[pre_pulse:peak_pulse+1]-fit[0][-1],ensamble_peak_shape[pre_pulse:peak_pulse+1],bounds_error=False,fill_value=(ensamble_peak_shape[pre_pulse],ensamble_peak_shape[peak_pulse]))
			interpolator_mean = interp1d(peak_shape_time[pre_pulse:peak_pulse+1]-fit[0][-1],ensamble_peak_mean_shape[pre_pulse:peak_pulse+1],bounds_error=False,fill_value=(ensamble_peak_mean_shape[pre_pulse],ensamble_peak_mean_shape[peak_pulse]))
			im = ax[0,0].plot(1000*peak_shape_time[peak_pulse:post_pulse],interpolator(fit[0][-1]-peak_shape_time[peak_pulse:post_pulse]),'--k',linewidth=1,label='mirror of raise')
			im = ax[1,0].plot(1000*peak_shape_time[peak_pulse:post_pulse],interpolator_mean(fit[0][-1]-peak_shape_time[peak_pulse:post_pulse]),'--k',linewidth=1,label='mirror of raise')
			im = ax[0,1].plot(1000*peak_shape_time[peak_pulse:post_pulse],interpolator(fit[0][-1]-peak_shape_time[peak_pulse:post_pulse]),'--k',linewidth=1,label='mirror of raise')
			im = ax[1,1].plot(1000*peak_shape_time[peak_pulse:post_pulse],interpolator_mean(fit[0][-1]-peak_shape_time[peak_pulse:post_pulse]),'--k',linewidth=1,label='mirror of raise')

			ax[0,0].set_xlim(left=-2,right=2)
			ax[0,0].set_ylim(bottom=0,top=1.5*ensamble_peak_shape.max())
			ax[0,0].grid()
			ax[0,1].set_xlim(left=0,right=15)
			ax[0,1].set_ylim(bottom=-1,top=max(3,ensamble_peak_shape[np.abs(peak_shape_time-1e-3).argmin()]))
			ax[0,1].grid()
			ensamble_peak_shape_full = ensamble_peak_shape
			ensamble_peak_shape = ensamble_peak_shape[peak_pulse:]-interpolator(fit[0][-1]-peak_shape_time[peak_pulse:])
			im = ax[0,0].plot(1000*peak_shape_time[peak_pulse:],ensamble_peak_shape,'-k',linewidth=2,label='decrease-mirror')
			ax[1,0].set_xlim(left=-2,right=2)
			ax[1,0].set_ylim(bottom=0,top=1.5*ensamble_peak_mean_shape.max())
			ax[1,0].grid()
			ax[1,1].set_xlim(left=0,right=15)
			ax[1,1].set_ylim(bottom=-1,top=max(3,ensamble_peak_mean_shape[np.abs(peak_shape_time-1e-3).argmin()]))
			ax[1,1].grid()
			ensamble_peak_mean_shape_full = ensamble_peak_mean_shape
			ensamble_peak_mean_shape = ensamble_peak_mean_shape[peak_pulse:]-interpolator_mean(fit[0][-1]-peak_shape_time[peak_pulse:])
			im = ax[1,0].plot(1000*peak_shape_time[peak_pulse:],ensamble_peak_mean_shape,'-k',linewidth=2,label='decrease-mirror')
			ax[0,0].legend(loc='best', fontsize='x-small')
			ax[1,0].legend(loc='best', fontsize='x-small')
			ax[0,1].legend(loc='best', fontsize='x-small')
			ax[1,1].legend(loc='best', fontsize='x-small')
			plot_index += 1

			ensamble_peak_shape[ensamble_peak_shape<0]=0
			ensamble_peak_shape += max_counts_before_peaks
			ensamble_peak_shape_full += max_counts_before_peaks
			ensamble_peak_mean_shape[ensamble_peak_mean_shape<0]=0
			ensamble_peak_mean_shape += mean_counts_before_peaks
			ensamble_peak_mean_shape_full += mean_counts_before_peaks

			ensamble_peak_shape = np.exp(counts_to_temperature(ensamble_peak_shape))+20
			ensamble_peak_mean_shape = np.exp(counts_to_temperature(ensamble_peak_mean_shape))+20
			ensamble_peak_shape_full = np.exp(counts_to_temperature(ensamble_peak_shape_full))+20
			ensamble_peak_mean_shape_full = np.exp(counts_to_temperature(ensamble_peak_mean_shape_full))+20
			emissivity_ensamble_peak_shape = np.exp(counts_to_emissivity(ensamble_peak_shape))
			emissivity_ensamble_peak_mean_shape = np.exp(counts_to_emissivity(ensamble_peak_mean_shape))
			emissivity_ensamble_peak_mean_shape_full = np.exp(counts_to_emissivity(ensamble_peak_mean_shape_full))

			temp_time = peak_shape_time[peak_pulse:]
			im = ax[0,2].plot(1000*temp_time,ensamble_peak_shape,'-k',linewidth=2)
			im = ax[0,2].plot([1000*temp_time[ensamble_peak_shape.argmax()]]*2,[ensamble_peak_shape.min(),ensamble_peak_shape.max()],'--',label='dT=%.3g' %(ensamble_peak_shape[ensamble_peak_shape.argmax()]-ensamble_peak_shape.min()))
			im = ax[0,2].plot([1000*temp_time[np.abs(temp_time-1e-3).argmin()]]*2,[ensamble_peak_shape.min(),ensamble_peak_shape.max()],'--',label='dT=%.3g' %(ensamble_peak_shape[np.abs(temp_time-1e-3).argmin()]-ensamble_peak_shape.min()))
			im = ax[0,2].plot([1000*temp_time[np.abs(temp_time-1.5e-3).argmin()]]*2,[ensamble_peak_shape.min(),ensamble_peak_shape.max()],'--g',label='dT=%.3g' %(ensamble_peak_shape[np.abs(temp_time-1.5e-3).argmin()]-ensamble_peak_shape.min()))
			im = ax[0,3].plot(1000*peak_shape_time,ensamble_peak_shape_full,'k')
			im = ax[1,2].plot(1000*temp_time,ensamble_peak_mean_shape,'-k',linewidth=2)
			im = ax[1,2].plot([1000*temp_time[ensamble_peak_mean_shape.argmax()]]*2,[ensamble_peak_mean_shape.min(),ensamble_peak_mean_shape.max()],'--',label='dT=%.3g' %(ensamble_peak_mean_shape[ensamble_peak_mean_shape.argmax()]-ensamble_peak_mean_shape.min()))
			im = ax[1,2].plot([1000*temp_time[np.abs(temp_time-1e-3).argmin()]]*2,[ensamble_peak_mean_shape.min(),ensamble_peak_mean_shape.max()],'--',label='dT=%.3g' %(ensamble_peak_mean_shape[np.abs(temp_time-1e-3).argmin()]-ensamble_peak_mean_shape.min()))
			im = ax[1,2].plot([1000*temp_time[np.abs(temp_time-1.5e-3).argmin()]]*2,[ensamble_peak_mean_shape.min(),ensamble_peak_mean_shape.max()],'--g',label='dT=%.3g' %(ensamble_peak_mean_shape[np.abs(temp_time-1.5e-3).argmin()]-ensamble_peak_mean_shape.min()))
			im = ax[1,3].plot(1000*peak_shape_time,ensamble_peak_mean_shape_full,'k')

			end_end_pulse = np.abs(temp_time*1000-1.5).argmin()
			post_pulse = np.abs(temp_time*1000-30).argmin()
			SS_time = int(1e-3*frequency*10)
			# temp = np.log(ensamble_peak_shape[end_end_pulse:post_pulse]-ensamble_peak_shape[-SS_time:].mean())
			# tau_down = np.polyfit(temp_time[end_end_pulse:post_pulse][np.isfinite(temp)],temp[np.isfinite(temp)],1)
			# min_target_temperature = np.exp(np.polyval((tau_down),temp_time))+ensamble_peak_shape[-SS_time:].mean()

			sigmaSB=5.6704e-08 #[W/(m2 K4)]
			if False:	# the exponential decay is an idea I had based on heat transport, but it is not representative of heat transport in time!
				def exponential_decay(time,*args):
					out = args[1]*np.exp(-time/args[2]) + args[0]
					return out

				guess = [ensamble_peak_shape[-SS_time:].mean(),ensamble_peak_shape[np.abs(temp_time-1e-3).argmin()],3e-3]
				bds=[[20,0,1e-3],[np.inf,np.inf,1e-1]]
				fit = curve_fit(exponential_decay, temp_time[end_end_pulse:post_pulse], ensamble_peak_shape[end_end_pulse:post_pulse], guess,bounds=bds,maxfev=int(1e4))
				min_target_temperature = exponential_decay(temp_time,*fit[0])
				# # area_of_interest = 2*np.pi*IR_shape[3]**2 * (0.3e-3)**2
				if False:	# fixed area
					area_of_interest = 2*np.pi*(0.026/2)**2	# 26mm is the diameter of the standard magnum target area_of_interest_sigma
				sigmaSB=5.6704e-08 #[W/(m2 K4)]
				# target_effected_thickness = 0.2e-3
				target_effected_thickness = 0.2e-3	# 1mm thickness of the target (approx.)
				# min_power = thermal_conductivity*(np.exp(np.polyval((tau_down),temp_time)))/target_effected_thickness + 1*sigmaSB*((min_target_temperature+273.15)**4 - 300**4)
				# min_power = area_of_interest*np.sum(min_power)/(10*frequency)
				# max_power = heat_capacity*density*area_of_interest*target_effected_thickness*np.exp(np.polyval((tau_down),0))
				min_power = thermal_conductivity*(min_target_temperature-fit[0][0])/target_effected_thickness + emissivity_ensamble_peak_shape*sigmaSB*((ensamble_peak_shape_full[peak_pulse:]+273.15)**4 - 300**4)
				min_power = area_of_interest*np.sum(min_power)/(10*frequency)
				# max_power = thermal_conductivity*(ensamble_peak_shape_full-ensamble_peak_shape_full.min())/target_effected_thickness + emissivity_ensamble_peak_shape_full*sigmaSB*((ensamble_peak_shape_full+273.15)**4 - 300**4)
				# max_power = area_of_interest*np.sum(max_power)/frequency
				max_power = heat_capacity*density*area_of_interest*target_effected_thickness*(fit[0][1])
				ax[0,2].plot(1000*temp_time,min_target_temperature,'--r',linewidth=2,label='T SS=%.3g°C\ndT=%.3g°C\ntau=%.3gms\nEnergy\nfrom area=%.3gJ\nfrom Tmax=%.3gJ' %(fit[0][0],fit[0][1],fit[0][2]*1000,min_power,max_power))
				ax[0,3].plot(1000*temp_time,min_target_temperature,'--r',linewidth=2)
				ax[0,2].set_ylim(bottom=fit[0][0]-2,top=10+min_target_temperature.max())

			# this aproach comes from a reference given me by Bruce: EVAPORATION FOR HEAT PULSES ON Ni, MO, W AND ATJ GRAPHITE AS FIRST WALL MATERIALS
			def semi_infinite_sink(time,*args):
				out = args[1]*2/((np.pi*thermal_conductivity*heat_capacity*density)**0.5) *(time**0.5 - (time-args[2])**0.5) + args[0]
				# print(out)
				return out

			guess = [ensamble_peak_shape[-SS_time:].mean(),1e6,1e-3]
			bds=[[20,0,1e-5],[np.inf,np.inf,temp_time[end_end_pulse]]]
			fit = curve_fit(semi_infinite_sink, temp_time[end_end_pulse:post_pulse], ensamble_peak_shape[end_end_pulse:post_pulse], guess,bounds=bds,maxfev=int(1e4))
			ax[0,2].plot(1000*temp_time,semi_infinite_sink(temp_time,*fit[0]),'--m',linewidth=2,label='semi-infinite solid\nT SS=%.3g°C\nEdens=%.3gJ/m2\ntau=%.3gms\nEnergy=%.3gJ' %(fit[0][0],fit[0][1]*fit[0][2],fit[0][2]*1000,fit[0][1]*fit[0][2]*area_of_interest))
			ax[0,3].plot(1000*temp_time,semi_infinite_sink(temp_time,*fit[0]),'--m',linewidth=2)
			ax[0,2].set_ylim(bottom=fit[0][0]-2,top=10+semi_infinite_sink(1.5e-3,*fit[0]))

			df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
			# df_log.loc[j,['pulse_en [J]']] = min_power
			df_log.loc[j,['pulse_en_semi_inf [J]']] = fit[0][1]*fit[0][2]*area_of_interest
			df_log.loc[j,['area_of_interest [m2]']] = area_of_interest
			df_log.to_csv(path_or_buf='/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv')

			if False:
				guess = [ensamble_peak_mean_shape[-SS_time:].mean(),ensamble_peak_mean_shape[np.abs(temp_time-1e-3).argmin()],3e-3]
				bds=[[20,0,1e-3],[np.inf,np.inf,1e-1]]
				fit = curve_fit(exponential_decay, temp_time[end_end_pulse:post_pulse], ensamble_peak_mean_shape[end_end_pulse:post_pulse], guess,bounds=bds,maxfev=int(1e4))
				min_target_temperature = exponential_decay(temp_time,*fit[0])
				if False:	# fixed area
					area_of_interest = 2*np.pi*(0.026/2)**2	# 26mm is the diameter of the standard magnum target
				sigmaSB=5.6704e-08 #[W/(m2 K4)]
				# target_effected_thickness = 0.7e-3	# 1mm thickness of the target (approx.)
				min_power = thermal_conductivity*(min_target_temperature-fit[0][0])/target_effected_thickness + emissivity_ensamble_peak_shape*sigmaSB*((ensamble_peak_mean_shape_full[peak_pulse:]+273.15)**4 - 300**4)
				min_power = area_of_interest*np.sum(min_power)/(10*frequency)
				max_power = heat_capacity*density*area_of_interest*target_effected_thickness*(fit[0][1])
				ax[1,2].plot(1000*temp_time,min_target_temperature,'--r',linewidth=2,label='T SS=%.3g°C\ndT=%.3g°C\ntau=%.3gms\nEnergy\nfrom area=%.3gJ\nfrom Tmax=%.3gJ' %(fit[0][0],fit[0][1],fit[0][2]*1000,min_power,max_power))
				ax[1,3].plot(1000*temp_time,min_target_temperature,'--r',linewidth=2)
				ax[1,2].set_ylim(bottom=fit[0][0]-2,top=10+min_target_temperature.max())

			guess = [ensamble_peak_mean_shape[-SS_time:].mean(),1e6,1e-3]
			bds=[[20,0,1e-5],[np.inf,np.inf,temp_time[end_end_pulse]]]
			fit = curve_fit(semi_infinite_sink, temp_time[end_end_pulse:post_pulse], ensamble_peak_mean_shape[end_end_pulse:post_pulse], guess,bounds=bds,maxfev=int(1e4))
			ax[1,2].plot(1000*temp_time,semi_infinite_sink(temp_time,*fit[0]),'--m',linewidth=2,label='semi-infinite solid\nT SS=%.3g°C\nEdens=%.3gJ/m2\ntau=%.3gms\nEnergy=%.3gJ' %(fit[0][0],fit[0][1]*fit[0][2],fit[0][2]*1000,fit[0][1]*fit[0][2]*area_of_interest))
			ax[1,3].plot(1000*temp_time,semi_infinite_sink(temp_time,*fit[0]),'--m',linewidth=2)
			ax[1,2].set_ylim(bottom=fit[0][0]-2,top=10+semi_infinite_sink(1.5e-3,*fit[0]))
	except:
		print('fit of the pulse shape failed')

	ax[0,2].set_xlim(left=0,right=15)
	# ax[plot_index,0].set_ylim(bottom=ensamble_peak_shape.min(),top=1.5*ensamble_peak_shape.max())
	ax[0,2].legend(loc='best', fontsize='x-small')
	ax[0,2].grid()
	ax[0,2].set_xlabel('time [ms]')
	ax[0,2].set_ylabel('temperature [°C]')
	ax[0,2].set_title('small area max temp\narea=%.3gm2' %(area_of_interest))
	ax[1,2].set_xlim(left=0,right=15)
	# ax[plot_index,0].set_ylim(bottom=ensamble_peak_shape.min(),top=1.5*ensamble_peak_shape.max())
	ax[1,2].legend(loc='best', fontsize='x-small')
	ax[1,2].grid()
	ax[1,2].set_xlabel('time [ms]')
	ax[1,2].set_ylabel('temperature [°C]')
	# ax[plot_index,1].set_ylim(top=2*min_target_temperature.max())
	# ax[plot_index,1].set_yscale('log')
	ax[1,2].set_title('large area mean temp')

	ax[0,3].set_xlim(left=-2,right=15)
	ax[0,3].legend(loc='best', fontsize='x-small')
	ax[0,3].grid()
	ax[0,3].set_xlabel('time [ms]')
	ax[0,3].set_ylabel('temperature [°C]')
	ax[0,3].set_title('small area max full temp')
	ax[1,3].set_xlim(left=-2,right=15)
	ax[1,3].legend(loc='best', fontsize='x-small')
	ax[1,3].grid()
	ax[1,3].set_xlabel('time [ms]')
	ax[1,3].set_ylabel('temperature [°C]')
	ax[1,3].set_title('small area max full temp')

	figure_index+=1
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()




	plt.figure(figsize=(20, 10))
	plt.plot(np.arange(len(selected_1_mean))/frequency,selected_1_mean,label='mean, centre '+str([IR_shape[0],IR_shape[1]])+' r='+str(IR_shape[2]))
	# plt.plot(np.arange(len(selected_1_mean))/frequency,selected_2_mean,label='mean, centre '+str([IR_shape[0],IR_shape[1]])+' r='+str(IR_shape[3]))
	# plt.plot(np.arange(len(selected_1_mean))/frequency,selected_2_max,'+',label='max, centre '+str([IR_shape[0],IR_shape[1]])+' r='+str(IR_shape[3]))
	plt.plot([0,len(selected_1_mean)/frequency],[max_temp_before_peaks,max_temp_before_peaks],'--',label='max temp before ELM pulse (%.3g°C)' %(max_temp_before_peaks))
	plt.plot([0,len(selected_1_mean)/frequency],[max_temp_before_peaks+ELM_temp_jump,max_temp_before_peaks+ELM_temp_jump],'--',label='max temp before ELM pulse + median ELM jump (%.3g°C)' %(ELM_temp_jump))
	plt.plot([0,len(selected_1_mean)/frequency],[max_temp_before_peaks+ELM_temp_jump_after,max_temp_before_peaks+ELM_temp_jump_after],'--',label='max temp before ELM pulse + median ELM jump after pulse (%.3g°C)' %(ELM_temp_jump_after))
	plt.plot([0,len(selected_1_mean)/frequency],[max_temp_before_peaks+low_end_interval_ELM_jumps,max_temp_before_peaks+low_end_interval_ELM_jumps],'-.',label='max temp before ELM pulse + ELM jump after pulse 80% interval')
	plt.plot([0,len(selected_1_mean)/frequency],[max_temp_before_peaks+up_end_interval_ELM_jumps,max_temp_before_peaks+up_end_interval_ELM_jumps],'-.')
	plt.plot((np.arange(len(selected_1_mean))/frequency)[peaks],temp_before_peaks,'^',label='temp before pulse')
	plt.plot((np.arange(len(selected_1_mean))/frequency)[peaks],np.max([selected_1_mean[peaks-1],selected_1_mean[peaks],selected_1_mean[peaks-1]],axis=0),'v',label='temp at pulse')
	plt.plot((np.arange(len(selected_1_mean))/frequency)[peaks],selected_1_mean[peaks+round(1.5/1000*frequency)],'o',label='temp 1.5ms after pulse')
	plt.plot((np.arange(len(selected_1_mean))/frequency)[strongest_very_good_pulse],np.max([selected_1_mean[strongest_very_good_pulse-1],selected_1_mean[strongest_very_good_pulse],selected_1_mean[strongest_very_good_pulse-1]]),'y*',markersize=20,label='sample pulse for area check')
	# plt.plot([background_interval/frequency,background_interval/frequency],[np.min(selected_2_mean),np.max(selected_2_max)],'k--',label='limit for background in IR trace '+str(IR_reference))
	plt.grid()
	plt.title(pre_title+'Current trace for '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms' %(frequency,int_time*1000))
	plt.xlabel('time [s]')
	plt.ylabel('Temperature [°C]')
	plt.legend(loc='best')
	figure_index+=1
	# plt.show()
	plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
	plt.close()

	if isinstance(fname_current_trace,str):
		if os.path.exists(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/'+fname_current_trace+'.tsf'):

			if (len(bad_pulses_indexes)>2 and not bad_pulses_indexes[0]==-100):
				energy_per_pulse_good = []
				for i_energy_per_pulse_good in range(len(energy_per_pulse)):
					if not (i_energy_per_pulse_good+1 in bad_pulses_indexes):
						energy_per_pulse_good.append(energy_per_pulse[i_energy_per_pulse_good])

				select = np.array([not((value) in miss_pulses) for value in np.arange(1,number_of_pulses+1)])
				peaks = np.array(peaks_initial[peaks_initial>background_interval][np.array(score).argmax():][:number_of_pulses])
				peaks = peaks[select[:len(peaks)]]
				proms = get_proms(selected_2_max,peaks)[0]

				plt.figure(figsize=(20, 10))
				plt.plot(energy_per_pulse[select][:len(proms)], proms,'+b',label='prominence of the temperature peak')
				plt.plot([np.mean(energy_per_pulse_good)]*2,[np.min(proms),np.max(proms)],'k')
				plt.plot([np.mean(energy_per_pulse_good)+np.std(energy_per_pulse_good)]*2,[np.min(proms),np.max(proms)],'--k')
				plt.plot([np.mean(energy_per_pulse_good)-np.std(energy_per_pulse_good)]*2,[np.min(proms),np.max(proms)],'--k')
				y = np.array([y for _, y in sorted(zip(energy_per_pulse[select][:len(proms)], proms))])
				x = np.sort(energy_per_pulse[select][:len(proms)])
				if len(y)%10!=0:
					x = x[:(len(y)//10)*10]
					y = y[:(len(y)//10)*10]
				y = y.reshape((len(y)//10,10))
				x = x.reshape((len(x)//10,10))
				plt.plot(np.mean(x,axis=-1), np.mean(y,axis=-1),'--b')
				fit = scipy.stats.linregress(energy_per_pulse[select][:len(proms)],proms)
				plt.plot(np.sort(energy_per_pulse[select]), np.polyval(fit[:2],np.sort(energy_per_pulse[select])),':b',label='linear fit, coeffs=%.3g,%.3g, R2=%.3g' %(fit[0],fit[1],fit[2]**2))

				temp = np.max([selected_1_mean[np.array(peaks)-1],selected_1_mean[np.array(peaks)],selected_1_mean[np.array(peaks)+1]],axis=0)
				plt.plot(energy_per_pulse[select][:len(temp)], temp,'or',label='maximum hight of the temperature peak')
				plt.plot([np.mean(energy_per_pulse_good)]*2,[np.min(temp),np.max(temp)],'k')
				plt.plot([np.mean(energy_per_pulse_good)+np.std(energy_per_pulse_good)]*2,[np.min(temp),np.max(temp)],'--k')
				plt.plot([np.mean(energy_per_pulse_good)-np.std(energy_per_pulse_good)]*2,[np.min(temp),np.max(temp)],'--k')
				y = np.array([y for _, y in sorted(zip(energy_per_pulse[select][:len(temp)], temp))])
				x = np.sort(energy_per_pulse[select][:len(proms)])
				if len(y)%10!=0:
					x = x[:(len(y)//10)*10]
					y = y[:(len(y)//10)*10]
				y = y.reshape((len(y)//10,10))
				x = x.reshape((len(x)//10,10))
				plt.plot(np.mean(x,axis=-1), np.mean(y,axis=-1),'--r')
				fit = scipy.stats.linregress(energy_per_pulse[select][:len(temp)],temp)
				plt.plot(np.sort(energy_per_pulse[select]), np.polyval(fit[:2],np.sort(energy_per_pulse[select])),':r',label='linear fit, coeffs=%.3g,%.3g, R2=%.3g' %(fit[0],fit[1],fit[2]**2))

				if peaks[-1]+round(1.5/1000*frequency)>len(selected_1_mean):
					peaks = peaks[:-1]
					select[-1] = False
				temp = []
				for index in range(10):
					temp.append(selected_1_mean[peaks-index-5])
				temp = np.mean(temp,axis=0)
				temp = selected_1_mean[peaks+round(1.5/1000*frequency)] - temp
				plt.plot(energy_per_pulse[select][:len(peaks)], temp,'xg',label='temperature after 1.5ms of peak-before')
				plt.plot([np.mean(energy_per_pulse_good)]*2,[np.min(temp),np.max(temp)],'k')
				plt.plot([np.mean(energy_per_pulse_good)+np.std(energy_per_pulse_good)]*2,[np.min(temp),np.max(temp)],'--k')
				plt.plot([np.mean(energy_per_pulse_good)-np.std(energy_per_pulse_good)]*2,[np.min(temp),np.max(temp)],'--k')
				y = np.array([y for _, y in sorted(zip(energy_per_pulse[select][:len(temp)], temp))])
				x = np.sort(energy_per_pulse[select][:len(proms)])
				if len(y)%10!=0:
					x = x[:(len(y)//10)*10]
					y = y[:(len(y)//10)*10]
				y = y.reshape((len(y)//10,10))
				x = x.reshape((len(x)//10,10))
				plt.plot(np.mean(x,axis=-1), np.mean(y,axis=-1),'--g')
				fit = scipy.stats.linregress(energy_per_pulse[select][:len(temp)],temp)
				plt.plot(np.sort(energy_per_pulse[select]), np.polyval(fit[:2],np.sort(energy_per_pulse[select])),':g',label='linear fit, coeffs=%.3g,%.3g, R2=%.3g' %(fit[0],fit[1],fit[2]**2))

				plt.legend(loc='best')
				plt.grid()
				plt.title(pre_title+'Comparison of effect on the target for different power upstream (using small area mean) for '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms\n good pulses energy %.3g std %.3g' %(frequency,int_time*1000,np.mean(energy_per_pulse_good),np.std(energy_per_pulse_good)))
				plt.xlabel('Power from the source [J]')
				plt.ylabel('Temperature increase at target [°C]')
				figure_index+=1
				plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
				plt.close()

			peaks = np.array(peaks_initial[peaks_initial>background_interval][np.array(score).argmax():][:number_of_pulses])
			peaks = peaks[select[:len(peaks)]]
			proms = get_proms(selected_2_max,peaks)[0]

			how_much_to_shift = (np.abs(energy_per_pulse-np.max(energy_per_pulse[[np.logical_and(value not in np.array(bad_pulses_indexes)-1,value<28*1.5) for value in np.arange(len(energy_per_pulse))]])).argmin())%28-5
			# peaks_initial = find_peaks(selected_1_mean,distance=incremental_step*frequency*0.9)[0]
			peaks = np.array(peaks_initial[peaks_initial>background_interval][np.array(score).argmax():][:number_of_pulses])
			# proms = get_proms(selected_1_mean,peaks)[0]
			if False:	# this was only the peak temperature, modified to be temp after - temp before
				temp = np.max([selected_1_mean[np.array(peaks)-1],selected_1_mean[np.array(peaks)],selected_1_mean[np.array(peaks)+1]],axis=0)
			else:
				if peaks[-1]+round(1.5/1000*frequency)>len(selected_1_mean):
					peaks = peaks[:-1]
					select[-1] = False
				temp = []
				for index in range(10):
					temp.append(selected_1_mean[peaks-index-5])
				temp = np.mean(temp,axis=0)
				temp = selected_1_mean[peaks+round(1.5/1000*frequency)] - temp

			energy_per_pulse_shifted = energy_per_pulse[how_much_to_shift:].tolist() + energy_per_pulse[:how_much_to_shift].tolist()
			temp_shifted = temp[how_much_to_shift:].tolist() + temp[:how_much_to_shift].tolist()
			time_shifted = (np.arange(len(selected_1_mean))/frequency)[peaks][how_much_to_shift:].tolist() + (np.arange(len(selected_1_mean))/frequency)[peaks][:how_much_to_shift].tolist()
			bad_pulses_indexes_shifted =np.array(bad_pulses_indexes)
			group_being_analysed = 0
			while group_being_analysed<100:
				select = np.arange(len(energy_per_pulse_shifted))/29
				select = np.logical_and(select<(group_being_analysed+1),select>=(group_being_analysed))
				if np.sum(select)==0:
					break
				bad_pulses_in_the_interval = [value in bad_pulses_indexes_shifted for value in np.arange(29*group_being_analysed,28*(group_being_analysed+1))]
				if np.sum(bad_pulses_in_the_interval)%2==0:
					energy_per_pulse_shifted = energy_per_pulse_shifted[:(group_being_analysed+1)*29] + [0] + energy_per_pulse_shifted[(group_being_analysed+1)*29:]
					temp_shifted = temp_shifted[:(group_being_analysed+1)*29] + [0] + temp_shifted[(group_being_analysed+1)*29:]
					time_shifted = time_shifted[:(group_being_analysed+1)*29] + [0] + time_shifted[(group_being_analysed+1)*29:]
					bad_pulses_indexes_shifted[bad_pulses_indexes_shifted>((group_being_analysed+1)*28)]+=1
				group_being_analysed+=1

			plt.figure(figsize=(20, 10))
			cmap = plt.cm.rainbow
			norm=plt.Normalize(vmin=0, vmax=29-1)
			for index in range(29):
				plt.plot(energy_per_pulse_shifted[index::29][:min(len(energy_per_pulse_shifted[index::29]),len(time_shifted[index::29]))], temp_shifted[index::29][:min(len(energy_per_pulse_shifted[index::29]),len(time_shifted[index::29]))],'o',color=cmap(norm(index)))
			# plt.plot(energy_per_pulse,proms,'+')
			plt.grid()
			plt.title(pre_title+'Comparison of effect on the target for different power upstream\nsame point color should be the same capacitor (using small area mean)\nfor '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms\n good pulses energy %.3g std %.3g' %(frequency,int_time*1000,np.mean(energy_per_pulse_good),np.std(energy_per_pulse_good)))
			plt.xlabel('Pulse energy from the source [J]')
			plt.ylabel('Temperature increase at target [°C]')
			plt.ylim(bottom=np.min(np.array(temp_shifted)[np.array(temp_shifted)>0]))
			figure_index+=1
			plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
			plt.close()

			plt.figure(figsize=(20, 10))
			# peaks_initial = find_peaks(selected_1_mean,distance=incremental_step*frequency*0.9)[0]
			cmap = plt.cm.rainbow
			norm=plt.Normalize(vmin=0, vmax=29-1)
			for index in range(29):
				plt.plot(time_shifted[index::29][:min(len(energy_per_pulse_shifted[index::29]),len(time_shifted[index::29]))],energy_per_pulse_shifted[index::29][:min(len(energy_per_pulse_shifted[index::29]),len(time_shifted[index::29]))],'o',color=cmap(norm(index)))
			# plt.plot(energy_per_pulse,proms,'+')
			plt.grid()
			plt.plot([0,np.max(time_shifted)],[np.mean(energy_per_pulse_good)]*2,'--k')
			plt.plot([0,np.max(time_shifted)],[np.mean(energy_per_pulse_good)+np.std(energy_per_pulse_good)]*2,'--y')
			plt.plot([0,np.max(time_shifted)],[np.mean(energy_per_pulse_good)-np.std(energy_per_pulse_good)]*2,'--y')
			plt.title(pre_title+'Energy released at the source\nsame point color should be the same capacitor\nfor '+str(j)+', IR trace '+IR_trace+'\n frequency %.3gHz, int time %.3gms\n good pulses energy %.3g std %.3g' %(frequency,int_time*1000,np.mean(energy_per_pulse_good),np.std(energy_per_pulse_good)))
			plt.xlabel('time [s]')
			plt.ylabel('Pulse energy from the source [J]')
			figure_index+=1
			plt.savefig(path_where_to_save_everything +'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_'+str(figure_index)+'.eps', bbox_inches='tight')
			plt.close()



	df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
	df_log.loc[j,['DT_pulse','T_pre_pulse']] = (ELM_temp_jump_after,max_temp_before_peaks)
	df_log.loc[j,['DT_pulse_low','DT_pulse_high']] = (low_end_interval_ELM_jumps,up_end_interval_ELM_jumps)
	df_log.to_csv(path_or_buf='/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv')

	print('DONE analysing item n '+str(j))
	# return 'success '+str(j)


	# del spectra,magnitude,freq,corrected_frames,corrected_spectra,corrected_magnitude,selected_x,selected_y,selected_1,selected_2,peaks,proms,


# pool = Pool(number_cpu_available)
# composed_array = [*pool.map(calc_stuff, enumerate(np.flip(all_j,axis=0)))]
# pool.close()
# pool.join()
# print(composed_array)
