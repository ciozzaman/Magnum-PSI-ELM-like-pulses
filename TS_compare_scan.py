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
from scipy.signal import find_peaks

os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data/functions/ADAS_opacity')
from adas import read_adf15
from idlbridge import export_procedure

fdir = '/home/ffederic/work/Collaboratory/test/experimental_data'
df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
df_settings = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_3.csv',index_col=0)

TS_size=[-4.149230769230769056e+01,4.416923076923076508e+01]

compare_peak_Te = []
compare_peak_dTe = []
compare_peak_ne = []
compare_peak_dne = []
distance_all=[]
pressure_all=[]

to_scan = [99,98,96,97]
# to_scan = [95,89,88,87,86,85]
# to_scan = [91,90,85,92,93,94]
# to_scan = [70,69,68,67,66]
# for merge_ID_target in [91,90,85,92,93,94]:
# for merge_ID_target in [95,89,88,87,86,85]:
for merge_ID_target in to_scan:
# for merge_ID_target in range(96,100,1):
# for merge_ID_target in range(17,32,1):
	print('merge '+str(merge_ID_target))
	try:
		if not os.path.exists('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)):
			os.makedirs('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target))

		merge_Te_prof_multipulse = []
		merge_dTe_multipulse = []
		merge_ne_prof_multipulse = []
		merge_dne_multipulse = []
		merge_time = []
		merge_Te_prof_multipulse_SS = []
		merge_dTe_multipulse_SS = []
		merge_ne_prof_multipulse_SS = []
		merge_dne_multipulse_SS = []
		if True:
			all_j=find_index_of_file(merge_ID_target,df_settings,df_log)
			# plt.figure(merge_ID_target*100+1)
			# plt.figure(figsize=(20, 10))
			for ij,j in enumerate(all_j):
				(folder,date,sequence,untitled,TSTrig) = df_log.loc[j,['folder','date','sequence','untitled','TSTrig']]
				(distance,magnetic_field,pressure,pulse_voltage) = df_log.loc[j,['T_axial','B','p_n [Pa]','Vc']]


				print('j='+str(j))
				TSTrig = int(TSTrig)
				if np.isnan(TSTrig):
					continue
				try:
					Te_prof_multipulse = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/TS data/Te_prof_multipulse_'+folder+'_'+str(TSTrig)+'.npy')
					print('number of pulses ' + str(np.shape(Te_prof_multipulse)[0]))
					dTe_multipulse = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/TS data/dTe_multipulse_'+folder+'_'+str(TSTrig)+'.npy')
					ne_prof_multipulse = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/TS data/ne_prof_multipulse_'+folder+'_'+str(TSTrig)+'.npy')
					dne_multipulse = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/TS data/dne_multipulse_'+folder+'_'+str(TSTrig)+'.npy')
				except Exception as e:
					print('in merge ' + str(merge_ID_target) + ' TSTrig ' + str(TSTrig) + ' missing, exception '+e)
					continue

				# if merge_ID_target<=31:
				# 	Te_prof_multipulse = np.flip(Te_prof_multipulse,axis=0)
				# 	dTe_multipulse = np.flip(dTe_multipulse, axis=0)
				# 	ne_prof_multipulse = np.flip(ne_prof_multipulse, axis=0)
				# 	dne_multipulse = np.flip(dne_multipulse, axis=0)

				if max(Te_prof_multipulse[0])>0:
					merge_Te_prof_multipulse_SS.append(Te_prof_multipulse[0])
					merge_dTe_multipulse_SS.append(dTe_multipulse[0])
					merge_ne_prof_multipulse_SS.append(ne_prof_multipulse[0])
					merge_dne_multipulse_SS.append(dne_multipulse[0])

				(CB_to_OES_initial_delay,incremental_step,first_pulse_at_this_frame,bad_pulses_indexes,number_of_pulses) = df_log.loc[j,['CB_to_OES_initial_delay','incremental_step','first_pulse_at_this_frame','bad_pulses_indexes','number_of_pulses']]
				if bad_pulses_indexes=='':
					bad_pulses_indexes=[0]
				elif (isinstance(bad_pulses_indexes, float) or isinstance(bad_pulses_indexes, int)):
					bad_pulses_indexes=[bad_pulses_indexes]
				else:
					bad_pulses_indexes = bad_pulses_indexes.replace(' ', '').split(',')
					bad_pulses_indexes = list(map(int, bad_pulses_indexes))

				if merge_ID_target<66:
					bad_pulses_indexes = np.array(bad_pulses_indexes) +1

				# color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro','paleturquoise']
				# plt.plot(np.linspace(1,len(Te_prof_multipulse),len(Te_prof_multipulse)),np.mean(np.sort(Te_prof_multipulse,axis=1)[:,-5:],axis=1),'o'+'C'+str(ij),label='TSTrig '+str(TSTrig))
				# plt.plot(((np.array(bad_pulses_indexes) +1) * [[1], [1]])[0], (np.ones_like(bad_pulses_indexes) * [[np.max(np.sort(Te_prof_multipulse, axis=1)[:, -5:])],[np.min(np.sort(Te_prof_multipulse, axis=1)[:, -5:])]])[0],'--' + 'C'+str(ij),label='TSTrig '+str(TSTrig)+' bad pulses')
				# plt.plot(((np.array(bad_pulses_indexes) +1)*[[1],[1]]),(np.ones_like(bad_pulses_indexes)*[[np.max(np.sort(Te_prof_multipulse,axis=1)[:,-5:])],[np.min(np.sort(Te_prof_multipulse,axis=1)[:,-5:])]]),'--'+'C'+str(ij))
				# plt.title('Peak Te for merge ' + str(merge_ID_target)+' and bad pulses')
				# plt.legend(loc='best')
				# # plt.pause(0.001)
				# # plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/TS_bad_pulses_merge_' + str(merge_ID_target) + '.eps', bbox_inches='tight')
				# # plt.close()

				for index in range(len(Te_prof_multipulse)):
					# if (-CB_to_OES_initial_delay-incremental_step*index <-3):	#I want to exlude some unnecessary steady state		#13/08/19 .....why?
					# 	continue
					if (-CB_to_OES_initial_delay-incremental_step*(index-2+1) in merge_time):
						double_location = (np.array(merge_time)==-CB_to_OES_initial_delay-incremental_step*(index-2+1)).argmax()
						if np.max(merge_Te_prof_multipulse[double_location])>0:
							continue
						else:
							# if (index == 0 or ((index + 1) in bad_pulses_indexes)):
							if (index == 0 or ((index-1+1) in bad_pulses_indexes)):	# 13/08/19 VERY IMPORTANT:  the first pulse is the first data, that was over written with the steady state calculated by Han!
								# print('index '+str(index)+' is bad')
								continue
							merge_Te_prof_multipulse[double_location]=Te_prof_multipulse[index]
							merge_dTe_multipulse[double_location]=dTe_multipulse[index]
							merge_ne_prof_multipulse[double_location]=ne_prof_multipulse[index]
							merge_dne_multipulse[double_location]=dne_multipulse[index]
					else:
						# if (index==0 or ((index+1) in bad_pulses_indexes) ):
						if (index == 0 or ((index-1+1) in bad_pulses_indexes)):	# 13/08/19 VERY IMPORTANT:  the first pulse is the first data, that was over written with the steady state calculated by Han!
							merge_Te_prof_multipulse.append(np.zeros_like(Te_prof_multipulse[0]))
							merge_dTe_multipulse.append(np.zeros_like(Te_prof_multipulse[0]))
							merge_ne_prof_multipulse.append(np.zeros_like(Te_prof_multipulse[0]))
							merge_dne_multipulse.append(np.zeros_like(Te_prof_multipulse[0]))
							merge_time.append(-CB_to_OES_initial_delay-incremental_step*(index-2+1))
							# print('index '+str(index)+' is bad')
							continue
						merge_Te_prof_multipulse.append(Te_prof_multipulse[index])
						merge_dTe_multipulse.append(dTe_multipulse[index])
						merge_ne_prof_multipulse.append(ne_prof_multipulse[index])
						merge_dne_multipulse.append(dne_multipulse[index])
						merge_time.append(-CB_to_OES_initial_delay-incremental_step*(index-2+1))
				print(np.shape(merge_time))

			# plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/TS_bad_pulses_merge_' + str(merge_ID_target) + '.eps', bbox_inches='tight')
			# plt.close()

			# if len(merge_Te_prof_multipulse_SS)>0:
			# 	fig = plt.figure(figsize=(20, 10))
			# 	fig.add_subplot(1, 2, 1)
			# 	plt.errorbar(np.linspace(TS_size[0],TS_size[1],len(Te_prof_multipulse[0])),np.mean(merge_Te_prof_multipulse_SS,axis=0),yerr=np.max(merge_dTe_multipulse_SS,axis=0))
			# 	plt.title('Steady state Te for merge ' + str(merge_ID_target) + ' before the pulse')
			# 	plt.xlabel('radial location [mm]')
			# 	plt.ylabel('temperature [eV]')
			# 	plt.grid()
			# 	fig.add_subplot(1, 2, 2)
			# 	plt.errorbar(np.linspace(TS_size[0],TS_size[1],len(Te_prof_multipulse[0])),np.mean(merge_ne_prof_multipulse_SS,axis=0),yerr=np.max(merge_dne_multipulse_SS,axis=0))
			# 	plt.title('Steady state ne for merge ' + str(merge_ID_target) + ' before the pulse')
			# 	plt.xlabel('radial location [mm]')
			# 	plt.ylabel('electron density [10^20 #/m^3]')
			# 	plt.grid()
			# 	plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/TS_SS_before_merge_' + str(merge_ID_target) + '.eps', bbox_inches='tight')
			# 	plt.close()
			#
			# 	np.savez_compressed('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) +'/TS_SS_data_merge_'+str(merge_ID_target),merge_Te_prof_multipulse=np.mean(merge_Te_prof_multipulse_SS,axis=0),merge_dTe_multipulse=np.mean(merge_dTe_multipulse_SS,axis=0),merge_ne_prof_multipulse=np.mean(merge_ne_prof_multipulse_SS,axis=0),merge_dne_multipulse=np.mean(merge_dne_multipulse_SS,axis=0))


			merge_Te_prof_multipulse = np.array(merge_Te_prof_multipulse)
			merge_dTe_multipulse = np.array(merge_dTe_multipulse)
			merge_ne_prof_multipulse = np.array(merge_ne_prof_multipulse)
			merge_dne_multipulse = np.array(merge_dne_multipulse)
			merge_time = np.array(merge_time)

			merge_Te_prof_multipulse = np.array([merge_Te_prof_multipulse for _, merge_Te_prof_multipulse in sorted(zip(merge_time, merge_Te_prof_multipulse))])
			merge_dTe_multipulse = np.array([merge_dTe_multipulse for _, merge_dTe_multipulse in sorted(zip(merge_time, merge_dTe_multipulse))])
			merge_ne_prof_multipulse = np.array([merge_ne_prof_multipulse for _, merge_ne_prof_multipulse in sorted(zip(merge_time, merge_ne_prof_multipulse))])
			merge_dne_multipulse = np.array([merge_dne_multipulse for _, merge_dne_multipulse in sorted(zip(merge_time, merge_dne_multipulse))])
			merge_time = np.sort(merge_time)
			merge_time+=0.19		#internal time of thompson scattering


		# fig=plt.figure(figsize=(10, 20))
		# # fig=plt.figure(merge_ID_target * 100 + 2)
		# columns = 4
		# rows = 5
		# fig.add_subplot(1, 2, 1)
		# plt.imshow(merge_Te_prof_multipulse, 'rainbow',extent=[TS_size[0],TS_size[1],max(merge_time),min(merge_time)],aspect=50)
		# plt.title('Te for merge ' + str(merge_ID_target))
		# plt.xlabel('radial location [mm]')
		# plt.ylabel('time [ms]')
		# plt.colorbar().set_label('[eV]')
		# fig.add_subplot(1, 2, 2)
		# plt.imshow(merge_ne_prof_multipulse, 'rainbow',extent=[TS_size[0],TS_size[1],max(merge_time),min(merge_time)],aspect=50)
		# plt.title('ne for merge ' + str(merge_ID_target))
		# plt.xlabel('radial location [mm]')
		# plt.ylabel('time [ms]')
		# plt.colorbar().set_label('[10^20 #/m^3]')
		# # plt.pause(0.001)
		# plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/TS_values_merge_' + str(merge_ID_target) + '.eps',bbox_inches='tight')
		# plt.close()

		index_t0 = (np.abs(merge_time)).argmin()
		index_t2 = (np.abs(merge_time-2)).argmin()
		# fig=plt.figure(figsize=(10, 20))
		# # fig=plt.figure(merge_ID_target * 100 + 2)
		# columns = 4
		# rows = 5
		# fig.add_subplot(1, 2, 1)
		# plt.imshow(merge_Te_prof_multipulse[index_t0:index_t2], 'rainbow',extent=[TS_size[0],TS_size[1],2,0],aspect=100)
		# plt.title('Te for merge ' + str(merge_ID_target))
		# plt.xlabel('radial location [mm]')
		# plt.ylabel('time [ms]')
		# plt.colorbar().set_label('[eV]')
		# fig.add_subplot(1, 2, 2)
		# plt.imshow(merge_ne_prof_multipulse[index_t0:index_t2], 'rainbow',extent=[TS_size[0],TS_size[1],2,0],aspect=100)
		# plt.title('ne for merge ' + str(merge_ID_target))
		# plt.xlabel('radial location [mm]')
		# plt.ylabel('time [ms]')
		# plt.colorbar().set_label('[10^20 #/m^3]')
		# # plt.pause(0.001)
		# plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/TS_values_short_merge_' + str(merge_ID_target) + '.eps',bbox_inches='tight')
		# plt.close()
		#
		# fig=plt.figure(figsize=(10, 20))
		# # fig=plt.figure(merge_ID_target * 100 + 2)
		# columns = 4
		# rows = 5
		# fig.add_subplot(1, 2, 1)
		# temp_print = cp.deepcopy(merge_Te_prof_multipulse)
		# temp_print[merge_Te_prof_multipulse>1]=0
		# plt.imshow(temp_print, 'rainbow',extent=[TS_size[0],TS_size[1],max(merge_time),min(merge_time)],aspect=50)
		# plt.title('Te for merge ' + str(merge_ID_target))
		# plt.xlabel('radial location [mm]')
		# plt.ylabel('time [ms]')
		# plt.colorbar().set_label('[eV]')
		# fig.add_subplot(1, 2, 2)
		# temp_print = cp.deepcopy(merge_ne_prof_multipulse)
		# temp_print[merge_Te_prof_multipulse>1]=0
		# plt.imshow(temp_print, 'rainbow',extent=[TS_size[0],TS_size[1],max(merge_time),min(merge_time)],aspect=50)
		# plt.title('ne for merge ' + str(merge_ID_target))
		# plt.xlabel('radial location [mm]')
		# plt.ylabel('time [ms]')
		# plt.colorbar().set_label('[10^20 #/m^3]')
		# # plt.pause(0.001)
		# plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/TS_values_merge_' + str(merge_ID_target) + '_Tmax1eV.eps',bbox_inches='tight')
		# plt.close()
		#
		# fig=plt.figure(figsize=(10, 20))
		# # fig=plt.figure(merge_ID_target * 100 + 2)
		# columns = 4
		# rows = 5
		# fig.add_subplot(1, 2, 1)
		# temp_print = cp.deepcopy(merge_Te_prof_multipulse)
		# temp_print[merge_Te_prof_multipulse<1]=1
		# plt.imshow(temp_print, 'rainbow',extent=[TS_size[0],TS_size[1],max(merge_time),min(merge_time)],aspect=50)
		# plt.title('Te for merge ' + str(merge_ID_target))
		# plt.xlabel('radial location [mm]')
		# plt.ylabel('time [ms]')
		# plt.colorbar().set_label('[eV]')
		# fig.add_subplot(1, 2, 2)
		# temp_print = cp.deepcopy(merge_ne_prof_multipulse)
		# temp_print[merge_Te_prof_multipulse<1]=np.nanmax(temp_print)
		# plt.imshow(temp_print, 'rainbow',extent=[TS_size[0],TS_size[1],max(merge_time),min(merge_time)],aspect=50)
		# plt.title('ne for merge ' + str(merge_ID_target))
		# plt.xlabel('radial location [mm]')
		# plt.ylabel('time [ms]')
		# plt.colorbar().set_label('[10^20 #/m^3]')
		# # plt.pause(0.001)
		# plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/TS_values_merge_' + str(merge_ID_target) + '_Tmin1eV.eps',bbox_inches='tight')
		# plt.close()
		#
		# fig=plt.figure(figsize=(10, 20))
		# # fig=plt.figure(merge_ID_target * 100 + 2)
		# columns = 4
		# rows = 5
		# fig.add_subplot(1, 2, 1)
		# temp_print = cp.deepcopy(merge_Te_prof_multipulse)
		# temp_print[merge_Te_prof_multipulse<0.5]=0
		# temp_print[merge_Te_prof_multipulse >1.2] = 0
		# plt.imshow(temp_print, 'rainbow',extent=[TS_size[0],TS_size[1],max(merge_time),min(merge_time)],aspect=50)
		# plt.title('Te for merge ' + str(merge_ID_target))
		# plt.xlabel('radial location [mm]')
		# plt.ylabel('time [ms]')
		# plt.colorbar().set_label('[eV]')
		# fig.add_subplot(1, 2, 2)
		# temp_print = cp.deepcopy(merge_ne_prof_multipulse)
		# temp_print[merge_Te_prof_multipulse<0.5]=0
		# temp_print[merge_Te_prof_multipulse>1.2]=0
		# plt.imshow(temp_print, 'rainbow',extent=[TS_size[0],TS_size[1],max(merge_time),min(merge_time)],aspect=50)
		# plt.title('ne for merge ' + str(merge_ID_target))
		# plt.xlabel('radial location [mm]')
		# plt.ylabel('time [ms]')
		# plt.colorbar().set_label('[10^20 #/m^3]')
		# # plt.pause(0.001)
		# plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/TS_values_merge_' + str(merge_ID_target) + '_MAR.eps',bbox_inches='tight')
		# plt.close()
		#
		# fig=plt.figure(figsize=(10, 20))
		# # fig=plt.figure(merge_ID_target * 100 + 3)
		# columns = 4
		# rows = 5
		# fig.add_subplot(1, 2, 1)
		# plt.imshow(merge_dTe_multipulse, 'rainbow',extent=[TS_size[0],TS_size[1],max(merge_time),min(merge_time)],aspect=50)
		# plt.title('dTe for merge ' + str(merge_ID_target))
		# plt.xlabel('radial location [mm]')
		# plt.ylabel('time [ms]')
		# plt.colorbar().set_label('[eV]')
		# fig.add_subplot(1, 2, 2)
		# plt.imshow(merge_dne_multipulse, 'rainbow',extent=[TS_size[0],TS_size[1],max(merge_time),min(merge_time)],aspect=50)
		# plt.title('dne for merge ' + str(merge_ID_target))
		# plt.xlabel('radial location [mm]')
		# plt.ylabel('time [ms]')
		# plt.colorbar().set_label('[10^20 #/m^3]')
		# # plt.pause(0.001)
		# plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/TS_uncertanties_merge_' + str(merge_ID_target) + '.eps',bbox_inches='tight')
		# plt.close()

		peak_temp_marker = np.mean(np.sort(merge_Te_prof_multipulse,axis=-1)[:,-3:],axis=-1)
		peak_dens_marker = np.mean(np.sort(merge_ne_prof_multipulse,axis=-1)[:,-3:],axis=-1)
		peaks_Te=[]
		for index in range(np.shape(merge_Te_prof_multipulse)[-1]):
			peaks_Te.append( np.array([x for _, x in sorted(zip(peak_temp_marker, merge_Te_prof_multipulse[:,index]))]))
		peaks_Te = np.mean(np.array(peaks_Te).T[-3:],axis=0)
		peaks_dTe=[]
		for index in range(np.shape(merge_dTe_multipulse)[-1]):
			peaks_dTe.append( np.array([x for _, x in sorted(zip(peak_temp_marker, merge_dTe_multipulse[:,index]))]))
		peaks_dTe = np.max(np.array(peaks_dTe).T[-3:],axis=0)
		peaks_ne=[]
		for index in range(np.shape(merge_ne_prof_multipulse)[-1]):
			peaks_ne.append( np.array([x for _, x in sorted(zip(peak_temp_marker, merge_ne_prof_multipulse[:,index]))]))
		peaks_ne = np.mean(np.array(peaks_ne).T[-3:],axis=0)
		peaks_dne=[]
		for index in range(np.shape(merge_dne_multipulse)[-1]):
			peaks_dne.append( np.array([x for _, x in sorted(zip(peak_temp_marker, merge_dne_multipulse[:,index]))]))
		peaks_dne = np.max(np.array(peaks_dne).T[-3:],axis=0)


		initial_time = (np.abs(merge_time+0.25)).argmin()
		final_time = (np.abs(merge_time-1.5)).argmin()
		compare_peak_Te.append(peak_temp_marker[initial_time:final_time])
		compare_peak_ne.append(peak_dens_marker[initial_time:final_time])
		compare_peak_dTe.append(np.max(merge_dTe_multipulse,axis=1)[initial_time:final_time])
		compare_peak_dne.append(np.max(merge_dne_multipulse,axis=1)[initial_time:final_time])


		distance_all.append(distance)
		pressure_all.append(pressure)

	except Exception as e:
		print('merge '+str(merge_ID_target)+' FAILED!, exception '+e)

merge_time=merge_time[initial_time:final_time]



fig=plt.figure(figsize=(10, 10))
fig.suptitle('magnetic field %.3gT, pulse voltage %.3gV, OES/TS - target distance %.3gmm' %(magnetic_field,pulse_voltage,distance))
fig.add_subplot(2, 1, 1)
for index,peak_temp_marker in enumerate(compare_peak_Te):
	temp1=[]
	temp2=[]
	temp3=[]
	for i in range(len(peak_temp_marker)):
		if (peak_temp_marker[i]>0 and np.isfinite(peak_temp_marker[i])):
			temp1.append(peak_temp_marker[i])
			temp2.append(compare_peak_dTe[index][i])
			temp3.append(merge_time[i])
	# plt.errorbar(merge_time,peak_temp_marker,yerr=compare_peak_dTe,label=str(index))
	plt.errorbar(temp3,temp1,yerr=temp2,label='OES-target %.3gmm, press %.3gPa' %(distance_all[index],pressure_all[index]))
plt.title('Peak Te in '+str(to_scan))
plt.ylabel('temperature [eV]')
plt.xlabel('time [ms]')
plt.legend(loc='best')
fig.add_subplot(2, 1, 2)
for index,peak_dens_marker in enumerate(compare_peak_ne):
	temp1=[]
	temp2=[]
	temp3=[]
	for i in range(len(peak_temp_marker)):
		if (peak_dens_marker[i]>0 and np.isfinite(peak_dens_marker[i])):
			temp1.append(peak_dens_marker[i])
			temp2.append(compare_peak_dne[index][i])
			temp3.append(merge_time[i])
	# plt.errorbar(merge_time,peak_dens_marker,fmt='none',marker='+',yerr=compare_peak_dne[index],label=str(index))
	plt.errorbar(temp3,temp1,yerr=temp2,label='OES-target %.3gmm, press %.3gPa' %(distance_all[index],pressure_all[index]))
plt.title('Peak ne in '+str(to_scan))
plt.ylabel('density [10^20 #/m^3]')
plt.xlabel('time [ms]')
plt.legend(loc='best')
plt.pause(0.001)
# plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/TS_peak_evolution_merge_' + str(merge_ID_target) + '.eps',bbox_inches='tight')
# plt.close()

# plt.figure()
# plt.plot([1.2,6.2,21.2,36.2,56.2],np.max(compare_peak_Te,axis=1))
# plt.pause(0.01)
#
# plt.figure()
# plt.plot([1.2,6.2,21.2,36.2,56.2],np.max(compare_peak_ne,axis=1))
# plt.pause(0.01)

'''
		# fig = plt.figure(figsize=(20, 10))
		# fig.add_subplot(1, 2, 1)
		# plt.errorbar(np.linspace(TS_size[0],TS_size[1], len(peaks_dTe)), peaks_Te,yerr=peaks_dTe)
		# plt.title('Peak Te for merge ' + str(merge_ID_target)+' time '+str(merge_time[peak_temp_marker.argmax()])+'ms')
		# plt.xlabel('radial location [mm]')
		# plt.ylabel('temperature [eV]')
		# fig.add_subplot(1, 2, 2)
		# plt.errorbar(np.linspace(TS_size[0],TS_size[1], len(peaks_dne)), peaks_ne,yerr=peaks_dne)
		# plt.title('ne at peak Te for merge ' + str(merge_ID_target))
		# plt.xlabel('radial location [mm]')
		# plt.ylabel('electron density [10^20 #/m^3]')
		# plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/TS_peak_merge_' + str(merge_ID_target) + '.eps', bbox_inches='tight')
		# plt.close()

		# fig=plt.figure(figsize=(20, 10))
		# initial_time = (np.abs(merge_time+2)).argmin()
		# columns = 4
		# rows = 5
		# fig.add_subplot(2, 1, 1)
		# where_temp_peak = np.max(merge_Te_prof_multipulse,axis=0)
		# plt.plot(merge_time[initial_time:],peak_temp_marker[initial_time:])
		# plt.title('Peak Te for merge ' + str(merge_ID_target))
		# plt.ylabel('[eV]')
		# plt.xlabel('time [ms]')
		# fig.add_subplot(2, 1, 2)
		# plt.plot(merge_time[initial_time:],peak_dens_marker[initial_time:])
		# plt.title('Peak ne for merge ' + str(merge_ID_target))
		# plt.ylabel('10^20 #/m^3')
		# plt.xlabel('time [ms]')
		# # plt.pause(0.001)
		# plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/TS_peak_evolution_merge_' + str(merge_ID_target) + '.eps',bbox_inches='tight')
		# plt.close()

		initial_time = (np.abs(merge_time+2)).argmin()
		where_temp_peak = np.max(merge_Te_prof_multipulse,axis=0)


		if False:	# this bit is to try to estimate opacity, but presently does not work
			radious=1	#cm
			aspect_ratio=100	#length over radious
			ev_to_K = 11600
			average_Te = cp.deepcopy(merge_Te_prof_multipulse)
			average_Te[average_Te==0] = np.nan
			average_Te = np.nanmean(average_Te,axis=1)[index_t0:index_t2]
			average_Te[np.isnan(average_Te)]=0
			average_ne = cp.deepcopy(merge_ne_prof_multipulse)
			average_ne[average_ne==0] = np.nan
			average_ne = np.nanmean(average_ne,axis=1)[index_t0:index_t2]
			average_ne[np.isnan(average_ne)]=0
			pecfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/pec12#h_balmer#h0.dat'
			file='./pec.pass'
			excitation_rates_opacity4,excitation_rates_opacity5,recombination_rates_opacity4,recombination_rates_opacity5 = [],[],[],[]
			excitation_rates_no_opacity4,excitation_rates_no_opacity5,recombination_rates_no_opacity4,recombination_rates_no_opacity5 = [],[],[],[]
			for index in range(len(average_Te)):
				# if index==5:
				Te_value,ne_value,Te_peak_value,ne_peak_value = average_Te[index],average_ne[index],peak_temp_marker[index_t0:index_t2][index],peak_dens_marker[index_t0:index_t2][index]
				get_opacity_adf15 = export_procedure("use_adas214_208") # te, ne, l, aspect_ratio
				# get_opacity_adf15(Te_value,ne_value*10 ** (20 - 6),ne_value*10 ** (20 - 6), radious, aspect_ratio, 'Cylinder','Linear')
				# get_opacity_adf15(Te_peak_value,ne_peak_value*10 ** (20 - 6),np.logspace(np.log10(ne_peak_value*10 ** (20 - 6)),np.log10(0.02*ne_peak_value*10 ** (20 - 6)),5), radious, aspect_ratio, 'Cylinder','Linear')
				get_opacity_adf15(Te_peak_value,Te_value,ne_peak_value*10 ** (20 - 6),ne_peak_value*10 ** (20 - 6), radious, aspect_ratio, 'Cylinder','Linear')
				block=5
				recombination_rates_opacity4.append(read_adf15(file, block, Te_peak_value, ne_peak_value* 10 ** (20 - 6))[0])
				block=4
				recombination_rates_opacity5.append(read_adf15(file, block, Te_peak_value, ne_peak_value* 10 ** (20 - 6))[0])
				isel=20
				recombination_rates_no_opacity4.append(read_adf15(pecfile, isel, Te_peak_value, ne_peak_value* 10 ** (20 - 6))[0])  # ADAS database is in cm^3   # photons s^-1 cm^-3
				isel=21
				recombination_rates_no_opacity5.append(read_adf15(pecfile, isel, Te_peak_value, ne_peak_value* 10 ** (20 - 6))[0])  # ADAS database is in cm^3   # photons s^-1 cm^-3
				block=2
				excitation_rates_opacity4.append(read_adf15(file, block, Te_peak_value, ne_peak_value* 10 ** (20 - 6))[0])
				block=1
				excitation_rates_opacity5.append(read_adf15(file, block, Te_peak_value, ne_peak_value* 10 ** (20 - 6))[0])
				isel=2
				excitation_rates_no_opacity4.append(read_adf15(pecfile, isel, Te_peak_value, ne_peak_value* 10 ** (20 - 6))[0])  # ADAS database is in cm^3   # photons s^-1 cm^-3
				isel=3
				excitation_rates_no_opacity5.append(read_adf15(pecfile, isel, Te_peak_value, ne_peak_value* 10 ** (20 - 6))[0])  # ADAS database is in cm^3   # photons s^-1 cm^-3

			excitation_rates_opacity4 = np.array(excitation_rates_opacity4)
			excitation_rates_opacity5 = np.array(excitation_rates_opacity5)
			recombination_rates_opacity4 = np.array(recombination_rates_opacity4)
			recombination_rates_opacity5 = np.array(recombination_rates_opacity5)
			excitation_rates_no_opacity4 = np.array(excitation_rates_no_opacity4)
			excitation_rates_no_opacity5 = np.array(excitation_rates_no_opacity5)
			recombination_rates_no_opacity4 = np.array(recombination_rates_no_opacity4)
			recombination_rates_no_opacity5 = np.array(recombination_rates_no_opacity5)


			excitation_rates_opacity4[np.isnan(excitation_rates_opacity4)] = 0
			excitation_rates_opacity5[np.isnan(excitation_rates_opacity5)] = 0
			recombination_rates_opacity4[np.isnan(recombination_rates_opacity4)] = 0
			recombination_rates_opacity5[np.isnan(recombination_rates_opacity5)] = 0
			excitation_rates_no_opacity4[np.isnan(excitation_rates_no_opacity4)] = 0
			excitation_rates_no_opacity5[np.isnan(excitation_rates_no_opacity5)] = 0
			recombination_rates_no_opacity4[np.isnan(recombination_rates_no_opacity4)] = 0
			recombination_rates_no_opacity5[np.isnan(recombination_rates_no_opacity5)] = 0
			# excitation_rates_opacity4[peak_temp_marker<0.2] = 0
			# excitation_rates_opacity5[peak_temp_marker<0.2] = 0
			# excitation_rates_no_opacity4[peak_temp_marker<0.2] = 0
			# excitation_rates_no_opacity5[peak_temp_marker<0.2] = 0



			fig=plt.figure(figsize=(20, 10))
			fig.add_subplot(2, 1, 1)
			plt.plot(merge_time[index_t0:index_t2],recombination_rates_no_opacity4,label='4>2 without opacity')
			plt.plot(merge_time[index_t0:index_t2],recombination_rates_opacity4,label='4>2 with opacity')
			plt.plot(merge_time[index_t0:index_t2],recombination_rates_no_opacity5,label='5>2 without opacity')
			plt.plot(merge_time[index_t0:index_t2],recombination_rates_opacity5,label='5>2 with opacity')
			plt.legend(loc='best')
			plt.grid()
			plt.semilogy()
			# plt.pause(0.01)
			plt.title('Recombination PEC from ADAS 214 ' + str(merge_ID_target))
			plt.ylabel('[# s^-1 cm^-3 / (cm^-3)^2]')
			plt.xlabel('time [ms]')
			fig.add_subplot(2, 1, 2)
			plt.plot(merge_time[index_t0:index_t2],10*peak_temp_marker[index_t0:index_t2],label='Peak Te [0.1*eV]')
			plt.plot(merge_time[index_t0:index_t2],10*average_Te,'--',label='Average Te [0.1*eV]')
			plt.plot(merge_time[index_t0:index_t2],peak_dens_marker[index_t0:index_t2],label='Peak ne [# 10^20 m^-3]')
			plt.plot(merge_time[index_t0:index_t2],average_ne,'--',label='Average ne [# 10^20 m^-3]')
			plt.title('Te and ne for merge ' + str(merge_ID_target))
			plt.xlabel('time [ms]')
			plt.legend(loc='best')
			plt.grid()
			# plt.pause(0.001)
			plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/TS_opacity_effect_recombination_merge_' + str(merge_ID_target) + '.eps',bbox_inches='tight')
			plt.close()

			fig=plt.figure(figsize=(20, 10))
			fig.add_subplot(2, 1, 1)
			plt.plot(merge_time[index_t0:index_t2],excitation_rates_no_opacity4,label='4>2 without opacity')
			plt.plot(merge_time[index_t0:index_t2],excitation_rates_opacity4,label='4>2 with opacity')
			plt.plot(merge_time[index_t0:index_t2],excitation_rates_no_opacity5,label='5>2 without opacity')
			plt.plot(merge_time[index_t0:index_t2],excitation_rates_opacity5,label='5>2 with opacity')
			plt.ylim(top=2*np.max([*excitation_rates_no_opacity4,*excitation_rates_opacity4,*excitation_rates_no_opacity5,*excitation_rates_opacity5]),bottom=1e-25)
			plt.legend(loc='best')
			plt.grid()
			plt.semilogy()
			# plt.pause(0.01)
			plt.title('Excitation PEC from ADAS 214, merge ' + str(merge_ID_target))
			plt.ylabel('[# s^-1 cm^-3 / (cm^-3)^2]')
			plt.xlabel('time [ms]')
			fig.add_subplot(2, 1, 2)
			plt.plot(merge_time[index_t0:index_t2],10*peak_temp_marker[index_t0:index_t2],label='Peak Te [0.1*eV]')
			plt.plot(merge_time[index_t0:index_t2],10*average_Te,'--',label='Average Te [0.1*eV]')
			plt.plot(merge_time[index_t0:index_t2],peak_dens_marker[index_t0:index_t2],label='Peak ne [# 10^20 m^-3]')
			plt.plot(merge_time[index_t0:index_t2],average_ne,'--',label='Average ne [# 10^20 m^-3]')
			plt.title('Te and ne for merge ' + str(merge_ID_target))
			plt.xlabel('time [ms]')
			plt.legend(loc='best')
			plt.grid()
			# plt.pause(0.001)
			plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) + '/TS_opacity_effect_excitation_merge_' + str(merge_ID_target) + '.eps',bbox_inches='tight')
			plt.close()


		# np.savez_compressed('/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) +'/TS_data_merge_'+str(merge_ID_target),merge_Te_prof_multipulse=merge_Te_prof_multipulse,merge_dTe_multipulse=merge_dTe_multipulse,merge_ne_prof_multipulse=merge_ne_prof_multipulse,merge_dne_multipulse=merge_dne_multipulse,merge_time=merge_time)

	except Exception as e:
		print('merge '+str(merge_ID_target)+' FAILED!, exception '+e)
'''
