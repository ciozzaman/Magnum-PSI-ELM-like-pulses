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

from pycine.color import color_pipeline, resize
from pycine.raw import read_frames
from pycine.file import read_header
from functions.fabio_add import all_file_names,find_index_of_file

fdir = '/home/ffederic/work/Collaboratory/test/experimental_data'
df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
df_settings = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_3.csv',index_col=0)


# median_50_all = []
# median_100_all = []
# median_150_all = []
# median_200_all = []
# median_250_all = []
# plt.figure()
# # for file_index in [46,45,43,44]:
# 	# folder = '/home/ffederic/work/Collaboratory/test/experimental_data/2019-07-04/01/fast_camera/'
# # for file_index in [18,17,16,15,19,20]:
# for j in [260,258,255,253,262,264]:
# 	(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
# 	(file_index,target_OES_distance,magnetic_field,SS_pressure,pulse_voltage) = df_log.loc[j,['fast_camera_trace','T_axial','B','p_n [Pa]','Vc']]
#
# 	full_folder = '/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/'+"{0:0=2d}".format(sequence)+'/fast_camera/'
# 	type = '.cine'
# 	filenames = np.array(all_file_names(full_folder, type))
# 	# select = [name[:2]==str(file_index) for name in filenames]
# 	select = [name[:2]=="{0:0=2d}".format(int(file_index)) for name in filenames]
# 	filenames = filenames[select]
#
# 	# median_50 = []
# 	# median_100 = []
# 	# median_150 = []
# 	median_50 = np.zeros((len(filenames),2*300))
# 	median_100 = np.zeros((len(filenames),2*300))
# 	median_150 = np.zeros((len(filenames),2*300))
# 	median_200 = np.zeros((len(filenames),2*300))
# 	median_250 = np.zeros((len(filenames),2*300))
#
# 	# plt.figure()
# 	for i_filename,filename in enumerate(filenames):
# 		raw_images, setup, bpp = read_frames(full_folder+filename)#,start_frame=1,count=1)
# 		# header = read_header('/home/ffederic/work/Collaboratory/test/experimental_data/2019-05-03/01/fast_camera/02_1.cine')
#
# 		raw_data = []
# 		for image in raw_images:
# 			raw_data.append(image)
# 		raw_data = np.array(raw_data).astype('int')
#
# 		cleaned = medfilt(raw_data-raw_data[np.sum(raw_data,axis=(-1,-2)).argmin()],[1,3,3])
#
# 		maximum = np.max(np.sum(cleaned[:,:,50],axis=1))
# 		minimum = np.min(np.sum(cleaned[:,:,50],axis=1))
# 		# select = (np.sum(cleaned[:,:,50],axis=1)>(maximum-minimum)/10).argmax()
# 		select = np.sum(cleaned[:,:,50],axis=1).argmax()
#
# 		# plt.figure()
# 		# plt.imshow(cleaned[select],'rainbow')
# 		# plt.colorbar()
# 		# plt.plot([50,50],[0,256],'--k')
# 		# plt.plot([100,100],[0,256],'--k')
# 		# plt.plot([150,150],[0,256],'--k')
# 		# plt.title(full_folder+filename)
# 		# plt.pause(0.01)
#
# 		# if i_filename==0:
# 		# 		median_50 = np.zeros((len(filenames),2*len(cleaned)))
# 		# 		median_100 = np.zeros((len(filenames),2*len(cleaned)))
# 		# 		median_150 = np.zeros((len(filenames),2*len(cleaned)))
# 		# 		median_200 = np.zeros((len(filenames),2*len(cleaned)))
# 		# 		median_250 = np.zeros((len(filenames),2*len(cleaned)))
#
# 		median_50[i_filename,-select+300:len(cleaned)-select+300] = np.mean(cleaned[:,60:200,50],axis=1)
# 		median_100[i_filename,-select+300:len(cleaned)-select+300] = np.mean(cleaned[:,60:200,100],axis=1)
# 		median_150[i_filename,-select+300:len(cleaned)-select+300] = np.mean(cleaned[:,60:200,150],axis=1)
# 		median_200[i_filename,-select+300:len(cleaned)-select+300] = np.mean(cleaned[:,60:200,200],axis=1)
# 		median_250[i_filename,-select+300:len(cleaned)-select+300] = np.mean(cleaned[:,60:200,250],axis=1)
# 		# median_100.append(np.sum(cleaned[select:,:,100],axis=1))
# 		# median_150.append(np.sum(cleaned[select:,:,150],axis=1))
#
# 	# 	plt.plot(np.arange(-select,len(cleaned)-select),np.sum(cleaned[:,:,50],axis=1),'--',label=filename+'__50')
# 	# 	plt.plot(np.arange(-select,len(cleaned)-select),np.sum(cleaned[:,:,100],axis=1),':',label=filename+'__100')
# 	# 	plt.plot(np.arange(-select,len(cleaned)-select),np.sum(cleaned[:,:,200],axis=1),label=filename+'__150')
# 	# plt.legend(loc='best')
# 	# plt.title(folder+filename)
# 	# plt.grid()
# 	# plt.pause(0.01)
#
# 	plt.plot(np.arange(-300,300),np.median(median_50,axis=0),'--',label='target/OES distance %.3gmm' %(target_OES_distance)+', h pixel 50')
# 	plt.plot(np.arange(-300,300),np.median(median_100,axis=0),':',label='target/OES distance %.3gmm' %(target_OES_distance)+', h pixel 100')
# 	plt.plot(np.arange(-300,300),np.median(median_200,axis=0),label='target/OES distance %.3gmm' %(target_OES_distance)+', h pixel 150')
# 	plt.legend(loc='best')
# 	# plt.title(full_folder+filename)
# 	plt.grid()
# 	plt.pause(0.01)
#
# 	median_50_all.append(np.median(median_50,axis=0))
# 	median_100_all.append(np.median(median_100,axis=0))
# 	median_150_all.append(np.median(median_150,axis=0))
# 	median_200_all.append(np.median(median_200,axis=0))
# 	median_250_all.append(np.median(median_250,axis=0))
#
# plt.legend(loc='best')
# plt.title('Target/OES distance scan magnetic_field %.3gT,steady state pressure %.3gPa,ELM pulse voltage %.3gV' %(magnetic_field,SS_pressure,pulse_voltage))
# plt.grid()
# plt.pause(0.01)




plt.close('all')
figure_index = 1







median_50_all = []
median_100_all = []
median_OES_all = []
median_target_all = []
median_250_all = []
SS_pressure_all = []
averaged_profile_all = []
target_OES_distance_all = []
plasma_diameter_50 = []
plasma_diameter_100 = []
plasma_diameter_OES = []
plasma_diameter_target = []
figure_index += 2
plt.figure(figure_index,figsize=(12,6))
plt.figure(figure_index+1,figsize=(12,6))
plt.figure(figure_index+2,figsize=(12,6))
plt.figure(figure_index+3,figsize=(12,6))
# for file_index in [46,45,43,44]:
	# folder = '/home/ffederic/work/Collaboratory/test/experimental_data/2019-07-04/01/fast_camera/'
# for file_index in [18,17,16,15,19,20]:
# for j in [260,258,255,253,262,264]:
counter=0
# for merge_ID_target in [30,27,28,29]:	# distance scan
# for merge_ID_target in [22,23,17,24,25,26]:	# distance scan
for merge_ID_target in [99,98,96,97]:	# pressure scan
# for merge_ID_target in [70,69,68,67,66]:	# pressure scan
# for merge_ID_target in [95,89,88,87,86,85]:	# pressure scan
# for merge_ID_target in [72,71,66,73]:	# distance scan
# for merge_ID_target in [82,81,80,79,83,84]:	# distance scan
# for merge_ID_target in [91,90,85,92,93]:	# distance scan
	all_j=find_index_of_file(merge_ID_target,df_settings,df_log,only_OES=True)
	averaged_profile = []

	median_50 = []
	# plt.figure()
	for j in all_j:
		(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
		sequence = int(sequence)
		(file_index,target_OES_distance,magnetic_field,SS_pressure,pulse_voltage) = df_log.loc[j,['fast_camera_trace','T_axial','B','p_n [Pa]','Vc']]
		pulse_voltage = int(pulse_voltage)
		if np.isnan(file_index):
			continue
		full_folder = '/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/'+"{0:0=2d}".format(sequence)+'/fast_camera/'
		type = '.cine'
		filenames = np.array(all_file_names(full_folder, type))
		# select = [name[:2]==str(file_index) for name in filenames]
		select = [name[:2]=="{0:0=2d}".format(int(file_index)) for name in filenames]
		filenames = filenames[select]

		mm_per_pixel = 15/44	# mm/#
		target_location = 230*mm_per_pixel +target_OES_distance - 31.2
		target_location_left = target_location - 16*mm_per_pixel
		target_location_right = target_location - 11*mm_per_pixel
		target_location_left_pixel = int(round(target_location/mm_per_pixel))-16
		target_location_right_pixel = int(round(target_location/mm_per_pixel))-11
		# LOS_size = 3.8	# mm diameter
		# in reality it's smaller than the resolution. looking at acually feeding light of the right wavelength the imprint is very small, with a vague halo around the main spot, but a weak one
		LOS_size = 1	# mm diameter
		OES_location = 230*mm_per_pixel - 31.2
		OES_location_left = 230*mm_per_pixel - 31.2 - LOS_size/2
		OES_location_right = 230*mm_per_pixel - 31.2 + LOS_size/2
		OES_location_left_pixel = int(round(OES_location_left/mm_per_pixel))
		OES_location_right_pixel = int(round(OES_location_right/mm_per_pixel))

		# median_50 = []
		# median_100 = []
		# median_OES = []

		for i_filename,filename in enumerate(filenames):
			raw_images, setup, bpp = read_frames(full_folder+filename)#,start_frame=1,count=1)
			# header = read_header('/home/ffederic/work/Collaboratory/test/experimental_data/2019-05-03/01/fast_camera/02_1.cine')
			framerate = setup.FrameRate
			raw_data = []
			for image in raw_images:
				raw_data.append(image)
			raw_data = np.array(raw_data).astype('int')
			raw_data = raw_data[np.max(raw_data,axis=(1,2))>0]

			# counts are normalised for a 1ms exposure
			raw_data = raw_data/(setup.ShutterNs/1000)

			if len(averaged_profile)==0:
				averaged_profile = raw_data.tolist()
			else:
				averaged_profile.extend(raw_data)

			cleaned = medfilt(raw_data-raw_data[np.sum(raw_data,axis=(-1,-2)).argmin()],[1,3,3])

			maximum = np.max(np.sum(cleaned[:,:,50],axis=1))
			minimum = np.min(np.sum(cleaned[:,:,50],axis=1))
			# select = (np.sum(cleaned[:,:,50],axis=1)>(maximum-minimum)/10).argmax()
			# select = np.sum(cleaned[:,:,150:230],axis=(1,2)).argmax()
			select = (np.sum(raw_data>0,axis=(1,2))>5).argmax()	# I do this because I want them all aligned to when the pulse frst arrive in the camera field of view

			# select_for_plot = np.sum(cleaned[:,60:200,target_location_left_pixel:target_location_right_pixel+1],axis=(1,2)).argmax()
			# plt.figure()
			# plt.imshow(cleaned[select_for_plot],'rainbow',origin='lower',extent=[0,np.shape(cleaned[select_for_plot])[0]*mm_per_pixel,0,np.shape(cleaned[select_for_plot])[1]*mm_per_pixel])
			# # plt.imshow(cleaned[select],'rainbow',origin='lower')
			# plt.colorbar()
			# plt.plot([OES_location_left,OES_location_left],[60*mm_per_pixel,200*mm_per_pixel],'--k',label='OES location')
			# plt.plot([OES_location_right,OES_location_right],[60*mm_per_pixel,200*mm_per_pixel],'--k')
			# plt.plot([47*mm_per_pixel,47*mm_per_pixel],[60*mm_per_pixel,200*mm_per_pixel],':k')
			# plt.plot([(53-1)*mm_per_pixel,(53-1)*mm_per_pixel],[60*mm_per_pixel,200*mm_per_pixel],':k')
			# plt.plot([97*mm_per_pixel,97*mm_per_pixel],[60*mm_per_pixel,200*mm_per_pixel],':k')
			# plt.plot([(103-1)*mm_per_pixel,(103-1)*mm_per_pixel],[60*mm_per_pixel,200*mm_per_pixel],':k')
			# plt.plot([214*mm_per_pixel,214*mm_per_pixel],[60*mm_per_pixel,200*mm_per_pixel],':k')
			# plt.plot([(220-1)*mm_per_pixel,(220-1)*mm_per_pixel],[60*mm_per_pixel,200*mm_per_pixel],':k')
			# plt.title(full_folder+filename+'\nmagnetic_field %.3gT,steady state pressure %.3gPa,target/OES distance %.3gmm,ELM pulse voltage %.3gV\nsampled regions' %(magnetic_field,SS_pressure,target_OES_distance,pulse_voltage))
			# plt.legend(loc='best')
			# plt.xlabel('[mm]')
			# plt.ylabel('[mm]')
			# plt.grid()
			# plt.pause(0.01)

			# if i_filename==0:
			# 		median_50 = np.zeros((len(filenames),2*len(cleaned)))
			# 		median_100 = np.zeros((len(filenames),2*len(cleaned)))
			# 		median_OES = np.zeros((len(filenames),2*len(cleaned)))
			# 		median_200 = np.zeros((len(filenames),2*len(cleaned)))
			# 		median_250 = np.zeros((len(filenames),2*len(cleaned)))

			if median_50==[]:
				median_50 = np.zeros((1,2*300))
				median_100 = np.zeros((1,2*300))
				median_OES = np.zeros((1,2*300))
				median_target = np.zeros((1,2*300))
				median_250 = np.zeros((1,2*300))

				# median_50[0,-select+300:len(cleaned)-select+300] = np.mean(cleaned[:,60:200,47:53],axis=(1,2))
				# median_100[0,-select+300:len(cleaned)-select+300] = np.mean(cleaned[:,60:200,97:103],axis=(1,2))
				# median_OES[0,-select+300:len(cleaned)-select+300] = np.mean(cleaned[:,60:200,OES_location_left_pixel:OES_location_right_pixel+1],axis=(1,2))
				# median_target[0,-select+300:len(cleaned)-select+300] = np.mean(cleaned[:,60:200,target_location_left_pixel:target_location_right_pixel+1],axis=(1,2))
				# median_250[0,-select+300:len(cleaned)-select+300] = np.mean(cleaned[:,60:200,250:255],axis=(1,2))
			else:
				median_50 = np.concatenate((median_50,np.zeros((1,2*300))))
				median_100 = np.concatenate((median_100,np.zeros((1,2*300))))
				median_OES = np.concatenate((median_OES,np.zeros((1,2*300))))
				median_target = np.concatenate((median_target,np.zeros((1,2*300))))
				median_250 = np.concatenate((median_250,np.zeros((1,2*300))))

			median_50[-1,-select+300:len(cleaned)-select+300] = np.mean(cleaned[:,60:200,47:53],axis=(1,2))
			median_100[-1,-select+300:len(cleaned)-select+300] = np.mean(cleaned[:,60:200,97:103],axis=(1,2))
			median_OES[-1,-select+300:len(cleaned)-select+300] = np.mean(cleaned[:,60:200,OES_location_left_pixel:OES_location_right_pixel+1],axis=(1,2))
			median_target[-1,-select+300:len(cleaned)-select+300] = np.mean(cleaned[:,60:200,target_location_left_pixel:target_location_right_pixel+1],axis=(1,2))
			median_250[-1,-select+300:len(cleaned)-select+300] = np.mean(cleaned[:,60:200,250:255],axis=(1,2))

	averaged_profile = np.mean(averaged_profile,axis=0)
	averaged_profile_all.append(averaged_profile)
	temp = np.mean(averaged_profile[:,47:53],axis=1)
	axis_pixel = max(1,temp.argmax())
	axis_loc_50 = axis_pixel*mm_per_pixel
	# top_loc_50 = ((np.mean(averaged_profile[axis_pixel:,47:53],axis=1)>5).argmin() + axis_pixel)*mm_per_pixel
	# bottom_loc_50 = (np.mean(averaged_profile[:axis_pixel,47:53],axis=1)<5).argmin()*mm_per_pixel
	top_loc_50 = (np.abs(temp[axis_pixel:]-temp.argmax()/10).argmin()+axis_pixel)*mm_per_pixel
	bottom_loc_50 = (np.abs(temp[:axis_pixel]-temp.argmax()/10).argmin())*mm_per_pixel
	plasma_diameter_50.append(top_loc_50-bottom_loc_50)
	temp = np.mean(averaged_profile[:,97:103],axis=1)
	axis_pixel = max(1,temp.argmax())
	axis_loc_100 = axis_pixel*mm_per_pixel
	# top_loc_100 = ((np.mean(averaged_profile[axis_pixel:,97:103],axis=1)>5).argmin() + axis_pixel)*mm_per_pixel
	# bottom_loc_100 = (np.mean(averaged_profile[:axis_pixel,97:103],axis=1)<5).argmin()*mm_per_pixel
	top_loc_100 = (np.abs(temp[axis_pixel:]-temp.argmax()/10).argmin()+axis_pixel)*mm_per_pixel
	bottom_loc_100 = (np.abs(temp[:axis_pixel]-temp.argmax()/10).argmin())*mm_per_pixel
	plasma_diameter_100.append(top_loc_100-bottom_loc_100)
	temp = np.mean(averaged_profile[:,OES_location_left_pixel:OES_location_right_pixel+1],axis=1)
	axis_pixel = max(1,temp.argmax())
	axis_loc_OES = axis_pixel*mm_per_pixel
	# top_loc_OES = ((np.mean(averaged_profile[axis_pixel:,OES_location_left_pixel:OES_location_right_pixel+1],axis=1)>5).argmin() + axis_pixel)*mm_per_pixel
	# bottom_loc_OES = (np.mean(averaged_profile[:axis_pixel,OES_location_left_pixel:OES_location_right_pixel+1],axis=1)<5).argmin()*mm_per_pixel
	top_loc_OES = (np.abs(temp[axis_pixel:]-temp.argmax()/10).argmin()+axis_pixel)*mm_per_pixel
	bottom_loc_OES = (np.abs(temp[:axis_pixel]-temp.argmax()/10).argmin())*mm_per_pixel
	plasma_diameter_OES.append(top_loc_OES-bottom_loc_OES)
	temp = np.mean(averaged_profile[:,target_location_left_pixel:target_location_right_pixel+1],axis=1)
	axis_pixel = max(1,temp.argmax())
	axis_loc_target = axis_pixel*mm_per_pixel
	# top_loc_target = ((np.mean(averaged_profile[axis_pixel:,target_location_left_pixel:target_location_right_pixel+1],axis=1)>5).argmin() + axis_pixel)*mm_per_pixel
	# bottom_loc_target = (np.mean(averaged_profile[:axis_pixel,target_location_left_pixel:target_location_right_pixel+1],axis=1)<5).argmin()*mm_per_pixel
	top_loc_target = (np.abs(temp[axis_pixel:]-temp.argmax()/10).argmin()+axis_pixel)*mm_per_pixel
	bottom_loc_target = (np.abs(temp[:axis_pixel]-temp.argmax()/10).argmin())*mm_per_pixel
	plasma_diameter_target.append(top_loc_target-bottom_loc_target)

			# median_100.append(np.sum(cleaned[select:,:,100],axis=1))
			# median_OES.append(np.sum(cleaned[select:,:,166],axis=1))

	# 		plt.plot(np.arange(-select,len(cleaned)-select),np.mean(cleaned[:,60:200,50],axis=1),'--',label=filename+'__50')
	# 		plt.plot(np.arange(-select,len(cleaned)-select),np.mean(cleaned[:,60:200,100],axis=1),':',label=filename+'__100')
	# 		plt.plot(np.arange(-select,len(cleaned)-select),np.mean(cleaned[:,60:200,166],axis=1),label=filename+'__OES')
	# 		plt.plot(np.arange(-select,len(cleaned)-select),np.mean(cleaned[:,60:200,223],axis=1),label=filename+'__223')
	# plt.legend(loc='best')
	# plt.title(folder+filename)
	# plt.grid()
	# plt.pause(0.01)
	plt.figure(figure_index)
	if counter==0:
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_50,axis=0),'--C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', %.3gmm from OES' %(OES_location-50*mm_per_pixel))
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_100,axis=0),':C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', %.3gmm from OES' %(OES_location-100*mm_per_pixel))
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_OES,axis=0),'C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', OES location')
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_50,axis=0),'--C'+str(counter),label='pressure %.3gPa' %(SS_pressure)+', %.3gmm from OES' %(OES_location-50*mm_per_pixel))
		plt.plot([(np.arange(-300,300)*1000/framerate)[np.median(median_50,axis=0).argmax()]]*2,[0,np.median(median_50,axis=0).max()],'--C'+str(counter))
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_100,axis=0),':C'+str(counter),label='pressure %.3gPa' %(SS_pressure)+', %.3gmm from OES' %(OES_location-100*mm_per_pixel))
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_OES,axis=0),'C'+str(counter),label='pressure %.3gPa' %(SS_pressure)+', OES location')
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_223,axis=0),'-.C'+str(counter),label='pressure %.3gPa' %(SS_pressure)+', -%.3gmm from OES' %((223-100)*mm_per_pixel))
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_target,axis=0),'-.C'+str(counter),label='pressure %.3gPa' %(SS_pressure)+', target location')
		plt.plot([(np.arange(-300,300)*1000/framerate)[np.median(median_target,axis=0).argmax()]]*2,[0,np.median(median_target,axis=0).max()],'-.C'+str(counter))
	else:
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_50,axis=0),'--C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', %.3gmm from OES' %(OES_location-50*mm_per_pixel))
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_100,axis=0),':C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', %.3gmm from OES' %(OES_location-100*mm_per_pixel))
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_OES,axis=0),'C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', OES location')
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_50,axis=0),'--C'+str(counter))
		plt.plot([(np.arange(-300,300)*1000/framerate)[np.median(median_50,axis=0).argmax()]]*2,[0,np.median(median_50,axis=0).max()],'--C'+str(counter))
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_100,axis=0),':C'+str(counter))
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_OES,axis=0),'C'+str(counter))
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_target,axis=0),'-.C'+str(counter),label='pressure %.3gPa' %(SS_pressure))
		plt.plot([(np.arange(-300,300)*1000/framerate)[np.median(median_target,axis=0).argmax()]]*2,[0,np.median(median_target,axis=0).max()],'-.C'+str(counter))
		plt.legend(loc='best')
		# plt.title(full_folder+filename)
		plt.grid()
		plt.pause(0.01)

	plt.figure(figure_index+1)
	if counter==0:
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_50,axis=0),'--C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', %.3gmm from OES' %(OES_location-50*mm_per_pixel))
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_100,axis=0),':C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', %.3gmm from OES' %(OES_location-100*mm_per_pixel))
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_OES,axis=0),'C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', OES location')
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_50,axis=0)/np.median(median_target,axis=0),'--C'+str(counter),label='pressure %.3gPa' %(SS_pressure)+', %.3gmm from OES' %(OES_location-50*mm_per_pixel))
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_100,axis=0)/np.median(median_target,axis=0),':C'+str(counter),label='pressure %.3gPa' %(SS_pressure)+', %.3gmm from OES' %(OES_location-100*mm_per_pixel))
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_OES,axis=0)/np.median(median_target,axis=0),'C'+str(counter),label='pressure %.3gPa' %(SS_pressure)+', OES location')
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_223,axis=0),'-.C'+str(counter),label='pressure %.3gPa' %(SS_pressure)+', -%.3gmm from OES' %((223-100)*mm_per_pixel))
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_target,axis=0)/np.median(median_OES,axis=0),'-.C'+str(counter),label='pressure %.3gPa' %(SS_pressure)+', target location')
	else:
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_50,axis=0),'--C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', %.3gmm from OES' %(OES_location-50*mm_per_pixel))
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_100,axis=0),':C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', %.3gmm from OES' %(OES_location-100*mm_per_pixel))
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_OES,axis=0),'C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', OES location')
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_50,axis=0)/np.median(median_target,axis=0),'--C'+str(counter),label='pressure %.3gPa' %(SS_pressure))
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_100,axis=0)/np.median(median_target,axis=0),':C'+str(counter))
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_OES,axis=0)/np.median(median_target,axis=0),'C'+str(counter))
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_target,axis=0)/np.median(median_OES,axis=0),'-.C'+str(counter))
		plt.legend(loc='best')
		# plt.title(full_folder+filename)
		plt.grid()
		plt.pause(0.01)

	plt.figure(figure_index+2)
	if counter==0:
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_50,axis=0),'--C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', %.3gmm from OES' %(OES_location-50*mm_per_pixel))
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_100,axis=0),':C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', %.3gmm from OES' %(OES_location-100*mm_per_pixel))
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_OES,axis=0),'C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', OES location')
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_50,axis=0)/(plasma_diameter_50[-1]**2),'--C'+str(counter),label='pressure %.3gPa' %(SS_pressure)+', %.3gmm from OES' %(OES_location-50*mm_per_pixel))
		plt.plot([(np.arange(-300,300)*1000/framerate)[np.median(median_50,axis=0).argmax()]]*2,[0,np.median(median_50,axis=0).max()/(plasma_diameter_50[-1]**2)],'--C'+str(counter))
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_100,axis=0)/(plasma_diameter_100[-1]**2),':C'+str(counter),label='pressure %.3gPa' %(SS_pressure)+', %.3gmm from OES' %(OES_location-100*mm_per_pixel))
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_OES,axis=0)/(plasma_diameter_OES[-1]**2),'C'+str(counter),label='pressure %.3gPa' %(SS_pressure)+', OES location')
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_223,axis=0),'-.C'+str(counter),label='pressure %.3gPa' %(SS_pressure)+', -%.3gmm from OES' %((223-100)*mm_per_pixel))
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_target,axis=0)/(plasma_diameter_target[-1]**2),'-.C'+str(counter),label='pressure %.3gPa' %(SS_pressure)+', target location')
		plt.plot([(np.arange(-300,300)*1000/framerate)[np.median(median_target,axis=0).argmax()]]*2,[0,np.median(median_target,axis=0).max()/(plasma_diameter_target[-1]**2)],'-.C'+str(counter))
	else:
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_50,axis=0),'--C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', %.3gmm from OES' %(OES_location-50*mm_per_pixel))
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_100,axis=0),':C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', %.3gmm from OES' %(OES_location-100*mm_per_pixel))
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_OES,axis=0),'C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', OES location')
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_50,axis=0)/(plasma_diameter_50[-1]**2),'--C'+str(counter))
		plt.plot([(np.arange(-300,300)*1000/framerate)[np.median(median_50,axis=0).argmax()]]*2,[0,np.median(median_50,axis=0).max()/(plasma_diameter_50[-1]**2)],'--C'+str(counter))
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_100,axis=0)/(plasma_diameter_100[-1]**2),':C'+str(counter))
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_OES,axis=0)/(plasma_diameter_OES[-1]**2),'C'+str(counter))
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_target,axis=0)/(plasma_diameter_target[-1]**2),'-.C'+str(counter),label='pressure %.3gPa' %(SS_pressure))
		plt.plot([(np.arange(-300,300)*1000/framerate)[np.median(median_target,axis=0).argmax()]]*2,[0,np.median(median_target,axis=0).max()/(plasma_diameter_target[-1]**2)],'-.C'+str(counter))
		plt.legend(loc='best')
		# plt.title(full_folder+filename)
		plt.grid()
		plt.pause(0.01)

	plt.figure(figure_index+3)
	if counter==0:
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_50,axis=0),'--C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', %.3gmm from OES' %(OES_location-50*mm_per_pixel))
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_100,axis=0),':C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', %.3gmm from OES' %(OES_location-100*mm_per_pixel))
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_OES,axis=0),'C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', OES location')
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_50,axis=0)/np.median(median_target,axis=0)/(plasma_diameter_50[-1]**2)*(plasma_diameter_target[-1]**2),'--C'+str(counter),label='pressure %.3gPa' %(SS_pressure)+', %.3gmm from OES' %(OES_location-50*mm_per_pixel))
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_100,axis=0)/np.median(median_target,axis=0)/(plasma_diameter_100[-1]**2)*(plasma_diameter_target[-1]**2),':C'+str(counter),label='pressure %.3gPa' %(SS_pressure)+', %.3gmm from OES' %(OES_location-100*mm_per_pixel))
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_OES,axis=0)/np.median(median_target,axis=0)/(plasma_diameter_OES[-1]**2)*(plasma_diameter_target[-1]**2),'C'+str(counter),label='pressure %.3gPa' %(SS_pressure)+', OES location')
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_223,axis=0),'-.C'+str(counter),label='pressure %.3gPa' %(SS_pressure)+', -%.3gmm from OES' %((223-100)*mm_per_pixel))
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_target,axis=0)/np.median(median_OES,axis=0),'-.C'+str(counter),label='pressure %.3gPa' %(SS_pressure)+', target location')
	else:
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_50,axis=0),'--C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', %.3gmm from OES' %(OES_location-50*mm_per_pixel))
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_100,axis=0),':C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', %.3gmm from OES' %(OES_location-100*mm_per_pixel))
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_OES,axis=0),'C'+str(counter),label='target/OES distance %.3gmm' %(target_OES_distance)+', OES location')
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_50,axis=0)/np.median(median_target,axis=0)/(plasma_diameter_50[-1]**2)*(plasma_diameter_target[-1]**2),'--C'+str(counter),label='pressure %.3gPa' %(SS_pressure))
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_100,axis=0)/np.median(median_target,axis=0)/(plasma_diameter_100[-1]**2)*(plasma_diameter_target[-1]**2),':C'+str(counter))
		plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_OES,axis=0)/np.median(median_target,axis=0)/(plasma_diameter_OES[-1]**2)*(plasma_diameter_target[-1]**2),'C'+str(counter))
		# plt.plot(np.arange(-300,300)*1000/framerate,np.median(median_target,axis=0)/np.median(median_OES,axis=0),'-.C'+str(counter))
		plt.legend(loc='best')
		# plt.title(full_folder+filename)
		plt.grid()
		plt.pause(0.01)
	counter+=1

	SS_pressure_all.append(SS_pressure)
	target_OES_distance_all.append(target_OES_distance)

	median_50_all.append(np.median(median_50,axis=0))
	median_100_all.append(np.median(median_100,axis=0))
	median_OES_all.append(np.median(median_OES,axis=0))
	median_target_all.append(np.median(median_target,axis=0))
	median_250_all.append(np.median(median_250,axis=0))


plt.figure(figure_index)
plt.legend(loc='best', fontsize='x-small')
# plt.title('Target/OES distance scan magnetic_field %.3gT,steady state pressure %.3gPa,ELM pulse voltage %.3gV' %(magnetic_field,SS_pressure,pulse_voltage)+'\naverage of the fast camera counts between vertical pixels 60 and 200 at the specified location')
plt.title('Pressure scan magnetic_field %.3gT,target/OES distance %.3gmm,ELM pulse voltage %.3gV' %(magnetic_field,target_OES_distance,pulse_voltage)+'\naverage of the fast camera counts between vertical pixels 60 and 200 at the specified location')
plt.grid()
plt.xlabel('time [ms]')
plt.ylabel('average counts [au]')
plt.pause(0.01)
plt.figure(figure_index+1)
plt.legend(loc='best', fontsize='x-small')
# plt.title('Target/OES distance scan magnetic_field %.3gT,steady state pressure %.3gPa,ELM pulse voltage %.3gV' %(magnetic_field,SS_pressure,pulse_voltage)+'\naverage of the fast camera counts between vertical pixels 60 and 200 at the specified location')
plt.title('Pressure scan magnetic_field %.3gT,target/OES distance %.3gmm,ELM pulse voltage %.3gV' %(magnetic_field,target_OES_distance,pulse_voltage)+'\nRatio of emissivity in a location over at the target')
plt.grid()
plt.xlabel('time [ms]')
plt.ylabel('average counts ratio [au]')
plt.yscale('log')
plt.pause(0.01)
plt.figure(figure_index+2)
plt.legend(loc='best', fontsize='x-small')
# plt.title('Target/OES distance scan magnetic_field %.3gT,steady state pressure %.3gPa,ELM pulse voltage %.3gV' %(magnetic_field,SS_pressure,pulse_voltage)+'\naverage of the fast camera counts between vertical pixels 60 and 200 at the specified location')
plt.title('Pressure scan magnetic_field %.3gT,target/OES distance %.3gmm,ELM pulse voltage %.3gV' %(magnetic_field,target_OES_distance,pulse_voltage)+'\naverage of the fast camera counts between vertical pixels 60 and 200 at the specified location\ndivided by plasma diameter square')
plt.grid()
plt.xlabel('time [ms]')
plt.ylabel('average counts/diameter [au/mm]')
plt.pause(0.01)
plt.figure(figure_index+3)
plt.legend(loc='best', fontsize='x-small')
# plt.title('Target/OES distance scan magnetic_field %.3gT,steady state pressure %.3gPa,ELM pulse voltage %.3gV' %(magnetic_field,SS_pressure,pulse_voltage)+'\naverage of the fast camera counts between vertical pixels 60 and 200 at the specified location')
plt.title('Pressure scan magnetic_field %.3gT,target/OES distance %.3gmm,ELM pulse voltage %.3gV' %(magnetic_field,target_OES_distance,pulse_voltage)+'\nRatio of emissivity in a location over at the target\ndivided by plasma diameter square')
plt.grid()
plt.xlabel('time [ms]')
plt.ylabel('average counts ratio/diameter [au/mm]')
plt.yscale('log')
plt.pause(0.01)


plt.figure(figsize=(12,6))
temp = np.divide(median_50_all,median_target_all)
temp[np.isinf(temp)]=0
plt.plot(SS_pressure_all,np.nanmax(temp,axis=-1),'C0',label='emission %.3gmm from OES/target location\npeak ratio' %(OES_location-50*mm_per_pixel))
plt.plot(SS_pressure_all,np.nanmean(temp,axis=-1),'--C0',label='emission %.3gmm from OES/target location\nmean ratio' %(OES_location-50*mm_per_pixel))
plt.plot(SS_pressure_all,np.diagonal(temp[:,np.array(median_target_all).argmax(axis=1)]),':C0',label='emission %.3gmm from OES/target location\nratio at max target emissivity' %(OES_location-50*mm_per_pixel))
temp = np.divide(median_100_all,median_target_all)
temp[np.isinf(temp)]=0
plt.plot(SS_pressure_all,np.nanmax(temp,axis=-1),'C1',label='emission %.3gmm from OES/target location' %(OES_location-100*mm_per_pixel))
plt.plot(SS_pressure_all,np.nanmean(temp,axis=-1),'--C1')
plt.plot(SS_pressure_all,np.diagonal(temp[:,np.array(median_target_all).argmax(axis=1)]),':C1')
temp = np.divide(median_OES_all,median_target_all)
temp[np.isinf(temp)]=0
plt.plot(SS_pressure_all,np.nanmax(temp,axis=-1),'C2',label='emission OES/target location')
plt.plot(SS_pressure_all,np.nanmean(temp,axis=-1),'--C2')
plt.plot(SS_pressure_all,np.diagonal(temp[:,np.array(median_target_all).argmax(axis=1)]),':C2')
plt.grid()
plt.legend(loc='best', fontsize='x-small')
# plt.title('Target/OES distance scan magnetic_field %.3gT,steady state pressure %.3gPa,ELM pulse voltage %.3gV' %(magnetic_field,SS_pressure,pulse_voltage)+'\naverage of the fast camera counts between vertical pixels 60 and 200 at the specified location')
plt.title('Pressure scan magnetic_field %.3gT,target/OES distance %.3gmm,ELM pulse voltage %.3gV' %(magnetic_field,target_OES_distance,pulse_voltage)+'\nRatio of emissivity in a location over at the target')
plt.xlabel('Pressure [Pa]')
plt.ylabel('Emission ratio [au]')
plt.ylim(bottom=1e-3)
plt.yscale('log')
plt.pause(0.01)

plt.figure(figsize=(12,6))
averaged_profile_all = np.array(averaged_profile_all)
# temp = np.divide(median_50_all,median_target_all)
# temp[np.isinf(temp)]=0
plt.plot(SS_pressure_all,np.mean(averaged_profile_all[:,60:200,47:53],axis=(1,2))/np.mean(averaged_profile_all[:,60:200,target_location_left_pixel:target_location_right_pixel+1],axis=(1,2)),'-C0',label='emission %.3gmm from OES/target location' %(OES_location-50*mm_per_pixel))
# temp = np.divide(median_100_all,median_target_all)
# temp[np.isinf(temp)]=0
plt.plot(SS_pressure_all,np.mean(averaged_profile_all[:,60:200,97:103],axis=(1,2))/np.mean(averaged_profile_all[:,60:200,target_location_left_pixel:target_location_right_pixel+1],axis=(1,2)),'-C1',label='emission %.3gmm from OES/target location' %(OES_location-100*mm_per_pixel))
# temp = np.divide(median_OES_all,median_target_all)
# temp[np.isinf(temp)]=0
plt.plot(SS_pressure_all,np.mean(averaged_profile_all[:,60:200,OES_location_left_pixel:OES_location_right_pixel+1],axis=(1,2))/np.mean(averaged_profile_all[:,60:200,target_location_left_pixel:target_location_right_pixel+1],axis=(1,2)),'-C2',label='emission OES/target location')
plt.grid()
plt.legend(loc='best', fontsize='x-small')
# plt.title('Target/OES distance scan magnetic_field %.3gT,steady state pressure %.3gPa,ELM pulse voltage %.3gV' %(magnetic_field,SS_pressure,pulse_voltage)+'\naverage of the fast camera counts between vertical pixels 60 and 200 at the specified location')
plt.title('Pressure scan magnetic_field %.3gT,target/OES distance %.3gmm,ELM pulse voltage %.3gV' %(magnetic_field,target_OES_distance,pulse_voltage)+'\nRatio of mean emissivity in a location over at the target')
plt.xlabel('Pressure [Pa]')
plt.ylabel('Emission ratio [au]')
plt.ylim(bottom=1e-3)
plt.yscale('log')
plt.pause(0.01)

plt.figure(figsize=(12,6))
temp = (np.divide(median_50_all,median_target_all).T*(np.divide(plasma_diameter_target,plasma_diameter_50)**2)).T
temp[np.isinf(temp)]=0
plt.plot(SS_pressure_all,np.nanmax(temp,axis=-1),'C0',label='emission %.3gmm from OES/target location\npeak ratio' %(OES_location-50*mm_per_pixel))
plt.plot(SS_pressure_all,np.nanmean(temp,axis=-1),'--C0',label='emission %.3gmm from OES/target location\nmean ratio' %(OES_location-50*mm_per_pixel))
plt.plot(SS_pressure_all,np.diagonal(temp[:,np.array(median_target_all).argmax(axis=1)]),':C0',label='emission %.3gmm from OES/target location\nratio at max target emissivity' %(OES_location-50*mm_per_pixel))
temp = (np.divide(median_100_all,median_target_all).T*(np.divide(plasma_diameter_target,plasma_diameter_100)**2)).T
temp[np.isinf(temp)]=0
plt.plot(SS_pressure_all,np.nanmax(temp,axis=-1),'C1',label='emission %.3gmm from OES/target location' %(OES_location-100*mm_per_pixel))
plt.plot(SS_pressure_all,np.nanmean(temp,axis=-1),'--C1')
plt.plot(SS_pressure_all,np.diagonal(temp[:,np.array(median_target_all).argmax(axis=1)]),':C1')
temp = (np.divide(median_OES_all,median_target_all).T*(np.divide(plasma_diameter_target,plasma_diameter_OES)**2)).T
temp[np.isinf(temp)]=0
plt.plot(SS_pressure_all,np.nanmax(temp,axis=-1),'C2',label='emission OES/target location')
plt.plot(SS_pressure_all,np.nanmean(temp,axis=-1),'--C2')
plt.plot(SS_pressure_all,np.diagonal(temp[:,np.array(median_target_all).argmax(axis=1)]),':C2')
plt.grid()
plt.legend(loc='best', fontsize='x-small')
# plt.title('Target/OES distance scan magnetic_field %.3gT,steady state pressure %.3gPa,ELM pulse voltage %.3gV' %(magnetic_field,SS_pressure,pulse_voltage)+'\naverage of the fast camera counts between vertical pixels 60 and 200 at the specified location')
plt.title('Pressure scan magnetic_field %.3gT,target/OES distance %.3gmm,ELM pulse voltage %.3gV' %(magnetic_field,target_OES_distance,pulse_voltage)+'\nRatio of emissivity in a location over at the target\ndivided by relative plasma diameter square')
plt.xlabel('Pressure [Pa]')
plt.ylabel('Emission ratio/diameter [au/mm]')
# plt.ylim(bottom=1e-3)
plt.yscale('log')
plt.pause(0.01)

plt.figure(figsize=(12,6))
# temp = (np.divide(median_50_all,median_target_all).T/(np.array(plasma_diameter_50)*np.array(plasma_diameter_target))**2).T
# temp[np.isinf(temp)]=0
plt.plot(SS_pressure_all,np.mean(averaged_profile_all[:,60:200,47:53],axis=(1,2))/np.mean(averaged_profile_all[:,60:200,target_location_left_pixel:target_location_right_pixel+1],axis=(1,2))*(np.divide(plasma_diameter_target,plasma_diameter_50)**2),'-C0',label='emission %.3gmm from OES/target location' %(OES_location-50*mm_per_pixel))
# temp = (np.divide(median_100_all,median_target_all).T/(np.array(plasma_diameter_100)*np.array(plasma_diameter_target))**2).T
# temp[np.isinf(temp)]=0
plt.plot(SS_pressure_all,np.mean(averaged_profile_all[:,60:200,97:103],axis=(1,2))/np.mean(averaged_profile_all[:,60:200,target_location_left_pixel:target_location_right_pixel+1],axis=(1,2))*(np.divide(plasma_diameter_target,plasma_diameter_100)**2),'-C1',label='emission %.3gmm from OES/target location' %(OES_location-100*mm_per_pixel))
# temp = (np.divide(median_OES_all,median_target_all).T/(np.array(plasma_diameter_OES)*np.array(plasma_diameter_target))**2).T
# temp[np.isinf(temp)]=0
plt.plot(SS_pressure_all,np.mean(averaged_profile_all[:,60:200,OES_location_left_pixel:OES_location_right_pixel+1],axis=(1,2))/np.mean(averaged_profile_all[:,60:200,target_location_left_pixel:target_location_right_pixel+1],axis=(1,2))*(np.divide(plasma_diameter_target,plasma_diameter_OES)**2),'-C2',label='emission OES/target location')
plt.grid()
plt.legend(loc='best', fontsize='x-small')
# plt.title('Target/OES distance scan magnetic_field %.3gT,steady state pressure %.3gPa,ELM pulse voltage %.3gV' %(magnetic_field,SS_pressure,pulse_voltage)+'\naverage of the fast camera counts between vertical pixels 60 and 200 at the specified location')
plt.title('Pressure scan magnetic_field %.3gT,target/OES distance %.3gmm,ELM pulse voltage %.3gV' %(magnetic_field,target_OES_distance,pulse_voltage)+'\nRatio of mean emissivity in a location over at the target\ndivided by relative plasma diameter square')
plt.xlabel('Pressure [Pa]')
plt.ylabel('Emission ratio/diameter [au/mm]')
# plt.ylim(bottom=1e-3)
plt.yscale('log')
plt.pause(0.01)

plt.rcParams.update({'font.size': 20})
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise', 'teal', 'olive']
fig = plt.figure(figsize=(13,8))
ax1 = fig.add_subplot(1,1,1)
# if merge_ID_target in [85,97]:
# 	ax2 = fig.add_subplot(5,5,7, facecolor='w',yscale='linear')
ax1.plot([],[],'w',label='Pressure')
for i in range(len(averaged_profile_all)):
	temp_temp = np.mean(averaged_profile_all[i],axis=0)
	temp_temp[np.arange(np.shape(averaged_profile_all)[1])>230]=0
	target_location = 230*mm_per_pixel +target_OES_distance_all[i] - 31.2
	ax1.plot(np.arange(np.shape(averaged_profile_all)[1])*mm_per_pixel-target_location,temp_temp,color=color[i],label='%.3gPa' %(SS_pressure_all[i]))
# plt.yscale('log')
ax1.axvline(x=0,color='k',linestyle='-.',label='target')
ax1.axvline(x=OES_location_left-target_location,color='k',linestyle='-',label='TS/OES')
ax1.axvline(x=OES_location_right-target_location,color='k',linestyle='-')
ax1.set_title('Pressure scan magnetic_field %.3gT,target/OES distance %.3gmm,ELM pulse voltage %.3gV' %(magnetic_field,target_OES_distance,pulse_voltage)+'\nAverage of emissivity in the radial direction\n')
ax1.set_xlabel('longitudinal position [mm]')
ax1.set_ylabel('Average brightness [au]')
ax1.grid()
ax1.set_yscale('log')
ax1.set_ylim(bottom=0.1)
if merge_ID_target==85:
	ax2 = plt.axes([.2, .54, .2, .29], facecolor='w',yscale='linear')
	ax1.plot(np.array([75,78,78,75,75])-target_location,[600,600,800,800,600],'k--',label='linear detail')
	for i in range(len(averaged_profile_all)):
		temp_temp = np.mean(averaged_profile_all[i],axis=0)
		temp_temp[np.arange(np.shape(averaged_profile_all)[1])>230]=0
		ax2.plot(np.arange(np.shape(averaged_profile_all)[1])*mm_per_pixel,temp_temp,color=color[i])
	ax2.set_xlim(left=75,right=78)
	ax2.set_ylim(bottom=600,top=800)
	ax2.set_yscale('linear')
	ax2.set_title('linear detail')
# elif merge_ID_target==97:
# 	ax2 = plt.axes([.2, .5, .2, .32], facecolor='w',yscale='linear')
# 	ax1.plot(np.array([75,78,78,75,75])-target_location,[100,100,210,210,100],'k--',label='linear detail')
# 	for i in range(len(averaged_profile_all)):
# 		temp_temp = np.mean(averaged_profile_all[i],axis=0)
# 		temp_temp[np.arange(np.shape(averaged_profile_all)[1])>230]=0
# 		ax2.plot(np.arange(np.shape(averaged_profile_all)[1])*mm_per_pixel,temp_temp,color=color[i])
# 	ax2.set_xlim(left=75,right=78)
# 	ax2.set_ylim(bottom=100,top=210)
# 	ax2.set_yscale('linear')
# 	ax2.set_title('linear detail')
ax1.legend(loc=3, fontsize='x-small')
ax1.set_xlim(right=5)
ax1.set_ylim(bottom=1)
plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/fast camera/' +'39'+'.png', bbox_inches='tight')
# plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/fast camera/' +'20'+'.png', bbox_inches='tight')
plt.close()
# plt.pause(0.01)



plt.figure(figsize=(12,6))
plt.plot(SS_pressure_all,plasma_diameter_50,'C0',label='%.3gmm from OES' %(OES_location-50*mm_per_pixel))
plt.plot(SS_pressure_all,plasma_diameter_100,'C1',label='%.3gmm from OES' %(OES_location-100*mm_per_pixel))
plt.plot(SS_pressure_all,plasma_diameter_OES,'C2',label='OES location')
plt.plot(SS_pressure_all,plasma_diameter_target,'C3',label='target location')
plt.grid()
plt.legend(loc='best', fontsize='x-small')
# plt.title('Target/OES distance scan magnetic_field %.3gT,steady state pressure %.3gPa,ELM pulse voltage %.3gV' %(magnetic_field,SS_pressure,pulse_voltage)+'\naverage of the fast camera counts between vertical pixels 60 and 200 at the specified location')
plt.title('Pressure scan magnetic_field %.3gT,target/OES distance %.3gmm,ELM pulse voltage %.3gV' %(magnetic_field,target_OES_distance,pulse_voltage)+'\nPlasma diameter (where averaged counts are >5)')
plt.xlabel('Pressure [Pa]')
plt.ylabel('Plasma diameter [mm]')
# plt.ylim(bottom=1e-4)
# plt.yscale('log')
plt.pause(0.01)






# Data from Bayesian analysis

pressure = [0.3,0.5158,8.17,11.847,15,0.22,0.38,5.9908,10.956]
fraction_of_energy_removed = [0.252663622526636,0.244343891402715,0.828054298642534,0.897435897435897,0.843137254901961,0.209677419354839,0.29,0.076451612903226,0.040322580645161]
ELM_pulse_energy = [65.7,66.3,66.3,66.3,66.3,31,31,31,31]

plt.figure(figsize=(12,6))
plt.tricontourf(pressure, ELM_pulse_energy, fraction_of_energy_removed,2,cmap='rainbow')
plt.plot(pressure, ELM_pulse_energy,'+k')
plt.xlabel('Pressure [Pa]')
plt.ylabel('ELM pulse energy [J]')
plt.colorbar().set_label('Fraction of energy removed from plasma column [au]')
plt.pause(0.01)






# analysis of steady state plasma from Gijs, dating back to 2017
file_id = [9318,9319,9320,9321,9322,9323,9324,9325,9326,9327,9328,9329,9330,9331,9332,9333,9334,9335,9336,9337]
Te = [4.46,4.73,4.18,2.71,3.07,0,3.94,3.34,3.14,3.02,2.52,2.34,1.49,1.53,0.779,0.848,0.394,0.381,0,0]	# eV
ne = np.array([6.71,6.24,6.29,1.07,0.875,0,11.1,11.3,14.2,13.6,16.5,16.5,21.2,19.8,24.1,20.1,20.0,16.6,0,0])*1e19	# #/m3
z_position = [-170,-110,-90,-170,-110,-90,-170,-100,-170,-100,-170,-100,-170,-100,-170,-100,-170,-100,-170,-100]	# mm
exposure = [3000,900,900,10000,3000,3000,10000,10000,10000,10000,10000,10000,10000,10000,2000,2000,2000,2000,10000,10000]	# microseconds
target_chamber_pressure = [0.25,0.26,0.26,0.52,0.52,0.52,0.27,0.29,0.54,0.52,0.99,0.99,2.02,2.01,4.47,4.48,8.18,8.33,16.8,17.3]	# Pa
mm_per_pixel = 15/44	# mm/#

for i_j,j in enumerate(file_id):
	full_folder = '/home/ffederic/work/Collaboratory/test/experimental_data/fast camera/DetachmentMovieRawCine/'
	type = '.cine'
	filename = str(j) + type
	raw_images, setup, bpp = read_frames(full_folder+filename)#,start_frame=1,count=1)
	# header = read_header('/home/ffederic/work/Collaboratory/test/experimental_data/2019-05-03/01/fast_camera/02_1.cine')
	framerate = setup.FrameRate
	raw_data = []
	for image in raw_images:
		raw_data.append(image)
	raw_data = np.array(raw_data).astype('int')
	raw_data = raw_data[np.max(raw_data,axis=(1,2))>0]

	average = np.median(raw_data,axis=0)

	overexposed_local = False
	if raw_data.max()>=4095:
		overexposed_local = True
		overexposed_mask = np.zeros(np.shape(raw_data[0]), dtype=bool)
		overexposed_mask[np.max(raw_data,axis=0)>=4095]=True
		average[overexposed_mask] = np.nan

	# counts are normalised for a 1ms exposure
	average = average/(setup.ShutterNs/1000)

	target_location = 230*mm_per_pixel +  -(z_position[i_j]+78.8)-5  - 31.2	# this is for a tungsten dummy 1 target, as I think it was for theis set of experiment
	LOS_size = 1	# mm diameter
	LOS_number = 40
	LOS_interval = 1.06478505470992	# 10/02/2020 from	Calculate magnification_FF.xlsx
	LOS_full_size = (LOS_number-1)*LOS_interval + LOS_size
	TS_size = [-4.149230769230769056e+01, 4.416923076923076508e+01]
	TS_LOS_full_size = TS_size[1]-TS_size[0]
	TS_LOS_interval = (TS_size[1]-TS_size[0])/65
	OES_location = 230*mm_per_pixel - 31.2
	OES_location_left = 230*mm_per_pixel - 31.2 - LOS_size/2
	OES_location_right = 230*mm_per_pixel - 31.2 + LOS_size/2
	OES_location_left_pixel = int(round(OES_location_left/mm_per_pixel))
	OES_location_right_pixel = int(round(OES_location_right/mm_per_pixel))
	TS_location_left = 230*mm_per_pixel - 31.2 - TS_LOS_interval/2
	TS_location_right = 230*mm_per_pixel - 31.2 + TS_LOS_interval/2

	plt.figure(figsize=(8,6))
	global_plasma_cecntre = (np.abs(np.cumsum(np.nansum(raw_data[:,:,60:],axis=(0,2)))/np.nansum(raw_data[:,:,60:int(target_location//mm_per_pixel)])-0.5).argmin()+0.5) * mm_per_pixel
	plt.imshow(average,'rainbow',origin='lower',extent=[0-target_location,np.shape(raw_data)[1]*mm_per_pixel-target_location,0-global_plasma_cecntre,np.shape(raw_data)[2]*mm_per_pixel-global_plasma_cecntre])
	plt.axvline(x=0,color='k',linestyle='--',label='target')

	plt.axvline(x=OES_location_left-target_location,color='k',label='OES/TS location')
	plt.axvline(x=OES_location_right-target_location,color='k')

	# plt.imshow(cleaned[select],'rainbow',origin='lower')
	if overexposed_local:
		plt.colorbar().set_label('average counts [au] (overexposed)')
	else:
		plt.colorbar().set_label('average counts [au]')
	plt.title(full_folder+filename+'\nmagnetic_field %.3gT,steady state pressure %.3gPa,target/OES distance %.3gmm,no ELM pulse\n average profile' %(1.2,target_chamber_pressure[i_j],-(z_position[i_j]+78.8)),fontsize=12)
	plt.legend(loc='upper left', fontsize='x-small')
	plt.xlabel('longitudinal position [mm]')
	plt.ylabel('radial position [mm]')
	plt.grid()
	# plt.pause(0.01)
	plt.savefig(full_folder +'/fast_camera'+'_' + filename +'.eps', bbox_inches='tight')
	plt.close('all')

plt.figure(figsize=(13,8))
mm_per_pixel = 15/44	# mm/#
plt.plot([],[],'w',label='Pressure')
for i_j,j in enumerate(file_id):
	# if not(j in [9325,9329,9333,9337]):
	# if not(j in [9325,9329,9337]):
	if not(j in [9325,9331,9337]):
		continue
	full_folder = '/home/ffederic/work/Collaboratory/test/experimental_data/fast camera/DetachmentMovieRawCine/'
	type = '.cine'
	filename = str(j) + type
	raw_images, setup, bpp = read_frames(full_folder+filename)#,start_frame=1,count=1)
	# header = read_header('/home/ffederic/work/Collaboratory/test/experimental_data/2019-05-03/01/fast_camera/02_1.cine')
	framerate = setup.FrameRate
	raw_data = []
	for image in raw_images:
		raw_data.append(image)
	raw_data = np.array(raw_data).astype('int')
	raw_data = raw_data[np.max(raw_data,axis=(1,2))>0]

	average = np.median(raw_data,axis=0)

	# counts are normalised for a 1ms exposure
	average = average/(setup.ShutterNs/1000)

	target_location = 230*mm_per_pixel +  -(z_position[i_j]+78.8)-5  - 31.2	# this is for a tungsten dummy 1 target, as I think it was for theis set of experiment
	LOS_size = 1	# mm diameter
	LOS_number = 40
	LOS_interval = 1.06478505470992	# 10/02/2020 from	Calculate magnification_FF.xlsx
	LOS_full_size = (LOS_number-1)*LOS_interval + LOS_size
	TS_size = [-4.149230769230769056e+01, 4.416923076923076508e+01]
	TS_LOS_full_size = TS_size[1]-TS_size[0]
	TS_LOS_interval = (TS_size[1]-TS_size[0])/65
	OES_location = 230*mm_per_pixel - 31.2
	OES_location_left = 230*mm_per_pixel - 31.2 - LOS_size/2
	OES_location_right = 230*mm_per_pixel - 31.2 + LOS_size/2
	OES_location_left_pixel = int(round(OES_location_left/mm_per_pixel))
	OES_location_right_pixel = int(round(OES_location_right/mm_per_pixel))
	TS_location_left = 230*mm_per_pixel - 31.2 - TS_LOS_interval/2
	TS_location_right = 230*mm_per_pixel - 31.2 + TS_LOS_interval/2

	plt.plot(np.arange(np.shape(average)[0])*mm_per_pixel-target_location,np.mean(average[110:150],axis=0),label='%.3gPa' %(target_chamber_pressure[i_j]))
plt.title('steady state samples average profile\nmagnetic_field %.3gT,target/OES distance %.3gmm,no ELM pulse\n' %(1.2,-(z_position[i_j]+78.8)),fontsize=12)
# plt.plot([195*mm_per_pixel]*2,[0,np.mean(average,axis=0).max()],'k-.',label='target')
# plt.plot([195*mm_per_pixel]*2,[0,0.5],'k-.',label='target')
plt.axvline(x=0,color='k',linestyle='-.',label='target')
plt.axvline(x=OES_location_left-target_location,color='k',label='OES/TS location')
plt.axvline(x=OES_location_right-target_location,color='k')
plt.legend(loc='best', fontsize='x-small')
plt.xlabel('longitudinal position [mm]')
plt.ylabel('Average brightness [au]')
plt.ylim(top=6e-1,bottom=1e-2)
plt.xlim(left=0-target_location,right=70-target_location)
plt.grid()
plt.semilogy()
# plt.pause(0.01)
plt.savefig(full_folder +'/SS_scan4' +'.png', bbox_inches='tight')
plt.close('all')




# ani = coleval.movie_from_data(np.array([cleaned]), 5000, 0.01/1000,'horizontal coord [pixels]','vertical coord [pixels]','Intersity [au]')

plt.figure()
plt.plot(cleaned[22,:,50])
plt.pause(0.01)

plt.figure()
plt.imshow(cleaned[:,:,50].T)
plt.colorbar()
plt.pause(0.01)

plt.figure()
plt.imshow(cleaned[:,:,100].T)
plt.colorbar()
plt.pause(0.01)

plt.figure()
plt.imshow(cleaned[:,:,150].T)
plt.colorbar()
plt.pause(0.01)

plt.figure()
plt.imshow(cleaned[:,:,200].T)
plt.colorbar()
plt.pause(0.01)


plt.figure()
plt.plot(np.sum(cleaned[:,:,50],axis=1),label='50')
plt.plot(np.sum(cleaned[:,:,100],axis=1),label='100')
plt.plot(np.sum(cleaned[:,:,150],axis=1),label='150')
plt.legend(loc='best')
plt.pause(0.01)
