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




merge_ID_target_multipulse = np.flip([66,67,68,69,70,71,72,73,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92, 93, 94,95,96,97,98,99],axis=0)
# merge_ID_target_multipulse = np.flip([66,67,68,69,70,71,72,73,75,76,77,78,79],axis=0)



for merge_ID_target in merge_ID_target_multipulse:
	print('Working on ' + str(merge_ID_target))
	overexposed = False
	overexposed_mask = np.zeros((256,256), dtype=bool)

	path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)

	all_j=find_index_of_file(merge_ID_target,df_settings,df_log,only_OES=True)

	full_saved_file_dict = dict([])
	full_saved_file_dict['all_j'] = []

	good_records = 0
	all_averaged_profile = []
	for j in all_j:
		(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
		(file_index,target_OES_distance,magnetic_field,SS_pressure,pulse_voltage) = df_log.loc[j,['fast_camera_trace','T_axial','B','p_n [Pa]','Vc']]
		if np.isnan(file_index):
			continue
		full_folder = '/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/'+"{0:0=2d}".format(sequence)+'/fast_camera/'
		type = '.cine'
		filenames = np.array(all_file_names(full_folder, type))
		# select = [name[:2]==str(file_index) for name in filenames]
		select = [name[:2]=="{0:0=2d}".format(int(file_index)) for name in filenames]
		filenames = filenames[select]

		full_saved_file_dict['all_j'] = full_saved_file_dict['all_j']+[str(j)]
		full_saved_file_dict[str(j)] = dict([])
		full_saved_file_dict[str(j)]['filenames'] = filenames

		# median_50 = []
		# median_100 = []
		# median_166 = []

		for i_filename,filename in enumerate(filenames):
			raw_images, setup, bpp = read_frames(full_folder+filename)#,start_frame=1,count=1)
			# header = read_header('/home/ffederic/work/Collaboratory/test/experimental_data/2019-05-03/01/fast_camera/02_1.cine')
			framerate = setup.FrameRate
			integration_time = setup.ShutterNs/1000	# ms
			saturation_counts = setup.Conv8Max	# counts
			raw_data = []
			for image in raw_images:
				raw_data.append(image)
			raw_data = np.array(raw_data).astype('int')
			raw_data = raw_data[np.max(raw_data,axis=(1,2))>0]

			overexposed_local = False
			if raw_data.max()>=saturation_counts:
				overexposed_local = True
				overexposed = True
				overexposed_mask[np.max(raw_data,axis=0)>=saturation_counts]=True

			full_saved_file_dict[str(j)][filename] = dict([])
			full_saved_file_dict[str(j)][filename]['framerate'] = framerate
			full_saved_file_dict[str(j)][filename]['integration_time'] = integration_time	# ms
			full_saved_file_dict[str(j)][filename]['saturation_counts'] = saturation_counts	# counts
			full_saved_file_dict[str(j)][filename]['raw_data'] = raw_data

			# counts are normalised for a 1ms exposure
			raw_data = raw_data/(setup.ShutterNs/1000)

			if len(all_averaged_profile)==0:
				all_averaged_profile = [raw_data.tolist()]
			else:
				all_averaged_profile.append(raw_data)
			good_records +=1

			averaged_profile = np.mean(raw_data,axis=0)
			full_saved_file_dict[str(j)][filename]['averaged_profile'] = averaged_profile	# counts are normalised for a 1ms exposure


			temp = np.mean(averaged_profile[:,47:53],axis=1)
			axis_pixel = max(1,temp.argmax())
			axis_loc_50 = axis_pixel*15/44
			# top_loc_50 = ((np.mean(averaged_profile[axis_pixel:,47:53],axis=1)>5).argmin() + axis_pixel)*15/44
			# bottom_loc_50 = (np.mean(averaged_profile[:axis_pixel,47:53],axis=1)<5).argmin()*15/44
			top_loc_50 = (np.abs(temp[axis_pixel:]-temp.argmax()/10).argmin()+axis_pixel)*15/44
			bottom_loc_50 = (np.abs(temp[:axis_pixel]-temp.argmax()/10).argmin())*15/44
			plasma_diameter_50 = (top_loc_50-bottom_loc_50)
			temp = np.mean(averaged_profile[:,97:103],axis=1)
			axis_pixel = max(1,temp.argmax())
			axis_loc_100 = axis_pixel*15/44
			# top_loc_100 = ((np.mean(averaged_profile[axis_pixel:,97:103],axis=1)>5).argmin() + axis_pixel)*15/44
			# bottom_loc_100 = (np.mean(averaged_profile[:axis_pixel,97:103],axis=1)<5).argmin()*15/44
			top_loc_100 = (np.abs(temp[axis_pixel:]-temp.argmax()/10).argmin()+axis_pixel)*15/44
			bottom_loc_100 = (np.abs(temp[:axis_pixel]-temp.argmax()/10).argmin())*15/44
			plasma_diameter_100 = (top_loc_100-bottom_loc_100)
			temp = np.mean(averaged_profile[:,158:172],axis=1)
			axis_pixel = max(1,temp.argmax())
			axis_loc_166 = axis_pixel*15/44
			# top_loc_166 = ((np.mean(averaged_profile[axis_pixel:,158:172],axis=1)>5).argmin() + axis_pixel)*15/44
			# bottom_loc_166 = (np.mean(averaged_profile[:axis_pixel,158:172],axis=1)<5).argmin()*15/44
			top_loc_166 = (np.abs(temp[axis_pixel:]-temp.argmax()/10).argmin()+axis_pixel)*15/44
			bottom_loc_166 = (np.abs(temp[:axis_pixel]-temp.argmax()/10).argmin())*15/44
			plasma_diameter_166 = (top_loc_166-bottom_loc_166)
			temp = np.mean(averaged_profile[:,214:220],axis=1)
			axis_pixel = max(1,temp.argmax())
			axis_loc_target = axis_pixel*15/44
			# top_loc_target = ((np.mean(averaged_profile[axis_pixel:,214:220],axis=1)>5).argmin() + axis_pixel)*15/44
			# bottom_loc_target = (np.mean(averaged_profile[:axis_pixel,214:220],axis=1)<5).argmin()*15/44
			top_loc_target = (np.abs(temp[axis_pixel:]-temp.argmax()/10).argmin()+axis_pixel)*15/44
			bottom_loc_target = (np.abs(temp[:axis_pixel]-temp.argmax()/10).argmin())*15/44
			plasma_diameter_target = (top_loc_target-bottom_loc_target)

			plt.figure(figsize=(8,6))
			plt.imshow(averaged_profile,'rainbow',origin='lower',extent=[0,np.shape(raw_data)[1]*15/44,0,np.shape(raw_data)[2]*15/44])
			# plt.imshow(cleaned[select],'rainbow',origin='lower')
			if overexposed_local:
				plt.colorbar().set_label('average counts [au] (overexposed)')
			else:
				plt.colorbar().set_label('average counts [au]')
			plt.plot([47*15/44,47*15/44],[60*15/44,200*15/44],'--k',label='%.3gmm from OES, ' %((166-50)*15/44) +r'$\phi$'+ ' %.3gmm' %(plasma_diameter_50))
			plt.plot([(53-1)*15/44,(53-1)*15/44],[60*15/44,200*15/44],'--k')
			plt.plot([47*15/44,(53-1)*15/44],[axis_loc_50]*2,'-.r',linewidth=1)
			plt.plot([47*15/44,(53-1)*15/44],[top_loc_50]*2,'r',linewidth=4)
			plt.plot([47*15/44,(53-1)*15/44],[bottom_loc_50]*2,'r',linewidth=4)
			plt.plot([97*15/44,97*15/44],[60*15/44,200*15/44],':k',label='%.3gmm from OES, ' %((166-100)*15/44) +r'$\phi$'+ ' %.3gmm' %(plasma_diameter_100))
			plt.plot([(103-1)*15/44,(103-1)*15/44],[60*15/44,200*15/44],':k')
			plt.plot([97*15/44,(103-1)*15/44],[axis_loc_100]*2,'-.r',linewidth=1)
			plt.plot([97*15/44,(103-1)*15/44],[top_loc_100]*2,'r',linewidth=4)
			plt.plot([97*15/44,(103-1)*15/44],[bottom_loc_100]*2,'r',linewidth=4)
			plt.plot([158*15/44,158*15/44],[60*15/44,200*15/44],'k',label='OES location, ' +r'$\phi$'+ ' %.3gmm' %(plasma_diameter_166))
			plt.plot([(172-1)*15/44,(172-1)*15/44],[60*15/44,200*15/44],'k')
			plt.plot([158*15/44,(172-1)*15/44],[axis_loc_166]*2,'-.r',linewidth=1)
			plt.plot([158*15/44,(172-1)*15/44],[top_loc_166]*2,'r',linewidth=4)
			plt.plot([158*15/44,(172-1)*15/44],[bottom_loc_166]*2,'r',linewidth=4)
			plt.plot([214*15/44,214*15/44],[60*15/44,200*15/44],'-.k',label='target proximity, ' +r'$\phi$'+ ' %.3gmm' %(plasma_diameter_target))
			plt.plot([(220-1)*15/44,(220-1)*15/44],[60*15/44,200*15/44],'-.k')
			plt.plot([214*15/44,(220-1)*15/44],[axis_loc_target]*2,'-.r',linewidth=1)
			plt.plot([214*15/44,(220-1)*15/44],[top_loc_target]*2,'r',linewidth=4)
			plt.plot([214*15/44,(220-1)*15/44],[bottom_loc_target]*2,'r',linewidth=4)
			plt.title(full_folder+filename+'\nmagnetic_field %.3gT,steady state pressure %.3gPa,target/OES distance %.3gmm,ELM pulse voltage %.3gV\n average profile' %(magnetic_field,SS_pressure,target_OES_distance,pulse_voltage),fontsize=12)
			plt.legend(loc='best', fontsize='x-small')
			plt.xlabel('longitudinal position [mm]')
			plt.ylabel('radial position [mm]')
			plt.grid()
			# plt.pause(0.01)
			plt.savefig(path_where_to_save_everything +'/fast_camera_merge_'+str(merge_ID_target)+'_item_'+str(j)+'_' + filename +'.eps', bbox_inches='tight')
			plt.close('all')

			if overexposed_local:
				averaged_profile[np.max(raw_data,axis=0)>=saturation_counts] = np.nan
				length = 351+target_OES_distance	# mm distance skimmer to OES/TS + OES/TS to target
				total_volume = ((np.max([plasma_diameter_50,plasma_diameter_100,plasma_diameter_166,plasma_diameter_target,plasma_diameter_target])/2)**2)*np.pi*length
				overexposed_volume_max = np.max([plasma_diameter_50,plasma_diameter_100,plasma_diameter_166,plasma_diameter_target,plasma_diameter_target])*np.sum((np.max(raw_data,axis=0)>=saturation_counts))*((15/44)**2)
				fraction_overexposed_volume = overexposed_volume_max/total_volume * 100


			plt.figure(figsize=(8,6))
			plt.imshow(averaged_profile,'rainbow',origin='lower',extent=[0,np.shape(raw_data)[1]*15/44,0,np.shape(raw_data)[2]*15/44])
			# plt.imshow(cleaned[select],'rainbow',origin='lower')
			if overexposed_local:
				plt.colorbar().set_label('average counts [au] (%.3g%% volume overexposed)' %(fraction_overexposed_volume),fontsize=13)
			else:
				plt.colorbar().set_label('average counts [au]')
			if plasma_diameter_50>15/44*2:
				plt.plot([47*15/44,47*15/44],[60*15/44,200*15/44],'--k',label='%.3gmm from OES, ' %((166-50)*15/44) +r'$\phi$'+ ' %.3gmm' %(plasma_diameter_50))
				plt.plot([(53-1)*15/44,(53-1)*15/44],[60*15/44,200*15/44],'--k')
				plt.plot([47*15/44,(53-1)*15/44],[axis_loc_50]*2,'-.r',linewidth=1)
				plt.plot([47*15/44,(53-1)*15/44],[top_loc_50]*2,'r',linewidth=4)
				plt.plot([47*15/44,(53-1)*15/44],[bottom_loc_50]*2,'r',linewidth=4)
			if plasma_diameter_100>15/44*2:
				plt.plot([97*15/44,97*15/44],[60*15/44,200*15/44],':k',label='%.3gmm from OES, ' %((166-100)*15/44) +r'$\phi$'+ ' %.3gmm' %(plasma_diameter_100))
				plt.plot([(103-1)*15/44,(103-1)*15/44],[60*15/44,200*15/44],':k')
				plt.plot([97*15/44,(103-1)*15/44],[axis_loc_100]*2,'-.r',linewidth=1)
				plt.plot([97*15/44,(103-1)*15/44],[top_loc_100]*2,'r',linewidth=4)
				plt.plot([97*15/44,(103-1)*15/44],[bottom_loc_100]*2,'r',linewidth=4)
			if plasma_diameter_166>15/44*2:
				plt.plot([158*15/44,158*15/44],[60*15/44,200*15/44],'k',label='OES location, ' +r'$\phi$'+ ' %.3gmm' %(plasma_diameter_166))
				plt.plot([(172-1)*15/44,(172-1)*15/44],[60*15/44,200*15/44],'k')
				plt.plot([158*15/44,(172-1)*15/44],[axis_loc_166]*2,'-.r',linewidth=1)
				plt.plot([158*15/44,(172-1)*15/44],[top_loc_166]*2,'r',linewidth=4)
				plt.plot([158*15/44,(172-1)*15/44],[bottom_loc_166]*2,'r',linewidth=4)
			if plasma_diameter_target>15/44*2:
				plt.plot([214*15/44,214*15/44],[60*15/44,200*15/44],'-.k',label='target proximity, ' +r'$\phi$'+ ' %.3gmm' %(plasma_diameter_target))
				plt.plot([(220-1)*15/44,(220-1)*15/44],[60*15/44,200*15/44],'-.k')
				plt.plot([214*15/44,(220-1)*15/44],[axis_loc_target]*2,'-.r',linewidth=1)
				plt.plot([214*15/44,(220-1)*15/44],[top_loc_target]*2,'r',linewidth=4)
				plt.plot([214*15/44,(220-1)*15/44],[bottom_loc_target]*2,'r',linewidth=4)
			plt.title(full_folder+filename+'\nmagnetic_field %.3gT,steady state pressure %.3gPa,target/OES distance %.3gmm,ELM pulse voltage %.3gV\n average profile' %(magnetic_field,SS_pressure,target_OES_distance,pulse_voltage),fontsize=12)
			plt.legend(loc='best', fontsize='x-small')
			plt.xlabel('longitudinal position [mm]')
			plt.ylabel('radial position [mm]')
			plt.grid()
			# plt.pause(0.01)
			plt.savefig(path_where_to_save_everything +'/fast_camera_merge_'+str(merge_ID_target)+'_item_'+str(j)+'_' + filename +'2.eps', bbox_inches='tight')
			plt.close('all')

			ani = coleval.movie_from_data(np.array([raw_data]), framerate, integration_time,'horizontal coord [pixels]','vertical coord [pixels]','Intersity [au]')
			ani.save(path_where_to_save_everything +'/fast_camera_merge_'+str(merge_ID_target)+'_item_'+str(j)+'_' + filename + '.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
			# ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_original.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
			plt.close()

			video_length = len(raw_data)
			downsample = int(np.ceil(video_length/15))
			ani = coleval.movie_from_data(np.array([raw_data[::downsample][:15]]), framerate/downsample, integration_time,'horizontal coord [pixels]','vertical coord [pixels]','Intersity [au]')
			ani.save(path_where_to_save_everything +'/fast_camera_merge_'+str(merge_ID_target)+'_item_'+str(j)+'_' + filename + '_2.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
			# ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_original.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
			plt.close()

			temp = np.nanmean(raw_data,axis=1)
			temp[np.nanmax(raw_data,axis=1)>=saturation_counts] = np.nan
			plt.figure(figsize=(8,6))
			plt.imshow(temp,'rainbow',origin='lower',extent=[0,np.shape(raw_data)[2]*15/44,0,len(raw_data)/framerate*1e3],aspect='auto',norm=LogNorm())
			# plt.imshow(cleaned[select],'rainbow',origin='lower')
			if overexposed_local:
				plt.colorbar().set_label('average counts [au] (%.3g%% volume overexposed)' %(fraction_overexposed_volume),fontsize=13)
			else:
				plt.colorbar().set_label('average counts [au]')
			plt.plot([158*15/44,158*15/44],[0,len(raw_data)/framerate*1e3],'b',label='OES location')
			plt.plot([(172-1)*15/44,(172-1)*15/44],[0,len(raw_data)/framerate*1e3],'b')
			plt.title(full_folder+filename+'\nmagnetic_field %.3gT,steady state pressure %.3gPa,target/OES distance %.3gmm,ELM pulse voltage %.3gV\n temporal evolution of the radially averaged profile' %(magnetic_field,SS_pressure,target_OES_distance,pulse_voltage),fontsize=12)
			plt.legend(loc='best', fontsize='x-small')
			plt.xlabel('longitudinal position [mm]')
			plt.ylabel('time [ms]')
			plt.grid()
			# plt.pause(0.01)
			plt.savefig(path_where_to_save_everything +'/fast_camera_merge_'+str(merge_ID_target)+'_item_'+str(j)+'_' + filename +'3.eps', bbox_inches='tight')
			plt.close('all')


	temp = np.min([np.shape(value)[0] for value in all_averaged_profile])
	all_averaged_profile = [value[:temp] for value in all_averaged_profile]
	full_saved_file_dict['averaged_profile'] = np.mean(all_averaged_profile,axis=(0,1))	# counts are normalised for a 1ms exposure
	fast_camera_record_duration = np.shape(all_averaged_profile)[1] / framerate	# [s]
	full_saved_file_dict['record_duration'] = fast_camera_record_duration
	results_summary = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/results_summary.csv',index_col=0)
	results_summary.loc[merge_ID_target,['fast_camera_record_duration']]=fast_camera_record_duration
	results_summary.to_csv(path_or_buf='/home/ffederic/work/Collaboratory/test/experimental_data/results_summary.csv')

	averaged_profile = np.mean(all_averaged_profile,axis=(0,1))

	temp = np.mean(averaged_profile[:,47:53],axis=1)
	axis_pixel = max(1,temp.argmax())
	axis_loc_50 = axis_pixel*15/44
	# top_loc_50 = ((np.mean(averaged_profile[axis_pixel:,47:53],axis=1)>5).argmin() + axis_pixel)*15/44
	# bottom_loc_50 = (np.mean(averaged_profile[:axis_pixel,47:53],axis=1)<5).argmin()*15/44
	top_loc_50 = (np.abs(temp[axis_pixel:]-temp.argmax()/10).argmin()+axis_pixel)*15/44
	bottom_loc_50 = (np.abs(temp[:axis_pixel]-temp.argmax()/10).argmin())*15/44
	plasma_diameter_50 = (top_loc_50-bottom_loc_50)
	temp = np.mean(averaged_profile[:,97:103],axis=1)
	axis_pixel = max(1,temp.argmax())
	axis_loc_100 = axis_pixel*15/44
	# top_loc_100 = ((np.mean(averaged_profile[axis_pixel:,97:103],axis=1)>5).argmin() + axis_pixel)*15/44
	# bottom_loc_100 = (np.mean(averaged_profile[:axis_pixel,97:103],axis=1)<5).argmin()*15/44
	top_loc_100 = (np.abs(temp[axis_pixel:]-temp.argmax()/10).argmin()+axis_pixel)*15/44
	bottom_loc_100 = (np.abs(temp[:axis_pixel]-temp.argmax()/10).argmin())*15/44
	plasma_diameter_100 = (top_loc_100-bottom_loc_100)
	temp = np.mean(averaged_profile[:,158:172],axis=1)
	axis_pixel = max(1,temp.argmax())
	axis_loc_166 = axis_pixel*15/44
	# top_loc_166 = ((np.mean(averaged_profile[axis_pixel:,158:172],axis=1)>5).argmin() + axis_pixel)*15/44
	# bottom_loc_166 = (np.mean(averaged_profile[:axis_pixel,158:172],axis=1)<5).argmin()*15/44
	top_loc_166 = (np.abs(temp[axis_pixel:]-temp.argmax()/10).argmin()+axis_pixel)*15/44
	bottom_loc_166 = (np.abs(temp[:axis_pixel]-temp.argmax()/10).argmin())*15/44
	plasma_diameter_166 = (top_loc_166-bottom_loc_166)
	temp = np.mean(averaged_profile[:,214:220],axis=1)
	axis_pixel = max(1,temp.argmax())
	axis_loc_target = axis_pixel*15/44
	# top_loc_target = ((np.mean(averaged_profile[axis_pixel:,214:220],axis=1)>5).argmin() + axis_pixel)*15/44
	# bottom_loc_target = (np.mean(averaged_profile[:axis_pixel,214:220],axis=1)<5).argmin()*15/44
	top_loc_target = (np.abs(temp[axis_pixel:]-temp.argmax()/10).argmin()+axis_pixel)*15/44
	bottom_loc_target = (np.abs(temp[:axis_pixel]-temp.argmax()/10).argmin())*15/44
	plasma_diameter_target = (top_loc_target-bottom_loc_target)

	plt.figure(figsize=(8,6))
	plt.imshow(averaged_profile,'rainbow',origin='lower',extent=[0,np.shape(raw_data)[1]*15/44,0,np.shape(raw_data)[2]*15/44],vmax=4096/2)
	# plt.imshow(cleaned[select],'rainbow',origin='lower')
	if overexposed:
		plt.colorbar().set_label('average counts [au] (overexposed)')
	else:
		plt.colorbar().set_label('average counts [au]')
	plt.plot([47*15/44,47*15/44],[60*15/44,200*15/44],'--k',label='%.3gmm from OES, ' %((166-50)*15/44) +r'$\phi$'+ ' %.3gmm' %(plasma_diameter_50))
	plt.plot([(53-1)*15/44,(53-1)*15/44],[60*15/44,200*15/44],'--k')
	plt.plot([47*15/44,(53-1)*15/44],[axis_loc_50]*2,'-.r',linewidth=1)
	plt.plot([47*15/44,(53-1)*15/44],[top_loc_50]*2,'r',linewidth=4)
	plt.plot([47*15/44,(53-1)*15/44],[bottom_loc_50]*2,'r',linewidth=4)
	plt.plot([97*15/44,97*15/44],[60*15/44,200*15/44],':k',label='%.3gmm from OES, ' %((166-100)*15/44) +r'$\phi$'+ ' %.3gmm' %(plasma_diameter_100))
	plt.plot([(103-1)*15/44,(103-1)*15/44],[60*15/44,200*15/44],':k')
	plt.plot([97*15/44,(103-1)*15/44],[axis_loc_100]*2,'-.r',linewidth=1)
	plt.plot([97*15/44,(103-1)*15/44],[top_loc_100]*2,'r',linewidth=4)
	plt.plot([97*15/44,(103-1)*15/44],[bottom_loc_100]*2,'r',linewidth=4)
	plt.plot([158*15/44,158*15/44],[60*15/44,200*15/44],'k',label='OES location, ' +r'$\phi$'+ ' %.3gmm' %(plasma_diameter_166))
	plt.plot([(172-1)*15/44,(172-1)*15/44],[60*15/44,200*15/44],'k')
	plt.plot([158*15/44,(172-1)*15/44],[axis_loc_166]*2,'-.r',linewidth=1)
	plt.plot([158*15/44,(172-1)*15/44],[top_loc_166]*2,'r',linewidth=4)
	plt.plot([158*15/44,(172-1)*15/44],[bottom_loc_166]*2,'r',linewidth=4)
	plt.plot([214*15/44,214*15/44],[60*15/44,200*15/44],'-.k',label='target proximity, ' +r'$\phi$'+ ' %.3gmm' %(plasma_diameter_target))
	plt.plot([(220-1)*15/44,(220-1)*15/44],[60*15/44,200*15/44],'-.k')
	plt.plot([214*15/44,(220-1)*15/44],[axis_loc_target]*2,'-.r',linewidth=1)
	plt.plot([214*15/44,(220-1)*15/44],[top_loc_target]*2,'r',linewidth=4)
	plt.plot([214*15/44,(220-1)*15/44],[bottom_loc_target]*2,'r',linewidth=4)
	plt.title('magnetic_field %.3gT,steady state pressure %.3gPa,target/OES distance %.3gmm,ELM pulse voltage %.3gV\n average profile' %(magnetic_field,SS_pressure,target_OES_distance,pulse_voltage),fontsize=12)
	plt.legend(loc='best', fontsize='x-small')
	plt.xlabel('longitudinal position [mm]')
	plt.ylabel('radial position [mm]')
	plt.grid()
	# plt.pause(0.01)
	plt.yscale('log')
	plt.savefig(path_where_to_save_everything +'/fast_camera_merge_'+str(merge_ID_target)+'_average'+'.eps', bbox_inches='tight')
	plt.close('all')

	if overexposed:
		full_saved_file_dict['overexposed_mask'] = overexposed_mask
		averaged_profile[overexposed_mask] = np.nan
		length = 351+target_OES_distance	# mm distance skimmer to OES/TS + OES/TS to target
		total_volume = ((np.max([plasma_diameter_50,plasma_diameter_100,plasma_diameter_166,plasma_diameter_target,plasma_diameter_target])/2)**2)*np.pi*length
		overexposed_volume_max = np.max([plasma_diameter_50,plasma_diameter_100,plasma_diameter_166,plasma_diameter_target,plasma_diameter_target])*np.sum(overexposed_mask)*((15/44)**2)
		fraction_overexposed_volume = overexposed_volume_max/total_volume * 100
	else:
		full_saved_file_dict['overexposed_mask'] = np.zeros_like(averaged_profile).astype(bool)

	plt.figure(figsize=(8,6))
	plt.imshow(averaged_profile,'rainbow',origin='lower',extent=[0,np.shape(raw_data)[1]*15/44,0,np.shape(raw_data)[2]*15/44])
	# plt.imshow(cleaned[select],'rainbow',origin='lower')
	if overexposed:
		plt.colorbar().set_label('average counts [au] (%.3g%% volume overexposed)' %(fraction_overexposed_volume),fontsize=13)
	else:
		plt.colorbar().set_label('average counts [au]')
	if plasma_diameter_50>15/44*2:
		plt.plot([47*15/44,47*15/44],[60*15/44,200*15/44],'--k',label='%.3gmm from OES, ' %((166-50)*15/44) +r'$\phi$'+ ' %.3gmm' %(plasma_diameter_50))
		plt.plot([(53-1)*15/44,(53-1)*15/44],[60*15/44,200*15/44],'--k')
		plt.plot([47*15/44,(53-1)*15/44],[axis_loc_50]*2,'-.r',linewidth=1)
		plt.plot([47*15/44,(53-1)*15/44],[top_loc_50]*2,'r',linewidth=4)
		plt.plot([47*15/44,(53-1)*15/44],[bottom_loc_50]*2,'r',linewidth=4)
	if plasma_diameter_100>15/44*2:
		plt.plot([97*15/44,97*15/44],[60*15/44,200*15/44],':k',label='%.3gmm from OES, ' %((166-100)*15/44) +r'$\phi$'+ ' %.3gmm' %(plasma_diameter_100))
		plt.plot([(103-1)*15/44,(103-1)*15/44],[60*15/44,200*15/44],':k')
		plt.plot([97*15/44,(103-1)*15/44],[axis_loc_100]*2,'-.r',linewidth=1)
		plt.plot([97*15/44,(103-1)*15/44],[top_loc_100]*2,'r',linewidth=4)
		plt.plot([97*15/44,(103-1)*15/44],[bottom_loc_100]*2,'r',linewidth=4)
	if plasma_diameter_166>15/44*2:
		plt.plot([158*15/44,158*15/44],[60*15/44,200*15/44],'k',label='OES location, ' +r'$\phi$'+ ' %.3gmm' %(plasma_diameter_166))
		plt.plot([(172-1)*15/44,(172-1)*15/44],[60*15/44,200*15/44],'k')
		plt.plot([158*15/44,(172-1)*15/44],[axis_loc_166]*2,'-.r',linewidth=1)
		plt.plot([158*15/44,(172-1)*15/44],[top_loc_166]*2,'r',linewidth=4)
		plt.plot([158*15/44,(172-1)*15/44],[bottom_loc_166]*2,'r',linewidth=4)
	if plasma_diameter_target>15/44*2:
		plt.plot([214*15/44,214*15/44],[60*15/44,200*15/44],'-.k',label='target proximity, ' +r'$\phi$'+ ' %.3gmm' %(plasma_diameter_target))
		plt.plot([(220-1)*15/44,(220-1)*15/44],[60*15/44,200*15/44],'-.k')
		plt.plot([214*15/44,(220-1)*15/44],[axis_loc_target]*2,'-.r',linewidth=1)
		plt.plot([214*15/44,(220-1)*15/44],[top_loc_target]*2,'r',linewidth=4)
		plt.plot([214*15/44,(220-1)*15/44],[bottom_loc_target]*2,'r',linewidth=4)
	plt.title('magnetic_field %.3gT,steady state pressure %.3gPa,target/OES distance %.3gmm,ELM pulse voltage %.3gV\n average profile' %(magnetic_field,SS_pressure,target_OES_distance,pulse_voltage),fontsize=12)
	plt.legend(loc='best', fontsize='x-small')
	plt.xlabel('longitudinal position [mm]')
	plt.ylabel('radial position [mm]')
	plt.grid()
	# plt.pause(0.01)
	plt.savefig(
	path_where_to_save_everything +'/fast_camera_merge_'+str(merge_ID_target)+'_average'+'2.eps', bbox_inches='tight')
	plt.close('all')

	plt.figure(figsize=(10,3))
	plt.imshow(averaged_profile,'rainbow',origin='lower',extent=[0,np.shape(raw_data)[1]*15/44,0,np.shape(raw_data)[2]*15/44])
	# plt.imshow(cleaned[select],'rainbow',origin='lower')
	if overexposed:
		plt.colorbar().set_label('average counts [au]\n%.3g%% vol overexposed' %(fraction_overexposed_volume),fontsize=13)
	else:
		plt.colorbar().set_label('average counts [au]')
	plt.plot([158*15/44,158*15/44],[60*15/44,200*15/44],'k',label='OES location')
	plt.plot([(172-1)*15/44,(172-1)*15/44],[60*15/44,200*15/44],'k')
	plt.plot([230*15/44]*2,[60*15/44,200*15/44],'k-.',label='target')
	plt.ylim(bottom=30,top=65)
	plt.title('magnetic_field %.3gT,steady state pressure %.3gPa,target/OES distance %.3gmm,ELM pulse voltage %.3gV\n average profile' %(magnetic_field,SS_pressure,target_OES_distance,pulse_voltage),fontsize=12)
	plt.legend(loc='best', fontsize='x-small')
	plt.xlabel('longitudinal position [mm]')
	plt.ylabel('radial position [mm]')
	# plt.yticks(fontfamily=ticker.FormatStrFormatter('% 1.0f'))
	plt.grid()
	# plt.pause(0.01)
	plt.savefig(path_where_to_save_everything +'/fast_camera_merge_'+str(merge_ID_target)+'_average'+'3.eps', bbox_inches='tight')
	plt.close('all')

	plt.figure(figsize=(10,5))
	plt.plot(np.arange(np.shape(raw_data)[1])*15/44,np.mean(averaged_profile[:,158:172],axis=1))
	plt.title('magnetic_field %.3gT,steady state pressure %.3gPa,target/OES distance %.3gmm,ELM pulse voltage %.3gV\n radial average brightness at OES location (scaled to 1ms int time)' %(magnetic_field,SS_pressure,target_OES_distance,pulse_voltage),fontsize=12)
	plt.ylabel('counts [au]')
	plt.xlabel('radial position [mm]')
	# plt.yticks(fontfamily=ticker.FormatStrFormatter('% 1.0f'))
	plt.grid()
	# plt.pause(0.01)
	plt.savefig(path_where_to_save_everything +'/fast_camera_merge_'+str(merge_ID_target)+'_average'+'4.eps', bbox_inches='tight')
	plt.close('all')
	full_saved_file_dict['radial_average_brightness_1ms_int_time'] = np.mean(averaged_profile[:,158:172],axis=1)	# counts are normalised for a 1ms exposure

	temp = np.nanmean(all_averaged_profile,axis=(0,2))
	temp[np.nanmax(all_averaged_profile,axis=(0,2))>=saturation_counts] = np.nan
	plt.figure(figsize=(8,6))
	plt.imshow(temp,'rainbow',origin='lower',extent=[0,np.shape(raw_data)[2]*15/44,0,len(raw_data)/framerate*1e3],aspect='auto',norm=LogNorm(),vmax=800,vmin=1)
	# plt.imshow(cleaned[select],'rainbow',origin='lower')
	if overexposed_local:
		plt.colorbar().set_label('average counts [au] (%.3g%% volume overexposed)' %(fraction_overexposed_volume),fontsize=13)
	else:
		plt.colorbar().set_label('average counts [au]')
	plt.plot([158*15/44,158*15/44],[0,len(raw_data)/framerate*1e3],'b',label='OES location')
	plt.plot([(172-1)*15/44,(172-1)*15/44],[0,len(raw_data)/framerate*1e3],'b')
	plt.title('magnetic_field %.3gT,steady state pressure %.3gPa,target/OES distance %.3gmm,ELM pulse voltage %.3gV\n temporal evolution of the radially averaged profile' %(magnetic_field,SS_pressure,target_OES_distance,pulse_voltage),fontsize=12)
	plt.legend(loc='best', fontsize='x-small')
	plt.xlabel('longitudinal position [mm]')
	plt.ylabel('time [ms]')
	plt.grid()
	# plt.pause(0.01)
	plt.savefig(path_where_to_save_everything +'/fast_camera_merge_'+str(merge_ID_target)+'_average'+'5.eps', bbox_inches='tight')
	plt.close('all')

	ani = coleval.movie_from_data(np.array([np.nanmean(all_averaged_profile,axis=(0))]), framerate, integration_time,'horizontal coord [pixels]','vertical coord [pixels]','Intersity [au]')
	ani.save(path_where_to_save_everything +'/fast_camera_merge_'+str(merge_ID_target)+'_average' + '6.mp4', fps=5, writer='ffmpeg',codec='mpeg4')
	# ani.save(path_where_to_save_everything+'/file_index_' + str(j) +'_IR_trace_'+IR_trace + '_original.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
	plt.close()


	np.savez_compressed(path_where_to_save_everything +'/fast_camera_merge_'+str(merge_ID_target), bbox_inches='tight')
