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

# I follow the the digital level temperature conversion from Y. Li and Thomas Morgan




# temp=[]
# merge_id_all = [66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85, 95, 86, 87, 89, 92, 93, 94, 96, 97, 98, 88, 99]
# # merge_id_all = [97]
# for merge_id in merge_id_all:
# 	temp.extend(find_index_of_file(merge_id,df_settings,df_log,only_OES=False))
# all_j=[]
# for i in range(len(df_log)):
# 	IR_trace,IR_reference = df_log.loc[i,['IR_trace','IR_reference']]
# 	# if (isinstance(IR_trace, str) and np.isnan(df_log.loc[i,['DT_pulse']][0])):
# 	if (isinstance(IR_trace, str) and (i in temp)):
# 		# if i<=265:
# 		all_j.append(i)

def bin2dec_for_list(array):
	import numpy as np
	temp = []
	for value in array:
		temp.append(int(value,2))
	return np.array(temp)


def DOM52sec_for_list(array):
	import numpy as np
	date_time_stamp2sec = []
	for value in array:
		if np.isfinite(value)==True:
			date_time_stamp2sec.append( DOM52sec(value) )
	return np.array(date_time_stamp2sec)

def DOM52sec_for_list(array):
	import numpy as np
	import datetime as dt
	date_time_stamp2sec = []
	for value in array:
		if np.isfinite(value)==True:
			# date_time_stamp2sec.append( dt.datetime.fromtimestamp(int(DOM52sec(value))) )
			date_time_stamp2sec.append( dt.datetime.fromtimestamp(DOM52sec(value)) )
	return np.array(date_time_stamp2sec)


def DOM52sec(value):
	if np.isfinite(value)==True:
		out = int("{0:b}".format(int(value))[:-32],2)+int("{0:b}".format(int(value))[-32:],2)/(2**32)
		return out
	else:
		return np.nan

def specemi(Lambda,T,de):
	e0 = 0.4205 *(Lambda**((1.88949 *Lambda**3+0.24191) /(Lambda**3+1.90197))) **-1;
	e1 = 5.55537e-5+1.77717e-6*e0-8.43353e-4*e0**2+9.82596e-4*e0**3;
	e = e0 + e1*T + de;
	return e

def dl2temp_generator(de,beta_sample,tau,t_exp,target_material,out_emissivity=False):

	beta_cam = np.arccos(77/92)*360/(2*np.pi) + beta_sample;                     # camera view angle in Magnum-PSI

	c = 299792458;                                                               # [m/s], speed of light in a vaccum
	h = 6.62607015e-34;                                                       # [J*s], Planck constant
	k = 1.380649e-23;                                                           # [J/K], Boltzmann constant

	lambda_cam = np.arange(3.97,4.01,0.001)*1e-6;                            # [m], wavelength, camera spectral range for the 60% filter


	# https://www.plansee.com/en/materials/molybdenum.html
	moly_emissivity_interp = interp1d([0,280,500,1000,1500,2000,2500],[0.08,0.1,0.125,0.181,0.243,0.294,0.33],fill_value="extrapolate")

	T_ref = np.arange(300,30000,1);                                                     # [K]

	e_cal = np.zeros_like(T_ref,dtype=np.float)
	phi_bb = np.zeros_like(T_ref,dtype=np.float)
	for ii in range(len(T_ref)):
		if target_material=='tungsten dummy':
			e_cal[ii] = np.mean(np.min([specemi(lambda_cam*1e6,T_ref[ii],de),np.ones_like(lambda_cam)],axis=0))
		else:
			# https://www.plansee.com/en/materials/molybdenum.html
			e_cal[ii] = moly_emissivity_interp(T_ref[ii]+273.15)+de
		# spectral radiance, [W/Sr/m3] to average photon fluence including emissivity
		phi_bb[ii] = e_cal[ii] * np.mean(2 * h * c**2 /(lambda_cam**5) *  ((np.exp(h * c / (lambda_cam * k * T_ref[ii])) - 1)**-1) /(h * c /lambda_cam) * lambda_cam ) * t_exp/1e6

	# print(e_cal)
	# to real photon flux

	phi_real = phi_bb * tau

	# to convert photon fluence into digital level,
	# coefficients come from the dl2fluence script, with raw data from Jordy
	dl_real = phi_real * 6.9825e-16 -3.74;
	# to substract the bb @ T==300 K
	dl_real = dl_real - dl_real[0];

	# the Table

	dl_cal = np.arange(0,20000,1)
	temp = interpolate.interp1d(dl_real,T_ref,kind='slinear')
	temp_cal = temp(dl_cal)
	dl2temp = interpolate.interp1d(dl_cal,temp_cal,kind='slinear')

	if out_emissivity:
		return dl2temp,T_ref,e_cal
	else:
		return dl2temp


plt.figure()
all_j = [231,232,233,234,244,245,393,394,305,306]
time_shift = [1562076516.351,1562076678.708+4,1562077276.157,1562077372.03,1562080690.253,1562080859.278,1562165683.122,1562165819.007,1562169014.003,1562169074.379]
de = [0.07,0.07,0.2,0.2,0.4,0.4,0.05,0.05,-0.1,-0.1]                                                                           # emissivity offset, -0.03 for polished tungsten
for i_j,j in enumerate(all_j):

	print('analysing item n '+str(j))

	df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
	(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
	(IR_trace,IR_reference,IR_shape,magnetic_field,target_OES_distance,target_chamber_pressure,capacitor_voltage,target_material) = df_log.loc[j,['IR_trace','IR_reference','IR_shape','B','T_axial','p_n [Pa]','Vc','Target']]
	(CB_to_OES_initial_delay,incremental_step,number_of_pulses) = df_log.loc[j,['CB_to_OES_initial_delay','incremental_step','number_of_pulses']]
	number_of_pulses = int(number_of_pulses)
	(folder,date,sequence,untitled,target_material,fname_current_trace,first_pulse_at_this_frame,bad_pulses_indexes) = df_log.loc[j,['folder','date','sequence','untitled','Target','current_trace_file','first_pulse_at_this_frame','bad_pulses_indexes']]

	merge_found = 0
	for i in range(len(df_settings)):
		a,b = df_settings.loc[i,['Hb','merge_ID']]
		if a==j:
			merge_ID=b
			merge_found=1
			print('Equivalent to merge '+str(merge_ID))
			break

	pre_title = 'merge %.3g, B=%.3gT, pos=%.3gmm, P=%.3gPa, ELMen=%.3gJ \n' %(merge_ID,magnetic_field,target_OES_distance,target_chamber_pressure,0.5*(capacitor_voltage**2)*150e-6)

	incremental_step = 0.1	# I do this because what I mean really is the time between pulses, that is always 0.1s

	IR_shape = np.array((IR_shape.replace(' ', '').split(','))).astype(int)

	if merge_found:
		path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID)
	else:
		path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/'+"{0:0=2g}".format(sequence)+'/IR_camera'
	figure_index = 0

	header = ryptw.readPTWHeader('/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/'+"{0:0=2g}".format(sequence)+'/IR_camera/'+IR_reference+'.ptw')
	# ryptw.showHeader(header)

	f = 1/header.h_CEDIPAquisitionPeriod                                                                        # [Hz], frame rate
	t_exp = round(header.h_CEDIPIntegrationTime,9)*1e6                                                                   # [mus], camera integration time

	if t_exp == 100:
		background_interval = int(6*f)
	elif t_exp == 400:
		background_interval = int(5*f)
	elif t_exp == 1000:
		background_interval = int(4*f)
	else:
		print('error, only 1, 0.1 and 0.4ms int time now')

	print('mark7')

	# frames = frames.flatten().reshape((np.shape(frames)[1],np.shape(frames)[2],np.shape(frames)[0])).T
	frames = []
	for i in range(1,background_interval):
		frame = ryptw.getPTWFrames(header, [i])[0][0]
		frames.append(frame.flatten().reshape(np.shape(frame.T)).T)
	print('mark6')

	dark = np.mean(frames,axis=0)

	header = ryptw.readPTWHeader('/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/'+"{0:0=2g}".format(sequence)+'/IR_camera/'+IR_trace+'.ptw')

	print('mark7')

	# frames = frames.flatten().reshape((np.shape(frames)[1],np.shape(frames)[2],np.shape(frames)[0])).T
	frames = []
	for i in range(header.h_firstframe,header.h_lastframe):
		frame = ryptw.getPTWFrames(header, [i])[0][0]
		frames.append(frame.flatten().reshape(np.shape(frame.T)).T)
	print('mark6')

	dl_tot = np.array(frames)
	dl_ave = np.mean(dl_tot,axis=(1,2))



	T_pyro = 615;                                                              # pyrometer reading, [K]
	beta_sample = 0;                                                           # sample tilting of the target manipulator?
	AOI_center = [IR_shape[1],IR_shape[0]];                                                   # position to calculate [X,Y]; always the center for calibration
	# AOI_edge = [52,IR_shape[0]]#[25,8];
	f = 1/header.h_CEDIPAquisitionPeriod                                                                        # [Hz], frame rate
	t_exp = round(header.h_CEDIPIntegrationTime,9)*1e6                                                                   # [mus], camera integration time
	# tau = 0.145;                                                                    # optical transmission
	tau = 0.145;                                                                    # optical transmission

	p2d_h = 25/77;                                                              # 25 mm == 77 pixels without sample tilting
	p2d_v = 25/92;

	dl2temp = dl2temp_generator(de[i_j],beta_sample,tau,t_exp,target_material)


	# DL to temp
	# Background subtraction
	dl_bkg = dark                                                                # if the real background is recorded; a manual one can be added
	dl_sub = np.abs(dl_tot - dl_bkg);
	# dl_sub_center = reshape(dl_sub(Nrow-AOI_center(2),AOI_center(1),:),[],1);                     # DL at a the center
	# dl_sub_edge = reshape(dl_sub(Nrow-AOI_edge(2),AOI_edge(1),:),[],1);
	dl_sub_center = dl_sub[:,AOI_center[1],AOI_center[0]]                     # DL at a the center
	# dl_sub_edge = dl_sub[:,AOI_edge[1],AOI_edge[0]]

	temp_sub_center = dl2temp(dl_sub_center+1);                                                                   # central temp.
	# temp field
	temp_sub_full = dl2temp(dl_sub+1)

	selected_x = np.zeros_like(dl_tot[0])+np.arange(np.shape(dl_tot)[-1])
	selected_y = (np.zeros_like(dl_tot[0]).T+np.arange(np.shape(dl_tot)[-2])).T
	selected_1 = ((selected_x-IR_shape[1])**2 + (selected_y-IR_shape[0])**2)<IR_shape[2]**2
	selected_2 = ((selected_x-IR_shape[1])**2 + (selected_y-IR_shape[0])**2)<IR_shape[3]**2

	# selected_1_mean_counts = np.mean(dl_tot[:,selected_1],axis=-1)
	# selected_1_max_counts = np.max(dl_tot[:,selected_1],axis=-1)
	# selected_2_mean_counts = np.mean(dl_tot[:,selected_2],axis=-1)
	# selected_2_max_counts = np.max(dl_tot[:,selected_2],axis=-1)
	#
	# selected_1_mean_temp = dl2temp(selected_1_mean_counts+1)
	# selected_1_max_temp = dl2temp(selected_1_max_counts+1)
	# selected_2_mean_temp = dl2temp(selected_2_mean_counts+1)
	# selected_2_max_temp = dl2temp(selected_2_max_counts+1)

	selected_1_mean_temp = np.mean(temp_sub_full[:,selected_1],axis=-1)
	selected_1_max_temp = np.max(temp_sub_full[:,selected_1],axis=-1)
	selected_2_mean_temp = np.mean(temp_sub_full[:,selected_2],axis=-1)
	selected_2_max_temp = np.max(temp_sub_full[:,selected_2],axis=-1)

	dl_max = np.max(dl_ave)
	idx_max_frame = dl_ave.argmax();
	time_IR = np.linspace(0,len(temp_sub_center)/f,len(temp_sub_center));                # [s]
	time_IR += time_shift[i_j]
	temp_max = np.max(temp_sub_center);
	temp_min = np.min(temp_sub_center);

	# plt.plot(dl_ave)
	if i_j==0:
		plt.plot(time_IR,temp_sub_center,'C0',label='temp_sub_center')
		# plt.plot(np.arange(0,len(temp_sub_center))/f,temp_sub_edge,label='temp_sub_edge')
		plt.plot(time_IR,selected_1_mean_temp,'C1',label='selected_1_mean_temp')
		plt.plot(time_IR,selected_1_max_temp,'--C2',label='selected_1_max_temp')
		plt.plot(time_IR,selected_2_mean_temp,'C3',label='selected_2_mean_temp')
		plt.plot(time_IR,selected_2_max_temp,'--C4',label='selected_2_max_temp')
		# plt.plot([time_IR.min(),time_IR.max()],[T_pyro]*2,'C5',label='T_pyro')
	else:
		plt.plot(time_IR,temp_sub_center,'C0')
		# plt.plot(np.arange(0,len(temp_sub_center))/f,temp_sub_edge,label='temp_sub_edge')
		plt.plot(time_IR,selected_1_mean_temp,'C1')
		plt.plot(time_IR,selected_1_max_temp,'--C2')
		plt.plot(time_IR,selected_2_mean_temp,'C3')
		plt.plot(time_IR,selected_2_max_temp,'--C4')
		# plt.plot([time_IR.min(),time_IR.max()],[T_pyro]*2,'C5')


multi_1 = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/2019-07-02/20190702_1.csv',index_col=False,sep='|')
index1 = list(multi_1.head(0))
for index in range(len(index1)//3):
	index1[index*3]=index1[index*3+1]
	index1[index*3+2]=index1[index*3+1]
multi_1 = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/2019-07-02/20190702_1.csv',index_col=False,header=1,sep='|')
index2 = list(multi_1.head(0))
index = [_+' '+__ for _,__ in zip(index1,index2)]
for i_text,text in enumerate(index):
	if (text.find('.'))!=-1:
		index[i_text] = text[:text.find('.')]
multi_1 = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/2019-07-02/20190702_1.csv',index_col=False,header=2,sep='|')
multi_1_array = np.array(multi_1)

# plt.title(index[i_index*3+2])
# for i_index in range(0,13):
# for i_index in [4,11]:
i_index = 11
date_time_stamp = multi_1[multi_1.keys()[i_index*3]]
date_time_stamp2time = DOM52sec_for_list(date_time_stamp)
temp = []
for i in range(len(date_time_stamp2time)):
	temp.append(date_time_stamp2time[i].timestamp())
temp = np.array(temp)
# temp -= temp[376]
# date_time_stamp2time = [dt.datetime.fromtimestamp(int(_)) for _ in date_time_stamp2sec//1 ]
plt.errorbar(temp,multi_1[multi_1.keys()[i_index*3+2]][:len(date_time_stamp2time)]+273.15,yerr=multi_1[multi_1.keys()[i_index*3+2+3]][:len(date_time_stamp2time)],label=index[i_index*3+2])

multi_1 = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/2019-07-03/20190703_1.csv',index_col=False,sep='|')
index1 = list(multi_1.head(0))
for index in range(len(index1)//3):
	index1[index*3]=index1[index*3+1]
	index1[index*3+2]=index1[index*3+1]
multi_1 = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/2019-07-03/20190703_1.csv',index_col=False,header=1,sep='|')
index2 = list(multi_1.head(0))
index = [_+' '+__ for _,__ in zip(index1,index2)]
for i_text,text in enumerate(index):
	if (text.find('.'))!=-1:
		index[i_text] = text[:text.find('.')]
multi_1 = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/2019-07-03/20190703_1.csv',index_col=False,header=2,sep='|')
multi_1_array = np.array(multi_1)

# plt.title(index[i_index*3+2])
# for i_index in range(0,13):
# for i_index in [4,11]:
i_index = 11
date_time_stamp = multi_1[multi_1.keys()[i_index*3]]
date_time_stamp2time = DOM52sec_for_list(date_time_stamp)
temp = []
for i in range(len(date_time_stamp2time)):
	temp.append(date_time_stamp2time[i].timestamp())
temp = np.array(temp)
# temp -= temp[376]
# date_time_stamp2time = [dt.datetime.fromtimestamp(int(_)) for _ in date_time_stamp2sec//1 ]
plt.errorbar(temp,multi_1[multi_1.keys()[i_index*3+2]][:len(date_time_stamp2time)]+273.15,yerr=multi_1[multi_1.keys()[i_index*3+2+3]][:len(date_time_stamp2time)],label=index[i_index*3+2])


plt.legend(loc='best', fontsize='xx-small')
plt.grid()
plt.title(pre_title+'emissivity addition %.500s, window transmissivity %.3g' %(str(de),tau))
plt.ylabel('Temperature [K]')
plt.xlabel('time [s]')
plt.pause(0.1)






# j=232
# print('analysing item n '+str(j))
#
# df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
# (folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
# (IR_trace,IR_reference,IR_shape,magnetic_field,target_OES_distance,target_chamber_pressure,capacitor_voltage) = df_log.loc[j,['IR_trace','IR_reference','IR_shape','B','T_axial','p_n [Pa]','Vc']]
# (CB_to_OES_initial_delay,incremental_step,number_of_pulses) = df_log.loc[j,['CB_to_OES_initial_delay','incremental_step','number_of_pulses']]
# number_of_pulses = int(number_of_pulses)
# (folder,date,sequence,untitled,target_material,fname_current_trace,first_pulse_at_this_frame,bad_pulses_indexes) = df_log.loc[j,['folder','date','sequence','untitled','Target','current_trace_file','first_pulse_at_this_frame','bad_pulses_indexes']]
#
# merge_found = 0
# for i in range(len(df_settings)):
# 	a,b = df_settings.loc[i,['Hb','merge_ID']]
# 	if a==j:
# 		merge_ID=b
# 		merge_found=1
# 		print('Equivalent to merge '+str(merge_ID))
# 		break
#
# pre_title = 'merge %.3g, B=%.3gT, pos=%.3gmm, P=%.3gPa, ELMen=%.3gJ \n' %(merge_ID,magnetic_field,target_OES_distance,target_chamber_pressure,0.5*(capacitor_voltage**2)*150e-6)
#
# incremental_step = 0.1	# I do this because what I mean really is the time between pulses, that is always 0.1s
#
# IR_shape = np.array((IR_shape.replace(' ', '').split(','))).astype(int)
#
# if merge_found:
# 	path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID)
# else:
# 	path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/'+"{0:0=2g}".format(sequence)+'/IR_camera'
# figure_index = 0
#
# header = ryptw.readPTWHeader('/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/'+"{0:0=2g}".format(sequence)+'/IR_camera/'+IR_reference+'.ptw')
# # ryptw.showHeader(header)
#
# if int_time == 1e-4:
# 	background_interval = int(6*frequency)
# elif int_time == 4e-4:
# 	background_interval = int(5*frequency)
# elif int_time == 1e-3:
# 	background_interval = int(4*frequency)
# else:
# 	print('error, only 1, 0.1 and 0.4ms int time now')
#
# print('mark7')
#
# # frames = frames.flatten().reshape((np.shape(frames)[1],np.shape(frames)[2],np.shape(frames)[0])).T
# frames = []
# for i in range(1,background_interval):
# 	frame = ryptw.getPTWFrames(header, [i])[0][0]
# 	frames.append(frame.flatten().reshape(np.shape(frame.T)).T)
# print('mark6')
#
# dark = np.mean(frames,axis=0)
#
# header = ryptw.readPTWHeader('/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/'+"{0:0=2g}".format(sequence)+'/IR_camera/'+IR_trace+'.ptw')
#
# print('mark7')
#
# # frames = frames.flatten().reshape((np.shape(frames)[1],np.shape(frames)[2],np.shape(frames)[0])).T
# frames = []
# for i in range(header.h_firstframe,header.h_lastframe):
# 	frame = ryptw.getPTWFrames(header, [i])[0][0]
# 	frames.append(frame.flatten().reshape(np.shape(frame.T)).T)
# print('mark6')
#
# dl_tot = np.array(frames)
# dl_ave = np.mean(dl_tot,axis=(1,2))
#
# T_pyro = 615;                                                              # pyrometer reading, [K]
# beta_sample = 0;                                                           # sample tilting of the target manipulator?
# AOI_center = [IR_shape[1],IR_shape[0]];                                                   # position to calculate [X,Y]; always the center for calibration
# AOI_edge = [52,IR_shape[0]]#[25,8];
# de = +0.75;                                                                           # emissivity offset, -0.03 for polished tungsten
# f = 1/header.h_CEDIPAquisitionPeriod                                                                        # [Hz], frame rate
# t_exp = round(header.h_CEDIPIntegrationTime,9)*1e6                                                                   # [mus], camera integration time
# tau = 0.145;                                                                    # optical transmission
#
# f = 1/header.h_CEDIPAquisitionPeriod                                                                        # [Hz], frame rate
# t_exp = round(header.h_CEDIPIntegrationTime,9)*1e6                                                                   # [mus], camera integration time
#
# p2d_h = 25/77;                                                              # 25 mm == 77 pixels without sample tilting
# p2d_v = 25/92;
#
# dl2temp = dl2temp_generator(de,beta_sample,tau)
#
# # DL to temp
# # Background subtraction
# dl_bkg = dark                                                                # if the real background is recorded; a manual one can be added
# dl_sub = np.abs(dl_tot - dl_bkg);
# # dl_sub_center = reshape(dl_sub(Nrow-AOI_center(2),AOI_center(1),:),[],1);                     # DL at a the center
# # dl_sub_edge = reshape(dl_sub(Nrow-AOI_edge(2),AOI_edge(1),:),[],1);
# dl_sub_center = dl_sub[:,AOI_center[1],AOI_center[0]]                     # DL at a the center
# dl_sub_edge = dl_sub[:,AOI_edge[1],AOI_edge[0]]
#
# temp_sub_center = dl2temp(dl_sub_center+1);                                                                   # central temp.
# temp_sub_edge = dl2temp(dl_sub_edge+1);
# # temp field
# temp_sub_full = dl2temp(dl_sub+1)
#
# dl_max = np.max(dl_ave)
# idx_max_frame = dl_ave.argmax();
# time_IR = np.linspace(0,len(temp_sub_center)/f,len(temp_sub_center));                # [s]
# temp_max = np.max(temp_sub_center);
# temp_min = np.min(temp_sub_center);
#
# plt.figure()
# # plt.plot(dl_ave)
# plt.plot(np.arange(0,len(temp_sub_center))/f,temp_sub_center,label='temp_sub_center')
# plt.plot(np.arange(0,len(temp_sub_center))/f,temp_sub_edge,label='temp_sub_edge')
# plt.plot([0,len(temp_sub_center)/f],[T_pyro]*2,label='T_pyro')
#
#
#
# multi_1 = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/2019-07-02/20190702_1.csv',index_col=False,sep='|')
# index1 = list(multi_1.head(0))
# for index in range(len(index1)//3):
# 	index1[index*3]=index1[index*3+1]
# 	index1[index*3+2]=index1[index*3+1]
# multi_1 = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/2019-07-02/20190702_1.csv',index_col=False,header=1,sep='|')
# index2 = list(multi_1.head(0))
# index = [_+' '+__ for _,__ in zip(index1,index2)]
# for i_text,text in enumerate(index):
# 	if (text.find('.'))!=-1:
# 		index[i_text] = text[:text.find('.')]
# multi_1 = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/2019-07-02/20190702_1.csv',index_col=False,header=2,sep='|')
# multi_1_array = np.array(multi_1)
#
# # plt.title(index[i_index*3+2])
# # for i_index in range(0,13):
# # for i_index in [4,11]:
# i_index = 11
# date_time_stamp = multi_1[multi_1.keys()[i_index*3]]
# date_time_stamp2time = DOM52sec_for_list(date_time_stamp)
# temp = []
# for i in range(len(date_time_stamp2time)):
# 	temp.append(date_time_stamp2time[i].timestamp())
# temp = np.array(temp)
# temp -= temp[405]
# # date_time_stamp2time = [dt.datetime.fromtimestamp(int(_)) for _ in date_time_stamp2sec//1 ]
# plt.errorbar(temp,multi_1[multi_1.keys()[i_index*3+2]][:len(date_time_stamp2time)],yerr=multi_1[multi_1.keys()[i_index*3+2+3]][:len(date_time_stamp2time)],label=index[i_index*3+2])
# plt.legend(loc='best')
# plt.grid()
# plt.pause(0.1)
