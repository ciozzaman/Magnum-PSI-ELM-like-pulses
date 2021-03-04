import numpy as np
import matplotlib.pyplot as plt
import os,sys



def all_file_names(extpath,type):

	# This utility returns a list with all filenames of a defined type in a given folder, orderen alphabetically and in number

	# path=os.getcwd()
	path=extpath
	print('path =', path)

	# position=[]
	# for i in range(len(path)):
	# 	if path[i]=='/':
	# 		position.append(i)
	# position=max(position)
	# lastpath=path[position+1:]
	# print('lastpath',lastpath)

	f = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		f.extend(filenames)
	#	break

	filenames=f
	# print('filenames',filenames)


	temp=[]

	typelen=len(type)
	for index in range(len(filenames)):
		# print(filenames[index])
		if filenames[index][-typelen:]==type:
			temp.append(filenames[index])
			# filenames=np.delete(filenames,index)
			# print('suca')
	filenames=temp

	if len(filenames)==0:
		print('ERROR - there are no files of type '+type+' in path '+path)
		return ()

	if len(filenames)==1:
		print('len(filenames)',len(filenames))
		return filenames

	filenames=order_filenames(filenames)

	print('len(filenames)',len(filenames))
	return filenames

#######################################################################################

def order_filenames(filenames):

	# 13/05/2018 THIS FUNCTION IS INTODUCED TO FIX A BUG IN THE CASE filenames CONTAINS .npy FILES AND NOT .csv

	extention=filenames[0][-4:]
	if ((extention=='.csv') or (extention=='.CSV')):
		return order_filenames_csv(filenames)
	else:
		return sorted(filenames, key=str.lower)

######################################################################################

def order_filenames_csv(filenames):

	# section to be sure that the files are ordered in the proper numeric order
	# CAUTION!! THE FOLDER MUST CONTAIN MAX 9999999 FILES!

	reference=[]
	filenamescorr=[]
	for i in range(len(filenames)):
		# print(filenames[i])
		start=0
		end=len(filenames[i])
		for j in range(len(filenames[i])):
			if ((filenames[i][j]=='-') or (filenames[i][j]=='_')): #modified 12/08/2018 from "_" to "-"
				start=j
			elif filenames[i][j]=='.':
				end=j
		index=filenames[i][start+1:end]
		#print('index',index)

		# Commented 21/08/2018
		# if int(index)<10:
		# 	temp=filenames[i][:start+1]+'000000'+filenames[i][end-1:]
		# 	filenamescorr.append(temp)
		# elif int(index)<100:
		# 	temp=filenames[i][:start+1]+'00000'+filenames[i][end-2:]
		# 	filenamescorr.append(temp)
		# 	# print(filenames[i],temp)
		# elif int(index)<1000:
		# 	temp=filenames[i][:start+1]+'0000'+filenames[i][end-3:]
		# 	filenamescorr.append(temp)
		# elif int(index)<10000:
		# 	temp=filenames[i][:start+1]+'000'+filenames[i][end-4:]
		# 	filenamescorr.append(temp)
		# elif int(index)<100000:
		# 	temp=filenames[i][:start+1]+'00'+filenames[i][end-5:]
		# 	filenamescorr.append(temp)
		# elif int(index)<1000000:
		# 	temp=filenames[i][:start+1]+'0'+filenames[i][end-6:]
		# 	filenamescorr.append(temp)
		# else:
		# 	filenamescorr.append(filenames[i])
		reference.append(index)
	reference=np.array(reference)
	filenamesnew=np.array([filenames for _,filenames in sorted(zip(reference,filenames))])

	# Commented 21/08/2018
	# filenamescorr=sorted(filenamescorr, key=str.lower)
	# filenamesnew=[]
	# for i in range(len(filenamescorr)):
	# 	# print(filenamescorr[i])
	# 	for j in range(len(filenamescorr[i])):
	# 		if ((filenames[i][j]=='-') or (filenames[i][j]=='_')): #modified 12/08/2018 from "_" to "-"
	# 			start=j
	# 		elif filenamescorr[i][j]=='.':
	# 			end=j
	# 	index=filenamescorr[i][start+1:end]
	# 	# if int(index)<10:
	# 	# 	temp=filenamescorr[i][:start+1]+str(int(index))+filenamescorr[i][end:]
	# 	# 	filenamesnew.append(temp)
	# 	# elif int(index)<100:
	# 	# 	temp=filenamescorr[i][:start+1]+str(int(index))+filenamescorr[i][end:]
	# 	# 	filenamesnew.append(temp)
	# 	# 	# print(filenames[i],temp)
	# 	# if int(index)<1000000:
	# 	temp=filenamescorr[i][:start+1]+str(int(index))+filenamescorr[i][end:]
	# 	filenamesnew.append(temp)
	# 	# 	# print(filenamescorr[i],temp)
	# 	# else:
	# 	# 	filenamesnew.append(filenamescorr[i])
	filenames=filenamesnew

	return(filenames)


####################################################################################################

def find_nearest_index(array,value):

	# 14/08/2018 This function returns the index of the closer value to "value" inside an array

	array_shape=np.shape(array)

	index = np.abs(np.add(array,-value)).argmin()
	residual_index=index
	cycle=1
	done=0
	position_min=np.zeros(len(array_shape),dtype=int)
	while done!=1:
		length=array_shape[-cycle]
		if residual_index<length:
			position_min[-cycle]=residual_index
			done=1
		else:
			position_min[-cycle]=round(((residual_index/length) %1) *length +0.000000000000001)
			residual_index=residual_index//length
			cycle+=1

	return position_min[0]

####################################################################################################


def multi_gaussian(n):
	import numpy as np
	def polyadd(x, *params):
		#print(np.shape(x))
		params=np.array(params)
		shape=np.shape(x)
		temp=np.zeros(shape)
		# if (len(shape)>1) & (len(np.shape(params))==1):
		# 	params=np.reshape(params,(shape[-2],shape[-1],n))
		for i in range(n):
			# print('np.shape(temp),np.shape(params),np.shape(x)',temp,params,x)
			# print('params[:][:][i]',params[:][:][i])
			# print('params',params)
			para=params[i*3:(i+1)*3]	#mean coordinate, sigma^2, amplitude
			#print(para)
			x2=np.power(x-para[0],2)
			gauss = para[2]/(2*np.pi*(para[1]**2))*np.exp(-x2/(2*(para[1]**2)))
			temp=np.add(temp,gauss)
		#print(np.shape(temp+params[-1]))
		#return temp+params[-1]+params[-2]*np.array(x)
		return temp+params[-1]
	return polyadd


####################################################################################################


def load_dark(j,df_settings,df_log,fdir,geom):
	import numpy as np
	from functions.spectools import rotate,do_tilt, binData
	from PIL import Image

	jDark = np.where((df_log['folder']==df_log.loc[j,'Dark_folder']) & (df_log['sequence']==df_log.loc[j,'Dark_sequence']) & (df_log['untitled']==df_log.loc[j,'Dark_untitled']) & (df_log['type'] == 'Dark'))[0][0]
	(fDark,dDark,sDark,uDark) = df_log.loc[jDark,['folder','date','sequence','untitled']]
	type = '.tif'
	filenames_Dark = all_file_names(fdir+'/'+fDark+'/'+"{0:0=2g}".format(sDark)+'/Untitled_'+str(int(uDark))+'/Pos0', type)
	# dataDark=0
	# dataDark_count=0
	dataDark=[]
	for index,filename in enumerate(filenames_Dark):
		fname = fdir+'/'+fDark+'/'+"{0:0=2g}".format(sDark)+'/Untitled_'+str(int(uDark))+'/Pos0/'+filename
		im = Image.open(fname)
		data = np.array(im)
		# dataDark+=data
		dataDark.append(data)
		# dataDark_count+=1
	dataDark = np.mean(dataDark,axis=0)
	dataDark=rotate(dataDark.astype('float'),geom['angle'])
	if np.shape(geom['tilt'][0]) == ():
		dataDark=do_tilt(dataDark,geom['tilt'][0])
	else:
		dataDark = four_point_transform(dataDark, geom['tilt'][0])
	# dataDark=dataDark/dataDark_count
	return dataDark

###################################################################################################


def find_index_of_file(merge_ID_target,df_settings,df_log,type='Hb',only_OES=False):

	out=[]
	for i in range(len(df_settings)):
			j,merge_ID = df_settings.loc[i,[type,'merge_ID']]
			if not np.isnan(j):
				if (merge_ID!=merge_ID_target or (only_OES and (df_settings.loc[i,['notes']][0]=='OES data missing'))):
					#print('bad'+str(j))
					continue
				out.append(j)
	return out

####################################################################################################

def get_metadata(fdir,folder,sequence,untitled,filename_metadata,pulse_frequency_Hz=10):
	import numpy as np

	# incremental_step*=1000	# ms

	# List of camera characteristics
	# Photometrics Prime95B 25MM
	# Serial Number: A17M203010
	Gain_16_bit = 0.88
	RMS_noise_16_bit = 1.8
	Gain_sensitivity = 0.56
	RMS_noise_sensitivity = 2
	Gain_balanced = 1.14
	RMS_noise_balanced = 2.8
	Gain_full_well = 2.44
	RMS_noise_full_well = 4.3

	bof=[]
	eof=[]
	metadata= open(os.path.join(fdir+'/'+folder+'/'+"{0:0=2g}".format(sequence)+'/Untitled_'+str(int(untitled))+'/Pos0', filename_metadata), 'r')
	flag=0
	roi_lb=[0,0]
	roi_tr=[0,0]
	elapsed_time=[]
	PixelType=[]
	Gain=[]
	Noise=[]
	TimeStampMsec=[]
	real_exposure_time=[]
	PVCAM_TimeStampBOF = []
	PVCAM_TimeStamp = []
	for irow,row in enumerate(metadata):
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
			real_exposure_time.append(int(row[start:end])*0.000001)
		if row.find('ElapsedTime-ms') !=-1:
			#print('found eof')
			start=0
			end=len(row)
			for index,value in enumerate(row):
				if (value.isdigit() and start==0):
					start=int(index)
			for index, value in enumerate(np.flip(list(row),axis=0)):
				if (not value.isdigit() and start!=0 and end==len(row)):
					end=int(len(row)-index-2)
			#print(row[start:end])
			elapsed_time.append(float(row[start:end].replace(',','.')))
		if row.find('TimeStampMsec') !=-1:
			#print('found eof')
			start=0
			end=len(row)
			for index,value in enumerate(row):
				if (value.isdigit() and start==0):
					start=int(index)
			for index, value in enumerate(np.flip(list(row),axis=0)):
				if (not value.isdigit() and start!=0 and end==len(row)):
					end=int(len(row)-index-3)
			#print(row[start:end])
			TimeStampMsec.append(float(row[start:end].replace(',','.')))
		if row.find('PVCAM-TimeStampBOF') !=-1:
			#print('found eof')
			start=0
			end=len(row)
			for index,value in enumerate(row):
				if (value.isdigit() and start==0):
					start=int(index)
			for index, value in enumerate(np.flip(list(row),axis=0)):
				if (not value.isdigit() and start!=0 and end==len(row)):
					end=int(len(row)-index-3)
			#print(row[start:end])
			PVCAM_TimeStampBOF.append(float(row[start:end].replace(',','.')))
		if row.find('PVCAM-TimeStamp') !=-1:
			#print('found eof')
			start=0
			end=len(row)
			for index,value in enumerate(row):
				if (value.isdigit() and start==0):
					start=int(index)
			for index, value in enumerate(np.flip(list(row),axis=0)):
				if (not value.isdigit() and start!=0 and end==len(row)):
					end=int(len(row)-index-3)
			#print(row[start:end])
			PVCAM_TimeStamp.append(float(row[start:end].replace(',','.')))
		if row.find('PM Cam-StartTime-ms') !=-1:
			#print('found eof')
			start=0
			end=len(row)
			for index,value in enumerate(row):
				if (value.isdigit() and start==0):
					start=int(index)
			for index, value in enumerate(np.flip(list(row),axis=0)):
				if (not value.isdigit() and start!=0 and end==len(row)):
					end=int(len(row)-index-3)
			#print(row[start:end])
			PM_Cam_StartTime_ms = float(row[start:end].replace(',','.'))
		if row.find('PM Cam-PixelType') !=-1:
			#print('found eof')
			start=0
			end=len(row)
			for index,value in enumerate(row):
				if (value.isdigit() and start==0):
					start=int(index)
				elif (not value.isdigit() and start!=0 and end==len(row)):
					end=int(index)
			#print(row[start:end])
			PixelType.append(float(row[start:end]))
			if float(row[start:end])==16:
				Gain[len(PixelType)-1]=Gain_16_bit	#	this values came from Louis Keal (Photometrics) email of 24/06/19
				Noise[len(PixelType)-1]=RMS_noise_16_bit	#	this values came from Louis Keal (Photometrics) email of 24/06/19
		if row.find('PM Cam-Gain') !=-1:
			#print('found eof')
			start=0
			end=len(row)
			for index,value in enumerate(row):
				if (value.isdigit() and start==0):
					start=int(index)
				elif (not value.isdigit() and start!=0 and end==len(row)):
					end=int(index)
			#print(row[start:end])
			# Gain.append(float(row[start:end]))
			temp_1 = int(row[start:end])	#	these values came from Louis Keal (Photometrics) email of 24/06/19
			temp_2 = int(row[start:end])	#	these values came from Louis Keal (Photometrics) email of 24/06/19
			if temp_1==1:
				temp_1=Gain_full_well
				temp_2=RMS_noise_full_well
			elif temp_1==2:
				temp_1=Gain_balanced
				temp_2=RMS_noise_balanced
			elif temp_1 == 3:
				temp_1 = Gain_sensitivity
				temp_2=RMS_noise_sensitivity
			Gain.append(temp_1)
			Noise.append(temp_2)
			# Gain.append( 2**(np.abs(int(row[start:end])-3)) )	#	THIS MODIFICATION IS FOR AN EASYER HANDLING WHEN MERGING DATA

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
	elapsed_time=np.array(elapsed_time)
	TimeStampMsec=np.array(TimeStampMsec)
	PVCAM_TimeStamp_ms=np.array(PVCAM_TimeStamp)/10
	PVCAM_TimeStampBOF_ms=np.array(PVCAM_TimeStampBOF)/10
	timestamp_regularised_shift = np.mean(TimeStampMsec[1:]-PM_Cam_StartTime_ms-np.arange(len(TimeStampMsec)-1)*(1/pulse_frequency_Hz))
	TimeStampMsec_corrected = PM_Cam_StartTime_ms + (np.arange(len(TimeStampMsec))-1)*(1/pulse_frequency_Hz) + timestamp_regularised_shift
	TimeStampMsec_corrected[0] = TimeStampMsec_corrected[1]-np.diff(bof[:2])[0]*1e-6
	bof_regularised_shift = np.mean(TimeStampMsec-bof*1e-6)
	bof_corrected_ms = bof*1e-6+bof_regularised_shift
	metadata.close()

	time_info = dict([('elapsed_time',elapsed_time),('TimeStampMsec',TimeStampMsec),('PM_Cam_StartTime_ms',PM_Cam_StartTime_ms),('TimeStampMsec_corrected',TimeStampMsec_corrected),('bof_corrected_ms',bof_corrected_ms),('PVCAM_TimeStamp_ms',PVCAM_TimeStamp_ms),('PVCAM_TimeStampBOF_ms',PVCAM_TimeStampBOF_ms)])
	return bof,eof,roi_lb,roi_tr,time_info,real_exposure_time,PixelType,Gain,Noise



####################################################################################################

def examine_current_trace(fdir_long,fname_current_trace,number_pulses,minimum_pulse = 0.0002,want_the_prominences=False,want_the_power_per_pulse=False,want_the_mean_power_profile=False,time_between_pulses=0.1001,want_the_plot_of_mean_power_profile=False):

	# 14/05/2019 Script to analyse the current traces of Magnum-PSI to spot good or bad pulses

	import pandas as pd
	from scipy.signal import find_peaks, peak_prominences as get_proms
	import numpy as np
	from scipy.optimize import curve_fit
	from scipy.stats import mode
	from scipy import interpolate
	gauss = lambda x, A, sig, x0, y0: A * np.exp(-((x - x0) / sig) ** 4)+y0

	current_traces = pd.read_csv(fdir_long + fname_current_trace+'.tsf',index_col=False, delimiter='\t')
	current_traces_time = current_traces['Time [s]']
	current_traces_total = current_traces['I_Src_AC [A]']
	voltage_traces_total = current_traces['U_Src_DC [V]']
	target_voltage_traces_total = current_traces['U_Tar_DC [V]']
	power_traces_total = np.abs( current_traces_total * voltage_traces_total )
	# plt.figure()
	# plt.plot(current_traces_time,current_traces_total)
	# plt.pause(0.001)
	time_resolution = np.mean(np.diff(current_traces_time))
	peaks = find_peaks(current_traces_total, width=int(minimum_pulse / time_resolution), height=current_traces_total.max() / 3)[0]
	difference = np.diff(np.sort(peaks))
	temp = []
	skip = 0
	for index, value in enumerate(peaks):
		if skip == 1:
			skip = 0
			continue
		if index == len(peaks) - 1:
			temp.append(value)
			continue
		if (difference[index] < np.mean(difference) * 0.5):
			if current_traces_total[value] < current_traces_total[peaks[index + 1]]:
				continue
			else:
				skip = 1
		temp.append(value)

	peaks = np.array(temp)
	difference = np.diff(np.sort(peaks))

	peaks = np.array([peaks for _, peaks in sorted(zip(current_traces_total[peaks], peaks))])

	if (current_traces_total[peaks]).max() < 1.25 * np.mean(current_traces_total[peaks]):
		peaks_good = np.sort(peaks)
		peaks_double = np.array([])
	elif np.max(np.diff(current_traces_total[peaks])[:-1])<20:
		peaks_good = np.sort(peaks[:np.diff(current_traces_total[peaks]).argmax() + 1])
		peaks_double = np.sort(peaks[np.diff(current_traces_total[peaks]).argmax() + 1:])
	else:
		peaks_good = np.sort(peaks[:np.diff(current_traces_total[peaks])[:-1].argmax() + 1])
		peaks_double = np.sort(peaks[np.diff(current_traces_total[peaks])[:-1].argmax() + 1:])


	calculated_interval = np.median(difference)
	if calculated_interval>np.mean(difference):
		calculated_interval = int(np.sum(difference)/(len(difference)+len(peaks_double)))
	peaks = np.sort(peaks)

	peaks_missing = []
	multiplier = 1
	calculated_interval_internal = calculated_interval
	while ( ((len(peaks_missing)==0 or len(peaks_missing)>len(peaks))) and multiplier<2 ) :
		for index, value in enumerate(peaks):
			if index == len(peaks) - 1:
				continue
			jump=1
			while difference[index] > np.mean(difference) * 1.25:
				# peaks_missing.append(int((peaks[index + 1] + peaks[index]) / 2))
				where_additional_peak_should_be = int(peaks[index] + jump*calculated_interval_internal)
				# peaks_missing.append(int(peaks[index] + jump*calculated_interval_internal))
				peaks_missing.append((current_traces_total[int(where_additional_peak_should_be-calculated_interval_internal*0.05):int(where_additional_peak_should_be+calculated_interval_internal*0.05)]).idxmax())
				difference[index] -=calculated_interval_internal
				jump+=1
		peaks_missing = np.array(peaks_missing)
		multiplier +=0.1
		calculated_interval_internal = calculated_interval*multiplier

	all_peaks = np.sort(peaks_good.tolist() + peaks_double.tolist() + peaks_missing.tolist())
	if len(peaks_double) == 0:
		while current_traces_time[all_peaks[0]] % 1 > 0.213:
			# peaks_missing = np.concatenate(([int(min(all_peaks) - np.mean(np.diff(all_peaks)))], peaks_missing))
			# all_peaks = np.concatenate(([int(min(all_peaks) - np.mean(np.diff(all_peaks)))], all_peaks))
			where_additional_peak_should_be = int(min(all_peaks) - np.mean(np.diff(all_peaks)))
			peaks_missing = np.concatenate(([(current_traces_total[int(where_additional_peak_should_be-calculated_interval*0.05):int(where_additional_peak_should_be+calculated_interval*0.05)]).idxmax()], peaks_missing))
			all_peaks = np.concatenate(([(current_traces_total[int(where_additional_peak_should_be-calculated_interval*0.05):int(where_additional_peak_should_be+calculated_interval*0.05)]).idxmax()], all_peaks))
	else:
		while (min(all_peaks) == min(peaks_double) or current_traces_time[all_peaks[0]] % 1 > 0.213):
			# peaks_missing = np.concatenate(([int(min(all_peaks) - np.mean(np.diff(all_peaks)))], peaks_missing))
			# all_peaks = np.concatenate(([int(min(all_peaks) - np.mean(np.diff(all_peaks)))], all_peaks))
			where_additional_peak_should_be = int(min(all_peaks) - np.mean(np.diff(all_peaks)))
			peaks_missing = np.concatenate(([(current_traces_total[int(where_additional_peak_should_be-calculated_interval*0.05):int(where_additional_peak_should_be+calculated_interval*0.05)]).idxmax()], peaks_missing))
			all_peaks = np.concatenate(([(current_traces_total[int(where_additional_peak_should_be-calculated_interval*0.05):int(where_additional_peak_should_be+calculated_interval*0.05)]).idxmax()], all_peaks))

	# peaks_missing = np.concatenate(([int(min(all_peaks) - np.mean(np.diff(all_peaks)))], peaks_missing))
	# all_peaks = np.concatenate(([int(min(all_peaks)-np.mean(np.diff(all_peaks)))],all_peaks))

	while len(all_peaks) < number_pulses:
		# peaks_missing = np.concatenate((peaks_missing, [int(max(all_peaks) + np.mean(np.diff(all_peaks)))]))
		# all_peaks = np.concatenate((all_peaks, [int(max(all_peaks) + np.mean(np.diff(all_peaks)))]))
		where_additional_peak_should_be = int(max(all_peaks) + np.mean(np.diff(all_peaks)))
		peaks_missing = np.concatenate((peaks_missing, [(current_traces_total[int(where_additional_peak_should_be-calculated_interval*0.05):int(where_additional_peak_should_be+calculated_interval*0.05)]).idxmax()]))
		all_peaks = np.concatenate((all_peaks, [(current_traces_total[int(where_additional_peak_should_be-calculated_interval*0.05):int(where_additional_peak_should_be+calculated_interval*0.05)]).idxmax()]))
	# while len(all_peaks)>number_pulses:
	# 	if all_peaks[-1] in peaks_good:
	# 		peaks_good = peaks_good[:-1]
	# 	all_peaks = all_peaks[:-1]

	test=[]
	for index in range(len(peaks_good)):
		test.append(not peaks_good[index] in peaks_missing)
	peaks_good = peaks_good[test]


	peak_of_the_peak = []
	start_of_the_peak = []
	for peak in np.sort(peaks_good):
		xx = current_traces_time[peak-int(0.05//time_resolution) : peak]#+int(0.05//time_resolution)]
		yy = current_traces_total[peak-int(0.05//time_resolution) : peak]#+int(0.05//time_resolution)]
		p0 = [np.max(yy), 0.005, current_traces_time[peak], mode(yy)[0][0]]
		fit_1 = curve_fit(gauss, xx, yy, p0, maxfev=100000)
		noise = yy[0:-int(fit_1[0][1]*3/time_resolution)]
		noise_mean = np.mean(noise)
		noise_std = np.std(noise)
		threshold = 2*(np.max(noise) - noise_mean) + noise_mean
		xx_2 = np.array(xx[[*(yy > threshold)[1:],False]])[0:2]
		yy_2 = np.array(yy[[*(yy > threshold)[1:],False]])[0:2]
		fit = np.polyfit(xx_2, yy_2, 1)
		# start_of_the_peak.append(np.array(xx[yy > threshold])[0])	#	this use the beginning of the pulse profile
		start_of_the_peak.append((threshold-fit[1])/fit[0])	#	this use the beginning of the pulse profile but interpolated
		# plt.figure()
		# plt.plot(xx,yy)
		# plt.plot(xx, gauss(xx, *fit_1[0]))
		# plt.plot([np.array(xx[yy > threshold])[0],np.array(xx[yy > threshold])[0]],[np.min(yy),np.max(yy)],'k')
		# plt.plot([(threshold-fit[1])/fit[0],(threshold-fit[1])/fit[0]],[np.min(yy),np.max(yy)],'r')
		# plt.pause(0.001)
		# start_of_the_peak.append(fit[0][2])	# this use to the peak of the gaussian
		# start_of_the_peak.append(np.array(xx[gauss(xx, *fit[0])>fit[0][-1]*1.5])[0]) 	#	this use the beginning of the gaussian, in an imprecisse way
		# start_of_the_peak.append(fit_1[0][2]-(-(fit_1[0][1]**4)*np.log(0.5*fit_1[0][-1]/fit_1[0][0]))**0.25)	#	this use the beginning of the gaussian
		xx = current_traces_time[peak-3 : peak+4]#+int(0.05//time_resolution)]
		yy = power_traces_total[peak-3 : peak+4]#+int(0.05//time_resolution)]
		fit = np.polyfit(xx,yy,2)
		peak_of_the_peak.append(-fit[1]/(2*fit[0]))
	start_of_the_peak.extend(current_traces_time[peaks_missing])
	start_of_the_peak.extend(current_traces_time[peaks_double])
	peak_of_the_peak.extend(current_traces_time[peaks_missing])
	peak_of_the_peak.extend(current_traces_time[peaks_double])
	time_of_pulses = dict([('start_of_the_peak',np.sort(start_of_the_peak)),('peak_of_the_peak',np.sort(peak_of_the_peak))])
	# counter = collections.Counter(np.diff(start_of_the_peak)//0.0001)
	# x=list(counter.keys())
	# y=np.array(list(counter.values()))
	# y = np.array([y for _, y in sorted(zip(x, y))])
	# x = np.sort(x)
	# plt.figure()
	# plt.plot(x,y)
	# plt.pause(0.001)


	all_peaks = np.sort(all_peaks)
	bad = []
	good = []
	any = []
	miss = []
	double = []
	for index, value in enumerate(all_peaks):
		if not (value in peaks_good):
			bad.append(index + 1)
		else:
			good.append(index + 1)
		if ((value in peaks_double) or (value in peaks_good)):
			any.append(index + 1)

		if value in peaks_missing:
			miss.append(index+1)
		if value in peaks_double:
			double.append(index+1)
	bad = np.array(bad)
	good = np.array(good)
	any = np.array(any)
	miss = np.array(miss)
	double = np.array(double)

	print('double pulses = '+str(len(peaks_double)))
	print('missing pulses = ' + str(len(peaks_missing)))
	print('good pulses = ' + str(len(good)))

	prominences = get_proms(current_traces_total,all_peaks)[0]
	prominences[prominences==0]=np.max(prominences[prominences>0])

	if want_the_prominences==True:
		return bad, good[0], any[0], any[-1], miss,double, good, time_of_pulses, prominences
	elif want_the_power_per_pulse==True:
		# power_traces = np.abs(	current_traces_total * voltage_traces_total )
		# steady_state_power = np.median(power_traces)
		# spectra = np.fft.fft(power_traces[:min(np.abs(current_traces_time-2).argmin(),all_peaks[0]-int(np.median(np.diff(all_peaks))/2))])
		# spectra = np.fft.fft(power_traces)
		# phase = np.angle(spectra)
		# magnitude = 2 * np.abs(spectra) / len(spectra)
		# freq = np.fft.fftfreq(len(magnitude), d=time_resolution)
		# power_traces_corrected = cp.deepcopy(power_traces)
		# df = int(4/np.median(np.diff(freq)))
		# for frequency_to_rem in np.flip(np.arange(1,21)*50,axis=0):
		# 	index_to_rem = np.abs(freq-frequency_to_rem).argmin()
		# 	index_to_rem = index_to_rem + (magnitude[index_to_rem-df:index_to_rem+df]).argmax() - df
		# 	spectra[index_to_rem]=0
		# for frequency_to_rem in np.flip(np.arange(-20,0)*50,axis=0):
		# 	index_to_rem = np.abs(freq-frequency_to_rem).argmin()
		# 	index_to_rem = index_to_rem + (magnitude[index_to_rem-df:index_to_rem+df]).argmax() - df
		# 	spectra[index_to_rem]=0
		# power_traces_corrected = np.fft.ifft(spectra)
			# power_traces_corrected = power_traces_corrected-magnitude[index_to_rem]*np.cos(phase[index_to_rem] + 2 * np.pi * freq[index_to_rem] * np.arange(len(current_traces_time)) * time_resolution)
			# spectra = np.fft.fft(power_traces_corrected[:min(np.abs(current_traces_time-2).argmin(),all_peaks[0]-int(np.median(np.diff(all_peaks))/2))])
			# phase = np.angle(spectra)
			# magnitude = 2 * np.abs(spectra) / len(spectra)
		# steady_state_power = np.max(magnitude)/2

		# the way the steady state power is calculated prior 04/03/2021 does not work, the intervai i sample is just too small
		exclusion_zone_around_peak_right = 10e-3	# +/- ms
		exclusion_zone_around_peak_right = int(exclusion_zone_around_peak_right/time_resolution)
		exclusion_zone_around_peak_left = 4e-3	# +/- ms
		exclusion_zone_around_peak_left = int(exclusion_zone_around_peak_left/time_resolution)
		temp = np.arange(len(power_traces_total))
		good_times = np.ones_like(power_traces_total).astype(bool)
		for i_peak_pos,peak_pos in enumerate(all_peaks):
			good_times[np.logical_and(temp>peak_pos-exclusion_zone_around_peak_left,temp<peak_pos+exclusion_zone_around_peak_right)] = False
		steady_state_power = np.mean(power_traces_total[good_times])
		mean_steady_state_power = np.mean(power_traces_total[good_times])
		mean_steady_state_power_std = np.std(power_traces_total[good_times])

		energy_per_pulse = []
		duration_per_pulse = []
		energy_per_pulse_good_pulses = []
		duration_per_pulse_good_pulses = []
		# all_steady_state_power = []
		peak_shape = []
		current_peak_shape = []
		voltage_peak_shape = []
		target_voltage_peak_shape = []
		interval_between_pulses = np.median(np.diff(all_peaks))
		interval_to_find_peak = int(0.0004/time_resolution)
		if want_the_plot_of_mean_power_profile:
			plt.figure()
		for i_peak_pos,peak_pos in enumerate(all_peaks):
			if i_peak_pos+1 in miss:
				energy_per_pulse.append(0)
				duration_per_pulse.append(0)
			else:
				if False:	# upon inspection this method performs very poorly to identify reliably start and end of the pulse
					left = np.logical_and(np.logical_and(np.array((power_traces_total<steady_state_power)), np.arange(len(power_traces_total))>(peak_pos-calculated_interval*0.9)), np.arange(len(power_traces_total))<(peak_pos+calculated_interval*0.9)) * np.arange(len(power_traces_total))
					left = np.abs(np.array((power_traces_total<steady_state_power)) * np.arange(len(power_traces_total)) - peak_pos)
					right = left * (np.arange(len(power_traces_total))>=peak_pos)
					right[right==0] = np.max(right)
					right = right.argmin()+1
					left = left * (np.arange(len(power_traces_total))<=peak_pos)
					left[left==0] = np.max(left)
					left = left.argmin()
					# energy_per_pulse.append(np.sum(power_traces[left:right]*np.diff(current_traces_time[left:right+1])))
					energy_per_pulse.append(np.trapz(power_traces_total[left:right]-steady_state_power, current_traces_time[left:right]))
					# plt.plot(current_traces_time[left:right+100]-current_traces_time[left],power_traces[left:right+100])#,plt.plot(power_traces[left:right],'+');plt.pause(0.01)
				else:
					left = peak_pos-int(interval_between_pulses*2/3)
					right = peak_pos+int(interval_between_pulses*2/3)
					power_traces_corrected = np.array(power_traces_total[left:right])
					current_traces_corrected = np.array(current_traces_total[left:right])
					voltage_traces_corrected = np.array(voltage_traces_total[left:right])
					target_voltage_traces_corrected = np.array(target_voltage_traces_total[left:right])
					current_traces_time_corrected = np.array(current_traces_time[left:right])
					# spectra = np.fft.fft(power_traces_corrected)
					# phase = np.angle(spectra)
					# magnitude = 2 * np.abs(spectra) / len(spectra)
					# freq = np.fft.fftfreq(len(magnitude), d=time_resolution)
					# df = int(50/np.median(np.diff(freq)))
					# steady_state_power = np.max(magnitude)/2
					# for frequency_to_rem in np.flip([10000,20000],axis=0):
					# 	index_to_rem = np.abs(freq-frequency_to_rem).argmin()
					# 	index_to_rem = index_to_rem + (magnitude[index_to_rem-df:index_to_rem+df]).argmax() - df
					# 	power_traces_corrected = power_traces_corrected-magnitude[index_to_rem]*np.cos(phase[index_to_rem] + 2 * np.pi * freq[index_to_rem] * np.arange(len(power_traces_corrected)) * time_resolution)
					# tck = interpolate.splrep(current_traces_time[left:right], power_traces_corrected,s=4000000000,k=2)
					# interpolated = interpolate.splev(current_traces_time[left:right], tck)
					interpolated = np.convolve(power_traces_corrected, np.ones((interval_to_find_peak))/interval_to_find_peak , mode='same')
					real_peak_loc = np.arange(left,right)[interpolated.argmax()]
					# steady_state_power = np.mean([*power_traces_total[real_peak_loc-int(2e-3/time_resolution):real_peak_loc-int(1e-3/time_resolution)]])#,*power_traces_corrected[real_peak_loc+int(10e-3/time_resolution):real_peak_loc+int(13e-3/time_resolution)]])
					# plt.figure();plt.plot(current_traces_time[left:right]-current_traces_time[real_peak_loc],power_traces_corrected-steady_state_power);plt.plot(current_traces_time[left:right]-current_traces_time[real_peak_loc],interpolated-steady_state_power,'--');plt.grid();plt.pause(0.01)
					real_peak_loc = interpolated.argmax()
					if False:	# I want the real start and end of the pulse, this approximation is not enough
						left = real_peak_loc-int(1e-3/time_resolution)
						right = real_peak_loc+int(1.5e-3/time_resolution)
					else:
						left = real_peak_loc - (np.flip(power_traces_corrected[:real_peak_loc],axis=0)-steady_state_power>0).argmin()
						if (power_traces_corrected[left]-steady_state_power)>(power_traces_corrected.max()-steady_state_power)*0.1:
							left -= 1
						right = real_peak_loc + (power_traces_corrected[real_peak_loc:]-steady_state_power>0).argmin()
						if (power_traces_corrected[right-1]-steady_state_power)>(power_traces_corrected.max()-steady_state_power)*0.1:
							right += 1
						# plt.plot(current_traces_time[left:right]-current_traces_time[real_peak_loc],power_traces_corrected[left:right]-steady_state_power,'o');plt.pause(0.01)
					# steady_state_power = np.mean([*power_traces_corrected[real_peak_loc-int(5e-3/time_resolution):real_peak_loc-int(1e-3/time_resolution)]])#,*power_traces_corrected[real_peak_loc+int(10e-3/time_resolution):real_peak_loc+int(13e-3/time_resolution)]])
					# energy_per_pulse.append(np.trapz(power_traces_corrected, current_traces_time[left:right])-np.mean([power_traces_corrected[:int(np.median(np.diff(all_peaks))*1/2)],power_traces_corrected[-int(np.median(np.diff(all_peaks))*1/2):]])*(current_traces_time[right]-current_traces_time[left]))
					if False:	# modified because in the energy of the pulse it is absolutely included also the energy of the SS
						energy_per_pulse.append(np.trapz(power_traces_corrected[real_peak_loc-int(1e-3/time_resolution):real_peak_loc+int(1.5e-3/time_resolution)], current_traces_time[left:right])-steady_state_power*(current_traces_time[right]-current_traces_time[left]))
					else:
						energy_per_pulse.append(np.trapz(power_traces_corrected[left:right], current_traces_time_corrected[left:right]))
						duration_per_pulse.append(current_traces_time_corrected[right-1]-current_traces_time_corrected[left])
					# all_steady_state_power.append(steady_state_power)
					if not (i_peak_pos+1 in double):
						peak_shape.append(power_traces_corrected[real_peak_loc-int(interval_between_pulses*0.6):real_peak_loc+int(interval_between_pulses*0.6)]-steady_state_power)
						current_peak_shape.append(current_traces_corrected[real_peak_loc-int(interval_between_pulses*0.6):real_peak_loc+int(interval_between_pulses*0.6)])
						voltage_peak_shape.append(voltage_traces_corrected[real_peak_loc-int(interval_between_pulses*0.6):real_peak_loc+int(interval_between_pulses*0.6)])
						target_voltage_peak_shape.append(target_voltage_traces_corrected[real_peak_loc-int(interval_between_pulses*0.6):real_peak_loc+int(interval_between_pulses*0.6)])
						# plt.plot(current_traces_time[left+real_peak_loc-int(interval_between_pulses*0.6):left+real_peak_loc+int(interval_between_pulses*0.6)]-time_between_pulses*i_peak_pos,power_traces_corrected[real_peak_loc-int(interval_between_pulses*0.6):real_peak_loc+int(interval_between_pulses*0.6)]-steady_state_power)#,plt.plot(power_traces[left:right],'+');plt.pause(0.01)
						energy_per_pulse_good_pulses.append(np.trapz(power_traces_corrected[left:right], current_traces_time_corrected[left:right]))
						duration_per_pulse_good_pulses.append(current_traces_time_corrected[right-1]-current_traces_time_corrected[left])
						if want_the_plot_of_mean_power_profile:
							plt.plot(np.array(power_traces_corrected[real_peak_loc-int(interval_between_pulses*0.6):real_peak_loc+int(interval_between_pulses*0.6)]-steady_state_power))#,plt.plot(power_traces[left:right],'+');plt.pause(0.01)
							plt.plot(np.array(interpolated[real_peak_loc-int(interval_between_pulses*0.6):real_peak_loc+int(interval_between_pulses*0.6)]-steady_state_power),'--')#,plt.plot(power_traces[left:right],'+');plt.pause(0.01)
							# plt.plot(np.arange(len(np.array(power_traces_corrected[real_peak_loc-int(interval_between_pulses*0.6):real_peak_loc+int(interval_between_pulses*0.6)]-steady_state_power)))[real_peak_loc],interpolated.max(),'+')
		mean_peak_shape = np.mean(peak_shape,axis=0)
		mean_current_peak_shape = np.mean(current_peak_shape,axis=0)
		mean_voltage_peak_shape = np.mean(voltage_peak_shape,axis=0)
		mean_target_voltage_peak_shape = np.mean(target_voltage_peak_shape,axis=0)
		mean_peak_std = np.std(peak_shape,axis=0)
		mean_current_peak_std = np.std(current_peak_shape,axis=0)
		mean_voltage_peak_std = np.std(voltage_peak_shape,axis=0)
		mean_target_voltage_peak_std = np.std(target_voltage_peak_shape,axis=0)
		# mean_steady_state_power = np.mean(all_steady_state_power)
		# mean_steady_state_power_std = np.std(all_steady_state_power)
		if want_the_plot_of_mean_power_profile:
			plt.errorbar(np.arange(len(mean_peak_shape)),mean_peak_shape,yerr=mean_peak_std,color='k')
			plt.title(fdir_long+'\n'+fname_current_trace)
			plt.grid()
			plt.pause(0.01)
		try:	# this extra is to get a window around the pulse from the start of the pulse and not the peak
			left = mean_peak_shape.argmax() - int(2e-3/time_resolution)
			pulse_start = np.abs(np.cumsum(mean_peak_shape[left:]) - np.cumsum(mean_peak_shape[left:]).max()/50).argmin()	# 50 is arbitrary
			mean_peak_shape = mean_peak_shape[left+pulse_start-int(interval_between_pulses/2):left+pulse_start+int(interval_between_pulses/2)]
			mean_current_peak_shape = mean_current_peak_shape[left+pulse_start-int(interval_between_pulses/2):left+pulse_start+int(interval_between_pulses/2)]
			mean_voltage_peak_shape = mean_voltage_peak_shape[left+pulse_start-int(interval_between_pulses/2):left+pulse_start+int(interval_between_pulses/2)]
			mean_target_voltage_peak_shape = mean_target_voltage_peak_shape[left+pulse_start-int(interval_between_pulses/2):left+pulse_start+int(interval_between_pulses/2)]
			mean_peak_std = mean_peak_std[left+pulse_start-int(interval_between_pulses/2):left+pulse_start+int(interval_between_pulses/2)]
			mean_current_peak_std = mean_current_peak_std[left+pulse_start-int(interval_between_pulses/2):left+pulse_start+int(interval_between_pulses/2)]
			mean_voltage_peak_std = mean_voltage_peak_std[left+pulse_start-int(interval_between_pulses/2):left+pulse_start+int(interval_between_pulses/2)]
			mean_target_voltage_peak_std = mean_target_voltage_peak_std[left+pulse_start-int(interval_between_pulses/2):left+pulse_start+int(interval_between_pulses/2)]
		except:
			print('failed to find the start of the averaged pulse')
		# plt.figure();
		# plt.plot(energy_per_pulse,':k');plt.pause(0.01)
		if want_the_mean_power_profile==True:
			return bad, good[0], any[0], any[-1], miss,double, good, time_of_pulses, np.array(energy_per_pulse), np.array(duration_per_pulse),np.median(energy_per_pulse_good_pulses),np.median(duration_per_pulse_good_pulses),mean_peak_shape,mean_peak_std,mean_steady_state_power,mean_steady_state_power_std,time_resolution,mean_current_peak_shape,mean_current_peak_std,mean_voltage_peak_shape,mean_voltage_peak_std,mean_target_voltage_peak_shape,mean_target_voltage_peak_std
		else:
			return bad, good[0], any[0], any[-1], miss,double, good, time_of_pulses, np.array(energy_per_pulse), np.array(duration_per_pulse),np.median(energy_per_pulse_good_pulses),np.median(duration_per_pulse_good_pulses)
	else:
		return bad, good[0], any[0], any[-1], miss, double, good, time_of_pulses


#################################################################################################################

def movie_from_data(data,framerate,integration=1,xlabel=(),ylabel=(),barlabel=(),cmap='rainbow',timesteps='auto',extvmin='auto',extvmax='auto'):

	import matplotlib.animation as animation

	fig = plt.figure()
	ax = fig.add_subplot(111)

	# I like to position my colorbars this way, but you don't have to
	# div = make_axes_locatable(ax)
	# cax = div.append_axes('right', '5%', '5%')

	# def f(x, y):
	#	 return np.exp(x) + np.sin(y)

	# x = np.linspace(0, 1, 120)
	# y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

	# This is now a list of arrays rather than a list of artists
	frames = [None]*len(data[0])
	frames[0]=data[0,0]

	for i in range(len(data[0])):
		# x	   += 1
		# curVals  = f(x, y)
		frames[i]=(data[0,i])

	cv0 = frames[0]
	im = ax.imshow(cv0,cmap, origin='lower') # Here make an AxesImage rather than contour


	cb = fig.colorbar(im).set_label(barlabel)
	cb = ax.set_xlabel(xlabel)
	cb = ax.set_ylabel(ylabel)
	tx = ax.set_title('Frame 0')


	if timesteps=='auto':
		def animate(i):
			arr = frames[i]
			if extvmax=='auto':
				vmax = np.max(arr)
			elif extvmax=='allmax':
				vmax = np.max(data)
			else:
				vmax = extvmax

			if extvmin=='auto':
				vmin = np.min(arr)
			elif extvmin=='allmin':
				vmin = np.min(data)
			else:
				vmin = extvmin
			im.set_data(arr)
			im.set_clim(vmin, vmax)
			tx.set_text('Frame {0}'.format(i)+', FR '+str(framerate)+'Hz, t '+str(np.around(0+i/framerate,decimals=3))+'s int '+str(integration)+'ms')

			# In this version you don't have to do anything to the colorbar,
			# it updates itself when the mappable it watches (im) changes
	else:
		def animate(i):
			arr = frames[i]
			if extvmax=='auto':
				vmax = np.max(arr)
			elif extvmax=='allmax':
				vmax = np.max(data)
			else:
				vmax = extvmax

			if extvmin=='auto':
				vmin = np.min(arr)
			elif extvmin=='allmin':
				vmin = np.min(data)
			else:
				vmin = extvmin
			im.set_data(arr)
			im.set_clim(vmin, vmax)
			tx.set_text('Frame {0}'.format(i)+', t '+str(timesteps[i])+'s int '+str(integration)+'ms')

	ani = animation.FuncAnimation(fig, animate, frames=len(data[0]))

	return ani


#######################################################################################################################################################

def get_angle_no_lines(data, nCh=40,bininterv_est=25):
	# nCh=40;nLines=2
	# nCh=40;nLines=7
	from matplotlib import pyplot as plt
	import numpy as np
	from scipy.signal import find_peaks, peak_prominences as get_proms
	import copy

	Nx, Ny = data.shape
	Spectrum = np.sum(data, axis=0)  # Chord-integrated spectrum
	# Spectrum.dtype = 'int'
	iNoise = Spectrum[20:].argmin() + 20
	Noise = data[:750, iNoise];
	Noise = max(Noise) - min(Noise)
	iBright = Spectrum[20:].argmax() + 20
	RHM = 30
	LHM = 30
	dw = np.shape(data)[1] / 20
	num_points = 5
	iBins = [[] for i in range(num_points)]  # Vertical position of chord edge
	min_wave = dw

	max_wave = int(Ny - dw)
	test=1
	while test!=39:
		vertData = np.mean(data[:, max_wave - LHM:max_wave + RHM], axis=1)
		# plt.figure();plt.plot(vertData);plt.pause(0.01)
		binPeaks = find_peaks(-vertData, distance=bininterv_est)[0]
		binProms = get_proms(-vertData, binPeaks)[0]
		high_Bins = sorted(binPeaks[np.argpartition(-binProms, nCh - 2)[:nCh - 1]])
		diffI = np.diff(high_Bins)
		iM = min(vertData.argmax(), high_Bins[-2])
		iiM = [iM < foo for foo in high_Bins].index(1)
		try:
			i0 = iiM - list(diffI[iiM::-1] > 1.2 * np.median(diffI)).index(1) + 1
		except ValueError:
			i0 = None
		try:
			ie = iiM + list(diffI[iiM:] > 1.2 * np.median(diffI)).index(1) + 1
		except ValueError:
			ie = None
		test=len(high_Bins[i0:ie])
		max_wave-=int(dw/2)

	test = 1
	while test != 2:
		wavel_points = np.linspace(min_wave, max_wave, num_points).astype(int)
		for index, iLine in enumerate(wavel_points):
			vertData = np.mean(data[:, iLine - LHM:iLine + RHM], axis=1)
			# plt.figure();plt.plot(vertData);plt.pause(0.01)
			binPeaks = find_peaks(-vertData, distance=bininterv_est)[0]
			binProms = get_proms(-vertData, binPeaks)[0]

			iBins[index] = sorted(binPeaks[np.argpartition(-binProms, nCh - 2)[:nCh - 1]])
			diffI = np.diff(iBins[index])

			iM = min(vertData.argmax(), iBins[index][-2])
			# print(iBins[i])
			# print(iM)
			iiM = [iM < foo for foo in iBins[index]].index(1)
			try:
				i0 = iiM - list(diffI[iiM::-1] > 1.2 * np.median(diffI)).index(1) + 1
			except ValueError:
				i0 = None
			try:
				ie = iiM + list(diffI[iiM:] > 1.2 * np.median(diffI)).index(1) + 1
			except ValueError:
				ie = None
			iBins[index] = iBins[index][i0:ie]
		test = len(np.shape(iBins))
		print(min_wave)
		min_wave += dw
	# if len(np.shape(iBins))==2:	   # this should be done!!!!
	iBins = np.array(iBins)
	centre_location = np.mean(iBins, axis=(-1))
	central_bin = np.abs(np.array(iBins[-1]) - Nx / 2).argmin()
	binFit = np.polyfit(wavel_points, iBins[:, central_bin], 1)
	# iFit = np.polyval(binFit, range(len(iBins[i])))
	phi = np.arctan(binFit[0]) * 180 / np.pi

	return phi



#######################################################################################################################################################

def do_tilt_no_lines(data, nCh=40,bininterv_est='auto',return_4_points=False):
	# nCh=40;nLines=2
	# nCh=40;nLines=7
	from matplotlib import pyplot as plt
	import numpy as np
	from scipy.signal import find_peaks, peak_prominences as get_proms
	import copy

	Nx, Ny = data.shape
	Spectrum = np.sum(data, axis=0)  # Chord-integrated spectrum
	# Spectrum.dtype = 'int'
	iNoise = Spectrum[20:].argmin() + 20
	Noise = data[:750, iNoise];
	Noise = max(Noise) - min(Noise)
	iBright = Spectrum[20:].argmax() + 20
	RHM = 30
	LHM = 30
	dw = np.shape(data)[1] / 20
	num_points = 5
	# iBins = [[] for i in range(num_points)]  # Vertical position of chord edge
	min_wave = dw

	max_wave = int(Ny - dw)
	test=1
	while test!=39:
		vertData = np.mean(data[:, max_wave - LHM:max_wave + RHM], axis=1)
		# plt.figure();plt.plot(vertData);plt.pause(0.01)
		if bininterv_est!='auto':
			binPeaks = find_peaks(-vertData, distance=bininterv_est)[0]
			binProms = get_proms(-vertData, binPeaks)[0]
		else:	# introduced 21/06/2020
			temp = []
			for presumed_distance in [19,20,21,22,23,24,25]:
				# temp.append(np.mean(get_proms(-vertData, find_peaks(-vertData, distance=presumed_distance)[0])[0]))
				peaks = find_peaks(-vertData, distance=presumed_distance)[0]
				temp.append(np.mean(get_proms(-vertData, peaks[np.logical_and(peaks>15,peaks<Nx-15)])[0])*(len(peaks[np.logical_and(peaks>15,peaks<Nx-15)])==nCh-1))
			presumed_distance = [19,20,21,22,23,24,25][np.array(temp).argmax()]
			binPeaks = find_peaks(-vertData, distance=presumed_distance)[0]
			binProms = get_proms(-vertData, binPeaks)[0]
		high_Bins = sorted(binPeaks[np.argpartition(-binProms, nCh - 2)[:nCh - 1]])
		diffI = np.diff(high_Bins)
		iM = min(vertData.argmax(), high_Bins[-2])
		iiM = [iM < foo for foo in high_Bins].index(1)
		try:
			i0 = iiM - list(diffI[iiM::-1] > 1.2 * np.median(diffI)).index(1) + 1
		except ValueError:
			i0 = None
		try:
			ie = iiM + list(diffI[iiM:] > 1.2 * np.median(diffI)).index(1) + 1
		except ValueError:
			ie = None
		test=len(high_Bins[i0:ie])
		max_wave-=int(dw/2)

	test = 1
	while test != 2:
		iBins = [[] for i in range(num_points)]  # Vertical position of chord edge
		wavel_points = np.linspace(min_wave, max_wave, num_points).astype(int)
		for index, iLine in enumerate(wavel_points):
			vertData = np.mean(data[:, iLine - LHM:iLine + RHM], axis=1)
			# plt.figure();plt.plot(vertData);plt.pause(0.01)
			if bininterv_est!='auto':
				binPeaks = find_peaks(-vertData, distance=bininterv_est)[0]
				presumed_distance = bininterv_est
				while ( len(binPeaks) < (nCh - 2) and presumed_distance>3 ):
					presumed_distance -= 2
					binPeaks = find_peaks(-vertData, distance=presumed_distance)[0]
				binProms = get_proms(-vertData, binPeaks)[0]
			else:	# introduced 21/06/2020
				temp = []
				for presumed_distance in [19,20,21,22,23,24,25]:
					# temp.append(np.mean(get_proms(-vertData, find_peaks(-vertData, distance=presumed_distance)[0])[0]))
					peaks = find_peaks(-vertData, distance=presumed_distance)[0]
					temp.append(np.mean(get_proms(-vertData, peaks[np.logical_and(peaks>15,peaks<Nx-15)])[0])*(len(peaks[np.logical_and(peaks>15,peaks<Nx-15)])==nCh-1))
				presumed_distance = [19,20,21,22,23,24,25][np.array(temp).argmax()]
				binPeaks = find_peaks(-vertData, distance=presumed_distance)[0]
				binProms = get_proms(-vertData, binPeaks)[0]

			iBins[index] = sorted(binPeaks[np.argpartition(-binProms, nCh - 2)[:nCh - 1]])
			diffI = np.diff(iBins[index])

			iM = min(vertData.argmax(), iBins[index][-2])
			# print(iBins[i])
			# print(iM)
			iiM = [iM < foo for foo in iBins[index]].index(1)
			try:
				i0 = iiM - list(diffI[iiM::-1] > 1.2 * np.median(diffI)).index(1) + 1
			except ValueError:
				i0 = None
			try:
				ie = iiM + list(diffI[iiM:] > 1.2 * np.median(diffI)).index(1) + 1
			except ValueError:
				ie = None
			iBins[index] = iBins[index][i0:ie]
		test = len(np.shape(iBins))
		if test !=2:
			print('extra check done, len(iBins[0])=' + str(len(iBins[0])))
			if len(iBins[0])==nCh-1:
				temp1 = []
				temp2 = []
				for index2 in range(len(iBins)):
					if len(iBins[index2])==nCh-1:
						temp1.append(iBins[index2])
						temp2.append(wavel_points[index2])
				if len(temp1)>=3:
					iBins = np.array(temp1)
					test = len(np.shape(iBins))
					wavel_points = np.array(temp2)
		if test==2:
			if np.abs(np.array(iBins) - np.median(np.array(iBins),axis=0)).max()>10:	#this is to avoid the unfortunate case that len(np.shape(iBins)) just by chance
				test=1
		print(min_wave)
		min_wave += dw
	# if len(np.shape(iBins))==2:	   # this should be done!!!!
	print('shape of iBins ' +str(np.shape(iBins)))
	iBins = np.array(iBins)

	# shift = np.mean(iBins, axis=1)
	# shift -= shift[0]
	# binInterv,bin0 = np.polyfit(np.tile(np.arange(1,nCh),len(wavel_points)),(iBins-np.repeat([shift],nCh-1,axis=0).T).flatten(),1)
	# tilt,bin00 = np.polyfit(np.array(wavel_points),shift+bin0,1)
	# # tilt = np.arctan(tilt) * 180 / np.pi
	#
	# # tilt = -tilt
	# sgna = do_tilt(data,tilt)
	# plt.figure();plt.plot(data[:,1500]);plt.pause(0.0001)
	# plt.plot(data[:,500]);plt.pause(0.0001)
	#
	# plt.plot(sgna[:,1500]);plt.pause(0.0001)
	# plt.plot(sgna[:,500]);plt.pause(0.0001)
	#
	# plt.figure();plt.imshow(sgna);plt.pause(0.01)



	iBins=iBins.tolist()
	for index in range(len(iBins)):
		interv = np.mean(np.diff(iBins[index]))
		iBins[index].append(np.min(iBins[index])-3*interv)		#I suppose that there is always enough pixels for this
		iBins[index].append(np.max(iBins[index])+3*interv)
	iBins = np.array(iBins)
	iBins = np.sort(iBins,axis=(-1))
	binFit = np.polyfit(wavel_points, iBins[:, 0], 1)
	iFit_down = np.polyval(binFit, [0,Ny])
	binFit = np.polyfit(wavel_points, iBins[:, -1], 1)
	iFit_up = np.polyval(binFit, [0,Ny])

	pts_1 = np.array([(iFit_up[0],0),(iFit_up[-1],Ny),(iFit_down[0],0),(iFit_down[-1],Ny)])
	pts_1 = np.flip(pts_1,axis=(1))

	if return_4_points==True:
		return pts_1

	print('Points for 4 points deformation '+str(pts_1))

	data = four_point_transform(data, pts_1)

	return data



#####################################################################################################################################################################

def four_point_transform(image, pts_1):
	import numpy as np
	import cv2

	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts_1)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))


	dst = np.array([
		[0, 0],
		[maxWidth , 0],
		[maxWidth , maxHeight ],
		[0, maxHeight ]], dtype="float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# FF 2019/08/22
	# I want to account for the fact that I do not want to conserve the color, but the amount of light that arrives on the camera sensor
	multiplier = np.linspace((bl[1]-tl[1]),(br[1]-tr[1]),maxWidth)
	multiplier = multiplier / maxHeight

	# return the warped image
	return warped * multiplier

######################################################################################################################################################################

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype="float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

######################################################################################################################################################################


def get_bin_and_interv_no_lines(data, nCh=40,bininterv_est='auto'):
	# nCh=40;nLines=2
	# nCh=40;nLines=7
	from matplotlib import pyplot as plt
	import numpy as np
	from scipy.signal import find_peaks, peak_prominences as get_proms
	import copy

	Nx, Ny = data.shape
	Spectrum = np.sum(data, axis=0)  # Chord-integrated spectrum
	# Spectrum.dtype = 'int'
	iNoise = Spectrum[20:].argmin() + 20
	Noise = data[:750, iNoise];
	Noise = max(Noise) - min(Noise)
	iBright = Spectrum[20:].argmax() + 20
	RHM = 30
	LHM = 30
	dw = np.shape(data)[1] / 20
	num_points = 5
	# iBins = [[] for i in range(num_points)]  # Vertical position of chord edge
	min_wave = dw

	max_wave = int(Ny - dw)
	test=1
	while test!=39:
		vertData = np.mean(data[:, max_wave - LHM:max_wave + RHM], axis=1)
		# plt.figure();plt.plot(vertData);plt.pause(0.01)
		if bininterv_est!='auto':
			binPeaks = find_peaks(-vertData, distance=bininterv_est)[0]
			binProms = get_proms(-vertData, binPeaks)[0]
		else:	# introduced 21/06/2020
			temp = []
			for presumed_distance in [19,20,21,22,23,24,25]:
				# temp.append(np.mean(get_proms(-vertData, find_peaks(-vertData, distance=presumed_distance)[0])[0]))
				peaks = find_peaks(-vertData, distance=presumed_distance)[0]
				temp.append(np.mean(get_proms(-vertData, peaks)[0])*(len(peaks)==nCh-1))
			presumed_distance = [19,20,21,22,23,24,25][np.array(temp).argmax()]
			# presumed_distance = [19,20,21,22,23,24,25][np.abs(np.array(temp)-nCh-1).argmin()]
			binPeaks = find_peaks(-vertData, distance=presumed_distance)[0]
			binProms = get_proms(-vertData, binPeaks)[0]
		high_Bins = sorted(binPeaks[np.argpartition(-binProms, nCh - 2)[:nCh - 1]])
		diffI = np.diff(high_Bins)
		iM = min(vertData.argmax(), high_Bins[-2])
		iiM = [iM < foo for foo in high_Bins].index(1)
		try:
			i0 = iiM - list(diffI[iiM::-1] > 1.2 * np.median(diffI)).index(1) + 1
		except ValueError:
			i0 = None
		try:
			ie = iiM + list(diffI[iiM:] > 1.2 * np.median(diffI)).index(1) + 1
		except ValueError:
			ie = None
		test=len(high_Bins[i0:ie])
		max_wave-=int(dw/2)

	test = 1
	while test != 2:
		iBins = [[] for i in range(num_points)]  # Vertical position of chord edge
		wavel_points = np.linspace(min_wave, Ny - dw, num_points).astype(int)
		for index, iLine in enumerate(wavel_points):
			vertData = np.mean(data[:, iLine - LHM:iLine + RHM], axis=1)
			# plt.figure();plt.plot(vertData);plt.pause(0.01)
			if bininterv_est!='auto':
				binPeaks = find_peaks(-vertData, distance=bininterv_est)[0]
				presumed_distance = 25
				while (len(binPeaks) < (nCh - 2) and presumed_distance>3):
					presumed_distance -= 2
					binPeaks = find_peaks(-vertData, distance=presumed_distance)[0]
				binProms = get_proms(-vertData, binPeaks)[0]
			else:	# introduced 21/06/2020
				temp = []
				for presumed_distance in [19,20,21,22,23,24,25]:
					# temp.append(np.mean(get_proms(-vertData, find_peaks(-vertData, distance=presumed_distance)[0])[0]))
					peaks = find_peaks(-vertData, distance=presumed_distance)[0]
					temp.append(np.mean(get_proms(-vertData, peaks)[0])*(len(peaks)==nCh-1))
				presumed_distance = [19,20,21,22,23,24,25][np.array(temp).argmax()]
				binPeaks = find_peaks(-vertData, distance=presumed_distance)[0]
				binProms = get_proms(-vertData, binPeaks)[0]

			iBins[index] = sorted(binPeaks[np.argpartition(-binProms, min(len(binPeaks)-1,nCh - 2))[:nCh - 1]])
			diffI = np.diff(iBins[index])

			iM = min(vertData.argmax(), iBins[index][-2])
			# print(iBins[i])
			# print(iM)
			iiM = [iM < foo for foo in iBins[index]].index(1)
			try:
				i0 = iiM - list(diffI[iiM::-1] > 1.2 * np.median(diffI)).index(1) + 1
			except ValueError:
				i0 = None
			try:
				ie = iiM + list(diffI[iiM:] > 1.2 * np.median(diffI)).index(1) + 1
			except ValueError:
				ie = None
			iBins[index] = iBins[index][i0:ie]
		test = len(np.shape(iBins))
		if test !=2:
			print('extra check done, len(iBins[0])=' + str(len(iBins[0])))
			if len(iBins[0])==nCh-1:
				temp1 = []
				temp2 = []
				for index2 in range(len(iBins)):
					if len(iBins[index2])==nCh-1:
						temp1.append(iBins[index2])
						temp2.append(wavel_points[index2])
				if len(temp1)>=3:
					iBins = np.array(temp1)
					test = len(np.shape(iBins))
					wavel_points = np.array(temp2)
		if test==2:
			if np.abs(np.array(iBins) - np.median(np.array(iBins),axis=0)).max()>10:	#this is to avoid the unfortunate case that len(np.shape(iBins)) just by chance
				test=1
		print(min_wave)
		min_wave += dw
	# if len(np.shape(iBins))==2:	   # this should be done!!!!
	iBins = np.array(iBins)

	# shift = np.mean(iBins, axis=1)
	# shift -= shift[0]
	# binInterv,bin0 = np.polyfit(np.tile(np.arange(1,nCh),len(wavel_points)),(iBins-np.repeat([shift],nCh-1,axis=0).T).flatten(),1)
	# tilt,bin00 = np.polyfit(np.array(wavel_points),shift+bin0,1)
	# # tilt = np.arctan(tilt) * 180 / np.pi
	#
	# # tilt = -tilt
	# sgna = do_tilt(data,tilt)
	# plt.figure();plt.plot(data[:,1500]);plt.pause(0.0001)
	# plt.plot(data[:,500]);plt.pause(0.0001)
	#
	# plt.plot(sgna[:,1500]);plt.pause(0.0001)
	# plt.plot(sgna[:,500]);plt.pause(0.0001)
	#
	# plt.figure();plt.imshow(sgna);plt.pause(0.01)

	bin_interv = np.mean(np.diff(iBins),axis=(0,1))
	first_bin = np.mean(iBins[:,0])-bin_interv


	return first_bin, bin_interv

################################################################################################################################################

def get_bin_and_interv_specific_wavelength(data,wavelength_column, nCh=40,bininterv_est=25):	# bininterv_est was set to 25 26/06/2020 because the quality of the data this function is used on don't allows it
	# nCh=40;nLines=2
	# nCh=40;nLines=7
	from matplotlib import pyplot as plt
	import numpy as np
	from scipy.signal import find_peaks, peak_prominences as get_proms
	import copy

	if nCh%2!=0:
		print('get_bin_and_interv_specific_wavelength requires an even number of LOS')
		exit()

	Nx, Ny = data.shape
	Spectrum = np.sum(data, axis=0)  # Chord-integrated spectrum
	# Spectrum.dtype = 'int'
	iNoise = Spectrum[20:].argmin() + 20
	Noise = data[:750, iNoise];
	Noise = max(Noise) - min(Noise)
	iBright = Spectrum[20:].argmax() + 20
	RHM = 30
	LHM = 30

	vertData = np.mean(data[:, max(0,wavelength_column - LHM):min(wavelength_column + RHM,Ny)], axis=1)
	if bininterv_est!='auto':
		test=1
		while test!=nCh-1:
			print('expected interval of '+str(bininterv_est))
			# plt.figure();plt.plot(vertData);plt.pause(0.01)
			binPeaks = find_peaks(-vertData, distance=bininterv_est)[0]
			binProms = get_proms(-vertData, binPeaks)[0]
			if len(binPeaks)<nCh:
				bininterv_est-=1
				if bininterv_est==0:
					print('search of '+str(nCh)+' LOS at wavelength '+str(wavelength_column)+' failed')
					# return np.nan,np.nan
				continue
			Bins = sorted(binPeaks[np.argpartition(-binProms, nCh - 2)[:nCh - 1]])
			diffI = np.diff(Bins)
			iM = min(vertData.argmax(), Bins[-2])
			iiM = [iM < foo for foo in Bins].index(1)
			try:
				i0 = iiM - list(diffI[iiM::-1] > 1.2 * np.median(diffI)).index(1) + 1
			except ValueError:
				i0 = None
			try:
				ie = iiM + list(diffI[iiM:] > 1.2 * np.median(diffI)).index(1) + 1
			except ValueError:
				ie = None
			test=len(Bins[i0:ie])
			bininterv_est-=1
			if bininterv_est==0:
				print('search of '+str(nCh)+' LOS at wavelength '+str(wavelength_column)+' failed')
				return np.nan,np.nan
	else:	# introduced 21/06/2020
		temp = []
		for presumed_distance in [19,20,21,22,23,24,25]:
			# temp.append(np.mean(get_proms(-vertData, find_peaks(-vertData, distance=presumed_distance)[0])[0]))
			peaks = find_peaks(-vertData, distance=presumed_distance)[0]
			temp.append(np.mean(get_proms(-vertData, peaks)[0])*(len(peaks)==nCh-1))
		presumed_distance = [19,20,21,22,23,24,25][np.array(temp).argmax()]
		binPeaks = find_peaks(-vertData, distance=presumed_distance)[0]
		binProms = get_proms(-vertData, binPeaks)[0]
		Bins = sorted(binPeaks[np.argpartition(-binProms, nCh - 2)[:nCh - 1]])
		diffI = np.diff(Bins)
		iM = min(vertData.argmax(), Bins[-2])
		iiM = [iM < foo for foo in Bins].index(1)
		try:
			i0 = iiM - list(diffI[iiM::-1] > 1.2 * np.median(diffI)).index(1) + 1
		except ValueError:
			i0 = None
		try:
			ie = iiM + list(diffI[iiM:] > 1.2 * np.median(diffI)).index(1) + 1
		except ValueError:
			ie = None
		test=len(Bins[i0:ie])
		if test!=nCh-1:
			print('search of '+str(nCh)+' LOS at wavelength '+str(wavelength_column)+' failed')
			return np.nan,np.nan


	bin_interv = np.mean(np.diff(Bins[i0:ie]))
	first_bin = np.mean(Bins[i0:ie])-bin_interv*nCh/2	#this only works with an even number of LOS

	return first_bin, bin_interv

################################################################################################################################################

def fix_minimum_signal(data, treshold=500):
	import numpy as np

	data=np.array(data)
	Ny = np.shape(data)[1]
	# noise = data[:20].tolist() + data[-20:].tolist()
	# noise_std = np.std(noise, axis=(0, 1))
	# noise = np.mean(noise, axis=(0, 1))

	# noise =np.sort(np.array(data[:20].tolist() + data[-20:].tolist()).flatten())[:int(len(np.array(data[:20].tolist() + data[-20:].tolist()).flatten()) * treshold/(Ny**2))]
	# noise_std = np.std(noise)
	# noise = np.mean(noise)
	# print(noise)
	factor_std = 0
	noise=0

	# THIS SECTION SEEMS NOW NOT NECESSARY
	# std_record = []
	# while factor_std <= 0.2:
	# 	reconstruct = []
	# 	for irow, row in enumerate(data.tolist()):
	# 		correction = noise - np.mean(np.sort(row)[:int(len(row) / treshold)]) + factor_std * np.std(np.sort(row)[:int(len(row) / treshold)])
	# 		# print(np.std(np.sort(row)[:int(len(row) / treshold)]))
	# 		# minimum = np.min(row)
	# 		if correction<0:
	# 			correction=0
	# 		reconstruct.append(np.array(row) + correction)
	# 	# print(correction)
	# 	reconstruct = np.array(reconstruct)
	# 	test_std = np.sum(reconstruct, axis=(0))
	# 	test_std = test_std.argmin()
	# 	if test_std<=20:
	# 		test_std=20
	# 	elif test_std>np.shape(data)[1]-20:
	# 		test_std = np.shape(data)[1] - 20
	# 	test_std = np.std(reconstruct[:, test_std-20:test_std+20])
	# 	std_record.append(test_std)
	# 	print('factor_std ' + str(factor_std))
	# 	print('test_std ' + str(test_std))
	# 	factor_std += 0.02
	# factor_std = np.array(std_record).argmin() * 0.02
	reconstruct = []
	for irow, row in enumerate(data.tolist()):
		correction = noise - np.mean(np.sort(row)[:int(len(row) * treshold/(Ny*10))]) + factor_std * np.std(np.sort(row)[:int(len(row) * treshold/(Ny*100))])
		# correction = noise - np.mean(np.sort(row)[:int(len(row) / treshold)]) + factor_std * np.std(np.sort(row)[:int(len(row) / treshold)])
		# minimum = np.mean(np.sort(row)[:int(len(row) / treshold)]) + factor_std * np.std(np.sort(row)[:int(len(row) / treshold)])
		# minimum = np.min(row)
		if correction < 0:
			correction = 0
		# print(irow)
		# print(correction)
		reconstruct.append(np.array(row) + correction)
	data = np.array(reconstruct)

	inoise = np.sum(np.array(data[:20,10:-10].tolist() + data[-20:,10:-10].tolist()),axis=0).argmin()+10
	# noise =np.sort(np.array(data[:20].tolist() + data[-20:].tolist()).flatten())[:int(len(np.array(data[:20].tolist() + data[-20:].tolist()).flatten()) * treshold/(Ny*400))]
	noise =np.sort(np.array(data[:20,inoise-10:inoise+10].tolist() + data[-20:,inoise-10:inoise+10].tolist()).flatten())
	noise = np.mean(noise)

	# noise = data[:20].tolist() + data[-20:].tolist()
	# noise = np.mean(noise, axis=(0, 1))
	data -= noise

	# noise = np.mean(np.sort(data.flatten())[:int(len(data.flatten())*500 / (1608**2))])
	# data -= min(np.min(noise), 0)


	return data

#############################################################################################################################################

def fix_minimum_signal2_old(data, treshold=500):	#this became old 08/08/2019
	import numpy as np

	data=np.array(data)
	Ny = np.shape(data)[1]

	ipeak = np.convolve(np.sum(data[10:-10,10:-10],axis=0), np.ones((20,))/20, mode='valid').argmax()+10
	ipeak = np.convolve(np.sum(data[40:-40,ipeak-10:ipeak+10],axis=1), np.ones((20,))/20, mode='valid').argmax()+40
	inoise = np.convolve(np.sum(data[ipeak-10:ipeak+10,10:-10],axis=0), np.ones((30,))/30, mode='valid').argmin()+10

	# noise =np.sort(np.array(data[:20].tolist() + data[-20:].tolist()).flatten())[:int(len(np.array(data[:20].tolist() + data[-20:].tolist()).flatten()) * treshold/(Ny*400))]
	noise =np.sort(np.array(data[:20,inoise-10:inoise+10].tolist() + data[-20:,inoise-10:inoise+10].tolist()).flatten())

	# inoise = np.sum(np.array(data[:20,10:-10].tolist() + data[-20:,10:-10].tolist()),axis=0).argmin()+10
	# noise =np.sort(np.array(data[:20].tolist() + data[-20:].tolist()).flatten())[:int(len(np.array(data[:20].tolist() + data[-20:].tolist()).flatten()) * treshold/(Ny*400))]
	noise =np.sort(np.array(data[:20,inoise-10:inoise+10].tolist() + data[-20:,inoise-10:inoise+10].tolist()).flatten())


	# noise = data[:20].tolist() + data[-20:].tolist()
	# noise_std = np.std(noise, axis=(0, 1))
	# noise = np.mean(noise, axis=(0, 1))

	# noise =np.sort(np.array(data[:20].tolist() + data[-20:].tolist()).flatten())[:int(len(np.array(data[:20].tolist() + data[-20:].tolist()).flatten()) * treshold/(Ny**2))]
	# noise_std = np.std(noise)
	noise = np.mean(noise)
	# print(noise)
	factor_std = 0
	noise=0

	# THIS SECTION SEEMS NOW NOT NECESSARY
	# std_record = []
	# while factor_std <= 0.2:
	# 	reconstruct = []
	# 	for irow, row in enumerate(data.tolist()):
	# 		correction = noise - np.mean(np.sort(row)[:int(len(row) / treshold)]) + factor_std * np.std(np.sort(row)[:int(len(row) / treshold)])
	# 		# print(np.std(np.sort(row)[:int(len(row) / treshold)]))
	# 		# minimum = np.min(row)
	# 		if correction<0:
	# 			correction=0
	# 		reconstruct.append(np.array(row) + correction)
	# 	# print(correction)
	# 	reconstruct = np.array(reconstruct)
	# 	test_std = np.sum(reconstruct, axis=(0))
	# 	test_std = test_std.argmin()
	# 	if test_std<=20:
	# 		test_std=20
	# 	elif test_std>np.shape(data)[1]-20:
	# 		test_std = np.shape(data)[1] - 20
	# 	test_std = np.std(reconstruct[:, test_std-20:test_std+20])
	# 	std_record.append(test_std)
	# 	print('factor_std ' + str(factor_std))
	# 	print('test_std ' + str(test_std))
	# 	factor_std += 0.02
	# factor_std = np.array(std_record).argmin() * 0.02
	reconstruct = []
	for irow, row in enumerate(data.tolist()):
		correction = noise - np.mean(row[inoise-10:inoise+10])
		# correction = noise - np.mean(np.sort(row)[:int(len(row) / treshold)]) + factor_std * np.std(np.sort(row)[:int(len(row) / treshold)])
		# minimum = np.mean(np.sort(row)[:int(len(row) / treshold)]) + factor_std * np.std(np.sort(row)[:int(len(row) / treshold)])
		# minimum = np.min(row)
		if correction < 0:
			correction = 0
		# print(irow)
		# print(correction)
		reconstruct.append(np.array(row) + correction)
	data = np.array(reconstruct)

	# inoise = np.sum(np.array(data[:20,10:-10].tolist() + data[-20:,10:-10].tolist()),axis=0).argmin()+10
	# # noise =np.sort(np.array(data[:20].tolist() + data[-20:].tolist()).flatten())[:int(len(np.array(data[:20].tolist() + data[-20:].tolist()).flatten()) * treshold/(Ny*400))]
	# noise =np.sort(np.array(data[:20,inoise-10:inoise+10].tolist() + data[-20:,inoise-10:inoise+10].tolist()).flatten())
	# noise = np.mean(noise)
	#
	# # noise = data[:20].tolist() + data[-20:].tolist()
	# # noise = np.mean(noise, axis=(0, 1))
	# data -= noise

	# noise = np.mean(np.sort(data.flatten())[:int(len(data.flatten())*500 / (1608**2))])
	# data -= min(np.min(noise), 0)


	return data


#######################################################################################################################################################


def fix_minimum_signal2(data,plot_stats=False,treshold=5,fix_done_marker=False):	#created 08/08/2019
	import numpy as np

	data=np.array(data)
	Nx,Ny = np.shape(data)

	ipeak = np.convolve(np.sum(data[10:-10,10:-10],axis=0), np.ones((20,))/20, mode='valid').argmax()+10+10
	inoise1 = np.convolve(np.sum(data[5:39,ipeak-5:ipeak+5],axis=1), np.ones((10,))/10, mode='valid').argmin()+5+5
	inoise2 = Nx - (np.convolve(np.sum(data[-39:-5, ipeak - 5:ipeak + 5], axis=1), np.ones((10,)) / 10,mode='valid').argmin() + 5 + 5)
	ipeak = np.convolve(np.sum(data[40:-40,ipeak-10:ipeak+10],axis=1), np.ones((20,))/20, mode='valid').argmax()+40+10
	ipeak = np.convolve(np.sum(data[ipeak - 10:ipeak + 10,30:-30], axis=0), np.ones((20,)) / 20,mode='valid').argmin() + 30 + 10
	noise =np.array(data[inoise1-10:inoise1+10].tolist() + data[inoise2-10:inoise2+10].tolist())
	if np.std(noise)<treshold:
		noise = np.mean(noise)

		reconstruct = []
		for irow, row in enumerate(data.tolist()):
			inoise = np.convolve(row[30:-30], np.ones((30,)) / 30,mode='valid').argmin() + 30 + 15
			correction = noise - np.mean(row[inoise-30:inoise+30])
			if correction < 0:
				correction = 0
			# print(irow)
			# print(correction)
			reconstruct.append(np.array(row) + correction)
		reconstruct = np.array(reconstruct)

		noise2 =np.array(reconstruct[inoise1-10:inoise1+10].tolist() + reconstruct[inoise2-10:inoise2+10].tolist())
		noise2 = np.mean(noise2)

		reconstruct -= noise2 - noise

		if plot_stats==True:
			plt.figure()
			plt.imshow(data,'rainbow',origin='lower',vmax=min(noise+20,np.max(data)),vmin=np.min(reconstruct))
			plt.colorbar()
			plt.plot([5,Ny-5,Ny-5,5,5],[5,5,39,39,5],'k')
			plt.plot([ipeak-20, ipeak-20, ipeak+20, ipeak+20, ipeak-20],[5, Nx - 5, Nx - 5, 5, 5], 'r')
			plt.title('Original data \n in black: mean='+str(np.mean(data[5:40,5:Ny-5]))+' std='+str(np.std(data[5:40,5:Ny-5]))+'\n in red: mean='+str(np.mean(data[5:Nx-5,ipeak-20:ipeak+20]))+' std='+str(np.std(data[5:Nx-5,ipeak-20:ipeak+20])))
			plt.pause(0.001)

			plt.figure()
			plt.imshow(reconstruct,'rainbow',origin='lower',vmax=min(noise+20,np.max(data)),vmin=np.min(reconstruct))
			plt.colorbar()
			plt.plot([5,Ny-5,Ny-5,5,5],[5,5,39,39,5],'k')
			plt.plot([ipeak-20, ipeak-20, ipeak+20, ipeak+20, ipeak-20],[5, Nx - 5, Nx - 5, 5, 5], 'r')
			plt.title('Corrected data \n in black: mean='+str(np.mean(reconstruct[5:40,5:Ny-5]))+' std='+str(np.std(reconstruct[5:40,5:Ny-5]))+'\n in red: mean='+str(np.mean(reconstruct[5:Nx-5,ipeak-20:ipeak+20]))+' std='+str(np.std(reconstruct[5:Nx-5,ipeak-20:ipeak+20])))
			plt.pause(0.001)

		if fix_done_marker==True:
			return reconstruct,True
		else:
			return reconstruct

	else:
		print('Found noise std of '+str(np.std(noise))+' , higher than the preset limit of '+str(treshold)+' , therefore the row by row fixing of the minimum value is skipped')
		noise = np.mean(noise)
		if plot_stats==True:
			plt.figure()
			plt.imshow(data,'rainbow',origin='lower',vmax=min(noise+20,np.max(data)),vmin=np.min(data))
			plt.colorbar()
			plt.plot([5,Ny-5,Ny-5,5,5],[5,5,39,39,5],'--k')
			plt.plot([ipeak-20, ipeak-20, ipeak+20, ipeak+20, ipeak-20],[5, Nx - 5, Nx - 5, 5, 5], '--b')
			plt.title('Original data, minimum signal adjust not performed \n in black: mean='+str(np.mean(data[5:40,5:Ny-5]))+' std='+str(np.std(data[5:40,5:Ny-5]))+'\n in blue: mean='+str(np.mean(data[5:Nx-5,ipeak-20:ipeak+20]))+' std='+str(np.std(data[5:Nx-5,ipeak-20:ipeak+20])))
			plt.pause(0.001)
		if fix_done_marker == True:
			return data, False
		else:
			return data




#######################################################################################################################################################


def fix_minimum_signal3(data,convolution_width = 50,treshold_max_signal_right=5,treshold_max_signal_strength=20,max_shift=5.2,force_average_noise=False):	#created 18/11/2019 to address Matthew reinke comments

	# NOTE: it seems not to work with the calibration data "sensitive, 12bit" setting of the camera because the minimum

	import numpy as np

	data=np.array(data)
	Nx,Ny = np.shape(data)

	temp=[]
	for index in range(20):
		temp.append(np.convolve(data[index], np.ones((convolution_width,)) / convolution_width, mode='valid'))
		temp.append(np.convolve(data[-index], np.ones((convolution_width,)) / convolution_width, mode='valid'))
		# temp.append(np.min(data[index]))
		# temp.append(np.min(data[-index]))
	temp = np.array(temp)
	average_noise = np.median(np.min(temp,axis=1))
	# average_noise = np.median(temp)
	if force_average_noise!=False:
		average_noise = force_average_noise


	if np.min(data)>30:
		print('ERROR, the input of this function is supposed to be dark noise substraced')
		exit()

	corrected_data = []
	for index in range(Nx):
		averaged_data = np.convolve(data[index], np.ones((convolution_width,)) / convolution_width, mode='valid')
		if averaged_data[-1]>treshold_max_signal_right or np.min(averaged_data)<average_noise:
			if np.mean(averaged_data)-np.min(averaged_data)>treshold_max_signal_strength:
				corrected_data.append(data[index]+max_shift)
			else:
				corrected_data.append(data[index]+min(max_shift,-np.min(averaged_data)+average_noise))
		# averaged_data = np.convolve(data[index], np.ones((convolution_width,)) / convolution_width, mode='valid')
		# if averaged_data[-1]>treshold_max_signal_right or np.min(data[index])<average_noise:
		# 	if np.mean(averaged_data)-np.min(averaged_data)>treshold_max_signal_strength:
		# 		corrected_data.append(data[index]+max_shift)
		# 	else:
		# 		corrected_data.append(data[index]+min(max_shift,-np.min(data[index])+average_noise))
		else:
			corrected_data.append(data[index])
	corrected_data = np.array(corrected_data)

	return corrected_data


#######################################################################################################################################################


def fix_minimum_signal4(data,dataDark,convolution_width = 100,treshold_max_signal_right=5,treshold_max_signal_strength=20,max_shift=5.2):	#created 18/11/2019 to address Matthew reinke comments
	import numpy as np

	data=np.array(data)
	Nx,Ny = np.shape(data)

	temp=[]
	for index in range(20):
		temp.append(np.convolve(dataDark[index], np.ones((convolution_width,)) / convolution_width, mode='valid'))
		temp.append(np.convolve(dataDark[-index], np.ones((convolution_width,)) / convolution_width, mode='valid'))
		# temp.append(np.min(data[index]))
		# temp.append(np.min(data[-index]))
	temp = np.array(temp)
	average_noise = np.median(np.min(temp,axis=1))
	# average_noise = np.median(temp)

	# if np.min(data)>30:
	# 	print('ERROR, the input of this function is supposed to be dark noise substraced')
	# 	exit()

	corrected_data = []
	for index in range(Nx):
		averaged_data = np.convolve(data[index], np.ones((convolution_width,)) / convolution_width, mode='valid')
		if averaged_data[-1]>treshold_max_signal_right or np.min(averaged_data)<average_noise:
			# scaling = (np.max(data[index]) - data[index])/ np.max(data[index])
			if np.mean(averaged_data)-np.min(averaged_data)>treshold_max_signal_strength:
				corrected_data.append(data[index]*min(average_noise/(average_noise-max_shift),100/(100-max_shift)))
			else:
				corrected_data.append(data[index]*min(average_noise/np.min(averaged_data),100/(100-max_shift)))
		# averaged_data = np.convolve(data[index], np.ones((convolution_width,)) / convolution_width, mode='valid')
		# if averaged_data[-1]>treshold_max_signal_right or np.min(data[index])<average_noise:
		# 	if np.mean(averaged_data)-np.min(averaged_data)>treshold_max_signal_strength:
		# 		corrected_data.append(data[index]+max_shift)
		# 	else:
		# 		corrected_data.append(data[index]+min(max_shift,-np.min(data[index])+average_noise))
		else:
			corrected_data.append(data[index])
	corrected_data = np.array(corrected_data)

	return corrected_data


#######################################################################################################################################################

def fix_minimum_signal_calibration(data,column_averaging=30,range_rows=0,dx_to_etrapolate_to=180,nCh=40,counts_treshold_fixed_increase=106,intermediate_wavelength=1200,last_wavelength=1608,tilt_intermediate_column=[0,0],tilt_last_column=[0,0]):
	# created 27/10/2019
	import numpy as np
	from scipy.ndimage import median_filter
	from scipy.interpolate.interpolate import interp1d
	from scipy.signal import savgol_filter

	Nx,Ny = (np.array(data)).shape

	if tilt_intermediate_column==[0,0]:
		tilt_intermediate_column = get_bin_and_interv_specific_wavelength(fix_minimum_signal2(data),intermediate_wavelength)
	if 	tilt_last_column==[0,0]:
		tilt_last_column = get_bin_and_interv_specific_wavelength(fix_minimum_signal2(data),last_wavelength)
	if np.sum(np.isnan([*tilt_last_column,*tilt_intermediate_column]))>0:
		print('fix_minimum_signal_calibration aborted for not detecting LOS at column 1200 or 1608')
		return np.ones((len(data)))*np.nan
	window_of_signal = [np.floor(tilt_last_column[0]).astype('int')-10,np.ceil(tilt_last_column[0]+tilt_last_column[1]*nCh+10).astype('int')+100]

	# I suppose here that the minimum signal per row is kinda flat (it mostly is)
	temp=[]
	for row_averaging in range(-range_rows,range_rows+1):
		temp.append(np.mean(data[range_rows+row_averaging:len(data)-range_rows+row_averaging, -column_averaging:],axis=-1) - np.mean(data[range_rows+row_averaging:len(data)-range_rows+row_averaging, :100],axis=-1))
	interpolator = interp1d(np.linspace(range_rows, len(data)-range_rows-1, len(data)-range_rows), np.mean(temp,axis=0),fill_value=0.)
	tilt_extapolated_column = [tilt_last_column[0] + (tilt_last_column[0]-tilt_intermediate_column[0])/(last_wavelength-intermediate_wavelength)*dx_to_etrapolate_to,tilt_last_column[1] + (tilt_last_column[1]-tilt_intermediate_column[1])/(last_wavelength-intermediate_wavelength)*dx_to_etrapolate_to]
	equivalent_counts = []
	for row in range(range_rows,len(data)-range_rows):
		# test_row = 1200
		row_distance = np.abs(tilt_extapolated_column[0] + tilt_extapolated_column[1]*np.linspace(-10,nCh+10,nCh+21).astype('int') - row)
		interested_LOS = np.array([to_sort for _, to_sort in sorted(zip(row_distance, np.linspace(-10,nCh+10,nCh+21).astype('int')))])[:2]
		B = row - (tilt_extapolated_column[0] + min(interested_LOS) *tilt_extapolated_column[1])
		b = B * tilt_last_column[1] / tilt_extapolated_column[1]
		# row_corrisponding_to_test_row = np.around((b + tilt_last_column[0] + min(interested_LOS) *tilt_last_column[1]) + 1e-10)
		row_corrisponding_to_test_row = (b + tilt_last_column[0] + min(interested_LOS) *tilt_last_column[1])
		if row_corrisponding_to_test_row<0:
			row_corrisponding_to_test_row = 0
		if row_corrisponding_to_test_row>Ny:
			row_corrisponding_to_test_row = Ny
		# equivalent_positions.append(int(row_corrisponding_to_test_row))
		equivalent_counts.append(interpolator(row_corrisponding_to_test_row))
	equivalent_counts = np.array(equivalent_counts)
	rows_samples_min_signal = np.linspace(range_rows, len(data)-range_rows-1, len(data)-range_rows,dtype=int)[equivalent_counts>6]
	if len(rows_samples_min_signal)<(2*4*int(tilt_intermediate_column[1])):
		x_new = [0,Ny]
		y_new = [0,0]
	else:
		convolution_200_columns = []
		for index in range(Nx):
			convolution_200_columns.append(np.convolve(data[index],np.ones((200))/200,mode='valid'))
		convolution_200_columns = np.array(convolution_200_columns)
		sample_min_signal = np.min(convolution_200_columns,axis=-1)[rows_samples_min_signal]+6

		x_new = savgol_filter(rows_samples_min_signal,4*int(tilt_intermediate_column[1])-1,2)
		x_new[x_new<=0] = 0
		y_new = savgol_filter(sample_min_signal,4*int(tilt_intermediate_column[1])-1,2)
		y_new[y_new<=0] = 0

	# plt.figure()
	# plt.plot(rows_samples_min_signal,sample_min_signal,'+')
	# plt.plot(x_new,y_new)
	# plt.pause(0.01)

	interpolator_min_signal = interp1d(x_new, y_new,fill_value='extrapolate')
	interpolated_min_signal = interpolator_min_signal(np.linspace(0, len(data)-1, len(data)-range_rows,dtype=int))
	interpolated_min_signal[interpolated_min_signal<100]=100

	# now I can really calculate the corrected image

	temp=[]
	for row_averaging in range(-range_rows,range_rows+1):
		temp.append(np.mean(data[range_rows+row_averaging:len(data)-range_rows+row_averaging, -column_averaging:],axis=-1) - np.mean(data[range_rows+row_averaging:len(data)-range_rows+row_averaging, :100],axis=-1) + interpolated_min_signal[range_rows+row_averaging:len(data)-range_rows+row_averaging])
	interpolator = interp1d(np.linspace(range_rows, len(data)-range_rows-1, len(data)-range_rows), np.mean(temp,axis=0))
	tilt_extapolated_column = [tilt_last_column[0] + (tilt_last_column[0]-tilt_intermediate_column[0])/(last_wavelength-intermediate_wavelength)*dx_to_etrapolate_to,tilt_last_column[1] + (tilt_last_column[1]-tilt_intermediate_column[1])/(last_wavelength-intermediate_wavelength)*dx_to_etrapolate_to]
	equivalent_counts = []
	for row in range(range_rows,len(data)-range_rows):
		# test_row = 1200
		row_distance = np.abs(tilt_extapolated_column[0] + tilt_extapolated_column[1]*np.linspace(-10,nCh+10,nCh+21)- row)
		interested_LOS = np.array([to_sort for _, to_sort in sorted(zip(row_distance, np.linspace(-10,nCh+10,nCh+21)))])[:2]
		B = row - (tilt_extapolated_column[0] + min(interested_LOS) *tilt_extapolated_column[1])
		b = B * tilt_last_column[1] / tilt_extapolated_column[1]
		# row_corrisponding_to_test_row = np.around((b + tilt_last_column[0] + min(interested_LOS) *tilt_last_column[1]) + 1e-10)
		row_corrisponding_to_test_row = (b + tilt_last_column[0] + min(interested_LOS) *tilt_last_column[1])
		if row_corrisponding_to_test_row<0:
			row_corrisponding_to_test_row = 0
		if row_corrisponding_to_test_row>Ny:
			row_corrisponding_to_test_row = Ny
		# equivalent_positions.append(int(row_corrisponding_to_test_row))
		equivalent_counts.append(interpolator(row_corrisponding_to_test_row))
	equivalent_counts = np.array(equivalent_counts)

	additive_factor=[]
	additive_factor_sigma=[]
	for row in range(range_rows,len(data)-range_rows):
		if equivalent_counts[row]>counts_treshold_fixed_increase:
			additive_factor.append(6)
		else:
			additive_factor.append(max(6*(equivalent_counts[row]-100)/(counts_treshold_fixed_increase-100),0))
		if equivalent_counts[row]>counts_treshold_fixed_increase+2:
			additive_factor_sigma.append(0.5)
		else:
			additive_factor_sigma.append(1.5)
	additive_factor = np.array(additive_factor)
	additive_factor_sigma = np.array(additive_factor_sigma)

	return additive_factor,additive_factor_sigma


#######################################################################################################################################################

def apply_proportionality_calibration(data,real_counts_calibration,camera_counts_calibration):
	# created 27/10/2019
	import numpy as np
	from scipy.interpolate.interpolate import interp1d

	interpolator = interp1d(camera_counts_calibration,real_counts_calibration,fill_value='extrapolate')

	data_corrected = interpolator(data)
	return data_corrected

#######################################################################################################################################################

def fix_minimum_signal_experiment(data,intermediate_wavelength,last_wavelength,tilt_intermediate_column,tilt_last_column,column_averaging=30,range_rows=0,dx_to_etrapolate_to=180,nCh=40,counts_treshold_fixed_increase=106):
	# created 27/10/2019
	import numpy as np
	from scipy.ndimage import median_filter
	from scipy.interpolate.interpolate import interp1d
	from scipy.signal import savgol_filter

	Nx,Ny = (np.array(data)).shape

	data[data==0]=100	# to account for recomposed experimental data

	if np.sum(np.isnan([*tilt_last_column,*tilt_intermediate_column]))>0:
		print('fix_minimum_signal_calibration aborted for not detecting LOS at column '+str(intermediate_wavelength)+' or '+str(last_wavelength))
		return np.ones((len(data)))*np.nan
	window_of_signal = [np.floor(tilt_last_column[0]).astype('int')-10,np.ceil(tilt_last_column[0]+tilt_last_column[1]*nCh+10).astype('int')+100]

	# I cannot assume that the signal is flat on the left, as I did for the calibration
	convolution_100_columns = []
	for index in range(Nx):
		convolution_100_columns.append(np.convolve(data[index],np.ones((100))/100,mode='valid'))
	convolution_100_columns = np.array(convolution_100_columns)
	temp=[]
	for row_averaging in range(-range_rows,range_rows+1):
		temp.append(np.mean(data[range_rows+row_averaging:len(data)-range_rows+row_averaging, -column_averaging:],axis=-1) - np.min(convolution_100_columns,axis=-1))
	interpolator = interp1d(np.linspace(range_rows, len(data)-range_rows-1, len(data)-range_rows), np.mean(temp,axis=0))
	tilt_extapolated_column = [tilt_last_column[0] + (tilt_last_column[0]-tilt_intermediate_column[0])/(last_wavelength-intermediate_wavelength)*dx_to_etrapolate_to,tilt_last_column[1] + (tilt_last_column[1]-tilt_intermediate_column[1])/(last_wavelength-intermediate_wavelength)*dx_to_etrapolate_to]
	equivalent_counts = []
	for row in range(range_rows,len(data)-range_rows):
		# test_row = 1200
		row_distance = np.abs(tilt_extapolated_column[0] + tilt_extapolated_column[1]*np.linspace(-10,nCh+10,nCh+21).astype('int') - row)
		interested_LOS = np.array([to_sort for _, to_sort in sorted(zip(row_distance, np.linspace(-10,nCh+10,nCh+21).astype('int')))])[:2]
		B = row - (tilt_extapolated_column[0] + min(interested_LOS) *tilt_extapolated_column[1])
		b = B * tilt_last_column[1] / tilt_extapolated_column[1]
		# row_corrisponding_to_test_row = np.around((b + tilt_last_column[0] + min(interested_LOS) *tilt_last_column[1]) + 1e-10)
		row_corrisponding_to_test_row = (b + tilt_last_column[0] + min(interested_LOS) *tilt_last_column[1])
		if row_corrisponding_to_test_row<0:
			row_corrisponding_to_test_row = 0
		if row_corrisponding_to_test_row>Ny:
			row_corrisponding_to_test_row = Ny
		# equivalent_positions.append(int(row_corrisponding_to_test_row))
		equivalent_counts.append(interpolator(row_corrisponding_to_test_row))
	equivalent_counts = np.array(equivalent_counts)
	rows_samples_min_signal = np.linspace(range_rows, len(data)-range_rows-1, len(data)-range_rows,dtype=int)[equivalent_counts>6]
	# sample_min_signal = np.min(convolution_100_columns,axis=-1)[rows_samples_min_signal]+6
	flag=0
	if len(rows_samples_min_signal)<(2*4*int(tilt_intermediate_column[1])):
		x_new = [0,Ny]
		y_new = [0,0]
		flag=1
	else:
		sample_min_signal = np.min(convolution_100_columns,axis=-1)[rows_samples_min_signal]+6
		x_new = savgol_filter(rows_samples_min_signal,4*int(tilt_intermediate_column[1])-1,2)
		x_new[x_new<=0] = 0
		y_new = savgol_filter(sample_min_signal,4*int(tilt_intermediate_column[1])-1,2)
		y_new[y_new<=0] = 0

	interpolator_min_signal = interp1d(x_new, y_new,fill_value='extrapolate')
	interpolated_min_signal = interpolator_min_signal(np.linspace(0, len(data)-1, len(data)-range_rows,dtype=int))
	interpolated_min_signal[interpolated_min_signal<100]=100

	# plt.figure()
	# plt.plot(rows_samples_min_signal,sample_min_signal,'+')
	# plt.plot(x_new,y_new)
	# plt.plot(np.linspace(0, len(data)-1, len(data)-range_rows,dtype=int),interpolated_min_signal)
	# plt.pause(0.01)

	# I want to make sure that the minimum signal I detect is realistic, meaning that it should fade to nothing toward the sides
	if ( (interpolated_min_signal.argmax() in [0,len(interpolated_min_signal)-1]) and (flag==0) ):
		for addition_to_treshold in range(1,20):
			print('treshold for the detection of the minimum signal increased by '+str(addition_to_treshold))
			rows_samples_min_signal = np.linspace(range_rows, len(data)-range_rows-1, len(data)-range_rows,dtype=int)[equivalent_counts>6+addition_to_treshold]
			# sample_min_signal = np.min(convolution_100_columns,axis=-1)[rows_samples_min_signal]+6
			if len(rows_samples_min_signal)<(2*4*int(tilt_intermediate_column[1])):
				x_new = [0,Ny]
				y_new = [0,0]
				interpolator_min_signal = interp1d(x_new, y_new,fill_value='extrapolate')
				interpolated_min_signal = interpolator_min_signal(np.linspace(0, len(data)-1, len(data)-range_rows,dtype=int))
				interpolated_min_signal[interpolated_min_signal<100]=100
				break
			else:
				sample_min_signal = np.min(convolution_100_columns,axis=-1)[rows_samples_min_signal]+6
				x_new = savgol_filter(rows_samples_min_signal,4*int(tilt_intermediate_column[1])-1,2)
				x_new[x_new<=0] = 0
				y_new = savgol_filter(sample_min_signal,4*int(tilt_intermediate_column[1])-1,2)
				y_new[y_new<=0] = 0
				interpolator_min_signal = interp1d(x_new, y_new,fill_value='extrapolate')
				interpolated_min_signal = interpolator_min_signal(np.linspace(0, len(data)-1, len(data)-range_rows,dtype=int))
				interpolated_min_signal[interpolated_min_signal<100]=100

				# plt.figure()
				# plt.plot(rows_samples_min_signal,sample_min_signal,'+')
				# plt.plot(x_new,y_new)
				# plt.plot(np.linspace(0, len(data)-1, len(data)-range_rows,dtype=int),interpolated_min_signal)
				# plt.pause(0.01)

				# if not(interpolated_min_signal.argmax() in [0,len(interpolated_min_signal)-1]):
				# 	break
				if ( (np.diff(interpolated_min_signal[:2])>=0) and (np.diff(interpolated_min_signal[-2:])<=0) ):
					break

	# now I can really calculate the corrected image

	temp=[]
	for row_averaging in range(-range_rows,range_rows+1):
		temp.append(np.mean(data[range_rows+row_averaging:len(data)-range_rows+row_averaging, -column_averaging:],axis=-1) - np.min(convolution_100_columns[range_rows+row_averaging:len(data)-range_rows+row_averaging],axis=-1) + interpolated_min_signal[range_rows+row_averaging:len(data)-range_rows+row_averaging])
	interpolator = interp1d(np.linspace(range_rows, len(data)-range_rows-1, len(data)-range_rows), np.mean(temp,axis=0))
	tilt_extapolated_column = [tilt_last_column[0] + (tilt_last_column[0]-tilt_intermediate_column[0])/(last_wavelength-intermediate_wavelength)*dx_to_etrapolate_to,tilt_last_column[1] + (tilt_last_column[1]-tilt_intermediate_column[1])/(last_wavelength-intermediate_wavelength)*dx_to_etrapolate_to]
	equivalent_counts = []
	for row in range(range_rows,len(data)-range_rows):
		# test_row = 1200
		row_distance = np.abs(tilt_extapolated_column[0] + tilt_extapolated_column[1]*np.linspace(-10,nCh+10,nCh+21)- row)
		interested_LOS = np.array([to_sort for _, to_sort in sorted(zip(row_distance, np.linspace(-10,nCh+10,nCh+21)))])[:2]
		B = row - (tilt_extapolated_column[0] + min(interested_LOS) *tilt_extapolated_column[1])
		b = B * tilt_last_column[1] / tilt_extapolated_column[1]
		# row_corrisponding_to_test_row = np.around((b + tilt_last_column[0] + min(interested_LOS) *tilt_last_column[1]) + 1e-10)
		row_corrisponding_to_test_row = (b + tilt_last_column[0] + min(interested_LOS) *tilt_last_column[1])
		if row_corrisponding_to_test_row<0:
			row_corrisponding_to_test_row = 0
		if row_corrisponding_to_test_row>Ny:
			row_corrisponding_to_test_row = Ny
		# equivalent_positions.append(int(row_corrisponding_to_test_row))
		equivalent_counts.append(interpolator(row_corrisponding_to_test_row))
	equivalent_counts = np.array(equivalent_counts)

	additive_factor=[]
	additive_factor_sigma=[]
	for row in range(range_rows,len(data)-range_rows):
		if equivalent_counts[row]>counts_treshold_fixed_increase:
			additive_factor.append(6)
		else:
			additive_factor.append(max(6*(equivalent_counts[row]-100)/(counts_treshold_fixed_increase-100),0))
		if equivalent_counts[row]>counts_treshold_fixed_increase+2:
			additive_factor_sigma.append(0.5)
		else:
			additive_factor_sigma.append(1.5)
	additive_factor = np.array(additive_factor)
	additive_factor_sigma = np.array(additive_factor_sigma)

	return additive_factor,additive_factor_sigma

############################################################################################################################################################################################

def print_all_properties(obj):
	# created 29/09/2020 function that prints all properties of an object
  for attr in dir(obj):
    print("object.%s = %r" % (attr, getattr(obj, attr)))

##################################################################################################################################################################################################

def shift_between_TS_and_power_source(merge_ID_target,plot_report=False):
	# created 30/09/2020 from TS-current trace comparison.py in order to accomodate for different times for different magnetic fields
	import numpy as np
	import os,sys
	# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())
	os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
	from functions.spectools import rotate,do_tilt, binData, get_angle, get_tilt,get_angle_2
	from functions.Calibrate import do_waveL_Calib, do_Intensity_Calib
	from functions.fabio_add import find_nearest_index,multi_gaussian,all_file_names,load_dark,find_index_of_file,get_metadata,movie_from_data,get_angle_no_lines,do_tilt_no_lines,four_point_transform,fix_minimum_signal,fix_minimum_signal2,get_bin_and_interv_no_lines,examine_current_trace
	from functions.GetSpectrumGeometry import getGeom
	from functions.SpectralFit import doSpecFit_single_frame
	from functions.GaussFitData import doLateralfit_time_tependent
	import collections
	from scipy.interpolate.interpolate import interp1d

	from PIL import Image
	import xarray as xr
	import pandas as pd
	import copy
	from scipy.optimize import curve_fit
	from scipy import interpolate
	from scipy.signal import find_peaks, peak_prominences as get_proms
	from multiprocessing import Pool,cpu_count
	# number_cpu_available = cpu_count()
	# print('Number of cores available: '+str(number_cpu_available))


	os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data/functions')
	print(os.path.abspath(os.getcwd()))
	import pyradi.ryptw as ryptw

	fdir = '/home/ffederic/work/Collaboratory/test/experimental_data'
	df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
	df_settings = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_3.csv',index_col=0)

	all_j=find_index_of_file(merge_ID_target,df_settings,df_log,only_OES=True)
	path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) #+ '_back'

	merge_time_window = [-1,2]
	power_pulse_shape_time_dependent = []
	power_pulse_shape_time_dependent_std = []
	steady_state_power = []
	steady_state_power_std = []
	for j in all_j:
		(merge_folder,sequence,fname_current_trace) = df_log.loc[j,['folder','sequence','current_trace_file']]
		sequence = int(sequence)
		bad_pulses,first_good_pulse,first_pulse,last_pulse,miss_pulses,double_pulses,good_pulses, time_of_pulses, energy_per_pulse,duration_per_pulse,median_energy_delivered_good_pulses,median_duration_good_pulses,mean_peak_shape,mean_peak_std,mean_steady_state_power,mean_steady_state_power_std,time_resolution, = examine_current_trace(fdir+'/'+merge_folder+'/'+"{0:0=2d}".format(sequence)+'/', fname_current_trace, df_log.loc[j, ['number_of_pulses']][0],want_the_power_per_pulse=True,want_the_mean_power_profile=True)
		current_traces = pd.read_csv(fdir+'/'+merge_folder+'/'+"{0:0=2d}".format(sequence)+'/'+ fname_current_trace+'.tsf',index_col=False, delimiter='\t')
		current_traces_time = current_traces['Time [s]']
		current_traces_total = current_traces['I_Src_AC [A]']
		voltage_traces_total = current_traces['U_Src_DC [V]']
		# plt.figure()
		# plt.plot(current_traces_time,-current_traces_total*voltage_traces_total/np.max(-current_traces_total*voltage_traces_total),label='power')
		# plt.plot(current_traces_time,current_traces_total/np.max(current_traces_total),label='current')
		# plt.plot(current_traces_time,-voltage_traces_total/np.max(-voltage_traces_total),label='voltage')
		# plt.legend(loc='best')
		# plt.grid()
		# plt.pause(0.001)
		power_pulse_shape_time_dependent.append(mean_peak_shape)
		power_pulse_shape_time_dependent_std.append(mean_peak_std)
		steady_state_power.append(mean_steady_state_power)
		steady_state_power_std.append(mean_steady_state_power_std)
	power_pulse_shape_time_dependent = np.mean(power_pulse_shape_time_dependent,axis=0)
	power_pulse_shape_time_dependent_std = np.sum(0.25*np.array(power_pulse_shape_time_dependent_std)**2,axis=0)**0.5
	steady_state_power = np.mean(steady_state_power,axis=0)
	steady_state_power_std = np.sum(0.25*np.array(steady_state_power_std)**2,axis=0)**0.5
	power_pulse_shape =power_pulse_shape_time_dependent + steady_state_power
	power_pulse_shape_std = (power_pulse_shape_time_dependent_std**2+steady_state_power_std**2)**0.5

	path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target) #+ '_back'
	merge_Te_prof_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_Te_prof_multipulse']
	merge_dTe_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_dTe_multipulse']
	merge_ne_prof_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_ne_prof_multipulse']
	merge_dne_multipulse = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_dne_multipulse']
	merge_time_original = np.load(path_where_to_save_everything + '/TS_data_merge_' + str(merge_ID_target) + '.npz')['merge_time']
	# new_timesteps = np.load(path_where_to_save_everything + '/merge' + str(merge_ID_target) + '_new_timesteps.npy')
	new_timesteps = np.linspace(-0.5,1.5,num=41)
	dt = np.nanmedian(np.diff(new_timesteps))


	spatial_factor = 1
	time_shift_factor = 0

	dx = 1.06478505470992 / 1e3	# 10/02/2020 from	Calculate magnification_FF.xlsx
	xx = np.arange(40) * dx  # m
	xn = np.linspace(0, max(xx), 1000)
	number_of_radial_divisions = 21
	r = np.arange(number_of_radial_divisions)*dx
	dr = np.median(np.diff(r))

	merge_time = time_shift_factor + merge_time_original
	TS_dt = np.nanmedian(np.diff(merge_time))

	TS_size = [-4.149230769230769056e+01, 4.416923076923076508e+01]
	TS_r = TS_size[0] + np.linspace(0, 1, 65) * (TS_size[1] - TS_size[0])
	TS_dr = np.median(np.diff(TS_r)) / 1000
	gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
	profile_centres = []
	profile_sigma = []
	profile_centres_score = []
	for index in range(np.shape(merge_ne_prof_multipulse)[0]):
		yy = merge_ne_prof_multipulse[index]
		yy_sigma = merge_dne_multipulse[index]
		yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
		if np.sum(yy>0)==0:
			profile_centres.append(0)
			profile_sigma.append(10)
			profile_centres_score.append(np.max(TS_r))
			continue
		yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
		p0 = [np.max(yy), 10, 0]
		bds = [[0, 0, np.min(TS_r)], [np.inf, TS_size[1], np.max(TS_r)]]
		fit = curve_fit(gauss, TS_r, yy, p0, sigma=yy_sigma, maxfev=100000, bounds=bds)
		profile_centres.append(fit[0][-1])
		profile_sigma.append(fit[0][-2])
		profile_centres_score.append(fit[1][-1, -1])
	# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)
	profile_centres = np.array(profile_centres)
	profile_sigma = np.array(profile_sigma)
	profile_centres_score = np.array(profile_centres_score)
	# centre = np.nanmean(profile_centres[profile_centres_score < 1])
	centre = np.nansum(profile_centres/(profile_centres_score**1))/np.sum(1/profile_centres_score**1)
	TS_r_new = (TS_r - centre) / 1000
	print('TS profile centre at %.3gmm compared to the theoretical centre' %centre)
	# temp_r, temp_t = np.meshgrid(TS_r_new, merge_time)
	# plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse,vmin=0);plt.colorbar().set_label('Te [eV]');plt.pause(0.01)
	# plt.figure();plt.pcolor(temp_t,temp_r,merge_ne_prof_multipulse,vmin=0);plt.colorbar().set_label('ne [10^20 #/m^3]');plt.pause(0.01)


	left = power_pulse_shape_time_dependent.argmax() - int(2e-3/time_resolution)
	right = power_pulse_shape_time_dependent.argmax() + int(4e-3/time_resolution)

	interpolation = interp1d((np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000)[left:right],np.cumsum(power_pulse_shape_time_dependent[left:right]))
	t_temp = np.linspace(np.min((np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000)[left:right]),np.max((np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000)[left:right]),num=1000000)
	current_trace_start = t_temp[np.abs(interpolation(t_temp) - np.max(np.cumsum(power_pulse_shape_time_dependent[left:right]))/50).argmin()]
	# current_trace_start = (np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000)[current_trace_start]

	# plt.figure()
	# # plt.plot(np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000,power_pulse_shape_time_dependent)
	# plt.plot((np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000)[left:right],np.cumsum(power_pulse_shape_time_dependent[left:right]))
	# plt.plot((np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000)[left:right][[0,-1]],[np.max(np.cumsum(power_pulse_shape_time_dependent[left:right]))/50]*2,'--',label='1/50 of peak')
	# plt.xlabel('arbitrary time [ms]')
	# plt.ylabel('cumulative power transmitted by the power source')
	# plt.legend(loc='best')
	# plt.grid()
	# plt.pause(0.01)


	start_r = np.abs(r - 0).argmin()
	end_r = np.abs(r - 5).argmin() + 1
	r_crop = r[start_r:end_r]
	area = 2*np.pi*(np.abs(TS_r_new) + np.median(np.diff(TS_r_new))/2) * np.median(np.diff(TS_r_new))/2	# the last /2 is due to the fact that I didn't do the left/right average so I'm double counting the area
	# energy_flow = np.sum(merge_ne_prof_multipulse*(13.6 + merge_Te_prof_multipulse) * area,axis=-1)
	hydrogen_mass = 1.008*1.660*1e-27	# kg
	boltzmann_constant_J = 1.380649e-23	# J/K
	eV_to_K = 8.617333262145e-5	# eV/K
	adiabatic_collisional_velocity = ((merge_Te_prof_multipulse + 5/3 *merge_Te_prof_multipulse*eV_to_K)/eV_to_K*boltzmann_constant_J/hydrogen_mass)**0.5
	homogeneous_mach_number = 1	# as an approximate assumption I assume sonic flow
	energy_flow = np.sum(homogeneous_mach_number*adiabatic_collisional_velocity*merge_ne_prof_multipulse*(0.5*hydrogen_mass*(homogeneous_mach_number*adiabatic_collisional_velocity)**2 + (5*merge_Te_prof_multipulse+13.6 + 2.2)/eV_to_K*boltzmann_constant_J) * area,axis=-1)
	# plt.figure()
	# # plt.plot(merge_time_original,energy_flow)
	# # plt.plot(merge_time_original[[0,-1]],[np.median(energy_flow[merge_time_original<0])]*2,'--')
	# # plt.pause(0.001)
	# plt.plot(merge_time_original,np.cumsum(energy_flow-np.mean(energy_flow[merge_time_original<0])))
	# plt.plot(merge_time_original[[0,-1]],[np.cumsum(energy_flow-np.mean(energy_flow[merge_time_original<0])).max()/50]*2,'--',label='1/50 of peak')
	# plt.xlabel('arbitrary time [ms]')
	# plt.ylabel('cumulative energy transported by plasma')
	# plt.legend(loc='best')
	# plt.grid()
	# plt.pause(0.01)

	interpolation = interp1d(merge_time_original,np.cumsum(energy_flow-np.mean(energy_flow[merge_time_original<0])))
	t_temp = np.linspace(np.min(merge_time_original),np.max(merge_time_original),num=1000000)
	TS_start = t_temp[np.abs(interpolation(t_temp) - np.cumsum(energy_flow-np.mean(energy_flow[merge_time_original<0])).max()/50).argmin()]

	offset_current_trace = current_trace_start-TS_start

	if plot_report:
		plt.figure()
		# plt.plot(np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000,power_pulse_shape_time_dependent)
		plt.plot((np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000)[left:right]-current_trace_start,np.cumsum(power_pulse_shape_time_dependent[left:right])/np.max(np.cumsum(power_pulse_shape_time_dependent[left:right])),label='energy generated from power source')
		plt.plot((np.arange(len(power_pulse_shape_time_dependent))*time_resolution*1000)[left:right][[0,-1]]-current_trace_start,[1/50]*2,'--')

		plt.plot(merge_time_original-TS_start,np.cumsum(energy_flow-np.mean(energy_flow[merge_time_original<0]))/np.max(np.cumsum(energy_flow-np.mean(energy_flow[merge_time_original<0]))),label='energy transported by plasma (TS)')
		plt.plot(merge_time_original[[0,-1]]-TS_start,[1/50]*2,'--')
		plt.xlabel('Time from 1/50 of maximum energy transferred [s]')
		plt.ylabel('cumulative energy transferred [au]')
		plt.grid()
		plt.title('merge '+str(merge_ID_target)+' shift to remove from current trace to match TS = %.5gms' %(offset_current_trace))
		plt.legend(loc='best')
		plt.pause(0.01)

	return offset_current_trace

##############################################################################################################################################################################


def DOM52sec(value):
	# conversion of time format necessary for magnum timestamp
	if np.isfinite(value)==True:
		out = int("{0:b}".format(int(value))[:-32],2)+int("{0:b}".format(int(value))[-32:],2)/(2**32)
		return out
	else:
		return np.nan
