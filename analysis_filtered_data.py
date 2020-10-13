import numpy as np
import matplotlib.pyplot as plt
#import .functions
import os,sys
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
from functions.spectools import rotate,do_tilt, binData
from functions.fabio_add import find_nearest_index,multi_gaussian,all_file_names,load_dark,find_index_of_file,get_metadata,examine_current_trace
from PIL import Image
import xarray as xr
import pandas as pd
import copy
from scipy.optimize import curve_fit
from scipy import interpolate

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


calculate_geometry = False



#fdir = '/home/ff645/GoogleDrive/ff645/YPI - Fabio/Collaboratory/Experimental data/tests/test_data_folder'
#df_log = pd.read_csv('functions/Log/shots_2.csv',index_col=0)
#df_settings = pd.read_csv('functions/Log/settin	gs_2.csv',index_col=0)

fdir = '/home/ffederic/work/Collaboratory/test/experimental_data'
df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_1.csv',index_col=0)
df_settings = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_1.csv',index_col=0)


if calculate_geometry:
	geom=functions.getGeom(df_settings,df_log,fdir=fdir)
	print(geom)
else:
	geom = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
	geom.loc[0] = [0.01759,np.nan,29.202126,53.647099,53.647099]
geom_null = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
geom_null.loc[0] = [0,0,geom['binInterv'],geom['bin00a'],geom['bin00b']]

#waveLcoefs = functions.do_waveL_Calib(df_settings,df_log,geom,fdir=fdir)
#print('waveLcoefs= '+str(waveLcoefs))


# data_all = []
angle=[]
for merge_ID_target in range(32,39):
# for merge_ID_target in range(17,23):
	all_j=find_index_of_file(merge_ID_target,df_settings,df_log)
	for j in all_j:
		dataDark = load_dark(j,df_settings,df_log,fdir,geom_null)
		(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
		type = '.tif'
		filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)
		type = '.txt'
		filename_metadata = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)[0]

		# plt.figure()
		data_all = []
		for index, filename in enumerate(filenames):
			fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + \
					filename
			im = Image.open(fname)
			data = np.array(im)
			data = data - dataDark
			# data = fix_minimum_signal(data)
			data_all.append(data)
		data_all = np.array(data_all)
		data_mean = np.mean(data_all,axis=0)
		try:
			angle.append(get_angle_2(data_mean,nLines=2))
			print(angle)
		except:
			print('FAILED')

angle = np.array(angle)

#this is for merge 40-53
angle=np.array([[[-0.15547637, -0.1111757 ],
        [ 0.10017873,  0.02872499]],
       [[-0.04131632, -0.11591068],
        [ 0.04501964,  0.02899283]],
       [[-0.13251782, -0.11505956],
        [ 0.09531651,  0.02898415]],
       [[-0.90320104, -0.11498321],
        [ 0.34257669,  0.02990536]],
       [[ 0.36532655,         np.nan],
        [ 0.20287097,         np.nan]],
       [[ 0.26998885, -0.11153748],
        [ 0.06388269,  0.03666075]],
       [[-0.02532894, -0.11998517],
        [ 0.05026678,  0.02949288]],
       [[-0.03525347, -0.11631955],
        [ 0.04413133,  0.02828175]],
       [[-0.22622798, -0.11720355],
        [ 0.13487788,  0.02849795]],
       [[ 0.08502387, -0.11562485],
        [ 0.1140376 ,  0.03544645]],
       [[-0.01595292, -0.10181777],
        [ 0.22577497,  0.03374602]],
       [[-0.27618349, -0.10406049],
        [ 0.14834453,  0.03971919]],
       [[ 0.2030814 , -0.10800348],
        [ 0.20925125,  0.03598202]]])
#this is for merge 32-39
angle=np.array([[[ 0.2730194 , -0.41273367],
        [ 0.08101617,  0.18106157]],
       [[-0.10732836, -0.14319128],
        [ 0.03527036,  0.02908546]],
       [[-0.10372567, -0.12729041],
        [ 0.03195125,  0.02774662]],
       [[-0.10184881, -0.11794369],
        [ 0.03258391,  0.02847863]],
       [[-0.10737155, -0.12578377],
        [ 0.03264232,  0.02752683]],
       [[-0.24555832, -0.13526176],
        [ 0.10431149,  0.02889631]],
       [[-0.00367268, -0.12527252],
        [ 0.13268985,  0.08525873]]])


angle = np.nansum(angle[:,0]/ (angle[:,1]**2))/np.nansum(1/angle[:,1]**2)
# angle = np.mean(np.array(angle)[:,0],axis=0)[1]
# if np.abs(angle)>0.06:	#	I found out that substracting the dark before getting the tilt this bit is not necessary
# 	angle=0

#this is for merge 17-31
angle=np.array([[[  4.79062580e-04,  -3.30661311e+00],
        [  3.07423897e-02,   2.04739634e+00]],
       [[  5.09684100e-04,  -3.57878996e+00],
        [  3.56485972e-02,   2.23862290e+00]],
       [[  1.55266771e-01,  -8.07649546e+00],
        [  6.19553147e-02,   5.92879191e+00]],
       [[  1.03106811e-01,  -1.15023946e+00],
        [  5.99342478e-02,   1.54722789e+00]],
       [[  3.50525206e-01,  -2.17179622e-01],
        [  5.34677605e-01,   9.76643043e-01]],
       [[ -3.21472285e-03,  -2.72015250e+00],
        [  3.01999859e-02,   1.86273076e+00]],
       [[  6.55596839e-03,  -6.73318104e+00],
        [  3.96822724e-02,   3.73957374e+00]],
       [[  1.02834845e-01,  -8.08562599e-01],
        [  4.69265441e-02,   1.41984426e+00]],
       [[  1.94582013e-01,   2.25621013e+00],
        [  4.92299985e-02,   1.83155843e+00]],
       [[  2.79298362e-01,   2.73942078e-01],
        [  1.11315057e-01,   8.55720385e-01]],
       [[  8.36889904e-04,  -2.20820110e-01],
        [  3.02258991e-02,   3.72428561e+00]],
       [[  4.95666803e-03,  -3.21360468e+00],
        [  3.55664479e-02,   1.79946681e+00]],
       [[  1.54491086e-01,   2.73947013e-02],
        [  6.23547188e-02,   7.14922386e-01]],
       [[  3.69809413e-02,   1.09553374e+00],
        [  4.48839508e-02,   1.13687841e+00]],
       [[  6.77275873e-02,  -1.08550765e+00],
        [  4.30078444e-02,   7.20892273e-01]],
       [[  2.21433483e-03,  -4.09028448e+00],
        [  3.07715583e-02,   1.99152138e+00]],
       [[  7.99032160e-03,  -5.07480298e+00],
        [  3.17911685e-02,   3.11230465e+00]],
       [[  1.93480939e-02,  -2.01238786e+00],
        [  3.20384960e-02,   1.06870298e+00]],
       [[  7.72551858e-03,  -1.47389468e+00],
        [  3.75997557e-02,   1.20689456e+00]],
       [[  1.68332003e-03,  -1.66277297e+00],
        [  3.35171273e-02,   9.06861156e-01]],
       [[  2.27270549e-01,  -8.31315413e-01],
        [  7.04578233e-02,   8.61705456e-01]],
       [[  8.23684187e-03,   5.80773246e-01],
        [  4.08507844e-02,   4.84110693e+00]],
       [[ -2.08840784e-01,  -1.58313789e+00],
        [  1.45780825e-01,   7.00184905e-01]],
       [[  2.31434834e-01,   2.48770120e-01],
        [  6.98583523e-02,   9.34055979e-01]],
       [[  6.62731090e-01,  -1.55459683e+00],
        [  2.74018488e-01,   5.98804328e-01]],
       [[ -1.00641292e-01,  -7.64616654e-01],
        [  6.29741171e-02,   7.18497640e-01]],
       [[  1.10721360e-02,  -2.07798408e-01],
        [  3.63082448e-02,   2.51468597e+00]],
       [[  1.67700134e-01,  -2.65347156e+00],
        [  5.77659119e-02,   2.32042797e+00]]])



tilt=[]
for merge_ID_target in range(32,39):
# for merge_ID_target in range(17, 23):
	all_j=find_index_of_file(merge_ID_target,df_settings,df_log)
	for j in all_j:
		dataDark = load_dark(j,df_settings,df_log,fdir,geom_null)
		(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
		type = '.tif'
		filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)
		type = '.txt'
		filename_metadata = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)[0]

		# plt.figure()
		data_all = []
		for index, filename in enumerate(filenames):
			fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + \
					filename
			im = Image.open(fname)
			data = np.array(im)
			data = data - dataDark
			# data = fix_minimum_signal(data)
			data_all.append(data)
		data_all = np.array(data_all)
		data_mean = np.mean(data_all,axis=0)
		data_mean = rotate(data_mean, angle)
		tilt.append(get_tilt(data_mean,nLines=3))
		print(tilt)

tilt = np.array(tilt)

#this is for merge 40-53
tilt = np.array([[  2.80852227e+01,              np.nan,   4.65263158e+01],
       [  2.80842105e+01,              np.nan,   4.63927126e+01],
       [  2.80407895e+01,   4.87295510e-03,   4.31061272e+01],
       [  2.80902834e+01,              np.nan,   4.58353576e+01],
       [             np.nan,              np.nan,              np.nan],
       [  2.80815789e+01,              np.nan,   4.57786775e+01],
       [  2.80318826e+01,   3.82875044e-03,   4.45592388e+01],
       [  2.80963563e+01,              np.nan,   4.60472335e+01],
       [  2.80995951e+01,              np.nan,   4.59824561e+01],
       [  2.80734818e+01,              np.nan,   4.66329285e+01],
       [             np.nan,              np.nan,              np.nan],
       [  2.81125506e+01,              np.nan,   4.57233468e+01],
       [  2.80759109e+01,              np.nan,   4.66099865e+01]])

tilt = np.nanmean(tilt,axis=0)

#this is for merge 17-31
tilt = np.array([[  2.93241903e+01,   2.56914009e-03,   4.60937975e+01],
       [             np.nan,              np.nan,              np.nan],
       [             np.nan,              np.nan,              np.nan],
       [             np.nan,              np.nan,              np.nan],
       [             np.nan,              np.nan,              np.nan],
       [  2.93096154e+01,   3.12326835e-03,   4.57934361e+01],
       [             np.nan,              np.nan,              np.nan],
       [             np.nan,              np.nan,              np.nan],
       [             np.nan,              np.nan,              np.nan],
       [             np.nan,              np.nan,              np.nan],
       [  2.93047571e+01,   2.61951539e-03,   4.63866980e+01],
       [  2.93173077e+01,   3.47589542e-03,   4.50205279e+01],
       [  2.94338057e+01,              np.nan,   4.75546559e+01],
       [             np.nan,              np.nan,              np.nan],
       [             np.nan,              np.nan,              np.nan],
       [  2.93047571e+01,   2.11161388e-03,   4.71172263e+01],
       [  2.94392713e+01,              np.nan,   4.73940621e+01],
       [  2.94362348e+01,              np.nan,   4.73778677e+01],
       [  2.94356275e+01,              np.nan,   4.73900135e+01],
       [  2.94311741e+01,              np.nan,   4.75303644e+01],
       [  2.94463563e+01,              np.nan,   4.71754386e+01],
       [  2.94467611e+01,              np.nan,   4.71929825e+01],
       [             np.nan,              np.nan,              np.nan],
       [             np.nan,              np.nan,              np.nan],
       [             np.nan,              np.nan,              np.nan],
       [  2.93163968e+01,   1.66894250e-03,   4.76822302e+01],
       [             np.nan,              np.nan,              np.nan],
       [             np.nan,              np.nan,              np.nan],
       [             np.nan,              np.nan,              np.nan],
       [             np.nan,              np.nan,              np.nan]])

#this is for merge 32-39
tilt = np.array([[             np.nan,              np.nan,              np.nan],
       [             np.nan,              np.nan,              np.nan],
       [  2.81770243e+01,   3.10004203e-03,   5.73164919e+01],
       [  2.81182186e+01,   3.33771823e-03,   5.84656910e+01],
       [  2.81085695e+01,   3.04448468e-03,   5.89287303e+01],
       [  2.81750000e+01,   2.93639557e-03,   5.69993970e+01],
       [  2.80354251e+01,   5.29287227e-03,   4.22762075e+01]])


waveLcoefs = []
geom = pd.DataFrame(columns=['angle', 'tilt', 'binInterv', 'bin00a', 'bin00b'])
geom.loc[0] = [angle, tilt[1], tilt[0], tilt[2], tilt[2]]  # from analysis_filtered_data.py
# for merge_ID_target in range(40,53):
for merge_ID_target in range(17, 26):
	all_j=find_index_of_file(merge_ID_target,df_settings,df_log)
	waveLcoefs.append(do_waveL_Calib(merge_ID_target, df_settings, df_log, geom, fdir=fdir))




	# index=0
	# result = 'n'
	# plt.ion()
	# plt.figure()
	# while result != 'e':
	# 	while int(index+first_frame-1)<0:
	# 		index+=1
	# 	fname = fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0/'+filenames[max(0,int(index+first_frame-1))]
	# 	print(fname)
	# 	im = Image.open(fname)
	# 	data = np.array(im)
	# 	plt.imshow(data, 'rainbow', origin='lower')
	# 	plt.title('Frame '+str(int(index+1)))
	# 	plt.draw()
	# 	result = input('what do you want to see? \n n: next, p: previous, e: escape, r: repeat from')
	# 	if result == 'n':
	# 		index+=1
	# 	if result == 'p':
	# 		index-=1
	# 	elif result == 'r':
	# 		index = int(input('from which index do you want to restart?'))-1
	# plt.close()







	# data_all=[]
	# for index in range(0,len(filenames)):
	# 	fname = fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0/'+filenames[index]
	# 	im = Image.open(fname)
	# 	data = np.array(im)
	# 	data_all.append(data)
	# data_all=np.array(data_all)
	# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())
	# ani = coleval.movie_from_data(np.array([data_all]), 1 / 0.05, row_shift,'Wavelength axis [pixles]', 'Row axis [pixles]', 'Intersity [au]')
	# # plt.show()
	# ani.save(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0/' + str(merge_ID_target) + '_exploratory_video' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
	#
	#
	#
	# num_pulses=last_frame+len(filenames)-first_frame
	# if num_pulses<=df_log.loc[j, ['number_of_pulses']][0]:
	# 	df_log.loc[j, ['number_of_pulses']] = num_pulses
	# 	df_log.loc[j, ['first_pulse_at_this_frame']] = first_frame
	# else:
	# 	while not (result in ['f','l']):
	# 		result = input('Number of pulses is '+str(num_pulses)+' rather than '+str(df_log.loc[j, ['number_of_pulses']][0])+' \n Do you consider more trustworthy the first or lase frame you found? \n f: first, l: last')
	# 		if result=='f':
	# 			df_log.loc[j, ['first_pulse_at_this_frame']] = first_frame
	# 		if result == 'l':
	# 			first_frame = len(filenames)+first_frame -df_log.loc[j, ['number_of_pulses']]
	# 			df_log.loc[j, ['first_pulse_at_this_frame']] = first_frame
	# df_log.to_csv(path_or_buf='/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_1.csv')

# df_log.loc[j, ['number_of_pulses']] = num_pulses
	# df_log.loc[j, ['first_pulse_at_this_frame']] = first_frame

# df_log.to_csv(path_or_buf='/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_1.csv')
#
# 		(bof, eof, roi_lb, roi_tr, elapsed_time) = get_metadata(fdir, folder, sequence, untitled, filename_metadata)
# 		(CB_to_OES_initial_delay, incremental_step, first_pulse_at_this_frame, bad_pulses_indexes, number_of_pulses) = \
# 		df_log.loc[j, ['CB_to_OES_initial_delay', 'incremental_step', 'first_pulse_at_this_frame', 'bad_pulses_indexes',
# 					   'number_of_pulses']]
# 		bad_pulses_indexes = bad_pulses_indexes.replace(' ', '').split(',')
# 		bad_pulses_indexes = list(map(int, bad_pulses_indexes))








merge_ID_target = int(input('insert the merge_ID as per file /home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_1.csv'))
all_j=find_index_of_file(merge_ID_target,df_settings,df_log)
for j in all_j:
	(folder,date,sequence,untitled,fname_current_trace) = df_log.loc[j,['folder','date','sequence','untitled','current_trace_file']]
	type = '.tif'
	filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)
	type = '.txt'
	filename_metadata = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)[0]
	# data_sum = []
	# for index, filename in enumerate(filenames):
	# 	fname = fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0/'+filenames[index]
	# 	im = Image.open(fname)
	# 	data = np.array(im)
	# 	data_sum.append(np.sum(data))
	# plt.figure()
	# plt.plot(data_sum)
	# plt.pause(0.001)

	index=0
	result = 'n'
	plt.ion()
	plt.figure()
	while result != 'y':
		fname = fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0/'+filenames[index]
		im = Image.open(fname)
		data = np.array(im)
		plt.imshow(data, 'rainbow', origin='lower')
		plt.title('Frame '+str(index))
		plt.draw()
		result = input('is here present the first pulse? \n u: unsure, y: yes, n: no, r: repeat from start')
		if result == 'y':
			first_frame = index+1
		elif result == 'r':
			index = int(input('from which index do you want to restart?'))-1
		index+=1
	# plt.close()

	index=-1
	result = 'n'
	# plt.ion()
	# plt.figure()
	while result != 'y':
		fname = fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0/'+filenames[index]
		im = Image.open(fname)
		data = np.array(im)
		plt.imshow(data, 'rainbow', origin='lower')
		plt.title('Frame '+str(index))
		plt.draw()
		result = input('is here present the last pulse? \n u: unsure, y: yes, n: no, r: repeat from start')
		if result == 'y':
			last_frame = index+1
		elif result == 'r':
			index=int(input('from which index do you want to restart?'))+1
		index-=1
	plt.close()

	bad_pulses,first_good_pulse,first_pulse,last_pulse = examine_current_trace(fdir+'/'+folder+'/', fname_current_trace, df_log.loc[j, ['number_of_pulses']][0])



	num_pulses=len(filenames)+last_frame-first_frame+1
	real_num_pulses = last_pulse-first_pulse+1
	if num_pulses==real_num_pulses:
		df_log.loc[j, ['first_pulse_at_this_frame']] = first_frame-first_good_pulse+1
		df_log.loc[j, ['bad_pulses_indexes']] = str(bad_pulses.tolist())[1:-1]
	else:
		while not (result in ['f','l']):
			result = input('Number of pulses is '+str(num_pulses)+' rather than '+str(real_num_pulses)+' \n Do you consider more trustworthy the first or lase frame you found? \n f: first, l: last')
			if result=='f':
				df_log.loc[j, ['first_pulse_at_this_frame']] = first_frame - first_good_pulse + 1
				df_log.loc[j, ['bad_pulses_indexes']] = str(bad_pulses.tolist())[1:-1]
			if result == 'l':
				df_log.loc[j, ['first_pulse_at_this_frame']] = len(filenames)-np.abs(last_frame) + 1 - real_num_pulses
				df_log.loc[j, ['bad_pulses_indexes']] = str(bad_pulses.tolist())[1:-1]



	df_log.to_csv(path_or_buf='/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_1.csv')










