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


data_filtered_all=[]
folder='2019-05-03'
date=np.nan
sequence=2
untitled=4
type = '.tif'
filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
for filename in filenames:
	fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
	im = Image.open(fname)
	data = np.array(im)
	data_filtered_all.append(data)
data_filtered_all=np.array(data_filtered_all)
# plt.figure();plt.imshow(data);plt.colorbar();plt.pause(0.001)
# plt.figure();plt.plot(data_filtered_all[:,937,1215]);plt.pause(0.01)
data_filtered_std=np.std(data_filtered_all,axis=0)
data_filtered=np.mean(data_filtered_all,axis=0)
plt.figure();plt.imshow(data_filtered);plt.colorbar();plt.pause(0.001)
plt.figure();plt.imshow(data_filtered_std);plt.colorbar();plt.pause(0.001)


data_unfiltered_all=[]
folder='2019-05-23'
date=np.nan
sequence=2
untitled=11
type = '.tif'
filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
for filename in filenames:
	fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
	im = Image.open(fname)
	data = np.array(im)
	data_unfiltered_all.append(data)
data_unfiltered_all=np.array(data_unfiltered_all)
# plt.figure();plt.imshow(data);plt.colorbar();plt.pause(0.001)
# plt.figure();plt.plot(data_unfiltered_all[:,945,1220]);plt.pause(0.01)
data_unfiltered_std=np.std(data_unfiltered_all,axis=0)
data_unfiltered=np.mean(data_unfiltered_all,axis=0)
plt.figure();plt.imshow(data_unfiltered);plt.colorbar();plt.pause(0.001)
plt.figure();plt.imshow(data_unfiltered_std);plt.colorbar();plt.pause(0.001)

plt.figure();plt.plot(data_filtered[:,786])
plt.plot(data_unfiltered[:,1219]);plt.pause(0.01)

filtered = np.sum(data_filtered[:,1170:1230],axis=(0,1))
unfiltered = np.sum(data_unfiltered[:,1190:1250],axis=(0,1))
attenuation= unfiltered/filtered



merge_ID_target = int(input('insert the merge_ID as per file /home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_1.csv'))
all_j=find_index_of_file(merge_ID_target,df_settings,df_log)
for j in all_j:
	(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
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
	for filename in filenames:
		fname = fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0/'+filename
		im = Image.open(fname)
		data = np.array(im)
		if data.max()>4000:
			print(fname)
			print('is overexposed')

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
			last_frame = index+2
		elif result == 'r':
			index=int(input('from which index do you want to restart?'))+1
		index-=1
	plt.close()

	result = input('Do you consider more trustworthy the first or lase frame you found? \n f: first, l: last')
	if result == 'f':
		last_frame = first_frame +df_log.loc[j, ['number_of_pulses']]
	if result == 'l':
		first_frame = len(filenames) + last_frame - df_log.loc[j, ['number_of_pulses']]+1

	print('first frame = '+str(first_frame))
	print('last frame = ' + str(last_frame))

	index=0
	result = 'n'
	plt.ion()
	plt.figure()
	while result != 'e':
		while int(index+first_frame-1)<0:
			index+=1
		fname = fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0/'+filenames[max(0,int(index+first_frame-1))]
		print(fname)
		im = Image.open(fname)
		data = np.array(im)
		plt.imshow(data, 'rainbow', origin='lower')
		plt.title('Frame '+str(int(index+1)))
		plt.draw()
		result = input('what do you want to see? \n n: next, p: previous, e: escape, r: repeat from')
		if result == 'n':
			index+=1
		if result == 'p':
			index-=1
		elif result == 'r':
			index = int(input('from which index do you want to restart?'))-1
	plt.close()


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










