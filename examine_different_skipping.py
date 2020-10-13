import numpy as np
import matplotlib.pyplot as plt
#import .functions
import os,sys
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
from functions.spectools import rotate,do_tilt, binData
from functions.fabio_add import find_nearest_index,multi_gaussian,all_file_names,load_dark,find_index_of_file,get_metadata,examine_current_trace,movie_from_data
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

merge_ID=3
# merge_ID = 20

if calculate_geometry:
	geom=getGeom(merge_ID,df_settings,df_log,fdir=fdir)
	print(geom)
else:
	geom = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
	geom.loc[0] = [0.01759,np.nan,29.202126,53.647099,53.647099]
geom_null = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
geom_null.loc[0] = [0,0,geom['binInterv'],geom['bin00a'],geom['bin00b']]


waveLcoefs = do_waveL_Calib(merge_ID,df_settings,df_log,geom,fdir=fdir)

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
df_sphere['Rad'] = df_sphere.RadRaw / I_nom * 1e4


binnedSens = do_Intensity_Calib(merge_ID,df_settings,df_log,df_calLog,df_sphere,geom,waveLcoefs,waves_to_calibrate=['Hb'],fdir=fdir)

binnedSens = np.ones(np.shape(binnedSens))


row_shift=2*10280/1000000	# ms
merge_ID_target = 4
composed_array_all = []
for index,skip in enumerateexamine_different_skipping.py([1,2,3,4,5]):
	composed_array_all.append(xr.open_dataarray('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(skip)+'row_composed_array.nc'))
	plt.figure()
	ani = movie_from_data(np.array([(composed_array_all[index].values)]), 1 / 0.01, row_shift, 'Wavelength axis [pixles]','Row axis [pixles]', 'Intersity [au]')
	plt.pause(0.001)

merge_tot_all = []
for skip in [1,2,3,4,5]:
	merge_tot_all.append(xr.open_dataarray('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(skip)+'row_merge_tot.nc'))

all_fits_all = []
for skip in [1,2,3,4,5]:
	all_fits_all.append(np.load('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(skip)+'row_all_fits.npy'))
# ani = movie_from_data(np.array([all_fits_all[0]]), 1 / 0.01, row_shift, 'Wavelength axis [pixles]','Row axis [pixles]', 'Intersity [au]')
plt.imshow(all_fits_all[0][:,:,0], 'rainbow',vmin=0, origin='lower');plt.pause(0.01)

difference=[]
for j in range(9):
	temp=[]
	for index,skip in enumerate([1,2,3,4,5]):
		temp.append(np.sqrt(np.nansum(np.power( all_fits_all[index][85:150,9:25,j]-all_fits_all[0][85:150,9:25,j],2),axis=(0,1)))/np.nansum(all_fits_all[0][85:150,9:25,j],axis=(0,1)))
	plt.plot(temp,label='line '+str(j+1))
	difference.append(temp)
plt.legend(loc='best')
plt.pause(0.001)

binned_data_all = []
for skip in [1,2,3,4,5]:
	binned_data_all.append(xr.open_dataarray('/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)+'_skip_of_'+str(skip)+'row_binned_data.nc'))
plt.figure();plt.imshow(binned_data_all[0][:,20]);plt.pause(0.01)





row_shift=2*10280/1000000	# ms
ani = coleval.movie_from_data(np.array([(composed_array_all[1].values)]), 1 / 0.01, row_shift,'Wavelength axis [pixles]', 'Row axis [pixles]', 'Intersity [au]')
plt.show()
exit()




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


	data_all=[]
	for index in range(0,len(filenames)):
		fname = fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0/'+filenames[index]
		im = Image.open(fname)
		data = np.array(im)
		data_all.append(data)
	data_all=np.array(data_all)
	exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())
	ani = coleval.movie_from_data(np.array([data_all]), 1 / 0.05, row_shift,'Wavelength axis [pixles]', 'Row axis [pixles]', 'Intersity [au]')
	# plt.show()
	ani.save(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0/' + str(merge_ID_target) + '_exploratory_video' + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])



	num_pulses=last_frame+len(filenames)-first_frame
	if num_pulses<=df_log.loc[j, ['number_of_pulses']][0]:
		df_log.loc[j, ['number_of_pulses']] = num_pulses
		df_log.loc[j, ['first_pulse_at_this_frame']] = first_frame
	else:
		while not (result in ['f','l']):
			result = input('Number of pulses is '+str(num_pulses)+' rather than '+str(df_log.loc[j, ['number_of_pulses']][0])+' \n Do you consider more trustworthy the first or lase frame you found? \n f: first, l: last')
			if result=='f':
				df_log.loc[j, ['first_pulse_at_this_frame']] = first_frame
			if result == 'l':
				first_frame = len(filenames)+first_frame -df_log.loc[j, ['number_of_pulses']]
				df_log.loc[j, ['first_pulse_at_this_frame']] = first_frame
	df_log.to_csv(path_or_buf='/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_1.csv')

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









