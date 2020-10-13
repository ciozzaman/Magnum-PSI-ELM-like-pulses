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





merge_ID_target = 40
all_j=find_index_of_file(merge_ID_target,df_settings,df_log)
for j in all_j:
	dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)
	(folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]
	type = '.tif'
	filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)
	type = '.txt'
	filename_metadata = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)[0]

	# plt.figure()
	data_all = []
	for index, filename in enumerate(filenames[20:30]):
		fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
		im = Image.open(fname)
		data = np.array(im)
		data = data - dataDark  # I checked that for 16 or 12 bit the dark is always around 100 counts
		data_all.append(data)
data_all = np.array(data_all)
# data_sum = np.sum(data_all[:, :, 1300:1500], axis=(-1, -2))

data_all2 = []
for index,data in enumerate(data_all):
	data_all2.append( fix_minimum_signal(data))
data_all2 = np.array(data_all2)

data_mean = np.mean(data_all, axis=(0))
data_mean2 = np.mean(data_all2, axis=(0))

plt.figure()
plt.plot(np.mean(data_mean,axis=0),label='original')
plt.plot(np.mean(data_mean2,axis=0),label='corrected')
plt.legend(loc='best')
# plt.title('intensity on the chord 20 with and wothout balancing filter')
# plt.xlabel('wavelength axis [au]')
# plt.ylabel('intensity [au]')
plt.pause(0.001)

