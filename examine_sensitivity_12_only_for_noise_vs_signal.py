import numpy as np
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())
#import .functions
import os,sys

from numpy.core.multiarray import ndarray

os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
from functions.spectools import get_angle,rotate,get_tilt,do_tilt,getFirstBin, binData, get_angle_2,binData_with_sigma,get_line_position,get_4_points
from functions.fabio_add import find_nearest_index,multi_gaussian,all_file_names,load_dark,find_index_of_file,get_metadata,examine_current_trace,movie_from_data,get_bin_and_interv_no_lines, four_point_transform, fix_minimum_signal, do_tilt_no_lines, fix_minimum_signal2,apply_proportionality_calibration,fix_minimum_signal_calibration,get_bin_and_interv_specific_wavelength,fix_minimum_signal_experiment
from functions.Calibrate import do_waveL_Calib,do_waveL_Calib_simplified
from PIL import Image
import xarray as xr
import pandas as pd
import copy
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.signal import gaussian
from skimage.transform import resize

exec(open("/home/ffederic/work/analysis_scripts/scripts/profile_smoothing.py").read())

n = np.arange(10,20)
waveLengths = [486.13615,434.0462,410.174,397.0072,388.9049,383.5384] + list((364.6*(n*n)/(n*n-4)))
# waveLengths = [486.13615,434.0462,410.174,397.0072,388.9049,383.5384]
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



#fdir = '/home/ff645/GoogleDrive/ff645/YPI - Fabio/Collaboratory/Experimental data/tests/test_data_folder'
#df_log = pd.read_csv('functions/Log/shots_2.csv',index_col=0)
#df_settings = pd.read_csv('functions/Log/settin	gs_2.csv',index_col=0)

fdir = '/home/ffederic/work/Collaboratory/test/experimental_data'
df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
df_settings = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_3.csv',index_col=0)



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
df_sphere['Rad'] = df_sphere.RadRaw / I_nom * 1e4	# I think this is just to go from cm-2 to m-2






#####	THIS BIT IS FOR THE EXPERIMENTS 66 onwards

limits_angle = [66, 70]
limits_tilt = [66, 70]
limits_wave = [66, 70]
figure_number = 0
where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/Calibrations/'


# int_time_long = [0.1, 1, 10, 100]
# calib_ID_long=[215, 216, 217, 218]
int_time_long = [0.02, 0.1, 1, 10, 100, 100, 200, 500]
calib_ID_long=[214, 215, 216, 217, 413, 218, 414, 415]	# I can't really compare 2xx and 4xx because I don't have a wavelength axis for 4xx

data_average = []
data_sigma = []
for iFile, j in enumerate(calib_ID_long):
	(folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
	type = '.tif'
	filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
	type = '.txt'
	temp = []
	for index, filename in enumerate(filenames):
		fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
		im = Image.open(fname)
		data = np.array(im)
		# data = (data - dataDark)
		# data = fix_minimum_signal2(data)

		temp.append(data)
	data_average.append(np.mean(temp, axis=0))
	data_sigma.append(np.std(temp, axis=0))


data_average_2 = []
data_sigma_2 = []
for iFile, j in enumerate(calib_ID_long):
	data_average_2.append(median_filter(data_average[iFile],[3,3]))
	data_sigma_2.append(median_filter(data_sigma[iFile],[3,3]))


plt.figure(figsize=(20, 10))
for iFile, j in enumerate(calib_ID_long):
	plt.plot(data_average[iFile][400:1250].flatten()[::10],data_sigma[iFile][400:1250].flatten()[::10],'+',label='int time '+str(int_time_long[iFile])+'ms')
plt.xlabel('signal [au]')
plt.ylabel('signal std [au]')
# plt.plot(np.unique(data_average_2),(np.unique(data_average_2)-100)**0.5,'--k')

std_vs_counts = lambda x, a,b,x0: np.maximum(2,a*(np.maximum(0,x-x0)**b))
fit = curve_fit(std_vs_counts,np.array(data_average)[:,400:1250].flatten(),np.array(data_sigma)[:,400:1250].flatten(),p0=[1,0.5,100],bounds=[[0,0,80],[np.inf,1,120]],verbose=2)
fit = [np.array([ 1.77615007,  0.45271138, 94.47173805]),np.array([[ 7.18422728e-07, -4.61178099e-08,  1.34693352e-06], [-4.61178099e-08,  3.00381572e-09, -8.51554623e-08], [ 1.34693352e-06, -8.51554623e-08,  7.35861433e-05]])]
plt.plot(np.unique(data_average),std_vs_counts(np.unique(data_average),*fit[0]),'--k',label='fit '+str(fit[0]))
plt.plot(np.unique(data_average),(np.unique(data_average)-100)**0.5,'--y',label='theoretical counts^0.5')
# fit1 = curve_fit(std_vs_counts,np.array(data_average[0]).flatten(),np.array(data_sigma[0]).flatten(),p0=[1,0.5,100],bounds=[[0,0,80],[np.inf,1,120]],verbose=2)
# plt.plot(np.unique(data_average_2),std_vs_counts(np.unique(data_average_2),*fit[0]),'--y')
plt.ylim(top=250)
plt.legend(loc='best',fontsize='x-small')
plt.grid()
plt.savefig(where_to_save_everything+'std_to_counts_correlation.eps', bbox_inches='tight')
plt.close()

plt.pause(0.01)
