import numpy as np
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_batch.py").read())
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

folder_to_check = '/home/ffederic/work/Collaboratory/test/experimental_data'
folder = '2019-07-04'
TS_size=[-4.149230769230769056e+01,4.416923076923076508e+01]
dt=100/1000	# ms

filenames = all_file_names('/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/TS data','.npy')
temp = []
for val in filenames:
	if val[:18] == 'ne_prof_multipulse':
		for j in range(len(val)):
			if ((val[j]=='-') or (val[j]=='_')): #modified 12/08/2018 from "_" to "-"
				start=j
			elif val[j]=='.':
				end=j
		temp.append(val[start+1:end])
filenames = np.sort(np.array(temp,dtype=int))

for i in filenames:
	print('working on '+str(i))

	Te_prof_multipulse = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/TS data/Te_prof_multipulse_'+folder+'_'+str(i)+'.npy')
	dTe_multipulse = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/TS data/dTe_multipulse_'+folder+'_'+str(i)+'.npy')
	ne_prof_multipulse = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/TS data/ne_prof_multipulse_'+folder+'_'+str(i)+'.npy')
	dne_multipulse = np.load('/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/TS data/dne_multipulse_'+folder+'_'+str(i)+'.npy')

	r = np.linspace(TS_size[0],TS_size[-1],num=np.shape(Te_prof_multipulse)[1])
	t = np.arange(np.shape(Te_prof_multipulse)[0])*dt
	r_full,t_full = np.meshgrid(r,t)

	fig, ax = plt.subplots( 1,2,figsize=(12, 15), squeeze=False)
	plot_index = 0
	im1 = ax[0,plot_index].pcolor(r_full,t_full,Te_prof_multipulse,cmap='rainbow')
	ax[0,plot_index].set_ylabel('time [ms]')
	ax[0,plot_index].set_xlabel('radious [mm]')
	fig.colorbar(im1,ax=ax[0,plot_index]).set_label('Te [eV]')  # ;plt.pause(0.01)
	plot_index +=1
	im2 = ax[0,plot_index].pcolor(r_full,t_full,ne_prof_multipulse,cmap='rainbow')
	ax[0,plot_index].set_ylabel('time [ms]')
	ax[0,plot_index].set_xlabel('radious [mm]')
	fig.colorbar(im2,ax=ax[0,plot_index]).set_label('density [10^20 # m^-3]')  # ;plt.pause(0.01)
	plt.savefig('/home/ffederic/work/Collaboratory/test/experimental_data/'+folder+'/TS data/Ts_prof2d_multipulse_'+folder+'_' + str(i) + '.png', bbox_inches='tight')
	plt.close()
