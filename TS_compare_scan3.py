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
from scipy.signal import find_peaks

os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data/functions/ADAS_opacity')
from adas import read_adf15
from idlbridge import export_procedure

fdir = '/home/ffederic/work/Collaboratory/test/experimental_data'
df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
df_settings = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_3.csv',index_col=0)
results_summary = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/results_summary.csv',index_col=0)

TS_size=[-4.149230769230769056e+01,4.416923076923076508e+01]
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro','paleturquoise']
line_style = ['-','--',':','-.']
marker = ['x','+']


fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('Pressure [Pa]')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax3.spines["right"].set_position(("axes", 1.1))
for i_to_scan,to_scan in enumerate([[99,98,96,97],[95,89,87,86,85]]):
	magnetic_field_all = []
	average_static_pressure = []
	average_Te = []
	average_ne = []
	target_chamber_pressure_2 = []

	for merge_ID_target in to_scan:
		print('Looking at '+str(merge_ID_target))
		target_chamber_pressure_2.append(np.float(results_summary.loc[merge_ID_target,['p_n [Pa]']]))
		average_static_pressure.append(np.float(results_summary.loc[merge_ID_target,['average_static_pressure']]))
		average_Te.append(np.float(results_summary.loc[merge_ID_target,['average_Te']]))
		average_ne.append(np.float(results_summary.loc[merge_ID_target,['average_ne']]))
		magnetic_field_all.append(np.float(results_summary.loc[merge_ID_target,['B']]))


	a1, = ax1.plot(target_chamber_pressure_2,average_Te,ls=line_style[i_to_scan],color='r',label='Te B=%.3gT' %(np.mean(magnetic_field_all)))
	a2, = ax2.plot(target_chamber_pressure_2,average_ne,ls=line_style[i_to_scan],color='b',label='ne B=%.3gT' %(np.mean(magnetic_field_all)))
	a3, = ax3.plot(target_chamber_pressure_2,average_static_pressure,ls=line_style[i_to_scan],color='g',label='pressure B=%.3gT' %(np.mean(magnetic_field_all)))
	# ax1.plot(target_chamber_pressure_SS_all,np.max(merge_Te_SS_all,axis=(1)),ls=line_style[i_to_scan],color='r',label='SS Te B=%.3gT' %(np.mean(magnetic_field_all)))
	# ax2.plot(target_chamber_pressure_SS_all,np.max(merge_ne_SS_all,axis=(1)),ls=line_style[i_to_scan],color='c',label='SS ne B=%.3gT' %(np.mean(magnetic_field_all)))

ax1.tick_params(axis='y', labelcolor=a1.get_color())
ax2.tick_params(axis='y', labelcolor=a2.get_color())
ax3.tick_params(axis='y', labelcolor=a3.get_color())
ax1.set_ylabel('average peak Te [eV]', color=a1.get_color())
ax2.set_ylabel('average peak ne [10^20 #/m3]', color=a2.get_color())  # we already handled the x-label with ax1
ax3.set_ylabel('average peak static pressure [Pa]', color=a3.get_color())  # we already handled the x-label with ax1
ax1.legend(loc=7, fontsize='x-small')
ax2.legend(loc=6, fontsize='x-small')
ax3.legend(loc=4, fontsize='x-small')
ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)
ax3.set_ylim(bottom=0)
ax1.grid()
# ax2.grid()
plt.pause(0.01)
