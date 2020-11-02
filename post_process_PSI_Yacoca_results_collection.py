# this script is to collect together results from the Bayesian search and IR camera.
# The outcome I want are the plots for the paper

import numpy as np
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())
# import matplotlib.pyplot as plt
#import .functions
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
from functions.fabio_add import find_index_of_file
import collections

from adas import read_adf15,read_adf11
import os,sys
from PIL import Image
import xarray as xr
import pandas as pd
import copy
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.signal import find_peaks
from multiprocessing import Pool,cpu_count
number_cpu_available = cpu_count()
print('Number of cores available: '+str(number_cpu_available))


fdir = '/home/ffederic/work/Collaboratory/test/experimental_data'
df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv', index_col=0)
df_settings = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_3.csv',index_col=0)

results_summary = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/results_summary.csv',index_col=0)
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise', 'teal', 'olive']

# merge_ID_target_multipulse = [95,89,88,87,86,85]
merge_ID_target_multipulse = [99,98,96,97]

target_chamber_pressure = []
target_chamber_pressure_2 = []
target_OES_distance = []
feed_rate_SLM = []
CB_pulse_energy = []
delivered_pulse_energy = []
magnetic_field = []
area_equiv_max_static_pressure = []
T_pre_pulse = []
DT_pulse = []
pulse_en_semi_inf = []
area_of_interest_IR = []

net_power_removed_plasma_column = []
net_power_removed_plasma_column_sigma = []
tot_rad_power = []
tot_rad_power_sigma = []
power_rec_neutral = []
power_rec_neutral_sigma = []
power_rad_mol = []
power_rad_mol_sigma = []
power_rad_excit = []
power_rad_excit_sigma = []
power_rad_rec_bremm = []
power_rad_rec_bremm_sigma = []
max_CX_energy = []
max_average_static_pressure = []

j_specific_target_chamber_pressure = []
j_specific_target_OES_distance = []
j_specific_feed_rate_SLM = []
j_specific_CB_pulse_energy = []
j_specific_delivered_pulse_energy = []
j_specific_magnetic_field = []
j_specific_T_pre_pulse = []
j_specific_DT_pulse = []
j_specific_pulse_en_semi_inf = []
j_specific_area_of_interest_IR = []
for merge_ID_target in merge_ID_target_multipulse:
	target_chamber_pressure_2.append(np.float(results_summary.loc[merge_ID_target,['p_n [Pa]']]))
	if merge_ID_target != 88:
		target_chamber_pressure.append(np.float(results_summary.loc[merge_ID_target,['p_n [Pa]']]))
		target_OES_distance.append(np.float(results_summary.loc[merge_ID_target,['T_axial']]))
		feed_rate_SLM.append(np.float(results_summary.loc[merge_ID_target,['Seed']]))
		CB_pulse_energy.append(np.float(results_summary.loc[merge_ID_target,['CB energy [J]']]))
		delivered_pulse_energy.append(np.float(results_summary.loc[merge_ID_target,['Delivered energy [J]']]))
		magnetic_field.append(np.float(results_summary.loc[merge_ID_target,['B']]))
		area_equiv_max_static_pressure.append(np.float(results_summary.loc[merge_ID_target,['area_equiv_max_static_pressure']]))

		net_power_removed_plasma_column.append(np.float(results_summary.loc[merge_ID_target,['net_power_removed_plasma_column']]))
		net_power_removed_plasma_column_sigma.append(np.float(results_summary.loc[merge_ID_target,['net_power_removed_plasma_column_sigma']]))
		tot_rad_power.append(np.float(results_summary.loc[merge_ID_target,['tot_rad_power']]))
		tot_rad_power_sigma.append(np.float(results_summary.loc[merge_ID_target,['tot_rad_power_sigma']]))
		power_rec_neutral.append(np.float(results_summary.loc[merge_ID_target,['power_rec_neutral']]))
		power_rec_neutral_sigma.append(np.float(results_summary.loc[merge_ID_target,['power_rec_neutral_sigma']]))
		power_rad_mol.append(np.float(results_summary.loc[merge_ID_target,['power_rad_mol']]))
		power_rad_mol_sigma.append(np.float(results_summary.loc[merge_ID_target,['power_rad_mol_sigma']]))
		power_rad_excit.append(np.float(results_summary.loc[merge_ID_target,['power_rad_excit']]))
		power_rad_excit_sigma.append(np.float(results_summary.loc[merge_ID_target,['power_rad_excit_sigma']]))
		power_rad_rec_bremm.append(np.float(results_summary.loc[merge_ID_target,['power_rad_rec_bremm']]))
		power_rad_rec_bremm_sigma.append(np.float(results_summary.loc[merge_ID_target,['power_rad_rec_bremm_sigma']]))
		max_CX_energy.append(np.float(results_summary.loc[merge_ID_target,['max_CX_energy']]))
		max_average_static_pressure.append(np.float(results_summary.loc[merge_ID_target,['max_average_static_pressure']]))

	temp1=[]
	temp2=[]
	temp3=[]
	temp4=[]
	temp5=[]
	all_j=find_index_of_file(merge_ID_target,df_settings,df_log,only_OES=True)
	for j in all_j:
		j_specific_target_chamber_pressure.append(np.float(results_summary.loc[merge_ID_target,['p_n [Pa]']]))
		j_specific_target_OES_distance.append(np.float(results_summary.loc[merge_ID_target,['T_axial']]))
		j_specific_feed_rate_SLM.append(np.float(results_summary.loc[merge_ID_target,['Seed']]))
		j_specific_CB_pulse_energy.append(np.float(results_summary.loc[merge_ID_target,['CB energy [J]']]))
		j_specific_delivered_pulse_energy.append(np.float(results_summary.loc[merge_ID_target,['Delivered energy [J]']]))
		j_specific_magnetic_field.append(np.float(results_summary.loc[merge_ID_target,['B']]))

		j_specific_T_pre_pulse.append(df_log.loc[j,['T_pre_pulse']])
		temp1.append(df_log.loc[j,['T_pre_pulse']])
		j_specific_DT_pulse.append(df_log.loc[j,['DT_pulse']])
		temp2.append(df_log.loc[j,['DT_pulse']])
		j_specific_pulse_en_semi_inf.append(df_log.loc[j,['pulse_en_semi_inf [J]']])
		temp3.append(df_log.loc[j,['pulse_en_semi_inf [J]']])
		j_specific_area_of_interest_IR.append(df_log.loc[j,['area_of_interest [m2]']])
		temp4.append(df_log.loc[j,['area_of_interest [m2]']])

	T_pre_pulse.append(np.nanmean(temp1))
	DT_pulse.append(np.nanmean(temp2))
	if merge_ID_target != 88:
		pulse_en_semi_inf.append(np.nanmean(temp3))
		area_of_interest_IR.append(np.nanmean(temp4))

# plt.figure(figsize=(10, 5))
# plt.plot(j_specific_target_chamber_pressure,j_specific_T_pre_pulse,'+b')
# plt.plot(target_chamber_pressure,T_pre_pulse,'b')
# plt.plot(j_specific_target_chamber_pressure,np.array(j_specific_DT_pulse),'+r')
# plt.plot(target_chamber_pressure,np.array(DT_pulse),'r')
# plt.grid()
# plt.pause(0.001)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.set_title('Pressure scan magnetic_field %.3gT,target/OES distance %.3gmm,ELM pulse voltage %.3gV' %(magnetic_field[0],target_OES_distance[0],(2*CB_pulse_energy[0]/150e-6)**0.5)+'\nIR data analysis')
ax1.set_xlabel('Pressure [Pa]')
ax1.set_ylabel('Temp before ELM-like pulse [°C]', color='tab:blue')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel(r'$\Delta T$'+' 1.5ms after ELM-like pulse [°C]', color='tab:red')  # we already handled the x-label with ax1
ax1.plot(j_specific_target_chamber_pressure,j_specific_T_pre_pulse,'+b')
ax1.plot(target_chamber_pressure_2,T_pre_pulse,'b')
ax2.plot(j_specific_target_chamber_pressure,np.array(j_specific_DT_pulse),'+r')
ax2.plot(target_chamber_pressure_2,np.array(DT_pulse),'r')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:red')
# ax1.legend(loc='best', fontsize='x-small')
# ax2.legend(loc='best', fontsize='x-small')
ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)
ax1.grid()
# ax2.grid()
plt.pause(0.01)


plt.figure(figsize=(10, 5))
plt.plot(target_chamber_pressure,delivered_pulse_energy,linewidth=3,color=color[0],label=r'$E_{up}$')
plt.errorbar(target_chamber_pressure,net_power_removed_plasma_column,yerr=np.array(net_power_removed_plasma_column_sigma),linewidth=3,color=color[1],capsize=5,label=r'$E_{rem}$')
plt.errorbar(target_chamber_pressure,tot_rad_power,yerr=np.array(tot_rad_power_sigma),capsize=5,color=color[2],label=r'$E_{rad}$')
plt.errorbar(target_chamber_pressure,power_rec_neutral,yerr=np.array(power_rec_neutral_sigma),capsize=5,color=color[3],label=r'$E_{neut \: rec}$')
plt.errorbar(target_chamber_pressure,power_rad_mol,yerr=np.array(power_rad_mol_sigma),linestyle='--',capsize=5,color=color[4],label=r'$E_{rad \: mol}$')
plt.errorbar(target_chamber_pressure,power_rad_excit,yerr=np.array(power_rad_excit_sigma),linestyle='--',capsize=5,color=color[5],label=r'$E_{ex}$')
plt.errorbar(target_chamber_pressure,power_rad_rec_bremm,yerr=np.array(power_rad_rec_bremm_sigma),linestyle='--',capsize=5,color=color[6],label=r'$E_{rad \: rec+brem}$')
plt.plot(target_chamber_pressure,max_CX_energy,linewidth=3,color=color[7],label=r'$E_{CX \: max}$')
plt.plot(target_chamber_pressure,np.array(pulse_en_semi_inf)/np.array(area_of_interest_IR)*np.array(area_equiv_max_static_pressure)*6,linewidth=3,color=color[8],label=r'$6 \cdot E_{target}$')
plt.plot(target_chamber_pressure,np.array(net_power_removed_plasma_column) + np.array(max_CX_energy),linestyle='--',color=color[9],label=r'$E_{rem}+E_{CX \: max}$')
plt.plot(target_chamber_pressure,np.array(pulse_en_semi_inf)/np.array(area_of_interest_IR)*np.array(area_equiv_max_static_pressure)*6 + np.array(net_power_removed_plasma_column) + np.array(max_CX_energy),linewidth=3,color=color[9],label=r'$6 \cdot E_{target} + E_{rem}+E_{CX \: max}$')
plt.legend(loc='best', fontsize='xx-small',ncol=2,handleheight=2, labelspacing=0.00005)
plt.grid()
plt.title('Pressure scan magnetic_field %.3gT,target/OES distance %.3gmm,ELM pulse voltage %.3gV' %(magnetic_field[0],target_OES_distance[0],(2*CB_pulse_energy[0]/150e-6)**0.5))
plt.xlabel('Pressure [Pa]')
plt.ylabel('Energy [J]')
plt.pause(0.001)

# plt.figure()
# plt.plot(target_chamber_pressure,max_average_static_pressure)
# plt.title('Pressure scan magnetic_field %.3gT,target/OES distance %.3gmm,ELM pulse voltage %.3gV' %(magnetic_field[0],target_OES_distance[0],(2*CB_pulse_energy[0]/150e-6)**0.5))
# plt.grid()
# plt.xlabel('Pressure [Pa]')
# plt.ylabel('max averaged plasma static pressure [Pa]')
# plt.pause(0.001)


# end
