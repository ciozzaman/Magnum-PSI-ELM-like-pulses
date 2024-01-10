import numpy as np
from scipy import interpolate
from scipy.io import loadmat
import os
import collect_and_eval as coleval


path = os.getcwd()

f = []
for (dirpath, dirnames, filenames) in os.walk(path):
	f.extend(dirnames)

for folder in f:
	filename = coleval.all_file_names(path+'/'+folder,'mat')[0]

	x = loadmat(path+'/'+folder+'/'+filename)

	arguments = []
	for item in enumerate(x):
		if not item[1][0]=='_':
			# print(item[1]+'_' + folder)
			# print(item[1][0] == '_')
			print(item[1] + '_' + folder + '  shape='+str(np.shape(x[item[1]])))
			if np.min(np.shape(x[item[1]]))==1:
				exec(item[1] + '_' + folder + "=x['" + item[1] + "'].reshape((np.max(np.shape(x['" + item[1] + "']))))")
			else:
				exec(item[1] + '_' + folder + "=x['" + item[1] + "']")
			if item[1]=='PEC':
				funcname = item[1] + '_' + folder
			else:
				arguments.append(item[1] + '_' + folder)
		# exec(item[1] + '_' + folder + "=x['" + item[1] + "']")











path = os.getcwd()

f = []
for (dirpath, dirnames, filenames) in os.walk(path):
	f.extend(dirnames)

for folder in f:
	f = []
	for (dirpath, dirnames, filenames) in os.walk(path+'/'+folder):
		f.extend(filenames)
	filenames = coleval.all_file_names(path + '/' + folder, 'dat')
	filenames = coleval.order_filenames(filenames)

	data=[]
	for fname in filenames[1:]:
		print(np.shape(np.genfromtxt(path + '/' + folder+'/'+fname)))
		data.append(np.genfromtxt(path + '/' + folder+'/'+fname))
	data=np.array(data)
	print('creation of PEC_' + folder)
	exec('PEC_' + folder + "=data")


# H2+ + H- → H(p) + H2
# n = 3-7
Te_HmH2p_N = PEC_HmH2p_N[:,:,0].flatten()
# Te_HmH2p_N = np.repeat(Te_HmH2p_N,len(excited_states)).reshape((len(Te_HmH2p_N),len(excited_states))).T.flatten()
T_H2p_HmH2p_N = PEC_HmH2p_N[:,:,1].flatten()
T_Hm_HmH2p_N = PEC_HmH2p_N[:,:,2].flatten()
n_e_HmH2p_N = PEC_HmH2p_N[:,:,3].flatten()
PEC_val_HmH2p_N = PEC_HmH2p_N[:,:,4].flatten()
n_H2p_HmH2p_N = 1.0000e+15	# #/m^3
T_H_HmH2p_N = 50000/11600	#K

excited_states_HmH2p_N=len(PEC_HmH2p_N)
excited_states_HmH2p_N = np.linspace(3,3+excited_states_HmH2p_N-1,excited_states_HmH2p_N).astype('int')
excited_states_HmH2p_N = np.repeat(excited_states_HmH2p_N,len(PEC_HmH2p_N[0]))

HmH2p_N_pop_coeff = interpolate.LinearNDInterpolator(np.log(np.array([excited_states_HmH2p_N, Te_HmH2p_N, T_H2p_HmH2p_N, T_Hm_HmH2p_N, n_e_HmH2p_N]).T),np.log(PEC_val_HmH2p_N))
def HmH2p_N_pop_coeff_full(x,function=HmH2p_N_pop_coeff,excited_states=excited_states_HmH2p_N, coord_1=Te_HmH2p_N, coord_2=T_H2p_HmH2p_N, coord_3=T_Hm_HmH2p_N, coord_4=n_e_HmH2p_N):
	import numpy as np
	x=np.array(x)
	x[x[:,0]<np.min(excited_states),0]=np.min(excited_states)
	x[x[:,0]>np.max(excited_states),0]=np.max(excited_states)

	x[x[:,1]<np.min(coord_1),1]=np.min(coord_1)
	x[x[:,1]>np.max(coord_1),1]=np.max(coord_1)

	x[x[:,2]<np.min(coord_2),2]=np.min(coord_2)
	x[x[:,2]>np.max(coord_2),2]=np.max(coord_2)

	x[x[:,3]<np.min(coord_3),3]=np.min(coord_3)
	x[x[:,3]>np.max(coord_3),3]=np.max(coord_3)

	x[x[:,4]<np.min(coord_4),4]=np.min(coord_4)
	x[x[:,4]>np.max(coord_4),4]=np.max(coord_4)

	print(x)
	out = function(np.log(x))
	return np.exp(out)




# H2+ + e- → H(p) + H(1)
# H2+ + e- → H(p) + H+ + e-
# n = 3-7
Te_H2p = PEC_H2p[:,:,0].flatten()
T_H_H2p = PEC_H2p[:,:,1].flatten()
n_e_H2p = PEC_H2p[:,:,2].flatten()
PEC_val_H2p = PEC_H2p[:,:,3].flatten()

excited_states_H2p=len(PEC_H2p)
excited_states_H2p = np.linspace(3,3+excited_states_H2p-1,excited_states_H2p).astype('int')
excited_states_H2p = np.repeat(excited_states_H2p,len(PEC_H2p[0]))

H2p_pop_coeff = interpolate.LinearNDInterpolator(np.log(np.array([excited_states_H2p, Te_H2p, T_H_H2p, n_e_H2p]).T),np.log(PEC_val_H2p))
def H2p_pop_coeff_full(x,function=H2p_pop_coeff,excited_states=excited_states_H2p, coord_1=Te_H2p, coord_2=T_H_H2p, coord_3=n_e_H2p):
	import numpy as np
	x = np.array(x)
	x[x[:, 0] < np.min(excited_states), 0] = np.min(excited_states)
	x[x[:, 0] > np.max(excited_states), 0] = np.max(excited_states)

	x[x[:, 1] < np.min(coord_1), 1] = np.min(coord_1)
	x[x[:, 1] > np.max(coord_1), 1] = np.max(coord_1)

	x[x[:, 2] < np.min(coord_2), 2] = np.min(coord_2)
	x[x[:, 2] > np.max(coord_2), 2] = np.max(coord_2)

	x[x[:, 3] < np.min(coord_3), 3] = np.min(coord_3)
	x[x[:, 3] > np.max(coord_3), 3] = np.max(coord_3)

	print(x)
	out = function(np.log(x))
	return np.exp(out)


# H(q) + e- → H(p>q) + e-
# n = 3-7
Te_H2 = PEC_H2[:,:,0]
T_H_H2 = PEC_H2[:,:,1]
n_e_H2 = PEC_H2[:,:,2]
PEC_val_H2 = PEC_H2[:,:,3]

excited_states_H2=len(PEC_H2)
excited_states_H2 = np.linspace(3,3+excited_states_H2-1,excited_states_H2).astype('int')
excited_states_H2 = np.repeat(excited_states_H2,len(PEC_H2[0]))

H2_pop_coeff = interpolate.LinearNDInterpolator(np.log(np.array([excited_states_H2, Te_H2, T_H_H2, n_e_H2]).T),np.log(PEC_val_H2))
def H2_pop_coeff_full(x,function=H2_pop_coeff,excited_states=excited_states_H2, coord_1=Te_H2, coord_2=T_H_H2, coord_3=n_e_H2):
	import numpy as np
	x = np.array(x)
	x[x[:, 0] < np.min(excited_states), 0] = np.min(excited_states)
	x[x[:, 0] > np.max(excited_states), 0] = np.max(excited_states)

	x[x[:, 1] < np.min(coord_1), 1] = np.min(coord_1)
	x[x[:, 1] > np.max(coord_1), 1] = np.max(coord_1)

	x[x[:, 2] < np.min(coord_2), 2] = np.min(coord_2)
	x[x[:, 2] > np.max(coord_2), 2] = np.max(coord_2)

	x[x[:, 3] < np.min(coord_3), 3] = np.min(coord_3)
	x[x[:, 3] > np.max(coord_3), 3] = np.max(coord_3)

	print(x)
	out = function(np.log(x))
	return np.exp(out)



# H+ + H- → H(p) + H
# n = 3-7
Te_HmHp_N = PEC_HmHp_N[:,:,0]
T_Hm_HmHp_N = PEC_HmHp_N[:,:,1]
T_Hp_HmHp_N = PEC_HmHp_N[:,:,2]
n_e_HmHp_N = PEC_HmHp_N[:,:,3]
PEC_val_HmHp_N = PEC_HmHp_N[:,:,4]
n_Hp_HmHp_N = 1.0000e+15	# #/m^3
T_H_HmHp_N = 8000/11600	#K

excited_states_HmHp_N=len(PEC_HmHp_N)
excited_states_HmHp_N = np.linspace(3,3+excited_states_HmHp_N-1,excited_states_HmHp_N).astype('int')
excited_states_HmHp_N = np.repeat(excited_states_HmHp_N,len(PEC_HmHp_N[0]))

HmHp_N_pop_coeff = interpolate.LinearNDInterpolator(np.log(np.array([excited_states_HmHp_N, Te_HmHp_N, T_Hm_HmHp_N, T_Hp_HmHp_N,n_e_HmHp_N]).T),np.log(PEC_val_HmHp_N))
def HmHp_N_pop_coeff_full(x,function=HmHp_N_pop_coeff,excited_states=excited_states_HmHp_N, coord_1=Te_HmHp_N, coord_2=T_Hm_HmHp_N, coord_3=T_Hp_HmHp_N, coord_4=n_e_HmHp_N):
	import numpy as np
	x = np.array(x)
	x[x[:, 0] < np.min(excited_states), 0] = np.min(excited_states)
	x[x[:, 0] > np.max(excited_states), 0] = np.max(excited_states)

	x[x[:, 1] < np.min(coord_1), 1] = np.min(coord_1)
	x[x[:, 1] > np.max(coord_1), 1] = np.max(coord_1)

	x[x[:, 2] < np.min(coord_2), 2] = np.min(coord_2)
	x[x[:, 2] > np.max(coord_2), 2] = np.max(coord_2)

	x[x[:, 3] < np.min(coord_3), 3] = np.min(coord_3)
	x[x[:, 3] > np.max(coord_3), 3] = np.max(coord_3)

	x[x[:,4]<np.min(coord_4),4]=np.min(coord_4)
	x[x[:,4]>np.max(coord_4),4]=np.max(coord_4)

	print(x)
	out = function(np.log(x))
	return np.exp(out)


# H3+ + e- → H(p) + H2
# n = 3-7
Te_H3p = PEC_H3p[:,:,0]
T_H3p_H3p = PEC_H3p[:,:,1]
T_H_H3p = PEC_H3p[:,:,2]
n_e_H3p = PEC_H3p[:,:,3]
PEC_val_H3p = PEC_H3p[:,:,4]
n_Hp_H3p = 1.0000e+15	# #/m^3


excited_states_H3p=len(PEC_H3p)
excited_states_H3p = np.linspace(3,3+excited_states_H3p-1,excited_states_H3p).astype('int')
excited_states_H3p = np.repeat(excited_states_H3p,len(PEC_H3p[0]))

H3p_pop_coeff = interpolate.LinearNDInterpolator(np.log(np.array([excited_states_H3p, Te_H3p, T_H3p_H3p, T_H_H3p,n_e_H3p]).T),np.log(PEC_val_H3p))
def H3p_pop_coeff_full(x,function=H3p_pop_coeff,excited_states=excited_states_H3p, coord_1=Te_H3p, coord_2=T_H3p_H3p, coord_3=T_H_H3p, coord_4=n_e_H3p):
	import numpy as np
	x = np.array(x)
	x[x[:, 0] < np.min(excited_states), 0] = np.min(excited_states)
	x[x[:, 0] > np.max(excited_states), 0] = np.max(excited_states)

	x[x[:, 1] < np.min(coord_1), 1] = np.min(coord_1)
	x[x[:, 1] > np.max(coord_1), 1] = np.max(coord_1)

	x[x[:, 2] < np.min(coord_2), 2] = np.min(coord_2)
	x[x[:, 2] > np.max(coord_2), 2] = np.max(coord_2)

	x[x[:, 3] < np.min(coord_3), 3] = np.min(coord_3)
	x[x[:, 3] > np.max(coord_3), 3] = np.max(coord_3)

	x[x[:,4]<np.min(coord_4),4]=np.min(coord_4)
	x[x[:,4]>np.max(coord_4),4]=np.max(coord_4)

	print(x)
	out = function(np.log(x))
	return np.exp(out)














exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())
# import matplotlib.pyplot as plt
#import .functions
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
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


# n = np.arange(10, 20)
# waveLengths = [656.45377,486.13615,434.0462,410.174,397.0072,388.9049,383.5384] + list((364.6*(n*n)/(n*n-4)))
waveLengths = [486.13615,434.0462,410.174,397.0072,388.9049,383.5384]
waveLengths_interp = interpolate.interp1d(waveLengths[:2], [1.39146712e+03,8.83761543e+02],fill_value='extrapolate')
#pathfiles = '/home/ff645/GoogleDrive/ff645/YPI - Fabio/Collaboratory/Experimental data/tests/2019-04-25/final_test/Untitled_1/Pos0'
# pathfiles = '/home/ff645/GoogleDrive/ff645/YPI - Fabio/Collaboratory/Experimental data/tests/temp - Desktop pc/Untitled_11/Pos0'
# pathfiles = '/home/ff645/GoogleDrive/ff645/YPI - Fabio/Collaboratory/Experimental data/tests/test_data_folder/2019-04-25/01/Untitled_1/Pos0'


for merge_ID_target in [85,86,87,88,89,92,93,94]:

	# merge_ID_target = 85
	high_line = 8
	OES_multiplier = 1



	figure_index = 0
	fdir = '/home/ffederic/work/Collaboratory/test/experimental_data'
	df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv',index_col=0)
	df_settings = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_3.csv',index_col=0)

	# calculate_geometry = False
	# merge_ID_target = 17	#	THIS IS GIVEN BY THE LAUNCHER
	for i in range(10):
		print('.')
	print('Starting to work on merge number'+str(merge_ID_target))
	for i in range(10):
		print('.')


	time_resolution_scan = False
	time_resolution_scan_improved = True
	time_resolution_extra_skip = 0




	started=0
	rows_range_for_interp = 25/3 # rows that I use for interpolation (box with twice this side length, not sphere)
	# if merge_ID_target>=66:
	# 	rows_range_for_interp = geom_store['binInterv'][0] / 6
	if time_resolution_scan:
		conventional_time_step = 0.01	# ms
	else:
		conventional_time_step = 0.05	# ms
	# interpolation_type = 'quadratic'	# 'linear' or 'quadratic'
	grade_of_interpolation = 3	#this is the exponent used for the weights of the interpolation for image resampling
	type_of_image = '12bit'	# '12bit' or '16bit'
	# if type_of_image=='12bit':
	row_shift=2*10280/1000000	# ms
	# elif type_of_image=='16bit':
	# 	print('Row shift to be checked')
	# 	exit()
	# time_range_for_interp = rows_range_for_interp*row_shift
	# merge_time_window=[-1,4]
	merge_time_window=[-10,10]
	overexposed_treshold = 3600
	path_where_to_save_everything = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target)



	new_timesteps = np.linspace(merge_time_window[0]+0.5, merge_time_window[1]-0.5,int((merge_time_window[1]-0.5-(merge_time_window[0]+0.5))/conventional_time_step+1))
	dt=np.nanmedian(np.diff(new_timesteps))


	dx = 18 / 40 * (50.5 / 27.4) / 1e3
	xx = np.arange(40) * dx  # m
	xn = np.linspace(0, max(xx), 1000)
	r = np.linspace(-max(xx) / 2, max(xx) / 2, 1000)
	r=r[::10]
	dr=np.median(np.diff(r))



	# first_time = np.min(new_timesteps)
	# last_time = np.max(new_timesteps)


	# Data from wikipedia
	energy_difference = np.array([1.89,2.55,2.86,3.03,3.13,3.19,3.23,3.26, 3.29 ])	#eV
	# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
	statistical_weigth = np.array([32,50,72,98,128,162,200,242,288])	#gi-gk
	einstein_coeff = np.array([8.4193e-2,2.53044e-2,9.7320e-3,4.3889e-3,2.2148e-3,1.2156e-3,7.1225e-4,4.3972e-4,2.8337e-4])*1e8	#1/s
	J_to_eV = 6.242e18
	# Used formula 2.3 in Rion Barrois thesys, 2017
	color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise']

	inverted_profiles = 4*np.pi*np.load(path_where_to_save_everything+'/inverted_profiles.npy')		# in W m^-3 sr^-1  *  4*pi sr  =  W m^-3
	all_fits = np.load(path_where_to_save_everything +'/merge'+str(merge_ID_target)+'_all_fits.npy')	# in W m^-2 sr^-1
	merge_Te_prof_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_Te_prof_multipulse']
	merge_dTe_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_dTe_multipulse']
	merge_ne_prof_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_ne_prof_multipulse']
	merge_dne_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_dne_multipulse']
	merge_time = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_time']
	TS_dt=np.nanmedian(np.diff(merge_time))

	pecfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/pec12#h_balmer#h0.dat'
	scdfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/scd12_h.dat'
	acdfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/acd12_h.dat'


	if np.max(merge_Te_prof_multipulse)<=0:
		print('merge'+str(merge_ID_target)+" has no recorded temperature")
		continue

	TS_size=[-4.149230769230769056e+01,4.416923076923076508e+01]
	TS_r=TS_size[0] + np.linspace(0,1,65)*(TS_size[1]- TS_size[0])
	TS_dr = np.median(np.diff(TS_r))/1000
	gauss = lambda x, A, sig, x0: A * np.exp(-((x - x0) / sig) ** 2)
	profile_centres = []
	profile_centres_score = []
	for index in range(np.shape(merge_Te_prof_multipulse)[0]):
		yy = merge_Te_prof_multipulse[index]
		p0 = [np.max(yy),10,0]
		bds = [[0,-40,np.min(TS_r)],[np.inf,40,np.max(TS_r)]]
		fit = curve_fit(gauss, TS_r, yy, p0, maxfev=100000, bounds=bds)
		profile_centres.append(fit[0][-1])
		profile_centres_score.append(fit[1][-1,-1])
		# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)
	profile_centres = np.array(profile_centres)
	profile_centres_score = np.array(profile_centres_score)
	centre = np.nanmean(profile_centres[profile_centres_score<1])
	TS_r_new = np.abs(TS_r-centre)/1000
	# temp_r, temp_t = np.meshgrid(TS_r_new, merge_time)
	# plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse,vmin=0);plt.colorbar().set_label('Te [eV]');plt.pause(0.01)
	# plt.figure();plt.pcolor(temp_t,temp_r,merge_ne_prof_multipulse,vmin=0);plt.colorbar().set_label('ne [10^20 #/m^3]');plt.pause(0.01)


	temp1=np.zeros_like(inverted_profiles[:,0])
	temp2=np.zeros_like(inverted_profiles[:,0])
	temp3=np.zeros_like(inverted_profiles[:,0])
	temp4=np.zeros_like(inverted_profiles[:,0])
	interp_range_t = max(dt,TS_dt)*1
	interp_range_r = max(dx,TS_dr)*1
	for i_t,value_t in enumerate(new_timesteps):
		if np.sum(np.abs(merge_time-value_t) < interp_range_t) == 0:
			continue
		for i_r,value_r in enumerate(np.abs(r)):
			if np.sum(np.abs(TS_r_new - value_r) < interp_range_r) == 0:
				continue
			temp1[i_t,i_r] = np.mean(merge_Te_prof_multipulse[np.abs(merge_time-value_t)<interp_range_t][:,np.abs(TS_r_new-value_r)<interp_range_r])
			temp2[i_t,i_r] = np.max(merge_dTe_multipulse[np.abs(merge_time-value_t)<interp_range_t][:,np.abs(TS_r_new-value_r)<interp_range_r])/(np.sum( np.isfinite(merge_dTe_multipulse[np.abs(merge_time-value_t)<interp_range_t][:,np.abs(TS_r_new-value_r)<interp_range_r]))**0.5)
			temp3[i_t,i_r] = np.mean(merge_ne_prof_multipulse[np.abs(merge_time-value_t)<interp_range_t][:,np.abs(TS_r_new-value_r)<interp_range_r])
			temp4[i_t,i_r] = np.max(merge_dne_multipulse[np.abs(merge_time-value_t)<interp_range_t][:,np.abs(TS_r_new-value_r)<interp_range_r])/(np.sum( np.isfinite(merge_dne_multipulse[np.abs(merge_time-value_t)<interp_range_t][:,np.abs(TS_r_new-value_r)<interp_range_r]))**0.5)

	merge_Te_prof_multipulse_interp=np.array(temp1)
	merge_dTe_prof_multipulse_interp=np.array(temp2)
	merge_ne_prof_multipulse_interp=np.array(temp3)
	merge_dne_prof_multipulse_interp=np.array(temp4)
	temp_r, temp_t = np.meshgrid(r, new_timesteps)
	# plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse_interp,cmap='rainbow');plt.colorbar().set_label('Te [eV]')#;plt.pause(0.01)
	# plt.axes().set_aspect(100)
	# plt.xlabel('time [ms]')
	# plt.ylabel('radial location [m]')
	# plt.title('Te axially averaged and smoothed to match OES resolution')
	# figure_index+=1
	# plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	# plt.close()
	#
	# plt.figure();plt.pcolor(temp_t,temp_r,merge_ne_prof_multipulse_interp,cmap='rainbow');plt.colorbar().set_label('ne [10^20 #/m^3]')#;plt.pause(0.01)
	# plt.axes().set_aspect(100)
	# plt.xlabel('time [ms]')
	# plt.ylabel('radial location [m]')
	# plt.title('ne axially averaged and smoothed to match OES resolution')
	# figure_index+=1
	# plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	# plt.close()


	# plt.figure();plt.pcolor(temp_t,temp_r,inverted_profiles[:,0]);plt.colorbar();plt.pause(0.01)




	# merge_Te_prof_multipulse_interp_fun = bisplrep((merge_time*np.ones((len(merge_time),len(TS_r_new))).T).T,TS_r_new*np.ones((len(merge_time),len(TS_r_new))), merge_Te_prof_multipulse,kx=3,ky=3,s=40)#len(merge_time)*len(TS_r_new)/10)
	# # d_interpolated = bisplev(pixels_centre_location[1], pixels_centre_location[0], foil_power_interpolator)
	# merge_Te_prof_multipulse_interp = bisplev(new_timesteps,np.sort(np.abs(r)), merge_Te_prof_multipulse_interp_fun)
	# temp_r, temp_t = np.meshgrid(np.sort(np.abs(r)), new_timesteps)
	# plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse_interp);plt.colorbar();plt.pause(0.01)
	#
	# merge_ne_prof_multipulse_interp_fun = bisplrep((merge_time*np.ones((len(merge_time),len(TS_r_new))).T).T,TS_r_new*np.ones((len(merge_time),len(TS_r_new))), merge_ne_prof_multipulse,kx=5,ky=5,s=100000)#len(merge_time)*len(TS_r_new)/10)
	# merge_ne_prof_multipulse_interp = bisplev(new_timesteps,np.sort(np.abs(r)), merge_ne_prof_multipulse_interp_fun)
	# temp_r, temp_t = np.meshgrid(np.sort(np.abs(r)), new_timesteps)
	# plt.figure();plt.pcolor(temp_t,temp_r,merge_ne_prof_multipulse_interp);plt.colorbar();plt.pause(0.01)
	#
	# # plt.figure();plt.imshow(merge_Te_prof_multipulse_interp,vmin=0);plt.colorbar();plt.pause(0.01)




	# I crop to the usefull stuff
	start_time = np.abs(new_timesteps-0).argmin()
	end_time = np.abs(new_timesteps-1.5).argmin()+1
	time_crop = new_timesteps[start_time:end_time]
	start_r = np.abs(r-0).argmin()
	end_r = np.abs(r-5).argmin()+1
	r_crop = r[start_r:end_r]
	temp_r, temp_t = np.meshgrid(r_crop, time_crop)
	merge_Te_prof_multipulse_interp_crop = merge_Te_prof_multipulse_interp[start_time:end_time,start_r:end_r]
	merge_ne_prof_multipulse_interp_crop = merge_ne_prof_multipulse_interp[start_time:end_time,start_r:end_r]
	merge_dTe_prof_multipulse_interp_crop = merge_dTe_prof_multipulse_interp[start_time:end_time,start_r:end_r]
	merge_dne_prof_multipulse_interp_crop = merge_dne_prof_multipulse_interp[start_time:end_time,start_r:end_r]
	inverted_profiles_crop = inverted_profiles[start_time:end_time,:,start_r:end_r]
	inverted_profiles_crop[np.isnan(inverted_profiles_crop)] = 0
	# inverted_profiles_crop[inverted_profiles_crop<0] = 0


	plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse_interp_crop,cmap='rainbow');plt.colorbar().set_label('Te [eV]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('Te axially averaged and smoothed to match OES resolution')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse_interp_crop,cmap='rainbow',vmax=0.2);plt.colorbar().set_label('Te [eV]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('Te axially averaged and smoothed to match OES resolution\nonly values<0.2eV where ADAS data become unreliable')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse_interp_crop,cmap='rainbow',vmax=1);plt.colorbar().set_label('Te [eV]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('Te axially averaged and smoothed to match OES resolution\nonly values<1eV where YACORA data become unreliable')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()


	temp = merge_dTe_prof_multipulse_interp_crop/merge_Te_prof_multipulse_interp_crop
	temp[np.isnan(temp)]=0
	plt.figure();plt.pcolor(temp_t,temp_r,temp,norm=LogNorm(),cmap='rainbow');plt.colorbar().set_label('relative error [au]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('merge_Te_prof_multipulse_interp_crop uncertainty from TS')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	plt.figure();plt.pcolor(temp_t,temp_r,merge_ne_prof_multipulse_interp_crop,cmap='rainbow');plt.colorbar().set_label('ne [10^20 #/m^3]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('ne axially averaged and smoothed to match OES resolution')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	temp = merge_dne_prof_multipulse_interp_crop/merge_ne_prof_multipulse_interp_crop
	temp[np.isnan(temp)]=0
	plt.figure();plt.pcolor(temp_t,temp_r,temp,norm=LogNorm(),cmap='rainbow');plt.colorbar().set_label('relative error [au]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('merge_ne_prof_multipulse_interp_crop uncertainty from TS')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()



	merge_Te_prof_multipulse_interp_crop_limited = cp.deepcopy(merge_Te_prof_multipulse_interp_crop)
	merge_Te_prof_multipulse_interp_crop_limited[merge_Te_prof_multipulse_interp_crop_limited<0.2]=0
	merge_ne_prof_multipulse_interp_crop_limited = cp.deepcopy(merge_ne_prof_multipulse_interp_crop)
	merge_ne_prof_multipulse_interp_crop_limited[merge_ne_prof_multipulse_interp_crop_limited<5e-07]=0
	excitation = []
	for isel in [2,3,4,5,6,7,8,9,10]:
		temp = read_adf15(pecfile, isel, merge_Te_prof_multipulse_interp_crop_limited.flatten(), (merge_ne_prof_multipulse_interp_crop_limited*10**(20-6)).flatten())[0]	# ADAS database is in cm^3   # photons s^-1 cm^-3
		temp[np.isnan(temp)] = 0
		temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
		excitation.append(temp)
	excitation = np.array(excitation)	# in # photons cm^-3 s^-1
	excitation = (excitation.T*(10**-6)*(energy_difference/J_to_eV)).T	# in W m^-3 / (# / m^3)**2

	recombination = []
	for isel in [20,21,22,23,24,25,26,27,28]:
		temp = read_adf15(pecfile, isel, merge_Te_prof_multipulse_interp_crop_limited.flatten(), (merge_ne_prof_multipulse_interp_crop_limited*10**(20-6)).flatten())[0]	# ADAS database is in cm^3   # photons s^-1 cm^-3
		temp[np.isnan(temp)] = 0
		temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
		recombination.append(temp)
	recombination = np.array(recombination)	# in # photons cm^-3 s^-1 / (# cm^-3)**2
	recombination = (recombination.T*(10**-6)*(energy_difference/J_to_eV)).T	# in W m^-3 / (# / m^3)**2



	recombination_emissivity = (recombination * (merge_ne_prof_multipulse_interp_crop_limited*(10**20))**2).astype('float')
	# plt.figure();plt.pcolor(temp_t,temp_r,recombination_emissivity[0],vmax=np.nanmax(inverted_profiles[:,0]));plt.colorbar();plt.pause(0.01)
	plt.figure();plt.pcolor(temp_t,temp_r,recombination_emissivity[high_line-4],cmap='rainbow');plt.colorbar().set_label('line emission [W m^-3]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('recombination_emissivity line'+str(high_line)+' from ne,Te')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	plt.figure();plt.pcolor(temp_t,temp_r,inverted_profiles_crop[:,high_line-4],cmap='rainbow');plt.colorbar().set_label('line emission [W m^-3]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('inverted_profiles_crop line'+str(high_line)+' from OES')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	plt.figure();plt.pcolor(temp_t,temp_r,recombination_emissivity[0],cmap='rainbow');plt.colorbar().set_label('line emission [W m^-3]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('recombination_emissivity line4 from ne,Te')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	plt.figure();plt.pcolor(temp_t,temp_r,inverted_profiles_crop[:,0],cmap='rainbow');plt.colorbar().set_label('line emission [W m^-3]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('inverted_profiles_crop line4 from OES')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	plt.figure();plt.pcolor(temp_t,temp_r,recombination_emissivity[high_line-4]/np.max(recombination_emissivity[high_line-4]),cmap='rainbow');plt.colorbar().set_label('relative line emission [au]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('relative recombination_emissivity line'+str(high_line)+' from ne,Te')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	plt.figure();plt.pcolor(temp_t,temp_r,excitation[high_line-4]/np.max(excitation[high_line-4]),cmap='rainbow');plt.colorbar().set_label('relative line emission [au]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('relative excitation line'+str(high_line)+' from ne,Te')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	# the peaks of excitataion and density are apart, so I scale up the OES measurements to match the expected data where I have the maximum total radiation

	# integrated_recombination_emissitivy = np.sum(recombination_emissivity[high_line-4]*dr*dt*2*np.pi*r_crop,axis=-1)
	# integrated_measured_emissitivy = np.sum(inverted_profiles_crop[:,high_line-4]*dr*dt*2*np.pi*r_crop,axis=-1)
	#
	#
	# difference_all=[]
	# for line in range(4,13):
	# 	OES_multiplier = 1
	# 	difference_all_line = inverted_profiles_crop[:,line-4]*OES_multiplier - recombination_emissivity[line-4]
	# 	min_mean = 2*np.std(difference_all_line)
	# 	while np.mean(difference_all_line)<min_mean:
	# 		OES_multiplier+=1.2
	# 		difference_all_line = inverted_profiles_crop[:, line - 4] * OES_multiplier - recombination_emissivity[line - 4]
	# 	print('line ' + str(line))
	# 	print(OES_multiplier)
	# 	difference_all.append(difference_all_line)
	# 	plt.figure();
	# 	plt.pcolor(temp_t, temp_r, difference_all_line,vmax=0, cmap='rainbow');
	# 	plt.colorbar().set_label('line emission [W m^-3 sr^-1]')  # ;plt.pause(0.01)
	# 	plt.axes().set_aspect(20)
	# 	plt.xlabel('time [ms]')
	# 	plt.ylabel('radial location [m]')
	# 	plt.title('difference between inverted_profiles and recombination_emissivity\n and recombination_emissivity for line' + str(line) + '\nI attribute this to excitation')
	# 	plt.close()
	# difference_all = np.array(difference_all)
	#
	#
	#
	# difference_all=[]
	# for line in range(4,13):
	# 	OES_multiplier = 1
	# 	difference_all_line = inverted_profiles_crop[:,line-4]*OES_multiplier - recombination_emissivity[line-4]
	# 	min_mean = 2*np.std(difference_all_line)
	# 	while np.mean(difference_all_line)<0:
	# 		OES_multiplier+=1.2
	# 		difference_all_line = inverted_profiles_crop[:, line - 4] * OES_multiplier - recombination_emissivity[line - 4]
	# 	print('line ' + str(line))
	# 	print(OES_multiplier)
	# 	difference_all.append(difference_all_line)
	# 	plt.figure();
	# 	plt.pcolor(temp_t, temp_r, difference_all_line,vmax=0, cmap='rainbow');
	# 	plt.colorbar().set_label('line emission [W m^-3 sr^-1]')  # ;plt.pause(0.01)
	# 	plt.axes().set_aspect(20)
	# 	plt.xlabel('time [ms]')
	# 	plt.ylabel('radial location [m]')
	# 	plt.title('difference between inverted_profiles and recombination_emissivity\n and recombination_emissivity for line' + str(line) + '\nI attribute this to excitation')
	# 	plt.close()
	# difference_all = np.array(difference_all)


	# if False:
	# 	OES_multiplier = integrated_recombination_emissitivy.max()/integrated_measured_emissitivy[integrated_recombination_emissitivy.argmax()]
	# 	OES_multiplier = recombination_emissivity[high_line-4].max()/(inverted_profiles_crop[:,high_line-4].flatten()[recombination_emissivity[high_line-4].argmax()])
	# elif True:
	# 	OES_multiplier = 1
	difference = inverted_profiles_crop[:,high_line-4]*OES_multiplier - recombination_emissivity[high_line-4]
	plt.figure();plt.pcolor(temp_t,temp_r,difference,cmap='rainbow',vmin=max(np.min(difference),-np.max(difference)));plt.colorbar().set_label('line emission [W m^-3 sr^-1]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('difference between inverted_profiles\n and recombination_emissivity for line'+str(high_line)+'\nI attribute this to excitation')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()


	# I want to find what is the best line to calculate the neutral hydrogen atomic density
	# best means that it fits best with ower lines

	if False:
		score = []
		for line_index in [6,7,8,9,10,11,12]:
			difference = inverted_profiles_crop[:, line_index - 4] * OES_multiplier - recombination_emissivity[line_index - 4]
			h_atomic_density = difference/(excitation[line_index-4] * (merge_ne_prof_multipulse_interp_crop*(10**20))).astype('float')
			h_atomic_density[np.logical_not(np.isfinite(h_atomic_density))]=0
			h_atomic_density[h_atomic_density<0]=0

			excitation_emissivity = (excitation * (merge_ne_prof_multipulse_interp_crop*(10**20))*(h_atomic_density)).astype('float')

			residual_emission=[]
			for index in range(len(recombination_emissivity)):
				residual_emission.append(inverted_profiles_crop[:,index]*OES_multiplier-recombination_emissivity[index]-excitation_emissivity[index])
			residual_emission=np.array(residual_emission)

			score.append(np.min(residual_emission[0]))
		print('score')
		print(score)


	h_atomic_density = difference/(excitation[high_line-4] * (merge_ne_prof_multipulse_interp_crop*(10**20))).astype('float')
	h_atomic_density[np.logical_not(np.isfinite(h_atomic_density))]=0
	h_atomic_density[h_atomic_density<0]=0
	if np.max(h_atomic_density[np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007])<=0:
		print('merge'+str(merge_ID_target)+" didn't allow for any neutral atomic hydrogen")
		continue
	plt.figure();plt.pcolor(temp_t,temp_r,h_atomic_density,norm=LogNorm(),cmap='rainbow',vmax=np.max(h_atomic_density[np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label('n0 [# m^-3]')#;plt.pause(0.01)
	# plt.figure();plt.pcolor(temp_t,temp_r,h_atomic_density,norm=LogNorm(),cmap='rainbow',vmax=np.);plt.colorbar().set_label('n0 [# m^-3]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('estimated hydrogen neutral atomic density from excitation')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()


	excitation_emissivity = (excitation * (merge_ne_prof_multipulse_interp_crop*(10**20))*(h_atomic_density)).astype('float')
	# plt.figure();plt.pcolor(temp_t,temp_r,excitation_emissivity[0],cmap='rainbow',vmax=np.max(excitation_emissivity[0][np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label('line emission [W m^-3 sr^-1]')#;plt.pause(0.01)
	plt.figure();plt.pcolor(temp_t,temp_r,excitation_emissivity[0],cmap='rainbow');plt.colorbar().set_label('line emission [W m^-3 sr^-1]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('excitation_emissivity line4')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	plt.figure();plt.pcolor(temp_t,temp_r,excitation_emissivity[high_line-4],cmap='rainbow',vmax=np.max(excitation_emissivity[high_line-4][np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label('line emission [W m^-3 sr^-1]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('excitation_emissivity line'+str(high_line))
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	print(np.max(excitation_emissivity))

	residual_emission=[]
	for index in range(len(recombination_emissivity)):
		residual_emission.append(inverted_profiles_crop[:,index]*OES_multiplier-recombination_emissivity[index]-excitation_emissivity[index])
	residual_emission=np.array(residual_emission)
	# residual_emission = inverted_profiles_crop-recombination_emissivity-excitation_emissivity
	plt.figure();plt.pcolor(temp_t,temp_r,residual_emission[0],cmap='rainbow',vmin=np.min(residual_emission[0][np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label('line emission [W m^-3 sr^-1]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('residual emissivity, attribuited to MAR, line4')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	temp = residual_emission[0]/recombination_emissivity[0]
	temp[np.logical_not( np.isfinite(temp))] = 0
	plt.figure();plt.pcolor(temp_t,temp_r,temp,cmap='rainbow',vmax=np.max(temp[np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]),vmin=np.min(temp[np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label(' relative line emission [au]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('residual emissivity, attribuited to MAR, line4')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	plt.figure();plt.pcolor(temp_t,temp_r,residual_emission[1],cmap='rainbow',vmin=np.min(residual_emission[1][np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label('line emission [W m^-3 sr^-1]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('residual emissivity, attribuited to MAR, line5')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	temp = residual_emission[1]/recombination_emissivity[1]
	temp[np.logical_not( np.isfinite(temp))] = 0
	plt.figure();plt.pcolor(temp_t,temp_r,temp,cmap='rainbow',vmax=np.max(temp[np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]),vmin=np.min(temp[np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label(' relative line emission [au]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('residual emissivity, attribuited to MAR, line5')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	plt.figure();plt.pcolor(temp_t,temp_r,residual_emission[2],cmap='rainbow',vmin=np.min(residual_emission[2][np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label('line emission [W m^-3 sr^-1]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('residual emissivity, attribuited to MAR, line6')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	temp = residual_emission[2]/recombination_emissivity[2]
	temp[np.logical_not( np.isfinite(temp))] = 0
	plt.figure();plt.pcolor(temp_t,temp_r,temp,cmap='rainbow',vmax=np.max(temp[np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]),vmin=np.min(temp[np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label(' relative line emission [au]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('residual emissivity, attribuited to MAR, line6')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	plt.figure();plt.pcolor(temp_t,temp_r,residual_emission[high_line-4],cmap='rainbow',vmin=np.min(residual_emission[high_line-4][np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label('line emission [W m^-3 sr^-1]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('residual emissivity, attribuited to MAR, line'+str(high_line))
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	temp = residual_emission[high_line-4]/recombination_emissivity[high_line-4]
	temp[np.logical_not( np.isfinite(temp))] = 0
	plt.figure();plt.pcolor(temp_t,temp_r,temp,cmap='rainbow',vmax=np.max(temp[np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]),vmin=np.min(temp[np.logical_and(time_crop>0.3,time_crop<0.8)][:,r_crop<0.007]));plt.colorbar().set_label(' relative line emission [au]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('residual emissivity, attribuited to MAR, line'+str(high_line))
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()


	temp = read_adf11(scdfile, 'scd', 1,1,1,merge_Te_prof_multipulse_interp_crop_limited.flatten(), (merge_ne_prof_multipulse_interp_crop_limited*10**(20-6)).flatten())
	temp[np.isnan(temp)] = 0
	effective_ionisation_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop)))*(10**-6)	# in ionisations m^-3 s-1 / (# / m^3)**2
	effective_ionisation_rates = (effective_ionisation_rates * (merge_ne_prof_multipulse_interp_crop*(10**20))*h_atomic_density).astype('float')
	plt.figure();plt.pcolor(temp_t,temp_r,effective_ionisation_rates,cmap='rainbow');plt.colorbar().set_label('effective_ionisation_rates [# m^-3 s-1]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('effective_ionisation_rates')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()


	temp = read_adf11(acdfile, 'acd', 1,1,1,merge_Te_prof_multipulse_interp_crop_limited.flatten(), (merge_ne_prof_multipulse_interp_crop_limited*10**(20-6)).flatten())
	temp[np.isnan(temp)] = 0
	effective_recombination_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop)))*(10**-6)	# in recombinations m^-3 s-1 / (# / m^3)**2
	effective_recombination_rates = (effective_recombination_rates * (merge_ne_prof_multipulse_interp_crop*(10**20))**2).astype('float')
	plt.figure();plt.pcolor(temp_t,temp_r,effective_recombination_rates,cmap='rainbow');plt.colorbar().set_label('effective_recombination_rates [# m^-3 s-1]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('effective_recombination_rates (three body plus radiative)')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()



	line_integrated_recombination = []
	x_local = xx - 17.4/1000
	dr_crop = np.median(np.diff(r_crop))
	for x_value in np.abs(x_local):
		y=np.zeros_like(r_crop)
		y_prime = np.zeros_like(r_crop)
		for i_r,r_value in enumerate(r_crop):
			# print(r_value)
			if (r_value+dr_crop/2)<(x_value):
				# print(str([r_value+dr_crop/2,x_value-dx/2]))
				continue
			elif (x_value>=(r_value - dr_crop/2) and x_value<=(r_value + dr_crop/2)):
				y[i_r] = np.sqrt((r_value+dr_crop/2)**2- x_value**2)
				# print([r_value+dr_crop/2,x_value])
				# print(y[i_r])
			elif r_value - dr_crop>0:
				y[i_r] = np.sqrt((r_value+dr_crop/2)**2 - x_value**2)
				y_prime[i_r] = np.sqrt((r_value - dr_crop / 2) ** 2 - x_value**2)
			else:
				y[i_r] = np.sqrt((r_value+dr_crop/2)**2)
				# print([r_value+dr_crop/2,x_value])
				# print(y[i_r])
		dy = 2*(y-y_prime)
		if np.sum(np.isnan(dy))>0:
			print(str(x_value)+' is bad')
			print(dy)
		line_integrated_recombination.append(np.sum(recombination_emissivity*dy,axis=-1))
	line_integrated_recombination = np.array(line_integrated_recombination)

	all_fits[np.isnan(all_fits)]=0
	plt.figure();plt.plot(np.nanmax(4*np.pi*all_fits,axis=(0,1)),label='from OES')
	plt.plot(np.max(line_integrated_recombination[:,:,np.logical_and(time_crop<0.9,time_crop>0.4)],axis=(0,-1)),label='expected')
	plt.legend(loc='best')
	plt.ylabel('line integrated emission peak value')
	plt.xlabel('line index-4')
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()


	# plt.figure();plt.imshow(line_integrated_recombination[:,0],cmap='rainbow',vmax=400);plt.colorbar().set_label('effective_recombination_rates [# m^-3 s-1]')#;plt.pause(0.01)
	# plt.figure();
	# plt.imshow(line_integrated_recombination[:, 5], cmap='rainbow');
	# plt.colorbar().set_label('effective_recombination_rates [# m^-3 s-1]')  # ;plt.pause(0.01)

	temp_xx, temp_t = np.meshgrid(x_local, time_crop)

	plt.figure();plt.pcolor(temp_t,temp_xx,line_integrated_recombination[:,0].T,cmap='rainbow');plt.colorbar().set_label('effective_recombination_rates [# m^-3 s-1]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('effective_recombination_rates (three body plus radiative) line'+str(4))
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	plt.figure();plt.pcolor(temp_t,temp_xx,line_integrated_recombination[:,5-4].T,cmap='rainbow');plt.colorbar().set_label('effective_recombination_rates [# m^-3 s-1]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('effective_recombination_rates (three body plus radiative) line'+str(5))
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	plt.figure();plt.pcolor(temp_t,temp_xx,line_integrated_recombination[:,6-4].T,cmap='rainbow');plt.colorbar().set_label('effective_recombination_rates [# m^-3 s-1]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('effective_recombination_rates (three body plus radiative) line'+str(6))
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	plt.figure();plt.pcolor(temp_t,temp_xx,line_integrated_recombination[:,high_line-4].T,cmap='rainbow');plt.colorbar().set_label('effective_recombination_rates [# m^-3 s-1]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('effective_recombination_rates (three body plus radiative) line'+str(high_line-4))
	figure_index+=1
	plt.savefig(path_where_to_save_everything + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

