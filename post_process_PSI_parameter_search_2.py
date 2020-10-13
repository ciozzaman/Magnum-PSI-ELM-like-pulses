import numpy as np
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
from scipy.optimize import curve_fit,least_squares
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

fdir = '/home/ffederic/work/Collaboratory/test/experimental_data'
df_log = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/shots_3.csv', index_col=0)
df_settings = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/functions/Log/settings_3.csv',index_col=0)

pecfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/pec12#h_balmer#h0.dat'
scdfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/scd12_h.dat'
acdfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/acd12_h.dat'


OES_multiplier=1
nH_ne=1
make_plots = False
time_shift_factor=0
spatial_factor=1



merge_ID_target = 85


figure_index = 0

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


mod = '/spatial_factor' + str(spatial_factor)
if not os.path.exists(path_where_to_save_everything + mod):
	os.makedirs(path_where_to_save_everything + mod)

dx = (18 / 40 * (50.5 / 27.4) / 1e3) * spatial_factor
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

inverted_profiles_original = 4*np.pi*np.load(path_where_to_save_everything+'/inverted_profiles.npy')		# in W m^-3 sr^-1  *  4*pi sr  =  W m^-3
all_fits = np.load(path_where_to_save_everything +'/merge'+str(merge_ID_target)+'_all_fits.npy')	# in W m^-2 sr^-1
merge_Te_prof_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_Te_prof_multipulse']
merge_dTe_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_dTe_multipulse']
merge_ne_prof_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_ne_prof_multipulse']
merge_dne_multipulse = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_dne_multipulse']
merge_time_original = np.load(path_where_to_save_everything+'/TS_data_merge_' + str(merge_ID_target) +'.npz')['merge_time']

mod = '/spatial_factor' + str(spatial_factor)+'/time_shift_factor' + str(time_shift_factor)
if not os.path.exists(path_where_to_save_everything + mod):
	os.makedirs(path_where_to_save_everything + mod)


merge_time = time_shift_factor + merge_time_original
inverted_profiles = (1 / spatial_factor) * inverted_profiles_original  # in W m^-3 sr^-1  *  4*pi sr  =  W m^-3
TS_dt=np.nanmedian(np.diff(merge_time))




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
interp_range_t = max(dt/2,TS_dt)*1
interp_range_r = max(dx/2,TS_dr)*1
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
all_fits_crop = all_fits[start_time:end_time]
# inverted_profiles_crop[inverted_profiles_crop<0] = 0

x_local = xx - spatial_factor * 17.4 / 1000
dr_crop = np.median(np.diff(r_crop))
all_dy = []
for x_value in np.abs(x_local):
	y = np.zeros_like(r_crop)
	y_prime = np.zeros_like(r_crop)
	for i_r, r_value in enumerate(r_crop):
		# print(r_value)
		if (r_value + dr_crop / 2) < (x_value):
			# print(str([r_value+dr_crop/2,x_value-dx/2]))
			continue
		elif (x_value >= (r_value - dr_crop / 2) and x_value <= (r_value + dr_crop / 2)):
			y[i_r] = np.sqrt((r_value + dr_crop / 2) ** 2 - x_value ** 2)
		# print([r_value+dr_crop/2,x_value])
		# print(y[i_r])
		elif r_value - dr_crop > 0:
			y[i_r] = np.sqrt((r_value + dr_crop / 2) ** 2 - x_value ** 2)
			y_prime[i_r] = np.sqrt((r_value - dr_crop / 2) ** 2 - x_value ** 2)
		else:
			y[i_r] = np.sqrt((r_value + dr_crop / 2) ** 2)
	# print([r_value+dr_crop/2,x_value])
	# print(y[i_r])
	dy = 2 * (y - y_prime)
	all_dy.append(dy)
	if np.sum(np.isnan(dy)) > 0:
		print(str(x_value) + ' is bad')
		print(dy)
# line_integrated_recombination.append(np.sum(recombination_emissivity*dy,axis=-1))
all_dy = np.array(all_dy)


if make_plots:
	plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse_interp_crop,cmap='rainbow');plt.colorbar().set_label('Te [eV]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('Te axially averaged and smoothed to match OES resolution')
	figure_index+=1
	plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse_interp_crop,cmap='rainbow',vmax=0.2);plt.colorbar().set_label('Te [eV]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('Te axially averaged and smoothed to match OES resolution\nonly values<0.2eV where ADAS data become unreliable')
	figure_index+=1
	plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse_interp_crop,cmap='rainbow',vmax=1);plt.colorbar().set_label('Te [eV]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('Te axially averaged and smoothed to match OES resolution\nonly values<1eV where YACORA data become unreliable')
	figure_index+=1
	plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()


	temp = merge_dTe_prof_multipulse_interp_crop/merge_Te_prof_multipulse_interp_crop
	temp[np.isnan(temp)]=0
	plt.figure();plt.pcolor(temp_t,temp_r,temp,norm=LogNorm(),cmap='rainbow');plt.colorbar().set_label('relative error [au]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('merge_Te_prof_multipulse_interp_crop uncertainty from TS')
	figure_index+=1
	plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	plt.figure();plt.pcolor(temp_t,temp_r,merge_ne_prof_multipulse_interp_crop,cmap='rainbow');plt.colorbar().set_label('ne [10^20 #/m^3]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('ne axially averaged and smoothed to match OES resolution')
	figure_index+=1
	plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
	plt.close()

	temp = merge_dne_prof_multipulse_interp_crop/merge_ne_prof_multipulse_interp_crop
	temp[np.isnan(temp)]=0
	plt.figure();plt.pcolor(temp_t,temp_r,temp,norm=LogNorm(),cmap='rainbow');plt.colorbar().set_label('relative error [au]')#;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]')
	plt.title('merge_ne_prof_multipulse_interp_crop uncertainty from TS')
	figure_index+=1
	plt.savefig(path_where_to_save_everything+mod + '/post_process_'+str(figure_index) + '.eps',bbox_inches='tight')
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


# I will try this:
# Use the excied states 7-12 to fit: ne, Te, nH
# to account for the uncertainty in the higher excited states I can use a decreasing weight for encreasing n
# let's do a linear thing n=7>1, n=8>1.1, ecc.


# example

class calc_output:
	def __init__(self, out):
		self.out = out

def func_recombination(n_list,ne,fraction_nHp,Te,pecfile=pecfile):
	from adas import read_adf15
	import numpy as np

	n_list_int = n_list+16

	def calc(n):
		# print(n,Te.flatten(),ne.flatten(),read_adf15(pecfile, n, Te.flatten(), ne.flatten() * (10 ** -6))[0])
		return n,calc_output(read_adf15(pecfile, n, Te.flatten(), ne.flatten() * (10 ** -6))[0])

	composed_row = map(calc, n_list_int)
	composed_row = set(composed_row)
	composed_row = list(composed_row)
	temp = []
	index=[]
	for row in composed_row:
		temp.append(row[1].out)
		index.append(row[0])
	temp = np.array(temp)
	temp = np.array([peaks for _, peaks in sorted(zip(index, temp))])

	# temp = []
	# for n in n_list:
	# 	temp.append(read_adf15(pecfile, n, Te.flatten(), ne.flatten()*(10**-6))[0])	# ADAS database is in cm^3   # photons s^-1 cm^-3
	# temp = np.array(temp)


	temp = temp.T * (10 ** -6) * (energy_difference[n_list-4] / J_to_eV)  # in W m^-3 / (# / m^3)**2
	# print(temp,ne,fraction_nHp)
	temp = temp.T*(((np.array(ne*(10**20))).flatten()**2) *np.array(fraction_nHp).flatten())
	# print(temp,ne,fraction_nHp)
	temp = temp.reshape((len(n_list),*np.shape(ne))).astype('float')

	return temp

def func_excitation(n_list,ne,nH,Te,pecfile=pecfile):
	from adas import read_adf15
	import numpy as np

	n_list_int = n_list-2

	def calc(n):
		# print(n,Te.flatten(),ne.flatten(),read_adf15(pecfile, n, Te.flatten(), ne.flatten() * (10 ** -6))[0])
		return n,calc_output(read_adf15(pecfile, n, Te.flatten(), ne.flatten() * (10 ** -6))[0])

	composed_row = map(calc, n_list_int)
	composed_row = set(composed_row)
	composed_row = list(composed_row)
	temp = []
	index=[]
	for row in composed_row:
		temp.append(row[1].out)
		index.append(row[0])
	temp = np.array(temp)
	temp = np.array([peaks for _, peaks in sorted(zip(index, temp))])

	# temp = []
	# for n in n_list:
	# 	temp.append(read_adf15(pecfile, n, Te.flatten(), ne.flatten()*(10**-6))[0])	# ADAS database is in cm^3   # photons s^-1 cm^-3
	# temp = np.array(temp)
	# print(temp,ne,fraction_nHp)
	temp = temp.T * (10 ** -6) * (energy_difference[n_list-4] / J_to_eV)  # in W m^-3 / (# / m^3)**2
	# print(temp,ne,fraction_nHp)
	temp = temp.T*np.array(ne*nH*(10**40)).flatten()
	# print(temp,ne,fraction_nHp)
	temp = temp.reshape((len(n_list),*np.shape(ne))).astype('float')
	return temp

def func_total_emission(n_list,ne,fraction_nHp,nH,Te,multiplier):
	import numpy as np
	ne = np.array(ne)
	nH = np.array(nH)
	fraction_nHp = np.array(fraction_nHp)
	Te = np.array(Te)
	multiplier = np.array(multiplier)

	print(ne,fraction_nHp,nH,Te,multiplier)
	# print(func_recombination(n_list,ne, fraction_nHp, Te) + func_excitation(n_list,ne,nH,Te))
	# print(ne,fraction_nHp,nH,Te,multiplier)
	return (1/multiplier)*(func_recombination(n_list,ne, fraction_nHp, Te) + func_excitation(n_list,ne,nH,Te))

my_time=0.5 #ms
my_r=0 #m

my_r_pos = np.abs(r_crop-my_r).argmin()
my_time_pos = np.abs(time_crop-my_time).argmin()

my_emission = inverted_profiles_crop[my_time_pos,:,my_r_pos]

bds=[[1e-6,0.1,1e-7,0.1,0.05],[1e2,1,1e2,20,20]]
guess = [50,0.5,0.1,5,1.2]
min_n = 7
max_n = 12
n_list = np.linspace(min_n, max_n, max_n - min_n + 1).astype('int')
# sigma = 1 +0.1*(n_list-6)
sigma = [2,2.5,1.3,1.5,1.8,2]
fit = curve_fit(func_total_emission,n_list,my_emission[n_list-4],sigma=sigma,p0=guess,bounds=bds,maxfev=1000000)

plt.figure();plt.plot(inverted_profiles_crop[10,n_list-4,16]);plt.pause(0.01)
plt.plot(func_total_emission(n_list,*fit[0]));plt.pause(0.01)


my_time1=0.4 #ms
my_time2=0.45 #ms
my_r1=0 #m
my_r2=0.005 #m


my_r_pos1 = np.abs(r_crop-my_r1).argmin()
my_r_pos2 = np.abs(r_crop-my_r2).argmin()
my_time_pos1 = np.abs(time_crop-my_time1).argmin()
my_time_pos2 = np.abs(time_crop-my_time2).argmin()

min_n = 7
max_n = 12
# n_list = np.linspace(min_n, max_n, max_n - min_n + 1).astype('int')
n_list = np.array([7,9,10,11,12])
n_weights = [0.6,1,1,1,0.6]

inverted_profiles_crop_restrict = inverted_profiles_crop[my_time_pos1:my_time_pos2,:,my_r_pos1:my_r_pos2]

def residual_ext(inverted_profiles_crop_restrict,n_list,n_weights=n_weights):
	# def residual(input):
	def residual(input):
		import numpy as np

		shape = np.shape(inverted_profiles_crop_restrict)

		ne = input[:shape[0]*shape[2]]
		fraction_nHp = input[shape[0]*shape[2]:2*shape[0]*shape[2]]
		nH = input[2*shape[0]*shape[2]:3*shape[0]*shape[2]]
		Te = input[3*shape[0]*shape[2]:4*shape[0]*shape[2]]
		multiplier = input[-1]
		temp1 = func_recombination(n_list,ne,fraction_nHp,Te) + func_excitation(n_list,ne,nH,Te)
		temp1[np.isnan(temp1)]=0

		# print(np.around(input[-1],decimals=3))
		temp2 = []
		for i_n,n in enumerate(n_list):
			# temp2.append(n_weights[i_n]*(temp1[i_n] - multiplier*inverted_profiles_crop_restrict[:,n-4].flatten())/(multiplier*inverted_profiles_crop_restrict[:,n-4].flatten()))
			temp2.append(n_weights[i_n]*(temp1[i_n] - multiplier*inverted_profiles_crop_restrict[:,n-4].flatten()))
		output = np.array(temp2).flatten()
		# print(np.around(output,decimals=3))
		return output
	return residual

guess = [*merge_ne_prof_multipulse_interp_crop_limited[my_time_pos1:my_time_pos2,my_r_pos1:my_r_pos2].flatten(),*np.ones_like(inverted_profiles_crop_restrict[:,0]).flatten(),*1e-5*np.ones_like(inverted_profiles_crop_restrict[:,0]).flatten(),*merge_Te_prof_multipulse_interp_crop_limited[my_time_pos1:my_time_pos2,my_r_pos1:my_r_pos2].flatten(),1.32]
guess = np.array(guess).astype('float').flatten()
bds=[[*1e-4*np.ones_like(inverted_profiles_crop_restrict[:,0]).flatten(),*0.001*np.ones_like(inverted_profiles_crop_restrict[:,0]).flatten(),*np.zeros_like(inverted_profiles_crop_restrict[:,0]).flatten(),*0.2*np.ones_like(inverted_profiles_crop_restrict[:,0]).flatten(),1.3],[*70*np.ones_like(inverted_profiles_crop_restrict[:,0]).flatten(),*np.ones_like(inverted_profiles_crop_restrict[:,0]).flatten(),*100*np.ones_like(inverted_profiles_crop_restrict[:,0]).flatten(),*15*np.ones_like(inverted_profiles_crop_restrict[:,0]).flatten(),1.33]]
sol = least_squares(residual_ext(inverted_profiles_crop_restrict,n_list), guess,bounds = bds, max_nfev = 60,verbose=2,gtol=1e-20,xtol=1e-20,ftol=1e-16,diff_step=0.001,x_scale='jac')
bds=[[*1e-4*np.ones_like(inverted_profiles_crop_restrict[:,0]).flatten(),*0.001*np.ones_like(inverted_profiles_crop_restrict[:,0]).flatten(),*np.zeros_like(inverted_profiles_crop_restrict[:,0]).flatten(),*0.2*np.ones_like(inverted_profiles_crop_restrict[:,0]).flatten(),0.9],[*70*np.ones_like(inverted_profiles_crop_restrict[:,0]).flatten(),*np.ones_like(inverted_profiles_crop_restrict[:,0]).flatten(),*100*np.ones_like(inverted_profiles_crop_restrict[:,0]).flatten(),*15*np.ones_like(inverted_profiles_crop_restrict[:,0]).flatten(),1.7]]
sol = least_squares(residual_ext(inverted_profiles_crop_restrict,n_list), sol.x,bounds = bds, max_nfev = 60,verbose=2,gtol=1e-20,xtol=1e-20,ftol=1e-16,diff_step=0.001,x_scale='jac')
print('Residual: %g' % abs(residual_ext(hx,hy,ht,P_left, P_right,P_top, P_bottom, P_init, conductivity,thickness,emissivity,diffusivity,sigmaSB,T_reference,Power_in,d2x,d2y,dt)(sol)).max())

def func_total_emission2(n_list,input):
	import numpy as np
	num_points = int((len(input)-1)//4)
	ne = np.array(input[:num_points])
	fraction_nHp = np.array(input[num_points:2*num_points])
	nH = np.array(input[2*num_points:3*num_points])
	Te = np.array(input[3*num_points:4*num_points])
	multiplier = np.array(input[-1])

	print(ne,fraction_nHp,nH,Te,multiplier)
	# print(func_recombination(n_list,ne, fraction_nHp, Te) + func_excitation(n_list,ne,nH,Te))
	# print(ne,fraction_nHp,nH,Te,multiplier)
	return (1/multiplier)*(func_recombination(n_list,ne, fraction_nHp, Te) + func_excitation(n_list,ne,nH,Te))


plt.figure();plt.plot(n_list,inverted_profiles_crop_restrict[:,n_list-4,0].T,'-',label='OES');plt.pause(0.01)
# plt.plot(n_list,func_total_emission(n_list,*guess));plt.pause(0.01)
plt.plot(n_list,func_total_emission2(n_list,sol.x),'--',label='estimated')
plt.xlabel('n')
plt.ylabel('emissivity [W m^-3]')
plt.legend(loc='best')
plt.title('ne,fraction_nHp,nH,Te,multiplier = '+str(sol.x)+'\n guess='+str(guess))
plt.pause(0.01)
