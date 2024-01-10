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
from scipy.optimize import curve_fit,least_squares
from scipy import interpolate
from scipy.signal import find_peaks, peak_prominences as get_proms
import time as tm
from multiprocessing import Pool,cpu_count
import datetime as dt
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

def bin2dec_for_list(array):
	import numpy as np
	temp = []
	for value in array:
		temp.append(int(value,2))
	return np.array(temp)


def DOM52sec_for_list(array):
	import numpy as np
	date_time_stamp2sec = []
	for value in array:
		if np.isfinite(value)==True:
			date_time_stamp2sec.append( DOM52sec(value) )
	return np.array(date_time_stamp2sec)

def DOM52sec_for_list(array):
	import numpy as np
	import datetime as dt
	date_time_stamp2sec = []
	for value in array:
		if np.isfinite(value)==True:
			# date_time_stamp2sec.append( dt.datetime.fromtimestamp(int(DOM52sec(value))) )
			date_time_stamp2sec.append( dt.datetime.fromtimestamp(DOM52sec(value)) )
	return np.array(date_time_stamp2sec)


def DOM52sec(value):
	if np.isfinite(value)==True:
		out = int("{0:b}".format(int(value))[:-32],2)+int("{0:b}".format(int(value))[-32:],2)/(2**32)
		return out
	else:
		return np.nan

file_full_path = '/home/ffederic/work/Collaboratory/test/experimental_data/2019-07-03/20190703_1.csv'
multi_1 = pd.read_csv(file_full_path,index_col=False,sep='|')
index1 = list(multi_1.head(0))
for index in range(len(index1)//3):
	index1[index*3]=index1[index*3+1]
	index1[index*3+2]=index1[index*3+1]
multi_1 = pd.read_csv(file_full_path,index_col=False,header=1,sep='|')
index2 = list(multi_1.head(0))
index = [_+' '+__ for _,__ in zip(index1,index2)]
for i_text,text in enumerate(index):
	if (text.find('.'))!=-1:
		index[i_text] = text[:text.find('.')]
multi_1 = pd.read_csv(file_full_path,index_col=False,header=2,sep='|')
multi_1_array = np.array(multi_1).T

date_time_stamp = multi_1[multi_1.keys()[6]]
# date_time_stamp_separated = []
# for value in date_time_stamp:
# 	if np.isfinite(value)==True:
# 		date_time_stamp_separated.append( ["{0:b}".format(int(value))[:-32],"{0:b}".format(int(value))[-32:]] )
# date_time_stamp_separated = np.array(date_time_stamp_separated)
date_time_stamp2sec = DOM52sec_for_list(date_time_stamp)

# temp = "{0:b}".format(int(timestamp))
# while len(temp)<64:
# 	temp = '0'+temp
# datetime = dt.datetime.fromtimestamp(int(temp[:32],2))
# time = int(temp[:32],2)/2**32

if False:	# I don't think this code is usefull anymore
	merge_ID_target=85
	all_j=find_index_of_file(merge_ID_target,df_settings,df_log,only_OES=True)
	current_trace = df_log.loc[all_j[1],['current_trace_file']]
	timestamp2 = int(current_trace.values[0][current_trace.values[0].find('_')+1:])

	row = np.abs(date_time_stamp2sec - DOM52sec(timestamp2)).argmin()
	# multi_1[multi_1.keys()[1]][row-5:row+5]

	if multi_1[multi_1.keys()[8]][row]==True:
		end_row = row
	else:
		add=0
		while multi_1[multi_1.keys()[8]][row+add]!=True:
			add+=1
		end_row = row + add

	add=1
	while multi_1[multi_1.keys()[8]][row-add]!=True:
		add+=1
	beginning_row = row - add

	beginning_sec = date_time_stamp2sec[beginning_row]
	end_sec = date_time_stamp2sec[end_row]


	date_time_stamp = multi_1[multi_1.keys()[24]]
	date_time_stamp2sec = DOM52sec_for_list(date_time_stamp)
	select = np.logical_and(date_time_stamp2sec>=beginning_sec-100,date_time_stamp2sec<=end_sec+400)
	date_time_stamp2time = [dt.datetime.fromtimestamp(int(_)).time() for _ in date_time_stamp2sec//1 ]

	plt.figure()
	plt.plot(np.array(date_time_stamp2time)[select],multi_1[multi_1.keys()[24+2]][select])
	plt.pause(0.1)

	date_time_stamp = multi_1[multi_1.keys()[18]]
	date_time_stamp2sec = DOM52sec_for_list(date_time_stamp)
	date_time_stamp2time = [dt.datetime.fromtimestamp(int(_)).time() for _ in date_time_stamp2sec//1 ]
	plt.figure()
	plt.plot(np.array(date_time_stamp2time)[np.isfinite(multi_1[multi_1.keys()[20]])],multi_1[multi_1.keys()[20]][np.isfinite(multi_1[multi_1.keys()[20]])])
	plt.pause(0.1)


	date_time_stamp = multi_1[multi_1.keys()[24]]
	date_time_stamp2sec = DOM52sec_for_list(date_time_stamp)
	date_time_stamp2time = [dt.datetime.fromtimestamp(int(_)).time() for _ in date_time_stamp2sec//1 ]
	plt.figure()
	plt.plot(np.array(date_time_stamp2time)[np.isfinite(multi_1[multi_1.keys()[26]])],multi_1[multi_1.keys()[26]][np.isfinite(multi_1[multi_1.keys()[26]])])
	plt.pause(0.1)
else:
	pass

plt.close('all')
for i_index in range(0,13):
	print(index[i_index*3+2])
	plt.figure()
	plt.title(index[i_index*3+2])
	date_time_stamp = multi_1[multi_1.keys()[i_index*3]]
	date_time_stamp2time = DOM52sec_for_list(date_time_stamp)
	# date_time_stamp2time = [dt.datetime.fromtimestamp(int(_)) for _ in date_time_stamp2sec//1 ]
	plt.plot(date_time_stamp2time,multi_1[multi_1.keys()[i_index*3+2]][:len(date_time_stamp2time)],label=index[i_index*3+2])
	plt.legend(loc='best')
	plt.grid()
plt.pause(0.1)

if False:
	# I want to find the time constant of the target. I can get it looking at its cooling from high temp
	# whith i_index=8
	temp = np.array(multi_1[multi_1.keys()[i_index*3]])
	x=np.array(multi_1[multi_1.keys()[i_index*3+1]])[np.isfinite(temp)][-138:-100]
	y=np.array(multi_1[multi_1.keys()[i_index*3+2]])[np.isfinite(temp)][-138:-100]
	y = y-y.min()
	time = np.arange(len(y))*2.

	def temp(t,tau,amp):
		t0=0
		print(str([t0,tau,amp]))
		out = np.zeros_like(t)
		out[t>=t0] = amp*(np.exp(-((t[t>=t0]-t0)/(tau))**1))
		return out

	guess=curve_fit(temp, time,y, p0=[10,1],bounds=[[0,0],[np.inf,np.inf]],verbose=2,maxfev=1e5)

	plt.figure()
	plt.plot(time,y)
	plt.plot(time,temp(time,*guess[0]),'+')




plt.close('all')
plt.figure()
# plt.title(index[i_index*3+2])
for i_index in range(0,7):
# for i_index in [4,11,0,1,2]:
	date_time_stamp = multi_1_array[i_index*3].astype(np.float32)
	temp = multi_1_array[i_index*3+2][np.isfinite(date_time_stamp)].astype(np.int)
	date_time_stamp2time = DOM52sec_for_list(multi_1_array[i_index*3][np.isfinite(date_time_stamp)].astype(np.int))
	# date_time_stamp2time = [dt.datetime.fromtimestamp(int(_)) for _ in date_time_stamp2sec//1 ]
	plt.plot(date_time_stamp2time,temp/np.max(temp),label=index[i_index*3+2])
	plt.legend(loc='best')
	plt.grid()
plt.pause(0.1)


plt.close('all')
plt.figure()
# plt.title(index[i_index*3+2])
# for i_index in range(0,13):
# for i_index in [4,11]:
i_index = 4
date_time_stamp = multi_1[multi_1.keys()[i_index*3]]
date_time_stamp2time = DOM52sec_for_list(date_time_stamp)
# date_time_stamp2time = [dt.datetime.fromtimestamp(int(_)) for _ in date_time_stamp2sec//1 ]
plt.plot(date_time_stamp2time,multi_1[multi_1.keys()[i_index*3+2]][:len(date_time_stamp2time)]/np.max(multi_1[multi_1.keys()[i_index*3+2]][:len(date_time_stamp2time)]),label=index[i_index*3+2])
i_index = 11
date_time_stamp = multi_1[multi_1.keys()[i_index*3]]
date_time_stamp2time = DOM52sec_for_list(date_time_stamp)
# date_time_stamp2time = [dt.datetime.fromtimestamp(int(_)) for _ in date_time_stamp2sec//1 ]
plt.errorbar(date_time_stamp2time,multi_1[multi_1.keys()[i_index*3+2]][:len(date_time_stamp2time)]/np.max(multi_1[multi_1.keys()[i_index*3+2]][:len(date_time_stamp2time)]),yerr=multi_1[multi_1.keys()[i_index*3+2+3]][:len(date_time_stamp2time)]/np.max(multi_1[multi_1.keys()[i_index*3+2]][:len(date_time_stamp2time)]),label=index[i_index*3+2])
plt.legend(loc='best')
plt.grid()
plt.pause(0.1)



multi_3 = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/2019-07-02/SkimmerData_2019-07-02.csv',index_col=False,sep=',')
index1 = list(multi_3.head(0))
for index in range(1,(len(index1)+1)//2):
	index1[index*2-1]=index1[index*2]
multi_3 = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/2019-07-02/SkimmerData_2019-07-02.csv',index_col=False,header=1,sep=',')
index2 = list(multi_3.head(0))
index = [_+' '+__ for _,__ in zip(index1,index2)]
for i_text,text in enumerate(index):
	if (text.find('.'))!=-1:
		index[i_text] = text[:text.find('.')]
multi_3 = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/2019-07-02/SkimmerData_2019-07-02.csv',index_col=False,header=2,sep=',')
multi_3_array = np.array(multi_3)

for i_index in range(11):
	print(index[i_index*2+1+1])
	date_time_stamp = multi_3[multi_3.keys()[i_index*2+1]]
	date_time_stamp2time = DOM52sec_for_list(date_time_stamp)
	# date_time_stamp2time = [dt.datetime.fromtimestamp(int(_)) for _ in date_time_stamp2sec//1 ]
	plt.figure()
	plt.title(index[i_index*2+1+1])
	print(index[i_index*2+1+1])
	plt.plot(date_time_stamp2time,multi_3[multi_3.keys()[i_index*2+1+1]][:len(date_time_stamp2time)])
	plt.pause(0.1)


if False:
	# I want to find the time constant of the target skimmer. I can get it looking at its cooling from high temp
	# whith i_index=1
	i_index = 1
	x=np.array(date_time_stamp2time)[51980:][:250]
	y=np.array(multi_3[multi_3.keys()[i_index*2+1+1]])[51980:][:250]
	y = y-y.min()
	time = np.arange(len(y))*1.

	def temp(t,tau,amp):
		t0=0
		print(str([t0,tau,amp]))
		out = np.zeros_like(t)
		out[t>=t0] = amp*(np.exp(-((t[t>=t0]-t0)/(tau))**1))
		return out

	guess=curve_fit(temp, time,y, p0=[10,1],bounds=[[0,0],[np.inf,np.inf]],verbose=2,maxfev=1e5)

	plt.figure()
	plt.plot(time,y)
	plt.plot(time,temp(time,*guess[0]),'+')


multi_3 = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/from_Tijs_target_skimmer_temp/new/flow-and-temp-cathode-3.csv',index_col=False,sep='|')
index1 = list(multi_3.head(0))
for index in range(1,(len(index1)+1)//3):
	index1[index*3-1]=index1[index*3]
multi_3 = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/from_Tijs_target_skimmer_temp/new/flow-and-temp-cathode-3.csv',index_col=False,header=1,sep='|')
index2 = list(multi_3.head(0))
index = [_+' '+__ for _,__ in zip(index1,index2)]
for i_text,text in enumerate(index):
	if (text.find('.'))!=-1:
		index[i_text] = text[:text.find('.')]
multi_3 = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/from_Tijs_target_skimmer_temp/new/flow-and-temp-cathode-3.csv',index_col=False,header=2,sep='|')
multi_3_array = np.array(multi_3)

for i_index in range(11):
	print(index[i_index*3+1])
	# time = []
	# for value in multi_3[multi_3.keys()[i_index*2+1]]:
	# 	try:
	# 		time.append(datetime.strptime(value, '%Y-%m-%d %H:%M:%S.%f'))
	# 	except:
	# 		time.append(datetime.strptime(value, '%Y-%m-%d %H:%M:%S'))
	# date_time_stamp2time = np.array(time)
	date_time_stamp = multi_3[multi_3.keys()[i_index*3]]
	date_time_stamp2time = DOM52sec_for_list(date_time_stamp)
	# date_time_stamp2time = [dt.datetime.fromtimestamp(int(_)) for _ in date_time_stamp2sec//1 ]
	plt.figure()
	plt.title(index[i_index*3+1])
	plt.plot(date_time_stamp2time,multi_3[multi_3.keys()[i_index*3+1+1]][:len(date_time_stamp2time)])
	plt.pause(0.1)

if False:
	# I want to find the time constant of the target skimmer. I can get it looking at its cooling from high temp
	# whith i_index=1
	i_index = 0
	x=np.array(date_time_stamp2time)[11355:][:120]
	y=np.array(multi_3[multi_3.keys()[i_index*3+1+1]])[11355:][:120]
	y = y-y.min()
	time = np.arange(len(y))*2.

	def temp(t,tau,amp):
		t0=0
		print(str([t0,tau,amp]))
		out = np.zeros_like(t)
		out[t>=t0] = amp*(np.exp(-((t[t>=t0]-t0)/(tau))**1))
		return out

	guess=curve_fit(temp, time,y, p0=[10,1],bounds=[[0,0],[np.inf,np.inf]],verbose=2,maxfev=1e5)

	plt.figure()
	plt.plot(time,y)
	plt.plot(time,temp(time,*guess[0]),'+')
if False:
	# I want to find the time constant of the anode. I can get it looking at its cooling from high temp
	# whith i_index=0
	i_index = 0
	x=np.array(date_time_stamp2time)[5995:][:45]
	y=np.array(multi_3[multi_3.keys()[i_index*3+1+1]])[5995:][:45]
	y = y-y.min()
	time = np.arange(len(y))*2.

	def temp(t,tau,amp):
		t0=0
		print(str([t0,tau,amp]))
		out = np.zeros_like(t)
		out[t>=t0] = amp*(np.exp(-((t[t>=t0]-t0)/(tau))**1))
		return out

	guess=curve_fit(temp, time,y, p0=[10,1],bounds=[[0,0],[np.inf,np.inf]],verbose=2,maxfev=1e5)

	plt.figure()
	plt.plot(time,y)
	plt.plot(time,temp(time,*guess[0]),'+')
if False:
	# I want to find the time constant of the cathode. I can get it looking at its cooling from high temp
	# whith i_index=0
	i_index = 0
	x=np.array(date_time_stamp2time)[5995:][:45]
	y=np.array(multi_3[multi_3.keys()[i_index*3+1+1]])[5995:][:45]
	y = y-y.min()
	time = np.arange(len(y))*2.

	def temp(t,tau,amp):
		t0=0
		print(str([t0,tau,amp]))
		out = np.zeros_like(t)
		out[t>=t0] = amp*(np.exp(-((t[t>=t0]-t0)/(tau))**1))
		return out

	guess=curve_fit(temp, time,y, p0=[10,1],bounds=[[0,0],[np.inf,np.inf]],verbose=2,maxfev=1e5)

	plt.figure()
	plt.plot(time,y)
	plt.plot(time,temp(time,*guess[0]),'+')



# from the ENERGY 2D forward model (see forward model Energy 2D excel file)
SS_heat_flux = lambda x : np.polyval([141184,-3e6],x)	# in degree C
ELM_heat_flux = lambda x : np.polyval([2e+6,5e+7],x)	# in degree C

def temp(t,t0,tau,amp):
	print(str([t0,tau,amp]))
	out = np.zeros_like(t)
	out[t>t0] = amp*(np.exp(-((t[t>t0]-t0)/(tau))**1))
	return out

time = np.linspace(0,58,num=len(multi_3[multi_3.keys()[0]][5846-20:5846+10]))
plt.figure()
plt.plot(multi_3[multi_3.keys()[0]][5846-20:5846+10]-multi_3[multi_3.keys()[0]][5846-20:5846+10].min(),multi_3[multi_3.keys()[2]][5846-20:5846+10])
plt.plot(multi_3[multi_3.keys()[1]][5846-20:5846+10],multi_3[multi_3.keys()[2]][5846-20:5846+10])
x=np.array(multi_3[multi_3.keys()[2]][5846-20:5846+10])-np.array(multi_3[multi_3.keys()[2]][5846-20:5846+10]).min()
guess=curve_fit(temp, time,x,sigma=1/(x+1), p0=[25,10,1],bounds=[[0,0,0],[time.max(),np.inf,np.inf]],verbose=2,maxfev=1e5)

plt.figure()
plt.plot(time,x)
plt.plot(time,temp(time,*guess[0]),'+')


# import IR camera data of the pressure scan
print_plots = False

record = []
filenames = ['Capture003_restrict','Capture004','Capture005','Capture006','Capture007','Capture008','Capture009','Capture010','Capture048','Capture049']
for filename in filenames:
	IR_read = pd.read_excel (r'/home/ffederic/work/Collaboratory/test/experimental_data/2019-07-03/01/IR_camera/'+filename+'.xlsx', sheet_name='Time evo',header=13)
	keys = IR_read.keys()
	peaks = find_peaks(IR_read['target Max  (°C)'],distance=300)[0]
	proms = get_proms(IR_read['target Max  (°C)'],peaks)[0]
	peaks = peaks[proms>100]
	temp_peaks = IR_read['target Max  (°C)'][peaks]
	temp=[]
	for index in range(10):
		temp.append(np.array(IR_read['target Mean  (°C)'])[peaks-index-5])
	mean_temp_before_peaks = np.mean(temp,axis=-0)
	temp=[]
	for index in range(10):
		temp.append(np.array(IR_read['target small Mean  (°C)'])[peaks-index-5])
	max_temp_before_peaks = np.mean(temp,axis=-0)
	record.append( [np.median(temp_peaks-mean_temp_before_peaks),np.mean(np.sort(mean_temp_before_peaks)[-50:]),np.median(temp_peaks-max_temp_before_peaks),np.mean(np.sort(max_temp_before_peaks)[-50:])] )
	if print_plots:
		plt.figure()
		plt.title('file '+filename)
		for key in keys[1:]:
			if key == 'point Level  (°C)':
				continue
			plt.plot((IR_read['(frames)']-1)*278*1e-6,IR_read[key],label=key)
		plt.plot(((IR_read['(frames)']-1)*278*1e-6)[[0,len(IR_read['(frames)'])-1]],np.median(IR_read['side Mean  (°C)'])*np.ones((2)),'--',label='side Mean  (°C) median')
		plt.plot(((IR_read['(frames)']-1)*278*1e-6)[[0,len(IR_read['(frames)'])-1]],np.median(IR_read['target Mean  (°C)'])*np.ones((2)),'--',label='target Mean  (°C) median')
		plt.plot(((IR_read['(frames)']-1)*278*1e-6)[[0,len(IR_read['(frames)'])-1]],np.max(IR_read['target Max  (°C)'])*np.ones((2)),'--',label='target Max  (°C) max')
		plt.plot(((IR_read['(frames)']-1)*278*1e-6)[[0,len(IR_read['(frames)'])-1]],np.median(temp_peaks)*np.ones((2)),'--',label='target Max  (°C) peaks median\ndt = '+str(np.median(temp_peaks-max_temp_before_peaks)))
		plt.xlabel('time [s]')
		plt.ylabel('temperature [°C]')
		plt.legend(loc='best')
		plt.pause(0.01)
record = np.array(record)
x_coord = [15.04,15.04,11.847,11.847,8.17,4.37,4.37,0.5158,0.296,0.296]
plt.figure()
# plt.plot(x_coord,record.T[0],'o',label='average temp increase after puslse')
# plt.plot(x_coord,record.T[1],'o',label='max temp before puslse')
plt.plot(x_coord,record.T[2],'x',label='average temp increase after puslse all max')
plt.plot(x_coord,record.T[3],'x',label='max temp before puslse all max')
plt.legend(loc='best')
plt.xlabel('Pressure [Pa]')
plt.ylabel('temperature [°C]')
plt.grid()
plt.pause(0.01)
# plt.figure()
# plt.plot([15.04,11.847,8.17,4.37,0.5158,0.296],SS_heat_flux(record.T[3]),label='SS heat flux')
# plt.plot([15.04,11.847,8.17,4.37,0.5158,0.296],ELM_heat_flux(record.T[2]),label='heat flux due to ELM')
# plt.legend(loc='best')
# plt.semilogy()
# plt.xlabel('Pressure [Pa]')
# plt.ylabel('maximum target heat flux [W m^-2]')
# plt.pause(0.01)

# import IR camera data of the distance scan
print_plots = False

record = []
filenames = ['Capture014','Capture015','Capture013','Capture012','Capture003_restrict','Capture004','Capture016','Capture017','Capture019','Capture018','Capture020','Capture021']
for filename in filenames:
	IR_read = pd.read_excel (r'/home/ffederic/work/Collaboratory/test/experimental_data/2019-07-03/01/IR_camera/'+filename+'.xlsx', sheet_name='Time evo',header=13)
	keys = IR_read.keys()
	peaks = find_peaks(IR_read['target Max  (°C)'],distance=300)[0]
	proms = get_proms(IR_read['target Max  (°C)'],peaks)[0]
	peaks = peaks[proms>100]
	temp_peaks = IR_read['target Max  (°C)'][peaks]
	temp=[]
	for index in range(10):
		temp.append(np.array(IR_read['target Mean  (°C)'])[peaks-index-5])
	mean_temp_before_peaks = np.mean(temp,axis=-0)
	temp=[]
	for index in range(10):
		temp.append(np.array(IR_read['target Max  (°C)'])[peaks-index-5])
	max_temp_before_peaks = np.mean(temp,axis=-0)
	record.append( [np.median(temp_peaks-mean_temp_before_peaks),np.mean(np.sort(mean_temp_before_peaks)[-50:]),np.median(temp_peaks-max_temp_before_peaks),np.mean(np.sort(max_temp_before_peaks)[-50:])] )
	for key in keys[1:]:
		if key == 'point Level  (°C)':
			continue
	if print_plots:
		plt.figure()
		plt.title('file '+filename)
		plt.plot((IR_read['(frames)']-1)*278*1e-6,IR_read[key],label=key)
		plt.plot(((IR_read['(frames)']-1)*278*1e-6)[[0,len(IR_read['(frames)'])-1]],np.median(IR_read['side Mean  (°C)'])*np.ones((2)),'--',label='side Mean  (°C) median')
		plt.plot(((IR_read['(frames)']-1)*278*1e-6)[[0,len(IR_read['(frames)'])-1]],np.median(IR_read['target Mean  (°C)'])*np.ones((2)),'--',label='target Mean  (°C) median')
		plt.plot(((IR_read['(frames)']-1)*278*1e-6)[[0,len(IR_read['(frames)'])-1]],np.max(IR_read['target Max  (°C)'])*np.ones((2)),'--',label='target Max  (°C) max')
		plt.plot(((IR_read['(frames)']-1)*278*1e-6)[[0,len(IR_read['(frames)'])-1]],np.median(temp_peaks)*np.ones((2)),'--',label='target Max  (°C) peaks median\ndt = '+str(np.median(temp_peaks-temp_before_peaks)))
		plt.xlabel('time [s]')
		plt.ylabel('temperature [°C]')
		plt.legend(loc='best')
		plt.pause(0.01)
record = np.array(record)
x_coord = [1.2,1.2,6.2,6.2,21.2,21.2,36.2,36.2,56.2,56.2,96.2,96.2]
plt.figure()
# plt.plot([1.2,6.2,21.2,36.2,56.2,96.2],record.T[0],label='average temp increase after puslse')
# plt.plot([1.2,6.2,21.2,36.2,56.2,96.2],record.T[1],label='max temp before puslse')
plt.plot(x_coord,record.T[2],'x',label='average temp increase after puslse all max')
plt.plot(x_coord,record.T[3],'x',label='max temp before puslse all max')
plt.plot([21.2,21.2],[np.max(record.T[2:]),np.min(record.T[2:])],'--',label='smallest distance with OES data')
plt.legend(loc='best')
plt.xlabel('distance [nn]')
plt.ylabel('temperature [°C]')
plt.grid()
plt.pause(0.01)
# plt.figure()
# plt.plot([1.2,6.2,21.2,36.2,56.2,96.2],SS_heat_flux(record.T[3]),label='SS heat flux')
# plt.plot([1.2,6.2,21.2,36.2,56.2,96.2],ELM_heat_flux(record.T[3]),label='heat flux due to ELM')
# plt.legend(loc='best')
# plt.semilogy()
# plt.xlabel('distance [nn]')
# plt.ylabel('maximum target heat flux [W m^-2]')
# plt.pause(0.01)
