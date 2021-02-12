import numpy as np
import xarray as xr
import math
import glob
import re
import matplotlib.pyplot as plt
import matplotlib.tri as ptri
import matplotlib.colors as colors
import matplotlib.artist as art
from operator import itemgetter, attrgetter, methodcaller
import readline
from tabulate import tabulate
import scipy.interpolate as interpolate
import matplotlib.pyplot  as plt
import os
import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def read_first(filename):
    data = []
    count=0
    f = open(filename,'r')
    for line in f:
        if 'species' in line:
            columns=line.split()
            list=columns[2].split(",")
            del(list[-1]) # remove last blank partition
        elif '<body>' in line:
            break

    for line in f:
        if line == '':
            break
        count = count + 1
        columns=line.split()
        data.append(float(columns[1]))
        data.append(float(columns[2]))
    f.close()
    data = np.reshape(data, (count,2)).T
    dat = xr.DataArray(data, coords=[['r','z'],range(0,count)], dims=['axis','index'])

    return list, count, dat

def read_data(file_array, list, count):
    data = []
    r_cup = []
    z_cup = []

    data_list=['n','t','vx','vy','vz']
    for file in file_array:
        f = open(file,'r')
        for line in f:
            if '<body>' in line:
                break

        for line in f:
            if line =='':
                break
            columns = line.split()
            i = 3
            for item in list:
                for n in range(i,i+5):
                    data.append(float(columns[n]))
                i=i+5
        f.close()
    data=np.reshape(data,(len(data_list),len(list),count,len(file_array)),order='F')

    dat = xr.DataArray(data,coords=[data_list, list, range(0,count), file_array],dims=['quantity','species','index','filename'])
    return dat

def read_triangle(nodes,elements):
    r,z,tri = [],[],[]
    f = open(nodes,'r')
    f.readline()
    for line in f:
        if '#' in line:
            break
        columns = line.split()
        r.append(float(columns[1]))
        z.append(float(columns[2]))
    f.close()
    f = open(elements,'r')
    f.readline()
    for line in f:
        if '#' in line:
            break
        columns = line.split()
        tri.append([columns[1],columns[2],columns[3]])
    f.close()
    return r,z,tri

Data = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/B2.5-Eunomia/'
file_array = ['027.dat','053.dat','10.dat','20.dat','44.dat']
pressure = [0.27,0.53,1.0,2.0,4.4]
boltzmann_constant_J = 1.380649e-23	# J/K
eV_to_K = 8.617333262145e-5	# eV/K
hydrogen_mass = 1.008*1.660*1e-27	# kg
print('Reading data file..')

figure_index=0
plt.close('all')


'''Data read only 1 time'''
data_tria = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/B2.5-Eunomia/'
list, grid_cells, data_grid = read_first(Data+file_array[0])
data_matrix = read_data([Data+file_array[0]], list, grid_cells)
z_data = data_grid.sel(axis='z').values
r_data = data_grid.sel(axis='r').values
r_tri, z_tri, tri = read_triangle(data_tria+'tria.1.node',data_tria+'tria.1.ele')
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise']
figure_index +=1
figure_1 = figure_index
plt.figure(figure_1)
figure_index +=1
figure_2 = figure_index
plt.figure(figure_2)
figure_index +=1
figure_3 = figure_index
plt.figure(figure_3)
figure_index +=1
figure_4 = figure_index
plt.figure(figure_4)
figure_index +=1
figure_5 = figure_index
plt.figure(figure_5)
figure_index +=1
figure_6 = figure_index
plt.figure(figure_6)
figure_index +=1
figure_7 = figure_index
plt.figure(figure_7)
figure_index +=1
figure_8 = figure_index
plt.figure(figure_8)
figure_index +=1
figure_9 = figure_index
plt.figure(figure_9)
figure_index +=1
figure_10 = figure_index
plt.figure(figure_10)
figure_index +=1
figure_11 = figure_index
plt.figure(figure_11)
figure_index +=1
figure_12 = figure_index
plt.figure(figure_12)
figure_index +=1
figure_13 = figure_index
plt.figure(figure_13)
figure_index +=1
figure_14 = figure_index
plt.figure(figure_14)
figure_index +=1
figure_15 = figure_index
plt.figure(figure_15)
figure_index +=1
figure_16 = figure_index
plt.figure(figure_16)
figure_index +=1
figure_17 = figure_index
plt.figure(figure_17)
figure_index +=1
figure_18 = figure_index
plt.figure(figure_18)
collect_x = []
collect_y = []
collect_x1 = []
collect_y1 = []
collect_temp_e = []
collect_dens_e = []
collect_temp_H2 = []
collect_dens_H2 = []
collect_temp_H = []
collect_dens_H = []
for index in range(len(file_array)):
	r_tri = np.asarray(r_tri)
	r_tri2 = -r_tri
	data_matrix = read_data([Data+file_array[index]], list, grid_cells)
	temp_e = data_matrix.sel(quantity='t', species='e', filename=Data+file_array[index]).values
	dens_e = data_matrix.sel(quantity='n', species='e', filename=Data+file_array[index]).values
	temp_H2 = (data_matrix.sel(quantity='t', species='H_2', filename=Data+file_array[index]).values)[0]
	dens_H2 = (data_matrix.sel(quantity='n', species='H_2', filename=Data+file_array[index]).values)[0]
	temp_H = (data_matrix.sel(quantity='t', species='H', filename=Data+file_array[index]).values)
	dens_H = (data_matrix.sel(quantity='n', species='H', filename=Data+file_array[index]).values)
	temp_Hp = (data_matrix.sel(quantity='t', species='H^+', filename=Data+file_array[index]).values)
	dens_Hp = (data_matrix.sel(quantity='n', species='H^+', filename=Data+file_array[index]).values)
	v_e = (data_matrix.sel(quantity='vz', species='e', filename=Data+file_array[index]).values)
	vz_Hp = (data_matrix.sel(quantity='vz', species='H^+', filename=Data+file_array[index]).values)
	vz_H = (data_matrix.sel(quantity='vz', species='H', filename=Data+file_array[index]).values)
	vz_H2 = (data_matrix.sel(quantity='vz', species='H_2', filename=Data+file_array[index]).values)[0]
	v_H = ((data_matrix.sel(quantity='vz', species='H', filename=Data+file_array[index]).values)**2 + (data_matrix.sel(quantity='vy', species='H', filename=Data+file_array[index]).values)**2 + (data_matrix.sel(quantity='vx', species='H', filename=Data+file_array[index]).values)**2)**0.5
	v_H2 = ((data_matrix.sel(quantity='vz', species='H_2', filename=Data+file_array[index]).values)[0]**2 + (data_matrix.sel(quantity='vy', species='H_2', filename=Data+file_array[index]).values)[0]**2 + (data_matrix.sel(quantity='vx', species='H_2', filename=Data+file_array[index]).values)[0]**2)**0.5
	ratio = dens_H2/dens_e
	select = np.logical_and(r_data>0,np.logical_and(r_data<0.02,np.logical_and(z_data>-0.35,np.logical_and(np.isfinite(ratio),ratio<1e5))))
	collect_x.append(temp_e[select])
	collect_y.append(ratio[select])
	collect_temp_e.append(temp_e[select])
	collect_dens_e.append(dens_e[select])
	collect_temp_H2.append(temp_H2[select])
	collect_dens_H2.append(dens_H2[select])
	collect_temp_H.append(temp_H[select])
	collect_dens_H.append(dens_H[select])
	plt.figure(figure_1)
	plt.plot(temp_e[select],ratio[select],'x',color=color[index],label='pressure %.3gPa' %(pressure[index]))
	plt.plot(temp_e,ratio,',',color=color[index])
	fit = np.polyfit(np.log(temp_e[select]),np.log(ratio[select]),2)
	plt.plot(temp_e[select],np.exp(np.polyval(fit,np.log(temp_e[select]))),'--',color=color[index])
	# max_nH2_ne_from_pressure = pressure[index]/(boltzmann_constant_J*temp_e[select]/eV_to_K)/dens_e[select]
	# plt.plot(temp_e[select],max_nH2_ne_from_pressure,'v',color=color[index])
	# plt.plot(temp_e[select],10*np.exp(np.polyval(fit,np.log(temp_e[select]))),'k--')
	# plt.plot(temp_e[select],0.1*np.exp(np.polyval(fit,np.log(temp_e[select]))),'k--')
	plt.figure(figure_2)
	ratio = (dens_H)/dens_e
	select = np.logical_and(r_data>0,np.logical_and(r_data<0.02,np.logical_and(z_data>-0.35,np.logical_and(np.isfinite(ratio),np.logical_and(True,True)))))
	collect_x1.append(temp_e[select])
	collect_y1.append(ratio[select])
	plt.plot((temp_e)[select],ratio[select],'x',color=color[index],label='pressure %.3gPa' %(pressure[index]))
	plt.plot(temp_e,ratio,',',color=color[index])
	plt.figure(figure_3)
	ratio = temp_H
	select = np.logical_and(r_data>0,np.logical_and(r_data<0.02,np.logical_and(z_data>-0.35,np.logical_and(np.isfinite(ratio),True))))
	plt.plot((temp_e)[select],ratio[select],'x',color=color[index],label='pressure %.3gPa' %(pressure[index]))
	plt.plot(temp_e,ratio,',',color=color[index])
	plt.figure(figure_4)
	ratio = temp_H2
	select = np.logical_and(r_data>0,np.logical_and(r_data<0.02,np.logical_and(z_data>-0.35,np.logical_and(np.isfinite(ratio),True))))
	plt.plot((temp_e)[select],ratio[select],'x',color=color[index],label='pressure %.3gPa' %(pressure[index]))
	plt.plot(temp_e,ratio,',',color=color[index])
	plt.figure(figure_5)
	ratio = temp_e
	select = np.logical_and(r_data>0,np.logical_and(r_data<0.001,np.logical_and(z_data>-0.35,np.logical_and(np.isfinite(ratio),True))))
	plt.plot((z_data)[select],ratio[select],'x',color=color[index],label='pressure %.3gPa' %(pressure[index]))
	plt.plot(z_data,ratio,',',color=color[index])
	plt.figure(figure_6)
	ratio = dens_e
	select = np.logical_and(r_data>0,np.logical_and(r_data<0.001,np.logical_and(z_data>-0.35,np.logical_and(np.isfinite(ratio),True))))
	plt.plot((z_data)[select],ratio[select],'x',color=color[index],label='pressure %.3gPa' %(pressure[index]))
	plt.plot(z_data,ratio,',',color=color[index])
	plt.figure(figure_7)
	ratio = temp_Hp
	select = np.logical_and(r_data>0,np.logical_and(r_data<0.02,np.logical_and(z_data>-0.35,np.logical_and(np.isfinite(ratio),True))))
	plt.plot((temp_e)[select],ratio[select],'x',color=color[index],label='pressure %.3gPa' %(pressure[index]))
	plt.plot(temp_e,ratio,',',color=color[index])
	plt.figure(figure_8)
	ratio = v_e
	select = np.logical_and(r_data>0,np.logical_and(r_data<0.001,np.logical_and(z_data>-0.35,np.logical_and(np.isfinite(ratio),True))))
	plt.plot((z_data)[select],ratio[select],'x',color=color[index],label='pressure %.3gPa' %(pressure[index]))
	plt.plot(z_data,ratio,',',color=color[index])
	plt.figure(figure_9)
	ratio = vz_Hp
	select = np.logical_and(r_data>0,np.logical_and(r_data<0.001,np.logical_and(z_data>-0.35,np.logical_and(np.isfinite(ratio),True))))
	plt.plot((z_data)[select],ratio[select],'x',color=color[index],label='pressure %.3gPa' %(pressure[index]))
	plt.plot(z_data,ratio,',',color=color[index])
	plt.figure(figure_10)
	ratio = (vz_Hp**2)*dens_Hp*hydrogen_mass + dens_Hp*temp_Hp/eV_to_K*boltzmann_constant_J + dens_e*temp_e/eV_to_K*boltzmann_constant_J
	select = np.logical_and(r_data>0,np.logical_and(r_data<0.001,np.logical_and(z_data>-0.35,np.logical_and(np.isfinite(ratio),True))))
	plt.plot((z_data)[select],ratio[select],'x',color=color[index],label='pressure %.3gPa' %(pressure[index]))
	plt.plot(z_data,ratio,',',color=color[index])
	plt.figure(figure_11)
	ratio = (vz_Hp**2)*dens_Hp*hydrogen_mass
	select = np.logical_and(r_data>0,np.logical_and(r_data<0.001,np.logical_and(z_data>-0.35,np.logical_and(np.isfinite(ratio),True))))
	plt.plot((z_data)[select],ratio[select],'x',color=color[index],label='pressure %.3gPa' %(pressure[index]))
	plt.plot(z_data,ratio,',',color=color[index])
	plt.figure(figure_12)
	ratio = vz_Hp/(((temp_e + 5/3*temp_Hp)/eV_to_K*boltzmann_constant_J/hydrogen_mass)**0.5)
	select = np.logical_and(r_data>0,np.logical_and(r_data<0.001,np.logical_and(z_data>-0.35,np.logical_and(np.isfinite(ratio),True))))
	plt.plot((z_data)[select],ratio[select],'x',color=color[index],label='pressure %.3gPa' %(pressure[index]))
	plt.plot(z_data,ratio,',',color=color[index])
	plt.figure(figure_13)
	ratio = vz_H
	select = np.logical_and(r_data>0,np.logical_and(r_data<0.001,np.logical_and(z_data>-0.35,np.logical_and(np.isfinite(ratio),True))))
	plt.plot((z_data)[select],ratio[select],'x',color=color[index],label='pressure %.3gPa' %(pressure[index]))
	plt.plot(z_data,ratio,',',color=color[index])
	plt.figure(figure_14)
	ratio = vz_H2
	select = np.logical_and(r_data>0,np.logical_and(r_data<0.001,np.logical_and(z_data>-0.35,np.logical_and(np.isfinite(ratio),True))))
	plt.plot((z_data)[select],ratio[select],'x',color=color[index],label='pressure %.3gPa' %(pressure[index]))
	plt.plot(z_data,ratio,',',color=color[index])
	plt.figure(figure_15)
	ratio = (vz_H)/vz_Hp
	select = np.logical_and(r_data>0,np.logical_and(r_data<0.02,np.logical_and(z_data>-0.35,np.logical_and(np.isfinite(ratio),np.logical_and(ratio<100,True)))))
	plt.plot((temp_e)[select],ratio[select],'x',color=color[index],label='pressure %.3gPa' %(pressure[index]))
	plt.plot(temp_e,ratio,',',color=color[index])
	plt.figure(figure_16)
	ratio = (vz_H2)/vz_Hp
	select = np.logical_and(r_data>0,np.logical_and(r_data<0.02,np.logical_and(z_data>-0.35,np.logical_and(np.isfinite(ratio),np.logical_and(ratio<100,True)))))
	plt.plot((temp_e)[select],ratio[select],'x',color=color[index],label='pressure %.3gPa' %(pressure[index]))
	plt.plot(temp_e,ratio,',',color=color[index])
	plt.figure(figure_17)
	ratio = vz_H/v_H
	select = np.logical_and(r_data>0,np.logical_and(r_data<0.001,np.logical_and(z_data>-0.35,np.logical_and(np.isfinite(ratio),True))))
	plt.plot((z_data)[select],ratio[select],'x',color=color[index],label='pressure %.3gPa' %(pressure[index]))
	plt.plot(z_data,ratio,',',color=color[index])
	plt.figure(figure_18)
	ratio = vz_H2/v_H2
	select = np.logical_and(r_data>0,np.logical_and(r_data<0.001,np.logical_and(z_data>-0.35,np.logical_and(np.isfinite(ratio),True))))
	plt.plot((z_data)[select],ratio[select],'x',color=color[index],label='pressure %.3gPa' %(pressure[index]))
	plt.plot(z_data,ratio,',',color=color[index])
# plt.plot([0.1,4],[1,1e-3],'k--')
# plt.plot([0.1,4],[1000,1],'k--')
collect_x = [item for sublist in collect_x for item in sublist]
collect_y = [item for sublist in collect_y for item in sublist]
collect_x1 = [item for sublist in collect_x1 for item in sublist]
collect_y1 = [item for sublist in collect_y1 for item in sublist]
collect_temp_e = np.array([item for sublist in collect_temp_e for item in sublist])
collect_dens_e = np.array([item for sublist in collect_dens_e for item in sublist])
collect_temp_H2 = np.array([item for sublist in collect_temp_H2 for item in sublist])
collect_dens_H2 = np.array([item for sublist in collect_dens_H2 for item in sublist])
collect_temp_H = np.array([item for sublist in collect_temp_H for item in sublist])
collect_dens_H = np.array([item for sublist in collect_dens_H for item in sublist])
plt.figure(figure_1)
fit = np.polyfit(np.log(collect_x),np.log(collect_y),2)
collect_x = np.sort(collect_x)
temp2 = np.exp(np.polyval(fit,np.log(collect_x)))
collect_x_max = np.exp(-fit[1]/(2*fit[0]))
temp2[collect_x<collect_x_max]=temp2.max()
plt.plot(collect_x,10*temp2,'k--',label='*10 and /10 of log log fit\n'+str(fit))
plt.plot(collect_x,0.1*temp2,'k--')
plt.plot(collect_x,temp2,'k')
plt.xlabel('Te')
plt.ylabel('n_H2/n_e')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A')
plt.legend(loc='best', fontsize='small'),plt.semilogy(),plt.semilogx()
plt.grid()
plt.figure(figure_2)
fit = np.polyfit(np.log(collect_x1),np.log(collect_y1),2)
collect_x1 = np.sort(collect_x1)
temp2 = np.exp(np.polyval(fit,np.log(collect_x1)))
limit_H_down = interpolate.interp1d(np.log([1e-5,0.1,0.5,2,4]),np.log([0.2,0.2,0.006,0.003,0.003]),fill_value='extrapolate')
temp2 = np.max([temp2,5*np.exp(limit_H_down(np.log(collect_x1)))],axis=0)
plt.plot(collect_x1,temp2,'k--',label='log log fit\n'+str(fit))
plt.xlabel('Te')
plt.ylabel('n_H/n_e')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A')
plt.plot([0.1,4],[20,20],'k:',label='boundaries')
# plt.plot([0.1,0.5,1.1,4],[7,5,0.3,0.3],'r:',label='excluding 1mm close to target')
plt.plot([0.1,0.5,2,4],[0.2,0.006,0.003,0.003],'k:')
plt.legend(loc='best', fontsize='small'),plt.semilogy(),plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_3)
plt.xlabel('Te')
plt.ylabel('T_H')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
plt.plot([0.1,3],[0.1,1.4],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy(),plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_4)
plt.xlabel('Te')
plt.ylabel('T_H2')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
# plt.plot([0.1,4],[0.12,0.3],'k--')
plt.plot([0.1,2,4],[0.1,0.3,0.3],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy(),plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_5)
plt.xlabel('z[mm]')
plt.ylabel('T_e')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A\n conditions at r<=1mm')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
# plt.plot([0.1,4],[0.12,0.3],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy()#,plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_6)
plt.xlabel('z[mm]')
plt.ylabel('n_e')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A\n conditions at r<=1mm')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
# plt.plot([0.1,4],[0.12,0.3],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy()#,plt.semilogx()
plt.pause(0.01)
plt.grid()
plt.figure(figure_7)
plt.xlabel('Te')
plt.ylabel('T_H+')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
plt.plot([0.1,4],[0.1,4],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy(),plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_8)
plt.xlabel('z[mm]')
plt.ylabel('v_e [m/s]')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A\n conditions at r<=1mm')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
# plt.plot([0.1,4],[0.12,0.3],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy()#,plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_9)
plt.xlabel('z[mm]')
plt.ylabel('vz_H+ [m/s]')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A\n conditions at r<=1mm')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
# plt.plot([0.1,4],[0.12,0.3],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy()#,plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_10)
plt.xlabel('z[mm]')
plt.ylabel('total pressure [Pa]')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A\n conditions at r<=1mm')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
# plt.plot([0.1,4],[0.12,0.3],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy()#,plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_11)
plt.xlabel('z[mm]')
plt.ylabel('dynamic pressure [Pa]')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A\n conditions at r<=1mm')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
# plt.plot([0.1,4],[0.12,0.3],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy()#,plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_12)
plt.xlabel('z[mm]')
plt.ylabel('Mach number [au]')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A\n conditions at r<=1mm')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
# plt.plot([0.1,4],[0.12,0.3],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy()#,plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_13)
plt.xlabel('z[mm]')
plt.ylabel('vz_H [m/s]')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A\n conditions at r<=1mm')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
# plt.plot([0.1,4],[0.12,0.3],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy()#,plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_14)
plt.xlabel('z[mm]')
plt.ylabel('vz_H2 [m/s]')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A\n conditions at r<=1mm')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
# plt.plot([0.1,4],[0.12,0.3],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy()#,plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_15)
plt.xlabel('Te')
plt.ylabel('vz_H/vz_H+')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A')
plt.legend(loc='best', fontsize='small'),plt.semilogy(),plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_16)
plt.xlabel('Te')
plt.ylabel('vz_H2/vz_H+')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A')
plt.legend(loc='best', fontsize='small'),plt.semilogy(),plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_17)
plt.xlabel('z[mm]')
plt.ylabel('vz_H/v_H [m/s]')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A\n conditions at r<=1mm')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
# plt.plot([0.1,4],[0.12,0.3],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy()#,plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_18)
plt.xlabel('z[mm]')
plt.ylabel('vz_H2/v_H2 [m/s]')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A\n conditions at r<=1mm')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
# plt.plot([0.1,4],[0.12,0.3],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy()#,plt.semilogx()
plt.grid()
plt.pause(0.01)


plt.figure()
plt.plot(collect_temp_e,(collect_dens_H+collect_dens_H2)/collect_dens_e,'+')
plt.semilogx()
plt.semilogy()
plt.pause(0.01)

plt.figure()
plt.plot(collect_dens_e*collect_temp_e,collect_temp_e,'+')
plt.semilogx()
plt.semilogy()
plt.pause(0.01)



'''Data read only 1 time'''
data_tria = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/B2.5-Eunomia/'
file_array = ['027.dat','053.dat','10.dat','20.dat','44.dat']
Data = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/B2.5-Eunomia/'
selected_file = file_array[-1]
list, grid_cells, data_grid = read_first(Data+selected_file)
r_tri, z_tri, tri = read_triangle(data_tria+'tria.1.node',data_tria+'tria.1.ele')

r_tri = np.asarray(r_tri)
r_tri2 = -r_tri
data_matrix = read_data([Data+selected_file], list, grid_cells)


# datafile=Data.split('/')
# datafile= datafile[1]+'_'+str(len(file_array))+'cyc'
# if os.path.exists(datafile):
    # data = xr.open_dataarray(datafile)
# else:
    # '''Read data'''
    # data = read_data(file_array, list, grid_cells)
    # data.to_netcdf(datafile)

fig, ax = plt.subplots(1,1,figsize=(18,4))
triang=ptri.Triangulation(z_tri,r_tri,tri)
triang2=ptri.Triangulation(z_tri,r_tri2,tri)
# ax.triplot(triang,'-w',linewidth=0.05)
# ax.triplot(triang2,'-w',linewidth=0.05)
# ax[2].set_xlabel('Z (m)', size=22)
# ax[0].set_ylabel('R (m)', size=22)
# ax[1].set_ylabel('R (m)', size=22)
# ax[2].set_ylabel('R (m)', size=22)

# ax[0].tick_params(labelsize=16)
# ax[1].tick_params(labelsize=16)
# ax[2].tick_params(labelsize=16)

temp_e = data_matrix.sel(quantity='t', species='e', filename=Data+selected_file).values
dens_e = data_matrix.sel(quantity='n', species='e', filename=Data+selected_file).values
vz_e = data_matrix.sel(quantity='vz', species='e', filename=Data+selected_file).values
temp_H2 = (data_matrix.sel(quantity='t', species='H_2', filename=Data+selected_file).values)[0]
dens_H2 = (data_matrix.sel(quantity='n', species='H_2', filename=Data+selected_file).values)[0]
temp_H = (data_matrix.sel(quantity='t', species='H', filename=Data+selected_file).values)
dens_H = (data_matrix.sel(quantity='n', species='H', filename=Data+selected_file).values)

# temp_H = data_matrix.sel(quantity='t', species='H', filename='grid045.dat').values
# dens_H = data_matrix.sel(quantity='n', species='H', filename='grid045.dat').values
# temp_H2 = data_matrix.sel(quantity='t', species='H_2', filename='grid045.dat').values
# dens_H2 = data_matrix.sel(quantity='n', species='H_2', filename='grid045.dat').values

# temp_e1 = data_matrix.sel(quantity='t', species='e', filename='mike14.dat').values
# temp_e2 = data_matrix.sel(quantity='t', species='e', filename='mike43.dat').values

# pressure_H = np.multiply(temp_H,dens_H)
# pressure_H2 = np.multiply(temp_H2[0], dens_H2[0])
# pressure = (pressure_H + pressure_H2) * 1.6e-19

z_data = data_grid.sel(axis='z').values
r_data = data_grid.sel(axis='r').values

# a=np.where(abs(z_data - 0.029) <= 0.001)

'''Make colormap like in Visit'''
cdict = {'red':   ((0.0,  0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (0.75,  1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'green': ((0.0,  0.0, 0.0),
                   (0.25, 1.0, 1.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  0.0, 0.0)),

         'blue':  ((0.0,  1.0, 1.0),
                   (0.25,  1.0, 1.0),
                   (0.5, 0.0, 0.0),
                   (1.0,  0.0, 0.0))}

visit = colors.LinearSegmentedColormap('visit', cdict)
plt.register_cmap(cmap=visit)

norm= colors.LogNorm(vmin=1e16, vmax=7e20)

#cs = ax.tripcolor(triang,value, norm=norm, cmap=plt.get_cmap('visit'))

cs = ax.tripcolor(triang,temp_e, cmap=plt.get_cmap('visit'))
cs2 = ax.tripcolor(triang2,temp_e, cmap=plt.get_cmap('visit'))

# cs = ax.tripcolor(triang,vz_e,cmap=plt.get_cmap('visit'))
# cs2 = ax.tripcolor(triang2,vz_e,cmap=plt.get_cmap('visit'))

# cs = ax.tripcolor(triang,dens_e*1e-20, cmap=plt.get_cmap('visit'))
# cs2 = ax.tripcolor(triang2,dens_e*1e-20, cmap=plt.get_cmap('visit'))

#ax.set_title(r'Li$^+$ density, $n_{Li^+}$ (m$^{-3}$)', size=22)
# ax.set_title(r'Temperature, (eV)', size=22)
ax.tick_params(labelsize=16)
ax.set_ylim(-0.25,0.25)
ax.set_xlim(-1.35,0.4)
ax.set_xlabel('Distance from TS/OES location [m]')
ax.set_ylabel('Radious [m]')
# ax.tripcolor(triang,temp_e1, vmin=0.0, vmax = 3.0, cmap=plt.get_cmap('visit'))
# ax.tripcolor(triang2,temp_e1, vmin=0.0, vmax = 3.0, cmap=plt.get_cmap('visit'))

#ax[2].tripcolor(triang,temp_e2, vmin=0.0, vmax = 3.0, cmap=plt.get_cmap('visit'))
#ax[2].tripcolor(triang2,temp_e2, vmin=0.0, vmax = 3.0, cmap=plt.get_cmap('visit'))

# ax.set_xlim(left=-0.01, right=0.06)
# ax.set_ylim(bottom=-0.385, top=0.1)

#ticks = np.geomspace(1e16,7e20,5)
ticks = np.linspace(0.0,3.0,5)
cbaxes = fig.add_axes([0.91, 0.1, 0.01, 0.8])
cb = fig.colorbar(cs, cax=cbaxes, ticks=ticks)

#formatted_ticks = ['%.2E' % i for i in ticks]
formatted_ticks = ['%.2f' % i for i in ticks]
cb.ax.set_yticklabels(formatted_ticks)
cb.ax.tick_params(labelsize=16)
cb.set_label('Temperature [eV]')
# plt.savefig('test.pdf',bbox_inches='tight')
plt.pause(0.01)
