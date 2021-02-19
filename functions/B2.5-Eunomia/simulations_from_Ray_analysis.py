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
# import matplotlib.pyplot  as plt
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
figure_index +=1
figure_19 = figure_index
plt.figure(figure_19)
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
	if False:
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
	plt.figure(figure_19)
	ratio = (dens_H)*temp_H2
	select = np.logical_and(r_data>0,np.logical_and(r_data<0.02,np.logical_and(z_data>-0.35,np.logical_and(np.isfinite(ratio),np.logical_and(True,True)))))
	plt.plot((temp_e*dens_e)[select],ratio[select],'x',color=color[index],label='pressure %.3gPa' %(pressure[index]))
	plt.plot(temp_e*dens_e,ratio,',',color=color[index])

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
plt.xlabel(r'$T_e$'+' [eV]')
plt.ylabel(r'$n_{H_2} / n_e$')
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
plt.xlabel(r'$T_e$'+' [eV]')
plt.ylabel(r'$n_H / n_e$')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A')
plt.plot([0.1,4],[20,20],'k:',label='boundaries')
# plt.plot([0.1,0.5,1.1,4],[7,5,0.3,0.3],'r:',label='excluding 1mm close to target')
plt.plot([0.1,0.5,2,4],[0.2,0.006,0.003,0.003],'k:')
plt.legend(loc='best', fontsize='small'),plt.semilogy(),plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_3)
plt.xlabel(r'$T_e$'+' [eV]')
plt.ylabel(r'$T_H$'+' [eV]')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
plt.plot([0.1,3],[0.1,1.4],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy(),plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_4)
plt.xlabel(r'$T_e$'+' [eV]')
plt.ylabel(r'$T_{H_2}$'+' [eV]')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
# plt.plot([0.1,4],[0.12,0.3],'k--')
plt.plot([0.1,2,4],[0.1,0.3,0.3],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy(),plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_5)
plt.xlabel('z[mm]')
plt.ylabel(r'$T_e$'+' [eV]')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A\n conditions at r<=1mm')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
# plt.plot([0.1,4],[0.12,0.3],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy()#,plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_6)
plt.xlabel('z[mm]')
plt.ylabel(r'$n_e$')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A\n conditions at r<=1mm')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
# plt.plot([0.1,4],[0.12,0.3],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy()#,plt.semilogx()
plt.pause(0.01)
plt.grid()
plt.figure(figure_7)
plt.xlabel(r'$T_e$'+' [eV]')
plt.ylabel(r'$T_{H^+}$'+' [eV]')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
plt.plot([0.1,4],[0.1,4],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy(),plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_8)
plt.xlabel('z[mm]')
plt.ylabel(r'$v_e$'+' [m/s]')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A\n conditions at r<=1mm')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
# plt.plot([0.1,4],[0.12,0.3],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy()#,plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_9)
plt.xlabel('z[mm]')
plt.ylabel(r'$vz_{H^+}$'+' [m/s]')
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
plt.ylabel(r'$vz_H$'+' [m/s]')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A\n conditions at r<=1mm')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
# plt.plot([0.1,4],[0.12,0.3],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy()#,plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_14)
plt.xlabel('z[mm]')
plt.ylabel(r'$vz_{H_2}$'+' [m/s]')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A\n conditions at r<=1mm')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
# plt.plot([0.1,4],[0.12,0.3],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy()#,plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_15)
plt.xlabel(r'$T_e$'+' [eV]')
plt.ylabel(r'$vz_H / vz_{H^+}$')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A')
plt.legend(loc='best', fontsize='small'),plt.semilogy(),plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_16)
plt.xlabel(r'$T_e$'+' [eV]')
plt.ylabel(r'$vz_{H_2} / vz_{H^+}$')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A')
plt.legend(loc='best', fontsize='small'),plt.semilogy(),plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_17)
plt.xlabel('z[mm]')
plt.ylabel(r'$vz_H / v_H$'+' [m/s]')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A\n conditions at r<=1mm')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
# plt.plot([0.1,4],[0.12,0.3],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy()#,plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_18)
plt.xlabel('z[mm]')
plt.ylabel(r'$vz_{H_2}/v_{H_2}$'+' [m/s]')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A\n conditions at r<=1mm')
# plt.plot([0.1,0.8,2.5,4],[5,5,0.2,0.2],'k--',label='boundaries')
# plt.plot([0.1,4],[0.12,0.3],'k--')
plt.legend(loc='best', fontsize='small'),plt.semilogy()#,plt.semilogx()
plt.grid()
plt.pause(0.01)
plt.figure(figure_19)
plt.xlabel(r'$T_e * n_e$'+' [eV]')
plt.ylabel(r'$n_{H_2} * T_{H_2}$')
plt.title('B = 1.2T, Plasma source of 4 slm , 120A')
plt.legend(loc='best', fontsize='small'),plt.semilogy(),plt.semilogx()
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






# Added 18/02/2021 only to plot an example of what the distribution of excited states is for atomic and molecular processes
exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())
os.chdir('/home/ffederic/work/Collaboratory/test/experimental_data')
exec(open("/home/ffederic/work/Collaboratory/test/experimental_data/functions/MolRad_Yacora/Yacora_FF/import_PECs_FF_2.py").read())

pecfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/pec12#h_balmer#h0.dat'
pecfile_2 = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/pec12#h_pju#h0.dat'
scdfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/scd12_h.dat'
acdfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/acd12_h.dat'
pltfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/plt12_h.dat'
prbfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/prb12_h.dat'
ccdfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/ccd96_h.dat'

# Data from wikipedia
energy_difference = np.array([2.54970, 2.85567, 3.02187, 3.12208, 3.18716, 3.23175, 3.26365, 3.28725, 3.30520])  # eV
energy_difference_full = np.array([10.1988,1.88867, 2.54970, 2.85567, 3.02187, 3.12208, 3.18716, 3.23175, 3.26365, 3.28725, 3.30520, 3.31917])  # eV
# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
statistical_weigth = np.array([32, 50, 72, 98, 128, 162, 200, 242, 288])  # gi-gk
einstein_coeff = np.array([8.4193e-2, 2.53044e-2, 9.7320e-3, 4.3889e-3, 2.2148e-3, 1.2156e-3, 7.1225e-4, 4.3972e-4, 2.8337e-4]) * 1e8  # 1/s
einstein_coeff_full = np.array([4.6986e+00, 4.4101e-01, 8.4193e-2, 2.53044e-2, 9.7320e-3, 4.3889e-3, 2.2148e-3, 1.2156e-3, 7.1225e-4, 4.3972e-4, 2.8337e-4, 1.8927e-04]) * 1e8  # 1/s
einstein_coeff_full_full = np.array([[4.6986,5.5751e-01,1.2785e-01,4.1250e-02,1.6440e-02,7.5684e-03,3.8694e-03,2.1425e-03,1.2631e-03,7.8340e-04,5.0659e-04,3.3927e-04],[0,0.44101,0.084193,0.0253044,0.009732,0.0043889,0.0022148,0.0012156,0.00071225,0.00043972,0.00028337,0.00018927],[0,0,8.9860e-02,2.2008e-02,7.7829e-03,3.3585e-03,1.6506e-03,8.9050e-04,5.1558e-04,3.1558e-04,2.0207e-04,1.3431e-04],[0,0,0,2.6993e-02,7.7110e-03,3.0415e-03,1.4242e-03,7.4593e-04,4.2347e-04,2.5565e-04,1.6205e-04,1.0689e-04],[0,0,0,0,1.0254e-02,3.2528e-03,1.3877e-03,6.9078e-04,3.7999e-04,2.2460e-04,1.4024e-04,9.1481e-05],[0,0,0,0,0,4.5608e-03,1.5609e-03,7.0652e-04,3.6881e-04,2.1096e-04,1.2884e-04,8.2716e-05],[0,0,0,0,0,0,2.2720e-03,8.2370e-04,3.9049e-04,2.1174e-04,1.2503e-04,7.8457e-05],[0,0,0,0,0,0,0,1.2328e-03,4.6762e-04,2.3007e-04,1.2870e-04,7.8037e-05],[0,0,0,0,0,0,0,0,7.1514e-04,2.8131e-04,1.4269e-04,8.1919e-05],[0,0,0,0,0,0,0,0,0,4.3766e-04,1.7740e-04,9.2309e-05],[0,0,0,0,0,0,0,0,0,0,2.7989e-04,1.1633e-04],[0,0,0,0,0,0,0,0,0,0,0,1.8569e-04]]) * 1e8  # 1/s
level_1 = (np.ones((13,13))*np.arange(1,14)).T
level_2 = (np.ones((13,13))*np.arange(1,14))
# Used formula 2.9 and 2.12 in Rion Barrois thesys, 2017
energy_difference_full_full = 1/level_1**2-1/level_2**2
energy_difference_full_full = 13.6*energy_difference_full_full[:-1,1:]
energy_difference_full_full[energy_difference_full_full<0]=0	# energy difference between energy levels [eV]
plank_constant_eV = 4.135667696e-15	# eV s
light_speed = 299792458	# m/s
photon_wavelength_full_full = plank_constant_eV * light_speed / energy_difference_full_full	# m
visible_light_flag_full_full = np.logical_and(photon_wavelength_full_full>=380*1e-9,photon_wavelength_full_full<=750*1e-9)
J_to_eV = 6.242e18
multiplicative_factor_full = energy_difference_full * einstein_coeff_full / J_to_eV
multiplicative_factor_full_full = np.sum(energy_difference_full_full * einstein_coeff_full_full / J_to_eV,axis=0)
multiplicative_factor_visible_light_full_full = np.sum(energy_difference_full_full * visible_light_flag_full_full * einstein_coeff_full_full / J_to_eV,axis=0)
au_to_kg = 1.66053906660e-27	# kg/au
# Used formula 2.3 in Rion Barrois thesys, 2017
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro','paleturquoise']
boltzmann_constant_J = 1.380649e-23	# J/K
eV_to_K = 8.617333262145e-5	# eV/K
avogadro_number = 6.02214076e23
hydrogen_mass = 1.008*1.660*1e-27	# kg
electron_mass = 9.10938356* 1e-31	# kg
ionisation_potential = 13.6	# eV
dissociation_potential = 2.2	# eV




# Example points

ne_values = np.array([1e20,4e21,2e21])
Te_values = np.array([0.5,2,8])


multiplicative_factor_full = energy_difference_full * einstein_coeff_full / J_to_eV
excitation_full = []
for isel in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
	if isel==0:
		temp = read_adf15(pecfile_2, 1, Te_values.flatten(),(ne_values * 10 ** (- 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
	else:
		temp = read_adf15(pecfile, isel, Te_values.flatten(),(ne_values * 10 ** (- 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
	temp[np.isnan(temp)] = 0
	temp[np.isinf(temp)] = 0
	# temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
	excitation_full.append(temp)
del temp
excitation_full = np.array(excitation_full)  # in # photons cm^-3 s^-1
excitation_full = (excitation_full.T * (10 ** -6) * (energy_difference_full / J_to_eV))  # in W m^-3 / (# / m^3)**2
excitation_full = (excitation_full /multiplicative_factor_full)  # in m^-3 / (# / m^3)**2

recombination_full = []
for isel in [0, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
	if isel==0:
		temp = read_adf15(pecfile_2, 67, Te_values.flatten(),(ne_values * 10 ** (- 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
	else:
		temp = read_adf15(pecfile, isel, Te_values.flatten(),(ne_values * 10 ** (- 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
	temp[np.isnan(temp)] = 0
	# temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
	recombination_full.append(temp)
del temp
recombination_full = np.array(recombination_full)  # in # photons cm^-3 s^-1 / (# cm^-3)**2
recombination_full = (recombination_full.T * (10 ** -6) * (energy_difference_full / J_to_eV))  # in W m^-3 / (# / m^3)**2
recombination_full = (recombination_full /multiplicative_factor_full)  # in # m^-3 / (# / m^3)**2



total_wavelengths = np.unique(excited_states_From_Hn_with_Hp)
T_Hp_values = Te_values/eV_to_K	# K
T_Hp_values[T_Hp_values<300]=300
T_Hm_values = np.exp(TH2_fit_from_simulations(np.log(Te_values)))/eV_to_K	# K
T_Hm_values[T_Hm_values<300]=300
T_H2p_values = np.exp(TH2_fit_from_simulations(np.log(Te_values)))/eV_to_K	# K
T_H2p_values[T_H2p_values<300]=300

record_nH_ne_values = []
for i_Te_for_nH_ne,Te_for_nH_ne in enumerate(Te_values):
	nH_ne_values = nH_ne_values_Te(Te_for_nH_ne,1)[0]
	record_nH_ne_values.append(nH_ne_values)
record_nH_ne_values = np.array(record_nH_ne_values)
nHp_ne_values = np.ones_like(ne_values)
nH_ne_excited_states_atomic_recomb = (recombination_full.T * nHp_ne_values * ne_values).T
emissivity_excited_states_atomic_recomb = (nH_ne_excited_states_atomic_recomb.T * ne_values).T * multiplicative_factor_full
nHex_nH_excited_states_atomic_recomb = (recombination_full.T * nHp_ne_values * ne_values / record_nH_ne_values).T
nH_ne_nHp_excited_states_atomic_recomb = recombination_full
nH_ne_excited_states_atomic_excit = (excitation_full.T * record_nH_ne_values * ne_values).T
emissivity_excited_states_atomic_excit = (nH_ne_excited_states_atomic_excit.T * ne_values).T * multiplicative_factor_full
nHex_nH_excited_states_atomic_excit = (excitation_full.T * record_nH_ne_values * ne_values / record_nH_ne_values).T
nH_ne_nH_excited_states_atomic_excit = excitation_full

record_nH2_ne_values = []
for i_Te_for_nH2_ne,Te_for_nH2_ne in enumerate(Te_values):
	nH2_ne_values = nH2_ne_values_Te(Te_for_nH2_ne,1)[0]
	record_nH2_ne_values.append(nH2_ne_values)
record_nH2_ne_values = np.array(record_nH2_ne_values)

record_nH2p_nH2_values = nH2p_nH2_values_Te_ne(Te_values,ne_values,1)[0]
nH2p_ne_values = record_nH2_ne_values * record_nH2p_nH2_values
record_nHm_nH2_values = nHm_nH2_values_Te(Te_values,1)[0]
nHm_ne_values = record_nH2_ne_values * record_nHm_nH2_values

nHp_ne_values = 1 - nH2p_ne_values + nHm_ne_values


coeff_1 = From_H2p_pop_coeff_full_extra(np.array([Te_values,ne_values]).T,total_wavelengths)
nH_ne_excited_states_mol_H2p = (coeff_1.T * nH2p_ne_values * ne_values).T
emissivity_excited_states_mol_H2p = (nH_ne_excited_states_mol_H2p.T * ne_values).T * multiplicative_factor_full
nHex_nH_excited_states_mol_H2p = (coeff_1.T * nH2p_ne_values * ne_values / record_nH_ne_values).T
nH_ne_nH2p_excited_states_mol_H2p = coeff_1
coeff_2 = From_H2_pop_coeff_full_extra(np.array([Te_values,ne_values]).T,total_wavelengths)
nH_ne_excited_states_mol_H2 = (coeff_2.T * record_nH2_ne_values * ne_values).T
emissivity_excited_states_mol_H2 = (nH_ne_excited_states_mol_H2.T * ne_values).T * multiplicative_factor_full
nHex_nH_excited_states_mol_H2 = (coeff_2.T * record_nH2_ne_values * ne_values / record_nH_ne_values).T
nH_ne_nH2_excited_states_mol_H2 = coeff_2
coeff_3 = From_Hn_with_Hp_pop_coeff_full_extra(np.array([Te_values,T_Hp_values,T_Hm_values,ne_values,nHp_ne_values*ne_values]).T ,total_wavelengths)
nH_ne_excited_states_mol_Hn_with_Hp = (coeff_3.T * nHm_ne_values * ne_values).T
emissivity_excited_states_mol_Hn_with_Hp = (nH_ne_excited_states_mol_Hn_with_Hp.T * ne_values).T * multiplicative_factor_full
nHex_nH_excited_states_mol_Hn_with_Hp = (coeff_3.T * nHm_ne_values * ne_values / record_nH_ne_values).T
nH_nHn_nHp_excited_states_mol_Hn_with_Hp = (coeff_3.T * ne_values /(nHp_ne_values * ne_values)).T
coeff_4 = From_Hn_with_H2p_pop_coeff_full_extra(np.array([Te_values,T_H2p_values,T_Hm_values,ne_values,nH2p_ne_values*ne_values]).T,total_wavelengths)
nH_ne_excited_states_mol_Hn_with_H2p = (coeff_4.T * nHm_ne_values * ne_values).T
emissivity_excited_states_mol_Hn_with_H2p = (nH_ne_excited_states_mol_Hn_with_H2p.T * ne_values).T * multiplicative_factor_full
nHex_nH_excited_states_mol_Hn_with_H2p = (coeff_4.T * nHm_ne_values * ne_values / record_nH_ne_values).T
nH_nHn_nH2p_excited_states_mol_Hn_with_H2p = (coeff_4.T * ne_values / (nH2p_ne_values * ne_values)).T

linestyle = ['-','--','-.']
plt.figure()
for index in range(len(ne_values)):
	max_nHex_nH = np.max([nHex_nH_excited_states_atomic_recomb[index],nHex_nH_excited_states_atomic_excit[index],nHex_nH_excited_states_mol_H2p[index],nHex_nH_excited_states_mol_H2[index],nHex_nH_excited_states_mol_Hn_with_Hp[index],nHex_nH_excited_states_mol_Hn_with_H2p[index]])
	if index == 0:
		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_atomic_recomb[index][:6+1],color=color[0],linestyle=linestyle[index],label='recombination (ADAS)\n'+r'$H^+ + e^- → H(p) + hν$'+'\n'+r'$H^+ + 2e^- → H(p) + e^-$')
		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_atomic_excit[index][:6+1],color=color[1],linestyle=linestyle[index],label='direct excitation (ADAS)\n'+r'$H(q) + e^- → H(p>q) + e^-$')
		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_mol_H2p[index][:6+1],color=color[2],linestyle=linestyle[index],label='H2+ dissociation (Yacora)\n'+r'${H_2}^+ + e^- → H(p) + H^+ + e^-$')
		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_mol_H2[index][:6+1],color=color[3],linestyle=linestyle[index],label='H2 dissociation (Yacora)\n'+r'$H_2 + e^- → H(p) + H(1) + e^-$')
		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_mol_Hn_with_Hp[index][:6+1],color=color[4],linestyle=linestyle[index],label='H+ mutual neutralisation (Yacora)\n'+r'$H^+ + H^- → H(p) + H(1)$')
		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_mol_Hn_with_H2p[index][:6+1],color=color[5],linestyle=linestyle[index],label='H2+ mutual neutralisation (Yacora)\n'+r'${H_2}^+ + H^- → H(p) + H_2$')
	else:
		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_atomic_recomb[index][:6+1],color=color[0],linestyle=linestyle[index])
		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_atomic_excit[index][:6+1],color=color[1],linestyle=linestyle[index])
		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_mol_H2p[index][:6+1],color=color[2],linestyle=linestyle[index])
		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_mol_H2[index][:6+1],color=color[3],linestyle=linestyle[index])
		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_mol_Hn_with_Hp[index][:6+1],color=color[4],linestyle=linestyle[index])
		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_mol_Hn_with_H2p[index][:6+1],color=color[5],linestyle=linestyle[index])

plt.xlabel('H excited state '+r'$p$')
plt.ylabel(r'$n_{H(p)} / n_H$')
temp = np.array([Te_values,ne_values/1e20]).T
plt.title('[Te[eV],ne['+r'$10^{20}$'+'#'+r'$/m^3$'+']] examples → "'+linestyle[0] +'"='+ str(temp[0])+ ', "'+linestyle[1] +'"='+ str(temp[1])+ ', "'+linestyle[2] +'"='+ str(temp[2]))
plt.legend(loc='lower right', fontsize='x-small')
plt.semilogy()
# plt.semilogx()
plt.grid()
plt.pause(0.01)

linestyle = ['-','--','-.']
plt.figure()
for index in range(len(ne_values)):
	# max_nH_ne = np.max([nH_ne_excited_states_atomic_recomb[index],nH_ne_excited_states_atomic_excit[index],nH_ne_excited_states_mol_H2p[index],nH_ne_excited_states_mol_H2[index],nH_ne_excited_states_mol_Hn_with_Hp[index],nH_ne_excited_states_mol_Hn_with_H2p[index]])
	if index == 0:
		plt.plot(total_wavelengths[:6+1],nH_ne_excited_states_atomic_recomb[index][:6+1]/(nH_ne_excited_states_atomic_recomb[index][2]),color=color[0],linestyle=linestyle[index],label='recombination (ADAS)\n'+r'$H^+ + e^- → H(p) + hν$'+'\n'+r'$H^+ + 2e^- → H(p) + e^-$')
		plt.plot(total_wavelengths[:6+1],nH_ne_excited_states_atomic_excit[index][:6+1]/(nH_ne_excited_states_atomic_excit[index][2]),color=color[1],linestyle=linestyle[index],label='direct excitation (ADAS)\n'+r'$H(q) + e^- → H(p>q) + e^-$')
		plt.plot(total_wavelengths[:6+1],nH_ne_excited_states_mol_H2p[index][:6+1]/(nH_ne_excited_states_mol_H2p[index][2]),color=color[2],linestyle=linestyle[index],label='H2+ dissociation (Yacora)\n'+r'${H_2}^+ + e^- → H(p) + H^+ + e^-$')
		plt.plot(total_wavelengths[:6+1],nH_ne_excited_states_mol_H2[index][:6+1]/(nH_ne_excited_states_mol_H2[index][2]),color=color[3],linestyle=linestyle[index],label='H2 dissociation (Yacora)\n'+r'$H_2 + e^- → H(p) + H(1) + e^-$')
		plt.plot(total_wavelengths[:6+1],nH_ne_excited_states_mol_Hn_with_Hp[index][:6+1]/(nH_ne_excited_states_mol_Hn_with_Hp[index][2]),color=color[4],linestyle=linestyle[index],label='H+ mutual neutralisation (Yacora)\n'+r'$H^+ + H^- → H(p) + H(1)$')
		plt.plot(total_wavelengths[:6+1],nH_ne_excited_states_mol_Hn_with_H2p[index][:6+1]/(nH_ne_excited_states_mol_Hn_with_H2p[index][2]),color=color[5],linestyle=linestyle[index],label='H2+ mutual neutralisation (Yacora)\n'+r'${H_2}^+ + H^- → H(p) + H_2$')
	else:
		plt.plot(total_wavelengths[:6+1],nH_ne_excited_states_atomic_recomb[index][:6+1]/(nH_ne_excited_states_atomic_recomb[index][2]),color=color[0],linestyle=linestyle[index])
		plt.plot(total_wavelengths[:6+1],nH_ne_excited_states_atomic_excit[index][:6+1]/(nH_ne_excited_states_atomic_excit[index][2]),color=color[1],linestyle=linestyle[index])
		plt.plot(total_wavelengths[:6+1],nH_ne_excited_states_mol_H2p[index][:6+1]/(nH_ne_excited_states_mol_H2p[index][2]),color=color[2],linestyle=linestyle[index])
		plt.plot(total_wavelengths[:6+1],nH_ne_excited_states_mol_H2[index][:6+1]/(nH_ne_excited_states_mol_H2[index][2]),color=color[3],linestyle=linestyle[index])
		plt.plot(total_wavelengths[:6+1],nH_ne_excited_states_mol_Hn_with_Hp[index][:6+1]/(nH_ne_excited_states_mol_Hn_with_Hp[index][2]),color=color[4],linestyle=linestyle[index])
		plt.plot(total_wavelengths[:6+1],nH_ne_excited_states_mol_Hn_with_H2p[index][:6+1]/(nH_ne_excited_states_mol_Hn_with_H2p[index][2]),color=color[5],linestyle=linestyle[index])

plt.xlabel('H excited state '+r'$p$')
plt.ylabel(r'$n_{H(p)} / n_{H(4)}$')
temp = np.array([Te_values,ne_values/1e20]).T
plt.title('[Te[eV],ne['+r'$10^{20}$'+'#'+r'$/m^3$'+']] examples → "'+linestyle[0] +'"='+ str(temp[0])+ ', "'+linestyle[1] +'"='+ str(temp[1])+ ', "'+linestyle[2] +'"='+ str(temp[2]))
plt.legend(loc='best', fontsize='x-small')
plt.semilogy()
# plt.semilogx()
plt.grid()
plt.pause(0.01)

linestyle = ['-','--','-.']
plt.figure()
for index in range(len(ne_values)):
	# max_nH_ne = np.max([nH_ne_excited_states_atomic_recomb[index],nH_ne_excited_states_atomic_excit[index],nH_ne_excited_states_mol_H2p[index],nH_ne_excited_states_mol_H2[index],nH_ne_excited_states_mol_Hn_with_Hp[index],nH_ne_excited_states_mol_Hn_with_H2p[index]])
	if index == 0:
		plt.plot(total_wavelengths[:6+1],emissivity_excited_states_atomic_recomb[index][:6+1]/(emissivity_excited_states_atomic_recomb[index][2]),color=color[0],linestyle=linestyle[index],label='recombination (ADAS)\n'+r'$H^+ + e^- → H(p) + hν$'+'\n'+r'$H^+ + 2e^- → H(p) + e^-$')
		plt.plot(total_wavelengths[:6+1],emissivity_excited_states_atomic_excit[index][:6+1]/(emissivity_excited_states_atomic_excit[index][2]),color=color[1],linestyle=linestyle[index],label='direct excitation (ADAS)\n'+r'$H(q) + e^- → H(p>q) + e^-$')
		plt.plot(total_wavelengths[:6+1],emissivity_excited_states_mol_H2p[index][:6+1]/(emissivity_excited_states_mol_H2p[index][2]),color=color[2],linestyle=linestyle[index],label='H2+ dissociation (Yacora)\n'+r'${H_2}^+ + e^- → H(p) + H^+ + e^-$')
		plt.plot(total_wavelengths[:6+1],emissivity_excited_states_mol_H2[index][:6+1]/(emissivity_excited_states_mol_H2[index][2]),color=color[3],linestyle=linestyle[index],label='H2 dissociation (Yacora)\n'+r'$H_2 + e^- → H(p) + H(1) + e^-$')
		plt.plot(total_wavelengths[:6+1],emissivity_excited_states_mol_Hn_with_Hp[index][:6+1]/(emissivity_excited_states_mol_Hn_with_Hp[index][2]),color=color[4],linestyle=linestyle[index],label='H+ mutual neutralisation (Yacora)\n'+r'$H^+ + H^- → H(p) + H(1)$')
		plt.plot(total_wavelengths[:6+1],emissivity_excited_states_mol_Hn_with_H2p[index][:6+1]/(emissivity_excited_states_mol_Hn_with_H2p[index][2]),color=color[5],linestyle=linestyle[index],label='H2+ mutual neutralisation (Yacora)\n'+r'${H_2}^+ + H^- → H(p) + H_2$')
	else:
		plt.plot(total_wavelengths[:6+1],emissivity_excited_states_atomic_recomb[index][:6+1]/(emissivity_excited_states_atomic_recomb[index][2]),color=color[0],linestyle=linestyle[index])
		plt.plot(total_wavelengths[:6+1],emissivity_excited_states_atomic_excit[index][:6+1]/(emissivity_excited_states_atomic_excit[index][2]),color=color[1],linestyle=linestyle[index])
		plt.plot(total_wavelengths[:6+1],emissivity_excited_states_mol_H2p[index][:6+1]/(emissivity_excited_states_mol_H2p[index][2]),color=color[2],linestyle=linestyle[index])
		plt.plot(total_wavelengths[:6+1],emissivity_excited_states_mol_H2[index][:6+1]/(emissivity_excited_states_mol_H2[index][2]),color=color[3],linestyle=linestyle[index])
		plt.plot(total_wavelengths[:6+1],emissivity_excited_states_mol_Hn_with_Hp[index][:6+1]/(emissivity_excited_states_mol_Hn_with_Hp[index][2]),color=color[4],linestyle=linestyle[index])
		plt.plot(total_wavelengths[:6+1],emissivity_excited_states_mol_Hn_with_H2p[index][:6+1]/(emissivity_excited_states_mol_Hn_with_H2p[index][2]),color=color[5],linestyle=linestyle[index])

plt.xlabel('H excited state '+r'$p$')
plt.ylabel('line ratio '+r'$\epsilon_{H(p)} / \epsilon_{H(4)}$')
temp = np.array([Te_values,ne_values/1e20]).T
plt.title('[Te[eV],ne['+r'$10^{20}$'+'#'+r'$/m^3$'+']] examples → "'+linestyle[0] +'"='+ str(temp[0])+ ', "'+linestyle[1] +'"='+ str(temp[1])+ ', "'+linestyle[2] +'"='+ str(temp[2]))
plt.legend(loc='best', fontsize='x-small')
plt.semilogy()
# plt.semilogx()
plt.grid()
plt.pause(0.01)


# linestyle = ['-','--',':']
# plt.figure()
# for index in range(len(ne_values)):
# 	max_nH_ne = np.max([nH_ne_excited_states_atomic_recomb[index],nH_ne_excited_states_atomic_excit[index],nH_ne_excited_states_mol_H2p[index],nH_ne_excited_states_mol_H2[index],nH_ne_excited_states_mol_Hn_with_Hp[index],nH_ne_excited_states_mol_Hn_with_H2p[index]])
# 	if index == 0:
# 		plt.plot(total_wavelengths[:6+1],nH_ne_nHp_excited_states_atomic_recomb[index][:6+1]/(nH_ne_nHp_excited_states_atomic_recomb[index][2]),color=color[0],linestyle=linestyle[index],label='recombination (ADAS)\n'+r'$H^+ + e^- → H(p) + hν$'+'\n'+r'$H^+ + 2e^- → H(p) + e^-$')
# 		plt.plot(total_wavelengths[:6+1],nH_ne_nH_excited_states_atomic_excit[index][:6+1]/(nH_ne_nH_excited_states_atomic_excit[index][2]),color=color[1],linestyle=linestyle[index],label='direct excitation (ADAS)\n'+r'$H(q) + e^- → H(p>q) + e^-$')
# 		plt.plot(total_wavelengths[:6+1],nH_ne_nH2p_excited_states_mol_H2p[index][:6+1]/(nH_ne_nH2p_excited_states_mol_H2p[index][2]),color=color[2],linestyle=linestyle[index],label='H2+ dissociation (Yacora)\n'+r'${H_2}^+ + e^- → H(p) + H^+ + e^-$')
# 		plt.plot(total_wavelengths[:6+1],nH_ne_nH2_excited_states_mol_H2[index][:6+1]/(nH_ne_nH2_excited_states_mol_H2[index][2]),color=color[3],linestyle=linestyle[index],label='H2 dissociation\n'+r'$H_2 + e^- → H(p) + H(1) + e^-$')
# 		plt.plot(total_wavelengths[:6+1],nH_nHn_nHp_excited_states_mol_Hn_with_Hp[index][:6+1]/(nH_nHn_nHp_excited_states_mol_Hn_with_Hp[index][2]),color=color[4],linestyle=linestyle[index],label='H+ mutual neutralisation\n'+r'$H^+ + H^- → H(p) + H(1)$')
# 		plt.plot(total_wavelengths[:6+1],nH_nHn_nH2p_excited_states_mol_Hn_with_H2p[index][:6+1]/(nH_nHn_nH2p_excited_states_mol_Hn_with_H2p[index][2]),color=color[5],linestyle=linestyle[index],label='H2+ mutual neutralisation\n'+r'${H_2}^+ + H^- → H(p) + H_2$')
# 	else:
# 		plt.plot(total_wavelengths[:6+1],nH_ne_nHp_excited_states_atomic_recomb[index][:6+1]/(nH_ne_nHp_excited_states_atomic_recomb[index][2]),color=color[0],linestyle=linestyle[index])
# 		plt.plot(total_wavelengths[:6+1],nH_ne_nH_excited_states_atomic_excit[index][:6+1]/(nH_ne_nH_excited_states_atomic_excit[index][2]),color=color[1],linestyle=linestyle[index])
# 		plt.plot(total_wavelengths[:6+1],nH_ne_nH2p_excited_states_mol_H2p[index][:6+1]/(nH_ne_nH2p_excited_states_mol_H2p[index][2]),color=color[2],linestyle=linestyle[index])
# 		plt.plot(total_wavelengths[:6+1],nH_ne_nH2_excited_states_mol_H2[index][:6+1]/(nH_ne_nH2_excited_states_mol_H2[index][2]),color=color[3],linestyle=linestyle[index])
# 		plt.plot(total_wavelengths[:6+1],nH_nHn_nHp_excited_states_mol_Hn_with_Hp[index][:6+1]/(nH_nHn_nHp_excited_states_mol_Hn_with_Hp[index][2]),color=color[4],linestyle=linestyle[index])
# 		plt.plot(total_wavelengths[:6+1],nH_nHn_nH2p_excited_states_mol_Hn_with_H2p[index][:6+1]/(nH_nHn_nH2p_excited_states_mol_Hn_with_H2p[index][2]),color=color[5],linestyle=linestyle[index])
#
# plt.xlabel('atomic hydrogen excited state')
# plt.ylabel('relative '+r'$n_{H(p)} / n_e$')
# # plt.title('B = 1.2T, Plasma source of 4 slm , 120A')
# plt.legend(loc='best', fontsize='x-small')
# plt.semilogy()
# # plt.semilogx()
# plt.pause(0.01)

# linestyle = ['-','--',':']
# plt.figure()
# for index in range(len(ne_values)):
# 	max_nH_ne = np.max([nHex_nH_excited_states_atomic_recomb[index],nHex_nH_excited_states_atomic_excit[index],nHex_nH_excited_states_mol_H2p[index],nHex_nH_excited_states_mol_H2[index],nHex_nH_excited_states_mol_Hn_with_Hp[index],nHex_nH_excited_states_mol_Hn_with_H2p[index]])
# 	if index == 0:
# 		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_atomic_recomb[index][:6+1]/(nHex_nH_excited_states_atomic_recomb[index][2]),color=color[0],linestyle=linestyle[index],label='recombination (ADAS)\n'+r'$H^+ + e^- → H(p) + hν$'+'\n'+r'$H^+ + 2e^- → H(p) + e^-$')
# 		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_atomic_excit[index][:6+1]/(nHex_nH_excited_states_atomic_excit[index][2]),color=color[1],linestyle=linestyle[index],label='direct excitation (ADAS)\n'+r'$H(q) + e^- → H(p>q) + e^-$')
# 		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_mol_H2p[index][:6+1]/(nHex_nH_excited_states_mol_H2p[index][2]),color=color[2],linestyle=linestyle[index],label='H2+ dissociation (Yacora)\n'+r'${H_2}^+ + e^- → H(p) + H^+ + e^-$')
# 		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_mol_H2[index][:6+1]/(nHex_nH_excited_states_mol_H2[index][2]),color=color[3],linestyle=linestyle[index],label='H2 dissociation\n'+r'$H_2 + e^- → H(p) + H(1) + e^-$')
# 		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_mol_Hn_with_Hp[index][:6+1]/(nHex_nH_excited_states_mol_Hn_with_Hp[index][2]),color=color[4],linestyle=linestyle[index],label='H+ mutual neutralisation\n'+r'$H^+ + H^- → H(p) + H(1)$')
# 		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_mol_Hn_with_H2p[index][:6+1]/(nHex_nH_excited_states_mol_Hn_with_H2p[index][2]),color=color[5],linestyle=linestyle[index],label='H2+ mutual neutralisation\n'+r'${H_2}^+ + H^- → H(p) + H_2$')
# 	else:
# 		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_atomic_recomb[index][:6+1]/(nHex_nH_excited_states_atomic_recomb[index][2]),color=color[0],linestyle=linestyle[index])
# 		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_atomic_excit[index][:6+1]/(nHex_nH_excited_states_atomic_excit[index][2]),color=color[1],linestyle=linestyle[index])
# 		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_mol_H2p[index][:6+1]/(nHex_nH_excited_states_mol_H2p[index][2]),color=color[2],linestyle=linestyle[index])
# 		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_mol_H2[index][:6+1]/(nHex_nH_excited_states_mol_H2[index][2]),color=color[3],linestyle=linestyle[index])
# 		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_mol_Hn_with_Hp[index][:6+1]/(nHex_nH_excited_states_mol_Hn_with_Hp[index][2]),color=color[4],linestyle=linestyle[index])
# 		plt.plot(total_wavelengths[:6+1],nHex_nH_excited_states_mol_Hn_with_H2p[index][:6+1]/(nHex_nH_excited_states_mol_Hn_with_H2p[index][2]),color=color[5],linestyle=linestyle[index])
#
# plt.xlabel('atomic hydrogen excited state')
# plt.ylabel('relative '+r'$n_{H(p)} / n_e$')
# # plt.title('B = 1.2T, Plasma source of 4 slm , 120A')
# plt.legend(loc='best', fontsize='x-small')
# plt.semilogy()
# # plt.semilogx()
# plt.pause(0.01)


#
#
