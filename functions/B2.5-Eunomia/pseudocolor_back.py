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
    data = np.reshape(data, (2, count))
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

Data = './'
file_array = ['mike045.dat','mike14.dat','mike43.dat']
print('Reading data file..')

'''Data read only 1 time'''
data_tria = './'
list, grid_cells, data_grid = read_first(file_array[0])
r_tri, z_tri, tri = read_triangle(data_tria+'tria.1.node',data_tria+'tria.1.ele')

r_tri = np.asarray(r_tri)
r_tri2 = -r_tri
data_matrix = read_data(file_array, list, grid_cells)

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

# temp_H = data_matrix.sel(quantity='t', species='H', filename='grid045.dat').values
# dens_H = data_matrix.sel(quantity='n', species='H', filename='grid045.dat').values
# temp_H2 = data_matrix.sel(quantity='t', species='H_2', filename='grid045.dat').values
# dens_H2 = data_matrix.sel(quantity='n', species='H_2', filename='grid045.dat').values

temp_e = data_matrix.sel(quantity='t', species='e', filename='mike045.dat').values
temp_e1 = data_matrix.sel(quantity='t', species='e', filename='mike14.dat').values
temp_e2 = data_matrix.sel(quantity='t', species='e', filename='mike43.dat').values

# pressure_H = np.multiply(temp_H,dens_H)
# pressure_H2 = np.multiply(temp_H2[0], dens_H2[0])
# pressure = (pressure_H + pressure_H2) * 1.6e-19

z_data = data_grid.sel(axis='z').values

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
cs = ax.tripcolor(triang,temp_e, vmin=0.0, vmax = 3.0, cmap=plt.get_cmap('visit'))
cs2 = ax.tripcolor(triang2,temp_e, vmin=0.0, vmax = 3.0, cmap=plt.get_cmap('visit'))

#ax.set_title(r'Li$^+$ density, $n_{Li^+}$ (m$^{-3}$)', size=22)
ax.set_title(r'Temperature, (eV)', size=22)
ax.tick_params(labelsize=16)
ax.set_ylim(-0.25,0.25)
ax.set_xlim(-1.35,0.4)
ax.tripcolor(triang,temp_e1, vmin=0.0, vmax = 3.0, cmap=plt.get_cmap('visit'))
ax.tripcolor(triang2,temp_e1, vmin=0.0, vmax = 3.0, cmap=plt.get_cmap('visit'))

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
plt.savefig('test.pdf',bbox_inches='tight')
plt.show()

