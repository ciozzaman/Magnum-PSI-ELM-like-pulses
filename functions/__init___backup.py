import numpy as np
from scipy.optimize import curve_fit
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import statistics as s
import csv
import os,sys
from astropy.io import fits
import matplotlib.animation as animation
import pandas
import scipy.stats
import peakutils
import collections
from scipy.ndimage import rotate
from skimage.transform import resize


#
# This is a collection of script that can be usefull in the interpretation of IR camera files
#



# SOMETHING USEFULL FOR PROFILING
#
#
# import cProfile
# import re
# cProfile.run(' SCRIPT OR SINGLE INSTRUCTION THAT I WANT TO PROFILE ')
#
#


def is_number(s):
    """ Returns True is string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False

def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2


#This function will generate a polinomial of n order GOOD FOR WHOLE FRAMES
def polygen(n):
	def polyadd(x, *params):
		params=np.array(params)
		shape=np.shape(x)
		temp=np.zeros(shape)
		# if (len(shape)>1) & (len(np.shape(params))==1):
		# 	params=np.reshape(params,(shape[-2],shape[-1],n))
		for i in range(n):
			# print('np.shape(temp),np.shape(params),np.shape(x)',temp,params,x)
			x2=np.power(x,i)
			# print('params[:][:][i]',params[:][:][i])
			# print('params',params)
			para=np.multiply(params[:,:,i],x2)
			temp=np.add(temp,para)
		return temp
	return polyadd

# #This function will generate a polinomial of n order GOOD FOR background fitting BUT NOT WORKING!! 2018/03/30
# def polygen2(n,shape):
# 	def polyadd(x, *params):
# 		params=np.array(params)
# 		x=np.reshape(x,shape)
# 		temp=np.zeros(shape)
# 		# if (len(shape)>1) & (len(np.shape(params))==1):
# 		params=np.reshape(params,(shape[-2],shape[-1],n))
# 		for i in range(n):
# 			# print('np.shape(temp),np.shape(params),np.shape(x)',temp,params,x)
# 			x2=np.power(x,i)
# 			# print('params[:][:][i]',params[:][:][i])
# 			# print('params',params)
# 			para=np.multiply(params[:,:,i],x2)
# 			temp=np.add(temp,para)
# 		return temp
# 	return polyadd

#This function will generate a polinomial of n order
def polygen3(n):
	def polyadd(x, *params):
		temp=0
		for i in range(n):
			temp+=params[i]*x**i
		return temp
	return polyadd


#############################################################################################

def order_filenames(filenames):

	# 13/05/2018 THIS FUNCTION IS INTODUCED TO FIX A BUG IN THE CASE filenames CONTAINS .npy FILES AND NOT .csv

	extention=filenames[0][-4:]
	if ((extention=='.csv') or (extention=='.CSV')):
		return order_filenames_csv(filenames)
	else:
		return sorted(filenames, key=str.lower)

######################################################################################################

def order_filenames_csv(filenames):

	# section to be sure that the files are ordered in the proper numeric order
	# CAUTION!! THE FOLDER MUST CONTAIN MAX 9999999 FILES!

	reference=[]
	filenamescorr=[]
	for i in range(len(filenames)):
		# print(filenames[i])
		start=0
		end=len(filenames[i])
		for j in range(len(filenames[i])):
			if ((filenames[i][j]=='-') or (filenames[i][j]=='_')): #modified 12/08/2018 from "_" to "-"
				start=j
			elif filenames[i][j]=='.':
				end=j
		index=filenames[i][start+1:end]
		#print('index',index)

		# Commented 21/08/2018
		# if int(index)<10:
		# 	temp=filenames[i][:start+1]+'000000'+filenames[i][end-1:]
		# 	filenamescorr.append(temp)
		# elif int(index)<100:
		# 	temp=filenames[i][:start+1]+'00000'+filenames[i][end-2:]
		# 	filenamescorr.append(temp)
		# 	# print(filenames[i],temp)
		# elif int(index)<1000:
		# 	temp=filenames[i][:start+1]+'0000'+filenames[i][end-3:]
		# 	filenamescorr.append(temp)
		# elif int(index)<10000:
		# 	temp=filenames[i][:start+1]+'000'+filenames[i][end-4:]
		# 	filenamescorr.append(temp)
		# elif int(index)<100000:
		# 	temp=filenames[i][:start+1]+'00'+filenames[i][end-5:]
		# 	filenamescorr.append(temp)
		# elif int(index)<1000000:
		# 	temp=filenames[i][:start+1]+'0'+filenames[i][end-6:]
		# 	filenamescorr.append(temp)
		# else:
		# 	filenamescorr.append(filenames[i])
		reference.append(index)
	reference=np.array(reference)
	filenamesnew=np.array([filenames for _,filenames in sorted(zip(reference,filenames))])

	# Commented 21/08/2018
	# filenamescorr=sorted(filenamescorr, key=str.lower)
	# filenamesnew=[]
	# for i in range(len(filenamescorr)):
	# 	# print(filenamescorr[i])
	# 	for j in range(len(filenamescorr[i])):
	# 		if ((filenames[i][j]=='-') or (filenames[i][j]=='_')): #modified 12/08/2018 from "_" to "-"
	# 			start=j
	# 		elif filenamescorr[i][j]=='.':
	# 			end=j
	# 	index=filenamescorr[i][start+1:end]
	# 	# if int(index)<10:
	# 	# 	temp=filenamescorr[i][:start+1]+str(int(index))+filenamescorr[i][end:]
	# 	# 	filenamesnew.append(temp)
	# 	# elif int(index)<100:
	# 	# 	temp=filenamescorr[i][:start+1]+str(int(index))+filenamescorr[i][end:]
	# 	# 	filenamesnew.append(temp)
	# 	# 	# print(filenames[i],temp)
	# 	# if int(index)<1000000:
	# 	temp=filenamescorr[i][:start+1]+str(int(index))+filenamescorr[i][end:]
	# 	filenamesnew.append(temp)
	# 	# 	# print(filenamescorr[i],temp)
	# 	# else:
	# 	# 	filenamesnew.append(filenamescorr[i])
	filenames=filenamesnew

	return(filenames)

#############################################################################################


def collect_subfolder(extpath):

	# THIS FUNCTION GENERATE THE .npy FILE CORRESPONDING TO A SINGLE FOLDER STARTING FROM ALL THE .csv FILES IN IT

	##print('sys.argv[0] =', sys.argv[0])
	#pathname = os.path.dirname(sys.argv[0])
	##print('path =', pathname)
	#print('full path =', os.path.abspath(pathname))
	#path=os.path.abspath(pathname)


	# path=os.getcwd()
	path=extpath
	print('path =', path)

	position=[]
	for i in range(len(path)):
		if path[i]=='/':
			position.append(i)
	position=max(position)
	lastpath=path[position+1:]
	# print('lastpath',lastpath)

	f = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		f.extend(filenames)
	#	break

	filenames=f
	# print('filenames',filenames)


	temp=[]
	print('len(filenames)',len(filenames))
	for index in range(len(filenames)):
		# print(filenames[index])
		if filenames[index][-3:]=='csv':
			temp.append(filenames[index])
			# filenames=np.delete(filenames,index)
			# print('suca')
	filenames=temp

	filenames=order_filenames(filenames)

	numfiles=len(filenames)


	filename=filenames[0]

	# print('os.path.join(path,filename)',os.path.join(path,filename))

	firstrow=-1
	with open(os.path.join(path,filename),'r') as csvfile:
		reader = csv.reader(csvfile)
		# print('reader',reader)
		pointer=0
		for row in reader:
	#		print('shape',np.shape(row))
			# print('row',row)
			if not row:
				temp='empty'
			else:
				temp=row[0]
			if (is_number(temp)) & (firstrow==-1):
				firstrow=pointer
				rowlen=np.shape(row)[0]
				pointer+=1
			elif is_number(temp) & firstrow>-1:
				pointer+=1
			else:
				ponter=0
	lastrow=pointer
	sizey=lastrow-firstrow
	sizex=rowlen

	data=np.zeros((1,numfiles,sizey,sizex))

	print('firstrow,lastrow,sizey,sizex',firstrow,lastrow,sizey,sizex)

	file=0
	for filename in filenames:
		with open(os.path.join(path,filename),'r') as csvfile:
			reader = csv.reader(csvfile)
			# print('reader',reader)
			tempdata=[]
			pointer=sizey-1
			for row in reader:
		#		print('shape',np.shape(row))
				# print('row',row)
				if not row:
					temp='empty'
				else:
					temp=row[0]
				if is_number(temp):
					for k in range(len(row)):
						# print('j,i,pointer,k',j,i,pointer,k)
						data[0,file,pointer,k]=(float(row[k]))
						# print('float(k)',float(row[k]))
					pointer-=1
					# print('row',row)
				else:
					ponter=0
			file+=1

	np.save(os.path.join(extpath,lastpath),data)

	# plt.pcolor(data[0,20,:,:])
	# plt.show()


##############################################################################################

def collect_subfolderfits(extpath,start='auto',stop='auto'):

	# 2018/01/16 Upgraded to handle, as optional, a limited range of frames

	##print('sys.argv[0] =', sys.argv[0])
	#pathname = os.path.dirname(sys.argv[0])
	##print('path =', pathname)
	#print('full path =', os.path.abspath(pathname))
	#path=os.path.abspath(pathname)


	# path=os.getcwd()
	path=extpath
	print('path =', path)

	position=[]
	for i in range(len(path)):
		if path[i]=='/':
			position.append(i)
	position=max(position)
	lastpath=path[position+1:]
	# print('lastpath',lastpath)

	f = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		f.extend(filenames)
	#	break

	filenames=f
	# print('filenames',filenames)

	filefits=[]
	temp=[]
	print('len(filenames)',len(filenames))
	for index in range(len(filenames)):
		# print(filenames[index])
		if filenames[index][-3:]=='csv':
			temp.append(filenames[index])
			# filenames=np.delete(filenames,index)
			# print('suca')
		elif filenames[index][-3:]=='fts':
			filefits.append(filenames[index])
	filenames=temp
	filenames=sorted(filenames, key=str.lower)

	numfiles=len(filenames)


	filename=filenames[0]

	# print('os.path.join(path,filename)',os.path.join(path,filename))

	firstrow=-1
	with open(os.path.join(path,filename),'r') as csvfile:
		reader = csv.reader(csvfile)
		# print('reader',reader)
		pointer=0
		for row in reader:
	#		print('shape',np.shape(row))
			# print('row',row)
			if not row:
				temp='empty'
			else:
				temp=row[0]
			if (is_number(temp)) & (firstrow==-1):
				firstrow=pointer
				rowlen=np.shape(row)[0]
				pointer+=1
			elif is_number(temp) & firstrow>-1:
				pointer+=1
			else:
				ponter=0
	lastrow=pointer
	sizey=lastrow-firstrow
	sizex=rowlen


	datafit=fits.open(os.path.join(path,filefits[0]))
	datafit=datafit[0].data

	datafit=datafit+32768
	lenfits=len(datafit)

	data=np.zeros((1,lenfits,sizey,sizex))
	datatest=np.zeros((1,numfiles,sizey,sizex))
	index=0
	for frame in datafit:
		# plt.pcolor(frame)
		# plt.show()
		frame=np.flip(frame,0)
		# plt.pcolor(frame)
		# plt.show()
		data[0,index,:,:]=frame
		index+=1


	print('firstrow,lastrow,sizey,sizex',firstrow,lastrow,sizey,sizex)

	file=0
	for filename in filenames:
		with open(os.path.join(path,filename),'r') as csvfile:
			reader = csv.reader(csvfile)
			# print('reader',reader)
			tempdata=[]
			pointer=sizey-1
			for row in reader:
		#		print('shape',np.shape(row))
				# print('row',row)
				if not row:
					temp='empty'
				else:
					temp=row[0]
				if is_number(temp):
					for k in range(len(row)):
						# print('j,i,pointer,k',j,i,pointer,k)
						datatest[0,file,pointer,k]=(float(row[k]))
						# print('float(k)',float(row[k]))
					pointer-=1
					# print('row',row)
				else:
					ponter=0
			file+=1


	# 2018/01/16 Section added to avoid the need of a separate "collect_subfolderfits_limited" function
	if start=='auto':
		force_start=0
	elif (start<0 or start>lenfits):
		print('The initial limit is out of range (a number of frame)')
		print('0 will be used instead of '+str(start))
		force_start=0
	else:
		force_start=start
	if stop=='auto':
		force_end=lenfits
	elif (stop<0 or stop>lenfits or stop<=force_start):
		print('The final limit to search for the oscillation ad erase it is out of range (a number of frame)')
		print(str(lenfits)+'s will be used instead of '+str(stop))
		force_end=lenfits
	else:
		force_end=stop

	# if start<0:
	# 	start=0
	# if start>lenfits:
	# 	start=lenfits
	# if stop>lenfits:
	# 	stop=lenfits
	# if stop<start:
	# 	print('there must be something wrong, you are giving start frame higher than stop one')
	# 	exit()
	datacropped=data[:,force_start:force_end,:,:]


	# if (not (np.array_equal(datatest[0,0],data[0,0]))):
	# 	print('there must be something wrong, the first frame of the FITS file do not match with the first csv files')
	# 	exit()
	if (not (np.array_equal(datatest[0,-1],data[0,-1]))):
		print('there must be something wrong, the last frame of the FITS file do not match with the last csv files')
		# exit()
	else:
		np.save(os.path.join(extpath,lastpath),datacropped)

	# plt.pcolor(data[0,20,:,:])
	# plt.show()

##############################################################################################

def collect_subfolderfits_limited(extpath,start,stop):

	# 10/05/2018 THIS FUNCTION IS EXACLT LIKE collect_subfolderfits BUT LIMIT THE FILE CREATED FROM THE start FRAME TO THE  finish ONE
	print('Use collect_subfolderfits instead of collect_subfolderfits_limited')
	exit()


	##print('sys.argv[0] =', sys.argv[0])
	#pathname = os.path.dirname(sys.argv[0])
	##print('path =', pathname)
	#print('full path =', os.path.abspath(pathname))
	#path=os.path.abspath(pathname)


	# path=os.getcwd()
	path=extpath
	print('path =', path)

	position=[]
	for i in range(len(path)):
		if path[i]=='/':
			position.append(i)
	position=max(position)
	lastpath=path[position+1:]
	# print('lastpath',lastpath)

	f = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		f.extend(filenames)
	#	break

	filenames=f
	# print('filenames',filenames)

	filefits=[]
	temp=[]
	print('len(filenames)',len(filenames))
	for index in range(len(filenames)):
		# print(filenames[index])
		if filenames[index][-3:]=='csv':
			temp.append(filenames[index])
			# filenames=np.delete(filenames,index)
			# print('suca')
		elif filenames[index][-3:]=='fts':
			filefits.append(filenames[index])
	filenames=temp
	filenames=sorted(filenames, key=str.lower)

	numfiles=len(filenames)


	filename=filenames[0]

	# print('os.path.join(path,filename)',os.path.join(path,filename))

	firstrow=-1
	with open(os.path.join(path,filename),'r') as csvfile:
		reader = csv.reader(csvfile)
		# print('reader',reader)
		pointer=0
		for row in reader:
	#		print('shape',np.shape(row))
			# print('row',row)
			if not row:
				temp='empty'
			else:
				temp=row[0]
			if (is_number(temp)) & (firstrow==-1):
				firstrow=pointer
				rowlen=np.shape(row)[0]
				pointer+=1
			elif is_number(temp) & firstrow>-1:
				pointer+=1
			else:
				ponter=0
	lastrow=pointer
	sizey=lastrow-firstrow
	sizex=rowlen


	datafit=fits.open(os.path.join(path,filefits[0]))
	datafit=datafit[0].data

	datafit=datafit+32767
	lenfits=len(datafit)

	data=np.zeros((1,lenfits,sizey,sizex))
	datatest=np.zeros((1,numfiles,sizey,sizex))
	index=0
	for frame in datafit:
		# plt.pcolor(frame)
		# plt.show()
		frame=np.flip(frame,0)
		# plt.pcolor(frame)
		# plt.show()
		data[0,index,:,:]=frame
		index+=1


	print('firstrow,lastrow,sizey,sizex',firstrow,lastrow,sizey,sizex)

	file=0
	for filename in filenames:
		with open(os.path.join(path,filename),'r') as csvfile:
			reader = csv.reader(csvfile)
			# print('reader',reader)
			tempdata=[]
			pointer=sizey-1
			for row in reader:
		#		print('shape',np.shape(row))
				# print('row',row)
				if not row:
					temp='empty'
				else:
					temp=row[0]
				if is_number(temp):
					for k in range(len(row)):
						# print('j,i,pointer,k',j,i,pointer,k)
						datatest[0,file,pointer,k]=(float(row[k]))
						# print('float(k)',float(row[k]))
					pointer-=1
					# print('row',row)
				else:
					ponter=0
			file+=1

	if (np.array_equal(datatest[0,0],data[0,0]))&(np.array_equal(datatest[0,-1],data[0,-1])):
		print('there must be something wrong, the first or last frame of the TITS file do not match with the two csv files')
		exit()


	if start<0:
		start=0
	if start>lenfits:
		start=lenfits
	if stop>lenfits:
		stop=lenfits
	if stop<start:
		print('there must be something wrong, you are giving start frame higher than stop one')
		exit()
	datacropped=data[:,start:stop,:,:]

	np.save(os.path.join(extpath,lastpath),datacropped)

	# plt.pcolor(data[0,20,:,:])
	# plt.show()

##############################################################################################

# make movie

def movie(extpath,framerate,integration,xlabel=(),ylabel=(),barlabel=(),cmap='rainbow',timesteps='auto',extvmin='auto',extvmax='auto'):

	path=extpath
	print('path =', path)

	position=[]
	for i in range(len(path)):
		if path[i]=='/':
			position.append(i)
	position=max(position)
	lastpath=path[position+1:]
	# print('lastpath',lastpath)

	f = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		f.extend(filenames)
	#	break

	filenames=f
	# print('filenames',filenames)

	filefits=[]
	temp=[]
	filemovie=[]
	print('len(filenames)',len(filenames))
	for index in range(len(filenames)):
		# print(filenames[index])
		if filenames[index][-3:]=='npy':
			temp.append(filenames[index])

	filenames=temp
	filenames=sorted(filenames, key=str.lower)

	numfiles=len(filenames)


	filename=filenames[0]

	data=np.load(os.path.join(path,filename))

	fig = plt.figure()
	ax = fig.add_subplot(111)

	# I like to position my colorbars this way, but you don't have to
	# div = make_axes_locatable(ax)
	# cax = div.append_axes('right', '5%', '5%')

	# def f(x, y):
	#     return np.exp(x) + np.sin(y)

	# x = np.linspace(0, 1, 120)
	# y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

	# This is now a list of arrays rather than a list of artists
	frames = [None]*len(data[0])
	frames[0]=data[0,0]

	for i in range(len(data[0])):
	    # x       += 1
	    # curVals  = f(x, y)
	    frames[i]=(data[0,i])

	cv0 = frames[0]
	im = ax.imshow(cv0,cmap, origin='lower') # Here make an AxesImage rather than contour
	cb = fig.colorbar(im).set_label(barlabel)
	cb = ax.set_xlabel(xlabel)
	cb = ax.set_ylabel(ylabel)
	tx = ax.set_title('Frame 0')



	if timesteps=='auto':
		def animate(i):
			arr = frames[i]
			if extvmax=='auto':
				vmax = np.max(arr)
			elif extvmax=='allmax':
				vmax = np.max(data)
			else:
				vmax = extvmax

			if extvmin=='auto':
				vmin = np.min(arr)
			elif extvmin=='allmin':
				vmin = np.min(data)
			else:
				vmin = extvmin

			im.set_data(arr)
			im.set_clim(vmin, vmax)
			tx.set_text('Frame {0}'.format(i)+', FR '+str(framerate)+'Hz, t '+str(np.around(0+i/framerate,decimals=3))+'s int '+str(integration)+'ms')
			# In this version you don't have to do anything to the colorbar,
			# it updates itself when the mappable it watches (im) changes
	else:
		def animate(i):
			arr = frames[i]
			if extvmax=='auto':
				vmax = np.max(arr)
			elif extvmax=='allmax':
				vmax = np.max(data)
			else:
				vmax = extvmax

			if extvmin=='auto':
				vmin = np.min(arr)
			elif extvmin=='allmin':
				vmin = np.min(data)
			else:
				vmin = extvmin

			im.set_data(arr)
			im.set_clim(vmin, vmax)
			tx.set_text('Frame {0}'.format(i)+', t '+str(timesteps[i])+'s int '+str(integration)+'ms')

	ani = animation.FuncAnimation(fig, animate, frames=len(data[0]))

	ani.save(os.path.join(extpath,lastpath)+'.mp4', fps=30, extra_args=['-vcodec', 'libx264'])


######################################################################################################

def movie_from_data(data,framerate,integration=1,xlabel=(),ylabel=(),barlabel=(),cmap='rainbow',timesteps='auto',extvmin='auto',extvmax='auto'):

	fig = plt.figure()
	ax = fig.add_subplot(111)

	# I like to position my colorbars this way, but you don't have to
	# div = make_axes_locatable(ax)
	# cax = div.append_axes('right', '5%', '5%')

	# def f(x, y):
	#     return np.exp(x) + np.sin(y)

	# x = np.linspace(0, 1, 120)
	# y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

	# This is now a list of arrays rather than a list of artists
	frames = [None]*len(data[0])
	frames[0]=data[0,0]

	for i in range(len(data[0])):
	    # x       += 1
	    # curVals  = f(x, y)
	    frames[i]=(data[0,i])

	cv0 = frames[0]
	im = ax.imshow(cv0,cmap, origin='lower') # Here make an AxesImage rather than contour


	cb = fig.colorbar(im).set_label(barlabel)
	cb = ax.set_xlabel(xlabel)
	cb = ax.set_ylabel(ylabel)
	tx = ax.set_title('Frame 0')


	if timesteps=='auto':
		def animate(i):
			arr = frames[i]
			if extvmax=='auto':
				vmax = np.max(arr)
			elif extvmax=='allmax':
				vmax = np.max(data)
			else:
				vmax = extvmax

			if extvmin=='auto':
				vmin = np.min(arr)
			elif extvmin=='allmin':
				vmin = np.min(data)
			else:
				vmin = extvmin
			im.set_data(arr)
			im.set_clim(vmin, vmax)
			tx.set_text('Frame {0}'.format(i)+', FR '+str(framerate)+'Hz, t '+str(np.around(0+i/framerate,decimals=3))+'s int '+str(integration)+'ms')

		    # In this version you don't have to do anything to the colorbar,
		    # it updates itself when the mappable it watches (im) changes
	else:
		def animate(i):
			arr = frames[i]
			if extvmax=='auto':
				vmax = np.max(arr)
			elif extvmax=='allmax':
				vmax = np.max(data)
			else:
				vmax = extvmax

			if extvmin=='auto':
				vmin = np.min(arr)
			elif extvmin=='allmin':
				vmin = np.min(data)
			else:
				vmin = extvmin
			im.set_data(arr)
			im.set_clim(vmin, vmax)
			tx.set_text('Frame {0}'.format(i)+', t '+str(timesteps[i])+'s int '+str(integration)+'ms')

	ani = animation.FuncAnimation(fig, animate, frames=len(data[0]))

	return ani


######################################################################################################


def collect_stat(extpath):

	##print('sys.argv[0] =', sys.argv[0])
	#pathname = os.path.dirname(sys.argv[0])
	##print('path =', pathname)
	#print('full path =', os.path.abspath(pathname))
	#path=os.path.abspath(pathname)


	# path=os.getcwd()
	path=extpath
	print('path =', path)


	filenames=all_file_names(path,'stat.npy')
	#
	# position=[]
	# for i in range(len(path)):
	# 	if path[i]=='/':
	# 		position.append(i)
	# position=max(position)
	# lastpath=path[position+1:]
	# # print('lastpath',lastpath)
	#
	# f = []
	# for (dirpath, dirnames, filenames) in os.walk(path):
	# 	f.extend(filenames)
	# #	break
	#
	# filenames=f
	# # print('filenames',filenames)
	#
	# filefits=[]
	# temp=[]
	# print('len(filenames)',len(filenames))
	# for index in range(len(filenames)):
	# 	# print(filenames[index])
	# 	if filenames[index][-8:]=='stat.npy':
	# 		temp.append(filenames[index])
	# 		# filenames=np.delete(filenames,index)
	# 		# print('suca')
	# filenames=temp
	# filenames=sorted(filenames, key=str.lower)
	#
	# numfiles=len(filenames)


	filename=filenames[0]

	data=np.load(os.path.join(extpath,filename))

	return data

########################################################################################################


def evaluate_back(extpath):

	# THIS FUNCTION PREPARE THE FILES THAT WILL BE USED FOR GETTING THE COUNTS TO TEMPERATURE CALIBRATION
	#
	# UPDATE 10/05/2018 I FOUND AN ERROR AND CHANGED datastat[0]=np.mean(data,axis=(-1,-2))  TO   datastat[0]=np.mean(data,axis=(0,1))

	##print('sys.argv[0] =', sys.argv[0])
	#pathname = os.path.dirname(sys.argv[0])
	##print('path =', pathname)
	#print('full path =', os.path.abspath(pathname))
	#path=os.path.abspath(pathname)

	path=extpath
	# path=os.getcwd()
	print('path =', path)

	position=[]
	for i in range(len(path)):
		if path[i]=='/':
			position.append(i)
	position=max(position)
	lastpath=path[position+1:]
	print('lastpath',lastpath)

	f = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		f.extend(filenames)
	#	break

	filenames=sorted(filenames, key=str.lower)
	temp=[]
	# print('len(filenames)',len(filenames))
	for index in range(len(filenames)):
		# print(filenames[index])
		if filenames[index][-3:]=='npy':
			temp.append(filenames[index])
			# filenames=np.delete(filenames,index)
			# print('suca')

	filenames=temp

	#filenames=np.delete(filenames,0)
	numfiles=len(filenames)


	print('filenames',filenames)


	data=np.load(os.path.join(path,filenames[0]))
	datashape=np.shape(data)[-2:]

	datastatshape=[2]
	datastatshape.extend(datashape)
	datastat=np.zeros(datastatshape)

	print('np.shape(datastatshape)',np.shape(datastat))

	datastat[0]=np.mean(data,axis=(0,1))
	datastat[1]=np.std(data,axis=(0,1))

	# for i in range(datashape[0]):
	# 	for j in range(datashape[1]):
	# 		datastat[0,i,j]=np.mean(data[0,:,i,j])
	# 		datastat[1,i,j]=np.std(data[0,:,i,j])

	# datastat.extend([np.mean(data[0,:,:,:]),np.std(data[0,:,:,:])])
	# datastat[2]=np.std(data[0,:,:,:])

	# plt.pcolor(datastat[0])
	# plt.colorbar()
	# plt.figure()
	# plt.pcolor(datastat[1])
	# plt.colorbar()
	# plt.show()

	datastat=datastat.tolist()
	datastat.append([np.mean(data[0,:,:,:]),np.std(data[0,:,:,:])])

	np.save(os.path.join(extpath,lastpath+'_stat'),datastat)

######################################################################################################################

def all_file_names(extpath,type):

	# This utility returns a list with all filenames of a defined type in a given folder, orderen alphabetically and in number

	# path=os.getcwd()
	path=extpath
	print('path =', path)

	# position=[]
	# for i in range(len(path)):
	# 	if path[i]=='/':
	# 		position.append(i)
	# position=max(position)
	# lastpath=path[position+1:]
	# print('lastpath',lastpath)

	f = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		f.extend(filenames)
	#	break

	filenames=f
	# print('filenames',filenames)


	temp=[]

	typelen=len(type)
	for index in range(len(filenames)):
		# print(filenames[index])
		if filenames[index][-typelen:]==type:
			temp.append(filenames[index])
			# filenames=np.delete(filenames,index)
			# print('suca')
	filenames=temp

	if len(filenames)==0:
		print('ERROR - there are no files of type '+type+' in path '+path)
		return ()

	if len(filenames)==1:
		print('len(filenames)',len(filenames))
		return filenames

	filenames=order_filenames(filenames)

	print('len(filenames)',len(filenames))
	return filenames

######################################################################################################

def read_csv(extpath,filenames):

	# SLOW VERSION OF READ_CSV WITH csv.reader

	path=extpath
	print('path =', path)

	numfiles=len(filenames)


	filename=filenames[0]

	# print('os.path.join(path,filename)',os.path.join(path,filename))

	firstrow=-1
	with open(os.path.join(path,filename),'r') as csvfile:
		reader = csv.reader(csvfile)
		# print('reader',reader)
		pointer=0
		for row in reader:
	#		print('shape',np.shape(row))
			# print('row',row)
			if not row:
				temp='empty'
			else:
				temp=row[0]
			if (is_number(temp)) & (firstrow==-1):
				firstrow=pointer
				rowlen=np.shape(row)[0]
				pointer+=1
			elif is_number(temp) & firstrow>-1:
				pointer+=1
			else:
				ponter=0
	lastrow=pointer
	sizey=lastrow-firstrow
	sizex=rowlen


	data=np.zeros((1,numfiles,sizey,sizex))

	print('firstrow,lastrow,sizey,sizex',firstrow,lastrow,sizey,sizex)

	file=0
	for filename in filenames:
		with open(os.path.join(path,filename),'r') as csvfile:
			reader = csv.reader(csvfile)
			# print('reader',reader)
			tempdata=[]
			pointer=sizey-1
			for row in reader:
		#		print('shape',np.shape(row))
				# print('row',row)
				if not row:
					temp='empty'
				else:
					temp=row[0]
				if is_number(temp):
					for k in range(len(row)):
						# print('j,i,pointer,k',j,i,pointer,k)
						data[0,file,pointer,k]=(float(row[k]))
						# print('float(k)',float(row[k]))
					pointer-=1
					# print('row',row)
				else:
					ponter=0
			file+=1

	return data

###########################################################################################################

def read_csv2(extpath,filenames):

	# FAST VERSION OF READ_CVS WITH PANDA READER

	path=extpath
	print('path =', path)

	numfiles=len(filenames)


	filename=filenames[0]

	# print('os.path.join(path,filename)',os.path.join(path,filename))

	firstrow=-1
	csvfile=path+'/'+filename
	reader = pandas.read_csv(csvfile,sep='\r')
	reader=reader.values
	# print('reader',reader)
	pointer=0
	for row in reader:
#		print('shape',np.shape(row))
		# print('row',row)
		# if np.fromstring(row[0],sep=',').size>0:
			# print('sti cazzi2')
			# print(row)
		# print((np.fromstring(row[0],sep=',')).size>0)
		if (((np.fromstring(row[0],sep=',')).size>0) & (firstrow==-1)):
			# print((np.fromstring(row[0],sep=',')).size>0)
			# print('pointer,firstrow',pointer,firstrow)
			firstrow=pointer
			rowlen=np.shape(np.fromstring(row[0],sep=','))[0]
			pointer+=1
		# elif ((np.fromstring(row[0],sep=',')).size>0) & (firstrow>-1):
		# 	pointer+=1
		else:
			pointer+=1

	lastrow=pointer
	sizey=lastrow-firstrow
	sizex=rowlen


	data=np.zeros((1,numfiles,sizey,sizex))

	print('firstrow,lastrow,sizey,sizex',firstrow,lastrow,sizey,sizex)

	file=0
	for filename in filenames:
		csvfile=path+'/'+filename
		# print('csvfile',csvfile)
		reader = pandas.read_csv(csvfile,sep='\r')
		reader=reader.values
		reader=np.delete(reader,(np.linspace(0,firstrow-1,firstrow).astype(int)))
		# print('reader',reader)
		tempdata=[]
		pointer=sizey-1
		# pointer=sizey-1
		for row in reader:
			# tempdata.append(row[0])
			# tempdata=np.array(tempdata[firstrow:])
			data[0,file,pointer]=np.fromstring(row,sep=',')
			# print('np.fromstring(row[0],sep=',')',row)
			pointer-=1
		file+=1
	# data=data.astype(float)
	# print('suca')
	return data

###########################################################################################################


def read_fits(extpath,filenames,filefits):

	if len(filenames)>2:
		print('this is wrong! to make this function work toy must provide as second agrument the first and last frame of the movie as .csv')
		exit()

	numfiles=len(filenames)


	filename=filenames[0]

	# print('os.path.join(path,filename)',os.path.join(path,filename))

	firstrow=-1
	with open(os.path.join(path,filename),'r') as csvfile:
		reader = csv.reader(csvfile)
		# print('reader',reader)
		pointer=0
		for row in reader:
	#		print('shape',np.shape(row))
			# print('row',row)
			if not row:
				temp='empty'
			else:
				temp=row[0]
			if (is_number(temp)) & (firstrow==-1):
				firstrow=pointer
				rowlen=np.shape(row)[0]
				pointer+=1
			elif is_number(temp) & firstrow>-1:
				pointer+=1
			else:
				ponter=0
	lastrow=pointer
	sizey=lastrow-firstrow
	sizex=rowlen


	datafit=fits.open(os.path.join(path,filefits[0]))
	datafit=datafit[0].data

	datafit=datafit+32767
	lenfits=len(datafit)

	data=np.zeros((1,lenfits,sizey,sizex))
	datatest=np.zeros((1,numfiles,sizey,sizex))
	index=0
	for frame in datafit:
		# plt.pcolor(frame)
		# plt.show()
		frame=np.flip(frame,0)
		# plt.pcolor(frame)
		# plt.show()
		data[0,index,:,:]=frame
		index+=1


	print('firstrow,lastrow,sizey,sizex',firstrow,lastrow,sizey,sizex)

	file=0
	for filename in filenames:
		with open(os.path.join(path,filename),'r') as csvfile:
			reader = csv.reader(csvfile)
			# print('reader',reader)
			tempdata=[]
			pointer=sizey-1
			for row in reader:
		#		print('shape',np.shape(row))
				# print('row',row)
				if not row:
					temp='empty'
				else:
					temp=row[0]
				if is_number(temp):
					for k in range(len(row)):
						# print('j,i,pointer,k',j,i,pointer,k)
						datatest[0,file,pointer,k]=(float(row[k]))
						# print('float(k)',float(row[k]))
					pointer-=1
					# print('row',row)
				else:
					ponter=0
			file+=1

	if (np.array_equal(datatest[0,0],data[0,0]))&(np.array_equal(datatest[0,-1],data[0,-1])):
		print('there must be something wrong, the first or last frame of the TITS file do not match with the two csv files')
		exit()

	return data

################################################################################################

def count_to_temp_poly(data,params,errparams,errdata=(0,0)):
	print('ERROR: count_to_temp_poly was deleted 2018/04/02 because count_to_temp_poly2 has a better formulation for the error')
	exit()

	polygrade=len(params[0,0])

	shape=np.shape(data)
	datatemp=np.zeros(shape)

	if np.max(errdata)==0:
		errdata=np.zeros(shape)

	errdatatemp=np.zeros(shape)
	numframes=shape[1]
	sizey=shape[-2]
	sizex=shape[-1]
	sumi2=sum(np.power(np.linspace(1,polygrade-1,polygrade-1),2))

	# OLD VERSION WHEN POLYGEN WAS ABLE TO PRECESSO ONLY SCALARS 2018/03/29
	# for i in range(numframes):
	# 	for j in range(sizey):
	# 		index=0
	# 		for count in data[0,i,j]:
	# 			datatemp[0,i,j,index]=polygen(polygrade)(count,*params[j,index])
	# 			errdatatemp[0,i,j,index]=count*np.sqrt(sum([sumi2*errdata[0,i,j,index]**2/count**2]+np.power(np.divide(errparams[j,index],params[j,index]),2)))
	# 			index+=1

	paramerr=np.sum(np.power(np.divide(errparams,params),2),axis=-1)
	index=0
	for frame in data[0]:
		# print('np.shape(frame)',np.shape(frame))
		# print('np.shape(params)',np.shape(params))
		datatemp[0,index]=polygen(polygrade)(frame,*params)
		errdatatemp[0,index]=np.multiply(datatemp[0,index],np.sqrt(np.add(sumi2*np.divide(np.power(errdata[0,index],2),np.power(frame,2)),paramerr)))
		index+=1

	return datatemp,errdatatemp

################################################################################################

def count_to_temp_poly2(data,params,errparams,errdata=(0,0),averaged_params=False):

	polygrade=len(params[0,0])

	shape=np.shape(data)
	datatemp=np.zeros(shape)

	if np.max(errdata)==0:
		errdata=np.zeros(shape)

	errdatatemp=np.zeros(shape)
	numframes=shape[1]
	sizey=shape[-2]
	sizex=shape[-1]
	#sumi2=sum(np.power(np.linspace(1,polygrade-1,polygrade-1),2))

	# paramerr=np.sum(np.power(np.divide(errparams,params),2),axis=-1)

	if averaged_params==True:
		params_mean=np.mean(params,axis=(0,1))
		params_to_use=np.ones(np.shape(params))*params_mean
		errparams_mean=np.mean(errparams,axis=(0,1))
		errparams_to_use=np.ones(np.shape(errparams))*errparams_mean
	else:
		params_to_use=params
		errparams_to_use=errparams

	index=0
	for frame in data[0]:
		# print('np.shape(frame)',np.shape(frame))
		# print('np.shape(params)',np.shape(params))
		datatemp[0,index]=polygen(polygrade)(frame,*params_to_use)
		temp=0
		for i in range(polygrade):
			temp+=np.power(np.multiply(errparams_to_use[:,:,i],np.power(frame,i)),2)
			if i>0:
				temp+=np.power(np.multiply(np.multiply(np.multiply(params_to_use[:,:,i],np.power(frame,i-1)),i),errdata[0,index]),2)
		# errdatatemp[0,index]=np.sqrt(np.add(sumi2*np.divide(np.power(errdata[0,index],2),np.power(frame,2)),paramerr))
		errdatatemp[0,index]=np.sqrt(temp)
		index+=1




	return datatemp,errdatatemp

################################################################################################

def build_poly_coeff(temperature,files,int,path,nmax):
	# modified 2018-10-08 to build the coefficient only for 1 degree of polinomial
	while np.shape(temperature[0])!=():
		temperature=np.concatenate(temperature)
		files=np.concatenate(files)
	meancounttot=[]
	meancountstdtot=[]
	for file in files:
		data=collect_stat(file)
		meancounttot.append(np.array(data[0]))
		meancountstdtot.append(np.array(data[1]))

	meancounttot=np.array(meancounttot)
	meancountstdtot=np.array(meancountstdtot)
	# nmax=5
	shapex=np.shape(meancounttot[0])[0]
	shapey=np.shape(meancounttot[0])[1]
	score=np.zeros((nmax-1,shapex,shapey))

	#for n in range(2,nmax+1):
	n=nmax
	guess=np.ones(n)
	guess,temp2=curve_fit(polygen3(n), meancounttot[:,0,0],temperature, p0=guess, maxfev=100000000)

	coeff=np.zeros((shapex,shapey,n))
	errcoeff=np.zeros((shapex,shapey,n))

	for j in range(shapex):
		for k in range(shapey):
			x=np.array(meancounttot[:,j,k])
			xerr=np.array(meancountstdtot[:,j,k])
			temp1,temp2=curve_fit(polygen3(n),x, temperature, p0=guess, maxfev=100000000)
			yerr=(polygen3(n)((x+xerr),*temp1)-polygen3(n)((x-xerr),*temp1))/2
			temp1,temp2=curve_fit(polygen3(n),x, temperature, p0=temp1, sigma=yerr, maxfev=100000000)
			# yerr=(polygen3(n)((x+xerr),*temp1)-polygen3(n)((x-xerr),*temp1))/2
			guess=temp1
			coeff[j,k,:]=temp1
			errcoeff[j,k,:]=np.sqrt(np.diagonal(temp2))
			score[n-2,j,k]=rsquared(temperature,polygen3(n)(x,*temp1))
	np.save(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)+'ms'),coeff)
	np.save(os.path.join(path,'errcoeffpolydeg'+str(n)+'int'+str(int)+'ms'),errcoeff)

	print('for a polinomial of degree '+str(n-1)+' the R^2 score is '+str(np.sum(score[n-2])))


###############################################################################################################

def build_poly_coeff2(temperature,files,int,path,nmax):
	print('ERROR: builf_poly_coeff2 was deleted 2018/04/02 because curve_fit must be used without the flag absolute_sigma=True on for significant parameters covariance matrix')
	exit()

	while np.shape(temperature[0])!=():
		temperature=np.concatenate(temperature)
		files=np.concatenate(files)
	meancounttot=[]
	meancountstdtot=[]
	for file in files:
		data=collect_stat(file)
		meancounttot.append(np.array(data[0]))
		meancountstdtot.append(np.array(data[1]))

	meancounttot=np.array(meancounttot)
	meancountstdtot=np.array(meancountstdtot)
	# nmax=5
	shapex=np.shape(meancounttot[0])[0]
	shapey=np.shape(meancounttot[0])[1]
	score=np.zeros((nmax-1,shapex,shapey))

	for n in range(2,nmax+1):
		guess=np.ones(n)
		guess,temp2=curve_fit(polygen3(n), meancounttot[:,0,0],temperature, p0=guess, maxfev=100000000)

		coeff=np.zeros((shapex,shapey,n))
		errcoeff=np.zeros((shapex,shapey,n))

		for j in range(shapex):
			for k in range(shapey):
				x=np.array(meancounttot[:,j,k])
				xerr=np.array(meancountstdtot[:,j,k])
				temp1,temp2=curve_fit(polygen3(n),x, temperature, p0=guess,absolute_sigma=True)
				yerr=(polygen3(n)((x+xerr),*temp1)-polygen3(n)((x-xerr),*temp1))/2
				temp1,temp2=curve_fit(polygen3(n),x, temperature, p0=temp1, sigma=yerr, maxfev=100000000,absolute_sigma=True)
				yerr=(polygen3(n)((x+xerr),*temp1)-polygen3(n)((x-xerr),*temp1))/2
				guess=temp1
				coeff[j,k,:]=temp1
				errcoeff[j,k,:]=np.sqrt(np.diagonal(temp2))
				score[n-2,j,k]=rsquared(temperature,polygen3(n)(x,*temp1))
		np.save(os.path.join(path,'coeffpolydeg'+str(n)+'int'+str(int)+'ms'),coeff)
		np.save(os.path.join(path,'errcoeffpolydeg'+str(n)+'int'+str(int)+'ms'),errcoeff)

		print('for a polinomial with '+str(n)+' coefficients the R^2 score is '+str(np.sum(score[n-2])))

################################################################################################

def build_multiple_poly_coeff(temperaturehot,temperaturecold,fileshot,filescold,inttime,framerate,pathparam,nmax):
	# 08/10/2018 THIS CALCULATE FROM MULTIPLE HOT>ROOM AND COLD >ROOM CYCLES THE COEFFICIENTS FOR ALL THE POSSIBLE COMBINATIONS

	for i in range(len(temperaturehot)):
		temperaturehot[i]=flatten_full(temperaturehot[i])
	for i in range(len(temperaturecold)):
		temperaturecold[i]=flatten_full(temperaturecold[i])
	for i in range(len(fileshot)):
		fileshot[i]=flatten_full(fileshot[i])
	for i in range(len(filescold)):
		filescold[i]=flatten_full(filescold[i])

	for i in range(len(temperaturehot)):
		if len(temperaturehot[i])!=len(fileshot[i]):
			print('Error, temperaturehot'+str(i)+' and fileshot'+str(i)+' length is different')
			exit()
	for i in range(len(temperaturecold)):
		if len(temperaturecold[i])!=len(filescold[i]):
			print('Error, temperaturecold'+str(i)+' and filescold'+str(i)+' length is different')
			exit()

	lengthhot=len(temperaturehot)
	lengthcold=len(temperaturecold)

	# This lies must be placed outside of the function to make it work
	# fileshot=np.array([fileshot1,fileshot2])
	# temperaturehot=np.array([temperaturehot1,temperaturehot2])
	# filescold=np.array([filescold1,filescold2])
	# temperaturecold=np.array([temperaturecold1,temperaturecold2])

	# THIS COMPUTE THE 1-1, 1-2, 2-1, 2-2 PARAMETERS
	for i in range(lengthhot):
		for j in range(lengthcold):
			path=pathparam+'/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(nmax)+'/'+str(i+1)+'-'+str(j+1)
			if not os.path.exists(path):
				os.makedirs(path)
			temperature=[temperaturehot[i],temperaturecold[j]]
			files=[fileshot[i],filescold[j]]
			build_poly_coeff(temperature,files,inttime,path,nmax)

################################################################################################

def build_average_poly_coeff(temperaturehot,temperaturecold,fileshot,filescold,inttime,framerate,pathparam,nmax):
	# 08/10/2018 THIS MAKES THE AVERAGE OF COEFFICIENTS FROM MULTIPLE HOT>ROOM AND COLD >ROOM CYCLES THE COEFFICIENTS

	lengthhot=len(temperaturehot)
	lengthcold=len(temperaturecold)

	first=True
	for i in range(lengthhot):
		for j in range(lengthcold):
			path=pathparam+'/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(nmax)+'/'+str(i+1)+'-'+str(j+1)
			params=np.load(os.path.join(path,'coeffpolydeg'+str(nmax)+'int'+str(inttime)+'ms'+'.npy'))
			if first==True:
				shape=np.shape(params)
				shape=np.concatenate(((lengthhot,lengthcold),shape))
				parameters=np.zeros(shape)
				first=False
			parameters[i,j]=params

	meanparameters=np.mean(parameters,axis=(0,1))
	stdparameters=np.std(parameters,axis=(0,1))

	path=pathparam+'/'+str(inttime)+'ms'+str(framerate)+'Hz'+'/'+'numcoeff'+str(nmax)+'/average'
	if not os.path.exists(path):
		os.makedirs(path)
	np.save(os.path.join(path,'coeffpolydeg'+str(nmax)+'int'+str(inttime)+'ms'),meanparameters)
	np.save(os.path.join(path,'errcoeffpolydeg'+str(nmax)+'int'+str(inttime)+'ms'),stdparameters)


#####################################################################################################

def average_frame(frame,pixelmean,extremedelete=False):

	# Does the average over pixelmean x pixelmean pixels of frame
	#
	# If the flag extremedelete is True for every mean the maximuna and minimum values are deleted
	#
	if pixelmean==1:
		return frame
	shapeorig=np.shape(frame)
	shapeaver=(np.divide(shapeorig,pixelmean)).astype(int)
	frameaver=np.zeros(shapeaver)
	for i in range(shapeaver[0]):
		if ((i+1)*pixelmean)>shapeorig[0]:
			indexi=shapeorig[0]-1
		else:
			indexi=(i+1)*pixelmean
		for j in range(shapeaver[1]):
			if ((j+1)*pixelmean)>shapeorig[1]:
				indexj=shapeorig[1]-1
			else:
				indexj=(j+1)*pixelmean
			flat=np.ravel(frame[i*pixelmean:int(indexi),j*pixelmean:int(indexj)])
			if extremedelete:
				if len(flat)>3:
					flat=np.delete(flat,np.argmax(flat))
					flat=np.delete(flat,np.argmin(flat))
			frameaver[i,j]=np.mean(flat)
			# print(i*pixelmean,indexi,j*pixelmean,indexj,flat)

	return frameaver

#####################################################################################################

def average_multiple_frames(frames,pixelmean,timemean=1,extremedelete=False):

	# Does the average over pixelmean x pixelmean pixels of multiple frames
	#
	# If the flag extremedelete is True for every mean the maximuna and minimum values are deleted
	#

	shapeorig=np.shape(frames)
	nframes=shapeorig[1]
	# framesaver=[]
	# framesaver.append([None] * nframes)
	# framesaver=np.array(framesaver)
	framesaver = [None]*nframes
	for i in range(nframes):
		framesaver[i]=average_frame(frames[0,i],pixelmean,extremedelete)

	if timemean>1:
		timemean = int(timemean)
		reduced_frames = int(nframes/timemean)
		# temp = []
		# temp.append([None] * reduced_frames)
		# temp = np.array(temp)
		temp=[None] * reduced_frames
		for index in range(reduced_frames):
			temp[index] = np.mean(framesaver[index*timemean:(index+1)*timemean],axis=(0))
		return np.array([temp])

	return np.array([framesaver])


#####################################################################################################

def average_multiple_frames2(frames,pixelmean,timemean=1):

	# Created to use the faster function skimage.transform.resize
	# Does the average over pixelmean x pixelmean pixels of multiple frames
	#
	import numpy as np
	from skimage.transform import resize

	shapeorig=np.shape(frames)
	nframes=shapeorig[1]
	shapeaver=(np.divide(shapeorig[-2:],pixelmean)).astype(int)
	# framesaver=[]
	# framesaver.append([None] * nframes)
	# framesaver=np.array(framesaver)
	framesaver = [None]*nframes
	for i in range(nframes):
		# framesaver[i]=average_frame(frames[0,i],pixelmean,extremedelete)
		framesaver[i] =resize(frames[0,i], shapeaver,order=1)
	framesaver = np.array(framesaver)

	if timemean>1:
		timemean = int(timemean)
		reduced_frames = int(nframes/timemean)
		# temp = []
		# temp.append([None] * reduced_frames)
		# temp = np.array(temp)
		temp=[None] * reduced_frames
		for index in range(reduced_frames):
			temp[index] = np.mean(framesaver[index*timemean:(index+1)*timemean],axis=(0))
		return np.array([temp])

	return np.array([framesaver])


#####################################################################################################

def flatten(array):

	# 11/06/2018
	# This function flatten any array of one level.
	# If it is already in one level it return is the way it is
	#
	array=np.array(array)
	length=np.shape(array)[0]

	done=0
	for item in array:
		if np.shape(item)!=():
			done+=1
	if done==0:
		return array

	temp=[]
	lengthinside=np.zeros(length)
	for i in range(length):
		temp2=np.array(array[i])
		lengthinside=np.shape(temp2)
		if lengthinside==():
			temp.append(array[i])
		else:
			for j in range(lengthinside[0]):
				temp.append(temp2[j])
	temp=np.array(temp)
	return temp

#####################################################################################################

def flatten_full(array):
	# this function flatten an array fully

	while np.shape(array[0])!=():
		array=flatten(array)

	return array

#####################################################################################################

def ddx(array,dx,axis,otheraxis=(),howcropotheraxis=0):

	# this function makes the central difference derivative on the axis AXIS. it reduces the size of the array in that direction by two pixels
	# at the same time it can reduce the size ao the array in other directions too of any number of pixels.

	if axis==0:
		temp=np.divide(array[2:]-array[:-2],2*dx)
	elif axis==1:
		temp=np.divide(array[:,2:]-array[:,:-2],2*dx)
	elif axis==2:
		temp=np.divide(array[:,:,2:]-array[:,:,:-2],2*dx)
	elif axis==3:
		temp=np.divide(array[:,:,:,2:]-array[:,:,:,:-2],2*dx)
	elif axis==4:
		temp=np.divide(array[:,:,:,:,2:]-array[:,:,:,:,:-2],2*dx)
	elif axis==5:
		temp=np.divide(array[:,:,:,:,:,2:]-array[:,:,:,:,:,:-2],2*dx)
	elif axis==6:
		temp=np.divide(array[:,:,:,:,:,:,2:]-array[:,:,:,:,:,:,:-2],2*dx)


	if howcropotheraxis==0:
		return temp

	if otheraxis==():
		print('if you specify the number of pixels to crop tou must specify too the axis where to do that')
		exit()

	numaxis=len(np.shape(array))
	for i in range(len(otheraxis)):
		if otheraxis[i]<0:
			otheraxis[i]=numaxis-otheraxis[i]

	if howcropotheraxis/2-howcropotheraxis//2!=0:
		print('the amount of pixels you want to crop the array must be even, to crop of half on one side and half on the other')
		exit()
	htc=howcropotheraxis//2
	if 0 in otheraxis:
		temp=temp[htc:-htc]
	if 1 in otheraxis:
		temp=temp[:,htc:-htc]
	if 2 in otheraxis:
		temp=temp[:,:,htc:-htc]
	if 3 in otheraxis:
		temp=temp[:,:,:,htc:-htc]
	if 4 in otheraxis:
		temp=temp[:,:,:,:,htc:-htc]
	if 5 in otheraxis:
		temp=temp[:,:,:,:,:,htc:-htc]
	if 6 in otheraxis:
		temp=temp[:,:,:,:,:,:,htc:-htc]

	return temp

#####################################################################################################

def d2dx2(array,dx,axis,otheraxis=(),howcropotheraxis=0):

	# this function makes the tentral difference second derivative on the axis AXIS. it reduces the size of the array in that direction by two pixels
	# at the same time it can reduce the size ao the array in other directions too of any number of pixels.

	if axis==0:
		temp=np.divide(array[2:]-np.multiply(2,array[1:-1])+array[:-2],dx**2)
	elif axis==1:
		temp=np.divide(array[:,2:]-np.multiply(2,array[:,1:-1])+array[:,:-2],dx**2)
	elif axis==2:
		temp=np.divide(array[:,:,2:]-np.multiply(2,array[:,:,1:-1])+array[:,:,:-2],dx**2)
	elif axis==3:
		temp=np.divide(array[:,:,:,2:]-np.multiply(2,array[:,:,:,1:-1])+array[:,:,:,:-2],dx**2)
	elif axis==4:
		temp=np.divide(array[:,:,:,:,2:]-np.multiply(2,array[:,:,:,:,1:-1])+array[:,:,:,:,:-2],dx**2)
	elif axis==5:
		temp=np.divide(array[:,:,:,:,:,2:]-np.multiply(2,array[:,:,:,:,:,1:-1])+array[:,:,:,:,:,:-2],dx**2)
	elif axis==6:
		temp=np.divide(array[:,:,:,:,:,:,2:]-np.multiply(2,array[:,:,:,:,:,:,1:-1])+array[:,:,:,:,:,:,:-2],dx**2)


	if howcropotheraxis==0:
		return temp
	if howcropotheraxis/2-howcropotheraxis//2!=0:
		print('the amount of pixels you want to crop the array must be even, to crop of half on one side and half on the other')
		exit()

	numaxis=len(np.shape(array))
	for i in range(len(otheraxis)):
		if otheraxis[i]<0:
			otheraxis[i]=numaxis-otheraxis[i]

	if howcropotheraxis/2-howcropotheraxis//2!=0:
		print('the amount of pixels you want to crop the array must be even, to crop of half on one side and half on the other')
		exit()
	htc=howcropotheraxis//2
	if 0 in otheraxis:
		temp=temp[htc:-htc]
	if 1 in otheraxis:
		temp=temp[:,htc:-htc]
	if 2 in otheraxis:
		temp=temp[:,:,htc:-htc]
	if 3 in otheraxis:
		temp=temp[:,:,:,htc:-htc]
	if 4 in otheraxis:
		temp=temp[:,:,:,:,htc:-htc]
	if 5 in otheraxis:
		temp=temp[:,:,:,:,:,htc:-htc]
	if 6 in otheraxis:
		temp=temp[:,:,:,:,:,:,htc:-htc]

	return temp


#####################################################################################################

def save_timestamp(extpath):

	# 09/08/2018 This function lools at the CSV files and saves the timestamp of the firs and last one in a _timestamp.npy file

	# path=os.getcwd()

	path=extpath
	print('path =', path)

	position=[]
	for i in range(len(path)):
		if path[i]=='/':
			position.append(i)
	position=max(position)
	lastpath=path[position+1:]
	# print('lastpath',lastpath)

	f = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		f.extend(filenames)
	#	break

	filenames=f
	# print('filenames',filenames)

	filefits=[]
	temp=[]
	print('len(filenames)',len(filenames))
	for index in range(len(filenames)):
		# print(filenames[index])
		if filenames[index][-3:]=='csv':
			temp.append(filenames[index])
			# filenames=np.delete(filenames,index)
			# print('suca')
		# elif filenames[index][-3:]=='fts':
		# 	filefits.append(filenames[index])
	filenames=temp
	filenames=sorted(filenames, key=str.lower)

	timestamp=[]
	# file=0
	for filename in [filenames[0],filenames[-1]]:
		with open(os.path.join(path,filename),'r') as csvfile:
			reader = csv.reader(csvfile)
			# print('reader',reader)
			for row in reader:
				if not not row:
				# else:
					if row[0][0:4]=='Time':
						time=row[0][7:]
						ampm=int(time[0:3])
						hh=int(time[4:6])
						mm=int(time[7:9])
						ss=float(time[10:])
						timess=ss+60*(mm+60*(hh+ampm*12))
						timestamp.append(timess)



	timestamp.append(np.mean(timestamp))
	timestamp=np.array(timestamp)
	np.save(os.path.join(extpath,lastpath+'_timestamp'),timestamp)


####################################################################################################

def search_background_timestamp(extpath,ref_directory):

	# 09/08/2018 This function lools at the timestamp in "extpath" and compare it with all the timestamps in the directories indicated by "ref_directory"
	# then I pick the two that are closer and interpolate in between the two to get the proper background


	ref_directories=[]
	for (dirpath, dirnames, filenames) in os.walk(ref_directory):
		ref_directories.extend(dirnames)


	ref_directories=order_filenames(ref_directories)

	type="_timestamp.npy"
	time=[]
	for directory in ref_directories:
		filename=all_file_names(os.path.join(ref_directory,directory),type)[0]
		timestamp=np.load(os.path.join(ref_directory,directory,filename))
		time.append(timestamp[2])

	time=np.array(time)
	if np.sum(time!=np.sort(time))>0:
		print('Something is wrong, the order of the files does not follows a cronological order, as it should')
		exit()


	filename=all_file_names(os.path.join(extpath),type)[0]
	specific_time=np.load(os.path.join(extpath,filename))[2]
	specific_filename=filename[:-14]

	index=np.searchsorted(time,specific_time)-1

	print('found '+str(specific_time)+' between '+str(time[index])+' and '+str(time[index+1]))

	type="_stat.npy"
	filename=all_file_names(os.path.join(ref_directory,ref_directories[index]),type)[0]
	pre_ref=np.load(os.path.join(ref_directory,ref_directories[index],filename))[0]

	filename=all_file_names(os.path.join(ref_directory,ref_directories[index+1]),type)[0]
	post_ref=np.load(os.path.join(ref_directory,ref_directories[index+1],filename))[0]

	dt=time[index+1]-time[index]
	reference=np.multiply(pre_ref,(time[index+1]-specific_time)/dt)+np.multiply(post_ref,(specific_time-time[index])/dt)

	np.save(os.path.join(extpath,specific_filename+'_reference'),reference)


####################################################################################################

def find_nearest_index(array,value):

	# 14/08/2018 This function returns the index of the closer value to "value" inside an array

	array_shape=np.shape(array)

	index = np.abs(np.add(array,-value)).argmin()
	residual_index=index
	cycle=1
	done=0
	position_min=np.zeros(len(array_shape),dtype=int)
	while done!=1:
		length=array_shape[-cycle]
		if residual_index<length:
			position_min[-cycle]=residual_index
			done=1
		else:
			position_min[-cycle]=round(((residual_index/length) %1) *length +0.000000000000001)
			residual_index=residual_index//length
			cycle+=1

	return position_min[0]


###################################################################################################

def track_change_from_baseframe(data,poscentred,basecounts):

	# Created 27/12/2018 to clean up some legacy data. Now this function is not necessay anymore
	# This function evaluate the average counts on a limited number of locations [poscentered] and in a small window around them [a].
	# Then this is compared with same locations and window for a reference frame, and the whole record will need be scaled of the difference [base_counts_correction]


	data_mean_difference = []
	data_mean_difference_std = []

	for pos in poscentred:
		for a in [5, 10]:
			temp1 = np.mean(data[0, :, pos[1] - a:pos[1] + 1 + a, pos[0] - a:pos[0] + 1 + a], axis=(-1, -2, -3))
			temp1std = np.std(np.mean(data[0, :, pos[1] - a:pos[1] + 1 + a, pos[0] - a:pos[0] + 1 + a], axis=(-1, -2)))
			temp2 = np.mean(basecounts[0, :, pos[1] - a:pos[1] + 1 + a, pos[0] - a:pos[0] + 1 + a], axis=(-1, -2, -3))
			temp2std = np.std(
				np.mean(basecounts[0, :, pos[1] - a:pos[1] + 1 + a, pos[0] - a:pos[0] + 1 + a], axis=(-1, -2)))
			data_mean_difference.append(temp1 - temp2)
			data_mean_difference_std.append(temp1std + temp2std)
	data_mean_difference = np.array(data_mean_difference)
	data_mean_difference_std = np.array(data_mean_difference_std)
	guess = [1]
	base_counts_correction, temp2 = curve_fit(costant,
											  np.linspace(1, len(data_mean_difference), len(data_mean_difference)),
											  data_mean_difference, sigma=data_mean_difference_std, p0=guess,
											  maxfev=100000000)
	print('background correction = ' + str(int(base_counts_correction * 1000) / 1000))

	return base_counts_correction







###################################################################################################

def clear_oscillation_central(data,framerate,oscillation_search_window_begin='auto',oscillation_search_window_end='auto',plot_conparison=False):

	# Created 01/11/2018
	# This function take the raw counts. analyse the fast fourier transform in a selectable interval IN THE CENTER OF THE FRAME.
	# Then search for the peak between 20Hz and 34Hz  and substract the oscillation found to the counts
	print('Use clear_oscillation_central2 instead of clear_oscillation_central')
	exit()


	print('shape of data array is '+str(np.shape(data))+', it should be (x,frames,v pixel,h pixel)')

	data=data[0]
	if oscillation_search_window_begin=='auto':
		force_start=0
	elif (oscillation_search_window_begin<0 or oscillation_search_window_begin>len(data)):
		print('The initial limit to search for the oscillation ad erase it is out of range (a time in seconds)')
		print('0s will be used instead')
		force_start=0
	else:
		force_start=int(oscillation_search_window_begin/framerate)

	if oscillation_search_window_end=='auto':
		force_end=len(data)
	elif (oscillation_search_window_end<0 or oscillation_search_window_end>(len(data)*framerate) or oscillation_search_window_end<=(force_start*framerate)):
		print('The final limit to search for the oscillation ad erase it is out of range (a time in seconds)')
		print(str(int(len(data)//(2*framerate)))+'s will be used instead')
		force_end=int(len(data)//(2*framerate))
	else:
		force_end=int(oscillation_search_window_end*framerate)

	window = 10
	datasection = data

	if plot_conparison==True:
		poscentred = [[15, 80], [80, 80], [70, 200], [160, 133], [250, 200]]

		spectra_orig = np.fft.fft(data, axis=0)
		# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
		magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
		phase = np.angle(spectra_orig)
		freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)

		color = ['m', 'c', 'y', 'b', 'r', 'k', 'g', 'm']
		for i in range(len(poscentred)):
			pos = poscentred[i]
			y = np.mean(magnitude[:, pos[0] - window:pos[0] + window, pos[1] - window:pos[1] + window], axis=(-1, -2))
			# y=magnitude[:,pos[0],pos[1]]
			y = np.array([y for _, y in sorted(zip(freq, y))])
			x = np.sort(freq)
			plt.plot(x, y, color[i], label='original data at the point ' + str(pos))
		# plt.title()

		plt.figure(1)
		plt.title('Amplitued from fast Fourier transform for different groups of ' + str(window * 2) + 'x' + str(
			window * 2) + ' pixels, framerate ' + str(framerate)+'Hz' )
		plt.xlabel('Frequency [Hz]')
		plt.ylabel('Amplitude [au]')
		plt.grid()
		plt.semilogy()
		plt.legend()
	# plt.show()


	sections = 31  # number found with practice, no specific mathematical reasons
	max_time = 5  # seconds of record that I can use to filter the signal. I assume to start from zero
	poscentre = [np.shape(data)[1] // 2, np.shape(data)[2] // 2]
	record_magnitude = []
	record_phase = []
	record_freq = []
	peak_freq_record = []
	peak_value_record = []
	section_frames_record = []

	# I restrict the window over which I search for the oscillation
	datarestricted=data[force_start:force_end]

	if oscillation_search_window_end == 'auto':
		if (len(datarestricted) / framerate) <= 1:
			max_start = int(sections // 2)
		else:

			max_start = min(int(1 + sections / 2), int(
				max_time * framerate / (len(datarestricted) / sections)))  # I can use only a part of the record to filter the signal
	else:
		max_start = int(oscillation_search_window_end * framerate / (len(datarestricted) / sections))

	if (len(datarestricted) / framerate) <= 1:
		min_start = max(1, int(0.2 * framerate / (len(datarestricted) / sections)))


	else:
		extra = 0
		while ((max_start - int(max_start / (5 / 2.5)) + extra) < 7):	# 7 is just a try. this is in a way to have enough fitting to compare
			extra+=1
		min_start = max(1,int(max_start / (5 / 2.5)) - extra)
		# min_start = max(1, int(max_start / (5 / 2.5)) )  # with too little intervals it can interpret noise for signal

	for i in range(min_start, max_start):
		section_frames = (i) * (len(datarestricted) // sections)
		section_frames_record.append(section_frames)
		datasection = datarestricted[0:section_frames, poscentre[0] - window:poscentre[0] + window, poscentre[1] - window:poscentre[1] + window]
		spectra = np.fft.fft(datasection, axis=0)
		magnitude = 2 * np.abs(spectra) / len(spectra)
		record_magnitude.append(magnitude[0:len(magnitude) // 2])
		phase = np.angle(spectra)
		record_phase.append(phase[0:len(magnitude) // 2])
		freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
		record_freq.append(freq[0:len(magnitude) // 2])
		magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
		y = np.array(
			[magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
		x = np.sort(freq)
		if plot_conparison == True:
			plt.figure(2)
			plt.plot(x, y, label='size of the analysed window ' + str(section_frames / framerate))
		index_20 = int(find_nearest_index(x, 20))  # I restric the window over which I do the peak search
		index_34 = int(find_nearest_index(x, 34))
		index_7 = int(find_nearest_index(x, 7))
		index_n7 = int(find_nearest_index(x, -7))
		index_n20 = int(find_nearest_index(x, -20))
		index_n34 = int(find_nearest_index(x, -34))
		index_0 = int(find_nearest_index(x, 0))
		noise = np.mean(np.array(
			y[3:index_n34].tolist() + y[index_n20:index_n7].tolist() + y[index_7:index_20].tolist() + y[
																									  index_34:-3].tolist()),
						axis=(-1))
		temp = peakutils.indexes(y[index_20:index_34], thres=noise + np.abs(magnitude.min()),
								 min_dist=(index_34 - index_20) // 2)
		if len(temp) == 1:
			peak_index = index_20 + int(temp)
			peak_freq_record.append(x[peak_index])
			peak_value = float(y[peak_index])
			peak_value_record.append(peak_value)
	record_magnitude = np.array(record_magnitude)
	record_phase = np.array(record_phase)
	record_freq = np.array(record_freq)
	peak_freq_record = np.array(peak_freq_record)
	peak_value_record = np.array(peak_value_record)
	section_frames_record = np.array(section_frames_record)
	if plot_conparison==True:
		plt.figure(2)
		plt.title('Amplitued from fast Fourier transform averaged in a wondow of ' + str(window) + 'pixels around ' + str(
			poscentre) + ', framerate ' + str(framerate) + 'Hz')
		plt.xlabel('Frequency [Hz]')
		plt.ylabel('Amplitude [au]')
		plt.grid()
		plt.semilogy()
		plt.legend()


	# I find the highest peak and that will be the one I use
	index = int(find_nearest_index(peak_value_record, max(peak_value_record)+1))
	section_frames = section_frames_record[index]
	datasection = datarestricted[0:section_frames]
	spectra = np.fft.fft(datasection, axis=0)
	# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
	magnitude = 2 * np.abs(spectra) / len(spectra)
	phase = np.angle(spectra)
	freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
	freq_to_erase = peak_freq_record[index]
	freq_to_erase_index = int(find_nearest_index(freq, freq_to_erase))
	framenumber = np.linspace(0, len(data) - 1, len(data)) - force_start
	data2 = data - np.multiply(magnitude[freq_to_erase_index], np.cos(np.repeat(np.expand_dims(phase[freq_to_erase_index], axis=0), len(data), axis=0) + np.repeat(np.expand_dims(np.repeat(np.expand_dims(2 * np.pi * freq_to_erase * framenumber / framerate, axis=-1),np.shape(data)[1], axis=-1), axis=-1), np.shape(data)[2], axis=-1)))

	if plot_conparison==True:
		plt.figure(1)
		datasection2 = data2
		spectra = np.fft.fft(datasection2, axis=0)
		# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
		magnitude2 = 2 * np.abs(spectra) / len(spectra)
		phase2 = np.angle(spectra)
		freq = np.fft.fftfreq(len(magnitude2), d=1 / framerate)
		for i in range(len(poscentred)):
			pos = poscentred[i]
			y = np.mean(magnitude2[:, pos[0] - window:pos[0] + window, pos[1] - window:pos[1] + window], axis=(-1, -2))
			# y=magnitude[:,pos[0],pos[1]]
			y = np.array([y for _, y in sorted(zip(freq, y))])
			x = np.sort(freq)
			plt.plot(x, y, color[i] + '--',
					 label='data at the point ' + str(pos) + ', ' + str(freq_to_erase) + 'Hz oscillation substracted')
		# plt.title()


		plt.grid()
		plt.semilogy()
		plt.legend()
		plt.pause(0.0001)




	print('stats of the oscillation removal')
	print('with window of size '+str(section_frames_record[index]/framerate)+'s of '+str(len(data)/framerate)+'s of record')
	print('found oscillation of frequency '+str(freq_to_erase)+'Hz')

	return np.array([data2])


###################################################################################################

def clear_oscillation_central2(data,framerate,oscillation_search_window_begin='auto',oscillation_search_window_end='auto',min_frequency_to_erase=20,max_frequency_to_erase=34,plot_conparison=False):

	# Created 15/02/2019
	# This function take the raw counts. analyse the fast fourier transform in a selectable interval IN THE CENTER OF THE FRAME.
	# Then search for the peak between 20Hz and 34Hz  and substract the oscillation found to the counts
	# The difference from clear_oscillation_central is that instead of splitting the interval of interest in smaller chunks and analyse them I analyse the full window available and shift the first and last point.
	# This aproach seems much more efficient

	print('shape of data array is '+str(np.shape(data))+', it should be (x,frames,v pixel,h pixel)')

	data=data[0]
	if oscillation_search_window_begin=='auto':
		force_start=0
	elif (oscillation_search_window_begin<0 or oscillation_search_window_begin>len(data)):
		print('The initial limit to search for the oscillation ad erase it is out of range (a time in seconds)')
		print('0s will be used instead')
		force_start=0
	else:
		force_start=int(oscillation_search_window_begin/framerate)

	if oscillation_search_window_end=='auto':
		force_end=len(data)
	elif (oscillation_search_window_end<0 or oscillation_search_window_end>(len(data)*framerate) or oscillation_search_window_end<=(force_start*framerate)):
		print('The final limit to search for the oscillation ad erase it is out of range (a time in seconds)')
		print(str(int(len(data)//(2*framerate)))+'s will be used instead')
		force_end=int(len(data)//(2*framerate))
	else:
		force_end=int(oscillation_search_window_end*framerate)


	central_freq_for_search = (max_frequency_to_erase-min_frequency_to_erase)/2+min_frequency_to_erase
	if (framerate<2*central_freq_for_search):
		print('There is a problem. The framerate is too low to try to extract the oscillation')
		print('The minimum framrate for doing it is 2*oscillation frequency to detect. Therefore '+str(np.around(2*central_freq_for_search,decimals=1))+'Hz, in this case.')
		print('See http://www.skillbank.co.uk/SignalConversion/rate.htm')
		exit()

	window = 1	# Previously found that as long as the fft is averaged over at least 4 pixels the peak shape and location does not change
	datasection = data

	if plot_conparison==True:
		poscentred = [[15, 80], [80, 80], [70, 200], [160, 133], [250, 200]]

		# spectra_orig = np.fft.fft(data, axis=0)
		# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
		# magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
		# phase = np.angle(spectra_orig)
		# freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)

		color = ['m', 'c', 'y', 'b', 'r', 'k', 'g', 'm']
		for i in range(len(poscentred)):
			pos = poscentred[i]
			spectra_orig = np.fft.fft(data[:, pos[0] - window:pos[0] + window, pos[1] - window:pos[1] + window], axis=0)
			magnitude = 2 * np.abs(spectra_orig) / len(spectra_orig)
			freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
			y = np.mean(magnitude, axis=(-1, -2))
			# y=magnitude[:,pos[0],pos[1]]
			y = np.array([y for _, y in sorted(zip(freq, y))])
			x = np.sort(freq)
			plt.plot(x, y, color[i], label='original data at the point ' + str(pos))
		# plt.title()

		plt.figure(1)
		plt.title('Amplitued from fast Fourier transform for different groups of ' + str(window * 2) + 'x' + str(
			window * 2) + ' pixels, framerate ' + str(framerate)+'Hz' )
		plt.xlabel('Frequency [Hz]')
		plt.ylabel('Amplitude [au]')
		plt.grid()
		plt.semilogy()
		plt.legend()
	# plt.show()



	frames_for_oscillation = framerate//central_freq_for_search

	number_of_waves = 3
	fft_window_move = int(number_of_waves*frames_for_oscillation)
	step = 1
	if fft_window_move<10:
		number_of_waves=10//frames_for_oscillation	# I want to scan for at least 10 frame shifts
	while int(fft_window_move/step)>80:
		step+=1		#if framerate is too high i will skip some of the shifts to limit the number of Fourier transforms to 100


	# I restrict the window over which I search for the oscillation
	datarestricted = data[force_start:force_end]#
	len_data_restricted = len(datarestricted)
	poscentre = [np.shape(data)[1] // 2, np.shape(data)[2] // 2]

	while fft_window_move>(len_data_restricted/5):
		fft_window_move-=1		# I want that the majority of the data I analyse remains the same

	if oscillation_search_window_end == 'auto':
		if (len(datarestricted) / framerate) <= 1:
			max_start = int(sections // 2)
		else:
			max_start = int(5*framerate)  # I use 5 seconds of record
	else:
		max_start = force_end

	if oscillation_search_window_begin == 'auto':
		min_start = 0
	else:
		min_start = force_start

	section_frames = max_start - min_start-fft_window_move

	record_magnitude = []
	record_phase = []
	record_freq = []
	peak_freq_record = []
	peak_value_record = []

	for i in range(int(fft_window_move/step)):
		shift=i*step
		datasection = datarestricted[min_start:max_start-fft_window_move+shift, poscentre[0] -1 :poscentre[0] + window, poscentre[1] - 1:poscentre[1] + window]
		spectra = np.fft.fft(datasection, axis=0)
		magnitude = 2 * np.abs(spectra) / len(spectra)
		record_magnitude.append(magnitude[0:len(magnitude) // 2])
		phase = np.angle(spectra)
		record_phase.append(phase[0:len(magnitude) // 2])
		freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
		record_freq.append(freq[0:len(magnitude) // 2])
		magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
		y = np.array(
			[magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
		x = np.sort(freq)
		if plot_conparison == True:
			plt.figure(2)
			plt.plot(x, y, label='Applied shift of ' + str(shift))
		index_min_freq = int(find_nearest_index(x, min_frequency_to_erase))  # I restric the window over which I do the peak search
		index_max_freq = int(find_nearest_index(x, max_frequency_to_erase))
		index_7 = int(find_nearest_index(x, 7))
		index_n7 = int(find_nearest_index(x, -7))
		index_min_freq_n = int(find_nearest_index(x, -min_frequency_to_erase))
		index_max_freq_n = int(find_nearest_index(x, -max_frequency_to_erase))
		index_0 = int(find_nearest_index(x, 0))
		# noise = np.mean(np.array(
		# 	y[3:index_max_freq_n].tolist() + y[index_min_freq_n:index_n7].tolist() + y[index_7:index_min_freq].tolist() + y[
		# 																									index_max_freq:-3].tolist()),
		# 				axis=(-1))
		# temp = peakutils.indexes(y[index_min_freq:index_max_freq], thres=noise + np.abs(magnitude.min()),
		# 						 min_dist=(index_max_freq - index_min_freq) // 2)
		temp = int(find_nearest_index(y[index_min_freq:index_max_freq],(y[index_min_freq:index_max_freq]).max()))
		# if len(temp) == 1:
		peak_index = index_min_freq + int(temp)
		peak_freq_record.append(x[peak_index])
		peak_value = float(y[peak_index])
		peak_value_record.append(peak_value)
	record_magnitude = np.array(record_magnitude)
	record_phase = np.array(record_phase)
	record_freq = np.array(record_freq)
	peak_freq_record = np.array(peak_freq_record)
	peak_value_record = np.array(peak_value_record)
	if plot_conparison==True:
		plt.figure(2)
		plt.title('Amplitued from fast Fourier transform averaged in a wondow of ' + str(window+1) + 'pixels around ' + str(
			poscentre) + ', framerate ' + str(framerate) + 'Hz')
		plt.xlabel('Frequency [Hz]')
		plt.ylabel('Amplitude [au]')
		plt.grid()
		plt.semilogy()
		plt.legend()



	# I find the highest peak and that will be the one I use
	index = int(find_nearest_index(peak_value_record, max(peak_value_record)+1))
	shift = index * step
	datasection = datarestricted[min_start:max_start-fft_window_move+shift]
	spectra = np.fft.fft(datasection, axis=0)
	# magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
	magnitude = 2 * np.abs(spectra) / len(spectra)
	phase = np.angle(spectra)
	freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
	freq_to_erase = peak_freq_record[index]
	freq_to_erase_index = int(find_nearest_index(freq, freq_to_erase))
	framenumber = np.linspace(0, len(data) - 1, len(data)) - min_start
	data2 = data - np.multiply(magnitude[freq_to_erase_index], np.cos(np.repeat(np.expand_dims(phase[freq_to_erase_index], axis=0), len(data), axis=0) + np.repeat(np.expand_dims(np.repeat(np.expand_dims(2 * np.pi * freq_to_erase * framenumber / framerate, axis=-1),np.shape(data)[1], axis=-1), axis=-1), np.shape(data)[2], axis=-1)))

	# section only for stats
	datasection = data[:, poscentre[0] - 1:poscentre[0] + window, poscentre[1] - 1:poscentre[1] + window]
	spectra = np.fft.fft(datasection, axis=0)
	magnitude = 2 * np.abs(spectra) / len(spectra)
	phase = np.angle(spectra)
	freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
	magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
	y = np.array(
		[magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
	x = np.sort(freq)
	index_min_freq = int(find_nearest_index(x, min_frequency_to_erase))  # I restric the window over which I do the peak search
	index_max_freq = int(find_nearest_index(x, max_frequency_to_erase))
	index_7 = int(find_nearest_index(x, 7))
	index_n7 = int(find_nearest_index(x, -7))
	index_min_freq_n = int(find_nearest_index(x, -min_frequency_to_erase))
	index_max_freq_n = int(find_nearest_index(x, -max_frequency_to_erase))
	index_0 = int(find_nearest_index(x, 0))
	noise = (np.array(y[3:index_max_freq_n].tolist() + y[index_min_freq_n:index_n7].tolist() + y[index_7:index_min_freq].tolist() + y[index_max_freq:-3].tolist())).max()
	temp = int(find_nearest_index(y[index_min_freq:index_max_freq], (y[index_min_freq:index_max_freq]).max()))
	peak_index = index_min_freq + int(temp)
	peak_value_pre_filter = float(y[peak_index])

	datasection = data2[:, poscentre[0] - 1:poscentre[0] + window, poscentre[1] - 1:poscentre[1] + window]
	spectra = np.fft.fft(datasection, axis=0)
	magnitude = 2 * np.abs(spectra) / len(spectra)
	phase = np.angle(spectra)
	freq = np.fft.fftfreq(len(magnitude), d=1 / framerate)
	magnitude_space_averaged = np.mean(magnitude, axis=(-1, -2))
	y = np.array(
		[magnitude_space_averaged for _, magnitude_space_averaged in sorted(zip(freq, magnitude_space_averaged))])
	x = np.sort(freq)
	temp = int(find_nearest_index(y[index_min_freq:index_max_freq], (y[index_min_freq:index_max_freq]).max()))
	peak_index = index_min_freq + int(temp)
	peak_value_post_filter = float(y[peak_index])


	if plot_conparison==True:
		plt.figure(1)
		datasection2 = data2
		# spectra = np.fft.fft(datasection2, axis=0)
		# # magnitude=np.sqrt(np.add(np.power(real,2),np.power(imag,2)))
		# magnitude2 = 2 * np.abs(spectra) / len(spectra)
		# phase2 = np.angle(spectra)
		# freq = np.fft.fftfreq(len(magnitude2), d=1 / framerate)
		for i in range(len(poscentred)):
			pos = poscentred[i]
			spectra = np.fft.fft(datasection2[:, pos[0] - window:pos[0] + window, pos[1] - window:pos[1] + window], axis=0)
			magnitude2 = 2 * np.abs(spectra) / len(spectra)
			freq = np.fft.fftfreq(len(magnitude2), d=1 / framerate)
			y = np.mean(magnitude2, axis=(-1, -2))
			# y=magnitude[:,pos[0],pos[1]]
			y = np.array([y for _, y in sorted(zip(freq, y))])
			x = np.sort(freq)
			plt.plot(x, y, color[i] + '--',
					 label='data at the point ' + str(pos) + ', ' + str(np.around(freq_to_erase,decimals=2)) + 'Hz oscillation substracted')
		# plt.title()


		plt.grid()
		plt.semilogy()
		plt.legend()
		plt.pause(0.0001)




	print('stats of the oscillation removal')
	print('with window of size '+str(np.around(section_frames/framerate,decimals=1))+'s of '+str(len(data)/framerate)+'s of record')
	print('found oscillation of frequency '+str(freq_to_erase)+'Hz')
	print('On the foil centre oscillation magnitude reduced from '+str(peak_value_pre_filter)+'[au] to '+str(peak_value_post_filter)+'[au] \nwith an approximate maximum noise level of '+str(noise)+'[au]')

	return np.array([data2])




###################################################################################################


def abs_angle(angle,deg=False):
	import numpy as np

	out = angle
	if deg==False:
		if out<=0:
			out=angle+2*np.pi
	elif deg==True:
		if out<=0:
			out=angle+360

	return out



###################################################################################################



def split_fixed_length(A,length):
	A=np.array(A)
	length = int(length)
	B=[]
	while len(A)>length:
		B.append(A[:length].tolist())
		A=A[length:]
	if len(A)>0:
		B.append(A.tolist())
	return np.array(B)



###################################################################################################


# function to print on screen a whole array even if it is very large

def fullprint(*args, **kwargs):
	from pprint import pprint
	import numpy as np
	opt = np.get_printoptions()
	np.set_printoptions(threshold=np.inf)
	pprint(*args, **kwargs)
	np.set_printoptions(**opt)



###################################################################################################


def find_dead_pixels(data, start_interval='auto',end_interval='auto', framerate='auto',from_data=True,treshold_for_bad_std=10,treshold_for_bad_difference=13):

	# Created 26/02/2019
	# function that finds the dead pixels. 'data' can be a string with the path of the record or the data itself.
	# if the path is given the oscillation filtering is done
	# default tresholds for std and difference found 25/02/2019

	# Output legend:
		# 3 = treshold for difference with neighbouring pixels trepassed
		# 6 = treshold for std trepassed
		# 9 = both treshold trepassed

	if from_data==False:
		if framerate=='auto':
			print('if you load from file you must specify the framerate')
			exit()
		path=data
		print('Loading a recod from '+path)
		filenames = all_file_names(path, '.npy')[0]
		data = np.load(os.path.join(path, filenames))
		data2 = clear_oscillation_central2(data, framerate, plot_conparison=False)
		data = np.array(data2)
	else:
		data = np.array(data)

	if end_interval=='auto':
		end_interval=np.shape(data)[1]
	else:
		if framerate=='auto':
			print('if you specify an end time you must specify the framerate')
			exit()
		else:
			end_interval=end_interval*framerate
			if (end_interval>np.shape(data)[1]):
				end_interval = np.shape(data)[1]

	if start_interval=='auto':
		start_interval=0
	else:
		if framerate=='auto':
			print('if you specify a start time you must specify the framerate')
			exit()
		else:
			start_interval=start_interval*framerate
			if (start_interval>end_interval):
				print("You set the start interval after the end one, I'm going to use zero seconds")
				start_interval = 0


	data=data[0,start_interval:end_interval]
	mean=np.mean(data,axis=(0))
	std=np.std(data,axis=(0))
	flag_ceck=np.zeros(np.shape(std))
	flag = np.ones(np.shape(std))
	for i in range(np.shape(std)[0]):
		for j in range(np.shape(std)[1]):
			if std[i,j]>treshold_for_bad_std:
				flag_ceck[i,j]=6
				flag[i, j] = 0
	for i in range(1, np.shape(std)[0] - 1):
		for j in range(1, np.shape(std)[1] - 1):
			temp = (mean[i - 1, j - 1:j + 2] * flag[i - 1, j - 1:j + 2]).tolist() + [
				(mean[i, j - 1] * flag[i, j - 1]).tolist()] + [(mean[i, j + 1] * flag[i, j + 1]).tolist()] + (
						   mean[i + 1, j - 1:j + 2] * flag[i + 1, j - 1:j + 2]).tolist()
			temp2 = [x for x in temp if x != 0]
			if (mean[i, j] > max(temp2) + treshold_for_bad_difference or mean[i, j] < min(
					temp2) - treshold_for_bad_difference):
				flag_ceck[i, j] += 3

	counter = collections.Counter(flatten_full(flag_ceck))
	print('Number of pixels that trepass the limit for count difference with neighbours '+str(counter.get(3)))
	print('Number of pixels that trepass the limit for standard deviation '+str(counter.get(6)))
	print('Number of pixels that trepass both limits '+str(counter.get(9)))

	return flag_ceck


###################################################################################################


def replace_dead_pixels(data,flag, framerate='auto',from_data=True):

	# Created 26/02/2019
	# function replace dead pixels with the average from their neighbours

	if from_data==False:
		if framerate=='auto':
			print('if you load from file you must specify the framerate')
			exit()
		path=data
		print('Loading a recod from '+path)
		filenames = all_file_names(path, '.npy')[0]
		data = np.load(os.path.join(path, filenames))
		data2 = clear_oscillation_central2(data, framerate, plot_conparison=False)
		data = np.array(data2)[0]
	else:
		data = np.array(data)[0]

	flag=np.array(flag)

	if np.shape(data[0])!=np.shape(flag):
		print('There is something wrong, the shape of the data and the dead pixels map you want to use are not the same')
		exit()

	data2=np.array([data])

	for i in range(np.shape(data)[1]):
		for j in range(np.shape(data)[2]):
			if flag[i,j]!=0:
				temp=[]
				for i_ in [-1,0,1]:
					for j_ in [-1, 0, 1]:
						if (i_!=0 and j_!=0):
							if flag[i+i_,j+j_]==0:
								temp.append(data[:,i+i_, j +j_])
				data2[0,:,i,j]=np.mean(temp,axis=(0))

	return data2


###################################################################################################


def record_rotation(data,rotangle, foilcenter= [160, 133],	foilhorizw = 0.09, foilvertw = 0.07,foilhorizwpixel = 240,precisionincrease = 10):

	# Created 26/02/2019
	# function created to rotate and crop a whole record given central point and rotation angle

	mean = data[0,0]

	testrot = mean
	foilrot = rotangle * 2 * np.pi / 360
	foilrotdeg = rotangle
	foilvertwpixel = np.int(np.around((foilhorizwpixel * foilvertw) / foilhorizw))
	r = ((foilhorizwpixel ** 2 + foilvertwpixel ** 2) ** 0.5) / 2  # HALF DIAGONAL
	a = foilvertwpixel / np.cos(foilrot)
	tgalpha = np.tan(foilrot)
	delta = -(a ** 2) / 4 + (1 + tgalpha ** 2) * (r ** 2)
	foilx = np.add(foilcenter[0], [(-0.5 * a * tgalpha + delta ** 0.5) / (1 + tgalpha ** 2),
								   (-0.5 * a * tgalpha - delta ** 0.5) / (1 + tgalpha ** 2),
								   (0.5 * a * tgalpha - delta ** 0.5) / (1 + tgalpha ** 2),
								   (0.5 * a * tgalpha + delta ** 0.5) / (1 + tgalpha ** 2),
								   (-0.5 * a * tgalpha + delta ** 0.5) / (1 + tgalpha ** 2)])
	foily = np.add(foilcenter[1] - tgalpha * foilcenter[0],
				   [tgalpha * foilx[0] + a / 2, tgalpha * foilx[1] + a / 2, tgalpha * foilx[2] - a / 2,
					tgalpha * foilx[3] - a / 2, tgalpha * foilx[0] + a / 2])
	foilxint = (np.rint(foilx)).astype(int)
	foilyint = (np.rint(foily)).astype(int)


	dummy = np.ones(np.multiply(np.shape(testrot), precisionincrease))
	dummy[foilcenter[1] * precisionincrease, foilcenter[0] * precisionincrease] = 2
	dummy[int(foily[0] * precisionincrease), int(foilx[0] * precisionincrease)] = 3
	dummy[int(foily[1] * precisionincrease), int(foilx[1] * precisionincrease)] = 4
	dummy[int(foily[2] * precisionincrease), int(foilx[2] * precisionincrease)] = 5
	dummy[int(foily[3] * precisionincrease), int(foilx[3] * precisionincrease)] = 6
	dummy2 = rotate(dummy, foilrotdeg, axes=(-1, -2), order=0)
	foilcenterrot = (
		np.rint(
			[np.where(dummy2 == 2)[1][0] / precisionincrease, np.where(dummy2 == 2)[0][0] / precisionincrease])).astype(
		int)
	foilxrot = (
		np.rint([np.where(dummy2 == 3)[1][0] / precisionincrease, np.where(dummy2 == 4)[1][0] / precisionincrease,
				 np.where(dummy2 == 5)[1][0] / precisionincrease, np.where(dummy2 == 6)[1][0] / precisionincrease,
				 np.where(dummy2 == 3)[1][0] / precisionincrease])).astype(int)
	foilyrot = (
		np.rint([np.where(dummy2 == 3)[0][0] / precisionincrease, np.where(dummy2 == 4)[0][0] / precisionincrease,
				 np.where(dummy2 == 5)[0][0] / precisionincrease, np.where(dummy2 == 6)[0][0] / precisionincrease,
				 np.where(dummy2 == 3)[0][0] / precisionincrease])).astype(int)

	foillx = min(foilxrot)
	foilrx = max(foilxrot)
	foilhorizwpixel = foilrx - foillx
	foildw = min(foilyrot)
	foilup = max(foilyrot)
	foilvertwpixel = foilup - foildw

	datarot = rotate(data, foilrotdeg, axes=(-1, -2))
	datacrop = datarot[:, :, foildw:foilup, foillx:foilrx]

	return datacrop


#####################################################################################################

def sensitivities_matrix_averaging_foil_pixels(sensitivities,h_pixels,pixelmean_h,pixelmean_v):

	# Created 27/02/2019
	# Does the average of a homogeneous sensitivity matrix on foil pixels
	# It can return a non homogeneous foil, and loose resolution due to the residual of h_pixels/pixelmean_h and v_pixels/pixelmean_v
	#
	#


	shapeorig=np.shape(sensitivities)
	npixels=shapeorig[0]
	nvoxels=shapeorig[1]

	v_pixels=npixels//h_pixels

	h_end_pixels = h_pixels//pixelmean_h
	v_end_pixels = v_pixels // pixelmean_v

	print('the final number of pixels is (vertical,horizontal)'+str([v_end_pixels,h_end_pixels]))
	print('the loss of pixels is (vertical,horizontal)'+str([v_pixels-v_end_pixels*pixelmean_v,h_pixels-h_end_pixels*pixelmean_h]))

	h_shift = (h_pixels-h_end_pixels*pixelmean_h)//2
	v_shift = (v_pixels-v_end_pixels*pixelmean_v)//2

	sensitivities_averaged=[]
	for voxel in range(nvoxels):
		voxel_sensitivity=[]
		foil_image=np.array(split_fixed_length(sensitivities[:,voxel],h_pixels))
		for i in range(v_end_pixels):
			for j in range(h_end_pixels):
				voxel_sensitivity.append(np.mean(foil_image[v_shift+i*pixelmean_v:v_shift+(i+1)*pixelmean_v,h_shift+j*pixelmean_h:h_shift+(j+1)*pixelmean_h]))
		sensitivities_averaged.append(voxel_sensitivity)

	sensitivities_averaged=np.array(sensitivities_averaged)

	return sensitivities_averaged.T


#####################################################################################################

def sensitivities_matrix_averaging_foil_pixels_extra(sensitivities,h_pixels,pixelmean_h):

	# Created 27/02/2019
	# Does the average of a homogeneous sensitivity matrix on foil pixels
	# It can loose resolution due to the residual of h_pixels/pixelmean_h and v_pixels/pixelmean_v
	#
	# Additionally from sensitivities_matrix_averaging_foil_pixels this function calculate also the pixels obtained from shifting the area of the average, one by one
	# the additional constrain is that the foil grid will be homogeneous


	pixelmean_v=pixelmean_h
	shapeorig=np.shape(sensitivities)
	npixels=shapeorig[0]
	nvoxels=shapeorig[1]

	v_pixels=npixels//h_pixels

	h_end_pixels = h_pixels//pixelmean_h
	v_end_pixels = v_pixels // pixelmean_v

	print('the final number of pixels is (vertical,horizontal)'+str([v_end_pixels,h_end_pixels]))
	print('the loss of pixels is (vertical,horizontal)' + str([v_pixels - v_end_pixels * pixelmean_v, h_pixels - h_end_pixels * pixelmean_h]))
	print('but the sensityvity matrix generated will be sized '+str([nvoxels,v_end_pixels*h_end_pixels+(v_end_pixels-1)*(h_end_pixels-1)*(pixelmean_h-1)]))
	print('instead of (vertical,horizontal)'+str([nvoxels,v_end_pixels*h_end_pixels]))

	h_shift = (h_pixels-h_end_pixels*pixelmean_h)//2
	v_shift = (v_pixels-v_end_pixels*pixelmean_v)//2


	sensitivities_averaged=[]
	for voxel in range(nvoxels):
		voxel_sensitivity=[]
		foil_image=np.array(split_fixed_length(sensitivities[:,voxel],h_pixels))
		for i in range(v_end_pixels):
			for j in range(h_end_pixels):
				voxel_sensitivity.append(np.mean(foil_image[v_shift+i*pixelmean_v:v_shift+(i+1)*pixelmean_v,h_shift+j*pixelmean_h:h_shift+(j+1)*pixelmean_h]))
		sensitivities_averaged.append(voxel_sensitivity)
	intermediate=(np.array(sensitivities_averaged).T).tolist()

	print('check 1')
	print('intermediate matrix size is ' + str(np.shape(intermediate)))


	for extra_shift in range(1,pixelmean_h,1):
		sensitivities_averaged = []
		for voxel in range(nvoxels):
			voxel_sensitivity=[]
			foil_image=np.array(split_fixed_length(sensitivities[:,voxel],h_pixels))
			for i in range(v_end_pixels-1):
				for j in range(h_end_pixels-1):
					voxel_sensitivity.append(np.mean(foil_image[extra_shift+v_shift+i*pixelmean_v:extra_shift+v_shift+(i+1)*pixelmean_v,extra_shift+h_shift+j*pixelmean_h:extra_shift+h_shift+(j+1)*pixelmean_h]))
			sensitivities_averaged.append(voxel_sensitivity)
		print('check ' + str(extra_shift + 1))
		print('sensitivities_averaged matrix size is ' + str(np.shape(sensitivities_averaged)))
		intermediate.extend((np.array(sensitivities_averaged).T).tolist())

	intermediate=np.array(intermediate)

	print('sensitivity matrix size is '+str(np.shape(intermediate)))

	return intermediate

#####################################################################################################

def foil_measurement_averaging_foil_pixels_extra(d,h_pixels,pixelmean_h):

	# Created 27/02/2019
	# Does the average of a homogeneous sensitivity matrix on foil pixels
	# It can loose resolution due to the residual of h_pixels/pixelmean_h and v_pixels/pixelmean_v
	#
	# Additionally, this function calculate also the pixels obtained from shifting the area of the average, one by one
	# the additional constrain is that the grid foil will be homogeneous




	pixelmean_v=pixelmean_h
	shapeorig=np.shape(d)
	npixels=shapeorig[0]

	v_pixels=npixels//h_pixels

	h_end_pixels = h_pixels//pixelmean_h
	v_end_pixels = v_pixels // pixelmean_v

	print('the final number of pixels is (vertical,horizontal)'+str([v_end_pixels,h_end_pixels]))
	print('but the foil reading matrix generated will be sized '+str([1,v_end_pixels*h_end_pixels+(v_end_pixels-1)*(h_end_pixels-1)*(pixelmean_h-1)]))
	print('the loss of pixels is (vertical,horizontal)'+str([v_pixels-v_end_pixels*pixelmean_v,h_pixels-h_end_pixels*pixelmean_h]))

	h_shift = (h_pixels-h_end_pixels*pixelmean_h)//2
	v_shift = (v_pixels-v_end_pixels*pixelmean_v)//2

	foil_image = np.array(split_fixed_length(d, h_pixels))

	averaged_foil_reading=[]
	for i in range(v_end_pixels):
		for j in range(h_end_pixels):
			averaged_foil_reading.append(np.mean(foil_image[v_shift+i*pixelmean_v:v_shift+(i+1)*pixelmean_v,h_shift+j*pixelmean_h:h_shift+(j+1)*pixelmean_h]))



	for extra_shift in range(1,pixelmean_h,1):
		for i in range(v_end_pixels-1):
			for j in range(h_end_pixels-1):
				averaged_foil_reading.append(np.mean(foil_image[extra_shift+v_shift+i*pixelmean_v:extra_shift+v_shift+(i+1)*pixelmean_v,extra_shift+h_shift+j*pixelmean_h:extra_shift+h_shift+(j+1)*pixelmean_h]))

	averaged_foil_reading=np.array(averaged_foil_reading)

	print('power on the foil array size is '+str(np.shape(averaged_foil_reading)))

	return averaged_foil_reading



#####################################################################################################

def sensitivities_matrix_averaging_foil_pixels_loseless(sensitivities,h_pixels,pixelmean_h,pixelmean_v):

	# Created 28/03/2019
	# Does the average of a homogeneous sensitivity matrix on foil pixels
	# Different from sensitivities_matrix_averaging_foil_pixels because I do it loseless. I do not discard the last pixels
	#
	#


	shapeorig=np.shape(sensitivities)
	npixels=shapeorig[0]
	nvoxels=shapeorig[1]

	v_pixels=npixels//h_pixels

	h_end_pixels = np.ceil(h_pixels/pixelmean_h).astype('int')
	v_end_pixels = np.ceil(v_pixels/pixelmean_v).astype('int')
	pixels_averaged = pixelmean_h*pixelmean_v

	print('the final number of pixels is (vertical,horizontal)'+str([v_end_pixels,h_end_pixels]))

	sensitivities_averaged=[]
	for voxel in range(nvoxels):
		voxel_sensitivity=[]
		foil_image=np.array(split_fixed_length(sensitivities[:,voxel],h_pixels))
		for i in range(v_end_pixels):
			for j in range(h_end_pixels):
				voxel_sensitivity.append(np.sum(foil_image[i*pixelmean_v:(i+1)*pixelmean_v,j*pixelmean_h:(j+1)*pixelmean_h])/pixels_averaged)
		sensitivities_averaged.append(voxel_sensitivity)

	sensitivities_averaged=np.array(sensitivities_averaged)


	return sensitivities_averaged.T


#####################################################################################################

def foil_measurement_averaging_foil_pixels_loseless(d,h_pixels,pixelmean_h,pixelmean_v):

	# Created 01/04/2019
	# Does the average of a regular sensitivity matrix on foil pixels
	#


	shapeorig=np.shape(d)
	npixels=shapeorig[0]

	v_pixels=npixels//h_pixels

	h_end_pixels = np.ceil(h_pixels/pixelmean_h).astype('int')
	v_end_pixels = np.ceil(v_pixels/pixelmean_v).astype('int')
	pixels_averaged = pixelmean_h*pixelmean_v

	print('the final number of pixels is (vertical,horizontal)'+str([v_end_pixels,h_end_pixels]))

	foil_image = np.array(split_fixed_length(d, h_pixels))

	averaged_foil_reading=[]
	for i in range(v_end_pixels):
		for j in range(h_end_pixels):
			averaged_foil_reading.append(np.sum(foil_image[i*pixelmean_v:(i+1)*pixelmean_v,j*pixelmean_h:(j+1)*pixelmean_h])/pixels_averaged)


	averaged_foil_reading=np.array(averaged_foil_reading)

	print('power on the foil array size is '+str(np.shape(averaged_foil_reading)))

	return averaged_foil_reading


#####################################################################################################

def sensitivities_matrix_averaging_foil_pixels_extra_loseless(sensitivities,h_pixels,pixelmean_h,pixelmean_v):

	# Created 30/03/2019
	# Does the average of a regular sensitivity matrix on foil pixels
	# Additionally from sensitivities_matrix_averaging_foil_pixels this function calculate also the pixels obtained from shifting the area of the average, one by one
	#
	# Different from sensitivities_matrix_averaging_foil_pixels_extra because I do it loseless. I do not discard the last pixels
	#
	#


	shapeorig=np.shape(sensitivities)
	npixels=shapeorig[0]
	nvoxels=shapeorig[1]

	v_pixels=npixels//h_pixels

	h_end_pixels = np.ceil(h_pixels/pixelmean_h).astype('int')
	v_end_pixels = np.ceil(v_pixels/pixelmean_v).astype('int')
	pixels_averaged = pixelmean_h*pixelmean_v

	print('the final number of pixels is (vertical,horizontal)'+str([v_end_pixels,h_end_pixels]))
	print('but the sensityvity matrix generated will be sized '+str([nvoxels,v_end_pixels*h_end_pixels+(v_end_pixels-1)*(h_end_pixels-1)*(pixelmean_h-1)*(pixelmean_v-1)]))
	print('instead of (vertical,horizontal)'+str([nvoxels,v_end_pixels*h_end_pixels]))


	sensitivities_averaged=[]
	for voxel in range(nvoxels):
		voxel_sensitivity=[]
		foil_image=np.array(split_fixed_length(sensitivities[:,voxel],h_pixels))
		for i in range(v_end_pixels):
			for j in range(h_end_pixels):
				voxel_sensitivity.append(np.sum(foil_image[i*pixelmean_v:(i+1)*pixelmean_v,j*pixelmean_h:(j+1)*pixelmean_h])/pixels_averaged)
		sensitivities_averaged.append(voxel_sensitivity)
	intermediate=(np.array(sensitivities_averaged).T).tolist()

	print('check 1')
	print('intermediate matrix size is ' + str(np.shape(intermediate)))

	for extra_v_shift in range(0, pixelmean_v, 1):
		for extra_h_shift in range(0,pixelmean_h,1):
			if (extra_v_shift+extra_v_shift)==0:
				continue
			sensitivities_averaged = []
			for voxel in range(nvoxels):
				voxel_sensitivity=[]
				foil_image=np.array(split_fixed_length(sensitivities[:,voxel],h_pixels))
				if extra_v_shift==0:
					for i in range(v_end_pixels):
						for j in range(h_end_pixels-1):
							voxel_sensitivity.append(np.sum(foil_image[extra_v_shift+i*pixelmean_v:extra_v_shift+(i+1)*pixelmean_v,extra_h_shift+j*pixelmean_h:extra_h_shift+(j+1)*pixelmean_h])/pixels_averaged)
				elif extra_h_shift == 0:
					for i in range(v_end_pixels-1):
						for j in range(h_end_pixels):
							voxel_sensitivity.append(np.sum(foil_image[extra_v_shift+i*pixelmean_v:extra_v_shift+(i+1)*pixelmean_v,extra_h_shift+j*pixelmean_h:extra_h_shift+(j+1)*pixelmean_h])/pixels_averaged)
				else:
					for i in range(v_end_pixels-1):
						for j in range(h_end_pixels-1):
							voxel_sensitivity.append(np.sum(foil_image[extra_v_shift+i*pixelmean_v:extra_v_shift+(i+1)*pixelmean_v,extra_h_shift+j*pixelmean_h:extra_h_shift+(j+1)*pixelmean_h])/pixels_averaged)
				sensitivities_averaged.append(voxel_sensitivity)
			print('check ' + str(extra_v_shift + extra_h_shift + 1))
			print('sensitivities_averaged matrix size is ' + str(np.shape(sensitivities_averaged)))
			intermediate.extend((np.array(sensitivities_averaged).T).tolist())

	intermediate=np.array(intermediate)

	print('sensitivity matrix size is '+str(np.shape(intermediate)))

	return intermediate

#####################################################################################################

def foil_measurement_averaging_foil_pixels_extra_loseless(d,h_pixels,pixelmean_h,pixelmean_v):

	# Created 30/03/2019
	# Does the average of a rgular sensitivity matrix on foil pixels
	#
	# Additionally, this function calculate also the pixels obtained from shifting the area of the average, one by one



	shapeorig=np.shape(d)
	npixels=shapeorig[0]

	v_pixels=npixels//h_pixels

	h_end_pixels = np.ceil(h_pixels/pixelmean_h).astype('int')
	v_end_pixels = np.ceil(v_pixels/pixelmean_v).astype('int')
	pixels_averaged = pixelmean_h*pixelmean_v

	print('the final number of pixels is (vertical,horizontal)'+str([v_end_pixels,h_end_pixels]))
	print('but the foil reading matrix generated will be sized '+str([1,v_end_pixels*h_end_pixels+(v_end_pixels-1)*(h_end_pixels-1)*(pixelmean_h-1)*(pixelmean_v-1)]))

	foil_image = np.array(split_fixed_length(d, h_pixels))

	averaged_foil_reading=[]
	for i in range(v_end_pixels):
		for j in range(h_end_pixels):
			averaged_foil_reading.append(np.sum(foil_image[i*pixelmean_v:(i+1)*pixelmean_v,j*pixelmean_h:(j+1)*pixelmean_h])/pixels_averaged)

	for extra_v_shift in range(0, pixelmean_v, 1):
		for extra_h_shift in range(0,pixelmean_h,1):
			if (extra_v_shift+extra_v_shift)==0:
				continue
			if extra_v_shift == 0:
				for i in range(v_end_pixels):
					for j in range(h_end_pixels-1):
						averaged_foil_reading.append(np.sum(foil_image[extra_v_shift+i*pixelmean_v:extra_v_shift+(i+1)*pixelmean_v,extra_h_shift+j*pixelmean_h:extra_h_shift+(j+1)*pixelmean_h])/pixels_averaged)
			elif extra_h_shift == 0:
				for i in range(v_end_pixels-1):
					for j in range(h_end_pixels):
						averaged_foil_reading.append(np.sum(foil_image[extra_v_shift+i*pixelmean_v:extra_v_shift+(i+1)*pixelmean_v,extra_h_shift+j*pixelmean_h:extra_h_shift+(j+1)*pixelmean_h])/pixels_averaged)
			else:
				for i in range(v_end_pixels-1):
					for j in range(h_end_pixels-1):
						averaged_foil_reading.append(np.sum(foil_image[extra_v_shift+i*pixelmean_v:extra_v_shift+(i+1)*pixelmean_v,extra_h_shift+j*pixelmean_h:extra_h_shift+(j+1)*pixelmean_h])/pixels_averaged)

	averaged_foil_reading=np.array(averaged_foil_reading)

	print('power on the foil array size is '+str(np.shape(averaged_foil_reading)))

	return averaged_foil_reading



