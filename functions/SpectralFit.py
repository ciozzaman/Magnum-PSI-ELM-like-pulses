# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:59:28 2019

@author: akkermans
"""
import numpy as np
import pandas as pd
# from winspec import SpeFile
from .spectools import rotate,do_tilt, binData
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import copy as cp
import os
from scipy.signal import savgol_filter
from scipy.signal import medfilt
from scipy.interpolate import interp1d
import time as tm

from uncertainties import ufloat,unumpy,correlated_values
from uncertainties.unumpy import exp,nominal_values,std_devs,sqrt

fdir = 'data/'

#

def doSpecFit(df_settings,df_log,geom,waveLcoefs,binnedSens,Nx = 2048,nCh = 40):
	blockGauss = lambda x,A,w,sig,x0: A*(erf((x-x0+w)/sig)-erf((x-x0-w)/sig))/2*np.sqrt(np.pi)
	triGauss = lambda x,A,a,sig,x0: A*sig/a*((np.exp(-((x-x0)/sig)**2)-np.exp(-((x-x0-a)/sig)**2))/2 + (x-x0)/sig*np.sqrt(np.pi)/2*(erf((a-x+x0)/sig)+erf((x-x0)/sig)))
	#shapeFun = lambda x,A1,A2,w,sig,x0 : (triGauss(x,(A2-A1),2*w,sig,x0) + blockGauss(x,A1,w,sig,x0+w))/1.77
	#InstrFun = lambda x,A,x0 : (triGauss(x,(-.6*A),2*1.86,.31,x0) + blockGauss(x,A,1.86,.31,x0+1.86))/1.77
	InstrFun = lambda x,A,x0 : (triGauss(x,(-.75*A),2*1.46,.5,x0) + blockGauss(x,A,1.46,.5,x0+1.46))/1.77
	multPeakFun = lambda x,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10: sum([InstrFun(x,Ai,x0-0.207) for Ai,x0 in zip([A1,A2,A3,A4,A5,A6,A7,A8,A9,A10],waveLengths[5:15])])

	n = np.arange(10,20)
	waveLengths = [656.45377,486.13615,434.0462,410.174,397.0072,388.9049,383.5384] + list((364.6*(n*n)/(n*n-4)))
	lamb = np.array([np.polyval(waveLcoefs[0],np.arange(Nx)),np.polyval(waveLcoefs[1],np.arange(Nx)),np.polyval(waveLcoefs[1],np.arange(Nx))])

	peakUnitArea = -np.trapz(InstrFun(lamb[0],1,600),x=lamb[0])

	linesSeen = [range(2),range(1,11),range(1,11)]
	iSens = [[np.abs(lamb[i]-waveLengths[line]).argmin() for line in lines]for i,lines in enumerate(linesSeen)] #Indexes of peaks where we check sensitivity
	dfs_lineRads = [pd.DataFrame(index=range(nCh),columns=['n3','n4']) ,	# Fitted radiance of each line for the different exposure types
	pd.DataFrame(index=range(nCh),columns=['n4','n5','n6','n7','n8','n9','n10','n11','n12']) ,
	pd.DataFrame(index=range(nCh),columns=['n4','n5','n6','n7','n8','n9','n10','n11','n12']) ]

	df_lineRad = pd.DataFrame(index=range(nCh),columns=['n3','n4','n5','n6','n7','n8','n9','n10','n11','n12']) #Fitted radiance of each line, combined between different exposure types

	dfs_totIntens = [pd.DataFrame(index=range(len(df_settings)),columns=['n3','n4']) ,
	pd.DataFrame(index=range(len(df_settings)),columns=['n4','n5','n6','n7','n8','n9','n10','n11','n12']) ,
	pd.DataFrame(index=range(len(df_settings)),columns=['n4','n5','n6','n7','n8','n9','n10','n11','n12']) ]

	for iSet in range(len(df_settings)):
		for iTyp,typ in enumerate(['Ha','Hb','Hc']):
			j = df_settings.loc[iSet,typ]
			if not np.isnan(j):
				(folder,date,time) = df_log.loc[j,['folder','date','time']]
				fname = folder+'/'+date+time+'.spe'

				file = SpeFile(fdir+fname)
				data = file.data[0].T

				df_log.loc[j,'Dark']
				df_log.loc[df_log['type'] == 'Dark']

				jDark = np.where((df_log['time']==df_log.loc[j,'Dark']) & (df_log['type'] == 'Dark'))[0][0]

				(fDark,dDark,tDark) = df_log.loc[jDark,['folder','date','time']]
				fnDark = fDark+'/'+dDark+tDark+'.spe'
				fileDark = SpeFile(fdir+fnDark)
				dataDark = fileDark.data[0].T

				data -= dataDark
				data = rotate(data,geom['angle'])
				data = do_tilt(data,geom['tilt'])

				if typ=='Ha':
					binnedData = binData(data,geom['bin00a'],geom['binInterv'],check_overExp=False)
					for iBin in range(len(binnedData)):
						for h,l0 in enumerate(waveLengths[:2]):
							iFit = np.logical_and(lamb[iTyp] > l0-5, lamb[iTyp] < l0+5)
							xFit = lamb[iTyp][iFit]
							yFit = binnedData[iBin,iFit]
							bgFit = np.polyfit(xFit[[0,-1]],yFit[[0,-1]],1)
							yFit -= np.polyval(bgFit,xFit)
							p0 = [max(yFit),l0]

							fit = curve_fit(InstrFun,xFit,yFit,p0)
							dfs_lineRads[iTyp].loc[iBin][h] = fit[0][0]*peakUnitArea/binnedSens[iTyp,iBin,iSens[iTyp][h]]
					'''
					plt.figure()
					plt.plot(xFit,yFit)
					plt.plot(xFit,InstrFun(xFit,*fit[0]))
					'''
				else:
					binnedData,oExpLim = binData(data,geom['bin00b'],geom['binInterv'],check_overExp=True,nCh=40,treshold=np.inf)

					def calc_fit(iBin,lamb=lamb,iTyp=iTyp,oExpLim=oExpLim,waveLengths=waveLengths,binnedData=binnedData,InstrFun=InstrFun,peakUnitArea=peakUnitArea,binnedSens=binnedSens,iSens=iSens,multPeakFun=multPeakFun):
						import numpy as np
						from scipy.optimize import curve_fit
						from scipy.signal import find_peaks
						global dfs_lineRads,df_lineRad

						lambMax = lamb[iTyp][oExpLim[iBin]] - 2
						first = np.searchsorted(waveLengths < lambMax, 1)  # First non-overexposed line to be measured
						dfs_lineRads[iTyp].loc[iBin][:first - 1] = np.nan
						for h, l0 in enumerate(waveLengths[first:5], first - 1):
							iFit = np.logical_and(lamb[iTyp] > l0 - 5, lamb[iTyp] < l0 + 5)
							xFit = lamb[iTyp][iFit]
							yFit = binnedData[iBin, iFit]
							bgFit = np.polyfit(xFit[[0, -1]], yFit[[0, -1]], 1)
							yFit -= np.polyval(bgFit, xFit)
							p0 = [max(yFit), l0]

							fit = curve_fit(InstrFun, xFit, yFit, p0)
							dfs_lineRads[iTyp].loc[iBin][h] = fit[0][0] * peakUnitArea / binnedSens[iTyp, iBin, iSens[iTyp][h]]
							df_lineRad.loc[iBin][h + 1] = fit[0][0] * peakUnitArea / binnedSens[iTyp, iBin, iSens[iTyp][h]]

						iFit = np.logical_and(lamb[iTyp] > 370, lamb[iTyp] < 392)
						xFit = lamb[iTyp][iFit]
						yFit = binnedData[iBin, iFit]
						bgFit = np.polyfit(xFit[[0, -1]], yFit[[0, -1]], 1)
						yFit -= np.polyval(bgFit, xFit)

						ipks = find_peaks(yFit)[0]
						A0 = yFit[ipks[[np.abs((l0 - xFit[ipks])).argmin() for l0 in waveLengths[5:15]]]]

						fit = curve_fit(multPeakFun, xFit, yFit, A0)
						if first < 6:
							dfs_lineRads[iTyp].loc[iBin][-5:] = fit[0][:5] / binnedSens[iTyp, iBin, iSens[iTyp][5:10]]
							df_lineRad.loc[iBin][-5:] = fit[0][:5] / binnedSens[iTyp, iBin, iSens[iTyp][5:10]]
						else:
							dfs_lineRads[iTyp].loc[iBin][first - 10:] = fit[0][first - 5:5] / binnedSens[iTyp, iBin, iSens[iTyp][first:10]]
							df_lineRad.loc[iBin][first - 10:] = fit[0][first - 5:5] / binnedSens[iTyp, iBin, iSens[iTyp][first:10]]

						return iBin

					trash = map(calc_fit, range(len(binnedData)))
					trash = set(trash)



					# for iBin in range(len(binnedData)):
					#	 lambMax = lamb[iTyp][oExpLim[iBin]]-2
					#	 first = np.searchsorted(waveLengths < lambMax,1) #First non-overexposed line to be measured
					#	 dfs_lineRads[iTyp].loc[iBin][:first-1]=np.nan
					#	 for h,l0 in enumerate(waveLengths[first:5],first-1):
					#		 iFit = np.logical_and(lamb[iTyp] > l0-5, lamb[iTyp] < l0+5)
					#		 xFit = lamb[iTyp][iFit]
					#		 yFit = binnedData[iBin,iFit]
					#		 bgFit = np.polyfit(xFit[[0,-1]],yFit[[0,-1]],1)
					#		 yFit -= np.polyval(bgFit,xFit)
					#		 p0 = [max(yFit),l0]
					#
					#		 fit = curve_fit(InstrFun,xFit,yFit,p0)
					#		 dfs_lineRads[iTyp].loc[iBin][h] = fit[0][0]*peakUnitArea/binnedSens[iTyp,iBin,iSens[iTyp][h]]
					#		 df_lineRad.loc[iBin][h+1] = fit[0][0]*peakUnitArea/binnedSens[iTyp,iBin,iSens[iTyp][h]]
					#
					#	 iFit = np.logical_and(lamb[iTyp] > 370, lamb[iTyp] < 392)
					#	 xFit = lamb[iTyp][iFit]
					#	 yFit = binnedData[iBin,iFit]
					#	 bgFit = np.polyfit(xFit[[0,-1]],yFit[[0,-1]],1)
					#	 yFit -= np.polyval(bgFit,xFit)
					#
					#	 ipks = find_peaks(yFit)[0]
					#	 A0 = yFit[ipks[[np.abs((l0- xFit[ipks])).argmin() for l0 in waveLengths[5:15]]]]
					#
					#	 fit = curve_fit(multPeakFun,xFit,yFit,A0)
					#	 if first < 6:
					#		 dfs_lineRads[iTyp].loc[iBin][-5:] = fit[0][:5]/binnedSens[iTyp,iBin,iSens[iTyp][5:10]]
					#		 df_lineRad.loc[iBin][-5:] = fit[0][:5]/binnedSens[iTyp,iBin,iSens[iTyp][5:10]]
					#	 else:
					#		 dfs_lineRads[iTyp].loc[iBin][first-10:] = fit[0][first-5:5]/binnedSens[iTyp,iBin,iSens[iTyp][first:10]]
					#		 df_lineRad.loc[iBin][first-10:] = fit[0][first-5:5]/binnedSens[iTyp,iBin,iSens[iTyp][first:10]]
				dfs_totIntens[iTyp].loc[iSet] = dfs_lineRads[iTyp].sum(skipna=0)
				df_lineRad.n3 = dfs_lineRads[0].n3
				df_lineRad.to_csv('Radiance/%i.csv'%iSet)# Write to file


				if typ !='Ha':
					plt.figure()
					plt.plot(xFit,yFit)
					plt.plot(xFit,multPeakFun(xFit,*fit[0]))
					#plt.plot(xFit,multPeakFun(xFit,*A0))
				#plt.plot(xFit,np.polyval(bgFit,xFit))
				'''
				'''
				'''
				plt.figure()
				plt.plot(dfs_lineRads[1].n5)
				plt.plot(dfs_lineRads[2].n5)
				plt.plot(df_lineRad.n5)
				plt.pause(0.001)
				'''

# Check variation of Hb between the two exposure ranges;
# Variation of high lines between two exposure times looks OK
# Rigid Instrument function gives semi-poor fits with structural ~5% area error; acceptable for now

# Fit instrument function on different peaks
'''
lineShapeCoefs = np.zeros([5,5])
for h,l0 in enumerate(waveLengths[1:6]):
	iFit = np.logical_and(lamb > l0-5, lamb < l0+5)
	xFit = lamb[iFit]
	yFit = binnedData[20,iFit]
	if h==4:
		xFit = xFit[:76]
		yFit = yFit[:76]
	bgFit = np.polyfit(xFit[[0,-1]],yFit[[0,-1]],1)
	yFit -= np.polyval(bgFit,xFit)
	p0 = [max(yFit),max(yFit)/2,1.5,.2,l0]

	fit = curve_fit(shapeFun,xFit,yFit,p0)

	plt.figure();
	plt.plot(xFit,yFit)
	plt.plot(xFit,shapeFun(xFit,*fit[0]))
	#plt.plot(xFit,shapeFun(xFit,*p0))
	lineShapeCoefs[h] = fit[0]
'''


def doSpecFit_single_frame(binnedData,df_settings, df_log, geom, waveLcoefs, binnedSens,path_where_to_save_everything,time, type='Hb', pure_gaussian=True,perform_convolution=True,first_line_in_multipulse=3,total_number_of_lines_in_multipulse=10):
	import matplotlib.pyplot as plt

	nCh,Nx = binnedData.shape

	if pure_gaussian==True:
		InstrFun = lambda x, A, x0:(1/np.sqrt(2*np.pi*(0.85**2)))*A*np.exp(-np.power(x-x0,2)/(2*(0.85**2)))
		multPeakFun = lambda x, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10: sum([InstrFun(x, Ai, x0) for Ai, x0 in zip([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10], waveLengths[5:15])])
		InstrFun_biased = lambda x, A, x0, m,y0:InstrFun(x, A, x0) + m*x+y0
		multPeakFun_biased = lambda x, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10,m,y0: sum([InstrFun(x, Ai, x0) for Ai, x0 in zip([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10], waveLengths[5:15])])+m*x+y0
		multPeakFun_biased_2 = lambda x, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10,x01,x02,x03,x04,x05,x06,x07,x08,x09,x010, m, y0: sum([InstrFun(x, Ai, x0i) for Ai, x0i in zip([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10], [x01, x02, x03, x04, x05, x06, x07, x08, x09, x010])]) + m * x + y0
		full_InstrFun = lambda x, A, sig, x0:(1/np.sqrt(2*np.pi*(sig**2)))*A*np.exp(-np.power(x-x0,2)/(2*(sig**2)))
		full_InstrFun_biased = lambda x, A, sig, x0, m,y0:full_InstrFun(x, A, sig, x0) + m*x+y0
		full_InstrFun_biased_quad = lambda x, A, sig, x0, m1,m,y0:full_InstrFun(x, A, sig, x0) + m1*(x-x0)**2 + m*(x-x0)+y0
		full_InstrFun_biased_no_neg = lambda x, A, sig, x0, c,y0:full_InstrFun(x, A, sig, x0) + y0*c*x+y0
		multPeakFun_biased_10 = lambda x, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, sig, x01,x02,x03,x04,x05,x06,x07,x08,x09,x010, m1,m, y0: sum([full_InstrFun(x, Ai, sigi, x0i) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10], [sig, sig, sig, sig, sig, sig, sig, sig, sig, sig], [x01, x02, x03, x04, x05, x06, x07, x08, x09, x010])]) + m1*(x-np.mean([x01,x02,x03,x04,x05,x06,x07,x08,x09,x010]))**2 + m * (x-np.mean([x01,x02,x03,x04,x05,x06,x07,x08,x09,x010])) + y0
		multPeakFun_biased_10_3 = lambda x, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, sig, x01,x02,x03,x04,x05,x06,x07,x08,x09,x010, y_min,y_max, m: sum([full_InstrFun(x, Ai, sigi, x0i) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10], [sig, sig, sig, sig, sig, sig, sig, sig, sig, sig], [x01, x02, x03, x04, x05, x06, x07, x08, x09, x010])]) + ((y_min-y_max-m*(np.min(x)-np.max(x)))/((np.min(x)-np.max(x))**2))*(x-np.max(x))**2 + m* (x-np.max(x)) + y_max
		multPeakFun_biased_6 = lambda x, A1, A2, A3, A4, A5, A6, sig, x01,x02,x03,x04,x05, x06, m1,m, y0: sum([full_InstrFun(x, Ai, sigi, x0i) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6], [sig, sig, sig, sig, sig, sig], [x01, x02, x03, x04, x05, x06])]) + m1*(x-(x01+x06)/2)**2 + m * (x-(x01+x06)/2) + y0
		multPeakFun_biased_6_1 = lambda x, A1, A2, A3, A4, A5, A6, sig, x01,x02,x03,x04,x05, x06, m1,m, y0,x_ref: sum([full_InstrFun(x, Ai, sigi, x0i) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6], [sig, sig, sig, sig, sig, sig], [x01, x02, x03, x04, x05, x06])]) + m1*(x-x_ref)**2 + m * (x-x_ref) + y0
		multPeakFun_biased_6_2 = lambda x, A1, A2, A3, A4, A5, A6, sig, x01,x02,x03,x04,x05, x06, m1,c, y0: sum([full_InstrFun(x, Ai, sigi, x0i) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6], [sig, sig, sig, sig, sig, sig], [x01, x02, x03, x04, x05, x06])]) + m1*(x-np.max(x))**2 + (-m1*(np.min(x)-np.max(x)) -y0/(np.min(x)-np.max(x)) - c) * (x-np.max(x)) + y0
		multPeakFun_biased_6_3 = lambda x, A1, A2, A3, A4, A5, A6, sig, x01,x02,x03,x04,x05, x06, y_min,y_max, m: sum([full_InstrFun(x, Ai, sigi, x0i) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6], [sig, sig, sig, sig, sig, sig], [x01, x02, x03, x04, x05, x06])]) + ((y_min-y_max-m*(np.min(x)-np.max(x)))/((np.min(x)-np.max(x))**2))*(x-np.max(x))**2 + m* (x-np.max(x)) + y_max
		# def multPeakFun_biased_6_1(x_max,x_min):
		#	 multPeakFun_biased_6 = lambda x, A1, A2, A3, A4, A5, A6, sig, x01,x02,x03,x04,x05, x06, m1,m, y0: sum([full_InstrFun(x, Ai, sigi, x0i) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6], [sig, sig, sig, sig, sig, sig], [x01, x02, x03, x04, x05, x06])]) + m1*(x-(x01+x06)/2)**2 + m * (x-(x01+x06)/2) + y0
		#	 def polyadd(x, *params):
		#		 temp=0
		#		 for i in range(n):
		#			 temp+=params[i]*x**i
		#		 return temp
		#	 return polyadd


	else:
		blockGauss = lambda x, A, w, sig, x0: A * (erf((x - x0 + w) / sig) - erf((x - x0 - w) / sig)) / 2 * np.sqrt(np.pi)
		triGauss = lambda x, A, a, sig, x0: A * sig / a * ((np.exp(-((x - x0) / sig) ** 2) - np.exp(-((x - x0 - a) / sig) ** 2)) / 2 + (x - x0) / sig * np.sqrt(np.pi) / 2 * (erf((a - x + x0) / sig) + erf((x - x0) / sig)))
		# shapeFun = lambda x,A1,A2,w,sig,x0 : (triGauss(x,(A2-A1),2*w,sig,x0) + blockGauss(x,A1,w,sig,x0+w))/1.77
		# InstrFun = lambda x,A,x0 : (triGauss(x,(-.6*A),2*1.86,.31,x0) + blockGauss(x,A,1.86,.31,x0+1.86))/1.77
		# InstrFun = lambda x, A, x0: (triGauss(x, (-.75 * A), 2 * 1.46, .5, x0) + blockGauss(x, A, 1.46, .5,x0 + 1.46)) / 1.77
		InstrFun = lambda x, A, x0: (triGauss(x, (-.75 * A), 2 * 1.46, .5, x0) + blockGauss(x, A, 1.46, .5,x0 + 1.46)) / 1.77
		# all_InstrFun = lambda x, A1, w1, sig1, x01, A2, a2, sig2, x02: (triGauss(x, (-.75 * A2), a2, sig2, x02) + blockGauss(x, A1, w1, sig1,x01))
		# all_InstrFun = lambda x, A1, w1, sig1, x01, A2, a2, sig2, x02: (triGauss(x, (-.75 * A2), a2, sig2, x02) + blockGauss(x, A1, w1, sig1,x01))
		multPeakFun = lambda x, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10: sum([InstrFun(x, Ai, x0 - 0.207) for Ai, x0 in zip([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10], waveLengths[5:15])])
		InstrFun_biased = lambda x, A, x0, m,y0:InstrFun(x, A, x0) + m*x+y0
		multPeakFun_biased = lambda x, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10,m,y0: sum([InstrFun(x, Ai, x0) for Ai, x0 in zip([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10], waveLengths[5:15])])+m*x+y0

	# if np.random.rand()<0.1:
	if np.min(np.abs(time - np.array([-0.2,0.2,0.55,1])))<=0.05:
		print('will plot the spectral fit at time '+str(time))
		if not os.path.exists(path_where_to_save_everything+'/spectral_fit_example'):
			os.makedirs(path_where_to_save_everything+'/spectral_fit_example')


	n = np.arange(10, 20)
	waveLengths = [656.45377, 486.13615, 434.0462, 410.174, 397.0072, 388.9049, 383.5384] + list(
		(364.6 * (n * n) / (n * n - 4)))
	lamb = np.array([np.polyval(waveLcoefs[0], np.arange(Nx)), np.polyval(waveLcoefs[1], np.arange(Nx)),
					 np.polyval(waveLcoefs[1], np.arange(Nx))])

	peakUnitArea = np.abs(np.trapz(InstrFun(lamb[1], 1, np.mean(lamb[1])), x=lamb[1]))

	linesSeen = [range(2), range(1, 11), range(1, 11)]
	iSens = [[np.abs(lamb[i] - waveLengths[line]).argmin() for line in lines] for i, lines in enumerate(linesSeen)]  # Indexes of peaks where we check sensitivity
	dfs_lineRads = [pd.DataFrame(index=range(nCh), columns=['n3', 'n4']),
					# Fitted radiance of each line for the different exposure types
					pd.DataFrame(index=range(nCh), columns=['n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12']),
					pd.DataFrame(index=range(nCh), columns=['n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12'])]

	df_lineRad = pd.DataFrame(index=range(nCh), columns=['n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11',
														 'n12'])  # Fitted radiance of each line, combined between different exposure types

	dfs_totIntens = [pd.DataFrame(index=range(len(df_settings)), columns=['n3', 'n4']),
					 pd.DataFrame(index=range(len(df_settings)),
								  columns=['n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12']),
					 pd.DataFrame(index=range(len(df_settings)),
								  columns=['n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12'])]

	iSet=0
	# for iSet in range(len(df_settings)):
	for iTyp, typ in enumerate(['Ha', 'Hb', 'Hc']):
		if typ!=type:
			continue

		# j = df_settings.loc[iSet, typ]
		# if not np.isnan(j):
		#	 (folder, date, time) = df_log.loc[j, ['folder', 'date', 'time']]
		#	 fname = folder + '/' + date + time + '.spe'
		#
		#	 file = SpeFile(fdir + fname)
		#	 data = file.data[0].T
		#
		#	 df_log.loc[j, 'Dark']
		#	 df_log.loc[df_log['type'] == 'Dark']
		#
		#	 jDark = np.where((df_log['time'] == df_log.loc[j, 'Dark']) & (df_log['type'] == 'Dark'))[0][0]
		#
		#	 (fDark, dDark, tDark) = df_log.loc[jDark, ['folder', 'date', 'time']]
		#	 fnDark = fDark + '/' + dDark + tDark + '.spe'
		#	 fileDark = SpeFile(fdir + fnDark)
		#	 dataDark = fileDark.data[0].T
		#
		#	 data -= dataDark
		#	 data = rotate(data, geom['angle'])
		#	 data = do_tilt(data, geom['tilt'])

		if typ == 'Ha':
			# binnedData = binData(data, geom['bin00a'], geom['binInterv'], check_overExp=False)
			oExpLim = np.zeros((len(binnedData))).astype('int')
			for iBin in range(len(binnedData)):
				for h, l0 in enumerate(waveLengths[:2]):
					iFit = np.logical_and(lamb[iTyp] > l0 - 5, lamb[iTyp] < l0 + 5)
					xFit = lamb[iTyp][iFit]
					yFit = binnedData[iBin, iFit]
					bgFit = np.polyfit(xFit[[0, -1]], yFit[[0, -1]], 1)
					yFit -= np.polyval(bgFit, xFit)
					p0 = [max(yFit), l0]

					fit = curve_fit(InstrFun, xFit, yFit, p0)
					dfs_lineRads[iTyp].loc[iBin][h] = fit[0][0] * peakUnitArea / binnedSens[
						iTyp, iBin, iSens[iTyp][h]]
			'''
			plt.figure()
			plt.plot(xFit,yFit)
			plt.plot(xFit,InstrFun(xFit,*fit[0]))
			'''
		elif typ == 'Hb':
			#section added to try to find a proper counts offset for the lines
			record_all_sigma=[]
			# binnedData, oExpLim = binData(data, geom['bin00b'][0], geom['binInterv'][0], check_overExp=True, nCh=40)
			oExpLim=np.zeros((len(binnedData))).astype('int')

			# highest_peak = int(np.median(binnedData.argmax(axis=1)))
			highest_peak = int(np.sum(binnedData,axis=0).argmax())
			highest_peak_wavel = lamb[iTyp][highest_peak]
			lambMax = lamb[iTyp][Nx - 1 - oExpLim[int(nCh/2)]] - 2
			first = np.searchsorted(waveLengths < lambMax, 1)  # First non-overexposed line to be measured
			highest_peak_number = (np.abs(waveLengths[first:5]-highest_peak_wavel)).argmin()
			# print('highest_peak_number '+str(highest_peak_number))
			fit_initial_record = []
			for iBin in range(len(binnedData)):
				if np.max(binnedData[iBin])<1:
					print('bin ' + str(iBin) + ' missing')
					continue
				lambMax = lamb[iTyp][Nx - 1 - oExpLim[iBin]] - 2
				first = np.searchsorted(waveLengths < lambMax, 1)  # First non-overexposed line to be measured
				dfs_lineRads[iTyp].loc[iBin][:first - 1] = np.nan
				try:
					for h, l0 in enumerate(waveLengths[first:first_line_in_multipulse+1], first - 1):
						if h>highest_peak_number+1:	# usually there are at least 2 good lines
							continue
						# iFit = np.logical_and(lamb[iTyp] > l0 + min(np.diff(waveLengths)[h + 1] / 2, -5),
						#					   lamb[iTyp] < min(l0 - min(np.diff(waveLengths)[h] / 2, -5),max(lamb[iTyp])))
						iFit = np.logical_and(lamb[iTyp] > l0 + min(np.diff(waveLengths)[h + 1] / 2, -5),
											  lamb[iTyp] < l0 - min(np.diff(waveLengths)[h] / 2, -5))
						# iFit = np.logical_and(lamb[iTyp] > l0 - 5,lamb[iTyp] < min(l0 + 5, max(lamb[iTyp])))
						xFit = lamb[iTyp][iFit]
						yFit = binnedData[iBin, iFit]

						if np.max(yFit)<1:
							print('bin ' + str(iBin) + ' missing')
							continue

						bds = [[0, 0, l0-5, -np.inf, -np.inf],
							   # [max(yFit) * 100, np.inf, max(xFit), np.inf, np.inf]]
							   [max(yFit) * 100, 5, l0+5, np.inf, np.inf]]
						if pure_gaussian == True:
							p0 = [max(yFit), 0.8, l0, 0, yFit[[0]]]
							try:
								fit = curve_fit(full_InstrFun_biased, xFit, yFit, p0, maxfev=1000000, bounds=bds)
								record_all_sigma.append([fit[0][1], fit[1][1, 1]])
							except:
								fit = [[0, 1, 0, 0, 0]]
								print('bin ' + str(iBin) + ' line ' + str(h + 4) + ' line missing')

						# if h == 1:
						# plt.figure()
						#	 # plt.title('wavelength ' + str(l0) + ', line ' + str(h + 4) + '\n fit ' + str(
						#	 #	 fit[0]) + ' uncertainty ' + str(np.diag(fit[1])))
						# plt.plot(xFit, yFit)
						#	 # # plt.plot(xFit, InstrFun(xFit,*fit[0])-np.polyval(bgFit, xFit))
						# plt.plot(xFit, full_InstrFun_biased(xFit, *fit[0]))
						# plt.pause(0.001)
						# if h==0:
						#	 print(fit[0][1])

				except:
					print('bin ' + str(iBin) + ' preliminary fit failed')
			record_all_sigma = np.abs(np.array(record_all_sigma))
			mean_sigma =  np.nansum(record_all_sigma[:,0]/ (record_all_sigma[:,1]**2))/np.nansum(1/record_all_sigma[:,1]**2)
			mean_sigma_deviation = np.std(record_all_sigma[:,0])
			min_sigma = max(mean_sigma - mean_sigma_deviation*4 , 0.4)
			print('min_sigma = '+str(min_sigma))
			max_sigma = min(mean_sigma + mean_sigma_deviation*4 , 1.6)
			print('max_sigma = ' + str(max_sigma))
			for iBin in range(len(binnedData)):
				if np.max(binnedData[iBin])<1:
					print('bin ' + str(iBin) + ' missing')
					continue
				if  ((np.min(np.abs(time - np.array([-0.2,0.2,0.55,1])))<=0.1) and (iBin in [5,20,30])):
					tm.sleep(np.random.random()*10)
					plt.figure(figsize=(20, 10))
					plt.plot(lamb[iTyp],binnedData[iBin],label='binned data')

				# selected_for_background = np.logical_and(lamb[iTyp]>505,np.arange(Nx)<1550) + np.logical_and(lamb[iTyp]>445,lamb[iTyp]<470)+ np.logical_and(lamb[iTyp]>420,lamb[iTyp]<425)+ np.logical_and(lamb[iTyp]>402,lamb[iTyp]<405)
				# xFit = lamb[iTyp][selected_for_background]
				# yFit = binnedData[iBin, selected_for_background]
				# # plt.figure()
				# # plt.plot(xFit,yFit)
				# # plt.plot(xFit,medfilt(yFit,11))
				# xFit_filtered = savgol_filter(xFit,51,2)
				# yFit_filtered = savgol_filter(medfilt(yFit,11),51,2)
				# # plt.plot(xFit_filtered,yFit_filtered,'x')
				# while np.sum(np.diff(xFit_filtered)<0)>0:
				#	 yFit_filtered = yFit_filtered[:-1][np.diff(xFit_filtered)>0]
				#	 xFit_filtered = xFit_filtered[:-1][np.diff(xFit_filtered)>0]
				# # plt.plot(xFit_filtered,yFit_filtered)
				# # plt.pause(0.01)
				# background_interpolation = interp1d(xFit_filtered,yFit_filtered,kind='linear',fill_value='extrapolate')

				temp = []
				temp_x = []
				for limit_a,limit_b in [[505,np.inf],[465,470],[445,450],[420,425],[401,404]]:
					selected_for_background = np.logical_and(np.logical_and(lamb[iTyp]>limit_a,lamb[iTyp]<limit_b) , np.arange(Nx)<1550)
					temp.append( np.mean(np.sort(binnedData[iBin, selected_for_background] ) [:np.sum(selected_for_background)//2] ) )
					temp_x.append( np.mean(lamb[iTyp][selected_for_background]) )
				background_interpolation = interp1d(temp_x,temp,kind='linear',fill_value='extrapolate')
				# plt.figure()
				# plt.plot(lamb[iTyp],background_interpolation(lamb[iTyp]))
				# plt.plot(temp_x,background_interpolation(temp_x),'+')
				# plt.semilogy()
				# plt.pause(0.01)
				lambMax = lamb[iTyp][Nx-1-oExpLim[iBin]] - 2
				first = np.searchsorted(waveLengths < lambMax, 1)  # First non-overexposed line to be measured
				dfs_lineRads[iTyp].loc[iBin][:first - 1] = np.nan
				try:
					for h, l0 in enumerate(waveLengths[first:first_line_in_multipulse+first], first - 1):
						# print(h)
						# print(h) print_example
						# iFit = np.logical_and(lamb[iTyp] > l0 + min(np.diff(waveLengths)[h+1]/2,-5), lamb[iTyp] < min(l0-min(np.diff(waveLengths)[h]/2,-5),max(lamb[iTyp])))
						iFit = np.logical_and(lamb[iTyp] > l0 + min(np.diff(waveLengths)[h+1]/2,-5), lamb[iTyp] < l0-min(np.diff(waveLengths)[h]/2,-5))
						iFit = np.logical_and(np.logical_and(iFit, np.arange(Nx)>20),np.arange(Nx)<Nx-20)	# added 16/02/2020 to afoid effects from tilting the image
						# iFit = np.logical_and(lamb[iTyp] > l0 - 5,lamb[iTyp] < min(l0 + 5, max(lamb[iTyp])))
						xFit = lamb[iTyp][iFit]
						yFit = binnedData[iBin, iFit] - background_interpolation(xFit)

						if np.max(yFit)<1:
							print('bin ' + str(iBin) + ' missing')
							continue


						# bds = [[0, min(xFit), -np.inf, -np.inf],
						#		[max(yFit)*100, max(xFit), np.inf, np.inf]]
						# if pure_gaussian == True:
						#	 if h>0:
						#		 p0 = [max(yFit), l0, fit[0][2], fit[0][3]]
						#	 else:
						#		 p0 = [max(yFit), l0, 0, yFit[[0]]]   #  A, sig, x0, m,y0
						#	 try:
						#		 fit = curve_fit(InstrFun_biased, xFit, yFit, p0, maxfev=100000000, bounds=bds)
						#	 except:
						#		 fit = [[0, 0, 0, 0]]
						#		 print('bin ' + str(iBin) + ' line ' + str(h + 4) + ' line missing')
						# else:
						#	 bgFit = np.polyfit(xFit[[0 -1]], yFit[[0, -1]], 1)
						#	 yFit -= np.polyval(bgFit, xFit)
						#	 p0 = [max(yFit), l0]
						#	 fit = curve_fit(InstrFun, xFit, yFit, p0)
						# dfs_lineRads[iTyp].loc[iBin][h] = fit[0][0] * peakUnitArea / binnedSens[
						#		 iTyp, iBin, iSens[iTyp][h]]
						# df_lineRad.loc[iBin][h + 1] = fit[0][0] * peakUnitArea / binnedSens[
						#	 iTyp, iBin, iSens[iTyp][h]]
						# # plt.figure()
						# # plt.plot(xFit, yFit)
						# # plt.plot(xFit, InstrFun_biased(xFit, *fit[0]))
						# # plt.pause(0.001)



						# bds = [[0,min_sigma, min(xFit), -np.inf, -np.inf],
						#		[max(yFit)*100,max_sigma, max(xFit), np.inf, np.inf]]
						# bds = [[0,min_sigma, l0-waveLcoefs[1][1]*30, -np.inf, -np.inf],
						#		[max(yFit)*100,max_sigma, l0+waveLcoefs[1][1]*30, np.inf, np.inf]]
						# bds = [[0,min_sigma, l0-waveLcoefs[1][1]*30, -1/np.max(xFit), -np.inf],
						#		[max(yFit)*100,max_sigma, l0+waveLcoefs[1][1]*30, np.inf, np.inf]]
						# bds = [[0,min_sigma, l0-waveLcoefs[1][1]*30, -np.inf,-np.inf, -np.inf],
						#		[max(yFit)*100,max_sigma, l0+waveLcoefs[1][1]*30, 0,np.inf, np.inf]]
						bds = [[0,min_sigma, l0-waveLcoefs[1][1]*30, -0.000001,-0.000001, -0.000001],
							   [max(yFit)*100,max_sigma, l0+waveLcoefs[1][1]*30, 0.000001,0.000001, 0.000001]]
						if pure_gaussian == True:
							if h>0:
								p0 = [max(yFit), fit[0][1],l0, *fit[0][3:]]
							else:
								# p0 = [max(yFit),(min_sigma+max_sigma)/2, l0, (np.mean(yFit[-10:-1])-np.mean(yFit[0:10]))/(xFit[-5] - xFit[5]),np.mean(yFit[-10:-1])- xFit[5]*(np.mean(yFit[-10:-1])-np.mean(yFit[0:10]))/(xFit[-5] - xFit[5])]
								# p0 = [max(yFit),(min_sigma+max_sigma)/2, l0, 0,np.mean(yFit[-10:-1])- xFit[5]*(np.mean(yFit[-10:-1])-np.mean(yFit[0:10]))/(xFit[-5] - xFit[5])]
								# p0 = [max(yFit),(min_sigma+max_sigma)/2, l0, 0,0,np.mean(yFit[-10:-1])]
								p0 = [max(yFit),(min_sigma+max_sigma)/2, l0, 0,0,0]
							try:
								fit = curve_fit(full_InstrFun_biased_quad, xFit, yFit, p0, maxfev=10000000, bounds=bds)
								# fit = curve_fit(full_InstrFun_biased, xFit, yFit, p0, maxfev=10000000, bounds=bds)
								pixel_location_of_the_peak = (np.abs(lamb[iTyp] - fit[0][2])).argmin()
								wavel_location_of_the_peak = fit[0][2]
								# sensitivity = binnedSens[iTyp, iBin, (np.abs(lamb[iTyp] - fit[0][2])).argmin()]
								if  ( (np.min(np.abs(time - np.array([-0.2,0.2,0.55,1])))<=0.06) and (iBin in [5,20,30]) ):
									plt.plot(xFit,full_InstrFun_biased_quad(xFit,*fit[0])+background_interpolation(xFit),label='n='+str(h+first+3)+'->2, line at %.5g nm, ' % wavel_location_of_the_peak + ' instead of %.5g nm' % l0 + ', sigma=%.4g nm' % fit[0][1])
									plt.plot(xFit,background_interpolation(xFit),'k--')
							except:
								fit = [[0,1,0,0,0]]
								# sensitivity = binnedSens[iTyp, iBin, iSens[iTyp][h]]
								pixel_location_of_the_peak = iSens[iTyp][h]
								wavel_location_of_the_peak = l0
								print('bin '+str(iBin)+' line '+str(h+4)+' line missing')
						else:
							bgFit = np.polyfit(xFit[[0 -1]], yFit[[0, -1]], 1)
							yFit -= np.polyval(bgFit, xFit)
							p0 = [max(yFit), l0]
							fit = curve_fit(InstrFun, xFit, yFit, p0)
							pixel_location_of_the_peak = iSens[iTyp][h]
							wavel_location_of_the_peak = l0
							# sensitivity = binnedSens[iTyp, iBin, iSens[iTyp][h]]
						if perform_convolution==True:
							sensitivity = binnedSens[iTyp, iBin, iFit]
							gaussian_weigths = full_InstrFun(xFit,1,fit[0][1],wavel_location_of_the_peak)/np.sum(full_InstrFun(xFit,1,fit[0][1],wavel_location_of_the_peak))
							sensitivity = np.sum(np.array(sensitivity)*gaussian_weigths)
						else:
							sensitivity = binnedSens[iTyp, iBin, pixel_location_of_the_peak]


						# dfs_lineRads[iTyp].loc[iBin][h] = np.abs(np.trapz(full_InstrFun_biased(xFit, *fit[0][:-2],0,0), x=xFit)) / binnedSens[iTyp, iBin, iSens[iTyp][h]]
						# df_lineRad.loc[iBin][h + 1] = np.abs(np.trapz(full_InstrFun_biased(lamb[1], *fit[0]), x=lamb[1])) / binnedSens[iTyp, iBin, iSens[iTyp][h]]
						# dfs_lineRads[iTyp].loc[iBin][h] = np.abs(np.trapz(full_InstrFun_biased_no_neg(xFit, *fit[0][:-2],0,0), x=xFit)) / sensitivity
						# df_lineRad.loc[iBin][h + 1] = np.abs(np.trapz(full_InstrFun_biased_no_neg(lamb[1], *fit[0]), x=lamb[1])) / sensitivity
						dfs_lineRads[iTyp].loc[iBin][h] = np.abs(np.trapz(full_InstrFun_biased_quad(xFit, *fit[0][:3],*np.zeros_like(fit[0][3:])), x=xFit)) / sensitivity
						df_lineRad.loc[iBin][h + 1] = np.abs(np.trapz(full_InstrFun_biased_quad(lamb[1], *fit[0][:3],*np.zeros_like(fit[0][3:])), x=lamb[1])) / sensitivity
						# if h == 3:
						# plt.figure()
						# plt.title('wavelength '+str(l0)+', line '+str(h+4)+'\n fit '+str(fit[0])+' uncertainty '+str(np.diag(fit[1])))
						# plt.plot(xFit, yFit)
						# # plt.plot(xFit, InstrFun(xFit,*fit[0])-np.polyval(bgFit, xFit))
						# plt.plot(xFit, full_InstrFun_biased(xFit, *fit[0]))
						# plt.pause(0.001)
						# if h==0:
						#	 print(fit[0][1])

						# record_all_sigma.append(fit[0][1])


					# iFit = np.logical_and(lamb[iTyp] > waveLengths[14]+np.diff(waveLengths)[14]/2, lamb[iTyp] < waveLengths[5]-np.diff(waveLengths)[4]/2)
					iFit = np.logical_and(lamb[iTyp] > waveLengths[total_number_of_lines_in_multipulse+first_line_in_multipulse+first-1]+np.diff(waveLengths)[total_number_of_lines_in_multipulse+first_line_in_multipulse+first-1]/2, lamb[iTyp] < waveLengths[first_line_in_multipulse+first]-np.diff(waveLengths)[first_line_in_multipulse+first-1]/2)
					iFit = np.logical_and(np.logical_and(iFit, np.arange(Nx)>20),np.arange(Nx)<Nx-20)	# added 16/02/2020 to afoid effects from tilting the image
					xFit = lamb[iTyp][iFit]
					yFit = binnedData[iBin, iFit]
					# bds = [[0,0,0,0,0,0,0,0,0,0, -np.inf, -np.inf],[max(yFit) * 100,max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100,np.inf, np.inf]]
					# bds_2 = [[0,0,0,0,0,0,0,0,0,0, *(np.array(waveLengths[5:15])-0.15), -np.inf, -np.inf],[max(yFit) * 100,max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100,*(np.array(waveLengths[5:15])+0.15),np.inf, np.inf]]
					# bds_3 = [[0,0,0,0,0,0,0,0,0,0, min_sigma, *(np.array(waveLengths[5:15])-0.2), -np.inf, -np.inf],[max(yFit) * 100,max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max_sigma,*(np.array(waveLengths[5:15])+0.2),np.inf, np.inf]]
					# bds_3 = [[*np.zeros((total_number_of_lines_in_multipulse)), min_sigma, *(np.array(waveLengths[first_line_in_multipulse+first:total_number_of_lines_in_multipulse+first_line_in_multipulse+first])-0.2), -np.inf, -np.inf, -np.inf],[*(np.ones((total_number_of_lines_in_multipulse))*max(yFit) * 100), max_sigma,*(np.array(waveLengths[first_line_in_multipulse+first:total_number_of_lines_in_multipulse+first_line_in_multipulse+first])+0.2),0,np.inf, np.inf]]
					if pure_gaussian == True:
						success=9
						while success!=-1:
							# print(success)
							try:
								ipks = find_peaks(yFit)[0]
								# A0 = yFit[ipks[[np.abs((l0 - xFit[ipks])).argmin() for l0 in waveLengths[5:15]]]].tolist()
								A0 = yFit[ipks[[np.abs((l0 - xFit[ipks])).argmin() for l0 in waveLengths[first_line_in_multipulse+first:total_number_of_lines_in_multipulse+first_line_in_multipulse+first]]]].tolist()
								# A0.append(fit[0][2])
								# A0.append(fit[0][3])
								# fit = curve_fit(multPeakFun_biased, xFit, yFit, A0, maxfev=100000000, bounds=bds)
								A0.append(max(min(mean_sigma,max_sigma),min_sigma))
								A0.extend(waveLengths[first_line_in_multipulse+first:total_number_of_lines_in_multipulse+first_line_in_multipulse+first])
								# A0.append(fit[0][3])
								# A0.append(fit[0][4])
								# A0.append(fit[0][5])
								# A0.extend(np.zeros((3)))
								y_max_base_noise = np.mean(np.sort(binnedData[iBin, np.logical_and(lamb[iTyp]>402,lamb[iTyp]<405)])[:int(np.sum(np.logical_and(lamb[iTyp]>402,lamb[iTyp]<405))/3)])
								y_max_base_noise_slope = np.mean(np.sort(binnedData[iBin, np.logical_and(lamb[iTyp]>391,lamb[iTyp]<394)])[:int(np.sum(np.logical_and(lamb[iTyp]>391,lamb[iTyp]<394))/3)])
								y_max_base_noise_slope_2 = np.mean(np.sort(binnedData[iBin, np.logical_and(lamb[iTyp]>385.1,lamb[iTyp]<386.5)])[:int(np.sum(np.logical_and(lamb[iTyp]>385.1,lamb[iTyp]<386.5))/3)])
								y_max_base_noise_slope_3 = np.mean(np.sort(binnedData[iBin, np.logical_and(lamb[iTyp]>374,lamb[iTyp]<375)])[:int(np.sum(np.logical_and(lamb[iTyp]>374,lamb[iTyp]<375))/3)])
								y_min_base_noise = np.mean(np.sort(binnedData[iBin, np.logical_and(lamb[iTyp]>np.min(xFit)-2,lamb[iTyp]<np.min(xFit)+2)])[:int(np.sum(np.logical_and(lamb[iTyp]>np.min(xFit)-2,lamb[iTyp]<np.min(xFit)+2))/3)])
								min_slope = np.max([0,(y_max_base_noise-y_max_base_noise_slope)/((401+404)/2 - (391+394)/2) , (y_max_base_noise-y_max_base_noise_slope_2)/((401+404)/2 - (385.1+386.5)/2) , (y_max_base_noise-y_max_base_noise_slope_3)/((401+404)/2 - (374+375)/2)  ])
								if fit[0][0]>3:	# I do it only if the last strong peak is actually strong
									min_slope=max(min_slope,np.diff(background_interpolation([350,351]))/1)
								A0.append(min(0.2*y_min_base_noise,y_max_base_noise-min_slope*(np.max(xFit) - np.min(xFit))))
								A0.append(y_max_base_noise)
								A0.append(max(0.1,min_slope))
								bds_3 = [[*np.zeros((total_number_of_lines_in_multipulse)), min_sigma, *(np.array(waveLengths[first_line_in_multipulse+first:total_number_of_lines_in_multipulse+first_line_in_multipulse+first])-0.5), -np.inf, 0.3*y_max_base_noise,min_slope],[*(np.ones((total_number_of_lines_in_multipulse))*max(yFit) * 100), max_sigma,*(np.array(waveLengths[first_line_in_multipulse+first:total_number_of_lines_in_multipulse+first_line_in_multipulse+first])+0.5),min(1.1*y_min_base_noise,y_max_base_noise-min_slope*(np.max(xFit) - np.min(xFit))),1*y_max_base_noise,np.inf]]
								if total_number_of_lines_in_multipulse==10:
									fit_multiline = curve_fit(multPeakFun_biased_10_3, xFit, yFit, A0, maxfev=10000000, bounds=bds_3)
								elif total_number_of_lines_in_multipulse==6:
									fit_multiline = curve_fit(multPeakFun_biased_6_3, xFit, yFit, A0, maxfev=10000000, bounds=bds_3)
								else:
									print('you asked to put '+str(total_number_of_lines_in_multipulse)+' lines in the multPeakFun_biased routine\nbut now it can accept only 6 or 10')
									exit()
								# peakUnitArea = np.abs(np.trapz(full_InstrFun(lamb[1], fit_multiline[0][total_number_of_lines_in_multipulse], 1, np.mean(lamb[1])), x=lamb[1]))
								success=-1
								# plt.figure()
								# plt.plot(xFit, yFit)
								# # plt.plot(xFit, multPeakFun(xFit, *fit[0]))
								# plt.plot(xFit, multPeakFun_biased_10(xFit, *fit_multiline[0]))
								# plt.pause(0.001)
								if  ( (np.min(np.abs(time - np.array([-0.2,0.2,0.55,1])))<=0.06) and (iBin in [5,20,30]) ):
									for iline in range(total_number_of_lines_in_multipulse):
										if fit_multiline[0][iline]<0:
											fit_multiline[0][iline]=0
										if fit_multiline[0][iline]>0:
											# temp_amplitudes = np.zeros((len(waveLengths[5:15])))
											temp_amplitudes = np.zeros((total_number_of_lines_in_multipulse))
											temp_amplitudes[iline]=fit_multiline[0][iline]
											if total_number_of_lines_in_multipulse==10:
												plt.plot(xFit,multPeakFun_biased_10_3(xFit,*temp_amplitudes,*fit_multiline[0][total_number_of_lines_in_multipulse:]),label='n='+str(iline+3+first+first_line_in_multipulse)+'->2, line at %.5g nm, ' % fit_multiline[0][total_number_of_lines_in_multipulse+1+iline] + ' instead of %.5g nm' % A0[total_number_of_lines_in_multipulse+1+iline])
											elif total_number_of_lines_in_multipulse==6:
												plt.plot(xFit,multPeakFun_biased_6_3(xFit,*temp_amplitudes,*fit_multiline[0][total_number_of_lines_in_multipulse:]),label='n='+str(iline+3+first+first_line_in_multipulse)+'->2, line at %.5g nm, ' % fit_multiline[0][total_number_of_lines_in_multipulse+1+iline] + ' instead of %.5g nm' % A0[total_number_of_lines_in_multipulse+1+iline])
											else:
												print('you asked to put '+str(total_number_of_lines_in_multipulse)+' lines in the multPeakFun_biased routine\nbut now it can accept only 6 or 10')
												exit()
									extended_y_min = (fit_multiline[0][-3] - fit_multiline[0][-2] - fit_multiline[0][-1]*(np.min(xFit)-np.max(xFit)))*((np.min(lamb[iTyp][lamb[iTyp]>350])-np.max(xFit))**2)/((np.min(xFit)-np.max(xFit))**2) + fit_multiline[0][-1]*(np.min(lamb[iTyp][lamb[iTyp]>350])-np.max(xFit))+fit_multiline[0][-2]
									if total_number_of_lines_in_multipulse==10:
										plt.plot(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],multPeakFun_biased_10_3(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],*fit_multiline[0][:-3],extended_y_min,*fit_multiline[0][-2:]),'r--',linewidth=0.6,label='sum of n='+str(3+first+first_line_in_multipulse)+'/'+str(3+first+total_number_of_lines_in_multipulse+first_line_in_multipulse-1)+'->2, sigma=%.4g nm' % fit_multiline[0][total_number_of_lines_in_multipulse])
										plt.plot(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],multPeakFun_biased_10_3(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],*np.zeros((total_number_of_lines_in_multipulse)),*fit_multiline[0][total_number_of_lines_in_multipulse:-3],extended_y_min,*fit_multiline[0][-2:]),'k--',label='background, quadratic coeff on multipeak %.5g' % ((fit_multiline[0][-3] - fit_multiline[0][-2] - fit_multiline[0][-1]*(np.min(xFit)-np.max(xFit)))/((np.min(xFit)-np.max(xFit))**2)) )
									elif total_number_of_lines_in_multipulse==6:
										plt.plot(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],multPeakFun_biased_6_3(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],*fit_multiline[0][:-3],extended_y_min,*fit_multiline[0][-2:]),'r--',linewidth=0.6,label='sum of n='+str(3+first+first_line_in_multipulse)+'/'+str(3+first+total_number_of_lines_in_multipulse+first_line_in_multipulse-1)+'->2, sigma=%.4g nm' % fit_multiline[0][total_number_of_lines_in_multipulse])
										plt.plot(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],multPeakFun_biased_6_3(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],*np.zeros((total_number_of_lines_in_multipulse)),*fit_multiline[0][total_number_of_lines_in_multipulse:-3],extended_y_min,*fit_multiline[0][-2:]),'k--',label='background, quadratic coeff on multipeak %.5g' % ((fit_multiline[0][-3] - fit_multiline[0][-2] - fit_multiline[0][-1]*(np.min(xFit)-np.max(xFit)))/((np.min(xFit)-np.max(xFit))**2)) )
									else:
										print('you asked to put '+str(total_number_of_lines_in_multipulse)+' lines in the multPeakFun_biased routine\nbut now it can accept only 6 or 10')
										exit()
									# plt.plot(xFit,multPeakFun_biased_10(xFit,np.zeros(iline-1)*fit_multiline[0]),label='n=8-18->2')
							except:
								bds_3[0][success] = -abs(max(yFit)) * 100
								if success<=4:
									print('bin '+str(iBin)+' line '+str(success+first_line_in_multipulse+1+3)+' constrain on positive peak removed')
								success -= 1
								fit_multiline = np.array([A0])
								fit_multiline[0,:total_number_of_lines_in_multipulse]=0
						for iline in range(total_number_of_lines_in_multipulse):
							if fit_multiline[0][iline]<0:
								fit_multiline[0][iline]=0
						pixel_location_of_the_peak = (np.abs(lamb[iTyp] - np.array([fit_multiline[0][total_number_of_lines_in_multipulse+1:2*total_number_of_lines_in_multipulse+1]]).T)).argmin(axis=1)
						wavel_location_of_the_peak = fit_multiline[0][total_number_of_lines_in_multipulse+1:2*total_number_of_lines_in_multipulse+1]
						# sensitivity = binnedSens[iTyp, iBin, (np.abs(lamb[iTyp] - np.array([fit_multiline[0][11:21]]).T)).argmin(axis=1)]
					else:
						bgFit = np.polyfit(xFit[[0, -1]], yFit[[0, -1]], 1)
						yFit -= np.polyval(bgFit, xFit)
						ipks = find_peaks(yFit)[0]
						A0 = yFit[ipks[[(np.abs(l0 - xFit[ipks])).argmin() for l0 in waveLengths[first_line_in_multipulse+first:total_number_of_lines_in_multipulse+first_line_in_multipulse+first]]]]
						fit = curve_fit(multPeakFun, xFit, yFit, A0)
						sensitivity = binnedSens[iTyp, iBin, iSens[iTyp][first:10]]

					if perform_convolution==True:
						sensitivity = binnedSens[iTyp, iBin]
						temp = []
						for line_index,line in enumerate(wavel_location_of_the_peak):
							gaussian_weigths = full_InstrFun(lamb[1],1,fit_multiline[0][total_number_of_lines_in_multipulse],line)/np.sum(full_InstrFun(lamb[1],1,fit_multiline[0][total_number_of_lines_in_multipulse],line))
							temp.append(np.sum(sensitivity * gaussian_weigths))
						sensitivity = np.array(temp)
					else:
						sensitivity = binnedSens[iTyp, iBin, pixel_location_of_the_peak]
					peakUnitArea = np.abs(np.trapz(full_InstrFun(lamb[1], fit_multiline[0][total_number_of_lines_in_multipulse], 1, np.mean(wavel_location_of_the_peak)), x=lamb[1]))


					# plt.figure()
					# plt.plot(xFit, yFit)
					# # plt.plot(xFit, multPeakFun(xFit, *fit[0]))
					# plt.plot(xFit, multPeakFun_biased_10(xFit, *fit[0]))
					# plt.pause(0.001)
					if first < 6:
						# dfs_lineRads[iTyp].loc[iBin][-5:] = fit[0][:5]*peakUnitArea / binnedSens[iTyp, iBin, iSens[iTyp][5:10]]
						# df_lineRad.loc[iBin][-5:] = fit[0][:5]*peakUnitArea / binnedSens[iTyp, iBin, iSens[iTyp][5:10]]
						# dfs_lineRads[iTyp].loc[iBin][-5:] = fit_multiline[0][:5]*peakUnitArea / sensitivity[5:10]
						# df_lineRad.loc[iBin][-5:] = fit_multiline[0][:5]*peakUnitArea / sensitivity[5:10]
						dfs_lineRads[iTyp].loc[iBin][first_line_in_multipulse:] = fit_multiline[0][:len(dfs_lineRads[iTyp].loc[iBin])-first_line_in_multipulse]*peakUnitArea / sensitivity[:len(dfs_lineRads[iTyp].loc[iBin])-first_line_in_multipulse]
						df_lineRad.loc[iBin][first_line_in_multipulse+first:] = fit_multiline[0][:len(dfs_lineRads[iTyp].loc[iBin])-first_line_in_multipulse]*peakUnitArea / sensitivity[:len(dfs_lineRads[iTyp].loc[iBin])-first_line_in_multipulse]
					else:
						print('this part of code is not updated')
						exit()
						# dfs_lineRads[iTyp].loc[iBin][first - 10:] = fit[0][first - 5:5]*peakUnitArea / binnedSens[iTyp, iBin, iSens[iTyp][first:10]]
						# df_lineRad.loc[iBin][first - 10:] = fit[0][first - 5:5]*peakUnitArea / binnedSens[iTyp, iBin, iSens[iTyp][first:10]]
						# dfs_lineRads[iTyp].loc[iBin][first - 10:] = fit[0][first - 5:5]*peakUnitArea / sensitivity[first:10]
						# df_lineRad.loc[iBin][first - 10:] = fit[0][first - 5:5]*peakUnitArea / sensitivity[first:10]
						dfs_lineRads[iTyp].loc[iBin][first - 10:] = fit[0][first - 5:5]*peakUnitArea / sensitivity[first:10]
						df_lineRad.loc[iBin][first - 10:] = fit[0][first - 5:5]*peakUnitArea / sensitivity[first:10]
				except:
					print('bin '+str(iBin)+' fit failed')
				if  ( (np.min(np.abs(time - np.array([-0.2,0.2,0.55,1])))<=0.06) and (iBin in [5,20,30]) ):
					print('file 1   '+path_where_to_save_everything+'/spectral_fit_example'+'/fit_time_' + str(np.around(time, decimals=3)) +'_ms_bin_'+str(iBin)+ '.eps')
					plt.title('Emission profile fitting for time '+str(time)+' ms and bin '+str(iBin) + ' sigma range %.5g %.5g mm' % (min_sigma,max_sigma) )
					plt.grid()
					plt.legend(loc='best')
					plt.xlabel('wavelength [nm]')
					plt.ylabel('counts [au]')
					plt.ylim(np.min(binnedData[iBin][:-100]),np.max(binnedData[iBin]))
					print('file 2   '+path_where_to_save_everything+'/spectral_fit_example'+'/fit_time_' + str(np.around(time, decimals=3)) +'_ms_bin_'+str(iBin)+ '.eps')
					plt.semilogy()
					save_done = 0
					save_index=1
					try:	# All this trie are because of a known issues when saving lots of figure inside of multiprocessor
						plt.savefig(path_where_to_save_everything+'/spectral_fit_example'+'/fit_time_' + str(np.around(time, decimals=3)) +'_ms_bin_'+str(iBin)+ '.eps',bbox_inches='tight')
					except Exception as e:
						print(path_where_to_save_everything+'/spectral_fit_example'+'/fit_time_' + str(np.around(time, decimals=3)) +'_ms_bin_'+str(iBin)+ '.eps save try number '+str(1)+' failed. Reason %s' % e)
						while save_done==0 and save_index<100:
							try:
								tm.sleep(np.random.random()**2)
								plt.savefig(path_where_to_save_everything+'/spectral_fit_example'+'/fit_time_' + str(np.around(time, decimals=3)) +'_ms_bin_'+str(iBin)+ '.eps',bbox_inches='tight')
								print(path_where_to_save_everything+'/spectral_fit_example'+'/fit_time_' + str(np.around(time, decimals=3)) +'_ms_bin_'+str(iBin)+ '.eps save try number '+str(save_index+1)+' successfull')
								save_done=1
							except Exception as e:
								print(path_where_to_save_everything+'/spectral_fit_example'+'/fit_time_' + str(np.around(time, decimals=3)) +'_ms_bin_'+str(iBin)+ '.eps save try number '+str(save_index+1)+' failed. Reason %s' % e)
								save_index+=1
					plt.close()
					print('file 3   '+path_where_to_save_everything+'/spectral_fit_example'+'/fit_time_' + str(np.around(time, decimals=3)) +'_ms_bin_'+str(iBin)+ '.eps')
		dfs_totIntens[iTyp].loc[iSet] = dfs_lineRads[iTyp].sum(skipna=0)
		df_lineRad.n3 = dfs_lineRads[0].n3
		# df_lineRad.to_csv('Radiance/%i.csv' % iSet)  # Write to file


		return dfs_lineRads[iTyp]
# if typ != 'Ha':
# plt.figure()
# plt.plot(xFit, yFit)
# plt.plot(xFit, multPeakFun(xFit, *fit[0]))
# plt.pause(0.001)
# plt.plot(xFit,multPeakFun(xFit,*A0))
# # plt.plot(xFit,np.polyval(bgFit,xFit))
'''
plt.figure()
plt.plot(dfs_lineRads[1].n5)
plt.plot(dfs_lineRads[2].n5)
plt.plot(df_lineRad.n5)
plt.pause(0.001)
'''


# Check variation of Hb between the two exposure ranges;
# Variation of high lines between two exposure times looks OK
# Rigid Instrument function gives semi-poor fits with structural ~5% area error; acceptable for now

# Fit instrument function on different peaks
'''
lineShapeCoefs = np.zeros([5,5])
for h,l0 in enumerate(waveLengths[1:6]):
	iFit = np.logical_and(lamb > l0-5, lamb < l0+5)
	xFit = lamb[iFit]
	yFit = binnedData[20,iFit]
	if h==4:
		xFit = xFit[:76]
		yFit = yFit[:76]
	bgFit = np.polyfit(xFit[[0,-1]],yFit[[0,-1]],1)
	yFit -= np.polyval(bgFit,xFit)
	p0 = [max(yFit),max(yFit)/2,1.5,.2,l0]

	fit = curve_fit(shapeFun,xFit,yFit,p0)

	plt.figure();
	plt.plot(xFit,yFit)
	plt.plot(xFit,shapeFun(xFit,*fit[0]))
	#plt.plot(xFit,shapeFun(xFit,*p0))
	lineShapeCoefs[h] = fit[0]
'''


def doSpecFit_single_frame_with_sigma(binnedData,binnedData_sigma,df_settings, df_log, geom, waveLcoefs, binnedSens,binnedSens_sigma,path_where_to_save_everything,time,to_print, type='Hb', pure_gaussian=True,perform_convolution=True,first_line_in_multipulse=3,total_number_of_lines_in_multipulse=10,binnedSens_waveLcoefs='auto'):
	import matplotlib.pyplot as plt

	nCh,Nx = binnedData.shape
	if not (binnedSens_waveLcoefs!='auto'):
		binnedSens_waveLcoefs = waveLcoefs

	if pure_gaussian==True:
		InstrFun = lambda x, A, x0:(1/np.sqrt(2*np.pi*(0.85**2)))*A*np.exp(-np.power(x-x0,2)/(2*(0.85**2)))
		multPeakFun = lambda x, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10: sum([InstrFun(x, Ai, x0) for Ai, x0 in zip([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10], waveLengths[5:15])])
		InstrFun_biased = lambda x, A, x0, m,y0:InstrFun(x, A, x0) + m*x+y0
		multPeakFun_biased = lambda x, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10,m,y0: sum([InstrFun(x, Ai, x0) for Ai, x0 in zip([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10], waveLengths[5:15])])+m*x+y0
		multPeakFun_biased_2 = lambda x, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10,x01,x02,x03,x04,x05,x06,x07,x08,x09,x010, m, y0: sum([InstrFun(x, Ai, x0i) for Ai, x0i in zip([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10], [x01, x02, x03, x04, x05, x06, x07, x08, x09, x010])]) + m * x + y0
		full_InstrFun = lambda x, A, sig, x0:(1/np.sqrt(2*np.pi*(sig**2)))*A*np.exp(-np.power(x-x0,2)/(2*(sig**2)))
		full_InstrFun_biased = lambda x, A, sig, x0, m,y0:full_InstrFun(x, A, sig, x0) + m*x+y0
		full_InstrFun_biased_quad = lambda x, A, sig, x0, m1,m,y0:full_InstrFun(x, A, sig, x0) + m1*(x-x0)**2 + m*(x-x0)+y0
		full_InstrFun_biased_no_neg = lambda x, A, sig, x0, c,y0:full_InstrFun(x, A, sig, x0) + y0*c*x+y0
		multPeakFun_biased_10 = lambda x, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, sig, x01,x02,x03,x04,x05,x06,x07,x08,x09,x010, m1,m, y0: sum([full_InstrFun(x, Ai, sigi, x0i) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10], [sig, sig, sig, sig, sig, sig, sig, sig, sig, sig], [x01, x02, x03, x04, x05, x06, x07, x08, x09, x010])]) + m1*(x-np.mean([x01,x02,x03,x04,x05,x06,x07,x08,x09,x010]))**2 + m * (x-np.mean([x01,x02,x03,x04,x05,x06,x07,x08,x09,x010])) + y0
		multPeakFun_biased_10_3 = lambda x, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, sig, x01,x02,x03,x04,x05,x06,x07,x08,x09,x010, y_min,y_max, m,P: sum([full_InstrFun(x, Ai, sigi, x0i) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10], [sig, sig, sig, sig, sig, sig, sig, sig, sig, sig], [x01, x02, x03, x04, x05, x06, x07, x08, x09, x010])]) + np.max([np.ones_like(x)*max(y_min,0),((y_min-y_max-m*(np.min(x)-np.max(x)))/(((np.min(x)-np.max(x))**2)**P))*(((x-np.max(x))**2)**P) + m* (x-np.max(x)) + y_max],axis=0)
		multPeakFun_biased_6 = lambda x, A1, A2, A3, A4, A5, A6, sig, x01,x02,x03,x04,x05, x06, m1,m, y0: sum([full_InstrFun(x, Ai, sigi, x0i) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6], [sig, sig, sig, sig, sig, sig], [x01, x02, x03, x04, x05, x06])]) + m1*(x-(x01+x06)/2)**2 + m * (x-(x01+x06)/2) + y0
		multPeakFun_biased_6_1 = lambda x, A1, A2, A3, A4, A5, A6, sig, x01,x02,x03,x04,x05, x06, m1,m, y0,x_ref: sum([full_InstrFun(x, Ai, sigi, x0i) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6], [sig, sig, sig, sig, sig, sig], [x01, x02, x03, x04, x05, x06])]) + m1*(x-x_ref)**2 + m * (x-x_ref) + y0
		multPeakFun_biased_6_2 = lambda x, A1, A2, A3, A4, A5, A6, sig, x01,x02,x03,x04,x05, x06, m1,c, y0: sum([full_InstrFun(x, Ai, sigi, x0i) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6], [sig, sig, sig, sig, sig, sig], [x01, x02, x03, x04, x05, x06])]) + m1*(x-np.max(x))**2 + (-m1*(np.min(x)-np.max(x)) -y0/(np.min(x)-np.max(x)) - c) * (x-np.max(x)) + y0
		multPeakFun_biased_6_3 = lambda x, A1, A2, A3, A4, A5, A6, sig, x01,x02,x03,x04,x05, x06, y_min,y_max, m,P: sum([full_InstrFun(x, Ai, sigi, x0i) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6], [sig, sig, sig, sig, sig, sig], [x01, x02, x03, x04, x05, x06])]) + np.max([np.ones_like(x)*max(y_min,0),((y_min-y_max-m*(np.min(x)-np.max(x)))/(((np.min(x)-np.max(x))**2)**P))*(((x-np.max(x))**2)**P) + m* (x-np.max(x)) + y_max],axis=0)

		full_InstrFun_sigma = lambda x, A, gaus_sig, x0,A_sig,gaus_sig_sig,x0_sig:full_InstrFun(x, A, gaus_sig, x0) * np.sqrt( (A_sig/A)**2 + (2*(x-x0)*x0_sig/(gaus_sig**2))**2 + (2*((x-x0)**2)*gaus_sig_sig/(gaus_sig**3))**2 )
		full_InstrFun_with_error_included = lambda x, A, sig, x0:(1/sqrt(2*np.pi*(sig**2)))*A*exp(-np.power((x-x0)/sig,2)/2)
		multPeakFun_biased_10_3_with_error_included = lambda x, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, sig, x01,x02,x03,x04,x05,x06,x07,x08,x09,x010, y_min,y_max, m,P: sum([full_InstrFun_with_error_included(x, Ai, sigi, x0i) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10], [sig, sig, sig, sig, sig, sig, sig, sig, sig, sig], [x01, x02, x03, x04, x05, x06, x07, x08, x09, x010])]) + np.max([np.ones_like(x)*max(y_min,0),((y_min-y_max-m*(np.min(x)-np.max(x)))/(((np.min(x)-np.max(x))**2)**P))*(((x-np.max(x))**2)**P) + m* (x-np.max(x)) + y_max],axis=0)
		multPeakFun_biased_6_3_with_error_included = lambda x, A1, A2, A3, A4, A5, A6, sig, x01,x02,x03,x04,x05, x06, y_min,y_max, m,P: sum([full_InstrFun_with_error_included(x, Ai, sigi, x0i) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6], [sig, sig, sig, sig, sig, sig], [x01, x02, x03, x04, x05, x06])]) + np.max([np.ones_like(x)*max(y_min,0),((y_min-y_max-m*(np.min(x)-np.max(x)))/(((np.min(x)-np.max(x))**2)**P))*(((x-np.max(x))**2)**P) + m* (x-np.max(x)) + y_max],axis=0)

		super_gaussian = lambda x, A, sig, x0,P:(1/np.sqrt(2*np.pi*(sig**2)))*A*np.exp(-np.power(np.power(x-x0,2)/(2*(sig**2)),P))
		stark_gaussian = lambda x, A, sig, x0,dist,P:(1/np.sqrt(2*np.pi*(sig**2)))*A*np.exp(-np.power(np.power(x-(x0+dist),2)/(2*(sig**2)),P)) + (1/np.sqrt(2*np.pi*(sig**2)))*A*np.exp(-np.power(np.power(x-(x0-dist),2)/(2*(sig**2)),P))
		gauss_plus_stark =lambda x, A, sig, x0,P, A1_A, sig_sig, dist_sig:super_gaussian(x, A, sig, x0,P) + stark_gaussian(x,A*A1_A,sig*sig_sig,x0,sig*dist_sig,P)
		def multPeakFun_biased_6_3_stark(ext_Ai_A, ext_sigi_sig, ext_disti_sig,fixed_pos):
			function = lambda x, A1, A2, A3, A4, A5, A6, sig, y_min,y_max, m,P: sum([full_InstrFun(x, Ai, sigi, x0i)+stark_gaussian(x,Ai_A*Ai,sigi_sig*sigi,x0i,disti_sig*sigi_sig*sigi,1)*(stark_gaussian(x,Ai_A*Ai,sigi_sig*sigi,x0i,disti_sig*sigi_sig*sigi,1)-full_InstrFun(x, Ai, sigi, x0i)>0) for Ai, sigi, x0i, Ai_A, sigi_sig, disti_sig in zip([A1, A2, A3, A4, A5, A6], [sig, sig, sig, sig, sig, sig], fixed_pos, ext_Ai_A, ext_sigi_sig, ext_disti_sig)]) + np.max([np.ones_like(x)*max(y_min,0),((y_min-y_max-m*(np.min(x)-np.max(x)))/(((np.min(x)-np.max(x))**2)**P))*(((x-np.max(x))**2)**P) + m* (x-np.max(x)) + y_max],axis=0)
			return function
		def multPeakFun_biased_10_3_stark(ext_Ai_A, ext_sigi_sig, ext_disti_sig,fixed_pos):
			function = lambda x, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, sig, y_min,y_max, m, P: sum([full_InstrFun(x, Ai, sigi, x0i)+stark_gaussian(x,Ai_A*Ai,sigi_sig*sigi,x0i,disti_sig*sigi_sig*sigi,1)*(stark_gaussian(x,Ai_A*Ai,sigi_sig*sigi,x0i,disti_sig*sigi_sig*sigi,1)-full_InstrFun(x, Ai, sigi, x0i)>0) for Ai, sigi, x0i, Ai_A, sigi_sig, disti_sig in zip([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10], [sig, sig, sig, sig, sig, sig, sig, sig, sig, sig], fixed_pos, ext_Ai_A, ext_sigi_sig, ext_disti_sig)]) + np.max([np.ones_like(x)*max(y_min,0),((y_min-y_max-m*(np.min(x)-np.max(x)))/(((np.min(x)-np.max(x))**2)**P))*(((x-np.max(x))**2)**P) + m* (x-np.max(x)) + y_max],axis=0)
			return function

		def multPeakFun_biased_10_4(ext_peak_locations):
			function = lambda x, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, sig, delta_x0, y_min,y_max, m,P: sum([full_InstrFun(x, Ai, sigi, x0i+delta_x0) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10], [sig, sig, sig, sig, sig, sig, sig, sig, sig, sig], ext_peak_locations)]) + np.max([np.ones_like(x)*max(min(y_min,y_max),0),((y_min-y_max-m*(np.min(x)-np.max(x)))/(((np.min(x)-np.max(x))**2)**P))*(((x-np.max(x))**2)**P) + m* (x-np.max(x)) + y_max],axis=0)
			return function
		def multPeakFun_10_4(ext_peak_locations):
			function = lambda x, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, sig, delta_x0: sum([full_InstrFun(x, Ai, sigi, x0i+delta_x0) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10], [sig, sig, sig, sig, sig, sig, sig, sig, sig, sig], ext_peak_locations)])
			return function
		def multPeakFun_biased_only_bias():
			function = lambda x, x_min,x_max, y_min,y_max, m,P: np.max([np.ones_like(x)*max(min(y_min,y_max),0),((y_min-y_max-m*(x_min-x_max))/(((x_min-x_max)**2)**P))*(((x-x_max)**2)**P) + m* (x-x_max) + y_max],axis=0)
			return function
		def multPeakFun_biased_6_4(ext_peak_locations):
			function = lambda x, A1, A2, A3, A4, A5, A6, sig, delta_x0, y_min,y_max, m,P: sum([full_InstrFun(x, Ai, sigi, x0i+delta_x0) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6], [sig, sig, sig, sig, sig, sig], ext_peak_locations)]) + np.max([np.ones_like(x)*max(min(y_min,y_max),0),((y_min-y_max-m*(np.min(x)-np.max(x)))/(((np.min(x)-np.max(x))**2)**P))*(((x-np.max(x))**2)**P) + m* (x-np.max(x)) + y_max],axis=0)
			return function
		def multPeakFun_biased_10_4_with_error_included(ext_peak_locations):
			function = lambda x, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, sig, delta_x0, y_min,y_max, m,P: sum([full_InstrFun_with_error_included(x, Ai, sigi, x0i+delta_x0) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10], [sig, sig, sig, sig, sig, sig, sig, sig, sig, sig], ext_peak_locations)]) + np.max([np.ones_like(x)*max(min(y_min,y_max),0),((y_min-y_max-m*(np.min(x)-np.max(x)))/(((np.min(x)-np.max(x))**2)**P))*(((x-np.max(x))**2)**P) + m* (x-np.max(x)) + y_max],axis=0)
			return function
		def multPeakFun_10_4_with_error_included(ext_peak_locations):
			function = lambda x, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, sig, delta_x0: sum([full_InstrFun_with_error_included(x, Ai, sigi, x0i+delta_x0) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10], [sig, sig, sig, sig, sig, sig, sig, sig, sig, sig], ext_peak_locations)])
			return function
		def multPeakFun_biased_6_4_with_error_included(ext_peak_locations):
			function = lambda x, A1, A2, A3, A4, A5, A6, sig, delta_x0, y_min,y_max, m,P: sum([full_InstrFun_with_error_included(x, Ai, sigi, x0i+delta_x0) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6], [sig, sig, sig, sig, sig, sig], ext_peak_locations)]) + np.max([np.ones_like(x)*max(y_min,0),((y_min-y_max-m*(np.min(x)-np.max(x)))/(((np.min(x)-np.max(x))**2)**P))*(((x-np.max(x))**2)**P) + m* (x-np.max(x)) + y_max],axis=0)
			return function
		# def multPeakFun_biased_6_1(x_max,x_min):
		#	 multPeakFun_biased_6 = lambda x, A1, A2, A3, A4, A5, A6, sig, x01,x02,x03,x04,x05, x06, m1,m, y0: sum([full_InstrFun(x, Ai, sigi, x0i) for Ai, sigi, x0i in zip([A1, A2, A3, A4, A5, A6], [sig, sig, sig, sig, sig, sig], [x01, x02, x03, x04, x05, x06])]) + m1*(x-(x01+x06)/2)**2 + m * (x-(x01+x06)/2) + y0
		#	 def polyadd(x, *params):
		#		 temp=0
		#		 for i in range(n):
		#			 temp+=params[i]*x**i
		#		 return temp
		#	 return polyadd


	else:
		blockGauss = lambda x, A, w, sig, x0: A * (erf((x - x0 + w) / sig) - erf((x - x0 - w) / sig)) / 2 * np.sqrt(np.pi)
		triGauss = lambda x, A, a, sig, x0: A * sig / a * ((np.exp(-((x - x0) / sig) ** 2) - np.exp(-((x - x0 - a) / sig) ** 2)) / 2 + (x - x0) / sig * np.sqrt(np.pi) / 2 * (erf((a - x + x0) / sig) + erf((x - x0) / sig)))
		# shapeFun = lambda x,A1,A2,w,sig,x0 : (triGauss(x,(A2-A1),2*w,sig,x0) + blockGauss(x,A1,w,sig,x0+w))/1.77
		# InstrFun = lambda x,A,x0 : (triGauss(x,(-.6*A),2*1.86,.31,x0) + blockGauss(x,A,1.86,.31,x0+1.86))/1.77
		# InstrFun = lambda x, A, x0: (triGauss(x, (-.75 * A), 2 * 1.46, .5, x0) + blockGauss(x, A, 1.46, .5,x0 + 1.46)) / 1.77
		InstrFun = lambda x, A, x0: (triGauss(x, (-.75 * A), 2 * 1.46, .5, x0) + blockGauss(x, A, 1.46, .5,x0 + 1.46)) / 1.77
		# all_InstrFun = lambda x, A1, w1, sig1, x01, A2, a2, sig2, x02: (triGauss(x, (-.75 * A2), a2, sig2, x02) + blockGauss(x, A1, w1, sig1,x01))
		# all_InstrFun = lambda x, A1, w1, sig1, x01, A2, a2, sig2, x02: (triGauss(x, (-.75 * A2), a2, sig2, x02) + blockGauss(x, A1, w1, sig1,x01))
		multPeakFun = lambda x, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10: sum([InstrFun(x, Ai, x0 - 0.207) for Ai, x0 in zip([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10], waveLengths[5:15])])
		InstrFun_biased = lambda x, A, x0, m,y0:InstrFun(x, A, x0) + m*x+y0
		multPeakFun_biased = lambda x, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10,m,y0: sum([InstrFun(x, Ai, x0) for Ai, x0 in zip([A1, A2, A3, A4, A5, A6, A7, A8, A9, A10], waveLengths[5:15])])+m*x+y0

	# if np.random.rand()<0.1:
	if to_print:
		if not os.path.exists(path_where_to_save_everything+'/spectral_fit_example'):
			os.makedirs(path_where_to_save_everything+'/spectral_fit_example')


	n = np.arange(10, 20)
	waveLengths = [656.45377, 486.13615, 434.0462, 410.174, 397.0072, 388.9049, 383.5384] + list(
		(364.6 * (n * n) / (n * n - 4)))
	lamb = np.array([np.polyval(waveLcoefs[0], np.arange(Nx)), np.polyval(waveLcoefs[1], np.arange(Nx)),
					 np.polyval(waveLcoefs[1], np.arange(Nx))])

	peakUnitArea = np.abs(np.trapz(InstrFun(lamb[1], 1, np.mean(lamb[1])), x=lamb[1]))

	linesSeen = [range(2), range(1, 11), range(1, 11)]
	iSens = [[np.abs(lamb[i] - waveLengths[line]).argmin() for line in lines] for i, lines in enumerate(linesSeen)]  # Indexes of peaks where we check sensitivity
	dfs_lineRads = [pd.DataFrame(index=range(nCh), columns=['n3', 'n4'],dtype=float),
					# Fitted radiance of each line for the different exposure types
					pd.DataFrame(index=range(nCh), columns=['n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12'],dtype=float),
					pd.DataFrame(index=range(nCh), columns=['n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12'],dtype=float)]

	df_lineRad = pd.DataFrame(index=range(nCh), columns=['n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11','n12'],dtype=float)  # Fitted radiance of each line, combined between different exposure types

	dfs_lineRads_sigma = [pd.DataFrame(index=range(nCh), columns=['n3', 'n4'],dtype=float),
					# Fitted radiance of each line for the different exposure types
					pd.DataFrame(index=range(nCh), columns=['n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12'],dtype=float),
					pd.DataFrame(index=range(nCh), columns=['n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12'],dtype=float)]

	df_lineRad_sigma = pd.DataFrame(index=range(nCh), columns=['n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11','n12'],dtype=float)  # Fitted radiance of each line, combined between different exposure types

	dfs_totIntens = [pd.DataFrame(index=range(len(df_settings)), columns=['n3', 'n4'],dtype=float),
					 pd.DataFrame(index=range(len(df_settings)),
								  columns=['n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12'],dtype=float),
					 pd.DataFrame(index=range(len(df_settings)),
								  columns=['n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12'],dtype=float)]

	iSet=0
	# for iSet in range(len(df_settings)):
	for iTyp, typ in enumerate(['Ha', 'Hb', 'Hc']):
		if typ!=type:
			continue

		# j = df_settings.loc[iSet, typ]
		# if not np.isnan(j):
		#	 (folder, date, time) = df_log.loc[j, ['folder', 'date', 'time']]
		#	 fname = folder + '/' + date + time + '.spe'
		#
		#	 file = SpeFile(fdir + fname)
		#	 data = file.data[0].T
		#
		#	 df_log.loc[j, 'Dark']
		#	 df_log.loc[df_log['type'] == 'Dark']
		#
		#	 jDark = np.where((df_log['time'] == df_log.loc[j, 'Dark']) & (df_log['type'] == 'Dark'))[0][0]
		#
		#	 (fDark, dDark, tDark) = df_log.loc[jDark, ['folder', 'date', 'time']]
		#	 fnDark = fDark + '/' + dDark + tDark + '.spe'
		#	 fileDark = SpeFile(fdir + fnDark)
		#	 dataDark = fileDark.data[0].T
		#
		#	 data -= dataDark
		#	 data = rotate(data, geom['angle'])
		#	 data = do_tilt(data, geom['tilt'])

		if typ == 'Ha':
			# binnedData = binData(data, geom['bin00a'], geom['binInterv'], check_overExp=False)
			oExpLim = np.zeros((len(binnedData))).astype('int')
			for iBin in range(len(binnedData)):
				for h, l0 in enumerate(waveLengths[:2]):
					iFit = np.logical_and(lamb[iTyp] > l0 - 5, lamb[iTyp] < l0 + 5)
					xFit = lamb[iTyp][iFit]
					yFit = binnedData[iBin, iFit]
					bgFit = np.polyfit(xFit[[0, -1]], yFit[[0, -1]], 1)
					yFit -= np.polyval(bgFit, xFit)
					p0 = [max(yFit), l0]

					fit = curve_fit(InstrFun, xFit, yFit, p0)
					dfs_lineRads[iTyp].loc[iBin][h] = fit[0][0] * peakUnitArea / binnedSens[iTyp, iBin, iSens[iTyp][h]]
			'''
			plt.figure()
			plt.plot(xFit,yFit)
			plt.plot(xFit,InstrFun(xFit,*fit[0]))
			'''
		elif typ == 'Hb':

			#section added to try to find a proper counts offset for the lines
			record_all_sigma=[]
			# binnedData, oExpLim = binData(data, geom['bin00b'][0], geom['binInterv'][0], check_overExp=True, nCh=40)
			oExpLim=np.zeros((len(binnedData))).astype('int')

			# highest_peak = int(np.median(binnedData.argmax(axis=1)))
			highest_peak = int(np.sum(binnedData,axis=0).argmax())
			highest_peak_wavel = lamb[iTyp][highest_peak]
			lambMax = lamb[iTyp][Nx - 1 - oExpLim[int(nCh/2)]] - 2
			first = np.searchsorted(waveLengths < lambMax, 1)  # First non-overexposed line to be measured
			highest_peak_number = (np.abs(waveLengths[first:5]-highest_peak_wavel)).argmin()
			# print('highest_peak_number '+str(highest_peak_number))
			fit_initial_record = []
			for iBin in range(len(binnedData)):
				# I do this because using 1 as treshold doesn't really work when you have number scaled with gain and proportionality correction
				noise_floor = np.median(binnedData[iBin])
				std_around_noise_floor = np.std(binnedData[iBin][binnedData[iBin]<=noise_floor].tolist()+(-binnedData[iBin][binnedData[iBin]<=noise_floor]).tolist())
				threshold_for_no_signal = noise_floor+2*std_around_noise_floor
				# print(threshold_for_no_signal)

				# if np.max(binnedData[iBin])<1:
				if np.max(binnedData[iBin])<threshold_for_no_signal:
					print('bin ' + str(iBin) + ' missing')
					continue
				lambMax = lamb[iTyp][Nx - 1 - oExpLim[iBin]] - 2
				first = np.searchsorted(waveLengths < lambMax, 1)  # First non-overexposed line to be measured
				dfs_lineRads[iTyp].loc[iBin][:first - 1] = np.nan
				dfs_lineRads_sigma[iTyp].loc[iBin][:first - 1] = np.nan
				try:
					for h, l0 in enumerate(waveLengths[first:first_line_in_multipulse+1], first - 1):
						if h>highest_peak_number+1:	# usually there are at least 2 good lines
							continue
						# iFit = np.logical_and(lamb[iTyp] > l0 + min(np.diff(waveLengths)[h + 1] / 2, -5),
						#					   lamb[iTyp] < min(l0 - min(np.diff(waveLengths)[h] / 2, -5),max(lamb[iTyp])))
						iFit = np.logical_and(np.logical_and(lamb[iTyp] > l0 + min(np.diff(waveLengths)[h + 1] / 2, -5),
											  lamb[iTyp] < l0 - min(np.diff(waveLengths)[h] / 2, -5)),np.arange(Nx)<Nx-20)
						# iFit = np.logical_and(lamb[iTyp] > l0 - 5,lamb[iTyp] < min(l0 + 5, max(lamb[iTyp])))
						xFit = lamb[iTyp][iFit]
						yFit = binnedData[iBin, iFit]
						yFit_sigma = binnedData_sigma[iBin, iFit]

						# if np.max(yFit)<1:
						if np.max(yFit)<threshold_for_no_signal:
							print('bin ' + str(iBin) + ' line ' + str(h + 4) + ' missing')
							continue

						bds = [[0, 0, l0-5, -np.inf, -np.inf],
							   # [max(yFit) * 100, np.inf, max(xFit), np.inf, np.inf]]
							   [max(yFit) * 100, 5, l0+5, np.inf, np.inf]]
						if pure_gaussian == True:
							p0 = [max(yFit), 0.8, max(min(xFit[yFit.argmax()],l0+5),l0-5), 0, yFit[0]]
							try:
								fit = curve_fit(full_InstrFun_biased, xFit, yFit, p0,sigma=yFit_sigma, absolute_sigma=True, maxfev=1000000, bounds=bds)
								record_all_sigma.append([fit[0][1], fit[1][1, 1]])
							except:
								fit = [[0, 1, 0, 0, 0]]
								print('bin ' + str(iBin) + ' line ' + str(h + 4) + ' line missing')

						# if h == 1:
						# plt.figure()
						#	 # plt.title('wavelength ' + str(l0) + ', line ' + str(h + 4) + '\n fit ' + str(
						#	 #	 fit[0]) + ' uncertainty ' + str(np.diag(fit[1])))
						# plt.plot(xFit, yFit)
						#	 # # plt.plot(xFit, InstrFun(xFit,*fit[0])-np.polyval(bgFit, xFit))
						# plt.plot(xFit, full_InstrFun_biased(xFit, *fit[0]))
						# plt.pause(0.001)
						# if h==0:
						#	 print(fit[0][1])

				except:
					print('bin ' + str(iBin) + ' preliminary fit failed')
			record_all_sigma = np.abs(np.array(record_all_sigma))
			record_all_sigma[:,1][record_all_sigma[:,1]<1e-5]=1e5
			mean_sigma =  np.nansum(record_all_sigma[:,0]/ (record_all_sigma[:,1]**2))/np.nansum(1/record_all_sigma[:,1]**2)
			mean_sigma_deviation = np.std(record_all_sigma[:,0])
			min_sigma = max(mean_sigma - mean_sigma_deviation*5 , 0.4)
			print('min_gaussiam_sigma = '+str(min_sigma))
			max_sigma = min(mean_sigma + mean_sigma_deviation*4 , 1.6)
			print('max_gaussiam_sigma = ' + str(max_sigma))

			# up to here I only fit the lines to find the max and min reasonable size (width) of the lines

			for iBin in range(len(binnedData)):

				# I do this because using 1 as treshold doesn't really work when you have number scaled with gain and proportionality correction
				noise_floor = np.median(binnedData[iBin])
				std_around_noise_floor = np.std(binnedData[iBin][binnedData[iBin]<=noise_floor].tolist()+(-binnedData[iBin][binnedData[iBin]<=noise_floor]).tolist())
				threshold_for_no_signal = noise_floor+2*std_around_noise_floor

				# if np.max(binnedData[iBin])<1:
				if np.max(binnedData[iBin])<threshold_for_no_signal:
					print('bin ' + str(iBin) + ' missing')
					continue

				sensitivity = binnedSens[iTyp, iBin]
				sensitivity_sigma = binnedSens_sigma[iTyp, iBin]
				waveLcoefs_iTyp = binnedSens_waveLcoefs[iTyp]
				sensitivity_x = np.polyval(waveLcoefs_iTyp,np.arange(len(sensitivity)))
				sensitivity_dx = np.concatenate([np.diff(sensitivity_x),[np.diff(sensitivity_x)[-1]]])
				sensitivity_interpolator = interp1d(sensitivity_x,sensitivity,kind='linear',fill_value=tuple(sensitivity[[0,-1]]),bounds_error=False)
				sensitivity_sigma_interpolator = interp1d(sensitivity_x,sensitivity_sigma,kind='linear',fill_value=tuple(sensitivity_sigma[[0,-1]]),bounds_error=False)
				sensitivity_density_interpolator = interp1d(sensitivity_x,sensitivity/sensitivity_dx,kind='linear',fill_value=tuple((sensitivity/sensitivity_dx)[[0,-1]]),bounds_error=False)
				sensitivity_density_sigma_interpolator = interp1d(sensitivity_x,sensitivity_sigma/sensitivity_dx,kind='linear',fill_value=tuple((sensitivity_sigma/sensitivity_dx)[[0,-1]]),bounds_error=False)

				def sensitivity_calculator_with_uncertainty(xFit_int):
					dxFit_int = np.concatenate([np.diff(xFit_int),[np.diff(xFit_int)[-1]]])
					true_sensitivity_1 = sensitivity_density_interpolator(xFit_int-dxFit_int/2)
					true_sensitivity_2 = sensitivity_density_interpolator(xFit_int+dxFit_int/2)
					true_sensitivity = (true_sensitivity_1+true_sensitivity_2)/2*dxFit_int
					true_sensitivity_sigma_1 = sensitivity_density_sigma_interpolator(xFit_int-dxFit_int/2)
					true_sensitivity_sigma_2 = sensitivity_density_sigma_interpolator(xFit_int+dxFit_int/2)
					true_sensitivity_sigma = (true_sensitivity_sigma_1+true_sensitivity_sigma_2)/2*dxFit_int
					true_sensitivity_full = unumpy.uarray(true_sensitivity,true_sensitivity_sigma)	# counts W-1 m2 sr nm
					return true_sensitivity_full

				if  (to_print and iBin in [5,20,30]):
					tm.sleep(np.random.random()*10)
					plt.figure(figsize=(30, 15))

					# section added to print in W rather than counts
					dlamb = np.concatenate([np.diff(lamb[iTyp]),[np.diff(lamb[iTyp])[-1]]])
					true_sensitivity_full = sensitivity_calculator_with_uncertainty(lamb[iTyp])	# counts W-1 m2 sr nm
					yFit_full = unumpy.uarray(binnedData[iBin],binnedData_sigma[iBin])	# counts
					yFit_full = yFit_full / true_sensitivity_full	# W m-2 sr-1 nm-1
					yFit_full_sigma = std_devs(yFit_full)	# W m-2 sr-1 nm-1
					yFit_full = nominal_values(yFit_full)

					# plt.plot(lamb[iTyp],binnedData[iBin],'b')
					# # plt.errorbar(lamb[iTyp],binnedData[iBin],'c',yerr=binnedData_sigma[iBin],label='binned data')
					# plt.fill_between(lamb[iTyp], np.max([binnedData[iBin]-binnedData_sigma[iBin],np.ones_like(binnedData[iBin])*max(np.min(binnedData[iBin]),0.0001)],axis=0),binnedData[iBin]+binnedData_sigma[iBin],np.max([binnedData[iBin]-binnedData_sigma[iBin],np.ones_like(binnedData[iBin])*max(np.min(binnedData[iBin]),0.0001)],axis=0)<binnedData[iBin]+binnedData_sigma[iBin],color='c',alpha=0.01)

					plt.plot(lamb[iTyp],yFit_full,'b')
					plt.fill_between(lamb[iTyp], np.max([yFit_full-yFit_full_sigma,np.ones_like(yFit_full)*max(np.min(yFit_full),0.00000001)],axis=0),yFit_full+yFit_full_sigma,np.max([yFit_full-yFit_full_sigma,np.ones_like(yFit_full)*max(np.min(yFit_full),0.00000001)],axis=0)<yFit_full+yFit_full_sigma,color='c',alpha=0.01)

				# selected_for_background = np.logical_and(lamb[iTyp]>505,np.arange(Nx)<1550) + np.logical_and(lamb[iTyp]>445,lamb[iTyp]<470)+ np.logical_and(lamb[iTyp]>420,lamb[iTyp]<425)+ np.logical_and(lamb[iTyp]>402,lamb[iTyp]<405)
				# xFit = lamb[iTyp][selected_for_background]
				# yFit = binnedData[iBin, selected_for_background]
				# # plt.figure()
				# # plt.plot(xFit,yFit)
				# # plt.plot(xFit,medfilt(yFit,11))
				# xFit_filtered = savgol_filter(xFit,51,2)
				# yFit_filtered = savgol_filter(medfilt(yFit,11),51,2)
				# # plt.plot(xFit_filtered,yFit_filtered,'x')
				# while np.sum(np.diff(xFit_filtered)<0)>0:
				#	 yFit_filtered = yFit_filtered[:-1][np.diff(xFit_filtered)>0]
				#	 xFit_filtered = xFit_filtered[:-1][np.diff(xFit_filtered)>0]
				# # plt.plot(xFit_filtered,yFit_filtered)
				# # plt.pause(0.01)
				# background_interpolation = interp1d(xFit_filtered,yFit_filtered,kind='linear',fill_value='extrapolate')

				temp = []
				temp_x = []
				for limit_a,limit_b in [[505,np.inf],[465,470],[445,450],[420,425],[401,404]]:
					selected_for_background = np.logical_and(np.logical_and(lamb[iTyp]>limit_a,lamb[iTyp]<limit_b) , np.arange(Nx)<1550)
					temp.append( np.mean(np.sort(binnedData[iBin, selected_for_background] ) [:np.sum(selected_for_background)//2] ) )
					temp_x.append( np.mean(lamb[iTyp][selected_for_background]) )
				background_interpolation = interp1d(temp_x,temp,kind='linear',fill_value='extrapolate')
				limit_a,limit_b = [391,394]
				selected_for_background = np.logical_and(np.logical_and(lamb[iTyp]>limit_a,lamb[iTyp]<limit_b) , np.arange(Nx)<1550)
				temp_add = np.mean(np.sort(binnedData[iBin, selected_for_background] ) [:np.sum(selected_for_background)//2] )
				if np.diff(background_interpolation([402,405]))>0 and temp_add/background_interpolation(393)<2:
					temp_add = np.mean([temp_add,np.mean(background_interpolation(lamb[iTyp][selected_for_background]))])
				else:
					temp_add = min(temp_add,temp[-1])
				temp.append(temp_add)
				temp_x.append( np.mean(lamb[iTyp][selected_for_background]) )
				background_interpolation = interp1d(temp_x,temp,kind='linear',fill_value='extrapolate')
				# plt.figure()
				# plt.plot(lamb[iTyp],background_interpolation(lamb[iTyp]))
				# plt.plot(temp_x,background_interpolation(temp_x),'+')
				# plt.semilogy()
				# plt.pause(0.01)
				lambMax = lamb[iTyp][Nx-1-oExpLim[iBin]] - 2
				first = np.searchsorted(waveLengths < lambMax, 1)  # First non-overexposed line to be measured
				dfs_lineRads[iTyp].loc[iBin][:first - 1] = np.nan
				dfs_lineRads_sigma[iTyp].loc[iBin][:first - 1] = np.nan

				fit_stark_record = []
				record_peak_shift = []
				try:	# here I fit the lines that I can fit independently
					for h, l0 in enumerate(waveLengths[first:first_line_in_multipulse+first+1], first - 1):
						# print(h)
						# print(h) print_example
						# iFit = np.logical_and(lamb[iTyp] > l0 + min(np.diff(waveLengths)[h+1]/2,-5), lamb[iTyp] < min(l0-min(np.diff(waveLengths)[h]/2,-5),max(lamb[iTyp])))
						iFit = np.logical_and(lamb[iTyp] > l0 + min(np.diff(waveLengths)[h+1]/2,-5), lamb[iTyp] < l0-min(np.diff(waveLengths)[h]/2,-5))
						iFit = np.logical_and(np.logical_and(iFit, np.arange(Nx)>20),np.arange(Nx)<Nx-20)	# added 16/02/2020 to afoid effects from tilting the image
						# iFit = np.logical_and(lamb[iTyp] > l0 - 5,lamb[iTyp] < min(l0 + 5, max(lamb[iTyp])))
						xFit = lamb[iTyp][iFit]
						yFit = binnedData[iBin, iFit] - background_interpolation(xFit)
						yFit_sigma = binnedData_sigma[iBin, iFit]

						# here I go from counts to W
						true_sensitivity_full = sensitivity_calculator_with_uncertainty(xFit)	# counts W-1 m2 sr nm
						true_sensitivity_sigma = std_devs(true_sensitivity_full)	# counts W-1 m2 sr nm
						true_sensitivity = nominal_values(true_sensitivity_full)
						yFit_full = unumpy.uarray(yFit,yFit_sigma)	# counts

						yFit = yFit_full / true_sensitivity_full
						yFit_sigma = std_devs(yFit)
						yFit = nominal_values(yFit)

						if np.max(yFit)<=0:
							print('bin ' + str(iBin) + ' missing')
							continue


						# bds = [[0, min(xFit), -np.inf, -np.inf],
						#		[max(yFit)*100, max(xFit), np.inf, np.inf]]
						# if pure_gaussian == True:
						#	 if h>0:
						#		 p0 = [max(yFit), l0, fit[0][2], fit[0][3]]
						#	 else:
						#		 p0 = [max(yFit), l0, 0, yFit[[0]]]   #  A, sig, x0, m,y0
						#	 try:
						#		 fit = curve_fit(InstrFun_biased, xFit, yFit, p0, maxfev=100000000, bounds=bds)
						#	 except:
						#		 fit = [[0, 0, 0, 0]]
						#		 print('bin ' + str(iBin) + ' line ' + str(h + 4) + ' line missing')
						# else:
						#	 bgFit = np.polyfit(xFit[[0 -1]], yFit[[0, -1]], 1)
						#	 yFit -= np.polyval(bgFit, xFit)
						#	 p0 = [max(yFit), l0]
						#	 fit = curve_fit(InstrFun, xFit, yFit, p0)
						# dfs_lineRads[iTyp].loc[iBin][h] = fit[0][0] * peakUnitArea / binnedSens[
						#		 iTyp, iBin, iSens[iTyp][h]]
						# df_lineRad.loc[iBin][h + 1] = fit[0][0] * peakUnitArea / binnedSens[
						#	 iTyp, iBin, iSens[iTyp][h]]
						# # plt.figure()
						# # plt.plot(xFit, yFit)
						# # plt.plot(xFit, InstrFun_biased(xFit, *fit[0]))
						# # plt.pause(0.001)



						# bds = [[0,min_sigma, min(xFit), -np.inf, -np.inf],
						#		[max(yFit)*100,max_sigma, max(xFit), np.inf, np.inf]]
						# bds = [[0,min_sigma, l0-waveLcoefs[1][1]*30, -np.inf, -np.inf],
						#		[max(yFit)*100,max_sigma, l0+waveLcoefs[1][1]*30, np.inf, np.inf]]
						# bds = [[0,min_sigma, l0-waveLcoefs[1][1]*30, -1/np.max(xFit), -np.inf],
						#		[max(yFit)*100,max_sigma, l0+waveLcoefs[1][1]*30, np.inf, np.inf]]
						# bds = [[0,min_sigma, l0-waveLcoefs[1][1]*30, -np.inf,-np.inf, -np.inf],
						#		[max(yFit)*100,max_sigma, l0+waveLcoefs[1][1]*30, 0,np.inf, np.inf]]
						# p0_stark = [0.01,2,0.001]
						# bds_stark = [[0,0.1,1e-5],[1e-1,5,0.01]]
						p0_stark = [0.01,2,0]
						bds_stark = [[0,0.1,-1],[1e-1,5,1]]
						scale_stark = [0.01,2,0.001]
						if h<first_line_in_multipulse:	# do this only for the lines that can be properly fit independently
							bds = [[0,min_sigma, l0-waveLcoefs[1][1]*30],
								   [max(yFit)*100,max_sigma, l0+waveLcoefs[1][1]*30]]
							allowed_range = [np.abs(xFit-bds[0][-1]).argmin()+1,np.abs(xFit-bds[1][-1]).argmin()-1]
							if pure_gaussian == True:
								if h>0:
									p0 = [max(yFit), fit[0][1],l0]
								else:
									# p0 = [max(yFit),(min_sigma+max_sigma)/2, l0, (np.mean(yFit[-10:-1])-np.mean(yFit[0:10]))/(xFit[-5] - xFit[5]),np.mean(yFit[-10:-1])- xFit[5]*(np.mean(yFit[-10:-1])-np.mean(yFit[0:10]))/(xFit[-5] - xFit[5])]
									# p0 = [max(yFit),(min_sigma+max_sigma)/2, l0, 0,np.mean(yFit[-10:-1])- xFit[5]*(np.mean(yFit[-10:-1])-np.mean(yFit[0:10]))/(xFit[-5] - xFit[5])]
									# p0 = [max(yFit),(min_sigma+max_sigma)/2, l0, 0,0,np.mean(yFit[-10:-1])]
									p0 = [max(yFit),(min_sigma+max_sigma)/2, l0]
								try:
									fit = curve_fit(full_InstrFun, xFit, yFit, p0,sigma=yFit_sigma, absolute_sigma=True, maxfev=10000000, bounds=bds)
									# fit = curve_fit(full_InstrFun_biased, xFit, yFit, p0, maxfev=10000000, bounds=bds)
									pixel_location_of_the_peak = (np.abs(lamb[iTyp] - fit[0][2])).argmin()
									wavel_location_of_the_peak = fit[0][2]
									wavel_location_of_the_peak_sigma = np.sqrt(fit[1][2,2])
									gaussian_sigma_of_the_peak = fit[0][1]
									gaussian_sigma_of_the_peak_sigma = np.sqrt(fit[1][1,1])
									amplitude_of_the_peak = fit[0][0]
									amplitude_of_the_peak_sigma = np.sqrt(fit[1][0,0])
									# sensitivity = binnedSens[iTyp, iBin, (np.abs(lamb[iTyp] - fit[0][2])).argmin()]
									if  (to_print and iBin in [5,20,30]):
										temp_gaus_peak = full_InstrFun_with_error_included(xFit, *correlated_values(fit[0],fit[1]) )
										plt.errorbar(xFit,nominal_values(temp_gaus_peak)+background_interpolation(xFit)/true_sensitivity,yerr=std_devs(temp_gaus_peak),label='n='+str(h+first+3)+'->2, x0=%.5g+/-%.3g' %(wavel_location_of_the_peak,wavel_location_of_the_peak_sigma) + ' instead of %.5g nm' % l0 + '\namp=%.3g+/-%.3g au' % (amplitude_of_the_peak/((2*np.pi*gaussian_sigma_of_the_peak**2)**0.5),amplitude_of_the_peak_sigma/((2*np.pi*gaussian_sigma_of_the_peak**2)**0.5))+ ', \u03C3=%.4g+/-%.3g nm' % (gaussian_sigma_of_the_peak,gaussian_sigma_of_the_peak_sigma) )
										plt.plot(xFit,background_interpolation(xFit)/true_sensitivity,'k--')
										plt.plot([l0,l0],[np.min(nominal_values(temp_gaus_peak)+background_interpolation(xFit)/true_sensitivity),np.max(nominal_values(temp_gaus_peak)+background_interpolation(xFit)/true_sensitivity)],'--',color='r',linewidth=0.5)
										plt.plot([wavel_location_of_the_peak,wavel_location_of_the_peak],[np.min(nominal_values(temp_gaus_peak)+background_interpolation(xFit)/true_sensitivity),np.max(nominal_values(temp_gaus_peak)+background_interpolation(xFit)/true_sensitivity)],'--',color='k',linewidth=0.5)
								except Exception as e:
									fit = [[0,1,l0],np.ones((len(p0),len(p0)))]
									# sensitivity = binnedSens[iTyp, iBin, iSens[iTyp][h]]
									pixel_location_of_the_peak = iSens[iTyp][h]
									wavel_location_of_the_peak = l0
									wavel_location_of_the_peak_sigma = 1
									gaussian_sigma_of_the_peak = 1
									gaussian_sigma_of_the_peak_sigma = 1
									amplitude_of_the_peak = 0
									amplitude_of_the_peak_sigma = 1
									print('bin '+str(iBin)+' line '+str(h+4)+' line missing because of '+str(e))
								record_peak_shift.append(fit[0][2]-l0)
								try:
									# yFit_residual = yFit-full_InstrFun(xFit,*fit[0])
									yFit_residual = cp.deepcopy(yFit)#-full_InstrFun(xFit,*fit[0])
									# select = yFit_residual>full_InstrFun(xFit,*fit[0])
									# select = yFit>full_InstrFun(xFit,*fit[0])	# yFit_residualhas already subtracted the peak. this way I exclude more than I should
									select = np.logical_or(xFit<fit[0][2]-2.5*fit[0][1],xFit>fit[0][2]+2.5*fit[0][1])	# yFit_residualhas already subtracted the peak. this way I look at averything out of 2.5 sigma of the main peak
									yFit_stark = yFit_residual[select]
									xFit_stark = xFit[select]
									yFit_sigma_stark = yFit_sigma[select]*(((xFit_stark-fit[0][2])/fit[0][2])**2)
									# p0_stark = [*fit[0][:-1],1,0.01,2,0.5]
									# bds_stark = [[fit[0][0]*0.1,min_sigma,1,1e-3,1,0.3],[fit[0][0]*10,max_sigma,1.5,1e-1,5,1]]
									# gauss_plus_stark_fixed =lambda x, A, sig, P, A1_A, sig_sig, dist_sig:super_gaussian(x, A, sig, fit[0][-1],P) + stark_gaussian(x,A*A1_A,sig*sig_sig,fit[0][-1],sig*dist_sig,P)
									# fit_stark = curve_fit(gauss_plus_stark_fixed, xFit, yFit, p0_stark,sigma=yFit_sigma, absolute_sigma=True, maxfev=1000000, bounds=bds_stark,x_scale=p0_stark,ftol=1e-11,xtol=1e-11,verbose=2)
									# plt.plot(xFit,gauss_plus_stark_fixed(xFit,*fit_stark[0])+background_interpolation(xFit),label=str(np.around(fit_stark[0],decimals=3)))
									# plt.plot(xFit,gauss_plus_stark_fixed(xFit,*fit_stark[0])-super_gaussian(xFit,*fit_stark[0][:2],fit[0][-1],fit_stark[0][2])+background_interpolation(xFit),label='stark')
									# plt.plot(xFit,yFit-full_InstrFun(xFit,*fit[0]),label='maybe stark')
									# stark_gaussian_fixed =lambda x, A1_A, sig_sig, dist_sig:stark_gaussian(x,fit[0][0]*A1_A,fit[0][1]*sig_sig,fit[0][2],fit[0][1]*dist_sig,1)	# i think it does not make sense to include sig_sig in dist=fit[0][1]*sig_sig*dist_sig . why???
									stark_gaussian_fixed =lambda x, A1_A, sig_sig, dist_sig:stark_gaussian(x,fit[0][0]*A1_A,fit[0][1]*sig_sig,fit[0][2],fit[0][1]*sig_sig*dist_sig,1)
									fit_stark = curve_fit(stark_gaussian_fixed, xFit_stark, yFit_stark, p0_stark,sigma=yFit_sigma_stark, absolute_sigma=True, maxfev=1000000, bounds=bds_stark,x_scale=scale_stark,ftol=1e-14,xtol=1e-15)
									if  (to_print and iBin in [5,20,30]):
										plt.plot(xFit,stark_gaussian_fixed(xFit,*fit_stark[0])+background_interpolation(xFit)/true_sensitivity,label='line broadening: amplitude=%.3g, sigma %.3g times main gaussian' %(fit_stark[0][0],fit_stark[0][1])+'\ndistance between broading peaks %.3gnm (%.3g)' %(fit[0][1]*fit_stark[0][1]*fit_stark[0][2],fit_stark[0][2]))
										plt.plot(xFit,stark_gaussian_fixed(xFit,*fit_stark[0])+full_InstrFun(xFit,*fit[0])+background_interpolation(xFit)/true_sensitivity)
										# plt.plot(xFit,yFit-full_InstrFun(xFit,*fit[0])+background_interpolation(xFit),label='maybe stark')
									fit_stark_record.append(fit_stark)
								except:
									print('bin '+str(h)+'stark failed')
									fit_stark = np.array([ 0.0       ,  0.81317458, 0        ])
							else:
								bgFit = np.polyfit(xFit[[0 -1]], yFit[[0, -1]], 1)
								yFit -= np.polyval(bgFit, xFit)
								p0 = [max(yFit), l0]
								fit = curve_fit(InstrFun, xFit, yFit, p0)
								pixel_location_of_the_peak = iSens[iTyp][h]
								wavel_location_of_the_peak = l0
								# sensitivity = binnedSens[iTyp, iBin, iSens[iTyp][h]]
							if perform_convolution==True:
								if False:	# this is if yFit does not have the calibration
									temp_sensitivity = unumpy.uarray(sensitivity,sensitivity_sigma)	# counts W-1 m2 sr nm
									# gaussian_weigths = full_InstrFun(xFit,1,fit[0][1],wavel_location_of_the_peak)/np.sum(full_InstrFun(xFit,1,fit[0][1],wavel_location_of_the_peak))
									# sensitivity = np.sum(np.array(sensitivity)*gaussian_weigths)
									temp_gaus_peak = full_InstrFun_with_error_included(lamb[iTyp], *correlated_values(fit[0],fit[1]) )	# counts
									# gaus_peak = full_InstrFun(lamb[iTyp], *fit[0])
									temp = np.abs(np.trapz(temp_gaus_peak/temp_sensitivity , x=lamb[iTyp]))	# W m-2 sr-1
								else:	# in reality here I don't do a convolution, I just sum over the background
									width_to_consider = np.nanmax([fit[0][1]*3,fit_stark[0][1]*fit[0][1]*2])
									select = np.logical_and(xFit>fit[0][2]-width_to_consider,xFit<fit[0][2]+width_to_consider)
									temp = np.trapz(unumpy.uarray(yFit[select],yFit_sigma[select]) , x=xFit[select])	# W m-2 sr-1
								# dfs_lineRads[iTyp].loc[iBin][h] = np.abs(np.trapz(gaus_peak/sensitivity , x=lamb[iTyp]))
								# df_lineRad.loc[iBin][h + 1] = np.abs(np.trapz(gaus_peak/sensitivity , x=lamb[iTyp]))
								# gaus_peak_sigma = full_InstrFun_sigma(lamb[iTyp],*fit[0],amplitude_of_the_peak_sigma,gaussian_sigma_of_the_peak_sigma,wavel_location_of_the_peak_sigma)
								# dfs_lineRads_sigma[iTyp].loc[iBin][h] = np.sqrt(np.nansum( ((lamb[iTyp][1:]-lamb[iTyp][:-1])**2)*((gaus_peak[1:]/sensitivity[1:])**2 *( (gaus_peak_sigma[1:]/gaus_peak[1:])**2 + (sensitivity_sigma[1:]/sensitivity[1:])**2) + (gaus_peak[:-1]/sensitivity[:-1])**2 *( (gaus_peak_sigma[:-1]/gaus_peak[:-1])**2 + (sensitivity_sigma[:-1]/sensitivity[:-1])**2)) ))/2
								# df_lineRad_sigma.loc[iBin][h + 1] = np.sqrt(np.nansum( ((lamb[iTyp][1:]-lamb[iTyp][:-1])**2)*((gaus_peak[1:]/sensitivity[1:])**2 *( (gaus_peak_sigma[1:]/gaus_peak[1:])**2 + (sensitivity_sigma[1:]/sensitivity[1:])**2) + (gaus_peak[:-1]/sensitivity[:-1])**2 *( (gaus_peak_sigma[:-1]/gaus_peak[:-1])**2 + (sensitivity_sigma[:-1]/sensitivity[:-1])**2)) ))/2
								dfs_lineRads[iTyp].loc[iBin][h] = nominal_values(temp)	# W m-2 sr-1
								df_lineRad.loc[iBin][h + 1] = nominal_values(temp)	# W m-2 sr-1
								dfs_lineRads_sigma[iTyp].loc[iBin][h] = std_devs(temp)	# W m-2 sr-1
								df_lineRad_sigma.loc[iBin][h + 1] = std_devs(temp)	# W m-2 sr-1
							else:
								print("this shouldn't happen, iBin"+str(iBin)+' h'+str(h))
								# sensitivity = binnedSens[iTyp, iBin, pixel_location_of_the_peak]
								# dfs_lineRads[iTyp].loc[iBin][h] = np.abs(np.trapz(full_InstrFun(xFit, *fit[0]), x=xFit)) / sensitivity
								# df_lineRad.loc[iBin][h + 1] = np.abs(np.trapz(full_InstrFun(lamb[iTyp], *fit[0]), x=lamb[iTyp])) / sensitivity
								dfs_lineRads[iTyp].loc[iBin][h] = np.abs(np.trapz(full_InstrFun(xFit, *fit[0]), x=xFit)) / binnedSens[iTyp, iBin, pixel_location_of_the_peak]	# W m-2 sr-1
								df_lineRad.loc[iBin][h + 1] = np.abs(np.trapz(full_InstrFun(lamb[iTyp], *fit[0]) / sensitivity, x=lamb[iTyp]))	# W m-2 sr-1
						else:	# this is to fit the first of the lines that are too close each other to be looked at indepemdently
							bds = [[0,min_sigma, l0-waveLcoefs[1][1]*30],
								   [max(yFit)*100,max_sigma, l0+waveLcoefs[1][1]*30]]
							allowed_range = [np.abs(xFit-bds[0][-1]).argmin()+1,np.abs(xFit-bds[1][-1]).argmin()-1]
							if pure_gaussian == True:
								p0 = [max(yFit), fit[0][1],l0]
								try:
									fit_additional = curve_fit(full_InstrFun, xFit, yFit, p0,sigma=yFit_sigma, absolute_sigma=True, maxfev=10000000, bounds=bds)
									record_peak_shift.append(fit_additional[0][2]-l0)
									# fit = curve_fit(full_InstrFun_biased, xFit, yFit, p0, maxfev=10000000, bounds=bds)
									pixel_location_of_the_peak = (np.abs(lamb[iTyp] - fit_additional[0][2])).argmin()
									wavel_location_of_the_peak = fit_additional[0][2]
									wavel_location_of_the_peak_sigma = np.sqrt(fit_additional[1][2,2])
									gaussian_sigma_of_the_peak = fit_additional[0][1]
									gaussian_sigma_of_the_peak_sigma = np.sqrt(fit_additional[1][1,1])
									amplitude_of_the_peak = fit_additional[0][0]
									amplitude_of_the_peak_sigma = np.sqrt(fit_additional[1][0,0])
									# sensitivity = binnedSens[iTyp, iBin, (np.abs(lamb[iTyp] - fit[0][2])).argmin()]
									if  (to_print and iBin in [5,20,30]):
										temp_gaus_peak = full_InstrFun_with_error_included(xFit, *correlated_values(fit_additional[0],fit_additional[1]) )
										plt.errorbar(xFit,nominal_values(temp_gaus_peak)+background_interpolation(xFit)/true_sensitivity,yerr=std_devs(temp_gaus_peak),label='n='+str(h+first+3)+'->2, x0=%.5g+/-%.3g' %(wavel_location_of_the_peak,wavel_location_of_the_peak_sigma) + ' instead of %.5g nm' % l0 + '\namp=%.3g+/-%.3g au' % (amplitude_of_the_peak/((2*np.pi*gaussian_sigma_of_the_peak**2)**0.5),amplitude_of_the_peak_sigma/((2*np.pi*gaussian_sigma_of_the_peak**2)**0.5))+ ', \u03C3=%.4g+/-%.3g nm' % (gaussian_sigma_of_the_peak,gaussian_sigma_of_the_peak_sigma) )
										plt.plot(xFit,background_interpolation(xFit)/true_sensitivity,'k--')
										plt.plot([l0,l0],[np.min(nominal_values(temp_gaus_peak)+background_interpolation(xFit)/true_sensitivity),np.max(nominal_values(temp_gaus_peak)+background_interpolation(xFit)/true_sensitivity)],'--',color='r',linewidth=0.5)
										plt.plot([wavel_location_of_the_peak,wavel_location_of_the_peak],[np.min(nominal_values(temp_gaus_peak)+background_interpolation(xFit)/true_sensitivity),np.max(nominal_values(temp_gaus_peak)+background_interpolation(xFit)/true_sensitivity)],'--',color='k',linewidth=0.5)
									yFit_residual = yFit-full_InstrFun(xFit,*fit_additional[0])
									select = yFit_residual>full_InstrFun(xFit,*fit_additional[0])
									yFit_stark = yFit_residual[select]
									xFit_stark = xFit[select]
									yFit_sigma_stark = yFit_sigma[select]*(((xFit_stark-fit_additional[0][2])/fit_additional[0][2])**2)
									# p0_stark = [*fit[0][:-1],1,0.01,2,0.5]
									# bds_stark = [[fit[0][0]*0.1,min_sigma,1,1e-3,1,0.3],[fit[0][0]*10,max_sigma,1.5,1e-1,5,1]]
									# gauss_plus_stark_fixed =lambda x, A, sig, P, A1_A, sig_sig, dist_sig:super_gaussian(x, A, sig, fit[0][-1],P) + stark_gaussian(x,A*A1_A,sig*sig_sig,fit[0][-1],sig*dist_sig,P)
									# fit_stark = curve_fit(gauss_plus_stark_fixed, xFit, yFit, p0_stark,sigma=yFit_sigma, absolute_sigma=True, maxfev=1000000, bounds=bds_stark,x_scale=p0_stark,ftol=1e-11,xtol=1e-11,verbose=2)
									# plt.plot(xFit,gauss_plus_stark_fixed(xFit,*fit_stark[0])+background_interpolation(xFit),label=str(np.around(fit_stark[0],decimals=3)))
									# plt.plot(xFit,gauss_plus_stark_fixed(xFit,*fit_stark[0])-super_gaussian(xFit,*fit_stark[0][:2],fit[0][-1],fit_stark[0][2])+background_interpolation(xFit),label='stark')
									# plt.plot(xFit,yFit-full_InstrFun(xFit,*fit[0]),label='maybe stark')
									stark_gaussian_fixed =lambda x, A1_A, sig_sig, dist_sig:stark_gaussian(x,fit_additional[0][0]*A1_A,fit_additional[0][1]*sig_sig,fit_additional[0][2],fit_additional[0][1]*sig_sig*dist_sig,1)
									fit_stark = curve_fit(stark_gaussian_fixed, xFit_stark, yFit_stark, p0_stark,sigma=yFit_sigma_stark, absolute_sigma=True, maxfev=1000000, bounds=bds_stark,x_scale=scale_stark,ftol=1e-14,xtol=1e-15)
									if  (to_print and iBin in [5,20,30]):
										plt.plot(xFit,stark_gaussian_fixed(xFit,*fit_stark[0])+background_interpolation(xFit)/true_sensitivity,label='line broadening: amplitude=%.3g, sigma %.3g times main gaussian' %(fit_stark[0][0],fit_stark[0][1])+'\ndistance between broading peaks %.3gnm (%.3g)' %(fit_additional[0][1]*fit_stark[0][1]*fit_stark[0][2],fit_stark[0][2]))
										# plt.plot(xFit,stark_gaussian_fixed(xFit,*fit_stark[0])+full_InstrFun(xFit,*fit_additional[0])+background_interpolation(xFit),label='n=%.3g full' %(h+first+3))
										# plt.plot(xFit,yFit-full_InstrFun(xFit,*fit[0])+background_interpolation(xFit),label='maybe stark')
									fit_stark_record.append(fit_stark)
								except Exception as e:
									print('bin '+str(iBin)+' line '+str(h+4)+' line missing because of '+str(e)+'relative broadening skipped')

					temp1 = []
					temp2 = []
					for i in range(len(fit_stark_record)):
						temp1.append(fit_stark_record[i][0])
						temp2.append(fit_stark_record[i][1])
					fit_stark_record = np.array(temp1)
					fit_stark_record_score = np.array(temp2)
					fit_stark_record_score[fit_stark_record_score==0]=max(1,np.nanmax(fit_stark_record_score))*10
					p0_stark = [1,1,1]
					def parabola(x, *params):

						a = params[0]
						b = params[1]
						c = params[2]
						return a*(x**2)+b*x+c
					try:
						# bds_stark = [[0,0,0],[np.inf,1e-6,np.inf]]
						fit_stark_amplitude = curve_fit(parabola, np.arange(len(fit_stark_record))+3, fit_stark_record[:,0], p0_stark,sigma=fit_stark_record_score[:,0,0]**0.5, absolute_sigma=True, maxfev=1000000)#, bounds=bds_stark)
						stark_high_lines_amp = np.polyval(fit_stark_amplitude[0],np.arange(first_line_in_multipulse+3,first_line_in_multipulse+3+total_number_of_lines_in_multipulse))
						stark_high_lines_amp[stark_high_lines_amp<0]=0
					except Exception as e:
						print('bin '+str(iBin)+' stark_high_lines_amp failed. Reason: %s' % str(e))
						stark_high_lines_amp = np.ones((total_number_of_lines_in_multipulse))*np.nanmedian(fit_stark_record[:,0])
					p0_stark = [1,1]
					def line(x, *params):
						m = params[0]
						q = params[1]
						return m*x+q
					try:
						fit_stark_sigma = curve_fit(line, np.arange(len(fit_stark_record))+3, np.log(fit_stark_record[:,1]), p0_stark,sigma=fit_stark_record_score[:,1,1]**0.5, absolute_sigma=True, maxfev=1000000)#, bounds=bds_stark)
						stark_high_lines_sigma = np.exp(np.polyval(fit_stark_sigma[0],np.arange(first_line_in_multipulse+3,first_line_in_multipulse+3+total_number_of_lines_in_multipulse)))
						stark_high_lines_sigma[stark_high_lines_sigma<0]=0
					except Exception as e:
						print('bin '+str(iBin)+' stark_high_lines_sigma failed. Reason: %s' % str(e))
						stark_high_lines_sigma = np.ones((total_number_of_lines_in_multipulse))*np.nanmedian(fit_stark_record[:,1])
					if False:
						fit_stark_dist = curve_fit(line, np.arange(len(fit_stark_record))+3, np.log(fit_stark_record[:,2]), p0_stark,sigma=fit_stark_record_score[:,2,2]**0.5, absolute_sigma=True, maxfev=1000000)#, bounds=bds_stark)
						stark_high_lines_dist = np.exp(np.polyval(fit_stark_dist[0],np.arange(first_line_in_multipulse+3,first_line_in_multipulse+3+total_number_of_lines_in_multipulse)))
						stark_high_lines_dist[stark_high_lines_dist<0]=0
						stark_high_lines_dist[stark_high_lines_dist>0.01]=0.01
					else:
						stark_high_lines_dist=np.ones((total_number_of_lines_in_multipulse))*min(0.01,np.sum(fit_stark_record[:,2]/fit_stark_record_score[:,2,2])/np.sum(1/fit_stark_record_score[:,2,2]))
					record_peak_shift = np.nanmedian(record_peak_shift)




						# dfs_lineRads[iTyp].loc[iBin][h] = np.abs(np.trapz(full_InstrFun_biased(xFit, *fit[0][:-2],0,0), x=xFit)) / binnedSens[iTyp, iBin, iSens[iTyp][h]]
						# df_lineRad.loc[iBin][h + 1] = np.abs(np.trapz(full_InstrFun_biased(lamb[1], *fit[0]), x=lamb[1])) / binnedSens[iTyp, iBin, iSens[iTyp][h]]
						# dfs_lineRads[iTyp].loc[iBin][h] = np.abs(np.trapz(full_InstrFun_biased_no_neg(xFit, *fit[0][:-2],0,0), x=xFit)) / sensitivity
						# df_lineRad.loc[iBin][h + 1] = np.abs(np.trapz(full_InstrFun_biased_no_neg(lamb[1], *fit[0]), x=lamb[1])) / sensitivity
						# dfs_lineRads[iTyp].loc[iBin][h] = np.abs(np.trapz(full_InstrFun_biased_quad(xFit, *fit[0][:3],*np.zeros_like(fit[0][3:])), x=xFit)) / sensitivity
						# df_lineRad.loc[iBin][h + 1] = np.abs(np.trapz(full_InstrFun_biased_quad(lamb[1], *fit[0][:3],*np.zeros_like(fit[0][3:])), x=lamb[1])) / sensitivity
						# if h == 3:
						# plt.figure()
						# plt.title('wavelength '+str(l0)+', line '+str(h+4)+'\n fit '+str(fit[0])+' uncertainty '+str(np.diag(fit[1])))
						# plt.plot(xFit, yFit)
						# # plt.plot(xFit, InstrFun(xFit,*fit[0])-np.polyval(bgFit, xFit))
						# plt.plot(xFit, full_InstrFun_biased(xFit, *fit[0]))
						# plt.pause(0.001)
						# if h==0:
						#	 print(fit[0][1])

						# record_all_sigma.append(fit[0][1])


					# iFit = np.logical_and(lamb[iTyp] > waveLengths[14]+np.diff(waveLengths)[14]/2, lamb[iTyp] < waveLengths[5]-np.diff(waveLengths)[4]/2)
					iFit = np.logical_and(lamb[iTyp] > waveLengths[total_number_of_lines_in_multipulse+first_line_in_multipulse+first-1]+np.diff(waveLengths)[total_number_of_lines_in_multipulse+first_line_in_multipulse+first-1]/2, lamb[iTyp] < 405)	# waveLengths[first_line_in_multipulse+first]-np.diff(waveLengths)[first_line_in_multipulse+first-1]/2)	# 2023-03-27 modified to easier calculation of background level
					iFit = np.logical_and(np.logical_and(iFit, np.arange(Nx)>20),np.arange(Nx)<Nx-20)	# added 16/02/2020 to avoid effects from tilting the image
					xFit = lamb[iTyp][iFit]
					yFit = binnedData[iBin, iFit]
					yFit_sigma = binnedData_sigma[iBin, iFit]

					true_sensitivity_full = sensitivity_calculator_with_uncertainty(xFit)	# counts W-1 m2 sr nm
					true_sensitivity_sigma = std_devs(true_sensitivity_full)	# counts W-1 m2 sr nm
					true_sensitivity = nominal_values(true_sensitivity_full)
					yFit_full = unumpy.uarray(yFit,yFit_sigma)	# counts

					yFit_sens = yFit_full / true_sensitivity_full
					yFit_sens_sigma = std_devs(yFit_sens)
					yFit_sens = nominal_values(yFit_sens)

					# to fit the line that overlap each other I do it in counts, to can keep the way I already developed


					# bds = [[0,0,0,0,0,0,0,0,0,0, -np.inf, -np.inf],[max(yFit) * 100,max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100,np.inf, np.inf]]
					# bds_2 = [[0,0,0,0,0,0,0,0,0,0, *(np.array(waveLengths[5:15])-0.15), -np.inf, -np.inf],[max(yFit) * 100,max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100,*(np.array(waveLengths[5:15])+0.15),np.inf, np.inf]]
					# bds_3 = [[0,0,0,0,0,0,0,0,0,0, min_sigma, *(np.array(waveLengths[5:15])-0.2), -np.inf, -np.inf],[max(yFit) * 100,max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max(yFit) * 100, max_sigma,*(np.array(waveLengths[5:15])+0.2),np.inf, np.inf]]
					# bds_3 = [[*np.zeros((total_number_of_lines_in_multipulse)), min_sigma, *(np.array(waveLengths[first_line_in_multipulse+first:total_number_of_lines_in_multipulse+first_line_in_multipulse+first])-0.2), -np.inf, -np.inf, -np.inf],[*(np.ones((total_number_of_lines_in_multipulse))*max(yFit) * 100), max_sigma,*(np.array(waveLengths[first_line_in_multipulse+first:total_number_of_lines_in_multipulse+first_line_in_multipulse+first])+0.2),0,np.inf, np.inf]]
					if pure_gaussian == True:
						success=9
						while success!=-1:
							# print(success)
							try:
								ipks = find_peaks(yFit)[0]
								# A0 = yFit[ipks[[np.abs((l0 - xFit[ipks])).argmin() for l0 in waveLengths[5:15]]]].tolist()
								A0 = yFit[ipks[[np.abs((l0 - xFit[ipks])).argmin() for l0 in waveLengths[first_line_in_multipulse+first:total_number_of_lines_in_multipulse+first_line_in_multipulse+first]]]].tolist()
								# A0.append(fit[0][2])
								# A0.append(fit[0][3])
								# fit = curve_fit(multPeakFun_biased, xFit, yFit, A0, maxfev=100000000, bounds=bds)
								A0.append(max(min(mean_sigma,max_sigma),min_sigma))
								# A0.extend(waveLengths[first_line_in_multipulse+first:total_number_of_lines_in_multipulse+first_line_in_multipulse+first])
								A0.append(record_peak_shift)
								ext_peak_locations = np.array(waveLengths[first_line_in_multipulse+first:total_number_of_lines_in_multipulse+first_line_in_multipulse+first])
								# A0.append(fit[0][3])
								# A0.append(fit[0][4])
								# A0.append(fit[0][5])
								# A0.extend(np.zeros((3)))
								# y_max_base_noise = np.mean(np.sort(binnedData[iBin, np.logical_and(lamb[iTyp]>402,lamb[iTyp]<405)])[:int(np.sum(np.logical_and(lamb[iTyp]>402,lamb[iTyp]<405))/3)])
								# y_max_base_noise_slope = np.mean(np.sort(binnedData[iBin, np.logical_and(lamb[iTyp]>391,lamb[iTyp]<394)])[:int(np.sum(np.logical_and(lamb[iTyp]>391,lamb[iTyp]<394))/3)])
								# y_max_base_noise_slope_2 = np.mean(np.sort(binnedData[iBin, np.logical_and(lamb[iTyp]>385.1,lamb[iTyp]<386.5)])[:int(np.sum(np.logical_and(lamb[iTyp]>385.1,lamb[iTyp]<386.5))/3)])
								# y_max_base_noise_slope_3 = np.mean(np.sort(binnedData[iBin, np.logical_and(lamb[iTyp]>374,lamb[iTyp]<375)])[:int(np.sum(np.logical_and(lamb[iTyp]>374,lamb[iTyp]<375))/3)])
								# y_min_base_noise = np.mean(np.sort(binnedData[iBin, np.logical_and(lamb[iTyp]>np.min(xFit)-10,lamb[iTyp]<np.min(xFit)+2)])[:int(np.sum(np.logical_and(lamb[iTyp]>np.min(xFit)-10,lamb[iTyp]<np.min(xFit)+2))/3)])
								# min_slope = np.max([0,np.min([(y_max_base_noise-y_max_base_noise_slope)/((401+404)/2 - (391+394)/2) , (y_max_base_noise-y_max_base_noise_slope_2)/((401+404)/2 - (385.1+386.5)/2) , (y_max_base_noise-y_max_base_noise_slope_3)/((401+404)/2 - (374+375)/2)])  ])
								# if fit[0][0]>3:	# I do it only if the last strong peak is actually strong
								#	 min_slope=np.max([min_slope,np.nanmean([np.diff(background_interpolation(xFit[-2:]+2))/np.diff(xFit[-2:]+2),np.diff(background_interpolation(xFit[-2:]))/np.diff(xFit[-2:])])])
								# A0.append(min(y_min_base_noise,max(0.1,y_max_base_noise-min_slope*(np.max(xFit) - np.min(xFit)))))
								# A0.append(y_max_base_noise)

								# start calculation of the background for the lines that overlap each other
								# y_max_base_noise = np.mean(np.sort(binnedData[iBin, np.logical_and(lamb[iTyp]>402,lamb[iTyp]<405)])[:int(np.sum(np.logical_and(lamb[iTyp]>402,lamb[iTyp]<405))/3)])
								# y_max_base_noise_slope = np.mean(np.sort(binnedData[iBin, np.logical_and(lamb[iTyp]>391,lamb[iTyp]<394)])[:int(np.sum(np.logical_and(lamb[iTyp]>391,lamb[iTyp]<394))/3)])
								# y_max_base_noise_slope_2 = np.mean(np.sort(binnedData[iBin, np.logical_and(lamb[iTyp]>385.1,lamb[iTyp]<386.5)])[:int(np.sum(np.logical_and(lamb[iTyp]>385.1,lamb[iTyp]<386.5))/3)])
								# y_max_base_noise_slope_3 = np.mean(np.sort(binnedData[iBin, np.logical_and(lamb[iTyp]>374,lamb[iTyp]<375)])[:int(np.sum(np.logical_and(lamb[iTyp]>374,lamb[iTyp]<375))/3)])
								# y_min_base_noise = max(0.1,np.mean(np.sort(binnedData[iBin, np.logical_and(lamb[iTyp]>np.min(xFit)-10,lamb[iTyp]<np.min(xFit)+2)])[:int(np.sum(np.logical_and(lamb[iTyp]>np.min(xFit)-10,lamb[iTyp]<np.min(xFit)+2))/3)]))
								# 2023-03-27: more general formula using yFit and xFit
								y_max_base_noise = np.mean(np.sort(yFit[ np.logical_and(xFit>402,xFit<405)])[:int(np.sum(np.logical_and(xFit>402,xFit<405))/3)])
								y_max_base_noise_slope = np.mean(np.sort(yFit[ np.logical_and(xFit>391,xFit<394)])[:int(np.sum(np.logical_and(xFit>391,xFit<394))/3)])
								y_max_base_noise_slope_2 = np.mean(np.sort(yFit[ np.logical_and(xFit>385.1,xFit<386.5)])[:int(np.sum(np.logical_and(xFit>385.1,xFit<386.5))/3)])
								y_max_base_noise_slope_3 = np.mean(np.sort(yFit[ np.logical_and(xFit>374,xFit<375)])[:int(np.sum(np.logical_and(xFit>374,xFit<375))/3)])
								y_min_base_noise = max(0.1/100,np.mean(np.sort(yFit[ np.logical_and(xFit>np.min(xFit)-10,xFit<np.min(xFit)+2)])[:int(np.sum(np.logical_and(xFit>np.min(xFit)-10,xFit<np.min(xFit)+2))/3)]))
								if background_interpolation(400)<100:
									min_slope = np.max([0,np.min([(y_max_base_noise-y_max_base_noise_slope)/((401+404)/2 - (391+394)/2) , (y_max_base_noise-y_max_base_noise_slope_2)/((401+404)/2 - (385.1+386.5)/2) , (y_max_base_noise-y_max_base_noise_slope_3)/((401+404)/2 - (374+375)/2)])  ])
									if fit[0][0]>3:	# I do it only if the last strong peak is actually strong
										min_slope=np.max([min_slope,np.nanmean([np.diff(background_interpolation(xFit[-2:]+2))/np.diff(xFit[-2:]+2),np.diff(background_interpolation(xFit[-2:]))/np.diff(xFit[-2:])])])
									A0.append(min(y_min_base_noise,max(0.1,y_max_base_noise-min_slope*(np.max(xFit) - np.min(xFit)))))
									# A0.append(y_max_base_noise)
									A0.append(background_interpolation(xFit.max()))
									min_slope = min(min_slope,y_max_base_noise/(np.max(xFit)-np.min(xFit)))	#not sure if usefull, added to avoid the background to go negative, 26/06/2020
									max_slope = max(0.1,1.1 * min_slope)
									min_y_min_base_noise = 0
									A0.append(min(max_slope,max(0.1,min_slope)))
									min_slope = min_slope*0.9
									A0.append(1)
									# bds_3 = [[*np.zeros((total_number_of_lines_in_multipulse)), min_sigma, *(np.array(waveLengths[first_line_in_multipulse+first:total_number_of_lines_in_multipulse+first_line_in_multipulse+first])-0.5), min_y_min_base_noise, 0.5*y_max_base_noise,min_slope,1-1e-6],[*(np.ones((total_number_of_lines_in_multipulse))*max(yFit) * 100), max_sigma,*(np.array(waveLengths[first_line_in_multipulse+first:total_number_of_lines_in_multipulse+first_line_in_multipulse+first])+0.5),y_min_base_noise,1*y_max_base_noise,max(0.1,max_slope),1+1e-6]]
									bds_4 = [[*np.zeros((total_number_of_lines_in_multipulse)), min_sigma, min(record_peak_shift,-0.5), min_y_min_base_noise, background_interpolation(xFit.max())*0.9,min_slope,1-1e-6],[*(np.ones((total_number_of_lines_in_multipulse))*max(yFit) * 100), max_sigma,max(record_peak_shift,+0.5),y_min_base_noise,background_interpolation(xFit.max())*1.1,max(0.1,max_slope),1+1e-6]]
								else:
									temp = np.array([(y_max_base_noise-y_max_base_noise_slope)/((401+404)/2 - (391+394)/2) , (y_max_base_noise-y_max_base_noise_slope_2)/((401+404)/2 - (385.1+386.5)/2) , (y_max_base_noise-y_max_base_noise_slope_3)/((401+404)/2 - (374+375)/2)])
									temp = temp[temp>0]
									min_slope = np.max([0,np.min(temp)])
									if fit[0][0]>3:	# I do it only if the last strong peak is actually strong
										min_slope=np.max([min_slope,np.nanmean([np.diff(background_interpolation(xFit[-2:]+2))/np.diff(xFit[-2:]+2),np.diff(background_interpolation(xFit[-2:]))/np.diff(xFit[-2:])])])
									A0.append(min(y_min_base_noise,max(0.1,y_max_base_noise-min_slope*(np.max(xFit) - np.min(xFit)))))
									# A0.append(y_max_base_noise)
									A0.append(background_interpolation(xFit.max()))
									max_slope = max(0.1,max(1.5*y_max_base_noise/(np.max(xFit)-np.min(xFit)),2*min_slope))
									# min_slope = min(min_slope,y_max_base_noise/(np.max(xFit)-np.min(xFit)))	#not sure if usefull, added to avoid the background to go negative, 26/06/2020
									min_y_min_base_noise = -np.inf
									A0.append(max(max_slope,2*max(0.1,min_slope)))
									A0.append(3)
									# bds_3 = [[*np.zeros((total_number_of_lines_in_multipulse)), min_sigma, *(np.array(waveLengths[first_line_in_multipulse+first:total_number_of_lines_in_multipulse+first_line_in_multipulse+first])-0.5), -np.inf, 0.3*y_max_base_noise,min_slope],[*(np.ones((total_number_of_lines_in_multipulse))*max(yFit) * 100), max_sigma,*(np.array(waveLengths[first_line_in_multipulse+first:total_number_of_lines_in_multipulse+first_line_in_multipulse+first])+0.5),min(1.1*y_min_base_noise,y_max_base_noise-min_slope*(np.max(xFit) - np.min(xFit))),1*y_max_base_noise,np.inf]]
									# bds_3 = [[*np.zeros((total_number_of_lines_in_multipulse)), min_sigma, *(np.array(waveLengths[first_line_in_multipulse+first:total_number_of_lines_in_multipulse+first_line_in_multipulse+first])-0.5), min_y_min_base_noise, 0.5*y_max_base_noise,min_slope,0.8],[*(np.ones((total_number_of_lines_in_multipulse))*max(yFit) * 100), max_sigma,*(np.array(waveLengths[first_line_in_multipulse+first:total_number_of_lines_in_multipulse+first_line_in_multipulse+first])+0.5),y_min_base_noise,1*y_max_base_noise,max(0.1,max_slope),4]]
									bds_4 = [[*np.zeros((total_number_of_lines_in_multipulse)), min_sigma, min(record_peak_shift,-0.5), min_y_min_base_noise, background_interpolation(xFit.max())*0.9,min_slope,0.8],[*(np.ones((total_number_of_lines_in_multipulse))*max(yFit) * 100), max_sigma,max(record_peak_shift,+0.5),y_min_base_noise,background_interpolation(xFit.max())*1.1,max(0.1,max_slope),4]]
								if total_number_of_lines_in_multipulse==10:
									# fit_multiline = curve_fit(multPeakFun_biased_10_3, xFit, yFit, A0,sigma=yFit_sigma, absolute_sigma=True, maxfev=10000000, bounds=bds_3,ftol=1e-13,xtol=1e-12)
									# fit_multiline_no_pos = np.concatenate((fit_multiline[0][0:11],fit_multiline[0][-4:]))
									# bds_3_stark = [[*np.zeros((total_number_of_lines_in_multipulse)), min_sigma, *(np.min([1.2*fit_multiline_no_pos[-4:],0.8*fit_multiline_no_pos[-4:]],axis=0))],[*(np.ones((total_number_of_lines_in_multipulse))*max(yFit) * 100), max_sigma,*(np.max([1.2*fit_multiline_no_pos[-4:],0.8*fit_multiline_no_pos[-4:]],axis=0))]]
									# fit_multiline_stark = curve_fit(multPeakFun_biased_10_3_stark(stark_high_lines_amp,stark_high_lines_sigma,stark_high_lines_dist,fit_multiline[0][total_number_of_lines_in_multipulse+1:2*total_number_of_lines_in_multipulse+1]), xFit, yFit, fit_multiline_no_pos,sigma=yFit_sigma, absolute_sigma=True, maxfev=10000000, bounds=bds_3_stark,ftol=1e-13)

									# guidance: multPeakFun_biased_10_4(ext_peak_locations) = function of
									# x, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, sig, delta_x0, y_min,y_max, m,P
									fit_multiline = curve_fit(multPeakFun_biased_10_4(ext_peak_locations), xFit, yFit, A0,sigma=yFit_sigma, absolute_sigma=True, maxfev=10000000, bounds=bds_4,ftol=1e-13,xtol=1e-13)
									fit_multiline_no_pos = np.concatenate((fit_multiline[0][0:11],fit_multiline[0][-4:]))
									bds_4_stark = [[*np.zeros((total_number_of_lines_in_multipulse)), min_sigma, *(np.min([1.2*fit_multiline_no_pos[-4:],0.8*fit_multiline_no_pos[-4:]],axis=0))],[*(np.ones((total_number_of_lines_in_multipulse))*max(yFit) * 100), max_sigma,*(np.max([1.2*fit_multiline_no_pos[-4:],0.8*fit_multiline_no_pos[-4:]],axis=0))]]
									fit_multiline_stark = curve_fit(multPeakFun_biased_10_3_stark(stark_high_lines_amp,stark_high_lines_sigma,stark_high_lines_dist,ext_peak_locations+fit_multiline[0][total_number_of_lines_in_multipulse+1]), xFit, yFit,fit_multiline_no_pos,sigma=yFit_sigma, absolute_sigma=True, maxfev=10000000, bounds=bds_4_stark,ftol=1e-13)
								elif total_number_of_lines_in_multipulse==6:
									fit_multiline = curve_fit(multPeakFun_biased_6_3, xFit, yFit, A0,sigma=yFit_sigma, absolute_sigma=True, maxfev=10000000, bounds=bds_3,ftol=1e-13)
								else:
									print('you asked to put '+str(total_number_of_lines_in_multipulse)+' lines in the multPeakFun_biased routine\nbut now it can accept only 6 or 10')
									exit()
								# peakUnitArea = np.abs(np.trapz(full_InstrFun(lamb[1], fit_multiline[0][total_number_of_lines_in_multipulse], 1, np.mean(lamb[1])), x=lamb[1]))
								success=-1
								fit_multiline_only_bias = np.array([xFit.min(),xFit.max()]+fit_multiline[0][-4:].tolist())
								# plt.figure()
								# plt.plot(xFit, yFit)
								# # plt.plot(xFit, multPeakFun(xFit, *fit[0]))
								# plt.plot(xFit, multPeakFun_biased_10(xFit, *fit_multiline[0]))
								# plt.pause(0.001)
								if  (to_print and iBin in [5,20,30]):
									total_stark_emission = 0
									for iline in range(total_number_of_lines_in_multipulse):
										if fit_multiline[0][iline]<0:
											fit_multiline[0][iline]=0
										if fit_multiline[0][iline]>0:
											# temp_amplitudes = np.zeros((len(waveLengths[5:15])))
											temp_amplitudes = np.zeros((total_number_of_lines_in_multipulse))
											temp_amplitudes[iline]=fit_multiline[0][iline]
											fit_multiline_iline = cp.deepcopy(fit_multiline)
											fit_multiline_iline[0][np.logical_and(np.arange(len(fit_multiline[0]))!=iline,np.arange(len(fit_multiline[0]))<=total_number_of_lines_in_multipulse-1)] = 0
											fit_multiline_iline[1][np.logical_and(np.arange(len(fit_multiline[0]))!=iline,np.arange(len(fit_multiline[0]))<=total_number_of_lines_in_multipulse-1),:] = 0
											fit_multiline_iline[1][:,np.logical_and(np.arange(len(fit_multiline[0]))!=iline,np.arange(len(fit_multiline[0]))<=total_number_of_lines_in_multipulse-1)] = 0
											fit_multiline_iline = correlated_values(fit_multiline_iline[0],fit_multiline_iline[1])
											if total_number_of_lines_in_multipulse==10:
												# # temp = multPeakFun_biased_10_3_with_error_included(xFit,*fit_multiline_iline)
												# temp = multPeakFun_biased_10_3(xFit,*nominal_values(fit_multiline_iline))
												# # plt.plot(xFit,multPeakFun_biased_10_3(xFit,*temp_amplitudes,*fit_multiline[0][total_number_of_lines_in_multipulse:]),label='n='+str(iline+3+first+first_line_in_multipulse)+'->2, x0=%.5g+/-%.3g' % (fit_multiline[0][total_number_of_lines_in_multipulse+1+iline],(fit_multiline[1][total_number_of_lines_in_multipulse+1+iline,total_number_of_lines_in_multipulse+1+iline])**0.5) + ' instead of %.5g nm' % A0[total_number_of_lines_in_multipulse+1+iline]+ ', amp=%.3g+/-%.3g au' % (fit_multiline[0][iline],(fit_multiline[1][iline,iline])**0.5))
												# plt.plot(xFit,temp,label='n='+str(iline+3+first+first_line_in_multipulse)+'->2, x0=%.5g+/-%.3g' % (fit_multiline[0][total_number_of_lines_in_multipulse+1+iline],(fit_multiline[1][total_number_of_lines_in_multipulse+1+iline,total_number_of_lines_in_multipulse+1+iline])**0.5) + ' instead of %.5g nm' % A0[total_number_of_lines_in_multipulse+1+iline]+ ', amp=%.3g+/-%.3g au' % (fit_multiline[0][iline]/((2*np.pi*fit_multiline[0][total_number_of_lines_in_multipulse]**2)**0.5),(fit_multiline[1][iline,iline])**0.5/(2*np.pi*fit_multiline[0][total_number_of_lines_in_multipulse]**2)))
												# plt.plot([A0[total_number_of_lines_in_multipulse+1+iline],A0[total_number_of_lines_in_multipulse+1+iline]],nominal_values([np.min(temp),np.max(temp)]),'--',color='r',linewidth=0.5)
												# plt.plot([fit_multiline[0][total_number_of_lines_in_multipulse+1+iline],fit_multiline[0][total_number_of_lines_in_multipulse+1+iline]],nominal_values([np.min(temp),np.max(temp)]),'--',color='k',linewidth=0.5)
												temp = multPeakFun_biased_10_4(ext_peak_locations)(xFit,*nominal_values(fit_multiline_iline))/true_sensitivity
												plt.plot(xFit,temp,label='n='+str(iline+3+first+first_line_in_multipulse)+'->2, x0=%.5g+/-%.3g' % (ext_peak_locations[iline]+fit_multiline[0][total_number_of_lines_in_multipulse+1],(fit_multiline[1][total_number_of_lines_in_multipulse+1,total_number_of_lines_in_multipulse+1])**0.5) + ' instead of %.5g nm' % ext_peak_locations[iline]+ ', amp=%.3g+/-%.3g au' % (fit_multiline[0][iline]/((2*np.pi*fit_multiline[0][total_number_of_lines_in_multipulse]**2)**0.5),(fit_multiline[1][iline,iline])**0.5/(2*np.pi*fit_multiline[0][total_number_of_lines_in_multipulse]**2)))
												plt.plot([ext_peak_locations[iline],ext_peak_locations[iline]],nominal_values([np.min(temp),np.max(temp)]),'--',color='r',linewidth=0.5)
												plt.plot([ext_peak_locations[iline]+fit_multiline[0][total_number_of_lines_in_multipulse+1],ext_peak_locations[iline]+fit_multiline[0][total_number_of_lines_in_multipulse+1]],nominal_values([np.min(temp),np.max(temp)]),'--',color='k',linewidth=0.5)
											elif total_number_of_lines_in_multipulse==6:
												# # temp = multPeakFun_biased_6_3_with_error_included(xFit,*fit_multiline_iline)
												# temp = multPeakFun_biased_6_3(xFit,*nominal_values(fit_multiline_iline))
												# # plt.plot(xFit,multPeakFun_biased_6_3(xFit,*temp_amplitudes,*fit_multiline[0][total_number_of_lines_in_multipulse:]),label='n='+str(iline+3+first+first_line_in_multipulse)+'->2, x0=%.5g+/-%.3g' % (fit_multiline[0][total_number_of_lines_in_multipulse+1+iline],(fit_multiline[1][total_number_of_lines_in_multipulse+1+iline,total_number_of_lines_in_multipulse+1+iline])**0.5) + ' instead of %.5g nm' % A0[total_number_of_lines_in_multipulse+1+iline]+ ', amp=%.3g+/-%.3g au' % (fit_multiline[0][iline],(fit_multiline[1][iline,iline])**0.5))
												# plt.plot(xFit,temp,label='n='+str(iline+3+first+first_line_in_multipulse)+'->2, x0=%.5g+/-%.3g' % (fit_multiline[0][total_number_of_lines_in_multipulse+1+iline],(fit_multiline[1][total_number_of_lines_in_multipulse+1+iline,total_number_of_lines_in_multipulse+1+iline])**0.5) + ' instead of %.5g nm' % A0[total_number_of_lines_in_multipulse+1+iline]+ ', amp=%.3g+/-%.3g au' % (fit_multiline[0][iline]/((2*np.pi*fit_multiline[0][total_number_of_lines_in_multipulse]**2)**0.5),(fit_multiline[1][iline,iline])**0.5/(2*np.pi*fit_multiline[0][total_number_of_lines_in_multipulse]**2)))
												# plt.plot([A0[total_number_of_lines_in_multipulse+1+iline],A0[total_number_of_lines_in_multipulse+1+iline]],nominal_values([np.min(temp),np.max(temp)]),'--',color='r',linewidth=0.5)
												# plt.plot([fit_multiline[0][total_number_of_lines_in_multipulse+1+iline],fit_multiline[0][total_number_of_lines_in_multipulse+1+iline]],nominal_values([np.min(temp),np.max(temp)]),'--',color='k',linewidth=0.5)
												temp = multPeakFun_biased_6_4(xFit,*nominal_values(fit_multiline_iline))/true_sensitivity
												plt.plot(xFit,temp,label='n='+str(iline+3+first+first_line_in_multipulse)+'->2, x0=%.5g+/-%.3g' % (ext_peak_locations[iline]+fit_multiline[0][total_number_of_lines_in_multipulse+1],(fit_multiline[1][total_number_of_lines_in_multipulse+1,total_number_of_lines_in_multipulse+1])**0.5) + ' instead of %.5g nm' % ext_peak_locations[iline]+ ', amp=%.3g+/-%.3g au' % (fit_multiline[0][iline]/((2*np.pi*fit_multiline[0][total_number_of_lines_in_multipulse]**2)**0.5),(fit_multiline[1][iline,iline])**0.5/(2*np.pi*fit_multiline[0][total_number_of_lines_in_multipulse]**2)))
												plt.plot([ext_peak_locations[iline],ext_peak_locations[iline]],nominal_values([np.min(temp),np.max(temp)]),'--',color='r',linewidth=0.5)
												plt.plot([ext_peak_locations[iline]+fit_multiline[0][total_number_of_lines_in_multipulse+1],ext_peak_locations[iline]+fit_multiline[0][total_number_of_lines_in_multipulse+1]],nominal_values([np.min(temp),np.max(temp)]),'--',color='k',linewidth=0.5)
											else:
												print('you asked to put '+str(total_number_of_lines_in_multipulse)+' lines in the multPeakFun_biased routine\nbut now it can accept only 6 or 10')
												exit()
											stark_emission = stark_gaussian(xFit,fit_multiline[0][iline]*stark_high_lines_amp[iline],fit_multiline[0][total_number_of_lines_in_multipulse]*stark_high_lines_sigma[iline],ext_peak_locations[iline]+fit_multiline[0][total_number_of_lines_in_multipulse+1],stark_high_lines_dist[iline],1)
											total_stark_emission+=stark_emission/true_sensitivity
											# plt.plot(xFit,stark_emission,label='stark n='+str(iline+3+first+first_line_in_multipulse))
									extended_y_min = (fit_multiline[0][-4] - fit_multiline[0][-3] - fit_multiline[0][-2]*(np.min(xFit)-np.max(xFit)))*(((np.min(lamb[iTyp][lamb[iTyp]>350])-np.max(xFit))**2)**fit_multiline[0][-1])/(((np.min(xFit)-np.max(xFit))**2)**fit_multiline[0][-1]) + fit_multiline[0][-2]*(np.min(lamb[iTyp][lamb[iTyp]>350])-np.max(xFit))+fit_multiline[0][-3]
									fit_multiline_2 = cp.deepcopy(fit_multiline)
									# fit_multiline_2[0][-3]=extended_y_min
									# fit_multiline_only_bias[0][:-4]=0
									# fit_multiline_only_bias[0][-4]=extended_y_min
									# fit_multiline_only_bias[1][:-4,:]=0
									# fit_multiline_only_bias[1][:,:-4]=0
									# fit_multiline_only_bias[0][total_number_of_lines_in_multipulse]=1
									if total_number_of_lines_in_multipulse==10:
										# # plt.plot(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],multPeakFun_biased_10_3(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],*fit_multiline[0][:-3],extended_y_min,*fit_multiline[0][-2:]),'r--',linewidth=0.6,label='sum of n='+str(3+first+first_line_in_multipulse)+'/'+str(3+first+total_number_of_lines_in_multipulse+first_line_in_multipulse-1)+'->2, \u03C3=%.4g+/-%.3g nm' %(fit_multiline[0][total_number_of_lines_in_multipulse],(fit_multiline[1][total_number_of_lines_in_multipulse,total_number_of_lines_in_multipulse])**0.5))
										# temp = multPeakFun_biased_10_3_with_error_included(xFit,*correlated_values(fit_multiline_2[0],fit_multiline_2[1]))
										# plt.errorbar(xFit,nominal_values(temp),linestyle='--',color='r',yerr=std_devs(temp),linewidth=1,label='sum of n='+str(3+first+first_line_in_multipulse)+'/'+str(3+first+total_number_of_lines_in_multipulse+first_line_in_multipulse-1)+'->2, \u03C3=%.4g+/-%.3g nm' %(fit_multiline[0][total_number_of_lines_in_multipulse],(fit_multiline[1][total_number_of_lines_in_multipulse,total_number_of_lines_in_multipulse])**0.5))
										# # plt.plot(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],multPeakFun_biased_10_3(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],*np.zeros((total_number_of_lines_in_multipulse)),*fit_multiline[0][total_number_of_lines_in_multipulse:-3],extended_y_min,*fit_multiline[0][-2:]),'k--',label='background, quadratic coeff on multipeak %.5g' % ((fit_multiline[0][-3] - fit_multiline[0][-2] - fit_multiline[0][-1]*(np.min(xFit)-np.max(xFit)))/((np.min(xFit)-np.max(xFit))**2)) )
										# # temp1 = multPeakFun_biased_10_3_with_error_included(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],*correlated_values(fit_multiline_only_bias[0],fit_multiline_only_bias[1]))
										# temp1 = multPeakFun_biased_10_3(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],*fit_multiline_only_bias[0])
										# plt.plot(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],temp1,'k--',label='background, count min=%.3g, a=%.3g, b=%.3g, c=%.3g, P=%.3g' % (fit_multiline[0][-4],(fit_multiline[0][-4] - fit_multiline[0][-3] - fit_multiline[0][-2]*(np.min(xFit)-np.max(xFit)))/((np.min(xFit)-np.max(xFit))**2),fit_multiline[0][-2],fit_multiline[0][-3],fit_multiline[0][-1]) )
										# plt.plot(xFit,multPeakFun_biased_10_3_stark(stark_high_lines_amp,stark_high_lines_sigma,stark_high_lines_dist,fit_multiline[0][total_number_of_lines_in_multipulse+1:2*total_number_of_lines_in_multipulse+1])(xFit,*fit_multiline_stark[0]),'--',label='stark fit multi peak')
										# plt.plot(xFit,multPeakFun_biased_10_3_stark(stark_high_lines_amp,stark_high_lines_sigma,stark_high_lines_dist,fit_multiline[0][total_number_of_lines_in_multipulse+1:2*total_number_of_lines_in_multipulse+1])(xFit,*np.zeros((total_number_of_lines_in_multipulse)),*fit_multiline_stark[0][total_number_of_lines_in_multipulse:]),'--',label='stark background')
										# plt.plot(xFit,multPeakFun_biased_10_3_stark(stark_high_lines_amp,stark_high_lines_sigma,stark_high_lines_dist,fit_multiline[0][total_number_of_lines_in_multipulse+1:2*total_number_of_lines_in_multipulse+1])(xFit,*fit_multiline_no_pos),'--',label='stark initial guess')
										temp = multPeakFun_biased_10_4_with_error_included(ext_peak_locations)(xFit,*correlated_values(fit_multiline_2[0],fit_multiline_2[1]))/true_sensitivity
										plt.errorbar(xFit,nominal_values(temp),linestyle='--',color='r',yerr=std_devs(temp),linewidth=1,label='sum of n='+str(3+first+first_line_in_multipulse)+'/'+str(3+first+total_number_of_lines_in_multipulse+first_line_in_multipulse-1)+'->2, \u03C3=%.4g+/-%.3g nm' %(fit_multiline[0][total_number_of_lines_in_multipulse],(fit_multiline[1][total_number_of_lines_in_multipulse,total_number_of_lines_in_multipulse])**0.5))
										gna = (((sensitivity_density_interpolator(lamb[iTyp]-dlamb/2) + sensitivity_density_interpolator(lamb[iTyp]+dlamb/2))/2)*dlamb)[np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))]
										temp1 = multPeakFun_biased_only_bias()(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],*fit_multiline_only_bias)/gna
										plt.plot(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],temp1,'k--',label='background, count min=%.3g, a=%.3g, b=%.3g, c=%.3g, P=%.3g' % (fit_multiline[0][-4],(fit_multiline[0][-4] - fit_multiline[0][-3] - fit_multiline[0][-2]*(np.min(xFit)-np.max(xFit)))/((np.min(xFit)-np.max(xFit))**2),fit_multiline[0][-2],fit_multiline[0][-3],fit_multiline[0][-1]) )
										plt.plot(xFit,multPeakFun_biased_10_3_stark(stark_high_lines_amp,stark_high_lines_sigma,stark_high_lines_dist,ext_peak_locations+fit_multiline[0][total_number_of_lines_in_multipulse+1])(xFit,*fit_multiline_stark[0])/true_sensitivity,'--',label='stark fit multi peak')
										plt.plot(xFit,multPeakFun_biased_10_3_stark(stark_high_lines_amp,stark_high_lines_sigma,stark_high_lines_dist,ext_peak_locations+fit_multiline[0][total_number_of_lines_in_multipulse+1])(xFit,*np.zeros((total_number_of_lines_in_multipulse)),*fit_multiline_stark[0][total_number_of_lines_in_multipulse:])/true_sensitivity,'--',label='stark background')
										plt.plot(xFit,multPeakFun_biased_10_3_stark(stark_high_lines_amp,stark_high_lines_sigma,stark_high_lines_dist,ext_peak_locations+fit_multiline[0][total_number_of_lines_in_multipulse+1])(xFit,*fit_multiline_no_pos)/true_sensitivity,'--',label='stark initial guess')
									elif total_number_of_lines_in_multipulse==6:
										# plt.plot(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],multPeakFun_biased_6_3(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],*fit_multiline[0][:-3],extended_y_min,*fit_multiline[0][-2:]),'r--',linewidth=0.6,label='sum of n='+str(3+first+first_line_in_multipulse)+'/'+str(3+first+total_number_of_lines_in_multipulse+first_line_in_multipulse-1)+'->2, \u03C3=%.4g+/-%.3g nm' %(fit_multiline[0][total_number_of_lines_in_multipulse],(fit_multiline[1][total_number_of_lines_in_multipulse,total_number_of_lines_in_multipulse])**0.5))
										temp = multPeakFun_biased_6_3_with_error_included(xFit,*correlated_values(fit_multiline_2[0],fit_multiline_2[1]))/true_sensitivity/true_sensitivity
										plt.errorbar(xFit,nominal_values(temp),'r--',yerr=std_devs(temp),linewidth=1,label='sum of n='+str(3+first+first_line_in_multipulse)+'/'+str(3+first+total_number_of_lines_in_multipulse+first_line_in_multipulse-1)+'->2, \u03C3=%.4g+/-%.3g nm' %(fit_multiline[0][total_number_of_lines_in_multipulse],(fit_multiline[1][total_number_of_lines_in_multipulse,total_number_of_lines_in_multipulse])**0.5))
										# plt.plot(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],multPeakFun_biased_6_3(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],*np.zeros((total_number_of_lines_in_multipulse)),*fit_multiline[0][total_number_of_lines_in_multipulse:-3],extended_y_min,*fit_multiline[0][-2:]),'k--',label='background, quadratic coeff on multipeak %.5g' % ((fit_multiline[0][-3] - fit_multiline[0][-2] - fit_multiline[0][-1]*(np.min(xFit)-np.max(xFit)))/((np.min(xFit)-np.max(xFit))**2)) )
										# temp1 = multPeakFun_biased_10_3_with_error_included(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],*correlated_values(fit_multiline_only_bias[0],fit_multiline_only_bias[1]))
										temp1 = multPeakFun_biased_6_3(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],*fit_multiline_only_bias[0])/true_sensitivity
										plt.plot(lamb[iTyp][np.logical_and(lamb[iTyp]>350,lamb[iTyp]<=np.max(xFit))],temp1,'k--',label='background, count min=%.3g, a=%.3g, b=%.3g, c=%.3g' % (fit_multiline[0][-4],(fit_multiline[0][-4] - fit_multiline[0][-3] - fit_multiline[0][-2]*(np.min(xFit)-np.max(xFit)))/((np.min(xFit)-np.max(xFit))**2),fit_multiline[0][-2],fit_multiline[0][-3]) )
									else:
										print('you asked to put '+str(total_number_of_lines_in_multipulse)+' lines in the multPeakFun_biased routine\nbut now it can accept only 6 or 10')
										exit()
									plt.plot(xFit,total_stark_emission+multPeakFun_biased_only_bias()(xFit,*fit_multiline_only_bias)/true_sensitivity,label='expected stark multi peak emission')
									# plt.plot(xFit,nominal_values(temp)+total_stark_emission,label='total (+stark) multi peak')
									# plt.plot(xFit,multPeakFun_biased_10(xFit,np.zeros(iline-1)*fit_multiline[0]),label='n=8-18->2')
							except:
								bds_4[0][success] = -abs(max(yFit)) * 100
								if success<=4:
									print('bin '+str(iBin)+' line '+str(success+first_line_in_multipulse+1+3)+' constrain on positive peak removed')
								success -= 1
								fit_multiline = np.array([A0])
								fit_multiline[0,:total_number_of_lines_in_multipulse]=0
						for iline in range(total_number_of_lines_in_multipulse):
							if fit_multiline[0][iline]<0:
								fit_multiline[0][iline]=0
						pixel_location_of_the_peak = (np.abs(lamb[iTyp] - np.array([ext_peak_locations + fit_multiline[0][total_number_of_lines_in_multipulse+1]]).T)).argmin(axis=1)
						wavel_location_of_the_peak = ext_peak_locations + fit_multiline[0][total_number_of_lines_in_multipulse+1]
						# sensitivity = binnedSens[iTyp, iBin, (np.abs(lamb[iTyp] - np.array([fit_multiline[0][11:21]]).T)).argmin(axis=1)]
					else:
						bgFit = np.polyfit(xFit[[0, -1]], yFit[[0, -1]], 1)
						yFit -= np.polyval(bgFit, xFit)
						ipks = find_peaks(yFit)[0]
						A0 = yFit[ipks[[(np.abs(l0 - xFit[ipks])).argmin() for l0 in waveLengths[first_line_in_multipulse+first:total_number_of_lines_in_multipulse+first_line_in_multipulse+first]]]]
						fit = curve_fit(multPeakFun, xFit, yFit, A0)
						sensitivity = binnedSens[iTyp, iBin, iSens[iTyp][first:10]]

					if perform_convolution==True:
						# sensitivity = binnedSens[iTyp, iBin]
						# sensitivity_sigma = binnedSens_sigma[iTyp, iBin]
						fitted_profile_no_bias = multPeakFun_10_4_with_error_included(ext_peak_locations)(xFit,*correlated_values(fit_multiline[0][:-4],fit_multiline[1][:-4,:-4]))
						multPeakFun_bias = multPeakFun_biased_only_bias()(xFit,*fit_multiline_only_bias)
						select_fitted_profile_no_bias_higher_than_threshold = fitted_profile_no_bias>1e-10	# arbitrary to avoid numerical problems
						# fitted_profile_no_bias = fitted_profile_no_bias[select_fitted_profile_no_bias_higher_than_threshold]
						true_sensitivity_full = sensitivity_calculator_with_uncertainty(xFit)	# counts W-1 m2 sr nm
						true_sensitivity_sigma = std_devs(true_sensitivity_full)	# counts W-1 m2 sr nm
						true_sensitivity = nominal_values(true_sensitivity_full)
						# plt.plot(xFit,(yFit-multPeakFun_bias)/true_sensitivity,'--')
						for line_index,line in enumerate(wavel_location_of_the_peak[:len(dfs_lineRads[iTyp].loc[iBin])-first_line_in_multipulse]):
							fit_line=[[],[]]
							for i_coefficient in range(len(fit_multiline[0])):
								# if i_coefficient in [line_index,total_number_of_lines_in_multipulse,line_index+total_number_of_lines_in_multipulse+1]:
								if i_coefficient in [line_index,total_number_of_lines_in_multipulse,total_number_of_lines_in_multipulse+1]:
									fit_line[0].append(fit_multiline[0][i_coefficient])
									temp = []
									for j_coefficient in range(len(fit_multiline[0])):
										if j_coefficient in [line_index,total_number_of_lines_in_multipulse,total_number_of_lines_in_multipulse+1]:
											temp.append(fit_multiline[1][i_coefficient,j_coefficient])
									fit_line[1].append(temp)
							coeff = np.array(correlated_values(fit_line[0],fit_line[1]))
							coeff[2]+=ext_peak_locations[line_index]
							# yFit = unumpy.uarray(sensitivity,sensitivity_sigma)
							width_to_consider = np.nanmax([nominal_values(coeff[1])*3,stark_high_lines_sigma[line_index]*nominal_values(coeff[1])*2])	# I use the stark forecast width as stark width
							select = np.logical_and(xFit>nominal_values(coeff[2])-width_to_consider,xFit<nominal_values(coeff[2])+width_to_consider)
							select = np.logical_and(select,select_fitted_profile_no_bias_higher_than_threshold)
							# temp_gaus_peak = full_InstrFun_with_error_included(xFit, *coeff)
							# plt.plot(xFit[select],nominal_values((yFit-multPeakFun_bias)*temp_gaus_peak/fitted_profile_no_bias)[select]/(true_sensitivity[select]))
							temp_gaus_peak = full_InstrFun_with_error_included(xFit, *coeff)[select]# + multPeakFun_biased_only_bias()(lamb[iTyp],*fit_multiline_only_bias)
							temp = np.trapz((unumpy.uarray((yFit-multPeakFun_bias),yFit_sigma)[select]/(true_sensitivity_full[select])) * (temp_gaus_peak/(fitted_profile_no_bias[select])) , x=xFit[select])	# W m-2 sr-1
							# temp = np.abs(np.trapz(temp_gaus_peak/temp_sensitivity , x=lamb[iTyp]))
							# dfs_lineRads[iTyp].loc[iBin][first_line_in_multipulse+line_index] = np.abs(np.trapz(gaus_peak/sensitivity , x=lamb[iTyp]))
							# df_lineRad.loc[iBin][first_line_in_multipulse+first+line_index] = np.abs(np.trapz(gaus_peak/sensitivity , x=lamb[iTyp]))
							# gaus_peak_sigma = full_InstrFun_sigma(lamb[iTyp],fit_multiline[0][line_index],fit_multiline[0][total_number_of_lines_in_multipulse],line,(fit_multiline[1][line_index,line_index])**0.5,(fit_multiline[1][total_number_of_lines_in_multipulse,total_number_of_lines_in_multipulse])**0.5,(fit_multiline[1][total_number_of_lines_in_multipulse+1+line_index,total_number_of_lines_in_multipulse+1+line_index])**0.5)
							# dfs_lineRads_sigma[iTyp].loc[iBin][first_line_in_multipulse+line_index] = np.sqrt(np.nansum( ((lamb[iTyp][1:]-lamb[iTyp][:-1])**2)*((gaus_peak[1:]/sensitivity[1:])**2 *( (gaus_peak_sigma[1:]/gaus_peak[1:])**2 + (sensitivity_sigma[1:]/sensitivity[1:])**2) + (gaus_peak[:-1]/sensitivity[:-1])**2 *( (gaus_peak_sigma[:-1]/gaus_peak[:-1])**2 + (sensitivity_sigma[:-1]/sensitivity[:-1])**2)) ))/2
							# df_lineRad_sigma.loc[iBin][first_line_in_multipulse+first+line_index] = np.sqrt(np.nansum( ((lamb[iTyp][1:]-lamb[iTyp][:-1])**2)*((gaus_peak[1:]/sensitivity[1:])**2 *( (gaus_peak_sigma[1:]/gaus_peak[1:])**2 + (sensitivity_sigma[1:]/sensitivity[1:])**2) + (gaus_peak[:-1]/sensitivity[:-1])**2 *( (gaus_peak_sigma[:-1]/gaus_peak[:-1])**2 + (sensitivity_sigma[:-1]/sensitivity[:-1])**2)) ))/2
							dfs_lineRads[iTyp].loc[iBin][first_line_in_multipulse+line_index] = nominal_values(temp)
							df_lineRad.loc[iBin][first_line_in_multipulse+first+line_index] = nominal_values(temp)
							dfs_lineRads_sigma[iTyp].loc[iBin][first_line_in_multipulse+line_index] = std_devs(temp)
							df_lineRad_sigma.loc[iBin][first_line_in_multipulse+first+line_index] = std_devs(temp)
					else:
						sensitivity = binnedSens[iTyp, iBin, pixel_location_of_the_peak]
						peakUnitArea = np.abs(np.trapz(full_InstrFun(lamb[1], fit_multiline[0][total_number_of_lines_in_multipulse], 1, np.mean(wavel_location_of_the_peak)), x=lamb[1]))
						if first < 6:
							dfs_lineRads[iTyp].loc[iBin][first_line_in_multipulse:] = fit_multiline[0][:len(dfs_lineRads[iTyp].loc[iBin])-first_line_in_multipulse]*peakUnitArea / sensitivity[:len(dfs_lineRads[iTyp].loc[iBin])-first_line_in_multipulse]
							df_lineRad.loc[iBin][first_line_in_multipulse+first:] = fit_multiline[0][:len(dfs_lineRads[iTyp].loc[iBin])-first_line_in_multipulse]*peakUnitArea / sensitivity[:len(dfs_lineRads[iTyp].loc[iBin])-first_line_in_multipulse]
						else:
							print('this part of code is not updated')
							exit()
							dfs_lineRads[iTyp].loc[iBin][first - 10:] = fit[0][first - 5:5]*peakUnitArea / sensitivity[first:10]
							df_lineRad.loc[iBin][first - 10:] = fit[0][first - 5:5]*peakUnitArea / sensitivity[first:10]
				except Exception as e:
					print('bin '+str(iBin)+' fit failed. Reason: %s' % str(e))
				if  (to_print and iBin in [5,20,30]):
					print('file 1   '+path_where_to_save_everything+'/spectral_fit_example'+'/fit_time_' + str(np.around(time, decimals=3)) +'_ms_bin_'+str(iBin)+ '.eps')
					plt.title('Emission profile fitting for time %.3g' %(time) + ' ms and bin '+str(iBin) + ' sigma range %.5g/%.5g mm' % (min_sigma,max_sigma) )
					plt.grid()
					plt.legend(loc='best',fontsize='xx-small')
					plt.xlabel('wavelength [nm]')
					plt.ylabel('counts [au]')

					yFit = binnedData[iBin]
					dxFit = np.concatenate([np.diff(lamb[iTyp]),[np.diff(lamb[iTyp])[-1]]])
					true_sensitivity_1 = sensitivity_density_interpolator(lamb[iTyp]-dxFit/2)
					true_sensitivity_2 = sensitivity_density_interpolator(lamb[iTyp]+dxFit/2)
					true_sensitivity = (true_sensitivity_1+true_sensitivity_2)/2*dxFit

					plt.ylim(max(0.1/1000,np.min((binnedData[iBin]/true_sensitivity)[:-100])),np.max(binnedData[iBin]/true_sensitivity))
					print('file 2   '+path_where_to_save_everything+'/spectral_fit_example'+'/fit_time_' + str(np.around(time, decimals=3)) +'_ms_bin_'+str(iBin)+ '.eps')
					plt.semilogy()
					save_done = 0
					save_index=1
					try:	# All this trie are because of a known issues when saving lots of figure inside of multiprocessor
						plt.savefig(path_where_to_save_everything+'/spectral_fit_example'+'/fit_time_' + str(np.around(time, decimals=3)) +'_ms_bin_'+str(iBin)+ '.eps',bbox_inches='tight')
					except Exception as e:
						print(path_where_to_save_everything+'/spectral_fit_example'+'/fit_time_' + str(np.around(time, decimals=3)) +'_ms_bin_'+str(iBin)+ '.eps save try number '+str(1)+' failed. Reason %s' % str(e))
						while save_done==0 and save_index<100:
							try:
								tm.sleep(np.random.random()**2)
								plt.savefig(path_where_to_save_everything+'/spectral_fit_example'+'/fit_time_' + str(np.around(time, decimals=3)) +'_ms_bin_'+str(iBin)+ '.eps',bbox_inches='tight')
								print(path_where_to_save_everything+'/spectral_fit_example'+'/fit_time_' + str(np.around(time, decimals=3)) +'_ms_bin_'+str(iBin)+ '.eps save try number '+str(save_index+1)+' successfull')
								save_done=1
							except Exception as e:
								print(path_where_to_save_everything+'/spectral_fit_example'+'/fit_time_' + str(np.around(time, decimals=3)) +'_ms_bin_'+str(iBin)+ '.eps save try number '+str(save_index+1)+' failed. Reason %s' % str(e))
								save_index+=1
					plt.close('all')
					print('file 3   '+path_where_to_save_everything+'/spectral_fit_example'+'/fit_time_' + str(np.around(time, decimals=3)) +'_ms_bin_'+str(iBin)+ '.eps')

		dfs_totIntens[iTyp].loc[iSet] = dfs_lineRads[iTyp].sum(skipna=0)
		df_lineRad.n3 = dfs_lineRads[0].n3
		# df_lineRad.to_csv('Radiance/%i.csv' % iSet)  # Write to file


		return dfs_lineRads[iTyp],dfs_lineRads_sigma[iTyp]
