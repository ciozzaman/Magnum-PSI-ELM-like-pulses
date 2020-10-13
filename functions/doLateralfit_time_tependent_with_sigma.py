# def doLateralfit_time_tependent_with_sigma(df_settings,all_fits,all_fits_sigma, merge_ID_target, new_timesteps,dx,xx,r,same_centre_every_line=False,same_centre_all_time=True,max_central_depression_amplitude=-1,min_central_depression_sigma=0.05,force_glogal_center=0):




try:
	print('number_of_radial_divisions '+str(number_of_radial_divisions))
except:
	number_of_radial_divisions=21
try:
	print('same_centre_every_line '+str(same_centre_every_line))
except:
	same_centre_every_line=False
try:
	print('same_centre_all_time '+str(same_centre_all_time))
except:
	same_centre_all_time=True
try:
	print('max_central_depression_amplitude '+str(max_central_depression_amplitude))
except:
	max_central_depression_amplitude=-1
try:
	print('min_central_depression_sigma '+str(min_central_depression_sigma))
except:
	min_central_depression_sigma=0.1


import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import copy as cp
from uncertainties import ufloat,unumpy,correlated_values
from uncertainties.unumpy import exp,nominal_values,std_devs,erf
path = '/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)
# pdir1 = './plots/LateralGauss/'
# pdir2 = './plots/RadialGauss/'
# ddir = './results/Radiance/'
# odir = './results/'

# for cdir in [pdir1, pdir2, odir]:
#	 if not os.path.isdir(cdir):
#		 os.makedirs(cdir)

nLine = np.shape(all_fits)[-1]
first_time = np.min(new_timesteps)
last_time = np.max(new_timesteps)
time_res = np.mean(np.diff(new_timesteps))

gauss = lambda x, A, sig, x0: A * np.exp(-((x - x0) / sig) ** 2)
gauss_with_error_included = lambda x, A, sig, x0: A * exp(-((x - x0) / sig) ** 2)
gauss_bias = lambda x, A, sig, x0,m,q: A * np.exp(-((x - x0) / sig) ** 2)+q+x*m
# gauss2 = lambda x,A1,sig1,A2,sig2,x0: gauss(x,A1,sig1,x0) + gauss(x,A2,sig2,x0)
gauss3 = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: gauss(x, A1, sig1, x0) + gauss(x, A2, sig2, x0) + gauss(x, A3,sig3,x0)
gauss3_bias = lambda x, A1, sig1, A2, sig2, A3, sig3, x0,m,q: gauss(x, A1, sig1, x0) + gauss(x, A2, sig2, x0) + gauss(x, A3,sig3,x0)+q+x*m
gaussJac = lambda x, A, sig, x0: np.array([gauss(x, 1, sig, x0), gauss(x, A, sig, x0) * 2 * (x - x0) ** 2 / sig ** 3,gauss(x, A, sig, x0) * 2 * (x - x0) / sig ** 2])
gauss3Jac = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: [np.concatenate([a[:2], a[3:5], a[6:8], [sum(a[2::3])]]) for a in [np.concatenate([gaussJac(x, A1, sig1, x0), gaussJac(x, A2, sig2, x0), gaussJac(x, A3, sig3, x0)])]][0]
AInvgauss = lambda x, A, sig, x0: gauss(x, A / sig / np.sqrt(np.pi), sig, 0)
AInvgauss_with_error_included = lambda x, A, sig, x0: gauss_with_error_included(x, A / sig / np.sqrt(np.pi), sig, 0)
AInvgauss_with_error_included_2 = lambda x, A, sig, x0: gauss_with_error_included(x, A * sig / np.sqrt(np.pi), sig, 0)
AInvgauss3 = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: AInvgauss(x, A1, sig1, 0) + AInvgauss(x, A2, sig2,0) + AInvgauss(x, A3,sig3,0)
AInvgaussJac = lambda x, A, sig, x0: np.array([gauss(x, 1 / np.sqrt(np.pi) / sig, sig, 0),gauss(x, A, sig, 0) * (2 * x ** 2 - sig ** 2) / sig ** 4 / np.sqrt(np.pi), gauss(x, A, sig, 0) * 2 * x / sig ** 3 / np.sqrt(np.pi)])
AInvgauss3Jac = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: [np.concatenate([a[:2], a[3:5], a[6:8], [sum(a[2::3])]]) for a in [np.concatenate([AInvgaussJac(x, A1, sig1, 0), AInvgaussJac(x, A2, sig2, 0), AInvgaussJac(x, A3, sig3, 0)])]][0]
LRvecmul = lambda vec, mat: np.dot(vec, np.matmul(mat, vec))
gauss3_2 = lambda x, A1, sig1, c_A2, c_sig2, c_A3, c_sig3,x0: gauss(x, A1, sig1, x0) + gauss(x,A1*c_A2,c_sig2*sig1,x0) + gauss(x, c_A3*c_sig3*(A1 +A1*c_A2/c_sig2), c_sig3*sig1, x0)
AInvgauss3_locked_2 = lambda x, A1, sig1, c_A2, c_sig2, c_A3, c_sig3: AInvgauss(x, A1, sig1, 0) + AInvgauss(x, A1*c_A2,c_sig2*sig1,0) + AInvgauss(x, c_A3*c_sig3*(A1 +A1*c_A2/c_sig2), c_sig3*sig1, 0)
AInvgauss3_locked_2_with_error_included = lambda x, A1, sig1, A2, c_sig2, c_A3, c_sig3: AInvgauss_with_error_included(x, A1, sig1, 0) + AInvgauss_with_error_included(x, A2,c_sig2*sig1,0) + AInvgauss_with_error_included(x, c_A3*c_sig3*(A1 +A2/c_sig2), c_sig3*sig1, 0)
AInvgauss3_locked_3_with_error_included = lambda x, A1, sig1, A2, c_sig2, c_A3, c_sig3: AInvgauss_with_error_included_2(x, A1, sig1, 0) + AInvgauss_with_error_included_2(x, A2,c_sig2*sig1,0) + AInvgauss_with_error_included_2(x, c_A3*c_sig3*(A1 +A2/c_sig2), c_sig3*sig1, 0)
gauss_sigma = lambda x, A, gaus_sig, x0,A_sig,gaus_sig_sig,x0_sig:gauss(x, A, gaus_sig, x0) * np.sqrt( (A_sig/A)**2 + (2*(x-x0)*x0_sig/(gaus_sig**2))**2 + (2*((x-x0)**2)*gaus_sig_sig/(gaus_sig**3))**2 )
AInvgauss_sigma_1 = lambda x, A1, sig1, A1_sigma, sig1_sigma: AInvgauss(x, A1, sig1, 0) * np.sqrt( (A1_sigma/A1)**2 + (1+4*(x**4)/(sig1**4))*(sig1_sigma/sig1)**2 )
AInvgauss_sigma_2 = lambda x, sig1, A2, c_sig2, sig1_sigma, A2_sigma, c_sig2_sigma: AInvgauss(x, A2, c_sig2*sig1, 0) * np.sqrt( (A2_sigma/A2)**2 + (1+4*(x**4)/((c_sig2*sig1)**4))*((sig1_sigma/sig1)**2 + (c_sig2_sigma/c_sig2)**2) )
AInvgauss_sigma_3 = lambda x, A1, sig1, A2, c_sig2, c_A3, c_sig3, A1_sigma, sig1_sigma, A2_sigma, c_sig2_sigma, c_A3_sigma, c_sig3_sigma: AInvgauss(x, c_A3*c_sig3*(A1 +A2/c_sig2), c_sig3*sig1, 0) * np.sqrt( (c_A3_sigma/c_A3)**2 + (A1_sigma**2 + (A2_sigma/c_sig2)**2 + (A2*c_sig2_sigma/c_sig2**2)**2)/((A1 +A2/c_sig2)**2) + 4*(x**4)/((c_sig3*sig1)**4)*(c_sig3_sigma/c_sig3)**2 + (1+4*(x**4)/((c_sig3*sig1)**4))*(sig1_sigma/sig1)**2 )
AInvgauss3_locked_2_sigma = lambda x, A1, sig1, A2, c_sig2, c_A3, c_sig3, A1_sigma, sig1_sigma, A2_sigma, c_sig2_sigma, c_A3_sigma, c_sig3_sigma: np.sqrt( (AInvgauss_sigma_1(x, A1, sig1, A1_sigma, sig1_sigma))**2 + (AInvgauss_sigma_2(x, sig1, A2, c_sig2, sig1_sigma, A2_sigma, c_sig2_sigma))**2 + (AInvgauss_sigma_3(x, A1, sig1, A2, c_sig2, c_A3, c_sig3, A1_sigma, sig1_sigma, A2_sigma, c_sig2_sigma, c_A3_sigma, c_sig3_sigma ))**2 )
erf_with_error_included = lambda x, A, sig, x0: -0.5 * (np.pi**0.5) * sig * A * erf(-(x - x0) / sig)
A_erf_with_error_included = lambda x, A, sig, x0: erf_with_error_included(x, A / sig / np.sqrt(np.pi), sig, 0)
A_erf_3_locked_2_with_error_included = lambda x, A1, sig1, c_A2, c_sig2, c_A3, c_sig3: A_erf_with_error_included(x, A1, sig1, 0) + A_erf_with_error_included(x, A1*c_A2,c_sig2*sig1,0) + A_erf_with_error_included(x, c_A3*c_sig3*(A1 +A1*c_A2/c_sig2), c_sig3*sig1, 0)
# bds2 = [[0,1e-4,0,1e-4,0],[np.inf,np.inf,np.inf,np.inf,np.inf]]
bds3 = [[0, 1e-4, 0, 1e-4, -np.inf, 1e-4, 0], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, np.inf]]

# dx = 18 / 40 * (50.5 / 27.4) / 1e3
# xx = np.arange(40) * dx  # m
xn = np.linspace(0, max(xx), 1000)
# r = np.linspace(-max(xx) / 2, max(xx) / 2, 1000)
# bds = [[-np.inf, dx, -np.inf, dx, -np.inf, dx, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, max(xx)]]
bds = [[0, dx, -np.inf, dx, -np.inf, dx, 0], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, max(xx)]]
bds4 = [[0, dx, 0], [np.inf, np.inf, max(xx)]]
fitCoefs = np.zeros((len(all_fits), nLine, 7))
fitCoefs_sigma = np.zeros((len(all_fits), nLine, 7,7))
# fitCoefs = np.zeros((len(all_fits), nLine, 8))
leg = ['n = %i' % i for i in range(3, 13)]


minimum_level = np.min(all_fits[all_fits>0])
if force_glogal_center!=0:
	profile_centres = np.zeros((len(all_fits),nLine))
	profile_centre = np.ones((len(all_fits),nLine))*force_glogal_center

elif (same_centre_every_line==True and same_centre_all_time==True):
	profile_centres = np.zeros((len(all_fits),nLine))
	profile_centres_score = np.zeros((len(all_fits),nLine))
	# fit = [[0, 4e-3, 20 * dx],np.array([[dx,dx],[dx,dx]])]
	# for iSet in range(len(all_fits)):
	# 	# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
	# 	for iR in range(np.shape(all_fits)[-1]):
	# 		# # for col in ['n5']:
	# 		yy = all_fits[iSet,:,iR]
	# 		for iCorr, value in enumerate(yy):
	# 			if np.isnan(value):
	# 				yy[iCorr]=0
	# 		# if not yy.isnull().any():
	# 		# print(str(iSet)+' , '+str(iR))
	# 		borders = np.linspace(0,len(yy),len(yy)+1)[np.diff(np.array([0,*yy,0])==0)].astype('int')
	# 		if len(borders)==0:
	# 			xx_good = xx
	# 			yy_good = yy
	# 		else:
	# 			xx_good = xx[borders[0]:borders[-1]]
	# 			yy_good = yy[borders[0]:borders[-1]]
	# 		yy_good[yy_good==0]=minimum_level
	# 		ym = np.nanmax(yy_good)
	# 		# p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 3, 1e-3, 20 * dx]
	# 		if fit[1][-1,-1]<dx:
	# 			p0 = [ym, fit[0][-2], fit[0][-1]]
	# 		else:
	# 			p0 = [ym, 4e-3, 20 * dx]
	# 		if iR>0:
	# 			bds4 = [[0, dx, profile_centres[iSet,iR-1]-3*dx], [max(yy_good)*10, max(xx_good), profile_centres[iSet,iR-1]+3*dx]]
	# 		else:
	# 			bds4 = [[0, dx, 10*dx], [max(yy_good)*10, max(xx_good), 30*dx]]
	# 			# bds4 = [[0, dx, 0], [np.inf, np.inf, max(xx)]]
	# 		# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
	# 		try:
	# 			# fit = curve_fit(gauss, xx, yy, p0, maxfev=100000, bounds=np.array(bds)[:,4:])
	# 			fit = curve_fit(gauss, xx_good, yy_good, p0, maxfev=100000, bounds=np.array(bds4))
	# 		except:
	# 			fit=[p0,np.array([[np.inf]])]
	# 		# if iSet==250:
	# 		# 	plt.figure()
	# 		# 	plt.plot(xx, yy)
	# 		# 	plt.plot(xx, gauss(xx, *fit[0]))
	# 		# 	plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
	# 		# 	plt.title('score: '+str(np.diag(fit[1])))
	# 		# 	plt.title('fit: ' + str(fit[0]))
	# 		# 	plt.pause(0.01)
	# 		profile_centres[iSet,iR]=fit[0][-1]
	# 		# profile_centres_score[iSet, iR] = fit[1][-1,-1]
	# 		profile_centres_score[iSet, iR] = np.sum(((yy_good - gauss(xx_good, *fit[0]))/yy_good) ** 2)


	class calc_stuff_output:
		def __init__(self, iSet, profile_centres,profile_centres_score):
			self.iSet = iSet
			self.profile_centres = profile_centres
			self.profile_centres_score = profile_centres_score

	# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
	def calc_fit(iSet,all_fits=all_fits, dx=dx, profile_centres=profile_centres,profile_centres_score=profile_centres_score,minimum_level=minimum_level):
		profile_centres = profile_centres[iSet]
		profile_centres_score = profile_centres_score[iSet]
		fit = [[0, 4e-3, 20 * dx], np.array([[dx, dx], [dx, dx]])]
		for iR in range(np.shape(all_fits)[-1]):

			# # for col in ['n5']:
			yy = all_fits[iSet,:,iR]
			for iCorr, value in enumerate(yy):
				if np.isnan(value):
					yy[iCorr]=0
			# if not yy.isnull().any():
			# print(str(iSet)+' , '+str(iR))
			borders = np.linspace(0,len(yy),len(yy)+1)[np.diff(np.array([0,*yy,0])==0)].astype('int')
			if len(borders)==0:
				xx_good = xx
				yy_good = yy
			else:
				xx_good = xx[borders[0]:borders[-1]]
				yy_good = yy[borders[0]:borders[-1]]
			yy_good[yy_good==0]=minimum_level
			ym = np.nanmax(yy_good)
			# p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 3, 1e-3, 20 * dx]
			if fit[1][-1,-1]<dx:
				p0 = [ym, fit[0][-2], fit[0][-1]]
			else:
				p0 = [ym, 4e-3, 20 * dx]
			if iR>0:
				bds4 = [[0, dx, profile_centres[iR-1]-3*dx], [max(yy_good)*10, max(xx_good), profile_centres[iR-1]+3*dx]]
			else:
				bds4 = [[0, dx, 10*dx], [max(yy_good)*10, max(xx_good), 30*dx]]
				# bds4 = [[0, dx, 0], [np.inf, np.inf, max(xx)]]
			# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
			try:
				# fit = curve_fit(gauss, xx, yy, p0, maxfev=100000, bounds=np.array(bds)[:,4:])
				fit = curve_fit(gauss, xx_good, yy_good, p0, maxfev=100000, bounds=np.array(bds4))
			except:
				fit=[p0,np.array([[np.inf]])]
			# if iSet==250:
			# 	plt.figure()
			# 	plt.plot(xx, yy)
			# 	plt.plot(xx, gauss(xx, *fit[0]))
			# 	plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
			# 	plt.title('score: '+str(np.diag(fit[1])))
			# 	plt.title('fit: ' + str(fit[0]))
			# 	plt.pause(0.01)
			profile_centres[iR]=fit[0][-1]
			# profile_centres_score[iSet, iR] = fit[1][-1,-1]
			profile_centres_score[iR] = np.sum(((yy_good - gauss(xx_good, *fit[0]))/yy_good) ** 2)

		output = calc_stuff_output(iSet, profile_centres,profile_centres_score)
		return output

	composed_row = map(calc_fit, range(len(all_fits)))
	composed_row = set(composed_row)
	composed_row = list(composed_row)

	temp=[]
	for i in range(len(composed_row)):
		temp.append(composed_row[i].iSet)
	iSet = np.array(temp)
	composed_row = np.array([peaks for _, peaks in sorted(zip(iSet, composed_row))])

	temp=[]
	for i in range(len(composed_row)):
		temp.append(composed_row[i].profile_centres)
	profile_centres = np.array(temp)
	temp=[]
	for i in range(len(composed_row)):
		temp.append(composed_row[i].profile_centres_score)
	profile_centres_score = np.array(temp)



	# profile_centre = np.mean(profile_centres,axis=-1)
	profile_centre = np.nansum(profile_centres/(profile_centres_score**2), axis=(0,1))/np.nansum(1/(profile_centres_score**2), axis=(0,1))
	for iSet in range(len(all_fits)):
		# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
		for iR in range(np.shape(all_fits)[-1]):
			# # for col in ['n5']:
			yy = all_fits[iSet, :, iR]
			for iCorr, value in enumerate(yy):
				if np.isnan(value):
					yy[iCorr] = 0
			borders = np.linspace(0,len(yy),len(yy)+1)[np.diff(np.array([0,*yy,0])==0)].astype('int')
			if len(borders)==0:
				xx_good = xx
				yy_good = yy
			else:
				xx_good = xx[borders[0]:borders[-1]]
				yy_good = yy[borders[0]:borders[-1]]
			# if not yy.isnull().any():
			ym = np.nanmax(yy_good)
			p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 4, 0.9e-3, profile_centre]
			bds = [[0, dx, -np.inf, dx, -np.inf, dx, profile_centre-2*dx], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, profile_centre+2*dx]]
			# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
			try:
				fit = curve_fit(gauss3, xx_good, yy_good, p0, maxfev=100000, bounds=bds)
			except:
				fit=[p0]
			profile_centres[iSet, iR] = fit[0][-1]
			if len(fit)==1:
				profile_centres_score[iSet, iR] = 1
			else:
				# profile_centres_score[iSet, iR] = fit[1][-1, -1]
				# profile_centres_score[iSet, iR] = fit[1][-1, -1] / ((np.nanmean(yy)/ym)**4)
				profile_centres_score[iSet, iR] = np.sum(((yy_good - gauss3(xx_good, *fit[0]))/yy_good) ** 2)
		# profile_centres_score[iSet, iR] = np.sum((yy - gauss3(xx, *fit[0])) ** 2)
		# plt.figure()
			# plt.plot(xx, yy)
			# plt.plot(xx, gauss3(xx, *fit[0]))
			# plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
			# plt.title('score: '+str(np.diag(fit[1])))
			# plt.pause(0.01)
	# profile_centre = np.mean(profile_centres, axis=-1)
	profile_centre = np.nansum(profile_centres/(profile_centres_score**2), axis=(0,1))/np.nansum(1/(profile_centres_score**2), axis=(0,1))
	for iSet in range(len(all_fits)):
		# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
		for iR in range(np.shape(all_fits)[-1]):
			# # for col in ['n5']:
			yy = all_fits[iSet, :, iR]
			for iCorr, value in enumerate(yy):
				if np.isnan(value):
					yy[iCorr] = 0
			borders = np.linspace(0, len(yy), len(yy) + 1)[np.diff(np.array([0, *yy, 0]) == 0)].astype('int')
			if len(borders)==0:
				xx_good = xx
				yy_good = yy
			else:
				xx_good = xx[borders[0]:borders[-1]]
				yy_good = yy[borders[0]:borders[-1]]
			# if not yy.isnull().any():
			ym = np.nanmax(yy_good)
			# p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 4, 0.9e-3, profile_centre]
			# bds = [[0, dx, -np.inf, dx, -np.inf, dx, profile_centre - 2 * dx],[np.inf, np.inf, np.inf, np.inf, 0, np.inf, profile_centre + 2 * dx]]
			p0 = [ym *0.9, 6*dx, ym * .05, 300, -0.25, 0.5,profile_centre]
			bds = [[0, 0, 0, 1, max_central_depression_amplitude, min_central_depression_sigma,profile_centre - 2 * dx],[max(yy)*100, np.inf, max(yy)*10, 10000, 0, 1,profile_centre + 2 * dx]]
			x_scale = [ym,dx,ym,1,1,1,profile_centre]
			# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
			try:
				# fit = curve_fit(gauss3, xx_good, yy_good, p0, maxfev=100000, bounds=bds)
				fit = curve_fit(gauss3_2, xx_good, yy_good, p0, maxfev=100000, bounds=bds,x_scale=x_scale)
			except:
				fit=[p0]
			profile_centres[iSet, iR] = fit[0][-1]
			if len(fit)==1:
				profile_centres_score[iSet, iR] = 1
			else:
				# profile_centres_score[iSet, iR] = fit[1][-1, -1]
				# profile_centres_score[iSet, iR] = fit[1][-1, -1] / ((np.nanmean(yy)/ym)**4)
				profile_centres_score[iSet, iR] = np.sum(((yy_good - gauss3(xx_good, *fit[0]))/yy_good) ** 2)
			# profile_centres_score[iSet, iR] = np.sum((yy - gauss3(xx, *fit[0])) ** 2)
			# if iSet in [35,70]:
			# 	plt.figure()
			# 	plt.plot(xx, yy)
			# 	plt.plot(xx, gauss3(xx, *fit[0]))
			# 	plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
			# 	plt.title('score: '+str(np.diag(fit[1])))
			# 	plt.pause(0.01)
	# profile_centre = np.mean(profile_centres, axis=-1)
	profile_centre = np.nansum(profile_centres/(profile_centres_score**2), axis=(0,1))/np.nansum(1/(profile_centres_score**2), axis=(0,1))
	print('profile_centre')
	print(profile_centre)
	profile_centre = np.ones_like(profile_centres)*profile_centre
	# profile_centre = np.convolve(profile_centre, np.ones((len(profile_centre)//5)) / (len(profile_centre)//5), mode='valid')

elif same_centre_all_time==True:
	profile_centres = np.zeros((len(all_fits),nLine))
	profile_centres_score = np.zeros((len(all_fits),nLine))
	# fit = [[0, 4e-3, 20 * dx,0,0],dx*np.ones((5,5))]
	fit = [[0, 5*dx, 20 * dx],dx*np.ones((3,3))]
	for iSet in range(len(all_fits)):
		# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
		for iR in range(np.shape(all_fits)[-1]):
			# # for col in ['n5']:
			yy = all_fits[iSet,:,iR]
			yy_sigma = all_fits_sigma[iSet,:,iR]
			for iCorr, value in enumerate(yy):
				if np.isnan(value):
					yy[iCorr]=0
					yy_sigma[iCorr]=np.nanmax(yy_sigma)
			# if not yy.isnull().any():
			# print(str(iSet)+' , '+str(iR))
			borders = np.linspace(0,len(yy),len(yy)+1)[np.diff(np.array([0,*yy,0])==0)].astype('int')
			if len(borders)==0:
				xx_good = xx
				yy_good = yy
				yy_good_sigma = yy_sigma
			else:
				xx_good = xx[borders[0]:borders[-1]]
				yy_good = yy[borders[0]:borders[-1]]
				yy_good_sigma = yy_sigma[borders[0]:borders[-1]]
			# yy_good[yy_good==0]=minimum_level
			ym = np.nanmax(yy_good)
			# p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 3, 1e-3, 20 * dx]
			if ((fit[1][2,2])**0.5<dx/2 and iR>0 and (np.abs(fit[0][1]-20*dx)<5*dx)):
				# p0 = [ym, fit[0][1], fit[0][2],0,0]
				p0 = [ym, fit[0][1], fit[0][2]]
			else:
				# p0 = [ym, 4e-3, 20 * dx,0,0]
				p0 = [ym, 5*dx, 20 * dx]
				fit[0][1]=4e-3
				fit[0][2]=20 * dx
			if iR>0:
				# bds4 = [[0, dx, fit[0][2]-3*dx,-np.inf,-np.inf], [max(yy_good)*10, max(xx_good), fit[0][2]+3*dx,np.inf,np.inf]]
				bds4 = [[0, dx, fit[0][2]-3*dx], [max(yy_good)*10, max(xx_good), fit[0][2]+3*dx]]
			else:
				# bds4 = [[0, dx, 10*dx,-np.inf,-np.inf], [max(yy_good)*10, max(xx_good), 30*dx,np.inf,np.inf]]
				bds4 = [[0, dx, 10*dx], [max(yy_good)*10, max(xx_good), 30*dx]]
				# bds4 = [[0, dx, 0], [max(yy_good)*10, max(xx_good), max(xx_good)]]
			# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
			try:
				# fit = curve_fit(gauss, xx, yy, p0, maxfev=100000, bounds=np.array(bds)[:,4:])
				# fit = curve_fit(gauss_bias, xx_good, yy_good, p0, maxfev=10000, bounds=np.array(bds4))
				fit = curve_fit(gauss, xx_good, yy_good, p0,sigma=yy_good_sigma, absolute_sigma=True, maxfev=10000, bounds=np.array(bds4))
				if fit[1][2,2]==0:
					fit[1][2, 2]=np.inf
			except:
				print('preliminary fit failed for line '+str(iR)+' iSet '+str(iSet))
				fit=[p0,np.inf*np.ones((len(p0),len(p0)))]
			# if iR==8:
			# plt.figure()
			# plt.plot(xx, yy)
			# plt.plot(xx, gauss_bias(xx, *fit[0]))
			# plt.plot([fit[0][2],fit[0][2]],[min(yy),max(yy)])
			# plt.title('fit: ' + str(fit[0])+'\n score: '+str(np.diag(fit[1])))
			# # plt.title('fit: ' + str(fit[0]))
			# plt.pause(0.01)
			# if ((fit[1][-1,-1])**0.5>dx or (iR>0 and abs(profile_centres[iSet,iR-1]-fit[0][-1])>dx*3)):
			# 	fit[0][-1] = 20 * dx
			# 	fit[1][-1,-1] = np.inf
			profile_centres[iSet,iR]=fit[0][2]
			profile_centres_score[iSet, iR] = fit[1][2,2]
			# profile_centres_score[iSet, iR] = np.sum(((yy_good - gauss(xx_good, *fit[0]))/np.max(yy_good)) ** 2)
	# profile_centre = np.mean(profile_centres,axis=-1)
	profile_centre = np.nansum(profile_centres/(profile_centres_score**2), axis=0)/np.nansum(1/(profile_centres_score**2), axis=0)
	profile_centre = np.ones_like(profile_centre)*np.median(profile_centre)
	for iSet in range(len(all_fits)):
		# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
		for iR in range(np.shape(all_fits)[-1]):
			# # for col in ['n5']:
			yy = all_fits[iSet, :, iR]
			yy_sigma = all_fits_sigma[iSet, :, iR]
			for iCorr, value in enumerate(yy):
				if np.isnan(value):
					yy[iCorr] = 0
					yy_sigma[iCorr] = np.nanmax(yy_sigma)
			borders = np.linspace(0,len(yy),len(yy)+1)[np.diff(np.array([0,*yy,0])==0)].astype('int')
			if len(borders)==0:
				xx_good = xx
				yy_good = yy
				yy_good_sigma = yy_sigma
			else:
				xx_good = xx[borders[0]:borders[-1]]
				yy_good = yy[borders[0]:borders[-1]]
				yy_good_sigma = yy_sigma[borders[0]:borders[-1]]
			# if not yy.isnull().any():
			ym = np.nanmax(yy_good)
			p0 = [ym / 6, 30*dx, ym * .8, 5*dx, -ym / 4, 1.1*dx, profile_centre[iR]]
			bds = [[0, dx, -np.inf, dx, -np.inf, dx, profile_centre[iR]-2*dx], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, profile_centre[iR]+2*dx]]
			# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
			try:
				fit = curve_fit(gauss3, xx_good, yy_good, p0,sigma=yy_good_sigma, absolute_sigma=True, maxfev=10000, bounds=bds)
			except:
				print('preliminary fit failed for line '+str(iR)+' iSet '+str(iSet))
				fit=[p0,np.inf*np.ones((len(p0),len(p0)))]
			profile_centres[iSet, iR] = fit[0][-1]
			if len(fit)==1:
				profile_centres_score[iSet, iR] = 1
			else:
				profile_centres_score[iSet, iR] = fit[1][-1, -1]
				# profile_centres_score[iSet, iR] = fit[1][-1, -1] / ((np.nanmean(yy)/ym)**4)
				# profile_centres_score[iSet, iR] = np.sum(((yy_good - gauss3(xx_good, *fit[0]))/yy_good) ** 2)
			# profile_centres_score[iSet, iR] = np.sum((yy - gauss3(xx, *fit[0])) ** 2)
			# plt.figure()
			# plt.plot(xx, yy)
			# plt.plot(xx, gauss3(xx, *fit[0]))
			# plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
			# plt.title('score: '+str(np.diag(fit[1])))
			# plt.pause(0.01)
	# profile_centre = np.mean(profile_centres, axis=-1)
	profile_centre = np.nansum(profile_centres/(profile_centres_score**2), axis=0)/np.nansum(1/(profile_centres_score**2), axis=0)
	for iSet in range(len(all_fits)):
		# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
		for iR in range(np.shape(all_fits)[-1]):
			# # for col in ['n5']:
			yy = all_fits[iSet, :, iR]
			yy_sigma = all_fits_sigma[iSet, :, iR]
			for iCorr, value in enumerate(yy):
				if np.isnan(value):
					yy[iCorr] = 0
					yy_sigma[iCorr] = np.nanmax(yy_sigma)
			borders = np.linspace(0, len(yy), len(yy) + 1)[np.diff(np.array([0, *yy, 0]) == 0)].astype('int')
			if len(borders)==0:
				xx_good = xx
				yy_good = yy
				yy_good_sigma = yy_sigma
			else:
				xx_good = xx[borders[0]:borders[-1]]
				yy_good = yy[borders[0]:borders[-1]]
				yy_good_sigma = yy_sigma[borders[0]:borders[-1]]
			# if not yy.isnull().any():
			ym = np.nanmax(yy_good)
			# p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 4, 0.9e-3, profile_centre]
			# bds = [[0, dx, -np.inf, dx, -np.inf, dx, profile_centre - 2 * dx],[np.inf, np.inf, np.inf, np.inf, 0, np.inf, profile_centre + 2 * dx]]
			p0 = [ym *0.9, 6*dx, ym * .05, 300, -0.25, 0.5,profile_centre[iR]]
			bds = [[0, 0, 0, 1, max_central_depression_amplitude, min_central_depression_sigma,profile_centre[iR] - 2 * dx],[max(yy)*100, np.inf, max(yy)*10, np.inf, 0, 1,profile_centre[iR] + 2 * dx]]
			# x_scale = [ym,dx,ym,1,1,1,profile_centre[iR]]
			x_scale = np.abs(p0)
			# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
			try:
				# fit = curve_fit(gauss3, xx_good, yy_good, p0, maxfev=100000, bounds=bds)
				fit = curve_fit(gauss3_2, xx_good, yy_good, p0,sigma=yy_good_sigma, absolute_sigma=True, maxfev=100000, bounds=bds,x_scale=x_scale)
			except:
				print('fit failed for line '+str(iR)+' iSet '+str(iSet))
				fit=[p0,np.inf*np.ones((len(p0),len(p0)))]
			profile_centres[iSet, iR] = fit[0][-1]
			if len(fit)==1:
				profile_centres_score[iSet, iR] = 1
			else:
				# profile_centres_score[iSet, iR] = fit[1][-1, -1]
				# profile_centres_score[iSet, iR] = fit[1][-1, -1] / ((np.nanmean(yy)/ym)**4)
				profile_centres_score[iSet, iR] = np.sum(((yy_good - gauss3_2(xx_good, *fit[0]))/yy_good) ** 2)
			# profile_centres_score[iSet, iR] = np.sum((yy - gauss3(xx, *fit[0])) ** 2)
			# if iSet in [35,70]:
			# plt.figure()
			# plt.plot(xx, yy)
			# plt.plot(xx, gauss3(xx, *fit[0]))
			# plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
			# plt.title('score: '+str(np.diag(fit[1])))
			# plt.pause(0.01)
	# profile_centre = np.mean(profile_centres, axis=-1)
	profile_centre = np.nansum(profile_centres/(profile_centres_score**2), axis=0)/np.nansum(1/(profile_centres_score**2), axis=0)
	print('profile_centre')
	print(profile_centre)
	profile_centre = np.ones_like(profile_centres)*profile_centre
	# profile_centre = np.convolve(profile_centre, np.ones((len(profile_centre)//5)) / (len(profile_centre)//5), mode='valid')






class calc_stuff_output:
	def __init__(self, iSet,iR, fit):
		self.iSet = iSet
		self.iR = iR
		self.fit = fit

def calc_stuff(arg):
	iSet = arg[0]
	iR = arg[1]
	example_fit = arg[2]
	example_fit_sigma = arg[3]
	gauss3_locked_2 = lambda x, A1, sig1, c_A2, c_sig2, c_A3, c_sig3: gauss(x, A1, sig1, profile_centre[iSet,iR]) + gauss(x,A1*c_A2,c_sig2*sig1,profile_centre[iSet,iR]) + gauss(x, c_A3*c_sig3*(A1 +A1*c_A2/c_sig2), c_sig3*sig1, profile_centre[iSet,iR])
	gauss3_locked_2_bias = lambda x, A1, sig1, c_A2, c_sig2, c_A3, c_sig3, m, q: gauss(x, A1, sig1,profile_centre[iR]) + gauss(x, A1*c_A2,c_sig2 * sig1,profile_centre[iR]) + gauss(x, c_A3 * c_sig3 * (A1 + A1*c_A2 / c_sig2), c_sig3 * sig1, profile_centre[iR]) + q + x * m
	yy = all_fits[iSet, :, iR]
	yy_sigma = all_fits_sigma[iSet, :, iR]
	xx_int = xx
	if np.sum(np.isnan(yy))+np.sum(np.isnan(yy_sigma))>0:
		select = np.logical_not(np.logical_or(np.isnan(yy),np.isnan(yy_sigma)))
		yy = yy[select]
		yy_sigma = yy_sigma[select]
		xx_int = xx_int[select]
	# yy[np.isnan(yy)]=0
	# yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
	# for iCorr, value in enumerate(yy):
	# 	if np.isnan(value):
	# 		yy[iCorr] = 0
	temp_all_fits = cp.deepcopy(yy)
	if np.max(yy)>100*np.sort(yy)[-2]:
		pos1 = np.max([yy.argmax()-1,0])
		pos2 = np.min([yy.argmax()+1,len(yy)-1])
		yy[yy.argmax()]=np.mean(yy[[pos1,pos2]])
		yy_sigma[yy.argmax()]=np.sqrt(np.sum(np.array(yy_sigma[[pos1,pos2]])**2))/2

	borders = np.linspace(0, len(yy), len(yy) + 1)[np.diff(np.array([0, *yy, 0]) == 0)].astype('int')
	if len(borders)==0:
		xx_good = xx_int
		yy_good = yy
		yy_good_sigma = yy_sigma
	else:
		if borders[0]!=0:	# this should not be necessary but I cannot avoid a smear of composed_array when dealing with data missing in lower lines
			borders[0]+=5
		xx_good = xx_int[borders[0]:borders[-1]]
		yy_good = yy[borders[0]:borders[-1]]
		yy_good_sigma = yy_sigma[borders[0]:borders[-1]]

	# xx_good = np.abs(xx_good-profile_centre[iSet,iR])
	# plt.figure();plt.plot(xx_good,yy_good)
	# yy_good = np.array([peaks for _, peaks in sorted(zip(xx_good, yy_good))])
	# yy_good = np.array([yy_good[0],*np.convolve(yy_good,np.ones((3))/3,'valid'),yy_good[-1]])
	# xx_good = np.sort(xx_good)
	# plt.plot(xx_good, yy_good)
	# plt.pause(0.01)

	# temp = unumpy.uarray([np.max(yy_good),np.max(yy_good)], [10*np.max(yy_good),10*np.max(yy_good)])
	# max_central_depression_amplitude_max = max_central_depression_amplitude
	# fit = [-np.ones((6))]

	# while ((np.max(std_devs(temp))>=4*np.max(nominal_values(temp)) and -fit[0][4]>0.85) and max_central_depression_amplitude_max>0.7):
	ym = np.nanmax(yy_good)
	# inner_dip_sig = 0.9e-3
	inner_dip_sig = 0.15
	# bds = [[0, 0, 0, 0, -max(yy)*10, 0],[max(yy)*100, np.inf, max(yy)*100, np.inf, 0, np.inf]]
	bds = [[0, 0, 0, 1, max_central_depression_amplitude, min_central_depression_sigma,profile_centre[iSet,iR]-0.2*dx],[max(yy)*100, np.inf, 10, np.inf, 0, 1,profile_centre[iSet,iR]+0.2*dx]]
	# p0 = [ym / 5, 20e-3, ym * .8, 4e-1, -ym / 2, 0.9e-3]
	# p0 = [ym *0.9, 6e-3, ym * .05, 4e-1, -ym / 4, inner_dip_sig]
	if iR>0:
		bds = [[0, 0, 0, 1, max_central_depression_amplitude, min_central_depression_sigma,profile_centre[iSet,iR]-0.2*dx],[max(yy)*100, np.inf, 10, max(20,1.5*example_fit[min(2,iR-1)][3]), 0, 1,profile_centre[iSet,iR]+0.2*dx]]
		p0 = [example_fit[min(2,iR-1)][0]*ym/np.nanmax(all_fits[iSet, :, min(2,iR-1)]), *example_fit[min(2,iR-1)][1:]]
		x_scale = [p0[0]*0.2,dx,0.1,1,1,1,profile_centre[iSet,iR]]
	else:
		bds = [[0, 0, 0, 1, max_central_depression_amplitude, min_central_depression_sigma,profile_centre[iSet,iR]-0.2*dx],[max(yy)*100, np.inf, 10, np.inf, 0, 1,profile_centre[iSet,iR]+0.2*dx]]
		p0 = [ym *0.9, 6*dx, .05, 10, -0.25, inner_dip_sig,profile_centre[iSet,iR]]
		x_scale = [p0[0]*0.2,dx,0.1,1,1,1,profile_centre[iSet,iR]]
	# p0 = [ym / 5, 20e-3, ym * .8, 4e-3, -ym / 5, 1e-3,profile_centre[iSet]]
	# bds = [[0, 0, 0, 1, -1, 1e-10, -np.inf, -1e-10],[max(yy) * 100, np.inf, max(yy) * 100, np.inf, 0, 1, np.inf, 1e-10]]
	# p0 = [ym * 0.9, 5e-3, ym * .05, 67, -0.25, inner_dip_sig, 0, 0]
	try:
		try:
			fit = curve_fit(gauss3_2, xx_good, yy_good, p0,sigma=yy_good_sigma, absolute_sigma=True, maxfev=1000000, bounds=np.array(bds),x_scale=x_scale,ftol=1e-15,loss='soft_l1',xtol=1e-11,gtol=1e-11)
			if iR==0 and fit[0][0]/(np.diag(fit[1])[0]**0.5)<1e-8:
				p0 = [ym *0.2, 6*dx, .8, 10, -0.25, inner_dip_sig,profile_centre[iSet,iR]]
				x_scale = [p0[0]*0.2,dx,0.1,1,1,1,profile_centre[iSet,iR]]
				fit = curve_fit(gauss3_2, xx_good, yy_good, p0,sigma=yy_good_sigma, absolute_sigma=True, maxfev=1000000, bounds=np.array(bds),x_scale=x_scale,ftol=1e-15,loss='soft_l1',xtol=1e-11,gtol=1e-11)
		except:
			try:
				fit = curve_fit(gauss3_2, xx_good, yy_good, p0,sigma=yy_good_sigma, absolute_sigma=True, maxfev=1000000, bounds=np.array(bds),x_scale=x_scale,ftol=1e-11,loss='soft_l1')
			except:
				fit = curve_fit(gauss3_2, xx_good, yy_good, p0,sigma=yy_good_sigma, absolute_sigma=True, maxfev=1000000, bounds=np.array(bds),x_scale=x_scale,loss='soft_l1')
		# fit = curve_fit(gauss3_locked_2_bias, xx_good, yy_good, p0, maxfev=100000, bounds=np.array(bds))
		# while (AInvgauss3_locked_2(0, *fit[0]) < 0 and inner_dip_sig < 2e-3):
		while (AInvgauss3_locked_2(0, *fit[0][:-1]) < 0 and inner_dip_sig < 0.3):
		# while (AInvgauss3_locked_2(0, *fit[0][:-2]) < 0 and inner_dip_sig < 0.3):
			# inner_dip_sig+=(0.9e-3)/10
			print('small peak sigma increased, time slice '+str(iSet)+', line '+str(iR))
			inner_dip_sig+=0.015
			# p0 = [ym * 0.9, 6e-3, ym * .05, 4e-1, -ym / 3, inner_dip_sig]
			p0 = [*fit[0][:-2],inner_dip_sig,profile_centre[iSet,iR]]
			# p0 = [*fit[0][:5], inner_dip_sig, 0, 0]
			try:
				fit = curve_fit(gauss3_2, xx_good, yy_good, p0,sigma=yy_good_sigma, absolute_sigma=True, maxfev=1000000, bounds=np.array(bds),x_scale=x_scale,ftol=1e-15,loss='soft_l1')
			except:
				fit = curve_fit(gauss3_2, xx_good, yy_good, p0,sigma=yy_good_sigma, absolute_sigma=True, maxfev=1000000, bounds=np.array(bds),x_scale=x_scale,loss='soft_l1')
			# fit = curve_fit(gauss3_locked_2_bias, xx_good, yy_good, p0, maxfev=100000, bounds=np.array(bds))
		# if fit[0][-2]>0:
		# 	inner_dip_sig = 0.9e-3
		# 	p0 = [ym *0.9, 6e-3, ym * .05, 4e-1, -ym / 3, inner_dip_sig]
	except Exception as e:
		print('fitting failed for time slice '+str(iSet)+', line '+str(iR) + ' because '+str(e))
		fit=[[0,5e-3,0, 67, -0.25,0.15,profile_centre[iSet,iR]],np.ones((7,7))*np.nan]

	output = calc_stuff_output(iSet,iR, fit)
	return output

for iR in range(np.shape(all_fits)[-1]):
	all_indexes = []
	for iSet in range(len(all_fits)):
		all_indexes.append([iSet,iR,fitCoefs[iSet],fitCoefs_sigma[iSet]])

	pool = Pool(number_cpu_available,maxtasksperchild=1)
	all_results = [*pool.map(calc_stuff, all_indexes)]
	pool.close()
	pool.join()
	pool.terminate()
	del pool

	for i in range(len(all_results)):
		fitCoefs[int(all_results[i].iSet), int(all_results[i].iR)] = all_results[i].fit[0]
		fitCoefs_sigma[int(all_results[i].iSet), int(all_results[i].iR)] = all_results[i].fit[1]
	print('line '+str(iR)+' done')
fitCoefs_sigma[np.isnan(fitCoefs_sigma)]=np.nanmax(fitCoefs_sigma)

all_fits_fitted = np.zeros_like(all_fits)
all_fits_residuals = np.zeros_like(all_fits)
inverted_profiles = np.zeros((len(all_fits),nLine,len(r)))
inverted_profiles_sigma = np.zeros((len(all_fits),nLine,len(r)))
for iSet in range(len(all_fits)):
	# gauss3_locked = lambda x, A1, sig1, A2, sig2, A3, sig3: gauss(x, A1, sig1, profile_centre[iSet]) + gauss(x, A2, sig2, profile_centre[iSet]) + gauss(x,A3,sig3,profile_centre[iSet])
	# AInvgauss3_locked = lambda x, A1, sig1, A2, sig2, A3, sig3: AInvgauss(x, A1, sig1, 0) + AInvgauss(x, A2, sig2,0) + AInvgauss(x,A3,sig3,0)

	# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
	if  (np.min(np.abs(new_timesteps[iSet] - np.array([-0.2,0.2,0.55,1])))<=0.06):
		print('will plot the spectral fit at time '+str(new_timesteps[iSet]))
		if not os.path.exists(path+'/abel_inversion_example'):
			os.makedirs(path+'/abel_inversion_example')
		fig, ax = plt.subplots( np.shape(all_fits)[-1],2,figsize=(15, 25), squeeze=True)
		fig.suptitle('Abel inversion for time %.3g ,' %new_timesteps[iSet] +'ms\n   same_centre_every_line=%1s,same_centre_all_time=%1s,max_central_depression_amplitude=%3g,min_central_depression_sigma=%3g,force_glogal_center=%3g  ' %(same_centre_every_line,same_centre_all_time,max_central_depression_amplitude,min_central_depression_sigma,force_glogal_center) )


	for iR in range(np.shape(all_fits)[-1]):
		# gauss3_locked = lambda x, A1, sig1, A2, sig2, A3, sig3: gauss(x, A1, sig1, profile_centre[iSet,iR]) + gauss(x,A2,sig2,profile_centre[iSet,iR]) + gauss(x, A3, sig3, profile_centre[iSet,iR])
		# AInvgauss3_locked = lambda x, A1, sig1, A2, sig2, A3, sig3: AInvgauss(x, A1, sig1, 0) + AInvgauss(x, A2,sig2,0) + AInvgauss(x, A3, sig3, 0)
		gauss3_locked_2 = lambda x, A1, sig1, c_A2, c_sig2, c_A3, c_sig3: gauss(x, A1, sig1, profile_centre[iSet,iR]) + gauss(x,A1*c_A2,c_sig2*sig1,profile_centre[iSet,iR]) + gauss(x, c_A3*c_sig3*(A1 +A1*c_A2/c_sig2), c_sig3*sig1, profile_centre[iSet,iR])
		gauss3_locked_2_bias = lambda x, A1, sig1, c_A2, c_sig2, c_A3, c_sig3, m, q: gauss(x, A1, sig1,profile_centre[iR]) + gauss(x, A1*c_A2,c_sig2 * sig1,profile_centre[iR]) + gauss(x, c_A3 * c_sig3 * (A1 + A1*c_A2 / c_sig2), c_sig3 * sig1, profile_centre[iR]) + q + x * m
		yy = all_fits[iSet, :, iR]
		temp_all_fits = cp.deepcopy(yy)
		temp_all_fits[np.isnan(temp_all_fits)]=0
		yy_sigma = all_fits_sigma[iSet, :, iR]
		xx_int = xx
		if np.sum(np.isnan(yy))+np.sum(np.isnan(yy_sigma))>0:
			select = np.logical_not(np.logical_or(np.isnan(yy),np.isnan(yy_sigma)))
			yy = yy[select]
			yy_sigma = yy_sigma[select]
			xx_int = xx_int[select]
		# for iCorr, value in enumerate(yy):
		# 	if np.isnan(value):
		# 		yy[iCorr] = 0
		if np.max(yy)>100*np.sort(yy)[-2]:
			pos1 = np.max([yy.argmax()-1,0])
			pos2 = np.min([yy.argmax()+1,len(yy)-1])
			yy[yy.argmax()]=np.mean(yy[[pos1,pos2]])
			yy_sigma[yy.argmax()]=np.sqrt(np.sum(np.array(yy_sigma[[pos1,pos2]])**2))/2

		borders = np.linspace(0, len(yy), len(yy) + 1)[np.diff(np.array([0, *yy, 0]) == 0)].astype('int')
		if len(borders)==0:
			xx_good = xx_int
			yy_good = yy
			yy_good_sigma = yy_sigma
		else:
			if borders[0]!=0:	# this should not be necessary but I cannot avoid a smear of composed_array when dealing with data missing in lower lines
				borders[0]+=5
			xx_good = xx_int[borders[0]:borders[-1]]
			yy_good = yy[borders[0]:borders[-1]]
			yy_good_sigma = yy_sigma[borders[0]:borders[-1]]

		# xx_good = np.abs(xx_good-profile_centre[iSet,iR])
		# plt.figure();plt.plot(xx_good,yy_good)
		# yy_good = np.array([peaks for _, peaks in sorted(zip(xx_good, yy_good))])
		# yy_good = np.array([yy_good[0],*np.convolve(yy_good,np.ones((3))/3,'valid'),yy_good[-1]])
		# xx_good = np.sort(xx_good)
		# plt.plot(xx_good, yy_good)
		# plt.pause(0.01)

		fit = [fitCoefs[iSet, iR],fitCoefs_sigma[iSet, iR]]

		if  (np.min(np.abs(new_timesteps[iSet] - np.array([-0.2,0.2,0.55,1])))<=0.06):
			im = ax[iR,0].errorbar(1000*xx_good, yy_good,yerr=yy_good_sigma,color='c',label='line integrated data')
			im = ax[iR,0].plot([1000*fit[0][-1],1000*fit[0][-1]],[np.min([yy_good,gauss3_2(xx_good, *fit[0])]),np.max([yy_good,gauss3_2(xx_good, *fit[0])])],'k--')
			im = ax[iR,0].plot(1000*xx, gauss3_2(xx, *fit[0]),'m',label='fit')
			im = ax[iR,0].plot(1000*xx, gauss(xx, fit[0][0]*fit[0][2],fit[0][1]*fit[0][3],fit[0][-1]),'y--',label='base')
			im = ax[iR,0].plot(1000*xx, gauss(xx, fit[0][0],fit[0][1],fit[0][-1])+gauss(xx, fit[0][4]*fit[0][5]*fit[0][0]*(1+fit[0][2]/fit[0][3]),fit[0][5]*fit[0][1],fit[0][-1]),'g--',label='peak')
			ax[iR,0].grid()
			if iR==0:
				ax[iR,0].set_title('cyan=line int data, magenta=fit\nline n='+str(iR+4)+'->2\n ')
			else:
				ax[iR,0].set_title('line n='+str(iR+4)+'->2\n ')
			if iR==(np.shape(all_fits)[-1]-1):
				ax[iR,0].set_xlabel('x [mm]')
				ax[iR,0].set_ylabel('brightness [W m-2 sr-1]')
			else:
				ax[iR,0].set_xticks([])

		# if iR==0:
		# 	plt.figure()
		# 	# # plt.plot(1000*xx, yy)
		# 	plt.title('iSet='+str(iSet)+' line='+str(iR)+'\n'+str(fit[0])+'\n R2 '+str(np.sum((temp_all_fits - gauss3_2(xx,*fit[0]))**2)))
		# 	plt.errorbar(1000*xx_good, yy_good,yerr=yy_good_sigma)
		# 	# # plt.plot(1000*xx, gauss3_locked(xx, *[*fit[0][:2],0,0,0,0]))
		# 	# # plt.plot(1000*xx, gauss3_locked(xx, *[*fit[0][:4],0,0]))
		# 	# plt.plot([1000*profile_centre[iSet,iR],1000*profile_centre[iSet,iR]],[np.min(yy_good),np.max(yy_good)],'--k')
		# 	plt.plot(1000*xx, gauss3_2(xx, *fit[0]))
		# 	# plt.ylabel('emission line intensity [au]')
		# 	# plt.xlabel('impact radius [mm]')
		# 	# # plt.plot(xx, gauss3(xx, *fit[0]))
		# 	plt.pause(0.01)
		# inverted_profiles[iSet, iR] = AInvgauss3_locked(r, *fit[0])
		# all_fits_fitted[iSet, :, iR] = gauss3_locked(xx,*fit[0])
		# inverted_profiles[iSet, iR] = AInvgauss3_locked_2(r, *fit[0])
		# inverted_profiles_sigma[iSet, iR] = AInvgauss3_locked_2_sigma(r, *fit[0],*np.sqrt(np.diag(fit[1])))
		profile_centre[iSet,iR]=fit[0][-1]
		# temp = AInvgauss3_locked_2_with_error_included(r, unumpy.uarray(fit[0][0]*np.ones_like(r),(fit[1][0,0]**0.5)*np.ones_like(r)), unumpy.uarray(fit[0][1]*np.ones_like(r),(fit[1][1,1]**0.5)*np.ones_like(r)), unumpy.uarray(fit[0][2]*np.ones_like(r),(fit[1][2,2]**0.5)*np.ones_like(r)), unumpy.uarray(fit[0][3]*np.ones_like(r),(fit[1][3,3]**0.5)*np.ones_like(r)), unumpy.uarray(fit[0][4]*np.ones_like(r),(fit[1][4,4]**0.5)*np.ones_like(r)), unumpy.uarray(fit[0][5]*np.ones_like(r),(fit[1][5,5]**0.5)*np.ones_like(r)) )
		# temp = AInvgauss3_locked_2_with_error_included(r, *correlated_values(fit[0][:-1],fit[1][:-1,:-1]) )
		# temp = A_erf_3_locked_2_with_error_included(np.array([0,*(r+dx/2)]), *correlated_values(fit[0][:-1],fit[1][:-1,:-1]) )
		# temp = np.diff(temp)/np.array([dx/2,*np.diff(r)])
		temp = AInvgauss3_locked_3_with_error_included(np.array([0,*(r+dx/2)]), *correlated_values(fit[0][:-1],fit[1][:-1,:-1]) )
		temp = -np.diff(temp)/np.diff(np.array([0,*(r+dx/2)])**2)
		inverted_profiles[iSet, iR] = nominal_values(temp)
		inverted_profiles_sigma[iSet, iR] = std_devs(temp)
		all_fits_fitted[iSet, :, iR] = gauss3_2(xx,*fit[0])
		# inverted_profiles[iSet, iR] = AInvgauss3_locked_2(r, *fit[0][:-2])
		# all_fits_fitted[iSet, :, iR] = gauss3_locked_2_bias(xx,*fit[0])
		all_fits_residuals[iSet, :, iR] = (temp_all_fits - all_fits_fitted[iSet, :, iR])#/np.abs(np.nanmax(all_fits_fitted[iSet, :, iR]))
		if  (np.min(np.abs(new_timesteps[iSet] - np.array([-0.2,0.2,0.55,1])))<=0.06):
			im = ax[iR,1].errorbar(1000*r, nominal_values(temp),yerr=std_devs(temp),color='b',label='inverted data')
			im = ax[iR,1].plot(1000*r, std_devs(temp),'r',label='error')
			ax[iR,1].grid()
			if iR==0:
				ax[iR,1].set_title('blue=inverted prof, red=error\nfit='+str(''.join(['%.3g+/-%.3g, ' %(_,__) for _,__ in zip(fit[0],np.sqrt(np.diag(fit[1])))])[:-2]))
			else:
				ax[iR,1].set_title('fit='+str(''.join(['%.3g+/-%.3g, ' %(_,__) for _,__ in zip(fit[0],np.sqrt(np.diag(fit[1])))])[:-2]))
			if iR==(np.shape(all_fits)[-1]-1):
				ax[iR,1].set_xlabel('r [mm]')
				ax[iR,1].set_ylabel('emissivity [W m^-3 sr^-1]')
				ax[iR,1].yaxis.set_label_position('right')
				ax[iR,1].yaxis.tick_right()
			else:
				ax[iR,1].set_xticks([])
	if  (np.min(np.abs(new_timesteps[iSet] - np.array([-0.2,0.2,0.55,1])))<=0.06):
		plt.savefig(path+'/abel_inversion_example/abe_inversion_time_' + str(np.around(new_timesteps[iSet], decimals=3)) +'_ms.eps', bbox_inches='tight')
		plt.close('all')


		# if iR==0:
		# 	plt.figure()
		# 	# plt.errorbar(1000*r, AInvgauss3_locked_2(r, *fit[0]),yerr=AInvgauss3_locked_2_sigma(r, *fit[0],*np.sqrt(np.diag(fit[1]))))
		# 	plt.errorbar(1000*r, nominal_values(temp),yerr=std_devs(temp))
		# 	plt.plot(1000*r, nominal_values(temp))
		# 	plt.plot(1000*r, std_devs(temp))
		# 	# plt.ylabel('emission line intensity [au]')
		# 	# plt.xlabel('radius [mm]')
		# 	plt.title('iSet='+str(iSet)+' line='+str(iR)+'\n'+str(fit[0])+'\n R2 '+str(np.sum((temp_all_fits - gauss3_2(xx,*fit[0]))**2)))
		# 	plt.pause(0.01)
		# max_central_depression_amplitude_max-=0.05


if not os.path.exists(path):
	os.makedirs(path)
for iR in range(nLine):
	fig,ax = plt.subplots(figsize=(5, 10))
	# IM = ax.imshow(inverted_profiles[:,iR],'rainbow',vmax=np.nanmax(np.mean(np.sort(inverted_profiles[:,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(r-dx/2),1000*max(r+dx/2),last_time+conventional_time_step/2,first_time-conventional_time_step/2],aspect=50)
	IM = ax.imshow(inverted_profiles[:,iR],'rainbow',extent=[1000*min(r-dx/2),1000*max(r+dx/2),last_time+conventional_time_step/2,first_time-conventional_time_step/2],aspect=50)
	cbar = fig.colorbar(IM)
	cbar.set_label('Emissivity [W m^-3 sr^-1]')
	CS = ax.contour(r*1000,new_timesteps,inverted_profiles[:,iR],colors='k',linewidths=1)
	cbar.add_lines(CS)
	plt.title('Line n='+str(iR+4) + r'$\to$' + '2')
	plt.xlabel('radius [mm]')
	plt.ylabel('time [ms]')
	plt.savefig('%s/abel_inverted_profile%s.eps' % (path, iR+4), bbox_inches='tight')
	plt.close()
initial_time = int((np.abs(new_timesteps-0)).argmin())
final_time = int((np.abs(new_timesteps -2)).argmin())
for iR in range(nLine):
	fig,ax = plt.subplots(figsize=(5, 10))
	# vmax = max(inverted_profiles_sigma[:,iR].max(),inverted_profiles[:,iR].max())
	vmax = np.nanmax(np.mean(np.sort(inverted_profiles_sigma[initial_time:final_time+1,iR],axis=-1)[:,-10:-1],axis=-1))
	# IM = ax.imshow(inverted_profiles_sigma[:,iR],'rainbow',vmax=np.nanmax(np.mean(np.sort(inverted_profiles_sigma[:,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(r-dx/2),1000*max(r+dx/2),last_time+conventional_time_step/2,first_time-conventional_time_step/2],aspect=50)
	IM = ax.imshow(inverted_profiles_sigma[:,iR],'rainbow',vmax=vmax,extent=[1000*min(r-dx/2),1000*max(r+dx/2),last_time+conventional_time_step/2,first_time-conventional_time_step/2],aspect=50)
	cbar = fig.colorbar(IM)
	cbar.set_label('Emissivity [W m^-3 sr^-1]')
	CS = ax.contour(r*1000,new_timesteps,inverted_profiles_sigma[:,iR],colors='k',linewidths=1,vmax=vmax)
	cbar.add_lines(CS)
	plt.title('Line n='+str(iR+4) + r'$\to$' + '2' + '\nuncertainty')
	plt.xlabel('radius [mm]')
	plt.ylabel('time [ms]')
	plt.savefig('%s/abel_inverted_profile_sigma%s.eps' % (path, iR+4), bbox_inches='tight')
	plt.close()
initial_time = int((np.abs(new_timesteps-0)).argmin())
final_time = int((np.abs(new_timesteps -2)).argmin())
for iR in range(nLine):
	fig,ax = plt.subplots(figsize=(5, 10))
	# IM = ax.imshow(inverted_profiles[initial_time:final_time+1,iR],'rainbow',vmin=0,vmax=np.nanmax(np.mean(np.sort(inverted_profiles[initial_time:final_time+1,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(r-dx/2),1000*max(r+dx/2),new_timesteps[final_time]+conventional_time_step/2,new_timesteps[initial_time]-conventional_time_step/2],aspect=50)
	IM = ax.imshow(inverted_profiles[initial_time:final_time+1,iR],'rainbow',extent=[1000*min(r-dx/2),1000*max(r+dx/2),new_timesteps[final_time]+conventional_time_step/2,new_timesteps[initial_time]-conventional_time_step/2],aspect=50)
	cbar = fig.colorbar(IM)
	cbar.set_label('Emissivity [W m^-3 sr^-1]')
	CS = ax.contour(r*1000,new_timesteps[initial_time:final_time+1],inverted_profiles[initial_time:final_time+1,iR],colors='k',linewidths=1)
	cbar.add_lines(CS)
	plt.title('Line n='+str(iR+4) + r'$\to$' + '2')
	plt.xlabel('radius [mm]')
	plt.ylabel('time [ms]')
	plt.savefig('%s/abel_inverted_profile%s_short.eps' % (path, iR+4), bbox_inches='tight')
	plt.close()
for iR in range(nLine):
	fig,ax = plt.subplots(figsize=(5, 10))
	IM = ax.imshow(inverted_profiles[initial_time:final_time+1,iR]/inverted_profiles_sigma[initial_time:final_time+1,iR],'rainbow',vmin=0,vmax=np.nanmax(np.mean(np.sort(inverted_profiles[initial_time:final_time+1,iR]/inverted_profiles_sigma[initial_time:final_time+1,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(r-dx/2),1000*max(r+dx/2),new_timesteps[final_time]+conventional_time_step/2,new_timesteps[initial_time]-conventional_time_step/2],aspect=50)
	# IM = ax.imshow(inverted_profiles[initial_time:final_time+1,iR]/inverted_profiles_sigma[initial_time:final_time+1,iR],'rainbow',extent=[1000*min(r-dx/2),1000*max(r+dx/2),new_timesteps[final_time]+conventional_time_step/2,new_timesteps[initial_time]-conventional_time_step/2],aspect=50)
	cbar = fig.colorbar(IM)
	cbar.set_label('Signal/Noise [au]')
	CS = ax.contour(r*1000,new_timesteps[initial_time:final_time+1],inverted_profiles[initial_time:final_time+1,iR]/inverted_profiles_sigma[initial_time:final_time+1,iR],colors='k',linewidths=1)
	cbar.add_lines(CS)
	plt.title('SNR Line n='+str(iR+4) + r'$\to$' + '2')
	plt.xlabel('radius [mm]')
	plt.ylabel('time [ms]')
	plt.savefig('%s/abel_inverted_profile_SNR%s_short.eps' % (path, iR+4), bbox_inches='tight')
	plt.close()
for iR in range(nLine):
	fig,ax = plt.subplots(figsize=(5, 10))
	# vmax = max(inverted_profiles_sigma[:,iR].max(),inverted_profiles[:,iR].max())
	vmax=np.nanmax(np.mean(np.sort(inverted_profiles_sigma[initial_time:final_time+1,iR],axis=-1)[:,-10:-1],axis=-1))
	# IM = ax.imshow(inverted_profiles_sigma[initial_time:final_time+1,iR],'rainbow',vmin=0,vmax=np.nanmax(np.mean(np.sort(inverted_profiles_sigma[initial_time:final_time+1,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(r-dx/2),1000*max(r+dx/2),new_timesteps[final_time]+conventional_time_step/2,new_timesteps[initial_time]-conventional_time_step/2],aspect=50)
	IM = ax.imshow(inverted_profiles_sigma[initial_time:final_time+1,iR],'rainbow',vmax=vmax,extent=[1000*min(r-dx/2),1000*max(r+dx/2),new_timesteps[final_time]+conventional_time_step/2,new_timesteps[initial_time]-conventional_time_step/2],aspect=50)
	cbar = fig.colorbar(IM)
	cbar.set_label('Emissivity [W m^-3 sr^-1]')
	CS = ax.contour(r*1000,new_timesteps[initial_time:final_time+1],inverted_profiles_sigma[initial_time:final_time+1,iR],colors='k',linewidths=1)
	cbar.add_lines(CS)
	plt.title('Line n='+str(iR+4) + r'$\to$' + '2' + '\nuncertainty')
	plt.xlabel('radius [mm]')
	plt.ylabel('time [ms]')
	plt.savefig('%s/abel_inverted_profile_sigma%s_short.eps' % (path, iR+4), bbox_inches='tight')
	plt.close()
for iR in range(nLine):
	plt.figure(figsize=(5, 10))
	max_value = np.median(np.max(inverted_profiles[:6,iR], axis=1))
	plt.imshow(inverted_profiles[:,iR],'rainbow',vmin=0,vmax=max_value,extent=[1000*min(r-dx/2),1000*max(r+dx/2),last_time+conventional_time_step/2,first_time-conventional_time_step/2],aspect=20)
	plt.colorbar().set_label('Emissivity [W m^-3 sr^-1]')
	# plt.contour(r*1000,new_timesteps,inverted_profiles[:,iR],cmap='rainbow',linewidths=2,vmin=0,vmax=max_value)
	plt.title('SS Line n='+str(iR+4) + r'$\to$' + '2')
	plt.xlabel('radius [mm]')
	plt.ylabel('time [ms]')
	plt.savefig('%s/abel_inverted_profile_SS_%s.eps' % (path, iR+4), bbox_inches='tight')
	plt.close()
for iR in range(nLine):
	fig,ax = plt.subplots(figsize=(10 , 10))
	# IM = ax.imshow(all_fits[:,:,iR],'rainbow',vmin=0,vmax=np.nanmax(np.mean(np.sort(all_fits[:,:,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(xx-dx/2),1000*max(xx+dx/2),last_time+conventional_time_step/2,first_time-conventional_time_step/2],aspect=20)
	IM = ax.imshow(all_fits[:,:,iR],'rainbow',extent=[1000*min(xx-dx/2),1000*max(xx+dx/2),last_time+conventional_time_step/2,first_time-conventional_time_step/2],aspect=20)
	ax.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, int((last_time -first_time) // time_res) + 2), 'k', linewidth=0.4,label='used centre')
	plt.title('Line n='+str(iR+4) + r'$\to$' + '2' +'\n used centre black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)\nfitted profiles = magenta')
	if np.sum(np.array(profile_centres)!=0)>0:
		ax.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, int((last_time -first_time) // time_res) + 2), '--r', linewidth=0.4, label='local centre')
		plt.title('Line n='+str(iR+4) + r'$\to$' + '2' +'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centres[:,iR],axis=(0)), decimals=1))+'mm) \n used centre black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)\nfitted profiles = magenta')
	cbar = fig.colorbar(IM)
	cbar.set_label('Line integrated brightness [W m^-2 sr^-1]')
	CS = ax.contour(xx*1000,new_timesteps,all_fits[:,:,iR],colors='k',linewidths=1)
	cbar.add_lines(CS)
	CS2 = ax.contour(xx*1000,new_timesteps,all_fits_fitted[:,:,iR],colors='m',linewidths=1,linestyles='--')
	cbar.add_lines(CS2)
	plt.xlabel('LOS location [mm]')
	plt.ylabel('time [ms]')
	plt.savefig('%s/line_integrted_profile%s.eps' % (path, iR+4), bbox_inches='tight')
	plt.close()
for iR in range(nLine):
	fig,ax = plt.subplots(figsize=(10 , 10))
	IM = ax.imshow(all_fits[:,:,iR]/all_fits_sigma[:,:,iR],'rainbow',vmin=0,vmax=np.nanmax(np.mean(np.sort(all_fits[:,:,iR]/all_fits_sigma[:,:,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(xx-dx/2),1000*max(xx+dx/2),last_time+conventional_time_step/2,first_time-conventional_time_step/2],aspect=20)
	# IM = ax.imshow(all_fits[:,:,iR]/all_fits_sigma[:,:,iR],'rainbow',extent=[1000*min(xx-dx/2),1000*max(xx+dx/2),last_time+conventional_time_step/2,first_time-conventional_time_step/2],aspect=20)
	ax.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, int((last_time -first_time) // time_res) + 2), 'k', linewidth=0.4,label='used centre')
	plt.title('SNR Line n='+str(iR+4) + r'$\to$' + '2' +'\n used centre black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
	if np.sum(np.array(profile_centres)!=0)>0:
		ax.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, int((last_time -first_time) // time_res) + 2), '--r', linewidth=0.4, label='local centre')
		plt.title('SNR Line n='+str(iR+4) + r'$\to$' + '2' +'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centres[:,iR],axis=(0)), decimals=1))+'mm) \n used centre black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
	cbar = fig.colorbar(IM)
	cbar.set_label('Line integrated brightness [W m^-2 sr^-1]')
	CS = ax.contour(xx*1000,new_timesteps,all_fits[:,:,iR]/all_fits_sigma[:,:,iR],colors='k',linewidths=1)
	cbar.add_lines(CS)
	plt.xlabel('LOS location [mm]')
	plt.ylabel('time [ms]')
	plt.savefig('%s/line_integrted_profile_SNR%s.eps' % (path, iR+4), bbox_inches='tight')
	plt.close()
for iR in range(nLine):
	fig,ax = plt.subplots(figsize=(10 , 10))
	# vmax = max(all_fits_sigma[:,:,iR].max(),all_fits[:,:,iR].max())
	vmax=np.nanmax(np.mean(np.sort(all_fits_sigma[:,:,iR],axis=-1)[:,-10:-1],axis=-1))
	# IM = ax.imshow(all_fits_sigma[:,:,iR],'rainbow',vmin=0,vmax=np.nanmax(np.mean(np.sort(all_fits_sigma[:,:,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(xx-dx/2),1000*max(xx+dx/2),last_time+conventional_time_step/2,first_time-conventional_time_step/2],aspect=20)
	IM = ax.imshow(all_fits_sigma[:,:,iR],'rainbow',vmax=vmax,extent=[1000*min(xx-dx/2),1000*max(xx+dx/2),last_time+conventional_time_step/2,first_time-conventional_time_step/2],aspect=20)
	ax.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), 'k', linewidth=0.4,label='used centre')
	plt.title('Line n='+str(iR+4) + r'$\to$' + '2\nuncertainty' +'\n used centre black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
	if np.sum(np.array(profile_centres)!=0)>0:
		ax.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), '--r', linewidth=0.4, label='local centre')
		plt.title('Line n='+str(iR+4) + r'$\to$' + '2\nuncertainty' +'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centres[:,iR],axis=(0)), decimals=1))+'mm) \n used centre black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
	cbar = fig.colorbar(IM)
	cbar.set_label('Line integrated brightness sigma[W m^-2 sr^-1]')
	CS = ax.contour(xx*1000,new_timesteps,all_fits_sigma[:,:,iR],colors='k',linewidths=1)
	cbar.add_lines(CS)
	plt.xlabel('LOS location [mm]')
	plt.ylabel('time [ms]')
	plt.savefig('%s/line_integrted_profile_sigma%s.eps' % (path, iR+4), bbox_inches='tight')
	plt.close()
for iR in range(nLine):
	fig,ax = plt.subplots(figsize=(10 , 10))
	vmax = max(all_fits_fitted[:,:,iR].max(),max(all_fits_sigma[:,:,iR].max(),all_fits[:,:,iR].max()))
	IM = ax.imshow(all_fits_fitted[:,:,iR],'rainbow',vmax=vmax,extent=[1000*min(xx-dx/2),1000*max(xx+dx/2),last_time+conventional_time_step/2,first_time-conventional_time_step/2],aspect=20)
	ax.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), 'k', linewidth=0.4,label='used centre')
	plt.title('Fitting of Line n='+str(iR+4) + r'$\to$' + '2' +'\n used centre black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
	if np.sum(np.array(profile_centres)!=0)>0:
		ax.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), '--r', linewidth=0.4, label='local centre')
		plt.title('Fitting of Line n='+str(iR+4) + r'$\to$' + '2' +'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centres[:,iR],axis=(0)), decimals=1))+'mm) \n used centre black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
	cbar = fig.colorbar(IM)
	cbar.set_label('Line integrated brightness [W m^-2 sr^-1]')
	CS = ax.contour(xx*1000,new_timesteps,all_fits_fitted[:,:,iR],colors='k',linewidths=1)
	cbar.add_lines(CS)
	plt.xlabel('LOS location [mm]')
	plt.ylabel('time [ms]')
	plt.savefig('%s/line_integrted_profile_fitting%s.eps' % (path, iR+4), bbox_inches='tight')
	plt.close()
for iR in range(nLine):
	fig,ax = plt.subplots(figsize=(10 , 10))
	IM = ax.imshow(all_fits_residuals[:,:,iR],'rainbow',extent=[1000*min(xx-dx/2),1000*max(xx+dx/2),last_time+conventional_time_step/2,first_time-conventional_time_step/2],aspect=20)
	ax.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), 'k', linewidth=0.4,label='used centre')
	plt.title('Fitting residuals of Line n='+str(iR+4) + r'$\to$' + '2' +'\n used centre black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
	if np.sum(np.array(profile_centres)!=0)>0:
		ax.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), '--r', linewidth=0.4, label='local centre')
		plt.title('Fitting residuals of Line n='+str(iR+4) + r'$\to$' + '2' +'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centres[:,iR],axis=(0)), decimals=1))+'mm) \n used centre black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
	cbar = fig.colorbar(IM)
	cbar.set_label('Deviation from fit [au]')# over maximum fitted value [au]')
	CS = ax.contour(xx*1000,new_timesteps,all_fits_residuals[:,:,iR],colors='k',linewidths=1)
	cbar.add_lines(CS)
	plt.xlabel('LOS location [mm]')
	plt.ylabel('time [ms]')
	plt.savefig('%s/line_integrted_profile_fitting_residuals%s.eps' % (path, iR+4), bbox_inches='tight')
	plt.close()
for iR in range(nLine):
	fig,ax = plt.subplots(figsize=(10 , 10))
	max_value = np.median(np.max(all_fits[:6,:,iR], axis=1))
	IM = ax.imshow(all_fits[:,:,iR],'rainbow',vmin=0,vmax=max_value,extent=[1000*min(xx-dx/2),1000*max(xx+dx/2),last_time+conventional_time_step/2,first_time-conventional_time_step/2],aspect=20)
	ax.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), 'k', linewidth=0.4,label='used centre')
	plt.title('Line n='+str(iR+4) + r'$\to$' + '2' +'\n used centre black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
	if np.sum(np.array(profile_centres)!=0)>0:
		ax.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), '--r', linewidth=0.4, label='local centre')
		plt.title('Line n='+str(iR+4) + r'$\to$' + '2' +'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm) \n used centre black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
	cbar = fig.colorbar(IM)
	cbar.set_label('SS Line integrated brightness [W m^-2 sr^-1]')
	# CS = ax.contour(xx*1000,new_timesteps,all_fits[:,:,iR],colors='k',linewidths=1)
	# cbar.add_lines(CS)
	plt.xlabel('LOS location [mm]')
	plt.ylabel('time [ms]')
	plt.savefig('%s/line_integrted_profile_SS_%s.eps' % (path, iR+4), bbox_inches='tight')
	plt.close()
np.save('%s/profileFits.npy' % (path), fitCoefs)
np.save('%s/profileFits_sigma.npy' % (path), fitCoefs_sigma)
np.save('%s/profile_centre.npy' % (path), profile_centre)
np.save('%s/inverted_profiles.npy' % (path), inverted_profiles)
np.save('%s/inverted_profiles_sigma.npy' % (path), inverted_profiles_sigma)

del number_of_radial_divisions,same_centre_every_line,same_centre_all_time,max_central_depression_amplitude,min_central_depression_sigma
