def doLateralfit(df_settings):

	nLine = 10

	import os
	import pandas as pd
	import numpy as np
	from scipy.optimize import curve_fit
	import matplotlib.pyplot as plt


	pdir1 = './plots/LateralGauss/'
	pdir2 = './plots/RadialGauss/'
	ddir = './results/Radiance/'
	odir = './results/'


	for cdir in [pdir1,pdir2,odir]:
		if not os.path.isdir(cdir):
			os.makedirs(cdir)


	gauss = lambda x,A,sig,x0: A*np.exp(-((x-x0)/sig)**2)
	#gauss2 = lambda x,A1,sig1,A2,sig2,x0: gauss(x,A1,sig1,x0) + gauss(x,A2,sig2,x0)
	gauss3 = lambda x,A1,sig1,A2,sig2,A3,sig3,x0: gauss(x,A1,sig1,x0) + gauss(x,A2,sig2,x0) + gauss(x,A3,sig3,x0)
	gaussJac = lambda x,A,sig,x0: np.array([gauss(x,1,sig,x0) , gauss(x,A,sig,x0)*2*(x-x0)**2/sig**3 , gauss(x,A,sig,x0)*2*(x-x0)/sig**2])
	gauss3Jac = lambda x,A1,sig1,A2,sig2,A3,sig3,x0:   [np.concatenate([a[:2],a[3:5],a[6:8],[sum(a[2::3])]]) for a in [np.concatenate([gaussJac(x,A1,sig1,x0),gaussJac(x,A2,sig2,x0),gaussJac(x,A3,sig3,x0)])]][0]
	AInvgauss = lambda x,A,sig,x0: gauss(x,A/(sig/np.sqrt(np.pi)),sig,0)
	AInvgauss3 = lambda x,A1,sig1,A2,sig2,A3,sig3,x0: AInvgauss(x,A1,sig1,0) + AInvgauss(x,A2,sig2,0) + AInvgauss(x,A3,sig3,0)
	AInvgaussJac = lambda x,A,sig,x0: np.array([gauss(x,1/np.sqrt(np.pi)/sig,sig,0) , gauss(x,A,sig,0)*(2*x**2 - sig**2)/sig**4/np.sqrt(np.pi) , gauss(x,A,sig,0)*2*x/sig**3/np.sqrt(np.pi)])
	AInvgauss3Jac = lambda x,A1,sig1,A2,sig2,A3,sig3,x0:   [np.concatenate([a[:2],a[3:5],a[6:8],[sum(a[2::3])]]) for a in [np.concatenate([AInvgaussJac(x,A1,sig1,0),AInvgaussJac(x,A2,sig2,0),AInvgaussJac(x,A3,sig3,0)])]][0]
	LRvecmul = lambda vec,mat:  np.dot(vec,np.matmul(mat,vec))
	#bds2 = [[0,1e-4,0,1e-4,0],[np.inf,np.inf,np.inf,np.inf,np.inf]]
	bds3 = [[0,1e-4,0,1e-4,-np.inf,1e-4,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,np.inf]]

	dx = 18/40*(50.5/27.4)/1e3
	xx = np.arange(40)*dx #m
	xn = np.linspace(0,max(xx),1000)
	r = np.linspace(-max(xx)/2,max(xx)/2,1000)
	bds = [[-np.inf,dx,-np.inf,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,max(xx)]]
	fitCoefs = np.zeros((len(df_settings),nLine,len(bds3[0])))
	leg = ['n = %i'%i for i in range(3,13)]

	for iSet in range(len(df_settings)):
		df_lineRads = pd.read_csv('%s%i.csv'%(ddir,iSet))
		for iC,col in enumerate(df_lineRads.columns):
		#for col in ['n5']:
			yy = df_lineRads[col]
			if not yy.isnull().any():
				ym = max(yy)
				p0 = [ym/5,20e-3,ym*.8,4e-3,-ym/5,1e-3,20*dx]
				#bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
				fit = curve_fit(gauss3,xx,yy,p0,maxfev=100000,bounds=bds)
				fitCoefs[iSet,iC] = fit[0]

				invProf = AInvgauss3(r,*fit[0])
				plt.figure()
				plt.plot(xx*1e3,yy)
				plt.plot(xn*1e3,gauss3(xn,*fit[0]))
				plt.xlabel('Lateral distance [mm]')
				plt.ylabel('Radiance [a.u.]')
				plt.title('Setting %i, n = %s'%(iSet,col[1:]))
				plt.savefig('%s%i_%s.png'%(pdir1,iSet,col),bbox_inches='tight')
				plt.savefig('%s%s_%i.png'%(pdir1,col,iSet),bbox_inches='tight')

				plt.figure()
				plt.plot(r*1e3,invProf)
				plt.xlabel('Radial distance [mm]')
				plt.ylabel('Emission coefficient [a.u.]')
				plt.savefig('%s%i_%s.png'%(pdir2,iSet,col),bbox_inches='tight')
				plt.savefig('%s%s_%i.png'%(pdir2,col,iSet),bbox_inches='tight')

		plt.figure()
		for coef in fitCoefs[iSet]:
			plt.plot(r*1e3,AInvgauss3(r,*coef))
		plt.xlabel('Radial distance [mm]')
		plt.ylabel('Emission coefficient [W sr^{-1} m^{-3)}]')
		plt.legend(leg)
		plt.savefig('%sall_%i.png'%(pdir2,iSet),bbox_inches='tight')

		plt.figure()
		for coef in fitCoefs[iSet]:
			plt.semilogy(r*1e3,AInvgauss3(r,*coef))
		plt.xlabel('Radial distance [mm]')
		plt.ylabel('Emission coefficient [W sr^{-1} m^{-3)}]')
		plt.legend(leg)
		plt.savefig('%sallLog_%i.png'%(pdir2,iSet),bbox_inches='tight')
		plt.close('all')
	np.save('./results/profileFits.npy',fitCoefs)

	# Checking profile shape
	'''
	df_rMax = pd.DataFrame(index = range(len(df_settings)),columns=df_lineRads.columns)
	df_rdec = pd.DataFrame(index = range(len(df_settings)),columns=df_lineRads.columns)
	for iSet in range(len(df_settings)):
		for iC,col in enumerate(df_lineRads.columns):
			invProf = AInvgauss3(r,*fitCoefs[iSet,iC])
			df_rMax[col].loc[iSet] = np.abs(r[invProf.argmax()])
			df_rdec[col].loc[iSet] = np.abs(r[(invProf > max(invProf)/np.exp(1)).argmax()])

	plt.figure()
	plt.plot(df_rMax.n4,'.')

	rMax = [2,3,3]

	plt.figure()
	plt.plot(df_rdec.n4,'.')

	rDec = [5,5,6]
	'''



	# Fitting functions for untransformed Gaussians
	'''
	Agauss = lambda x,A,sig,x0: gauss(x,A*sp.sqrt(sp.pi)*sig,sig,x0)
	Agauss2 = lambda x,A1,sig1,A2,sig2,x0: Agauss(x,A1,sig1,x0) + Agauss(x,A2,sig2,x0)
	Agauss3 = lambda x,A1,sig1,A2,sig2,A3,sig3,x0: Agauss(x,A1,sig1,x0) + Agauss(x,A2,sig2,x0) + Agauss(x,A3,sig3,x0)
	AgaussJac = lambda x,A,sig,x0: np.array([gauss(x,sig*sp.sqrt(sp.pi),sig,x0) , gauss(x,A,sig,x0)*(2*(x-x0)**2+sig**2)*sp.sqrt(sp.pi)/sig**2 , gauss(x,A,sig,x0)*2*(x-x0)*sp.sqrt(sp.pi)/sig])
	Agauss3Jac = lambda x,A1,sig1,A2,sig2,A3,sig3,x0:   [np.concatenate([a[:2],a[3:5],a[6:8],[sum(a[2::3])]]) for a in [np.concatenate([AgaussJac(x,A1,sig1,x0),AgaussJac(x,A2,sig2,x0),AgaussJac(x,A3,sig3,x0)])]][0]
	'''
	# Plot individual Gaussians
	'''
	plt.figure()
	eb=plt.errorbar(xx[use],yy[use],yerr=ye[use],ls='',marker='.',label='Measurements')
	p,=plt.plot(xn,yfit,'r')
	plt.fill_between(xn,yfit-Dyfit,yfit+Dyfit,color='r',alpha=0.2)
	fb,=plt.fill(sp.nan,sp.nan,'r',alpha=0.2)
	y1 = list(map(lambda x: gauss(x,coefs[0],coefs[1],coefs[-1]),xn))
	y2 = list(map(lambda x: gauss(x,coefs[2],coefs[3],coefs[-1]),xn))
	y3 = list(map(lambda x: gauss(x,coefs[4],coefs[5],coefs[-1]),xn))
	g1,g2,g3 = plt.plot(xn,y1,xn,y2,xn,y3)
	plt.legend([eb,(p,fb),g1,g2,g3],['Measurements',nGstr+' fit','Gauss #1','Gauss #2','Gauss #3'])
	plt.xlabel('Lateral distance [mm]')
	plt.ylabel('Radiance [a.u.]')
	'''
	# Test covariance
	'''
	k=6
	y = np.array(list(map(lambda x:gauss3(x,*coefs),xn)))
	er = np.zeros(coefvar.shape)
	er[k]=coefvar[k]
	Dy = np.sqrt(list(map(lambda x:gauss3cov(x,*np.concatenate([coefs,er])),xn)))
	plt.figure()
	plt.fill_between(xn,y-Dy,y+Dy,alpha=0.2,color='r')
	plt.plot(xn,y,'r')
	'''
	# Plot both double and triple Gauss
	'''
	yfit2 = list(map(lambda x:gauss2(x,*coefs2),xn))
	yfit3 = list(map(lambda x:gauss3(x,*coefs3),xn))
	plt.figure()
	eb=plt.errorbar(xx[use],yy[use],yerr=ye[use],ls='',marker='.',label='Measurements')
	p2,p3=plt.plot(xn,yfit2,xn,yfit3)
	plt.legend([eb,p2,p3],['Measurements','Double Gauss fit','Triple Gauss fit'])
	plt.xlabel('Lateral distance [mm]')
	plt.ylabel('Radiance [a.u.]')
	'''
	# Abel invert the fitted coefficients and covariances (incorrect)
	'''
	Invcoefs = np.copy(coefs)
	Invcoefs[-1]=0
	Invcoefs[:-1:2]/=(Invcoefs[1::2]*sp.sqrt(sp.pi))
	Invcoefcov = np.copy(coefcov)
	for i in range(Invcoefs.shape[0]):
		Invcoefcov[:-1:2,i]/=(Invcoefs[1::2]*sp.sqrt(sp.pi))
		Invcoefcov[i,:-1:2]/=(Invcoefs[1::2]*sp.sqrt(sp.pi))
	'''
	# Check the (incorrect) re-transformation of the Abel inversion
	'''
	ReTranscoefs = np.copy(Invcoefs)
	ReTranscoefs[-1] = coefs[-1]
	ReTranscoefs[:-1:2]*=(Invcoefs[1::2]*sp.sqrt(sp.pi))
	ReTranscoefcov = np.copy(Invcoefcov)
	for i in range(ReTranscoefs.shape[0]):
		ReTranscoefcov[:-1:2,i]*=(ReTranscoefs[1::2]*sp.sqrt(sp.pi))
		ReTranscoefcov[i,:-1:2]*=(ReTranscoefs[1::2]*sp.sqrt(sp.pi))
	yReTrans = np.array(list(map(lambda x:gauss3(x,*ReTranscoefs),xn)))
	#Fit uncertainty as J*Sigma*J', with J the Jacobian (partial derivatives wrt the fit parameters) and Sigma the parameter covariance
	DyReTrans = np.sqrt(list(map(lambda x:LRvecmul(gauss3Jac(x,*ReTranscoefs),ReTranscoefcov),xn)))
	p,=plt.plot(1e3*xn,yReTrans)
	plt.fill_between(1e3*xn,yReTrans-DyReTrans,yReTrans+DyReTrans,color=p.get_color(),alpha=0.2)
	plt.xlabel('Radial distance [mm]')
	plt.ylabel('Emission coefficient [a.u.]')
	'''
	# Plot initial guess
	'''
	plt.figure()
	plt.errorbar(xx[use],yy[use],yerr=ye[use],ls='',marker='.')
	plt.plot(xn,list(map(gauss2,xn,*[np.ones(xn.shape)*c for c in p02])),xn,list(map(gauss3,xn,*[np.ones(xn.shape)*c for c in p03])))
	plt.legend(['Measurements','Double gauss','Triple gauss'])
	plt.xlabel('Lateral distance')
	plt.ylabel('Radiance')
	'''
	# Compare dogbox and trf optimisation methods: trf seems to give smaller residuals, and be more robust to initial conditions.
	#But it cheats by using broad negative Gaussians...
	'''
	coefs1,pcov1 = sp.optimize.curve_fit(gauss3,xx[use],yy[use],p0=p03,bounds=bds3,method='dogbox')#sigma=ye,
	yfit1 = list(map(gauss3,xn,*[np.ones(xn.shape)*c for c in coefs1]))
	sqres1 = sum(list(map(gauss3,xx,*[np.ones(xn.shape)*c for c in coefs1]))-yy)**2
	coefs2,pcov2 = sp.optimize.curve_fit(gauss3,xx[use],yy[use],p0=p03,bounds=bds3,method='trf')#sigma=ye,
	yfit2 = list(map(gauss3,xn,*[np.ones(xn.shape)*c for c in coefs2]))
	sqres2 = sum(list(map(gauss3,xx,*[np.ones(xn.shape)*c for c in coefs2]))-yy)**2
	print('dogbox square residual:',sqres1)
	print('trf square residual:',sqres2)
	if sqres1 < sqres2:
		coefs = coefs1
	else:
		coefs = coefs2
	'''
	# Plot residuals
	'''
	plt.figure()
	plt.plot(xx[use],res2,'.',xx[use],res3,'.')
	plt.legend(['Double gauss','Triple gauss'])
	plt.xlabel('Lateral distance')
	plt.ylabel('Radiance')
	'''


def doLateralfit_time_tependent(df_settings,all_fits, merge_ID_target, new_timesteps,dx,xx,r,same_centre_every_line=False,same_centre_all_time=True,force_glogal_center=0,max_central_depression_amplitude=-1,min_central_depression_sigma=0.1):

	import os
	import pandas as pd
	import numpy as np
	from scipy.optimize import curve_fit
	import matplotlib.pyplot as plt
	import copy as cp

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
	gauss_bias = lambda x, A, sig, x0,m,q: A * np.exp(-((x - x0) / sig) ** 2)+q+x*m
	# gauss2 = lambda x,A1,sig1,A2,sig2,x0: gauss(x,A1,sig1,x0) + gauss(x,A2,sig2,x0)
	gauss3 = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: gauss(x, A1, sig1, x0) + gauss(x, A2, sig2, x0) + gauss(x, A3,sig3,x0)
	gauss3_bias = lambda x, A1, sig1, A2, sig2, A3, sig3, x0,m,q: gauss(x, A1, sig1, x0) + gauss(x, A2, sig2, x0) + gauss(x, A3,sig3,x0)+q+x*m
	gaussJac = lambda x, A, sig, x0: np.array([gauss(x, 1, sig, x0), gauss(x, A, sig, x0) * 2 * (x - x0) ** 2 / sig ** 3,gauss(x, A, sig, x0) * 2 * (x - x0) / sig ** 2])
	gauss3Jac = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: [np.concatenate([a[:2], a[3:5], a[6:8], [sum(a[2::3])]]) for a in [np.concatenate([gaussJac(x, A1, sig1, x0), gaussJac(x, A2, sig2, x0), gaussJac(x, A3, sig3, x0)])]][0]
	AInvgauss = lambda x, A, sig, x0: gauss(x, A / sig / np.sqrt(np.pi), sig, 0)
	AInvgauss3 = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: AInvgauss(x, A1, sig1, 0) + AInvgauss(x, A2, sig2,0) + AInvgauss(x, A3,sig3,0)
	AInvgaussJac = lambda x, A, sig, x0: np.array([gauss(x, 1 / np.sqrt(np.pi) / sig, sig, 0),gauss(x, A, sig, 0) * (2 * x ** 2 - sig ** 2) / sig ** 4 / np.sqrt(np.pi), gauss(x, A, sig, 0) * 2 * x / sig ** 3 / np.sqrt(np.pi)])
	AInvgauss3Jac = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: [np.concatenate([a[:2], a[3:5], a[6:8], [sum(a[2::3])]]) for a in [np.concatenate([AInvgaussJac(x, A1, sig1, 0), AInvgaussJac(x, A2, sig2, 0), AInvgaussJac(x, A3, sig3, 0)])]][0]
	LRvecmul = lambda vec, mat: np.dot(vec, np.matmul(mat, vec))
	gauss3_2 = lambda x, A1, sig1, A2, c_sig2, c_A3, c_sig3,x0: gauss(x, A1, sig1, x0) + gauss(x,A2,c_sig2*sig1,x0) + gauss(x, c_A3*c_sig3*(A1 +A2/c_sig2), c_sig3*sig1, x0)
	# bds2 = [[0,1e-4,0,1e-4,0],[np.inf,np.inf,np.inf,np.inf,np.inf]]
	bds3 = [[0, 1e-4, 0, 1e-4, -np.inf, 1e-4, 0], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, np.inf]]

	# dx = 18 / 40 * (50.5 / 27.4) / 1e3
	# xx = np.arange(40) * dx  # m
	xn = np.linspace(0, max(xx), 1000)
	# r = np.linspace(-max(xx) / 2, max(xx) / 2, 1000)
	# bds = [[-np.inf, dx, -np.inf, dx, -np.inf, dx, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, max(xx)]]
	bds = [[0, dx, -np.inf, dx, -np.inf, dx, 0], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, max(xx)]]
	bds4 = [[0, dx, 0], [np.inf, np.inf, max(xx)]]
	fitCoefs = np.zeros((len(all_fits), nLine, 6))
	# fitCoefs = np.zeros((len(all_fits), nLine, 8))
	leg = ['n = %i' % i for i in range(3, 13)]

	# profile_centres = np.zeros((len(all_fits),nLine))
	# profile_centres_score = np.zeros((len(all_fits),nLine))
	# fit = [[0, 0, 20 * dx]]
	# for iSet in range(len(all_fits)):
	# 	# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
	# 	for iR in range(np.shape(all_fits)[-1]):
	# 		# # for col in ['n5']:
	# 		yy = all_fits[iSet,:,iR]
	# 		for iCorr, value in enumerate(yy):
	# 			if np.isnan(value):
	# 				yy[iCorr]=0
	# 		# if not yy.isnull().any():
	# 		ym = np.nanmax(yy)
	# 		# p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 3, 1e-3, 20 * dx]
	# 		p0 = [ym, 4e-3, fit[0][-1]]
	# 		bds4 = [[0, dx, 0], [np.inf, np.inf, max(xx)]]
	# 		# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
	# 		try:
	# 			# fit = curve_fit(gauss, xx, yy, p0, maxfev=100000, bounds=np.array(bds)[:,4:])
	# 			fit = curve_fit(gauss, xx, yy, p0, maxfev=100000, bounds=np.array(bds4))
	# 		except:
	# 			fit=[p0]
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
	# 		profile_centres_score[iSet, iR] = np.sum(((yy - gauss(xx, *fit[0]))/yy) ** 2)
	# # profile_centre = np.mean(profile_centres,axis=-1)
	# profile_centre = np.sum(profile_centres/(profile_centres_score**2), axis=-1)/np.sum(1/(profile_centres_score**2), axis=-1)
	# for iSet in range(len(all_fits)):
	# 	# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
	# 	for iR in range(np.shape(all_fits)[-1]):
	# 		# # for col in ['n5']:
	# 		yy = all_fits[iSet, :, iR]
	# 		for iCorr, value in enumerate(yy):
	# 			if np.isnan(value):
	# 				yy[iCorr] = 0
	# 		# if not yy.isnull().any():
	# 		ym = np.nanmax(yy)
	# 		p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 4, 0.9e-3, profile_centre[iSet]]
	# 		bds = [[0, dx, -np.inf, dx, -np.inf, dx, profile_centre[iSet]-2*dx], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, profile_centre[iSet]+2*dx]]
	# 		# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
	# 		try:
	# 			fit = curve_fit(gauss3, xx, yy, p0, maxfev=100000, bounds=bds)
	# 		except:
	# 			fit=[p0]
	# 		profile_centres[iSet, iR] = fit[0][-1]
	# 		if len(fit)==1:
	# 			profile_centres_score[iSet, iR] = 1
	# 		else:
	# 			# profile_centres_score[iSet, iR] = fit[1][-1, -1]
	# 			# profile_centres_score[iSet, iR] = fit[1][-1, -1] / ((np.nanmean(yy)/ym)**4)
	# 			profile_centres_score[iSet, iR] = np.sum(((yy - gauss3(xx, *fit[0]))/yy) ** 2)
	# 	# profile_centres_score[iSet, iR] = np.sum((yy - gauss3(xx, *fit[0])) ** 2)
	# 	# plt.figure()
	# 		# plt.plot(xx, yy)
	# 		# plt.plot(xx, gauss3(xx, *fit[0]))
	# 		# plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
	# 		# plt.title('score: '+str(np.diag(fit[1])))
	# 		# plt.pause(0.01)
	# # profile_centre = np.mean(profile_centres, axis=-1)
	# profile_centre = np.sum(profile_centres/(profile_centres_score**2), axis=-1)/np.sum(1/(profile_centres_score**2), axis=-1)
	# for iSet in range(len(all_fits)):
	# 	# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
	# 	for iR in range(np.shape(all_fits)[-1]):
	# 		# # for col in ['n5']:
	# 		yy = all_fits[iSet, :, iR]
	# 		for iCorr, value in enumerate(yy):
	# 			if np.isnan(value):
	# 				yy[iCorr] = 0
	# 		# if not yy.isnull().any():
	# 		ym = np.nanmax(yy)
	# 		p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 4, 0.9e-3, profile_centre[iSet]]
	# 		bds = [[0, dx, -np.inf, dx, -np.inf, dx, profile_centre[iSet] - 2 * dx],[np.inf, np.inf, np.inf, np.inf, 0, np.inf, profile_centre[iSet] + 2 * dx]]
	# 		# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
	# 		try:
	# 			fit = curve_fit(gauss3, xx, yy, p0, maxfev=100000, bounds=bds)
	# 		except:
	# 			fit=[p0]
	# 		profile_centres[iSet, iR] = fit[0][-1]
	# 		if len(fit)==1:
	# 			profile_centres_score[iSet, iR] = 1
	# 		else:
	# 			# profile_centres_score[iSet, iR] = fit[1][-1, -1]
	# 			# profile_centres_score[iSet, iR] = fit[1][-1, -1] / ((np.nanmean(yy)/ym)**4)
	# 			profile_centres_score[iSet, iR] = np.sum(((yy - gauss3(xx, *fit[0]))/yy) ** 2)
	# 		# profile_centres_score[iSet, iR] = np.sum((yy - gauss3(xx, *fit[0])) ** 2)
	# 		# if iSet in [35,70]:
	# 		# 	plt.figure()
	# 		# 	plt.plot(xx, yy)
	# 		# 	plt.plot(xx, gauss3(xx, *fit[0]))
	# 		# 	plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
	# 		# 	plt.title('score: '+str(np.diag(fit[1])))
	# 		# 	plt.pause(0.01)
	# # profile_centre = np.mean(profile_centres, axis=-1)
	# profile_centre = np.sum(profile_centres/(profile_centres_score**2), axis=-1)/np.sum(1/(profile_centres_score**2), axis=-1)

	minimum_level = np.min(all_fits[all_fits!=0])
	if (same_centre_every_line==True and same_centre_all_time==True):
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
				bds = [[0, 0, 0, 1, max_central_depression_amplitude, min_central_depression_sigma,profile_centre - 2 * dx],[max(yy)*100, np.inf, max(yy)*10, np.inf, 0, 1,profile_centre + 2 * dx]]
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
					fit = curve_fit(gauss, xx_good, yy_good, p0, maxfev=10000, bounds=np.array(bds4))
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
				p0 = [ym / 6, 30*dx, ym * .8, 5*dx, -ym / 4, 1.1*dx, profile_centre[iR]]
				bds = [[0, dx, -np.inf, dx, -np.inf, dx, profile_centre[iR]-2*dx], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, profile_centre[iR]+2*dx]]
				# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
				try:
					fit = curve_fit(gauss3, xx_good, yy_good, p0, maxfev=10000, bounds=bds)
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
				p0 = [ym *0.9, 6*dx, ym * .05, 300, -0.25, 0.5,profile_centre[iR]]
				bds = [[0, 0, 0, 1, max_central_depression_amplitude, min_central_depression_sigma,profile_centre[iR] - 2 * dx],[max(yy)*100, np.inf, max(yy)*10, np.inf, 0, 1,profile_centre[iR] + 2 * dx]]
				x_scale = [ym,dx,ym,1,1,1,profile_centre[iR]]
				# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
				try:
					# fit = curve_fit(gauss3, xx_good, yy_good, p0, maxfev=100000, bounds=bds)
					fit = curve_fit(gauss3_2, xx_good, yy_good, p0, maxfev=100000, bounds=bds,x_scale=x_scale)
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


	# profile_centres = np.zeros((len(all_fits)))
	# profile_centres_score = np.zeros((len(all_fits)))
	# fit = [[0, 20e-3, 0, 4e-3, 0, 0.9e-3, 20 * dx]]
	# p0_1 = [0, 0, 0, 0, 0, 0, fit[0][-1]]
	# iR = int(np.median(all_fits.argmax(axis=-1)))	#I want the stronger signal
	# for iSet in range(len(all_fits)):
	# 	# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
	# 	# for iR in range(np.shape(all_fits)[-1]):
	# 	# iR=0
	# 	# # for col in ['n5']:
	# 	yy = all_fits[iSet,:,iR]
	# 	for iCorr, value in enumerate(yy):
	# 		if np.isnan(value):
	# 			yy[iCorr]=0
	# 	# if not yy.isnull().any():
	# 	ym = np.nanmax(yy)
	# 	# p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 3, 1e-3, 20 * dx]
	# 	bds = [[0, dx, -np.inf, dx, -np.inf, dx, 0], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, max(xx)]]
	# 	p0 = [ym / 6, fit[0][1], ym * .8, fit[0][3], -ym / 4, fit[0][5], fit[0][-1]]
	# 	# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
	# 	try:
	# 		# fit = curve_fit(gauss, xx, yy, p0, maxfev=100000, bounds=np.array(bds)[:,4:])
	# 		fit = curve_fit(gauss3, xx, yy, p0, maxfev=100000, bounds=bds)
	# 	except:
	# 		fit=[p0]
	# 	if iSet>1:
	# 		p0_1 = [0, 0, 0, 0, 0, 0, profile_centres[iSet-2]]
	# 	# plt.figure()
	# 	# plt.plot(xx, yy)
	# 	# plt.plot(xx, gauss3(xx, *fit[0]))
	# 	# plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
	# 	# plt.title('score: '+str(np.diag(fit[1])))
	# 	# plt.pause(0.01)
	# 	profile_centres[iSet]=fit[0][-1]
	# 	profile_centres_score[iSet] = np.sum((yy - gauss3(xx, *fit[0]))**2)*((fit[0][-1]-(p0_1[-1]+p0[-1])/2)**2)/(fit[0][4]**2)
	# # profile_centre = np.mean(profile_centres,axis=-1)
	# profile_centre = np.sum(profile_centres/(profile_centres_score**2), axis=-1)/np.sum(1/(profile_centres_score**2), axis=-1)
	#
	# fit = [[0, 20e-3, 0, 4e-3, 0, 2e-3, profile_centre]]
	# p0_1 = [0, 0, 0, 0, 0, 0, fit[0][-1]]
	# iR = int(np.median(all_fits.argmax(axis=-1)))	#I want the stronger signal
	# for iSet in range(len(all_fits)):
	# 	# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
	# 	# for iR in range(np.shape(all_fits)[-1]):
	# 	# iR=0
	# 	# # for col in ['n5']:
	# 	yy = all_fits[iSet,:,iR]
	# 	for iCorr, value in enumerate(yy):
	# 		if np.isnan(value):
	# 			yy[iCorr]=0
	# 	# if not yy.isnull().any():
	# 	ym = np.nanmax(yy)
	# 	# p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 3, 1e-3, 20 * dx]
	# 	p0 = [ym / 6, fit[0][1], ym * .8, fit[0][3], -ym / 2, 2e-3, fit[0][-1]]
	# 	# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
	# 	bds = [[0, 5*dx, 0, 3*dx, -np.inf, dx, fit[0][-1]-dx/4], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, fit[0][-1]+dx/4]]
	# 	try:
	# 		# fit = curve_fit(gauss, xx, yy, p0, maxfev=100000, bounds=np.array(bds)[:,4:])
	# 		fit = curve_fit(gauss3, xx, yy, p0, maxfev=100000, bounds=bds)
	# 	except:
	# 		fit=[p0]
	# 	if iSet>1:
	# 		p0_1 = [0, 0, 0, 0, 0, 0, profile_centres[iSet-2]]
	# 	if iSet in [208, 210, 212, 211,324]:
	# 		plt.figure()
	# 		plt.plot(xx, yy)
	# 		plt.plot(xx, gauss3(xx, *fit[0]))
	# 		plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
	# 		# plt.title('score: '+str(np.diag(fit[1])))
	# 		plt.title('fit: '+str(fit[0]))
	# 		plt.pause(0.01)
	# 	profile_centres[iSet]=fit[0][-1]
	# 	profile_centres_score[iSet] = np.sum((yy - gauss3(xx, *fit[0]))**2)*((fit[0][-1]-(p0_1[-1]+p0[-1])/2)**2)/(fit[0][4]**2)
	# # profile_centre = np.mean(profile_centres,axis=-1)
	# profile_centre = np.sum(profile_centres/(profile_centres_score**2), axis=-1)/np.sum(1/(profile_centres_score**2), axis=-1)
	# # profile_centre = np.convolve(profile_centre, np.ones((30,))/30, mode='valid')

	if force_glogal_center!=0:
		# profile_centres = np.zeros((len(all_fits),nLine))
		profile_centre = np.ones((len(all_fits),nLine))*force_glogal_center

	all_fits_fitted = np.zeros_like(all_fits)
	all_fits_residuals = np.zeros_like(all_fits)
	inverted_profiles = np.zeros((len(all_fits),nLine,len(r)))
	for iSet in range(len(all_fits)):
		# gauss3_locked = lambda x, A1, sig1, A2, sig2, A3, sig3: gauss(x, A1, sig1, profile_centre[iSet]) + gauss(x, A2, sig2, profile_centre[iSet]) + gauss(x,A3,sig3,profile_centre[iSet])
		# AInvgauss3_locked = lambda x, A1, sig1, A2, sig2, A3, sig3: AInvgauss(x, A1, sig1, 0) + AInvgauss(x, A2, sig2,0) + AInvgauss(x,A3,sig3,0)

		# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
		for iR in range(np.shape(all_fits)[-1]):
			# gauss3_locked = lambda x, A1, sig1, A2, sig2, A3, sig3: gauss(x, A1, sig1, profile_centre[iSet,iR]) + gauss(x,A2,sig2,profile_centre[iSet,iR]) + gauss(x, A3, sig3, profile_centre[iSet,iR])
			# AInvgauss3_locked = lambda x, A1, sig1, A2, sig2, A3, sig3: AInvgauss(x, A1, sig1, 0) + AInvgauss(x, A2,sig2,0) + AInvgauss(x, A3, sig3, 0)
			gauss3_locked_2 = lambda x, A1, sig1, A2, c_sig2, c_A3, c_sig3: gauss(x, A1, sig1, profile_centre[iSet,iR]) + gauss(x,A2,c_sig2*sig1,profile_centre[iSet,iR]) + gauss(x, c_A3*c_sig3*(A1 +A2/c_sig2), c_sig3*sig1, profile_centre[iSet,iR])
			gauss3_locked_2_bias = lambda x, A1, sig1, A2, c_sig2, c_A3, c_sig3, m, q: gauss(x, A1, sig1,profile_centre[iR]) + gauss(x, A2,c_sig2 * sig1,profile_centre[iR]) + gauss(x, c_A3 * c_sig3 * (A1 + A2 / c_sig2), c_sig3 * sig1, profile_centre[iR]) + q + x * m
			AInvgauss3_locked_2 = lambda x, A1, sig1, A2, c_sig2, c_A3, c_sig3: AInvgauss(x, A1, sig1, 0) + AInvgauss(x, A2,c_sig2*sig1,0) + AInvgauss(x, c_A3*c_sig3*(A1 +A2/c_sig2), c_sig3*sig1, 0)
			yy = all_fits[iSet, :, iR]
			yy[np.isnan(yy)]=0
			# for iCorr, value in enumerate(yy):
			# 	if np.isnan(value):
			# 		yy[iCorr] = 0
			temp_all_fits = cp.deepcopy(yy)
			if np.max(yy)>100*np.sort(yy)[-2]:
				pos1 = np.max([yy.argmax()-1,0])
				pos2 = np.min([yy.argmax()+1,len(yy)-1])
				yy[yy.argmax()]=np.mean(yy[[pos1,pos2]])

			borders = np.linspace(0, len(yy), len(yy) + 1)[np.diff(np.array([0, *yy, 0]) == 0)].astype('int')
			if len(borders)==0:
				xx_good = xx
				yy_good = yy
			else:
				if borders[0]!=0:	# this should not be necessary but I cannot avoid a smear of composed_array when dealing with data missing in lower lines
					borders[0]+=5
				xx_good = xx[borders[0]:borders[-1]]
				yy_good = yy[borders[0]:borders[-1]]

			# xx_good = np.abs(xx_good-profile_centre[iSet,iR])
			# plt.figure();plt.plot(xx_good,yy_good)
			# yy_good = np.array([peaks for _, peaks in sorted(zip(xx_good, yy_good))])
			# yy_good = np.array([yy_good[0],*np.convolve(yy_good,np.ones((3))/3,'valid'),yy_good[-1]])
			# xx_good = np.sort(xx_good)
			# plt.plot(xx_good, yy_good)
			# plt.pause(0.01)


			ym = np.nanmax(yy_good)
			# inner_dip_sig = 0.9e-3
			inner_dip_sig = 0.15
			# bds = [[0, 0, 0, 0, -max(yy)*10, 0],[max(yy)*100, np.inf, max(yy)*100, np.inf, 0, np.inf]]
			bds = [[0, 0, 0, 1, max_central_depression_amplitude, min_central_depression_sigma],[max(yy)*100, np.inf, max(yy)*1000, np.inf, 0, 1]]
			# p0 = [ym / 5, 20e-3, ym * .8, 4e-1, -ym / 2, 0.9e-3]
			# p0 = [ym *0.9, 6e-3, ym * .05, 4e-1, -ym / 4, inner_dip_sig]
			p0 = [ym *0.9, 6*dx, ym * .05, 300, -0.25, inner_dip_sig]
			x_scale = [ym,dx,ym,1,1,1]
			# p0 = [ym / 5, 20e-3, ym * .8, 4e-3, -ym / 5, 1e-3,profile_centre[iSet]]
			# bds = [[0, 0, 0, 1, -1, 1e-10, -np.inf, -np.inf],[max(yy) * 100, np.inf, max(yy) * 100, np.inf, 0, 1, np.inf, np.inf]]
			# p0 = [ym * 0.9, 5e-3, ym * .05, 67, -0.25, inner_dip_sig, 0, 0]
			try:
				fit = curve_fit(gauss3_locked_2, xx_good, yy_good, p0, maxfev=1000000, bounds=np.array(bds),x_scale=x_scale)
				# fit = curve_fit(gauss3_locked_2_bias, xx_good, yy_good, p0, maxfev=100000, bounds=np.array(bds))
				# while (AInvgauss3_locked_2(0, *fit[0]) < 0 and inner_dip_sig < 2e-3):
				while (AInvgauss3_locked_2(0, *fit[0]) < 0 and inner_dip_sig < 0.3):
				# while (AInvgauss3_locked_2(0, *fit[0][:-2]) < 0 and inner_dip_sig < 0.3):
					# inner_dip_sig+=(0.9e-3)/10
					print('small peak sigma increased')
					inner_dip_sig+=0.015
					# p0 = [ym * 0.9, 6e-3, ym * .05, 4e-1, -ym / 3, inner_dip_sig]
					p0 = [*fit[0][:-1],inner_dip_sig]
					# p0 = [*fit[0][:5], inner_dip_sig, 0, 0]
					fit = curve_fit(gauss3_locked_2, xx_good, yy_good, p0, maxfev=1000000, bounds=np.array(bds),x_scale=x_scale)
					# fit = curve_fit(gauss3_locked_2_bias, xx_good, yy_good, p0, maxfev=100000, bounds=np.array(bds))
				# if fit[0][-2]>0:
				# 	inner_dip_sig = 0.9e-3
				# 	p0 = [ym *0.9, 6e-3, ym * .05, 4e-1, -ym / 3, inner_dip_sig]
			except:
				print('fitting failed for time slice '+str(iSet)+', line '+str(iR))
				fit=[[0,5e-3,0, 67, -0.25,0.15]]
				# fit = [[0, 5e-3, 0, 67, -0.25, 0.15, 0, 0]]
			# fit = curve_fit(gauss3_locked, xx_good, yy_good, p0, maxfev=100000, bounds=bds)
			# plt.figure()
			# # # plt.plot(1000*xx, yy)
			# plt.title('iSet='+str(iSet)+' line='+str(iR)+'\n'+str(fit[0])+'\n R2 '+str(np.sum((temp_all_fits - gauss3_locked_2(xx,*fit[0]))**2)))
			# plt.plot(1000*xx_good, yy_good)
			# # # plt.plot(1000*xx, gauss3_locked(xx, *[*fit[0][:2],0,0,0,0]))
			# # # plt.plot(1000*xx, gauss3_locked(xx, *[*fit[0][:4],0,0]))
			# # plt.plot([1000*profile_centre[iSet,iR],1000*profile_centre[iSet,iR]],[np.min(yy_good),np.max(yy_good)],'--k')
			# plt.plot(1000*xx, gauss3_locked_2(xx, *fit[0]))
			# # plt.ylabel('emission line intensity [au]')
			# # plt.xlabel('impact radius [mm]')
			# # # plt.plot(xx, gauss3(xx, *fit[0]))
			# plt.pause(0.01)
			fitCoefs[iSet, iR] = fit[0]
			# inverted_profiles[iSet, iR] = AInvgauss3_locked(r, *fit[0])
			# all_fits_fitted[iSet, :, iR] = gauss3_locked(xx,*fit[0])
			inverted_profiles[iSet, iR] = AInvgauss3_locked_2(r, *fit[0])
			all_fits_fitted[iSet, :, iR] = gauss3_locked_2(xx,*fit[0])
			# inverted_profiles[iSet, iR] = AInvgauss3_locked_2(r, *fit[0][:-2])
			# all_fits_fitted[iSet, :, iR] = gauss3_locked_2_bias(xx,*fit[0])
			all_fits_residuals[iSet, :, iR] = (temp_all_fits - all_fits_fitted[iSet, :, iR])#/np.abs(np.nanmax(all_fits_fitted[iSet, :, iR]))
			# plt.figure()
			# plt.plot(1000*r, AInvgauss3_locked_2(r, *fit[0]))
			# # plt.ylabel('emission line intensity [au]')
			# # plt.xlabel('radius [mm]')
			# plt.title('iSet='+str(iSet)+' line='+str(iR)+'\n'+str(fit[0])+'\n R2 '+str(np.sum((temp_all_fits - gauss3_locked_2(xx,*fit[0]))**2)))
			# plt.pause(0.01)

	path = '/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)
	if not os.path.exists(path):
		os.makedirs(path)
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(inverted_profiles[:,iR],'rainbow',vmax=np.nanmax(np.mean(np.sort(inverted_profiles[:,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(r),1000*max(r),last_time,first_time],aspect=50)
		plt.title('Line '+str(iR+4))
		plt.xlabel('radius [mm]')
		plt.ylabel('time [ms]')
		plt.colorbar().set_label('Emissivity [W m^-3 sr^-1]')
		plt.savefig('%s/abel_inverted_profile%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	initial_time = int((np.abs(new_timesteps)).argmin())
	final_time = int((np.abs(new_timesteps -2)).argmin())
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(inverted_profiles[initial_time:final_time+1,iR],'rainbow',vmin=0,vmax=np.nanmax(np.mean(np.sort(inverted_profiles[initial_time:final_time+1,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(r),1000*max(r),new_timesteps[final_time],new_timesteps[initial_time]],aspect=50)
		plt.title('Line '+str(iR+4))
		plt.xlabel('radius [mm]')
		plt.ylabel('time [ms]')
		plt.colorbar().set_label('Emissivity [W m^-3 sr^-1]')
		plt.savefig('%s/abel_inverted_profile%s_short.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		max_value = np.median(np.max(inverted_profiles[:6,iR], axis=1))
		plt.imshow(inverted_profiles[:,iR],'rainbow',vmin=0,vmax=max_value,extent=[1000*min(r),1000*max(r),last_time,first_time],aspect=20)
		plt.title('Line '+str(iR+4))
		plt.xlabel('radius [mm]')
		plt.ylabel('time [ms]')
		plt.colorbar().set_label('Emissivity [W m^-3 sr^-1]')
		plt.savefig('%s/abel_inverted_profile_SS_%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(all_fits[:,:,iR],'rainbow',vmin=0,vmax=np.nanmax(np.mean(np.sort(all_fits[:,:,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(xx),1000*max(xx),last_time,first_time],aspect=20)
		plt.title('Line '+str(iR+4)+'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centres[:,iR],axis=(0)), decimals=1))+'mm) \n used black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
		plt.xlabel('LOS location [mm]')
		plt.ylabel('time [ms]')
		plt.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), 'k', linewidth=0.4,label='used centre')
		plt.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), '--r', linewidth=0.4, label='local centre')
		plt.colorbar().set_label('Line integrated brightness [W m^-2 sr^-1]')
		plt.savefig('%s/line_integrted_profile%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(all_fits_fitted[:,:,iR],'rainbow',extent=[1000*min(xx),1000*max(xx),last_time,first_time],aspect=20)
		plt.title('Fitting of line '+str(iR+4)+'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centres[:,iR],axis=(0)), decimals=1))+'mm) \n used black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
		plt.xlabel('LOS location [mm]')
		plt.ylabel('time [ms]')
		plt.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), 'k', linewidth=0.4,label='used centre')
		plt.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), '--r', linewidth=0.4, label='local centre')
		plt.colorbar().set_label('Line integrated brightness [W m^-2 sr^-1]')
		plt.savefig('%s/line_integrted_profile_fitting%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(all_fits_residuals[:,:,iR],'rainbow',extent=[1000*min(xx),1000*max(xx),last_time,first_time],aspect=20)
		plt.title('Fitting residuals of line '+str(iR+4)+'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centres[:,iR],axis=(0)), decimals=1))+'mm) \n used black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
		plt.xlabel('LOS location [mm]')
		plt.ylabel('time [ms]')
		plt.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), 'k', linewidth=0.4,label='used centre')
		plt.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), '--r', linewidth=0.4, label='local centre')
		plt.colorbar().set_label('Deviation from fit [au]')# over maximum fitted value [au]')
		plt.savefig('%s/line_integrted_profile_fitting_residuals%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		max_value = np.median(np.max(all_fits[:6,:,iR], axis=1))
		plt.imshow(all_fits[:,:,iR],'rainbow',vmin=0,vmax=max_value,extent=[1000*min(xx),1000*max(xx),last_time,first_time],aspect=20)
		plt.title('Line '+str(iR+4)+'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm) \n used black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
		plt.xlabel('LOS location [mm]')
		plt.ylabel('time [ms]')
		plt.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), 'k', linewidth=0.4,label='used centre')
		plt.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), '--r', linewidth=0.4, label='local centre')
		plt.colorbar().set_label('Line integrated brightness [W m^-2 sr^-1]')
		plt.savefig('%s/line_integrted_profile_SS_%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	np.save('%s/profileFits.npy' % (path), fitCoefs)
	np.save('%s/profile_centre.npy' % (path), profile_centre)
	np.save('%s/inverted_profiles.npy' % (path), inverted_profiles)

	# 	# invProf = AInvgauss3(r, *fit[0])
	# 		plt.figure()
	# 		plt.plot(xx * 1e3, yy)
	# 		plt.plot(xn * 1e3, gauss3_locked(xn, *fit[0]))
	# 		plt.xlabel('Lateral distance [mm]')
	# 		plt.ylabel('Radiance [a.u.]')
	# 		plt.pause(0.01)
	# 		plt.title('Setting %i, n = %s' % (iSet, iR))
	#
	# 		plt.savefig('%s%i_%s.png' % (pdir1, iSet, col), bbox_inches='tight')
	# 		plt.savefig('%s%s_%i.png' % (pdir1, col, iSet), bbox_inches='tight')
	#
	# 		plt.figure()
	# 		plt.plot(r * 1e3, invProf)
	# 		plt.xlabel('Radial distance [mm]')
	# 		plt.ylabel('Emission coefficient [a.u.]')
	# 		plt.savefig('%s%i_%s.png' % (pdir2, iSet, col), bbox_inches='tight')
	# 		plt.savefig('%s%s_%i.png' % (pdir2, col, iSet), bbox_inches='tight')
	#
	#
	# 	plt.figure()
	# 	for coef in fitCoefs[iSet]:
	# 		plt.plot(r * 1e3, AInvgauss3(r, *coef))
	# 	plt.xlabel('Radial distance [mm]')
	# 	plt.ylabel('Emission coefficient [W sr^{-1} m^{-3)}]')
	# 	plt.legend(leg)
	# 	plt.savefig('%sall_%i.png' % (pdir2, iSet), bbox_inches='tight')
	#
	# 	plt.figure()
	# 	for coef in fitCoefs[iSet]:
	# 		plt.semilogy(r * 1e3, AInvgauss3(r, *coef))
	# 	plt.xlabel('Radial distance [mm]')
	# 	plt.ylabel('Emission coefficient [W sr^{-1} m^{-3)}]')
	# 	plt.legend(leg)
	# 	plt.savefig('%sallLog_%i.png' % (pdir2, iSet), bbox_inches='tight')
	# 	plt.close('all')
	# np.save('./results/profileFits.npy', fitCoefs)

def doLateralfit_time_tependent_with_sigma(df_settings,all_fits,all_fits_sigma, merge_ID_target, new_timesteps,dx,xx,r,same_centre_every_line=False,same_centre_all_time=True,max_central_depression_amplitude=-1,min_central_depression_sigma=0.05,force_glogal_center=0):

	import os
	import pandas as pd
	import numpy as np
	from scipy.optimize import curve_fit
	import matplotlib.pyplot as plt
	import copy as cp
	from uncertainties import ufloat,unumpy,correlated_values
	from uncertainties.unumpy import exp,nominal_values,std_devs
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
	AInvgauss3 = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: AInvgauss(x, A1, sig1, 0) + AInvgauss(x, A2, sig2,0) + AInvgauss(x, A3,sig3,0)
	AInvgaussJac = lambda x, A, sig, x0: np.array([gauss(x, 1 / np.sqrt(np.pi) / sig, sig, 0),gauss(x, A, sig, 0) * (2 * x ** 2 - sig ** 2) / sig ** 4 / np.sqrt(np.pi), gauss(x, A, sig, 0) * 2 * x / sig ** 3 / np.sqrt(np.pi)])
	AInvgauss3Jac = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: [np.concatenate([a[:2], a[3:5], a[6:8], [sum(a[2::3])]]) for a in [np.concatenate([AInvgaussJac(x, A1, sig1, 0), AInvgaussJac(x, A2, sig2, 0), AInvgaussJac(x, A3, sig3, 0)])]][0]
	LRvecmul = lambda vec, mat: np.dot(vec, np.matmul(mat, vec))
	gauss3_2 = lambda x, A1, sig1, c_A2, c_sig2, c_A3, c_sig3,x0: gauss(x, A1, sig1, x0) + gauss(x,A1*c_A2,c_sig2*sig1,x0) + gauss(x, c_A3*c_sig3*(A1 +A1*c_A2/c_sig2), c_sig3*sig1, x0)
	AInvgauss3_locked_2 = lambda x, A1, sig1, c_A2, c_sig2, c_A3, c_sig3: AInvgauss(x, A1, sig1, 0) + AInvgauss(x, A1*c_A2,c_sig2*sig1,0) + AInvgauss(x, c_A3*c_sig3*(A1 +A1*c_A2/c_sig2), c_sig3*sig1, 0)
	AInvgauss3_locked_2_with_error_included = lambda x, A1, sig1, A2, c_sig2, c_A3, c_sig3: AInvgauss_with_error_included(x, A1, sig1, 0) + AInvgauss_with_error_included(x, A2,c_sig2*sig1,0) + AInvgauss_with_error_included(x, c_A3*c_sig3*(A1 +A2/c_sig2), c_sig3*sig1, 0)
	gauss_sigma = lambda x, A, gaus_sig, x0,A_sig,gaus_sig_sig,x0_sig:gauss(x, A, gaus_sig, x0) * np.sqrt( (A_sig/A)**2 + (2*(x-x0)*x0_sig/(gaus_sig**2))**2 + (2*((x-x0)**2)*gaus_sig_sig/(gaus_sig**3))**2 )
	AInvgauss_sigma_1 = lambda x, A1, sig1, A1_sigma, sig1_sigma: AInvgauss(x, A1, sig1, 0) * np.sqrt( (A1_sigma/A1)**2 + (1+4*(x**4)/(sig1**4))*(sig1_sigma/sig1)**2 )
	AInvgauss_sigma_2 = lambda x, sig1, A2, c_sig2, sig1_sigma, A2_sigma, c_sig2_sigma: AInvgauss(x, A2, c_sig2*sig1, 0) * np.sqrt( (A2_sigma/A2)**2 + (1+4*(x**4)/((c_sig2*sig1)**4))*((sig1_sigma/sig1)**2 + (c_sig2_sigma/c_sig2)**2) )
	AInvgauss_sigma_3 = lambda x, A1, sig1, A2, c_sig2, c_A3, c_sig3, A1_sigma, sig1_sigma, A2_sigma, c_sig2_sigma, c_A3_sigma, c_sig3_sigma: AInvgauss(x, c_A3*c_sig3*(A1 +A2/c_sig2), c_sig3*sig1, 0) * np.sqrt( (c_A3_sigma/c_A3)**2 + (A1_sigma**2 + (A2_sigma/c_sig2)**2 + (A2*c_sig2_sigma/c_sig2**2)**2)/((A1 +A2/c_sig2)**2) + 4*(x**4)/((c_sig3*sig1)**4)*(c_sig3_sigma/c_sig3)**2 + (1+4*(x**4)/((c_sig3*sig1)**4))*(sig1_sigma/sig1)**2 )
	AInvgauss3_locked_2_sigma = lambda x, A1, sig1, A2, c_sig2, c_A3, c_sig3, A1_sigma, sig1_sigma, A2_sigma, c_sig2_sigma, c_A3_sigma, c_sig3_sigma: np.sqrt( (AInvgauss_sigma_1(x, A1, sig1, A1_sigma, sig1_sigma))**2 + (AInvgauss_sigma_2(x, sig1, A2, c_sig2, sig1_sigma, A2_sigma, c_sig2_sigma))**2 + (AInvgauss_sigma_3(x, A1, sig1, A2, c_sig2, c_A3, c_sig3, A1_sigma, sig1_sigma, A2_sigma, c_sig2_sigma, c_A3_sigma, c_sig3_sigma ))**2 )
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
	# fitCoefs = np.zeros((len(all_fits), nLine, 8))
	leg = ['n = %i' % i for i in range(3, 13)]


	minimum_level = np.min(all_fits[all_fits!=0])
	if (same_centre_every_line==True and same_centre_all_time==True):
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



	if force_glogal_center!=0:
		# profile_centres = np.zeros((len(all_fits),nLine))
		profile_centre = np.ones((len(all_fits),nLine))*force_glogal_center

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
			yy[np.isnan(yy)]=0
			yy_sigma = all_fits_sigma[iSet, :, iR]
			yy_sigma[np.isnan(yy)]=np.nanmax(yy_sigma)
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
				xx_good = xx
				yy_good = yy
				yy_good_sigma = yy_sigma
			else:
				if borders[0]!=0:	# this should not be necessary but I cannot avoid a smear of composed_array when dealing with data missing in lower lines
					borders[0]+=5
				xx_good = xx[borders[0]:borders[-1]]
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
			bds = [[0, 0, 0, 1, max_central_depression_amplitude, min_central_depression_sigma,profile_centre[iSet,iR]-2*dx],[max(yy)*100, np.inf, 1, np.inf, 0, 1,profile_centre[iSet,iR]+2*dx]]
			# p0 = [ym / 5, 20e-3, ym * .8, 4e-1, -ym / 2, 0.9e-3]
			# p0 = [ym *0.9, 6e-3, ym * .05, 4e-1, -ym / 4, inner_dip_sig]
			p0 = [ym *0.9, 6*dx, .05, 300, -0.25, inner_dip_sig,profile_centre[iSet,iR]]
			x_scale = [ym,dx,1,1,1,1,profile_centre[iSet,iR]]
			# p0 = [ym / 5, 20e-3, ym * .8, 4e-3, -ym / 5, 1e-3,profile_centre[iSet]]
			# bds = [[0, 0, 0, 1, -1, 1e-10, -np.inf, -np.inf],[max(yy) * 100, np.inf, max(yy) * 100, np.inf, 0, 1, np.inf, np.inf]]
			# p0 = [ym * 0.9, 5e-3, ym * .05, 67, -0.25, inner_dip_sig, 0, 0]
			try:
				fit = curve_fit(gauss3_2, xx_good, yy_good, p0,sigma=yy_good_sigma, absolute_sigma=True, maxfev=1000000, xtol=1e-15, gtol=1e-15, bounds=np.array(bds),x_scale=x_scale,ftol=1e-15,loss='soft_l1')
				# fit = curve_fit(gauss3_locked_2_bias, xx_good, yy_good, p0, maxfev=100000, bounds=np.array(bds))
				# while (AInvgauss3_locked_2(0, *fit[0]) < 0 and inner_dip_sig < 2e-3):
				while (AInvgauss3_locked_2(0, *fit[0][:-1]) < 0 and inner_dip_sig < 0.3):
				# while (AInvgauss3_locked_2(0, *fit[0][:-2]) < 0 and inner_dip_sig < 0.3):
					# inner_dip_sig+=(0.9e-3)/10
					print('small peak sigma increased, time slice '+str(iSet)+', line '+str(iR))
					inner_dip_sig+=0.015
					# p0 = [ym * 0.9, 6e-3, ym * .05, 4e-1, -ym / 3, inner_dip_sig]
					p0 = [*fit[0][:-1],inner_dip_sig,profile_centre[iSet,iR]]
					# p0 = [*fit[0][:5], inner_dip_sig, 0, 0]
					fit = curve_fit(gauss3_2, xx_good, yy_good, p0,sigma=yy_good_sigma, absolute_sigma=True, maxfev=1000000, xtol=1e-15, gtol=1e-15, bounds=np.array(bds),x_scale=x_scale,ftol=1e-15,loss='soft_l1')
					# fit = curve_fit(gauss3_locked_2_bias, xx_good, yy_good, p0, maxfev=100000, bounds=np.array(bds))
				# if fit[0][-2]>0:
				# 	inner_dip_sig = 0.9e-3
				# 	p0 = [ym *0.9, 6e-3, ym * .05, 4e-1, -ym / 3, inner_dip_sig]
			except:
				print('fitting failed for time slice '+str(iSet)+', line '+str(iR))
				fit=[[0,5e-3,0, 67, -0.25,0.15,profile_centre[iSet,iR]],np.ones((7,7))]
				# fit = [[0, 5e-3, 0, 67, -0.25, 0.15, 0, 0]]
			# fit = curve_fit(gauss3_locked, xx_good, yy_good, p0, maxfev=100000, bounds=bds)
			if  (np.min(np.abs(new_timesteps[iSet] - np.array([-0.2,0.2,0.55,1])))<=0.06):
				im = ax[iR,0].errorbar(1000*xx_good, yy_good,yerr=yy_good_sigma,color='c',label='line integrated data')
				im = ax[iR,0].plot([1000*fit[0][-1],1000*fit[0][-1]],[np.min([yy_good,gauss3_2(xx, *fit[0])]),np.max([yy_good,gauss3_2(xx, *fit[0])])],'k--')
				im = ax[iR,0].plot(1000*xx, gauss3_2(xx, *fit[0]),'m',label='fit')
				ax[iR,0].set_ylabel('brightness [W m-2 sr-1]')
				ax[iR,0].grid()
				if iR==0:
					ax[iR,0].set_title('cyan=line int data, magenta=fit\nline n='+str(iR+4)+'->2')
				else:
					ax[iR,0].set_title('line n='+str(iR+4)+'->2')
				if iR==(np.shape(all_fits)[-1]-1):
					ax[iR,0].set_xlabel('x [mm]')
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
			fitCoefs[iSet, iR] = fit[0]
			# inverted_profiles[iSet, iR] = AInvgauss3_locked(r, *fit[0])
			# all_fits_fitted[iSet, :, iR] = gauss3_locked(xx,*fit[0])
			# inverted_profiles[iSet, iR] = AInvgauss3_locked_2(r, *fit[0])
			# inverted_profiles_sigma[iSet, iR] = AInvgauss3_locked_2_sigma(r, *fit[0],*np.sqrt(np.diag(fit[1])))
			profile_centre[iSet,iR]=fit[0][-1]
			# temp = AInvgauss3_locked_2_with_error_included(r, unumpy.uarray(fit[0][0]*np.ones_like(r),(fit[1][0,0]**0.5)*np.ones_like(r)), unumpy.uarray(fit[0][1]*np.ones_like(r),(fit[1][1,1]**0.5)*np.ones_like(r)), unumpy.uarray(fit[0][2]*np.ones_like(r),(fit[1][2,2]**0.5)*np.ones_like(r)), unumpy.uarray(fit[0][3]*np.ones_like(r),(fit[1][3,3]**0.5)*np.ones_like(r)), unumpy.uarray(fit[0][4]*np.ones_like(r),(fit[1][4,4]**0.5)*np.ones_like(r)), unumpy.uarray(fit[0][5]*np.ones_like(r),(fit[1][5,5]**0.5)*np.ones_like(r)) )
			temp = AInvgauss3_locked_2_with_error_included(r, *correlated_values(fit[0][:-1],fit[1][:-1,:-1]) )
			inverted_profiles[iSet, iR] = nominal_values(temp)
			inverted_profiles_sigma[iSet, iR] = std_devs(temp)
			all_fits_fitted[iSet, :, iR] = gauss3_2(xx,*fit[0])
			# inverted_profiles[iSet, iR] = AInvgauss3_locked_2(r, *fit[0][:-2])
			# all_fits_fitted[iSet, :, iR] = gauss3_locked_2_bias(xx,*fit[0])
			all_fits_residuals[iSet, :, iR] = (temp_all_fits - all_fits_fitted[iSet, :, iR])#/np.abs(np.nanmax(all_fits_fitted[iSet, :, iR]))
			if  (np.min(np.abs(new_timesteps[iSet] - np.array([-0.2,0.2,0.55,1])))<=0.06):
				im = ax[iR,1].errorbar(1000*r, nominal_values(temp),yerr=std_devs(temp),color='b',label='inverted data')
				im = ax[iR,1].plot(1000*r, std_devs(temp),'r',label='error')
				ax[iR,1].set_ylabel('emissivity [W m^-3 sr^-1]')
				ax[iR,1].yaxis.set_label_position('right')
				ax[iR,1].yaxis.tick_right()
				ax[iR,1].grid()
				if iR==0:
					ax[iR,1].set_title('blue=inverted prof, red=error\nfit='+str(''.join(['%.3g+/-%.3g, ' %(_,__) for _,__ in zip(fit[0],np.sqrt(np.diag(fit[1])))])[:-2]))
				else:
					ax[iR,1].set_title('fit='+str(''.join(['%.3g+/-%.3g, ' %(_,__) for _,__ in zip(fit[0],np.sqrt(np.diag(fit[1])))])[:-2]))
				if iR==(np.shape(all_fits)[-1]-1):
					ax[iR,1].set_xlabel('r [mm]')
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
		plt.figure(figsize=(10, 20))
		plt.imshow(inverted_profiles[:,iR],'rainbow',vmax=np.nanmax(np.mean(np.sort(inverted_profiles[:,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(r),1000*max(r),last_time,first_time],aspect=50)
		plt.title('Line '+str(iR+4))
		plt.xlabel('radius [mm]')
		plt.ylabel('time [ms]')
		plt.colorbar().set_label('Emissivity [W m^-3 sr^-1]')
		plt.savefig('%s/abel_inverted_profile%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	initial_time = int((np.abs(new_timesteps)).argmin())
	final_time = int((np.abs(new_timesteps -2)).argmin())
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(inverted_profiles_sigma[:,iR],'rainbow',vmax=np.nanmax(np.mean(np.sort(inverted_profiles_sigma[:,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(r),1000*max(r),last_time,first_time],aspect=50)
		plt.title('Line '+str(iR+4))
		plt.xlabel('radius [mm]')
		plt.ylabel('time [ms]')
		plt.colorbar().set_label('Emissivity [W m^-3 sr^-1]')
		plt.savefig('%s/abel_inverted_profile_sigma%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	initial_time = int((np.abs(new_timesteps)).argmin())
	final_time = int((np.abs(new_timesteps -2)).argmin())
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(inverted_profiles[initial_time:final_time+1,iR],'rainbow',vmin=0,vmax=np.nanmax(np.mean(np.sort(inverted_profiles[initial_time:final_time+1,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(r),1000*max(r),new_timesteps[final_time],new_timesteps[initial_time]],aspect=50)
		plt.title('Line '+str(iR+4))
		plt.xlabel('radius [mm]')
		plt.ylabel('time [ms]')
		plt.colorbar().set_label('Emissivity [W m^-3 sr^-1]')
		plt.savefig('%s/abel_inverted_profile%s_short.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(inverted_profiles[initial_time:final_time+1,iR]/inverted_profiles_sigma[initial_time:final_time+1,iR],'rainbow',vmin=0,vmax=np.nanmax(np.mean(np.sort(inverted_profiles[initial_time:final_time+1,iR]/inverted_profiles_sigma[initial_time:final_time+1,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(r),1000*max(r),new_timesteps[final_time],new_timesteps[initial_time]],aspect=50)
		plt.title('Line '+str(iR+4))
		plt.xlabel('radius [mm]')
		plt.ylabel('time [ms]')
		plt.colorbar().set_label('Signal/Noise [au]')
		plt.savefig('%s/abel_inverted_profile_SNR%s_short.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(inverted_profiles_sigma[initial_time:final_time+1,iR],'rainbow',vmin=0,vmax=np.nanmax(np.mean(np.sort(inverted_profiles_sigma[initial_time:final_time+1,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(r),1000*max(r),new_timesteps[final_time],new_timesteps[initial_time]],aspect=50)
		plt.title('Line '+str(iR+4))
		plt.xlabel('radius [mm]')
		plt.ylabel('time [ms]')
		plt.colorbar().set_label('Emissivity [W m^-3 sr^-1]')
		plt.savefig('%s/abel_inverted_profile_sigma%s_short.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		max_value = np.median(np.max(inverted_profiles[:6,iR], axis=1))
		plt.imshow(inverted_profiles[:,iR],'rainbow',vmin=0,vmax=max_value,extent=[1000*min(r),1000*max(r),last_time,first_time],aspect=20)
		plt.title('Line '+str(iR+4))
		plt.xlabel('radius [mm]')
		plt.ylabel('time [ms]')
		plt.colorbar().set_label('Emissivity [W m^-3 sr^-1]')
		plt.savefig('%s/abel_inverted_profile_SS_%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(all_fits[:,:,iR],'rainbow',vmin=0,vmax=np.nanmax(np.mean(np.sort(all_fits[:,:,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(xx),1000*max(xx),last_time,first_time],aspect=20)
		plt.title('Line '+str(iR+4)+'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centres[:,iR],axis=(0)), decimals=1))+'mm) \n used black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
		plt.xlabel('LOS location [mm]')
		plt.ylabel('time [ms]')
		plt.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), 'k', linewidth=0.4,label='used centre')
		plt.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), '--r', linewidth=0.4, label='local centre')
		plt.colorbar().set_label('Line integrated brightness [W m^-2 sr^-1]')
		plt.savefig('%s/line_integrted_profile%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(all_fits[:,:,iR]/all_fits_sigma[:,:,iR],'rainbow',vmin=0,vmax=np.nanmax(np.mean(np.sort(all_fits[:,:,iR]/all_fits_sigma[:,:,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(xx),1000*max(xx),last_time,first_time],aspect=20)
		plt.title('Line '+str(iR+4)+'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centres[:,iR],axis=(0)), decimals=1))+'mm) \n used black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
		plt.xlabel('LOS location [mm]')
		plt.ylabel('time [ms]')
		plt.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), 'k', linewidth=0.4,label='used centre')
		plt.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), '--r', linewidth=0.4, label='local centre')
		plt.colorbar().set_label('Line integrated brightness [W m^-2 sr^-1]')
		plt.savefig('%s/line_integrted_profile_SNR%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(all_fits_sigma[:,:,iR],'rainbow',vmin=0,vmax=np.nanmax(np.mean(np.sort(all_fits_sigma[:,:,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(xx),1000*max(xx),last_time,first_time],aspect=20)
		plt.title('Line '+str(iR+4)+'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centres[:,iR],axis=(0)), decimals=1))+'mm) \n used black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
		plt.xlabel('LOS location [mm]')
		plt.ylabel('time [ms]')
		plt.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), 'k', linewidth=0.4,label='used centre')
		plt.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), '--r', linewidth=0.4, label='local centre')
		plt.colorbar().set_label('Line integrated brightness sigma[W m^-2 sr^-1]')
		plt.savefig('%s/line_integrted_profile_sigma%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(all_fits_fitted[:,:,iR],'rainbow',extent=[1000*min(xx),1000*max(xx),last_time,first_time],aspect=20)
		plt.title('Fitting of line '+str(iR+4)+'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centres[:,iR],axis=(0)), decimals=1))+'mm) \n used black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
		plt.xlabel('LOS location [mm]')
		plt.ylabel('time [ms]')
		plt.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), 'k', linewidth=0.4,label='used centre')
		plt.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), '--r', linewidth=0.4, label='local centre')
		plt.colorbar().set_label('Line integrated brightness [W m^-2 sr^-1]')
		plt.savefig('%s/line_integrted_profile_fitting%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(all_fits_residuals[:,:,iR],'rainbow',extent=[1000*min(xx),1000*max(xx),last_time,first_time],aspect=20)
		plt.title('Fitting residuals of line '+str(iR+4)+'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centres[:,iR],axis=(0)), decimals=1))+'mm) \n used black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
		plt.xlabel('LOS location [mm]')
		plt.ylabel('time [ms]')
		plt.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), 'k', linewidth=0.4,label='used centre')
		plt.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), '--r', linewidth=0.4, label='local centre')
		plt.colorbar().set_label('Deviation from fit [au]')# over maximum fitted value [au]')
		plt.savefig('%s/line_integrted_profile_fitting_residuals%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		max_value = np.median(np.max(all_fits[:6,:,iR], axis=1))
		plt.imshow(all_fits[:,:,iR],'rainbow',vmin=0,vmax=max_value,extent=[1000*min(xx),1000*max(xx),last_time,first_time],aspect=20)
		plt.title('Line '+str(iR+4)+'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm) \n used black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
		plt.xlabel('LOS location [mm]')
		plt.ylabel('time [ms]')
		plt.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), 'k', linewidth=0.4,label='used centre')
		plt.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), '--r', linewidth=0.4, label='local centre')
		plt.colorbar().set_label('Line integrated brightness [W m^-2 sr^-1]')
		plt.savefig('%s/line_integrted_profile_SS_%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	np.save('%s/profileFits.npy' % (path), fitCoefs)
	np.save('%s/profile_centre.npy' % (path), profile_centre)
	np.save('%s/inverted_profiles.npy' % (path), inverted_profiles)
	np.save('%s/inverted_profiles_sigma.npy' % (path), inverted_profiles_sigma)

	# 	# invProf = AInvgauss3(r, *fit[0])
	# 		plt.figure()
	# 		plt.plot(xx * 1e3, yy)
	# 		plt.plot(xn * 1e3, gauss3_locked(xn, *fit[0]))
	# 		plt.xlabel('Lateral distance [mm]')
	# 		plt.ylabel('Radiance [a.u.]')
	# 		plt.pause(0.01)
	# 		plt.title('Setting %i, n = %s' % (iSet, iR))
	#
	# 		plt.savefig('%s%i_%s.png' % (pdir1, iSet, col), bbox_inches='tight')
	# 		plt.savefig('%s%s_%i.png' % (pdir1, col, iSet), bbox_inches='tight')
	#
	# 		plt.figure()
	# 		plt.plot(r * 1e3, invProf)
	# 		plt.xlabel('Radial distance [mm]')
	# 		plt.ylabel('Emission coefficient [a.u.]')
	# 		plt.savefig('%s%i_%s.png' % (pdir2, iSet, col), bbox_inches='tight')
	# 		plt.savefig('%s%s_%i.png' % (pdir2, col, iSet), bbox_inches='tight')
	#
	#
	# 	plt.figure()
	# 	for coef in fitCoefs[iSet]:
	# 		plt.plot(r * 1e3, AInvgauss3(r, *coef))
	# 	plt.xlabel('Radial distance [mm]')
	# 	plt.ylabel('Emission coefficient [W sr^{-1} m^{-3)}]')
	# 	plt.legend(leg)
	# 	plt.savefig('%sall_%i.png' % (pdir2, iSet), bbox_inches='tight')
	#
	# 	plt.figure()
	# 	for coef in fitCoefs[iSet]:
	# 		plt.semilogy(r * 1e3, AInvgauss3(r, *coef))
	# 	plt.xlabel('Radial distance [mm]')
	# 	plt.ylabel('Emission coefficient [W sr^{-1} m^{-3)}]')
	# 	plt.legend(leg)
	# 	plt.savefig('%sallLog_%i.png' % (pdir2, iSet), bbox_inches='tight')
	# 	plt.close('all')
	# np.save('./results/profileFits.npy', fitCoefs)



def doLateralfit_single(df_settings,all_fits_ss, merge_ID_target,dx,xx,r,same_centre_every_line=True,force_glogal_center=0):

	import os
	import pandas as pd
	import numpy as np
	from scipy.optimize import curve_fit
	import matplotlib.pyplot as plt
	import copy as cp


	# pdir1 = './plots/LateralGauss/'
	# pdir2 = './plots/RadialGauss/'
	# ddir = './results/Radiance/'
	# odir = './results/'

	# for cdir in [pdir1, pdir2, odir]:
	#	 if not os.path.isdir(cdir):
	#		 os.makedirs(cdir)

	nLine = np.shape(all_fits_ss)[-1]


	gauss = lambda x, A, sig, x0: A * np.exp(-((x - x0) / sig) ** 2)
	gauss_bias = lambda x, A, sig, x0,m,q: A * np.exp(-((x - x0) / sig) ** 2)+q+x*m
	# gauss2 = lambda x,A1,sig1,A2,sig2,x0: gauss(x,A1,sig1,x0) + gauss(x,A2,sig2,x0)
	gauss3 = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: gauss(x, A1, sig1, x0) + gauss(x, A2, sig2, x0) + gauss(x, A3,sig3,x0)
	gauss3_bias = lambda x, A1, sig1, A2, sig2, A3, sig3, x0,m,q: gauss(x, A1, sig1, x0) + gauss(x, A2, sig2, x0) + gauss(x, A3,sig3,x0)+q+x*m
	gaussJac = lambda x, A, sig, x0: np.array([gauss(x, 1, sig, x0), gauss(x, A, sig, x0) * 2 * (x - x0) ** 2 / sig ** 3,gauss(x, A, sig, x0) * 2 * (x - x0) / sig ** 2])
	gauss3Jac = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: [np.concatenate([a[:2], a[3:5], a[6:8], [sum(a[2::3])]]) for a in [np.concatenate([gaussJac(x, A1, sig1, x0), gaussJac(x, A2, sig2, x0), gaussJac(x, A3, sig3, x0)])]][0]
	AInvgauss = lambda x, A, sig, x0: gauss(x, A / sig / np.sqrt(np.pi), sig, 0)
	AInvgauss3 = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: AInvgauss(x, A1, sig1, 0) + AInvgauss(x, A2, sig2,0) + AInvgauss(x, A3,sig3,0)
	AInvgaussJac = lambda x, A, sig, x0: np.array([gauss(x, 1 / np.sqrt(np.pi) / sig, sig, 0),gauss(x, A, sig, 0) * (2 * x ** 2 - sig ** 2) / sig ** 4 / np.sqrt(np.pi), gauss(x, A, sig, 0) * 2 * x / sig ** 3 / np.sqrt(np.pi)])
	AInvgauss3Jac = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: [np.concatenate([a[:2], a[3:5], a[6:8], [sum(a[2::3])]]) for a in[np.concatenate([AInvgaussJac(x, A1, sig1, 0), AInvgaussJac(x, A2, sig2, 0), AInvgaussJac(x, A3, sig3, 0)])]][0]
	LRvecmul = lambda vec, mat: np.dot(vec, np.matmul(mat, vec))
	# bds2 = [[0,1e-4,0,1e-4,0],[np.inf,np.inf,np.inf,np.inf,np.inf]]
	bds3 = [[0, 1e-4, 0, 1e-4, -np.inf, 1e-4, 0], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, np.inf]]

	# dx = 18 / 40 * (50.5 / 27.4) / 1e3
	# xx = np.arange(40) * dx  # m
	xn = np.linspace(0, max(xx), 1000)
	# r = np.linspace(-max(xx) / 2, max(xx) / 2, 1000)
	# bds = [[-np.inf, dx, -np.inf, dx, -np.inf, dx, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, max(xx)]]
	bds = [[0, dx, -np.inf, dx, -np.inf, dx, 0], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, max(xx)]]
	bds4 = [[0, dx, 0], [np.inf, np.inf, max(xx)]]
	fitCoefs = np.zeros(( nLine, 8))
	leg = ['n = %i' % i for i in range(3, 13)]

	# if same_centre_every_line==True:
	# 	profile_centres = np.ones_like((nLine))*np.nan
	# 	profile_centres_score = np.zeros((nLine))
	# 	fit = [[0, 4e-3, 20 * dx], np.ones((3, 3)) * dx]
	# 	for iR in range(np.shape(all_fits_ss)[-1]):
	# 		# # for col in ['n5']:
	# 		yy = all_fits_ss[:,iR]
	# 		yy[np.isnan(yy)] = 0
	# 		# if not yy.isnull().any():
	# 		borders = np.linspace(0,len(yy),len(yy)+1)[np.diff(np.array([0,*yy,0])==0)].astype('int')
	# 		xx_good = xx[borders[0]:borders[-1]]
	# 		yy_good = yy[borders[0]:borders[-1]]
	# 		ym = np.nanmax(yy_good)
	# 		# p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 3, 1e-3, 20 * dx]
	# 		# p0 = [ym, 4e-3, fit[0][-1]]
	# 		p0 = [ym, 4e-3, 20 * dx,0,0]
	# 		bds4 = [[0, dx, 0,-1,-np.min(yy)], [np.inf, np.inf, max(xx),1,np.max(yy)]]
	# 		# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
	# 		try:
	# 			# fit = curve_fit(gauss, xx, yy, p0, maxfev=100000, bounds=np.array(bds)[:,4:])
	# 			# fit = curve_fit(gauss, xx_good, yy_good, p0, maxfev=100000, bounds=np.array(bds4))
	# 			fit = curve_fit(gauss_bias, xx_good, yy_good, p0, maxfev=100000, bounds=np.array(bds4))
	# 		except:
	# 			fit=[p0]
	# 		# if iSet==250:
	# 		# plt.figure()
	# 		# plt.plot(xx, yy)
	# 		# plt.plot(xx, gauss_bias(xx, *fit[0]))
	# 		# plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
	# 		# plt.title('fit: ' + str(fit[0])+'\n score: '+str(np.diag(fit[1])))
	# 		# # plt.title('fit: ' + str(fit[0]))
	# 		# plt.pause(0.01)
	# 		profile_centres[iR]=fit[0][2]
	# 		# profile_centres_score[iSet, iR] = fit[1][-1,-1]
	# 		profile_centres_score[iR] = np.sum(((yy_good - gauss_bias(xx_good, *fit[0]))/yy_good) ** 2)
	# 	# profile_centre = np.mean(profile_centres,axis=-1)
	# 	profile_centre = np.sum(profile_centres/(profile_centres_score**2), axis=-1)/np.sum(1/(profile_centres_score**2), axis=-1)
	# 	for iR in range(np.shape(all_fits_ss)[-1]):
	# 		# gauss3 = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: gauss(x, A1, sig1, x0) + gauss(x, A2, sig2, x0) + gauss(x,A3,sig3,x0)
	# 		yy = all_fits_ss[ :, iR]
	# 		yy[np.isnan(yy)] = 0
	# 		borders = np.linspace(0,len(yy),len(yy)+1)[np.diff(np.array([0,*yy,0])==0)].astype('int')
	# 		xx_good = xx[borders[0]:borders[-1]]
	# 		yy_good = yy[borders[0]:borders[-1]]
	# 		# if not yy.isnull().any():
	# 		ym = np.nanmax(yy_good)
	# 		p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 4, 0.9e-3, profile_centre]
	# 		bds = [[0, dx, -np.inf, dx, -np.inf, dx, profile_centre-2*dx], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, profile_centre+2*dx]]
	# 		# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
	# 		try:
	# 			fit = curve_fit(gauss3, xx_good, yy_good, p0, maxfev=100000, bounds=bds)
	# 		except:
	# 			fit=[p0]
	# 		profile_centres[ iR] = fit[0][-1]
	# 		if len(fit)==1:
	# 			profile_centres_score [iR] = 1
	# 		else:
	# 			# profile_centres_score[iSet, iR] = fit[1][-1, -1]
	# 			# profile_centres_score[iSet, iR] = fit[1][-1, -1] / ((np.nanmean(yy)/ym)**4)
	# 			profile_centres_score[ iR] = np.sum(((yy_good - gauss3(xx_good, *fit[0]))/yy_good) ** 2)
	# 		# profile_centres_score[iSet, iR] = np.sum((yy - gauss3(xx, *fit[0])) ** 2)
	# 		# plt.figure()
	# 		# plt.plot(xx, yy)
	# 		# plt.plot(xx, gauss3(xx, *fit[0]))
	# 		# plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
	# 		# plt.title('fit: ' + str(fit[0]) + '\n score: ' + str(np.diag(fit[1])))
	# 		# plt.pause(0.01)
	# 	# profile_centre = np.mean(profile_centres, axis=-1)
	# 	profile_centre = np.sum(profile_centres/(profile_centres_score**2), axis=-1)/np.sum(1/(profile_centres_score**2), axis=-1)
	# 	for iR in range(np.shape(all_fits_ss)[-1]):
	# 		# # for col in ['n5']:
	# 		yy = all_fits_ss[ :, iR]
	# 		yy[np.isnan(yy)] = 0
	# 		borders = np.linspace(0, len(yy), len(yy) + 1)[np.diff(np.array([0, *yy, 0]) == 0)].astype('int')
	# 		xx_good = xx[borders[0]:borders[-1]]
	# 		yy_good = yy[borders[0]:borders[-1]]
	# 		# if not yy.isnull().any():
	# 		ym = np.nanmax(yy_good)
	# 		p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 4, 0.9e-3, profile_centre]
	# 		bds = [[0, dx, -np.inf, dx, -np.inf, dx, profile_centre - 2 * dx],[np.inf, np.inf, np.inf, np.inf, 0, np.inf, profile_centre + 2 * dx]]
	# 		# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
	# 		try:
	# 			fit = curve_fit(gauss3, xx_good, yy_good, p0, maxfev=100000, bounds=bds)
	# 		except:
	# 			fit=[p0]
	# 		profile_centres[ iR] = fit[0][-1]
	# 		if len(fit)==1:
	# 			profile_centres_score[ iR] = 1
	# 		else:
	# 			# profile_centres_score[iSet, iR] = fit[1][-1, -1]
	# 			# profile_centres_score[iSet, iR] = fit[1][-1, -1] / ((np.nanmean(yy)/ym)**4)
	# 			profile_centres_score[ iR] = np.sum(((yy_good - gauss3(xx_good, *fit[0]))/yy_good) ** 2)
	# 		# profile_centres_score[iSet, iR] = np.sum((yy - gauss3(xx, *fit[0])) ** 2)
	# 		# if iSet in [35,70]:
	# 		# plt.figure()
	# 		# plt.plot(xx, yy)
	# 		# plt.plot(xx, gauss3(xx, *fit[0]))
	# 		# plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
	# 		# plt.title('fit: ' + str(fit[0])+'\n score: '+str(np.diag(fit[1])))
	# 		# plt.pause(0.01)
	# 	# profile_centre = np.mean(profile_centres, axis=-1)
	# 	profile_centre = np.sum(profile_centres/(profile_centres_score**2), axis=-1)/np.sum(1/(profile_centres_score**2), axis=-1)
	# 	profile_centre = np.ones_like(profile_centres)*np.nanmedian(profile_centre)
	# 	# profile_centre = np.convolve(profile_centre, np.ones((len(profile_centre)//5)) / (len(profile_centre)//5), mode='valid')
	#
	# else:
	profile_centres = np.zeros((nLine))
	profile_centres_score = np.zeros((nLine))
	# fit = [[0, 0, 20 * dx]]
	fit = [[0, 4e-3, 20 * dx],np.ones((3,3))*dx]
	for iR in range(np.shape(all_fits_ss)[-1]):
		# # for col in ['n5']:
		yy = all_fits_ss[:, iR]
		yy[np.isnan(yy)] = 0
		# if not yy.isnull().any():
		borders = np.linspace(0, len(yy), len(yy) + 1)[np.diff(np.array([0, *yy, 0]) == 0)].astype('int')
		xx_good = xx[borders[0]:borders[-1]]
		yy_good = yy[borders[0]:borders[-1]]
		ym = np.nanmax(yy_good)
		# p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 3, 1e-3, 20 * dx]
		# p0 = [ym, 4e-3, fit[0][-1]]
		# p0 = [ym, 4e-3, 20 * dx, 0, 0]
		# bds4 = [[0, dx, 0, -1, -np.min(yy)], [np.inf, np.inf, max(xx), 1, np.max(yy)]]
		if (fit[1][2, 2]) ** 0.5 < dx / 2:
			p0 = [ym, fit[0][1], fit[0][2],0,0]
		else:
			p0 = [ym, 5 * dx, 20 * dx,0,0]
		if iR > 0:
			bds4 = [[0, dx, min(profile_centres[iR - 1] - 3 * dx,p0[2]),-np.inf,-np.inf],[max(yy_good) * 10, max(xx_good), max(profile_centres[iR - 1] + 3 * dx,p0[2]),np.inf,np.inf]]
		else:
			bds4 = [[0, dx, 10 * dx,-np.inf,-np.inf], [max(yy_good) * 10, max(xx_good), 30 * dx,np.inf,np.inf]]

		# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
		try:
			# fit = curve_fit(gauss, xx, yy, p0, maxfev=100000, bounds=np.array(bds)[:,4:])
			# fit = curve_fit(gauss, xx_good, yy_good, p0, maxfev=100000, bounds=np.array(bds4))
			fit = curve_fit(gauss_bias, xx_good, yy_good, p0, maxfev=100000, bounds=np.array(bds4))
		except:
			print('preliminary SS fit failed for line '+str(iR))
			fit = [p0,np.ones((3,3))*np.inf]
		# if iSet==250:
		# plt.figure()
		# plt.plot(xx, yy)
		# plt.plot(xx, gauss_bias(xx, *fit[0]))
		# plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
		# plt.title('fit: ' + str(fit[0])+'\n score: '+str(np.diag(fit[1])))
		# # plt.title('fit: ' + str(fit[0]))
		# plt.pause(0.01)
		profile_centres[iR] = fit[0][2]
		profile_centres_score[iR] = fit[1][-1,-1]
		# profile_centres_score[iR] = np.sum(((yy_good - gauss_bias(xx_good, *fit[0])) / yy_good) ** 2)
	# profile_centre = np.mean(profile_centres,axis=-1)
	profile_centre = np.sum(profile_centres / (profile_centres_score ** 2), axis=-1) / np.sum(1 / (profile_centres_score ** 2), axis=-1)
	for iR in range(np.shape(all_fits_ss)[-1]):
		# gauss3 = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: gauss(x, A1, sig1, x0) + gauss(x, A2, sig2, x0) + gauss(x,A3,sig3,x0)
		yy = all_fits_ss[:, iR]
		yy[np.isnan(yy)] = 0
		borders = np.linspace(0, len(yy), len(yy) + 1)[np.diff(np.array([0, *yy, 0]) == 0)].astype('int')
		xx_good = xx[borders[0]:borders[-1]]
		yy_good = yy[borders[0]:borders[-1]]
		# if not yy.isnull().any():
		ym = np.nanmax(yy_good)
		p0 = [ym / 6, 20*dx, ym * .8, 5*dx, -ym / 4, 1.1*dx, profile_centre,0,0]
		bds = [[0, dx, -np.inf, dx, -np.inf, dx, profile_centre - 2 * dx,-np.inf,-np.inf],
			   [np.inf, np.inf, np.inf, np.inf, 0, np.inf, profile_centre + 2 * dx,np.inf,np.inf]]
		# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
		try:
			fit = curve_fit(gauss3_bias, xx_good, yy_good, p0, maxfev=100000, bounds=bds)
		except:
			print('preliminary SS fit failed for line '+str(iR))
			fit = [p0]
		profile_centres[iR] = fit[0][6]
		if len(fit) == 1:
			profile_centres_score[iR] = 1
		else:
			# profile_centres_score[iSet, iR] = fit[1][-1, -1]
			# profile_centres_score[iSet, iR] = fit[1][-1, -1] / ((np.nanmean(yy)/ym)**4)
			profile_centres_score[iR] = np.sum(((yy_good - gauss3_bias(xx_good, *fit[0])) / yy_good) ** 2)
	# profile_centres_score[iSet, iR] = np.sum((yy - gauss3(xx, *fit[0])) ** 2)
	# plt.figure()
	# plt.plot(xx, yy)
	# plt.plot(xx, gauss3(xx, *fit[0]))
	# plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
	# plt.title('fit: ' + str(fit[0]) + '\n score: ' + str(np.diag(fit[1])))
	# plt.pause(0.01)
	# profile_centre = np.mean(profile_centres, axis=-1)
	profile_centre = np.sum(profile_centres / (profile_centres_score ** 2), axis=-1) / np.sum(1 / (profile_centres_score ** 2), axis=-1)
	profile_centres = np.ones_like(profile_centres) * np.nanmedian(profile_centre)

	if False:	# for some reason this doesn't work
		for iR in range(np.shape(all_fits_ss)[-1]):
			# # for col in ['n5']:
			yy = all_fits_ss[:, iR]
			yy[np.isnan(yy)] = 0
			borders = np.linspace(0, len(yy), len(yy) + 1)[np.diff(np.array([0, *yy, 0]) == 0)].astype('int')
			xx_good = xx[borders[0]:borders[-1]]
			yy_good = yy[borders[0]:borders[-1]]
			# if not yy.isnull().any():
			ym = np.nanmax(yy_good)
			p0 = [ym / 6, 20*dx, ym * .8, 5*dx, -ym / 4, 1.1*dx, profile_centres[iR],0,0]
			bds = [[0, dx, 10*dx, dx, -np.inf, 0.5*dx, profile_centres[iR] - 2 * dx,-np.inf,-np.inf],
				   [np.inf, np.inf, np.inf, 20*dx, 0, 5*dx, profile_centres[iR] + 2 * dx,np.inf,np.min(yy_good)]]
			# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
			try:
				fit = curve_fit(gauss3_bias, xx_good, yy_good, p0, maxfev=100000, bounds=bds)
			except:
				print('SS fit failed for line '+str(iR))
				fit = [p0,[1]]
			profile_centres[iR] = fit[0][6]
			if len(fit) == 1:
				profile_centres_score[iR] = 1
			else:
				# profile_centres_score[iSet, iR] = fit[1][-1, -1]
				# profile_centres_score[iSet, iR] = fit[1][-1, -1] / ((np.nanmean(yy)/ym)**4)
				profile_centres_score[iR] = np.sum(((yy_good - gauss3_bias(xx_good, *fit[0])) / yy_good) ** 2)
			# profile_centres_score[iSet, iR] = np.sum((yy - gauss3(xx, *fit[0])) ** 2)
			# if iSet in [35,70]:
			plt.figure()
			plt.plot(xx_good, yy_good)
			plt.plot(xx_good, gauss3_bias(xx_good, *fit[0]))
			# plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
			plt.title('iR '+str(iR)+' fit: ' + str(fit[0])+'\n score: '+str(np.diag(fit[1])))
			plt.pause(0.01)
			# profile_centre = np.mean(profile_centres, axis=-1)
			# profile_centre = np.sum(profile_centres / (profile_centres_score ** 2), axis=-1) / np.sum(1 / (profile_centres_score ** 2), axis=-1)

	if force_glogal_center!=0:
		profile_centre = np.ones_like(profile_centres)*force_glogal_center
	elif same_centre_every_line == True:
		profile_centre = np.sum(profile_centres/(profile_centres_score**2), axis=-1)/np.sum(1/(profile_centres_score**2), axis=-1)
		profile_centre = np.ones_like(profile_centres)*np.nanmedian(profile_centre)
	else:
		profile_centre = cp.deepcopy(profile_centres)
	# profile_centre = np.convolve(profile_centre, np.ones((len(profile_centre)//5)) / (len(profile_centre)//5), mode='valid')



	all_fits_fitted = np.zeros_like(all_fits_ss)
	all_fits_residuals = np.zeros_like(all_fits_ss)
	inverted_profiles = np.zeros((nLine,len(r)))

	# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
	for iR in range(np.shape(all_fits_ss)[-1]):
		# gauss3_locked = lambda x, A1, sig1, A2, sig2, A3, sig3: gauss(x, A1, sig1, profile_centre[iR]) + gauss(x, A2, sig2,profile_centre[iR]) + gauss(x, A3, sig3, profile_centre[iR])
		# AInvgauss3_locked = lambda x, A1, sig1, A2, sig2, A3, sig3: AInvgauss(x, A1, sig1, 0) + AInvgauss(x, A2, sig2,0) + AInvgauss(x, A3, sig3, 0)
		gauss3_locked_2_bias = lambda x, A1, sig1, A2, c_sig2, c_A3, c_sig3, m, q: gauss(x, A1, sig1,profile_centre[iR]) + gauss(x, A2,c_sig2 * sig1,profile_centre[iR]) + gauss(x, c_A3*c_sig3*(A1 +A2/c_sig2), c_sig3 * sig1, profile_centre[iR]) + q + x * m
		AInvgauss3_locked_2 = lambda x, A1, sig1, A2, c_sig2, c_A3, c_sig3: AInvgauss(x, A1, sig1, 0) + AInvgauss(x, A2,c_sig2 * sig1,0) + AInvgauss(x, c_A3*c_sig3*(A1 +A2/c_sig2), c_sig3 * sig1, 0)
		yy = all_fits_ss[:, iR]
		for iCorr, value in enumerate(yy):
			if np.isnan(value):
				yy[iCorr] = 0

		temp_all_fits = cp.deepcopy(yy)
		if np.max(yy)>100*np.sort(yy)[-2]:
			pos1 = np.max([yy.argmax()-1,0])
			pos2 = np.min([yy.argmax()+1,len(yy)-1])
			yy[yy.argmax()]=np.mean(yy[[pos1,pos2]])

		borders = np.linspace(0, len(yy), len(yy) + 1)[np.diff(np.array([0, *yy, 0]) == 0)].astype('int')
		xx_good = xx[borders[0]:borders[-1]]
		yy_good = yy[borders[0]:borders[-1]]

		ym = np.nanmax(yy_good)
		# inner_dip_sig = 0.
		inner_dip_sig = 0.15
		# bds = [[0, -np.inf, 0, -np.inf, -max(yy), -np.inf],[max(yy)*100, np.inf, max(yy)*100, 0, np.inf]]
		# bds = [[0, dx, -np.inf, dx, -np.inf, dx, 0], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, max(xx)]]
		bds = [[0, 0, 0, 1, -1, 1e-10,-np.inf,-np.inf],[max(yy)*100, np.inf, max(yy)*100, np.inf, 0, 1,np.inf,np.inf]]
		# p0 = [ym / 5, 20e-3, ym * .8, 4e-1, -ym / 2, 0.9e-3]
		# p0 = [ym *0.9, 6e-3, ym * .05, 4e-1, -ym / 4, inner_dip_sig]
		p0 = [ym *0.9, 5e-3, ym * .05, 67, -0.25, inner_dip_sig,0,0]
		# p0 = [ym / 5, 20e-3, ym * .8, 4e-3, -ym / 5, 1e-3,profile_centre[iSet]]
		try:
			fit = curve_fit(gauss3_locked_2_bias, xx_good, yy_good, p0, maxfev=100000, bounds=np.array(bds))
			while (AInvgauss3_locked_2(0, *fit[0][:-2]) < 0 and inner_dip_sig < 0.3):
				# inner_dip_sig+=(0.9e-3)/10
				inner_dip_sig += 0.015
				# p0 = [ym * 0.9, 6e-3, ym * .05, 4e-1, -ym / 3, inner_dip_sig]
				p0 = [*fit[0][:5], inner_dip_sig,0,0]
				fit = curve_fit(gauss3_locked_2_bias, xx_good, yy_good, p0, maxfev=100000, bounds=np.array(bds))
			# if fit[0][-2]>0:
			# 	inner_dip_sig = 0.9e-3
			# 	p0 = [ym *0.9, 6e-3, ym * .05, 4e-1, -ym / 3, inner_dip_sig]
		except:
			print('fitting failed for line '+str(iR))
			fit = [[0,5e-3,0, 67, -0.25,0.15,0,0]]
		# fit = curve_fit(gauss3_locked, xx_good, yy_good, p0, maxfev=100000, bounds=bds)
		# plt.figure()
		# plt.plot(xx, yy)
		# plt.plot(xx, gauss3_locked(xx, *[*fit[0][:2],0,0,0,0]))
		# plt.plot(xx, gauss3_locked(xx, *[*fit[0][:4],0,0]))
		# plt.plot(xx, gauss3_locked(xx, *fit[0]))
		# plt.plot(xx, gauss3(xx, *fit[0]))
		# plt.pause(0.01)
		fitCoefs[ iR] = fit[0]
		inverted_profiles[ iR] = AInvgauss3_locked_2(r, *fit[0][:-2])
		all_fits_fitted[ :, iR] = gauss3_locked_2_bias(xx,*fit[0])
		all_fits_residuals[ :, iR] = (temp_all_fits - all_fits_fitted[ :, iR])
		# plt.figure()
		# plt.plot(r[::20], AInvgauss3_locked(r[::20], *fit[0]))
		# plt.pause(0.01)

	path = '/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)
	if not os.path.exists(path):
		os.makedirs(path)

	color = ['b', 'r', 'm', 'y', 'g', 'c', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise']
	plt.figure(figsize=(20, 10))
	for iR in range(nLine):
		plt.plot(1000*xx,all_fits_ss[:,iR],color[iR],label='line '+str(iR+4),linewidth=0.7)
		plt.plot(1000*xx, all_fits_fitted[:,iR],color=color[iR],linestyle='--',linewidth=0.4)
		# plt.plot(1000*xx, np.abs(all_fits_residuals[:,iR]),color=color[iR],linestyle=':',linewidth=0.4)
		plt.plot(1000*xx, fitCoefs[ iR][-1]+xx*fitCoefs[ iR][-2],color=color[iR],linestyle=':',linewidth=0.4)
		plt.plot([1000*profile_centre[iR],1000*profile_centre[iR]],[np.min(all_fits_ss),np.max(all_fits_ss)],color=color[iR],linestyle='-.')
	plt.title("Steady state line integrated profiles\n'--'=fitting, ':'=bias\n mean centre "+str(np.around(1000*np.mean(profile_centre,axis=0), decimals=1))+'mm')
	plt.xlabel('LOS location [mm]')
	plt.ylabel('Line integrated brightness [W m^-2 sr^-1]')
	plt.legend(loc='best')
	plt.semilogy()
	plt.grid(b=True, which='both')
	plt.ylim(np.min(all_fits_ss)*0.9,np.max(all_fits_ss)*1.1)
	plt.savefig('%s/SS_line_profile.eps' % (path), bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	for iR in range(nLine):
		plt.plot(1000*r,inverted_profiles[iR],color=color[iR],label='line '+str(iR+4))
	plt.title('Steady state line profiles \n mean centre '+str(np.around(1000*np.mean(profile_centre,axis=0), decimals=1))+'mm')
	plt.xlabel('radius [mm]')
	plt.ylabel('Emissivity [W m^-3 sr^-1]')
	plt.legend(loc='best')
	plt.savefig('%s/SS_abel_inverted_line_profile.eps' % (path), bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	for iR in range(nLine-1):
		plt.plot(1000*r,inverted_profiles[iR+1]/inverted_profiles[iR],color=color[iR],label='line ratio '+str(iR+4+1)+'/'+str(iR+4))
	plt.title('Steady state line ratio profiles \n mean centre '+str(np.around(1000*np.mean(profile_centre,axis=0), decimals=1))+'mm')
	plt.xlabel('radius [mm]')
	plt.ylabel('Line ratio [au]')
	plt.legend(loc='best')
	# plt.ylim(bottom=0)
	plt.semilogy()
	plt.grid(b=True, which='both')
	plt.savefig('%s/SS_abel_inverted_line_ratio_profile.eps' % (path), bbox_inches='tight')
	plt.close()
	np.save('%s/SS_profileFits.npy' % (path), fitCoefs)
	np.save('%s/SS_inverted_profiles.npy' % (path), inverted_profiles)

	if force_glogal_center==0:
		return profile_centre

def doLateralfit_single_with_sigma(df_settings,all_fits_ss,all_fits_ss_sigma, merge_ID_target,dx,xx,r,number_of_radial_divisions=21,same_centre_every_line=True,force_glogal_center=0,max_central_depression_amplitude=-1,min_central_depression_sigma=0.1):

	import os
	import pandas as pd
	import numpy as np
	from scipy.optimize import curve_fit
	import matplotlib.pyplot as plt
	import copy as cp
	from uncertainties import ufloat,unumpy,correlated_values
	from uncertainties.unumpy import exp,nominal_values,std_devs,erf


	# pdir1 = './plots/LateralGauss/'
	# pdir2 = './plots/RadialGauss/'
	# ddir = './results/Radiance/'
	# odir = './results/'

	# for cdir in [pdir1, pdir2, odir]:
	#	 if not os.path.isdir(cdir):
	#		 os.makedirs(cdir)

	nLine = np.shape(all_fits_ss)[-1]


	gauss = lambda x, A, sig, x0: A * np.exp(-((x - x0) / sig) ** 2)
	gauss_bias = lambda x, A, sig, x0,m,q: A * np.exp(-((x - x0) / sig) ** 2)+q+x*m
	# gauss2 = lambda x,A1,sig1,A2,sig2,x0: gauss(x,A1,sig1,x0) + gauss(x,A2,sig2,x0)
	gauss3 = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: gauss(x, A1, sig1, x0) + gauss(x, A2, sig2, x0) + gauss(x, A3,sig3,x0)
	gauss3_bias = lambda x, A1, sig1, A2, sig2, A3, sig3, x0,m,q: gauss(x, A1, sig1, x0) + gauss(x, A2, sig2, x0) + gauss(x, A3,sig3,x0)+q+x*m
	gaussJac = lambda x, A, sig, x0: np.array([gauss(x, 1, sig, x0), gauss(x, A, sig, x0) * 2 * (x - x0) ** 2 / sig ** 3,gauss(x, A, sig, x0) * 2 * (x - x0) / sig ** 2])
	gauss3Jac = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: [np.concatenate([a[:2], a[3:5], a[6:8], [sum(a[2::3])]]) for a in [np.concatenate([gaussJac(x, A1, sig1, x0), gaussJac(x, A2, sig2, x0), gaussJac(x, A3, sig3, x0)])]][0]
	AInvgauss = lambda x, A, sig, x0: gauss(x, A / sig / np.sqrt(np.pi), sig, 0)
	AInvgauss3 = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: AInvgauss(x, A1, sig1, 0) + AInvgauss(x, A2, sig2,0) + AInvgauss(x, A3,sig3,0)
	AInvgaussJac = lambda x, A, sig, x0: np.array([gauss(x, 1 / np.sqrt(np.pi) / sig, sig, 0),gauss(x, A, sig, 0) * (2 * x ** 2 - sig ** 2) / sig ** 4 / np.sqrt(np.pi), gauss(x, A, sig, 0) * 2 * x / sig ** 3 / np.sqrt(np.pi)])
	AInvgauss3Jac = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: [np.concatenate([a[:2], a[3:5], a[6:8], [sum(a[2::3])]]) for a in[np.concatenate([AInvgaussJac(x, A1, sig1, 0), AInvgaussJac(x, A2, sig2, 0), AInvgaussJac(x, A3, sig3, 0)])]][0]
	LRvecmul = lambda vec, mat: np.dot(vec, np.matmul(mat, vec))
	gauss_with_error_included = lambda x, A, sig, x0: A * exp(-((x - x0) / sig) ** 2)
	AInvgauss_with_error_included = lambda x, A, sig, x0: gauss_with_error_included(x, A / sig / np.sqrt(np.pi), sig, 0)
	AInvgauss3_locked_2_with_error_included = lambda x, A1, sig1, c_A2, c_sig2, c_A3, c_sig3: AInvgauss_with_error_included(x, A1, sig1, 0) + AInvgauss_with_error_included(x, A1*c_A2,c_sig2*sig1,0) + AInvgauss_with_error_included(x, c_A3*c_sig3*(A1 +A1*c_A2/c_sig2), c_sig3*sig1, 0)
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
	fitCoefs = np.zeros(( nLine, 8))
	leg = ['n = %i' % i for i in range(3, 13)]

	profile_centres = np.zeros((nLine))
	profile_centres_score = np.zeros((nLine))
	# fit = [[0, 0, 20 * dx]]
	fit = [[0, 4e-3, 20 * dx],np.ones((3,3))*dx]
	for iR in range(np.shape(all_fits_ss)[-1]):
		# # for col in ['n5']:
		yy = all_fits_ss[:, iR]
		yy[np.isnan(yy)] = 0
		# if not yy.isnull().any():
		borders = np.linspace(0, len(yy), len(yy) + 1)[np.diff(np.array([0, *yy, 0]) == 0)].astype('int')
		xx_good = xx[borders[0]:borders[-1]]
		yy_good = yy[borders[0]:borders[-1]]
		yy_sigma = all_fits_ss_sigma[:, iR]
		yy_good_sigma = yy_sigma[borders[0]:borders[-1]]
		ym = np.nanmax(yy_good)
		# p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 3, 1e-3, 20 * dx]
		# p0 = [ym, 4e-3, fit[0][-1]]
		# p0 = [ym, 4e-3, 20 * dx, 0, 0]
		# bds4 = [[0, dx, 0, -1, -np.min(yy)], [np.inf, np.inf, max(xx), 1, np.max(yy)]]
		if (fit[1][2, 2]) ** 0.5 < dx / 2:
			p0 = [ym, fit[0][1], fit[0][2],0,0]
		else:
			p0 = [ym, 5 * dx, 20 * dx,0,0]
		if iR > 0:
			bds4 = [[0, dx, min(profile_centres[iR - 1] - 3 * dx,p0[2]),-np.inf,-np.inf],[max(yy_good) * 10, max(xx_good), max(profile_centres[iR - 1] + 3 * dx,p0[2]),np.inf,np.inf]]
		else:
			bds4 = [[0, dx, 10 * dx,-np.inf,-np.inf], [max(yy_good) * 10, max(xx_good), 30 * dx,np.inf,np.inf]]

		# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
		try:
			# fit = curve_fit(gauss, xx, yy, p0, maxfev=100000, bounds=np.array(bds)[:,4:])
			# fit = curve_fit(gauss, xx_good, yy_good, p0, maxfev=100000, bounds=np.array(bds4))
			fit = curve_fit(gauss_bias, xx_good, yy_good, p0,sigma=yy_good_sigma, absolute_sigma=True, maxfev=100000, bounds=np.array(bds4))
		except:
			print('preliminary SS fit failed for line '+str(iR))
			fit = [p0,np.ones((3,3))*np.inf]
		# if iSet==250:
		# plt.figure()
		# plt.plot(xx, yy)
		# plt.plot(xx, gauss_bias(xx, *fit[0]))
		# plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
		# plt.title('fit: ' + str(fit[0])+'\n score: '+str(np.diag(fit[1])))
		# # plt.title('fit: ' + str(fit[0]))
		# plt.pause(0.01)
		profile_centres[iR] = fit[0][2]
		profile_centres_score[iR] = fit[1][-1,-1]
		# profile_centres_score[iR] = np.sum(((yy_good - gauss_bias(xx_good, *fit[0])) / yy_good) ** 2)
	# profile_centre = np.mean(profile_centres,axis=-1)
	profile_centre = np.sum(profile_centres / (profile_centres_score ** 2), axis=-1) / np.sum(1 / (profile_centres_score ** 2), axis=-1)
	for iR in range(np.shape(all_fits_ss)[-1]):
		# gauss3 = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: gauss(x, A1, sig1, x0) + gauss(x, A2, sig2, x0) + gauss(x,A3,sig3,x0)
		yy = all_fits_ss[:, iR]
		yy[np.isnan(yy)] = 0
		borders = np.linspace(0, len(yy), len(yy) + 1)[np.diff(np.array([0, *yy, 0]) == 0)].astype('int')
		xx_good = xx[borders[0]:borders[-1]]
		yy_good = yy[borders[0]:borders[-1]]
		yy_sigma = all_fits_ss_sigma[:, iR]
		yy_good_sigma = yy_sigma[borders[0]:borders[-1]]
		# if not yy.isnull().any():
		ym = np.nanmax(yy_good)
		p0 = [ym / 6, 20*dx, ym * .8, 5*dx, -ym / 4, 1.1*dx, profile_centre,0,0]
		bds = [[0, dx, -np.inf, dx, -np.inf, dx, profile_centre - 2 * dx,-np.inf,-np.inf],
			   [np.inf, np.inf, np.inf, np.inf, 0, np.inf, profile_centre + 2 * dx,np.inf,np.inf]]
		# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
		try:
			fit = curve_fit(gauss3_bias, xx_good, yy_good, p0,sigma=yy_good_sigma, absolute_sigma=True, maxfev=100000, bounds=bds)
		except:
			print('preliminary SS fit failed for line '+str(iR))
			fit = [p0]
		profile_centres[iR] = fit[0][6]
		if len(fit) == 1:
			profile_centres_score[iR] = 1
		else:
			# profile_centres_score[iSet, iR] = fit[1][-1, -1]
			# profile_centres_score[iSet, iR] = fit[1][-1, -1] / ((np.nanmean(yy)/ym)**4)
			profile_centres_score[iR] = np.sum(((yy_good - gauss3_bias(xx_good, *fit[0])) / yy_good) ** 2)
	# profile_centres_score[iSet, iR] = np.sum((yy - gauss3(xx, *fit[0])) ** 2)
	# plt.figure()
	# plt.plot(xx, yy)
	# plt.plot(xx, gauss3(xx, *fit[0]))
	# plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
	# plt.title('fit: ' + str(fit[0]) + '\n score: ' + str(np.diag(fit[1])))
	# plt.pause(0.01)
	# profile_centre = np.mean(profile_centres, axis=-1)
	# profile_centre = np.sum(profile_centres / (profile_centres_score ** 2), axis=-1) / np.sum(1 / (profile_centres_score ** 2), axis=-1)
	# profile_centres = np.ones_like(profile_centres) * np.nanmedian(profile_centre)

	if False:	# for some reason this doesn't work
		for iR in range(np.shape(all_fits_ss)[-1]):
			# # for col in ['n5']:
			yy = all_fits_ss[:, iR]
			yy[np.isnan(yy)] = 0
			borders = np.linspace(0, len(yy), len(yy) + 1)[np.diff(np.array([0, *yy, 0]) == 0)].astype('int')
			xx_good = xx[borders[0]:borders[-1]]
			yy_good = yy[borders[0]:borders[-1]]
			# if not yy.isnull().any():
			ym = np.nanmax(yy_good)
			p0 = [ym / 6, 20*dx, ym * .8, 5*dx, -ym / 4, 1.1*dx, profile_centres[iR],0,0]
			bds = [[0, dx, 10*dx, dx, -np.inf, 0.5*dx, profile_centres[iR] - 2 * dx,-np.inf,-np.inf],
				   [np.inf, np.inf, np.inf, 20*dx, 0, 5*dx, profile_centres[iR] + 2 * dx,np.inf,np.min(yy_good)]]
			# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
			try:
				fit = curve_fit(gauss3_bias, xx_good, yy_good, p0, maxfev=100000, bounds=bds)
			except:
				print('SS fit failed for line '+str(iR))
				fit = [p0,[1]]
			profile_centres[iR] = fit[0][6]
			if len(fit) == 1:
				profile_centres_score[iR] = 1
			else:
				# profile_centres_score[iSet, iR] = fit[1][-1, -1]
				# profile_centres_score[iSet, iR] = fit[1][-1, -1] / ((np.nanmean(yy)/ym)**4)
				profile_centres_score[iR] = np.sum(((yy_good - gauss3_bias(xx_good, *fit[0])) / yy_good) ** 2)
			# profile_centres_score[iSet, iR] = np.sum((yy - gauss3(xx, *fit[0])) ** 2)
			# if iSet in [35,70]:
			plt.figure()
			plt.plot(xx_good, yy_good)
			plt.plot(xx_good, gauss3_bias(xx_good, *fit[0]))
			# plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
			plt.title('iR '+str(iR)+' fit: ' + str(fit[0])+'\n score: '+str(np.diag(fit[1])))
			plt.pause(0.01)
			# profile_centre = np.mean(profile_centres, axis=-1)
			# profile_centre = np.sum(profile_centres / (profile_centres_score ** 2), axis=-1) / np.sum(1 / (profile_centres_score ** 2), axis=-1)

	if force_glogal_center!=0:
		profile_centre = np.ones_like(profile_centres)*force_glogal_center
	elif same_centre_every_line == True:
		profile_centre = np.sum(profile_centres/(profile_centres_score**2), axis=-1)/np.sum(1/(profile_centres_score**2), axis=-1)
		profile_centre = np.ones_like(profile_centres)*np.nanmedian(profile_centre)
	else:
		profile_centre = cp.deepcopy(profile_centres)
	# profile_centre = np.convolve(profile_centre, np.ones((len(profile_centre)//5)) / (len(profile_centre)//5), mode='valid')


	path = '/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)
	if not os.path.exists(path):
		os.makedirs(path)

	if False:	# this aproach assumes "infinite" precision, sampling only one point
		all_fits_fitted = np.zeros_like(all_fits_ss)
		all_fits_residuals = np.zeros_like(all_fits_ss)
		inverted_profiles = np.zeros((nLine,len(r)))
		inverted_profiles_sigma = np.zeros((nLine,len(r)))

		# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
		for iR in range(np.shape(all_fits_ss)[-1]):
			# gauss3_locked = lambda x, A1, sig1, A2, sig2, A3, sig3: gauss(x, A1, sig1, profile_centre[iR]) + gauss(x, A2, sig2,profile_centre[iR]) + gauss(x, A3, sig3, profile_centre[iR])
			# AInvgauss3_locked = lambda x, A1, sig1, A2, sig2, A3, sig3: AInvgauss(x, A1, sig1, 0) + AInvgauss(x, A2, sig2,0) + AInvgauss(x, A3, sig3, 0)
			gauss3_locked_2_bias = lambda x, A1, sig1, A2, c_sig2, c_A3, c_sig3, m, q: gauss(x, A1, sig1,profile_centre[iR]) + gauss(x, A2,c_sig2 * sig1,profile_centre[iR]) + gauss(x, c_A3*c_sig3*(A1 +A2/c_sig2), c_sig3 * sig1, profile_centre[iR]) + q + x * m
			AInvgauss3_locked_2 = lambda x, A1, sig1, A2, c_sig2, c_A3, c_sig3: AInvgauss(x, A1, sig1, 0) + AInvgauss(x, A2,c_sig2 * sig1,0) + AInvgauss(x, c_A3*c_sig3*(A1 +A2/c_sig2), c_sig3 * sig1, 0)
			yy = all_fits_ss[:, iR]
			for iCorr, value in enumerate(yy):
				if np.isnan(value):
					yy[iCorr] = 0

			temp_all_fits = cp.deepcopy(yy)
			if np.max(yy)>100*np.sort(yy)[-2]:
				pos1 = np.max([yy.argmax()-1,0])
				pos2 = np.min([yy.argmax()+1,len(yy)-1])
				yy[yy.argmax()]=np.mean(yy[[pos1,pos2]])

			borders = np.linspace(0, len(yy), len(yy) + 1)[np.diff(np.array([0, *yy, 0]) == 0)].astype('int')
			xx_good = xx[borders[0]:borders[-1]]
			yy_good = yy[borders[0]:borders[-1]]
			# yy_good[yy_good==0]=1e-8
			yy_sigma = all_fits_ss_sigma[:, iR]
			yy_good_sigma = yy_sigma[borders[0]:borders[-1]]
			yy_good_sigma[np.isnan(yy_good_sigma)]=100*np.nanmax(yy_good_sigma)

			ym = np.nanmax(yy_good)
			# inner_dip_sig = 0.
			inner_dip_sig = 0.15
			# bds = [[0, -np.inf, 0, -np.inf, -max(yy), -np.inf],[max(yy)*100, np.inf, max(yy)*100, 0, np.inf]]
			# bds = [[0, dx, -np.inf, dx, -np.inf, dx, 0], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, max(xx)]]
			bds = [[0, 0, 0, 1, -1, 1e-10,-np.inf,-np.inf],[max(yy)*100, np.inf, max(yy)*100, 10000, 0, 1,np.inf,np.inf]]
			# p0 = [ym / 5, 20e-3, ym * .8, 4e-1, -ym / 2, 0.9e-3]
			# p0 = [ym *0.9, 6e-3, ym * .05, 4e-1, -ym / 4, inner_dip_sig]
			p0 = [ym *0.9, 5e-3, ym * .05, 67, -0.25, inner_dip_sig,0,0]
			# p0 = [ym / 5, 20e-3, ym * .8, 4e-3, -ym / 5, 1e-3,profile_centre[iSet]]
			try:
				fit = curve_fit(gauss3_locked_2_bias, xx_good, yy_good, p0,sigma=yy_good_sigma, absolute_sigma=True, maxfev=100000, bounds=np.array(bds))
				while (AInvgauss3_locked_2(0, *fit[0][:-2]) < 0 and inner_dip_sig < 0.3):
					# inner_dip_sig+=(0.9e-3)/10
					inner_dip_sig += 0.015
					# p0 = [ym * 0.9, 6e-3, ym * .05, 4e-1, -ym / 3, inner_dip_sig]
					p0 = [*fit[0][:5], inner_dip_sig,0,0]
					fit = curve_fit(gauss3_locked_2_bias, xx_good, yy_good, p0,sigma=yy_good_sigma, absolute_sigma=True, maxfev=100000, bounds=np.array(bds))
				# if fit[0][-2]>0:
				# 	inner_dip_sig = 0.9e-3
				# 	p0 = [ym *0.9, 6e-3, ym * .05, 4e-1, -ym / 3, inner_dip_sig]
			except:
				print('fitting failed for line '+str(iR))
				fit = [[0,5e-3,0, 67, -0.25,0.15,0,0],np.ones((8,8))]
			# fit = curve_fit(gauss3_locked, xx_good, yy_good, p0, maxfev=100000, bounds=bds)
			# plt.figure()
			# plt.plot(xx, yy)
			# plt.plot(xx, gauss3_locked(xx, *[*fit[0][:2],0,0,0,0]))
			# plt.plot(xx, gauss3_locked(xx, *[*fit[0][:4],0,0]))
			# plt.plot(xx, gauss3_locked(xx, *fit[0]))
			# plt.plot(xx, gauss3(xx, *fit[0]))
			# plt.pause(0.01)
			fitCoefs[ iR] = fit[0]
			temp = AInvgauss3_locked_2_with_error_included(r, *correlated_values(fit[0][:-2],fit[1][:-2,:-2]) )
			inverted_profiles[ iR] = nominal_values(temp)
			inverted_profiles_sigma[ iR] = std_devs(temp)
			# inverted_profiles[ iR] = AInvgauss3_locked_2(r, *fit[0][:-2])
			all_fits_fitted[ :, iR] = gauss3_locked_2_bias(xx,*fit[0])
			all_fits_residuals[ :, iR] = (temp_all_fits - all_fits_fitted[ :, iR])
			# plt.figure()
			# plt.plot(r[::20], AInvgauss3_locked(r[::20], *fit[0]))
			# plt.pause(0.01)
	elif True:	# here I average the result over a given dx

		print('will plot the spectral fit for SS')
		if not os.path.exists(path+'/abel_inversion_example'):
			os.makedirs(path+'/abel_inversion_example')
		fig, ax = plt.subplots( np.shape(all_fits_ss)[-1],2,figsize=(15, 25), squeeze=True)
		fig.suptitle('Abel inversion for SS\n   same_centre_every_line=%1s,max_central_depression_amplitude=%3g,min_central_depression_sigma=%3g,force_glogal_center=%3g  ' %(same_centre_every_line,max_central_depression_amplitude,min_central_depression_sigma,force_glogal_center) )


		r_new = np.arange(number_of_radial_divisions)*dx
		r_fine = np.arange(number_of_radial_divisions*100)*dx/100
		all_fits_fitted = np.zeros_like(all_fits_ss)
		all_fits_residuals = np.zeros_like(all_fits_ss)
		inverted_profiles_fine = np.zeros((nLine,len(r_fine)))
		inverted_profiles = np.zeros((nLine,number_of_radial_divisions))
		inverted_profiles_sigma = np.zeros((nLine,number_of_radial_divisions))

		# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
		for iR in range(np.shape(all_fits_ss)[-1]):
			# gauss3_locked = lambda x, A1, sig1, A2, sig2, A3, sig3: gauss(x, A1, sig1, profile_centre[iR]) + gauss(x, A2, sig2,profile_centre[iR]) + gauss(x, A3, sig3, profile_centre[iR])
			# AInvgauss3_locked = lambda x, A1, sig1, A2, sig2, A3, sig3: AInvgauss(x, A1, sig1, 0) + AInvgauss(x, A2, sig2,0) + AInvgauss(x, A3, sig3, 0)
			gauss3_locked_2_bias = lambda x, A1, sig1, c_A2, c_sig2, c_A3, c_sig3, m, q: gauss(x, A1, sig1,profile_centre[iR]) + gauss(x, A1*c_A2,c_sig2 * sig1,profile_centre[iR]) + gauss(x, c_A3*c_sig3*(A1 +A1*c_A2/c_sig2), c_sig3 * sig1, profile_centre[iR]) + q + (x-profile_centre[iR]) * m
			AInvgauss3_locked_2 = lambda x, A1, sig1, c_A2, c_sig2, c_A3, c_sig3: AInvgauss(x, A1, sig1, 0) + AInvgauss(x, A1*c_A2,c_sig2 * sig1,0) + AInvgauss(x, c_A3*c_sig3*(A1 +A1*c_A2/c_sig2), c_sig3 * sig1, 0)
			yy = all_fits_ss[:, iR]
			for iCorr, value in enumerate(yy):
				if np.isnan(value):
					yy[iCorr] = 0

			temp_all_fits = cp.deepcopy(yy)
			if np.max(yy)>100*np.sort(yy)[-2]:
				pos1 = np.max([yy.argmax()-1,0])
				pos2 = np.min([yy.argmax()+1,len(yy)-1])
				yy[yy.argmax()]=np.mean(yy[[pos1,pos2]])

			borders = np.linspace(0, len(yy), len(yy) + 1)[np.diff(np.array([0, *yy, 0]) == 0)].astype('int')
			xx_good = xx[borders[0]:borders[-1]]
			yy_good = yy[borders[0]:borders[-1]]
			# yy_good[yy_good==0]=1e-8
			yy_sigma = all_fits_ss_sigma[:, iR]
			yy_good_sigma = yy_sigma[borders[0]:borders[-1]]
			yy_good_sigma[np.isnan(yy_good_sigma)]=100*np.nanmax(yy_good_sigma)

			ym = np.nanmax(yy_good)
			# inner_dip_sig = 0.
			inner_dip_sig = 0.15
			# bds = [[0, -np.inf, 0, -np.inf, -max(yy), -np.inf],[max(yy)*100, np.inf, max(yy)*100, 0, np.inf]]
			# bds = [[0, dx, -np.inf, dx, -np.inf, dx, 0], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, max(xx)]]
			bds = [[0, 0, 0, 1, max_central_depression_amplitude, min_central_depression_sigma,-np.inf,-1e-10],[max(yy)*10000, np.inf, 1, 100, 0, 1,np.inf,1e-10]]
			# p0 = [ym / 5, 20e-3, ym * .8, 4e-1, -ym / 2, 0.9e-3]
			# p0 = [ym *0.9, 6e-3, ym * .05, 4e-1, -ym / 4, inner_dip_sig]
			p0 = [ym *0.9, 5e-3, .05, 10, -0.25, inner_dip_sig,0,0]
			# p0 = [ym / 5, 20e-3, ym * .8, 4e-3, -ym / 5, 1e-3,profile_centre[iSet]]
			x_scale = [ym *0.9, 5e-3, .05, 67, 0.25, inner_dip_sig,1,1]
			try:
				fit = curve_fit(gauss3_locked_2_bias, xx_good, yy_good, p0,sigma=yy_good_sigma, absolute_sigma=True,x_scale=x_scale,ftol=1e-14, maxfev=100000, bounds=np.array(bds))
				while (AInvgauss3_locked_2(0, *fit[0][:-2]) < 0 and inner_dip_sig < 0.3):
					# inner_dip_sig+=(0.9e-3)/10
					inner_dip_sig += 0.015
					# p0 = [ym * 0.9, 6e-3, ym * .05, 4e-1, -ym / 3, inner_dip_sig]
					p0 = [*fit[0][:5], inner_dip_sig,0,0]
					fit = curve_fit(gauss3_locked_2_bias, xx_good, yy_good, p0,sigma=yy_good_sigma, absolute_sigma=True,x_scale=x_scale,ftol=1e-14, maxfev=100000, bounds=np.array(bds))
					print('enlarged the initial value of the inner dip line '+str(iR+4))
				# if fit[0][-2]>0:
				# 	inner_dip_sig = 0.9e-3
				# 	p0 = [ym *0.9, 6e-3, ym * .05, 4e-1, -ym / 3, inner_dip_sig]
			except:
				print('fitting failed for line '+str(iR))
				fit = [[0,5e-3,0, 67, -0.25,0.15,0,0],np.ones((8,8))]

			im = ax[iR,0].errorbar(1000*xx_good, yy_good,yerr=yy_good_sigma,color='c',label='line integrated data')
			im = ax[iR,0].plot([1000*profile_centre[iR],1000*profile_centre[iR]],[np.min([yy_good,gauss3_locked_2_bias(xx, *fit[0])]),np.max([yy_good,gauss3_locked_2_bias(xx, *fit[0])])],'k--')
			im = ax[iR,0].plot(1000*xx, gauss3_locked_2_bias(xx, *fit[0]),'m',label='fit')
			ax[iR,0].set_ylabel('brightness [W m-2 sr-1]')
			ax[iR,0].grid()
			if iR==0:
				ax[iR,0].set_title('cyan=line int data, magenta=fit\nline n='+str(iR+4)+'->2\n ')
			else:
				ax[iR,0].set_title('line n='+str(iR+4)+'->2\n ')
			if iR==(np.shape(all_fits_ss)[-1]-1):
				ax[iR,0].set_xlabel('x [mm]')
			else:
				ax[iR,0].set_xticks([])

			# fit = curve_fit(gauss3_locked, xx_good, yy_good, p0, maxfev=100000, bounds=bds)
			# plt.figure()
			# plt.plot(xx, yy)
			# plt.plot(xx, gauss3_locked(xx, *[*fit[0][:2],0,0,0,0]))
			# plt.plot(xx, gauss3_locked(xx, *[*fit[0][:4],0,0]))
			# plt.plot(xx, gauss3_locked(xx, *fit[0]))
			# plt.plot(xx, gauss3(xx, *fit[0]))
			# plt.pause(0.01)
			fitCoefs[ iR] = fit[0]
			temp = A_erf_3_locked_2_with_error_included(np.array([0,*(r_new+dx/2)]), *correlated_values(fit[0][:-2],fit[1][:-2,:-2]) )
			temp = np.diff(temp)/np.array([dx/2,*(np.ones(number_of_radial_divisions-1)*dx)])
			inverted_profiles[ iR] = nominal_values(temp)
			inverted_profiles_sigma[ iR] = std_devs(temp)
			temp = AInvgauss3_locked_2_with_error_included(r_fine, *correlated_values(fit[0][:-2],fit[1][:-2,:-2]) )
			inverted_profiles_fine[ iR] = nominal_values(temp)
			# inverted_profiles[ iR] = AInvgauss3_locked_2(r, *fit[0][:-2])
			all_fits_fitted[ :, iR] = gauss3_locked_2_bias(xx,*fit[0])
			all_fits_residuals[ :, iR] = (temp_all_fits - all_fits_fitted[ :, iR])
			# plt.figure()
			# plt.plot(r[::20], AInvgauss3_locked(r[::20], *fit[0]))
			# plt.pause(0.01)

			im = ax[iR,1].errorbar(1000*r, inverted_profiles[ iR],yerr=inverted_profiles_sigma[ iR],color='b',label='inverted data')
			im = ax[iR,1].plot(1000*r, inverted_profiles_sigma[ iR],'r',label='error')
			ax[iR,1].set_ylabel('emissivity [W m^-3 sr^-1]')
			ax[iR,1].yaxis.set_label_position('right')
			ax[iR,1].yaxis.tick_right()
			ax[iR,1].grid()
			if iR==0:
				ax[iR,1].set_title('blue=inverted prof, red=error\nfit='+str(''.join(['%.3g+/-%.3g, ' %(_,__) for _,__ in zip(fit[0],np.sqrt(np.diag(fit[1])))])[:-2]))
			else:
				ax[iR,1].set_title('fit='+str(''.join(['%.3g+/-%.3g, ' %(_,__) for _,__ in zip(fit[0],np.sqrt(np.diag(fit[1])))])[:-2]))
			if iR==(np.shape(all_fits_ss)[-1]-1):
				ax[iR,1].set_xlabel('r [mm]')
			else:
				ax[iR,1].set_xticks([])
		plt.savefig(path+'/abel_inversion_example/abe_inversion_SS.eps', bbox_inches='tight')
		plt.close('all')

	color = ['b', 'r', 'm', 'y', 'g', 'c', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise']
	plt.figure(figsize=(20, 10))
	for iR in range(nLine):
		plt.errorbar(1000*xx,all_fits_ss[:,iR],yerr=all_fits_ss_sigma[:,iR],color=color[iR],label='line '+str(iR+4)+', m=%.3g' %(fitCoefs[ iR][-2]),linewidth=0.7)
		plt.plot(1000*xx, all_fits_fitted[:,iR],color=color[iR],linestyle='--',linewidth=0.4)
		# plt.plot(1000*xx, np.abs(all_fits_residuals[:,iR]),color=color[iR],linestyle=':',linewidth=0.4)
		plt.plot(1000*xx, fitCoefs[ iR][-1]+(xx-profile_centre[iR])*fitCoefs[ iR][-2],color=color[iR],linestyle=':',linewidth=0.4)
		plt.plot([1000*profile_centre[iR],1000*profile_centre[iR]],[np.min(all_fits_ss),np.max(all_fits_ss)],color=color[iR],linestyle='-.')
	plt.title("Steady state line integrated profiles\n'--'=fitting, ':'=bias\n mean centre "+str(np.around(1000*np.mean(profile_centre,axis=0), decimals=1))+'mm')
	plt.xlabel('LOS location [mm]')
	plt.ylabel('Line integrated brightness [W m^-2 sr^-1]')
	plt.legend(loc='best',fontsize='x-small')
	plt.semilogy()
	plt.grid(b=True, which='both')
	plt.ylim(max(1e-4,np.min(all_fits_ss)*0.9),np.max(all_fits_ss)*1.1)
	plt.savefig('%s/SS_line_profile.eps' % (path), bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	for iR in range(nLine):
		if np.max(inverted_profiles_sigma[iR])>10*np.max(inverted_profiles[iR]):
			plt.plot(1000*r_new,inverted_profiles[iR],color=color[iR],marker='+',label='line '+str(iR+4)+' $\sigma_{max}$=%.3g' %(np.max(inverted_profiles_sigma[iR])))
		else:
			plt.errorbar(1000*r_new,inverted_profiles[iR],yerr=inverted_profiles_sigma[iR],color=color[iR],marker='+',label='line '+str(iR+4))
		plt.plot(1000*r_fine,inverted_profiles_fine[iR],color=color[iR],linestyle='--',label='fine inversion, line '+str(iR+4))
	plt.title('Steady state line profiles \n mean centre '+str(np.around(1000*np.mean(profile_centre,axis=0), decimals=1))+'mm')
	plt.xlabel('radius [mm]')
	plt.ylabel('Emissivity [W m^-3 sr^-1]')
	plt.semilogy()
	plt.legend(loc='best',fontsize='x-small')
	plt.ylim(bottom=np.min(inverted_profiles[-1]),top=np.max(inverted_profiles[0]))
	plt.savefig('%s/SS_abel_inverted_line_profile.eps' % (path), bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10))
	for iR in range(nLine-1):
		plt.plot(1000*r_new,inverted_profiles[iR+1]/inverted_profiles[iR],color=color[iR],label='line ratio '+str(iR+4+1)+'/'+str(iR+4))
	plt.title('Steady state line ratio profiles \n mean centre '+str(np.around(1000*np.mean(profile_centre,axis=0), decimals=1))+'mm')
	plt.xlabel('radius [mm]')
	plt.ylabel('Line ratio [au]')
	plt.legend(loc='best')
	# plt.ylim(bottom=0)
	plt.semilogy()
	plt.grid(b=True, which='both')
	plt.ylim(bottom=1e-2)
	plt.savefig('%s/SS_abel_inverted_line_ratio_profile.eps' % (path), bbox_inches='tight')
	plt.close()
	np.save('%s/SS_profileFits.npy' % (path), fitCoefs)
	np.save('%s/SS_inverted_profiles.npy' % (path), inverted_profiles)
	np.save('%s/SS_inverted_profiles_sigma.npy' % (path), inverted_profiles_sigma)

	if force_glogal_center==0:
		return profile_centre


def find_plasma_axis(df_settings,all_fits, merge_ID_target, new_timesteps,dx,xx,r,same_centre_every_line=True,same_centre_all_time=True,restrict_to_steady_state=False,max_central_depression_amplitude=-1,min_central_depression_sigma=0.1):

	import os
	import pandas as pd
	import numpy as np
	from scipy.optimize import curve_fit
	import matplotlib.pyplot as plt
	import copy as cp

	nLine = np.shape(all_fits)[-1]
	first_time = np.min(new_timesteps)
	last_time = np.max(new_timesteps)
	time_res = np.mean(np.diff(new_timesteps))

	gauss = lambda x, A, sig, x0: A * np.exp(-((x - x0) / sig) ** 2)
	gauss_bias = lambda x, A, sig, x0,m,q: A * np.exp(-((x - x0) / sig) ** 2)+q+x*m
	# gauss2 = lambda x,A1,sig1,A2,sig2,x0: gauss(x,A1,sig1,x0) + gauss(x,A2,sig2,x0)
	gauss3 = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: gauss(x, A1, sig1, x0) + gauss(x, A2, sig2, x0) + gauss(x, A3,sig3,x0)
	gauss3_bias = lambda x, A1, sig1, A2, sig2, A3, sig3, x0,m,q: gauss(x, A1, sig1, x0) + gauss(x, A2, sig2, x0) + gauss(x, A3,sig3,x0)+q+x*m
	gaussJac = lambda x, A, sig, x0: np.array([gauss(x, 1, sig, x0), gauss(x, A, sig, x0) * 2 * (x - x0) ** 2 / sig ** 3,gauss(x, A, sig, x0) * 2 * (x - x0) / sig ** 2])
	gauss3Jac = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: [np.concatenate([a[:2], a[3:5], a[6:8], [sum(a[2::3])]]) for a in [np.concatenate([gaussJac(x, A1, sig1, x0), gaussJac(x, A2, sig2, x0), gaussJac(x, A3, sig3, x0)])]][0]
	AInvgauss = lambda x, A, sig, x0: gauss(x, A / sig / np.sqrt(np.pi), sig, 0)
	AInvgauss3 = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: AInvgauss(x, A1, sig1, 0) + AInvgauss(x, A2, sig2,0) + AInvgauss(x, A3,sig3,0)
	AInvgaussJac = lambda x, A, sig, x0: np.array([gauss(x, 1 / np.sqrt(np.pi) / sig, sig, 0),gauss(x, A, sig, 0) * (2 * x ** 2 - sig ** 2) / sig ** 4 / np.sqrt(np.pi), gauss(x, A, sig, 0) * 2 * x / sig ** 3 / np.sqrt(np.pi)])
	AInvgauss3Jac = lambda x, A1, sig1, A2, sig2, A3, sig3, x0: [np.concatenate([a[:2], a[3:5], a[6:8], [sum(a[2::3])]]) for a in [np.concatenate([AInvgaussJac(x, A1, sig1, 0), AInvgaussJac(x, A2, sig2, 0), AInvgaussJac(x, A3, sig3, 0)])]][0]
	LRvecmul = lambda vec, mat: np.dot(vec, np.matmul(mat, vec))
	gauss3_2 = lambda x, A1, sig1, c_A2, c_sig2, c_A3, c_sig3,x0: gauss(x, A1, sig1, x0) + gauss(x,A1*c_A2,c_sig2*sig1,x0) + gauss(x, c_A3*c_sig3*(A1 +A1*c_A2/c_sig2), c_sig3*sig1, x0)
	# bds2 = [[0,1e-4,0,1e-4,0],[np.inf,np.inf,np.inf,np.inf,np.inf]]
	bds3 = [[0, 1e-4, 0, 1e-4, -np.inf, 1e-4, 0], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, np.inf]]

	# dx = 18 / 40 * (50.5 / 27.4) / 1e3
	# xx = np.arange(40) * dx  # m
	xn = np.linspace(0, max(xx), 1000)
	# r = np.linspace(-max(xx) / 2, max(xx) / 2, 1000)
	# bds = [[-np.inf, dx, -np.inf, dx, -np.inf, dx, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, max(xx)]]
	bds = [[0, dx, -np.inf, dx, -np.inf, dx, 0], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, max(xx)]]
	bds4 = [[0, dx, 0], [np.inf, np.inf, max(xx)]]
	fitCoefs = np.zeros((len(all_fits), nLine, 6))
	# fitCoefs = np.zeros((len(all_fits), nLine, 8))
	leg = ['n = %i' % i for i in range(3, 13)]

	all_fits[np.isnan(all_fits)]=0

	time_step = int((np.abs(new_timesteps+0.05)).argmin())	#I select the time 0.05 ms before the pulse
	all_fits_ss = cp.deepcopy(all_fits[:time_step])
	all_fits_ss[all_fits_ss==0]=np.nan
	all_fits_ss = np.nanmean(all_fits_ss,axis=0)
	profile_centres = np.zeros((nLine))
	profile_centres_score = np.zeros((nLine))

	if restrict_to_steady_state==True:
		fit = [[0, 4e-3, 20 * dx],np.ones((3,3))*dx]
		for iR in range(np.shape(all_fits_ss)[-1]):
			# # for col in ['n5']:
			yy = all_fits_ss[:, iR]
			yy[np.isnan(yy)] = 0
			# if not yy.isnull().any():
			borders = np.linspace(0, len(yy), len(yy) + 1)[np.diff(np.array([0, *yy, 0]) == 0)].astype('int')
			xx_good = xx[borders[0]:borders[-1]]
			yy_good = yy[borders[0]:borders[-1]]
			ym = np.nanmax(yy_good)
			if (fit[1][2, 2]) ** 0.5 < dx / 2:
				p0 = [ym, fit[0][1], fit[0][2],0,0]
			else:
				p0 = [ym, 4e-3, 20 * dx,0,0]
			if iR > 0:
				bds4 = [[0, dx, profile_centres[iR - 1] - 3 * dx,-np.inf,-np.inf],[max(yy_good) * 10, max(xx_good), profile_centres[iR - 1] + 3 * dx,np.inf,np.inf]]
			else:
				bds4 = [[0, dx, 10 * dx,-np.inf,-np.inf], [max(yy_good) * 10, max(xx_good), 30 * dx,np.inf,np.inf]]

			try:
				fit = curve_fit(gauss_bias, xx_good, yy_good, p0, maxfev=100000, bounds=np.array(bds4))
			except:
				fit = [p0,np.ones((3,3))*np.inf]
			profile_centres[iR] = fit[0][2]
			profile_centres_score[iR] = fit[1][-1,-1]
		profile_centre = np.sum(profile_centres / (profile_centres_score ** 2), axis=-1) / np.sum(1 / (profile_centres_score ** 2), axis=-1)
		for iR in range(np.shape(all_fits_ss)[-1]):
			yy = all_fits_ss[:, iR]
			yy[np.isnan(yy)] = 0
			borders = np.linspace(0, len(yy), len(yy) + 1)[np.diff(np.array([0, *yy, 0]) == 0)].astype('int')
			xx_good = xx[borders[0]:borders[-1]]
			yy_good = yy[borders[0]:borders[-1]]
			ym = np.nanmax(yy_good)
			p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 4, 0.9e-3, profile_centre,0,0]
			bds = [[0, dx, -np.inf, dx, -np.inf, dx, profile_centre - 2 * dx,-np.inf,-np.inf],
				   [np.inf, np.inf, np.inf, np.inf, 0, np.inf, profile_centre + 2 * dx,np.inf,np.inf]]
			try:
				fit = curve_fit(gauss3_bias, xx_good, yy_good, p0, maxfev=100000, bounds=bds)
			except:
				fit = [p0]
			profile_centres[iR] = fit[0][6]
			if len(fit) == 1:
				profile_centres_score[iR] = 1
			else:
				profile_centres_score[iR] = np.sum(((yy_good - gauss3_bias(xx_good, *fit[0])) / yy_good) ** 2)
		profile_centre = np.sum(profile_centres / (profile_centres_score ** 2), axis=-1) / np.sum(1 / (profile_centres_score ** 2), axis=-1)
		profile_centre = np.ones_like(profile_centres) * np.nanmedian(profile_centre)
		for iR in range(np.shape(all_fits_ss)[-1]):
			# # for col in ['n5']:
			yy = all_fits_ss[:, iR]
			yy[np.isnan(yy)] = 0
			borders = np.linspace(0, len(yy), len(yy) + 1)[np.diff(np.array([0, *yy, 0]) == 0)].astype('int')
			xx_good = xx[borders[0]:borders[-1]]
			yy_good = yy[borders[0]:borders[-1]]
			# if not yy.isnull().any():
			ym = np.nanmax(yy_good)
			p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 4, 0.9e-3, profile_centres[iR],0,0]
			bds = [[0, dx, -np.inf, dx, -np.inf, dx, profile_centres[iR] - 2 * dx,-np.inf,-np.inf],
				   [np.inf, np.inf, np.inf, np.inf, 0, np.inf, profile_centres[iR] + 2 * dx,np.inf,np.inf]]
			# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
			try:
				fit = curve_fit(gauss3_bias, xx_good, yy_good, p0, maxfev=100000, bounds=bds)
			except:
				fit = [p0]
			profile_centres[iR] = fit[0][6]
			if len(fit) == 1:
				profile_centres_score[iR] = 1
			else:
				profile_centres_score[iR] = np.sum(((yy_good - gauss3_bias(xx_good, *fit[0])) / yy_good) ** 2)

		if same_centre_every_line == True:
			profile_centre = np.sum(profile_centres/(profile_centres_score**2), axis=-1)/np.sum(1/(profile_centres_score**2), axis=-1)
			profile_centre = np.ones_like(profile_centres)*np.nanmedian(profile_centre)
		else:
			profile_centre = cp.deepcopy(profile_centres)
	else:
		selected = []
		minimum_level = np.min(all_fits[all_fits!=0])
		selected_lines = []
		for iR in range(np.shape(all_fits)[-1]):
			high_level = np.mean(np.sort(all_fits[:,:,iR].flatten())[-10:])*0.2	#	0.2 is completely arbitrary
			selected.append(all_fits[np.mean(all_fits[:,:,iR],axis=1)>high_level,:,iR])
			if np.sum(np.mean(all_fits[:,:,iR],axis=1)>high_level)>0:
				selected_lines.append(iR)

		profile_centres = np.zeros((len(all_fits),nLine))
		profile_centres_score = np.zeros((len(all_fits),nLine))

		if (same_centre_every_line==True and same_centre_all_time==True):
			profile_centres = []
			profile_centres_score = []
			result = []
			result_score = []

			fit = [[0, 4e-3, 20 * dx],np.array([[dx,dx],[dx,dx]])]
			for index_iR,iR in enumerate(selected_lines):
				profile_centres.append([])
				profile_centres_score.append([])

				# # for col in ['n5']:
				for iSet in range(len(selected[iR])):
					# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
					yy = selected[iR][iSet,:]
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
					if iR>0 and profile_centres[index_iR-1]!=[]:
						bds4 = [[0, dx, np.mean(profile_centres[index_iR-1])-3*dx], [max(yy_good)*10, max(xx_good), np.mean(profile_centres[index_iR-1])+3*dx]]
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
					profile_centres[index_iR].append(fit[0][-1])
					# profile_centres_score[iSet, iR] = fit[1][-1,-1]
					profile_centres_score[index_iR].append(np.sum(((yy_good - gauss(xx_good, *fit[0]))/np.max(yy_good)) ** 2))
				result.append( np.nansum(np.divide(profile_centres[index_iR],(np.power(profile_centres_score[index_iR],2))))/np.nansum(1/(np.power(profile_centres_score[index_iR],2)), axis=(0)) )
				result_score.append( np.nansum(1/(np.power(profile_centres_score[index_iR],2)), axis=(0)) )
			profile_centre = np.sum(np.multiply(result,result_score))/np.sum(result_score)

			profile_centres = []
			profile_centres_score = []
			result = []
			result_score = []

			for index_iR,iR in enumerate(selected_lines):
				profile_centres.append([])
				profile_centres_score.append([])

				# # for col in ['n5']:
				for iSet in range(len(selected[iR])):
					# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
					# # for col in ['n5']:
					yy = selected[iR][iSet,:]
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
					p0 = [ym / 6, 30*dx, ym * .8, 5*dx, -ym / 4, dx, profile_centre]
					bds = [[0, dx, -np.inf, dx, -np.inf, dx, profile_centre-2*dx], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, profile_centre+2*dx]]
					# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
					try:
						fit = curve_fit(gauss3, xx_good, yy_good, p0, maxfev=100000, bounds=bds)
					except:
						fit=[p0]
					profile_centres[index_iR].append(fit[0][-1])
					if len(fit)==1:
						profile_centres_score[index_iR].append(1)
					else:
						# profile_centres_score[iSet, iR] = fit[1][-1, -1]
						# profile_centres_score[iSet, iR] = fit[1][-1, -1] / ((np.nanmean(yy)/ym)**4)
						profile_centres_score[index_iR].append( np.sum(((yy_good - gauss3(xx_good, *fit[0]))/np.max(yy_good)) ** 2) )
				# profile_centres_score[iSet, iR] = np.sum((yy - gauss3(xx, *fit[0])) ** 2)
				# plt.figure()
					# plt.plot(xx, yy)
					# plt.plot(xx, gauss3(xx, *fit[0]))
					# plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
					# plt.title('score: '+str(np.diag(fit[1])))
					# plt.pause(0.01)
				result.append( np.nansum(np.divide(profile_centres[index_iR],(np.power(profile_centres_score[index_iR],2))))/np.nansum(1/(np.power(profile_centres_score[index_iR],2)), axis=(0)) )
				result_score.append( np.nansum(1/(np.power(profile_centres_score[index_iR],2)), axis=(0)) )
			profile_centre = np.sum(np.multiply(result,result_score))/np.sum(result_score)

			profile_centres = []
			profile_centres_score = []
			result = []
			result_score = []

			for index_iR,iR in enumerate(selected_lines):
				profile_centres.append([])
				profile_centres_score.append([])

				# # for col in ['n5']:
				for iSet in range(len(selected[iR])):
					# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
					# # for col in ['n5']:
					yy = selected[iR][iSet,:]
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
					# p0 = [ym / 6, 30*dx, ym * .8, 5*dx, -ym / 4, dx, profile_centre]
					# bds = [[0, dx, -np.inf, dx, -np.inf, dx, profile_centre - 2 * dx],[np.inf, np.inf, np.inf, np.inf, 0, np.inf, profile_centre + 2 * dx]]
					bds = [[0, 0, 0, 1, max_central_depression_amplitude, min_central_depression_sigma,profile_centre - 2 * dx],[max(yy)*100, np.inf, 1, 100, 0, 1,profile_centre + 2 * dx]]
					p0 = [ym *0.9, 6*dx, .05, 10, -0.25, 0.5,profile_centre]
					x_scale = [ym,dx,1,1,1,1,profile_centre]
					# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
					try:
						# fit = curve_fit(gauss3, xx_good, yy_good, p0, maxfev=100000, bounds=bds)
						fit = curve_fit(gauss3_2, xx_good, yy_good, p0, maxfev=100000, bounds=bds,x_scale=x_scale)
					except:
						fit=[p0]
					profile_centres[index_iR].append(fit[0][-1])
					if len(fit)==1:
						profile_centres_score[index_iR].append( 1)
					else:
						profile_centres_score[index_iR].append( fit[1][-1, -1] )
						# profile_centres_score[iSet, iR] = fit[1][-1, -1] / ((np.nanmean(yy)/ym)**4)
						# profile_centres_score[index_iR].append( np.sum(((yy_good - gauss3(xx_good, *fit[0]))/np.max(yy_good)) ** 2) )
						# profile_centres_score[index_iR].append( np.sum(((yy_good - gauss3_2(xx_good, *fit[0]))/np.max(yy_good)) ** 2) )
					# profile_centres_score[iSet, iR] = np.sum((yy - gauss3(xx, *fit[0])) ** 2)
					# if iSet in [35,70]:
					# plt.figure()
					# plt.plot(xx, yy)
					# plt.plot(xx, gauss3_2(xx, *fit[0]))
					# plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
					# plt.title('fit '+str(fit[0])+'\nsigma: '+str(np.sqrt(np.diag(fit[1]))))
					# plt.pause(0.01)
					# profile_centre = np.mean(profile_centres, axis=-1)
				result.append( np.nansum(np.divide(profile_centres[index_iR],(np.power(profile_centres_score[index_iR],2))))/np.nansum(1/(np.power(profile_centres_score[index_iR],2)), axis=(0)) )
				result_score.append( np.nansum(1/(np.power(profile_centres_score[index_iR],2)), axis=(0)) )
			profile_centre = np.sum(np.multiply(result,result_score))/np.sum(result_score)
			print('profile_centre')
			print(profile_centre)
			profile_centres = np.ones_like(all_fits[:,0])*profile_centre
			# profile_centre = np.convolve(profile_centre, np.ones((len(profile_centre)//5)) / (len(profile_centre)//5), mode='valid')

		elif same_centre_all_time==True:
			# this section is not prepared yet
			print('this section of find_plasma_axis is not prepared yet')
			exit()
			profile_centres = np.zeros((len(all_fits),nLine))
			profile_centres_score = np.zeros((len(all_fits),nLine))
			# fit = [[0, 4e-3, 20 * dx,0,0],dx*np.ones((5,5))]
			fit = [[0, 4e-3, 20 * dx],dx*np.ones((3,3))]
			for iSet in range(len(all_fits)):
				# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
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
					# yy_good[yy_good==0]=minimum_level
					ym = np.nanmax(yy_good)
					# p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 3, 1e-3, 20 * dx]
					if ((fit[1][2,2])**0.5<dx/2 and iR>0 and (np.abs(fit[0][1]-20*dx)<5*dx)):
						# p0 = [ym, fit[0][1], fit[0][2],0,0]
						p0 = [ym, fit[0][1], fit[0][2]]
					else:
						# p0 = [ym, 4e-3, 20 * dx,0,0]
						p0 = [ym, 4e-3, 20 * dx]
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
						fit = curve_fit(gauss, xx_good, yy_good, p0, maxfev=10000, bounds=np.array(bds4))
						if fit[1][2,2]==0:
							fit[1][2, 2]=np.inf
					except:
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
					p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 4, 0.9e-3, profile_centre[iR]]
					bds = [[0, dx, -np.inf, dx, -np.inf, dx, profile_centre[iR]-2*dx], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, profile_centre[iR]+2*dx]]
					# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
					try:
						fit = curve_fit(gauss3, xx_good, yy_good, p0, maxfev=10000, bounds=bds)
					except:
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
					p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 4, 0.9e-3, profile_centre[iR]]
					bds = [[0, dx, -np.inf, dx, -np.inf, dx, profile_centre[iR] - 2 * dx],[np.inf, np.inf, np.inf, np.inf, 0, np.inf, profile_centre[iR] + 2 * dx]]
					# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
					try:
						fit = curve_fit(gauss3, xx_good, yy_good, p0, maxfev=100000, bounds=bds)
					except:
						fit=[p0,np.inf*np.ones((len(p0),len(p0)))]
					profile_centres[iSet, iR] = fit[0][-1]
					if len(fit)==1:
						profile_centres_score[iSet, iR] = 1
					else:
						profile_centres_score[iSet, iR] = fit[1][-1, -1]
						# profile_centres_score[iSet, iR] = fit[1][-1, -1] / ((np.nanmean(yy)/ym)**4)
						# profile_centres_score[iSet, iR] = np.sum(((yy_good - gauss3(xx_good, *fit[0]))/yy_good) ** 2)
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

	return profile_centres

	# profile_centres = np.zeros((len(all_fits)))
	# profile_centres_score = np.zeros((len(all_fits)))
	# fit = [[0, 20e-3, 0, 4e-3, 0, 0.9e-3, 20 * dx]]
	# p0_1 = [0, 0, 0, 0, 0, 0, fit[0][-1]]
	# iR = int(np.median(all_fits.argmax(axis=-1)))	#I want the stronger signal
	# for iSet in range(len(all_fits)):
	# 	# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
	# 	# for iR in range(np.shape(all_fits)[-1]):
	# 	# iR=0
	# 	# # for col in ['n5']:
	# 	yy = all_fits[iSet,:,iR]
	# 	for iCorr, value in enumerate(yy):
	# 		if np.isnan(value):
	# 			yy[iCorr]=0
	# 	# if not yy.isnull().any():
	# 	ym = np.nanmax(yy)
	# 	# p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 3, 1e-3, 20 * dx]
	# 	bds = [[0, dx, -np.inf, dx, -np.inf, dx, 0], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, max(xx)]]
	# 	p0 = [ym / 6, fit[0][1], ym * .8, fit[0][3], -ym / 4, fit[0][5], fit[0][-1]]
	# 	# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
	# 	try:
	# 		# fit = curve_fit(gauss, xx, yy, p0, maxfev=100000, bounds=np.array(bds)[:,4:])
	# 		fit = curve_fit(gauss3, xx, yy, p0, maxfev=100000, bounds=bds)
	# 	except:
	# 		fit=[p0]
	# 	if iSet>1:
	# 		p0_1 = [0, 0, 0, 0, 0, 0, profile_centres[iSet-2]]
	# 	# plt.figure()
	# 	# plt.plot(xx, yy)
	# 	# plt.plot(xx, gauss3(xx, *fit[0]))
	# 	# plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
	# 	# plt.title('score: '+str(np.diag(fit[1])))
	# 	# plt.pause(0.01)
	# 	profile_centres[iSet]=fit[0][-1]
	# 	profile_centres_score[iSet] = np.sum((yy - gauss3(xx, *fit[0]))**2)*((fit[0][-1]-(p0_1[-1]+p0[-1])/2)**2)/(fit[0][4]**2)
	# # profile_centre = np.mean(profile_centres,axis=-1)
	# profile_centre = np.sum(profile_centres/(profile_centres_score**2), axis=-1)/np.sum(1/(profile_centres_score**2), axis=-1)
	#
	# fit = [[0, 20e-3, 0, 4e-3, 0, 2e-3, profile_centre]]
	# p0_1 = [0, 0, 0, 0, 0, 0, fit[0][-1]]
	# iR = int(np.median(all_fits.argmax(axis=-1)))	#I want the stronger signal
	# for iSet in range(len(all_fits)):
	# 	# df_lineRads = pd.read_csv('%s%i.csv' % (ddir, iSet))
	# 	# for iR in range(np.shape(all_fits)[-1]):
	# 	# iR=0
	# 	# # for col in ['n5']:
	# 	yy = all_fits[iSet,:,iR]
	# 	for iCorr, value in enumerate(yy):
	# 		if np.isnan(value):
	# 			yy[iCorr]=0
	# 	# if not yy.isnull().any():
	# 	ym = np.nanmax(yy)
	# 	# p0 = [ym / 6, 20e-3, ym * .8, 4e-3, -ym / 3, 1e-3, 20 * dx]
	# 	p0 = [ym / 6, fit[0][1], ym * .8, fit[0][3], -ym / 2, 2e-3, fit[0][-1]]
	# 	# bds = [[0,dx,0,dx,-np.inf,dx,0],[np.inf,np.inf,np.inf,np.inf,0,np.inf,max(xx)]]
	# 	bds = [[0, 5*dx, 0, 3*dx, -np.inf, dx, fit[0][-1]-dx/4], [np.inf, np.inf, np.inf, np.inf, 0, np.inf, fit[0][-1]+dx/4]]
	# 	try:
	# 		# fit = curve_fit(gauss, xx, yy, p0, maxfev=100000, bounds=np.array(bds)[:,4:])
	# 		fit = curve_fit(gauss3, xx, yy, p0, maxfev=100000, bounds=bds)
	# 	except:
	# 		fit=[p0]
	# 	if iSet>1:
	# 		p0_1 = [0, 0, 0, 0, 0, 0, profile_centres[iSet-2]]
	# 	if iSet in [208, 210, 212, 211,324]:
	# 		plt.figure()
	# 		plt.plot(xx, yy)
	# 		plt.plot(xx, gauss3(xx, *fit[0]))
	# 		plt.plot([fit[0][-1],fit[0][-1]],[min(yy),max(yy)])
	# 		# plt.title('score: '+str(np.diag(fit[1])))
	# 		plt.title('fit: '+str(fit[0]))
	# 		plt.pause(0.01)
	# 	profile_centres[iSet]=fit[0][-1]
	# 	profile_centres_score[iSet] = np.sum((yy - gauss3(xx, *fit[0]))**2)*((fit[0][-1]-(p0_1[-1]+p0[-1])/2)**2)/(fit[0][4]**2)
	# # profile_centre = np.mean(profile_centres,axis=-1)
	# profile_centre = np.sum(profile_centres/(profile_centres_score**2), axis=-1)/np.sum(1/(profile_centres_score**2), axis=-1)
	# # profile_centre = np.convolve(profile_centre, np.ones((30,))/30, mode='valid')


def doLateralfit_single_with_sigma_pure_Abel(df_settings,all_fits_ss,all_fits_ss_sigma, merge_ID_target, new_timesteps,dx,xx,r,number_of_radial_divisions=21,N=11,same_centre_every_line=True,same_centre_all_time=True,force_glogal_center=0):
	# technique from A New Method for Numerical Abel-Inversion, Oliver, J., 2013
	import scipy.integrate as integrate
	import abel
	import numpy as np
	from scipy.optimize import curve_fit
	import matplotlib.pyplot as plt
	import copy as cp
	import os


	nLine = np.shape(all_fits_ss)[-1]


	if False:	# here I fit the brightness with a Fourier series
		i_time=198
		force_glogal_center=0.0224395234639
		N=11
		for line in range(9):
			xx_int = xx- force_glogal_center
			xx_int_check = np.linspace(np.min(xx_int), np.max(xx_int) , 1000)
			r_int = np.linspace(0, max(np.abs(xx_int)) , 1000)
			dr_int=np.median(np.diff(r_int))
			# r_int = r[r<=np.max(np.abs(xx_int))]
			R = np.max(np.abs(xx_int))
			F_0=(R**2-xx_int**2)*0.5
			all_F_n = [F_0]
			for n in range(1,N):
				temp=[]
				for x_value in xx_int:
					selection=np.logical_and(np.abs(r_int)>=np.abs(x_value) , np.abs(r_int)<=R)
					# temp.append((n*np.pi/R)* integrate.quad(lambda x: (np.sin(n*np.pi*x/R))*((x**2-force_glogal_center**2)**0.5), force_glogal_center, R) )
					temp.append(-((-1)**n)*(n*np.pi/R)* np.trapz((np.sin(n*np.pi*r_int[selection]/R))*((r_int[selection]**2-x_value**2)**0.5), x=r_int[selection]) )
				all_F_n.append(temp)
			all_F_n=np.array(all_F_n)

			def mother_F_star(all_F_n):
				def F_star(x,*args):
					return 2*np.sum(args*all_F_n.T,axis=1)
				return F_star

			def mother_F_star_limited(all_F_n):
				def F_star(x,*args):
					A_N = args[-1]-np.sum(args[1:-1]*(1-(-1)**np.linspace(1,N-2,N-2)*np.cos(np.pi*((2*N-3)/(2*N-2))*np.linspace(1,N-2,N-2)) ))
					return 2*np.sum(np.array([*args[:-1],A_N])*all_F_n.T,axis=1)
				return F_star

			F_0_check=(R**2-xx_int_check**2)*0.5
			all_F_n_check = [F_0_check]
			for n in range(1,N):
				temp=[]
				for x_value in xx_int_check:
					selection=np.logical_and(np.abs(r_int)>=np.abs(x_value) , np.abs(r_int)<=R)
					# temp.append((n*np.pi/R)* integrate.quad(lambda x: (np.sin(n*np.pi*x/R))*((x**2-force_glogal_center**2)**0.5), force_glogal_center, R) )
					temp.append(-((-1)**n)*(n*np.pi/R)* np.trapz((np.sin(n*np.pi*r_int[selection]/R))*((r_int[selection]**2-x_value**2)**0.5), x=r_int[selection]) )
				all_F_n_check.append(temp)
			all_F_n_check=np.array(all_F_n_check)

			# def mother_F_star_check(all_F_n_check):
			# 	def F_star(x,*args):
			# 		return 2*np.sum(args*all_F_n_check.T,axis=1)
			# 	return F_star

			# fake_x=np.arange(len(all_fits[200,:,0]))
			# guess=np.ones((N))
			# guess[0]=0
			# bds=[[-0.0001,*-np.ones((N-1))*np.inf],[0.0001,*np.ones((N-1))*np.inf]]
			# fit = curve_fit(mother_F_star(all_F_n),fake_x,all_fits[200,:,0],guess,maxfev=1000000,bounds=bds)
			# plt.figure(N*2);plt.title('N '+str(N));plt.plot(xx_int,mother_F_star(all_F_n)(fake_x,*fit[0]));plt.plot(xx_int_check,mother_F_star(all_F_n_check)(fake_x,*fit[0]));plt.plot(xx_int,all_fits[200,:,0]);plt.plot(xx_int,mother_F_star(all_F_n)(fake_x,*fit[0])-all_fits[200,:,0]);plt.pause(0.01)

			fake_x=np.arange(len(all_fits[i_time,:,0]))
			guess=np.ones((N))
			guess[-1]=0
			# guess[0]=0
			# bds=[[-0.0001,*-np.ones((N-2))*np.inf,0],[0.0001,*np.ones((N-1))*np.inf]]
			bds=[[*-np.ones((N-1))*np.inf,0],[*np.ones((N))*np.inf]]
			fit_limited = curve_fit(mother_F_star_limited(all_F_n),fake_x,all_fits[i_time,:,line],guess,bounds=bds,maxfev=1000000)
			plt.figure(line);plt.title('N '+str(N));plt.plot(xx_int,mother_F_star_limited(all_F_n)(fake_x,*fit_limited[0]));plt.plot(xx_int_check,mother_F_star_limited(all_F_n_check)(fake_x,*fit_limited[0]),'+');plt.plot(xx_int,all_fits[i_time,:,line]);plt.plot(xx_int,mother_F_star_limited(all_F_n)(fake_x,*fit_limited[0])-all_fits[i_time,:,line]);plt.pause(0.01)


			f_0=np.ones_like(r_int)
			all_f_n = [f_0]
			for n in range(1,N):
				all_f_n.append(1-((-1)**n)*np.cos(n*np.pi*r_int/R))
			all_f_n=np.array(all_f_n)

			def mother_f_star(all_f_n):
				def F_star(x,*args):
					return np.sum(args*all_f_n.T,axis=1)
				return F_star

			def mother_f_star_limited(all_f_n):
				def F_star(x,*args):
					A_N = args[-1]-np.sum(args[1:-1]*(1-(-1)**np.linspace(1,N-2,N-2)*np.cos(np.pi*((2*N-3)/(2*N-2))*np.linspace(1,N-2,N-2)) ))
					return np.sum(np.array([*args[:-1],A_N])*all_f_n.T,axis=1)
				return F_star

			fit_limited[0][0]=0
			plt.figure(1000);#plt.plot(r_int,mother_f_star(all_f_n)(fake_x,*fit[0]),label='line='+str(line)+' N='+str(N)+' non limited')
			plt.plot(r_int,np.pi*mother_f_star_limited(all_f_n)(fake_x,*fit_limited[0]),'C'+str(line),label='Fourier line='+str(line)+' N='+str(N)+' limited')



			gauss = lambda x,A,sig,x0: A*np.exp(-((x-x0)/sig)**2)
			AInvgauss = lambda x,A,sig,x0: gauss(x,A/(sig/np.sqrt(np.pi)),sig,0)
			gauss3_locked_2 = lambda x, A1, sig1, A2, c_sig2, c_A3, c_sig3: gauss(x, A1, sig1, 0) + gauss(x,A2,c_sig2*sig1,0) + gauss(x, c_A3*c_sig3*(A1 +A2/c_sig2), c_sig3*sig1, 0)
			gauss3_locked_2_bias = lambda x, A1, sig1, A2, c_sig2, c_A3, c_sig3, m, q: gauss(x, A1, sig1, 0) + gauss(x,A2,c_sig2*sig1,0) + gauss(x, c_A3*c_sig3*(A1 +A2/c_sig2), c_sig3*sig1, 0) + q + x * m
			AInvgauss3_locked_2 = lambda x, A1, sig1, A2, c_sig2, c_A3, c_sig3: AInvgauss(x, A1, sig1, 0) + AInvgauss(x, A2,c_sig2*sig1,0) + AInvgauss(x, c_A3*c_sig3*(A1 +A2/c_sig2), c_sig3*sig1, 0)
			yy = all_fits[i_time, :, line]

			ym = np.nanmax(yy)
			inner_dip_sig = 0.15
			bds = [[0, 0, 0, 1, -1, 1e-10],[max(yy)*100, np.inf, np.max(yy)*100, np.inf, 0, 1]]
			p0 = [ym *0.9, 6*dx, ym * .05, 300, -0.25, inner_dip_sig]
			fit = curve_fit(gauss3_locked_2, xx_int, yy, p0, maxfev=1000000, bounds=np.array(bds))
			while (AInvgauss3_locked_2(0, *fit[0]) < 0 and inner_dip_sig < 0.3):
				inner_dip_sig+=0.015
				p0 = [*fit[0][:-1],inner_dip_sig]
				fit = curve_fit(gauss3_locked_2, xx_int, yy, p0, maxfev=1000000, bounds=np.array(bds))
			plt.figure(1000);#plt.plot(r_int,mother_f_star(all_f_n)(fake_x,*fit[0]),label='line='+str(line)+' N='+str(N)+' non limited')
			plt.plot(r_int,AInvgauss3_locked_2(r_int,*fit[0]),'C'+str(line)+'--',label='Gaussian line non biased='+str(line)+' N='+str(N))
			plt.grid()
			plt.legend(loc='best')
			plt.figure(line);plt.plot(xx_int,gauss3_locked_2(xx_int,*fit[0]),'--',label='Gaussian line non biased='+str(line)+' N='+str(N))
			plt.legend(loc='best')

			ym = np.nanmax(yy)
			inner_dip_sig = 0.15
			bds = [[0, 0, 0, 1, -1, 1e-10,-np.inf,-np.inf],[max(yy)*100, np.inf, np.max(yy)*100, np.inf, 0, 1,np.inf,np.min(yy)]]
			p0 = [ym *0.9, 6*dx, ym * .05, 300, -0.25, inner_dip_sig,0,np.min(yy)]
			fit = curve_fit(gauss3_locked_2_bias, xx_int, yy, p0, maxfev=1000000, bounds=np.array(bds))
			while (AInvgauss3_locked_2(0, *fit[0][:-2]) < 0 and inner_dip_sig < 0.3):
				inner_dip_sig+=0.015
				p0 = [*fit[0][:-1],inner_dip_sig,0,np.min(yy)]
				fit = curve_fit(gauss3_locked_2_bias, xx_int, yy, p0, maxfev=1000000, bounds=np.array(bds))
			plt.figure(1000);#plt.plot(r_int,mother_f_star(all_f_n)(fake_x,*fit[0]),label='line='+str(line)+' N='+str(N)+' non limited')
			plt.plot(r_int,AInvgauss3_locked_2(r_int,*fit[0][:-2]),'C'+str(line)+':',label='Gaussian line biased='+str(line)+' N='+str(N))
			plt.grid()
			plt.legend(loc='best')
			plt.figure(line);plt.plot(xx_int,gauss3_locked_2_bias(xx_int,*fit[0]),':',label='Gaussian line biased='+str(line)+' N='+str(N))
			plt.legend(loc='best')

			plt.pause(0.01)

			#  At the end: the biased fitting works, but was is the point of that or it's physical justification?
			# why there is a linear rediation added on top of the signal?
			# it might be reflections, but for now I don't check for that. the wide gaussian is there, anyway, to pick a flattish signal

	elif True:	# method using pyAbel suggested by Gijs

		all_fits_fitted = np.zeros_like(all_fits_ss)
		all_fits_residuals = np.zeros_like(all_fits_ss)
		inverted_profiles = np.zeros((nLine,number_of_radial_divisions))
		inverted_profiles_sigma = np.zeros((nLine,number_of_radial_divisions))
		fitCoefs = np.zeros((np.shape(all_fits_ss)[-1],2))

		interp_range_r = dx
		# weights_r = (xx-force_glogal_center)/interp_range_r
		all_fits_ss_interp = np.zeros((number_of_radial_divisions,np.shape(all_fits_ss)[-1]))
		all_fits_ss_sigma_interp = np.zeros((number_of_radial_divisions,np.shape(all_fits_ss)[-1]))
		r_new = xx - force_glogal_center
		for iR in range(np.shape(all_fits_ss)[-1]):
			temp = all_fits_ss[:,iR]
			temp[temp==0] = np.nan
			max_coord = min(np.max(r_new[np.logical_not(np.isnan(temp))]),np.abs(np.min(r_new[np.logical_not(np.isnan(temp))])))
			right_high = np.nanmean(temp[np.logical_and(r_new<=max_coord+dx/10,r_new>=max_coord-4*dx-dx/10)])
			right_high_r = np.nanmean(r_new[np.logical_and(r_new<=max_coord+dx/10,r_new>=max_coord-4*dx-dx/10)])
			left_high = np.nanmean(temp[np.logical_and(r_new>=-max_coord-dx/10,r_new<=-max_coord+4*dx+dx/10)])
			left_high_r = np.nanmean(r_new[np.logical_and(r_new>=-max_coord-dx/10,r_new<=-max_coord+4*dx+dx/10)])
			fit = np.polyfit([left_high_r,right_high_r],[left_high,right_high],1)
			fitCoefs[iR] = fit
			to_fit = all_fits_ss[:,iR]-np.polyval([fit[0],0],r_new)-np.min(np.polyval([fit[0],0],r_new))
			for i_r, value_r in enumerate(np.arange(number_of_radial_divisions)*dx):
				if np.sum(np.abs(np.abs(r_new) - value_r) < interp_range_r) == 0:
					continue
				selected_values = np.abs(np.abs(r_new) - value_r) < interp_range_r
				selected_values[np.isnan(to_fit)] = False
				# weights = np.abs(weights_r[selected_values]+1e-6)
				# weights = np.ones_like(weights_r[selected_values])
				weights = np.ones((np.sum(selected_values)))
				if np.sum(selected_values) == 0:
					continue
				all_fits_ss_interp[i_r,iR] = np.sum(to_fit[selected_values]*weights / all_fits_ss_sigma[selected_values,iR]) / np.sum(weights / all_fits_ss_sigma[selected_values,iR])
				if False: 	# suggestion from Daljeet: use the "worst case scenario" in therms of uncertainty
					merge_dTe_prof_multipulse_interp[i_r] = 1/(np.sum(1 / merge_dTe_multipulse[selected_values]))*(np.sum( ((merge_Te_prof_multipulse_interp[i_r]-merge_Te_prof_multipulse[selected_values])/merge_dTe_multipulse[selected_values])**2 )**0.5)
				else:
					all_fits_ss_sigma_interp_temp = 1/(np.sum(1 / all_fits_ss_sigma[selected_values,iR]))*(np.sum( ((all_fits_ss_interp[i_r,iR]-to_fit[selected_values])/all_fits_ss_sigma[selected_values,iR])**2 )**0.5)
					all_fits_ss_sigma_interp[i_r,iR] = max(all_fits_ss_sigma_interp_temp,(np.max(to_fit[selected_values])-np.min(to_fit[selected_values]))/2 )
				all_fits_ss_interp[np.isnan(all_fits_ss_interp)]=0
				all_fits_ss_sigma_interp[np.isnan(all_fits_ss_sigma_interp)]=0

			inverted_profiles[ iR] = abel.dasch.three_point_transform(all_fits_ss_interp[:,iR],dr=dx,direction='inverse')
			inverted_profiles[ iR][inverted_profiles[ iR]<0]=0

			# it works, and it is smooth at the centre, that is good, BUT the signal can go negative. it happens far from the centre and for high lines, so it's ok
			# this function does not deal with uncertainty. but this is based on the central difference derivative

			for i_r in np.abs(np.arange(1-number_of_radial_divisions,1)):
				if i_r==number_of_radial_divisions-1:
					inverted_profiles_sigma[ iR,i_r] = all_fits_ss_sigma_interp[i_r,iR]
				else:
					inverted_profiles_sigma[ iR,i_r] = all_fits_ss_sigma_interp[i_r,iR]+inverted_profiles_sigma[ iR,i_r+1]


		# plt.figure();plt.plot(xx-force_glogal_center,all_fits_ss[:,0]);plt.plot(np.arange(number_of_radial_divisions)*dx,all_fits_ss_interp[:,0]);plt.grid();plt.pause(0.01)
		# np.abs(xx-force_glogal_center).argmin()
		# all_fits_ss[:,0]
		# gna=abel.dasch.three_point_transform(all_fits_ss_interp[:,0],dr=dx,direction='inverse')
		# plt.figure();plt.plot(np.arange(number_of_radial_divisions)*dx,gna);plt.grid();plt.pause(0.01)

		# I might think of a way to obtain this
		# all_fits_fitted[ :, iR] = gauss3_locked_2_bias(xx,*fit[0])
		# all_fits_residuals[ :, iR] = (temp_all_fits - all_fits_fitted[ :, iR])

		path = '/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)
		if not os.path.exists(path):
			os.makedirs(path)

		color = ['b', 'r', 'm', 'y', 'g', 'c', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise']
		plt.figure(figsize=(20, 10))
		for iR in range(nLine):
			plt.errorbar(xx,all_fits_ss[:,iR],yerr=all_fits_ss_sigma[:,iR],color=color[iR],label='line '+str(iR+4),linewidth=0.7)
			plt.errorbar(np.arange(number_of_radial_divisions)*dx+force_glogal_center,all_fits_ss_interp[:,iR]+np.polyval([fitCoefs[iR,0],0],np.arange(number_of_radial_divisions)*dx)-np.min(np.polyval([fitCoefs[iR,0],0],np.arange(number_of_radial_divisions)*dx)),yerr=all_fits_ss_sigma_interp[:,iR],linestyle='--',color=color[iR],label='interpolated line '+str(iR+4),linewidth=0.7)
			# plt.plot(1000*xx, all_fits_fitted[:,iR],color=color[iR],linestyle='--',linewidth=0.4)
			# # plt.plot(1000*xx, np.abs(all_fits_residuals[:,iR]),color=color[iR],linestyle=':',linewidth=0.4)
			plt.plot(xx, np.polyval(fitCoefs[iR],r_new),color=color[iR],linestyle=':',linewidth=0.4)
			plt.plot([force_glogal_center,force_glogal_center],[np.nanmin(all_fits_ss),np.nanmax(all_fits_ss)],color=color[iR],linestyle='-.')
		plt.title("Steady state line integrated profiles\n'--'=fitting, ':'=bias\n mean centre "+str(np.around(force_glogal_center*1000, decimals=1))+'m')
		plt.xlabel('LOS location [m]')
		plt.ylabel('Line integrated brightness [W m^-2 sr^-1]')
		plt.legend(loc='best',fontsize='x-small')
		plt.semilogy()
		plt.grid(b=True, which='both')
		plt.ylim(max(1e-4,np.nanmin(all_fits_ss)*0.9),np.nanmax(all_fits_ss)*1.1)
		plt.savefig('%s/SS_pure_abel_line_profile.eps' % (path), bbox_inches='tight')
		plt.close()

		plt.figure(figsize=(20, 10))
		for iR in range(nLine):
			plt.errorbar(np.arange(number_of_radial_divisions)*dx,inverted_profiles[iR],yerr=inverted_profiles_sigma[iR],color=color[iR],label='line '+str(iR+4))
		plt.title('Steady state line profiles \n mean centre '+str(np.around(force_glogal_center*1000, decimals=1))+'mm')
		plt.xlabel('radius [mm]')
		plt.ylabel('Emissivity [W m^-3 sr^-1]')
		plt.semilogy()
		plt.legend(loc='best',fontsize='x-small')
		plt.ylim(bottom=np.min(inverted_profiles[-1]),top=np.max(inverted_profiles[0]))
		plt.savefig('%s/SS_pure_abel_abel_inverted_line_profile.eps' % (path), bbox_inches='tight')
		plt.close()

		plt.figure(figsize=(20, 10))
		for iR in range(nLine-1):
			plt.plot(np.arange(number_of_radial_divisions)*dx,inverted_profiles[iR+1]/inverted_profiles[iR],color=color[iR],label='line ratio '+str(iR+4+1)+'/'+str(iR+4))
		plt.title('Steady state line ratio profiles \n mean centre '+str(np.around(force_glogal_center*1000, decimals=1))+'mm')
		plt.xlabel('radius [mm]')
		plt.ylabel('Line ratio [au]')
		plt.legend(loc='best')
		# plt.ylim(bottom=0)
		plt.semilogy()
		plt.grid(b=True, which='both')
		# plt.ylim(bottom=1e-2)
		plt.savefig('%s/SS_pure_abel_abel_inverted_line_ratio_profile.eps' % (path), bbox_inches='tight')
		plt.close()
		# np.save('%s/SS_profileFits.npy' % (path), fitCoefs)
		np.save('%s/SS_pure_abel_inverted_profiles.npy' % (path), inverted_profiles)
		np.save('%s/SS_pure_abel_inverted_profiles_sigma.npy' % (path), inverted_profiles_sigma)


def doLateralfit_time_tependent_LOS_overlapping(df_settings,all_fits,all_fits_sigma, merge_ID_target, new_timesteps,dx,xx,r,N,force_glogal_center):
	# forward model that I had inspiration from from a talk on Tom Farley method for thomographic inversion. instead of doing an actual inversion I only fit the expected brightness due to each intersection of plasma ring and LOS. in this way I can factor in also the overlapping between LOS
	import scipy.integrate as integrate
	import os
	import pandas as pd
	import numpy as np
	from scipy.optimize import curve_fit
	import matplotlib.pyplot as plt
	import copy as cp
	from uncertainties import ufloat,unumpy,correlated_values
	from uncertainties.unumpy import exp,nominal_values,std_devs
	from scipy.optimize import least_squares
	from scipy.linalg.decomp_svd import svd

	path = '/home/ffederic/work/Collaboratory/test/experimental_data/merge'+str(merge_ID_target)
	nLine = np.shape(all_fits)[-1]
	first_time = np.min(new_timesteps)
	last_time = np.max(new_timesteps)
	time_res = np.mean(np.diff(new_timesteps))


	print('NOTE: A METHOD TO FORCE EMISSIVITY REGULARISATION MUST BE ADDED TO THE FULL CONVOLUTE INVERSION')

	dr_new = 1.5*1.338461/1000	# mm, just an example. this is same as TS
	radius_LOS = 1.9/1000	# mm
	new_intervals = int(25/1000//dr_new)-1	# just because light fall off quickly after ~x=35mm
	r_max_new = (new_intervals)*dr_new	# mm
	r_new = np.linspace(0,r_max_new,num=new_intervals+1)
	r_actual_new = r_new[:-1]+np.mean(np.diff(r_new))/2


	emissivity_new = np.zeros((np.shape(all_fits)[0],new_intervals,np.shape(all_fits)[-1]))
	brightness_new = np.zeros_like(all_fits)
	weights = np.zeros((len(xx),new_intervals))

	# plt.figure()
	for i_int_r in range(new_intervals):
		# plt.figure()
		Rmin = i_int_r*dr_new
		Rmax = (i_int_r+1)*dr_new
		for i_x,x in enumerate(xx-force_glogal_center):
			if ((x-radius_LOS)>Rmax or (x+radius_LOS)<-Rmax):
				continue
			plasma_R = np.linspace(max(x-radius_LOS,-Rmax),min(x+radius_LOS,Rmax),num=40)
			# integral = 4 * ((Rmax**2 - plasma_R**2)*(radius_LOS**2-(x-plasma_R)**2))**0.5
			integral = 4 * ((Rmax**2 - plasma_R**2)*(radius_LOS**2-(x-plasma_R)**2))**0.5
			integral[np.isnan(integral)] = 0
			temp = np.trapz(integral,plasma_R)
			if ((x-radius_LOS)<Rmin and (x+radius_LOS)>-Rmin):
				plasma_R = np.linspace(max(x-radius_LOS,-Rmin),min(x+radius_LOS,Rmin),num=40)
				integral = 4 * ((Rmin**2 - plasma_R**2)*(radius_LOS**2-(x-plasma_R)**2))**0.5
				integral[np.isnan(integral)] = 0
				temp -= np.trapz(integral,plasma_R)
			# print([i_int_r,i_x,temp])
			weights[i_x,i_int_r]=temp
		# 	plt.plot(x,temp,'x')
		# plt.pause(0.01)

	# plt.figure()
	# plt.imshow(weights)
	# plt.pause(0.01)

	def forward_model(weights):
		def forward_model_int(*arg):
			emissivity = arg[1:]
			brightness = np.sum(weights*emissivity,axis=-1)
			return brightness
		return forward_model_int

	# guess = np.ones((np.shape(weights)[1]))*1e10
	# bds = [np.zeros((np.shape(weights)[1])),np.ones((np.shape(weights)[1]))*np.inf]
	# fit = curve_fit(forward_model(weights),[0],all_fits[20,:,0],p0=guess,bounds=bds)
	# plt.figure()
	# plt.plot(xx-force_glogal_center,all_fits[20,:,0])
	# plt.plot(xx-force_glogal_center,forward_model(weights)(0,*fit[0]))
	# plt.pause(0.01)
	#
	# plt.figure()
	# plt.plot(r_new[:-1]+np.mean(np.diff(r_new)),fit[0])
	# plt.pause(0.01)

	guess = np.ones((np.shape(weights)[1]))*1e10
	bds = [np.zeros((np.shape(weights)[1])),np.ones((np.shape(weights)[1]))*np.inf]
	for iSet in range(np.shape(all_fits)[0]):
		for iR in range(np.shape(all_fits)[-1]):
			fit = curve_fit(forward_model(weights),[0],all_fits[iSet,:,iR],p0=guess,bounds=bds,xtol=1e-11)
			emissivity_new[iSet,:,iR]=fit[0]
			brightness_new[iSet,:,iR]=forward_model(weights)(0,*fit[0])

	# plt.figure()
	# plt.imshow(emissivity_new[:,:,3],'rainbow',extent=[1000*min(r_new),1000*max(r_new),new_timesteps[-1],new_timesteps[0]],aspect=20)
	# plt.colorbar()
	# plt.pause(0.01)
	#
	# plt.figure()
	# plt.imshow(brightness_new[:,:,0])
	# plt.colorbar()
	# plt.pause(0.01)


	if False:	# in this model I have no regularisation enforced

		def forward_model_full(*arg):
			emissivity = arg[2:]
			force_glogal_center = arg[1]
			weights = np.zeros((len(xx),new_intervals))
			for i_int_r in range(new_intervals):
				# plt.figure()
				Rmin = i_int_r*dr_new
				Rmax = (i_int_r+1)*dr_new
				for i_x,x in enumerate(xx-force_glogal_center):
					if ((x-radius_LOS)>Rmax or (x+radius_LOS)<-Rmax):
						continue
					plasma_R = np.linspace(max(x-radius_LOS,-Rmax),min(x+radius_LOS,Rmax),num=40)
					# integral = 4 * ((Rmax**2 - plasma_R**2)*(radius_LOS**2-(x-plasma_R)**2))**0.5
					integral = 4 * ((Rmax**2 - plasma_R**2)*(radius_LOS**2-(x-plasma_R)**2))**0.5
					integral[np.isnan(integral)] = 0
					temp = np.trapz(integral,plasma_R)
					if ((x-radius_LOS)<Rmin and (x+radius_LOS)>-Rmin):
						plasma_R = np.linspace(max(x-radius_LOS,-Rmin),min(x+radius_LOS,Rmin),num=40)
						integral = 4 * ((Rmin**2 - plasma_R**2)*(radius_LOS**2-(x-plasma_R)**2))**0.5
						integral[np.isnan(integral)] = 0
						temp -= np.trapz(integral,plasma_R)
					# print([i_int_r,i_x,temp])
					weights[i_x,i_int_r]=temp
				# 	plt.plot(x,temp,'x')
				# plt.pause(0.01)
			brightness = np.sum(weights*emissivity,axis=-1)
			return brightness

		# guess = [force_glogal_center,*emissivity_new[20,:,0]]
		# bds = [[force_glogal_center-2*dx,*np.zeros_like(guess[1:])],[force_glogal_center+2*dx,*np.ones_like(guess[1:])*np.inf]]
		# fit = curve_fit(forward_model_full,[0],all_fits[20,:,0],p0=guess,bounds=bds)
		# plt.figure()
		# plt.plot(xx-fit[0][0],all_fits[20,:,0])
		# plt.plot(xx-fit[0][0],forward_model_full(0,*fit[0]))
		# plt.pause(0.01)
		#
		# plt.figure()
		# plt.plot(r_new[:-1]+np.mean(np.diff(r_new)),fit[0][1:])
		# plt.pause(0.01)

		all_fits_fitted = np.zeros_like(all_fits)
		all_fits_residuals = np.zeros_like(all_fits)
		inverted_profiles = np.zeros((len(all_fits),nLine,np.shape(weights)[1]))
		inverted_profiles_sigma = np.zeros((len(all_fits),nLine,np.shape(weights)[1]))
		profile_centre = np.zeros((np.shape(all_fits)[0],np.shape(all_fits)[-1]))
		bds = [[force_glogal_center-2*dx,*np.zeros((np.shape(weights)[1]))],[force_glogal_center+2*dx,*np.ones((np.shape(weights)[1]))*np.inf]]
		for iSet in range(np.shape(all_fits)[0]):

			if  (np.min(np.abs(new_timesteps[iSet] - np.array([-0.2,0.2,0.55,1])))<=0.06):
				print('will plot the spectral fit at time '+str(new_timesteps[iSet]))
				if not os.path.exists(path+'/abel_inversion_example'):
					os.makedirs(path+'/abel_inversion_example')
				fig, ax = plt.subplots( np.shape(all_fits)[-1],2,figsize=(15, 25), squeeze=True)
				fig.suptitle('Abel inversion for time %.3g ,' %new_timesteps[iSet] +'ms\n   same_centre_every_line=%1s,same_centre_all_time=%1s,max_central_depression_amplitude=%3g,min_central_depression_sigma=%3g,force_glogal_center=%3g  ' %(same_centre_every_line,same_centre_all_time,max_central_depression_amplitude,min_central_depression_sigma,force_glogal_center) )

			for iR in range(np.shape(all_fits)[-1]):

				yy = all_fits[iSet, :, iR]
				yy[np.isnan(yy)]=0
				yy_sigma = all_fits_sigma[iSet, :, iR]
				yy_sigma[np.isnan(yy)]=np.nanmax(yy_sigma)
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
					xx_good = xx
					yy_good = yy
					yy_good_sigma = yy_sigma
				else:
					if borders[0]!=0:	# this should not be necessary but I cannot avoid a smear of composed_array when dealing with data missing in lower lines
						borders[0]+=5
					xx_good = xx[borders[0]:borders[-1]]
					yy_good = yy[borders[0]:borders[-1]]
					yy_good_sigma = yy_sigma[borders[0]:borders[-1]]


				guess = [force_glogal_center,*emissivity_new[iSet,:,iR]]
				x_scale = [force_glogal_center,*emissivity_new[iSet,:,iR]]
				# fit = curve_fit(forward_model_full,[0],all_fits[iSet,:,iR],p0=guess,bounds=bds,gtol=1e-10)
				fit = curve_fit(forward_model_full,[0], yy_good, p0=guess,sigma=yy_good_sigma, absolute_sigma=True, maxfev=1000000, xtol=1e-15, gtol=1e-15, bounds=np.array(bds),x_scale=x_scale,ftol=1e-15)

				# emissivity_new[iSet,:,iR]=fit[0][1:]
				# brightness_new[iSet,:,iR]=forward_model_full(0,*fit[0])
				# profile_centres[iSet,iR]=fit[0][0]
				temp = correlated_values(fit[0][1:],fit[1][1:,1:])
				profile_centre[iSet,iR]=fit[0][0]
				inverted_profiles[iSet, iR] = nominal_values(temp)
				inverted_profiles_sigma[iSet, iR] = std_devs(temp)
				all_fits_fitted[iSet, :, iR] = forward_model_full(0,*fit[0])
				all_fits_residuals[iSet, :, iR] = (temp_all_fits - all_fits_fitted[iSet, :, iR])#/np.abs(np.nanmax(all_fits_fitted[iSet, :, iR]))

				if  (np.min(np.abs(new_timesteps[iSet] - np.array([-0.2,0.2,0.55,1])))<=0.06):
					im = ax[iR,0].errorbar(1000*xx_good, yy_good,yerr=yy_good_sigma,color='c',label='line integrated data')
					im = ax[iR,0].plot([1000*profile_centre[iSet,iR],1000*profile_centre[iSet,iR]],[np.min([yy_good,all_fits_fitted[iSet, :, iR]]),np.max([yy_good,all_fits_fitted[iSet, :, iR]])],'k--')
					im = ax[iR,0].plot(1000*xx, all_fits_fitted[iSet, :, iR],'m',label='fit')
					ax[iR,0].set_ylabel('brightness [W m-2 sr-1]')
					ax[iR,0].grid()
					if iR==0:
						ax[iR,0].set_title('cyan=line int data, magenta=fit\nline n='+str(iR+4)+'->2')
					else:
						ax[iR,0].set_title('line n='+str(iR+4)+'->2')
					if iR==(np.shape(all_fits)[-1]-1):
						ax[iR,0].set_xlabel('x [mm]')
					else:
						ax[iR,0].set_xticks([])

					im = ax[iR,1].errorbar(1000*r, nominal_values(temp),yerr=std_devs(temp),color='b',label='inverted data')
					im = ax[iR,1].plot(1000*r, std_devs(temp),'r',label='error')
					ax[iR,1].set_ylabel('emissivity [W m^-3 sr^-1]')
					ax[iR,1].yaxis.set_label_position('right')
					ax[iR,1].yaxis.tick_right()
					ax[iR,1].grid()
					if iR==0:
						ax[iR,1].set_title('blue=inverted prof, red=error\nfit='+str(''.join(['%.3g+/-%.3g, ' %(_,__) for _,__ in zip(fit[0],np.sqrt(np.diag(fit[1])))])))
					else:
						ax[iR,1].set_title('fit='+str(''.join(['%.3g+/-%.3g, ' %(_,__) for _,__ in zip(fit[0],np.sqrt(np.diag(fit[1])))])))
					if iR==(np.shape(all_fits)[-1]-1):
						ax[iR,1].set_xlabel('r [mm]')
					else:
						ax[iR,1].set_xticks([])
			if  (np.min(np.abs(new_timesteps[iSet] - np.array([-0.2,0.2,0.55,1])))<=0.06):
				plt.savefig(path+'/abel_inversion_example/abe_inversion_time_' + str(np.around(new_timesteps[iSet], decimals=3)) +'_ms.eps', bbox_inches='tight')
				plt.close('all')

	else:	# this isncludes a cost function		UNTESTED


		def forward_model_regularised(reg_param,yy_good):
			def forward_model_full(*arg):
				emissivity = arg[1:]
				force_glogal_center = arg[0]
				weights = np.zeros((len(xx),new_intervals))
				for i_int_r in range(new_intervals):
					# plt.figure()
					Rmin = i_int_r*dr_new
					Rmax = (i_int_r+1)*dr_new
					for i_x,x in enumerate(xx-force_glogal_center):
						if ((x-radius_LOS)>Rmax or (x+radius_LOS)<-Rmax):
							continue
						plasma_R = np.linspace(max(x-radius_LOS,-Rmax),min(x+radius_LOS,Rmax),num=40)
						# integral = 4 * ((Rmax**2 - plasma_R**2)*(radius_LOS**2-(x-plasma_R)**2))**0.5
						integral = 4 * ((Rmax**2 - plasma_R**2)*(radius_LOS**2-(x-plasma_R)**2))**0.5
						integral[np.isnan(integral)] = 0
						temp = np.trapz(integral,plasma_R)
						if ((x-radius_LOS)<Rmin and (x+radius_LOS)>-Rmin):
							plasma_R = np.linspace(max(x-radius_LOS,-Rmin),min(x+radius_LOS,Rmin),num=40)
							integral = 4 * ((Rmin**2 - plasma_R**2)*(radius_LOS**2-(x-plasma_R)**2))**0.5
							integral[np.isnan(integral)] = 0
							temp -= np.trapz(integral,plasma_R)
						# print([i_int_r,i_x,temp])
						weights[i_x,i_int_r]=temp
					# 	plt.plot(x,temp,'x')
					# plt.pause(0.01)
				brightness = np.sum(weights*emissivity,axis=-1)
				second_der = np.array([brightness[0]-2*brightness[1]+brightness[2],*(brightness[2:]-2*brightness[1:-1]+brightness[:-2]),brightness[-1]-2*brightness[-2]+brightness[-3]])
				return (brightness-yy_good)**2 + reg_param**2 * (np.array(second_der)**2)
			return forward_model_full

		values = merge_values_medianfiltered[:,z]	# modified 29/03/2020 to do the filtering only once
		values_sigma = merge_values_sigma_medianfiltered[:,z]
		# guess[-1]=np.nanmean(values)
		guess = [1,1,1,1,1,1,np.nanmean(values)]
		fit = least_squares(forward_model_regularised(reg_param,yy_good),guess,bounds=bds,max_nfev=100000,ftol=1e-15,gtol=1e-15,xtol=1e-15)
		_, s, VT = svd(fit.jac, full_matrices=False)	# this method is from https://stackoverflow.com/questions/40187517/getting-covariance-matrix-of-fitted-parameters-from-scipy-optimize-least-squares
		threshold = np.finfo(float).eps * max(fit.jac.shape) * s[0]
		s = s[s > threshold]
		VT = VT[:s.size]
		pcov = np.dot(VT.T / s**2, VT)



	profile_centres = cp.deepcopy(profile_centre)
	if not os.path.exists(path):
		os.makedirs(path)
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(inverted_profiles[:,iR],'rainbow',vmax=np.nanmax(np.mean(np.sort(inverted_profiles[:,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(r_actual_new),1000*max(r_actual_new),last_time,first_time],aspect=50)
		plt.title('Line '+str(iR+4))
		plt.xlabel('radius [mm]')
		plt.ylabel('time [ms]')
		plt.colorbar().set_label('Emissivity [W m^-3 sr^-1]')
		plt.savefig('%s/abel_inverted_profile%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	initial_time = int((np.abs(new_timesteps)).argmin())
	final_time = int((np.abs(new_timesteps -2)).argmin())
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(inverted_profiles_sigma[:,iR],'rainbow',vmax=np.nanmax(np.mean(np.sort(inverted_profiles_sigma[:,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(r_actual_new),1000*max(r_actual_new),last_time,first_time],aspect=50)
		plt.title('Line '+str(iR+4))
		plt.xlabel('radius [mm]')
		plt.ylabel('time [ms]')
		plt.colorbar().set_label('Emissivity [W m^-3 sr^-1]')
		plt.savefig('%s/abel_inverted_profile_sigma%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	initial_time = int((np.abs(new_timesteps)).argmin())
	final_time = int((np.abs(new_timesteps -2)).argmin())
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(inverted_profiles[initial_time:final_time+1,iR],'rainbow',vmin=0,vmax=np.nanmax(np.mean(np.sort(inverted_profiles[initial_time:final_time+1,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(r_actual_new),1000*max(r_actual_new),new_timesteps[final_time],new_timesteps[initial_time]],aspect=50)
		plt.title('Line '+str(iR+4))
		plt.xlabel('radius [mm]')
		plt.ylabel('time [ms]')
		plt.colorbar().set_label('Emissivity [W m^-3 sr^-1]')
		plt.savefig('%s/abel_inverted_profile%s_short.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(inverted_profiles[initial_time:final_time+1,iR]/inverted_profiles_sigma[initial_time:final_time+1,iR],'rainbow',vmin=0,vmax=np.nanmax(np.mean(np.sort(inverted_profiles[initial_time:final_time+1,iR]/inverted_profiles_sigma[initial_time:final_time+1,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(r_actual_new),1000*max(r_actual_new),new_timesteps[final_time],new_timesteps[initial_time]],aspect=50)
		plt.title('Line '+str(iR+4))
		plt.xlabel('radius [mm]')
		plt.ylabel('time [ms]')
		plt.colorbar().set_label('Signal/Noise [au]')
		plt.savefig('%s/abel_inverted_profile_SNR%s_short.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(inverted_profiles_sigma[initial_time:final_time+1,iR],'rainbow',vmin=0,vmax=np.nanmax(np.mean(np.sort(inverted_profiles_sigma[initial_time:final_time+1,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(r_actual_new),1000*max(r_actual_new),new_timesteps[final_time],new_timesteps[initial_time]],aspect=50)
		plt.title('Line '+str(iR+4))
		plt.xlabel('radius [mm]')
		plt.ylabel('time [ms]')
		plt.colorbar().set_label('Emissivity [W m^-3 sr^-1]')
		plt.savefig('%s/abel_inverted_profile_sigma%s_short.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		max_value = np.median(np.max(inverted_profiles[:6,iR], axis=1))
		plt.imshow(inverted_profiles[:,iR],'rainbow',vmin=0,vmax=max_value,extent=[1000*min(r_actual_new),1000*max(r_actual_new),last_time,first_time],aspect=20)
		plt.title('Line '+str(iR+4))
		plt.xlabel('radius [mm]')
		plt.ylabel('time [ms]')
		plt.colorbar().set_label('Emissivity [W m^-3 sr^-1]')
		plt.savefig('%s/abel_inverted_profile_SS_%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(all_fits[:,:,iR],'rainbow',vmin=0,vmax=np.nanmax(np.mean(np.sort(all_fits[:,:,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(xx),1000*max(xx),last_time,first_time],aspect=20)
		plt.title('Line '+str(iR+4)+'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centres[:,iR],axis=(0)), decimals=1))+'mm) \n used black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
		plt.xlabel('LOS location [mm]')
		plt.ylabel('time [ms]')
		plt.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), 'k', linewidth=0.4,label='used centre')
		plt.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), '--r', linewidth=0.4, label='local centre')
		plt.colorbar().set_label('Line integrated brightness [W m^-2 sr^-1]')
		plt.savefig('%s/line_integrted_profile%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(all_fits[:,:,iR]/all_fits_sigma[:,:,iR],'rainbow',vmin=0,vmax=np.nanmax(np.mean(np.sort(all_fits[:,:,iR]/all_fits_sigma[:,:,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(xx),1000*max(xx),last_time,first_time],aspect=20)
		plt.title('Line '+str(iR+4)+'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centres[:,iR],axis=(0)), decimals=1))+'mm) \n used black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
		plt.xlabel('LOS location [mm]')
		plt.ylabel('time [ms]')
		plt.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), 'k', linewidth=0.4,label='used centre')
		plt.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), '--r', linewidth=0.4, label='local centre')
		plt.colorbar().set_label('Line integrated brightness [W m^-2 sr^-1]')
		plt.savefig('%s/line_integrted_profile_SNR%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(all_fits_sigma[:,:,iR],'rainbow',vmin=0,vmax=np.nanmax(np.mean(np.sort(all_fits_sigma[:,:,iR],axis=-1)[:,-10:-1],axis=-1)),extent=[1000*min(xx),1000*max(xx),last_time,first_time],aspect=20)
		plt.title('Line '+str(iR+4)+'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centres[:,iR],axis=(0)), decimals=1))+'mm) \n used black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
		plt.xlabel('LOS location [mm]')
		plt.ylabel('time [ms]')
		plt.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), 'k', linewidth=0.4,label='used centre')
		plt.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), '--r', linewidth=0.4, label='local centre')
		plt.colorbar().set_label('Line integrated brightness sigma[W m^-2 sr^-1]')
		plt.savefig('%s/line_integrted_profile_sigma%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(all_fits_fitted[:,:,iR],'rainbow',extent=[1000*min(xx),1000*max(xx),last_time,first_time],aspect=20)
		plt.title('Fitting of line '+str(iR+4)+'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centres[:,iR],axis=(0)), decimals=1))+'mm) \n used black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
		plt.xlabel('LOS location [mm]')
		plt.ylabel('time [ms]')
		plt.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), 'k', linewidth=0.4,label='used centre')
		plt.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), '--r', linewidth=0.4, label='local centre')
		plt.colorbar().set_label('Line integrated brightness [W m^-2 sr^-1]')
		plt.savefig('%s/line_integrted_profile_fitting%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		plt.imshow(all_fits_residuals[:,:,iR],'rainbow',extent=[1000*min(xx),1000*max(xx),last_time,first_time],aspect=20)
		plt.title('Fitting residuals of line '+str(iR+4)+'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centres[:,iR],axis=(0)), decimals=1))+'mm) \n used black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
		plt.xlabel('LOS location [mm]')
		plt.ylabel('time [ms]')
		plt.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), 'k', linewidth=0.4,label='used centre')
		plt.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), '--r', linewidth=0.4, label='local centre')
		plt.colorbar().set_label('Deviation from fit [au]')# over maximum fitted value [au]')
		plt.savefig('%s/line_integrted_profile_fitting_residuals%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	for iR in range(nLine):
		plt.figure(figsize=(10, 20))
		max_value = np.median(np.max(all_fits[:6,:,iR], axis=1))
		plt.imshow(all_fits[:,:,iR],'rainbow',vmin=0,vmax=max_value,extent=[1000*min(xx),1000*max(xx),last_time,first_time],aspect=20)
		plt.title('Line '+str(iR+4)+'\n local centre red (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm) \n used black (mean '+str(np.around(1000*np.nanmean(profile_centre[:,iR],axis=(0)), decimals=1))+'mm)')
		plt.xlabel('LOS location [mm]')
		plt.ylabel('time [ms]')
		plt.plot(1000*profile_centre[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), 'k', linewidth=0.4,label='used centre')
		plt.plot(1000*profile_centres[:,iR], np.linspace(first_time, last_time, (last_time -first_time) // time_res + 2), '--r', linewidth=0.4, label='local centre')
		plt.colorbar().set_label('Line integrated brightness [W m^-2 sr^-1]')
		plt.savefig('%s/line_integrted_profile_SS_%s.eps' % (path, iR+4), bbox_inches='tight')
		plt.close()
	np.save('%s/profileFits.npy' % (path), fitCoefs)
	np.save('%s/profile_centre.npy' % (path), profile_centre)
	np.save('%s/inverted_profiles.npy' % (path), inverted_profiles)
	np.save('%s/inverted_profiles_sigma.npy' % (path), inverted_profiles_sigma)

	return r_actual_new
