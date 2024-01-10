# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:46:02 2018

@author: akkermans
"""
#Requires pandas, numpy, and attached scripts

import pandas as pd
from nonOESData import getnonOES
from GetSpectrumGeometry import getGeom
from Calibrate import do_waveL_Calib,do_Intensity_Calib
from SpectralFit import doSpecFit
from numpy import save,load
#from tools import spectools

if False: # Collect non-OES data (requires database access)
    logdir = 'Log/'
    logfname = 'log.xlsx'
    
    df_log = pd.read_excel(logdir+logfname)
    df_log.columns = ['folder','date','time','trigNum','TSTrig','gratPos','t_exp','frames','ln/mm','gratTilt','I','B','Flow','pump%','Seed','p_n','Te','Ne','type','Dark','Calo0','Notes']
    df_log['folder'] = df_log['folder'].astype(str)
    
    (df_log,df_settings) = getnonOES(df_log)
    
    df_log.to_csv('Log/shots.csv')
    df_settings.to_csv('Log/settings.csv')
else:# Load non-OES data
    df_log = pd.read_csv('Log/shots.csv',index_col=0)
    df_settings = pd.read_csv('Log/settings.csv',index_col=0)
    
if True: # Calculate the geometrical factors of the misalignment of the spectrometer/camera system
    geom=getGeom(df_settings,df_log)
    
else: # Load the geometry values
    geom = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
    geom.loc[0] = [0.6254567079875826,-0.02752172070353893,33.996963562753045,272.16079733663486,223.1034637876741]

if True: # Calculate the sensor dispersion and sensitivity from calibration data
    waveLcoefs = do_waveL_Calib(df_settings,df_log,geom)
    
    logdir = 'Log/'
    calfname = 'calLog.xlsx'
    
    df_calLog = pd.read_excel(logdir+calfname)
    df_calLog.columns = ['folder','date','time','gratPos','t_exp','frames','ln/mm','gratTilt','I_det','type','exper','Dark']
    df_calLog['folder'] = df_calLog['folder'].astype(str)
    
    
    spherefname = r'Labsphere_Radiance.xls'
    df_sphere = pd.read_excel(logdir+spherefname)
    I_nom = df_sphere.columns[1].split()[-3:-1] # Nominal calibration current as string in A
    I_nom = float(I_nom[0])*10**(int(I_nom[1][-2:])+6) # float, \muA
    df_sphere.columns = ['waveL','RadRaw']
    df_sphere.units = ['nm','W cm-2 sr-1 nm-1','W m-2 sr-1 nm-1 \muA-1']
    df_sphere['Rad'] = df_sphere.RadRaw / I_nom * 1e4
    
    binnedSens = do_Intensity_Calib(df_calLog,df_sphere,geom,waveLcoefs)
    save('Calibrations/Sensitivity.npy',binnedSens) 
else: #Load the dispersion and sensitvity values
    waveLcoefs = [[-1.04328399e-07, -8.81793255e-02,  6.63244197e+02],[-1.04328399e-07, -8.81793255e-02,  5.08522477e+02]]
    binnedSens = load('Calibrations/Sensitivity.npy')
    
doSpecFit(df_settings,df_log,geom,waveLcoefs,binnedSens) #Do spectral fit and apply calibration, obtaining lateral distribution of radiance for each spectral line 