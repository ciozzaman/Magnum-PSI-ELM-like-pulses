# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:27:09 2018

@author: akkermans
"""

# Fetches 
# - plasma temperature and density from thomson scattering
# - target heat flux from calorimetry and
# - background neutral pressure in the target chamber.
# for all OES measurements.
# Exponential fits of pressure and heat flux are taken through the entire 
# time-range for which the relevant machine settings were applied.

def getnonOES(df_log):
    from codac.datastore import client as c
    from scipy.optimize import curve_fit
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    
    db = c.connect('magdat')
    
    expdec = lambda t,A0,A,tauinv: A0+A*np.exp(-t*tauinv) #Function with which pres and calori data will be fitted
    
    # Definition of database variables    
    v_TaChPres = db.root._Raw.TaChPresBara          # Pressure in target chamber - barametric gauge
    v_SourceH2 = db.root._Raw.Mfc2Setpoint          # H2 Gas flow through source
    v_SeedH2 = db.root._Raw.Mfc5Setpoint            # H2 Gas flow through seeding valve
    v_pump = db.root._Raw.Rp5TaChSpeed              # Target chamber roots pump speed
    v_I = db.root._Raw.SourceCurrentPs              # Cathode current 
    v_Calo = db.root._Raw.TargetCalTempDiffCorr     # Calorimetry in/out temperature difference
    v_Dump = db.root._Raw.BeamDumpDown              # Whether the beam dump is NOT blocking the plasma
    v_Water = db.root._Raw.TargetWflowTotal         # Cooling water flow to target holder
    trange = c.TimeRange(c.datetotime(c.datetime.datetime(*[int(x) for x in min(list(df_log.folder)).split('-')])), # Complete time range for whole experimental campaign
                         c.datetotime(c.datetime.datetime(*[int(x) for x in max(list(df_log.folder)).split('-')]+[23])))
    
    # Reading out of database variables
    (t_so,SourceH2) = v_SourceH2.read(trange,unit=c.PREF_UNIT)
    (t_se,SeedH2) = v_SeedH2.read(trange,unit=c.PREF_UNIT)
    (t_pu,pump) = v_pump.read(trange,unit=c.PREF_UNIT)
    (t_I,I) = v_I.read(trange,unit=c.PREF_UNIT)
    (t_du,dump) = v_Dump.read(trange,unit=c.PREF_UNIT)
    
    SourceH2 = np.around(SourceH2,1)
    SeedH2 = np.around(SeedH2,1)
    pump = np.around(pump,0)

    # Conversion and calibration factors for calorimetry
    CaloCalFac = 1.099      # Calibration factor for calorimetry using electrical heater
    CaloCalFac_er = 0.013
    Calo0_er = np.nanstd(list(set(df_log['Calo0'])))    # Random fluctuation level of the baseline of the calorimeter
    CaloConv = .997*4186/60    # Conversion factor from (Flow[l/min] * Delta T [K]) to Power
    
    # Fetches pressure and heat flux from calorimetry, and fits steady-state values
    for (i,d,t,typ) in zip(df_log.index.values,df_log['folder'],df_log['time'],df_log['type']):
        if not type(typ)==str and np.isnan(typ):
            # find the time frame for which seeding was constant
            t0 = c.datetotime(c.datetime.datetime(int(d[:4]),int(d[5:7]),int(d[8:10]),int(t[:2]),int(t[3:5]),int(t[6:])))
            t0_se, SeedH20 = v_SeedH2.getNext(t0,unit=c.PREF_UNIT)
            i0_se = list(t_se).index(t0_se)
            SeedH20 = SeedH2[i0_se]
            
            is_se = i0_se-list(SeedH2[i0_se::-1]==SeedH20).index(0)+1
            if_se = list(SeedH2[i0_se:]==SeedH20).index(0)+i0_se-1
            tf_se = t_se[if_se]
            ts_se = t_se[is_se]
            
            # find the time frame for which source gas flow was constant
            (t0_so,SourceH20) = v_SourceH2.getNearest(t0,unit=c.PREF_UNIT)
            (ts_so,SourceH2s) = v_SourceH2.getPrevious(ts_se,unit=c.PREF_UNIT)
            (tf_so,SourceH2f) = v_SourceH2.getNext(tf_se,unit=c.PREF_UNIT)
            i0_so = list(t_so).index(t0_so)
            is_so = list(t_so).index(ts_so)
            if_so = list(t_so).index(tf_so)
            
            SourceH20 = SourceH2[i0_so]
            
            if any(SourceH2[i0_so:if_so+1]!=SourceH20):
                if_so = list(SourceH2[i0_so:if_so+1]==SourceH20).index(0)+i0_so-1
                tf = t_so[if_so]
            else:
                tf = tf_se
                
            if any(SourceH2[is_so:i0_so]!=SourceH20):
                is_so = i0_so-list(SourceH2[i0_so:is_so-1:-1]==SourceH20).index(0)+1
                ts = t_so[is_so]
            else:
                ts = ts_se
            
            # find the time frame for which the pumping rate was constant
            (t0_pu,pump0) = v_pump.getNearest(t0,unit=c.PREF_UNIT)
            (ts_pu,pumps) = v_pump.getPrevious(ts,unit=c.PREF_UNIT)
            (tf_pu,pumpf) = v_pump.getNext(tf,unit=c.PREF_UNIT)
            i0_pu = list(t_pu).index(t0_pu)
            is_pu = list(t_pu).index(ts_pu)
            if_pu = list(t_pu).index(tf_pu)
            pump0 = pump[i0_pu]
            
            if any(pump[i0_pu:if_pu+1]!=pump0):
                if_pu = list(pump[i0_pu:if_pu+1]==pump0).index(0)+i0_pu-1
                tf = t_pu[if_pu]
                
            if any(pump[is_pu:i0_pu]!=pump0):
                is_pu = i0_pu-list(pump[i0_pu:is_pu-1:-1]==pump0).index(0)+1
                ts = t_pu[is_pu]
                
            # find the time frame for which plasma current was constant
            (t0_I,I0) = v_I.getNearest(t0,unit=c.PREF_UNIT)
            (ts_I,s0) = v_I.getPrevious(ts,unit=c.PREF_UNIT)
            (tf_I,If) = v_I.getNext(tf,unit=c.PREF_UNIT)
            i0_I = list(t_I).index(t0_I)
            is_I = list(t_I).index(ts_I)
            if_I = list(t_I).index(tf_I)
            
            if any(np.abs(I[i0_I:if_I+1]-I0) > 5):
                if_I = list(np.abs(I[i0_I:if_I+1]-I0)  < 5).index(0)+i0_I-1
                tf = t_I[if_I]
            if any(np.abs(I[is_I:i0_I]-I0)> 5):
                is_I = i0_I - list(np.abs(I[i0_I:is_I-1:-1]-I0)  < 5).index(0)+1
                ts = t_I[is_I]
            
            # find the time frame for which the beam dump was outside the plasma
            (t0_du,dump0) = v_Dump.getNearest(t0,unit=c.PREF_UNIT)
            (ts_du,dumps) = v_Dump.getPrevious(ts,unit=c.PREF_UNIT)
            (tf_du,dumpf) = v_Dump.getNext(tf,unit=c.PREF_UNIT)
            i0_du = list(t_du).index(t0_du)
            is_du = list(t_du).index(ts_du)
            if_du = list(t_du).index(tf_du)
            
            if not all(dump[i0_du:if_du]):
                if_du = list(dump[i0_du:if_du]).index(0)+i0_du
                tf = t_du[if_du]
                
            if i0_du==if_du:
                i0_du-=1
                
            if not all(dump[is_du:i0_du]):
                is_du = i0_du-list(dump[i0_du:is_du-1:-1]).index(0)+1
                ts = t_du[is_du]
            
            
            tConst = c.TimeRange(ts,tf) # time-range during which the relevant settings are constantly applied
            (t_p,TaChPres) = v_TaChPres.read(tConst,unit=c.PREF_UNIT)
            (t_C,Calo) = v_Calo.read(tConst,unit=c.PREF_UNIT)
            
            if len(t_C) > 2:
                t_p = (t_p - t_p[0])/2**32  # Time conversion to seconds
                t_C = (t_C - t_C[0])/2**32
                
                # Make exponential fits of pressure and T difference. Start the Calorimetry fit when pressure is neatly converged
                (p_coef,p_cov) = curve_fit(expdec,t_p,TaChPres,[TaChPres[-1],TaChPres[0]-TaChPres[-1],1/2.5],bounds = ((-np.inf,-np.inf,1/10),(np.inf,np.inf,1/1)))
                
                t0C = t_p[list(p_coef[0] > TaChPres).index(p_coef[1] > 0)] + 5
                
                (Calo_coef,Calo_cov) = curve_fit(expdec,t_C[t_C > t0C],Calo[t_C > t0C],[Calo[-1],Calo[0]-Calo[-1],1/30],bounds = ((-np.inf,-np.inf,1/60),(np.inf,np.inf,1/10)))
                                
                df_log.loc[i,'TaChPres'] = p_coef[0]                    # Target chamber background pressure
                p_er1 = np.std(TaChPres-expdec(t_p,*p_coef))            # Error from noise in data and non-exponential signal behavior
                p_er2 = abs(p_coef[1])*np.exp(-p_coef[-1]*t_p[-1])      # Error from too short measurement when fit has not reached steady-state
                df_log.loc[i,'TaChPres_er'] = np.sqrt((p_er1**2+p_er2**2))
                
                Calo_er1 = np.std(Calo[t_C > t0C]-expdec(t_C[t_C > t0C],*Calo_coef)) # Error from noise in data and non-exponential signal behavior
                Calo_er2 = abs(Calo_coef[1])*np.exp(-Calo_coef[-1]*t_C[-1]) # Error from too short measurement when fit has not reached steady-state
                flow = np.mean(v_Water.read(tConst,unit=c.PREF_UNIT)[1])
                dT = (Calo_coef[0] - df_log.loc[i,'Calo0'])
                
                df_log.loc[i,'Calori'] = dT*flow*CaloConv/CaloCalFac    # Target power loading
                df_log.loc[i,'Calori_er'] = abs(df_log.loc[i,'Calori'])*np.sqrt((CaloCalFac_er/CaloCalFac)**2+((Calo_er1**2+Calo_er2**2+Calo0_er**2)/dT**2))
    
    # Write a log sorted by measurement settings rather than individual measurements
    
    df_settings = pd.DataFrame(columns = ['I','pump%','Seed','TaChPres','TaChPres_er','Calori','Calori_er','Ha','Hb','Hc','Q22','Q11','Q00'])
    j=0
    # For each anode current setting...
    for I in np.unique(df_log['I'])[~np.isnan(np.unique(df_log['I']))]:
        log_for_I = df_log.loc[df_log['I']==I].copy()
        
        # Just to make every setting unique in terms of seeding, corrected later
        log_for_I.loc[log_for_I['pump%'] == 70,'Seed'] = -.2 
        log_for_I.loc[log_for_I['pump%'] == 82,'Seed'] = -.5
        
        # ... and each seeding setting
        for seed in np.unique(log_for_I['Seed'])[~np.isnan(np.unique(log_for_I['Seed']))]:
            
            log_spec_setting = log_for_I.loc[log_for_I['Seed'] == seed].copy()
            
            if (log_spec_setting['Seed'] < 0).any():
                log_spec_setting['Seed'] = 0
                
            # Note down the settings; they are identical, so the mean operation is unnecessary
            df_settings.loc[j,['I','pump%','Seed']] = np.mean(log_spec_setting[['I','pump%','Seed']])
            
            n = len(log_spec_setting)
            if n > 1:
                # For multiple instances of the same setting, the resulting measurements are weightedby inverse square of fitting error
                w = 1/(log_spec_setting['TaChPres_er']*log_spec_setting['TaChPres_er'])
                xav = np.average(log_spec_setting['TaChPres'],weights = w)
                df_settings.loc[j,['TaChPres']] = xav
                # The weighted sample variance is calculated, applying Bessel's correction
                df_settings.loc[j,['TaChPres_er']] = np.sqrt(n/(n-1)*np.average((log_spec_setting['TaChPres']-xav)**2,weights = w))
                w = 1/(log_spec_setting['Calori_er']*log_spec_setting['Calori_er'])
                xav = np.average(log_spec_setting['Calori'],weights = w)
                df_settings.loc[j,['Calori']] = xav
                df_settings.loc[j,['Calori_er']] = np.sqrt(n/(n-1)*np.average((log_spec_setting['Calori']-xav)**2,weights = w))
            else:
                # For single instances, that instance's value and error are used
                df_settings.loc[j,['TaChPres','TaChPres_er','Calori','Calori_er']] = np.mean(log_spec_setting[['TaChPres','TaChPres_er','Calori','Calori_er']])
            
            # The different wavelength regimes are identified by the position of the diffraction grating in the log
            for i in log_spec_setting.index:
                if (log_spec_setting.loc[i,['gratPos']] == 964).all(): 
                    df_settings.loc[j,['Ha']] = i                       # Balmer alpha and beta
                elif (log_spec_setting.loc[i,['gratPos']] == 765).all(): 
                    df_settings.loc[j,['Hb']] = i                       # Balmer beta through to the continuum, short exposure time
                    df_settings.loc[j,['Hc']] = i+1                     # Balmer beta through to the continuum, long exposure time
                elif (log_spec_setting.loc[i,['gratPos']] == 12608).all():
                    df_settings.loc[j,['Q22']] = i                      # Higher wavelengths of the Fulcher band, incl. Q(2-2) branch
                elif (log_spec_setting.loc[i,['gratPos']] == 12461).all():
                    df_settings.loc[j,['Q11']] = i                      # Middle wavelengths of the Fulcher band, incl. Q(1-1) branch
                elif (log_spec_setting.loc[i,['gratPos']] == 12294).all():
                    df_settings.loc[j,['Q00']] = i                      # Higher wavelengths of the Fulcher band, incl. Q(0-0) branch
            j+=1

    # Get Thomson measurements and save to file.
    df_settings['Te'] = 0
    df_settings['Ne'] = 0
    v_Te = db.root._Raw.TsProfTe
    v_Ne = db.root._Raw.TsProfNe
    v_dTe = db.root._Raw.TsProfTe_d
    v_dNe = db.root._Raw.TsProfNe_d
    v_Rad = db.root._Raw.TsRadCoords
    v_Seq = db.root._Raw.TsSequenceNr
    for i in range(j):
        times = []
        for var in ['Ha','Hb']: # Find TS measurements during all exposures with the same settings
            l = df_settings.loc[i,var]
            if not pd.isnull(l):
                seqNr = df_log.loc[l,'TSTrig']
                if not pd.isnull(seqNr):
                    seqStart = v_Seq.search(seqNr, c.CompOperation.CompEqual, c.infinity())[0][0]
                    seqEnd = v_Seq.getNext(seqStart)[0]
                
                    timerange = c.TimeRange(seqStart, seqEnd-1)
                    times += v_Te.getTimes(timerange, 1, 0)
            Tsdata = []
            if len(times) > 0:
                for t in times:
                    #Collect TS data
                    coords = v_Rad.getAt(times[0])
                    Te = v_Te.getAt(times[0], unit=c.U.PREF)
                    dTe= v_dTe.getAt(times[0], unit=c.U.PREF)
                    Ne = v_Ne.getAt(times[0], unit=c.U.PREF)
                    dNe = v_dNe.getAt(times[0], unit=c.U.PREF)
                    Tsdata.append(np.array([coords,Te,dTe,Ne,dNe]))
                Tsdata = np.concatenate(Tsdata,axis=1)
                np.savetxt('./TSProfiles/%s.txt'%i,Tsdata.T)    #save TS data
                df_settings.loc[i,'Te'] = np.max(Tsdata[1])
                df_settings.loc[i,'Ne'] = np.max(Tsdata[3])
    return(df_log,df_settings)
    
    '''
plt.figure();
for I in [80,140,175]:
    loc = df_settings['I'] == I
    plt.plot(df_settings.loc[loc,'TaChPres'],df_settings.loc[loc,'Te'])
plt.xlabel('Neutral pressure [Pa]')
plt.ylabel('Te [eV]')
plt.legend(['Low Ne','Mid Ne','High Ne'])

plt.figure();
for I in [80,140,175]:
    loc = df_settings['I'] == I
    plt.plot(df_settings.loc[loc,'TaChPres'],df_settings.loc[loc,'Ne'])
plt.xlabel('Neutral pressure [Pa]')
plt.ylabel('Ne [m-3]')
plt.legend(['Low Ne','Mid Ne','High Ne'])

plt.figure();
for I in [80,140,175]:
    loc = df_settings['I'] == I
    plt.loglog(df_settings.loc[loc,'Ne'],df_settings.loc[loc,'Te'])
plt.xlabel('Ne [m-3]')
plt.ylabel('Te [eV]')
plt.legend(['Low Ne','Mid Ne','High Ne'])
    '''