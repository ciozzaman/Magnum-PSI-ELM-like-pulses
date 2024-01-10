def get_angle(data,nCh=40,nLines=1,no_line_present=False):
    #nCh=40;nLines=2
    #nCh=40;nLines=7
    from matplotlib import pyplot as plt
    import numpy as np
    from scipy.signal import find_peaks, peak_prominences as get_proms
    from scipy.optimize import curve_fit
    from scipy.special import erf
    import copy


    #Gauss = lambda x,A0,A,x0,sig: A0+A*np.exp(-((np.array(x)-x0)/sig)**2)
    #Gauss2 = lambda x,A0,A1,x01,sig1,A2,x02,sig2 : A0 + Gauss(x,A1,x01,sig1) + Gauss(x,A2,x02,sig2)
    #blockGauss2 = lambda x,A0,Al,Ar,wl,wr,sig,x0: A0+(erf((x-x0+wl)/sig)*Al + erf((x-x0)/sig)*(Ar-Al) - erf((x-x0-wr)/sig)*Ar )/2
    #block = lambda x,A0,A,w,x0: A0+A*(np.abs(x-x0)<w)

    phi = [[] for i in range(nLines)]
    d_phi = [[] for i in range(nLines)]

    blockGauss = lambda x,A0,A,w,sig,x0: A0+A*(erf((x-x0+w)/sig)-erf((x-x0-w)/sig))/2
    Nx,Ny = data.shape
    Spectrum = np.sum(data,axis=0)  # Chord-integrated spectrum
    #Spectrum.dtype = 'int'
    iNoise = Spectrum[20:].argmin()+20
    Noise = data[:750,iNoise];Noise = max(Noise)-min(Noise)
    iBright = Spectrum[20:].argmax()+20
    HM = (Spectrum[iNoise]+ Spectrum[iBright])/2
    RHM = list((Spectrum[iBright:] < HM)).index(True)
    LHM = list((Spectrum[iBright::-1] < HM)).index(True)

    peaks = find_peaks(Spectrum[2:])[0]    # Approximate horizontal postitions of spectral lines
    proms = get_proms(Spectrum[2:],peaks)[0]

    iLines = sorted(peaks[np.argpartition(-proms,nLines-1)[:nLines]]+2)

    iChords = [[] for i in range(nLines)]  # Vertical position of chord centres
    iBins = [[] for i in range(nLines)]    # Vertical position of chord edge
    iPeak = [[] for i in range(nLines)]    # Horizontal position of lines per chord

    '''
    plt.figure()
    plt.imshow(data)
    plt.figure();
    plt.plot(Spectrum)
    plt.plot(iLines,Spectrum[iLines]+(max(Spectrum)-min(Spectrum))/10,'vr')
    '''
    for i,iLine in enumerate(iLines):
        vertData = np.mean(data[:,iLine-LHM:iLine+RHM],axis=1)
        #plt.figure();plt.plot(vertData);plt.pause(0.01)
        binPeaks = find_peaks(-vertData,distance = 25)[0]
        binProms = get_proms(-vertData,binPeaks)[0]

        iBins[i] = sorted(binPeaks[np.argpartition(-binProms,nCh-2)[:nCh-1]])
        diffI = np.diff(iBins[i])


        iM = min(vertData.argmax(),iBins[i][-2])
        #print(iBins[i])
        #print(iM)
        iiM = [iM < foo for foo in iBins[i]].index(1)
        try:
            i0 = iiM-list(diffI[iiM::-1] > 1.2*np.median(diffI)).index(1)+1
        except ValueError:
            i0 = None
        try:
            ie = iiM+list(diffI[iiM:] > 1.2*np.median(diffI)).index(1)+1
        except ValueError:
            ie = None
        iBins[i] = iBins[i][i0:ie]

        if len(iBins[i]) > 5:
            binFit = np.polyfit(range(len(iBins[i])),iBins[i],1)
            iFit = np.polyval(binFit,range(len(iBins[i])))

            iBins[i] = np.vectorize(int)(np.round(np.concatenate([[np.polyval(binFit,-1)],iFit,[np.polyval(binFit,len(iBins[i]))]])))
            for jj,j in enumerate(iBins[i][:-1]):
                iChords[i].append(j + np.argmax(vertData[j:iBins[i][jj+1]]))

            '''
            plt.figure()
            plt.plot(vertData)
            plt.plot(iBins[i],vertData[iBins[i]]-10,'^r')
            '''
            '''
            plt.figure()
            plt.imshow(data,aspect='auto')
            plt.plot([iLine]*2,plt.ylim(),'r')
            '''
            '''
            plt.figure()
            plt.imshow(data,aspect='auto')
            for y in iBins[i]:
                plt.plot(plt.xlim(),[y]*2,'r')
            yl = iBins[i][0]-50,iBins[i][-1]+50
            plt.ylim(iBins[i][0]-50,iBins[i][-1]+50)
            xl = max(0,iLine-100)
            xr = min(2047,iLine+100)
            plt.xlim(xl,xr)
            '''
            i_0 = max(0,iLine-2*LHM)
            i_f = min(iLine+RHM+LHM,Nx)
            if i_f<i_0:
                temp=copy.deepcopy(i_0)
                i_0=copy.deepcopy(i_f)
                i_f=copy.deepcopy(temp)
            i_s = range(i_0,i_f)
            for ii,iChord in enumerate(iChords[i]):
                #print('iChords')
                #print(iChords)
                #print('i_0,i_f')
                #print(i_0)
                #print(i_f)
                #print('np.shape(data)')
                #print(np.shape(data))
                horData = data[iChord,i_0:i_f]
                #print('horData')
                #print(horData)
                p0 = [horData[0],max(horData),RHM,LHM/4,horData.argmax()+i_0-5] #BlockGauss
                try:
                    specFit = curve_fit(blockGauss,i_s,horData,p0 = p0)[0]
                    iPeak[i].append(specFit[-1])
                except RuntimeError:
                    iPeak[i].append(np.nan)


                '''
                plt.figure()
                plt.plot(i_s,data[iChord,i_0:i_f])
                plt.plot(i_s,blockGauss(i_s,*specFit))
                #plt.plot(i_s,blockGauss(i_s,*np.array(p0)))
                '''

            ixfin = np.isfinite(iChords[i]) & np.isfinite(iPeak[i])
            coefs,cov=np.polyfit(np.array(iChords[i])[ixfin],np.array(iPeak[i])[ixfin],1,cov=True)

            '''
            plt.figure()
            plt.plot(iChords[i],iPeak[i],'.')
            plt.plot(iChords[i],np.polyval(coefs,iChords[i]))

            plt.figure()
            plt.imshow(data)
            plt.plot(iPeak[i],iChords[i],'.r')
            '''
            #print('coefs[0]')
            #print(coefs[0])
            phi[i] = -np.arctan(coefs[0])*180/np.pi
            d_phi[i] = np.abs(1/(1+coefs[0]*coefs[0])*np.sqrt(cov[0,0])*180/np.pi)
        else:
            phi[i] = np.nan
            d_phi[i] = np.nan
    return(phi,d_phi)


def get_angle_2(data, nCh=40, nLines=2,max_angle=20,bininterv_est='auto'):
    # max_angle added 12/08/2019

    # nCh=40;nLines=2
    # nCh=40;nLines=7
    from matplotlib import pyplot as plt
    import numpy as np
    from scipy.signal import find_peaks, peak_prominences as get_proms
    from scipy.optimize import curve_fit
    from scipy.special import erf
    import copy

    Gauss = lambda x, A0, A, x0, sig: A0 + A * np.exp(-((np.array(x) - x0) / sig) ** 2)
    # Gauss2 = lambda x,A0,A1,x01,sig1,A2,x02,sig2 : A0 + Gauss(x,A1,x01,sig1) + Gauss(x,A2,x02,sig2)
    # blockGauss2 = lambda x,A0,Al,Ar,wl,wr,sig,x0: A0+(erf((x-x0+wl)/sig)*Al + erf((x-x0)/sig)*(Ar-Al) - erf((x-x0-wr)/sig)*Ar )/2
    # block = lambda x,A0,A,w,x0: A0+A*(np.abs(x-x0)<w)

    phi = [[] for i in range(nLines)]
    d_phi = [[] for i in range(nLines)]

    blockGauss = lambda x, A0, A, w, sig, x0: A0 + A * (erf((x - x0 + w) / sig) - erf((x - x0 - w) / sig)) / 2
    Nx, Ny = data.shape
    Spectrum = np.sum(data, axis=0)  # Chord-integrated spectrum
    # Spectrum.dtype = 'int'
    iNoise = Spectrum[20:].argmin() + 20
    Noise = data[:750, iNoise];
    Noise = max(Noise) - min(Noise)
    iBright = Spectrum[20:].argmax() + 20
    HM = (Spectrum[iNoise] + Spectrum[iBright]) / 2
    RHM = list((Spectrum[iBright:] < HM)).index(True)
    LHM = list((Spectrum[iBright::-1] < HM)).index(True)

    peaks = find_peaks(Spectrum[2:])[0]  # Approximate horizontal postitions of spectral lines
    proms = get_proms(Spectrum[2:], peaks)[0]

    iLines = sorted(peaks[np.argpartition(-proms, nLines - 1)[:nLines]] + 2)
    print(iLines)

    iChords = [[] for i in range(nLines)]  # Vertical position of chord centres
    iBins = [[] for i in range(nLines)]  # Vertical position of chord edge
    iPeak = [[] for i in range(nLines)]  # Horizontal position of lines per chord

    '''
    plt.figure()
    plt.imshow(data)
    plt.figure();
    plt.plot(Spectrum)
    plt.plot(iLines,Spectrum[iLines]+(max(Spectrum)-min(Spectrum))/10,'vr')
    '''
    for i, iLine in enumerate(iLines):
        vertData = np.mean(data[:, iLine - LHM:iLine + RHM], axis=1)
        # plt.figure();plt.plot(vertData);plt.pause(0.01)
        if bininterv_est!='auto':
            binPeaks = find_peaks(-vertData, distance=25)[0]
            presumed_distance = 25
            while len(binPeaks)<(nCh-2):
                presumed_distance-=2
                binPeaks = find_peaks(-vertData, distance=presumed_distance)[0]
            binProms = get_proms(-vertData, binPeaks)[0]
        else:    # introduced 21/06/2020
            temp = []
            for presumed_distance in [19,20,21,22,23,24,25]:
                # temp.append(np.mean(get_proms(-vertData, find_peaks(-vertData, distance=presumed_distance)[0])[0]))
                peaks = find_peaks(-vertData, distance=presumed_distance)[0]
                temp.append(np.mean(get_proms(-vertData, peaks)[0])*(len(peaks)==nCh-1))
            presumed_distance = [19,20,21,22,23,24,25][np.array(temp).argmax()]
            binPeaks = find_peaks(-vertData, distance=presumed_distance)[0]
            binProms = get_proms(-vertData, binPeaks)[0]

        iBins[i] = sorted(binPeaks[np.argpartition(-binProms, nCh - 2)[:nCh - 1]])
        diffI = np.diff(iBins[i])

        iM = min(vertData.argmax(), iBins[i][-2])
        # print(iBins[i])
        # print(iM)
        iiM = [iM < foo for foo in iBins[i]].index(1)
        try:
            i0 = iiM - list(diffI[iiM::-1] > 1.2 * np.median(diffI)).index(1) + 1
        except ValueError:
            i0 = None
        try:
            ie = iiM + list(diffI[iiM:] > 1.2 * np.median(diffI)).index(1) + 1
        except ValueError:
            ie = None
        iBins[i] = iBins[i][i0:ie]

        if len(iBins[i]) > 5:
            binFit = np.polyfit(range(len(iBins[i])), iBins[i], 1)
            iFit = np.polyval(binFit, range(len(iBins[i])))

            iBins[i] = np.vectorize(int)(
                np.round(np.concatenate([[np.polyval(binFit, -1)], iFit, [np.polyval(binFit, len(iBins[i]))]])))
            for jj, j in enumerate(iBins[i][:-1]):
                iChords[i].append(j + np.argmax(vertData[j:iBins[i][jj + 1]]))

            '''
            plt.figure()
            plt.plot(vertData)
            plt.plot(iBins[i],vertData[iBins[i]]-10,'^r')
            '''
            '''
            plt.figure()
            plt.imshow(data,aspect='auto')
            plt.plot([iLine]*2,plt.ylim(),'r')
            '''
            '''
            plt.figure()
            plt.imshow(data,aspect='auto')
            for y in iBins[i]:
                plt.plot(plt.xlim(),[y]*2,'r')
            yl = iBins[i][0]-50,iBins[i][-1]+50
            plt.ylim(iBins[i][0]-50,iBins[i][-1]+50)
            xl = max(0,iLine-100)
            xr = min(2047,iLine+100)
            plt.xlim(xl,xr)
            '''
            i_0 = max(0, iLine - 2 * LHM)
            i_f = min(iLine + RHM + LHM, Nx)
            if i_f < i_0:
                temp = copy.deepcopy(i_0)
                i_0 = copy.deepcopy(i_f)
                i_f = copy.deepcopy(temp)
            i_s = range(i_0, i_f)
            for ii, iChord in enumerate(iChords[i]):
                # print('iChords')
                # print(iChords)
                # print('i_0,i_f')
                # print(i_0)
                # print(i_f)
                # print('np.shape(data)')
                # print(np.shape(data))
                # horData = data[iChord,i_0:i_f]
                horData = np.mean(data[iBins[i][ii]:iBins[i][ii + 1], i_0:i_f], axis=0)
                # print('horData')
                # print(horData)
                # p0 = [horData[0],max(horData),RHM,LHM/4,horData.argmax()+i_0-5] #BlockGauss
                p0 = [horData[0], max(horData), horData.argmax() + i_0 - 5, LHM / 4]  # Gauss
                try:
                    # specFit = curve_fit(blockGauss,i_s,horData,p0 = p0)[0]
                    # iPeak[i].append(specFit[-1])
                    specFit = curve_fit(Gauss, i_s, horData, p0=p0)[0]
                    iPeak[i].append(specFit[-2])
                except RuntimeError:
                    iPeak[i].append(np.nan)

                '''
                plt.figure()
                plt.plot(i_s,data[iChord,i_0:i_f])
                plt.plot(i_s,blockGauss(i_s,*specFit))
                #plt.plot(i_s,blockGauss(i_s,*np.array(p0)))
                '''

            ixfin = np.isfinite(iChords[i]) & np.isfinite(iPeak[i])
            coefs, cov = np.polyfit(np.array(iChords[i])[ixfin], np.array(iPeak[i])[ixfin], 1, cov=True)

            '''
            plt.figure()
            plt.plot(iChords[i],iPeak[i],'.')
            plt.plot(iChords[i],np.polyval(coefs,iChords[i]))

            plt.figure()
            plt.imshow(data)
            plt.plot(iPeak[i],iChords[i],'.r')
            '''
            # print('coefs[0]')
            # print(coefs[0])
            phi[i] = -np.arctan(coefs[0]) * 180 / np.pi
            if not np.isnan(phi[i]):
                if abs(phi[i])>max_angle:
                    d_phi[i]=1000000
                else:
                    d_phi[i] = np.abs(1 / (1 + coefs[0] * coefs[0]) * np.sqrt(cov[0, 0]) * 180 / np.pi)
            else:
                d_phi[i] = np.nan
        else:
            phi[i] = np.nan
            d_phi[i] = np.nan
    return (phi, d_phi)


def rotate(data,phi):
    import cv2
    import numpy as np
    Nx,Ny = data.shape
    M = cv2.getRotationMatrix2D((Nx/2,Ny/2),phi,1)
    data = cv2.warpAffine(np.array(data).astype('float'),M,(Ny,Nx))
    return (data)

def get_tilt(data,nCh=40,nLines=5,plot_reports=False,bininterv_est='auto'):
    #nCh=40;nLines=2
    #nCh=40;nLines=7
    from matplotlib import pyplot as plt
    import numpy as np
    from scipy.signal import find_peaks, peak_prominences as get_proms
    from scipy.optimize import curve_fit
    from scipy.special import erf

    Nx,Ny = data.shape
    Spectrum = np.sum(data,axis=0)  # Chord-integrated spectrum
    Spectrum-=min(Spectrum)
    #Spectrum.dtype = 'int'
    iNoise = Spectrum[20:-20].argmin()+20
    Noise = data[20:-20,iNoise];Noise = max(Noise)-min(Noise)
    iBright = Spectrum[20:-20].argmax()+20
    HM = (Spectrum[iBright]-Spectrum[iNoise]) / 2 + Spectrum[iNoise]
    #HM = (Spectrum[iNoise]+ Spectrum[iBright])/2
    RHM = list((Spectrum[iBright:] < HM)).index(True)
    LHM = list((Spectrum[iBright::-1] < HM)).index(True)

    peaks = find_peaks(Spectrum[2:])[0]    # Approximate horizontal postitions of spectral lines
    proms = get_proms(Spectrum[2:],peaks)[0]

    iLines = sorted(peaks[np.argpartition(-proms,nLines-1)[:nLines]]+2)

    iBins = [[] for i in range(nLines)]    # Vertical position of chord edge
    useLine = np.zeros(nLines,dtype='bool')

    if plot_reports==True:
        plt.figure()
        plt.imshow(data)
        plt.figure();
        plt.plot(Spectrum)
        plt.plot(iLines,Spectrum[iLines]+(max(Spectrum)-min(Spectrum))/10,'vr')


        plt.figure()
        plt.imshow(data,'rainbow', origin='lower')

    for i,iLine in enumerate(iLines):
        vertData = np.mean(data[:,iLine-LHM:iLine+RHM],axis=1)
        #plt.figure();plt.plot(vertData);plt.pause(0.01)
        if bininterv_est!='auto':
            binPeaks = find_peaks(-vertData,distance = 20)[0]
            presumed_distance = 20
            while len(binPeaks)<(nCh-2):
                presumed_distance-=2
                binPeaks = find_peaks(-vertData, distance=presumed_distance)[0]
            binProms = get_proms(-vertData,binPeaks)[0]
        else:    # introduced 21/06/2020
            temp = []
            for presumed_distance in [19,20,21,22,23,24,25]:
                # temp.append(np.mean(get_proms(-vertData, find_peaks(-vertData, distance=presumed_distance)[0])[0]))
                peaks = find_peaks(-vertData, distance=presumed_distance)[0]
                temp.append(np.mean(get_proms(-vertData, peaks)[0])*(len(peaks)==nCh-1))
            presumed_distance = [19,20,21,22,23,24,25][np.array(temp).argmax()]
            binPeaks = find_peaks(-vertData, distance=presumed_distance)[0]
            binProms = get_proms(-vertData, binPeaks)[0]

        iBins[i] = sorted(binPeaks[np.argpartition(-binProms,nCh-2)[:nCh-1]])
        diffI = np.diff(iBins[i])


        iM = min(vertData.argmax(),iBins[i][-2])
        iiM = [iM < foo for foo in iBins[i]].index(1)
        #print(iBins[i])
        #print(iM)
        #print(diffI)
        #print(np.median(diffI))
        try:
            i0 = iiM-list(diffI[iiM::-1] > 1.2*np.median(diffI)).index(1)+1
        except ValueError:
            i0 = None
        try:
            ie = iiM+list(diffI[iiM:] > 1.2*np.median(diffI)).index(1)+1
        except ValueError:
            ie = None
        iBins[i] = iBins[i][i0:ie]

        if len(iBins[i]) == nCh-1:
            useLine[i] = True
        #print(ie)
        #print(i0)
        #print(len(iBins[i]))
        if plot_reports == True:
            plt.plot((iLine + 30) * np.ones_like(iBins[i]), iBins[i], '<r')
    if plot_reports == True:
        plt.pause(0.01)

        plt.plot((iLine+30)*np.ones_like(iBins[i]),iBins[i],'<r')
    plt.pause(0.01)


    if sum(useLine) > 1:
        usedBins = np.array([bins for bins,use in zip(iBins,useLine) if use])
        shift = np.mean(usedBins,axis=1);shift-=shift[0]
        binInterv,bin0 = np.polyfit(np.tile(np.arange(1,nCh),sum(useLine)),(usedBins-np.repeat([shift],nCh-1,axis=0).T).flatten(),1)
        tilt,bin00 = np.polyfit(np.array(iLines)[useLine],shift+bin0,1)
        return(binInterv,tilt,bin00)

    elif sum(useLine) == 1:
        usedBins = np.array([bins for bins,use in zip(iBins,useLine) if use])
        #print('iBins ='+str(iBins))
        #print('useLine ='+str(useLine))
        #print('usedBins ='+str(usedBins))
        shift = np.mean(usedBins,axis=1);shift-=shift[0]
        #print('shift ='+str(shift))
        #print('np.tile(np.arange(1,nCh),sum(useLine)) ='+str(np.tile(np.arange(1,nCh),sum(useLine))))
        #print('(usedBins-np.repeat([shift],nCh-1,axis=0).T).flatten() ='+str((usedBins-np.repeat([shift],nCh-1,axis=0).T).flatten()))
        binInterv,bin0 = np.polyfit(np.tile(np.arange(1,nCh),sum(useLine)),(usedBins-np.repeat([shift],nCh-1,axis=0).T).flatten(),1)
        return(binInterv,np.nan,bin0)
    else:
        return(np.nan,np.nan,np.nan)

    '''
    x = np.arange(data.shape[0])
    y0 = bin00+0*x
    #y0 = bin00 + x*tilt
    plt.figure()
    plt.imshow(data)
    for iB in range(nCh+1):
        plt.plot(x,y0+binInterv*iB,'r')
    plt.tight_layout()
    '''


def get_line_position(data,nLines,nCh=40):
    #nCh=40;nLines=2
    #nCh=40;nLines=7
    import numpy as np
    from scipy.signal import find_peaks, peak_prominences as get_proms

    Nx,Ny = data.shape
    Spectrum = np.sum(data,axis=0)  # Chord-integrated spectrum
    Spectrum-=min(Spectrum)
    #Spectrum.dtype = 'int'
    iNoise = Spectrum[20:-20].argmin()+20
    Noise = data[20:-20,iNoise];Noise = max(Noise)-min(Noise)
    iBright = Spectrum[20:-20].argmax()+20

    peaks = find_peaks(Spectrum[2:])[0]    # Approximate horizontal postitions of spectral lines
    proms = get_proms(Spectrum[2:],peaks)[0]

    iLines = sorted(peaks[np.argpartition(-proms,nLines-1)[:nLines]]+2)

    return iLines

def get_4_points(data, nCh=40, nLines=5,plot_reports=False,return_4_points=False,bininterv_est='auto'):
    # nCh=40;nLines=2
    # nCh=40;nLines=7
    from matplotlib import pyplot as plt
    import numpy as np
    from scipy.signal import find_peaks, peak_prominences as get_proms
    from scipy.optimize import curve_fit
    from scipy.special import erf

    Nx, Ny = data.shape
    data[data==0]=np.min(data[data>0])
    Spectrum = np.sum(data, axis=0)  # Chord-integrated spectrum
    Spectrum -= min(Spectrum)
    # Spectrum.dtype = 'int'
    iNoise = Spectrum[20:-20].argmin() + 20
    Noise = data[20:-20, iNoise];
    Noise = max(Noise) - min(Noise)
    iBright = Spectrum[20:-20].argmax() + 20
    HM = (Spectrum[iBright]-Spectrum[iNoise]) / 2 + Spectrum[iNoise]
    # HM = (Spectrum[iNoise]+ Spectrum[iBright])/2
    RHM = list((Spectrum[iBright:] < HM)).index(True)
    LHM = list((Spectrum[iBright::-1] < HM)).index(True)

    peaks = find_peaks(Spectrum[2:])[0]  # Approximate horizontal postitions of spectral lines
    proms = get_proms(Spectrum[2:], peaks)[0]

    iLines = sorted(peaks[np.argpartition(-proms, nLines - 1)[:nLines]] + 2)

    iBins = [[] for i in range(nLines)]  # Vertical position of chord edge
    useLine = np.zeros(nLines, dtype='bool')

    if plot_reports==True:
        plt.figure()
        plt.imshow(data)
        plt.figure();
        plt.plot(Spectrum)
        plt.plot(iLines,Spectrum[iLines]+(max(Spectrum)-min(Spectrum))/10,'vr')


        plt.figure()
        plt.imshow(data,'rainbow', origin='lower')

    for i, iLine in enumerate(iLines):
        vertData = np.mean(data[:, iLine - LHM:iLine + RHM], axis=1)
        # plt.figure();plt.plot(vertData);plt.pause(0.01)
        if bininterv_est!='auto':
            binPeaks = find_peaks(-vertData, distance=20)[0]
            presumed_distance = 20
            while len(binPeaks) < (nCh - 2):
                presumed_distance -= 2
                binPeaks = find_peaks(-vertData, distance=presumed_distance)[0]
            binProms = get_proms(-vertData, binPeaks)[0]
        else:    # introduced 21/06/2020
            temp = []
            for presumed_distance in [19,20,21,22,23,24,25]:
                # temp.append(np.mean(get_proms(-vertData, find_peaks(-vertData, distance=presumed_distance)[0])[0]))
                peaks = find_peaks(-vertData, distance=presumed_distance)[0]
                temp.append(np.mean(get_proms(-vertData, peaks[np.logical_and(peaks>15,peaks<Nx-15)])[0])*(len(peaks[np.logical_and(peaks>15,peaks<Nx-15)])==nCh-1))
            presumed_distance = [19,20,21,22,23,24,25][np.array(temp).argmax()]
            binPeaks = find_peaks(-vertData, distance=presumed_distance)[0]
            binProms = get_proms(-vertData, binPeaks)[0]

        iBins[i] = sorted(binPeaks[np.argpartition(-binProms, nCh - 2)[:nCh - 1]])
        diffI = np.diff(iBins[i])

        iM = min(vertData.argmax(), iBins[i][-2])
        iiM = [iM < foo for foo in iBins[i]].index(1)
        # print(iBins[i])
        # print(iM)
        # print(diffI)
        # print(np.median(diffI))
        try:
            i0 = iiM - list(diffI[iiM::-1] > 1.2 * np.median(diffI)).index(1) + 1
        except ValueError:
            i0 = None
        try:
            ie = iiM + list(diffI[iiM:] > 1.2 * np.median(diffI)).index(1) + 1
        except ValueError:
            ie = None
        iBins[i] = iBins[i][i0:ie]

        if len(iBins[i]) == nCh - 1:
            useLine[i] = True
        # print(ie)
        # print(i0)
        # print(len(iBins[i]))
        if plot_reports==True:
            plt.plot((iLine + 30) * np.ones_like(iBins[i]), iBins[i], '<r')
    if plot_reports==True:
        plt.pause(0.01)

    if sum(useLine) > 1:
        usedBins = np.array([bins for bins, use in zip(iBins, useLine) if use])
        usediLines = np.array([bins for bins, use in zip(iLines, useLine) if use])
        print(usediLines)

        usedBins = usedBins.tolist()
        for index in range(len(usedBins)):
            interv = np.mean(np.diff(usedBins[index]))
            usedBins[index].append(np.min(usedBins[index]) - 2 * interv)  # I suppose that there is always enough pixels for this
            usedBins[index].append(np.max(usedBins[index]) + 2 * interv)
        usedBins = np.array(usedBins)
        usedBins = np.sort(usedBins, axis=(-1))
        binFit = np.polyfit(usediLines, usedBins[:, 0], 1)
        iFit_down = np.polyval(binFit, [0, Ny])
        binFit = np.polyfit(usediLines, usedBins[:, -1], 1)
        iFit_up = np.polyval(binFit, [0, Ny])

        pts_1 = np.array([(iFit_up[0], 0), (iFit_up[-1], Ny), (iFit_down[0], 0), (iFit_down[-1], Ny)])
        pts_1 = np.flip(pts_1, axis=(1))

        if plot_reports==True:
            plt.figure()
            plt.imshow(data,'rainbow', origin='lower')
            plt.plot(pts_1.T[0,:2],pts_1.T[1,:2])
            plt.plot(pts_1.T[0,2:], pts_1.T[1, 2:])
            plt.pause(0.001)

        return pts_1

    else:
        return np.ones((4,2))*np.nan


def do_tilt(data,tilt):
    import numpy as np
    import cv2
    import numbers
    if not isinstance(tilt, numbers.Real):
        #if tilt==np.nan:
        return data
    Nx,Ny = data.shape
    pts1 = np.float32([[0,0],[0,Ny],[Nx,tilt*Nx]])
    pts2 = np.float32([[0,0],[0,Ny],[Nx,0]])

    M = cv2.getAffineTransform(pts1,pts2)

    data = cv2.warpAffine(np.array(data).astype('float'),M,(Ny,Nx))
    return data
'''
x = np.arange(data.shape[0])
plt.figure()
plt.imshow(data)
plt.tight_layout()

for iB in range(60):
    plt.plot(x,np.repeat(bin0%binInterv+binInterv*iB,Nx),'r')
'''
def getFirstBin(data,nCh=40,nLines=2,binInterv=34,calibration=False):
    #nCh=40;nLines=2
    #nCh=40;nLines=7
    import numpy as np
    from scipy.signal import find_peaks, peak_prominences as get_proms
    #from scipy.optimize import curve_fit
    #from scipy.special import erf

    Nx,Ny = data.shape
    Spectrum = np.sum(data,axis=0)  # Chord-integrated spectrum
    Spectrum-=min(Spectrum)
    if not calibration:
        #Spectrum.dtype = 'int'
        iNoise = Spectrum[20:-20].argmin()+20
        Noise = data[20:750,iNoise];Noise = max(Noise)-min(Noise)
        iBright = Spectrum[20:-20].argmax()+20
        HM = (Spectrum[iNoise]+ Spectrum[iBright])/4+Spectrum[iNoise]
        try:
            RHM = list((Spectrum[iBright:] < HM)).index(True)
            LHM = list((Spectrum[iBright::-1] < HM)).index(True)
        except ValueError:
            return np.nan


        peaks = find_peaks(Spectrum[2:])[0]    # Approximate horizontal postitions of spectral lines
        proms = get_proms(Spectrum[2:],peaks)[0]

        iLines = sorted(peaks[np.argpartition(-proms,nLines-1)[:nLines]]+2)
    else:
        iBright = Spectrum[50:-50].argmin() + 50
        RHM=iBright+20
        LHM=iBright-20
        nLines=1
        iLines=[iBright]
        data=-data

    iBins = [[] for i in range(nLines)]    # Vertical position of chord edge
    useLine = np.zeros(nLines,dtype='bool')
    '''
    plt.figure()
    plt.imshow(data)
    plt.figure();
    plt.plot(Spectrum)
    plt.plot(iLines,Spectrum[iLines]+(max(Spectrum)-min(Spectrum))/10,'vr')
    '''
    binIntervs = binInterv*np.arange(1,nCh)

    for i,iLine in enumerate(iLines):
        vertData = np.mean(data[:,max(iLine-LHM,0):min(iLine+RHM,Ny-1)],axis=1)
        binPeaks = find_peaks(-vertData,distance = 25)[0]
        temp=[]
        for value in binPeaks:
            if (value>5 and value<len(vertData)-5):
                temp.append(value)
        binPeaks=np.array(temp)
        binProms = get_proms(-vertData,binPeaks)[0]

        iBins[i] = sorted(binPeaks[np.argpartition(-binProms,nCh-2)[:nCh-1]])
        # if (iBins[i][0]<5 and calibration):
        #     iBins[i] = sorted(binPeaks[np.argpartition(-binProms,nCh-2)[:nCh]])[1:]
        diffI = np.diff(iBins[i])


        #iM = vertData[:-10].argmax()
        iM = min(vertData.argmax(),iBins[i][-2])
        iiM = [iM < foo for foo in iBins[i]].index(1)
        treshold = 1.2
        if calibration:
            treshold=2
        try:
            i0 = iiM-list(diffI[iiM::-1] > treshold*np.median(diffI)).index(1)+1
        except ValueError:
            i0 = iiM-len(diffI[iiM::-1])+1
        try:
            ie = iiM+list(diffI[iiM:] > treshold*np.median(diffI)).index(1)+1
        except ValueError:
            ie = iiM+len(diffI[iiM:])+1
        iBins[i] = iBins[i][i0:ie]

        if len(iBins[i]) == nCh-1:
            useLine[i] = True
    #print('useLine'+str(useLine))
    if sum(useLine) > 0:
        usedBins = np.array([bins-binIntervs for bins,use in zip(iBins,useLine) if use])
        return(np.mean(usedBins))

    else:
        return(np.nan)

def binData_old(data,bin00,binInterv,nCh=40,check_overExp = False,treshold=64e3):
    # Original routine from Gijs, dismissed 31/01/2020

    #bin00 = geom['bin00a'];binInterv = geom['binInterv'];check_overExp = False
    #bin00 = geom['bin00b'];binInterv = geom['binInterv'];check_overExp = False
    #bin00 = geom['bin00b'];binInterv = geom['binInterv'];check_overExp = True
    import numpy as np
    oExpLim = np.zeros(nCh,dtype = 'int')
    binnedData = np.zeros([nCh,data.shape[1]])
    ie = bin00
    i = 0
    while i<nCh:
        i0=ie
        ie=i0+binInterv
        ist = int(round(i0))
        ien = int(round(ie))
        #print('np.shape(data)'+str(np.shape(data)))
        binnedData[i,:] = np.sum(data[ist:ien,:],axis=0)
        if check_overExp:
            oExpLim[i] = np.max(np.append((np.max( data[ist:ien,:] ,axis=0) > treshold).nonzero(),0))
        i+=1
    if not check_overExp:
        return binnedData
    else:
        return binnedData,oExpLim


def binData(data,bin00,binInterv,nCh=40,check_overExp = False,treshold=64e3):
    #bin00 = geom['bin00a'];binInterv = geom['binInterv'];check_overExp = False
    #bin00 = geom['bin00b'];binInterv = geom['binInterv'];check_overExp = False
    #bin00 = geom['bin00b'];binInterv = geom['binInterv'];check_overExp = True
    import numpy as np
    oExpLim = np.zeros(nCh,dtype = 'int')
    binnedData = np.zeros([nCh,data.shape[1]])
    ie = bin00
    i = 0
    while i<nCh:
        i0=ie
        ie=i0+binInterv
        ist_min = int(np.floor(i0))
        ist_max = int(np.ceil(i0))
        ien_min = int(np.floor(ie))
        ien_max = int(np.ceil(ie))

        mask = np.ones((ien_max-ist_min))
        if ist_min<ist_max:
            mask[0]=ist_max-i0
        if ien_max>ien_min:
            mask[-1]=ie-ien_min

        #print('np.shape(data)'+str(np.shape(data)))
        binnedData[i,:] = np.sum((data[ist_min:ien_max,:].T*mask).T,axis=0)
        if check_overExp:
            oExpLim[i] = np.max(np.append((np.max( data[ist_min:ien_max,:] ,axis=0) > treshold).nonzero(),0))
        i+=1
    if not check_overExp:
        return binnedData
    else:
        return binnedData,oExpLim


def binData_with_sigma(data,data_sigma,bin00,binInterv,nCh=40,check_overExp = False,treshold=64e3):
    #bin00 = geom['bin00a'];binInterv = geom['binInterv'];check_overExp = False
    #bin00 = geom['bin00b'];binInterv = geom['binInterv'];check_overExp = False
    #bin00 = geom['bin00b'];binInterv = geom['binInterv'];check_overExp = True
    import numpy as np
    oExpLim = np.zeros(nCh,dtype = 'int')
    binnedData = np.zeros([nCh,data.shape[1]])
    binnedData_sigma = np.zeros([nCh,data.shape[1]])
    ie = bin00
    i = 0
    while i<nCh:
        i0=ie
        ie=i0+binInterv
        ist_min = int(np.floor(i0))
        ist_max = int(np.ceil(i0))
        ien_min = int(np.floor(ie))
        ien_max = int(np.ceil(ie))

        mask = np.ones((ien_max-ist_min))
        if ist_min<ist_max:
            mask[0]=ist_max-i0
        if ien_max>ien_min:
            mask[-1]=ie-ien_min

        #print('np.shape(data)'+str(np.shape(data)))
        binnedData[i,:] = np.sum((data[ist_min:ien_max,:].T*mask).T,axis=0)
        binnedData_sigma[i,:] = np.sqrt(np.sum(((data_sigma[ist_min:ien_max,:].T*mask)**2).T,axis=0))
        if check_overExp:
            oExpLim[i] = np.max(np.append((np.max( data[ist_min:ien_max,:] ,axis=0) > treshold).nonzero(),0))
        i+=1
    if not check_overExp:
        return binnedData,binnedData_sigma
    else:
        return binnedData,binnedData_sigma,oExpLim
