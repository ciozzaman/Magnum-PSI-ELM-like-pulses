def do_waveL_Calib(merge_ID,df_settings,df_log,geom,nCh = 40,Nx = 1608,fdir = '/home/ff645/GoogleDrive/ff645/YPI - Fabio/Collaboratory/Experimental data/tests/test_data_folder'):
    import numpy as np
    from matplotlib import pyplot as plt
    import os
    from functions.winspec import SpeFile
    from .spectools import rotate,do_tilt, binData
    from scipy.signal import find_peaks, peak_prominences as get_proms
    from scipy.optimize import curve_fit
    from scipy.special import erf
    from .fabio_add import all_file_names,order_filenames,order_filenames_csv,load_dark, find_index_of_file,fix_minimum_signal2,four_point_transform
    from PIL import Image
    import numbers
    import pandas as pd

    geom_null = pd.DataFrame(columns=['angle', 'tilt', 'binInterv', 'bin00a', 'bin00b'])
    geom_null.loc[0] = [0, 0, 0, 0, 0]

    #fdir = '/home/ff645/Downloads/OES Fabio/data/'

    n = np.arange(10,20)
    waveLengths = [656.45377,486.13615,434.0462,410.174,397.0072,388.9049,383.5384] + list((364.6*(n*n)/(n*n-4)))


    waveLfits = np.zeros([3,len(df_settings),2])*np.nan


    for k,typ in enumerate(['Hb','Hc']):
        # for iFile in range(len(df_settings)):
            # Find vertical position of fibres for low wavelength
            # j = df_settings.loc[iFile,typ]
            # if not np.isnan(j):
        if k>0:
            continue
            # all_j_0 = find_index_of_file(merge_ID, df_settings, df_log, type=typ)

        all_j = find_index_of_file(merge_ID, df_settings, df_log,type=typ)
        if all_j==[]:
            continue
        for iFile,j in enumerate(all_j):
            # if k>0:
            #     if j in all_j_0:
            #         continue
            (folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]

            #jDark = np.where((df_log['folder']==df_log.loc[j,'Dark_folder']) & (df_log['sequence']==df_log.loc[j,'Dark_sequence']) & (df_log['type'] == 'Dark'))[0][0]
            #(fDark,dDark,sDark) = df_log.loc[jDark,['folder','date','sequence']]
            #type = '.tif'
            #filenames_Dark = all_file_names(fdir+'/'+fDark+'/'+"{0:0=2d}".format(sDark)+'/Untitled_1/Pos0', type)
            #dataDark=[]
            #for index,filename in enumerate(filenames_Dark[:5]):
            #    fname = fdir+'/'+fDark+'/'+"{0:0=2d}".format(sDark)+'/Untitled_1/Pos0/'+filename
            #    im = Image.open(fname)
            #    data = np.array(im)
            #    dataDark.append(data)
            #dataDark=np.mean(dataDark,axis=0)
            dataDark = load_dark(j,df_settings,df_log,fdir,geom_null)

            type = '.tif'
            filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)
            binnedData_collect=[]
            for index,filename in enumerate(filenames):
                fname = fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0/'+filename

                #file = SpeFile(fdir+fname)
                #data = file.data[0].T
                im = Image.open(fname)
                data = np.array(im)
                data = fix_minimum_signal2(data)

                #fnDark = fDark+'/'+dDark+tDark+'.spe'
                #fileDark = SpeFile(fdir+fnDark)
                #dataDark = fileDark.data[0].T
                #data =data - dataDark
                data =data - dataDark
                data = rotate(data,geom['angle'])
                # print(geom['tilt'])
                if np.shape(geom['tilt'][0]) == ():
                    if isinstance(geom['tilt'], numbers.Number):
                        tilt=geom['tilt']
                    else:
                        tilt=0
                    # print(geom)
                    data = do_tilt(data,tilt)
                else:
                    data = four_point_transform(data,geom['tilt'][0])

                print(fname)
                #binnedData = binData(data,geom['bin00b'],geom['binInterv'],check_overExp=False)
                binnedData,oExpLim = binData(data,geom['bin00b'],geom['binInterv'],check_overExp=True)
                binnedData_collect.append(binnedData)
            # plt.figure();plt.plot((np.mean(binnedData_collect,axis=0)).T);plt.title(fname);plt.pause(0.001)
            # plt.figure();plt.imshow(data, origin='lower');plt.title(fname);plt.pause(0.001)
            #plt.figure();plt.plot(np.mean(data,axis=0));plt.show()
            # print(np.shape(binnedData))
            #print(np.shape(oExpLim))

            binnedData=np.sum(binnedData_collect,axis=0)

            binSpec = np.sum(binnedData,axis=0)
            peaks = find_peaks(binSpec)[0]    # horizontal postitions of spectral lines
            peaks = peaks[  np.logical_and(peaks>15, peaks<Nx-15) ] #I get rid of artefacts due to rotation and tilting
            proms = get_proms(binSpec,peaks)[0]
            temp=[peaks[0]]            # here I want to avoid that molecular lines between H beta and gamma are used
            for i in range(1,len(peaks)):
                if proms[i]>np.max([0,*proms[:i]]):
                    temp.append(peaks[i])
            peaks=np.array(temp)
            proms = get_proms(binSpec,peaks)[0]
            binSpec_peaks=binSpec[peaks]
            if (np.sort(proms)[-4]/np.sort(proms)[-5] < 1.2):   # test in case the fourth peak is not resolvable
                nLines = 3
                iLines = sorted(peaks[np.argpartition(-proms, nLines - 1)[:nLines]] + 2)  # select the peaks with the highest prominences
                binSpec_peaks_test = sorted(binSpec_peaks[np.argpartition(-proms, nLines - 1)[:nLines]] + 2)
                if binSpec_peaks_test[-1]>binSpec_peaks_test[-2]:
                    coefs = np.polyfit(iLines,np.flip(waveLengths[1:nLines+1],axis=0),2)
                    temp_score = np.sum((np.polyval(coefs, iLines) - np.flip(waveLengths[1:nLines + 1], axis=0)) ** 2)
                else:
                    coefs = np.polyfit(iLines, waveLengths[1:nLines + 1], 2)
                    temp_score = np.sum((np.polyval(coefs, iLines) - waveLengths[1:nLines + 1]) ** 2)
                bestcoefs = np.array([ coefs ]).T
                score = [[1]]
                print('fourth line too weak for detection, '+str((np.sort(proms)[-4]/np.sort(proms)[-5]))+' respect to fifth')
                print('across all picture, with ' + str(nLines) + ' lines , ' + str(coefs) + ' score ' + str(temp_score))

            else:
                bestcoefs = np.zeros([3,nCh])
                score = []
                for i in range(nCh):
                    nLines = 4
                    binSpec = binnedData[i,oExpLim[i]:]
                    peaks = find_peaks(binSpec)[0]    # horizontal postitions of spectral lines
                    peaks = peaks[  np.logical_and(peaks>15, peaks<Nx-15) ] #I get rid of artefacts due to rotation and tilting
                    proms = get_proms(binSpec,peaks)[0]
                    temp=[peaks[0]]            # here I want to avoid that molecular lines between H beta and gamma are used
                    for i1 in range(1,len(peaks)):
                        if proms[i1]>np.max([0,*proms[:i1]]):
                            temp.append(peaks[i1])
                    peaks=np.array(temp)
                    proms = get_proms(binSpec,peaks)[0]
                    binSpec_peaks=binSpec[peaks]
                    proms = get_proms(binSpec,peaks)[0]
                    peaks += oExpLim[i]
                    iLines = sorted(peaks[np.argpartition(-proms,nLines-1)[:nLines]]+2) #select the peaks with the highest prominences
                    binSpec_peaks_test = sorted(binSpec_peaks[np.argpartition(-proms,nLines-1)[:nLines]]+2)
                    if binSpec_peaks_test[-1]>binSpec_peaks_test[-2]:
                        coefs = np.polyfit(iLines,np.flip(waveLengths[1:nLines+1],axis=0),2)
                        temp_score = np.sum((np.polyval(coefs, iLines) - np.flip(waveLengths[1:nLines + 1], axis=0)) ** 2)
                    else:
                        coefs = np.polyfit(iLines, waveLengths[1:nLines + 1], 2)
                        temp_score = np.sum((np.polyval(coefs, iLines) - waveLengths[1:nLines + 1]) ** 2)

                    # bestcoefs[:, i] = np.nan
                    while ( (abs(Nx*Nx*coefs[0])< 0 or coefs[1]<0) and (nLines < len(waveLengths[1:])) ):    # it was Nx*Nx*coefs[0])< 2     before
                        bestcoefs[:,i] = coefs
                        nLines += 1
                        if nLines==len(peaks):
                            nLines -=1
                            break
                        iLines = sorted(peaks[np.argpartition(-proms,nLines-1)[:nLines]]+2)
                        binSpec_peaks_test = sorted(binSpec_peaks[np.argpartition(-proms, nLines - 1)[:nLines]] + 2)
                        if binSpec_peaks_test[-1] > binSpec_peaks_test[-2]:
                            coefs = np.polyfit(iLines, np.flip(waveLengths[1:nLines + 1], axis=0), 2)
                            temp_score = np.sum((np.polyval(coefs,iLines)-np.flip(waveLengths[1:nLines + 1], axis=0))**2)
                        else:
                            coefs = np.polyfit(iLines, waveLengths[1:nLines + 1], 2)
                            temp_score = np.sum((np.polyval(coefs,iLines)-waveLengths[1:nLines + 1])**2)
                        # coefs = np.polyfit(iLines,waveLengths[1:nLines+1],2)
                        # print(coefs)

                    if nLines==len(waveLengths[1:]):
                        coefs = [np.nan,np.nan,np.nan]
                        temp_score=10000000000000000000
                    score.append(temp_score)
                    print('channel '+str(i)+' with lines '+str(nLines)+' , '+str(coefs)+' score '+str(temp_score))
                    bestcoefs[:,i] = coefs
                    '''
                    plt.figure()
                    plt.plot(binnedData[i,:])
                    plt.plot(iLines,binnedData[i,iLines]+(max(binnedData[i,:])-min(binnedData[i,:]))/10,'vr')
                    '''
            waveLfits[:,iFile,k] = np.nansum((bestcoefs/np.array(score)).astype(float),axis=1)/np.nansum(1/np.array(score))
        # else:
        #     waveLfits[:,iFile,k] = np.nan

            '''
            np.nanmedian(waveLfits[0]*Nx*Nx)
            np.nanmedian(waveLfits[1]*Nx)
            np.nanmedian(waveLfits[2])

            np.nanstd(waveLfits[0]*Nx*Nx)
            np.nanstd(waveLfits[1]*Nx)
            np.nanstd(waveLfits[2])

            -0.43758502057226456
            -180.590510870422
            508.53625179701703

            0.16213891905160668
            0.14046100012705637
            0.05334330335232047
            '''
    Hbcoefs = np.nanmedian(waveLfits.reshape([3,len(df_settings)*2]),axis=1)

    Halocs = []
    all_j = find_index_of_file(merge_ID, df_settings, df_log, type='Ha')
    if all_j != []:
        for iFile, j in enumerate(all_j):
            (folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
            # for iFile in range(len(df_settings)):
            #     j = df_settings.loc[iFile,'Ha']
            #     if not np.isnan(j):
            #         (folder,date,sequence) = df_log.loc[j,['folder','date','sequence']]
            dataDark = load_dark(j, df_settings, df_log, fdir, geom)
            # jDark = np.where((df_log['folder']==df_log.loc[j,'Dark_folder']) & (df_log['sequence']==df_log.loc[j,'Dark_sequence']) & (df_log['type'] == 'Dark'))[0][0]
            # (fDark,dDark,sDark) = df_log.loc[jDark,['folder','date','sequence']]
            # type = '.tif'
            # filenames_Dark = all_file_names(fdir+'/'+fDark+'/'+"{0:0=2d}".format(sDark)+'/Untitled_1/Pos0', type)
            # dataDark=[]
            # for index,filename in enumerate(filenames_Dark[:5]):
            #     fname = fdir+'/'+fDark+'/'+"{0:0=2d}".format(sDark)+'/Untitled_1/Pos0/'+filename
            #     im = Image.open(fname)
            #     data = np.array(im)
            #     dataDark.append(data)
            # dataDark=np.mean(dataDark,axis=0)

            type = '.tif'
            filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)
            binnedData_collect=[]
            for index,filename in enumerate(filenames):
                fname = fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0/'+filename

                #file = SpeFile(fdir+fname)
                #data = file.data[0].T
                im = Image.open(fname)
                data = np.array(im)


                #fnDark = fDark+'/'+dDark+tDark+'.spe'
                #fileDark = SpeFile(fdir+fnDark)
                #dataDark = fileDark.data[0].T
                #data =data - dataDark
                data =data - dataDark
                data = rotate(data,geom['angle'])
                data = do_tilt(data,tilt)

                binnedData = binData(data,geom['bin00a'],geom['binInterv'])
                Halocs.append(np.median(binnedData.argmax(axis=1)))
        Hacoefs = Hbcoefs.copy()
        Hacoefs[-1] += waveLengths[0] - np.polyval(Hacoefs,np.median(Halocs))
    else:
        Hacoefs = [np.nan,np.nan,np.nan]
    waveLCalcoefs = np.vstack([Hacoefs,Hbcoefs])
    return waveLCalcoefs


def do_waveL_Calib_simplified(binnedData,type='Hb',oExpLim='auto'):
    import numpy as np
    from matplotlib import pyplot as plt
    import os
    from functions.winspec import SpeFile
    from scipy.signal import find_peaks, peak_prominences as get_proms
    from scipy.optimize import curve_fit
    from scipy.special import erf
    from PIL import Image
    import numbers
    import pandas as pd

    nCh,Nx = np.shape(binnedData)

    n = np.arange(10,20)
    waveLengths = [656.45377,486.13615,434.0462,410.174,397.0072,388.9049,383.5384] + list((364.6*(n*n)/(n*n-4)))
    if oExpLim=='auto':
        oExpLim = np.zeros((nCh)).astype(int)

    waveLfits = np.zeros([3,2])*np.nan


    for k,typ in enumerate(['Hb','Hc']):
        # for iFile in range(len(df_settings)):
            # Find vertical position of fibres for low wavelength
            # j = df_settings.loc[iFile,typ]
            # if not np.isnan(j):
        if type!=typ:
            continue
        if k>0:
            continue
            # all_j_0 = find_index_of_file(merge_ID, df_settings, df_log, type=typ)
        Hbcoefs = []
        binSpec = np.sum(binnedData,axis=0)
        peaks = find_peaks(binSpec)[0]    # horizontal postitions of spectral lines
        peaks = peaks[np.logical_and(peaks>15, peaks<Nx-15) ] #I get rid of artefacts due to rotation and tilting
        proms = get_proms(binSpec,peaks)[0]
        temp=[peaks[0]]            # here I want to avoid that molecular lines between H beta and gamma are used
        for i in range(1,len(peaks)):
            if proms[i]>np.max([0,*proms[:i]]):
                temp.append(peaks[i])
        peaks=np.array(temp)
        proms = get_proms(binSpec,peaks)[0]
        binSpec_peaks=binSpec[peaks]
        if (np.sort(proms)[-4]/np.sort(proms)[-5] < 1.2):   # test in case the fourth peak is not resolvable
            nLines = 3
            iLines = sorted(peaks[np.argpartition(-proms, nLines - 1)[:nLines]] + 2)  # select the peaks with the highest prominences
            binSpec_peaks_test = sorted(binSpec_peaks[np.argpartition(-proms, nLines - 1)[:nLines]] + 2)
            if binSpec_peaks_test[-1]>binSpec_peaks_test[-2]:
                coefs = np.polyfit(iLines,np.flip(waveLengths[1:nLines+1],axis=0),2)
                temp_score = np.sum((np.polyval(coefs, iLines) - np.flip(waveLengths[1:nLines + 1], axis=0)) ** 2)
            else:
                coefs = np.polyfit(iLines, waveLengths[1:nLines + 1], 2)
                temp_score = np.sum((np.polyval(coefs, iLines) - waveLengths[1:nLines + 1]) ** 2)
            bestcoefs = np.array([ coefs ]).T
            score = [[1]]
            print('fourth line too weak for detection, '+str((np.sort(proms)[-4]/np.sort(proms)[-5]))+' respect to fifth')
            print('across all picture, with ' + str(nLines) + ' lines , ' + str(coefs) + ' score ' + str(temp_score))

        else:
            bestcoefs = np.zeros([3,nCh])
            score = []
            for i in range(nCh):
                nLines = 4
                binSpec = binnedData[i,oExpLim[i]:]
                peaks = find_peaks(binSpec)[0]    # horizontal postitions of spectral lines
                peaks = peaks[  np.logical_and(peaks>15, peaks<Nx-15) ] #I get rid of artefacts due to rotation and tilting
                proms = get_proms(binSpec,peaks)[0]
                temp=[peaks[0]]            # here I want to avoid that molecular lines between H beta and gamma are used
                for i1 in range(1,len(peaks)):
                    if proms[i1]>np.max([0,*proms[:i1]]):
                        temp.append(peaks[i1])
                peaks=np.array(temp)
                proms = get_proms(binSpec,peaks)[0]
                binSpec_peaks=binSpec[peaks]
                proms = get_proms(binSpec,peaks)[0]
                peaks += oExpLim[i]
                iLines = sorted(peaks[np.argpartition(-proms,nLines-1)[:nLines]]+2) #select the peaks with the highest prominences
                binSpec_peaks_test = sorted(binSpec_peaks[np.argpartition(-proms,nLines-1)[:nLines]]+2)
                if binSpec_peaks_test[-1]>binSpec_peaks_test[-2]:
                    coefs = np.polyfit(iLines,np.flip(waveLengths[1:nLines+1],axis=0),2)
                    temp_score = np.sum((np.polyval(coefs, iLines) - np.flip(waveLengths[1:nLines + 1], axis=0)) ** 2)
                else:
                    coefs = np.polyfit(iLines, waveLengths[1:nLines + 1], 2)
                    temp_score = np.sum((np.polyval(coefs, iLines) - waveLengths[1:nLines + 1]) ** 2)

                # bestcoefs[:, i] = np.nan
                while ( (abs(Nx*Nx*coefs[0])< 0 or coefs[1]<0) and (nLines < len(waveLengths[1:])) ):    # it was Nx*Nx*coefs[0])< 2     before
                    bestcoefs[:,i] = coefs
                    nLines += 1
                    if nLines==len(peaks):
                        nLines -=1
                        break
                    iLines = sorted(peaks[np.argpartition(-proms,nLines-1)[:nLines]]+2)
                    binSpec_peaks_test = sorted(binSpec_peaks[np.argpartition(-proms, nLines - 1)[:nLines]] + 2)
                    if binSpec_peaks_test[-1] > binSpec_peaks_test[-2]:
                        coefs = np.polyfit(iLines, np.flip(waveLengths[1:nLines + 1], axis=0), 2)
                        temp_score = np.sum((np.polyval(coefs,iLines)-np.flip(waveLengths[1:nLines + 1], axis=0))**2)
                    else:
                        coefs = np.polyfit(iLines, waveLengths[1:nLines + 1], 2)
                        temp_score = np.sum((np.polyval(coefs,iLines)-waveLengths[1:nLines + 1])**2)
                    # coefs = np.polyfit(iLines,waveLengths[1:nLines+1],2)
                    # print(coefs)

                if nLines==len(waveLengths[1:]):
                    coefs = [np.nan,np.nan,np.nan]
                    temp_score=10000000000000000000
                score.append(temp_score)
                print('channel '+str(i)+' with lines '+str(nLines)+' , '+str(coefs)+' score '+str(temp_score))
                bestcoefs[:,i] = coefs
                '''
                plt.figure()
                plt.plot(binnedData[i,:])
                plt.plot(iLines,binnedData[i,iLines]+(max(binnedData[i,:])-min(binnedData[i,:]))/10,'vr')
                '''
        Hbcoefs = np.nansum((bestcoefs/np.array(score)).astype(float),axis=1)/np.nansum(1/np.array(score))

    Hacoefs = [np.nan,np.nan,np.nan]
    Halocs = []
    if typ=='Ha':
        Halocsnp=median(binnedData.argmax(axis=1))
        Hacoefs = Hbcoefs.copy()
        Hacoefs[-1] += waveLengths[0] - np.polyval(Hacoefs,np.median(Halocs))
    else:
        Hacoefs = [np.nan,np.nan,np.nan]
    waveLCalcoefs = np.vstack([Hacoefs,Hbcoefs])
    return waveLCalcoefs


def do_waveL_Calib_2(merge_ID, df_settings, df_log, geom,binnedSens, nCh=40, Nx=1608,fdir='/home/ff645/GoogleDrive/ff645/YPI - Fabio/Collaboratory/Experimental data/tests/test_data_folder'):
    import numpy as np
    from matplotlib import pyplot as plt
    import os
    from functions.winspec import SpeFile
    from .spectools import rotate, do_tilt, binData
    from scipy.signal import find_peaks, peak_prominences as get_proms
    from scipy.optimize import curve_fit
    from scipy.special import erf
    from .fabio_add import find_nearest_index, multi_gaussian, all_file_names, load_dark, find_index_of_file, \
        get_metadata, movie_from_data, get_angle_no_lines, do_tilt_no_lines, four_point_transform, fix_minimum_signal, \
        get_bin_and_interv_no_lines
    from PIL import Image
    import numbers

    # fdir = '/home/ff645/Downloads/OES Fabio/data/'

    n = np.arange(10, 20)
    waveLengths = [656.45377, 486.13615, 434.0462, 410.174, 397.0072, 388.9049, 383.5384] + list(
        (364.6 * (n * n) / (n * n - 4)))

    waveLfits = np.zeros([3, len(df_settings), 2]) * np.nan

    for k, typ in enumerate(['Hb', 'Hc']):
        # for iFile in range(len(df_settings)):
        # Find vertical position of fibres for low wavelength
        # j = df_settings.loc[iFile,typ]
        # if not np.isnan(j):
        if k == 0:
            all_j_0 = find_index_of_file(merge_ID, df_settings, df_log, type=typ)

        all_j = find_index_of_file(merge_ID, df_settings, df_log, type=typ)
        if all_j == []:
            continue
        for iFile, j in enumerate(all_j):
            if k > 0:
                if j in all_j_0:
                    continue
            (folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]

            # jDark = np.where((df_log['folder']==df_log.loc[j,'Dark_folder']) & (df_log['sequence']==df_log.loc[j,'Dark_sequence']) & (df_log['type'] == 'Dark'))[0][0]
            # (fDark,dDark,sDark) = df_log.loc[jDark,['folder','date','sequence']]
            # type = '.tif'
            # filenames_Dark = all_file_names(fdir+'/'+fDark+'/'+"{0:0=2d}".format(sDark)+'/Untitled_1/Pos0', type)
            # dataDark=[]
            # for index,filename in enumerate(filenames_Dark[:5]):
            #    fname = fdir+'/'+fDark+'/'+"{0:0=2d}".format(sDark)+'/Untitled_1/Pos0/'+filename
            #    im = Image.open(fname)
            #    data = np.array(im)
            #    dataDark.append(data)
            # dataDark=np.mean(dataDark,axis=0)
            dataDark = load_dark(j, df_settings, df_log, fdir, geom)

            type = '.tif'
            filenames = all_file_names(
                fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
            binnedData_collect = []
            for index, filename in enumerate(filenames):
                fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(
                    untitled) + '/Pos0/' + filename

                # file = SpeFile(fdir+fname)
                # data = file.data[0].T
                im = Image.open(fname)
                data = np.array(im)

                # fnDark = fDark+'/'+dDark+tDark+'.spe'
                # fileDark = SpeFile(fdir+fnDark)
                # dataDark = fileDark.data[0].T
                # data =data - dataDark
                data = data - dataDark
                data = rotate(data, geom['angle'])
                # print(geom['tilt'])
                if isinstance(geom['tilt'], numbers.Number):
                    tilt = geom['tilt']
                else:
                    tilt = 0
                # print(geom)
                data = do_tilt(data, tilt)

                print(fname)
                # binnedData = binData(data,geom['bin00b'],geom['binInterv'],check_overExp=False)
                binnedData, oExpLim = binData(data, geom['bin00b'], geom['binInterv'], check_overExp=True)
                binnedData_collect.append(binnedData)
            # plt.figure();plt.plot((np.mean(binnedData_collect,axis=0)).T);plt.title(fname);plt.pause(0.001)
            # plt.figure();plt.imshow(data, origin='lower');plt.title(fname);plt.pause(0.001)
            # plt.figure();plt.plot(np.mean(data,axis=0));plt.show()
            # print(np.shape(binnedData))
            # print(np.shape(oExpLim))

        binnedData = np.sum(binnedData_collect, axis=0)
        # binnedData = binnedData/binnedSens[1]
        binnedData = np.sum(binnedData,axis=(0))
        bestcoefs = np.zeros((3))
        # for i in range(nCh):
        nLines = 3
        binSpec = binnedData
        peaks = find_peaks(binSpec)[0]  # horizontal postitions of spectral lines
        binSpec_peaks = binSpec[peaks]
        proms = get_proms(binSpec, peaks)[0]
        # peaks += oExpLim[i]
        iLines = sorted(peaks[np.argpartition(-proms, nLines - 1)[
                              :nLines]] + 2)  # select the peaks with the highest prominences
        binSpec_peaks_test = sorted(binSpec_peaks[np.argpartition(-proms, nLines - 1)[:nLines]] + 2)
        if binSpec_peaks_test[-1] > binSpec_peaks_test[-2]:
            coefs = np.polyfit(iLines, np.flip(waveLengths[1:nLines + 1], axis=0), 2)
        else:
            coefs = np.polyfit(iLines, waveLengths[1:nLines + 1], 2)

        bestcoefs[:] = coefs
        # bestcoefs[:, i] = np.nan
        waveLdiff = np.polyval(coefs, waveLengths[1:nLines + 1]) - waveLengths[1:nLines + 1]
        waveLdiff_test = np.power(np.sum(np.power(waveLdiff, 2)), 0.5)
        while ((abs(Nx * Nx * coefs[0]) < 2 or waveLdiff_test > 10) and nLines < 6):
            bestcoefs[:] = coefs
            nLines += 1
            iLines = sorted(peaks[np.argpartition(-proms, nLines - 1)[:nLines]] + 2)
            binSpec_peaks_test = sorted(binSpec_peaks[np.argpartition(-proms, nLines - 1)[:nLines]] + 2)
            if binSpec_peaks_test[-1] > binSpec_peaks_test[-2]:
                coefs = np.polyfit(iLines, np.flip(waveLengths[1:nLines + 1], axis=0), 2)
            else:
                coefs = np.polyfit(iLines, waveLengths[1:nLines + 1], 2)
            waveLdiff = np.polyval(coefs, waveLengths[1:nLines + 1]) - waveLengths[1:nLines + 1]
            waveLdiff_test = np.power(np.sum(np.power(waveLdiff, 2)), 0.5)
            # coefs = np.polyfit(iLines,waveLengths[1:nLines+1],2)
            print(coefs)
            print(waveLdiff)
            print(waveLdiff_test)
        if not (abs(Nx * Nx * coefs[0]) < 2 and waveLdiff_test > 10):
            print('fit failed for chord ' )
            bestcoefs[:] = np.ones_like(coefs) * np.nan

        '''
        plt.figure()
        plt.plot(binnedData[i,:])
        plt.plot(iLines,binnedData[i,iLines]+(max(binnedData[i,:])-min(binnedData[i,:]))/10,'vr')
        '''
        waveLfits[:, iFile, k] = np.nanmedian(bestcoefs, axis=1)
        # else:
        #     waveLfits[:,iFile,k] = np.nan

        '''
        np.nanmedian(waveLfits[0]*Nx*Nx)
        np.nanmedian(waveLfits[1]*Nx)
        np.nanmedian(waveLfits[2])

        np.nanstd(waveLfits[0]*Nx*Nx)
        np.nanstd(waveLfits[1]*Nx)
        np.nanstd(waveLfits[2])

        -0.43758502057226456
        -180.590510870422
        508.53625179701703

        0.16213891905160668
        0.14046100012705637
        0.05334330335232047
        '''


    Hbcoefs = np.nanmedian(waveLfits.reshape([3, len(df_settings) * 2]), axis=1)

    Halocs = []
    all_j = find_index_of_file(merge_ID, df_settings, df_log, type='Ha')
    if all_j != []:
        for iFile, j in enumerate(all_j):
            (folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]
            # for iFile in range(len(df_settings)):
            #     j = df_settings.loc[iFile,'Ha']
            #     if not np.isnan(j):
            #         (folder,date,sequence) = df_log.loc[j,['folder','date','sequence']]
            dataDark = load_dark(j, df_settings, df_log, fdir, geom)
            # jDark = np.where((df_log['folder']==df_log.loc[j,'Dark_folder']) & (df_log['sequence']==df_log.loc[j,'Dark_sequence']) & (df_log['type'] == 'Dark'))[0][0]
            # (fDark,dDark,sDark) = df_log.loc[jDark,['folder','date','sequence']]
            # type = '.tif'
            # filenames_Dark = all_file_names(fdir+'/'+fDark+'/'+"{0:0=2d}".format(sDark)+'/Untitled_1/Pos0', type)
            # dataDark=[]
            # for index,filename in enumerate(filenames_Dark[:5]):
            #     fname = fdir+'/'+fDark+'/'+"{0:0=2d}".format(sDark)+'/Untitled_1/Pos0/'+filename
            #     im = Image.open(fname)
            #     data = np.array(im)
            #     dataDark.append(data)
            # dataDark=np.mean(dataDark,axis=0)

            type = '.tif'
            filenames = all_file_names(
                fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
            binnedData_collect = []
            for index, filename in enumerate(filenames):
                fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(
                    untitled) + '/Pos0/' + filename

                # file = SpeFile(fdir+fname)
                # data = file.data[0].T
                im = Image.open(fname)
                data = np.array(im)

                # fnDark = fDark+'/'+dDark+tDark+'.spe'
                # fileDark = SpeFile(fdir+fnDark)
                # dataDark = fileDark.data[0].T
                # data =data - dataDark
                data = data - dataDark
                data = rotate(data, geom['angle'])
                data = do_tilt(data, tilt)

                binnedData = binData(data, geom['bin00a'], geom['binInterv'])
                Halocs.append(np.median(binnedData.argmax(axis=1)))
        Hacoefs = Hbcoefs.copy()
        Hacoefs[-1] += waveLengths[0] - np.polyval(Hacoefs, np.median(Halocs))
    else:
        Hacoefs = [np.nan, np.nan, np.nan]
    waveLCalcoefs = np.vstack([Hacoefs, Hbcoefs])
    return waveLCalcoefs


def do_Intensity_Calib(merge_ID,df_settings,df_log,df_calLog,df_sphere,geom_null,waveLcoefs,nCh=40,Nx=1608,waves_to_calibrate=['Ha','Hb','Hc'],fdir = '/home/ff645/GoogleDrive/ff645/YPI - Fabio/Collaboratory/Experimental data/tests/test_data_folder'):
    # from .winspec import SpeFile
    from .spectools import rotate,do_tilt,binData,getFirstBin
    import numpy as np
    from scipy.interpolate import interp1d
    from .fabio_add import all_file_names,order_filenames,order_filenames_csv,load_dark, find_index_of_file
    from PIL import Image
    import numbers

    #nCh=40;Nx=2048

    # fdir = '/home/ff645/Downloads/OES Fabio/data/'

    spline=interp1d(df_sphere.waveL,df_sphere.Rad,kind='cubic') #Spline interp of sphere radiance

    #Planck gives poor fit to data
    '''
    from scipy.optimize import curve_fit
    c = 299792458
    h = 6.62607004e-34
    k = 1.38064852e-23
    planck = lambda x,A,T: A*2*h*c*c/(x/1e9)**5/(np.exp(h*c/(x/1e9)/k/T)-1)
    fit = curve_fit(planck,df_sphere.waveL,df_sphere.Rad,p0 = [1e-15,5e3])
    plt.figure();plt.plot(df_sphere.waveL,df_sphere.Rad)
    plt.plot(df_sphere.waveL,planck(df_sphere.waveL,*fit[0]))
    '''
    #Spline works well for 350-700nm
    '''
    plt.figure();plt.plot(df_sphere.waveL,df_sphere.Rad)
    waveL = np.arange(310,2400,1)
    plt.plot(waveL,spline(waveL))
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Sphere radiance [W m-2 sr-1 nm-1 $\mu$A-1]')
    '''

    sensBinned = np.zeros([3,nCh,Nx])

    for i,typ in enumerate(waves_to_calibrate):
        all_j = find_index_of_file(merge_ID, df_settings, df_log,type=typ)
        if all_j==[]:
            continue
        # j0 = list(df_calLog['exper']).index(typ)
        #
        # jLong = df_calLog.t_exp.loc[np.logical_and(df_calLog.gratPos == df_calLog.gratPos.loc[j0],df_calLog.type == df_calLog.type.loc[j0])].argmax()

        # (folder,date,time) = df_calLog.loc[jLong,['folder','date','time']]
        # fname = folder+'/'+date+time+'.spe'
        #
        # file = SpeFile(fdir+fname)
        # data = file.data[0].T.astype('float32')
        #
        # jDark = np.where((df_calLog['time']==df_calLog.loc[jLong,'Dark']) & (df_calLog['type'] == 'Dark'))[0][0]
        #
        # (fDark,dDark,tDark) = df_calLog.loc[jDark,['folder','date','time']]
        # fnDark = fDark+'/'+dDark+tDark+'.spe'
        # fileDark = SpeFile(fdir+fnDark)
        # dataDark = fileDark.data[0].T.astype('float32')
        #
        # data -= dataDark
        # data = rotate(data,geom['angle'])
        # data = do_tilt(data,geom['tilt'])

        data_all = []
        bin0_all = []
        temp=[]
        for iFile, j in enumerate(all_j):
            temp.append(df_log.loc[j, ['Calib_ID']])
        calib_ID = np.unique(temp).astype(int)
        for iFile, j in enumerate(calib_ID):
            (folder, date, sequence, untitled) = df_log.loc[j, ['folder', 'date', 'sequence', 'untitled']]

            dataDark = load_dark(j, df_settings, df_log, fdir, geom_null)

            type = '.tif'
            filenames = all_file_names(fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0', type)
            data_sum = 0
            temp = []
            for index, filename in enumerate(filenames):
                fname = fdir + '/' + folder + '/' + "{0:0=2d}".format(sequence) + '/Untitled_' + str(untitled) + '/Pos0/' + filename
                # print(fname)
                im = Image.open(fname)
                data = np.array(im)
                # data = data - dataDark
                # data = rotate(data, geom['angle'])
                # data = do_tilt(data, geom['tilt'])

                # noise = data[:20].tolist()+data[-20:].tolist()
                # noise_std = np.std(noise,axis=(0,1))
                # noise = np.mean(noise,axis=(0,1))
                # test_std=10000
                # factor_std = 0
                # std_record=[]
                # while factor_std<=2:
                #     reconstruct = []
                #     for irow,row in enumerate(data.tolist()):
                #         minimum=np.mean(np.sort(row)[:int(len(row)/40)])+factor_std*np.std(np.sort(row)[:int(len(row)/40)])
                #         # minimum = np.min(row)
                #         reconstruct.append(np.array(row)+noise-minimum)
                #     reconstruct=np.array(reconstruct)
                #     test_std = np.std(reconstruct[:,:40])
                #     std_record.append(test_std)
                #     print('factor_std '+str(factor_std))
                #     print('test_std ' + str(test_std))
                #     factor_std +=0.1

                data_sum+=data
                temp.append(data)
            temp = np.mean(temp, axis=0)
            temp = temp - dataDark
            angle = get_angle_no_lines(temp)
            temp = rotate(temp, angle)
            temp = do_tilt_no_lines(temp)
            data_all.append(temp/df_log.loc[j, ['Calib_ID']][0])

            data_sum = data_sum - dataDark*len(filenames)
            angle = get_angle_no_lines(data_sum)
            data_sum = rotate(data_sum, angle)
            data_sum = do_tilt_no_lines(data_sum)

            # data_sum = rotate(data_sum, geom['angle'])
            # data_sum = do_tilt(data_sum, geom['tilt'])
            if typ == 'Ha':
                fracBin = geom.bin00a-round(geom.bin00a)
            else:
                fracBin = geom.bin00b-round(geom.bin00b)
            try:
                first_bin=getFirstBin(data_sum,nLines=1,binInterv=geom.binInterv.loc[0],calibration=True)
            except:
                print('fail')
                continue
            if np.isnan(first_bin):
                print('fail')
                continue
            bin0=round(first_bin)+fracBin
            bin0_all.append(bin0)
        bin0=np.median(bin0_all)
        # bin0=geom.bin00a.loc[0]

        data = np.mean(data_all, axis=0)

        longCal = binData(data,bin0-geom.binInterv,geom['binInterv'],nCh = 42)
        topDark = longCal[0]
        botDark = longCal[-1]
        wtDark = np.arange(1,41)/41
        calBinned = longCal[1:41]-(np.outer(wtDark,topDark)+np.outer(wtDark[::-1],botDark))


        # # TO CHECK: MAYBE THIS BIT IS NOT NECESSARY
        # if typ == 'Ha':
        #     for k in range(len(longCal)):
        #         coef = np.polyfit(np.arange(1440,Nx),longCal[k][1440:Nx],1)
        #         longCal[k][1440:] = np.polyval(coef,np.arange(1440,Nx))
        # else:
        #     longCal-=np.mean(longCal[:,1740:1760],axis=1)[:,np.newaxis]
        #     longCal[:,1760:] = 0

        '''
        plt.figure();plt.plot(np.polyval(waveLcoefs[int(typ!='Ha')],np.arange(data.shape[0])),longCal.T)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Intensity [counts] - %i ms'%df_calLog.t_exp[jLong])
        '''

        # if  j0 == jLong:
        # calBinned = longCal/df_calLog.I_det.loc[jLong]
        # else:
        #     # (folder,date,time) = df_calLog.loc[j0,['folder','date','time']]
        #     # fname = folder+'/'+date+time+'.spe'
            # #
        #     # file = SpeFile(fdir+fname)
        #     # data = file.data[0].T.astype('float32')
        #     #
        #     # jDark = np.where((df_calLog['time']==df_calLog.loc[j0,'Dark']) & (df_calLog['type'] == 'Dark'))[0][0]
        #     #
        #     # (fDark,dDark,tDark) = df_calLog.loc[jDark,['folder','date','time']]
        #     # fnDark = fDark+'/'+dDark+tDark+'.spe'
        #     # fileDark = SpeFile(fdir+fnDark)
        #     # dataDark = fileDark.data[0].T.astype('float32')
        #     #
        #     # data -= dataDark
        #     # data = rotate(data,geom['angle'])
        #     # data = do_tilt(data,geom['tilt'])
        #
        #
        #
        #     shortCal = binData(data,bin0-geom.binInterv,geom['binInterv'],nCh = 42)
        #     topDark = shortCal[0]
        #     botDark = shortCal[-1]
        #     wtDark = np.arange(1,41)/41
        #     shortCal = shortCal[1:41]-(np.outer(wtDark,topDark)+np.outer(wtDark[::-1],botDark))
        #
        #     # TO CHECK: MAYBE THIS BIT IS NOT NECESSARY
        #     # if typ == 'Ha':
        #     #     for k in range(len(shortCal)):
        #     #         coef = np.polyfit(np.arange(1440,Nx),shortCal[k][1440:Nx],1)
        #     #         shortCal[k][1440:] = np.polyval(coef,np.arange(1440,Nx))
        #     # else:
        #     #     shortCal-=np.mean(longCal[:,1740:1760],axis=1)[:,np.newaxis]
        #     #     shortCal[:,1760:] = 0
        #     calBinned = longCal*(np.sum(shortCal)/np.sum(longCal)) / df_calLog.I_det.loc[j0]
        if typ == 'Ha':
            sensBinned[0] = calBinned / spline(np.polyval(waveLcoefs[0],np.arange(data.shape[1])))
        elif typ == 'Hb':
            sensBinned[1] = calBinned / spline(np.polyval(waveLcoefs[1],np.arange(data.shape[1])))
        elif typ == 'Hc':
            sensBinned[2] = calBinned / spline(np.polyval(waveLcoefs[2],np.arange(data.shape[1])))
    return sensBinned
'''
        plt.figure();plt.plot(np.polyval(waveLcoefs[int(typ!='Ha')],np.arange(data.shape[0])),calBinned.T)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Intensity [counts/$\mu$A] - %i ms'%df_calLog.t_exp[j0])

        plt.figure();plt.plot(np.polyval(waveLcoefs[int(typ!='Ha')],np.arange(data.shape[0])),sensBinned[i].T)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Sensitivity [counts/(W m-2 sr-1 nm-1)] - %i ms'%df_calLog.t_exp[j0])
'''
'''
j = 7

(folder,date,time) = df_calLog.loc[j,['folder','date','time']]
fname = folder+'/'+date+time+'.spe'

file = SpeFile(fdir+fname)
data = file.data[0].T.astype('float32')
data = rotate(data,geom['angle'])
data = do_tilt(data,geom['tilt'])
bin0 = round(getFirstBin(data,nLines=2,binInterv=geom.binInterv.loc[0]))+geom.bin00a-round(geom.bin00a)
Halamp = binData(data,bin0,geom['binInterv'],nCh = 40)
plt.figure();plt.plot(np.polyval(waveLcoefs[0],np.arange(data.shape[0])),Halamp.T)


j = 8

(folder,date,time) = df_calLog.loc[j,['folder','date','time']]
fname = folder+'/'+date+time+'.spe'

file = SpeFile(fdir+fname)
data = file.data[0].T.astype('float32')
data = rotate(data,geom['angle'])
data = do_tilt(data,geom['tilt'])
bin0 = round(getFirstBin(data,nLines=2,binInterv=geom.binInterv.loc[0]))+geom.bin00b-round(geom.bin00b)
Hblamp = binData(data,bin0-binInterv*10,geom['binInterv'],nCh = 60)
plt.figure();plt.plot(np.polyval(waveLcoefs[1],np.arange(data.shape[0])),Hblamp.T)
'''
'''
    plt.figure()
    plt.plot(np.polyval(waveLcoefs[0],np.arange(data.shape[0])),sensBinned[0].T)
    plt.plot(np.polyval(waveLcoefs[1],np.arange(data.shape[0])),sensBinned[1].T)
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Sensitivity [counts/(W m-2 sr-1 nm-1)] - %i ms'%df_calLog.t_exp[j0])
    '''
'''
    for typ in ['Ha','Hb','Hc']:
        j0 = list(df_calLog['exper']).index(typ)
        bin0Ca = []
        js = np.where(np.logical_and(df_calLog.gratPos == df_calLog.gratPos.loc[j0],df_calLog.type == df_calLog.type.loc[j0]))[0]
        for j in js:
            (folder,date,time) = df_calLog.loc[j,['folder','date','time']]
            fname = folder+'/'+date+time+'.spe'

            file = SpeFile(fdir+fname)
            data = file.data[0].T.astype('float32')

            df_calLog.loc[j,'Dark']
            df_calLog.loc[df_log['type'] == 'Dark']

            jDark = np.where((df_calLog['time']==df_calLog.loc[j,'Dark']) & (df_calLog['type'] == 'Dark'))[0][0]

            (fDark,dDark,tDark) = df_calLog.loc[jDark,['folder','date','time']]
            fnDark = fDark+'/'+dDark+tDark+'.spe'
            fileDark = SpeFile(fdir+fnDark)
            dataDark = fileDark.data[0].T.astype('float32')

            data -= dataDark
            data = rotate(data,geom['angle'])
            data = do_tilt(data,geom['tilt'])

            bin0Ca.append(getFirstBin(data,nLines=2,binInterv=geom.binInterv.loc[0]))
        bin0Ca = round(np.nanmedian(bin0Ca))+geom.bin00a-round(geom.bin00a)


        calData = np.zeros([len(js),42, 2048])
        for jj,j in enumerate(js):
            (folder,date,time) = df_calLog.loc[j,['folder','date','time']]
            fname = folder+'/'+date+time+'.spe'

            file = SpeFile(fdir+fname)
            data = file.data[0].T.astype('float32')

            df_calLog.loc[j,'Dark']
            df_calLog.loc[df_log['type'] == 'Dark']

            jDark = np.where((df_calLog['time']==df_calLog.loc[j,'Dark']) & (df_calLog['type'] == 'Dark'))[0][0]

            (fDark,dDark,tDark) = df_calLog.loc[jDark,['folder','date','time']]
            fnDark = fDark+'/'+dDark+tDark+'.spe'
            fileDark = SpeFile(fdir+fnDark)
            dataDark = fileDark.data[0].T.astype('float32')

            data -= dataDark
            data = rotate(data,geom['angle'])
            data = do_tilt(data,geom['tilt'])
            calData[jj] = binData(data,geom['bin00b'],geom['binInterv'],nCh = 42)
'''

'''
        plt.figure()
        for jj,j in enumerate(js):
            plt.plot(calData[jj,-2,:]/df_calLog.t_exp[j])

        plt.figure()
        k = 41
        plt.loglog(df_calLog.t_exp[js],np.sum(calData[:,k,:],axis = 1),'.')
        xf = np.logspace(2,4,1e3)
        y = np.sum(calData[:,k,:],axis = 1)
        coefs = np.polyfit(df_calLog.t_exp[js],y,1,w=1/np.sqrt(y))
        yf = np.polyval(coefs,xf)
        plt.plot(xf,yf)


        plt.figure()
        plt.loglog(df_calLog.t_exp[js],np.sum(calData[:,k,:],axis = 1),'.')
        coefs = [y[-1]/df_calLog.t_exp[js[-1]],50*y[-1]/df_calLog.t_exp[js[-1]]]
        yf = np.polyval(coefs,xf)
        plt.plot(xf,yf)


        yplot = np.sum(calData[:,:40,:],axis = 2)/np.vstack([np.array(df_calLog.t_exp[js])]*40).T
        dyplot = np.sqrt(np.sum(calData[:,:40,:],axis = 2))/np.vstack([np.array(df_calLog.t_exp[js])]*40).T
        dyplot/=np.vstack([yplot[-1,:]]*7)
        yplot/=np.vstack([yplot[-1,:]]*7)
        plt.figure()
        for y,dy in zip(yplot.T,dyplot.T):
            plt.errorbar(df_calLog.t_exp[js],y,yerr=dy,ls='',marker='.')
        plt.xscale('log')
'''

# Instrument function: will probably work nicely after intensity calibration
'''
lamb = np.polyval(Hbcoefs,range(2048))
i = 5
plt.figure();plt.plot(lamb,binnedData[i,:])


lamb = np.polyval(waveLCalcoefs,range(2048))
plt.figure();
for x in waveLengths[1:7]:
    ix = np.logical_and(lamb < x + 5,lamb > x - 5)
    y = binnedData[0,ix]
    y -= min(y); y/=max(y)
    plt.plot(lamb[ix]-x,y)

lamb = np.polyval(waveLCalcoefs,range(2048))
plt.figure();
x = waveLengths[3]
ix = np.logical_and(lamb < x + 5,lamb > x - 5)
y = binnedData[7,ix]
y -= min(y); y/=max(y)
plt.plot(lamb[ix]-x,y)


Hacoef = waveLCalcoefs.copy()
Hacoef[-1] += waveLengths[0] - np.polyval(waveLCalcoefs,binnedData[20,:].argmax())
Halamb = np.polyval(Hacoef,range(2048))
plt.figure();plt.plot(Halamb,binnedData[20,:])

'''

'''
blockGauss = lambda x,A,w,sig,x0: A*(erf((x-x0+w)/sig)-erf((x-x0-w)/sig))/2*np.sqrt(np.pi)
triGauss = lambda x,A,a,sig,x0: A*sig/a*((np.exp(-((x-x0)/sig)**2)-np.exp(-((x-x0-a)/sig)**2))/2 + (x-x0)/sig*np.sqrt(np.pi)/2*(erf((a-x+x0)/sig)+erf((x-x0)/sig)))
shapeFun = lambda x,A1,A2,w,sig,x0 : (triGauss(x,(A2-A1),2*w,sig,x0) + blockGauss(x,A1,w,sig,x0+w))/1.77


lineShapeCoefs = np.zeros([4,5])
for h,l0 in enumerate(waveLengths[1:5]):
    iFit = np.logical_and(lamb > l0-5, lamb < l0+5)
    xFit = lamb[iFit]
    yFit = binnedData[i,iFit]
    bgFit = np.polyfit(xFit[[0,-1]],yFit[[0,-1]],1)
    yFit -= np.polyval(bgFit,xFit)
    p0 = [max(yFit),max(yFit)/2,1,.2,l0]

    fit = curve_fit(shapeFun,xFit,yFit,p0)

    plt.figure();
    plt.plot(xFit,yFit,'.')
    plt.plot(xFit,shapeFun(xFit,*fit[0]))
    lineShapeCoefs[h] = fit[0]

'''
