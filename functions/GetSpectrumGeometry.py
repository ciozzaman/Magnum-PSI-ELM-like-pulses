def getGeom(merge_ID,df_settings,df_log,fdir = '/home/ff645/GoogleDrive/ff645/YPI - Fabio/Collaboratory/Experimental data/tests/test_data_folder'):
    import pandas as pd
    import numpy as np
    #from matplotlib import pyplot as plt
    import os
    # from .winspec import SpeFile
    from .spectools import get_angle,rotate,get_tilt,do_tilt,getFirstBin
    from .fabio_add import all_file_names,order_filenames,order_filenames_csv
    from PIL import Image
    import numbers
    import copy 
    
    #fdir = '/home/ff645/Downloads/OES Fabio/data/'
    df_angles = pd.DataFrame(columns=['Ha angle','Ha er','Hb angle','Hb er','Intensity'])
    df_anglesLong = pd.DataFrame(columns=['Hb angle','Hb er','Hc angle','Hc er','Hd angle','Hd er','He angle','He er','Hf angle','Hf er','Hg angle','Hg er','Hh angle','Hh er','Intensity'])
    
    data_mean=[]
    all_angle=[]
    all_er=[]
    for i in range(len(df_settings)):
        data_all=0
        #all_angle=[]
        #all_er=[]
        # Find rotation angle for high wavelength: low wavelength matches this
        # j = df_settings.loc[i,'Hb']
        j, merge_ID_j = df_settings.loc[i, ['Hb', 'merge_ID']]
        if merge_ID_j!=merge_ID:
            continue
        if not np.isnan(j):
            (folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]

            jDark = np.where((df_log['folder']==df_log.loc[j,'Dark_folder']) & (df_log['sequence']==df_log.loc[j,'Dark_sequence']) & (df_log['untitled']==df_log.loc[j,'Dark_untitled']) & (df_log['type'] == 'Dark'))[0][0]
            (fDark,dDark,sDark,uDark) = df_log.loc[jDark,['folder','date','sequence','untitled']]
            type = '.tif'
            filenames_Dark = all_file_names(fdir+'/'+fDark+'/'+"{0:0=2d}".format(sDark)+'/Untitled_'+str(uDark)+'/Pos0', type)
            dataDark=0
            dataDark_count=0
            for index,filename in enumerate(filenames_Dark[:5]):
                fname = fdir+'/'+fDark+'/'+"{0:0=2d}".format(sDark)+'/Untitled_'+str(uDark)+'/Pos0/'+filename
                im = Image.open(fname)
                data = np.array(im)
                dataDark+=data
                dataDark_count+=1
            dataDark=dataDark/dataDark_count

            type = '.tif'
            filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)
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
                #data_mean.append(np.mean(data))
                data_all+=np.array(data)
                #angle, er= get_angle(data,nLines=1)
                #print('single angle='+str(angle))
                #all_angle.append(angle)
                #all_er.append(er)
            angle, er= get_angle(data_all,nLines=3)
            all_angle.append(angle)
            print('angle')
            print(angle)
            '''
            plt.figure()
            plt.imshow(data)    
            plt.figure()
            plt.plot(np.sum(data,axis=0))
            
            plt.figure()
            plt.imshow(dataDark)    
            '''
            #df_angles.loc[i,'Intensity'] = np.mean(data_mean)
            #df_angles.loc[i,['Ha angle','Hb angle']],df_angles.loc[i,['Ha er','Hb er']] = np.median(all_angle),np.median(all_er)
        
    #angle = np.median(df_angles['Ha angle'])
    angle = np.nanmedian(all_angle)  
    #angle = 0.6254567079875826
    print('Angle='+str(angle))
    df_tilt = pd.DataFrame(columns=['tilt','binInterv','bin00'])
    
    all_binInterv=[]
    all_tilt=[]
    all_bin00a=[]
    for i in range(len(df_settings)):
	
        data_all=0
        #all_angle=[]
        #all_er=[]
        # Find rotation angle for high wavelength: low wavelength matches this
        # j = df_settings.loc[i,'Hb']
        j, merge_ID_j = df_settings.loc[i, ['Hb', 'merge_ID']]
        if merge_ID_j!=merge_ID:
            continue
        if not np.isnan(j):
            (folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]

            jDark = np.where((df_log['folder']==df_log.loc[j,'Dark_folder']) & (df_log['sequence']==df_log.loc[j,'Dark_sequence']) & (df_log['untitled']==df_log.loc[j,'Dark_untitled']) & (df_log['type'] == 'Dark'))[0][0]
            (fDark,dDark,sDark,uDark) = df_log.loc[jDark,['folder','date','sequence','untitled']]
            type = '.tif'
            filenames_Dark = all_file_names(fdir+'/'+fDark+'/'+"{0:0=2d}".format(sDark)+'/Untitled_'+str(uDark)+'/Pos0', type)
            dataDark=0
            dataDark_count=0
            for index,filename in enumerate(filenames_Dark[:5]):
                fname = fdir+'/'+fDark+'/'+"{0:0=2d}".format(sDark)+'/Untitled_'+str(uDark)+'/Pos0/'+filename
                im = Image.open(fname)
                data = np.array(im)
                dataDark+=data
                dataDark_count+=1
            dataDark=dataDark/dataDark_count

            type = '.tif'
            filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)
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
                data = rotate(data,angle)
                data_all+=np.array(data)
        all_binInterv.append(get_tilt(data_all,nLines=4)[0])
        all_tilt.append(get_tilt(data_all,nLines=4)[1])
        all_bin00a.append(get_tilt(data_all,nLines=4)[2])
        #df_tilt.loc[i,['binInterv','tilt','bin00']] = get_tilt(data_all)
    print(all_tilt)
    #all_tilt = [x for x in all_tilt if isinstance(x, numbers.Number)]
    #if all_tilt==[]:
    #    tilt=0
    #else:
    #tilt=np.nanmedian(all_tilt)


    
    #tilt,binInterv,bin00a = df_tilt.loc[np.any(~pd.isnull(df_tilt),axis=1)].apply(np.median)
    binInterv=np.nanmedian(all_binInterv)
    tilt=np.nanmedian(all_tilt)
    bin00a=np.nanmedian(all_bin00a)

    print('tilt='+str(tilt))
    print('binInterv='+str(binInterv))
    print('bin00a='+str(bin00a))
    '''
    tilt = -0.02752172070353893
    binInterv = 33.996963562753045
    bin00a = 272.16079733663486
    '''
   
    data = do_tilt(data,tilt)
    bin00b = []
    
    for i in range(len(df_settings)):
        # Find vertical position of fibres for low wavelength
        data_all=0
        # j = df_settings.loc[i,'Hb']
        j, merge_ID_j = df_settings.loc[i, ['Hb', 'merge_ID']]
        if merge_ID_j!=merge_ID:
            continue
        if not np.isnan(j):
            (folder,date,sequence,untitled) = df_log.loc[j,['folder','date','sequence','untitled']]

            jDark = np.where((df_log['folder']==df_log.loc[j,'Dark_folder']) & (df_log['sequence']==df_log.loc[j,'Dark_sequence']) & (df_log['untitled']==df_log.loc[j,'Dark_untitled']) & (df_log['type'] == 'Dark'))[0][0]
            (fDark,dDark,sDark,uDark) = df_log.loc[jDark,['folder','date','sequence','untitled']]
            type = '.tif'
            filenames_Dark = all_file_names(fdir+'/'+fDark+'/'+"{0:0=2d}".format(sDark)+'/Untitled_'+str(uDark)+'/Pos0', type)
            dataDark=0
            dataDark_count=0
            for index,filename in enumerate(filenames_Dark[:5]):
                fname = fdir+'/'+fDark+'/'+"{0:0=2d}".format(sDark)+'/Untitled_'+str(uDark)+'/Pos0/'+filename
                im = Image.open(fname)
                data = np.array(im)
                dataDark+=data
                dataDark_count+=1
            dataDark=dataDark/dataDark_count

            type = '.tif'
            filenames = all_file_names(fdir+'/'+folder+'/'+"{0:0=2d}".format(sequence)+'/Untitled_'+str(untitled)+'/Pos0', type)
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
                data = rotate(data,angle)
                data = do_tilt(data,tilt)
                data_all+=data
        bin00b.append(getFirstBin(data_all,nLines=3,binInterv=binInterv))
    print(bin00b)
    if not isinstance(np.nanmedian(bin00b), numbers.Rational):
        bin00b=copy.deepcopy(bin00a);print('replaced')
    else:
        bin00b=np.nanmedian(bin00b)
    # bin00b = 223.1034637876741
    print(tilt)
    geom = pd.DataFrame(columns = ['angle','tilt','binInterv','bin00a','bin00b'])
    geom.loc[0] = [angle,tilt,binInterv,bin00a,bin00b]
    return geom
