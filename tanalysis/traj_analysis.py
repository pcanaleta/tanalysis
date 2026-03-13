import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def xml_to_xlsx(dirname:str, xyscale:float, zdist:float, dt:int):
    '''
    This function converts all xml files contained in a folder into a merged excel file with all data. One file is created for each condition
    directory tree should be something like:
        -Tracks
            -xml_files  <--- path given to the function
                -Condition1:
                    -1.xml
                    -2.xml
                    -...
                -Condition2:
                    -1.xml
                    -...
                -...
    Converted excel files will be saved in a folder names excel_files which will be contained in the Tracks folder. Excel files will have the condition folders names.

    Args:
        dirname (string): path to the folder containing the folders where xml files are saved
        xyscale (float): conversion rate from pixels to real units
        zdist (floar): real distance between z planes   
        dt (int): time step between frames
    '''
    i = 0
    xlsxdata = []
    for filename in os.listdir(dirname):
        tree = ET.parse(os.path.join(dirname,filename))
        root = tree.getroot()
        for traj in root:
            for timeframe in traj:
                id = i
                t = np.double(timeframe.get('t'))*dt
                x = np.double(timeframe.get('x'))*xyscale
                y = np.double(timeframe.get('y'))*xyscale
                z = np.double(timeframe.get('z'))*zdist
                data = np.array([id, t, x, y, z])
                xlsxdata.append(data)
            i += 1           
    axlsxdata = np.array(xlsxdata)

    #create a directory where excel tracks will be saved
    savedir = os.path.split(dirname)[0].replace('xml_tracks','excel_tracks')
    if not os.path.exists(savedir):
        os.makedirs(os.path.abspath(savedir))
    
    #saving tracks in excel file
    name = os.path.split(dirname)
    savename = fr'{savedir}\{name[-1]}.xlsx'
    df = pd.DataFrame({'id': axlsxdata[:,0], 'time': axlsxdata[:,1], 'x': axlsxdata[:,2], 'y': axlsxdata[:,3], 'z': axlsxdata[:,4]})
    df.to_excel(savename, sheet_name='trajectories', index=False)  
    return

def get_traj(dirname:str):
    '''
    This function gets the tracks in an excel file and crops them in order to have all tracks with the same length as the shortest track.
    Excel files need to have the following column order: [track_id, t, x, y, (z)].

    Args:
        dirname (string): path to the Excel file

    Returns:
        tracks: pandas dataframe containing the position information from each track
        name: name of the track
    '''
    fname = os.path.split(dirname)[-1]
    ext = os.path.splitext(fname)[-1].lower()

    # Read excel documents
    try:
        tracks = pd.read_excel(dirname)
    except:         
        print(f'Could not read: {dirname}')
    
    tracks['frame'] = np.int8(tracks['time']/np.min(np.uint32(np.diff(tracks['time'])))) # Add frame column in dataframe based on time column
    for id in pd.unique(tracks['id']):
        min_val = np.min(tracks.loc[tracks['id']==id, 'frame'])
        tracks.loc[tracks['id']==id, 'frame'] -= min_val # Set starting frame to 0 in all id
    tracks = tracks.set_index(['id', 'frame'])    
    name = fname.replace(ext,'')
    return tracks, name

def filter_traj(dirname:str, filter_values:dict):
    '''
    This function applys selected filters to the original tracks.

    Args:
        dirname (str): path to the original tracks
        filter_values (dict): dict containing the filter values. Possible filter parameters are: ['track_duration', 'total_distance', 'mean_velocity', 'minmax_velocity']. Filter values are composed of a min and a max value.

    Returns:
        tracks: pandas dataframe containing the position information from each track
        name: name of the track
    '''
    df, name = get_traj(dirname)
    valid_ids = []
    for id in np.unique(df.index.get_level_values(0)):
        ctrack = df.loc[id,:]
        total_distance = np.sum(np.linalg.norm(np.diff(ctrack[['x','y','z']], axis=0), axis=-1))
        mean_speed = np.mean(np.linalg.norm(np.diff(ctrack[['x','y','z']], axis=0), axis=-1)/(np.diff(ctrack[['time']], axis=0)))
        track_duration = np.max(ctrack[['time']])-np.min(ctrack[['time']])
        # Compare given filter values to the current track and store only the valid ids
        comparison = [filter_values['track_duration'][0]<=track_duration<=filter_values['track_duration'][1], filter_values['total_distance'][0]<=total_distance<=filter_values['total_distance'][1], filter_values['mean_velocity'][0]<=mean_speed<=filter_values['mean_velocity'][1]]
        if all(comparison):
            valid_ids.append(id)
        else:
            continue
    filtered_tracks = df.loc[valid_ids]
    return filtered_tracks, name

def crop_traj(dirname:str, filter_tracks:bool=False, filter_values:dict={}):
    '''
    This function reads the tracks in the given excel files and crops them in order to all have the same length.

    Args:
        dirname (str): path to the folder containing the excel files
        filter_tracks (bool): whether to perform track filtering or not. Defaults to False
        filter_values (dict): dict containing the filter values in case of performing the filter. Defaults to None

    Returns:
        tracks: pandas dataframe containing the position information from each track
        name: name of the track
    '''
    if filter_tracks==True:
        tracks, name = filter_traj(dirname, filter_values=filter_values)
    else:
        tracks, name = get_traj(dirname)

    min_len = np.inf
    # Find the length of the shortest track
    for id in np.unique(tracks.index.get_level_values(0)):
        t_len = tracks.loc[id].index[-1]
        if t_len<min_len:
            min_len = t_len
    # Reindex the dataframe to crop all tracks to the same length and fill empty spaces
    iterables = pd.MultiIndex.from_product([np.unique(tracks.index.get_level_values(0)), np.arange(min_len+1)])
    tracks = tracks.reindex(index=iterables)
    return tracks, name

def velocity(tracks:pd.DataFrame, names:str, timelapse_units:str, savedir:str="", save_results:bool=True):
    '''
    This function is used to calculate the velocity of the cells. It performs the mean between the velocities of all tracks and generates a file containing the velocity for each time frame.

    Args:
        tracks (list[np.ndarray]): list of track arrays. Track arrays must be in order [id,t,x,y,(z)]
        names (list[str]): list of names of the track conditions
        timelapse_units (str): time units of the tracks (s, min, h)
        savedir (str): path where resulting excels will be saved. Defaults to None
        save_results (bool): save excel with the results of the function. Defaults to True

    Returns:
        list: list of mean speeds of the tracks
    '''    
    v_file = []
    v_params_file = []
    ids = np.unique(tracks.index.get_level_values(0))
    for id in ids:
        track = tracks.loc[id]
        dtrack = np.diff(track, axis=0)
        dt = np.unique(dtrack[:,0])[0]
        vtrack = dtrack[:,1:]/dt
        vr = np.linalg.norm(vtrack, axis=-1)
        v = np.array([vr, vtrack[:,0], vtrack[:,1], vtrack[:,2]]).T
        v_params = np.array([id, 
                             *np.nanmean(v, axis=0), 
                             *np.nanstd(v, axis=0), 
                             *np.nanmedian(v, axis=0),
                             *np.nanmax(np.abs(v), axis=0), 
                             *np.nanmin(np.abs(v), axis=0)])
        final_v = []
        f = 1
        for frame in v:
            final_v.append(np.asarray([id, f*dt, *frame]))
            f+=1
        v_file.append(final_v)
        v_params_file.append(v_params)

    v_file = np.asarray(v_file)
    v_file = v_file.reshape((v_file.shape[0]*v_file.shape[1], v_file.shape[2]))
    v_params_file = np.asarray(v_params_file)
    if save_results:
        if not os.path.isdir(savedir):
            raise ValueError ("File has not been saves: Save directory is not valid")
        savename = f'{os.path.join(savedir, names)}_velocity.xlsx'
        df1 = pd.DataFrame(v_file, columns=['id', f'time {timelapse_units}', 'r_speed', 'x_speed', 'y_speed', 'z_speed'])
        df2 = pd.DataFrame(v_params_file, columns=['id', 'r_mean', 'x_mean', 'y_mean', 'z_mean',
                                                'r_std', 'x_std', 'y_std', 'z_std',
                                                'r_median', 'x_median', 'y_median', 'z_median',
                                                'r_absmax', 'x_absmax', 'y_absmax', 'z_absmax',
                                                'r_absmin', 'x_absmin', 'y_absmin', 'z_absmin'])
        with pd.ExcelWriter(savename, mode='w', engine='openpyxl') as writer:
            df1.to_excel(writer, sheet_name='track_speed', index=False)
            df2.to_excel(writer, sheet_name='speed_params', index=False)
        
    return v_file, v_params_file

def ezmsd_old(xyz:np.ndarray) -> np.ndarray:
    '''
    This function calculates the msd for the given trajectory. It can accept 1D trajectories or multiple dimension ones.

    Args:
        xyz (array): trajectories used to calculate msd

    Returns:
        msdr (array): calculated msd for given trajectory
    '''
    s = np.shape(xyz)
    fn = s[0]
    msdr = np.zeros(fn-1)
    if len(s)!=1:
        for tlag in range(1,fn):
            dxyz = xyz[tlag:,:] - xyz[:-tlag,:]
            msdr[tlag-1] = np.mean(np.sum(dxyz**2, axis=1))
    else:
        for tlag in range(1,fn):
            dxyz = xyz[tlag:] - xyz[:-tlag]
            msdr[tlag-1] = np.mean(dxyz**2)
    return msdr

def ezmsd(xyz:pd.DataFrame) -> np.ndarray:
    '''
    This function calculates the msd for the given trajectory. It can accept 1D trajectories or multiple dimension ones.

    Args:
        txyz (array): trajectories used to calculate msd, they are ordered as [t,x,y,z]

    Returns:
        msdr (array): calculated msd for given trajectory
    '''
    track_msd = []
    for tlag in range(1, len(xyz)):
        msd = np.mean(np.nansum((np.asarray(xyz[tlag:]) - np.asarray(xyz[:-tlag]))**2, axis=-1))
        track_msd.append(msd)
    return np.asarray(track_msd)

def get_msd(tracks:pd.DataFrame, names:str, timelapse_units:str, savedir:str="", save_results:bool=True):
    '''
    This function determines the msd for each file in the directory, with the corresponding error.
    Excel files need to have the following column order: [track_id, t, x, y, (z)].

    Args:
        tracks (pd.DataFrame): DataFrame containing all tracks in the file. Tracks must be in order [id,t,x,y,(z)]
        names (str): name of the track conditions
        timelapse_units (str): time units of the tracks (s, min, h)
        savedir (str): path where resulting excels will be saved. Defaults to None
        save_results (bool): save excel with the results of the function. Defaults to True

    Returns:
        final_msds: dataframe of msds for all tracks
        mean_msds: dataframe of the mean msd
        diff_coef: dataframe of diffusion coefficient for all tracks
    ''' 
    ids = np.unique(tracks.index.get_level_values(0))
    frames = tracks.loc[ids[0]].index
    final_msds = pd.DataFrame(data=np.arange(1,len(frames)), index=np.arange(1,len(frames)), columns=['frame']) 
    diff_coef = pd.DataFrame(data=['D'], index=[ids[0]], columns=['diff_coef'])
    # Calculate msd for each track in the file
    for id in ids:
        track = tracks.loc[id]
        dt = np.unique(np.diff(track, axis=0)[:,0][0])
        xyz = track.iloc[:,slice(1,None,1)]
        msd0 = ezmsd(xyz)
        final_msds[f'msd_{id}'] = msd0
        regr = linregress(np.log10(np.array(frames[1:])), np.log10(msd0))
        D = abs(regr.slope/(2*len(xyz.columns)))
        diff_coef[f'D_{id}'] = D
    mean_msd = np.mean(final_msds, axis=1)
    std_msd = np.std(final_msds, axis=1)
    mean_msds = pd.DataFrame({f'dt ({timelapse_units})': np.arange(1, len(frames))*dt, 'msd': mean_msd, 'std_dev': std_msd})
    #save the file
    if save_results:
        if not os.path.isdir(savedir):
            raise ValueError ("Save directory is not valid")
        savename = f'{os.path.join(savedir,names)}_msd.xlsx'
        with pd.ExcelWriter(savename, mode='w', engine='openpyxl') as writer:
            mean_msds.to_excel(writer, sheet_name='mean_msd', index=False)
            final_msds.to_excel(writer, sheet_name='track_msd', index=False)
            diff_coef.to_excel(writer, sheet_name='diff_coef', index=False)
    return final_msds, mean_msds, diff_coef

def directionality_tortuosity(tracks:pd.DataFrame, names:str, timelapse_units:str, savedir:str="", save_results:bool=True):
    '''
    This function determines the directionality and the tortuosity for each track in the given files.
    Excel files need to have the following column order: [track_id, t, x, y, (z)].

    Args:
        tracks (pd.DataFrame): dataframe of tracks. Tracks must be in order [id,t,x,y,(z)]
        names (str): name of the track conditions
        timelapse_units (str): time units of the tracks (s, min, h)
        savedir (str): path where resulting excels will be saved. Defaults to None
        save_results (bool): save excel with the results of the function. Defaults to True

    Returns:
        df_dir_tort: dataframe with calculated directionality and tortuosity
    '''
    dir_tort = []
    ids = np.unique(tracks.index.get_level_values(0))
    # calculate directionality and tortuosity for each track
    for id in ids:
        xyz = tracks.loc[id].iloc[:,slice(1,None,1)]
        d_euclidean = np.sqrt(np.nansum((xyz.iloc[-1]-xyz.iloc[0])**2, axis=0))
        d_total = np.nansum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))
        dir_tort.append([id, d_euclidean/d_total, d_total/d_euclidean])
    df_dir_tort = pd.DataFrame(np.asarray(dir_tort), columns=['id', 'directionality', 'tortuosity'])

    # save results in excel file
    if save_results:
        if not os.path.isdir(savedir):
            raise ValueError ("Save directory is not valid")
        savename = f'{os.path.join(savedir, names)}_directionality_tortuosity.xlsx'
        with pd.ExcelWriter(savename, mode='w', engine='openpyxl') as writer:
            df_dir_tort.to_excel(writer, sheet_name='direct_tortuos', index=False)

    return df_dir_tort

def spatial_coverage(tracks:pd.DataFrame, names:str, timelapse_units:str, savedir:str="", save_results:bool=True):
    '''
    This function determines the spatial coverage for each track in the given files.
    Excel files need to have the following column order: [track_id, t, x, y, (z)].

    Args:
        tracks (pd.DataFrame): dataframe of tracks. Tracks must be in order [id,t,x,y,(z)]
        names (str): name of the track conditions
        timelapse_units (str): time units of the tracks (s, min, h)
        savedir (str): path where resulting excels will be saved. Defaults to None
        save_results (bool): save excel with the results of the function. Defaults to True

    Returns:
        df_sp_cov: dataframe of calculated spatial coverage values 
    '''
    sp_cov = []
    ids = np.unique(tracks.index.get_level_values(0))
    # calculate spatial coverage of the tracks
    for id in ids:
        xyz = tracks.loc[id].iloc[:,slice(1,None,1)].dropna()
        spatial_coverage = ConvexHull(xyz, qhull_options='QJ').volume
        sp_cov.append([id, spatial_coverage])
    df_sp_cov = pd.DataFrame(np.asarray(sp_cov), columns=['id', 'spatial_coverage'])

    # save results in excel file
    if save_results:
        if not os.path.isdir(savedir):
            raise ValueError ("Save directory is not valid")
        savename = f'{os.path.join(savedir, names)}_spatial_coverage.xlsx'
        with pd.ExcelWriter(savename, mode='w', engine='openpyxl') as writer:
            df_sp_cov.to_excel(writer, sheet_name='sp_cov', index=False)

    return df_sp_cov

def turning_angle(tracks:pd.DataFrame, names:str, timelapse_units:str, savedir:str="", save_results:bool=True):
    '''
    This function calculates the total turning angle of each track in the given files.
    Excel files need to have the following column order: [track_id, t, x, y, (z)].

    Args:
        tracks (pd.DataFrame): dataframes of tracks. Tracks must be in order [id,t,x,y,(z)]
        names (str): name of the track conditions
        timelapse_units (str): time units of the tracks (s, min, h)
        savedir (str): path where resulting excels will be saved. Defaults to None
        save_results (bool): save excel with the results of the function. Defaults to True

    Returns:
        df_turning_angle: dataframe containing the total turning angle and persistance for each track in file
    '''
    turning_angle = []
    ids = np.unique(tracks.index.get_level_values(0))
    # calculate total turning angle and persistence of the tracks
    for id in ids:
        xyz = tracks.loc[id].iloc[:,slice(1,None,1)].dropna()
        dxyz = np.diff(xyz, axis=0)
        total_turning_angle = 0
        persistence = []
        turn_angle = []
        for i in range(1, len(dxyz)):
            dir1 = dxyz[i-1,:]
            dir2 = dxyz[i,:]
            if np.linalg.norm(dir1)==0 or np.linalg.norm(dir2)==0:
                # ignore frames where cell has not moved
                continue
            cos_angle = np.dot(dir1, dir2)/(np.linalg.norm(dir1)*np.linalg.norm(dir2))
            cos_angle = np.clip(cos_angle, -1, 1)
            persistence.append(cos_angle)
            turn_angle.append(np.degrees(np.arccos(cos_angle))) ####################################################
            total_turning_angle += np.degrees(np.arccos(cos_angle))
        turning_angle.append([id, total_turning_angle, abs(np.mean(persistence))])
    df_turning_angle = pd.DataFrame(turning_angle, columns=['id', 'total_turning_angle', 'persistence'])

    # save results in excel file
    if save_results:
        if not os .path.isdir(savedir):
            raise ValueError ("Save directory is not valid")
        savename = f'{os.path.join(savedir, names)}_turning_angle.xlsx'
        with pd.ExcelWriter(savename, mode='w', engine='openpyxl') as writer:
            df_turning_angle.to_excel(writer, sheet_name='tt_angle', index=False)

    return df_turning_angle

def get_acf(tracks:pd.DataFrame, names:str, timelapse_units:str, savedir:str="", save_results:bool=True):
    '''
    This function determines the acf for each file in the directory, with the corresponding error.
    Excel files need to have the following column order: [track_id, t, x, y, (z)].

    Args:
        tracks (pd.DataFrame): dataframe of tracks. Tracks must be in order [id,t,x,y,(z)]
        names (str): name of the track condition
        timelapse_units (str): time units of the tracks (s, min, h)
        savedir (str): path where resulting excels will be saved. Defaults to None
        save_results (bool): save excel with the results of the function. Defaults to True

    Returns:
        final_acfs: dataframe containing the calculated acf for each track
        mean_acfs: dataframe with the mean acf values
    '''
    ids = np.unique(tracks.index.get_level_values(0))
    frames = tracks.loc[ids[0]].index
    final_acfs = pd.DataFrame(data=np.arange(1,len(frames)-1), index=np.arange(1,len(frames)-1), columns=['frame']) 
    # Calculate acf for each track in the file
    for id in ids:
        track = tracks.loc[id]
        dt = np.unique(np.diff(track, axis=0)[:,0][0])
        xyz = track.iloc[:,slice(1,None,1)]
        dxyz = np.diff(xyz, axis=0)
        track_acf = []
        for tlag in range(1, len(xyz)-1):
            acf = np.mean(np.nansum(dxyz[tlag:]*dxyz[:-tlag], axis=1))
            track_acf.append(acf)
        acf0 = np.asarray(track_acf)
        final_acfs[f'acfs_{id}'] = np.abs(acf0)
    mean_acf = np.mean(final_acfs, axis=1)
    std_acf = np.std(final_acfs, axis=1)
    mean_acfs = pd.DataFrame({f'dt ({timelapse_units})': np.arange(1, len(frames)-1)*dt, 'acf': mean_acf, 'std_dev': std_acf})
    #save the file
    if save_results:
        if not os.path.isdir(savedir):
            raise ValueError ("Save directory is not valid")
        savename = f'{os.path.join(savedir,names)}_acf.xlsx'
        with pd.ExcelWriter(savename, mode='w', engine='openpyxl') as writer:
            mean_acfs.to_excel(writer, sheet_name='mean_acf', index=False)
            final_acfs.to_excel(writer, sheet_name='track_acf', index=False)
    return final_acfs, mean_acfs

def tPRW3D(x:np.ndarray, P:float, S:float, SE:float):
    return 3*(S**2)*P*(x-P*(1-np.exp(-x/P)))+6*SE**2

def tPRW2D(x:np.ndarray, P:float, S:float, SE:float):
    return 2*(S**2)*P*(x-P*(1-np.exp(-x/P)))+4*SE**2

def tPRW1D(x:np.ndarray, P:float, S:float, SE:float):
    return (S**2)*P*(x-P*(1-np.exp(-x/P)))+2*SE**2

###########################################

def fit_APRW(tracks:pd.DataFrame, names:str, savedir:str):
    '''
    This functions rotates the trajectories to the primary axis of migration (p) and calculates the msd for the 
    primary and non primary axes.

    Args:
        dirname (string): path to the folder containing the original trajectories
        dt (int): trajectory time step size
        dim (int): dimensionality of the tracks (2 for 2D or 3 for 3D)
        tlag (int): defaults to 1. Time lag to determine the rotational matrix
    '''
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    ids = np.unique(tracks.index.get_level_values(0))
    for id in ids:
        track = tracks.loc[id]
        xyz = track.iloc[:,slice(1,None,1)]
        xyzr = xyz - np.mean(xyz, axis=0)
        dxyz = np.diff(xyz, axis=0)
        U,S,V = np.linalg.svd(dxyz, full_matrices=True)
        xyzrot = xyz@np.transpose(V) ######

    for file in tracks:
        params_list = []
        for track in file:
            t = track[:,1]
            xyz = track[:,2:]
            dt = np.min(np.diff(track[:,1]))
            used_len = np.uint16(np.round(len(track))/4)
            dtxyz = np.diff(xyz, axis=0)
            xyzr = xyz-np.mean(xyz, axis=0) #major axis of trajectories
            U,S,V = np.linalg.svd(dtxyz, full_matrices=True)
            xyzrot = xyz@np.transpose(V)
            txyz_list=[]
            for time,row in zip(t,xyzrot):
                txyz_list.append([time/dt, *row])
            txyz=np.array(txyz_list)
            
            dim = len(xyz[0,:])
            time = np.linspace(1, used_len, used_len)
            t = time*dt
            wif = (2*time**2+1)/(3*time*(used_len-time+1))
            msdp0 = ezmsd(txyz[:,:2])[:,1]
            msdp = msdp0[:used_len]
            wtp = (msdp*wif**2)
            msdnp0 = ezmsd(txyz[:,:3:2])[:,1]
            if dim==3:
                msdnp0 += ezmsd(txyz[:,::3])[:,1]
            msdnp = msdnp0[:used_len]
            wtnp = (msdnp*wif**2)
                        
            if dim==3:
                poptp, pcovp = curve_fit(tPRW1D, t, msdp, p0=(3*dt,1,1), bounds=([0, 0, 0], [1000,10,100]), method='trf', maxfev=10000)#, sigma=wtp, absolute_sigma=True)
                poptnp, pcovnp = curve_fit(tPRW2D, t, msdnp, p0=(3*dt,1,1), bounds=([0, 0, 0], [1000,10,100]), method='trf', maxfev=10000)#, sigma=wtnp, absolute_sigma=True)
            if dim==2:
                poptp, pcovp = curve_fit(tPRW1D, t, msdp, p0=(3*dt,1,1), bounds=([0, 0, 0], [1000,10,100]), method='trf', maxfev=10000, sigma=wtp, absolute_sigma=True)
                poptnp, pcovnp = curve_fit(tPRW1D, t, msdnp, p0=(3*dt,1,1), bounds=([0, 0, 0], [1000,10,100]), method='trf', maxfev=10000, sigma=wtnp, absolute_sigma=True)

            rmsep=np.sqrt(np.mean((msdp-tPRW1D(t,*poptp))**2))
            rmsenp=np.sqrt(np.mean((msdnp-tPRW2D(t,*poptnp))**2))

            Pp,Sp,SEp = poptp
            Pnp,Snp,SEnp = poptnp
            params_list.append([Pp,Sp,SEp,rmsep])
            params_list.append([Pnp,Snp,SEnp,rmsenp])

        params=np.array(params_list)

        savename = f'{os.path.join(savedir,names[name])}_APRW_params.xlsx'
        pd.DataFrame({'P':params[:,0], 'S':params[:,1], 'SE':params[:,2], 'rmse':params[:,3]}).to_excel(savename, sheet_name='params', index=False)
        name = name+1

    return params

def sim_APRW(dirname:str, tracks:list[np.ndarray], Nmax:int=50, repeats:int=30, subsamples:int=100):
    '''
    This function simulates different tracks with the parameters determined in the fitting. The function saves an .xlsx file with all simulated tracks

    Args:
        dirname (string): path to the directory containing the files with the parameters
        tracks (list): list of arrays containing the values of the tracks. Used to extract som necessary data. Order of the tracks must be [id, t, x, y, z]
        dim (int): dimensionality of the tracks (2 for 2D and 3 for 3D)
        Nmax (int): maximum number of time lags that will be calculated when simulating the tracks
        repeats (int): number of simulated tracks per set of parameters
    '''
    subsample = subsamples
    for file in tracks:
        for track in file:
            tlag = np.min(np.diff(track[:,1]))
            dim = len(track[0,2:])
    
    dt = tlag/subsample 
    savedir = fr'{dirname}\Simulations'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    params = []
    savenames = []
    for fname in os.listdir(dirname):
        ext = os.path.splitext(fname)[-1].lower()
        if ext=='.xlsx' or ext=='.xls':
            df = pd.read_excel(os.path.abspath(os.path.join(dirname,fname))).to_numpy()
            params.append(df)
            newname = fname.replace('_params.xlsx', '_sim_tracks.xlsx')
            savename = os.path.join(savedir, newname)
            savenames.append(savename)

    ss = 10 ** np.ceil(np.log10(repeats))
    n = 0

    for file in params:
        xys00 = []
        simxy = []
        count = 1
        for i in range(0, len(file), 2):
            P = file[i:i+2,0]
            S = file[i:i+2,1]
            mSE = file[i:i+2,2]
            beta = 1/P
            alpha = (S**2)*beta
            map = np.array([max(0, 1-dt/P[0]), max(0, 1-dt/P[1])])
            mFR = np.sqrt(alpha*dt)*dt

            if dim==3:
                ap = np.append(map, map[-1])
                FR = np.append(mFR, mFR[-1])
                SE = np.append(mSE, mSE[-1])
            elif dim==2:
                ap = map
                FR = mFR
                SE = mSE

            fnum = Nmax*subsample
            for rep in range(0, repeats):
                dr = np.zeros((fnum, dim))
                xyz = np.zeros((fnum, dim))

                for i in range(0, fnum-1):
                    for d in range(0, dim):
                        dr[i+1,d] = FR[d]*np.random.randn() + ap[d]*dr[i,d]

                for k in range(0, fnum-1):
                    for kd in range(0, dim):
                        xyz[k+1, kd] = xyz[k,kd] + dr[k,kd]

                exyz = xyz[::subsample, :]
                oxyz = exyz - np.ones((exyz.shape[0], 1)) * exyz[0, :] 
           
                #including position noise
                for kd in range(0, dim):
                    oxyz[:,kd] = oxyz[:,kd] + np.random.randn(len(oxyz[:,kd]))*SE[kd]

                #rotate the trajectories randomly
                if dim==3:
                    thetax = np.random.rand()*2*np.pi
                    Rx = np.array([[1, 0, 0],
                                   [np.cos(thetax), -np.sin(thetax), 0],
                                   [0, np.sin(thetax), np.cos(thetax)]])
      
                    thetay = np.random.rand()*2*np.pi
                    Ry = np.array([[np.cos(thetay), 0, np.sin(thetay)],
                                   [0, 1, 0],
                                   [-np.sin(thetay), 0, np.cos(thetay)]])
       
                    thetaz = np.random.rand()*2*np.pi
                    Rz = np.array([[np.cos(thetaz), -np.sin(thetaz), 0],
                                   [np.sin(thetaz), np.cos(thetaz), 0],
                                   [0, 0, 1]])

                    Rm = np.dot(Rx, np.dot(Ry, Rz))
                    rxyz = np.dot(oxyz, Rm) #rotate matrix

                elif dim==2:
                    theta=np.random.rand()*2*np.pi
                    Rm = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
                    rxyz=np.dot(oxyz,Rm)

                xyss0 = np.column_stack((np.ones(rxyz.shape[0]) * count * ss + rep, np.arange(0, len(rxyz))*tlag, rxyz))
                xys00.append(xyss0)  # for output to excel
                simxy.append(xyss0)  # for output variable
            count = count+1

        pd.DataFrame(np.vstack(xys00)).to_excel(savenames[n], sheet_name='sim_APRW', index=False)
        n = n+1  
    return

def PDF_dR(tracks:list[np.ndarray], names:list[str], timelapse_units:str, tlag:int, savedir:str="", save_results:bool=True):
    '''
    This function calculates the probability density function (PDF) of the displacement. 
    For a given time lag, it calculates all displacements and distributes them along the given number of bins, up to the maximum displacement assigned.

    Args:
        tracks (list[np.ndarray]): list of track arrays. Track arrays must be in order [id,t,x,y,(z)]
        names (list[str]): list of names of the track conditions
        timelapse_units (str): time units of the tracks (s, min, h)
        tlag (int): value of the timelag that will be used in the PDF calculation
        savedir (str): path where resulting excels will be saved. Defaults to None
        save_results (bool): save excel with the results of the function. Defaults to True

    returns:
        list: list of lists for each file. Each file list corresponds to two arrays, the first one for the histogram data and the second one for the bins.
    '''
    if not os.path.exists(savedir):
        os.makedirs(os.path.abspath(savedir))

    all_d_euc = []
    tracks_d_euc = []
    for file in tracks:
        file_d_euc = []
        #calculate PDF_dR for each track in the file
        for track in file:
            d_euc = np.linalg.norm(track[tlag:,2:]-track[:-tlag,2:], axis=-1)
            file_d_euc.append(np.mean(d_euc))
            all_d_euc.append(np.mean(d_euc))
        tracks_d_euc.append(np.array(all_d_euc))

    _, all_bins = np.histogram(all_d_euc, bins='auto')
    name = 0
    for file in tracks_d_euc:
        hist, bins = np.histogram(file, all_bins)
        #save PDF_dR into excel files, if multiple files have been readed, multiple excel files will be created
        if save_results:
            if not os.path.isdir(savedir):
                raise ValueError ("Save directory is not valid")
            savename = f'{os.path.join(savedir,names[name])}_PDF_dR.xlsx'
            df1 = DataFrame({'bins': bins[:-1], 'hist': hist})
            with pd.ExcelWriter(savename, mode='w', engine='openpyxl') as writer:
                df1.to_excel(writer, sheet_name='PDF_dR', index=False)
            name = name+1
    return

################################################

def PDF_dtheta(tracks:list[np.ndarray], names:list[str], timelapse_units:str, tlag:int, savedir:str="", save_results:bool=True):
    '''
    This function calculates the probability density funtion (PDF) of the angle. 
    Given a time lag, the function calculates the angle turned (between 0 and 180º) between the current position and the position one time lag after.

    Args:
        tracks (list[np.ndarray]): list of track arrays. Track arrays must be in order [id,t,x,y,(z)]
        names (list[str]): list of names of the track conditions
        timelapse_units (str): time units of the tracks (s, min, h)
        tlag (int): value of the timelag that will be used in the PDF calculation
        savedir (str): path where resulting excels will be saved. Defaults to None
        save_results (bool): save excel with the results of the function. Defaults to True
    '''
    if not os.path.exists(savedir):
        os.makedirs(os.path.abspath(savedir))


    all_files_PDF_dtheta = []
    bin_distr = np.linspace(0, 180, binn)

    for file in all_files:
        angle = []
        #calculate PDF_dtheta for each track in the file
        for track in file:
            t = track[:,1]
            xyz = track[:,2:]
            for j in range(0, len(t)-np.uint16(tlag/dt)-1):
                dxyzt = xyz[j+1]-xyz[j]
                dxyztau = xyz[j+np.uint16(tlag/dt)+1]-xyz[j+np.uint16(tlag/dt)]
                theta = np.arccos(np.dot(dxyzt, dxyztau)/(np.sqrt(np.sum(dxyzt**2))*np.sqrt(np.sum(dxyztau**2))))
                angle.append(theta/np.pi*180)
        hist, bins = np.histogram(np.array(angle), bin_distr, density=True)
        all_files_PDF_dtheta.append([hist, bins[0:-1]])
    
    #save PDF_dtheta into excel files, if multiple files have been readed, multiple excel files will be created
    name = 0
    if save_xlsx==True:
        for file in all_files_PDF_dtheta:
            savename = f'{os.path.join(savedir,name_list[name])}_PDF_dtheta_{tlag}{tunit}.xlsx'
            df = DataFrame({'bins': file[1], 'PDF_dtheta': file[0]})
            df.to_excel(savename, sheet_name='PDF_dtheta', index=False)
            name = name+1

    return

def polarity_dR(dirname, dt, tlag, binn=20, savedir=None):
    '''
    This function is used to determine the polarity of the velocity. It detrmines the primary axis and rotates the 
    '''
    all_files, name_list = get_traj(dirname)

    if os.path.isfile(dirname):
        raise ValueError('Error: please indicate the path to the folder where results will be saved.')

    if savedir==None:
        savedir = os.path.abspath(fr'{dirname}\Results')
        if not os.path.exists(savedir):
            os.makedirs(savedir)
    
    all_files_PDF_dtheta = []
    bin_distr = np.linspace(0, 360, binn)

    plt.figure()
    for file in all_files:
        angle = []
        #calculate polarity_dR for each track in the file
        for track in file:
            t = track[:,1]
            xyz = track[:,2:]
            dxyz = np.diff(xyz, axis=0)
            U, S, V = np.linalg.svd(dxyz, full_matrices=False) #the numpy function returns Vt, but we want the orthogonal vectors in the columns
            dxyzr = np.dot(dxyz, np.transpose(V))

            theta = np.arctan(dxyzr[:,1]/dxyzr[:,0])
            angle = theta*180/np.pi
            for i in range(0,len(angle)):
                if angle[i]<0:
                    angle[i]=angle[i]+360
            
            dr = np.sqrt(dxyzr[0]**2+dxyzr[1]**2)


    return

def fit_PRW(dirname, dt, dim):
    '''
    This function calculates the msd for every track and fits a persisten random walk model in each one, obtaining the parameters P, S, and SE for every track.

    Args:
        filename (string): path to the msd file.
        dt (int): trajectory time step size
        dim (int): dimensionality of the tracks (2 for 2D or 3 for 3D)
    '''
    track_list, name_list = get_traj(dirname)

    savedir = os.path.abspath(fr'{dirname}\PRW')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    name = 0
    psse = []
    for file in track_list:
        file_psse = []
        for track in file:
            used_len = np.uint16(np.round(len(track)/2)) #use only the first 1/3 of tracks as they contain les error
            msd = ezmsd(track[:used_len+1,:])
            t = np.linspace(1, used_len, used_len)

            #fit the values in PRW_msd equation
            if dim == 2:
                popt, pcov = curve_fit(tPRW2D, t*dt, msd, p0=(10, 0.01, 1), bounds=(0, [1000, 20, 100]), method='trf', maxfev=1000)
            elif dim == 3:
                popt, pcov = curve_fit(tPRW3D, t*dt, msd, p0=(10, 0.01, 1), bounds=(0, [1000, 20, 100]), method='trf', maxfev=1000)
            file_psse.append(np.array(popt))
        params = np.array(file_psse)
        
        psse.append(params)
        savename = f'{os.path.join(savedir, name_list[name])}_PRW_params.xlsx'
        savedf = DataFrame({'P':params[:,0], 'S':params[:,1], 'SE':params[:,2]})
        savedf.to_excel(savename, sheet_name='params', index=False)
        name = name+1
        
    return params

def sim_PRW(dirname, tlag, dim, Nmax=50, repeats=20, subsamples=100):
    '''
    This function simulates different tracks with the parameters determined in the fitting. The function saves an .xlsx file with all simulated tracks

    Args:
        dirname (string): path to the directory containing the files with the parameters
        tlag (int): time step of the tracks (needs to be the same one used in the fitting)
        dim (int): dimensionality of the tracks (2 for 2D and 3 for 3D)
        Nmax (int): maximum number of time lags that will be calculated when simulating the tracks
        repeats (int): number of simulated tracks per set of parameters
    '''
    subsample = subsamples
    dt = tlag/subsample 
    savedir = fr'{dirname}\Simulations'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    params = []
    savenames = []
    for fname in os.listdir(dirname):
        ext = os.path.splitext(fname)[-1].lower()
        if ext=='.xlsx' or ext=='.xls':
            df = pd.read_excel(os.path.abspath(os.path.join(dirname,fname))).to_numpy()
            params.append(df)
            newname = fname.replace('_params.xlsx', '_sim_tracks.xlsx')
            savename = os.path.join(savedir, newname)
            savenames.append(savename)


    ss = 10 ** np.ceil(np.log10(repeats))
    n = 0
    
    for file in params:
        xys00 = []
        simxy = []
        count = 1
        for track in file:
            P,S,SE = track
            beta = 1/P
            alpha = S**2*beta
            ap = max(0, 1-beta*dt)
            FR = np.sqrt(alpha*dt)*dt
    
            fnum = Nmax*subsample
            for rep in range(0, repeats):
                dr = np.zeros((fnum, dim))
                xyz = np.zeros((fnum, dim))

                for i in range(0, fnum-1):
                    for d in range(0, dim):
                        dr[i+1,d] = FR*np.random.randn() + ap*dr[i,d]

                for k in range(0, fnum-1):
                    for kd in range(0, dim):
                        xyz[k+1, kd] = xyz[k,kd] + dr[k,kd]

                exyz = xyz[::subsample, :]
                oxyz = exyz - np.ones((exyz.shape[0], 1)) * exyz[0, :] #
           
                #including position noise
                for k in range(0, Nmax):
                    for kd in range(0, dim):
                        oxyz[k,kd] = oxyz[k,kd] + np.random.randn()*SE

                #rotate the trajectories randomly
                if dim==3:
                    thetax = np.random.rand()*2*np.pi
                    Rx = np.array([[1, 0, 0],
                                   [np.cos(thetax), -np.sin(thetax), 0],
                                   [0, np.sin(thetax), np.cos(thetax)]])
      
                    thetay = np.random.rand()*2*np.pi
                    Ry = np.array([[np.cos(thetay), 0, np.sin(thetay)],
                                   [0, 1, 0],
                                   [-np.sin(thetay), 0, np.cos(thetay)]])
       
                    thetaz = np.random.rand()*2*np.pi
                    Rz = np.array([[np.cos(thetaz), -np.sin(thetaz), 0],
                                   [np.sin(thetaz), np.cos(thetaz), 0],
                                   [0, 0, 1]])

                    Rm = np.dot(Rx, np.dot(Ry, Rz))
                    rxyz = np.dot(oxyz, Rm) #rotate matrix

                elif dim==2:
                    theta=np.random.rand()*2*np.pi
                    Rm = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
                    rxyz=np.dot(oxyz,Rm)

                xyss0 = np.column_stack((np.ones(rxyz.shape[0]) * count * ss + rep, np.arange(0, len(rxyz))*tlag, rxyz))
                xys00.append(xyss0)  # for output to excel
                simxy.append(xyss0)  # for output variable
            count = count+1

        pd.DataFrame(np.vstack(xys00)).to_excel(savenames[n], sheet_name='sim_PRW', index=False)
        n = n+1           
    return
