import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import warnings

with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

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

def cohen_d(group_a:pd.DataFrame, group_b:pd.DataFrame):
    '''
    This function takes as arguments two dataframes and calculates the size effect between them.

    Args:
        group_a (pd.DataFrame): one of the groups to calculate cohen's d value
        group_b (pd.DataFrame): one of the groups to calculate cohen's d value
    
    Returns:
        d_value: float value of calculated cohen's d effect size
    '''
    mean_a = np.nanmean(group_a)
    mean_b = np.nanmean(group_b)
    std_a = np.std(group_a)
    std_b = np.std(group_b)
    n_a = len(group_a)
    n_b = len(group_b)
    stdp = np.sqrt(((n_a-1)*std_a**2 + (n_b-1)*std_b**2)/(n_a+n_b-2))
    d_value = (np.abs(mean_a - mean_b))/stdp
    return d_value

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
    
    t_lens = []
    total_frames = []
    for id in np.unique(tracks['id']):
        t_lens.append(len(tracks.loc[tracks['id']==id, 'time']))
    for t_len in t_lens:
        frames = np.arange(t_len)
        for frame in frames:
            total_frames.append(frame)
    frames = np.asarray(total_frames).reshape(len(tracks))
    tracks['frame'] = frames
    tracks = tracks.set_index(['id', 'frame'])    
    name = fname.replace(ext,'')
    return tracks, name

def filter_traj(dirname:str, filter_values:dict):
    '''
    This function applys selected filters to the original tracks.

    Args:
        dirname (str): path to the original tracks
        filter_values (dict): dict containing the filter values. Possible filter parameters are: ['track_duration', 'total_distance', 'mean_velocity']. 
                              Filter values are composed of a min and a max value.

    Returns:
        tracks: pandas dataframe containing the position information from each track
        name: name of the track
    '''
    df, name = get_traj(dirname)
    valid_ids = []
    for id in np.unique(df.index.get_level_values(0)):
        ctrack = df.loc[id,:]
        total_distance = np.nansum(np.abs(np.diff(np.linalg.norm(ctrack[['x','y','z']], axis=1), axis=0)))
        mean_speed = np.nanmean(np.abs(np.diff(np.linalg.norm(ctrack[['x','y','z']], axis=1), axis=0))/(np.diff(ctrack[['time']], axis=0)))
        track_duration = np.max(ctrack[['time']])-np.min(ctrack[['time']])
        # Compare given filter values to the current track and store only the valid ids
        comparison = [filter_values['track_duration'][0]<=track_duration<=filter_values['track_duration'][1], 
                      filter_values['total_distance'][0]<=total_distance<=filter_values['total_distance'][1], 
                      filter_values['mean_velocity'][0]<=mean_speed<=filter_values['mean_velocity'][1]]
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
        t_len = len(tracks.loc[id])
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
    dims = len(xyz.shape)
    for tlag in range(1, len(xyz)-1):
        if dims > 1:
            msd_sum = np.sum((np.asarray(xyz[tlag:]).astype(np.double) - np.asarray(xyz[:-tlag]).astype(np.double))**2, axis=-1)
            msd = np.nanmean(msd_sum)
        else:
            msd = np.nanmean((np.asarray(xyz[tlag:]).astype(np.double) - np.asarray(xyz[:-tlag]).astype(np.double))**2)
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
    final_msds = pd.DataFrame(data=np.arange(1,len(frames)-1), index=np.arange(1,len(frames)-1), columns=['frame']) 
    alpha_coef = pd.DataFrame(data=['alpha'], index=[ids[0]], columns=['alpha_coef'])
    diff_coef = pd.DataFrame(data=['D'], index=[ids[0]], columns=['diff_coef'])
    # Calculate msd for each track in the file
    for id in ids:
        track = tracks.loc[id]
        dt = np.unique(np.diff(track, axis=0)[:,0][0])
        xyz = track.iloc[:,slice(1,None,1)]
        msd0 = ezmsd(xyz)
        final_msds[f'msd_{id}'] = msd0.astype(np.double)
        regr = linregress(np.log10(np.array(frames[1:-1])*dt), np.log10(msd0))
        alpha = regr.slope
        D = regr.intercept/(2*len(track.iloc[0, slice(1, None, 1)]))
        alpha_coef[f'alpha_{id}'] = alpha
        diff_coef[f'D_{id}'] = D
    mean_msd = np.mean(final_msds, axis=1)
    std_msd = np.std(final_msds, axis=1)/np.sqrt(len(final_msds)) #standard error of the mean (std/sqrt(n))
    mean_msds = pd.DataFrame({f'dt ({timelapse_units})': np.arange(1, len(frames)-1)*dt, 'msd': mean_msd, 'std_dev': std_msd})
    #save the file
    if save_results:
        if not os.path.isdir(savedir):
            raise ValueError ("Save directory is not valid")
        savename = f'{os.path.join(savedir,names)}_msd.xlsx'
        with pd.ExcelWriter(savename, mode='w', engine='openpyxl') as writer:
            mean_msds.to_excel(writer, sheet_name='mean_msd', index=False)
            final_msds.to_excel(writer, sheet_name='track_msd', index=False)
            alpha_coef.to_excel(writer, sheet_name='alpha_coef', index=False)
            diff_coef.to_excel(writer, sheet_name='diff_coef', index=False)
    return final_msds, mean_msds, alpha_coef, diff_coef

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
        if d_total==0:
            d_total = 0.00001
        if d_euclidean==0:
            d_euclidean = 0.00001
        dir_tort.append([id, d_euclidean/d_total, d_total/d_euclidean, d_total, d_euclidean])
    df_dir_tort = pd.DataFrame(np.asarray(dir_tort), columns=['id', 'directionality', 'tortuosity', 'total distance', 'net distance'])

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
    final_acfs = pd.DataFrame(data=np.arange(1,len(frames)-2), index=np.arange(1,len(frames)-2), columns=['frame']) 
    # Calculate acf for each track in the file
    for id in ids:
        track = tracks.loc[id]
        dt = np.unique(np.diff(track, axis=0)[:,0][0])
        xyz = track.iloc[:,slice(1,None,1)]
        dxyz = np.diff(xyz, axis=0)
        track_acf = []
        for tlag in range(1, len(xyz)-2):
            acf = np.nanmean(np.sum(dxyz[tlag:]*dxyz[:-tlag], axis=1))
            track_acf.append(acf)
        acf0 = np.asarray(track_acf)
        final_acfs[f'acfs_{id}'] = acf0.astype(np.double)
    mean_acf = np.mean(final_acfs, axis=1)
    std_acf = np.std(final_acfs, axis=1)/np.sqrt(len(final_acfs))
    mean_acfs = pd.DataFrame({f'dt ({timelapse_units})': np.arange(1, len(frames)-2)*dt, 'acf': mean_acf, 'std_dev': std_acf})
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

def fit_APRW(tracks:pd.DataFrame, names:str, savedir:str):
    '''
    This functions rotates the trajectories to the primary axis of migration (p) and calculates the msd for the 
    primary and non primary axes.

    Args:
        tracks (pd.DataFrame): dataframe containing the trajectories of the file
        names (str): name of the condition of the tracks in the file
        savedir (str): folder path where the parameters will be saved
    '''
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    params_list = []
    ids = np.unique(tracks.index.get_level_values(0))
    for id in ids:
        track = tracks.loc[id]
        xyz = track.iloc[:,slice(1,None,1)].dropna()
        xyzr = xyz - np.mean(xyz, axis=0)
        dxyz = np.diff(xyz, axis=0)
        U,S,V = np.linalg.svd(dxyz, full_matrices=True)
        xyzrot = xyz@V.T 

        dt = np.unique(np.diff(track, axis=0)[:,0][0])[0]
        frames = xyz.index
        tlag = frames[1:]
        
        msdp = ezmsd(xyzrot.iloc[:,0])
        poptp, _ = curve_fit(tPRW1D, tlag*dt, msdp, p0=(100,0.1,1), bounds=([0, 0, 0], [1000,100/dt,10]), 
                             method='trf', maxfev=10000)
        msdnp = ezmsd(xyzrot.iloc[:,1])
        rmsep = np.sqrt(np.mean((msdp - tPRW1D(tlag*dt, *poptp))**2))
        if len(xyz.columns) == 3:
            msdnp += ezmsd(xyzrot.iloc[:,2])
            poptnp, _ = curve_fit(tPRW2D, tlag*dt, msdnp, p0=(100,0.1,1), bounds=([0, 0, 0], [1000,100/dt,10]), 
                                 method='trf', maxfev=10000)
            rmsenp = np.sqrt(np.mean((msdnp - tPRW2D(tlag*dt, *poptnp))**2))
        else:
            poptnp, _ = curve_fit(tPRW1D, tlag*dt, msdnp, p0=(100,0.1,1), bounds=([0, 0, 0], [1000,100/dt,10]), 
                                 method='trf', maxfev=10000)
            rmsenp = np.sqrt(np.mean((msdnp - tPRW1D(tlag*dt, *poptnp))**2))

        params_list.append([*poptp, rmsep])
        params_list.append([*poptnp, rmsenp])
    index = pd.MultiIndex.from_product([ids,['p', 'np']])
    df_params = pd.DataFrame(np.array(params_list), index=index, columns=['P','S','SE','rmse'])
    if os.path.isdir(savedir):
        savename = f'{os.path.join(savedir,names)}_APRW_params.xlsx'
        with pd.ExcelWriter(savename, mode='w', engine='openpyxl') as writer:
            df_params.to_excel(writer, sheet_name='APRW_params', index=True)
    
    return df_params

def sim_APRW(params:pd.DataFrame, tracks:pd.DataFrame, names:str, Nmax:int=50, repeats:int=30, subsamples:int=100, savedir:str=None):
    '''
    This function simulates different tracks with the parameters determined in the fitting. The function saves an .xlsx file with all simulated tracks

    Args:
        params (pd.DataFrame): dataframe with the parameters calculated in fit_APRW function
        tracks (pd.DataFrame): dataframe containing the trajectories of the file
        names (str): name of the condition of the tracks
        Nmax (int): maximum number of time lags that will be calculated when simulating the tracks
        repeats (int): number of simulated tracks per set of parameters
        subsamples (int): number of divisions between time lags. Used for calculation of the simulated tracks
        savedir (str): path to the folder where simulated tracks will be saved
    '''
    ids = np.unique(params.index.get_level_values(0))

    #Extract tlag and dimensions from original tracks
    track = tracks.loc[ids[0]]
    dim = len(track.columns)-1
    tlag = np.unique(np.diff(track['time']))[0]
    dt = tlag/subsamples
    ss = 10 ** np.ceil(np.log10(repeats))
    #Simulation of given number of repeats for each track
    xys00 = []
    count = 0
    for id in ids:
        track_params = params.loc[id]
        beta = 1/track_params['P']
        alpha = (track_params['S']**2)*beta
        FR = np.asarray(np.sqrt(alpha*dt)*dt)
        ap = np.array((max(0, 1-dt/track_params['P'].iloc[0]), max(0, 1-dt/track_params['P'].iloc[1])))
        SE = np.asarray(track_params['SE'])
        fnum = Nmax*subsamples
        if dim==3:
            ap = np.append(ap, ap[-1])
            FR = np.append(FR, FR[-1])
            SE = np.append(SE, SE[-1])

        for rep in range(0, repeats):
            dr = np.zeros((fnum, dim))
            xyz = np.zeros((fnum, dim))

            for i in range(0, fnum-1):
                dr[i+1,:] = FR*np.random.randn(3) + ap*dr[i,:]
                xyz[i+1,:] = xyz[i,:] + dr[i,:]
            exyz = xyz[::subsamples, :]
            oxyz = exyz - np.ones((exyz.shape[0], 1))*exyz[0,:]
            oxyz = oxyz + np.random.randn(len(oxyz[:]),dim)*SE #Position noise

            if dim==3:
                theta = np.random.randn(3)*2*np.pi
                Rx = np.array([[1, 0, 0],
                               [np.cos(theta[0]), -np.sin(theta[0]), 0],
                               [0, np.sin(theta[0]), np.cos(theta[0])]])
      
                Ry = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                               [0, 1, 0],
                               [-np.sin(theta[1]), 0, np.cos(theta[1])]])

                Rz = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                               [np.sin(theta[2]), np.cos(theta[2]), 0],
                               [0, 0, 1]])
            
                Rm = Rx@Ry@Rz
                rxyz = oxyz@Rm
        
            elif dim==2:
                theta = np.random.randn()*2*np.pi
                Rm = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
                rxyz = oxyz@Rm
        
            xyss0 = np.column_stack((np.ones(rxyz.shape[0])*count*ss + rep, np.arange(0, len(rxyz))*tlag, rxyz))
            xys00.append(xyss0)
        count += 1
    savename = fr'{savedir}\Simulations\{names}_sim_APRW.xlsx'
    sim_tracks = pd.DataFrame(np.vstack(xys00), columns=['id', 'time', 'x', 'y', 'z'])
    #{'id':np.vstack(xys00)[:,0], 'time':np.vstack(xys00)[:,1], 'x':np.vstack(xys00)[:,2], 'y':np.vstack(xys00)[:,3], 'z':np.vstack(xys00)[:,4]}
    with pd.ExcelWriter(savename, mode='w', engine='openpyxl') as writer:
        sim_tracks.to_excel(writer, sheet_name='sim_APRW', index=False)
    return sim_tracks

def PDF(tracks:pd.DataFrame, names:str, timelapse_units:str, tlag:int, savedir:str="", save_results:bool=True):
    '''
    This function calculates the probability density function (PDF) of the displacement. 
    For a given time lag, it calculates all displacements and distributes them along the given number of bins, up to the maximum displacement assigned.

    Args:
        tracks (pd.DataFrame): dataframes of tracks. Tracks must be in order [id,t,x,y,(z)]
        names (str): name of the track conditions
        timelapse_units (str): time units of the tracks (s, min, h)
        tlag (int): value of the timelag that will be used in the PDF calculation
        savedir (str): path where resulting excels will be saved. Defaults to None
        save_results (bool): save excel with the results of the function. Defaults to True

    returns:
        df_PDF: dataframe containing the information of the bins and the histogram created
    '''
    ids = np.unique(tracks.index.get_level_values(0))
    file_d_euc = [] # one row per track
    file_angle = [] # several rows per track
    for id in ids:
        track = tracks.loc[id][['x','y','z']]
        d_euc = np.linalg.norm(np.asarray(track.loc[tlag:])-np.asarray(track.loc[:len(track)-tlag-1]), axis=1) #Euclidean distance 
        dxyz = np.diff(track.loc[:len(track)-tlag], axis=0)
        dxyztau = np.asarray(track.loc[tlag:])-np.asarray(track.loc[:len(track)-tlag-1])
        norm = np.linalg.norm(dxyz, axis=1)*np.linalg.norm(dxyztau, axis=1)
        norm[norm==0] = np.nan
        theta = np.arccos(np.divide(np.vecdot(dxyz,dxyztau),(norm)))/np.pi*180
        file_d_euc.append(np.nanmean(d_euc))
        file_angle.append(np.nanmean(theta))
    df_PDF = pd.DataFrame({'id': ids, 'net_dist':file_d_euc, 'mean_angle':file_angle})
    if save_results:
        if not os.path.exists(savedir):
            os.makedirs(os.path.abspath(savedir))
        savename = f'{os.path.join(savedir, names)}_PDF.xlsx'
        with pd.ExcelWriter(savename, mode='w', engine='openpyxl') as writer:
            df_PDF.to_excel(writer, sheet_name='PDF', index=False)
    return df_PDF

################################################

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
        savedf = pd.DataFrame({'P':params[:,0], 'S':params[:,1], 'SE':params[:,2]})
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
