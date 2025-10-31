import numpy as np
import pandas as pd
from pandas import DataFrame
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def xml_to_xlsx(dirname, xyscale, zdist, dt):
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
        dt (int): time step between frames
        xyscale (float): conversion rate from pixels to real units
        zdist (floar): real distance between z planes        
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
    df = DataFrame({'id': axlsxdata[:,0], 'time': axlsxdata[:,1], 'x': axlsxdata[:,2], 'y': axlsxdata[:,3], 'z': axlsxdata[:,4]})
    df.to_excel(savename, sheet_name='trajectories', index=False)  
    return

def get_traj(dirname):
    '''
    This function gets all the tracks in an excel file and creates an array containing every track.
    Excel files need to have the following column order: [track_id, t, x, y, (z)].

    Args:
        dirname (string): path to the folder containing the Excel files

    Returns.
        list: list of track arrays    
    '''
    #read files in dirname and return the dataframe as a numpy array
    #a list of numpy arrays is created, with one array element per file
    file_list = []
    file_names = []
    if os.path.isdir(dirname):
        for fname in os.listdir(dirname):
            ext = os.path.splitext(fname)[-1].lower()
            if ext=='.xlsx' or ext=='.xls':
                file_names.append(fname.replace(ext,''))
                df = pd.read_excel(os.path.join(dirname,fname))
                np_df = df.to_numpy()
                file_list.append(np_df)
    elif os.path.isfile(dirname):
        ext = os.path.splitext(dirname)[-1].lower()
        if ext=='.xlsx' or ext=='.xls':
            file_names.append(os.path.split(dirname)[-1].replace(ext,''))
            df = pd.read_excel(os.path.abspath(dirname))
            np_df = df.to_numpy()
            file_list.append(np_df)

    #get the trajectories of the given files, with the length of all trajectories being the minimum of the dataset
    file_tracks = []
    for file in file_list:
        all_tracks = []
        all_tracks_len = []
        ids = np.unique(file[:,0]) #get tracks id from the dataset

        #separate tracks in different 2D arrays
        for track_id in ids:
            track = []
            for i in range(0, len(file[:,0])):
                if file[i,0]==track_id:
                    track.append(file[i,:])
            track = np.array(track)
            track_len = len(track)
            all_tracks_len.append(track_len) #collecting the length of al tracks
            all_tracks.append(track)

        #making all trajectories have the same length, being that length the minimum one
        min_len = min(all_tracks_len)
        cropped_tracks = []
        for track in all_tracks:
            cropped_tracks.append(track[:min_len])
        c_tracks = np.array(cropped_tracks)        
        file_tracks.append(c_tracks)

    return file_tracks, file_names

def velocity(dirname, dt, save_xlsx=True, savedir=None):
    '''
    This function is used to calculate the velocity of the cells. It performs the mean between the velocities of all tracks and generates a file containing the velocity for each time frame.

    Args:
        dirname (string): path to the folder containing the Excel files
        dt (int): trajectory time step size
        save_xlsx (bool): Defults to True. Determines if an Excel file with the results of the function will be saved.
        savedir (string): Defaults to None. Path to the folder where results will be saved. If set to None, a folder will be created in the directory containing the tracks.

    Returns:
        list: list of arrays where each array corresponds to the mean velocities of a  given file.

    TODO:
        could be interesting to save velocity for each track to perform metrics like in CellTracksColab
    '''
    all_files, name_list = get_traj(dirname)

    if os.path.isfile(dirname):
        raise ValueError('Error: please indicate the path to the folder where results will be saved.')

    if savedir==None:
        if not os.path.exists(os.path.abspath(fr'{dirname}\Results')):
            os.makedirs(os.path.abspath(fr'{dirname}\Results'))
        savedir = os.path.abspath(fr'{dirname}\Results')
    
    name = 0
    for file in all_files:
        file_velocity = []
        time = np.linspace(1, len(file[0])-1, len(file[0])-1)
        #calculate velocity for each track in the file
        for track in file:
            xyz = track[:,2:]
            dxyz = np.diff(xyz, axis=0)
            vxyz = np.abs(dxyz/dt)
            vr = np.sqrt(vxyz[:,0]**2+vxyz[:,1]**2+vxyz[:,2]**2)
            v = np.array([vr, vxyz[:,0], vxyz[:,1], vxyz[:,2]])
            file_velocity.append(np.transpose(v))

        mean_velocity = np.zeros((np.shape(np.transpose(v))))
        for k in range(0,np.shape(v)[1]):
            mv = []
            for track in file_velocity:
                mv.append(track[k,:])
            mean_velocity[k,:] = np.median(mv, axis=0)
    
        #save velocity into excel files, if multiple files have been readed, multiple excel files will be created
        if save_xlsx==True:
            savename = f'{os.path.join(savedir,name_list[name])}_velocity.xlsx'
            df = DataFrame({'time': time*dt, 'r_velocity': abs(mean_velocity[:,0]), 'x_velocity': abs(mean_velocity[:,1]), 'y_velocity': abs(mean_velocity[:,2]), 'z_velocity': abs(mean_velocity[:,3])})
            df.to_excel(savename, sheet_name='velocity', index=False)
            name = name+1
    return

def ezmsd(xyz):
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
        for tlag in range(1, fn):
            dxyz = xyz[tlag:,:] - xyz[:-tlag,:]
            msdr[tlag-1] = np.mean(np.sum(dxyz**2, axis=1))
    else:
        for tlag in range(1,fn):
            dxyz = xyz[tlag:] - xyz[:-tlag]
            msdr[tlag-1] = np.mean(dxyz**2)
    return msdr

def get_msd(dirname, dt, save_xlsx=True, savedir=None, show_fig=False):
    '''
    
    This function determines the msd for each file in the directory, with the corresponding error.
    Excel files need to have the following column order: [track_id, t, x, y, (z)].

    Args:
        dirname (string): path to the folder containing the Excel files.
        dt (int): trajectory time step size
        save_xlsx (bool): Defults to True. Determines if an Excel file with the results of the function will be saved.
        savedir (string): Defaults to None. Path to the folder where results will be saved. If set to None, a folder will be created in the directory containing the tracks.
        show_fig (bool): Defaults to False. It is used to work directly with tracks and not loading an exel. Tracks and names should be input in respective params.

    Returns:
        list: list of mean msd arrays, one array per file in the directory. Mean msd arrays columns correspond to [time lag, mean msd, error msd].

    TODO:
        determine cumulated error from msd calculation in every track and mean calculation.    
    '''
    all_files, name_list = get_traj(dirname)
    
    if savedir==None:
        if os.path.isfile(dirname):
            raise ValueError('Error: please indicate the path to the folder where results will be saved.')
        if not os.path.exists(os.path.abspath(fr'{dirname}\Results')):
            os.makedirs(os.path.abspath(fr'{dirname}\Results'))
        savedir = os.path.abspath(fr'{dirname}\Results')

    all_files_msd = []
    all_files_mean_msd = []
    name = 0
    if show_fig==True:
        plt.figure()
    for file in all_files:
        msdt = []
        t = np.linspace(1, len(file[0])-1, len(file[0])-1) #define time
        #calculate msd for each track in the file
        for track in file:
            xyz = track[:,2:]
            msd0 = ezmsd(xyz)
            msdt.append(msd0)
            msd = np.array(msdt)

        #calculate mean msd
        mean_msd = np.zeros((len(msd[0])))
        std_dev = np.zeros((len(msd[0])))
        for k in range(0, len(msd[0])):
            mean_msd[k] = np.mean(msd[:,k])
            std_dev[k] = np.std(msd[:,k])/np.sqrt(len(msd[:,k]))

        #adding msd to all files list
        all_files_msd.append(np.array(msd))
        all_files_mean_msd.append(mean_msd)

        #save the file
        if save_xlsx==True:
            savename = f'{os.path.join(savedir,name_list[name])}_MSD.xlsx'
            df = DataFrame({'Time lag': t*dt, 'msd': mean_msd, 'std': std_dev})
            df.to_excel(savename, sheet_name='MeanMSD', index=False)

        if show_fig==True:
            plt.plot(t*dt, mean_msd, label=name_list[name])
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
        
        name = name+1

    if show_fig==True:
        plt.show()

    return

def get_acf(dirname, dt, save_xlsx=True, savedir=None, show_fig=False):
    '''
    This function determines the acf for each file in the directory, with the corresponding error.
    Excel files need to have the following column order: [track_id, t, x, y, (z)].

    Args:
        dirname (string): path to the folder containing the Excel files
        dt (int): trajectory time step size
        save_xlsx (bool): Defults to True. Determines if an Excel file with the results of the function will be saved.
        savedir (string): Defaults to None. Path to the folder where results will be saved. If set to None, a folder will be created in the directory containing the tracks.

    Returns:
        list: list of mean acf arrays, one array per file in the directory. Mean acf arrays columns correspond to [time lag, mean acf, error acf].

    TODO:
        Some tracks present strange behaviour when simulated. Possible index error
    '''
    all_files, name_list = get_traj(dirname)

    if os.path.isfile(dirname):
        raise ValueError('Error: please indicate the path to the folder where results will be saved.')

    if savedir==None:
        if not os.path.exists(os.path.abspath(fr'{dirname}\Results')):
            os.makedirs(os.path.abspath(fr'{dirname}\Results'))
        savedir = os.path.abspath(fr'{dirname}\Results')

    if show_fig==True:
        plt.figure()
    name=0
    for file in all_files:
        t = np.linspace(1, len(file[0])-2, len(file[0])-2)
        acf_list = []
        #calculate acf for each track in the file
        for track in file:
            xyz = track[:,2:]
            dxyz = xyz[1:,:]-xyz[:-1,:]
            M = len(dxyz[:,0])
            acft = []
            #calculate the acf for the current track
            for tlag in range(0, M-1):
                acft.append(np.sum(np.sum(dxyz[:M-tlag,:]*dxyz[tlag:M,:], axis=1))/(M-tlag))
            acf = np.array(acft)
            acf_list.append(acf)
            aacf = np.array(acf_list)

        #calculate mean acf
        mean_acf = np.zeros((len(acf)))
        std_dev = np.zeros((len(acf)))
        for k in range(0, len(acf)):
            mean_acf[k] = np.mean(aacf[:,k])
            std_dev[k] = np.std(aacf[:,k])/np.sqrt(len(aacf[:,k]))

        norm = sum(abs(mean_acf[:]))

        if show_fig==True:
            plt.plot(t*dt, mean_acf/norm, label=name_list[name])
            plt.legend()

        #save the file
        if save_xlsx==True:
            savename = f'{os.path.join(savedir,name_list[name])}_ACF.xlsx'
            df = DataFrame({'Time lag': t*dt, 'acf': abs(mean_acf/norm), 'std': std_dev/norm})
            df.to_excel(savename, sheet_name='MeanACF', index=False)

        name = name+1

    if show_fig==True:
        plt.show()

    return

def PDF_dR(dirname, dt, tlag, dmax=30, binn=50, tunit='min', save_xlsx=True, savedir=None):
    '''
    This function calculates the probability density function (PDF) of the displacement. 
    For a given time lag, it calculates all displacements and distributes them along the given number of bins, up to the maximum displacement assigned.

    Args:
        dirname (string): path to the folder containing the Excel files
        dt (int): trajectory time step size
        tlag (int): time lag where probability density function of the displacement will be calculated.
        dmax (float): maximum distance cells can travel in the selected time lag.
        binn (int): number of bins between 0 and dmax.
        tunit (string): Defaults to min. Indicates the dt and tlag unit. It will be used to save in the file name. 
        save_xlsx (bool): Defults to True. Determines if an Excel file with the results of the function will be saved.
        savedir (string): Defaults to None. Path to the folder where results will be saved. If set to None, a folder will be created in the directory containing the tracks.

    returns:
        list: list of lists for each file. Each file list corresponds to two arrays, the first one for the histogram data and the second one for the bins.
    '''
    all_files, name_list = get_traj(dirname)

    if os.path.isfile(dirname):
        raise ValueError('Error: please indicate the path to the folder where results will be saved.')

    if savedir==None:
        if not os.path.exists(os.path.abspath(fr'{dirname}\Results')):
            os.makedirs(os.path.abspath(fr'{dirname}\Results'))
        savedir = os.path.abspath(fr'{dirname}\Results')

    all_files_PDF_dR = []
    bin_distr = np.linspace(0, dmax, binn)
    ntlag = int(tlag/dt)
    for file in all_files:
        displ = []
        #calculate PDF_dR for each track in the file
        for track in file:
            t = track[:,1]
            xyz = track[:,2:]
            dxyz = xyz[ntlag:,:]-xyz[:-ntlag,:]
            displ.append(np.mean(dxyz,axis=1))
        hist, bins = np.histogram(np.array(displ), bin_distr, density=True)
        all_files_PDF_dR.append([hist, bins[0:-1]])

    #save PDF_dR into excel files, if multiple files have been readed, multiple excel files will be created
    name = 0
    if save_xlsx==True:
        for file in all_files_PDF_dR:
            savename = f'{os.path.join(savedir,name_list[name])}_PDF_dR_{tlag}{tunit}.xlsx'
            df = DataFrame({'bins': file[1], 'PDF_dR': file[0]})
            df.to_excel(savename, sheet_name='PDF_dR', index=False)
            name = name+1

    return

def PDF_dtheta(dirname, dt, tlag, binn=9, tunit='min', save_xlsx=True, savedir=None):
    '''
    This function calculates the probability density funtion (PDF) of the angle. 
    Given a time lag, the function calculates the angle turned (between 0 and 180º) between the current position and the position one time lag after.

    Args:
        dirname (string): path to the folder containing the Excel files
        dt (int): trajectory time step size
        tlag (int): time lag where probability density function of the turning angle will be calculated.
        binn (int): number of bins between 0 and 180.
        tunit (string): Defaults to min. Indicates the dt and tlag unit. It will be used to save in the file name. 
        save_xlsx (bool): Defults to True. Determines if an Excel file with the results of the function will be saved.
        savedir (string): Defaults to None. Path to the folder where results will be saved. If set to None, a folder will be created in the directory containing the tracks.
    '''
    all_files, name_list = get_traj(dirname)

    if os.path.isfile(dirname):
        raise ValueError('Error: please indicate the path to the folder where results will be saved.')

    if savedir==None:
        if not os.path.exists(os.path.abspath(fr'{dirname}\Results')):
            os.makedirs(os.path.abspath(fr'{dirname}\Results'))
        savedir = os.path.abspath(fr'{dirname}\Results')

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

def polarity_dR(dirname, dt, tlag, binn = 20, savedir = None):
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

def tPRW3D(x, P, S, SE):
    return 3*(S**2)*P*(x-P*(1-np.exp(-x/P)))+6*SE**2

def tPRW2D(x, P, S, SE):
    return 2*(S**2)*P*(x-P*(1-np.exp(-x/P)))+4*SE**2

def tPRW1D(x, P, S, SE):
    return (S**2)*P*(x-P*(1-np.exp(-x/P)))+2*SE**2

def fit_PRW(dirname, dt, dim):
    '''
    This function calculates the msd for every track and fits a persisten random walk model in each one, obtaining the parameters P, S, and SE for every track.

    Args:
        filename (string): path to the msd file.
        dt (int): trajectory time step size
        dim (int): dimensionality of the tracks (2 for 2D or 3 for 3D)

    TODO:
        add error of the fittings and R^2
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

def fit_APRW(dirname, dt, dim, tlag=1):
    '''
    This functions rotates the trajectories to the primary axis of migration (p) and calculates the msd for the primary and non primary axis.

    Args:
        dirname (string): path to the folder containing the original trajectories
        dt (int): trajectory time step size
        dim (int): dimensionality of the tracks (2 for 2D or 3 for 3D)
        tlag (int): defaults to 1. Time lag to determine the rotational matrix
        
    TODO:
        Fitting process has large mse when using more than 1/4 of the data, further investgation needed
    '''
    track_list, name_list = get_traj(dirname)

    savedir = os.path.abspath(fr'{dirname}\APRW')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    name=0
    for file in track_list:
        params_list = []
        for track in file:
            xyz = track[:,2:]
            used_len = np.uint16(np.round(len(track))/4)
            dxyz = xyz[tlag:,:]-xyz[:-tlag,:]
            xyzr = xyz-np.mean(xyz, axis=0) #major axis of trajectories
            U,S,V = np.linalg.svd(dxyz, full_matrices=True)
            xyzrot = xyz@np.transpose(V)
            
            time = np.linspace(1, used_len, used_len)
            t = time*dt
            wif = (2*time**2+1)/(3*time*(used_len-time+1))
            msdp0 = ezmsd(xyzrot[:,0])
            msdp = msdp0[:used_len]
            wtp = (msdp*wif**2)
            msdnp0 = ezmsd(xyzrot[:,1])
            if dim==3:
                msdnp0 += ezmsd(xyzrot[:,2])
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

            """ plt.figure()
            plt.plot(t, msdp, color='C0')
            plt.plot(t, msdnp, color='C1')
            plt.plot(t, tPRW1D(t, *poptp), color='C0', linestyle='dashed')
            plt.plot(t, tPRW2D(t, *poptnp), color='C1', linestyle='dashed')
            plt.show() """

            Pp,Sp,SEp = poptp
            Pnp,Snp,SEnp = poptnp
            params_list.append([Pp,Sp,SEp,rmsep])
            params_list.append([Pnp,Snp,SEnp,rmsenp])

        params=np.array(params_list)

        savename = f'{os.path.join(savedir,name_list[name])}_APRW_params.xlsx'
        pd.DataFrame({'P':params[:,0], 'S':params[:,1], 'SE':params[:,2], 'rmse':params[:,3]}).to_excel(savename, sheet_name='params', index=False)
        name = name+1

    return params

def sim_APRW(dirname, tlag, dim, Nmax=50, repeats=30, subsamples=100):
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
        for i in range(0, len(file), 2):
            P = file[i:i+2,0]
            S = file[i:i+2,1]
            mSE = file[i:i+2,2]
            beta = 1/P
            alpha = (S**2)*beta
            map = [max(0, 1-dt/P[0]), max(0, 1-dt/P[1])]
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