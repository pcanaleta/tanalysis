import os
import tifffile as tiff
import numpy as np
import shutil
import liffile as lif

try:
    from readlif.reader import LifFile # type: ignore
    READLIF = True
except:
    READLIF = False

try:
    from cellpose import io, models, train, denoise # type: ignore
    CELLPOSE = True
except:
    CELLPOSE = False

def imread(dirname:str, tiles:bool=False, gpu:bool=False):
    '''
    This function reads images from files or directories. It only accepts .tif, .tiff or .lif files for now. 
    .tif or .tiff files should be images of the channel to read, as the function is not prepared to accept files with this extensions with multiple channels.
    For .lif files, the function is prepared to read raw .lif images with 1 images in the file and channel of interest being the second channel.

    Args:
        dirname (string): Path to the file or folder containging files to read
        channel (int): channel to read in lif files. Defaults to 1 
        tiles (bool): read tiles images (True) or not (False). Defaults to False

    Returns:
        tuple: A tuple containing the list of image arrays, the value (int) of the image dimensions, a list of the file names and a dictionary containing additional info
            im_list: list of ndarrays of the readed images
            dim: int value of the dimensions of the image. All images have to be the same size
            im_name: list of names of all 
            im_info: {
                'scale': returns the conversion values for the images. x,y,z values are in pixels/micron and t values in frame/sec
                'mosaic_position': returns a list of tuples of the position of all tiles in the lif file. The tuples contain (col, row, x_pos, y_pos). Only available if tiles=True}
    
    Raises:
        ImportError: if liffile is not installed in the venv
        ValueError: if files do not have supported extensions (.tif, .tiff, .lif)
        ValueError: if submited directory is empty
        ValueError: if image in the directory have different dimensions
    '''
    dirname = os.path.abspath(dirname)
    im_list = [] #list of read images
    im_name = [] #list of read images' names
    im_info = {}

    #List all given files
    file_list = []
    if os.path.isfile(dirname):
        file_list.append(dirname)
    elif os.path.isdir(dirname):
        if len(os.listdir(dirname)) == 0:
            raise ValueError('ERROR: submitted directory is empty')
        else:
            for fname in os.listdir(dirname):
                file_list.append(os.path.join(dirname,fname))
    else:
        raise ValueError('ERROR: no directory or file submitted')
    
    #Read all files in file_list
    i=0
    for file in file_list:
        ext = os.path.splitext(file)[-1].lower()
        file_name = os.path.split(file)[-1].replace(ext, "")
        im_name.append(f'{file_name.replace(ext, '')}-{i}')
        i=i+1
        #For tiff files
        if ext==".tif" or ext==".tiff":
            image = np.asarray(tiff.imread(file))
            im_list.append(image)
        #For lif files
        elif ext==".lif":
            if not READLIF:
                raise ImportError('ERROR: cannot import liffile, please use: pip install -U readlif[all]')
            else:
                for i in range(0,10):
                    try:
                        im = lif.imread(file, image=i)
                        im_list.append(im)
                    except:
                        continue
                lif_file = LifFile(file)
                for image_0 in lif_file.get_iter_image():
                    im_info['scale'] = image_0.info['scale']
                    if tiles==True:
                        im_info['mosaic_position'] = image_0.info['mosaic_position']
        else:
            raise ValueError('ERROR: submited file does not have a supported extension (.tif, .tiff, .lif)')

    return im_list, im_name, im_info

def cellposeseg(images:list[np.ndarray], dim:int, im_name:list[str], savedir:str, modelpath:str="",):
    '''
    This function segmentates the images with the model selected. The segmented images are saved in the specified directory. 
    
    Args:
        images (list): list of images in array format
        dim (int): number of dimensions of the images
        im_name (list): list of lists of image names
        savedir (string): path to directory where images will be saved
        modelpath (string): path to pretained model 

    Returns:
        list: List of directories where images are saved

    Raises:
        ImportError: if cellpose package is not installed
        ValueError: if the image/s selected are not 2D or 3d
    '''
    if not CELLPOSE:
        raise ImportError('ERROR: Cellpose package is not installed, please use: pip install cellpose[all]')
    
    io.logger_setup()

    model = models.CellposeModel(gpu=True)

    if os.path.isfile(modelpath):
        model = models.CellposeModel(gpu=True, pretrained_model=modelpath)

    if dim == 3:
        do_3D = True
    elif dim == 2:
        do_3D = False
    else:
        raise ValueError ('ERROR: selected image has to be 2D or 3D')

    for image, name in zip(images, im_name):
        timer=0
        temp_savedir = os.path.join(savedir, name)
        if not os.path.exists(temp_savedir):
            os.makedirs(temp_savedir)
            print(temp_savedir)
        for time_frame in image:
            masks, flows, styles = model.eval(time_frame, do_3D=do_3D, z_axis=0, normalize={'percentile':[1,100]})
            io.save_masks(time_frame, masks, flows, f'{name}_T{timer}.tif', tif=do_3D, png=not(do_3D), savedir=temp_savedir)
            timer=timer+1 
    return temp_savedir

def concatenate(dirname, remove_original=False):
    '''
    This function concatenates all images in the given directory and saves the resulting image. Original images can be removed.

    Args:
        dirname (string): path of the folder containing the images to concatenate
        remove_original (boolean, optional): option to remove original images. Defaults to False

    Raises:
        ValueError: if path given is not a folder
    '''
    if os.path.isdir(dirname):
        images, names, info = imread(dirname)
        newname = f'{os.path.abspath(os.path.join(dirname,names[0].replace('_T0_cp_masks','')))}.tiff'
        im_concat = np.stack(images, 0)
        
        if remove_original == True:
            try:
                shutil.rmtree(dirname)
                os.makedirs(dirname)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (dirname, e))
        
        axes = 'TZYX'
        
        tiff.imwrite(
            newname, 
            im_concat, 
            imagej=True,
            metadata={
                'axes':axes
            })

    else:
        raise ValueError('ERROR: path to a folder is needed')
    
    return

def cellposeseg_bigdata(dirname, zarr_path, chunks={0:256,1:256,2:'auto'}, blocksize={0:256,1:256,2:'auto'}, time_frames=False, model=None, do_3D=True):

    if not CELLPOSE:
        raise ImportError('ERROR: Cellpose package is not installed, please use: pip install cellpose')

    from cellpose.contrib.distributed_segmentation import distributed_eval, numpy_array_to_zarr # type: ignore
    
    images, dim, im_name = imread(dirname, n_images=1)

    #convert the numpy array to a zarr

    if time_frames == True:
        im_tf = images
        dim = dim+1

    image_names = []
    for j in range(0, len(im_name)):
        names = []
        for i in range(0, len(im_tf[j])):
            names.append(f'{im_name[j]}_T{i}')
        image_names.append(names)

    data_zarr=[]
    
    for k in range(0,len(im_tf)):
        for image in im_tf:
            data_zarr.append(numpy_array_to_zarr(zarr_path, image[k], chunks))

    #parametrize cellpose
    if model==None:
        model_kwargs = {'gpu':True, 'model_type':'cyto3'}
    else:
        model_kwargs = {'gpu':True, 'pretrained_model':os.path.abspath(model)}

    eval_kwargs = {'diameter':30,
                   'z_axis':0,
                   'channels':[0,0],
                   'do_3D':do_3D}
    
    #define compute resources for local workstation
    cluster_kwargs = {
        'n_workers':1,
        'ncpus':8,
        'memory_limit':'64GB',
        'threads_per_worker':1
    }

    #segmentation
    for j in range(0, len(data_zarr)):
        segments, boxes = distributed_eval(
            input_zarr=data_zarr[j],
            blocksize=blocksize,
            write_path=f'{os.path.join(dirname,im_name[j])}.zarr',
            model_kwargs=model_kwargs,
            eval_kwargs=eval_kwargs,
            cluster_kwargs=cluster_kwargs
    )

    return