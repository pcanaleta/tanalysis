import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import exposure

### Translation Computation ###
def pcm(image1:np.ndarray, image2:np.ndarray):
    """
    This function performs the peak correlation matrix as described in MIST algorithms. Input images have to be the same size.

    Args:
        image1 (ndArray): 2D array image
        image2 (ndArray): 2D array image

    Returns:
        PCM (ndArray): 2D peak correlation array with the same size as the original images
    """
    assert image1.ndim == image2.ndim == 2
    assert image1.shape == image2.shape
    F1 = np.fft.fft2(image1)
    F2 = np.fft.fft2(image2)
    FC = F1 * np.conjugate(F2)
    PCM = np.fft.ifft2(FC/np.abs(FC))
    return PCM.real.astype(np.float32)

def multiPeakMax(PCM:np.ndarray):
    """
    This function finds the multiple discrete peaks in a pcm matrix

    Args:
        PCM (ndArray): 2D array with peak correlation values

    Returns:
        tuple: array of n tuples (x,y,val) where x,y correspond to the peak location described in matrix indices
    """
    row, col = np.unravel_index(np.argsort(np.ravel(PCM)), PCM.shape)
    vals = PCM[row[::-1], col[::-1]]
    return np.array((row[::-1], col[::-1], vals))

def ncc(image1:np.ndarray, image2:np.ndarray):
    """
    This function returns the normalized cross correlation coefficient between two arrays. The two arrays must have the same size

    Args:
        image1 (ndArray): 2D array corresponding to one of the images
        image2 (ndArray): 2D array corresponding to the second image

    Returns:
        float: normalized cross correlation coefficient    
    """
    assert image1.shape == image2.shape
    I1 = image1.flatten()
    I2 = image2.flatten()
    n = np.dot(I1 - np.mean(I1), I2 - np.mean(I2))
    d = np.linalg.norm(I1 - np.mean(I1)) * np.linalg.norm(I2 - np.mean(I2))
    return n/d

def extractOverlapSubregion(image:np.ndarray, row:int, col:int):
    """
    This function extracts the overlapping region with another image given the coordinates where the overlapping begins

    Args:
        image (ndArray): 2D image array where we want to determine the overlapping region
        row (int): index of the row where the overlapping begins
        col (int): index of the column where the overlapping begins

    Returns:
        ndArray: array containing the overlapping region of the given image
    """
    H,W = image.shape
    colstart = int(max(0, min(col, W, key=int), key=int))
    colend = int(max(0, min(col+W, W, key=int), key=int))
    rowstart = int(max(0, min(row, H, key=int), key=int))
    rowend = int(max(0, min(row+H, H, key=int), key=int))
    return image[rowstart:rowend, colstart:colend]

def interpretTranslation(image1: np.ndarray, image2: np.ndarray, rowin, colin, rowmin, rowmax, colmin, colmax, n=8):
    """
    This function computes all the possible coordinate combinations when overlapping two images and extracts the set of coordinates with the highest ncc value, which will correspond to the translation between the two images

    Args:
        image1 (ndArray): 2D image array
        image2 (ndArray): 2D image array
        xin (int): row index coordinate
        yin (int): column index coordinate 
        perc_threshold (int): maximum accepted percentage of overlapping between the two images

    Returns:
        ndArray: tuple containing the ncc,x,y of the translation between image1 and image2
    """
    assert image1.shape == image2.shape
    _ncc = -np.inf
    x = 0
    y = 0
    H,W = image1.shape

    rowmagss = [rowin, H-rowin]
    colmagss = [colin, W-colin]
    _poss = []
    for rowmag in rowmagss:
        for colmag in colmagss:
            for rowsign in [-1,1]:
                for colsign in [-1,1]:
                    _poss.append([rowmag*rowsign, colmag*colsign])
    poss = np.array(_poss)
    valid_ind = (
        (rowmin < poss[:, 0, :])
        & (poss[:, 0, :] < rowmax)
        & (poss[:, 0, :] != 0)
        & (colmin < poss[:, 1, :])
        & (poss[:, 1, :] < colmax)
        & (poss[:, 1, :] != 0)
    )  
    valid_ind = np.any(valid_ind, axis=0)

    for pos in np.moveaxis(poss[:,:,valid_ind], -1, 0)[:int(n)]:
        for rowval, colval in pos:
            if (colmin <= colval) and (colval <= colmax) and (rowmin <= rowval) and (rowval <= rowmax):
                subI1 = extractOverlapSubregion(image1, rowval, colval)
                subI2 = extractOverlapSubregion(image2, -rowval, -colval)
                ncc_val = ncc(subI1, subI2)
                if ncc_val>_ncc:
                    _ncc = float(ncc_val)
                    x = int(rowval)
                    y = int(colval)
    return np.asarray([_ncc,x,y])

def pciam(image1:np.ndarray, image2:np.ndarray, n=8):
    """
    This function finds the north and west translation between two images. It performs the PCM between the images, extracts n peaks from the PCM and interpretes the coordinates of the peaks for each peak

    Args:
        image1 (ndArray): 2D image array
        image2 (ndArray): 2D image array

    Returns:
        ndArray: tuple containing the peak with the max ncc value (ncc, x, y)
    """
    PCM = pcm(image1, image2)
    H, W = np.shape(image1)
    rowin, colin, val = multiPeakMax(PCM)
    max_peak = np.asarray(interpretTranslation(image1, image2, rowin, colin, -H, H, -W, W, n))
    return max_peak

def make_grid(im_list, positions):
    '''
    This function converts the resulting image list from the imread function to a list of dictionaries where each dictionary corresponds to a timeframe. 
    The keys in the dictionary correspond to the row/col position of the tile assigned to the key

    Args:
        im_list (list): list of arrays obtained from imread function
        positions (tuple): list of tuples obtained from imread function
    '''
    nrow=0
    ncol=0
    grid_list=[]
    for im in im_list:
        img = []
        for timestep in im:
            grid={}
            for pos, tile in zip(positions, timestep):
                if pos[1]+1>nrow:
                    nrow=pos[1]+1
                if pos[0]+1>ncol:
                    ncol=pos[0]+1
                grid[f'{pos[1]}{pos[0]}']=tile
            img.append(grid)
        grid_list.append(img)
    return grid_list, nrow, ncol

def translationComputation(imgs, positions, n=8) -> np.ndarray:
    """
    This is the final function to obtain the translation vectors for all the tiles to obtain the resulting image

    Args:
        imgs (ndArray): array of tiles with shape (t,m,z,x,y)
        positions (list): list of tuples containing the position of the tiles
    """
    #Translating the given image array to the format needed to proceed
    grid_list, nrow, ncol = make_grid(imgs, positions)
    
    #Finding the peaks for some t and z
    translations_list = []
    for img in grid_list:
        translations = []
        for t in tqdm(range(0, len(imgs[0]), int(len(imgs[0])/10)), 'Calculating translation vectors'):
            grid_ = img[t]
            for z in range(0, len(imgs[0][0]), int(len(imgs[0][0])/4)):
                Tvcol=[]
                Throw=[]
                Tvrow=[]
                Thcol=[]
                nccv=-np.inf
                ncch=-np.inf
                for row in np.arange(nrow):
                    for col in np.arange(ncol):
                        im2 = grid_[f'{row}{col}'][z]
                        H,W = im2.shape
                        if row!=0:
                            im = grid_[f'{row-1}{col}'][z]
                            nccv_, Tvrow_, Tvcol_ = pciam(im, im2, n)
                            if 0.5<nccv_ and abs(Tvrow_)>=int(H*0.8):
                                Tvcol.append(Tvcol_)
                                Tvrow.append(Tvrow_)
                        if col!=0:
                            im = grid_[f'{row}{col-1}'][z]
                            ncch_, Throw_, Thcol_ = pciam(im, im2, n)
                            if 0.5<ncch_ and abs(Thcol_)>=int(W*0.8):
                                Throw.append(Throw_)
                                Thcol.append(Thcol_)
                if Throw==[]:
                    Throw=[0]
                if Tvcol==[]:
                    Tvcol=[0]
                if Tvrow==[]:
                    Tvrow=[0]
                if Thcol==[]:
                    Thcol=[0]
                Tv = [int(np.average(Tvrow)), int(np.average(Tvcol))]
                Th = [int(np.average(Throw)), int(np.average(Thcol))]
                rr = Th[0]
                rc = Tv[1]
                drow = Th[1]-rr
                dcol = Tv[0]-rc
                translations.append([drow, rr, dcol, rc])
        print('All vectors calculated!')
        arr_translations = np.asarray(translations)
        drow, rr, dcol, rc = int(np.median(arr_translations[:,0])), int(np.median(arr_translations[:,1])), int(np.median(arr_translations[:,2])), int(np.median(arr_translations[:,3]))
        translations_list.append([drow, rr, dcol, rc])
        print(translations_list)
    return translations_list

def image_reconstruction(imgs, positions, n=8):
    '''
    This function reconstructs the mosaic image using translation vectors for the tiles. Calculation of this translation vectors
    '''
    grid_list, nrow, ncol = make_grid(imgs, positions)
    translations_list = translationComputation(imgs, positions, n)

    res_img_list = []
    for trans_set, grid in zip(translations_list, grid_list):
        abs_translations = {}
        minr=0
        minc=0
        drow, rr, dcol, rc = trans_set
        for row in np.arange(nrow):
            for col in np.arange(ncol):
                abs_translations[f'{row}{col}'] = [int(row*(drow+rr)+col*rr), int(row*rc+col*(dcol+rc))]
                minr_ = abs_translations[f'{row}{col}'][0]
                minc_ = abs_translations[f'{row}{col}'][1]
                if minr_<minr:
                    minr=minr_
                if minc_<minc:
                    minc=minc_
        H,W = imgs[0].shape[-2], imgs[0].shape[-1]
        Hmax,Wmax = abs_translations[f'{nrow-1}{ncol-1}']
        rerr = abs(minr)
        cerr = abs(minc)
        t_result = []
        for grid_t in grid:
            z_result = []
            for z in np.arange(imgs[0].shape[-3]):
                result = np.zeros((Hmax+H+2*rerr, Wmax+W+2*cerr))
                for trans in abs_translations:
                    srow = abs_translations[trans][0]+rerr
                    scol = abs_translations[trans][1]+cerr
                    erow = srow+H
                    ecol = scol+W
                    result[srow:erow,scol:ecol] = grid_t[trans][z]+result[srow:erow,scol:ecol]
                z_result.append(result)
            t_result.append(z_result)
        res_img = np.asarray(t_result)
        res_img_list.append(res_img)
    return res_img_list