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

def translationComputation(imgs, positions, n=8) -> np.ndarray:
    """
    This is the final function to obtain the translation vectors for all the tiles to obtain the resulting image

    Args:
        imgs (ndArray): array of tiles with shape (t,m,z,x,y)
        positions (list): list of tuples containing the position of the tiles
    """
    #Translating the given image array to the format needed to proceed
    nrow=0
    ncol=0
    eq_img = []
    img = []
    for timestep in imgs[0]:
        grid={}
        eq_grid={}
        for pos, tile in zip(positions, timestep):
            #Recorder of number of rows and cols
            if pos[1]+1>nrow:
                nrow=pos[1]+1
            if pos[0]+1>ncol:
                ncol=pos[0]+1
            #eq_tile=exposure.equalize_adapthist(tile, clip_limit=0.75) #Sometimes is better to equalize the tiles to find the translation values
            grid[f'{pos[1]}{pos[0]}']=tile
            #eq_grid[f'{pos[1]}{pos[0]}']=eq_tile
        #eq_img.append(eq_grid)
        img.append(grid)
    
    #Finding the peaks for some t and z
    translations = []
    for t in tqdm(range(0, len(imgs[0]), int(len(imgs[0])/10))):
        grid_ = img[t]
        for z in range(0, len(imgs[0][0]), int(len(imgs[0][0])/4)):
            Tvcol=[]
            Throw=[]
            nccv=-np.inf
            ncch=-np.inf
            for row in np.arange(nrow):
                for col in np.arange(ncol):
                    im2 = grid_[f'{row}{col}'][z]
                    H,W = im2.shape
                    if row!=0:
                        im = grid_[f'{row-1}{col}'][z]
                        nccv_, Tvrow_, Tvcol_ = pciam(im, im2, n)
                        if abs(Tvcol_)<int(W/5):
                            Tvcol.append(Tvcol_)
                        if nccv<nccv_:
                            nccv=nccv_
                            Tvrow=Tvrow_
                    if col!=0:
                        im = grid_[f'{row}{col-1}'][z]
                        ncch_, Throw_, Thcol_ = pciam(im, im2, n)
                        if abs(Throw_)<int(H/5):
                            Throw.append(Throw_)
                        if ncch<ncch_:
                            ncch=ncch_
                            Thcol=Thcol_
            if Throw==[]:
                Throw=[0]
            if Tvcol==[]:
                Tvcol=[0]
            Tv = [int(abs(Tvrow)), int(np.average(Tvcol))]
            Th = [int(np.average(Throw)), int(abs(Thcol))]
            rr = Th[0]
            rc = Tv[1]
            drow = Th[1]-rr
            dcol = Tv[0]-rc
            translations.append([drow, rr, dcol, rc])

    return translations

