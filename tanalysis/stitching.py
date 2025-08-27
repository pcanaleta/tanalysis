import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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

def interpretTranslation(image1: np.ndarray, image2: np.ndarray, rowin, colin, rowmin, rowmax, colmin, colmax, n=5):
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

def pciam(image1:np.ndarray, image2:np.ndarray):
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
    rowin, colin, _ = multiPeakMax(PCM)
    max_peak = np.asarray(interpretTranslation(image1, image2, rowin, colin, -H, H, -W, W))
    return max_peak

#############################################################################Fix this function#########################################################################################
def translationComputation(imgGrid, final_shape) -> np.ndarray:
    """
    imgGrid = arr[ims] with shape (mosaic_row,mosaic_col) where row and col are extracted from metadata
    """
    #change final shape for imgGrid.shape
    transComp = np.empty((np.shape(imgGrid)))
    for row in range(0,final_shape[0]):
        for col in range(0,final_shape[1]):
            Tv,Th = None
            im1 = imgGrid[row,col]
            if row is not final_shape[0]:
                im2v = imgGrid[row+1,col]
                Tv = pciam(im1,im2v,True)
            if col is not final_shape[1]:
                im2h = imgGrid[row,col+1]
                Th = pciam(im1,im2h,False)
            transComp[row,col] = (Tv,Th)
    return transComp  

### Translation Optimization ###

##### Compute Image Overlap #####
def computeMle(model, T):
    likelihood = 0 #initialize likelihood
    for t in T:
        normLikelihood = np.exp((-(t-model[1])**2/(2*model[2]**2)))/(model[2]*np.sqrt(2*np.pi)) #likelihood that t belongs to N
        uniformLikelihood = 1/100 #likelihood that t belongs to uniform distribution
        p = model[0]/100
        l = p*uniformLikelihood + (1-p)*normLikelihood
        likelihood = likelihood + np.log(abs(l))
    return likelihood

def percentileResolutionMleHillClimb(model, T):
    done = False
    neighbours = []
    while not done:
        for neighbour in neighbours: #neighbours differ from model by distance of 1 in a single dimension
            temp = model
    return model

def computeImageOverlap(imgGrid:np.ndarray, T:np.ndarray, direction:str):
    H,W = imgGrid[0].shape
    if direction == 'North':
        T = 100*T/H
    else:
        T = 100*T/W

    bestModel = [0,0,0 -np.inf] #model contains [probUniform, mu, sigma, likelihood]

    maxStallCount = 20 #termination condition
    stallCount = 0

    while stallCount < maxStallCount:
        model = [100*np.random.rand(), 100*np.random.rand(), 100*np.random.rand(), np.nan]
        model = percentileResolutionMleHillClimb(model, T)

        if model[3] > bestModel[3]:
            bestModel = model
            stallCount = 0
        else:
            stallCount = stallCount + 1

    overlap = 100 - bestModel[1]
    return overlap

##### Compute Stage Repeatability #####

def filterByOverlapAndCorrelation(imgGrid, T, overlap, pou, direction):
    H,W = imgGrid[0].shape

    if direction == 'North':
        range = (H-(overlap+pou)*H/100, H-(overlap-pou)*H/100)
    else:
        range = (W-(overlap+pou)*W/100, W-(overlap-pou)*W/100)

    validTranslations = []
    for t in T:
        if direction == 'North':
            if t[0][0]:
                if range[0] <= t[0] <= range[1]:
                    validTranslations.append(t)
        else:
                if range[0] <= t[1] <= range[1]:
                    validTranslations.append(t)

    return validTranslations

def filterOutliers(T, direction):
    validTranslations = []
    w = 1.5

    if direction == 'North':
        q2 = np.median(T[:,1])
        q1 = np.median(T[:,1]<q2)
        q3 = np.median(T[:,1]>q2)
        iqd = abs(q3-q1)

        for t in T:
            if (q1-w*iqd) < t[0]:
                return

def estimateEmptyRowColumn(imgGrid, translations, validTranslations, direction):
    H,W = imgGrid.shape
    if direction == "North":
        for row in range(0,H):
            return

def computeRepeatability(imgGrid, T, overlap, pou, direction):
    return

### Image Composition ###

def composeImage(imgGrid, T):
    return