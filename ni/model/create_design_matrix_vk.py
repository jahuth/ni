# create_design_matrix_vk.py
# -*- coding: utf-8 -*-
#
# (c) 2012 IKW Universität Osnabrück
# ported to Python by Robert Costa <rcosta@uni-osnabrueck.de>
#
# This version is based on the original Matlab code written by ??????
# in ???? 20??.
#
from __future__ import division
import numpy as np
from math import factorial

#function [mD covariates mOrder] = jCreateDesignMatrix9(V1,o)
def create_design_matrix_vk(V1, o):
    """Fills free rows in the current design matrix, deduced from size(mD) and
    len(freeCov), corresponding to a single covariate according to the spline
    bases of Volterra kernels. The current kernel(s) and the respective
    numbers of covariates that will be computed for each kernel is deduced
    from masterIndex by determining the position in hypothetical upper
    triangular part of hypercube with number of dimensions corresponding to
    current kernel order. Using only the 'upper triangular part' of the
    hypercube reflects the symmetry of the kernels which stems from the fact
    that only a single spline is used as basis function.

    saves covariate information in cell array 'covariates', format is
    {kernelOrder  relativePositionInKernel  productTermsOfV1}

    Anpassung für Gordon: masterIndex, log, C, mD, freeCov werden berechnet
    statt übergeben.
    """

    masterIndex = 2
    #log = -1
    sV1 = V1.shape

    if sV1[0] > sV1[1]:
        V1 = V1.conj().transpose()
        C = sV1[1]
        NmD = sV1[0]
    else:
        C = sV1[0]
        NmD = sV1[1]
    
    freeCov = 0
    for cDim in range(1, o + 1):
        freeCov += factorial(C + cDim - 1) / (factorial(cDim) * factorial(C - 1))
    
    freeCov = round(freeCov)
    mD = np.zeros((freeCov, NmD))

    #fprintf('creating %i design matrix entries\n', freeCov)
    #if log != -1:
    #    fprintf(log, 'creating %i design matrix entries\n', freeCov)

    # determine model order and corresponding number of covariates
    mOrderCurrent, oCov = detModelOrder(masterIndex, C)

    # determine kernels that can be computed w.r.t. free slots in design matrix
    kernels, mOrder = detKernels(freeCov, masterIndex, oCov, mOrderCurrent, C)
    kS = kernels.shape

    # current index in design matrix
    tmp = mD.shape
    mIndex = tmp[0] - freeCov + 1

    # current position in mOrder kernel w.r.t. masterIndex
    if mOrderCurrent == 0:
        kPos = 1
    else:
        kPos = masterIndex - numCov(C, mOrderCurrent - 1)

    #covariates = cell(tmp[0], 3) # TODO replace cell array with numpy object arrays
    #j = 1
    covariates = []

    # do for as many kernels as freeCov allows
    for i in range(1, kS[0] + 1):
        num = kernels[i - 1, 1] # number of covariates to be computed
        o = kernels[i - 1, 0] # kernel order

        #fprintf('computing %i of %i covariates for kernel of order %i\n', num, round(upTriHalf(C, o)), o)
        #if log != -1:
        #    fprintf(log, 'computing %i of %i covariates for kernel of order %i\n', num, upTriHalf(C, o), o)

        if o == 0:
            mD[mIndex - 1, :] = np.ones((tmp[1], 1))
            mIndex += 1
            #covariates[j - 1, :] = {0, 1, 0}
            #j += 1
            covariates.append([0, 1, 0])
        elif o == 1:
            s = V1.shape
            #mD(mIndex:mIndex+s(1)-1,:) = V1;
            mD[mIndex-1:(mIndex-1) + (s[0]-1) +1, :] = V1
            mIndex = mIndex + s[0]
            for i2 in range(1, s[0]+1):
                #covariates[j-1, :] = {1, i2, i2}
                #j += 1
                covariates.append([1, i2, np.array([i2])]);
        else:
            for k in range(1, num+1):
                #covariates[j-1, 0:1] = {o, kPos} NOTE o, kpos are appended later

                #fprintf('mIndex: %f\n', mIndex)

                mD[mIndex-1, :], prodTerms = computeCovariate(kPos, o, C, V1)

                #fprintf('o %i kPos %i: ', o, kPos)
                #fprintf('%i ', prodTerms)
                #fprintf('\n')

                #covariates[j-1, 2] = prodTerms
                #j += 1
                covariates.append([o, kPos, prodTerms])

                masterIndex += 1
                mIndex += 1
                kPos += 1
            
            kPos = 1

    mD = mD.conj().transpose()
    return mD, covariates, mOrder

#function [mD kernels] = computeCovariate(index,o,C,V1)
def computeCovariate(index, o, C, V1):
    """Computes a row of the designMatrix corresponding to a certain covariate.
    """
    #if log != -1:
    #    fprintf(log, 'computing covariate index %i order %i\n', index, o)
    Cc = C
    mD = 1
    productTerms = np.zeros((o, 1))
    for dim in range(1, o + 1):
        k = 0
        tmp = 0
        tmp2 = 0
        tmpC = Cc
        while tmp < index:
            tmp2 = tmp
            tmp += upTriHalf(tmpC, o - dim)
            tmpC -= 1
            k += 1
        
        productTerms[dim-1] = k
        if dim == o:
            productTerms[dim-1] = k + (C - Cc)
        
        index -= tmp2
        Cc = (Cc - k) + 1

    # productTerms
    kernels = np.zeros((o,))
    for i in range(1, o + 1):
        #mD = mD .* V1[productTerms[i], :]
        mD = mD * V1[int(productTerms[i-1])-1, :] # TODO check if * in Python is .* in Matlab
        kernels[i-1] = productTerms[i-1]

    return mD, kernels

#function num = numCov(C,complexity)
def numCov(C, complexity):
    """Computes number of covariates in a model for which len(complexity)
    symmetric kernels are assumed.
    """
    num = 0
    for cDim in range(0, complexity+1):
        num += factorial(C + cDim - 1) / (factorial(cDim) * factorial(C - 1))
    
    return round(num)

#function num = upTriHalf(C,cDim)
def upTriHalf(C, cDim):
    """Computes number of elements in upper triangular half of hybercube.
    """
    #fprintf('C %i, cDim %i\n', C, cDim)
    num = factorial(C + cDim - 1) / (factorial(cDim) * factorial(C - 1))
    return round(num)

#function [order numCov] = detModelOrder(masterIndex,C)
def detModelOrder(masterIndex, C):
    """Determines model order and corresponding number of covariates.
    """
    numCov = 0
    order = 0
    while numCov < masterIndex:
        numCov += factorial(C + order - 1) / (factorial(order) * factorial(C - 1))
        #fprintf('numCov %f\n', numCov)
        order += 1
    
    order -= 1
    return order, numCov

#function [kernels mOrder] = detKernels(freeCov, masterIndex, oCov, mOrder, C)
def detKernels(freeCov, masterIndex, oCov, mOrder, C):
    """Determines from the number of free slots in Designmatrix len(freeCov)
    and the current masterIndex how many covariates for which Volterra
    coefficient can be computed. Updates model order mOrder.
    """
    #i = 1
    tmp = int(round(min(oCov - masterIndex + 1, freeCov)))
    #kernels = np.array([])
    kernels = []
    while tmp < freeCov:
        #kernels[i-1, :] = [mOrder, tmp]
        kernels.append([mOrder, tmp])
        mOrder += 1
        #i += 1
        freeCov -= tmp
        tmp = int(round(min(freeCov, factorial(C + mOrder - 1) / (factorial(mOrder) * factorial(C - 1)))))
    
    #kernels[i-1, :] = [mOrder, tmp]
    kernels.append([mOrder, tmp])
    return np.array(kernels), mOrder
