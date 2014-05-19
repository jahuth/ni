"""
.. module:: ni.model.designmatrix
   :platform: Unix
   :synopsis: Creates Design Matrices

.. moduleauthor:: Jacob Huth, Robert Costa

"""
from ni.tools.project import *

import numpy as np
import pylab as pl

import scipy.signal
import create_splines as cs
import create_design_matrix_vk as cdm
import json
import ni.tools.pickler

def convolve_spikes(spikes,kernel):
    """
        Convolves a spike train with a kernel by adding the kernel onto every spiketime.
    """
    output = np.zeros((spikes.shape[0]+kernel.shape[0]+1,kernel.shape[1]))
    for i in np.where(spikes)[0]:
        output[(i+1):(i+1+kernel.shape[0]),:] = output[(i+1):(i+1+kernel.shape[0]),:] + kernel
    return output[:len(spikes),:]

def convolve_spikes_2d(spikes_a,spikes_b,kernel_a,kernel_b):
    """
        Does a 2d convolution
    """
    output = np.zeros((spikes_a.shape[0]+kernel_a.shape[0],kernel_a.shape[1]*kernel_b.shape[1]))
    for k_i in range(kernel_a.shape[1]):
        for k_j in range(kernel_b.shape[1]):
            mat = np.zeros((kernel_a.shape[0],kernel_b.shape[0]))
            #for l_1 in range(kernel_a.shape[0]):
            #    for l_2 in range(kernel_b.shape[0]):
            #        mat[l_1,l_2] = kernel_a[l_1,k_i] * kernel_b[l_2,k_j]
            for i in np.where(spikes_a)[0]:
                for j in np.where(spikes_b[(i+1):(i+1+kernel_b.shape[0])])[0]:
                    if j < kernel_b.shape[0]:
                        output[(i+j):(i+j+kernel_a.shape[0]),k_i * kernel_b.shape[1] + k_j] = output[(i+j):(i+j+kernel_a.shape[0]),k_i * kernel_b.shape[1] + k_j] + kernel_a[:,k_i] * kernel_b[j,k_j] #mat[:,j-i]
    return output[:spikes_a.shape[0],:]

class HistoryDesignMatrix:
    """
        Internal helper class
    """
    def __init__(self,spikes,history_length=100,knot_number=5,order_flag=1,kernel=False):
        if type(kernel) == bool:
            self.history_kernel = cs.create_splines_logspace(history_length, knot_number, 0)
        else:
            self.history_kernel = kernel
        self.matrix = "Not initialized!"
        try:
            if order_flag == 1:
                #print spikes.shape
                #print self.history_kernel.shape
                matrix = convolve_spikes(spikes, self.history_kernel)
                #print matrix.shape
                self.matrix = matrix[:len(spikes)]
                #print self.matrix.shape
            else:
                self.covariate_matrix, self.covariates, self.morder = cdm.create_design_matrix_vk(self.history_kernel, order_flag)
                #print spikes.shape
                #print self.covariate_matrix.shape
                matrix = scipy.signal.convolve2d(spikes, self.covariate_matrix)
                #print matrix.shape
                self.matrix = matrix[:len(spikes)]
                #print self.matrix.shape
        except:
            err(spikes)
            err(spikes.shape)
            err(self.history_kernel)
            report()
            raise

class Component(ni.tools.pickler.Picklable):
    """
        Design Matrix Component

        header: name of the kernel component
        kernel: kernel that will be tiled to fill the design matrix
    """
    def __init__(self,header="Undefined",kernel=0):
        self.kernel = kernel
        self.header = header
        try:
            self.width = self.kernel.shape[1]
        except:
            self.width = 1
        if type(header) == str or type(header) == dict:
            self.loads(header)
        if type(self.kernel) == str:
            self.kernel = self.kernel.replace("np.ndarray","np.array")
    def getSplines(self,data=[]):
        if type(self.kernel) == str:
            return eval(self.kernel)
        else:
            return self.kernel
    def __str__(self):
        if type(self.kernel) == str:
            s = "{\"header\": \""+self.header+"\", \"type\": \"simple\", \"kernel\": \"" + self.kernel + "\"}"
        else:
            s = "{\"header\": \""+self.header+"\", \"type\": \"simple\", \"kernel\": [[" + "],[".join([",".join([str(e) for e in k]) for k in self.kernel]) + "]]}"
        return s

class RateComponent(Component):
    """
        Rate Design Matrix Component

        header: name of the kernel component
        knots: Number of knots
        length: length of the component. Will be multiplied

        kernel: use this kernel instead of a newly created one
    """
    def __init__(self,header="rate",knots=10,length=1000,kernel =False):
        self.header = header
        self.knots = knots
        self.length = length
        if type(kernel) == bool:
            self.kernel = cs.create_splines_linspace(self.length, self.knots, 0)
        else:
            self.kernel = kernel
        if type(self.kernel) == str:
            self.width = eval(self.kernel).shape[1]
        else:
            self.width = self.kernel.shape[1]
        if type(header) == str or type(header) == dict:
            self.loads(header)
        if type(self.kernel) == str:
            self.kernel = self.kernel.replace("np.ndarray","np.array")
    def getSplines(self,data=[]):
        if type(self.kernel) == str:
            return eval(self.kernel)
        else:
            return self.kernel
    def __str__(self):
        if type(self.kernel) == str:
            s = "{\"header\": \""+self.header+"\", \"type\": \"rate\", \"kernel\": \"" + self.kernel + "\"}"
        else:
            s = "{\"header\": \""+self.header+"\", \"type\": \"rate\", \"kernel\": [[" + "],[".join([",".join([str(e) for e in k]) for k in self.kernel]) + "]]}"
        return s

class AdaptiveRateComponent(Component):
    """
        Rate Design Matrix Component

        header: name of the kernel component
        rate: a rate function that determines
        exponent: the rate function will be taken to thins power to have a higher selctivity for high firing rates
        knots: Number of knots
        length: length of the component. Will be multiplied

        kernel: use this kernel instead of a newly created one
    """
    def __init__(self,header="rate",rate=False,exponent=2,knots=10,length=1000,kernel =False):
        self.rate = rate
        self.exponent = exponent
        self.knots = knots
        if type(rate) == bool:
            self.knot_points = np.linspace(0,length,knots-1)
        else:
            self.rate_derivative = np.abs(np.append([0], self.rate[:-1] - self.rate[1:]))
            rate_cumsum = (self.rate_derivative**self.exponent).cumsum()
            self.knot_points = [np.sum(rate_cumsum < n) for n in np.linspace(0,rate_cumsum.max(),knots-1)]
        self.header = header
        self.length = length
        if type(kernel) == bool:
            self.kernel = cs.create_splines(self.length, self.knots-1,0, lambda l,n: self.knot_points)
        else:
            self.kernel = kernel
        if type(self.kernel) == str:
            self.width = eval(self.kernel).shape[1]
        else:
            self.width = self.kernel.shape[1]
        if type(header) == str or type(header) == dict:
            self.loads(header)
        if type(self.kernel) == str:
            self.kernel = self.kernel.replace("np.ndarray","np.array")
    def getSplines(self,data=[]):
        if type(self.kernel) == str:
            return eval(self.kernel)
        else:
            return self.kernel
    def __str__(self):
        if type(self.kernel) == str:
            s = "{\"header\": \""+self.header+"\", \"type\": \"rate\", \"kernel\": \"" + self.kernel + "\"}"
        else:
            s = "{\"header\": \""+self.header+"\", \"type\": \"rate\", \"kernel\": [[" + "],[".join([",".join([str(e) for e in k]) for k in self.kernel]) + "]]}"
        return s


class HistoryComponent(Component):
    """
        History Design Matrix Component

        Will be convolved with spikes before fitting

        header: name of the kernel component
        channel: which channel the kernel should be convolved with (default 0)
        history_length: length of the kernel 
        knot_number: number of knots (will be logspaced)
        order_flag: default 0 (no higher order interactions)

        kernel: use this kernel instead of a newly created one

    Atm only order 1 interactions

    """
    def __init__(self,header='autohistory',channel=0,history_length=100,knot_number=4, order_flag=1,kernel=False,delete_last_spline=True):
        if type(kernel) == bool:
            self.kernel = "cs.create_splines_logspace("+str(history_length)+", "+str(knot_number)+", "+str(delete_last_spline)+")"
        else:
            self.kernel = kernel
        self.header = header
        self.knot_number = knot_number
        self.history_length = history_length
        self.channel = channel
        if type(self.kernel) == str:
            self.width = eval(self.kernel).shape[1]
        else:
            self.width = self.kernel.shape[1]
        if type(header) == str or type(header) == dict:
            self.loads(header)
        if type(self.kernel) == str:
            self.kernel = self.kernel.replace("np.ndarray","np.array")
    def getSplines(self,channels=[]):
        if channels == []:
            if type(self.kernel) == str:
                return eval(self.kernel)
            else:
                return self.kernel
        else:
            #print self.channel,"/",len(channels)
            if type(self.kernel) == str:
                kernel = eval(self.kernel)
            else:
                kernel = self.kernel
            if len(channels) > self.channel:
                matrix = convolve_spikes(channels[self.channel], kernel)
            else:
                raise Exception(str(self.channel) + " is greater than " + str(len(channels)) + "!")
            return matrix
    def __str__(self):
        if type(self.kernel) == str:
            s = "{\"header\": \""+self.header+"\", \"type\": \"history\", \"channel\": "+str(self.channel)+", \"kernel\": \"" + self.kernel + "\"}"
        else:
            s = "{\"header\": \""+self.header+"\", \"type\": \"history\", \"channel\": "+str(self.channel)+", \"kernel\": [[" + "],[".join([",".join([str(e) for e in k]) for k in self.kernel]) + "]]}"
        return s

class SecondOrderHistoryComponent(Component):
    """
        History Design Matrix Component with Second Order Kernels

        Will be convolved with spikes before fitting

        header: name of the kernel component
        channel: which channel the kernel should be convolved with (default 0)
        history_length: length of the kernel 
        knot_number: number of knots (will be logspaced)
        order_flag: default 0 (no higher order interactions)

        kernel: use this kernel instead of a newly created one

    Atm only order 1 interactions

    """
    def __init__(self,header='autohistory',channel_1=0,channel_2=0,history_length=100,knot_number=4, order_flag=1,kernel_1=False,kernel_2=False,delete_last_spline=True):
        if type(kernel_1) == bool:
            self.kernel_1 = "cs.create_splines_logspace("+str(history_length)+", "+str(knot_number)+", "+str(delete_last_spline)+")"
        else:
            self.kernel_1 = kernel_1
        if type(kernel_2) == bool:
            self.kernel_2 = "cs.create_splines_logspace("+str(history_length)+", "+str(knot_number)+", "+str(delete_last_spline)+")"
        else:
            self.kernel_2 = kernel_2
        self.header = header
        self.knot_number = knot_number
        self.history_length = history_length
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        if type(self.kernel_1) == str:
            if type(self.kernel_2) == str:
                self.width = eval(self.kernel_1).shape[1] * eval(self.kernel_2).shape[1]
            else:
                self.width = eval(self.kernel_1).shape[1] * self.kernel_2.shape[1]
        else:
            if type(self.kernel_2) == str:
                self.width = self.kernel_1.shape[1] * eval(self.kernel_2).shape[1]
            else:
                self.width = self.kernel_1.shape[1] * self.kernel_2.shape[1]
        if type(header) == str or type(header) == dict:
            self.loads(header)
        if type(self.kernel_1) == str:
            self.kernel_1 = self.kernel_1.replace("np.ndarray","np.array")
        if type(self.kernel_2) == str:
            self.kernel_2 = self.kernel_2.replace("np.ndarray","np.array")
    def getSplines(self,channels=[],get_1d_splines=False,beta=False):
        if type(self.kernel_1) == str:
            kernel_1 = eval(self.kernel_1)
        else:
            kernel_1 = self.kernel_1
        if type(self.kernel_2) == str:
            kernel_2 = eval(self.kernel_2)
        else:
            kernel_2 = self.kernel_2
        if get_1d_splines:
            return np.concatenate([kernel_1,kernel_2],axis=1)
        if channels == []:
            mats = np.zeros((kernel_1.shape[1]*kernel_2.shape[1],kernel_1.shape[0],kernel_2.shape[0]))
            for i in range(kernel_1.shape[1]):
                for j in range(kernel_2.shape[1]):
                    mat = np.zeros((kernel_1.shape[0],kernel_2.shape[0]))
                    for l_1 in range(kernel_1.shape[0]):
                        for l_2 in range(kernel_2.shape[0]):
                            mat[l_1,l_2] = kernel_1[l_1,i] * kernel_2[l_2,j]
                    mats[i*kernel_2.shape[1]+j,:,:] = mat
            return mats
        else:
            if len(channels) > self.channel_1 and len(channels) > self.channel_2:
                matrix = np.zeros((channels[0].shape[0],kernel_1.shape[1]*kernel_2.shape[1]))
                splines_1 = convolve_spikes(channels[self.channel_1], kernel_1)
                splines_2 = convolve_spikes(channels[self.channel_2], kernel_2)
                for i_1 in range(splines_1.shape[1]):
                    for i_2 in range(i_1,splines_2.shape[1]):
                        matrix[:,i_1*kernel_2.shape[1] + i_2] = splines_1[:,i_1]*splines_2[:,i_2]
                #matrix = convolve_spikes_2d(channels[self.channel_1],channels[self.channel_2], kernel_1, kernel_2)
            else:
                raise Exception(str(self.channel_1) + " or " + str(self.channel_2) + " is greater than " + str(len(channels)) + "!")
            if type(beta) == bool:
                return matrix
            else:
                return np.sum(beta*matrix,1)
    def __str__(self):
        if type(self.kernel_1) == str and type(self.kernel_2) == str:
            s = "{\"header\": \""+self.header+"\", \"type\": \"2ndOrderHistory\", \"channel_1\": "+str(self.channel_1)+", \"channel_2\": "+str(self.channel_2)+", \"kernel_1\": \"" + self.kernel_1 + "\", \"kernel_2\": \"" + self.kernel_2 + "\"}"
        else:
            s = "{\"header\": \""+self.header+"\", \"type\": \"2ndOrderHistory\", \"channel_1\": "+str(self.channel_1)+", \"channel_2\": "+str(self.channel_2)+", \"kernel_1\": [[" + "],[".join([",".join([str(e) for e in k]) for k in self.kernel_1]) + "]], \"kernel_2\": [[" + "],[".join([",".join([str(e) for e in k]) for k in self.kernel_2]) + "]]}"
        return s

class DesignMatrixTemplate(ni.tools.pickler.Picklable):
    """
        Most important class for Design Matrices

        Uses components that are then combined into an actual design matrix:

            >>> DesignMatrixTemplate(data.nr_trials * data.time_bins)
            >>> kernel = cs.create_splines_logspace(self.configuration.history_length, self.configuration.knot_number, 0)
            >>> design_template.add(designmatrix.HistoryComponent('autohistory', kernel=kernel))
            >>> design_template.add(designmatrix.HistoryComponent('crosshistory'+str(2), channel=2, kernel = kernel))
            >>> design_template.add(designmatrix.RateComponent('rate',self.configuration.knots_rate,trial_length))
            >>> design_template.add(designmatrix.Component('constant',np.ones((1,1))))
            >>> design_template.combine(data)

    """
    def __init__(self,length,trial_length=0):
        """
            length: overall length
            trial_length: length of one trial
        """
        self.component_width = 0
        self.components = []
        self.length = length
        self.header = []
        self._header = []
        self.mask = []
        self.trial_length = trial_length
        if type(length) == str or type(length) == dict:
            self.length = 0
            self.loads(length)
    def add(self, component):
        """ Adds a component """
        self.components.append(component)
        self.header.append(component.header)
        self.component_width = self.component_width + component.width
        self.mask.extend([True]*component.width)
    def combine(self,data):
        """
            combines the design matrix template into an actual design matrix.

            It needs an ni.Data instance for this to place the history splines.

        """
        channels = []
        #if type(data.data) == pandas.core.frame.DataFrame:
        #    spike_train_all_trial_ = np.array(data.data.stack())
        #    spike_train_all_trial = np.reshape(spike_train_all_trial_,(spike_train_all_trial_.shape[0],1))
        #else:
        #    spike_train_all_trial_ = np.array(data.data)
        #    spike_train_all_trial = np.reshape(spike_train_all_trial_,(spike_train_all_trial_.shape[0],1))
        for c in range(data.nr_cells):
            channels.append(data.cell(c).getFlattend())

        d = DesignMatrix(channels[0].shape[0], self.component_width)
        d.setMask(self.mask)
        for c in self.components:
            d.add(c.getSplines(channels),c.header)
        #if self.mask.shape[0] == d.matrix.shape[1] + 1:
        #    d.matrix = d.matrix[:,~self.mask[:-1]]
        self._header = d.header
        #print "--"
        #print d.matrix.shapeload
        return d.matrix
    def get(self,filt):
        """ returns the splines of the first component, the header of which matches `filt` """
        indx = [i for i in xrange(len(self.header)) if filt == self.header[i]]
        return self.components[indx[0]].getSplines()
    def get_components(self,filt):
        """ returns all component, the header of which matches `filt` """
        comps = [self.components[i] for i in xrange(len(self.header)) if filt == self.header[i]]
        return comps
    def getIndex(self,filt):
        """ returns the index (design matrix rows) of the component matching `filt`"""
        indx = [i for i in xrange(len(self._header)) if filt == self._header[i]]
        return indx
    def getMask(self,filt):
        """ reurns the mask of the component matching `filt` """
        indx = [self.mask[i] for i in xrange(len(self._header)) if filt == self._header[i]]
        return indx
    def setMask(self, mask):
        """ sets a mask (list of boolean values), which design matrix rows to use. Default is all True. If `mask` is shorter` than the desgin matrix, all following values are assumed True. """
        self.mask = mask
    def __str__(self):
        s = "[" + ",\n".join([c.__str__() for c in self.components]) + "]"
        return s
        



class DesignMatrix(ni.tools.pickler.Picklable):
    """
        Use :class:`DesignMatrixTemplate` to create a design matrix.

        This class computes an actual matrix, where :class:`DesignMatrixTemplate` can be saved before the matrix is instanciated.
    """
    def __init__(self,length,width=1):
        try:
            self._matrix = np.zeros((length,width))#np.matrix([0]*length).transpose()
            self.matrix = np.zeros((length,width))#np.matrix([0]*length).transpose()
        except:
            err('length :' + str(length) + ' width: ' + str(width))
            report()
            raise
        self.mask = np.array([True]*width)
        self._header = []
        self.header = []
        self.length = length
        self.width_used = 0
    def add(self, splines, header):
        l = splines.shape[1]
        if self.width_used + l > self._matrix.shape[1]:
            self._matrix = np.concatenate((self._matrix,np.zeros((self.length,l-(self._matrix.shape[1]-self.width_used)))),1)
            #raise Warning('Matrix not big enough!', 'Initiate with wider matrix to save time.')
            err("Please initiate with wider matrix to save time.")
        mult = int(np.ceil(float(self.length) / splines.shape[0]))
        splines_full_length = splines
        if mult > 1:
            #for i in xrange(mult-1):
            #    splines_full_length = np.concatenate((splines_full_length,splines),0)
            splines_full_length = np.tile(splines,(mult,1))
        try:
            self._matrix[:,self.width_used:(self.width_used+l)] = splines_full_length[:self.length]
        except:
            err('self._matrix.shape' + str(self._matrix.shape))
            err('splines_full_length.shape' + str(splines_full_length.shape))
            err('self.width_used' + str(self.width_used))
            err('l' + str(l))
            err('self.length' + str(self.length))
            report()
            raise
        #else:
        #    self.matrix = np.concatenate((self.matrix, splines), 1)
        if isinstance(header, str):
            self._header.extend([header] * l)
        else:
            if l == len(header):
                self._header.extend(header)
            else:
                self._header.extend(['Undefined'] * l)
        self.width_used = self.width_used + l
        if self.mask.shape[0] < self.width_used:
            while self.mask.shape[0] < self.width_used:
                self.mask = np.concatenate([self.mask,[True]],0)
        self.matrix = self._matrix[:,self.mask[:self.width_used]]
        self.header = [self._header[i] for i in np.where(self.mask[:self.width_used])[0]]
    def setMask(self, mask):
        if mask == np.ndarray([True]) or mask == np.array([6.89897521e-310]):
            mask = np.array([True])
        if type(mask) == bool:
            #dbg("Tried to set mask to " + str(mask))
            mask = np.array([mask])
        if type(mask) == list:
            mask = np.array(mask)
        if mask.shape[0] < self.width_used:
            while mask.shape[0] < self.width_used:
                mask = np.concatenate([mask,[True]],0)
        self.mask = mask
        self.matrix = self._matrix[:,self.mask[:self.width_used]]
        self.header = [self._header[i] for i in np.where(self.mask[:self.width_used])[0]]
    def clip(self):
        self.matrix = self.matrix[:,:self.width_used]
    def addLinSpline(self, knots, header, length=0):
        if length == 0:
            self.add(cs.create_splines_linspace(self.length-1, knots, 0), header)
        else:
            self.add(cs.create_splines_linspace(length-1, knots, 0), header)
    def addLogSpline(self, knots, header, length=0):
        if length == 0:
            self.add(cs.create_splines_logspace(self.length-1, knots, 0), header)
        else:
            self.add(cs.create_splines_logspace(length-1, knots, 0), header)
    def plot(self,filt=""):
        if filt=="":
            pl.plot(self.matrix)
        else:
            pl.plot(self.get(filt))
    def get(self,filt):
        return self.matrix[:,self.getIndex(filt)]
    def getIndex(self,filt):
        indx = [i for i in xrange(len(self.header)) if filt == self.header[i]]
        return indx
    def getMask(self,filt):
        indx = [self.mask[i] for i in xrange(len(self._header)) if filt == self._header[i]]
        return indx
            

