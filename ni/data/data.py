"""
.. module:: ni.data.data
   :platform: Unix
   :synopsis: Storing Point Process Data

.. moduleauthor:: Jacob Huth

.. todo::
		Use different internal representations, depending on use. Ie. Spike times vs. binary array

.. todo::
		Lazy loading and prevention from data duplicates where unnecessary. See also: `indexing view versus copy <http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy>`_

Storing Spike Data in Python with Pandas
--------------------------------------------------------

The `pandas package <http://pandas.pydata.org/>`_ allows for easy storage of large data objects in python. The structure that is used by this toolbox is the pandas :py:class:`pandas.MultiIndexedFrame` which is a :py:class:`pandas.DataFrame` / `pandas.DataFrame <http://pandas.pydata.org/pandas-docs/dev/dsintro.html#dataframe>`_ with an Index that has multiple levels.

The index contains at least the levels ``'Cell'``, ``'Trial'`` and ``'Condition'``. Additional Indizex can be used (eg. ``'Bootstrap Sample'`` for Bootstrap Samples), but keep in mind that when fitting a model only ``'Cell'`` and ``'Trial'`` should remain, all other dimensions will be collapsed as more sets of Trials which may be indistinguishable after the fit.


===========  =====  ======= ===================================
Condition    Cell   Trial   *t* (Timeseries of specific trial)
===========  =====  ======= ===================================
0            0      0       0,0,0,0,1,0,0,0,0,1,0...      
0            0      1       0,0,0,1,0,0,0,0,1,0,0...
0            0      2       0,0,1,0,1,0,0,1,0,1,0...
0            1      0       0,0,0,1,0,0,0,0,0,0,0...
0            1      1       0,0,0,0,0,1,0,0,0,1,0...
...          ...    ...     ...
1            0      0       0,0,1,0,0,0,0,0,0,0,1...
1            0      1       0,0,0,0,0,1,0,1,0,0,0...
...          ...    ...     ...
===========  =====  ======= ===================================


To put your own data into a :py:class:`pandas.DataFrame`, so it can be used by the models in this toolbox create a MultiIndex for example like this::

	import ni
	import pandas as pd
	d = []
	tuples = []
	for con in range(nr_conditions):
		for t in range(nr_trials):
			for c in range(nr_cells):
					spikes = list(ni.model.pointprocess.getBinary(Spike_times_STC.all_SUA[0][0].spike_times[con,t,c].flatten()*1000))
					if spikes != []:
						d.append(spikes)
						tuples.append((con,t,c))
	index = pd.MultiIndex.from_tuples(tuples, names=['Condition','Trial','Cell'])
	data = ni.data.data.Data(pd.DataFrame(d, index = index))

If you only have one trial if several cells or one cell with a few trials, it can be indexed like this:

	from ni.data.data import Data
	import pandas as pd
	
	index = pd.MultiIndex.from_tuples([(0,0,i) for i in range(len(d))], names=['Condition','Cell','Trial'])
	data = Data(pd.DataFrame(d, index = index))

To use the data you can use :py:func:`ni.data.data.Data.filter`::

	only_first_trials = data.filter(0, level='Trial')

	# filter returns a copy of the Data object

	only_the_first_trial = data.filter(0, level='Trial').filter(0, level='Cell').filter(0, level='Condition') 

	only_the_first_trial = data.condition(0).cell(0).trial(0) # condition(), cell() and trial() are shortcuts to filter that set *level* accordingly

	only_some_trials  = data.trial(range(3,15))
	# using slices, ranges or boolean indexing causes the DataFrame to be indexed again from 0 to N, in this case 0:11

Also ix and xs pandas operations can be useful::

	plot(data.data.ix[(0,0,0):(0,3,-1)].transpose().cumsum())
	plot(data.data.xs(0,level='Condition').xs(0,level='Cell').ix[:5].transpose().cumsum())


"""
from ni.tools.project import *
#import ni.model.ip as _ip
#reload(ni.model.ip)
import pandas
import numpy as np
import scipy
import os
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as pl
from ni.tools.html_view import View

MILLISECOND_RESOLUTION = 1000

def saveToFile(path,o):
	""" saves a DataFrame-like to a file """
	return pandas.DataFrame(o).to_pickle(path)

def loadFromFile(path):
	""" loads a pandas DataFrame from a file """
	return pandas.read_pickle(path)


def merge(datas,dim,keys = False):
	"""

	merges multiple Data instances into one::

		data = ni.data.data.merge([ni.data.data.Date(f) for f in ['data1.pkl','data2.pkl','data3.pkl']], dim = 'Data File')

	"""
	cells = 0
	trials = 0
	conditions = 0
	time_bins = 0
	if keys == False:
		keys = range(len(datas))
	df = pandas.concat([m.data for m in datas], keys = keys, names = [dim])
	return Data(df)


def matrix_to_dataframe(matrix, dimensions):
	""" conerts a trial x cells matrix into a DataFrame """
	tup = []
	for t1 in range(self.nr_trials):
		for t2 in range(self.nr_cells):
			tup.append((0,t1,t2))
	df = pandas.DataFrame(matrix)
	df.index = pandas.MultiIndex.from_tuples(tup, names=['Condition','Trial','Cell'])
	return df

class Data:
	"""
		Spike data container

		Contains a panda Data Frame with MultiIndex.
		Can save to and load from files.

		The Index contains at least Trial, Cell and Condition and can be extended.


	"""
	def __init__(self,matrix, dimensions = [], other_spikes = False, key_index="i", resolution=MILLISECOND_RESOLUTION):
		"""
			Can be initialized with a DataFrame, filename or Data instance

			**resolution**

					resolution in bins per second
		"""
		self.desc = ""
		self.type = "binary_array"
		self.nr_cells = 1
		self.nr_trials = 1
		self.nr_conditions = 1
		self.time_bins = 1
		self.other_spikes = other_spikes
		self.data = pandas.DataFrame(np.zeros((1,1)))
		self.resolution = resolution
		if type(matrix) == str:
			matrix = pandas.read_pickle(matrix)
		if isinstance(matrix, Data):
			self = copy(matrix)
			self.__class__ = Data
			if not type(other_spikes) == bool:
				self.other_spikes = Data(other_spikes)
		if type(matrix) == np.ndarray:
			if dimensions != []:
				if len(matrix.shape) == 2:
					self.data = pandas.DataFrame(matrix)
					if 'Trial' in dimensions:
						self.nr_trials = matrix.shape[dimensions.index('Trial')]
					if 'Cell' in dimensions:
						self.nr_trials = matrix.shape[dimensions.index('Cell')]
					if 'Condition' in dimensions:
						self.nr_conditions = matrix.shape[dimensions.index('Condition')]
					if 'Time' in dimensions:
						self.trial_length = matrix.shape[dimensions.index('Time')]
					tup = []
					for t1 in range(self.nr_trials):
						for t2 in range(self.nr_cells):
							tup.append((0,t1,t2))
					self.data.index = pandas.MultiIndex.from_tuples(tup, names=['Condition','Trial','Cell'])
				elif len(matrix.shape) == 3:
					raise Exception("Not implemented.")
			elif len(matrix.shape) == 3:
				self.data = pandas.DataFrame(matrix.reshape((matrix.shape[0]*matrix.shape[1],matrix.shape[2])))
				tup = []
				self.nr_trials = matrix.shape[0]
				self.nr_cells = matrix.shape[1]
				self.time_bins = matrix.shape[2]
				for t1 in range(self.nr_trials):
					for t2 in range(self.nr_cells):
						tup.append((0,t1,t2))
				self.data.index = pandas.MultiIndex.from_tuples(tup, names=['Condition','Trial','Cell'])
				self.trial_length = int(self.time_bins/self.nr_trials)
			elif len(matrix.shape) == 2:
				self.data = pandas.DataFrame(matrix)
				self.nr_trials = matrix.shape[0]
				self.nr_cells = 1
				self.time_bins = matrix.shape[1]
				tup=[]
				for t1 in range(self.nr_trials):
					tup.append((0,t1,0))
				self.data.index = pandas.MultiIndex.from_tuples(tup, names=['Condition','Trial','Cell'])
				self.trial_length = int(self.time_bins/self.nr_trials)
			elif len(matrix.shape) == 1:
				self.data = pandas.DataFrame(matrix)
				self.nr_trials = 1
				self.nr_cells = 1
				self.time_bins = matrix.shape[0]
				self.trial_length = int(self.time_bins/self.nr_trials)
			else:
				raise Exception("Matrix has incompatible dimensions. Consider using a pandas.DataFrame.")
		elif type(matrix) == pandas.core.frame.DataFrame:
			self.data = matrix.fillna(value=0)
			ind = dict(zip(*[matrix.index.names, range(len(matrix.index.names))]))
			self.nr_conditions = 1
			self.nr_trials = 1
			self.nr_cells = 1
			if type(matrix.index) == pandas.MultiIndex:
				if 'Condition' in ind:
					self.nr_conditions = matrix.index.levshape[ind['Condition']]
				if 'Trial' in ind:
					self.nr_trials = matrix.index.levshape[ind['Trial']]
				if 'Cell' in ind:
					self.nr_cells = matrix.index.levshape[ind['Cell']]
				self.time_bins = matrix.shape[1]
				self.trial_length = self.time_bins#int(self.time_bins/self.nr_trials)
			else:
				if 'Condition' in ind:
					self.nr_conditions = matrix.index.shape[ind['Condition']]
				if 'Trial' in ind:
					self.nr_trials = matrix.index.shape[ind['Trial']]
				if 'Cell' in ind:
					self.nr_cells = matrix.index.shape[ind['Cell']]
				self.time_bins = matrix.shape[1]
				self.trial_length = self.time_bins#int(self.time_bins/self.nr_trials)
		elif type(matrix) == pandas.core.series.Series:
			self.data = matrix.fillna(value=0)
			self.nr_conditions = 1
			self.nr_trials = 1
			self.nr_cells = 1
			self.time_bins = matrix.shape[0]
			self.trial_length = self.time_bins
		elif type(matrix) == list:
			if isinstance(matrix[0], Data):
				self.data = pandas.concat([m.data for m in matrix], keys = range(len(matrix)), names = [key_index])
			if type(self.data.index) == pandas.MultiIndex:
				ind = dict(zip(*[self.data.index.names, range(len(self.data.index.names))]))
				if 'Condition' in ind:
					self.nr_conditions = self.data.index.levshape[ind['Condition']]
				if 'Trial' in ind:
					self.nr_trials = self.data.index.levshape[ind['Trial']]
				if 'Cell' in ind:
					self.nr_cells = self.data.index.levshape[ind['Cell']]
				self.time_bins = self.data.shape[1]
				self.trial_length = self.time_bins#int(self.time_bins/self.nr_trials)
		else:
			self.data = np.array([0])
			self.nr_trials = 0
			self.nr_cells = 0
			self.time_bins = 0
			self.trial_length = self.time_bins
		if not type(other_spikes) == bool and not isinstance(other_spikes, Data):
			self.other_spikes = Data(other_spikes)
		else:
			self.other_spikes = other_spikes
	def cell(self,cells=False):
		"""filters for an array of cells -> see :py:func:`ni.data.data.Data.filter`"""
		if (cells ==[0] or cells == 0) and self.nr_cells == 1:
			return self
		return self.filter(cells,'Cell')
	def condition(self,conditions=False):
		"""filters for an array of conditions -> see :py:func:`ni.data.data.Data.filter`"""
		return self.filter(conditions,'Condition')
	def trial(self,trials=False):
		"""filters for an array of trials -> see :py:func:`ni.data.data.Data.filter`"""
		return self.filter(trials,'Trial')
	def filter(self,array=False,level='Cell'):
		"""filters for arbitrary index levels
			`array` a number, list or numpy array of indizes that are to be filtered
			`level` the level of index that is to be filtered. Default: 'Cell' 
		"""
		if type(array) == bool:
			#dbg("called for nothing")
			return self
		else:
			if type(array) == int or type(array) == float:
				array = [int(array)]
			if type(self.data) == pandas.DataFrame:
				if type(self.data.index) == pandas.Int64Index:
					data = pandas.concat([self.data.ix[i] for i in array],keys=range(len(array)),names=[level])
					return Data(data,self.other_spikes)
				elif type(self.data.index) == pandas.MultiIndex:
					data = pandas.concat([self.data.xs(i,level=level) for i in array],keys=range(len(array)),names=[level])
					data.index = pandas.MultiIndex.from_tuples(data.index.tolist(), names=data.index.names)
					return Data(data,self.other_spikes)
				else:
					raise Exception("Unrecognized DataFrame index")
			else:
				raise Exception("Unrecognized Data")
	def firing_rate(self,smooth_width=0,trials=False):
		"""
			computes the firing rate of the data for each cell separately.
		"""
		if type(trials) is bool:
			channels = []
			for cell in range(self.nr_cells):
				n = self.cell(cell)
				d = n.data.sum(0)
				channels.append(scipy.ndimage.gaussian_filter(d,smooth_width))
			return channels
		else:
			channels = []
			for cell in range(self.nr_cells):
				n = self.cell(cell).trial(trials)
				d = n.data.sum(0)
				channels.append(scipy.ndimage.gaussian_filter(d,smooth_width))
			return channels
	def interspike_intervals(self,smooth_width=0,trials=False):
		"""
			computes inter spike intervalls in the data for each cell separately.
		"""
		if type(trials) is bool:
			channels = []
			for cell in range(self.nr_cells):
				n = self.cell(cell)
				d = n.data.sum(0)
				channels.append(scipy.ndimage.gaussian_filter(d,smooth_width))
			return channels
		else:
			channels = []
			for cell in range(self.nr_cells):
				n = self.cell(cell).trial(trials)
				d = n.data.sum(0)
				channels.append(scipy.ndimage.gaussian_filter(d,smooth_width))
			return channels
	def as_series(self):
		"""
		Returns one timeseries, collapsing all indizes.

		The output has dimensions of (N,1) with N being length of one trial x nr_trials x nr_cells x nr_conditions (x additonal indices).

		If cells, conditions or trials should be separated, use :func:`as_list_of_series` instead.
		"""
		if type(self.data) is pandas.core.frame.DataFrame:
			data = self.data.stack()
			data = data.reshape((data.shape[0],1))
			return data
		else:
			data = self.data.reshape((np.prod(self.data.shape),1))
			return data
	def as_list_of_series(self,list_conditions=True,list_cells=True,list_trials=False,list_additional_indizes=True):
		"""
		Returns one timeseries, collapsing only certain indizes (on default only trials). All non collapsed indizes
		"""
		if list_conditions and self.nr_conditions > 1:
			return [self.condition(c).as_list_of_series() for c in range(self.nr_conditions)]
		if list_cells and self.nr_cells > 1:
			return [self.cell(c).as_list_of_series() for c in range(self.nr_cells)]
		if list_trials and self.nr_trials > 1:
			return [self.trial(t).as_list_of_series() for t in range(self.nr_trials)]

		if list_additional_indizes and type(self.data.index) == pandas.MultiIndex:
			ind = dict(zip(*[self.data.index.names, range(len(self.data.index.names))]))
			for n in ind:
				if not n == 'Trial' and not n == 'Condition' and not n == 'Cell':
					if self.data.index.levshape[ind[n]] > 1:
						return [self.filter(a, level=n).as_list_of_series() for a in range(self.data.index.levshape[ind[n]])]
		return self.as_series() 
	def getFlattend(self,all_in_one=True,trials=False):
		"""
		.. deprecated:: 0.1
			Use :func:`as_list_of_series` and :func:`as_series` instead
		Returns one timeseries for all trials.

		The *all_in_one* flag determines whether ``'Cell'`` and ``'Condition'`` should also be collapsed. If set to *False* and the number of Conditions and/or Cells is greater than 1, a list of timeseries will be returned. If both are greater than 1, then a list containing for each condition a list with a time series for each cell.

		"""
		#print "getFlattend"
		if not all_in_one:
			if self.nr_conditions > 1:
				#print "collapsing conditions"
				return [self.condition(c).getFlattend() for c in range(self.nr_conditions)]
			if self.nr_cells > 1:
				#print "collapsing cells"
				return [self.cell(c).getFlattend() for c in range(self.nr_cells)]
		spike_train_all_trial = []
		if not type(trials) is list:
			if type(trials) is bool:
				#print type(self.data.index)
				if type(self.data) is pandas.core.frame.DataFrame:
					data = self.data.stack()
					data = data.reshape((data.shape[0],1))
					#print data.shape
					return data
				else:
					data = self.data.reshape((np.prod(self.data.shape),1))
					#print data.shape
					return data
			else:
				if type(trials) is int and trials <= self.nr_trials:
					trials = range(trials)
					data = self.trial(trials).data.stack()
					data = data.reshape((data.shape[0],1))
					return data
				else: # Whatever data is now
					raise Exception("Unrecognized trial indices")
					trials = range(self.nr_trials)
					data = self.trial(trials).data.stack()
					data = data.reshape((data.shape[0],1))
					return data
		for trial in trials:
			spike_train = []
			if type(self.data.index) is pandas.core.index.Int64Index:
				spike_train = self.data.ix[trial]
			else:
				spike_train = self.data.xs(trial,level='Trial')
			spikes = np.where(spike_train)
			spike_train_all_trial.extend(spike_train)
		spike_train_all_trial_ = np.array(spike_train_all_trial)
		spike_train_all_trial = np.reshape(spike_train_all_trial_,(spike_train_all_trial_.shape[0],1))
		return spike_train_all_trial
	def shape(self,level):
		"""
			Returns the shape of the sepcified level::

				>>> data.shape('Trial')
					100

				>>> data.shape('Cell') == data.nr_cells
					True

		"""
		ind = dict(zip(*[self.data.index.names, range(len(self.data.index.names))]))
		if level in ind:
			return self.data.index.levshape[ind[level]]
	def __str__(self):
		"""
			Returns a string representation of the Data Object.

		"""
		s =  "Spike data: " + str(self.nr_conditions) + " Condition(s) " + str(self.nr_trials) + " Trial(s) of " + str(self.nr_cells) + " Cell(s) in " + str(self.time_bins) + " Time step(s)."
		if type(self.data.index) == pandas.MultiIndex:
			ind = dict(zip(*[self.data.index.names, range(len(self.data.index.names))]))
			additional = [str(n) + " (" +str(self.data.index.levshape[ind[n]]) + ") " for n in ind if not n == 'Trial' and not n == 'Condition' and not n == 'Cell']
			if len(additional) > 0:
				s = s + " Additional indices: " + ", ".join(additional) 
		if not type(self.other_spikes) == bool:
			s = s + " Also there is additional data:\n"
			s = s + str(self.other_spikes)
		else:
			s = s + " No other data."
		return s
	def to_pickle(self,path):
		"""
			Saves the DataFrame to a file
		"""
		print "Saving to "+path+"... "
		print self
		print self.data
		print pandas.DataFrame(self.data)
		pandas.DataFrame(self.data).to_pickle(path)
		if not self.desc == "":
			with open(path+".info","w") as f:
				f.write(self.desc)
				f.close()
	def read_pickle(self,path):
		"""
			Loads a DataFrame from a file
		"""
		self.data = pandas.read_pickle(path)
		if os.path.exists(path+".info"):
			with open(path+".info","r") as f:
				self.desc = f.read()
		return self
	def html_view(self):
		view = View()
		data_prefix = ""
		for c in range(self.nr_conditions):
			cond_data = self.condition(c)
			with view.figure(data_prefix + "/tabs/Condition " + str(c) + "/Firing Rate/Plot/#4/Mean/tabs/Axis 0"):
				pl.plot(np.mean(np.array(cond_data.firing_rate()).transpose(),axis=0))
			with view.figure(data_prefix + "/tabs/Condition " + str(c) + "/Firing Rate/Plot/#4/Mean/tabs/Axis 1"):
				pl.plot(np.mean(np.array(cond_data.firing_rate()).transpose(),axis=1))
			with view.figure(data_prefix + "/tabs/Condition " + str(c) + "/Firing Rate/Plot/#4/Mean/tabs/Axis 1 (smoothed)"):
				pl.plot(gaussian_filter(np.mean(np.array(cond_data.firing_rate()).transpose(),axis=1),20))
			for cell in range(cond_data.nr_cells):
				cell_data = cond_data.cell(cell)
				with view.figure(data_prefix + "/tabs/Condition " + str(c) + "/Spikes/tabs/Cell " + str(cell) + "/Plot"):
					pl.imshow(1-cell_data.data,interpolation='nearest',aspect=20,cmap='gray')
		return view