"""
.. module:: ni.data.monkey
   :platform: Unix
   :synopsis: Loads Data into a Panda Data Frame

"""

import ni
import scipy.io
import numpy as np
import pandas as pd
import itertools
import ni.model.pointprocess
import ni.data.data

MILLISECOND_RESOLUTION = 1000
#path = '/work/jahuth/'
#path = ni.config.get("ni.data.monkey.path") #"/net/store/ni/happycortex/statmodelling/data/"

available_trials = ['101a03', '104a10', '107a03', '108a08', '112a03', '101a03', '104a11', '107a04', '109a04', '112b02', '101a04', '105a04', '108a05', '110a03', '113a04', '102a09', '105a05', '108a06', '111a03', '113a05', '103a03', '106a03', '108a07', '111a04']
available_files = ['101a03', '104a10', '107a03', '108a08', '112a03', '101a03', '104a11', '107a04', '109a04', '112b02', '101a04', '105a04', '108a05', '110a03', '113a04', '102a09', '105a05', '108a06', '111a03', '113a05', '103a03', '106a03', '108a07', '111a04']

def Data(file_nr='101a03',resolution=MILLISECOND_RESOLUTION,trial=[],condition=[],cell=[]):
	"""
	Loads Data into a Data Frame

	Expects a file number. Available file numbers are in ni.data.monkey.available_files::

		>>> print ni.data.monkey.available_files
			['101a03', '104a10', '107a03', '108a08', '112a03', '101a03', '104a11', '107a04', '109a04', '112b02', '101a04', '105a04', '108a05', '110a03', '113a04', '102a09', '105a05', '108a06', '111a03', '113a05', '103a03', '106a03', '108a07', '111a04']

	**trial**

		number of trial to load or list of trials to load.
		Non-existent trial numbers are ignored.

	**condition**

		number of condition to load or list of conditions to load.
		Non-existent condition numbers are ignored.

	**cell**

		number of cell to load or list of cells to load.
		Non-existent cell numbers are ignored.

	Example::

		data = ni.data.monkey.Data(trial_nr = ni.data.monkey.available_trials[3], trial=range(10), condition = 0)


	"""
	d = _Data(file_nr,resolution,trial=trial,condition=condition,cell=cell)
	return ni.data.data.Data(d.data)

class _Data:
	"""
	Loads Data into a Data Frame
	"""
	def __init__(self,trial_nr='101a03',resolution=MILLISECOND_RESOLUTION,trial=[],condition=[],cell=[]):
		path = ni.config.get("ni.data.monkey.path")
		mat = scipy.io.loadmat(path + 'nic'+str(trial_nr)+'.mat')
		try:
			self.init_dot_indexing(mat,resolution,trial=trial,condition=condition,cell=cell)
		except:
			self.init_square_brackets_indexing(mat,resolution,trial=trial,condition=condition,cell=cell)
	def init_dot_indexing(self,mat,resolution=MILLISECOND_RESOLUTION,trial=[],condition=[],cell=[]):
		""" one of two possible indexing methods (differs by python version and file) """
		self.Spike_times_STC = mat['Data'][0][0].Spike_times_STC[0][0]
		self.T = mat['Data'][0][0].T[0][0]
		self.Stimulus = mat['Data'][0][0].Stimlus[0][0]
		self.name = mat['Data'][0][0].name
		self.nr_cells = np.size(self.Spike_times_STC.all_SUA[0][0].spike_times,2);
		self.nr_conditions = self.Spike_times_STC.all_SUA[0][0].Nr_stimlus[0][0]
		self.nr_trials = np.size(self.Spike_times_STC.all_SUA[0][0].spike_times,1);
		self.time_bins = 10000
		d = []
		tuples = []
		for con in range(self.nr_conditions):
			if condition == [] or con == condition or (type(condition) == list and con in condition):
				for t in range(self.nr_trials):
					if trial == [] or t == trial or (type(trial) == list and t in trial):
						for c in range(self.nr_cells):
							if cell == [] or c == cell or (type(cell) == list and c in cell):
								spikes = list(ni.model.pointprocess.getBinary(self.Spike_times_STC.all_SUA[0][0].spike_times[con,t,c].flatten()*resolution))
								if spikes != []:
									d.append(spikes)
									tuples.append((con,t,c))
		index = pd.MultiIndex.from_tuples(tuples, names=['Condition','Trial','Cell'])
		self.data = pd.DataFrame(d, index = index)
		self.time_bins = len(self.data.columns)
	def init_square_brackets_indexing(self,mat,resolution=MILLISECOND_RESOLUTION,trial=[],condition=[],cell=[]):
		""" one of two possible indexing methods (differs by python version and file) """
		self.Spike_times_STC = mat['Data'][0][0]['Spike_times_STC'][0][0]
		self.T = mat['Data'][0][0].T[0][0]
		self.Stimulus = mat['Data'][0][0]['Stimlus'][0][0]
		self.name = mat['Data'][0][0]['name']
		self.nr_cells = np.size(self.Spike_times_STC['all_SUA'][0][0]['spike_times'],2);
		self.nr_conditions = self.Spike_times_STC['all_SUA'][0][0]['Nr_stimlus'][0][0]
		self.nr_trials = np.size(self.Spike_times_STC['all_SUA'][0][0]['spike_times'],1);
		self.time_bins = 10000
		d = []
		tuples = []
		for con in range(self.nr_conditions):
			if condition == [] or con == condition or (type(condition) == list and con in condition):
				for t in range(self.nr_trials):
					if trial == [] or t == trial or (type(trial) == list and t in trial):
						for c in range(self.nr_cells):
							if cell == [] or c == cell or (type(cell) == list and c in cell):
								spikes = list(ni.model.pointprocess.getBinary(self.Spike_times_STC['all_SUA'][0][0]['spike_times'][con,t,c].flatten()*resolution))
								if spikes != []:
									d.append(spikes)
									tuples.append((con,t,c))
		index = pd.MultiIndex.from_tuples(tuples, names=['Condition','Trial','Cell'])
		self.data = pd.DataFrame(d, index = index)
		self.time_bins = len(self.data.columns)
	def __str__(self):
		s =  "Spike data: " + str(self.nr_conditions) + " Condition(s) " + str(self.nr_trials) + " Trial(s) of " + str(self.nr_cells) + " Cell(s) in " + str(self.time_bins) + " Time step(s)."
		return s

