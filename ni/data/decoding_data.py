"""
.. module:: ni.data.decoding_data
   :platform: Unix
   :synopsis: Loads Data into a Panda Data Frame

.. moduleauthor:: Jacob Huth

Loads Data into a Panda Data Frame

"""
import scipy.io
import numpy as np
import pandas as pd
import itertools
import ni.model.pointprocess
reload (ni.model.pointprocess)

def get():
	mat = scipy.io.loadmat('ni/data/Group.mat')
	data = {}
	data['mat'] = mat
	data['Nr_cells'] = np.size(mat['Spikes_Group_Condition_A'],0);
	data['Nr_conditions']   = 2;
	data['Nr_trials'] = np.size(mat['Spikes_Group_Condition_A'],1);
	data['Trial_start'] = 0.5; # That's when the movie starts
	data['Fixation_dot_color_change'] = 4.3;
	data['Max_trial_length'] = 3.5;
	return data

class Cell:
	def __init__(self,data):
		#print data[0]*1000
		self.train = ni.model.pointprocess.getCounts(data[0]*1000)

class Trial:
	def __init__(self):
		self.nr_cells = 0
		self.cells = []
	def addCell(self,data):
		self.cells.append(Cell(data))
		self.nr_cells = self.nr_cells + 1
	def getMatrix(self):
		return np.array([c.train for c in self.cells])

class DecodingData:
	"""
	Loads Data into a Panda Data Frame
		
	"""
	def __init__(self):
		mat = scipy.io.loadmat('ni/data/Group.mat')
		self.nr_cells = np.size(mat['Spikes_Group_Condition_A'],0);
		self.nr_conditions = 2
		self.nr_trials = np.size(mat['Spikes_Group_Condition_A'],1);
		self.markers = {'trial_start': 500, 'Fixation_dot_color_change': 4300, 'max_trial_end': 3500}
		tuples = list(itertools.product(*[range(self.nr_conditions),range(self.nr_trials),range(self.nr_cells)]))
		index = pd.MultiIndex.from_tuples(tuples, names=['Condition','Trial','Cell'])
		d = []
		for m in [mat['Spikes_Group_Condition_A'],mat['Spikes_Group_Condition_B']]:
			for t in xrange(self.nr_trials):
				for c in xrange(self.nr_cells):
						spikes = list(ni.model.pointprocess.getBinary(m[c,t][0]*1000))
						d.append(spikes)
		#print d
		df = pd.DataFrame(d, index = index)
		self.data = df
		#self.trials = [Trial() for trial in xrange(self.nr_trials)]
		#for c in xrange(self.nr_cells):
		#	for t in xrange(self.nr_trials):
		#		self.trials[t].addCell(mat['Spikes_Group_Condition_A'][c,t])
