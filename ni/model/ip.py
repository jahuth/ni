"""
.. module:: ni.model.ip
   :platform: Unix
   :synopsis: Inhomogeneous Pointprocess Model

.. moduleauthor:: Jacob Huth <jahuth@uos.de>

Inhomogeneous Pointprocess Generalized Linear Model
--------------------------------------------------------


Adapted from FMTP by Robert Costa


A `generalized linear model <http://en.wikipedia.org/wiki/Generalized_linear_model>`_ predicts a variable *Y* with a linear predictor and a link function.
The linear predictor of the form:
	:math:`\eta = X\cdot\\beta`

Where **X** is a matrix consisting of rows of values that correspond to a specific point in time of the modeled process.
Each row my model a certain aspect (ie. time in trial, time after spike of some neuron) and is then weighted by the corresponding :math:`\\beta` parameter value. Each aspect may be scaled arbitrarily and shifted.
This weighted matrix is then added up into a firing probability that is passed on to the link function.
	.. image:: _static/examples/design_matrix_cross_component.png
	.. image:: _static/examples/design_matrix_rate_component.png


The link function in our case of pointprocesses (ie. a poisson, bernoulli or binomial distribution, depending on notation) it is either the *log* or *logit* function (:math:`ln\left(\\frac{\mu}{(1-\mu)}\\right)`).

Uses one of two backends :py:mod:`ni.model.backend_glm` and :py:mod:`ni.model.backend_elasticnet`

"""
from __future__ import division
from ni.tools.project import *

import ni.config

import numpy as np
import pylab as pl
import scipy
import scipy.ndimage
import pandas
import statsmodels.api as sm
import statsmodels.genmod.families.family
from copy import copy

import create_splines as cs
reload(cs)
import create_design_matrix_vk as cdm

import designmatrix

import ni.model.ip_generator

import ni.data.data
reload(ni.data.data)

from ni.data.data import Data as Data

import backend_elasticnet as backend_model
import backend_glm
import backend_elasticnet
reload(backend_model)

import ni.tools.pickler
from ni.tools.html_view import View

class Configuration(ni.tools.pickler.Picklable):
	"""
	The following values are the defaults used:	

		self.backend = "glm"

			The backend used. Valid options: "glm" and "elasticnet"

		self.history_length = 100
			
			Length of the history kernel

		self.knot_number = 3

			Number of knots in ?the history kernel?

		self.order_flag = 2

			Something

			.. todo::

				Find out what this is

		self.knots_rate = 10

			Knots of the firing rate kernel (knots/second)

		Look at the [source] for a full list of defaults.

	"""
	def __init__(self,c=False):
		self.history_length = ni.config.get('model.ip.history_length',100)
		self.knot_number = ni.config.get('model.ip.knot_number',3)
		self.order_flag = 1
		self.knots_rate = ni.config.get('model.ip.knots_rate',10)
		self.design = 'new'
		self.mask = np.array([True])
		self.custom_kernels = []
		self.custom_components = []
		self.autohistory = True
		self.crosshistory = True
		self.rate = True
		self.constant = True
		self.be_memory_efficient = True
		self.adaptive_rate = False
		self.adaptive_rate_smooth_width = ni.config.get('model.ip.adaptive_rate_smooth_width',20)
		self.cell = 0
		self.backend = ni.config.get('model.ip.backend',"glm")
		self.delete_last_spline = True 
		self.autohistory_2d = False
		self.name = "Inhomogeneous Point Process"
		if type(c) == str:
			self.loads(c)
		if type(c) == dict:
			self.__dict__.update(c)
			self.eval_dict()
		if isinstance(c, Configuration):
			self.loads(c.dumps())
		if self.mask == np.array([  6.91730533e-310]): # glitch in numpy
			self.mask = [True]
	def __str__(self):
		return "ni.model.ip.Configuration(\""+ni.tools.pickler.dumps(self.__dict__).replace('"', '\\"')+"\")"

class FittedModel(ni.tools.pickler.Picklable):
	"""
	When initialized via Model.fit() it contains a copy of the configuration, a link to the model it was fitted from and fitting parameters:

		FittedModel. **fit**

			modelFit Output

		FittedModel. **design**
		
			The DesignMatrix used. Use *design.matrix* for the actual matrix or design.get('...') to extract only the rows that correspond to a keyword.


	"""
	def __init__(self, model):

		if type(model) == str or type(model) == dict:
			self.loads(model)
			if type(self.beta) == str:
				self.beta = eval(self.beta.replace("np.ndarray","np.array"))
		else:
			if isinstance(model, Model):
				self.model = model
			else:
				raise Exception("Initialized without a model.")
	@property
	def complexity(self):
		try:
			return len(self.beta)
		except:
			return False
	"""def getBareModel(self):
		return {'beta':self.beta, 'design': str(self.design), 'statistics': self.statistics }
	def fromBareModel(self,bm):
		self.beta = bm["beta"]
		self.statistics = bm["statistics"]
		self.design = bm["design"]"""
	def getParams(self):
		return [np.sum(self.design.get(h) * self.beta[self.design.getIndex(h)],1) for h in np.unique(self.design.header)]
	def getPvalues(self):
		return dict((h,self.statistics.pvalues[self.design.getIndex(h)]) for h in np.unique(self.design.header))
	def pvalues_by_component(self):
		return dict((h,self.statistics.pvalues[self.design.getIndex(h)]) for h in np.unique(self.design.header))
	def plotParams(self,x=-1):
		figs = {}
		for h in np.unique(self.design.header):
			fig = pl.figure()
			pl.plot(np.sum(self.design.get(h)[:x] * self.beta[self.design.getIndex(h)],1))
			pl.title(h)
			figs[h] = fig
		return figs
	def plot_prototypes(self):
		figs = {}
		for h in np.unique(self.design.header):
			fig = pl.figure()
			splines = self.design.get(h)
			if 'rate' in h:
				pl.plot(np.sum(self.design.get(h)[:self.design.trial_length] * self.beta[self.design.getIndex(h)],1))
				pl.title(h)
			elif len(splines.shape) == 1 or (splines.shape[0] == 1):
				pl.plot(np.sum(self.design.get(h) * self.beta[self.design.getIndex(h)],1),'o')
				pl.title(h)
			elif len(splines.shape) == 2:
				pl.plot(np.sum(self.design.get(h) * self.beta[self.design.getIndex(h)],1))
				pl.title(h)
			elif len(splines.shape) == 3:
				slices = np.zeros(splines.shape)
				for (i, ind) in zip(range(splines.shape[0]),self.design.getIndex(h)):
					slices[i,:,:] = splines[i,:,:] * self.beta[ind]
				pl.imshow(slices.sum(axis=0),cmap='jet')
				figs[h + '_sum'] = fig
				fig = pl.figure()
				for i in range(len(slices)):
					pl.subplot(np.ceil(np.sqrt(slices.shape[0])),np.ceil(np.sqrt(slices.shape[0])),i+1)
					pl.imshow(slices[i],vmin=np.percentile(slices,1),vmax=np.percentile(slices,99),cmap='jet')
				pl.suptitle(h)
				figs[h] = fig
			else:
				pl.plot(np.sum(self.design.get(h) * self.beta[self.design.getIndex(h)],1))
				pl.title(h)
			figs[h] = fig
		return figs
	def prototypes(self):
		prot = {}
		for h in np.unique(self.design.header):
			splines = self.design.get(h)
			if len(splines.shape) == 1:
				prot[h] = self.design.get(h) * self.beta[self.design.getIndex(h)]
			elif len(splines.shape) == 2:
				prot[h] = np.sum(self.design.get(h) * self.beta[self.design.getIndex(h)],1)
			elif len(splines.shape) == 3:
				prot[h] = np.zeros(splines.shape)
				for (i, ind) in zip(range(splines.shape[0]),self.design.getIndex(h)):
					prot[h][i,:,:] = splines[i,:,:] * self.beta[ind]
			"""
			if 'rate' in h or 'constant' in h:
				prot[h] = np.sum(self.design.get(h) * self.beta[self.design.getIndex(h)],1)
			elif 'autohistory' in h or 'crosshistory' in h:
				prot[h] = np.sum(self.design.get(h) * self.beta[self.design.getIndex(h)],1)
			else:
				prot[h] = np.sum(self.design.get(h) * self.beta[self.design.getIndex(h)],1)
			"""
		return prot
	"""
	def getWeights(self):
		prot = {}
		for h in np.unique(self.design.header):
			splines = self.design.get(h)
			if len(splines.shape) == 3:
				prot[h] = np.mean(np.sum([splines[i] * self.beta[self.design.getIndex(h)][i] for i in range(splines.shape[0])],0))
			else:
				prot[h] = np.mean(np.sum(splines * self.beta[self.design.getIndex(h)],1))
		return prot"""
	def history_model(self, n = 'autohistory'):
		"""
			TODO: sort out what is saved where
		"""
		return np.sum(self.design.history_kernel * self.beta[self.design.getIndex(n)],1)
	def firing_rate_model(self):
		rate_design = self.design.getIndex('rate')
		return self.design.get('rate')[:self.design.trial_length,:]*self.beta[rate_design]
	def plot_firing_rate_model(self):
		rate_design = self.design.getIndex('rate')
		return plot(np.sum(self.design.get('rate')[:self.design.trial_length,:]*self.beta[rate_design]))
	def generate(self, bins=-1):
		"""
		Generates new spike trains from the extracted staistics

		Currently uses rate model and autohistory.

			**bins**
		
				How many bins should be generated (should be multiples of trial_length)
				
		"""
		spikes = []
		if bins < 0:
			bins = self.design.trial_length
		prototypes = self.getPrototypes()
		ps = []
		autohistory = prototypes['autohistory']#*self.firing_rate*self.firing_rate/abs(np.mean(prototypes['autohistory']))
		rate = self.firing_rate_model().sum(1) + prototypes['constant']
		time = np.zeros(bins)
		for i in range(self.design.trial_length):
			rand = np.random.rand()
			p = rate[i]
			ps.append(p)
			if rand < self.family_fitted_function(p):
				time[i] = 1
				spikes.append(i)
				kernel_end = np.min([autohistory.shape[0],len(rate) - i])
				rate[i:i+kernel_end] = rate[i:i+kernel_end] + autohistory[:kernel_end]
		return (spikes,time,np.array(ps))
	def family_fitted_function(self,p):
		"""
			only implemented family: Binomial
		"""
		return sm.families.Binomial().fitted(p)
	def predict(self,data):
		"""
		Using the model this will predict a firing probability function according to a design matrix.
		"""
		if isinstance(data, Data) or isinstance(data, ni.data.data.Data):
			dm = self.design.combine(data)
		else:
			dm = data

		return self.model.backend.predict(self.beta, dm)
	def compare(self,data):
		"""
		Using the model this will predict a firing probability function according to a design matrix.

		Returns:

			**Deviance_all**: dv, 
			**LogLikelihood_all**: ll, 
			**Deviance**: dv/nr_trials, 
			**LogLikelihood**: ll/nr_trials, 
			**llf**: Likelihood function over time 
			**ll**: np.sum(ll)/nr_trials
		"""
		return self.model.compare(data, self.predict(data))
	def to_pickle(self,path):
		#{'beta':self.beta, 'design': str(self.design)}
		raise Exception("Not implemented yet.")
	def read_pickle(self,path):
		raise Exception("Not implemented yet.")
	def dumps(self):
		return ni.tools.pickler.dumps({'beta': self.beta, 'model': self.model,'statistics': self.statistics})
	def html_view(self):
		view = View()
		model_prefix = self.model.name + "/"
		view.add(model_prefix + "#2/beta",self.beta)
		for key in self.configuration.__dict__:
			view.add(model_prefix + "#3/tabs/Configuration/table/"+key,self.configuration.__dict__[key])
		view.add(model_prefix + "#3/tabs/Design","")
		prot_nr = 0
		prototypes = self.plot_prototypes()
		for p in prototypes:
			prot_nr = prot_nr + 1
			view.savefig(model_prefix + "#3/tabs/Prototypes/tabs/"+str(p), fig = prototypes[p])
		prototypes = self.prototypes()
		if 'autohistory2d' in prototypes and 'autohistory' in prototypes:
			with view.figure(model_prefix + "#3/tabs/Prototypes/tabs/autohistory_2d+autohistory"):
				for i in range(prototypes["autohistory2d"].shape[2]):
					pl.plot(sum(prototypes["autohistory2d"],0)[i,:]+prototypes["autohistory"],'g:')
				pl.plot(prototypes["autohistory"],'b-')
		for c in self.design.components:
			if type(c) != str:
				with view.figure(model_prefix + "#3/tabs/Design/#2/tabs/"+c.header+"/#2/Splines"):
					splines = c.getSplines()
					if len(splines.shape) == 1:
						pl.plot(splines,'-o')
					elif len(splines.shape) == 2:
						if splines.shape[0] == 1 or splines.shape[1] == 1:
							pl.plot(splines,'-o')
						else:
							pl.plot(splines)
					elif len(splines.shape) == 3:
						for i in range(splines.shape[0]):
							pl.subplot(np.ceil(np.sqrt(splines.shape[0])),np.ceil(np.sqrt(splines.shape[0])),i+1)
							pl.imshow(splines[i,:,:],interpolation='nearest')
			else:
				view.add(model_prefix + "#3/tabs/Design/#2/tabs/##/component",str(c))
		return view
class Model(ni.tools.pickler.Picklable):
	def __init__(self, configuration = None, nr_bins = 0):
		"""
		Does an model with history kernel etc.

			**nr_bins** if undefined, use maximal spike time

		Uses one of two backends :py:module:`ni.model.backend_glm` and :py:module:`ni.model.backend_elasticnet`


		"""
		if configuration == None:
			configuration = Configuration()
		if type(configuration) == dict:
			configuration = Configuration(configuration)
		self.configuration = configuration
		self.name = self.configuration.name
		self.history_kernel = "log kernel"
		self.nr_bins = nr_bins
		self.loads(configuration)
		if self.configuration.backend == "glm":
			self._backend = backend_glm
		elif self.configuration.backend == "elasticnet":
			self._backend = backend_elasticnet
		else:
			raise Exception("Backend '"+self.configuration.backend+"' is not a Model backend.")
		#self.rate_splines = cs.create_splines_linspace(nr_bins, self.configuration.knots_rate, 0)
	@property
	def backend(self):
	    return self._backend
	@backend.setter
	def backend(self, b):
		if b == "glm":
			self._backend = backend_glm
		elif b == "elasticnet":
			self._backend = backend_elasticnet
	def predict(self,beta,data):
		dm = self.dm(data)
		x = beta
		return self.backend.predict(x,dm)
	def compare(self,data,p,nr_trials=1):
		binomial = statsmodels.genmod.families.family.Binomial()
		x = self.x(data)
		x = x.squeeze()
		p = p.squeeze()
		p[p<=0] = 0.000000001
		dv = binomial.deviance(x,p)
		ll_bin = x * np.log(p) + (1-x)*np.log(1-p)
		ll_bin[np.isnan(ll_bin)] = np.min(ll_bin)
		ll = binomial.loglike(x,p)
		if isinstance(data, Data) or isinstance(data, ni.data.data.Data):
			nr_trials = data.nr_trials
		return {'Deviance': dv/nr_trials, 'Deviance_all': dv, 'LogLikelihood': ll/nr_trials, 'LogLikelihood_all': ll, 'llf': ll_bin, 'll': np.sum(ll_bin)/nr_trials}
		#return self.backend.compare(x,p,nr_trails)
	def generateDesignMatrix(self,data,trial_length):
		design_template = designmatrix.DesignMatrixTemplate(data.nr_trials * data.time_bins,trial_length)
		log_kernel = cs.create_splines_logspace(self.configuration.history_length, self.configuration.knot_number, self.configuration.delete_last_spline)
		if self.configuration.autohistory:
			kernel = self.history_kernel
			if kernel == "log kernel":
				kernel = log_kernel
			if self.configuration.custom_kernels != "list:":
				for c in self.configuration.custom_kernels:
					if c['Name'] == 'autohistory':
						kernel = c['Kernel']
			design_template.add(designmatrix.HistoryComponent('autohistory', channel=self.configuration.cell, kernel=kernel, delete_last_spline=self.configuration.delete_last_spline))
		if self.configuration.autohistory_2d:
			kernel_1 = self.history_kernel
			kernel_2 = self.history_kernel
			if kernel_1 == "log kernel":
				kernel_1 = log_kernel
			if kernel_2 == "log kernel":
				kernel_2 = log_kernel
			if self.configuration.custom_kernels != "list:":
				for c in self.configuration.custom_kernels:
					if c['Name'] == 'autohistory2d':
						kernel_1 = c['Kernel']
						kernel_2 = c['Kernel']
					if c['Name'] == 'autohistory2d_1':
						kernel_1 = c['Kernel']
					if c['Name'] == 'autohistory2d_2':
						kernel_2 = c['Kernel']
			design_template.add(designmatrix.SecondOrderHistoryComponent('autohistory2d', channel_1=self.configuration.cell, channel_2=self.configuration.cell, kernel_1=kernel_1, kernel_2=kernel_2, delete_last_spline=self.configuration.delete_last_spline))
		# Generating crosshistory splines for all trials
		crosshistories = []
		if  type(self.configuration.crosshistory) == int or type(self.configuration.crosshistory) == float:
			self.configuration.crosshistory = [int(self.configuration.crosshistory)]
		if self.configuration.crosshistory == True or type(self.configuration.crosshistory) == list:
			for i in range(data.nr_cells):
				if i == self.configuration.cell:
					continue
				if self.configuration.crosshistory == True or i in self.configuration.crosshistory:
					kernel = self.history_kernel
					if kernel == "log kernel":
						kernel = log_kernel
					for c in self.configuration.custom_kernels:
						if c['Name'] == 'crosshistory'+str(i):
							kernel = c['Kernel']
					design_template.add(designmatrix.HistoryComponent('crosshistory'+str(i), channel=i, kernel = kernel, delete_last_spline=self.configuration.delete_last_spline))
		if self.configuration.rate:
			added_rate = False
			for c in self.configuration.custom_kernels:
					if c['Name'] == 'rate':
						design_template.add(designmatrix.RateComponent('rate',kernel = c['Kernel']))
						added_rate = True
			if not added_rate:
				if self.configuration.adaptive_rate:
					rate = data.firing_rate(self.configuration.adaptive_rate_smooth_width)[0]
					design_template.add(designmatrix.AdaptiveRateComponent('rate',rate,self.configuration.knots_rate,trial_length))
				else:
					design_template.add(designmatrix.RateComponent('rate',self.configuration.knots_rate,trial_length))
		if self.configuration.constant:
			design_template.add(designmatrix.Component('constant',np.ones((1,1))))
		##log("Combining Design Matrix")
		for c in self.configuration.custom_components:
			design_template.add(c)
		design = design_template
		design.setMask(self.configuration.mask)
		return design
	def x(self, in_spikes):
		if isinstance(in_spikes, Data) or isinstance(in_spikes, ni.data.data.Data):
			data = in_spikes
		else:
			return in_spikes
		if data.nr_cells == 1:
			x = data.getFlattend()
		else:
			x = data.cell(self.configuration.cell).getFlattend()
		return x.squeeze()
	def dm(self, in_spikes, design = False):
		"""
		Creates a design matrix from data and self.design
	
			**in_spikes** `ni.data.data.Data` instance

		"""
		if isinstance(in_spikes, Data) or isinstance(in_spikes, ni.data.data.Data):
			data = in_spikes
		else:
			return in_spikes

		if design == False:
			design = self.generateDesignMatrix(data,data.time_bins)
		self.last_generated_design = design
		dm = design.combine(data)
		return dm
	def fit(self, data=None,beta=None,x= None,dm = None, nr_trials=None):
		"""
		Fits the model
	
			**in_spikes** `ni.data.data.Data` instance


		example::

			from scipy.ndimage import gaussian_filter
			import ni
			model = ni.model.ip.Model(ni.model.ip.Configuration({'crosshistory':False}))
			data = ni.data.monkey.Data()
			data = data.condition(0).trial(range(int(data.nr_trials/2)))
			dm = model.dm(data)
			x = model.x(data)
			from sklearn import linear_model
			betas = []
			fm = model.fit(data)
			betas.append(fm.beta)
			print "fitted."
			for clf in [linear_model.LinearRegression(), linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])]:
				clf.fit(dm,x)
				betas.append(clf.coef_)

				figure()
				plot(clf.coef_.transpose(),'.')
				title('coefficients')
				prediction = np.dot(dm,clf.coef_.transpose())
				figure()
				plot(prediction)
				title('prediction')
				ll = x * log(prediction) + (len(x)-x)*log(1-prediction)
				figure()
				plot(ll)
				title('ll')
				print np.sum(ll)

		"""
		fittedmodel = FittedModel(self)
		fittedmodel.configuration = copy(self.configuration)

		fittedmodel.history_kernel = self.history_kernel
		if fittedmodel.history_kernel == "log kernel":
				fittedmodel.history_kernel = cs.create_splines_logspace(self.configuration.history_length, self.configuration.knot_number, 0)
		spike_train_all_trial = []

		if data is not None:
			spike_train_all_trial = data.getFlattend()
			firing_rate = np.mean(spike_train_all_trial)
			fittedmodel.firing_rate = firing_rate
			fittedmodel.trial_length = data.time_bins
			fittedmodel.design = self.generateDesignMatrix(data,data.time_bins)
			if dm is None:
				dm = self.dm(data,fittedmodel.design)
			if x is None:
				x = self.x(data)
		w = np.where(dm.transpose())[0]
		cnt = [np.sum(w== i) for i in range(dm.shape[1])]

		if sum(np.array(cnt) >= dm.shape[0]) > 0:
			log("!! "+str(sum(np.array(cnt) == dm.shape[0]))+ " Components are only 0. \n"+str(sum(np.array(cnt) <= dm.shape[0]*0.1))+" are mostly 0. "+str(sum(np.array(cnt) <= dm.shape[0]*0.5))+" are half 0.")
		else:
			log(""+str(sum(np.array(cnt) == dm.shape[0]))+ " Components are only 0. \n"+str(sum(np.array(cnt) <= dm.shape[0]*0.1))+" are mostly 0. "+str(sum(np.array(cnt) <= dm.shape[0]*0.5))+" are half 0.")
		zeroed_components = [i for i in range(len(cnt)) if cnt[i] == 0]

		fittedmodel.backend_model = self.backend.Model()
		if beta is None:
			fit = fittedmodel.backend_model.fit(x, dm) # NOTE frequently produces LinAlgError: SVD did not converge
			if False:
				# This might be introduced at a later data:
				# 	Components that are 0 can be excluded from the fitting process and then virtually reinserted in the beta and params attributes.
				beta = fit.params
				i_z = 0
				for i in range(len(cnt)):
					if i in zeroed_components:
						beta.append(0)
					else:
						beta.append(fit.params[i_z])
						i_z = i_z + 1
				fit.params = beta
			else:
				for z in zeroed_components:
					fit.params[z] = 0
			fittedmodel.beta = fit.params
		else:
			fit = self.backend.Fit(f=None,m=fittedmodel.backend_model)
			fit.params = beta
			fittedmodel.beta = beta
		fittedmodel.fit = fit
		if "llf" in fit.statistics:
			fit.statistics["llf_all"] = fit.statistics["llf"]
			if hasattr(data,'nr_trials'):
				fit.statistics["llf"] = fit.statistics["llf"]/data.nr_trials
			elif nr_trials is not None:
				fit.statistics["llf"] = fit.statistics["llf"]/nr_trials
		fittedmodel.statistics = fit.statistics
		#log("done fitting.")
		return fittedmodel
	def fit_with_design_matrix(self, fittedmodel, spike_train_all_trial, dm):
		fittedmodel.fit = fittedmodel.backend_model.fit(spike_train_all_trial, dm) 
		return fittedmodel
	def to_pickle(self,path):
		raise Exception("Not implemented yet.")
	def read_pickle(self,path):
		raise Exception("Not implemented yet.")
	def html_view(self):
		view = View()
		model_prefix = self.name + "/"
		for key in self.configuration.__dict__:
			view.add(model_prefix + "#3/tabs/Configuration/table/"+key,self.configuration.__dict__[key])
		return view

class MultiChannelModel(ni.tools.pickler.Picklable):
	def __init__(self,configuration={}):
		self.configuration = configuration
		self.models = []
	def append(self,m):
		self.models.append(m)

def generate_to_file(path,data,eval_trials,use_cells=[0],eval_bootstrap_repetitions=10):
	"""
		This function fits two models and generates data for each saved to path + 'data0.pkl' and path + 'data1.pkl'

		.. todo::

			split into more usefull and modular functions
	"""
	bootstrap_datas = []
	subjob("Fitting generative Model 1")
	cross_fits = []
	for cell in use_cells:
		subjob("fitting models with auto and crosshistory for cell " + str(cell))
		log("Fitting cell for generative model",4)
		fit_data = data.cell(cell)
		other_cells = [c for c in use_cells if c != cell]
		fit_data.other_spikes = data.cell(other_cells)
		c = Configuration()
		#c.crosshistory = False
		c.knots_rate = 30
		c.history_length = 100
		model = Model(c,fit_data.time_bins)
		model.name = 'Original Model Cell ' + str(cell)
		log("starting to fit...",1)
		#print fit_data
		fit = model.fit(fit_data)
		cross_fits.append(fit)
		superjob()
	superjob()

	bootstrap_data = []
	for r in range(eval_bootstrap_repetitions):
		#print r,
		gs = []
		gs_other = []
		for t in range(eval_trials):
			(spikes,p) = ni.model.ip_generator.generate(cross_fits)
			gs.append(spikes.transpose())
		bootstrap_data.append(Data(np.array(gs)))
	bootstrap_datas.append(pandas.merge(bootstrap_data,'Bootstrap Sample'))

	subjob("Fitting generative Model 2")
	cross_fits = []
	for cell in use_cells:
		subjob("fitting models with auto and crosshistory for cell " + str(cell))
		log("Fitting cell for generative model",4)
		fit_data = data.cell(cell)
		other_cells = [c for c in use_cells if c != cell]
		fit_data.other_spikes = data.cell(other_cells)
		c = Configuration()
		c.autohistory = False
		c.crosshistory = False
		c.knots_rate = 10
		model = Model(c,fit_data.time_bins)
		model.name = 'Original Model Cell ' + str(cell)
		log("starting to fit...",1)
		fit = model.fit(fit_data)
		cross_fits.append(fit)
		superjob()
	superjob()

	bootstrap_data = []
	for r in range(eval_bootstrap_repetitions):
		#print r,
		gs = []
		gs_other = []
		for t in range(eval_trials):
			(spikes,p) = ni.model.ip_generator.generate(cross_fits)
			gs.append(spikes.transpose())
		bootstrap_data.append(Data(np.array(gs)))
	bootstrap_datas.append(merge(bootstrap_data,'Bootstrap Sample'))
	log("Generated Bootstrap Data",3)
	#store = pandas.HDFStore(path)
	#store['data0'] = bootstrap_datas[0]
	#store['data1'] = bootstrap_datas[1]
	#data0 = pandas.DataFrame(bootstrap_datas[0])
	bootstrap_datas[0].to_pickle(path+'data0.pkl')
	#data1 = pandas.DataFrame(bootstrap_datas[0])
	bootstrap_datas[1].to_pickle(path+'data1.pkl')