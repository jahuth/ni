"""
.. module:: ni.model.backend_glm
   :platform: Unix
   :synopsis: Model Backend using statsmodels.api.GLM

.. moduleauthor:: Jacob Huth <jahuth@uos.de>

This module provides a backend to the .ip model. It wraps the statsmodels.api.GLM object.

"""
import statsmodels.api as sm
import statsmodels

import statsmodels.genmod.families.family

import sklearn.linear_model as linear_model 

import numpy as np

class Configuration:
	"""
		Default Values:

			be_memory_efficient = True
				Does not keep the data with which it is fitted.
	"""
	def __init__(self):
		self.crossvalidation = True
		self.alpha = 0.5
		self.l1_ratio = 1
		self.be_memory_efficient = True
class Fit:
	def __init__(self, f, m):
		self.model  = m
		if self.model.configuration.be_memory_efficient:
			self.fit = None
		else:
			self.fit = f
		try:
			self.params = f.params
			self.pvalues = f.pvalues
			self.family = f.family
			self.llf = f.llf
			self.aic = f.aic
			self.bic = f.bic
			self.family = f.family
			self.tvalues = f.tvalues
			self.summary = str(f.summary())
			self.statistics = {'llf': f.llf, 'aic':  f.aic, 'bic': f.bic, 'pvalues': f.pvalues, 'tvalues': f.tvalues}
		except:
			self.params = np.zeros(1)
			self.statistics = {}
	def predict(self,X=False):
		return self.family.fitted(np.dot(self.params,X))
		#return self.fit.predict(exog=X)

class Model:
	def __init__(self, c = False):
		if type(c)== bool:
			c = Configuration()
		self.configuration = c
		self.model = False
	def fit(self,y,X):
		self.model = sm.GLM(y, X, family=sm.families.Binomial())
		try:
			fit = Fit(self.model.fit(),self)
		except:
			print "Fitting failed!"
			fit = Fit("Fitting failed.",self)
			fit.params = np.zeros(X.shape[1])
		if self.configuration.be_memory_efficient:
			self.model = None
		return fit

def predict(x,dm):
	binomial = statsmodels.genmod.families.family.Binomial()
	prediction = binomial.fitted(np.dot(dm,np.squeeze(x)))
	prediction.shape = (prediction.shape[0],1)
	return prediction
def compare(x,p,nr_trials=1):
	x = x.squeeze()
	p = p.squeeze()
	binomial = statsmodels.genmod.families.family.Binomial()
	dv = binomial.deviance(x,p)
	ll = binomial.loglike(x,p)
	if isinstance(data, Data) or isinstance(data, ni.data.data.Data):
		nr_trials = data.nr_trials
	return {'Deviance': dv/nr_trials, 'Deviance_all': dv, 'LogLikelihood': ll/nr_trials, 'LogLikelihood_all': ll}
