"""
.. module:: ni.model.backend_elasticnet
   :platform: Unix
   :synopsis: Model Backend using sklearn.linear_model.ElasticNet

.. moduleauthor:: Jacob Huth <jahuth@uos.de>

"""

import sklearn.linear_model as linear_model 
import numpy as np

class Configuration:
	"""
		Default Values:

			crossvalidation = True

				If true, alpha and l1_ratio will be calculated by crossvalidation.

			alpha = 0.5

			l1_ratio = 1

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
			self.params = f.coef_
			self.statistics = { 'alpha': f.alpha, 'coef_path_': f.coef_path_, 'intercept': f.intercept_ }
		except:
			self.params = np.zeros(1)
			self.statistics = {}
	def predict(self,X=False):
		return self.fit.predict(X)

class Model:
	def __init__(self, c = False):
		if type(c)== bool:
			c = Configuration()
		self.configuration = c
		if c.crossvalidation == True:
			self.model = linear_model.ElasticNetCV(fit_intercept=False)#(alpha=self.configuration.alpha, l1_ratio=self.configuration.l1_ratio)
		else:
			self.model = linear_model.ElasticNet(fit_intercept=False,alpha=c.alpha, l1_ratio=c.l1_ratio)
	def fit(self,x,dm):
		return Fit(self.model.fit(dm,x.squeeze()),self)
def predict(x,dm):
	#model = linear_model.ElasticNetCV(fit_intercept=False)
	#model.intercept_ = 0
	#model.coef_ = x
	prediction = np.dot(dm,x)#model.predict(dm)
	return prediction.squeeze()
def compare(x,p,nr_trials=1):
	x = x.squeeze()
	p = p.squeeze()
	binomial = statsmodels.genmod.families.family.Binomial()
	dv = binomial.deviance(x,p)
	ll = binomial.loglike(x,p)
	if isinstance(data, Data) or isinstance(data, ni.data.data.Data):
		nr_trials = data.nr_trials
	return {'Deviance': dv/nr_trials, 'Deviance_all': dv, 'LogLikelihood': ll/nr_trials, 'LogLikelihood_all': ll}
