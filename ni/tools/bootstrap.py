"""
.. module:: ni.tools.bootstrap
   :platform: Unix
   :synopsis: Provides Bootstrapping Methods for models.

.. moduleauthor:: Jacob Huth <jahuth@uos.de>

"""
import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib.pyplot import figure, plot, imshow, subplot, legend, title, show
from numpy import var, abs, sum, mean
import pandas
import random 
import pylab
import matplotlib as mpl
import scipy
from ni.tools.plot import *
from ni.tools.statcollector import StatCollector

def merge(stats):
	"""
		.. todo :: implement merge function for bootstrap data that calculates EIC etc.

	"""
	result = StatCollector()
	for s in stats:
		pass 
	pass

def description(prefix='',additional_information=""):
	"""
		Describes the common bootstrap output variables as a dictionary. `additional_information` will be appended to each entry, `prefix` will be prepended to each key.
	"""
	d = {}
	for b in bootstrap_results:
		d[prefix+b] = bootstrap_results[b] + additional_information
	return d

bootstrap_results = {
		'name': "Name of the Model", 
		'AIC': "AIC Information criterion",
		'EIC': "EIC",
		'EICE': ["EIC of one Bootstrap Sample", "Bootstrap Sample"],
		'EICE2': ["uncorrected EIC of one Bootstrap Sample", "Bootstrap Sample"], 
		'EICE_bias': ["EIC bias", "Bootstrap Sample"],
		'EICE_bias_uncorrected': ["uncorrected EIC bias", "Bootstrap Sample"],
		'llf_test_model':["LogLikelihood of actual model on test data", "Test Data Nr."], 
		'llf_train':["LogLikelihood of BS model on BS data", "Bootstrap Sample"],
		'llf_boot':["LogLikelihood of BS model on actual training data", "Bootstrap Sample"], 
		'llf_test':["LogLikelihood of BS model on test data", "Bootstrap Sample", "Test Data Nr."],
		'beta': "Parameters of the model fitted with actual data",
		'boot_betas': ["Parameters of the model fitted with a Bootstrap Sample", "Bootstrap Sample"],
		'complexity': "complexity of the model (ie. number of parameters)"
}

def bootstrap(bootstrap_repetitions,model,data,test_data=[],shuffle=True,prefix='',bootstrap_data=[]):
	"""
		A helper function that performs bootstrap evaluation of models.

		A Model `model` is fitted with some data `data`, called "actual data" or "D" and subsequently on all of a number of bootstrap samples "D*_n" for n in range(`bootstrap_repetitions`).
		This yields an `actual fit` and `bootstrap_repetitions` times `boot fit` (or `fit*`) for each sample.

		Use bootstrap_results for explanations on the dimensions of the result.

		`bootstrap_repetitions`

		`model`

		`data`

		`test_data`=[]

		`shuffle`=True

		`prefix`=''
			String that is prefixed to the results.

		`bootstrap_data`=[]
			If new data instead of trial shuffling is to be used as bootstrap data, this data should be passed here.
			The `ni.data.data.Data` Instance should contain an additional index `Bootstrap Sample`
	"""
	if bootstrap_data != []:
		return bootstrap_samples(bootstrap_data,model,data,test_data,shuffle,prefix)
	else:
		return bootstrap_trials(bootstrap_repetitions,model,data,test_data,shuffle,prefix)

def bootstrap_trials(bootstrap_repetitions,model,data,test_data=[],shuffle=True,prefix='',bootstrap_data=[]):
	"""
		Performs bootstrap evaluation by trial shuffling.

		A Model `model` is fitted with some data `data`, called "actual data" or "D" and subsequently on all of a number of bootstrap samples "D*_n" for n in range(`bootstrap_repetitions`).
		This yields an `actual fit` and `bootstrap_repetitions` times `boot fit` (or `fit*`) for each sample.

		Use bootstrap_results for explanations on the dimensions of the result.

		`bootstrap_repetitions`

			Number of bootstrap repetitions

		`model`

			Model to be evaluated. It needs to provide an `x()`, `dm()` and `fit(x=, dm=)`/`fit(data)` method.

		`data`


		`test_data`=[]

		`shuffle`=True

		`prefix`=''
			String that is prefixed to the results.

		`bootstrap_data`=[]
			If new data instead of trial shuffling is to be used as bootstrap data, this data should be passed here.
			The `ni.data.data.Data` Instance should contain an additional index `Bootstrap Sample`
	"""
	EICE = []; EICE2 = []; EICE_bias = []; EICE_bias_uncorrected = []; EIC_aic = []; EIC_bic = []
	dm = model.dm(data)
	x = model.x(data)
	actual_fit = model.fit(x=x,dm=dm)
	actual_fit_fit_aic = np.nan
	try:
		actual_fit_fit_aic = actual_fit.fit.aic
	except:
		pass
	model_prediction_fit = model.compare(x,model.predict(actual_fit.beta,dm))
	llf_test_model = [model.compare(t,model.predict(actual_fit.beta,t))['LogLikelihood'] for t in test_data]
	llf_train = []; llf_boot = []; llf_test = []; llf_test2 = []; betas = []
	print bootstrap_repetitions
	for boot_rep in range(bootstrap_repetitions):
		print boot_rep,
		bootstrap_trials = np.array(np.floor(np.random.rand(int(np.ceil(data.nr_trials)))*data.nr_trials),int)
		boot_data = data.trial(bootstrap_trials)
		boot_fit = model.fit(boot_data)
		boot_dm = model.dm(boot_data)
		boot_x = model.x(boot_data)
		boot_prediction = model.compare(x,model.predict(boot_fit.beta,dm))
		model_prediction_boot = model.compare(boot_x,model.predict(actual_fit.beta,boot_dm))
		boot_boot_prediction = model.compare(boot_x,model.predict(boot_fit.beta,boot_dm))
		llf_train.append(boot_boot_prediction['LogLikelihood'])
		llf_boot.append(boot_prediction['LogLikelihood'])
		llf_test.append([model.compare(t,model.predict(boot_fit.beta,t))['LogLikelihood'] for t in test_data])
		# EIC = -2 M(D) + 2 ( M*(D*) - M(D*) + M(D) - M*(D) )
		EICE.append(-2*model_prediction_fit['LogLikelihood'] + 2*(boot_boot_prediction['LogLikelihood'] - model_prediction_boot['LogLikelihood'] + model_prediction_fit['LogLikelihood'] - boot_prediction['LogLikelihood']))
		EICE2.append(-2*model_prediction_fit['LogLikelihood'] + 2*(boot_boot_prediction['LogLikelihood'] - boot_prediction['LogLikelihood']))
		EICE_bias.append(2*(boot_boot_prediction['LogLikelihood'] - model_prediction_boot['LogLikelihood'] + model_prediction_fit['LogLikelihood'] - boot_prediction['LogLikelihood']))
		EICE_bias_uncorrected.append(2*(boot_boot_prediction['LogLikelihood'] - boot_prediction['LogLikelihood']))
		# Knishi & Kitagawa 2007, p.198 Formula (8.23)
		try:
			EIC_aic.append(boot_fit.fit.aic)
			EIC_bic.append(boot_fit.fit.bic)
		except:
			pass
		betas.append(boot_fit.beta)
	res = {
		prefix+'name': model.name, 
		prefix+'statistics': actual_fit.statistics, 
		prefix+'AIC':actual_fit_fit_aic,
		prefix+'EIC':np.mean(EICE),
		prefix+'EICE':EICE,
		prefix+'EICE2':EICE2, 
		prefix+'EICE_bias':EICE_bias,
		prefix+'EICE_bias_uncorrected':EICE_bias_uncorrected,
		prefix+'llf_test_model':llf_test_model, 
		prefix+'llf_train':llf_train, 
		prefix+'llf_boot':llf_boot, 
		prefix+'llf_test':llf_test, 
		#prefix+'boot_aic':EIC_aic, 
		#prefix+'boot_bic':EIC_bic,,
		prefix+'beta':actual_fit.beta,
		prefix+'boot_betas':betas,
		prefix+'complexity':len(actual_fit.beta),
		prefix+'llf_train_model': model_prediction_fit['LogLikelihood']
		}
	if 'pvalues' in actual_fit.statistics:
		res['pvalues'] = actual_fit.statistics['pvalues']
	return res

def bootstrap_samples(bootstrap_data,model,data,test_data=[],shuffle=False,prefix='',boot_dim='Bootstrap Sample'):
	"""
		Performs bootstrap evaluation with bootstrap data.

		A Model `model` is fitted with some data `data`, called "actual data" or "D" and subsequently on all of a number of bootstrap samples "D*_n" for n in range(`bootstrap_repetitions`).
		This yields an `actual fit` and `bootstrap_repetitions` times `boot fit` (or `fit*`) for each sample.

		Use bootstrap_results for explanations on the dimensions of the result.

		`bootstrap_data`
			If new data instead of trial shuffling is to be used as bootstrap data, this data should be passed here.
			The `ni.data.data.Data` Instance should contain an additional index `Bootstrap Sample`

		`model`

			Model to be evaluated. It needs to provide an `x()`, `dm()` and `fit(x=, dm=)`/`fit(data)` method.

		`data`

		`test_data`=[]

		`shuffle`=True

		`prefix`=''
			String that is prefixed to the results.

		`boot_dim`
			The `ni.data.data.Data` Instance should contain an additional index `Bootstrap Sample` or be a list. If some other index should be used as bootstrap samples, `boot_dim` can be set to that.

	"""
	EICE = []; EICE2 = []; EICE_bias = []; EICE_bias_uncorrected = []; EIC_aic = []; EIC_bic = []
	dm = model.dm(data)
	x = model.x(data)
	actual_fit = model.fit(x=x,dm=dm)
	actual_fit_fit_aic = np.nan
	try:
		actual_fit_fit_aic = actual_fit.fit.aic
	except:
		pass
	model_prediction_fit = model.compare(x,model.predict(actual_fit.beta,dm))
	llf_test_model = [model.compare(t,model.predict(actual_fit.beta,t))['LogLikelihood'] for t in test_data]
	llf_train = []; llf_boot = []; llf_test = []; llf_test2 = []; betas = []
	bootstrap_repetitions = bootstrap_data.shape(boot_dim)
	print bootstrap_repetitions
	for boot_rep in range(bootstrap_repetitions):
		print boot_rep,
		if type(bootstrap_data) == list:
			boot_data = bootstrap_data[boot_rep]
		else:
			boot_data = bootstrap_data.filter(boot_rep,boot_dim)
		if shuffle:
			bootstrap_trials = np.array(np.floor(np.random.rand(int(np.ceil(boot_data.nr_trials)))*boot_data.nr_trials),int)
			boot_data = boot_data.trial(bootstrap_trials)
		boot_fit = model.fit(boot_data)
		boot_dm = model.dm(boot_data)
		boot_x = model.x(boot_data)
		boot_prediction = model.compare(x,model.predict(boot_fit.beta,dm))
		model_prediction_boot = model.compare(boot_x,model.predict(actual_fit.beta,boot_dm))
		boot_boot_prediction = model.compare(boot_x,model.predict(boot_fit.beta,boot_dm))
		llf_train.append(boot_boot_prediction['LogLikelihood'])
		llf_boot.append(boot_prediction['LogLikelihood'])
		llf_test.append([model.compare(t,model.predict(boot_fit.beta,t))['LogLikelihood'] for t in test_data])
		# EIC = -2 M(D) + 2 ( M*(D*) - M(D*) + M(D) - M*(D) )
		EICE.append(-2*model_prediction_fit['LogLikelihood'] + 2*(boot_boot_prediction['LogLikelihood'] - model_prediction_boot['LogLikelihood'] + model_prediction_fit['LogLikelihood'] - boot_prediction['LogLikelihood']))
		EICE2.append(-2*model_prediction_fit['LogLikelihood'] + 2*(boot_boot_prediction['LogLikelihood'] - boot_prediction['LogLikelihood']))
		EICE_bias.append(2*(boot_boot_prediction['LogLikelihood'] - model_prediction_boot['LogLikelihood'] + model_prediction_fit['LogLikelihood'] - boot_prediction['LogLikelihood']))
		EICE_bias_uncorrected.append(2*(boot_boot_prediction['LogLikelihood'] - boot_prediction['LogLikelihood']))
		# Knishi & Kitagawa 2007, p.198 Formula (8.23)
		try:
			EIC_aic.append(boot_fit.fit.aic)
			EIC_bic.append(boot_fit.fit.bic)
		except:
			pass
		betas.append(boot_fit.beta)
	res = {
		prefix+'name': model.name, 
		prefix+'statistics': actual_fit.statistics, 
		prefix+'AIC':actual_fit_fit_aic,
		prefix+'EIC':np.mean(EICE),
		prefix+'EICE':EICE,
		prefix+'EICE2':EICE2, 
		prefix+'EICE_bias':EICE_bias,
		prefix+'EICE_bias_uncorrected':EICE_bias_uncorrected,
		prefix+'llf_test_model':llf_test_model, 
		prefix+'llf_train':llf_train, 
		prefix+'llf_boot':llf_boot, 
		prefix+'llf_test':llf_test, 
		#prefix+'boot_aic':EIC_aic, 
		#prefix+'boot_bic':EIC_bic,,
		prefix+'beta':actual_fit.beta,
		prefix+'boot_betas':betas,
		prefix+'complexity':len(actual_fit.beta),
		prefix+'llf_train_model': model_prediction_fit['LogLikelihood']
		}
	if 'pvalues' in actual_fit.statistics:
		res['pvalues'] = actual_fit.statistics['pvalues']
	return res

def bootstrap_time(bootstrap_repetitions,model,data,test_data=[],prefix=''):
	"""
		Performs bootstrap evaluation of models.

		A Model `model` is fitted with some data `data`, called "actual data" or "D" and subsequently on all of a number of bootstrap samples "D*_n" for n in range(`bootstrap_repetitions`).
		This yields an `actual fit` and `bootstrap_repetitions` times `boot fit` (or `fit*`) for each sample.

		Use bootstrap_results for explanations on the dimensions of the result.

		`bootstrap_repetitions`

		`model`

			Model to be evaluated. It needs to provide an `x()`, `dm()` and `fit(x=, dm=)` method.

		`data`

		`test_data`=[]

		`prefix`=''
			String that is prefixed to the results.

	"""
	EICE = []; EICE2 = []; EICE_bias = []; EICE_bias_uncorrected = []; EIC_aic = []; EIC_bic = []
	dm = model.dm(data)
	x = model.x(data)
	actual_fit = model.fit(x=x,dm=dm)
	actual_fit_fit_aic = np.nan
	try:
		actual_fit_fit_aic = actual_fit.fit.aic
	except:
		pass
	model_prediction_fit = model.compare(x,model.predict(actual_fit.beta,dm))
	llf_test_model = [model.compare(t,model.predict(actual_fit.beta,t))['LogLikelihood'] for t in test_data]
	llf_train = []; llf_boot = []; llf_test = []; llf_test2 = []; betas = []
	print bootstrap_repetitions
	for boot_rep in range(bootstrap_repetitions):
		print boot_rep,
		bootstrap_time = np.array(np.floor(np.random.rand(int(np.ceil(dm.shape[0])))*dm.shape[0]),int)
		boot_dm = dm[bootstrap_time,:]
		boot_fit = model.fit(x=x, dm=boot_dm)
		boot_x = x
		boot_prediction = model.compare(x,model.predict(boot_fit.beta,dm))
		model_prediction_boot = model.compare(boot_x,model.predict(actual_fit.beta,boot_dm))
		boot_boot_prediction = model.compare(boot_x,model.predict(boot_fit.beta,boot_dm))
		llf_train.append(boot_boot_prediction['LogLikelihood'])
		llf_boot.append(boot_prediction['LogLikelihood'])
		llf_test.append([model.compare(t,model.predict(boot_fit.beta,t))['LogLikelihood'] for t in test_data])
		# EIC = -2 M(D) + 2 ( M*(D*) - M(D*) + M(D) - M*(D) )
		EICE.append(-2*model_prediction_fit['LogLikelihood'] + 2*(boot_boot_prediction['LogLikelihood'] - model_prediction_boot['LogLikelihood'] + model_prediction_fit['LogLikelihood'] - boot_prediction['LogLikelihood']))
		EICE2.append(-2*model_prediction_fit['LogLikelihood'] + 2*(boot_boot_prediction['LogLikelihood'] - boot_prediction['LogLikelihood']))
		EICE_bias.append(2*(boot_boot_prediction['LogLikelihood'] - model_prediction_boot['LogLikelihood'] + model_prediction_fit['LogLikelihood'] - boot_prediction['LogLikelihood']))
		EICE_bias_uncorrected.append(2*(boot_boot_prediction['LogLikelihood'] - boot_prediction['LogLikelihood']))
		# Knishi & Kitagawa 2007, p.198 Formula (8.23)
		try:
			EIC_aic.append(boot_fit.fit.aic)
			EIC_bic.append(boot_fit.fit.bic)
		except:
			pass
		betas.append(boot_fit.beta)
	res = {
		prefix+'name': model.name, 
		prefix+'statistics': actual_fit.statistics, 
		prefix+'AIC':actual_fit_fit_aic,
		prefix+'EIC':np.mean(EICE),
		prefix+'EICE':EICE,
		prefix+'EICE2':EICE2, 
		prefix+'EICE_bias':EICE_bias,
		prefix+'EICE_bias_uncorrected':EICE_bias_uncorrected,
		prefix+'llf_test_model':llf_test_model, 
		prefix+'llf_train':llf_train, 
		prefix+'llf_boot':llf_boot, 
		prefix+'llf_test':llf_test, 
		#prefix+'boot_aic':EIC_aic, 
		#prefix+'boot_bic':EIC_bic,,
		prefix+'beta':actual_fit.beta,
		prefix+'boot_betas':betas,
		prefix+'complexity':len(actual_fit.beta),
		prefix+'llf_train_model': model_prediction_fit['LogLikelihood']
		}
	if 'pvalues' in actual_fit.statistics:
		res['pvalues'] = actual_fit.statistics['pvalues']
	return res
	

def generate(model,bootstrap_repetitions):
	return [model.generate() for r in range(bootstrap_repetitions)]

def plotBootstrap(res,path):
	"""
		.. deprecated:: 0.1
			use the plot capabilities of the :class:`ni.tools.statcollector.StatCollector`.
	"""
	fig = figure()
	plot(res['EIC_aic'],'-+')
	plot(res['EICE'],'o')
	plot(res['EICE2'],'+')
	plot([0,len(res['EICE'])],[np.mean(res['EICE']),np.mean(res['EICE'])],'--')
	plot([0,len(res['llf_train'])],[res['AIC'],res['AIC']],':')
	lgd=legend(['AIC*','EIC*','EIC* uncorrected','EIC','AIC'],bbox_to_anchor=(1.05, 1), loc=2)
	fig.savefig(path+'EIC_dist.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
	plot([0,len(res['EICE'])],[np.mean(res['EICE']),np.mean(res['EICE'])],'--')
	plot([0,len(res['llf_train'])],[res['AIC'],res['AIC']],':')
	lgd=legend(['EIC','AIC'],bbox_to_anchor=(1.05, 1), loc=2)
	fig.savefig(path+'EIC.png', bbox_extra_artists=(lgd,), bbox_inches='tight')


def plotCompareBootstrap(reses,path):
	"""
		.. deprecated:: 0.1
			use the plot capabilities of the :class:`ni.tools.statcollector.StatCollector`.
	"""
	formats = [s1+s2 for s2 in ["-",":","-.","--"] for s1 in mpl.rcParams['axes.color_cycle']]
	colors = mpl.rcParams['axes.color_cycle'] * 5
	markers = [s2 for s2 in ["o","*","+.","^"] for s1 in mpl.rcParams['axes.color_cycle']]
	fig = figure()
	for res in reses:
		plot(res['llf_train'],'-o')
		plot(res['llf_test'],'-*')
		plot(res['llf_test2'],'-')
	legend(['LL Training','LL Test']*len(reses))
	fig.savefig(path+'llf_compare.png')
	fig = figure()
	leg = []
	for res in reses:
		if 'EICE' in res:
			_n, _bins, _patches = pyplot.hist(np.array(res['EICE']).flatten(), 50, normed=1, alpha=0.75)
			leg.append(res['name'])
	lgd=legend(leg,bbox_to_anchor=(1.05, 1), loc=2)
	pyplot.xlabel('EIC')
	fig.savefig(path+'compare_EIC_distributions.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
	for width in [10,20,50,100]:
		fig = pl.figure()
		leg = []
		i = 0
		for res in reses:
			if 'EICE' in res:
				format =formats[i]
				markerformat = markers[i] + colors[i]
				if res['name'] == 'Reduced Model':
					format = "r:"
					markerformat = 'r.'
					i = i - 1
				plotHist(res, 'EICE', format, markerformat,width)
				i = i + 1
				leg.append(res['name'])
				leg.append(res['name']+' Mean')
		lgd=pl.legend(leg,bbox_to_anchor=(1.05, 1), loc=2)
		fig.savefig(path+'fig3_compare_EIC_distributions_'+str(width)+'.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
	fig = figure()
	leg = []
	for res in reses:
		if 'llf_test' in res:
			_n, _bins, _patches = pyplot.hist(np.array(res['llf_test']).flatten(), 50, normed=1, alpha=0.75)
			leg.append(res['name'])
	pyplot.xlabel('Likelihood')
	lgd=legend(leg,bbox_to_anchor=(1.05, 1), loc=2)
	fig.savefig(path+'compare_llf_distributions.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
	for width in [10,20,50,100]:
		fig = pl.figure(figsize=(10,10))
		sp = 0
		for key in ['llf_train','llf_test','llf_test2']:
			pl.subplot(3,1,sp, title=key)
			sp = sp + 1
			leg = []
			i = 0
			for res in reses:
				if key in res:
					format =formats[i]
					markerformat = markers[i] + colors[i]
					if res['name'] == 'Reduced Model':
						format = "r:"
						markerformat = 'r.'
						i = i - 1
					plotHist(res,key,format,markerformat,width)
					leg.append(res['name'])
					leg.append(res['name'] + ' Mean')
				i = i + 1
		lgd=pl.legend(leg,bbox_to_anchor=(1.05, 1), loc=2)
		fig.savefig(path+'fig3_compare_llf_distributions_'+str(width)+'.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
	fig = figure()
	for res in reses:
		plot(res['EIC_aic'],'-+')
		#plot(EIC_bic,'-*')
		plot(res['EICE'],'o')
		plot([0,len(res['EICE'])],[np.mean(res['EICE']),np.mean(res['EICE'])],'--')
		plot([0,len(res['llf_train'])],[res['AIC'],res['AIC']],':')
	lgd=legend([a+' '+r['name'] for r in reses for a in ['AIC*','EIC*','EIC','AIC']],bbox_to_anchor=(1.05, 1), loc=2)
	fig.savefig(path+'compare_EIC_dist.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
	fig = pl.figure()
	i = 0
	for res in reses:
		pl.plot([i,i], [np.mean(res['EICE']),res['AIC']],'o')
		i = i + 1
	pl.plot([np.mean(res['EICE']) for res in reses],'b--')
	pl.plot([res['AIC'] for res in reses],'g:')
	leg = [r['name'] for r in reses]
	leg.extend(['EIC','AIC'])
	lgd=pl.legend(leg,bbox_to_anchor=(1.05, 1), loc=2)
	fig.savefig(path+'compare_EIC.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
