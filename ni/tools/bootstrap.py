import numpy as np

likelihood_Fun = lambda y,x,mu:  (-1)*(size(y)/2) * np.log(2*np.pi*x**2) - 1/((x**2)*(y-mu)**2)
"""
Calculates the likelihood for a binary vector and a predicted firing rate

.. math::
	
	-(size(y)/2) \cdot log(2 \cdot \pi \cdot x^2) - (1/x^2 \cdot (y-mu)^2)


"""

def evaluate(Model, Data, bootstrap_repetitions, return_all=False):
	"""
	Executes a certain number of bootstrap repetitions to calculate the bias of the likelihood he model computes

		**Model**

			A model object that is capable of loglikelihood estimation

		**Data**

			Data that is to be reshuffled. A bootstrap sample is drawn from this Data of the same length with each Element of Data being equally probable of being included.

		**bootstrap_repetitions**

			Number of repetitions

		**return_all**

			Default: False. Whether an array of all bootstrap biases should be returned or just the mean.

	Example::

		import ni.tools.bootstrap
		reload(ni.tools.bootstrap)
		import ni.model.pointprocess
		reload(ni.model.pointprocess)
		p1 = np.array([ni.model.pointprocess.createPoisson(0.1,1000).getCounts() for i in range(0,10)])
		p2 = np.array([ni.model.pointprocess.createPoisson(sin(numpy.array(range(0,200))*0.01)*0.5- 0.2,1000).getCounts() for i in range(0,10)])

		m1 = ni.model.pointprocess.SimpleFiringRateModel()

		ni.tools.bootstrap.evaluate(m1,p1,10000)
	
	Or to see the effect of increasing bootstrap size::
		
		[plot(np.cumsum(ni.tools.bootstrap.evaluate(m1,p1,1000,return_all=True))/range(1,1001)) for i in range(0,10)]

	.. image:: _static/increasing_bootstrap.png

	"""
	bias = np.zeros(bootstrap_repetitions)
	for i in range(0,bootstrap_repetitions):
		BootData = Data[np.array(np.floor(np.random.rand(np.size(Data,0))*np.size(Data,0)),int),:] # 2d Case
		#BootData = Data[np.array(np.floor(np.random.rand(size(Data))*size(Data)),int)] # One-D Case? -> useful?
		model = Model.fit(BootData)
		prediction = model.predict(Data)
		bias[i] = np.sum(model.loglikelihood(BootData,prediction)) - np.sum(model.loglikelihood(Data,prediction))
	if return_all:
		return bias
	return np.mean(bias)




def bootstrap(bootstrap_repetitions,model,data,other_data):
	EICE = []
	EICE_llf_train = []
	EICE_llf_test = []
	EIC_aic = []
	EIC_bic = []
	actual_fit = model.fit(data)
	for boot_rep in range(bootstrap_repetitions):
		print "####################### BOOTSTRAPPING ",model.name," ",boot_rep," Evaluating Cell ",cell," Reduced Model with threshold: ",threshold," #####################"
		bootstrap_trials = np.array(np.floor(np.random.rand(int(np.ceil(data.nr_trials)))*data.nr_trials),int)
		boot_data = data.condition(0).cell(cell).trial(bootstrap_trials)
		boot_data.other_spikes = other_data.condition(0).trial(bootstrap_trials)
		boot_fit = model.fit(boot_data)


		print "####################### BOOTSTRAPPING ",model.name," ",boot_rep," Evaluating Cell ",cell," Reduced Model with threshold: ",threshold," #####################"
		boot_prediction = boot_fit.predict(fit_data)
		EICE_llf_train.append(boot_fit.fit.fit.llf)
		EICE_llf_test.append(boot_prediction['LogLikelihood'])
		EICE.append(-2*actual_fit.fit.fit.llf + 2*(boot_fit.fit.fit.llf - boot_prediction['LogLikelihood']))
		EIC_aic.append(boot_fit.fit.fit.aic)
		EIC_bic.append(boot_fit.fit.fit.bic)
	return {'AIC':actual_fit.fit.fit.aic,'EICE':EICE, 'EICE_llf_train':EICE_llf_train, 'EICE_llf_test':EICE_llf_test, 'EIC_aic':EIC_aic, 'EIC_bic':EIC_bic }

def plotBootstrap(res,path):
	fig = figure()
	plot(res['EICE_llf_train'],'-o')
	plot(res['EICE_llf_test'],'-*')
	fig.savefig(path+'Likelihoods.png')
	fig = figure()
	plot(res['EIC_aic'],'-+')
	#plot(EIC_bic,'-*')
	plot(res['EICE'],'o')
	plot([0,len(res['EICE'])],[np.mean(res['EICE']),np.mean(res['EICE'])],'--')
	plot([0,len(res['EICE_llf_train'])],[res['AIC'],res['AIC']],':')
	legend(['AIC','EIC','E[EIC]'])
	fig.savefig(path+'EIC.png')

def plotCompareBootstrap(reses,path):
	fig = figure()
	for res in reses:
		plot(res['EICE_llf_train'],'-o')
		plot(res['EICE_llf_test'],'-*')
	legend(['LL Training','LL Test']*len(reses))
	fig.savefig(path+'compare_Likelihoods.png')
	fig = figure()
	for res in reses:
		plot(res['EIC_aic'],'-+')
		#plot(EIC_bic,'-*')
		plot(res['EICE'],'o')
		plot([0,len(res['EICE'])],[np.mean(res['EICE']),np.mean(res['EICE'])],'--')
		plot([0,len(res['EICE_llf_train'])],[res['AIC'],res['AIC']],':')
	legend(['AIC','EIC (dist)','EIC','AIC']*len(reses))
	fig.savefig(path+'compare_EIC.png')