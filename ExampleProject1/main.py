import ni
from ni import View, StatCollector
from matplotlib.pyplot import plot, figure, imshow
import matplotlib.pyplot as pylab
import numpy as np
import ni.tools.strap as bootstrap

## Parameters:
trial_number = ni.data.monkey.available_trials[0]
condition = 0

eval_bootstrap_repetitions = 50
eval_trials = 50

simple_model_configuration = ni.model.ip.Configuration({
									'knots_rate': 20,
									'autohistory': True,
									'autohistory_2d':False,
									'crosshistory':False,
									'history_length':60})
complex_model_configuration = ni.model.ip.Configuration({
									'knots_rate': 20,
									'autohistory': True,
									'autohistory_2d':True,
									'crosshistory':True,
									'history_length':60})
## End of Parameters

if type(parameters) == list:
	if len(parameters) > 0:
		if parameters[0] in ni.data.monkey.available_trials:
			trial_number = parameters[0]
	if len(parameters) > 1:
		try:
			condition = int(parameters[1])
		except:
			pass
elif type(parameters) == dict:
	if "trial_number" in parameters:
		trial_number = parameters["trial_number"]
	if "condition" in parameters:
		condition = parameters["condition"]
	if "eval_bootstrap_repetitions" in parameters:
		eval_bootstrap_repetitions = parameters["eval_bootstrap_repetitions"]
	if "eval_trials" in parameters:
		eval_trials = parameters["eval_trials"]

identifier = str(trial_number)+"_"+str(condition)

all_data = ni.data.monkey.Data(trial_number).condition(condition)
data = all_data.trial(range(int(all_data.nr_trials/2)))
test_data = all_data.trial(range(int(all_data.nr_trials/2),all_data.nr_trials))

use_cells = range(all_data.nr_cells)
model_cells = range(all_data.nr_cells)

prototypes = StatCollector()
stats  = StatCollector()
path = _current_session.path
job_path = _current_job.path

def to_path(l):
	return "/".join([str(c) for c in sorted(l)])

results_titles = {
	'llf_test_model':'Loglikelihood on test data',
	'EIC':'negative EIC',
	'AIC':'negative AIC',
	'llf_boot':'Loglikelihood on bootstrap data',
	'llf_train':'Bootmodel loglikelihood on training data',
	'llf_test':'Bootmodel loglikelihood on test data',
	'llf_train_model':'Model loglikelihood on training data',
	'EIC bias':'EICE_bias',
	'EICE2':'EICE uncorrected',
	'pvalues':'pvalues',
	'beta':'beta',
	'BIC':'negative BIC'
}
results_invert = {
	'llf_test_model': 1,
	'EIC': -1,
	'AIC': -1,
	'llf_boot':1,
	'llf_test':1,
	'llf_train':1,
	'llf_train_model':1,
	'EICE_bias':-1,
	'EICE2':-1,
	'pvalues':1,
	'beta':1,
	'BIC':-1
}

models = {}
for cell in model_cells:
	models[cell] = {}
	model = ni.model.ip.Model(ni.model.ip.Configuration({'cell':cell, 'autohistory':False, 'crosshistory':[cell]}))
	model.name = "Model " +str(cell)
	models[cell][model.name] = model
	model = ni.model.ip.Model(ni.model.ip.Configuration({'cell':cell, 'crosshistory':[cell]}))
	model.name = "Model " +str(cell)+ "/"  +str(cell)
	models[cell][model.name] = model

for cell in model_cells:
	for c1 in use_cells:
		if c1 != cell:
			crosshistory = sorted([cell,c1])
			model = ni.model.ip.Model(ni.model.ip.Configuration({'cell':cell, 'crosshistory':crosshistory}))
			model.name = "Model " +str(cell)+ "/" +str("/".join([str(c) for c in crosshistory]))
			models[cell][model.name] = model
for cell in model_cells:
	for c1 in use_cells:
		if c1 != cell:
			for c2 in use_cells:
				if c2 != cell:
					crosshistory = sorted([cell,c1,c2])
					model = ni.model.ip.Model(ni.model.ip.Configuration({'cell':cell, 'crosshistory':crosshistory}))
					model.name = "Model " +str(cell)+ "/" +str("/".join([str(c) for c in crosshistory]))
					models[cell][model.name] = model


E1 = 'reshuffle_trials_'
E1b = 'reshuffle_time_'
E2 = 'simple_model_'
E3 = 'complex_model_'
Es = [E1,E1b,E2,E3]

job "E1.1. Evaluate Models for Trial Reshuffling" for cell in model_cells for model in range(len(models[0])):
	m = models[cell][sorted(models[cell].keys())[model]]
	with View(job_path + '_results.html') as view:
		b = bootstrap.bootstrap_trials(50,m,data,test_data=[test_data])
		stats.addNode(m.name,b)
		for (dim, name) in [('EIC','EIC')]:
			with view.figure("Cells/tabs/Cell "+str(cell)+"/tabs/"+str(name)):
				plot(stats.get(dim))
		for k in stats.keys():
			with view.figure("Cells/tabs/Cell "+str(cell)+"/tabs/Betas/tabs/"+str(k)):
				for b in stats.filter(k).get('beta'): 
					plot(b)
				for b in stats.filter(k).get('boot_betas'): 
					for bb in b:
						plot(bb,'--')
		view.render(job_path + '_results.html')
		stats.save(path + identifier  + E1 + '_' + str(cell) + "_" + str(m.name).replace("/","_") + '_stats.stat')
job "E1.2. Saving Data":
	require previous
	stats.load_glob(path + identifier + E1 + '_*_Model*_stats.stat')
	stats.save(path + identifier + E1 + '_all_models.stat')

job "E1b.1. Evaluate Models for Trial Reshuffling" for cell in model_cells for model in range(len(models[0])):
	m = models[cell][sorted(models[cell].keys())[model]]
	with View(job_path + '_results.html') as view:
		b = bootstrap.bootstrap_time(50,m,data,test_data=[test_data])
		stats.addNode(m.name,b)
		for (dim, name) in [('EIC','EIC')]:
			with view.figure("Cells/tabs/Cell "+str(cell)+"/tabs/"+str(name)):
				plot(stats.get(dim))
		for k in stats.keys():
			with view.figure("Cells/tabs/Cell "+str(cell)+"/tabs/Betas/tabs/"+str(k)):
				for b in stats.filter(k).get('beta'): 
					plot(b)
				for b in stats.filter(k).get('boot_betas'): 
					for bb in b:
						plot(bb,'--')
		view.render(job_path + '_results.html')
		stats.save(path + identifier  + E1b + '_' + str(cell) + "_" + str(m.name).replace("/","_") + '_stats.stat')
job "E1b.2. Saving Data":
	require previous
	stats.load_glob(path + identifier + E1b + '_*_Model*_stats.stat')
	stats.save(path + identifier + E1b + '_all_models.stat')



job "E2.1. Fit All" for cell in model_cells:
	with View(job_path + '_results.html') as view:
		simple_model_configuration.cell = cell
		model = ni.model.ip.Model(simple_model_configuration)
		x = model.x(data)
		dm = model.dm(data)
		fm = model.fit(data)
		fm.save(path +identifier+'_model_' +str(cell) + '.mdl')
		view.add("Model/",fm)
		test_x = model.x(test_data)
		test_dm = model.dm(test_data)
		c_train = model.compare(x,model.predict(fm.beta,dm),nr_trials=data.nr_trials)
		c_test = model.compare(test_x,model.predict(fm.beta,test_dm),nr_trials=test_data.nr_trials)
		stats.addToNode([cell,'beta'],{'beta':fm.beta})
		if 'pvalues' in fm.statistics:
			stats.addToNode([cell,'pvalues'],{'pvalues':fm.statistics['pvalues']})
		stats.addToNode([cell,'training'],{'LogLikelihood':c_train['LogLikelihood'],'ll':c_train['ll']})
		stats.addToNode([cell,'test'],{'LogLikelihood':c_test['LogLikelihood'],'ll':c_test['ll']})
		view.add("#1/Cells/tabs/"+str(cell)+"/beta",fm.beta)
		view.add("Stats/",stats)
		stats.save(path +identifier+ E2+ '_'+str(cell)+'_stats.stat')
job "E2.2. Generate Data" for bootstrap_repetition in range(eval_bootstrap_repetitions):
	require previous
	with View(job_path + '_results.html') as view:
		crossmodels = [ni.tools.pickler.load(path +identifier+E2+ '_model_'+str(cell)+'.mdl') for cell in model_cells]
		print bootstrap_repetition, " Bootstrap Sample"
		gs = []
		gs_other = []
		for t in range(eval_trials):
			(spikes,p) = ni.model.ip_generator.generate(crossmodels)
			gs.append(spikes.transpose())
		d = ni.model.ip.Data(np.array(gs))
		d.to_pickle(path+identifier+E2+'_'+str(bootstrap_repetition)+'_data.pkl')
		view.add("Data",d)
job "E2.3. Evaluate Models" for cell in model_cells for model in range(len(models[0])):
	require previous
	m = models[cell][sorted(models[cell].keys())[model]]
	with View(job_path + '_results.html') as view:
		data_files = [path+identifier+E2+"_"+str(b)+"_data.pkl" for b in range(eval_bootstrap_repetitions)]
		bootstrap_data = ni.data.data.merge([ni.data.data.Data(d) for d in data_files],dim='Bootstrap Sample')
		print bootstrap_data.shape(level='Bootstrap Sample'), "Bootstrap Repetitions"
		all_data = ni.data.monkey.Data().condition(0)
		data = all_data.trial(range(int(all_data.nr_trials/2)))
		test_data = all_data.trial(range(int(all_data.nr_trials/2),all_data.nr_trials))
		
		b = bootstrap.bootstrap(bootstrap_data.shape(level='Bootstrap Sample'),m,data,test_data=[test_data],bootstrap_data=bootstrap_data)
		stats.addNode(m.name,b)
		for (dim, name) in [('EIC','EIC')]:
			with view.figure("Cells/tabs/Cell "+str(cell)+"/tabs/"+str(name)):
				plot(stats.get(dim))
		for k in stats.keys():
			with view.figure("Cells/tabs/Cell "+str(cell)+"/tabs/Betas/tabs/"+str(k)):
				for b in stats.filter(k).get('beta'): 
					plot(b)
				for b in stats.filter(k).get('boot_betas'): 
					for bb in b:
						plot(bb,'--')
		stats.save(path + identifier + E2+'_' + str(cell) + "_" + str(m.name).replace("/","_") + '_stats.stat')
job "E2.4. Saving Data":
	require previous
	stats.load_glob(path + identifier +E2+ '_*_Model*_stats.stat')
	stats.save(path + identifier +E2+ '_all_models.stat')


job "E3.1. Fit All" for cell in model_cells:
	with View(job_path + '_results.html') as view:
		complex_model_configuration.cell = cell
		model = ni.model.ip.Model(complex_model_configuration)
		x = model.x(data)
		dm = model.dm(data)
		fm = model.fit(data)
		fm.save(path +identifier+'_model_' +str(cell) + '.mdl')
		view.add("Model/",fm)
		test_x = model.x(test_data)
		test_dm = model.dm(test_data)
		c_train = model.compare(x,model.predict(fm.beta,dm),nr_trials=data.nr_trials)
		c_test = model.compare(test_x,model.predict(fm.beta,test_dm),nr_trials=test_data.nr_trials)
		stats.addToNode([cell,'beta'],{'beta':fm.beta})
		if 'pvalues' in fm.statistics:
			stats.addToNode([cell,'pvalues'],{'pvalues':fm.statistics['pvalues']})
		stats.addToNode([cell,'training'],{'LogLikelihood':c_train['LogLikelihood'],'ll':c_train['ll']})
		stats.addToNode([cell,'test'],{'LogLikelihood':c_test['LogLikelihood'],'ll':c_test['ll']})
		view.add("#1/Cells/tabs/"+str(cell)+"/beta",fm.beta)
		view.add("Stats/",stats)
		stats.save(path +identifier+E3+ '_'+str(cell)+'_stats.stat')
job "E3.2. Generate Data" for bootstrap_repetition in range(eval_bootstrap_repetitions):
	require previous
	with View(job_path + '_results.html') as view:
		crossmodels = [ni.tools.pickler.load(path +identifier+E3+ '_model_'+str(cell)+'.mdl') for cell in model_cells]
		print bootstrap_repetition, " Bootstrap Sample"
		gs = []
		gs_other = []
		for t in range(eval_trials):
			(spikes,p) = ni.model.ip_generator.generate(crossmodels)
			gs.append(spikes.transpose())
		d = ni.model.ip.Data(np.array(gs))
		d.to_pickle(path+identifier+E3+'_'+str(bootstrap_repetition)+'_data.pkl')
		view.add("Data",d)
job "E3.3. Evaluate Models" for cell in model_cells for model in range(len(models[0])):
	require previous
	m = models[cell][sorted(models[cell].keys())[model]]
	with View(job_path + '_results.html') as view:
		data_files = [path+identifier+E3+"_"+str(b)+"_data.pkl" for b in range(eval_bootstrap_repetitions)]
		bootstrap_data = ni.data.data.merge([ni.data.data.Data(d) for d in data_files],dim='Bootstrap Sample')
		print bootstrap_data.shape(level='Bootstrap Sample'), "Bootstrap Repetitions"
		all_data = ni.data.monkey.Data().condition(0)
		data = all_data.trial(range(int(all_data.nr_trials/2)))
		test_data = all_data.trial(range(int(all_data.nr_trials/2),all_data.nr_trials))
		
		b = bootstrap.bootstrap(bootstrap_data.shape(level='Bootstrap Sample'),m,data,test_data=[test_data],bootstrap_data=bootstrap_data)
		stats.addNode(m.name,b)
		for (dim, name) in [('EIC','EIC')]:
			with view.figure("Cells/tabs/Cell "+str(cell)+"/tabs/"+str(name)):
				plot(stats.get(dim))
		for k in stats.keys():
			with view.figure("Cells/tabs/Cell "+str(cell)+"/tabs/Betas/tabs/"+str(k)):
				for b in stats.filter(k).get('beta'): 
					plot(b)
				for b in stats.filter(k).get('boot_betas'): 
					for bb in b:
						plot(bb,'--')
		stats.save(path + identifier +E3+ '_' + str(cell) + "_" + str(m.name).replace("/","_") + '_stats.stat')
job "E3.4. Saving Data":
	require previous
	stats.load_glob(path + identifier +E3+ '_*_Model*_stats.stat')
	stats.save(path + identifier +E3+ '_all_models.stat')

job "Results 1. Plotting" for Ei in range(len(Es)):
	require "E1.2. Saving Data"
	require "E1b.2. Saving Data"
	require "E2.4. Saving Data"
	require "E3.4. Saving Data"
	E = Es[Ei]
	stats = StatCollector() 
	titles = results_titles
	invert = results_invert
	with View(job_path + '_results.html') as view:
		stats.load(path + identifier + E + '_all_models.stat')
		for k in stats.keys():
			stats.stats[k]['BIC'] = stats.stats[k]['statistics']['bic'] # get bic from 'statistics' into the statcollector
		all_l = []
		all_model_points = []
		for cell in model_cells:
			for dim in ['llf_test_model','EIC','AIC','llf_boot','llf_train','llf_train_model','llf_test','EICE_bias','EICE2','BIC']:
				print dim
				with view.figure("/tabs/"+E+"/Cells/tabs/Cell "+str(cell)+"/tabs/Trees From 2nd Level/tabs/"+dim,figsize=(7,10)):
					stats.filter('Model '+str(cell)).plotTree(dim)
					pylab.title(titles[dim])
				with view.figure("/tabs/"+E+"/Cells/tabs/Cell "+str(cell)+"/tabs/Trees From 2nd Level Inverse/tabs/"+dim,figsize=(7,10)):
					stats.filter('Model '+str(cell)).plotTree(dim,right_to_left=True)
					pylab.title(titles[dim])
			for dim in ['EICE','llf_boot','llf_train','llf_test','EICE_bias','EICE2','EICE_bias_uncorrected']:
				with view.figure("/tabs/"+E+"/Cumulative Mean/tabs/Cell "+str(cell)+"/tabs/Cumulative Mean/tabs/"+dim,figsize=(4,4)):
					for e in stats.filter(mp).get(dim):
						plot(np.cumsum(e)/(np.arange(len(e))+1),'k',alpha=0.5)
					pylab.xlabel("Number of Samples")
				with view.figure("/tabs/"+E+"/Cumulative Mean/tabs/Cell "+str(cell)+"/tabs/Cumulative Mean/tabs/"+dim+" centered",figsize=(4,4)):
					for e in stats.filter(mp).get(dim):
						plot((np.cumsum(e)/(np.arange(len(e))+1) - np.mean(e))[:-1],'k',alpha=0.5)
					pylab.xlabel("Number of Samples in mean")
					pylab.ylabel("Distance to mean")
		view.render(path+E+identifier+'_results.html')

job "Results 2. Plotting":
	require "E1.2. Saving Data"
	require "E1b.2. Saving Data"
	require "E2.4. Saving Data"
	require "E3.4. Saving Data"
	titles = results_titles
	invert = results_invert
	with View(job_path + '_bias_results.html') as view:
		stats1 = ni.StatCollector()
		stats1b = ni.StatCollector()
		stats2 = ni.StatCollector()
		stats3 = ni.StatCollector()
		stats1.load(path + identifier + E1 + '_all_models.stat')
		stats1b.load(path + identifier + E1b + '_all_models.stat')
		stats2.load(path + identifier + E2 + '_all_models.stat')
		stats3.load(path + identifier + E3 + '_all_models.stat')
		for k in stats1.keys():
			stats1.stats[k]['BIC'] = stats1.stats[k]['statistics']['bic'] # get bic from 'statistics' into the statcollector
		all_l = []
		all_model_points = []
		dim = "EIC"
		N = use_cells
		cell_gain_1 = []
		cell_gain_1b = []
		cell_gain_2 = []
		cell_gain_3 = []
		cell_gain_aic = []
		cell_gain_bic = []
		cell_gain_ll_test = []
		cell_gain_ll_train = []
		for cell in model_cells:
			mp = "Model "+str(cell)+"/"
			gain_1 = np.zeros(len(N))
			gain_1b = np.zeros(len(N))
			gain_2 = np.zeros(len(N))
			gain_3 = np.zeros(len(N))
			gain_aic = np.zeros(len(N))
			gain_bic = np.zeros(len(N))
			gain_ll_test = np.zeros(len(N))
			gain_ll_train = np.zeros(len(N))
			for i in N:
				print mp + to_path([cell,i])
				if i == cell:
					if mp + to_path([cell]) in stats1.keys():
						gain_ll_test[i] = np.mean(stats1.filter(mp + to_path([cell])).get('llf_test_model')[0])
						gain_ll_train[i] = np.mean(stats1.filter(mp + to_path([cell])).get('llf_train_model')[0])
				else:
					if mp + to_path([cell,i]) in stats1.keys():
						gain_ll_test[i] = np.mean(stats1.filter(mp + to_path([cell,i])).get('llf_test_model')[0])
						gain_ll_train[i] = np.mean(stats1.filter(mp + to_path([cell,i])).get('llf_train_model')[0])
				if mp + to_path([cell,i]) in stats1.keys():
					gain_aic[i] = np.mean(stats1.filter(mp + to_path([cell])).get('AIC')[0]) - np.mean(stats1.filter(mp + to_path([cell,i])).get('AIC')[0])
					gain_bic[i] = np.mean(stats1.filter(mp + to_path([cell])).get('BIC')[0]) - np.mean(stats1.filter(mp + to_path([cell,i])).get('BIC')[0])
				if mp + to_path([cell,i]) in stats1.keys():
					gain_1[i] = np.mean(stats1.filter(mp + to_path([cell])).get(dim)[0]) - np.mean(stats1.filter(mp + to_path([cell,i])).get(dim)[0])
				if mp + to_path([cell,i]) in stats1b.keys():
					gain_1b[i] = np.mean(stats1b.filter(mp + to_path([cell])).get(dim)[0]) - np.mean(stats1b.filter(mp + to_path([cell,i])).get(dim)[0])
				if mp + to_path([cell,i]) in stats2.keys():
					gain_2[i] = np.mean(stats2.filter(mp + to_path([cell])).get(dim)[0]) - np.mean(stats2.filter(mp + to_path([cell,i])).get(dim)[0])
				if mp + to_path([cell,i]) in stats3.keys():
					gain_3[i] = np.mean(stats3.filter(mp + to_path([cell])).get(dim)[0]) - np.mean(stats3.filter(mp + to_path([cell,i])).get(dim)[0])
			gain_ll_test = gain_ll_test - gain_ll_test[cell]
			gain_ll_train = gain_ll_train - gain_ll_train[cell]
			with view.figure("Gain/tabs/"+str(cell)+"/plot",figsize=(7,4)):
				plot(gain_1,'k')
				plot(gain_1b,'k',alpha=0.5)
				plot(gain_2,'k--')
				plot(gain_3,'k:')
				plot(gain_aic,'k--',alpha=0.5)
				plot(gain_bic,'k:',alpha=0.5)
				pylab.legend(["Trial reshuffled","Time reshuffled","simple model","complex model","AIC","BIC","Test ll"])
			with view.figure("Likelihoods/tabs/"+str(cell)+"/plot",figsize=(7,4)):
				plot(gain_ll_test,'k--')
				plot(gain_ll_train,'k')
				pylab.legend(["ll test","ll train"])
			cell_gain_1.append(gain_1)
			cell_gain_1b.append(gain_1b)
			cell_gain_2.append(gain_2)
			cell_gain_3.append(gain_3)
			cell_gain_aic.append(gain_aic)
			cell_gain_bic.append(gain_bic)
			cell_gain_ll_test.append(gain_ll_test)
			cell_gain_ll_train.append(gain_ll_train)
		def plotConnections(n,a):
			if np.max(a) > 0:
				a[a < 0] = 0
				ni.tools.plot.plotConnections(n,(a)/(np.max(a)))
			else:
				ni.tools.plot.plotConnections(n,0.01*np.ones((a.shape[0],a.shape[0])))
		for (dim,c) in [('EIC Trials reshuffle', cell_gain_1),('EIC Time reshuffle', cell_gain_1b),('EIC simple', cell_gain_2),('EIC complex', cell_gain_3),('AIC', cell_gain_aic),('BIC', cell_gain_bic),('ll test', cell_gain_ll_test),('ll train', cell_gain_ll_train)]:
			view.add("Number of connections/table/"+dim+"/pos",len(c) - (np.sum(np.array(c)<=0)-model_cells) )
			view.add("Number of connections/table/"+dim+"/neg",np.sum(np.array(c)<=0)-model_cells)
			with view.figure("Connections/tabs/"+dim+"/plot",figsize=(7,7)):
				plotConnections(len(N),np.array(c))
			with view.figure("Graphs/tabs/"+dim+"/plot",figsize=(7,7)):
				ni.tools.plot.plotNetwork(np.array(c))
			view.add("Connections/tabs/"+dim+"/table",np.array(c).tolist())
		view.render(path+E+identifier+'_results_2.html')
