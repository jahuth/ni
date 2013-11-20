import ni
from ni import View, StatCollector
from matplotlib.pyplot import plot, figure, imshow
import matplotlib.pyplot as pylab
import numpy as np
import ni.tools.strap as bootstrap
import glob

## Parameters:
trial_number = ni.data.monkey.available_trials[0]
condition = 0

eval_bootstrap_repetitions = 50
eval_trials = 50

knot_rates = [3,7,10,15,20,30,50,100,150,200]

E2_path = "ExampleProject1/sessions/session1/*simple*_data.pkl"
E3_path = "ExampleProject1/sessions/session1/*complex*_data.pkl"

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
model_cells = [0]#range(all_data.nr_cells)

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
	'EICE_bias':'EICE_bias',
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
	for knot_rate in knot_rates:
		model = ni.model.ip.Model(ni.model.ip.Configuration({'cell':cell, 'knots_rate': knot_rate, 'adaptive_rate':False, 'crosshistory':False}))
		model.name = "Model " +str(cell)+ "/rate/" + str(knot_rate)
		models[cell][model.name] = model
		model = ni.model.ip.Model(ni.model.ip.Configuration({'cell':cell, 'knots_rate': knot_rate, 'adaptive_rate':True, 'crosshistory':False}))
		model.name = "Model " +str(cell)+ "/adaptive/" + str(knot_rate)
		models[cell][model.name] = model

E1 = 'trial_reshuffle_'
E1b = 'time_reshuffle_'
E2 = 'simple_model_'
E3 = 'complex_model_'

Es = [E1, E1b, E2, E3]



job "E1.1. Evaluate Models for Trial Reshuffling" for cell in model_cells for model in range(len(models[0])):
	m = models[cell][sorted(models[cell].keys())[model]]
	with View(job_path + '_results.html') as view:
		b = bootstrap.bootstrap_trials(50,m,data,test_data=[test_data],shuffle=True)
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


job "E2.3. Evaluate Models" for cell in model_cells for model in range(len(models[0])):
	require previous
	m = models[cell][sorted(models[cell].keys())[model]]
	with View(job_path + '_results.html') as view:
		data_files = glob.glob(E2_path)
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


job "E3.3. Evaluate Models" for cell in model_cells for model in range(len(models[0])):
	require previous
	m = models[cell][sorted(models[cell].keys())[model]]
	with View(job_path + '_results.html') as view:
		data_files = glob.glob(E3_path)
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
	with View(job_path + '_results_' + E + 'plots.html') as view:
		stats.load(path + identifier + E + '_all_models.stat')
		for k in stats.keys():
			stats.stats[k]['BIC'] = stats.stats[k]['statistics']['bic'] # get bic from 'statistics' into the statcollector
		statsr = stats.rename_value_to_tree()
		for i in range(model_cells):
			for dim in ['EIC','AIC','BIC','EICE2','llf_test_model','llf_train_model']:
				with view.figure('Cells/tabs/'+str(i)+'/tree/'+dim):
					statsr.filter('Model 0').plotTree(dim)
		view.render(path+'results_'+E+identifier+'.html')
