"""
.. module:: ni.model.net_sim
   :platform: Unix
   :synopsis: Runs a small, simulated network

.. moduleauthor:: Jacob Huth

The Net Simulator is divided into a Configuration, Net and a Result object.

After configuration of the network it can be instantiated by calling `Net(conf)` with a valid configuration `conf`. This creates eg. random connectivity so that the simulation with the same network can be repeated multiple times.

.. todo::
	
	Add options for random number generator seeds, so that the *exact* same trial can be run over and over again.

.. testcode::
	
	c = ni.model.net_sim.SimulationConfiguration()
	c.Nneur = 10
	net = ni.model.net_sim.Net(c)
	print net
	net.plot_firing_rates()

.. testoutput::
	
	'ni.model.net_sim' Simulation Setup
	Timerange: (250, 10250)
	 10 channels with firing rates: 
		[12.815928361, 29.6328550796, 19.9415819867, 13.6710936491, 20.242131795, 11.4661487294, 11.5071338947, 10.2727521514, 24.2587596858, 13.1497981307]
	Firing Rates plot


.. image:: _static/net_sim_firing_rates.png


.. testcode::

	for i in range(1,11):
	    print i
	    res1 = net.simulate()
	    res1.plot_firing_rates()

	plot(numpy.array([r.num_spikes_per_channel for r in net.results]))
	plot([0]*len(net.results))


.. image:: _static/net_sim_trials_firing.png

"""
import time
import random, math
import numpy as np
import pylab
import pandas as pd
import ni.tools.progressbar
import pickle
import create_splines as cs
import pointprocess
import ni.data.data

class SimulationConfiguration:
	"""
	Configures the simulation. The default values are:

		**Nneur**  = 100
		**sparse_coeff** = 0.1
		**Trial_Time**  = 1000
		**prior_epoch**  = 250
		**Ntrials**  = 10
		**Nsec**  = Ntrials*Trial_Time/1000
		**Ntime** = Nsec*1000
		**eps**  =0.1
		**frate_mu**  = 1.0/25.0
		**Nhist** = 50
		**output**  = False
		**rate_function**  = False

	"""
	def __init__(self):
		self.Nneur = 10
		self.sparse_coeff=0.1
		self.Trial_Time = 1000
		self.prior_epoch = 250
		self.Ntrials = 100
		#self.Nsec = self.Ntrials*self.Trial_Time/1000
		#self.Ntime=self.Nsec*1000
		self.eps =0.1
		self.frate_mu = 1.0/25.0
		self.Nhist=50
		self.output = False
		self.rate_function = False
	@property
	def Nsec(self):
		return self.Ntrials*self.Trial_Time/1000
	@property
	def Ntime(self):
		return self.Nsec*1000

	

class SimulationResult:
	"""
	Holds the results of a simulation
	"""
	def __init__(self):
		self.sim_name = "ni.model.net_sim"
		self.log_time_begin = time.time()
		self.config = {}
		self.spikes = pd.DataFrame([0])
	def stopTimer(self):
		"""
		stops the internal timer
		"""
		self.log_time_end = time.time()
		self.log_time_duration = self.log_time_end - self.log_time_begin
	def store(self, data):
		"""
		stores data in the container
		"""
		self.spikes = pd.DataFrame(data)
		spikes = np.where(data==1)
		self.spike_times = np.array([ spikes[0][np.where(spikes[1]==i)] for i in range(0,self.config.Nneur)])
		self.num_spikes_per_channel =  self.spikes.sum()
		self.num_spikes = self.num_spikes_per_channel.sum()
		self.num_channels = len(self.spikes.T)
		self.timerange = (0,len(self.spikes))
	def plot(self):
		"""
		plots the resulting spike train
		"""
		[pylab.plot(np.where(self.spikes[i]>0)[0], self.spikes[i][self.spikes[i]>0]+i,'|', markersize=12)  for i in range(0,len(self.spikes.T))]
		pylab.ylim(0, len(self.spikes.T)+1)
		return
	def plot_firing_rates(self):
		"""
		plots the resulting firing rate
		"""
		print "Firing Rates plot"
		for i in xrange(self.num_channels):
			pointprocess.plotGaussed(self.spikes[i],100)
	def plot_firing_rates_per_channel(self):
		"""
		plots the firing rate for each channel
		"""
		print "Firing Rates plot"
		pylab.plot(self.spikes.mean())
	def __str__(self):
		return "'" +self.sim_name + "' Simulation Result\n" + "Took " + str(round(self.log_time_duration,2)) + "s to compute.\nTimerange: "+str(self.timerange)+"\n" + str(int(self.num_spikes)) + " Spikes in " +str(self.num_channels)+ " channels: \n\t[" + ", ".join([str(int(s)) for s in self.num_spikes_per_channel]) + "]"
	@property
	def data(self):
		datas = []
		for i in range(self.config.Nneur): 
			datas.append(self.spikes[i][self.config.prior_epoch:].reshape((self.config.Ntrials,self.config.Trial_Time)))
		return ni.data.data.Data(np.transpose(np.array(datas), (1, 0, 2)))


class Net:
	"""
	The Net Simulator class. Use with an Configuration instance.
	"""
	def __init__(self,config):
		self.sim_name = "ni.model.net_sim"
		self.config = config
		self.frate= np.random.rand(self.config.Nneur)*0.2+0.4
		self.frate=-(1/self.config.frate_mu)*np.log(1-self.frate)+1
		self.frate_function = np.tile(self.frate,self.config.prior_epoch+self.config.Ntime)
		if self.config.rate_function:
			rate_splines = cs.create_splines_linspace(self.config.prior_epoch+self.config.Ntime, 10, 0)
			rate_function = np.matrix(np.random.rand(self.config.Nneur, rate_splines.shape[1])) * np.matrix(rate_splines.conj().transpose())
			self.frate_function = rate_function[:, :(self.config.prior_epoch+self.config.Ntime)] 
			self.frate_function = self.frate_function / (np.mean(self.frate_function))
			self.frate_function =  np.array([np.array(self.frate_function[i,:].flatten() * self.frate[i]) for i in xrange(self.config.Nneur)])
			self.frate_function =  np.reshape(self.frate_function,(self.config.Nneur,self.config.prior_epoch+self.config.Ntime)).transpose()
		self.logit_lam0=np.log(self.frate/1000 /(1-self.frate/1000))
		# now make coupling we will use a simple 
		# exponential decay function
		t = np.array(range(0,self.config.Nhist))
		# self histories
		self.Jself= np.zeros((self.config.Nneur,self.config.Nhist))
		for i in range(1,self.config.Nneur):
			self.Jself[i,] = 3*np.random.rand()*(np.exp(-t/20) -5*np.exp(-t/2))
			self.Jself[i,1]=-10
		# Interaction Matrix
		self.Jall=np.zeros((self.config.Nneur,self.config.Nneur,self.config.Nhist))
		J_beta=np.zeros((self.config.Nneur,self.config.Nneur))
		tau_beta=np.zeros((self.config.Nneur,self.config.Nneur))
		#A=full(sprand(Nneur,Nneur,sparse_coeff)) == Nneur*Nneur*sparse_coeff non-zero entries
		A_entries = np.random.rand(self.config.Nneur*self.config.Nneur*self.config.sparse_coeff)
		A = np.zeros((self.config.Nneur,self.config.Nneur))
		for a in A_entries:
			A[np.random.randint(self.config.Nneur),np.random.randint(self.config.Nneur)] = a
		J_beta = (A-0.5)*0.5
		tau_beta[np.nonzero(A)] = np.random.rand(tau_beta[np.nonzero(A)].shape[0])*10+5
		for i in range(0,self.config.Nneur):
			for j in range(0,self.config.Nneur):
				self.Jall[i,j,] = J_beta[i,j]*np.exp(np.array(range(1,self.config.Nhist+1))*-1/(self.config.eps+tau_beta[i,j]))
		for i in range(0,self.config.Nneur):
			self.Jall[i,i,] = np.zeros((1,1,self.config.Nhist))
		self.results = []
	def __str__(self):
		return "'" +self.sim_name + "' Simulation Setup\nTimerange: "+str((self.config.prior_epoch,self.config.Ntime+self.config.prior_epoch))+"\n " +str(self.config.Nneur)+ " channels with firing rates: \n\t[" + ", ".join([str(s) for s in self.frate]) + "]"
	def save(self,filename):
		"""
		saves itself to a file
		"""
		f = open(fileame, "wb")
		pickle.dump(self, f)
		f.close()
		return 1
	def load(self,filename):
		"""
		loads itself from a file
		"""
		f = open(fileame, "rb")
		tmp_dict = pickle.load(f)
		f.close()
		self.__dict__.update(tmp_dict) 
		return 1
	def simulate(self):
		"""
		Simulates the network
		"""
		result = SimulationResult()
		result.config = self.config
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		#%  Simuluate
		spikes_all=np.zeros((self.config.Ntime+self.config.prior_epoch,self.config.Nneur))

		# Initialize
		for t in range(0,self.config.prior_epoch):
			for neur in range(0,self.config.Nneur):
				if np.random.rand() < self.frate[neur]/1000:
					spikes_all[t,neur] = 1
		#% Now simulate the rest
		last_spike_time = 100*np.ones(self.config.Nneur)
		if self.config.output:
			ni.tools.progressbar.progress_init()
		for t in range(self.config.prior_epoch,self.config.Ntime+self.config.prior_epoch):
			if self.config.output:
				ni.tools.progressbar.progress(t,self.config.Ntime)
			last_spike_time = last_spike_time + 1
			last_spike_time_new = last_spike_time
			for neur in range(0,self.config.Nneur):
				#% we are doing time to last spike to keep stable
				#% self histories
				self_interact = 0
				if last_spike_time[neur] < 50:
					self_interact = self_interact+self.Jself[neur,last_spike_time[neur]]
				#% cross histories
				interact = 0
				for j in range(0,self.config.Nneur):
					if neur != j:
						if last_spike_time[j] < 50:
							interact = interact + self.Jall[neur,j,last_spike_time[j]]
				exp_logit_lam=np.exp(np.log( (self.frate_function[t,neur]/1000) / (1-self.frate_function[t,neur]/1000) ) + self_interact +interact)
				p=exp_logit_lam/(1+exp_logit_lam)
				if np.random.rand()<=p:
					spikes_all[t,neur] = 1
					last_spike_time_new[neur] = 0

			last_spike_time = last_spike_time_new
		if self.config.output:
			ni.tools.progressbar.progress_end()
		pylab.plot(sum((spikes_all==1)))
		pylab.savefig('spikes')
		pylab.close()
		pylab.plot(sum((spikes_all==1).transpose()))
		pylab.savefig('spikes_over_time')
		pylab.close()
		#pretty but probably slow:
		result.store(spikes_all)
		result.spikes = pd.DataFrame(spikes_all)
		spikes = np.where(spikes_all==1)
		spike_times = np.array([ spikes[0][np.where(spikes[1]==i)] for i in range(0,self.config.Nneur)])
		result.spike_times = pd.DataFrame(spike_times)
		result.stopTimer()
		self.results.append(result)
		return result
	# Plots:
	def plot_interaction(self):
		"""
		plots the interactions of the network
		"""
		print "Interaction plot"
		for i in range(0,self.config.Nneur):
			subplot(5,self.config.Nneur/5,i+1)
			p = pylab.plot(self.Jall[i,:,:].T)
			print i, self.Jall[i,:,:].sum()
			#p.set_cmap('hot')
			pylab.title(str(i))
	def plot_firing_rates(self):
		"""
		plots the intended firing rates
		"""
		print "Firing Rates plot"
		pylab.plot(self.frate)

def simulate(config):
	"""
	creates a network and simulates it.
	"""
	result = SimulationResult()
	result.config = config
	frate= np.random.rand(config.Nneur)*0.4+0.3
	frate=-(1/config.frate_mu)*np.log(1-frate)+1
	logit_lam0=np.log(frate/1000 /(1-frate/1000))

	# now make coupling we will use a simple 
	# exponential decay function
	
	t = np.array(range(0,config.Nhist))

	# self histories

	Jself= np.zeros((config.Nneur,config.Nhist))

	for i in range(1,config.Nneur):
		Jself[i,] = 3*np.random.rand()*(np.exp(-t/20) -5*np.exp(-t/2))
		Jself[i,1]=-10

	pylab.plot(Jself)

	pylab.savefig('Jself')
	pylab.close()

	# Interaction Matrix

	Jall=np.zeros((config.Nneur,config.Nneur,config.Nhist))
	J_beta=np.zeros((config.Nneur,config.Nneur))
	tau_beta=np.zeros((config.Nneur,config.Nneur))
	#A=full(sprand(Nneur,Nneur,sparse_coeff)) == Nneur*Nneur*sparse_coeff non-zero entries
	A_entries = np.random.rand(config.Nneur*config.Nneur*config.sparse_coeff)
	A = np.zeros((config.Nneur,config.Nneur))
	for a in A_entries:
		A[np.random.randint(config.Nneur),np.random.randint(config.Nneur)] = a

	J_beta = (A-0.5)*0.5
	tau_beta[np.nonzero(A)] = np.random.rand(tau_beta[np.nonzero(A)].shape[0])*10+5

	for i in range(0,config.Nneur):
		for j in range(0,config.Nneur):
			Jall[i,j,] = J_beta[i,j]*np.exp(np.array(range(1,config.Nhist+1))*-1/(config.eps+tau_beta[i,j]))

	for i in range(0,config.Nneur):
		Jall[i,i,] = np.zeros((1,1,config.Nhist))

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#%  Simuluate
	spikes_all=np.zeros((config.Ntime+config.prior_epoch,config.Nneur))

	# Initialize
	for t in range(0,config.prior_epoch):
		for neur in range(0,config.Nneur):
			if np.random.rand() < frate[neur]/1000:
				spikes_all[t,neur] = 1


	#% Now simulate the rest
	last_spike_time = 100*np.ones(config.Nneur)
	if config.output:
		ni.tools.progressbar.progress_init()
	for t in range(config.prior_epoch,config.Ntime+config.prior_epoch):
		if config.output:
			ni.tools.progressbar.progress(t,config.Ntime)
		last_spike_time = last_spike_time + 1
		last_spike_time_new = last_spike_time
		for neur in range(0,config.Nneur):
			#% we are doing time to last spike to keep stable
			#% self histories
			self_interact = 0
			if last_spike_time[neur] < 50:
				self_interact = self_interact+Jself[neur,last_spike_time[neur]]
			#% cross histories
			interact = 0
			for j in range(0,config.Nneur):
				if neur != j:
					if last_spike_time[j] < 50:
						interact = interact + Jall[neur,j,last_spike_time[j]]
			exp_logit_lam=np.exp(np.log( (frate[neur]/1000) / (1-frate[neur]/1000) ) + self_interact +interact)
			p=exp_logit_lam/(1+exp_logit_lam)
			if np.random.rand()<=p:
				spikes_all[t,neur] = 1
				last_spike_time_new[neur] = 0

		last_spike_time = last_spike_time_new
	if config.output:
		ni.tools.progressbar.progress_end()
	pylab.plot(sum((spikes_all==1)))
	pylab.savefig('spikes')
	pylab.close()

	pylab.plot(sum((spikes_all==1).transpose()))
	pylab.savefig('spikes_over_time')
	pylab.close()

	#pretty but probably slow:

	result.spikes = pd.DataFrame(spikes_all)

	spikes = np.where(spikes_all==1)
	spike_times = np.array([ spikes[0][np.where(spikes[1]==i)] for i in range(0,config.Nneur)])
	return result


