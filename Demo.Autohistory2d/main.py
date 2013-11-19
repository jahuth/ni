import ni
from pylab import *


job "demo graphic":
	data = ni.data.monkey.Data(condition=0,cell=0)
	model = ni.model.ip.Model({'autohistory_2d':True})
	dm = model.dm(data)
	fm = model.fit(data)
	figsize(10,2)
	with ni.View("autohistory_2d_demo.html") as view:
		view.add("/#0/","""
			<h1>Second order history kernels</h1>
			The second order history kernel is computed with respect to two spikes and the distance between them.
			Two sets of splines are multiplied to get a set of matrizes.
			The rows of the matrix correspond to a specific distance of the last second spikes, the columns to time after the second spike.
			""")
		with view.figure('/#1/table/#1 spikes'):
			plot(data.as_series()[1000:4000])
		with view.figure('/#1/table/#2 autohistory'):
			for i in range(1,4):
				plot(dm[1000:4000,i])
			title('autohistory')
		with view.figure('/#1/table/#3 very close spikes'):
			for i in range(4,20,4):
				plot(dm[1000:4000,i])
				title('very close spikes')
		with view.figure('/#1/table/#4 close spikes'):
			for i in range(5,20,4):
				plot(dm[1000:4000,i])
				title('close spikes')
		with view.figure('/#1/table/#5 medium distance spikes'):
			for i in range(6,20,4):
				plot(dm[1000:4000,i])
				title('medium distance spikes')
		with view.figure('/#1/table/#6 large distance spikes'):
			for i in range(7,20,4):
				plot(dm[1000:4000,i])
				title('large distance spikes')
		figsize(10,10)
		with view.figure('/#1/table/#7 2d matrizes'):
			splines = model.last_generated_design.components[1].getSplines()
			for i in range(16):
				subplot(4,4,i+1)
				imshow(splines[i],interpolation='nearest')
		prot_figures = fm.plot_prototypes()
		prots = fm.prototypes()
		view.savefig("/#2/Fitted Component/table/#1/ single components",fig=prot_figures['autohistory2d'])
		view.savefig("/#2/Fitted Component/table/#2 Sum/",fig=prot_figures['autohistory2d_sum'])
		with view.figure("/#2/Fitted Component/table/#3 Sum+autohistory/"):
			plot(sum(prots['autohistory2d'],0) + prots['autohistory'],'g:')
			plot(prots['autohistory'],'b')
		with view.figure("/#2/Fitted Component/table/#4 Sum+autohistory/"):
			imshow(sum(prots['autohistory2d'],0) + prots['autohistory'],interpolation='nearest')



job "burst vs. regular":
	import ni
	with ni.View("_bursts3.html") as v:
		p = 0.01
		interval_burst = 5
		l = 2000
		bursts = 10
		burst_length_ratio = 0.5
		for burst_length_ratio in [0.1,0.2,0.5,0.9,1]:
			print p
			burst_length = (l/bursts)/2
			data_bursting = np.zeros(l)
			for b in range(0,l,l/bursts):
				data_bursting[range(b+interval_burst,b+burst_length,interval_burst)] = 1
			data_regular = np.zeros(l)
			interval_regular = interval_burst#int(l/sum(data_bursting))
			data_regular[range(interval_regular,l,interval_regular)] = 1
			for i in range(int(l*p)):
				data_bursting[randint(l)] = 1
			for i in range(int(l*p)):
				data_regular[randint(l)] = 1
			model = ni.model.ip.Model({'rate':False,'autohistory_2d':False, 'knot_number': 3, 'history_length':60})
			fm_regular = model.fit(ni.Data(data_regular))
			fm_bursting = model.fit(ni.Data(data_bursting))
			#model_2 = ni.model.ip.Model({'rate':False,'autohistory_2d':True, 'autohistory':False, 'knot_number': 3, 'history_length':30})
			#fm_2_regular = model_2.fit(ni.Data(data_regular))
			#fm_2_bursting = model_2.fit(ni.Data(data_bursting))
			prefix = "/tabs/"+str(burst_length_ratio)
			print prefix
			model_3 = ni.model.ip.Model({'rate':False,'autohistory_2d':True, 'autohistory':True, 'knot_number': 3, 'history_length':60})
			fm_3_regular = model_3.fit(ni.Data(data_regular))
			fm_3_bursting = model_3.fit(ni.Data(data_bursting))
			v.add(prefix+"/tabs/Regular/",fm_regular)
			v.add(prefix+"/tabs/Bursting/",fm_bursting)
			#view.add(prefix+"/tabs/Regular 2/",fm_2_regular)
			#view.add(prefix+"/tabs/Bursting 2/",fm_2_bursting)
			v.add(prefix+"/tabs/Regular 3/",fm_3_regular)
			v.add(prefix+"/tabs/Bursting 3/",fm_3_bursting)


job "Test Autohistory":
	view = ni.View()
	l = 2000
	data = np.zeros(l)
	interval = 60
	data[range(interval,l,interval)] = 1
	for i in range(int(l*0.001)):
		data[randint(l)] = 1
	data = ni.Data(data)
	model = ni.model.ip.Model({'rate':False,'autohistory_2d':True, 'knot_number': 6, 'history_length':90})
	fm = model.fit(data)
	prot = fm.getPrototypes
	for p in prot:
		with view.figure("Prototypes/tabs/"+str(p)):	
			plot(prot[p])
	with view.figure("positives & negatives"):
		plot(sum(dm[where(data.getFlattend()==1)[0],:]/sum(where(data.getFlattend()==1)),0))
		plot(sum(dm[where(data.getFlattend()==0)[0],:]/sum(where(data.getFlattend()==0)),0))
		xlabel('Parameter')
	view.render("_autohistory.html")

job "dummy 2":
	import ni
	view = ni.View()
	for p in [0.001,0.002,0.005,0.01,0.02,0.05]:
	    l = 2000
	    data = np.zeros(l)
	    interval = 20
	    data[range(interval,l,interval)] = 1
	    for i in range(int(l*p)):
	    	data[randint(l)] = 1
	    data = ni.Data(data)
	    model = ni.model.ip.Model({'rate':False,'autohistory_2d':False, 'knot_number': 6, 'history_length':90})
	    dm = model.dm(data)
	    fm = model.fit(data)
	    prot = fm.plotPrototypes()
	    for pr in prot:
	    	view.savefig("/tabs/"+str(p)+"/#2/Prototypes/tabs/"+str(pr), fig = prot[pr])
	    with view.figure("/tabs/"+str(p)+"/#1/positives & negatives"):
	    	plot(sum(dm[where(data.getFlattend()==1)[0],:]/sum(where(data.getFlattend()==1)),0))
	    	plot(sum(dm[where(data.getFlattend()==0)[0],:]/sum(where(data.getFlattend()==0)),0))
	    	xlabel('Parameter')
	view.render("_autohistory.html")