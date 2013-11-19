import ni
#models = [ni.tools.pickler.load("xmod/101a03_0_model_"+str(i)+".mdl") for i in range(11)]
#models = [ni.tools.pickler.load("xmod2/101a03_0_model_"+str(i)+".mdl") for i in range(11)]
data = ni.data.monkey.Data(condition=0,trial=range(20))
models = []
for i in range(data.nr_cells):
	model = ni.model.ip.Model({'cell': i, 'autohistory': True, 'autohistory_2d':True})
	models.append(model.fit(data))
r = ni.model.ip_generator.generate(models)
p = []
data = ni.data.data.merge([ ni.Data(rr) for rr in r[0].transpose()],dim='Cell')
for i in range(len(models)):
    p.append(models[i].predict(data))
for i in range(len(models)):
    figure()
    #subplot(211,title='generated')
    plot(r[1][i][500:1000],'k--')
    #subplot(212,title='prediction')
    plot(p[i][500:1000],'k:')
    legend(['generated','predicted'])
figure()
for i in range(len(models)):
    plot(r[1][i].flatten() - p[i].flatten())
