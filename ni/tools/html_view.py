"""
.. module:: ni.tools.html_view
   :platform: Unix
   :synopsis: HTML View Object for easy reporting of results

.. moduleauthor:: Jacob Huth <jahuth@uos.de>
 
This module can generate HTML output from text or objects that provide a .html_view() function::

	import ni
	view = ni.View()	# this is a shortcut for ni.tools.html_view.View
	view.add("#1/title","This is a test")
	view.add("#2/Some Example Models/tabs/",ni.model.ip.Model({'name': 'Basic Model'}))
	view.add("#2/Some Example Models/tabs/",ni.model.ip.Model({'autohistory_2d':True, 'name': 'Model with Higher Dimensional Autohistory'}))
	view.add("#2/Some Example Models/tabs/",ni.model.ip.Model({'rate':False, 'name': 'Model without Rate Component'}))
	view.add("#3/Some Example Data/tabs/1",ni.data.monkey.Data())
	view.render("this_is_a_test.html")


"""
#import xml.etree.ElementTree as ET

import numpy as np
import re
from copy import copy
import matplotlib.pyplot
import matplotlib.pyplot as pl
import base64, urllib
import StringIO
import glob

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def natural_sorted(l):
	""" sorts a sortable in human order (0 < 20 < 100) """
	ll = copy(l)
	ll.sort(key=natural_keys)
	return ll


class Figure:
	"""
	Figure Context Manager

	Can be used with the **with** statement::

		import ni
		v = ni.View()
		x = np.arange(0,10,0.1)
		with ni.tools.html_view.Figure(v,"some test"):
		    plot(cos(x)) 	# plots to a first plot 
		    with ni.tools.html_view.Figure(v,"some other test"):
		        plot(-1*np.array(x)) # plots to a second plot
		    plot(sin(x))	# plots to the first plot again
		v.render("context_manager_test.html")


	"""
	def __init__(self,view,path,close=True,figsize=False):
		self.view = view
		self.path = path
		self.close = close
		self.figsize = figsize
	def __enter__(self):
		if type(self.figsize) != bool:
			self.fig = matplotlib.pyplot.figure(figsize=self.figsize)
		else:
			self.fig = matplotlib.pyplot.figure()
	def __exit__(self, type, value, tb):
		if tb is None:
			self.view.savefig(self.path, fig=self.fig,close=self.close)
		else:
			self.view.add(self.path, "Exception occured")
			if self.close:
				matplotlib.pyplot.close(self.fig)
			

class View:
	def __init__(self,path=""):
		self.path = path
		self.tree = []
		self.tabs = 0
		self.ids = 0
		self.progress = "false"
		self.fig = ""
		self.title = ""
	def __enter__(self):
		return self
	def __exit__(self, type, value, tb):
		if self.path != "":
			self.render(self.path)
	def add(self,path,obj):
		if hasattr(obj,'html_view'):
			tmp_view = obj.html_view()
			for (t,o) in tmp_view.tree:
				self.tree.append( (path + t, o) )
		else:
			self.tree.append( (path, obj) )
	def save(self, filename):
		f = open(filename,"w")
		pickle.dump(self.tree,f)
		f.close()
	def load(self,filename):
		try:
			f = open(filename,"r")
			self.tree = pickle.load(f)
			f.close()
		except:
			print "Could not open file ",filename
			raise
	def loadList(self,filenames):
		return self.load_list(filenames)
	def load_list(self,filenames):
		self.tree = []
		for filename in filenames:
			try:
				f = open(filename,"r")
				self.tree.extend(pickle.load(f))
				f.close()
			except:
				print "Could not open file ",filename
	def load_glob(self,filename_template):
		return self.load_list(glob.glob(filename_template))
	def parse(self,tree):
		t = {}
		for (k,o) in tree:
			path = k.split("/")
			node = t
			ps = 0
			for p in path:
				ps = ps + 1
				if not p in node:
					node[p] = {}
				#if ps == len(path):
				#	node[p] = o
				node = node[p]
			node["text"] = o
		return t
	def process(self,obj, mode = "text"):
		code = ""
		if mode == "tabs":
			code = code + """
		<div class="tabs">
			  <ul>
	"""
			last_tabs = self.tabs
			tabs = last_tabs
			for t in natural_sorted(obj.keys()):
				tabs = tabs + 1
				title = t.split("/")
				style = ""
				if type(obj[t]) == dict and ".style" in obj[t].keys():
					style = str(obj[t][".style"]["text"])
				code = code + """
			<li><a href="#tabs-"""+str(tabs)+""" " style=" """+style+""" ">"""+t+"""</a></li>
	"""
			code = code + """
			  </ul>
	""" 
			self.tabs = tabs
			tabs = last_tabs
			for t in natural_sorted(obj.keys()):
				tabs = tabs + 1
				code = code + """
			  <div id=\"tabs-"""+str(tabs)+"""\">
			    <p>"""+self.process(obj[t])+"""</p>
			  </div>
	"""
			code = code + """		 
			</div>
	"""
		elif mode == "table":
			code = code + """
		<table>
	"""
			for t in natural_sorted(obj.keys()):
				code = code + """
			  <tr>
			  	<td class="header">"""+str(t)+"""</td>
			    <td>"""+self.process(obj[t],mode="td")+"""</td>
			  </tr>
	"""
			code = code + """		 
			</table>
	"""
		elif mode == "td":
			if type(obj) == list:
				for t in obj:
					code = code + """
				"""+str(t)+"""</td>
			<td>"""
			elif type(obj) == dict:
				for t in natural_sorted(obj.keys()):
					code = code + """
			  """+self.process(obj[t])+"""</td>
		<td>"""
			else:
				code = self.process(obj)
		elif mode == "title":
			code = code + """
		<h1>""" + str(obj) + """</h1>
"""
		elif mode == "hidden":
			if type(obj) == dict:
				for t in natural_sorted(obj.keys()):
					self.ids = self.ids + 1
					code = code + """
		<a class="show_hide_link" href="javascript:$('#hidden"""+str(self.ids)+"""').toggle()">+ show/hide """+str(t)+"""</a><div style="display: none;" id="hidden"""+str(self.ids)+"""">""" + self.process(obj[t]) + """</div>
"""
			else:
				self.ids = self.ids + 1
				code = code + """
		<a class="show_hide_link" href="javascript:$('#hidden"""+str(self.ids)+"""').toggle()">+ show/hide</a><div style="display: none;" id="hidden"""+str(self.ids)+"""">""" + self.process(obj) + """</div>
"""
		else:
			if type(obj) == dict:
				for t in natural_sorted(obj.keys()):
					if t == "":
						code = code + self.process(obj[t])
					elif t == ".style":
						code = code # for now, except tabs
					elif t == "text" or t[0] == "#":
						code = code + self.process(obj[t])
					elif t == "title":
						code = code + self.process(obj[t]["text"], mode="title")
					elif t == "hidden":
						code = code + self.process(obj[t], mode="hidden")
					elif t == "tabs":
						code = code + self.process(obj[t], mode="tabs")
					elif t == "table":
						code = code + self.process(obj[t], mode="table")
					else:
						code = code + self.process(t, mode="title")
						code = code + self.process(obj[t], mode="text")
			else:
				code = code + str(obj)
		return code
	def render(self,path,include_files=True):
		code = """<!doctype html>
<html>
	<head>
"""
		if self.title != "":
			code = code + "<title>"+str(self.title)+"</title>\n"
		if include_files == True:
			with open('html/jquery-1.10.1.min.js','r') as f:
				code = code + """
<script>
""" + f.read() + """
</script>
"""
			with open('html/jquery-ui.js','r') as f:
				code = code + """
<script>
""" + f.read() + """
</script>
"""
			with open('html/jquery-ui.css','r') as f:
				code = code + """
<style>
""" + f.read() + """
</style>
"""
		else:
			code = code + """
  <link rel="stylesheet" href="html/jquery-ui.css" />
  <script src="html/jquery-1.10.1.min.js"></script>
  <script src="html/jquery-ui.js"></script>
"""
		code = code + """  		<script>
		$(function() {
			$( ".tabs" ).tabs();
			$( "#progressbar" ).progressbar({
			      value: """ + str(self.progress) + """
			    });
			if (IsNumeric('""" + str(self.progress) + """')) {
				$( ".progress-label" ).text( '""" + str(self.progress) + """ %' );
			}
		});
		
		</script>
		  <style>
		  #progressbar .ui-progressbar-value {
		    background-color: #ccc;
		  }
		  .progress-label {
		    position: absolute;
		    left: 50%;
		    top: 4px;
		    font-weight: bold;
		    text-shadow: 1px 1px 0 #fff;
		  }
  		  a.show_hide_link {
		  	display: block;
		  	color: blue;
		  }
  		  a {
		  	color: grey;
		  	font-family: sans-serif;
		  }
		  </style>
	</head>

	<body>"""
		if self.progress != "false":
			code = code + """
	<div id="progressbar"><div class="progress-label"></div></div>
"""

		code = code + self.process(self.parse(self.tree))
		code = code + """		 
	</body>
</html>"""
		if path == "":
			return code
		else:
			with open(path,"w") as f:
				f.write(code)
	def has(self, path):
		for (p,o) in self.tree:
			if p == path:
				return True
		return False
	def figure(self,path="",close=True,figsize=False):
		"""
		Provides a Context Manager for figure management

		Should be used if plots are to be used in 

		Example::

			import ni
			v = ni.View()
			x = np.arange(0,10,0.1)
			with v.figure("some test"):
			    plot(cos(x))		# plot to a first plot
			    with v.figure("some other test"):
			        plot(-1*np.array(x))	# plot to a second plot
			    plot(sin(x))		# plot to the first plot again
			v.render("context_manager_test.html")

		"""
		if path == "":
			i = 0
			while self.has("figure "+str(i)):
				i = i + 1
			path = "figure "+str(i)
		self.fig = Figure(self,path,close=close,figsize=figsize)
		return self.fig
	def savefig(self,p="", fig="", close=True):
		if fig != "":
			imgdata = StringIO.StringIO()
			fig.savefig(imgdata, format='png')
			imgdata.seek(0) 
			image = base64.encodestring(imgdata.buf) 
			self.add(p,"<img src='data:image/png;base64," + urllib.quote(image) + "'>")
			if close:
				matplotlib.pyplot.close(fig)
		elif self.fig != "":
			if p != "":
				self.fig.path = p
			self.fig.__exit__(None,None,None)
			self.fig = ""
	def html_view(self):
		return self


"""

From here on, there is dead code.

def model_view(model):
	view = View()
	model_prefix = model.model.name + "/"
	view.add(model_prefix + "#1/title","Analysis of " + model.name)
	for key in model.configuration.__dict__:
		view.add(model_prefix + "#3/tabs/Configuration/table/"+key,model.configuration.__dict__[key])
	return view

def fittedmodel_view(model):
	view = View()
	model_prefix = model.model.name + "/"
	view.add(model_prefix + "#2/beta",model.beta)
	for key in model.configuration.__dict__:
		view.add(model_prefix + "#3/tabs/Configuration/table/"+key,model.configuration.__dict__[key])
	view.add(model_prefix + "#3/tabs/Design","")
	prot_nr = 0
	prototypes = model.plotPrototypes()
	for p in prototypes:
		prot_nr = prot_nr + 1
		view.fig = prototypes[p]
		view.savefig(model_prefix + "#3/tabs/Prototypes/#"+str(prot_nr)+"/"+str(p))
	for c in model.design.components:
		if type(c) != str:
			fig = view.figure()
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
					pl.imshow(splines[i,:,:])
			view.savefig(model_prefix + "#3/tabs/Design/#2/"+c.header+"/#2/Splines")
		else:
			view.add(model_prefix + "#3/tabs/Design/#2/tabs/##/component",str(c))
	return view

def stat_view(stats):
	stat_prefix = ""
	for k in stats.keys():
		view.add(stat_prefix + "#2/tabs/Models/"+k.replace("/","/tabs/"),k)
		node = stats.getNode(k)
		for prop in node:
			if type(node[prop]) == numpy.float64:
				view.add(stat_prefix + "#2/tabs/Models/"+k.replace("/","/tabs/")+"/#1/"+prop,prop+" = "+str(node[prop]))
			if type(node[prop]) == list:
				fig = view.figure()
				pl.plot(node[prop])
				view.savefig(stat_prefix + "#2/tabs/Models/"+k.replace("/","/tabs/")+"/Plots/tabs/"+prop)
	for d in stats.getDimensions():	
		x = stats.get(d)
		if type(x[0]) == numpy.string_:
				view.add(stat_prefix + "#2/tabs/Dimensions/tabs/"+d+"/",str(x))
		elif len(x.shape) == 1:
			try:
				fig = view.figure()
				pl.plot(x)
				view.savefig(stat_prefix + "#2/tabs/Dimensions/tabs/"+d+"/#1")
			except:
				view.add(stat_prefix + "#2/tabs/Dimensions/tabs/"+d+"/#1/failed",x)
		elif len(x.shape) == 2:
			try:
				fig = view.figure()
				pl.plot(x)
				view.savefig(stat_prefix + "#2/tabs/Dimensions/tabs/"+d+"/#1")
				fig = view.figure()
				pl.plot(x.transpose())
				view.savefig(stat_prefix + "#2/tabs/Dimensions/tabs/"+d+"/#2")
			except:
				fig = view.figure()
				for xx in x:
					for xxx in xx:
						pl.plot(xxx)
				view.savefig(stat_prefix + "#2/tabs/Dimensions/tabs/"+d+"/#1/failed")
		elif len(x.shape) == 3:
			try:
				fig = view.figure()
				pl.plot(np.mean(x,axis=0))
				view.savefig(stat_prefix + "#2/tabs/Dimensions/tabs/"+d+"/tabs/Axis 0/#1")
				fig = view.figure()
				pl.plot(np.mean(x,axis=0).transpose())
				view.savefig(stat_prefix + "#2/tabs/Dimensions/tabs/"+d+"/tabs/Axis 0/#2")
				fig = view.figure()
				pl.plot(np.mean(x,axis=1))
				view.savefig(stat_prefix + "#2/tabs/Dimensions/tabs/"+d+"/tabs/Axis 1/#1")
				fig = view.figure()
				pl.plot(np.mean(x,axis=1).transpose())
				view.savefig(stat_prefix + "#2/tabs/Dimensions/tabs/"+d+"/tabs/Axis 1/#2")
				fig = view.figure()
				pl.plot(np.mean(x,axis=2))
				view.savefig(stat_prefix + "#2/tabs/Dimensions/tabs/"+d+"/tabs/Axis 2/#1")
				fig = view.figure()
				pl.plot(np.mean(x,axis=2).transpose())
				view.savefig(stat_prefix + "#2/tabs/Dimensions/tabs/"+d+"/tabs/Axis 2/#2")
			except:
				view.add(stat_prefix + "#2/tabs/Dimensions/tabs/"+d+"/#1/failed",x)
	return view

def data_view(data):	
	view = View()
	data_prefix = ""
	for c in range(d.nr_conditions):
		cond_data = d.condition(c)
		fig = view.figure()
		pl.plot(np.mean(np.array(cond_data.getFiringRate()).transpose(),axis=0))
		view.savefig(data_prefix + "/tabs/Condition " + str(c) + "/Firing Rate/Plot/#4/Mean/tabs/Axis 0")
		fig = view.figure()
		pl.plot(np.mean(np.array(cond_data.getFiringRate()).transpose(),axis=1))
		view.savefig(data_prefix + "/tabs/Condition " + str(c) + "/Firing Rate/Plot/#4/Mean/tabs/Axis 1")
		fig = view.figure()
		pl.plot(gaussian_filter(np.mean(np.array(cond_data.getFiringRate()).transpose(),axis=1),20))
		view.savefig(data_prefix + "/tabs/Condition " + str(c) + "/Firing Rate/Plot/#4/Mean/tabs/Axis 1 (smoothed)")
		for cell in range(cond_data.nr_cells):
			cell_data = cond_data.cell(cell)
			fig = view.figure()
			pl.imshow(1-cell_data.data,interpolation='nearest',aspect=20,cmap='gray')
			view.savefig(data_prefix + "/tabs/Condition " + str(c) + "/Spikes/tabs/Cell " + str(cell) + "/Plot")
	return view

"""