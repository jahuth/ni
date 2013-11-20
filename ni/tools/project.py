"""
.. module:: ni.tools.project
   :platform: Unix
   :synopsis: Manages Python Projects using the ni toolbox

.. moduleauthor:: Jacob Huth <jahuth@uos.de>

NI Project Management

 - All steps in a configuration / simulation process will be logged to some folder structure
 - after the simulation and even after changing the original code, the results should still be viewable / interpretable with a project viewer
 - batches of runs should be easy to batch interpret (characteristic plots etc.)
 - metadata should contain among others:
	date
	software versions
	configuration options
	manual comments
 - saving of plots/data should be done by the project manager

"""

import os, sys, time, datetime, shutil
import xml.etree.ElementTree as ET
import traceback 
import hashlib
import pickle
import html_reporter
from guppy import hpy
from ni.tools.alert import alert
import ni.config
import glob 
import re 
import time
import socket
from copy import copy
import uuid
import matplotlib
from ni.tools.html_view import View
try:
	import IPython
except:
	IPython = None

def atoi(text):
    """ converts text containing numbers into ints / used by :func:`natural_keys`"""
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    (See Toothy's implementation in the comments of http://nedbatchelder.com/blog/200712/human_sorting.html )
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def natural_sorted(l):
    """ sorts a sortable in human order (0 < 20 < 100) """
    ll = copy(l)
    ll.sort(key=natural_keys)
    return ll

class VariableContainer:
	pass

class ListContainer:
	def __init__(self):
		self.list = []
	def append(self,msg_type,priority,date,job,txt):
		self.list.append((msg_type,priority,date,job,str(txt),txt))
		if len(self.list) > 50:
			self.list = self.list[-50:-1]
	def __str__(self):
		return str(self.list)
	def clear(self):
		self.list = []
	def __len__(self):
		return len(self.list)

class PickleContainer:
	def __init__(self,f):
		self.path = f
		self.list = []
	def append(self,msg_type,priority,date,job,txt):
		self.list.append((msg_type,priority,date,job,str(txt),txt))
	def __str__(self):
		return str(self.list)
	def clear(self):
		self.list = []
	def __len__(self):
		return len(self.list)

class LogContainer:
	def __init__(self,f):
		self.path = f
		self.list = []
	def append(self,msg_type,priority,date,job,txt):
		with open(self.path,"a") as logfile:
			logfile.write(">> " + str(job) + " <<\n" + str(date) + " [" + str(msg_type) +"/" + str(priority) +"]\n" + str(txt) + "\n\n")
	def __str__(self):
		return str(self.list)
	def clear(self):
		self.list = []
	def __len__(self):
		return len(self.list)


class Figure:
	"""
	Figure Context Manager

	Can be used with the **with** statement::

		import ni
		x = np.arange(0,10,0.1)
		with ni.figure("some_test.png"):
		    plot(cos(x)) 	# plots to a first plot 
		    with ni.figure("some_other_test.png"):
		        plot(-1*np.array(x)) # plots to a second plot
		    plot(sin(x))	# plots to the first plot again

	Or if they are to be used in an interactive console::

		import ni
		x = np.arange(0,10,0.1)
		with ni.figure("some_test.png",close=False):
			plot(cos(x)) 	# plots to a first plot 
				with ni.figure("some_other_test.png",close=False):
					plot(-1*np.array(x)) # plots to a second plot
			plot(sin(x))	# plots to the first plot again

	Both figures will be displayed, but the second one will remain available after the code. (But keep in mind that in the iPython pylab console, after every input, all figures will be closed)

	"""
	def __init__(self,path,display=False,close=True):
		self.path = path
		self.display = display
		self.last_fig = matplotlib.pyplot.gcf()
		self.close = close
	def __enter__(self):
		self.fig = matplotlib.pyplot.figure()
	def __exit__(self, type, value, tb):
		if self.path is not None and self.path != "":
			self.fig.savefig(self.path)
		if self.display:
			try:
				# trying to use ipython display
				IPython.core.display.display(self.fig)
			except:
				# otherwise presume that we run with some other gui backend. If we don't, nothing will happen.
				self.fig.show(warn=False)
		if self.close:
			matplotlib.pyplot.close(self.fig)
		else:
			matplotlib.pyplot.figure(self.last_fig.number)

def figure(path,display=False,close=True):
	"""


	Can be used with the **with** statement::

		import ni
		x = np.arange(0,10,0.1)
		with ni.figure("some_test.png"):
		    plot(cos(x)) 	# plots to a first plot 
		    with ni.figure("some_other_test.png"):
		        plot(-1*np.array(x)) # plots to a second plot
		    plot(sin(x))	# plots to the first plot again


	Or if they are to be used in an interactive console::


		import ni
		x = np.arange(0,10,0.1)
		with ni.figure("some_test.png",display=True):
		    plot(cos(x)) 	# plots to a first plot 
		    with ni.figure("some_other_test.png",close=False):
		        plot(-1*np.array(x)) # plots to a second plot
		    plot(sin(x))	# plots to the first plot again

	Both of these figures will be displayed, but the second one will remain open and can be activated again.


	"""
	return Figure(path,display=display,close=close)


class Job:
	def __init__(self, project, session, path, job_name="", job_number="", file= "", status='initializing...', dependencies=[]):
		self.project = project
		self.session = session
		self.job_name = job_name
		self.job_number = job_number
		self.status_path = path
		if self.status_path[-4:] == ".txt":
			self.status_path = self.status_path[:-4]
		self.path = self.session.path+"job_"+str(job_number)+"_"
		self.status = status
		self.file = file
		self.dependencies = dependencies
		self.host = str(self.project.hostname)+":"+str(self.project.instancename)
		self.last_touched = 'never'#time.asctime()
		self.get_status()
		self.current_activity = ""
	def can_run(self):
		self.session.update_jobs()
		for dep in self.dependencies:
			if str(dep) in self.session.job_status:
				for k in self.session.job_status[str(dep)].keys():
					if k != "done.":
						if self.session.job_status[str(dep)][k] > 0:
							return False
		return True
	def update(self):
		if os.path.exists(self.status_path+".txt"):
			try:
				with open(self.status_path+".txt","r") as f:
					s = f.read()
					ff = eval(s)
			except:
				return  self.__dict__
			if type(ff) == dict:
				self.__dict__.update(ff)
			return  self.__dict__
	def get_status(self):
		if os.path.exists(self.status_path+".txt"):
			try:
				with open(self.status_path+".txt","r") as f:
					s = f.read()
					ff = eval(s)
			except:
				return  self.__dict__
			if type(ff) == dict:
				self.__dict__.update(ff)
			return  self.__dict__
		else:
			self.save()
			return  self.__dict__
	def set_status(self,msg=""):
		if msg != "":
			self.status = msg
		self.host = str(self.project.hostname)+":"+str(self.project.instancename)
		self.last_touched = time.asctime()
		self.save()
	def set_activity(self,msg=""):
		if msg != "":
			self.current_activity = msg
		self.host = str(self.project.hostname)+":"+str(self.project.instancename)
		self.last_touched = time.asctime()
		self.project.job = self.job_name + self.current_activity
		self.save()
	def save(self):
		d = {}
		for k in self.__dict__.keys():
			if not k in ["project","session"]:
				d[k] = self.__dict__[k] 
		with open(self.status_path+".txt","w+") as f:
			f.write(str(d))
	def run(self,parameters=[]):
		source_file = self.session.path+self.file
		if not os.path.exists(source_file):
			raise Exception("File not found!")
		_Project = self.project
		l = locals()
		out = []
		l["parameters"] = parameters
		l["ThisProject"] = self.project
		l["_out"] = out
		l["_job"] = self.job_number
		l["_current_project"] = self.project
		l["_current_session"] = self.session
		l["_current_job"] = self
		l["__file__"] = self.file
		l["path"] = self.session.path
		l["job_path"] = self.path
		l["__name__"] = "__main__"
		l["tmp_path"] = ni.config.tmp_path
		l["status"] = self.set_activity
		l["log"] = self.project.log
		l["err"] = self.project.err
		l["dbg"] = self.project.dbg
		self.project.job = self.job_name
		self.project.containers.append(LogContainer(self.path + '_log.txt'))
		if os.path.exists(".git/refs/heads/master"):
			self.project.log('Last ni.toolbox Revision: ' +open(".git/refs/heads/master").read() + " from: " + time.ctime(os.stat(".git/refs/heads/master").st_mtime) + " (or slightly later)",1)
		self.project.log('Executing file ' + self.file + ' (on '+self.project.hostname+':'+self.project.instancename+') md5:'+str(hashlib.md5(open(source_file, 'rb').read()).hexdigest())+ '\n Last changed on: ' + time.ctime(os.stat(source_file).st_mtime),1)
		self.set_status("running...")
		try:
			#exec(compile(open(self.folder+self.main).read(), self.folder+self.main, 'exec'),globals(),l)
			execfile(source_file,l)
		except:
			variables = VariableContainer()
			for o in l.keys():
				variables.__dict__[o] = l[o]
			self.project.variables = variables
			self.project.err(traceback.format_exc())
			self.project.log("Writing to logfile...")
			alert("Error! See logfile.")
			self.set_status("failed.")
			with open(self.path + '_error.txt',"w") as logfile:
				logfile.write(traceback.format_exc())
				logfile.write(self.project.report(silent=True))
			raise
		alert("Success.")
		self.set_status("done.")
		self.project.log("Writing to logfile...")
		with open(self.path+"log.txt","w") as logfile:
			logfile.write(self.project.report(silent=True))
		variables = VariableContainer()
		for o in l.keys():
			variables.__dict__[o] = l[o]
		self.project.variables = variables#l
		return variables
	def html_view(self):
		view = View()
		view.title = self.project.name + " / " + self.session.path.split("/")[-2] + "/" + str(self.job_name) + " " + str(self.job_number) + " ni. toolbox job"
		for k in self.__dict__.keys():
			if k != "session" and k != "project":
				v = self.__dict__[k]
				view.add("/Info/table/"+str(k),v)
		if os.path.exists(self.path+"_results.html"):
			view.add("/Results","<a href=\""+"job_"+str(self.job_number)+"_"+"_results.html"+"\">results</a>")
		if os.path.exists(self.path+"results.html"):
			view.add("/Results","<a href=\""+"job_"+str(self.job_number)+"_"+"results.html"+"\">results</a>")
		if os.path.exists(self.path+"_log.txt"):
			view.add("/Log","<a href=\""+"job_"+str(self.job_number)+"_"+"_log.txt"+"\">log</a>")
		if os.path.exists(self.path+"_error.txt"):
			view.add("/Error","<a href=\""+"job_"+str(self.job_number)+"_"+"_error.txt"+"\">Error log</a>")
		if os.path.exists(self.path+"error.txt"):
			view.add("/Error","<a href=\""+"job_"+str(self.job_number)+"_error.txt"+"\">Error log</a>")
		return view

class Session:
	def __init__(self, project, path="", parameter_string = ""):
		self.project = project
		if path == "":
			run = 1
			path = self.project.folder+"sessions/session"+str(run)+"/"
			while os.path.exists(path):
				run = run + 1
				path = self.project.folder+"sessions/session"+str(run)+"/"
			self.path = path
		else:
			if os.path.exists(self.project.folder+"sessions/"+path):
				self.path = self.project.folder+"sessions/"+path
			elif os.path.exists(self.project.folder+path):
				self.path = self.project.folder+path
			elif os.path.exists(path):
				self.path = path
			else:
				self.path = path
			if self.path[-1] != "/":
				self.path = self.path + "/"
		self.status = ""
		self.jobs = {}
		self.job_status = {}
		self.job_counter = 0
		self.parameters = {}
		self.parameter_string = parameter_string
		try: 
			with open(self.path+"status.txt","r") as f:
				self.status = f.read()
		except:
			pass
		self.find_jobs()
	def set_status(self,msg=""):
		if not os.path.exists(self.path):
			os.makedirs(self.path)
		self.status = msg
		with open(self.path+"status.txt","w") as f:
			f.write(msg)
	def abandon(self):
		self.set_status("abandoned.")
	def get_status(self):
		return [self.jobs[j].get_status() for j in natural_sorted(self.jobs.keys())]
	def find_jobs(self):
		self.update_jobs()
	def setup_jobs(self,source_file,parameter_string=""):
		if not os.path.exists(self.path):
			os.makedirs(self.path)
		os.makedirs(self.path+"jobs/")
		os.makedirs(self.path+"job_code/")
		with open(self.path+"parameters.txt","w") as f:
			f.write(self.parameter_string)
		if os.path.exists(source_file):
			job_file = True
			try:
				jobs = self.parse_job_file(source_file,parameter_string)
			except Exception as e:
				if e.args[0] == "Not a job file":
					job_file = False
					return
				else:
					raise
			print len(jobs), " Jobs calculated."
			for job in jobs:
				with open(self.path+"job_code/job_"+str(job["nr"])+".py", "w") as code_file:
					code_file.write(job["code"])
				_o = self.add_job(job_name = job["job"],job_number = job["nr"], file="job_code/job_"+str(job["nr"])+".py", status= 'pending', dependencies= job["dependencies"])
	def update_job_files(self,source_file="",parameter_string=""):
		if source_file == "":
			source_file = self.project.folder + "main.py"
		if os.path.exists(source_file):
			job_file = True
			try:
				jobs = self.parse_job_file(source_file,parameter_string)
			except Exception as e:
				if e.args[0] == "Not a job file":
					job_file = False
					return
				else:
					raise
			print len(jobs), " Jobs calculated."
			for job in jobs:
				with open(self.path+"job_code/job_"+str(job["nr"])+".py", "w") as code_file:
					code_file.write(job["code"])
	def parse_job_file(self,filename,parameter_string=""):
		if parameter_string == "":
			parameter_string = self.parameter_string
			try: 
				with open(self.path+"parameters.txt","r") as f:
					parameter_string = f.read()
			except:
				pass
		new_files = {}
		quantifiers = {}
		tagged_lines = []
		preamble = []
		ordered_jobs = []
		dependencies = {}
		with open(filename) as source_file:
			source = source_file.readlines()
			job = "#preamble"
			last_indent = ""
			job_indent = ""
			job_indent_stack = []
			job_stack = []
			indent_specified = True
			for l in range(len(source)):
				if source[l].strip().startswith("## End of Parameters"):
					preamble.append("## Changed Parameters: ")
					for p in parameter_string.split("\n"):
						preamble.append(p)
				indent = re.match(r'^([\t ]*)',source[l]).group(1)
				line = re.match(r'^[\t ]*(.*)',source[l]).group(1)
				if not indent_specified:
					if not indent.startswith(job_indent):
						raise Exception("Parsing Error for indent at line "+str(l)+" in file: "+filename)
					job_indent = indent
					indent_specified = True
				if not indent.startswith(job_indent) and not source[l].strip() == "":
					while indent != job_indent:
						_job = job
						job = job_stack.pop()
						job_indent = job_indent_stack.pop()
				if line.startswith("job "):
					m = re.match(r'job *((\".*\")|(\'.*\')|(\"\"\".*\"\"\"))(.*:)',line)
					if m:
						name = m.group(1)
						if name.startswith('\"\"\"'):
							name = name[3:-3]
						else:
							name = name[1:-1]
						quantifier = m.group(5)
						job_stack.append(job)
						job_indent_stack.append(job_indent)
						job = name
						indent_specified = False
						quantifiers[job] = re.findall(r'for (.*?) in (.*?)(?=(?= for)|:)',quantifier)
						dependencies[job] = []
						ordered_jobs.append(job)
					else:
						raise Exception("Parsing Error in job ... : declaration at line "+str(l)+" in file: "+filename)
				elif line.startswith("require "):
					m = re.match(r'require *((\".*\")|(\'.*\')|(\"\"\".*\"\"\"))',line)
					if len(ordered_jobs) > 1:
						r = ordered_jobs[-2]
					else:
						r = ""
					if m:
						r = m.group(1)
						if name.startswith('\"\"\"'):
							r = r[3:-3]
						else:
							r = r[1:-1]
					dependencies[job].append(r)
				else:
					real_indent = indent[len(job_indent):]
					real_line = real_indent + line
					new_files[job] = real_line
					if job == "#preamble":
						preamble.append(real_line)
					tagged_lines.append((job,real_line))
		if len(set(ordered_jobs)) <= 1:
			raise Exception("Not a job file")
		preamble_vars = self.execute("\n".join(preamble),{'parameters':self.parameters})
		unique_jobs = set(new_files.keys())
		independent_jobs = []
		for j in ordered_jobs:
			independent = True
			for j_2 in ordered_jobs:
				if not j == j_2 and j_2.startswith(j):
					independent = False
			if independent and not j in independent_jobs:
				if len(independent_jobs) > 0:
					pass #dependencies[j] = [independent_jobs[-1]]
				else:
					pass #dependencies[j] = []
				independent_jobs.append(j)
		jobs = []
		j_nr = 0
		for i_j in independent_jobs:
			print "---------------------------------------"
			quantified_parameters = [""]
			if i_j in quantifiers:
				for q in quantifiers[i_j]:
					new_quantified_parameters = []
					for qp in quantified_parameters:
						for qi in eval(q[1],preamble_vars):
							new_quantified_parameters.append(qp + str(q[0]) + " = " + str(qi)+ "\n")
					quantified_parameters = new_quantified_parameters
			for qp in quantified_parameters:
				code = "\n".join(preamble) + "\n" + qp
				for (tag,line) in tagged_lines:
					if i_j.startswith(tag):
						code = code + line + "\n"
				jobs.append({'nr':j_nr,'job':i_j,'quantified_parameters': quantified_parameters,'code':code,'dependencies':dependencies[i_j]})
				j_nr = j_nr + 1
		return jobs
	def execute(self,code,local_vars = {}):
		if not "parameters" in local_vars:
			local_vars["parameters"] = []
		local_vars["_current_project"] = self.project
		local_vars["_current_session"] = self
		local_vars["_current_job"] = TemporaryJob(self.project,self,"Preamble")
		try:
			exec code in local_vars
		except:
			raise
		return local_vars
	def update_jobs(self):
		#self.jobs = {}
		job_status = {}
		if os.path.exists(self.path+"job_status/"):
				for j in os.listdir(self.path+"job_status/"):
					res = re.search("job_(.*)\.txt",j)
					if res != None:
						job = res.group(1)
						if job in self.jobs:
							self.jobs[job].update()
						else:
							self.jobs[job] = Job(self.project,self,self.path+"job_status/job_"+job)
						if self.jobs[job].job_name in job_status:
							if self.jobs[job].status in job_status[self.jobs[job].job_name]:
								job_status[self.jobs[job].job_name][self.jobs[job].status] = job_status[self.jobs[job].job_name][self.jobs[job].status] + 1
							else:
								job_status[self.jobs[job].job_name][self.jobs[job].status] = 1
						else:
							job_status[self.jobs[job].job_name] = { self.jobs[job].status: 1 }
		self.job_status = job_status
	def next_job(self, retry_failed = False, ignore_dependencies = False):
		for j in natural_sorted( self.jobs.keys() ):
			if self.jobs[j].status == "pending":
				return self.jobs[j]
			if retry_failed and self.jobs[j].status == "failed.":
				return self.jobs[j]
		raise Exception("End of Jobs")
	def add_job(self, job_name="", job_number="",**kwargs):
		if not os.path.exists(self.path+"job_status/"):
			os.makedirs(self.path+"job_status/")
		if job_number == "":
			while str(self.job_counter) in self.jobs.keys():
				self.job_counter = self.job_counter + 1
			job_number = self.job_counter
		self.jobs[job_number] = Job(self.project, self, self.path+"job_status/job_"+str(job_number), job_name, job_number,**kwargs)
		#if type(dic) == dict:
		#	self.jobs[job_number].__dict__.update(dic)
		#self.jobs[job_number].save()
		return self.jobs[job_number]
	def print_job_status(self):
		self.update_jobs()
		s = ""
		for job in self.job_status.keys():
			s = s + job + ": "
			if "done." in self.job_status[job]:
				s = s + str(self.job_status[job]["done."]) + " done "
			if "starting..." in self.job_status[job]:
				s = s + str(self.job_status[job]["starting..."]) + " starting "
			if "running..." in self.job_status[job]:
				s = s + str(self.job_status[job]["running..."]) + " running "
			if "pending" in self.job_status[job]:
				s = s + str(self.job_status[job]["pending"]) + " pending "
			if "failed." in self.job_status[job]:
				s = s + str(self.job_status[job]["failed."]) + " failed "
			s = s + "\n"
		return s
	def print_long_job_status(self):
		s = ""
		self.update_jobs()
		for job in self.jobs.keys():
			s = s + job + " " + self.jobs[job] + "\n"
		return s
	def reset_failed_jobs(self,which="failed.",to="pending"):
		self.update_jobs()
		for job in self.jobs.keys():
			if self.jobs[job].status == which:
				self.jobs[job].set_status(to)
	def save_html(self,path="session.html"):
		view = self.html_view()
		view.render(self.path+path)
	def html_view(self):
		view = View()
		view.title = self.project.name + " / " + self.path.split("/")[-2] + " ni. toolbox session"
		#view.add("#0/Project",self.project)
		for job in self.job_status.keys():
			if "pending" in self.job_status[job]:
				view.add("#1/Job Status/"+str(job)+"/table/pending", str(self.job_status[job]["pending"]))
			if "starting..." in self.job_status[job]:
				view.add("#1/Job Status/"+str(job)+"/table/starting", str(self.job_status[job]["starting..."]))
			if "done." in self.job_status[job]:
				view.add("#1/Job Status/"+str(job)+"/table/done", str(self.job_status[job]["done."]))
			if "running..." in self.job_status[job]:
				view.add("#1/Job Status/"+str(job)+"/table/running", str(self.job_status[job]["running..."]))
			if "failed." in self.job_status[job]:
				view.add("#1/Job Status/"+str(job)+"/table/failed", str(self.job_status[job]["failed."]))
		jobs = copy(self.jobs)
		for j in jobs.keys():
			view.add("#2/Jobs/tabs/"+str(jobs[j].job_name)+"/tabs/"+str(j),jobs[j])
			if jobs[j].status == "starting...":
				view.add("#2/Jobs/tabs/"+str(jobs[j].job_name)+"/tabs/"+str(j)+"/.style","background-color: lightyellow; color: yellow;")
			if jobs[j].status == "running...":
				view.add("#2/Jobs/tabs/"+str(jobs[j].job_name)+"/tabs/"+str(j)+"/.style","background-color: lightyellow; color: black;")
			if jobs[j].status == "failed.":
				view.add("#2/Jobs/tabs/"+str(jobs[j].job_name)+"/tabs/"+str(j)+"/.style","color: darkred;")
			if jobs[j].status == "done.":
				view.add("#2/Jobs/tabs/"+str(jobs[j].job_name)+"/tabs/"+str(j)+"/.style","color: darkgreen;")
		return view

class TemporaryJob(Job):
	def __init__(self,project,session,job_name):
		self.project = project
		self.session = session
		self.job_name = job_name
		self.job_number = str(uuid.uuid4())
		self.status_path = "temp_job_"+str(self.job_number)
		if self.status_path[-4:] == ".txt":
			self.status_path = self.status_path[:-4]
		self.path = self.session.path+"temp_job_"+str(self.job_number)+"_"
		self.status = "ghosting"
		self.file = file
		self.dependencies = []
		self.host = str(self.project.hostname)+":"+str(self.project.instancename)
		self.last_touched = 'never'#time.asctime()
		self.session.jobs[self.job_number] = self


class TemporarySession(Session):
	def __init__(self,project):
		self.project = project
		self.status = ""
		self.jobs = {}
		self.job_status = {}
		self.job_counter = 0
		self.path = "temp_session"

class Project:
	"""
	Project Class

	loads a Project folder (containing eg. a main.py file)
	"""
	def __init__(self, folder = "unsaved_project", name = ""):
		self.folder = folder
		if self.folder[-1] != "/":
			self.folder = self.folder + "/"
		if name == "":
			if folder == "unsaved_project":
				name = "New Project"
			else:
				name = self.folder.split("/")[-2]
		self.job_mode = "no jobs"
		self.name = name
		self.path = self.folder
		self.parameters = []
		self.logging = True
		self.code_files = []
		self.code = []
		self.errors = ListContainer()
		self.logs = ListContainer()
		self.dbgs = ListContainer()
		self.msgs = ListContainer()
		self.containers = [ListContainer()]
		self.current_job = False
		self.session = False
		self.sessions = []
		self.job_status = {}
		self.overall_status = {}
		self.retry_failed = True
		self.log_updated_on_priority = 1 # a msg with a greater or equal priority will cause a logifle update
		self.jobs = {}
		if os.path.exists(self.folder + "main.py"):
			self.main = "main.py"
		self.description = ""
		self.logo = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAABmJLR0QA/wD/AP%2BgvaeTAAAACXBI%0AWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3QoYCTQZbfmiOAAAABl0RVh0Q29tbWVudABDcmVhdGVk%0AIHdpdGggR0lNUFeBDhcAAAM/SURBVGje7VhbSBRhFP4m11ugm8iiiD2lIoUPvghC5g3qRbHoRRPx%0AFkj4VLG55hJWQij00hKuRMgi2VYPjqGIZiBroBCY6EKLpmup6Y66eYnNnXX370Ecmd2ZEXdNXJoP%0Ahplz/nMO/zcz5zJD0TRNEMRQAEBRUVFQbr67uxunEOSQCfyXBIYtw0ivT0d0VTRaP7QGH4ESXQnG%0Af4xjy7kF7Ttt8BGIU8Zx18pIZeBl9NjL391uNNFNmLHNoOFqQ/ARSIxNhL5aL1chySfgcDpg27DB%0AyljRO96Lrs9dsK5aQV7tTh62dRvaPrah50sPZplZOFgHkuKSkH8hH9prWsRGxfLieTwe2H/bsbS%2B%0ABMtPCwYmB9A/0Y95%2BzwX80gJpN9Px9TylI%2BeEAJdvw7at1pEhEUg73weMs5lYGR6BGPfxzC5MAnj%0AqBETTyagUqo4v77xPhQ8LTj6R0DTNBGCx%2BMhzDpDKvWVBDfAHcXPiomqRkXah9oJ62I5%2B52dHVKi%0AK%2BHsqvRVPjE3HZvE9NVEMh9k8mL6C5qmiWgOUBQFlVKFxuuNPP2CfQHmZjMqsisQqgjl9CEhIdAU%0AajjZZDH5xIyKjEJWahbaqtv%2BfQ7sIeFMAk8erB9EeFi4oG1yfDJ3PcPMiMZMiU85vlFCoeBzFNs8%0AAESGR%2B7nCsQTUyqGPMzJBGQCMoGTBZfLDZZ1%2B0/AyTp5MutiRW2911w7roBi0p3TSIsxIC3GgPfG%0Ab/4R8G5IcytzorazzKykvAfzgpknW1esgnYPb49g%2B48bzm03Ht0ZPTyB5V/LUHeqeTqNUYO1rTUf%0A29XNVdS/qefp1J1qMBsMJ7vdboxZx1DzsoZnV/e6Dov2Rd/GeHq/iYaFC2%2BVommaCP3Yyn2ciyHL%0AkLATKJRdLIPhlgEAUN5ajo5PHaLdNyc1B7WXa1H6vBSsW/wVbClugbpw/4aZBuZx7%2BbuTNX84hKy%0Ar5z1%2BbElOo0GAySnUbmMygQOSYCiKNEPGynsrXufhXyFbA6CWAz5FZIJyARkAieMACHCc4yY3nvd%0A%2ByzkK2RzEMRiHNgHAqn//tR1KX%2Bpvck5IBOQCcgETmgfCKT%2B%2B1PXpfyl9qbgPo6DFH8BkBmOWDZo%0AfhoAAAAASUVORK5CYII%3D%0A"
		self.hostname = socket.gethostname()
		self.instancename = str(uuid.uuid4())
		if os.path.exists(self.folder+"description.txt"):
			with open(self.folder+"description.txt","r") as f:
				self.description = f.read()
		if os.path.exists(self.folder):
			self.containers.append(LogContainer(self.folder + 'log.txt'))
			self.find_sessions()
			if len(self.sessions) > 0:
				self.session = self.sessions[-1]
				self.path = self.session.path
			self.update_job_status()
			self.print_job_status()
		if len(self.errors) > 0:
			print "Errors occured:"
			for e in self.errors:
				print "\t" + e
	def find_sessions(self):
		self.sessions = []
		if os.path.exists(self.folder+"sessions/"):
			for sess in natural_sorted(os.listdir(self.folder+"sessions/")):
				try:
					self.sessions.append(Session(self,sess))
				except:
					raise
		return self.sessions
	def select_session(self,path):
		for s in self.sessions:
			if path.startswith(s.path):
				self.session = s
			if (self.folder + path).startswith(s.path):
				self.session = s
	def last_run(self):
		if os.path.exists(self.folder+"run/"):
			last_run = False
			for j in natural_sorted(os.listdir(self.folder+"run/")):
				res = re.search("run(\d*)",j)
				if res != None:
					last_run = res.groups(1)[0]
			return last_run
		return False
	def get_parameters_from_job_file(self):
		parameters = []
		param_flag = False
		with open(self.folder+"main.py") as source_file:
			source = source_file.readlines()
			for l in range(len(source)):
				if source[l].startswith("## Parameters"):
					param_flag = True
				if param_flag:
					parameters.append(source[l])
				if source[l].startswith("## End of Parameters"):
					param_flag = False
		return "".join(parameters)
	def setup_jobs(self,parameter_string=""):
		self.session = Session(self,parameter_string=parameter_string)
		self.sessions.append(self.session)
		self.session.setup_jobs(self.folder+"main.py",parameter_string)
		self.update_job_status()
	def update_job_status(self):
		if self.session != False:
			self.session.update_jobs()
			self.job_status = self.session.job_status
			self.jobs = self.session.jobs
			self.overall_status = {}
			for j in self.session.jobs:
				ff = self.session.jobs[j]
				if not ff.status in self.overall_status:
					self.overall_status[ff.status] = 1
				else:
					self.overall_status[ff.status] = self.overall_status[ff.status] + 1
	def less_running_than(self,N):
		running = 0
		if "running..." in self.overall_status:
			running = running + self.overall_status["running..."]
		if "starting..." in self.overall_status:
			running = running + self.overall_status["starting..."]
		if running < N:
			return True
		return False
	def print_job_status(self):
		self.update_job_status()
		s = ""
		for job in self.job_status.keys():
			s = s + job + ": "
			if "done." in self.job_status[job]:
				s = s + str(self.job_status[job]["done."]) + " done "
			if "starting..." in self.job_status[job]:
				s = s + str(self.job_status[job]["starting..."]) + " starting "
			if "running..." in self.job_status[job]:
				s = s + str(self.job_status[job]["running..."]) + " running "
			if "pending" in self.job_status[job]:
				s = s + str(self.job_status[job]["pending"]) + " pending "
			if "failed." in self.job_status[job]:
				s = s + str(self.job_status[job]["failed."]) + " failed "
			s = s + "\n"
		return s
	def print_long_job_status(self):
		return self.session.print_long_job_status()
	def reset_failed_jobs(self):
		return self.session.reset_failed_jobs()
	def abandon(self):
		self.session.set_status("abandoned.")
	def set_session_status(self,msg="running..."):
		self.session.set_status(msg)
	def get_session_status(self,r=False):
		return self.session.status
	def job_activate(self,j,msg="running..."):
		if j in self.session.jobs:
			self.session.jobs[j].set_status(msg)
	def job_done(self,j):
		self.job_activate(j,"done.")
	def next_job(self, ignore_dependencies=False):
		self.update_job_status()
		return self.session.next_job(self.retry_failed, ignore_dependencies)
	def require_job(self,j):
		while True:
			self.update_job_status()
			if str(j) in self.job_status:
				all_done = True
				for k in self.job_status[str(j)].keys():
					if k != "done.":
						if self.job_status[str(j)][k] > 0:
							all_done = False
				if all_done:
					break
			print "Required Job not done."
			print "\n\n\nWaiting to retry..."
			time.sleep(10)
		return True
	def do_log(self,b):
		if type(b) == bool:
			self.logging = b
	def msg(self,msg_type,txt,priority = 0):
		if self.logging:
			for c in self.containers:
				c.append(msg_type,priority,datetime.datetime.now(),self.current_job,txt)
			#if priority >= self.log_updated_on_priority:
			#	self.msgs.append((msg_type,priority,datetime.datetime.now(),self.current_job,str(txt),txt))
			##if priority >= self.log_updated_on_priority:
			#with open(self.path+"log.txt","a") as logfile:
			#	logfile.write(">> " + str(self.current_job) + " <<\n" + str(datetime.datetime.now()) + " [" + str(msg_type) +"/" + str(priority) +"]\n" + str(txt) + "\n\n")
	def log(self,txt,priority = 0):
		if self.logging:
			#self.logs.append(str(txt))
			self.msg('log',txt,priority=priority)
		if priority > 0:
			print txt
	def dbg(self,txt,priority = -1):
		if self.logging:
			#self.dbgs.append(str(txt))
			self.msg('Debug',txt,priority=priority)
		if priority > 0:
			print "Dbg: ",str(txt)
	def err(self,txt,priority = 0):
		if self.logging:
			#self.errors.append(str(txt))
			self.msg('Error',txt,priority=priority)
		if priority >= 0:
			print "Error: ",str(txt)
	def clear(self):
		self.errors = ListContainer()
		self.logs = ListContainer()
		self.dbgs = ListContainer()
		self.msgs = ListContainer()
		for c in self.containers:
			c.clear()
		self.variables = {}
	def report(self,silent =False):
		#self.reportHTML()
		s = "          ---- Report ----\n\n"
		for m in self.containers[0].list:
			if m[1] > 0:
				s = s + ">> " + str(m[3]) + " <<\n" + str(m[2]) + " [" + str(m[0]) +"/" + str(m[1]) +"]\n" + str(m[4]) + "\n\n"
				if not silent:
					print str(m[1])," [",m[0],"] ",m[2]
		s = s + "\n          ----        ----\n"
		return s
	def reportHTML(self):
		rep = html_reporter.Reporter()
		for m in self.containers[0]:
			if m[1] > 0:
				rep.add(str(m[3]), {'date': m[2], 'type': str(m[0]), 'priority': m[1], 'msg': m[4]})
		rep.render(self.path+"report.html")
	def job(self,j):
		"""
		TODO: rename to something else
		"""
		self.current_job = str(j)
	def subjob(self,j):
		jj = self.current_job+ "/" + str(j)
		self.current_job = str(jj)
	def sibjob(self,j):
		"""
		Sibling Job

		Is on the same level as the previous job (ie. a child of its parent)
		"""
		if "/" in self.current_job:
			jj = self.current_job[:self.current_job.rfind("/")]
			self.current_job = jj
		jj = self.current_job + "/" + str(j)
		self.current_job = str(jj)
	def superjob(self):
		if "/" in self.current_job:
			jj = self.current_job[:self.current_job.rfind("/")]
			self.current_job = jj
	def save(self,name,val):
		self.log("Saving to "+self.path+str(name)+".p",1)
		f = open(self.path+str(name)+".p","w")
		pickle.dump(val,f)
		f.close()
	def dumpheap(self):
		h = hpy()
		h.heap().dump(self.path+"_heap_"+str(self.current_job.replace("/","_"))+".txt", "a")
		f = open(self.path+"_heap_"+str(self.current_job).replace("/","_")+"_str.txt","w+")
		f.write(str(h.heap()))
		f.close()
	def next(self):
		return self.run("next")
	def autorun(self):
		while True:
			try:
				if self.session.get_status() == "abandoned.":
					raise Exception("This session is abandoned.")
				next_job = self.session.next_job()
				next_job.set_status("starting...")
				try:
					print "Job " + str(next_job.job_number), ":"
					for dep in next_job.dependencies:
						self.require_job(dep)
				except:
					next_job.set_status("pending")
				next_job.run(self.parameters)
			except Exception as e:
				if e.args[0] == "End of Jobs":
					print "End of Jobs"
					pass
					break
				if e.args[0] == "This session is abandoned.":
					print "This session is abandoned."
					pass
					break
				else:
					print "Error occured:"
					print e
					raise
	def run(self,parameters=[],job=False):
		self.clear()
		if self.session == False:
			run = 1
			path = self.folder+"run/run"+str(run)+"/"
			while os.path.exists(path):
				run = run + 1
				path = self.folder+"run/run"+str(run)+"/"
			self.path = path
			os.makedirs(self.path)
			source_file = self.folder+self.main
			_Project = self
			l = locals()
			if job == "next":
				_job = self.next_job()
			else:
				_job = job
			print source_file
			out = []
			l["parameters"] = parameters
			l["ThisProject"] = self
			l["_out"] = out
			l["_job"] = _job
			l["_current_project"] = self
			l["__file__"] = self.folder+self.main
			l["path"] = self.path
			l["job_path"] = self.path+_job
			l["__name__"] = "__main__"
			l["tmp_path"] = ni.config.tmp_path
			self.containers.append(LogContainer(self.path+_job + '_log.txt'))
			if os.path.exists(".git/refs/heads/master"):
						log('Last ni.toolbox Revision: ' +open(".git/refs/heads/master").read() + " from: " + time.ctime(os.stat(".git/refs/heads/master").st_mtime) + " (or slightly later)",1)
			self.log('Executing file ' + self.folder+self.main + ' (on '+self.hostname+':'+self.instancename+') md5:'+str(hashlib.md5(open(self.folder+self.main, 'rb').read()).hexdigest())+ '\n Last changed on: ' + time.ctime(os.stat(self.folder+self.main).st_mtime),1)
			shutil.copy2(self.folder+self.main, self.path+self.main)
			self.log('Copied to: ' + self.path+self.main)
			self.job_activate(_job)
			try:
				#exec(compile(open(self.folder+self.main).read(), self.folder+self.main, 'exec'),globals(),l)
				execfile(source_file,l)
			except:
				#e = sys.exc_info()
				variables = VariableContainer()
				for o in l.keys():
					variables.__dict__[o] = l[o]
				self.variables = variables #l
				#self.save('variables',l)
				self.err(traceback.format_exc())
				self.log("Writing to logfile...")
				alert("Error! See logfile.")
				self.job_activate(_job,"failed.")
				with open(self.path+_job + '_error.txt',"w") as logfile:
					logfile.write(traceback.format_exc())
					logfile.write(self.report(silent=True))
				raise
			alert("Success.")
			self.job_done(_job)
			#self.save('variables',l)
			self.log("Writing to logfile...")
			with open(self.path+"log.txt","w") as logfile:
				logfile.write(self.report(silent=True))
			self.clear()
			variables = VariableContainer()
			for o in l.keys():
				variables.__dict__[o] = l[o]
			self.variables = variables#l
			return variables
		else:
			if str(job) in self.session.jobs:
				self.session.jobs[job].run()
	def save_html(self,path="project.html"):
		view = self.html_view()
		view.render(self.folder+path)
	def html_view(self):
		view = ni.View()
		view.title = self.name + " ni. toolbox project"
		view.add("/hidden/folder",self.folder)
		s = "<h1>"+self.name+"</h1>"
		if hasattr(self,'description'):
			s = s + self.description
		if hasattr(self,'logo'):
			s = s + "<div style=\"float: right;\"><img src=\""+self.logo+"\"></div>"
		s = s + "<h2>Sessions:</h2><ul>"
		for sess in range(len(self.sessions)):
			self.sessions[sess].save_html("session.html")
			s = s + "<li><a href=\""+self.sessions[sess].path[len(self.folder):]+"session.html\">"+str(self.sessions[sess].path[len(self.folder+"sessions/"):])+"</a></li>"
		s = s + "</ul>"
		view.add("",s)
		return view
	def execute(self,code,local_vars = {},session=False):
		if not "parameters" in local_vars:
			local_vars["parameters"] = []
			if type(session) == bool:
				session = TemporarySession(self,"Setup")
			local_vars["_current_project"] = self
			local_vars["_current_session"] = session
			local_vars["_current_job"] = TemporaryJob(self,session,"Preamble")
		try:
			exec code in local_vars
		except:
			raise
		return local_vars
	def parse_job_file(self,filename,session):
		new_files = {}
		quantifiers = {}
		tagged_lines = []
		preamble = []
		ordered_jobs = []
		dependencies = {}
		with open(filename) as source_file:
			source = source_file.readlines()
			job = "#preamble"
			last_indent = ""
			job_indent = ""
			job_indent_stack = []
			job_stack = []
			indent_specified = True
			for l in range(len(source)):
				indent = re.match(r'^([\t ]*)',source[l]).group(1)
				line = re.match(r'^[\t ]*(.*)',source[l]).group(1)
				if not indent_specified:
					if not indent.startswith(job_indent):
						raise Exception("Parsing Error for indent at line "+str(l)+" in file: "+filename)
					job_indent = indent
					indent_specified = True
				if not indent.startswith(job_indent) and not source[l].strip() == "":
					while indent != job_indent:
						_job = job
						job = job_stack.pop()
						job_indent = job_indent_stack.pop()
				if line.startswith("job "):
					m = re.match(r'job *((\".*\")|(\'.*\')|(\"\"\".*\"\"\"))(.*:)',line)
					if m:
						name = m.group(1)
						if name.startswith('\"\"\"'):
							name = name[3:-3]
						else:
							name = name[1:-1]
						quantifier = m.group(5)
						job_stack.append(job)
						job_indent_stack.append(job_indent)
						job = name
						indent_specified = False
						quantifiers[job] = re.findall(r'for (.*?) in (.*?)(?=(?= for)|:)',quantifier)
						dependencies[job] = []
						ordered_jobs.append(job)
					else:
						raise Exception("Parsing Error in job ... : declaration at line "+str(l)+" in file: "+filename)
				elif line.startswith("require "):
					m = re.match(r'require *((\".*\")|(\'.*\')|(\"\"\".*\"\"\"))',line)
					if len(ordered_jobs) > 1:
						r = ordered_jobs[-2]
					else:
						r = ""
					if m:
						r = m.group(1)
						if name.startswith('\"\"\"'):
							r = r[3:-3]
						else:
							r = r[1:-1]
					dependencies[job].append(r)
				else:
					real_indent = indent[len(job_indent):]
					real_line = real_indent + line
					new_files[job] = real_line
					if job == "#preamble":
						preamble.append(real_line)
					tagged_lines.append((job,real_line))
		if len(set(ordered_jobs)) <= 1:
			raise Exception("Not a job file")
		preamble_vars = self.execute("\n".join(preamble),session=session)
		unique_jobs = set(new_files.keys())
		independent_jobs = []
		for j in ordered_jobs:
			independent = True
			for j_2 in ordered_jobs:
				if not j == j_2 and j_2.startswith(j):
					independent = False
			if independent and not j in independent_jobs:
				if len(independent_jobs) > 0:
					pass #dependencies[j] = [independent_jobs[-1]]
				else:
					pass #dependencies[j] = []
				independent_jobs.append(j)
		jobs = []
		j_nr = 0
		for i_j in independent_jobs:
			print "---------------------------------------"
			quantified_parameters = [""]
			if i_j in quantifiers:
				for q in quantifiers[i_j]:
					new_quantified_parameters = []
					#print "--> ",q[0]," = ",q[1],"[i]"
					#print q[1], " = ", eval(q[1],preamble_vars)
					for qp in quantified_parameters:
						for qi in eval(q[1],preamble_vars):
							new_quantified_parameters.append(qp + str(q[0]) + " = " + str(qi)+ "\n")
					quantified_parameters = new_quantified_parameters
			for qp in quantified_parameters:
				code = "\n".join(preamble) + "\n" + qp
				for (tag,line) in tagged_lines:
					if i_j.startswith(tag):
						code = code + line + "\n"
				jobs.append({'nr':j_nr,'job':i_j,'quantified_parameters': quantified_parameters,'code':code,'dependencies':dependencies[i_j]})
				j_nr = j_nr + 1
		return jobs

try:
	_current_project.name
except:
	_current_project = Project()

def load(path):
	global _current_project
	_current_project = Project(path)
	return _current_project
def log(txt,priority = 0):
	_current_project.log(txt,priority)
def dbg(txt,priority = -1):
	_current_project.dbg(txt,priority)
def err(txt,priority = 0):
	_current_project.err(txt,priority)
def report(silent=False):
	return _current_project.report(silent)
def run():
	_current_project.run()

def save(name,val):
	_current_project.save(name,val)

def job(j):
	_current_project.job(j)

def subjob(j):
	_current_project.subjob(j)

def sibjob(j):
	_current_project.sibjob(j)

def superjob():
	_current_project.superjob()

def dumpheap():
	_current_project.dumpheap()
def do_log(b):
	_current_project.do_log(b)

def require_job(j):
	return _current_project.require_job(j)