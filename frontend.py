#!/usr/bin/python
# coding:UTF-8

import ni
import glob 
import ni.config
import re 
import time
import socket, os, sys, subprocess, readline
from copy import copy
import warnings
from threading  import Thread, Event
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty  # python 3.x

"""
class CursesWindow(object):
    def __enter__(project):
        curses.initscr()

    def __exit__(self):
        curses.endwin()
"""


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', str(text)) ]

def natural_sorted(l):
	ll = copy(l)
	ll.sort(key=natural_keys)
	return ll

ON_POSIX = 'posix' in sys.builtin_module_names

def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()

cli_options = ["help"]
def completer(text, state):
    global cli_options
    for cmd in cli_options:
        if cmd.startswith(text):
            if not state:
                return cmd
            else:
                state -= 1
import curses

def getProjects(s):
	projects = []
	ls = os.listdir(".")
	for d in ls:
		if d.startswith(s) and os.path.isdir(d):
			ls_2 = os.listdir(d)
			if 'main.py' in ls_2:
				projects.append(d)
	return projects

def globalCommands(y=0,x=0,max_y=2,max_x=80):
	global cli_options
	cli_options = ["help","status","setup jobs","autorun"]
	for f in os.listdir("."):
		cli_options.append("load "+f)
	cmd = ":"
	x = 0
	input_window = curses.newpad(2*max_y, 2*max_x)
	curses.cbreak()
	#input_window.nodelay(1)
	input_window.clear()
	input_window.addstr(0,0, " "*max_x, curses.A_REVERSE)
	input_window.addstr(1,0, " "*max_x, curses.A_REVERSE)
	#input_window.addstr(0,0, ":", curses.A_REVERSE)
	input_window.scrollok(True)
	input_window.refresh(0,0,y,x,y + max_y-1, x + max_x-1)
	while True:
		inp = input_window.getch()
		if inp > -1:
			if inp == 10:
				break
			elif inp == 127:
				if len(cmd) > 1:
					cmd = cmd[:-1]
			elif inp <= 256:
				cmd = cmd + chr(inp)
			else:
				cmd = cmd + "#"
			input_window.addstr(0,0," "*max_x, curses.A_REVERSE)	
		input_window.addstr(1,0, time.ctime(), curses.A_REVERSE)
		input_window.addstr(0,0, cmd, curses.A_REVERSE)
		input_window.refresh(0,0,y,x,y + max_y-1, x + max_x-1)
	cmd = cmd[1:]
	if cmd.strip().startswith("load "):
		print "loading: ",cmd.strip()[5:]
		project = ni.tools.project.load(cmd.strip()[5:])
	if cmd.strip().startswith("jobs"):
		content = curses.newpad(1000, 100)
		l = 0
		if type(project) != bool:
			for key in natural_sorted(project.jobs.keys()):
				j = project.jobs[key]
				content.addstr(l,0,j["status"])
				l = l + 1
				content.addstr(l,0,"\t"+str(j["nr"])+str(j["job"]))
				l = l + 1
				if j["dependencies"] != []:
					content.addstr(l,0,"\tDepends on: " + " and ".join(j["dependencies"]))
					l = l + 1
				content.addstr(l,0,"\t\t"+str(j["file"]))
				l = l + 1
				if "host" in j:
					content.addstr(l,0,"\truns on: " + j["host"])
					l = l + 1
				if "time" in j:
					content.addstr(l,0,"\truns since: " + j["time"])
					l = l + 1
		y = 0
		while True:
			content.refresh(y,0,2,0,max_y-2,max_x)
			x = content.getch()
			if chr(x) == 'a' or chr(x) == ' ':
				if y < l+1:
					y= y + 1
			if chr(x) == 'd':
				if y > 0:
					y= y - 1
			content.refresh(y,0,2,0,max_y-2,max_x)
			if x == 10 or chr(x) == 'q':
				break
	if cmd.strip().startswith("status"):
		if type(project) != bool:
			project.print_job_status()
	if cmd.strip().startswith("setup jobs"):
		if type(project) != bool:
			project.setup_jobs()
	if cmd.strip().startswith("autorun"):
		if type(project) != bool:
			project.autorun()
	if cmd == "python":
		python_window = curses.newpad(2*max_y, max_x)
		lines = [""]*max_y
		while 1:
			try:
				l = raw_input('python > ')
				lines.append('python > '+l)
			except EOFError:   # Catch Ctrl-D
				break
			if l == "q" or l == "quit":
				break
			else:
				try:
					lines.append(str(eval(l)).split("\n"))
				except:
					exception = sys.exc_info()
					lines.append("ERROR: "+str(exception[1]))
			for l in range(max_y):
				python_window.addstr(l,0,lines[l-max_y])
			python_window .refresh(0,0,y,x,y + max_y-1, x + max_x-1)
		del python_window 
	del input_window


run_prefixes = [ni.config.get("queue_command", "/usr/bin/qsub"),ni.config.get("run_command", os.getcwd()+"/run.sh"),os.getcwd()+"/run.py"]

class CursesWindow:
	def __init__(self):
		self.myscreen = False
		pass
	def getmaxyx(self):
		if self.myscreen == False:
			return (0,0)
		else:
			return self.myscreen.getmaxyx()
	def __enter__(self):
		self.myscreen = curses.initscr()
		curses.halfdelay(1)
		curses.noecho()
		curses.start_color()
		curses.init_pair(1, curses.COLOR_RED, curses.COLOR_WHITE)
		curses.cbreak()
		curses.curs_set(0)
		self.myscreen.keypad(1)
		self.max_y = self.myscreen.getmaxyx()[0]
		self.max_x = self.myscreen.getmaxyx()[1]
		return self
	def __exit__(self, type, value, tb):
		curses.endwin()

class CursesPad:
	def __init__(self):
		self.myscreen = False
	def getmaxyx(self):
		if self.myscreen == False:
			return (0,0)
		else:
			return self.myscreen.getmaxyx()
	def __enter__(self):
		self.myscreen = curses.initscr()
		curses.halfdelay(1)
		curses.noecho()
		curses.start_color()
		curses.init_pair(1, curses.COLOR_RED, curses.COLOR_WHITE)
		curses.cbreak()
		curses.curs_set(0)
		self.myscreen.keypad(1)
		return self
	def __exit__(self, type, value, tb):
		curses.endwin()

class FrontendPanel:
	def __init__(self, frontend, name="tab"):
		self.frontend = frontend
		self.name = name
		self.pad = curses.newpad(256, 100)
		self.x = 0
		self.y = 0
		self.max_y = 100
		self.selection = None
		self.commands = []
		self.msg = ""
	def refresh(self,from_y,from_x,to_y,to_x):
		self.update()
		self.pad.refresh(self.y,self.x,from_y,from_x,to_y,to_x)
		return self.pad
	def update(self):
		return self.pad
	def key(self,inp):
		if inp == curses.KEY_DOWN:
			if self.y < self.max_y+1:
				self.y= self.y + 1
		elif inp == curses.KEY_UP:
			if self.y > 0:
				self.y = self.y - 1
		return False
	def display_name(self):
		return self.name
	def cmd(self,command):
		if command == "close":
			self.name = "!"+self.name
		return False
	def close(self):
		self.frontend.tabs.remove(self)

class TextPanel(FrontendPanel):
	def __init__(self, frontend, name="tab", text=""):
		FrontendPanel.__init__(self,frontend)
		self.frontend = frontend
		self.name = name
		self.pad = curses.newpad(256, 100)
		self.x = 0
		self.y = 0
		self.max_y = 100
		self.max_x = 100
		self.padding = 4
		self.selection = None
		self.line_numbers = True
		self.commands = ["save ","load ","close"]
		self.filename = ""
		self.cursor_x = 0
		self.cursor_y = 0	
		self.set_text(text)
	def set_text(self,text):
		if type(text) == str:
			self.text = text.replace("\t","    ").split("\n")
		elif type(text) == list:
			self.set_text("\n".join(text))
		else:
			self.set_text(str(text))
	def refresh(self,from_y,from_x,to_y,to_x):
		self.update()
		self.pad.addstr((to_y-from_y), 0, " "+self.filename + " " + self.msg, curses.A_REVERSE)
		self.pad.refresh(0,self.x,from_y,from_x,to_y,to_x)
		return self.pad
	def update(self):
		l = 0
		self.pad.clear()
		y = -1
		if type(self.text) == str:
			text = self.text.split("\n")
		elif type(self.text) == list:
			text = self.text
		else:
			text = str(self.text).split("\n")
		for line in text:
			y = y + 1
			if y < self.y:
				continue
			if self.line_numbers:
				padding = self.padding - len(str(y)+" ")
				if padding < 0:
					padding = 0
				self.pad.addstr(l, padding, str(y) + " " + line)
			else:
				self.pad.addstr(l, self.padding, line)
			if y == self.cursor_y:
				if self.frontend.input_mode == "hotkeys":
					try:
						if self.cursor_x < len(line):
							self.pad.addstr(l, self.padding + self.cursor_x, line[self.cursor_x], curses.A_REVERSE)
						else:
							self.pad.addstr(l, self.padding + self.cursor_x, " ", curses.A_REVERSE)
					except:
						pass
			l = l + 1
			if l > 255:
				break
		return self.pad
	def key(self,inp):
		if self.frontend.input_mode == "hotkeys":
			if inp == 127 or inp == curses.KEY_BACKSPACE:
				try:
					if type(self.text) != list:
						self.text = self.text.split("\n")
					self.text[self.cursor_y] = self.text[self.cursor_y][:(self.cursor_x-1)] + self.text[self.cursor_y][self.cursor_x:]
					self.cursor_x = self.cursor_x - 1
					return True
				except:				
					pass
			elif inp == curses.KEY_DOWN:
				if self.cursor_y < self.max_y+1:
					self.cursor_y= self.cursor_y + 1
					if type(self.text) == list and len(self.text) <= self.cursor_y:
						self.text.append("")
				return True
			elif inp == curses.KEY_UP:
				if self.cursor_y > 0:
					self.cursor_y = self.cursor_y - 1
				return True
			elif inp == curses.KEY_LEFT:
				if self.cursor_x > 0:
					self.cursor_x = self.cursor_x - 1
				return True
			elif inp == curses.KEY_RIGHT:
				if self.cursor_x < self.max_x+1:
					self.cursor_x = self.cursor_x + 1
				return True
			elif inp == 27:
				self.frontend.input_mode = "shell"
				if hasattr(self,'on_exit'):
					self.save(self.filename)
					exec self.on_exit
				if hasattr(self,'on_exit_close'):
					self.close()
				self.msg = " Don't forget to type \"save\" to save."
				return True
			else:
				try:
					c = chr(inp)
					if type(self.text) != list:
						self.text = self.text.split("\n")
					self.text[self.cursor_y] = self.text[self.cursor_y][:self.cursor_x] + c + self.text[self.cursor_y][self.cursor_x:]
					self.cursor_x = self.cursor_x + 1
					return True
				except:
					pass
		else:
			if inp == ord('n'):
				self.line_numbers = not self.line_numbers
				self.update()
				return True
			if inp == curses.KEY_DOWN:
				if self.y < self.max_y+1:
					self.y= self.y + 1
			elif inp == curses.KEY_UP:
				if self.y > 0:
					self.y = self.y - 1
			return False
	def cmd(self,command):
		if command == "edit":
			self.frontend.input_mode = "hotkeys"
			self.msg = " Press <ESC> to end edit mode"
		elif command == "save" and self.filename != "":
			self.save(self.filename)
		elif command == "close":
			self.close()
		elif command.startswith("save ") and command != "save ":
			with open(command[5:],"w") as f:
				self.filename = command[5:]
				self.save(command[5:])
		return False
	def save(self,filename):
		if filename[0] == ":":
			if type(self.text) == list:
				self.text = "\n".join(self.text)
				exec filename[1:] + " = self.text"
			else:
				exec filename[1:] + " = self.text"
		else:
			with open(filename) as f:
				if type(self.text) == list:
					f.write("\n".join(self.text))
				else:
					f.write(self.text)

class DynamicTextPanel(TextPanel):
	def __init__(self, frontend, code="self.frontend.project.name", name="tab", text=""):
		FrontendPanel.__init__(self,frontend)
		self.frontend = frontend
		self.code = code
		self.text = text
		self.name = name
		self.pad = curses.newpad(256, 100)
		self.x = 0
		self.y = 0
		self.max_y = 100
		self.padding = 0
		self.error = ""
		self.selection = None
	def refresh(self,from_y,from_x,to_y,to_x):
		self.update()
		self.pad.refresh(self.y,self.x,from_y,from_x,to_y,to_x)
		return self.pad
	def update(self):
		try:
			self.text = eval(self.code)
		except:
			self.text = self.error
		l = self.padding
		if type(self.text) == str:
			for line in self.text.split("\n"):
				self.pad.addstr(l, self.padding, line)
				l = l + 1
		if type(self.text) == list:
			for o in self.text:
				for line in o.split("\n"):
					self.pad.addstr(l, self.padding, line)
					l = l + 1
		return self.pad
	def key(self,inp):
		if inp == curses.KEY_DOWN:
			if self.y < self.max_y+1:
				self.y= self.y + 1
		elif inp == curses.KEY_UP:
			if self.y > 0:
				self.y = self.y - 1
		return False


class JobViewPanel(FrontendPanel):
	def __init__(self, frontend, filter=[], name="tab"):
		FrontendPanel.__init__(self,frontend)
		self.frontend = frontend
		self.filter = filter
		self.name = name
		self.l_max = 256
		self.pad = curses.newpad(self.l_max, 100)
		self.x = 0
		self.y = 0
		self.selection = None
	def refresh(self,from_y,from_x,to_y,to_x):
		self.update()
		self.pad.refresh(0,0,from_y,from_x,to_y,to_x)
		return self.pad
	def update(self):
		self.pad.clear()
		l = 1
		y = -1
		max_y = 0
		if type(self.frontend.project) != bool:
			for key in natural_sorted(self.frontend.project.jobs.keys()):
				j = self.frontend.project.jobs[key]
				if self.filter == [] or j.status in self.filter:
					if l >= 100:
						continue
					max_y = max_y + 1
					y = y + 1
					if y < self.y - 1:
						continue
					if y == self.y:
						self.selection = j
						self.pad.addstr(l,0,"\t[ "+str(j.job_number)+": "+str(j.job_name)+ " [" + j.status + "] ("+str(j.file)+") ]", curses.A_BOLD)
						l = l + 1
						if j.dependencies != []:
							self.pad.addstr(l,0,"\t\tDepends on: " + " and ".join(j.dependencies), curses.A_BOLD)
							l = l + 1
						if hasattr(j,'host'):
							self.pad.addstr(l,0,"\t\truns on: " + j.host, curses.A_BOLD)
							l = l + 1
						if hasattr(j,'process'):
							self.pad.addstr(l,0,"\t\tprocess: " + j.process, curses.A_BOLD)
							l = l + 1
						if hasattr(j,'last_touched'):
							self.pad.addstr(l,0,"\t\truns since: " + str(j.last_touched), curses.A_BOLD)
							l = l + 1
					else:
						self.pad.addstr(l,0,"\t"+str(j.job_number)+": "+str(j.job_name)+ " [" + j.status +"] ("+str(j.file)+")")
						l = l + 1
						if hasattr(j,'host') and j.status == "running...":
							self.pad.addstr(l,0,"\truns on: " + j.host)
							l = l + 1
		self.max_y = max_y
		self.pad.addstr(0,0, " Jobs: " +self.name+"    ")
		return self.pad
	def key(self,inp):
		if inp == curses.KEY_DOWN:
			if self.y < self.max_y+1:
				self.y= self.y + 1
		elif inp == curses.KEY_UP:
			if self.y > 0:
				self.y = self.y - 1
		self.update()
		return False
	def display_name(self):
		if self.filter == []:
			return self.name 
		if self.filter[0] in self.frontend.project.overall_status:
			return self.name +" ["+str( self.frontend.project.overall_status[self.filter[0]])+"]"
		else:
			return self.name 
class StatusViewPanel(FrontendPanel):
	def __init__(self, frontend, name="tab"):
		FrontendPanel.__init__(self,frontend)
		self.frontend = frontend
		self.name = name
		self.l_max = 256
		self.pad = curses.newpad(self.l_max, 100)
		self.x = 0
		self.y = 0
		self.selection = None
	def refresh(self,from_y,from_x,to_y,to_x):
		self.update()
		self.pad.refresh(self.y,self.x,from_y,from_x,to_y,to_x)
		return self.pad
	def update(self):
		self.pad.clear()
		l = 1
		for job in natural_sorted(self.frontend.project.job_status.keys()):
			s = "\t\t"
			if "done." in self.frontend.project.job_status[job]:
				s = s + str(self.frontend.project.job_status[job]["done."]) + " done "
			if "starting..." in self.frontend.project.job_status[job]:
				s = s + str(self.frontend.project.job_status[job]["starting..."]) + " starting "
			if "running..." in self.frontend.project.job_status[job]:
				s = s + str(self.frontend.project.job_status[job]["running..."]) + " running "
			if "pending" in self.frontend.project.job_status[job]:
				s = s + str(self.frontend.project.job_status[job]["pending"]) + " pending "
			if "failed." in self.frontend.project.job_status[job]:
				s = s + str(self.frontend.project.job_status[job]["failed."]) + " failed "
			print ""
			self.pad.addstr(l,0,"\t " + job)
			self.pad.addstr(l+1,0,s)
			l = l + 2
		self.max_y = l
		return self.pad


class Frontend:
	def __init__(self):
		self.project = False
		self.max_threads = 1
		try:
			self.max_started_jobs = int(ni.config.get("frontend.max_started_jobs", 50))
		except:
			self.max_started_jobs = 50
		self.autorun_flag = False
		self.html_update_flag = False
		self.error_msgs = []
		self.error_times = []	
		self.update_tick = 0	
		self.processes = []
		self.process_out = []
		self.process_thread = []
		self.process_queue = []
		self.process_jobs = []
		self.job_threads = []
		self.run_prefixes = [ni.config.get("queue_command", "/usr/bin/qsub"),ni.config.get("run_command", os.getcwd()+"/run.sh"),os.getcwd()+"/run.py"]
		self.dead_processes = []
		self.parameters = ""
	def error_msg(self,msg):
		if len(self.error_msgs) > 0:
			if msg == self.error_msgs[-1]:
				return
		self.error_msgs.append(msg)
		self.error_times.append(time.time())
		if len(self.error_msgs) > 20:
			del self.error_msgs[0]
			del self.error_times[0]
	def update(self):
		while not self.shutdown.is_set():
			try:
				time.sleep(0.1)
				self.update_tick = self.update_tick + 1
				if self.update_tick > 100000:
					self.update_tick = 0
				if type(self.project) != bool:
					self.project.update_job_status()
					if self.html_update_flag == True and self.update_tick % 50 == 0:
						self.project.save_html()
					try:
						self.max_started_jobs = int(ni.config.get("frontend.max_started_jobs", 50))
					except:
						pass
					if self.autorun_flag and self.project.less_running_than(self.max_started_jobs):
						try:
							self.autorun()
						except Exception as e:
							self.error_msg(str(e))
							pass
				self.update_processes()
			except Exception as e:
				self.error_msg("Error: "+str(e))
				pass
	def autorun(self):#"/usr/bin/qsub",
		if type(self.project) == bool:
			raise Exception("No Project selected")
		if type(self.project.session) == bool:
			raise Exception("No Session selected")
		job = self.project.session.next_job()
		self.run(job)
	def add_thread(self, f, args=()):
		t = Thread(target = f, args=args)
		t.daemon = True # thread dies with the program
		t.start()
		self.job_threads.append(t)
	def lock_input(self, f, *args, **kwargs):
		self.input_locked = True
		with warnings.catch_warnings(record=True) as w:
			try:
				f(*args, **kwargs)
			except Exception as e:
				self.error_msg("Error: "+str(e))
			if len(w) > 0:
				for ww in w:
					self.error_msg("Warning: "+str(ww.message))
		self.input_locked = False
		self.lock_msg = " (waiting)"
	def wait(self, t):
		time.sleep(t)
	def run(self,job, override_requirements=False, run_local = False):
		if type(self.project) == bool:
			raise Exception("No Project selected")
		if type(self.project.session) == bool:
			raise Exception("No Session selected")
		if len(self.processes) > self.max_threads:
			raise Exception("Too many jobs running.")
		if job.can_run() or override_requirements:
			if run_local:
				t = Thread(target = job.run)
				t.daemon = True # thread dies with the program
				t.start()
				self.job_threads.append(t)
			else:
				start_line = copy(self.run_prefixes)
				start_line.extend([self.project.folder, self.project.session.path, str(job.job_number)])
				self.processes.append(subprocess.Popen(start_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
				start_line = " ".join(start_line)
				self.process_jobs.append(job)
				job.process = "pid:" + str(self.processes[-1].pid)
				job.set_status("starting...")
				q = Queue()
				t = Thread(target=enqueue_output, args=(self.processes[-1].stdout, q))
				self.process_queue.append(q)
				self.process_thread.append(t)
				self.process_out.append(start_line+"\n")
				t.daemon = True # thread dies with the program
				t.start()
		else:
			raise Exception("Dependencies not satisfied")
	def update_processes(self):
		for i in range(len(self.processes)):
			p = self.processes[i]
			try:
				p.poll()
				os.kill(p.pid, 0) # Sending a "0" signal to the process. This will fail if process has ended.
			except:
				self.dead_processes.append(self.process_out[i])
				del self.processes[i]
				del self.process_out[i]
				del self.process_queue[i]
				del self.process_thread[i]
	def process_list(self):
		s = [" "+str(len(self.dead_processes))+" finished processes "," "+str(len(self.processes))+" processes: "]
		for j in self.job_threads:
			s.append(str(j))
		for i in range(len(self.processes)):
			p = self.processes[i]
			try:
				p.poll()
				os.kill(p.pid, 0) # Sending a "0" signal to the process. This will fail if process has ended.
			except:
				self.dead_processes.append(self.process_out[i])
				del self.processes[i]
				del self.process_out[i]
				del self.process_queue[i]
				del self.process_thread[i]
			try:
				out = self.process_queue[i].get_nowait()
				if out.startswith("Your job "): # and out.endswith("has been submitted"):
					#"Your job 3388895 (\"ls\") has been submitted"
					self.process_jobs[i].process = out
					pass
				self.process_out[i] = process_out[i] +  out
			except:
				pass
			if len(self.process_out) > i:
				s.append(str(i)+ " " + str(p.pid) + "\n" + self.process_out[i])
			else:
				s.append(str(i)+ " " + str(p.pid) + "\n 00")
		for i in range(len(self.dead_processes)):
			s.append("+"+str(self.dead_processes[i][-1]))
			s.append(" "+str(self.dead_processes[i][-2]))
		s.append("----")
		return s
	def cmd(self,cmd):
		if cmd == "show jobs":
			self.display_mode = "jobs"
		elif cmd.startswith("job "):
			if isinstance(self.tabs[self.selected_tab].selection,ni.tools.project.Job):
				if cmd == "job reset":
					self.tabs[self.selected_tab].selection.set_status("pending")
				elif cmd == "job set to done":
					self.tabs[self.selected_tab].selection.set_status("done.")
				elif cmd == "job run":
					self.run(self.tabs[self.selected_tab].selection, override_requirements=True, run_local=False)
		elif cmd == "hotkey mode":
			self.input_mode = "hotkeys"
		elif cmd == "run":
			try:
				self.autorun()
			except:
				e = sys.exc_info()
				self.error_msg(str(e))
				pass
		elif cmd.startswith("run"):
			if cmd[4:].strip() in self.project.session.jobs:
				self.run(self.project.session.jobs[cmd[4:].strip()], override_requirements=True, run_local=False)
			else:
				self.error_msg("No such job "+cmd[4:])
		elif cmd == "processes":
			self.display_mode = "processes"
			if self.processes_tab in self.tabs:
				self.tabs.remove(self.processes_tab)
			else:
				self.tabs.append(self.processes_tab)
				self.selected_tab = len(self.tabs) - 1
		elif cmd == "errors":
			self.display_mode = "errors"
			if self.error_tab in self.tabs:
				self.tabs.remove(self.error_tab)
			else:
				self.tabs.append(self.error_tab)
				self.selected_tab = len(self.tabs) - 1
		elif cmd == "config" or cmd == "configuration":
			if self.config_tab in self.tabs:
				self.tabs.remove(self.config_tab)
			else:
				self.tabs.append(self.config_tab)
				self.selected_tab = len(self.tabs) - 1
		elif cmd == "help":
			if self.help_tab in self.tabs:
				self.tabs.remove(self.help_tab)
			else:
				self.tabs.append(self.help_tab)
				self.selected_tab = len(self.tabs) - 1
		elif cmd == "html update on":
			self.html_update_flag = True
		elif cmd == "html update off":
			self.html_update_flag = False
		elif cmd == "save project to html":
			if type(self.project) != bool:
				#if type(project.session) != bool:
				self.project.save_html()
		elif cmd == "save session to html":
			if type(self.project) != bool:
				if type(self.project.session) != bool:
					self.project.session.save_html()
		elif cmd == "autorun on":
			self.autorun_flag = True
		elif cmd == "autorun off":
			self.autorun_flag = False
		elif cmd == "reset failed jobs":
			self.project.session.reset_failed_jobs()
		elif cmd == "reset starting jobs":
			self.project.session.reset_failed_jobs("starting...")
		elif cmd == "reset running jobs":
			self.project.session.reset_failed_jobs("running...")
		elif cmd == "set failed jobs to done":
			self.project.session.reset_failed_jobs("failed.",to="done.")
		elif cmd == "session abandon":
			self.project.session.abandon()
		elif cmd == "wait":
			self.lock_msg = "Waiting. This may take a while."
			self.add_thread(self.lock_input, (self.wait, 1))
		elif cmd == "setup session":
			params = self.project.get_parameters_from_job_file()
			if len(params) > 0:
				t = TextPanel(self,"Setup",params)
				t.filename = ":self.frontend.parameters"
				t.msg = " Press <ESC> to exit edit mode."
				t.on_exit_close = True
				t.on_exit = "self.frontend.cmd(\"setup session with parameters\")"
				self.tabs.append(t)
				self.selected_tab = len(self.tabs) - 1
				self.input_mode = "hotkeys"
			else:
				self.lock_msg = "No parameters found. Setting up jobs. This may take a while."
				self.add_thread(self.lock_input, (self.project.setup_jobs,self.parameters))
		elif cmd == "setup session with parameters":
			self.lock_msg = "Setting up jobs. This may take a while."
			self.add_thread(self.lock_input, (self.project.setup_jobs,self.parameters))
		elif cmd == "parameters":
			params = self.parameters
			t = TextPanel(self,"Parameters",params)
			t.filename = ":self.frontend.parameters"
			self.tabs.append(t)
			self.selected_tab = len(self.tabs) - 1
		elif cmd == "setup session with default parameters":
			self.lock_msg = "Setting up jobs. This may take a while."
			self.add_thread(self.lock_input, (self.project.setup_jobs, ))
		elif cmd == "update job files":
			self.lock_msg = "Setting up jobs. This may take a while."
			self.add_thread(self.lock_input, (self.project.session.update_job_files, ))
		elif cmd == "quit":
			self.shutdown.set()
			return
		elif cmd.startswith("config set ") and cmd != "config set ":
			try:
				words = cmd.split(" ")
				ni.config.user.set(words[2], words[3])
				self.error_msg("set "+str(words[2])+" to "+ words[3])
			except:
				self.error_msg("could not set "+str(words[2])+" to "+ words[3])
		else:
			self.set_msg(' Don\'t know what to do with ' + cmd)
		cmd = ""
	def set_msg(self, msg):
		self.msg = msg
	def main(self):
		self.shutdown = Event()
		histfile = os.path.join(os.environ["HOME"], ".ni_console_history")
		self.input_locked = False
		self.input_mode = "hotkeys"
		self.input_mode = "shell"
		self.lock_msg = " (waiting)"
		self.max_threads = 10
		self.msg = ""
		self.update_thread = Thread(target=self.update)
		self.update_thread.daemon = False
		self.update_thread.start()
		self.commands = ["show jobs","hotkey mode","run","autorun on","autorun off",
												"reset failed jobs","reset running jobs","reset starting jobs",
												"quit","restart","setup session","processes","save project to html"
												,"save session to html","html update on","html update off","update job files", "set failed jobs to done", "errors"]
		try:
			with CursesWindow() as myscreen:
				cmd = ""
				max_y = myscreen.getmaxyx()[0]
				max_x = myscreen.getmaxyx()[0]
				toolbar_window = curses.newpad(3, 500)
				status_window = curses.newpad(1, 500)
				content = curses.newpad(256, 100)
				l_max = 200
				self.display_mode = "menu"
				cmd_completion = ""
				y = 0
				quit = False
				self.processes_tab = DynamicTextPanel(self,"self.frontend.process_list()","processes")
				self.error_tab = DynamicTextPanel(self,"self.frontend.error_msgs","errors")
				self.config_tab = DynamicTextPanel(self,"'Configuration options:\\n(change with <config set ...>)\\n\\n\\t' + '\\n\\t'.join([str(a) +': '+ str(b) for (a,b) in zip(ni.config.keys(),[ni.config.get(k) for k in ni.config.keys()])])","config")
				self.config_tab.padding = 2
				self.menu_tab = DynamicTextPanel(self,""" "Change between [ tabs ] with the left and right arrow keys.\nChange between sessions with page up/down.\n\nProject description: "+ self.frontend.project.description + "\n Session Status: " + self.frontend.project.session.status""","Info")
				self.menu_tab.padding = 2
				self.menu_tab.error = """Change between [ tabs ] with the left and right arrow keys.
Change between sessions with page up/down.

Type "help" to see commands."""
				self.status_tab = StatusViewPanel(self,"status")
				self.help_tab = DynamicTextPanel(self,"""\"\"\"
Change [ tabs ] with the left and right key. Scroll up and down with the up and down keys.

Press <TAB> to complete a command automatically.

  You can open tabs with:
     errors, processes, config (also this tab with help)
  The same command closes the tab again.

  You can enable or disable flags if a session is selected:
     autorun on/off (currently \"\"\" + ("on" if self.frontend.autorun_flag else "off") + \"\"\")
     html update on/off  (currently \"\"\" + ("on" if self.frontend.html_update_flag else "off") + \"\"\")

Some other commands:
\t""" + "\n\t".join(self.commands)+ "\"\"\"","help")
				self.tabs = [
					self.menu_tab,
					self.status_tab,
					JobViewPanel(self,[],"all jobs"),
					JobViewPanel(self,["pending"],"pending jobs"),
					JobViewPanel(self,["starting..."],"starting jobs"),
					JobViewPanel(self,["running..."],"running jobs"),
					JobViewPanel(self,["failed."],"failed jobs"),
					JobViewPanel(self,["done."],"done jobs")
				]

				self.tabs[0].text = "Menu"
				self.selected_tab = 0
				while not self.shutdown.is_set():
					max_y = myscreen.getmaxyx()[0]
					max_x = myscreen.getmaxyx()[1]
					if self.display_mode == "menu":
						if self.project == False:
							content.clear()
							content.addstr(2, 0, "   "+"\n   ".join(p for p in getProjects(cmd)) )
							if len(getProjects(cmd)) > 0:
								cmd_completion = getProjects(cmd)[0][len(cmd):]
							else:
								cmd_completion = ""
							if cmd == "":
								cmd_completion = ""
							toolbar_window.addstr(1, 0, " "*max_x)
							toolbar_window.addstr(1, 1, "Select Project: "+cmd)
							toolbar_window.addstr(1, 1 + len("Select Project: "+cmd), cmd_completion, curses.A_BOLD)
							toolbar_window.refresh(0,0,0,0,2, max_x-1)
							content.refresh(0,0,2,0,max_y-2,max_x)
							x = toolbar_window.getch()
							toolbar_window.refresh(0,0,0,0,2, max_x-1)
							content.refresh(0,0,2,0,max_y-2,max_x)
							if x > -1:
								if chr(x) == "!":
									self.display_mode = "python"
								elif chr(x) == ":":
									globalCommands(0,0,2,max_x)
								elif x == 10:
									if len(getProjects(cmd)) > 0:
										self.project = ni.tools.project.load(getProjects(cmd)[0])
								elif x == 127:
									if len(cmd) > 0:
										cmd = cmd[:-1]
								else:
									cmd = cmd + chr(x)
							time.sleep(0.1)
						if self.project != False:
							selected_row = -1
							cmd = ""
							while True:
								toolbar_window.addstr(0, 0, " "*max_x, curses.A_REVERSE)
								toolbar_window.addstr(0, 0, "Project: "+self.project.folder+" "+self.display_mode, curses.A_REVERSE)
								if self.project.sessions != []:
									s = ""
									for i in range(len(self.project.sessions)):
										if self.project.sessions[i].status == "abandoned.":
											if self.project.session == self.project.sessions[i]:
												s = s + " (.)"
											else:
												s = s + "  . "
										else:
											if self.project.session == self.project.sessions[i]:
												s = s + " ["+str(i)+"]"
											else:
												s = s + "  "+str(i)+" "
									if selected_row == 0:								
										toolbar_window.addstr(1, 0, " "*max_x, curses.A_REVERSE)
										if self.autorun_flag:
											toolbar_window.addstr(1, 0, "[autorun on] Session: "+s+"   "+self.project.session.path+" ", curses.A_REVERSE)
										else:
											toolbar_window.addstr(1, 0, "[autorun off] Session: "+s+"   "+self.project.session.path+" ", curses.A_REVERSE)
									else:
										toolbar_window.addstr(1, 0, " "*max_x)
										if self.autorun_flag:
											toolbar_window.addstr(1, 0, "[autorun on] Session: "+s+"   "+self.project.session.path+" ")
										else:
											toolbar_window.addstr(1, 0, "[autorun off] Session: "+s+"   "+self.project.session.path+" ")
								else:
									toolbar_window.addstr(1, 0, "This project contains no sessions. Create a session by typing > setup session")
								modes = ["menu","status","jobs","jobs:pending","jobs:starting...","jobs:running...","jobs:done.","jobs:failed."]
								mode_display = ["menu","status","jobs"]
								for stat in ["pending","starting...","running...","done.","failed."]:
									if stat in self.project.overall_status:
										mode_display.append(stat+" ["+str( self.project.overall_status[stat])+"]")
									else:
										mode_display.append(stat)

								self.selected_tab = (self.selected_tab%len(self.tabs))

								s = ""
								for i in range(len(self.tabs)):
									if i == self.selected_tab:
										s = s + " [ "+ self.tabs[i].display_name() + " ] "
									else:
										s = s + "   "+ self.tabs[i].display_name() + "   "
								if selected_row == 1:
									toolbar_window.addstr(2, 0, " "*max_x, curses.A_REVERSE)
									toolbar_window.addstr(2,0,s, curses.A_REVERSE)
								else:
									toolbar_window.addstr(2, 0, " "*max_x)
									toolbar_window.addstr(2,0,s)
								toolbar_window.refresh(0,0,0,0,2, max_x-1)
								l = 0
								status_window.addstr(0, 0, ' ' * (max_x))
								status_window.refresh(0,0,max_y-2, 0,max_y-1, max_x-1)
								content.keypad(1)
								if self.input_mode == "hotkeys":
									curses.curs_set(0)
									self.tabs[self.selected_tab].refresh(3,0,max_y-1,max_x)
									inp = content.getch()
									if not self.tabs[self.selected_tab].key(inp):
										if inp == curses.KEY_DOWN:
											if y < l+1:
												y= y + 1
										if inp == curses.KEY_UP:
											if y > 0:
												y= y - 1
										if inp == curses.KEY_LEFT:
											self.selected_tab = self.selected_tab - 1
										if inp == curses.KEY_RIGHT:
											self.selected_tab = self.selected_tab + 1
										if inp == curses.KEY_PPAGE:
											if self.project.sessions != []:
												for i in range(len(self.project.sessions)):
													if self.project.session == self.project.sessions[i]:
														break
												i = (i + 1)%len(self.project.sessions)
												self.project.session = self.project.sessions[i]
										if inp == curses.KEY_NPAGE:
											if self.project.sessions != []:
												for i in range(len(self.project.sessions)):
													if self.project.session == self.project.sessions[i]:
														break
												i = (i - 1)%len(self.project.sessions)
												self.project.session = self.project.sessions[i]
										if inp == 9 or inp == ord(">"):
											self.input_mode = "shell"
										if inp == curses.KEY_BREAK:
											return
										if inp > -1:
											if inp == ord('a'):
												autorun_flag = not autorun_flag
											if inp < 256:
												if chr(inp) == 'q':
													del content
													self.display_mode = "menu"
													self.project = False
													quit = True
													break
												if chr(inp) == 'j' or  chr(inp) == '0':
													self.display_mode = "jobs"
													y = 0
												if chr(inp) == 'r' or  chr(inp) == '1':
													self.display_mode = "jobs:running..."
													y = 0
												if chr(inp) == 'd' or  chr(inp) == '2':
													self.display_mode = "jobs:done."
													y = 0
												if chr(inp) == 'f' or  chr(inp) == '3':
													self.display_mode = "jobs:failed."
													y = 0
												if chr(inp) == '.':
													self.display_mode = "processes"
													y = 0
												if chr(inp) == 'p' or  chr(inp) == '4':
													self.display_mode = "jobs:pending"
													y = 0
												if chr(inp) == 'q':
													self.project = False
												if chr(inp) == 'c':
													toolbar_window.addstr(1,0,"Do you want to start a new process? [y/n]", curses.A_REVERSE)
													toolbar_window.refresh(0,0,0,0,2, max_x-1)
													while True:
														y_n = toolbar_window.getch()
														if y_n == ord('y'):
															try:
																self.autorun()
															except:
																e = sys.exc_info()
																msg = str(e)
																content.clear()
																content.addstr(0,0,"ERROR " + str(e) + "\n",curses.color_pair(1))
																content.refresh(y,0,2,0,max_y-2,max_x)
																x_ = content.getch()
																while x_ == -1:
																	x_ = content.getch()
															finally:
																content.addstr(0,0,"Spawned process. Press [.] to view processes.")
																content.refresh(y,0,2,0,max_y-2,max_x)
															break
														if y_n == ord('n'):
															break
												elif chr(inp) == 's':
													self.display_mode = "status"	
								elif self.input_mode == "shell":
									input_window = curses.newpad(3, max_x)
									curses.cbreak()
									input_window.addstr(0,0, " "*max_x)
									input_window.keypad(1)
									commands = ["show jobs","hotkey mode","run","autorun on","autorun off","config set ",
												"reset failed jobs","reset running jobs","reset starting jobs",
												"quit","restart","setup session","setup session with default parameters","processes","save project to html"
												,"save session to html","html update on","html update off","update job files", "set failed jobs to done", "errors"]
									if isinstance(self.tabs[self.selected_tab].selection,ni.tools.project.Job):
										for c in ["run", "reset", "set to done"]:
											commands.append("job "+c)
									for c in ["abandon", "reset failed jobs", "reset all jobs"]:
										commands.append("session "+c)
									for func in dir(self.project):
										if callable(func):
											commands.append("project "+func.__name__)
									if cmd.startswith("config set"):
										for k in ni.config.keys(recursive = True):
											commands.append("config set "+str(k)+" "+str(ni.config.get(k,"not set")))
									for c in self.tabs[self.selected_tab].commands:
										commands.append(c)
									self.commands = commands
									completions = " | ".join([c for c in commands if c.startswith(cmd)])
									if isinstance(self.tabs[self.selected_tab].selection,ni.tools.project.Job):
										completions = "job " + str(self.tabs[self.selected_tab].selection.job_number) + " / " + completions 
									if len(completions) > max_x:
										completions = completions[:max_x]
									curses.curs_set(1)
									input_window.addstr(1,0,' '*max_x)
									input_window.addstr(0,0,' '*max_x)
									if cmd != "":
										input_window.addstr(0,0,completions)
									if len(self.error_msgs) > 0:
										input_window.addstr(2, 0, str(len(self.error_msgs)) + " " + str(self.error_msgs[-1]))
									self.tabs[self.selected_tab].refresh(3,0,max_y-4,max_x)
									if self.input_locked:
										input_window.addstr(1,0,' > ' + cmd + '  ' + self.lock_msg, curses.A_REVERSE)			
									else:
										input_window.addstr(1,0,' > ' + cmd)
									input_window.move(1,3+len(cmd))
									input_window.refresh(0,0,max_y-3,0,max_y-1, max_x-1)
									inp = input_window.getch()
									if not self.tabs[self.selected_tab].key(inp):
										if inp == curses.KEY_DOWN:
											if y < l+1:
												y= y + 1
										elif inp == curses.KEY_UP:
											if y > 0:
												y= y - 1
										elif inp == curses.KEY_LEFT:
											self.selected_tab = self.selected_tab - 1
											self.selected_tab = (self.selected_tab%len(self.tabs))
										elif inp == curses.KEY_RIGHT:
											self.selected_tab = self.selected_tab + 1
											self.selected_tab = (self.selected_tab%len(self.tabs))
										elif inp == curses.KEY_LEFT:
											for i in range(len(modes)):
												if self.display_mode == modes[i]:
													break
											i = (i - 1)%len(modes)
											self.display_mode = modes[i]
										elif inp == curses.KEY_RIGHT:
											for i in range(len(modes)):
												if self.display_mode == modes[i]:
													break
											i = (i + 1)%len(modes)
											self.display_mode = modes[i]
										elif inp == curses.KEY_PPAGE:
											if self.project.sessions != []:
												for i in range(len(self.project.sessions)):
													if self.project.session == self.project.sessions[i]:
														break
												i = (i + 1)%len(self.project.sessions)
												self.project.session = self.project.sessions[i]
										elif inp == curses.KEY_NPAGE:
											if self.project.sessions != []:
												for i in range(len(self.project.sessions)):
													if self.project.session == self.project.sessions[i]:
														break
												i = (i - 1)%len(self.project.sessions)
												self.project.session = self.project.sessions[i]
										elif inp == 9:
											for c in commands:
												if c.startswith(cmd):
													cmd = c
													break
										elif inp == curses.KEY_BREAK:
											return
										elif inp > 0:
											if inp == 127 or inp == curses.KEY_BACKSPACE:
												if len(cmd) > 0:
													cmd = cmd[:-1]
											elif inp != 10:
												try:
													cmd = cmd + chr(inp)
												except:
													pass
									input_window.addstr(1,0,' '*max_x)
									input_window.addstr(1,0,' > ' + cmd)
									if cmd != "":
										s = ""
										for c in commands:
											if c.startswith(cmd):
												s = s + c + " "
										input_window.addstr(0,0, s)
									if not self.input_locked and inp == 10 and cmd != "":
										if not self.tabs[self.selected_tab].cmd(cmd):
											self.cmd(cmd)
										if cmd == "quit":
											return
										cmd = ""
					if self.display_mode == "python":			
						python_window = curses.newpad(max_y, max_x)
						lines = [""]*max_y
						while 1:
							try:
								cmd = ""
								while True:
									python_window.addstr(max_y-5,0,' '*max_x)
									python_window.addstr(max_y-5,0,'eval  > ' + cmd)
									python_window.refresh(0,0,2,0,max_y-1, max_x-1)
									inp = python_window.getch()
									if inp == 10:
										break
									if inp > 0:
										if inp == 127:
											if len(cmd) > 0:
												cmd = cmd[:-1]
										else:
											try:
												cmd = cmd + chr(inp)
											except:
												pass
									python_window.addstr(max_y-5,0,' '*max_x)
									python_window.addstr(max_y-5,0,'eval > ' + cmd)
									python_window.refresh(0,0,2,0,max_y-1, max_x-1)
								lines.append('python > '+cmd)
							except EOFError:   # Catch Ctrl-D
								break
							if cmd == "quit":
								break
							else:
								if cmd != "":
									try:
										lines.extend(str(eval(cmd)).split("\n"))
									except:
										exception = sys.exc_info()
										lines.append("ERROR: "+str(exception[1]))
									cmd = ""
							for l in range(max_y-5):
								python_window.addstr(l,0,lines[l-max_y+5])
							python_window.refresh(0,0,2,0,max_y-1, max_x-1)
						del python_window 
						cmd = ""
						self.display_mode = "menu"
		except:
			self.shutdown.set()
			raise
		finally:
			self.shutdown.set()


def main():
	f = Frontend()
	if len(sys.argv) > 1:
		try:
			if sys.argv[1] in getProjects(sys.argv[1]):
				p = sys.argv[1]
				if p[-1] == "/":
					p = p[:-1]
				f.project = ni.tools.project.load(sys.argv[1])
			else:
				msg = "Could not load project " + sys.argv[1]
		except:
			raise
	f.main()

if __name__ == '__main__':
        main()
