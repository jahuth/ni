"""
.. module:: ni.tools.pickler
   :platform: Unix
   :synopsis: Provides a method to save arbitrary objects to text files.

.. moduleauthor:: Jacob Huth <jahuth@uos.de>

Objects that inherit from :class:`Picklable` will have `save` and `load` functions to save them to a file.
They can be loaded again (without having to specify which class they are) with `ni.tools.pickler.load(filename)`.


"""


import ni
import numpy as np
import re
import json
import pickle
from types import ModuleType
import glob

ignored_types = []#["ni.model.backend_glm.Fit","ni.model.backend_elasticnet.Fit"]

def escape(s):
	""" escapes strings for save usage within strings 

	can be inverted by eg.::

		>>> s = eval(" \" some text \"  ")
		<<<  " some text "

	"""
	if type(s) == str:
		return json.dumps(s)

def dumps(o):
	if isinstance(o, ModuleType):
		return "<module " + o.__name__+ ">"
	elif type(o) == list:
		return "[" + ", ".join([dumps(oo) for oo in o]) + "]"
	elif type(o) == dict:
		return "{" + ", ".join(["'"+oo+"': "+dumps(o[oo]) for oo in o]) + "}"
	elif type(o) == str:
		return "\"" + o +"\""
	elif type(o) == int or  type(o) == bool:
		return "" + str(o) +""
	elif type(o) == np.ndarray:
		return "np.array(" + str(o.tolist()) +")"
	elif type(o) == np.float64:
		return "" + str(o.tolist()) +""
	elif getattr(o,"dumps", None):
		return o.dumps() 
	else:
		return "pickle.loads(\"\"\""+pickle.dumps(o)+"\"\"\")"#\'"+ len(o) + "\'\n"

def loads(s):
	s = s.replace("np.ndarray","np.array")
	try:
		o = eval(s)
	except:
		return s
	try:
		for k in o.__dict__:
			if type(k) == str:
				try:
					o.__dict__[k] = loads(o.__dict__[k])
				except:
					pass
	except:
		return o
	return o

def prettyfy(s):
	""" creates a pythonic version of nested lists/dictionaries """
	o = ""
	if isinstance(s, ModuleType):
		o = o + "<module " + s.__name__+ ">"
	elif type(s) == np.ndarray:
		o = o + "np.array(" + str(s.tolist()) + ")"
	elif type(s) == dict:
		o = o + "dict:\n"
		for key in s:
			out = prettyfy(s[key])
			o = o + "\t" + key + ":\n"
			o = o + "\n".join(["\t\t" + i for i in out.split("\n")])
			o = o + "\n"
	elif type(s) == list:
		o = o + "list:"
		for item in s:
			out = prettyfy(item)
			o = o + "\n".join(["\n\t" + i for i in out.split("\n")])
	else:
		try:
			s.__class__
			s.__dict__
			if str(type(s)) == "<type 'instance'>":
				o = o + "<class '"+str(s.__class__) + "'>:\n"
			else:
				o = o + ""+str(s.__class__) + ":\n"
			for key in s.__dict__:
				out = prettyfy(s.__dict__[key])
				o = o + "\t" + key + ":\n"
				o = o + "\n".join(["\t\t" + i for i in out.split("\n")])
				o = o + "\n"
		except:
			o = str(s)
	return "\n".join([line for line in o.split("\n") if line.strip() != ""])

is_tab = '\t'.__eq__

def parse_to_list(s, depth = 0):
	o = []
	head = ""
	layer = ""
	for line in s.split("\n"):
		tabs = re.match('\t*', line).group(0).count('\t')
		if tabs <= depth:
			if not layer == "":
				o.append([head, parse_to_list(layer,depth+1)])
				head = ""
				layer = ""
			if head != "":
				o.append(head)
			head = line.lstrip()
				#o.append(line.lstrip())
		elif tabs > depth:
			layer = layer + "\n" + line
	if not layer == "":
		o.append([head, parse_to_list(layer,depth+1)])
		head = ""
		#print tabs, line
	else:
		o.extend([head])
	if len(o) == 1:
		return o[0]
	return o

def escape( text, characters = "\"" ):
    for character in characters:
        text = text.replace( character, '\\' + character )
    return text

def parse(s):
	l = parse_to_list(s)
	#print l
	return parse_list(l)

def parse_list(ll):
	if type(ll) == str:
		try:
			return int(ll)
		except:
			try:
				return float(ll)
			except:
				return ll
	elif type(ll) == list:
		if len(ll) == 1:
			return parse_list(ll[0])
		elif len(ll) == 2 and type(ll[0]) == str:
			m = re.search('<class \'(.*)\'>', ll[0])
			if m != None:
				d = {}
				for lll in ll[1]:
					if type(lll[0]) == str and len(lll) > 1:
						if lll[0][-1] == ":": 
							d[lll[0][:-1]] = parse_list(lll[1])
						else:
							d[lll[0]] = parse_list(lll[1])
				if not m.group(1) in ignored_types:
					return m.group(1) + "(" + str(d) +")"
				return ""
			elif ll[0] == "list:":
				return parse_list(ll[1])
			elif ll[0] == "dict:":
				d = {}
				for lll in ll[1]:
					if type(lll[0]) == str:
						if lll[0][-1] == ":": 
							d[lll[0][:-1]] = parse_list(lll[1])
						else:
							d[lll[0]] = parse_list(lll[1])
				return d
			else:
				return [parse_list(lll) for lll in ll]#testing this
				if type(ll[1]) == str:
					try:
						return str(ll[0]) + ": " + float(ll[1])
					except:
						return str(ll[0]) + ": " + str(ll[1])
				else:
					return str(ll[0]) + ": " + str([parse_list(lll) for lll in ll[1]])
		else:
			return [parse_list(lll) for lll in ll]
	else:
		raise Exception("Unexpected Input!")


class Picklable(object):
	def __init__(self, p):
		if type(p) == str or type(p) == dict:
			self.loads(p)
	def dumps(self):
		return dumps(self.__dict__)
	def loads(self,s):
		if s == None:
			return
		if type(s) == dict:
			self.__dict__.update(s)
		elif type(s) == str:
			s = s.replace("np.ndarray","np.array")
			try:
				d = eval(s)
				if type(d) == dict:
					self.__dict__.update(d)
				else:
					return
			except:
				return s
		else:
			return
		for k in self.__dict__:
			if type(self.__dict__[k]) == str:
				if self.__dict__[k] == "list:":
					self.__dict__[k] = []
				else:
					try:
						self.__dict__[k] = eval(self.__dict__[k])
					except:
						pass
			elif type(self.__dict__[k]) == list:
				for key in range(len(self.__dict__[k])):
					if type(self.__dict__[k][key]) == str:
						try:
							self.__dict__[k][key] = eval(self.__dict__[k][key])
						except:
							pass
			elif type(self.__dict__[k]) == dict:
				for key in self.__dict__[k].keys():
					if type(self.__dict__[k][key]) == str:
						try:
							self.__dict__[k][key] = eval(self.__dict__[k][key])
						except:
							pass
		return self
	def save(self,file):
		with open(file,"w") as f:
			f.write(prettyfy(self))
	def load(self,file):
		with open(file,"r") as f:
			self.loads(parse(f.read()))
	def eval_dict(self):
		for k in self.__dict__:
			if type(self.__dict__[k]) == str:
				if self.__dict__[k] == "list:":
					self.__dict__[k] = []
				else:
					try:
						self.__dict__[k] = eval(self.__dict__[k])
					except:
						pass


def load(file):
	"""loads a pickled object"""
	with open(file,"r") as f:
		l = parse(f.read())
		if type(l) == str:
			return loads(l)
		else:
			if len(l) == 1:
				return loads(l[0])
			elif len(l) == 2 and l[1] == '':
				return loads(l[0])
			else:
				return [loads(ll) for ll in l]

def load_list(files):
	return [load(f) for f in files]
	
def load_glob(self,filename_template):
	return self.load_list(glob.glob(filename_template))
