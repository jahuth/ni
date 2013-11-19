
tmp_path = "/net/store/ni/happycortex/statmodelling/"


import json
config = {'tmp_path': "/net/store/ni/happycortex/statmodelling/"}

class Config:
	def __init__(self, filename='ni_config.json',fallback=None):
		self.config = {'tmp_path': "/net/store/ni/happycortex/statmodelling/", "ni.data.monkey.path":"/net/store/ni/happycortex/statmodelling/data/"}
		self.filename = filename
		self.fallback = fallback
	def load(self):
		try:
			with open(self.filename, 'r') as f:
				self.config = json.load(f)
		except:
			self.config = {}
		if self.fallback != None:
			self.fallback.load()
	def save(self):
		with open(self.filename, 'w') as f:
			json.dump(self.config, f)
		with open(self.filename+".txt", 'w') as f:
			for k in self.config.keys():
				f.write(str(k) + " " + str(self.config[k]))
	def set(self,name,value=True):
		self.config[name] = value
		self.save()
	def unset(self,name):
		del self.config[name]
		self.save()
	def get(self,name, default_value=False):
		if name in self.config:
			return self.config[name]
		if self.fallback != None:
			return self.fallback.get(name,default_value)
		return default_value
	def keys(self, recursive = False):
		if recursive and self.fallback != None:
			d = []
			d.extend(self.fallback.keys(recursive = True))
			d.extend(self.config.keys())
			return d
		return self.config.keys()

default = Config('ni_default_config.json')
system = Config('ni_system_config.json',default)
user = Config('ni_user_config.json',system)

def load():
	return user.load()
def save():
	return user.save()
def get(name, default_value=False):
	return user.get(name, default_value)
def set(name, value=True):
	return user.set(name, value)
def keys(recursive = False):
	return user.keys(recursive)
try:
	user.load()
except:
	pass
