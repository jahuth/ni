#import xml.etree.ElementTree as ET

import numpy as np

class Reporter:
	def __init__(self):
		#self.tree = ET.ElementTree(ET.Element('html'))
		#self.root = self.tree.getroot()
		self.code = ''
		self.paths = [['']]
		self.path_ids = {}
		self.last_path = ""
		self.i = 1
		self.last_path_id = 0
		self.tree = {"__id__":0,"__path__":"","__content__":[],"__nodes__":[]}
		self.last_task = ""
	def add(self,path,obj):
		self.last_task = obj
		path_components = path.split("/")
		node = self.tree
		p = ""
		for c in path_components:
			p = p + "/"+c
			if c in node:
				node = node[c]
			else:
				node["__nodes__"].append(c)
				node[c] = {"__id__":self.last_path_id + 1,"__path__":p,'__content__':[],"__nodes__":[]}
				self.last_path_id = self.last_path_id + 1
				node = node[c]
		node['__content__'].append(obj)
	def statsTree(self,tree):
		date_min = []
		date_max = []
		types = {}
		priorities = {}
		if "__content__" in tree:
			for c in tree["__content__"]:
				if 'type' in c:
					if c['type'] in types:
						types[c['type']] = types[c['type']] + 1
					else:
						types[c['type']] = 1
				if 'priority' in c:
					p = c['priority']
					if p in types:
						priorities[p] = priorities[p] + 1
					else:
						priorities[p] = 1
				if 'date' in c:
					if date_min == []:
						date_min = c['date']
						date_max = c['date']
					if c['date'] < date_min:
						date_min = c['date']
					if c['date'] > date_max:
						date_max = c['date']
		for k in tree["__nodes__"]:
			s = self.statsTree(tree[k])
			for x in s['types'].keys():
					if x in types:
						types[x] = types[x] + s['types'][x]
					else:
						types[x] = s['types'][x]
			for x in s['priorities'].keys():
					if x in priorities:
						priorities[x] = priorities[x] + s['priorities'][x]
					else:
						priorities[x] = s['priorities'][x]
			if date_min == []:
				date_min = s['date_min']
				date_max = s['date_max']
			if s['date_min'] < date_min:
				date_min = s['date_min']
			if s['date_max'] > date_max:
				date_max = s['date_max']
		return {'children': len(tree["__nodes__"]) + len(tree["__content__"]),'types':types,'date_min':date_min,'date_max':date_max,'priorities':priorities}
	def renderTree(self,tree):
		stats = self.statsTree(tree)
		score = 0
		sigma = 0
		for x in stats['priorities'].keys():
			score = score + x * stats['priorities'][x]
			sigma = sigma + stats['priorities'][x]
		score = float(score) / sigma
		s = "<div class=\"tree_header\"><h2>"+str(tree["__path__"]).split("/")[-1]+"</h2><a class=\"extender\" href='javascript:$(\"#"+str(tree["__id__"])+"\").slideToggle()'>+ "+str(tree["__path__"])+" ("+str(stats['children'])+")</a>\n"
		s = s + str(stats['date_min'].strftime("%H:%M:%S")) + "-" + str(stats['date_max'].strftime("%H:%M:%S")) + " <small>("+str(int(np.round((stats['date_max']-stats['date_min']).seconds)))+"s)</small>"
		for x in stats['types'].keys():
			s = s + " <b class=\"counter counter_"+x+"\">"+str(stats['types'][x])+" "+x+"</b> "
		hidden = ""
		if score < 0:
			hidden = " style=\"display:none;\""
		s = s +" <b>Priority: " + str(int(np.round(score))) + "</b></div><div class=\"tree\" id=\""+str(tree["__id__"])+"\""+hidden+">"
		if "__content__" in tree:
			for c in tree["__content__"]:
				s = s + self.process(c)+"\n"
		for k in tree["__nodes__"]:
			#if k != "__content__" and k != "__id__" and k != "__path__":
			s = s + str(self.renderTree(tree[k]))
		return s + "</div>"
	def process(self,obj):
		if 'type' in obj:
			if obj['type'] == "Debug":
				return "<div class=\"content debug priority"+str(obj['priority']).replace("-","n")+"\"><div class=\"type_indicator\">Debug</div>" + str(obj['msg']).replace("\n","<br/>\n") + "<div class=\"date\">"+str(obj['date'].strftime("%H:%M:%S"))+"</div></div>\n"
			if obj['type'] == "Error":
				return "<div class=\"content error priority"+str(obj['priority']).replace("-","n")+"\"><div class=\"type_indicator\">Error</div>" + str(obj['msg']).replace("\n","<br/>\n") + "<div class=\"date\">"+str(obj['date'].strftime("%H:%M:%S"))+"</div></div>\n"
			if obj['type'] == "log":
				return "<div class=\"content log priority"+str(obj['priority']).replace("-","n")+"\"><div class=\"type_indicator\">Log</div>" + str(obj['msg']).replace("\n","<br/>\n") + "<div class=\"date\">"+str(obj['date'].strftime("%H:%M:%S"))+"</div></div>\n"
		return "<div class=\"content\">" + str(obj).replace("\n","<br/>\n") + "</div>\n"
	def getPathIds(self,path_components):
		ids = []
		s = path_components[0]
		if s in self.path_ids:
				ids.append(str(self.path_ids[s]))
		for p in path_components[1:]:
			s = s + "/" + p
			if s in self.path_ids:
				ids.append(str(self.path_ids[s]))
		return " ".join(ids)
	def render(self,path):
		code = """<!doctype html>
		<html>
			<head>
				<script src="http://code.jquery.com/jquery-1.10.1.min.js"></script>
				<style>
					h1 {
						margin-left: 20px;

					}
					div {
						margin: 1px;
						padding: 2px;
					}
					.tree {
						border: 2px solid #eee;
						margin: 5px;
						margin-left: 2px;
						padding: 10px;
						-moz-border-radius: 15px;
						-webkit-border-radius: 15px;
						-khtml-border-radius: 15px;
						border-radius: 15px;
					}
					.content {
						padding: 5px;
						padding-left: 60px;
						position: relative;
						top: 0px;
						min-height: 50px;
						-moz-border-radius: 10px;
						-webkit-border-radius: 10px;
						-khtml-border-radius: 10px;
						border-radius: 10px;
						font-family: sans-serif;
					}
					a.extender {
						font-size: 12pt;
						color: #bbb;
						font-weight: bold;
						text-decoration:none;
					}
					a.extender large {
						fon-size: 48pt;
					}
					.log {
						padding-left: 45px;
						color: #222;
					}
					.error {
						background: #fbb;
						color: #800;
					}
					.debug {
						background: #dfd;
						padding-left: 65px;
						color: #080;
						display: none;
					}
					.counter {
						-moz-border-radius: 5px;
						-webkit-border-radius: 5px;
						-khtml-border-radius: 5px;
						border-radius: 5px;	
						padding: 2px				
					}
					.counter_log {
						background: #ddf;
						color: #008;
					}
					.counter_Error {
						background: #fbb;
						color: #800;
					}
					.counter_Debug {
						background: #dfd;
						color: #080;
					}
					.log_button {
						background: #ddf;
						color: #008;
					}
					.error_button {
						background: #fbb;
						color: #800;
					}
					.debug_button {
						background: #dfd;
						color: #080;
					}
					.type_indicator {
						font-size: 16pt;
						font-weight: 900;
						position: absolute;
						top: 0px;
						left: 0px;
						opacity:0.6;
						font-family: serif;
					}
					.date {
						padding: 5px;
						position: absolute;
						bottom: 0px;
						left: 0px;
					}
					.priorityn1 {
						font-size: 8pt;
					}
					.priorityn1 .type_indicator {
						opacity:0.4;				
					}
					.priority0 {
						font-size: 10pt;
					}
					.priority1 .type_indicator {
						opacity:1;				
					}
					.priority1 {
						font-size: 10pt;
						border-top: 1px solid black;
						border-bottom: 1px solid black;
					}
					#top_bar {
						position: fixed;
						top: 0px;
						left: 20px;
						background: #fff;
						border: 2px solid #ddd;
					}
					#top_bar a {
						font-weight: 900;
						text-decoration: none;
					}
					#top_bar a.struck {
						text-decoration: line-through;
					}
					.button {
						border: 2p solid #333;
						padding: 4px;
						margin: 2px;
					}
					.header {
						background: #444;
						width: 100%;
						height: 60px;
						margin: 0px;
						margin-bottom: 20px;
					}
					body {
						margin: 0px;
						padding: 0px;
					}
				</style>
			</head>
			<body><div class="header"></div><h1>Last Task</h1>
			"""+self.process(self.last_task)+"""
			<h1>Protocol</h1>
		""" + self.renderTree(self.tree) + """
			<div id="top_bar">Hide/Show:
			<a href="javascript:$('#debug_button').toggleClass('struck');$('.debug').toggle()" id="debug_button" class="button struck">Debug Messages</a>
			<a href="javascript:$('.log').toggle();$('#log_button').toggleClass('struck')" id="log_button" class="button">Log Messages</a> 
			<a href="javascript:$('.error').toggle();$('#error_button').toggleClass('struck')" id="error_button" class="button">Error Messages</a></div>
			</body>
		</html>
		"""
		with open(path,"w") as f:
			f.write(code)
