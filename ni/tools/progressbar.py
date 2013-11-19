"""
.. module:: ni.tools.progressbar
   :platform: Unix
   :synopsis: Showing a progressbar

.. moduleauthor:: Jacob Huth <jahuth@uos.de>

"""
import sys 
import math

toolbar_width = 40
last_p = 0.0
def progress_init():
	"""
	Undocumented
	"""
	# setup toolbar
	sys.stdout.write(" [%s]" % (" " * toolbar_width))
	sys.stdout.flush()
	sys.stdout.write("\b" * (toolbar_width+3)) # return to start of line, after '['

def progress(p):
	"""
	Undocumented
	"""
	sys.stdout.write(" [%s%s] " % ("-" * int(math.ceil(toolbar_width * p))," " * int(math.floor(toolbar_width * (1-p)))) + str(p))
	sys.stdout.flush()
	sys.stdout.write("\b" * (toolbar_width+4)) # return to start of line, after '['

def progress(a,b):
	"""
	Undocumented
	"""
	p = float(a)/float(b)
	line = " [%s%s] " % ("-" * int(math.ceil(toolbar_width * p))," " * int(math.floor(toolbar_width * (1-p)))) + str(a) + "/" + str(b)
	sys.stdout.write(line)
	sys.stdout.flush()
	sys.stdout.write("\b" * len(line)) # return to start of line, after '['

def progress_end():
	"""
	Undocumented
	"""
	sys.stdout.write("\n")
