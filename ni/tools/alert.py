"""
Helper Functions to alert when something is done
"""

from subprocess import call

def alert(msg):
	if call(["which","notify-seend"]) == 0:
		call(["notify-send",msg])
	print "ALERT: ", msg

