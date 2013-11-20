"""
Helper Functions to alert when something is done
"""

from subprocess import call

def alert(msg):
	"""
		alerts the user of something happening via `notify-send`. If it is not installed, the alert will be printed to the console.
	"""
	if call(["which","notify-send"]) == 0:
		call(["notify-send",msg])
	else:
		print "ALERT: ", msg

