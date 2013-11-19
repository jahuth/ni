#!/usr/bin/python

import sys
from ni.tools.project import *

print sys.argv[1]

if len(sys.argv) >= 3:
	print "loading ",sys.argv[1]
	p = load(sys.argv[1])
	p.select_session(sys.argv[2])
	job = p.session.jobs[sys.argv[3]]
	p.retry_failed = True
	print "parameters: ", p.parameters
	job.run(sys.argv[4:])
