import designmatrix
import ip
import pointprocess
import net_sim
import ip_generator

class RateModel(ip.Model):
	"""This is a shorthand class for an Inhomogenous Pointprocess model that contains only a RateComponent and nothing else.

	This is completely equivalent to using:

		ni.model.ip.Model({'name':'Rate Model','autohistory':False, 'crosshistory':False, 'knots_rate':knots_rate})
		
	"""
	def __init__(self,knots_rate=10):
		super(RateModel,self).__init__({'name':'Rate Model','autohistory':False, 'crosshistory':False, 'knots_rate':knots_rate})

class RateAndHistoryModel(ip.Model):
	"""This is a shorthand class for an Inhomogenous Pointprocess model that contains only a RateComponent, a Autohistory Component and nothing else.

	This is completely equivalent to using:

		ni.model.ip.Model({'name':'Rate Model with Autohistory Component','autohistory':True, 'crosshistory':False, 'knots_rate':knots_rate, 'history_length':history_length, 'knot_number':history_knots})
		
	"""
	def __init__(self,knots_rate=10,history_length=100,history_knots=4):
		super(RateAndHistoryModel,self).__init__({'name':'Rate Model with Autohistory Component','autohistory':True, 'crosshistory':False, 'knots_rate':knots_rate, 'history_length':history_length, 'knot_number':history_knots})
