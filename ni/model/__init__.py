import designmatrix
import ip
import pointprocess
import net_sim
import ip_generator

class BareModel(ip.Model):
	"""This is a shorthand class for an Inhomogenous Pointprocess model that contains no Components.

	This is completely equivalent to using:

		ni.model.ip.Model({'name':'Bare Model','autohistory':False, 'crosshistory':False, 'rate':False})
		
	"""
	def __init__(self,configuration={}):
		config = {'name':'Model','autohistory':False, 'crosshistory':False, 'rate':False}
		for k in configuration:
			config[k] = configuration[k]
		super(BareModel,self).__init__(config)
		self.loads(config)

class RateModel(ip.Model):
	"""This is a shorthand class for an Inhomogenous Pointprocess model that contains only a RateComponent and nothing else.

	This is completely equivalent to using:

		ni.model.ip.Model({'name':'Rate Model','autohistory':False, 'crosshistory':False, 'knots_rate':knots_rate})
		
	"""
	def __init__(self,knots_rate=10):
		if type(knots_rate) == int:
			super(RateModel,self).__init__({'name':'Rate Model','autohistory':False, 'crosshistory':False, 'knots_rate':knots_rate})
		elif type(knots_rate) == dict:
			conf = {'name':'Rate Model','autohistory':False, 'crosshistory':False,'knots_rate':10};
			conf.update(knots_rate)
		        super(RateModel,self).__init__(conf)

class RateAndHistoryModel(ip.Model):
	"""This is a shorthand class for an Inhomogenous Pointprocess model that contains only a RateComponent, a Autohistory Component and nothing else.

	This is completely equivalent to using:

		ni.model.ip.Model({'name':'Rate Model with Autohistory Component','autohistory':True, 'crosshistory':False, 'knots_rate':knots_rate, 'history_length':history_length, 'knot_number':history_knots})
		
	"""
	def __init__(self,knots_rate=10,history_length=100,history_knots=4):
		super(RateAndHistoryModel,self).__init__({'name':'Rate Model with Autohistory Component','autohistory':True, 'crosshistory':False, 'knots_rate':knots_rate, 'history_length':history_length, 'knot_number':history_knots})
		self.loads(knots_rate)
