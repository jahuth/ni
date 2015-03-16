model Package
=============

This package provides two kinds of models:

    * Generative pointprocess models (see Section :py:mod:`ni.model.pointproccess`)
    * GLM based inhomogeneous pointprocess (ip) models 

The pointprocess models can instantiate a given firing rate function with a spike train.
The ip models use "components" that are combined to fit a spike train using a GLM.


What is a "component"?
-------------------------
For the `ni.model` package, a component is a set of time series (or a way to generate these) in a Generalized Linear Model that is used to fit a set of spike trains. When a model is fitted to a set of spike trains, the components are used to compute the designmatrix. Each component provides 1 or more timeseries that represent the influence of a specific event or aspect that is expected to affect the firing rate at that specific point in time to a certain degree.

Eg. a component can be as simple as a constant value over time, or it can be 0 for most of the time, except at those time points where a specific event occurs (eg. a spike of a different neuron). The design matrix shown here is mostly 0 and contains three components (history, rate and a constant):

.. image:: _static/2_designmatrix_example.png



To model a more or less precise time course of the effect of an event, the component needs to create mutiple time series, each representing a time window relative to the event (usually following the event, if a causal link is assumed). It is convenient to use splines to model these windows, because they will produce a smooth time series when summed. 
These time windows can be arranged linearly (each spanning the same amount of time) or logarithmically (such that there is a higher resolution close to the event than further away). The length and number of the time windows can be adjusted, since their explanatory value depends on the modeled process.


The Rate component
---------------------
Given that each spike train is alligned in the same timeframe regarding a certain stimulus (or sequence of stimuli), a rate component models rate fluctuations that occur in every trial. This is done by creating a number of timeseries, each representing a portion of the trial time: eg. early in the trial, in the middle of the trial, at the end of the trial.

A  :class:`ni.model.designmatrix.RateComponent` will span the whole trial with a given number of knots while the class :class:`ni.model.designmatrix.AdaptiveRateComponent` will use the firing rate to space out the knots to cover (on average) an equal amount of spikes.

The history components
---------------------------
Since the spike data containers contain multiple spike trains, a component can access the spiketimes of each of the spike trains contained. The history components use these spike times to model the effect of a certain spike train on the dependent spike train.

    * **autohistory:** the effect of the spike train on itself. Eg. refractory period or periodic spiking
    * **crosshistory:** models the effect of another spike train on the one modeled
    * 2nd order **autohistory_2d:** for bursting neurons, the autohistory alone will not capture the behaviour of the neuron, as it has to either predict periodic activity for the whole trial or no auto effect at all. The 2nd order history takes into account two spikes, such that the end of a burst can be predicted.


Here you can see the slow time frame of a rate fluctuation (fixed to the time frame of each trial) and the fast time frame of a history component (each relative to a spike):

.. image:: _static/2_designmatrix_sum.png


Configuring a Model
---------------------------

The model class has a default behaviour that assumes you want to have one rate component, one autohistory componenet and a crosshistory component for each channel. This behaviour can be changed by providing a different configuration dictionary to the model.

Eg. to create a model without any of the components::

    model = ni.model.ip.Model({‘name’:’Model’,’autohistory’:False, ‘crosshistory’:False, 'rate':False})

Or, if additionally the 2nd order autohistory should be enabled::

    model = ni.model.ip.Model({‘name’:’Model’,’autohistory_2d’:True })

Important for models is also which spike train channel should be modeled by the others::

    model_0 = ni.mode.ip.Model({'cell':0})
    model_1 = ni.mode.ip.Model({'cell':1})
    model_2 = ni.mode.ip.Model({'cell':2})
    model_3 = ni.mode.ip.Model({'cell':3})

Instead of only true or false, crosshistory can also be a list of channels that should be used::

    model = ni.mode.ip.Model({'cell':1,'crosshistory':[2]}) # cell 1 modeled using activity of cell 2

Also there are a number of configuration options that are passed on to a component to change its behaviour:

    * Rate Component:
        * *knots_rate*: how many knots should be created
        * *adaptive_rate*: Whether the RateComponent or AdaptiveRateComponent should be used. The AdaptiveRateComponent can additionally be tweaked with:
            * adaptive_rate_exponent (default = 2)
            * adaptive_rate_smooth_width (default = 20)

    * History Components (autohistory and crosshistory):
        * *history_length*: total lenght of the time spanned by the history component
        * *history_knots*: number of knots to span

These options are set for the whole model at once::

    model = ni.mode.ip.Model({'knots_rate':10,'history_length':200}) # sets history_length for autohistory and crosshistory, knots_rate for rate component


Fitting a Model
---------------------------

An ip.Model can be fitted on a :class:`ni.data.data.Data` object and will produce a :class:`ni.model.ip.FittedModel`. This will contain the fitted coefficients and the design templates, such that the different components can be inspected individually::

    model = ni.model.ip.Model({'cell':2,'crosshistory':[1,3]})
    fm = model.fit(data.trial(range(data.nr_trials/2)))
    
    fm.plot_prototypes() # will plot each fitted component

    plot(fm.firing_rate_model().sum(1)) # will plot the firing rate components (rate + constant)

    plot(fm.statistics['pvalues']) # plots the pvalues for each coefficient


Creating custom components
---------------------------------

To be precise, what happens when a model is fitted is the following:

    * The model creates a :class:`ni.model.designmatrix.DesignMatrixTemplate` and adds components (which are subclasses of :class:`ni.model.designmatrix.Component`) according to its configuration to it:
        * when the Component classes are created they are provided with most of the information they need (eg. trial length)
    * The DesignMatrixTemplate is combined with the data which creates the actual designmatrix by:
        * For each component the function getSplines is called, providing the component with spikes of other neurons
        * The component returns a 2 dimensional numpy array that has the dimensions of *length of time* **x** *number of time series*
        * if the array does not have the length of the complete time series, it will be repeated until it fits. So a kernel that has the length of exactly one trial will be repeated for each trial (the ni models by default make all trials the same length).
    * The design matrix is then passed to the GLM fitting backend

So, if you want to add a component, this can be either implement a function *getSplines* that returns a 2d numpy array that has the correct dimensions (time bins x N), or you can use or inherit from :class:`ni.model.designmatrix.Component` and set the `self.kernel` attribute which will then be returned::

    my_kernel = ni.model.create_splines.create_splines_linspace(time_length, 6, 0) # creates a 6 knotted set of splines covering time_length
    c = Component(header='My own component',kernel=my_kernel)
    model = ni.model.ip.Model(custom_components = [c])

Applications can be eg. a rate component that is only applied to trials, while a second component is applied to the others. If two kinds of trials are alternated this could be done like this::

    from ni.model.designmatrix import Component
    my_kernel = ni.model.create_splines.create_splines_linspace(time_length, 6, 0) # creates a 6 knotted set of splines covering time_length
    is_trial_type_1 = repeat([0,1]*(number_of_trials/2),trial_length)
    is_trial_type_2 = 1 - is_trial_type_1
    c1 = Component(header='Trial Type 1 Rate',kernel=np.concatenate([my_kernel]*number_of_trials) * is_trial_type_1[:,np.newaxis])
    c2 = Component(header='Trial Type 2 Rate',kernel=np.concatenate([my_kernel]*number_of_trials) * is_trial_type_2[:,np.newaxis])
    model = ni.model.ip.Model({'custom_components': [c1,c2]},trial_length)

Or alternatively you can overwrite a kernel of the rate component::

    kernel = np.concatenate([
                np.concatenate([my_kernel]*number_of_trials) * is_trial_type_1[:,np.newaxis]),
                np.concatenate([my_kernel]*number_of_trials) * is_trial_type_2[:,np.newaxis])
            ])
    model = ni.model.RateModel({’rate’:True, ’custom_kernels’: [{’Name’:’rate’,’Kernel’: kernel}]})


Classes in the ni.model package
-------------------------------------


.. autoclass:: ni.model.BareModel
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: ni.model.RateModel
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: ni.model.RateAndHistoryModel
    :members:
    :undoc-members:
    :show-inheritance:




:mod:`ip` Module
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ni.model.ip
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`designmatrix` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ni.model.designmatrix
    :members:
    :undoc-members:
    :show-inheritance:


:mod:`net_sim` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ni.model.net_sim
    :members:
    :undoc-members:
    :show-inheritance:


:mod:`create_design_matrix_vk` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ni.model.create_design_matrix_vk
    :members:
    :undoc-members:
    :show-inheritance:


:mod:`create_splines` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ni.model.create_splines
    :members:
    :undoc-members:
    :show-inheritance:


:mod:`backend_elasticnet` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ni.model.backend_elasticnet
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`backend_glm` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: ni.model.backend_glm
    :members:
    :undoc-members:
    :show-inheritance:

