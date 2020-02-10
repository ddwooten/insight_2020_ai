Creator: Daniel Wooten
Licence: This repository is covered by the Eclipse public licence, please see
[spdx.org/licences/EPL-1.0.html] for the licence text.

# What Next?
## A Reinforcement Learning LSTM based recommendation engine
### Keywords: Pytorch, reinforcement learning, RL, RecSys, recommender systems

## Introduction
What is "What Next?" (WN)? WN is a framework for developing reinforcement
learning based recommendation systems. WN provides a deployment ready and
scaleable infrastructure upon which to implement any given RL algorithm. While
existing as a fairly low level API WN provides an excellent template from which
to expand out an existing RL model, tune hyperparameters, or experiement with
loss functions. WN was built with modularity and itteration in mind - these
choices being reflected in the straightforward architecture.

## Installation
The necessary Python 3 packages for WN to run include and are limited to

* PyTorch
* Pandas
* Numpy
* re
* math
* datetime

Personally I find it easiest to install the above pckages with conda.

Following the acquisition of the above packages clone this repo, adjust
the bash line in *main.py* and *test.py*, make both executable files on your
system, and follow the instructions in section **Setup** below.

## Setup
Once you've cloned into this repo you will find three sub-directories, *data*,
*src* and *models*. *data* is where all training data is kept. Such data is
stored in pickles and the curent hard coded name for the pickle containing
the training data is *clean.pk* - this may be altered in *main.py*. Inside of
*models* is where saved PyTorch model instances are kept following their
generation during training runs. This is also where models are loaded from
for model inference.

Inside of *src* are all the files that make WN tick. The first to understand
is *setup.txt* - this is a raw text file which holds the simulation
configuation. The contents of *setup.txt* - which should be edited by you - are
given below in a *key, value* format. 
*setup.txt* is not sensative to capitalization, linebreaks, or
whitespace - it operates as a ***comma*** seperated file. 

*train, [yes/no]
*model, [option] (currently this is only 'lstm')
*data, [path to training pickle]
*epochs, [integer number of training samples to run]
*model_agent_path, [path to agent model]
*model_critic_path, [path to critic model]

Once all these options have been set calling *main.py* will run the entire WN
model according to the parameters given in *setup.txt*.

## The internals

Before discussing the main program, a note on the file *test.py*. *test.py*
runs any tests enumerated by the *tests* dictionary found inside of 
*run_tests()*. At this time there is only one test which tests the reading of
*setup.txt* and the processing of its input dictionary. To run the tests simple
execute the *test.py* file - it will handle the rest and output, in the
executing dirctory, the test results in a file tittled *Test.test*.

Moving on to the main program, executed by calling *main.py*, there are two
modes of operation - training and inference. These two modese are activated, 
respectively, by the "yes" and "no" flag given to the "train" key in
*setup.txt*. In *main.py* the simulation framework is built whith the key
components, state and agent, being passed to the environment class. 

The environment class is contained in *environment.py* and manages the
interactions between the state and agent classes. The state class is contained
in *state.py* and the agent class in *agent.py*. The state class is largely
responsible for data handling, format, passing, and retreival. The agent class
is entirely responsible for model training and inference. The model
***classes*** - AgentModel and CriticModel - are both found inside *model.py*
and contain the build instructions as well as the forward pass instructions
for all the PyTorch models used in this framework.

In the case of model training, *main.py* will call *run_sim()* from the
environment class. Inside of said class, for training, *run_sim()* will call
*train_agent()*, a function of the environment class. *train_agent()* handles
two tasks. After calling the agent's setup function, *wake_agent()*,
*train_agent()* begins the agent training cycle. In this cycle
first the agent is asked to make an inference of the embeddings through the
*predict()* method of the agent class. These embeddings are passed to the
state class *predict* method which produces a randomly selected
"entity-to-be-recommended" - in this case a song - from a list of entities with
similar embeddings. This recommendation as selected by state is passed to agent
through the *propogation()* method which, in short, caculates and backpropogates
the gradients for the neural net to update itself with. Throughout this process
at pre-defined intervals *train_agent()* carries out its other task - saving
the PyTorch model to disk for later use.

Peering into the details of the agent class the first method which must be
called for training after instantiation is *wake_agent()* which then calls 
*add_model()* which executes a series of steps - calling the model constructors
for both AgentModel and CriticModel from the model class as well as assigning
loss and optimization functions for both the agent and critic models. Whenever
*predict()* is called the given user_history is passed off to *factorize()* 
which breaks down the user_history into numpy arrays which the models can then
use. The arrays which are produced are factors for both agent and critic. 
Following this factorization an inference is asked of the agent model. In the
case of *propogate()* the actual song selected by the state classes's 
*produce()* method is appended to the critic factors via the *add_prediction()*
method. These critic factors are then passed to a forward inference of the
critic model from which the reward is generated and used to arrive at the loss
for the agent. Once this loss has been backpropogated through the gradients of
the agent model the critic loss is returned from the *get_critic_loss()*
method and backpropogated through the critic's model. 

Currently the critic loss is assessed as the difference between the sample
standard deviation of the intra-song embedding distances with and without the
recommendation as compared to the standard deivation of the population, in
this case all songs that a user has listened to. 

In the case of inference, where training is set to "no", the environment class
calls *ready_agent()* instead of *wake_agent()* and only an inference cycle of
the agent is executed, no critic step is applied. *ready_agent()* itself loads
actor and critic models from the specified path given in *setup.txt*.

With regards to the internals of the state class, it has two primary methods
both called upon by the environment class - *get_random_user_history()* which
returns a random complete user history from all user historys while 
*produce()* selects a random song who's embedding's match the input 
*attributes*. If *produce()* is given a *relax* parameter a random
embedding is extended to accept values within the range of +/- relax/2.

And with all of that said, you should now be able to get WN up and running!

