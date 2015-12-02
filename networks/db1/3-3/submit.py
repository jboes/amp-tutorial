#!/usr/bin/env python
from amp import Amp
from amp.descriptor import *
from amp.regression import *

calc = Amp(label="./",
           descriptor=Behler(cutoff=6.0),
           regression=NeuralNetwork(hiddenlayers=(2, 3)))

calc.train("../train.db", # The training data
           cores=1,
           global_search=None, # not found the simulated annealing feature useful
           extend_variables=False) # feature does not work properly and will crash
