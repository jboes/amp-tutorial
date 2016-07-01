#!/usr/bin/env python
from amp import Amp
from amp.descriptor import Gaussian
from amp.regression import NeuralNetwork
from ase.db import connect
from amp import SimulatedAnnealing

db = connect('../../../database/master.db')

images = []
for d in db.select('train_set=True'):
    atoms = db.get_atoms(d.id)
    del atoms.constraints
    images += [atoms]

calc = Amp(label='./',
           dblabel='../../',
           descriptor=Gaussian(cutoff=6.5),
           regression=NeuralNetwork(hiddenlayers=(2, 3)))

calc.train(images=images,
	   data_format='db',
	   cores=48,
	   energy_goal=1e-2,
	   force_goal=1e-1,
	   global_search=SimulatedAnnealing(temperature=70,
					    steps=50),
	   extend_variables=False)
