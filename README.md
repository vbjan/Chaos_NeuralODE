# LED_NeuralODE
Bachelor Thesis at CSE-lab at ETH on: Learning effective dynamics of a system using Neural ODEs

Applying Neural ODEs to three scenarios of learning the dynamics of a chaotic system with:
  - full state observations
  - partial state observations
  - full state observations with prior knowledge of the dynamics
  
As examples the Lorenz system and Van der Pol oscillator are used as relatively simple chaotic dynamical systems.

  - "FullStateObsNODE.py": training and testing script of full observation model with and without prior knowledge
  - "PartialStateObsNODE.py": training and testing of partial observation model
  - "Datasets.py" - torch data sets of Lorenz system and Van der Pol system

Some examples of the learnt dynamics (with prior knowledge and full state observations):

![alt text](https://github.com/vbjan/LED_NeuralODE/blob/master/results_figures/SumKnowledge/0.6attractor.png)
![alt text](https://github.com/vbjan/LED_NeuralODE/blob/master/results_figures/SumKnowledge/0.6learntx.png)

