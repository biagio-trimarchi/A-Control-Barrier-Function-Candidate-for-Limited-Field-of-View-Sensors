A-Control-Barrier-Function-Candidate-for-Limited-Field-of-View-Sensors
======================================================================

This repository contains the simulated scenarios used to showcase the 
control approach presented in
[A Control Barrier Function Candidate for Limited Field of View Sensors]().

# Running the Simulations

###### Prerequisites
To run the simulation on your pc, install the requirement Python3 packages via

```Shell
pip3 install -r requirements.text
```

**N.B.** It is suggested to use [Virtual Environment](https://docs.python.org/3/library/venv.html).

###### Simulations
+ `visual_acceleration_control.py` contains the implementation of the control algorithm
	as reported on the article and showcase the performance that can be obtained
	when the agent under control can be modeled as a fully actuated rigid body

+ `visual_quadrotor.py` contains a "modified" version of the algorithm that
	showcase that can be obtained when the agent is a quadrotor actuaed in trust
	and torques

+ `visual_gates.py` is a more advanced example not reported on the article that
	shows a quadrotor racing trough a simple a track with gates

![Alt Text](Misc/quadrotor.gif)
