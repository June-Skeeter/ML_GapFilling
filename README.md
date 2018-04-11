This module is designed to use Long-short Term Memory (LSTM) Nerural Netowrks - for feature identification and gapfilling of flux time series

The general workflow is as follows:

1) Bayseian optimisation of a "full model" with all potential factors using gaussian process regression to identify the intial "optimal" number of nodes and timesteps

2) Itterative construction of models for feature identification from a pool of p potential features (f) using the N & T values identified in step 1)

	A) Start with 1 factor model, and loop through factores f1 ... fp, select the factor $f_{min}$ that yield the model with the losest MSE
	B) Increase the model size to 2 and train iteratively on feature $f_{min}$ and factors f1 ... fp-1, selecte factor $f_{min2}$
	C) Repeate B until MSE stops decreasing, select the smallest model where MSE is lower than previous models by some score ... 95% CI??

3) Repeate step 1 to optimize final model.  Probably benificial??
	A) Iteratively deconstruct final model??

4) Use optimized model to fill time series.
