# ETeval
Eye-movement event matching procedure and computation of the event level Cohen's Kappa as described in: 
```sh
@article{zemblys2018gazeNet,
  title={gazeNet: End-to-end eye-movement event detection with deep neural networks},
  author={Zemblys, Raimondas and Niehorster, Diederick C and Holmqvist, Kenneth},
  journal={Behavior research methods},
  year={2018},
}
```

This code is a generalised version of the eye-movement event matching procedure described in [Hooge et al. 2017](https://link.springer.com/article/10.3758/s13428-017-0955-x). Instead of looking for overlapping test-events occurring earliest in time, for each event in the reference stream, it looks for the event in the test stream that has the largest overlap, and then matches the events in the two streams.

## Using ETeval
To use ETeval place your datasets with events detected in the `root` folder, create a `job.json` file that contains a list of json objects that define the ground truth and test datasets, and run:

```sh
python run_eval.py --config job.json
```
For an example see `example_job.json`. All jobs files need to be a list of the following json objects:
```sh
{
    "root": "etdata", 			# Root folder of the data
    "gt": "lund2013_npy", 		# Ground truth dataset
    "pr": "lund2013_npy_gazeNet", 	# Test dataset
    "dataset": "lund2013-image-test",	# Dataset label
    "alg": "gazeNet"			# Algorithm label
}
```
For each job, ETeval will output scores for fixations, saccades and PSOs and generate a csv file with the results in your test data folder. 

**!!! Note that only the evaluation of fixations, saccades and PSO is currently supported and data with `status` flag `False` (as defined in the ground truth data) is not evaluated is the current version !!!**

### Data format
The internal data format used by ETeval is a structured numpy array with a following format:

```
dtype = np.dtype([
	('t', np.float64),	#time in seconds
	('x', np.float32),	#horizontal gaze direction in degrees
	('y', np.float32), 	#vertical gaze direction in degrees
	('status', np.bool),	#status flag. False means trackloss 
	('evt', np.uint8)	#event label:
					#0: Undefined
					#1: Fixation
					#2: Saccade
					#3: Post-saccadic oscillation
					#4: Smooth pursuit
					#5: Blink
])
```
TODO:
- Describe how to replicate one of the paper results


### Python environment
ETeval was developed and tested using Python 2.7, however it should also work if using Python 3.x. Code requires `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn` and `tqdm` libraries.
