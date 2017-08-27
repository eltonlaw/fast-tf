# fast-tf

A workflow template for fast model architecture experimentation with TensorFlow

## Layout

`setup.sh`: Shell script that setups distributed execution environment for `run.sh`

`run.sh`: Shell script that executes the `model_runner.py`

`model_runner.py`: Imports TensorFlow MetaGraph from `models.py`. A wrapper script containing boilerplate and argument parsing.

`models.py`: Holds the models you want to experiment with. Exports TensorFlow MetaGraph.


