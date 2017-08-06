# fast-tf

A workflow template for fast model architecture experimentation with TensorFlow

## Layout

`setup.sh`: Shell script that setups distributed execution environment for run.sh
`run.sh`: Shell script that executes the `model_runner.py`
`model_runner.py`: Just a wrapper around the models.py script, allowing you to pass in arguments from the command line
`models.py`: Holds the models you want to experiment with


