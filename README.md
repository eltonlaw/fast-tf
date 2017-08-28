# tf-template

A workflow template for fast model architecture experimentation with TensorFlow.

## Layout

`setup.sh`: Shell script that setups distributed execution environment for `run.sh`

`run.sh`: Shell script that executes the `model_runner.py`

`model_runner.py`: Imports TensorFlow MetaGraph from `models.py`. A wrapper script containing boilerplate and argument parsing.

`models.py`: Holds the models you want to experiment with. Exports TensorFlow MetaGraph.

## Using this Template

1) Setup:

```
./setup.sh
```

2) In the `models` folder create a new folder, the name of your experiment model. Needs an `__init__.py` file which exports to `__all__` a function called `model(...)`that returns a MetaGraphDef and only takes ones parameter `FLAGS` (the standard name for the object that holds command line arguments). Look at the structure of `test_experiment` if something is unclear. 

3) In `run.sh` change the variable name, `MODEL` to the name of your experimental model and feed in your hyperparameters.

## Notes

* Opted not to use TensorFlow flags interface, used Python's `argparse` module instead
* Opted not to use TensorFlow logging interfance, used Python's `logging` module instead (when TensorBoard logging isn't used)
