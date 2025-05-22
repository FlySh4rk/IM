# Implantmaster.ai

This document contains an essential highlight of the model developed for the implantmaster project.

## Most relevant files

Several iteration were made to progress from the initial proof of concept that was generating both the
input data and the labels in to 2D up to the final version that actually consumes data gathered from the
field by using the dedicated web app that was developed on purpose.

As a result many of the files contained in this packages are obsolete and are kept only for documentation
purposes but they will not be discussed further. 

In this document will only go through the final and most relevant
playbooks, source files, data files, which are:

 - [test/dataset_tests.py](test/dataset_tests.py) : this file contains several tests used to test the dataset generation but also
   to read the data collected from the field with the dedicated webapp and put it in a format better suited
   for the training
 - [src/dataset_reader.py](src/dataset_reader.py) : contains several utilities for generating the datasets for the different versions
   of the model and to process the production data
 - [prod_data/implantMaster-solution.json](prod_data/implantMaster-solution.json) : this is the production data in JSON format that was collected
   to be used for the training
 - [models/model_v4_train.ipynb](models/model_v4_train.ipynb) : This is the notebook used to train the last version of the model
 - [models/models.py](models/models.py) : Contains all the models developed, included the final one (see: `setup_model_v4`)
 - [data/model_v4_prod_v1](data/model_v4_prod_v1) : Contains the trained model suitable to be loaded and used for inferring

## Howtos

### Howto prepare the environment

In order to execute the code a properly configured python virtual env has to be prepared and used
by using these commands

```shell
pipenv install --python $(which python)
pipenv shell
```

#### Prerequisites

 - python installed
 - pipenv installed

### Howto repeat the training

To repeat the training (eventually with additional data), here's a short how to:

1. Replace `prod_data/implantMaster-solution.json` with the new data in the same JSON format
2. Run the test  `DataSetTestCase.test_write` to preprocess the data into a pickle dataset using
   the following command:
   ```shell
   python -m unittest dataset_tests.DataSetTestCase.test_write
   ```
       
3. Go through the [models/model_v4_train.ipynb](models/model_v4_train.ipynb) playbook to train
   the model and store it in [data/model_v4_prod_v1](data/model_v4_prod_v1).

**HEADSUP** : Running the training will override the existing trained model. Be careful and make a backup
copy before proceeding

### Howto running the trained model in infer mode

To run the trained model in infer mode you just have to load the pre-trained model using

```python
import tensorflow as tf
model = tf.keras.models.load_model("data/model_v4_prod_v1")
```

and then use it. You can check the notebooks in the [validation](validation) folder for some examples
on how to do it.
