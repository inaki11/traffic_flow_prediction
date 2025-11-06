# Sacyr Traffic Prediction Module

This repository contains the source code for Sacyr's project on **Traffic Prediction and Auxiliary Lane Opening**. The codebase is organized into three main modules, each serving a distinct purpose in the project lifecycle.

---

## 1. Introduction

The code in this repository supports a predictive model designed to forecast traffic conditions and determine optimal timings for opening auxiliary lanes. The project is structured into three core components: `Experiments Code`, `Service`, and `Development`.

---

## 2. Experiments Code

The `sacyr` directory houses the code template used to conduct all the **Deep Learning (DL)** experiments the project. This name will be the same of your project in wandb.

Before to run an experiment:
- set wandb login key {"key":"<your_key>"} as ~/wandb/login.json 


How to run an experiment:
- 1st introduce dataset into ~/data/datasets/sacyr
- 2nd decide Deep Learning Architecture and number of experiments to optimice the Bayesian Search
- 3rd run run_sweep.py script. More examples in module README
```python
python run_sweep.py --config configs/cnn_lstm.yaml --total-runs 100 --processes 1
```

---

## 3. Service 

The `Service` directory contains the production-ready, **Dockerized API service**.

This component is responsible for the operational deployment of the final model. It is designed to run continuously, performing the following key functions:
* Actively monitoring and reading real-time traffic updates from the database (BBDD).
* Processing the new data through the trained models.
* Inserting the resulting traffic predictions back into the database.
The Service module contains the following elements:
models

### Files and Folders:
**/models**
contains the best models provided by experimentation for predicting traffic flow on each of the loops from A3, with two directories, 1c and 2c, each representing a direction of traffic. Within each directory, there are five models that are assembled for each prediction.

**crontab.txt**
This config file set a periodical process that looks in the database for traffic updates, make predictions 24 hours forward and store them on the database, for each loop and traffic direction

**Dockerfile**
This file defines the container, copy the files and install dependencies to execute the internal code.

**main.py**
Main code that request data, make predictions and inset them back to the database

**pruebas.ipynb** 
just a notebook where I did try some things. Can be deleted but good to have just in case.

**start.sh** 
script that the container executes at start time.


### How to run:

From terminal set inside the service directory and run: 

```python
docker compose up --build
```

It may be posible to be needed to work on ports and network so the database can be reached from the container.


---

## 4. Development 

The `Development` directory contains all exploratory code, preliminary tests, and utility scripts created throughout the project's lifecycle.

This module serves as an archive of the analytical process and includes code for:
* Data exploration
* Data cleaning and preprocessing
* Data visualization
* Dataset creation and feature engineering

### main files

**análisis_experimentos.ipynb**
Represent in tables the different blocks of experiments carried out during the project to find the DL architecture, the amount of context, and the prediction horizon.

**create_dataset.ipynb**
Code that, based on already filtered and selected data, creates datasets to experiment with different features, contexts, and prediction horizons.

**EDA.ipynb**
Some visualizations we made early on the project

**filter_raw_data.ipynb**
Plot data and we filter the data so we could select valid loops.

