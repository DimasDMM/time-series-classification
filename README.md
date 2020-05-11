# Introduction

Time Series Classification (TSC) has been considered as one of the hardest problems in data mining. With the exponential increase of data availability (in particular, time series data) a wide variety of TSC algorithms have been proposed, although only some Deep Neural Network can solve this task. This growth also means that we depends more on automatic classification of time series data, and ideally, we should be able to apply algorithms with the ability to work in large datasets.

Although in the past, common approaches using RNNs to solve TSC did not succeed, nowdays this complex task is feasible. We show an overview of some Neural Networks which are successful in this problem. In particular, we show two approaches to make a Binary Time Series Classification by using LSTM with Dropout and LSTM with Attention.

# Content of repository

There are several notebooks where first I analyze the data and later I train some ML models.
- [Dataset Analysis](./notebooks/Data%20Analysis%20-%20Dataset%20Ford%20-%20Overview.ipynb)
- [Training with LSTM model](./notebooks/ModelTraining%20-%20Dataset%20Ford%20-%20LSTM.ipynb)
- [Training with Attention model](./notebooks/ModelTraining%20-%20Dataset%20Ford%20-%20Attention.ipynb)

Additionally, there is a report in PDF format which contains all the details about the process: [Report](./report.pdf)

# Note

I made this project as part of my Data Science Master's programme at Higher School of Economics. The *Boreholes Dataset* (described in the report) contains confidential data, thus is not included in the repository as well as the scripts and Docker containers used to preprocess it.

However, it is included a copy of the *Ford Dataset*, thus you should be able to train the different models and obtain the same results as me. Source of dataset: http://www.timeseriesclassification.com/description.php?Dataset=FordA
