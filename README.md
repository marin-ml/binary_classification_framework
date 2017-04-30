# README #

This README would normally document whatever steps are necessary to get your application up and running.

### Requirements ###

* Ubuntu 14.04
* Python 2.7
* tensorflow 0.10
* numpy


### How do I get set up? ###

* training_relu.py

    This is the script to train data from csv file for machine learning.

* predict.py

    This is the script to predict the data. The format is such as below:

        python predict.py csv_file
        python predict.py feature1 feature2 feature3 feature4 feature5 feature6

    First case is predict the csv file and save the result to csv file, too.

    Second case is predict the value for input by argument in command window and display the result in command window, too.

    for example:

        python predict.py validate.csv
        python predict.py 0.21 -0.3 0.5 1.25 0.2 2.0
