### 1. Project Overview

This repository contains the project **Disaster Response Pipeline** which is a part of **Udacity's Data Scientist Nanodegree** and deals with several aspects of the ETL pipeline, ML pipelines, and model deployment on a web-app. The aim of this project is to assist in forwarding the relevant disaster messages to the concerned departments/organizations for necessary action/response. 


### 2. Technical Summary

The project applies machine learning to classify real disaster response messages in 36 categories that represent different departments. The training data is obtained from the company Figure 8 that is first cleaned and transformed through several steps. Then a multi-output classifier (with random forest classification) is implemented and grid-search is used for hyperparameter tuning. Afterwards, the model is deployed on web using a flask app with several visualizations where any new message can be tested to check the relevant categories or the departments that need to be engaged for dealing with the particular disaster. The model evaluation reveals the precision, F1-score, and recall above 0.92.


### 3. Requirements

The following modules/packages have been used in this project:

1. **Pandas** and **Numpy** for data manipulation.
2. **SQL Alchemy** for creating databases and tables.
3. **NLTK** for tokenizing and creating features out of text
4. **Scikit-learn** for data modelling and evaluation
5. **Plotly** for data visualization


### 4. Files

The repository consists of the following files:

1. A CSV file consisting of all the disaster messages **messages.csv** in the *data* folder.
2. A CSV file consisting of all the labeled categories for each corresponding message **categories.csv** in the *data* folder.
3. The python file **process_data.py** in the *data* folder. This file loads the above provided CSV files and performs some data cleaning and transformation on them. It stores the results in the database file **DisasterResponse.db** in the *data* folder as well.
4. The python file **train_classifier** in the *models* folder. This file trains and perfroms multiclass classification. It generates a *classification.pkl* file but it isn't a part of this repository.
5. The python file **run.py** in the *app* folder. This file loads the cleaned data from the database and the trained model and creates a flask web-app link where you can visualize the dataset and also classify messages based on the trained model.
6. Two html files **master.html** and **go.html** in the *app/templates* folder that provide formatting for the web app.


### 5. Instruction on how to run the files

1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        **python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db**
    - To run ML pipeline that trains classifier and saves
        **python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl**
2. Run the following command in the app's directory to run your web app.
    `python run.py`
3. Go to http://0.0.0.0:3001/


### 6. Links and Acknowledgements

- **[Udacity](https://www.udacity.com)**: This project is part of the Data Scientist Nano Degree that I am doing in collaboration with Udacity.
- **[Figure Eight](http://figure-eight.com)**: The dataset containing the real-time disaster response messages and department categories is provided by this company.

