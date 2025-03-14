****Disaster Response Pipelines****


Dependencies:
Python 3.8+
Pandas, NumPy, Matplotlib, SQLAlchemy
Scikit-learn, NLTK
Flask


**Disaster Response Classification Project**

Description

This project is part of the Data Science Nanodegree Program by Udacity, in collaboration with Figure Eight. The dataset includes pre-labeled tweets and messages from real-life disaster events. The goal is to develop a Natural Language Processing (NLP) model to categorize messages in real-time.

Key Components
The project consists of three main sections:

Data Processing: An ETL pipeline to extract, clean, and save data in a SQLite database.
Machine Learning Pipeline: A model to classify text messages into multiple categories.
Web Application: A web app to display real-time message classification results.

How to Run the Program

1. Set Up Database and Train the Model
  Run the following commands in the project directory:
  
  Run the ETL Pipeline
  
  To clean and store processed data in the database:
  
  python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db
  
  Run the ML Pipeline
  
  To load data from the database, train the classifier, and save the model:
  python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl

2. Launch the Web App
  Run the following command in the app directory to start the web application:
  
  python run.py
  
  3. Access the Web App
  Visit the app in your browser at:
  http://0.0.0.0:3000/
  
  Additional Resources
  
  In the data and models folders, you’ll find two Jupyter notebooks to guide you through the project:
  
  ETL Preparation Notebook: Explains the ETL pipeline step by step.
  ML Pipeline Preparation Notebook: Covers the Machine Learning pipeline, including a grid search section for fine-tuning the model.
  
  Important Files
  
  app/templates/: HTML templates for the web application.
  
  data/process_data.py: Automates data cleaning, feature extraction, and storage in an SQLite database.
  
  models/train_classifier.py: Builds, trains, and exports the classification model as a .pkl file.
  
  run.py: Launches the Flask web application for real-time classification.

**ETL Pipeline Details**

Objective
Develop an ETL (Extract, Transform, Load) pipeline to process disaster response data, preparing it for analysis and machine learning tasks.

Steps
  Import Libraries and Load Datasets
  Load messages and categories datasets into Pandas DataFrames.
  Data Cleaning and Transformation
  Merge datasets on the id column.
  Split the categories column into individual binary category columns.
  Remove duplicates and handle missing values.
  Save Data into SQLite Database
  Store the cleaned dataset in a SQLite database table (DisasterMessages).
  Automation
  Write a script (process_data.py) to automate the ETL process.

**ML Pipeline Details**

Objective
Build and optimize a Machine Learning pipeline to classify disaster response messages into multiple categories.

Steps
  Load Data
  Load cleaned data from the SQLite database.
  Build the ML Pipeline
  Create a pipeline that includes data preprocessing and classification.
  Use MultiOutputClassifier (AdaBoostClassifier) for multi-category predictions.
  Train and Evaluate the Model
  Split the data into training and testing sets.
  Evaluate model performance using metrics like accuracy, precision, recall, and F1-score.
  Model Optimization
  Use grid search for hyperparameter tuning.
  Experiment with additional features and algorithms for improved results.
  Export the Model
  Save the trained model as a .pkl file using the pickle library.
  Automation
  Write a script (train_classifier.py) to automate the ML pipeline.

**Final Output**

  A cleaned dataset stored in disaster_response_db.db (SQLite database).
  A trained classification model saved as classifier.pkl.
  A web application for real-time message classification.

**Screen Snaps:**

![Screenshot 2025-01-13 at 20 30 49](https://github.com/user-attachments/assets/5467aae4-19a7-49e8-b808-af721e1bdd5b)

![Screenshot 2025-01-13 at 20 31 17](https://github.com/user-attachments/assets/481a7f30-91fc-4394-a3bf-604034845caa)

![Screenshot 2025-01-13 at 20 31 28](https://github.com/user-attachments/assets/40ed3eb9-2441-47e7-8916-26d3f1dff61d)



**Licensing, Authors, and Acknowledgements**

This application was developed as part of the Udacity Data Scientist Nanodegree program.

Licensing: 
The code and project materials are intended for educational purposes and follow Udacity's licensing guidelines.
Authors: The project was implemented by Amit Kumar, building on templates provided by Udacity.
Acknowledgements:
Udacity for providing the project structure and code templates.




