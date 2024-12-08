# -*- coding: utf-8 -*-
"""PasswordPredict.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vMuVsw4WCY1TH8qUsrPt4-wjxzkxjemC
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_scor
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
try:
    Data = pd.read_csv('data.csv',on_bad_lines='skip')
except Exception as e:
    print(f"Error loading the data: {e}")


# Extract features and labels
Passwords = Data['password']
Labels = Data['strength']

Passwords = Passwords.fillna('')

Passwords = Passwords.astype(str)

# Vectorize using TF-IDF with character n-grams
Vectorize = TfidfVectorizer(analyzer='char', ngram_range=(1,3), max_features=5000)  # Limit the feature space


try:
    Password_Vector = Vectorize.fit_transform(Passwords)
except Exception as e:
    print(f"Vectorization failed: {e}")

joblib.dump(Vectorize, 'Vectorize.joblib')
# Convert sparse matrix to array for compatibility
Input_Dataset = Password_Vector.toarray()
Output_Dataset = Labels


# Split into train/test sets
Input_Train, Input_Test, Output_Train, Output_Test = train_test_split(
    Input_Dataset, Output_Dataset, test_size=0.2, random_state=42
)

# Create and train the Logistic Regression model
Model = LogisticRegression(max_iter=200, solver='liblinear')  # Added solver and iteration limit
Model.fit(Input_Train, Output_Train)
joblib.dump(Model, 'Password.joblib')
# Test predictions
Prediction = Model.predict(Input_Test)
print(Prediction)
# Evaluate performance
Accuracy = accuracy_score(Output_Test, Prediction)

# Output the results
print(f"Accuracy: {Accuracy}")

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and vectorizer
Model = joblib.load('Password_RF.joblib')
Vectorize = joblib.load('Vectorize_RF.joblib')



# Predict using the loaded model
for _ in range(5):
  try:
      Password = input("Enter a password to predict its strength: ")
      Password_Vector = Vectorize.transform([Password]).toarray()

      P = Model.predict(Password_Vector)
      print("Predictions:")
      print(P[0])
  except ValueError as e:
      print(f"Error during prediction: {e}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

chunksize = 10000
model = LogisticRegression(max_iter=200, solver='liblinear')

Vectorize = TfidfVectorizer(analyzer='char', ngram_range=(1,3), max_features=5000)

for chunk in pd.read_csv('data.csv', chunksize=chunksize, on_bad_lines='skip'):
    # Extract features and labels from the current chunk
    Passwords = chunk['password'].fillna('').astype(str)
    Labels = chunk['strength']

    # Vectorize the passwords in the chunk
    Password_Vector = Vectorize.fit_transform(Passwords).toarray()

    # Split the chunk into train/test sets
    Input_Train, Input_Test, Output_Train, Output_Test = train_test_split(
        Password_Vector, Labels, test_size=0.2, random_state=42
    )

    # Train the model on the current chunk
    model.fit(Input_Train, Output_Train)



joblib.dump(model, 'Password.joblib')
joblib.dump(Vectorize, 'Vectorize.joblib')

Prediction = model.predict(Input_Test)
print(Prediction)

Accuracy = accuracy_score(Output_Test, Prediction)
print(f"Accuracy: {Accuracy}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

chunksize = 10000
# Initialize RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

Vectorize = TfidfVectorizer(analyzer='char', ngram_range=(1,3), max_features=5000)

for chunk in pd.read_csv('data.csv', chunksize=chunksize, on_bad_lines='skip'):
    # Extract features and labels from the current chunk
    Passwords = chunk['password'].fillna('').astype(str)
    Labels = chunk['strength']

    # Vectorize the passwords in the chunk
    Password_Vector = Vectorize.fit_transform(Passwords).toarray()

    # Split the chunk into train/test sets
    Input_Train, Input_Test, Output_Train, Output_Test = train_test_split(
        Password_Vector, Labels, test_size=0.2, random_state=42
    )

    # Train the model on the current chunk
    model.fit(Input_Train, Output_Train)


# Save the model and vectorizer
joblib.dump(model, 'Password_RF.joblib')  # Save as a different file
joblib.dump(Vectorize, 'Vectorize_RF.joblib') # Save as a different file

# Make predictions and evaluate performance
Prediction = model.predict(Input_Test)
print(Prediction)

Accuracy = accuracy_score(Output_Test, Prediction)
print(f"Accuracy: {Accuracy}")

"""THis is anothe model

"""

import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

chunksize = 10000
sample_size = 50000
test_size = 0.2


# Step 1: Debugging - Check the sample data columns
try:
    sample_data = pd.read_csv('data.csv', nrows=sample_size, on_bad_lines='skip')
    print("Columns in DataFrame:", sample_data.columns)  # Debugging check
    sample_data.columns = sample_data.columns.str.strip()  # Remove trailing/leading spaces
    if 'password' not in sample_data.columns:
        raise KeyError("'password' column does NOT exist in the data. Columns found are: " + str(sample_data.columns))

    # Ensure 'strength' column exists
    if 'strength' not in sample_data.columns:
        raise KeyError("'strength' column does NOT exist in the data. Columns found are: " + str(sample_data.columns))

    # Preprocess sample data
    Passwords_sample = sample_data['password'].fillna('').astype(str)
    Labels_sample = sample_data['strength'].astype(int)  # Ensure labels are integers
except Exception as e:
    print(f"Error reading sample data: {e}")
    exit()


# Initialize and fit vectorizer
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=5000)
vectorizer.fit(Passwords_sample)
joblib.dump(vectorizer, 'Vectorize_RF.joblib')

# Split sample data for initial training and testing
X_sample = vectorizer.transform(Passwords_sample)  # Vectorize the sample data
Input_Train, Input_Test, Output_Train, Output_Test = train_test_split(
    X_sample, Labels_sample, test_size=test_size, random_state=42
)

# Step 2: Initialize Incremental Model
model = SGDClassifier(random_state=42, loss='log_loss')  # 'log' is the same as log_loss
model.partial_fit(Input_Train, Output_Train, classes=[0, 1, 2])  # Pass all possible classes

# Step 3: Process chunks incrementally and train the model
try:
    for chunk in pd.read_csv('data.csv', chunksize=chunksize, skiprows=sample_size, on_bad_lines='skip'):
        # Debugging - Ensure the chunk contains expected columns
        chunk.columns = chunk.columns.str.strip()  # Clean column names
        if 'password' not in chunk.columns or 'strength' not in chunk.columns:
            raise KeyError("Expected columns ('password', 'strength') are missing in chunk. Columns: " + str(chunk.columns))

        # Process each chunk
        Passwords = chunk['password'].fillna('').astype(str)
        Labels = chunk['strength'].astype(int)  # Ensure labels are integers

        # Transform using the vectorizer
        Password_Vector = vectorizer.transform(Passwords)

        # Incrementally train the model
        model.partial_fit(Password_Vector, Labels)
except Exception as e:
    print(f"Error during incremental chunk training: {e}")
    exit()

# Save the trained model
joblib.dump(model, 'Password_RF.joblib')

# Step 4: Evaluate the model
try:
    # Predict on the test set
    Prediction = model.predict(Input_Test)
    accuracy = accuracy_score(Output_Test, Prediction)
    print(f"Accuracy: {accuracy}")
    print(classification_report(Output_Test, Prediction))
except Exception as e:
    print(f"Error during evaluation: {e}")

"""Random forest with only 100000 data"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

# Constants
num_rows_to_train = 140000  # Number of rows to train on
Vectorize = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=5000)

# Read only the first 100,000 rows from the CSV
data = pd.read_csv('data.csv', nrows=num_rows_to_train, on_bad_lines='skip')

# Extract features and labels
Passwords = data['password'].fillna('').astype(str)
Labels = data['strength']

Input_Train, Input_Test, Output_Train, Output_Test = train_test_split(
    Passwords, Labels, test_size=0.3, random_state=42
)

# Vectorize the passwords
Password_Vector = Vectorize.fit_transform(Input_Train).toarray()
Test_Vector_input = Vectorize.transform(Input_Test).toarray()

# Initialize RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the RandomForestClassifier on all 100,000 rows
model.fit(Password_Vector, Output_Train)

# Save the model and vectorizer
joblib.dump(model, 'Password_RF.joblib')
joblib.dump(Vectorize, 'Vectorize_RF.joblib')

# Evaluate model predictions on training data itself (since no split is performed)
Prediction = model.predict(Input_Test)
accuracy = accuracy_score(Output_Test, Prediction)

print(f"Training accuracy (on the first 140,000 rows): {accuracy}")

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd

data = pd.read_csv('data.csv', on_bad_lines='skip',nrows=140000)

Passwords = data['password'].fillna('').astype(str)
Lables = data['strength']

Input_Train, Input_Test, Output_Train, Output_Test = train_test_split(
    Passwords, Labels, test_size=0.05, random_state=42
)


# Load the saved model and vectorizer
Model = joblib.load('Password_RF.joblib')
Vectorize = joblib.load('Vectorize_RF.joblib')

Input_Test = Vectorize.transform(Input_Test).toarray()

Prediction = Model.predict(Input_Test)
print(Prediction)

Accuracy = accuracy_score(Output_Test, Prediction)
print(f"Accuracy: {Accuracy}")