import joblib

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
      if P[0] == 1:
         print("Medium")

      elif P[0] == 0:
         print("Weak")

      else:
         print("strong")


  except ValueError as e:
      print(f"Error during prediction: {e}")
