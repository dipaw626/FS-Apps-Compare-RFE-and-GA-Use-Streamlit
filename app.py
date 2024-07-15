# !pip install -U scikit-learn scipy matplotlib
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# feature selection and train test
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn_genetic import GAFeatureSelectionCV
# from sklearn.preprocessing import StandardScaler
# from sklearn import preprocessing
from deap import base
from deap import creator
from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import LabelEncoder

# algorithm
from sklearn.svm import SVC
# from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from IPython.display import display

st.set_page_config(layout='wide')

# Initialize models
models = [
('Logistic Regression', LogisticRegression()),
('K-Nearest Neighbors', KNeighborsClassifier()),
('Support Vector Machines', SVC()),
('Random Forest', RandomForestClassifier()),
('AdaBoost', AdaBoostClassifier()),
('Naive Bayes', GaussianNB())
]
    
# Fungsi untuk evaluasi model
def evaluate_model(model, X_test, y_test):
  y_pred = model.predict(X_test)
  accuracy = "{:.2f}%".format(accuracy_score(y_test, y_pred)*100)
  precision = "{:.2f}%".format(precision_score(y_test, y_pred, average='weighted')*100)
  recall = "{:.2f}%".format(recall_score(y_test, y_pred, average='weighted')*100)
  f1 = "{:.2f}%".format(f1_score(y_test, y_pred, average='weighted')*100)
  return accuracy, precision, recall, f1

def Input_CSV():
  # Initialize session_state
  if 'input_csv' not in st.session_state:
      st.session_state.input_csv = None

  st.title("CSV File Upload Page")

  # Create a file uploader widget
  input_csv = st.file_uploader("Upload your CSV file")

  if input_csv is not None:
      # Check if the file is empty or not
    content_type, extension = input_csv.type, input_csv.name.split(".")[-1]
    if content_type == "application/vnd.ms-excel" or extension.lower() != "csv":
        st.warning("Please upload a valid CSV file.")
        st.session_state.input_csv = None
    else:
      # Check if the file is empty or not
      try:
          df = pd.read_csv(input_csv)
          if df.empty:
              st.warning("The uploaded CSV file is empty.")
          else:
              st.session_state.input_csv = df  # Store the DataFrame in session_state
      except pd.errors.EmptyDataError:
          st.warning("The uploaded CSV file is empty or contains no valid data.")
      except Exception as e:
          st.error(f"An error occurred while reading the CSV file: {e}")

  # Display the uploaded CSV file (if available)
  if st.session_state.input_csv is not None:
      st.title("Uploaded CSV Data")
      st.dataframe(st.session_state.input_csv)


def Classification():
  
  # if 'input_csv' not in st.session_state:
  #   st.session_state.input_csv = None
  
  st.title("Evaluation Calculation Using Selection Features")
  st.header("Real Data CSV") 
  
  df = st.session_state.input_csv
  st.dataframe(df)
  col = df.columns.values
    
  st.divider()
  
  # stepp 1
  st.header("Step 1 Choose Target and Conversion")
    
  col1,col2,col3 = st.columns(3)
  
  with col1:
    # Check if the value has changed
    if 'target_variable' not in st.session_state:
        # If the value has changed, update it in session state
        lastindex = len(col) - 1
        target_variable = st.selectbox('select target or y variable',(col), index=lastindex) 
        st.session_state.target_variable = target_variable
    else:
      col1 = col.tolist()
      index = col1.index(st.session_state.target_variable)
      target_variable = st.selectbox('Select target or y variable', col, index=index)
    st.session_state.target_variable = target_variable
    
    X = df.drop(target_variable,axis=1)
    y = df[target_variable] 
    
    class_count = st.session_state.input_csv[target_variable].value_counts().reset_index()
    class_count.columns = [f'{target_variable}', 'Number of occurrences']
    st.write(class_count)
      

  with col2:
    if 'custom_conversion' not in st.session_state:
      st.session_state.custom_conversion = ""
      
    if target_variable in st.session_state.input_csv.columns:
      if st.session_state.input_csv[target_variable].dtype == 'object':
        example = '''***|The paramater input is class/target/y : conversion_value|
                |Example B:1, Me: 2, and so on|***'''
        st.markdown(example)
        st.session_state.custom_conversion = st.text_input('Input Conversion for Selected Variable', '')
        custom_conversion_dict = {}
        converted_values = None
        
        # Check if the user provided custom conversion input
        if st.session_state.custom_conversion:
            # Split the input into individual key-value pairs
            if not all(":" in pair for pair in st.session_state.custom_conversion.split(",")):
                st.warning("The input format does not match. Make sure each pair has a 'target/y : value' format.")
            else:
                try:
                    key_value_pairs = st.session_state.custom_conversion.split(',')
            
                    # Flag untuk menandai apakah semua kunci sesuai dengan dataset
                    all_keys_valid = True
                    
                    # Proses each key-value pair and add it to the custom conversion dictionary
                    for pair in key_value_pairs:
                        # Split the input into key and value
                        key, value = pair.split(':')
                        
                        # Trim whitespaces from key and value
                        key = key.strip()
                        value = value.strip()
                        
                        # Validasi apakah kunci ada dalam dataset
                        if key not in st.session_state.input_csv[target_variable].unique():
                            st.warning(f"'{key}' not found in the dataset.")
                            all_keys_valid = False
                        else:
                            # Tambahkan kunci ke dalam custom conversion dictionary
                            custom_conversion_dict[key] = float(value)

                    # Jika semua kunci sesuai dengan dataset
                    if all_keys_valid:
                        # Map and convert values using the custom conversion dictionary
                        converted_values = st.session_state.input_csv[target_variable].map(custom_conversion_dict).astype(float)
                    else:
                        # Jika tidak semua kunci sesuai dengan dataset, reset custom_conversion_dict
                        custom_conversion_dict = {}
                        
                except ValueError as ve:
                    st.error(f"Error: {ve}")
                    st.warning("Make sure the input format matches 'key : value, key : value'.")
                
        # Update DataFrame jika nilai telah dikonversi
        if converted_values is not None:
            st.session_state.input_csv[target_variable] = converted_values
            X = st.session_state.input_csv.drop(target_variable, axis=1)
            y = st.session_state.input_csv[target_variable]
   
     
  with col3:
    if st.session_state.custom_conversion is not None:
      st.write("Input Value: ", st.session_state.custom_conversion)
    else:
      st.write("Please Input Your Conversion")
    
    st.write("so here the conversion result of y or target variable", y)
      
  st.divider()
  
  # stepp 2
  st.header("Step 2 Set Split Data")
  col4,col5 = st.columns(2)
  
  with col4:
    if 'size' not in st.session_state:
      value=0.0
      size = st.slider('Test size', 0.0, 1.00, value)
      st.session_state.size = size
    else:
      size = st.slider('Test size', 0.0, 1.00, st.session_state.size)
    st.session_state.size = size
  
  with col5:
    if 'random' not in st.session_state:
      value=0
      random = st.number_input('Random seed (0-100)',min_value=0, max_value=100, value=value)
      st.session_state.random = random
    else:
      random = st.number_input('Random seed (0-100)',min_value=0, max_value=100, value=st.session_state.random)
    st.session_state.random = random
  st.divider()
  
  # stepp 3
  st.header("Step 3 Choose Feature Selection")
  
  # _____FEATURE SELECTION____
  
  Feature_selection = st.radio(
    "Feature selection",
    ["***Recursive Feature Selection***", "***Genetic Algorithm***"],
    captions = ["Running under 1 minute", "Running around 1 minute"]
  )
  
  st.markdown("<br>", unsafe_allow_html=True)
  
  if Feature_selection == '***Genetic Algorithm***':
    # Feature Selection ____GA____
    st.subheader('GA Selected Features')
    if st.button("Process Selections"):
      st.session_state.GA = "GA"
      
      svm = SVC(gamma='auto')
      ga = GAFeatureSelectionCV(svm,generations=10,cv=3)
      ga.fit(X, y)
      new_features_ga = ga.support_
      creator.create("FitnessMax", base.Fitness, weights=(1.0,))
      creator.create("Individual", list, fitness=creator.FitnessMax)
      
      if 'X_ga' not in st.session_state:
        # remove false features
        columns_to_remove = X.columns.values[np.logical_not(new_features_ga)]
        X_ga = X.drop(columns=columns_to_remove)
        st.session_state.X_ga = X_ga
      else:
        # remove false features
        columns_to_remove = X.columns.values[np.logical_not(new_features_ga)]
        X_ga = X.drop(columns=columns_to_remove)
      st.session_state.X_ga = X_ga
      
      #drop target or y
      selected_features = ", ".join(X_ga.columns.tolist())
      st.write(f'Selected Features:  ({selected_features})')
      
      if 'X_trainGA' not in  st.session_state:
        X_trainGA, X_testGA, y_trainGA, y_testGA = train_test_split(X_ga, y, test_size=size, random_state=random)
        st.session_state.X_trainGA = X_trainGA
        st.session_state.X_testGA = X_testGA
        st.session_state.y_trainGA = y_trainGA
        st.session_state.y_testGA = y_testGA
      else:
        X_trainGA, X_testGA, y_trainGA, y_testGA = train_test_split(X_ga, y, test_size=size, random_state=random)
      st.session_state.X_trainGA = X_trainGA
      st.session_state.X_testGA = X_testGA
      st.session_state.y_trainGA = y_trainGA
      st.session_state.y_testGA = y_testGA 
      
      # _____EVALUATION GA____
      st.subheader("GA Evaluation")
      
      # Create a dictionary to store the models
      saved_modelsGA = {}
      
      # Melatih dan mengevaluasi model
      resultsGA = []
      for name, model in models:
        model.fit(X_trainGA, y_trainGA)
        saved_modelsGA[name] = model
        accuracy, precision, recall, f1 = evaluate_model(model, X_testGA,
        y_testGA)
        resultsGA.append([name, accuracy, precision, recall, f1])
      
      # Save the dictionary with all models to a single file
      with open('all_modelsGA.pkl', 'wb') as models_fileGA:
          pickle.dump(saved_modelsGA, models_fileGA)
          
      # Menampilkan hasil dalam tabel perbandingan
      results_df_ga = pd.DataFrame(resultsGA, columns=['Model', 'Accuracy',
      'Precision', 'Recall', 'F1 Score'])
      
      # Create a unique key for each st.data_editor
      idx = 'a'
      unique_key = f'results_df_ga_{idx}' 
      st.data_editor(results_df_ga, key=unique_key)
      # eval = display(results_df)
      st.divider()
    
  else:
    # Feature Selection ____RFE_____
    # select the number of features
    if 'n_features_to_select' not in st.session_state:
      value=1
      n_features_to_select = st.slider("Select the number of features to keep", 1, X.shape[1], value)
      st.session_state.n_features_to_select = n_features_to_select
    else:
      n_features_to_select = st.slider("Select the number of features to keep", 1, X.shape[1], st.session_state.n_features_to_select)
    st.session_state.n_features_to_select = n_features_to_select
    
    st.subheader('RFE Selected Features')
    if st.button("Process Selections"):
      st.session_state.RFE = "RFE"
      
      estimator = SVC(kernel="linear")
      selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
      selector.fit(X, y)
      new_features_rfe= selector.support_
      
      if 'X_rfe' not in st.session_state:
        # remove false features
        columns_to_remove = X.columns.values[np.logical_not(new_features_rfe)]
        X_rfe = X.drop(columns=columns_to_remove)
        st.session_state.X_rfe = X_rfe
      else:
        # remove false features
        columns_to_remove = X.columns.values[np.logical_not(new_features_rfe)]
        X_rfe = X.drop(columns=columns_to_remove)
      st.session_state.X_rfe = X_rfe
      
      #drop target or y
      selected_features = ", ".join(X_rfe.columns.tolist())
      st.write(f'Selected Features:  ({selected_features})')
      
      if 'X_trainRFE' not in  st.session_state:
        X_trainRFE, X_testRFE, y_trainRFE, y_testRFE = train_test_split(X_rfe, y, test_size=size, random_state=random)
        st.session_state.X_trainRFE = X_trainRFE
        st.session_state.X_testRFE = X_testRFE
        st.session_state.y_trainRFE = y_trainRFE
        st.session_state.y_testRFE = y_testRFE
      else:
        X_trainRFE, X_testRFE, y_trainRFE, y_testRFE = train_test_split(X_rfe, y, test_size=size, random_state=random)
      st.session_state.X_trainRFE = X_trainRFE
      st.session_state.X_testRFE = X_testRFE
      st.session_state.y_trainRFE = y_trainRFE
      st.session_state.y_testRFE = y_testRFE 
          
      
      # _____EVALUATION RFE____
      st.subheader("RFE Evaluation")
    
      # Create a dictionary to store the models
      saved_modelsRFE = {}
      
      # Melatih dan mengevaluasi model
      resultsRFE = []
      for name, model in models:
        model.fit(X_trainRFE, y_trainRFE)
        saved_modelsRFE[name] = model
        accuracy, precision, recall, f1 = evaluate_model(model, X_testRFE,
        y_testRFE)
        resultsRFE.append([name, accuracy, precision, recall, f1])
          
      # Save the dictionary with all models to a single file
      with open('all_modelsRFE.pkl', 'wb') as models_fileRFE:
          pickle.dump(saved_modelsRFE, models_fileRFE)
      
      # Menampilkan hasil dalam tabel perbandingan
      results_df_rfe = pd.DataFrame(resultsRFE, columns=['Model', 'Accuracy',
      'Precision', 'Recall', 'F1 Score'])

      # Create a unique key for each st.data_editor
      idx = 'b'
      unique_key = f'results_df_rfe_{idx}'  
      st.data_editor(results_df_rfe, key=unique_key)
      # eval = display(results_df)
      # st.table(results_df_rfe)
      st.divider()
    
def Prediction():

  # if 'input_csv' not in st.session_state:
  #   st.session_state.input_csv = None
  
  # Create a selectbox to choose between feature sets
  feature_set = st.selectbox("Select Feature Set for Prediction:", ["Recursive Feature Selection", "Genetic Algorithm"])
  
  if feature_set == "Recursive Feature Selection" and 'X_rfe' in st.session_state:
    X_rfe = st.session_state.X_rfe
    st.subheader(f"Input {X_rfe.shape[1]} Feature Values for Prediction Using {st.session_state.RFE}:")
    
    # Load all models from the file
    with open('all_modelsRFE.pkl', 'rb') as model_file:
      saved_modelsRFE = pickle.load(model_file)
    
    # Create an empty dictionary to store feature values
    feature_values = {}
    # accuracies = {}
    
    pred1, pred2 = st.columns(2)
    
    with pred1:
      # Dynamically create number input fields for each feature
      for feature in X_rfe.columns:
          feature_values[feature] = st.number_input(f"{feature}:", min_value=0.0, step=0.001, format="%.3f")

    # Add a button to trigger the prediction
    if st.button("Predict"):
      with pred2:
        st.subheader("Prediction Results:")
        
        # Convert the feature values to a numpy array for prediction
        input_data = np.array(list(feature_values.values())).reshape(1, -1)

        for model_name, model in saved_modelsRFE.items():
            prediction = model.predict(input_data)
            # y_true = 1  # Define the true label according to your dataset
            # accuracy = accuracy_score([y_true], prediction)
            # accuracies[model_name] = accuracy
            # if prediction[0] == 0:
            #     st.write(f"{model_name} Accuracy - {accuracy * 100:.2f}% : The people in winsconsin diagnosed with benign breast cancer")
            # else:
            #     st.write(f"{model_name} Accuracy - {accuracy * 100:.2f}% : The people in winsconsin diagnosed with malignant breast cancer")
            if prediction[0] == 0:
                st.info(f"{model_name} : The people in winsconsin diagnosed with benign breast cancer")
            else:
                st.info(f"{model_name} : The people in winsconsin diagnosed with malignant breast cancer")
        
  elif feature_set == "Genetic Algorithm" and 'X_ga' in st.session_state:
    X_ga = st.session_state.X_ga
    st.subheader(f"Input {X_ga.shape[1]} Feature Values for Prediction Using {st.session_state.GA}:")
    
    # Load all models from the file
    with open('all_modelsGA.pkl', 'rb') as model_file:
      saved_modelsGA = pickle.load(model_file)
    
    # Create an empty dictionary to store feature values and accuracies
    feature_values = {}
    # accuracies = {}
    
    pred3, pred4 = st.columns(2)
    
    with pred3:
    # Dynamically create number input fields for each feature
      for feature in X_ga.columns:
          feature_values[feature] = st.number_input(f"{feature}:", min_value=0.0, step=0.001, format="%.3f")

    # Add a button to trigger the prediction
    if st.button("Predict"):
      with pred4:
        st.subheader("Prediction Results:")
        
        # Convert the feature values to a numpy array for prediction
        input_data = np.array(list(feature_values.values())).reshape(1, -1)

        for model_name, model in saved_modelsGA.items():
            prediction = model.predict(input_data)
            # y_true = 1  # Define the true label according to your dataset
            # accuracy = accuracy_score([y_true], prediction)
            # accuracies[model_name] = accuracy
            # if prediction[0] == 0:
            #     st.write(f"{model_name} Accuracy - {accuracy * 100:.2f}% : The people in winsconsin diagnosed with benign breast cancer")
            # else:
            #     st.write(f"{model_name} Accuracy - {accuracy * 100:.2f}% : The people in winsconsin diagnosed with malignant breast cancer")
            if prediction[0] == 0:
                st.info(f"{model_name} : The people in winsconsin diagnosed with benign breast cancer")
            else:
                st.info(f"{model_name} : The people in winsconsin diagnosed with malignant breast cancer")
  
  else:
    st.warning("Please perform the Classification step first before making predictions")
  
  
page = st.sidebar.selectbox("Select a page:", ("Input CSV", "Classification", "Prediction"))
if page == "Input CSV":
    Input_CSV()
    
elif page == "Classification":
  if 'input_csv' not in st.session_state:
    st.session_state.input_csv = None
  if st.session_state.input_csv is not None:
    Classification()
  else:
    st.warning("Please upload a CSV file in the 'Input CSV' page.")
    
elif page == "Prediction":
  if 'input_csv' not in st.session_state:
    st.session_state.input_csv = None
  if st.session_state.input_csv is not None:
    Prediction()
  else:
    st.warning("Please upload a CSV file in the 'Input CSV' page.")
    
    