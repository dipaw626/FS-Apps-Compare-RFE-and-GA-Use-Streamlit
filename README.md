# Comparison of Genetic Algorithm and Recursive Feature Elimination on High Dimensional Data with Streamlit
This project aims to compare RFE and GA feature selection, with evaluation performance using 6 supervised algorithms namely Logistic Regression, K-Nearest Neighbors, Support Vector Machines, Random Forest, AdaBoost, and Naive Bayes on a dataset called WDBC.

### Dataset Source : https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
### Dataset Download : https://bit.ly/DatasetAndAlgorithm

The source dataset above is a raw dataset from research in Wisconsin, then for the easy-to-read file it has been made into csv in the download dataset.

#### Framework for website: "Streamlit"
#### Build supervised algorithm and feature selection: "Google Collab"
#### Python libraries that are used: "pandas, numpy, pickle, sklearn"

## HOW TO RUNNING
1. open terminal and type "python -m streamlit run app.py" or "streamlit run app.py" in Visual Studio Code
2. the web auto directly to open in browser or if not, just click the Local URL.
3. Input the dataset in Input CSV, after that click Classification section
4. Input all boundaries like data split and others, and choose Features Selection between RFE or GA to perform classification
5. After that, wait the perform until showing the result of the model.
6. After got the model from the result's algorithm, go to the Prediction section and choose either.
7. Input the number from that selection features, and after all filled then click prediction to showing the prediction from six algorithm.

## OUTPUT ORDER OF WEBSITE USAGE
##### i. Input CSV
![i1](https://github.com/dipaw626/Comparing-RFE-and-GA-Using-Streamlit/blob/main/Output%20Web/Input%20CSV%20File.jpeg)

##### ii. Classification
![c1](https://github.com/dipaw626/Comparing-RFE-and-GA-Using-Streamlit/blob/main/Output%20Web/Classification_Choose%20Target%20and%20Conversion.jpeg)

![c2](https://github.com/dipaw626/Comparing-RFE-and-GA-Using-Streamlit/blob/main/Output%20Web/Classification_Set%20Split%20Data.jpeg)

![c3](https://github.com/dipaw626/Comparing-RFE-and-GA-Using-Streamlit/blob/main/Output%20Web/Classification_Choose%20Feature%20Selection.jpeg)

![c4](https://github.com/dipaw626/Comparing-RFE-and-GA-Using-Streamlit/blob/main/Output%20Web/Classification_Selected%20Features%20and%20Evaluation.jpeg)

##### iii. Prediction
![p1](https://github.com/dipaw626/Comparing-RFE-and-GA-Using-Streamlit/blob/main/Output%20Web/Prediction_Feature%20Selection.jpeg)


