**Overview:**
This project aims to create a predictive model that suggests a hobby (academics, sports, art) for an individual based on certain parameters provided in the dataset. The model will analyze various features of individuals, such as academic performance, participation in sports activities, and engagement in artistic endeavors, to predict their most suitable hobby category.


**Data Preprocessing:**
To prepare the data for model training and testing:
- Categorical variables are encoded using label encoding to convert strings into numeric values.
- The dataset is split into training and testing sets using the train_test_split function from Scikit-learn. This enables testing the models on independent data to 
  assess their performance accurately.


**Model Training and Evaluation:**
The following machine learning models are trained and evaluated for predicting hobby categories:

**Logistic Regression:**
Trained using the logistic regression algorithm from Scikit-learn.
Achieved accuracy of 0.9140893470790378.

**Support Vector Machine (SVM):**
Implemented using the SVM classifier from Scikit-learn.
Attained accuracy of 0.9381443298969072.

**Random Forest:**
Utilized the random forest classifier from Scikit-learn.
Obtained accuracy of 0.9278350515463918.

**Decision Tree:**
Constructed using the decision tree classifier from Scikit-learn.
Achieved accuracy of 0.8900343642611683.

**Conclusion:**
The hobby prediction model demonstrates varying accuracies across different machine learning algorithms. Further optimization and fine-tuning of the models may be required to improve accuracy and generalize well to unseen data. Additionally, feature engineering and model selection techniques could be explored to enhance the model's predictive performance.
