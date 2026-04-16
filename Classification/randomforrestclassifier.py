import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier


class RandomForrestACDC:
    def __init__(self, test_patients, train_patients):
        self.test_patients = pd.DataFrame(test_patients)             # input of patients as list, but needed in dataframe for further processing 
        self.train_patients = pd.DataFrame(train_patients)           # input of patients as list, but needed in dataframe for further processing
        self.best_rf = []

    def trainer(self):
        '''Function that sets up a random forrest classifier for classifying the heart disease
        with the help of patient information and information extracted from the segmentation mask
        of the training data'''
        
        # seperate the features from the target, the target is the classification of the patient indciated in 'Group'
        x,y = self.train_patients.loc[:, (self.train_patients.columns != 'Group')], self.train_patients['Group']

        # convert categories of 'Group' to numbers because RFC cannot deal with strings
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(y)

        # apply standard scaler value; all features have zero mean and unit variance, potentially boost the performance of random forest
        self.scaler = StandardScaler()
        x_train = self.scaler.fit_transform(x)

        # adding hyper parameters with the help of a param_grid
        param_grid = {
            'n_estimators': [10,  100, 125, 200 ], #number of trees in the forest
            'max_depth': [3, 5, 10,20, None], # max level of each level of each tree
            'max_features': ['sqrt', 'log2', None], #max number of features to take into account
            'min_samples_leaf': [1, 2, 4], # minimal number of samples required to be in a leaf node, which cannot be split further
            'bootstrap': [True, False]
        }

        # initialize the random forrest classifier
        rf_model = RandomForestClassifier(random_state=42)

        # set up a grid search algorithm to find the best random forrest classifier
        grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
        grid_search.fit(x_train, y_train)
        
        # safe the most accurate random forrest classifier, so it can be used in the future
        self.best_rf = grid_search.best_estimator_
    
        return self.scaler, self.label_encoder, self.scaler


    def tester(self):
        '''Function that tests the random forrest classifier for classifying the heart disease
        with the help of patient information and information extracted from the segmentation mask
        of the test data'''

        x_test,y_test = self.test_patients.loc[:, (self.test_patients.columns != 'Group')], self.test_patients['Group']
        
        self.x_test = self.scaler.transform(x_test)
        self.y_test = self.label_encoder.transform(y_test)
        self.y_pred = self.best_rf.predict(self.x_test)

        self.test_accuracy = accuracy_score(self.y_test, self.y_pred)

        print(f"Test Accuracy:{self.test_accuracy:.4f}")
        print("\nClassification Report:\n", classification_report(self.y_test, self.y_pred))
        print("\nConfusion Matrix:\n", confusion_matrix(self.y_test, self.y_pred))

        return self.test_accuracy
    
    def roc_curve(self): 
        '''Function that creates a ROC curve of the test data'''
        # Get predicted probabilities
        y_score = self.best_rf.predict_proba(self.x_test)

        # Ensure y_test is one-hot encoded to be able to create ROC curves
        y_test_bin = label_binarize(self.y_test, classes=[0, 1, 2, 3, 4])
        
        # Initialize plot
        plt.figure(figsize=(8, 6))

        # Colors for different classes
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        class_names = ['DCM', 'HCM', 'MINF', 'NOR', 'RV']

        AUC = []
        # Plot ROC curve for each class
        for i in range(5):  # Loop over each class
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])  # Compute ROC
            roc_auc = auc(fpr, tpr)  # Compute AUC
            plt.plot(fpr, tpr, color=colors[i])
            AUC.append(roc_auc)
            plt.plot(fpr, tpr, color=colors[i], label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

        # Plot diagonal reference line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)

        # Customize the plot
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('One-vs-Rest (OvR) ROC Curves for Multi-Class Classification')
        plt.legend(loc='lower right')
        plt.show()
        return 


    
    def load_model(self, rf_path='best_rf_model.pkl', scaler_path='scaler.pkl', encoder_path='label_encoder.pkl'):
        '''Function that enables the use of a saved model, so there is no need to perform the
        whole training before testing'''

        self.best_rf = joblib.load(rf_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        
        return 
