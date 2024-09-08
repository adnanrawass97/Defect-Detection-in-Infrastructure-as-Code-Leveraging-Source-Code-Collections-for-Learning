
from sklearn import linear_model,preprocessing
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score,f1_score
from sklearn.model_selection import cross_val_score,cross_val_predict,train_test_split
from sklearn.metrics import precision_score, recall_score
import os
import time
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
def main(file_path,Result_file_path,iteration,fold2Use):
    # Load data
    all_features, labels =load_data(file_path)
    
    # Perform L1-penalized logistic regression to select features
    selected_indices_for_features =performPenalizedLogiRegression_n(all_features, labels)
    print("Total selected feature count:", len(selected_indices_for_features))
    print("-" * 50)

    # Create the selected feature matrix
    selected_features = createSelectedFeatures(all_features, selected_indices_for_features)
    print("Selected feature dataset size:", np.shape(selected_features))
    print("-" * 50)
    # Train the Random Forest model
      # Split the data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(selected_features, labels, test_size=0.2, random_state=42)  # Split data
    #trained_model = performRF(X_train, y_train,fold2Use, "R.F")
    print("Labels are", labels)
    performIterativeModeling(Result_file_path,selected_features, labels, fold2Use, iteration)


def performPenalizedLogiRegression_n(allFeatureParam, allLabelParam):
    feat_index_to_ret = []
    index = 0
    # Print a descriptive message
    print("<------------ Performing Logistic Regression ------------->")
    #normalizing 
    allFeatureParam_normalized = preprocessing.scale(allFeatureParam)
    # Initialization Logistic Regression model with L1 penalty

    logistic_model = linear_model.LogisticRegression(penalty='l1', solver='liblinear', max_iter=30)
    param_grid = {'C': np.logspace(-4, 4, 10)}
    
    # Performing grid search with cross-validation
    grid_search = GridSearchCV(logistic_model, param_grid, cv=2, scoring='accuracy')
    grid_search.fit(allFeatureParam_normalized, allLabelParam)
    
    # Retrieve the best parameter
    best_c = grid_search.best_params_['C']
    print(f"Optimal C: {best_c}")
    
    # Train the final model with the optimal C
    logisticRModel = linear_model.LogisticRegression(C=best_c, penalty='l1', solver='liblinear')

    # Fiting the model into the data
    logisticRModel.fit(allFeatureParam, allLabelParam)

    # Printing the mean accuracy score of the model
    print("Output of score (mean accuracy): {:.4f}".format(logisticRModel.score(allFeatureParam, allLabelParam)))
    print("the feature number  after regression ")
    # Retrieving the coefficients of the features
    feature_coeffs = logisticRModel.coef_[0]

    # Print the coefficients
    print("Output of coefficients: {}".format(feature_coeffs))

    # Looping through the coefficients to identify non-zero indices
    for x in np.nditer(feature_coeffs):
        if x != 0:
            feat_index_to_ret.append(index)
        index += 1
    #Retrieving the index      
    return feat_index_to_ret

def load_data(file_path):
    # Load the CSV file
    full_dataset_from_csv = pd.read_csv(file_path)
    print("Shape of full dataset:", full_dataset_from_csv.shape)
    print("-" * 50)
    # Label is in column 1, features start from column 3
    labels = full_dataset_from_csv.iloc[:, 1].values
    all_features = full_dataset_from_csv.iloc[:, 3:].values

    return all_features, labels


def performPenalizedLogiRegression(allFeatureParam, allLabelParam):
    feat_index_to_ret = []
    index = 0
    # Print a descriptive message
    print("<------------ Performing Logistic Regression ------------->")
    #normalizing 
    allFeatureParam_normalized = preprocessing.scale(allFeatureParam)
    # Initialization Logistic Regression model with L1 penalty
    logisticRModel = linear_model.LogisticRegression(C=100, penalty='l1', solver='liblinear')

    # Fiting the model into the data
    logisticRModel.fit(allFeatureParam, allLabelParam)

    # Printing the mean accuracy score of the model
    print("Output of score (mean accuracy): {:.4f}".format(logisticRModel.score(allFeatureParam, allLabelParam)))
    print("the feature number  after regression ")
    # Retrieving the coefficients of the features
    feature_coeffs = logisticRModel.coef_[0]

    # Print the coefficients
    print("Output of coefficients: {}".format(feature_coeffs))

    # Looping through the coefficients to identify non-zero indices
    for x in np.nditer(feature_coeffs):
        if x != 0:
            feat_index_to_ret.append(index)
        index += 1
    #Retrieving the index      
    return feat_index_to_ret

def createSelectedFeatures(allFeatureParam, selectedIndices):
    feature_dataset_to_ret = []
    for ind_ in selectedIndices:
        features_for_this_index = allFeatureParam[:, ind_]
        feature_dataset_to_ret.append(features_for_this_index)
    
    feature_dataset_to_ret = np.array(feature_dataset_to_ret)
    feature_dataset_to_ret = feature_dataset_to_ret.transpose()
    print("the feature are " ,feature_dataset_to_ret)
    return feature_dataset_to_ret


#random forest classfier
def performRF(featureParam, labelParam, foldParam, infoP):
    theRndForestModel = RandomForestClassifier()
    #theRndForestModel = RandomForestClassifier(max_features = int(1),    max_leaf_nodes = int(44),
                                          #min_samples_split=16, min_samples_leaf=8,
                                          #n_estimators=int(17))

    auc, precision, recall, f1= perform_cross_validation(theRndForestModel, featureParam, labelParam, foldParam, infoP)
    print("For {} area under ROC is: {}".format(infoP, auc))
    return auc, precision, recall, f1 # Return both metrics and model


def dumpContentIntoFile(strP, fileP):
  fileToWrite = open( fileP, 'w');
  fileToWrite.write(strP );
  fileToWrite.close()
  return str(os.stat(fileP).st_size)

def dumpPredPerfValuesToFile(iterations, predPerfVector, fileName):
   str2write=''
   headerStr='AUC,PRECISION,RECALL,F1'
   for c in range(iterations):
     auc   = predPerfVector[0][c]
     prec  = predPerfVector[1][c]
     recal  = predPerfVector[2][c]
     f1  = predPerfVector[3][c]

       
     str2write = str2write + str(auc) + ',' + str(prec) + ',' + str(recal) + ',' + str(f1)  + '\n'
     

   str2write = headerStr + '\n' + str2write
   bytes_ = dumpContentIntoFile(str2write, fileName)
   print ("Created {} of {} bytes".format(fileName, bytes_))


def dumpPredPerfValuesToFile_old(iterations, cv_pred_perf_vector, test_pred_perf_vector, file_name):
    """
    Dump cross-validation and testing performance values to a file.

    Parameters:
    - iterations: int, number of iterations.
    - cv_pred_perf_vector: tuple, cross-validation metrics (AUC, Precision, Recall, F1).
    - test_pred_perf_vector: tuple, testing metrics (AUC, Precision, Recall, F1).
    - file_name: str, the file path to save the metrics.
    """
    str2write = ''
    header_str = 'Metric,AUC,Precision,Recall,F1'
    
    for c in range(iterations):
        # Cross-validation metrics
        auc_cv = cv_pred_perf_vector[0][c]
        prec_cv = cv_pred_perf_vector[1][c]
        recal_cv = cv_pred_perf_vector[2][c]
        f1_cv = cv_pred_perf_vector[3][c]
        
        # Testing metrics
        auc_test = test_pred_perf_vector[0][c]
        prec_test = test_pred_perf_vector[1][c]
        recal_test = test_pred_perf_vector[2][c]
        f1_test = test_pred_perf_vector[3][c]

        # Append metrics to the string
        str2write += 'Cross-Validation,' + f'{auc_cv},{prec_cv},{recal_cv},{f1_cv}\n'
        str2write += 'Testing,' + f'{auc_test},{prec_test},{recal_test},{f1_test}\n'

    str2write = header_str + '\n' + str2write
    bytes_ = dumpContentIntoFile(str2write, file_name)
    print(f"Created {file_name} of {bytes_} bytes")




def evalClassifier(actualLabels, predictedLabels):

  target_labels =  ['0', '1']

  print( classification_report(actualLabels, predictedLabels, target_names=target_labels))

  #Getting the confusion matrix
 
  #conf_matr_output = confusion_matrix(actualLabels, predictedLabels)
  print( "Confusion matrix start")
  #print conf_matr_output
  conf_matr_output = pd.crosstab(actualLabels, predictedLabels, rownames=['True'], colnames=['Predicted'], margins=True)
  print (conf_matr_output)
  print ("Confusion matrix end")
  prec = precision_score(actualLabels, predictedLabels, average='binary')
  recall = recall_score(actualLabels, predictedLabels, average='binary')
  f1=f1_score(actualLabels, predictedLabels, average='binary')
  area_roc_output = roc_auc_score(actualLabels, predictedLabels)
  return area_roc_output, prec, recall, f1


def perform_cross_validation(classiferP, featuresP, labelsP, cross_vali_param, infoP):
  print ("-----Cross Validation#{}(Start)-----".format(infoP))
  predicted_labels = cross_val_predict(classiferP, featuresP , labelsP, cv=cross_vali_param)
  area_roc_output, prec, recall, f1 = evalClassifier(labelsP, predicted_labels)
  print ("-----Cross Validation#{}(End)-----".format(infoP))
  return area_roc_output, prec, recall,f1



def performIterativeModeling_del(result_file_path, feature_param, label_param, fold_param, iteration_p):
    # Array that contain metrics from each iteration random forest  with training data 
    rf_training_auc_lst, rf_training_prec_lst, rf_training_rec_lst, rf_training_f1_lst = [], [], [], []
    
    # Array that contain metrics from each iteration in random forest with test data 
    rf_test_auc_lst, rf_test_prec_lst, rf_test_recall_lst, rf_test_f1_lst = [], [], [], []
    
    for ind in range(iteration_p):
        # Generating a unique random seed using current timestamp and additional random value to make unique 
        random_seed = int(time.time()) + np.random.randint(0, 10000)

        # Split data into training and testing sets with the unique random seed
        X_train, X_test, y_train, y_test = train_test_split(feature_param, label_param, test_size=0.3, random_state=random_seed )

         #Initialize the Random Forest classifier
        
        rnd_forest_model = RandomForestClassifier(max_features = int(1),    max_leaf_nodes = int(44),
                                           min_samples_split=16, min_samples_leaf=8,
                                           n_estimators=int(17))

        rnd_forest_model = RandomForestClassifier()


        # Perform cross-validation
        auc_cv, precision_cv, recall_cv, f1_cv = perform_cross_validation(
            rnd_forest_model, X_train, y_train, fold_param, "Random Forest"
        )

        # Training the model on the training data
        rnd_forest_model.fit(X_train, y_train)

        # predictions on the testing data
        y_pred = rnd_forest_model.predict(X_test)
        y_pred_proba = rnd_forest_model.predict_proba(X_test)[:, 1]

        # Calculating performance metrics on the testing data
        auc_test = roc_auc_score(y_test, y_pred_proba)
        precision_test = precision_score(y_test, y_pred)
        recall_test = recall_score(y_test, y_pred)
        f1_test = f1_score(y_test, y_pred)

        # cross-validation metrics
        rf_training_auc_lst.append(auc_cv)
        rf_training_prec_lst.append(precision_cv)
        rf_training_rec_lst.append(recall_cv)
        rf_training_f1_lst.append(f1_cv)

        # Collect testing metrics
        rf_test_auc_lst.append(auc_test)
        rf_test_prec_lst.append(precision_test)
        rf_test_recall_lst.append(recall_test)
        rf_test_f1_lst.append(f1_test)

    # Print summary statistics for cross-validation metrics
    print("-" * 50)
    print("Cross-Validation Summary:")
    print("Summary: AUC, for: Random Forest, mean: {:.4f}, median: {:.4f}, max: {:.4f}, min: {:.4f}".format(
        np.mean(rf_training_auc_lst), np.median(rf_training_auc_lst), max(rf_training_auc_lst), min(rf_training_auc_lst)))
    print("Summary: Precision, for: Random Forest, mean: {:.4f}, median: {:.4f}, max: {:.4f}, min: {:.4f}".format(
        np.mean(rf_training_prec_lst), np.median(rf_training_prec_lst), max(rf_training_prec_lst), min(rf_training_prec_lst)))
    print("Summary: Recall, for: Random Forest, mean: {:.4f}, median: {:.4f}, max: {:.4f}, min: {:.4f}".format(
        np.mean(rf_training_rec_lst), np.median(rf_training_rec_lst), max(rf_training_rec_lst), min(rf_training_rec_lst)))
    print("Summary: F1 Score, for: Random Forest, mean: {:.4f}, median: {:.4f}, max: {:.4f}, min: {:.4f}".format(
        np.mean(rf_training_f1_lst), np.median(rf_training_f1_lst), max(rf_training_f1_lst), min(rf_training_f1_lst)))

    # Print summary statistics for testing metrics
    print("-" * 50)
    print("Testing Summary:")
    print("Summary: AUC, for: Random Forest, mean: {:.4f}, median: {:.4f}, max: {:.4f}, min: {:.4f}".format(
        np.mean(rf_test_auc_lst), np.median(rf_test_auc_lst), max(rf_test_auc_lst), min(rf_test_auc_lst)))
    print("Summary: Precision, for: Random Forest, mean: {:.4f}, median: {:.4f}, max: {:.4f}, min: {:.4f}".format(
        np.mean(rf_test_prec_lst), np.median(rf_test_prec_lst), max(rf_test_prec_lst), min(rf_test_prec_lst)))
    print("Summary: Recall, for: Random Forest, mean: {:.4f}, median: {:.4f}, max: {:.4f}, min: {:.4f}".format(
        np.mean(rf_test_recall_lst), np.median(rf_test_recall_lst), max(rf_test_recall_lst), min(rf_test_recall_lst)))
    print("Summary: F1 Score, for: Random Forest, mean: {:.4f}, median: {:.4f}, max: {:.4f}, min: {:.4f}".format(
        np.mean(rf_test_f1_lst), np.median(rf_test_f1_lst), max(rf_test_f1_lst), min(rf_test_f1_lst)))

    # Save the performance metrics to a file
    rf_cv_all_pred_perf_values = (rf_training_auc_lst, rf_training_prec_lst, rf_training_rec_lst, rf_training_f1_lst)
    rf_test_all_pred_perf_values = (rf_test_auc_lst, rf_test_prec_lst, rf_test_recall_lst, rf_test_f1_lst)
    
    base_dir = "/Users/adnan.alrawass/jupyter-1.0.0/project_snt/puppet_replication"
    result_csv_path = os.path.join(base_dir, result_file_path)
    
    dumpPredPerfValuesToFile(iteration_p, rf_cv_all_pred_perf_values, rf_test_all_pred_perf_values, result_csv_path)
    

def performIterativeModeling_old1(result_file_path, feature_param, label_param, fold_param, iteration_p):
    # Holders for cross-validation metrics
    rf_cv_auc_holder, rf_cv_prec_holder, rf_cv_recall_holder, rf_cv_f1_holder = [], [], [], []
    
    # Holders for testing metrics
    rf_test_auc_holder, rf_test_prec_holder, rf_test_recall_holder, rf_test_f1_holder = [], [], [], []
    
    for ind in range(iteration_p):
        # Generate a unique random seed using current timestamp and additional random value
        random_seed = int(time.time()) + np.random.randint(0, 10000)

        # Split data into training and testing sets with the unique random seed
        X_train, X_test, y_train, y_test = train_test_split(feature_param, label_param, test_size=0.3, random_state=random_seed)

        # Initialize the Random Forest classifier
        rnd_forest_model = RandomForestClassifier()

        # Perform cross-validation
        auc_cv, precision_cv, recall_cv, f1_cv = perform_cross_validation(
            rnd_forest_model, X_train, y_train, fold_param, "Random Forest"
        )

        # Train the model on the training set
        rnd_forest_model.fit(X_train, y_train)

        # Make predictions on the testing set
        y_pred = rnd_forest_model.predict(X_test)
        y_pred_proba = rnd_forest_model.predict_proba(X_test)[:, 1]

        # Calculate performance metrics on the testing set
        auc_test = roc_auc_score(y_test, y_pred_proba)
        precision_test = precision_score(y_test, y_pred)
        recall_test = recall_score(y_test, y_pred)
        f1_test = f1_score(y_test, y_pred)

        # Collect cross-validation metrics
        rf_cv_auc_holder.append(auc_cv)
        rf_cv_prec_holder.append(precision_cv)
        rf_cv_recall_holder.append(recall_cv)
        rf_cv_f1_holder.append(f1_cv)

        # Collect testing metrics
        rf_test_auc_holder.append(auc_test)
        rf_test_prec_holder.append(precision_test)
        rf_test_recall_holder.append(recall_test)
        rf_test_f1_holder.append(f1_test)

    # Printing summary statistics for cross-validation metrics
    print("-" * 50)
    print("Cross-Validation Summary:")
    print("Summary: AUC, for: Random Forest, mean: {:.4f}, median: {:.4f}, max: {:.4f}, min: {:.4f}".format(
        np.mean(rf_cv_auc_holder), np.median(rf_cv_auc_holder), max(rf_cv_auc_holder), min(rf_cv_auc_holder)))
    print("Summary: Precision, for: Random Forest, mean: {:.4f}, median: {:.4f}, max: {:.4f}, min: {:.4f}".format(
        np.mean(rf_cv_prec_holder), np.median(rf_cv_prec_holder), max(rf_cv_prec_holder), min(rf_cv_prec_holder)))
    print("Summary: Recall, for: Random Forest, mean: {:.4f}, median: {:.4f}, max: {:.4f}, min: {:.4f}".format(
        np.mean(rf_cv_recall_holder), np.median(rf_cv_recall_holder), max(rf_cv_recall_holder), min(rf_cv_recall_holder)))
    print("Summary: F1 Score, for: Random Forest, mean: {:.4f}, median: {:.4f}, max: {:.4f}, min: {:.4f}".format(
        np.mean(rf_cv_f1_holder), np.median(rf_cv_f1_holder), max(rf_cv_f1_holder), min(rf_cv_f1_holder)))

    # Printing summary statistics for testing metrics
    print("-" * 50)
    print("Testing Summary:")
    print("Summary: AUC, for: Random Forest, mean: {:.4f}, median: {:.4f}, max: {:.4f}, min: {:.4f}".format(
        np.mean(rf_test_auc_holder), np.median(rf_test_auc_holder), max(rf_test_auc_holder), min(rf_test_auc_holder)))
    print("Summary: Precision, for: Random Forest, mean: {:.4f}, median: {:.4f}, max: {:.4f}, min: {:.4f}".format(
        np.mean(rf_test_prec_holder), np.median(rf_test_prec_holder), max(rf_test_prec_holder), min(rf_test_prec_holder)))
    print("Summary: Recall, for: Random Forest, mean: {:.4f}, median: {:.4f}, max: {:.4f}, min: {:.4f}".format(
        np.mean(rf_test_recall_holder), np.median(rf_test_recall_holder), max(rf_test_recall_holder), min(rf_test_recall_holder)))
    print("Summary: F1 Score, for: Random Forest, mean: {:.4f}, median: {:.4f}, max: {:.4f}, min: {:.4f}".format(
        np.mean(rf_test_f1_holder), np.median(rf_test_f1_holder), max(rf_test_f1_holder), min(rf_test_f1_holder)))

    # Save the performance metrics to a file
    rf_cv_all_pred_perf_values = (rf_cv_auc_holder, rf_cv_prec_holder, rf_cv_recall_holder, rf_cv_f1_holder)
    rf_test_all_pred_perf_values = (rf_test_auc_holder, rf_test_prec_holder, rf_test_recall_holder, rf_test_f1_holder)
    
    base_dir = "/Users/adnan.alrawass/jupyter-1.0.0/project_snt/puppet_replication"
    result_csv_path = os.path.join(base_dir, result_file_path)






def performIterativeModeling(Result_file_path,featureParam, labelParam, foldParam, iterationP):
    rf_auc_holder, rf_prec_holder, rf_recall_holder, rf_f1_holder = [], [], [], []
    holder_rf = []
    

    for ind in range(iterationP):
        # Perform Random Forest modeling for current iteration
        rf_area_roc, rf_prec, rf_recall, rf_f1 = performRF(featureParam, labelParam, foldParam, "Random Forest")

        # Collect metrics for current iteration 
        holder_rf.append(rf_area_roc)
        rf_auc_holder.append(rf_area_roc)
        rf_prec_holder.append(rf_prec)
        rf_recall_holder.append(rf_recall)
        rf_f1_holder.append(rf_f1)
 
 
    print("-" * 50)
    print("Summary: AUC, for: {}, mean: {:.4f}, median: {:.4f}, max: {:.4f}, min: {:.4f}".format("Rand. Forest",
                                                                                                np.mean(holder_rf),
                                                                                                np.median(holder_rf),
                                                                                                max(holder_rf),
                                                                                                min(holder_rf)))
    print("*" * 25)
    print("Summary: Precision, for: {}, mean: {:.4f}, median: {:.4f}, max: {:.4f}, min: {:.4f}".format("Rand. Forest",
                                                                                                       np.mean(rf_prec_holder),
                                                                                                       np.median(rf_prec_holder),
                                                                                                       max(rf_prec_holder),
                                                                                                       min(rf_prec_holder)))
    print("*" * 25)
    print("Summary: Recall, for: {}, mean: {:.4f}, median: {:.4f}, max: {:.4f}, min: {:.4f}".format("Rand. Forest",
                                                                                                    np.mean(rf_recall_holder),
                                                                                                    np.median(rf_recall_holder),
                                                                                                    max(rf_recall_holder),
                                                                                                    min(rf_recall_holder)))
    print("*" * 25)
    print("Summary: F1 Score, for: {}, mean: {:.4f}, median: {:.4f}, max: {:.4f}, min: {:.4f}".format("Rand. Forest",
                                                                                                      np.mean(rf_f1_holder),
                                                                                                      np.median(rf_f1_holder),
                                                                                                      max(rf_f1_holder),
                                                                                                      min(rf_f1_holder)))

    print()
    print("*" * 25)
        
    rf_all_pred_perf_values = (holder_rf, rf_prec_holder, rf_recall_holder, rf_f1_holder)
    base_dir = "/Users/adnan.alrawass/jupyter-1.0.0/project_snt/puppet_replication"
    result_csv_path = os.path.join(base_dir,Result_file_path)
    dumpPredPerfValuesToFile(iterationP,rf_all_pred_perf_values,result_csv_path)
    #dumpPredPerfValuesToFile(iteration_p, rf_cv_all_pred_perf_values, rf_test_all_pred_perf_values, result_csv_path)
    print(result_csv_path)
    print("-" * 50)
    return rf_auc_holder, rf_prec_holder, rf_recall_holder, rf_f1_holder

print("Started at:", pd.Timestamp.now())
#Loading the structured_dtm.csv file
#file_path = "/Users/adnan.alrawass/jupyter-1.0.0/project_snt/puppet_replication/structured_dtm_tfidf.csv"
#full_dataset_from_csv = pd.read_csv(file_path)
#print("full_dataset_from_csv",full_dataset_from_csv)
#full_rows, full_cols = full_dataset_from_csv.shape
#print("Total number of columns", full_cols)

# Label is in column 1, features start from column 3
#labels = full_dataset_from_csv.iloc[:, 1].values
#all_features = full_dataset_from_csv.iloc[:, 3:].values

# Perform L1-penalized logistic regression to select features
#selected_indices_for_features = performPenalizedLogiRegression(all_features, labels)
#print("Total selected feature count:", len(selected_indices_for_features))
#print("-" * 50)

# Create the selected feature matrix
#selected_features = createSelectedFeatures(all_features, selected_indices_for_features)
#print("Selected feature dataset size:", np.shape(selected_features))
#print("-" * 50)


# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(selected_features, labels, test_size=0.2, random_state=42)

# Train the Random Forest model
#trained_model = performRF(X_train, y_train, "Training")
# this method runs the classifiers 'iteration' number of times
#iteration=1000
#fold2Use=10
#print("labels are",all_features)
#performIterativeModeling(selected_features,labels, fold2Use, iteration)
# print "-"*50
# Test the model
#y_pred = trained_model.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)
#print("\nTesting accuracy: {:.4f}".format(accuracy))

# Calculate testing AUC
#y_pred_proba = trained_model.predict_proba(X_test)
#test_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
#print("Testing AUC: {:.4f}".format(test_auc))

# Display classification report
#print("Classification Report:")
#print(classification_report(y_test, y_pred))

#print("-" * 50)
#print(f"The file used: {file_path}")
#print("-" * 50)

#num_features = all_features.shape[1]
#print(f"Number of features: {num_features}")
