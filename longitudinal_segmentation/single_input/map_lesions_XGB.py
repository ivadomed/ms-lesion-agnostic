"""
In this script, we perform lesion mapping between two timepoints using an XGBoost model to predict lesion correspondences based on features extracted from the lesions.

Inputs:
    - dataset_csv: Path to the csv dataset containing lesion features over time.
    - output_folder: Path to the folder where model and results will be saved.

Output:
    None

Author: Pierre-Louis Benveniste
"""
import os
import pandas as pd
import argparse
from loguru import logger
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, help='Path to the csv dataset containing lesion features over time')
    parser.add_argument('-o', '--output-folder', type=str, required=True, help='Path to the output folder where model and results will be stored')
    return parser.parse_args()

def distance_cylindrical(coord1, coord2, w_z=1.0, w_disk=1.0):
    """
    Computes a weighted Euclidean distance between two points in cylindrical coordinates.
    
    Parameters:
        coord1 (dict): Coordinates of the first point with keys 'r', 'theta', and 'z'.
        coord2 (dict): Coordinates of the second point with keys 'r', 'theta', and 'z'.
        w_z (float): Weight for the z-axis distance.
        w_disk (float): Weight for the distance in the disk plane (r, theta).
    
    Returns:
        float: The weighted Euclidean distance.
    """
    distance_disk = np.sqrt(coord1['r']**2 + coord2['r']**2 - 2 * coord1['r'] * coord2['r'] * np.cos(np.radians(coord1['theta'] - coord2['theta'])))
    z_dist = coord1['z'] - coord2['z']
    return np.sqrt(w_z * z_dist**2 + w_disk * distance_disk**2)


def main():
    args = parse_args()
    dataset_csv = args.data
    output_folder = args.output_folder

    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Create a logger
    logger.add(os.path.join(output_folder, 'lesion_mapping_XGB.log'))

    # Load dataset
    df = pd.read_csv(dataset_csv)

    # We transform the df to create one row per lesion pair in timepoint1 and timepoint2 (pairing only within the same subject)
    ## Columns are : subject,timepoint,group,z,r,theta,volume
    lesion_pairs = []
    subjects = df['subject'].unique()

    for subject in subjects:
        df_subject = df[df['subject'] == subject]
        timepoints = df_subject['timepoint'].unique()
        if len(timepoints) != 2:
            continue  # We only process subjects with exactly 2 timepoints
        tp1, tp2 = timepoints

        df_tp1 = df_subject[df_subject['timepoint'] == tp1]
        df_tp2 = df_subject[df_subject['timepoint'] == tp2]

        for _, lesion1 in df_tp1.iterrows():
            for _, lesion2 in df_tp2.iterrows():
                pair_features = {
                    'subject': subject,
                }
                # Add features from lesion1 and lesion2
                for col in df.columns:
                    if col not in ['subject', 'timepoint', 'group']:
                        pair_features[f'{col}1'] = lesion1[col]
                        pair_features[f'{col}2'] = lesion2[col]
                # Calculate explicit differences
                dz = lesion1['z'] - lesion2['z']
                dr = lesion1['r'] - lesion2['r']
                dtheta = lesion1['theta'] - lesion2['theta']
                d_vol = abs(lesion1['volume'] - lesion2['volume'])
                dist = distance_cylindrical({'r': lesion1['r'], 'theta': lesion1['theta'], 'z': lesion1['z']},
                                            {'r': lesion2['r'], 'theta': lesion2['theta'], 'z': lesion2['z']})
                # Add these to pair_features
                pair_features['dz'] = dz
                pair_features['dr'] = dr
                pair_features['dtheta'] = dtheta
                pair_features['dvol'] = d_vol
                pair_features['dist'] = dist
                # Add label if available (1 if they are the same lesion, else 0)
                pair_features['label'] = 1 if lesion1['group'] == lesion2['group'] else 0
                lesion_pairs.append(pair_features)

    # Create a DataFrame from lesion pairs
    df_pairs = pd.DataFrame(lesion_pairs)

    print(df_pairs.head())

    # Split the dataset into training and testing sets (done on subject level to avoid data leakage)
    subjects = df_pairs['subject'].unique()
    ## We arbitrarily choose that subjects from the Toronto site are used for testing
    test_subjects = [s for s in subjects if 'tor' in s]
    train_subjects = [s for s in subjects if s not in test_subjects]
    train_df = df_pairs[df_pairs['subject'].isin(train_subjects)]
    test_df = df_pairs[df_pairs['subject'].isin(test_subjects)]
    logger.info(f"Total subjects: {len(subjects)}, Training subjects: {len(train_subjects)}, Testing subjects: {len(test_subjects)}")

    # Split the dataset into features and labels
    feature_cols = [col for col in df_pairs.columns if col not in ['subject', 'label']]
    X_train = train_df[feature_cols]
    y_train = train_df['label']
    X_test = test_df[feature_cols]
    y_test = test_df['label']
    logger.info(f"Training set size: {X_train.shape[0]} pairs")
    logger.info(f"Testing set size: {X_test.shape[0]} pairs")

    # ## Strat 1
    # model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    # model.fit(X_train, y_train)
    # # print model parameters
    # logger.info(f"XGBoost model parameters: {model.get_params()}")

    # # Strat 2
    # # Train an XGBoost model
    # search_spaces = {
    #     'max_depth': Integer(3, 10), # Lower values prevent overfitting
    #     'min_child_weight': Integer(1, 10), # Higher values prevent overfitting # Suggested to go as high as 5 by LeChat
    #     'subsample': Real(0.5, 1), # Lower values prevent overfitting
    #     'colsample_bytree': Real(0.001, 1), # Lower values prevent overfitting # Suggested to go as low as 0.5 by LeChat
    #     'learning_rate': Real(0.01, 0.5, prior='log-uniform'),
    # }
    # # We define the model
    # model = XGBClassifier(seed=42, eval_metric='logloss')  # We set scale_pos_weight to the ratio of negative to positive samples
    # # We define the search
    # search = BayesSearchCV(model, search_spaces, n_iter=100, n_jobs=1, cv=3, random_state=42, scoring='roc_auc')
    # # We fit the search
    # search.fit(X_train, y_train)
    # model = search.best_estimator_

    ## Strat 3
    # # import the library
    # from sklearn.naive_bayes import MultinomialNB
    # # instantiate & fit
    # model = MultinomialNB().fit(X_train, y_train)

    ## Strat 4
    # from sklearn.linear_model import LogisticRegression
    # # instantiate & fit
    # lr=LogisticRegression(max_iter=5000)
    # lr.fit(X_train, y_train)
    # model = lr

    ## Strat 5
    # from sklearn.linear_model import SGDClassifier
    # # instantiate & fit
    # sgd=SGDClassifier()
    # sgd.fit(X_train, y_train)
    # model = sgd

    # Strat 6
    # from sklearn.neighbors import KNeighborsClassifier
    # # instantiate & fit
    # knn = KNeighborsClassifier(algorithm = 'brute', n_jobs=-1)
    # knn.fit(X_train, y_train)
    # model = knn

    # # Strat 7
    # from sklearn.svm import LinearSVC
    # # instantiate & fit
    # svm=LinearSVC(C=0.0001)
    # svm.fit(X_train, y_train)
    # model = svm

    # # Strat 8
    # # import the library
    # from sklearn.ensemble import AdaBoostClassifier
    # from sklearn.tree import DecisionTreeClassifier
    # # instantiate & fit
    # adb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),n_estimators=100,learning_rate=0.5)
    # adb.fit(X_train, y_train)
    # model = adb

    # Strat 9
    # import the library
    from keras import layers
    from keras import models
    from keras import optimizers
    from keras import losses
    from keras import regularizers
    from keras import metrics
    from tqdm.keras import TqdmCallback

    # add validation dataset
    validation_split=100
    x_validation=X_train[:validation_split]
    x_partial_train=X_train[validation_split:]
    y_validation=y_train[:validation_split]
    y_partial_train=y_train[validation_split:]

    from keras import layers, models, regularizers, Input

    def build_residual_mlp(input_dim):
        inputs = Input(shape=(input_dim,))
        
        # Block 1
        x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.001))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Block 2 (Residual)
        res = x # Skip connection start
        x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Add()([x, res]) # Skip connection add
        
        # Output
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model


    # build & compile model
    # model=models.Sequential()
    # model.add(layers.Dense(4,kernel_regularizer=regularizers.l2(0.003),activation='relu',input_shape=(14,)))
    # model.add(layers.Dropout(0.7))
    # model.add(layers.Dense(4,kernel_regularizer=regularizers.l2(0.003),activation='relu'))
    # model.add(layers.Dropout(0.7))
    # model.add(layers.Dense(1,activation='sigmoid'))
    
    # With an MLP
    model = build_residual_mlp(X_train.shape[1])
    
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['AUC'])

    # fir the model
    model.fit(x_partial_train,y_partial_train,epochs=500,batch_size=50,validation_data=(x_validation,y_validation), callbacks=[TqdmCallback(verbose=0)],verbose=0)

    # Predict on test set
    y_pred = model.predict(X_test)
    y_pred_labels = (y_pred > 0.5).astype(int).flatten()

    # Compute TP, FP, FN on the test set
    tp = sum((y_test == 1) & (y_pred_labels == 1))
    fp = sum((y_test == 0) & (y_pred_labels == 1))
    fn = sum((y_test == 1) & (y_pred_labels == 0))
    logger.info(f"Test set - True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test set - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")


    # # Evaluate the model
    # train_accuracy = model.score(X_train, y_train)
    # test_accuracy = model.score(X_test, y_test)
    # logger.info(f"Training accuracy: {train_accuracy:.4f}")
    # logger.info(f"Testing accuracy: {test_accuracy:.4f}")

    # # Compute TP, FP, FN on the training set
    # y_train_pred = model.predict(X_train)
    # tp = sum((y_train == 1) & (y_train_pred == 1))
    # fp = sum((y_train == 0) & (y_train_pred == 1))
    # fn = sum((y_train == 1) & (y_train_pred == 0))
    # logger.info(f"Training set - True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")

    # # Compute TP, FP, FN on the test set
    # y_pred = model.predict(X_test)
    # tp = sum((y_test == 1) & (y_pred == 1))
    # fp = sum((y_test == 0) & (y_pred == 1))
    # fn = sum((y_test == 1) & (y_pred == 0))
    # logger.info(f"Test set - True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")

    return None


if __name__ == "__main__":

    main()