# LogP-Prediction

Random forest regression method for the prediction of the octanol-water partion coefficient (logP), hydration energy and solvation energy. RF method includes recursive feature eliminations and hyperparameter optimisation. Recursive feature elimination removes features until a maximum of 100 out of the 1000+ features remains in order to improve computation speed. Feature importance is ranked and saved to enable analysis of key descriptors.

Analysis of the results can be found in my included thesis from page 57 onwards.

# Code example can be found below. [Can be found here](../main/logP/RF_09.03_opt_RFEC.py)
```
  import pandas as pd
  import numpy as np

  #Using Skicit-learn, import train test split, RF, random cross validation optimisation, recursive feature elimination, and metrics
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestRegressor as rfr
  from sklearn.model_selection import RandomizedSearchCV as rcv
  from sklearn.feature_selection import RFECV
  from sklearn import metrics

  #Seed for reproducable results
  seed = 121212
  np.random.seed(seed)

  #Defining variables for averages
  avgr2 = 0
  avgrmse = 0
  avgmae = 0
  avgsdep = 0

  #Defining hyperpara's
  #max number of features to consider at each split
  max_features = ['sqrt', 'log2',
                  0.333]
  #max number of levels in tree
  max_depth = [2, 3, 5, 8, 10, 13, 15, 20]
  #Minimum number of samples to spilt a node
  min_samples_split = [2, 3, 4, 5, 7, 10]
  #Minimum number of samples required at each leaf node
  min_samples_leaf = [1, 2, 3, 5]

  grid_param = {'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf}

  #Set how many times the RF model will run
  runtimes = 100

  #Set up dataframe for storing data
  stored_preds = pd.read_csv('predictions_template.csv')
  stored_opt = pd.DataFrame()
  stored_import = pd.DataFrame(pd.read_csv('features_template.csv'))
  stored_metrics = pd.DataFrame()

  #Running model for n times
  for n in range(runtimes):
      # Read in data as pandas dataframe
      features = pd.read_csv('mordredRF.csv')

      # Label = the value we want to predict; logP
      labels = np.array(features['logP'])

      # Features = what the rf will use\
      # Remove logP, axis = 1 refers to columns, unsure what it does
      features = features.drop('logP', axis=1)

      # Saving feature names for later use, this takes the strings for the first row
      feature_list = list(features.columns)

      # Convert to numpy array
      features = np.array(features)

      # Split the data into training and testing sets, 75/25 split
      train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                                random_state=seed * n)

      # Save test feature ID's
      test_names = test_features[:, 0]
      # Drop ID's from features
      test_features = np.delete(test_features, 0, 1)
      train_features = np.delete(train_features, 0, 1)
      feature_list = np.delete(feature_list, np.where(feature_list == "logP"))
      feature_list = np.delete(feature_list, np.where(feature_list == "ID"))
      test_features = pd.DataFrame(test_features, columns=feature_list)

      # Instantiate model
      rf = rfr(n_estimators=500, random_state=seed * n)

      # Run optimisation
      rf_opt = rcv(estimator=rf, param_distributions=grid_param, n_iter=1000, cv=5, verbose=1, random_state=seed * n,
                 n_jobs=-1, scoring='neg_mean_squared_error')

      # Train model on training data
      rf_opt.fit(train_features, train_labels)

      # Update model with optimised hyperparameters
      best_params = rf_opt.best_params_
      rf.set_params(**best_params)

      # Saving optimisation results
      best_params = pd.Series(best_params)
      best_params = best_params.to_frame().transpose()
      stored_opt = stored_opt.append(best_params)

      # Recursive feature elimination cross validation with optimised hyperparameters, first with 100 removed per step till at least 400 remain...
      rf_opt_RFEC = RFECV(estimator=rf, step=100, min_features_to_select=400, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
      train_features = rf_opt_RFEC.fit_transform(train_features, train_labels)
      feature_list = feature_list[rf_opt_RFEC.support_]

      # Then 25 removed until 200 remain...
      rf_opt_RFEC = RFECV(estimator=rf, step=25, min_features_to_select=200, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
      train_features = rf_opt_RFEC.fit_transform(train_features, train_labels)
      feature_list = feature_list[rf_opt_RFEC.support_]

      # Then 10 removed until 100 remain...
      rf_opt_RFEC = RFECV(estimator=rf, step=10, min_features_to_select=100, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
      train_features = rf_opt_RFEC.fit_transform(train_features, train_labels)
      feature_list = feature_list[rf_opt_RFEC.support_]

      # Finally stepwise removal of features, RFE was done this way to decrease computational expensive
      rf_opt_RFEC = RFECV(estimator=rf, step=1, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
      train_features = rf_opt_RFEC.fit_transform(train_features, train_labels)
      feature_list = feature_list[rf_opt_RFEC.support_]

      # Predict logP
      rf.fit(train_features, train_labels)
      test_features = test_features[feature_list]
      predictions = rf.predict(test_features.values)

      # Store predictions
      pred = np.stack((test_names, predictions), axis=-1)
      pred = pd.DataFrame(data=pred,
                          index=np.array(range(1, 28)),
                          columns=np.array(range(1, 3)))
      pred = pred.rename(columns={1: "ID", 2: "Pred"})
      stored_preds = pd.merge(stored_preds, pred, on="ID", how='left').fillna(0)

      # Store recursive feature rankings and number of features remaining per split
      import_feat = pd.DataFrame(feature_list).T
      importance = pd.Series(rf.feature_importances_)
      importance.name = 1
      import_feat = import_feat.append(importance)
      import_feat.columns = import_feat.iloc[0]
      import_feat = import_feat.drop(import_feat.index[0])
      frames = [stored_import, import_feat]
      stored_import = pd.concat(frames)

      # Metrics
      r2 = metrics.r2_score(test_labels, predictions)
      rmse = np.sqrt(metrics.mean_squared_error(test_labels, predictions))
      mae = metrics.mean_absolute_error(test_labels, predictions)
      sdep = rmse - (mae ** 2)

      # Add to average metrics
      avgr2 = avgr2 + r2
      avgrmse = avgrmse + rmse
      avgmae = avgmae + mae
      avgsdep = avgsdep + sdep
      metrics_values = pd.DataFrame([r2, rmse, mae, sdep]).T
      stored_metrics = stored_metrics.append(metrics_values)
      print('For run number', n, 'R2:', r2, 'RMSE:', rmse, 'Bias:', mae, 'SDEP:', sdep)

      # Store values to csv
      stored_preds.to_csv("stored_predictions.csv")
      stored_opt.to_csv("stored_opt.csv")
      stored_import.to_csv("stored_import.csv")
      stored_metrics.to_csv("stored_metrics.csv")

  avgr2 = avgr2 / runtimes
  avgrmse = avgrmse / runtimes
  avgmae = avgmae / runtimes
  avgsdep = avgsdep / runtimes

  metrics_values = pd.DataFrame([avgr2, avgrmse, avgmae, avgsdep]).T
  stored_metrics = stored_metrics.append(metrics_values)
  print('Average: R2:', avgr2, 'RMSE:', avgrmse, 'Bias:', avgmae, 'SDEP', avgsdep)

  stored_preds.to_csv("stored_predictions.csv")
  stored_opt.to_csv("stored_opt.csv")
  stored_import.to_csv("stored_import.csv")
  stored_metrics.to_csv("stored_metrics.csv")
```
