from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

def training(df_train):
    numerical_columns = ['col_1', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7', 'col_8', 'col_9', 'col_10', 'col_11', 'col_12', 'col_14', 'col_15', 'col_16', 'col_18', 'col_19', 'col_21']

    # 使用KNNImputer補充數值特徵的缺失值
    imputer = KNNImputer(n_neighbors=5)
    df_train.loc[:, numerical_columns] = imputer.fit_transform(df_train[numerical_columns])

    categorical_columns = ['col_2', 'col_13', 'col_17', 'col_20']
    for col in categorical_columns:
        df_train[col].fillna(df_train[col].mode()[0], inplace=True)

    # numerical_columns = ['col_1', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7', 'col_8', 'col_9', 'col_10', 'col_11', 'col_12', 'col_14', 'col_15', 'col_16', 'col_18', 'col_19', 'col_21']
    # df_train.loc[:, numerical_columns] = df_train[numerical_columns].fillna(df_train[numerical_columns].median())

    # categorical_columns = ['col_2', 'col_13', 'col_17', 'col_20']
    # for col in categorical_columns:
    #     df_train[col].fillna(df_train[col].mode()[0], inplace=True)

    X = df_train.drop(columns=['CreditScore'])
    y = df_train['CreditScore']

    # 使用 SMOTE 进行过取样
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, shuffle=True)

    xgb_classifier = XGBClassifier(
        learning_rate=0.2,
        n_estimators=1000,
        max_depth=10,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        n_jobs=-1,
        random_state=42,
        gamma=0.3
    )

    xgb_classifier.fit(X_train, y_train)

    y_pred = xgb_classifier.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average='macro')

    print("Macro-F1 Score:", macro_f1)

    return xgb_classifier, X.columns

def predicting(xgb_model, df_test, relevant_features):
    numerical_columns = ['col_1', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7', 'col_8', 'col_9', 'col_10', 'col_11', 'col_12', 'col_14', 'col_15', 'col_16', 'col_18', 'col_19', 'col_21']

    df_test.loc[:, numerical_columns] = df_test[numerical_columns].fillna(df_test[numerical_columns].median())
    # # 使用KNNImputer補充數值特徵的缺失值
    # imputer = KNNImputer(n_neighbors=5)
    # df_test.loc[:, numerical_columns] = imputer.fit_transform(df_test[numerical_columns])

    categorical_columns = ['col_2', 'col_13', 'col_17', 'col_20']
    for col in categorical_columns:
        df_test[col].fillna(df_test[col].mode()[0], inplace=True)

    # numerical_columns = ['col_1', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7', 'col_8', 'col_9', 'col_10', 'col_11', 'col_12', 'col_14', 'col_15', 'col_16', 'col_18', 'col_19', 'col_21']
    # df_test.loc[:, numerical_columns] = df_test[numerical_columns].fillna(df_test[numerical_columns].median())

    # categorical_columns = ['col_2', 'col_13', 'col_17', 'col_20']
    # for col in categorical_columns:
    #     df_test[col].fillna(df_test[col].mode()[0], inplace=True)

    selected_features = df_test[relevant_features]

    y_pred = xgb_model.predict(selected_features)

    output = pd.DataFrame({'label': y_pred})
    output.to_csv('answer.csv', index_label='Id')

rf_model, relevant_features = training(df)
predicting(rf_model, df_test, relevant_features)