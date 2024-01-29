import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

def load_dataset(link):
	return pd.read_csv(link)

def clean_data(df):
    df = df.copy()

	#Convert Date to separate day, month and year columns and drop Date
    df['Day'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    cols = df.columns.tolist()
    cols = cols[-3:] + cols[:-3]
    df = df[cols]
    df = df.drop('Date', axis=1)    
    
    #Remove empty rows for RainTomorrow since that is our label and RainToday for simplicity
    df = df[(df['RainTomorrow'].notna()) & (df['RainToday'].notna())]

    #Remove all columns with NA ratio greater than 30% threshold
    threshold = 0.3
    size = df.shape[0]
    exclusion_list = []
    for col in df.columns.tolist():
        na_ratio = df[col].isna().sum()/size
        if na_ratio > threshold:
            exclusion_list.append(col)
    df = df.drop(exclusion_list, axis=1)

    #Perform mean imputation to all remaining numeric columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    #Perform mode imputation to all remaining categorical columns
    cat_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    #Convert all categorical columns to numeric factor values
    cat_cols = cat_cols + ['RainToday', 'RainTomorrow', 'Location', 'Year']
    for col in cat_cols:
        df[col] = pd.factorize(df[col])[0]
        

    return df

def split_x_y(x):
     return np.hsplit(x, [20])

def split_data(df):
    train, validate, test = np.split(
        df.sample(frac=1, random_state=42).to_numpy(), [int(.7*len(df)), int(.85*len(df))]
    )

    X_train, y_train = split_x_y(train)
    X_validate, y_validate = split_x_y(validate)
    X_test, y_test = split_x_y(test)

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    y_validate = to_categorical(y_validate, num_classes=2)

    X_train_n = normalize(X_train)
    X_validate_n = normalize(X_validate)
    X_test_n = normalize(X_test)

    return (X_train_n, y_train), (X_validate_n, y_validate), (X_test_n, y_test)

def build_model():
    model = keras.Sequential()
    model.add(layers.Dense(16, input_dim = 20, activation= 'relu'))
    model.add(layers.Dense(32, activation = 'relu'))
    model.add(layers.Dense(2, activation = 'sigmoid'))
    #model.summary()
    return model

def normalize(X):
    return (X-np.min(X))/(np.max(X)-np.min(X))

def run_model():
    df = load_dataset("weatherAUS.csv")
    df = clean_data(df)
    (X_train, y_train), (X_validate, y_validate), (X_test, y_test) = split_data(df)
    model = build_model()

    model.compile(
        loss="binary_crossentropy", 
        optimizer="adam", 
        metrics=["accuracy"]
    )
    model.fit(X_train, y_train,epochs = 15, batch_size = 64, validation_data=(X_validate, y_validate))
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

run_model()
