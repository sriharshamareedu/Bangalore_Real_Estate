import pickle
import json
import numpy as np


__locations = None
__data_columns = None
__model = None

def predict_home_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)

def get_location_names():
    return __locations


def get_data_columns():
    return __data_columns

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __data_columns
    global __locations

    with open("./artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

    global __model
    with open("./artifacts/bangalore_home_prices_model.pickle", 'rb') as f:
        __model = pickle.load(f)
    print("loading saved artifacts...done")


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(predict_home_price('1st Phase JP Nagar', 1000, 3, 3))
    print(predict_home_price('1st Phase JP Nagar', 1000, 2, 2))
    print(predict_home_price('9th phase jp nagar', 1000, 2, 2)) # other location
    print(predict_home_price('5th block hbr layout', 1000, 2, 2))  # other location
