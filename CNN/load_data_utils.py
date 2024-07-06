from pandas import read_csv, concat, merge, DataFrame
import numpy as np
from math import ceil

def load_to_df(input_filenames, output_filename, standardized_delta = None, prefix = ''):
    # Load in the sensor data
    dfs = []
    for filename in input_filenames:
        df = read_csv(prefix + filename)
        dfs.append(df)
        
    input_data = concat(dfs)

    '''
    STANDARDIZE INPUT DATA
    Sometimes some timestamps are missing so we want to fill in missing 
    timestamps to make sure maintain a standard number of samples per minute
    '''

    # Sets a standard delta between sensors so we can upsample UWB
    if standardized_delta:
        delta = standardized_delta
    else:
        # Get start and end timestamps
        start_timestamp = input_data.iloc[0]['timestamp']
        end_timestamp = input_data.iloc[-1]['timestamp']
        delta = input_data.iloc[1]['timestamp'] - input_data.iloc[0]['timestamp']

    timestamp_range = np.arange(start_timestamp,end_timestamp+delta,delta)
    timestamp_range = np.round(timestamp_range,1)
    # Add existing data to the full df
    standardized_sensor_data = DataFrame(timestamp_range,columns=['timestamp'])
    standardized_sensor_data = merge(standardized_sensor_data,input_data, how='outer', on='timestamp')
    # Fill the data with ffil
    standardized_sensor_data.ffill(inplace=True)

    # Load associated behavior labels
    behavior_labels = read_csv(prefix + output_filename)

    return standardized_sensor_data, behavior_labels


'''
window size is given in minutes
stride is given in minutes

After hyperparameter tuning, defualt sizes are the best tested
'''
def create_rolling_window_data(sensor_data_df, behavior_labels_df, window_size = 30, stride = 5):

    # Get base time difference size
    # Use 3 and 2 in case there is a problem with the first index
    groundtruth_base_time = behavior_labels_df['timestamp'][4] - behavior_labels_df['timestamp'][3]
    # print("Base time is: " + str(groundtruth_base_time))
    input_base_time = sensor_data_df['timestamp'][4] - sensor_data_df['timestamp'][3]

    # Make the base time 5 minutes to make processing much faster
    stride_time = groundtruth_base_time * stride
    window = groundtruth_base_time * window_size
    # Get grountruths in time window
    X,y = [], []

    for start_time in np.arange(behavior_labels_df.iloc[0]['timestamp'],behavior_labels_df.iloc[-1]['timestamp'], stride_time):
        end_time = start_time + window
        # Groundtruth is given in one minute time windows, so split input data every minute
        behavior_labels_data_for_time_window = behavior_labels_df.loc[(behavior_labels_df['timestamp'] >= start_time) & (
            behavior_labels_df['timestamp'] < end_time)]
        
        # Check to make sure this isn't a transition period
        # also get rid of unknown behavior (0)
        a = behavior_labels_data_for_time_window["behavior"].to_list()
        if not (len(set(a)) == 1) or a[0] == 0:
            continue

        # Get associated sensor data
        input_data_for_time_window = sensor_data_df.loc[(sensor_data_df['timestamp'] >= start_time) & (
            sensor_data_df['timestamp'] < end_time)]
        
        # Drop columns that we don't want to use
        unused_features = ['timestamp','mag_x_uT','mag_y_uT','mag_z_uT','pressure_Pa','elevation'] #,'coord_x_cm', 'coord_y_cm']
        unused_cols = input_data_for_time_window.columns.intersection(unused_features)
        input_data_for_time_window = input_data_for_time_window.drop(columns=unused_cols)
        
        # Get rid of empty windows
        if len(input_data_for_time_window) == 0:
            continue

        # Make sure we have consistent shape (Standardize to 600 readings per window)
        sensor_data_list = input_data_for_time_window
        expected_readings = int(ceil(window/input_base_time))

        if len(sensor_data_list) != expected_readings:
            continue
        
        # Add X data
        X.append(sensor_data_list)

        # Add y data
        y.append(a[0])
        

    return np.array(X),np.array(y)