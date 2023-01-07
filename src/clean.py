"""Clean and reshape incoming data"""

import os
import numpy as np
import pandas as pd

def create_demand_df(df):
    # convert date to pandas datetime object
    df["date_time"] = pd.to_datetime(df['date_time'], format='%Y-%m-%d %H:%M:%S')

    # create separate date and hour columns to aid with pivoting
    df["date"] = df["date_time"].dt.date  # Add separate date string 
    df["hour"] = df["date_time"].dt.hour  # Add separate hour string

    # Drop all columns except "Cleaned Demand (MW)"
    df_dropped = df.drop(["raw demand (MW)", "category", "forecast demand (MW)", "date_time"], axis=1)
    df_demand = df_dropped.pivot(index = "date", columns = "hour", values = "cleaned demand (MW)")

    df_demand.index = pd.to_datetime(df_demand.index)

    # return the pivoted df
    return df_demand

def create_temp_df(df):
    # convert date to pandas datetime object
    df.columns = ['date_time', 'Temperature (K)']
    df["date_time"] = pd.to_datetime(df['date_time'])

    # create separate date and hour columns to aid with pivoting
    df["date"] = df["date_time"].dt.date  # Add separate date string 
    df["hour"] = df["date_time"].dt.hour  # Add separate hour string

    df_temp = pd.DataFrame()

    temp_max = df.groupby('date')['Temperature (K)'].agg(np.max)

    df_temp['Temperature (F)'] = (1.8 * (temp_max - 273.15)) + 32

    print(df_temp)

    return df_temp



def clean_data(input_path, output_path):
    # clean all demand and temp files in input directory
    demand_path = os.path.join(input_path, "demand")
    for filename in os.listdir(demand_path):
        # only concerned with CSVs here
        if not filename.endswith('csv'): continue
        file_path = os.path.join(demand_path, filename)
        name = filename.split('.')[0] + "_demand.csv"
        output_name = os.path.join(output_path, name)

        # create out directory 
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # read in raw data to a dataframe
        raw_data = pd.read_csv(file_path)

        # create the df_demand pandas DataFrame with hour as column, row as date, and value as demand
        df_demand = create_demand_df(raw_data)
        df_demand.to_csv(output_name)

    if 'temp' in os.listdir(input_path):
        temp_path = os.path.join(input_path, "temp")
        for filename in os.listdir(temp_path):
            # only concerned with CSVs here
            if not filename.endswith('csv'): continue
            file_path = os.path.join(temp_path, filename)
            name = filename.split('.')[0] + "_temp.csv"
            output_name = os.path.join(output_path, name)
            
            # create out directory 
            if not os.path.exists(output_path):
                os.mkdir(output_path)

            # read in raw data to a dataframe
            raw_data = pd.read_csv(file_path)
    
            # create the df_demand pandas DataFrame with hour as column, row as date, and value as demand
            df_demand = create_temp_df(raw_data)
            df_demand.to_csv(output_name)

