#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import csv
import numpy as np


# Process data
def processAirfoilData(directory='data'):
    '''
    Get airfoil shape from preprocessed numpy files and CFD
    information from webscraped csv files.
    Files are retrieved from the airfoil-data directory by default.
    Output is a pandas dataframe containing the following information:
    - Airfoil name
    - Airfoil shape
    - Angle of attack
    - Reynolds number
    - N_crit
    - C_l
    - C_d
    - C_m
    '''
    # Initialize lists
    aoa_list=[]
    airfoil_list=[]
    Re_list=[]
    Cl_list=[]
    Cd_list=[]
    ClCd_list=[]
    afdir_list=[]

    # Create a counter
    i=0

    # Go through files
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath=os.path.join(root, file)

            # Get airfoil directory to retrieve numpy files later
            afdir=root.replace(directory+'/', '') # Get folder name to get airfoil shape numpy file name

            # Get data from csv files
            if 'csv' in file and '-n5.' not in file and 'airfoil_' not in file:
                with open(filepath) as f:

                    # Read csv
                    r=csv.reader(f)
                    data=list(r)

                    # Get general airfoil information
                    airfoil_name=data[2][1]
                    Re=data[3][1]

                    # For each angle of attack, append airfoil name, Reynolds number, Ncrit,
                    # and corresponding force coefficients to respective lists
                    for row in range(11, len(data)):
                        aoa_list.append(data[row][0])
                        Cl_list.append(data[row][1])
                        Cd_list.append(data[row][2])
                        ClCd_list.append(float(data[row][1])/float(data[row][2]))
                        airfoil_list.append(airfoil_name)
                        Re_list.append(Re)
                        afdir_list.append(afdir)

                # Increment counter and print file number every 1000 files
                i+=1
                if i%1000==0:
                    print(i, 'files processed.')

    # Notify that all files have been processed
    print('All',i,'files processed!')
    print('Creating dataframe...')

    # Initialize dataframe
    airfoil_df=pd.DataFrame(columns=['Name','Directory','AoA','Re','Cl','Cd','Cl/Cd'])

    # Add data lists to dataframe
    airfoil_df['Name']=airfoil_list
    airfoil_df['Directory']=afdir_list
    airfoil_df['AoA']=aoa_list
    airfoil_df['Re']=Re_list
    airfoil_df['Cl']=Cl_list
    airfoil_df['Cd']=Cd_list
    airfoil_df['Cl/Cd']=ClCd_list

    # Notify that dataframe has been created
    print('Dataframe created!')

    return airfoil_df


# Fix data types in dataframe
def fixDfDtypes(airfoil_df, datatypes=['str', 'str', 'float', 'float', 'float', 'float', 'float']):
    # Fix data types
    airfoil_df['Name']=airfoil_df['Name'].astype(datatypes[0])
    airfoil_df['Directory']=airfoil_df['Directory'].astype(datatypes[1])
    airfoil_df['AoA']=airfoil_df['AoA'].astype(datatypes[2])
    airfoil_df['Re']=airfoil_df['Re'].astype(datatypes[3])
    airfoil_df['Cl']=airfoil_df['Cl'].astype(datatypes[4])
    airfoil_df['Cd']=airfoil_df['Cd'].astype(datatypes[5])
    airfoil_df['Cl/Cd']=airfoil_df['Cl/Cd'].astype(datatypes[6])

    return airfoil_df


def normalizeData(csv_file):
    norm_csv_file=csv_file.replace('.csv', '')+'_normalized.csv'
    mstd_csv_file=csv_file.replace('.csv', '')+'_mean_std.csv'

    df=pd.read_csv(csv_file).drop('Unnamed: 0', axis=1)
    variables=['AoA','Re','Cl','Cd','Cl/Cd']
    mean=df.loc[:, variables].mean()
    std=df.loc[:, variables].std()
    df.loc[:, variables]=(df.loc[:, variables]-mean)/std
    df.to_csv(norm_csv_file)

    df=pd.DataFrame({'mean':mean, 'std':std})
    df.to_csv(mstd_csv_file)


# Run process airfoil data function
airfoil_df=processAirfoilData()

# Run fix data types function
airfoil_df=fixDfDtypes(airfoil_df)
airfoil_df.dtypes

# Save dataframe
airfoil_df.to_csv('processed_data/airfoil_data.csv')

# Create normalized data csv and save mean and standard deviation values
normalizeData('processed_data/airfoil_data.csv')
