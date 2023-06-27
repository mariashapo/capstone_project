import pandas as pd
import numpy as np
import re

#HIGH LEVEL OVERVIEWS

def overview (df):
    '''
    Use:
        Provides a high-level overview of the input dataframe.
    Input:
        df [pandas.core.frame.DataFrame] : df to be analysed
    Return:
        - Print the shape of the input dataframe
        - pandas.core.frame.DataFrame : df containing overview of the column data types, null values, and sample rows

    '''
    #print df shape
    print(f'The dataframe shape is {df.shape}')
    # Create new dataframe with data types, null values, and sample row information
    preview_df=pd.DataFrame({
        #data types for each columns
        'Data Types':df.dtypes,\
        #absolute number of null values in each column
        'Total Null Values':df.isna().sum(), \
        #percentage number of null values in each column
        'Null Values Percentage':df.isna().sum()*100/df.shape[0],\
        #first row
        'Sample Value Head':df.head(1).T.iloc[:,0],\
        #last row
        'Sample Value Tail':df.tail(1).T.iloc[:,0],\
        #random sample row
        'Sample Value':df.sample().T.iloc[:,0]})

    # Rename index axis
    preview_df.rename_axis('Column_Name',inplace=True)
    return preview_df


#

def get_columns_with_regex(df,reg_exp=None,return_ct=False):
    '''
    Use:
        Returns a filtered dataframe that contains columns with names that match the given regular expression.
    Input:
        df (pandas.core.frame.DataFrame): The dataframe to be filtered.
        reg_exp (str): The regular expression to match against column names. Default is None
        return_ct(bol): Specify if census tract column should be returned, Default is False
    Return:
        pandas.core.frame.DataFrame: A new dataframe containing only columns that match the regular expression and census tract column if return_ct set to True
    
    '''

    try:
        if reg_exp is None:
            df_result=df
        else:
            filtered_df = df.loc[:, [bool(re.search(reg_exp, col)) for col in df.columns]]
            df_result=filtered_df

    except re.error:
        raise ValueError(f"{reg_exp} is not a valid regular expression")
    
    if return_ct==True:
        if 'CENSUS_TRACT' in df.columns:
            df_result=pd.concat([df_result,df['CENSUS_TRACT']],axis=1)
        elif 'Census_Tract' in df.columns:
            df_result=pd.concat([df_result,df['Census_Tract']],axis=1)

    return df_result
    

### INCOME EDA

#CLEANING
def census_column_selector(df,di,remove_first_row=False):
    '''
    Use:
        This function selects columns speicified by the dictory keys and renames the columns to the dictionary values

    Input:
        df (pandas.DataFrame): The input DataFrame to select columns from.
        col_di (dict): A dictionary with column names as keys and new names as values.
        remove_first_row (bool): Optional flag to remove the first row of the DataFrame. Default is False.
        
    Returns:
        pandas.DataFrame: A new DataFrame containing only the selected columns and renamed columns.    
    '''

    #copy dataframe to avoid overwritting the original
    df_temp=df.copy()

    #remove first row if required
    if remove_first_row==True:
        df_temp=df_temp.iloc[1:,:]

    # extract last 6 characters of GEO_ID column (last 6 char=census tract number)
    df_temp['GEO_ID']=df_temp['GEO_ID'].str[-6:]

    #select columns using dictionary 'di' keys and rename using the same dictionary
    df_temp=df_temp[di.keys()].rename(columns=di)

    # convert selected columns to numeric data type
    df_temp = df_temp.apply(pd.to_numeric, errors='coerce')
    # if data is missing or non-numeric, (errors='coerce') converts it to nan

    # convert 0 and 0.0 values to NaNs
    df_temp.replace({0.0:np.nan,0:np.nan},inplace=True)

    return df_temp



def census_tract_cleaning(df,di_conversion,relevant='all'):

    '''
    Use:
        Clean Census Income Dataset: replace ct numbers, merge rows with the same ct and select relevant rows
    Inputs:
        df (pandas.DataFrame): The input DataFrame to clean
        di_conversion (dict): A dictionary to update ct numbers
        relevant (list) : A list of planning application ct numbers
    Returns:
        pandas.DataFrame: A new cleaned DataFrame
    '''

    #replace census tracts
    df['Census_Tract']=df['Census_Tract'].replace(di_conversion)

    count_col=[col for col in df.columns if 'Count' in col]

    #merge rows with the same ct
    df=pd.concat([
        #for the same ct, average all columns
        df.drop(columns=count_col).groupby('Census_Tract').mean(),\
        #for the same ct, sum household counts
        df.groupby('Census_Tract')[count_col].sum()]\
        #reset index to set ct back as a column
        ,axis=1).reset_index()

    if relevant!='all':
        #select only rows in a list `relevant` & reset index
        df=df[df['Census_Tract'].isin(relevant)].reset_index(drop=True)

    #replace any zeros with nan
    df=df.replace({0:np.nan,0.0:np.nan})

    return df


#CALCULATINF DISTNACE BETWEEN TRACTS, FINDING THE CLOSEST CENSUS TRACT AND MERGING TRACTS TOGETHER

import math

def distance(lat1, lon1, lat2, lon2):
    '''
    Use:
        This function calculates the distance (in kilometers) between two geographic points given their respective 
        latitude and longitude coordinates.

    Inputs:
        lat1 (float): Latitude of the first point
        lon1 (float): Longitude of the first point
        lat2 (float): Latitude of the second point
        lon2 (float): Longitude of the second point

    Return:
        float: The distance between the two points in kilometers, rounded to 4 decimal places

    '''
    R = 6371  # radius of the Earth in kilometers
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Calculate the difference in latitude and longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Apply the Haversine formula
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    # Round the result to 4 decimal places and return
    return round(distance,4)


#find a census tract closest to the input tract
def closest_tract_search(tract,df):

    '''
    Use:
        This function takes in a census tract ID and a pandas DataFrame containing the latitude and longitude 
        coordinates of all census tracts. This function finds the census tract that is closest to the input tract from the 
        based on all the latitude and longitude coordinates in the dataframe.
    
    Inputs:
        tract (int): Input census tract
        df (pandas DataFrame): DataFrame containing the latitude and longitude coordinates of census tracts to be searched from
        #note df should only contain distinct census tracts
    
    Return:
        closest_tract (int): Census tract ID of the closest tract to the input tract
    '''

    # Get the latitude and longitude coordinates of the input tract
    lat_base=df[df['CENSUS_TRACT']==tract]['LATITUDE']
    lon_base=df[df['CENSUS_TRACT']==tract]['LONGITUDE']

    # Set initial minimum distance to inf
    min_dist=float('inf')

    #Iterate over all census tracts and calculate all distances
    for tract_search in df['CENSUS_TRACT']:
        # Skip the input tract
        if tract_search==tract:
            continue
        
        # Get the lat and long of the tract being compared
        lat_search=df[df['CENSUS_TRACT']==tract_search]['LATITUDE']
        lon_search=df[df['CENSUS_TRACT']==tract_search]['LONGITUDE']

        # Calculate the distance between the two tracts
        dist=distance(lat_base, lon_base, lat_search, lon_search)
        # if a closer tract is closer update the minimum distance and closest tract ID 
        if dist<min_dist:
            min_dist=dist
            closest_tract=tract_search

    return closest_tract



#Update census tract and average if two previously distinct tracts were merged into one
#filter on census tracts present in the Permit Dataset

def small_tracts_merge(df,dictionary):
    '''
    Use:
        This function updates census tracts and averages Income values, but sums Household_Count if two previously distinct tracts were merged into one.

    Inputs:
        df (pandas DataFrame): input dataset containing Census_Tract, Income, and Household_Count columns
        dictionary (dict): a dictionary with information to merge census tracts (old tract : new tract)
    
    Return:
        df (pandas DataFrame): merged dataset with updated census tract numbers and averaged Income values and summed Household_Counts.
        

    '''
    #copy input to avoid overwriting the input
    df_temp=df.copy()

    # Replace old census tract numbers with new ones using the input dictionary
    df_temp['Census_Tract']=df_temp['Census_Tract'].replace(dictionary)

    #Average Income values, but sum Household_Count
    df_temp=pd.concat([df_temp.groupby('Census_Tract').mean().drop(columns='Household_Count'),df_temp.groupby('Census_Tract')['Household_Count'].sum()],axis=1).reset_index()

    return df_temp



#COMBINED INCOME CHANGE METRIC TO EVALUATE GENTRIFICATION

#LOG

#THIS CAN BE DELETED LATER AND REPLACED WITH THE FUNCTION AFTER LATER ON
def comb_change_calc(start,finish):
    '''
    Use
        Calculates the combined change metric between incomes from two years using the formula: 
        log(abs(abs_change))*(perc_change)
    
    Inputs:
        start (float or int): The initial value (income for the start year - year the prediction is made FROM)
        finish (float or int): The final value (income for the final year - year the prediction is made FOR)
    
    Returns:
        x (float): The combined change metric for two incomes.
    '''
    perc_change=(finish-start)/start
    abs_chage=abs(finish-start)

    if perc_change ==0:
        perc_change=1e-10
    if abs_chage ==0:
        abs_chage=1e-10

    x=np.log(abs_chage)*(perc_change)

    return x

#now let's define a function that selects the correct columns from the whole dataframe and applies combined change calculatation for specific years

#POWER

def comb_change_calc_power(start,finish,perc_power):
    '''
    Use
        Calculates the combined change metric between incomes from two years using the formula: 
        (abs_change)*(perc_change)^perc_power
    
    Inputs:
        start (float or int): The initial value (income for the start year - year the prediction is made FROM)
        finish (float or int): The final value (income for the final year - year the prediction is made FOR)
    
    Returns:
        x (float): The combined change metric for two incomes.
    '''
    perc_change=finish/start
    abs_chage=abs(finish-start)

    x=(abs_chage)*(perc_change)**(perc_power)

    return x

#now let's define a function that selects the correct columns from the whole dataframe and applies combined change calculatation for specific years

def comb_change(df,year_start,n,mode='log',perc_power=2):

    '''
    Use:
        Calculate the combined change of income metric for specific years.
    Inputs:
        df (DataFrame): DataFrame containing Mean_Income data.
        year_start (int): Current year (the year we are making the prediction from).
        n (int): The number of years ahead we are making the prediction for.
        mode (str): ['log','power'] (default: 'log')
        perc_power (int): the power to raise the percentage change to 
    Returns:
        A list of combined change metric of income values.
    '''

    #select income values from df for year_start (current year) by selecting the Mean_Income column for the given year
    str_current='Mean_Income_'+str(year_start)
    income_val_current=list(df[str_current])

    #year_final for which we are making the prediction (future year)

    #find the final year
    year_final=year_start+n

    #select the column containing Mean Income for the final year
    str_final='Mean_Income_'+str(year_final)
    income_val_final=list(df[str_final])

    comb_change_li=[]
    if mode=='log':
        for (a,b) in zip(income_val_current,income_val_final):
            comb_change_li.append(comb_change_calc(a,b))
    elif mode=='power':
        for (a,b) in zip(income_val_current,income_val_final):
            comb_change_li.append(comb_change_calc_power(a,b,perc_power))

    return comb_change_li

#####

def change_plot(df,year_start,year_end,mode='absolute',ranking='max',reg_ex='Mean',perc_power=2):
    '''
    Use:
        This function plots the change in income over a period between 2 years, and filters for census tracts that meet certain criteria.
    
    Inputs:
        df: dataframe containing income data
        year_start: start year for the income comparison
        year_end: end year for the income comparison
        mode: type of income change to evaluate ['absolute','percentage','combined_log','combined_power'] (default: 'absolute')
        ranking: specifies whether to filter the top or bottom census tracts ['min',''max'] (default: 'max')
        reg_ex: regular expression to use for column selection ['Mean','Median'], (default: 'Mean')
        perc_power: power to raise the percentage difference to - only for mode=='combined_power' (default: 2)

    
    Output:
        A line plot showing the change in income for census tracts that meet the specified criteria, as well as the mean income over the years.
    '''
    #DATA PREPARATION

    df_copy=df.copy()
    #set Census_Tract as index so that census tracts appear in the legend box
    df_copy=df_copy.set_index('Census_Tract')

    #selecting mean or median columns using the custom function defined previously
    df_temp=get_columns_with_regex(df_copy,reg_ex)

    #convert year to column position
    start_pos=year_start-2010
    end_pos=year_end-2010

    #rename columns to only include years
    df_temp.columns=[(i[-4::]) for i in df_temp.columns]

    #compute the difference basen on the mode specified
    if mode =='absolute':
        df_diff=df_temp.iloc[:,end_pos]-df_temp.iloc[:,start_pos]
    elif mode =='percentage':
        df_diff=((df_temp.iloc[:,end_pos]-df_temp.iloc[:,start_pos])/df_temp.iloc[:,start_pos])
    #combined metric takes into account both absolute and percentage change in income, which helps to ensure that tracts of different income levels are considered
    elif mode=='combined_log':
        df_diff=np.log(abs(df_temp.iloc[:,end_pos]-df_temp.iloc[:,start_pos]))*((df_temp.iloc[:,end_pos]-df_temp.iloc[:,start_pos])/df_temp.iloc[:,start_pos])
        #log of absolute change is taken to reduce its impact due to absolute change being a greater number than percentage change
    elif mode=='combined_power':
        df_diff=(abs(df_temp.iloc[:,end_pos]-df_temp.iloc[:,start_pos]))*((df_temp.iloc[:,end_pos]-df_temp.iloc[:,start_pos])/df_temp.iloc[:,start_pos])**perc_power
        #raising percentage change to the power of 2 

    #select if display top 10 maximum and minimal growth
    if ranking=='max':
        order_state=False
    elif ranking=='min':
        order_state=True

    #filter the tracts for which the specified type of growth occured and select TOP 10
    filtered_tracts=df_diff.sort_values(ascending=order_state).iloc[:10].index

    #Plot every 5th census tract in the dataset with low transparency
    sns.lineplot(data=df_temp.iloc[range(0,df_temp.shape[0],5)].T,dashes=False,alpha=0.1,legend=False)

    #Plot TOP 10 in full colours
    sns.lineplot(data=df_temp[df_temp.index.isin(filtered_tracts)].T,dashes=False,alpha=0.9)

    # Plot the mean income over the years
    sns.lineplot(data=pd.DataFrame(df_temp.mean(axis=0).T.rename('mean')),markers=True)
    #sns.lineplot(data=pd.DataFrame(df_temp.median(axis=0).T.rename('median')),markers=True)

    plt.xlabel('year')
    plt.ylabel('Income per census tract')


#####

#define a function to Replaces all zeros in a DataFrame
def replace_zeros(dataframe,skip_columns=['CENSUS_TRACT']):
    """
    Use:
        Replaces all zeros in a DataFrame with a very small non-zero value.

    Inputs:
        dataframe (pandas.DataFrame): The input DataFrame.
        skip_columns (list): List of columns to skip, default = ['CENSUS_TRACT']
    Returns:
        pandas.DataFrame: The modified DataFrame.
    """
    
    small_value=1e-10

    # Create a copy to avoid modifying the original
    df = dataframe.copy()

    # Iterate over each column in the DataFrame
    for col in df.columns:

        # Check if the column has numeric data and is not a column to skip
        if pd.api.types.is_numeric_dtype(df[col]) and col not in skip_columns:

            # Check if there are any zeros in the column
            if (df[col] == 0).any():

                # Replace the zeros with the small_value
                df[col] = df[col].replace(0, small_value)

    return df






def pipe_to_df(X,pipe_obj,model_ref='model'):
    '''
    X(pandas.DataFrame) : dataframe used for modelling
    pipe_obj(pipeline obkect) : pipeline with the model inside it 
    model_ref(str) : name of the model perameter in the pipeline

    '''
    #obtain the model from the pipeline 
    model_temp=pipe_obj.named_steps[model_ref]

    #save model coefficients
    coefficients = model_temp.coef_[0]

    #extract column names from X and create a dataframe combining feature names and coefficients
    df_result=pd.DataFrame({'feature':X.columns,'coefficients':coefficients})

    return df_result

#Define a function to add census tract latitude and longitude coordinates back

def coord(df, df_coord):
    """
    Use:
        This merges input the first dataframe, df, with the second, df_coord, adding coordinates for each Census Tract. 
    
    Inputs:
        df (pandas.DataFrame): The DataFrame containing Census Tract information to be merged with coordinates.
        df_coord (pandas.DataFrame): The DataFrame containing the coordinates for each Census Tract.
    
    Returns:
        df_result (pandas.DataFrame): A new DataFrame with Census Tract information and corresponding coordinates. 
    """
    df_result=df.copy()
    df_result=df.merge(df_coord, how='left', left_on='Census_Tract', right_on='CENSUS_TRACT')
    return df_result


## GROUPING BY CENSUS TRACT AND YEAR

def df_window_multi_type(df,year,t):

    '''
    Combines descriptions & numeric data
    Inputs:
    #df_description to only contain description columns
    '''
    #copy dataframe to avoid accidental overwriting
    df_temp=df.copy()

    if 'YEAR' not in df.columns:
        df_temp['YEAR']=df['ISSUE_DATE'].dt.year
        df_temp=df_temp.drop(columns='ISSUE_DATE')

    assert df_temp.index.is_monotonic_increasing, 'Check Indexing: Should be a simple arithmetic sequence'

    df_temp=df_temp.set_index(['Census_Tract','YEAR'])

    #to set 'YEAR' index as a column
    df_temp=df_temp.reset_index(level=1)

    #select relevant years
    #year+1 as the range end to esnure data for the current year is also included
    df_temp=df_temp[df_temp['YEAR'].isin(range(year-t,year+1))].drop(columns='YEAR')

    #instantiate the output dataframe
    df_result=pd.DataFrame()

    #select columns with distriptions
    obj_cols = df_temp.select_dtypes(include=['object']).columns
    num_cols = df_temp.select_dtypes(include=['number']).columns

    ### CAN ADD MORE FEATURES HERE ###

    #Taking averages for numeric columns
    for col in num_cols:
        df_result[col]=df_temp.groupby(level=0)[col].mean()

    #Concatenating qualitative columns
    for col in obj_cols:
        #need to keep in mind that some stings (descriptions) might be missing
        df_result[col]=df_temp.groupby(level=0)[col].apply(lambda x: ' '.join(str(i) for i in x))

    return df_result


## Column Tranformer conducting One Hot Encoding

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class CustomOneHotEncoder_CT(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.encoder = OneHotEncoder(**kwargs)
        self.ohe_col=None
    
    def fit(self, X, y=None):

        if 'WORK_DESCRIPTION' in X.columns:
            ohe_col=X.drop(columns='WORK_DESCRIPTION').select_dtypes(include=['object']).columns
        else: 
            ohe_col=X.select_dtypes(include=['object']).columns


        self.encoder.fit(X[ohe_col])
        self.ohe_col=ohe_col
        return self
    
    def transform(self, X):
        transformed_data = self.encoder.transform(X[self.ohe_col])
        column_names = self.encoder.get_feature_names_out()
        df = pd.DataFrame(transformed_data.toarray(), columns=column_names)
        df=pd.concat([X,df],axis=1)
        df = df.drop(columns=self.ohe_col)
        return df




def sel_col_model(df,reg_exp):
    '''
    This function return a filtered dataframe that contains given (reg_exp) in the column name
    Census_Tract and Year are set as index (if present)
    '''
    temp_df=df.copy()
    #df['Census_Tract']=df['Census_Tract'].astype('int32')        
    #temp_df=df.set_index(['Census_Tract','YEAR'],drop=True)
    new_df=temp_df.loc[:,[bool(re.search(reg_exp,col)) for col in temp_df]]
    if 'Census_Tract' in temp_df.columns:
        new_df=pd.concat([temp_df['Census_Tract'],new_df],axis=1)
    if 'YEAR' in temp_df.columns:
        new_df=pd.concat([temp_df['YEAR'],new_df],axis=1)
    
    return new_df


#OHE

#define a function to apply ohe to a specified list of categorical columns only and returns a new full dataframe

from sklearn.preprocessing import OneHotEncoder

def ohe_cat(df,col_li):
    '''
    Use:
        Apply OHE for the specified columns only
    
    Inputs:
        df (pandas.DataFrame): Input DataFrame
        col_li(list): List of column names to apply OHE
    
    Returns:
        df_result(Pandas DataFrame): Result DataFrame with OHE applied to specified columns'''

    #copy dataframe 
    df_result=df.copy()

    # Iterate through each column in the input list
    for col in col_li:

        #Instantiate the encoder
        enc = OneHotEncoder(sparse=False)

        #Encode and convert it into a new DataFrame with column names including the original column name
        df_temp=pd.DataFrame(enc.fit_transform(df[[col]]),columns=[col+'_'+i for i in enc.categories_[0]])

        # Concatenate the encoded df with the existing DataFrame
        df_result=pd.concat([df_result,df_temp],axis=1)

    # Drop the original categorical columns from the result DataFrame
    df_result=df_result.drop(columns=col_li)

    return df_result






#IGNORE AFTER THIS
##################################################################################


#OLD FUNCTIONS 
def df_window(df,year,t):
    '''
    This function takes the mean of all the X features for the years within the window for each geography (census tract)
    year - current year/ the year we are making the prediction from
    t - number of years in the past'''
    df_X=df[df['YEAR'].isin(range(year-t,year+1))].groupby('Census_Tract').mean()
   
    return df_X

def ohe_cat(df,col_li):
    df_result=df.copy()
    for col in col_li:
        #Instantiate the encoder
        enc = OneHotEncoder(sparse=False)
        #Encode

        df_temp=pd.DataFrame(enc.fit_transform(df[[col]]),columns=[col+'_'+i for i in enc.categories_[0]])
        df_result=pd.concat([df_result,df_temp],axis=1)

    df_result=df_result.drop(columns=col_li)

    return df_result


import random

def test_train(X,y,test_size):
    X_temp=X.copy()
    y_temp=y.copy()

    geo_set=set(X_temp['Census_Tract'])
    test_len=int(len(geo_set)*test_size)

    geo_test=random.sample(list(geo_set),k=test_len)
    test_mask=X_temp['Census_Tract'].isin(geo_test)

    y_test=y[test_mask]
    y_train=y[~test_mask]

    X_test=X_temp[test_mask]
    X_train=X_temp[~test_mask]

    return X_train,X_test,y_train,y_test



def target_class(df,year,p):
    '''
    This function creates the target column and drops all income columns apart from the current year
    '''
    #name of the column containing income for the current year
    current_income_col='Median_Income_'+str(year)

    #the year we are making the prediction for
    prediction_year=year+p

    #name of the column containing income for the year we are making the prediction for
    prediction_income_col='Median_Income_'+str(year)

    abs_diff=prediction_income_col-current_income_col
    perc_diff=prediction_income_col/current_income_col
    

    #generate names of all income columns
    income_col=[col for col in list(df.columns) if bool(re.search('Median',col))]
    #drop all income columns and create new df_out dataframe
    df_temp=df.drop(columns=income_col)
    #add the current income column back
    df_out['Income']=df[current_income_col]

#DELETE DELETE DELETE
def abs_perc_income_change(df,year_x,n):

    '''
    year_x - current year (the year we are making the prediction from)
    n - the number of years ahead we are making the prediction
    '''

    #select year_x (current year) income values from df 
    str_current='Median_Income_'+str(year_x)
    income_val_current=list(df[str_current])

    #year for which we are making the prediction (future year)
    year_future=year_x+n
    str_future='Median_Income_'+str(year_future)
    income_val_future=list(df[str_future])

    abs_perc_change_li=[]
    for (a,b) in zip(income_val_current,income_val_future):
        #abs_perc_change_li.append(abs_perc_change(a,b))
        abs_perc_change_li.append(b-a)

    return abs_perc_change_li
#



def abs_perc_income_change(df,year_x,n):

    '''
    year_x - current year (the year we are making the prediction from)
    n - the number of years ahead we are making the prediction
    '''

    #select year_x (current year) income values from df 
    str_current='Median_Income_'+str(year_x)
    income_val_current=list(df[str_current])

    #year for which we are making the prediction (future year)
    year_future=year_x+n
    str_future='Median_Income_'+str(year_future)
    income_val_future=list(df[str_future])

    abs_perc_change_li=[]
    for (a,b) in zip(income_val_current,income_val_future):
        abs_perc_change_li.append(abs_perc_change(a,b))
        #abs_perc_change_li.append(b-a)


    return abs_perc_change_li



def abs_perc_change(a,b):
    '''a values for the current year = year we are making prediction from
    b values for the future year = year we are making prediction for'''
    change=(b-a)*(b/a)**2
    return round(change,4)