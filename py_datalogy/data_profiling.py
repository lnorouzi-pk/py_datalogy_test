'''




'''

import pandas as pd
from pandas.api.types import *
from dateutil.parser import *
from eda_functions import *

import numpy as np

class DataProfiling(object):
    '''
    :Author: Leila Norouzi
    :Date: 27 Dec 2019
    :Description:

    Attributes
    ----------
    data: pd.DataFrame -- The data to be investigated
    path: str -- The path for saving the results
    numerical_columns: list -- The name of the numeric variables (columns) in the data
    object_columns: list -- The name of the non-numeric variables (columns) in the data
    datetime_columns: list -- The name of the date-time type variables (columns) in the data
    string_columns: list -- The name of the string type variables (columns) in the data
    numeric_data: pd.DataFrame -- The numeric variables (columns) of the data
    string_data: pd.DataFrame -- The string type variables (columns) of the data
    datetime_data: pd.DataFrame -- The date-time type variables (columns) of the data
    non_numeric_data: pd.DataFrame -- The non-numeric variables (columns) of the data
    numeric_profile_df: pd.DataFrame -- The data profile for numeric variables (columns)
    non_numeric_profile_df: pd.DataFrame -- The data profile for non-numeric variables (columns)
    total_profile_df: pd.DataFrame -- The data profile for all variables (columns)


    Methods
    --------
    datetime_identification: This functions looks at the non-numeric columns od the data and search if they have any
        date/time format. Then returns the name of those columns.
    numeric_profile: This function takes the numeric variables of the data and for each variable generates a basic data
        description such as
            - The type of the variable as numeric, the size of the data,
            - The max, min, mean, median, standard deviation  and the range of the variable,
            - Number of nulls, non-nulls and percentage of null values,
            - The unique values , number of unique values and their frequency counts
            - The longest and shortest length of the string version of values and their values, and the shortest length
             of the non-null value for each variable.
            - The top frequent value and the count is set to NA as they are set for non-numeric variables

    non_numeric_profile: This function takes the non-numeric variables of the data and for each variable generates a
        basic data description such as
            - The type of the variable as non-numeric, the size of the data,
            - The max and min values as teh first  and the last values when the variable is sorted
            - The median, standard deviation  and the range of the variable are set to NA
            - Number of nulls, non-nulls and percentage of null values,
            - The unique values , number of unique values and their frequency counts
            - The longest and shortest length of values and their values, and the shortest length of the non-null
              value for each variable.
            - The top most frequent value and the count
    '''

    def __init__(self, data:pd.DataFrame, path=''):
        """
        :Description: This functions looks at the non-numeric columns od the data and search if they have any
            date/time format. Then returns the name of those columns.

        :param data: The data to be investigated.
        :type data: pandas dataframe
        :param path: the path to save the plot in.
        :type path: str

        """

        self.data = data.copy(deep=True)
        self.path = path
        # self.features =list(data.columns)

        # Identifying the columns with numerical type
        self.numerical_columns = [cols for cols in data.columns
                                   if is_numeric_dtype(data[cols])
                                   and len(data[cols].dropna()) > 0
                                   and sum(data[cols].notnull()) > 0
                                   ]
        #Identifying the columns with object type
        self.object_columns = [cols for cols in data.columns
                                   if is_object_dtype(data[cols])
                                   and len(data[cols].dropna()) > 0
                                   and sum(data[cols].notnull()) > 0
                                   ]

        self.datetime_columns = self.datetime_identification()
        self.string_columns = list(
            filter(
                lambda x: x not in set(self.datetime_identification()), self.object_columns
            ))

        self.numeric_data = data[self.numerical_columns]
        self.string_data = data[self.string_columns]
        self.datetime_data = data[self.datetime_columns].applymap(lambda x: parse(x) if not x is np.nan else x )
        self.non_numeric_data = pd.concat([self.datetime_data, self.string_data], axis=1)

        self.numeric_profile_df = self.numeric_profile()
        self.non_numeric_profile_df = self.non_numeric_profile()
        self.total_profile_df = pd.concat([self.numeric_profile_df,self.non_numeric_profile_df], axis=1)



    def datetime_identification(self) -> list:
        '''
        :Description: This functions looks at the non-numeric columns od the data and search if they have any
        date/time format. Then returns the name of those columns.

        :return :  list --  date-time column names
        '''
        res=[]
        for x in self.object_columns:
            df = self.data[x].dropna()

            # to check if the content of teh column is date-time type or not
            try:
                parse(df[0])
                res = res + [x]
            except:
                pass

        return res


    def numeric_profile(self) -> pd.DataFrame:
        '''
        :Description: This function takes the numeric variables of the data and for each variable generates a basic data
        description such as
            - The type of the variable as numeric, the size of the data,
            - The max, min, mean, median, standard deviation  and the range of the variable,
            - Number of nulls, non-nulls and percentage of null values,
            - The unique values , number of unique values and their frequency counts
            - The longest and shortest length of the string version of values and their values, and the shortest length
             of the non-null value for each variable.
            - The top frequent value and the count is set to NA as they are set for non-numeric variables

        :return: pd.DataFrame --  A descriptive information about numerical variables of the data
        '''

        #Making a profile data frame
        prof = pd.DataFrame()

        for col in self.numerical_columns:
            df = self.data.loc[:,col]
            prof.loc['Type', col] = df.get_dtype_counts().index.to_list()
            prof.loc['Size', col] = len(df)
            prof.loc['MaxValue',col] = df.max()
            prof.loc['MinValue', col] = df.min()
            prof.loc['Top', col] = 'NA'
            prof.loc['Freq', col] = 'NA'
            prof.loc['MeanValue', col] = round(df.mean(),2)
            prof.loc['MedianValue', col] = round(df.median(),2)
            prof.loc['ModeValue', col] = df.mode()[0]
            prof.loc['StdValue', col] = round(np.std(df),2)
            prof.loc['Range', col] = df.max() - df.min()
            prof.loc['NumNull', col] = sum(df.isna() | df.isnull())
            prof.loc['NonNull', col] = \
                round(sum(df.isna() | df.isnull())*100/len(df),2)
            prof.loc['PercentNull', col] = df.count()

            # Calculate the 25, 75 and 95 percentiles of the values
            # prof.loc['q25', col] = np.percentile(df.dropna(), 25)
            # prof.loc['q75', col] = np.percentile(df.dropna(), 75)
            # prof.loc['q95', col] = np.percentile(df.dropna(), 95)

            prof.loc['UniqueValues', col] = df.unique()
            prof.loc['NumUnique', col] = df.nunique()

            hist = pd.DataFrame(pd.Series(df.value_counts()).reset_index().values, columns=['name', 'num'])
            prof.loc['ValueFreq',col] = hist.values
            prof.loc['LongestLength', col] = df.astype('str').str.len().max()
            prof.loc['LongestLengthVal', col] = df.astype('str').str.len().idxmax()
            prof.loc['ShortestLength', col] = df.astype('str').str.len().min()
            prof.loc['ShortestLengthVal', col] = df.astype('str').str.len().idxmin()
            prof.loc['ShortestLengthNonNullVal', col] = df.dropna().astype('str').str.len().idxmin()

        return prof

    def non_numeric_profile(self) -> pd.DataFrame:
        '''
        :Description: This function takes the non-numeric variables of the data and for each variable generates a
        basic data description such as
            - The type of the variable as non-numeric, the size of the data,
            - The max and min values as teh first  and the last values when the variable is sorted
            - The median, standard deviation  and the range of the variable are set to NA
            - Number of nulls, non-nulls and percentage of null values,
            - The unique values , number of unique values and their frequency counts
            - The longest and shortest length of values and their values, and the shortest length of the non-null
              value for each variable.
            - The top most frequent value and the count

        :return: pd.DataFrame -- A descriptive information about non-numerical variables of the data
        '''

        #Making a profile data frame
        prof = pd.DataFrame()

        for col in self.object_columns:
            df = self.data.loc[:,col]
            prof.loc['Type', col] = df.get_dtype_counts().index.to_list()
            prof.loc['Size', col] = len(df)
            prof.loc['MaxValue',col] = df.sort_values(ascending=False).reset_index(drop=True)[0]
            prof.loc['MinValue', col] = df.sort_values(ascending=True).reset_index(drop=True)[0]

            hist = pd.DataFrame(pd.Series(df.value_counts())
                                    .reset_index().values
                                    ,columns=['name','num'])

            # prof.loc['MeanValue', col] = hist.num.mean()
            # prof.loc['MedianValue', col] = hist.num.median()
            # prof.loc['ModeValue', col] = hist.num.mode()[0]
            # prof.loc['StdValue', col] = np.std(hist.num)
            # prof.loc['Range', col] = [prof.loc['MinValue', col],prof.loc['MaxValue',col]]

            prof.loc['Top', col] = hist.loc[0,'name']
            prof.loc['Freq', col] = hist.loc[0, 'num']

            prof.loc['MeanValue', col] = 'NA'
            prof.loc['MedianValue', col] = 'NA'
            prof.loc['ModeValue', col] = 'NA'
            prof.loc['StdValue', col] = 'NA'
            prof.loc['Range', col] = 'NA'


            prof.loc['NumNull', col] = sum(df.isna() | df.isnull())
            prof.loc['NonNull', col] = \
                round(sum(df.isna() | df.isnull())*100/len(df),2)
            prof.loc['PercentNull', col] = df.count()

            # Calculate the 25, 75 and 95 percentiles of the values
            # prof.loc['q25', col] = 'NA'
            # prof.loc['q75', col] = 'NA'
            # prof.loc['q95', col] = 'NA'

            prof.loc['UniqueValues', col] = df.unique()
            prof.loc['NumUnique', col] = df.nunique()
            prof.loc['ValueFreq',col] = hist.values
            prof.loc['LongestLength', col] = df.astype('str').str.len().max()
            prof.loc['LongestLengthVal', col] = df.astype('str').str.len().idxmax()
            prof.loc['ShortestLength', col] = df.astype('str').str.len().min()
            prof.loc['ShortestLengthVal', col] = df.astype('str').str.len().idxmin()
            prof.loc['ShortestLengthNonNullVal', col] = df.dropna().astype('str').str.len().idxmin()

        return prof

