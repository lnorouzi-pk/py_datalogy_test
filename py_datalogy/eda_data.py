"""
This is the integration of data profiling, null analysis and basic data visualization.
It generates a text file as the log of running code at the location of the working directory, named 'outputlog.txt'
It also generates data profiling as a excel file, Null analysis results and data visualization of data of the interest.
"""

import pandas as pd
# import numpy as np
import sys
import os
import warnings


from data_profiling import *
from eda_functions import *
from analyze_nulls import *

#To ignore warning while running the code
warnings.filterwarnings("ignore")

# the log file name
filename= 'outputlog.txt'
# To write the python console outputs in a text file, to save the records if it is needed.
sys.stdout = open(os.getcwd()+'/outputlog.txt', 'w')

class EDA(object):
    """
    :Description:

    Atributes
    ---------
        data: pandas data frame
            the data to be investigated
        path: str
            the path that the results will be saved. If the path is not existed, it will create one.
        visual: bool
            whether to plot the graphs or not
        save_file: bool
            whether to save the results or not



    Methods
    -------
        numeric_eda: This function focuses on the numeric variables/ columns of the data. It follows these steps:
                    1- Using data_profiling, it preforms the data profiling of the data and generates an excel
                    spreadsheet if it was indicated that the results are needed to be saved.
                    2- Using eda_funtion, it generates 6 different plots for every given numeric variables. If
                    save_pic is set to True it will save the results in given path.
                        1th plot: Box plot for all the values
                        2nd plot: Distribution of all values
                        3rd plot: Boxplot for quartiles (all values)
                        4th plot: Box plot without outliers
                        5th plot: Violin plot (<95% percentile)
                        6th plot: Histogram (<95% percentile)
                    3- Using eda_funtion, calculates the correlation between numeric variables of a given data and plots
                    the correlation in a heatmap plot.
                    4- Using analyze_nulls, plots the location of the null values in all numeric variables/ columns.
                    5- Using analyze_nulls, it plots the correlation between variables with null values
                    6- Using analyze_nulls, it plots the correlation with variables at the nulls location
                    7- Using analyze_nulls, to see which numeric variables are commonly null at the same time

                    The result may be shown in graphs and saved if the keywords were set.

        non_numeric_eda: This function focuses on the numeric variables/ columns of the data. It follows these steps:
                    1- Using data_profiling, it preforms the data profiling of the data and generates an excel
                    spreadsheet if it was indicated that the results are needed to be saved.
                    2- Using eda_funtion, it generates a histogram of distribution of all values for every given
                        non-numeric variables. If save_pic is set to True it will save the results in given path.
                    3- Using analyze_nulls, plots the location of the null values in all numeric variables/ columns.
                    4- Using analyze_nulls, it plots the correlation between variables with null values
                    5- Using analyze_nulls, it plots the correlation with variables at the nulls location
                    6- Using analyze_nulls, to see which numeric variables are commonly null at the same time

                    The result may be shown in graphs and saved if the keywords were set.

        total_eda: It will preform numeric_eda and non_numeric_eda for all data.

    :Example:
    >>> import seaborn as sns
    >>> from eda_data import *
    >>> # getting data from seaborn package
    >>> data = sns.load_dataset(name='tips')
    >>> # Getting the EDA of the data
    >>> ed = EDA(data)
    >>> ed.total_eda()

    """

    def __init__(self, data:pd.DataFrame, path='' ,save_file=True, visual=True):
        '''

        :param data: The data to be investigated.
        :type data: pd.DataFrame
        :param path: The path that results will be saved
        :type path: str
        :param save_file: Whether to save results or not
        :type save_file: bool
        :param visual: Whether to plot any graph or not
        :type visual: bool
        '''
        self.data = data
        try:
            os.stat(path)
            self.path = path
        except:
            os.mkdir(path)
            self.path = path
        self.save_file= save_file
        self.visual = visual

    def numeric_eda(self) -> None:
        '''
        This function focuses on the numeric variables/ columns of the data. It follows these steps:
            1- Using data_profiling, it preforms the data profiling of the data and generates an excel
            spreadsheet if it was indicated that the results are needed to be saved.
            2- Using eda_funtion, it generates 6 different plots for every given numeric variables. If
            save_pic is set to True it will save the results in given path.
                1th plot: Box plot for all the values
                2nd plot: Distribution of all values
                3rd plot: Boxplot for quartiles (all values)
                4th plot: Box plot without outliers
                5th plot: Violin plot (<95% percentile)
                6th plot: Histogram (<95% percentile)
            3- Using eda_funtion, calculates the correlation between numeric variables of a given data and plots
            the correlation in a heatmap plot.
            4- Using analyze_nulls, plots the location of the null values in all numeric variables/ columns.
            5- Using analyze_nulls, it plots the correlation between variables with null values
            6- Using analyze_nulls, it plots the correlation with variables at the nulls location
            7- Using analyze_nulls, to see which numeric variables are commonly null at the same time

            The result may be shown in graphs and saved if the keywords were set.
        :return:
        '''
        # Running the data profiling part
        dp = DataProfiling(self.data, path=self.path)

        df = dp.numeric_data
        print(df.info(verbose=True))
        NumericProfile = dp.numeric_profile_df
        print('\n--------------Printing data descriptive information-------------- :\n',NumericProfile)

        if self.save_file:
            file_name = self.path + 'data_profiling_numerical.xlsx'
            print('\n--------------Saving data descriptive information in %s-------------- :\n' % (file_name))
            with pd.ExcelWriter(file_name) as writer:  # doctest: +SKIP
                print('The data profiling results is saved in %s \n' % (file_name))
                NumericProfile.to_excel(writer, sheet_name='Numerical_data_profile')

        # Running data visualization part
        # plotting 6 graphs
        analyze_numeric(df, path=self.path)
        # plotting the correlation
        analyze_numeric_correlations(df, path=self.path)

        # Running null analysis part
        an = AnalyzeNulls(df, path=self.path)
        # The location of the null values
        an.locations(save_pic=self.save_file)

        print('\n\n--------------Correlation between varibales with null values--------------')
        cor_btwn_nulls_df = an.cor_btwn_nulls(visual=self.visual, save_pic=self.save_file)
        print(cor_btwn_nulls_df)

        print('\n\n------------Correlation with varibales at the nulls location--------------')
        corr_with_nulls_df= an.corr_with_nulls(visual=self.visual, save_pic=self.save_file)
        print(corr_with_nulls_df)

        print('\n\n------------boxplots by null--------------')
        for x in list(df.columns):
            an.boxplots_by_null(var=x)

        print('\n--------------null_combinations: Numeric variables--------------')
        print('\tSee which numeric variables are commonly null at the same time. A 1 denotes the column being null and a 0 denotes')
        print('\tit having a value. Freq counts the number of rows with this pattern.\n\n')
        print(an.null_combinations())

        pass


    def non_numeric_eda(self) -> None:
        '''
        This function focuses on the numeric variables/ columns of the data. It follows these steps:
            1- Using data_profiling, it preforms the data profiling of the data and generates an excel
            spreadsheet if it was indicated that the results are needed to be saved.
            2- Using eda_funtion, it generates a histogram of distribution of all values for every given
                non-numeric variables. If save_pic is set to True it will save the results in given path.
            3- Using analyze_nulls, plots the location of the null values in all numeric variables/ columns.
            4- Using analyze_nulls, it plots the correlation between variables with null values
            5- Using analyze_nulls, it plots the correlation with variables at the nulls location
            6- Using analyze_nulls, to see which numeric variables are commonly null at the same time

            The result may be shown in graphs and saved if the keywords were set.
        :return:
        '''
        # Running the data profiling part
        dp = DataProfiling(self.data, path=self.path)

        df = dp.non_numeric_data

        print(df.info(verbose=True))
        Non_NumericProfile = dp.non_numeric_profile_df
        print('\n--------------Printing data descriptive information-------------- :\n', Non_NumericProfile)

        if self.save_file:
            file_name = dp.path + 'data_profiling_non_numerical.xlsx'
            print('\n--------------Saving data descriptive information in %s-------------- :\n' % (file_name))
            with pd.ExcelWriter(file_name) as writer:  # doctest: +SKIP
                print('The data profiling results is saved in %s \n' % (file_name))
                Non_NumericProfile.to_excel(writer, sheet_name='Non_numerical_data_profile')

        # Running data visualization part
        # plotting the distribution
        analyze_cat(df, path=self.path)

        # Running null analysis part
        an = AnalyzeNulls(df, path=self.path)
        # The location of the null values
        an.locations()

        print('\n\n--------------Correlation between variables with null values--------------')
        cor_btwn_nulls_df = an.cor_btwn_nulls(visual=self.visual, save_pic=self.save_file)
        print(cor_btwn_nulls_df)

        print('\n\n------------Correlation with variables at the nulls location--------------')
        corr_with_nulls_df = an.corr_with_nulls(visual=self.visual, save_pic=self.save_file)
        print(corr_with_nulls_df)

        print('\n--------------null_combinations: Non_numeric variables--------------')
        print('\tSee which numeric variables are commonly null at the same time. A 1 denotes the column being '+\
              'null and a 0 denotes')
        print('\tit having a value. Freq counts the number of rows with this pattern.\n\n')
        print(an.null_combinations())

        pass


    def total_eda(self):
        '''
        This function first preform data profiling for numeric and non-numeric data if there are any. Then saves the
            results if a excel xlsx file. Next, it preform numeric_eda if the data has any numeric variables and
            non_numeric_eda if the data has any non-numeric variables.
        :return:
        '''


        dp = DataProfiling(self.data , self.path)

        print('The data has some numeric variables:\t %s \nThe data has some non-numeric variables:\t%s'
              %(len(dp.numerical_columns) > 0 , len(dp.object_columns)>0))
        file_name = dp.path+'data_profiling_total.xlsx'
        print('\n--------------Saving data descriptive information in %s-------------- :\n' % (file_name))
        with pd.ExcelWriter(file_name) as writer:  # doctest: +SKIP
            print('The data profiling results is saved in %s \n'%(file_name))
            if len(dp.numerical_columns) > 0:
                dp.numeric_profile_df.to_excel(writer, sheet_name='Numerical_data_profile')
                self.numeric_eda()
            if len(dp.object_columns)>0 :
                dp.non_numeric_profile_df.to_excel(writer, sheet_name='Non-numerical_data_profile')
                self.non_numeric_eda()
            if ((len(dp.numerical_columns) > 0) & (len(dp.object_columns)>0)):
                dp.total_profile_df.to_excel(writer, sheet_name='All_data_profile')

        pass



