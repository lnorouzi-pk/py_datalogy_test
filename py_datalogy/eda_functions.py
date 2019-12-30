"""
    :Author: Zach Clement, Leila Norouzi (editor)
    :Date: 27 Dec 2019
    :Description: Functions providing explanatory data analysis for a given data set.

    ...


    Methods
    -------
    analyze_numeric(df,save_pic=True, path='')
        Generates 6 different plots for every given numeric variables. If save_pic is set to True it will save the
        results in given path.

    analyze_cat(df,save_pic=True, path='')
        Generates a histogram for every given non-numerical variables. If save_pic is set to True it will save the
        results in given path.

    analyze_numeric_correlations(df,save_pic=True, path='')
        Generates a heatmap plot for correlation between numeric variables. If save_pic is set to True it will save the
        results in given path.

    analyze_pair(df:pd.DataFrame, x1: str, x2:str, save_pic=True, path='')
        Group the data by X1 and X2 and then counts the frequency of appearance of those values. If save_pic is set to
        True it will save the results in given path.
"""


import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns



# Colors for different percentiles
perc_25_color = 'gold'
perc_50_color = 'mediumaquamarine'
perc_75_color = 'deepskyblue'
perc_95_color = 'peachpuff'
perc_color = [perc_25_color, perc_50_color, perc_75_color, perc_95_color]

def analyze_numeric(df,save_pic=True, path='') -> None:
    '''
    :Description: Generates 6 different plots for every given numeric variables. If save_pic is set to True it will
    save the results in given path.
        1th plot: Box plot for all the values
        2nd plot: Distribution of all values
        3rd plot: Boxplot for quartiles (all values)
        4th plot: Box plot without outliers
        5th plot: Violin plot (<95% percentile)
        6th plot: Histogram (<95% percentile)

    :param df: The data to be investigated.
    :type df: pandas data frame
    :param save_pic: if the user wants to save the results. default is True to save the results as a png file.
    :type save_pic: bool

    :return: None, 6  plots for EDA data visualization.

    '''

    num_cols = [cols for cols in df.columns if is_numeric_dtype(df[cols]) and len(df[cols].dropna()) > 0]

    iter_len = len(num_cols)

    print('Path inside analyze_numeric',path)

    # For each numeric column in the list
    for x, col_name in enumerate(num_cols):

        print(x + 1, " of ", iter_len, " completed   ", col_name)

        # Create a copy of the column values without nulls or NA
        no_null_col = df[col_name].dropna()

        # Calculate the 95 percentile of the values
        q25 = np.percentile(no_null_col, 25)
        q75 = np.percentile(no_null_col, 75)
        q95 = np.percentile(no_null_col, 95)

        # Plot the graphs
        fig, ax = plt.subplots(2, 3,figsize=(17, 9),constrained_layout=True)
        fig.suptitle("Profile of column  " + str(col_name).strip()+'\n', fontsize=25)


        # figure 1

        ax[0,0].set_title("Box plot for all the values")
        plt.setp(ax[0,0].get_xticklabels(), ha="right", rotation=35)
        plt.setp(ax[0,0].get_yticklabels(), ha="right")
        ax[0,0].boxplot(no_null_col)


        # figure 2
        ax[0,1].set_title("Distribution of all values")
        plt.setp(ax[0,1].get_xticklabels(), ha="right", rotation=35)
        plt.setp(ax[0,1].get_yticklabels(), ha="right")
        ax[0,1].hist(no_null_col)

        # figure 3
        ax[0,2].set_title("Boxplot for quartiles (all values)")
        if len(no_null_col.value_counts()) >= 4:
            df['quartiles']=pd.qcut(df[col_name],
                4, duplicates='drop' , precision=1)
            sns.boxplot(data=df, y=col_name, x='quartiles', ax=ax[0,2],
                        palette=sns.color_palette(perc_color))
        plt.setp(ax[0,2].get_xticklabels(), ha="right", rotation=35)
        plt.setp(ax[0,2].get_yticklabels(), ha="right")


        # figure 4
        ax[1,0].set_title("Box plot without outliers")
        plt.setp(ax[1,0].get_xticklabels(), ha="right", rotation=35)
        plt.setp(ax[1,0].get_yticklabels(), ha="right")
        ax[1,0].boxplot(no_null_col, showfliers=False)

        # figure 5
        ax[1,1].set_title("Violin plot (<95% percentile)")
        plt.setp(ax[1,1].get_xticklabels(), ha="right", rotation=35)
        plt.setp(ax[1,1].get_yticklabels(), ha="right")
        ax[1,1].violinplot(no_null_col[no_null_col <= q95])

        # figure 6
        # Histogram with bin ranges, counts and percentile color
        ax[1,2].set_title("Histogram (<95% percentile)")
        plt.setp(ax[1,2].get_xticklabels(), ha="right", rotation=35)
        plt.setp(ax[1,2].get_yticklabels(), ha="right")

        # Take only the data less than 95 percentile
        data = no_null_col[no_null_col <= q95]

        counts, bins, patches = ax[1,2].hist(data, bins=10, facecolor=perc_50_color, edgecolor='gray')

        # Set the ticks to be at the edges of the bins.
        ax[1,2].set_xticks(bins.round(2))
        plt.xticks(rotation=70)

        # Change the colors of bars at the edges
        for patch, leftside, rightside in zip(patches, bins[:-1], bins[1:]):
            if rightside < q25:
                patch.set_facecolor(perc_25_color)
            elif leftside > q95:
                patch.set_facecolor(perc_95_color)
            elif leftside > q75:
                patch.set_facecolor(perc_75_color)

        # Calculate bar centre to display the count of data points and %
        bin_x_centers = 0.5 * np.diff(bins) + bins[:-1]
        bin_y_centers = ax[1,2].get_yticks()[1] * 0.25

        # Display the the count of data points and % for each bar in histogram
        for i in range(len(bins) - 1):
            bin_label = "{0:,}".format(counts[i]) + "  ({0:,.2f}%)".format((counts[i] / counts.sum()) * 100)
            plt.text(bin_x_centers[i], bin_y_centers, bin_label, rotation=90, rotation_mode='anchor')

        # create legend
        handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in perc_color]
        labels = ["0-25 Percentile", "25-50 Percentile", "50-75 Percentile", ">95 Percentile"]
        ax[1,2].legend(handles, labels, bbox_to_anchor=(0.5, 0., 0.85, 0.99), loc= 'upper right')

        fig_name = path+'EDA_' + col_name
        if save_pic==True :
            fig.savefig(fig_name, dpi=50)

        plt.show()
        plt.close(fig)

        df.drop(u'quartiles', axis=1, inplace=True)

        pass



def analyze_cat(df,save_pic=True, path='') -> None:
    '''
    :Description: This function plot normalized frequency values of the variable in a bar chart for frequency larger than 25%.

    :param df: The data to be investigated.
    :type df: pandas data frame
    :param save_pic: if the user wants to save the results. default is True to save the results as a png file.
    :type save_pic: bool
    :param path: the path to save the plot in.
    :type path: str

    :return: None, a plot.
    '''
    obj_cols = [cols for cols in df.columns if is_string_dtype(df[cols]) and len(df[cols].dropna()) > 0]

    # For each object column in the list
    for x, col_name in enumerate(obj_cols):
        # print(x + 1, " of ", iter_len, " completed   ", col_name)
        values_freq_threshold = 25

        # If unique values count is below the threshold value then store the details of unique values
        # normalize True/False for counts
        col_unique_vals = df[col_name].value_counts(normalize=True, sort=True)
        #generating a data frame for the normalized count value data
        f = pd.DataFrame(np.array(col_unique_vals.head(values_freq_threshold).reset_index()),columns=['Values','Count'])

        # Plot the graphs
        fig, ax = plt.subplots(figsize=(17, 9), constrained_layout=True)
        fig.suptitle("Profile of column  " + str(col_name).strip() , fontsize=25)
        ax.bar(f.Values,f.Count,color= perc_color[1])
        ax.set_title("Normalized bar chart for top 25 values")
        plt.xticks(rotation=90)
        ax.set_ylabel('Count')
        ax.set_xlabel('Values')
        for p in ax.patches:
            ax.annotate(str(round(p.get_height(), 2)), (p.get_x() , p.get_height() * 1.01))

        fig_name = path+'EDA_' + str(col_name).strip()+'.png'
        if save_pic: fig.savefig(fig_name, dpi= 100)
        plt.show()
        plt.close(fig)

        pass


def analyze_numeric_correlations(df,save_pic=True, path='') -> None:
    '''
    :Description: This function calculates the correlation between numeric variables of a given data and plots the
    correlation in a heatmap plot.

    :param df: The data to be investigated.
    :type df: pandas dataframe
    :param save_pic: if the user wants to save the results. default is True to save the results as a png file.
    :type save_pic: bool
    :param path: the path to save the plot in.
    :type path: str

    :return:  None, a plot.
    '''
    fig, ax = plt.subplots(figsize=(17, 8), constrained_layout=True)

    corr_data = df.corr()
    g= sns.heatmap(corr_data,
                mask=np.zeros_like(corr_data, dtype=np.bool),
                cmap=sns.diverging_palette(20, 220, as_cmap=True),
                vmin=-1, vmax=1,
                square=True,
                ax=ax)
    g.set_yticklabels(g.get_yticklabels(), rotation=0)
    ax.set_title("Correlation between variables")
    ax.set_xlabel('Varibales')
    ax.set_ylabel('Varibales')

    fig_name = path+'EDA_numeric_cor_plot.png'
    if save_pic: fig.savefig(fig_name,  dpi=100)

    plt.show()
    plt.close(fig)

    pass

def analyze_pair(df:pd.DataFrame, x1: str, x2:str, save_pic=True, path='') -> None:
    '''
    :Description: It takes two variables of a data. Groups the data based on those variables and count
    the frequency of each group.

    :param df: The data
    :type df: pandas dataframe
    :param x1: First variable to be grouped by
    :type x1: str
    :param x2: Second variable to be grouped by
    :type x2: str
    :param save_pic: if the user wants to save the results. default is True to save the results as a png file.
    :type save_pic: bool
    :param path: the path to save the plot in.
    :type path: str

    :return:
    '''
    fig, ax = plt.subplots(figsize=(17, 8), constrained_layout=True)

    data = pd.DataFrame(df.groupby([x1, x2])[x1, x2].size(), columns=['paired_count']).reset_index()
    mmin = data.paied_count.min()
    mmax = data.paied_count.max()
    mavg = np.mean([mmin,mmax])

    print(data.head(),mmin,mmax)
    data = data.pivot(x1, x2, "paired_count")
    g= sns.heatmap(data,ax=ax,vmin=mmin, vmax=mmax, center=mavg)
    g.set_yticklabels(g.get_yticklabels(), rotation=0)

    ax.set_title("Frequency of %s and %s appeared together \n"%(x1,x2))

    fig_name = path+'EDA_pair_'+x1+'_'+x2+'.png'
    if save_pic: fig.savefig(fig_name, dpi=100)
    plt.show()
    plt.close(fig)
    pass



