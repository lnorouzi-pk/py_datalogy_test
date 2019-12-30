


import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


class AnalyzeNulls(object):
    """
    :Author: Isaac Slaughter, Leila Norouzi (editor)
    :Date: 27 Dec 2019
    :Description: A class  for determining whether data is Missing Completely At Random, or if other variables influence a variable's
    missingness. Many functions loosely based on R's VIM package.
    Class for analyzing whether variables are omitted completely at random, or conditionally on other variables.
    The AnalyzeNulls object should be imported then instantiated with the data in question. Descriptions of functions
    are provided in their docstrings.

    ...

    Attributes
    ----------
    data : pandas data frame
        The data to be investigated
    path : str
        The path for saving the results
    missing_columns_indexer : list
        A list of column names indexes that contain null values
    missing_columns : list
        A list of column names that contain null values
    missing_data_df : Pandas data frame
        The data that include columns containing null values in their entries
    null_indicators : Pandas data frame
        If the entries in the missing_data_df is null or not

    Methods
    -------
    locations(save_pic=True)
        Shows where in data missing values are located in a heat map plot.

    cor_btwn_nulls(visual=True, save_pic=True)
        Generates a dataframe showing which columns's nulls are correlated to which other column's nulls.
        If visual is set to True it plots a heatmap.

    corr_with_nulls(visual=True, save_pic=True)
        Generates a dataframe to show which columns have correlation with a variable being null.
        If visual is set to True it plots a heatmap.

    null_combinations()
        Generates a dataframe to show which variables are commonly null at the same time.
        A 1 denotes the column being null and a 0 denotes it having a value. Freq counts the number of rows with this
        pattern.

    boxplots_by_null(var: str, save_pic=True)
        View distribution in variables, split based on whether the observation has var null.

    pairwise_null_comparison(x: str, y: str)
        View a scatter plot showing x vs. y. For variables where x is missing, show a heatmap of their distribution
        in y, and vice versa.


    """
    def __init__(self, data: pd.DataFrame, path=''):
        '''

        :param data: teh data to be investigated
        :type data: pd.DataFrame
        :param path: The path to save the results


        
        '''
        self.path = path
        self.data = data.copy(deep=True)
        missing_columns_indexer = np.any(self.data.isnull(),axis = 0)
        self.missing_columns = self.data.columns[missing_columns_indexer]
        self.missing_data_df = self.data[self.missing_columns]
        self.null_indicators = self.missing_data_df.isnull()

    def locations(self, save_pic=True) -> None:

        """
        See where in data missing values are located. Nulls are shown in white.

        :param save_pic: Whether save the results or not. Default is True to save pictures.
        :type save_pic: bool.

        :return: None
        """
        fig = plt.figure(figsize= (15,8))
        main_ax = plt.subplot2grid((8, 20), (0, 0), colspan=19, rowspan= 8)
        legend_ax = plt.subplot2grid((8, 20), (5, 19), colspan=1,rowspan=1)

        value_to_int = {j: i for i, j in enumerate([0, 1])}

        g = sns.heatmap(self.data.isnull().transpose(),vmax=1, vmin= 0, center=0.5,ax=main_ax, cmap="Reds", cbar=False)
        g.set_yticklabels(g.get_yticklabels(), rotation=0)

        main_ax.set_title("Missing values' locations\n",fontsize='large')
        main_ax.set_xlabel('Index')
        main_ax.set_ylabel('Column names')

        legend_ax.axis('off')

        # reconstruct color map
        colors = plt.cm.Reds(np.linspace(0, 1, len(value_to_int)))

        # add color map to legend
        patches = [mpatches.Patch(facecolor=c, edgecolor=c) for c in colors]
        legend = legend_ax.legend(patches,
                                  ['Non-nulls','Nulls'],
                                  handlelength=0.9, loc='lower left')
        for t in legend.get_texts():
            t.set_ha("left")

        plt.tight_layout()

        fig_name = self.path+'Nulls_location.png'
        if save_pic: fig.savefig(fig_name, dpi=100)
        plt.show()
        plt.close(fig)

        pass



    def cor_btwn_nulls(self, visual=True, save_pic=True) -> pd.DataFrame:
        """
        See which columns's nulls are correlated to which other column's nulls. E.g. if Var A and Var B are null
        for all the same rows, they will have correlation 1.

        :param visual: Whether draw the results or not. Default is True to save pictures.
        :type visual: bool.
        :param save_pic: Whether save the results or not. Default is True to save pictures.
        :type save_pic: bool.

        :return: c: pd.DataFrame -- Correlation between nulls.
        """
        null_locations = self.missing_data_df.isnull()
        null_locations.columns = null_locations.columns
        corr_data = null_locations.corr()

        if visual :
            fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
            g = sns.heatmap(corr_data,
                        mask=np.zeros_like(corr_data, dtype=np.bool),
                        cmap=sns.diverging_palette(20, 220, as_cmap=True),
                        vmin=-1, vmax=1,
                        square=True,
                        ax=ax)
            ax.set_title('Correlation between nulls\n\n', fontsize='large')
            ax.set_xlabel('Column names')
            ax.set_ylabel('Column names')
            g.set_yticklabels(g.get_yticklabels(), rotation=0)

            # plt.tight_layout()

            fig_name = self.path+'cor_btwn_nulls.png'
            if save_pic: fig.savefig(fig_name, dpi=100)

            plt.show()
            plt.close(fig)

        return corr_data

    def corr_with_nulls(self, visual=True, save_pic=True) -> pd.DataFrame:
        """
        See which columns have correlation with a variable being null. The rows correspond to a variable and the columns
        are an indicator for whether a variable is null. E.g. the data at left would result in the following table
                     Row | A   | B   |                     |   | A_is_null | B_is_null |
                         |-----|-----|                     |---|-----------|-----------|
                       0 | 2   | NaN |      ->             | A | NaN       | 1         |
                       1 | 1   | 3.7 |                     | B | 0         | NaN       |
                       2 | 2   | NaN |
                       3 | NaN | 3.8 |
                       4 | 1   | 3.8 |
                       5 | NaN | 3.7 |
        When calculating correlation, rows are omitted if the variable is null in a pairwise fashion.  E.g. for
        calculating correlation between A and the indicator B_is_null, rows 3 and 5 are omitted. Only numeric columns
        are included as rows, and only columns with nulls are included as columns.

        :param visual: Whether draw the results or not. Default is True to save pictures.
        :type visual: bool.
        :param save_pic: Whether save the results or not. Default is True to save pictures.
        :type save_pic: bool.
        :return: correlations: pd.DataFrame -- Correlations as described above
        """
        null_indicators = self.null_indicators.copy(deep=True)
        null_indicators.columns = self.null_indicators.columns + '_is_null'
        n_indicators = len(null_indicators.columns)
        data_with_null_indicators = pd.concat([null_indicators, self.data], axis=1)
        correlations = data_with_null_indicators.corr()
        correlations = correlations.iloc[n_indicators:,:n_indicators]

        if visual :
            fig, ax = plt.subplots(figsize=(17, 8), constrained_layout=True)
            g = sns.heatmap(correlations,
                        mask=np.zeros_like(correlations, dtype=np.bool),
                        vmin=-1, vmax=1, center= 0.5,
                        # square=True,
                        ax=ax)
            ax.set_title('Correlation between nulls and variables\n\n', fontsize='large')
            ax.set_xlabel('Column names')
            ax.set_ylabel('Column names')
            g.set_yticklabels(g.get_yticklabels(), rotation=0)

            fig_name = self.path+'corr_with_nulls.png'
            if save_pic: fig.savefig(fig_name, dpi=100)

            plt.show()
            plt.close(fig)

        return correlations


    def null_combinations(self) -> pd.DataFrame:
        """
        ???????????
        See which variables are commonly null at the same time. A 1 denotes the column being null and a 0 denotes
        it having a value. Freq counts the number of rows with this pattern.

        :return: pd.DataFrame -- pattern vs. number of rows showing pattern
        """
        # print(len(self.missing_columns))
        if len(self.missing_columns) >0:
            # Count number of times each pattern shows up
            self.null_indicators['freq'] = 1
            patterns = self.null_indicators.groupby(self.missing_columns.to_list()).agg("sum")



            # Turn missing columns back into columns, instead of as index
            patterns = patterns.reset_index()

            # Sort so highest frequency patterns show together
            patterns = patterns.sort_values('freq', axis= 0, ascending=False)

            # Reset meaningless index
            patterns = patterns.reset_index(drop=True)

            # convert boolean to numeric
            patterns = patterns * 1

            # remove frequency variable
            self.null_indicators.drop('freq', axis=1)
        else :
            print('!!!There is no null value in this variable!!!')
            patterns=None

        return patterns

    def boxplots_by_null(self, var: str, save_pic=True):
        """
        View distribution in variables, split based on whether the observation has var null.

        :param var: Variable to create null indicator for.
        :type var: str
        :param save_pic: Whether save the results or not. Default is True to save pictures.
        :type save_pic: bool.

        :return: None

        """
        if type(var) is not str:
            raise TypeError("Var must be string")

        boxplot_df = self.data.copy(deep = True)
        boxplot_df[f'{var}_is_null'] = boxplot_df[var].isnull()
        boxplot_df = boxplot_df.drop(var, axis=1)
        number_of_numeric_col = boxplot_df.select_dtypes(include=[np.number]).shape[1]
        fig, axes = plt.subplots(ncols=number_of_numeric_col, figsize = (17,8))
        boxplot_df.boxplot(by=f'{var}_is_null', return_type='axes', ax=axes)

        if number_of_numeric_col >1 :
            for x in axes:
                x.set_xlabel(var+' is null')
        else: axes.set_xlabel(var+' is null')

        fig.suptitle(f'Boxplots grouped by whether {var} is null')

        fig_name = self.path+'boxplots_by_null.png'
        if save_pic: fig.savefig(fig_name, dpi=100)

        plt.show()
        plt.close(fig)
        pass

    def pairwise_null_comparison(self, x: str, y: str) -> None:
        """
        View a scatter plot showing x vs. y. For variables where x is missing, show a heatmap of their distribution
        in y, and vice versa.

        :param x: Name of x variable, Must be numeric.
        :type x: str
        :param y: Name of y variable, must be numeric.
        :type y: str

        :return: None
        """
        if type(x) is not str:
            raise TypeError("x must be string")
        if type(y) is not str:
            raise TypeError("y must be string")
        if not is_numeric_dtype(self.data[x]):
            raise ValueError("x must refer to a numeric variable")
        if not is_numeric_dtype(self.data[y]):
            raise ValueError("y must refer to a numeric variable")

        x_non_null_indexer = self.data[x].notna()
        y_non_null_indexer = self.data[y].notna()
        x_and_y_present = self.data[[x, y]].loc[(x_non_null_indexer & y_non_null_indexer)]
        only_x_present = self.data[[x, y]].loc[(x_non_null_indexer & ~y_non_null_indexer)]
        only_y_present = self.data[[x, y]].loc[(~x_non_null_indexer & y_non_null_indexer)]

        # Define plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots( 2, 2, figsize= (17,8), gridspec_kw={'width_ratios': [1, 35],
                                                                        'height_ratios': [35,1]})
        # Setup scatter
        ax2.scatter(x_and_y_present[x].to_list(), x_and_y_present[y].to_list())
        ax2.tick_params(
            which='both',
            labelleft=True,
            labelbottom=True
        )

        # Get bounds of scatter, for use by heatmaps
        ymin, ymax = ax2.get_ybound()
        yrange = ymax - ymin
        xmin, xmax = ax2.get_xbound()
        xrange = xmax - xmin

        # Calculate Y heatmap
        yheatmap, _, _ = np.histogram2d(np.zeros(only_y_present[y].shape),
                                        only_y_present[y].to_list(),
                                        bins=(1, np.linspace(ymin, ymax)))
        yheatmap = yheatmap.T
        ybox = [0, yrange/35, ymin, ymax]

        # Calculate X heatmap
        xheatmap, _, _ = np.histogram2d(only_x_present[x].to_list(),
                                        np.zeros(only_x_present[x].shape),
                                        bins=(np.linspace(xmin, xmax), 1))
        xheatmap = xheatmap.T
        xbox = [xmin, xmax, 0, xrange/50]

        # Plot heatmaps
        highest_frequency = max(np.max(xheatmap), np.max(yheatmap))
        defining_heatmap = (xheatmap, yheatmap)[np.argmax([np.max(xheatmap), np.max(yheatmap)])]
        ax1.imshow(yheatmap, extent=ybox, origin='lower', cmap='Reds',vmin=0, vmax=highest_frequency)
        ax4.imshow(xheatmap, extent=xbox, origin='lower', cmap='Reds',vmin=0, vmax=highest_frequency)

        # Formatting
        ax1.set_ylabel(y)
        ax4.set_xlabel(x)
        ax1.tick_params(
            axis='both',
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelleft=False,
            labelbottom=False,
            bottom=False,
            top=False
        )
        ax4.tick_params(
            axis='both',
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelleft=False,
            labelbottom=False,
            bottom=False,
            top=False
        )
        ax3.axis('off')
        fig.suptitle(f'{x} vs. {y}', fontsize= 14)
        ax2.set_title("Heatmaps show frequency of observations where one variable is null")
        plt.show()

        plt.close('fig')
        pass