# LIBRARIES
# Pandas for data manipulation and analysis
import pandas as pd
import numpy as np

# Check if the dtype of a column is object
from pandas.api.types import is_object_dtype

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns

def data_explore(data):
    '''
    General information about the dataset (number of rows, columns, duplicated rows and number of features corresponding to each type)
    
    Parameters:
    data (pandas.DataFrame): A dataframe needed to be explored
        
    Return:
    pandas.DataFrame: A dataframe with basic information about the input dataframe
    
    '''
    # Number of rows, columns and duplicated rows
    info = [['Rows', data.shape[0]],
            ['Features', data.shape[1]],
            ['Duplicate Rows', data.duplicated().sum()]]
    datainfo = pd.DataFrame(info, columns=['index', 'info']).set_index('index')
    
    # Number of features corresponding to each type
    dtype = pd.DataFrame(data.dtypes.value_counts()).rename(columns={'count': 'info'})
    
    # Final result
    datainfo = pd.concat([datainfo, dtype], axis=0)
    
    return datainfo

# Check Missing Value
def check_nan(data, axis=1):
    '''
    Return the number and percentage of NaN value corresponding to chosen axis in dataset
    
    Parameters:
    - data (pandas.DataFrame): Data that contains features you want to explore
    - axis (bool/int) {True, False, 1, 0}: default=1
        + If True or 1, return missing value per column
        + If False or 0, return missing value per row
        
    Return:
    pandas.DataFrame: A dataframe contains information about the number and percentage of NaN values corresponding to chosen axis
    
    '''
    output = pd.DataFrame(data.isnull().sum(1-axis), columns=['nan']).sort_values(by='nan', ascending=False)
    output = output[output['nan']>0]
    output['%nan'] = output['nan']/data.shape[1-axis]*100
    return output
    

# All Feature Exploration
def multi_features_explore(data, reverse=False):
    '''
    Explore multiple features of the dataset
    
    Parameters:
    - data (pandas.DataFrame): Data that contains features you want to explore
    - reverse (bool): default=False
        + If False, output will be a DataFrame with feature's name as columns
        + If True, output will be a DataFrame with feature's name as indexes
        
    Return:
    pandas.DataFrame: A dataframe of basic information about all features of the input data
    
    '''
    colinfo = []
    for col in data:
        colinfo.append(feature_explore(data[col]))
    if not reverse:
        return pd.concat(colinfo, axis=1).T
    else:
        return pd.concat(colinfo, axis=1)
    
# Single Feature Exploration
def feature_explore(feature):
    '''
    General information of a specific feature
    
    Parameters:
    feature (pandas.Series): Feature needed to be explored
    
    Return:
    pandas.DataFrame:  A dataframe of basic information about a specific feature
    
    '''
    info = [['dtype', feature.dtype],
            ['nonnull', feature.notnull().sum()],
            ['%nonnull', round(feature.notnull().sum()/feature.shape[0], 2)],
            ['nan', feature.isnull().sum()],
            ['%nan', round(feature.isnull().sum()/feature.shape[0], 2)],
            ['nunique', feature.nunique()],
            ['nunique_nan', len(feature.unique())]]
    if is_object_dtype(feature):
        info.extend([['unique', feature.unique()],
                     ['frequency', feature.value_counts().to_dict()],
                     ['%value', feature.value_counts(normalize=True).round(2).to_dict()],
                     ['most', feature.mode().values]])
    else:
        if feature.nunique() <= 10:
            info.extend([['unique', feature.unique()],
                         ['frequency', feature.value_counts().to_dict()],
                         ['%value', feature.value_counts(normalize=True).round(2).to_dict()]])
        info.extend([['max', feature.max()],
                     ['min', feature.min()],
                     ['mean', feature.mean()],
                     ['std', feature.std()]])
    output = pd.DataFrame(info, columns=['index', feature.name])
    output = output.set_index('index')
    return pd.DataFrame(output)

# Distribution Of Numerical Features Using Histplot
def num_dist_histplot(data, nrows=1, ncols=1, figsize=(5,5), fontsize=10, bins=30, limit=-1):
    '''
    Using histplot to show the general distribution of at least one numerical feature
    
    Parameters:
    - data (pandas.DataFrame): A dataframe with all numerical features needed to be plot
    - nrows (int) Number of rows of subplots,  default=1
    - ncols (int): Number of cols of subplots, default=1
    - figsize (tuple): A tuple contain two values (width, height) of a figure, default=(5, 5)
    - fontsize (int or float): Size of any text appear in the plot, default=10 
    - bins (int): Number of bins, default=30
    - limit (int): The actual number of features needed to be plotted. If your actual features needed to be plotted is less than the subplots created, you can limit the subplots used. Default=-1, that means the actual number of features are equal to the number of subplots.
        
    '''
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    cols = list(data.columns)
    index = 0
    if nrows == 1 and ncols == 1:
        sns.histplot(data[cols[index]].sort_values(), bins=bins, ax=ax)
        ax.set_ylabel("")
        ax.set_xlabel(xlabel=cols[index], fontsize=fontsize)
    elif (nrows == 1) or (ncols == 1):
        for i in ax:
            i.tick_params(labelsize=fontsize)
            sns.histplot(data[cols[index]].sort_values(), bins=bins, ax=i)
            i.set_ylabel("")
            i.set_xlabel(xlabel=cols[index], fontsize=fontsize)
            index += 1
            if index == limit:
                break
    else:
        for row in ax:
            for i in row:
                i.tick_params(labelsize=fontsize)
                sns.histplot(data[cols[index]].sort_values(), bins=bins, ax=i)
                i.set_ylabel("")
                i.set_xlabel(xlabel=cols[index], fontsize=fontsize)
                index += 1
                if index == limit:
                    break

# Distribution Of Numerical Features Using Boxplot                    
def num_dist_boxplot(data, nrows=1, ncols=1, figsize=(5,5), fontsize=10, limit=-1):
    '''
    Using boxplot to show the general distribution of at least one numerical feature, especially information of the outliers
    
    Parameters:
    - data (pandas.DataFrame): A dataframe with all numerical features needed to be plot
    - nrows (int) Number of rows of subplots,  default=1
    - ncols (int): Number of cols of subplots, default=1
    - figsize (tuple): A tuple contain two values (width, height) of a figure, default=(5, 5)
    - fontsize (int or float): Size of any text appear in the plot, default=10 
    - limit (int): The actual number of features needed to be plotted. If your actual features needed to be plotted is less than the subplots created, you can limit the subplots used. Default=-1, that means the actual number of features are equal to the number of subplots.
        
    '''
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    cols = list(data.columns)
    index = 0
    if nrows == 1 and ncols == 1:
        data.boxplot(column=cols[index], ax=ax, fontsize=fontsize)
    elif (nrows == 1) or (ncols == 1):
        for i in ax:
            data.boxplot(column=cols[index], ax=i, fontsize=fontsize)
            index += 1
            if index == limit:
                break
    else:
        for row in ax:
            for i in row:
                data.boxplot(column=cols[index], ax=i, fontsize=fontsize)
                index += 1
                if index == limit:
                    break
                
def num_kdeplot(data, cols, color, nrows=1, ncols=1, figsize=(5, 5), fontsize=10, limit=-1):
    '''
    Visualize the effect of some numerical features on the target using kdeplot.
    
    Parameters
    - data (pandas.DataFrame): A dataframe with all numerical features needed to be visualized
    - cols (list): A list of features name needed to be visualized
    - colors (str): Name of the feature that will be used as a categorical separation of data, allowing you to color the chart based on its unique values.
    - nrows (int): Number of rows of subplots, default=1
    - ncols (int): Number of cols of subplots, default=1
    - figsize (tuple): A tuple contain two values (width, height) of a figure, default=(5, 5)
    - fontsize (int or float): Size of any text appeared in the plot, default=10
    - limit (int): The actual number of features needed to be plotted. If your actual features needed to be plotted is less than the subplots created, you can limit the subplots used. Default=-1, that means the actual number of features are equal to the number of subplots.  
        
    '''
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    index = 0
    if nrows == 1 and ncols == 1:
        for val in data[color].unique():
            sns.kdeplot(data.loc[data[color] == val, cols[index]], label = f'{color} = {val}', ax=ax, common_norm=False)
        ax.tick_params(labelsize=fontsize)
        ax.set_ylabel("")
        ax.set_xlabel(xlabel=cols[index], fontsize=fontsize)
        ax.legend(fontsize=fontsize)
    elif (nrows == 1) or (ncols == 1):
        for i in ax:
            for val in data[color].unique():
                sns.kdeplot(data.loc[data[color] == val, cols[index]], label = f'{color} = {val}', ax=i, common_norm=False)
            i.tick_params(labelsize=fontsize)
            i.set_ylabel("")
            i.set_xlabel(xlabel=cols[index], fontsize=fontsize)
            i.legend(fontsize=fontsize)
            index += 1
            if index == limit:
                break
    else:
        for row in ax:
            for i in row:
                for val in data[color].unique():
                    sns.kdeplot(data.loc[data[color] == val, cols[index]], label = f'{color} = {val}', ax=i, common_norm=False)
                i.tick_params(labelsize=fontsize)
                i.set_ylabel("")
                i.set_xlabel(xlabel=cols[index], fontsize=fontsize)
                i.legend(fontsize=fontsize)
                index += 1
                if index == limit:
                    break
                    
def dist_base_barplot(data, cols, color, nrows=1, ncols=1, figsize=(5, 5), fontsize=10, limit=-1, norm=True):
    '''
    Visualize the distribution of some features based on the distribution of a specific features uisng barplot.
    
    Parameters
    - data (pandas.DataFrame): A dataframe with all features needed to be plot
    - col (str): A feature will appear on x-axis
    - colors (list): A list of features name that are used as a categorical separation of data, allowing you to color the bars based on their unique values.
    - nrows (int): Number of rows of subplots, default=1
    - ncols (int): Number of cols of subplots, default=1
    - figsize (tuple): A tuple contain two values (width, height) of a figure, default=(5, 5)
    - fontsize (int or float): Size of any text appeared in the plot, default=10
    - limit (int): The actual number of features needed to be plotted. If your actual features needed to be plotted is less than the subplots created, you can limit the subplots used. Default=-1, that means the actual number of features are equal to the number of subplots. 
    - norm (bool): default=True
        + If True, normalize the data before visualizing.
        + If False, using the original data to visualize.
        
    '''
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    index = 0
    if nrows == 1 and ncols == 1:
        temp = data.groupby(by=cols, as_index=False)[color[index]].value_counts(normalize=norm)
        sns.barplot(data=temp, x=cols, y=temp.columns[-1], hue=color[index], ax=ax)
        ax.tick_params(labelsize=fontsize)
        ax.set_ylabel("")
        ax.set_xlabel(xlabel=color[index], fontsize=fontsize)
        ax.legend(fontsize=fontsize, bbox_to_anchor=(1, 1))
    elif (nrows == 1) or (ncols == 1):
        for i in ax:
            temp = data.groupby(by=cols, as_index=False)[color[index]].value_counts(normalize=norm)
            sns.barplot(data=temp, x=cols, y=temp.columns[-1], hue=color[index], ax=i)
            i.tick_params(labelsize=fontsize)
            i.set_ylabel("")
            i.set_xlabel(xlabel=color[index], fontsize=fontsize)
            if data[color[index]].nunique() <= 10:
                i.legend(fontsize=fontsize, bbox_to_anchor=(1, 1), loc='upper left')
            index += 1
            if index == limit:
                break
    else:
        for row in ax:
            for i in row:
                temp = data.groupby(by=cols, as_index=False)[color[index]].value_counts(normalize=norm)
                sns.barplot(data=temp, x=cols, y=temp.columns[-1], hue=color[index], ax=i)
                i.tick_params(labelsize=fontsize)
                i.set_ylabel("")
                i.set_xlabel(xlabel=color[index], fontsize=fontsize)
                if data[color[index]].nunique() <= 10:
                    i.legend(fontsize=fontsize, bbox_to_anchor=(1, 1))
                index += 1
                if index == limit:
                    break

def continuous_plot(data, col, plot=['dist', 'box'], figsize=(20, 8)):
    '''
    Creates continuous plots for a specified column in a given DataFrame.
    
    Parameters:
    - data (DataFrame): The DataFrame containing the data.
    - col (str): The column in the DataFrame for which the plots are to be created.
    - plot (list of str): The type of plots to be created. It should be a list of either 'dist' or 'box'.
    - figsize (tuple of int): The size of the figure to be displayed.
    
    Returns:
    None: This function does not return any value.
    '''
    num_sub = len(plot)
    fig, ax = plt.subplots(1, num_sub, figsize=figsize)
    
    for i, p in enumerate(plot):
        if p == 'dist':
            # Plotting kernel density estimate for both target classes
            sns.kdeplot(data.loc[data['TARGET'] == 0, col], label='0', ax=ax[i])
            sns.kdeplot(data.loc[data['TARGET'] == 1, col], label='1', ax=ax[i])
            
            # Setting legend, xlabel, and ylabel
            ax[i].legend()
            ax[i].set_xlabel(col)
            ax[i].set_ylabel('Density')
        elif p == 'box':
            # Plotting boxplot for the specified column and target classes
            sns.boxplot(x=data['TARGET'], y=data[col], ax=ax[i])

def check_corr(data, heatmap=True, figsize=(20, 20)):
    '''
    Checks the correlation between features in a DataFrame.

    Parameters:
    - data (DataFrame): The input DataFrame containing numerical features.
    - heatmap (bool, optional): default=True
        + If True, displays a heatmap of the correlation matrix.
        + If False, returns the correlation matrix.

    Returns:
    DataFrame or None: Returns the correlation matrix if heatmap is False, otherwise, None.
    '''
    # Check if heatmap argument is True to create and display the heatmap
    if heatmap:
        # Create a mask for the upper triangle of the heatmap
        mask = np.triu(np.ones_like(data.corr()))

        # Create the heatmap figure
        plt.figure(figsize=figsize)

        # Generate the heatmap using seaborn, annotating the cells with correlation values
        sns.heatmap(data.corr(), cmap="YlGnBu", annot=True, mask=mask)

    # If heatmap argument is False, return the correlation matrix
    else:
        return data.corr()