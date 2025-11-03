import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
#import geopandas as gpd


def plot_categories(data, small_cat, big_cat, head, x,y):
    sns.set(style= "dark")

    palette = sns.color_palette("tab20", n_colors=len(data[big_cat].unique()))

    fig, axes = plt.subplots(int(len(data[big_cat].unique())/3) + 1, 3, figsize=(x, y))

    for ax, value, color in zip(axes.flatten(), data[big_cat].dropna().unique(), palette):
        
        data[small_cat][data[big_cat] == value].value_counts().head(head).plot(kind= "bar", ax=ax, color = color)
        ax.set_title(value, fontsize = 15)
        ax.tick_params(axis='x', rotation=0)
        ax.set_xlabel("")

    plt.suptitle(f"{small_cat} per {big_cat}", fontsize=18, y=1.01)
    plt.tight_layout()



def num_per_cat(data, numerical_var, cat_var, rotation = 0):
    sns.set()

    # Computing mean income per education level
    CLV_mean = data.groupby(cat_var)[numerical_var].mean().reset_index().sort_values(by=numerical_var, ascending= False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=CLV_mean, x=cat_var, y=numerical_var, hue = cat_var, legend=False)

    plt.title(f"Average {numerical_var} by {cat_var}")
    plt.xlabel(cat_var)
    plt.ylabel(numerical_var)
    plt.xticks()
    plt.tick_params(axis="x", rotation = rotation)
    plt.tight_layout()
    plt.show()