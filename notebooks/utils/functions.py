import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import silhouette_score, make_scorer
from scipy.stats import randint, uniform

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
    sns.set(style= "darkgrid")

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

def clustering_scorer(estimator, X):
    """Custom scorer for clustering that uses transformed data"""
    try:
        # Fit the estimator
        labels = estimator.fit_predict(X)
        
        # Check for valid clustering
        n_clusters = len(set(labels))
        if n_clusters < 2:
            print(f"Warning: Only {n_clusters} cluster(s) found")
            return -1
        
        if n_clusters >= len(X):
            print(f"Warning: Too many clusters ({n_clusters})")
            return -1
        
        # Use the TRANSFORMED data stored in the pipeline
        X_transformed = estimator.X_
        
        # Additional check: ensure we have enough samples
        if len(X_transformed) < 2:
            print("Warning: Not enough samples after transformation")
            return -1
        
        score = silhouette_score(X_transformed, labels)
        return score
        
    except Exception as e:
        print(f"Scoring error: {e}")
        return -1  # Return bad score instead of nan
