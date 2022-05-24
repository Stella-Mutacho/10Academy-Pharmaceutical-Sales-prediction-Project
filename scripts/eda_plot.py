import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns




class Plot:
    def __init__(self) -> None:
        """Initilize class."""
        pass
    def plot_hist(self, df: pd.DataFrame, column: str, color: str) -> None:
        """Plot the hist of the column.

        Args:
            df (pd.DataFrame): Dataframe to be plotted.
            column (str): column to be plotted.
            color (str): color of the histogram.
        """
        # plt.figure(figsize=(15, 10))
        # fig, ax = plt.subplots(1, figsize=(12, 7))
        sns.displot(data=df, x=column, color=color,
                    kde=True, height=7, aspect=2)
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        plt.show()

    def plot_count(self, df: pd.DataFrame, column: str) -> None:
        """Plot the count of the column.

        Args:
            df (pd.DataFrame): Dataframe to be plotted.
            column (str): column to be plotted.
        """
        plt.figure(figsize=(12, 7))
        sns.countplot(data=df, x=column)
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        plt.show()

    def plot_bar(self, df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str) -> None:
        """Plot bar of the column.

        Args:
            df (pd.DataFrame): Dataframe to be plotted.
            x_col (str): column to be plotted.
        """
        plt.figure(figsize=(12, 7))
        sns.barplot(data=df, x=x_col, y=y_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.show()

    def plot_heatmap(self, df: pd.DataFrame, title: str, cbar=False) -> None:
        """Plot Heat map of the dataset.

        Args:
            df (pd.DataFrame): Dataframe to be plotted.
            title (str): title of chart.
        """
        # num_cols = df.select_dtypes(include=np.number).columns
        plt.figure(figsize=(12, 7))
        sns.heatmap(df, annot=True, cmap='viridis', vmin=0,
                    vmax=1, fmt='.2f', linewidths=.7, cbar=cbar)
        plt.title(title, size=18, fontweight='bold')
        plt.show()

    def plot_box(self, df: pd.DataFrame, x_col: str, title: str) -> None:
        """Plot box chart of the column.

        Args:
            df (pd.DataFrame): Dataframe to be plotted.
            x_col (str): column to be plotted.
            title (str): title of chart.
        """
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=df, x=x_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.show()

    def plot_box_multi(self, df: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
        """Plot the box chart for multiple column.

        Args:
            df (pd.DataFrame): Dataframe to be plotted.
            column (str): column to be plotted.
        """
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=df, x=x_col, y=y_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def plot_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, title: str, hue: str, style: str) -> None:
        """Plot Scatter chart of the data.

        Args:
            df (pd.DataFrame): Dataframe to be plotted.
            column (str): column to be plotted.
        """
        plt.figure(figsize=(12, 7))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, style=style)
        plt.title(title, size=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def plot_pie(self, data, labels, title) -> None:
        """Plot pie chart of the data.

        Args:
            data (list): Data to be plotted.
            labels (list): labels of the data.
            colors (list): colors of the data.
        """
        plt.figure(figsize=(12, 7))
        colors = sns.color_palette('bright')
        plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')
        plt.title(title, size=20)
        plt.show()
