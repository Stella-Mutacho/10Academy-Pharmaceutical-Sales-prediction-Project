import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
# Importing the SimpleImputer class from sklearn
from sklearn.impute import SimpleImputer


class CleanDataFrame:

    @staticmethod
    def get_numerical_columns(df: pd.DataFrame) -> list:
        numerical_columns = df.select_dtypes(include='number').columns.tolist()
        return numerical_columns

    @staticmethod
    def get_categorical_columns(df: pd.DataFrame) -> list:
        categorical_columns = df.select_dtypes(
            include=['object']).columns.tolist()
        return categorical_columns
    def fix_datatypes(self, df: pd.DataFrame, column: str = None, to_type: type = None) -> pd.DataFrame:
        """
        Takes in the train and test dataframes and casts columns to proper data types.
        Time -> from string to datetime.
        """
        datetime_columns = ['Date' ]
        string_columns = []
        df_columns = df.columns
        for col in string_columns:
            if col in df_columns:
                df[col] = df[col].astype(str)
        for col in datetime_columns:
            if col in df_columns:
                df[col] = pd.to_datetime(df[col])
        if column and to_type:
            df[column] = df[column].astype(to_type)

        return df
    def percent_missing(self, df):
        """
        Print out the percentage of missing entries in a dataframe
        """
        # Calculate total number of cells in dataframe
        totalCells = np.product(df.shape)

        # Count number of missing values per column
        missingCount = df.isnull().sum()

        # Calculate total number of missing values
        totalMissing = missingCount.sum()

        # Calculate percentage of missing values
        print("The dataset contains", round(
            ((totalMissing/totalCells) * 100), 2), "%", "missing values.")

    def get_mct(self, series: pd.Series, measure: str):
        """
        get mean, median or mode depending on measure
        """
        measure = measure.lower()
        if measure == "mean":
            return series.mean()
        elif measure == "median":
            return series.median()
        elif measure == "mode":
            return series.mode()[0]

    def replace_missing(self, df: pd.DataFrame, columns: str, method: str) -> pd.DataFrame:

        for column in columns:
            nulls = df[column].isnull()
            indecies = [i for i, v in zip(nulls.index, nulls.values) if v]
            replace_with = self.get_mct(df[column], method)
            df.loc[indecies, column] = replace_with

        return df
    def fill_missing_median(df):
        """
        # Function to fill categorical nulls with median
        """
        col=['CompetitionDistance','CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']
        for column in col:
            imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
            imp_median = imp_median.fit(df[[column]])
            df[column] = imp_median.transform(df[[column]]).ravel()
    
        return df
    
    def fill_missing_zero(df):
        """
        # Function to fill categorical nulls with 0
        """
        col=['Promo2SinceWeek','Promo2SinceYear']
        for column in col:
            imp_median = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
            imp_median = imp_median.fit(df[[column]])
            df[column] = imp_median.transform(df[[column]]).ravel()
        
        return df
    def fill_missing_mode(df):
        """
        # Function to fill categorical nulls with mode
        """
        col=['PromoInterval']
        for column in col:
            imp_median = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            imp_median = imp_median.fit(df[[column]])
            df[column] = imp_median.transform(df[[column]]).ravel()
    
        return df

    def remove_null_row(self, df: pd.DataFrame, columns: str) -> pd.DataFrame:
        for column in columns:
            df = df[~ df[column].isna()]

        return df

    def normal_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        scaller = StandardScaler()
        scalled = pd.DataFrame(scaller.fit_transform(
            df[self.get_numerical_columns(df)]))
        scalled.columns = self.get_numerical_columns(df)

        return scalled

    def minmax_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        scaller = MinMaxScaler()
        scalled = pd.DataFrame(
            scaller.fit_transform(
                df[self.get_numerical_columns(df)]),
            columns=self.get_numerical_columns(df)
        )

        return scalled

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        normalizer = Normalizer()
        normalized = pd.DataFrame(
            normalizer.fit_transform(
                df[self.get_numerical_columns(df)]),
            columns=self.get_numerical_columns(df)
        )

        return normalized

    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This checkes if there are any duplicated entries for a user
        And remove the duplicated rows
        """
        df = df.drop_duplicates(subset='Id')

        return df

    def date_to_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This converts the date column into the day of the week
        """
        df['Day'] = pd.to_datetime(df['Date']).dt.day_name().values

        return df
    def date_to_month(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This converts the date column into the month
        """
        df['Month'] = df['Date'].apply(lambda x: int(str(x)[5:7]))

        return df
    def date_to_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This converts the date column into the year 
        """
        df['Month'] = df['Date'].apply(lambda x: int(str(x)[:4]))

        return df
    def holidays(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This generates a column showing if the date is that holiday or not holiday 
        """
        df['Month'] = df['Date'].apply(lambda x: int(str(x)[:4]))
        #handling categorical attributes(by creating a binary feature for each category)
        df['IsPublicHoliday'] = df['StateHoliday'].map(lambda x: 1 if x=='a' else 0)
        df['IsEasterHoliday'] = df['StateHoliday'].map(lambda x: 1 if x=='b' else 0)
        df['IsChristmasHoliday'] = df['StateHoliday'].map(lambda x: 1 if x=='c' else 0)
        

        return df
    def day_to_weekday(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This converts the day column into the description of weekend or weekend 
        """
        df['IsWeekend'] = df['Day'].map(lambda x: 1 if x=='Saturday' or x=='Sunday' else 0)
        df['IsWeekday'] = df['Day'].map(lambda x: 1 if x!='Saturday' or x!='Sunday' else 0)

        return df
    def drop_columns(self, df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
        """
        Drops columns that are not essesntial for modeling
        """
        if not columns:
            columns = ['auction_id', 'date', 'yes', 'no', 'device_make']
        df.drop(columns=columns, inplace=True)

        return df
    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This runs a series of cleaner methods on the df passed to it. 
        """
        df = self.date_to_day(df)
        df = self.date_to_month(df)
        df = self.date_to_year(df)
        df = self.fill_missing_mode(df)
        df = self.fill_missing_zero(df)
        df = self.holidays(df)
        df = self.day_to_weekday(df)
        df.reset_index(drop=True, inplace=True)
        return df
    


    