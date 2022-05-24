import json
import pandas as pd


class ReadData:
    
    """
    this class contains functions to parse json, excel and csv data into a pandas dataframe
    
    Return
    ------
    dataframe
    """
    def init():
        pass
    def read_json(self,json_file: str, dfExctractor)->pd.DataFrame:
        """
        json file reader to open and read json files into a dataframe
        Args:
        -----
        json_file: str - path of a json file

        Returns
        -------
        A dataframe of the json file
        """

        data = []
        for item in open(json_file,'r'):
            data.append(json.loads(item))
        return dfExtractor(data)
    
    def read_csv(self, csv_file):
        """
        csv file reader to open and read csv files into a dataframe
        Args:
        -----
        csv_file: str - path of the csv file

        Returns
        -------
        A dataframe of the csv file
        """
        data = pd.read_csv(csv_file)
        return data
    
    def read_excel(self,excel_file, startRow=0)->pd.DataFrame:
        """
        excel file reader to open and read excel files into a dataframe
        Args:
        -----
        excel_file: str - path of the excel file

        Returns
        -------
        A dataframe of the excel file
        """
        data = pd.read_excel(excel_file)
        return data
    
