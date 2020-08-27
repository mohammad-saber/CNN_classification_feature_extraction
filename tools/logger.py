
import numpy as np
import pandas as pd
import codecs


def save_txt(txt_path, *args):
    '''
    Log a text file
    '''
    with codecs.open(txt_path, "a", "utf-8") as log_file:   # "a" : append
        for line in args:
            log_file.write(line + "\n")


def save_excel(excel_path, args):
    """
    logger function to write input data to an excel file.
    excel_path: path to save excel file
    args: List containing excel sheet name and data. Data structure is a dictionary. [("sheet name", dictionary)]
    Dictionary key is used as column name, and dictionary value is used as data. 
    """
    excel_writer = pd.ExcelWriter(excel_path)

    for (sheet_name, data) in args:
        result_excel = pd.DataFrame.from_dict(data=data)
        result_excel.to_excel(excel_writer, sheet_name, index=False)
    excel_writer.save()


def save_csv(csv_path, data):
    '''
    Save data in a csv file.
    data is a Numpy array
    ''' 
    with open(csv_path, 'a') as f:
        # x_numpy.tofile(f, sep=',', format='%10.5f')
        np.savetxt(f, data, delimiter=",", fmt='%.5f')   # it saves only 5 digts after decimal point to reduce csv file size
        # np.savetxt(f, data, delimiter=",")


