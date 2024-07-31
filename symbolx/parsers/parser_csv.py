import numpy as np
import json
import csv
import os
import zipfile
from io import TextIOWrapper
from ..utils import _get_files_path_list, _convert_symbol_name_to_tuple


def symbol_parser_csv(folder: str, symbol_names: list=[], zip_extension=None, **kwargs):
    '''
    Parse all symbols from a folder and returns a dictionary
    '''
    symbol_dict_with_value_type = {}
    for symbs in symbol_names:
        symb_tp = _convert_symbol_name_to_tuple(symbs)
        symbol_dict_with_value_type[symb_tp] = None

    file_list = _get_files_path_list(folder=folder, zip_extension=zip_extension, file_extension='csv')
    symbol_list = []
    for file in file_list:
        symbol_name = os.path.splitext(os.path.basename(file))[0]
        if zip_extension is not None:
            scenario_name = os.path.splitext(os.path.basename(os.path.dirname(file)))[0]
        else:
            scenario_name = os.path.basename(os.path.dirname(file))
        if (symbol_name, 'v') in symbol_dict_with_value_type if len(symbol_dict_with_value_type) != 0 else True:
            symbol_dict = {}
            # This fields are mandatory for a parser
            symbol_dict['symbol_name'] = symbol_name
            symbol_dict['value_type']  = 'v'
            symbol_dict['path']        = file
            symbol_dict['scenario_name'] = scenario_name
            # Until here
            # you can add more (custom) attributes. It must be added also see handler.py def add_custom_attr() and be an attribute for loader
            symbol_dict['zip_extension'] = zip_extension
            symbol_list.append(symbol_dict)
    return symbol_list

def load_csv(path:str, symbol_name:str, zip_extension=None, **kwargs):
    '''
    Load custom csv file.
    '''
    if zip_extension is not None:
        path_parts = path.split(zip_extension+os.sep)
        zip_fpath = path_parts[0] + zip_extension
        file_name = symbol_name
        file_data = f"{file_name}.csv"
        file_coords = f"{file_name}.json"
        with zipfile.ZipFile(zip_fpath, mode="r") as zip_io:
            with zip_io.open(file_data) as the_file:
                CSV_IO = TextIOWrapper(the_file, 'utf-8')
                reader = csv.reader(CSV_IO, delimiter=',')
                for row in reader:
                    file_load_headers = row
                    break
                file_load_dims = file_load_headers[:-1] # remove last column
            with zip_io.open(file_data) as the_file:
                CSV_IO = TextIOWrapper(the_file, 'utf-8')
                file_load_data =  np.loadtxt(CSV_IO, delimiter=',', skiprows=1, dtype=np.object_)
            index = {}
            for i, dim in enumerate(file_load_dims):
                arr = file_load_data[:,i]
                if np.char.isnumeric(arr[0]):
                    arr = arr.astype(np.integer)
                index[dim] = arr
            if len(file_load_dims) > 0:
                value = file_load_data[:,-1].astype('float64')
            else:
                value = file_load_data.astype('float64')
            if file_coords in zip_io.filelist:
                with zip_io.open(file_coords) as jsonfile:
                    coords = json.load(jsonfile)
                if len(coords) == 0:
                    coords = None
            else:
                coords=None
        return {'data': (index, value), 'coords': coords}
    else:
        folder = os.path.dirname(path)
        file_name = symbol_name
        file_data = f"{file_name}.csv"
        file_coords = f"{file_name}.json"
        path_data = os.path.join(folder, file_data)
        path_coords = os.path.join(folder, file_coords)
        with open(path_data,'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                file_load_headers = row
                break
        file_load_dims = file_load_headers[:-1] # remove last column
        file_load_data =  np.loadtxt(path_data, delimiter=',', skiprows=1, dtype=object) # np.genfromtxt()
        index = {}
        for i, dim in enumerate(file_load_dims):
            arr = file_load_data[:,i]
            if np.char.isnumeric(arr[0]):
                arr = arr.astype(np.integer)
            index[dim] = arr
        if file_load_dims:
            value = file_load_data[:,-1].astype('float64')
        else:
            value = file_load_data.astype('float64')
        if os.path.exists(path_coords):
            with open(path_coords,'r') as jsonfile:
                coords = json.load(jsonfile)
        else:
            coords=None
        return {'data': (index, value), 'coords': coords}

