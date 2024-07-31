import karray as ka
import json
import os
from ..utils import _get_files_path_list, _convert_symbol_name_to_tuple
import zipfile


def symbol_parser_feather(folder: str, symbol_names: list=[], zip_extension=None, **kwargs):
    '''
    Parse all symbols from a folder and returns a dictionary
    '''

    tool_name = kwargs['tool_name'] if 'tool_name' in kwargs else 'dieter.jl'
    symbol_dict_with_value_type = {}
    for symbs in symbol_names:
        symb_tp = _convert_symbol_name_to_tuple(symbs)
        symbol_dict_with_value_type[symb_tp] = None

    file_list = _get_files_path_list(folder=folder, zip_extension=zip_extension, file_extension='feather')

    symbol_list = []
    for file in file_list:
        if zip_extension is not None:
            path_parts = file.split(zip_extension+os.sep)
            zip_fpath = path_parts[0] + zip_extension
            target_fpath = path_parts[1]
            with zipfile.ZipFile(zip_fpath, mode="r") as zip_io:
                with zip_io.open(target_fpath) as the_file:
                    symbol_info = _info_feather(the_file, tool_name)
        else:
            symbol_info = _info_feather(file, tool_name)
        if (symbol_info['symbol_name'], symbol_info['value_type']) in symbol_dict_with_value_type if len(symbol_dict_with_value_type) != 0 else True:
            symbol_dict = {}
            # This fields are mandatory for a parser
            symbol_dict['symbol_name'] = symbol_info['symbol_name']
            symbol_dict['value_type']  = symbol_info['value_type']
            symbol_dict['path']        = file
            symbol_dict['scenario_name'] = symbol_info['scenario_name']
            # Until here
            # you can add more (custom) attributes. It must be added also see handler.py def add_custom_attr() and be an attribute for loader
            symbol_dict['zip_extension'] = zip_extension
            symbol_list.append(symbol_dict)
    return symbol_list

def load_feather(path:str, zip_extension, **kwargs):
    '''
    Load custom feather file.
    '''
    if "use_threads" in kwargs:
        threads = kwargs["use_threads"]
    else:
        threads = True
    if "with_" in kwargs:
        with_ = kwargs["with_"]
    else:
        with_ = "pandas"
    if zip_extension is not None:
        path_parts = path.split(zip_extension+os.sep)
        zip_fpath = path_parts[0] + zip_extension
        target_fpath = path_parts[1]
        with zipfile.ZipFile(zip_fpath, mode="r") as zip_io:
            with zip_io.open(target_fpath) as the_file:
                arr_dict = ka.from_feather_to_dict(the_file, use_threads=threads, with_=with_)
        return arr_dict
    else:
        return ka.from_feather_to_dict(path, use_threads=threads, with_=with_)

def _info_feather(path:str, tool_name:str):
    '''
    Load symbol info from feather file.
    '''
    import pyarrow.feather as ft
    table = ft.read_table(path)
    meta_bstring = table.schema.metadata
    assert tool_name.encode() in meta_bstring, f"{tool_name=} does not exist as key for file metadata {list(meta_bstring)}"
    restored_meta_json = meta_bstring[tool_name.encode()]
    restored_meta = json.loads(restored_meta_json)
    symbol_name = restored_meta['symbol_name']
    value_type = restored_meta['value_type']
    scenario_id = restored_meta['scenario_name']
    return {'symbol_name': symbol_name, 'value_type': value_type, 'scenario_name': scenario_id}
