
import yaml
import pyarrow.feather as ft
import numpy as np
import glob
import os


def load_scenario_info(path):
    '''
    Load scenario info from yaml file.
    '''
    with open(path,'r') as stream:
        info = yaml.load(stream,Loader=yaml.FullLoader)
    return info


#### Load symbols from arrow - feather file ####

def _unflatten_dict(a, result = None, sep = '_'):

    if result is None:
        result = dict()
    for k, v in a.items():
        if isinstance(k, str):
            k, *rest = k.split(sep, 1)
        elif isinstance(k, int):
            rest = []
        if rest:
            if rest[0].isdigit():
                _unflatten_dict({int(rest[0]): v}, result.setdefault(k, {}), sep = sep)
            else:
                _unflatten_dict({rest[0]: v}, result.setdefault(k, {}), sep = sep)
        else:
            result[k] = v
    return result

def _metadata_manipulation(metadata, sep = '.'):
    meta_dict = _unflatten_dict(metadata, sep = sep)
    new_dict = {}
    for key, value in meta_dict.items():
        if isinstance(value, dict):
            new_dict[key] = _metadata_manipulation(value, sep = sep)
        else:
            new_dict[key] = value
    return new_dict

def _sort_metadata(meta_dict):
    new_dict = {}
    for key, value in meta_dict.items():
        if isinstance(value, dict):
            keys_sorted_list = sorted(list(value.keys()))
            if all([isinstance(k, int) for k in keys_sorted_list]):
                new_dict[key] = [value[i] for i in keys_sorted_list]
            else:
                new_dict[key] = _sort_metadata(value)
        else:
            new_dict[key] = value
    return new_dict

def _get_metadata(metadata, sep = '.'):
    meta_dict = _metadata_manipulation(metadata, sep = sep)
    meta_dict = _sort_metadata(meta_dict)
    for key, value in meta_dict.items():
        if key == "coords":
            meta_dict[key] = {dim: meta_dict[key][dim] for dim in meta_dict['dims']}
    return meta_dict

def load_feather(path:str):
    '''
    Load custom feather file.
    '''
    table = ft.read_table(path)
    meta_bstring = table.schema.metadata
    meta_strings = {key.decode('utf-8'): value.decode('utf-8') for (key, value) in meta_bstring.items()}
    meta_custom = {}
    for (key, value) in meta_strings.items():
        k = key.split('.')
        if k[0] in ['symbol','value_type','dims','coords','scenario_id']:
            meta_custom[key] = value
    metadata = _get_metadata(meta_custom, sep = '.')
    coo = table.to_pandas().to_numpy()
    return {'symbol_name': metadata['symbol'], 
            'coo':coo, 
            'dims': metadata['dims'], 
            'coords': metadata['coords'], 
            'value_type': metadata['value_type']}

def _info_feather(path:str):
    '''
    Load symbol info from feather file.
    '''
    table = ft.read_table(path)
    meta_bstring = table.schema.metadata
    symbol_name = meta_bstring[b'symbol'].decode('utf-8')
    value_type = meta_bstring[b'value_type'].decode('utf-8')
    scenario_id = meta_bstring[b'scenario_id'].decode('utf-8')
    return {'symbol_name': symbol_name, 'value_type': value_type, 'scenario_id': scenario_id}

def symbol_parser_feather(folder: str, symbol_names: list=[]):
    '''
    Parse all symbols from a folder and returns a dictionary
    '''
    symbol_dict_with_value_type = {}
    for symbs in symbol_names:
        symb_tp = _convert_symbol_name_to_tuple(symbs)
        symbol_dict_with_value_type[symb_tp] = None

    file_list = glob.glob(os.path.join(folder,'*/*.feather'))
    symbol_list = []
    for file in file_list:
        symbol_info = _info_feather(file)
        if (symbol_info['symbol_name'], symbol_info['value_type']) in symbol_dict_with_value_type if len(symbol_dict_with_value_type) != 0 else True:
            symbol_dict = {}
            symbol_dict['symbol_name'] = symbol_info['symbol_name']
            symbol_dict['value_type']  = symbol_info['value_type']
            symbol_dict['scenario_id'] = symbol_info['scenario_id']
            symbol_dict['path']        = file
            symbol_list.append(symbol_dict)
    return symbol_list

def _convert_symbol_name_to_tuple(symbol_name: str):
    '''
    Convert symbol name to tuple.
    '''
    symb_list = symbol_name.split('.')
    if len(symb_list) == 1:
        symb_tp = (symb_list[0],'v')
    elif len(symb_list) == 2:
        symb_tp = (symb_list[0], symb_list[1])
    else:
        raise ValueError(f"Symbol name '{symbol_name}' is not valid")
    return symb_tp


def symbol_parser_gdx(folder: str, symbol_names: list=[]):
    '''
    Parse all symbols from a folder and returns a dictionary
    '''
    symbol_dict_with_value_type = {}
    for symbs in symbol_names:
        symb_tp = _convert_symbol_name_to_tuple(symbs)
        symbol_dict_with_value_type[symb_tp] = None

    file_list = glob.glob(os.path.join(folder,'*/*.gdx'))
    symbol_list = []
    for file in file_list:
        scen_id = os.path.basename(file).split('.')[0]
        for (name, symb_type, nrdims) in _symbols_list_from_gdx(file):
            if symb_type == 0: # set
                options = []
            elif symb_type == 1: # parameter
                options = ['v']
            elif symb_type == 2: # variable
                options = ['v', 'm']
            elif symb_type == 3: # equation
                options = ['v', 'm']
            for value_type in options:
                symb_tp = (name, value_type)
                if symb_tp in symbol_dict_with_value_type:
                    symbol_list.append({'symbol_name':symb_tp[0],
                                        'value_type':symb_tp[1],
                                        'scenario_id':scen_id,
                                        'path':file})
                else:
                    if not symbol_names:
                        symbol_list.append({'symbol_name':symb_tp[0],
                                            'value_type':symb_tp[1],
                                            'scenario_id':scen_id,
                                            'path':file})
    return symbol_list


#### Load symbols from gdx file ####

def set_gams_dir(gams_dir: str = None):
    """ 
    This function will add GAMS.exe temporarily to the PATH environment variable.

    WARNING: An incorrect path may cause python crashes!!!. Make sure GAMS path is correct.
    """

    from gdxcc import (
        gdxCreateD,
        new_gdxHandle_tp,
        gdxClose,
        gdxFree,
        GMS_SSSIZE,
    )

    gdxHandle = new_gdxHandle_tp()
    gdxCreateD(gdxHandle, gams_dir, GMS_SSSIZE)
    gdxClose(gdxHandle)
    gdxFree(gdxHandle)
    return True


def _symbols_list_from_gdx(filename: str = None, gams_dir: str = None):
    """ It returns a list of symbols' names contained in the GDX file

    Args:
        gams_dir (str, optional): GAMS.exe path, if None the API looks at environment variables. Defaults to None.
        filename (str, optional): GDX filename. Defaults to None.

    Raises:
        Exception: GDX file does not exist or is failed

    Returns:
        list: a list of symbol's names contained in the GDX file
    """

    from gdxcc import (
        gdxSystemInfo,
        gdxSymbolInfo,
        gdxCreateD,
        gdxOpenRead,
        gdxDataReadDone,
        new_gdxHandle_tp,
        gdxClose,
        gdxFree,
        GMS_SSSIZE,
    )

    gdxHandle = new_gdxHandle_tp()
    gdxCreateD(gdxHandle, gams_dir, GMS_SSSIZE)
    gdxOpenRead(gdxHandle, filename)
    exists, nSymb, nElem = gdxSystemInfo(gdxHandle)
    symbols = []
    for symNr in range(nSymb):
        ret, name, nrdims, symb_type = gdxSymbolInfo(gdxHandle, symNr)
        symbols.append((name, symb_type, nrdims))
    gdxDataReadDone(gdxHandle)
    gdxClose(gdxHandle)
    gdxFree(gdxHandle)
    return symbols

def _gdx_get_symbol_array_str(symbol_name: str, gdx_file: str,  gams_dir: str=None):
    from gams2numpy import Gams2Numpy

    g2np = Gams2Numpy(gams_dir)
    uel_map = g2np.gdxGetUelList(gdx_file)
    arr = g2np.gdxReadSymbolStr(gdx_file, symbol_name,uel_map)
    return arr

def _gdx_get_symbol_data_dict(symbol_name: str, gdx_file: str, gams_dir: str=None):

    from gdxcc import (
        gdxSymbolInfo,
        gdxFindSymbol,
        gdxSymbolGetDomainX,
        gdxSymbolInfoX,
        gdxCreateD,
        gdxOpenRead,
        new_gdxHandle_tp,
        gdxClose,
        gdxFree,
        GMS_SSSIZE,
    )

    gdxHandle = new_gdxHandle_tp()
    ret, msg = gdxCreateD(gdxHandle, gams_dir, GMS_SSSIZE)
    ret, msg = gdxOpenRead(gdxHandle, gdx_file)
    assert ret, f"Failed to open '{gdx_file}'"
    ret, symidx = gdxFindSymbol(gdxHandle, symbol_name)
    assert ret, f"Symbol {symbol_name} not found in {gdx_file}"
    if not ret:
        return None
    _, name, NrDims, data_type = gdxSymbolInfo(gdxHandle, symidx)
    _, gdx_domain = gdxSymbolGetDomainX(gdxHandle, symidx)
    _, NrRecs, _, description = gdxSymbolInfoX(gdxHandle, symidx)
    gdxClose(gdxHandle)
    gdxFree(gdxHandle)

    data = {}
    data['symbol'] = symbol_name
    data['dims'] = gdx_domain
    data['coords'] = {dim: list(np.sort(_gdx_get_symbol_array_str(symbol_name=dim, gdx_file=gdx_file, gams_dir=gams_dir)[:,0])) for dim in gdx_domain}
    return data

def load_gdx(symbol_name: str, value_type: str='v', gdx_file: str='', gams_dir: str= None):
    '''
    Load custom GDX file.

    Parameters
    ----------
    symbol_name : str
        Name of the symbol to be extracted.
    value_type : str, optional
        Type of the symbol to be extracted. The default is 'v'.
    gdx_file : str
        Path to the gdx file.
    gams_dir : str, optional

    '''
    value_types = {'v':0, 'm':1, 'lo':2, 'up':3, 'scale':4}
    assert value_type in value_types.keys(), f'value_type must be one of the following: {value_types.keys()}'
    metadata = _gdx_get_symbol_data_dict(symbol_name=symbol_name, gdx_file=gdx_file, gams_dir=gams_dir)
    symbol = _gdx_get_symbol_array_str(symbol_name=symbol_name, gdx_file=gdx_file, gams_dir=gams_dir)
    nrdims = len(metadata['dims'])
    col_index = nrdims + value_types[value_type]
    raw_coo = symbol[:, list(range(nrdims)) + [col_index]]
    mask = raw_coo[:,nrdims] != 0.0
    coo = raw_coo[mask,:]
    return {'symbol_name': symbol_name,
            'coo':coo,
            'dims': metadata['dims'],
            'coords': metadata['coords'],
            'value_type': value_type}



