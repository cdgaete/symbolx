import glob
import os
import re
import uuid
import json
from typing import Callable
from .parser import load_scenario_info
import karray as ka
import pandas as pd

from .settings import settings, allowed_string


class DataCollection:
    def __init__(self):
        '''
        Collect scenarios per symbol
        '''
        self.collector = {}
        self.config_files = None
        self.config = None
        self.data = None
        self.symbol_name_list = None
        self.symbol_valuetype_dict = None
        self.short_names = None
        self.metadata_template = None
        self.scenarios_metadata = None
        self.symbols_book = None

    def add_collector(self, collector_name:str, parser:Callable, loader:Callable):
        self.collector[collector_name] = {}
        self.collector[collector_name]['parser'] = parser
        self.collector[collector_name]['loader'] = loader
        self.add_symbol_list(collector_name)
        self.add_custom_attr(collector_name)


    def add_folder(self, collector_name:str, folder:str):
        self.collector[collector_name]['folder'] = folder

    def add_symbol_list(self, collector_name:str, symbol_list:list=[]): # optional
        self.collector[collector_name]['symbol_list'] = symbol_list

    def add_custom_attr(self, collector_name:str, **kwargs):
        self.collector[collector_name]['custom'] = kwargs

    def adquire(self, id_integer=True):
        self.config_files = []
        self.config = {}
        self.data = []

        for collector in self.collector:
            folder = self.collector[collector]['folder']
            symbol_list = self.collector[collector]['symbol_list']
            custom_attr = self.collector[collector]['custom']
            parser = self.collector[collector]['parser']
            dyn_config_path = os.path.join(folder,"*/*_config.yml")
            config_files = glob.glob(dyn_config_path)
            self.config_files += config_files
            for symbol_info_dict in parser(folder, symbol_list):
                symbol_info_dict['collector'] = collector
                for attr in custom_attr:
                    symbol_info_dict[attr] = custom_attr[attr]
                self.data.append(symbol_info_dict)

        assert len(self.config_files) > 0, "No config files found"
        for config_file in self.config_files:
            config = load_scenario_info(config_file)
            self.config[config['id']] = config
        self.config = dict(sorted(self.config.items()))
        assert len(self.config) == len(self.config_files), "Config files with same id found"

        self._get_symbol_lists()
        self._scenario_name_shortener(id_integer)
        self._get_metadata_template()
        self._get_all_scenario_metadata()
        self._join_all_symbols()


    def _get_symbol_lists(self):
        list_of_symbols = []
        symbols_and_value_type = {}
        for symb_info in self.data:
            list_of_symbols.append(symb_info['symbol_name'])
            symbols_and_value_type[(symb_info['symbol_name'], symb_info['value_type'])] = None

        self.symbol_name_list = sorted(list(set(list_of_symbols)))
        self.symbol_valuetype_dict = dict(sorted(symbols_and_value_type.items()))

        return None
    
    def _scenario_name_shortener(self, id_integer=True):
        flag = False
        pattern = re.compile(r"(\d+)", re.IGNORECASE)
        names = []
        numbs = []
        shortnames = {}
        for scen in self.config:
            name = scen
            names.append(name)
            if pattern.search(name) != None:
                numbs.append(int(pattern.search(name)[0]))
            else:
                flag = True
        nrmax = max(numbs)
        for i in range(1,11):
            result = nrmax//10**i
            if result <= 1:
                digitM = i
                break

        names = sorted(names)
        number = len(names)
        for i in range(1,11):
            result = number//10**i
            if result <= 1:
                digitL = i
                break
        digit = max([digitL, digitM])
        if not flag:
            names_set = list(set(names))
            if len(names) == len(names_set):
                if len(names) == len(set(numbs)):
                    for name in names:
                        if id_integer:
                            shortname = int(pattern.search(name)[0])
                        else:
                            shortname = "S" + pattern.search(name)[0].zfill(digit)
                        shortnames[name] = shortname
                else:
                    flag = True
            else:
                flag = True
        if flag:
            for n, name in enumerate(names):
                if id_integer:
                    shortname = n
                else:
                    shortname = "S" + str(n).zfill(digit)
                shortnames[name] = shortname
        self.short_names = shortnames
        return None

    def _get_metadata_template(self):
        items_collector = []
        for scen in self.config:
            items_collector += list(self.config[scen]["config"])
        items = list(set(items_collector))
        self.metadata_template = {item:None for item in items}
        return None

    def _get_scenario_metadata(self, scen:str):
        scenario_metadata = {}
        for key in self.metadata_template:
            if key in self.config[scen]["config"]:
                scenario_metadata[key] = self.config[scen]["config"][key]
            else:
                scenario_metadata[key] = None
        return scenario_metadata

    def _get_all_scenario_metadata(self):
        all_scenario_metadata = {}
        for scen in self.config:
            all_scenario_metadata[scen] = self._get_scenario_metadata(scen)
        self.scenarios_metadata = all_scenario_metadata
        return None

    def _get_symbol_metadata(self, symbol_name:str, value_type:str):
        symbol_metadata = {}
        for scen in self.config:
            if (symbol_name,value_type) in self.symbol_valuetype_dict:
                symbol_metadata[scen] = self.scenarios_metadata[scen]
            else:
                print(f"{symbol_name} not found in {scen}")
                symbol_metadata[scen] = self.metadata_template
        return symbol_metadata

    def _join_scenarios_by_symbol(self, symbol_name:str, value_type:str='v'):
        """
        symbol
        """
        for data in self.data:
            if data['symbol_name'] == symbol_name and data['value_type'] == value_type:
                if self.symbols_book is None:
                    self.symbols_book = {}
                if (symbol_name, value_type) not in self.symbols_book:
                    self.symbols_book[(symbol_name, value_type)] = {}
                if 'short_names' not in self.symbols_book[(symbol_name, value_type)]:
                    self.symbols_book[(symbol_name, value_type)]['short_names'] = self.short_names
                if 'metadata' not in self.symbols_book[(symbol_name, value_type)]:
                    self.symbols_book[(symbol_name, value_type)]['metadata'] = self._get_metadata(self._get_symbol_metadata(symbol_name, value_type))
                if 'scenario_data' not in self.symbols_book[(symbol_name, value_type)]:
                    self.symbols_book[(symbol_name, value_type)]['scenario_data'] = {}
                self.symbols_book[(symbol_name, value_type)]['scenario_data'][data['scenario_id']] = data
        self.symbols_book[(symbol_name, value_type)]['scenario_data'] = dict(sorted(self.symbols_book[(symbol_name, value_type)]['scenario_data'].items()))

    def _get_metadata(self, raw_metadata):
        short_id = self.short_names
        dc = {}
        for k, v in raw_metadata.items():
            dc[short_id[k]] = {}
            for key, value in v.items():
                dc[short_id[k]][key] = value
        return pd.DataFrame(dc).transpose().to_dict()

    def _join_all_symbols(self):
        for symb in self.symbol_valuetype_dict:
            self._join_scenarios_by_symbol(*symb)
        return None

    def __repr__(self):
        return '''DataCollection()'''


class SymbolsHandler:
    def __init__(self, method:str, folder_path:str=None, obj:DataCollection=None):
        ''' 
        method: "folder" or "object"
        kwargs:
            folder_path: path to folder with symbol files
            object: DataCollection object
        '''
        assert isinstance(method, str), "Arg 'method' must be a string."
        assert method in ["folder", "object"], "Arg 'method' must be either 'folder' or 'object'"
        self.method = method
        self.folder_path = None
        self.symbols_book = None
        self.symbol_handler_token = str(uuid.uuid4()) # TODO: this can be changed by hashing the input file
        self.order = ['id']
        self.saved_symbols = {}
        self.token_info = None
        self.input_method(method=method, folder_path=folder_path, obj=obj)


    def input_method(self, method:str, **kwargs):
        if method == "object":
            self.from_object(obj=kwargs['obj'])
        elif method == "folder":
            self.from_folder(folder_path=kwargs['folder_path'])
        else:
            raise Exception('A method mus be provided from either "object" or "folder"')

    def from_object(self, obj:DataCollection):
        self.symbols_book = obj.symbols_book
        self.collector = obj.collector
        self.short_names = obj.short_names
        for n_v in self.symbols_book:
            settings.append((*n_v,self.symbol_handler_token))


    def from_folder(self, folder_path:str=None):
        self.token_info = {}
        self.folder_path = folder_path
        files = glob.glob(os.path.join(self.folder_path, "*.feather"))
        self.symbols_book = {}
        for file in files:
            loaded_file_dict = from_feather_dict(file)
            self.symbols_book[(loaded_file_dict['name'],loaded_file_dict['value_type'])] = file

            token = loaded_file_dict['symbol_handler_token']
            settings.append((loaded_file_dict['name'],loaded_file_dict['value_type'],loaded_file_dict['symbol_handler_token']))

            if token not in self.token_info:
                self.token_info[token] = {}
            self.token_info[token][(loaded_file_dict['name'],loaded_file_dict['value_type'])] = file
        if len(self.token_info) > 1:
            print(f"There are Symbols' files with different symbol_handler_token in {folder_path}")
            print("       Symbol's scenario short names or id's might be in conflict.")
            print("       Make sure all Symbols are from the same symbol_handler_token")
            print("       Otherwise, it may happen, for example, different scenarios have the same id.")
            print("       See 'token_info' attribute of SymbolsHandler for more information")


    def append(self, **kwargs):
        for name in kwargs:
            symbol = kwargs[name]
            assert len(set(name).difference(set(allowed_string))) == 0, f"Symbol name '{name}' contains special characters. Please, change the name of the symbol. Allowed chars are: {allowed_string}"
            symbol.name = name
            self.saved_symbols[(name, symbol.value_type)] = symbol

    def save(self, folder_path=None):
        if folder_path is None:
            assert self.folder_path is not None, "folder_path must be provided"
            folder_path = self.folder_path
        os.makedirs(folder_path, exist_ok=True)
        for symbol in self.saved_symbols.values():
            file_name = f"{symbol.name}.{symbol.value_type}.{symbol.symbol_handler_token}.feather"
            file_path = os.path.join(folder_path,file_name)
            symbol.to_feather(file_path)

    def get_info(self, symbol_name, value_type):
        assert (symbol_name, value_type) in self.symbols_book, f"Pairs {(symbol_name, value_type)} not present in the symbols_book attribute of SymbolsHandler"
        if isinstance(self.symbols_book[(symbol_name, value_type)], dict):
            return self.symbols_book[(symbol_name, value_type)]
        elif isinstance(self.symbols_book[(symbol_name, value_type)], str):
            return self.symbols_book[(symbol_name, value_type)]

    def __repr__(self):
        return f'''SymbolsHandler(method='{self.method}')'''

# TODO: metadata dictionary keys are always strings. Needs to be fixed
def from_feather_dict(path):
    arr, restored_table = ka.from_feather(path, with_table=True)
    custom_meta_key = 'symbolx'
    restored_meta_json = restored_table.schema.metadata[custom_meta_key.encode()]
    restored_meta = json.loads(restored_meta_json)
    return dict(array=arr,**restored_meta)