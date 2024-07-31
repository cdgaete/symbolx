import re
from typing import Callable
from .utils import load_scenario_info, _get_files_path_list, select_scenario_files


class DataCollection:
    def __init__(self):
        '''
        Collect relevant data from raw scenarios output and organizes it per symbol or variable. No input is required. 
        This class defines a collector with custom parsers and loaders of the raw scenarios output.
        It also requires custom attributes that are the parser and loader input arguments.
        This structure helps to use personalized parsers and loaders depending on the type and format of the optimization tools' output.
        
        Examples
        --------
        Example 1: parser and loader from arrow tables and feather file format

            >>> import symbolx as syx
            >>> from symbolx import DataCollection
            >>> DC = DataCollection()
            >>> DC.add_collector(collector_name='opt_model', parser=syx.symbol_parser_feather, loader=syx.load_feather)
            >>> DC.add_folder(collector_name='opt_model', './raw_model_output')
            >>> DC.add_custom_attr(collector_name='opt_model', with_='pandas')
            >>> DC.adquire(id_integer=True, zip_extension=None)


        Example 2: parser and loader from GAMS GDX file format

            >>> import os
            >>> import symbolx as syx
            >>> from symbolx import DataCollection
            >>> DC = DataCollection()
            >>> DC.add_collector(collector_name='gdx_collector', parser=syx.symbol_parser_gdx, loader=syx.load_gdx)
            >>> DC.add_folder('gdx_collector', './gdx_output')
            >>> DC.add_custom_attr(collector_name='gdx_collector', inf_to_zero=True, verbose=False)
            >>> DC.adquire(id_integer=False, gams_dir=os.getenv("GAMS_DIR"), zip_extension=None)

        
        '''
        self.collector = {}
        self.scenarios_path = None
        self.config = None
        self.data = None
        self.symbol_name_list = None
        self.symbol_valuetype_dict = None
        self.short_names = None
        self.metadata_template = None
        self.scenarios_metadata = None
        self.symbols_book = None

    def add_collector(self, collector_name:str, parser:Callable, loader:Callable):
        '''
        Add a collector to the DataCollection object.

        Parameters
        ----------
        collector_name : str
            Define or set the name of the collector.
        parser : Callable
            The parser function.
        loader : Callable
            The loader function.
            
        Returns
        -------
        None.

        '''

        self.collector[collector_name] = {}
        self.collector[collector_name]['parser'] = parser
        self.collector[collector_name]['loader'] = loader
        self.add_symbol_list(collector_name)
        self.add_custom_attr(collector_name)

    def add_folder(self, collector_name:str, folder:str):
        '''
        Add a folder to the DataCollection object.

        Parameters
        ----------
        collector_name : str
            The name of the collector.
        folder : str
            The path of the folder.

        Returns
        -------
        None.

        '''

        self.collector[collector_name]['folder'] = folder

    def add_symbol_list(self, collector_name:str, symbol_list:list=[]): # optional
        '''
        Add a symbol list to the DataCollection object.
        
        Parameters
        ----------
        collector_name : str
            The name of the collector.
        symbol_list : list
            The symbol list.

        Returns
        -------
        None.

        '''

        self.collector[collector_name]['symbol_list'] = symbol_list

    def add_custom_attr(self, collector_name:str, **kwargs):
        '''
        Add custom attributes to the DataCollection object.

        Parameters
        ----------
        collector_name : str
            The name of the collector.
        kwargs : dict
            The custom attributes.

        Returns
        -------
        None.

        '''
        
        self.collector[collector_name]['custom'] = kwargs

    def adquire(self, id_integer=True, serializer='json', zip_extension=None, **kwargs):
        '''
        Adquire metadata from the collectors.

        Parameters
        ----------
        id_integer : bool
            A key feature is dealing with scenario results. The dimension 'id' represents the id of the scenarios. Each individual scenario id could be either numbered strings or integers.
        serializer : str
            Scenario metadata is provided through serialization formats. The default is 'json'. A JSON file has to be present in every scenario folder.
        zip_extension : str
            Scenario folders that contain files with optimization problem output can be contained as compressed files. The default is None. Options are 'zip' or '7z'.
        kwargs : dict
            The custom attributes.

        Returns
        -------
        None.

        '''

        self.scenarios_path = []
        self.config = {}
        self.data = []

        for collector in self.collector:
            folder = self.collector[collector]['folder']
            symbol_list = self.collector[collector]['symbol_list']
            custom_attr = self.collector[collector]['custom']
            parser = self.collector[collector]['parser']
            scenarios_path = _get_files_path_list(folder=folder, zip_extension=zip_extension, file_extension=serializer)
            self.scenarios_path += select_scenario_files(scenarios_path, zip_extension=zip_extension)
            for symbol_info_dict in parser(folder=folder, symbol_names=symbol_list, zip_extension=zip_extension, **kwargs):
                symbol_info_dict['collector'] = collector
                for attr in custom_attr:
                    symbol_info_dict[attr] = custom_attr[attr]
                self.data.append(symbol_info_dict)

        assert len(self.scenarios_path) > 0, "No scenario folder found"
        for scenario_path in self.scenarios_path:
            config = load_scenario_info(scenario_path,serializer=serializer, zip_extension=zip_extension)
            self.config[config['name']] = config
        self.config = dict(sorted(self.config.items()))
        assert len(self.config) == len(self.scenarios_path), "Config files with same id found"

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
            if pattern.search(name) is not None:
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
            items_collector += list(self.config[scen]["metadata"])
        items = list(set(items_collector))
        self.metadata_template = {item:None for item in items}
        return None

    def _get_scenario_metadata(self, scen:str):
        scenario_metadata = {}
        for key in self.metadata_template:
            if key in self.config[scen]["metadata"]:
                scenario_metadata[key] = self.config[scen]["metadata"][key]
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
                self.symbols_book[(symbol_name, value_type)]['scenario_data'][data['scenario_name']] = data
        self.symbols_book[(symbol_name, value_type)]['scenario_data'] = dict(sorted(self.symbols_book[(symbol_name, value_type)]['scenario_data'].items()))

    def _get_metadata(self, raw_metadata):
        short_id = self.short_names
        dc = {}
        for k, v in raw_metadata.items():
            dc[short_id[k]] = {}
            for key, value in v.items():
                dc[short_id[k]][key] = value
        # new
        tdc = {}
        for ids, details in dc.items():
            for key in details:
                if key not in tdc:
                    tdc[key] = {}
                tdc[key][ids] = details[key]
        return tdc
        # return pd.DataFrame(dc).transpose().to_dict()

    def _join_all_symbols(self):
        for symb in self.symbol_valuetype_dict:
            self._join_scenarios_by_symbol(*symb)
        return None

    def __repr__(self):
        return '''DataCollection()'''
