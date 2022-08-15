import glob
import os
import re
import copy
from typing import Callable

from .parser import load_scenario_info


class SymbolHandler:
    def __init__(self):
        '''
        Collect scenarios per symbol
        '''
        self.collector = {}
        self.config_files = None
        self.config = None
        self.data = None

    def add_collector(self, colector_name:str, parser:Callable, loader:Callable):
        self.collector[colector_name] = {}
        self.collector[colector_name]['parser'] = parser
        self.collector[colector_name]['loader'] = loader
        self.add_symbol_list(colector_name, symbol_list=[])

    def add_folder(self, collector_name:str, folder:str):
        self.collector[collector_name]['folder'] = folder

    def add_symbol_list(self, collector_name:str, symbol_list:list=[]): # optional
        self.collector[collector_name]['symbol_list'] = symbol_list

    def adquire(self):
        self.config_files = []
        self.config = {}
        self.data = []

        for collector in self.collector:
            folder = self.collector[collector]['folder']
            symbol_list = self.collector[collector]['symbol_list']
            parser = self.collector[collector]['parser']
            dyn_config_path = os.path.join(folder,"*/*_config.yml")
            config_files = glob.glob(dyn_config_path)
            self.config_files += config_files
            for symbol_info_dict in parser(folder, symbol_list):
                symbol_info_dict['collector'] = collector
                self.data.append(symbol_info_dict)

        assert len(self.config_files) > 0, "No config files found"
        for config_file in self.config_files:
            config = load_scenario_info(config_file)
            self.config[config['id']] = config
        assert len(self.config) == len(self.config_files), "Config files with same id found"


    def collectinfo(self):

        self.symbols = self._list_all_symbols()
        self.shortscennames = self._scenario_name_shortener(self.data)
        self.loopitems = self.get_loopitems(self.data)
        self.pathsbook = dict()


    def _list_all_symbols(self):
        list_with_symbols_and_value_type = []
        for symb_info in self.data:
            list_with_symbols_and_value_type.append('.'.join([symb_info['symbol_name'], symb_info['value_type']]))
        return sorted(list(set(list_with_symbols_and_value_type)))
    
    def _scenario_name_shortener(self):
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
                        shortname = "S" + pattern.search(name)[0].zfill(digit)
                        shortnames[name] = shortname
                else:
                    flag = True
            else:
                flag = True
        if flag:
            for n, name in enumerate(names):
                shortname = "S" + str(n).zfill(digit)
                shortnames[name] = shortname
        return shortnames

    def get_loopitems(self):
        loopkeys = []
        for scen in self.config:
            loopkeys += list(self.config[scen]["config"].keys())
        loopset = list(set(loopkeys))
        loopitems = {}
        for loop in loopset:
            loopitems[loop] = None
        return loopitems

    def get_modifiers(self, scen, loopitems):
        loops = {}
        for key in loopitems:
            if key in scen["loop"].keys():
                loops[key] = scen["loop"][key]
        return loops

    def join_scens_by_symbol(self, symbol):
        """
        symbol
        """
        sceninfo_dict = dict()
        for indx, scen in enumerate(self.data):
            if symbol in scen.keys():
                sceninfo_dict[indx] = scen
        outputdict = {}
        for idx, scenario in sceninfo_dict.items():
            symb_seudo_df = {}
            symb_seudo_df['symbol'] = symbol
            symb_seudo_df['scenario_id'] = scenario['scenario']
            symb_seudo_df['short_id'] = self.shortscennames[scenario['scenario']]
            symb_seudo_df['dims'] = scenario[symbol]['dims']
            symb_seudo_df['nrdims'] = scenario[symbol]['nrdims']
            symb_seudo_df['nrrecs'] = scenario[symbol]['nrrecs']
            symb_seudo_df['uelmap'] = scenario[symbol]['uelmap']
            symb_seudo_df['gdx_path'] = scenario[symbol]['gdx_path']
            symb_seudo_df['dim_coords'] = scenario[symbol]['dim_coords']
            outputdict[idx] = symb_seudo_df

        symblist = [v for v in outputdict.values()]

        flag = -1
        modifiers = dict()
        for ix, scen in enumerate(self.data):
            if symbol in scen.keys():
                modifiers[self.shortscennames[scen["scenario"]]] = self.get_modifiers(
                    scen, self.loopitems
                )
                flag = ix
            else:
                logger.info(f'   Symbol "{symbol}" is not in {scen["scenario"]}')

        if flag > -1:

            for i, scen in enumerate(self.data):
                if symbol in self.data[i]:
                    idx = i
                    break

            symbol_data_example = self.data[idx][symbol]
            symbdict = {}
            symbdict['name'] = symbol
            symbdict['dims'] = symbol_data_example['dims']
            symbdict['type'] = symbol_data_example['type']
            symbdict['symb_desc'] = symbol_data_example['symb_desc']
            symbdict['data'] = symblist
            symbdict["scen"] = self.shortscennames
            symbdict["loop"] = list(self.loopitems.keys())
            symbdict["modifiers"] = modifiers
            symbdict["reporting"] = []

            if symbol not in self.pathsbook.keys():
                self.pathsbook[symbol] = {}
            self.pathsbook[symbol]['v'] = symbdict
            self.pathsbook[symbol]['m'] = symbdict
            self.pathsbook[symbol]['lo'] = symbdict
            self.pathsbook[symbol]['up'] = symbdict

            return symbdict
        else:
            logger.info(f'Symbol "{symbol}" does not exist in any scenario')
            return None

    def join_all_symbols(self, warningshow=True):
        for symb in self.symbols:
            self.join_scens_by_symbol(symb)
        if warningshow:
            logger.info('In a new python script or notebook you can access the data with this snippet: \n   from dieterpy import SymbolsHandler, Symbol \n   SH = SymbolsHandler("folder") \n   logger.info(SH.reporting) \n   Z = Symbol(name="Z", value_type="v", symbol_handler=SH) \n   Z.df  # <- Pandas DataFrame')
