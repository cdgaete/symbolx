import karray as ka
import pandas as pd
import numpy as np
import os
import pyarrow.feather as ft
import json
from typing import Union
from .handler import SymbolsHandler, from_feather_dict
from .settings import settings, allowed_string, value_type_name_map


def build_array(symbol_name:str, value_type:str, symbol_handler:SymbolsHandler):
    """
    Create an array of all existing scenarios
    """
    list_of_arrays = []
    for scenario_id in symbol_handler.get_info(symbol_name, value_type)['scenario_data']:
        array_with_id = insert_id_dim(symbol_name, value_type, scenario_id, symbol_handler)
        list_of_arrays.append(array_with_id)
    return ka.concat(list_of_arrays)

def insert_id_dim(symbol_name:str, value_type:str, scenario_id:str, symbol_handler:SymbolsHandler):
    """
    Insert dimension id with the corresponding scenario_id.
    """
    single_symbol = symbol_handler.get_info(symbol_name, value_type)['scenario_data'][scenario_id]
    single_array_dict = symbol_handler.collector[single_symbol['collector']]['loader'](**single_symbol)
    oarray = ka.array(order=symbol_handler.order, **single_array_dict)
    narray = oarray.add_dim('id',[symbol_handler.short_names[scenario_id]])
    return narray

def from_feather(path):
    return Symbol(**from_feather_dict(path))


class Symbol:
    def __init__(
        self,
        name: str=                      None,
        value_type: str=                None,
        metadata: dict=                 None,
        array: ka.array=                None,
        symbol_handler_token: str=      None,
        symbol_handler: SymbolsHandler= None,
        ):
        '''
        A class for creating symbols.
        '''
        self.__dict__["_repo"] = {}
        if symbol_handler is not None:
            if 'object' == symbol_handler.method:
                self.build(symbol_handler, name, value_type)
            elif 'folder' == symbol_handler.method:
                self.load(symbol_handler, name, value_type)
        else:
            if settings.exists((name,value_type,symbol_handler_token)):
                print("The symbol name has already been taken by another Symbol stored in SymbolsHandler.")
                print("Please choose another name to avoid overwriting the existing symbol when saving the current symbol.")
            self.name=                 name
            self.value_type=           value_type
            self.metadata=             metadata
            self.array=                array
            self.symbol_handler_token= symbol_handler_token
        self.check_input()


    def __setattr__(self, name, value):
        self._repo[name] = value

    def __getattr__(self, name):
        if name == "xxxxx":
            # Here if does not exist or is None: if statement
            # generate and save: self._repo[name] = function or value
            # finally handover: return self._repo[name]
            return None
        else:
            return self._repo[name]

    def build(self, symbol_handler:SymbolsHandler, name: str=None, value_type: str=None):
        assert isinstance(symbol_handler, SymbolsHandler), "'symbol_handler' must be a SymbolsHandler object"
        assert isinstance(name, str), "'name' must be a string"
        assert isinstance(value_type, str), "'value_type' must be a string"
        print(f"{name:<25} value_type: {value_type_name_map[value_type]} ({value_type})")

        key = (name,value_type)
        assert key in symbol_handler.symbols_book, f"'{key}' is not present in 'symbol_handler.symbols_book' dictionary"

        self.name=                 name
        self.value_type=           value_type
        self.metadata=             symbol_handler.get_info(*key)['metadata']
        self.array=                build_array(name, value_type, symbol_handler)
        self.symbol_handler_token= symbol_handler.symbol_handler_token
        
    def load(self, symbol_handler:SymbolsHandler, name: str=None, value_type: str=None):
        assert isinstance(symbol_handler, SymbolsHandler), "'symbol_handler' must be a SymbolsHandler object"
        assert isinstance(name, str), "'name' must be a string"
        assert isinstance(value_type, str), "'value_type' must be a string"
        print(f"{name:<25} value_type: {value_type_name_map[value_type]} ({value_type})")

        key = (name,value_type)
        assert key in symbol_handler.symbols_book, f"'{key}' is not present in 'symbol_handler.symbols_book' dictionary"

        file_path = symbol_handler.get_info(*key)
        symbol_dict = from_feather_dict(file_path)
        self.name=                 name
        self.value_type=           value_type
        self.metadata=             symbol_dict['metadata']
        self.array=                symbol_dict['array']
        self.symbol_handler_token= symbol_dict['symbol_handler_token']

    def to_arrow(self):
        table = self.array.to_arrow()
        existing_meta = table.schema.metadata
        custom_meta_key = 'symbolx'
        custom_metadata = {}
        attr = ['name', 'value_type', 'metadata','symbol_handler_token']
        for k,v in self._repo.items():
            if k in attr:
                custom_metadata[k] = v

        custom_meta_json = json.dumps(custom_metadata)
        existing_meta = table.schema.metadata
        combined_meta = {custom_meta_key.encode() : custom_meta_json.encode(),**existing_meta}
        table = table.replace_schema_metadata(combined_meta)
        return table

    def to_feather(self, path:str):
        sets = set(allowed_string)
        sets.add(os.path.sep)
        joint = ''.join(sorted(sets))
        assert len(set(path).difference(sets)) == 0, f"There are/is special characters in path '{path}'. Allowed chars are: {joint}"

        table = self.to_arrow()
        ft.write_feather(table, path)
        print(f"{path}")
        return None

    def check_input(self):
        assert self.name is not None and isinstance(self.name, str), "Name of symbol must be provided."
        assert self.value_type is not None and isinstance(self.value_type, str), "Value type of symbol must be provided."
        assert self.metadata is not None and isinstance(self.metadata, dict), "metadata of symbol must be provided."
        assert self.array is not None and isinstance(self.array, ka.array), "Array must be provided."
        assert self.symbol_handler_token is not None and isinstance(self.symbol_handler_token, str), "Symbol handler token must be provided."

    @property
    def dims(self):
        return self.array.dims

    def set_array_from_pandas(self, df, order=['id']):
        array = ka.from_pandas(df, order=order)
        # TODO: Include sanity checks
        self.array = array
        return None

    def get_df(self):
        return self.array.to_dataframe()

    @property
    def dfm(self):
        dfm = self.get_df().reset_index()
        for k, v in self.metadata.items():
            dfm[k] = dfm["id"].map(v)
        return dfm

    @property
    def dfc(self):
        dfc = self.get_df().reset_index()
        for k, v in self.metadata.items():
            if 'custom_' in k:
                dfc[k] = dfc["id"].map(v)
        return dfc

    def metadata_union(self, other=None):
        self_metadata = self.metadata
        other_metadata = other.metadata
        new_metadata = {}
        for elem in self_metadata.keys():
            new_metadata[elem] = {**self_metadata[elem],**other_metadata[elem]}
        return new_metadata

    def new_symbol(self, array, new_name, other=None):
        if isinstance(other, Symbol):
            assert self.symbol_handler_token == other.symbol_handler_token, "Symbol handler tokens must be the same"
            new_metadata = self.metadata_union(other)
        elif other is None or isinstance(other,(int,float)):
            new_metadata = self.metadata
        else:
            raise Exception(f'other must be either a Symbol object, int, float or None, but it is: {str(type(other))}')
        new_object = Symbol(name=new_name, value_type='v',
                            metadata=new_metadata, array=array, 
                            symbol_handler_token=self.symbol_handler_token)
        return new_object

    def __add__(self, other):
        flag = False
        if isinstance(other, (int, float)):
            new_array = self.array + other
            new_name =  f"({self.name})+{str(other)}"
            return self.new_symbol(new_array, new_name, other)
        elif isinstance(other, Symbol):
            if set(self.dims) == set(other.dims):
                new_array = self.array + other.array
                new_name = f"({self.name})+({other.name})"
                return self.new_symbol(new_array, new_name, other)
            else:
                flag = True
            if flag:
                raise Exception(f'dims are not equal. {self.name} Dims: {self.dims}, {other.name} Dims: {other.dims}')
        else:
            raise Exception(f'{type(other)} is not supported')

    def __sub__(self, other):
        flag = False
        if isinstance(other, (int, float)):
            new_array = self.array - other
            new_name =  f"({self.name})-{str(other)}"
            return self.new_symbol(new_array, new_name, other)
        elif isinstance(other, Symbol):
            if set(self.dims) == set(other.dims):
                new_array = self.array - other.array
                new_name = f"({self.name})-({other.name})"
                return self.new_symbol(new_array, new_name, other)
            else:
                flag = True
            if flag:
                raise Exception(f'dims are not equal. {self.name} Dims: {self.dims}, {other.name} Dims: {other.dims}')
        else:
            raise Exception(f'{type(other)} is not supported')

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_array = self.array * other
            new_name = f"({self.name})*{str(other)}"
            return self.new_symbol(new_array, new_name, other)
        elif isinstance(other, Symbol):
            diffdims = list(set(self.dims).symmetric_difference(other.dims))
            lendiff = len(diffdims)
            if set(self.dims) == set(other.dims):
                new_array = self.array * other.array
                new_name = f"({self.name})*({other.name})"
                return self.new_symbol(new_array, new_name, other)
            elif lendiff == 1:
                new_array = self.array * other.array
                new_name = f"({self.name})*({other.name})"
                return self.new_symbol(new_array, new_name, other)
            elif lendiff > 1:
                common_dims = list(set(self.dims).intersection(other.dims))
                if len(common_dims) > 0:
                    new_array = self.array * other.array
                    new_name = f"({self.name})*({other.name})"
                    return self.new_symbol(new_array, new_name, other)
                else:
                    raise Exception(f"The difference in dimensions is greater than one: '{diffdims}' and has no common dimensions")
        else:
            raise Exception(f'{type(other)} is not supported')

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            new_array = self.array / other
            new_name = f"({self.name})/{str(other)}"
            return self.new_symbol(new_array, new_name, other)
        elif isinstance(other, object):
            diffdims = set(self.dims).symmetric_difference(other.dims)
            lendiff = len(diffdims)
            if set(self.dims) == set(other.dims):
                new_array = self.array / other.array
                new_name = f"({self.name})/({other.name})"
                return self.new_symbol(new_array, new_name, other)
            elif lendiff == 1:
                new_array = self.array / other.array
                new_name = f"({self.name})/({other.name})"
                return self.new_symbol(new_array, new_name, other)

            elif lendiff > 1:
                raise Exception(f"The difference in dimensions is greater than one: '{diffdims}'")
        else:
            raise Exception("The second term is not known, must be a int, float or a Symbol object")

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            new_array =  other + self.array
            new_name = f"{str(other)}+({self.name})"
            return self.new_symbol(new_array, new_name, other)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            new_array = other - self.array
            new_name = f"{str(other)}-({self.name})"
            return self.new_symbol(new_array, new_name, other)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            new_array = other*self.array
            new_name = f"{str(other)}*({self.name})"
            return self.new_symbol(new_array, new_name, other)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            new_array = other/self.array
            new_name = f"{str(other)}/({self.name})"
            return self.new_symbol(new_array, new_name, other)




    # @staticmethod
    # def reduce(array: Union[xr.DataArray,sc.DataArray], dim: str, func: str = 'sum'):
    #     if func == 'sum':
    #         new_array = array.sum(dim=dim)
    #     elif func == 'mean':
    #         new_array = array.mean(dim=dim)
    #     return new_array

    # def dimreduce(self, dim: str='h', aggfunc='sum'):
    #     new_dims = list(set(self.dims).symmetric_difference([dim]))
    #     new_name = f"({self.name}).dimreduce({dim})"
    #     new_array = self.reduce(self.array, dim, aggfunc)
    #     return self.new_symbol(new_array, new_name, new_dims)

    # def rename(self, new_name: str):
    #     new_array = self.array
    #     new_dims = self.dims
    #     return self.new_symbol(new_array, new_name)

    def expand_dim(self, new_dim, unique_coord):
        if settings.ARRAY_MODULE == 'scipp':
            nparr = self.array.data.values
            dims = self.array.dims
            coords = {dim: self.array.coords[dim] for dim in self.array.coords}
            new_data = {}
            new_data['data'] = sc.array(dims=dims + (new_dim,), values=nparr[..., np.newaxis].tolist())
            new_data['coords'] = coords
            new_data['coords']['id'] = sc.array(dims=[new_dim], values=[unique_coord], unit=None)
            arr_plus_new_coords = sc.DataArray(**new_data)
            arr_plus_new_coords.name = 'value'
            return arr_plus_new_coords

        elif settings.ARRAY_MODULE == 'xarray':
            arr_plus_new_coords = self.array.assign_coords(**{new_dim:unique_coord})
            arr_plus_new_coords = arr_plus_new_coords.expand_dims(dim=new_dim, axis=-1)
            arr_plus_new_coords.name = 'value'
            return arr_plus_new_coords


# # Until here
#     def refdiff(self, reference_id=0):
#         if not self.get('dims'):
#             data = self.df.drop("symbol", axis=1)
#             data['value'] = data['value'] - data[data["id"] == reference_id]['value'].values
#         else:
#             dataframes = []
#             for ix, df in self.df.drop("symbol", axis=1).groupby(self.get('dims')):
#                 df['value'] = df['value'] - df[df["id"] == reference_id]['value'].values
#                 dataframes.append(df)
#             data = pd.concat(dataframes)
#         name = self.get('name')+'_diff_on_'+ reference_id
#         dt = data[['id'] + self.get("dims") + ['value']].reset_index(drop=True)
#         return self.new_symbol(dt, name, self.get('dims'))

#     def create_mix(self, criteria):
#         ''' '''
#         combination = self.create_combination(criteria)
#         order = criteria.keys()
#         return self._find_ids_by_tuple(order,combination)

#     def create_combination(self, criteria: dict):
#         return list(itertools.product(*criteria.values()))


#     def _find_ids_by_tuple(self,key_order,combination):
#         groups = {}
#         for i, pair in enumerate(combination):
#             config = {}
#             for k, v in zip(key_order, pair):
#                 config[k] = ('==',v)
#             groups[i] = list(self.find_ids(**config))
#         return groups

#     def _ref_diff_group(self,refs,groups, verbose=False):
#         symbols = []
#         for key in groups:
#             if len(refs[key]) == 0:
#                 if verbose:
#                     logger.info(f"{refs} for key = {key} no reference id found")
#                     logger.info(groups)
#                 continue
#             else:
#                 refdiff_symbol = self.shrink_by_id(groups[key]).refdiff(refs[key][0])
#                 symbols.append(refdiff_symbol)
#         return sum(symbols)

#     def refdiff_by_sections(self, criteria_dict, criteria_ref_dict, verbose=False):
#         ''' '''
#         groups = self.create_mix(criteria_dict)
#         refs = self.create_mix({**criteria_dict,**criteria_ref_dict})
#         return self._ref_diff_group(refs,groups,verbose)

#     def refdiff_by_sections_tuple(self, key_ref: str, key_order: list, combination: list, verbose: bool=False):
#         ''' '''
#         combination_no_ref = []
#         index = key_order.index(key_ref)
#         for cluster in combination:
#             cluster_list = list(cluster)
#             cluster_list.pop(index)
#             no_ref_cluster = tuple(cluster_list)
#             combination_no_ref.append(no_ref_cluster)
#         order_no_ref = [key for key in key_order if key != key_ref]
#         groups = self._find_ids_by_tuple(order_no_ref,combination_no_ref)
#         refs = self._find_ids_by_tuple(key_order,combination)
#         return self._ref_diff_group(refs,groups,verbose)

    @property
    def items(self):
        if len(self.dims) > 0:
            elements = {}
            for dim in self.dims:
                elements[dim] = list(self.array.coords[dim])
            return elements
        else:
            raise Exception("Symbol has no dimension") #TODO: Check the existence of one dimension 'id' at least

    def rename_dim(self, old_dim: str, new_dim: str):
        """ This function renames a dimension in a symbol.

        Args:
            old_dim (str): dimension to be renamed
            new_dim (str): new dimension name

        Returns:
            symbol: Returns a new symbol with the renamed dimension.
        """
        new_array = self.array.rename({old_dim: new_dim})
        new_name = f"({self.name}).rename_dim({old_dim},{new_dim})"
        new_dims = self.dims + [new_dim]
        new_dims.remove(old_dim)
        return self.new_symbol(new_array, new_name, new_dims)
        
    def add_dim(self, dim_name: str, value: Union[str,dict]):
        '''
        dim_name: new dimension name
        value:  if value is a string, the dimension column will contain this value only.
                if value is a dict, the dict must look like {column_header:{column_element: new_element_name}}
                where column_header must currently exists and all column_elements must have a new_element_name.
        '''
        if isinstance(value, str):
            new_array = self.expand_dim(dim_name, value)
            new_name = f"({self.name}).add_dim({dim_name},{value})"
            new_dims = self.dims + [dim_name]
            return self.new_symbol(new_array, new_name, new_dims)

        elif isinstance(value, dict):
            df = self.df.copy()
            df.insert(-3, dim_name, None)
            key = list(value.keys())[0]
            val = value[key]
            df[dim_name] = df[key].map(val)
            self.set_array_from_df(df)
            new_array = self.array
            new_name = f"({self.name}).rename_dim({dim_name},{key}:map_dict)"
            new_dims = self.dims + [dim_name]
            return self.new_symbol(new_array, new_name, new_dims)
        else:
            raise Exception('value is neither str nor dict')

#     def round(self, decimals:int):
#         df = self.df.set_index(['id'] + self.get("dims")).drop("symbol", axis=1)
#         df['value'] = df['value'].round(decimals)
#         return self.new_symbol(self, df, f"{self.name}.round({str(decimals)})", self.get("dims"))

#     def elems2str(self, by='h', string='t', digits=4):
#         df = self.df.copy()
#         df[by] = df[by].apply(lambda x: string+str(x).zfill(digits))
#         new_object = self*1
#         new_object.dims = self.get('dims')
#         new_object.df = df
#         new_object.info = self.info
#         new_object.name = f"({self.get('name')}).elems2str({by},{string},{str(digits)})"
#         return new_object

#     def elems2int(self, by='h'):
#         df = self.df.copy()
#         df.loc[:, by] = df[by].str.extract(r"(\d+)", expand=False).astype("int16")
#         new_object = self*1
#         new_object.dims = self.get('dims')
#         new_object.df = df
#         new_object.info = self.info
#         new_object.name = f"({self.get('name')}).elems2int({by})"
#         return new_object

#     def replacezero(self, by=1):
#         df = self.df.copy()
#         df['value'] = df['value'].replace(0.0,by)
#         new_object = self*1
#         new_object.dims = self.get('dims')
#         new_object.df = df
#         new_object.info = self.info
#         new_object.name = f"({self.get('name')}).replacezero(by={str(by)})"
#         return new_object

#     def replaceall(self, by=1):
#         df = self.df.copy()
#         df['value'] = by
#         new_object = self*1
#         new_object.dims = self.get('dims')
#         new_object.df = df
#         new_object.info = self.info
#         new_object.name = f"({self.get('name')}).replaceall(by={str(by)})"
#         return new_object

#     def replacenan(self, by=0):
#         df = self.df.copy()
#         df['value'] = df['value'].fillna(by)
#         new_object = self*1
#         new_object.dims = self.get('dims')
#         new_object.df = df
#         new_object.info = self.info
#         new_object.name = f"({self.get('name')}).replacenan(by={str(by)})"

#         return new_object

#     def fillelems(self):
#         dims = self.dims
#         if len(dims) == 0:
#             raise Exception("Symbol without dimensions")
#         elif len(dims) == 1:
#             df = pd.concat({'value':
#                                 self.df.drop("symbol", axis=1)
#                                 .pivot_table(
#                                             index=['id'],
#                                             columns=dims,
#                                             values='value',
#                                             aggfunc=sum,
#                                             )
#                                 .fillna(0)
#                                 .sort_index()
#                             },
#                             names=[''], axis=1,
#                             ).stack(dims)
#         elif len(dims) > 1:
#             if 'h' in dims:
#                 dim = 'h'
#             else:
#                 dim = dims[0]
#             common_dims = [elem for elem in dims if elem != dim]
#             df = self.reorganize(dim, common_dims)
#             df = df.stack(common_dims)
#         new_object = self*1
#         new_object.dims = self.get('dims')
#         new_object.df = df.reset_index()
#         new_object.info = self.info
#         new_object.name = f"({self.get('name')}).fillelems()"
#         return new_object
    
#     def find_ids(self, **karg):
#         '''
#         find ids whose headings comply the criteria to the value indicated
#         dictionary heading as key and value as tuple of (operator, value)
#         Example: Z.find_ids(**{'time_series_scen':('==','NaN'),'co2price(n,tech)':('<',80)})
#         '''
#         dc = self.get('modifiers')
#         collector = []
#         for k,v in dc.items():
#             if k in karg.keys():
#                 flag = False
#                 nan_str = False
#                 id_list = []
#                 for k2, v2 in v.items():
#                     if isinstance(v2,str):
#                         if isinstance(karg[k][1],str):
#                             if eval(f"'{v2}' {karg[k][0]} '{karg[k][1]}'"):
#                                 id_list.append(k2)
#                                 if not flag:
#                                     flag = True
#                             if karg[k][1] != 'NaN' and v2 == 'NaN':
#                                 nan_str = True
#                         else:
#                             continue
#                     elif np.isnan(v2):
#                         if np.isnan(karg[k][1]):
#                             if karg[k][0] == '==':
#                                 id_list.append(k2)
#                                 if not flag:
#                                     flag = True
#                         else:
#                             if karg[k][0] == '!=':
#                                 id_list.append(k2)
#                                 if not flag:
#                                     flag = True
#                             elif karg[k][0] != '==':
#                                 logger.info(f"{k} in {k2} has NaN value for condition '{karg[k][0]} {str(karg[k][1])}'. Not included")
                            
#                     elif np.isnan(karg[k][1]):
#                         if karg[k][0] == '!=':
#                             id_list.append(k2)
#                             if not flag:
#                                 flag = True
#                         else:
#                             continue
#                     elif eval(f"{v2} {karg[k][0]} {karg[k][1]}"):
#                         id_list.append(k2)
#                         if not flag:
#                             flag = True
#                 collector.append(set(id_list))
#                 if not flag:
#                     logger.info(f"Column '{k}' does not contain '{karg[k][1]}'")
#                 if nan_str:
#                     logger.info(f"Column '{k}' has 'NaN' as string. You can filter such string too.")
#         not_present = []
#         for cond in karg.keys():
#             if cond in dc.keys():
#                 pass
#             else:
#                 not_present.append(cond)
#         if not_present:
#             str_cond = ";".join(not_present)
#             logger.info(f"{str_cond} not in symbol's data")
#         return set.intersection(*collector)

#     def id_info(self,ID):
#         '''Gives informaion about the ID

#         Args:
#             ID (str): ID of the scenario
#         Returns:
#               A dictionary with the following Modifiers as keys and the corresponding value.
#         Example:
#            >>> Z.id_info('S0001')
#         '''
#         dc = dict()
#         for k, v in self.get('modifiers').items():
#             if ID in v.keys():
#                 dc[k] = v[ID]
#         return dc
    
#     def shrink(self, **karg):
#         ''' 
#         Shrinks the symbol to keep only those rows that comply the given criteria.
#         karg is a dictionary of symbol sets as key and elements of the set as value.
#         sets and elements must be present in the symbol.
        
#         eg:
#         Z.shrink(**{'tech':['pv','bio'],'h':[1,2,3,4]})
        
#         returns a new symbol
#         '''
#         for key, value in karg.items():
#             if key in self.dims:
#                 if set(value).issubset(self.items[key]):
#                     pass
#                 else:
#                     not_present = set(value) - (set(value) & set(self.items[key]))
#                     raise Exception(f"{not_present} is/are not in {self.items[key]}")
#             else:
#                 raise Exception(f"'{key}' is not in {self.dims} for symbol {self.name}")
#         query_code = " & ".join([f"{k} in {v}" for k,v in karg.items()])
#         df = self.df.copy().query(query_code)

#         new_object = self*1
#         new_object.dims = self.get('dims')
#         new_object.df = df
#         new_object.info = self.info
#         new_object.name = f"({self.get('name')}).shrink({','.join(['='.join([k,str(v)]) for k,v in karg.items()])})"
#         return new_object

#     def shrink_by_id(self, id_list):
#         '''
#         Shrinks the symbol to keep only those rows that comply the given criteria.
#         id_list is a list of ids to keep.
#         '''
#         ids = sorted(list(self.get('modifiers')['run'].keys()))

#         if set(id_list).issubset(ids):
#             pass
#         else:
#             not_present = sorted(list(set(id_list) - (set(id_list) & set(ids))))
#             # logger.info(f"WARNING: {not_present} is/are not in {ids}")
#         query_code = f"id in {id_list}"
#         df = self.df.copy().query(query_code)
#         new_object = self*1
#         new_object.dims = self.get('dims')
#         new_object.df = df
#         new_object.info = self.info
#         new_object.name = f"({self.get('name')}).shrink_by_id({id_list})"
#         return new_object

#     def shrink_by_attr(self, **kargs):
#         ''' 
#         shrink_with_attributes generates new symbol based on other attributes of the dataframe. Attributes can be seen with Symbol.get('modifiers').
#         Shrink the symbol to keep only the row that comply the criteria in kargs.
#         kargs is a dictionary of symbol attributes as key and elements of the attribute columns as value.
#         attributes and attribute's elements must be present in the symbol.

#         eg:
#         Z.shrink(**{'run':[0,1],'country_set':['NA']})

#         returns a new symbol
#         '''
#         for key, value in kargs.items():
#             dc = self.get('modifiers')
#             if key in dc.keys():
#                 if set(value).issubset(set(dc[key].values())):
#                     pass
#                 else:
#                     not_present = set(value) - (set(value) & set(dc[key].values()))
#                     # raise Exception(f"{not_present} is/are not in {list(set(dc[key].values()))}")
#             else:
#                 raise Exception(f"'{key}' is not in {list(dc.keys())} for symbol {self.name}")
#         query_code = " & ".join([f"{k} in {v}" for k,v in kargs.items()])
#         df = self.dfm.copy().query(query_code)
#         new_object = self*1
#         new_object.dims = self.get('dims')
#         new_object.df = df.loc[:,['id','symbol'] + self.get('dims') + ['value']]
#         new_object.info = self.info
#         new_object.name = f"({self.get('name')}).shrink_by_attr({','.join(['='.join([k,str(v)]) for k,v in kargs.items()])})"
#         return new_object
    
#     def transform(self, subset_of_sets=['n'], func='sum', condition='!=', value=0):
#         '''
#         transform consists of providing a list of sets to group the dataframe and apply the function. Then the result is compared with a condition and a value. If the condition is true, the resulting rows are kept, otherwise it is dropped. 
#         subset_of_sets is a list of sets that are present in the symbol.
#         At least one set must be left out of subset_of_sets to apply the function.
        
#         eg: N_TECH.transform(subset_of_sets=['n'], func='sum', condition='!=', value=0)
#             As N_TECH has 'n' and 'tech' as sets, the function is applied to 'n' and agregating all 'tech' and the result is compared with '!= 0'.
#             The final result is a dataframe without elements of 'n' that has a sum of element of 'tech' equal to zero.
#             It is a way to clean up the dataframe by removing elements of a set that are not needed.
#         '''
        
#         ops = {'>': operator.gt,
#                 '<': operator.lt,
#                 '>=': operator.ge,
#                 '<=': operator.le,
#                 '==': operator.eq,
#                 '!=': operator.ne}
        
#         keep = ops[condition](self.df.groupby(["id"]+subset_of_sets)["value"].transform(func), value)
#         df = self.df.loc[keep]
#         new_object = self*1
#         new_object.dims = self.get('dims')
#         new_object.df = df
#         new_object.info = self.info
#         new_object.name = f"({self.get('name')}).transform({subset_of_sets},{func},{condition},{value})"
#         return new_object

    # def __repr__(self):
    #     return f'''Symbol(name='{self.name}', \n       value_type='{self.value_type}', \n       dims={self.dims}'''










# def symbols_operation_on_gpu(symbol_operation):     
#     @wraps(symbol_operation)
#     def wrapper(cls, other):
#         if settings.GPU:
#             if isinstance(cls, Symbol) and isinstance(other, Symbol):
#                 with cupy.cuda.Device(0):
#                     logger.info('Using GPU')
#                     cls.array = cls.array.astype(float).cupy.as_cupy()
#                     other.array = other.array.astype(float).cupy.as_cupy()
#                     new_cls = symbol_operation(cls, other)
#                     new_cls.array = new_cls.array.astype(float).cupy.as_numpy()
#                     cls.array = cls.array.astype(float).cupy.as_numpy()
#                     other.array = other.array.astype(float).cupy.as_numpy()
#                 logger.info('Finished GPU')
#                 return new_cls
#             else:
#                 return symbol_operation(cls, other)
#         else:
#             return symbol_operation(cls, other)
#     return wrapper