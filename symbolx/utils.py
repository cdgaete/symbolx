import glob
import json
import os
import shutil
import pathlib
import zipfile
from zipfile import ZIP_DEFLATED, ZIP_LZMA, ZIP_BZIP2

EXT_CONF = {'7z':ZIP_LZMA, 'zip': ZIP_DEFLATED, 'bz2':ZIP_BZIP2}

def _get_files_path_list_from_zip(folder: str, ext: str, zip_ext: str='7z'):
    zip_file_list = glob.glob(os.path.join(folder,f'*.{zip_ext}'))
    files_path = []
    for zip_path in zip_file_list:
        with zipfile.ZipFile(zip_path, mode="r") as zfile:
            for fpath in zfile.namelist():
                complete_path = os.path.join(zip_path,fpath)
                if f".{ext}" in fpath:
                    files_path.append(complete_path)
    return sorted(files_path)

def _get_files_path_list_from_folder(folder: str, ext='*'):
    return glob.glob(os.path.join(folder,f'*/*.{ext}'))

def _get_files_path_list(folder:str, zip_extension:str, file_extension:str):
    if zip_extension is not None:
        file_list = _get_files_path_list_from_zip(folder=folder, ext=file_extension, zip_ext=zip_extension)
        assert len(file_list) > 0, f"No files found in {folder} or in its .{zip_extension} files"
    else:
        file_list = _get_files_path_list_from_folder(folder=folder, ext=file_extension)
        assert len(file_list) > 0, f"No files found in {folder}"
    return sorted(file_list)

def load_scenario_info(path, serializer='json', zip_extension=None):
    '''
    Load scenario info from file.
    '''
    options = ['json','yml']
    assert serializer in options, f"Serializer options are {options}. Provided: '{serializer}'."
    if zip_extension is not None:
        path_parts = path.split(zip_extension+os.sep)
        zip_fpath = path_parts[0] + zip_extension
        target_fpath = path_parts[1]
        with zipfile.ZipFile(zip_fpath, mode="r") as zip_io:
            with zip_io.open(target_fpath) as the_file:
                if serializer == 'json':
                    info = _load_scenario_info_json(the_file)
                elif serializer == 'yml':
                    info = _load_scenario_info_yaml(the_file)
        return info
    else:
        if serializer == 'json':
            return _load_scenario_info_json(path)
        elif serializer == 'yml':
            return _load_scenario_info_yaml(path)

def _load_scenario_info_json(path):
    '''
    Load scenario info from json file.
    '''
    if isinstance(path, str):
        base_name = os.path.basename(path).rstrip('.json')
        with open(path,'r') as stream:
            info = json.load(stream)
            assert all([key in info for key in ['name','metadata']])
            assert info['name'] == base_name, f"Folder name '{base_name}' and 'name':'{info['name']}' do not match."
        return info
    elif isinstance(path, zipfile.ZipExtFile):
        return json.load(path)


def _load_scenario_info_yaml(path):
    '''
    Load scenario info from yaml file compatible with dieterpy <= 1.6.0.
    '''
    import yaml
    if isinstance(path, str):
        base_name = os.path.basename(path).rstrip('_config.yml')
        info = {}
        with open(path,'r') as stream:
            info_ = yaml.load(stream,Loader=yaml.FullLoader)
            info['name'] = info_['id']
            info['metadata'] = info_['config']
            assert info['name'] == base_name, f"Folder name '{base_name}' and 'name':'{info['name']}' do not match."
        return info
    elif isinstance(path, zipfile.ZipExtFile):
        info = {}
        info_ = yaml.load(path,Loader=yaml.FullLoader)
        info['name'] = info_['id']
        info['metadata'] = info_['config']
        # assert info['name'] == base_name, f"Folder name '{base_name}' and 'name':'{info['name']}' do not match."
        return info


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


def compress_subdirs(folder: str, zip_extension: str, compresslevel: int=9, allowZip64: bool=True, delete=True):
    child_folders = [path for path in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder, path))]
    for childir in child_folders:
        zip_folder = os.path.join(folder, childir)
        compress_dir(zip_folder, zip_extension, compresslevel, allowZip64, delete)
    return None


def compress_dir(folder_path: str, zip_extension: str, compresslevel: int=9, allowZip64: bool=True, delete=True):
    assert zip_extension in EXT_CONF, f"{zip_extension=}. Valid options are: {list(EXT_CONF)}"
    directory = pathlib.Path(folder_path)
    with zipfile.ZipFile(f"{folder_path}.{zip_extension}", "w", EXT_CONF[zip_extension], compresslevel=compresslevel, allowZip64=allowZip64) as archive:
        for file_path in directory.rglob("*"):
            archive.write(file_path, arcname=file_path.relative_to(directory))
    if delete:
        try:
            shutil.rmtree(folder_path)
        except OSError:
            print(f"Error folder not deleted {folder_path}")
            # TODO: logger with details about the error (permission denied?)
    return None

def unzip(source_filename, dest_dir, delete=False):
    with zipfile.ZipFile(source_filename) as zf:
        zf.extractall(dest_dir)
    if delete:
        try:
            os.remove(source_filename)
        except OSError as e:
            # If it fails, inform the user.
            print("Error: %s - %s." % (e.filename, e.strerror))

def unzip_all(folder, zip_extension='7z', delete=False):
    options = list(EXT_CONF)
    assert zip_extension in options
    zip_files_found = glob.glob(os.path.join(folder, f'*.{zip_extension}'))
    for file_path in zip_files_found:
        dest_dir = os.path.splitext(file_path)[0]
        unzip(file_path, dest_dir, delete)

def select_scenario_files(scenarios_path, zip_extension=None):
    if zip_extension is None:
        file_list = []
        for file in scenarios_path:
            A = os.path.basename(os.path.dirname(file))
            B = os.path.basename(file).split('.')[0]
            if A == B:
                file_list.append(file)
        return file_list
    else:
        file_list = []
        for file in scenarios_path:
            path_parts = file.split(zip_extension+os.sep)
            zip_fpath = path_parts[0] + zip_extension
            target_fpath = path_parts[1]
            A = os.path.basename(zip_fpath).split('.')[0]
            B = os.path.basename(target_fpath).split('.')[0]
            if A == B:
                file_list.append(file)
        return file_list