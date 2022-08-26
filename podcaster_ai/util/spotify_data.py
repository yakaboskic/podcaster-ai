import os, json
import tqdm
import pandas as pd

from boxsdk import JWTAuth, Client
from boxsdk.object.file import File as BoxFile
from boxsdk.object.folder import Folder as BoxFolder
from collections import defaultdict


# Box credentials
CREDENTIALS_FILENAME = '915983714_g0pco3fr_config.json'
CREDENTIALS_PATH = os.path.join(
        os.path.dirname(__file__),
        '../..', 
        'data', 
        CREDENTIALS_FILENAME
        )
PRIVATE_KEY_PATH = os.path.join(
        os.path.dirname(CREDENTIALS_PATH),
        'box-private-key',
        )

# Spotify File Constants and IDs
METADATA_ID = '664983823420'
TMP_METADATA_FILEPATH = '/tmp/spotify-metadata.tsv'
SPOTIFY_PODCASTS_ROOTDIR = 'Spotify-Podcasts'

def get_client(
        ):
    """ Returns your Box python SDK client.
    See https://github.com/box/box-python-sdk for details.
    """
    with open(CREDENTIALS_PATH, 'r') as f_:
        creds = json.load(f_)
    # Load Service Account Client
    service_account_auth = JWTAuth(
            client_id=creds["boxAppSettings"]["clientID"],
            client_secret=creds["boxAppSettings"]["clientSecret"],
            enterprise_id=creds["enterpriseID"],
            jwt_key_id=creds["boxAppSettings"]["appAuth"]["publicKeyID"],
            rsa_private_key_file_sys_path=PRIVATE_KEY_PATH,
            rsa_private_key_passphrase=creds["boxAppSettings"]["appAuth"]["passphrase"],
            )
    service_account_client = Client(service_account_auth)
    # Load User Client
    app_user = service_account_client.user(user_id='18540353656')
    app_user_auth = JWTAuth(
            client_id=creds["boxAppSettings"]["clientID"],
            client_secret=creds["boxAppSettings"]["clientSecret"],
            enterprise_id=creds["enterpriseID"],
            user=app_user,
            jwt_key_id=creds["boxAppSettings"]["appAuth"]["publicKeyID"],
            rsa_private_key_file_sys_path=PRIVATE_KEY_PATH,
            rsa_private_key_passphrase=creds["boxAppSettings"]["appAuth"]["passphrase"],
            )
    app_user_auth.authenticate_user()
    return Client(app_user_auth)

def get_metadata(client, file_id=None, download_path=None):
    """ Pulls the Metadata file from the Spotify repo. This file contains every podcast's information
    and identification.
    """
    if os.path.exists(TMP_METADATA_FILEPATH):
        return pd.read_csv(TMP_METADATA_FILEPATH, sep='\t', header=0)
    if download_path is None:
        download_path = TMP_METADATA_FILEPATH
    if file_id is None:
        file_id = METADATA_ID
    # Download metadata file
    with open(TMP_METADATA_FILEPATH, 'wb') as f_:
        client.file(file_id).download_to(f_)
    # Read file back into Dataframe
    return pd.read_csv(TMP_METADATA_FILEPATH, sep='\t', header=0)

def walk_box_dir(client, folder, fileids_map, dir_blacklist):
    for item in client.folder(folder_id=folder.id).get_items():
        if type(item) == BoxFolder:
            if item.name in dir_blacklist:
                continue
            print(f'Going into folder: {item.name}')
            fileids_map = walk_box_dir(client, item, fileids_map, dir_blacklist)
            print(f'Coming out of folder: {item.name}')
        elif type(item) == BoxFile:
            #print(f'Found file: {item.name}')
            fileids_map[item.name] = item.id
        else:
            print(f'Unrecognized Box Type: {type(item)} in folder: {folder_id}. Skipping...')
            continue
    return fileids_map

def get_box_fileids(client, dir_blacklist=None):
    root_folder_id = client.root_folder().get().id
    fileids_map = {}
    if dir_blacklist is None:
        dir_blacklist = []
    for item in client.folder(folder_id=root_folder_id).get_items():
        if item.name == SPOTIFY_PODCASTS_ROOTDIR:
            fileids_map = walk_box_dir(client, item, fileids_map, dir_blacklist)
    return fileids_map

def get_file(client, box_filename, fileids_map=None, fileids_map_path=None, save_dir='/tmp'):
    if fileids_map_path is not None:
        with open(fileids_map_path, 'r') as f_:
            fileids_map = json.load(f_)
    save_path = os.path.join(save_dir, box_filename)
    try:
        with open(save_path, 'wb') as f_:
            client.file(fileids_map[box_filename]).download_to(f_)
    except:
        print(f'Could not download file: {box_filename}.')
        return None
    return save_path
