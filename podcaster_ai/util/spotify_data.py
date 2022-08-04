import os, json
import pandas as pd
from boxsdk import JWTAuth, Client

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
