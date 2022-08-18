import unittest

from podcaster_ai.util.spotify_data import *

class TestSpotifyData(unittest.TestCase):

    def test_get_client(self):
        client = get_client()

    def test_get_metadata(self):
        client = get_client()
        df = get_metadata(client)
        print(df)

    def test_get_audiofile(self):
        client = get_client()
        fileids_map_path = 'data/spotify-fileids.json'
        audio_filename = '009TPwk7i2wA7UA8QEjAo4.ogg'
        print(get_file(client, audio_filename, fileids_map_path=fileids_map_path))
