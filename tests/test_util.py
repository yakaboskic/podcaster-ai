import unittest

from podcaster_ai.util.spotify_data import *

class TestSpotifyData(unittest.TestCase):

    def test_get_client(self):
        client = get_client()

    def test_get_metadata(self):
        client = get_client()
        df = get_metadata(client)
        print(df)
