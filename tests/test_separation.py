import unittest
import os

from podcaster_ai.separation import separate

class TestDetection(unittest.TestCase):

    def test_009TPwk7i2wA7UA8QEjAo4_full(self):
        path_to_audio = '/tmp/segments/009TPwk7i2wA7UA8QEjAo4-copy_0-22.ogg'
        separate(path_to_audio)

    def test_009TPwk7i2wA7UA8QEjAo4_vocal(self):
        path_to_audio = '/tmp/segments/009TPwk7i2wA7UA8QEjAo4-copy_0-22.ogg'
        separate(path_to_audio, only_vocals=True, output_dir='only_vocals')

    def test_009TPwk7i2wA7UA8QEjAo4_fast(self):
        path_to_audio = '/tmp/segments/009TPwk7i2wA7UA8QEjAo4-copy_0-22.ogg'
        separate(path_to_audio, fast=True, output_dir='fast')
        

