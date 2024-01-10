import unittest
import os

from podcaster_ai.detection import detect, load_detection_model

class TestDetection(unittest.TestCase):

    def setUp(self):
        self.model = load_detection_model()

    def test_example_1(self):
        path_to_audio = 'tests/example-1.wav'
        truth = {
                "music": [(0.0, 3.053061224489796)],
                "speech":  [(2.0253968253968253, 8.0)],
                }
        preds = detect(self.model, path_to_audio)
        self.assertDictEqual(preds, truth)

    def test_example_2(self):
        path_to_audio = 'tests/example-2.wav'
        truth = {
                "music": [(0.0, 8.0)],
                "speech":  [(5.976417233560091, 8.0)],
                }
        preds = detect(self.model, path_to_audio)
        self.assertDictEqual(preds, truth)

    def test_example_3(self):
        path_to_audio = 'tests/example-3.wav'
        truth = {
                "music": [(0.0, 8.0)],
                "speech": [(0.0, 4.729251700680272)],
                }
        preds = detect(self.model, path_to_audio)
        self.assertDictEqual(preds, truth)

    def test_this_american_life(self):
        path_to_audio = 'data/this_american_life-241-20_acts_in_60_minutes-quarter.mp3'
        preds = detect(self.model, path_to_audio)
        print(preds)
