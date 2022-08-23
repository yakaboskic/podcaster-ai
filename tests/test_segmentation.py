import unittest
import os

from podcaster_ai.segmentation import segment

class TestSegmentation(unittest.TestCase):

    def test_example_1(self):
        path_to_audio = 'tests/example-1.wav'
        output_filepath = 'tests/test_example_1_segment.ogg'

        if os.path.exists(output_filepath):
            os.remove(output_filepath)

        audio_segment = segment(path_to_audio, start_time=1, end_time=3, output_filepath='tests/test_example_1_segment.ogg')
        self.assertTrue(os.path.exists(output_filepath))
        self.assertEqual(audio_segment.duration_seconds, 2)
