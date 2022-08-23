import unittest
import os

from podcaster_ai.pipelines.detect_and_segment import DetectAndSegmentPipeline

class TestDetectAndSegment(unittest.TestCase):

    def test_009TPwk7i2wA7UA8QEjAo4_local(self):
        path_to_audio = '/tmp/009TPwk7i2wA7UA8QEjAo4-copy.ogg'
        pipeline = DetectAndSegmentPipeline(files=[path_to_audio], output_dir='/tmp/segments')
        seg_paths, thresholded = pipeline.run()
        print(seg_paths)
        print(thresholded)
    
    def test_009TPwk7i2wA7UA8QEjAo4_box(self):
        path_to_audio = '009TPwk7i2wA7UA8QEjAo4.ogg'
        pipeline = DetectAndSegmentPipeline(box_files=[path_to_audio], output_dir='/tmp/segments')
        seg_paths, thresholded = pipeline.run()
        print(seg_paths)
        print(thresholded)
