import os
import tqdm
import numpy as np
from pydub import AudioSegment

from podcaster_ai.detection import detect, load_detection_model
from podcaster_ai.segmentation import segment
from podcaster_ai.util.spotify_data import *

class DetectAndSegmentPipeline:
    def __init__(
            self,
            files:list=None,
            box_files:list=None,
            total_music_threshold_seconds:int=30,
            segment_music_threshold_seconds:int=5,
            output_dir:str=None,
            path_to_detection_model_weights:str='models/model_d-DS.h5',
            fileids_map_path = 'data/spotify-fileids.json',
            ):
        self.files = files
        self.box_files = box_files
        self.total_music_threshold_seconds = total_music_threshold_seconds
        self.segment_music_threshold_seconds = segment_music_threshold_seconds
        self.output_dir = output_dir
        self.path_to_detection_model_weights = path_to_detection_model_weights
        self.fileids_map_path = fileids_map_path
        if self.box_files:
            self.client = get_client()
        else:
            self.client = None

    def run(self, verbose=False):
        model = load_detection_model(self.path_to_detection_model_weights)
        below_threshold = []
        segmentation_dict = {}
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if self.files:
            for f_ in tqdm.tqdm(self.files, disable=not verbose, leave=False, desc='Processing local files'):
                segmentation_dict, below_threshold = self.detect_and_segment(f_, model, segmentation_dict, below_threshold)
        if self.box_files:
            for f_ in tqdm.tqdm(self.box_files, disable=not verbose, leave=False, desc='Processing box files'):
                segmentation_dict, below_threshold = self.detect_and_segment(f_, model, segmentation_dict, below_threshold, is_box_file=True)
        return segmentation_dict, below_threshold

    def detect_and_segment(self, audio_file, model, segmentation_dict, below_threshold, is_box_file=False):
        # Download box file to tmp before processing
        if is_box_file:
            audio_file = get_file(self.client, audio_file, fileids_map_path=self.fileids_map_path)
        res = detect(model, audio_file)
        if self.above_music_threshold(res):
            segmentation_paths = self.segment_detection_results(res, audio_file)
            segmentation_dict[audio_file] = segmentation_paths
        else:
            below_threshold.append(audio_file)
        # Delete the box audio file after processing
        if is_box_file:
            os.remove(audio_file)
        return segmentation_dict, below_threshold

    def segment_detection_results(self, detection_results, audio_file):
        music_res = detection_results["music"]
        audio = AudioSegment.from_ogg(audio_file)
        # OS name operations
        audio_dir, audio_name_wext = os.path.split(audio_file)
        audio_name, audio_ext = os.path.splitext(audio_name_wext)
        segmentation_paths = []
        # Process each segmentation
        for m_res_start, m_res_end in music_res:
            # Convert to ints
            m_res_start = int(np.floor(m_res_start))
            m_res_end = int(np.ceil(m_res_end))
            # Test if music segment is longer than threshold
            if m_res_end - m_res_start < self.segment_music_threshold_seconds:
                continue
            segment_name = f'{audio_name}_{m_res_start}-{m_res_end}{audio_ext}'
            segment_savepath = os.path.join(self.output_dir, segment_name)
            _ = segment(audio=audio, start_time=m_res_start, end_time=m_res_end, output_filepath=segment_savepath)
            segmentation_paths.append(segment_savepath)
        return segmentation_paths

    def above_music_threshold(self, detection_results):
        music_res = detection_results["music"]
        total_music_time = 0
        for m_res_start, m_res_end in music_res:
            total_music_time += m_res_end - m_res_start
        if total_music_time < self.total_music_threshold_seconds:
            return False
        return True
