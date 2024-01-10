import os
import logging
import tqdm
import numpy as np
import ray
from pydub import AudioSegment

from podcaster_ai.detection import detect, load_detection_model
from podcaster_ai.segmentation import segment
from podcaster_ai.util.spotify_data import *
from podcaster_ai.util.mp import MPLogger

class DetectAndSegmentPipeline:
    def __init__(
            self,
            files:list=None,
            box_files:list=None,
            total_music_threshold_seconds:int=30,
            segment_music_threshold_seconds:int=5,
            output_dir:str=None,
            path_to_detection_model_weights:str='data/model_d-DS.h5',
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

    def run(self, verbose=False, logger=None):
        model = load_detection_model(self.path_to_detection_model_weights)
        below_threshold = []
        segmentation_dict = {}
        detection_problems = []
        segmentation_problems = []
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if self.files:
            if logger:
                logger.initialize_loop_reporting()
            total = len(self.files)
            for i, f_ in tqdm.tqdm(enumerate(self.files), total=total, disable=not verbose, leave=False, desc='Processing local files'):
                segmentation_dict, below_threshold, detection_problems, segmentation_problems = self.detect_and_segment(
                        f_,
                        model,
                        segmentation_dict,
                        below_threshold,
                        detection_problems,
                        segmentation_problems,
                        )
                if logger:
                    logger.report(i=i, total=total)
        if self.box_files:
            if logger:
                logger.initialize_loop_reporting()
            total = len(self.box_files)
            for i, f_ in tqdm.tqdm(enumerate(self.box_files), total=total, disable=not verbose, leave=False, desc='Processing box files'):
                segmentation_dict, below_threshold, detection_problems, segmentation_problems = self.detect_and_segment(
                        f_,
                        model,
                        segmentation_dict,
                        below_threshold,
                        detection_problems,
                        segmentation_problems,
                        is_box_file=True,
                        )
                if logger:
                    logger.report(i=i, total=total)
        return segmentation_dict, below_threshold, detection_problems, segmentation_problems

    def detect_and_segment(
            self,
            audio_filename,
            model,
            segmentation_dict,
            below_threshold,
            detection_problems,
            segmentation_problems,
            is_box_file=False
            ):
        # Download box file to tmp before processing
        if is_box_file:
            audio_file = get_file(self.client, audio_filename, fileids_map_path=self.fileids_map_path)
            if audio_file is None:
                return segmentation_dict, below_threshold, detection_problems, segmentation_problems
        else:
            audio_file = audio_filename
            audio_filename = os.path.basename(audio_file)
        try:
            res = detect(model, audio_file)
        except:
            # Delete the box audio file after processing
            if is_box_file:
                os.remove(audio_file)
            print(f'Detection issue with {audio_filename}')
            detection_problems.append(audio_filename)
            return segmentation_dict, below_threshold, detection_problems, segmentation_problems
        if self.above_music_threshold(res):
            try:
                segmentation_paths = self.segment_detection_results(res, audio_file)
            except:
                # Delete the box audio file after processing
                if is_box_file:
                    os.remove(audio_file)
                print(f'Segmentation issue with {audio_filename}')
                segmentation_problems.append(audio_filename)
                return segmentation_dict, below_threshold, detection_problems, segmentation_problems
            segmentation_dict[audio_file] = segmentation_paths
        else:
            below_threshold.append(audio_filename)
        # Delete the box audio file after processing
        if is_box_file:
            os.remove(audio_file)
        return segmentation_dict, below_threshold, detection_problems, segmentation_problems

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
        try:
            music_res = detection_results["music"]
        except KeyError:
            # Means to music was detected
            return False
        total_music_time = 0
        for m_res_start, m_res_end in music_res:
            total_music_time += m_res_end - m_res_start
        if total_music_time < self.total_music_threshold_seconds:
            return False
        return True

@ray.remote(num_cpus=1)
def detect_and_segment_distributed(
            files:list=None,
            box_files:list=None,
            total_music_threshold_seconds:int=30,
            segment_music_threshold_seconds:int=5,
            output_dir:str=None,
            path_to_detection_model_weights:str='data/model_d-DS.h5',
            fileids_map_path = 'data/spotify-fileids.json',
            position=0,
            ):
    # Build pipeline
    pipeline = DetectAndSegmentPipeline(
            files=files,
            box_files=box_files,
            total_music_threshold_seconds=total_music_threshold_seconds,
            segment_music_threshold_seconds=segment_music_threshold_seconds,
            output_dir=output_dir,
            path_to_detection_model_weights=path_to_detection_model_weights,
            fileids_map_path=fileids_map_path,
            )
    # run pipeline
    logger = MPLogger(f'Worker', logging.INFO, id=position, loop_report_time=60)
    seg_paths, thresholded, detection_problems, segmentation_problems = pipeline.run(verbose=False, logger=logger)
    logger.info('Complete.')
    return (seg_paths, thresholded, detection_problems, segmentation_problems)
