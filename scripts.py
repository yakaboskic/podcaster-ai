import subprocess
import json
import sys
import os
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from podcaster_ai.util.spotify_data import get_client, get_box_fileids
from podcaster_ai.pipelines.detect_and_segment import detect_and_segment_distributed

def test():
    """
    Run all unittests. Equivalent to:
    `poetry run python -u -m unittest discover`
    """
    subprocess.run(
        ['python', '-u', '-m', 'unittest', 'discover']
    )

def test_util():
    """
    Run all unittests. Equivalent to:
    `poetry run python -u -m unittest tests/test_util.py'
    """
    subprocess.run(
        ['python', '-u', '-m', 'unittest', 'tests/test_util.py']
    )

def test_detection():
    """
    Run all unittests. Equivalent to:
    `poetry run python -u -m unittest tests/test_detection.py'
    """
    subprocess.run(
        ['python', '-u', '-m', 'unittest', 'tests/test_detection.py']
    )

def test_segmentation():
    """
    Run all unittests. Equivalent to:
    `poetry run python -u -m unittest tests/test_segmentation.py'
    """
    subprocess.run(
        ['python', '-u', '-m', 'unittest', 'tests/test_segmentation.py']
    )

def test_separation():
    """
    Run all unittests. Equivalent to:
    `poetry run python -u -m unittest tests/test_separation.py'
    """
    subprocess.run(
        ['python', '-u', '-m', 'unittest', 'tests/test_separation.py']
    )

def test_pipelines():
    """
    Run all unittests. Equivalent to:
    `poetry run python -u -m unittest tests/test_pipelines.py'
    """
    subprocess.run(
        ['python', '-u', '-m', 'unittest', 'tests/test_pipelines.py']
    )

def fileids():
    """ Get a json file map of all filenames to box file ids.
    """
    DIR_BLACKLIST = [
            'opensmile',
            'podcast_pyserini_indices',
            'yamnet',
            'podcasts-audio-summarization-2020-testset',
            'podcasts-audio-summarization-2021-testset',
            ]
    client = get_client()
    fileids_map = get_box_fileids(client, dir_blacklist=DIR_BLACKLIST)
    with open('spotify-fileids.json', 'w') as f_:
        json.dump(fileids_map, f_)

def collect_audio_data():
    import compress_pickle
    import numpy as np
    import concurrent.futures as cf
    import tqdm
    import librosa
    import audioread.ffdec
    from operator import itemgetter

    dir_path = sys.argv[-2]
    out_path = sys.argv[-1]

    files = os.listdir(dir_path)

    def process_file(path):
        name, _ = os.path.splitext(path)
        aro = audioread.ffdec.FFmpegAudioFile(os.path.join(dir_path, path))
        x, _ = librosa.load(aro)
        return (int(name), x)

    with cf.ThreadPoolExecutor() as executor:
        results = list(tqdm.tqdm(executor.map(process_file, files), total=len(files)))

    results = sorted(results, key=itemgetter(0))
    np_res = [arr for _, arr in results]
    with open(out_path, 'wb') as data_file:
        compress_pickle.dump(np_res, data_file)
    print('Complete.')

def detect_segment_spotify():
    # First arg is number of cpus per node
    num_cpus = int(sys.argv[-2])
    # Second arg is output_dir
    output_dir = sys.argv[-1]
    # Read output dir and build set of already processed files
    processed_files = os.listdir(output_dir)
    processed_files_set = set()
    for f_ in processed_files:
        filename, ext = os.path.splitext(f_)
        if ext == 'ogg':
            audio_name, _ = filename.split('_')
            processed_files_set.add(f'{audio_name}.{ext}')
    # Read output dir and read any files to not process in the future
    no_process_file = os.path.join(output_dir, 'no_process.json')
    no_process_dict = {}
    if os.path.exists(no_process_file):
        with open(no_process_file, 'r') as f_:
            no_process_dict = json.load(f_)
    # Add no process files to processed files above
    for f in no_process_dict:
        processed_files_set.add(f_)

    with open('data/spotify-fileids.json', 'r') as f_:
        fileids = json.load(f_)
    # Load box file ids
    box_files = list(set(fileids.keys()) - processed_files_set)[:1500]
    print(f'Will process {len(box_files)} files that have not yet been considered')
    
    # Initialize ray
    ray.init(address='auto')
    
    # Build placement group
    num_nodes = len(ray.nodes())
    bundles = [{"CPU": num_cpus} for _ in range(num_nodes)]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    # Batch files
    batched_files = [[] for _ in range(num_nodes * num_cpus)]
    while box_files:
        for i in range(len(batched_files)):
            try:
                f = box_files.pop()
                batched_files[i].append(f)
            except IndexError:
                break
    bundled_batches = [[] for _ in range(num_nodes)]
    while batched_files:
        for i in range(num_nodes):
            try:
                batch = batched_files.pop()
                bundled_batches[i].append(batch)
            except IndexError:
                break
    
    # Setup results
    result_ids = []
    worker_idx = 0
    for bundle_idx, bundle_batch in enumerate(bundled_batches):
        for batch in bundle_batch:
            result_ids.append(
                    detect_and_segment_distributed.options(
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=bundle_idx,
                            )
                        ).remote(
                            box_files=batch,
                            position=worker_idx,
                            output_dir=output_dir,
                            )
                        )
            worker_idx += 1
    # Collect results
    seg_paths = {}
    thresholded = []
    detect_problems = []
    seg_problems = []
    while result_ids:
        done_ids, result_ids = ray.wait(result_ids)
        _seg_paths, _thresholded, _detect_problems, _seg_problems = ray.get(done_ids)[0]
        seg_paths.update(_seg_paths)
        thresholded.extend(_thresholded)
        detect_problems.extend(_detect_problems)
        seg_problems.extend(_seg_problems)
    # Build json result structure
    results = {
            "Successful": seg_paths,
            "Below Threshold": thresholded,
            "Detection Problems": detect_problems,
            "Segmentation Problems": seg_problems,
            }
    # Add any files that couldn't be processed to the no_process json file
    print('Writing no process file.')
    for f in thresholded:
        no_process_dict[f] = "below_threshold"
    for f in detect_problems:
        no_process_dict[f] = "detection_problem"
    for f in seg_problems:
        no_process_dict[f] = "segmentation_problem"
    with open(no_process_file, 'w') as f_:
        json.dump(no_process_dict, f_, indent=2)
    print('Writing json results file.')
    with open('detect_segment_spotify_results.json', 'w') as res_file:
        json.dump(results, res_file, indent=2)
