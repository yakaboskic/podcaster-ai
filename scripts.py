import subprocess
import json
import sys
import os
import ray
import pandas as pd
import numpy as np
import logging
import compress_pickle
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy



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
    #from podcaster_ai.util.spotify_data import get_client, get_box_fileids
    from podcaster_ai.pipelines.detect_and_segment import detect_and_segment_distributed
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

    local_dir = '/home/public/data/spotify/gimlet'
    local_files = os.listdir(local_dir)

    #with open('data/spotify-fileids.json', 'r') as f_:
    #    fileids = json.load(f_)
    ## Load box file ids
    local_files = list(set(local_files) - processed_files_set)
    # Put absolute path on local files
    local_files = [os.path.join(local_dir, f) for f in local_files]
    #print(f'Will process {len(box_files)} files that have not yet been considered')

    # Initialize ray
    ray.init(address='auto')
    
    # Build placement group
    num_nodes = len(ray.nodes())
    bundles = [{"CPU": num_cpus} for _ in range(num_nodes)]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    # Batch files
    batched_files = [[] for _ in range(num_nodes * num_cpus)]
    while local_files:
        for i in range(len(batched_files)):
            try:
                f = local_files.pop()
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
                            files=batch,
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

def download_gimlet():
    from podcaster_ai.util.spotify_data import get_client, get_file
    from podcaster_ai.util.mp import MPLogger
    # First arg is number of cpus per node
    num_cpus = int(sys.argv[-1])
    download_dir = '/tmp/gimlet'
    metadata = pd.read_csv('/home/cyakaboski/data/metadata.tsv', delimiter='\t')
    gimlet_meta = metadata[metadata.publisher == 'Gimlet']
    episodes = []
    for episode in gimlet_meta.episode_filename_prefix:
        episodes.append(f'{episode}.ogg')
    box_files = episodes
    
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

    # simple remote function to download
    @ray.remote(num_cpus=1)
    def download(
            box_files,
            worker_idx,
            save_dir,
            fileids_map_path = '/home/cyakaboski/src/python/projects/podcaster-ai/data/spotify-fileids.json',
            ):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logger = MPLogger('Worker', logging.INFO, id=worker_idx, loop_report_time=60)
        client = get_client()
        logger.initialize_loop_reporting()
        for i, f in enumerate(box_files):
            _ = get_file(client, f, fileids_map_path=fileids_map_path, save_dir=save_dir)
            logger.report(i=i, total=len(box_files))
        logger.info('Complete.')
        return 

    # Setup results
    result_ids = []
    worker_idx = 0
    for bundle_idx, bundle_batch in enumerate(bundled_batches):
        for batch in bundle_batch:
            result_ids.append(
                    download.options(
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=bundle_idx,
                            )
                        ).remote(
                            batch,
                            worker_idx,
                            download_dir,
                            )
                        )
            worker_idx += 1
    # Wait for results
    while result_ids:
        done_ids, result_ids = ray.wait(result_ids)
        _ = ray.get(done_ids)
    print('Complete.')

def separate_gimlet():
    from podcaster_ai.separation import separate
    from podcaster_ai.util.mp import MPLogger
    # First arg is number of cpus per node
    num_cpus = int(sys.argv[-1])
    
    # Get files to separate
    segments_dir = '/home/public/data/spotify/gimlet-segments'
    output_dir = '/home/public/data/spotify/gimlet-seperated'
    files = os.listdir(segments_dir)
    files = [os.path.join(segments_dir, f) for f in files]

    # Initialize ray
    ray.init(address='auto')
    
    # Build placement group
    num_nodes = len(ray.nodes())
    bundles = [{"CPU": num_cpus} for _ in range(num_nodes)]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    # Batch files
    batched_files = [[] for _ in range(num_nodes * num_cpus)]
    while files:
        for i in range(len(batched_files)):
            try:
                f = files.pop()
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

    # simple remote function to download
    @ray.remote(num_cpus=1)
    def separate_distributed(
            files,
            worker_idx,
            output_dir,
            ):
        # Redirect stdout for cleaner logs
        logger = MPLogger('Worker', logging.INFO, id=worker_idx, loop_report_time=60)
        logger.initialize_loop_reporting()
        for i, f in enumerate(files):
            separate(f, output_dir=output_dir, only_vocals=True, fast=True, no_output=True)
            logger.report(i=i, total=len(files))
        logger.info('Complete.')
        return 

    # Setup results
    result_ids = []
    worker_idx = 0
    for bundle_idx, bundle_batch in enumerate(bundled_batches):
        for batch in bundle_batch:
            result_ids.append(
                    separate_distributed.options(
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=bundle_idx,
                            )
                        ).remote(
                            batch,
                            worker_idx,
                            output_dir,
                            )
                        )
            worker_idx += 1
    # Wait for results
    while result_ids:
        done_ids, result_ids = ray.wait(result_ids)
        _ = ray.get(done_ids)
    print('Complete.')

def predict_gimlet_midlevel():
    from podcaster_ai.midlevel_features import predict_midlevel_from_audiofiles
    from podcaster_ai.util.mp import MPLogger
    # First arg is number of cpus per node
    num_cpus = int(sys.argv[-1])
    
    # Load dataset to get column names
    with open('soundtrack-emotion-midlevel-complete-discretized_3.lz4', 'rb') as f_:
        _, _, cols = compress_pickle.load(f_)

    # Get files to separate
    separate_dir = '/home/public/data/spotify/gimlet-seperated/83fc094f'
    model_path = '/home/cyakaboski/src/python/projects/podcaster-ai/models/midlevel-model.h5'
    dirs = os.listdir(separate_dir)
    files = [os.path.join(separate_dir, d, 'no_vocals.wav') for d in dirs]

    # Initialize ray
    ray.init(address='auto')
    
    # Build placement group
    num_nodes = len(ray.nodes())
    bundles = [{"CPU": num_cpus} for _ in range(num_nodes)]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    # Batch files
    batched_files = [[] for _ in range(num_nodes * num_cpus)]
    while files:
        for i in range(len(batched_files)):
            try:
                f = files.pop()
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

    # simple remote function to download
    @ray.remote(num_cpus=1)
    def predict_distributed(
            files,
            model_path,
            worker_idx,
            ):
        # Redirect stdout for cleaner logs
        logger = MPLogger('Worker', logging.INFO, id=worker_idx, loop_report_time=60)
        logger.initialize_loop_reporting()
        preds = predict_midlevel_from_audiofiles(files, model_path, logger)
        logger.info('Complete.')
        return files, preds

    # Setup results
    result_ids = []
    worker_idx = 0
    for bundle_idx, bundle_batch in enumerate(bundled_batches):
        for batch in bundle_batch:
            result_ids.append(
                    predict_distributed.options(
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=bundle_idx,
                            )
                        ).remote(
                            batch,
                            model_path,
                            worker_idx,
                            )
                        )
            worker_idx += 1
    # Wait for results
    results = []
    while result_ids:
        done_ids, result_ids = ray.wait(result_ids)
        y_files, preds = ray.get(done_ids)[0]
        for f, pred in zip(y_files, preds):
            f_name = os.path.split(os.path.split(f)[0])[1]
            name, time_int = f_name.split('_')
            start_int, end_int = time_int.split('-')
            results.append([name, start_int, end_int] + pred)
    df = pd.DataFrame.from_records(np.array(results), columns = ['file_name', 'start_time', 'end_time'] + cols)
    df.to_csv('gimlet_midlevel_predictions.csv')
    print('Complete.')

def gimlet_topics():
    from podcaster_ai.topic import find_topics
    metadata = pd.read_csv('/home/cyakaboski/data/metadata.tsv', delimiter='\t')
    unique_desc = list(metadata.show_description.unique().astype('U'))
    unique_desc_df = pd.DataFrame(unique_desc, columns=['show_description'])
    print(unique_desc_df)
    print(f'Num descriptions {len(unique_desc)}')
    stop_words = list(set(pd.read_csv('data/stopwords.csv').iloc[:,0].values.astype('U')))
    ass, words, _ = find_topics(unique_desc, n_components=50, verbose=1, stop_words=stop_words)
    unique_desc_df['ass'] = ass
    gimlet_desc = metadata[metadata.publisher == 'Gimlet'].show_description.unique().astype('U')
    print(unique_desc_df[unique_desc_df.show_description.isin(gimlet_desc)])
    #print(ass)
    print(words)
