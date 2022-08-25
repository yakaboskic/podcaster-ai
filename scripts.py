import subprocess
import json
import sys
import os

from podcaster_ai.util.spotify_data import get_client, get_box_fileids

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
