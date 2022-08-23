import subprocess
import json

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
