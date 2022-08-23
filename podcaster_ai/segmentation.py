import numpy as np
from pydub import AudioSegment


def segment(path_to_audio=None, audio=None, start_time=-1, end_time=-1, output_filepath=None):
    """ Function that will segment an audio clip. Assumes time is in seconds.
    """
    # pydub works in milliseconds so convert
    if start_time != -1:
        start_time = int(np.floor(1000 * start_time))
    if end_time != -1:
        end_time = int(np.ceil(1000 * end_time))
    # Load ogg audio
    if not audio:
        audio = AudioSegment.from_ogg(path_to_audio)
    # Segment
    if start_time == -1 and end_time == -1:
        audio_segment = audio
    elif start_time == -1:
        audio_segment = audio[:end_time]
    elif end_time == -1:
        audio_segment = audio[start_time:]
    else:
        audio_segment = audio[start_time:end_time]
    if output_filepath:
        audio_segment.export(output_filepath, format="ogg")
    return audio_segment
