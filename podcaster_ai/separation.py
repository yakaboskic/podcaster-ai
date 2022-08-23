import subprocess


def separate(path_to_audio, output_dir=None, only_vocals=False, fast=False):
    """ Wrapper for separating audio with Facebook Demucs.
    """
    cmd = ['python3', '-m', 'demucs', '-d', 'cpu']
    if output_dir:
        cmd.extend(['-o', output_dir])
    if only_vocals:
        cmd.extend(['--two-stems', 'vocals'])
    if fast:
        cmd.extend(['-n', '83fc094f'])
    cmd.append(path_to_audio)
    subprocess.run(
            cmd
    )
