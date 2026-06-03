#!/usr/bin/env python
"""Extract audio tracks from movies and write them to audio/

Usage: convertaudio.py FILENAME
where FILENAME is a text file containing filenames of movies, one per line.
"""
import sys
import os
from os.path import basename, dirname
import subprocess
from tqdm import tqdm


def process(movie, output):
    subprocess.check_call(['ffmpeg',
                           '-i', movie,
                           '-vn', '-ac', '1', '-ar', '16000',
                           'audio/' + output
                           ])


def main():
    os.makedirs('audio', exist_ok=True)
    if len(sys.argv) < 2:
        print(__doc__)
        return
    with open(sys.argv[1]) as inp:
        movies = inp.read().strip().splitlines()
    for n, movie in enumerate(movies, 1):
        print(f'{n}/{len(movies)}. {movie}')
        output = basename(dirname(movie)).removesuffix('/') + '.mp3'
        process(movie, output)


if __name__ == '__main__':
    main()
