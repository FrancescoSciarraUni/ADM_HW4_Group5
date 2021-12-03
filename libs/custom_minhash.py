import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import subprocess
import librosa
import librosa.display
#import IPython.display as ipd
import itertools
import json
from collections import defaultdict
from random import randrange
from random import *
from pathlib import Path, PurePath
from tqdm.notebook import tqdm
from utility_function import *

N_TRACKS = 1413
HOP_SIZE = 512
DURATION = 30  # TODO: to be tuned!
THRESHOLD = 0  # TODO: to be tuned!


def get_duration():
    return DURATION


def get_threshold():
    return THRESHOLD


def set_threshold(thrshld):
    global THRESHOLD
    THRESHOLD = thrshld


def set_duration(duration):
    global DURATION
    DURATION = duration


def isPrime(n):
    '''
    :param n: a integer number
    :return: true or false if n is a prime number or not
    '''
    
    if n <= 1:
        return False

    # check from 2 to n-1
    for i in range(2, n):
        if n % i == 0:
            return False

    return True

def getPrime(n):

    '''
    :param n: a integer number
    :return: return the biggest prime number smaller than n
    '''
    
    out = []
    for i in range(2, n + 1):
        if isPrime(i):
            out.append(i)
    return out[-1]


def get_ch_matrix(tracks, DURATION, HOP_SIZE):

    '''

    :param tracks: list of tracks
    :param DURATION: duration of the sample
    :param HOP_SIZE: integer
    :return: return characteristics matrix that has number of columns equal to the number of the tracks in the dataset and has number
            of rows equal to the max peak in the tracks. matrix[i][j] = 0 if the peak is not in the track, otherwise matrix[i][j] = 1 if the peak is in the track
    '''
    
    dict_tracks = {}
    for idx, audio in tqdm(enumerate(tracks)):
        track, sr, onset_env, peaks = load_audio_picks(audio, DURATION, HOP_SIZE)
        # print(peaks)
        if idx not in dict_tracks.keys():
            dict_tracks[idx] = peaks.tolist()
    n_rows = max(list(itertools.chain(*dict_tracks.values()))) + 1
    matrix = np.zeros((n_rows, len(dict_tracks.values())))
    for i in range(n_rows):
        for j in range(len(dict_tracks.values())):
            if i in dict_tracks[j]:
                # print(i,j,dict_tracks[j])
                matrix[i][j] = 1
    return matrix


# CAPIRE COME PRENDERE IL MAX_ROWS (DOVREBBE ESSERE IL NUMERO MASSIMO DI RIGHE DELLA MATRICE DI TUTTE LE CANZONI)

def get_signature_matrix(debug_matrix, n_hash, max_rows):


    '''
    :param debug_matrix: matrix of characteristics
    :param n_hash: number of random hash function that we want to use to represent a song
    :param max_rows: numbers of rows of the signature matrix
    :return: return the hash signature matrix, that contains for every song its hash signature
    '''
    
    rows, cols = debug_matrix.shape
    sig_matrix = np.ones((n_hash, cols)) * np.inf
    seed(10)
    prime = getPrime(max_rows * 2)
    l_random = [randrange(prime - 1) for i in range(n_hash)]
    for row in range(rows):
        for col in range(cols):
            l_supp = []
            for h in l_random:
                l_supp.append((row * h + 1) % prime)
            if debug_matrix[row][col] == 1:
                for i in range(n_hash):
                    if l_supp[i] < sig_matrix[i][col]:
                        sig_matrix[i][col] = l_supp[i]
    return sig_matrix

def jaccard_sim(query_matrix,db_matrix):
    '''
    :param query_matrix:  is one of the column of the signature matrix for the query songs
    :param db_matrix: signature matrix for all songs in the dataset
    '''
    dist = 0
    j = 0
    l_out = []
    for i in range(len(db_matrix.T)):
        #forse conviene ritornarla già con il reshape la query matrix
        s_query = set(query_matrix.reshape(query_matrix.shape[0]))
        s_db = set(db_matrix.T[i])
        intersection = s_query.intersection(s_db)
        #intersection = len(np.intersect1d(query_matrix.reshape(query_matrix.shape[0],),db_matrix.T[i]))
        union = s_query.union(s_db)
        j = float(len(intersection)) / len(union)
        l_out.append([j,i])
        if j >= dist:
            index_song = i
            dist = j
    #return dist,index_song,
    return l_out



