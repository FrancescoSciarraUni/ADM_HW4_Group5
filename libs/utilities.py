import json
from collections import defaultdict
from custom_minhash import *
from pathlib import Path, PurePath   

def matrix_to_json(sig_matrix,file_name):
    '''

    :param sig_matrix: hash signature matrix
    :param file_name: file name where the matrix as dict will be written
    :return: dictionary that represenent the hash signature matrix
    '''
    
    dict_sig = {}
    Path("./json").mkdir(parents=True, exist_ok=True)
    for i in range(len(sig_matrix.T)):
        dict_sig[i] = list(sig_matrix.T[i])
    with open("json/"+file_name, 'w') as fp:
        json.dump(dict_sig, fp)
    return dict_sig

def track_list_to_json(tracks,path_out):

    '''
    :param tracks: the list of tracks
    :param path_out: path where the json will be written
    :return: a dictionary id_track: (artist, song name)
    '''
        

    track_list_dict = {}
    Path("./json").mkdir(parents=True, exist_ok=True)
    for i in enumerate(tracks):
        full_name = str(i[1]).split('\\')[-1]
        artist = str(i[1]).split('\\')[-3]
        name = full_name.split('-')[-1][:-4]
        track_list_dict[i[0]] = (artist,name)
    with open("json/"+path_out, "w") as outfile:
        json.dump(track_list_dict, outfile)
    return track_list_dict


def search_song(path_query, path_ds, treshold=THRESHOLD):

    '''
    :param path_query: file name of the json file that contains the signature matrix for query songs
    :param path_ds: file name of the json file that contains the signature matrix for dataset songs
    :param treshold: treshold that can be modified to search similar songs
    :return: dictionary made like this {query song id: list of song similar to the query song}
    '''
    
    bins_dict = defaultdict(list)
    l_dist = []
    # q_sig_matrix = CARICO IL JSON DELLE SIGNATURE DELLE QUERY NELLA MATRICE
    with open(f'./json/{path_query}') as json_file:
        queries_dict = json.load(json_file)
    queries_m = np.array([list(v) for v in queries_dict.values()]).T
    with open(f'./json/{path_ds}') as json_file:
        d_dict = json.load(json_file)
    d_m = np.array([list(v) for v in d_dict.values()]).T

    for i in range(len(queries_m[0])):
        l_q = queries_m[:, i]
        l_dist.append(jaccard_sim(l_q, d_m))
    for distances in range(len(l_dist)):
        for distance in l_dist[distances]:
            if distances in bins_dict.keys():
                if distance[0] >= treshold:
                    bins_dict[distances].append(distance)
            else:
                if distance[0] >= treshold:
                    bins_dict[distances] = [distance]
    for k in range(len(bins_dict.keys())):
        bins_dict[k].sort(key = lambda x: x[0])
        bins_dict[k] = bins_dict[k]
    return bins_dict


def get_song_name(bins_dict):

    '''
    :param bins_dict: dictionary made like this {query song id: list of couples (similarity to the query song,song_id)}
    :return: dictionary that has as values a list of list [similarity, [artist name, song name]]
    '''
    with open(f'./json/track_list.json') as json_file:
        track_list= json.load(json_file)
    for k,v in bins_dict.items():
        for i in v:
            song_name = track_list[str(i[1])]
            i[1] = song_name
    return bins_dict
    # track_list = carico track_list.json
    # distance, song_index =  jaccard_sim(query_matrix, db_matrix)
    # return track_list[song_index]
    
    
def get_rows(cluster_dict, point_matrix):
    '''

    :param cluster_dict: the dictionary that represent how data points are divided into clusters, whit keys the cluster id and as keys the list of row indexes that are in the cluster
    :param point_matrix: a multidimensional numpy array
    :return: a dictionary that contains the data points instead of index
    '''

    s_dict = cluster_dict
    for k, v in s_dict.items():
        supp = []
        for val in v:
            supp.append(point_matrix[val])
        s_dict[k] = supp
    return s_dict