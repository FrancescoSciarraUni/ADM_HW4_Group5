import pandas as pd
import numpy as np
from collections import defaultdict
import math
import seaborn as sns
import matplotlib.pyplot as plt
import utilities


def k_means_vecchio(X, k):
    '''
        :param X: multidimensional numpy array
        :param k: number of clusters
    '''
    clusters = np.zeros(X.shape[0])
    conv = False
    number_of_rows = X.shape[0]
    random_indices = np.random.choice(number_of_rows,
                                      size=k,
                                      replace=False)
    # display random rows
    centers = X[random_indices, :]

    while not conv:
        test_d = defaultdict(list)
        for i in range(len(X)):
            new_dist = float('inf')
            for k in range(len(centers)):
                # print(centroids[k])
                sum_ = 0
                for h in range(len(centers[0])):
                    sum_ += (X[i][h] - centers[k][h]) ** 2
                # print("-------",sum_)
                dist = np.sqrt(sum_)
                # print("distance = ",dist)
                # print("values==>",X[i],centroids[k],new_dist,dist)
                if new_dist > dist:
                    new_dist = dist
                    # IN QUELLO NUOVO HO CAMBIATO QUESTA RIGA
                    point_min_dist = X[i]
                    clusters[i] = k
                    dict_ind = k
            # print("point_min_dist==>",point_min_dist,dict_ind)
            # print()
            test_d[dict_ind].append(point_min_dist)
        new_centers = []
        for i in range(len(test_d.keys())):
            np_l = np.array(test_d[i])
            centr = np_l.mean(axis=0)
            new_centers.append(centr)
        new_centers = np.array(new_centers)
        # print(new_centers)
        # print()
        # print(centers)
        if (centers == new_centers).all():
            conv = True
        else:
            centers = new_centers
    return centers, test_d


def k_means(X, k):
    '''
    
    :param X: multidimensional numpy array
    :param k: number of clusters
    :return: centers of the clusters and a dictionary that represent the clustering {id of the cluster: rows' indexes of the original dataset}
    '''
    
    clusters = np.zeros(X.shape[0])
    # centers = data.sample(n=k).values
    conv = False
    number_of_rows = X.shape[0]
    random_indices = np.random.choice(number_of_rows,
                                      size=k,
                                      replace=False)
    # display random rows
    centers = X[random_indices, :]

    while not conv:
        test_d = defaultdict(list)
        for i in range(len(X)):
            new_dist = float('inf')
            for k in range(len(centers)):
                sum_ = 0
                for h in range(len(centers[0])):
                    sum_ += (X[i][h] - centers[k][h]) ** 2
                dist = np.sqrt(sum_)
                if new_dist > dist:
                    new_dist = dist
                    # point_min_dist = X[i]
                    # QUESTA è LA DIFFERENZA CON QUELLO VECCHIO point_mind_dist è diventato solo l'indice
                    # prima era X[i]
                    point_min_dist = i
                    clusters[i] = k
                    dict_ind = k

            test_d[dict_ind].append(point_min_dist)
        new_centers = []
        for i in range(len(test_d.keys())):
            np_l = np.array(X[test_d[i]])
            centr = np_l.mean(axis=0)
            new_centers.append(centr)
        new_centers = np.array(new_centers)
        # print(new_centers)
        # print()
        # print(centers)
        if (centers == new_centers).all():
            conv = True
        else:
            centers = new_centers
    return centers, test_d


def elbow(x_test,max_k):
    '''

    :param x_test: a numpy multidimensional array that represent our dataset after the component analysis
    :param max_k: maximun number of clusters that we want to test
    :return: return the WCSS list and the plot related to them
    '''
    
    cost_list = []
    for k in range(1, max_k):
        centr,clstrs = k_means(x_test,k)
        clstrs = utilities.get_rows(clstrs, x_test)
        sum_ = 0
        for i in range(len(centr)):
            s = 0
            for j in clstrs[i]:
                #for k in range
                #print(centr[i] - j)
                #print(j,centr[i])

                for k in range(len(j)):
                    #print(j[k])
                    #print(centr[i][k])
                    #print(math.sqrt((centr[i][k] - j[k])**2))
                    #s+=math.sqrt((centr[i][k] - j[k])**2)
                    s+=(centr[i][k] - j[k])**2
                #sum_+=s
                sum_+=math.sqrt(s)
        cost_list.append(sum_)
    sns.lineplot(x=range(1,len(cost_list)+1), y=cost_list, marker='o')
    sns.set(rc={'figure.figsize':(6,7)})
    plt.xlabel('k')
    plt.ylabel('WCSS')
    plt.show()
    return cost_list

def silhoutte_method(x_test,n_clusters):
    '''

    :param x_test: a numpy multidimensional array that represent our dataset after the component analysis
    :param n_clusters: maximun number of clusters that we want to test
    :return: the average silhouette score list
    '''

    avrg_sil = []
    for cl in range(1,n_clusters):
        out = k_means(x_test,cl)
        st = np.zeros(x_test.shape[0])
        for k,v in out[1].items():
            for val in v:
                st[val] = k
        for i in range(len(st)):
            si_score_list = []
            in_cl_d = []
            out_cl_d = []
            cluster = st[i]
            #indici di tutti i punti che sono nello stesso cluster
            all_points_in_cluster = np.where(st == cluster)

            #indici di tutti i punti che non sono nello stesso cluster
            all_points_not_in_cluster = np.where(st !=cluster)
            for p1 in x_test[all_points_in_cluster[0]]:
                d_in = 0
                d_out = 0
                for p2 in x_test[all_points_in_cluster[0]]:
                    if (p1!=p2).all():
                        #print(p1,p2,np.linalg.norm(p1-p2))
                        d_in+=np.linalg.norm(p1-p2)
                in_cl_d.append(d_in)
                for p2 in x_test[all_points_not_in_cluster[0]]:
                    d_out += np.linalg.norm(p1-p2)
                out_cl_d.append(d_out)
            for s in range(len(in_cl_d)):
                si_score = (out_cl_d[s] - in_cl_d[s])/max(out_cl_d[s],in_cl_d[s])
                si_score_list.append(si_score)
        avrg_sil.append(np.mean(si_score_list))
        
    sns.lineplot(x=range(1,len(avrg_sil)+1), y=avrg_sil, marker='o')
    sns.set(rc={'figure.figsize':(6,7)})
    plt.xlabel('k')
    plt.ylabel('silhouette')
    plt.show()
    return avrg_sil