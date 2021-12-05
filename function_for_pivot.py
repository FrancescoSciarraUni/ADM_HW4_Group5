import pandas as pd
import json
import numpy as np
import seaborn as sns
import sklearn
import math
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#This function take as input the database that we want to use for analisis, like track_duration
# and add columns named cluster0, cluster1 ..  all filled with NaN values

def adding_clusters_columns(database, number_of_clusters):
    for j in range(number_of_clusters):
        i=str(j)
        database.loc[:, "cluster"+i] = np.nan
    database.loc[:, "groupization"] = ""
    return database

#this function take as input the database with the cluster columns and the result of k_means 
#and return the dataframe with the update index of the cluster 
def update_cluster_columns(dataframe, cluster_dictionary, number_of_features):
    for key, value in cluster_dictionary.items():
        for i in value:
            row=int(i)
            a=int(key)
            columns=int(number_of_features+a)
            dataframe.iat[row,columns]=1
    return dataframe

#this function create list in which the values are equally divided from min to max

def get_range(database , feature):
    column=database[feature]
    max_value=column.max()
    min_value=column.min()
    range_values=np.arange(min_value, max_value ,((max_value-min_value)/4))
    range_values=np.append(range_values, max_value)    
    return range_values   


#This funciont fills the columns groupization using a list of integer values
#classifing the values in one of 4 groups
def filling_groupization(dataframe, column_to_group ,group_interval):
    col_name="groupization"
    index1=int(dataframe.columns.get_loc(col_name))
    index2=int(dataframe.columns.get_loc(column_to_group))
    for i in range(len(dataframe)):
        a= (dataframe.iloc[i,index2])
        
        if a <= group_interval[1]:
            dataframe.iat[i,index1]=str("values between "+str(round(group_interval[0],2))+" and "+str(round(group_interval[1],2))) 
        elif  a <= group_interval[2]:
            dataframe.iat[i,index1]=str("values between "+str(round(group_interval[1],2))+" and "+str(round(group_interval[2],2))) 
        elif  a <= group_interval[3]:
            dataframe.iat[i,index1]=str("values between "+str(round(group_interval[2],2))+" and "+str(round(group_interval[3],2))) 
        else:
            dataframe.iat[i,index1]=str("values between "+str(round(group_interval[3],2))+" and "+str(round(group_interval[4],2))) 
    return dataframe



#to not have to write manually all the clusters
def name_of_cluster(number_of_clusters):
    number_of_clusters=12
    list_of_cluster=[]
    for j in range(number_of_clusters):
            i=str(j)
            list_of_cluster.append("cluster"+i)
    return list_of_cluster



def table_percentage(table, list1):
    for i in list1:
        d = table[i]/table[i].sum()
        table[i] = (d)*100
        table[i] = table[i].round(3).astype(str)
        table[i] = table[i].map(lambda x : x + '%')
    return table



def genres_to_int(s):
    to_remove = ['[',']']
    l = s.split(',')
    genre = l[0]    
    for i in to_remove:
        genre = genre.replace(i,'')
    if len(genre) > 0:
        return int(genre)
    else:
        return None
    
def filling_groupization_for_track_genres(dataframe, column_to_group ,group_interval):
    col_name="groupization"
    index1=int(dataframe.columns.get_loc(col_name))
    index2=int(dataframe.columns.get_loc(column_to_group))
    k=max(group_interval)
    group_interval.sort()
    for i in range(len(dataframe)):
        a= (dataframe.iloc[i,index2])
        for j in range(len(group_interval)):
            if int(a) == int(21):
                dataframe.iat[i,index1]=str("Hip-Hop")
            elif int(a) == int(10):
                dataframe.iat[i,index1]=str("Pop")
            elif int(a) == int(17):
                dataframe.iat[i,index1]=str("Folk")
            elif int(a) == int(4):
                dataframe.iat[i,index1]=str("Jazz")
            elif int(a) == int(15):
                dataframe.iat[i,index1]=str("Electronic")
            elif int(a) == int(2):
                dataframe.iat[i,index1]=str("International")
            elif int(a) == int(3):
                dataframe.iat[i,index1]=str("Blues")
            elif int(a) == int(33):
                dataframe.iat[i,index1]=str("Folk")
            elif int(a) == int(5):
                dataframe.iat[i,index1]=str("Classical")
            elif int(a) == int(8):
                dataframe.iat[i,index1]=str("Old-Time/ Historic")
            elif int(a) == int(17):
                dataframe.iat[i,index1]=str("Folk")
            elif int(a) == int(286):
                dataframe.iat[i,index1]=str("Electronic")
            elif int(a) == int(18):
                dataframe.iat[i,index1]=str("Instrumental")
            elif int(a) == int(538):
                dataframe.iat[i,index1]=str("Instrumental")
            elif int(a) == int(240):
                dataframe.iat[i,index1]=str("Electronic")
            elif int(a) == int(514):
                dataframe.iat[i,index1]=str("Experimental")
            elif int(a) == int(360):
                dataframe.iat[i,index1]=str("Experimental")
            else:
                dataframe.iat[i,index1]=str("Rock")
            
            
    return dataframe    
    
 