# ADM_HW4_Group5

 - __main.ipynb__: the notebook that contains the homework
 - __silhouette.ipynb__: this notebook contains the result of our silhoutte analysis. We used another notebook because the running time of the analysis was very long, in this way we could run the main notebook while the analysis was finishing
- __clustering.py__ : This python library which contains the functions that are needed to achieve clustering. It contains our kmeans implementation and the two methods (silhoutte and elbow) that we used to find the best number of clusters

- __custom_minhash.py__ : This python library contains functions to compute the hash signature matrix starting from the dataset 

- __utilities.py__ : some utilities function that helped us with the work. 

- __utility_function.py__ : we moved the function, that were in __AudioSignals.ipynb__ ,into this python file. This library contains __convert_mp3_to_wav__ __plot_spectrogram_and_picks__ __load_audio_picks__

- __function_for_pivot.py__: this is a library that contains function that we used to create pivot table.

- __dataset__ : https://www.kaggle.com/dhrumil140396/mp3s32k. This dataset that contains .mp3 files divided by artist and album

- __/json__ : directory that contains the .json files that represent the hash signature matrix for the dataset songs (sig_matrix.json) and for the query songs (queries_sig.json). It also contains a json that map song id to the tuple (artist, song name) 

