from surprise import SVD
from surprise import SVDpp
import os
import csv
from surprise import dump
from collections import defaultdict
from surprise import Dataset
from surprise import accuracy
from surprise import KNNBasic
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import get_dataset_dir
from surprise.model_selection import GridSearchCV
import io

def percentConfidence(prediction, actual):
    """
    Returns the percent that a prediction is to the actual rating done by the user
    """
    return (10 - abs(2*(actual-prediction)))*10

def read_item_names():
    """Read the u.item file from MovieLens 100-k dataset and return two
    mappings to convert raw ids into movie names and movie names into raw ids.
    """

    file_name = './movies.csv'
    rid_to_name = {}
    name_to_rid = {}
    with open(file_name, 'r') as f:
        reader = csv.reader(f, dialect="excel")
        for line in reader:
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''
    Return precision and recall at k metrics for each user.
        Precision = Proportion of recommended items that are relevant
        Recall = Proportion of relevant items that are recommended
    '''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings[:k])
        # print("N_rel: "+str(n_rel))

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        # print("N_rec_k: "+str(n_rec_k))

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])
        # print("n_rel_and_rec_k: "+str(n_rel_and_rec_k))

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = float(n_rel_and_rec_k) / float(n_rec_k) if n_rec_k != 0 else 1

        # print("Precisions: "+str(precisions[uid]))

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = float(n_rel_and_rec_k) / float(n_rel) if n_rel != 0 else 1

        # print("Recall: "+str(recalls[uid]))

    return precisions, recalls

def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est, true_r))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def getUserTop(top_n, user):
    """
    Get the top k recommendations for a given `user`

    `top_n:` a dictionary of the top k reccommendations for a given user

    `user:` internal user id(uid) used in the datasets like in ml-latest-parsed.csv

    Returns:
        a list containing the top k reccomendations for the given user
    """
    matches = [ [iid for (iid, _) in user_ratings] for uid,user_ratings in top_n.items() if uid == user]
    return matches

# Load the movielens-100k dataset (download it if needed),
reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5,5), skip_lines=1 )
data = Dataset.load_from_file('./ml-latest-parsed.csv', reader=reader)

trainset = data.build_full_trainset()

testset = trainset.build_anti_testset()
#testset = trainset.build_anti_testset()

#trainset, testset = train_test_split(data, test_size=.3)

# We'll use the famous SVD algorithm.
print("Creating Model")
sim_options = {'name':'cosine', 'user_based':True, 'min_support':2}
algo = KNNBasic(k=40, min_k=2, sim_options=sim_options)

algo = SVDpp()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset, verbose=False)

rid_to_name, name_to_rid = read_item_names()

top_n = get_top_n(predictions, n=8)

user_predictions_table = open("recommendations2.csv", "w+")
user_predictions_table.write("userId,movieId,movieName,prediction,trueValue\n")
user_predictions_readable = open("recommendations_readable2.txt", "w+")
writer = csv.writer(user_predictions_table)

# Write the predictions into a human readable format and into a .csv file for analysis
for uid, user_ratings in top_n.items():
    line1 = "__________UID: "+str(uid)+"____________\n"
    user_predictions_readable.write(line1)
    for iid, estimation, true_r in user_ratings:
        movie_name = str((rid_to_name[str(iid)]))
        # print(movie_name)
        part1 = '['+str(iid)+" "+str(movie_name)+'|'+str(estimation)
        part2 = '|'+str(true_r)+'|'+str(round(percentConfidence(estimation, true_r), 3))+"%"+']\n'
        line2 = part1 + part2
        user_predictions_readable.write(line2)
        data = [[uid, iid, movie_name, round(estimation, 3), true_r]]
        writer.writerows(data)
    user_predictions_readable.write(" \n")

user_predictions_readable.close()
user_predictions_table.close()

#print("Top 10 movies from user 130", getUserTop(top_n,'130'))
#print("Top N recommendations ", top_n)
#neighbor = algo.get_neighbors(1,1)
#print(neighbor)

# Then compute RMSE
accuracy.rmse(predictions)
accuracy.mae(predictions)
precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4.0)
print("The precision was: "+str(sum(precision for precision in precisions.values())/len(precisions)))
print("The recall was:"+str(sum(recall for recall in recalls.values())/len(precisions)))

# Dump algorithm and reload it.
#file_name = os.path.expanduser('~/.surprise_data/dump_file')
#dump.dump(file_name, algo=algo)
#loaded_predictions, loaded_algo = dump.load(file_name)

#cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

#print("10 nearest neighbors of 1: ", neighbors)

#param_grid = {'k': range(1,5), 'min_k': range(1,5)}
#gs = GridSearchCV(KNNBasic, param_grid, measures=['rmse', 'mae'], cv=3)

#gs.fit(data)

# best RMSE score
#print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
#print(gs.best_params['rmse'])

