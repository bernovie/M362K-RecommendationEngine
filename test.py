from surprise import SVD
from surprise import SVDpp
import os
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
    return (10 - abs(2*(actual-prediction)))*10

def read_item_names():
    """Read the u.item file from MovieLens 100-k dataset and return two
    mappings to convert raw ids into movie names and movie names into raw ids.
    """

    file_name = './ml-latest-small/movies.csv'
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split(',')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid

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
    matches = [ [iid for (iid, _) in user_ratings] for uid,user_ratings in top_n.items() if uid == user]
    return matches

# Load the movielens-100k dataset (download it if needed),
reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5,5), skip_lines=1 )
data = Dataset.load_from_file('./ml-latest-small/ratings.csv', reader=reader)

trainset = data.build_full_trainset()

testset = trainset.build_anti_testset()

#trainset, testset = train_test_split(data, test_size=.3)

# We'll use the famous SVD algorithm.
sim_options = {'name':'cosine', 'user_based':True, 'min_support':2}
#algo = KNNBasic(k=40, min_k=2, sim_options=sim_options)
algo = SVDpp()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset, verbose=False)

rid_to_name, name_to_rid = read_item_names()

top_n = get_top_n(predictions, n=8)

for uid, user_ratings in top_n.items():
    print"__________UID:", uid, "____________"
    for iid, estimation, true_r in user_ratings:
        print '[',iid, rid_to_name[str(iid)],'|',estimation,'|',true_r,'|',str(round(percentConfidence(estimation, true_r), 3))+"%",']'
    print("")

#print("Top 10 movies from user 130", getUserTop(top_n,'130'))
#print("Top N recommendations ", top_n)
#neighbor = algo.get_neighbors(1,1)
#print(neighbor)

# Then compute RMSE
accuracy.rmse(predictions)
accuracy.mae(predictions)

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

