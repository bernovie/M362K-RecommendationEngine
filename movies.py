import io  # needed because of weird encoding of u.item file

from surprise import KNNBaseline
from surprise import Dataset
from surprise import get_dataset_dir

def read_user_names():
    """Read the u.item file from MovieLens 100-k dataset and return two
    mappings to convert raw ids into movie names and movie names into raw ids.
    """

    file_name = get_dataset_dir() + '/ml-100k/ml-100k/u.data'
    user_id = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('\t')
            user_id[line[0]] = line[0]

    return user_id

# First, train the algortihm to compute the similarities between items
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
sim_options = {'name': 'cosine', 'user_based': True, 'min_support':10 }
algo = KNNBaseline(sim_options=sim_options)
algo.fit(trainset)

user_neighbors = algo.get_neighbors(130, k=10)

user_neighbors = (algo.trainset.to_raw_iid(inner_id)
                    for inner_id in user_neighbors) 

print('The 10 nearest neighbors of User 130 are:')
for user in user_neighbors:
    print(user)