from dataset import Dataset
from reader import Reader

def create_ratings(path, rating_scale):
    reader = Reader(line_format='user item rating', sep=' ', rating_scale=rating_scale)
    data = Dataset.load_from_file(path, reader=reader)
    trainset = data.build_full_trainset()
    return trainset

if __name__ == '__main__':
    rpath = "C:/gjw2/Rec/community/Datasets/flixster/ratings.txt"
    rating_scale = (0.5, 5)
    trainset = create_ratings(rpath, rating_scale)
    print(trainset.n_users)
