from dataset import Dataset
from reader import Reader

def create_testings(path, rating_scale):
    reader = Reader(line_format='user item rating', sep=' ', rating_scale=rating_scale)
    data = Dataset.load_from_file(path, reader=reader)
    testset = data.build_full_trainset().build_testset()
    return testset

if __name__ == '__main__':
    vpath = "C:/gjw2/Rec/Datasets/filmtrust/cvtest1.txt"
    rating_scale = (0.5, 4)
    testset = create_testings(vpath, rating_scale)
    print(len(testset))
