from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

def build_and_train_model(track_counts):
    reader = Reader(rating_scale=(1, 100))
    data = Dataset.load_from_df(track_counts[['user_id', 'track', 'track_count']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    algo = SVD()
    algo.fit(trainset)

    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)

    return algo, rmse
