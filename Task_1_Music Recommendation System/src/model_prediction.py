def get_top_n_recommendations(algo, trainset, custom_user_id, n=20):
    items_to_rate = [item for item in trainset.all_items() if item not in trainset.ur[custom_user_id]]
    item_ratings = [(item, algo.predict(custom_user_id, item).est) for item in items_to_rate]
    item_ratings.sort(key=lambda x: x[1], reverse=True)
    top_n_recommendations = item_ratings[:n]
    top_n_recommendations_with_names = [(trainset.to_raw_iid(item_id), rating) for item_id, rating in top_n_recommendations]
    return top_n_recommendations_with_names
