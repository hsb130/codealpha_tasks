import joblib
from model_prediction import get_top_n_recommendations

def predict_recommendations(model_load_path, custom_user_id):
    # Load the trained model from the joblib file
    loaded_algo = joblib.load(model_load_path)
    # Prediction using the loaded model
    recommendations = get_top_n_recommendations(loaded_algo, loaded_algo.trainset, custom_user_id)
    return recommendations

if __name__ == "__main__":
    # Define model load path
    model_load_path = 'your_model_save_path_here.joblib'

    # Input custom user ID
    custom_user_id = input("Enter the custom user ID: ")

    # Predict recommendations using the loaded model
    recommendations = predict_recommendations(model_load_path, custom_user_id)
    print("Top Recommendations using loaded model:")
    print(recommendations)
