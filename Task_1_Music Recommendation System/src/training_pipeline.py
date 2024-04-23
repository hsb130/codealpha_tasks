from data_collection import read_csv
from data_transformation import clean_data
from model_training import build_and_train_model
import joblib

def main(file_path, model_save_path):
    # Data Collection
    df = read_csv(file_path)
    
    # Data Transformation
    track_counts = clean_data(df)
    
    # Model Building and Training
    algo, rmse = build_and_train_model(track_counts)
    
    # Save the trained model to a file
    joblib.dump(algo, model_save_path)
    print("Model saved to:", model_save_path)

if __name__ == "__main__":
    # Define file path and model save path
    file_path = "E:\Hasib's Github\ML Intern@CodeAlpha\Task_1_Music Recommendation System\Datasets\dataset_2.csv"
    model_save_path = 'music_recommender_model.joblib'

    # Execute the main function
    main(file_path, model_save_path)
