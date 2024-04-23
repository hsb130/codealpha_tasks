import pandas as pd

def read_csv(file_path):
    df = pd.read_csv(file_path, on_bad_lines='skip')
    return df
