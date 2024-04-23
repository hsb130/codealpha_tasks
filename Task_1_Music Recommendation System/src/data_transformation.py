import re

def clean_data(df):
    # Rename columns
    column_rename_mapping = {
        ' "artistname"':'artist',
        ' "trackname"': 'track',
        ' "playlistname"': 'playlist'
    }
    df = df.rename(columns=column_rename_mapping)

    # Clean text values
    columns_to_clean = ['artist', 'track', 'playlist']
    for col_name in columns_to_clean:
        df[col_name] = df[col_name].astype(str)
        df[col_name] = df[col_name].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    # Group by 'user_id' and 'track' and count occurrences
    track_counts = df.groupby(['user_id', 'track']).size().reset_index(name='track_count')

    return track_counts
