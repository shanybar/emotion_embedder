import os
import librosa
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def run_eda(data_df):
    '''
    Runs a short EDA and plotting the distribution of the labels,
     and a visualization of the audio length
    '''

    print("Shape of dataframe:", data_df.shape)
    plt.figure(figsize=(5, 5))
    sns.countplot(x='label', data=data_df)
    plt.title('Distribution of Labels')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.histplot(data=data_df, x='duration', hue='label', kde=True)
    plt.title('Distribution of audio length by labels')
    plt.show()


def create_annotation_csv(rootdir):
    filenames = []
    labels = []

    durations = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filename = os.path.join(subdir, file)
            if os.path.isfile(filename) and filename.endswith(".wav"):
                filenames.append(filename)
                metadata_list = file.split('-')
                label = metadata_list[2]
                labels.append(label)
                y, sr = librosa.load(filename)
                duration = librosa.get_duration(y=y, sr=sr)
                durations.append(duration)

    all_data_dict = {'filename': filenames, 'label': labels , 'duration': durations}
    data_df = pd.DataFrame(all_data_dict)
    run_eda(data_df)

    filenames_train, filenames_test, labels_train,\
        labels_test = train_test_split(filenames, labels, test_size=0.2, random_state=48)

    train_data_dict = {'filename':filenames_train, 'label':labels_train}
    test_data_dict = {'filename':filenames_test, 'label':labels_test}
    train_metadata_df = pd.DataFrame(train_data_dict)
    test_metadata_df = pd.DataFrame(test_data_dict)
    train_metadata_df.to_csv(os.path.join (rootdir, 'train_annotations.csv'), index=False)
    test_metadata_df.to_csv(os.path.join (rootdir, 'val_annotations.csv'), index=False)

if __name__ == '__main__':
    create_annotation_csv(rootdir="C:\\Users\\shany\\PycharmProjects\\emotion_embedder\\resources")