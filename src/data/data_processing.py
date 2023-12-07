import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_annotation_csv(rootdir):
    filenames = []
    labels = []

    duration = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filename = os.path.join(subdir, file)
            if os.path.isfile(filename) and filename.endswith(".wav"):
                filenames.append(filename)
                metadata_list = file.split('-')
                label = metadata_list[2]
                labels.append(label)

    filenames_train, filenames_test, labels_train,\
        labels_test = train_test_split(filenames, labels, test_size=0.2, random_state=48)
    filenames_val, filenames_test, labels_val,\
        labels_test = train_test_split(filenames_test, labels_test, test_size=0.5, random_state=48)

    train_data_dict = {'filename':filenames_train, 'label':labels_train}
    val_data_dict = {'filename':filenames_val, 'label':labels_val}
    test_data_dict = {'filename':filenames_test, 'label':labels_test}
    train_metadata_df = pd.DataFrame(train_data_dict)
    val_metadata_df = pd.DataFrame(val_data_dict)
    test_metadata_df = pd.DataFrame(test_data_dict)
    train_metadata_df.to_csv(os.path.join (rootdir, 'train_annotations.csv'), index=False)
    val_metadata_df.to_csv(os.path.join (rootdir, 'validation_annotations.csv'), index=False)
    test_metadata_df.to_csv(os.path.join (rootdir, 'test_annotations.csv'), index=False)

create_annotation_csv(rootdir="C:\\Users\\shany\\PycharmProjects\\emotion_embedder\\resources")