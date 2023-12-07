import os
import glob
import pandas as pd


def create_annotation_csv(rootdir):
    filenames = []
    labels = []
    statements = []
    duration = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filename = os.path.join(subdir, file)
            if os.path.isfile(filename) and filename.endswith(".wav"):
                filenames.append(filename)
                metadata_list = file.split('-')
                label = metadata_list[2]
                labels.append(label)
                statement = metadata_list[4]
                statements.append(statement)
    data_dict = {'filename':filenames, 'label':labels, 'statement':statements}
    metadata_df = pd.DataFrame(data_dict)
    metadata_df.to_csv(rootdir, index=False)



create_annotation_csv(rootdir="C:\\Users\\shany\\PycharmProjects\\emotion_embedder\\resources")