from datetime import datetime
from datetime import timedelta
import os

from joblib import load
import pandas as pd

from Audioset.AudiosetAnalysis_aug import AudiosetAnalysis
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

audio_dir = "../data/Indonesia2018sliced"
vggish_files = "Audioset"
output_folder_1min = "vggish_features"

an = AudiosetAnalysis()
an.setup()

f = "../data/Indonesia2018sliced/BaF5.1732H.671907872.180831.NT1822.wav"

# Feature extract
# Extract timestamp from filename
t2 = f.split(".")[1][0:4]
t1 = f.split(".")[3]
t = t1 + t2 + "00"
recording_start_time = pd.to_datetime(t, format="%y%m%d%H%M%S")
slice_time = recording_start_time - timedelta(milliseconds=960)
mean_slice_time = recording_start_time - timedelta(minutes=1)

results_df_1sec = pd.DataFrame()
results_df_1min = pd.DataFrame()

# Extract timestamp from filename
t2 = f.split(".")[1][0:4]
t1 = f.split(".")[3]
t = t1 + t2 + "00"
recording_start_time = pd.to_datetime(t, format="%y%m%d%H%M%S")
slice_time = recording_start_time - timedelta(milliseconds=960)
mean_slice_time = recording_start_time - timedelta(minutes=1)

# Calculate feature values
# results = an.analyse_audio(path)
aug_count = 0
for results in an.analyse_audio(f):
    # Uncomment for 0.96s results:
    # r1sec = results['raw_audioset_feats_960ms']
    # for count, r1sec in enumerate(r1sec):
    #     slice_time = slice_time + timedelta(milliseconds=960)
    #     string_time = slice_time.strftime('%H.%M.%S.%f')[:-4]
    #     result_name = f[:-4] + 'T' + string_time + '.wav'
    #     #result_name = f[:-4]+'T'+str(count+1)+'.wav' #use this line if not using ST timestamped files
    #     results_df_1sec[result_name] = pd.Series(results['raw_audioset_feats_960ms'][count])

    # Save 1 min results:
    r1min = results["raw_audioset_feats_59520ms"]
    for count, r1min in enumerate(r1min):
        # store the timestamp
        mean_slice_time = mean_slice_time + timedelta(minutes=1)
        string_time = mean_slice_time.strftime("%H.%M.%S.%f")[:-4]
        result_name = f[:-4] + "T" + string_time + "A" + str(aug_count) + ".wav"
        # result_name = f[:-4]+'T'+str(count+1)+'.wav' #use this line if not using ST timestamped files
        results_df_1min[result_name] = pd.Series(results["raw_audioset_feats_59520ms"][count])
    aug_count += 1

# Save a timestamped csv with 1 min results
# now = datetime.now()
# time_now = now.strftime("%H.%M.%S")
results_df_1min.to_csv(output_folder_1min + "inference.csv")
print(f"Feature extraction is finished")

# Inference
# Features of audio files, obtained by VGGish feature extractor.
path = "vggish_features/inference.csv"
num_classes = 2
labels = ["Healthy", "Degraded"]

data = pd.read_csv(path)
temp_df = data.reset_index() # put index in order
temp_df = temp_df.iloc[: , 2:]  # Remove unnecessary index
temp_df = temp_df.T  # Transpose to match indices format
temp_df = temp_df.reset_index()  # Re-add the index
df = temp_df.rename(columns={"index": "minute"})

model = load("models/random_forest_model_7_0.joblib")


def get_feats(file):
    # Create new dataframes for the train, val and test data for each class.
    file_df = df[df["minute"].isin(file)]

    # Add a column with the class
    file_df.insert(1, "class", "Healthy")
    feats = file_df.iloc[:, 2:].to_numpy()
    labels = file_df.iloc[:, 1].to_numpy()

    return feats, labels

feats, labels = get_feats(f)
acc = model.score(feats, labels)
print(acc)
