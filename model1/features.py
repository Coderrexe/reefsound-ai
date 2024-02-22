from datetime import datetime
from datetime import timedelta
import os

import pandas as pd

from Audioset.AudiosetAnalysis_aug import AudiosetAnalysis

audio_dir = "../../data/HD_audio"
# Path to folder containing VGGish files
vggish_files = "Audioset"
# Output folder for results
output_folder_1min = "../Results"

an = AudiosetAnalysis()
an.setup()

# Select audio files
all_fs = os.listdir(audio_dir)
audio_fs = [f for f in all_fs if ".wav" in f.lower() or ".mp3" in f.lower()]

results_df_1sec = pd.DataFrame()
results_df_1min = pd.DataFrame() 

# Feature extraction loop
for f in audio_fs:
    """This loop takes the current filename, rips the timestamp, appends the corresponding length of time being analysed
    additively to a new name, calculates VGGish features from each 0.96s chunk, averages these for each 1min file and saves 
    the results to a csv."""
    path = os.path.join(audio_dir, f)
    print(f)
    
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
    for results in an.analyse_audio(path):
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
            #store the timestamp
            mean_slice_time = mean_slice_time + timedelta(minutes=1)
            string_time = mean_slice_time.strftime("%H.%M.%S.%f")[:-4]
            result_name = f[:-4] + "T" + string_time + "A" + str(aug_count) + ".wav"
            #result_name = f[:-4]+'T'+str(count+1)+'.wav' #use this line if not using ST timestamped files
            results_df_1min[result_name] = pd.Series(results["raw_audioset_feats_59520ms"][count])
        aug_count += 1

# Save a timestamped csv with 1 min results
now = datetime.now()
time_now = now.strftime("%H.%M.%S")
results_df_1min.to_csv(output_folder_1min + "/pretrained_CNN_features_" + time_now + ".csv")
print(f"Feature extraction is finished. Results are saved at {output_folder_1min + '/pretrained_CNN_features_' + time_now + '.csv'}")
