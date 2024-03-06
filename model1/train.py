from joblib import dump, load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Features of audio files, obtained by VGGish feature extractor.
path = "vggish_features/pretrained_CNN_features_21.08.33.csv"
num_classes = 2
labels = ["Healthy", "Degraded"]

data = pd.read_csv(path)
temp_df = data.reset_index() # put index in order
temp_df = temp_df.iloc[: , 2:]  # Remove unnecessary index
temp_df = temp_df.T  # Transpose to match indices format
temp_df = temp_df.reset_index()  # Re-add the index
df = temp_df.rename(columns={"index": "minute"})


def put_files_in_splits(train_deployments, val_deployments, test_deployments):
    """This function uses the IDs of the train, val and test sets generated above to select the actual recordings.
    This generates train_files, val_files, test_files."""
    train_files = []
    val_files = []
    test_files = []

    for index, row in df.iterrows():
        filename = (df["minute"][index]) 
        namePt1 = filename.split(".")[0]
        namePt2 = filename.split(".")[1]
        ID = namePt1 + "." + namePt2

        if ID in train_deployments:
            train_files.append(filename)
        if ID in val_deployments:
            val_files.append(filename)
        if ID in test_deployments:
            test_files.append(filename)

    # print("Number and list of validation files:")
    # print(len(val_files))
    # print(val_files)
    # print("Number and list of test files:")
    # print(len(test_files))
    # print(test_files)
    # print("Number and list of training files:")
    # print(len(train_files))
    # print(train_files)
            
    return train_files, val_files, test_files


def get_class(filename):
    t = filename.split(".")[1][4:5]
    return t


def split_by_class(train_files, val_files, test_files):
    degraded_train_files = []
    healthy_train_files = []

    degraded_val_files = []
    healthy_val_files = []

    degraded_test_files = []
    healthy_test_files = []

    for f in train_files:
        if get_class(f) == "D":
            degraded_train_files.append(f)
        
        if get_class(f) == "H":
            healthy_train_files.append(f)

    for f in val_files:
        if get_class(f) == "D":
            degraded_val_files.append(f)
        
        if get_class(f) == "H":
            healthy_val_files.append(f)

    for f in test_files:
        if get_class(f) == "D":
            degraded_test_files.append(f)
        
        if get_class(f) == "H":
            healthy_test_files.append(f)

    return degraded_train_files, healthy_train_files, degraded_val_files, healthy_val_files, degraded_test_files, healthy_test_files


def remake_dfs_for_splits(degraded_train_files, healthy_train_files, degraded_val_files, healthy_val_files, degraded_test_files, healthy_test_files):
    # Create new dataframes for the train, val and test data for each class.
    healthy_train_df = df[df["minute"].isin(healthy_train_files)]
    degraded_train_df = df[df["minute"].isin(degraded_train_files)]

    healthy_val_df = df[df["minute"].isin(healthy_val_files)]
    degraded_val_df = df[df["minute"].isin(degraded_val_files)]

    healthy_test_df = df[df["minute"].isin(healthy_test_files)]
    degraded_test_df = df[df["minute"].isin(degraded_test_files)]

    # Add a column with the class
    healthy_train_df.insert(1, "class", "Healthy")
    degraded_train_df.insert(1, "class", "Degraded")
    healthy_val_df.insert(1, "class", "Healthy")
    degraded_val_df.insert(1, "class", "Degraded")
    healthy_test_df.insert(1, "class", "Healthy")
    degraded_test_df.insert(1, "class", "Degraded")

    # Now combine dataframes for each class into dataframes for tje train, val and test (so these remain separate)
    train_unshuffled = pd.concat([healthy_train_df, degraded_train_df])
    val_df = pd.concat([healthy_val_df, degraded_val_df])
    test_df = pd.concat([healthy_test_df, degraded_test_df])

    # Shuffle train (this is not done to to test or val so that predictions can be more easily studied)
    train_df = train_unshuffled.sample(frac = 1) 

    # Now create arrays of the features and labels that can be input to RFs to train or inference
    train_feats = train_df.iloc[:, 2:].to_numpy()
    train_labels = train_df.iloc[:, 1].to_numpy()

    validation_feats = val_df.iloc[:, 2:].to_numpy()
    validation_labels = val_df.iloc[:, 1].to_numpy()

    test_feats = test_df.iloc[:, 2:].to_numpy()
    test_labels = test_df.iloc[:, 1].to_numpy()

    return train_feats, train_labels, validation_feats, validation_labels, test_feats, test_labels


# This function takes the parts of a filename that make it unique
# This uses Tims naming convention, specific to the 2018 Indonesia data
def get_identifier(filename):
    # Find part of the name that corresponds to the deployment
    t0 = filename.split(".")[0]
    t1 = filename.split(".")[1][0:5]
    t = t0 + "." + t1
    return t


# Function to get unique values within an array
def unique(list1):
    x = np.array(list1)
    return np.unique(x)


def get_class(filename):
    # Find part of the name that corresponds to the deployment
    # Adapted the get_identifier function above to only get class (e.g healthy)
    deployment_ID = filename.split(".")[1][4:5]
    return deployment_ID


def train_test_val_split(df):
    IDs = []
    for index, row in df.iterrows():
        filename = (df["minute"][index])
        IDs.append(get_identifier(filename))
    all_IDs = unique(IDs)

    # Use the above function to get a list of unique deployment IDs (approx 30 for healthy, and again for degraded)
    unshuffled_unique_deployments = unique(all_IDs) #so for the real data this will give a big long order list

    # Ensure the same random shuffle is made each time for a CV, so the order is conserved across the 3 methods
    np.random.seed(42)

    # Shuffle this list
    shuffled_unique_deployments = np.random.permutation(unshuffled_unique_deployments)

    # Create df of all deployments and their class
    d = {"Deployment": shuffled_unique_deployments}
    d1 = pd.DataFrame(data=d)

    df_withclasses = d1

    # Add a column to the DF that contains class
    new_list = []
    for i in range(len(df_withclasses)):
        new_list.append(get_class(df_withclasses["Deployment"][i]))

    df_withclasses.insert(1, "Class", new_list)

    # Use the new df to split the deployment ID"s into train/val/test sets
    # Pick 15% (8 deployments) of data as test data
    validation, the_rest = sklearn.model_selection.train_test_split(df_withclasses, test_size=0.85, stratify=df_withclasses["Class"])
    # Pick 0.15*0.85% (8 deployments) of the_rest to be the val data
    train, test = sklearn.model_selection.train_test_split(the_rest, test_size=0.15, stratify=the_rest["Class"])

    # Convert these to numpy arrays
    train_deployments = np.array(train["Deployment"])
    val_deployments = np.array(validation["Deployment"])
    test_deployments = np.array(test["Deployment"]) 
    np.random.seed() #now lift the seed so that randomisation can be used again in the rest of the script
    
    return train_deployments, val_deployments, test_deployments


"""Run the model for 100 cross validation splits and for each:
Train on the train data for 50 repeats
For each repeat it inference on the validation data
If the accuracy on the val data is higher than the accuracy of a prev, repeat it will then inference on the test data and save this. 
If a future repeat on the val data is more accurate it will overwrite the previous acc score for the test data.
It will then output the test data accuraciesfor each of the 100 repeats"""

all_saved_test_accs = []
ConfusionMatrix = np.zeros((num_classes, num_classes), dtype=float)

for i in range(100):
    repeat = i+1
    thisCV_saved_test_accs = []
    print("Training cross val: " + str(repeat))
    train_deployments, val_deployments, test_deployments = train_test_val_split(df)
    train_files, val_files, test_files = put_files_in_splits(train_deployments, val_deployments, test_deployments)
    degraded_train_files, healthy_train_files, degraded_val_files, healthy_val_files, degraded_test_files, healthy_test_files = split_by_class(train_files, val_files, test_files)
    train_feats, train_labels, val_feats, val_labels, test_feats, test_labels  = remake_dfs_for_splits(degraded_train_files, healthy_train_files,
                                                                                                    degraded_val_files, healthy_val_files, degraded_test_files, healthy_test_files)
    print(f"Training on {len(train_feats)} samples, validating on {len(val_feats)} samples, testing on {len(test_feats)} samples")

    # print(test_deployments)
    #accuracy_scores = []
    val_accuracy_score = 0
    for k in range(50):  # Picked 50 as 50 epochs was used to train the VGGish
        model = RandomForestClassifier(n_jobs = -1,random_state=k)
        # This trains 50 RFs and chooses the best
        #print("Inferencing on validation data, repeat: " + str(k))
        model.fit(train_feats, train_labels)
        # print(f"train accuracy: {model.score(train_feats, train_labels)}")
        new_val_acc = model.score(val_feats, val_labels)
        print(new_val_acc)
        if new_val_acc > val_accuracy_score:
            val_accuracy_score = new_val_acc
            test_acc = model.score(test_feats, test_labels)
            test_predictions = model.predict(test_feats)
            # Get confusion matrix values
            best_ConfusionMatrix = confusion_matrix(test_labels, test_predictions, labels=labels)
            dump(model, f"models/random_forest_model_{repeat}_{k}.joblib")  # Save the model
        #val_accuracy_scores.append(model.score(val_feats, val_labels))
    thisCV_saved_test_accs.append(test_acc)
    all_saved_test_accs.append(test_acc)
    ConfusionMatrix = np.add(ConfusionMatrix, best_ConfusionMatrix)
    print("Accuracies for cross validation split number: "+ str(repeat))
    print(thisCV_saved_test_accs)
    thisCV_saved_test_accs = []

# Accuracy and standard deviation across all CVs
print("Completed RFs:")
print(len(all_saved_test_accs))


def Average(lst):
    return sum(lst) / len(lst)


mean_accuracy = Average(all_saved_test_accs)
stdev = np.std(all_saved_test_accs)

print("saved_test_accs: ")
print(all_saved_test_accs)
result = "Mean accuracy with standard deviation = {} (ﾱ{})".format(str(mean_accuracy), str(stdev))
print(result)
print()
print(repr(ConfusionMatrix))

figure = plt.figure(figsize=(10, 10))

index = range(len(all_saved_test_accs))
plt.scatter(index, all_saved_test_accs, label = "test acc", color = "r")
plt.xlabel("epoches")
plt.ylabel("accuracy")
plt.ylim(0, 1.1)

plt.savefig("model_results/vggish_randomforest_accuracy.png")
# plt.show()

array = np.around(ConfusionMatrix.astype("float") / ConfusionMatrix.sum(axis=1)[:, np.newaxis], 2)

df_cm = pd.DataFrame(array, index=["Healthy", "Degraded"], columns=["Healthy", "Degraded"])
plt.figure(figsize=(10,10))
cmap = sns.cm.rocket_r
ax = sns.heatmap(df_cm, annot=True, annot_kws={"fontsize": 25}, fmt="g", cmap=cmap)
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 25)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 25)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

plt.savefig("model_results/vggish_randomforest_confusion_matrix.png")
