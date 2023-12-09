import os
import csv
from sklearn.utils import shuffle
import pandas as pd

# Load data
current_directory = os.environ['HOME']
data_path = os.path.join(current_directory, "radimagenet/Anders/Eksperiment2/Data/poc_breast_dataset/Labels.csv")
data_df = pd.read_csv(data_path,header=0,dtype=str, sep=',')

pocBreastData = shuffle(data_df)
train_pocBreastData = pocBreastData [0:2880] #80% af 3600 = 2880
val_pocBreastData = pocBreastData [2880:3060] #5% af 3600 = 180
test_pocBreastData = pocBreastData [3060:] #15% af 3600 = 540
#ITU: datasplit p ̊a 80% træning, 5% validering og 15% test.


#write to csv
#folder_path = "radimagenet/Anders/Eksperiment2/Data/poc_breast_dataset/Images/"
folder_path = ""

#train
relative_path = 'radimagenet/Anders/Eksperiment2/Data/Datasplits/train_pocData.csv'
subset_path = os.path.join(current_directory, relative_path)

with open(subset_path , 'w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(['filepath', 'label'])
    for i in range(len(train_pocBreastData)):
        #print(train_pocBreastData.iloc[i,0])
        #print(train_pocBreastData.iloc[i,1])
        writer.writerow([train_pocBreastData.iloc[i,0],train_pocBreastData.iloc[i,1]])


#Validation
relative_path = 'radimagenet/Anders/Eksperiment2/Data/Datasplits/val_pocData.csv'
subset_path = os.path.join(current_directory, relative_path)
with open(subset_path , 'w', newline='') as file:
    writer = csv.writer(file)
     
    writer.writerow(['filepath', 'label'])
    for i in range(len(val_pocBreastData)):
        #print(train_pocBreastData.iloc[i,0])
        #print(train_pocBreastData.iloc[i,1])
        writer.writerow([val_pocBreastData.iloc[i,0],val_pocBreastData.iloc[i,1]])


#Test
relative_path = 'radimagenet/Anders/Eksperiment2/Data/Datasplits/test_pocData.csv'
subset_path = os.path.join(current_directory, relative_path)
with open(subset_path , 'w', newline='') as file:
    writer = csv.writer(file)
     
    writer.writerow(['filepath', 'label'])
    for i in range(len(test_pocBreastData)):
        #print(train_pocBreastData.iloc[i,0])
        #print(train_pocBreastData.iloc[i,1])
        writer.writerow([test_pocBreastData.iloc[i,0],test_pocBreastData.iloc[i,1]])

print('')
print('file names and labels are saved')
print('')