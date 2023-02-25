import pandas as pd
from surprise.model_selection import train_test_split, cross_validate
from surprise import Dataset, Reader, BaselineOnly

def readHeader(file_path):
    tsv_file = open(file_path)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    list = []
    for two in read_tsv:
        list.append(column)
    return list

def createPredictionDataFrame(testDf, prediction_testset):
    rowID = testDf.RowID.tolist()
    uid = []
    est = []
    i = 0
    while i < len(prediction_testset):
        uid.append(prediction_testset[i].uid)
        est.append(prediction_testset[i].est)
        i += 1

    df = pd.DataFrame(list(zip(rowID,uid, est)),columns =['RowID','ReviewerID', 'Label'])
    return df

# File path
headerFile = "../S3862092/header.tsv" 
trainFile = "../S3862092/train.tsv"
testFile = "../S3862092/test.tsv"
valFile = "../S3862092/val.tsv"
featuresFile = "../S3862092/features.tsv"
headerFeatures = "../S3862092/header-features.tsv"

# Convert header series to a list
headerDf = pd.read_csv(headerFile, sep = "\t", header = 0)
headerFeaturesDf = pd.read_csv(headerFeatures, sep = '\t', header = 0)
header = list(headerDf.columns)
headerFeatures = list(headerFeaturesDf.columns)

#Dataframes
trainDf = pd.read_csv(trainFile, sep = "\t", names = header)
testDf = pd.read_csv(testFile, sep = "\t", names = header[:len(header)])
valDf = pd.read_csv(valFile, sep = "\t", names = header)
featuresDf = pd.read_csv(featuresFile, sep = '\t', names = headerFeatures)

# Create new DataFrame containing some features from features.tsv
features_selected = featuresDf.drop(["DayofWeek", "Month", "DayofMonth", "Year"], axis = 1)
merged = trainDf.merge(features_selected, how='inner', left_on='RowID', right_on='RowID')

# It appears that beers with ABV of more than 20, recieve very little ratings, 
# so any drinks above ABV of 20 will be removed from the data
merged_new = pd.DataFrame(columns = merged.columns)
for col in merged.columns:
    merged_new[col] = merged[col]
indexNames = merged_new[ merged_new['ABV'] >= 20 ].index
merged_new.drop(indexNames , inplace=True)

# Create model
cleant = merged_new[['ReviewerID','BeerID','Label']]
cleanv = valDf.drop(['RowID','BeerName','BeerType'],axis=1)
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(cleant[['ReviewerID','BeerID',
                                    'Label']],reader)

datav = Dataset.load_from_df(cleanv[['ReviewerID','BeerID',
                                     'Label']],reader)
trainset = data.build_full_trainset()
NA,valset = train_test_split(datav, test_size=1.0)
bsl_options = {'method': 'als', 'n_epochs': 15}
model = BaselineOnly(bsl_options = bsl_options)
model.fit(trainset)

# Predict on test set
testData = Dataset.load_from_df(testDf[['ReviewerID','BeerID','Label']],reader)
NA,testset = train_test_split(testData, test_size=1.0, shuffle = False)
prediction = model.test(testset)

# Prepare output file
resultDf = createPredictionDataFrame(testDf, prediction)
resultDf = resultDf.drop("ReviewerID", axis = 1)
resultDf.to_csv("../S3862092/A3-2.tsv", sep = "\t", header = False ,index = False)