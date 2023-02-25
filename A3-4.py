import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

def text_clean_1(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def text_clean_2(text):
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

def readHeader(file_path):
    tsv_file = open(file_path)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    list = []
    for two in read_tsv:
        list.append(column)
    return list

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
testDf = pd.read_csv(testFile, sep = "\t", names = header)
valDf = pd.read_csv(valFile, sep = "\t", names = header)
featuresDf = pd.read_csv(featuresFile, sep = '\t', names = headerFeatures)

# Merge some features from features.tsv into train.tsv
features_selected = featuresDf.drop(["DayofWeek", "Month", "DayofMonth", "Year"], axis = 1)
merged = trainDf.merge(features_selected, how='inner', left_on='RowID', right_on='RowID')

merged_new = pd.DataFrame(columns = merged.columns)
for col in merged.columns:
    merged_new[col] = merged[col]
indexNames = merged_new[ merged_new['ABV'] >= 20 ].index
merged_new.drop(indexNames , inplace=True)

# Remove all rows whose "Text" column is NaN
for i in merged_new[merged_new['Text'].isnull()].index.tolist():
        merged_new = merged_new.drop(i)
        
# Apply first level cleaning
#This function converts to lower-case, removes square bracket, removes numbers and punctuation
cleaned1 = lambda x: text_clean_1(x)
merged_new['cleanedText'] = pd.DataFrame(merged_new.Text.apply(cleaned1))
   
# Apply second level cleaning
cleaned2 = lambda x: text_clean_2(x)
merged_new['cleanedText_new'] = pd.DataFrame(merged_new['cleanedText'].apply(cleaned2))

# Train data
X = merged_new["cleanedText_new"]
y = merged_new["Label"]
tvec = TfidfVectorizer()
gbr = GradientBoostingRegressor()
model = Pipeline([('vectorizer',tvec),('regressor',gbr)])
model.fit(X, y)

# Merge some features from features.tsv into test.tsv
features_selected = featuresDf.drop(["DayofWeek", "Month", "DayofMonth", "Year"], axis = 1)
mergedTest = testDf.merge(features_selected, how='inner', left_on='RowID', right_on='RowID')

# values with NaN will be set to "empty"
for i in mergedTest[mergedTest['Text'].isnull()].index.tolist():
        mergedTest.Text.loc[i] = "empty"

# Predict on test set
Xtest = mergedTest["Text"]
prediction = model.predict(Xtest)
prediction = pd.Series(prediction)
prediction = pd.DataFrame(prediction, columns = ["Prediction"])

# Prepare output file
outputDf = pd.DataFrame(columns = ["RowID", "Prediction"])
outputDf["RowID"] = testDf["RowID"]
outputDf["Prediction"] = prediction["Prediction"]
outputDf.to_csv("../S3862092/A3-4.tsv", sep = "\t", header = False ,index = False)