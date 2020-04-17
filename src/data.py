
# must be run once on your machine
# import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords
import pandas as pd
import glob
import json
import string

root_path = '../data/CORD-19-research-challenge/'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str,
    'doi': str
})
meta_df.head()
meta_df.info()

# from https://www.kaggle.com/ivanegapratama/covid-eda-initial-exploration-tool
class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            if "abstract" in content:
                for entry in content['abstract']:
                    self.abstract.append(entry['text'])
                self.abstract = '\n'.join(self.abstract)
            # Body text
            if "body_text" in content:
                for entry in content['body_text']:
                    self.body_text.append(entry['text'])
                self.body_text = '\n'.join(self.body_text)
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'

def loadAllData():
    all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
    first_row = FileReader(all_json[0])  # TODO: replace with loading all data
    print(first_row)

    # all json files
    # files = []
    # for json in all_json:
    #     files += [FileReader(json)]

    # loading singe file
    # not all files have abstracts. it turns out the first 5000 don't, but most of the last 5000 do (hence for testing, the -1)
    fileObjs = []
    for i in range(1, 50):
        fileObjs += [FileReader(all_json[-1*i])]

    return fileObjs

# from https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/
def cleanText(textStr):
    tokens = textStr.split(" ")
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


fileObjs = loadAllData()
# pulling out abstracts
json_with_abstracts = []
for ele in fileObjs:
    if len(ele.abstract) != 0:
        json_with_abstracts += [ele]

abstracts = []
for f in json_with_abstracts:
    cleaned_entry = cleanText(f.abstract)
    abstracts += [cleaned_entry]

print()
