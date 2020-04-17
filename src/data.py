from nltk.corpus import stopwords
import pandas as pd
import glob
import json

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

    # loading 5000 files
    # not all files have abstracts. it turns out the first 5000 don't, but most of the last 5000 do (hence for testing, the -1)
    files = []
    for i in range(0, 5000):
        files += [FileReader(all_json[-1*i])]

    # pulling out abstracts
    json_with_abstracts = []
    for ele in files:
        if len(ele.abstract) != 0:
            json_with_abstracts += [ele]

    print()

loadAllData()


def getAbstracts(file):
    pass