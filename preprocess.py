import sys
import os
import sqlite3
import pandas as pd
import wordfreq
from sklearn.model_selection import train_test_split

# Parse .sql files from loanwords_gairaigo repo and convert to csv format
def to_csv(sql_path, out_dir):
    db = sqlite3.connect(os.path.join(out_dir, "merged.db"))
    query = open(sql_path, encoding='utf-8').read()
    cursor = db.cursor()
    cursor.executescript(query)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table_name in tables:
        table_name = table_name[0]
        table = pd.read_sql_query("SELECT * from %s" % table_name, db)
        table.to_csv(table_name + '.csv', index_label='index')
    cursor.close()
    db.close()

# Load csv and generate train/test split
# also augment with word frequency
def split_data(csv_path, val_split=0.1, 
                out_train="train.csv", out_val="val.csv"):
    df = pd.read_csv(csv_path, encoding="utf-8", delimiter=',')
    probs = [wordfreq.word_frequency(str(x), 'en') for x in df["english"]]
    df['frequency'] = probs
    # create train and validation set 
    train, val = train_test_split(df, test_size=val_split)
    train.to_csv(out_train, index=False)
    val.to_csv(out_val, index=False) 

# usage: python preprocess.py <path to loadwords_gairaigo> <output dir>
if __name__ == "__main__":
    to_csv(os.path.join(sys.argv[1], "db", "merged.sql"), 
           sys.argv[2])
    split_data(os.path.join(sys.argv[2], "merged.csv"))