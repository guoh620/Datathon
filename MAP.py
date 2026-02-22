import pandas as pd
import re

data = pd.read_csv("learning_traces.13m.csv")

user = data[data["user_id"] == "u:bBMK"]

hours = {"fr" : 675,
         "en" : 600,
         "es"  : 675, 
         "de" : 750,
         "pt" : 675, 
         "it" : 675}

speech = {"n" : 1,
          "v" : 2,
          "adj" : 3,
          "pr" : 4,
          "adv" : 5
          }

hours_sum = 600+675+700+900+1100+2200


data["diff_language"] = 1 - (data["learning_language"].map(hours))/hours_sum

def extract_pos(cell):
    pos = cell.split("<")[1].strip(">")
    if pos.startswith("v") or pos == "sep":
        return 3
    if pos in ["n", "ant", "cog", "np"]:
        return 2
    if pos == "num":
        return 1
    if pos in ["acr", "adj", "atn", "comp", "dem", "det", "detnt", "enc", "itg",
                "obj", "ord", "pos", "pro", "qnt", "ref", 
                "rel", "sint", "sup", "tn"]:
        return 4
    if pos in ["prn", "pron", "prpers"]:
        return 5
    if pos == "pr": 
        return 6
    if pos in ["adv", "cnjadv", "preadv"]:
        return 7
    return None

def extract_len(cell):
    word = cell.split("/")[1].split("<")[0]
    return(len(word))

data["POS"] = data["lexeme_string"].apply(extract_pos)
data["len"] = data["lexeme_string"].apply(extract_len)



print(data[["lexeme_string", "POS", "len"]].head())
print(data["POS"].isna().sum())



