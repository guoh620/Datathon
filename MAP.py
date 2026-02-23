import pandas as pd
import re

data = pd.read_csv("learning_traces.13m.csv")

hours = {"fr" : 675,
         "en" : 600,
         "es"  : 675, 
         "de" : 750,
         "pt" : 675, 
         "it" : 675}

hours_sum = 600+675+700+900+1100+2200

data["diff_language"] = 1 - (data["learning_language"].map(hours))/hours_sum

def extract_pos(cell):
    #extract difficulty of the word 
    pos = cell.split("<")[1].strip(">")

    if pos.startswith("v") or pos == "sep":
        return 0.89
    if pos in ["n", "ant", "cog", "np"]:
        return 0.92
    if pos == "num":
        return 0.95
    if pos in ["acr", "adj", "atn", "comp", "dem", "det", "detnt", "enc", "itg",
                "obj", "ord", "pos", "pro", "qnt", "ref", 
                "rel", "sint", "sup", "tn"]:
        return 0.65
    if pos in ["prn", "pron", "prpers"]:
        return 0.531
    if pos == "pr": 
        return 0.469
    if pos in ["adv", "cnjadv", "preadv"]:
        return 0.45
    return None

def extract_len(cell):
    #extract the word that is being learnt 
    word = cell.split("/")[1].split("<")[0]
    return(len(word))

#create the 2 cols for parts of speech and length of the word
data["POS"] = data["lexeme_string"].apply(extract_pos)
data["len"] = data["lexeme_string"].apply(extract_len)

#filter out NAs
data = data[data["POS"].isna() == False]

data.to_csv("filtered_spaced_rep.csv")

