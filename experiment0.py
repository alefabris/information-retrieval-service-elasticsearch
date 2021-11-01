import string
import sys
import getopt
import os.path
import json
import subprocess
import pandas as pd
import nltk
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.cluster import KMeans
from nltk.corpus import stopwords as sw
from string import punctuation as pun
from nltk.tokenize import TweetTokenizer as tt
from nltk.stem.snowball import SnowballStemmer as ss
from sklearn.feature_extraction.text import CountVectorizer
from elasticsearch import Elasticsearch
from sklearn.model_selection import ParameterGrid
nltk.download("stopwords")


INDEXNAME = 'cv19index'
VERBOSE = True
SIZE = 1000

FILEQUERY = "query1.json"
FILERETRIEVALS = "retrieved.txt"

es = Elasticsearch([{'host':'localhost','port':9200}])

def resToText(fileout,response,query = "1", n = 1000, tag = "tag"):
    rank = 0
    for hit in response["hits"]["hits"]:
        rank += 1
        line = "\t".join([str(query),"Q0",str(hit["_source"]['cord_uid']),str(rank),str(hit['_score']),str(tag)+"\n"])
        fileout.write(line)
        if rank > n: 
            break


def pulisci_parola(s):
    return "".join(re.findall("[a-zA-Z0-9]+", s))

def add_spaces(doc):
    string = ""
    for word in doc:
        string = string + word + " "
    return string[0:-1]

def get_stem_matrix(text):
    tokenizer = tt(preserve_case=False, reduce_len=True, strip_handles=True)
    text = [tokenizer.tokenize(t) for t in text]

    stop_words = sw.words("english")+list(pun)
    stemmer = ss("english")

    stemmed_text = []

    for i in range(len(text)):
        stemmed_text.append([])
        for j in range(len(text[i])):
            s = pulisci_parola(text[i][j])
            if s not in stop_words and s !="":
                stemmed_text[i].append(stemmer.stem(s))
                

    vec = CountVectorizer()
    X = vec.fit_transform(add_spaces(doc) for doc in stemmed_text)
    X = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
    return X
    

def standard_retrieval(index, body, size):
    return es.search(index = index, body = body, size = size)


def clustering_based_retrieval(index, body, size, K = 2, TR = 1):
    
    K = int(K)
    TR = int(TR)
    response = es.search(index = index, body = body, size = size)
    res = []
    onlytext = []
    for hit in response["hits"]["hits"]:
        temp = []
        temp.append(hit["_id"])
        temp.append(hit["_score"])
        title = str(hit["_source"]["title"])
        abstract = str(hit["_source"]["abstract"])
        text = ""
        text += ( title + " ") * TR
        text += abstract
        temp.append(title)
        temp.append(abstract)
        res.append(temp)
        onlytext.append(text)

    
    X = get_stem_matrix(onlytext)
    
    kmeans = KMeans(n_clusters = K, random_state = 123)
    kmeans.fit(X)
    clusters = kmeans.predict(X)
    freq_clusters = [0]*K
    for c in clusters: freq_clusters[c] += 1
    
    res = pd.DataFrame(res, columns = ["id", "score", "title", "abstract"])
    res["cluster_rel"] = [freq_clusters[c] for c in clusters]
    res["score"] = [1]*len(res)
    res = res.sort_values(["cluster_rel","score"],ascending = [False,False]).reset_index(drop = True)
    
    response = dict()
    response["hits"] = dict()
    response["hits"]["hits"] = []
    for i in range(len(res)):
        temp_dict = dict()
        temp_dict['_score'] = res.iloc[i,1]
        temp_dict["_source"] = dict()
        temp_dict["_source"]["cord_uid"] = res.iloc[i,0]
        response["hits"]["hits"].append(temp_dict)
      
    
    return response
        
    


def evaluate(retrieval_function = standard_retrieval):
    infile = open(FILEQUERY,'r')
    oufile = open(FILERETRIEVALS,'w')
    for tuttoIlFile in infile:
        queries = json.loads(tuttoIlFile)["topics"]["topic"]
        for query in queries:
            querytext = query["query"]["#text"] + query["question"]["#text"]
            num = query["@number"]       
            query_dict = {
                "query": {
                    "bool": {
                        "should": [
                            { "match": { "title"	:	querytext } },
                            { "match": { "abstract"	:	querytext } }
                            ]
                        }
                    }
                }
            response = retrieval_function(index=INDEXNAME,body=query_dict,size = SIZE) 
            resToText(response=response, query=num, n=SIZE, tag="tag", fileout=oufile)
    oufile.close() 
    infile.close()

def evaluate_kmeans(K = 2, TR = 1):
    infile = open(FILEQUERY,'r')
    oufile = open(FILERETRIEVALS,'w')
    for tuttoIlFile in infile:
        queries = json.loads(tuttoIlFile)["topics"]["topic"]
        for query in queries:
            querytext = query["query"]["#text"] + query["question"]["#text"]
            num = query["@number"]       
            query_dict = {
                "query": {
                    "bool": {
                        "should": [
                            { "match": { "title"	:	querytext } },
                            { "match": { "abstract"	:	querytext } }
                            ]
                        }
                    }
                }
            response = clustering_based_retrieval(index = INDEXNAME, body = query_dict, size = SIZE, K = K, TR = TR) 
            resToText(response=response, query=num, n=SIZE, tag="tag", fileout=oufile)
    oufile.close() 
    infile.close()
    
def get_map():
    result = subprocess.run(["./trec_eval", "qrels.txt", FILERETRIEVALS, "-m", "map"], stdout=subprocess.PIPE)
    s = result.stdout.decode('utf-8')
    return float(s.split()[2])

def get_gm_map():
    result = subprocess.run(["./trec_eval", "qrels.txt", FILERETRIEVALS, "-m", "gm_map"], stdout=subprocess.PIPE)
    s = result.stdout.decode('utf-8')
    return float(s.split()[2])

def get_recip_rank():
    result = subprocess.run(["./trec_eval", "qrels.txt", FILERETRIEVALS, "-m", "recip_rank"], stdout=subprocess.PIPE)
    s = result.stdout.decode('utf-8')
    return float(s.split()[2])

def get_Rprec():
    result = subprocess.run(["./trec_eval", "qrels.txt", FILERETRIEVALS, "-m", "Rprec"], stdout=subprocess.PIPE)
    s = result.stdout.decode('utf-8')
    return float(s.split()[2])

def get_bpref():
    result = subprocess.run(["./trec_eval", "qrels.txt", FILERETRIEVALS, "-m", "bpref"], stdout=subprocess.PIPE)
    s = result.stdout.decode('utf-8')
    return float(s.split()[2])

def get_partial_precision():
    after = [5,10,15,20,30,100,200,500,1000]
    p = []
    for a in after:
        result = subprocess.run(["./trec_eval", "qrels.txt", FILERETRIEVALS, "-m", "P." + str(a)], stdout=subprocess.PIPE)
        p.append(result.stdout.decode('utf-8').split()[2])
    return p,after


######################################################################################

K_val = np.arange(2,10)
TR_val = np.arange(1,5)
best_map = float("-inf")
for K in tqdm(K_val):
    for TR in TR_val:
        evaluate_kmeans(K,TR)
        m = get_map()
        if best_map < m:
            best_map = m
            best_K = K
            best_TR = TR

print("\n#########\nBest K: " + str(best_K))
print("\n#########\nBestTR: " + str(best_TR))

print("...\n...\n...\nperforming es standard retrieval...")
evaluate()
es_p,after = get_partial_precision()

print("...\n...\n...\nperforming clustering based retrieval...\n")
evaluate_kmeans(best_K, best_TR)
kmeans_p,after = get_partial_precision()