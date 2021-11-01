import string
import sys
import getopt
import os.path
import json
from elasticsearch import Elasticsearch

INDEXNAME = 'cv19index'
VERBOSE = True

FILEQUERY = "query1.json"

es = Elasticsearch([{'host':'localhost','port':9200}])

def resToText(fileout,response,query = "1", n = 1000, tag = "tag"):
    rank = 0
    if VERBOSE:
        print("processing query ",query,"...")
    for hit in response['hits']['hits']:
        rank += 1
        line = "\t".join([str(query),"Q0",str(hit["_source"]['cord_uid']),str(rank),str(hit['_score']),str(tag)+"\n"])
        fileout.write(line)
        if rank > 1000: 
            break

infile = open(FILEQUERY,'r')
oufile = open("output.txt",'w')

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
        response = es.search(index=INDEXNAME,body=query_dict,size = 1000) 
        resToText(response=response, query=num, n=1000, tag="tag", fileout=oufile)

oufile.close() 
         
if VERBOSE:
        print("\ndone")