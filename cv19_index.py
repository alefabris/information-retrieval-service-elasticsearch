import string
import sys
import getopt
import os.path
import json
from elasticsearch import Elasticsearch

INDEXNAME = 'cv19index'
DOCTYPE   = 'cv19doc'
VERBOSE   = True

FILEIDS = "docids1.txt"
FILEDOCS = "docs1.json"

es = Elasticsearch([{'host':'localhost','port':9200}])

infileid = open(FILEIDS,'r')
infiledocs = open(FILEDOCS,'r')

validlist = list()
stringa = infileid.readline()[0:-1]
validlist.append(stringa)
while stringa is not "":
   stringa = infileid.readline()[0:-1]
   validlist.append(stringa)

ndoc = 0
for doc in infiledocs:
    if len(doc) > 0:
        ndoc += 1
        cv19_doc = json.loads(doc)
        this_id = str(cv19_doc["cord_uid"])
        if this_id is not None and this_id in validlist:
            if VERBOSE:
                print(FILEDOCS,ndoc,this_id,'...',end='')                 
            res = es.index(index=INDEXNAME,
                           doc_type=DOCTYPE,
                           id=this_id,
                           body=cv19_doc)
            if VERBOSE:
                print("added")                 
    if VERBOSE:
        print("done")

if VERBOSE:
    print("done")



    
