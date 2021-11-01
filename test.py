import string
import sys
import getopt
import os.path
import json
from elasticsearch import Elasticsearch

INDEXNAME = 'cv19index'
DOCTYPE   = 'cv19doc'
VERBOSE   = True

FILEIDS = "docids.txt"
FILEDOCS = "docs.json"

es = Elasticsearch([{'host':'localhost','port':9200}])

infileid = open(FILEIDS,'r')
infiledocs = open(FILEDOCS,'r')

validlist = list()
stringa = infileid.readline()[0:-1]
validlist.append(stringa)
while stringa is not "":
   stringa = infileid.readline()[0:-1]
   validlist.append(stringa)

for doc in infiledocs:
    if len(doc) > 0:
        cv19_doc = json.loads(doc)
        this_id = str(cv19_doc["cord_uid"])
        if this_id is not None and this_id in validlist:
            print(this_id," ",cv19_doc["title"] )