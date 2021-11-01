from elasticsearch import Elasticsearch
from flask import Flask, redirect, url_for, render_template, request

INDEXNAME = 'cv19index'

app = Flask(__name__)
es = Elasticsearch([{'host':'localhost','port':9200}])

#--- Homepage ---#
@app.route("/")
@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        q = request.form["query"]
        return results(q)
    else:
        return render_template("home.html")

@app.route("/results")
def results(q):
    query_dict = {
                "query": {
                    "bool": {
                        "should": [
                            { "match": { "title"	:	q } },
                            { "match": { "abstract"	:	q } }
                            ]
                        }
                    }
                }
    n_docs = 100
    response = es.search(index = INDEXNAME, body = query_dict, size = n_docs)
    documents = []
    n_chars = 225
    for hit in response["hits"]["hits"]:
        temp = dict()
        temp["id"] = (hit["_id"])
        temp["title"] = str(hit["_source"]["title"])
        temp["abstract"] = str(hit["_source"]["abstract"])
        temp["authors"] = str(hit["_source"]["authors"])
        temp["link"] = str(hit["_source"]["url"])
        if len(temp["abstract"]) > n_chars:
            temp["abs_first"] = temp["abstract"][0:n_chars]
            temp["abs_last"] = temp["abstract"][n_chars:]
        else: 
            temp["abs_first"] = temp["abstract"]
            temp["abs_last"] = ""
        documents.append(temp)
        
    return render_template("results.html", documents = documents, q = q)


if __name__ == "__main__":
    app.run(debug = True)
