# Creation an Information Retrieval Service with ElasticSearch

## Summary

The problem assigned to me was of *Information Retrieval* on the data of the *first round of COVID-19*. In this project I have experimented with two sorting methods different from the basic one implemented by Elasticsearch, that is, reordering the documents found with BM25 through clustering. In the first *pure clustering* approach, the order is established in the first instance by the cluster to which it belongs and subsequently the order for documents belonging to the same cluster is established by means of the BM25 score. In the second *hybrid* approach, on the other hand, the cluster modifies the score value which therefore determines the new order.
The hybrid approach returned similar performance to the BM25, while with the classic clustering worse results were obtained. We then surrounded the IR with a simple graphical interface developed by us, so as to improve the *user experience*.

## Index
1. [Introduction](#intro)<br>
2. [Starting point](#sp)<br>
3. [Methodology](#me)<br>
4. [Experiments](#ex)<br>
5. [Interface](#int)<br>
6. [Bibliographical references](#br)<br>

## 1. Introduction<a id=intro> </a>

The main objective of this project is to put into practice the concepts learned in the Databases 2 course and add others learned in other subjects and independently.
In this project in particular we have deepened the clustering method: in particular trying to exploit the cluster hypotesis, ie that the relevant elements are similar to each other. Clustering was applied to documents retrieved from a query using BM25, the default method applied by Elasticsearch, in order to reorder them and provide the user with the relevant documents faster.
In addition, we have created a graphical interface for the IR service to provide a minimum level of user experience.
As far as programming is concerned, the python language was used for the creation of the new *Information Retrieval* methods and for the back-end part of the *Web Application*. The HTML and CSS languages ​​were instead used for the creation of the front-end part of the *Web Application*.


## 2. Starting point<a id=sp> </a>

The methods used in this Mini-project, as mentioned above, are BM25 and clustering, used in combination using two different approaches:
* *sorting by clustering*;
* *hybrid sorting between BM25 and clustering*.
BM25 is a function of finding a probabilistic model, given by the relationship between the likelihood of the hypothesis that the document is relevant and that that the document is not. The BM25 sorting algorithm therefore weighs the relationship between the document and the query.
The most common scoring function for BM25 is:

![Figure 1](https://github.com/alefabris/)

* *r<sub>i</sub>* - number of documents found containing the i-th term
* *R* - number of documents found
* *n<sub>i</sub>* - number of documents containing the i-th term
* *N* - number of documents
* *k<sub>1</sub>* - parameter that controls the scaling function
* *k<sub>2</sub>* - parameter that controls the scaling function
* *b* - parameter that controls how the length of a document affects the relevance score
* *dl* - length of the document
* *avdl* - average of document length
* *Q* - set of query terms


## 3. Methodology<a id=me> </a>

In order to cluster the documents found, we went to carry out the pre-processing phases, starting from the title and abstract of the documents, defined below.
* **Tokenization**, that is the reduction of the text into smaller parts defined precisely tokens;
* **Stop words removal**, i.e. the removal of punctuation and stop words, words belonging to a specific language that serve as a link in certain sentences, but are extraneous to the lexicon of the context to be analyzed. Among these are, for example, conjunctions, adverbs, words that appear in the text to be analyzed too frequently or too rarely. The list of stop words relating to the English language is used, the language present in all documents;
* **Stemming**, or the reduction of words to their root. The words obtained at the end of this process contain only certain characters, of the standard alphanumeric type;
* **Creation of the matrix of the stems**, that is a matrix with many
rows as the number of documents considered, as many columns as there are unique stems found in the documents and containing the frequency of appearance of the stem in individual documents.
Subsequently, we implemented the clustering performed on the stem matrix, obtained from the results of the BM25 method (explained above). As a clustering methodology, we used a partition method in which the number of groups is initially fixed. Precisely, the algorithm we used is K-means, an unsupervised machine learning technique. In this type of cluster, K is set equal to the number of clusters we want to obtain.
The algorithm that defines the partition is the following:
* **1.** K centroids are initialized, chosen at random from the documents;
* **2.** Each document is assigned to a cluster, in particular the specific document is assigned to the cluster identified by the closest centroid (the centroid that minimizes the distance, in our case expressed as Euclidean norm);
* **3.** The centroids are recalculated: A centroid z<sub>k</sub> of the *k-th* cluster is the midpoint between all the points that represent the objects within the *k-th* cluster in space, namely:

	![Figure 2](https://github.com/alefabris/)
  
	* *x<sub>i</sub>* - object
	* x<sub>i</sub> - vector representing xi
	* *I* - event indicator function *x<sub>i</sub>* ∈ *z<sub>k</sub>*
* **4.** 2-3 iterates until the partition remains unchanged for 2 iterations.
As a first reordering approach we considered the cluster of relevant documents to be the cluster with more elements, and we ordered the documents by placing the largest clusters first and then sorting the documents within the cluster using the score. As previously stated, being the k-means method a partition method, the number of groups is fixed prior to their creation, therefore to obtain the best possible result using this methodology we have opted for an optimization of this hyperparameter K, precisely we have tried values ​​from 2 to 9. The best result in terms of *map*, that is *Mean Average Precision*, was equal to 2.
The number of clusters was not the only hyperparameter we optimized, in fact another one we used was *TITLERELEVANCE*; this hyperparameter represents the weight of the title with respect to the abstract of the documents found. We made the choice to insert it because it seemed plausible that the title had a greater relevance than the abstract in terms of retrieval. The best result in terms of map was equal to 1, therefore the hypothesis that the title is more relevant than the abstract has been refuted. The values ​​we entered within the hyperparameter grid were all integers from 0 to 4.
As a second sorting approach, we opted for a hybrid sorting between the cluster and the BM25. What we did was therefore to take a K equal to 2 (best value found in the previous method) and we identified the best cluster through a summary index, that is the average, to the score attributed. At this point, we have two clusters with two corresponding indexes of synthesis of the score, we therefore know which of the two clusters is the best from the point of view of the *score* (synthesized), we therefore define the cluster with this highest index as the *best cluster* . To the *scores* relating to the documents belonging to the *best cluster*, let's add and multiply two parameters, respectively *alpha* and *beta*.
As previously done, we built a grid of hyperparameters within which we entered values ​​for alpha, respectively all integers from 0 to 4; as regards beta we used all the values ​​starting from 0.75 up to 2 (with steps of 0.25). The optimized hyperparameters are respectively equal to 2 and 1. Therefore, only the additive constant was significant.


## 4. Experiments<a id=ex> </a>

By comparing the precision of the three methods in various numbers of documents, it can be seen that the BM25 method and the hybrid approach are similar in terms of precision; furthermore, the poor precision of the method based on *k-means* is also noted. Below we can see these values in comparison:

![Figure 3](https://github.com/alefabris/)

**(a) Precision of Methods**

![Figure 4](https://github.com/alefabris/)

**(b) First 20 documents (c) First 100 documents (d) First 1000 documents**

As seen from the table and graphs, we obtained a method, using the approach we define as hybrid, with precision very similar to the basic one implemented by *ElasticSearch* through BM25. The method based mainly on k-means, on the other hand, gave disappointing results. That said, we have kept the standard *ElasticSearch* ordering in our *web application*, as the increase in the computational cost to calculate the matrix of styles and to run *k-means* does not generate an increase in performance that justifies its use.
To reproduce the experiments, your computer must meet certain requirements:
* having installed *ElasticSearch*, more precisely the version we use is 7.6.2;
* having installed python3, more precisely the version we use is 3.6;
* having installed all the python modules we use, for more information about these modules consult the README.txt file;
* having installed a web browser.
The operating system we used, as during the course, was LinuxMint.
First of all, the ElasticSearch server must be started as follows:
open the terminal in the folder that contains the *ElasticSearch* installation and type ```elasticsearch-7.6.2/bin/elasticsearch``` (if you are using another version replace "7.6.2" with your version) send the command and wait for the opening the server.
To reproduce the two experiments, go to the folder entered by us on moodle, click with the right mouse button and open the terminal and execute the following commands:
* **Index documents within *ElasticSearch*** using the command: ```python3 cv 19index.py```;
(You only need to do it once) ´
* **Experiment 1**, that is the search for the best values ​​of *K* and *TITLERELEVANCE* in the sorting described in the first approach, using the command: ```python3 experiment0.py```;
**ATTENTION: it can take several hours to process** (Note that the results will be printed on the terminal)
* **Experiment 2**, that is the search for the best *alpha* and *beta* values ​​in the sorting described in the second approach, using the command: ```python3 experiment1.py```;
**ATTENTION: it can take several hours to process** (Note that the results will be printed on the terminal)


## 5. Interface<a id=int> </a>

As mentioned initially, we decided to develop a graphical interface, which was developed using *HTML* and *CSS*, while maintaining minimal graphics. In addition to the search bar, we have implemented for example a hypertext link on the writing of the home page: *Information Retrieval* that leads to the *Wikipedia* page of that topic. Also on the *home page*, we have inserted hyperlinks that redirect to our university emails.
As for the *research page*, we have made the text: *Information Retrieval* clickable, this click takes you back to the home page, like some of the best known IR services. Once a search has been carried out, the first 100 documents found appear on the *research page*. Each document found is presented within a gray rectangle, with slightly rounded corners, within this box there are: *title* and the first 225 characters of the *abstract*, followed by *...see more*. The *title* and *see more* are hyperlinks that refer to the relative web address which contains the entire document found.

![Figure 5](https://github.com/alefabris/)

**(e) IR home page (f) IR research page**

To use the graphical interface you need to have the *ElasticSearch* server active, go to the folder entered by us on moodle, right-click, open the terminal and execute the following commands:
* ```python3 cv 19index.py``` (Not needed if you have done this before)
* ```python3 web app.py```
* connect to the browser and go to ```http://127.0.0.1:5000```


## 6. Bibliographical references<a id=br> </a>

**1.** W. Bruce Croft, Donald Metzler, Trevor Strohman: Search Engine, Information
Retrieval in Practice. Addison Wesley, 2009 Oct 2017
