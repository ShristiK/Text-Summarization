import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.approximation as nxaa
from collections import OrderedDict, deque
import copy
import operator

from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import pandas
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


# G=nx.Graph()
# G.add_node(1)
# G.add_node(2)
# G.add_node(3)
# G.add_node(4)
# G.add_node(5)
# G.add_edges_from([(1,2),(1,3),(2,4),(3,4),(2,5),(3,5)]) 


### reading data file

df = pandas.read_csv('C:/Users/Akriti Garg/Downloads/tennis_articles_v4.csv')


### cleaning sentences, by removing non alphabet characters and converting to lower case letters

s = ""
d = {}
for a in df['article_text']:
      s += a

sentences = sent_tokenize(s)
clean_sentences = []
for s in sentences:
    temp = re.sub("[^a-zA-Z]"," ",s)
    temp = temp.lower()
    clean_sentences.append(temp)
    d[temp] = s 



### defined a functiom for removing stop words which are downloaded from NLTk's list of english stop words

stop_words = stopwords.words('english')
def rem_stop(s):
    var = ""
    words = nltk.word_tokenize(s)
    for w in words:
        if( w not in stop_words):
           var+=w+" "
    return var



### removed the stop words using the function defined above

dict = {}
clean = []
# print clean_sentences
for s in clean_sentences:
    temp = rem_stop(s)
    clean.append(temp)
    dict[temp] = d[s]


### loaded pre trained word2vec model from Gensim

from gensim.models import KeyedVectors
filename = 'Downloads/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)



### creating vector representation of sentences after extracting word vectors


word_embeddings = {}
words = list(model.wv.vocab)
for a in words:
    word_embeddings[a]=model[a]


sentence_vectors = []
for i in clean:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((300,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((300,))
  sentence_vectors.append(v)


### generating the final summary after producing the graph using networkx 

sentence_similarity_martix = np.zeros([len(sentences), len(sentences)])
for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sentence_similarity_martix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,300), sentence_vectors[j].reshape(1,300))[0,0]

G = nx.from_numpy_array(sentence_similarity_martix)

assert nx.is_connected(G)

### finding minimum connected dominating set using a greedy approach

G2 = copy.deepcopy(G)

		# Step 1: initialization
		# take the node with maximum degree as the starting node
starting_node = max(dict(G2.degree()).items(), key=operator.itemgetter(1))[0] 
fixed_nodes = {starting_node}

		# Enqueue the neighbor nodes of starting node to Q in descending order by their degree
neighbor_nodes = G2.neighbors(starting_node)
neighbor_nodes_sorted =list( OrderedDict(sorted(dict(G2.degree(neighbor_nodes)).items(), key=operator.itemgetter(1), reverse=True)).keys())

priority_queue = deque(neighbor_nodes_sorted) # a priority queue is maintained centrally to decide whether an element would be a part of CDS.
# print([starting_node]+neighbor_nodes_sorted)
inserted_set = set(neighbor_nodes_sorted + [starting_node])

		# Step 2: calculate the cds
while priority_queue:
	u = priority_queue.pop()

			# check if the graph after removing u is still connected
	rest_graph = copy.deepcopy(G2)
	rest_graph.remove_node(u)

	if nx.is_connected(rest_graph):
		G2.remove_node(u)
	else: # is not connected 
		fixed_nodes.add(u)

				# add neighbors of u to the priority queue, which never are inserted into Q
		inserted_neighbors = set(G2.neighbors(u)) - inserted_set
		inserted_neighbors_sorted = OrderedDict(sorted(dict(G2.degree(inserted_neighbors)).items(),
																key=operator.itemgetter(1), reverse=True)).keys()

		priority_queue.extend(inserted_neighbors_sorted)
		inserted_set.update(inserted_neighbors_sorted)

		# Step 3: verify the result
assert nx.is_dominating_set(G, fixed_nodes) and nx.is_connected(G.subgraph(fixed_nodes))

print fixed_nodes
