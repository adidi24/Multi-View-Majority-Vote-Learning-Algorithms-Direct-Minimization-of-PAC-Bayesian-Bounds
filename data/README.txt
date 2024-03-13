===== Presentation =====
This dataset have been extracted from the Reuters RCV1/RCV2 Multilingual test collection that can be found on http://multilingreuters.iit.nrc.ca/ReutersMultiLingualMultiView.htm

author: Clément Grimal <Clement.Grimal@imag.fr>
		 http://membres-lig.imag.fr/grimal/
		 Questions, suggestions or comments are appreciated!
		
date: February, 2012


===== Description =====
The archive contains 6 samples of 1200 documents, balanced over the 6 labels (E21,CCAT,M11,GCAT,C15,ECAT). Each sample is made of 5 views (EN,FR,GR,IT,SP) on the same documents. The documents were initially in english, and the FR, GR, IT, and SP views corresponds to the words of their traductions respectively in French, German, Italian and Spanish.

The documents have been selected randomly, and the 2000 words have been selected with the k-medoids algorithm.

===== Files =====
All the files are encoded in UTF8.

reutersEN_<sample>_<view>.mtx -- 
	the documents-words matrix, containing the tf-idf scores, in the Matrix Market coordinate format (sparse).

reutersEN_<sample>_<view>.maprow.txt -- 
	the mapping between the rows of the matrix and the id of the document in the original collection.

reutersEN_<sample>_<view>.mapcol.txt -- 
	the mapping between the columns of the matrix and the id of the word in the original collection.

reutersEN_act.txt --
	contains the list of the affectations of the documents to a topic.

labels.txt --
	contains the list of the different labels, in the order of the affectations found in reutersEN_act.txt