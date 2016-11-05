#!/usr/bin/python
'''
@File:           WikiSearch.py
@Description:    PySpark application for retrieving top K wikipedia pages relevant
                 to search query. This application uses tf-idf inverted index data 
                 structure built over wikipedia corpus.
@Author:         Chetan Borse
@EMail:          cborse@uncc.edu
@Created on:     10/31/2016
@Usage:          spark-submit --master yarn --deploy-mode client
                    src/WikiSearch.py
                    -i "output/Spark/InvertedIndexer/InvertedIndex"
                    -q "parks in usa"
                    -k 100
                    -o "output/Spark/InvertedIndexer/WikiSearch"
@python_version: 2.6
===============================================================================
'''


import argparse
import warnings

from pyspark import SparkContext, SparkConf

warnings.filterwarnings("ignore")


query = ""


def FindMatchingWikiPage(token):
    """
    Find wikipedia pages matching to a given search query.

    @param: token    => token from tf-idf inverted index file

    @yields:         => Tuple of (<wikipedia page>, <term frequency>)
    """
    term = ''
    wikiPage = ''
    termFrequency = 0.0

    if not token:
        return

    # Extract term, wikipedia page and term frequency from token
    term = token.split("$#$#$")[0]
    wikiPage = token.split("$#$#$")[1].rsplit("\t", 1)[0]
    termFrequency = float(token.split("$#$#$")[1].rsplit("\t", 1)[1])

    # If a valid term, wikipedia page and term frequency exist, then yield
    # (<wikipedia page>, <term frequency>)
    if term and wikiPage and termFrequency:
        for q in query:
            if q == term:
                yield (wikiPage, termFrequency)


def WikiSearch(**args):
    """
    Entry point for Wikipedia Search application, which retrieves top K 
    wikipedia pages relevant to search query.
    """
    global query;

    # Read arguments
    invertedIndex = args['inverted_index']
    query = args['query']
    resultSize = args['top_k']
    output = args['output']

    # Create SparkContext object
    conf = SparkConf()
    conf.setAppName("WikiSearch")
    sc = SparkContext(conf=conf)

    # Read in the inverted index built over wikipedia pages
    invertedIndex = sc.textFile(invertedIndex)

    # Preprocess search query
    query = query.split()
    query = map(lambda x: x.strip().lower(), query)
    query = map(lambda x: x.replace("[", "").replace("]", ""), query)
    query = filter(None, query)

    if query:
        # Find relevance scores of wikipedia pages that match to a given
        # search query
        wikiPageRelevance = invertedIndex.flatMap(FindMatchingWikiPage) \
                                         .filter(lambda x: x != None) \
                                         .reduceByKey(lambda x, y: x + y)

        # Retrieve top K wikipedia pages relevant to search query
        topWikiPage = wikiPageRelevance.sortBy(lambda x: x[1], ascending=False, numPartitions=1) \
                                       .take(resultSize)

        # Save results
        topWikiPage = sc.parallelize(topWikiPage) \
                        .coalesce(1, shuffle=False) \
                        .map(lambda x: x[0] + '\t' + str(x[1]))
        topWikiPage.saveAsTextFile(output)
    else:
        print "Please enter a valid query!"

    # Shut down SparkContext
    sc.stop()


if __name__ == "__main__":
    """
    Entry point.
    """
    # Argument parser
    parser = argparse.ArgumentParser(description='Wikipedia Search Application',
                                     prog='spark-submit --master yarn --deploy-mode client \
                                           src/WikiSearch.py \
                                           -i <inverted_index> \
                                           -q <query> \
                                           -k <top k result> \
                                           -o <output>')

    parser.add_argument("-i", "--inverted_index", type=str, default="output/Spark/InvertedIndexer/InvertedIndex",
                        help="Wikipedia inverted index, default: output/Spark/InvertedIndexer/InvertedIndex")
    parser.add_argument("-q", "--query", type=str, required=True,
                        help="Search query.")
    parser.add_argument("-k", "--top_k", type=int, default=100,
                        help="Top K results to be retrieved, default: 100")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output directory for storing the top K wikipedia pages relevant to search query.")

    # Read user inputs
    args = vars(parser.parse_args())

    # Run Wikipedia Search Application
    WikiSearch(**args)

