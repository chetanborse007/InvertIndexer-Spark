#!/usr/bin/python
'''
@File:           InvertedIndexer.py
@Description:    PySpark application for creating inverted index data structure 
                 over corpus of wikipedia pages.
@Author:         Chetan Borse
@EMail:          cborse@uncc.edu
@Created on:     10/31/2016
@Usage:          spark-submit --master yarn --deploy-mode client
                    src/InvertedIndexer.py
                    -i "input/InvertedIndexer"
                    -o "output/Spark/InvertedIndexer/InvertedIndex"
@python_version: 2.6
===============================================================================
'''


import argparse
import re
import math
import warnings

from pyspark import SparkContext, SparkConf

warnings.filterwarnings("ignore")


corpusSize = 0


def ExtractTitleAndText(wikiPage):
    """
    Extract title and text for a given wikipedia page. Tokenize content of 
    text body.

    @param: wikiPage => Wikipedia page

    @yields:         => Tuple of (<Term, Title>, <1>)
    """
    title = ''
    text = ''
    terms = []

    if not wikiPage:
        return

    # Patterns for extracting title and text body from wikipedia page
    titleMatcher = re.match(r'.*<title>(.*?)</title>.*', wikiPage, re.M)
    textMatcher = re.match(r'.*<text.*>(.*?)</text>.*', wikiPage, re.M)

    # Extract title of wikipedia page
    if titleMatcher:
        title = titleMatcher.group(1).strip().lower()

    # Extract text body of wikipedia page
    if textMatcher:
        text = textMatcher.group(1).strip().lower()

    # Tokenize terms in text body of wikipedia page
    if text:
        terms = text.split()
        terms = map(lambda x: x.strip(), terms)
        terms = map(lambda x: x.replace("[", "").replace("]", ""), terms)
        terms = filter(None, terms)

    # If a valid title and terms exist, then yield;
    # (<Term, Title>, <1>)
    if title and terms:
        for term in terms:
            yield (term + "$#$#$" + title, 1)


def TermFrequencyCalculator(term):
    """
    Calculate logarithmic term frequency for a given term in the given 
    wikipedia page.

    @param: term    => term

    @returns:       => Returns (<Term>, <Logarithmic term frequency>)
    """
    if term[1] > 0:
        termFrequency = 1 + math.log10(term[1])
    else:
        termFrequency = 0

    return (term[0], termFrequency)


def FindUniqueWikiPages(token):
    """
    Find unique wikipedia pages.

    @param: token   => token from term frequency data structure

    @returns:       => Tuple of (<Title>, <1>)
    """
    title = token[0].split("$#$#$")[1]
    return (title, 1)


def PostingGenerator(token):
    """
    Generate posting file over wikipedia pages.

    @param: token   => token from term frequency data structure

    @returns:       => Tuple of (<Term>, <Title, Term Frequency>)
    """
    term = token[0].split("$#$#$")[0]
    title = token[0].split("$#$#$")[1]
    termFrequency = token[1]
    return (term, [title + "$#$#$" + str(termFrequency)])


def TFIDFCalculator(token):
    """
    Calculate TF-IDF score for a given term in the given wikipedia page.

    @param: token   => token from posting file

    @yields:        => Tuple of (<Term, Title>, <TF-IDF score>)
    """
    # Compute document frequency
    wikiPageFrequency = len(token[1])

    # Compute inverse document frequency
    inverseWikiPageFrequency = math.log10(1 + (corpusSize / (1 + wikiPageFrequency)))

    # Compute tf-idf score
    for t in token[1]:
        wikiPage = t.split("$#$#$")[0]
        termFrequency = t.split("$#$#$")[1]
        
        score = float(termFrequency) * inverseWikiPageFrequency
        
        yield (token[0] + "$#$#$" + wikiPage, score)


def InvertedIndexer(**args):
    """
    Entry point for Wikipedia Inverted Indexer application, which computes 
    inverted index over corpus of wikipedia page.
    """
    global corpusSize;

    # Read arguments
    input = args['input']
    output = args['output']

    # Create SparkContext object
    conf = SparkConf()
    conf.setAppName("WikiInvertedIndexer")
    sc = SparkContext(conf=conf)

    # Read in the corpus of wikipedia pages
    input = sc.textFile(input)

    # Calculate logarithmic term frequency for a given term 
    # in the given wikipedia page
    termFrequency = input.flatMap(ExtractTitleAndText) \
                         .filter(lambda x: x != None) \
                         .reduceByKey(lambda x, y: x + y) \
                         .map(TermFrequencyCalculator)

    # Count the number of valid wikipedia pages in corpus
    corpusSize = termFrequency.map(FindUniqueWikiPages) \
                              .filter(lambda x: x != None) \
                              .reduceByKey(lambda x, y: x + y) \
                              .count()

    # Generate inverted index using TF-IDF score computed for a given term 
    # in the given wikipedia page.
    invertedIndex = termFrequency.map(PostingGenerator) \
                                 .filter(lambda x: x != None) \
                                 .reduceByKey(lambda x, y: x + y) \
                                 .flatMap(TFIDFCalculator)

    # Save results
    invertedIndex = invertedIndex.map(lambda x: x[0] + '\t' + str(x[1]))
    invertedIndex.saveAsTextFile(output)

    # Shut down SparkContext
    sc.stop()


if __name__ == "__main__":
    """
    Entry point.
    """
    # Argument parser
    parser = argparse.ArgumentParser(description='Wikipedia Inverted Indexer Application',
                                     prog='spark-submit --master yarn --deploy-mode client \
                                           src/InvertedIndexer.py \
                                           -i <input> \
                                           -o <output>')

    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input for Wikipedia Inverted Index computations.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output directory for storing the Wikipedia Inverted Index.")

    # Read user inputs
    args = vars(parser.parse_args())

    # Run Wikipedia Inverted Index Application
    InvertedIndexer(**args)

