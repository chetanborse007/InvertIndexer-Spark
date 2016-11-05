# Create new input directory on HDFS and copy input files from local
# filesystem to HDFS filesystem.
hadoop fs -rm -r input/InvertedIndexer
hadoop fs -mkdir -p input/InvertedIndexer
hadoop fs -put input/* input/InvertedIndexer


# Create new output directories on HDFS and delete existing output 
# directories generated by previous execution instance of 
# MapReduce application.
hadoop fs -mkdir -p output/Spark/InvertedIndexer
hadoop fs -rm -r output/Spark/InvertedIndexer


# Run MapReduce application on HDFS.
spark-submit --master yarn --deploy-mode client src/InvertedIndexer.py -i "input/InvertedIndexer" -o "output/Spark/InvertedIndexer/InvertedIndex"


# Get final output from HDFS.
rm -r InvertedIndexer/
hadoop fs -get output/Spark/InvertedIndexer
