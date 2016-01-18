__author__ = 'pranavgoel'

from pyspark import SparkConf,SparkContext
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import sys


def main():
    '''
    takes one input argument :: Location of the directory for training and test data files.
    :return: Print output on console for the area under the ROC curve.
    '''

    conf = SparkConf().setAppName("MLPipeline")
    sc = SparkContext(conf=conf)

    # Read training data as a DataFrame
    sqlCt = SQLContext(sc)
    trainDF = sqlCt.read.parquet("20news_train.parquet")

    # Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features", numFeatures=1000)
    lr = LogisticRegression(maxIter=20, regParam=0.1)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

    # Fit the pipeline to training data.
    model = pipeline.fit(trainDF)

    numFeatures = (1000, 5000, 10000)
    regParam = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    paramGrid = ParamGridBuilder().addGrid(hashingTF.numFeatures, numFeatures).addGrid(lr.regParam, regParam).build()


    cv = CrossValidator().setEstimator(pipeline).setEvaluator(BinaryClassificationEvaluator()).setEstimatorParamMaps(paramGrid).setNumFolds(2)

    # Evaluate the model on testing data
    testDF = sqlCt.read.parquet("20news_test.parquet")
    prediction = model.transform(testDF)
    evaluator = BinaryClassificationEvaluator()


    model_cv = cv.fit(trainDF)
    prediction_cv = model_cv.transform(testDF)
    print evaluator.evaluate(prediction)
    print evaluator.evaluate(prediction_cv)


if __name__ == "__main__":
    main()