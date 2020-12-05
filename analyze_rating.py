import warnings
import pandas as pd
from pyspark.ml import *
from pyspark.ml.classification import *
from pyspark.ml.feature import *
from pyspark.ml.param import *
from pyspark.ml.tuning import *
from pyspark.ml.evaluation import *
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import rand
from sklearn.metrics import classification_report
from time import time
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
from pyspark.sql.functions import when
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator

warnings.filterwarnings('ignore')

# Create Spark session
spark = SparkSession.builder \
    .appName('AMS') \
    .master('local') \
    .getOrCreate()


# Convert rating to label
data = spark.read.json('data/reviews_Musical_Instruments_5.json')
review = data.select(['reviewerID', 'reviewText', 'overall'])
review = review.withColumn('label', when(data["overall"] > 3.0, 1).otherwise(0))

# compute tf-idf

tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
review_word = tokenizer.transform(review)

# remove stop words
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
removed_review = remover.transform(review_word)

# Convert to TF words vector
hashingTF = HashingTF(inputCol="filtered", outputCol="tf")
tf_review = hashingTF.transform(removed_review)

# Convert to TF*IDF words vector
idf = IDF(inputCol="tf", outputCol="features")

idfModel = idf.fit(tf_review)
tfidf_review = idfModel.transform(tf_review)

train, test = tfidf_review.randomSplit([0.8, 0.2], seed=42)

train = train.cache()
test = test.cache()
print('Sample number in the train set : {}'.format(train.count()))
print('Sample number in the test set : {}'.format(test.count()))

nb = NaiveBayes(featuresCol='features', labelCol='label', predictionCol='prediction',
                smoothing=1.0, modelType="multinomial")

# svc = LinearSVC(featuresCol='features', labelCol='label', predictionCol='prediction', regParam=1)
model = nb.fit(train)
prediction = model.transform(test)

acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

accuracy_score = acc_evaluator.evaluate(prediction)
f1_score = f1_evaluator.evaluate(prediction)

print('Accuracy:', accuracy_score)
print('F1-score:', f1_score)