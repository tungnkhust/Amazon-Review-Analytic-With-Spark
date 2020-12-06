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
from pyspark.sql import SQLContext, Row, Column
from pyspark.sql.functions import when, udf, countDistinct
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator

warnings.filterwarnings('ignore')

# Create Spark session
spark = SparkSession.builder \
    .appName('AMS') \
    .master('local') \
    .getOrCreate()

# -------------------------------------------
print('-'*50)
data = spark.read.json('data/*')
# data.show(n=10)

# -------------------------------------------
print('-'*50)
print('Num of review:\t{}'.format(data.count()))

# -------------------------------------------
print('-'*50)
no_reviewer = data.select(countDistinct('reviewerID'))
print('Num of reviewer:\t{}'.format(no_reviewer.first()[0]))

# -------------------------------------------
print('-'*50)
print('Num of product:\t{}'.format(data.select(countDistinct('asin')).first()[0]))

# -------------------------------------------
print('-'*50)
print('Num of rating:\t{}'.format(data.select(countDistinct('overall')).first()[0]))
print('No of review by rating:')
count_rating = data.groupBy('overall').count()
print('rating\tcount')
for r in count_rating.collect():
    print('{}\t{}'.format(r['overall'], r['count']))

# -------------------------------------------
print('-'*50)
print('Thống kê số lượng nhận xét theo mỗi người dùng')
count_reviewer = data.groupBy('reviewerID').count()
count_reviewer = count_reviewer.withColumn('scope', when(count_reviewer['count'] <= 5, '5')
                                           .when(count_reviewer['count'] < 10, '10')
                                           .when(count_reviewer['count'] < 15, '15')
                                           .when(count_reviewer['count'] < 20, '20').otherwise('>20'))

hier_count_reviewer = count_reviewer.groupBy('scope').count()
print('scope\tcount')
for r in hier_count_reviewer.collect():
    print('{}\t{}'.format(r['scope'], r['count']))

# -------------------------------------------
print('-'*50)
print('Thống kế số lượng nhận xét theo mỗi sản phẩm')
count_reviewer = data.groupBy('asin').count()
count_reviewer = count_reviewer.withColumn('scope', when(count_reviewer['count'] <= 5, '5')
                                           .when(count_reviewer['count'] < 10, '10')
                                           .when(count_reviewer['count'] < 15, '15')
                                           .when(count_reviewer['count'] < 20, '20').otherwise('>20'))

hier_count_reviewer = count_reviewer.groupBy('scope').count()
print('scope\tcount')
for r in hier_count_reviewer.collect():
    print('{}\t{}'.format(r['scope'], r['count']))

# -------------------------------------------
print('-'*50)
print('Thống kế số lượng người dùng nhận xét theo mỗi sản phẩm')
count_reviewer = data.groupBy('asin').agg(countDistinct("reviewerID").alias('count'))
count_reviewer = count_reviewer.withColumn('scope', when(count_reviewer['count'] <= 5, '5')
                                           .when(count_reviewer['count'] < 10, '10')
                                           .when(count_reviewer['count'] < 15, '15')
                                           .when(count_reviewer['count'] < 20, '20').otherwise('>20'))

hier_count_reviewer = count_reviewer.groupBy('scope').count()
print('scope\tcount')
for r in hier_count_reviewer.collect():
    print('{}\t{}'.format(r['scope'], r['count']))

# -------------------------------------------
print('-'*50)
print('Thống kế số nhận xét được xác thực')
verify_review = data.groupBy('verified').count()
print('verified\tcount')
for r in verify_review.collect():
    print('{}\t{}'.format(r['verified'], r['count']))

# -------------------------------------------
print('-'*50)
print('Thống kế số nhận xét được xác thực (verified-True) theo rating')
true_rating = data.filter("verified == True")
true_rating = true_rating.groupBy('overall').count()
print('rating\tcount')
for r in true_rating.collect():
    print('{}\t{}'.format(r['overall'], r['count']))

# -------------------------------------------
print('-'*50)
print('Thống kế số nhận xét được xác thực (verified-False) theo rating')
true_rating = data.filter("verified == False")
true_rating = true_rating.groupBy('overall').count()
print('rating\tcount')
for r in true_rating.collect():
    print('{}\t{}'.format(r['overall'], r['count']))