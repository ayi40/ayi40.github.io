---
title: Spark
date: 2024/5/13 20:46:25
categories:
  - [ML, BDC]
---

Spark

<!-- more -->

# 搭建环境

[Taming Big Data with Apache Spark and Python - Getting Started - Sundog Education with Frank Kane (sundog-education.com)](https://www.sundog-education.com/spark-python/)

**Note**：Step12使用`` pyspark``指令前记得先进入anaconda中的环境。

# Introduction

## Overview

TBC

# Resilient Distributed Data Set（RDD）弹性分布式数据集

RDD是数据集，我们通常对一个RDD做一些操作去获得另外一个RDD。我们需要实例化一个对象``SparkContext``来执行这些操作。

## Basic Operation

### Transforming RDD

+ map
+ flatmap
+ filter: removing information potentially that you don't  care about
+ distinct
+ sample
+ union, intersection, subtract, cartesian

### RDD Action

+ collect
+ count： 统计RDD中value出现的次数
+ countByValue
+ take
+ top
+ reduce
+ ... and more ...

### Example：统计user对movie的评分

```python
from pyspark import SparkConf, SparkContext
import collections

# setMaster指定在单个主机(local)还是在集群(cluster)中运行，这里我们暂时使用单线程
conf = SparkConf().setMaster("local").setAppName("RatingsHistogram")
# 创建SparkContext
sc = SparkContext(conf = conf)

# 使用sc.textFile创建RDD，text中每一行（整行文本）对应RDD中一个值
# exp：
#user_id movie_id rating timestep
#196 242 4 991232423
#186 302 3 984927391
#......
lines = sc.textFile("file:///SparkCourse/ml-100k/u.data")
# 使用map与lambda对RDD进行transform
ratings = lines.map(lambda x: x.split()[2])
# 对新的RDD进行action
result = ratings.countByValue()

sortedResults = collections.OrderedDict(sorted(result.items()))
for key, value in sortedResults.items():
    print("%s %i" % (key, value))
```

## Key-Value RDD

### Create Key-Value RDD

```
totalsByAge = rdd.map(lambda x:(x,1))
```

键值对的value不一定非得是一个值，也可以是列表

### Special Action

+ reduceByKey(): combine values with the same key using some function.

  exp: use ``rdd.reduceByKey(lambda x,y:x+y)`` to adds values up

+ groupByKey(): Group values with the same key

+ sortByKey(): Sort RDD by key values

+ keys(), values(): create an RDD of just the keys, or just the values

### Example: 统计一定年龄段的朋友有多少

```python
# datasets
# indexd name age friends_num
# 0 Will 33 385
# 1 Jean 33 2
# 2 Huge 55 221
# 3 Luke 40 465
#...

from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("FriendsByAge")
sc = SparkContext(conf = conf)

# transform to key-value RDD
def parseLine(line):
    fields = line.split(',')
    age = int(fields[2])
    numFriends = int(fields[3])
    return (age, numFriends)

lines = sc.textFile("file:///SparkCourse/fakefriends.csv")
rdd = lines.map(parseLine)

# transform (33,385) to (33, (385, 1)),and them sum up respectively, 385用来计算总朋友数，1用来计算人头数用于求平均
totalsByAge = rdd.mapValues(lambda x: (x, 1)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
averagesByAge = totalsByAge.mapValues(lambda x: x[0] / x[1])

results = averagesByAge.collect()
for result in results:
    print(result)

```

## Filtering RDD

map：对RDD处理，input和output始终是一对一关系

flatmap： 可以从一个value生成多个values

e.g. (The quick red fox ...)		——》	lines.flatmap(lambda x:x.split())	——》	(The) (quick) (red) (fox)...

### Example: 找出气候站一年的最低温低

```python
# dataset
# wether sation code, date, type, temp, other
# ITE00100554, 18000101, TMAX, -75,,,E 

from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("MinTemperatures")
sc = SparkContext(conf = conf)

def parseLine(line):
    fields = line.split(',')
    stationID = fields[0]
    entryType = fields[2]
    temperature = float(fields[3]) * 0.1 * (9.0 / 5.0) + 32.0
    return (stationID, entryType, temperature)

lines = sc.textFile("file:///SparkCourse/1800.csv")
parsedLines = lines.map(parseLine)
minTemps = parsedLines.filter(lambda x: "TMIN" in x[1])
stationTemps = minTemps.map(lambda x: (x[0], x[2]))
minTemps = stationTemps.reduceByKey(lambda x, y: min(x,y))
results = minTemps.collect();

for result in results:
    print(result[0] + "\t{:.2f}F".format(result[1]))
```

## Flatmap

### Example： 统计文本中词汇出现次数

```python
from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf = conf)

input = sc.textFile("file:///sparkcourse/book.txt")
words = input.flatMap(lambda x: x.split())
wordCounts = words.countByValue()

for word, count in wordCounts.items():
    cleanWord = word.encode('ascii', 'ignore')
    if (cleanWord):
        print(cleanWord.decode() + " " + str(count))

```

上面代码只通过空格分解，会出现``spark,``这种情况，我们下面用正则表达式改进一下

```python
import re
from pyspark import SparkConf, SparkContext

def normalizeWords(text):
    return re.compile(r'\W+', re.UNICODE).split(text.lower())

conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf = conf)

input = sc.textFile("file:///sparkcourse/book.txt")
words = input.flatMap(normalizeWords)
wordCounts = words.countByValue()

for word, count in wordCounts.items():
    cleanWord = word.encode('ascii', 'ignore')
    if (cleanWord):
        print(cleanWord.decode() + " " + str(count))
```

加上排序

```python
import re
from pyspark import SparkConf, SparkContext

def normalizeWords(text):
    return re.compile(r'\W+', re.UNICODE).split(text.lower())

conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf = conf)

input = sc.textFile("file:///sparkcourse/book.txt")
words = input.flatMap(normalizeWords)

# 用另外一种方法实现词频统计
wordCounts = words.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)
# key，value转换，然后用sordbykey方法
wordCountsSorted = wordCounts.map(lambda x: (x[1], x[0])).sortByKey()
results = wordCountsSorted.collect()

for result in results:
    count = str(result[0])
    word = result[1].encode('ascii', 'ignore')
    if (word):
        print(word.decode() + ":\t\t" + count)
```

# Spark SQL

 一种dataframe，可以用sql语句查询，可以与rdd互相转换

## Example：将RDD转为SparkSQL

```python
from pyspark.sql import SparkSession
from pyspark.sql import Row

# Create a SparkSessiony用于操作SparkSQL
# spark.getOrCreate()与spark.close()相对应
spark = SparkSession.builder.appName("SparkSQL").getOrCreate()

def mapper(line):
    fields = line.split(',')
    return Row(ID=int(fields[0]), name=str(fields[1].encode("utf-8")), \
               age=int(fields[2]), numFriends=int(fields[3]))

# 创建一个value是Row的RDD
lines = spark.sparkContext.textFile("fakefriends.csv")
people = lines.map(mapper)

# 利用RDD创建DataFrame
# Infer the schema, and register the DataFrame as a table.
# cache是将这个表存入内存里
schemaPeople = spark.createDataFrame(people).cache()
# 要对DataFrame进行操作，需要创建一个临时View（如果View已经存在则替换）
schemaPeople.createOrReplaceTempView("people")

# SQL can be run over DataFrames that have been registered as a table.
# return a dataframe
# 这里people对应View的名字，age对于创建Row是给的名字
teenagers = spark.sql("SELECT * FROM people WHERE age >= 13 AND age <= 19")

# The results of SQL queries are RDDs and support all the normal RDD operations.
for teen in teenagers.collect():
  print(teen)

# We can also use functions instead of SQL queries:
schemaPeople.groupBy("age").count().orderBy("age").show()

spark.stop()

```

## Example：直接打开DataFrame+用执行代码处理数据

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SparkSQL").getOrCreate()

# 读取csv文件
# option("header", "true")表明这个csv文件有header
# option("inferSchema", "true")要求推理检测模式
people = spark.read.option("header", "true").option("inferSchema", "true")\
    .csv("file:///SparkCourse/fakefriends-header.csv")

# print属性名与属性类型
print("Here is our inferred schema:")
people.printSchema()

print("Let's display the name column:")
people.select("name").show()

print("Filter out anyone over 21:")
people.filter(people.age < 21).show()

print("Group by age")
people.groupBy("age").count().show()

print("Make everyone 10 years older:")
people.select(people.name, people.age + 10).show()

spark.stop()

```

## Example：计算某个年龄平均有几个朋友

```python
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions as func

spark = SparkSession.builder.appName("FriendsByAge").getOrCreate()

lines = spark.read.option("header", "true").option("inferSchema", "true").csv("file:///SparkCourse/fakefriends-header.csv")

# Select only age and numFriends columns
friendsByAge = lines.select("age", "friends")

# From friendsByAge we group by "age" and then compute average
friendsByAge.groupBy("age").avg("friends").show()

# Sorted
friendsByAge.groupBy("age").avg("friends").sort("age").show()

# Formatted more nicely
# agg()聚合多个命令，func.round()取小数点后几位
friendsByAge.groupBy("age").agg(func.round(func.avg("friends"), 2)).sort("age").show()

# With a custom column name
# alias()可以自定义列的名字
friendsByAge.groupBy("age").agg(func.round(func.avg("friends"), 2)
  .alias("friends_avg")).sort("age").show()

spark.stop()

```

## func

Passing columns as parameters

+ func.explode(): similar to flatmap
+ func.split()
+ func.lower()

### Example: WordCounting处理非结构数据

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as func

spark = SparkSession.builder.appName("WordCount").getOrCreate()

# Read each line of my book into a dataframe
inputDF = spark.read.text("file:///SparkCourse/book.txt")

# Split using a regular expression that extracts words
words = inputDF.select(func.explode(func.split(inputDF.value, "\\W+")).alias("word"))
wordsWithoutEmptyString = words.filter(words.word != "")

# Normalize everything to lowercase
lowercaseWords = wordsWithoutEmptyString.select(func.lower(wordsWithoutEmptyString.word).alias("word"))

# Count up the occurrences of each word
wordCounts = lowercaseWords.groupBy("word").count()

# Sort by counts
wordCountsSorted = wordCounts.sort("count")

# Show the results.
wordCountsSorted.show(wordCountsSorted.count())
```

### Example: 找最大最小温度

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

spark = SparkSession.builder.appName("MinTemperatures").getOrCreate()

# define the schema
# 根据列顺序分配
schema = StructType([ \
                     StructField("stationID", StringType(), True), \
                     StructField("date", IntegerType(), True), \
                     StructField("measure_type", StringType(), True), \
                     StructField("temperature", FloatType(), True)])

# // Read the file as dataframe
df = spark.read.schema(schema).csv("file:///SparkCourse/1800.csv")
df.printSchema()

# Filter out all but TMIN entries
minTemps = df.filter(df.measure_type == "TMIN")

# Select only stationID and temperature
stationTemps = minTemps.select("stationID", "temperature")

# Aggregate to find minimum temperature for every station
minTempsByStation = stationTemps.groupBy("stationID").min("temperature")
# 当有show()等action才会开始真正执行上面的代码
minTempsByStation.show()

# Convert temperature to fahrenheit and sort the dataset
# withColumn()新建一列名为“temperature",value是第二个参数
# 创建好新列后再进行select
minTempsByStationF = minTempsByStation.withColumn("temperature",
                                                  func.round(func.col("min(temperature)") * 0.1 * (9.0 / 5.0) + 32.0, 2))\
                                                  .select("stationID", "temperature").sort("temperature")
                                                  
# Collect, format, and print the results
results = minTempsByStationF.collect()

for result in results:
    print(result[0] + "\t{:.2f}F".format(result[1]))
    
spark.stop()
```

### Exercise：customer_order

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

spark = SparkSession.builder.appName("TotalSpentByCustomer").master("local[*]").getOrCreate()

# Create schema when reading customer-orders
customerOrderSchema = StructType([ \
                                  StructField("cust_id", IntegerType(), True),
                                  StructField("item_id", IntegerType(), True),
                                  StructField("amount_spent", FloatType(), True)
                                  ])

# Load up the data into spark dataset
customersDF = spark.read.schema(customerOrderSchema).csv("file:///SparkCourse/customer-orders.csv")

totalByCustomer = customersDF.groupBy("cust_id").agg(func.round(func.sum("amount_spent"), 2) \
                                      .alias("total_spent"))

totalByCustomerSorted = totalByCustomer.sort("total_spent")

# totalByCustomerSorted.count()算出整个table一共多少行，这是为了输出整个表
totalByCustomerSorted.show(totalByCustomerSorted.count())

spark.stop()
```

## Advanced Example

### 找出最受欢迎的电影-最多评分数的电影

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import StructType, StructField, IntegerType, LongType

spark = SparkSession.builder.appName("PopularMovies").getOrCreate()

# Create schema when reading u.data
schema = StructType([ \
                     StructField("userID", IntegerType(), True), \
                     StructField("movieID", IntegerType(), True), \
                     StructField("rating", IntegerType(), True), \
                     StructField("timestamp", LongType(), True)])

# Load up movie data as dataframe
# option("sep", "\t")说明以"\t"为分割符
moviesDF = spark.read.option("sep", "\t").schema(schema).csv("file:///SparkCourse/ml-100k/u.data")

# Some SQL-style magic to sort all movies by popularity in one line!
topMovieIDs = moviesDF.groupBy("movieID").count().orderBy(func.desc("count"))

# Grab the top 10
topMovieIDs.show(10)

# Stop the session
spark.stop()
```

### Broadcast

将一个变量分发到cluster上每个点

example：给电影ID找对于的电影名字

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import StructType, StructField, IntegerType, LongType
import codecs

def loadMovieNames():
    movieNames = {}
    # CHANGE THIS TO THE PATH TO YOUR u.ITEM FILE:
    with codecs.open("E:/SparkCourse/ml-100k/u.ITEM", "r", encoding='ISO-8859-1', errors='ignore') as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames

spark = SparkSession.builder.appName("PopularMovies").getOrCreate()

# 将loadMovieNames方法return的字典分发到cluster上所有节点
nameDict = spark.sparkContext.broadcast(loadMovieNames())

# Create schema when reading u.data
schema = StructType([ \
                     StructField("userID", IntegerType(), True), \
                     StructField("movieID", IntegerType(), True), \
                     StructField("rating", IntegerType(), True), \
                     StructField("timestamp", LongType(), True)])

# Load up movie data as dataframe
moviesDF = spark.read.option("sep", "\t").schema(schema).csv("file:///SparkCourse/ml-100k/u.data")

movieCounts = moviesDF.groupBy("movieID").count()

# Create a user-defined function to look up movie names from our broadcasted dictionary
def lookupName(movieID):
    return nameDict.value[movieID]
lookupNameUDF = func.udf(lookupName)

# Add a movieTitle column using our new udf
moviesWithNames = movieCounts.withColumn("movieTitle", lookupNameUDF(func.col("movieID")))

# Sort the results
# 新的一种排序方法
sortedMoviesWithNames = moviesWithNames.orderBy(func.desc("count"))

# Grab the top 10
sortedMoviesWithNames.show(10, False)

# Stop the session
spark.stop()

```



## Example: find the most popular superhero

数据集：

+ Marvel-graph.txt

  4395 7483 9475 7483 

  4802 3939 ...

  每行第一个ID是superhero ID，后面跟着的是在漫画中和这个superhero同时出现过的superhero ID

  一个超级英雄可能多次在每一行的第一个出现

+ Marvel-names.txt

  superhero ID与名字的映射



### Mission1：找出最受欢迎的superhero

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

spark = SparkSession.builder.appName("MostPopularSuperhero").getOrCreate()

schema = StructType([ \
                     StructField("id", IntegerType(), True), \
                     StructField("name", StringType(), True)])

names = spark.read.schema(schema).option("sep", " ").csv("file:///SparkCourse/Marvel-names.txt")

# 暂时不在意这个datafram的schema
lines = spark.read.text("file:///SparkCourse/Marvel-graph.txt")

# Small tweak vs. what's shown in the video: we trim each line of whitespace as that could
# throw off the counts.
connections = lines.withColumn("id", func.split(func.trim(func.col("value")), " ")[0]) \
    .withColumn("connections", func.size(func.split(func.trim(func.col("value")), " ")) - 1) \
    .groupBy("id").agg(func.sum("connections").alias("connections"))
    
mostPopular = connections.sort(func.col("connections").desc()).first()

mostPopularName = names.filter(func.col("id") == mostPopular[0]).select("name").first()

print(mostPopularName[0] + " is the most popular superhero with " + str(mostPopular[1]) + " co-appearances.")

```

### Mission2：找出最不起眼的superhero

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

spark = SparkSession.builder.appName("MostObscureSuperheroes").getOrCreate()

schema = StructType([ \
                     StructField("id", IntegerType(), True), \
                     StructField("name", StringType(), True)])

names = spark.read.schema(schema).option("sep", " ").csv("file:///SparkCourse/Marvel-names.txt")

lines = spark.read.text("file:///SparkCourse/Marvel-graph.txt")

# Small tweak vs. what's shown in the video: we trim whitespace from each line as this
# could throw the counts off by one.
connections = lines.withColumn("id", func.split(func.trim(func.col("value")), " ")[0]) \
    .withColumn("connections", func.size(func.split(func.trim(func.col("value")), " ")) - 1) \
    .groupBy("id").agg(func.sum("connections").alias("connections"))
    
minConnectionCount = connections.agg(func.min("connections")).first()[0]

minConnections = connections.filter(func.col("connections") == minConnectionCount)

# 使用join方法联合两个表
minConnectionsWithNames = minConnections.join(names, "id")

print("The following characters have only " + str(minConnectionCount) + " connection(s):")

minConnectionsWithNames.select("name").show()
```

