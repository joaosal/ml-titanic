
/*
# First, we read and cleanup data
*/
val data = spark.read.
  option("inferSchema", true).
  option("header", true).
  csv("hdfs:///user/ljiang/titanic/titanic.csv").toDF
  
/*  
#Data Dictionary

##Variable	Definition	Key
survival	Survival	0 = No, 1 = Yes
pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex	Sex	
Age	Age in years	
sibsp	# of siblings / spouses aboard the Titanic	
parch	# of parents / children aboard the Titanic	
ticket	Ticket number	
fare	Passenger fare	
cabin	Cabin number	
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
#Variable Notes

pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.
  */
data.show
data.printSchema
  
data.describe().filter($"summary" === "count").show

import org.apache.spark.sql.functions._
val cleandata = data.drop("Cabin").
  drop("Embarked").
  filter($"Age".isNotNull)
  
cleandata.describe().filter($"summary" === "count").show

/*
# Splitting Data for both training and testing. Cache both dataset for performance
*/
  
val Array(trainData, testData) = cleandata.randomSplit(Array(0.8, 0.2), seed=1L)
trainData.cache()
testData.cache()


import org.apache.spark.ml.feature.StringIndexer  
val sexIndexer = new StringIndexer().
  setInputCol("Sex").
  setOutputCol("SexIndex")
  
  
import org.apache.spark.ml.feature.VectorAssembler
val inputCols = Array("Pclass", "SexIndex", "Age","SibSp", "Parch", "Fare")
val assembler = new VectorAssembler().
  setInputCols(inputCols).
  setOutputCol("features")

  
import org.apache.spark.ml.feature.VectorIndexer
val vectorIndexer = new VectorIndexer().
  setInputCol("features").
  setOutputCol("IndexedFeatures").
  setMaxCategories(3)
  

import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import scala.util.Random
val classifier = new DecisionTreeClassifier().
  setSeed(1L).
  setLabelCol("Survived").
  setFeaturesCol("IndexedFeatures").
  setPredictionCol("Prediction").
  setMaxDepth(9).
  setMaxBins(100).
  setImpurity("gini")
  

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator  
// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().
  setLabelCol("Survived").
  setPredictionCol("Prediction").
  setMetricName("accuracy")

  
import org.apache.spark.ml.{Pipeline, PipelineModel}

val pipeline = new Pipeline().setStages(Array(sexIndexer, assembler, vectorIndexer, classifier))
val model = pipeline.fit(trainData)
val trainPredictions = model.transform(trainData)
val testPredictions = model.transform(testData)  
  
  
val trainAccuracy = evaluator.evaluate(trainPredictions)
println("Train Data Accuracy = " + trainAccuracy)  
  
val testAccuracy = evaluator.evaluate(testPredictions)
println("Test Data Accuracy = " + testAccuracy) 

val dtModel = model.stages.last.asInstanceOf[DecisionTreeClassificationModel]  
println(dtModel.toDebugString)
println(dtModel.extractParamMap)

/*
# Choose the best hyperparameters using validation set
*/
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator, TrainValidationSplit}
import org.apache.spark.ml.feature.VectorIndexerModel
val paramGrid = new ParamGridBuilder().
  addGrid(classifier.maxDepth, Array(5,7,9)).
  //addGrid(classifier.maxBins, Array(20, 50, 100)).
  addGrid(classifier.impurity, Array("gini", "entropy")).
  build()

val trainValidationSplit = new TrainValidationSplit().
  setSeed(1L).
  setEstimator(pipeline).
  setEvaluator(evaluator).
  setEstimatorParamMaps(paramGrid).
  setTrainRatio(0.8)
  
val validatorModel = trainValidationSplit.fit(trainData)
val bestModel = validatorModel.bestModel  
println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)
val myVectorIndexer = bestModel.asInstanceOf[PipelineModel].stages(2).asInstanceOf[VectorIndexerModel]

 
val bestDTModel = bestModel.asInstanceOf[PipelineModel].stages.last.asInstanceOf[DecisionTreeClassificationModel]  
bestDTModel.featureImportances.toArray.zip(inputCols).sorted.reverse.foreach(println) 

  
val trainAccuracy = evaluator.evaluate(bestModel.transform(trainData))
println("Train Data Accuracy = " + trainAccuracy)  
  
val testAccuracy = evaluator.evaluate(bestModel.transform(testData))  
println("Test Data Accuracy = " + testAccuracy) 


  
  