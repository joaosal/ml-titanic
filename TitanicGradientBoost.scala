
/*
# First, we read and cleanup data
*/
val data = spark.read.
  option("inferSchema", true).
  option("header", true).
  csv("hdfs:///user/ljiang/titanic/titanic.csv").toDF
  
data.show
data.printSchema
  
data.describe().filter($"summary" === "count").show

import org.apache.spark.sql.functions._
val cleandata = data.drop("Cabin").
  drop("Embarked").
  filter($"Age".isNotNull)
  
cleandata.describe().filter($"summary" === "count").show
  
val Array(trainData, testData) = cleandata.randomSplit(Array(0.8, 0.2), seed=1L)
trainData.cache()
testData.cache()
  
  
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
  

import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import scala.util.Random
  


val classifier = new GBTClassifier().
  setSeed(1L).
  setLabelCol("Survived").
  setFeaturesCol("IndexedFeatures").
  setPredictionCol("Prediction")



import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator  
val evaluator = new MulticlassClassificationEvaluator().
  setLabelCol("Survived").
  setPredictionCol("Prediction").
  setMetricName("accuracy")


  
import org.apache.spark.ml.{Pipeline, PipelineModel}

val pipeline = new Pipeline().setStages(Array(pclassIndexer,sexIndexer, assembler, vectorIndexer, classifier))


/*
# Choose the best hyperparameters using validation set
*/
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator, TrainValidationSplit}

val paramGrid = new ParamGridBuilder().
  addGrid(classifier.maxIter, Array(10, 25)).
  addGrid(classifier.maxDepth, Array(2,5)).
  build()

val trainValidationSplit = new TrainValidationSplit().
  setEstimator(pipeline).
  setEvaluator(evaluator).
  setEstimatorParamMaps(paramGrid).
  setTrainRatio(0.8)
  
val validatorModel = trainValidationSplit.fit(trainData)
val bestModel = validatorModel.bestModel  
println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)

  
val trainPredictions = bestModel.transform(trainData)
val testPredictions = bestModel.transform(testData)  
  
  
val trainAccuracy = evaluator.evaluate(trainPredictions)
println("Train Data Accuracy = " + trainAccuracy)  
  
val testAccuracy = evaluator.evaluate(testPredictions)
println("Test Data Accuracy = " + testAccuracy) 