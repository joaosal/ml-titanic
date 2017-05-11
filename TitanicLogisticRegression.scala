
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
  
val Array(trainData, testData) = cleandata.randomSplit(Array(0.8, 0.2), seed=1L)
trainData.cache()
testData.cache()


  
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
val pclassOneHotEncoder = new OneHotEncoder().
  setInputCol("Pclass").
  setOutputCol("PclassVector")
  
val sexIndexer = new StringIndexer().
  setInputCol("Sex").
  setOutputCol("SexIndex")

  
  
import org.apache.spark.ml.feature.VectorAssembler
val inputCols = Array("PclassVector", "SexIndex", "Age","SibSp", "Parch", "Fare")
val assembler = new VectorAssembler().
  setInputCols(inputCols).
  setOutputCol("features")
   

import org.apache.spark.ml.classification.{LogisticRegression,LogisticRegressionModel}
import scala.util.Random
val classifier = new LogisticRegression().
  setLabelCol("Survived").
  setFeaturesCol("features").
  setPredictionCol("Prediction").
  setMaxIter(100)
  
  

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator  
// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().
  setLabelCol("Survived").
  setPredictionCol("Prediction").
  setMetricName("accuracy")

  
import org.apache.spark.ml.{Pipeline, PipelineModel}

val pipeline = new Pipeline().setStages(Array(pclassOneHotEncoder,sexIndexer, assembler,  classifier))


/*
# Choose the best hyperparameters using validation set
*/
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator, TrainValidationSplit}

val paramGrid = new ParamGridBuilder().
  addGrid(classifier.regParam, Array(0.0, 0.01, 0.1)).
  addGrid(classifier.elasticNetParam, Array(0.0, 0.5, 1.0)).
  build()

val trainValidationSplit = new TrainValidationSplit().
  setEstimator(pipeline).
  setEvaluator(evaluator).
  setEstimatorParamMaps(paramGrid).
  setTrainRatio(0.8)
  
val validatorModel = trainValidationSplit.fit(trainData)
val bestModel = validatorModel.bestModel  
println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)

val bestLRModel = bestModel.asInstanceOf[PipelineModel].stages.last.asInstanceOf[LogisticRegressionModel]  
println(s"Coefficients: ${bestLRModel.coefficients} Intercept: ${bestLRModel.intercept}")

val trainPredictions =  bestModel.transform(trainData)
val testPredictions = bestModel.transform(testData) 
  
val trainAccuracy = evaluator.evaluate(trainPredictions)
println("Train Data Accuracy = " + trainAccuracy)  
  
val testAccuracy = evaluator.evaluate(testPredictions)
println("Test Data Accuracy = " + testAccuracy) 


testPredictions.filter($"Survived"!==$"Prediction").
  select("Name","Age", "Sex","Pclass","Survived", "Prediction", "probability").
  show(truncate=false)
  
  
  
