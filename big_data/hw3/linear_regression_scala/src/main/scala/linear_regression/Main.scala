package linear_regression

import breeze.linalg._
import breeze.numerics._

import java.io.File

object Main {

    def main(args: Array[String]): Unit = {

        // Read data from files
        // trainPath is args(0) e.g. "C:\\Users\\Otabek Nazarov\\IdeaProjects\\linear_regression_scala\\house_prices_train.csv"
        val dataFileTrain = new File(args(0))
        val trainData = csvread(dataFileTrain,',', skipLines = 1)

        //  testPath is args(1) e.g. "C:\\Users\\Otabek Nazarov\\IdeaProjects\\linear_regression_scala\\house_prices_test.csv"
        val dataFileTest = new File(args(1))
        val testData = csvread(dataFileTest,',', skipLines = 1)

        // Read data
        val xData = trainData(::, 1 to (trainData.cols - 2))
        val yData = trainData(::, (trainData.cols - 1))

        // Set train and validation split
        val SPLIT = 0.8
        val splitRow = (xData.rows * SPLIT).toInt

        // Training set
        val xTrain = xData(0 to splitRow - 1, ::)
        val yTrain = yData(0 to splitRow - 1)

        // Validation set
        val xValid = xData(splitRow to xData.rows - 1, ::)
        val yValid = yData(splitRow to yData.length - 1)

        // Test set
        val xTest = testData(::, 1 to (testData.cols - 2))
        val yTest = testData(::, (testData.cols - 1))

        // Fit linear regression model and predict validation prices
        val linReg = new LinearRegression(xTrain, yTrain)
        val yValidPred = linReg.predict(xValid)
        val yTestPred = linReg.predict(xTest)

        // Print the results of model predictions
        print("Average housing price: ")
        println((sum(yTrain) / yTrain.length).toInt) // 1206551
        print("Mean square error for predicted housing prices on validation set: ")
        print(getRMSE(yValid, yValidPred).toInt) // 218814

        // Save model predictions to file
        val outputFile = new File("predicted_prices.txt")
        csvwrite(outputFile, yTestPred.toDenseMatrix, separator = ',')
    }


    class LinearRegression(var X: DenseMatrix[Double], var y: DenseVector[Double]) {

        // Calculate coefficients of linear regression
        var beta = inv(X.t * X) * X.t * y

        def getX = this.X

        def getBeta() = this.beta

        def getY = this.y

        // Run prediction of model for a given data
        def predict(testX: DenseMatrix[Double]) : DenseVector[Double] = {
            return testX * this.beta
        }
    }


    def getRMSE(predictedY: DenseVector[Double], trueY: DenseVector[Double]): Double = {
        // Calculate root mean square error between two datasets
        return sqrt(sum(pow((trueY - predictedY), 2)) / trueY.length)
    }
}
