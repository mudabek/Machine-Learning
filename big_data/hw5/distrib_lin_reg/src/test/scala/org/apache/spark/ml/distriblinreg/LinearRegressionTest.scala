package org.apache.spark.ml.distriblinreg

import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import breeze.linalg.{*, DenseMatrix, DenseVector}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vectors}
import org.apache.spark.sql.{DataFrame}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

    val delta = 0.0001
    val thetas = DenseVector(1.5, 0.3, -0.7)
    val X = DenseMatrix.rand(1000, 3)
    val shift = DenseVector.rand(1000) * 0.001
    val y = X * thetas + shift

    val data = DenseMatrix.horzcat(X, y.asDenseMatrix.t)

    import sqlc.implicits._
    val _df = data(*, ::).iterator
        .map(x => (x(0), x(1), x(2), x(3)))
        .toSeq.toDF("x1", "x2", "x3", "y")


    val assembler = new VectorAssembler()
        .setInputCols(Array("x1", "x2", "x3"))
        .setOutputCol("features")

    val df = assembler.transform(_df).select("features", "y")

    private def checkMSE(model: LinearRegressionModel): Unit = {
        val evaluator = new RegressionEvaluator()
            .setLabelCol("y")
            .setPredictionCol("y_pred")
            .setMetricName("mse")
        val mse = evaluator.evaluate(model.transform(df))
        mse should be <= 0.1
    }

    "Estimator" should "have correct thetas" in {
        val linRegModel = new LinearRegression()
            .setInputCol("features")
            .setOutputCol("y")
        val model = linRegModel.fit(df)
        val modelThetas = model.getThetas
        modelThetas(0) should be(thetas(0) +- delta)
        modelThetas(1) should be(thetas(1) +- delta)
        modelThetas(2) should be(thetas(2) +- delta)
    }

    "Estimator" should "work after re-read" in {

        val pipeline = new Pipeline().setStages(
            Array(
                new LinearRegression()
                    .setInputCol("X")
                    .setOutputCol("y")
                    .setPredictionCol("y_pred")
            )
        )

        val tmpFolder = Files.createTempDir()

        pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

        val reRead = Pipeline.load(tmpFolder.getAbsolutePath)

        val model = reRead.fit(data).stages(0).asInstanceOf[LinearRegressionModel]

        checkMSE(model)
    }

    "Model" should "work after re-read" in {

        val pipeline = new Pipeline().setStages(
            Array(
                new LinearRegression()
                    .setInputCol("X")
                    .setOutputCol("y")
                    .setPredictionCol("y_pred")
            )
        )

        val model = pipeline.fit(data)

        val tmpFolder = Files.createTempDir()

        model.write.overwrite().save(tmpFolder.getAbsolutePath)

        val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)

        checkMSE(reRead.stages(0).asInstanceOf[LinearRegressionModel])
    }
}

object LinearRegressionTest extends WithSpark {

    lazy val _vectors = Seq(
        Vectors.dense(13.5, 12),
        Vectors.dense(-1, 0)
    )

    lazy val _data: DataFrame = {
        import sqlc.implicits._
        _vectors.map(x => Tuple1(x)).toDF("features")
    }
}