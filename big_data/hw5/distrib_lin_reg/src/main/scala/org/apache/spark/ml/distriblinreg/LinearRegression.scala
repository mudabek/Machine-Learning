package org.apache.spark.ml.distriblinreg

import breeze.linalg.{functions, sum, DenseVector => BreezeDenseVector}
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{Param, ParamMap, DoubleParam}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions.lit
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.{StructType, DoubleType}
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.ml.feature.VectorAssembler

trait LinRegParams extends HasInputCol with HasOutputCol {

    protected def validateAndTransformSchema(schema: StructType): StructType = {
        SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())
        SchemaUtils.checkColumnType(schema, getOutputCol, DoubleType)
        schema
    }
}

trait LinearRegressionParams extends HasInputCol with HasOutputCol {

    def setInputCol(value: String) : this.type = set(inputCol, value)
    def setOutputCol(value: String): this.type = set(outputCol, value)

    // learning rate of linear regression
    val lr = new DoubleParam(this, "lr", "learning rate")
    def setLR(value: Double) : this.type = set(lr, value)
    setDefault(lr -> 1e-3)

    // column to store predictions
    val predCol: Param[String] = new Param[String](this, "y_pred", "y_pred")
    def setPredictionCol(value: String): this.type = set(predCol, value)
    def getPredictionCol: String = $(predCol)
    setDefault(predCol, "y_pred")

    protected def validateAndTransformSchema(schema: StructType): StructType = {
        SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

        if (schema.fieldNames.contains($(outputCol))) {
            SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
            schema
        } else {
            SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
        }
    }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
    with DefaultParamsWritable {

    def this() = this(Identifiable.randomUID("LinearRegression"))

    override def fit(dataset: Dataset[_]): LinearRegressionModel = {

        implicit val encoder : Encoder[Vector] = ExpressionEncoder()

        val assembler = new VectorAssembler()
            .setInputCols(Array($(inputCol), "ones", $(outputCol)))
            .setOutputCol("result")

        val resultColumn = dataset.withColumn("result", lit(1))
        val trainData: Dataset[Vector] = assembler
            .transform(resultColumn)
            .select(col="result")
            .as[Vector]

        var thetas = BreezeDenseVector.rand[Double](MetadataUtils.getNumFeatures(dataset, $(inputCol)) + 1)
        var thetasOld = thetas.copy
        thetasOld = 99999.0 * thetasOld

        while (functions.euclideanDistance(thetas.toDenseVector, thetasOld.toDenseVector) > 1e-4) {
            val summary = trainData.rdd.mapPartitions((data: Iterator[Vector]) => {
                val result = new MultivariateOnlineSummarizer()
                data.foreach(cur => {
                    val X = cur.asBreeze(0 to thetas.size - 1).toDenseVector
                    val y = cur.asBreeze(-1)
                    val eps = sum(X * thetas) - y
                    val deltaThetas = X * eps
                    result.add(mllib.linalg.Vectors.fromBreeze(deltaThetas))
                })
                Iterator(result)
            }).reduce(_ merge _)
            thetasOld = thetas.copy
            thetas = thetas - summary.mean.asBreeze * $(lr)
        }
        copyValues(new LinearRegressionModel(thetas).setParent(this))
    }

    override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

    override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[distriblinreg](
                                                    override val uid: String,
                                                    val thetas: BreezeDenseVector[Double])
    extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {


    private[distriblinreg] def this(thetas: BreezeDenseVector[Double]) =
        this(Identifiable.randomUID("LinearRegressionModel"), thetas)

    override def copy(extra: ParamMap): LinearRegressionModel = defaultCopy(extra)

    override def transform(dataset: Dataset[_]): DataFrame = {
        val transformUdf = {
            dataset.sqlContext.udf.register(uid + "_transform",
                (x : Vector) => {
                    sum(x.asBreeze.toDenseVector * thetas(0 to thetas.size - 2)) + thetas(-1)
                }
            )
        }
        dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
    }

    override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

    override def write: MLWriter = new DefaultParamsWriter(this) {
        override protected def saveImpl(path: String): Unit = {
            super.saveImpl(path)

            val vectors = Tuple1(Vectors.fromBreeze(thetas))

            sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/thetas")
        }
    }

    def getThetas: BreezeDenseVector[Double] = {thetas}
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
    override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
        override def load(path: String): LinearRegressionModel = {
            val metadata = DefaultParamsReader.loadMetadata(path, sc)

            val vectors = sqlContext.read.parquet(path + "/thetas")

            // Used to convert untyped dataframes to datasets with vectors
            implicit val encoder : Encoder[Vector] = ExpressionEncoder()

            val thetas =  vectors.select(vectors("_1").as[Vector]).first().asBreeze.toDenseVector

            val model = new LinearRegressionModel(thetas)
            metadata.getAndSetParams(model)
            model
        }
    }
}
