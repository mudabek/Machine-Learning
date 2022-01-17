import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import scala.collection.mutable.HashMap

object TF_IDF {
    def main(args: Array[String]): Unit = {

        // Create spark session
        val spark = SparkSession.builder()
            .master("local[*]")
            .appName("tf-idf")
            .getOrCreate()

        // Read data file, lowercase text, remove special characters, split strings into array
        var df = spark.read
            .option("header", "true")
            .option("interschema", "true")
            .csv("tripadvisor_hotel_reviews.csv")
            .withColumn("Review", lower(col("Review")))
            .withColumn("Review", regexp_replace(col("Review"), "[^a-z _]", ""))
            .withColumn("Review", split(col("Review"), " "))

        // Calculate term frequencies of each review
        df = df.withColumn("doc_id", monotonically_increasing_id()) // add unique id to each review
        val columns = df.columns.map(col):+(explode(col("Review")) as "token") // bring out each doc token as row
        var termFrequencies = df.select(columns: _*) // merge tables

        termFrequencies = termFrequencies
            .groupBy("doc_id", "token")
            .agg(count("Review") as "term_freq")

        // Calculate document frequencies and get top N = 100 of them
        val N = 100
        val totalReviewCount = df.count()
        val docFrequencies = termFrequencies
            .groupBy("token")
            .agg(countDistinct("doc_id") as "doc_freq")
            .orderBy(desc("doc_freq"))
            .limit(N)
            .withColumn("idf", log(lit(totalReviewCount) / (col("doc_freq") + 1) + 1))

        // Join two tables and calculate tf-idf
        val joinedTable = termFrequencies
            .join(docFrequencies, "token")
            .withColumn("tf-idf", col("idf") * col("term_freq"))
            .orderBy("doc_id")

        // Pivot the table
        val pivotDF = joinedTable
            .groupBy("doc_id")
            .pivot("token")
            .mean("tf-idf")
            .show()

        // Print the table to see the result
        print(pivotDF)
    }
}
