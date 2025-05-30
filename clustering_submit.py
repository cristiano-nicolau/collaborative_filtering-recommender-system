"""
Item-Item Collaborative Filtering with Clustering using PySpark
This script implements an item-item collaborative filtering recommendation system using clustering techniques with PySpark.
It downloads the specified MovieLens dataset, processes it, and applies KMeans clustering to group similar items.

It then computes cosine similarities within clusters and generates predictions based on user ratings.

This script requires PySpark and pandas to be installed in your Python environment.

Usage:
    spark-submit clustering_submit.py -d small -o ./output/ -k 20

Parameters:
    -d small: Specify the dataset to use (options: small, ml-1m, ml-10m, ml-20m, ml-25m)
    -o ./output/: Specify the output directory to save results
    -k 20: Specify the number of clusters for KMeans (default is set to sqrt(num_items) if not provided)

Created by: Cristiano Nicolau
"""


import argparse
import os
import time
import math
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, struct, sum as sql_sum, udf
from pyspark.ml.linalg import Vectors
from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.ml.feature import Normalizer
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import DenseVector
from pyspark.sql.types import DoubleType

class DatasetManager:
    DATASET_LINKS = {
        "small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "ml-10m": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
        "ml-20m": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
        "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    }
    DATASET_PATHS = {
        "small": "./data/ml-latest-small/ratings.csv",
        "ml-1m": "./data/ml-1m/ratings.csv",
        "ml-10m": "./data/ml-10m/ratings.csv",
        "ml-20m": "./data/ml-20m/ratings.csv",
        "ml-25m": "./data/ml-25m/ratings.csv",
    }
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset_path = self.DATASET_PATHS[dataset_name]

    def ensure_dataset(self):
        if not os.path.exists(self.dataset_path):
            self.download_and_extract_dataset()
            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(f"Failed to find dataset at {self.dataset_path} after download.")
            
    def download_and_extract_dataset(self):
        import zipfile
        import urllib.request
        url = self.DATASET_LINKS[self.dataset_name]
        zip_path = f"./data/{self.dataset_name}.zip"
        out_dir = f"./data/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(out_dir)
        os.remove(zip_path)
        print(f"Downloaded and extracted {self.dataset_name}.")
        # se o ratings dor .dat transforma para .csv
        if self.dataset_name in ["ml-1m", "ml-10m", "ml-20m", "ml-25m"]:
            ratings_path = os.path.join(out_dir, self.dataset_name, "ratings.dat")
            if os.path.exists(ratings_path):
                df = pd.read_csv(ratings_path, sep="::", header=None, names=["userId", "movieId", "rating", "timestamp"])
                df.to_csv(self.DATASET_PATHS[self.dataset_name], index=False)
                print(f"Converted {ratings_path} to CSV format at {self.DATASET_PATHS[self.dataset_name]}.")

def cosine_sim(v1, v2):
    return float(v1.dot(v2)) / (float(v1.norm(2)) * float(v2.norm(2)))

def to_sparse_vector(user_ratings, size):
    sorted_pairs = sorted(user_ratings, key=lambda x: x.userId)
    indices = [x.userId - 1 for x in sorted_pairs]
    values = [float(x.rating) for x in sorted_pairs]
    return Vectors.sparse(size, indices, values)

class ClusteringRecommender:
    def __init__(self, dataset, output_dir, k=None):
        self.dataset = dataset
        self.output_dir = output_dir
        self.k = k
        self.spark = None
        self.ratings = None
        self.test = None
        self.rmse = None
        self.mae = None

    def start_spark(self):
        self.spark = SparkSession.builder \
            .appName("ItemItemCF_Clustering") \
            .config("spark.driver.memory", "16g") \
            .config("spark.executor.memory", "16g") \
            .config("spark.executor.cores", "4") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")

    def load_data(self, dataset_path):
        data = self.spark.read.csv(dataset_path, header=True, inferSchema=True) \
            .select("userId", "movieId", "rating")
        self.ratings, self.test = data.randomSplit([0.9, 0.1], seed=42)
        self.ratings.cache()
        self.test.cache()

    def fit_predict(self):
        spark = self.spark
        spark.udf.register("to_sparse_vector", to_sparse_vector)

        item_user = self.ratings.groupBy("movieId") \
            .agg(collect_list(struct("userId", "rating")).alias("user_ratings"))
        num_users = self.ratings.select("userId").distinct().count()
        item_vectors_rdd = item_user.rdd.map(
            lambda row: Row(
                movieId=row["movieId"],
                features=to_sparse_vector(row["user_ratings"], num_users)
            )
        )

        item_vectors = spark.createDataFrame(item_vectors_rdd)

        normalizer = Normalizer(inputCol="features", outputCol="norm_features", p=2.0)
        normalized_item_vectors = normalizer.transform(item_vectors)
        num_items = item_vectors.count()

        self.k = self.k if self.k is not None else max(10, min(100, int(math.sqrt(num_items))))
        print(f"Using {k} clusters for KMeans.")
        kmeans = KMeans(k=k, seed=42, featuresCol="norm_features", predictionCol="cluster")
        kmeans_model = kmeans.fit(normalized_item_vectors)
        
        clustered_items = kmeans_model.transform(normalized_item_vectors).select("movieId", "cluster", "features")
        cosine_sim_udf = udf(lambda x, y: float(cosine_sim(x, y)), returnType=DoubleType())
        cross_joined = clustered_items.alias("a").join(
            clustered_items.alias("b"),
            (col("a.cluster") == col("b.cluster")) & (col("a.movieId") < col("b.movieId"))
        )
        similarities = cross_joined.withColumn(
            "cosine_sim", cosine_sim_udf(col("a.features"), col("b.features"))
        ).select(
            col("a.movieId").alias("i_mv"),
            col("b.movieId").alias("j_mv"),
            "cosine_sim"
        )
        similarities = similarities.union(
            similarities.selectExpr("j_mv as i_mv", "i_mv as j_mv", "cosine_sim")
        ).cache()
        test_neighbors = self.test.alias("t") \
            .join(similarities.alias("s"), col("t.movieId") == col("s.i_mv"))
        test_with_ratings = test_neighbors \
            .join(self.ratings.alias("r"), (col("t.userId") == col("r.userId")) & (col("s.j_mv") == col("r.movieId"))) \
            .select(
                col("t.userId"),
                col("t.movieId").alias("target_movie"),
                col("s.j_mv").alias("neighbor_movie"),
                col("s.cosine_sim"),
                col("r.rating").alias("neighbor_rating")
            )
        weighted_sums = test_with_ratings.groupBy("userId", "target_movie").agg(
            sql_sum(col("cosine_sim") * col("neighbor_rating")).alias("weighted_rating_sum"),
            sql_sum(F.abs(col("cosine_sim"))).alias("similarity_sum")
        )
        predictions = weighted_sums.withColumn(
            "pred_rating",
            F.when(
                col("similarity_sum") > 0,
                col("weighted_rating_sum") / col("similarity_sum")
            ).otherwise(None)
        ).filter(col("pred_rating").isNotNull()) \
         .select("userId", "target_movie", "pred_rating")
        final_results = predictions.alias("p") \
            .join(self.test.alias("t"), (col("p.userId") == col("t.userId")) & (col("p.target_movie") == col("t.movieId"))) \
            .select(
                col("p.userId"),
                col("p.target_movie"),
                col("p.pred_rating"),
                col("t.rating").alias("actual_rating")
            )
        final_results_filtered = final_results.filter(col("pred_rating").isNotNull())
        evaluator = RegressionEvaluator(
            labelCol="actual_rating",
            predictionCol="pred_rating",
            metricName="rmse"
        )
        self.rmse = evaluator.evaluate(final_results_filtered)
        mae_evaluator = RegressionEvaluator(
            labelCol="actual_rating",
            predictionCol="pred_rating",
            metricName="mae"
        )
        self.mae = mae_evaluator.evaluate(final_results_filtered)
        return final_results_filtered
    def save_results(self, summary, output_path):
        if os.path.exists(output_path):
            summary.to_csv(output_path, mode='a', header=False, index=False)
        else:
            summary.to_csv(output_path, index=False)
    def run(self, dataset_path):
        start_time = time.time()
        self.start_spark()
        self.load_data(dataset_path)
        print(f"Loaded dataset {self.dataset} with {self.ratings.count()} ratings and {self.test.count()} test samples.")
        print("Starting item-item collaborative filtering with clustering...")
        results = self.fit_predict()
        print("Item-item collaborative filtering with clustering completed.")
        print(f"RMSE: {self.rmse}, MAE: {self.mae}")
        execution_time = round(time.time() - start_time, 2)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        summary = pd.DataFrame([{
            'Dataset': self.dataset,
            'Train Size': self.ratings.count(),
            'Test Size': self.test.count(),
            'Similarity': 'Cosine (within cluster)',
            'Clustering': f"KMeans (k={self.k})",
            'RMSE': self.rmse,
            'MAE': self.mae,
            'Execution Time (s)': execution_time
        }])
        output_path = os.path.join(self.output_dir, 'item_item_cf_summary.csv')
        self.save_results(summary, output_path)
        print(f"Results saved to {output_path}")
        print(f"Total execution time: {execution_time} seconds")
        self.spark.stop()

def main():
    parser = argparse.ArgumentParser(description="Item-Item CF with Clustering (PySpark)")
    parser.add_argument('-d','--dataset', type=str, default='small', choices=DatasetManager.DATASET_PATHS.keys(), help='Dataset to use')
    parser.add_argument('-o', '--output_dir', type=str, default='./output/', help='Output directory')
    parser.add_argument('-k','--number_k', type=int, default=None, help='Number of clusters for KMeans (default: sqrt(num_items))')
    args = parser.parse_args()
    dataset_manager = DatasetManager(args.dataset)
    dataset_manager.ensure_dataset()
    recommender = ClusteringRecommender(
        dataset=args.dataset,
        output_dir=args.output_dir,
        k=args.number_k
    )
    recommender.run(dataset_manager.dataset_path)

if __name__ == "__main__":
    main()
