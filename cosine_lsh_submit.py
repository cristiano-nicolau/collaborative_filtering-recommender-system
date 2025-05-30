"""
Item-Item Collaborative Filtering with LSH using Cosine Similarity (PySpark)
This script implements an item-item collaborative filtering recommender system using Locality Sensitive Hashing (LSH) to approximate cosine similarity.

It downloads the specified dataset if it is not already present, processes the data to create item vectors, applies LSH for approximate nearest neighbor search,
and evaluates the recommender system using RMSE and MAE metrics. The results are saved to a specified output directory.

This script requires PySpark and pandas to be installed in your Python environment.

Usage:
    spark-submit cosine_lsh_submit.py -d small -b 1.5 -ht 8 -st 1.0 -o ./output/

Parameters:
    -d, --dataset: The dataset to use (default: small). Available options: small, ml-1m, ml-10m, ml-20m, ml-25m.
    -b, --bucket_length: The bucket length for LSH (default: 1.5).
    -ht, --num_hash_tables: The number of hash tables for LSH (default: 8).
    -st, --similarity_threshold: The similarity threshold for LSH (default: 1.0).  
    -o, --output_dir: The directory to save the output results (default: ./output/).

Created by: Cristiano Nicolau
"""    

import argparse
import os
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, struct, sum as sql_sum, udf, lit
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.ml.feature import Normalizer, BucketedRandomProjectionLSH
from pyspark.ml.evaluation import RegressionEvaluator

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

class CosineLSHRecommender:
    def __init__(self, dataset, bucket_length, num_hash_tables, similarity_threshold, output_dir):
        self.dataset = dataset
        self.bucket_length = bucket_length
        self.num_hash_tables = num_hash_tables
        self.similarity_threshold = similarity_threshold
        self.output_dir = output_dir
        self.spark = None
        self.ratings = None
        self.test = None
        self.rmse = None
        self.mae = None

    def start_spark(self):
        self.spark = SparkSession.builder \
            .appName("ItemItemCF") \
            .config("spark.driver.memory", "16g") \
            .config("spark.executor.memory", "16g") \
            .config("spark.executor.cores", "4") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")

    @staticmethod
    def to_sparse_vector(user_ratings, size):
        sorted_pairs = sorted(user_ratings, key=lambda x: x.userId)
        indices = [x.userId - 1 for x in sorted_pairs]
        values = [float(x.rating) for x in sorted_pairs]
        return Vectors.sparse(size, indices, values)
    
    def load_data(self, dataset_path):
        data = self.spark.read.csv(dataset_path, header=True, inferSchema=True) \
            .select("userId", "movieId", "rating")
        self.ratings, self.test = data.randomSplit([0.9, 0.1], seed=42)
        self.ratings.cache()
        self.test.cache()

    def fit_predict(self):
        spark = self.spark
        item_user = self.ratings.groupBy("movieId") \
            .agg(collect_list(struct("userId", "rating")).alias("user_ratings"))
        num_users = self.ratings.select("userId").distinct().count()
        item_vectors_rdd = item_user.rdd.map(
            lambda row: Row(
                movieId=row["movieId"],
                features=CosineLSHRecommender.to_sparse_vector(row["user_ratings"], num_users)
            )
        )
        item_vectors = spark.createDataFrame(item_vectors_rdd)
        normalizer = Normalizer(inputCol="features", outputCol="norm_features", p=2.0)
        normalized_item_vectors = normalizer.transform(item_vectors)
        lsh = BucketedRandomProjectionLSH(
            inputCol="norm_features",
            outputCol="hashes",
            bucketLength=self.bucket_length,
            numHashTables=self.num_hash_tables
        )
        lsh_model = lsh.fit(normalized_item_vectors)
        neighbors = lsh_model.approxSimilarityJoin(
            normalized_item_vectors,
            normalized_item_vectors,
            threshold=self.similarity_threshold,
            distCol="distance"
        ).filter(col("datasetA.movieId") != col("datasetB.movieId"))
        neighbors_cosine = neighbors.withColumn(
            "cosine_sim",
            1 - (col("distance") ** 2) / 2
        ).select(
            col("datasetA.movieId").alias("i_mv"),
            col("datasetB.movieId").alias("j_mv"),
            "cosine_sim"
        )
        similarities = neighbors_cosine.union(
            neighbors_cosine.selectExpr("j_mv as i_mv", "i_mv as j_mv", "cosine_sim")
        )
        similarities.cache()
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
        print(f"Running Item-Item CF with LSH using Cosine Similarity...")
        print(f"Using bucket length: {self.bucket_length}, number of hash tables: {self.num_hash_tables}, similarity threshold: {self.similarity_threshold}")
        results = self.fit_predict()
        print("Item-item collaborative filtering with LSH completed.")
        print(f"RMSE: {self.rmse}, MAE: {self.mae}")
        execution_time = round(time.time() - start_time, 2)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        summary = pd.DataFrame([{
            'Dataset': self.dataset,
            'Train Size': self.ratings.count(),
            'Test Size': self.test.count(),
            'Similarity': 'Cosine (approximated with LSH)',
            'Similarity Threshold': self.similarity_threshold,
            'Num Hash Tables': self.num_hash_tables,
            'Bucket Length': self.bucket_length,
            'RMSE': self.rmse,
            'MAE': self.mae,
            'Execution Time (s)': execution_time
        }])
        output_path = os.path.join(self.output_dir, 'item_item_cf_summary.csv')
        self.save_results(summary, output_path)
        print(f"Results saved to {output_path}")
        print(f"Execution time: {execution_time} seconds")
        self.spark.stop()

def main():
    parser = argparse.ArgumentParser(description="Item-Item CF with LSH (PySpark)")
    parser.add_argument('-d','--dataset', type=str, default='small', choices=DatasetManager.DATASET_PATHS.keys(), help='Dataset to use')
    parser.add_argument('-b','--bucket_length', type=float, default=1.5, help='LSH bucket length')
    parser.add_argument('-ht','--num_hash_tables', type=int, default=8, help='Number of LSH hash tables')
    parser.add_argument('-st','--similarity_threshold', type=float, default=1.0, help='LSH similarity threshold (Euclidean)')
    parser.add_argument('-o','--output_dir', type=str, default='./output/', help='Output directory')
    args = parser.parse_args()
    dataset_manager = DatasetManager(args.dataset)
    dataset_manager.ensure_dataset()
    recommender = CosineLSHRecommender(
        dataset=args.dataset,
        bucket_length=args.bucket_length,
        num_hash_tables=args.num_hash_tables,
        similarity_threshold=args.similarity_threshold,
        output_dir=args.output_dir
    )
    recommender.run(dataset_manager.dataset_path)

if __name__ == "__main__":
    main()
