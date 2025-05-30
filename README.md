# Item-Item Collaborative Filtering Recommender System (PySpark)

This project implements scalable item-item collaborative filtering algorithms for movie recommendation using PySpark and the MovieLens datasets. Two main approaches are provided:

- **Clustering-based Collaborative Filtering**: Uses KMeans to cluster items and computes cosine similarities within clusters for efficient neighbor search.
- **LSH-based Collaborative Filtering**: Uses Locality Sensitive Hashing (LSH) to approximate cosine similarity and efficiently find similar items in large datasets.

## Features
- **Automatic Dataset Download**: Downloads and prepares MovieLens datasets (small, 1M, 10M, 20M, 25M) as needed.
- **Efficient Vectorization**: Represents each item as a sparse vector of user ratings for scalable similarity computation.
- **Clustering (KMeans)**: Groups items to restrict similarity search to within clusters, improving speed and scalability.
- **LSH (BucketedRandomProjectionLSH)**: Approximates cosine similarity using LSH for fast neighbor search in high-dimensional spaces.
- **Evaluation Metrics**: Computes RMSE and MAE on a held-out test set.
- **Results Logging**: Saves experiment results (dataset, parameters, metrics, execution time) to CSV for easy comparison.

## Requirements
- Python 3.7+
- PySpark
- pandas

## Usage

### 1. Clustering-based Collaborative Filtering
Run with:
```sh
spark-submit clustering_submit.py -d small -o ./output/ -k 20
```
- `-d`: Dataset to use (`small`, `ml-1m`, `ml-10m`, `ml-20m`, `ml-25m`)
- `-o`: Output directory (default: `./output/`)
- `-k`: Number of clusters for KMeans (default: sqrt(num_items))

### 2. LSH-based Collaborative Filtering
Run with:
```sh
spark-submit cosine_lsh_submit.py -d small -b 1.5 -ht 8 -st 1.0 -o ./output/
```
- `-d`: Dataset to use
- `-b`: LSH bucket length (default: 1.5)
- `-ht`: Number of LSH hash tables (default: 8)
- `-st`: Similarity threshold (default: 1.0)
- `-o`: Output directory

### Output

Results are saved in `output/item_item_cf_summary.csv` after each experiment. This CSV file provides a comprehensive summary of all runs, making it easy to compare different algorithms, datasets, and parameter settings.

Each row in the CSV contains the following columns (depending on the method used):

- **Dataset**: The MovieLens dataset used (e.g., `small`, `ml-1m`).
- **Train Size**: Number of ratings in the training set.
- **Test Size**: Number of ratings in the test set.
- **Similarity**: The similarity metric or method used (e.g., `Cosine (approximated with LSH)`, `Cosine (within cluster)`).
- **Similarity Threshold**: (LSH only) The Euclidean distance threshold for considering items as neighbors.
- **Num Hash Tables**: (LSH only) Number of hash tables used in LSH.
- **Bucket Length**: (LSH only) The bucket length parameter for LSH.
- **Clustering**: (Clustering only) KMeans configuration, e.g., `KMeans (k=96)`.
- **RMSE**: Root Mean Squared Error of the predictions on the test set.
- **MAE**: Mean Absolute Error of the predictions on the test set.
- **Execution Time (s)**: Total execution time for the experiment (in seconds).


The notebook `data/to_csv.ipynb` is provided to help convert MovieLens `.dat` rating files (such as `ratings.dat`) into standard CSV format (`ratings.csv`). This is necessary because some MovieLens datasets are distributed in a double-colon-separated format, which is not directly compatible with pandas or PySpark. The notebook reads each `.dat` file, parses it, and writes a corresponding `.csv` file in the same folder. This ensures all scripts in this project can seamlessly load the ratings data for any supported dataset.

## Notebooks
- `cosine_clustering.ipynb`: Step-by-step clustering-based approach with explanations and code.
- `cosine_lsh.ipynb`: Step-by-step LSH-based approach with explanations and code.

## References
- [MovieLens Datasets](https://grouplens.org/datasets/movielens/)
- B. Sarwar et al., "Item-based Collaborative Filtering Recommendation Algorithms"
- George Karypis, "Evaluation of Item-Based Top-N Recommendation Algorithms"

## Author
Cristiano Nicolau

