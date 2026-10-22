# Big Data Performance Benchmark: Civil Comments Dataset

Empirical benchmarks measured on **Apache PySpark 4.2.0** with **Java 21 Runtime** processing the 1.8M-row Civil Comments dataset (`civil_comments_train.csv`, ~816 MB).

## Benchmark Results Table

| Operation | Input Records | Output Records | Execution Time |
| :--- | :--- | :--- | :--- |
| **CSV Ingestion (1.8M rows)** | 1,804,874 | 1,804,874 | **18.167 s** |
| **Data Quality & Cleaning** | 1,804,874 | 1,804,874 | **104.589 s** |
| **Transformations & Feature Engineering** | 1,804,874 | 1,804,874 | **51.998 s** |
| **Spark SQL Analytics (7 Queries)** | 1,804,874 | 1,804,874 | **10.960 s** |
| **Spark Window Functions (Rankings & Cum. Sums)** | 1,804,874 | 20 | **21.774 s** |
| **Parquet Write (Snappy, Partitioned by Year)** | 1,804,874 | 1,804,874 | **24.854 s** |
| **Parquet Read & Verification** | 1,804,874 | 1,804,874 | **1.015 s** |
| **Total Pipeline Runtime** | 1,804,874 | 1,804,874 | **248.370 s** |

## Storage & I/O Comparison

| Format | Path | Size | Compression | Read Time |
| :--- | :--- | :--- | :--- | :--- |
| **CSV (Source)** | `civil_comments_train.csv` | **778.40 MB** | None | 18.167 s |
| **Parquet (Processed)** | `data/parquet/civil_comments/` | **41.83 MB** | Snappy (Partitioned by Year) | **1.015 s** |

### Key I/O Takeaways
- **Storage Savings**: Parquet reduced disk footprint by **94.6%** (778.40 MB down to 41.83 MB).
- **Read Speedup**: Querying Parquet was **17.90x faster** than reading the uncompressed CSV.
- **Partitioning Strategy**: Partitioning by `year` ensured efficient column pruning and prevented excessive small file fragmentation.

---
*All timings represent actual measured execution times on this machine.*
