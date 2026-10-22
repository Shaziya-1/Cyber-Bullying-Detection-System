import os
import sys

def get_spark_session(app_name="CivilCommentsBigDataAnalytics", memory="4g"):
    java_21_path = r"C:\Users\Admin\jdk-21.0.12.1+1"
    if os.path.exists(java_21_path):
        os.environ["JAVA_HOME"] = java_21_path
        bin_path = os.path.join(java_21_path, "bin")
        if bin_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")

    # Configure Hadoop Home for Windows Parquet/Hadoop I/O
    hadoop_path = r"C:\Users\Admin\hadoop"
    if os.path.exists(hadoop_path):
        os.environ["HADOOP_HOME"] = hadoop_path
        hadoop_bin = os.path.join(hadoop_path, "bin")
        if hadoop_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", memory)
        .config("spark.executor.memory", memory)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
