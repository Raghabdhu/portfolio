import pyarrow.parquet as pq

file_path = "c:\\Users\\dell0\\Documents\\GitHub\\Data-Analytics-group-project\\NYCdata\\sampled_taxi_weather_data.parquet"
parquet_file = pq.ParquetFile(file_path)

# Print metadata
print(parquet_file.metadata)

# Print schema
print(parquet_file.schema)

# Print number of rows
print(parquet_file.metadata.num_rows)