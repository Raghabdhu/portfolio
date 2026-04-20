import pandas as pd
import glob
import pyarrow.parquet as pq
import os
from datetime import datetime

# Define the base directory path dynamically (relative path)
base_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(base_folder, "NYCdata")

# Get a list of all Parquet files, excluding 'combined_sample_data.parquet'
parquet_files = [f for f in glob.glob(os.path.join(data_folder, "*.parquet")) if "combined_sample_data.parquet" not in f]

print(f"Total files found: {len(parquet_files)}")
if not parquet_files:
    raise FileNotFoundError("No Parquet files found. Please check the directory path.")

# Load and clean the weather data
csv_file_path = os.path.join(base_folder, "NYC_Weather_2016_2022.csv")
cleaned_csv_file_path = os.path.join(base_folder, "NYC_Weather_2016_2022_cleaned.csv")

print(f"\nProcessing CSV file: {csv_file_path}")
df_weather = pd.read_csv(csv_file_path)

# Print column names to inspect
print("Weather data columns:", df_weather.columns)

# Drop rows where any column is missing (check all columns for NA values)
df_weather.dropna(inplace=True)

# Remove duplicate rows
df_weather.drop_duplicates(inplace=True)

# Convert the 'time' column to datetime format and extract the date
df_weather['date'] = pd.to_datetime(df_weather['time']).dt.date  # Use 'time' column for weather data

# Ensure unique dates in weather data
df_weather.drop_duplicates(subset=['date'], inplace=True)

# Save cleaned CSV file (overwrite if exists)
if os.path.exists(cleaned_csv_file_path):
    os.remove(cleaned_csv_file_path)
    print("Existing cleaned CSV file removed.")
df_weather.to_csv(cleaned_csv_file_path, index=False)
print("Cleaned CSV file saved.")

# Initialize an empty list to store sampled and merged dataframes
sampled_dfs = []

# Process each Parquet file incrementally
for file in parquet_files:
    try:
        print(f"\nProcessing file: {file}")
        df = pd.read_parquet(file, engine='pyarrow')  # Use PyArrow for efficiency

        # Check original row count
        original_rows = len(df)
        
        # Handle rows with 'hvfhs_license_num' as 'HV0005'
        mask = df['hvfhs_license_num'] == 'HV0005'
        df.loc[mask, 'originating_base_num'] = df.loc[mask, 'dispatching_base_num']
        df.loc[mask, 'on_scene_datetime'] = df.loc[mask, 'request_datetime']

        # Define essential columns (all except 'airport_fee' and 'wav_match_flag')
        essential_columns = [col for col in df.columns if col not in ['airport_fee', 'wav_match_flag']]
        
        # Drop rows where essential columns are missing
        df = df.dropna(subset=essential_columns)

        # Remove duplicate rows
        df = df.drop_duplicates()

        cleaned_rows = len(df)
        print(f"Original rows: {original_rows}, After dropping NA, duplicates, and handling HV0005: {cleaned_rows}")

        # If dataset is too small, do not include it
        if cleaned_rows < 100:
            print(f"Skipping file {file} - Too few valid rows after cleaning.")
            continue

        # Convert the 'request_datetime' column to a date for merging
        df['request_date'] = pd.to_datetime(df['request_datetime']).dt.date  # Use 'request_datetime' for taxi data

        # Merge with weather data on the date
        merged_df = pd.merge(df, df_weather, left_on='request_date', right_on='date', how='inner')

        # Drop the extra date columns used for merging
        merged_df.drop(columns=['request_date', 'date'], inplace=True)

        # Debug: Print rows before and after merge
        print(f"Rows before merge: {len(df)}")
        print(f"Rows after merge: {len(merged_df)}")

        # Take a random sample of 200,000 rows from the merged dataset
        sample_size = min(200000, len(merged_df))
        sampled_df = merged_df.sample(n=sample_size, random_state=42)  # Random sampling

        print(f"Sampled {len(sampled_df)} rows from merged data for file: {file}")

        # Append the sampled dataframe to the list
        sampled_dfs.append(sampled_df)

    except MemoryError as e:
        print(f"Memory error while loading {file}: {e}")

# Ensure dataframes were loaded
if not sampled_dfs:
    raise RuntimeError("No files could be loaded due to memory issues.")

# Concatenate all sampled dataframes
final_sampled_df = pd.concat(sampled_dfs, ignore_index=True)

print(f"\nFinal sampled dataset has {final_sampled_df.shape[0]} rows and {final_sampled_df.shape[1]} columns.")

# Define output file path
output_file = os.path.join(data_folder, "sampled_taxi_weather_data.parquet")

# Remove existing file if it exists
if os.path.exists(output_file):
    os.remove(output_file)
    print("Existing 'sampled_taxi_weather_data.parquet' file removed.")

# Save the final sampled file using PyArrow
final_sampled_df.to_parquet(output_file, index=False, engine='pyarrow')
print("Final sampled dataset saved as 'sampled_taxi_weather_data.parquet'.")