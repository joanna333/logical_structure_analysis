import os
import pandas as pd
import csv

# Define the folder containing the CSV files
folder_path = 'RelationShip_Question_Finetuning'

# List to hold all the dataframes
dfs = []

# Loop through the files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a CSV
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)

        # Read the CSV file with error handling, ensuring all rows and columns are appended
        try:
            # Read the CSV file, skip malformed lines, and take the first row as header
            df = pd.read_csv(file_path, on_bad_lines='skip', header=0)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Check if there are any dataframes to combine
if dfs:
    # Combine all the dataframes into one, filling missing columns with NaN
    combined_df = pd.concat(dfs, ignore_index=True, sort=False)

    # Remove rows where all values are NaN
    combined_df.dropna(how='all', inplace=True)

    # Remove empty columns (those with all NaN values)
    combined_df.dropna(axis=1, how='all', inplace=True)

    # Save the combined dataframe to a new CSV file
    combined_df.to_csv('combined_output.csv', index=False, quoting=csv.QUOTE_MINIMAL)

    print("CSV files have been combined successfully!")
else:
    print("No valid CSV files found in the folder.")
