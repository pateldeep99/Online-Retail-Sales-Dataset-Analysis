import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('online_retail_sales_dataset.csv')

# Randomly sample 20,000 rows
df_sample = df.sample(n=40856, random_state=42)
# Randomly introduce NaN values in 1,000 random rows
num_nulls = 4000  # Number of rows to introduce NaNs in
for col in df_sample.columns:
    # Randomly select 1,000 indices in the column and set them to NaN
    null_indices = np.random.choice(df_sample.index, num_nulls, replace=False)
    df_sample.loc[null_indices, col] = np.nan

if 'customer_gender' in df_sample.columns:
        other_indices = df_sample[df_sample['customer_gender'] == 'Other'].index
        num_to_convert = int(0.3 * len(other_indices))  # 40% of "Other"

        # Randomly select 40% of "Other" indices
        convert_indices = np.random.choice(other_indices, num_to_convert, replace=False)

        # Convert selected values to "Female"
        df_sample.loc[convert_indices, 'customer_gender'] = 'Female'
if 'customer_gender' in df_sample.columns:
        other_indices = df_sample[df_sample['customer_gender'] == 'Male'].index
        num_to_convert = int(0.1 * len(other_indices))  # 40% of "Other"

        # Randomly select 40% of "Other" indices
        convert_indices = np.random.choice(other_indices, num_to_convert, replace=False)

        # Convert selected values to "Female"
        df_sample.loc[convert_indices, 'customer_gender']='Female'
# Save the modified data with NaNs to a new CSV file
df_sample.to_csv('sampled_online_retail_sales_dataset.csv', index=False)

print(f"Random null values have been added in {num_nulls} rows and saved to 'sampled_with_nulls_file.csv'.")
