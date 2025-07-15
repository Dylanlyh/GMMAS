import pandas as pd

# Read the CSV file
df = pd.read_csv('/public/home/hpc226511030/GMMAS/12.11_pq_UPS_test_data_0.05_0.9.csv')

# Modify the last column for rows with confidence < 0.95
df.loc[df['confidence'] < 0.99, df.columns[-1]] = -1
df.loc[df['uncertainty'] > 0.01, df.columns[-1]] = -1

# Save the modified DataFrame back to the CSV file
df.to_csv('/public/home/hpc226511030/GMMAS/12.12_pq_UPS_test_data_0.01_0.99.csv', index=False)
