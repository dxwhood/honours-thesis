import pandas as pd

# Load the datasets
df_random = pd.read_csv('taxi_random_dataset.csv')
df_expert = pd.read_csv('taxi_q_expert_dataset.csv')
df_medium = pd.read_csv('taxi_q_medium_dataset.csv')

# Concatenate datasets
df_mixed = pd.concat([df_random, df_medium, df_expert], ignore_index=True)

# Save the mixed dataset
df_mixed.to_csv('taxi_mixed_dataset.csv', index=False)
