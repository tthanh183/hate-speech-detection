import pandas as pd

# Load the CSV file
df = pd.read_csv('classificated_data.csv')  # Replace with the actual path to your CSV file

# Convert 0 and 1 to 1, and 2 to 0
df['class'] = df['class'].apply(lambda x: 1 if x in [0, 1] else 0)

# Save the modified DataFrame to a new CSV file
df.to_csv('modified_file.csv', index=False)
