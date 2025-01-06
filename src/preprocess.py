# Import necessary libraries
import pandas as pd

# Load the dataset
file_path = r"C:\Users\anil_\Documents\GitHub\SteamSalesPrediction\data\games.csv"  # Adjust the path to your manually cleaned file
games_df = pd.read_csv(file_path, header=0)  # Ensure header is correctly read

# Drop unnecessary columns
columns_to_drop = [
    'Header image',
    'Website',
    'Support url',
    'Support email',
    'Movies',
    'Notes',
    'Full audio languages',
    'Metacritic score',
    'Recommendations',
    'Metacritic url',
    'DiscountDLC count 1',
    'User score',
    'Score rank',
    'Reviews'
]
games_df = games_df.drop(columns=[col for col in columns_to_drop if col in games_df.columns], errors='ignore')

# Drop the last column (z. column)
if games_df.columns[-1]:
    games_df = games_df.iloc[:, :-1]

# Drop the G-th column (7th column in this case, zero-indexed)
g_column_index = 6  # Adjust for zero-based indexing
if len(games_df.columns) > g_column_index:
    games_df = games_df.drop(games_df.columns[g_column_index], axis=1)

# Fill missing values
# Categorical columns: Fill with 'Unknown'
categorical_columns = games_df.select_dtypes(include=['object']).columns
games_df[categorical_columns] = games_df[categorical_columns].fillna('Unknown')

# Numerical columns: Fill with median
numerical_columns = games_df.select_dtypes(include=['number']).columns
games_df[numerical_columns] = games_df[numerical_columns].fillna(games_df[numerical_columns].median())

# Save the dataset after filling missing values
output_file_path = r"C:\Users\anil_\Documents\GitHub\SteamSalesPrediction\data\games_cleaned.csv"
games_df.to_csv(output_file_path, index=False)

print("Missing values filled and dataset saved to", output_file_path)
