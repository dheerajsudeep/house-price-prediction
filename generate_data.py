import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

area = np.random.randint(500, 5000, n)
bedrooms = np.random.randint(1, 6, n)
bathrooms = np.random.randint(1, 4, n)
stories = np.random.randint(1, 4, n)
parking = np.random.randint(0, 4, n)

price = (
    area * 250
    + bedrooms * 50000
    + bathrooms * 30000
    + stories * 20000
    + parking * 15000
    + np.random.randint(-50000, 50000, n)
)

df = pd.DataFrame({
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'parking': parking,
    'price': price
})

df.to_csv('house_data.csv', index=False)
print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())
