import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from acweather import read_dataset, clean_data, train_model, calculate_median

class TestWeatherPrediction(unittest.TestCase):

    def setUp(self):
        # Create a mock dataset for testing
        self.data = {
            'temperature': np.random.uniform(15, 35, 100),
            'humidity': np.random.uniform(40, 90, 100),
            'wind_speed': np.random.uniform(0, 20, 100),
            'precipitation': np.random.randint(0, 2, 100)
        }
        self.df = pd.DataFrame(self.data)

        # Save the DataFrame to a CSV file for testing
        self.test_file = 'test_weather_data.csv'
        self.df.to_csv(self.test_file, index=False)

    def test_read_dataset(self):
        result = read_dataset(self.test_file)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, self.df.shape)  # Check if the dimensions match

    def test_clean_data(self):
        # Test the data cleaning process
        cleaned_df = clean_data(self.df, ['temperature', 'humidity', 'wind_speed'], 'precipitation')
        self.assertIn('temperature', cleaned_df.columns)
        self.assertIn('humidity', cleaned_df.columns)
        self.assertNotIn('NaN', cleaned_df)  # Ensure no NaN values remain
        self.assertEqual(cleaned_df.isnull().sum().sum(), 0)  # Ensure no NaN values

    def test_train_model(self):
        # Test the training of the model
        cleaned_df = clean_data(self.df, ['temperature', 'humidity', 'wind_speed'], 'precipitation')
        model, features, target = train_model(cleaned_df)
        self.assertIsInstance(model, RandomForestRegressor)  # Check if the model is a RandomForestRegressor
        self.assertGreater(len(features), 0)  # Check if features were returned
        self.assertEqual(cleaned_df.iloc[:, -1].values.all(), target.values.all())  # Target should be the last column

# PLACEHOLDER: does_calculate_standard_deviation_work(self)

# PLACEHOLDER: does_calculate_range_work(self)

# PLACEHOLDER: does_calculate_median_work(self)


class TestCalculateMedian(unittest.TestCase):

    def test_ordered_median(self):
        self.assertEqual(calculate_median([1,2,3,4,7]), 3)
        
    def test_unordered_median(self):
            self.assertEqual(calculate_median([1, 4, 3, 5, 2]), 3)

    def test_negative_median(self):
        self.assertEqual(calculate_median([-5, -3, -1, 0, 1]), -1)

    def test_float_median(self):
        self.assertEqual(calculate_median([-10.5, -5.5, 3.5, 4.5, 5.5]), 3.5)

    def test_negative_float_median(self):
        self.assertEqual(calculate_median([-3.1, -4.3, -6.5, -7.6, -10.3]), -6.5)

    def test_all_zero_median(self):
        self.assertEqual(calculate_median([0, 0, 0, 0, 0]), 0)

    def test_even_number_of_elements_median(self):
        self.assertEqual(calculate_median([1, 2, 3, 4]), 2.5)

    




# PLACEHOLDER: does_calculate_variance_work(self)

if __name__ == '__main__':
    unittest.main()
