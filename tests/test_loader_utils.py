import unittest
from toddbo import loader_utils

class TestLoaderUtils(unittest.TestCase):

    def test_load_data(self):
        # Test the load_data function
        data = loader_utils.load_data()
        self.assertIsNotNone(data)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_preprocess_data(self):
        # Test the preprocess_data function
        raw_data = ["Hello, world!", "This is a test."]
        processed_data = loader_utils.preprocess_data(raw_data)
        self.assertIsNotNone(processed_data)
        self.assertIsInstance(processed_data, list)
        self.assertEqual(len(processed_data), len(raw_data))
        self.assertEqual(processed_data[0], "hello world")
        self.assertEqual(processed_data[1], "this is a test")

if __name__ == '__main__':
    unittest.main()