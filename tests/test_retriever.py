import unittest
from unittest.mock import patch
from toddbo.retriever import retrieve_resume_records

class TestRetriever(unittest.TestCase):

    @patch('toddbo.retriever.load_pinecone')
    def test_retrieve_resume_records(self, mock_load_pinecone):
        # Mock the load_pinecone function
        mock_load_pinecone.return_value = "mock_retriever"

        # Test the retrieve_resume_records function
        retriever = retrieve_resume_records()
        self.assertEqual(retriever, "mock_retriever")
        mock_load_pinecone.assert_called_once()

if __name__ == '__main__':
    unittest.main()
