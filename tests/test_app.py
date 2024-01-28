import unittest
from unittest.mock import patch
import app

class TestApp(unittest.TestCase):

    def test_page_title(self):
        with patch('streamlit.set_page_config') as mock_set_page_config:
            app.main()
            mock_set_page_config.assert_called_once_with(page_title="Todd's ResumeBot", layout="wide")

    def test_header_text(self):
        with patch('streamlit.title') as mock_title, \
             patch('streamlit.header') as mock_header:
            app.main()
            mock_title.assert_called_once_with("Todd's Resume Bot")
            mock_header.assert_called_once_with("Ask my resume questions!")

    def test_sidebar(self):
        with patch('streamlit.sidebar.subheader') as mock_subheader:
            app.main()
            mock_subheader.assert_called_once_with("Coming Soon!")

if __name__ == '__main__':
    unittest.main()