import os
from dotenv import load_dotenv

load_dotenv()

DOCUMENT_ID=os.environ.get("DOCUMENT_ID")
FOLDER_ID= os.environ.get("FOLDER_ID") 
GOOGLE_API_KEY= os.environ.get("GOOGLE_API_KEY") 
GOOGLE_APPLICATION_CREDENTIALS= os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_ACCOUNT_FILE= os.environ.get("GOOGLE_ACCOUNT_FILE")
SCOPES=["https://www.googleapis.com/auth/drive.metadata.readonly",'https://www.googleapis.com/auth/drive','https://www.googleapis.com/auth/drive.file','https://www.googleapis.com/auth/drive.appdata']
