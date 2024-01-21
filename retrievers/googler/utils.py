
import os.path
import shutil
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.errors import HttpError
from env import (SCOPES, FOLDER_ID)


class GoogleDrive:

    # If modifying these scopes, delete the file token.json.
    def __init__(self, SCOPES = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/drive.file',
        'https://www.googleapis.com/auth/drive.appdata'
        ]
        ):
        self.utils = GoogleUtils()
        self.SCOPES = SCOPES
        self.creds = self.establish_creds()
        self.endpoint = self.connect_to_endpoint()
        self.DOWNLOAD_PATH = "drive"
        self.token_path = "data/credentials/token.json"

    def connect_to_endpoint(self):
        try: 
            self.endpoint = build('drive', 'v3', credentials=self.creds)
            return self.endpoint
        except Exception as e:
            print(e)

    def establish_creds(self):
        creds = None
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        self.makedirs()
        if os.path.exists("data/credentials/token.json"):
            creds = Credentials.from_authorized_user_file("data/credentials/token.json", self.SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file("data/credentials/credentials.json", self.SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open("data/credentials/token.json", "w") as token:
                token.write(creds.to_json())
        return creds

    def fetch_docs(self):
        try:

            # Call the Drive v3 API
            results = (
                self.endpoint.files()
                .list(pageSize=10, fields="nextPageToken, files(id, name, mimeType)")
                .execute()
            )
            items = results.get("files", [])

            if not items:
                print("No files found.")
            return items
        except HttpError as error:
            # TODO(developer) - Handle errors from drive API.
            print(f"An error occurred: {error}")

    def get_doc(self, fileID, MIME='text/plain', encoding="utf-8"):
        try:
            results = self.endpoint.files().export(fileId=fileID, mimeType=MIME).execute()
            if results:
                 return results.decode(encoding)
            return results
        except HttpError as error:
            # TODO(developer) - Handle errors from drive API.
            print(f"An error occurred: {error}")

    def close(self):
        os.remove("data/credentials/token.json")

    def makedirs(self):
        if not os.path.exists("data/credentials/"):
            os.makedirs("data/credentials") 

    def download_file(self, file_id, file_name, DOWNLOAD_PATH):
        request = self.endpoint.files().get_media(fileId=file_id)

        if not os.path.exists(DOWNLOAD_PATH):
            os.makedirs(DOWNLOAD_PATH)

        with open(os.path.join(DOWNLOAD_PATH, file_name), 'wb+') as file:
            downloader = request.execute()
            file.write(downloader)


class Remove:

    def __init__(self):
        self.exclude = ['application/vnd.google-apps.folder',
                        'application/vnd.google.colaboratory']

    def folder(self, folder):
        shutil.rmtree(folder, ignore_errors=True)

    def loaded_documents(self, app_docs, all_docs):
        already_loaded = [d.metadata['title'] for d in app_docs]
        return [doc for doc in all_docs if doc["name"] not in already_loaded]
    
    def mimetypes(self, documents, mimeTypes=None):
        mimeTypes = self.exclude if mimeTypes is None else mimeTypes
        removed = []
        for mimeType in mimeTypes:
            for document in documents:
                if document["mimeType"] == mimeType:
                    popped = documents.pop(documents.index(document))
                    removed.append(popped)
        return documents
    

from datetime import datetime
import subprocess

class GoogleUtils:

    def __init__(self, LOCATION=None, PROJECT_ID=None):
        self.set_project_id(PROJECT_ID=PROJECT_ID)
        self.set_location(LOCATION=LOCATION)
        self.UID = datetime.now().strftime("%m%d%H%M")

    def set_project_id(self, PROJECT_ID=None):
        if PROJECT_ID is None:
            PROJECT_ID = self.get_gcloud_project()
            PROJECT_ID = PROJECT_ID[0]
            if PROJECT_ID == "(unset)":
                print(f"Please set the project ID manually below")

            # define project information
            if PROJECT_ID == "(unset)":
                PROJECT_ID = "[your-project-id]"  # @param {type:"string"}
        self.PROJECT_ID = PROJECT_ID

    def set_location(self, LOCATION=None):
        self.LOCATION = "us-central1" if LOCATION is None else LOCATION

    @staticmethod
    def get_gcloud_project():
        try:
            # Run the gcloud command and capture its output
            result = subprocess.run(['gcloud', 'config', 'get-value', 'project'], stdout=subprocess.PIPE, text=True, check=True)
            
            # Extract and return the project ID
            project_id = result.stdout.strip()
            return project_id
        except subprocess.CalledProcessError as e:
            # Handle errors, if any
            print(f"Error: {e}")
            return None