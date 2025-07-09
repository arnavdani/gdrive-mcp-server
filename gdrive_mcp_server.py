import os.path
import base64
import requests
import io
import sys

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from pdfminer.high_level import extract_text
from google import genai
from google.genai import types
from mcp.server.fastmcp import FastMCP


## -- setup server -- ##
mcp = FastMCP("gdrive")
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
CREDENTIALS_FILE = "./credentials.json"
TOKEN_FILE = "./token.json"
MAX_CONTENT_BYTES = 300_000
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


## -- auth -- ##
def connect_to_drive():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    try:
        service = build('drive', 'v3', credentials=creds)
        return service
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None

def summarize_with_gemini(text_content: str) -> str:
    """Sends text to Gemini Flash for summarization."""
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=text_content),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config = types.ThinkingConfig(
            thinking_budget=-1,
        ),
        response_mime_type="text/plain",
    )
    chunks = []
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        chunks.append(chunk.text)

    return "".join(chunks)



## MCP proper ##


@mcp.tool()
def list_files(max_results: int = 50) -> str:
    """
    List the names, IDs, and types in the user's Google Drive.

    Args:
        max_results (int): Maximum number of files to list.

    Returns:
        str: formatted string containing the list of files.
    """
    service = connect_to_drive()
    if not service:
        return "Authentication failed"

    try:
        results = (
            service.files().list(
                pageSize=max_results,
                fields="nextPageToken, files(id, name, mimeType)"
            ).execute()
        )
        items = results.get("files", [])
        if not items:
            return "No files found."
        output = "ID\tName\tType\n"
        for item in items:
            item_type = "Folder" if item["mimeType"] == "application/vnd.google-apps.folder" else "File"
            output += f"{item['id']}\t{item['name']}\t{item_type}\n"
        return output

    except HttpError as error:
        return f"An error occurred: {error}"


# @mcp.tool()
# def get_file_content(file_id: str) -> str:
#     """
#     Retrieves and returns file contents from a specific file.
#     For google docs, converted to plain text

#     For all other formats, converted to Base64 encoded URI
#     """
#     service = connect_to_drive()
#     if not service:
#         return "Authentication failed"

#     try:
#         file = service.files().get(fileId=file_id, fields="mimeType").execute()
#         file_type = file.get('mimeType')
#         request = None

#         if file_type == "application/vnd.google-apps.document":
#             request = service.files().export(fileId=file_id, mimeType="text/plain")
#             file_bytes = request.execute()
#             if len(file_bytes) > MAX_CONTENT_BYTES:
#                 return file_bytes[:MAX_CONTENT_BYTES].decode('utf-8', errors='ignore')
#             return file_bytes.decode('utf-8')
#         else:
#             file_bytes = base64.b64encode(service.files().get_media(fileId=file_id).execute())
#             if len(file_bytes) > MAX_CONTENT_BYTES:
#                 file_bytes = file_bytes[:MAX_CONTENT_BYTES]
#             data = file_bytes.decode('utf-8', errors="ignore")
#             return f"data:{file_type};base64,{data}"
#     except HttpError as error:
#         return f"An error occurred: {error}"
#     except Exception as e:
#         return f"An error occurred: {e}"

@mcp.tool()
def search_files(query: str, max_results: int = 10) -> str:
    """
    Searches for files in Google Drive matching the query in their name or content.
    Returns up to 10 most relevant files with their names and IDs.
    """
    service = connect_to_drive()
    if not service:
        return "Error: Could not connect to Google Drive."

    try:
        # Escape single quotes in the query to prevent API errors
        escaped_query = query.replace("'", "\\'")

        # Construct the search query string for the API
        search_query = f"(name contains '{escaped_query}' or fullText contains '{escaped_query}') and trashed = false"

        results = (
            service.files()
            .list(
                q=search_query,
                pageSize=max_results,
                fields="nextPageToken, files(id, name, mimeType)",
            )
            .execute()
        )
        items = results.get("files", [])

        if not items:
            return f"No files found matching your search for '{query}'."

        output = f"Found {len(items)} files matching your search:\n"
        for item in items:
            item_type = "Folder" if item["mimeType"] == "application/vnd.google-apps.folder" else "File"
            output += f"- Name: {item['name']}, ID: {item['id']}, Type: {item_type}\n"
        return output

    except HttpError as error:
        return f"An error occurred during search: {error}"

@mcp.tool()
def summarize_pdf(file_id: str, prompt: str = "Summarize this PDF in 200 words using bullet points") -> str:
    """
    Downloads a PDF from Google Drive, extracts its text, and returns a summary from Gemini.

    Use the prompt parameter to add specific details you would want.
    """
    service = connect_to_drive()
    if not service:
        return "Error: Could not connect to Google Drive."

    try:
        # Verify it's a PDF
        file_metadata = service.files().get(fileId=file_id, fields='mimeType').execute()
        if file_metadata.get('mimeType') != 'application/pdf':
            return "Error: This tool only works with PDF files."

        # Download the PDF content
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()

        fh.seek(0)

        # Extract text from the PDF bytes
        extracted_text = extract_text(fh)

        if not extracted_text.strip():
            return "Could not extract any text from the PDF. It might be an image-based PDF."

        # Summarize the extracted text with Gemini
        summary = summarize_with_gemini(extracted_text)

        return summary

    except HttpError as error:
        return f"An error occurred with the Google Drive API: {error}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"



if __name__ == "__main__":
    mcp.run(transport="stdio")
