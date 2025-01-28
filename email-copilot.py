#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "litellm",
#   "python-dotenv",
#   "google-api-python-client",
#   "google-auth-httplib2",
#   "google-auth-oauthlib",
#   "google-auth",
# ]
# ///
import base64
import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.discovery import Resource
from litellm import completion

dotenv.load_dotenv()

# If modifying these SCOPES, delete the token file
# SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
SCOPES = ["https://mail.google.com/"]

CREDENTIALS_FILE = os.getenv("CREDENTIALS_FILE")
TOKEN_KEY_FILE = os.getenv("TOKEN_KEY_FILE")
FIRST_NAME = os.getenv("FIRST_NAME")
LAST_NAME = os.getenv("LAST_NAME")

FILTER_INSTRUCTION = f"""
Your task is to assist in managing the Gmail inbox of a busy individual by filtering out promotional emails from her personal (i.e., not work) account.
Your primary focus is to ensure that emails from individual people, whether they are known family members (with the same last name), close acquaintances, or potential contacts {FIRST_NAME} might be interested in hearing from, are not ignored.
You need to distinguish between promotional, automated, or mass-sent emails and personal communications.
Respond with "True" if the email is promotional and should be ignored based on the below criteria, or "False" otherwise.
Remember to prioritize personal communications and ensure emails from genuine individuals are not filtered out.
Criteria for Ignoring an Email:
 - The email is promotional: It contains offers, discounts, or is marketing a product or service.
 - The email is automated: It is sent by a system or service automatically, and not a real person.
 - The email appears to be mass-sent or from a non-essential mailing list: It does not address {FIRST_NAME} by name, lacks personal context that would indicate its personally written to her, or is from a mailing list that does not pertain to her interests or work.

Special Consideration:
 - Exception: If the email is from an actual person, especially a family member (with the same last name), a close acquaintance, or a potential contact {FIRST_NAME} might be interested in, and contains personalized information indicating a one-to-one communication, do not mark it for ignoring regardless of the promotional content.
 - Additionally, do not ignore emails requiring an action to be taken for important matters, such as needing to send a payment via Venmo, but ignore requests for non-essential actions like purchasing discounted items or signing up for rewards programs.

Be cautious: If theres any doubt about whether an email is promotional or personal, respond with "False".

The user message you will receive will have the following format:

Subject: <email subject>
To: <to names, to emails>
From: <from name, from email>
Cc: <cc names, cc emails>
Gmail labels: <labels>
Body: <plaintext body of the email>

Your response must be:  "True" or "False"
"""


def get_gmail_service():
    creds = None
    # The file stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(TOKEN_KEY_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_KEY_FILE, SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(TOKEN_KEY_FILE, "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def fetch_emails(
    gmail: Resource, page_token: Optional[str]
) -> Tuple[List[Dict[str, Union[str, List[str]]]], Optional[str]]:
    try:
        results = (
            gmail.users()
            .messages()
            .list(
                userId="me",
                labelIds=["UNREAD"],
                pageToken=page_token,  # Include the page token in the request if there is one
            )
            .execute()
        )
    except Exception as e:
        print(f"Failed to fetch emails: {e}")
        return [], None

    messages: List[Dict[str, Union[str, List[str]]]] = results.get("messages", [])
    page_token = results.get("nextPageToken")
    return messages, page_token


def parse_email_data(
    gmail: Resource, message_info: Dict[str, Union[str, List[str]]]
) -> Dict[str, Union[str, List[str]]]:
    # Fetch email data with 'full' format
    try:
        msg = (
            gmail.users()
            .messages()
            .get(userId="me", id=message_info["id"], format="full")
            .execute()
        )
    except Exception as e:
        print(f"Failed to fetch email data: {e}")
        return {}

    try:
        headers = msg["payload"]["headers"]
        subject = next(
            header["value"] for header in headers if header["name"] == "Subject"
        )
        to = next(header["value"] for header in headers if header["name"] == "To")
        sender = next(header["value"] for header in headers if header["name"] == "From")
        cc = next(
            (header["value"] for header in headers if header["name"] == "Cc"), None
        )
    except Exception as e:
        print(f"Failed to parse email data: {e}")
        return {}

    # Extract the plain text body
    parts = msg["payload"].get("parts", [])
    for part in parts:
        if part["mimeType"] == "text/plain":
            body = part["body"].get("data", "")
            body = base64.urlsafe_b64decode(body.encode("ASCII")).decode("utf-8")
            break
    else:
        body = ""

    # Parse email data
    email_data_parsed: Dict[str, Union[str, List[str]]] = {
        "subject": subject,
        "to": to,
        "from": sender,
        "cc": cc,
        "labels": msg["labelIds"],
        "body": body,
    }
    return email_data_parsed


def evaluate_email(
    email_data: Dict[str, Union[str, List[str]]],
) -> bool:
    MAX_EMAIL_LEN = 3000
    system_message: Dict[str, str] = {
        "role": "system",
        "content": FILTER_INSTRUCTION,
    }
    truncated_body = email_data["body"][:MAX_EMAIL_LEN] + (
        "..." if len(email_data["body"]) > MAX_EMAIL_LEN else ""
    )
    user_message: Dict[str, str] = {
        "role": "user",
        "content": (
            f"Subject: {email_data['subject']}\n"
            f"To: {email_data['to']}\n"
            f"From: {email_data['from']}\n"
            f"Cc: {email_data['cc']}\n"
            f"Gmail labels: {email_data['labels']}\n"
            f"Body: {truncated_body}"
        ),
    }

    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            messages = [system_message, user_message]
            response = completion(
                model="ollama_chat/llama3.2:latest",
                messages=messages,
                api_base="http://localhost:11434",
                temperature=0.0,
                max_tokens=1,
            )
            return response.choices[0].message.content.strip() == "True"
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                print(
                    f"Failed to evaluate email with GPT-4 after {max_retries} attempts: {e}"
                )
                return False
            print(f"Attempt {retry_count} failed, retrying...")
    return False


def mark_email_as_read(
    gmail: Resource, message_info: Dict[str, Union[str, List[str]]]
) -> int:
    try:
        gmail.users().messages().delete(userId="me", id=message_info["id"]).execute()
        print("✅ Email deleted successfully")
        return 1
    except Exception as e:
        print(f"❌ Failed to delete email: {e}")
        return 0


def process_email(
    gmail: Resource,
    message_info: Dict[str, Union[str, List[str]]],
    email_data_parsed: Dict[str, Union[str, List[str]]],
) -> int:
    email_subject = email_data_parsed.get("subject")
    email_sender = email_data_parsed.get("from")
    if evaluate_email(email_data_parsed):
        user_input = input(
            f"🗄️ Email with subject '{email_subject}' from '{email_sender}' is not worth the time. Do you want to delete this email? (Y/n): "
        ).lower()
        if not user_input or user_input == "y":
            return mark_email_as_read(gmail, message_info)
        else:
            print("Skipping this email")
            return 0
    else:
        print(
            f"👪 Email with subject '{email_subject}' from '{email_sender}' looks important ❗, leaving as unread"
        )
    return 0


def report_statistics(
    total_unread_emails: int, total_pages_fetched: int, total_marked_as_read: int
) -> None:
    print(f"Total number of unread emails fetched: {total_unread_emails}")
    print(f"Total number of pages fetched: {total_pages_fetched}")
    print(f"Total number of emails marked as read: {total_marked_as_read}")
    print(
        f"Final number of unread emails: {total_unread_emails - total_marked_as_read}"
    )


def main():
    gmail = get_gmail_service()

    page_token: Optional[str] = None

    total_unread_emails = 0
    total_pages_fetched = 0
    total_marked_as_read = 0

    while True:  # Continue looping until no more pages of messages
        # Fetch unread emails
        messages, page_token = fetch_emails(gmail, page_token)
        total_pages_fetched += 1
        print(f"Fetched page {total_pages_fetched} of emails")

        total_unread_emails += len(messages)
        for message_info in messages:
            # Fetch and parse email data
            email_data_parsed = parse_email_data(gmail, message_info)

            # Process email
            total_marked_as_read += process_email(
                gmail,
                message_info,
                email_data_parsed,
            )

        if not page_token:
            break  # Exit the loop if there are no more pages of messages

    report_statistics(total_unread_emails, total_pages_fetched, total_marked_as_read)


if __name__ == "__main__":
    main()
