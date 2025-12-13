from fastapi import HTTPException
from dotenv import load_dotenv
import os
import boto3

# Load AWS credentials from the .env file
load_dotenv(".env")

AWS_ACCESS_KEY = os.getenv("ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")


def upload_html_to_s3(html_content: str, html_filename: str) -> str:
    """
    Uploads the HTML content directly to AWS S3 without saving to disk.
    Returns the S3 file URL.
    """

    print("ACCESS_KEY_ID:", repr(AWS_ACCESS_KEY))
    print("SECRET_ACCESS_KEY:", "SET" if AWS_SECRET_KEY else "MISSING")
    print("REGION:", AWS_REGION)
    print("BUCKET:", BUCKET_NAME)


    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )

        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=html_filename,
            Body=html_content.encode("utf-8"),
            ContentType="text/html; charset=utf-8",
            # ACL="public-read"  # keep commented unless intentionally public
        )

        file_url = (
            f"https://{BUCKET_NAME}.s3."
            f"{AWS_REGION}.amazonaws.com/{html_filename}"
        )

        return file_url

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload HTML to S3: {str(e)}"
        )
