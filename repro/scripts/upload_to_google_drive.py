#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow


SCOPES = ["https://www.googleapis.com/auth/drive"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload reproducibility artifacts to Google Drive and print public links."
    )
    p.add_argument("--folder-id", required=True, help="Destination Google Drive folder id.")
    p.add_argument(
        "--mode",
        choices=["oauth", "service-account"],
        default="oauth",
        help="Auth mode for Google Drive API.",
    )
    p.add_argument(
        "--client-secrets",
        type=Path,
        default=Path("google_client_secret.json"),
        help="OAuth client secrets JSON (used in oauth mode).",
    )
    p.add_argument(
        "--service-account-json",
        type=Path,
        default=Path("google_service_account.json"),
        help="Service account JSON (used in service-account mode).",
    )
    p.add_argument(
        "--token-cache",
        type=Path,
        default=Path(".cache/google_drive_token.json"),
        help="Token cache JSON path for oauth mode.",
    )
    p.add_argument(
        "--make-public",
        action="store_true",
        help="Set uploaded files to public read access and print public links.",
    )
    p.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Files to upload.",
    )
    return p.parse_args()


def build_service_oauth(client_secrets: Path, token_cache: Path):
    token_cache.parent.mkdir(parents=True, exist_ok=True)
    if token_cache.exists():
        from google.oauth2.credentials import Credentials

        creds = Credentials.from_authorized_user_file(str(token_cache), SCOPES)
    else:
        if not client_secrets.exists():
            raise FileNotFoundError(f"Client secrets not found: {client_secrets}")
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES)
        creds = flow.run_local_server(port=0)
        token_cache.write_text(creds.to_json(), encoding="utf-8")
    return build("drive", "v3", credentials=creds)


def build_service_account(sa_json: Path):
    if not sa_json.exists():
        raise FileNotFoundError(f"Service account json not found: {sa_json}")
    creds = service_account.Credentials.from_service_account_file(str(sa_json), scopes=SCOPES)
    return build("drive", "v3", credentials=creds)


def upload_file(service, folder_id: str, local_path: Path, make_public: bool) -> dict:
    if not local_path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")

    media = MediaFileUpload(str(local_path), resumable=True)
    meta = {"name": local_path.name, "parents": [folder_id]}

    created = (
        service.files()
        .create(body=meta, media_body=media, fields="id,name,size,webViewLink")
        .execute()
    )

    if make_public:
        service.permissions().create(
            fileId=created["id"],
            body={"type": "anyone", "role": "reader"},
        ).execute()
        created["public_link"] = f"https://drive.google.com/file/d/{created['id']}/view?usp=sharing"
    return created


def print_manifest(rows: Iterable[dict]) -> None:
    rows = list(rows)
    print("\nUpload result:\n")
    for r in rows:
        size_mb = float(r.get("size", 0)) / (1024 * 1024)
        link = r.get("public_link") or r.get("webViewLink", "")
        print(f"- {r['name']} | {size_mb:.1f} MB")
        print(f"  {link}")

    manifest = {"files": rows}
    Path("repro/upload_manifest_google_drive.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nManifest saved to repro/upload_manifest_google_drive.json")


def main() -> int:
    args = parse_args()
    files = [Path(f).expanduser().resolve() for f in args.files]

    if args.mode == "oauth":
        service = build_service_oauth(args.client_secrets, args.token_cache)
    else:
        service = build_service_account(args.service_account_json)

    uploaded = []
    for f in files:
        print(f"Uploading: {f}")
        uploaded.append(upload_file(service, args.folder_id, f, args.make_public))

    print_manifest(uploaded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

