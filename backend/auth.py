"""
auth.py — Google OAuth + Email OTP + Gmail SMTP
"""

import os
import json
import smtplib
import asyncio
import random
import urllib.parse
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse, HTMLResponse
from pydantic import BaseModel
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

router = APIRouter()

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GMAIL_ADDRESS        = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD   = os.getenv("GMAIL_APP_PASSWORD", "")
SECRET_KEY           = os.getenv("SECRET_KEY", "dev-secret")
FRONTEND_URL         = os.getenv("FRONTEND_URL", "http://localhost:3000")
BACKEND_URL          = os.getenv("BACKEND_URL", "http://localhost:8000")

REDIRECT_URI = BACKEND_URL + "/auth/google/callback"
TOKEN_MAX_AGE = 60 * 60 * 24 * 7
OTP_EXPIRY_MIN = 10

serializer = URLSafeTimedSerializer(SECRET_KEY)
_otp_store: dict = {}


def create_token(user: dict) -> str:
    return serializer.dumps(user, salt="bms-auth")

def decode_token(token: str) -> Optional[dict]:
    try:
        return serializer.loads(token, salt="bms-auth", max_age=TOKEN_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None

def get_token_from_request(request: Request) -> Optional[str]:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


@router.get("/auth/debug")
async def debug():
    return {
        "GOOGLE_CLIENT_ID_set":     bool(GOOGLE_CLIENT_ID),
        "GOOGLE_CLIENT_SECRET_set": bool(GOOGLE_CLIENT_SECRET),
        "GMAIL_ADDRESS_set":        bool(GMAIL_ADDRESS),
        "REDIRECT_URI":             REDIRECT_URI,
        "FRONTEND_URL":             FRONTEND_URL,
        "client_id_preview":        GOOGLE_CLIENT_ID[:20] + "..." if GOOGLE_CLIENT_ID else "NOT SET",
    }


@router.get("/auth/google")
async def google_login():
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(500, "GOOGLE_CLIENT_ID not set in .env")

    params = urllib.parse.urlencode({
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  REDIRECT_URI,
        "response_type": "code",
        "scope":         "openid email profile",
        "access_type":   "offline",
        "prompt":        "select_account",
    })
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + params
    print(f"[GOOGLE] Redirecting to: {url[:80]}...")
    return RedirectResponse(url)


@router.get("/auth/google/callback")
async def google_callback(request: Request):
    """Handle Google OAuth callback."""
    params = dict(request.query_params)
    print(f"[CALLBACK] params received: {list(params.keys())}")

    error = params.get("error", "")
    code  = params.get("code", "")

    if error:
        print(f"[CALLBACK] Google error: {error}")
        return RedirectResponse(f"{FRONTEND_URL}?auth_error={urllib.parse.quote(error)}")

    if not code:
        print("[CALLBACK] No code in params")
        return RedirectResponse(f"{FRONTEND_URL}?auth_error=no_code")

    # Exchange code for token
    print(f"[CALLBACK] Exchanging code, REDIRECT_URI={REDIRECT_URI}")
    async with httpx.AsyncClient(timeout=20) as client:
        tok_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code":          code,
                "client_id":     GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri":  REDIRECT_URI,
                "grant_type":    "authorization_code",
            }
        )

    print(f"[CALLBACK] Token exchange status: {tok_resp.status_code}")
    if tok_resp.status_code != 200:
        err_body = tok_resp.text[:500]
        print(f"[CALLBACK] Token error: {err_body}")
        return RedirectResponse(
            f"{FRONTEND_URL}?auth_error=token_failed&msg={urllib.parse.quote(err_body[:100])}"
        )

    tok_data = tok_resp.json()
    access_token = tok_data.get("access_token", "")
    print(f"[CALLBACK] Got access_token: {bool(access_token)}")

    # Fetch user profile
    async with httpx.AsyncClient(timeout=10) as client:
        prof_resp = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"}
        )

    print(f"[CALLBACK] Profile status: {prof_resp.status_code}")
    if prof_resp.status_code != 200:
        return RedirectResponse(f"{FRONTEND_URL}?auth_error=profile_failed")

    g = prof_resp.json()
    print(f"[CALLBACK] Logged in: {g.get('email')}")

    user = {
        "id":         g.get("id", ""),
        "email":      g.get("email", ""),
        "name":       g.get("name", ""),
        "picture":    g.get("picture", ""),
        "given_name": g.get("given_name", ""),
        "login_at":   datetime.utcnow().isoformat(),
        "provider":   "google",
    }
    token = create_token(user)

    # Redirect directly to frontend with token + user in URL params
    # React reads from URL on load — avoids all localStorage timing issues
    encoded_token = urllib.parse.quote(token, safe='')
    encoded_user  = urllib.parse.quote(json.dumps(user), safe='')
    redirect_url  = (
        FRONTEND_URL
        + "?auth=google"
        + "&token=" + encoded_token
        + "&user="  + encoded_user
    )
    print(f"[CALLBACK] Redirecting to {FRONTEND_URL}?auth=google&token=...")
    return RedirectResponse(redirect_url)


@router.get("/auth/me")
async def get_me(request: Request):
    token = get_token_from_request(request)
    if not token:
        raise HTTPException(401, "No token")
    user = decode_token(token)
    if not user:
        raise HTTPException(401, "Invalid or expired token")
    return user


@router.post("/auth/logout")
async def logout():
    return {"status": "logged out"}


# ── OTP ───────────────────────────────────────────────────────────

class OTPRequest(BaseModel):
    email: str
    name:  str = ""

class OTPVerify(BaseModel):
    email: str
    otp:   str


def _smtp_send(to: str, subject: str, html: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"BMS ML Pipeline <{GMAIL_ADDRESS}>"
    msg["To"]      = to
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        s.sendmail(GMAIL_ADDRESS, to, msg.as_string())


@router.post("/auth/otp/send")
async def send_otp(req: OTPRequest):
    email = req.email.strip().lower()
    if "@" not in email:
        raise HTTPException(400, "Invalid email address.")
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        raise HTTPException(500, "Gmail SMTP not configured in .env")

    otp = str(random.randint(100000, 999999))
    _otp_store[email] = {
        "otp":     otp,
        "expires": datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MIN),
        "name":    req.name.strip(),
    }

    greeting = f"Hi {req.name.strip()}," if req.name.strip() else "Hi,"
    html = (
        "<html><body style='font-family:Arial;max-width:480px;margin:0 auto'>"
        "<div style='background:#cc0000;padding:20px;border-radius:8px 8px 0 0'>"
        "<h2 style='color:#fff;margin:0'>BMS ML Pipeline — Login Code</h2>"
        "</div>"
        "<div style='background:#fff;padding:28px;border:1px solid #e0e0e0;border-radius:0 0 8px 8px'>"
        f"<p>{greeting}</p>"
        "<p>Your one-time login code:</p>"
        "<div style='text-align:center;margin:24px 0'>"
        f"<span style='font-size:42px;font-weight:800;letter-spacing:14px;color:#cc0000;"
        "font-family:Courier New;background:#fff5f5;padding:16px 28px;"
        f"border-radius:10px;border:2px solid rgba(204,0,0,0.2)'>{otp}</span>"
        "</div>"
        f"<p style='color:#666;font-size:13px'>Valid for <strong>{OTP_EXPIRY_MIN} minutes</strong>.</p>"
        "<p style='color:#999;font-size:11px'>BMS Hackathon · Team Data Minds</p>"
        "</div></body></html>"
    )

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _smtp_send, email,
                                   f"BMS Pipeline login code: {otp}", html)
    except Exception as e:
        _otp_store.pop(email, None)
        raise HTTPException(500, f"Email failed: {e}")

    return {"status": "sent", "message": f"Code sent to {email}. Check inbox and spam."}


@router.post("/auth/otp/verify")
async def verify_otp(req: OTPVerify):
    email = req.email.strip().lower()
    entry = _otp_store.get(email)
    if not entry:
        raise HTTPException(400, "No code found. Request a new one.")
    if datetime.utcnow() > entry["expires"]:
        _otp_store.pop(email, None)
        raise HTTPException(400, "Code expired. Request a new one.")
    if req.otp.strip() != entry["otp"]:
        raise HTTPException(400, "Incorrect code.")

    _otp_store.pop(email, None)
    name = entry.get("name") or email.split("@")[0].replace(".", " ").title()
    user = {
        "id":         f"otp_{email}",
        "email":      email,
        "name":       name,
        "given_name": name.split()[0] if name else "",
        "picture":    None,
        "login_at":   datetime.utcnow().isoformat(),
        "provider":   "email_otp",
    }
    token = create_token(user)
    return {"token": token, "user": user}


# ── Support ───────────────────────────────────────────────────────

class SupportRequest(BaseModel):
    name:     str
    email:    str
    subject:  str
    message:  str
    category: str = "general"


@router.post("/auth/support")
async def send_support(req: SupportRequest, request: Request):
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        raise HTTPException(500, "Gmail SMTP not configured in .env")

    token  = get_token_from_request(request)
    user   = decode_token(token) if token else None
    sender = user["email"] if user else req.email

    html = (
        "<html><body style='font-family:Arial;max-width:600px;margin:0 auto'>"
        "<div style='background:#cc0000;padding:20px;border-radius:8px 8px 0 0'>"
        "<h2 style='color:#fff;margin:0'>BMS Support Request</h2>"
        "</div>"
        "<div style='background:#f9f9f9;padding:24px;border:1px solid #e0e0e0;border-radius:0 0 8px 8px'>"
        f"<p><b>From:</b> {req.name} &lt;{sender}&gt;</p>"
        f"<p><b>Category:</b> {req.category}</p>"
        f"<p><b>Subject:</b> {req.subject}</p>"
        "<hr>"
        f"<div style='white-space:pre-wrap'>{req.message}</div>"
        "</div></body></html>"
    )

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _smtp_send, GMAIL_ADDRESS,
                                   f"[BMS Support] {req.subject}", html)
        return {"status": "sent"}
    except Exception as e:
        raise HTTPException(500, f"Email failed: {e}")