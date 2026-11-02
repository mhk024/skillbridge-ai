"""Registration, login and password handling."""

import re

import bcrypt
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.database import operations as ops
from src.database.models import User

BCRYPT_ROUNDS = 12  # tests lower this for speed
MAX_PASSWORD_BYTES = 72  # bcrypt limit
MIN_PASSWORD_LENGTH = 8

_USERNAME_RE = re.compile(r"^[a-z0-9_]{3,30}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_dummy_hash: str | None = None


class AuthError(Exception):
    """Base class. Messages are safe to show to the user."""


class ValidationError(AuthError):
    pass


class UserExistsError(AuthError):
    pass


class AuthenticationError(AuthError):
    pass


def validate_username(username: str) -> str:
    cleaned = (username or "").strip().lower()
    if not _USERNAME_RE.match(cleaned):
        raise ValidationError(
            "Username must be 3-30 characters: letters, numbers and underscores only."
        )
    return cleaned


def validate_email(email: str) -> str:
    cleaned = (email or "").strip().lower()
    if len(cleaned) > 254 or not _EMAIL_RE.match(cleaned):
        raise ValidationError("Please enter a valid email address.")
    return cleaned


def validate_password(password: str) -> None:
    password = password or ""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValidationError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise ValidationError(f"Password is too long (maximum {MAX_PASSWORD_BYTES} bytes).")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        raise ValidationError("Password must contain at least one letter and one number.")


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except (ValueError, TypeError):
        return False  # malformed hash or over-long password


def register_user(
    session: Session,
    username: str,
    email: str,
    password: str,
    full_name: str | None = None,
) -> User:
    username = validate_username(username)
    email = validate_email(email)
    validate_password(password)
    full_name = (full_name or "").strip()[:100] or None

    if ops.get_user_by_username(session, username):
        raise UserExistsError("That username is already taken.")
    if ops.get_user_by_email(session, email):
        raise UserExistsError("An account with that email already exists.")

    try:
        return ops.create_user(
            session,
            username=username,
            email=email,
            password_hash=hash_password(password),
            full_name=full_name,
        )
    except IntegrityError as exc:  # two sign-ups racing each other
        raise UserExistsError("That username or email is already registered.") from exc


def authenticate(session: Session, identifier: str, password: str) -> User:
    """Log in with username or email. Raises AuthenticationError on any failure."""
    global _dummy_hash
    identifier = (identifier or "").strip().lower()
    if not identifier or not password:
        raise ValidationError("Enter your username (or email) and password.")

    if "@" in identifier:
        user = ops.get_user_by_email(session, identifier)
    else:
        user = ops.get_user_by_username(session, identifier)

    if user is None:
        if _dummy_hash is None:
            _dummy_hash = hash_password("dummy-password-1")
        verify_password(password, _dummy_hash)  # equalize timing
        raise AuthenticationError("Invalid username or password.")

    if not verify_password(password, user.password_hash):
        raise AuthenticationError("Invalid username or password.")
    return user