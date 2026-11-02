import pytest

from src.auth import service as auth
from src.database.connection import session_scope
from src.database import operations as ops


def register(engine, username="alice_01", email="alice@example.com", password="Passw0rdA"):
    with session_scope(engine) as s:
        return auth.register_user(s, username, email, password, full_name="Alice")


def test_register_creates_user_with_normalized_fields(engine):
    user = register(engine, username="  Alice_01 ", email="Alice@Example.com")
    assert user.id is not None
    assert user.username == "alice_01"
    assert user.email == "alice@example.com"


def test_password_is_hashed_not_plaintext(engine):
    register(engine, password="Passw0rdA")
    with session_scope(engine) as s:
        user = ops.get_user_by_username(s, "alice_01")
        assert user.password_hash != "Passw0rdA"
        assert user.password_hash.startswith("$2")  # bcrypt format
        assert "Passw0rdA" not in user.password_hash


def test_same_password_gets_different_hashes():
    assert auth.hash_password("Passw0rdA") != auth.hash_password("Passw0rdA")


def test_verify_password():
    h = auth.hash_password("Passw0rdA")
    assert auth.verify_password("Passw0rdA", h) is True
    assert auth.verify_password("wrong-pass1", h) is False


def test_verify_password_with_malformed_hash_returns_false():
    assert auth.verify_password("Passw0rdA", "not-a-hash") is False


def test_duplicate_username_rejected(engine):
    register(engine)
    with pytest.raises(auth.UserExistsError):
        register(engine, email="other@example.com")


def test_duplicate_email_rejected_case_insensitive(engine):
    register(engine)
    with pytest.raises(auth.UserExistsError):
        register(engine, username="bob_02", email="ALICE@example.com")


@pytest.mark.parametrize("bad", ["ab", "has space", "bad-name!", "x" * 31, ""])
def test_invalid_usernames_rejected(engine, bad):
    with pytest.raises(auth.ValidationError):
        register(engine, username=bad)


@pytest.mark.parametrize("bad", ["", "no-at-sign", "a@b", "a b@c.com"])
def test_invalid_emails_rejected(engine, bad):
    with pytest.raises(auth.ValidationError):
        register(engine, email=bad)


@pytest.mark.parametrize("bad", ["short1", "allletters", "12345678", "a1" * 40])
def test_weak_or_overlong_passwords_rejected(engine, bad):
    with pytest.raises(auth.ValidationError):
        register(engine, password=bad)


def test_login_by_username_and_email(engine):
    register(engine)
    with session_scope(engine) as s:
        assert auth.authenticate(s, "alice_01", "Passw0rdA").username == "alice_01"
        assert auth.authenticate(s, "ALICE@example.com", "Passw0rdA").username == "alice_01"


def test_login_wrong_password(engine):
    register(engine)
    with session_scope(engine) as s:
        with pytest.raises(auth.AuthenticationError):
            auth.authenticate(s, "alice_01", "WrongPass1")


def test_login_unknown_user_gives_same_message(engine):
    register(engine)
    with session_scope(engine) as s:
        with pytest.raises(auth.AuthenticationError) as unknown:
            auth.authenticate(s, "nobody", "Passw0rdA")
        with pytest.raises(auth.AuthenticationError) as wrong:
            auth.authenticate(s, "alice_01", "WrongPass1")
    assert str(unknown.value) == str(wrong.value)


def test_login_requires_both_fields(engine):
    with session_scope(engine) as s:
        with pytest.raises(auth.ValidationError):
            auth.authenticate(s, "", "")