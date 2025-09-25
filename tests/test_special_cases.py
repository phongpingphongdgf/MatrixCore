from fastapi.testclient import TestClient
from api.dashboard import app, CORE

def pack_iv(iv: tuple[int, int]) -> int:
    return (iv[0] << 32) | (iv[1] & 0xFFFFFFFF)

def test_word_len1_roundtrip():
    CORE.reset()
    iv = CORE.encode_word_text("A")
    out = CORE.decode_word_iv(iv)
    assert out["text"] == "A"

def test_sentence_one_word():
    CORE.reset()
    iv = CORE.encode_sentence_text("Hello")
    out = CORE.decode_sentence_iv(iv)
    assert out["text"] == "Hello"

def test_message_one_letter():
    CORE.reset()
    iv = CORE.encode_message("x")
    out = CORE.decode_message_iv(iv)
    assert out["text"].strip() == "x"

def test_decode_token_endpoints():
    CORE.reset()
    # Готовим IV'ы напрямую и проверяем API-варианты с token=<u64>
    w_iv = CORE.encode_word_text("Z")
    s_iv = CORE.encode_sentence_text("Z")     # предложение из 1 слова
    m_iv = CORE.encode_message("Z")           # сообщение из 1 буквы/слова/предложения

    client = TestClient(app)
    # word_token
    r = client.get("/api/decode/word_token", params={"token": pack_iv(w_iv)})
    assert r.status_code == 200 and r.json()["text"] == "Z"
    # sentence_token
    r = client.get("/api/decode/sentence_token", params={"token": pack_iv(s_iv)})
    assert r.status_code == 200 and r.json()["text"].strip() == "Z"
    # message_token
    r = client.get("/api/decode/message_token", params={"token": pack_iv(m_iv)})
    assert r.status_code == 200 and "Z" in r.json()["text"]
