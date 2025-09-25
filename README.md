# MatrixCore (v0.4.7)

[![CI](https://github.com/phongpingphongdgf/MatrixCore/actions/workflows/ci.yml/badge.svg)](https://github.com/phongpingphongdgf/MatrixCore/actions/workflows/ci.yml)

Прототип «ядра» потоковой упаковки/индексации (слова → предложения → сообщения) с зеркальными таблицами A/B, счётчиками и API на FastAPI.

## Быстрый старт
```bash
# 1) создать и активировать venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) зависимости
python -m pip install -U pip
pip install -r api/requirements.txt

# 3) тесты
pytest -q

# 4) сервер API (локально)
uvicorn api.dashboard:app --reload --host 127.0.0.1 --port 8000
