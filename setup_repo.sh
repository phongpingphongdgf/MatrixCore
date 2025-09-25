#!/usr/bin/env bash
set -euo pipefail

# setup_repo.sh — инициализация структуры MatrixCore (v0.4.2)

ROOT="$(pwd)"
echo "Working dir: $ROOT"

# --- 1) Папки ---
mkdir -p api
mkdir -p excel
mkdir -p tests
mkdir -p .github/workflows

# --- 2) README ---
cat > README.md <<'EOF'
# MatrixCore (v0.4.2)

Прототип "ядра" для потоковой упаковки/индексации текста (слова → предложения → сообщения) с зеркальными таблицами A/B, телеметрией и API на FastAPI.

## Структура
