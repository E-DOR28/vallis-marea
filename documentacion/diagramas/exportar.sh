#!/bin/sh
# Exporta cada .mmd de esta carpeta a SVG y PNG.
#
# Requiere mermaid-cli (comando `mmdc`) y un Chrome o Chromium instalado.
#   npm install @mermaid-js/mermaid-cli       (con PUPPETEER_SKIP_DOWNLOAD=1 si ya hay Chrome)
#
# Variables opcionales:
#   MMDC         ruta de mmdc            (por defecto, el que esté en PATH)
#   CHROME_PATH  ruta del ejecutable de Chrome o Chromium
set -eu

CARPETA="$(cd "$(dirname "$0")" && pwd)"
MMDC="${MMDC:-mmdc}"
SALIDA="${CARPETA}/exportados"
mkdir -p "$SALIDA"

CONFIG="$(mktemp)"
trap 'rm -f "$CONFIG"' EXIT
if [ -n "${CHROME_PATH:-}" ]; then
    printf '{"executablePath": "%s", "args": ["--no-sandbox"]}\n' "$CHROME_PATH" > "$CONFIG"
else
    printf '{"args": ["--no-sandbox"]}\n' > "$CONFIG"
fi

for fuente in "$CARPETA"/*.mmd; do
    base="$(basename "$fuente" .mmd)"
    "$MMDC" -q -i "$fuente" -o "$SALIDA/$base.svg" -p "$CONFIG" -b white
    "$MMDC" -q -i "$fuente" -o "$SALIDA/$base.png" -p "$CONFIG" -b white -s 2
    echo "ok  $base"
done
