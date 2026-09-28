#!/usr/bin/env bash
# reports/REPORT.md -> reports/REPORT.pdf (pandoc + headless Chrome). Relative links to animations, results and the README
# become links to the GitHub repository so they work from the PDF.
set -euo pipefail
cd "$(dirname "$0")/../reports"
REPO=${REPO:-https://github.com/CyberChickZ/meposer-repro/blob/main}
CHROME=${CHROME:-"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}
TMP=$(mktemp -d)
python3 - "$REPO" > "$TMP/report.md" <<'PY'
import re, sys
repo = sys.argv[1]
s = open("REPORT.md").read()
s = re.sub(r"(?<!!)\[([^\]]*)\]\((?!http)([^)]+)\)",
           lambda m: "[" + m.group(1) + "](" + repo + "/" + ("README.md" if m.group(2) == "../README.md" else "reports/" + m.group(2)) + ")", s)
print(s)
PY
pandoc "$TMP/report.md" -f gfm -t html5 --standalone --metadata title="MEPoser reproduction" --css report.css -o .report_build.html
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --allow-file-access-from-files --print-to-pdf="$PWD/REPORT.pdf" "file://$PWD/.report_build.html" 2>/dev/null
rm -rf "$TMP" .report_build.html
echo "wrote reports/REPORT.pdf"
