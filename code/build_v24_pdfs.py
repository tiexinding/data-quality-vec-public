"""Build v2.4 CN + EN PDFs."""
import pypandoc, re, os
from pathlib import Path

WORK = Path("/home/dingdang-ws/wsl-projects/claudecode/NPM_v13_commit_260324/NPM_v13_complete/20_NPM/0-update/数据战略地图/s_bar_战略地图_工作区")

# Same emoji map as before (PDF-safe replacement)
EMOJI_MAP = {
    "✅": "[OK]", "🟢": "[A]", "🟡": "[B]", "🟠": "[C]", "🔴": "[!]",
    "⚠️": "(!) ", "⚠": "(!) ", "🔥": "[*]", "🆕": "[NEW]", "⭐": "*",
    "🎯": "Goal:", "📊": "Data:", "📋": "List:", "📦": "Pack:", "📄": "Doc:",
    "📁": "Dir:", "📈": "", "📉": "", "💥": "*KEY*:", "🧬": "",
    "🔧": "Fix:", "🚀": "[Run]", "💡": "Note:", "💬": "Quote:", "🙏": "",
    "🔬": "Test:", "🌅": "", "🌙": "", "🔄": "", "⏱️": "Time:",
    "⏳": "[wait]", "🌐": "", "💾": "", "🤖": "", "🟦": "[A]",
    "🆚": "vs", "🎨": "", "📌": "Note:",
    "①": "(1)", "②": "(2)", "③": "(3)", "④": "(4)", "⑤": "(5)",
    "⑥": "(6)", "⑦": "(7)", "⑧": "(8)", "⑨": "(9)", "⑩": "(10)",
    "️": "",
}

def sanitize(text):
    for k, v in EMOJI_MAP.items():
        text = text.replace(k, v)
    return text

def build_pdf(md_path, pdf_path):
    src = Path(md_path).read_text(encoding="utf-8")
    sanitized = sanitize(src)
    print_md = Path(md_path).parent / (Path(md_path).stem + ".__print__.md")
    print_md.write_text(sanitized, encoding="utf-8")
    args = [
        "--pdf-engine=xelatex",
        f"--resource-path={WORK}:{WORK}/ablation_results_20260424/figures:{WORK}/ablation_results_20260424",
        "-V", "geometry:margin=2.2cm",
        "-V", "colorlinks=true",
        "-V", "fontsize=10pt",
        "-V", "linestretch=1.2",
        "-V", "CJKmainfont=Noto Sans CJK SC",
        "-V", "mainfont=DejaVu Serif",
    ]
    print(f"\n=== building {pdf_path} ===")
    pypandoc.convert_file(str(print_md), 'pdf', outputfile=str(pdf_path), extra_args=args)
    sz = os.path.getsize(pdf_path)
    print(f"OK · {sz/1024:.0f} KB")

build_pdf(
    WORK / "阶梯1_技术报告_中文版_v2.4_sentence-transformers主体_20260425.md",
    WORK / "阶梯1_技术报告_中文版_v2.4_PDF_20260425.pdf",
)
build_pdf(
    WORK / "Stage1_Technical_Report_EN_v2.4_sentence-transformers_main_20260425.md",
    WORK / "Stage1_Technical_Report_EN_v2.4_PDF_20260425.pdf",
)
print("\n=== Done ===")
