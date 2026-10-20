"""Windows desktop entry point."""

from __future__ import annotations

from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.bridge import DesktopBridge
else:
    from .bridge import DesktopBridge


def load_inline_html(index: Path) -> str:
    """Inline Vite's local assets so WebView2 needs no localhost HTTP server."""
    document = index.read_text(encoding="utf-8")
    script_match = re.search(r'<script\b[^>]*\bsrc="([^"]+)"[^>]*></script>', document)
    style_match = re.search(r'<link\b[^>]*\bhref="([^"]+\.css)"[^>]*>', document)
    if not script_match or not style_match:
        raise ValueError("前端构建产物缺少脚本或样式")
    script = (index.parent / script_match.group(1)).resolve().read_text(encoding="utf-8")
    style = (index.parent / style_match.group(1)).resolve().read_text(encoding="utf-8")
    script = script.replace("</script", "<\\/script")
    document = document.replace(script_match.group(0), f'<script type="module">{script}</script>')
    document = document.replace(style_match.group(0), f"<style>{style}</style>")
    return document


def main() -> None:
    import webview

    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    index = root / "dist" / "index.html"
    if not index.is_file():
        raise SystemExit("前端资源不存在。请先在 desktop 目录运行 npm run build。")
    html = load_inline_html(index)

    bridge = DesktopBridge()
    window = webview.create_window(
        "水印消除",
        html=html,
        js_api=bridge,
        width=1160,
        height=800,
        min_size=(820, 620),
        background_color="#f7f8fa",
    )
    bridge._window = window
    window.events.closed += bridge._dispose

    profile_root = bridge._data_root / "WebView2"
    profile_root.mkdir(parents=True, exist_ok=True)
    webview.start(
        gui="edgechromium",
        debug=False,
        private_mode=True,
        storage_path=str(profile_root),
        icon=str(root / "assets" / "app_icon.ico"),
    )


if __name__ == "__main__":
    main()
