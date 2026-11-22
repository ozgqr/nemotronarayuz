import streamlit as st
from openai import OpenAI
import json
import os
import uuid
import zipfile
import io
import re
import base64
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

# ==========================================
# OPSİYONEL KÜTÜPHANELER
# ==========================================
try:
    import tiktoken
    ENCODER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    ENCODER = None
    class FallbackEncoder:
        def encode(self, text): return text.split()
        def decode(self, tokens): return " ".join(tokens)
    ENCODER = FallbackEncoder()

try:
    import fitz; HAS_PYMUPDF = True
except ImportError: HAS_PYMUPDF = False

try:
    from docx import Document; HAS_DOCX = True
except ImportError: HAS_DOCX = False

try:
    from PIL import Image
    import pytesseract
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    HAS_OCR = True
except ImportError: HAS_OCR = False

try:
    import wikipedia; wikipedia.set_lang("tr"); HAS_WIKI = True
except ImportError: HAS_WIKI = False

try:
    from duckduckgo_search import DDGS; HAS_DDG = True
except ImportError: HAS_DDG = False

# ==========================================
# 1. KONFİGÜRASYON
# ==========================================
st.set_page_config(page_title="Nvidia Nemotron 3 Ultra", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

BACKGROUND_IMAGE_PATH = r"C:\Users\hindi\OneDrive\Masaüstü\ozgurfoto\canon\canoneos7d\09-2026\IMG_8170.JPG"
DEFAULT_API_KEY = "Enter API Key"

AVAILABLE_MODELS = ["nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-4-340b-instruct", "nvidia/nemotron-3.5-bolt-e4b"]
CHATS_DIR = Path("chats"); INDEX_FILE = CHATS_DIR / "index.json"
SUPPORTED_TEXT_EXT = {".txt", ".py", ".md", ".json", ".js", ".ts", ".html", ".css", ".yaml", ".yml", ".ini", ".cfg", ".log", ".csv", ".xml", ".sh", ".bat", ".ps1", ".dockerfile", ".gitignore", ".env", ".toml", ".rs", ".go", ".java", ".cpp", ".c", ".h", ".cs", ".php", ".rb", ".swift", ".kt", ".scala", ".r", ".m", ".pl", ".lua", ".vim", ".sql"}
SUPPORTED_BINARY_EXT = {".pdf", ".docx"}
SUPPORTED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}
SUPPORTED_ARCHIVE_EXT = {".zip"}
ALL_SUPPORTED_EXT = SUPPORTED_TEXT_EXT | SUPPORTED_BINARY_EXT | SUPPORTED_IMAGE_EXT | SUPPORTED_ARCHIVE_EXT

THINKING_PHRASES = ["🧠 Düşünüyorum...", "🔍 Bağlam analiz ediliyor...", "⚙️ Mantık kuruluyor...", "🎨 Yanıt tasarlanıyor...", "📚 Bilgi taranıyor...", "✨ Fikirler birleştiriliyor...", "🚀 Nemotron uyanıyor...", "🧩 Parçalar bir araya getiriliyor..."]

# ==========================================
# 2. VERİ YAPILARI
# ==========================================
@dataclass
class ChatMessage:
    role: str; content: str
    usage: Dict = field(default_factory=dict)
    files: List[Dict] = field(default_factory=list) # Mesaja eklenen dosyalar (geçici görsel dahil)
    project_files: Dict[str, str] = field(default_factory=dict)

@dataclass
class ChatSession:
    id: str; title: str; created_at: str; updated_at: str
    messages: List[ChatMessage] = field(default_factory=list)
    context_files: List[Dict] = field(default_factory=list) # Kalıcı bağlam dosyaları (PDF, ZIP, Kod vb.)

# ==========================================
# 3. STORAGE & YARDIMCI FONKSİYONLAR
# ==========================================
def ensure_storage():
    CHATS_DIR.mkdir(exist_ok=True)
    if not INDEX_FILE.exists(): INDEX_FILE.write_text("[]", encoding="utf-8")

def load_index() -> list:
    ensure_storage()
    try: return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except: return []

def save_index(index: list):
    try: INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    except: st.toast("İndeks kaydedilemedi!", icon="❌")

def load_chat(chat_id: str) -> ChatSession | None:
    file_path = CHATS_DIR / f"chat_{chat_id}.json"
    if file_path.exists():
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            msgs = [ChatMessage(**m) if isinstance(m, dict) else m for m in data.get("messages", [])]
            return ChatSession(id=data["id"], title=data["title"], created_at=data["created_at"], updated_at=data["updated_at"], messages=msgs, context_files=data.get("files", []))
        except Exception as e: st.error(f"Yükleme hatası: {e}")
    return None

def save_chat(chat: ChatSession):
    chat.updated_at = datetime.now().isoformat()
    data = {"id": chat.id, "title": chat.title, "created_at": chat.created_at, "updated_at": chat.updated_at, "messages": [m.__dict__ for m in chat.messages], "files": chat.context_files}
    try:
        (CHATS_DIR / f"chat_{chat.id}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        idx = load_index(); idx = [c for c in idx if c["id"] != chat.id]; idx.insert(0, {"id": chat.id, "title": chat.title, "updated_at": chat.updated_at}); save_index(idx)
    except Exception as e: st.toast(f"Kayıt hatası: {e}", icon="❌")

def delete_chat(chat_id: str):
    (CHATS_DIR / f"chat_{chat_id}.json").unlink(missing_ok=True)
    save_index([c for c in load_index() if c["id"] != chat_id])

def create_new_chat() -> ChatSession:
    cid = uuid.uuid4().hex[:12]; now = datetime.now().isoformat()
    chat = ChatSession(id=cid, title="Yeni Sohbet", created_at=now, updated_at=now)
    save_chat(chat); return chat

def count_tokens(text: str) -> int:
    try: return len(ENCODER.encode(text))
    except: return len(text) // 3

# ==========================================
# 4. DOSYA İŞLEME MOTORU
# ==========================================
def extract_text_from_bytes(file_name: str, file_bytes: bytes) -> Tuple[str, str]:
    ext = Path(file_name).suffix.lower()
    if ext in SUPPORTED_ARCHIVE_EXT:
        full_text = ""
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as zf:
                for name in zf.namelist():
                    if name.startswith("__MACOSX") or name.endswith("/"): continue
                    with zf.open(name) as f:
                        inner_bytes = f.read()
                        if len(inner_bytes) > 2_000_000: continue
                        inner_text, _ = extract_text_from_bytes(name, inner_bytes)
                        if inner_text.strip(): full_text += f"\n--- FILE: {name} ---\n{inner_text}\n"
            return full_text, "zip"
        except: return f"[ZIP Okunamadı]", "zip"
    if ext == ".pdf":
        if not HAS_PYMUPDF: return "[PyMuPDF yok: `pip install pymupdf`]", "pdf"
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf"); text = "\n".join([page.get_text() for page in doc]); doc.close()
            return text, "pdf"
        except Exception as e: return f"[PDF Hatası: {e}]", "pdf"
    if ext == ".docx":
        if not HAS_DOCX: return "[python-docx yok]", "docx"
        try:
            doc = Document(io.BytesIO(file_bytes)); return "\n".join([p.text for p in doc.paragraphs]), "docx"
        except Exception as e: return f"[DOCX Hatası: {e}]", "docx"
    if ext in SUPPORTED_IMAGE_EXT:
        if not HAS_OCR: return "[OCR için Pillow/pytesseract kurun + Tesseract sistemde olmalı]", "image"
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert('L')
            text = pytesseract.image_to_string(img, lang='tur+eng')
            return f"[Görsel OCR: {file_name}]\n{text}", "image_ocr"
        except Exception as e: return f"[OCR Hatası: {e}]", "image"
    if ext in SUPPORTED_TEXT_EXT or ext == "":
        for enc in ('utf-8', 'latin-1', 'cp1254', 'iso-8859-9'):
            try: return file_bytes.decode(enc), "text"
            except UnicodeDecodeError: continue
        return "[Binary/Dekod Edilemedi]", "binary"
    return f"[Desteklenmeyen: {ext}]", "unknown"

def process_uploaded_files(uploaded_files) -> List[Dict]:
    processed = []
    for uf in uploaded_files:
        if uf.size > 10 * 1024 * 1024: st.toast(f"⚠️ `{uf.name}` çok büyük (Max 10MB), atlandı.", icon="⚠️"); continue
        bytes_data = uf.getvalue()
        text, ftype = extract_text_from_bytes(uf.name, bytes_data)
        processed.append({"name": uf.name, "content": text, "type": ftype, "size": uf.size, "tokens": count_tokens(text), "raw_bytes": bytes_data, "mime": uf.type})
    return processed

# ==========================================
# 5. PROJE BİRLEŞTİRİCİ
# ==========================================
class ProjectMerger:
    HEADER_PATTERN = re.compile(r'^(?:#{1,3}|-{3,})\s*FILE\s*:\s*(.+?)\s*(?:#{1,3}|-{3,})?\s*$', re.MULTILINE | re.IGNORECASE)
    def __init__(self, raw_text: str): self.raw_text = raw_text; self.files: Dict[str, str] = {}; self._parse()
    def _parse(self):
        matches = list(self.HEADER_PATTERN.finditer(self.raw_text))
        if not matches: return
        for i, match in enumerate(matches):
            header_end = match.end(); next_start = matches[i + 1].start() if i + 1 < len(matches) else len(self.raw_text)
            raw_path = match.group(1).strip(); content = self.raw_text[header_end:next_start].strip()
            clean_path = self._sanitize_path(raw_path)
            if clean_path: self.files[clean_path] = content
    def _sanitize_path(self, raw: str) -> str:
        raw = raw.replace('\\', '/').strip(); parts = [p for p in raw.split('/') if p and p not in ('..', '.')]; return '/'.join(parts) if parts else ""
    def get_tree_md(self) -> str:
        if not self.files: return ""
        root = {}
        for path, content in self.files.items():
            node = root
            for i, part in enumerate(path.split('/')):
                if i == len(path.split('/')) - 1: node[part] = f"📄 {part} ({len(content)} chr)"
                else:
                    if part not in node: node[part] = {}
                    node = node[part]
        def build(d, pre=""):
            lines = []
            for i, (n, v) in enumerate(d.items()):
                is_last = i == len(d)-1; con = "└── " if is_last else "├── "; lines.append(f"{pre}{con}{n}")
                if isinstance(v, dict): ext = "    " if is_last else "│   "; lines.extend(build(v, pre+ext))
            return lines
        return "📁 **Proje Yapısı**\n```text\n" + "\n".join(build(root)) + "\n```"
    def create_zip(self) -> bytes:
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED) as zf:
            for p, c in self.files.items(): zf.writestr(p, c.encode('utf-8'))
        mem.seek(0); return mem.read()

# ==========================================
# 6. ARAŞTIRMA AJANI
# ==========================================
class ResearchAgent:
    @staticmethod
    def search_wiki(query: str, sentences=5) -> Dict:
        if not HAS_WIKI: return {"text": "Wikipedia kütüphanesi yok.", "url": "", "title": ""}
        try:
            results = wikipedia.search(query)
            if not results: return {"text": "Sonuç bulunamadı.", "url": "", "title": ""}
            page = wikipedia.page(results[0], auto_suggest=False)
            summary = wikipedia.summary(page.title, sentences=sentences)
            return {"text": f"**{page.title}**\n\n{summary}", "url": page.url, "title": page.title}
        except wikipedia.DisambiguationError as e: return {"text": f"Çoklu anlam: {', '.join(e.options[:5])}", "url": "", "title": "Disambiguation"}
        except Exception as e: return {"text": f"Hata: {e}", "url": "", "title": "Error"}
    @staticmethod
    def search_web(query: str, max_results=5) -> List[Dict]:
        if not HAS_DDG: return [{"text": "duckduckgo-search kütüphanesi yok", "url": "", "title": "Error"}]
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, region='tr-tr', safesearch='moderate', max_results=max_results))
                formatted = []
                for r in results:
                    if 'error' not in r: formatted.append({"text": f"**{r.get('title','')}**\n{r.get('body','')}", "url": r.get('href',''), "title": r.get('title','')})
                return formatted if formatted else [{"text": "Sonuç bulunamadı.", "url": "", "title": ""}]
        except Exception as e: return [{"text": f"Hata: {e}", "url": "", "title": "Error"}]

# ==========================================
# 7. UI STİL VE ARKA PLAN (GÜNCELLENDİ: Şeffaf Input + Copy Btn CSS)
# ==========================================
def get_base64_bg(image_path: str) -> str:
    try:
        with open(image_path, "rb") as f: return base64.b64encode(f.read()).decode()
    except: return ""

def inject_custom_css():
    bg_b64 = get_base64_bg(BACKGROUND_IMAGE_PATH)
    bg_css = f"background-image: url('data:image/jpeg;base64,{bg_b64}'); background-size: cover; background-position: center; background-attachment: fixed;" if bg_b64 else "background-color: #0e1117;"

    css = f"""
    <style>
    .stApp {{ {bg_css} font-family: 'Segoe UI', 'Inter', sans-serif; }}
    .main .block-container {{ background: rgba(15, 18, 25, 0.85); backdrop-filter: blur(12px); border-radius: 24px; padding: 2rem; margin-top: 1rem; margin-bottom: 1rem; border: 1px solid rgba(255,255,255,0.05); box-shadow: 0 8px 32px rgba(0,0,0,0.4); padding-bottom: 4rem; }} /* Alt boşluk input için */
    #MainMenu, header, footer {{ visibility: hidden; }} .stDeployButton {{ display: none; }}

    section[data-testid="stSidebar"] {{ background: rgba(20, 24, 35, 0.95) !important; backdrop-filter: blur(10px); border-right: 1px solid rgba(255,255,255,0.05); width: 320px !important; }}
    .sidebar-title {{ font-size: 1.3rem; font-weight: 700; color: #fff; padding: 1rem; border-bottom: 1px solid rgba(255,255,255,0.1); display: flex; align-items: center; gap: 0.5rem; }}
    .sidebar-section {{ padding: 0 1rem 1rem; }}

    .chat-item {{ display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; border-radius: 8px; margin-bottom: 0.3rem; transition: 0.2s; animation: fadeIn 0.2s ease; }}
    .chat-item:hover {{ background: rgba(255,255,255,0.05); }}
    .chat-title {{ font-size: 0.9rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #eee; display:flex; align-items:center; gap:0.5rem; }}
    .chat-date {{ font-size: 0.7rem; color: #888; margin-left: 0.5rem; }}
    .chat-del-btn {{ background: transparent; border: none; color: #666; cursor: pointer; padding: 0.2rem; opacity: 0; transition: 0.2s; }}
    .chat-item:hover .chat-del-btn {{ opacity: 1; }} .chat-del-btn:hover {{ color: #ff4444; }}

    .stChatMessage {{ background: transparent !important; border-radius: 16px; padding: 1rem 1.25rem; margin-bottom: 1rem; animation: fadeInUp 0.4s ease-out; border: 1px solid rgba(255,255,255,0.04); }}
    @keyframes fadeInUp {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    @keyframes fadeIn {{ from {{ opacity: 0; transform: translateX(-10px); }} to {{ opacity: 1; transform: translateX(0); }} }}

    [data-testid="stChatMessage"][aria-label="user"] {{ background: linear-gradient(135deg, rgba(76, 175, 80, 0.15), rgba(56, 142, 60, 0.1)) !important; border-color: rgba(76, 175, 80, 0.3); margin-left: 15%; }}
    [data-testid="stChatMessage"][aria-label="user"] .stMarkdown {{ color: #e8f5e9; }}
    [data-testid="stChatMessage"][aria-label="assistant"] {{ background: rgba(255,255,255,0.03) !important; border-color: rgba(255,255,255,0.05); margin-right: 15%; }}
    [data-testid="stChatMessageAvatar"] {{ background: linear-gradient(135deg, #4CAF50, #2E7D32) !important; border: 2px solid rgba(255,255,255,0.1); }}
    [data-testid="stChatMessageAvatar"][aria-label="assistant"] {{ background: linear-gradient(135deg, #1a1a2e, #16213e) !important; border-color: #4CAF50; }}

    .thinking-badge {{ display: inline-flex; align-items: center; gap: 0.5rem; background: rgba(76, 175, 80, 0.15); border: 1px solid rgba(76, 175, 80, 0.3); color: #a5d6a7; padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.5rem; animation: pulse 1.5s infinite; }}
    @keyframes pulse {{ 0% {{ box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.4); }} 70% {{ box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }} 100% {{ box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }} }}
    .thinking-spinner {{ width: 14px; height: 14px; border: 2px solid rgba(76, 175, 80, 0.3); border-top-color: #4CAF50; border-radius: 50%; animation: spin 0.8s linear infinite; }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

    .project-card {{ background: rgba(0,0,0,0.2); border: 1px solid rgba(76, 175, 80, 0.2); border-radius: 12px; padding: 1rem; margin-top: 1rem; animation: slideDown 0.3s ease; }}
    @keyframes slideDown {{ from {{ opacity: 0; max-height: 0; }} to {{ opacity: 1; max-height: 500px; }} }}
    .project-tree {{ font-family: 'Monospace', monospace; font-size: 0.85rem; background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; white-space: pre-wrap; color: #c8e6c9; }}

    .context-tag {{ display: inline-flex; align-items: center; gap: 0.4rem; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); padding: 0.3rem 0.7rem; border-radius: 20px; font-size: 0.8rem; color: #bbb; margin: 0.2rem; }}
    .context-tag.pending {{ border-color: #ffc107; background: rgba(255, 193, 7, 0.1); color: #ffc107; }}

    .token-bar {{ display: flex; gap: 1rem; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.05); font-size: 0.8rem; color: #888; }}
    .token-item {{ display: flex; align-items: center; gap: 0.4rem; background: rgba(255,255,255,0.03); padding: 0.3rem 0.7rem; border-radius: 8px; }}

    /* --- YENİ: Şeffaf Chat Input Barı --- */
    /* Container (Input'un çevresindeki wrapper) */
    div[data-testid="stChatInputContainer"] {{
        background: transparent !important; 
        border-top: none !important; 
        box-shadow: none !important;
        padding-top: 0.5rem;
    }}
    /* İçerdeki form/alan */
    div[data-testid="stChatInputContainer"] > div {{
        background: rgba(255, 255, 255, 0.02) !important; /* Çok hafif cam efektsiz */
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 14px !important;
        backdrop-filter: blur(5px);
    }}
    /* Textarea kendisi */
    textarea[data-testid="stChatInputTextArea"] {{
        background: transparent !important; 
        color: #fff !important; 
        border: none !important; 
        box-shadow: none !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
    }}
    /* Placeholder rengi */
    textarea[data-testid="stChatInputTextArea"]::placeholder {{ color: #888 !important; }}
    /* Gönder butonu alanı */
    div[data-testid="stChatInputContainer"] button[kind="secondary"] {{ background: transparent !important; color: #aaa !important; }}
    div[data-testid="stChatInputContainer"] button[kind="secondary"]:hover {{ color: #4CAF50 !important; }}

    /* --- YENİ: Kod Blokları Alt Kopyala Butonu (CSS Hazırlığı) --- */
    .stCodeBlock {{ position: relative !important; padding-bottom: 36px !important; /* Buton için yer ayır */ }}
    .custom-copy-btn {{
        position: absolute !important; bottom: 6px !important; left: 8px !important; z-index: 10 !important;
        background: rgba(255,255,255,0.1) !important; border: 1px solid rgba(255,255,255,0.2) !important;
        color: #ddd !important; padding: 4px 10px !important; border-radius: 6px !important;
        font-size: 0.75rem !important; cursor: pointer !important; opacity: 0.7 !important;
        transition: all 0.2s !important; display: flex !important; align-items: center !important; gap: 4px !important;
        font-family: 'Segoe UI', sans-serif !important;
    }}
    .custom-copy-btn:hover {{ opacity: 1 !important; background: rgba(76, 175, 80, 0.3) !important; border-color: #4CAF50 !important; color: #a5d6a7 !important; }}
    .custom-copy-btn.copied {{ background: #4CAF50 !important; border-color: #4CAF50 !important; color: white !important; }}

    /* Dialog Styling */
    div[role="dialog"] {{ background: rgba(20, 24, 35, 0.98) !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 20px !important; box-shadow: 0 20px 60px rgba(0,0,0,0.5) !important; }}
    div[role="dialog"] .stButton > button[kind="primary"] {{ background: linear-gradient(90deg, #4CAF50, #2E7D32) !important; border: none !important; border-radius: 10px !important; }}
    div[role="dialog"] .stTextInput > div > div > input, div[role="dialog"] .stSelectbox > div > div, div[role="dialog"] .stSlider > div > div {{ background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.1) !important; color: #fff !important; border-radius: 10px !important; }}

    ::-webkit-scrollbar {{ width: 6px; }} ::-webkit-scrollbar-track {{ background: transparent; }} ::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.1); border-radius: 3px; }} ::-webkit-scrollbar-thumb:hover {{ background: rgba(76, 175, 80, 0.5); }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ==========================================
# 8. JAVASCRIPT ENJEKSİYONU (Kod Blokları Alt Kopyala)
# ==========================================
def inject_copy_button_js():
    # Bu script sayfadaki tüm code bloklarını izler, altına buton ekler.
    # Streamlit'in kendi kopyalama butonu (sağ üst) dokunulmaz.
    js_code = """
    <script>
    (function() {
        const addButtons = () => {
            // Streamlit code block wrapper: div.stCodeBlock veya .element-container pre
            // Yeni Streamlit sürümleri: div[data-testid="stCodeBlock"] veya sadece .stCodeBlock
            document.querySelectorAll('div.stCodeBlock, div[data-testid="stCodeBlock"]').forEach(block => {
                if (block.dataset.copyBtnAdded === 'true') return;

                // Pre/Code elementi bul
                const pre = block.querySelector('pre');
                if (!pre) return;

                // Buton zaten varsa ekleme
                if (block.querySelector('.custom-copy-btn')) return;

                const btn = document.createElement('button');
                btn.className = 'custom-copy-btn';
                btn.innerHTML = '📋 Kopyala';
                btn.title = 'Kodu kopyala';

                btn.onclick = async (e) => {
                    e.stopPropagation();
                    const code = pre.querySelector('code') ? pre.querySelector('code').innerText : pre.innerText;
                    try {
                        await navigator.clipboard.writeText(code);
                        btn.innerHTML = '✅ Kopyalandı';
                        btn.classList.add('copied');
                        setTimeout(() => {
                            btn.innerHTML = '📋 Kopyala';
                            btn.classList.remove('copied');
                        }, 2000);
                    } catch (err) {
                        btn.innerHTML = '❌ Hata';
                        setTimeout(() => { btn.innerHTML = '📋 Kopyala'; }, 2000);
                    }
                };

                block.style.position = 'relative';
                block.appendChild(btn);
                block.dataset.copyBtnAdded = 'true';
            });
        };

        // İlk yükleme
        addButtons();

        // Stream / Dinamik içerik için Observer
        const observer = new MutationObserver((mutations) => {
            let shouldCheck = false;
            for (const m of mutations) {
                if (m.addedNodes.length > 0) { shouldCheck = true; break; }
            }
            if (shouldCheck) addButtons();
        });

        observer.observe(document.body, { childList: true, subtree: true });

        // Cleanup (Sayfa değişince observer durmaz ama Streamlit SPA olduğu için gerek yok, 
        // ancak performans için limitli tutulabilir. Basitlik için body izleniyor.)
    })();
    </script>
    """
    st.components.v1.html(js_code, height=0, width=0)

# ==========================================
# 9. SESSION STATE BAŞLATMA (GÜNCELLENDİ: pending_image)
# ==========================================
def init_session():
    ensure_storage()
    defaults = {
        "active_chat_id": None, "chat_session": None, "messages": [],
        "api_key": DEFAULT_API_KEY, "selected_model": AVAILABLE_MODELS[0], "temperature": 0.5,
        "show_research": False, "research_results": [],
        "last_usage": {"input": 0, "output": 0},
        "context_files_uploader_key": 0,
        "sidebar_history_open": True,
        # YENİ: Bekleyen tek kullanımlık görsel
        "pending_image_context": None, 
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    if not st.session_state.active_chat_id:
        idx = load_index()
        st.session_state.active_chat_id = idx[0]["id"] if idx else create_new_chat().id

    if st.session_state.chat_session is None or st.session_state.chat_session.id != st.session_state.active_chat_id:
        session = load_chat(st.session_state.active_chat_id)
        if session: st.session_state.chat_session = session; st.session_state.messages = session.messages
        else: new_chat = create_new_chat(); st.session_state.active_chat_id = new_chat.id; st.session_state.chat_session = new_chat; st.session_state.messages = []

init_session()

# ==========================================
# 10. DİYALOG (MODAL) TANIMLAMALARI
# ==========================================
@st.dialog("⚙️ Ayarlar", width="large")
def settings_dialog():
    st.markdown("#### Model & API")
    st.session_state.api_key = st.text_input("NVIDIA API Key", value=st.session_state.api_key, type="password", label_visibility="collapsed", placeholder="nvapi-...", key="dialog_api_key")
    st.session_state.selected_model = st.selectbox("Model", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(st.session_state.selected_model) if st.session_state.selected_model in AVAILABLE_MODELS else 0, key="dialog_model")
    st.session_state.temperature = st.slider("Yaratıcılık (Temperature)", 0.0, 1.0, st.session_state.temperature, 0.1, key="dialog_temp")
    st.markdown("---"); st.markdown("#### 🎨 Arayüz"); st.caption("Arka plan resmi kod seviyesinde `BACKGROUND_IMAGE_PATH` değişkeninden ayarlanır.")
    if st.button("Kapat", use_container_width=True, key="dialog_close_btn"): st.rerun()

# ==========================================
# 11. SIDEBAR RENDER (GÜNCELLENDİ: Pending Image Gösterimi)
# ==========================================
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🟢 Nvidia Nemotron 3 Ultra</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("💬 Sohbet", use_container_width=True, type="primary" if not st.session_state.show_research else "secondary", key="nav_chat"): st.session_state.show_research = False; st.rerun()
        with c2:
            if st.button("🔍 Araştır", use_container_width=True, type="primary" if st.session_state.show_research else "secondary", key="nav_research"): st.session_state.show_research = not st.session_state.show_research; st.rerun()

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)

        if st.session_state.show_research:
            render_research_panel()
        else:
            render_history_folder()

        st.markdown('</div>', unsafe_allow_html=True)

        # --- YENİ: Bekleyen Görsel Önizlemesi (Sidebar Altında veya Üstünde) ---
        if st.session_state.pending_image_context:
            st.markdown("---")
            st.markdown("#### 🖼️ Bir Sonraki Mesaj İçin Hazır")
            p = st.session_state.pending_image_context
            # Küçük önizleme
            if p.get("mime", "").startswith("image/"):
                st.image(p["raw_bytes"], width=150, caption=f"'{p['name']}' gönderilecek.")
            else:
                st.info(f"📄 {p['name']} (Metin olarak işlenecek)")

            if st.button("❌ İptal Et / Temizle", use_container_width=True, key="cancel_pending_img"):
                st.session_state.pending_image_context = None
                st.rerun()

        st.markdown("---")
        c_new, c_set = st.columns([3, 1])
        with c_new:
            if st.button("➕ Yeni Sohbet", use_container_width=True, type="primary"): new_chat_action()
        with c_set:
            if st.button("⚙️", use_container_width=True, help="Ayarları Aç", key="settings_gear"): settings_dialog()

        render_context_uploader()

def render_history_folder():
    idx = load_index()
    with st.expander(f"📁 **Geçmiş** ({len(idx)})", expanded=st.session_state.sidebar_history_open):
        if not idx: st.caption("Henüz sohbet yok.")
        else:
            for chat in idx:
                is_active = chat["id"] == st.session_state.active_chat_id
                try: dt = datetime.fromisoformat(chat["updated_at"]).strftime("%d %b %H:%M")
                except: dt = ""
                c1, c2 = st.columns([0.85, 0.15])
                with c1:
                    btn_type = "primary" if is_active else "secondary"
                    label = f"{'🟢 ' if is_active else ''}{chat['title']}\n<small style='color:#888'>{dt}</small>"
                    if st.button(label, key=f"sel_{chat['id']}", use_container_width=True, type=btn_type, help="Aç"): switch_chat(chat["id"])
                with c2:
                    if st.button("🗑️", key=f"del_{chat['id']}", help="Sil", use_container_width=True): delete_chat_action(chat["id"])

def render_research_panel():
    st.markdown("### 🔍 Araştırma Ajanı"); st.caption("Wikipedia & Web'den anlık bilgi toplayın. Sonuçları doğrulayın.")
    query = st.text_input("Ne araştırayım?", placeholder="Örn: Yapay zeka tarihi, Python asyncio...", key="research_query", label_visibility="collapsed")
    col_w, col_web = st.columns(2)
    with col_w:
        if st.button("📖 Wikipedia", use_container_width=True, disabled=not HAS_WIKI):
            if query:
                with st.spinner("Wikipedia taranıyor..."):
                    res = ResearchAgent.search_wiki(query)
                    st.session_state.research_results.insert(0, {"source": "Wikipedia", "query": query, "data": res, "time": datetime.now().strftime("%H:%M")}); st.rerun()
    with col_web:
        if st.button("🌐 Web (DDG)", use_container_width=True, disabled=not HAS_DDG):
            if query:
                with st.spinner("Web aranıyor..."):
                    res_list = ResearchAgent.search_web(query)
                    for r in res_list: st.session_state.research_results.insert(0, {"source": "Web (DDG)", "query": query, "data": r, "time": datetime.now().strftime("%H:%M")}); st.rerun()
    st.markdown("---")
    if not st.session_state.research_results: st.info("Henüz araştırma yapılmadı.")
    else:
        for i, r in enumerate(st.session_state.research_results):
            data = r.get("data", {}); text = data.get("text", "İçerik yok"); url = data.get("url", ""); title = data.get("title", r['query'])
            with st.container(border=True):
                st.markdown(f"**{title}**  \n<small>{r['source']} • {r['time']}</small>", unsafe_allow_html=True)
                st.markdown(text)
                c_link, c_verify, c_del = st.columns([3, 2, 1])
                if url: c_link.markdown(f"[🔗 Kaynağı Ziyaret Et]({url})", unsafe_allow_html=True)
                if c_verify.button("✅ Doğrula / Düzelt", key=f"verify_{i}", use_container_width=True, help="Model bu bilgiyi kendi bilgisiyla karşılaştırır ve düzeltir."):
                    with st.spinner("Model doğruluyor..."):
                        verification_prompt = f"Aşağıdaki bilgiyi doğrula. Eğer hata varsa düzelt, doğruysa onayla. Sadece sonucu ver.\n\nKonu: {r['query']}\n\nBilgi:\n{text}"
                        try:
                            client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=st.session_state.api_key)
                            resp = client.chat.completions.create(model=st.session_state.selected_model, messages=[{"role": "user", "content": verification_prompt}], temperature=0.1)
                            result = resp.choices[0].message.content
                            st.session_state.research_results[i]["verification"] = result; st.toast("Doğrulama tamamlandı.", icon="✅")
                        except Exception as e: st.session_state.research_results[i]["verification"] = f"Hata: {e}"
                        st.rerun()
                if c_del.button("🗑️", key=f"del_res_{i}", use_container_width=True): st.session_state.research_results.pop(i); st.rerun()
                if "verification" in r: st.markdown("---"); st.markdown(f"**🤖 Model Doğrulaması:**\n{r['verification']}")

def render_context_uploader():
    st.markdown("---"); st.markdown("#### 📂 Bağlam Dosyaları (Kalıcı)")
    session = st.session_state.chat_session
    if session and session.context_files:
        for i, f in enumerate(session.context_files):
            c1, c2 = st.columns([0.8, 0.2])
            tok = f.get('tokens', count_tokens(f.get('content','')))
            c1.markdown(f'<span class="context-tag">📄 {f["name"]} <span style="color:#888">({tok/1000:.1f}k tok)</span></span>', unsafe_allow_html=True)
            if c2.button("✕", key=f"rm_ctx_{i}", use_container_width=True): session.context_files.pop(i); save_chat(session); st.rerun()
    else: st.caption("Bu sohbete eklenmiş kalıcı dosya yok.")

    # Uploader: Sadece görsel seçilirse "Pending" yap, diğerleri kalıcı ekle
    uploaded = st.file_uploader("Dosya Ekle (Görsel = Tek Kullanımlık, Diğerleri = Kalıcı)", type=list(ALL_SUPPORTED_EXT), accept_multiple_files=True, label_visibility="collapsed", key=f"uploader_{st.session_state.context_files_uploader_key}")
    if uploaded:
        new_files = process_uploaded_files(uploaded)
        if new_files:
            # İlk dosya görsel mi kontrol et (Basit mantık: Yüklenen ilk dosya görselse pending yap)
            # Veya kullanıcıya sor? İstek: "bir görsel yüklediğimde... sayılsın". 
            # Mantık: Yüklenen dosyalar içinde görsel varsa onları pending yap, diğerleri kalıcı ekle.

            pending_imgs = [f for f in new_files if f["type"] in ("image", "image_ocr")]
            permanent_files = [f for f in new_files if f["type"] not in ("image", "image_ocr")]

            if permanent_files:
                session.context_files.extend(permanent_files)
                save_chat(session)
                st.toast(f"✅ {len(permanent_files)} dosya kalıcı bağlama eklendi.", icon="📂")

            if pending_imgs:
                # Sadece sonuncuyu veya hepsini? İstek: "bir görsel". Tek alalım.
                # Eğer birden fazla görsel yüklendiyse sonuncuyu pending yap, diğerlerini kalıcıya atabiliriz veya hepsini pending yapamazsınız (tek mesaj için).
                # En mantıklısı: Son yüklenen görseli pending yap.
                img_to_pending = pending_imgs[-1]
                st.session_state.pending_image_context = img_to_pending
                st.toast(f"🖼️ '{img_to_pending['name']}' bir sonraki mesajınız için hazır.", icon="🖼️")

                # Eğer diğer görseller varsa onları kalıcıya ekle (veya sil). Burada kalıcıya ekleyelim.
                for p_img in pending_imgs[:-1]:
                    session.context_files.append(p_img)
                save_chat(session)

            st.session_state.context_files_uploader_key += 1
            st.rerun()

# ==========================================
# 12. ANA SAYFA RENDER (CHAT)
# ==========================================
def build_system_prompt(files: List[Dict]) -> str:
    if not files: return ""
    parts = ["# SİSTEM TALİMATI: Kullanıcı aşağıdaki dosyaları bağlam olarak sağladı. Cevap verirken bu dosyalara atıfta bulunun.\n"]
    total_tokens = 0; MAX_CTX = 90000
    for f in files:
        content = f["content"]; f_tokens = count_tokens(content)
        if total_tokens + f_tokens > MAX_CTX:
            content = content[: (MAX_CTX - total_tokens) * 3 ] + "\n... [KIRPİLDİ] ..."; f_tokens = count_tokens(content)
        parts.append(f"--- DOSYA: {f['name']} ({f['type']}) ---\n{content}\n")
        total_tokens += f_tokens
        if total_tokens >= MAX_CTX: break
    return "\n".join(parts)

def render_project_card(files: Dict[str, str], msg_idx: int):
    merger = ProjectMerger(""); merger.files = files; zip_bytes = merger.create_zip()
    st.markdown(f"""<div class="project-card"><h4 style="margin-top:0; color:#a5d6a7; display:flex; align-items:center; gap:0.5rem;">📦 Proje Oluşturuldu ({len(files)} dosya)</h4><div class="project-tree">{merger.get_tree_md()}</div></div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2: st.download_button(label="📥 Projeyi ZIP Olarak İndir", data=zip_bytes, file_name=f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip", mime="application/zip", use_container_width=True, type="primary", key=f"dl_proj_{msg_idx}")

def render_message(msg: ChatMessage, idx: int):
    avatar = "🧑‍💻" if msg.role == "user" else "🤖"
    with st.chat_message(msg.role, avatar=avatar):
        st.markdown(msg.content)
        if msg.role == "user" and msg.files:
            cols = st.columns(min(len(msg.files), 4))
            for j, f in enumerate(msg.files): cols[j % 4].caption(f"📎 {f['name']}")
        if msg.role == "assistant" and msg.project_files: render_project_card(msg.project_files, idx)
        if msg.usage:
            u = msg.usage
            st.markdown(f"""<div class="token-bar"><span class="token-item">📥 Input: <b>{u.get('input', 0):,}</b> tok</span><span class="token-item">📤 Output: <b>{u.get('output', 0):,}</b> tok</span><span class="token-item">📊 Total: <b>{u.get('input', 0) + u.get('output', 0):,}</b> tok</span></div>""", unsafe_allow_html=True)

def render_main_chat():
    session = st.session_state.chat_session
    if not session: return
    st.markdown(f"<h1 style='margin-bottom:0.2rem;'>💬 {session.title}</h1>", unsafe_allow_html=True)
    st.caption(f"Model: `{st.session_state.selected_model}` | Temp: {st.session_state.temperature} | Kalıcı Bağlam: {len(session.context_files)}")
    with st.container():
        for i, msg in enumerate(st.session_state.messages): render_message(msg, i)
    handle_user_input()

# ==========================================
# 13. KULLANICI GİRİŞİ VE STREAMING (GÜNCELLENDİ: Pending Image Kullanımı)
# ==========================================
def handle_user_input():
    session = st.session_state.chat_session
    if not session: return

    # Chat Input
    if prompt := st.chat_input("Nemotron'a mesaj gönder... (Shift+Enter = Yeni Satır)", key="main_chat_input"):
        # 1. Mesaj dosyalarını topla: Kalıcı bağlam + Bekleyen Görsel (varsa)
        current_turn_files = session.context_files.copy() # Kalıcı olanlar her mesajda gider (veya isterseniz sadece ilk mesajda gider, ama prompt "bağlam dosyaları" dediği için her seferinde gitmesi normaldir).

        # YENİ MANTIK: Eğer pending image varsa, BU mesaja ekle ve temizle.
        if st.session_state.pending_image_context:
            current_turn_files.append(st.session_state.pending_image_context)
            st.session_state.pending_image_context = None # Tüketildi, silindi.

        user_msg = ChatMessage(role="user", content=prompt, files=current_turn_files)
        st.session_state.messages.append(user_msg)
        session.messages.append(user_msg)

        if session.title == "Yeni Sohbet" and len(st.session_state.messages) == 1:
            threading.Thread(target=generate_title_async, args=(prompt,), daemon=True).start()

        st.rerun()

def generate_title_async(first_prompt: str):
    try:
        title = first_prompt.strip().split('\n')[0][:50]
        chat_session = st.session_state.chat_session
        if chat_session: chat_session.title = title if title else "Yeni Sohbet"; save_chat(chat_session)
    except: pass

def generate_assistant_response():
    if not st.session_state.messages: return
    last_msg = st.session_state.messages[-1]
    if last_msg.role != "user": return
    session = st.session_state.chat_session
    if not session: return

    with st.chat_message("assistant", avatar="🤖"):
        thinking_ph = st.empty(); content_ph = st.empty()

        # System Prompt: Sadece Kalıcı Bağlam Dosyaları (context_files)
        # Not: Geçici görsel (pending) zaten user_msg.files içine eklendi, model ona user mesajı olarak bakacak.
        # Sistem promptuna kalıcı dosyaları (RAG gibi) koyuyoruz.
        system_prompt = build_system_prompt(session.context_files)

        api_messages = []
        if system_prompt: api_messages.append({"role": "system", "content": system_prompt})

        # Son 20 mesaj (Geçici görsel dahil user mesajında zaten var)
        for m in st.session_state.messages[-20:]: api_messages.append({"role": m.role, "content": m.content})

        est_input_tokens = count_tokens(system_prompt) + sum(count_tokens(m.content) for m in st.session_state.messages[-20:])
        full_response = ""; final_usage = {"input": 0, "output": 0}

        try:
            client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=st.session_state.api_key)
            stream = client.chat.completions.create(model=st.session_state.selected_model, messages=api_messages, temperature=st.session_state.temperature, stream=True)
            first_chunk = True; phrase_idx = 0; start_time = time.time()
            for chunk in stream:
                elapsed = time.time() - start_time
                if elapsed > 0.5:
                    thinking_ph.markdown(f'<div class="thinking-badge"><div class="thinking-spinner"></div>{THINKING_PHRASES[phrase_idx % len(THINKING_PHRASES)]}</div>', unsafe_allow_html=True)
                    phrase_idx += 1
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        if first_chunk: thinking_ph.empty(); first_chunk = False
                        full_response += delta.content; content_ph.markdown(full_response + "▌")
                if hasattr(chunk, 'usage') and chunk.usage:
                    final_usage["input"] = chunk.usage.prompt_tokens; final_usage["output"] = chunk.usage.completion_tokens

            content_ph.markdown(full_response)
            if final_usage["input"] == 0:
                final_usage["input"] = est_input_tokens; final_usage["output"] = count_tokens(full_response)
            st.session_state.last_usage = final_usage

            project_files = {}
            if "### FILE:" in full_response or "--- FILE:" in full_response:
                merger = ProjectMerger(full_response)
                if merger.files: project_files = merger.files; st.toast(f"📦 {len(project_files)} dosyalık proje tespit edildi!", icon="📦")

            assistant_msg = ChatMessage(role="assistant", content=full_response, usage=final_usage, project_files=project_files)
            st.session_state.messages.append(assistant_msg); session.messages.append(assistant_msg); save_chat(session)
            st.rerun()
        except Exception as e:
            thinking_ph.empty()
            error_msg = str(e)
            if "401" in error_msg: error_msg = "❌ Geçersiz API Key (401). Ayarlardan kontrol edin."
            elif "404" in error_msg: error_msg = f"❌ Model bulunamadı (404): `{st.session_state.selected_model}`. Başka model deneyin."
            elif "429" in error_msg: error_msg = "⏳ Rate Limit (429). Biraz bekleyin."
            else: error_msg = f"❌ Hata: {error_msg}"
            content_ph.error(error_msg)

# ==========================================
# 14. STATE FONKSİYONLARI
# ==========================================
def switch_chat(chat_id: str):
    if chat_id == st.session_state.active_chat_id: return
    session = load_chat(chat_id)
    if session: 
        st.session_state.active_chat_id = chat_id
        st.session_state.chat_session = session
        st.session_state.messages = session.messages
        st.session_state.show_research = False
        st.session_state.pending_image_context = None # Sohbet değişince temizle
        st.rerun()

def new_chat_action():
    chat = create_new_chat()
    st.session_state.active_chat_id = chat.id
    st.session_state.chat_session = chat
    st.session_state.messages = []
    st.session_state.show_research = False
    st.session_state.pending_image_context = None
    st.rerun()

def delete_chat_action(chat_id: str):
    delete_chat(chat_id)
    if chat_id == st.session_state.active_chat_id:
        idx = load_index()
        if idx: switch_chat(idx[0]["id"])
        else: new_chat_action()
    else: st.rerun()

# ==========================================
# 15. ANA AKIŞ
# ==========================================
inject_custom_css()
inject_copy_button_js() # JS Enjekte Et

# Sidebar
render_sidebar()

# Main Chat
render_main_chat()

# Assistant Trigger
if st.session_state.messages and st.session_state.messages[-1].role == "user":
    generate_assistant_response()