#!/usr/bin/env python3
# qmt.py — QMT with customtkinter UI, dataset tab & QDG integration
# Version: 0.3.0-complete
# Made by Quixoticode-style — adapted for your environment
#
# Save as qmt.py and run: python qmt.py
# Recommended: pip install customtkinter pillow transformers torch
#
# Notes:
# - The UI will try to use customtkinter (dark theme). If it's missing, a simpler tkinter UI is used.
# - Put qmt_icon.ico in the same folder to see a proper taskbar icon on Windows.
# - QDG launcher attempts to spawn qdg.py from the same directory.
#
import os
import sys
import time
import json
import math
import threading
import queue
import glob
import subprocess
import webbrowser
import traceback
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, IterableDataset

# tkinter + filedialog + messagebox
import tkinter as tk
from tkinter import filedialog, messagebox

# prefer customtkinter
try:
    import customtkinter as ctk
    CTK_AVAILABLE = True
except Exception:
    CTK_AVAILABLE = False

# optional heavy deps
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_PATH = os.path.join(ROOT, "qmt.log")
OUT_LLM_DIR = os.path.join(ROOT, "llm")
DATASETS_DIR = os.path.join(ROOT, "datasets")
os.makedirs(OUT_LLM_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

# -------------------------
# Logging (thread-safe)
# -------------------------
_ui_queue = queue.Queue()  # global UI queue for cross-instance events

_log_lock = threading.Lock()
def log(msg: str, to_ui: bool = True):
    """Thread-safe logging. If to_ui=True the message is queued for the UI."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with _log_lock:
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            # best-effort: print if file write fails
            print("Log write failed:", line)
    print(line)
    if to_ui:
        try:
            _ui_queue.put(("log", line))
        except Exception:
            pass

# -------------------------
# Utilities
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def sanitize_name(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_")
    return safe or "model"

def smart_read_json_lines(path: str):
    items = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline()
            f.seek(0)
            if first.strip().startswith("[") or first.strip().startswith("{"):
                try:
                    data = json.load(f)
                    return data if isinstance(data, list) else [data]
                except Exception:
                    pass
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    items.append({"text": line})
    except Exception as e:
        log(f"smart_read_json_lines failed for {path}: {e}")
    return items

def next_id(prefix="ds"):
    return f"{prefix}_{int(time.time()*1000) % 1000000}"

# simple dataset wrapper (if torch available)
# -------------------------
# OPTIMIZED: Streaming Dataset
# -------------------------
if TORCH_AVAILABLE:
    class StreamingTextDataset(IterableDataset):
        def __init__(self, dataset_configs, tokenizer, max_len):
            self.dataset_configs = dataset_configs
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __iter__(self):
            for ds_config in self.dataset_configs:
                path = ds_config.get("path")
                if not path or not os.path.exists(path):
                    log(f"Dataset not found, skipping: {path}", to_ui=True)
                    continue

                fmt = ds_config.get("format", "prompt_completion")
                m = ds_config.get("map", {})
                
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        
                        text_to_tokenize = self._parse_line(line, fmt, m)
                        
                        if text_to_tokenize and len(text_to_tokenize) > 10:
                            try:
                                encodings = self.tokenizer(
                                    text_to_tokenize,
                                    truncation=True,
                                    max_length=self.max_len,
                                    return_tensors="pt"
                                )
                                yield {
                                    "input_ids": encodings["input_ids"].squeeze(0),
                                    "labels": encodings["input_ids"].squeeze(0).clone()
                                }
                            except Exception as e:
                                log(f"Tokenization failed for line: '{line[:50]}...': {e}", to_ui=False)
        
        def _parse_line(self, line, fmt, m):
            try:
                if fmt in ("json_text", "jsonl", "prompt_completion"):
                    item = json.loads(line)
                    if fmt == "prompt_completion":
                        pkey = m.get("prompt_key", "prompt")
                        ckey = m.get("completion_key", "completion")
                        p = item.get(pkey, "")
                        c = item.get(ckey, "")
                        return f"{p}\n{c}".strip()
                    else:
                        tkey = m.get("text_key", "text")
                        return str(item.get(tkey, json.dumps(item, ensure_ascii=False)))
                return line # Plain text
            except (json.JSONDecodeError, TypeError):
                return line # Treat as plain text on error

# -------------------------
# QMT App (customtkinter)
# -------------------------
class QMTApp:
    def __init__(self, root):
        self.root = root
        self.version = "QMT 0.3.0 (Quixoticode)"
        self.ui_queue = queue.Queue()
        self.datasets = []
        self.training_thread = None
        self.stop_event = threading.Event()

        # model/source state
        self.model_source_var = tk.StringVar(value="hf")  # hf/local/ollama
        self.local_model_dir = ""
        self.ollama_models = []

        # UI default examples
        self.base_model_examples = [
            "distilgpt2",
            "gpt2",
            "facebook/opt-125m",
            "google/flan-t5-small"
        ]

        # optimization options
        self.low_memory_mode = tk.BooleanVar(value=False)
        self.use_8bit = tk.BooleanVar(value=False)
        self.force_cpu = tk.BooleanVar(value=False)

        # initialize UI
        if CTK_AVAILABLE:
            try:
                ctk.set_appearance_mode("dark")
                ctk.set_default_color_theme("blue")
            except Exception:
                pass
            try:
                self._build_ctk_ui()
            except Exception as e:
                log(f"Fatal UI error building CTK UI: {e}\n{traceback.format_exc()}", to_ui=False)
                messagebox.showerror("UI Fehler", f"Fehler beim Erstellen der CTK-UI: {e}")
                # fallback to basic UI
                self._build_basic_ui()
        else:
            # fallback to classic tkinter UI (simpler)
            try:
                messagebox.showwarning("customtkinter nicht gefunden",
                    "customtkinter nicht installiert — starte die Anwendung erneut nachdem du 'pip install customtkinter' ausgeführt hast.\nFahre mit einfacher tkinter-UI.")
            except Exception:
                pass
            self._build_basic_ui()

        # start pollers
        self._start_ui_poller()
        log("QMT initialized", to_ui=True)

    # -------------------------
    # Icon helper
    # -------------------------
    def _set_app_icon(self, icon_path="qmt_icon.ico"):
        """Try to set app icon (Windows: .ico required)."""
        try:
            ip = os.path.join(ROOT, icon_path)
            if os.path.exists(ip):
                if sys.platform.startswith("win"):
                    try:
                        self.root.iconbitmap(ip)
                    except Exception as e:
                        log(f"root.iconbitmap failed: {e}", to_ui=False)
                else:
                    # other platforms: try PhotoImage (PNG recommended)
                    try:
                        img = tk.PhotoImage(file=ip)
                        self.root.iconphoto(False, img)
                    except Exception:
                        # try find png alternative
                        png_alt = os.path.join(ROOT, "qmt_icon.png")
                        if os.path.exists(png_alt):
                            try:
                                img = tk.PhotoImage(file=png_alt)
                                self.root.iconphoto(False, img)
                            except Exception:
                                pass
                log(f"App icon configured (checked): {ip}", to_ui=False)
            else:
                log(f"Icon not found: {ip}", to_ui=False)
        except Exception as e:
            log(f"Failed to set icon: {e}", to_ui=False)

    # -------------------------
    # Build UI (customtkinter)
    # -------------------------
    def _build_ctk_ui(self):
        # clear root
        for w in self.root.winfo_children():
            try: w.destroy()
            except Exception: pass

        self.root.title("QMT — Quick Model Trainer (Quixoticode)")
        self._set_app_icon("qmt_icon.ico")

        self.root.geometry("1160x820")
        main = ctk.CTkFrame(self.root, corner_radius=8)
        main.pack(fill="both", expand=True, padx=12, pady=12)

        top = ctk.CTkFrame(main, corner_radius=6)
        top.pack(fill="x", padx=6, pady=(6,8))

        lbl_title = ctk.CTkLabel(top, text="QMT — Quick Model Trainer", font=("Roboto", 18, "bold"))
        lbl_title.pack(side="left", padx=(8,12))
        lbl_ver = ctk.CTkLabel(top, text=self.version)
        lbl_ver.pack(side="left")

        # docs / github
        def _open_docs(): webbrowser.open("https://docs.quixoticode.de/qmt", new=2)
        def _open_github(): webbrowser.open("https://github.com/Quixoticode", new=2)
        btn_docs = ctk.CTkButton(top, text="Docs (de)", width=100, command=_open_docs)
        btn_docs.pack(side="right", padx=(6,8))
        btn_gh = ctk.CTkButton(top, text="Quixoticode", width=140, command=_open_github)
        btn_gh.pack(side="right", padx=(6,0))

        # content area with tabview
        content = ctk.CTkFrame(main, corner_radius=6)
        content.pack(fill="both", expand=True, padx=6, pady=6)

        tabview = ctk.CTkTabview(content, width=1000)
        tabview.pack(fill="both", expand=True, padx=6, pady=6)
        tabview.add("Model")
        tabview.add("Datasets")
        tabview.add("Training")
        tabview.add("Logs")

        # --- Model Tab ---
        model_fr = ctk.CTkFrame(tabview.tab("Model"))
        model_fr.pack(fill="both", expand=True, padx=8, pady=8)

        # metadata area
        md_fr = ctk.CTkFrame(model_fr)
        md_fr.pack(fill="x", padx=8, pady=8)

        ctk.CTkLabel(md_fr, text="Name:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.ent_name = ctk.CTkEntry(md_fr, width=360)
        self.ent_name.grid(row=0, column=1, sticky="w", padx=6, pady=6)

        ctk.CTkLabel(md_fr, text="Autor:").grid(row=0, column=2, sticky="w", padx=6, pady=6)
        self.ent_author = ctk.CTkEntry(md_fr, width=220)
        self.ent_author.grid(row=0, column=3, sticky="w", padx=6, pady=6)

        ctk.CTkLabel(md_fr, text="Base (HF/local/ollama):").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        self.cmb_base = ctk.CTkComboBox(md_fr, values=self.base_model_examples, width=360)
        self.cmb_base.set(self.base_model_examples[0])
        self.cmb_base.grid(row=1, column=1, sticky="w", padx=6, pady=6)

        # model source options
        src_fr = ctk.CTkFrame(md_fr)
        src_fr.grid(row=2, column=0, columnspan=4, sticky="we", padx=6, pady=6)
        ctk.CTkLabel(src_fr, text="Modell-Quelle:").pack(side="left", padx=(0,8))
        try:
            self.rb_hf = ctk.CTkRadioButton(src_fr, text="HuggingFace", variable=self.model_source_var, value="hf")
            self.rb_hf.pack(side="left", padx=6)
            self.rb_local = ctk.CTkRadioButton(src_fr, text="Lokal", variable=self.model_source_var, value="local")
            self.rb_local.pack(side="left", padx=6)
            self.rb_ollama = ctk.CTkRadioButton(src_fr, text="Ollama (CLI)", variable=self.model_source_var, value="ollama")
            self.rb_ollama.pack(side="left", padx=6)
        except Exception:
            # older customtk version: use regular Radiobuttons
            tk.Radiobutton(src_fr, text="HuggingFace", variable=self.model_source_var, value="hf").pack(side="left", padx=6)
            tk.Radiobutton(src_fr, text="Lokal", variable=self.model_source_var, value="local").pack(side="left", padx=6)
            tk.Radiobutton(src_fr, text="Ollama (CLI)", variable=self.model_source_var, value="ollama").pack(side="left", padx=6)

        # local model select
        ctk.CTkLabel(md_fr, text="Lokales Modell-Verzeichnis:").grid(row=3, column=0, sticky="w", padx=6, pady=6)
        self.ent_local_model = ctk.CTkEntry(md_fr, width=600)
        self.ent_local_model.grid(row=3, column=1, columnspan=2, sticky="w", padx=6, pady=6)
        btn_pick_local = ctk.CTkButton(md_fr, text="Wähle Ordner...", width=140, command=self._pick_local_model_dir)
        btn_pick_local.grid(row=3, column=3, sticky="w", padx=6, pady=6)

        # Ollama combobox + refresh
        ctk.CTkLabel(md_fr, text="Ollama Modell (falls Ollama):").grid(row=4, column=0, sticky="w", padx=6, pady=6)
        self.cmb_ollama = ctk.CTkComboBox(md_fr, values=self.ollama_models, width=520)
        self.cmb_ollama.grid(row=4, column=1, columnspan=2, sticky="w", padx=6, pady=6)
        btn_refresh_ollama = ctk.CTkButton(md_fr, text="Refresh Ollama", width=140, command=self._refresh_ollama_async)
        btn_refresh_ollama.grid(row=4, column=3, sticky="w", padx=6, pady=6)

        # system prompt
        ctk.CTkLabel(model_fr, text="System prompt (optional):").pack(anchor="w", padx=12, pady=(8,0))
        self.txt_sys = ctk.CTkTextbox(model_fr, height=80)
        self.txt_sys.pack(fill="x", padx=12, pady=(4,8))

        # optimization options
        opt_fr = ctk.CTkFrame(model_fr)
        opt_fr.pack(fill="x", padx=12, pady=(4,12))
        ctk.CTkLabel(opt_fr, text="Optimizations for low-end hardware:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.chk_lowmem = ctk.CTkCheckBox(opt_fr, text="Low memory mode (reduce batch/max tokens)", variable=self.low_memory_mode, command=self._apply_lowmem_hint)
        self.chk_lowmem.grid(row=0, column=1, sticky="w", padx=6, pady=6)
        self.chk_8bit = ctk.CTkCheckBox(opt_fr, text="Use 8-bit (bitsandbytes) if available", variable=self.use_8bit)
        self.chk_8bit.grid(row=0, column=2, sticky="w", padx=6, pady=6)
        self.chk_cpu = ctk.CTkCheckBox(opt_fr, text="Force CPU (no GPU)", variable=self.force_cpu)
        self.chk_cpu.grid(row=0, column=3, sticky="w", padx=6, pady=6)
        btn_autotune = ctk.CTkButton(opt_fr, text="Auto-tune for GPU", command=self._auto_tune_for_gpu)
        btn_autotune.grid(row=0, column=4, sticky="e", padx=6, pady=6)

        # --- Datasets Tab ---
        ds_fr = ctk.CTkFrame(tabview.tab("Datasets"))
        ds_fr.pack(fill="both", expand=True, padx=8, pady=8)
        top_ds = ctk.CTkFrame(ds_fr)
        top_ds.pack(fill="x", padx=8, pady=(6,8))
        ctk.CTkButton(top_ds, text="Datei hinzufügen", command=self._add_dataset_files, width=160).pack(side="left", padx=(0,8))
        ctk.CTkButton(top_ds, text="URL hinzufügen", command=self._add_dataset_url, width=160).pack(side="left", padx=(0,8))
        ctk.CTkButton(top_ds, text="Konfigurieren", command=self._config_dataset, width=140).pack(side="left", padx=(0,8))
        ctk.CTkButton(top_ds, text="Entfernen", command=self._remove_dataset, width=120).pack(side="left", padx=(0,8))
        ctk.CTkButton(top_ds, text="Open QDG", command=self._open_qdg, width=120).pack(side="right", padx=(8,0))

        # datasets list
        self.lst_datasets = tk.Listbox(ds_fr, height=18, bd=0, highlightthickness=0)
        self.lst_datasets.pack(fill="both", expand=True, padx=8, pady=(0,8))
        ds_scroll = tk.Scrollbar(ds_fr, command=self.lst_datasets.yview)
        ds_scroll.pack(side="right", fill="y", pady=(0,8))
        self.lst_datasets.config(yscrollcommand=ds_scroll.set)

        # --- Training Tab ---
        tr_fr = ctk.CTkFrame(tabview.tab("Training"))
        tr_fr.pack(fill="both", expand=True, padx=8, pady=8)

        left_tr = ctk.CTkFrame(tr_fr)
        left_tr.pack(side="left", fill="y", padx=(8,6), pady=6)

        right_tr = ctk.CTkFrame(tr_fr)
        right_tr.pack(side="right", fill="both", expand=True, padx=(6,8), pady=6)

        # hyperparams
        ctk.CTkLabel(left_tr, text="Epochen:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.spin_epochs = ctk.CTkEntry(left_tr, width=120); self.spin_epochs.grid(row=0, column=1, padx=6, pady=6); self.spin_epochs.insert(0, "3")

        ctk.CTkLabel(left_tr, text="Batchgröße:").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        self.spin_bs = ctk.CTkEntry(left_tr, width=120); self.spin_bs.grid(row=1, column=1, padx=6, pady=6); self.spin_bs.insert(0, "2")
        
        # --- NEU: Gradient Accumulation ---
        ctk.CTkLabel(left_tr, text="Grad Accum Steps:").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        self.spin_grad_accum = ctk.CTkEntry(left_tr, width=120); self.spin_grad_accum.grid(row=2, column=1, padx=6, pady=6); self.spin_grad_accum.insert(0, "4")
        # --- ENDE NEU ---

        ctk.CTkLabel(left_tr, text="LR:").grid(row=3, column=0, sticky="w", padx=6, pady=6)
        self.ent_lr = ctk.CTkEntry(left_tr, width=120); self.ent_lr.grid(row=3, column=1, padx=6, pady=6); self.ent_lr.insert(0, "5e-5")

        ctk.CTkLabel(left_tr, text="Max tokens:").grid(row=4, column=0, sticky="w", padx=6, pady=6)
        self.spin_maxlen = ctk.CTkEntry(left_tr, width=120); self.spin_maxlen.grid(row=4, column=1, padx=6, pady=6); self.spin_maxlen.insert(0, "512")

        self.btn_start = ctk.CTkButton(left_tr, text="Start Training", command=self._start_training, fg_color="#16a34a")
        self.btn_start.grid(row=5, column=0, columnspan=2, sticky="we", padx=6, pady=(12,6))
        self.btn_stop = ctk.CTkButton(left_tr, text="Stop Training", command=self._stop_training, fg_color="#ef4444")
        self.btn_stop.grid(row=6, column=0, columnspan=2, sticky="we", padx=6, pady=(0,6))
        self.btn_stop.configure(state="disabled")
        # progress bars & logs
        ctk.CTkLabel(right_tr, text="Progress").pack(anchor="w", padx=8, pady=(6,4))
        self.pb_total = ctk.CTkProgressBar(right_tr)
        self.pb_total.set(0.0); self.pb_total.pack(fill="x", padx=12, pady=(0,8))
        self.pb_epoch = ctk.CTkProgressBar(right_tr); self.pb_epoch.set(0.0); self.pb_epoch.pack(fill="x", padx=12, pady=(0,8))
        self.pb_batch = ctk.CTkProgressBar(right_tr); self.pb_batch.set(0.0); self.pb_batch.pack(fill="x", padx=12, pady=(0,8))

        ctk.CTkLabel(right_tr, text="Log").pack(anchor="w", padx=8, pady=(8,4))
        self.txt_log = ctk.CTkTextbox(right_tr, height=16)
        self.txt_log.pack(fill="both", expand=True, padx=12, pady=(0,12))

        # --- Logs Tab ---
        logs_fr = ctk.CTkFrame(tabview.tab("Logs"))
        logs_fr.pack(fill="both", expand=True, padx=8, pady=8)
        ctk.CTkLabel(logs_fr, text="Raw Log (file)").pack(anchor="w", padx=12, pady=(8,6))
        self.txt_raw = ctk.CTkTextbox(logs_fr, height=30)
        self.txt_raw.pack(fill="both", expand=True, padx=12, pady=(0,12))
        btn_reload_log = ctk.CTkButton(logs_fr, text="Reload log file", command=self._reload_logfile)
        btn_reload_log.pack(padx=12, pady=(0,8))

        # initial UI seeds
        try:
            self.ent_local_model.insert(0, "")
        except Exception:
            pass
        self._refresh_ollama_async()

    # -------------------------
    # Basic fallback UI (tkinter) - kept minimal
    # -------------------------
    def _build_basic_ui(self):
        self.root.title("QMT — Quick Model Trainer (basic UI)")
        self.root.geometry("1000x700")
        frm = tk.Frame(self.root, padx=8, pady=8)
        frm.pack(fill="both", expand=True)
        tk.Label(frm, text="QMT — basic UI (customtkinter not installed)", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        tk.Button(frm, text="Open QDG", command=self._open_qdg).pack(anchor="w", pady=(6,0))
        tk.Button(frm, text="Reload Log", command=self._reload_logfile).pack(anchor="w", pady=(6,0))
        self._append_raw("UI started (basic)")

    # -------------------------
    # UI helpers (append-only: avoid recursive logging)
    # -------------------------
    def _append_log_ui(self, text: str):
        """Append into the visible log textbox and also write to file (no UI requeue to avoid recursion)."""
        try:
            if CTK_AVAILABLE and hasattr(self, "txt_log"):
                self.txt_log.insert("end", text + "\n")
                self.txt_log.see("end")
            elif hasattr(self, "txt_raw"):
                self.txt_raw.insert("end", text + "\n")
                self.txt_raw.see("end")
        except Exception:
            pass
        # write to file only (don't requeue to UI)
        log(text, to_ui=False)

    def _append_raw(self, text: str):
        try:
            if CTK_AVAILABLE and hasattr(self, "txt_raw"):
                self.txt_raw.insert("end", text + "\n")
                self.txt_raw.see("end")
            else:
                print(text)
        except Exception:
            print(text)
        log(text, to_ui=False)

    def _reload_logfile(self):
        try:
            if os.path.exists(LOG_PATH):
                with open(LOG_PATH, "r", encoding="utf-8") as f:
                    data = f.read()
                if CTK_AVAILABLE and hasattr(self, "txt_raw"):
                    try:
                        self.txt_raw.delete("0.0", "end")
                        self.txt_raw.insert("0.0", data)
                    except Exception:
                        pass
                else:
                    # show truncated if too large
                    messagebox.showinfo("Logfile", data[:10000])
            else:
                messagebox.showinfo("Logfile", "No log file yet.")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot read log file: {e}")

    # -------------------------
    # Ollama helpers
    # -------------------------
    def _list_ollama_models(self):
        try:
            p = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=6)
            out = p.stdout.strip() or p.stderr.strip()
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            return lines
        except Exception as e:
            log(f"Ollama list failed: {e}", to_ui=True)
            return []

    def _refresh_ollama_async(self):
        threading.Thread(target=self._refresh_ollama_bg, daemon=True).start()

    def _refresh_ollama_bg(self):
        models = self._list_ollama_models()
        self.ollama_models = models
        try:
            if CTK_AVAILABLE and hasattr(self, "cmb_ollama"):
                try:
                    self.cmb_ollama.configure(values=models)
                    if models:
                        self.cmb_ollama.set(models[0])
                except Exception:
                    pass
        except Exception:
            pass
        _ui_queue.put(("log", f"Ollama models refreshed: {len(models)}"))

    # -------------------------
    # Local model folder picker
    # -------------------------
    def _pick_local_model_dir(self):
        """Open a folder dialog, set local model dir and switch model source to 'local'."""
        try:
            d = filedialog.askdirectory(title="Lokales Modell-Verzeichnis wählen", initialdir=OUT_LLM_DIR)
            if d:
                self.local_model_dir = d
                try:
                    # update field in UI
                    if hasattr(self, "ent_local_model"):
                        try:
                            self.ent_local_model.delete(0, "end")
                            self.ent_local_model.insert(0, d)
                        except Exception:
                            pass
                except Exception:
                    pass
                # set source to local
                try:
                    self.model_source_var.set("local")
                except Exception:
                    pass
                log(f"Local model dir set: {d}", to_ui=True)
        except Exception as e:
            log(f"_pick_local_model_dir failed: {e}", to_ui=True)

    # -------------------------
    # Dataset UI handlers
    # -------------------------
    def _add_dataset_files(self):
        p = filedialog.askopenfilenames(title="Dataset Dateien wählen", filetypes=[("All files", "*.*")])
        if not p:
            return
        for f in p:
            fp = os.path.abspath(f)
            ds = {"id": next_id("ds"), "source": "file", "path": fp, "orig": fp, "format": "json_text",
                  "map": {"text_key": "text", "prompt_key": "prompt", "completion_key": "completion"}}
            self.datasets.append(ds)
            try:
                self.lst_datasets.insert("end", f"{os.path.basename(fp)}  [{ds['format']}]")
            except Exception:
                pass
            log(f"Dataset added: {fp}", to_ui=True)

    def _add_dataset_url(self):
        # ask url and download into datasets folder
        from tkinter.simpledialog import askstring
        url = askstring("Dataset URL", "Gib die URL zu einer JSON/JSONL/TXT Datei ein:")
        if not url:
            return
        datasets_dir = ensure_dir(DATASETS_DIR)
        local_name = os.path.join(datasets_dir, sanitize_name(os.path.basename(url)) or "dl_" + next_id())
        if not os.path.splitext(local_name)[1]:
            local_name += ".dat"
        ds = {"id": next_id("ds"), "source": "url", "path": local_name, "orig": url, "format": "json_text",
              "map": {"text_key": "text", "prompt_key": "prompt", "completion_key": "completion"}}
        self.datasets.append(ds)
        try:
            self.lst_datasets.insert("end", f"{url} (downloading...)")
        except Exception:
            pass
        threading.Thread(target=self._download_dataset_bg, args=(ds, len(self.datasets)-1, local_name), daemon=True).start()
        log(f"Started download: {url}", to_ui=True)

    def _download_dataset_bg(self, ds, list_idx, local_name):
        url = ds.get("orig")
        try:
            import urllib.request, urllib.error
            req = urllib.request.Request(url, headers={"User-Agent": "QMT/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                total = getattr(resp, "length", None) or 0
                chunk = 8192
                downloaded = 0
                with open(local_name, "wb") as outf:
                    while True:
                        data = resp.read(chunk)
                        if not data:
                            break
                        outf.write(data)
                        downloaded += len(data)
                        if total:
                            pct = int(downloaded * 100 / total)
                            _ui_queue.put(("progress_download", pct))
                        else:
                            _ui_queue.put(("progress_download", None))
            # update list item
            try:
                self.lst_datasets.delete(list_idx)
                self.lst_datasets.insert(list_idx, f"{os.path.basename(url)}  [downloaded]")
            except Exception:
                pass
            ds["path"] = os.path.abspath(local_name)
            ds["source"] = "file"
            log(f"Download complete: {url}", to_ui=True)
        except Exception as e:
            log(f"Download failed {url}: {e}", to_ui=True)
            try:
                self.lst_datasets.delete(list_idx)
                self.lst_datasets.insert(list_idx, f"{url}  [failed]")
            except Exception:
                pass
        finally:
            _ui_queue.put(("progress_download", 0))

    def _config_dataset(self):
        sel = None
        try:
            sel = self.lst_datasets.curselection()
        except Exception:
            pass
        if not sel:
            messagebox.showinfo("Keine Auswahl", "Bitte ein Dataset wählen.")
            return
        idx = sel[0]
        ds = self.datasets[idx]
        win = tk.Toplevel(self.root)
        win.title("Dataset konfigurieren")
        win.geometry("520x260")
        tk.Label(win, text=f"Quelle: {ds.get('orig') or ds.get('path')}").pack(anchor="w", padx=8, pady=(8,0))
        tk.Label(win, text="Format:").pack(anchor="w", padx=8, pady=(8,0))
        var_fmt = tk.StringVar(value=ds.get("format", "json_text"))
        try:
            cmb = tk.ttk.Combobox(win, values=["json_text", "jsonl", "prompt_completion", "kv_json", "plain"], textvariable=var_fmt)
        except Exception:
            cmb = None
        if cmb:
            cmb.pack(fill="x", padx=8, pady=6)
        frame = tk.Frame(win)
        frame.pack(fill="x", padx=8, pady=6)
        tk.Label(frame, text="Text key:").grid(row=0, column=0, sticky="w")
        ent_text = tk.Entry(frame); ent_text.grid(row=0, column=1, sticky="we", padx=6); ent_text.insert(0, ds.get("map", {}).get("text_key", "text"))
        tk.Label(frame, text="Prompt key:").grid(row=1, column=0, sticky="w")
        ent_p = tk.Entry(frame); ent_p.grid(row=1, column=1, sticky="we", padx=6); ent_p.insert(0, ds.get("map", {}).get("prompt_key", "prompt"))
        tk.Label(frame, text="Completion key:").grid(row=2, column=0, sticky="w")
        ent_c = tk.Entry(frame); ent_c.grid(row=2, column=1, sticky="we", padx=6); ent_c.insert(0, ds.get("map", {}).get("completion_key", "completion"))
        frame.columnconfigure(1, weight=1)
        def apply_and_close():
            ds["format"] = var_fmt.get()
            ds["map"] = {"text_key": ent_text.get().strip(), "prompt_key": ent_p.get().strip(), "completion_key": ent_c.get().strip()}
            try:
                self.lst_datasets.delete(idx)
                self.lst_datasets.insert(idx, f"{os.path.basename(ds.get('orig') or ds.get('path'))}  [{ds['format']}]")
            except Exception:
                pass
            log(f"Dataset configured: {ds.get('orig') or ds.get('path')} as {ds['format']}", to_ui=True)
            win.destroy()
        tk.Button(win, text="OK", command=apply_and_close).pack(pady=8)

    def _remove_dataset(self):
        sel = ()
        try:
            sel = self.lst_datasets.curselection()
        except Exception:
            pass
        if not sel:
            return
        idx = sel[0]
        ds = self.datasets.pop(idx)
        try:
            self.lst_datasets.delete(idx)
        except Exception:
            pass
        log(f"Dataset removed: {ds.get('orig') or ds.get('path')}", to_ui=True)

    # -------------------------
    # QDG launcher
    # -------------------------
    def _open_qdg(self):
        script = os.path.join(ROOT, "qdg.py")
        if os.path.exists(script):
            try:
                # spawn qdg in new process to keep UI responsive
                if sys.platform.startswith("win"):
                    subprocess.Popen([sys.executable, script], creationflags=subprocess.CREATE_NEW_CONSOLE)
                else:
                    subprocess.Popen([sys.executable, script])
                self._append_log_ui("QDG gestartet.")
            except Exception as e:
                self._append_log_ui(f"Fehler beim Starten von QDG: {e}")
        else:
            self._append_log_ui("qdg.py nicht gefunden im selben Ordner.")

    # -------------------------
    # Checkpoints & metadata
    # -------------------------
    def find_latest_checkpoint(self, outdir: str):
        candidates = []
        patterns = [os.path.join(outdir, "checkpoint-epoch*"), os.path.join(outdir, "checkpoint*"), os.path.join(outdir, "sim-checkpoint*")]
        for pat in patterns:
            for p in glob.glob(pat):
                if os.path.isdir(p):
                    candidates.append(p)
        extras = ["pytorch_model.bin", "tf_model.h5", "config.json", "tokenizer.json", "tokenizer_config.json"]
        if os.path.isdir(outdir) and any(os.path.exists(os.path.join(outdir, f)) for f in extras):
            candidates.append(outdir)
        if not candidates:
            return None
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    def load_checkpoint_meta(self, ckpt_dir: str):
        meta_file = os.path.join(ckpt_dir, "checkpoint_meta.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log(f"Failed to read checkpoint_meta.json: {e}", to_ui=True)
        return {}

    def _import_checkpoint(self):
        sel = filedialog.askdirectory(title="Wähle Ordner mit Checkpoint / Modell (z.B. ./llm/<name>)", initialdir=OUT_LLM_DIR)
        if not sel:
            return
        meta_file = os.path.join(sel, "metadata.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                # populate UI
                try:
                    self.ent_name.delete(0, "end"); self.ent_name.insert(0, meta.get("name", ""))
                    self.ent_author.delete(0, "end"); self.ent_author.insert(0, meta.get("author", ""))
                    self.txt_sys.delete("0.0", "end"); self.txt_sys.insert("0.0", meta.get("system_prompt", ""))
                    self.cmb_base.set(meta.get("base_model", self.base_model_examples[0]))
                    self.datasets = meta.get("datasets", []) or []
                    # refresh dataset listbox
                    try:
                        self.lst_datasets.delete(0, "end")
                        for ds in self.datasets:
                            self.lst_datasets.insert("end", os.path.basename(ds.get("orig") or ds.get("path", "?")) + f" [{ds.get('format','?')}]")
                    except Exception:
                        pass
                except Exception:
                    pass
                self._append_log_ui(f"Checkpoint/Config importiert aus {sel}")
            except Exception as e:
                messagebox.showerror("Fehler", f"Meta lesen fehlgeschlagen: {e}")
                log(f"Import checkpoint failed: {e}", to_ui=True)
        else:
            self._append_log_ui(f"Kein metadata.json in {sel}. Setze als local model dir.")
            self.local_model_dir = sel
            try:
                self.ent_local_model.delete(0, "end"); self.ent_local_model.insert(0, sel)
            except Exception:
                pass
            self.model_source_var.set("local")

    def _save_interrupted_checkpoint(self, model, tokenizer, outdir, epoch, step):
        try:
            ckpt_dir = os.path.join(outdir, f"checkpoint-interrupt-epoch{epoch}-step{step}")
            ensure_dir(ckpt_dir)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            meta_info = {"epoch": epoch, "step": step, "timestamp": time.time(), "interrupted": True}
            with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w", encoding="utf-8") as mf:
                json.dump(meta_info, mf, indent=2, ensure_ascii=False)
            _ui_queue.put(("log", f"Interrupt-Checkpoint saved: {ckpt_dir}"))
            log(f"Interrupt-Checkpoint saved: {ckpt_dir}", to_ui=False)
        except Exception as e:
            log(f"Failed to save interrupt checkpoint: {e}", to_ui=True)

    # -------------------------
    # Training control
    # -------------------------
    def _start_training(self):
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showinfo("Training läuft", "Training läuft bereits.")
            return
        name = self.ent_name.get().strip()
        if not name:
            messagebox.showinfo("Name fehlt", "Gib einen Modellnamen an.")
            return
        if not self.datasets:
            messagebox.showinfo("Datasets fehlen", "Füge mindestens ein Dataset hinzu.")
            return
        
        try:
            params = {
                "epochs": int(self.spin_epochs.get()),
                "batch_size": int(self.spin_bs.get()),
                "grad_accum": int(self.spin_grad_accum.get()),
                "lr": float(self.ent_lr.get()),
                "max_len": int(self.spin_maxlen.get()),
                "base_model": self.cmb_base.get().strip()
            }
        except Exception as e:
            messagebox.showerror("Parameter Fehler", f"Überprüfe Parameter: {e}")
            return

        outdir = ensure_dir(os.path.join(OUT_LLM_DIR, sanitize_name(name)))
        
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.stop_event.clear()
        
        self.training_thread = threading.Thread(target=self._training_bg, args=(outdir, params), daemon=True)
        self.training_thread.start()
        log(f"Training started for {name} (outdir={outdir})", to_ui=True)


    def _training_bg(self, outdir, params):
        try:
            if not (TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE):
                raise RuntimeError("transformers and torch are required for training.")

            log(f"Starting training with params: {params}", to_ui=True)
            device = torch.device("cuda" if torch.cuda.is_available() and not self.force_cpu.get() else "cpu")
            log(f"Using device: {device}", to_ui=True)

            tokenizer = AutoTokenizer.from_pretrained(params['base_model'], use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(params['base_model'])
            model.to(device)
            log(f"Base model '{params['base_model']}' loaded to {device}", to_ui=True)

            dataset = StreamingTextDataset(self.datasets, tokenizer, params['max_len'])
            
            def collate_fn(batch):
                input_ids = [item['input_ids'] for item in batch]
                padded_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
                return {"input_ids": padded_ids, "labels": padded_ids.clone()}
            
            dataloader = DataLoader(dataset, batch_size=params['batch_size'], collate_fn=collate_fn)
            optimizer = AdamW(model.parameters(), lr=params['lr'])

            model.train()
            global_step = 0
            for epoch in range(params['epochs']):
                if self.stop_event.is_set(): break
                log(f"--- Starting Epoch {epoch + 1}/{params['epochs']} ---", to_ui=True)
                
                for i, batch in enumerate(dataloader):
                    if self.stop_event.is_set(): break
                    
                    input_ids = batch['input_ids'].to(device)
                    outputs = model(input_ids, labels=input_ids)
                    loss = outputs.loss / params['grad_accum']
                    loss.backward()
                    
                    if (i + 1) % params['grad_accum'] == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1
                        _ui_queue.put(("log", f"E{epoch+1}, Step {global_step}, Loss: {loss.item() * params['grad_accum']:.4f}"))

                if not self.stop_event.is_set():
                    self._save_interrupted_checkpoint(model, tokenizer, outdir, epoch + 1, global_step) # Re-using this function for epoch checkpoints
                    _ui_queue.put(("progress_epoch", int(100 * (epoch+1)/params['epochs'])))
        
        except Exception as e:
            tb = traceback.format_exc()
            log(f"Training failed: {e}\n{tb}", to_ui=True)
            _ui_queue.put(("error", f"Training failed: {e}"))
        finally:
            self._finalize_training_ui()

    def _stop_training(self):
        if not (self.training_thread and getattr(self.training_thread, "is_alive", lambda: False)()):
            return
        self.stop_event.set()
        log("Stop requested by user", to_ui=True)
        self._append_log_ui("Stop requested — saving checkpoint...")
        try:
            self.btn_stop.configure(state="disabled")
        except Exception:
            pass

    # -------------------------
    # Training background (real or simulated)
    # -------------------------
    def _training_bg(self, name, outdir, epochs, batch_size, lr, max_len):
        base_model = self.cmb_base.get().strip() if hasattr(self, "cmb_base") else "distilgpt2"
        # prepare dataset texts
        try:
            inputs = self._prepare_training_texts(max_len)
        except Exception as e:
            log(f"Dataset prepare failed: {e}", to_ui=True)
            _ui_queue.put(("error", f"Dataset prepare failed: {e}"))
            self._finalize_training_ui()
            return

        total_items = len(inputs)
        log(f"Prepared {total_items} training items", to_ui=True)
        # real training if transformers+torch available and user didn't force CPU only incorrectly
        can_train = TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE and (not self.force_cpu.get())
        if can_train:
            _ui_queue.put(("log", "Transformers+Torch gefunden — starte echtes Training"))
            tokenizer = None
            model = None
            start_epoch = 1
            ckpt_dir = self.find_latest_checkpoint(outdir)
            if ckpt_dir:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
                    model = AutoModelForCausalLM.from_pretrained(ckpt_dir)
                    meta = self.load_checkpoint_meta(ckpt_dir)
                    prev_epoch = int(meta.get("epoch", 0))
                    start_epoch = prev_epoch + 1
                    log(f"Resuming from checkpoint {ckpt_dir}, last epoch={prev_epoch}", to_ui=True)
                except Exception as e:
                    log(f"Failed to resume from checkpoint {ckpt_dir}: {e}", to_ui=True)
                    tokenizer = None
                    model = None
            # load base model depending on source selection
            if model is None or tokenizer is None:
                try:
                    if self.model_source_var.get() == "local" and self.local_model_dir:
                        tokenizer = AutoTokenizer.from_pretrained(self.local_model_dir, use_fast=True)
                        model = AutoModelForCausalLM.from_pretrained(self.local_model_dir)
                        log(f"Loaded local model from {self.local_model_dir}", to_ui=True)
                    elif self.model_source_var.get() == "ollama":
                        # For Ollama-based training/resume we currently cannot load model in HF API — user should export/push to HF/local
                        raise RuntimeError("Ollama as base for real training not yet supported in-process")
                    else:
                        # HF repo or id
                        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
                        model = AutoModelForCausalLM.from_pretrained(base_model)
                        log(f"Base model '{base_model}' loaded", to_ui=True)
                except Exception as e:
                    log(f"Cannot load base model '{base_model}': {e}", to_ui=True)
                    # fallback
                    try:
                        tokenizer = AutoTokenizer.from_pretrained("distilgpt2", use_fast=True)
                        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
                        log("Fallback tokenizer/model: distilgpt2 loaded", to_ui=True)
                    except Exception as e2:
                        log(f"Fallback failed: {e2}", to_ui=True)
                        tokenizer = None
                        model = None

            if model is None or tokenizer is None:
                _ui_queue.put(("error", "Model oder Tokenizer konnten nicht geladen werden. Schau qmt log an."))
                self._finalize_training_ui()
                return

            device = torch.device("cuda" if torch.cuda.is_available() and not self.force_cpu.get() else "cpu")
            model.to(device)
            # tokenize
            encodings = {"input_ids": []}
            for i, txt in enumerate(inputs):
                if self.stop_event.is_set():
                    break
                try:
                    ids = tokenizer.encode(txt, truncation=True, max_length=max_len)
                except Exception:
                    # best-effort truncation
                    try:
                        ids = tokenizer.encode(txt[:max_len], truncation=True, max_length=max_len)
                    except Exception:
                        ids = []
                if len(ids) < 1:
                    continue
                encodings["input_ids"].append(ids)
                if i % 200 == 0:
                    _ui_queue.put(("progress_total", int(100 * (i / max(1, total_items)))))
            if len(encodings["input_ids"]) == 0:
                _ui_queue.put(("error", "No tokenized inputs. Abort."))
                self._finalize_training_ui()
                return

            dataset = SimpleTextDataset(encodings)
            try:
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_pad)
            except Exception as e:
                log(f"DataLoader failed: {e}", to_ui=True)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=self._collate_pad)

            optimizer = AdamW(model.parameters(), lr=lr)
            total_steps = epochs * math.ceil(len(dataset) / max(1, batch_size))
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)

            step = 0
            # training loop
            for epoch in range(start_epoch, epochs + 1):
                if self.stop_event.is_set():
                    log("Stop event detected before epoch start", to_ui=True)
                    break
                model.train()
                epoch_loss = 0.0
                num_batches = math.ceil(len(dataset) / max(1, batch_size))
                interrupted = False
                for b_idx, batch in enumerate(dataloader, start=1):
                    if self.stop_event.is_set():
                        log("Stop event detected during training loop", to_ui=True)
                        try:
                            self._save_interrupted_checkpoint(model, tokenizer, outdir, epoch, step)
                        except Exception as e:
                            log(f"Error saving interrupted checkpoint: {e}", to_ui=True)
                        interrupted = True
                        break
                    inputs_ids = batch["input_ids"].to(device)
                    labels = inputs_ids
                    outputs = model(inputs_ids, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    epoch_loss += loss.item() if loss is not None else 0.0
                    step += 1
                    # update progress
                    _ui_queue.put(("progress_batch", int(100 * (b_idx / max(1, num_batches)))))
                    _ui_queue.put(("progress_epoch", int(100 * ((epoch - 1 + b_idx / num_batches) / max(1, epochs)))))
                    _ui_queue.put(("progress_total", int(100 * (step / max(1, total_steps)))))
                if interrupted:
                    log("Training interrupted during epoch; exiting loop", to_ui=True)
                    break
                log(f"Epoch {epoch} finished, loss={epoch_loss / max(1, num_batches):.4f}", to_ui=True)
                # save checkpoint each epoch
                try:
                    ckpt_dir = os.path.join(outdir, f"checkpoint-epoch{epoch}")
                    ensure_dir(ckpt_dir)
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    meta_info = {"epoch": epoch, "step": step, "timestamp": time.time()}
                    with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w", encoding="utf-8") as mf:
                        json.dump(meta_info, mf, indent=2, ensure_ascii=False)
                    log(f"Checkpoint saved: {ckpt_dir}", to_ui=True)
                except Exception as e:
                    log(f"Failed to save checkpoint: {e}", to_ui=True)
                if self.stop_event.is_set():
                    break

            # final save
            try:
                model.save_pretrained(outdir)
                tokenizer.save_pretrained(outdir)
                final_meta = {"trained_epochs": epochs, "last_step": step, "timestamp": time.time()}
                with open(os.path.join(outdir, "checkpoint_meta.json"), "w", encoding="utf-8") as mf:
                    json.dump(final_meta, mf, indent=2, ensure_ascii=False)
                _ui_queue.put(("info", f"Training finished. Model saved to {outdir}"))
                log(f"Final model saved to {outdir}", to_ui=True)
            except Exception as e:
                log(f"Error saving final model: {e}", to_ui=True)
        else:
            # simulated training (UI-friendly)
            _ui_queue.put(("info", "Transformers/Torch nicht verfügbar oder CPU-forced — simuliere Training"))
            total_steps = epochs * 100
            cur = 0
            try:
                for epoch in range(1, epochs + 1):
                    if self.stop_event.is_set():
                        break
                    for batch in range(100):
                        if self.stop_event.is_set():
                            break
                        time.sleep(0.05)
                        cur += 1
                        pct_total = int(100 * (cur / total_steps))
                        _ui_queue.put(("progress_total", pct_total))
                        _ui_queue.put(("progress_epoch", int(100 * (epoch / epochs))))
                        _ui_queue.put(("progress_batch", int(100 * (batch / 100))))
                    # simulated checkpoint
                    ckpt_dir = os.path.join(outdir, f"sim-checkpoint-epoch{epoch}")
                    ensure_dir(ckpt_dir)
                    with open(os.path.join(ckpt_dir, "sim.txt"), "w", encoding="utf-8") as f:
                        f.write(f"Simulated checkpoint epoch {epoch}\n")
                    meta_info = {"epoch": epoch, "sim": True, "timestamp": time.time()}
                    with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w", encoding="utf-8") as mf:
                        json.dump(meta_info, mf, indent=2, ensure_ascii=False)
                    log(f"Simulated checkpoint saved: {ckpt_dir}", to_ui=True)
                    if self.stop_event.is_set():
                        break
                _ui_queue.put(("info", f"Simulation finished. Checkpoints saved in {outdir}"))
            except Exception as e:
                log(f"Simulation error: {e}", to_ui=True)
        # finalize
        self._finalize_training_ui()

    def _finalize_training_ui(self):
        try:
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
        except Exception:
            pass
        _ui_queue.put(("progress_total", 100))
        _ui_queue.put(("progress_epoch", 100))
        _ui_queue.put(("progress_batch", 100))
        if self.stop_event.is_set():
            self._append_log_ui("Training abgebrochen. Checkpoints gespeichert.")
            log("Training aborted by user", to_ui=True)
        else:
            self._append_log_ui("Training abgeschlossen.")
            log("Training finished normally", to_ui=True)

    def _collate_pad(self, batch):
        import torch as _torch
        input_ids = [b["input_ids"] for b in batch]
        maxlen = max(len(x) for x in input_ids)
        padded = []
        for ids in input_ids:
            arr = ids.tolist() if hasattr(ids, "tolist") else ids
            pad_len = maxlen - len(arr)
            padded.append(arr + [0] * pad_len)
        return {"input_ids": _torch.tensor(padded, dtype=_torch.long)}

    def _prepare_training_texts(self, max_len):
        all_texts = []
        for ds in self.datasets:
            path = ds.get("path")
            if not path or not os.path.exists(path):
                log(f"Dataset not ready: {path}", to_ui=True)
                continue
            fmt = ds.get("format", "json_text")
            m = ds.get("map", {})
            if fmt in ("json_text", "jsonl"):
                items = smart_read_json_lines(path)
                for it in items:
                    if isinstance(it, dict):
                        if m.get("text_key") and it.get(m["text_key"]):
                            all_texts.append(it[m["text_key"]])
                        elif it.get("text"):
                            all_texts.append(it["text"])
                        else:
                            all_texts.append(json.dumps(it, ensure_ascii=False))
                    else:
                        all_texts.append(str(it))
            elif fmt == "prompt_completion":
                items = smart_read_json_lines(path)
                pkey = m.get("prompt_key", "prompt")
                ckey = m.get("completion_key", "completion")
                for it in items:
                    if isinstance(it, dict):
                        p = it.get(pkey, "")
                        c = it.get(ckey, "")
                        merged = f"{p}\n{c}"
                        all_texts.append(merged)
                    else:
                        all_texts.append(str(it))
            elif fmt == "kv_json":
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        for k, v in data.items():
                            all_texts.append(f"{k}: {v}")
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                for k, v in item.items():
                                    all_texts.append(f"{k}: {v}")
                except Exception:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                all_texts.append(line)
            elif fmt == "plain":
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                    chunk_size = max(64, max_len * 3)
                    for i in range(0, len(txt), chunk_size):
                        all_texts.append(txt[i:i + chunk_size])
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_texts.append(line)
        out = [t.strip() for t in all_texts if t and len(t.strip()) > 5]
        return out

    # -------------------------
    # UI poller for queues
    # -------------------------
    def _start_ui_poller(self):
        self.root.after(200, self._ui_poller)

    def _ui_poller(self):
        # drain global _ui_queue first (messages from non-instance contexts)
        try:
            while True:
                typ, payload = _ui_queue.get_nowait()
                if typ == "log":
                    self._append_log_ui(payload)
                elif typ == "progress_download":
                    # unused visual for now
                    pass
                elif typ == "info":
                    self._append_log_ui(str(payload))
                elif typ == "error":
                    self._append_log_ui("ERROR: " + str(payload))
                else:
                    self._append_log_ui(str((typ, payload)))
        except queue.Empty:
            pass

        # drain instance UI queue (e.g., progress updates)
        try:
            while True:
                item = self.ui_queue.get_nowait()
                if not item:
                    continue
                typ = item[0]
                payload = item[1] if len(item) > 1 else None
                if typ == "dataset_update":
                    idx, text, dsid = payload
                    try:
                        self.lst_datasets.delete(idx)
                        self.lst_datasets.insert(idx, text)
                    except Exception:
                        pass
                elif typ == "ollama_refresh":
                    models = payload or []
                    try:
                        if CTK_AVAILABLE and hasattr(self, "cmb_ollama"):
                            self.cmb_ollama.configure(values=models)
                            if models:
                                self.cmb_ollama.set(models[0])
                    except Exception:
                        pass
                elif typ == "progress_total":
                    try:
                        self.pb_total.set(float(payload) / 100.0)
                    except Exception:
                        pass
                elif typ == "progress_epoch":
                    try:
                        self.pb_epoch.set(float(payload) / 100.0)
                    except Exception:
                        pass
                elif typ == "progress_batch":
                    try:
                        self.pb_batch.set(float(payload) / 100.0)
                    except Exception:
                        pass
                elif typ == "info":
                    self._append_log_ui(str(payload))
                elif typ == "error":
                    self._append_log_ui("ERROR: " + str(payload))
                elif typ == "log":
                    self._append_log_ui(str(payload))
                else:
                    self._append_log_ui(str(item))
        except queue.Empty:
            pass

        # schedule next poll
        try:
            self.root.after(200, self._ui_poller)
        except Exception:
            pass

    # -------------------------
    # Optimization helpers
    # -------------------------
    def _apply_lowmem_hint(self):
        """Adjust UI fields for low memory operation."""
        if self.low_memory_mode.get():
            try:
                self.spin_bs.delete(0, "end"); self.spin_bs.insert(0, "1")
                self.spin_maxlen.delete(0, "end"); self.spin_maxlen.insert(0, "128")
                self._append_log_ui("Low memory mode applied: batch=1, maxlen=128")
            except Exception:
                pass
        else:
            try:
                self.spin_bs.delete(0, "end"); self.spin_bs.insert(0, "4")
                self.spin_maxlen.delete(0, "end"); self.spin_maxlen.insert(0, "256")
                self._append_log_ui("Low memory mode disabled")
            except Exception:
                pass

    def _auto_tune_for_gpu(self):
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            messagebox.showinfo("No GPU", "Keine GPU erkannt oder torch nicht installiert.")
            return
        try:
            prop = torch.cuda.get_device_properties(0)
            total_mem = prop.total_memory  # bytes
            gb = total_mem / (1024 ** 3)
            # heuristics
            if gb < 6:
                bs = 1; maxlen = 128
            elif gb < 10:
                bs = 2; maxlen = 192
            elif gb < 16:
                bs = 4; maxlen = 256
            else:
                bs = 8; maxlen = 512
            self.spin_bs.delete(0, "end"); self.spin_bs.insert(0, str(bs))
            self.spin_maxlen.delete(0, "end"); self.spin_maxlen.insert(0, str(maxlen))
            self._append_log_ui(f"Auto-tune: GPU {prop.name} {gb:.1f}GB -> batch={bs}, maxlen={maxlen}")
        except Exception as e:
            messagebox.showerror("Auto-tune failed", str(e))
            log(f"Auto-tune failed: {e}", to_ui=True)

    # -------------------------
    # Config save
    # -------------------------
    def _save_config(self):
        name = ""
        try:
            name = self.ent_name.get().strip()
        except Exception:
            pass
        if not name:
            messagebox.showinfo("Missing", "Please set a model name first.")
            return
        safe = sanitize_name(name)
        outdir = ensure_dir(os.path.join(OUT_LLM_DIR, safe))
        meta = {"name": name, "author": (self.ent_author.get().strip() if hasattr(self, "ent_author") else ""), "base_model": (self.cmb_base.get().strip() if hasattr(self, "cmb_base") else ""), "datasets": self.datasets, "system_prompt": self.txt_sys.get("0.0", "end").strip() if hasattr(self, "txt_sys") else "", "version": self.version}
        try:
            with open(os.path.join(outdir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("Gespeichert", f"Konfiguration in {outdir} gespeichert.")
            log(f"Konfig gespeichert: {outdir}", to_ui=True)
        except Exception as e:
            messagebox.showerror("Fehler", f"Speichern fehlgeschlagen: {e}")
            log(f"Save config failed: {e}", to_ui=True)


# -------------------------
# Entrypoint
# -------------------------
def main():
    if not CTK_AVAILABLE:
        # Ask user to install customtkinter; we keep basic fallback but recommend CTK.
        root = tk.Tk()
        root.withdraw()
        try:
            res = messagebox.askyesno("customtkinter fehlt", "customtkinter ist nicht installiert. Möchtest du es jetzt installieren? (pip install customtkinter)\n\nJa -> Öffne Anleitung, Nein -> Starte einfache UI")
            if res:
                webbrowser.open("https://pypi.org/project/customtkinter/", new=2)
        except Exception:
            pass
        root.destroy()
    # create CTk root if available, else fallback to Tk
    try:
        if CTK_AVAILABLE:
            root = ctk.CTk()
        else:
            root = tk.Tk()
        app = QMTApp(root)
        root.mainloop()
    except Exception as e:
        log(f"Fatal UI error: {e}\n{traceback.format_exc()}", to_ui=False)
        print("Fatal UI error:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
