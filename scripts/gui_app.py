import os
import sys
import yaml
import queue
import tempfile
import threading
import subprocess
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

ROOT = Path(__file__).resolve().parents[1]

TR = {
    "ru": {
        "title": "LawnForge-NET GUI",
        "settings": "РќР°СЃС‚СЂРѕР№РєРё",
        "training": "РћР±СѓС‡РµРЅРёРµ",
        "inference": "РРЅС„РµСЂРµРЅСЃ",
        "status": "РЎС‚Р°С‚СѓСЃС‹",
        "logs": "РџРѕРґСЂРѕР±РЅС‹Р№ Р»РѕРі",
        "lang": "РЇР·С‹Рє",
        "import": "РРјРїРѕСЂС‚ YAML",
        "export": "Р­РєСЃРїРѕСЂС‚ YAML",
        "prepare": "РџРѕРґРіРѕС‚РѕРІРёС‚СЊ",
        "train": "РћР±СѓС‡РёС‚СЊ",
        "prepare_train": "РџРѕРґРіРѕС‚РѕРІРёС‚СЊ+РћР±СѓС‡РёС‚СЊ",
        "start": "РЎС‚Р°СЂС‚ batch",
        "stop": "РЎС‚РѕРї",
        "add": "Р”РѕР±Р°РІРёС‚СЊ",
        "remove": "РЈРґР°Р»РёС‚СЊ РІС‹Р±СЂР°РЅРЅС‹Рµ",
        "clear": "РћС‡РёСЃС‚РёС‚СЊ",
        "torch": "Torch",
        "cuda": "CUDA",
        "model": "РњРѕРґРµР»СЊ Р·Р°РіСЂСѓР¶РµРЅР°",
        "refresh": "РћР±РЅРѕРІРёС‚СЊ СЃС‚Р°С‚СѓСЃС‹",
        "minimal": "РўРѕР»СЊРєРѕ LAZ+GPKG",
        "no_files": "Р”РѕР±Р°РІСЊС‚Рµ С„Р°Р№Р»С‹ РґР»СЏ РёРЅС„РµСЂРµРЅСЃР°",
        "busy": "РЈР¶Рµ РІС‹РїРѕР»РЅСЏРµС‚СЃСЏ Р·Р°РґР°С‡Р°",
        "open_settings": "РќР°СЃС‚СЂРѕР№РєРё",
        "close": "Р—Р°РєСЂС‹С‚СЊ",
    },
    "en": {
        "title": "LawnForge-NET GUI",
        "settings": "Settings",
        "training": "Training",
        "inference": "Inference",
        "status": "Status",
        "logs": "Detailed log",
        "lang": "Language",
        "import": "Import YAML",
        "export": "Export YAML",
        "prepare": "Prepare",
        "train": "Train",
        "prepare_train": "Prepare+Train",
        "start": "Start batch",
        "stop": "Stop",
        "add": "Add",
        "remove": "Remove selected",
        "clear": "Clear",
        "torch": "Torch",
        "cuda": "CUDA",
        "model": "Model loaded",
        "refresh": "Refresh status",
        "minimal": "Only LAZ+GPKG",
        "no_files": "Add input files",
        "busy": "Task is already running",
        "open_settings": "Settings",
        "close": "Close",
    },
}

DEFAULT = {
    "seed": 42,
    "raw_train": str(ROOT / "data/raw/train"),
    "raw_val": str(ROOT / "data/raw/val"),
    "prepared_train": str(ROOT / "data/prepared/train"),
    "prepared_val": str(ROOT / "data/prepared/val"),
    "models": str(ROOT / "artifacts/models"),
    "cell_size": "0.15",
    "tile_size_m": "50.0",
    "tile_overlap_m": "2.0",
    "min_points_per_cell": "1",
    "channels": "count,z_min,z_max,z_mean,z_std,intensity_mean,intensity_std,ndhm",
    "batch_size": "8",
    "num_workers": "0",
    "epochs_stage2": "35",
    "lr": "0.001",
    "weight_decay": "0.0001",
    "amp": True,
    "bSloopSmooth": True,
    "cloth_resolution": "0.2",
    "rigidness": "3",
    "class_threshold": "0.35",
    "time_step": "0.65",
    "interations": "700",
    "inference_chunk_points": "2000000",
    "min_lawn_polygon_area_m2": "1.0",
    "weights_out": str(ROOT / "artifacts/models/stage2_lawn_vs_nonlawn.pt"),
    "weights_in": str(ROOT / "artifacts/models/stage2_lawn_vs_nonlawn.pt"),
    "out_root": str(ROOT / "outputs"),
    "minimal_outputs": True,
}


def ts():
    return datetime.now().strftime("%H:%M:%S")


class App:
    def __init__(self, root):
        self.root = root
        self.lang = tk.StringVar(value="ru")
        self.vars = {}
        for k, v in DEFAULT.items():
            self.vars[k] = tk.BooleanVar(value=v) if isinstance(v, bool) else tk.StringVar(value=str(v))
        self.log_q = queue.Queue()
        self.stop_flag = threading.Event()
        self.proc = None
        self.worker = None
        self.files = []
        self.settings_win = None
        self._build()
        self.root.after(100, self._drain)
        self.refresh_status()

    def t(self, k):
        return TR[self.lang.get()].get(k, k)

    def _log(self, m):
        self.log_q.put(f"[{ts()}] {m}")

    def _screen_geometry(self):
        w = self.root.winfo_screenwidth()
        h = self.root.winfo_screenheight()
        width = max(1100, int(w * 0.9))
        height = max(800, int(h * 0.85))
        x = int((w - width) / 2)
        y = int((h - height) / 2)
        return f"{width}x{height}+{x}+{y}"

    def _build(self):
        self.root.title(self.t("title"))
        self.root.geometry(self._screen_geometry())
        self.root.minsize(1100, 800)

        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(top, text=self.t("lang")).pack(side=tk.LEFT)
        cmb = ttk.Combobox(top, textvariable=self.lang, values=["ru", "en"], width=6, state="readonly")
        cmb.pack(side=tk.LEFT, padx=6)
        cmb.bind("<<ComboboxSelected>>", lambda e: self._rebuild())

        ttk.Button(top, text=self.t("open_settings"), command=self.open_settings).pack(side=tk.RIGHT, padx=6)

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.tab_tr = ttk.Frame(self.nb)
        self.tab_inf = ttk.Frame(self.nb)
        self.tab_st = ttk.Frame(self.nb)
        self.nb.add(self.tab_tr, text=self.t("training"))
        self.nb.add(self.tab_inf, text=self.t("inference"))
        self.nb.add(self.tab_st, text=self.t("status"))
        self._train_tab()
        self._infer_tab()
        self._status_tab()

        lf = ttk.LabelFrame(self.root, text=self.t("logs"))
        lf.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.log = ScrolledText(lf, height=18)
        self.log.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.log.configure(state=tk.DISABLED)

    def _rebuild(self):
        for w in self.root.winfo_children():
            w.destroy()
        if self.settings_win:
            self.settings_win.destroy()
            self.settings_win = None
        self._build()
        self.refresh_status()

    def open_settings(self):
        if self.settings_win and self.settings_win.winfo_exists():
            self.settings_win.lift()
            return
        self.settings_win = tk.Toplevel(self.root)
        self.settings_win.title(self.t("settings"))
        self.settings_win.geometry("900x700")
        self.settings_win.minsize(900, 700)
        self._settings_tab(self.settings_win)

    def _add_row(self, parent, r, label, var, browse=None):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=5, pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=r, column=1, sticky="ew", padx=5, pady=4)
        if browse:
            ttk.Button(parent, text="...", width=3, command=browse).grid(row=r, column=2, padx=5, pady=4)

    def _settings_tab(self, host):
        host.columnconfigure(1, weight=1)
        fields = [
            ("seed", "seed"),
            ("raw_train", "raw_train"),
            ("raw_val", "raw_val"),
            ("prepared_train", "prepared_train"),
            ("prepared_val", "prepared_val"),
            ("models", "models"),
            ("cell_size", "cell_size"),
            ("tile_size_m", "tile_size_m"),
            ("tile_overlap_m", "tile_overlap_m"),
            ("min_points_per_cell", "min_points_per_cell"),
            ("channels", "channels"),
            ("batch_size", "batch_size"),
            ("num_workers", "num_workers"),
            ("epochs_stage2", "epochs_stage2"),
            ("lr", "lr"),
            ("weight_decay", "weight_decay"),
            ("cloth_resolution", "cloth_resolution"),
            ("rigidness", "rigidness"),
            ("class_threshold", "class_threshold"),
            ("time_step", "time_step"),
            ("interations", "interations"),
            ("inference_chunk_points", "inference_chunk_points"),
            ("min_lawn_polygon_area_m2", "min_lawn_polygon_area_m2"),
        ]
        for i, (k, lab) in enumerate(fields):
            self._add_row(host, i, lab, self.vars[k])
        ttk.Checkbutton(host, text="amp", variable=self.vars["amp"]).grid(row=len(fields), column=0, sticky="w", padx=5, pady=3)
        ttk.Checkbutton(host, text="bSloopSmooth", variable=self.vars["bSloopSmooth"]).grid(row=len(fields)+1, column=0, sticky="w", padx=5, pady=3)
        b = ttk.Frame(host)
        b.grid(row=len(fields)+2, column=0, columnspan=3, sticky="w", padx=5, pady=8)
        ttk.Button(b, text=self.t("import"), command=self.import_yaml).pack(side=tk.LEFT, padx=4)
        ttk.Button(b, text=self.t("export"), command=self.export_yaml).pack(side=tk.LEFT, padx=4)
        ttk.Button(b, text=self.t("close"), command=self.settings_win.destroy).pack(side=tk.LEFT, padx=4)

    def _train_tab(self):
        f = self.tab_tr
        f.columnconfigure(1, weight=1)
        self._add_row(f, 0, "weights_out", self.vars["weights_out"], lambda: self._browse_save(self.vars["weights_out"], ".pt"))
        b = ttk.Frame(f)
        b.grid(row=1, column=0, columnspan=3, sticky="w", padx=5, pady=8)
        ttk.Button(b, text=self.t("prepare"), command=self.run_prepare).pack(side=tk.LEFT, padx=4)
        ttk.Button(b, text=self.t("train"), command=self.run_train).pack(side=tk.LEFT, padx=4)
        ttk.Button(b, text=self.t("prepare_train"), command=self.run_prepare_train).pack(side=tk.LEFT, padx=4)

    def _infer_tab(self):
        f = self.tab_inf
        f.columnconfigure(1, weight=1)
        f.rowconfigure(3, weight=1)
        self._add_row(f, 0, "weights_in", self.vars["weights_in"], lambda: self._browse_open(self.vars["weights_in"], [("PyTorch", "*.pt"), ("All", "*.*")]))
        self._add_row(f, 1, "out_root", self.vars["out_root"], lambda: self._browse_dir(self.vars["out_root"]))
        ttk.Checkbutton(f, text=self.t("minimal"), variable=self.vars["minimal_outputs"]).grid(row=2, column=0, sticky="w", padx=5, pady=4)
        box = ttk.LabelFrame(f, text="files")
        box.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)
        self.listbox = tk.Listbox(box, selectmode=tk.EXTENDED)
        self.listbox.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        ttk.Button(box, text=self.t("add"), command=self.add_files).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Button(box, text=self.t("remove"), command=self.remove_files).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        ttk.Button(box, text=self.t("clear"), command=self.clear_files).grid(row=1, column=2, sticky="w", padx=5, pady=5)
        b = ttk.Frame(f)
        b.grid(row=4, column=0, columnspan=3, sticky="w", padx=5, pady=8)
        ttk.Button(b, text=self.t("start"), command=self.run_batch).pack(side=tk.LEFT, padx=4)
        ttk.Button(b, text=self.t("stop"), command=self.stop).pack(side=tk.LEFT, padx=4)

    def _status_tab(self):
        f = self.tab_st
        self.s_torch = tk.StringVar()
        self.s_cuda = tk.StringVar()
        self.s_model = tk.StringVar()
        ttk.Label(f, textvariable=self.s_torch).pack(anchor="w", padx=8, pady=6)
        ttk.Label(f, textvariable=self.s_cuda).pack(anchor="w", padx=8, pady=6)
        ttk.Label(f, textvariable=self.s_model).pack(anchor="w", padx=8, pady=6)
        ttk.Button(f, text=self.t("refresh"), command=self.refresh_status).pack(anchor="w", padx=8, pady=8)

    def _cfg(self):
        ch = [x.strip() for x in self.vars["channels"].get().split(",") if x.strip()]
        return {
            "seed": int(self.vars["seed"].get()),
            "paths": {
                "raw_train": self.vars["raw_train"].get(),
                "raw_val": self.vars["raw_val"].get(),
                "prepared_train": self.vars["prepared_train"].get(),
                "prepared_val": self.vars["prepared_val"].get(),
                "models": self.vars["models"].get(),
            },
            "raster": {
                "cell_size": float(self.vars["cell_size"].get()),
                "tile_size_m": float(self.vars["tile_size_m"].get()),
                "tile_overlap_m": float(self.vars["tile_overlap_m"].get()),
                "min_points_per_cell": int(self.vars["min_points_per_cell"].get()),
            },
            "features": {"channels": ch},
            "training": {
                "batch_size": int(self.vars["batch_size"].get()),
                "num_workers": int(self.vars["num_workers"].get()),
                "epochs_stage2": int(self.vars["epochs_stage2"].get()),
                "lr": float(self.vars["lr"].get()),
                "weight_decay": float(self.vars["weight_decay"].get()),
                "amp": bool(self.vars["amp"].get()),
            },
            "stage2": {"valid_labels": [1, 2]},
            "csf": {
                "bSloopSmooth": bool(self.vars["bSloopSmooth"].get()),
                "cloth_resolution": float(self.vars["cloth_resolution"].get()),
                "rigidness": int(self.vars["rigidness"].get()),
                "class_threshold": float(self.vars["class_threshold"].get()),
                "time_step": float(self.vars["time_step"].get()),
                "interations": int(self.vars["interations"].get()),
            },
            "inference": {
                "inference_chunk_points": int(self.vars["inference_chunk_points"].get()),
                "min_lawn_polygon_area_m2": float(self.vars["min_lawn_polygon_area_m2"].get()),
            },
        }

    def _write_tmp_cfg(self):
        fd, p = tempfile.mkstemp(prefix="25d_cfg_", suffix=".yaml")
        os.close(fd)
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._cfg(), f, allow_unicode=True, sort_keys=False)
        return p

    def import_yaml(self):
        p = filedialog.askopenfilename(filetypes=[("YAML", "*.yaml"), ("All", "*.*")])
        if not p:
            return
        with open(p, "r", encoding="utf-8") as f:
            c = yaml.safe_load(f)
        try:
            self.vars["seed"].set(str(c["seed"]))
            self.vars["raw_train"].set(c["paths"]["raw_train"])
            self.vars["raw_val"].set(c["paths"]["raw_val"])
            self.vars["prepared_train"].set(c["paths"]["prepared_train"])
            self.vars["prepared_val"].set(c["paths"]["prepared_val"])
            self.vars["models"].set(c["paths"]["models"])
            self.vars["cell_size"].set(str(c["raster"]["cell_size"]))
            self.vars["tile_size_m"].set(str(c["raster"]["tile_size_m"]))
            self.vars["tile_overlap_m"].set(str(c["raster"]["tile_overlap_m"]))
            self.vars["min_points_per_cell"].set(str(c["raster"]["min_points_per_cell"]))
            self.vars["channels"].set(",".join(c["features"]["channels"]))
            self.vars["batch_size"].set(str(c["training"]["batch_size"]))
            self.vars["num_workers"].set(str(c["training"]["num_workers"]))
            self.vars["epochs_stage2"].set(str(c["training"]["epochs_stage2"]))
            self.vars["lr"].set(str(c["training"]["lr"]))
            self.vars["weight_decay"].set(str(c["training"]["weight_decay"]))
            self.vars["amp"].set(bool(c["training"]["amp"]))
            self.vars["bSloopSmooth"].set(bool(c["csf"]["bSloopSmooth"]))
            self.vars["cloth_resolution"].set(str(c["csf"]["cloth_resolution"]))
            self.vars["rigidness"].set(str(c["csf"]["rigidness"]))
            self.vars["class_threshold"].set(str(c["csf"]["class_threshold"]))
            self.vars["time_step"].set(str(c["csf"]["time_step"]))
            self.vars["interations"].set(str(c["csf"]["interations"]))
            self.vars["inference_chunk_points"].set(str(c["inference"]["inference_chunk_points"]))
            self.vars["min_lawn_polygon_area_m2"].set(str(c["inference"]["min_lawn_polygon_area_m2"]))
        except Exception as e:
            self._log(f"Import parse warning: {e}")
        self._log(f"Imported config: {p}")

    def export_yaml(self):
        p = filedialog.asksaveasfilename(defaultextension=".yaml", filetypes=[("YAML", "*.yaml"), ("All", "*.*")])
        if not p:
            return
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._cfg(), f, allow_unicode=True, sort_keys=False)
        self._log(f"Exported config: {p}")

    def _browse_open(self, var, filt):
        p = filedialog.askopenfilename(filetypes=filt)
        if p:
            var.set(p)
            self.refresh_status()

    def _browse_save(self, var, ext):
        p = filedialog.asksaveasfilename(defaultextension=ext, filetypes=[("All", "*.*")])
        if p:
            var.set(p)

    def _browse_dir(self, var):
        p = filedialog.askdirectory()
        if p:
            var.set(p)

    def add_files(self):
        ps = filedialog.askopenfilenames(filetypes=[("Point clouds", "*.laz *.las *.npz *.csv"), ("All", "*.*")])
        for p in ps:
            self.files.append(p)
            self.listbox.insert(tk.END, p)
        self._log(f"Added files: {len(ps)}")

    def remove_files(self):
        idx = list(self.listbox.curselection())
        idx.reverse()
        for i in idx:
            del self.files[i]
            self.listbox.delete(i)

    def clear_files(self):
        self.files = []
        self.listbox.delete(0, tk.END)

    def refresh_status(self):
        ok_t, ok_c, ok_m = False, False, False
        try:
            import torch  # type: ignore
            ok_t = True
            ok_c = bool(torch.cuda.is_available())
            w = self.vars["weights_in"].get().strip()
            if w and Path(w).exists():
                ck = torch.load(w, map_location="cpu")
                ok_m = isinstance(ck, dict) and "model_state" in ck
        except Exception:
            pass
        self.s_torch.set(f"{self.t('torch')}: {'OK' if ok_t else 'FAIL'}")
        self.s_cuda.set(f"{self.t('cuda')}: {'OK' if ok_c else 'FAIL'}")
        self.s_model.set(f"{self.t('model')}: {'OK' if ok_m else 'FAIL'}")

    def _drain(self):
        try:
            while True:
                s = self.log_q.get_nowait()
                self.log.configure(state=tk.NORMAL)
                self.log.insert(tk.END, s + "\n")
                self.log.see(tk.END)
                self.log.configure(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.root.after(100, self._drain)

    def _run(self, cmd):
        self._log("CMD: " + " ".join(cmd))
        self.proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in self.proc.stdout:
            self._log(line.rstrip())
            if self.stop_flag.is_set():
                break
        rc = self.proc.wait()
        self.proc = None
        return -1 if self.stop_flag.is_set() else rc

    def _ensure_idle(self):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Busy", self.t("busy"))
            return False
        self.stop_flag.clear()
        return True

    def run_prepare(self):
        if not self._ensure_idle():
            return
        self.worker = threading.Thread(target=self._w_prepare, daemon=True)
        self.worker.start()

    def run_train(self):
        if not self._ensure_idle():
            return
        self.worker = threading.Thread(target=self._w_train, daemon=True)
        self.worker.start()

    def run_prepare_train(self):
        if not self._ensure_idle():
            return
        self.worker = threading.Thread(target=self._w_prepare_train, daemon=True)
        self.worker.start()

    def run_batch(self):
        if not self._ensure_idle():
            return
        if not self.files:
            messagebox.showwarning("No files", self.t("no_files"))
            return
        self.worker = threading.Thread(target=self._w_batch, daemon=True)
        self.worker.start()

    def _w_prepare(self):
        c = self._write_tmp_cfg()
        self._log(f"Temp config: {c}")
        self._log(f"prepare rc={self._run([sys.executable, 'scripts/prepare_dataset.py', '--config', c])}")

    def _w_train(self):
        c = self._write_tmp_cfg()
        self._log(f"Temp config: {c}")
        cmd = [sys.executable, "scripts/train_stage2.py", "--config", c, "--weights-out", self.vars["weights_out"].get().strip()]
        self._log(f"train rc={self._run(cmd)}")
        self.refresh_status()

    def _w_prepare_train(self):
        c = self._write_tmp_cfg()
        self._log(f"Temp config: {c}")
        rc = self._run([sys.executable, "scripts/prepare_dataset.py", "--config", c])
        self._log(f"prepare rc={rc}")
        if rc == 0 and not self.stop_flag.is_set():
            rc2 = self._run([sys.executable, "scripts/train_stage2.py", "--config", c, "--weights-out", self.vars["weights_out"].get().strip()])
            self._log(f"train rc={rc2}")
        self.refresh_status()

    def _w_batch(self):
        c = self._write_tmp_cfg()
        self._log(f"Temp config: {c}")
        out_root = Path(self.vars["out_root"].get().strip())
        out_root.mkdir(parents=True, exist_ok=True)
        w = self.vars["weights_in"].get().strip()
        minimal = bool(self.vars["minimal_outputs"].get())
        for i, s in enumerate(self.files, 1):
            if self.stop_flag.is_set():
                self._log("stopped by user")
                return
            out = out_root / Path(s).stem
            out.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "scripts/infer_scene.py", "--config", c, "--scene", s, "--out-dir", str(out), "--weights", w]
            if minimal:
                cmd.append("--minimal-outputs")
            rc = self._run(cmd)
            self._log(f"[{i}/{len(self.files)}] {Path(s).name} rc={rc}")
            if rc != 0:
                return

    def stop(self):
        self.stop_flag.set()
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

