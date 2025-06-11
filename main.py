import os
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import time
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import shutil
import platform
import subprocess
from twilio.rest import Client
import requests
import webbrowser
from dotenv import load_dotenv

# twilio config

load_dotenv()  # loads variables from .env into environment

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
OWNER_WHATSAPP_TO = os.environ.get("OWNER_WHATSAPP_TO", "whatsapp:+918700843968")

def upload_to_tempsh(file_path):
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post('https://temp.sh/upload', files=files, timeout=30)
        if response.status_code == 200 and response.text.strip():
            url = response.text.strip()
            return url
        else:
            return None
    except Exception:
        return None

def send_whatsapp_alert(message, media_url=None):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        kwargs = {
            "body": message,
            "from_": TWILIO_WHATSAPP_FROM,
            "to": OWNER_WHATSAPP_TO
        }
        client.messages.create(**kwargs)
    except Exception:
        pass

# app config
OUTPUT_DIR = "captured_videos"
VIOLENT_DIR = os.path.join(OUTPUT_DIR, "violent")
TEMP_VIDEO_PATH = os.path.join(OUTPUT_DIR, "temp_clip.mp4")
MODEL_PATH = "crime_detection_model.tflite"
ASSETS_DIR = "assets"
LOGO_PATH = os.path.join(ASSETS_DIR, "watchdog_logo.png")
ICON_PATH = os.path.join(ASSETS_DIR, "watchdog_icon.ico")

DEFAULTS = {
    "camera_id": 0,
    "violence_threshold": 85.0,
    "clip_duration": 5,
    "resolution": "640x480",
    "output_dir": OUTPUT_DIR
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIOLENT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def extract_frames(video_path, num_frames=16):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        frame_count += 1
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    return np.array(frames)

def preprocess_frame(frame):
    frame = frame.astype(np.float32) / 255.0
    frame -= np.array([0.485, 0.456, 0.406])
    frame /= np.array([0.229, 0.224, 0.225])
    return frame

def process_video_for_inference(video_path, num_frames=16):
    frames = extract_frames(video_path, num_frames)
    processed_frames = np.array([preprocess_frame(frame) for frame in frames])
    processed_frames = np.expand_dims(processed_frames, axis=0)
    return processed_frames, frames

def predict_crime_tflite(interpreter, video_path):
    frames, _ = process_video_for_inference(video_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], frames)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_class = np.argmax(output_data[0])
    confidence = float(output_data[0][pred_class] * 100)
    prediction = "Violent" if pred_class == 1 else "Non-violent"
    return prediction, confidence

class VideoProcessor:
    def __init__(self, model_path, settings, update_callback=None, status_callback=None, mini_update_callback=None):
        self.interpreter = load_tflite_model(model_path)
        self.cap = None
        self.video_queue = queue.Queue()
        self.is_running = False
        self.processing_thread = None
        self.status_lock = threading.Lock()
        self.update_callback = update_callback
        self.status_callback = status_callback
        self.mini_update_callback = mini_update_callback
        self.alert_history = []
        self.settings = settings
        self.consecutive_violence_count = 0
        self.prolonged_alert_sent = False
        self.violence_detected = False
        self.violence_confidence = 0.0
        self.last_violence_time = 0
        self.overlay_active = False

    def start_capture(self):
        camera_id = self.settings["camera_id"]
        frame_width, frame_height = map(int, self.settings["resolution"].split("x"))
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            if self.status_callback:
                self.status_callback(f"Error: Failed to open camera {camera_id}")
            return False
        self.is_running = True
        self.processing_thread = threading.Thread(target=self.process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        return True

    def process_queue(self):
        while self.is_running:
            try:
                if not self.video_queue.empty():
                    video_path, timestamp = self.video_queue.get()
                    prediction, confidence = predict_crime_tflite(self.interpreter, video_path)
                    with self.status_lock:
                        if prediction == "Violent" and confidence >= self.settings['violence_threshold']:
                            self.consecutive_violence_count += 1
                            self.violence_detected = True
                            self.violence_confidence = confidence
                            self.last_violence_time = time.time()
                            self.overlay_active = True
                            # save violent video
                            violent_path = os.path.join(VIOLENT_DIR, f"{timestamp}_violent_{confidence:.1f}.mp4")
                            shutil.copy2(video_path, violent_path)
                            self.alert_history.append({
                                "timestamp": timestamp,
                                "confidence": confidence,
                                "path": violent_path
                            })
                            if self.mini_update_callback:
                                self.mini_update_callback(f"VIOLENCE DETECTED! ({confidence:.1f}%) at {timestamp}")
                            # prolonged alert
                            if self.consecutive_violence_count >= 2 and not self.prolonged_alert_sent:
                                temp_url = upload_to_tempsh(violent_path)
                                alert_msg = (
                                    "ALERT: Prolonged violence detected by Watchdog AI!\n"
                                    f"Time: {timestamp}\n"
                                    f"Confidence: {confidence:.1f}%\n"
                                    "Here is the video available for preview:\n"
                                    f"{temp_url if temp_url else '[Upload failed]'}"
                                )
                                send_whatsapp_alert(alert_msg, media_url=temp_url if temp_url else None)
                                self.prolonged_alert_sent = True
                        else:
                            self.consecutive_violence_count = 0
                            self.prolonged_alert_sent = False
                            self.violence_detected = False
                            self.violence_confidence = 0.0
                            if self.mini_update_callback:
                                self.mini_update_callback(f"Normal activity at {timestamp}")
                    self.video_queue.task_done()
                else:
                    time.sleep(0.1)
            except Exception:
                time.sleep(1)

    def capture_and_save(self):
        frame_width, frame_height = map(int, self.settings["resolution"].split("x"))
        fps = 30
        clip_duration = self.settings["clip_duration"]
        frames_per_clip = clip_duration * fps
        frame_count = 0
        current_frames = []
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                if self.status_callback:
                    self.status_callback("Error: Failed to read from camera")
                break
            current_frames.append(frame)
            frame_count += 1
            display_frame = frame.copy()
            now = time.time()
            # overlay logic
            with self.status_lock:
                violence_status = self.violence_detected
                violence_conf = self.violence_confidence
                overlay_active = self.overlay_active
                if overlay_active and (now - self.last_violence_time > 15):
                    self.overlay_active = False
                    overlay_active = False
            # draw overlay
            if overlay_active:
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
                cv2.putText(display_frame, f"VIOLENCE DETECTED! ({violence_conf:.1f}%)",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # mini status
            mini_status = ""
            if overlay_active:
                mini_status = f"VIOLENCE DETECTED! ({violence_conf:.1f}%)"
            else:
                mini_status = "Normal activity"
            cv2.putText(display_frame, mini_status, (10, frame_height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display_frame, timestamp, (10, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if self.update_callback and frame_count % 5 == 0:
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                self.update_callback(rgb_frame)
            if frame_count >= frames_per_clip:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(TEMP_VIDEO_PATH, fourcc, fps,
                                     (current_frames[0].shape[1], current_frames[0].shape[0]))
                for f in current_frames:
                    out.write(f)
                out.release()
                self.video_queue.put((TEMP_VIDEO_PATH, ts))
                frame_count = 0
                current_frames = []
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.stop()

    def stop(self):
        self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=3.0)
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.status_callback:
            self.status_callback("Surveillance stopped")

    def get_alert_history(self):
        return self.alert_history

def open_folder(path):
    path = os.path.abspath(path)
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])

class WatchdogApp:
    def __init__(self, root):
        self.root = root
        self.root.title("WatchDog AI - Surveillance System")
        self.root.geometry("1100x700")
        self.root.configure(bg="#181A1B")
        try:
            if os.path.exists(ICON_PATH):
                self.root.iconbitmap(ICON_PATH)
        except Exception:
            pass
        self.settings = DEFAULTS.copy()
        self.video_processor = None
        self.capture_thread = None
        self.is_running = False
        self.current_frame = None
        self.status_var = tk.StringVar(value="Ready")
        self.mini_update_var = tk.StringVar(value="System ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, anchor=tk.W, style="Status.TLabel")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.mini_update_label = tk.Label(self.root, textvariable=self.mini_update_var,
                                          font=("Segoe UI", 10), bg="#222", fg="#fff", anchor="w")
        self.mini_update_label.place(x=10, y=self.root.winfo_height()-50, anchor="sw")
        self.root.bind("<Configure>", self._on_resize)
        self.setup_style()
        self.create_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _on_resize(self, event):
        self.mini_update_label.place(x=10, y=self.root.winfo_height()-50, anchor="sw")

    def setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#181A1B")
        style.configure("TLabel", background="#181A1B", foreground="#E4E6EB", font=("Segoe UI", 12))
        style.configure("Header.TLabel", background="#181A1B", foreground="#E4E6EB", font=("Segoe UI", 22, "bold"))
        style.configure("TButton", font=("Segoe UI", 12, "bold"), background="#23272A", foreground="#E4E6EB", borderwidth=0, relief="flat")
        style.configure("Status.TLabel", background="#222", foreground="#E4E6EB", font=("Segoe UI", 10))
        style.configure("Treeview", background="#222", foreground="#E4E6EB", rowheight=28, fieldbackground="#222", borderwidth=0)
        style.configure("Treeview.Heading", background="#23272A", foreground="#E4E6EB", font=("Segoe UI", 12, "bold"))
        style.map("TButton", background=[("active", "#2C2F33")])

    def create_layout(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.tab_monitor = ttk.Frame(self.notebook)
        self.tab_settings = ttk.Frame(self.notebook)
        self.tab_alerts = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_monitor, text="Monitor")
        self.notebook.add(self.tab_settings, text="Settings")
        self.notebook.add(self.tab_alerts, text="Alerts")
        self.create_monitor_tab()
        self.create_settings_tab()
        self.create_alerts_tab()

    def create_monitor_tab(self):
        f = self.tab_monitor
        header = ttk.Frame(f)
        header.pack(fill=tk.X, pady=10)
        if os.path.exists(LOGO_PATH):
            logo_img = Image.open(LOGO_PATH).resize((90, 90), Image.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            logo_label = ttk.Label(header, image=self.logo_photo, background="#181A1B")
            logo_label.pack(side=tk.LEFT, padx=(15, 20))
        ttk.Label(header, text="WATCHDOG AI", style="Header.TLabel").pack(side=tk.LEFT, anchor=tk.NW)
        self.video_label = ttk.Label(f)
        self.video_label.pack(pady=25)
        self.set_default_video_frame()
        controls = ttk.Frame(f)
        controls.pack(pady=10)
        self.start_btn_text = tk.StringVar(value="Start Surveillance")
        self.start_btn = ttk.Button(controls, textvariable=self.start_btn_text, command=self.toggle_surveillance, width=20)
        self.start_btn.pack(side=tk.LEFT, padx=6)
        ttk.Button(controls, text="Open Violent Videos Folder", command=lambda: open_folder(VIOLENT_DIR)).pack(side=tk.LEFT, padx=6)

    def set_default_video_frame(self):
        default_img = np.ones((320, 540, 3), dtype=np.uint8) * 34
        cv2.putText(default_img, "Live feed will appear here", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 180), 2)
        self.update_video_frame(cv2.cvtColor(default_img, cv2.COLOR_BGR2RGB))

    def create_settings_tab(self):
        f = self.tab_settings
        cam_frame = ttk.LabelFrame(f, text="Camera", padding=12)
        cam_frame.pack(fill=tk.X, pady=12, padx=12)
        ttk.Label(cam_frame, text="Camera ID:").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.camera_id_var = tk.IntVar(value=self.settings["camera_id"])
        ttk.Spinbox(cam_frame, from_=0, to=10, textvariable=self.camera_id_var, width=6).grid(row=0, column=1, sticky=tk.W, padx=4)
        ttk.Button(cam_frame, text="Test Camera", command=self.test_camera).grid(row=0, column=2, padx=8)
        det_frame = ttk.LabelFrame(f, text="Detection", padding=12)
        det_frame.pack(fill=tk.X, pady=12, padx=12)
        ttk.Label(det_frame, text="Violence Threshold (%):").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.threshold_var = tk.DoubleVar(value=self.settings["violence_threshold"])
        ttk.Scale(det_frame, from_=50, to=99, variable=self.threshold_var, orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=4)
        ttk.Label(det_frame, textvariable=self.threshold_var).grid(row=0, column=2, padx=4)
        vid_frame = ttk.LabelFrame(f, text="Video", padding=12)
        vid_frame.pack(fill=tk.X, pady=12, padx=12)
        ttk.Label(vid_frame, text="Clip Duration (s):").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.duration_var = tk.IntVar(value=self.settings["clip_duration"])
        ttk.Spinbox(vid_frame, from_=1, to=30, textvariable=self.duration_var, width=6).grid(row=0, column=1, padx=4)
        ttk.Label(vid_frame, text="Resolution:").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.resolution_var = tk.StringVar(value=self.settings["resolution"])
        ttk.Combobox(vid_frame, values=["640x480", "800x600", "1280x720", "1920x1080"],
                     textvariable=self.resolution_var, width=12, state="readonly").grid(row=1, column=1, padx=4)
        store_frame = ttk.LabelFrame(f, text="Storage", padding=12)
        store_frame.pack(fill=tk.X, pady=12, padx=12)
        ttk.Label(store_frame, text="Output Directory:").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.output_dir_var = tk.StringVar(value=self.settings["output_dir"])
        ttk.Entry(store_frame, textvariable=self.output_dir_var, width=30).grid(row=0, column=1, padx=4)
        ttk.Button(store_frame, text="Open", command=lambda: open_folder(self.output_dir_var.get())).grid(row=0, column=2, padx=4)
        ttk.Button(f, text="Apply Settings", command=self.save_settings).pack(pady=18)

    def create_alerts_tab(self):
        f = self.tab_alerts
        paned = ttk.PanedWindow(f, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=8, padx=8)
        left = ttk.Frame(paned)
        paned.add(left, weight=3)
        stats_frame = ttk.LabelFrame(left, text="Alert Statistics")
        stats_frame.pack(fill=tk.X, pady=5, padx=5)
        self.stats_var = tk.StringVar(value="Total Alerts: 0 | Today: 0")
        ttk.Label(stats_frame, textvariable=self.stats_var).pack(pady=5)
        alerts_frame = ttk.LabelFrame(left, text="Detected Incidents")
        alerts_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        filter_frame = ttk.Frame(alerts_frame)
        filter_frame.pack(fill=tk.X, pady=5)
        ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT, padx=5)
        self.filter_var = tk.StringVar()
        ttk.Entry(filter_frame, textvariable=self.filter_var, width=25).pack(side=tk.LEFT, padx=5)
        ttk.Button(filter_frame, text="Search", command=self.filter_alerts).pack(side=tk.LEFT, padx=5)
        ttk.Button(filter_frame, text="Clear", command=lambda: [self.filter_var.set(""), self.refresh_alerts()]).pack(side=tk.LEFT)
        tree_frame = ttk.Frame(alerts_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        ysb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        xsb = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        xsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.alerts_tree = ttk.Treeview(
            tree_frame, columns=("timestamp", "confidence", "path"),
            show="headings", yscrollcommand=ysb.set, xscrollcommand=xsb.set
        )
        ysb.config(command=self.alerts_tree.yview)
        xsb.config(command=self.alerts_tree.xview)
        self.alerts_tree.heading("timestamp", text="Timestamp")
        self.alerts_tree.heading("confidence", text="Confidence")
        self.alerts_tree.heading("path", text="Video File")
        self.alerts_tree.column("timestamp", width=150)
        self.alerts_tree.column("confidence", width=100)
        self.alerts_tree.column("path", width=260)
        self.alerts_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(paned)
        paned.add(right, weight=2)
        preview_frame = ttk.LabelFrame(right, text="Alert Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.details_text = tk.Text(right, height=8, wrap=tk.WORD)
        self.details_text.pack(fill=tk.X, pady=5, padx=5)
        self.details_text.insert(tk.END, "Select an alert to view details")
        self.details_text.config(state=tk.DISABLED)
        btn_frame = ttk.Frame(f)
        btn_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Button(btn_frame, text="Open Selected", command=self.open_selected_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Refresh List", command=self.refresh_alerts).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Export Report", command=self.export_alert_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear All", command=self.clear_alerts).pack(side=tk.LEFT, padx=5)
        self.refresh_alerts()
        self.alerts_tree.bind("<<TreeviewSelect>>", self.on_alert_selected)
        self.alerts_tree.bind("<Double-1>", lambda e: self.open_selected_video())

    def refresh_alerts(self, filter_text=""):
        try:
            for item in self.alerts_tree.get_children():
                self.alerts_tree.delete(item)
            alerts = []
            if self.video_processor:
                alerts = self.video_processor.get_alert_history()
            if os.path.exists(VIOLENT_DIR):
                for filename in os.listdir(VIOLENT_DIR):
                    if filename.endswith(".mp4"):
                        try:
                            parts = filename.split("_")
                            timestamp = parts[0] + "_" + parts[1]
                            conf_part = parts[2] if len(parts) > 2 else "unknown"
                            confidence = float(conf_part.split(".")[0])
                            file_path = os.path.join(VIOLENT_DIR, filename)
                            if not any(a["path"] == file_path for a in alerts):
                                alerts.append({
                                    "timestamp": timestamp,
                                    "confidence": confidence,
                                    "path": file_path
                                })
                        except:
                            file_path = os.path.join(VIOLENT_DIR, filename)
                            alerts.append({
                                "timestamp": filename,
                                "confidence": 0.0,
                                "path": file_path
                            })
            alerts.sort(key=lambda x: x["timestamp"], reverse=True)
            if filter_text:
                filter_text = filter_text.lower()
                alerts = [a for a in alerts if filter_text in a["timestamp"].lower() or
                          filter_text in str(a["confidence"]).lower() or
                          filter_text in a["path"].lower()]
            for alert in alerts:
                self.alerts_tree.insert(
                    "", "end",
                    values=(
                        alert["timestamp"],
                        f"{alert['confidence']:.1f}%" if alert["confidence"] > 0 else "Unknown",
                        alert["path"]
                    )
                )
            today = datetime.now().strftime("%Y%m%d")
            today_count = sum(1 for a in alerts if a["timestamp"].startswith(today))
            self.stats_var.set(f"Total Alerts: {len(alerts)} | Today: {today_count}")
        except Exception:
            pass

    def filter_alerts(self):
        self.refresh_alerts(filter_text=self.filter_var.get())

    def on_alert_selected(self, event):
        try:
            selected = self.alerts_tree.selection()
            if not selected:
                return
            item_values = self.alerts_tree.item(selected[0], "values")
            video_path = item_values[2]
            timestamp = item_values[0]
            confidence = item_values[1]
            if not os.path.exists(video_path):
                self.update_preview_image(None)
                self.update_details_text(f"File not found: {video_path}")
                return
            self.extract_and_show_preview(video_path)
            details = f"Timestamp: {timestamp}\nConfidence: {confidence}\nFile: {os.path.basename(video_path)}\nFull Path: {video_path}\n"
            self.update_details_text(details)
        except Exception:
            pass

    def extract_and_show_preview(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.update_preview_image(None)
                return
            cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            cap.release()
            if ret:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], 30), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.putText(frame, "VIOLENT CONTENT", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                self.update_preview_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                self.update_preview_image(None)
        except Exception:
            self.update_preview_image(None)

    def update_preview_image(self, frame):
        try:
            if frame is None:
                frame = np.ones((180, 320, 3), dtype=np.uint8) * 44
                cv2.putText(frame, "No preview", (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
            h, w = frame.shape[:2]
            max_w, max_h = 320, 180
            if w > max_w or h > max_h:
                ratio = min(max_w / w, max_h / h)
                frame = cv2.resize(frame, (int(w * ratio), int(h * ratio)))
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(image=img)
            self.preview_label.config(image=img_tk)
            self.preview_label.image = img_tk
        except Exception:
            pass

    def update_details_text(self, text):
        try:
            self.details_text.config(state=tk.NORMAL)
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(tk.END, text)
            self.details_text.config(state=tk.DISABLED)
        except Exception:
            pass

    def export_alert_report(self):
        try:
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"watchdog_report_{date_str}.txt"
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialfile=filename
            )
            if not file_path:
                return
            alerts = []
            for item in self.alerts_tree.get_children():
                values = self.alerts_tree.item(item, "values")
                alerts.append({
                    "timestamp": values[0],
                    "confidence": values[1],
                    "path": values[2]
                })
            with open(file_path, 'w') as f:
                f.write("WatchDog Surveillance System - Alert Report\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Alerts: {len(alerts)}\n")
                f.write("-" * 70 + "\n\n")
                for i, alert in enumerate(alerts, 1):
                    f.write(f"Alert #{i}:\n")
                    f.write(f" Timestamp: {alert['timestamp']}\n")
                    f.write(f" Confidence: {alert['confidence']}\n")
                    f.write(f" Video: {alert['path']}\n\n")
            messagebox.showinfo("Export Complete", f"Report saved to:\n{file_path}")
        except Exception:
            pass

    def open_selected_video(self):
        try:
            selected = self.alerts_tree.selection()
            if not selected:
                messagebox.showinfo("Selection", "Please select a video to open")
                return
            video_path = self.alerts_tree.item(selected[0], "values")[2]
            if not os.path.exists(video_path):
                messagebox.showerror("File Error", f"File not found: {video_path}")
                return
            open_folder(video_path)
        except Exception:
            pass

    def clear_alerts(self):
        try:
            result = messagebox.askyesnocancel(
                "Clear Alerts",
                "Do you want to delete the video files as well?\n\n"
                "Yes - Delete files\n"
                "No - Keep files but clear list\n"
                "Cancel - Do nothing"
            )
            if result is None:
                return
            if result:
                if os.path.exists(VIOLENT_DIR):
                    for filename in os.listdir(VIOLENT_DIR):
                        if filename.endswith(".mp4"):
                            try:
                                os.remove(os.path.join(VIOLENT_DIR, filename))
                            except Exception:
                                pass
            for item in self.alerts_tree.get_children():
                self.alerts_tree.delete(item)
            if self.video_processor:
                self.video_processor.alert_history = []
            messagebox.showinfo("Clear Alerts", "Alerts cleared successfully")
        except Exception:
            pass

    def toggle_surveillance(self):
        if self.is_running:
            self.stop_surveillance()
        else:
            self.start_surveillance()

    def start_surveillance(self):
        try:
            self.apply_settings()
            if not os.path.exists(MODEL_PATH):
                messagebox.showerror("Error", f"Model file not found: {MODEL_PATH}")
                return
            self.video_processor = VideoProcessor(
                MODEL_PATH,
                self.settings,
                update_callback=self.update_video_frame,
                status_callback=self.update_status,
                mini_update_callback=self.update_mini_status
            )
            success = self.video_processor.start_capture()
            if not success:
                messagebox.showerror("Error", f"Failed to open camera {self.settings['camera_id']}")
                return
            self.is_running = True
            self.capture_thread = threading.Thread(target=self.video_processor.capture_and_save)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            self.start_btn_text.set("Stop Surveillance")
            self.update_status("Surveillance started")
            self.update_mini_status("Surveillance started")
            self.notebook.select(0)
        except Exception:
            pass

    def stop_surveillance(self):
        if self.video_processor:
            self.video_processor.stop()
        self.is_running = False
        self.start_btn_text.set("Start Surveillance")
        self.update_status("Surveillance stopped")
        self.update_mini_status("Surveillance stopped")
        self.set_default_video_frame()
        self.refresh_alerts()

    def update_video_frame(self, frame):
        try:
            h, w = frame.shape[:2]
            max_w, max_h = 900, 540
            if w > max_w or h > max_h:
                ratio = min(max_w / w, max_h / h)
                frame = cv2.resize(frame, (int(w * ratio), int(h * ratio)))
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=img_tk)
            self.video_label.image = img_tk
        except Exception:
            pass

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def update_mini_status(self, message):
        self.mini_update_var.set(message)
        self.root.update_idletasks()

    def save_settings(self):
        self.apply_settings()
        messagebox.showinfo("Settings", "Settings applied successfully.")

    def apply_settings(self):
        self.settings["camera_id"] = self.camera_id_var.get()
        self.settings["violence_threshold"] = self.threshold_var.get()
        self.settings["clip_duration"] = self.duration_var.get()
        self.settings["resolution"] = self.resolution_var.get()
        self.settings["output_dir"] = self.output_dir_var.get()
        global OUTPUT_DIR, VIOLENT_DIR
        OUTPUT_DIR = self.settings["output_dir"]
        VIOLENT_DIR = os.path.join(OUTPUT_DIR, "violent")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(VIOLENT_DIR, exist_ok=True)

    def test_camera(self):
        try:
            cam_id = self.camera_id_var.get()
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                messagebox.showerror("Camera Test", f"Failed to open camera {cam_id}")
                return
            ret, frame = cap.read()
            if not ret:
                cap.release()
                messagebox.showerror("Camera Test", "Failed to capture frame from camera")
                return
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.update_video_frame(frame_rgb)
            cap.release()
            messagebox.showinfo("Camera Test", f"Successfully connected to camera {cam_id}")
        except Exception:
            pass

    def on_close(self):
        try:
            if self.is_running and self.video_processor:
                self.video_processor.stop()
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=1.0)
            cv2.destroyAllWindows()
            self.root.destroy()
        except Exception:
            self.root.destroy()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VIOLENT_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)
    root = tk.Tk()
    app = WatchdogApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
