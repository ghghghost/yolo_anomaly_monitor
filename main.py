"""YOLO-based real-time work-zone anomaly monitor."""

from __future__ import annotations

import argparse
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Dict, Optional, Tuple

import cv2
from PIL import Image, ImageTk

from monitor_core import (
    AnomalyStateMachine,
    Detection,
    MonitorStatus,
    Zone,
    centered_zone,
)


COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

KOREAN_NAMES = {
    "person": "사람", "bicycle": "자전거", "car": "자동차", "motorcycle": "오토바이",
    "airplane": "비행기", "bus": "버스", "train": "기차", "truck": "트럭", "boat": "보트",
    "traffic light": "신호등", "fire hydrant": "소화전", "stop sign": "정지 표지판",
    "parking meter": "주차 미터기", "bench": "벤치", "bird": "새", "cat": "고양이",
    "dog": "개", "horse": "말", "sheep": "양", "cow": "소", "elephant": "코끼리",
    "bear": "곰", "zebra": "얼룩말", "giraffe": "기린", "backpack": "백팩",
    "umbrella": "우산", "handbag": "핸드백", "tie": "넥타이", "suitcase": "여행가방",
    "frisbee": "프리스비", "skis": "스키", "snowboard": "스노보드", "sports ball": "공",
    "kite": "연", "baseball bat": "야구방망이", "baseball glove": "야구글러브",
    "skateboard": "스케이트보드", "surfboard": "서핑보드", "tennis racket": "테니스라켓",
    "bottle": "병", "wine glass": "와인잔", "cup": "컵", "fork": "포크", "knife": "칼",
    "spoon": "숟가락", "bowl": "그릇", "banana": "바나나", "apple": "사과",
    "sandwich": "샌드위치", "orange": "오렌지", "broccoli": "브로콜리", "carrot": "당근",
    "hot dog": "핫도그", "pizza": "피자", "donut": "도넛", "cake": "케이크",
    "chair": "의자", "couch": "소파", "potted plant": "화분", "bed": "침대",
    "dining table": "식탁", "toilet": "변기", "tv": "TV", "laptop": "노트북",
    "mouse": "마우스", "remote": "리모컨", "keyboard": "키보드", "cell phone": "휴대전화",
    "microwave": "전자레인지", "oven": "오븐", "toaster": "토스터", "sink": "싱크대",
    "refrigerator": "냉장고", "book": "책", "clock": "시계", "vase": "꽃병",
    "scissors": "가위", "teddy bear": "곰인형", "hair drier": "헤어드라이어",
    "toothbrush": "칫솔",
}

DISPLAY_TO_LABEL = {f"{label} ({KOREAN_NAMES[label]})": label for label in COCO_LABELS}

STATUS_STYLE: Dict[MonitorStatus, Tuple[str, str]] = {
    MonitorStatus.WAITING: ("감지 대기 중", "#64748b"),
    MonitorStatus.NORMAL: ("정상 · 작업 영역 안", "#16a34a"),
    MonitorStatus.WARNING: ("경고 · 작업 영역 이탈", "#dc2626"),
    MonitorStatus.NOT_DETECTED: ("감지되지 않음", "#d97706"),
    MonitorStatus.RECOVERED: ("복구됨 · 정상 영역 복귀", "#0891b2"),
}


class DetectorWorker(threading.Thread):
    def __init__(
        self,
        output: queue.Queue,
        stop_event: threading.Event,
        camera_index: int,
        model_path: str,
        target_getter,
        confidence_getter,
        zone_ratio_getter,
        frames_getter,
    ):
        super().__init__(daemon=True)
        self.output = output
        self.stop_event = stop_event
        self.camera_index = camera_index
        self.model_path = model_path
        self.target_getter = target_getter
        self.confidence_getter = confidence_getter
        self.zone_ratio_getter = zone_ratio_getter
        self.frames_getter = frames_getter

    def _put_latest(self, item) -> None:
        try:
            while True:
                self.output.get_nowait()
        except queue.Empty:
            pass
        self.output.put_nowait(item)

    def run(self) -> None:
        capture = None
        try:
            from ultralytics import YOLO

            self._put_latest(("info", "YOLO 모델을 불러오는 중입니다…"))
            model = YOLO(self.model_path)
            names = model.names
            name_to_id = {str(name): int(class_id) for class_id, name in names.items()}

            if self.camera_index < 0:
                raise RuntimeError("카메라 번호는 0 이상이어야 합니다.")
            capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(self.camera_index)
            if not capture.isOpened():
                raise RuntimeError(f"카메라 {self.camera_index}을(를) 열 수 없습니다.")

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            state = AnomalyStateMachine(self.frames_getter())
            last_frames_setting = self.frames_getter()
            fps_value = 0.0
            fps_count = 0
            fps_started = time.perf_counter()

            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("카메라 영상을 읽지 못했습니다.")

                target = self.target_getter()
                confidence = self.confidence_getter()
                consecutive = self.frames_getter()
                if consecutive != last_frames_setting:
                    state = AnomalyStateMachine(consecutive)
                    last_frames_setting = consecutive

                class_id = name_to_id.get(target)
                if class_id is None:
                    raise RuntimeError(f"모델에 '{target}' 라벨이 없습니다.")

                results = model.predict(
                    source=frame,
                    conf=confidence,
                    classes=[class_id],
                    verbose=False,
                )
                best: Optional[Detection] = None
                boxes = results[0].boxes
                if boxes is not None:
                    for box in boxes:
                        score = float(box.conf[0])
                        coords = tuple(int(v) for v in box.xyxy[0].tolist())
                        candidate = Detection(coords, score)
                        if best is None or candidate.confidence > best.confidence:
                            best = candidate

                height, width = frame.shape[:2]
                zone = centered_zone(width, height, self.zone_ratio_getter())
                status = state.update(best, zone)
                fps_count += 1
                elapsed = time.perf_counter() - fps_started
                if elapsed >= 1.0:
                    fps_value = fps_count / elapsed
                    fps_count = 0
                    fps_started = time.perf_counter()

                annotated = draw_overlay(frame, zone, best, target, status, fps_value)
                self._put_latest(("frame", annotated, status, best, fps_value))
        except Exception as exc:
            self._put_latest(("error", str(exc)))
        finally:
            if capture is not None:
                capture.release()


def draw_overlay(
    frame, zone: Zone, detection: Optional[Detection], target: str,
    status: MonitorStatus, fps: float,
):
    color_bgr = {
        MonitorStatus.NORMAL: (46, 160, 67), MonitorStatus.RECOVERED: (178, 145, 8),
        MonitorStatus.WARNING: (38, 38, 220), MonitorStatus.NOT_DETECTED: (6, 126, 217),
        MonitorStatus.WAITING: (139, 116, 100),
    }[status]
    overlay = frame.copy()
    cv2.rectangle(overlay, (zone.left, zone.top), (zone.right, zone.bottom), (34, 197, 94), -1)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    cv2.rectangle(frame, (zone.left, zone.top), (zone.right, zone.bottom), (34, 197, 94), 2)
    cv2.putText(frame, "TARGET ZONE", (zone.left + 8, zone.top + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (34, 197, 94), 2, cv2.LINE_AA)

    if detection:
        x1, y1, x2, y2 = detection.box
        cx, cy = detection.center
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 3)
        cv2.circle(frame, (cx, cy), 6, color_bgr, -1)
        label = f"{target} {detection.confidence:.2f}"
        cv2.rectangle(frame, (x1, max(0, y1 - 29)), (x1 + 190, y1), color_bgr, -1)
        cv2.putText(frame, label, (x1 + 6, max(19, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (255, 255, 255), 2, cv2.LINE_AA)

    status_en = {
        MonitorStatus.WAITING: "WAITING", MonitorStatus.NORMAL: "NORMAL",
        MonitorStatus.WARNING: "WARNING - OUT OF ZONE",
        MonitorStatus.NOT_DETECTED: "OBJECT NOT DETECTED", MonitorStatus.RECOVERED: "RECOVERED",
    }[status]
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 54), (18, 24, 38), -1)
    cv2.putText(frame, status_en, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                color_bgr, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {fps:.1f}", (frame.shape[1] - 120, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 225, 232), 1, cv2.LINE_AA)
    return frame


class MonitorApp:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.worker: Optional[DetectorWorker] = None
        self.stop_event: Optional[threading.Event] = None
        self.result_queue: queue.Queue = queue.Queue(maxsize=2)
        self.photo = None
        self.latest_status = MonitorStatus.WAITING
        self.runtime_target = args.target if args.target in COCO_LABELS else "cup"
        self.runtime_confidence = float(args.confidence)
        self.runtime_zone = float(args.zone)
        self.runtime_frames = max(1, int(args.frames))

        root.title("YOLO 작업 상태 모니터")
        root.geometry("1180x760")
        root.minsize(900, 620)
        root.configure(bg="#0f172a")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<q>", lambda _event: self.close())
        root.bind("<Q>", lambda _event: self.close())

        self._build_ui()
        self._poll_results()
        if args.autostart:
            root.after(300, self.start)

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Panel.TFrame", background="#172033")
        style.configure("TLabel", background="#172033", foreground="#e2e8f0", font=("Malgun Gothic", 10))
        style.configure("Title.TLabel", background="#0f172a", foreground="#f8fafc", font=("Malgun Gothic", 18, "bold"))
        style.configure("Sub.TLabel", background="#0f172a", foreground="#94a3b8", font=("Malgun Gothic", 10))
        style.configure("TButton", font=("Malgun Gothic", 10, "bold"), padding=(10, 8))
        style.configure("TCombobox", padding=6)

        header = ttk.Frame(self.root, style="Panel.TFrame")
        header.configure(style="Panel.TFrame")
        header.pack(fill="x", padx=18, pady=(16, 10))
        title_wrap = tk.Frame(header, bg="#172033")
        title_wrap.pack(side="left", padx=16, pady=12)
        tk.Label(title_wrap, text="실시간 작업 상태 모니터", bg="#172033", fg="#f8fafc",
                 font=("Malgun Gothic", 17, "bold")).pack(anchor="w")
        tk.Label(title_wrap, text="YOLO 객체 위치 기반 이상 감지", bg="#172033", fg="#94a3b8",
                 font=("Malgun Gothic", 9)).pack(anchor="w")
        self.status_badge = tk.Label(header, text="  감지 대기 중  ", bg="#64748b", fg="white",
                                     font=("Malgun Gothic", 12, "bold"), padx=10, pady=8)
        self.status_badge.pack(side="right", padx=16)

        body = tk.Frame(self.root, bg="#0f172a")
        body.pack(fill="both", expand=True, padx=18, pady=(0, 16))
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        video_frame = tk.Frame(body, bg="#020617", highlightbackground="#334155", highlightthickness=1)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.video_label = tk.Label(video_frame, text="카메라를 시작하면 영상이 표시됩니다",
                                    bg="#020617", fg="#64748b", font=("Malgun Gothic", 13))
        self.video_label.pack(fill="both", expand=True)

        panel = ttk.Frame(body, style="Panel.TFrame", width=270)
        panel.grid(row=0, column=1, sticky="ns")
        panel.grid_propagate(False)

        ttk.Label(panel, text="감지 설정", font=("Malgun Gothic", 13, "bold")).pack(anchor="w", padx=18, pady=(20, 14))
        ttk.Label(panel, text="감지할 물체").pack(anchor="w", padx=18)
        values = list(DISPLAY_TO_LABEL.keys())
        default_label = self.args.target if self.args.target in COCO_LABELS else "cup"
        self.target_var = tk.StringVar(value=f"{default_label} ({KOREAN_NAMES[default_label]})")
        self.target_combo = ttk.Combobox(panel, textvariable=self.target_var, values=values, state="readonly")
        self.target_combo.pack(fill="x", padx=18, pady=(5, 14))
        self.target_combo.bind("<<ComboboxSelected>>", self._sync_runtime_settings)

        ttk.Label(panel, text="카메라 번호").pack(anchor="w", padx=18)
        self.camera_var = tk.IntVar(value=self.args.camera)
        ttk.Spinbox(panel, from_=0, to=9, textvariable=self.camera_var, width=8).pack(anchor="w", padx=18, pady=(5, 14))

        ttk.Label(panel, text="신뢰도 임계값").pack(anchor="w", padx=18)
        self.confidence_var = tk.DoubleVar(value=self.args.confidence)
        confidence_scale = ttk.Scale(panel, from_=0.1, to=0.9, variable=self.confidence_var,
                                     command=lambda _v: self._update_value_labels())
        confidence_scale.pack(fill="x", padx=18, pady=(5, 0))
        self.confidence_label = ttk.Label(panel, text="")
        self.confidence_label.pack(anchor="e", padx=18, pady=(0, 12))

        ttk.Label(panel, text="중앙 작업 영역 크기").pack(anchor="w", padx=18)
        self.zone_var = tk.DoubleVar(value=self.args.zone)
        ttk.Scale(panel, from_=0.3, to=0.9, variable=self.zone_var,
                  command=lambda _v: self._update_value_labels()).pack(fill="x", padx=18, pady=(5, 0))
        self.zone_label = ttk.Label(panel, text="")
        self.zone_label.pack(anchor="e", padx=18, pady=(0, 12))

        ttk.Label(panel, text="판정 연속 프레임").pack(anchor="w", padx=18)
        self.frames_var = tk.IntVar(value=self.args.frames)
        frames_spin = ttk.Spinbox(panel, from_=1, to=30, textvariable=self.frames_var, width=8,
                                  command=self._sync_runtime_settings)
        frames_spin.pack(anchor="w", padx=18, pady=(5, 18))
        frames_spin.bind("<FocusOut>", self._sync_runtime_settings)

        self.start_button = ttk.Button(panel, text="카메라 시작", command=self.start)
        self.start_button.pack(fill="x", padx=18, pady=(0, 8))
        self.stop_button = ttk.Button(panel, text="중지", command=self.stop, state="disabled")
        self.stop_button.pack(fill="x", padx=18)
        self.detail_label = ttk.Label(panel, text="Q 키: 프로그램 종료", foreground="#94a3b8")
        self.detail_label.pack(side="bottom", anchor="w", padx=18, pady=18)
        self._update_value_labels()

    def _update_value_labels(self) -> None:
        self.confidence_label.configure(text=f"{self.confidence_var.get():.2f}")
        self.zone_label.configure(text=f"화면 너비·높이의 {self.zone_var.get() * 100:.0f}%")
        self._sync_runtime_settings()

    def _sync_runtime_settings(self, _event=None) -> None:
        """Copy Tk values on the UI thread for safe access by the worker thread."""
        self.runtime_target = DISPLAY_TO_LABEL.get(self.target_var.get(), "cup")
        self.runtime_confidence = float(self.confidence_var.get())
        self.runtime_zone = float(self.zone_var.get())
        try:
            self.runtime_frames = max(1, int(self.frames_var.get()))
        except (tk.TclError, ValueError):
            pass

    def _selected_target(self) -> str:
        return self.runtime_target

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self._sync_runtime_settings()
        self.result_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.worker = DetectorWorker(
            self.result_queue, self.stop_event, int(self.camera_var.get()), self.args.model,
            lambda: self.runtime_target, lambda: self.runtime_confidence,
            lambda: self.runtime_zone, lambda: self.runtime_frames,
        )
        self.worker.start()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.detail_label.configure(text="모델 준비 중… (첫 실행 시 다운로드될 수 있습니다)")
        self._set_status(MonitorStatus.WAITING)

    def stop(self) -> None:
        if self.stop_event:
            self.stop_event.set()
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.video_label.configure(image="", text="카메라가 중지되었습니다")
        self.photo = None
        self.detail_label.configure(text="Q 키: 프로그램 종료")
        self._set_status(MonitorStatus.WAITING)

    def _set_status(self, status: MonitorStatus) -> None:
        self.latest_status = status
        text, color = STATUS_STYLE[status]
        self.status_badge.configure(text=f"  {text}  ", bg=color)

    def _show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_width = max(320, self.video_label.winfo_width() - 4)
        max_height = max(240, self.video_label.winfo_height() - 4)
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.photo, text="")

    def _poll_results(self) -> None:
        try:
            while True:
                item = self.result_queue.get_nowait()
                if item[0] == "frame":
                    _, frame, status, detection, fps = item
                    self._show_frame(frame)
                    self._set_status(status)
                    if detection:
                        self.detail_label.configure(
                            text=f"{self._selected_target()} · 신뢰도 {detection.confidence:.2f} · {fps:.1f} FPS"
                        )
                    else:
                        self.detail_label.configure(text=f"{self._selected_target()} 탐색 중 · {fps:.1f} FPS")
                elif item[0] == "info":
                    self.detail_label.configure(text=item[1])
                elif item[0] == "error":
                    self.stop()
                    messagebox.showerror("실행 오류", item[1])
        except queue.Empty:
            pass
        self.root.after(30, self._poll_results)

    def close(self) -> None:
        if self.stop_event:
            self.stop_event.set()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 실시간 작업 영역 이상 감지")
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO 모델 파일/이름")
    parser.add_argument("--camera", type=int, default=0, help="카메라 번호")
    parser.add_argument("--target", choices=COCO_LABELS, default="cup", help="초기 COCO 라벨")
    parser.add_argument("--confidence", type=float, default=0.4, help="신뢰도 임계값")
    parser.add_argument("--zone", type=float, default=0.5, help="중앙 영역 비율")
    parser.add_argument("--frames", type=int, default=5, help="상태 판정 연속 프레임 수")
    parser.add_argument("--autostart", action="store_true", help="실행 즉시 카메라 시작")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path(args.model).suffix and Path(args.model).parent != Path(".") and not Path(args.model).exists():
        raise SystemExit(f"모델 파일을 찾을 수 없습니다: {args.model}")
    root = tk.Tk()
    MonitorApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()

