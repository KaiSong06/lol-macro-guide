"""PySide6 desktop control panel for lolcoach."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from lolcoach import autolaunch
from lolcoach.config import Config
from lolcoach.config_writer import load_config_for_edit, save_config
from lolcoach.controller import CoachController, CoachStatus
from lolcoach.health import HealthResult, run_all_checks
from lolcoach.log_reader import filter_records, list_log_files, read_log_file
from lolcoach.template_builder import DEFAULT_OUTPUT_DIR, download_all

try:  # pragma: no cover - import availability is environment-specific
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFormLayout,
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMenu,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QStackedWidget,
        QStyle,
        QSystemTrayIcon,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "PySide6 is required for the desktop UI. Install with: pip install PySide6"
    ) from exc


class MainWindow(QMainWindow):
    """Primary desktop control panel."""

    def __init__(
        self,
        *,
        controller: CoachController | None = None,
        config_path: Path = Path("config.yaml"),
    ) -> None:
        super().__init__()
        self.config_path = config_path
        self.controller = controller or CoachController(config_path=config_path)
        self.controller.add_listener(self._apply_status)
        self.config, self.raw_config = load_config_for_edit(config_path)

        self.setWindowTitle("lolcoach")
        self.resize(1120, 720)
        self.setStyleSheet(_STYLE)

        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        for label in ("Dashboard", "Setup", "Coach", "Logs", "Settings"):
            QListWidgetItem(label, self.nav)
        self.nav.currentRowChanged.connect(self._switch_page)

        self.pages = QStackedWidget()
        self.dashboard_page = self._build_dashboard_page()
        self.setup_page = self._build_setup_page()
        self.coach_page = self._build_coach_page()
        self.logs_page = self._build_logs_page()
        self.settings_page = self._build_settings_page()
        for page in (
            self.dashboard_page,
            self.setup_page,
            self.coach_page,
            self.logs_page,
            self.settings_page,
        ):
            self.pages.addWidget(page)

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.nav, 0)
        layout.addWidget(self.pages, 1)
        self.setCentralWidget(root)

        self.tray = self._build_tray()
        self.nav.setCurrentRow(0)
        self._apply_status(self.controller.status_snapshot())
        self._refresh_health()
        self._refresh_logs()

    def _switch_page(self, row: int) -> None:
        self.pages.setCurrentIndex(max(0, row))

    def _build_dashboard_page(self) -> QWidget:
        page = _page()
        self.state_label = QLabel("Ready")
        self.state_label.setObjectName("state")
        self.message_label = QLabel("Ready to start")
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.start_button.clicked.connect(self.controller.start)
        self.stop_button.clicked.connect(self.controller.stop)

        controls = QHBoxLayout()
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)

        self.game_summary = QLabel("No active game")
        self.last_error = QLabel("")
        self.callout_list = QListWidget()

        page.layout().addWidget(
            _panel("Coach Status", self.state_label, self.message_label, controls)
        )
        page.layout().addWidget(_panel("Current Game", self.game_summary))
        page.layout().addWidget(_panel("Recent Callouts", self.callout_list))
        page.layout().addWidget(_panel("Last Error", self.last_error))
        return page

    def _build_setup_page(self) -> QWidget:
        page = _page()
        self.health_table = QTableWidget(0, 4)
        self.health_table.setHorizontalHeaderLabels(["Check", "Status", "Detail", "Action"])
        self.health_table.horizontalHeader().setStretchLastSection(True)

        refresh = QPushButton("Run Health Checks")
        refresh.clicked.connect(self._refresh_health)
        templates = QPushButton("Download Templates")
        templates.clicked.connect(self._download_templates)
        calibrate = QPushButton("Run Calibration")
        calibrate.clicked.connect(self._launch_calibration)

        controls = QHBoxLayout()
        controls.addWidget(refresh)
        controls.addWidget(templates)
        controls.addWidget(calibrate)
        controls.addStretch(1)

        page.layout().addWidget(_panel("Setup Health", self.health_table, controls))
        return page

    def _build_coach_page(self) -> QWidget:
        page = _page()
        self.coach_status = QLabel("Coach is idle")
        self.runtime_detail = QTextEdit()
        self.runtime_detail.setReadOnly(True)
        page.layout().addWidget(_panel("Runtime", self.coach_status, self.runtime_detail))
        return page

    def _build_logs_page(self) -> QWidget:
        page = _page()
        self.log_files = QComboBox()
        self.log_filter = QLineEdit()
        self.log_filter.setPlaceholderText("Filter logs")
        self.log_filter.textChanged.connect(self._refresh_log_records)
        self.log_files.currentIndexChanged.connect(self._refresh_log_records)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_logs)
        open_folder = QPushButton("Open Log Folder")
        open_folder.clicked.connect(self._open_log_folder)

        controls = QHBoxLayout()
        controls.addWidget(self.log_files)
        controls.addWidget(self.log_filter)
        controls.addWidget(refresh)
        controls.addWidget(open_folder)

        self.log_table = QTableWidget(0, 4)
        self.log_table.setHorizontalHeaderLabels(["Time", "Type", "Summary", "File"])
        self.log_table.horizontalHeader().setStretchLastSection(True)
        page.layout().addWidget(_panel("Session Logs", controls, self.log_table))
        return page

    def _build_settings_page(self) -> QWidget:
        page = _page()
        form_widget = QWidget()
        form = QFormLayout(form_widget)
        self.model_input = QLineEdit(self.config.inference.model)
        self.timeout_input = _spin(self.config.inference.timeout_ms, 1, 120_000)
        self.capture_interval_input = _spin(self.config.capture.interval_ms, 100, 10_000)
        self.confidence_input = _spin(self.config.decisions.confidence_threshold, 1, 10)
        self.cooldown_input = _spin(self.config.decisions.cooldown_seconds, 0, 120)
        self.piper_path_input = QLineEdit(self.config.tts.piper_path)
        self.voice_dir_input = QLineEdit(self.config.tts.voice_dir)
        self.output_device_input = QLineEdit(self.config.tts.output_device)
        self.log_dir_input = QLineEdit(self.config.logging.directory)
        self.autolaunch_checkbox = QCheckBox("Launch lolcoach when a match starts")
        self.autolaunch_status = QLabel("")
        self._refresh_autolaunch_status()
        self.autolaunch_checkbox.toggled.connect(self._set_autolaunch_enabled)

        form.addRow("Ollama model", self.model_input)
        form.addRow("Inference timeout (ms)", self.timeout_input)
        form.addRow("Capture interval (ms)", self.capture_interval_input)
        form.addRow("Confidence threshold", self.confidence_input)
        form.addRow("Cooldown (s)", self.cooldown_input)
        form.addRow("Piper path", self.piper_path_input)
        form.addRow("Voice directory", self.voice_dir_input)
        form.addRow("Output device", self.output_device_input)
        form.addRow("Log directory", self.log_dir_input)
        form.addRow("Auto launch", self.autolaunch_checkbox)
        form.addRow("Auto launch status", self.autolaunch_status)

        save = QPushButton("Save Settings")
        save.clicked.connect(self._save_settings)
        page.layout().addWidget(_panel("Settings", form_widget, save))
        return page

    def _build_tray(self) -> QSystemTrayIcon:
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        tray = QSystemTrayIcon(icon, self)
        tray.setToolTip("lolcoach")
        start = QAction("Start Coach", self)
        stop = QAction("Stop Coach", self)
        show = QAction("Show", self)
        quit_action = QAction("Quit", self)
        start.triggered.connect(self.controller.start)
        stop.triggered.connect(self.controller.stop)
        show.triggered.connect(self.show)
        quit_action.triggered.connect(QApplication.instance().quit)
        menu = QMenu(self)
        menu.addAction(start)
        menu.addAction(stop)
        menu.addAction(show)
        menu.addAction(quit_action)
        tray.setContextMenu(menu)
        tray.setIcon(icon)
        tray.show()
        return tray

    def _apply_status(self, status: CoachStatus) -> None:
        self.state_label.setText(status.state)
        self.message_label.setText(status.message)
        self.start_button.setEnabled(not status.is_running)
        self.stop_button.setEnabled(status.is_running)
        self.coach_status.setText(status.message)
        self.last_error.setText(status.last_error or "No recent errors")
        self.callout_list.clear()
        self.callout_list.addItems(list(status.latest_callouts) or ["No callouts yet"])
        if status.game_state is None or not status.game_state.all_champions:
            self.game_summary.setText("No active game")
            self.runtime_detail.setPlainText("Waiting for League Live Client data.")
            return
        state = status.game_state
        self.game_summary.setText(
            f"{state.active_player_champion or 'Unknown'} | "
            f"{int(state.game_time_seconds // 60)}:{int(state.game_time_seconds % 60):02d}"
        )
        self.runtime_detail.setPlainText(
            "\n".join(
                [
                    f"Allies: {', '.join(sorted(state.ally_champions))}",
                    f"Enemies: {', '.join(sorted(state.enemy_champions))}",
                    f"Enemy jungler: {state.enemy_jungler_champion_name or 'unknown'}",
                    f"Last seen: {state.enemy_jungler_last_seen or 'not seen'}",
                ]
            )
        )

    def _refresh_health(self) -> None:
        self.config, self.raw_config = load_config_for_edit(self.config_path)
        self._render_health(run_all_checks(self.config))

    def _render_health(self, results: tuple[HealthResult, ...]) -> None:
        self.health_table.setRowCount(len(results))
        for row, result in enumerate(results):
            values = [result.name, result.status, result.detail, result.remediation]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 1:
                    item.setForeground(Qt.GlobalColor.green if result.ok else Qt.GlobalColor.yellow)
                self.health_table.setItem(row, col, item)

    def _download_templates(self) -> None:
        downloaded, skipped = download_all(DEFAULT_OUTPUT_DIR)
        QMessageBox.information(
            self,
            "Templates",
            f"Template download complete: {downloaded} new, {skipped} skipped.",
        )
        self._refresh_health()

    def _launch_calibration(self) -> None:
        subprocess.Popen([sys.executable, "-m", "lolcoach.calibrate"])  # noqa: S603

    def _refresh_logs(self) -> None:
        self.config, self.raw_config = load_config_for_edit(self.config_path)
        self.log_files.clear()
        for path in list_log_files(Path(self.config.logging.directory)):
            self.log_files.addItem(path.name, str(path))
        self._refresh_log_records()

    def _refresh_log_records(self) -> None:
        path_raw = self.log_files.currentData()
        records = read_log_file(Path(path_raw)) if path_raw else ()
        records = filter_records(records, query=self.log_filter.text())
        self.log_table.setRowCount(len(records))
        for row, record in enumerate(records):
            summary = str(
                record.fields.get("decision") or record.fields.get("reason") or record.fields
            )
            values = [
                "" if record.timestamp is None else f"{record.timestamp:.3f}",
                record.event_type,
                summary,
                record.path.name,
            ]
            for col, value in enumerate(values):
                self.log_table.setItem(row, col, QTableWidgetItem(value))

    def _open_log_folder(self) -> None:
        directory = Path(self.config.logging.directory)
        directory.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(directory)])  # noqa: S603,S607
        else:
            QMessageBox.information(self, "Logs", str(directory.resolve()))

    def _refresh_autolaunch_status(self) -> None:
        status = autolaunch.get_status()
        self.autolaunch_checkbox.blockSignals(True)
        self.autolaunch_checkbox.setChecked(status.enabled)
        self.autolaunch_checkbox.setEnabled(status.supported)
        self.autolaunch_checkbox.blockSignals(False)
        self.autolaunch_status.setText(status.detail)

    def _set_autolaunch_enabled(self, enabled: bool) -> None:
        try:
            status = autolaunch.set_enabled(self.config_path, enabled)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.autolaunch_status.setText(f"Failed: {exc}")
            self.autolaunch_checkbox.blockSignals(True)
            self.autolaunch_checkbox.setChecked(False)
            self.autolaunch_checkbox.blockSignals(False)
            QMessageBox.warning(self, "Auto launch", str(exc))
            return
        self.autolaunch_status.setText(status.detail)
        self.autolaunch_checkbox.setChecked(status.enabled)

    def _save_settings(self) -> None:
        config = Config(
            capture=self.config.capture.__class__(
                interval_ms=self.capture_interval_input.value(),
                minimap_roi=self.config.capture.minimap_roi,
            ),
            riot_api=self.config.riot_api,
            inference=self.config.inference.__class__(
                model=self.model_input.text().strip(),
                timeout_ms=self.timeout_input.value(),
                ollama_host=self.config.inference.ollama_host,
                keep_alive=self.config.inference.keep_alive,
            ),
            decisions=self.config.decisions.__class__(
                cooldown_seconds=self.cooldown_input.value(),
                confidence_threshold=self.confidence_input.value(),
                dedup_window_seconds=self.config.decisions.dedup_window_seconds,
                staleness_threshold_seconds=self.config.decisions.staleness_threshold_seconds,
            ),
            tts=self.config.tts.__class__(
                voice=self.config.tts.voice,
                piper_path=self.piper_path_input.text().strip(),
                voice_dir=self.voice_dir_input.text().strip(),
                speed=self.config.tts.speed,
                output_device=self.output_device_input.text().strip(),
                interrupt_confidence_delta=self.config.tts.interrupt_confidence_delta,
                interrupt_categories=self.config.tts.interrupt_categories,
            ),
            logging=self.config.logging.__class__(
                level=self.config.logging.level,
                directory=self.log_dir_input.text().strip(),
            ),
        )
        save_config(self.config_path, config, existing_raw=self.raw_config)
        self.config, self.raw_config = load_config_for_edit(self.config_path)
        QMessageBox.information(self, "Settings", "Settings saved.")


def run_ui(argv: list[str] | None = None) -> int:
    argv = argv or []
    config_path = _config_path_from_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("lolcoach")
    app.setQuitOnLastWindowClosed(False)
    window = MainWindow(config_path=config_path)
    if "--auto-start" in argv:
        window.controller.start()
    if "--minimized" not in argv:
        window.show()
    else:
        window.hide()
    return int(app.exec())


def _config_path_from_args(argv: list[str]) -> Path:
    for idx, arg in enumerate(argv):
        if arg == "--config" and idx + 1 < len(argv):
            return Path(argv[idx + 1])
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
    return Path("config.yaml")


def _page() -> QWidget:
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)
    return page


def _panel(title: str, *widgets: object) -> QFrame:
    panel = QFrame()
    panel.setObjectName("panel")
    layout = QVBoxLayout(panel)
    heading = QLabel(title)
    heading.setObjectName("heading")
    layout.addWidget(heading)
    for widget in widgets:
        if isinstance(widget, QHBoxLayout):
            layout.addLayout(widget)
        elif isinstance(widget, QVBoxLayout):
            layout.addLayout(widget)
        elif isinstance(widget, QWidget):
            layout.addWidget(widget)
    return panel


def _spin(value: int, minimum: int, maximum: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    return spin


_STYLE = """
QMainWindow, QWidget {
    background: #101418;
    color: #e8edf2;
    font-size: 13px;
}
QListWidget#nav {
    min-width: 190px;
    max-width: 190px;
    background: #151b21;
    border: 0;
    padding: 16px 8px;
}
QListWidget#nav::item {
    padding: 10px 12px;
    border-radius: 6px;
}
QListWidget#nav::item:selected {
    background: #2b6f7f;
    color: #ffffff;
}
QFrame#panel {
    background: #171e25;
    border: 1px solid #29323a;
    border-radius: 8px;
    padding: 14px;
}
QLabel#state {
    font-size: 30px;
    font-weight: 700;
}
QLabel#heading {
    font-size: 15px;
    font-weight: 700;
    color: #f4f7fa;
}
QPushButton {
    background: #2b6f7f;
    color: #ffffff;
    border: 0;
    border-radius: 6px;
    padding: 8px 14px;
}
QPushButton:disabled {
    background: #2d343b;
    color: #85919b;
}
QLineEdit, QSpinBox, QComboBox, QTextEdit, QTableWidget, QListWidget {
    background: #0f1317;
    border: 1px solid #303a43;
    border-radius: 6px;
    color: #e8edf2;
    padding: 6px;
}
QHeaderView::section {
    background: #202831;
    color: #dbe3ea;
    border: 0;
    padding: 6px;
}
"""
