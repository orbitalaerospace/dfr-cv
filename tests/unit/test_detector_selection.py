"""Unit and integration tests for Milestone M6.1 explicit detector selection."""

from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from src.detectors.ultralytics_adapter import (
    DEFAULT_COCO_MAPPING,
    DEFAULT_SEADRONESSEE_MAPPING,
    UltralyticsDetectorAdapter,
    resolve_model_class_mapping,
)
from src.domain.enums import TargetClass
from src.pipeline.track_video import SUPPORTED_DETECTORS, main, run_video_tracking


def test_supported_detectors_registry():
    """Verify registry contains supported models and default paths."""
    assert "seadronessee" in SUPPORTED_DETECTORS
    assert "yolo11n" in SUPPORTED_DETECTORS
    assert Path(SUPPORTED_DETECTORS["seadronessee"]["weights"]).name == "seadronessee-yolov8n.pt"
    assert Path(SUPPORTED_DETECTORS["yolo11n"]["weights"]).name == "yolo11n.pt"


def test_default_selection_uses_seadronessee(monkeypatch, tmp_path):
    """Verify default detector selection loads SeaDronesSee without explicit arguments."""
    captured_args = {}

    def mock_adapter_init(self, weights_path, *args, **kwargs):
        captured_args["weights_path"] = Path(weights_path)
        captured_args["detector_name"] = kwargs.get("detector_name")
        self.device = "cpu"
        self.model = MagicMock()
        self.class_mapping = {}

    with patch.object(UltralyticsDetectorAdapter, "__init__", mock_adapter_init):
        # We simulate source video exists
        source_dummy = tmp_path / "dummy.mp4"
        source_dummy.touch()
        out_dummy = tmp_path / "out.mp4"

        with patch("cv2.VideoCapture") as mock_cap:
            mock_inst = MagicMock()
            mock_inst.isOpened.return_value = False  # Exit right after init
            mock_cap.return_value = mock_inst

            ret = run_video_tracking(
                source_path=source_dummy,
                output_path=out_dummy,
            )

    assert ret == 1  # exited at cap.isOpened()
    assert captured_args["weights_path"] == Path("models/seadronessee-yolov8n.pt")
    assert captured_args["detector_name"] == "seadronessee_yolov8n"


def test_explicit_selection_loads_requested_detector(tmp_path):
    """Verify explicit selection loads the correct weights and detector label for each mode."""
    source_dummy = tmp_path / "dummy.mp4"
    source_dummy.touch()
    out_dummy = tmp_path / "out.mp4"

    for det_key, expected_weights, expected_name in [
        ("seadronessee", "models/seadronessee-yolov8n.pt", "seadronessee_yolov8n"),
        ("yolo11n", "models/yolo11n.pt", "yolo11n_coco"),
    ]:
        captured_args = {}

        def mock_init(self, weights_path, *args, **kwargs):
            captured_args["weights_path"] = Path(weights_path)
            captured_args["detector_name"] = kwargs.get("detector_name")
            self.device = "cpu"
            self.model = MagicMock()
            self.class_mapping = {}

        with patch.object(UltralyticsDetectorAdapter, "__init__", mock_init):
            with patch("cv2.VideoCapture") as mock_cap:
                mock_inst = MagicMock()
                mock_inst.isOpened.return_value = False
                mock_cap.return_value = mock_inst

                run_video_tracking(
                    source_path=source_dummy,
                    output_path=out_dummy,
                    detector=det_key,
                )

        assert captured_args["weights_path"] == Path(expected_weights)
        assert captured_args["detector_name"] == expected_name


def test_custom_weights_override_takes_precedence(tmp_path):
    """Verify that explicitly passed custom weights override detector defaults."""
    source_dummy = tmp_path / "dummy.mp4"
    source_dummy.touch()
    out_dummy = tmp_path / "out.mp4"
    custom_weights = "models/custom_fine_tuned.pt"

    captured = {}

    def mock_init(self, weights_path, *args, **kwargs):
        captured["weights_path"] = Path(weights_path)
        captured["detector_name"] = kwargs.get("detector_name")
        self.device = "cpu"
        self.model = MagicMock()
        self.class_mapping = {}

    with patch.object(UltralyticsDetectorAdapter, "__init__", mock_init):
        with patch("cv2.VideoCapture") as mock_cap:
            mock_inst = MagicMock()
            mock_inst.isOpened.return_value = False
            mock_cap.return_value = mock_inst

            run_video_tracking(
                source_path=source_dummy,
                output_path=out_dummy,
                weights_path=custom_weights,
                detector="yolo11n",
            )

    assert captured["weights_path"] == Path(custom_weights)
    assert captured["detector_name"] == "yolo11n_coco"


def test_invalid_detector_rejected_clearly(tmp_path):
    """Verify that invalid detector strings are rejected with ValueError."""
    source_dummy = tmp_path / "dummy.mp4"
    source_dummy.touch()
    out_dummy = tmp_path / "out.mp4"

    with pytest.raises(ValueError, match="Unsupported detector 'unknown_detector'"):
        run_video_tracking(
            source_path=source_dummy,
            output_path=out_dummy,
            detector="unknown_detector",
        )


def test_cli_rejects_invalid_detector(monkeypatch):
    """Verify CLI argument parser rejects invalid choices with exit code 2."""
    test_args = ["track_video.py", "--source", "fake.mp4", "--detector", "invalid_choice"]
    monkeypatch.setattr("sys.argv", test_args)

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_model_specific_class_mappings():
    """Verify model-specific class mapping resolution for SeaDronesSee vs COCO."""
    # SeaDronesSee native names
    seadronessee_names = {0: "swimmer", 1: "boat", 2: "jetski", 3: "life_saving_appliances", 4: "buoy"}
    sds_mapping = resolve_model_class_mapping(seadronessee_names)
    assert sds_mapping[0] == TargetClass.SWIMMER
    assert sds_mapping[1] == TargetClass.WATERCRAFT
    assert sds_mapping[2] == TargetClass.WATERCRAFT
    assert sds_mapping[3] == TargetClass.LIFE_SAVING_APPLIANCE
    assert sds_mapping[4] == TargetClass.BUOY

    # COCO native names (80 classes)
    coco_names = {0: "person", 1: "bicycle", 2: "car", 8: "boat"}
    coco_mapping = resolve_model_class_mapping(coco_names)
    assert coco_mapping[0] == TargetClass.PERSON
    assert coco_mapping[8] == TargetClass.WATERCRAFT
    assert 1 not in coco_mapping
    assert 2 not in coco_mapping


def test_pipeline_smoke_both_detectors(tmp_path):
    """End-to-end integration test: run both detectors for 2 frames on authentic video."""
    video_source = "data/samples/aerial_water_sample.mp4"
    if not Path(video_source).exists():
        pytest.skip(f"Video {video_source} not found.")

    for det_key in ["seadronessee", "yolo11n"]:
        out_video = tmp_path / f"{det_key}_smoke.mp4"
        out_json = tmp_path / f"{det_key}_smoke.json"
        out_alerts = tmp_path / f"{det_key}_smoke.jsonl"

        ret = run_video_tracking(
            source_path=video_source,
            output_path=out_video,
            detector=det_key,
            max_frames=2,
            export_json=out_json,
            export_alerts=out_alerts,
            device="cpu",
        )
        assert ret == 0
        assert out_video.exists()
        assert out_json.exists()
        assert out_alerts.exists()
