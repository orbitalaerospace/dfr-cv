import unittest
from unittest.mock import patch, MagicMock
from src.use_cases.alpr import run_alpr
import sys

class TestALPR(unittest.TestCase):
    @patch('src.use_cases.alpr.cv2')
    @patch('src.use_cases.alpr.YOLO')
    @patch('src.use_cases.alpr.easyocr')
    def test_alpr_initialization(self, mock_easyocr, mock_yolo, mock_cv2):
        # Mock easyocr reader
        mock_reader = MagicMock()
        mock_easyocr.Reader.return_value = mock_reader
        
        # Mock VideoCapture to break loop immediately
        mock_cap = MagicMock()
        mock_cap.isOpened.side_effect = [True, False]
        mock_cap.read.return_value = (True, "fake_frame")
        mock_cv2.VideoCapture.return_value = mock_cap
        
        # Mock YOLO results
        mock_result = MagicMock()
        mock_box = MagicMock()
        
        # Fake bounding box (x1, y1, x2, y2)
        import torch
        mock_box.xyxy = [torch.tensor([10, 10, 100, 50])]
        
        # Mock boxes array
        class MockBoxes(list):
            def __init__(self, item):
                self.item = item
            def __iter__(self):
                yield self.item
                
        mock_result.boxes = MockBoxes(mock_box)
        mock_result.plot.return_value = "annotated_frame"
        
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.return_value = [mock_result]
        mock_yolo.return_value = mock_yolo_instance
        
        # To avoid actual numpy slicing errors in the test, we mock the frame itself
        # This is a bit tricky with raw strings, but since we just want to test initialization:
        # We can patch HAS_OCR to True to ensure the function runs
        with patch('src.use_cases.alpr.HAS_OCR', True):
            # We catch exceptions because string slicing "fake_frame"[y:y, x:x] will fail,
            # but we just want to ensure it tries to load YOLO and EasyOCR
            try:
                run_alpr(source='fake', weights='fake_plate.pt')
            except TypeError:
                pass
                
        # Assert YOLO and EasyOCR were initialized
        mock_easyocr.Reader.assert_called_with(['en'], gpu=False)
        mock_yolo.assert_called_with('fake_plate.pt')

if __name__ == '__main__':
    unittest.main()
