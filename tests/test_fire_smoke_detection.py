import unittest
from unittest.mock import patch, MagicMock
from src.use_cases.fire_smoke_detection import fire_smoke_detection

class TestFireSmokeDetection(unittest.TestCase):
    @patch('src.use_cases.fire_smoke_detection.cv2')
    @patch('src.use_cases.fire_smoke_detection.YOLO')
    def test_fire_smoke_detection_initialization(self, mock_yolo, mock_cv2):
        # Setup mock to break loop immediately
        mock_cap = MagicMock()
        mock_cap.isOpened.side_effect = [True, False]
        mock_cap.read.return_value = (True, "fake_frame")
        mock_cv2.VideoCapture.return_value = mock_cap
        
        # Setup mock YOLO results
        mock_result = MagicMock()
        
        # Mock the len() and iteration of boxes to simulate 1 fire (cls 0) and 1 smoke (cls 1)
        class MockBoxes:
            def __init__(self):
                self.cls = [0, 1]
            def __len__(self):
                return 2
                
        mock_result.boxes = MockBoxes()
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.return_value = [mock_result]
        mock_yolo.return_value = mock_yolo_instance

        # Run the function with a fake weights file
        fire_smoke_detection(source='fake', weights='fake_fire.pt')
        
        # Assert YOLO was called correctly with fire/smoke classes
        mock_yolo_instance.assert_called_with("fake_frame", classes=[0, 1], verbose=False)

if __name__ == '__main__':
    unittest.main()
