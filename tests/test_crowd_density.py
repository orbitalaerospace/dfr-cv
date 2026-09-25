import unittest
from unittest.mock import patch, MagicMock
from src.use_cases.crowd_density import crowd_density

class TestCrowdDensity(unittest.TestCase):
    @patch('src.use_cases.crowd_density.cv2')
    @patch('src.use_cases.crowd_density.YOLO')
    def test_crowd_density_initialization(self, mock_yolo, mock_cv2):
        # Setup mock to break loop immediately
        mock_cap = MagicMock()
        mock_cap.isOpened.side_effect = [True, False]
        mock_cap.read.return_value = (True, "fake_frame")
        mock_cv2.VideoCapture.return_value = mock_cap
        
        # Setup mock YOLO results
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.cls = [0, 1, 3] # simulate 2 humans, 1 vehicle
        
        # Mock the len() of boxes
        class MockBoxes:
            def __init__(self):
                self.cls = [0, 1, 3]
            def __len__(self):
                return 3
                
        mock_result.boxes = MockBoxes()
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.return_value = [mock_result]
        mock_yolo.return_value = mock_yolo_instance

        # Run the function
        crowd_density(source='fake', weights='fake.pt')
        
        # Assert YOLO was called correctly
        mock_yolo_instance.assert_called_with("fake_frame", verbose=False)

if __name__ == '__main__':
    unittest.main()
