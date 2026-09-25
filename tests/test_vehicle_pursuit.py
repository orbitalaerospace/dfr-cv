import unittest
from unittest.mock import patch, MagicMock
from src.use_cases.vehicle_pursuit import vehicle_pursuit

class TestVehiclePursuit(unittest.TestCase):
    @patch('src.use_cases.vehicle_pursuit.cv2')
    @patch('src.use_cases.vehicle_pursuit.YOLO')
    def test_vehicle_pursuit_initialization(self, mock_yolo, mock_cv2):
        # Setup mock to break loop immediately
        mock_cap = MagicMock()
        mock_cap.isOpened.side_effect = [True, False]
        mock_cap.read.return_value = (True, "fake_frame")
        mock_cv2.VideoCapture.return_value = mock_cap
        
        # Setup mock YOLO tracking results
        mock_result = MagicMock()
        mock_result.boxes.id = None
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.track.return_value = [mock_result]
        mock_yolo.return_value = mock_yolo_instance

        # Run the function
        vehicle_pursuit(source='fake', weights='fake.pt')
        
        # Assert YOLO tracker was called with correct vehicle classes
        mock_yolo_instance.track.assert_called_with(
            "fake_frame", classes=[3, 4, 5, 9], persist=True, tracker="bytetrack.yaml", verbose=False
        )

if __name__ == '__main__':
    unittest.main()
