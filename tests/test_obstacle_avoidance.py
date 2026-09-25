import unittest
from unittest.mock import patch, MagicMock
from src.use_cases.obstacle_avoidance import obstacle_avoidance

class TestObstacleAvoidance(unittest.TestCase):
    @patch('src.use_cases.obstacle_avoidance.cv2')
    @patch('src.use_cases.obstacle_avoidance.YOLO')
    def test_obstacle_avoidance_initialization(self, mock_yolo, mock_cv2):
        # Setup mock to break loop immediately
        mock_cap = MagicMock()
        mock_cap.isOpened.side_effect = [True, False]
        mock_cap.read.return_value = (True, "fake_frame")
        mock_cv2.VideoCapture.return_value = mock_cap
        
        # Setup mock YOLO tracker results
        mock_result = MagicMock()
        mock_result.boxes.id = None # Simulate no objects tracked on first frame
        mock_result.plot.return_value = "annotated_frame"
        
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.track.return_value = [mock_result]
        mock_yolo.return_value = mock_yolo_instance

        # Run the function
        obstacle_avoidance(source='fake', weights='fake_weights.pt')
        
        # Assert YOLO tracker was called correctly
        mock_yolo_instance.track.assert_called_with(
            "fake_frame", persist=True, tracker="bytetrack.yaml", verbose=False
        )

if __name__ == '__main__':
    unittest.main()
