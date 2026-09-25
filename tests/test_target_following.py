import unittest
from unittest.mock import patch, MagicMock
from src.use_cases.target_following import target_following

class TestTargetFollowing(unittest.TestCase):
    @patch('src.use_cases.target_following.cv2')
    @patch('src.use_cases.target_following.YOLO')
    def test_target_following_initialization(self, mock_yolo, mock_cv2):
        # Setup mock to break loop immediately
        mock_cap = MagicMock()
        mock_cap.isOpened.side_effect = [True, False]
        mock_cap.read.return_value = (True, "fake_frame")
        mock_cap.get.return_value = 640 # Fake width/height
        mock_cv2.VideoCapture.return_value = mock_cap
        
        # Setup mock YOLO results (empty to test safe initialization)
        mock_result = MagicMock()
        
        class MockBoxes(list):
            pass
            
        mock_result.boxes = MockBoxes()
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.return_value = [mock_result]
        mock_yolo.return_value = mock_yolo_instance

        # Run the function
        target_following(source='fake', weights='fake.pt')
        
        # Assert YOLO was called correctly
        mock_yolo_instance.assert_called_with("fake_frame", verbose=False)

if __name__ == '__main__':
    unittest.main()
