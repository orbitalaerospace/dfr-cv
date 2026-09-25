import unittest
from unittest.mock import patch, MagicMock
from src.use_cases.gesture_recognition import gesture_recognition

class TestGestureRecognition(unittest.TestCase):
    @patch('src.use_cases.gesture_recognition.cv2')
    @patch('src.use_cases.gesture_recognition.YOLO')
    def test_gesture_recognition_initialization(self, mock_yolo, mock_cv2):
        # Setup mock to break loop immediately
        mock_cap = MagicMock()
        mock_cap.isOpened.side_effect = [True, False]
        mock_cap.read.return_value = (True, "fake_frame")
        mock_cv2.VideoCapture.return_value = mock_cap
        
        # Setup mock YOLO results
        mock_result = MagicMock()
        
        # Mock keypoints for pose estimation (empty for test safety)
        class MockKeypoints:
            def __init__(self):
                self.xy = []
                
        mock_result.keypoints = MockKeypoints()
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.return_value = [mock_result]
        mock_yolo.return_value = mock_yolo_instance

        # Run the function
        gesture_recognition(source='fake', weights='fake_pose.pt')
        
        # Assert YOLO was called correctly
        mock_yolo_instance.assert_called_with("fake_frame", verbose=False)

if __name__ == '__main__':
    unittest.main()
