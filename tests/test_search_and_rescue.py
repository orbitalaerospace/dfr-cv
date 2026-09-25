import unittest
from unittest.mock import patch, MagicMock
from src.use_cases.search_and_rescue import search_and_rescue

class TestSearchAndRescue(unittest.TestCase):
    @patch('src.use_cases.search_and_rescue.cv2')
    @patch('src.use_cases.search_and_rescue.YOLO')
    def test_search_and_rescue_initialization(self, mock_yolo, mock_cv2):
        # Setup mock to break loop immediately after one frame
        mock_cap = MagicMock()
        mock_cap.isOpened.side_effect = [True, False] 
        mock_cap.read.return_value = (True, "fake_frame")
        mock_cv2.VideoCapture.return_value = mock_cap
        
        # Setup mock YOLO results
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.return_value = [mock_result]
        mock_yolo.return_value = mock_yolo_instance

        # Run the function
        search_and_rescue(source='fake', weights='fake.pt')

        # Assert YOLO was called with correct arguments (filtering humans)
        mock_yolo_instance.assert_called_with("fake_frame", classes=[0, 1], verbose=False)
        mock_cv2.VideoCapture.assert_called_with('fake')

if __name__ == '__main__':
    unittest.main()
