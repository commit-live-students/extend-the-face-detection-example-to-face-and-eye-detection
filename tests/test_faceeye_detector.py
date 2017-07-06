from unittest import TestCase


class TestFaceeye_detector(TestCase):
    def test_faceeye_detector(self):
        from build import face_plot
        x = face_plot("./images/face01.jpg")
        print(x)

