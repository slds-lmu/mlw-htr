import pytest
import unittest
import numpy as np
from PIL import Image
from lectiomat import  Lectiomat

class TestLectiomatImgSeg(unittest.TestCase):

    def setUp(self):
        pass
    
    def test_lectiomat_init(self):
        test_lm = Lectiomat()
        assert test_lm is not None

    def test_img_seg_0(self):
        ex: str = [
            "./tests/assets/ex1.png",
            "./tests/assets/ex2.png"
        ]
        test_lm = Lectiomat()
        results = test_lm.apply_img_seg(ex)
        for path, res in zip(ex, results):
            img = np.asarray(Image.open(path))
            assert np.all(np.shape(img)[0:2] > np.shape(res)[0:2])

    def test_img_seg_1(self):
        ex: str = ["./tests/assets/ex3.png"]
        test_lm = Lectiomat()
        with pytest.raises(Exception):
            _ = test_lm.apply_img_seg(ex)

    def test_img_seg_1(self):
        test_lm = Lectiomat()
        with pytest.raises(Exception):
            _ = test_lm.apply_img_seg("./tests/assets/ex3.png")

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()