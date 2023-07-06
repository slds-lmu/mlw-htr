import pytest
import unittest
from functools import reduce
from lectiomat import  Lectiomat

class TestLectiomatImgSeg(unittest.TestCase):

    def setUp(self):
        pass

    def test_ocr_0(self):
        ex: str = [
            "./tests/assets/ex1.png",
            "./tests/assets/ex2.png"
        ]
        test_lm = Lectiomat()
        results = test_lm.apply_img_seg(ex)
        results = test_lm.apply_ocr(results)
        assert len(results) == len(ex)
        assert True not in list(map(lambda e: len(e) == 0, results))
        assert reduce(lambda acc, e: isinstance(e, str) and acc, results, True)
    
    def test_ocr_1(self):
        ex: str = [
            "./tests/assets/ex1.png",
            "./tests/assets/ex2.png"
        ]
        test_lm = Lectiomat()
        results = test_lm(ex)
        assert len(results) == len(ex)
        assert True not in list(map(lambda e: len(e) == 0, results))
        assert reduce(lambda acc, e: isinstance(e, str) and acc, results, True)

    def test_ocr_2(self):
        ex: str = [
            "./tests/assets/ex1.png",
            "./tests/assets/ex3.png"
        ]
        test_lm = Lectiomat()
        with pytest.raises(Exception):
            _ = test_lm(ex)

    def test_ocr_3(self):
        test_lm = Lectiomat()
        result = test_lm("./tests/assets/ex1.png")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_ocr_4(self):
        ex: str = [
            "./tests/assets/ex1.png",
            "./tests/assets/ex2.png"
        ]
        test_lm = Lectiomat()
        results = test_lm.apply_img_seg(ex)
        with pytest.raises(Exception):
            _ = test_lm.apply_ocr(results[0])

    def test_ocr_5(self):
        ex: str = [
            "./tests/assets/ex3.png",
            "./tests/assets/ex2.png"
        ]
        test_lm = Lectiomat()
        with pytest.raises(Exception)  as exc_info:
            results = test_lm.apply_img_seg(ex)
            _ = test_lm.apply_ocr(results[0])
        assert str(exc_info.value) == "ERROR -> No Lemma found on images: ['./tests/assets/ex3.png']!"
    
    def test_ocr_6(self):
        ex: str = [
            "./tests/assets/ex3.png",
            "./tests/assets/ex2.png"
        ]
        test_lm = Lectiomat()
        test_lm("./tests/assets/ex2.png")
        test_lm(["./tests/assets/ex2.png"])
        test_lm([
            "./tests/assets/ex2.png",
            "./tests/assets/ex2.png",
            "./tests/assets/ex1.png"])

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()