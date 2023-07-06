import pytest
import unittest
import os
import pandas as pd
import time
from src.OCR.MLWDataset import MLWDataset

class TestMLWDataset(unittest.TestCase):
    path: str = "test.json"
    ex: list = ['952.jpg', '953.jpg', '954.jpg']
    def setUp(self):
        ex_data_json: str = '[{"id": 952, "lemma": "kaheio"}, {"id": 953, "lemma": "kai"}, {"id": 954, "lemma": "kakos"}, {"id": 955, "lemma": "kalendae"}, {"id": 956, "lemma": "kalendae"}, {"id": 957, "lemma": "kalendae"}, {"id": 958, "lemma": "kalendae"}, {"id": 959, "lemma": "kalendae"}, {"id": 960, "lemma": "kalendae"}, {"id": 961, "lemma": "kalendae"}, {"id": 962, "lemma": "kalendae"}, {"id": 963, "lemma": "kalendae"}, {"id": 964, "lemma": "kalendae"}, {"id": 965, "lemma": "kalendae"}, {"id": 966, "lemma": "kalendae"}, {"id": 967, "lemma": "kalendae"}, {"id": 968, "lemma": "kalendae"}, {"id": 969, "lemma": "kalendae"}, {"id": 970, "lemma": "kalendae"}, {"id": 971, "lemma": "kalendae"}, {"id": 972, "lemma": "kalendae"}, {"id": 973, "lemma": "kalendae"}, {"id": 974, "lemma": "kalendae"}, {"id": 975, "lemma": "kalendae"}, {"id": 976, "lemma": "kalendae"}, {"id": 977, "lemma": "kalendae"}, {"id": 978, "lemma": "kalendae"}, {"id": 979, "lemma": "kalendae"}, {"id": 980, "lemma": "kalendae"}, {"id": 981, "lemma": "kalendae"}, {"id": 982, "lemma": "kalendae"}, {"id": 983, "lemma": "kalendae"}, {"id": 984, "lemma": "kalendae"}, {"id": 985, "lemma": "kalendae"}, {"id": 986, "lemma": "kalendae"}, {"id": 987, "lemma": "kalendae"}, {"id": 988, "lemma": "kalendae"}, {"id": 989, "lemma": "kalendae"}, {"id": 990, "lemma": "kalendae"}, {"id": 991, "lemma": "kalendae"}, {"id": 992, "lemma": "kalendae"}, {"id": 993, "lemma": "kalendae"}, {"id": 994, "lemma": "kalendae"}, {"id": 995, "lemma": "kalendae"}, {"id": 996, "lemma": "kalendae"}, {"id": 997, "lemma": "kalendae"}, {"id": 998, "lemma": "kalendae"}, {"id": 999, "lemma": "kalendae"}, {"id": 1000, "lemma": "kalendae"}, {"id": 1001, "lemma": "kalendae"}, {"id": 1002, "lemma": "kalendae"}, {"id": 1003, "lemma": "kalendae"}, {"id": 1004, "lemma": "kalendae"}, {"id": 1005, "lemma": "kalendae"}, {"id": 1006, "lemma": "kalendae"}, {"id": 1007, "lemma": "kalendae"}, {"id": 1008, "lemma": "kalendae"}, {"id": 1009, "lemma": "kalendae"}, {"id": 1010, "lemma": "kalendae"}, {"id": 1011, "lemma": "kalendae"}, {"id": 1012, "lemma": "kalendae"}, {"id": 1013, "lemma": "kalendae"}, {"id": 1014, "lemma": "kalendae"}, {"id": 1015, "lemma": "kalendae"}, {"id": 1016, "lemma": "kalendae"}, {"id": 1017, "lemma": "kalendae"}, {"id": 1018, "lemma": "kalendae"}, {"id": 1019, "lemma": "kalendae"}, {"id": 1020, "lemma": "kalendae"}, {"id": 1021, "lemma": "kalendae"}, {"id": 1022, "lemma": "kalendae"}, {"id": 1023, "lemma": "kalendae"}, {"id": 1024, "lemma": "kalendae"}, {"id": 1025, "lemma": "kalendae"}, {"id": 1026, "lemma": "kalendae"}, {"id": 1027, "lemma": "kalendae"}, {"id": 1028, "lemma": "kalendae"}, {"id": 1029, "lemma": "kalendae"}, {"id": 1030, "lemma": "kalendae"}, {"id": 1031, "lemma": "kalendae"}, {"id": 1032, "lemma": "kalendae"}, {"id": 1033, "lemma": "kalendae"}, {"id": 1034, "lemma": "kalendae"}, {"id": 1035, "lemma": "kalendae"}, {"id": 1036, "lemma": "kalendae"}, {"id": 1037, "lemma": "kalendae"}, {"id": 1038, "lemma": "kalendae"}, {"id": 1039, "lemma": "kalendae"}, {"id": 1040, "lemma": "kalendae"}, {"id": 1041, "lemma": "kalendae"}, {"id": 1042, "lemma": "kalendae"}, {"id": 1043, "lemma": "kalendae"}, {"id": 1044, "lemma": "kalendae"}, {"id": 1045, "lemma": "kalendae"}, {"id": 1046, "lemma": "kalendae"}, {"id": 1047, "lemma": "kalendae"}, {"id": 1048, "lemma": "kalendae"}, {"id": 1049, "lemma": "kalendae"}, {"id": 1050, "lemma": "kalendae"}, {"id": 1051, "lemma": "kalendae"}, {"id": 1052, "lemma": "kalendae"}, {"id": 1053, "lemma": "kalendae"}, {"id": 1054, "lemma": "kalendae"}, {"id": 1055, "lemma": "kalendae"}, {"id": 1056, "lemma": "kalendae"}, {"id": 1057, "lemma": "kalendae"}, {"id": 1058, "lemma": "kalendae"}, {"id": 1059, "lemma": "kalendae"}, {"id": 1060, "lemma": "kalendae"}, {"id": 1061, "lemma": "kalendae"}, {"id": 1062, "lemma": "kalendae"}, {"id": 1063, "lemma": "kalendae"}, {"id": 1064, "lemma": "kalendae"}, {"id": 1065, "lemma": "kalendae"}]'
        f = open(self.path, 'w')
        f.write(ex_data_json)
        f.close()
        for name in self.ex:
            f = open(name, 'w')
            f.write('')
            f.close()
    
    def test_str(self):
        df: pd.DataFrame = pd.read_json(self.path)
        test_ds: MLWDataset = MLWDataset(df, '.', None, None)
        self.assertEqual(len(test_ds), len(self.ex))

    def tearDown(self):
        os.remove(self.path)
        for name in self.ex:
            os.remove(name)

if __name__ == "__main__":
    unittest.main()