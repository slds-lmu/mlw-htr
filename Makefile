py_interpreter=`which python`

tokenizer:
	${py_interpreter} src/Tokenization/make_tokenizer.py

train-ocr:
	${py_interpreter} src/OCR/main.py

eval-ocr:
	${py_interpreter} src/OCR/main.py global_config.goal=eval

run-debug:
	${py_interpreter} src/OCR/main.py global_config.debug=True

split:
	${py_interpreter} src/OCR/train_test_split.py --config ./config/OCR/config.yaml --target-folder ./
