PY = .venv/bin/python

# train
train:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PY) src/main.py

# run
run:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/results.ipynb
