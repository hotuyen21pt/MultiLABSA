FROM python:3.12-slim

# fasttext-wheel ships prebuilt wheels (no C++ build needed); pandas for CSVs.
# numpy<2 is required: fasttext's predict() uses np.array(..., copy=False),
# which NumPy 2.x turns into a hard error.
RUN pip install --no-cache-dir "numpy<2" pandas fasttext-wheel

WORKDIR /app

# Source + data are mounted at runtime (-v), not copied, to keep the image small.
CMD ["python", "add_language.py"]
