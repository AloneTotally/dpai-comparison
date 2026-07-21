# Geometry n(t) comparison — setup

## 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Launch Jupyter

```bash
jupyter notebook geometry_comparison.ipynb
```

or, if you prefer JupyterLab:

```bash
jupyter lab geometry_comparison.ipynb
```

This opens the notebook in your browser. Run all cells with **Cell → Run All** (or `Shift+Enter` through each cell).

## 4. Use your real data

By default the notebook uses synthetic demo curves. To use your own COMSOL export:

1. Save your data as a CSV with columns `geometry`, `t`, `n` (any column order, case doesn't matter).
2. In the second code cell, set:
   ```python
   CSV_PATH = "your_file.csv"
   ```
3. Re-run all cells.

Optionally change `T_PROCESS` (in seconds) in the ranking cell to check a different operating time.

## Editing in VS Code instead

If you'd rather not use the browser Jupyter UI, VS Code with the **Jupyter** and **Python** extensions installed can open and run `.ipynb` files directly — just point it at the same virtual environment as the interpreter.
