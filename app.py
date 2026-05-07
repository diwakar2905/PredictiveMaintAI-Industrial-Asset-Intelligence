"""
app.py — Render shim
---------------------
Render is invoking `streamlit run app.py` by default. This small shim forwards
execution to the actual dashboard script at `dashboard/app.py` so that the
container runs correctly whether Render calls `app.py` or the Dockerfile CMD.
"""
import runpy

runpy.run_path('dashboard/app.py', run_name='__main__')
