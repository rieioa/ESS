import pandas as pd
from pathlib import Path

files = sorted(Path('../preprocessed_data').glob('batch*_preprocessed_cycles_*.csv'))
df = pd.read_csv(files[0])
print(df.columns.tolist())
print(df.head(30))
