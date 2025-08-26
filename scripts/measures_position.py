import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('../dataset/dataset.csv')
os.makedirs('../plots', exist_ok=True)
metrics = ['psnr_dl', 'ssim_dl', 'mse_dl']

agg_funcs = []
for m in metrics:
    agg_funcs.extend([
        (f'{m}_mean', (m, 'mean')),
        (f'{m}_mode', (m, lambda x: x.mode().iloc[0] if not x.mode().empty else None)),
        (f'{m}_median', (m, 'median')),
        (f'{m}_q1', (m, lambda x: x.quantile(0.25))),
        (f'{m}_q3', (m, lambda x: x.quantile(0.75))),
    ])

summary = df.groupby('algoritmo').agg(**dict(agg_funcs))
summary = summary.T

plt.figure(figsize=(14, 6))
sns.heatmap(summary, annot=True, fmt='.3f', cmap='Blues', cbar=False)
plt.title('Position Measures per Algorithm')
plt.tight_layout()
plt.savefig('plots/summary_position.png')
plt.close()