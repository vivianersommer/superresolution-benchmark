import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('../dataset/dataset.csv')
os.makedirs('../plots', exist_ok=True)

psnr_summary = df.groupby('algoritmo')['psnr_dl'].agg(
    Mean='mean',
    Mode=lambda x: x.mode().iloc[0] if not x.mode().empty else None,
    Median='median',
    Q1=lambda x: x.quantile(0.25),
    Q3=lambda x: x.quantile(0.75)
)
ssim_summary = df.groupby('algoritmo')['ssim_dl'].agg(
    Mean='mean',
    Mode=lambda x: x.mode().iloc[0] if not x.mode().empty else None,
    Median='median',
    Q1=lambda x: x.quantile(0.25),
    Q3=lambda x: x.quantile(0.75)
)
mse_summary = df.groupby('algoritmo')['mse_dl'].agg(
    Mean='mean',
    Mode=lambda x: x.mode().iloc[0] if not x.mode().empty else None,
    Median='median',
    Q1=lambda x: x.quantile(0.25),
    Q3=lambda x: x.quantile(0.75)
)

psnr_summary.to_csv('plots/psnr_summary.csv')
ssim_summary.to_csv('plots/ssim_summary.csv')
mse_summary.to_csv('plots/mse_summary.csv')

plt.figure(figsize=(8, 4))
sns.heatmap(psnr_summary, annot=True, fmt='.3f', cmap='Blues', cbar=False)
plt.title('PSNR - Medidas de Posição')
plt.tight_layout()
plt.savefig('plots/psnr_summary.png')
plt.close()

plt.figure(figsize=(8, 4))
sns.heatmap(ssim_summary, annot=True, fmt='.3f', cmap='Greens', cbar=False)
plt.title('SSIM - Medidas de Posição')
plt.tight_layout()
plt.savefig('plots/ssim_summary.png')
plt.close()

plt.figure(figsize=(8, 4))
sns.heatmap(mse_summary, annot=True, fmt='.3f', cmap='Reds', cbar=False)
plt.title('MSE - Medidas de Posição')
plt.tight_layout()
plt.savefig('plots/mse_summary.png')
plt.close()