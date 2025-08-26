import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('../dataset/dataset.csv')
os.makedirs('../plots', exist_ok=True)

def amplitude(x):
    return x.max() - x.min()

def cv(x):
    return x.std() / x.mean() if x.mean() != 0 else None

psnr_disp = df.groupby('algoritmo')['psnr_dl'].agg(
    Std='std',
    Var='var',
    Range=amplitude,
    CV=cv
)
ssim_disp = df.groupby('algoritmo')['ssim_dl'].agg(
    Std='std',
    Var='var',
    Range=amplitude,
    CV=cv
)
mse_disp = df.groupby('algoritmo')['mse_dl'].agg(
    Std='std',
    Var='var',
    Range=amplitude,
    CV=cv
)

psnr_disp.to_csv('plots/psnr_dispersion.csv')
ssim_disp.to_csv('plots/ssim_dispersion.csv')
mse_disp.to_csv('plots/mse_dispersion.csv')

plt.figure(figsize=(8, 4))
sns.heatmap(psnr_disp, annot=True, fmt='.3f', cmap='Purples', cbar=False)
plt.title('PSNR - Medidas de Dispersão')
plt.tight_layout()
plt.savefig('plots/psnr_dispersion.png')
plt.close()

plt.figure(figsize=(8, 4))
sns.heatmap(ssim_disp, annot=True, fmt='.3f', cmap='Oranges', cbar=False)
plt.title('SSIM - Medidas de Dispersão')
plt.tight_layout()
plt.savefig('plots/ssim_dispersion.png')
plt.close()

plt.figure(figsize=(8, 4))
sns.heatmap(mse_disp, annot=True, fmt='.3f', cmap='Greys', cbar=False)
plt.title('MSE - Medidas de Dispersão')
plt.tight_layout()
plt.savefig('plots/mse_dispersion.png')
plt.close()