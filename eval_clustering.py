import pandas as pd
from PIL import Image
import io
import seaborn as sns
import matplotlib.pyplot as plt
import torch


out_dir = "outputs/basic_mmID_4096_discriminator1.0_latent1.0_MSE_JointTraining_NoExpansion/"
file_path = out_dir + "Pairwise Pearson Correlation Coefficient of Latents.png"

def read_csv_from_image(image_path):
    """
    Reads embedded CSV metadata from a PNG file and returns a pandas DataFrame.
    """
    # 1. Open the image file
    with Image.open(image_path) as img:
        # 2. Access the metadata dictionary via .info
        # The key must match the one used in plt.savefig ('Plot data')
        #img.load()
        csv_data = img.info.get('Plot data')
        #print(img.info)
        
    if not csv_data:
        raise ValueError(f"No 'Plot data' metadata found in {image_path}")

    # 3. Convert the string into a file-like object and read into pandas
    df = pd.read_csv(io.StringIO(csv_data))
    
    return df

df = read_csv_from_image(file_path)
df.set_index('Model Name', inplace=True)
print(df)

sns.clustermap(df)
plt.show()

row_norm_tensor = torch.tensor(df.iloc[:, :].values.astype(float))
row_norm_tensor = row_norm_tensor / row_norm_tensor.diag().unsqueeze(0)

col_norm_tensor = torch.tensor(df.iloc[:, :].values.astype(float))
col_norm_tensor = col_norm_tensor / col_norm_tensor.diag().unsqueeze(1)

df.iloc[:, :] = row_norm_tensor.numpy()
sns.clustermap(df)
plt.show()

df.iloc[:, :] = col_norm_tensor.numpy()
sns.clustermap(df)
plt.show()


