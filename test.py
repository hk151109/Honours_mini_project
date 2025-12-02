# test_loader.py
from torchvision import datasets
import torchvision.transforms as T
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pathlib import Path
DATA_ROOT = "./wildfire-prediction-dataset"
tf = T.Compose([T.Resize((224,224)), T.ToTensor()])

ds = datasets.ImageFolder(str(Path(DATA_ROOT)/"train"), transform=tf)
print("Classes:", ds.classes, "Total:", len(ds))
bad=[]
for i in range(min(200, len(ds))):
    try:
        img, lbl = ds[i]
        if i%50==0: print("Loaded", i)
    except Exception as e:
        bad.append((i, ds.samples[i][0], repr(e)))
        print("FAILED", i, ds.samples[i][0], e)
print("Done. bad:", len(bad))
