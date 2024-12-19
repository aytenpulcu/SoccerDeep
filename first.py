import os
import zipfile
import argparse
import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader=SoccerNetDownloader("../SoccerNet")
list_splits = ["train", "valid", "test", "challenge"]
#%%
mySoccerNetDownloader.downloadDataTask(task="spotting-ball-2024", split=list_splits, password="s0cc3rn3t")
#%%
for split in list_splits:
    print(f"Unzipping {split}.zip ...")
    subtask_data_dir = os.path.join("../SoccerNet", "spotting-ball-2024")
    zip_filename = os.path.join(subtask_data_dir, f"{split}.zip")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(subtask_data_dir)
    print(f"... done unzipping {split}.zip")
