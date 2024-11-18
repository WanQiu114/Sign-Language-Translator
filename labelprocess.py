from transformers import AutoTokenizer
from transformers import BertJapaneseTokenizer, BertForMaskedLM
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
import json
import glob as gb
    

# effectively rename the rows by adding prefix, e.g., "x" to "face0_x", when prefix = "face" and number = 0
# df is the DataFrame that needed to be renamed
# number is the #frame 
def renameprefix(df,prefix:str,number:int):
    new_columns = {col: f"{prefix}{number}_{col}" for col in df.columns}
    df.rename(columns=new_columns, inplace=True)



# iterate over all json files having certain pattern in its name, and convert them into csv file. 
json_files = gb.glob("./*.pose_data.json")

for json_file in json_files:
    with open(f'./{json_file}', 'r') as file:
        data = json.load(file)
    # checking the maximum number of keys
    max_index = max(range(len(data)))


    result = pd.DataFrame()

    faceprefix = "face"
    poseprefix = "pose"
    lefthandprefix = "left_hand"
    righthandprefix  = "right_hand"

    # in iteration, extract x y z position of face, pose and hands from the json file,
    # rename them, integrated together and output as csv file
    for index in range(max_index+1):
        face_data = data[index][faceprefix]
        pose_data = data[index][poseprefix]
        left_hand_data = data[index][lefthandprefix]
        right_hand_data = data[index][righthandprefix]

        df_face = pd.DataFrame(face_data)
        df_pose = pd.DataFrame(pose_data)
        df_left_hand = pd.DataFrame(left_hand_data)
        df_right_hand = pd.DataFrame(right_hand_data)


        renameprefix(df_face,faceprefix,index)
        renameprefix(df_pose,poseprefix,index)
        renameprefix(df_left_hand,lefthandprefix,index)
        renameprefix(df_right_hand,righthandprefix,index)
        if result.empty:
            result = pd.concat([df_face, df_pose, df_left_hand, df_right_hand], axis=1)
        else:
            result = pd.concat([result,df_face, df_pose, df_left_hand, df_right_hand], axis=1)
    result.to_csv(f'{json_file}_combined.csv', index=False)


    #renamecsvfile("combined_data.csv")

print("done")
