import numpy as np
import pandas as pd
import re
import glob as gb
import os
import copy


FACESIZE=80
POSESIZE=33
HANDSIZE=21

#define the empty NumPy array for face, pose and hand
FACE_EMPTY=np.zeros((FACESIZE, 3))
POSE_EMPTY=np.zeros((POSESIZE, 3))
HAND_EMPTY=np.zeros((HANDSIZE, 3))




def extract_frame_number(column_name):
    match = re.search(r'\d+', column_name)
    return int(match.group()) if match else None

def extract_position_number(column_name):

    patterns = {'face': 0,'pose': 1,'left_hand': 2,'right_hand': 3}
    
    for pattern, code in patterns.items():
        if re.search(pattern, column_name):
            return code
    return None

#converting any csv into a numpy file.
csv_files = gb.glob("./*.pose_data.json_combined.csv")
for csv_file in csv_files:

    file_path = csv_file
    data = pd.read_csv(file_path)


    num_columns = data.shape[1]

    last_frame_head = FACE_EMPTY
    last_frame_pose = POSE_EMPTY
    last_frame_left_hand = HAND_EMPTY
    last_frame_right_hand = HAND_EMPTY

    counter =0
    while counter<num_columns//3:
        column_name = data.columns[counter*3]



        if counter*3 !=num_columns:
            if (counter+1)*3 >=num_columns:
                next_column_name = "DNE"
            else:
                next_column_name = data.columns[(counter+1)*3]

        current_frame = extract_frame_number(column_name)
        next_column_frame = extract_frame_number(next_column_name)

        if next_column_frame is None:
            coordinates1 = data.iloc[:, counter*3:counter*3+3].dropna().head(FACESIZE).to_numpy()
            coordinates = np.vstack((coordinates,coordinates1))

            if column_name.startswith("face"):
                coordinates = np.vstack((coordinates,last_frame_pose))
                coordinates = np.vstack((coordinates,last_frame_left_hand))
                coordinates = np.vstack((coordinates,last_frame_right_hand))
            elif column_name.startswith("pose"):
                coordinates = np.vstack((coordinates,last_frame_left_hand))
                coordinates = np.vstack((coordinates,last_frame_right_hand))
            elif column_name.startswith("left"):
                coordinates = np.vstack((coordinates,last_frame_right_hand))
            
            break

        if counter == 0 :
            #by convention, we only select first FACESIZE lines of data
            if not column_name.startswith("face0"):
                coordinates = FACE_EMPTY
                last_frame_head = FACE_EMPTY
                if not column_name.startswith("pose0"):
                    coordinates = np.vstack((coordinates,POSE_EMPTY))
                    last_frame_pose = POSE_EMPTY
                    if not column_name.startswith("left0"):
                        coordinates = np.vstack((coordinates,HAND_EMPTY))
                        last_frame_left_hand = HAND_EMPTY

                coordinates1 = data.iloc[:, counter*3:counter*3+3].dropna().head(FACESIZE).to_numpy()
                coordinates = np.vstack((coordinates,coordinates1))

                if column_name.startswith("face0"):
                    last_frame_head = copy.deepcopy(coordinates1)
                elif column_name.startswith("pose0"):
                    last_frame_pose = copy.deepcopy(coordinates1)
                elif column_name.startswith("left0"):
                    last_frame_left_hand = copy.deepcopy(coordinates1)
                    is_left = True
                elif column_name.startswith("right0"):
                    last_frame_right_hand = copy.deepcopy(coordinates1)
                    is_left = False

            else:
                coordinates = data.iloc[:, counter*3:counter*3+3].dropna().head(FACESIZE).to_numpy()

                if column_name.startswith("face0"):
                    last_frame_head = copy.deepcopy(coordinates)
                elif column_name.startswith("pose0"):
                    last_frame_pose = copy.deepcopy(coordinates)
                elif column_name.startswith("left0"):
                    last_frame_left_hand = copy.deepcopy(coordinates)
                    is_left = True
                elif column_name.startswith("right0"):
                    last_frame_right_hand = copy.deepcopy(coordinates)
                    is_left = False

            if extract_frame_number(next_column_name)==1:
                if next_column_name.startswith("face"):
                    if coordinates.size == FACESIZE*3 :
                    #coordinates = data.iloc[:, counter*3:counter*3+3].dropna().head(FACESIZE).to_numpy()
                        coordinates = np.vstack((coordinates,last_frame_pose))
                        coordinates = np.vstack((coordinates,last_frame_left_hand))
                        coordinates = np.vstack((coordinates,last_frame_right_hand))
                        
                    elif coordinates.size == POSESIZE*3:
                        coordinates = np.vstack((coordinates,last_frame_left_hand))
                        coordinates = np.vstack((coordinates,last_frame_right_hand))
                        
                    elif coordinates.size == HANDSIZE*3 and is_left == True:
                        coordinates = np.vstack((coordinates,last_frame_right_hand))
                        
                    
                elif next_column_name.startswith("pose"):
                    if coordinates.size == FACESIZE*3 :
                    #coordinates = data.iloc[:, counter*3:counter*3+3].dropna().head(FACESIZE).to_numpy()
                        coordinates = np.vstack((coordinates,last_frame_pose))
                        coordinates = np.vstack((coordinates,last_frame_left_hand))
                        coordinates = np.vstack((coordinates,last_frame_right_hand))
                        
                        coordinates = np.vstack((coordinates,last_frame_head))
                    elif coordinates.size == POSESIZE*3:
                        coordinates = np.vstack((coordinates,last_frame_left_hand))
                        coordinates = np.vstack((coordinates,last_frame_right_hand))
                        
                        coordinates = np.vstack((coordinates,last_frame_head))
                    elif coordinates.size == HANDSIZE*3 and is_left == True:
                        coordinates = np.vstack((coordinates,last_frame_right_hand))
                        
                        coordinates = np.vstack((coordinates,last_frame_head))
                    elif coordinates.size == HANDSIZE*3 and is_left == False:
                        coordinates = np.vstack((coordinates,last_frame_head))
                elif next_column_name.startswith("left"):
                    #coordinates = data.iloc[:, counter*3:counter*3+3].dropna().head(FACESIZE).to_numpy()
                    if coordinates.size == FACESIZE*3 :
                        coordinates = np.vstack((coordinates,last_frame_pose))
                        coordinates = np.vstack((coordinates,last_frame_left_hand))
                        coordinates = np.vstack((coordinates,last_frame_right_hand))
                        
                        coordinates = np.vstack((coordinates,last_frame_head))
                        coordinates = np.vstack((coordinates,last_frame_pose))
                    elif coordinates.size == POSESIZE*3:
                        coordinates = np.vstack((coordinates,last_frame_left_hand))
                        coordinates = np.vstack((coordinates,last_frame_right_hand))
                        
                        coordinates = np.vstack((coordinates,last_frame_head))
                        coordinates = np.vstack((coordinates,last_frame_pose))
                    elif coordinates.size == HANDSIZE*3 and is_left == True:
                        coordinates = np.vstack((coordinates,last_frame_right_hand))
                        
                        coordinates = np.vstack((coordinates,last_frame_head))
                        coordinates = np.vstack((coordinates,last_frame_pose))
                    elif coordinates.size == HANDSIZE*3 and is_left == False:
                        coordinates = np.vstack((coordinates,last_frame_head))
                        coordinates = np.vstack((coordinates,last_frame_pose))
                
                elif next_column_name.startswith("right"):
                    if coordinates.size == FACESIZE*3 :
                        coordinates = np.vstack((coordinates,last_frame_pose))
                        coordinates = np.vstack((coordinates,last_frame_left_hand))
                        coordinates = np.vstack((coordinates,last_frame_right_hand))
                        
                        coordinates = np.vstack((coordinates,last_frame_head))
                        coordinates = np.vstack((coordinates,last_frame_pose))
                        coordinates = np.vstack((coordinates,last_frame_left_hand))
                    elif coordinates.size == POSESIZE*3:
                        coordinates = np.vstack((coordinates,last_frame_left_hand))
                        coordinates = np.vstack((coordinates,last_frame_right_hand))
                        
                        coordinates = np.vstack((coordinates,last_frame_head))
                        coordinates = np.vstack((coordinates,last_frame_pose))
                        coordinates = np.vstack((coordinates,last_frame_left_hand))
                    elif coordinates.size == HANDSIZE*3 and is_left == True:
                        coordinates = np.vstack((coordinates,last_frame_right_hand))
                        
                        coordinates = np.vstack((coordinates,last_frame_head))
                        coordinates = np.vstack((coordinates,last_frame_pose))
                        coordinates = np.vstack((coordinates,last_frame_left_hand))
                    elif coordinates.size == HANDSIZE*3 and is_left == False:
                        coordinates = np.vstack((coordinates,last_frame_head))
                        coordinates = np.vstack((coordinates,last_frame_pose))
                        coordinates = np.vstack((coordinates,last_frame_left_hand))
                #-----------------------#
        else :
            
            coordinates1 = data.iloc[:, counter*3:counter*3+3].dropna().head(FACESIZE).to_numpy()
            if column_name.startswith("face"):
                if current_frame == next_column_frame:
                    if next_column_name.startswith("pose"):
                        coordinates = np.vstack((coordinates,coordinates1))
                        last_frame_head = copy.deepcopy(coordinates1)
                    elif next_column_name.startswith("left"):
                        
                        coordinates = np.vstack((coordinates,coordinates1))
                        coordinates = np.vstack((coordinates,last_frame_pose))
                        last_frame_head = copy.deepcopy(coordinates1)
                    elif next_column_name.startswith("right"):
                        
                        coordinates = np.vstack((coordinates,coordinates1))
                        coordinates = np.vstack((coordinates,last_frame_pose))
                        coordinates = np.vstack((coordinates,last_frame_left_hand))
                        last_frame_head = copy.deepcopy(coordinates1)
                else:
                    
                    coordinates = np.vstack((coordinates,coordinates1))
                    coordinates = np.vstack((coordinates,last_frame_pose))
                    coordinates = np.vstack((coordinates,last_frame_left_hand))
                    coordinates = np.vstack((coordinates,last_frame_right_hand))
                    last_frame_head = copy.deepcopy(coordinates1)

                    if next_column_name.startswith("pose"):
                        coordinates = np.vstack((coordinates,last_frame_head))
                    elif next_column_name.startswith("left"):
                        coordinates = np.vstack((coordinates,last_frame_head))
                        coordinates = np.vstack((coordinates,last_frame_pose))
                    elif next_column_name.startswith("right"):
                        coordinates = np.vstack((coordinates,last_frame_head))
                        coordinates = np.vstack((coordinates,last_frame_pose))
                        coordinates = np.vstack((coordinates,last_frame_left_hand))

                

            elif column_name.startswith("pose"):
                if current_frame == next_column_frame:
                    if next_column_name.startswith("left"):
                        coordinates = np.vstack((coordinates,coordinates1))
                        last_frame_pose = copy.deepcopy(coordinates1)
                    elif next_column_name.startswith("right"):
                        coordinates = np.vstack((coordinates,coordinates1))
                        coordinates = np.vstack((coordinates,last_frame_left_hand))
                        last_frame_pose = copy.deepcopy(coordinates1)
                else:
                    coordinates = np.vstack((coordinates,coordinates1))
                    coordinates = np.vstack((coordinates,last_frame_left_hand))
                    coordinates = np.vstack((coordinates,last_frame_right_hand))
                    last_frame_pose = copy.deepcopy(coordinates1)

                    if next_column_name.startswith("pose"):
                        coordinates = np.vstack((coordinates,last_frame_head))
                    elif next_column_name.startswith("left"):
                        coordinates = np.vstack((coordinates,last_frame_head))
                        coordinates = np.vstack((coordinates,last_frame_pose))
                    elif next_column_name.startswith("right"):
                        coordinates = np.vstack((coordinates,last_frame_head))
                        coordinates = np.vstack((coordinates,last_frame_pose))
                        coordinates = np.vstack((coordinates,last_frame_left_hand))

                
            elif column_name.startswith("left"):
                if current_frame == next_column_frame:
                    coordinates = np.vstack((coordinates,coordinates1))
                    last_frame_left_hand = copy.deepcopy(coordinates1)

                else:
                    coordinates = np.vstack((coordinates,coordinates1))
                    coordinates = np.vstack((coordinates,last_frame_right_hand))
                    last_frame_left_hand = copy.deepcopy(coordinates1)

                    if next_column_name.startswith("pose"):
                        coordinates = np.vstack((coordinates,last_frame_head))
                    elif next_column_name.startswith("left"):
                        coordinates = np.vstack((coordinates,last_frame_head))
                        coordinates = np.vstack((coordinates,last_frame_pose))
                    elif next_column_name.startswith("right"):
                        coordinates = np.vstack((coordinates,last_frame_head))
                        coordinates = np.vstack((coordinates,last_frame_pose))
                        coordinates = np.vstack((coordinates,last_frame_left_hand))

                
            elif column_name.startswith("right"):
                coordinates = np.vstack((coordinates,coordinates1))
                last_frame_right_hand = copy.deepcopy(coordinates1)

                if next_column_name.startswith("pose"):
                    coordinates = np.vstack((coordinates,last_frame_head))
                elif next_column_name.startswith("left"):
                    coordinates = np.vstack((coordinates,last_frame_head))
                    coordinates = np.vstack((coordinates,last_frame_pose))
                elif next_column_name.startswith("right"):
                    coordinates = np.vstack((coordinates,last_frame_head))
                    coordinates = np.vstack((coordinates,last_frame_pose))
                    coordinates = np.vstack((coordinates,last_frame_left_hand))

                
            
        counter = counter + 1 
        

    print(coordinates.shape)
    coordinates=coordinates.flatten()
    print(coordinates.shape)

    directory, full_filename = os.path.split(file_path)
    new_filename = os.path.splitext(full_filename)[0] + '.npy'
    output_file_path = os.path.join(directory, new_filename)
    np.save(output_file_path, coordinates)
#    np.savetxt('output1.txt', coordinates, delimiter='\t')

#np.savetxt('output1.txt', coordinates, delimiter='\t')
#print(num_columns)



