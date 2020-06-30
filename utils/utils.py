import os
import shutil

def count_data():
    count_dict = {}
    data_dir = "data/flowers"

    for d in os.listdir(data_dir):
        n_images = (len([f for f in os.listdir(data_dir + "/" + d)] ))
        count_dict[d] = n_images
    return count_dict

def train_test(count_dict,train_size = 0.7):

    data_dir = "data/flowers"

    for path in ("data/train", "data/test"):
        for class_ in count_dict.keys():
            class_path = path + "/" + class_

            if not os.path.exists(class_path):
                os.mkdir(class_path)
            else:
                shutil.rmtree(class_path)
                os.mkdir(class_path)

    for key in count_dict:
        class_dir = data_dir + "/" + key
        
        class_dir_train = "data/train" + "/" + key
        class_dir_test = "data/test" + "/" + key

        n_train = round(count_dict[key] * .7)
        n_test = round(count_dict[key] * (1 - .7))
        
        train_i = 0
        for image in os.listdir(class_dir):
            if train_i <= n_train:
                shutil.move(class_dir + "/" + image, class_dir_test)
                
            else:
              
                shutil.move(class_dir + "/" + image, class_dir_train)
            
            train_i += 1

        


