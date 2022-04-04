from concept_reader import *
import typing as tp
import torchvision.transforms as transforms
from PIL import Image
import os
import pickle
import sys
# Run this script to add the concept examples to the files.



def preprocess_images_to_tensors(img_path: str, imglen_px=448):
    """ Cropping an preprocessing as we used it for our concept inputs. """
    tr = transforms.Compose([
            transforms.Resize((imglen_px//7)*8),
            transforms.CenterCrop(imglen_px),
            transforms.ToTensor(),
        ])
    with Image.open(img_path) as pil_img:
        ret_tensor = tr(pil_img)
    return ret_tensor


def get_filename_to_fullpath_dict(path_args: tp.List[str]):
    """ Get a dict of all filenames -> full path. Make sure each filename is unique. """
    ret_dict = {}
    for path_arg in path_args:
        print("Collecting files in path:", path_arg)
        for root, dirs, files in os.walk(path_arg, topdown=False):
            for name in files:
                my_filename = os.path.join(root, name)
                # print(my_filename, root)
                name_only = my_filename.split("/")[-1].split("\\")[-1]
                ret_dict[name] = my_filename
    print("Paths scanned. Found", len(ret_dict), "files.")
    return ret_dict


def add_images_to_concepts(concept_dir, search_paths: tp.List[str]):
    """ A preprocessing step required if you have downloaded the concepts without the example images
        from github. This function will retrieve the corresponding images from the datasets (that need to be stored on your machine).
        concept_dir: Path with the folders containing the pickle concepts.
        search_path: The paths to look for the images (for the oringinal papers, these are the val dirs of AwA and places365)
        Return number of concepts successfully processed.
    """
    filelist = get_filename_to_fullpath_dict(search_paths)

    # First read all concepts in concept_dir.
    c_count = 0
    for item in os.listdir(concept_dir):
        curr_dir = os.path.join(concept_dir, item)
        if os.path.isdir(curr_dir):
            for conceptf in os.listdir(curr_dir):
                file_name = os.path.join(curr_dir, conceptf)
                if os.path.isfile(file_name) and file_name.endswith(".pickle"):
                    # Try to read this concepts
                    try:
                        with open(file_name, 'rb') as pickleread:
                            my_concept = pickle.load(pickleread)
                        for example in my_concept:
                            search_name = example.path.split("/")[-1].split("\\")[-1] # only consider filename as key
                            full_name = filelist[search_name]
                            img_tensor = preprocess_images_to_tensors(full_name)
                            example.image_tensor = img_tensor
                        with open(file_name, 'wb') as picklewrite:
                            pickle.dump(my_concept, picklewrite)
                        print("Processed ", file_name)
                        c_count += 1
                    except ValueError:
                        print("No image file found for name", search_name)
                    except (FileNotFoundError, pickle.UnpicklingError):
                        print("Error reading file", file_name)

    return c_count

if __name__ == "__main___":
    if len(sys.argv) <= 2:
        print("Usage: python3 add_images_script.py <concept_path> <dataset_path1> <dataset_path2> ... <dataset_pathN>")
    else:
        print(sys.argv)
        ret = add_images_to_concepts(sys.argv[1], sys.argv[2:])
        print("Successfully added images to", ret, "concepts")