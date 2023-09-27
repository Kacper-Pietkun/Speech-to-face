import os
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from deepface import DeepFace
import pandas as pd
from tqdm import tqdm
import shutil
from statistics import mean


ACCEPTED_IMAGE_EXTENSIONS = ['.jpg']

parser = ArgumentParser(description="Spliting FaceDecoderDataset into train/val/test sets but \
                        in a way that each set contains the same ration for race, age group and gender")

parser.add_argument("--dataset-path", type=str, required=True,
                    help="Path to the root folder of the whole dataset")

parser.add_argument("--save-path", type=str, required=True,
                    help="Path to the root folder where train/validation/test splits will be saved")

parser.add_argument("--train-size", type=float, default=0.8,
                    help="Proportion of the dataset to include in train split (should be between 0.0 and 1.0) \
                        test and validation set sizes are the same")

parser.add_argument("--race", action="store_true",
                    help="Set if you want to stratify dataseet based on people's race")

parser.add_argument("--age", action="store_true",
                    help="Set if you want to stratify dataseet based on people's age")

parser.add_argument("--gender", action="store_true",
                    help="Set if you want to stratify dataseet based on people's gender")


def save_set(set, split_type, args):
    for row in set.iterrows():
        person_directory_name = os.path.basename(os.path.normpath(row[1]["path"]))
        src_path = os.path.join(args.dataset_path, person_directory_name)
        dest_path = os.path.join(args.save_path, split_type, person_directory_name)
        shutil.copytree(src_path, dest_path)


def get_age_group(age):
    age_groups = {
        # Group Name and and the age from which one is included in that group
        "Child": 0,
        "YoungAdult": 18,
        "MiddleAgedAdult": 30,
        "OldAgedAdults": 45
    }
    selected_key = ""
    for key, value in age_groups.items():
        if age < value:
            break
        selected_key = key 
    assert selected_key != ""
    return selected_key


def analyze_image(file_path, actions):
    try:
        analysis = DeepFace.analyze(file_path, actions=actions)
        analysis = analysis[0]
    except ValueError as e:
        # if Face was not detected, return default values
        return {
            "age": 35,
            "dominant_race": "white",
            "dominant_gender": "Man"
        }
    return analysis


def analyze_person(person_folder, actions, args):
    """
    Determine statistics of the person basing on all of images in this person's directory
    """
    person_statistics = {
        "age_groups": [],
        "races": [],
        "genders": []
    }
    for file_name in os.listdir(person_folder):
        file_path = os.path.join(person_folder, file_name)
        if os.path.isfile(file_path):
            _, extension = os.path.splitext(file_name)
            if extension not in ACCEPTED_IMAGE_EXTENSIONS:
                continue
            analysis = analyze_image(file_path, actions)
            if args.age:
                person_statistics["age_groups"].append(analysis["age"])
            if args.race:
                person_statistics["races"].append(analysis["dominant_race"])
            if args.gender:
                person_statistics["genders"].append(analysis["dominant_gender"])

    processed_person = {
        "path": person_folder,
        "age_group": get_age_group(mean(person_statistics["age_groups"])) if args.age else "NaN",
        "race": max(person_statistics["races"], key=person_statistics["races"].count) if args.race else "NaN",
        "gender": max(person_statistics["genders"], key=person_statistics["genders"].count) if args.gender else "NaN"
    }
    return processed_person


def main():
    args = parser.parse_args()

    actions = []
    if args.age:
        actions.append("age")
    if args.gender:
        actions.append("gender")
    if args.race:
        actions.append("race")

    dataset_rows = []
    for directory_name in tqdm(os.listdir(args.dataset_path)):
        person_directory = os.path.join(args.dataset_path, directory_name)
        if os.path.isdir(person_directory):
            try:
                dataset_rows.append(analyze_person(person_directory, actions, args))
            except:
                continue

    tagged_dataset_df = pd.DataFrame(dataset_rows)
    tagged_dataset_df["merged_label"] = (tagged_dataset_df["age_group"].astype(str) + "_" if args.age else "_") + \
                                        (tagged_dataset_df["race"].astype(str) + "_" if args.race else "_") + \
                                        (tagged_dataset_df["gender"].astype(str) if args.gender else "_")

    train_set_x, rest_x, _, rest_y = train_test_split(tagged_dataset_df, tagged_dataset_df["merged_label"], random_state=42)
    val_set_x, test_set_x, _, _ = train_test_split(rest_x, rest_y, random_state=42)
    save_set(train_set_x, "train", args)
    save_set(val_set_x, "validation", args)
    save_set(test_set_x, "test", args)


if __name__ == "__main__":
    main()
