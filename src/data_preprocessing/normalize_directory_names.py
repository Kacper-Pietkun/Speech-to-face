import os
import string
import shutil
from argparse import ArgumentParser
from tqdm import tqdm


parser = ArgumentParser(description="Normalizing directory names so that they consist of ASCII letters only")

parser.add_argument("--data-dir", required=True, type=str,
                    help="Absolute path to the root directory of the dataset")


def main():
    args = parser.parse_args()
    for directory_name in tqdm(os.listdir(args.data_dir)):
        directory_path = os.path.join(args.data_dir, directory_name)
        if os.path.isdir(directory_path):
            normalized_directory_name = ''.join(c for c in directory_name if c in string.printable)
            new_directory_path = os.path.join(args.data_dir, normalized_directory_name)
            try:
                os.rename(directory_path, new_directory_path)
            except FileExistsError as e:
                shutil.copytree(directory_path, new_directory_path, dirs_exist_ok=True)
                shutil.rmtree(directory_path)


if __name__ == "__main__":
    main()
