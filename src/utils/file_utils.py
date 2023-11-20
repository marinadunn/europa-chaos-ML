import os
import shutil


def clear_and_remake_directory(dir_path):
    """
    Clears and recreates a directory if it exists, or creates it if it doesn't.

    Parameters:
        dir_path (str): The directory path.
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def make_dir(dir_path):
    """
    Creates a directory if it doesn't exist.

    Parameters:
        dir_path (str): The directory path.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def create_output_csv(csv_path, csv_header):
    """
    Creates a new CSV file with the given header.

    Parameters:
        csv_path (str): The path to the CSV file.
        csv_header (str): The header string to write to the CSV file.
    """
    with open(csv_path, "w") as csv_file:
        csv_file.write(csv_header)


def append_input_to_file(file_path, input_data):
    """
    Appends input data to a file.

    Parameters:
        file_path (str): The path to the file.
        input_data (str): The data to append to the file.
    """
    with open(file_path, "a") as csv_file:
        csv_file.write(f"{input_data}\n")


def clear_file(file_path):
    """
    Clears the content of a file.

    Parameters:
        file_path (str): The path to the file.
    """
    with open(file_path, "w") as csv_file:
        csv_file.write("")
