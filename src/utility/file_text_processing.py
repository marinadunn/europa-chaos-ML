def create_output_csv(csv_path, csv_header):
    with open(csv_path, "w") as csv_file:
        csv_file.write(csv_header)

def append_input_to_file(file_path, input):
    with open(file_path, "a") as csv_file:
        text_line = f"{input}\n"
        csv_file.write(text_line)

def clear_file(file_path):
    with open(file_path, "w") as csv_file:
        text_line = f""
        csv_file.write(text_line)