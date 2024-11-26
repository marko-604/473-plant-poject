import os
import xml.etree.ElementTree as ET
import pandas as pd


#------------------------------|
NUM_OF_FILES = 700
#------------------------------|
TEST_DATA_DIR = "test"
TEST_OUTPUT_CSV = "test.csv"
#------------------------------|
TRAIN_DATA_DIR = "train"
TRAIN_OUTPUT_CSV = "train.csv"
#------------------------------|

#Purpose of makeCSV.py:

# Simplify the reading process of metadata of each image. Previously frequent errors when reading xml files would
# ignore certain values or misplace the correct categorization. Forming a single csv file for each train and test
# allows for easy data verification and increased accuracy.


def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = {
        "ObservationId": root.find("ObservationId").text if root.find("ObservationId") is not None else None,
        "MediaId": None,
        "Vote": root.find("Vote").text if root.find("Vote") is not None else None,
        "Content": root.find("Content").text if root.find("Content") is not None else None,
        "ClassId": root.find("ClassId").text if root.find("ClassId") is not None else None,
        "Family": root.find("Family").text if root.find("Family") is not None else None,
        "Genus": root.find("Genus").text if root.find("Genus") is not None else None,
        "Species": root.find("Species").text if root.find("Species") is not None else None,
        "Author": root.find("Author").text if root.find("Author") is not None else None,
        "Date": root.find("Date").text if root.find("Date") is not None else None,
        "Location": root.find("Location").text if root.find("Location") is not None else None,
        "Latitude": root.find("Latitude").text if root.find("Latitude") is not None else None,
        "Longitude": root.find("Longitude").text if root.find("Longitude") is not None else None,
        "YearInCLEF": root.find("YearInCLEF").text if root.find("YearInCLEF") is not None else None,
        "ObservationId2014": root.find("ObservationId2014").text if root.find("ObservationId2014") is not None else None,
        "ImageId2014": root.find("ImageId2014").text if root.find("ImageId2014") is not None else None,
        "LearnTag": root.find("LearnTag").text if root.find("LearnTag") is not None else None,
    }

    return data

def XML_TO_CSV(data_dir, output_csv, num_files=NUM_OF_FILES):
    all_metadata = []

    for i in range(1, num_files + 1):
        file_name = f"{i}.xml"
        file_path = os.path.join(data_dir, file_name)

        if os.path.exists(file_path):
            data = parse_xml(file_path)

            data["MediaId"] = i

            all_metadata.append(data)
            print(f"Processed {i}/{num_files} files...")
        else:
            print(f"Warning: {file_name} does not exist.")

    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(output_csv, index=False)

    print(f"Data written to {output_csv}.")

XML_TO_CSV(TEST_DATA_DIR, TEST_OUTPUT_CSV, num_files=NUM_OF_FILES)
XML_TO_CSV(TRAIN_DATA_DIR, TRAIN_OUTPUT_CSV, num_files=NUM_OF_FILES)
