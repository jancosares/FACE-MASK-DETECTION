import os
import xml.etree.ElementTree as ET
import re

# Set your dataset path
dataset_path = "C:/Users/Janine/Desktop/3RD YR - 2ND SEM/CSC 126/FINAL PROJECT/dataset"

annotations_dir = os.path.join(dataset_path, "annotations")
labels_dir = os.path.join(dataset_path, "labels")
train_labels_dir = os.path.join(labels_dir, "train")
val_labels_dir = os.path.join(labels_dir, "val")

# Create folders if not exist
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Define your classes - make sure these match exactly the Pascal VOC labels in your XMLs
classes = ["with_mask", "without_mask", "mask_weared_incorrect"]

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

# Regex to extract number from filename
pattern = re.compile(r'maksssksksss(\d+)\.xml')

for xml_file in os.listdir(annotations_dir):
    if not xml_file.endswith('.xml'):
        continue

    # Parse XML
    tree = ET.parse(os.path.join(annotations_dir, xml_file))
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # Prepare output txt filename (change .xml to .txt)
    txt_filename = xml_file.replace('.xml', '.txt')

    # Determine if file goes to train or val
    match = pattern.match(xml_file)
    if not match:
        print(f"Skipping {xml_file} due to no match")
        continue
    index = int(match.group(1))

    if index <= 682:
        label_path = os.path.join(train_labels_dir, txt_filename)
    else:
        label_path = os.path.join(val_labels_dir, txt_filename)

    with open(label_path, 'w') as out_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (
                float(xmlbox.find('xmin').text),
                float(xmlbox.find('xmax').text),
                float(xmlbox.find('ymin').text),
                float(xmlbox.find('ymax').text),
            )
            bb = convert((w, h), b)
            out_file.write(f"{cls_id} {' '.join(f'{a:.6f}' for a in bb)}\n")

print("Conversion and sorting done!")