"""Local (per-region) cortical measures parsed from ``*.aparc.stats``.

Complements :mod:`glo_feat`: the same stats files are re-parsed to keep the
region-wise columns (thickness and curvature per parcel) that form the local
half of the *brain morphology* pretext target.

Called by :mod:`extract_all_features`; not meant to be run standalone.
"""
import re


def sentence_to_dict(sentence, prefix):
    result = {}
    for i, line in enumerate(sentence.split("\n")[:-1]):
        elements = line.split(", ")
        if i !=2: continue
        key = elements[2]
        value = float(elements[3])
        result[f"{prefix}_{key}"] = value
    return result

def sentence_to_dict2(sentence, prefix):
    headers = sentence.split("\n")[0].split(" ")[:19]
    data = sentence.split("\n")[1:]
    data_dict = {}
    for row in data[:-1]:
        row_data = re.split(r'\s+', row)
        for i, header in enumerate(headers[3:]):
            if i == 3:
                data_dict[f"{prefix}_{row_data[0]}_{header}"] = float(row_data[i+1].strip())
            if i == 5:
                data_dict[f"{prefix}_{row_data[0]}_{header}"] = float(row_data[i+1].strip()) * 20

    return data_dict

def file_to_dict(path, prefix):
    with open(path, "r") as file:
        contents = file.read()

    result = {}
    sentence = contents[contents.find("# Measure Cortex, NumVert,"):contents.find("# NTableCols 10")]
    result.update(sentence_to_dict(sentence, prefix))
    sentence2 = contents[contents.find("# ColHeaders"):]
    result.update(sentence_to_dict2(sentence2, prefix))

    return result
