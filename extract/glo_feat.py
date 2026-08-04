"""Global cortical measures parsed from FreeSurfer/FastSurfer ``*.aparc.stats``.

Used as the *brain morphology* pretext target: whole-hemisphere summary
measures (cortical volume, surface area, mean thickness, ...) plus per-region
columns, parsed straight from the stats files written by the recon pipeline.

Called by :mod:`extract_all_features`; not meant to be run standalone.
"""
import re


def sentence_to_dict(sentence, prefix):
    result = {}
    for i, line in enumerate(sentence.split("\n")[:-1]):
        elements = line.split(", ")
        if i ==3: break
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
        for i, header in enumerate(headers[3:10]):
            data_dict[f"{prefix}_{row_data[0]}_{header}"] = float(row_data[i+1].strip())

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
