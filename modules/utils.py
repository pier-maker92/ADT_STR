from pathlib import Path


def midi_explorer(dataset, path):  # path(str)->paths(list)
    if dataset == "lakh":
        return list(Path(path).glob("*.mid"))

    if dataset == "lakh_matched":
        return list(Path(path).rglob("*.mid"))

    if dataset == "guitarset":
        return list(Path(path).glob("*.jams"))

    if dataset == "maestro":
        files = []
        with open(path.split(",")[0] + "/maestro-v3.0.0.csv", "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                if row[2] == path.split(",")[1]:
                    files.append(path.split(",")[0] + "/" + row[4])
        return files
    if dataset == "slakh":
        return list(Path(path).glob("**/MIDI/*.mid"))
    if dataset == "molina":
        return list(Path(path).glob("*.GroundTruth.txt"))
    if dataset == "phenicx":
        return list(Path(path).glob("*.mid"))

    if dataset == "urmp":
        return list(Path(path).rglob("Notes*.txt"))

    if dataset == "gaps":
        return list(Path(path).glob("*.mid"))

    if dataset == "other":
        return list(Path(path).glob("*.mid"))

    if dataset == "maps":
        return list(Path(path).glob("*.txt"))
