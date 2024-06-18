import os
import json

def InfoImage(filepath):
    fichiers_text = []
    for fichier in os.listdir(filepath):
        if fichier.endswith(".txt"):
            with open(os.path.join(filepath, fichier), 'r') as fp:
                file_info = {}  # Dictionnaire pour stocker les informations du fichier
                lines = fp.readlines()
                for line in lines:
                    if '=' in line:
                        parts = line.strip().split("=")
                        if len(parts) == 2:  # Vérifier s'il y a bien deux parties
                            key, value = parts
                            file_info[key] = value
                if "ImageName" in file_info and "Date" in file_info:
                    fichiers_text.append(file_info)
    return fichiers_text

# Chemin du répertoire contenant les fichiers texte
filepath = "D:/Stage/UsersVeryOLD_C/giov"

# Appel de la fonction et récupération des résultats

Giov_2009 = InfoImage(filepath)

# Sauvegarde des données dans un fichier JSON
with open("Data/Giov_2009.json", "w") as json_file:
    json.dump(Giov_2009, json_file)

print("Données sauvegardées dans Amir_Mohammed.json")