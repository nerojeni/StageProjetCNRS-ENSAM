""" Mettre les fichiers dans un dossier et nommé le dossier avec le nom du matériel étudié
    et mettre le chemin d'accès vers ce dossier lors de l'exécution du programme.
    !!!!! Ne pas exécuter plusieurs fois le code, ca va inscrire plusieurs fois les lignes !!!!!
    Exécuter une fois et vérifier les fichiers 
    L'exécution est assez rapide."""

import os

def write_folder_name_in_files(folder_path):
    # Récupérer le nom du dossier
    folder_name = os.path.basename(folder_path)
    
    # Parcourir tous les fichiers dans le dossier
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Vérifier si c'est un fichier
        if os.path.isfile(file_path):
            with open(file_path, 'a') as file:
                # Écrire le nom du dossier à la fin du fichier
                file.write(f"\nMaterial={folder_name}")

def process_all_subfolders(parent_folder_path):
    # Parcourir tous les sous-dossiers dans le dossier parent
    for subfolder_name in os.listdir(parent_folder_path):
        subfolder_path = os.path.join(parent_folder_path, subfolder_name)
        
        # Vérifier si c'est un dossier
        if os.path.isdir(subfolder_path):
            # Appeler la fonction pour écrire le nom du sous-dossier dans chaque fichier
            write_folder_name_in_files(subfolder_path)

# Spécifiez le chemin de votre dossier parent
parent_folder_path = 'C:/Users/neroj/Materiaux'

# Appeler la fonction pour traiter tous les sous-dossiers
process_all_subfolders(parent_folder_path)

