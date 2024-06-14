import datetime
from dash import html, dcc, Dash, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import json
from io import BytesIO
import base64
import zipfile
import os
import shutil
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend non interactif
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import plotly.graph_objs as go
from pathlib import Path
from scipy import signal
import pywt
from scipy.stats import entropy
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
from skimage.color import rgb2gray

# Initialisation des variables pour éviter les erreurs (lors des appels dans les layouts)
results_json = {}
unique_images = {}
image_parameters = {}
final_json = {}
keys_to_extract = [
    'AcceleratingVoltage', 'DecelerationVoltage', 'EmissionCurrent',
    'WorkingDistance', 'PixelSize', 'SignalName', 'Magnificient',
    'LensMode', 'ScanSpeed'
]

# Fonction pour convertir une image en base64 pour affichage
def image_to_base64(image_path):
    image = io.imread(image_path) 
    image = remove_black_band(image)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    fig.patch.set_facecolor('black')
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return "data:image/png;base64," + img_str

""" Fonction calculant la somme des différences absolues de ligne consécutive 
    divisé par par la valeur maximale de cette somme. 
    Donc elle permet d'observer les variations interlignes pour chaque image. """
def F_1(list_image):
    result_F1 = np.zeros(len(list_image))

    for i, name in enumerate(list_image):
        image = np.array(io.imread(name, as_gray=True)) / 256 # Elle permet de mettre la valeur de pixel entre 0 et 1 
        result_F1[i] = np.sum(np.abs(np.diff(image, axis=0)))
    result_F1 = result_F1 - np.min(result_F1)
    return result_F1 / np.max(result_F1)


""" Fonction calculant la même chose que F_1 
    mais la somme de la différence est élevée au carrée. """
def F_2(list_image):
    result_F2 = np.zeros(np.shape(list_image))

    for i, name in enumerate(list_image):
        image = np.array(io.imread(name, as_gray=True)) / 256
        result_F2[i] = np.sum(np.diff(image, axis=0) ** 2)
    result_F2 = result_F2 - np.min(result_F2)
    return result_F2 / np.max(result_F2)


""" Capture les changements dans les variations d'intensité de pixels 
    --> Changement dans la texture ou motifs de l'image. """
def F_3(list_image):
    result_F3 = np.zeros(np.shape(list_image))

    for i, name in enumerate(list_image):
        image = np.array(io.imread(name, as_gray=True)) / 256
        result_F3[i] = np.sum(np.diff(image, n=2, axis=0) ** 2) #n=2 --> seconde dérivée discrète
    result_F3 = result_F3 - np.min(result_F3)
    return result_F3 / np.max(result_F3)


""" Utilise l'opérateur de Sobel (filtre de convultion) pour détecter les bords (intensité des bords)
    valeur élevée indique forte présence de bord == image avec beaucoup de détails, de contours nets.
    valeur faible indique faible présence de bord == image homogène ou flou. """
def F_4(list_image):
    result_F4 = np.zeros(np.shape(list_image))

    Sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    for i, name in enumerate(list_image):
        image = np.array(io.imread(name, as_gray=True)) / 256
        Cx = signal.convolve2d(image, Sx, mode='same', boundary='fill')
        Cy = signal.convolve2d(image, Sy, mode='same', boundary='fill')
        result_F4[i] = np.sum(Cx ** 2 + Cy ** 2)
    result_F4 = result_F4 - np.min(result_F4)
    return result_F4 / np.max(result_F4) 


"""" Calcule une mesure de la variation dans une liste d'image en utilisant les gradients
     et les dérivées secondes des images. 
     Dx, Dy représente respectivement les variatations horizontal et vertical des intensités de pixels.
     Lxx, Lyy capturent la courbure de l'image dans les directions x, y.
     
     @return somme des carrés des dérivées secondes dans une mesure de l'intensité des variations
     de courbe et de texture dans l'image divisé par la valeur maximale de cette somme. 
     
     valeur élevée indique des détails fins dans l'image."""
def F_5(list_image):
    result_F5 = np.zeros(np.shape(list_image))

    for i, name in enumerate(list_image):
        image = np.array(io.imread(name, as_gray=True)) / 256
        Dx, Dy = np.gradient(image, edge_order=1)
        Lxx, Lxy = np.gradient(Dx, edge_order=1)
        Lxy, Lyy = np.gradient(Dy, edge_order=1)
        result_F5[i] = np.sum(Lxx ** 2 + Lyy ** 2)
    result_F5 = result_F5 - np.min(result_F5)
    return result_F5 / np.max(result_F5)


""" Détecte les variations de texture à l'aide d'un filtre Laplacien (opérateur diff mettant 
    en évidence les zones de variations rapide des intensités de pixels (bords, textures))
    
    @return sommme de la valeur de convolution divisée par maximum de cette somme. 
    
    valeur élevée indique une forte variation de texture dans l'image. """
def F_6(list_image):
    result_F6 = np.zeros(np.shape(list_image))

    L = np.array([[-1, -4, -1], [-4, 20, -4], [-1, -4, -1]])
    
    for i, name in enumerate(list_image):
        image = np.array(io.imread(name, as_gray=True)) / 256
        C = signal.convolve2d(image, L, mode='same', boundary='fill')
        result_F6[i] = np.sum(C ** 2)
    result_F6 = result_F6 - np.min(result_F6)
    return result_F6 / np.max(result_F6)


""" Fonction utilisant la transformée en ondelette (wavelett transform)
    <=> technique mathématique pour décomposer une fonction ou un signal
    en composantes à différentes échelles et résolutions. 
    
    DWT décompose image en 4 sous parties : 
    --> LL : composante base fréquence dans les directions horizontales / verticales (approximativement) ; 
    --> LH : composante base fréquence en horizontale et haute fréquence en verticale (détails horizontaux) ;
    --> HL : composante haute fréquence en horizontale et base fréquence en verticale (détails verticaux) ;
    --> HH : composante haute fréquence dans les deux directions (détails diagonaux).
    
    @return somme des valeurs absolues de ces coefficients de détails (LH, HL, HH) {mesurant l'intensité des variations
    de texture et de détails dans l'image} / max de cette somme. 
    
    valeur élevée indique une forte présence de détails et de textures dans l'image et inversement. """
def F_7(list_image):
    result_F7 = np.zeros(np.shape(list_image))

    for i, name in enumerate(list_image):
        image = np.array(io.imread(name, as_gray=True)) / 256
        coeffs2 = pywt.dwt2(image, 'db6')
        LL, (LH, HL, HH) = coeffs2
        result_F7[i] = np.sum(np.abs(LH) + np.abs(HL) + np.abs(HH))
    result_F7 = result_F7 - np.min(result_F7)
    return result_F7 / np.max(result_F7)


""" Fonction qui utilise la décomposition en ondelette avec ajustement de la moyenne.
    Similaire à la F_7 mais la somme des coeff est différente 
    ici on calcule la différence entre la valeur absolue du coef - la moyenne de la valeur absolue du coef.
    
    @return (somme des des différences au carré / par le produite de la taille d'image) / divisé par la valeur max. """
def F_8(list_image):
    result = np.zeros(np.shape(list_image))
    
    for i, name in enumerate(list_image):
        image = np.array(io.imread(name, as_gray=True)) / 256
        LL, (LH, HL, HH) = pywt.dwt2(image, 'db6')
        LHm = np.abs(LH) - np.mean(np.abs(LH))
        HLm = np.abs(HL) - np.mean(np.abs(HL))
        HHm = np.abs(HH) - np.mean(np.abs(HH))
        result[i] = np.sum(LHm ** 2 + HLm ** 2 + HHm ** 2) / np.prod(np.shape(image))
    result = result - np.min(result)
    return result / np.max(result)


""" Fonction pareil que F_8 mais sans les valeurs absolues. """
def F_9(list_image):
    result = np.zeros(np.shape(list_image))
    
    for i, name in enumerate(list_image):
        image = np.array(io.imread(name, as_gray=True)) / 256
        LL, (LH, HL, HH) = pywt.dwt2(image, 'db6')
        LHm = LH - np.mean(LH)
        HLm = HL - np.mean(HL)
        HHm = HH - np.mean(HH)
        result[i] = np.sum(LHm ** 2 + HLm ** 2 + HHm ** 2) / np.prod(np.shape(image))
    result = result - np.min(result)
    return result / np.max(result)


""" Calcule la variance des pixels de chaque image. Dipsersion des intensités des pixels autour de la moyenne
    <=> quantification de la variabilité de l'intensité de l'image. 
    valeur élevée indique une forte variabilité de l'intensité des pixels == contrastes marqués / variations importantes."""
def F_10(list_image):
    result = np.zeros(np.shape(list_image))
    
    for i, name in enumerate(list_image):
        image = np.array(io.imread(name, as_gray=True)) / 256
        mean = np.mean(image)
        result[i] = np.sum((image - mean) ** 2) / np.prod(np.shape(image))
    result = result - np.min(result)
    return result / np.max(result)


""" Similaire à la F_10 mais calcule la variance relative (var / moy)
    valeur élevée signifie qu'il y a une forte variabilité relative de l'intensité
    des pixels par rapport à la moyenne == contrastes marqués. """
def F_11(list_image):
    result = np.zeros(np.shape(list_image))
    
    for i, name in enumerate(list_image):
        image = np.array(io.imread(name, as_gray=True)) / 256
        mean = np.mean(image)
        result[i] = np.sum((image - mean) ** 2) / (np.prod(np.shape(image)) * mean)
    result = result - np.min(result)
    return result / np.max(result)


""" AutoCorrelation"""
def F_12(list_image):
    result = np.zeros(np.shape(list_image))
    
    for i, name in enumerate(list_image):
        image = np.array(io.imread(name, as_gray=True)) / 256
        
        # Décaler l'image de 1 pixel vers la droite
        img_1 = np.roll(image, shift=1, axis=0)
        
        # Remplacer la première ligne décalée par des zéros 
        img_1[0, :] = 0
        
        # Décaler l'image de 2 pixels vers la droite
        img_2 = np.roll(image, shift=2, axis=0)
        
        # Remplacer les premières lignes décalées par des zéros 
        img_2[:2]=0
        
        sum1 = np.sum(image * img_1)
        sum2 = np.sum(image * img_2)
        
        result[i] = sum1 - sum2
        
    result = result - np.min(result)
        
    return result / np.max(result) 


""" Standard-deviation-based correlation"""
def F_13(list_image,axis=0):
    result = np.zeros(len(list_image))
    
    for i, name in enumerate(list_image):
        # Lire l'image en niveaux de gris et la normaliser
        image = np.array(io.imread(name, as_gray=True)) / 256
        H, W = np.shape(image)
        mean = np.mean(image)
        
        # Décaler l'image de 1 pixel vers la droite
        img_1 = np.roll(image, shift=1, axis=axis)
        
        # Remplacer la première ligne décalée par des zéros pour éviter les artefacts
        if axis == 0 : 
            img_1[0, :] = 0
        elif axis == 1 : 
            img_1[:,0] = 0
        else : 
            print('erreur F_13')
            break
        
        # Calculer la somme des produits des pixels adjacents
        product_sum = np.sum(image * img_1)
        
        # Appliquer la formule
        result[i] = product_sum - H * W * (mean ** 2)
        
    result = result - np.min(result)
    
    return result / np.max(result)

""" Range Algorithm - différence entre l'intensité max et min d'une image"""
def F_14(list_image):
    result = np.zeros(len(list_image))
    
    for i, name in enumerate(list_image):
        image = np.array(io.imread(name, as_gray=True))
        
        hist = [np.count_nonzero(image==j) for j in range (256)]
        
        hist1 = []
        for h in hist : 
            if h > 0 : 
                hist1.append(h)
        
        max = np.max(hist1)
        min =  np.min(hist1)
        
        result[i] = max - min
        
    result = result - np.min(result)
            
    return result / np.max(result)
    
    
""" Entropy algorithm -- image focus a plus d'information que image défocus """
def F_15(list_image):
    result = np.zeros(len(list_image))
    
    for i, name in enumerate(list_image):
        # Lire l'image en niveaux de gris et la normaliser
        image = np.array(io.imread(name, as_gray=True)) / 256
        # Calculer l'histogramme des intensités des pixels et normaliser pour obtenir p(i)
        pi, _ = np.histogram(image, bins=256, range=(0, 1))
        pi = list(filter(lambda p: p>0, np.ravel(pi)))
        result[i] = entropy(pi, base = 2)
    
    result = result - np.min(result)
        
    return result / np.max(result)

""" Intuitive Algorithms """    
def F_16(list_image):
    result = np.zeros(len(list_image))
    
    for i, name in enumerate(list_image):
        # Lire l'image en niveaux de gris et la normaliser
        image = np.array(io.imread(name, as_gray=True)) / 256
        θ = np.mean(image)
        image = image >= θ 
        result[i] = np.sum(image)
    
    result = result - np.min(result)
     
    return result / np.max(result)


""" Thresholded pixel count """
def F_17(list_image): 
    result = np.zeros(len(list_image))
    
    for i, name in enumerate(list_image):
        # Lire l'image en niveaux de gris et la normaliser
        image = np.array(io.imread(name, as_gray=True)) / 256
        
        θ = np.mean(image)
        
       # Créer une image binaire où les pixels inférieurs ou égaux au seuil sont 1, sinon 0
        inf_theta = np.where(image <= θ)[0]
        
        # Compter le nombre de pixels inférieurs ou égaux au seuil
        result[i] = np.shape(inf_theta)[0]
    
    result = result - np.min(result)
        
    return result / np.max(result)

""" Image power """
def F_18(list_image):
    result = np.zeros(len(list_image))
    
    for i, name in enumerate(list_image):
        # Lire l'image en niveaux de gris et la normaliser
        image = np.array(io.imread(name, as_gray=True)) / 256
        
        θ = np.mean(image)
        
        image = image >= θ
        
        result[i] = np.sum(image**2)
    
    result = result - np.min(result)
        
    return result / np.max(result)

functions = [F_1, F_2, F_3, F_4, F_5, F_6, F_7, F_8, F_9, F_10, F_11, F_12, F_13, F_14, F_15, F_16, F_17, F_18]

# Fonction pour sauvegarder les nouveaux keywords
def save_keywords(keywords, file_path='keywords.json'):
    with open(file_path, 'w') as f:
        json.dump(keywords, f)

# Fonction pour récupérer les keywords du fichier json
def load_keywords(file_path='keywords.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return []

# Fonction pour sauvegarder les nouveaux types de matériaux
def save_materials(materials, file_path='materials.json'):
    with open(file_path, 'w') as f:
        json.dump(materials, f)

# Fonction pour récupérer les matériaux
def load_materials(file_path='materiaux.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return []

# Fonction pour enlever la bande noire
def remove_black_band(image, threshold=0.04, proportion_check=0.13):
    # Si l'image est en couleur, convertissez-la en niveaux de gris
    if len(image.shape) == 3:
        image = rgb2gray(image)
    
    # Proportion de l'image à vérifier pour la bande noire
    check_height = int(image.shape[0] * proportion_check)
    black = image[-check_height:, :]
    black_band = 0
    
    h, w = black.shape # taille de la bande noire
                
    for i in range(h):
        for j in range(w):
            pixel = black[i, j]
            if pixel >= 0 and pixel <= threshold:  
                black_band += 1
                
    # Vérifier si la bande noire doit être enlevée
    if black_band >= (h * w) * 0.5:  
        image = image[:-check_height, :]  # Enlever la bande noire
        
    return image

# Fonction pour traiter les images et les paramètres d'un dossier
def process_images_and_params(uploaded_files, save_dir):
    list_image = []
    params_json = {}
    
    # Récupération de la date du jour 
    creation_date = datetime.date.today()
    creation_date = creation_date.isoformat()

    # Sauvegarde temporaire pour le fonction de l'application
    temp_dir = Path(save_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    for file_content, filename in uploaded_files:
        content_type, content_string = file_content.split(',')
        decoded = base64.b64decode(content_string) # décoder les fichiers pour la lecture 
        file_path = temp_dir / filename
        with open(file_path, "wb") as f:
            f.write(decoded)
        if filename.endswith(('.bmp', '.jpg', '.png', 'tif')):
            list_image.append(str(file_path))
        elif filename.endswith('.txt'): # récupération des paramètres de tous les fichiers 
            image_name = filename.rsplit('.', 1)[0]
            with open(file_path, "r") as fp:
                file_info = {}
                lines = fp.readlines()
                for line in lines:
                    if '=' in line:
                        parts = line.strip().split("=")
                        if len(parts) == 2:
                            key, value = parts
                            file_info[key] = value
                params_json[image_name] = file_info

    image_names = [Path(img).stem for img in list_image]
    results_json = {}

    # Exécution des fonctions pour la liste d'image 
    for i, func in enumerate(functions, start=1):
        results = func(list_image)
        if results.size == 0:
            continue
        results_json[f'F_{i}'] = [
            {'path': list_image[j], 'name': image_names[j], 'result': results[j]}
            for j in range(len(list_image))
        ]

    for image_path in list_image:
        image = io.imread(image_path)
        image = remove_black_band(image)
        io.imsave(image_path, image)  # Enregistrer l'image modifiée

    # pour la création du json
    final_json = {
        "creation_date" : creation_date,
        "parameters": params_json,
        "results": results_json
    }
    
    # création du json dans le dossier temporaire
    json_path = temp_dir / "Results_Functions.json"
    if not json_path.exists(): # si le json n'existe pas déjà
        with open(json_path, "w") as f:
            json.dump(final_json, f, indent=4) # création du fichier json

    # Collecter keywords et Material :
    new_keywords = []
    new_materials = []
    for image_name, params in params_json.items():
        for key, value in params.items():
            if key in ['KeyWord1', 'KeyWord2']:
                new_keywords.append(value)
            elif key == 'Material':
                if value not in new_materials:
                    new_materials.append(value)

    existing_keywords = load_keywords()
    all_keywords = existing_keywords + new_keywords
    save_keywords(all_keywords)
    save_materials(new_materials)

    return final_json

# Fonction pour récupérer les paramètres d'image
def get_image_parameters(final_json):
    parameters = final_json.get("parameters", {})

    image_parameters = {}
    for image_name, param in parameters.items():
        image_parameters[image_name] = param

    return image_parameters

# Créer l'application Dash
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://bootswatch.com/5/solar/bootstrap.min.css'], suppress_callback_exceptions=True)

# Layout pour la page d'accueil avec la barre de navigation
navbar = dbc.NavbarSimple(
    brand="Analyse Images Microscopiques",
    brand_href="/",
    color="dark",
    dark=True,
    children=[
        dbc.NavItem(dbc.NavLink("Accueil", href="/accueil")),
        dbc.NavItem(dbc.NavLink("Tableau de Bord", href="/dashboard")),
        dbc.NavItem(dbc.NavLink("Paramètres Images", href="/paramsDashboard")),
        dbc.NavItem(dbc.NavLink("Toutes les Images", href="/photos"))
    ]
)

# Layout principal avec la barre de navigation et un conteneur pour le contenu dynamique
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content'),
    dcc.Input(id='input-keywords', type='text', value=json.dumps(load_keywords()), style={'display': 'none'}),
    dcc.Input(id='function_dropdown3', type='text', value=json.dumps([]), style={'display': 'none'}),
    dcc.Input(id='function_dropdown4', type='text', value=json.dumps([]), style={'display': 'none'}),
    dcc.Input(id='input-materiaux', type='text', value=json.dumps(load_materials()), style={'display': 'none'})
])

# La page d'accueil où on importe les fichiers
home_layout = html.Div(
    style={
        'display': 'flex',
        'flexDirection': 'column',
        'justifyContent': 'center',
        'alignItems': 'center',
        'height': '100vh',
        'textAlign': 'center'
    },
    children=[
        html.H1("Outils pour Analyser les Images Microscopiques", className='text-primary my-4'),
        html.H3("Insérer les images et paramètres associés afin de les traiter", className='text-secondary my-4'),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files', style={'color': '#007bff', 'text-decoration': 'underline'})
            ]),
            style={
                'width': '500px',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '20px 0',
                'backgroundColor': '#f9f9f9'
            },
            multiple=True
        ),
        html.Div(id='json-download-div', children=[
            html.Button("Download JSON", id='download-json-button', className='btn btn-primary', style={'display': 'none'}),
            dcc.Download(id='download-json')
        ], style={'textAlign': 'center', 'margin': '20px 0'}),
    ]
)

# Le tableau de bord principal qui contient seulement le résultat des fonctions individuellement et/ou simultanément
dashboard_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Filtres", className='text-center text-primary my-4'),
            html.H4("Sélectionnez un ou plusieurs matériaux", className='text-secondary mb-2', style={'color': '#123C69'}),
            dcc.Dropdown(
                id='material-dropdown-dashboard',
                options=[],
                multi=False,
                className='mb-4'
            ),
            html.H4("Sélectionnez les fonctions", className='text-secondary mb-2', style={'color': '#123C69'}),
            html.H5("A. Derivative-Based Algorithms", className='text-secondary mb-2', style={'color': '#123C69'}),
            dcc.Dropdown(
                id='function-dropdown-1',
                options=[{'label': f'F_{i}', 'value': i} for i in range(1, 10)],
                multi=True,
                className='mb-4'
            ),
            html.H5("B. Statistics-Based Algorithms", className='text-secondary mb-2', style={'color': '#123C69'}),
            dcc.Dropdown(
                id='function-dropdown-2',
                options=[{'label': f'F_{i}', 'value': i} for i in range(10, 14)],
                multi=True,
                className='mb-4'
            ),
            html.H5("C. Histogram-Based Algorithms", className='text-secondary mb-2', style={'color': '#123C69'}),
            dcc.Dropdown(
                id='function-dropdown-3',
                options=[{'label': f'F_{i}', 'value': i} for i in range(14, 19)],
                multi=True,
                className='mb-4'
            ),
            html.H4("Fonction individuelle", className='text-secondary mb-2'),
            dcc.Dropdown(
                id='individual-function-dropdown',
                options=[{'label': f'F_{i}', 'value': i} for i in range(1, 19)],
                className='mb-4'
            )
        ], xs=12, sm=12, md=12, lg=2, xl=2, style={"background-color": "white"}),

        dbc.Col([
            # Widget pour le graphique global
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("Images optimales et non optimales selon différentes fonctions", className='text-center text-success mb-3'),
                        dcc.Graph(id='function-graph', style={'margin': '1px', 'height': '500px', 'width': '100%'})
                    ], className='p-3 mb-4', style={'background-color': 'white', 'border-radius': '6px'})
                ], width=12),

                dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle("Image sélectionnée")),
                    dbc.ModalBody(id="modal-content"),
                ],
                id="modal",
                is_open=False,
                )
            ]),

            # Widget pour le graphique individuel avec les images associées
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("Graphique Individuel Optimal et Non Optimal", className='text-center text-success mb-3'),
                        dcc.Graph(id='individual-function-graph', style={'margin-right': '100px', 'height': '500px', 'width': '100%'})
                    ], className='p-3 mb-4', style={'background-color': 'white', 'border-radius': '6px'})
                ], width=12),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id='individual-function-image-optimal', style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),
                    html.P(id='optimal-image-name', style={'text-align': 'center', 'color': 'white'})
                ]),
                dbc.Col([
                    html.Div(id='individual-function-image-non-optimal', style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),
                    html.P(id='non-optimal-image-name', style={'text-align': 'center', 'color': 'white'})
                ])
            ], style={'display': 'flex', 'align-items': 'center'}),

            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle("Image sélectionnée")),
                dbc.ModalBody(id="modal-content2"),
            ],
            id="modal2",
            is_open=False,
            )
        ], xs=12, sm=12, md=12, lg=10, xl=10)

    ])

], fluid=True)

# Le second tableau de bord - interaction entre paramètres et résultats des fonctions
params_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Filtres", className='text-center text-primary my-4'),
            html.H4("Sélectionnez un ou plusieurs matériaux", className='text-secondary mb-2', style={'color': '#123C69'}),
            dcc.Dropdown(
                id='material-dropdown-dashboard',
                options=[],
                multi=False,
                className='mb-4'
            ),
            html.H4("A. Derivative-Based Algorithms ", className='text-secondary mb-2', style={'color': '#123C69'}),
            dcc.Dropdown(
                id='param-function-dropdown-1',
                options=[{'label': f'F_{i}', 'value': i} for i in range(1, 10)],
                value=1,
                multi=False,
                className='mb-4'
            ),
            html.H4("B. Statistics-Based Algorithms ", className='text-secondary mb-2', style={'color': '#123C69'}),
            dcc.Dropdown(
                id='param-function-dropdown-2',
                options=[{'label': f'F_{i}', 'value': i} for i in range(10, 14)],
                value=1,
                multi=False,
                className='mb-4'
            ),
            html.H4("C. Histogram-Based Algorithms ", className='text-secondary mb-2', style={'color': '#123C69'}),
            dcc.Dropdown(
                id='param-function-dropdown-3',
                options=[{'label': f'F_{i}', 'value': i} for i in range(14, 19)],
                value=1,
                multi=False,
                className='mb-4'
            ),
        ], xs=12, sm=12, md=12, lg=2, xl=2, style={"background-color": "white"}),

        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H2('Les combinaisons de paramètres utilisés selon le résultat des fonctions', className='text-center text-primary my-4'),
                        dcc.Graph(id='parallel-graph', style={'height': '500px', 'width': '100%'}),
                    ], className='p-3 mb-4', style={'background-color': 'white', 'border-radius': '6px'}),
                ], xs=12, sm=12, md=12, lg=12, xl=12),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H2("Les valeurs de paramètres utilisées pour les images optimales", className='text-center text-primary my-4'),
                        dcc.Graph(id='param-graph', style={'height': '800px', 'width': '100%'}),
                    ], className='p-3 mb-4', style={'background-color': 'white', 'border-radius': '6px'}),
                    dbc.Modal([
                        dbc.ModalHeader(dbc.ModalTitle("Images générées avec ce paramètre")),
                        dbc.ModalBody(id="modal-content3"),
                    ],
                    id="modal3",
                    is_open=False),
                ], xs=12, sm=12, md=12, lg=6, xl=6),

                dbc.Col([
                    html.Div([
                        html.H2("Les valeurs de paramètres les plus utilisées pour les images optimales", className='text-center text-primary my-4'),
                        dcc.Graph(id="hist-freq-param", style={'height': '800px', 'width': '100%'}),
                    ], className='p-3 mb-4', style={'background-color': 'white', 'border-radius': '6px'}),
                ], xs=12, sm=12, md=12, lg=6, xl=6),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H2("Pourcentage de fonctions considérant ces images comme peu optimales", className='text-center text-primary my-4'),
                        dcc.Graph(id='pie-chart-img-non-optimal', style={'height': '400px', 'width': '100%'}),
                        dbc.Modal([
                            dbc.ModalHeader(dbc.ModalTitle("Image sélectionnée")),
                            dbc.ModalBody(id="modal-content5"),
                        ],
                        id="modal5",
                        is_open=False),
                    ], className='p-3 mb-4', style={'background-color': 'white', 'border-radius': '6px'}),
                ], xs=12, sm=12, md=12, lg=6, xl=6),

                dbc.Col([
                    html.Div([
                        html.H2("Pourcentage de fonctions considérant ces images comme optimales", className='text-center text-primary my-4'),
                        dcc.Graph(id="pie-chart", style={'height': '400px', 'width': '100%'}),
                        dbc.Modal([
                            dbc.ModalHeader(dbc.ModalTitle("Image sélectionnée")),
                            dbc.ModalBody(id="modal-content4"),
                        ],
                        id="modal4",
                        is_open=False),
                    ], className='p-3 mb-4', style={'background-color': 'white', 'border-radius': '6px'}),
                ], xs=12, sm=12, md=12, lg=6, xl=6),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H2("Les KeyWords qui reviennent le plus souvent", className='text-center text-primary my-4'),
                        dcc.Graph(id="nuage-de-mot-keyword", style={'height': '400px', 'width': '100%', 'background-color': '#fff'}),
                    ], className='p-3 mb-4', style={'background-color': 'white', 'border-radius': '6px', 'overflow': 'hidden'}),
                ], xs=12, sm=12, md=12, lg=12, xl=6),
            ]),
        ], xs=12, sm=12, md=12, lg=10, xl=10),
    ]),
], fluid=True)

# Affichage de toutes les photos
photos_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Toutes les images", className='text-center text-primary my-4'),
            html.H4("Sélectionnez un ou plusieurs matériaux", className='text-secondary mb-2', style={'color': '#123C69'}),
            dcc.Dropdown(
                id='material-dropdown-dashboard',
                options=[],
                multi=False,
                className='mb-4'
            ),
            html.Div([
                dcc.Dropdown(
                    id='image-dropdown',
                    options=[],
                    value=[],
                    multi=True,
                    className='mb-4',
                    style={'display': 'none'}
                ),
                html.Div(id='photo-display', style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'flex-wrap': 'wrap'})
             ])
        ], width=12)
    ])
], fluid=True)


# Callback pour mettre à jour les graphiques
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
# utile pour que la barre de navigation soit opérationnelle, elle renvoit aux différents layouts
def display_page(pathname):
    if pathname == '/accueil':
        return home_layout
    elif pathname == '/dashboard':
        return dashboard_layout
    elif pathname == '/paramsDashboard':
        return params_layout
    elif pathname == '/photos':
        return photos_layout
    else:
        return html.Div([])


@app.callback(
    [Output('input-keywords', 'value'),
     Output('input-materiaux', 'value'),
     Output('download-json-button', 'style')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
# récupération des fichiers importés, vérification de l'existence de Json sinon création
def process_uploaded_files(list_of_contents, list_of_names):
    global results_json, unique_images, image_parameters, final_json
    if list_of_contents is not None:
        uploaded_files = [(c, n) for c, n in zip(list_of_contents, list_of_names)]

        # Save directory
        save_dir = "path/to/save/directory"  # Change this to the appropriate directory path

        # Check if JSON file exists in the uploaded files
        json_file = None
        for contents, filename in uploaded_files:
            if filename.endswith('.json'):
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                json_file = json.loads(decoded)
                break

        if json_file:
            final_json = json_file
            json_generated = False
        else:
            final_json = process_images_and_params(uploaded_files, save_dir)
            json_generated = True

        results_json = final_json['results']
        image_parameters = get_image_parameters(final_json)

        # Extraire images uniques pour empecher l'affichage double
        unique_images = {}
        for func_name, func_results in results_json.items():
            for item in func_results:
                if isinstance(item, dict):  # verifier si c'est un dictionnaire
                    if item['path'] not in unique_images:
                        unique_images[item['path']] = item['name']
                else:
                    print(f"Format incorrect : {item}")

        # Récupérer les données 
        existing_keywords = load_keywords()
        existing_materials = load_materials()

        # Collecter nouvelles données
        new_keywords = []
        new_materials = []
        for image_name, params in image_parameters.items():
            for key, value in params.items():
                if key in ['KeyWord1', 'KeyWord2']:
                    new_keywords.append(value)
                elif key == 'Material':
                    if value not in new_materials:
                        new_materials.append(value)

        # Combiner les nouvelles données et les anciennes
        all_keywords = existing_keywords + new_keywords
        all_materials = existing_materials + new_materials

        # Sauvegarder
        save_keywords(all_keywords)
        save_materials(all_materials)

        download_button_style = {'display': 'block'} if json_generated else {'display': 'none'} # apparait s'il n'y avait pas de JSON.
        return json.dumps(all_keywords), json.dumps(all_materials), download_button_style

    return json.dumps([]), json.dumps([]), {'display': 'none'}

@app.callback(
    Output('material-dropdown-dashboard', 'options'),
    [Input('input-materiaux', 'value')]
)
# Label pour les matériaux - sa mise à jour en fonction de l'importation
def update_material_dropdown(materials):
    materials_list = json.loads(materials)
    options = [{'label': mat, 'value': mat} for mat in materials_list]
    return options


@app.callback(
    [Output('function-graph', 'figure'),
     Output('modal-content', 'children'), 
     Output('modal', 'is_open')],
    [Input('function-dropdown-1', 'value'),
     Input('function-dropdown-2', 'value'),
     Input('function-dropdown-3', 'value'),
     Input('material-dropdown-dashboard', 'value'),
     Input('function-graph', 'clickData')],
    [State('modal', 'is_open')]
)
# Le graphique qui regroupe le résultat de plusieurs fonctions , avec possibilité d'afficher les images
def update_global_graph(selected_function_1, selected_function_2, selected_function_3, selected_material, clickData, is_open):
    selected_functions = [selected_function_1, selected_function_2, selected_function_3]
    selected_functions = [f for sublist in selected_functions if sublist is not None for f in sublist]  # Flatten the list and filter out None
    ctx = callback_context

    fig = go.Figure()

    if selected_functions:
        for selected_function in selected_functions:
            func_name = f'F_{selected_function}'
            if func_name not in results_json:
                continue

            func_results = results_json[func_name]
            if selected_material:
                func_results = [item for item in func_results if image_parameters[item['name']].get('Material') == selected_material]

            results = [item['result'] for item in func_results]
            image_names = [item['name'] for item in func_results]

            # Déterminer la catégorie et la couleur
            if 1 <= selected_function <= 9:
                category = 'A'
                color = 'blue'
            elif 10 <= selected_function <= 13:
                category = 'B'
                color = 'green'
            elif 14 <= selected_function <= 19:
                category = 'C'
                color = 'red'
            else:
                category = 'Unknown'
                color = 'black'

            fig.add_trace(go.Scatter(
                x=image_names, y=results, mode='lines+markers', name=f'F{selected_function} (Catégorie {category})', line=dict(color=color)
            ))

            fig.update_layout(
                plot_bgcolor='white'
            )

            fig.update_xaxes(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='lightgrey',
                gridcolor='white'
            )

            fig.update_yaxes(
                mirror=True,
                ticks='outside',
                showline=True,
                linecolor='lightgrey',
                gridcolor='white'
            )

            if results:
                max_idx = np.argmax(results)
                min_idx = np.argmin(results)

                fig.add_vline(x=max_idx, line=dict(color='green', dash='dash'), name=f'Optimal - {image_names[max_idx]}')
                fig.add_vline(x=min_idx, line=dict(color='red', dash='dash'), name=f'Non Optimal - {image_names[min_idx]}')

    # Ajouter des annotations pour les légendes
    fig.add_annotation(
        x=1, y=1.2,
        text="Image Optimale",
        showarrow=False,
        font=dict(color="green"),
        align="left",
        xref="paper",
        yref="paper"
    )
    fig.add_annotation(
        x=1, y=1.1,
        text="Image Non Optimale",
        showarrow=False,
        font=dict(color="red"),
        align="left",
        xref="paper",
        yref="paper"
    )

    fig.update_layout(
        title='Résultats des fonctions sélectionnées',
        xaxis_title='Images',
        yaxis_title='Résultats',
        hovermode='x unified'
    )

    if ctx.triggered and 'function-graph.clickData' in ctx.triggered[0]['prop_id']:
        point_index = clickData['points'][0]['pointIndex']
        selected_image_name = image_names[point_index]
        selected_image_path = [item['path'] for item in func_results if item['name'] == selected_image_name][0]
        selected_image_base64 = image_to_base64(selected_image_path)

        # Collecter les résultats de toutes les fonctions pour l'image sélectionnée
        results_all_functions = {}
        for selected_function in selected_functions:
            func_name = f'F_{selected_function}'
            result = [item['result'] for item in results_json[func_name] if item['name'] == selected_image_name][0]
            results_all_functions[f'F{selected_function}'] = result

        modal_content = html.Div([
            html.Img(src=selected_image_base64, style={'height': '100%', 'width': '100%'}),
            html.P(f"Image sélectionnée : {selected_image_name}"),
            html.P("Résultats de toutes les fonctions :"),
            html.Ul([html.Li(f"{func}: {res}") for func, res in results_all_functions.items()])
        ])
        return fig, modal_content, True

    return fig, "", is_open

@app.callback(
    [Output('individual-function-graph', 'figure'), 
     Output('modal-content2', 'children'), 
     Output('modal2', 'is_open'),
     Output('individual-function-image-optimal', 'children'), 
     Output('individual-function-image-non-optimal', 'children'), 
     Output('optimal-image-name', 'children'), 
     Output('non-optimal-image-name', 'children')],
    [Input('individual-function-dropdown', 'value'),
     Input('material-dropdown-dashboard', 'value'),
     Input('individual-function-graph', 'clickData')],
    [State('modal2', 'is_open')]
)
# fonction pour résultat d'une fonction sélectionnée ainsi que l'affichage des images optimales et non optimales
def update_individual_graph(selected_function, selected_material, clickData, is_open):
    if not selected_function:
        return go.Figure(), "", False, None, None, "", ""
    
    func_name = f'F_{selected_function}'
    if func_name not in results_json:
        return go.Figure(), "", False, None, None, "", ""
    
    func_results = results_json[func_name]
    if selected_material:
        func_results = [item for item in func_results if image_parameters[item['name']].get('Material') == selected_material]

    results = [item['result'] for item in func_results]
    image_names = [item['name'] for item in func_results]
    image_paths = [item['path'] for item in func_results]

    # Créer le graphique interactif
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=image_names, y=results, mode='lines+markers', name=f'F{selected_function}'
    ))
    
    fig.update_layout(
        plot_bgcolor='white'
    )
            
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='lightgrey',
        gridcolor='white'
    )
    
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='lightgrey',
        gridcolor='white'
    )

    max_idx = np.argmax(results)
    min_idx = np.argmin(results)

    fig.add_vline(x=max_idx, line=dict(color='green', dash='dash'), name=f'Optimal - {image_names[max_idx]}')
    fig.add_vline(x=min_idx, line=dict(color='red', dash='dash'), name=f'Non Optimal - {image_names[min_idx]}')

    # Ajouter des annotations pour les légendes
    fig.add_annotation(
        x=1, y=1.2,
        text="Image Optimale",
        showarrow=False,
        font=dict(color="green"),
        align="left",
        xref="paper",
        yref="paper"
    )
    fig.add_annotation(
        x=1, y=1.1,
        text="Image Non Optimale",
        showarrow=False,
        font=dict(color="red"),
        align="left",
        xref="paper",
        yref="paper"
    )

    fig.update_layout(
        title=f'Résultats de la fonction F{selected_function}',
        xaxis_title='Images',
        yaxis_title='Résultats',
        hovermode='x unified'
    )

    optimal_image_path = image_paths[max_idx]
    optimal_image_base64 = image_to_base64(optimal_image_path)
    optimal_img = html.Img(src=optimal_image_base64, style={'height': '100%', 'width': '100%'})

    non_optimal_image_path = image_paths[min_idx]
    non_optimal_image_base64 = image_to_base64(non_optimal_image_path)
    non_optimal_img = html.Img(src=non_optimal_image_base64, style={'height': '100%', 'width': '100%'})

    optimal_image_name = f"Image optimale : {image_names[max_idx]}"
    non_optimal_image_name = f"Image non optimale : {image_names[min_idx]}"

    ctx = callback_context

    if ctx.triggered and 'individual-function-graph.clickData' in ctx.triggered[0]['prop_id']:
        point_index = clickData['points'][0]['pointIndex']
        selected_image_name = image_names[point_index]
        selected_image_path = [item['path'] for item in func_results if item['name'] == selected_image_name][0]
        selected_image_base64 = image_to_base64(selected_image_path)

        # Collecter les résultats de toutes les fonctions pour l'image sélectionnée
        results_all_functions = {}
        for i in range(1, 19):
            func_name = f'F_{i}'
            result = [item['result'] for item in results_json[func_name] if item['name'] == selected_image_name][0]
            results_all_functions[f'F{i}'] = result

        modal_content = html.Div([
            html.Img(src=selected_image_base64, style={'height': '100%', 'width': '100%'}),
            html.P(f"Image sélectionnée : {selected_image_name}"),
            html.P("Résultat de la fonction :"),
            html.Ul([html.Li(f"{func}: {res}") for func, res in results_all_functions.items()])
        ])
        return fig, modal_content, True, optimal_img, non_optimal_img, optimal_image_name, non_optimal_image_name

    return fig, "", is_open, optimal_img, non_optimal_img, optimal_image_name, non_optimal_image_name


@app.callback(
    [Output('param-graph', 'figure'),
     Output('modal-content3', 'children'),
     Output('modal3', 'is_open')],
    [Input('param-function-dropdown-1', 'value'),
     Input('param-function-dropdown-2', 'value'),
     Input('param-function-dropdown-3', 'value'),
     Input('material-dropdown-dashboard', 'value'),
     Input('param-graph', 'clickData')],
    [State('modal3', 'is_open')]
)
# Graphique intéractif sur les paramètres suivant le résultat de la fonction sélectionnée
def update_param_graph(selected_function_1, selected_function_2, selected_function_3, selected_material, clickData, is_open):
    selected_functions = [selected_function_1, selected_function_2, selected_function_3]
    selected_functions = [f for f in selected_functions if f is not None]

    if not selected_functions:
        return go.Figure(), "", False

    optimal_threshold = 0.9  # définition d'une variable pour les récupérations des résultats supérieurs à celui-ci
    optimal_params = []

    all_func_results = []
    for selected_function in selected_functions:
        func_name = f'F_{selected_function}'
        if func_name in results_json:
            func_results = results_json.get(func_name, [])
            if selected_material:
                func_results = [item for item in func_results if image_parameters[item['name']].get('Material') == selected_material]
            all_func_results.extend(func_results)

    results = [round(item['result'], 2) for item in all_func_results]
    image_names = [item['name'] for item in all_func_results]

    # Extraction des paramètres pour les résultats proches de 1
    for idx, result in enumerate(results):
        if result >= optimal_threshold:
            image_name = image_names[idx]
            if image_name in image_parameters:
                params = {k: v for k, v in image_parameters[image_name].items() if k in keys_to_extract}
                params['Results'] = result  
                if params:
                    for k, v in params.items():  
                        if k == "WorkingDistance":
                            v = v.split(" ")[0]
                            v = round(float(v) / 1000, 2)
                            params[k] = f'{v} mm'  
                        elif k == "PixelSize":
                            params[k] = round(float(v), 1)  
                    optimal_params.append(params)

    # Convertion en DataFrame
    df = pd.DataFrame(optimal_params)

    if df.empty:
        return go.Figure(), "", False

    param_freq = df.melt(id_vars=['Results'], var_name='Parameter', value_name='Value')

    # Group by 'Parameter' and 'Value' to get the highest result for each group
    param_freq = param_freq.loc[param_freq.groupby(['Parameter', 'Value'])['Results'].idxmax()]

    # Calculate the frequency of each parameter-value pair
    param_freq['Frequency'] = param_freq.groupby(['Parameter', 'Value'])['Value'].transform('count')

    fig = px.scatter(param_freq, x='Parameter', y='Value', size='Results',
                     color='Results', hover_data=['Frequency'],
                     title=f'Pour les images avec un résultat > 0.9 pour les fonctions sélectionnées',
                     )

    fig.update_layout(
        plot_bgcolor='white',
        xaxis_title='Paramètre',
        yaxis_title='Valeur',
        hovermode='closest',
        height=800  
    )

    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='lightgrey',
        gridcolor='white',
        tickmode='linear',
        tick0=0,
        dtick=1
    )

    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='lightgrey',
        gridcolor='white',
        tickmode='array',
        tickvals=param_freq['Value'].unique() 
    )

    ctx = callback_context

    if ctx.triggered and 'param-graph.clickData' in ctx.triggered[0]['prop_id']:
        point_index = clickData['points'][0]['pointIndex']
        selected_param = param_freq.iloc[point_index]
        parameter = selected_param['Parameter']
        value = selected_param['Value']

        # Filter matching images
        matching_images = []
        for image_name, params in image_parameters.items():
            if parameter in params:
                param_value = params[parameter]
                if parameter == "WorkingDistance":
                    param_value = f'{round(float(param_value.split(" ")[0]) / 1000, 2)} mm'
                elif parameter == "PixelSize":
                    param_value = round(float(param_value), 1)
                
                if str(param_value) == str(value):
                    try:
                        result = next(item['result'] for item in all_func_results if item['name'] == image_name)
                        image_path = next(item['path'] for item in all_func_results if item['name'] == image_name)
                        image_base64 = image_to_base64(image_path)
                        matching_images.append((image_name, result, image_base64))
                    except StopIteration:
                        continue
                    
        modal_content = html.Div([
            html.H4(f'Images avec {parameter} = {value}'),
            html.Ul([html.Li([html.Img(src=img_base64, style={'height': '100px', 'margin-right': '10px', 'margin-bottom': '10px'}), f'{name}: {result}']) for name, result, img_base64 in matching_images], style={'margin-bottom': '10px'})
        ])
        
        return fig, modal_content, True
    
    return fig, "", is_open


@app.callback(
    Output('parallel-graph', 'figure'),
    [Input('param-function-dropdown-1', 'value'),
     Input('param-function-dropdown-2', 'value'),
     Input('param-function-dropdown-3', 'value'),
     Input('material-dropdown-dashboard', 'value')]
)

# Création du tracé des coordonnées parallèles - ensemble de paramètres utilisés selon résultat
def parallel_coordinates(selected_function_1, selected_function_2, selected_function_3, selected_material):
    selected_functions = [selected_function_1, selected_function_2, selected_function_3]
    selected_functions = [f for f in selected_functions if f is not None]

    if not selected_functions:
        return go.Figure()

    all_params = []

    for selected_function in selected_functions:
        func_name = f'F_{selected_function}'
        if func_name in results_json:
            func_results = results_json.get(func_name, [])
            if selected_material:
                func_results = [item for item in func_results if image_parameters[item['name']].get('Material') == selected_material]
            results = [round(item['result'], 2) for item in func_results]
            image_names = [item['name'] for item in func_results]

            for idx, res in enumerate(results):
                image_name = image_names[idx]
                if image_name in image_parameters:
                    params = {k: v for k, v in image_parameters[image_name].items() if k in [
                        'AcceleratingVoltage', 'DecelerationVoltage', 'EmissionCurrent',
                        'WorkingDistance', 'PixelSize', 'SignalName', 'Magnificient',
                        'LensMode', 'ScanSpeed'
                    ]}
                    params['Results'] = res
                    params['Image'] = image_name 
                    if params:
                        for k, v in params.items():
                            if k == "WorkingDistance":
                                try:
                                    v = float(v.split(" ")[0])
                                    params[k] = round(v / 1000, 2)
                                except ValueError:
                                    continue
                            elif k == "PixelSize":
                                try:
                                    params[k] = round(float(v), 1)
                                except ValueError:
                                    continue
                        all_params.append(params)

    df = pd.DataFrame(all_params)

    # Convert specific columns to categorical
    for col in ['SignalName', 'LensMode', 'Magnificient', 'ScanSpeed', 'AcceleratingVoltage', 'DecelerationVoltage', 'EmissionCurrent', 'PixelSize']:
        if col in df.columns:
            df[col] = df[col].astype('category')

    if df.empty:
        return go.Figure()

    # Define dimensions for the parallel coordinates plot
    dimensions = []
    if 'AcceleratingVoltage' in df.columns:
        dimensions.append(dict(label='Accelerating Voltage', values=df['AcceleratingVoltage'].cat.codes, tickvals=list(range(len(df['AcceleratingVoltage'].cat.categories))), ticktext=df['AcceleratingVoltage'].cat.categories))
    if 'DecelerationVoltage' in df.columns:
        dimensions.append(dict(label='Deceleration Voltage', values=df['DecelerationVoltage'].cat.codes, tickvals=list(range(len(df['DecelerationVoltage'].cat.categories))), ticktext=df['DecelerationVoltage'].cat.categories))
    if 'EmissionCurrent' in df.columns:
        dimensions.append(dict(label='Emission Current', values=df['EmissionCurrent'].cat.codes, tickvals=list(range(len(df['EmissionCurrent'].cat.categories))), ticktext=df['EmissionCurrent'].cat.categories))
    if 'WorkingDistance' in df.columns:
        dimensions.append(dict(label='Working Distance en mm', values=df['WorkingDistance']))
    if 'PixelSize' in df.columns:
        dimensions.append(dict(label='Pixel Size', values=df['PixelSize'].cat.codes, tickvals=list(range(len(df['PixelSize'].cat.categories))), ticktext=df['PixelSize'].cat.categories))
    if 'SignalName' in df.columns:
        dimensions.append(dict(label='Signal Name', values=df['SignalName'].cat.codes, tickvals=list(range(len(df['SignalName'].cat.categories))), ticktext=df['SignalName'].cat.categories))
    if 'Magnificient' in df.columns:
        dimensions.append(dict(label='Magnificient', values=df['Magnificient'].cat.codes, tickvals=list(range(len(df['Magnificient'].cat.categories))), ticktext=df['Magnificient'].cat.categories))
    if 'LensMode' in df.columns:
        dimensions.append(dict(label='Lens Mode', values=df['LensMode'].cat.codes, tickvals=list(range(len(df['LensMode'].cat.categories))), ticktext=df['LensMode'].cat.categories))
    if 'ScanSpeed' in df.columns:
        dimensions.append(dict(label='Scan Speed', values=df['ScanSpeed'].cat.codes, tickvals=list(range(len(df['ScanSpeed'].cat.categories))), ticktext=df['ScanSpeed'].cat.categories))
    if 'Results' in df.columns:
        dimensions.append(dict(label='Results', values=df['Results']))

    fig = go.Figure(data=
        go.Parcoords(
            line=dict(color=df['Results'], colorscale=[[0, 'red'], [1, 'green']], showscale=True),
            dimensions=dimensions,
            customdata=df['Image']
        )
    )

    return fig


@app.callback(
    Output("hist-freq-param", 'figure'),
    [Input('param-function-dropdown-1', 'value'),
     Input('param-function-dropdown-2', 'value'),
     Input('param-function-dropdown-3', 'value'),
     Input('material-dropdown-dashboard', 'value')]
)
# Création de l'histogramme pour afficher les valeurs de paramètres les plus utilisés 
def update_hist_param(selected_function_1, selected_function_2, selected_function_3, selected_material):
    selected_functions = [selected_function_1, selected_function_2, selected_function_3]
    selected_functions = [f for f in selected_functions if f is not None]

    if not selected_functions:
        return go.Figure()  

    optimal_threshold = 0.9  
    optimal_params = []

    all_func_results = []
    for selected_function in selected_functions:
        func_name = f'F_{selected_function}'
        if func_name in results_json:
            func_results = results_json.get(func_name, [])
            if selected_material:
                func_results = [item for item in func_results if image_parameters[item['name']].get('Material') == selected_material]
            results = [item['result'] for item in func_results]
            image_names = [item['name'] for item in func_results]

            for idx, result in enumerate(results): # résultat proche de 1
                if result >= optimal_threshold:
                    image_name = image_names[idx]
                    if image_name in image_parameters:
                        params = {k: v for k, v in image_parameters[image_name].items() if k in keys_to_extract}
                        if params:
                            for k, v in params.items():
                                if k == "WorkingDistance":
                                    v = float(v.split(" ")[0])
                                    v = round(v / 1000, 2)
                                    params[k] = f'{v} mm'
                                elif k == "PixelSize":
                                    params[k] = round(float(v), 1)
                            optimal_params.append(params)
                            
    df = pd.DataFrame(optimal_params)

    if df.empty:
        return go.Figure()

    df = df.applymap(str)

    param_freq = df.apply(pd.Series.value_counts).fillna(0)

    param_freq = param_freq.stack().reset_index()
    param_freq.columns = ['Value', 'Parameter', 'Frequency']

    max_freq = param_freq.loc[param_freq.groupby('Parameter')['Frequency'].idxmax()]

    fig = px.histogram(max_freq, x='Value', y='Frequency', color='Parameter',
                       title=f'Pour les images avec un résultat > 0.9 pour les fonctions sélectionnées',
                       barmode='overlay')

    fig.update_layout(
        plot_bgcolor='white',
        xaxis_title='Valeur',
        yaxis_title='Fréquence',
        hovermode='closest'
    )

    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='lightgrey',
        gridcolor='white',
        tickvals=max_freq['Value'].unique() 
    )

    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='lightgrey',
        gridcolor='white'
    )

    return fig

@app.callback(
    Output('photo-display', 'children'),
    [Input('image-dropdown', 'value')]
)
# Affichage des photos 
def display_photos(image_paths):
    if not unique_images:
        return html.Div("Aucune image disponible.", className='text-warning')

    if not image_paths:
        image_paths = list(unique_images.keys())

    image_elements = []
    for image_path in image_paths:
        image_base64 = image_to_base64(image_path)
        image_name = unique_images[image_path]
        image_elements.append(html.Div([
            html.Img(src=image_base64, style={'height': '150px', 'margin': '10px'}),
            html.P(image_name, className='text-muted')
        ], style={'display': 'inline-block', 'text-align': 'center'}))

    return html.Div(image_elements, style={'text-align': 'center'})


@app.callback(
    [Output('pie-chart', 'figure'),
     Output('modal4', 'is_open'),
     Output('modal-content4', 'children')],
    [Input('function_dropdown3', 'value'),
     Input('pie-chart', 'clickData')]
)
# Pie chart pour refarder le pourcentage de fonctions considérant les images comme optimales
def update_pie_chart(function_value, clickData):
    if not results_json:
        return go.Figure(), False, None

    # Déterminer les images optimales pour toutes les fonctions
    optimal_images_counts = {}
    hover_data = {}
    image_parameters = get_image_parameters(final_json)
    
    for i in range(1, 19):
        func_name = f'F_{i}'
        if func_name not in results_json:
            continue
        results = [item['result'] for item in results_json[func_name]]
        image_names = [item['name'] for item in results_json[func_name]]
        
        # Trouver l'image avec la valeur maximale pour cette fonction
        max_result = max(results)
        optimal_image = image_names[results.index(max_result)]
        
        # Mettre à jour les comptes et les données de hover
        if optimal_image in optimal_images_counts:
            optimal_images_counts[optimal_image] += 1
        else:
            optimal_images_counts[optimal_image] = 1
            hover_data[optimal_image] = {}
            
        hover_data[optimal_image][f'F_{i}'] = max_result
    
    # Créer une DataFrame avec les résultats et les noms des images
    df = pd.DataFrame({
        'Image': list(optimal_images_counts.keys()),
        'Count': list(optimal_images_counts.values())
    })

    # Ajouter les résultats des fonctions en hover_data
    df['hover'] = [hover_data[image] for image in df['Image']]
    
    # Créer le graphique circulaire
    fig = px.pie(df, values='Count', names='Image',  
                    hover_data={'Image': True, 'hover': False})

    # Mettre à jour les traces pour afficher les données de hover
    fig.update_traces(hovertemplate='<b>%{label}</b><br>Count: %{value}<br>')
    
    # Vérifier si une section a été cliquée pour afficher le modal
    if clickData:
        selected_image = clickData['points'][0]['label']
        selected_image_path = next(item['path'] for item in results_json['F_1'] if item['name'] == selected_image)
        selected_image_base64 = image_to_base64(selected_image_path)

        # Récupérer les paramètres de l'image sélectionnée
        image_params = image_parameters.get(selected_image, {})

        modal_content = html.Div([
            html.Img(src=selected_image_base64, style={'height': '100%', 'width': '100%'}),
            html.P(f"Image sélectionnée : {selected_image}"),
            html.P("Résultats des fonctions :"),
            html.Ul([html.Li(f"{key}: {value}") for key, value in hover_data[selected_image].items()]),
            html.P("Paramètres de l'image :"),
            html.Ul([html.Li(f"{param}: {value}") for param, value in image_params.items()])
        ])

        return fig, True, modal_content

    return fig, False, None


@app.callback(
    [Output('pie-chart-img-non-optimal', 'figure'),
     Output('modal5', 'is_open'),
     Output('modal-content5', 'children')],
    [Input('function_dropdown4', 'value'),
     Input('pie-chart-img-non-optimal', 'clickData')]
)
# Pie chart pour refarder le pourcentage de fonctions considérant les images comme non optimales
def update_pie_chart_non_optimal(function_value, clickData):
    if not results_json:
        return go.Figure(), False, None

    # Déterminer les images optimales pour toutes les fonctions
    non_optimal_images_counts = {}
    hover_data = {}
    image_parameters = get_image_parameters(final_json)
    
    for i in range(1, 19):
        func_name = f'F_{i}'
        if func_name not in results_json:
            continue
        results = [item['result'] for item in results_json[func_name]]
        image_names = [item['name'] for item in results_json[func_name]]
        
        # Trouver l'image avec la valeur maximale pour cette fonction
        min_result = min(results)
        non_optimal_image = image_names[results.index(min_result)]
        
        # Mettre à jour les comptes et les données de hover
        if non_optimal_image in non_optimal_images_counts:
            non_optimal_images_counts[non_optimal_image] += 1
        else:
            non_optimal_images_counts[non_optimal_image] = 1
            hover_data[non_optimal_image] = {}
            
        hover_data[non_optimal_image][f'F_{i}'] = min_result
    
    # Créer une DataFrame avec les résultats et les noms des images
    df = pd.DataFrame({
        'Image': list(non_optimal_images_counts.keys()),
        'Count': list(non_optimal_images_counts.values())
    })

    # Ajouter les résultats des fonctions en hover_data
    df['hover'] = [hover_data[image] for image in df['Image']]
    
    # Créer le graphique circulaire
    fig = px.pie(df, values='Count', names='Image',  
                    hover_data={'Image': True, 'hover': False})

    # Mettre à jour les traces pour afficher les données de hover
    fig.update_traces(hovertemplate='<b>%{label}</b><br>Count: %{value}<br>')
    
    # Vérifier si une section a été cliquée pour afficher le modal
    if clickData:
        selected_image = clickData['points'][0]['label']
        selected_image_path = next(item['path'] for item in results_json['F_1'] if item['name'] == selected_image)
        selected_image_base64 = image_to_base64(selected_image_path)

        # Récupérer les paramètres de l'image sélectionnée
        image_params = image_parameters.get(selected_image, {})

        modal_content = html.Div([
            html.Img(src=selected_image_base64, style={'height': '100%', 'width': '100%'}),
            html.P(f"Image sélectionnée : {selected_image}"),
            html.P("Résultats des fonctions :"),
            html.Ul([html.Li(f"{key}: {value}") for key, value in hover_data[selected_image].items()]),
            html.P("Paramètres de l'image :"),
            html.Ul([html.Li(f"{param}: {value}") for param, value in image_params.items()])
        ])

        return fig, True, modal_content

    return fig, False, None


@app.callback(
    Output("nuage-de-mot-keyword", 'figure'),
    [Input('input-keywords', 'value')]
)
# Affichage des keywords
def display_kW(keywords):
    if not keywords:
        return go.Figure() 

    keywords = json.loads(keywords)

    if not keywords:
        return go.Figure()  

    word_counts = Counter(keywords)

    wordcloud = WordCloud(background_color='white', max_font_size=300).generate_from_frequencies(word_counts)

    # Conversion du nuage de mot en une image pour l'affichage. 
    img = BytesIO()
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)

    encoded_image = base64.b64encode(img.read()).decode()

    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=f'data:image/png;base64,{encoded_image}',
            xref="paper", yref="paper",
            x=0, y=1,
            sizex=1, sizey=1,
            xanchor="left", yanchor="top"
        )
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig

@app.callback(
    Output('download-json', 'data'),
    [Input('download-json-button', 'n_clicks')],
    [State('input-keywords', 'value')]
)
# Téléchargement du Json
def download_json(n_clicks, keywords):
    if n_clicks is None:
        return None

    keywords = json.loads(keywords)
    final_json['keywords'] = keywords

    return dict(content=json.dumps(final_json, indent=4), filename='Results_Functions.json')

if __name__ == '__main__':
    app.run_server(debug=True)
