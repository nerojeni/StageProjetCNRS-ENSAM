import json
import pandas as pd
import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.dash_table import DataTable
import plotly.express as px
import plotly.graph_objs as go


# Chemin absolu du dossier contenant vos fichiers JSON
data_folder = "C:/Users/neroj/OneDrive/Bureau/BUT INF FA2/StageProjet/RecupInfoImage/Data/" # Changer le chemin d'accès

# Liste des noms de fichiers
file_names = [
    os.path.join(data_folder, "2023_06_19.json"),
    os.path.join(data_folder, "Camille_2021.json"),
    os.path.join(data_folder, "Camille_2021bis.json"),
    os.path.join(data_folder, "Amir_Mohammed.json"),
    os.path.join(data_folder, "Antoine_2018_AnCS11C.json"),
    os.path.join(data_folder, "Burak_2017_ENRF.json"),
    os.path.join(data_folder, "Burak_2017_ERG.json"),
    os.path.join(data_folder, "Giov_2009.json"),
    os.path.join(data_folder, "Hiam_2022.json"),
    os.path.join(data_folder, "Imen_2021.json"),
    os.path.join(data_folder, "Imen_2021_Nylon.json"),
    os.path.join(data_folder, "Martina_2024.json"),
    os.path.join(data_folder, "Zehoua_SLM_Inco625_2023.json"),
    os.path.join(data_folder, "Zehoua_Enzo_New.json")
]


# Charger les données depuis les fichiers JSON et créer les DataFrames
dfs = []
for file_name in file_names:
    with open(file_name, "r") as json_file:
        data = json.load(json_file)
        df = pd.DataFrame(data)
        dfs.append(df)

# Concaténer les DataFrames
df = pd.concat(dfs, ignore_index=True)

# Convertir certaines colonnes en nombres
numeric_columns = ['AcceleratingVoltage', 'DecelerationVoltage', 'EmissionCurrent', 'WorkingDistance', 'PixelSize']
for col in numeric_columns:
    if col != 'SignalName':  # Sauf pour la colonne "SignalName"
        df[col] = df[col].str.extract(r'(\d+)').astype(float)
df['WorkingDistance'] /= 1000  # Convertir µm en mm

# Options de liste déroulante pour le mode de lentille et la vitesse de numérisation
options_mode_lentille = [{'label': mode, 'value': mode} for mode in df['LensMode'].unique()]
options_mode_lentille.append({'label': 'Tous', 'value': 'Tous'})

options_vitesse_num = [{'label': speed, 'value': speed} for speed in df['ScanSpeed'].unique()]
options_vitesse_num.append({'label': 'Tous', 'value': 'Tous'})

colonnes = ['AcceleratingVoltage', 'DecelerationVoltage', 'EmissionCurrent', 'WorkingDistance', 'PixelSize']
options_axes = [{'label': col, 'value': col} for col in colonnes]

# Ajout des graphiques pour les moyennes, minima et maxima pour chaque colonne
import plotly.graph_objs as go

def create_stats_scatter_chart(df, column_name, color_mean='blue', color_min='red', color_max='green'):
    mean_value = df[column_name].mean()
    min_value = df[column_name].min()
    max_value = df[column_name].max()

    fig = go.Figure(data=[
        go.Scatter(
            x=['Moyenne', 'Minimum', 'Maximum'],
            y=[mean_value, min_value, max_value],
            mode='markers',
            marker=dict(color=[color_mean, color_min, color_max])
        )
    ])

    fig.update_layout(title=f"Statistiques pour {column_name}",
                      xaxis_title="Statistique",
                      yaxis_title="Valeur",xaxis_tickformat=".3f",yaxis_tickformat='.3f')


    return fig



# Initialiser l'application Dash
app = dash.Dash(__name__,
                external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'])

# Layout du tableau de bord
app.layout = html.Div([
    html.H1("Tableau de Bord", className="text-center text-primary mb-4"),
    html.Div([
        html.Label("Choisir le Mode de Lentille", className="dropdown-label"),
        dcc.Dropdown(
            id='mode-lentille-dropdown',
            options=options_mode_lentille,
            value='Tous',
            multi=True,
            clearable=True,
            placeholder="Sélectionnez un ou plusieurs modes de lentille"
        ),
        html.Label("Choisir la Vitesse de Numérisation", className="dropdown-label"),
        dcc.Dropdown(
            id='vitesse-num-dropdown',
            options=options_vitesse_num,
            value='Tous',
            multi=True,
            clearable=True,
            placeholder="Sélectionnez une ou plusieurs vitesses de numérisation"
        ),
    ], className="dropdown-container"),
    # Section du graphique personnalisé
    html.Div([
        html.Label("Choisir l'Axe X pour le Graphique Personnalisé", className="dropdown-label"),
        dcc.Dropdown(id='axe-x-graphique-personnalise', options=options_axes, value='AcceleratingVoltage', clearable=False,
                     placeholder="Sélectionnez l'axe X"),
        html.Label("Choisir l'Axe Y pour le Graphique Personnalisé", className="dropdown-label"),
        dcc.Dropdown(id='axe-y-graphique-personnalise', options=options_axes, value='EmissionCurrent', clearable=False,
                     placeholder="Sélectionnez l'axe Y"),
    ], className="dropdown-container"),
    dcc.Graph(id='graphique-personnalise'),

    # Section de l'histogramme
    html.Div([
        html.Label("Choisir le Paramètre", className="dropdown-label"),
        dcc.Dropdown(id='parametre-x-histogramme', options=options_axes, value='EmissionCurrent', clearable=False,
                     placeholder="Sélectionnez le paramètre")
    ], className="dropdown-container"),
    dcc.Graph(id='histogramme-parametre'),


    # Autres graphiques
    html.Div([
        dcc.Graph(id='graphique-tension-acceleration'),
        dcc.Graph(id='graphique-tension-deceleration'),
    ], className="row justify-content-center mt-4"),
    html.Div([
        dcc.Graph(id='graphique-courant-emission'),
        dcc.Graph(id='graphique-distance-travail'),
    ], className="row justify-content-center mt-4"),
    html.Div([
        dcc.Graph(id='nuage-de-points'),
        dcc.Graph(id='histogramme-mode-lentille'),
    ], className="row justify-content-center mt-4"),
    html.Div([
        dcc.Graph(id="graphique-taille-donnees")
    ], className="row justify-content-center mt-4"),

    # Widgets pour afficher les moyennes, minima et maxima
    html.Div([
        html.Div([
            html.Div(id='moyenne-tension-acceleration-container', className="moyenne-container"),
            html.Div(id='min-tension-acceleration-container', className="moyenne-container"),
            html.Div(id='max-tension-acceleration-container', className="moyenne-container"),
        ], className="col"),
        html.Div([
            html.Div(id='moyenne-tension-deceleration-container', className="moyenne-container"),
            html.Div(id='min-tension-deceleration-container', className="moyenne-container"),
            html.Div(id='max-tension-deceleration-container', className="moyenne-container"),
        ], className="col"),
        html.Div([
            html.Div(id='moyenne-courant-emission-container', className="moyenne-container"),
            html.Div(id='min-courant-emission-container', className="moyenne-container"),
            html.Div(id='max-courant-emission-container', className="moyenne-container"),
        ], className="col"),
        html.Div([
            html.Div(id='moyenne-distance-travail-container', className="moyenne-container"),
            html.Div(id='min-distance-travail-container', className="moyenne-container"),
            html.Div(id='max-distance-travail-container', className="moyenne-container"),
        ], className="col"),
    ], className="row justify-content-center mt-4"),

    # Section des graphiques de statistiques
    html.Div([
        dcc.Graph(id='graphique-tension-acceleration-stats'),
        dcc.Graph(id='graphique-tension-deceleration-stats'),
        dcc.Graph(id='graphique-courant-emission-stats'),
        dcc.Graph(id='graphique-distance-travail-stats'),
    ], className="row justify-content-center mt-4"),
])



@app.callback(
    [Output('graphique-tension-acceleration', 'figure'),
     Output('graphique-tension-deceleration', 'figure'),
     Output('graphique-courant-emission', 'figure'),
     Output('graphique-distance-travail', 'figure'),
     Output('nuage-de-points', 'figure'),
     Output('histogramme-mode-lentille', 'figure'),
     Output('graphique-taille-donnees', 'figure'),
     Output('graphique-personnalise', 'figure'),
     Output('histogramme-parametre', 'figure'),
     Output('moyenne-tension-acceleration-container', 'children'),
     Output('min-tension-acceleration-container', 'children'),
     Output('max-tension-acceleration-container', 'children'),
     Output('moyenne-tension-deceleration-container', 'children'),
     Output('min-tension-deceleration-container', 'children'),
     Output('max-tension-deceleration-container', 'children'),
     Output('moyenne-courant-emission-container', 'children'),
     Output('min-courant-emission-container', 'children'),
     Output('max-courant-emission-container', 'children'),
     Output('moyenne-distance-travail-container', 'children'),
     Output('min-distance-travail-container', 'children'),
     Output('max-distance-travail-container', 'children'),
     Output('graphique-tension-acceleration-stats', 'figure'),
     Output('graphique-tension-deceleration-stats', 'figure'),
     Output('graphique-courant-emission-stats', 'figure'),
     Output('graphique-distance-travail-stats', 'figure')],
    [Input('mode-lentille-dropdown', 'value'),
     Input('vitesse-num-dropdown', 'value'),
     Input('axe-x-graphique-personnalise', 'value'),
     Input('axe-y-graphique-personnalise', 'value'),
     Input('parametre-x-histogramme', 'value'),]
)

def mettre_a_jour_graphiques(mode_lentille_selectionne, vitesse_num_selectionnee, axe_x, axe_y, parametre_x):
    df_filtre = df
    if mode_lentille_selectionne and 'Tous' not in mode_lentille_selectionne:
        df_filtre = df_filtre[df_filtre['LensMode'].isin(mode_lentille_selectionne)]
    if vitesse_num_selectionnee and 'Tous' not in vitesse_num_selectionnee:
        df_filtre = df_filtre[df_filtre['ScanSpeed'].isin(vitesse_num_selectionnee)]

    # Formatage pour afficher les valeurs complètes sans 'k' dans la Tension d'Accélération
    fig_tension_acceleration = px.scatter(df_filtre, x='WorkingDistance', y='AcceleratingVoltage', color='LensMode',render_mode='webgl',
                                          title='AcceleratingVoltage(V) par WorkingDistance(mm)')
    fig_tension_acceleration.update_layout(yaxis_tickformat='.2f')

    # Formatage pour afficher les valeurs complètes sans 'k' dans la Tension de Décélération
    fig_tension_deceleration = px.scatter(df_filtre, x='ScanSpeed', y='DecelerationVoltage', color='LensMode',
                                          title='DeceleratingVoltage(V) par ScanSpeed')
    fig_tension_deceleration.update_layout(yaxis_tickformat='.2f')  # Afficher les valeurs complètes

    # Graphique pour le Courant d'Émission avec les valeurs complètes
    fig_courant_emission = px.scatter(df_filtre, x='ScanSpeed', y='EmissionCurrent', color='LensMode',
                                      title='EmissionCurrent(nA) par ScanSpeed')
    fig_courant_emission.update_layout(yaxis_tickformat='.2f')  # Afficher les valeurs complètes

    # Graphique pour la Distance de Travail, convertie en mm
    fig_distance_travail = px.scatter(df_filtre, x='ScanSpeed', y='WorkingDistance', color='LensMode',
                                      title='WorkingDistance(mm) par ScanSpeed')
    fig_distance_travail.update_layout(yaxis_tickformat='.3f')  # Afficher les valeurs complètes en mm

    fig_nuage_de_points = px.scatter(df_filtre, x='AcceleratingVoltage', y='EmissionCurrent', title='AccelerationVoltage(V) par EmissionCurrent(nA) et WorkingDistance(mm)',
                                     color='WorkingDistance', labels={'AcceleratingVoltage': 'Tension d\'Accélération (Volts)',
                                                                      'EmissionCurrent': 'Courant d\'Émission (nA)',
                                                                      'WorkingDistance': 'Distance de Travail (mm)'})

    fig_nuage_de_points.update_traces(marker=dict(size=8))
    fig_nuage_de_points.update_layout(xaxis_tickformat='.3f', yaxis_tickformat='.3f')

    # Graphique pour les axes personnalisés
    fig_personnalise = px.scatter(df_filtre, x=axe_x, y=axe_y, color='LensMode', title=f'{axe_y} par {axe_x}')
    fig_personnalise.update_layout(yaxis_tickformat=".3f", xaxis_tickformat=".3f")

    # Compter le nombre d'occurrences de chaque valeur unique du paramètre
    compte_parametre = df_filtre[parametre_x].value_counts().reset_index()
    compte_parametre.columns = [parametre_x, 'Compte']

    # Créer un histogramme à partir des données comptées
    fig_histogramme_parametre = px.bar(compte_parametre, x=parametre_x, y='Compte', title=f"Nombre d'images par {parametre_x}",
                                       color="Compte", labels={'Compte': 'Nombre d\'images'})

    # Mettre à jour le format de l'axe des abscisses
    fig_histogramme_parametre.update_layout(xaxis_tickformat=".0f")

    # Histogramme du nombre d'images par mode de lentille
    fig_histogramme_mode_lentille = px.histogram(df_filtre, x='LensMode', title='Nombre d\'images par Mode de Lentille')

    # Graphique pour la Taille des Données en utilisant les tailles réelles
    fig_taille_donnees_mode_lentille = px.scatter(df_filtre, x="DataSize", y="EmissionCurrent", color="LensMode",
                                                  title="Courant d\'Émission par Taille des Données")
    fig_taille_donnees_mode_lentille.update_layout(yaxis_tickformat=".3f")

    # Calcul des moyennes, minima et maxima
    moyenne_tension_acceleration = df_filtre['AcceleratingVoltage'].mean()
    min_tension_acceleration = df_filtre['AcceleratingVoltage'].min()
    max_tension_acceleration = df_filtre['AcceleratingVoltage'].max()

    moyenne_tension_deceleration = df_filtre['DecelerationVoltage'].mean()
    min_tension_deceleration = df_filtre['DecelerationVoltage'].min()
    max_tension_deceleration = df_filtre['DecelerationVoltage'].max()

    moyenne_courant_emission = df_filtre['EmissionCurrent'].mean()
    min_courant_emission = df_filtre['EmissionCurrent'].min()
    max_courant_emission = df_filtre['EmissionCurrent'].max()

    moyenne_distance_travail = df_filtre['WorkingDistance'].mean()
    min_distance_travail = df_filtre['WorkingDistance'].min()
    max_distance_travail = df_filtre['WorkingDistance'].max()

    # Formatage des résultats pour affichage
    moyenne_tension_acceleration_str = f"Moyenne AcceleratingVoltage(V): {moyenne_tension_acceleration:.2f}"
    min_tension_acceleration_str = f"Minimum AcceleratingVoltage(V): {min_tension_acceleration:.2f}"
    max_tension_acceleration_str = f"Maximum AcceleratingVoltage(V): {max_tension_acceleration:.2f}"

    moyenne_tension_deceleration_str = f"Moyenne DeceleratingVoltage(V): {moyenne_tension_deceleration:.2f}"
    min_tension_deceleration_str = f"Minimum DeceleratingVoltage(V): {min_tension_deceleration:.2f}"
    max_tension_deceleration_str = f"Maximum DeceleratingVoltage(V): {max_tension_deceleration:.2f}"

    moyenne_courant_emission_str = f"Moyenne EmissionCurrent(nA): {moyenne_courant_emission:.2f}"
    min_courant_emission_str = f"Minimum EmissionCurrent(nA): {min_courant_emission:.2f}"
    max_courant_emission_str = f"Maximum EmissionCurrent(nA): {max_courant_emission:.2f}"

    moyenne_distance_travail_str = f"Moyenne WorkingDistance(mm): {moyenne_distance_travail:.2f}"
    min_distance_travail_str = f"Minimum WorkingDistance(mm): {min_distance_travail:.2f}"
    max_distance_travail_str = f"Maximum WorkingDistance(mm): {max_distance_travail:.2f}"

    fig_tension_acceleration_stats = create_stats_scatter_chart(df_filtre, 'AcceleratingVoltage', color_mean='violet', color_min='blue', color_max='red')
    fig_tension_deceleration_stats = create_stats_scatter_chart(df_filtre, 'DecelerationVoltage', color_mean='violet', color_min='blue', color_max='red')
    fig_courant_emission_stats = create_stats_scatter_chart(df_filtre, 'EmissionCurrent', color_mean='violet', color_min='blue', color_max='red')
    fig_distance_travail_stats = create_stats_scatter_chart(df_filtre, 'WorkingDistance', color_mean='violet', color_min='blue', color_max='red')



    return [fig_tension_acceleration,
            fig_tension_deceleration,
            fig_courant_emission,
            fig_distance_travail,
            fig_nuage_de_points,
            fig_histogramme_mode_lentille,
            fig_taille_donnees_mode_lentille,
            fig_personnalise,
            fig_histogramme_parametre,
            moyenne_tension_acceleration_str,
            min_tension_acceleration_str,
            max_tension_acceleration_str,
            moyenne_tension_deceleration_str,
            min_tension_deceleration_str,
            max_tension_deceleration_str,
            moyenne_courant_emission_str,
            min_courant_emission_str,
            max_courant_emission_str,
            moyenne_distance_travail_str,
            min_distance_travail_str,
            max_distance_travail_str,
            fig_tension_acceleration_stats,
            fig_tension_deceleration_stats,
            fig_courant_emission_stats,
            fig_distance_travail_stats]


# Exécuter l'application
if __name__ == '__main__':
    app.run_server(debug=True)
