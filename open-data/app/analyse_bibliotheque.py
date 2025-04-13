import os
from attr import s
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
from haversine import haversine  
import requests
from bs4 import BeautifulSoup
import json
import pulp
import community as community_louvain
import community.community_louvain as community_louvain
import matplotlib.pyplot as plt
import numpy as np
import pulp
import streamlit
import streamlit as st
import folium
from streamlit_folium import folium_static

from pyvis.network import Network
from pulp import PULP_CBC_CMD, LpStatus
import os
import pandas as pd
import networkx as nx
import folium
import pulp
import streamlit as st
import plotly.express as px
from haversine import haversine
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from pyvis.network import Network

import networkx as nx
import matplotlib.pyplot as plt
import pulp
from community import community_louvain
import streamlit as st

# 📁 Importation du fichier contenant les fonctions de graphe et chemin
import graphe_ouvrages

# ===============================
# 🔲 MENU LATERAL DE NAVIGATION
# ===============================

choix = st.sidebar.radio(
    "🔎 Choisir une visualisation :",
    ["Réseau géographique", "Réseau par ouvrages", "Chemin optimal"]
)

# ===============================
# 🌍 GRAPHE CLASSIQUE PAR DISTANCE
# ===============================
if choix == "Réseau géographique":

    st.write("🌍 Réseaux Interactifs Des Bibliothèques En")

# ===============================
# 📚 GRAPHE PAR OUVRAGES DIFFÉRENTS
# ===============================
elif choix == "Réseau par ouvrages":
    graphe_ouvrages.afficher_graphe()

# ===============================
# 🧭 CHEMIN OPTIMAL POUR LIRE TOUS LES LIVRES
# ===============================
elif choix == "Chemin optimal":
    graphe_ouvrages.afficher_chemin_optimal()


# 📌 Étape 1 : Chargement et Nettoyage des Données

fichier_csv = "data/bibliotheques.csv"
df = pd.read_csv(fichier_csv, encoding="utf-8", delimiter=";")

df = df.dropna(subset=["Latitude", "Longitude", "Nom de l'établissement"])
df["Latitude"] = df["Latitude"].astype(float)
df["Longitude"] = df["Longitude"].astype(float)

# Réduction de la taille pour éviter les lenteurs
df = df.sample(n=200, random_state=42)  # au lieu de tout charger

print(f"✅ Nombre total de bibliothèques sélectionnées : {len(df)}")

print("✅ Étape 1 terminée : Données chargées et nettoyées")
# =====================================================
# 📌 ÉTAPE 2 DU PROJET :  Manipulation de graphes avec networkx
# =====================================================

# 📌 Étape 2.1 : Création du Graphe avec NetworkX

G = nx.Graph()

for _, row in df.iterrows():
    G.add_node(row["Nom de l'établissement"], 
               ville=row["Ville"], 
               region=row["Région"], 
               latitude=row["Latitude"], 
               longitude=row["Longitude"], 
               surface=row["Surface"])

# Seuil de distance et limitation des arêtes
seuil_distance_km = 100
max_edges = 5000
edge_count = 0

bibliotheques = list(G.nodes(data=True))

for i in range(len(bibliotheques)):
    for j in range(i + 1, len(bibliotheques)):
        if edge_count >= max_edges:
            break  

        biblio1 = bibliotheques[i]
        biblio2 = bibliotheques[j]

        coord1 = (biblio1[1]["latitude"], biblio1[1]["longitude"])
        coord2 = (biblio2[1]["latitude"], biblio2[1]["longitude"])
        distance = haversine(coord1, coord2)

        if distance < seuil_distance_km:
            G.add_edge(biblio1[0], biblio2[0], weight=distance)
            edge_count += 1

print(f"✅ Nombre total d'arêtes après limitation : {G.number_of_edges()}")
print("✅ Étape 2 terminée : Graphe créé")

# 📌 Étape 2.2 : Visualisation du Graphe

print("➡ Début de l'étape 3 : Visualisation du graphe")

# Réduire l'affichage à un sous-graphe de 200 bibliothèques max
if G.number_of_nodes() > 200:
    subgraph_nodes = list(G.nodes)[:200]
    G_subgraph = G.subgraph(subgraph_nodes)
    print("⚠ Affichage d'un sous-graphe de 200 bibliothèques au lieu de tout le graphe")
else:
    G_subgraph = G

# Affichage du graphe recentré
pos = nx.spring_layout(G_subgraph, k=0.3)

plt.figure(figsize=(12, 8))
nx.draw_networkx_edges(G_subgraph, pos, alpha=0.3, edge_color="gray")
nx.draw_networkx_nodes(G_subgraph, pos, node_size=50, node_color="blue")

plt.title("Graphe des bibliothèques")
plt.show(block=False)
plt.pause(3)
plt.close()

print("✅ Étape 3 terminée : Graphe affiché")

# 📌 Étape 2.3 : Affichage des Sommets et des Arêtes

print("\n📌 Liste des bibliothèques dans le graphe (10 premières affichées) :")
bibliotheques_list = list(G.nodes())[:10]
print(bibliotheques_list)
if len(G.nodes()) > 10:
    print("... (", len(G.nodes()) - 10, "bibliothèques supplémentaires non affichées)")

print("\n📌 Liste des connexions (arêtes) entre bibliothèques (10 premières affichées) :")
edges_list = list(G.edges())[:10]
print(edges_list)
if len(G.edges()) > 10:
    print("... (", len(G.edges()) - 10, "connexions supplémentaires non affichées)")

print(f"\n✅ Le graphe contient {G.number_of_nodes()} bibliothèques.")
print(f"✅ Il y a {G.number_of_edges()} connexions entre elles.")

# 📌 Étape 2.4: Analyse du Graphe (Bibliothèques Centrales)

degree_centrality = nx.degree_centrality(G)
top_bibliotheques = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

print("\n📌 Top 10 des bibliothèques les plus connectées :")
for bibliotheque, degre in top_bibliotheques:
    print(f"{bibliotheque} - Score de centralité : {degre:.3f}")

print("✅ Étape 2.4 terminée : Analyse des bibliothèques centrales.")

# 📌 Étape 2.5 : Génération de Graphes de Familles Connues

n = 10  
p = 0.3  

G_erdos = nx.erdos_renyi_graph(n, p)
G_barabasi = nx.barabasi_albert_graph(n=10, m=2)
G_bipartite = nx.complete_bipartite_graph(5, 5)
G_cycle = nx.cycle_graph(6)

print("\n📌 Exemples de graphes générés :")
print("G_erdos:", G_erdos.number_of_nodes(), "nœuds,", G_erdos.number_of_edges(), "arêtes.")
print("G_barabasi:", G_barabasi.number_of_nodes(), "nœuds,", G_barabasi.number_of_edges(), "arêtes.")
print("G_bipartite:", G_bipartite.number_of_nodes(), "nœuds,", G_bipartite.number_of_edges(), "arêtes.")
print("G_cycle:", G_cycle.number_of_nodes(), "nœuds,", G_cycle.number_of_edges(), "arêtes.")

print("✅ Étape 2.5 terminée : Génération de graphes terminée.")

# 📌 Étape 2.6 : Chargement et Visualisation d’un Graphe depuis un Fichier

nom_fichier = "test.txt"

if not os.path.exists(nom_fichier):
    print(f"⚠ Erreur : Le fichier '{nom_fichier}' n'existe pas !")
    print("👉 Vérifie le nom du fichier ou crée un fichier test.txt dans le bon dossier.")
else:
    def charger_graphe(source):
        G = nx.Graph()
        with open(source, 'r', encoding='utf-8') as fichier:
            fichier.readline()
            contenu = fichier.read()
            liste_couples = contenu.split("\n")
            for couple in liste_couples:
                if couple.strip() == "":
                    continue
                couple_etudiants = couple.split(";")
                if len(couple_etudiants) == 2:
                    G.add_edge(couple_etudiants[0], couple_etudiants[1])
        return G

    G_fichier = charger_graphe(nom_fichier)

    plt.figure(figsize=(10, 6))
    nx.draw_shell(G_fichier, with_labels=False, node_size=50, node_color="blue", edge_color="gray")
    plt.savefig('plotgraph.png', dpi=300)
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    print("✅ Étape 2.6 terminée : Chargement et visualisation du graphe depuis un fichier réussis.")
# =====================================================
# 📌 ÉTAPE 4 DU PROJET : Récupération d’informations (Scraping, APIs)
# =====================================================

# 📌 **4.1 - Scraping de données Web**
print("➡ Début de l'étape 4.1 : Scraping des données Web")

url = "https://example.com/"  # Remplace par un site réel
response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'lxml')
    all_links = soup.find_all("a")  # Récupérer tous les liens
    links_data = [{"Texte": link.text, "URL": link.get('href')} for link in all_links[:5]]  # Limité à 5 liens
    print("✅ Scraping terminé : 5 premiers liens récupérés")
else:
    print(f"⚠ Échec du scraping, code d'erreur : {response.status_code}")

# 📌 **4.2 - Requêtes API**
print("➡ Début de l'étape 4.2 : Récupération des données via API")

api_url = "https://jsonplaceholder.typicode.com/posts"
response = requests.get(api_url)

if response.status_code == 200:
    articles_json = response.json()
    with open("data/articles.json", "w", encoding='utf-8') as f:
        json.dump(articles_json, f, ensure_ascii=False, indent=2)
    print("✅ API accessible, données sauvegardées dans articles.json")
else:
    print(f"⚠ Erreur API : {response.status_code}")


# 📌 **4.3 - Recherche sur Yahoo**
print("➡ Début de l'étape 4.3 : Recherche Yahoo")

def requete_yahoo(chaine):
    headers = {
        'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                      '(KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582'
    }
    params = {'p': chaine}
    
    response = requests.get('https://fr.search.yahoo.com/search', headers=headers, params=params)
    
    if "Nous n'avons trouvé aucun résultat" in response.text:
        print(f"🔍 Recherche Yahoo '{chaine}': Aucun résultat trouvé")
        return 0
    else:
        print(f"🔍 Recherche Yahoo '{chaine}': Des résultats ont été trouvés")
        return 1  # Remplacer par un vrai comptage si besoin

requete_yahoo("bibliothèques")

print("✅ Étape 4 terminée : Scraping et récupération d’informations effectués")


# =====================================================
# 📌 ÉTAPE 5 DU PROJET : Formats de données (CSV, JSON, XLSX)
# =====================================================

# 📌 **5.1 - Sauvegarde et chargement au format CSV**
print("➡ Début de l'étape 5.1 : Sauvegarde des données en CSV")

# Convertir les nœuds en DataFrame
nodes_data = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index").reset_index()
nodes_data.rename(columns={"index": "Nom de l'établissement"}, inplace=True)

# Sauvegarde en CSV
nodes_data.to_csv("bibliotheques_graph.csv", index=False)
print("✅ Données du graphe sauvegardées dans bibliotheques_graph.csv")

# 📌 **5.2 - Sauvegarde et chargement au format JSON**
print("➡ Début de l'étape 5.2 : Sauvegarde des données en JSON")

# Sauvegarde du graphe en JSON
with open("data/bibliotheques_graph.json", "w", encoding="utf-8") as f:
    json.dump(nx.node_link_data(G), f, ensure_ascii=False, indent=2)

print("✅ Données du graphe sauvegardées dans bibliotheques_graph.json")

# 📌 **5.3 - Sauvegarde et chargement au format Excel (XLSX)**
print("➡ Début de l'étape 5.3 : Sauvegarde des données en XLSX")

# Sauvegarde en Excel
nodes_data.to_excel("bibliotheques_graph.xlsx", sheet_name="Bibliothèques", index=False)

print("✅ Données du graphe sauvegardées dans bibliotheques_graph.xlsx")

# 📌 **5.4 - Export du graphe pour Gephi (GEXF)**
print("➡ Début de l'étape 5.4 : Export du graphe pour Gephi")

nx.write_gexf(G, "bibliotheques_graph.gexf")
print("✅ Export du graphe terminé : bibliotheques_graph.gexf")




import networkx as nx
import matplotlib.pyplot as plt
import pulp
from community import community_louvain
import streamlit as st

# Assurez-vous que le graphe 'G' est déjà défini au préalable
# Exemple : G = nx.erdos_renyi_graph(100, 0.05) ou un autre graphe

# =====================================================
# 📌 ÉTAPE 7 : Détection des communautés avec Louvain
# =====================================================
st.subheader("🌐 Détection des communautés avec Louvain")

# Application de l'algorithme de Louvain
partition = community_louvain.best_partition(G)
num_communities = len(set(partition.values()))
st.write(f"✅ Nombre de communautés trouvées : {num_communities}")

# Ajout des communautés comme attributs des nœuds
for node, comm_id in partition.items():
    G.nodes[node]['community'] = comm_id

# Visualisation des communautés avec Streamlit
pos = nx.spring_layout(G, k=0.3)  # Position des nœuds
colors = [partition[node] for node in G.nodes()]  # Attribution des couleurs
cmap = plt.cm.viridis  # Palette de couleurs

plt.figure(figsize=(12, 8))
nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color="gray")
sc = nx.draw_networkx_nodes(G, pos, node_size=50, node_color=colors, cmap=cmap, alpha=0.8)

# Ajout de la légende
plt.colorbar(sc, label="Communauté ID")
plt.title("Détection des communautés avec l'algorithme de Louvain")

# Affichage avec Streamlit
st.pyplot(plt)

# =====================================================
# 📌 ÉTAPE 8 : Expérimentation et réglage des distances et poids
# =====================================================
st.subheader("🔍 Expérimentation et réglage des distances et poids")

# Tester plusieurs seuils de distance et observer l'impact
distances_a_tester = [50, 100, 150, 200]  # Seuils de distance en km

for seuil in distances_a_tester:
    G_temp = nx.Graph()
    edge_count = 0

    for i in range(len(bibliotheques)):
        for j in range(i + 1, len(bibliotheques)):
            if edge_count >= max_edges:
                break  
            
            biblio1 = bibliotheques[i]
            biblio2 = bibliotheques[j]
            coord1 = (biblio1[1]["latitude"], biblio1[1]["longitude"])
            coord2 = (biblio2[1]["latitude"], biblio2[1]["longitude"])
            distance = haversine(coord1, coord2)

            if distance < seuil:
                G_temp.add_edge(biblio1[0], biblio2[0], weight=distance)
                edge_count += 1

    st.write(f"📌 Seuil de distance : {seuil} km → Nombre d'arêtes : {G_temp.number_of_edges()}")

# =====================================================
# 📌 Étape 9.1 : Réduction du graphe avant optimisation
# =====================================================
st.subheader("📉 Réduction du graphe avant optimisation")

# Sélection des 200 bibliothèques les plus connectées
top_200_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:200]

# Création d’un sous-graphe avec les bibliothèques les plus importantes
G_reduit = G.subgraph(top_200_nodes).copy()

st.write(f"✅ Nouveau graphe pour optimisation : {G_reduit.number_of_nodes()} nœuds, {G_reduit.number_of_edges()} arêtes.")

# =====================================================
# 📌 ÉTAPE 9.2 : Optimisation avancée - Minimum Dominating Set
# =====================================================
st.subheader("⚙️ Optimisation du Minimum Dominating Set")

# Liste des nœuds du graphe réduit
nodes = list(G_reduit.nodes())

# Définition des variables binaires
x = pulp.LpVariable.dicts("x", nodes, cat=pulp.LpBinary)

# Création du problème d'optimisation
problem = pulp.LpProblem("Minimum_Dominating_Set", pulp.LpMinimize)

# Objectif : minimiser le nombre de bibliothèques sélectionnées
problem += pulp.lpSum([x[i] for i in nodes])

# Contrainte de domination : chaque bibliothèque doit être couverte
for i in nodes:
    neighbors = list(G_reduit.neighbors(i)) + [i]  # Ajouter la bibliothèque elle-même
    problem += pulp.lpSum([x[n] for n in neighbors]) >= 1

st.write("📌 Optimisation en cours...")

# Résolution du problème avec gestion des erreurs
try:
    problem.solve()
    if pulp.LpStatus[problem.status] == "Optimal":
        dominating_set = [i for i in nodes if pulp.value(x[i]) == 1]
        st.write(f"✅ Nombre de bibliothèques sélectionnées pour couvrir tout le réseau : {len(dominating_set)}")
        st.write(f"📌 Bibliothèques sélectionnées : {dominating_set[:10]} ...")
    else:
        st.write(f"⚠ Problème non résolu, statut : {pulp.LpStatus[problem.status]}")
except Exception as e:
    st.write(f"❌ Erreur lors de la résolution : {e}")

# Visualisation du Minimum Dominating Set
pos = nx.spring_layout(G_reduit, k=0.3)

plt.figure(figsize=(12, 8))
nx.draw_networkx_edges(G_reduit, pos, alpha=0.3, edge_color="gray")
nx.draw_networkx_nodes(G_reduit, pos, node_size=50, node_color="blue", alpha=0.5)
nx.draw_networkx_nodes(G_reduit, pos, nodelist=dominating_set, node_size=100, node_color="red", alpha=1)

plt.title("Minimum Dominating Set Connexe")

# Affichage avec Streamlit
st.pyplot(plt)

# =====================================================
# 📌 ÉTAPE 10 : Problématique avancée : Minimum Dominating Set Connexe
# =====================================================
st.subheader("🔧 Minimum Dominating Set Connexe")

# Réduction du graphe pour tester si trop lent
if G.number_of_nodes() > 200:
    sub_nodes = list(G.nodes())[:200]  # Prend un sous-ensemble de 200 nœuds
    G_reduit = G.subgraph(sub_nodes)  # Crée un sous-graphe réduit
else:
    G_reduit = G

nodes = list(G_reduit.nodes())  # Liste des nœuds du sous-graphe

# Définition des variables de décision
x = pulp.LpVariable.dicts('x', nodes, cat=pulp.LpBinary)

# Définition du problème d'optimisation
problem = pulp.LpProblem('MDS', pulp.LpMinimize)

# Fonction objectif : minimiser le nombre de nœuds sélectionnés
problem += pulp.lpSum([x[i] for i in nodes])

# Contrainte : chaque nœud doit être dominé (lui-même ou un voisin)
added_constraints = set()
for i in nodes:
    neighbors_i = tuple(sorted(set(G_reduit.neighbors(i)) | {i}))  # Trie et enlève les doublons
    if neighbors_i not in added_constraints:  # Vérifie si la contrainte existe déjà
        problem += pulp.lpSum([x[n] for n in neighbors_i]) >= 1
        added_constraints.add(neighbors_i)  # Ajoute la contrainte à l'ensemble

# Utilisation du solveur CBC
cbc_path = r"C:\Users\lydia\Downloads\Cbc-releases.2.10.12-windows-2022-msvs-v17-Release-x64\bin\cbc.exe"

from pulp import COIN_CMD

#solver = COIN_CMD(path=cbc_path, msg=True, keepFiles=True)
status = problem.solve(pulp.PULP_CBC_CMD())

# Extraction du Minimum Dominating Set (MDS)
dominating_set = [i for i in nodes if pulp.value(x[i]) == 1]
st.write(f"✅ Nombre de bibliothèques sélectionnées pour couvrir tout le réseau : {len(dominating_set)}")
st.write(f"📌 Bibliothèques sélectionnées : {dominating_set[:10]}...")  # Affiche seulement 10 bibliothèques pour éviter la surcharge

# Visualisation du MDS sur le graphe
pos = nx.spring_layout(G_reduit, k=0.3)

plt.figure(figsize=(12, 8))
nx.draw_networkx_edges(G_reduit, pos, alpha=0.3, edge_color="gray")
nx.draw_networkx_nodes(G_reduit, pos, node_size=50, node_color="blue", alpha=0.5)
nx.draw_networkx_nodes(G_reduit, pos, nodelist=dominating_set, node_size=100, node_color="red", alpha=1)
plt.title("Minimum Dominating Set Connexe")

# Affichage avec Streamlit
st.pyplot(plt)


# 📌 Charger une partie des données pour éviter la surcharge mémoire
fichier_csv = "data/bibliotheques.csv"
df = pd.read_csv(fichier_csv, encoding="utf-8", delimiter=";", nrows=1000)  # Charger uniquement les 1000 premières lignes

# 📌 Nettoyage des données
df = df.dropna(subset=["Latitude", "Longitude", "Surface", "Nombre de bénévoles"])

# 📌 Création de la carte interactive avec Folium
st.title("📚 Analyse des Bibliothèques en France")
st.write("Carte interactive des bibliothèques et analyse du réseau de connexions.")

map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
mymap = folium.Map(location=map_center, zoom_start=6)
marker_cluster = MarkerCluster().add_to(mymap)

for _, row in df.iterrows():
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=f"{row['Nom de l\'établissement']} - {row['Ville']} ({row['Surface']} m²)",
        icon=folium.Icon(color="blue", icon="book"),
    ).add_to(marker_cluster)

st_folium(mymap)  # 📌 Affichage de la carte interactive

# 📌 Graphique interactif des bibliothèques par région
df_counts = df["Région"].value_counts()
fig = px.bar(df_counts, x=df_counts.index, y=df_counts.values, labels={'x': "Région", 'y': "Nombre de Bibliothèques"}, title="📊 Nombre de Bibliothèques par Région")
fig.update_layout(showlegend=False)
st.plotly_chart(fig)

# 📌 Ajout de filtres interactifs
region = st.selectbox("📍 Choisir une région :", df["Région"].unique())
surface_min, surface_max = st.slider("📏 Filtrer par surface (m²) :", int(df["Surface"].min()), int(df["Surface"].max()), (500, 5000))

# 📌 Filtrer selon la sélection
df_filtered = df[(df["Région"] == region) & (df["Surface"].between(surface_min, surface_max))]
st.write(f"📌 Bibliothèques en {region} entre {surface_min} et {surface_max} m² :")
st.dataframe(df_filtered)

# 📌 Création du graphe des bibliothèques
st.subheader("🔗 Réseau des Bibliothèques")
G = nx.Graph()
for _, row in df.iterrows():
    G.add_node(row["Nom de l'établissement"], ville=row["Ville"], latitude=row["Latitude"], longitude=row["Longitude"])

seuil_distance_km = st.sidebar.slider("🔍 Distance max entre bibliothèques (km)", 10, 200, 50, 10)
for i, biblio1 in enumerate(G.nodes(data=True)):
    for j, biblio2 in enumerate(G.nodes(data=True)):
        if i >= j:
            continue
        coord1 = (biblio1[1]["latitude"], biblio1[1]["longitude"])
        coord2 = (biblio2[1]["latitude"], biblio2[1]["longitude"])
        distance = haversine(coord1, coord2)
        if distance < seuil_distance_km:
            G.add_edge(biblio1[0], biblio2[0], weight=distance)

st.sidebar.write(f"📌 Nombre total de connexions : {G.number_of_edges()}")

# 📌 Réseau interactif avec Pyvis
st.subheader("📌 Réseau interactif des bibliothèques")
net = Network(height="600px", width="100%", notebook=False)
for node, data in G.nodes(data=True):
    net.add_node(node, label=node)

for edge in G.edges():
    net.add_edge(edge[0], edge[1])

net.save_graph("graph.html")
st.components.v1.html(open("assets/graph.html", "r", encoding="utf-8").read(), height=600)

st.success("🚀 Application interactive terminée ! Explorez les bibliothèques et connexions en temps réel.")
