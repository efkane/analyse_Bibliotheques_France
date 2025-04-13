import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math
import streamlit as st

def haversine_distance(coord1, coord2):
    R = 6371
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def afficher_graphe():
    st.subheader("📚 Réseau par ouvrages différents et proximité géographique")

    df = pd.read_excel("data/bibliotheques_graph_ouvrages_6plus.xlsx")

    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row["Nom de l'établissement"],
                   latitude=row["latitude"],
                   longitude=row["longitude"],
                   ouvrages=row["Ouvrages"].split(", "))

    ouvrages_cibles = ["Python", "SQL", "Docker"]
    biblios_valides = [
        (row["Nom de l'établissement"], (row["latitude"], row["longitude"]), row["Ouvrages"].split(", "))
        for _, row in df.iterrows()
        if not all(ouvrage in row["Ouvrages"].split(", ") for ouvrage in ouvrages_cibles)
    ]

    for i in range(len(biblios_valides)):
        for j in range(i + 1, len(biblios_valides)):
            nom1, coord1, ouvrages1 = biblios_valides[i]
            nom2, coord2, ouvrages2 = biblios_valides[j]
            if haversine_distance(coord1, coord2) < 100:
                G.add_edge(nom1, nom2)

    pos = nx.spring_layout(G, k=0.3)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color="orange")
    plt.title("Réseau des bibliothèques (ouvrages différents + proximité)")
    st.pyplot(plt)


#chemin optiaml
def afficher_chemin_optimal():
    import pandas as pd
    import math
    import itertools
    import folium
    from folium.plugins import AntPath
    import streamlit as st

    st.subheader("📍 Chemin optimal pour lire tous les ouvrages")

    def haversine_distance(coord1, coord2):
        R = 6371
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)
        a = math.sin(d_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    df = pd.read_excel("data/bibliotheques_graph_ouvrages_6plus.xlsx")
    df["Ouvrages"] = df["Ouvrages"].apply(lambda x: x.split(", "))
    biblios = df[["Nom de l'établissement", "latitude", "longitude", "Ouvrages"]]
    ouvrages_uniques = set(itertools.chain.from_iterable(biblios["Ouvrages"]))
    biblio_list = biblios.head(25).to_dict("records")

    min_combo = None
    min_distance = float("inf")

    for r in range(2, 7):
        for combo in itertools.combinations(biblio_list, r):
            union_ouvrages = set(itertools.chain.from_iterable([b["Ouvrages"] for b in combo]))
            if ouvrages_uniques.issubset(union_ouvrages):
                coords = [(b["latitude"], b["longitude"]) for b in combo]
                nom_bibs = [b["Nom de l'établissement"] for b in combo]
                for perm in itertools.permutations(list(zip(nom_bibs, coords))):
                    dist = 0
                    for i in range(len(perm) - 1):
                        dist += haversine_distance(perm[i][1], perm[i + 1][1])
                    if dist < min_distance:
                        min_distance = dist
                        min_combo = [p[0] for p in perm]
        if min_combo:
            break

    bibs_chemin = [b for b in biblio_list if b["Nom de l'établissement"] in min_combo]
    bibs_ordonnes = sorted(bibs_chemin, key=lambda x: min_combo.index(x["Nom de l'établissement"]))
    latitudes = [b["latitude"] for b in bibs_ordonnes]
    longitudes = [b["longitude"] for b in bibs_ordonnes]
    centre = [sum(latitudes)/len(latitudes), sum(longitudes)/len(longitudes)]
    carte = folium.Map(location=centre, zoom_start=6)

    for i, b in enumerate(bibs_ordonnes):
        nom_biblio = b["Nom de l'établissement"]
        folium.Marker(
            location=[b["latitude"], b["longitude"]],
            popup=f"{i+1}. {nom_biblio}",
            icon=folium.Icon(color="blue", icon="book")
        ).add_to(carte)

    chemin_coords = [(b["latitude"], b["longitude"]) for b in bibs_ordonnes]
    AntPath(locations=chemin_coords, color="red", weight=4).add_to(carte)

    carte.save("chemin_optimal_bibliotheques.html")
    with open("assets/chemin_optimal_bibliotheques.html", "r", encoding="utf-8") as f:
        html = f.read()
        st.components.v1.html(html, height=600)

    st.success(f"📚 Ouvrages couverts : {len(ouvrages_uniques)} | 🏛️ Bibliothèques visitées : {len(min_combo)} | 🛣️ Distance totale : {round(min_distance, 2)} km")
