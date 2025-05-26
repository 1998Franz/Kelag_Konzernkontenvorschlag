import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import streamlit as st

# Optional: Ordnerwechsel für lokale Nutzung (für Cloud ggf. anpassen oder auskommentieren)
os.chdir(r"C:\Users\fhaim\OneDrive\Desktop\Kelag_Konzernkontenplan\Projekt")

st.set_page_config(page_title="Kelag Kontenplan Matching", layout="centered")
st.title("Kelag Konzernkontenplan: Sachkonto-Matching")

# 1. Excel einlesen
excel_pfad = "Konzernkontenplan_template.xlsx"
df = pd.read_excel(excel_pfad)

# Modell laden (cache für Speed)
modell_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
@st.cache_resource
def lade_modell():
    return SentenceTransformer(modell_name)
modell = lade_modell()

# GUI Inputs
konto_art = st.radio(
    "Art des neuen Sachkontos",
    ["Bilanz", "GuV"]
)

eingabe_bezeichnung = st.text_input("Bezeichnung des neuen Sachkontos")
eingabe_beschreibung = st.text_area("Beschreibung des neuen Sachkontos", height=100)

if st.button("Sachkonto-Vorschläge berechnen"):
    # 2. Filter nach Kontenart
    if konto_art.lower() == "bilanz":
        startziffern = ('1', '2', '3', '4', '5')
    else:
        startziffern = ('6', '7', '8', '9')
    df_filtered = df[df["Sachkontonummer"].astype(str).str.startswith(startziffern)]

    # 3. Vergleichstext bauen
    def kombiniere_textzeile(row):
        teile = [
            str(row.get("Kontenbezeichnung", "")),
            str(row.get("Beschreibung", "")),
            f"Positiv: {row.get('Positiv', '')}",
            f"Negativ: {row.get('Negativ', '')}",
        ]
        return " ".join([t for t in teile if t and str(t).lower() != 'nan'])
    df_filtered["vergleichstext"] = df_filtered.apply(kombiniere_textzeile, axis=1)

    # 4. Embeddings
    alle_embeddings = modell.encode(df_filtered["vergleichstext"].tolist(), convert_to_tensor=True)
    eingabe_text = f"{eingabe_bezeichnung} {eingabe_beschreibung}"
    eingabe_embedding = modell.encode([eingabe_text], convert_to_tensor=True)

    # 5. Ähnlichkeit berechnen und alle Treffer mit Score > 0.5 (50%) nehmen
    aehnlichkeit = util.pytorch_cos_sim(eingabe_embedding, alle_embeddings)[0]
    relevante_indices = (aehnlichkeit > 0.5).nonzero().tolist()
    relevante_indices = sorted([idx[0] for idx in relevante_indices], key=lambda i: float(aehnlichkeit[i]), reverse=True)

    if not relevante_indices:
        st.warning("Keine Sachkonten mit einer Wahrscheinlichkeit >50% gefunden.")
    else:
        treffer = []
        for idx in relevante_indices:
            treffer.append({
                "Score": float(aehnlichkeit[idx]),
                "Sachkontonummer": df_filtered.iloc[idx]['Sachkontonummer'],
                "Kontenbezeichnung": df_filtered.iloc[idx]['Kontenbezeichnung'],
                "Beschreibung": df_filtered.iloc[idx]['Beschreibung'],
                "Positiv": df_filtered.iloc[idx]['Positiv'],
                "Negativ": df_filtered.iloc[idx]['Negativ'],
                "Position neu": df_filtered.iloc[idx]['Position neu'],
                "Positionsbeschreibung neu": df_filtered.iloc[idx]['Positionsbeschreibung neu'],
            })

        # Input-Zeile als Kopf einfügen
        input_info = {
            "Score": "INPUT",
            "Sachkontonummer": "",
            "Kontenbezeichnung": eingabe_bezeichnung,
            "Beschreibung": eingabe_beschreibung,
            "Positiv": "",
            "Negativ": "",
            "Position neu": konto_art,
            "Positionsbeschreibung neu": "",
        }
        result_df = pd.DataFrame([input_info] + treffer)

        st.success(f"{len(treffer)} Sachkonten mit Score >50% gefunden.")
        st.dataframe(result_df, hide_index=True)

        # Download-Link für Excel
        output_path = "Matching_Ergebnis_offline.xlsx"
        result_df.to_excel(output_path, index=False)
        with open(output_path, "rb") as f:
            st.download_button("Ergebnis als Excel herunterladen", f, file_name=output_path)

else:
    st.info("Bitte Bezeichnung und Beschreibung eingeben und auf 'Sachkonto-Vorschläge berechnen' klicken.")

