import pandas as pd
from sentence_transformers import SentenceTransformer, util
import streamlit as st

st.set_page_config(page_title="Kelag Kontenplan Matching", layout="centered")
st.title("Kelag Konzernkontenplan: Sachkonto-Mapping alt auf neu")

# 1. Excel einlesen
excel_pfad = "Konzernkontenplan_template.xlsx"
df = pd.read_excel(excel_pfad)

# Modell laden (stärkere Variante, mpnet!)
modell_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
@st.cache_resource
def lade_modell():
    return SentenceTransformer(modell_name)
modell = lade_modell()

# GUI Inputs
konto_art = st.radio(
    "Art des neuen Sachkontos",
    ["Bilanz", "GuV"]
)

untertyp = ""
guv_untertyp = ""
if konto_art == "Bilanz":
    untertyp = st.selectbox(
        "Bilanz-Untertyp auswählen",
        ["Aktiv", "Passiv EK", "Passiv FK"]
    )
elif konto_art == "GuV":
    guv_untertyp = st.selectbox(
        "GuV-Untertyp auswählen",
        ["Ertrag", "Aufwand", "Finanzergebnis", "Ertragsteuerung"]
    )

eingabe_bezeichnung = st.text_input("Bezeichnung des neuen Sachkontos")
eingabe_beschreibung = st.text_area("Beschreibung des neuen Sachkontos", height=100)

if st.button("Sachkonto-Vorschläge berechnen"):
    # 2. Filter nach Kontenart und Untertyp
    if konto_art == "GuV":
        if guv_untertyp == "Ertrag":
            startziffern = ('6',)
            df_filtered = df[df["Sachkontonummer"].astype(str).str.startswith(startziffern)]
            konto_info = "GuV - Ertrag"
        elif guv_untertyp == "Aufwand":
            startziffern = ('7',)
            df_filtered = df[df["Sachkontonummer"].astype(str).str.startswith(startziffern)]
            konto_info = "GuV - Aufwand"
        elif guv_untertyp == "Finanzergebnis":
            # Konten mit 80 bis 85 am Anfang
            startziffern = tuple(str(i) for i in range(80, 86))  # "80", ..., "85"
            df_filtered = df[df["Sachkontonummer"].astype(str).str[:2].isin(startziffern)]
            konto_info = "GuV - Finanzergebnis"
        elif guv_untertyp == "Ertragsteuerung":
            # Konten mit 87 am Anfang
            df_filtered = df[df["Sachkontonummer"].astype(str).str[:2] == '87']
            konto_info = "GuV - Ertragsteuerung"
        else:
            # Fallback alle GuV
            startziffern = ('6', '7', '8', '9')
            df_filtered = df[df["Sachkontonummer"].astype(str).str.startswith(startziffern)]
            konto_info = "GuV"
    else:  # Bilanz
        if untertyp == "Aktiv":
            startziffern = ('1', '2')
            df_filtered = df[df["Sachkontonummer"].astype(str).str.startswith(startziffern)]
            konto_info = "Bilanz - Aktiv"
        elif untertyp == "Passiv EK":
            startziffern = ('3',)
            df_filtered = df[df["Sachkontonummer"].astype(str).str.startswith(startziffern)]
            konto_info = "Bilanz - Passiv EK"
        elif untertyp == "Passiv FK":
            startziffern = ('4', '5')
            df_filtered = df[df["Sachkontonummer"].astype(str).str.startswith(startziffern)]
            konto_info = "Bilanz - Passiv FK"
        else:
            startziffern = ('1', '2', '3', '4', '5')
            df_filtered = df[df["Sachkontonummer"].astype(str).str.startswith(startziffern)]
            konto_info = "Bilanz"

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

    # 5. Ähnlichkeit berechnen und alle Treffer mit Score > 0.60 (60%) nehmen
    aehnlichkeit = util.pytorch_cos_sim(eingabe_embedding, alle_embeddings)[0]
    relevante_indices = (aehnlichkeit > 0.60).nonzero().tolist()
    relevante_indices = sorted([idx[0] for idx in relevante_indices], key=lambda i: float(aehnlichkeit[i]), reverse=True)

    if not relevante_indices:
        st.warning("Keine Sachkonten mit einer Wahrscheinlichkeit >60% gefunden.")
    else:
        treffer = []
        for idx in relevante_indices:
            treffer.append({
                "Score": round(float(aehnlichkeit[idx]), 2),
                "Sachkontonummer": df_filtered.iloc[idx]['Sachkontonummer'],
                "Kontenbezeichnung": df_filtered.iloc[idx]['Kontenbezeichnung'],
                "Beschreibung": df_filtered.iloc[idx]['Beschreibung'],
                "Positiv": df_filtered.iloc[idx]['Positiv'],
                "Negativ": df_filtered.iloc[idx]['Negativ'],
                "Position neu": df_filtered.iloc[idx]['Position neu'],
                "Positionsbeschreibung neu": df_filtered.iloc[idx]['Positionsbeschreibung neu'],
            })

        # Input-Zeile als Kopf einfügen (enthält Art + Untertyp)
        input_info = {
            "Score": "INPUT",
            "Sachkontonummer": "",
            "Kontenbezeichnung": eingabe_bezeichnung,
            "Beschreibung": eingabe_beschreibung,
            "Positiv": "",
            "Negativ": "",
            "Position neu": konto_info,
            "Positionsbeschreibung neu": "",
        }
        result_df = pd.DataFrame([input_info] + treffer)

        st.success(f"{len(treffer)} Sachkonten mit Score >60% gefunden.")
        st.dataframe(result_df, hide_index=True)

        # Download-Link für Excel
        output_path = "Matching_Ergebnis_offline.xlsx"
        result_df.to_excel(output_path, index=False)
        with open(output_path, "rb") as f:
            st.download_button("Ergebnis als Excel herunterladen", f, file_name=output_path)

else:
    st.info("Bitte Bezeichnung und Beschreibung eingeben und auf 'Sachkonto-Vorschläge berechnen' klicken.")
