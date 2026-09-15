import os
import io
import unicodedata
import re
import pandas as pd
import glob

PATH_DATA = "/home/data/projets-aps/projet6/data/data_ewe/speech_ug/parquet/Ewe_parquet/data_ewe"

def normalize_text(s):
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKC", s).lower().strip()
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith(("P", "S")))
    return re.sub(r"\s+", " ", s).strip()

print("--- Purification du jeu de TRAIN (Anti-Fuite Textuelle) ---")

# 1. Charger les textes de Validation et Test pour créer la liste d'exclusion
print("Chargement des cibles d'exclusion (Test et Validation)...")
df_test = pd.read_parquet(f"{PATH_DATA}/ewe_test_clean.parquet", columns=["transcription"])
df_val = pd.read_parquet(f"{PATH_DATA}/ewe_validation_clean.parquet", columns=["transcription"])

exclude_texts = set()
for df in [df_test, df_val]:
    if "transcription" in df.columns:
        exclude_texts.update(df["transcription"].dropna().apply(normalize_text).unique())

print(f"-> Nombre de phrases uniques à bannir du Train : {len(exclude_texts)}")

# 2. Nettoyer les fichiers de Train sur place
train_files = glob.glob(f"{PATH_DATA}/ewe-train-*.parquet")
if os.path.exists(f"{PATH_DATA}/ewe_train.parquet"):
    train_files.append(f"{PATH_DATA}/ewe_train.parquet")

total_removed = 0

for f in train_files:
    print(f"Traitement de {os.path.basename(f)}...")
    df_train = pd.read_parquet(f)
    
    if "transcription" in df_train.columns:
        initial_len = len(df_train)
        
        # Filtrer : On garde uniquement les lignes dont le texte normalisé n'est PAS dans exclude_texts
        normalized_train = df_train["transcription"].dropna().apply(normalize_text)
        mask_keep = ~normalized_train.isin(exclude_texts)
        
        df_train_clean = df_train[mask_keep]
        removed = initial_len - len(df_train_clean)
        total_removed += removed
        
        # Écraser le fichier d'origine avec la version propre
        df_train_clean.to_parquet(f, index=False)
        print(f"   -> {removed} lignes supprimées ({len(df_train_clean)} restantes)")

print(f"\nPurification terminée ! {total_removed} segments retirés du Train global.")
print("Vous pouvez relancer votre script d'audit pour confirmer le score de 0 overlap.")
