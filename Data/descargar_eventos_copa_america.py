# scripts/descargar_eventos_copa_america.py

import os
import pandas as pd
from statsbombpy import sb

# Crear carpeta donde guardarás los datos
os.makedirs('data/eventos_copa_america', exist_ok=True)

# IDs de competición y temporada
competition_id = 223  # Copa América
season_id = 282       # Año 2024

# Obtener todos los partidos
matches = sb.matches(competition_id=competition_id, season_id=season_id)
match_ids = matches['match_id'].tolist()

# Lista para guardar todos los eventos
todos_los_eventos = []

print(f"🔄 Descargando eventos de {len(match_ids)} partidos de la Copa América 2024...")

for match_id in match_ids:
    try:
        events = sb.events(match_id=match_id)
        events['match_id'] = match_id
        todos_los_eventos.append(events)

        # Guardar JSON individual por partido (opcional)
        ruta_json = f"data/eventos_copa_america/eventos_{match_id}.json"
        events.to_json(ruta_json, orient='records', indent=2)

        print(f"✅ Partido {match_id} descargado.")
    except Exception as e:
        print(f"❌ Error en partido {match_id}: {e}")

# Unir todos los eventos en un solo DataFrame
df_eventos_copa = pd.concat(todos_los_eventos, ignore_index=True)

# Guardar CSV consolidado
ruta_csv = "data/eventos_copa_america/eventos_copa_america_2024.csv"
df_eventos_copa.to_csv(ruta_csv, index=False, encoding='utf-8')
print(f"\n✅ Todos los eventos guardados en: {ruta_csv}")
