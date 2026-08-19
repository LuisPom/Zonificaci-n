import pandas as pd
import numpy as np
import folium
from folium import plugins, Element
from folium.plugins import MarkerCluster, HeatMap, Fullscreen
from branca.colormap import linear
import branca
import webbrowser
import os

# =============================================================================
# CARGA DE DATOS
# =============================================================================

# Cargar el dataset principal con asignaciones de clústeres y misiones
df = pd.read_csv('dataset_with_missions.csv')
#df = pd.read_csv('evaluation_results/operational_interpretation/dataset_with_missions.csv')
#csv_path_missions = os.path.join('evaluation_results', 'operational_interpretation', 'dataset_with_missions.csv')
#df = pd.read_csv(csv_path_missions)

# Verificar columnas necesarias
required_cols = [
    'centroid_lon', 'centroid_lat', 'cluster', 'operational_mission',
    'is_sensitive_ecosystem', 'area_ha', 'ecosystem', 'province', 'canton',
    'category', 'estimated_trees', 'num_ecosystems'
]
for col in required_cols:
    if col not in df.columns:
        print(f"Advertencia: columna '{col}' no encontrada. Verificando alternativas...")

# Si no existe 'is_sensitive_ecosystem', crearla basada en palabras clave
if 'is_sensitive_ecosystem' not in df.columns:
    df['is_sensitive_ecosystem'] = df['ecosystem'].str.contains(
        'paramo|Manglar|Herbazal de paramo|Bosque siempreverde montano alto',
        case=False, na=False
    ).astype(int)

# Asegurar que cluster sea int
df['cluster'] = df['cluster'].fillna(-1).astype(int)

# Si no existe operational_mission, crearla basada en cluster y área
if 'operational_mission' not in df.columns:
    def assign_mission(row):
        cluster = row['cluster']
        area = row['area_ha']
        if cluster == -1:
            return 'Outlier / Not Recommended'
        elif area > 50000:
            return 'Large Area - Multiple Missions or Multi-Robot'
        elif area > 10000:
            return 'Medium Area - Standard Mission'
        elif area > 1000:
            return 'Small Area - Short Mission (Low autonomy)'
        else:
            return 'Special Configuration Required'
    df['operational_mission'] = df.apply(assign_mission, axis=1)

# =============================================================================
# DEFINICIÓN DE PALETAS DE COLORES Y ESTILOS
# =============================================================================

# Colores para clústeres HDBSCAN (-1, 0, 1, 2, 3)
CLUSTER_COLORS = {
    -1: '#808080',   # Gris - Ruido
    0: '#1f77b4',    # Azul
    1: '#2ca02c',    # Verde
    2: '#ff7f0e',    # Naranja
    3: '#9467bd',    # Púrpura
}

# Nombres descriptivos para clústeres
CLUSTER_NAMES = {
    -1: 'Ruido / No asignado',
    0: 'Clúster 0 - Áreas Andinas Occidentales',
    1: 'Clúster 1 - Bosques de Tierras Bajas y Chocó',
    2: 'Clúster 2 - Grandes Extensiones Amazónicas',
    3: 'Clúster 3 - Intervenciones y Áreas Pequeñas',
}

# Colores para misiones operacionales
MISSION_COLORS = {
    'Outlier / Not Recommended': '#808080',
    'Medium Area - Standard Mission': '#1f77b4',
    'Large Area - Multiple Missions or Multi-Robot': '#ff7f0e',
    'Small Area - Short Mission (Low autonomy)': '#2ca02c',
    'Special Configuration Required': '#9467bd',
}

# Íconos para misiones (FontAwesome)
MISSION_ICONS = {
    'Outlier / Not Recommended': 'exclamation-triangle',
    'Medium Area - Standard Mission': 'robot',
    'Large Area - Multiple Missions or Multi-Robot': 'drone',
    'Small Area - Short Mission (Low autonomy)': 'microchip',
    'Special Configuration Required': 'cogs',
}

# Emojis para misiones (alternativa simple)
MISSION_EMOJIS = {
    'Outlier / Not Recommended': '🚫',
    'Medium Area - Standard Mission': '🤖',
    'Large Area - Multiple Missions or Multi-Robot': '🚁',
    'Small Area - Short Mission (Low autonomy)': '⚙️',
    'Special Configuration Required': '🔧',
}

# Descripción de robots para cada misión
ROBOT_DESCRIPTIONS = {
    'Outlier / Not Recommended': 'No se recomienda intervención robótica',
    'Medium Area - Standard Mission': 'Robot terrestre autónomo con navegación estándar',
    'Large Area - Multiple Missions or Multi-Robot': 'Múltiples robots o UAVs con coordinación',
    'Small Area - Short Mission (Low autonomy)': 'Robot compacto con autonomía limitada',
    'Special Configuration Required': 'Robot con configuración especializada',
}

# =============================================================================
# PREPARACIÓN DE DATOS PARA EL MAPA
# =============================================================================

# Limpiar datos: eliminar filas con coordenadas nulas
df = df.dropna(subset=['centroid_lon', 'centroid_lat'])

# Crear una columna de tamaño de marcador basado en área
df['marker_radius'] = np.clip(np.log1p(df['area_ha']) * 0.8, 3, 25)

# Crear columna de color de clúster
df['cluster_color'] = df['cluster'].map(CLUSTER_COLORS)

# Crear columna de nombre de clúster
df['cluster_name'] = df['cluster'].map(CLUSTER_NAMES)

# Crear una columna de popup con información detallada
def create_popup(row):
    popup_text = f"""
    <div style="font-family: Arial, sans-serif; font-size: 12px; max-width: 300px;">
        <h4 style="margin: 2px 0; color: #2c3e50;">{row['category']}</h4>
        <hr style="margin: 4px 0;">
        <b>Área:</b> {row['area_ha']:.2f} ha<br>
        <b>Provincia:</b> {row['province']}<br>
        <b>Cantón:</b> {row['canton']}<br>
        <b>Ecosistema:</b> {row['ecosystem'][:100]}...<br>
        <b>Clúster:</b> {row['cluster_name']}<br>
        <b>Misión:</b> {row['operational_mission']}<br>
        <b>Robot:</b> {ROBOT_DESCRIPTIONS.get(row['operational_mission'], 'No especificado')}<br>
        <b>Árboles estimados:</b> {int(row['estimated_trees']):,}<br>
        <b>Ecosistemas:</b> {int(row['num_ecosystems'])}<br>
        <b>Sensible:</b> {'✅ Sí' if row['is_sensitive_ecosystem'] else '❌ No'}
    </div>
    """
    return popup_text

df['popup_html'] = df.apply(create_popup, axis=1)

# =============================================================================
# CREACIÓN DEL MAPA BASE
# =============================================================================

# Centro del mapa (Ecuador)
center_lat = -1.5
center_lon = -78.5

# Crear mapa base con un estilo limpio
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=6,
    tiles='CartoDB positron',
    control_scale=True,
    width='100%',
    height='100%'
)

# Añadir capa de satélite como opción
folium.TileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Satélite'
).add_to(m)

folium.TileLayer(
    'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    attr='CartoDB',
    name='CartoDB Light'
).add_to(m)

# =============================================================================
# CAPA 1: CLÚSTERES (HDBSCAN)
# =============================================================================

cluster_group = folium.FeatureGroup(name='📊 Clústeres HDBSCAN', show=True)

# Crear un MarkerCluster para los puntos de clúster
marker_cluster = MarkerCluster(
    name='Clústeres',
    overlay=True,
    control=False,
    show=True
).add_to(cluster_group)

for idx, row in df.iterrows():
    # Determinar color
    color = CLUSTER_COLORS.get(row['cluster'], '#808080')
    
    # Crear marcador circular
    folium.CircleMarker(
        location=[row['centroid_lat'], row['centroid_lon']],
        radius=row['marker_radius'],
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        weight=1,
        popup=folium.Popup(row['popup_html'], max_width=350),
        tooltip=f"{row['category']} - {row['province']} (Clúster {row['cluster']})",
        name=f"Clúster {row['cluster']}"
    ).add_to(marker_cluster)

cluster_group.add_to(m)

# =============================================================================
# CAPA 2: MISIONES OPERACIONALES
# =============================================================================

mission_group = folium.FeatureGroup(name='🤖 Misiones Operacionales', show=True)

# Agrupar por misión para mejor visualización
for mission, group in df.groupby('operational_mission'):
    color = MISSION_COLORS.get(mission, '#808080')
    emoji = MISSION_EMOJIS.get(mission, '❓')
    
    # Subgrupo para cada misión
    for idx, row in group.iterrows():
        folium.CircleMarker(
            location=[row['centroid_lat'], row['centroid_lon']],
            radius=row['marker_radius'] * 0.8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.5,
            weight=0.5,
            popup=folium.Popup(row['popup_html'], max_width=350),
            tooltip=f"{emoji} {mission} - {row['province']}",
        ).add_to(mission_group)

mission_group.add_to(m)

# =============================================================================
# CAPA 3: ECOSISTEMAS SENSIBLES
# =============================================================================

sensitive_group = folium.FeatureGroup(name='🌿 Ecosistemas Sensibles', show=True)

sensitive_df = df[df['is_sensitive_ecosystem'] == True]

for idx, row in sensitive_df.iterrows():
    folium.CircleMarker(
        location=[row['centroid_lat'], row['centroid_lon']],
        radius=row['marker_radius'] * 1.3,
        color='#e74c3c',
        fill=True,
        fill_color='#e74c3c',
        fill_opacity=0.3,
        weight=2,
        popup=folium.Popup(row['popup_html'], max_width=350),
        tooltip=f"🌿 Ecosistema Sensible - {row['province']}",
        dash_array='5, 5'
    ).add_to(sensitive_group)

sensitive_group.add_to(m)

# =============================================================================
# CAPA 4: ANOMALÍAS DE CONSENSO
# =============================================================================

# Cargar anomalías de consenso
try:
    #anomaly_df = pd.read_csv('evaluation_results/anomaly_detection/anomaly_detection_results.csv')
    #csv_path_results = os.path.join('evaluation_results', 'operational_interpretation', 'anomaly_detection_results.csv')
    #df = pd.read_csv(csv_path_results)
    df = pd.read_csv('anomaly_detection_results.csv')
    anomaly_indices = set()
    if 'anomaly_consensus' in anomaly_df.columns:
        anomaly_indices = set(anomaly_df[anomaly_df['anomaly_consensus'] == 1].index)
    else:
        # Buscar por índice o usar columna de anomalía
        for col in ['anomaly_consensus', 'anomaly_if', 'anomaly_lof', 'anomaly_dbscan']:
            if col in anomaly_df.columns:
                anomaly_indices.update(set(anomaly_df[anomaly_df[col] == 1].index))
    
    # Crear grupo de anomalías
    anomaly_group = folium.FeatureGroup(name='⚠️ Anomalías de Consenso', show=True)
    
    for idx in anomaly_indices:
        if idx < len(df):
            row = df.iloc[idx]
            folium.CircleMarker(
                location=[row['centroid_lat'], row['centroid_lon']],
                radius=row['marker_radius'] * 1.5,
                color='#e74c3c',
                fill=True,
                fill_color='#e74c3c',
                fill_opacity=0.2,
                weight=3,
                popup=folium.Popup(
                    f"<b>⚠️ ANOMALÍA DE CONSENSO</b><br>{row['popup_html']}",
                    max_width=350
                ),
                tooltip=f"⚠️ Anomalía - {row['province']}",
                dash_array='4, 4'
            ).add_to(anomaly_group)
    
    anomaly_group.add_to(m)
except Exception as e:
    print(f"No se pudo cargar anomalías: {e}")

# =============================================================================
# CAPA 5: MAPA DE CALOR (DENSIDAD DE ÁREAS)
# =============================================================================

heat_group = folium.FeatureGroup(name='🔥 Mapa de Calor (Densidad)', show=False)

# Preparar datos para heatmap (lat, lon, peso)
heat_data = []
for idx, row in df.iterrows():
    # Peso basado en área normalizada
    weight = np.log1p(row['area_ha']) / 10
    heat_data.append([row['centroid_lat'], row['centroid_lon'], weight])

HeatMap(heat_data, radius=15, blur=20, max_zoom=8, min_opacity=0.3).add_to(heat_group)
heat_group.add_to(m)

# =============================================================================
# CAPA 6: LÍMITES PROVINCIALES (SIMULADOS CON CÍRCULOS)
# =============================================================================

# Agrupar por provincia y mostrar como clusters
province_group = folium.FeatureGroup(name='📍 Provincias (Agrupadas)', show=False)

for province, group in df.groupby('province'):
    # Centroide de la provincia
    lat_mean = group['centroid_lat'].mean()
    lon_mean = group['centroid_lon'].mean()
    count = len(group)
    area_sum = group['area_ha'].sum()
    
    folium.Circle(
        location=[lat_mean, lon_mean],
        radius=np.sqrt(area_sum) * 50,  # Radio proporcional al área
        color='#2c3e50',
        fill=True,
        fill_color='#2c3e50',
        fill_opacity=0.05,
        weight=1,
        popup=f"""
        <b>{province}</b><br>
        Áreas: {count}<br>
        Superficie total: {area_sum:,.0f} ha
        """
    ).add_to(province_group)

province_group.add_to(m)

# =============================================================================
# LEYENDA PERSONALIZADA
# =============================================================================

legend_html = '''
<div style="position: fixed; 
     bottom: 30px; left: 30px; 
     width: 280px; 
     background-color: white; 
     border: 2px solid #2c3e50;
     border-radius: 8px;
     padding: 12px 15px;
     z-index: 9999;
     font-family: Arial, sans-serif;
     font-size: 12px;
     box-shadow: 0 4px 8px rgba(0,0,0,0.2);
     max-height: 400px;
     overflow-y: auto;">
    <h4 style="margin: 0 0 8px 0; color: #2c3e50; text-align: center;">📊 Leyenda</h4>
    <hr style="margin: 4px 0;">
    <b style="color: #2c3e50;">Clústeres HDBSCAN</b><br>
    <span style="color: #808080;">●</span> -1: Ruido<br>
    <span style="color: #1f77b4;">●</span> 0: Andino Occidental<br>
    <span style="color: #2ca02c;">●</span> 1: Tierras Bajas / Chocó<br>
    <span style="color: #ff7f0e;">●</span> 2: Amazónico<br>
    <span style="color: #9467bd;">●</span> 3: Intervenciones<br>
    <hr style="margin: 4px 0;">
    <b style="color: #2c3e50;">Misiones</b><br>
    <span style="color: #1f77b4;">🤖</span> Estándar<br>
    <span style="color: #ff7f0e;">🚁</span> Múltiples / Multi-Robot<br>
    <span style="color: #2ca02c;">⚙️</span> Corta duración<br>
    <span style="color: #9467bd;">🔧</span> Configuración especial<br>
    <span style="color: #808080;">🚫</span> No recomendado<br>
    <hr style="margin: 4px 0;">
    <span style="color: #e74c3c;">⚠️</span> <b>Anomalía de consenso</b><br>
    <span style="color: #e74c3c;">🌿</span> <b>Ecosistema sensible</b>
</div>
'''

m.get_root().html.add_child(Element(legend_html))

# =============================================================================
# CONTROLES ADICIONALES
# =============================================================================

# Añadir control de capas
folium.LayerControl(collapsed=False).add_to(m)

# Añadir pantalla completa
Fullscreen().add_to(m)

# Añadir minimapa
plugins.MiniMap(
    toggle_display=True,
    position='bottomright'
).add_to(m)

# Añadir medidor de distancia
plugins.MeasureControl(
    position='topleft',
    primary_length_unit='kilometers',
    secondary_length_unit='meters',
    primary_area_unit='hectares',
    secondary_area_unit='acres'
).add_to(m)

# Añadir botón de geolocalización
plugins.LocateControl().add_to(m)

# =============================================================================
# GUARDAR Y ABRIR EL MAPA
# =============================================================================

output_file = 'mapa_clustering_ecuador.html'
m.save(output_file)

print(f"✅ Mapa guardado como: {output_file}")
print(f"📊 Total de registros: {len(df)}")
print(f"🔢 Clústeres: {sorted(df['cluster'].unique())}")
print(f"📋 Misiones: {df['operational_mission'].nunique()}")
print(f"🌿 Ecosistemas sensibles: {df['is_sensitive_ecosystem'].sum()}")
print(f"📍 Provincias: {df['province'].nunique()}")
print("\n📋 Distribución de clústeres:")
for cluster in sorted(df['cluster'].unique()):
    count = len(df[df['cluster'] == cluster])
    name = CLUSTER_NAMES.get(cluster, f'Clúster {cluster}')
    print(f"  Clúster {cluster} ({name}): {count} áreas")

print("\n📋 Distribución de misiones:")
for mission, count in df['operational_mission'].value_counts().items():
    print(f"  {mission}: {count} áreas")

# Intentar abrir en el navegador
try:
    webbrowser.open(f'file://{os.path.abspath(output_file)}')
    print("\n🌐 Abriendo mapa en el navegador...")
except:
    print(f"\n📂 Abre manualmente: {os.path.abspath(output_file)}")

# =============================================================================
# ANÁLISIS ADICIONAL: ESTADÍSTICAS POR CLÚSTER
# =============================================================================

print("\n" + "=" * 60)
print("📊 ESTADÍSTICAS POR CLÚSTER")
print("=" * 60)

cluster_stats = df.groupby('cluster').agg({
    'area_ha': ['count', 'sum', 'mean', 'std'],
    'estimated_trees': ['sum', 'mean'],
    'num_ecosystems': ['mean'],
    'is_sensitive_ecosystem': ['sum'],
    'province': lambda x: x.nunique()
}).round(2)

print(cluster_stats)

print("\n" + "=" * 60)
print("🤖 RECOMENDACIONES DE ROBOTS POR MISIÓN")
print("=" * 60)

for mission, desc in ROBOT_DESCRIPTIONS.items():
    count = len(df[df['operational_mission'] == mission])
    print(f"\n{mission} ({count} áreas):")
    print(f"  → {desc}")
    if mission == 'Large Area - Multiple Missions or Multi-Robot':
        print("  → Robots recomendados: UAVs de ala fija o múltiples robots terrestres")
        print("  → Acceso: Vía aérea o terrestre con coordinación de múltiples unidades")
    elif mission == 'Medium Area - Standard Mission':
        print("  → Robot recomendado: Robot terrestre autónomo (ej. Husky, Jackal)")
        print("  → Acceso: Vía terrestre con navegación autónoma")
    elif mission == 'Small Area - Short Mission (Low autonomy)':
        print("  → Robot recomendado: Robot compacto (ej. Turtlebot, Tello)")
        print("  → Acceso: Vía terrestre con autonomía limitada, requiere supervisión")
    elif mission == 'Special Configuration Required':
        print("  → Robot recomendado: Robot con configuración especial (ej. anfibio, trepador)")
        print("  → Acceso: Depende del terreno, puede requerir acceso especial")
    else:  # Outlier
        print("  → No se recomienda intervención robótica")
        print("  → Requiere evaluación adicional del terreno")

print("\n" + "=" * 60)
print("✅ Análisis completado. El mapa interactivo contiene:")
print("   - 6 capas visuales (clústeres, misiones, sensibles, anomalías, calor, provincias)")
print("   - Leyenda personalizada")
print("   - Controles de zoom, medición y pantalla completa")
print("   - Popups con información detallada de cada área")
print("=" * 60)
