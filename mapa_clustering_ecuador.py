#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mapa interactivo de clustering y anomalías para datos de reforestación en Ecuador.
Genera un mapa HTML con capas de clústeres, misiones operacionales, ecosistemas sensibles,
anomalías de consenso, mapa de calor y agrupación por provincias.
Diseñado para ejecutarse en GitHub Actions y publicarse en GitHub Pages.
"""

import os
import sys
import webbrowser
import pandas as pd
import numpy as np
import folium
from folium import plugins, Element
from folium.plugins import MarkerCluster, HeatMap, Fullscreen, MiniMap, MeasureControl, LocateControl

# =============================================================================
# CONFIGURACIÓN DE ARCHIVOS
# =============================================================================

MAIN_CSV = 'dataset_with_missions.csv'        # Archivo principal con clusters y misiones
ANOMALY_CSV = 'anomaly_detection_results.csv'  # Archivo de anomalías (opcional)
OUTPUT_HTML = 'index.html'                    # Nombre para GitHub Pages

# =============================================================================
# CARGA DE DATOS PRINCIPAL (CON VALIDACIONES)
# =============================================================================

def cargar_datos_principales():
    """Carga el CSV principal y verifica que existan las columnas necesarias."""
    try:
        df = pd.read_csv(MAIN_CSV, encoding='utf-8')
        print(f"✅ Archivo '{MAIN_CSV}' cargado correctamente. Filas: {len(df)}")
    except FileNotFoundError:
        print(f"❌ No se encontró '{MAIN_CSV}'. Verifica que el archivo esté en la raíz.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error al leer '{MAIN_CSV}': {e}")
        sys.exit(1)

    # Columnas esenciales que deben existir
    columnas_requeridas = [
        'centroid_lon', 'centroid_lat', 'cluster', 'is_sensitive_ecosystem',
        'operational_mission', 'area_ha', 'ecosystem', 'province', 'canton',
        'category', 'estimated_trees', 'num_ecosystems'
    ]
    faltantes = [col for col in columnas_requeridas if col not in df.columns]
    if faltantes:
        print(f"⚠️ Faltan columnas: {faltantes}. Se intentará crear automáticamente si es posible.")
        # Crear columnas faltantes con valores por defecto
        for col in faltantes:
            if col == 'cluster':
                df['cluster'] = -1
            elif col == 'is_sensitive_ecosystem':
                df['is_sensitive_ecosystem'] = 0
            elif col == 'operational_mission':
                df['operational_mission'] = 'No definida'
            elif col == 'num_ecosystems':
                df['num_ecosystems'] = 1
            elif col == 'estimated_trees':
                df['estimated_trees'] = 0
            else:
                df[col] = ''

    # Asegurar tipos de datos correctos
    df['cluster'] = df['cluster'].fillna(-1).astype(int)
    df['is_sensitive_ecosystem'] = df['is_sensitive_ecosystem'].fillna(0).astype(int)
    df['area_ha'] = df['area_ha'].fillna(0).astype(float)
    df['estimated_trees'] = df['estimated_trees'].fillna(0).astype(int)
    df['num_ecosystems'] = df['num_ecosystems'].fillna(1).astype(int)

    # Convertir columnas de texto a string para evitar errores
    for col in ['ecosystem', 'province', 'canton', 'category', 'operational_mission']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    print("✅ Datos preparados correctamente.")
    return df

# =============================================================================
# CARGA DE ANOMALÍAS (OPCIONAL)
# =============================================================================

def cargar_anomalias():
    """Carga el archivo de anomalías si existe, devuelve un set de índices de filas anómalas."""
    try:
        anomaly_df = pd.read_csv(ANOMALY_CSV, encoding='utf-8')
        print(f"✅ Archivo '{ANOMALY_CSV}' cargado.")
        # Buscar columna de anomalía de consenso
        if 'anomaly_consensus' in anomaly_df.columns:
            indices = set(anomaly_df[anomaly_df['anomaly_consensus'] == 1].index)
            print(f"   Encontradas {len(indices)} anomalías de consenso.")
            return indices
        else:
            # Si no existe, buscar cualquier columna que empiece con 'anomaly_'
            cols = [c for c in anomaly_df.columns if c.startswith('anomaly_')]
            if cols:
                mask = anomaly_df[cols].sum(axis=1) > 0
                indices = set(anomaly_df[mask].index)
                print(f"   Encontradas {len(indices)} anomalías en al menos un método.")
                return indices
            else:
                print("   No se encontraron columnas de anomalía. Se ignorará esta capa.")
                return set()
    except FileNotFoundError:
        print(f"⚠️ Archivo '{ANOMALY_CSV}' no encontrado. Se omitirá la capa de anomalías.")
        return set()
    except Exception as e:
        print(f"⚠️ Error al cargar anomalías: {e}. Se omitirá la capa.")
        return set()

# =============================================================================
# DEFINICIÓN DE ESTILOS Y ETIQUETAS
# =============================================================================

CLUSTER_COLORS = {
    -1: '#808080',   # Gris - Ruido
    0: '#1f77b4',    # Azul
    1: '#2ca02c',    # Verde
    2: '#ff7f0e',    # Naranja
    3: '#9467bd',    # Púrpura
}

CLUSTER_NAMES = {
    -1: 'Ruido / No asignado',
    0: 'Clúster 0 - Áreas Andinas Occidentales',
    1: 'Clúster 1 - Bosques de Tierras Bajas y Chocó',
    2: 'Clúster 2 - Grandes Extensiones Amazónicas',
    3: 'Clúster 3 - Intervenciones y Áreas Pequeñas',
}

MISSION_COLORS = {
    'Outlier / Not Recommended': '#808080',
    'Medium Area - Standard Mission': '#1f77b4',
    'Large Area - Multiple Missions or Multi-Robot': '#ff7f0e',
    'Small Area - Short Mission (Low autonomy)': '#2ca02c',
    'Special Configuration Required': '#9467bd',
}

MISSION_EMOJIS = {
    'Outlier / Not Recommended': '🚫',
    'Medium Area - Standard Mission': '🤖',
    'Large Area - Multiple Missions or Multi-Robot': '🚁',
    'Small Area - Short Mission (Low autonomy)': '⚙️',
    'Special Configuration Required': '🔧',
}

ROBOT_DESCRIPTIONS = {
    'Outlier / Not Recommended': 'No se recomienda intervención robótica',
    'Medium Area - Standard Mission': 'Robot terrestre autónomo con navegación estándar',
    'Large Area - Multiple Missions or Multi-Robot': 'Múltiples robots o UAVs con coordinación',
    'Small Area - Short Mission (Low autonomy)': 'Robot compacto con autonomía limitada',
    'Special Configuration Required': 'Robot con configuración especializada',
}

def crear_popup(row):
    """Genera el contenido HTML del popup para cada marcador."""
    area = row.get('area_ha', 0)
    province = row.get('province', '')
    canton = row.get('canton', '')
    ecosystem = row.get('ecosystem', '')[:100]
    cluster_name = CLUSTER_NAMES.get(row.get('cluster', -1), 'Desconocido')
    mission = row.get('operational_mission', 'No definida')
    robot_desc = ROBOT_DESCRIPTIONS.get(mission, '')
    trees = int(row.get('estimated_trees', 0)) if pd.notna(row.get('estimated_trees')) else 0
    num_ecos = int(row.get('num_ecosystems', 0)) if pd.notna(row.get('num_ecosystems')) else 0
    sensitive = row.get('is_sensitive_ecosystem', 0)

    return f"""
    <div style="font-family: Arial, sans-serif; font-size: 12px; max-width: 300px;">
        <h4 style="margin: 2px 0; color: #2c3e50;">{row.get('category', '')}</h4>
        <hr style="margin: 4px 0;">
        <b>Área:</b> {area:,.2f} ha<br>
        <b>Provincia:</b> {province}<br>
        <b>Cantón:</b> {canton}<br>
        <b>Ecosistema:</b> {ecosystem}...<br>
        <b>Clúster:</b> {cluster_name}<br>
        <b>Misión:</b> {mission}<br>
        <b>Robot:</b> {robot_desc}<br>
        <b>Árboles estimados:</b> {trees:,}<br>
        <b>N° ecosistemas:</b> {num_ecos}<br>
        <b>Ecosistema sensible:</b> {'✅ Sí' if sensitive else '❌ No'}
    </div>
    """

# =============================================================================
# CREACIÓN DEL MAPA
# =============================================================================

def crear_mapa(df, indices_anomalias):
    """Construye el mapa folium con todas las capas."""
    # Centro de Ecuador
    center_lat = -1.5
    center_lon = -78.5

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='CartoDB positron',
        control_scale=True,
        width='100%',
        height='100%'
    )

    # Capas base opcionales
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

    # Preparar datos para marcadores
    df['marker_radius'] = np.clip(np.log1p(df['area_ha']) * 0.8, 3, 25)
    df['cluster_color'] = df['cluster'].map(CLUSTER_COLORS)
    df['cluster_name'] = df['cluster'].map(CLUSTER_NAMES)
    df['popup_html'] = df.apply(crear_popup, axis=1)

    # --- CAPA 1: Clústeres (principal) ---
    cluster_group = folium.FeatureGroup(name='📊 Clústeres HDBSCAN', show=True)
    marker_cluster = MarkerCluster(name='Clústeres').add_to(cluster_group)

    for _, row in df.iterrows():
        color = CLUSTER_COLORS.get(row['cluster'], '#808080')
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
        ).add_to(marker_cluster)
    cluster_group.add_to(m)

    # --- CAPA 2: Misiones operacionales ---
    mission_group = folium.FeatureGroup(name='🤖 Misiones Operacionales', show=True)
    for mission, group in df.groupby('operational_mission'):
        color = MISSION_COLORS.get(mission, '#808080')
        emoji = MISSION_EMOJIS.get(mission, '❓')
        for _, row in group.iterrows():
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

    # --- CAPA 3: Ecosistemas sensibles ---
    sensitive_group = folium.FeatureGroup(name='🌿 Ecosistemas Sensibles', show=True)
    sensitive_df = df[df['is_sensitive_ecosystem'] == 1]
    for _, row in sensitive_df.iterrows():
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

    # --- CAPA 4: Anomalías de consenso ---
    if indices_anomalias:
        anomaly_group = folium.FeatureGroup(name='⚠️ Anomalías de Consenso', show=True)
        for idx in indices_anomalias:
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

    # --- CAPA 5: Mapa de calor (densidad) ---
    heat_group = folium.FeatureGroup(name='🔥 Mapa de Calor (Densidad)', show=False)
    heat_data = []
    for _, row in df.iterrows():
        weight = np.log1p(row['area_ha']) / 10
        heat_data.append([row['centroid_lat'], row['centroid_lon'], weight])
    HeatMap(heat_data, radius=15, blur=20, max_zoom=8, min_opacity=0.3).add_to(heat_group)
    heat_group.add_to(m)

    # --- CAPA 6: Provincias agrupadas ---
    province_group = folium.FeatureGroup(name='📍 Provincias (Agrupadas)', show=False)
    for province, group in df.groupby('province'):
        lat_mean = group['centroid_lat'].mean()
        lon_mean = group['centroid_lon'].mean()
        area_sum = group['area_ha'].sum()
        folium.Circle(
            location=[lat_mean, lon_mean],
            radius=np.sqrt(area_sum) * 50,
            color='#2c3e50',
            fill=True,
            fill_color='#2c3e50',
            fill_opacity=0.05,
            weight=1,
            popup=f"""
            <b>{province}</b><br>
            Áreas: {len(group)}<br>
            Superficie total: {area_sum:,.0f} ha
            """
        ).add_to(province_group)
    province_group.add_to(m)

    # --- Leyenda personalizada ---
    legend_html = '''
    <div style="position: fixed; bottom: 30px; left: 30px; width: 280px; 
         background-color: white; border: 2px solid #2c3e50; border-radius: 8px;
         padding: 12px 15px; z-index: 9999; font-family: Arial, sans-serif;
         font-size: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);
         max-height: 400px; overflow-y: auto;">
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

    # --- Controles adicionales ---
    folium.LayerControl(collapsed=False).add_to(m)
    Fullscreen().add_to(m)
    MiniMap(toggle_display=True, position='bottomright').add_to(m)
    MeasureControl(
        position='topleft',
        primary_length_unit='kilometers',
        secondary_length_unit='meters',
        primary_area_unit='hectares',
        secondary_area_unit='acres'
    ).add_to(m)
    LocateControl().add_to(m)

    return m

# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def main():
    print("=" * 60)
    print("🌍 Generador de Mapa Interactivo - Clustering Ecuador")
    print("=" * 60)

    # 1. Cargar datos
    df = cargar_datos_principales()
    indices_anomalias = cargar_anomalias()

    # 2. Crear mapa
    mapa = crear_mapa(df, indices_anomalias)

    # 3. Guardar como index.html (para GitHub Pages)
    mapa.save(OUTPUT_HTML)
    print(f"\n✅ Mapa guardado como: {OUTPUT_HTML}")

    # 4. Mostrar resumen
    print("\n📊 Resumen de datos:")
    print(f"   - Total de áreas: {len(df)}")
    print(f"   - Clústeres: {sorted(df['cluster'].unique())}")
    print(f"   - Misiones: {df['operational_mission'].nunique()}")
    print(f"   - Ecosistemas sensibles: {df['is_sensitive_ecosystem'].sum()}")
    print(f"   - Provincias: {df['province'].nunique()}")
    if indices_anomalias:
        print(f"   - Anomalías de consenso: {len(indices_anomalias)}")

    print("\n✅ Proceso completado. El mapa está listo para ser publicado.")

if __name__ == "__main__":
    main()
