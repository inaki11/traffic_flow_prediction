# Reimportar las bibliotecas necesarias tras el reinicio del entorno
import pandas as pd
import folium
import xml.etree.ElementTree as ET
import psycopg2
import pandas as pd

def parse_linestring(mapa,wkt):
    # Eliminar "LINESTRING(" y ")" del texto
    wkt = wkt.replace("LINESTRING(", "").replace(")", "")
    # Separar los puntos por coma y convertir a tuplas de coordenadas
    coordenadas = [tuple(map(float, point.strip().split())) for point in wkt.split(",")]
    coordenadas_invertidas = [(lat, lon) for lon, lat in coordenadas]  # Invertir posiciones
    folium.PolyLine(coordenadas_invertidas, color="blue", weight=15, opacity=1.0).add_to(mapa)
    return mapa

def parse_multilinestring(mapa,wkt):
    # Eliminar "MULTILINESTRING((" y "))" del texto
    wkt = wkt.replace("MULTILINESTRING((", "").replace("))", "")
    # Separar los segmentos por paréntesis y convertir a listas de coordenadas
    segmentos = wkt.split("),(")
    coordenadas = [[tuple(map(float, point.strip().split())) for point in segmento.split(",")] for segmento in segmentos]
    # Añadir la MULTILINESTRING al mapa como una línea
    for segment in coordenadas:
        segment_invertidas = [(lat, lon) for lon, lat in segment]  # Invertir posiciones
        folium.PolyLine(segment_invertidas, color="blue", weight=15, opacity=1).add_to(mapa)
        
    return mapa


def dibujarRuta(mapa, latOrig, lonOrig, latDest, lonDest):
    # Configurar la conexión
    conexion = psycopg2.connect(
        dbname="BITAL_SACYR_ICONICA",
        user="adminpg",
        password="KRZEy5fxQyNmmVjeUseQ",
        host="srvbu-bd",  # o la IP del servidor
        port="5432"  # Puerto por defecto de PostgreSQL
    )

    # Crear un cursor
    cursor = conexion.cursor()

    # Ejecutar una consulta y cargar los datos en un DataFrame
    consulta = "SELECT ST_AsText(drawPath("+str(latOrig)+","+str(lonOrig)+","+str(latDest)+","+str(lonDest)+"));"
    cursor.execute(consulta)
    resultados = cursor.fetchall()

    # Recorrer los resultados e imprimirlos
    for linestring_wkt in resultados:    
        # Convertir la línea en una lista de coordenadas GPS
        if "MULTILINESTRING" in linestring_wkt[0]:
            mapa = parse_multilinestring(mapa, linestring_wkt[0])
        elif "LINESTRING" in linestring_wkt[0]:
            mapa = parse_linestring(mapa, linestring_wkt[0])   
        
    # Cerrar la conexión
    cursor.close()
    conexion.close()
    
    return mapa

def procesarCamaras():
    # Ruta del archivo XML subido
    xml_file_path = "Camaras.xml"

    # Parsear el archivo XML
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Espacio de nombres utilizado en el XML
    namespace = {"d2": "http://datex2.eu/schema/1_0/1_0"}

    # Extraer datos relevantes (ubicaciones de las cámaras)
    data = []
    for location in root.findall(".//d2:predefinedLocation", namespace):
        cam_id = location.get("id")
        name = location.find(".//d2:value", namespace).text
        lat = location.find(".//d2:latitude", namespace).text
        lon = location.find(".//d2:longitude", namespace).text
        road = location.find(".//d2:roadName/d2:value", namespace).text if location.find(".//d2:roadName/d2:value", namespace) is not None else "N/A"

        data.append({"ID": cam_id, "Nombre": name, "Latitud": lat, "Longitud": lon, "Carretera": road})

    # Convertir la lista de diccionarios en un DataFrame de pandas
    df_camaras = pd.DataFrame(data)
    
    return df_camaras



####### Config #######
camaras = False
espiras = True
dibujar_ruta = False
ruta = [40.405269, -3.656057,40.385670, -3.605679]
csv_ayuntamiento_file_path = "datos_ayuntamiento_espiras_treal.csv"
csv_A3_file_path = "Coordenadas_Estaciones_A-3.csv"
#######################

print(f"Opciones de configuración seleccionadas para la creación del mapa:\ndibujar_camaras:{camaras}\ndibujar_espiras:{espiras}\ndibujar_ruta:{dibujar_ruta}")
print(f"ficheros de entrada:\n{csv_ayuntamiento_file_path}\n{csv_A3_file_path}")
if dibujar_ruta:
    print(f"Coordenadas de origen y destino de la ruta: {ruta}")
   
# Procesado de camaras
if camaras:
    df_camaras = procesarCamaras()

# Recargar el archivo csv
if espiras:
    # Espiras Ayuntamiento
    df_espiras_ayuntamiento = pd.read_csv(csv_ayuntamiento_file_path)
    # Espiras A-3
    df_espiras_a3 = pd.read_csv(csv_A3_file_path)
    print(df_espiras_a3.head())
# Crear un mapa centrado en la media de los puntos
mapa = folium.Map(location=[40.439447,-3.683699], zoom_start=14)

# Añadir los puntos del DataFrame al mapa
if espiras:
    # Espiras Ayuntamiento
    for _, row in df_espiras_ayuntamiento.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=row["idelem"],
            tooltip=folium.Tooltip(row["idelem"], permanent=True)
        ).add_to(mapa)

    # Espiras A-3 con un color diferente
    for _, row in df_espiras_a3.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=row["id"],
            icon=folium.Icon(color="red"),
            tooltip=folium.Tooltip(row["id"], permanent=True)
        ).add_to(mapa)


if camaras:
    for _, row in df_camaras.iterrows():    
        folium.Marker(location=[row["Latitud"], row["Longitud"]], popup=row["ID"], icon=folium.Icon(color="green")).add_to(mapa)

if dibujar_ruta:    
    mapa = dibujarRuta(mapa, *ruta)


# Guardar el mapa en un archivo HTML
mapa_path = "mapa_leaflet.html"
mapa.save(mapa_path)

# Mostrar el enlace de descarga
print(f"Se ha creado el mapa en el fichero: {mapa_path}")
