import requests
import pandas as pd
import xml.etree.ElementTree as ET
from pyproj import Proj, transform
from datetime import datetime
import time, schedule


def descargar_xml(url, nombre_archivo):
    try:
        respuesta = requests.get(url)
        respuesta.raise_for_status()  # Lanza un error si la solicitud falla
        
        with open(nombre_archivo, 'wb') as archivo:
            archivo.write(respuesta.content)
        
        print(f"Archivo XML guardado como {nombre_archivo}")
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar el archivo: {e}")

def xml_a_dataframe(nombre_archivo):
    try:
        # Parsear el archivo XML
        tree = ET.parse(nombre_archivo)
        root = tree.getroot()
        # Extraer la fecha_hora
        fecha_hora = root.find("fecha_hora").text
        print(fecha_hora)
        # Convertir a timestamp
        fecha_hora_dt = datetime.strptime(fecha_hora, "%d/%m/%Y %H:%M:%S")
        timestamp = fecha_hora_dt.timestamp()

        # Mostrar el timestamp        
        print(timestamp)
        

        # Extraer los datos de cada elemento <pm> dentro de <pms>
        data = []
        for pm in root.findall('.//pm'):
            row = {child.tag: child.text for child in pm}
            data.append(row)

        # Convertir la lista de diccionarios en un DataFrame de pandas
        df = pd.DataFrame(data)
        utm_proj = Proj(proj="utm", zone=30, ellps="WGS84", datum="WGS84")
        wgs84_proj = Proj(proj="latlong", datum="WGS84")
        st_x_o = df['st_x']
        st_y_o = df['st_y']
        # Coordenadas en formato UTM (convertimos la coma en punto flotante)
        st_x = st_x_o.str.replace(',', '.').astype(float)
        st_y = st_y_o.str.replace(',', '.').astype(float)

        # Convertir a latitud y longitud
        df['lon'], df['lat'] = transform(utm_proj, wgs84_proj, st_x, st_y)
       
        # Definir el área (esquina superior izquierda y esquina inferior derecha)
        #esquina_superior_izquierda = (40.439447,-3.683699)  
        #esquina_inferior_derecha = (38.912811,-1.775669)

        # Aplicar la función al DataFrame
        # df_verificado = verificar_coordenadas(df, esquina_superior_izquierda, esquina_inferior_derecha)

        # csv_file_path = str(int(timestamp*100)) +"verificado.csv"        
        # df_verificado.to_csv(csv_file_path, index=False, header=True)
        
        # csv_file_path = str(int(timestamp*100)) +".csv"
        # df.to_csv(csv_file_path, index=False, header=True)

        return df
    except Exception as e:
        print(f"Error al convertir XML a DataFrame: {e}")
        return None

def verificar_coordenadas(df, esquina1, esquina2):
    """
    Añade una columna al DataFrame indicando si la coordenada GPS está dentro del área rectangular.
    
    :param df: DataFrame con columnas 'lat' y 'lon'.
    :param esquina1: Tupla (latitud, longitud) de la primera esquina del área.
    :param esquina2: Tupla (latitud, longitud) de la segunda esquina del área.
    :return: DataFrame con columna adicional 'dentro' (True/False).
    """

    lat_min = min(esquina1[0], esquina2[0]) 
    lat_max = max(esquina1[0], esquina2[0])
    lon_min = min(esquina1[1], esquina2[1])
    lon_max = max(esquina1[1], esquina2[1])
    
    #print(esquina1)
    #print(esquina2)
    
    df["dentro"] = df.apply(lambda row: lat_min <= row["lat"] <= lat_max and 
                                       lon_min <= row["lon"] <= lon_max, axis=1)
    return df


# Ejemplo de uso:
url_xml = "https://informo.madrid.es/informo/tmadrid/pm.xml"  # Reemplaza con la URL del XML

espiras_de_interes = [3492, 3493, 6791, 6792, 3600, 3838, 10178, 10179, 6753, 6754,6130, 6131, 3781, 10202, 10203, 10204,6444, 6449, 10353, 6445,6505, 6506, 6507]

def worker():
    descargar_xml(url_xml, "archivo_descargado.xml")
    df = xml_a_dataframe("archivo_descargado.xml")
    # filtramos por las espiras de interes
    df['idelem'] = pd.to_numeric(df['idelem'], errors='coerce')
    df = df[df['idelem'].isin(espiras_de_interes)]
    
    # Guardamos el archivo filtrado
    csv_file_path = "datos_ayuntamiento_espiras_treal.csv"  
    df.to_csv(csv_file_path, index=False, header=True)



#Lanzar el RPA inicialmente y luego cada hora
worker()

#Scheduler
schedule.every().minute.do(worker)

while True:
    schedule.run_pending()
    time.sleep(600)
