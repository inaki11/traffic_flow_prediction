import psycopg2
import pandas as pd
import numpy as np
import os

#############################################
#      CONEXIÓN BD Y ESPIRAS OBJETIVO       #
#############################################

print("---------------  SIMULACION TR  ------------------")

# cargar toda la tabla B_Elementos
conexion = psycopg2.connect(
    host="srvbu-bd",
    port="5432",
    database="BITAL_SACYR_ICONICA",
    user="adminpg",
    password="KRZEy5fxQyNmmVjeUseQ",
)

cursor = conexion.cursor()

cursor.execute('SELECT * FROM public."B_Elementos" WHERE "Fo_Tipoelemento" = 5;')
# Obtener todos los registros
registros = cursor.fetchall()

# crear un df con pandas
columnas = [desc[0] for desc in cursor.description]
df_elementos = pd.DataFrame(registros, columns=columnas)

# Filtramos los nombres en ["M-108-0", "M-139-0", "M-43-0", "M-267-0", "CU-801", "M-109-0", "M-110-0"]
df_elementos = df_elementos[
    df_elementos["Nombre"].isin(
        ["M-108-0", "M-139-0", "M-43-0", "M-267-0", "CU-801-0", "M-109-0", "M-110-0"]
    )
]

# seleccionamos los pares Nombre, Id
for nombre, Id in zip(df_elementos["Nombre"], df_elementos["Id"]):
    print(f'"{nombre}": {Id},')


print(df_elementos)


ids = df_elementos["Id"].dropna().astype(int).tolist()

if not ids:
    print("No hay Id en df_elementos.")
else:
    consulta = 'SELECT * FROM public."HIS_Elemento_Posiciones" WHERE "Fo_Elemento" = ANY(%s::int[]) ORDER BY "Fo_Elemento";'
    cursor.execute(consulta, (ids,))
    resultados = cursor.fetchall()
    columnas_pos = [desc[0] for desc in cursor.description]
    df_posiciones = pd.DataFrame(resultados, columns=columnas_pos)
    print(df_posiciones.head(5))

#############################################
#    CARGAMOS DATOS FILTRADOS DESDE CSV    #
#############################################

# load C:\Users\inaki.campo\Desktop\Sacyr_Ionica\sw-modelo-rutas-trafico\data\datos_sacyr
df_2022 = pd.read_csv("/app/a3_22_24_filtered.csv")
# sustituimos el ultimo caracter del combre de la estacion por un 0
df_2022["Estación"] = df_2022["Estación"].str[:-1] + "0"
print(df_2022.head())


#####################################################################
#    SELECCIONAMOS UN MES COMPLETO DE CADA ESPIRA PARA SIMULAR      #
#####################################################################

df_2022_with_period = (
    df_2022.assign(Dia_dt=pd.to_datetime(df_2022["Dia"], errors="coerce"))
    .dropna(subset=["Dia_dt"])
    .assign(
        Periodo=lambda d: d["Dia_dt"].dt.to_period("M"),
        Dia_norm=lambda d: d["Dia_dt"].dt.normalize(),
        DaysInMonth=lambda d: d["Dia_dt"].dt.daysinmonth,
    )
)

df_coverage = (
    df_2022_with_period.groupby(["Estación", "Periodo"])
    .agg(unique_days=("Dia_norm", "nunique"), days_in_month=("DaysInMonth", "max"))
    .reset_index()
)

first_full_period = (
    df_coverage.loc[df_coverage["unique_days"] == df_coverage["days_in_month"]]
    .sort_values(["Estación", "Periodo"])
    .drop_duplicates(subset="Estación")
)

df_first_full_month = df_2022_with_period.merge(
    first_full_period[["Estación", "Periodo"]],
    on=["Estación", "Periodo"],
    how="inner",
).drop(columns=["Dia_dt", "Periodo", "Dia_norm", "DaysInMonth"])

df_first_full_month["Dia"] = pd.to_datetime(
    df_first_full_month["Dia"], errors="coerce"
).dt.day


print(df_first_full_month.head(20))
print(f"Filas seleccionadas: {len(df_first_full_month)}")

########################################################################################
#        FILTRAMOS EL DIA DE MES Y HORA DE ESTE MES QUE COINCIDE CON EL ACTUAL         #
########################################################################################

# Obtenemos el dia de mes actual y hora en hora peninsular española
from datetime import datetime, timezone
from datetime import timedelta

now_utc = datetime.now(timezone.utc)
now_peninsular = now_utc.astimezone(timezone(timedelta(hours=1)))
current_day = now_peninsular.day
current_hour = now_peninsular.hour
# transformamos la hora a formato datetime string HH:MM:SS
current_hour_str = f"{current_hour:02d}:00:00"
print(f"Dia actual peninsular: {current_day}")
print(f"Hora actual peninsular: {current_hour_str}")

# Seleccionamos para cada estación la fila del día actual y hora
df_selected = df_first_full_month[
    (
        (df_first_full_month["Dia"] == current_day)
        & (df_first_full_month["Hora"] == current_hour_str)
    )
]

# transformamos la columna dia al dia actual dia mes y año formato 2022-01-01
current_date_str = now_peninsular.strftime("%Y-%m-%d")
df_selected = df_selected.assign(Dia=current_date_str)

print(df_selected)

##################################################
#   TRANSFORMAMOS AL FORMATO PARA LA INSERCIÓN   #
##################################################

# Nos quedamos solo con las estaciones presentes en B_Elementos
df_tmp = df_selected[df_selected["Estación"].isin(df_elementos["Nombre"])].copy()

print(df_elementos["Nombre"])
print(df_tmp.head(5))

# --- INICIO DE LA CORRECCIÓN ---

# 1. Convertimos 'Dia' a datetime.
#    Si 'Dia' es solo la fecha (ej: "2022-01-01"), se pondrá a medianoche (00:00:00)
df_tmp["Dia"] = pd.to_datetime(df_tmp["Dia"], errors="coerce")

# 2. Procesamos la 'Hora'
#    Asumimos que 'Hora' es un string de tiempo (ej: "01:00" o "01:00:00")
#    Convertimos a datetime y extraemos el componente .dt.hour
hora_int = pd.to_datetime(df_tmp["Hora"], format="%H:%M:%S", errors="coerce").dt.hour

#    Si el método anterior falla (ej: 'Hora' SÍ era un entero 0-23 pero
#    guardado como string "1", "2"), hora_int estará lleno de NaNs.
#    En ese caso, probamos tu método original (pd.to_numeric).
if hora_int.isnull().all():
    print("Formato 'HH:MM' fallido. Reintentando con 'pd.to_numeric'...")
    hora_int = pd.to_numeric(df_tmp["Hora"], errors="coerce")

# 3. Creamos la columna de hora final (rellenando posibles NaNs)
df_tmp["Hora_int"] = hora_int.fillna(0).astype(int)

# 4. Creamos la marca de tiempo completa (Fecha + Hora)
#    (He quitado el +5 años de prueba para mantener los datos reales)
df_tmp["Fecha"] = df_tmp["Dia"] + pd.to_timedelta(df_tmp["Hora_int"], unit="h")

# --- FIN DE LA CORRECCIÓN ---


# Ponemos los datos en formato wide donde cada fila es una estación+fecha
# Este pivot AHORA sí usará la 'Fecha' con granularidad horaria.
df_wide = df_tmp.pivot_table(
    index=["Estación", "Fecha"],
    columns="Calzada",
    values="intensidad",
    aggfunc="sum",
).reset_index()

print("tras wide (corregido)")
print(df_wide.head(5))  # Ahora deberías ver 24 filas por día (si hay datos)


# Mapeamos Fo_Elemento desde B_Elementos
id_map = df_elementos.set_index("Nombre")["Id"]
df_wide["Fo_Elemento"] = df_wide["Estación"].map(id_map)
# display(df_wide.head(5)) # Esta línea ya no es necesaria, se mostró antes

# Renombramos columnas
calzada_map = {1: "IntensidadCarrilAsc", 2: "IntensidadCarrilDesc"}
df_wide = df_wide.rename(
    columns={c: calzada_map[c] for c in calzada_map if c in df_wide.columns}
)

# Aseguramos que ambas columnas existan
for col in ("IntensidadCarrilAsc", "IntensidadCarrilDesc"):
    if col not in df_wide:
        df_wide[col] = np.nan
    df_wide[col] = pd.to_numeric(df_wide[col], errors="coerce")

# Limpieza y mapeo (tu código original está bien)
df_wide = df_wide.dropna(subset=["Fo_Elemento"])
df_wide["Fo_Elemento"] = df_wide["Fo_Elemento"].astype(int)

coord_from_pos = df_posiciones.groupby("Fo_Elemento")["Coordenadas"].first()
coord_from_elem = df_elementos.set_index("Id")["TR_Coordenadas"]
df_wide["Coordenadas"] = (
    df_wide["Fo_Elemento"]
    .map(coord_from_pos)
    .fillna(df_wide["Fo_Elemento"].map(coord_from_elem))
)

df_wide["Ocupacion"] = None
df_wide["VelMedia"] = None
df_wide["Sentido_EsAscendente"] = None
df_wide["Carriles"] = pd.to_numeric(
    df_wide["Estación"].map(df_elementos.set_index("Nombre")["TR_Carriles"]),
    errors="coerce",
)

# Creamos el df final
df_his = (
    df_wide[
        [
            "Fo_Elemento",
            "Coordenadas",
            "Ocupacion",
            "IntensidadCarrilAsc",
            "VelMedia",
            "Fecha",
            "Sentido_EsAscendente",
            "Carriles",
            "IntensidadCarrilDesc",
        ]
    ]
    .sort_values(["Fo_Elemento", "Fecha"])
    .reset_index(drop=True)
)

print(df_his.head(5))
print(f"Filas preparadas: {len(df_his)}")

##################################
#        INSERCIÓN EN BD         #
##################################


from psycopg2.extras import execute_values

## Insertamos los datos en la tabla HIS_Elemento_Posiciones
# Columnas destino en la tabla
cols_destino = (
    '"Fo_Elemento","Coordenadas","Ocupacion",'
    '"IntensidadCarrilAsc","VelMedia","Fecha","IntensidadCarrilDesc"'
)

# Preparar registros con conversión de NA -> None y tipos nativos
registros_insert = [
    (
        int(row.Fo_Elemento) if pd.notna(row.Fo_Elemento) else None,
        row.Coordenadas,
        row.Ocupacion,
        int(row.IntensidadCarrilAsc) if pd.notna(row.IntensidadCarrilAsc) else None,
        row.VelMedia,
        row.Fecha.to_pydatetime() if hasattr(row.Fecha, "to_pydatetime") else row.Fecha,
        int(row.IntensidadCarrilDesc) if pd.notna(row.IntensidadCarrilDesc) else None,
    )
    for _, row in df_his.iterrows()
]

sql_insert = f'INSERT INTO public."HIS_Elemento_Posiciones" ({cols_destino}) VALUES %s'

try:
    execute_values(cursor, sql_insert, registros_insert, page_size=1000)
    conexion.commit()
    print(f"Filas insertadas: {len(registros_insert)}")
except Exception as e:
    conexion.rollback()
    print(f"Error al insertar: {e}")

print("---------------  SIMULACION TR  ------------------")
