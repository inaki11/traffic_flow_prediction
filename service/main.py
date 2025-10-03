import torch
import datetime
from zoneinfo import ZoneInfo
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import numpy as np
import os
import holidays
from omegaconf import OmegaConf
from models import get_model


def get_device():
    """Return the best available device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("⚠️ No GPU detected, using CPU instead")
        return torch.device("cpu")


def main():
    # Select device
    device = get_device()

    # Example: create a tensor and move it to the right device
    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)
    z = x + y

    print(f"Tensor computation result on {device}:")
    print(z)

    # Print current date and time
    fecha_actual = datetime.datetime.now(ZoneInfo("Europe/Madrid"))
    print(f"Current date and time: {fecha_actual}")
    # truncamos minutos y segundos
    fecha_actual = fecha_actual.replace(minute=0, second=0, microsecond=0)
    print(f"Fecha actual truncada: {fecha_actual}")

    def crear_input_predicciones_24h(nombre_espira, df_24h):
        print(f"----  Creacion input 24h para {nombre_espira}  -----")
        # fetures dia laborable: Calculamos las fechas del dia objetivo, anterior y posterior
        dia_objetivo = df_24h.iloc[-1]["Fecha"].date()
        print(f"Dia objetivo: {dia_objetivo}")
        dia_posterior = dia_objetivo + pd.Timedelta(days=1)
        print(f"Dia posterior: {dia_posterior}")
        dia_anterior = dia_objetivo - pd.Timedelta(days=1)
        print(f"Dia anterior: {dia_anterior}")

        # Creamos las ventanas de las dos calzadas, 1c es creciente y 2c es decreciente. Formato float32
        ventana_1c = (
            df_24h[["IntensidadCarrilAsc"]].tail(24).T.to_numpy().astype(np.float32)
        )
        ventana_2c = (
            df_24h[["IntensidadCarrilDesc"]].tail(24).T.to_numpy().astype(np.float32)
        )

        md_holidays = holidays.Spain(prov="MD")

        laborable = pd.to_datetime(dia_objetivo).dayofweek < 5
        # Comprobamos si el dia objetivo es laborable para mirar si fue festivo
        if laborable:
            laborable = not (pd.to_datetime(dia_objetivo) in md_holidays)
        new_col = np.array([int(laborable) for _ in range(24)]).reshape(1, 24)
        ventana_1c = np.concatenate((ventana_1c, new_col), axis=0)
        ventana_2c = np.concatenate((ventana_2c, new_col), axis=0)

        # Comprobamos si el dia anterior es laborable
        laborable_anterior = pd.to_datetime(dia_anterior).dayofweek < 5
        if laborable_anterior:
            laborable_anterior = not (pd.to_datetime(dia_anterior) in md_holidays)
        new_col = np.array([int(laborable_anterior) for _ in range(24)]).reshape(1, 24)
        ventana_1c = np.concatenate((ventana_1c, new_col), axis=0)
        ventana_2c = np.concatenate((ventana_2c, new_col), axis=0)

        # Comprobamos si el dia posterior es laborable
        laborable_posterior = pd.to_datetime(dia_posterior).dayofweek < 5
        if laborable_posterior:
            laborable_posterior = not (pd.to_datetime(dia_posterior) in md_holidays)
        new_col = np.array([int(laborable_posterior) for _ in range(24)]).reshape(1, 24)
        ventana_1c = np.concatenate((ventana_1c, new_col), axis=0)
        ventana_2c = np.concatenate((ventana_2c, new_col), axis=0)

        # Convertimos a tensores de torch
        ventana_1c = torch.tensor(ventana_1c).unsqueeze(0)  # shape (1, 4, 24)
        ventana_2c = torch.tensor(ventana_2c).unsqueeze(0)  # shape (1, 4, 24)

        # Hacemos reshape a (1, 24, 4)
        ventana_1c = ventana_1c.permute(0, 2, 1)  # shape (1, 24, 4)
        ventana_2c = ventana_2c.permute(0, 2, 1)  # shape (1, 24, 4)

        # convertimos a float32
        ventana_1c = ventana_1c.float()
        ventana_2c = ventana_2c.float()

        # load model from ./models/{id}.
        print(ventana_1c.shape, ventana_2c.shape)
        print("---- Fin creacion input ----")
        return ventana_1c, ventana_2c

    def predicciones_24h(nombre_espira, input_c1, input_c2, device="cuda"):
        print(f"----  Predicciones 24h para {nombre_espira} -----")

        # predicciones 5-fold CV. Cargamos los 5 modelos de la id para la calzada 1
        preds_c1 = []
        preds_c2 = []
        for fold in range(5):
            # model path absoluto del fichero, cuya ubicacion desconocemos
            # this_dir = os.getcwd()
            this_dir = os.path.dirname(__file__)
            model_path_c1 = os.path.join(
                this_dir, "models", f"{str(nombre_espira)}-1c", f"best_{fold}.pth"
            )
            model_path_c2 = os.path.join(
                this_dir, "models", f"{str(nombre_espira)}-2c", f"best_{fold}.pth"
            )
            print(f"Model path C1: {model_path_c1}")
            print(f"Model path C2: {model_path_c2}")
            # load models config
            config_path_c1 = os.path.join(
                this_dir, "models", f"{str(nombre_espira)}-1c", "config.yaml"
            )
            config_path_c2 = os.path.join(
                this_dir, "models", f"{str(nombre_espira)}-2c", "config.yaml"
            )
            config_c1 = OmegaConf.load(config_path_c1)
            config_c2 = OmegaConf.load(config_path_c2)
            print(config_c1)
            print(config_c2)
            # Instanciamos los objetos modelo
            input_size, output_size = input_c1[0].shape, 24
            print(f"Input size: {input_size}, Output size: {output_size}")
            model_1 = get_model(input_size, output_size, config_c1.model)
            model_2 = get_model(input_size, output_size, config_c2.model)
            model_1.to(device)
            model_2.to(device)
            input_c1 = input_c1.to(device)
            input_c2 = input_c2.to(device)
            print(
                f"Input C1 device: {input_c1.device}, Input C2 device: {input_c2.device}"
            )
            # cargamos checkpint en el modelo
            load_checkpoint(model_1, model_path_c1, device)
            load_checkpoint(model_2, model_path_c2, device)
            model_1.eval()
            model_2.eval()
            # inferencia
            with torch.no_grad():
                pred_1 = model_1(input_c1).cpu().squeeze().numpy()  # shape (24,)
                pred_2 = model_2(input_c2).cpu().squeeze().numpy()  # shape (24,)
                preds_c1.append(pred_1)
                preds_c2.append(pred_2)
        # media de las predicciones
        pred_c1_mean = np.mean(preds_c1, axis=0)
        pred_c2_mean = np.mean(preds_c2, axis=0)
        print(f"Predicción sin desescalar calzada 1c (media 5 folds): {pred_c1_mean}")
        print(f"Predicción sin desescalar calzada 2c (media 5 folds): {pred_c2_mean}")
        print("-------- Fin predicciones ---------")
        return pred_c1_mean, pred_c2_mean

    def load_checkpoint(model, checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)

    def compute_trasvase(
        row,
        trafico_minimo=800,
        flujo_maximo_carril_perjudicado=1600,
        factor_maximo_perjudicado_beneficiado=1.1,
    ):

        asc_is_bigger = row["IntensidadCarrilAsc"] >= row["IntensidadCarrilDesc"]

        # identify the calzada with higher and lower flow
        if asc_is_bigger:
            high_flow = row["IntensidadCarrilAsc"]
            low_flow = row["IntensidadCarrilDesc"]
            n_high = row["n_carriles"]
            n_low = row["n_carriles"]
        else:
            high_flow = row["IntensidadCarrilDesc"]
            low_flow = row["IntensidadCarrilAsc"]
            n_high = row["n_carriles"]
            n_low = row["n_carriles"]

        # original average per lane using original number of carriles
        orig_avg_high = high_flow / n_high
        orig_avg_low = low_flow / n_low
        new_avg_high = high_flow / (n_high + 1)
        new_avg_low = low_flow / (n_low - 1)

        # if both flows are lower than 800 then no trasvase
        if orig_avg_high < trafico_minimo and orig_avg_low < trafico_minimo:
            if asc_is_bigger:
                return False, False, new_avg_high, new_avg_low
            else:
                return False, False, new_avg_low, new_avg_high

        # if the disadvantaged flow is higher than the maximum, dont recommend trasvase
        elif new_avg_low > flujo_maximo_carril_perjudicado:
            if asc_is_bigger:
                return False, False, new_avg_high, new_avg_low
            else:
                return False, False, new_avg_low, new_avg_high

        # if the modified flow of the disadvantaged lane is more than the advantaged lane, by factor_maximo_perjudicado_beneficiado, dont recommend trasvase
        elif new_avg_low > new_avg_high * factor_maximo_perjudicado_beneficiado:
            if asc_is_bigger:
                return False, False, new_avg_high, new_avg_low
            else:
                return False, False, new_avg_low, new_avg_high

        else:
            if asc_is_bigger:
                return True, False, new_avg_high, new_avg_low
            else:
                return False, True, new_avg_low, new_avg_high

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
    print(df_elementos.head(10))

    # Filtramos los nombres en M-108-0, M-139-0, M-43-0
    df_elementos = df_elementos[
        df_elementos["Nombre"].isin(["M-108-0", "M-139-0", "M-43-0"])  # TO DO
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

    #####################################
    #   Predicciones para cada espira   #
    #####################################

    # fecha_actual = pd.Timestamp.now()
    # fecha_actual = pd.to_datetime("2027-01-03 23:00:00")
    print(f"Fecha actual: {fecha_actual}")

    trafico_minimo = 800
    flujo_maximo_carril_perjudicado = 1600
    factor_maximo_perjudicado_beneficiado = 1.1
    print(
        f"Tráfico mínimo: {trafico_minimo}, Factor de Flujo máximo de carril perjudicado respecto beneficiado: {factor_maximo_perjudicado_beneficiado}"
    )

    predicciones_espiras = {}
    df_posiciones["Fecha"] = pd.to_datetime(df_posiciones["Fecha"], errors="coerce")

    # Localizamos solo si es naive, gestionando horas inexistentes (cambio a horario de verano) y ambiguas (otoño)
    if df_posiciones["Fecha"].dt.tz is None:
        df_posiciones["Fecha"] = df_posiciones["Fecha"].dt.tz_localize(
            "Europe/Madrid",
            nonexistent="shift_forward",  # opciones: 'NaT', 'shift_forward', 'shift_backward'
            ambiguous="NaT",  # opciones: 'infer', 'NaT', True/False
        )

    # seleccionamos las ultimas 48h registradas para cada Id de Fo_Elemento en df_posiciones
    for Id, nombre_espira, n_carriles in df_elementos[
        ["Id", "Nombre", "TR_Carriles"]
    ].values:
        print(f"--- Creando Datos Input para espira {nombre_espira}  ID = {Id} ---")
        df_tmp = df_posiciones[df_posiciones["Fo_Elemento"] == Id].copy()
        # float
        df_tmp[["IntensidadCarrilAsc", "IntensidadCarrilDesc"]] = df_tmp[
            ["IntensidadCarrilAsc", "IntensidadCarrilDesc"]
        ].astype(np.float32)
        # Hacemos un escalado standard scaler de las columnas IntensidadCarrilAsc y IntensidadCarrilDesc
        medias = df_tmp[["IntensidadCarrilAsc", "IntensidadCarrilDesc"]].mean()
        stds = df_tmp[["IntensidadCarrilAsc", "IntensidadCarrilDesc"]].std()
        df_tmp["IntensidadCarrilAsc"] = (
            df_tmp["IntensidadCarrilAsc"] - medias["IntensidadCarrilAsc"]
        ) / stds["IntensidadCarrilAsc"]
        df_tmp["IntensidadCarrilDesc"] = (
            df_tmp["IntensidadCarrilDesc"] - medias["IntensidadCarrilDesc"]
        ) / stds["IntensidadCarrilDesc"]
        print(
            f"Media y std de IntensidadCarrilAsc: {medias['IntensidadCarrilAsc']}, {stds['IntensidadCarrilAsc']}"
        )
        print(
            f"Media y std de IntensidadCarrilDesc: {medias['IntensidadCarrilDesc']}, {stds['IntensidadCarrilDesc']}"
        )

        if not df_tmp.empty:
            min_fecha = fecha_actual - pd.Timedelta(hours=48)
            df_48h = df_tmp[
                (df_tmp["Fecha"] > min_fecha) & (df_tmp["Fecha"] <= fecha_actual)
            ]
            print(f"{nombre_espira}  Id: {Id}, Registros últimas 48h: {len(df_48h)}")

            # Por cada df, cogemos las ultimas 24 medidas (si hay tantas)
            if df_48h.shape[0] > 24:
                df_24h = df_48h.nlargest(24, "Fecha")
                df_24h = df_24h[
                    ["Fecha", "IntensidadCarrilAsc", "IntensidadCarrilDesc"]
                ]
                print(df_24h.head(3))
                min_fecha_24h = df_24h["Fecha"].min()
                max_fecha_24h = df_24h["Fecha"].max()
                print(
                    f"{nombre_espira}  Id: {Id}, Min Fecha 24h: {min_fecha_24h}, Max Fecha 24h: {max_fecha_24h}"
                )
                print(f"Procedemos a predecir las próximas 24h para Id: {Id}")
                horas_a_predecir = pd.date_range(
                    start=max_fecha_24h + pd.Timedelta(hours=1), periods=24, freq="h"
                )
                print(
                    f"horas a predecir: {horas_a_predecir[0]} - {horas_a_predecir[-1]}"
                )

                input_c1, input_c2 = crear_input_predicciones_24h(nombre_espira, df_24h)

                pred_c1, pred_c2 = predicciones_24h(
                    nombre_espira, input_c1, input_c2, device
                )

                # desescalamos las predicciones
                pred_c1 = (pred_c1 * stds["IntensidadCarrilAsc"]) + medias[
                    "IntensidadCarrilAsc"
                ]
                pred_c2 = (pred_c2 * stds["IntensidadCarrilDesc"]) + medias[
                    "IntensidadCarrilDesc"
                ]
                print(f"Predicción desescalada calzada 1c (media 5 folds): {pred_c1}")
                print(f"Predicción desescalada calzada 2c (media 5 folds): {pred_c2}")
                # Creamos un df con las predicciones y las horas a predecir
                df_preds = pd.DataFrame(
                    {
                        "Fecha": horas_a_predecir,
                        "IntensidadCarrilAsc": pred_c1,
                        "IntensidadCarrilDesc": pred_c2,
                        "n_carriles": n_carriles,
                        "Fo_Elemento": Id,
                    }
                )
                df_preds[
                    [
                        "AbrirCarrilAsc",
                        "AbrirCarrilDesc",
                        "IntensidadPredCarrilAsc",
                        "IntensidadPredCarrilDesc",
                    ]
                ] = df_preds.apply(
                    lambda row: compute_trasvase(
                        row,
                        trafico_minimo,
                        flujo_maximo_carril_perjudicado,
                        factor_maximo_perjudicado_beneficiado=1.1,
                    ),
                    axis=1,
                    result_type="expand",
                )

                # Dividimos IntensidadCarrilAsc, IntensidadCarrilDesc por n_carriles para tener la intensidad por carril
                df_preds["IntensidadCarrilAsc"] = (
                    df_preds["IntensidadCarrilAsc"] / df_preds["n_carriles"]
                )
                df_preds["IntensidadCarrilDesc"] = (
                    df_preds["IntensidadCarrilDesc"] / df_preds["n_carriles"]
                )

                print(df_preds.head(3))

                # guardamos las predicciones en el diccionario con clave Id-1c y Id-2c
                predicciones_espiras[f"{Id}"] = df_preds

                print("--------------------------------------------------")

            else:
                print(
                    f"Id: {Id} no tiene suficientes registros en las últimas 48h para predicciones (tiene {df_48h.shape[0]})."
                )

        else:
            print(f"Id: {Id} no tiene registros en df_posiciones.")

    ##############################################
    #   Insercion de las predicciones en la BD   #
    ##############################################

    conexion = psycopg2.connect(
        host="srvbu-bd",
        port="5432",
        database="BITAL_SACYR_ICONICA",
        user="adminpg",
        password="KRZEy5fxQyNmmVjeUseQ",
    )

    cursor = conexion.cursor()

    for k, v in predicciones_espiras.items():
        # insert into B_PrediccionTraficoe
        # Preparamos e insertamos (omitimos n_carriles)
        cols_sql = '"Fo_Elemento","Fecha","IntensidadCarrilAsc","IntensidadCarrilDesc","AbrirCarrilAsc","AbrirCarrilDesc","IntensidadPredCarrilAsc","IntensidadPredCarrilDesc"'
        registros = [
            (
                int(r.Fo_Elemento),
                r.Fecha.to_pydatetime(),
                float(r.IntensidadCarrilAsc),
                float(r.IntensidadCarrilDesc),
                bool(r.AbrirCarrilAsc),
                bool(r.AbrirCarrilDesc),
                float(r.IntensidadPredCarrilAsc),
                float(r.IntensidadPredCarrilDesc),
            )
            for r in v.itertuples(index=False)
        ]
        if registros:
            sql = f'INSERT INTO public."B_PrediccionTrafico" ({cols_sql}) VALUES %s'
            try:
                execute_values(cursor, sql, registros, page_size=100)
                conexion.commit()
                print(f"Insertadas {len(registros)} filas para Fo_Elemento {k}")
            except Exception as e:
                conexion.rollback()
                print(f"Error insertando Fo_Elemento {k}: {e}")


if __name__ == "__main__":
    main()
