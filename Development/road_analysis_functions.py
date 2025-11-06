import pandas as pd
import holidays
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def suma_vehiculos_vias(df, calzada):
    """ This function will sum the vehicles flow of all lanes of a given roadway (calzada) for each time step (Dia, Hora).
    It will return a dataframe with the columns Estación, Dia, Hora and V_Total."""
    df_calzada = df[df['Calzada'] == calzada].copy()
    # we will sum the flow of the two lanes for each time step (formm by columns Dia and Hora)
    df_calzada['V_Total'] = df_calzada.groupby(['Dia', 'Hora'])['V_Total'].transform('sum')
    # we drop the columns that are not needed
    df_calzada = df_calzada[['Estación', 'Dia', 'Hora', 'V_Total']]
    # we will drop the duplicate rows, keeping the first one
    df_calzada = df_calzada.drop_duplicates(keep='first')
    # we will reset the index
    df_calzada = df_calzada.reset_index(drop=True)

    #print(df_calzada.head(30))
    return df_calzada 
    

def agregar_y_comparar_calzadas(df):
    """ This function will compare the two roadways (calzadas) of a given station (estacion).
    It will merge the two roadways df in a inner way, so we will get only rows with data in both roadways.
    It will return a dataframe with the columns Estación, Dia, Hora, V_Total_1, V_Total_2 and Diff.
    V_Total_1 is the sum of the vehicles flow of all lanes of the first roadway (calzada 1) for each time step (Dia, Hora).
    V_Total_2 is the sum of the vehicles flow of all lanes of the second roadway (calzada 2) for each time step (Dia, Hora).
    Diff is the difference between the two roadways. A positive value means that the growing km roadway has more flow than the decreasing km roadway."""

    df_1 = suma_vehiculos_vias(df, 1)
    df_2 = suma_vehiculos_vias(df, 2)
    # add a column n_vias to each dataframe, that indicates the number of lanes of the roadway  
    df_1['n_carriles_1'] = df[df['Calzada'] == 1]['Carril'].nunique()
    df_2['n_carriles_2'] = df[df['Calzada'] == 2]['Carril'].nunique()
    # merge the two dataframes on the columns 'Dia' and 'Hora'
    df_1 = df_1.rename(columns={'V_Total': 'V_Total_1'})
    df_2 = df_2.rename(columns={'V_Total': 'V_Total_2'})
    # Add a column that divide the the flow by the number of lanes
    df_1['Avg_1'] = df_1['V_Total_1'] / df_1['n_carriles_1']
    df_2['Avg_2'] = df_2['V_Total_2'] / df_2['n_carriles_2']
    # Add a ideal but impossible average flow column. (split flow without taking into account the direction of the flow)
    df_2['Flujo_Medio'] = (df_1['V_Total_1'] + df_2['V_Total_2']) / (df_1['n_carriles_1'] + df_2['n_carriles_2'])
    # drop the column 'Estación' from df_2
    df_2 = df_2.drop(columns=['Estación'])
    # merge the two dataframes on the columns 'Dia' and 'Hora'
    df = df_1.merge(df_2, on=['Dia', 'Hora'], how='inner')
    # we drop the rows from days that has not 24 hours of data
    #print("len df before filter incomplete days:", len(df))
    df.groupby(['Dia']).filter(lambda x: len(x) == 24)
    #print("len df after filter incomplete days:", len(df))
    # add the difference between the two calzadas
    df['Diff'] = df['V_Total_1'] - df['V_Total_2']
    return df


def agregar_dia_mes_esLaborable(df):
    # add the weekday of the date
    df['Dia'] = pd.to_datetime(df['Dia'])
    df['weekday'] = df['Dia'].dt.day_name()
    df['month'] = df['Dia'].dt.month_name()
    # workday is a boolean value that indicates if the day is not a weekend or a holiday in madrid
    holidays_madrid = holidays.Spain(prov='MD')
    df['workday'] = df['Dia'].apply(lambda x: x not in holidays_madrid and x.weekday() < 5)
    return df

def plot_diferencia_calzadas_grouped(df, estacion, weekdays=True):
    month_colors = {
        'January': 'blue',
        'February': 'green',
        'March': 'red',
        'April': 'cyan',
        'May': 'magenta',
        'June': 'orange',
        'July': 'purple',
        'August': 'brown',
        'September': 'pink',
        'October': 'gray',
        'November': 'olive',
        'December': 'black'
    }

    # Define the desired weekday order
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Filter only available weekdays in data, preserving the order
    available_weekdays = [day for day in weekday_order if day in df['weekday'].unique()]

    if weekdays:
        n_plots = len(available_weekdays)
    else:
        n_plots = 0    
    # Create subplots with a shared y-axis
    fig, axs = plt.subplots(2 + n_plots, 1, figsize=(6, 3 * (2 + n_plots)), sharey=True)
    # Handle the case with only one subplot
    if n_plots == 1:
        axs = [axs]

    # plot all workdays in the same plot
    df_workdays = df[df['workday']]
    used_months = df_workdays['month'].unique()
    handles = [
        mpatches.Patch(color=month_colors.get(month, 'white'), label=month)
        for month in used_months
    ]

    for day, group in df_workdays.groupby(df_workdays['Dia'].dt.date):
        day_month = group.iloc[0]['month']
        color = month_colors.get(day_month, 'white')
        axs[0].plot(group['Hora'], group['Diff'], marker='o', color=color)
        axs[0].set_title(f'Workdays - Difference between roadway for {estacion}')
        axs[0].set_xlabel('Hora')
        axs[0].set_ylabel('Difference')
        axs[0].set_ylim(-3000, 3000)
        axs[0].legend(handles=handles, loc='upper right')
                      
    # plot all weekends and holidays in the same plot
    df_weekends = df[~df['workday']]
    used_months = df_weekends['month'].unique()
    handles = [
        mpatches.Patch(color=month_colors.get(month, 'white'), label=month)
        for month in used_months
    ]
    for day, group in df_weekends.groupby(df_weekends['Dia'].dt.date):
        day_month = group.iloc[0]['month']
        color = month_colors.get(day_month, 'white')
        axs[1].plot(group['Hora'], group['Diff'], marker='o', color=color)
        axs[1].set_title(f'Weekends and Holidays - Difference between roadway for {estacion}')
        axs[1].set_xlabel('Hora')
        axs[1].set_ylabel('Difference')
        axs[1].set_ylim(-3000, 3000)
        axs[1].legend(handles=handles, loc='upper right')
        
    if weekdays:
        # Plot each weekday in a separate subplot
        for ax, weekday in zip(axs[2:], available_weekdays):
            df_weekday = df[df['weekday'] == weekday]
            for day, group in df_weekday.groupby(df_weekday['Dia'].dt.date):
                day_month = group.iloc[0]['month']
                color = month_colors.get(day_month, 'white')
                ax.plot(group['Hora'], group['Diff'], marker='o', color=color)

            # Create legend patches for only the months present in this weekday
            used_months = df_weekday['month'].unique()
            handles = [
                mpatches.Patch(color=month_colors.get(month, 'white'), label=month)
                for month in used_months
            ]

            ax.set_title(f'{weekday} - Difference between roadway for {estacion}')
            ax.set_xlabel('Hora')
            ax.set_ylabel('Difference')
            ax.set_ylim(-3000, 3000)
            ax.legend(handles=handles, loc='upper right')

    plt.tight_layout()
    plt.show()
    return fig, axs


def plot_distribution_by_station(df_estacion, station_name):    
    print(f"Station: {station_name}, Number of rows: {len(df_estacion)}")
    
    fig = plt.figure(figsize=(9, 4))

    plt.subplot(1, 2, 1)
    plt.xlim(0, 8000)
    plt.ylim(0, 1000)
    plt.hist(df_estacion['V_Total_1'], bins=int(df_estacion['V_Total_1'].max()/200), color='blue', alpha=0.7)
    plt.xlabel('Flujo Calzada 1')
    plt.ylabel('Frecuencia')
    n_carriles_1 = df_estacion['n_carriles_1'].iloc[0]
    plt.title(f'({station_name}) - Nº Carriles = {n_carriles_1}')

    # Right subplot: distribution of Calzada 2
    plt.subplot(1, 2, 2)
    plt.xlim(0, 8000)
    plt.ylim(0, 1000)
    plt.hist(df_estacion['V_Total_2'], bins=int(df_estacion['V_Total_2'].max()/200), color='green', alpha=0.7)
    plt.xlabel('Flujo Calzada 2')
    plt.ylabel('Frecuencia')
    n_carriles_2 = df_estacion['n_carriles_2'].iloc[0]
    plt.title(f'({station_name}) - Nº Carriles = {n_carriles_2}')

    plt.tight_layout()
    plt.show()
    return fig

def compute_trasvase(row, trafico_minimo=1500, factor_mejora_minimo=1.3):
    # if both flows are lower than 1500 then no trasvase
    if row['V_Total_1'] < trafico_minimo and row['V_Total_2'] < trafico_minimo:
        return False
    
    flujo_medio = row['Flujo_Medio']
    
    # identify the calzada with higher and lower flow
    if row['V_Total_1'] >= row['V_Total_2']:
        high_flow = row['V_Total_1']
        low_flow = row['V_Total_2']
        n_high = row['n_carriles_1']
        n_low = row['n_carriles_2']
    else:
        high_flow = row['V_Total_2']
        low_flow = row['V_Total_1']
        n_high = row['n_carriles_2']
        n_low = row['n_carriles_1']

    # if n_low < 2 then no trasvase
    if n_low < 2:
        return False
    
    # original average per lane using original number of carriles
    orig_avg_high = high_flow / n_high
    orig_avg_low = low_flow / n_low
    original_avg_error = abs(flujo_medio - orig_avg_high)*n_high + abs(flujo_medio - orig_avg_low)*n_low

    modified_high = high_flow / (n_high + 1)
    modified_low = low_flow / (n_low - 1)
    modified_avg_error = abs(flujo_medio - modified_high)*(n_high + 1) + abs(flujo_medio - modified_low)*(n_low - 1)
    # if the modified average error is lower than the original average error then we will do the trasvase
    return modified_avg_error * factor_mejora_minimo < original_avg_error

def plot_trasveses_por_hora(df, estacion):
    # Agrupamos los datos por la hora y contamos el número de trasvases (True)
    trasvases_por_hora = df.groupby('Hora')['trasvase'].sum().reindex(range(24), fill_value=0)
    # count no_trasvases_por_hora that is all False values
    no_trasvases_por_hora = df.groupby('Hora')['trasvase'].apply(lambda x: (~x).sum()).reindex(range(24), fill_value=0)
    """
    # Plotting the trasvases
    plt.figure(figsize=(7, 4))
    plt.bar(trasvases_por_hora.index, trasvases_por_hora.values, color='#66B3FF')
    plt.title(f"Trasvases por hora en estación {estacion}")
    plt.xlabel("Hora del día")
    plt.ylabel("Número de trasvases")
    plt.xticks(range(24))
    plt.show()
    """
    # Plotting the trasvases and no_trasvases together
    fig = plt.figure(figsize=(7, 4))
    plt.bar(trasvases_por_hora.index, trasvases_por_hora.values, color='green', label='Si Trasvase')
    plt.bar(no_trasvases_por_hora.index, no_trasvases_por_hora.values, bottom=trasvases_por_hora.values, color='orange', label='No Trasvase')
    plt.title(f"Trasvases y No Trasvases por hora en estación {estacion}")
    plt.xlabel("Hora del día")
    plt.ylabel("Número de eventos")
    plt.xticks(range(24))
    plt.legend()
    plt.show()
    return fig


def filter_invalid_days(df, max_0s):
    df_filtered = pd.DataFrame()
    for station in df['Estación'].unique():
        station_data = df[df['Estación'] == station]
        
        # Filtro por ejemplo dias con mas de 24 valores = 0, que en este caso como tiene hasta 6 carriles en algunas 
        # estaciones puede ser desde 1/4 hasta 1/6 de los valores horarios totales por dia
        invalid_days = (
            station_data[station_data['intensidad'] == 0]
            .groupby('Dia')
            .filter(lambda x: len(x) > max_0s)['Dia']  # Parametro de 24h por ejemplo
            .unique()
        )
        # Excluir los días identificados
        df_station_filtered = station_data[~station_data['Dia'].isin(invalid_days)]
        df_filtered = pd.concat([df_filtered, df_station_filtered], ignore_index=True)
    return df_filtered