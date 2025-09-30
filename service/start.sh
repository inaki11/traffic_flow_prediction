#!/bin/bash
set -e

# Activar conda
source /opt/conda/etc/profile.d/conda.sh
conda activate sacyr
conda list

# Iniciar cron
echo "Iniciando cron..."
# Iniciar el servicio cron
service cron start

# Mantener el contenedor vivo
tail -f /dev/null

