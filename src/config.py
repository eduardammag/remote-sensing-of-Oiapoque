import os

# Diretório raiz do projeto 
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Pasta onde ficam os dados de entrada
INPUT_DIR = os.path.join(ROOT_DIR, "input")

# Pasta de saída
OUTDIR = os.path.join(ROOT_DIR, "output")

# Cria pasta output se não existir
os.makedirs(OUTDIR, exist_ok=True)

csv_out = os.path.join(OUTDIR, "metricas_fragstats_por_celula.csv")
EPS = 1e-6

# PARÂMETROS ESPACIAIS
# CRS de trabalho
CRS_TARGET = "EPSG:31982"

# Resolução raster usada nas análises
RASTER_RESOLUTION = 50

# GRADE
GRADE_PATH = os.path.join(INPUT_DIR, "OIAPOQUE_2x2km.shp")

# CAMADAS DE USO DO SOLO
LAYER_LIST = [(os.path.join(INPUT_DIR, "TC2022_polyconic.gpkg"), "CLASSE")]

# CAMADAS AUXILIARES
ROAD_PATH = os.path.join(INPUT_DIR, "Estrada_QuickOMS.gpkg")
HYDRO_PATH = os.path.join(INPUT_DIR, "HIDRO_AMZLEGAL.shp")
INDIGENOUS_PATH = os.path.join(INPUT_DIR, "Area_indigena_oiapoque.gpkg")
DUMP_PATH = os.path.join(INPUT_DIR, "Lixão_oiapoque.shp")

# RASTERS
POP_RASTER = os.path.join(INPUT_DIR,"GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0_R9_C13_RECC.tif")
BUILT_RASTER = os.path.join(INPUT_DIR,"GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0_R9_C13_REC.tif")

# Número de células para teste (None = todas)
TEST_N = None

# Paralelização
N_WORKERS = 8