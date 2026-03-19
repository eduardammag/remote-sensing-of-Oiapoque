import os
import warnings

# CONFIGURAÇÃO GERAL

# Evita poluição de warnings no terminal
warnings.filterwarnings("ignore")

# Diretório raiz do projeto (onde está o script principal)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Pasta onde ficam os dados de entrada
INPUT_DIR = os.path.join(ROOT_DIR, "input")

# Pasta de saída
OUTDIR = os.path.join(ROOT_DIR, "output")

# Cria pasta output se não existir
os.makedirs(OUTDIR, exist_ok=True)

# PARÂMETROS ESPACIAIS

# CRS de trabalho (recomendado usar projeção métrica)
# Exemplo: SIRGAS 2000 / UTM zona 22N
CRS_TARGET = "EPSG:31982"

# Resolução raster usada nas análises
RASTER_RESOLUTION = 50

# GRADE

GRADE_PATH = os.path.join(INPUT_DIR, "OIAPOQUE_2x2km.shp")

# CAMADAS DE USO DO SOLO

LAYER_LIST = [
    (os.path.join(INPUT_DIR, "TC2022_polyconic.gpkg"), "CLASSE")
    # ,(os.path.join(INPUT_DIR, "TC_2022_PRIMARIA.gpkg"), "CLASSE"),
    # (os.path.join(INPUT_DIR, "TC_2022_URBANA.gpkg"), "CLASSE"),
    # (os.path.join(INPUT_DIR, "TC_2022_mineracao.gpkg"), "CLASSE"),
    # (os.path.join(INPUT_DIR, "TC_2022_desmatamento do ano.gpkg"), "CLASSE"),
    # (os.path.join(INPUT_DIR, "TC_2022_VS.gpkg"), "CLASSE"),
]

# da linha 38 ate 42 são classes de uso e cobertura do TC2022_polyconic
#nao precisam extrair as métricas da paisagem deles, pois já foram extraídos 
# do dado anterior, apenas 37 
# pode acessar tudo agregado ou separado

# CAMADAS AUXILIARES
ROAD_PATH = os.path.join(INPUT_DIR, "Estrada_QuickOMS.gpkg")
HYDRO_PATH = os.path.join(INPUT_DIR, "HIDRO_AMZLEGAL.shp")
INDIGENOUS_PATH = os.path.join(INPUT_DIR, "Area_indigena_oiapoque.gpkg")
DUMP_PATH = os.path.join(INPUT_DIR, "Lixão_oiapoque.shp")
MINING_PATH = os.path.join(INPUT_DIR, "TC_2022_mineracao.gpkg")
 
#tirar repetição da mineração  


# RASTERS

POP_RASTER = os.path.join(
    INPUT_DIR,
    "GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0_R9_C13_RECC.tif"
)

BUILT_RASTER = os.path.join(
    INPUT_DIR,
    "GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0_R9_C13_REC.tif"
)

# EXECUÇÃO
# Número de células para teste (None = todas)
TEST_N = None

# Paralelização
# Windows costuma dar problema com multiprocess
N_WORKERS = 1