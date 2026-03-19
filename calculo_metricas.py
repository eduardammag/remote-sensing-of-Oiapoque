import numpy as np
from scipy import ndimage
import warnings

def compute_metrics(arr, res, core_dist=100):
    """
    Calcula métricas de paisagem inspiradas no FRAGSTATS,
    usando conectividade 4 e perímetro por faces expostas.

    Parâmetros
    ----------
    arr : 2D numpy array
        Raster com códigos de classes
    res : float
        Resolução do pixel (metros)
    core_dist : float
        Distância mínima para definição de área núcleo (m)

    Retorna
    -------
    metrics_class : dict
        Métricas por classe
    metrics_land : dict
        Métricas da paisagem
    """

    # ================================
    # PARÂMETROS GERAIS
    # ================================

    pixel_area = res ** 2                      # área do pixel (m²)
    total_land_area = arr.size * pixel_area    # área total da paisagem (m²)
    total_land_ha = total_land_area / 10000    # área total (hectares)

    # Considera apenas classes válidas
    classes = [v for v in np.unique(arr) if v not in (0, -1)]

    # Estrutura de conectividade 4 (rook's case)
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]])

    metrics_class = {}

    # Listas auxiliares para métricas de paisagem
    all_patch_areas = []
    all_patch_perimeters = []

    # ================================
    # LOOP POR CLASSE
    # ================================
    for cls in classes:

        mask = arr == cls

        # Identificação de patches com conectividade 4
        labeled, n_patches = ndimage.label(mask, structure=structure)

        if n_patches == 0:
            continue

        # ----------------------------
        # ÁREA DOS PATCHES
        # ----------------------------
        patch_pixel_counts = ndimage.sum(
            mask, labeled, index=np.arange(1, n_patches + 1)
        )

        patch_areas = patch_pixel_counts * pixel_area  # m²

        # ----------------------------
        # PERÍMETRO (faces expostas)
        # ----------------------------
        patch_perimeters = []

        for i in range(1, n_patches + 1):
            p = labeled == i

            # Conta faces expostas em cada direção (N, S, E, W)
            perimeter = (
                np.sum(p & ~np.roll(p,  1, axis=0)) +
                np.sum(p & ~np.roll(p, -1, axis=0)) +
                np.sum(p & ~np.roll(p,  1, axis=1)) +
                np.sum(p & ~np.roll(p, -1, axis=1))
            ) * res

            patch_perimeters.append(perimeter)

        patch_perimeters = np.array(patch_perimeters)

        # ----------------------------
        # SHAPE INDEX (Fragstats)
        # SHAPE = (0.25 * P) / sqrt(A)
        # ----------------------------
        shape_index = (0.25 * patch_perimeters) / np.sqrt(patch_areas)

        # ----------------------------
        # ÁREA NÚCLEO (CORE AREA)
        # ----------------------------
        core_areas = []

        for i in range(1, n_patches + 1):
            p = labeled == i
            dist = ndimage.distance_transform_edt(p) * res
            core_area = np.sum(dist >= core_dist) * pixel_area
            core_areas.append(core_area)

        core_areas = np.array(core_areas)

        # ----------------------------
        # MÉTRICAS POR CLASSE
        # ----------------------------
        class_area = patch_areas.sum()

        metrics_class[cls] = {

            # Áreas
            "CA": class_area,                          # Class Area (m²)
            "PLAND": (class_area / total_land_area) * 100,

            # Patches
            "NP": n_patches,
            "PD": n_patches / total_land_ha * 100,     # patches / 100 ha

            # Tamanho
            "AREA_MN": patch_areas.mean(),
            "LPI": (patch_areas.max() / total_land_area) * 100,

            # Bordas
            "ED": patch_perimeters.sum() / total_land_ha,  # m/ha
            "PERIM_MN": patch_perimeters.mean(),

            # Forma
            "SHAPE_MN": shape_index.mean(),

            # Núcleo
            "CORE_MN": core_areas.mean()
        }

        # Acumula para métricas de paisagem
        all_patch_areas.extend(patch_areas)
        all_patch_perimeters.extend(patch_perimeters)

    # ================================
    # MÉTRICAS DA PAISAGEM
    # ================================
    if all_patch_areas:

        all_patch_areas = np.array(all_patch_areas)
        all_patch_perimeters = np.array(all_patch_perimeters)

        metrics_land = {

            "TA": all_patch_areas.sum(),                       # Total Area
            "NP": len(all_patch_areas),
            "PD": len(all_patch_areas) / total_land_ha * 100,

            "AREA_MN": all_patch_areas.mean(),
            "LPI": (all_patch_areas.max() / total_land_area) * 100,

            "ED": all_patch_perimeters.sum() / total_land_ha,

            "SHAPE_MN": np.mean(
                (0.25 * all_patch_perimeters) / np.sqrt(all_patch_areas)
            )
        }

    else:
        metrics_land = {}

    return metrics_class, metrics_land


"""
---------------- MÉTRICAS CALCULADAS (ESTILO FRAGSTATS) ----------------

As métricas abaixo são calculadas a partir de uma paisagem rasterizada,
utilizando conectividade 4 (rook’s case) e perímetro baseado em faces
expostas dos pixels, seguindo a lógica conceitual do software FRAGSTATS.

---------------------------------------------------------------------
MÉTRICAS POR CLASSE
(cada classe representa um tipo de cobertura/uso do solo)
---------------------------------------------------------------------

- CA (Class Area)
  Área total ocupada pela classe na paisagem (m²).
  Corresponde à soma das áreas de todos os patches da classe.

- NP (Number of Patches)
  Número total de patches distintos da classe.
  Indica o grau de fragmentação da classe.

- PD (Patch Density)
  Densidade de patches da classe (patches por 100 hectares).
  Permite comparar fragmentação entre paisagens de tamanhos diferentes.
  
  Fórmula:
  PD = NP / área_da_paisagem (ha) × 100

- AREA_MN (Mean Patch Area)
  Área média dos patches da classe (m²).
  Valores menores indicam maior fragmentação.

- LPI (Largest Patch Index)
  Índice do maior patch da classe (%).
  Representa a porcentagem da paisagem ocupada pelo maior patch da classe.
  
  Fórmula:
  LPI = (área do maior patch / área total da paisagem) × 100

- PLAND (Percentage of Landscape)
  Percentual da paisagem ocupado pela classe (%).
  Mede a dominância relativa da classe na paisagem.

- PERIM_MN (Mean Patch Perimeter)
  Perímetro médio dos patches da classe (m).
  O perímetro é calculado contando as faces expostas dos pixels
  (conectividade 4), conforme abordagem do FRAGSTATS.

- ED (Edge Density)
  Densidade de borda da classe (m/ha).
  Mede a quantidade de borda por unidade de área da paisagem.
  
  Fórmula:
  ED = soma dos perímetros dos patches / área da paisagem (ha)

- SHAPE_MN (Mean Shape Index)
  Índice médio de forma dos patches da classe.
  Quantifica a complexidade geométrica dos patches.
  
  Fórmula (FRAGSTATS):
  SHAPE = (0.25 × perímetro) / √área
  
  Interpretação:
  - Valor mínimo ≈ 1 → patch compacto (quadrado)
  - Valores maiores → formas mais irregulares e complexas

- CORE_MN (Mean Core Area)
  Área núcleo média dos patches da classe (m²).
  Representa a área interna do patch localizada a uma distância
  igual ou maior que `core_dist` em relação à borda.
  
  Patches pequenos podem apresentar CORE_MN = 0.

---------------------------------------------------------------------
MÉTRICAS DA PAISAGEM
(considerando todas as classes simultaneamente)
---------------------------------------------------------------------

- TA (Total Area)
  Área total ocupada por todos os patches da paisagem (m²),
  excluindo pixels de NoData.

- NP (Number of Patches)
  Número total de patches na paisagem (todas as classes).

- PD (Patch Density)
  Densidade total de patches da paisagem (patches por 100 hectares).
  Mede o nível geral de fragmentação da paisagem.

- AREA_MN (Mean Patch Area)
  Área média de todos os patches da paisagem (m²).

- LPI (Largest Patch Index)
  Percentual da paisagem ocupado pelo maior patch,
  independentemente da classe (%).

- ED (Edge Density)
  Densidade total de borda da paisagem (m/ha),
  considerando o somatório dos perímetros de todos os patches.

- SHAPE_MN (Mean Shape Index)
  Índice médio de forma de todos os patches da paisagem,
  refletindo a complexidade espacial geral.

---------------------------------------------------------------------
NOTAS METODOLÓGICAS
---------------------------------------------------------------------

- Apenas pixels válidos são considerados; valores 0 e -1 são tratados
  como NoData e excluídos dos cálculos.
- Todas as métricas são derivadas de dados rasterizados; portanto,
  os resultados dependem da resolução espacial do raster.
- A identificação de patches utiliza conectividade 4 (rook’s case),
  compatível com a configuração padrão do FRAGSTATS.
- O cálculo do perímetro baseia-se na contagem de faces expostas dos
  pixels, evitando superestimação causada por diagonais.
- As métricas são inspiradas no software FRAGSTATS, mas implementadas
  inteiramente em Python (NumPy + SciPy).
- A métrica CORE_MN pode ser ajustada por meio do parâmetro `core_dist`,
  expresso em metros, devendo ser compatível com a resolução do raster.

"""
