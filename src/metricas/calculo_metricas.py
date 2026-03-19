import numpy as np
from scipy import ndimage


def compute_adjacency(arr, res):

    classes = [v for v in np.unique(arr) if v not in (0, -1)]
    adjacency = {c: {} for c in classes}

    rows, cols = arr.shape

    for i in range(rows - 1):
        for j in range(cols - 1):

            c = arr[i, j]

            if c in (0, -1):
                continue

            r = arr[i, j + 1]
            if r != c and r not in (0, -1):
                adjacency[c][r] = adjacency[c].get(r, 0) + res
                adjacency.setdefault(r, {})
                adjacency[r][c] = adjacency[r].get(c, 0) + res

            b = arr[i + 1, j]
            if b != c and b not in (0, -1):
                adjacency[c][b] = adjacency[c].get(b, 0) + res
                adjacency.setdefault(b, {})
                adjacency[b][c] = adjacency[b].get(c, 0) + res

    return adjacency


def compute_IJI(adjacency):

    edges = []

    for c1 in adjacency:
        for c2 in adjacency[c1]:
            if c1 < c2:
                edges.append(adjacency[c1][c2])

    edges = np.array(edges)

    if len(edges) < 3:
        return None

    pij = edges / edges.sum()
    m = len(edges)

    iji = -(np.sum(pij * np.log(pij))) / np.log(m - 1) * 100

    return iji


def compute_border_objects(arr, res):

    structure = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]])

    pixel_area = res ** 2

    labeled, n = ndimage.label(arr > 0, structure=structure)

    border_ids = set()

    border_ids.update(np.unique(labeled[0,:]))
    border_ids.update(np.unique(labeled[-1,:]))
    border_ids.update(np.unique(labeled[:,0]))
    border_ids.update(np.unique(labeled[:,-1]))

    border_ids.discard(0)

    areas = []

    for obj in border_ids:
        mask = labeled == obj
        area = np.sum(mask) * pixel_area
        areas.append(area)

    if not areas:
        return None, None, None

    areas = np.array(areas)

    TABO = areas.max() / 10000
    BIA = areas.max() / 10000
    TAOBIA = areas.sum() / 10000

    return TABO, BIA, TAOBIA


def compute_metrics(arr, res, core_dist=100):

    pixel_area = res ** 2
    total_land_area = arr.size * pixel_area
    total_land_ha = total_land_area / 10000

    classes = [v for v in np.unique(arr) if v not in (0, -1)]

    structure = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]])

    metrics_class = {}

    all_patch_areas = []
    all_patch_perimeters = []

    for cls in classes:

        mask = arr == cls

        labeled, n_patches = ndimage.label(mask, structure=structure)

        if n_patches == 0:
            continue

        patch_pixel_counts = ndimage.sum(
            mask, labeled, index=np.arange(1, n_patches + 1)
        )

        patch_areas = patch_pixel_counts * pixel_area

        patch_perimeters = []

        for i in range(1, n_patches + 1):

            p = labeled == i

            perimeter = (
                np.sum(p & ~np.roll(p,  1, axis=0)) +
                np.sum(p & ~np.roll(p, -1, axis=0)) +
                np.sum(p & ~np.roll(p,  1, axis=1)) +
                np.sum(p & ~np.roll(p, -1, axis=1))
            ) * res

            patch_perimeters.append(perimeter)

        patch_perimeters = np.array(patch_perimeters)

        shape_index = (0.25 * patch_perimeters) / np.sqrt(patch_areas)

        core_areas = []

        for i in range(1, n_patches + 1):

            p = labeled == i
            dist = ndimage.distance_transform_edt(p) * res
            core_area = np.sum(dist >= core_dist) * pixel_area

            core_areas.append(core_area)

        core_areas = np.array(core_areas)

        class_area = patch_areas.sum()

        MPS = patch_areas.mean() / 10000
        PSSD = patch_areas.std(ddof=0) / 10000
        PSCOV = (PSSD / MPS) * 100 if MPS > 0 else 0

        MSI = np.mean(
            patch_perimeters / (2 * np.sqrt(np.pi * patch_areas))
        )

        AWMSI = np.sum(
            (patch_perimeters / (2*np.sqrt(np.pi*patch_areas))) *
            (patch_areas / class_area)
        )

        MPFD = np.mean(
            (2 * np.log(patch_perimeters)) /
            np.log(patch_areas)
        )

        AWMPFD = np.sum(
            ((2*np.log(patch_perimeters)) /
             np.log(patch_areas)) *
            (patch_areas / class_area)
        )

        MPAR = np.mean(patch_perimeters / patch_areas)

        LSI = patch_perimeters.sum() / (2 * np.sqrt(np.pi * total_land_area))

        metrics_class[cls] = {

            "CA": class_area,
            "PLAND": (class_area / total_land_area) * 100,
            "NP": n_patches,
            "PD": n_patches / total_land_ha * 100,

            "AREA_MN": patch_areas.mean(),
            "LPI": (patch_areas.max() / total_land_area) * 100,

            "ED": patch_perimeters.sum() / total_land_ha,
            "PERIM_MN": patch_perimeters.mean(),

            "SHAPE_MN": shape_index.mean(),
            "CORE_MN": core_areas.mean(),

            "c_CA": class_area / 10000,
            "c_PERCENTLAND": (class_area / total_land_area) * 100,
            "c_PD": (n_patches / total_land_area) * 10000 * 100,
            "c_MPS": MPS,
            "c_PSSD": PSSD,
            "c_PSCOV": PSCOV,
            "c_LSI": LSI,
            "c_MSI": MSI,
            "c_AWMSI": AWMSI,
            "c_MPFD": MPFD,
            "c_AWMPFD": AWMPFD,
            "c_MPAR": MPAR,
            "c_ED": patch_perimeters.sum() / total_land_ha,
            "c_TE": patch_perimeters.sum(),
        }

        all_patch_areas.extend(patch_areas)
        all_patch_perimeters.extend(patch_perimeters)

    if all_patch_areas:

        all_patch_areas = np.array(all_patch_areas)
        all_patch_perimeters = np.array(all_patch_perimeters)

        proportions = []

        for cls in classes:
            area_cls = np.sum(arr == cls) * pixel_area
            proportions.append(area_cls / total_land_area)

        proportions = np.array(proportions)

        SHDI = -np.sum(proportions * np.log(proportions + 1e-12))
        SIDI = 1 - np.sum(proportions ** 2)

        m = len(proportions)

        SHEI = SHDI / np.log(m) if m > 1 else 0
        SIEI = SIDI / (1 - 1/m) if m > 1 else 0

        adjacency = compute_adjacency(arr, res)
        IJI = compute_IJI(adjacency)

        TABO, BIA, TAOBIA = compute_border_objects(arr, res)

        metrics_land = {

            "TA": all_patch_areas.sum(),
            "NP": len(all_patch_areas),
            "PD": len(all_patch_areas) / total_land_ha * 100,

            "AREA_MN": all_patch_areas.mean(),
            "LPI": (all_patch_areas.max() / total_land_area) * 100,

            "ED": all_patch_perimeters.sum() / total_land_ha,

            "SHAPE_MN": np.mean(
                (0.25 * all_patch_perimeters) /
                np.sqrt(all_patch_areas)
            ),

            "PR": m,
            "PRD": (m / total_land_area) * 10000 * 100,

            "SHDI": SHDI,
            "SIDI": SIDI,
            "SHEI": SHEI,
            "SIEI": SIEI,

            "IJI": IJI,
            "TABO": TABO,
            "BIA": BIA,
            "TAOBIA": TAOBIA
        }

    else:
        metrics_land = {}

    return metrics_class, metrics_land




landscape_metrics_definitions = {
"c_CA": "Class Area: soma das áreas de todos os patches pertencentes a uma determinada classe dentro da paisagem.",
"c_PERCENTLAND": "Percentage of Landscape: porcentagem da paisagem ocupada por uma determinada classe.",
"c_PD": "Patch Density: número de patches de uma classe dividido pela área total da paisagem, padronizado para 100 hectares.",
"c_MPS": "Mean Patch Size: tamanho médio dos patches de uma classe na paisagem.",
"c_PSSD": "Patch Size Standard Deviation: desvio padrão do tamanho dos patches de uma classe.",
"c_PSCOV": "Patch Size Coefficient of Variation: medida da variabilidade do tamanho dos patches em relação ao tamanho médio.",
"c_LSI": "Landscape Shape Index: índice que mede a complexidade da forma da paisagem considerando o comprimento total das bordas.",
"c_MSI": "Mean Shape Index: média da complexidade da forma dos patches de uma classe.",
"c_AWMSI": "Area Weighted Mean Shape Index: média da complexidade da forma dos patches ponderada pela área de cada patch.",
"c_MPFD": "Mean Patch Fractal Dimension: dimensão fractal média dos patches, indicando o grau de complexidade da forma.",
"c_AWMPFD": "Area Weighted Mean Patch Fractal Dimension: dimensão fractal média ponderada pela área dos patches.",
"c_ED": "Edge Density: comprimento total das bordas entre patches dividido pela área da paisagem.",
"c_MPAR": "Mean Perimeter Area Ratio: média da razão entre o perímetro e a área dos patches.",
"c_NP": "Number of Patches: número total de patches de uma determinada classe.",
"c_TE": "Total Edge: comprimento total das bordas associadas aos patches de uma classe.",
"PR": "Patch Richness: número de diferentes tipos de classes presentes na paisagem.",
"PRD": "Patch Richness Density: número de classes presentes na paisagem dividido pela área total, padronizado para 100 hectares.",
"SHDI": "Shannon Diversity Index: índice que mede a diversidade da paisagem considerando a proporção de cada classe.",
"SIDI": "Simpson Diversity Index: índice de diversidade que mede a probabilidade de dois pixels escolhidos aleatoriamente pertencerem a classes diferentes.",
"SHEI": "Shannon Evenness Index: mede o quão uniformemente as classes estão distribuídas na paisagem com base no índice de Shannon.",
"SIEI": "Simpson Evenness Index: mede a uniformidade da distribuição das classes na paisagem com base no índice de Simpson.",
"IJI": "Interspersion and Juxtaposition Index: mede o grau de mistura e adjacência entre diferentes classes na paisagem.",
"TABO": "Total Area of the Biggest Object: área total do maior objeto que intersecta a paisagem.",
"BIA": "Biggest Intersection Area: maior área de interseção entre um objeto e a paisagem.",
"TAOBIA": "Total Area of Object with Biggest Intersection: área total do objeto que possui a maior interseção com a paisagem."
}



