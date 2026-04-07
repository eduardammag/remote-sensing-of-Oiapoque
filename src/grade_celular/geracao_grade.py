from src.grade_celular.espaco_celular import load_limite, create_grid, clip_grid, save_grid, plot_grid

def gerar_grade():
    print("\n[ETAPA 1] Gerando grade celular...")

    limite = load_limite("input/Limite_oiapoque_certo_dissolv.gpkg")
    grid = create_grid(limite, cell_size=2000)
    grid_clip = clip_grid(grid, limite)
    save_grid(grid_clip, "grade_2km_oiapoque.gpkg")
    plot_grid(limite, grid_clip)
    
    print("Grade gerada com sucesso.\n")
