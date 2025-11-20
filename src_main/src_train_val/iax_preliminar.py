import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src_main.src_train_val.metrics import eval_metrics

# =========================
# Funciones IAX 
# =========================

def log_iax_metrics(model_name: str, model, X_test: pd.DataFrame, df_out: pd.DataFrame):
    """
    Calcula e imprime métricas IAX: Feature Importances/Coeficientes y Sesgo (WAPE por Segmento).
    """
    print(f"\n--- IAX Metrics para {model_name} ---")

    # 1. Interpretación: Feature Importances Globales / Coeficientes
    importances = None
    
    # Manejar modelos de árbol (RandomForest, XGBoost, LightGBM)
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=X_test.columns)
        importances_title = "Feature Importances"
    
    # Manejar modelos lineales (Ridge)
    elif hasattr(model, 'coef_'):
        # Usamos el valor absoluto del coeficiente para medir la importancia relativa al cambio.
        importances = pd.Series(model.coef_, index=X_test.columns).abs()
        importances_title = "Coeficientes (Valor Absoluto)"
        
    # Manejar el Baseline
    elif model_name == "Baseline_no_change":
        print("El Baseline no tiene Feature Importances significativas.")
        pass

    if importances is not None:
        try:
            importances = importances.sort_values(ascending=False).head(10)
            
            print(f"Top 10 {importances_title}:")
            print(importances.to_string(float_format=lambda x: f"{x:.4f}"))
            
            # Gráfico de Importancia (Opcional)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importances.values, y=importances.index)
            plt.title(f'{importances_title}: {model_name}')
            plt.tight_layout()
            # Guardar el gráfico para la Auditabilidad
            plt.savefig(f'./data/output/iax_plots/{model_name}_feature_importance.png')
            plt.close()
            print(f"Gráfico de {importances_title} guardado.")

        except Exception as e:
            print(f"[WARN] Error calculando/graficando importances para {model_name}: {e}")


    # 2. Auditabilidad: Análisis de Sesgo (WAPE por Segmento)
    if 'segmento' in df_out.columns:
        # Calcular todas las métricas por segmento (incluyendo WAPE y R2)
        bias_check = df_out.groupby("segmento").apply(lambda x: pd.Series(eval_metrics(x)))
        
        print("\nAnálisis de Sesgo: Métrica Clave WAPE Desglosada (Menor es mejor)")
        # Mostramos las métricas de sesgo más relevantes: WAPE (error relativo), R2 (poder predictivo) y n (tamaño)
        segment_metrics = bias_check.loc[:, ["WAPE", "R2", "n"]].sort_values(by="WAPE", ascending=False)
        print(segment_metrics.to_string(float_format=lambda x: f"{x:.4f}"))
        
        # Robustez: Identificar segmentos con alto sesgo (ej. WAPE > 10% más que el promedio global)
        global_metrics = eval_metrics(df_out)
        global_wape_calc = global_metrics["WAPE"]
        
        # Filtramos segmentos con WAPE significativamente peor
        high_bias_segments = segment_metrics[segment_metrics['WAPE'] > global_wape_calc * 1.10]
        
        if not high_bias_segments.empty:
            print("\nSegmentos con Alto Sesgo (WAPE más de 10% peor que el promedio):")
            print(high_bias_segments.to_string(float_format=lambda x: f"{x:.4f}"))
        
    else:
        print("[WARN] Columna 'segmento' no encontrada en df_out para análisis de sesgo.")

    print("-----------------------------------")
    
# Asegúrate de tener esta inicialización de carpeta
import os
if not os.path.exists("./data/output/iax_plots"):
    os.makedirs("./data/output/iax_plots")
