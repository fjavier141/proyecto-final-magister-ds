/*
    created at: 2024-06-10
    updated at: 2024-06-10

    descripcion : Consulta agregada de ventas por tipo y mix, pivotada y unida con tablas dimensionales.
*/

WITH
  ventas_agregadas_tipo_mix AS (
    SELECT
      id_cliente,
      id_periodo,
      CASE
        WHEN id_categoria = 1 THEN 'Alcholicos'
        ELSE 'Analcoholicos'
      END AS Tipo_Producto,
      tipo_mix,
      COUNT(1) AS num_transacciones,
      SUM(liq_um) AS ventas
    FROM
      stg.venta_historica
    WHERE
      liq_um > 0 
    GROUP BY
      id_cliente,
      id_periodo,
      CASE
        WHEN id_categoria = 1 THEN 'Alcholicos'
        ELSE 'Analcoholicos'
      END,
      tipo_mix
  ),
  ventas_pivotadas_mes AS (
    SELECT
      id_cliente,
      id_periodo,
      -- Ventas y transacciones totales (agregando todos los tipos/mix)
      SUM(ventas) AS venta_mensual_total,
      SUM(num_transacciones) AS transacciones_mes_total,
      -- Pivot de 'Tipo_Producto'
      SUM(
        CASE
          WHEN Tipo_Producto = 'Alcholicos' THEN ventas
          ELSE 0
        END
      ) AS ventas_alcoholicos,
      SUM(
        CASE
          WHEN Tipo_Producto = 'Analcoholicos' THEN ventas
          ELSE 0
        END
      ) AS ventas_analcoholicos,
      -- Conteo de 'tipo_mix' distintos este mes (una medida de variedad)
      COUNT(DISTINCT tipo_mix) AS variedad_mix_mes      
    FROM
      ventas_agregadas_tipo_mix
    GROUP BY
      id_cliente,
      id_periodo
  )
--- Consulta Final: Unir la data mensual PIVOTADA con todas las tablas
SELECT
  -- Llaves principales
  v.id_cliente,
  v.id_periodo,
  -- Métricas de venta (del Paso 2)
  v.venta_mensual_total,
  v.transacciones_mes_total,
  v.ventas_alcoholicos,
  v.ventas_analcoholicos,
  v.variedad_mix_mes,
  -- Proporción de alcohol (un feature muy útil para segmentar)
  (v.ventas_alcoholicos / NULLIF(v.venta_mensual_total, 0)) AS pct_venta_alcohol,
  -- Info de Clientes (dimensional)
  c.id_barrio,
  c.id_comuna,
  c.segmento,
  c.canal,
  c.descr_flag_patente,
  -- Info de Segmentos (dimensional)
  s.descripcion_segmento,
  s.canal,
  -- Info de Distritos/Barrios (dimensional)
  d.indice_gse,
  d.n_habitantes,
  d.n_ptos_interes,
  d.superficie_km2,
  d.densidad_hab,
  -- Info Macroeconómica (dimensional, unida por mes)
  m.uf,
  m.dolar,
  m.ipc,
  m.imacec,
  m.tpm,
  m.tasa_desempleo
FROM
  ventas_pivotadas_mes AS v
  LEFT JOIN stg.base_clientes AS c ON v.id_cliente = c.id_cliente
  LEFT JOIN stg.info_distritos AS d ON c.id_barrio = d.id_barrio
  LEFT JOIN stg.info_segmentos AS s ON c.segmento = s.segmento
  --- Los datos macros se unen con un mes anterior, ya que se publican el mes siguiente:
  LEFT JOIN stg.datos_macro AS m ON v.id_periodo = m.id_periodo -1  
ORDER BY
  v.id_cliente,
  v.id_periodo;
