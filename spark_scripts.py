from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, lit, to_date, col, format_number

spark = SparkSession.builder.appName('petroleo').getOrCreate()
petroleo = spark.read.csv('preco_ipedata.csv', header=True)
dolar = spark.read.csv('preco_dolar.csv', header=True)
conflitos = spark.read.csv('conflitos.csv', header=True).drop("_c0", "event_id_cnty")

df = spark.read.csv('em_data_disasters.csv', header=True)
df_tratado = df.select(
    df["Year"].alias("ano"),
    df["Start Month"].alias("mes_inicio"),
    df["Start Day"].alias("dia_inicio"),
    df["End Year"].alias("ano_termino"),
    df["End Month"].alias("mes_termino"),
    df["End Day"].alias("dia_termino"),
    df["Country"].alias("pais"),
    df["Disaster Type"].alias("tipo_desastre"),
    df["Total Deaths"].alias("total_mortes"),
    df["No Injured"].alias("numero_feridos"),
    df["No Affected"].alias("numero_afetados"),
    df["No Homeless"].alias("numero_desabrigados"),
    df["Total Damages ('000 US$)"].alias("danos_totais_mil_usd")
)

df_tratado = df_tratado.withColumn(
    "data_inicio",
    to_date(
        concat(
            df_tratado["ano"],
            lit("-"),
            df_tratado["mes_inicio"],
            lit("-"),
            df_tratado["dia_inicio"]
        ),
        "yyyy-M-d"
    )
)

df_tratado = df_tratado.withColumn(
    "data_termino",
    to_date(
        concat(
            df_tratado["ano_termino"],
            lit("-"),
            df_tratado["mes_termino"],
            lit("-"),
            df_tratado["dia_termino"]
        ),
        "yyyy-M-d"
    )
)
desastres = df_tratado.filter(
    col("data_inicio").isNotNull() & col("data_termino").isNotNull()
)

dj_nasdaq = spark.read.csv('dowjones_nasdaq.csv', header=True)


dj_nasdaq = dj_nasdaq.withColumn("data", to_date(col("data"), "yyyy-MM-dd"))
dj_nasdaq = dj_nasdaq.withColumn("valor_petroleo", format_number(col("valor_petroleo").cast("float"), 4))
dj_nasdaq = dj_nasdaq.withColumn("valor_dow_jones", format_number(col("valor_dow_jones").cast("float"), 4))
dj_nasdaq = dj_nasdaq.withColumn("valor_nasdaq", format_number(col("valor_nasdaq").cast("float"), 4))

desastres = desastres.withColumn("data_inicio", col("data_inicio").cast("date"))
desastres = desastres.withColumn("data_termino", col("data_termino").cast("date"))

conflitos = conflitos.withColumn("data", to_date(col("data"), "yyyy-MM-dd"))
conflitos = conflitos.withColumn("fatalidades", col("fatalidades").cast("int"))

dolar = dolar.withColumn("data", to_date(col("data"), "yyyy-MM-dd"))
dolar = dolar.withColumn("preco_dolar", format_number(col("preco").cast("float"), 4))

petroleo = petroleo.withColumn("data", to_date(col("data"), "yyyy-MM-dd"))
petroleo = petroleo.withColumn("preco_petroleo", format_number(col("preco").cast("float"), 4))


combinado = dj_nasdaq.join(petroleo, "data", "outer") \
                     .join(dolar, "data", "outer") \
                     .join(conflitos, "data", "outer")


combinado = combinado.join(desastres, combinado.data >= desastres.data_inicio, "outer")


combinado = combinado.select(
    col("data").alias("data"),
    col("preco_petroleo").alias("preco_petroleo"),
    col("valor_dow_jones").alias("indice_dow_jones"),
    col("valor_nasdaq").alias("indice_nasdaq"),
    col("preco_dolar").alias("preco_dolar"),
    col("evento").alias("tipo_conflito"),
    col("fatalidades").alias("fatalidades_conflito"),
    col("tipo_desastre").alias("tipo_desastre"),
    col("total_mortes").alias("mortes_desastre"),
    col("numero_afetados").alias("afetados_desastre"),
    col("danos_totais_mil_usd").alias("danos_desastre_usd")
)

combinado.filter("data > '2019-12-31'").coalesce(1).write.parquet("/content/drive/MyDrive/Data/resultados")