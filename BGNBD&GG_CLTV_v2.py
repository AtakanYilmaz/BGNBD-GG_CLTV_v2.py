##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

# 1. Verinin Hazırlanması (Data Preperation)
# 2. BG-NBD Modeli ile Expected Sales Forecasting
# 3. Gamma-Gamma Modeli ile Expected Average Profit
# 4. BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
# 6. Çalışmanın fonksiyonlaştırılması
# 7. Sonuçların Veri Tabanına Gönderilmesi

# CLTV = (Customer_Value / Churn_Rate) x Profit_margin.
# Customer_Value = Average_Order_Value * Purchase_Frequency

##############################################################
# 1. Verinin Hazırlanması (Data Preperation)
##############################################################

##########################
# Gerekli Kütüphane ve Fonksiyonlar
##########################


# pip install lifetimes
# pip install sqlalchemy
# conda install -c anaconda mysql-connector-python
# conda install -c conda-forge mysql


from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



# credentials.

creds = {'user': 'group_4',
         'passwd': 'miuul',
         'host': '34.79.73.237',
         'port': 3306,
         'db': 'group_4'}

connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'

conn = create_engine(connstr.format(**creds))

pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)


retail_2010_2011_df = pd.read_sql_query("select * from online_retail_2010_2011", conn)

df = retail_2010_2011_df.copy()

df.shape
df.info()
df.head()

#########################
# Veri Ön İşleme
#########################

df.describe().T
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C",na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011,12,11)
df["InvoiceDate"].max()

# Lifetime Veri Yapısının Hazırlanması

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (daha önce analiz gününe göre, burada kullanıcı özelinde)
# T: Analiz tarihinden ne kadar süre önce ilk satın alma yapılmış. Haftalık.
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary_value: satın alma başına ortalama kazanç


uk_cltv = df[df["Country"] == "United Kingdom"]

uk_cltv = uk_cltv.groupby("CustomerID", ).agg({"InvoiceDate" : [lambda date: (date.max() - date.min()).days,
                                                                lambda date: (today_date - date.min()).days],
                                                   "Invoice" : lambda num: num.nunique(),
                                                   "TotalPrice" : lambda price: price.sum()})

uk_cltv.columns = ["recency", "T", "frequency", "monetary"]

uk_cltv["monetary"] = uk_cltv["monetary"] / uk_cltv["frequency"]

uk_cltv = uk_cltv[uk_cltv["frequency"] > 1]

uk_cltv = uk_cltv[uk_cltv["monetary"] > 0]

uk_cltv["recency"] = uk_cltv["recency"] / 7

uk_cltv["T"] = uk_cltv["T"] / 7

# 2. BG-NBD Modelinin Kurulması

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(frequency=uk_cltv["frequency"],recency=uk_cltv["recency"],T=uk_cltv["T"])

uk_cltv["expected_purc_1_week"] = bgf.predict(1,
                                              uk_cltv['frequency'],
                                              uk_cltv['recency'],
                                              uk_cltv['T'])
uk_cltv["expected_purc_1_month"] = bgf.predict(4,
                                               uk_cltv["frequency"],
                                               uk_cltv["recency"],
                                               uk_cltv["T"])
# 3.GAMMA-GAMMA Modelinin Kurulması

ggf = GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(uk_cltv["frequency"], uk_cltv["T"])

ggf.conditional_expected_average_profit(frequency=uk_cltv["frequency"],
        monetary_value=uk_cltv["monetary"])

uk_cltv["expected_average_profit_clv"] = ggf.conditional_expected_average_profit(uk_cltv["frequency"],
                                                                                 uk_cltv["monetary"])

# 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.

cltv = ggf.customer_lifetime_value(bgf,
                                   uk_cltv["frequency"],
                                   uk_cltv["recency"],
                                   uk_cltv["T"],
                                   uk_cltv["monetary"],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv.head()
cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(50)

cltv_final = uk_cltv.merge(cltv, on="CustomerID", how="left")

cltv_final.sort_values(by="clv", ascending=False).head(10)

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(cltv_final[["clv"]])

cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])
cltv_final.sort_values(by="scaled_clv", ascending=False).head(10)

"""
 CustomerID  recency       T  frequency  monetary  expected_purc_1_week        clv  scaled_clv
2486  18102.0000  52.2857 52.5714         60 3859.7391                0.9653 91722.9517      1.0000
589   14096.0000  13.8571 14.5714         17 3163.5882                0.7230 54684.1854      0.5962
2184  17450.0000  51.2857 52.5714         46 2863.2749                0.7451 52478.8478      0.5721
2213  17511.0000  52.8571 53.4286         31 2933.9431                0.5074 36566.7539      0.3987
1804  16684.0000  50.4286 51.2857         28 2209.9691                0.4767 25848.8026      0.2818
406   13694.0000  52.7143 53.4286         50 1275.7005                0.7982 25064.5339      0.2733
587   14088.0000  44.5714 46.1429         13 3864.5546                0.2597 24430.3687      0.2663
1173  15311.0000  53.2857 53.4286         91  667.7791                1.4286 23515.3876      0.2564
133   13089.0000  52.2857 52.8571         97  606.3625                1.5320 22897.0428      0.2496
1485  16000.0000   0.0000  0.4286          3 2335.1200                0.4159 21536.9938      0.2348

"""

# GÖREV 2

cltv_one_month = ggf.customer_lifetime_value(bgf,
                                   uk_cltv["frequency"],
                                   uk_cltv["recency"],
                                   uk_cltv["T"],
                                   uk_cltv["monetary"],
                                   time=1,
                                   freq="W",
                                   discount_rate=0.01)

cltv_one_month.head(10)
"""
CustomerID
12747.0000    327.3742
12748.0000   2150.4701
12749.0000    567.6384
12820.0000    102.0335
12822.0000    245.0516
12823.0000    179.6022
12826.0000    126.0968
12827.0000    164.1896
12828.0000    189.0606
12829.0000      2.5160
"""
cltv_twelve_month = ggf.customer_lifetime_value(bgf,
                                   uk_cltv["frequency"],
                                   uk_cltv["recency"],
                                   uk_cltv["T"],
                                   uk_cltv["monetary"],
                                   time=12,
                                   freq="W",
                                   discount_rate=0.01)

cltv_twelve_month.head(10)
"""
CustomerID
12747.0000    3595.1196
12748.0000   23647.1362
12749.0000    6144.4972
12820.0000    1115.8438
12822.0000    2587.6800
12823.0000    1960.2485
12826.0000    1383.7348
12827.0000    1711.5544
12828.0000    2021.4631
12829.0000      27.5499
"""
# GOREV 3

cltv_final["segments"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D","C","B","A"])


# GOREV 4
cltv_final.head()

cltv_final["CustomerID"] = cltv_final["CustomerID"].astype(int)

cltv_final.to_sql("Atakan_Yilmaz", con=conn, if_exists="replace", index=False)

pd.read_sql_query("select * from Atakan_Yilmaz limit 10", conn)
