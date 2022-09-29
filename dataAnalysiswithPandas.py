import pandas as pd
from pyparsing import col
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df=sns.load_dataset("titanic")
df.head()

df.age.head()

type(df["age"].head())

df[["age"]].head()

type(df[["age"]].head())

df[["age","alive"]]

col_names=["age", "adult_male", "alive"]

df[col_names]

df["age2"]=df["age"]**2
df

df["age3"]=df["age"]/df["age2"]
df

df.drop(col_names, axis=1).head()
df

#sütunlarda içerisinde age ifadesi geçenlerin dışındakileri seçelim

df.loc[:,~df.columns.str.contains("age")].head()



#iloc integer based selection  0 dan 3 e kadar seçer

df.iloc[0:3]

# loc ise 0-1-2-3 3 dahil seçer
df.loc[0:3]



df.iloc[0:3,0:3]

#age değişkeninden 0-3 e kadar 3 dahil satırları getirir

df.loc[0:3,"age"]



#ilgili sütunlardaki belirli satırın bilgisini çekmek için

col_names=["age","embarked","alive"]

df.loc[0:3, col_names]



#koşullu seçim işlemleri

#verisetinde yaşı 50 den yüksek olanları seçelim

df[df["age"] > 50].head()

#verisetinde yaşı 50 den büyük olan kaç kişi var

df[df["age"]>50]["age"].count()

#verisetinde yaşı 50 den büyük olanların age ve class bilgilerini getirelim

df.loc[df["age"]>50,["age","class"]].head()


#verisetinde yaşı 50 den büyük olan erkeklerin age ve class bilgilerini seçelim


df.loc[(df["age"]>50) & (df["sex"]=="male"),["age","class"]].head()


#verisetinde yaşı 50 den büyük olan erkeklerin embark_town değişkenindeki Cherbourg limanını seçenleri getirelim

df.loc[(df["age"]>50) & (df["sex"]=="male")&(df["embark_town"]=="Cherbourg"),["age","class","embark_town"]].head()

#verisetinde yaşı 50 den büyük olan erkeklerin embark_town değişkenindeki Cherbourg ya da Southampton limanını seçenleri getirelim


df_new= df.loc[(df["age"]>50) &
       (df["sex"]=="male")&
       (df["embark_town"]=="Cherbourg")|(df["embark_town"]=="Southampton"),["age","class","embark_town"]]


df_new.head()

df_new["embark_town"].value_counts()


#Toplulaştırma ve Gruplama Aggregation and Grouping


df["age"].mean()


#cinsiyete göre yaş ortalaması

df.groupby("sex")["age"].mean()


df.groupby("sex").agg({"age":"mean"})

#cinsiyete göre yaş ortalaması ve toplamları

df.groupby("sex").agg({"age":["mean","sum"]})


#cinsiyete göre gemiye biniş limanının frekans bilgisini bulalım

df.groupby("sex").agg({"age":["mean","sum"], "embark_town":"count"})

#cinsiyete göre yaş bilgisinin ortalama ve toplamı ile survived bilgisini ortalamasını bulalım

df.groupby("sex").agg({"age":["mean","sum"], "survived":"mean"})


#cinsiyete ve limana biniş yerine göre kırılımın yaş ve survived bilgilerinin ortalamalarını bulalım
#ve cinsiyet bilgilerinin sayısını bulalım



df.groupby(["sex", "embark_town"]).agg({"age":"mean" , "survived":"mean", "sex":"count"})


#pivot table

#pivot table de ilk argüman kesişimde görmek istediğimiz argüman,
#ikinci değişken satır değişkeni
#üçüncü değişken sütun bilgisidir
#kesişim değerinin ortalamasını alır ön tanımlı değer ortalamadır


df.pivot_table("survived","sex", "embarked")


#keşisim değişkeninin standart sapma değerleri verilsin

df.pivot_table("survived","sex", "embarked", aggfunc="std")


#iki farklı sütun seçilebilir

df.pivot_table("survived","sex", ["embarked","class"])


#sayısal bir değişkeni kategorik bir değişkene çevirmek için cut fonksiyonu kullanılır
#Elinizdeki sayısal değişkeni tanımlayamıyorsanız o zaman çevrekliklerine göre değiştirmek için qcut fonksiyonu kullanılır

df["new_age"]= pd.cut(df["age"], [0,10,18,25,40,90])

df.head()

#yaş-cinsiyet kırılımda hayatta kalma oranlarını pivot table olarak alalım 

df.pivot_table("survived", "sex", "new_age")


#cinsiyet-yaş ve bilet sınıfı kırılıma göre hayatta kalma oranlarını pivot table de gösterelim

df.pivot_table("survived", "sex", ["new_age","class"])



####################################################################
#####   apply & lambda 
####################################################################




# apply satır ya da sutünlarda otomatik olarak fonksiyon tanımlamaya yarar


# lambda bir fonksiyon tanımlama şeklidir ancak sadece kullan at şeklinde fonksiyon kullanma imkanı sağlar


#yeni değişkenler oluşturalım

df["age_2"]=df["age"]*2

df["age_3"]=df["age"]*5

df.head()

#İçinde yaş ifadesi geçen değişkenlerini 10'a bölelim
#öncelikle dataframedeki sütunlarda age ifade olanları bulalım daha sonra apply komutu ile 
#içindeki değişkenlerde gezelim lambda ile her değişkeni 10 a bölelim

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()



#İçinde yaş ifadesi geçen değişkenlerinin her birinin ortalamasını kendisinden çıkartıp standart sapmasına bölelim


df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x-x.mean())/x.std()).head()


#Yukarıdaki işlemi fonksiyonlaştıralım


def standart_scaler(col_name):
    return (col_name-col_name.mean())/col_name.std()



df.loc[:,df.columns.str.contains("age")].apply(standart_scaler).head()


#Yukarıdaki işlemi kaydetmek için

df.loc[:,df.columns.str.contains("age")]=df.loc[:,df.columns.str.contains("age")].apply(standart_scaler).head()



df.head()

#############################################################################
# Birleştirme İşlemleri (Join)
#############################################################################


import numpy as np
import pandas as pd

m = np.random.randint(1,30,size=(5,3))

df1 = pd.DataFrame(m,columns=["var1","var2","var3"])


df2 = df1+99

# oluşturduğumuz df1 ve df2 dataframeleri alt alta birleştirmek istersek concat fonksiyonu kullanılır

pd.concat([df1,df2])

# Dikket edilecek nokta indeks bilgilerini düzeltmek gerektir
# Bundan dolayı ignore_index= True dememiz gerekir


pd.concat([df1,df2], ignore_index=True)



# merge fonksiyonu ile birleştirme işlemi

df1 = pd.DataFrame({'employees': ['john', 'dennis', 'mark', 'maria'],'group': ['accounting', 'engineering', 'engineering', 'hr']})

df2 = pd.DataFrame({'employees': ['mark', 'john', 'dennis', 'maria'], 'start_date': [2010, 2009, 2014, 2019]})

# yukarıda yer alan dataframelerde her çalışanın işe başlama tarihlerini elde etmeye çalışalım

# Öncelikle bu dataframeleri birleştirelim

pd.merge(df1,df2, on="employees")

# Yukarıda yer alan iki dataframe i birleştirelim ve bunun yanında müdür bilgilerini içeren
# df4 tablosunu da ekleyelim

df3=pd.merge(df1,df2)

df3.head()

df4 = pd.DataFrame({'group': ['accounting', 'engineering', 'hr'], 'manager': ['Murat', 'Uğur', 'Şükrü']})

pd.merge(df3, df4)

dict={"Parts": [10], "Berlin": [20]}

pd.DataFrame(dict)














