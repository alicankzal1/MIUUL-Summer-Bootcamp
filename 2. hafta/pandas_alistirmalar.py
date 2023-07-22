import pandas as pd
import seaborn as sns
#Görev 1:  Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
df = sns.load_dataset("titanic")
#Görev 2:  Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
print(df["sex"].value_counts())
#Görev3:  Her bir sutuna ait unique değerlerin sayısını bulunuz.
print(df.nunique())
#Görev4:  pclass değişkeninin unique değerlerinin sayısını bulunuz.
print(df["pclass"].nunique())
#Görev5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
print(df[["pclass", "parch"]].nunique())
#Görev6:  embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.
print(df["embarked"].dtype)
df["embarked"] = df["embarked"].astype("category")
print(df["embarked"].dtype)
#Görev7:  embarked değeri C olanların tüm bilgelerini gösteriniz.,
print(df[df["embarked"] == "C"])
#Görev8:  embarked değeri S olmayanların tüm bilgilerini gösteriniz.
print(df[df["embarked"] != "S"])
#Görev9:   Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
print(df[(df["age"] < 30) & (df["sex"] == "female")])
#Görev10:  Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
print(df[(df["fare"] > 500) | (df["age"] > 70)])
#Görev 11:  Her bir değişkendeki boş değerlerin toplamını bulunuz.
print(df.isna().sum())
#Görev 12:  who değişkenini dataframe’den çıkarınız.
df.drop("who", axis= 1, inplace= True)
print("who" in df.columns)
#Görev13:  deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri(mode) ile doldurunuz.
df.deck.fillna(df.deck.mode()[0], inplace= True)
print(df.deck)
#Görev14:  age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz.
df.age.fillna(df.age.median(), inplace= True)
#Görev15:  survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında 
# sum, count, mean değerlerinibulunuz
print(df.groupby(["pclass", "sex"]).agg({"survived" : ["sum", "count", "mean"]}))
#Görev16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 verecek bir
# fonksiyon yazın. Yazdığınız fonksiyonu kullanarak titanik verisetinde age_flag
# adında bir değişken oluşturunuz. (apply ve lambda yapılarını kullanınız)
df['age_flag'] = df['age'].apply(lambda x: 1 if x < 30 else 0)
print(df.head())
#Görev17:  Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
tips_df = sns.load_dataset("tips")
print(tips_df.head())
#Görev18:  Time değişkeninin kategorilerine(Dinner, Lunch)
# göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
print(tips_df.groupby("time").agg({"total_bill" : ["sum", "min", "max", "mean"]}))
#Görev19:  Günlere ve time göre total_bill değerlerinin toplamını, min, max ve
# ortalamasını bulunuz.
print(tips_df.groupby(["day", "time"]).agg({"total_bill" : ["sum", "min", "max", "mean"]}))
#Görev 20:  Lunch zamanına ve kadın müşterilere ait total_bill ve tip
#değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
tips_df[(tips_df.time == "Lunch") & (tips_df.sex == "Female")].groupby("day").agg({"total_bill" : ["sum", "min", "max", "mean"],
                                                                                   "tip" : ["sum", "min", "max", "mean"]})
#Görev 21:size'i 3'ten küçük, total_bill'i 10'dan
#büyük olan siparişlerin ortalaması nedir? (loc kullanınız)
print(tips_df.loc[(tips_df['size'] < 3) & (tips_df['total_bill'] > 10), :].mean())
#Görev22:  total_bill_tip_sum adında yeni bir değişken oluşturunuz.
#Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
tips_df["total_bill_tip_sum"] = tips_df["total_bill"] + tips_df["tip"]
#Görev23:  total_bill_tip_sum değişkenine göre büyükten küçüğe
# sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
new_df = tips_df.sort_values("total_bill_tip_sum", ascending= False).head(30)
print(new_df)
