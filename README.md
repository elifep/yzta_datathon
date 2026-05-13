# YZTA 2026 Datathon - Bilişsel Performans Skoru Tahmini

Bu proje, YZTA 2026 Datathon kapsamında bireylerin uyku düzeni, fiziksel aktivite, stres seviyesi, yaşam tarzı ve günlük alışkanlık verileri kullanılarak **`bilissel_performans_skoru`** değerinin tahmin edilmesi amacıyla geliştirilmiştir.

Problem yapısı bir **regresyon problemidir** ve amaç, her birey için 0 ile 10 arasında sürekli bir bilişsel performans skoru tahmin etmektir.

---

## Public Leaderboard Skoru 

Bu repository, tarafımdan üretilen en iyi submission olan **CatBoost Seed Blend** çözümünü içermektedir.

| Submission File | Method | Public Score |
|---|---|---|
| `catboost_seed_blend_submission.csv` | CatBoost Seed Blend | **1.20536** |

Final submission dosyası:

```text
submissions/catboost_seed_blend_submission.csv
```
---

## Değerlendirme Metriği

Yarışmada kullanılan ana performans metriği RMSE (Root Mean Squared Error) değeridir.

RMSE, model tahminleri ile gerçek değerler arasındaki farkı ölçer. Skor ne kadar düşükse modelin tahminleri o kadar başarılı kabul edilir.

Amaç: En düşük RMSE skoruna ulaşmak.

## Proje Klasör Yapısı

```text
yzta_datathon/
│
├── data/
│   ├── train.csv
│   ├── test_x.csv
│   └── sample_submission.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── 01_eda.py
│   └── 02_catboost_seed_blend.py
│
├── submissions/
│   ├── catboost_seed_blend_submission.csv
│   ├── catboost_seed_42_submission.csv
│   ├── catboost_seed_2024_submission.csv
│   ├── catboost_seed_2025_submission.csv
│   ├── catboost_seed_3407_submission.csv
│   ├── catboost_seed_777_submission.csv
│   └── seed_blend_results.csv
│
├── requirements.txt
├── .gitignore
└── README.md

```

## Dosya Açıklamaları

`data/train.csv`

Modelin eğitildiği ana veri setidir.

Bu dosyada hem özellik sütunları hem de tahmin edilmesi gereken hedef değişken bulunur.

Hedef değişken:

`bilissel_performans_skoru`

Bu dosya modelin verilerden ilişki öğrenmesi için kullanılır.  


`data/test_x.csv`
Modelin tahmin üretmesi gereken test veri setidir.

Bu dosyada hedef değişken bulunmaz. Eğitim sonrası model bu dosyadaki her satır için bilissel_performans_skoru tahmini üretir.  

`data/sample_submission.csv`

Yarışma sisteminin beklediği submission formatını gösteren örnek dosyadır.

Beklenen format:
```bash
id,bilissel_performans_skoru
1,0
2,0
```
Model tahminleri bu formata uygun şekilde oluşturulur. 

```text
notebooks/eda.ipynb
```
Jupyter Notebook formatında hazırlanmış veri analizi dosyasıdır.

Bu dosya, veri keşfi ve görsel analizler için kullanılmak üzere projeye eklenmiştir.

Notebook içerisinde yapılabilecek analizler:

- Veri boyutu inceleme
- Eksik değer analizi
- Hedef değişken dağılımı
- Korelasyon analizi
- Kategorik değişken dağılımları
- Train-test kategori karşılaştırması

Not: Ana EDA süreci ayrıca src/01_eda.py dosyasında script formatında da yer almaktadır.

```text
src/01_eda.py
```
Bu dosya **Exploratory Data Analysis (EDA)** sürecini içerir.

EDA aşamasında veri seti modelleme öncesinde incelenmiş ve temel veri kalitesi kontrolleri yapılmıştır.

Bu dosyada yapılan işlemler:

- Train ve test veri boyutlarının kontrolü
- Sütun isimlerinin incelenmesi
- Hedef değişkenin temel istatistiklerinin çıkarılması
- Eksik değer analizi
- Sayısal ve kategorik değişkenlerin ayrıştırılması
- Duplicate kayıt kontrolü
- Kategorik değişken dağılımlarının incelenmesi
- IQR yöntemi ile outlier kontrolü
- Sayısal değişkenlerin hedef değişkenle korelasyonlarının incelenmesi
- Test setinde train setinde bulunmayan kategori olup olmadığının kontrol edilmesi
- Hedef değişkenin çarpıklık değerinin incelenmesi

EDA sonucunda hedef değişkenle en güçlü ilişkili sayısal değişkenlerin özellikle şu sütunlar olduğu gözlemlenmiştir:
```text
stres_skoru
rem_yuzdesi
gunluk_calisma_saati
derin_uyku_yuzdesi
gecelik_uyanma_sayisi
uykuya_dalma_suresi_dk
```
Bu analizler, modelleme aşamasında kullanılan feature engineering kararlarına temel oluşturmuştur.

```text
src/02_catboost_seed_blend.py
```
Bu dosyada sırasıyla şu işlemler yapılır:

1. `train.csv` ve `test_x.csv` dosyaları okunur.
2. Eksik değer temizliği yapılır.
3. Yeni özellikler üretilir.
4. Kategorik değişkenler için target mean encoding uygulanır.
5. Combo featurelar oluşturulur.
6. CatBoostRegressor modeli farklı random seed değerleriyle eğitilir.
7. Her seed için 5-Fold Cross Validation uygulanır.
8. Farklı seed tahminleri ortalanarak final submission oluşturulur.

Kullanılan seed değerleri:
```text
42, 2024, 2025, 3407, 777
```
Final üretilen submission dosyası:
```text
submissions/catboost_seed_blend_submission.csv
```

## Submissions Klasörü
### Seed Bazlı Submission Dosyaları

Aşağıdaki dosyalar farklı random seed değerleri ile eğitilen CatBoost modellerinin test tahminlerini içerir:

- `catboost_seed_42_submission.csv`
- `catboost_seed_2024_submission.csv`
- `catboost_seed_2025_submission.csv`
- `catboost_seed_3407_submission.csv`
- `catboost_seed_777_submission.csv`

Bu dosyalar final blend submission üretiminde kullanılmıştır.

### `seed_blend_results.csv`

Bu dosya her seed için lokal doğrulama sonuçlarını içerir.

| Seed | Mean RMSE | OOF RMSE |
|---|---:|---:|
| 2025 | 1.216774 | 1.216795 |
| 2024 | 1.217031 | 1.217064 |
| 777 | 1.217316 | 1.217355 |
| 3407 | 1.217360 | 1.217371 |
| 42 | 1.217523 | 1.217572 |

### `catboost_seed_blend_submission.csv` 

Final submission dosyasıdır.

Bu dosya, farklı random seed değerleriyle eğitilen CatBoost modellerinin test tahminlerinin ortalaması alınarak oluşturulmuştur.

Public leaderboard skoru:
```text
1.20536
```
## Kullanılan Çözüm Yöntemi

Bu projede ana model olarak **CatBoostRegressor** kullanılmıştır.

CatBoost tercih edilme nedenleri:

- Kategorik değişkenlerle güçlü çalışması
- Tabular veri problemlerinde başarılı sonuçlar vermesi
- Sayısal ve kategorik değişkenleri birlikte kullanabilmesi
- Eksik ve karma veri yapılarında iyi performans göstermesi
- Regresyon problemleri için güçlü bir baseline sağlaması

## Veri Ön İşleme

Modelleme öncesinde eksik değer temizliği uygulanmıştır.

## Kategorik Değişkenler

Kategorik sütunlardaki eksik değerler aşağıdaki değer ile doldurulmuştur:
```text
Bilinmiyor
```
Bu sayede eksik kategorik değerler de model için anlamlı bir kategori haline getirilmiştir.

## Sayısal Değişkenler

Sayısal sütunlardaki eksik değerler ilgili sütunun medyan değeri ile doldurulmuştur.

Median kullanılmasının sebebi, ortalamaya göre aykırı değerlerden daha az etkilenmesidir.

## Feature Engineering

Modelin daha anlamlı ilişkiler öğrenebilmesi için mevcut değişkenlerden yeni özellikler üretilmiştir.
```text
uyku_kalite_skoru
```
```bash
rem_yuzdesi + derin_uyku_yuzdesi
```
REM uykusu ve derin uyku oranlarını birlikte değerlendirerek genel uyku kalitesini temsil eder.
```text
uyku_bozulma_skoru
```

```bash
uykuya_dalma_suresi_dk + gecelik_uyanma_sayisi * 5
```
Uykuya dalma süresi ve gece uyanma sayısını birlikte değerlendirerek uyku bölünmesini temsil eder.

```text
dijital_yuk
```
```bash
uyku_oncesi_ekran_suresi_dk + log1p(uyku_oncesi_kafein_mg)
```
Uyku öncesi ekran süresi ve kafein miktarını birlikte ele alır.

Kafein değişkeninde log1p kullanılarak yüksek değerlerin etkisi yumuşatılmıştır.

```text
fiziksel_saglik
```

```bash
gunluk_adim_sayisi / vucut_kitle_indeksi
```

Günlük hareketlilik ile vücut kitle indeksi arasındaki ilişkiyi temsil eder.

```text
is_stres_yuku
```

```bash
gunluk_calisma_saati * stres_skoru
```

Çalışma süresi ile stres seviyesinin birleşik etkisini temsil eder.

```text
rem_stres_denge
```

```bash
rem_yuzdesi / (stres_skoru + 1)
```
REM uykusu ile stres seviyesi arasındaki dengeyi ifade eder.

Paydada +1 kullanılarak sıfıra bölme riski engellenmiştir.
```text
yas_kare
```
```bash
yas ** 2
```

Yaş değişkeninin doğrusal olmayan etkisini yakalamak amacıyla oluşturulmuştur.

## Target Mean Encoding

Bazı kategorik değişkenlerin hedef değişken ile güçlü ilişkiler taşıyabileceği düşünülmüştür. Bu nedenle belirli kategorik sütunlar için target mean encoding uygulanmıştır.

Bu yöntemde her kategori için eğitim verisindeki ortalama bilissel_performans_skoru değeri hesaplanmış ve yeni sayısal özellik olarak modele verilmiştir.

Target mean encoding uygulanan sütunlar:

- `meslek`
- `ruh_sagligi_durumu`
- `kronotip`
- `gun_tipi`

Bu yaklaşım, kategorik değişkenlerin hedef değişkenle olan ortalama ilişkisini modele sayısal bir sinyal olarak aktarmayı amaçlar.

## Combo Features

Kategorik değişkenlerin tek başına etkilerinin yanında, birlikte oluşturdukları davranış desenleri de modele aktarılmıştır.

Bu amaçla birleşik kategorik değişkenler oluşturulmuştur.

meslek_ruh_combo
```bash
meslek + ruh_sagligi_durumu
```
Meslek grubu ile ruh sağlığı durumunun birlikte etkisini yakalamak için oluşturulmuştur.

ultra_combo
```bash
meslek + ruh_sagligi_durumu + gun_tipi
```
Meslek, ruh sağlığı durumu ve gün tipi değişkenlerinin birlikte oluşturduğu davranış örüntüsünü temsil eder.

Bu combo değişkenler için de target mean encoding uygulanmıştır.

## Modelleme Süreci

Modelleme aşamasında **5-Fold Cross Validation** kullanılmıştır.

Bu yöntemde eğitim verisi 5 parçaya ayrılmıştır. Her adımda:

- 4 parça eğitim için kullanılır.
- 1 parça doğrulama için kullanılır.
- Her fold için RMSE değeri hesaplanır.
- Fold sonuçlarının ortalaması modelin genel başarısı hakkında fikir verir.

Bu işlem her random seed için ayrı ayrı tekrarlanmıştır.

Daha sonra her seed ile üretilen test tahminleri ortalanarak final submission oluşturulmuştur.

Bu yaklaşım, tek modelin rastgelelikten kaynaklanan varyansını azaltmayı ve daha stabil bir tahmin üretmeyi amaçlar.

## Seed Blend Yaklaşımı

Seed blend yaklaşımında aynı model mimarisi farklı random seed değerleriyle birden fazla kez eğitilmiştir.

Kullanılan seed değerleri:
```text
42, 2024, 2025, 3407, 777
```
Her seed için:

- 5-Fold Cross Validation yapılmıştır.
- Test seti için tahmin üretilmiştir.
- Seed bazlı tahmin dosyası kaydedilmiştir.

Final aşamada tüm seed tahminleri ortalanmıştır.

Final blend dosyası:
```bash
submissions/catboost_seed_blend_submission.csv
```
Public leaderboard skoru:
```text
1.20536
```
## Çalıştırma Adımları

Projeyi çalıştırmak için önce gerekli paketler yüklenmelidir.
```bash
pip install -r requirements.txt
```
EDA dosyasını çalıştırmak için:
```bash
python src/01_eda.py
```
Model eğitimi ve submission üretimi için:
```bash
python src/02_catboost_seed_blend.py
```
## Gereksinimler

requirements.txt dosyasında kullanılan temel kütüphaneler:

```txt
pandas
numpy
scikit-learn
catboost
matplotlib
```

## Deneysel Yaklaşım

Proje sürecinde farklı modelleme ve feature engineering yaklaşımları değerlendirilmiştir.

Denemeler sonucunda:

- Basit CatBoost modeli güvenilir bir baseline sağlamıştır.
- LightGBM alternatifi denenmiştir ancak bu veri setinde CatBoost kadar güçlü sonuç vermemiştir.
- Çok agresif target mean ve combo encoding denemeleri bazı lokal validasyon sonuçlarını iyileştirse de public leaderboard üzerinde güvenilir bulunmamıştır.
- Farklı random seed değerleriyle CatBoost modelleri eğitilerek seed blend yaklaşımı denenmiştir.
- Final çözüm olarak CatBoost Seed Blend yaklaşımı seçilmiştir.

Bu nedenle final dosyasında daha dengeli, açıklanabilir ve yarışma sonucuyla uyumlu bir pipeline kullanılmıştır.

## Sonuç

Bu projede bireylerin uyku, stres, fiziksel aktivite ve günlük yaşam değişkenleri kullanılarak bilişsel performans skoru tahmin edilmiştir.

Kullanılan temel teknikler:

- Veri keşfi
- Eksik değer temizliği
- Feature engineering
- Target mean encoding
- Combo feature generation
- CatBoost regression
- 5-Fold Cross Validation
- Seed blend
- Submission üretimi

Final çözüm, CatBoostRegressor tabanlı farklı seed modellerinin tahminlerini ortalayarak oluşturulmuştur.

Elde edilen public leaderboard skoru:
```text
1.20536
```