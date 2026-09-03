# Seminar 2 — Nacrt seminarskog rada

> **Status nacrta: 3. septembar 2026.** Rad se piše paralelno sa finalnom serijom
> eksperimenata (Google Colab). Svi brojčani rezultati su **TBD (da se popune)**
> i eksplicitno su označeni. Tehničke činjenice u tekstu (postupci, hiperparametri,
> arhitekture, konfiguracije) odgovaraju finalnom kodu i **ne menjati bez provere**
> sa `Plan/paper-statement-guide.md` (delovi PART 1–13) — tamo je zabeležen razlog
> za svaku konfiguracionu odluku. Jezik i stil prate prethodni seminarski rad iz
> Mašinskog učenja („Mašinsko učenje u IDS“): gusta formalna proza, numerisane
> formule, slike/tabele/listinzi sa oznakama, reference u IEEE stilu [n].

---

## Naslovna strana (šablon)

- **Univerzitet u Novom Sadu, Prirodno-matematički fakultet**
- Departman za matematiku i informatiku
- **Seminar II (ID212)** — tema: *Konverzija tabelarnih podataka u slike i
  klasifikacija konvolucionim neuronskim mrežama*
- Student: Luka Momčilović 39d/25
- Komisija: prof. dr Miloš Radovanović, prof. dr Zoran Racković

---

# 1. Uvod

U velikom broju praktičnih domena podaci su organizovani **tabelarno** — svaki
primer opisan je vektorom atributa fiksne dužine, pri čemu atributi nemaju
unutrašnju prostornu strukturu. Klasični algoritmi mašinskog učenja (stabla
odlučivanja, ansambli, linearni modeli) prirodno su prilagođeni ovakvom zapisu,
dok konvolucione neuronske mreže (CNN), koje dominiraju obradom slika, zahtevaju
ulaz u obliku dvodimenzionalnog rastera. Poslednjih godina pojavio se zaseban
pravac istraživanja koji tabelarne vektore **preslikava u slike**, a zatim
klasifikaciju prepušta konvolucionim mrežama. Ideja vodilja je da se korelacije
između atributa mogu prostorno „spakovati“ tako da konvolucioni operatori, koji
su inherentno lokalni, uče interakcije između atributa bez eksplicitnog
ručnog konstruisanja karakteristika [1], [2], [3].

Motivacija za ovakav pristup nije trivijalna: preslikavanje u sliku je
**gubitna transformacija** — konačan broj atributa mora se rasporediti po
diskretnoj mreži piksela, pri čemu raspored direktno određuje šta konvolucioni
jezgra „vide“. Naivno rešenje (redom pakovati atribute u mrežu) ne koristi
informaciju o povezanosti atributa, dok naprednije metode pokušavaju da u prostor
slike prenesu **statističku strukturu podataka**: slični (korelisani) atributi
završavaju na susednim pozicijama, a različiti na udaljenim. U radu se upoređuje
pet reprezentativnih postupaka ove vrste — od naivnog pakovanja, preko
projekcionih metoda (DeepInsight, TINTO) do permutacionih metoda zasnovanih na
rangovima rastojanja (IGTD) — na tri heterogena skupa podataka i sa četiri
konvolucione arhitekture različitog kapaciteta, uključujući i modele sa
transfer učenjem (pretrenirane mreže na ImageNet-u).

Cilj rada je da se odgovori na sledeća istraživačka pitanja:

1. Da li izbor prostornog rasporeda atributa utiče na performanse CNN-a i koliko?
2. Da li transfer učenje sa slika prirodnog domena (ImageNet) pomaže ili odmаže
   na sintetičkim slikama tabelarnog porekla?
3. Kako se rezultati CNN + T2I odnose na klasične metode mašinskog učenja
   (Random Forest, XGBoost, MLP) na istim podacima?
4. Koji faktori — gustina slike, preklapanje atributa, kapacitet mreže, stopa
   učenja — objašnjavaju razlike u performansama?

Rad je organizovan na sledeći način. U **odeljku 2** dat je pregled stanja u
oblasti konverzije tabelarnih podataka u slike. **Odeljak 3** uvodi teorijske
osnove: postupke preslikavanja, konvolucione arhitekture i metrike. **Odeljak 4**
opisuje eksperimentalnu postavku: skupove podataka, protokol treninga i
hiperparametre. **Odeljak 5** prikazuje rezultate, a **odeljak 6** ih diskutuje.
**Odeljak 7** sadrži zaključak, a **odeljak 8** reference i priloge.

---

# 2. Pregled stanja

## 2.1 Konverzija tabelarnih podataka u slike kao istraživački pravac

Ideja da se tabelarni podaci konvertuju u slike nije nova — prvi sistematski
predlog u savremenoj formi dali su Sharma i saradnici u okviru metode
**DeepInsight** [1], gde se atributi projicuju u 2D ravan (t-SNE ili PCA),
a zatim mapiraju na pozicije piksela. Ubrzo zatim, Zhu i saradnici su u metodi
**IGTD** [2] predložili drugačiji princip: umesto projekcije, raspored atributa
se bira tako da se **rangovi rastojanja** između atributa što vernije poklope sa
rangovima rastojanja između pozicija u mreži piksela. Objavljen je i veći broj
varijanti i hibrida: **REFINED** (kombinacija karata udaljenosti i vizuelnog
sličnosti), **TINTO** [3], koji uvodi umekšavanje (blur) oko pozicija atributa
kako bi se stvorili kontinualni gradijenti pogodni za konvolucije, zatim
**S-IGTD** sa nadgledanom topologijom (rastojanja između atributa zasnovana na
klasnim sredinama) [4] i druge. Pojedini radovi eksperimentišu i sa alokacijom
različitog broja piksela po atributu srazmerno njihovoj informativnosti [5].

## 2.2 Empirijska evidencija i obim

Nedavna velika studija [6] (9 metoda, 24 skupa podataka) pokazuje da T2I pristupi
mogu da pariraju, a u pojedinim slučajevima i nadmaše klasične ansamble
(npr. Table2Image ostvaruje 0,879 naspram 0,868 tačnosti XGBoost-a), ali da
rezultati jako zavise od izbora metode i skupa. Sistematski pregledi literature
[S-LR materijali — SLR folder] ukazuju na nekoliko otvorenih pitanja: (i) fer
poređenje sa podešenim klasičnim modelima, (ii) uticaj gustine slike (udeo
nenultih piksela) na naučenost konvolucija, (iii) ponašanje pretreniranih mreža
na sintetičkim slikama koje se strukturno razlikuju od prirodnih, i (iv)
odsustvo standardizovanih dijagnostičkih mera kvaliteta preslikavanja (npr.
mera preklapanja atributa na istom pikselu).

## 2.3 Pozicija ovog rada

U odnosu na navedene studije, ovaj rad je užeg obima ali sistematičan: **isti
protokol** (iste podele podataka, isti hiperparametri, isti način evaluacije)
primenjen je na sve kombinacije postupaka i arhitektura, uz eksplicitne
kontrolne eksperimente (ablacije) kojima se proverava da li prostorna struktura
uopšte doprinosi rezultatu. Dodatno, rad transparentno beleži i dijagnostikuje
poznate zamke protokola — jednokratnu podelu skupa, nepodešene baselajne,
različite opsege vrednosti piksela između metoda i izbor stope učenja za
pretrenirane modele — što u literaturi često ostaje nedokumentovano.

---

# 3. Teorijske osnove

## 3.1 Tabelarni podaci i izazovi konverzije

Neka je $X \in \mathbb{R}^{N \times d}$ matrica podataka sa $N$ primera i
$d$ atributa, a $y$ vektor ciljnih klasa. Tabelarni zapis karakterišu:

- **odsustvo prostorne strukture** — redosled atributa je (uglavnom) proizvoljan;
- **mešoviti tipovi** — numerički i kategorički atributi (kategorički se najčešće
  kodiraju one-hot tehnikom, čime raste dimenzionalnost);
- **različite skale** — atributi se standardizuju (StandardScaler) na trening
  skupu, da transformacija ne zavisi od test podataka (sprečavanje curenja
  informacija);
- **klasni disbalans** — neravnomerna zastupljenost klasa zahteva poseban tretman
  u funkciji gubitka i izbor metrike.

Konverzija u sliku $I \in [0,1]^{h \times w}$ svodi se na dve odluke:
**(a)** gde u mreži piksela smestiti svaki atribut (mapiranje koordinata) i
**(b)** kako vrednost atributa preneti u intenzitet piksela. Prva odluka je
presudna i čini suštinsku razliku između metoda.

## 3.2 Postupci preslikavanja

### 3.2.1 Naivno mapiranje (osnovna linija)

Najjednostavniji postupak: vektor atributa se dopuni nulama do najbližeg
kvadrata $g = \lceil \sqrt{d} \rceil$ i preoblikuje u matricu $g \times g$,
redom (row-major). Ako je $g \neq h$ (ciljna veličina slike), matrica se
bikubičnom interpolacijom skalira na $h \times w$. Intenziteti se normalizuju
min-max skaliranjem na opseg $[0,1]$, pri čemu se minimum i maksimum računaju
**isključivo na trening skupu** (u ranijoj verziji koda normalizacija po
podskupu izazivala je različite opsege za trening/validaciju/test — curenje
informacija; ispravljeno čuvanjem statistika u fazi `fit`).

Prednosti: jednostavnost i determinizam. Mane: raspored prati ulazni redosled
atributa i ne vodi računa o njihovoj povezanosti; visoko korelisani atributi
mogu završiti na suprotnim krajevima slike, a dopunjena zona čini traku skoro
nultih vrednosti.

### 3.2.2 DeepInsight

DeepInsight [1] preslikava atribute u ravan primenom algoritma smanjenja
dimenzionalnosti (u ovom radu PCA, preko TINTOlib-a) nad **atributima kao
tačkama**: polazi se od matrice korelacija $C_{ij} = \mathrm{corr}(X_i, X_j)$,
odnosno rastojanja $1 - |C_{ij}|$, koja se ugrađuju u 2D koordinate
$p_i \in \mathbb{R}^2$. Koordinate se zatim skaliraju u opseg mreže piksela, a
svaki atribut dobija svoj piksel. Kako više atributa može da padne u istu ćeliju,
definiše se tzv. matrica gustine karakteristika (Feature Density Matrix — FDM);
pikseli sa više atributa čuvaju **srednju** vrednost (grupa „avg”), što je
gubitna kompresija. Zato se uvode dijagnostičke mere:

- $OF$ (Overlapped Features) — procenat atributa koji dele piksel sa nekim
  drugim atributom;
- $OP$ (Overlapped Pixels) — procenat aktivnih piksela sa više od jednog
  atributa.

### 3.2.3 IGTD

IGTD [2] ne koristi projekciju. Neka su $D_F$ i $D_P$ matrice rastojanja:
$D_F(i,j)$ — rastojanje između atributa $i$ i $j$ (Pearson-ova korelacija
pretvorena u rastojanje, u ovom radu $1 - |\rho|$), a $D_P(i,j)$ — Euklidovo
rastojanje između pozicija u mreži piksela. Za obe matrice računaju se matrice
**rangova** $R_F$ i $R_P$ (svako rastojanje dobija svoj rang među svim parovima).
Cilj je pronaći permutaciju atributa po pozicijama koja minimizuje Frobenijusovu
normu razlike rangova:

$$
\min_{\pi} \; \lVert R_F - R_P^{(\pi)} \rVert_F,
$$

što se rešava iterativnom zamenom (swap) pozicija dok se greška smanjuje
(maksimalno `max_step` iteracija, provera na svakih `val_step`). IGTD je po
konstrukciji **bez kolizija** ($OF = OP = 0$), a izlaz je gusti raster
vrednosti $[0, 255]$ koji se deli sa 255 radi uporedivosti sa ostalim metodama.

### 3.2.4 TINTO

TINTO [3] kombinuje projekcioni princip DeepInsight-a sa **umekšavanjem**:
nakon pozicioniranja atributa, svaka vrednost se „razmazuje“ na okolne piksele
Gausovim jezgrom (parametri: pojačanje amplifikacije $\approx \pi$, rastojanje
$2$, koraci $4$, broj ponavljanja $4$; sve vrednosti su podrazumevane
vrednosti TINTOlib biblioteke). Time nastaju kontinualni gradijenti koje
konvoluciona jezgra $3\times3$ mogu da „uhvate“ i na uređenoj prostornoj skali.
TINTO je jedina od korišćenih metoda koja **ne garantuje** opseg $[0,1]$ nakon
umekšavanja (vrhovi se kompresuju; izmereno je do ~0,30 na skupu Breast Cancer),
pa se njegov izlaz normalizuje statistikama prvog (trening) transforma — detalj
koji je presudan za ispravan rad pretreniranih mreža (videti 3.5).

### 3.2.5 S-IGTD (koncept; nije deo finalnog poređenja)

S-IGTD [4] modifikuje IGTD tako što se rastojanja atributa računaju iz
**klasnih sredina**: za atribut $j$, vektor $\mu_j = (\bar{x}_j^{(1)}, \dots,
\bar{x}_j^{(C)})$ srednjih vrednosti po klasama, pa $D_B(i,j) = 1 - |\mathrm{corr}
(\mu_i, \mu_j)|$. Time se diskriminativni atributi grupišu lokalno. Tokom
implementacije za ovaj rad utvrđeno je da ugradjena verzija (preko TINTOlib-a)
nije zaista koristila nadgledana rastojanja u optimizaciji rasporeda — njene
slike bile su identične IGTD slikama (provera: nula razlika u koordinatama i
pikselima). Umesto prikazivanja duplikata, **S-IGTD je izostavljena iz finalnog
poređenja**, a u ovom radu se pominje samo kao srodan pristup iz literature.

### 3.2.6 Dijagnostika kvaliteta preslikavanja

Pored $OF/OP$, prati se i **gustina slike** — udeo piksela sa vrednošću iznad
praga $0{,}01$ — jer konvolucije na gotovo praznoj slici uče dominantno
konstantan pozadinski signal. Gustina se meri po metodi i skupu (slika
`t2i_density_comparison.png`).

## 3.3 Konvolucione neuronske mreže

Konvolucioni sloj računa, za izlazni kanal $k$ i poziciju $(i,j)$,

$$
z_{i,j}^{(k)} = b_k + \sum_{c=1}^{C_{in}} \sum_{u=0}^{K-1}\sum_{v=0}^{K-1}
w_{k,c,u,v}\, x_{i+u,\,j+v}^{(c)},
$$

nakon čega sledi aktivacija $\mathrm{ReLU}(z) = \max(0, z)$, normalizacija po
minibatču (BatchNorm) i redukcija rezolucije (max pooling). Slojevi uče lokalne
obrasce; dublji slojevi kombinuju obrase u apstraktnije karakteristike.
Kapacitet i induktivna pristrasnost (lokalnost, translaciona invarijantnost)
čine CNN prirodnim kandidatom za obradu „sintetičkih slika“ T2I postupaka.

**ShallowCNN** — mreža iz ovog rada (~620K parametara): tri konvoluciona bloka
(32, 64, 128 kanala, jezgro $3\times3$, BatchNorm, ReLU, max pooling),
adaptivno prosečno grupisanje na $4\times4$ i klasifikator sa dva skrivena sloja
(256, Dropout 0,3). Odsustvo pretreniranosti i male dubine čine je „poštenom“
osnovom: ako napredni T2I raspored pomaže na ovoj mreži, efekat se ne može
pripisati kapacitetu ili transfer učenju.

**ResNet-18** — rezidualna arhitektura sa blokovima oblika
$y = \mathcal{F}(x, W) + x$, ~11M parametara. U radu se koristi u dve varijante:
pretrenirana na ImageNet-u (`pretrained=True`, ulaz 3 kanala sa ImageNet
normalizacijom) i od nule (`pretrained=False`, ulaz 1 kanal — siva slika).
Ova dva režima razdvajaju efekat **kapaciteta** od efekta **pretreniranosti**.

**ViT-Base/16** — Vision Transformer, ~86M parametara, patch veličine 16.
Slika $32\times32$ se bikubičnom interpolacijom skalira na $224\times224$
(196 tokena), svaki token je linearna projekcija $16\times16$ bloka uvećana
pozicionim enkodingom, a jezgro mreže je višeglavi self-attention:

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\mathsf{T}}}{\sqrt{d_k}}\right) V .
$$

## 3.4 Transfer learning i protokol treninga

Pretrenirane mreže (ResNet-18 i ViT na ImageNet-u) očekuju ulaz normalizovan
ImageNet statistikama (srednja vrednost $[0.485, 0.456, 0.406]$, standardno
odstupanje $[0.229, 0.224, 0.225]$). Siva slika se ponavljanjem kanala prevodi u
RGB pre normalizacije; prvi konvolucioni sloj zadržava originalne trokanalne
težine (ovo je bila kritična ispravka tokom razvoja — bez normalizacije
pretrenirani modeli su kolabirali na predviđanje jedne klase).

Funkcija gubitka je unakrsna entropija sa:
- **klasnim težinama** (inverzna učestanost klasa, sklearn `compute_class_weight`
  sa `'balanced'`) — kompenzacija disbalansa (Dry Bean ~6,6:1, Adult ~3,2:1);
- **label smoothing** $\varepsilon = 0{,}1$ — omekšavanje ciljnih raspodela radi
  manje samouverenih predikcija i bolje generalizacije; primenjeno uniformno na
  svim skupovima radi uporedivosti protokola.

Optimizator: Adam ($\beta_1 = 0.9$, $\beta_2 = 0.999$), sa L2 regularizacijom
(weight decay $10^{-4}$). Raspored stope učenja: ReduceLROnPlateau (faktor 0,5,
strpljenje 5 epocha po validacionom gubitku). Rano zaustavljanje sa strpljenjem
15 epoha po validacionom gubitku — dovoljno dugo da model „preživi“ posle
smanjenja stope (analiza: strpljenje 10 ostavljalo samo 5 epoha nakon smanjenja
stope). Maksimalno 50 epoha.

**Stopa učenja po arhitekturi.** U ranijim verzijama protokola svi modeli
trenirani su istom stopom $10^{-3}$. Pokazalo se da pretrenirani ViT pri
$10^{-3}$ **ne može da nauči ni trening skup** (gubitak zaključan na
$\log 2 \approx 0{,}698$ — odgovara uniformnim predikcijama), dok pri
$10^{-4}$ (uobičajeni opseg za fino podešavanje ViT-a) isti model uči normalno.
Zato finalni protokol koristi $lr_{ViT} = 10^{-4}$, a za sve ostale arhitekture
$10^{-3}$ — odluka zasnovana na kriterijumu „obučljivosti“, ne na pogađanju
performansi (detalji i dokazni eksperiment u PART 12 priručnika). **Unutar svake
arhitekture stopa je ista za sve T2I metode**, pa poređenje metoda ostaje
nekontaminirano.

Fino podešavanje pretreniranih modela rađeno je klasičnim treningom svih slojeva
sa jednom stopom učenja (prema gornjoj tabeli). Dvofazna strategija LP-FT
(linearno ispitivanje pa fino podešavanje) postoji u kodu i korišćena je **samo
u ablacionoj studiji** (odeljak 4.7) radi poređenja strategija — ne u glavnoj
tabeli.

## 3.5 Metrike performansi

Neka su $TP, FP, FN, TN$ brojevi tačno pozitivnih, lažno pozitivnih, lažno
negativnih i tačno negativnih predikcija. Definišu se:

$$
\mathrm{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}, \quad
\mathrm{Precision} = \frac{TP}{TP+FP}, \quad
\mathrm{Recall} = \frac{TP}{TP+FN}, \quad
F_1 = 2\,\frac{\mathrm{Precision}\cdot\mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}} .
$$

Za višeklasne probleme, makro verzije su aritmetičke sredine po klasama.
**Važna napomena o terminologiji u ovom radu:** za višeklasni skup Dry Bean
(7 klasa) prijavljuje se **makro-F1**. Za binarne skupove (Breast Cancer,
Adult Income) vrednost pod ključem `f1_macro` u kodu jeste zapravo F1
**pozitivne klase** (scikit `average='binary'`): za Breast Cancer pozitivna
klasa je benigna (većinska), za Adult Income pozitivna klasa je `>50K`
(manjinska). U tekstu i tabelama te vrednosti se dosledno označavaju kao
„F1 (pozitivna klasa)“, a ne kao makro-F1 — kako bi se izbegla netačna
interpretacija (vidi PART 13e priručnika).

Dopunske metrike: ROC-AUC (površina ispod ROC krive; za višeklasne probleme
makro one-vs-rest) i PR-AUC. Prikazuju se i matrice konfuzije i krive učenja.

---

# 4. Eksperimentalna postavka

## 4.1 Skupovi podataka i priprema

Odabrana su tri javno dostupna skupa koja pokrivaju različite režime:
binarni i višeklasni problem, mali i veliki broj primera, mali i veliki broj
atributa, izbalansiran i disbalansiran raspored klasa (Tabela 4.1).

**Tabela 4.1.** Skupovi podataka.
| Skup | Primeraka | Atributa (posle kodiranja) | Klasa | Napomena |
|---|---|---|---|---|
| Breast Cancer Wisconsin [7] | 569 | 30 | binarna | ~63:37 (benigni:maligni); svi numerički |
| Dry Bean [8] | ~13.611 | 16 | 7 klasa | najveći disbalans ~6,6:1; svi numerički |
| Adult Income [9] | ~48.842 | ~108 (one-hot) | binarna | 6 kategoričkih + 8 numeričkih; ~76:24 |

Priprema je zajednička za sve metode i modele:

1. Kategorički atributi Adult skupa kodiraju se **one-hot**; zvanična UCI
   podela train/test se spaja i ponovo deli (radi stratifikacije), bez curenja.
2. Redovi sa nedostajućim vrednostima uklanjaju se pre kodiranja.
3. Podaci se dele **stratifikovano** na trening/validaciju/test u odnosu
   80/10/10, fiksiranim semenom 42; **ista podela koristi se za sve metode**
   (CNN-e, baselajne i ablacije).
4. `StandardScaler` se fituje **isključivo na trening** delu i primenjuje na
   validaciju i test.

Curenje informacija sprečeno je na svim nivoima: skaliranje po treningu,
fiksna podela pre bilo kakvog preslikavanja i fituvanje T2I transformacija na
trening skupu (vrednosti min/max ili koordinatno mapiranje nikada ne zavise od
validacionih/test primera).

## 4.2 Dizajn eksperimenata

Eksperimentalna matrica obuhvata **4 T2I postupka × 3 skupa × 4 arhitekture =
48 CNN eksperimenata** plus **3 baselajna modela × 3 skupa = 9 eksperimenata**
(ukupno 57). (Petofti postupak, S-IGTD, prvobitno je planiran ali je izostavljen
— videti 3.2.5.) Rezultati se čuvaju po eksperimentu (JSON) sa svim metrikama,
istorijom treninga, trajanjem treninga i brojem epoha, a modeli (težine) se
čuvaju radi vizuelizacija (Grad-CAM).

## 4.3 Konfiguracija T2I metoda

Sve metode generišu **monohromatske slike 32×32**, normalizovane na $[0,1]$:
naivna metoda preko min-max statistika sa treninga, DeepInsight preko interne
MinMax skalacije TINTOlib-a, IGTD deljenjem izlaznog opsega $[0,255]$ sa 255,
a TINTO preko statistika prvog (trening) transforma. Fiksna veličina 32×32 za
sve skupove i metode održava **identičan platno** u svim ćelijama matrice —
jedina promenljiva između ćelija je raspored atributa (dijagnostika je pokazala
da povećanje platna na 128 ne pomaže: parametri umekšavanja TINTOlib-a ne
skaliraju se sa platnom). Generator slučajnih brojeva fiksiran je na 42 (globalno
i u TINTOlib konstruktorima), pa je generisanje slika determinističko po skupu.

**Tabela 4.2.** Konfiguracija T2I metoda (TINTOlib 1.3.1).
| Metoda | Prostorno mapiranje | Specifični parametri (podrazumevane vrednosti biblioteke) |
|---|---|---|
| Naive | redom, u mrežu | grid = ceil(sqrt(d)); bicubic resize; min-max sa treninga |
| DeepInsight | PCA → 2D → pikseli | MinMax interni; grupisanje kolizija: srednja vrednost |
| IGTD | rangovi rastojanja + swap | Pearson/Euklid, greška squared, max_step=1000, val_step=50 |
| TINTO | PCA → pikseli + blur | amplifikacija ≈ π, rastojanje 2, koraci 4, times 4, blur=True |

## 4.4 Arhitekture i hiperparametri

**Tabela 4.3.** Arhitekture i stopa učenja (svaka arhitektura konstantna za sve T2I metode).
| Arhitektura | Parametara | Ulaz | Stopa učenja | Napomena |
|---|---|---|---|---|
| ShallowCNN | ~620K | 1×32×32 | $10^{-3}$ | od nule, „poštena“ osnova |
| ResNet-18 (pretrenirani) | ~11M | 3×32×32 (ImageNet norm.) | $10^{-3}$ | transfer učenje |
| ResNet-18 (od nule) | ~11M | 1×32×32 | $10^{-3}$ | kontrola kapaciteta |
| ViT-Base/16 (pretrenirani) | ~86M | 3×224×224 (ImageNet norm., bicubic) | $10^{-4}$ | fino podešavanje (praksa za ViT) |

Zajednički hiperparametri za sve eksperimente: Adam, weight decay $10^{-4}$,
label smoothing 0,1, klasne težine `'balanced'`, rano zaustavljanje 15 epoha,
scheduler (faktor 0,5, strpljenje 5), maksimalno 50 epoha, batch 32, seme 42.
**Nikakvo podešavanje hiperparametara po metodi nije vršeno** — svesna odluka
radi fer poređenja, i ona važi i za baselajne modele. Listing 4.1 prikazuje
konfiguraciju treninga u kodu.

```
# run_all.py — zajednički train_config (izvod)
train_config = {
    'epochs': 50,
    'lr': ARCH_LR[cnn_arch],        # 1e-3, osim vit: 1e-4
    'weight_decay': 1e-4,
    'early_stopping_patience': 15,
    'label_smoothing': 0.1,
    'class_weights': class_weights, # sklearn 'balanced' na y_train
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}
```
Listing 4.1. Konfiguracija treninga CNN modela.

## 4.5 Baselajni (klasično ML)

Kao donja granica uporedivosti koriste se: **Random Forest**
(100 stabala, bez ograničenja dubine), **XGBoost** (100 stabala, dubina 6,
stopa 0,1) i **MLP** (slojevi 128-64, ReLU, rano zaustavljanje na 10%
validacije). Svi su trenirani **podrazumevanim/referentnim konfiguracijama bez
podešavanja**, isključivo na trening redu (istim redovima kao CNN — validacija
se koristi samo za rano zaustavljanje CNN modela i nikada nije deo treninga
baselajna). Namerno odsustvo podešavanja mora se imati u vidu pri čitanju
razlika CNN vs baselajn (ograničenje, odeljak 6).

## 4.6 Evaluacija i protokol validacije

Evaluacija se vrši na **jednom fiksiranom, stratifikovanom test skupu**
(10% svakog skupa), uz napomenu da se varijansa usled izbora podele ne meri
(nema unakrsne validacije — ograničenje). Za svaki eksperiment prijavljuju se:
Accuracy, Precision, Recall, F1 (sa terminologijom iz 3.5), ROC-AUC, PR-AUC,
matrica konfuzije i vreme treninga. Primarna metrika za rangiranje je F1.

## 4.7 Ablaciona studija

Da bi se dokazalo da **prostorna struktura** (a ne puka činjenica da je ulaz
„slika“) doprinosi rezultatu, sprovedene su tri ablacije (na
DeepInsight/ShallowCNN po skupu, uz preostale kombinacije gde je navedeno):

1. **Mešanje piksela (pixel shuffling):** ista permutacija piksela primenjena na
   sve test slike. Ako F1 značajno opadne (>0,02), CNN zaista koristi prostorni
   raspored; ako ne, transformacija se svodi na vektor.
2. **Redosled atributa (feature ordering):** porede se originalni, nasumični
   (fiksno seme), obrnuti i redosled sortiran po apsolutnoj korelaciji sa
   ciljnom klasom. Ako poredak utiče na F1, aranžman je relevantan.
3. **LP-FT vs direktno fino podešavanje** (pretrenirani ResNet-18): dvofazna
   strategija (10 epoha linearnog ispitivanja sa zamrznutim jezgrom + 40 epoha
   finog podešavanja, ukupno 50 — isti budžet kao glavna tabela) naspram
   direktnog treninga svih slojeva stopom iz `ARCH_LR` (ista kao u glavnoj
   tabeli, radi konzistentnosti sa rezultatima).

Granične vrednosti 0,02/0,01 služe samo za automatske oznake zaključka u
slikama; u tekstu se navode sirove razlike.

## 4.8 Infrastruktura i ponovljivost

Implementacija: Python, PyTorch, TINTOlib 1.3.1 (DeepInsight/IGTD/TINTO),
torchvision/timm (pretrenirani modeli), scikit-learn/XGBoost (baselajni).
Eksperimenti izvršeni na [GPU — TBD, npr. Colab T4], uz merenje vremena treninga
po eksperimentu. Rezultati se upisuju atomski (`.tmp` + rename), a nastavak
prekinutog izvršavanja prepoznaje kompletne JSON fajlove i pokreće samo
nedostajuće eksperimente; iskvareni fajlovi se automatski ponovo rade.

---

# 5. Rezultati

> Svi brojevi u ovom poglavlju su **TBD** — popunjavaju se nakon finalne serije
> eksperimenata (`run_all.py` na 48 CNN + 9 baselajna ćelija). Struktura i
> interpretativni okvir dati su unapred; ne unositi brojeve iz ranijih,
> nevalidnih verzija protokola (vidi PART 9g, 12a priručnika).

## 5.1 Pregled rezultata po skupovima

### 5.1.1 Breast Cancer Wisconsin

- Slika: `ch4_heatmap_breast_cancer.png` (T2I metode × arhitekture, F1).
- Diskusija: [TBD — koji raspored daje najviši F1; da li pretrenirani modeli
  pomažu na 398 trening primera; poređenje sa baselajnima iz
  `ch4_baseline_comparison.png`; ROC krive `ch4_roc_curves.png`].
- [TBD]: tabela 5.1 sa svim metrikama.

### 5.1.2 Dry Bean

- Višeklasni skup (7 klasa): prati se makro-F1, po-klasne performanse
  (`ch4_per_class_f1_dry_bean.png`) i matrica konfuzije.
- [TBD].

### 5.1.3 Adult Income

- Binarni disbalansiran skup (76:24); pozitivna klasa `>50K`.
- [TBD].

## 5.2 Poređenje sa baselajnima

Slika `ch4_baseline_comparison.png`; [TBD — da li neka CNN+T2I kombinacija
nadmašuje XGBoost i na kojoj margini; napomena da su baselajni nepodešeni].

## 5.3 Transfer učenje

- Uporediti `resnet` (pretrenirani) sa `resnet_scratch` (isti kapacitet, bez
  pretreniranosti) po metodi i skupu → efekat pretreniranosti.
- [TBD]; diskutovati u svetlu sintetičke prirode slika (odeljak 3.4) i
  činjenice da su pretrenirani modeli primali ImageNet normalizovan RGB ulaz.
- Za ViT: proveriti da li svaka ćelija pokazuje pad trening gubitka ispod
  ~0,7 u prvim epohama (potvrda ispravne stope učenja).

## 5.4 Dijagnostika slika i vizuelna analiza

- Primeri slika po metodi i skupu: `t2i_comparison_{dataset}.png`.
- Gustina: `t2i_density_comparison.png`; preklapanje: `ch4_overlap_diagnostics.png`.
- Grad-CAM (`ch4_gradcam_{dataset}.png`): [TBD — koji pikseli/regioni
  najviše doprinose klasifikaciji; interpretabilnost na ShallowCNN].

## 5.5 Vreme treninga

- `ch4_runtime_comparison.png`; [TBD — prosečno vreme po arhitekturi i metodi].

---

# 6. Diskusija

[Okvir za pisanje; sadržaj se finalizuje nakon rezultata.]

**Da li raspored atributa utiče na performanse?** [TBD] — osloniti se na glavnu
tabelu i ablacije: očekivani obrazac je da napredne metode (DeepInsight, IGTD,
TINTO) donose dobitak na ShallowCNN ako prostorna grupisanost zaista pomaže, i
da mešanje piksela obara F1 ako CNN koristi prostornu strukturu. Naivna metoda
služi kao donja granica; njena mana je redosledni raspored bez grupisanja
sličnih atributa (a ne gustina — gustina je uporediva sa ostalim metodama).

**Transfer učenje na sintetičkim slikama.** [TBD] — pretrenirani ResNet i ViT
uče opšte obrasce sa prirodnih slika; na T2I slikama raspodela se razlikuje.
Diskutovati odnos `resnet` vs `resnet_scratch`, kao i ViT rezultate uz napomenu
da je stopa učenja za ViT birana prema praksi finog podešavanja ($10^{-4}$), a
da su rane verzije protokola sa $10^{-3}$ kolabirale (gubitak zaključan na
$\log 2$) — ti rezultati nisu deo finalne tabele.

**Kapacitet vs metoda.** Rezultati odražavaju interakciju metode i kapaciteta;
poređenja metoda treba čitati **unutar** svake arhitekture.

**Ograničenja.**

1. Jedna stratifikovana podela po skupu — bez procene varijanse (bez unakrsne
   validacije i više pokretanja).
2. Mali broj primera (posebno Breast Cancer, 398 trening) uz duboke modele.
3. Baselajni (RF/XGB/MLP) nepodešeni — razlike CNN vs baselajn mogu se
   promeniti pod podešavanjem.
4. Fiksni hiperparametri za sve metode mogu pojedinoj metodi uskratiti njen
   optimalni režim.
5. TINTOlib biblioteka kao crna kutija za projekciju/optimizaciju (bit-level
   rezultati mogu zavisiti od verzije).
6. ViT ulaz je interpolisan sa 32×32 na 224×224 (bikubično) — interpolacioni
   artefakti su identični za sve metode, ali postoje.
7. S-IGTD nije evaluiran (videti 3.2.5) — rezultati se odnose na četiri metode.

---

# 7. Zaključak

[TBD — sažetak najvažnijih nalaza prema pitanjima iz Uvoda: (1) uloga rasporeda,
(2) transfer učenje, (3) odnos sa baselajnima, (4) faktori koji objašnjavaju
razlike.]

**Budući rad:** unakrsna validacija sa intervalima poverenja; podešavanje
hiperparametara (Optuna) za sve modele pod istim protokolom; adaptivno
određivanje veličine slike po broju atributa (protokol je u ovom radu fiksirao
32×32); selekcija atributa pre preslikavanja; prava nadgledana S-IGTD
implementacija; savremene arhitekture za tabelarne podatke kao gornja granica
(TabPFN i sl.); Grad-CAM na više arhitektura.

---

# 8. Reference i prilozi

## 8.1 Reference

> **Uputstvo:** konačan spisak urediti iz `Notebook/references.bib` (Zotero/
> BibTeX biblioteka pripremljena tokom prethodnih faza) i numerisati prema
> redosledu prvog citiranja. Slede ključne stavke (metapodatke autora potvrditi
> iz biblioteke):

1. A. Sharma i sar., *DeepInsight: metodologija pretvaranja netabelarnih podataka
   u slike za CNN arhitekture* — [1], 2019.
2. Y. Zhu i sar., *IGTD: pretvaranje tabelarnih podataka u slike za duboko
   učenje konvolucionim mrežama*, Scientific Reports, 2021 — [2].
3. TINTO/TINTOlib — konverzija tabelarnih podataka u slike sa umekšavanjem —
   [3], 2023–2025 (proveriti u .bib).
4. S-IGTD — nadgledana topologija (Zhang i sar., 2024) — [4].
5. Tabular Image / alokacija piksela po informativnosti — [5].
6. Liu i sar., *velika uporedna studija T2I metoda*, Information Fusion, 2026 — [6].
7. Breast Cancer Wisconsin (originalni skup; UCI ML Repository) — [7].
8. Dry Bean Dataset (Koklu i sar., UCI) — [8].
9. Adult Income (Kohavi i sar., UCI) — [9].
10. K. He i sar., *Deep Residual Learning*, CVPR 2016 (ResNet).
11. A. Dosovitskiy i sar., *An Image is Worth 16×16 Words*, ICLR 2021 (ViT).
12. D. Kingma i J. Ba, *Adam*, ICLR 2015.
13. C. Szegedy i sar., *Rethinking the Inception Architecture* (label smoothing), 2016.
14. A. Kumar i sar., *Fine-Tuning can Distort Pretrained Features* (LP-FT), ICLR 2022.
15. [Ostalo iz .bib — dopuniti.]

## 8.2 Prilozi

**Prilog A — Generisane slike i tabele** (mapiranje na fajlove):

| Slika | Fajl | Poglavlje |
|---|---|---|
| Primeri slika po metodi i skupu | `results/figures/t2i_comparison_{dataset}.png` | 3.2/5.4 |
| Gustina slika | `results/figures/t2i_density_comparison.png` | 5.4 |
| Glavni rezultati (heatmap) | `results/figures/ch4_heatmap_{dataset}.png` | 5.1 |
| Baselajni | `results/figures/ch4_baseline_comparison.png` | 5.2 |
| Per-class F1 (Dry Bean) | `results/figures/ch4_per_class_f1_dry_bean.png` | 5.1.2 |
| Krive učenja | `results/figures/ch4_training_curves_{dataset}.png` | 5.5 |
| Matrice konfuzije | `results/figures/ch4_confusion_matrices.png` | 5.1 |
| Ablacije (pixel shuffle, LP-FT) | `results/figures/ch4_ablation_*.png` | 5.x/4.7 |
| Gustina vs performanse | `results/figures/ch4_density_vs_performance.png` | 5.4 |
| ROC krive | `results/figures/ch4_roc_curves.png` | 5.1 |
| Vreme treninga | `results/figures/ch4_runtime_comparison.png` | 5.5 |
| Raspodela klasa | `results/figures/ch3_class_distribution.png` | 4.1 |
| Preklapanje (OF/OP) | `results/figures/ch4_overlap_diagnostics.png` | 3.2.6 |
| Grad-CAM | `results/figures/ch4_gradcam_{dataset}.png` | 5.4 |

**Prilog B — Kontrolna lista pri popunjavanju rezultata** (sve tvrdnje moraju
odgovarati kodu — svaka ima zabeležen razlog u `Plan/paper-statement-guide.md`):

- [ ] Popuniti TBD brojeve iz isključivo finalne serije (`results/`, ključ
      `lr` u JSON mora biti 1e-4 za vit, 1e-3 za ostale; bez `*_s_igtd*` fajlova).
- [ ] U tabelama F1 označiti: makro (Dry Bean) / pozitivna klasa (BC, Adult);
      nigde reč „makro“ za binarne skupove (PART 13e).
- [ ] Navesti da su baselajni nepodešeni i da CNN hiperparametri nisu podešavani
      po metodi (PART 13b, 1.3).
- [ ] Navesti jednu podelu 80/10/10 bez CV (PART 1.1) i da su svi modeli trenirani
      na istim trening redovima (PART 11.3).
- [ ] U metodologiji: ImageNet normalizacija + 3 kanala za pretrenirane,
      1 kanal za od-nule modele (PART 4.2/9a); opseg [0,1] po metodi (PART 10a).
- [ ] S-IGTD pomenuti samo kao srodan pristup (PART 13i, 8b).
- [ ] Ne koristiti tvrdnje iz PART 7 (LP-FT/CV/adaptivno) — nisu deo protokola.
- [ ] ViT: opisati stopu 1e-4 kao praksu finog podešavanja (PART 12).
- [ ] Reference prebaciti iz `Notebook/references.bib`.
