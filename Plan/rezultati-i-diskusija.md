# 6. Rezultati i diskusija — gotov tekst za Word

> **Kako se koristi ovaj fajl (ne kopirati u rad).**
> Sve od naslova „6. Rezultati i diskusija“ do kraja poglavlja „7. Zaključak“ je
> tekst koji ide u Word. Numeracija slika i tabela nastavlja brojanje iz poglavlja 5
> (tamo su već `Slika 5.1–5.5` i `Tabela 5.1–5.3`), pa su ovde redom
> `Slika 6.1–6.26` i `Tabela 6.1–6.14`. Uz svaku sliku stoji ime fajla u
> `results/figures/` — toliko je da znaš šta da ubaciš; ime fajla **ne treba** da
> ostane u konačnom tekstu. Deo „Napomene za tebe“ na kraju fajla **ne ide** u rad:
> tu su proverene vrednosti, šest mesta na kojima se ovaj tekst razlikuje od
> `Plan/seminar2-rad-nacrt.md`, spisak slika po važnosti i uputstvo za novi
> Zaključak.

---

# 6. Rezultati i diskusija

Svi rezultati u ovom poglavlju potiču iz jedne finalne serije eksperimenata
pokrenute kodom iz repozitorijuma: 4 postupka preslikavanja (naivno pakovanje,
TINTO, DeepInsight, IGTD) × 3 arhitekture (ShallowCNN, pretrenirani ResNet-18 i
ResNet-18 treniran od nule) × 3 skupa podataka = **36 konvolucionih ćelija**, uz
**9 klasičnih baselajna** (Random Forest, XGBoost, MLP) i **tri ablacione studije**
(mešanje piksela, raspored atributa, LP-FT). Protokol je jedinstven za sve ćelije:
jedna stratifikovana podela 70 : 10 : 20, fiksni hiperparametri, seed 42 za podelu
i za trening. Metrike su izračunate na test skupu, koji nijednom odlukom nije
korišćen: za binarne skupove (Breast Cancer, Adult Income) prijavljuje se **F1
pozitivne klase** — *benign* za Breast Cancer i `>50K` za Adult Income — a za
višeklasni Dry Bean **makro-F1** preko svih 7 klasa; uz to se prate i tačnost,
makro-F1 preko svih klasa i balansirana tačnost.

Rezultati su organizovani tako da se za svaki skup prvo prikazuje kompletna mreža
ćelija (toplotna mapa i tabela), pa poređenje sa baselajnima, pa ablacije i, na
kraju, provera stabilnosti. **Jedna napomena o čitanju:** glavna tabela je po jedan
run po ćeliji, a jedina podela je fiksna, pa razlike manje od ~1 procentnog poena
ne treba čitati kao rangiranje metoda; ćelije koje nose tvrdnje zato su dodatno
ponovljene kroz pet semena (§6.9), gde su i date kao srednja vrednost ± standardna
devijacija.

## 6.1 Breast Cancer Wisconsin

Skup sadrži 569 uzoraka, 30 numeričkih atributa i dve klase (nakon kodiranja
podela trening/test je 398 : 57 : 114, pri čemu je pozitivna klasa *benign* sa 72
od 114 test primera). Sa 30 atributa na slici 32×32 svaki atribut dobija sopstveni
piksel, pa je ovo najlakši slučaj za konverziju — nema gužve i nema gubitka
informacije u samom raspoređivanju.

**Tabela 6.1.** Breast Cancer — F1 pozitivne klase (u zagradi tačnost), u %.

| T2I metoda | ShallowCNN | ResNet-18 (pretreniran) | ResNet-18 (od nule) |
|---|---|---|---|
| Naivno | 95,77 (94,74) | 95,77 (94,74) | 95,04 (93,86) |
| TINTO | 96,45 (95,61) | 97,18 (96,49) | 97,22 (96,49) |
| DeepInsight | 96,45 (95,61) | 97,22 (96,49) | 96,50 (95,61) |
| IGTD | 96,60 (95,61) | 96,00 (94,74) | 95,83 (94,74) |

*Baselajni (F1 / tačnost):* RF 95,10 (93,86), XGBoost 95,83 (94,74), MLP 94,52 (92,98).

**Slika 6.1.** Toplotna mapa F1 pozitivne klase po ćelijama; vrste su T2I metode,
kolone arhitekture (`ch4_heatmap_breast_cancer.png`). Slika omogućava da se na
jednom mestu vidi da su sve vrednosti u uskom opsegu: tamnije nijanse (TINTO i
DeepInsight) stoje uz svetlije (naivno i IGTD), a najsvetlija ćelija je IGTD sa
ResNet-18 od nule.

Sve konvolucione ćelije leže u opsegu **95,04–97,22 %** F1, a tačnost u
93,86–96,49 %; najbolja je TINTO odnosno DeepInsight sa pretreniranim ResNet-18 i
ResNet-18 od nule (97,22 %). Raspon je dakle 2,18 pp. Nasuprot očekivanju da
konverzija u sliku „pokvari“ najlakši skup, ovo je jedini skup na kome konvoluciona
mreža **prelazi najbolji klasični baselajn** — 97,22 % prema 95,83 % kod XGBoost-a,
razlika +1,39 pp. Rang metoda je pri tome zavisan od arhitekture (DeepInsight daje
97,22 % sa pretreniranim, a 96,50 % sa ResNet-18 od nule), a pomaci su reda 1 pp,
tj. na granici šuma, pa se čitaju deskriptivno.

**Slika 6.2.** Krive učenja po ćelijama za Breast Cancer — gubitak na treningu
(isprekidana linija) i validaciji (puna linija) po epohi, uz tačnost na validaciji
(`ch4_training_curves_breast_cancer.png`). Sa slike se vidi da sve ćelije dostižu
validacioni minimum u prvih desetak epoha i da dalje ne divergiraju. Oscilovanje
validacionog gubitka postoji i ovde, ali ne menja ishod: najbolji checkpoint je
izabran u epohi u kojoj je model već naučen. (Ovaj skup je prikazan kao
reprezentativan; isti paneli postoje i za ostale skupove.)

**Slika 6.3.** Matrice konfuzije najbolje konvolucione mreže po skupu; na panelu za
Breast Cancer vidi se raspodela grešaka (`ch4_confusion_matrices.png`). Najbolja
ćelija (TINTO + ResNet-18 od nule) pogrešno klasifikuje svega **2 maligna uzorka i
2 benigna** (matrica [[40, 2], [2, 70]]), dakle po dve greške u svakoj klasi — tip
greške je izbalansiran i ne ukazuje na pristrasnost prema češćoj klasi. (Ista slika
sadrži i panele za Dry Bean i Adult Income, pa se jedna slika ubacuje samo jednom —
vidi §6.2 i §6.3.)

Zaključak za ovaj skup: kada broj atributa ne prelazi površinu slike, način
preslikavanja gotovo da ne menja ishod, a razlika između naivnog pakovanja i
projekcionih metoda (DeepInsight, TINTO) nije izmerljiva na jednom runu.

## 6.2 Dry Bean

Dry Bean ima 16 atributa, 7 klasa i 13 611 uzoraka (9527 : 1361 : 2723). Pošto su
tri klase (DERMASON, SEKER, SIRA) međusobno vizuelno najsličnije i najbrojnije,
pažnja se pored makro-F1 poklanja i po-klasnim rezultatima. Sa 16 atributa na slici
32×32 nema gotovo nikakvih kolizija (jedino TINTO i DeepInsight spajaju po dva
atributa u jedan piksel, §6.6), pa je i ovde prostor rasporeda slobodan.

**Tabela 6.2.** Dry Bean — makro-F1 (u zagradi tačnost), u %.

| T2I metoda | ShallowCNN | ResNet-18 (pretreniran) | ResNet-18 (od nule) |
|---|---|---|---|
| Naivno | 93,99 (92,80) | 93,15 (91,92) | 93,18 (92,03) |
| TINTO | 93,65 (92,58) | 93,60 (92,51) | 93,80 (92,66) |
| DeepInsight | 93,53 (92,32) | 93,79 (92,69) | 93,28 (92,18) |
| IGTD | 92,28 (90,93) | 90,33 (88,73) | 91,30 (89,79) |

*Baselajni (makro-F1 / tačnost):* RF 93,10 (91,81), XGBoost 93,91 (92,69), MLP 93,46 (92,40).

**Slika 6.4.** Toplotna mapa makro-F1 po ćelijama (`ch4_heatmap_dry_bean.png`). Za
razliku od panela za Breast Cancer, ovde se vidi vertikalni gradijent: kolona IGTD
je dosledno svetlija (najslabija), dok su ostale tri metode međusobno blizu i
redosled im se menja od arhitekture do arhitekture.

Sve metode osim IGTD-a dostižu **93,15–93,99 %** makro-F1, što odgovara
literaturnom plafonu za ovaj skup (tačnost 91,92–92,80 %); najbolja ćelija je
naivno pakovanje sa ShallowCNN-om (93,99 %). **IGTD je dosledno najslabiji**
(90,33–92,28 %), što se poklapa sa njegovom dijagnostikom rasporeda (§6.6):
IGTD nizove atributa raspoređuje u jedan red (traku) bez kolizija, pa mreža nema
prostornu strukturu koju bi mogla da iskoristi; zaostatak prema najboljoj metodi
dostiže **3,66 pp**.

**Tabela 6.3.** Dry Bean — po-klasni F1 dve suprotne ćelije (naivno + ShallowCNN,
najbolja; IGTD + pretrenirani ResNet-18, najslabija).

| Klasa | Udeo u testu | F1 (naivno + ShallowCNN) | F1 (IGTD + ResNet-18 PT) |
|---|---|---|---|
| BARBUNYA | 265 | 0,93 | 0,90 |
| BOMBAY | 104 | 1,00 | 1,00 |
| CALI | 326 | 0,94 | 0,92 |
| DERMASON | 709 | 0,92 | 0,90 |
| HOROZ | 386 | 0,96 | 0,88 |
| SEKER | 406 | 0,95 | 0,94 |
| SIRA | 527 | 0,87 | 0,78 |
| **makro** | 2723 | **0,94** | **0,90** |

**Slika 6.5.** Po-klasni F1 po T2I metodi za Dry Bean
(`ch4_per_class_f1_dry_bean.png`): grupe stubića po klasi, u okviru grupe po jedna
metoda. Sa slike je najvažnije da najmanja klasa BOMBAY (3,8 % testa) ne kolabira
ni u jednoj metodi (F1 = 1,00), dok se sve greške koncentrišu na klasu **SIRA**
(0,78–0,88) i na njene susede DERMASON i SEKER; to znači da model ne „preskače“
retke klase, nego meša tri klase koje su po vrednostima atributa najbliže.

**Slika 6.6.** Krive učenja za Dry Bean (`ch4_training_curves_dry_bean.png`):
gubitak na treningu i validaciji po epohi i tačnost na validaciji. Sa preko 9500
trening primera ResNet ćelije ne dostižu minimum u prvih nekoliko epoha, nego tek
nakon 20–40 epoha, pa se na ovom skupu rano zaustavljanje ponaša kako se očekuje.

Nijedna T2I metoda ne nadmašuje baselajn značajno: XGBoost (93,91 %) stoji iznad
svih ćelija osim naivnog pakovanja sa ShallowCNN-om (93,99 %), tj. **razlika je
+0,08 pp**, što je unutar šuma jedne podele. Drugim rečima, na skupu sa 16 atributa
konverzija u sliku ne donosi ni prednost ni gubitak u odnosu na klasičan ansambl
stabala.

## 6.3 Adult Income

Adult Income je najteži skup u studiji: 104 atributa nastala one-hot kodiranjem
kategoričkih obeležja, dve klase u odnosu ~75 : 25 i 31 654 : 4523 : 9045 primera.
Sto četiri atributa na 32×32 slici znači da je više od tri četvrtine atributa
prinuđeno da deli piksel sa nekim drugim, pa je ovo slučaj u kome konverzija u
sliku zaista jeste gubitna.

**Tabela 6.4.** Adult Income — F1 pozitivne klase `>50K` (u zagradi tačnost), u %.

| T2I metoda | ShallowCNN | ResNet-18 (pretreniran) | ResNet-18 (od nule) |
|---|---|---|---|
| Naivno | 68,94 (80,75) | **68,36 ± 0,82** (80,97 ± 1,62) † | 68,98 (81,05) |
| TINTO | 66,90 (78,65) | 67,18 (79,31) | 67,03 (78,84) |
| DeepInsight | 66,34 (78,57) | 66,19 (78,43) | 66,32 (78,28) |
| IGTD | 68,52 (80,56) | 67,71 (79,06) | 68,45 (80,22) |

† Ova ćelija nije citirana iz pojedinačnog runa glavne tabele, nego kao srednja
vrednost kroz pet semena (42–46); razlog je opisan u §6.9 i predstavlja artefakt
izbora checkpointa u jednom od pokretanja. Ćelije bez oznake su pojedinačni runovi
(seed 42), a njihove ponovljene vrednosti date su u Tabeli 6.12.

*Baselajni (F1 / tačnost):* RF 69,99 (84,27), XGBoost 71,43 (82,94), MLP 68,51 (81,19).

**Slika 6.7.** Toplotna mapa F1 pozitivne klase po ćelijama
(`ch4_heatmap_adult_income.png`). Ovde je raspored drugačiji nego na prethodna dva
skupa: najsvetlija vrsta (najniže vrednosti) je **DeepInsight** (66,19–66,34 %),
nešto tamnija TINTO (66,90–67,18 %), a najtamnije su naivno pakovanje i IGTD
(67,71–68,98 %). To je obrnut poredak od onog koji bi se očekivao ako bi
„pametniji“ raspored bio bolji — i upravo je to glavni nalaz ovog skupa.

**Slika 6.8.** ROC krive najbolje konvolucione mreže po T2I metodi
(`ch4_roc_curves.png`): na panelu za Adult Income krive su grupisane nisko i blizu
dijagonale, sa AUC u opsegu 0,889–0,910, dok su na panelima za Breast Cancer i Dry
Bean one blizu gornjeg levog ugla. Slika na jednom mestu pokazuje i da je ovaj
skup jedini na kome AUC pada ispod 0,91.

**Slika 6.9.** Krive učenja za Adult Income (`ch4_training_curves_adult_income.png`).
Ovo je jedini skup na kome se na krivama vidi izrazito **oscilovanje validacionog
gubitka** između epoha (gubitak se menja i za red veličine), što je direktan uvid u
nestabilnost treninga o kojoj govori §6.9.

Rezultati pokazuju tri stvari. Prvo, **naivno pakovanje i IGTD su najbolje metode
na ovom skupu** (68,94 % i 68,52 % sa ShallowCNN-om; 68,98 % i 68,45 % sa
ResNet-18 od nule), dok DeepInsight i TINTO zaostaju za 2–3 pp — konzistentno sa
činjenicom da su upravo njih dve najviše pogođene kolizijama atributa (§6.6). Drugo,
**nijedna konvoluciona ćelija ne dostiže XGBoost** (71,43 %); najbolja ćelija
u tabeli 6.4 je 68,98 % (naivno + ResNet-18 od nule), što je **−2,45 pp**, a MLP
(68,51 %) stoji na samom nivou najboljih konvolucionih ćelija. Treće, ovo nije razočaranje nego očekivan ishod: one-hot
kodirana kategorička obeležja nemaju prostornu strukturu koju bi konvolucija mogla
da iskoristi, a njihovo „pakovanje“ na 32×32 nužno spaja različite atribute u isti
piksel.

Zajedno sa prethodna dva skupa, slika je konzistentna: **konverzija u sliku je
konkurentna kada je broj atributa mali** (30 i 16 atributa), a **gubi od tabularnog
baselajna kada je broj atributa veliki** (104 atributa).

## 6.4 Poređenje sa klasičnim baselajnima

Baselajni su trenirani na **istim trening redovima** i ocenjeni **istom metrikom** kao
konvolucione ćelije. Sve tri metode su nepodešene (podrazumevani hiperparametri bez
pretrage), ali sa **balansiranim težinama klasa** (Random Forest `class_weight =
'balanced'`, XGBoost i MLP sa balansiranim težinama uzoraka) — ista korekcija koja
je primenjena i na konvolucione mreže, tako da poređenje nije „CNN sa balansom
protiv stabla bez balansa“. Ovo je ujedno i jedina intervencija u baselajne.

**Tabela 6.5.** Baselajni prema najboljoj konvolucionoj ćeliji po skupu.

| Skup | Metrika | RF | XGBoost | MLP | Najbolji baselajn | Najbolja CNN ćelija | Razlika |
|---|---|---|---|---|---|---|---|
| Breast Cancer | F1 *benign* | 95,10 | 95,83 | 94,52 | 95,83 | 97,22 (TINTO, od nule) | **+1,39** |
| Dry Bean | makro-F1 | 93,10 | 93,91 | 93,46 | 93,91 | 93,99 (naivno, ShallowCNN) | **+0,08** |
| Adult Income | F1 `>50K` | 69,99 | 71,43 | 68,51 | 71,43 | 68,98 (naivno, od nule) | **−2,45** |

Radi potpunosti, tačnost baselajna po skupu: Breast Cancer 93,86 / 94,74 / 92,98;
Dry Bean 91,81 / 92,69 / 92,40; Adult Income 84,27 / 82,94 / 81,19. Površina pod
ROC krivom: 0,992 / 0,995 / 0,981 (Breast), 0,992 / 0,995 / 0,995 (Dry Bean) i
0,903 / 0,925 / 0,903 (Adult). Na Adult Income-u Random Forest ima najvišu tačnost
među baseline-ima (84,27 %), a istovremeno **nižu F1 pozitivne klase od XGBoost-a**
(69,99 prema 71,43 %) — tipičan efekat disbalansa 75 : 25, u kome visoka globalna
tačnost dolazi od dobro pogođene većinske klase; balansirane težine taj efekat
ublažavaju, ali ga ne uklanjaju.

**Slika 6.10.** Poređenje baselajna i najbolje konvolucione mreže
(`ch4_baseline_comparison.png`): po jedan panel za svaki skup, sa tri stuba za
klasične metode i četvrtim stubom „Best CNN“ (sa nazivom metode u oznaci).
Vrednosti su ispisane iznad stubova. Slika je namerno data u tri panela sa
različitim opsezima metrike, jer su F1 pozitivne klase (na binarnim skupovima) i
makro-F1 (na Dry Bean-u) različite veličine i **ne smeju se porediti među
panelima** — porede se samo stubovi unutar istog panela.

Zaključak je trodelan: na **Breast Cancer** i **Dry Bean** konvoluciona mreža
dostiže nivo najboljeg baselajna (razlike +1,39 pp i +0,08 pp), a na **Adult
Income** tabularni baselajn (XGBoost) nadmašuje sve kombinacije T2I + CNN za
2,45 pp. Nijedan rezultat u ovom radu ne tvrdi nadmoć nad **podešenim** baselajnom:
baselajni su namerno ostavljeni na podrazumevanim vrednostima, pa bi pretraga
hiperparametara verovatno pomerila XGBoost i RF još naviše.

## 6.5 Efekat transfer učenja i kontrola ulaznog domena

Porede se dve ćelije istog kapaciteta: `resnet` (pretrenirani ResNet-18, ulaz 3
kanala sa ImageNet normalizacijom) i `resnet_scratch` (ResNet-18 od nule, ulaz 1
kanal sa sirovim vrednostima piksela). Razlika ΔF1 = F1(pretreniran) −
F1(od nule) zato meri **kombinovani** efekat pretreniranosti i ulaznog domena.

**Tabela 6.6.** ΔF1 (pp) — pretrenirani minus od-nule, po metodi i skupu.

| T2I metoda | Breast Cancer | Dry Bean | Adult Income |
|---|---|---|---|
| Naivno | +0,73 | −0,03 | **−0,28** † |
| TINTO | −0,04 | −0,20 | +0,15 |
| DeepInsight | +0,72 | +0,51 | −0,13 |
| IGTD | +0,17 | −0,97 | −0,74 |

† Ova vrednost je uparena razlika po semenu iz pet ponavljanja, sa intervalom
[−1,55; +0,98] (Tabela 6.13), a ne razlika pojedinačnih runova iz glavne tabele —
pojedinačni run te ćelije daje Δ = +0,10 pp, ali se, kao nestabilan, ne citira
(§6.9).

**Slika 6.11.** Efekat transfer učenja (`ch4_transfer_delta.png`): po jednoj grupi
stubova za svaki skup, u okviru grupe po jedna T2I metoda; ΔF1 je u procentnim
poenima, zelene šipke znače pozitivan efekat, crvene negativan. Za par koji je
ponovljen kroz semena (Adult Income + naivno) šipka nosi i **grešku od ±2
standardne greške**; neobarane šipke su pojedinačni runovi. Slika je važna kao
kontrola čitljivosti: najveća crvena šipka je IGTD na Dry Bean-u (−0,97 pp), a ne
nijedna od „dramatičnih“ vrednosti, što i jeste suština nalaza.

Nalaz je **odsustvo efekta, ne šteta**: ΔF1 ostaje unutar ±1 pp na svim skupovima
i metodama. Jedina ćelija koja je izgledala kao veliki negativan transfer
(F1 57,58 % na Adult Income-u sa naivnim slikama) pokazala se kao nestabilno
pokretanje, a ne kao efekat transfera (§6.9). Interpretacija je
jednostavna: filteri naučeni na prirodnim slikama detektuju ivice, teksture i
obilke, a T2I slike su sintetički rasteri bez takvih struktura, pa pretreniranost
ne donosi sistematsku prednost. Najveći negativni pomaci javljaju se tamo gde ulaz
najviše odstupa od ImageNet domena i gde slika nema prostornu grupisanost
(naivni redosledni raspored, IGTD traka).

Pošto Δ meša dve promene (pretreniranost i ulazni domen), sprovedena je **kontrola
ulaznog domena**: ista arhitektura od nule, ali sa 3-kanalnim ulazom koji je
ImageNet-normalizovan (isti oblik ulaza kao kod pretrenirane mreže).

**Tabela 6.7.** Kontrola ulaznog domena — ResNet-18 od nule, 1 kanal (glavna
tabela) prema 3 kanala sa ImageNet normalizacijom, F1 u %.

| Skup | Metoda | 1 kanal | 3 kanala | Δ |
|---|---|---|---|---|
| Adult Income | Naivno | 68,98 | 68,61 | −0,37 |
| Adult Income | TINTO | 67,03 | 65,90 | −1,13 |
| Adult Income | DeepInsight | 66,32 | 66,24 | −0,08 |
| Adult Income | IGTD | 68,45 | 68,44 | −0,01 |
| Breast Cancer | (opseg sve 4 metode) | — | — | 0,00 … −1,45 |
| Dry Bean | (opseg sve 4 metode) | — | — | +0,31 … +0,87 |

Kontrola pokazuje da **ni ulazni domen nije uzrok razlike**: na Adult Income-u se
sve četiri metode pomeraju najviše za 1,13 pp (u proseku oko −0,4 pp), a na Dry
Bean-u čak i u pozitivnom smeru. Drugim rečima, efekat transfera na T2I slikama
zaista je približno nula, a ne posledica toga što se poredi mreža sa kanalom više.

## 6.6 Kvalitet konverzije: raspored, kolizije i gustina

Ako konverzija u sliku treba da bude korisna, raspored atributa mora nositi
informaciju koja se ne dobija iz pojedinačnih vrednosti. Ovaj odeljak kvantifikuje
sam raspored — pre i nezavisno od performansi mreža.

**Slika 6.12.** Gde se crta koji atribut — Breast Cancer
(`ch3_feature_layout_breast_cancer.png`): četiri panela (po jedna T2I metoda), a u
svakom je položaj svakog atributa na 32×32 mreži; na skupovima sa do 30 atributa
pored svake tačke ispisan je i indeks atributa, a u naslovu panela broj atributa
koji dele piksel. Kod Breast Cancer-a se vidi da naivno pakovanje i IGTD nemaju
nijednu koliziju, dok TINTO i DeepInsight grupišu korelisane atribute blizu —
otud i njihova najbolja mesta u tabeli 6.1.

**Slika 6.13.** Isto za Dry Bean (`ch3_feature_layout_dry_bean.png`): TINTO i
DeepInsight imaju po jedan piksel sa dva atributa (12,5 % atributa), ostale dve
metode nijedan.

**Slika 6.14.** Isto za Adult Income (`ch3_feature_layout_adult_income.png`): ovde
su indeksi atributa izostavljeni (previše ih je za čitljivost), a naslovi panela
pokazuju pravu meru problema — **86 od 104 atributa** dele piksel kod TINTO-a i
**78 od 104** kod DeepInsight-a, dok naivno pakovanje i IGTD ostaju bez kolizija.

**Tabela 6.8.** Dijagnostika rasporeda po metodi i skupu (računato na trening
skupu). *OF* — udeo atributa koji dele piksel s nekim drugim; *OP* — udeo aktivnih
piksela koje deli više atributa; *ρ_S* — Spirmanova korelacija između rastojanja
dva atributa na slici i apsolutne Pirsonove korelacije njihovih vrednosti
(negativno znači „slični atributi blizu“).

| Skup (broj atributa) | Metoda | OF % | OP % | ρ_S |
|---|---|---|---|---|
| Breast Cancer (30) | Naivno | 0,0 | 0,0 | +0,04 |
| | TINTO | 13,3 | 7,1 | −0,36 |
| | DeepInsight | 6,7 | 3,4 | −0,43 |
| | IGTD | 0,0 | 0,0 | +0,02 |
| Dry Bean (16) | Naivno | 0,0 | 0,0 | −0,08 |
| | TINTO | 12,5 | 6,7 | −0,37 |
| | DeepInsight | 12,5 | 6,7 | −0,29 |
| | IGTD | 0,0 | 0,0 | −0,14 |
| Adult Income (104) | Naivno | 0,0 | 0,0 | −0,07 |
| | TINTO | **82,7** | 30,8 | **+0,48** |
| | DeepInsight | **75,0** | 23,5 | **+0,51** |
| | IGTD | 0,0 | 0,0 | −0,00 |

**Slika 6.15.** Izgled generisanih slika za Adult Income
(`t2i_comparison_adult_income.png`): kolone su metode (naivno, TINTO, DeepInsight,
IGTD), a redovi primeri (dva po klasi). Slika je vizuelni par za tabelu 6.8: kod
naivnog pakovanja i IGTD-a vidi se jedan osvetljen piksel po atributu, a kod TINTO-a
i DeepInsight-a se svetle tačke stapaju u mutne mrlje. (Isti paneli postoje i za
druge skupove: `t2i_comparison_breast_cancer.png`, `t2i_comparison_dry_bean.png`.)

**Slika 6.16.** Pokrivenost piksela (`t2i_density_comparison.png`): mreža panela po
metodi i skupu, sa jednim primerom najčešće klase po panelu i udelom piksela sa
vrednošću iznad 0,01 („non-zero pixels“) ispisanim ispod panela. Ovo **nije**
gustina atributa iz tabele 6.8 nego pokrivenost slike; slika pokazuje da TINTO
„razmazuje“ signal preko velikog dela slike (blur), dok naivno pakovanje ostavlja
sliku gotovo praznom.

**Slika 6.17.** Sličnost prema rastojanju (`ch4_arrangement_quality.png`): po
panelu za svaku kombinaciju skup × metoda, na apscisi je rastojanje dva atributa na
slici, na ordinati apsolutna Pirsonova korelacija njihovih vrednosti, a ρ_S je
ispisan u naslovu panela. Panel za Breast Cancer + DeepInsight (-0,43) pokazuje
jasan opadajući trend (slični atributi su bliže), dok paneli za naivno pakovanje i
IGTD nemaju trend; na Adult Income-u je trend obrnut (+0,48 / +0,51), što znači da
su korelisani atributi **razdvojeni**.

**Slika 6.18.** Dijagnostika preklapanja (`ch4_overlap_diagnostics.png`): dva
panela, OF % i OP % po metodi i skupu, sa numeričkim oznakama iznad stubova. Sa
nje se najbrže pročita da je preklapanje praktično isključivo problem TINTO-a i
DeepInsight-a i praktično isključivo problem skupa sa 104 atributa.

**Slika 6.19.** Performanse prema metodi i arhitekturi
(`ch4_density_vs_performance.png`): po jedan panel za skup, na apscisi je T2I
metoda, na ordinati F1, a u okviru svake metode po stub za arhitekturu;
gustina atributa skupa ispisana je u naslovu panela. Ovo je veza između tabele 6.8
i tabela 6.1–6.4 na jednom mestu: grupa stubova je najniža kod TINTO-a i
DeepInsight-a na Adult Income-u, i to nezavisno od arhitekture.

Iz ovoga sledi trodelni nalaz. (1) Na skupovima sa malo atributa raspored nema
kolizija i metoda ne menja ishod. (2) Na Adult Income-u je raspored **jeste**
ograničenje, ali ne zato što je „pogrešan“, nego zato što 104 atributa ne staje na
1024 piksela bez preklapanja — i tu veću cenu plaćaju metode koje grupišu atribute
(TINTO, DeepInsight), a ne naivno pakovanje i IGTD, koji atribute samo poređaju.
(3) Efekat rasporeda (2–3 pp) uporediv je sa efektom izbora metode, dok je efekat
izbora arhitekture pri tome manji (do 2,0 pp; §6.11).

## 6.7 Interpretabilnost modela (Grad-CAM)

Grad-CAM pokazuje koji pikseli ulazne slike najviše doprinose odluci mreže. Mape
su generisane za ShallowCNN (standardni konvolucioni slojevi daju najčitljivije
mape) za svaku od četiri T2I metode, a računate su u odnosu na **predviđenu**
klasu, što je standardna praksa za objašnjavanje odluke modela.

**Slika 6.20.** Grad-CAM za Breast Cancer (`ch4_gradcam_breast_cancer.png`):
četiri vrste (po jedna metoda), tri kolone — originalna slika, prekrivena slika
(Grad-CAM overlay) i sama toplotna mapa. Kod TINTO-a i DeepInsight-a se topli
pikseli grupišu upravo na koordinatama informativnih atributa, dok se kod IGTD-a
aktivacija razvlači preko cele trake, bez veze sa sadržajem.

**Slika 6.21.** Isto za Dry Bean (`ch4_gradcam_dry_bean.png`) — najjasniji primer
korisnosti rasporeda: TINTO toplotna masa sedi na malom broju informativnih
površina (mereno: 1,21× koncentracija u odnosu na celu sliku), dok DeepInsight na
ovoj rezoluciji ne pokazuje merljivu prostornu preferenciju.

**Slika 6.22.** Isto za Adult Income (`ch4_gradcam_adult_income.png`). Ovde je
TINTO-ova koncentracija najviša u celoj studiji (2,07×), ali ona dolazi uz cenu
kolizija iz tabele 6.8 — mreža jeste „gledala“ na pravo mesto, ali je na tom
mestu zbog preklapanja bilo više atributa.

*Kvantitativno čitanje aktivacija* (ista procedura i isti uzorci kao na slikama):
računate su (a) koncentracija pažnje na pikselima sa salijentnošću iznad 0,02 u
odnosu na celu sliku i (b) udeo najtoplijih 25 % salijentnosti koji padne na te
piksele. TINTO najdoslednije koncentriše pažnju na informativne površine — Breast
Cancer 1,08× i 57 %, Dry Bean 1,21× i 44 %, Adult Income 2,07× i 48 % (udeo
osvetljenih piksela 50 % / 34 % / 25 %). Naivno pakovanje pokazuje prostornu
preferenciju tek na Adult Income-u (1,51×; 20 % od 14 % osvetljenih piksela).
DeepInsight na ovoj rezoluciji ne pokazuje merljivu preferenciju: najtoplija masa
pada na atribute srazmerno njihovom malom udelu (2,6 % od 2,8 %; 1,9 % od 1,5 %;
1,1 % od 1,2 %), jer mape karakteristika veličine 4×4 ne razlučuju pojedinačne,
veoma retke tačke — to je ograničenje rezolucije, **ne** dokaz da raspored nije
korišćen. IGTD pokriva celu traku (udeo približno 100 %) uz dosledno **negativnu**
korelaciju salijentnosti i intenziteta (−0,16 do −0,23): pažnja mreže ne prati
sadržaj trake, što odgovara njegovim najslabijim rezultatima u tabeli 6.2.

Ovaj odeljak je i najkorisnije ograničenje za tumačenje celog poglavlja: Grad-CAM
pokazuje da mreža **jeste** koristila raspored kod TINTO-a i (delimično) kod
naivnog pakovanja, ali da to nije prevedeno u prednost na Adult Income-u, gde je
usko grlo preklapanje, a ne nedostatak prostorne pažnje.

## 6.8 Ablaciona studija

Ablacije prate isti protokol kao glavna serija (seed 42, rano zaustavljanje po
validacionom gubitku) i izvode se na fiksnoj kombinaciji DeepInsight + ShallowCNN
(mešanje piksela i raspored atributa), odnosno DeepInsight + pretrenirani ResNet-18
(LP-FT). Napomena o ponovljivosti: ćelija „original“ u svakoj ablaciji **nije**
reprodukcija ćelije glavne tabele — reč je o zasebnom pokretanju istog koda, pa se
vrednosti razlikuju do ~0,2 pp (npr. Dry Bean 93,37 % u ablaciji prema 93,53 % u
tabeli 6.2); brojevi ablacija se zato ne mešaju sa brojevima glavne tabele.

### 6.8.1 Mešanje piksela

Ablacija ima **dva kraka**. *Krak A*: model je treniran na originalnim slikama, a
testiran na slikama čiji su pikseli nasumično permutovani (ista permutacija za sve
slike, izračunata na trening skupu) — to meri **osetljivost na promenu ulaza**.
*Krak B*: model je ponovo treniran na permutovanim slikama i testiran na istoj
permutaciji — to je pravi test **da li raspored nosi informaciju**. Mešanje čuva
marginalne intenzitete (svaki piksel ostaje u slici), a uništava prostorni raspored.

**Tabela 6.9.** Mešanje piksela (DeepInsight + ShallowCNN), F1 u %.

| Skup | Original | Krak A (test permutovan) | Pad A (pp) | Krak B (trenirano na permutaciji) | B − original (pp) |
|---|---|---|---|---|---|
| Breast Cancer | 96,45 | 89,87 | −6,58 | 95,71 | −0,74 |
| Dry Bean | 93,37 | 8,16 | **−85,21** | 93,50 | +0,13 |
| Adult Income | 66,53 | 51,47 | −15,06 | 66,28 | −0,25 |

**Slika 6.23.** Mešanje piksela (`ch4_ablation_pixel_shuffling.png`): tri serije
stubova po skupu — originalni raspored, krak A i krak B. Slika je namerno
nacrtana sa sva tri kraka: da je prikazan samo krak A, čitalac bi video pad od
85 pp i pogrešan zaključak da raspored nosi informaciju.

Rezultat je jednoznačan: **krak A pokazuje veliku osetljivost** (na Dry Bean-u
model pada na nivo većinske klase, 8,16 %), ali **krak B se na sva tri skupa
vraća na nivo originala** (razlike ≤ 0,74 pp). To znači da CNN iste performanse
postiže i kada je raspored piksela permutovan — dakle raspored, na ovim skupovima,
nije nosilac informacije, a veliki pad u kraku A je samo posledica distribucionog
pomaka (model naučen na jednom rasporedu ne preživljava drugi). Zaključak se
iznosi **po skupu**, tek posle očitavanja oba kraka.

### 6.8.2 Raspored atributa

U ovoj ablaciji menja se redosled kolona na ulazu pre konverzije: originalni,
nasumični, po korelaciji atributa sa ciljnom promenljivom i obrnuti. Ista
permutacija kolona izračunata je na trening skupu i primenjena na sva tri skupa
(prethodna verzija je permutaciju računala zasebno po skupu, zbog čega su vrednosti
test skupa završavale na koordinatama naučenim iz trening skupa).

**Tabela 6.10.** Raspored atributa — F1 po poretku kolona, u %.

| Skup | Metoda | Original | Nasumični | Korelacija | Obrnuti | Raspon |
|---|---|---|---|---|---|---|
| Breast Cancer | DeepInsight | 96,45 | 96,45 | 96,45 | 96,45 | 0,00 |
| Breast Cancer | Naivno | 95,10 | 97,22 | 94,37 | 95,04 | 2,86 |
| Dry Bean | DeepInsight | 93,37 | 93,37 | 93,37 | 93,37 | 0,00 |
| Dry Bean | Naivno | 93,71 | 93,89 | 93,58 | 94,33 | 0,75 |
| Adult Income | DeepInsight | 66,53 | 66,53 | 66,53 | 66,53 | 0,00 |
| Adult Income | Naivno | 68,36 | 68,65 | 68,31 | 69,04 | 0,73 |

**Slika 6.24.** Raspored atributa (`ch4_ablation_feature_ordering.png`): dva
panela — jedan za DeepInsight, jedan za naivno pakovanje — sa četiri poretka kao
serije. Na panelu za DeepInsight sve četiri serije se **poklapaju**, a na panelu za
naivno pakovanje se razilaze.

Nalaz je dvodelan. Kod **DeepInsight-a je ablacija nerezultativna**: položaji
atributa se izvode iz njihovih međusobnih odnosa (projekcija), pa redosled kolona ne
menja generisanu sliku — sva četiri poretka daju identične rezultate na svakom
skupu (96,45 / 93,37 / 66,53 %). Kod **naivnog pakovanja redosled jeste vidljiv u
metrici**, ali je efekat mali i bez smera: raspon je 2,86 pp na Breast Cancer i
0,73–0,75 pp na Dry Bean-u i Adult Income-u, a znamo da je rasipanje usled semena na
ta dva skupa 0,36–0,37 pp (§6.9) — dakle raspon poretka je oko dva standardna
odstupanja i podjednako se lako pripisuje treningu kao redosledu. Uz to, najbolji
poredak je na Breast Cancer-u *nasumični*, a na Dry Bean-u i Adult Income-u
*obrnuti*, pa nema monotone degradacije koju bi hipoteza o „lošem“ redosledu
predviđala. Korektno čitanje: naivno pakovanje je **osetljivo** na redosled ulaza
(za razliku od DeepInsight-a, koji je invarijantan), ali taj efekat nije dovoljno
velik ni dosledan da se navede kao zaseban nalaz.

### 6.8.3 LP-FT prema direktnom finom podešavanju

**Tabela 6.11.** Direktno fino podešavanje prema LP-FT (pretrenirani ResNet-18),
F1 u % (broj epoha u zagradi).

| Skup | Direktno FT | LP-FT | Prednost LP-FT |
|---|---|---|---|
| Breast Cancer | 98,63 (37) | 91,04 (39) | **−7,59 pp** |
| Dry Bean | 93,74 (50) | 92,81 (40) | −0,93 pp |
| Adult Income | 66,44 (33) | 65,86 (38) | −0,59 pp |

**Slika 6.25.** LP-FT (`ch4_ablation_lpft.png`): po skupu dve šipke (direktno
fino podešavanje i LP-FT). Napomena za čitanje: na ovoj slici je oznaka ose
**F1 (%)**, a ne makro-F1 — za Breast Cancer i Adult Income to je F1 pozitivne
klase, a za Dry Bean makro-F1.

LP-FT (linearno sondiranje zamrznutog jezgra, pa fino podešavanje svih slojeva),
koji je u literaturi opisan kao pouzdaniji postupak, **na T2I slikama gubi**:
direktno fino podešavanje je bolje na Breast Cancer-u (−7,59 pp) i uporedivo na
preostala dva skupa. Interpretacija je konzistentna sa §6.5: zamrznuti ImageNet
filteri na sintetičkim rasterima daju slabe odlike, pa linearna faza „usidri“ glavu
na šum iz koga se fino podešavanje ne oporavlja u potpunosti.

*Ablacije su izvedene na jednoj kombinaciji metode i arhitekture (DeepInsight +
ShallowCNN, odnosno DeepInsight + pretrenirani ResNet-18); generalizacija na ostale
metode ostaje predlog za budući rad.*

## 6.9 Stabilnost kroz semena i ponovljivost

Glavna tabela (§6.1–6.3) je **jedan run po ćeliji**, na jednoj stratifikovanoj
podeli. Da to ne bi ostalo kao neizmerena nesigurnost, sedam ćelija koje nose
tvrdnje ponovljeno je kroz **pet semena (42–46)**; seme odlučuje i o podeli i o
treningu, pa rasipanje odgovara na pitanje „da li bi druga podela i drugi trening
dali drugačiji odgovor?“. Ponavljanja se čuvaju odvojeno, pa se brojevi glavne
tabele ne menjaju. Za razlike koje nose tvrdnje interval se računa **upareno po
semenu** (obe grane na istoj podeli), sa ±2 standardne greške.

**Tabela 6.12.** Stabilnost kroz pet semena (srednja vrednost ± standardna
devijacija, u %).

| Ćelija | F1 | Makro-F1 (sve klase) | Balansirana tačnost |
|---|---|---|---|
| Adult + naivno + ResNet (PT) | 68,36 ± 0,82 | 77,36 ± 1,14 | 81,59 ± 0,64 |
| Adult + naivno + ResNet (od nule) | 68,64 ± 0,75 | 77,23 ± 0,64 | 82,40 ± 0,48 |
| Adult + naivno + ShallowCNN | 68,76 ± 0,36 | 77,16 ± 0,33 | 82,75 ± 0,29 |
| Adult + TINTO + ShallowCNN | 66,61 ± 0,65 | 75,40 ± 0,71 | 81,15 ± 0,34 |
| Adult + DeepInsight + ShallowCNN | 66,43 ± 0,65 | 75,26 ± 0,49 | 81,02 ± 0,55 |
| Breast + TINTO + ShallowCNN | 95,42 ± 1,33 | 94,08 ± 1,46 | 94,86 ± 0,82 |
| Dry Bean + naivno + ShallowCNN | 93,89 ± 0,37 | 93,89 ± 0,37 | 93,98 ± 0,39 |

**Tabela 6.13.** Razlike koje nose tvrdnje, upareno po semenu (u procentnim
poenima; interval je srednja vrednost ± 2 standardne greške).

| Poređenje | Metrika | Δ | Interval | Zaključak |
|---|---|---|---|---|
| Pretrenirani − od nule, Adult + naivno | F1 pozitivne klase | −0,28 | [−1,55; +0,98] | nije razlučivo od šuma |
| Pretrenirani − od nule, Adult + naivno | makro-F1 (sve klase) | +0,14 | [−1,22; +1,49] | nije razlučivo od šuma |
| Pretrenirani − od nule, Adult + naivno | balansirana tačnost | −0,81 | [−1,75; +0,13] | nije razlučivo od šuma |
| Naivno − TINTO, Adult + ShallowCNN | F1 pozitivne klase | **+2,15** | [+1,54; +2,76] | razrešeno |
| Naivno − TINTO, Adult + ShallowCNN | makro-F1 (sve klase) | **+1,76** | [+1,02; +2,50] | razrešeno |
| Naivno − TINTO, Adult + ShallowCNN | balansirana tačnost | **+1,60** | [+1,38; +1,81] | razrešeno |

Dva nalaza iz ove tabele su najvažnija u celom poglavlju. Prvo, **razlika između
pretrenirane i mreže od nule na Adult Income-u ne postoji na nivou merenja** — sva
tri intervala sadrže nulu, pa se transfer učenje opisuje kao odsustvo efekta, a
nikako kao negativan transfer. Drugo, **jedina uparena razlika koja čisti svoj
interval je prednost naivnog pakovanja nad TINTO-om na Adult Income-u** (+2,15 pp),
i to je istovremeno potvrda nalaza iz §6.6 da su kolizije atributa stvarni
mehanizam, a ne slučajnost jednog runa.

**Ponovljivost i artefakt izbora checkpointa.** Ćelija Adult + naivno +
pretrenirani ResNet imala je u glavnoj tabeli F1 **57,58 %** (tačnost 64,70 %, što
je *ispod* većinske klase od 75,2 %), što je izgledalo kao najveći negativan
transfer u studiji. Ista konfiguracija, isti seed i isti podaci pušteni su devet puta u odvojenim
pokretanjima; dobijena su dva **disjunktna** ishoda, a unutar svake grupe rezultati
su identični do 15 cifara:

| Grupa | Broj pokretanja | F1 | Tačnost | Izabrana epoha | Epohe do zaustavljanja |
|---|---|---|---|---|---|
| A | 6 | 57,58 | 64,70 | 2 / 50 | 17 |
| B | 3 | 69,08 | 81,98 | 33 / 50 | 48 |

Razlika u ukupnom vremenu nije razlika u opremi: vreme po epohi je praktično isto u
svih devet pokretanja (26,7–27,4 s, i u grupi A i u grupi B, uz razlike unutar
grupe koje potvrđuju da su to zasebna pokretanja), pa je 460 s prema 1290 s
posledica isključivo broja epoha — grupa A se zaustavila u 17, a grupa B u 48.
Uz to, ponavljanje kroz pet semena (tabela 6.12) daje 68,36 ± 0,82 % i **nijedna**
od tih vrednosti nije blizu 57,58 %. Uzrok je vidljiv u zapisanoj istoriji treninga: validacioni gubitak pri
stopi učenja 1e-3 **osciluje** između 0,58 i 3,88, pa ako se slučajno desi da je
minimum već u 2. epohi, rano zaustavljanje (strpljenje 15 epoha) sačuva praktično
netreniran model i prijavi F1 od 57,58 %.

Zaključak je zato stroži od „to je bio jedan loš run“: **za tu ćeliju se ne citira
nijedna pojedinačna vrednost**, jer ista konfiguracija daje 57,58 % u šest od devet
pokretanja. Umesto pojedinačnog runa u tabeli 6.4 stoji srednja vrednost kroz pet
semena (68,36 ± 0,82 %) označena kao †, a posledica je da se nijedan pojedinačni
rezultat na ovom skupu ne tumači na nivou manjem od ~1 pp. Uz to, provera svih 36
ćelija po istom kriterijumu (izabrani checkpoint u prvim epohama i tačnost ispod
većinske klase) ne pronalazi nijednu **drugu** problematičnu ćeliju, pa je
nestabilnost lokalizovana na ovaj skup i ovu arhitekturu, a ne sistemska.

## 6.10 Vreme izvršavanja

Sva merenja su izvedena u istom Google Colab okruženju, jednom verzijom koda.
Zabeležena su tri vremena: `t2i_time_sec` (generisanje slika), `train_time_sec`
(samo trening) i `total_time_sec` (celo izvršavanje ćelije, uključujući generisanje,
trening i evaluaciju). Pošto je trening računski dominantan, uporedivost vremena
proverena je i po epohi: sve ćelije iste arhitekture izvršene su praktično istom
brzinom (ResNet-18 na Adult Income-u ≈ 27 s po epohi), što znači da su sva vremena
u tabeli 6.14 merena na istoj mašini i jednom okruženju.

**Tabela 6.14.** Vreme izvršavanja po skupu (12 konvolucionih ćelija po skupu;
medijane po arhitekturi, ukupno po skupu).

| Skup | Ukupno | ShallowCNN | ResNet-18 (PT) | ResNet-18 (od nule) | Najduža ćelija |
|---|---|---|---|---|---|
| Breast Cancer | 2,3 min (0,04 h) | 4,4 s | 11,2 s | 16,0 s | 18,6 s |
| Dry Bean | 58,0 min (0,97 h) | 73,0 s | 411,8 s | 434,7 s | 455,5 s |
| Adult Income | 209,8 min (3,50 h) | 301,5 s | 1464,6 s | 1276,2 s | 1705,5 s |
| **Ukupno (36 ćelija)** | **270 min (4,50 h)** | — | — | — | — |

Generisanje slika ostaje mali deo ukupnog vremena, ali nije zanemarljivo na
najvećem skupu: medijana po ćeliji na Adult Income-u iznosi **2,4 s** za naivno
pakovanje, **24,0 s** za IGTD, **33,1 s** za DeepInsight i **108,5 s** za TINTO
(TINTO upisuje sliku po primeru na disk, pa je zato najsporiji). Devet klasičnih
baselajna ukupno traje **20,6 s** (najduži pojedinačni run 6,3 s).

**Slika 6.26.** Ukupno vreme po ćeliji (`ch4_runtime_comparison.png`): po stubu
jedna ćelija, sa vremenom u sekundama. Slika pokazuje da su razlike između
arhitektura veće od razlika između metoda i da vreme skalira sa brojem primera
(Breast Cancer se trenira u sekundama, a Adult Income u desetinama minuta po
ćeliji).

Ovo je važan praktičan nalaz: **prednost klasičnog baselajna nije samo u tačnosti
na Adult Income-u, nego i u vremenu** — XGBoost za 0,9 s daje F1 bolji od svake
konvolucione ćelije kojoj je trebalo 4–28 minuta. Istovremeno, za skupove sa malo
atributa (Breast Cancer, Dry Bean) ukupno vreme ostaje u minutima, pa T2I + CNN
ostaje praktično izvodljiv pristup.

## 6.11 Sinteza: odgovori na istraživačka pitanja

**Da li T2I + CNN dostiže klasične metode?** Zavisi isključivo od broja atributa.
Na Breast Cancer-u (30 atributa) konvoluciona mreža **prelazi** najbolji baselajn za
1,39 pp, na Dry Bean-u (16 atributa) je sa njim **izjednačena** (+0,08 pp), a na
Adult Income-u (104 one-hot atributa) **zaostaje** 2,45 pp. Za skupove umerene
dimenzionalnosti, dakle, konverzija u sliku nije gubitak; za visokodimenzionalne
one-hot podatke jeste.

**Da li raspored atributa nosi informaciju?** Na ova tri skupa — **ne**. Krak B
ablacije mešanja piksela (§6.8.1) pokazuje da mreža trenirana na permutovanom
rasporedu postiže isti F1 kao mreža trenirana na originalnom (razlike ≤ 0,74 pp),
ablacija rasporeda je za DeepInsight nerezultativna zbog invarijantnosti, a
jednostavan naivni raspored nadmašuje „pametne“ projekcije na najtežem skupu.
Grad-CAM ipak pokazuje da mreža **koristi** raspored kada joj je ponuđen (TINTO
koncentriše pažnju 1,08–2,07× na informativne piksele), ali to se ne prevodi u
prednost na skupovima gde informacija staje u marginalne statistike atributa.
Prostorni raspored jeste relevantan — samo je njegov uticaj manji od izbora
arhitekture i režima treninga.

**Da li transfer učenje pomaže na sintetičkim slikama?** **Ne.** Razlika između
pretrenirane i mreže od nule ostaje unutar ±1 pp na svim skupovima i metodama, a
tamo gde je proverena kroz pet semena nije razlučiva od šuma. Kontrola ulaznog
domena (tabela 6.7) pokazuje da to nije artefakt različitog broja kanala, a LP-FT
ablacija daje isti smer: zamrznuti filteri sa prirodnih slika ne odgovaraju
sintetičkim rasterima.

**Šta najviše utiče na rezultat?** Ako se svaki efekat meri *unutar* ostalih
faktora (metoda unutar iste arhitekture, arhitektura unutar iste metode), poredak
po veličini je sledeći: (1) **IGTD na Dry Bean-u**, koji od ostalih metoda zaostaje
1,7–3,5 pp — i to je ujedno najveći pojedinačni efekat u celoj mreži (3,46 pp na
pretreniranom ResNet-u); (2) **izbor metode na Adult Income-u** (2,6–2,9 pp, gde
naivno pakovanje nadmašuje projekcione metode); (3) **izbor arhitekture**, koji je
svuda skroman — najviše 1,95 pp (IGTD na Dry Bean-u), a tipično ispod 1 pp;
(4) **pretreniranost**, unutar ±1 pp na svim skupovima i metodama. Drugim rečima,
kvantitativno najveće razlike u radu potiču od **jedne slabo prilagođene metode**
(IGTD sa trakastim rasporedom) i od **kolizija atributa na skupu sa 104 obeležja**,
a ne od pretreniranosti ili od suptilnih razlika u rasporedu. Nijedan od ovih
efekata nije dovoljno velik da bi se, na jednoj podeli i jednom runu, mogao
proglasiti pouzdanom razlikom bez ponavljanja kroz semena (§6.9).

## 6.12 Ograničenja istraživanja

1. **Jedna stratifikovana podela po skupu** u glavnoj tabeli; procena varijanse
   data je posebno i samo za sedam ćelija koje nose tvrdnje (§6.9). Unakrsna
   validacija nije sprovedena, pa su razlike manje od ~1 pp na pojedinačnim runovima
   nezaključive.
2. **Mali broj primera na Breast Cancer-u** (398 trening primera) uz duboke modele;
   jedna promena klase u test skupu (dva primera) menja F1 za oko 1 pp, što
   dodatno ograničava tumačenje razlika te veličine.
3. **Baselajni su nepodešeni** (osim balansiranih težina klasa, §6.4). Poređenje
   pokazuje da CNN dostiže *podrazumevani* baselajn, a ne da je bolji od podešenog.
4. **Fiksni hiperparametri za sve metode** mogu pojedinoj metodi uskratiti njen
   optimalni režim; posebno je pokazano da je stopa učenja kritična za ViT.
5. **Nestabilnost treninga pri stopi učenja 1e-3** na Adult Income-u (oscilovanje
   validacionog gubitka, §6.9) znači da jedan deo varijanse potiče od optimizacije,
   a ne od podataka ili metode; ta ćelija je zato prijavljena kao raspodela, ne kao
   pojedinačna vrednost.
6. **TINTOlib je korišćen kao crna kutija** za projekciju i optimizaciju rasporeda;
   rezultati na nivou pojedinačnih primera mogu zavisiti od verzije biblioteke.
7. **Tri arhitekture; ViT-Base/16 nije uključen** iz računskih razloga (zahteva GPU
   vreme), a S-IGTD nije evaluiran — rezultati se odnose na četiri metode i tri
   arhitekture.
8. **Baselajni su osetljivi na verziju biblioteke.** Sa istim seed-om i istim
   podacima, Random Forest daje 95,10 / 93,10 / 69,99 % u okruženju u kome je
   izvršena cela serija (Python 3.13.15, `scikit-learn` 1.9.1), a 96,55 / 93,48 /
   67,84 % pod `scikit-learn` 1.7.2; XGBoost i MLP se pri tome pomeraju za manje od
   0,01 pp. Pošto su konvolucione ćelije i baselajni mereni u **jednom okruženju**,
   poređenja u ovom poglavlju ostaju konzistentna, ali se apsolutne vrednosti
   baselajna ne mogu reprodukovati bez iste verzije biblioteke. Zbog toga je
   verzija biblioteke deo protokola i treba je navesti u opisu postavke.

---

# 7. Zaključak

U radu je ispitan pristup klasifikaciji tabelarnih podataka u kome se vektor
atributa najpre preslikava u sliku, a klasifikaciju zatim preuzima konvoluciona
neuronska mreža. Poređena su četiri postupka preslikavanja — naivno pakovanje,
projekcione metode DeepInsight i TINTO i permutaciona metoda IGTD — na tri
heterogena skupa (Breast Cancer, Dry Bean, Adult Income), uz tri konvolucione
arhitekture (ShallowCNN, pretrenirani ResNet-18 i ResNet-18 od nule), tri klasična
baselajna (Random Forest, XGBoost, MLP) i tri ablacione studije (mešanje piksela,
raspored atributa i LP-FT). Eksperimenti su izvedeni po jedinstvenom protokolu — 36
konvolucionih i 9 baselajnih ćelija, jedna stratifikovana podela, fiksni
hiperparametri i deterministički seed — uz dodatnu proveru stabilnosti kroz pet
semena za ćelije koje nose tvrdnje (6.9).

**Glavni zaključak je da konverzija u sliku nije univerzalno korisna, nego zavisi
od odnosa broja atributa i površine slike.** Na Breast Cancer-u (30 atributa na
32×32) najbolja konvoluciona ćelija prelazi najbolji klasični baselajn za 1,39 pp
(97,22 % prema 95,83 %), na Dry Bean-u (16 atributa) je sa njim izjednačena
(+0,08 pp), a na Adult Income-u (104 one-hot obeležja) zaostaje 2,45 pp (68,98 %
prema 71,43 %). Granica se ne pomera sa izborom metode, nego sa gustinom rasporeda:
samo na skupu sa 104 atributa konverzija je izgubila informaciju, i to je jedini
skup na kome je baselajn jasno bolji.

**Usko grlo nije konvoluciona mreža, nego gubitak informacije u samom
preslikavanju.** Kod TINTO-a i DeepInsight-a na Adult Income-u 86, odnosno 78
atributa od 104 dele piksel sa nekim drugim (OF = 82,7 % i 75,0 %), a Spirmanova
korelacija između rastojanja i sličnosti atributa postaje **pozitivna**
(+0,48 / +0,51) — korelisani atributi se, dakle, razdvajaju. Upravo te dve metode na
tom skupu i zaostaju, dok naivno pakovanje i IGTD, koji nemaju nijednu koliziju,
daju najbolje rezultate. Uparena razlika naivnog pakovanja nad TINTO-om na Adult
Income-u (+2,15 pp, interval [+1,54; +2,76]) jedina je razlika u celom radu koja
čisti svoj interval, što kolizije potvrđuje kao stvarni mehanizam, a ne kao
slučajnost jednog pokretanja.

Delimično drugačiji mehanizam vidi se kod IGTD-a, koji je dosledno najslabiji
(zaostatak 1,7–3,5 pp na Dry Bean-u) iako nema kolizija: njegov raspored svodi
atribute na jednu traku, pa slika nema lokalnu strukturu koju bi 3×3 jezgra mogla da
iskoriste, a Grad-CAM to i potvrđuje (pažnja mreže ne prati sadržaj trake, uz
negativnu korelaciju salijentnosti i intenziteta). Oba nalaza vode istom zaključku:
korisnost konverzije određuje da li dobijena slika zadržava **i** identitet
pojedinačnog atributa **i** prostornu lokalnost, a ne da li je raspored „pametan“ u
smislu optimizacije.

**Prostorni raspored jeste relevantan, ali na ovim skupovima nije nosilac
informacije.** Model treniran na originalnom rasporedu ne preživljava permutaciju
piksela (na Dry Bean-u makro-F1 pada za 85,21 pp, do nivoa većinske klase), ali
kada se na istoj permutaciji ponovo trenira, rezultat se vraća na nivo originala
(razlike ≤ 0,74 pp; 6.8.1) — dakle veliki pad meri osetljivost na promenu ulaza
(distribucioni pomak), a ne vrednost rasporeda. Isto pokazuju i ablacije rasporeda
atributa: DeepInsight je na redosled kolona invarijantan, jer položaje izvodi iz
međusobnih odnosa atributa, a naivno pakovanje jeste osetljivo, ali je efekat
(0,73–2,86 pp) u granicama rasipanja usled semena i bez doslednog smera.

**Transfer učenje na sintetičkim T2I slikama nije donelo prednost.** Razlika
pretrenirane i mreže od nule ostaje unutar ±1 pp na svim skupovima i metodama, a na
Adult Income-u sa naivnim slikama iznosi −0,28 pp, uz interval [−1,55; +0,98] koji
sadrži nulu. Kontrolnim eksperimentom (ista arhitektura od nule, ali sa 3-kanalnim
ImageNet-normalizovanim ulazom) pokazano je da razlika nije posledica različitog
ulaznog domena, a LP-FT ablacija daje isti smer — direktno fino podešavanje je bolje
od LP-FT-a (−7,59 pp na Breast Cancer-u). Ovaj nalaz je objavljiv kao **negativan
rezultat**: filteri naučeni na prirodnim slikama opisuju ivice, teksture i oblike
kojih na sintetičkim rasterima nema, pa se prednost pretreniranosti, uobičajena u
literaturi, na ovom tipu ulaza ne reprodukuje.

**Metodološki doprinos rada je i nalaz o granicama pouzdanosti ovakvih
poređenja.** Ćelija Adult + naivno + pretrenirani ResNet dala je F1 57,58 % u šest
od devet identičnih pokretanja (isti seed, isti podaci, ista brzina po epohi) i
69,08 % u preostala tri, jer pri stopi učenja 1e-3 validacioni gubitak osciluje, a
rano zaustavljanje povremeno sačuva checkpoint iz 2. epohe od 50 (6.9). Zaključak
je da se na jednoj podeli i jednom pokretanju razlike manje od ~1 pp ne smeju
čitati kao rangiranje; u ovom radu su zato ćelije koje nose tvrdnje ponovljene kroz
pet semena, a razlike su iskazane upareno po semenu i sa intervalom. Provera svih
36 ćelija po istom kriterijumu nije našla nijednu drugu takvu ćeliju.

**Praktično gledano, T2I + CNN ostaje izvodljiv pristup tamo gde je broj atributa
mali.** Ukupno vreme svih 36 konvolucionih ćelija iznosi 4,50 h, od čega skoro
četiri petine otpada na najveći skup, a generisanje slika (medijana po ćeliji na
tom skupu 2,4 s za naivno pakovanje do 108,5 s za TINTO) čini mali deo; devet
klasičnih baselajna ukupno traje 20,6 s. Na Adult Income-u XGBoost za 0,9 s postiže bolji F1 od svake
konvolucione ćelije kojoj je trebalo 4–28 minuta — pa se u inženjerskoj praksi, pre
uvođenja T2I-CNN postupka, isplati najpre proveriti klasičan ansambl.

Budući rad otvara se u četiri smera. Prvo, **protokolarna strogost**: unakrsna
validacija ili više podela po ćeliji, uz više semena, dala bi intervale pouzdanosti
i za ćelije koje ovde nisu ponavljane, a podešavanje baselajna pokazalo bi koliko se
razlika iz 6.4 menja pod pretragom hiperparametara. Drugo, **skaliranje na
visokodimenzionalne skupove**: pošto je na Adult Income-u usko grlo preklapanje,
sledeći korak je veći raster ili kompaktnija višekanalna kodiranja koja čuvaju
identitet atributa, a ne nova arhitektura. Treće, **arhitekture i režimi treninga**:
pretrenirani ViT-Base/16 nije ušao u finalnu seriju zbog računskih zahteva, a pilot
testiranje je pokazalo da mu je stopa učenja kritična ($10^{-4}$, dok rane verzije
sa $10^{-3}$ kolabiraju na predviđanje jedne klase); uz GPU budžet to ostaje
otvoreno pitanje, kao i S-IGTD kao supervizovana varijanta rasporeda. Četvrto,
**stabilnost izbora modela**: pokazano je da sama procedura ranog zaustavljanja
može odlučiti o rezultatu, pa bi izveštavanje trebalo da uz svaku ćeliju beleži
izabranu epohu i istoriju validacionog gubitka — što je u ovom radu i učinjeno.

Najkraće: informacija koju nosi sam prostorni raspored jeste na ovim skupovima
izmerljivo mala (razlike ≤ 0,74 pp), pa je i manja od uticaja izbora arhitekture
(do 1,95 pp) i od uticaja pretreniranosti (±1 pp). Ono što rezultat zaista menja
jeste da li preslikavanje čuva informaciju iz pojedinačnih atributa: kolizije na
skupu sa 104 obeležja i slomljena lokalnost trakastog rasporeda pomeraju F1 do
3,46 pp, više od bilo kog drugog faktora u ovom radu. T2I + CNN zato nije zamena za
klasične metode na tabelarnim podacima uopšte, nego specijalizovan pristup koji
opravdava svoju cenu na skupovima sa umerenim brojem atributa, gde dostizanje nivoa
baselajna — ili blaga prednost nad njim — dolazi uz interpretabilnost koju pruža
Grad-CAM.

---
---

# Napomene za tebe (ne ide u rad)

### 1. Šta je provereno, a šta je pretpostavljeno

Svi brojevi u tekstu su izvedeni iz fajlova, ne prepisani iz ranijeg nacrta:
`results/all_experiments.csv` (45 redova: 36 CNN + 9 baselajna) za tabele 6.1–6.5 i
6.14, `results/seed_summary.csv` + `results/stability_table.md` za 6.12–6.13,
`results/ablation_*.json` za 6.9–6.11, `results/scratch3ch/*.json` za 6.7, i
direktan proračun preklapanja i Spirmanove korelacije na trening skupu za tabelu 6.8
(korišćenjem istih funkcija koje crtaju sliku 6.18).

Posle toga je cela serija ponovo pokrenuta u Colab-u (ponovljeni su samo baselajni,
kao i obavezne završne obrade) i **nijedan broj u ovom tekstu se nije promenio**;
jedina izmena je vreme baselajna (ukupno 21,1 → 20,6 s), jer su tri šume ponovo
trenirane. Provera je urađena i mašinski: svi JSON zapisi, CSV, tabela stabilnosti
i svih 30 slika upoređeni su sa prethodnim stanjem — 200 od 204 fajla su
identični, a razlikuju se samo tri `baseline_*_rf.json` (polje `train_time_sec`) i
CSV (isti razlog).

### 2. Šest mesta na kojima se ovaj tekst razlikuje od `Plan/seminar2-rad-nacrt.md`

| # | Nacrt kaže | Ovde stoji | Zašto |
|---|---|---|---|
| 1 | Breast Cancer: „RF 96,55“, razlika prema CNN +0,67 pp | RF **95,10**, razlika **+1,39 pp** prema XGBoost-u (95,83) | `results/baseline_*_rf.json` i CSV sadrže 95,10 / 93,10 / 69,99, i to je potvrđeno ponovnim pokretanjem (napomena 3) |
| 2 | Adult: „78, odnosno 70 od 104 atributa u koliziji“ | **86** (TINTO) i **78** (DeepInsight) | Direktan proračun sa aktuelnim kodom; ponovljeno dvaput, ista vrednost |
| 3 | „ukupno ≈ 4,22 h“, Adult ≈ 3,21 h | **4,50 h**, Adult **3,50 h** | Zamenjena ćelija (Adult + naivno + ResNet) trajala je duže od odbačene |
| 4 | „artefakt izbora modela… ponavljanje je pokazalo da to nije svojstvo konfiguracije“ | Ista konfiguracija dala je 57,58 % u **6 od 9** ponovljenih pokretanja | Devet arhiviranih runova ćelije; vidi tabelu u §6.9 |
| 5 | „`audit_cells.py` pronalazi tačno ovu ćeliju i nijednu drugu“ | Skripta sada vraća **HIGH 0** | Trenutni zapis ćelije je zdrava grupa B, pa se nalaz ne reprodukuje na njoj |
| 6 | „efekat metode (~2–3 pp) manji je od efekta arhitekture (do ~3,7 pp)“ | Efekat **metode** je veći: do **3,46 pp** (metoda unutar iste arhitekture, Dry Bean + pretrenirani ResNet), a efekat **arhitekture** najviše **1,95 pp** (arhitektura unutar iste metode, IGTD na Dry Bean-u) | Tvrđena brojka 3,7 pp je razlika između *najbolje i najslabe ćelije u tabeli*, što meša metodu i arhitekturu; vidi §6.11 |

Tačke 4 i 5 su najvažnije: formulacija „artefakt“ sugeriše jednokratnu grešku, a
podaci pokazuju **ponovljivu nestabilnost**. Tekst u §6.9 je zato napisan strože —
nijedna pojedinačna vrednost te ćelije se ne citira. Ako želiš blažu formulaciju,
zameni pasus „Zaključak je zato stroži…“ rečenicom: *„Za tu ćeliju se zato ne
citira pojedinačna vrednost, već raspodela kroz pet semena.“*

Ista greška kao u tački 6 („uticaj rasporeda manji od uticaja arhitekture“) stajala
je i u **oba sažetka nacrta**. Po izmerenim brojevima to nije tačno u tom obliku:
raspored kao *nosilac informacije* utiče najmanje (≤ 0,74 pp), dok izbor metode i
kolizije utiču najviše (do 3,46 pp). Formulacija koju novi Zaključak koristi
(„informacija koju nosi sam prostorni raspored je izmerljivo mala, pa je manja i od
uticaja arhitekture i od uticaja pretreniranosti“) tu razliku čini eksplicitnom, a
ta ista formulacija je sada prenesena i u oba sažetka `seminar2-rad-nacrt.md`, pa tu
nema više šta da se menja.

### 3. RF baselajn — rešeno (potvrđeno ponovnim pokretanjem)

RF je bio jedina otvorena stavka: zapisani rezultati dali su **95,10 / 93,10 /
69,99**, a ponovno pokretanje istog koda na laptopu davalo je **96,55 / 93,48 /
67,84**. Baselajn je zato ponovo pokrenut **u okruženju u kome je izvršena cela
serija** (Colab, Python 3.13.15, `scikit-learn` 1.9.1, nakon što su stare RF
datoteke sklonjene u `results/backup_rf_before/`) i **reprodukovao je zapisane
vrednosti**: 95,10 / 93,10 / 69,99. XGBoost i MLP se pri tom nisu menjali, kao ni
jedna konvoluciona ćelija. Razlika je dakle bila u verziji biblioteke, ne u
rezultatima, pa **sve vrednosti u ovom tekstu ostaju na mestu** — vidi ograničenje 8
odmah ispod.

Za odbranu je važno da umeš da kažeš zašto se citiraju te vrednosti: sve 45 ćelija
(konvolucione i baselajni) merene su u **jednom okruženju**, pa su poređenja
unutrašnje konzistentna; pomeranje od 1,45 pp na Breast Cancer-u (RF) samo pokazuje
da je apsolutnu vrednost šume nemoguće reprodukovati bez iste verzije biblioteke —
éto još jednog razloga da se razlike manje od ~1 pp ne rangiraju (§6.9).

Ako budeš kontrolisao brojeve iz CSV-a ručno, dve stvari izgledaju kao odstupanje,
a nisu: (1) najbolja ćelija na Adult Income-u u sirovom CSV-u je 69,08 % (pojedinačni
run ćelije Adult + naivno + ResNet), a u tabeli 6.4 se za tu ćeliju koristi
srednja vrednost kroz semena (68,36 ± 0,82 %, oznaka †), pa je najbolja ćelija
tabele 68,98 % i razlika −2,45 pp — u sirovom CSV-u bi bila −2,35 pp; (2) u tabeli
6.6 je za Adult + naivno upotrebljena uparena razlika po semenu (−0,28 pp), pa se
Δ za taj par namerno razlikuje od razlike pojedinačnih vrednosti iz CSV-a (+0,10 pp).

### 4. Slike — šta je neophodno, a šta može da ispadne

**Neophodno (11):** 6.1 heatmap_breast · 6.4 heatmap_dry_bean · 6.7 heatmap_adult ·
6.10 baseline_comparison · 6.11 transfer_delta · 6.14 feature_layout_adult ·
6.18 overlap_diagnostics · 6.23 + 6.24 + 6.25 tri ablacije · 6.26 runtime.

**Vredi imati (10):** 6.2 / 6.6 / 6.9 krive učenja · 6.3 matrice konfuzije · 6.5
po-klasni F1 · 6.8 ROC · 6.15 t2i_comparison_adult · 6.12 i 6.13 feature_layout
(manji skupovi) · 6.17 arrangement_quality.

**Može da ispadne bez gubitka (5):** 6.16 t2i_density_comparison · 6.19
density_vs_performance · 6.20 / 6.21 / 6.22 Grad-CAM (ako štampa ide u boji, ostavi
samo 6.21 za Dry Bean, jer je najilustrativniji).

Napomena o Grad-CAM-u: troje slika je veliko (do ~25 000 px širine za Dry Bean u
punoj rezoluciji) — u Word ih ubaci u jednoj koloni i smanji na širinu teksta.

### 5. Novi Zaključak — šta zameniti u Wordu

Poglavlje **7. Zaključak** u fajlu iznad zamenjuje **celo** postojeće poglavlje 7
(odmah ispod naslova „7. Zaključak“, do naslova „8. Reference“). Stari tekst se
briše u celosti: on je bio izveden iz pregleda literature i nije pominjao nijedan
izmereni rezultat, a uz to je u njemu bilo i tehničkih netačnosti (npr. tvrdnja da
transfer učenje omogućava „augmentaciju“ u smislu koji nije primenjen u ovom radu,
i opis T2I-CNN pristupa kao regularizatora za HDLS scenarije, što ovaj rad nije
pokazao).

Struktura novog Zaključka je namerno ista kao struktura poglavlja 6 — svaki
pasus nosi poznatu brojku iz rezultata, `**podebljan**` vodi čitaoca kroz nalaz, a
reference na potpoglavlja su u obliku „6.9“, „6.8.1“ (izvor je ostavljen tako da ga
u Wordu možeš zameniti pravom unakrsnom referencom, jer se naslovi u Wordu
numerišu automatski).

Jedna ograda za odbranu: novi Zaključak tvrdi da je **„presudno da li preslikavanje
čuva informaciju iz pojedinačnih atributa“** i da je uticaj rasporeda manji od
uticaja arhitekture i režima treninga. To je direktno izmereno (tabela 6.8, tabela
6.10, tabela 6.13), ali je izvedeno sa **jedne podele po skupu**, uz pet
ponavljanja samo za sedam ćelija. Ako ti komisija traži jaču formulaciju, stroža
verzija je „na ovim skupovima i pod ovim protokolom“ na mestu prve rečenice
poslednjeg pasusa.

### 6. Nedostaci koje treba znati pre odbrane

- **Poglavlje 7 (Zaključak)** je sada napisano i nalazi se u ovom fajlu; u Wordu ga
  treba zameniti u celosti (vidi napomenu 5).
- **Sekcija 6 u Word-u** postoji samo kao naslov „6. Rezultati i diskusija“ i
  podnaslov „6.1 Breast Cancer Wisconsin“ bez teksta — ovaj fajl popunjava oba.
- **„Jedna mašina“** stoji, ali sa jednom ogradom: sedam ćelija iz tabele 6.12
  pokrenuto je pre commit-a koji dodaje zapis o checkpointu (ta izmena dodaje samo
  zapis, ne menja trening), pa je ispravna formulacija „jedna verzija koda, jedan
  Colab runtime“. Nova provera to sada i potkrepljuje: vreme po epohi je isto u
  svih devet ponavljanja ćelije iz §6.9 i u ćelijama glavne tabele (~27 s/epoch za
  ResNet-18 na Adult Income-u), pa je cela serija merena na istoj mašini i istom
  okruženju.
- **CPU ili GPU — rešeno u korist GPU-a, ali nacrt to još kaže pogrešno.** Polje
  `device` postoji u devet zapisa i svih devet kaže `cuda`. Nezavisna provera to
  potvrđuje: ResNet-18 na Adult Income-u ima 31 654 primera u treningu, pa je pri
  batch-u 32 to 989 koraka po epohi; izmereno je ≈ 27 s po epohi, dakle **27 ms po
  koraku** — red veličine ispod onoga što daje Colab CPU, a tačno u redu veličine
  GPU-a za 32×32 ulaz. **Preporuka: u tekstu navesti GPU (kao što Word na str. 37 i
  čini), a „CPU“ iz uvoda poglavlja 6 ukloniti.** U ovom fajlu je tvrdnja o uređaju
  namerno izostavljena (piše samo „jedno Google Colab okruženje“), pa ako želiš da
  je dodaš, dodaj je u §6.10. Nezavisno od uređaja, **vremena u tabeli 6.14 su zidna
  vremena i uporediva su međusobno** (dokaz: isto vreme po epohi u svim ćelijama).
