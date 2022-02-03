# Speed optimization
* [01] **Accessing an item from a list is about 12% faster than from a dict.**
* [02] **Accessing an item from a numpy array takes twice as long as from a list.**
* [03] **np.random.choice is 250 times slower than straightforward choosing item from a list of length 3.**
The difference increases as the list length increases.
*  [04]  **Ordered agent activation is approximately 25% faster than random activation.**

# Customer states
OrdinaryPearsonAgent.state
* -1 = "recovery"
* 0 = "susceptible"
* 1 = "incubation" it is the same as "exposed"
* 2 = "prodromal"
* 3 = "illness"
* 4 = "dead"

"recovery" = -1, for quick check if agent is health enough to go shopping: **agent.state <= 2**

# Notation
## BatchRunnerMP
### Result data type
Collecting data by BatchRunnerMP returns data as dictionary in which
key is a tuple that looks like (variable_params_values, fixed_params_values, num_of_iteration).
Order of items in _params is the same as given in BatchRunnerMP constructor.

Values in this dictionary are pandas dataframes containing indicated model parameters
for each day of simulation. 

# Questions
* [01] Order of activation first agents or cashiers or maybe agents from first neighbourhood then 
cashier from first neighbourhood then agents from second neighbourhood e.t.c.
* [02] Real number of prodromal days 3 or 4. For 4 it should be on average 55.93 prodromal peoples (matches Fig. 1).

# Some results
| N | iterations | BMP | maxiterationsperchild | RAM | time |
| --- | --- | --- | --- | --- | --- |
| 1000 | 10k | standard | N/A | 10.3 GB (65%) | 5 min 53 s |
| 1000 | 10k | modified | 100 | 8.5 GB (47%) | 6 min 28 s |
| 1000 | 20k | standard | N/A | 24.3 GB (94%) | 11 min 52 s |
| 1000 | 20k | modified | 200 |  |  |

# Problem
* It's monday 11:00 cashier was infected and is in incubation phase for 0h = 0 days.
* It's tuesday 11:00 cashier is in incubation phase for 24h = 1 day.
* It's wednesday 11:00 cashier is in incubation phase for 48h = 2 days.
* It's thursday 00:01 cashier is in which phase? Incubation duration: 61h = 2.55 days.

## Case 1
If we count with respect to beginning of new day, then the first reported moment in which cashier is in incubation
phase is tuesday 00:01 and cashier goes into prodromal phase on friday at 00:01. In that scenario 
incubation period is in fact bigger than reported due to last 13h from monday which
were counted as susceptible, but were incubation.

## Case 2
If we count with respect to the end of current day, then the first reported moment in which cashier is in incubation
phase is monday  23:59. New day begins. It's tuesday 00:01.
Should I decrease incubation period by 1 right now or wait for tuesday 23:59?
* If I decrease incubation period by 1 right now then cashier goes into prodromal phase on thursday at 00:01.
In that scenario incubation period is in fact smaller than reported due to first 11h from monday which
were counted as incubation, but were susceptible.

* If I decrease incubation period by 1 at 23:59 then cashier goes into prodromal phase on friday at 00:01.
In that scenario incubation period is in fact bigger than reported due to last 13h from monday which
were counted as susceptible, but were incubation.


## Case 3
Cashier goes into prodromal phase on thursday at 11:00.

### Counts with respect to the beginning of the day
Counted as susceptible on monday.
Counted as incubation on tuesday, wednesday, thursday.
Counted as prodromal on friday and so on.

### Counts with respect to the end of the day (currently used)
Counted as incubation on monday, tuesday, wednesday.
Counted as prodromal on thursday and so on.


## Case 4 
Cashier goes into prodromal phase on thursday at random hour.

### Counts with respect to the beginning of the day
Counted as susceptible on monday.
Counted as incubation on tuesday, wednesday, thursday.
Counted as prodromal on friday and so on.

### Counts with respect to the end of the day (currently used)
Counted as incubation on monday, tuesday, wednesday.
Counted as prodromal on thursday and so on.

# Wiadomości

## Wiadomość 1
Dzień dobry,
chciałbym przekazać Panu kilka informacji.

Jak się Pan pewnie domyśla, nie zdążę złożyć mojej pracy dyplomowej do końca semestru i
będę się ubiegał o przedłużenie czasu na realizację tej pracy. Jak podaje regulamin
(punkt 13 https://www.fizyka.pw.edu.pl/index.php/content/download/894/14169/file/Zasady%20wykonywania%20prac%20dyplomowych.pdf)
"Na wniosek studenta i promotora możliwe jest także przesunięcie terminu złożenia
pracy dyplomowej, nie dłużej jednak niż o 3 miesiące w stosunku do terminu, o którym mowa
w punkcie 12." mogę się strać o przedłużenie do 3 miesięcy. Niestety, o ile mi wiadomo, 
na stronie wydziału nie ma przygotowanego do tego celu wzoru podania i należy skorzystać, z
ogólnego wzoru podania (https://www.fizyka.pw.edu.pl/Studia2/Dokumenty-i-formularze).
Znalazłem jednak kilka takich formularzy na innych wydziałach i na ich podstawie uzupełniłem 
podanie ogólne (załącznik). Myślę, że jest ono odpowiednie, ale nie zaszkodzi dodatkowe
sprawdzenie przez Pana.

Podanie podpisane przez Pana i mnie, muszę złożyć w dziekanacie najpóźniej 16 lutego 2022.

Myślę, że realnie potrzebuję jeszcze jakieś 1,5 miesiąca (miesiąc na otrzymanie finalnych 
wyników i 2 tygodnie na napisanie), ale skoro i tak już ubiegam się o przedłużenie,
to we wniosku poproszę o te 3 miesiące (do 30 kwietnia 2022), bo pracę zawsze mogę
oddać wcześniej, a na wszelki wypadek jakiś zapas czasu dobrze mieć.


## Wiadomość 2
W tej wiadomości opiszę w skrócie, co robiłem w ostatnim czasie.
Tak jak wcześniej rozmawialiśmy, swoją uwagę przeniosłem na faktycznie
pierwsze dni pandemii (dane w excelu, które dostałem od Pana, a nie te, co
miałem z GUS). Przypomnę, że idea była taka, by wyniki z symulacji porównywać
z fragmentem danych rzeczywistych, powiedzmy pierwsze 80 dni od odnotowania
pierwszego zgonu w województwie. Poniżej wytłumaczę, czemu takie podejście
jest zbyt naiwne i nie daje dobrych wyników.

Zacząłem po prostu od wykreślenia death toll dla każdego z województw, 
w przeciągu 100 i 200 dni od początku prowadzenia statystyk.
(Rysunek real_death_toll_100 i real_death_toll_200)

Z tych rysunków dobrze widać, że w różnych województwach rozwój
pandemii w poszczególnych województwach był skrajnie różny, np. w
województwie A dnia 10 pojawił się pierwszy zgon, dnia 40 drugi, 50 trzeci
i dopiero od tego momentu death toll rósł "naturalnie". Taki opóźniony
zapłon sprawił, że w dniu 100 death toll = 30.
Z kolei w województwie B pierwszy zgon pojawił się również 10 dnia, ale już 
od tego momentu death toll rósł "naturalnie" i finalnie w setnym dniu było
nie 30 a 200 zmarłych.

Pomyślałem, że mój model nie jest w stanie oddać dynamiki województwa A,
jeśli zacząć pierwszego zgonu, ale gdyby zacząć od 30 zgonów to może się udać.
Idąc tym tropem, dla każdego z województw "na oko" odczytałem wartość death 
toll, powyżej której wykresy wyglądają ok. Wyszło mi, że jeśli utworzyć pięć
progów death toll (1, 20, 50, 100, 200) to w przypadku każdego z województw,
któryś z tych progów dość dobrze wyznacza, od ilu zgonów death toll rośnie 
"naturalnie". Wykresy poniżej.
(Rysunki real_death_toll_shifted_by_hand)

W tym momencie wydawało mi się, że mam to, czego szukałem (początkowy etap
pandemii dla każdego z województw). Niestety, zaraz potem pojawiły się dwa
nowe problemy.
Po pierwsze nie sądzę, by taka metoda "na oko" znajdowania początku
pandemii była akceptowalne, z tego względu, że ktoś inny mógłby znaleźć inne 
progi i inaczej przyporządkować im grupy województw i w konsekwencji dostać 
inne wyniki (zbyt duża arbitralność i poleganie na ocenie człowieka).
Po drugie zbyt duża rozbieżność w czasie. Jeśli spojrzy Pan na legendę z 
wykresów powyżej to w przypadku grupy województw o progu death toll = 1, 
dzień początkowy to mniej więcej koniec marca, a dla grupy województw o 
death toll = 50 i więcej, za dzień początkowy należałoby uznać środek
października, czyli różnica między grupami wynosi ponad pół roku!
Myślę, że ta różnica jest za duża oraz że błędnie jest uważać, by w
niektórych województwach pandemia zaczęła się dopiero w środku
jesieni 2020, bądź jeszcze później.

Ostatnie i aktualnie rozważane przeze mnie podejście do określenia początku
pandemii opiera się na ilości powiatów w danym województwie, w których
odnotowano przypadki śmiertelne. Pomyślałem, że kryterium mówiące, że
"dzień D uznaję, za początek pandemii w województwie W, wtedy i tylko 
wtedy gdy D jest pierwszym dniem, w którym P procent powiatów spośród
wszystkich powiatów należących do województwa W odnotowało co najmniej
jeden przypadek śmiertelny". Myślę, że jest uniwersalne kryterium, które
co ważne, ma pewne odzwierciedlenie w modelu (początkowa ilość
zainfekowanych kasjerów) i prawdopodobnie przy nim zostanę. 

Naturalnie nasuwa się pytanie, jakie P wybrać? Proponuję takie, że
dzień D dla różnych województw będzie podobny oraz podobny będzie również
death toll w tym dniu. 
Kolejnym pytaniem jest co zrobić z powiatami, dla których nie ma danych
o pierwszym śmiertelnym przypadku (w danych, które od Pana mam,
w wielu powiatach nikt nie umarł przez cały czas gromadzenia statystyk)?
Widzę tu dwie możliwości: (1) z powiatów, wśród których szukam tych P procent 
z przypadkiem śmiertelnym, mogę wyrzucić te powiaty, dla których brakuje
danych (lub nikt nie zmarł), (2) bądź je uwzględnić. Podejście 2 jest 
moim zdaniem bardziej "fizyczne" i myślę, by to je wykorzystać.
Dodatkowo rozpatrzenie wyglądają dni D zależnie od tego, czy te powiaty
uwzględnię, czy nie pomaga zdecydować się na odpowiednią wartość P, tj.
taką, przy której ta różnica w dniach D dla każdego z województw będzie 
nieduża.

Zrobiłem więc, dla różnych wartości P, kilka wykresów.
Na osi X znajduje się województwo a na osi Y dzień D oraz death toll
w tym dniu (załącznik). Dzień D oraz death toll zaznaczyłem zarówno
uwzględniając jak i odrzucając powiaty, dla których brakuje danych.
(Wykresy starting_days_by_percent_of_touched_counties)

Moim zdaniem P na poziomie 20% daje dość podobny dzień D dla 
większości województw (12 z 16) oraz ma tę zaletę, że dzień D
prawie nie zależy od tego, czy uwzględnić/odrzucić powiaty bez 
zgonów zgłoszonych kiedykolwiek.
(Wykres starting_days_by_percent_of_touched_counties P = 20)

Ok, to początek pandemii w danym województwie mniej więcej mam. Co teraz
przyjąć za koniec początkowego etapu pandemii? Aktualnie nie mam pomysłu,
jakie kryterium tutaj przyjąć. Zostawiłem to na później i aktualnie jest 
to po prostu dzień D+80 (80 dni - czas początkowego etapu pandemii).
Jeśli ma pan jakieś propozycje, co można tu zrobić, to chętnie je sprawdzę.


## Wiadomość 2
Pierwsze próby automatycznego dopasowania parametrów modelu do danych
rzeczywistych.

Jeśli mam już wybrany zakres dni z danych rzeczywistych, dla których
chciałbym dopasować parametry modelu, tak by death toll zgadzał się możliwie
dobrze, to po pierwsze potrzebuję jakoś mierzyć tę zgodność.
Realizuję to w taki sposób, że plotuję death toll rzeczywisty oraz 
z symulacji, a następnie przesuwam rzeczywisty wzdłuż osi X i dla każdego 
przesunięcia liczę odległość między interesującym mnie zakresem death toll
z danych rzeczywistych a odpowiadającym mu przedziałem death toll z
symulacji. Myślę, że poniższy rysunek dobrze przedstawia to, co
mam na myśli. Uściślając, wybieram takie przesunięcie, które 
minimalizuje sumę kwadratów różnic między death toll rzeczywistym a 
symulowanym (na wyróżnionym fragmencie krzywej).
(Rysunek shift_łódzkie)

Jak pokazuje powyższy wykres, dla województwa łódzkiego otrzymałem całkiem
dobrą zgodność. Nie jest to przypadek, bo skoro mam już funkcję, której
podaję dane rzeczywiste i symulowane, a zwraca ona błąd dopasowania (przy 
najlepszym możliwym przesunięciu wzdłuż osi X) to nic nie stoi na
przeszkodzie, by stworzyć funkcję, której pozwalam zmieniać jeden z 
parametrów modelu (np. betę), przeprowadzać symulacje i porównując 
błędy dopasowania dla różnych bet znaleźć tę optymalną. Logika tej funkcji:
1) zrób symulacje dla: [beta_init, 1.05*beta_init, 0.95*beta_init]
2) znajdź: [error_init, error_plus, error_minus]
3) if error_plus jest najmniejszy:
   1) beta = 1.05*beta_init, error = error_plus
   2) while True:
      1) zrób symulację dla beta = 1.05*beta i znajdź nowy error_plus
      2) if nowy error_plus > error: return beta
      3) else: beta=1.05*beta, error=error_plus
4) else if error_minus jest najmniejszy: postąp analogicznie do sytuacji
opisanej wyżej.

Na wykresie wyżej (łódzkie) dane zgodziły się dosyć dobrze, co daje nadzieję, że 
takie podejście może mieć sens, jednak jeśli taką samą procedurę zastosować do
woj. lubelskiego, otrzymuję takie dopasowanie jak na wykresie poniżej.
(Rysunek shift_lubelskie)


Algorytm jakoś sobie poradził ze znalezieniem bety, która zapewni w miarę dobrą
zgodność, jednak od razu widać, że coś z tym wykresem jest nie tak. Okazuje się, że
dla tego województwa próg P = 30% (wcześniej 20% wydawało się ok) jest
jeszcze zbyt mały, bo wynika z niego, że 
do dopasowania zostanie wzięty niemal poziomy fragment wykresu death toll, czyli
zbyt wczesny etap pandemii (prawie nie eskaluje w tym czasie), skutkiem czego jest
beta w wysokości 0.012, gdzie dla woj. łódzkiego było to 0.021 (wartość, która
wydaje się o wiele bardziej realna).

Myślę, że wynika z tego, że trzeba w jakiś sposób jeszcze bardziej udoskonalić
metodę znajdowania dnia D, lub może sprawdzić, jakie daje ona wyniki dla innych
województw tj. fizyczność kształtu krzywej i wartości bet i wtedy odrzucić 
województwa najbardziej odstające, o ile nie będzie ich dużo.


# Wiadomość 3
Z nowych postępów nad pracą to by było tyle.

W ostatnim czasie byłem bardziej zajęty kwestią zaliczeń innych przedmiotów,
przez co pracę rozwijałem wolniej niż zazwyczaj.

Jeszcze w kwestii postępów nad
pracą, których nie widać w wynikach, to spędziłem ostatnio sporo czasu na
porządkowaniu kodu, zwłaszcza na opracowywaniu i prezentowaniu wyników w postaci
wykresów. W pewnym momencie doszło do takiej sytuacji, że miałem kilkanaście
funkcji plotujących dane (2000 linii), z czego część dających podobne wyniki,
mające podobne nazwy, posługujące się wewnątrz fragmentami kodu kopiuj-wklej,
a biorące różne argumenty, aż sam zacząłem się w nich gubić. Do tego pisząc te
funkcje, oczywiście myślałem, że jak dają mi wykres, którego potrzebowałem
na tamtą chwilę, to już są ok i nie będę do nich wracał, w wyniku czego większość
z nich praktycznie została bez odpowiedniej dokumentacji, a jednak często zdarzała
się później potrzeba ich edycji, na którą traciłem zdecydowanie za dużo czasu.
W związku tym, zamiast brnąć dalej w tym kierunku, zabrałem się poprawę kodu,
bym w przyszłości mógł go wygodnie dalej rozwijać.