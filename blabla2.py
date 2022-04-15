ilość_powiatów = 20*20
ilość_domów_w_powiecie = 700
rozmiar_domu = 3


class Klient:
    pass


nr_domu = 0
for powiat_id in range(ilość_powiatów):
    for _ in range(ilość_domów_w_powiecie):
        nr_domu += 1
        for lokator_id in range(rozmiar_domu):
            klinet = Klient()
            stany_agnetów[nr_domu][lokator_id] = klinet.stan


