# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:59:58 2023

@author: Cristiano
"""

import numpy as np
import matplotlib.pyplot as plt

# Definire i parametri delle Gaussiane
num_gaussiane = 3
media_gaussiane = [1.0, 4.0, 7.0]
deviazione_standard_gaussiane = [0.5, 0.3, 0.8]
pesi_gaussiane = [0.4, 0.3, 0.3]

# Numero totale di campioni da generare
num_campioni = 10000

# Generare campioni dalla distribuzione mista
campioni = np.zeros(num_campioni)

for _ in range(num_campioni):
    scelta_componente = np.random.choice(num_gaussiane, p=pesi_gaussiane)
    media = media_gaussiane[scelta_componente]
    deviazione_standard = deviazione_standard_gaussiane[scelta_componente]
    campione = np.random.normal(loc=media, scale=deviazione_standard)
    campioni[_] = campione

# Plottare l'istogramma dei campioni
plt.hist(campioni, bins=50, density=True, alpha=0.5, color='b', label='Campioni')
plt.xlabel('Valore')
plt.ylabel('Densità')
plt.title('Distribuzione Mistura di Gaussiane')
plt.legend()
plt.show()