import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

def area_interseccion_circulos(x1, y1, r1, x2, y2, z1, r2):
    d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if d >= r1 + r2:
        return 0
    elif d <= abs(r1 - r2) and z1 < 0:
        return np.pi * (r2**2)
    elif d <= abs(r1 - r2) and z1 >= 0:
        return 0
    else:
        a = (r1**2 - r2**2 + d**2) / (2 * d)
        h = np.sqrt(r1**2 - a**2)
        term1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
        term2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
        term3 = 0.5 * np.sqrt((-d + r1 + r2)*(d + r1 - r2)*(d - r1 + r2)*(d + r1 + r2))
        return term1 + term2 - term3

def main():
    st.title("🌑 Simulación de Tránsito de Exoplaneta en Tiempo Real")
    st.sidebar.header("Parámetros de Simulación")
    # Parámetros
    Radio_star = 10
    Rpf = st.sidebar.slider("Radio del planeta / Radio estrella", 0.01, 0.4, 0.1, 0.01)
    Radio_planet = Rpf * Radio_star

    R_orbf = st.sidebar.slider("Radio orbital", 2.0, 10.0, 5.0, 0.1)
    Orbita = R_orbf * Radio_star

    Angulo_inclinacion = st.sidebar.slider("Inclinación (grados)", -90, 90, 0, 1)
    Inclinacion = np.radians(90 + Angulo_inclinacion)

    Pasos = 150
    Caja = 1.5 * Orbita

    placeholder = st.empty()

    brillo = []
    for frame in range(Pasos):
        # Cálculos orbitales
        theta = 2 * np.pi * frame / Pasos
        x = np.cos(theta) * Orbita
        y = np.sin(theta) * Orbita * np.cos(Inclinacion)
        z = np.sin(theta) * Orbita

        # Cálculo de brillo
        if z > 0:
            brillo_val = 100
        else:
            inter = area_interseccion_circulos(0, 0, Radio_star, x, y, z, Radio_planet)
            brillo_val = 100 * (1 - inter / (np.pi * Radio_star**2))
        brillo.append(brillo_val)

        # Figura
        fig, axs = plt.subplots(2, 1, figsize=(8, 10))

        # Subplot 1: órbita
        axs[0].set_xlim(-Caja, Caja)
        axs[0].set_ylim(-Caja, Caja)
        axs[0].set_aspect('equal')
        axs[0].set_title("Órbita del planeta")
        estrella = Circle((0, 0), Radio_star, color='yellow')
        planeta = Circle((x, y), Radio_planet, color='red')
        axs[0].add_patch(estrella)
        axs[0].add_patch(planeta)

        # Subplot 2: curva de luz
        axs[1].set_xlim(0, Pasos)
        axs[1].set_ylim(min(brillo) - 0.5, 101)
        axs[1].set_title("Curva de luz simulada")
        axs[1].set_xlabel("Tiempo (frames)")
        axs[1].set_ylabel("Brillo (%)")
        axs[1].plot(brillo, color='red')
        axs[1].scatter(frame, brillo_val, color='blue', zorder=5)

        placeholder.pyplot(fig)
        time.sleep(0.03)  # Velocidad de animación

if __name__ == "__main__":
    main()
