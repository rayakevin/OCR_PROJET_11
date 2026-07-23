# Dashboard Eagle-1

Instrument de suivi des expériences de la mission. Il lit directement les CSV,
JSON et fichiers `EvalCallback` produits par le notebook et les runs de la GUI.
Les identifiants techniques (`ppo_gamma_extended_selection`, `broad_t110`…) sont
traduits en libellés lisibles ; le reste reste factuel, sans habillage
pédagogique — les commentaires de lecture sont laissés à la présentation orale.

## Lancement

```bash
uv run streamlit run ASTRODYNAMICS/dashboard/app.py --server.port 8502
```

Dashboard : `http://localhost:8502`. Le port est précisé explicitement car
Streamlit prendrait sinon le port 8501, déjà utilisé par la GUI.

## Les six onglets

| Onglet | Contenu |
|---|---|
| **Synthèse** | Indicateurs finaux (moyenne, σ, médiane, réussite, pire épisode, longueur, carburant) et récompense moyenne par politique. |
| **Apprentissage** | Meilleur résultat par phase, et courbes `EvalCallback` filtrables par expérience. |
| **Comparaison** | Deux évaluations alignées : deltas, distributions superposées, apprentissage comparé, table d'hyperparamètres. |
| **Épisodes** | Filtres par issue et plage de récompense ; moyenne, σ, médiane, min/max recalculés sur la sélection. |
| **Trajectoire** | Télémétrie pas à pas (altitude, vitesse, angle), distribution des actions, récompense cumulée. |
| **Optuna** | Essais TPE, raffinement de gamma, importance PED-ANOVA, validation multi-seed. |

## Choix de lisibilité

**Emphase plutôt qu'arc-en-ciel.** Sur les vues de comparaison, une seule série
porte l'information et le reste passe en gris. Un graphique où chaque barre a sa
couleur oblige à faire des allers-retours avec la légende.

**Palette validée, pas choisie à l'œil.** Les couleurs passent les contrôles de
`validate_palette.js` : bande de luminosité, seuil de chroma, séparation en
vision daltonienne et contraste sur le fond. Conséquence concrète : le couple
vert/rouge habituel pour réussite/échec a été **écarté des graphiques**, car sa
séparation en deutéranopie ne vaut que ΔE 4,1 — deux séries indistinguables pour
une partie des lecteurs. Le couple bleu/orange retenu mesure 24,7. Le vert et le
rouge ne servent plus que dans les textes, toujours avec une icône.

**Le seuil de 200 est partout.** Chaque graphique de récompense porte la même
ligne pointillée, pour que la question « est-ce au-dessus de l'objectif ? » se
lise sans calcul.

**Pas de chiffre sur chaque barre.** Quand une barre porte déjà sa moustache
d'écart-type, y ajouter une étiquette crée une collision. L'axe situe, le survol
donne la valeur exacte.

## Tests

```bash
uv run pytest ASTRODYNAMICS/dashboard/tests -q
```

Les tests vérifient que les sources de données attendues existent et ont le bon
schéma. Le rendu lui-même est contrôlé avec le harnais `streamlit.testing`, qui
exécute l'application et remonte toute exception.
