# Dashboard Eagle-1

Le dashboard Streamlit raconte le passage de la politique aléatoire au pilote
PPO final. Il lit directement les CSV, JSON et fichiers `EvalCallback` produits
par le notebook et les runs enregistrés par la GUI.

Il est conçu pour rester lisible par quelqu'un qui découvre le projet : chaque
onglet répond à une question formulée en français courant, chaque indicateur
porte une infobulle expliquant ce qu'il mesure, et les identifiants techniques
des expériences (`ppo_gamma_extended_selection`, `broad_t110`…) sont traduits en
libellés lisibles.

## Lancement

```bash
uv run streamlit run ASTRODYNAMICS/dashboard/app.py --server.port 8502
```

Dashboard : `http://localhost:8502`. Le port est précisé explicitement car
Streamlit prendrait sinon le port 8501, déjà utilisé par la GUI.

## Les six onglets

| Onglet | Question à laquelle il répond |
|---|---|
| **L'essentiel** | Le pilote fonctionne-t-il ? Le chiffre clé, les indicateurs de fiabilité, et une comparaison avec les points de référence. |
| **Étapes de la mission** | Comment est-on passé du hasard au pilote final ? Les six étapes dans l'ordre, avec la question posée à chacune. |
| **Comparer deux modèles** | En quoi ces deux modèles diffèrent-ils ? Chiffres alignés, distributions superposées, courbes d'apprentissage et écarts de réglages. |
| **Épisodes** | Que se passe-t-il quand ça rate ? Filtres par issue et par plage de récompense, moyenne, écart-type, médiane, min/max. |
| **Un vol en détail** | Comment se déroule un atterrissage ? Altitude, vitesse, angle, décisions et récompense accumulée. |
| **Recherche des réglages** | D'où viennent les hyperparamètres ? Essais Optuna, réglage fin de gamma, importance des paramètres, validation multi-seed. |

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
