import type { Gloss } from './types'

export const en: Record<string, Gloss> = {
  strokerate: {
    term: 'stroke rate',
    def: 'Stroke rate counts arm strokes per minute while you are swimming. Swim speed is stroke rate multiplied by the distance you cover per stroke, so a higher rate only helps when your distance per stroke holds. This chart averages the rate over each length or each 100 m block.',
  },
  swimcadence: {
    term: 'swim cadence',
    def: 'Swim cadence counts the strokes it takes to cover one pool length. A lower count means you cover more distance with each stroke. Read it next to stroke rate and pace: fewer strokes per length with pace holding is a real efficiency gain, while fewer strokes with pace slipping usually means you are gliding. Only pool swims report it, and the count depends on the pool length.',
  },
  swolf: {
    term: 'SWOLF',
    def: 'SWOLF is swim golf. It adds the seconds a length took to the strokes that length used, so a lower score is better. Speed and efficiency both move it, which is why it drops when either improves. Compare it only within the same pool length, since a 25 m score and a 50 m score are not on the same scale.',
  },
  overreaching: {
    term: 'overreaching',
    def: 'Overreaching is flagged when HRV is low while training load rises quickly. Here, low HRV means at least one standard deviation below your baseline. A fast load increase means a high ACWR or a weekly ramp above 10%.',
  },
  oreadiness: {
    term: 'Oura Readiness',
    def: 'Oura Readiness is a daily score from 0 to 100. A score of 85 or higher is optimal. A score from 70 to 84 is good. A score below 70 calls for attention. Several days below 70 can point to accumulated strain.',
  },
  vo2max: {
    term: 'VO₂max',
    def: 'VO₂max is the maximum amount of oxygen your body can use during hard exercise. It is measured in millilitres per kilogram per minute. The 28 day trend uses observed values from the latest measurement method over at most 12 weeks. The cycling estimate uses FTP and body weight. A lab result sets its baseline when the lab also recorded body weight.',
  },
  ftp: {
    term: 'FTP hypothesis',
    def: 'FTP is the cycling power you may be able to hold for about one hour. This estimate comes from a treadmill VO₂max test, so it crosses from running to cycling and has low confidence. Use your heart rate zones until you complete a bike power test.',
  },
  fitage: {
    term: 'fitness age',
    def: 'Fitness age is the age with the same median VO₂max in the male FRIEND reference data. This chart limits the result to ages 20 to 80. A lower fitness age means your VO₂max is above the median for your calendar age.',
  },
  vam: {
    term: 'VAM',
    def: 'VAM is the vertical distance you climb in one hour. This page calculates it from elevation gain and moving time. It shows metres per hour or feet per hour based on your unit setting.',
  },
  radar: {
    term: 'abilities',
    def: 'Each sport has six scores from 0 to 100. Sprint and threshold use power or speed. Endurance uses 42 days of training load. Run uses stride length, cadence, and vertical oscillation. Bike uses 5 second sprint and 20 minute climbing power in W/kg relative to male WorldTour mean records, plus cadence. Swim uses pace and stroke rate. Bike and swim retain recovery. The dashed line shows the projected score 28 days from now.',
  },
  hrzones: {
    term: 'training zone distributions',
    def: 'This panel adds recorded time in each training zone across the selected sport and date range. Cycling switches between heart rate and seven FTP-based power zones. Running switches between heart rate and six fixed pace zones derived from the 10 km race time in ATHLETE. Smoothed running stream samples are adjusted for grade before they are assigned to a zone. Swimming uses heart rate.',
  },
  activitytelemetry: {
    term: 'activity telemetry',
    def: 'Each point is one activity. Power keeps measured and estimated values distinct. Cadence uses strokes per minute for swimming, revolutions per minute for cycling, and steps per minute for running. Skin temperature and Heat Strain Index are duration-weighted CORE observations. Missing sensor readings remain gaps.',
  },
  ef: {
    term: 'efficiency factor',
    def: 'Efficiency factor compares your pace or power with your heart rate. On the bike, it divides normalized power by average heart rate. Normalized power gives more weight to hard efforts. On the run, the calculation divides speed adjusted for hills by average heart rate. A higher value at the same effort means you are producing more output per heartbeat.',
  },
  decouple: {
    term: 'decoupling',
    def: 'Decoupling compares the pace or power you produced per heartbeat in the first and second half of a workout. A lower value is better. A value under 5% means you stayed steady. A value over 10% means you faded in the second half.',
  },
  legdist: {
    term: 'total distance',
    def: 'Total distance is the distance covered in this sport during the selected season. Swim distance is shown in metres. Bike and run distance use the kilometre or mile setting.',
  },
  legcount: {
    term: 'sessions',
    def: 'This is the number of workouts logged for this sport during the selected season.',
  },
  legtime: { term: 'total time', def: 'Total time adds the moving time from these workouts.' },
  herodist: {
    term: 'season total',
    def: 'Season total adds the distance from all three sports during the selected season. It uses the kilometre or mile setting.',
  },
}

export const fr: Record<string, Gloss> = {
  strokerate: {
    term: 'fréquence de nage',
    def: 'La fréquence de nage compte les coups de bras par minute pendant que tu nages. La vitesse en natation est la fréquence multipliée par la distance parcourue à chaque coup, donc une fréquence plus élevée ne sert que si la distance par coup tient. Ce graphique fait la moyenne sur chaque longueur ou chaque bloc de 100 m.',
  },
  swimcadence: {
    term: 'cadence de nage',
    def: "La cadence de nage compte les coups nécessaires pour couvrir une longueur de bassin. Moins de coups veut dire plus de distance parcourue à chaque coup. Lis-la avec la fréquence et l'allure : moins de coups par longueur en gardant l'allure est un vrai gain d'efficacité, alors que moins de coups en perdant de l'allure veut souvent dire que tu glisses. Seules les séances en bassin la rapportent, et le compte dépend de la longueur du bassin.",
  },
  swolf: {
    term: 'SWOLF',
    def: "SWOLF veut dire swim golf. Il additionne les secondes mises pour une longueur et les coups utilisés pour cette longueur, donc un score plus bas est meilleur. La vitesse et l'efficacité le font bouger toutes les deux, et il baisse dès que l'une des deux progresse. Compare-le seulement à longueur de bassin égale, car un score en 25 m et un score en 50 m ne sont pas sur la même échelle.",
  },
  overreaching: {
    term: 'surmenage',
    def: 'Le surmenage est signalé lorsque la VFC est basse pendant que la charge augmente rapidement. Ici, une VFC basse signifie au moins un écart type sous ta référence. Une hausse rapide signifie un ACWR élevé ou une progression hebdomadaire au-dessus de 10 %.',
  },
  oreadiness: {
    term: 'préparation Oura',
    def: 'La préparation Oura est une note quotidienne de 0 à 100. Une note de 85 ou plus est optimale. Une note de 70 à 84 est bonne. Une note sous 70 demande ton attention. Plusieurs jours sous 70 peuvent signaler une contrainte accumulée.',
  },
  vo2max: {
    term: 'VO₂max',
    def: "La VO₂max est la quantité maximale d'oxygène que ton corps peut utiliser pendant un effort intense. Elle est mesurée en millilitres par kilogramme par minute. La tendance sur 28 jours utilise les valeurs observées de la méthode la plus récente sur au plus 12 semaines. L'estimation à vélo utilise la FTP et le poids. Un résultat de laboratoire fixe sa référence lorsque le poids a aussi été mesuré.",
  },
  ftp: {
    term: 'hypothèse FTP',
    def: "La FTP est la puissance à vélo que tu pourrais tenir pendant environ une heure. Cette estimation vient d'un test de VO₂max sur tapis, donc elle passe de la course au vélo et reste peu fiable. Utilise tes zones de fréquence cardiaque jusqu'à ton prochain test de puissance à vélo.",
  },
  fitage: {
    term: 'âge physiologique',
    def: "L'âge physiologique est l'âge qui a la même VO₂max médiane dans les données de référence FRIEND pour les hommes. Ce graphique limite le résultat de 20 à 80 ans. Un âge physiologique plus bas signifie que ta VO₂max dépasse la médiane de ton âge réel.",
  },
  vam: {
    term: 'VAM',
    def: "La VAM est le dénivelé grimpé en une heure. Cette page la calcule avec le dénivelé et le temps en mouvement. Elle affiche des mètres par heure ou des pieds par heure selon ton choix d'unités.",
  },
  radar: {
    term: 'aptitudes',
    def: "Chaque sport a six notes de 0 à 100. Le sprint et le seuil utilisent la puissance ou la vitesse. L'endurance utilise 42 jours de charge d'entraînement. La course utilise la longueur de foulée, la cadence et l'oscillation verticale. Le vélo utilise la puissance de sprint sur 5 secondes et de grimpe sur 20 minutes en W/kg par rapport aux records moyens du WorldTour masculin, ainsi que la cadence. La natation utilise l'allure et la fréquence de nage. Le vélo et la natation conservent la récupération. La ligne pointillée montre la note prévue dans 28 jours.",
  },
  hrzones: {
    term: "répartition des zones d'entraînement",
    def: "Ce panneau additionne le temps enregistré dans chaque zone pour la discipline et la période choisies. Le vélo alterne entre la fréquence cardiaque et sept zones de puissance basées sur la FTP. La course alterne entre la fréquence cardiaque et six zones d'allure fixes dérivées du temps sur 10 km dans ATHLETE. Les échantillons lissés sont ajustés selon la pente avant leur attribution à une zone. La natation utilise la fréquence cardiaque.",
  },
  activitytelemetry: {
    term: 'télémétrie des activités',
    def: "Chaque point représente une activité. La puissance distingue les valeurs mesurées des estimations. La cadence utilise les coups par minute en natation, les tours par minute à vélo et les pas par minute en course. La température cutanée et l'indice de contrainte thermique sont des moyennes CORE pondérées par la durée. Les mesures manquantes restent des espaces vides.",
  },
  ef: {
    term: "facteur d'efficacité",
    def: "Le facteur d'efficacité compare ton allure ou ta puissance avec ta fréquence cardiaque. À vélo, il divise la puissance normalisée par la fréquence cardiaque moyenne. La puissance normalisée donne plus de poids aux efforts intenses. En course, le calcul divise la vitesse ajustée pour les côtes par la fréquence cardiaque moyenne. Une valeur plus haute au même effort signifie plus de rendement par battement.",
  },
  decouple: {
    term: 'découplage',
    def: "Le découplage compare l'allure ou la puissance produite par battement entre les deux moitiés de la séance. Une valeur plus basse est meilleure. Une valeur sous 5 % signifie que l'effort est resté stable. Une valeur au-dessus de 10 % signifie une baisse pendant la seconde moitié.",
  },
  legdist: {
    term: 'distance totale',
    def: 'La distance totale est la distance parcourue dans ce sport pendant la saison choisie. La natation est affichée en mètres. Le vélo et la course utilisent le réglage en kilomètres ou en milles.',
  },
  legcount: {
    term: 'séances',
    def: 'Ce nombre est le total des séances enregistrées pour ce sport pendant la saison choisie.',
  },
  legtime: {
    term: 'temps total',
    def: 'Le temps total additionne le temps en mouvement de ces séances.',
  },
  herodist: {
    term: 'total saison',
    def: 'Le total saison additionne la distance des trois sports pendant la saison choisie. Il utilise le réglage en kilomètres ou en milles.',
  },
}
