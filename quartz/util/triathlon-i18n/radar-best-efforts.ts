export const en: Record<string, string> = {
  'radar sprint bike definition':
    'Sprint uses your best measured 5 second power-to-weight effort in the past 42 days, with body weight from the effort date. The score is the percentage of the male WorldTour mean record, {benchmark} W/kg, capped at 100. A 28 day projection uses the recent W/kg trend when at least three distinct efforts are available; otherwise it holds the current score.',
  'radar sprint run definition':
    'Sprint uses your fastest recorded 30 second running speed. It shows your top speed during a short effort.',
  'radar sprint swim definition':
    'Sprint uses the fastest average speed from one recorded swim. Pool data is recorded by length, so this page does not estimate a shorter peak from within the swim.',
  'radar threshold bike definition':
    'Threshold uses FTP divided by body weight. FTP estimates the bike power you can hold for about one hour.',
  'radar threshold run definition':
    'Threshold uses the fastest running speed you can hold for a sustained effort after adjusting for hills.',
  'radar threshold swim definition':
    'Threshold uses critical swim speed. It estimates the pace you can hold during a long, steady swim from your recorded sustained efforts.',
  'radar endurance definition':
    'Endurance uses the 42 day training load for this sport. The score compares that load with the target share of your total training.',
  'radar pace swim definition':
    'Pace uses the fastest valid average pace from one swim, based on active time. Fewer seconds per 100 metres give a higher score.',
  'radar climb run definition':
    'Climb uses the vertical {unit} gained per hour of running. It counts moving time.',
  'radar climb bike definition':
    'Climb uses your best measured 20 minute power-to-weight effort in the past 42 days, with body weight from the effort date. The score is the percentage of the male WorldTour mean record, {benchmark} W/kg, capped at 100. This measures sustained climbing power. The 28 day projection uses the same W/kg scale.',
  'radar cadence bike definition':
    'Cadence compares your average pedal rate with 90 revolutions per minute. The score falls when your rate is above or below 90.',
  'radar cadence run definition':
    'Cadence compares your average step rate with 180 steps per minute. The score falls when your rate is above or below 180.',
  'radar stroke rate swim definition':
    'Stroke rate is the average from recorded swims that include both stroke counts and timing. Each rate is the stroke count divided by the time during which strokes were recorded. The target is 30 strokes per minute.',
  'radar recovery definition':
    'Recovery uses your average Oura Readiness score over the past 14 days. If Readiness is unavailable, it uses HRV.',
  'radar stride run definition':
    'Stride length is your 42 day average distance per running step. Native Apple Watch samples take priority. When those samples are unavailable, the value is estimated from speed and cadence. The score shows where the average falls within your recorded range.',
  'radar oscillation run definition':
    'Vertical oscillation is your 42 day average vertical movement per running step from Apple Watch. The score is inverted within your recorded range, so less movement plots farther from the centre. Pace, terrain, height, and running style affect the measurement.',
  'radar unit wkg definition':
    '$\\mathrm{W/kg}$ means watts per kilogram of body weight. A rider producing 270 W at 90 kg has 3.0 W/kg.',
  'radar unit ctl definition':
    'CTL is your average daily training load over 42 days. Recent days count more.',
  'radar unit fth definition':
    '$\\mathrm{ft/h}$ means vertical feet climbed per hour. This page converts metres to feet using $1\\,\\mathrm{m}=3.281\\,\\mathrm{ft}$.',
  'radar unit mh definition':
    '$\\mathrm{m/h}$ means vertical metres climbed per hour. This page divides elevation gain by moving uphill time and scales it to one hour.',
  'radar unit mspeed definition':
    '$\\mathrm{m/s}$ means metres travelled per second. Multiply the value by 3.6 to convert it to kilometres per hour.',
  'radar unit s100m definition':
    's/100 m means the seconds needed to swim 100 metres. A smaller value means a faster pace.',
  'radar unit rpm definition':
    'rpm means revolutions per minute. It measures how fast you turn the pedals.',
  'radar unit spm definition': 'spm means steps per minute. It measures your running cadence.',
  'radar unit strmin definition':
    'str/min means strokes per minute. The rate is the stroke count divided by the time during which strokes were recorded.',
  'radar unit readiness definition':
    "Readiness is Oura's daily recovery score from 0 to 100. It uses your sleep and HRV. It also uses resting heart rate and recent activity.",
  'radar unit ms definition':
    'ms means milliseconds. HRV measures the change in time between heartbeats. A value above your usual range can mean better recovery.',
  'radar unit stride definition':
    'Stride length is stored in metres per step. Imperial mode converts it to feet using 1 m = 3.281 ft.',
  'radar unit oscillation definition':
    'Vertical oscillation is stored in centimetres. Imperial mode converts it to inches using 1 in = 2.54 cm.',
  'radar unit default definition':
    'The raw value is the original measurement used to calculate this score from 0 to 100.',
  'best efforts': 'best efforts',
  'best efforts definition':
    'Distance efforts are the fastest continuous stretch of each activity, found from its distance stream and timed as elapsed time. Power efforts are mean watts over each window. Longest ride and elevation gain use whole-ride totals. Ranks cover every synced activity since the first effort date.',
  'best efforts sport': 'best efforts sport',
  'best efforts view': 'best efforts view',
  'best efforts category': 'best efforts category',
  analysis: 'analysis',
  'top 10': 'top 10',
  'all-time': 'all-time',
  'all-time rank': 'all-time',
  'best to date': 'best to date',
  PR: 'PR',
  rank: 'rank',
  'longest ride': 'longest ride',
  'filter by year': 'filter by year',
  'all years': 'all years',
  'sort efforts': 'sort',
  'newest first': 'newest first',
  'oldest first': 'oldest first',
  'high → low': 'high → low',
  'low → high': 'low → high',
  indoor: 'indoor',
  'no best efforts yet': 'no best efforts yet',
  'no efforts in the selected year': 'no efforts in the selected year',
  'no efforts in the selected period': 'no efforts in the selected period',
  'Half marathon': 'half marathon',
  HM: 'HM',
  Marathon: 'marathon',
  '1/2 mile': '0.5 mile',
  '1 mile': '1 mile',
  '2 mile': '2 mile',
  '5 mile': '5 mile',
  '10 mile': '10 mile',
  '50 mile': '50 mile',
  '100 mile': '100 mile',
}

export const fr: Record<string, string> = {
  'radar sprint bike definition':
    'Le sprint utilise ton meilleur effort mesuré sur 5 secondes en W/kg des 42 derniers jours, avec le poids connu à la date de cet effort. La note est le pourcentage du record moyen du WorldTour masculin, {benchmark} W/kg, plafonné à 100. La projection à 28 jours utilise la tendance récente en W/kg lorsque trois efforts distincts sont disponibles; sinon, elle conserve la note actuelle.',
  'radar sprint run definition':
    'Le sprint utilise ta vitesse de course la plus rapide sur 30 secondes. Il montre ta vitesse maximale pendant un effort court.',
  'radar sprint swim definition':
    "Le sprint utilise la vitesse moyenne la plus rapide d'une séance de natation. Les données de piscine sont enregistrées par longueur, donc cette page n'estime pas un pic plus court au sein de la séance.",
  'radar threshold bike definition':
    'Le seuil utilise la FTP divisée par ton poids. La FTP estime la puissance à vélo que tu peux tenir pendant environ une heure.',
  'radar threshold run definition':
    'Le seuil utilise la vitesse de course la plus rapide que tu peux tenir pendant un effort soutenu, après un ajustement pour les côtes.',
  'radar threshold swim definition':
    "Le seuil utilise la vitesse critique de natation. Elle estime l'allure que tu peux tenir pendant une longue nage régulière à partir de tes efforts soutenus.",
  'radar endurance definition':
    "L'endurance utilise la charge d'entraînement de ce sport sur 42 jours. La note compare cette charge avec la part cible de ton entraînement total.",
  'radar pace swim definition':
    "L'allure utilise la meilleure allure moyenne valide d'une séance de natation, selon le temps actif. Moins de secondes par 100 mètres donne une note plus haute.",
  'radar climb run definition':
    'La grimpe utilise les {unit} de dénivelé gagnés par heure de course. Elle compte le temps en mouvement.',
  'radar climb bike definition':
    'La grimpe utilise ton meilleur effort mesuré sur 20 minutes en W/kg des 42 derniers jours, avec le poids connu à la date de cet effort. La note est le pourcentage du record moyen du WorldTour masculin, {benchmark} W/kg, plafonné à 100. Elle mesure la puissance soutenue en montée. La projection à 28 jours utilise la même échelle en W/kg.',
  'radar cadence bike definition':
    'La cadence compare ta fréquence moyenne de pédalage avec 90 tours par minute. La note baisse lorsque ta fréquence est au-dessus ou en dessous de 90.',
  'radar cadence run definition':
    'La cadence compare ta fréquence moyenne de pas avec 180 pas par minute. La note baisse lorsque ta fréquence est au-dessus ou en dessous de 180.',
  'radar stroke rate swim definition':
    'La fréquence de nage est la moyenne des fréquences calculées pour les séances qui contiennent un nombre de coups et une durée. Chaque fréquence divise le nombre de coups par le temps pendant lequel ils ont été enregistrés. La cible est de 30 coups par minute.',
  'radar recovery definition':
    'La récupération utilise ton score Oura moyen des 14 derniers jours. Si le score Oura manque, elle utilise la VFC.',
  'radar stride run definition':
    "La longueur de foulée est ta distance moyenne par pas sur 42 jours. Les mesures natives de l'Apple Watch sont prioritaires. Lorsqu'elles manquent, la valeur est estimée avec la vitesse et la cadence. La note indique où cette moyenne se situe dans ta plage enregistrée.",
  'radar oscillation run definition':
    "L'oscillation verticale est ton mouvement vertical moyen par pas sur 42 jours, mesuré par l'Apple Watch. La note est inversée dans ta plage enregistrée, donc moins de mouvement s'affiche plus loin du centre. L'allure, le terrain, la taille et le style de course influencent la mesure.",
  'radar unit wkg definition':
    '$\\mathrm{W/kg}$ signifie watts par kilogramme de poids. Un cycliste qui produit 270 W à 90 kg a 3,0 W/kg.',
  'radar unit ctl definition':
    "La CTL est ta charge d'entraînement quotidienne moyenne sur 42 jours. Les jours récents comptent davantage.",
  'radar unit fth definition':
    '$\\mathrm{ft/h}$ signifie pieds de dénivelé par heure. Cette page convertit les mètres en pieds avec $1\\,\\mathrm{m}=3.281\\,\\mathrm{ft}$.',
  'radar unit mh definition':
    '$\\mathrm{m/h}$ signifie mètres de dénivelé par heure. Cette page divise le dénivelé par le temps de montée et ramène le résultat à une heure.',
  'radar unit mspeed definition':
    '$\\mathrm{m/s}$ signifie mètres parcourus par seconde. Multiplie la valeur par 3,6 pour la convertir en kilomètres par heure.',
  'radar unit s100m definition':
    's/100 m signifie le nombre de secondes nécessaires pour nager 100 mètres. Une valeur plus basse signifie une allure plus rapide.',
  'radar unit rpm definition':
    'rpm signifie tours par minute. Cette unité mesure la vitesse de pédalage.',
  'radar unit spm definition':
    'spm signifie pas par minute. Cette unité mesure ta cadence de course.',
  'radar unit strmin definition':
    'str/min signifie coups par minute. La fréquence divise le nombre de coups par le temps pendant lequel ils ont été enregistrés.',
  'radar unit readiness definition':
    "La préparation est le score quotidien de récupération d'Oura, de 0 à 100. Elle utilise ton sommeil et ta VFC. Elle utilise aussi ta fréquence cardiaque au repos et ton activité récente.",
  'radar unit ms definition':
    'ms signifie millisecondes. La VFC mesure la variation du temps entre les battements. Une valeur au-dessus de ta plage habituelle peut indiquer une meilleure récupération.',
  'radar unit stride definition':
    'La longueur de foulée est stockée en mètres par pas. Le mode impérial la convertit en pieds avec 1 m = 3,281 ft.',
  'radar unit oscillation definition':
    "L'oscillation verticale est stockée en centimètres. Le mode impérial la convertit en pouces avec 1 in = 2,54 cm.",
  'radar unit default definition':
    "La valeur brute est la mesure d'origine utilisée pour calculer cette note de 0 à 100.",
  'best efforts': 'meilleurs efforts',
  'best efforts definition':
    'Les efforts sur distance sont le tronçon continu le plus rapide de chaque activité, trouvé dans son flux de distance et chronométré en temps écoulé. Les efforts de puissance sont la moyenne des watts sur chaque fenêtre. La plus longue sortie et le dénivelé positif utilisent les totaux de la sortie entière. Les rangs couvrent toutes les activités synchronisées depuis la date du premier effort.',
  'best efforts sport': 'sport des meilleurs efforts',
  'best efforts view': 'vue des meilleurs efforts',
  'best efforts category': 'catégorie des meilleurs efforts',
  analysis: 'analyse',
  'top 10': '10 meilleurs',
  'all-time': 'record absolu',
  'all-time rank': 'absolu',
  'best to date': 'meilleur à ce jour',
  PR: 'RP',
  rank: 'rang',
  'longest ride': 'plus longue sortie',
  'filter by year': 'filtrer par année',
  'all years': 'toutes les années',
  'sort efforts': 'trier',
  'newest first': 'plus récents',
  'oldest first': 'plus anciens',
  'high → low': 'haut → bas',
  'low → high': 'bas → haut',
  indoor: 'intérieur',
  'no best efforts yet': 'aucun meilleur effort',
  'no efforts in the selected year': "aucun effort pour l'année choisie",
  'no efforts in the selected period': 'aucun effort pour la période choisie',
  'Half marathon': 'semi-marathon',
  HM: 'SM',
  Marathon: 'marathon',
  '1/2 mile': '0,5 mille',
  '1 mile': '1 mille',
  '2 mile': '2 milles',
  '5 mile': '5 milles',
  '10 mile': '10 milles',
  '50 mile': '50 milles',
  '100 mile': '100 milles',
}
