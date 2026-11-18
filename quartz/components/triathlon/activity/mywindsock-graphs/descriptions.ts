// Reader help per native graph key. Sign and angle conventions were checked against the archived
// series: Head/Tail Wind reads 360° where air speed exceeds ground speed, and wImpact% is positive there.
export const GRAPH_DESCRIPTIONS: Record<string, string> = {
  interval_designer:
    'Power through the ride with the CdA and total weight that the model uses. myWindsock uses this view to plan target power for course sections.',
  delta_compare:
    'Time difference in seconds against your rides before the last recorded performance change, such as a new position or new equipment.',
  delta_compare_avg: 'Spread of the time difference through the ride.',
  weather:
    'Temperature, precipitation and air density at your position and time. Denser air increases drag at the same speed.',
  windspeed:
    'Wind speed and gusts along the route at the standard 10 m measurement height. Wind at rider height is lower.',
  virt_elev:
    'Your elevation profile with the wind changed into climbing. A headwind adds height and a tailwind removes it. The gap to the real profile is the cost of the wind as climbing.',
  virt_grade:
    'Road gradient adjusted for the wind, next to the actual gradient. The gap shows how much steeper or flatter the wind made the road feel.',
  '3dcourse_virt': 'Elevation profile that the 3D route view uses.',
  airdist_acc:
    'Extra distance of air that you rode through, added up over the ride. The line rises in headwinds and falls in tailwinds. The last value is how much longer the ride was in air than on the road.',
  effective:
    'Speed of the air against you: ground speed plus the headwind part of the wind at rider height. Aerodynamic drag follows this speed.',
  sidewind:
    'Part of the wind that crosses your direction of travel. A strong crosswind pushes the bike sideways and increases yaw.',
  groundspd: 'Speed over the road from the recorded activity.',
  ground_dist: 'Minutes spent in each ground speed band.',
  diff: 'Air speed minus ground speed. Above zero is a net headwind. Below zero is a net tailwind.',
  direction:
    'Wind angle relative to your direction of travel. 360° is a direct headwind, 270° is a direct crosswind and 180° is a direct tailwind.',
  rollingavg:
    'Moving average of ground speed. It removes short changes so that the trend is clear.',
  wwatts:
    'Percent of your power that the wind added or removed at each moment. Above zero means that the wind made you work harder.',
  yaw: 'Angle between your direction of travel and the air that hits you. 0° is air from straight ahead. Crosswind makes the angle larger.',
  yawdist:
    'Time spent at each yaw angle. Compare it with the yaw range in wind tunnel data for wheels and frames.',
  cda: 'Drag area (CdA, m²) that the model calculates from power, speed, wind and gradient. Lower is more aerodynamic. CdA is the ride average and Live CdA is the estimate at each moment. Where available, Test Average and the shaded Test Range show the provider’s aero-test results.',
  cda_dist: 'Time spent at each CdA value.',
  brake:
    'Places where the model detects braking: the bike slowed more than power, gradient and drag can explain.',
  kj_acc:
    'Mechanical work that you did, added up over the ride in kilojoules. On a bike, 1 kJ of work is approximately 1 kcal of food energy.',
  power: 'Power output through the ride.',
  rollingavg_power: 'Moving average of power. It removes short surges so that the trend is clear.',
  pdc: 'Best average power that you held for each duration in this ride, from short sprints on the left to the full ride on the right.',
  wprime:
    'W′ balance: the energy above Critical Power that you have left, in joules. It falls when you ride above Critical Power and refills below it. Near zero means that you are close to exhaustion.',
  pcp: 'Moving average of power that gives more weight to hard efforts, as Normalized Power does. It shows the physiological cost of a variable effort.',
  grade: 'Road gradient through the ride, in percent.',
  gradient_dist: 'Minutes spent in each gradient band.',
  watts:
    'Share of the resistance from rolling, gravity, acceleration and air at each moment. When air resistance is the largest part, aerodynamics matter more than power to weight.',
}
