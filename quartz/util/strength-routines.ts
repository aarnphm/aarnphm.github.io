export interface StrengthRoutine {
  title: string
  focus: string
  warmup: readonly string[]
  exercises: readonly string[]
  cooldown: string
  rounds: number
  workS: number
  restS: number
  optionalRoundRestS?: number
}

export const STRENGTH_EXERCISE_ALIASES: Readonly<Record<string, string>> = {
  'Bodyweight squat with 2s pause at bottom': 'Bodyweight Squat with 2s Pause',
  'hip flexor stretch, calf stretch, standing quad stretch':
    'Hip Flexor, Calf, and Standing Quad Stretch Flow',
  'standing calf stretch, seated forward fold, pigeon stretch each side':
    'Standing Calf, Seated Forward Fold, and Pigeon Stretch Flow',
  "child's pose, thread-the-needle stretch each side": "Child's Pose and Thread-the-Needle Stretch",
  'full-body stretch flow: hip flexors, hamstrings, calves, lower back':
    'Hip Flexor, Hamstring, Calf, and Lower Back Stretch Flow',
}

// Defaults from the five recurring Strava prescriptions; explicit descriptions take precedence.
export const STRENGTH_ROUTINES: readonly StrengthRoutine[] = [
  {
    title: 'Foundational Stability',
    focus: 'single-leg control and glute activation',
    warmup: ['Hip Circles', 'Glute Bridge', 'Walking Knee Hugs', 'Ankle Bounces'],
    exercises: [
      'Single-Leg Glute Bridge',
      'Reverse Lunge to Knee Drive',
      'Side Plank with Hip Dip',
      'Bodyweight Squat with 2s Pause',
      'Bird Dog',
    ],
    cooldown: 'Hip Flexor, Calf, and Standing Quad Stretch Flow',
    rounds: 4,
    workS: 40,
    restS: 20,
    optionalRoundRestS: 60,
  },
  {
    title: 'Posterior Chain & Hips',
    focus: 'hamstrings, glutes, calves',
    warmup: ['Glute Bridge', 'Standing Calf Raise', 'Walking Lunge', 'Hip Hinge'],
    exercises: [
      'Single-Leg Calf Raise',
      'Glute Bridge March',
      'Curtsy Lunge',
      'Wall Sit',
      'Single-Leg Romanian Deadlift',
    ],
    cooldown: 'Standing Calf, Seated Forward Fold, and Pigeon Stretch Flow',
    rounds: 4,
    workS: 40,
    restS: 20,
  },
  {
    title: 'Power & Control',
    focus: 'controlled explosive movement',
    warmup: ['Leg Swings', 'Inchworms', 'Bodyweight Squat', 'Arm Circles and Torso Twists'],
    exercises: [
      'Squat Jump',
      'Plank Shoulder Taps',
      'Split Squat with Rear Foot Elevated (or Flat)',
      'Mountain Climbers',
      'Step-Back Lunge to Balance Hold',
    ],
    cooldown: 'Figure-4 Glute Stretch, Downward Dogs, Seated Hamstring Stretch',
    rounds: 4,
    workS: 40,
    restS: 20,
  },
  {
    title: 'Core (Anti-Rotation)',
    focus: 'protects the lower back from swim rotation and run impact',
    warmup: [
      'Cat-cow',
      'Standing Torso Rotations',
      'Glute Bridge',
      'Deep Squat Hold with Gentle Rocking',
    ],
    exercises: [
      'Dead Bug',
      'Forearm Plank',
      'Single-Leg Romanian Deadlift Reach',
      'Side Plank',
      'Superman Hold with Alternating Leg Lift',
    ],
    cooldown: "Child's Pose and Thread-the-Needle Stretch",
    rounds: 4,
    workS: 40,
    restS: 20,
  },
  {
    title: 'Fatigue Endurance',
    focus: 'muscular endurance under fatigue',
    warmup: ['Marching in Place', 'Bodyweight Squat', 'Glute Bridge', 'Arm Swings'],
    exercises: [
      'Bodyweight Squat',
      'Plank Hold',
      'Alternating Reverse Lunges',
      'Glute Bridge',
      'Standing Single-Leg Balance with Knee Lift',
    ],
    cooldown: 'Hip Flexor, Hamstring, Calf, and Lower Back Stretch Flow',
    rounds: 3,
    workS: 45,
    restS: 15,
  },
]
