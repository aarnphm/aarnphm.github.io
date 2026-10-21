import type { TriathlonMaintenance } from '../../../util/triathlon-maintenance'
import type { TriathlonRenderData } from '../../triathlon/render-data'
import { ShortcutHint } from '../../triathlon/shell/ShortcutHint'
import { EquipmentUsage } from '../../triathlon/tools/EquipmentUsage'
import { Maintenance } from '../../triathlon/tools/Maintenance'
import { PacePanel } from '../../triathlon/tools/PaceChart'

export const CONVERSIONS: [string, string][] = [
  ['pace', '/100m × 16.09 → /mi'],
  ['dist', 'km × 0.621 → mi'],
]

export const GEAR: [string, string[]][] = [
  [
    'Cervélo Soloist',
    [
      'Size: 56, Black, 7.39 kgs',
      'Fork: Cervélo All-Carbon, Tapered Soloist',
      'Handlebar: Cervélo HB13 Carbon, 31.8mm clamp',
      'Handlebar Sizing: Size 56 - 40cm',
      'Handlebar: Enve SES AR In-Route One-Piece Handlebar',
      'Stem: Cervélo ST36 Alloy',
      'Stem Sizing: Size 56 - 100mm',
      'Seatpost: Cervélo SP27 Carbon',
      'Saddle: Prologo Nago R4 PAS Tirox Lightweight',
      'Saddle: Selle Italia SLR 3D Elite L3 145mm',
      'Bottom Bracket: FSA, T47 BBright for 24mm spindle',
      'Bottom Bracket: CeramicSpeed T47 BBright for Shimano',
      'Headset: FSA IS2 1-1/4, 45° x 45° / 1-1/2, 36° x 45°',
      'Cervélo Aero Thru Axle Front, M12x1.5mm, 127mm length',
      'Cervélo Aero Thru Axle Rear, M12x1.5mm, 170.5mm length',
      'Front Wheel: Reserve 40, 12x100mm, 24H, centerlock, tubeless compatible',
      'Rear Wheel: Reserve 44, 12x142mm, 24H, centerlock, tubeless compatible',
      'Front Wheel: Reserve 42TA, DT Swiss 350, 12x100mm, 24H, centerlock, tubeless compatible',
      'Rear Wheel: Reserve 49TA, DT Swiss 350, 12x142mm, HG freehub 24H, centerlock, tubeless compatible',
      'Front Wheel: HUNT 54_58 Aerodynamicist UD Carbon Spoke',
      'Rear Wheel: HUNT 54_58 Aerodynamicist UD Carbon Spoke',
      'Tube: Pirelli P Zero TPU',
      'Tires: Pirelli P Zero Race SL-R 700x28c',
      'Front Tire: Pirelli P Zero Race TLR SL-R 700x28c',
      'Rear Tire: Pirelli P Zero Race TLR RS 700x28C',
      'Tires: Vittoria Corsa N.EXT TLR G2.0 700x29c',
      'Shifter/Break: Shimano Ultegra, R8170',
      'Crankset: Shimano Ultegra, R8100, 52/36T',
      'Chain: Shimano M8100',
      'Chain: CeramicSpeed UFO optimized chain, 12-speed (#4); base-chain model and UFO variant unrecorded',
      'Chainring: CarbonTi 2x, 54-40T',
      'Pulley Wheel: CeramicSpeed OSPW RS 5 Spoke',
      'Cassette: Shimano Ultegra, R8100, 11-34T, 12-Speed',
      'Front Derailleur: Shimano Ultegra, R8150',
      'Rear Derailleur: Shimano Ultegra Di2 with CeramicSpeed OSPW RS 5 Spoke',
      'Brake Rotors: Shimano CL800 Centerlock',
      'Front/Rear Brake Pads: BRAKCO Airpads Graphene Venting Disc Brake Pads',
      'Powermeter: Magene P715 S Pedal',
      'Bike Computer: Garmin Edge 1050',
      'HR monitor: Garmin HRM 600',
      'Radar: Garmin Varia RTL515',
      'Scale: Garmin Index S2',
      'Shoes: SPECIALIZED TORCH 2.0',
      'Socks: DANISH ENDURANCE Aero Socks',
      'Helmet: KASK Valegro',
    ],
  ],
  [
    'Canyon Speedmax CFR Di2 2027',
    [
      'Size: L, Pro Black, 9.3 kg',
      'Frame: Canyon Speedmax CFR, CFR carbon, 12x142mm',
      'Fork: Canyon Speedmax, CF carbon, 12x100mm',
      'Cockpit: Canyon AeroShield Pro (forearm length 395mm, shell size 230mm)',
      'Cockpit Post: Long-Mid',
      'Hydration: Canyon AeroModule FUEL 650 + BTC Beam',
      'Seatpost: Canyon Splitter Plate Pro, carbon w/ XLab Bottle Cage',
      'Saddle: Fizik Transiro Aeris LD R1 Adaptive',
      'Bottom Bracket: Shimano Pressfit BB92',
      'Front Wheel: DT Swiss ARC 1100 85 mm',
      'Rear Wheel: DT Swiss ARC 1100 Disc',
      'Front Tire: Continental Aero 111, 700x26c',
      'Rear Tire: Continental GP5000 TT TR, 700x28c',
      'Shifters: Shimano Dura-Ace Di2 R9160 Remote TT',
      'Crankset: Shimano Dura-Ace R9200, 165mm, 54/40T',
      'Chainring: Carbon-Ti 1x, 58T',
      'Chain: Shimano XTR',
      'Cassette: Shimano Dura-Ace CS-R9200, 11-34T, 12-Speed',
      'Front Derailleur: Shimano Dura-Ace Di2 FD-R9250',
      'Rear Derailleur: Shimano Dura-Ace Di2 RD-R9250',
      'Shift/Brake Levers: Shimano Dura-Ace Di2 ST-R9180',
      'Brake Rotors: Shimano Dura-Ace RT-CL900, 160mm front / 140mm rear',
      'Powermeter: Shimano Dura-Ace R9200, dual-sided',
      'Bottle: Canyon FUEL Aero Bottle',
      'Bottle Cage: XLAB Gorilla XT Carbon',
      'Pedals: Assioma PRO RL-2',
      'Bike Computer: Wahoo ELEMNT BOLT',
      "Shoes: Fi'zi:k Transiro Hydra Aeroweave Carbone",
      'Helmet: MET Drone Wide Body III',
    ],
  ],
  [
    'running',
    [
      'Shoes: HOKA Clifton 10',
      'Shoes: Saucony Endorphin Elite 3',
      'Shoes: On Cloudmonster 3 Hyper LS',
      'Hat: Ciele Athletic Gocap',
      'Socks: Saucony Inferno Cushion Mid 3-Pack Sock',
      'Pants: Salomon SHAKEOUT CORE 5',
      'Utilities: Zone3 Ultimate Race Number Belt',
      'Utilities: Salomon HIGH PULSE',
      'Utilities: Salomon SOFT FLASK 150ml/5oz 28',
    ],
  ],
  [
    'swim',
    [
      'Suit: 2XU Trisuit',
      'Suit: 2XU Wetsuit',
      'Goggles: Decathlon Anti-fog Swimming Goggles',
      'Goggles: Speedo Unisex Adult Swim Goggles Hydrospex Classic',
      'Cap: Speedo Unisex Adult Swim Cap Silicone',
      "Pants: Speedo Speedo Men's Swimsuit Endurance+",
      'Utilities: Speedo Ergo Ear Plug',
    ],
  ],
  [
    'wearables',
    [
      'Oura Ring 4',
      'Apple Watch Ultra 3',
      'Garmin Forerunner 970',
      'Apple AirPod Pro 3',
      'SHOKZ OpenRun Pro 2-Bone Conduction Headphones',
    ],
  ],
  [
    'fuel',
    [
      'mandarins',
      'apple',
      'banana',
      'cherry tart juice',
      'Nuun Hydration',
      'Precision Fuel & Hydration Chews, Gels, Carb Drinks',
      'Xact Energy Bars & Gels',
      'Maurten Gels & Carb Drinks',
      'Skratch Labs Super High-Carb Hydration Powder',
    ],
  ],
]

const BIKE_GEAR_GROUP_COUNT = 2

const STRAVA_GEAR_IDS: Readonly<Record<string, string>> = {
  'Cervélo Soloist': 'b18037537',
  'Canyon Speedmax CFR Di2 2027': 'b18595115',
  'Shoes: HOKA Clifton 10': 'g31765417',
  'Shoes: Saucony Endorphin Elite 3': 'g32047204',
  'Shoes: On Cloudmonster 3 Hyper LS': 'g34180562',
}

const GearRows = ({
  groups,
  equipment,
}: {
  groups: ReadonlyArray<readonly [string, readonly string[]]>
  equipment: TriathlonRenderData['equipment']
}) => (
  <>
    {groups.map(([label, items]) => (
      <div class="tri-gear-row">
        <span class="tri-gear-k" data-i18n={label}>
          {label}
        </span>
        <span class="tri-gear-v">
          <EquipmentUsage usage={equipment?.[STRAVA_GEAR_IDS[label]]} />
          {items.map(it => (
            <span class="tri-gear-li">
              · {it}
              <EquipmentUsage usage={equipment?.[STRAVA_GEAR_IDS[it]]} />
            </span>
          ))}
        </span>
      </div>
    ))}
  </>
)

export const GearPanel = ({
  page,
  maintenance = null,
  equipment,
}: {
  page?: boolean
  maintenance?: TriathlonMaintenance | null
  equipment?: TriathlonRenderData['equipment']
}) => (
  <div class="tri-gear-wrap">
    {!page && (
      <button
        class="tri-gear-btn tri-key-anchor"
        type="button"
        aria-expanded="false"
        aria-controls="tri-gear-panel"
      >
        <span data-i18n="gear">gear</span>
        <ShortcutHint>g g</ShortcutHint>
      </button>
    )}
    <div
      id={page ? undefined : 'tri-gear-panel'}
      class="tri-gear"
      aria-hidden={page ? 'false' : 'true'}
    >
      <div class="tri-gear-scroll">
        <GearRows groups={GEAR.slice(0, BIKE_GEAR_GROUP_COUNT)} equipment={equipment} />
        <Maintenance maintenance={maintenance} />
        <GearRows groups={GEAR.slice(BIKE_GEAR_GROUP_COUNT)} equipment={equipment} />
      </div>
    </div>
  </div>
)

export const ToolsPanel = ({
  maintenance,
  renderData,
}: {
  maintenance?: TriathlonMaintenance | null
  renderData?: TriathlonRenderData
}) => (
  <div class="tri-tools">
    <div class="tri-ana-bar">
      <div class="tri-tools-tabs" role="tablist" aria-label="tools" data-i18n-aria-label="tools">
        <button
          id="tri-tools-tab-tools"
          class="tri-tools-tab"
          type="button"
          role="tab"
          aria-selected="true"
          aria-controls="tri-tools-panel-tools"
          data-tools-tab="tools"
          data-i18n="tools"
        >
          tools
        </button>
        <button
          id="tri-tools-tab-pace-chart"
          class="tri-tools-tab"
          type="button"
          role="tab"
          aria-selected="false"
          aria-controls="tri-tools-panel-pace-chart"
          data-tools-tab="pace-chart"
          data-i18n="pace chart"
          tabindex={-1}
        >
          pace chart
        </button>
      </div>
    </div>
    <section
      id="tri-tools-panel-tools"
      class="tri-tools-sec"
      role="tabpanel"
      aria-labelledby="tri-tools-tab-tools"
      data-tools-panel="tools"
      data-keyboard-scroll
      tabindex={0}
    >
      <GearPanel page maintenance={maintenance} equipment={renderData?.equipment} />
    </section>
    <section
      id="tri-tools-panel-pace-chart"
      class="tri-tools-sec"
      role="tabpanel"
      aria-labelledby="tri-tools-tab-pace-chart"
      data-tools-panel="pace-chart"
      data-keyboard-scroll
      tabindex={0}
      hidden
    >
      <PacePanel page />
    </section>
  </div>
)
