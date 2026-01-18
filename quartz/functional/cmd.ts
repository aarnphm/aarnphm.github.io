export type Cmd<Effect> = Effect[]

export const none = <Effect>(): Cmd<Effect> => []

export const of = <Effect>(effect: Effect): Cmd<Effect> => [effect]

export const batch = <Effect>(...cmds: Cmd<Effect>[]): Cmd<Effect> => cmds.flat()
