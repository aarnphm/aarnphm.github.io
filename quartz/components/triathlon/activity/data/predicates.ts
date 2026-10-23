export const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)

export const nullableFinite = (value: unknown): value is number | null =>
  value === null || finite(value)

export const bounded = (value: unknown, minimum: number, maximum: number): value is number =>
  finite(value) && value >= minimum && value <= maximum

export const nullableBounded = (value: unknown, minimum: number, maximum: number): boolean =>
  value === null || bounded(value, minimum, maximum)

export const optionalNullableBounded = (
  value: unknown,
  minimum: number,
  maximum: number,
): boolean => value === undefined || nullableBounded(value, minimum, maximum)
