import { z } from 'zod'

const dependencySetSchema = z.object({
  files: z.array(z.string()),
  transclusions: z.array(z.object({ source: z.string(), consumer: z.string() })),
})
const ownerSchema = z.object({
  epoch: z.string(),
  mode: z.enum(['replace', 'patch']),
  dependencies: z.array(z.string()),
  outputs: z.array(z.string()),
  externalOutputs: z.array(z.string()),
})
const fileSchema = z.object({
  owners: z.array(z.string()),
  size: z.number().nonnegative(),
  sha256: z.string().regex(/^[a-f0-9]{64}$/),
  identity: z.string(),
  observed: z.boolean(),
  asset: z.object({ owner: z.string(), source: z.string() }).optional(),
})

export const outputReceiptSchema = z.object({
  version: z.literal(1),
  outputRoot: z.string(),
  inputRoot: z.string(),
  epoch: z.string(),
  validation: z.literal('full-tree-comparison'),
  reusable: z.literal(false),
  dependencyScope: z.literal('conservative-build-inputs'),
  dependencySets: z.record(z.string(), dependencySetSchema),
  inputIdentities: z.record(z.string(), z.string()),
  owners: z.record(z.string(), ownerSchema),
  files: z.record(z.string(), fileSchema),
  changed: z.array(z.string()),
  removed: z.array(z.object({ path: z.string(), owners: z.array(z.string()) })),
  unattributed: z.array(z.string()),
  overlaps: z.array(z.string()),
  preserved: z.array(z.string()),
})

export type OutputReceipt = z.infer<typeof outputReceiptSchema>
export type OutputDependencySet = z.infer<typeof dependencySetSchema>
export type OutputOwner = z.infer<typeof ownerSchema>

export type OutputReceiptTransaction = {
  outputRoot: string
  inputRoot: string
  stage: string
  epoch: string
  previous?: OutputReceipt
  owners: Map<string, OutputOwner>
  dependencySets: Map<string, OutputDependencySet>
  contentDependencies: WeakMap<object, string>
  observed: Set<string>
  assets: Map<string, { owner: string; source: string }>
}
