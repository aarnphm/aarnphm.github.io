import "./MethodologyTree"
import "./Tractatus"

export type { QuartzMdxComponent, QuartzMdxConstructor } from "./registry"
export {
  registerMdxComponent,
  getMdxComponent,
  getMdxComponentEntries,
  getMdxComponents,
} from "./registry"
export { MethodologyTree, MethodologyStep } from "./MethodologyTree"
export { Tractatus, TractatusPropo, TractatusRoot } from "./Tractatus"
