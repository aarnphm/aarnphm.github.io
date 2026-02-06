import remarkBreaks from 'remark-breaks'
import { QuartzTransformerPlugin } from '../../types/plugin'

export const HardLineBreaks: QuartzTransformerPlugin = () => {
  return {
    name: 'HardLineBreaks',
    markdownPlugins() {
      return [remarkBreaks]
    },
  }
}
