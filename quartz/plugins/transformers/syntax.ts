import { QuartzTransformerPlugin } from "../types"
import rehypePrettyCode, { Options as CodeOptions } from "rehype-pretty-code"

export const SyntaxHighlighting: QuartzTransformerPlugin = () => ({
  name: "SyntaxHighlighting",
  htmlPlugins() {
    return [
      [
        rehypePrettyCode,
        {
          keepBackground: false,
          theme: {
            dark: "rose-pine",
            light: "rose-pine-dawn",
          },
        } satisfies Partial<CodeOptions>,
      ],
    ]
  },
})
