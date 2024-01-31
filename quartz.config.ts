import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

const config: QuartzConfig = {
  configuration: {
    pageTitle: "aarnphm.xyz",
    locale: "en-US",
    enableSPA: true,
    enablePopovers: true,
    enableTexture: false,
    analytics: {
      provider: "plausible",
    },
    baseUrl: "aarnphm.xyz",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "created",
    defaultFooterStyle: "minimal",
    contentOnlyPages: ["uses", "dump/quotes", "curius", "influence"],
    theme: {
      typography: {
        header: "DM Serif Text",
        body: "EB Garamond",
        code: "Berkeley Mono",
      },
      colors: {
        lightMode: {
          light: "#fffaf3",
          lightgray: "#f2e9e1",
          gray: "#9893a5",
          darkgray: "#797593",
          dark: "#575279",
          secondary: "#d7827e",
          tertiary: "#b4637a",
          highlight: "rgba(143, 159, 169, 0.15)",
        },
        darkMode: {
          light: "#1f1d30",
          lightgray: "#26233a",
          gray: "#6e6a86",
          darkgray: "#908caa",
          dark: "#e0def4",
          secondary: "#ebbcba",
          tertiary: "#eb6f92",
          highlight: "rgba(143, 159, 169, 0.15)",
        },
      },
    },
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.TableOfContents(),
      Plugin.CreatedModifiedDate({ priority: ["frontmatter", "filesystem"] }),
      Plugin.Latex({ renderEngine: "katex" }),
      Plugin.SyntaxHighlighting(),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.CrawlLinks({
        markdownLinkResolution: "absolute",
        enableRawEmbed: {
          enable: true,
          extensions: [".py", ".m", ".go", ".c", ".java"],
          cdn: "https://raw.aarnphm.xyz/",
        },
      }),
      Plugin.Description(),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources({ fontOrigin: "googleFonts" }),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.CuriusPage(),
      Plugin.ContentIndex({ rssLimit: 40 }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.NotFoundPage(),
    ],
  },
}

export default config
