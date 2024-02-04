import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

const config: QuartzConfig = {
  configuration: {
    pageTitle: "aarnphm.xyz",
    enableSPA: true,
    enablePopovers: true,
    enableTexture: true,
    analytics: {
      provider: "plausible",
    },
    locale: "fr-FR",
    baseUrl: "aarnphm.xyz",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "created",
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
      Plugin.CreatedModifiedDate({
        // you can add 'git' here for last modified from Git
        // if you do rely on git for dates, ensure defaultDateType is 'modified'
        priority: ["frontmatter", "filesystem"],
      }),
      Plugin.Latex({ renderEngine: "katex" }),
      Plugin.SyntaxHighlighting(),
      Plugin.ObsidianFlavoredMarkdown(),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents({ maxDepth: 4, collapseByDefault: true }),
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
      Plugin.ZenPage({ slug: ["uses", "dump/quotes", "influence"] }),
      Plugin.ContentIndex({ rssLimit: 40 }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.NotFoundPage(),
    ],
  },
}

export default config
