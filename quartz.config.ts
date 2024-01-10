import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

const config: QuartzConfig = {
  configuration: {
    pageTitle: "Aaron's notes",
    enableSPA: true,
    enablePopovers: true,
    analytics: {
      provider: "plausible",
    },
    baseUrl: "aarnphm.xyz",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "created",
    theme: {
      typography: {
        header: "DM Serif Text",
        body: "Lora",
        code: "JetBrains Mono",
      },
      colors: {
        lightMode: {
          light: "#f8f0e7",
          lightgray: "#f2e9e1",
          gray: "#9893a5",
          darkgray: "#797593",
          dark: "#575279",
          secondary: "#b4637a ",
          tertiary: "#d7827e",
          highlight: "rgba(143, 159, 169, 0.15)",
        },
        darkMode: {
          light: "#1f1d30",
          lightgray: "#393552",
          gray: "#6e6a86",
          darkgray: "#908caa",
          dark: "#e0def4",
          secondary: "#eb6f92",
          tertiary: "#ea9a97",
          highlight: "rgba(143, 159, 169, 0.15)",
        },
      },
    },
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.TableOfContents(),
      Plugin.CreatedModifiedDate({ priority: ["frontmatter", "filesystem", "git"] }),
      Plugin.Latex({ renderEngine: "katex" }),
      Plugin.SyntaxHighlighting(),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.CrawlLinks({ markdownLinkResolution: "absolute" }),
      Plugin.Description(),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources({ fontOrigin: "googleFonts" }),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.NotFoundPage(),
    ],
  },
}

export default config
