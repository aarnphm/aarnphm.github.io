import { GlobalConfiguration, QuartzConfig } from "./quartz/cfg"
import { byDateAndAlphabetical } from "./quartz/components/PageList"
import * as Plugin from "./quartz/plugins"
import * as Component from "./quartz/components"
import { QuartzPluginData } from "./quartz/plugins/vfile"

const configuration: GlobalConfiguration = {
  pageTitle: "Aaron's notes",
  enableSPA: true,
  enablePopovers: true,
  analytics: {
    provider: "plausible",
  },
  locale: "fr-FR",
  baseUrl: "aarnphm.xyz",
  ignorePatterns: [
    "private",
    "templates",
    ".obsidian",
    "**.adoc",
    "**.zip",
    "**.epub",
    "**.docx",
    "**.lvbitx",
    "**.slx",
    "**.so",
    "**/*400232791*",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    "**.ignore.pdf",
    "capstone",
    "**/.conform*",
  ],
  defaultDateType: "created",
  theme: {
    cdnCaching: true,
    fontOrigin: "local",
    typography: {
      header: "Parclo Serif",
      body: "Parclo Serif",
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
        textHighlight: "rgba(246, 193, 119, 0.28)",
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
        textHighlight: "#b3aa0288",
      },
    },
  },
}

/**
 * Quartz 4.0 Configuration
 *
 * See https://quartz.jzhao.xyz/configuration for more information.
 */
const config: QuartzConfig = {
  configuration,
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({ priority: ["frontmatter", "filesystem"] }),
      Plugin.Aarnphm(),
      Plugin.Pseudocode(),
      Plugin.TikzJax(),
      Plugin.TelescopicText(),
      // FIXME: implement this
      // Plugin.Recipe(),
      // Plugin.Embeddings(),
      Plugin.Twitter(),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "rose-pine-dawn",
          dark: "rose-pine",
        },
        keepBackground: true,
      }),
      Plugin.Citations({ bibliography: "./content/References.bib" }),
      Plugin.ObsidianFlavoredMarkdown({ parseTags: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.CrawlLinks({
        markdownLinkResolution: "absolute",
        externalLinkIcon: true,
        lazyLoad: true,
        enableArxivEmbed: true,
        enableRawEmbed: true,
      }),
      Plugin.Description(),
      Plugin.Latex({
        renderEngine: "katex",
        customMacros: {
          "\\argmin": "\\mathop{\\operatorname{arg\\,min}}\\limits",
          "\\argmax": "\\mathop{\\operatorname{arg\\,max}}\\limits",
          "\\upgamma": "\\mathit{\\gamma}",
          "\\upphi": "\\mathit{\\phi}",
          "\\upbeta": "\\mathit{\\beta}",
          "\\upalpha": "\\mathit{\\alpha}",
          "\\uptheta": "\\mathit{\\theta}",
        },
        katexOptions: { strict: false },
      }),
      Plugin.GitHub({
        internalLinks: [
          "livingalonealone.com",
          "bentoml.com",
          "vllm.ai",
          "obsidian.md",
          "neovim.io",
        ],
      }),
      // Plugin.GoogleDocs(),
      Plugin.TableOfContents({ maxDepth: 5 }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.LLM(),
      Plugin.FolderPage({
        pageBody: Component.FolderContent({
          sort: (a: QuartzPluginData, b: QuartzPluginData): number => {
            // Check if either file has a folder tag
            const aHasFolder = a.frontmatter?.tags?.includes("folder") ?? false
            const bHasFolder = b.frontmatter?.tags?.includes("folder") ?? false

            // If one has folder tag and other doesn't, prioritize the one with folder tag
            if (aHasFolder && !bHasFolder) return -1
            else if (!aHasFolder && bHasFolder) return 1
            else return byDateAndAlphabetical(configuration)(a, b)
          },
          include: [
            ".pdf",
            ".py",
            ".go",
            ".c",
            ".m",
            ".cu",
            ".java",
            ".sql",
            ".js",
            ".ipynb",
            ".json",
          ],
          exclude: [/\.(ignore\.pdf)$/, /400232791/],
        }),
      }),
      Plugin.TagPage(),
      // Plugin.ArenaPage(),
      // Plugin.InfinitePoemPage(),
      Plugin.NotebookViewer(),
      Plugin.ContentIndex({ rssLimit: 60 }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.Favicon(),
      Plugin.NotFoundPage(),
      Plugin.CustomOgImages(),
      Plugin.PressKit(),
    ],
  },
}

export default config
