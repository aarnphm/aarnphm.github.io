import { GlobalConfiguration, QuartzConfig } from "./quartz/cfg"
import { byDateAndAlphabetical } from "./quartz/components/PageList"
import * as Plugin from "./quartz/plugins"
import { QuartzPluginData } from "./quartz/plugins/vfile"

/**
 * Quartz 4.0 Configuration
 *
 * See https://quartz.jzhao.xyz/configuration for more information.
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "Aaron's notes",
    pageTitleSuffix: " -- de lecture",
    enableSPA: true,
    enablePopovers: true,
    generateSocialImages: true,
    analytics: {
      provider: "plausible",
    },
    locale: "fr-FR",
    baseUrl: "aarnphm.xyz",
    ignorePatterns: [
      "private",
      "templates",
      ".obsidian",
      "joininteract",
      "**/sfwr-4g06ab/source",
      "**.adoc",
      "**/lab*/**",
    ],
    defaultDateType: "created",
    theme: {
      cdnCaching: true,
      fontOrigin: "local",
      typography: {
        header: "GT Sectra Display",
        body: "Cardo",
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
          textHighlight: "#fff23688",
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
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "filesystem"],
      }),
      Plugin.Pseudocode(),
      Plugin.Poetry(),
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
      Plugin.Citations({
        bibliographyFile: "./content/References.bib",
        linkCitations: true,
      }),
      Plugin.ObsidianFlavoredMarkdown(),
      Plugin.GitHubFlavoredMarkdown(),
      // Plugin.GitHub(),
      Plugin.TableOfContents(),
      Plugin.CrawlLinks({
        markdownLinkResolution: "absolute",
        externalLinkIcon: true,
        lazyLoad: true,
        enableRawEmbed: {
          enable: true,
          extensions: [".py", ".m", ".go", ".c", ".java", ".cpp", ".h", ".hpp", ".cu"],
          cdn: "https://raw.aarnphm.xyz/",
        },
      }),
      Plugin.Description(),
      Plugin.Latex({
        renderEngine: "katex",
        customMacros: {
          "\\argmin": "\\mathop{\\operatorname{arg\\,min}}\\limits",
          "\\argmax": "\\mathop{\\operatorname{arg\\,max}}\\limits",
        },
      }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage({
        sort: (a: QuartzPluginData, b: QuartzPluginData): number => {
          // Check if either file has a folder tag
          const aHasFolder = a.frontmatter?.tags?.includes("folder") ?? false
          const bHasFolder = b.frontmatter?.tags?.includes("folder") ?? false

          // If one has folder tag and other doesn't, prioritize the one with folder tag
          if (aHasFolder && !bHasFolder) return -1
          else if (!aHasFolder && bHasFolder) return 1
          else {
            return byDateAndAlphabetical({ defaultDateType: "created" } as GlobalConfiguration)(
              a,
              b,
            )
          }
        },
      }),
      Plugin.TagPage(),
      Plugin.CuriusPage(),
      Plugin.MenuPage(),
      Plugin.PoetryPage(),
      Plugin.NotebookPage(),
      // Plugin.InfinitePoemPage(),
      Plugin.ContentIndex({ rssLimit: 40 }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.NotFoundPage(),
    ],
  },
}

export default config
