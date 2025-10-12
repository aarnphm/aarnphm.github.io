import { GlobalConfiguration, QuartzConfig } from "./quartz/cfg"
import { byDateAndAlphabetical } from "./quartz/components/PageList"
import * as Plugin from "./quartz/plugins"
import * as Component from "./quartz/components"
import { QuartzPluginData } from "./quartz/plugins/vfile"

const model = "onnx-community/embeddinggemma-300m-ONNX" // onnx-community/Qwen3-Embedding-0.6B-ONNX, intfloat/multilingual-e5-large

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
    "**.pyc",
    "**.slx",
    "**.so",
    "**/*400232791*",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    "**.ignore.pdf",
    "capstone",
    "**/.conform*",
    "**/target",
    "**/data",
  ],
  defaultDateType: "created",
  theme: {
    cdnCaching: true,
    fontOrigin: "local",
    typography: {
      title: "Parclo Serif",
      header: "Parclo Serif",
      body: "PP Neue Montreal",
      code: "Berkeley Mono",
    },
    colors: {
      lightMode: {
        light: "rgb(255, 252, 240)",
        lightgray: "rgb(230, 228, 217)",
        gray: "rgb(183, 181, 172)",
        darkgray: "rgb(111, 110, 105)",
        dark: "rgb(16, 15, 15)",
        secondary: "rgb(205, 213, 151)",
        tertiary: "rgb(252, 193, 146)",
        highlight: "rgb(218, 216, 206)",
        textHighlight: "rgb(241, 214, 126)",
      },
      darkMode: {
        light: "rgb(16, 15, 15)",
        lightgray: "rgb(40, 39, 38)",
        gray: "rgb(87, 86, 83)",
        darkgray: "rgb(135, 133, 128)",
        dark: "rgb(206, 205, 195)",
        secondary: "rgb(205, 213, 151)",
        tertiary: "rgb(252, 193, 146)",
        highlight: "rgb(135, 154, 57)",
        textHighlight: "rgb(241, 214, 126)",
      },
    },
  },
  semanticSearch: {
    enable: true,
    model,
    aot: true,
    dims: 768,
    dtype: "fp32",
    shardSizeRows: 1024,
    hnsw: { M: 16, efConstruction: 200 },
    chunking: { chunkSize: 256, chunkOverlap: 64 },
    vllm: { concurrency: 16, batchSize: 128 },
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
      Plugin.TikzJax({ showConsole: false }),
      Plugin.TelescopicText(),
      // Convert code-file transcludes to code blocks before highlighting
      Plugin.CodeViewer(),
      Plugin.Twitter(),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "rose-pine-dawn",
          dark: "rose-pine",
        },
        keepBackground: true,
      }),
      Plugin.Citations({ bibliography: "./content/References.bib" }),
      Plugin.ObsidianBases(),
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
          "\\upeta": "\\mathit{\\eta}",
          "\\upbeta": "\\mathit{\\beta}",
          "\\upalpha": "\\mathit{\\alpha}",
          "\\uptheta": "\\mathit{\\theta}",
          // KaTeX does not support tabular/multicolumn. Provide safe fallbacks.
          // This macro drops alignment specifiers and yields only the cell content.
          // IMPORTANT: when spanning >1 columns, add explicit '&'s in source rows.
          "\\multicolumn": "#3",
          // Text micro symbol compatibility
          "\\textmu": "\\mu",
        },
        katexOptions: { strict: true, throwOnError: true },
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
      Plugin.TableOfContents({ maxDepth: 5 }),
      Plugin.LLM(),
      Plugin.Slides(),
      Plugin.Arena(),
    ],
    filters: [Plugin.RemoveDrafts(), Plugin.RemovePrivate()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.LLMText(),
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
            ".cpp",
            ".m",
            ".cu",
            ".java",
            ".sql",
            ".js",
            ".ipynb",
            ".json",
            ".csv",
            ".webp",
            ".mp4",
            ".lean",
            ".svg",
          ],
          exclude: [/\.(ignore\.pdf)$/, /400232791/],
          tags: [
            "ml",
            "interpretability",
            "philosophy",
            "serving",
            "love",
            "fiction",
            "math",
            "evergreen",
          ],
          lg: ["thoughts/mechanistic-interpretability", "thoughts/vllm"],
          sm: [
            "thoughts/love",
            "thoughts/LLMs",
            "thoughts/Connectionist-network",
            "thoughts/Tractatus",
            "thoughts/Transformers",
            "thoughts/Camus",
            "thoughts/Attention",
            "thoughts/Philosophy-and-Nietzsche",
            "thoughts/ethics",
            "thoughts/Existentialism",
            "thoughts/reductionism",
            "thoughts/GPU-programming",
          ],
        }),
      }),
      Plugin.TagPage(),
      Plugin.NotebookViewer(),
      Plugin.ContentIndex({ rssLimit: 60 }),
      Plugin.SemanticIndex(configuration.semanticSearch!),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.Favicon(),
      Plugin.NotFoundPage(),
      Plugin.CustomOgImages(),
      Plugin.PressKit(),
      Plugin.SlidesPage(),
      Plugin.ArenaPage(),
      Plugin.BaseViewPage(),
    ],
  },
}

export default config
