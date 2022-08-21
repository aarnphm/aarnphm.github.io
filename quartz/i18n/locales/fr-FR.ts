import { Translation } from "./definition"

export default {
  propertyDefaults: {
    title: "sans titre",
    description: "aucune description fournie",
  },
  components: {
    callout: {
      note: "note",
      abstract: "résumé",
      info: "info",
      todo: "à faire",
      tip: "conseil",
      success: "succès",
      question: "question",
      warning: "avertissement",
      failure: "échec",
      danger: "danger",
      bug: "bogue",
      example: "exemple",
      quote: "citation",
    },
    backlinks: {
      title: "liens retour",
      noBacklinksFound: "aucun lien retour trouvé",
    },
    themeToggle: {
      lightMode: "mode clair",
      darkMode: "mode sombre",
    },
    explorer: {
      title: "explorateur",
    },
    footer: {
      createdWith: "créé avec",
    },
    graph: {
      title: "vue graphique",
    },
    recentNotes: {
      title: "notes récentes",
      seeRemainingMore: ({ remaining }) => `voir ${remaining} de plus →`,
    },
    transcludes: {
      transcludeOf: ({ targetSlug }) => `transclusion de ${targetSlug}`,
      linkToOriginal: "lien vers l'original",
    },
    search: {
      title: "recherche",
      searchBarPlaceholder: "rechercher quelque chose",
    },
    tableOfContents: {
      title: "table des matières",
    },
    contentMeta: {
      readingTime: ({ minutes, words }) => `${minutes} min de lecture (${words} words)`,
    },
  },
  pages: {
    rss: {
      recentNotes: "notes récentes",
      lastFewNotes: ({ count }) => `les dernières ${count} notes`,
    },
    error: {
      title: "introuvable",
      notFound: "à venir",
    },
    folderContent: {
      folder: "dossier",
      itemsUnderFolder: ({ count }) =>
        count === 1 ? "1 élément sous ce dossier." : `${count} éléments sous ce dossier.`,
    },
    tagContent: {
      tag: "étiquette",
      tagIndex: "index des étiquettes",
      itemsUnderTag: ({ count }) =>
        count === 1 ? "1 élément avec cette étiquette." : `${count} éléments avec cette étiquette.`,
      showingFirst: ({ count }) => `affichage des premières ${count} étiquettes.`,
      totalTags: ({ count }) => `trouvé ${count} étiquettes au total.`,
    },
  },
} as const satisfies Translation
