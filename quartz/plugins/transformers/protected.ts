import { QuartzTransformerPlugin } from "../../types/plugin"
import content from "../../components/styles/protected.scss"

function getPasswordForPage(file: any): string {
  const frontmatter = file.data.frontmatter

  if (frontmatter?.password) {
    const customPassword = process.env[frontmatter.password]
    if (customPassword) {
      return customPassword
    }
  }

  const defaultPassword = process.env.PROTECTED_CONTENT_PASSWORD
  if (defaultPassword) {
    return defaultPassword
  }

  throw new Error(
    `No password found for protected page ${file.data.slug}. ` +
      `Set ${frontmatter?.password || "PROTECTED_CONTENT_PASSWORD"} environment variable.`,
  )
}

export const Protected: QuartzTransformerPlugin = () => {
  return {
    name: "Protected",
    htmlPlugins: (ctx) => [
      () => {
        return async (_tree, file) => {
          if (ctx.argv.watch && !ctx.argv.force) return

          const frontmatter = file.data.frontmatter
          if (!frontmatter?.protected) return

          const password = getPasswordForPage(file)
          file.data.protectedPassword = password
        }
      },
    ],
    externalResources() {
      return { css: [{ content, inline: true }] }
    },
  }
}
