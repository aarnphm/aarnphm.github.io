import { readFile } from 'node:fs/promises'
import ts from 'typescript'

export async function readImageIgnorePatterns(configPath: string): Promise<string[]> {
  const source = ts.createSourceFile(
    configPath,
    await readFile(configPath, 'utf8'),
    ts.ScriptTarget.Latest,
    true,
  )
  let patterns: string[] | undefined
  function visit(node: ts.Node): void {
    if (ts.isPropertyAssignment(node) && node.name.getText(source) === 'ignorePatterns') {
      if (!ts.isArrayLiteralExpression(node.initializer))
        throw new Error(
          'Image inventory requires literal ignorePatterns in the Quartz configuration.',
        )
      patterns = node.initializer.elements.map(element => {
        if (!ts.isStringLiteral(element))
          throw new Error(
            'Image inventory requires string ignorePatterns in the Quartz configuration.',
          )
        return element.text
      })
    }
    ts.forEachChild(node, visit)
  }
  visit(source)
  if (!patterns) throw new Error(`No ignorePatterns found in ${configPath}.`)
  return patterns
}
