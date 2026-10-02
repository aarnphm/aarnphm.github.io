import type { CompileContext, Extension, Token } from 'mdast-util-from-markdown'
import type { Options } from 'mdast-util-to-markdown'
import type { Processor } from 'unified'
import type { SpeechPhrase } from './types'
import { speech } from './syntax'

export { speech }
export type { SpeechPhrase }

export function speechFromMarkdown(): Extension {
  return { enter: { speechPhrase: enter }, exit: { speechPhrase: exit } }

  function enter(this: CompileContext, token: Token): undefined {
    return this.enter({ type: 'speechPhrase', children: [] }, token)
  }

  function exit(this: CompileContext, token: Token): undefined {
    return this.exit(token)
  }
}

export function speechToMarkdown(): Options {
  return {
    handlers: {
      speechPhrase(node, _parent, state, info) {
        if (node.type !== 'speechPhrase') return ''
        return state.containerPhrasing(node, info)
      },
    },
  }
}

export function remarkSpeech(this: Processor): void {
  const data = this.data()
  data.micromarkExtensions ??= []
  data.fromMarkdownExtensions ??= []
  data.micromarkExtensions.push(speech())
  data.fromMarkdownExtensions.push(speechFromMarkdown())
}
