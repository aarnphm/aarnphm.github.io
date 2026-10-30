import { Element, Root } from 'hast'
import { visit } from 'unist-util-visit'
import { ColorScheme } from '../../../util/theme'
import { classList } from './document'

export const applyEmailStyles = (
  root: Root,
  options: { headerFont: string; codeFont: string; colors: ColorScheme },
) => {
  const { headerFont, codeFont, colors } = options
  const codeBackground = colors.highlight || colors.lightgray
  const mergeStyle = (node: Element, style: string) => {
    const current = typeof node.properties?.style === 'string' ? node.properties.style : ''
    node.properties = node.properties ?? {}
    node.properties.style = current ? `${current}; ${style}` : style
  }
  visit(root, 'element', (node: Element, _index, parent) => {
    const dataCodeblock = node.properties?.dataCodeblock
    const classes = classList(node)
    const isSignature = classes.includes('signature')
    const isSms = dataCodeblock === 'sms' || classes.includes('text')
    if (node.tagName === 'p' || node.tagName === 'div') {
      if (isSms) {
        mergeStyle(
          node,
          [
            'padding-top: 24px',
            'padding-bottom: 24px',
            'padding-left: 16px',
            'font-style: italic',
            'border: 0',
            `border-top: 1px solid ${colors.lightgray}`,
            `border-bottom: 1px solid ${colors.lightgray}`,
          ].join('; '),
        )
      } else if (typeof dataCodeblock === 'string') {
        mergeStyle(
          node,
          [
            `background: ${codeBackground}`,
            `border: 1px solid ${colors.lightgray}`,
            'border-radius: 6px',
            'padding: 12px 14px',
            'margin: 0 0 18px 0',
            `font-family: ${codeFont}`,
            'font-size: 13px',
            'line-height: 1.6',
            'white-space: pre-wrap',
          ].join('; '),
        )
      }
    }
    if (node.tagName === 'p' && isSignature) {
      mergeStyle(
        node,
        [
          'display: flex',
          'flex-wrap: wrap',
          'justify-content: flex-end',
          'min-height: 51px',
          'margin: 0 0 18px 0',
        ].join('; '),
      )
    }
    switch (node.tagName) {
      case 'article':
        mergeStyle(node, 'margin: 1rem;')
        break
      case 'section':
        mergeStyle(node, 'margin: 1rem;')
        break
      case 'p':
        mergeStyle(node, 'margin: 0 0 18px 0;')
        break
      case 'h1':
        mergeStyle(
          node,
          `margin: 0 0 18px 0; font-size: 30px; line-height: 1.25; font-weight: 600; font-family: ${headerFont};`,
        )
        break
      case 'h2':
        mergeStyle(
          node,
          `margin: 28px 0 14px 0; font-size: 22px; line-height: 1.3; font-weight: 600; font-family: ${headerFont};`,
        )
        break
      case 'h3':
        mergeStyle(
          node,
          `margin: 22px 0 12px 0; font-size: 19px; line-height: 1.35; font-weight: 600; font-family: ${headerFont};`,
        )
        break
      case 'h4':
        mergeStyle(
          node,
          `margin: 18px 0 10px 0; font-size: 17px; line-height: 1.35; font-weight: 600; font-family: ${headerFont};`,
        )
        break
      case 'ul':
      case 'ol':
        mergeStyle(node, 'margin: 0 0 18px 0; padding: 0 0 0 24px;')
        break
      case 'li':
        mergeStyle(node, 'margin: 6px 0;')
        break
      case 'blockquote':
        mergeStyle(
          node,
          `margin: 22px 0; padding: 0 0 0 16px; border-left: 3px solid ${colors.lightgray}; color: ${colors.darkgray}; font-style: italic;`,
        )
        break
      case 'hr':
        mergeStyle(node, `border: 0; border-top: 1px solid ${colors.lightgray}; margin: 28px 0;`)
        break
      case 'img':
        mergeStyle(
          node,
          'max-width: 100%; height: auto; display: block; margin: 24px auto; border-radius: 6px;',
        )
        break
      case 'figure':
        mergeStyle(node, 'margin: 24px 0;')
        break
      case 'figcaption':
        mergeStyle(
          node,
          `margin-top: 10px; font-size: 14px; color: ${colors.darkgray}; text-align: center;`,
        )
        break
      case 'pre':
        mergeStyle(
          node,
          [
            `background: ${codeBackground}`,
            `border: 1px solid ${colors.lightgray}`,
            'border-radius: 6px',
            'padding: 12px 14px',
            'margin: 0 0 18px 0',
            'overflow-x: auto',
            'white-space: pre-wrap',
            `font-family: ${codeFont}`,
            'font-size: 13px',
            'line-height: 1.6',
          ].join('; '),
        )
        break
      case 'code': {
        const isBlock = parent?.type === 'element' && (parent as Element).tagName === 'pre'
        if (isBlock) {
          mergeStyle(node, 'background: transparent; padding: 0;')
        } else {
          mergeStyle(
            node,
            [
              `background: ${codeBackground}`,
              `border: 1px solid ${colors.lightgray}`,
              'border-radius: 4px',
              'padding: 0 3px',
              `font-family: ${codeFont}`,
              'font-size: 0.95em',
            ].join('; '),
          )
        }
        break
      }
      case 'a':
        mergeStyle(node, `color: ${colors.dark}; text-decoration: underline;`)
        break
      case 'table':
        mergeStyle(node, 'width: 100%; border-collapse: collapse; margin: 24px 0;')
        break
      case 'thead':
        mergeStyle(node, `background: ${colors.light};`)
        break
      case 'th':
        mergeStyle(
          node,
          `text-align: left; border-bottom: 1px solid ${colors.lightgray}; padding: 8px 6px;`,
        )
        break
      case 'td':
        mergeStyle(node, `border-bottom: 1px solid ${colors.lightgray}; padding: 8px 6px;`)
        break
      case 'sup':
        mergeStyle(node, 'font-size: 0.75em; line-height: 0;')
        break
      case 'small':
        mergeStyle(node, `font-size: 14px; color: ${colors.darkgray};`)
        break
      default:
        break
    }
  })
}

export const buildEmailHtml = (
  content: string,
  options: { fontCss: string; colors: ColorScheme; bodyFont: string },
): string => {
  const { fontCss, colors, bodyFont } = options
  const bodyStyle = [
    'margin: 0',
    'padding: 0',
    `background: ${colors.light}`,
    `color: ${colors.dark}`,
    '-webkit-text-size-adjust: 100%',
  ].join('; ')
  const outerTableStyle = ['width: 100%', `background: ${colors.light}`].join('; ')
  const containerStyle = [
    'width: 100%',
    'max-width: 640px',
    'background: #ffffff',
    `border: 1px solid ${colors.lightgray}`,
    'border-radius: 12px',
    'overflow: hidden',
  ].join('; ')
  const innerStyle = [
    'padding: 32px 36px 40px',
    `font-family: ${bodyFont}`,
    'font-size: 17px',
    'line-height: 1.75',
    `color: ${colors.dark}`,
    'text-align: left',
  ].join('; ')
  const headParts = [
    '<meta charset="utf-8">',
    '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
    fontCss ? `<style>${fontCss}</style>` : '',
  ].filter(Boolean)
  return [
    '<!doctype html>',
    '<html>',
    '<head>',
    ...headParts,
    '</head>',
    `<body dir="ltr" style="${bodyStyle}">`,
    `<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="${outerTableStyle}">`,
    '<tr>',
    '<td align="center" style="padding: 24px 12px;">',
    `<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="${containerStyle}">`,
    '<tr>',
    `<td style="${innerStyle}">`,
    content,
    '</td>',
    '</tr>',
    '</table>',
    '</td>',
    '</tr>',
    '</table>',
    '</body>',
    '</html>',
  ].join('')
}
