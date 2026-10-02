import type { RoughAnnotationType } from 'rough-notation/lib/model'
import { PluginSettingTab, Setting } from 'obsidian'
import type MarkedPlugin from '../main'
import { INTENSITIES, type Intensity } from './parser'

export const ANNOTATION_TYPES: Record<RoughAnnotationType, string> = {
  box: 'Box',
  underline: 'Underline',
  circle: 'Circle',
  highlight: 'Highlight',
  'strike-through': 'Strike through',
  'crossed-off': 'Crossed off',
  bracket: 'Brackets',
}

interface LevelStyle {
  type: RoughAnnotationType | 'default'
  color: string
}

export interface MarkerSettings {
  type: RoughAnnotationType
  levels: Record<Intensity, LevelStyle>
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

export function annotationType(value: unknown): RoughAnnotationType | undefined {
  for (const type of [
    'box',
    'underline',
    'circle',
    'highlight',
    'strike-through',
    'crossed-off',
    'bracket',
  ] as const) {
    if (value === type) return type
  }
}

export function loadSettings(value: unknown): MarkerSettings {
  const data = isRecord(value) ? value : {}
  const levels = isRecord(data.levels) ? data.levels : {}
  const style = (level: Intensity): LevelStyle => {
    const saved = levels[level]
    return {
      type: (isRecord(saved) && annotationType(saved.type)) || 'default',
      color:
        isRecord(saved) && typeof saved.color === 'string' && CSS.supports('color', saved.color)
          ? saved.color
          : '',
    }
  }
  return {
    type: annotationType(data.type) ?? 'box',
    levels: {
      h1: style('h1'),
      h2: style('h2'),
      h3: style('h3'),
      h4: style('h4'),
      h5: style('h5'),
      h6: style('h6'),
      h7: style('h7'),
    },
  }
}

export class MarkedSettingsTab extends PluginSettingTab {
  constructor(private plugin: MarkedPlugin) {
    super(plugin.app, plugin)
  }

  display(): void {
    const { containerEl } = this
    containerEl.empty()
    containerEl.addClass('garden-plugin-ui', 'garden-marked-settings')
    containerEl.createEl('p', {
      text: 'Use ::text:: or ::text{h1}:: through ::text{h7}::. Bare markers use h2. Changes apply to open notes immediately.',
    })
    new Setting(containerEl)
      .setName('Default annotation')
      .setDesc('Used by every level without its own override.')
      .addDropdown(dropdown => {
        dropdown
          .addOptions(ANNOTATION_TYPES)
          .setValue(this.plugin.settings.type)
          .onChange(async value => {
            this.plugin.settings.type = annotationType(value) ?? 'box'
            await this.plugin.saveSettings()
          })
      })
    for (const level of INTENSITIES) {
      new Setting(containerEl)
        .setName(`${level} annotation`)
        .setDesc(
          level === 'h2'
            ? 'Also applies to bare ::text:: markers.'
            : `Applies to ::text{${level}}::.`,
        )
        .addDropdown(dropdown => {
          dropdown
            .addOption('default', 'Use default')
            .addOptions(ANNOTATION_TYPES)
            .setValue(this.plugin.settings.levels[level].type)
            .onChange(async value => {
              this.plugin.settings.levels[level].type = annotationType(value) ?? 'default'
              await this.plugin.saveSettings()
            })
        })
        .addText(text => {
          text.setPlaceholder('Theme color').setValue(this.plugin.settings.levels[level].color)
          text.inputEl.setAttribute('aria-label', `${level} annotation color`)
          text.onChange(async value => {
            const color = value.trim()
            const valid = !color || CSS.supports('color', color)
            text.inputEl.setAttribute('aria-invalid', String(!valid))
            if (!valid) return
            this.plugin.settings.levels[level].color = color
            await this.plugin.saveSettings()
          })
        })
    }
    containerEl.createEl('p', {
      cls: 'setting-item-description',
      text: 'Colors follow the garden palette when left empty. Custom colors accept CSS colors such as #d14d41 or var(--text-accent). Brackets appear on both sides; highlights remain translucent.',
    })
  }
}
