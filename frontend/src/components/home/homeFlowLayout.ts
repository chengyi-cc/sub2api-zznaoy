import { iconData } from '@/components/common/modelIconData'

/** 首页线路图左侧的客户端。最后一项的文字走 i18n。 */
export const HOME_FLOW_TOOLS = ['Claude Code', 'Codex CLI', 'Gemini CLI'] as const

/**
 * 首页线路图右侧展示的模型。只是展示接入生态，不代表实际开放范围。
 * icon 对应 modelIconData 的键；labelKey 存在时文字走 i18n（home.designed.models.*）。
 */
export const HOME_FLOW_MODELS: ReadonlyArray<{ icon: string; label?: string; labelKey?: string }> = [
  { icon: 'claude', label: 'Claude' },
  { icon: 'openai', label: 'GPT' },
  { icon: 'gemini', label: 'Gemini' },
  { icon: 'xai', label: 'Grok' },
  { icon: 'deepseek', label: 'DeepSeek' },
  { icon: 'qwen', labelKey: 'qwen' },
  { icon: 'moonshot', label: 'Kimi' }
]

export interface FlowIcon {
  fill: string
  paths: string[]
}

export interface FlowChip {
  x: number
  y: number
  width: number
  height: number
}

export interface FlowLink {
  d: string
  delay: string
}

/** 线路图坐标系：viewBox = `0 -28 1200 400`。 */
export const FLOW_VIEWBOX = '0 -28 1200 400'
export const FLOW_HUB = { cx: 600, cy: 172, halfWidth: 68, halfHeight: 72 } as const
const TOOL_SIZE = { width: 172, height: 42 }
const MODEL_SIZE = { width: 168, height: 38 }

const round = (value: number) => Math.round(value * 10) / 10

function curve(x1: number, y1: number, x2: number, y2: number): string {
  const dx = (x2 - x1) * 0.5
  return `M${round(x1)},${round(y1)} C${round(x1 + dx)},${round(y1)} ${round(x2 - dx)},${round(y2)} ${round(x2)},${round(y2)}`
}

/** 近黑色的品牌图标在深色背景上会消失，改为跟随文字颜色。 */
export function flowIconFill(color: string): string {
  const match = /^#?([0-9a-f]{6})$/i.exec(color)
  if (!match) return 'currentColor'
  const value = parseInt(match[1], 16)
  const luminance = (0.299 * (value >> 16) + 0.587 * ((value >> 8) & 255) + 0.114 * (value & 255)) / 255
  return luminance < 0.25 ? 'currentColor' : color
}

export function flowIcon(key: string): FlowIcon | null {
  const icon = iconData[key]
  return icon ? { fill: flowIconFill(icon.color), paths: icon.paths } : null
}

/**
 * 计算线路图几何：左侧工具和右侧模型各自排成向外微凸的弧，
 * 连线汇入网关时收拢到网关边缘的中段。
 * 返回的 delay 让光流在各条线上错开出发。
 */
export function buildHomeFlowLayout(toolCount: number, modelCount: number) {
  const { cx, cy, halfWidth } = FLOW_HUB
  const toolSpan = 138
  const modelSpan = 161
  const toolStep = toolCount > 1 ? (toolSpan * 2) / (toolCount - 1) : 0
  const modelStep = modelCount > 1 ? (modelSpan * 2) / (modelCount - 1) : 0

  const tools: FlowChip[] = []
  const toolLinks: FlowLink[] = []
  for (let i = 0; i < toolCount; i++) {
    const y = toolCount > 1 ? cy - toolSpan + i * toolStep : cy
    const t = (y - cy) / toolSpan
    const x = 66 - 46 * (1 - t * t)
    tools.push({ x: round(x), y: round(y - TOOL_SIZE.height / 2), ...TOOL_SIZE })
    toolLinks.push({ d: curve(x + TOOL_SIZE.width, y, cx - halfWidth, cy + (y - cy) * 0.16), delay: `${(-i * 0.85).toFixed(2)}s` })
  }

  const models: FlowChip[] = []
  const modelLinks: FlowLink[] = []
  for (let i = 0; i < modelCount; i++) {
    const y = modelCount > 1 ? cy - modelSpan + i * modelStep : cy
    const t = (y - cy) / modelSpan
    const x = 962 + 50 * (1 - t * t)
    models.push({ x: round(x), y: round(y - MODEL_SIZE.height / 2), ...MODEL_SIZE })
    modelLinks.push({ d: curve(cx + halfWidth, cy + (y - cy) * 0.16, x, y), delay: `${(-(1.7 + i * 0.43)).toFixed(2)}s` })
  }

  return { tools, models, links: [...toolLinks, ...modelLinks] }
}
