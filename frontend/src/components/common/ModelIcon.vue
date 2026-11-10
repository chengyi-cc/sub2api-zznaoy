<template>
  <svg
    v-if="iconInfo"
    :width="size"
    :height="size"
    viewBox="0 0 24 24"
    xmlns="http://www.w3.org/2000/svg"
    class="model-icon"
    fill="currentColor"
    fill-rule="evenodd"
  >
    <path v-for="(p, idx) in iconInfo.paths" :key="idx" :d="p" :fill="iconInfo.color" />
  </svg>
  <span v-else class="model-icon-fallback" :style="{ width: size, height: size, fontSize: `calc(${size} * 0.5)` }">
    {{ fallbackText }}
  </span>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { iconData } from './modelIconData'

const props = withDefaults(defineProps<{
  model: string
  size?: string
}>(), {
  size: '18px'
})

const fallbackText = computed(() => props.model.charAt(0).toUpperCase())

const iconKey = computed(() => {
  const modelLower = props.model.toLowerCase()

  // OpenAI models
  if (modelLower.startsWith('gpt') || modelLower.startsWith('o1') ||
      modelLower.startsWith('o3') || modelLower.startsWith('o4') ||
      modelLower.includes('chatgpt') || modelLower.includes('dall-e') ||
      modelLower.includes('whisper') || modelLower.includes('tts-1') ||
      modelLower.includes('text-embedding-3') || modelLower.includes('text-moderation') ||
      modelLower.includes('babbage') || modelLower.includes('davinci') ||
      modelLower.includes('curie') || modelLower.includes('ada')) return 'openai'

  // Anthropic Claude
  if (modelLower.includes('claude')) return 'claude'

  // Google Gemini
  if (modelLower.includes('gemini') || modelLower.includes('gemma') ||
      modelLower.includes('learnlm') || modelLower.includes('imagen-') ||
      modelLower.includes('veo-')) return 'gemini'

  // Zhipu GLM
  if (modelLower.includes('glm') || modelLower.includes('chatglm') ||
      modelLower.includes('cogview') || modelLower.includes('cogvideo')) return 'zhipu'

  // Alibaba Qwen
  if (modelLower.includes('qwen') || modelLower.includes('qwq')) return 'qwen'

  // DeepSeek
  if (modelLower.includes('deepseek')) return 'deepseek'

  // Mistral
  if (modelLower.includes('mistral') || modelLower.includes('mixtral') ||
      modelLower.includes('codestral') || modelLower.includes('pixtral') ||
      modelLower.includes('voxtral') || modelLower.includes('magistral')) return 'mistral'

  // Meta Llama
  if (modelLower.includes('llama')) return 'meta'

  // Cohere
  if (modelLower.includes('command') || modelLower.includes('c4ai-') ||
      modelLower.includes('embed-')) return 'cohere'

  // Yi
  if (modelLower.startsWith('yi-') || modelLower.startsWith('yi ')) return 'yi'

  // xAI Grok
  if (modelLower.includes('grok')) return 'xai'

  // Moonshot
  if (modelLower.includes('moonshot') || modelLower.includes('kimi')) return 'moonshot'

  // Doubao (ByteDance)
  if (modelLower.includes('doubao')) return 'doubao'

  // MiniMax
  if (modelLower.includes('abab') || modelLower.includes('minimax')) return 'minimax'

  // Baidu Wenxin
  if (modelLower.includes('ernie') || modelLower.includes('wenxin')) return 'wenxin'

  // iFlytek Spark
  if (modelLower.includes('spark')) return 'spark'

  // Tencent Hunyuan
  if (modelLower.includes('hunyuan')) return 'hunyuan'

  // Cloudflare
  if (modelLower.includes('@cf/')) return 'cloudflare'

  // Midjourney
  if (modelLower.includes('mj_') || modelLower.includes('midjourney')) return 'midjourney'

  // Perplexity
  if (modelLower.includes('perplexity') || modelLower.includes('pplx')) return 'perplexity'

  // Jina
  if (modelLower.includes('jina')) return 'jina'

  // OpenRouter
  if (modelLower.includes('openrouter')) return 'openrouter'

  // Suno
  if (modelLower.includes('suno')) return 'suno'

  // Ollama
  if (modelLower.includes('ollama')) return 'ollama'

  // 360
  if (modelLower.includes('360')) return 'ai360'

  // Dify
  if (modelLower.includes('dify')) return 'dify'

  // Coze
  if (modelLower.includes('coze')) return 'coze'

  return null
})

const iconInfo = computed(() => iconKey.value ? iconData[iconKey.value] : null)
</script>

<style scoped>
.model-icon {
  flex-shrink: 0;
}
.model-icon-fallback {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  color: white;
  font-weight: 600;
  flex-shrink: 0;
}
</style>
