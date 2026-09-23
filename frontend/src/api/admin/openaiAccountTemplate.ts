import { apiClient } from '../client'
import type { OpenAIAccountTemplate } from '@/utils/openaiAccountTemplate'
const url = '/admin/settings/openai-account-template'
export const openaiAccountTemplateAPI = {
  async get() { return (await apiClient.get<OpenAIAccountTemplate>(url)).data },
  async save(template: OpenAIAccountTemplate) { return (await apiClient.put<OpenAIAccountTemplate>(url, template)).data }
}
