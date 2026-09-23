import { apiClient } from '../client'
import type { OpenAIAccountTemplates } from '@/utils/openaiAccountTemplate'
const url = '/admin/settings/openai-account-template'
export const openaiAccountTemplateAPI = {
  async get() { return (await apiClient.get<OpenAIAccountTemplates>(url)).data },
  async save(template: OpenAIAccountTemplates) { return (await apiClient.put<OpenAIAccountTemplates>(url, template)).data }
}
