import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'

export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  agentSteps?: AgentStep[]
  criticScore?: number
  totalTokens?: number
  durationMs?: number
}

export interface AgentStep {
  agent: string
  input: string
  output: string
  toolCalls: Record<string, unknown>[]
  durationMs: number
  tokens: number
}

export interface Goal {
  goal_id: string
  description: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress_pct: number
  tasks: TaskNode[]
  created_at: string
  updated_at: string
  result?: string
  error?: string
}

export interface TaskNode {
  task_id: string
  description: string
  depends_on: string[]
  status: string
  result?: string
  agent?: string
}

export interface BrokerEvent {
  id: string
  topic: string
  payload: Record<string, unknown>
  created_at: string
}

export interface SystemHealth {
  broker: boolean
  llm_engine: boolean
  memory: boolean
  agent_core: boolean
  voice: boolean
  tool_system: boolean
}

interface JarvisState {
  messages: Message[]
  goals: Goal[]
  brokerEvents: BrokerEvent[]
  isStreaming: boolean
  sessionId: string
  settings: {
    model: string
    temperature: number
    voice: string
    ttsEnabled: boolean
    streamingEnabled: boolean
  }
  systemHealth: SystemHealth
  activeAgents: string[]
  brokerWs: WebSocket | null
  voiceWs: WebSocket | null

  addMessage: (msg: Message) => void
  updateLastMessage: (content: string) => void
  setStreaming: (v: boolean) => void
  setGoals: (goals: Goal[]) => void
  updateGoal: (goal: Goal) => void
  addBrokerEvent: (event: BrokerEvent) => void
  updateSettings: (settings: Partial<JarvisState['settings']>) => void
  updateHealth: (health: Partial<SystemHealth>) => void
  setActiveAgents: (agents: string[]) => void
  setBrokerWs: (ws: WebSocket | null) => void
  setVoiceWs: (ws: WebSocket | null) => void
  clearMessages: () => void
}

export const useJarvisStore = create<JarvisState>()(
  subscribeWithSelector((set) => ({
    messages: [],
    goals: [],
    brokerEvents: [],
    isStreaming: false,
    sessionId: `session_${Date.now()}`,
    settings: {
      model: 'auto',
      temperature: 0.7,
      voice: 'en-US-AriaNeural',
      ttsEnabled: false,
      streamingEnabled: true,
    },
    systemHealth: {
      broker: false,
      llm_engine: false,
      memory: false,
      agent_core: false,
      voice: false,
      tool_system: false,
    },
    activeAgents: [],
    brokerWs: null,
    voiceWs: null,

    addMessage: (msg) =>
      set((s) => ({ messages: [...s.messages, msg] })),

    updateLastMessage: (content) =>
      set((s) => {
        const msgs = [...s.messages]
        if (msgs.length > 0) {
          msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], content }
        }
        return { messages: msgs }
      }),

    setStreaming: (v) => set({ isStreaming: v }),
    setGoals: (goals) => set({ goals }),
    updateGoal: (goal) =>
      set((s) => ({
        goals: s.goals.map((g) => (g.goal_id === goal.goal_id ? goal : g)),
      })),
    addBrokerEvent: (event) =>
      set((s) => ({
        brokerEvents: [event, ...s.brokerEvents].slice(0, 200),
      })),
    updateSettings: (settings) =>
      set((s) => ({ settings: { ...s.settings, ...settings } })),
    updateHealth: (health) =>
      set((s) => ({ systemHealth: { ...s.systemHealth, ...health } })),
    setActiveAgents: (agents) => set({ activeAgents: agents }),
    setBrokerWs: (ws) => set({ brokerWs: ws }),
    setVoiceWs: (ws) => set({ voiceWs: ws }),
    clearMessages: () => set({ messages: [] }),
  }))
)
