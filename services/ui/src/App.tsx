import { useEffect, useState } from 'react'
import { ChatPanel } from './components/ChatPanel'
import { AgentChainView } from './components/AgentChainView'
import { MemoryInspector } from './components/MemoryInspector'
import { GoalManager } from './components/GoalManager'
import { SystemDashboard } from './components/SystemDashboard'
import { SettingsPanel } from './components/SettingsPanel'
import { LogStream } from './components/LogStream'
import { useJarvisStore } from './stores/jarvisStore'
import { useBrokerConnection } from './hooks/useBrokerConnection'
import { useHealthPoller } from './hooks/useHealthPoller'

type Tab = 'chat' | 'agents' | 'memory' | 'goals' | 'logs' | 'settings'

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('chat')
  const systemHealth = useJarvisStore((s) => s.systemHealth)

  useBrokerConnection()
  useHealthPoller()

  const healthDot = (ok: boolean) =>
    ok ? 'bg-green-400' : 'bg-red-500'

  const tabs: { id: Tab; label: string }[] = [
    { id: 'chat', label: '💬 Chat' },
    { id: 'agents', label: '🤖 Agents' },
    { id: 'memory', label: '🧠 Memory' },
    { id: 'goals', label: '🎯 Goals' },
    { id: 'logs', label: '📋 Logs' },
    { id: 'settings', label: '⚙️ Settings' },
  ]

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-gray-100 font-mono">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 bg-gray-900 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <span className="text-xl font-bold text-cyan-400">⚡ JARVIS</span>
          <span className="text-xs text-gray-500">Autonomous AI OS v1.0</span>
        </div>

        {/* System health indicators */}
        <div className="flex items-center gap-4 text-xs text-gray-400">
          {Object.entries(systemHealth).map(([service, ok]) => (
            <div key={service} className="flex items-center gap-1">
              <div className={`w-2 h-2 rounded-full ${healthDot(ok)}`} />
              <span className="capitalize">{service.replace('_', ' ')}</span>
            </div>
          ))}
        </div>
      </header>

      {/* Navigation */}
      <nav className="flex gap-1 px-4 py-2 bg-gray-900 border-b border-gray-800">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 rounded text-sm transition-colors ${
              activeTab === tab.id
                ? 'bg-cyan-600 text-white'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {/* Main content */}
      <main className="flex-1 overflow-hidden">
        {activeTab === 'chat' && <ChatPanel />}
        {activeTab === 'agents' && <AgentChainView />}
        {activeTab === 'memory' && <MemoryInspector />}
        {activeTab === 'goals' && <GoalManager />}
        {activeTab === 'logs' && <LogStream />}
        {activeTab === 'settings' && <SettingsPanel />}
      </main>
    </div>
  )
}
