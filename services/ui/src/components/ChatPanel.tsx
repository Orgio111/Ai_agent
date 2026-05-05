import { useCallback, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { useJarvisStore, Message } from '../stores/jarvisStore'
import { VoiceButton } from './VoiceButton'
import axios from 'axios'

const AGENT_CORE_URL = import.meta.env.VITE_AGENT_CORE_URL || '/api'

export function ChatPanel() {
  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const {
    messages,
    isStreaming,
    sessionId,
    settings,
    addMessage,
    updateLastMessage,
    setStreaming,
  } = useJarvisStore()

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = useCallback(async () => {
    const text = input.trim()
    if (!text || isStreaming) return

    setInput('')
    const userMsg: Message = {
      id: `msg_${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date(),
    }
    addMessage(userMsg)

    const assistantMsg: Message = {
      id: `msg_${Date.now()}_a`,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    }
    addMessage(assistantMsg)
    setStreaming(true)

    try {
      if (settings.streamingEnabled) {
        const resp = await fetch(`${AGENT_CORE_URL}/run/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: text,
            session_id: sessionId,
            stream: true,
          }),
        })

        if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
        if (!resp.body) throw new Error('No response body')

        const reader = resp.body.getReader()
        const decoder = new TextDecoder()
        let accumulated = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          const lines = decoder.decode(value).split('\n')
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6).trim()
              if (data === '[DONE]') break
              try {
                const parsed = JSON.parse(data)
                if (parsed.delta) {
                  accumulated += parsed.delta
                  updateLastMessage(accumulated)
                }
              } catch {
                // Non-JSON chunk, append as-is
                accumulated += data
                updateLastMessage(accumulated)
              }
            }
          }
        }
      } else {
        const resp = await axios.post(`${AGENT_CORE_URL}/run`, {
          prompt: text,
          session_id: sessionId,
        })
        const data = resp.data
        updateLastMessage(data.result)

        // Update last message with full metadata
        const { messages: msgs, addMessage: add } = useJarvisStore.getState()
        const lastIdx = msgs.length - 1
        if (lastIdx >= 0) {
          const updated = {
            ...msgs[lastIdx],
            content: data.result,
            agentSteps: data.steps,
            criticScore: data.critic_score,
            totalTokens: data.total_tokens,
            durationMs: data.total_duration_ms,
          }
          useJarvisStore.setState({
            messages: [...msgs.slice(0, lastIdx), updated],
          })
        }
      }
    } catch (err) {
      updateLastMessage(`❌ Error: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setStreaming(false)
    }
  }, [input, isStreaming, sessionId, settings, addMessage, updateLastMessage, setStreaming])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const agentColor = (agent: string) => {
    const colors: Record<string, string> = {
      planner: 'text-purple-400',
      executor: 'text-blue-400',
      critic: 'text-red-400',
      researcher: 'text-green-400',
      optimizer: 'text-yellow-400',
    }
    return colors[agent] || 'text-gray-400'
  }

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-gray-600">
            <div className="text-4xl mb-4">⚡</div>
            <p className="text-lg">JARVIS Autonomous AI OS</p>
            <p className="text-sm mt-2">Ask me anything or give me a goal to accomplish.</p>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-4xl rounded-lg px-4 py-3 ${
                msg.role === 'user'
                  ? 'bg-cyan-700 text-white'
                  : 'bg-gray-800 text-gray-100'
              }`}
            >
              {/* Role label */}
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-bold text-gray-400 uppercase">
                  {msg.role === 'user' ? '👤 You' : '⚡ JARVIS'}
                </span>
                {msg.criticScore !== undefined && (
                  <span className={`text-xs ml-4 ${msg.criticScore >= 0.7 ? 'text-green-400' : 'text-red-400'}`}>
                    Score: {(msg.criticScore * 100).toFixed(0)}%
                  </span>
                )}
              </div>

              {/* Content */}
              <div className="prose prose-invert prose-sm max-w-none">
                <ReactMarkdown>{msg.content || '▋'}</ReactMarkdown>
              </div>

              {/* Agent steps accordion */}
              {msg.agentSteps && msg.agentSteps.length > 0 && (
                <details className="mt-3 border-t border-gray-700 pt-2">
                  <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-300">
                    🔍 Agent chain ({msg.agentSteps.length} steps
                    {msg.durationMs !== undefined ? ` · ${msg.durationMs.toFixed(0)}ms` : ''}
                    {msg.totalTokens !== undefined ? ` · ${msg.totalTokens} tokens` : ''})
                  </summary>
                  <div className="mt-2 space-y-2">
                    {msg.agentSteps.map((step, i) => (
                      <div key={i} className="text-xs bg-gray-900 rounded p-2">
                        <span className={`font-bold ${agentColor(step.agent)}`}>
                          [{step.agent.toUpperCase()}]
                        </span>{' '}
                        <span className="text-gray-400">{step.output.slice(0, 150)}...</span>
                        <span className="ml-2 text-gray-600">{step.durationMs.toFixed(0)}ms</span>
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </div>
          </div>
        ))}

        {isStreaming && (
          <div className="flex justify-start">
            <div className="bg-gray-800 rounded-lg px-4 py-3">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="p-4 border-t border-gray-800 bg-gray-900">
        <div className="flex gap-2 items-end">
          <VoiceButton />
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask JARVIS anything... (Enter to send, Shift+Enter for newline)"
            className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-sm text-gray-100 placeholder-gray-500 resize-none focus:outline-none focus:border-cyan-500 min-h-[52px] max-h-32"
            rows={1}
            disabled={isStreaming}
          />
          <button
            onClick={sendMessage}
            disabled={isStreaming || !input.trim()}
            className="px-6 py-3 bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg text-sm font-medium transition-colors"
          >
            {isStreaming ? '...' : 'Send'}
          </button>
        </div>
        <p className="text-xs text-gray-600 mt-2">
          Session: {useJarvisStore.getState().sessionId} · Model: {useJarvisStore.getState().settings.model}
        </p>
      </div>
    </div>
  )
}
