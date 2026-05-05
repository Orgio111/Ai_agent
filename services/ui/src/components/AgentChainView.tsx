import { useCallback, useEffect } from 'react'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
  useNodesState,
  useEdgesState,
} from 'reactflow'
import 'reactflow/dist/style.css'
import { useJarvisStore } from '../stores/jarvisStore'
import axios from 'axios'

const AGENT_CORE_URL = import.meta.env.VITE_AGENT_CORE_URL || '/api'

const AGENT_COLORS: Record<string, string> = {
  planner: '#7c3aed',
  executor: '#2563eb',
  critic: '#dc2626',
  researcher: '#16a34a',
  optimizer: '#d97706',
}

export function AgentChainView() {
  const { messages, activeAgents } = useJarvisStore()
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])

  const buildFlowFromMessages = useCallback(() => {
    const latestWithSteps = [...messages]
      .reverse()
      .find((m) => m.agentSteps && m.agentSteps.length > 0)

    if (!latestWithSteps?.agentSteps) return

    const steps = latestWithSteps.agentSteps
    const newNodes: Node[] = []
    const newEdges: Edge[] = []

    const startNode: Node = {
      id: 'user_input',
      type: 'input',
      position: { x: 400, y: 0 },
      data: {
        label: (
          <div className="text-xs">
            <div className="font-bold text-gray-200">👤 User Input</div>
            <div className="text-gray-400 truncate max-w-40">
              {latestWithSteps?.content?.slice(0, 50)}...
            </div>
          </div>
        ),
      },
      style: { background: '#1f2937', border: '1px solid #374151', color: 'white', borderRadius: 8 },
    }
    newNodes.push(startNode)

    steps.forEach((step, i) => {
      const nodeId = `step_${i}`
      const color = AGENT_COLORS[step.agent] || '#6b7280'
      const x = (i % 3) * 300 + 100
      const y = Math.floor(i / 3) * 150 + 100

      newNodes.push({
        id: nodeId,
        position: { x, y },
        data: {
          label: (
            <div className="text-xs p-1">
              <div className="font-bold" style={{ color }}>
                {step.agent.toUpperCase()}
              </div>
              <div className="text-gray-400 truncate max-w-36">
                {step.output.slice(0, 60)}
              </div>
              <div className="text-gray-600 mt-1">
                {step.durationMs.toFixed(0)}ms · {step.tokens} tok
              </div>
            </div>
          ),
        },
        style: {
          background: '#111827',
          border: `2px solid ${color}`,
          color: 'white',
          borderRadius: 8,
          minWidth: 150,
        },
      })

      const sourceId = i === 0 ? 'user_input' : `step_${i - 1}`
      newEdges.push({
        id: `e_${i}`,
        source: sourceId,
        target: nodeId,
        style: { stroke: color, strokeWidth: 2 },
        animated: activeAgents.includes(step.agent),
      })
    })

    const resultNode: Node = {
      id: 'result',
      type: 'output',
      position: { x: 400, y: steps.length * 60 + 200 },
      data: {
        label: (
          <div className="text-xs">
            <div className="font-bold text-green-400">✅ Result</div>
            <div className="text-gray-400">
              Score: {((latestWithSteps?.criticScore || 0) * 100).toFixed(0)}%
            </div>
          </div>
        ),
      },
      style: { background: '#064e3b', border: '1px solid #10b981', color: 'white', borderRadius: 8 },
    }
    newNodes.push(resultNode)

    if (steps.length > 0) {
      newEdges.push({
        id: 'e_result',
        source: `step_${steps.length - 1}`,
        target: 'result',
        style: { stroke: '#10b981', strokeWidth: 2 },
      })
    }

    setNodes(newNodes)
    setEdges(newEdges)
  }, [messages, activeAgents, setNodes, setEdges])

  useEffect(() => {
    buildFlowFromMessages()
  }, [buildFlowFromMessages])

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-gray-800 flex items-center justify-between">
        <h2 className="text-lg font-bold text-cyan-400">Agent Chain Visualization</h2>
        <div className="flex gap-2">
          {Object.entries(AGENT_COLORS).map(([name, color]) => (
            <div key={name} className="flex items-center gap-1 text-xs">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-gray-400 capitalize">{name}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="flex-1 bg-gray-950">
        {nodes.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-600">
            <div className="text-center">
              <div className="text-4xl mb-4">🤖</div>
              <p>Agent chain visualization will appear here after you send a message.</p>
            </div>
          </div>
        ) : (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            fitView
            className="bg-gray-950"
          >
            <Background color="#374151" gap={20} />
            <Controls className="bg-gray-900 border-gray-700" />
            <MiniMap className="bg-gray-900" nodeColor={(n) => n.style?.borderColor as string || '#374151'} />
          </ReactFlow>
        )}
      </div>
    </div>
  )
}
