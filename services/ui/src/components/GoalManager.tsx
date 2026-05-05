import { useEffect, useState } from 'react'
import axios from 'axios'
import { Goal, useJarvisStore } from '../stores/jarvisStore'

const AGENT_CORE_URL = import.meta.env.VITE_AGENT_CORE_URL || '/api'

const STATUS_COLORS: Record<string, string> = {
  pending: 'text-yellow-400 bg-yellow-900',
  running: 'text-blue-400 bg-blue-900',
  completed: 'text-green-400 bg-green-900',
  failed: 'text-red-400 bg-red-900',
  cancelled: 'text-gray-400 bg-gray-800',
}

export function GoalManager() {
  const { goals, setGoals, updateGoal } = useJarvisStore()
  const [newGoalDesc, setNewGoalDesc] = useState('')
  const [creating, setCreating] = useState(false)
  const [selectedGoal, setSelectedGoal] = useState<Goal | null>(null)

  useEffect(() => {
    const fetchGoals = async () => {
      try {
        const resp = await axios.get(`${AGENT_CORE_URL}/goals`)
        setGoals(resp.data.goals || [])
      } catch { /* service may not be running */ }
    }
    fetchGoals()
    const interval = setInterval(fetchGoals, 5000)
    return () => clearInterval(interval)
  }, [setGoals])

  const createGoal = async () => {
    if (!newGoalDesc.trim() || creating) return
    setCreating(true)
    try {
      const resp = await axios.post(`${AGENT_CORE_URL}/goals`, {
        description: newGoalDesc.trim(),
        priority: 5,
        auto_resume: true,
      })
      const goal = resp.data as Goal
      useJarvisStore.setState((s) => ({ goals: [...s.goals, goal] }))
      setNewGoalDesc('')
      setSelectedGoal(goal)
    } catch (err) {
      alert(`Failed to create goal: ${err}`)
    } finally {
      setCreating(false)
    }
  }

  const cancelGoal = async (goalId: string) => {
    try {
      await axios.delete(`${AGENT_CORE_URL}/goals/${goalId}`)
      useJarvisStore.setState((s) => ({
        goals: s.goals.map((g) =>
          g.goal_id === goalId ? { ...g, status: 'cancelled' as const } : g
        ),
      }))
    } catch { /* ignore */ }
  }

  return (
    <div className="flex h-full">
      {/* Goal list */}
      <div className="w-1/3 border-r border-gray-800 flex flex-col">
        <div className="p-4 border-b border-gray-800">
          <h2 className="text-lg font-bold text-cyan-400 mb-3">🎯 Autonomous Goals</h2>
          <div className="flex gap-2">
            <input
              value={newGoalDesc}
              onChange={(e) => setNewGoalDesc(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && createGoal()}
              placeholder="Describe a long-term goal..."
              className="flex-1 bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-cyan-500"
            />
            <button
              onClick={createGoal}
              disabled={creating || !newGoalDesc.trim()}
              className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded text-sm"
            >
              {creating ? '...' : '➕'}
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          {goals.length === 0 ? (
            <div className="p-4 text-gray-600 text-sm text-center">
              No goals yet. Create one above.
            </div>
          ) : (
            goals.map((goal) => (
              <div
                key={goal.goal_id}
                onClick={() => setSelectedGoal(goal)}
                className={`p-4 border-b border-gray-800 cursor-pointer hover:bg-gray-900 ${
                  selectedGoal?.goal_id === goal.goal_id ? 'bg-gray-900' : ''
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className={`text-xs px-2 py-0.5 rounded ${STATUS_COLORS[goal.status] || 'text-gray-400'}`}>
                    {goal.status}
                  </span>
                  <span className="text-xs text-gray-500">
                    {goal.tasks.length} tasks
                  </span>
                </div>
                <p className="text-sm text-gray-200 truncate">{goal.description}</p>
                {goal.status === 'running' && (
                  <div className="mt-2 bg-gray-800 rounded-full h-1.5">
                    <div
                      className="bg-blue-500 h-1.5 rounded-full transition-all"
                      style={{ width: `${goal.progress_pct}%` }}
                    />
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Goal detail */}
      <div className="flex-1 p-4 overflow-y-auto">
        {!selectedGoal ? (
          <div className="flex items-center justify-center h-full text-gray-600">
            <p>Select a goal to view details</p>
          </div>
        ) : (
          <div>
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="text-xl font-bold text-gray-100">{selectedGoal.description}</h3>
                <p className="text-xs text-gray-500 mt-1">
                  Created: {new Date(selectedGoal.created_at).toLocaleString()}
                </p>
              </div>
              {selectedGoal.status === 'running' && (
                <button
                  onClick={() => cancelGoal(selectedGoal.goal_id)}
                  className="px-3 py-1 bg-red-800 hover:bg-red-700 text-red-300 text-sm rounded"
                >
                  Cancel
                </button>
              )}
            </div>

            {/* Progress */}
            <div className="mb-4 bg-gray-800 rounded p-3">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">Progress</span>
                <span className="text-gray-200">{selectedGoal.progress_pct.toFixed(1)}%</span>
              </div>
              <div className="bg-gray-700 rounded-full h-2">
                <div
                  className="bg-cyan-500 h-2 rounded-full transition-all"
                  style={{ width: `${selectedGoal.progress_pct}%` }}
                />
              </div>
            </div>

            {/* Task list */}
            <h4 className="text-sm font-bold text-gray-400 mb-2">Task Graph</h4>
            <div className="space-y-2">
              {selectedGoal.tasks.map((task) => (
                <div
                  key={task.task_id}
                  className="bg-gray-800 rounded p-3 border-l-4 border-l-gray-600"
                >
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-gray-200">{task.task_id}</span>
                    <span className={`text-xs px-2 py-0.5 rounded ${STATUS_COLORS[task.status] || 'text-gray-400 bg-gray-700'}`}>
                      {task.status}
                    </span>
                  </div>
                  <p className="text-xs text-gray-400 mt-1">{task.description}</p>
                  {task.depends_on.length > 0 && (
                    <p className="text-xs text-gray-600 mt-1">
                      Depends on: {task.depends_on.join(', ')}
                    </p>
                  )}
                  {task.result && (
                    <p className="text-xs text-green-400 mt-1 truncate">{task.result.slice(0, 100)}</p>
                  )}
                </div>
              ))}
            </div>

            {/* Result */}
            {selectedGoal.result && (
              <div className="mt-4 bg-green-900 border border-green-700 rounded p-3">
                <h4 className="text-sm font-bold text-green-400 mb-2">✅ Result</h4>
                <p className="text-sm text-green-200 whitespace-pre-wrap">{selectedGoal.result}</p>
              </div>
            )}
            {selectedGoal.error && (
              <div className="mt-4 bg-red-900 border border-red-700 rounded p-3">
                <h4 className="text-sm font-bold text-red-400 mb-2">❌ Error</h4>
                <p className="text-sm text-red-200">{selectedGoal.error}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
