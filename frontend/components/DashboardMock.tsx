"use client";

import { motion } from "framer-motion";
import { useState, useEffect, useRef } from "react";
import { Play, Square, Loader2 } from "lucide-react";

const API_BASE = "http://localhost:5050";

interface TrainingState {
  running: boolean;
  iteration: number;
  total_iterations: number;
  progress: number;
  accuracy: number;
  last_task: string;
  learning_curve: number[];
  rewards: number[];
  heatmap: number[][];
}

export default function DashboardMock() {
  const [isTraining, setIsTraining] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [state, setState] = useState<TrainingState>({
    running: false,
    iteration: 0,
    total_iterations: 50,
    progress: 0,
    accuracy: 0,
    last_task: "—",
    learning_curve: [0.2, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.72, 0.74, 0.76],
    rewards: [0.1, -0.05, 0.15, 0.08, -0.02, 0.12, 0.05, 0.18, -0.03, 0.1, 0.07, 0.09],
    heatmap: Array(5).fill(null).map(() => Array(3).fill(0.3)),
  });
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  const startTraining = async () => {
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/train/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ num_steps: 50, strategy: "ucb1" }),
      });
      if (res.ok) {
        setIsTraining(true);
        pollStatus();
      } else {
        console.error("Start failed:", await res.text());
      }
    } catch (err) {
      console.error("Failed to start training:", err);
    }
    setIsLoading(false);
  };

  const stopTraining = async () => {
    try {
      await fetch(`${API_BASE}/api/train/stop`, { method: "POST" });
      setIsTraining(false);
      if (pollRef.current) clearTimeout(pollRef.current);
    } catch (err) {
      console.error("Failed to stop training:", err);
    }
  };

  const pollStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/train/status`);
      if (res.ok) {
        const data = await res.json();
        const teacher = data.teacher || {};
        const meta = teacher.metadata || {};
        const heatmapData = teacher.heatmap?.cumulative || [];
        
        const step = meta.step || 0;
        const totalSteps = meta.total_steps || 50;
        const accuracies = teacher.accuracies || [];
        const rewards = teacher.rewards || [];
        const taskData = teacher.current_task;
        const currentTask = taskData?.arm_name || (typeof taskData === "string" ? taskData : "—");
        
        setState(prev => ({
          ...prev,
          running: meta.running || false,
          iteration: step,
          total_iterations: totalSteps,
          progress: totalSteps > 0 ? step / totalSteps : 0,
          accuracy: accuracies.length > 0 ? accuracies[accuracies.length - 1] : 0,
          last_task: currentTask,
          learning_curve: accuracies.length > 0 ? accuracies : prev.learning_curve,
          rewards: rewards.length > 0 ? rewards.slice(-12) : prev.rewards,
          heatmap: heatmapData.length > 0 ? heatmapData : prev.heatmap,
        }));

        if (meta.running) {
          pollRef.current = setTimeout(pollStatus, 500);
        } else if (isTraining && step > 0) {
          // Training finished
          setIsTraining(false);
        }
      }
    } catch (err) {
      console.error("Polling error:", err);
      if (isTraining) {
        pollRef.current = setTimeout(pollStatus, 1000);
      }
    }
  };

  useEffect(() => {
    return () => {
      if (pollRef.current) clearTimeout(pollRef.current);
    };
  }, []);

  const stats = [
    { label: "Progress", value: `${state.iteration} / ${state.total_iterations}`, color: "text-purple-400" },
    { label: "Accuracy", value: `${(state.accuracy * 100).toFixed(1)}%`, color: "text-green-400" },
    { label: "Current Task", value: state.last_task, color: "text-blue-400" },
  ];

  return (
    <div className="space-y-6">
      {/* Stats Row */}
      <div className="grid grid-cols-3 gap-4">
        {stats.map((stat, index) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="p-4 rounded-xl glass text-center"
          >
            <p className="text-gray-400 text-sm">{stat.label}</p>
            <p className={`text-2xl font-bold ${stat.color} truncate`}>{stat.value}</p>
          </motion.div>
        ))}
      </div>

      {/* Progress Bar */}
      <div className="p-4 rounded-xl glass">
        <div className="flex justify-between text-sm mb-2">
          <span className="text-gray-400">Training Progress</span>
          <span className="text-white">{Math.round(state.progress * 100)}%</span>
        </div>
        <div className="h-3 bg-white/10 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${state.progress * 100}%` }}
            transition={{ duration: 0.3 }}
            className="h-full bg-gradient-to-r from-purple-500 to-blue-500 rounded-full"
          />
        </div>
      </div>

      {/* Mock Charts */}
      <div className="grid md:grid-cols-2 gap-4">
        {/* Learning Curve */}
        <div className="p-4 rounded-xl glass">
          <h4 className="text-sm font-semibold mb-4">📈 Learning Curve</h4>
          <div className="h-32 flex items-end space-x-1">
            {state.learning_curve.slice(-12).map((h, i) => (
              <motion.div
                key={i}
                initial={{ height: 0 }}
                animate={{ height: `${Math.min(h, 1) * 100}%` }}
                transition={{ duration: 0.3 }}
                className="flex-1 bg-gradient-to-t from-purple-500 to-blue-500 rounded-t"
              />
            ))}
          </div>
        </div>

        {/* Rewards */}
        <div className="p-4 rounded-xl glass">
          <h4 className="text-sm font-semibold mb-4">🎯 Rewards</h4>
          <div className="h-32 flex items-center space-x-1">
            {state.rewards.slice(-12).map((r, i) => (
              <motion.div
                key={i}
                initial={{ height: 0 }}
                animate={{ height: `${Math.abs(r) * 300}%` }}
                transition={{ duration: 0.3 }}
                className={`flex-1 rounded ${r >= 0 ? "bg-green-500" : "bg-red-500"}`}
                style={{ alignSelf: r >= 0 ? "flex-end" : "flex-start" }}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Heatmap Preview */}
      <div className="p-4 rounded-xl glass">
        <h4 className="text-sm font-semibold mb-4">🗺️ Curriculum Heatmap</h4>
        <div className="grid grid-cols-3 gap-1">
          {state.heatmap.flat().map((val, i) => (
            <motion.div
              key={i}
              className="h-8 rounded"
              animate={{
                backgroundColor: `rgba(147, 51, 234, ${0.1 + Math.min(val, 1) * 0.7})`,
              }}
              transition={{ duration: 0.3 }}
            />
          ))}
        </div>
        <div className="flex justify-between mt-2 text-xs text-gray-500">
          <span>Easy</span>
          <span>Medium</span>
          <span>Hard</span>
        </div>
      </div>

      {/* Start/Stop Button */}
      <motion.button
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        onClick={isTraining ? stopTraining : startTraining}
        disabled={isLoading}
        className={`w-full py-4 rounded-xl text-white font-semibold text-lg transition-all flex items-center justify-center space-x-2 ${
          isTraining
            ? "bg-gradient-to-r from-red-500 to-orange-500 hover:opacity-90"
            : "bg-gradient-to-r from-purple-500 to-blue-500 hover:opacity-90"
        } ${isLoading ? "opacity-50 cursor-not-allowed" : ""}`}
      >
        {isLoading ? (
          <Loader2 className="w-5 h-5 animate-spin" />
        ) : isTraining ? (
          <>
            <Square className="w-5 h-5" />
            <span>Stop Training</span>
          </>
        ) : (
          <>
            <Play className="w-5 h-5" />
            <span>Start Training</span>
          </>
        )}
      </motion.button>

      {/* Status indicator */}
      {isTraining && (
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center text-sm text-gray-400"
        >
          Training in progress... Polling backend at localhost:5050
        </motion.p>
      )}
    </div>
  );
}
