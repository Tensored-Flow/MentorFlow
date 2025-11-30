"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { TrendingUp, Activity, Sparkles, RefreshCcw } from "lucide-react";
import SkillDifficultyBar from "@/components/SkillDifficultyBar";
import {
  startTraining,
  stopTraining,
  trainingStep,
  getTrainingStatus,
  getTrainingProgress,
} from "@/lib/trainApi";

type TrainingResponse = {
  teacher: {
    accuracies: number[];
    rewards: number[];
    selected_arms: number[];
    arm_names: string[];
    final_accuracy: number;
    current_task: { arm_name: string; prompt: string; choices: string[] } | null;
    heatmap: {
      cumulative: number[][];
      num_arms: number;
      num_steps: number;
      arm_labels: string[];
    };
    family_totals?: number[];
    family_names?: string[];
    metadata: {
      step: number;
      total_steps: number;
      running: boolean;
    };
  };
};

type ProgressResponse = {
  step: number;
  total_steps: number;
  running: boolean;
};

const POLL_INTERVAL = 300;
const NUM_DIFFICULTIES = 5;

const buildPath = (values: number[], width: number, height: number) => {
  if (!values.length) return "";
  const sanitized = values.length === 1 ? [...values, values[0]] : values;
  const max = Math.max(...sanitized, 1);
  const step = width / (sanitized.length - 1 || 1);
  const path = sanitized
    .map((value, index) => {
      const x = index * step;
      const y = height - (Math.min(value, max) / max) * height;
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
  return path;
};

export default function TrainingPage() {
  const [statusData, setStatusData] = useState<TrainingResponse | null>(null);
  const [progress, setProgress] = useState<ProgressResponse>({ step: 0, total_steps: 1, running: false });
  const [isTraining, setIsTraining] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  const refreshStatus = async () => {
    try {
      const [statusResponse, progressResponse] = await Promise.all([
        getTrainingStatus(),
        getTrainingProgress(),
      ]);
      setStatusData(statusResponse as TrainingResponse);
      setProgress(progressResponse as ProgressResponse);
      setIsTraining(progressResponse.running);
      if (!progressResponse.running && pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "Failed to fetch status");
    }
  };

  const startPolling = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(refreshStatus, POLL_INTERVAL);
    refreshStatus();
  };

  const handleStart = async () => {
    setError(null);
    setIsStarting(true);
    try {
      await startTraining({ num_steps: 40, strategy: "ucb1" });
      await trainingStep();
      startPolling();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to start training");
    } finally {
      setIsStarting(false);
    }
  };

  const handleStop = async () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    try {
      await stopTraining();
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    refreshStatus();
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
      }
    };
  }, []);

  const metadata = statusData?.teacher.metadata ?? { step: 0, total_steps: 1, running: false };
  const progressValue = Math.min(100, (metadata.step / Math.max(metadata.total_steps, 1)) * 100);

  const currentDifficultyLevel = useMemo(() => {
    const arms = statusData?.teacher.selected_arms ?? [];
    if (!arms.length) return 1;
    const lastArm = arms[arms.length - 1];
    return (lastArm % NUM_DIFFICULTIES) + 1;
  }, [statusData]);

  const accuracies = statusData?.teacher.accuracies ?? [];
  const rewards = statusData?.teacher.rewards ?? [];
  const currentTask = statusData?.teacher.current_task;
  const uniqueArms = new Set(statusData?.teacher.selected_arms ?? []).size;
  const totalArms = statusData?.teacher.heatmap.num_arms ?? 0;
  const explorationRate = totalArms ? Math.round((uniqueArms / totalArms) * 100) : 0;

  const familyTotals = statusData?.teacher.family_totals ?? [];
  const familyNames = statusData?.teacher.family_names ?? [];
  const totalSelections = familyTotals.reduce((sum, value) => sum + value, 0) || 1;
  const distribution = familyTotals.map((value, index) => ({
    label: familyNames[index] ?? `Family ${index + 1}`,
    value: Math.round((value / totalSelections) * 100),
  }));

  const learningCurvePath = useMemo(() => buildPath(accuracies, 320, 140), [accuracies]);
  const rewardPath = useMemo(() => buildPath(rewards, 320, 120), [rewards]);

  return (
    <div className="min-h-screen bg-[#020013] text-white pt-28 pb-20 px-6">
      <div className="max-w-6xl mx-auto space-y-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-3"
        >
          <p className="text-sm uppercase tracking-[0.4em] text-white/50">insight studio</p>
          <h1 className="text-4xl md:text-5xl font-black">
            Watch the <span className="gradient-text">AI Learn</span>
          </h1>
          <p className="text-white/70 max-w-2xl mx-auto">
            Monitor how the Teacher agent balances exploration/exploitation, updates reward estimates, and tells
            you the next task that maximizes learning.
          </p>
        </motion.div>

        <motion.div
          className="flex flex-col gap-3"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-sm uppercase tracking-[0.3em] text-white/60">Progress</p>
              <p className="text-lg font-semibold">
                Step {metadata.step} of {metadata.total_steps}
              </p>
            </div>
            <div className="flex gap-3">
              <button
                onClick={handleStart}
                disabled={metadata.running || isStarting}
                className="px-6 py-3 rounded-full bg-gradient-to-r from-purple-500 to-blue-500 text-sm font-semibold shadow-[0_10px_30px_rgba(99,102,241,0.4)] disabled:opacity-40"
              >
                {metadata.running ? "Training..." : "Start Training"}
              </button>
              <button
                onClick={handleStop}
                className="px-6 py-3 rounded-full border border-white/30 text-sm font-semibold"
              >
                Stop
              </button>
            </div>
          </div>
          <div className="h-2 w-full rounded-full bg-white/10 overflow-hidden">
            <div className="h-full rounded-full bg-gradient-to-r from-purple-500 to-blue-500" style={{ width: `${progressValue}%` }} />
          </div>
          {error && <p className="text-sm text-red-400">{error}</p>}
        </motion.div>

        <motion.div
          className="grid lg:grid-cols-[1.3fr_0.7fr] gap-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="glass rounded-3xl border border-white/10 p-6 relative overflow-hidden">
            <div className="absolute inset-0 rounded-3xl border border-gradient-to-br from-purple-500 to-blue-500 opacity-20 pointer-events-none" />
            <div className="relative space-y-4">
              <div className="flex items-center gap-3 text-white/70">
                <TrendingUp className="w-5 h-5 text-purple-300" />
                <span className="uppercase tracking-[0.3em] text-xs">Learning Curve</span>
              </div>
              <div className="h-60 w-full bg-gradient-to-b from-white/5 to-transparent rounded-2xl overflow-hidden relative">
                <svg viewBox="0 0 320 160" className="w-full h-full">
                  {learningCurvePath && (
                    <path d={learningCurvePath} stroke="#a855f7" strokeWidth="3" fill="none" strokeLinecap="round" />
                  )}
                </svg>
              </div>
              <p className="text-sm text-white/60">
                Accuracy updates are driven by the RL teacher’s evaluations. Expect the glide to stabilize as the teacher converges.
              </p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="glass rounded-3xl border border-white/10 p-5 space-y-3">
              <div className="flex items-center justify-between text-sm text-white/70">
                <span>Teacher Insights</span>
                <Sparkles className="w-4 h-4 text-purple-300" />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-base font-semibold">
                  <span className="text-white/80">Exploration Rate</span>
                  <span>{explorationRate}%</span>
                </div>
                <div className="flex items-center justify-between text-base font-semibold">
                  <span className="text-white/80">Exploited Arms</span>
                  <span>
                    {uniqueArms} of {totalArms}
                  </span>
                </div>
                <div className="flex items-center justify-between text-base font-semibold">
                  <span className="text-white/80">Reward Trend</span>
                  <span>{(rewards[rewards.length - 1] ?? 0).toFixed(2)}</span>
                </div>
              </div>
            </div>

            <SkillDifficultyBar level={currentDifficultyLevel} compact />

            <div className="glass rounded-3xl border border-white/10 p-5 space-y-3">
              <p className="text-sm uppercase tracking-[0.3em] text-white/50">Task Family Distribution</p>
              <div className="space-y-3">
                {distribution.map((family) => (
                  <div key={family.label} className="space-y-1">
                    <div className="flex items-center justify-between text-sm text-white/70">
                      <span>{family.label}</span>
                      <span>{family.value}%</span>
                    </div>
                    <div className="h-2 rounded-full bg-white/10 overflow-hidden">
                      <div className="h-full rounded-full bg-gradient-to-r from-purple-500 to-blue-500" style={{ width: `${family.value}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass rounded-3xl border border-white/20 p-6 space-y-4"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm uppercase tracking-[0.3em] text-white/60">Current Task</p>
              <h2 className="text-2xl font-semibold text-white/90">
                {currentTask?.prompt ?? "Waiting for the teacher to pick a task..."}
              </h2>
            </div>
            <span className={`text-xs rounded-full px-3 py-1 ${metadata.running ? "bg-gradient-to-r from-purple-500 to-blue-500" : "bg-white/10"}`}>
              {metadata.running ? "Active" : "Idle"}
            </span>
          </div>
          <p className="text-white/70">
            {currentTask?.arm_name ? `Arm: ${currentTask.arm_name}` : "A task preview will appear here when training starts."}
          </p>
          <div className="h-32 w-full bg-[#0b0b17] rounded-2xl border border-white/10 p-4 grid grid-cols-2 gap-2">
            {currentTask?.choices?.map((choice, index) => (
              <div key={choice} className="rounded-xl bg-white/5 flex items-center justify-center text-sm text-white/70">{choice}</div>
            ))}
          </div>
          <div className="glass rounded-2xl border border-white/10 p-4">
            <div className="flex items-center justify-between text-sm text-white/60 mb-2">
              <span>Accuracy</span>
              <span>{(statusData?.teacher.final_accuracy ?? 0).toFixed(2)}%</span>
            </div>
            <div className="h-1 rounded-full bg-white/10 overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-purple-500 to-blue-500"
                style={{ width: `${Math.min(100, (statusData?.teacher.final_accuracy ?? 0) * 100)}%` }}
              />
            </div>
          </div>
          <div className="space-y-3">
            <div className="flex items-center justify-between text-sm text-white/70">
              <span>Reward Graph</span>
              <span>{(rewards[rewards.length - 1] ?? 0).toFixed(2)}</span>
            </div>
            <div className="h-32 w-full bg-gradient-to-b from-white/5 to-transparent rounded-2xl overflow-hidden relative">
              <svg viewBox="0 0 320 120" className="w-full h-full">
                {rewardPath && (
                  <path d={rewardPath} stroke="#22d3ee" strokeWidth="3" fill="none" strokeLinecap="round" />
                )}
              </svg>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
