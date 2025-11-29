"use client";

import { motion } from "framer-motion";
import DashboardMock from "@/components/DashboardMock";

export default function TrainingPage() {
  return (
    <div className="min-h-screen pt-28 pb-20 px-6">
      <div className="max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            Watch the <span className="gradient-text">AI Learn</span>
          </h1>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto">
            Observe our meta-curriculum system in action. The Teacher agent learns to select 
            optimal tasks for the Student agent using bandit algorithms.
          </p>
        </motion.div>

        {/* Strategy Selector */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-8 p-6 rounded-2xl glass"
        >
          <div className="flex flex-wrap items-end gap-6">
            <div className="flex-1 min-w-[200px]">
              <label className="block text-sm text-gray-400 mb-2">Teacher Strategy</label>
              <select className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 focus:border-purple-500 focus:outline-none transition-colors">
                <option value="ucb1">UCB1 (Exploration)</option>
                <option value="thompson">Thompson Sampling</option>
                <option value="epsilon_greedy">ε-Greedy</option>
                <option value="random">Random (Baseline)</option>
              </select>
            </div>
            <div className="w-32">
              <label className="block text-sm text-gray-400 mb-2">Steps</label>
              <input
                type="number"
                defaultValue={30}
                min={10}
                max={200}
                className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 focus:border-purple-500 focus:outline-none transition-colors"
              />
            </div>
          </div>
        </motion.div>

        {/* Dashboard */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <DashboardMock />
        </motion.div>

        {/* Info Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mt-12 grid md:grid-cols-3 gap-4"
        >
          {[
            {
              title: "Teacher Agent",
              description: "Uses multi-armed bandit algorithms to select which task type to train the student on next.",
            },
            {
              title: "Student Agent",
              description: "A PPO neural network that learns to solve coding & STEM tasks from the selected curriculum.",
            },
            {
              title: "Meta-Learning",
              description: "The teacher learns to teach by observing how student performance changes after each task.",
            },
          ].map((card, i) => (
            <div key={card.title} className="p-5 rounded-xl glass">
              <h4 className="font-semibold mb-2">{card.title}</h4>
              <p className="text-gray-400 text-sm">{card.description}</p>
            </div>
          ))}
        </motion.div>
      </div>
    </div>
  );
}
