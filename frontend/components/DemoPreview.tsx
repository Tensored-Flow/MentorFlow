"use client";

import { motion } from "framer-motion";

export default function DemoPreview() {
  return (
    <section className="py-24 px-6">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            See It In <span className="gradient-text">Action</span>
          </h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            A glimpse into how MentorFlow generates and sequences tasks for optimal learning.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-6">
          {/* Task Card */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="p-6 rounded-2xl glass"
          >
            <div className="flex items-center space-x-2 mb-4">
              <span className="px-3 py-1 rounded-full bg-purple-500/20 text-purple-400 text-sm">
                var_trace
              </span>
              <span className="px-3 py-1 rounded-full bg-green-500/20 text-green-400 text-sm">
                easy
              </span>
            </div>
            <div className="bg-[#1a1a1a] rounded-lg p-4 font-mono text-sm mb-4">
              <p className="text-gray-300">a = 5</p>
              <p className="text-gray-300">b = 3</p>
              <p className="text-gray-300">a = a + b</p>
              <p className="text-purple-400 mt-2">What is a?</p>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {["8", "5", "3", "15"].map((choice, i) => (
                <button
                  key={i}
                  className={`py-2 rounded-lg font-mono ${
                    i === 0
                      ? "bg-green-500/20 text-green-400 border border-green-500/50"
                      : "bg-white/5 text-gray-400 hover:bg-white/10"
                  } transition-colors`}
                >
                  {choice}
                </button>
              ))}
            </div>
          </motion.div>

          {/* Teacher Decision */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="p-6 rounded-2xl glass"
          >
            <h3 className="text-lg font-semibold mb-4">🤖 Teacher Decision</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Strategy</span>
                <span className="text-purple-400">UCB1</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Selected Arm</span>
                <span className="text-white">var_trace_easy</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Exploration</span>
                <span className="text-blue-400">High</span>
              </div>
              <div className="mt-4 p-3 rounded-lg bg-purple-500/10 border border-purple-500/20">
                <p className="text-sm text-gray-300">
                  "Starting with easier tasks to build foundational understanding..."
                </p>
              </div>
            </div>
          </motion.div>

          {/* Student Response */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="p-6 rounded-2xl glass"
          >
            <h3 className="text-lg font-semibold mb-4">🧠 Student Response</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Answer</span>
                <span className="text-green-400">8 ✓</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Confidence</span>
                <span className="text-white">92%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Reward</span>
                <span className="text-green-400">+0.15</span>
              </div>
              <div className="mt-4">
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-400">Accuracy</span>
                  <span className="text-white">78%</span>
                </div>
                <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                  <div className="h-full w-[78%] bg-gradient-to-r from-purple-500 to-blue-500 rounded-full" />
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
