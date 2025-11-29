"use client";

import { motion } from "framer-motion";

export default function DashboardMock() {
  return (
    <div className="space-y-6">
      {/* Stats Row */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: "Progress", value: "24 / 50", color: "text-purple-400" },
          { label: "Accuracy", value: "76.2%", color: "text-green-400" },
          { label: "Current Task", value: "loop_count", color: "text-blue-400" },
        ].map((stat, index) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="p-4 rounded-xl glass text-center"
          >
            <p className="text-gray-400 text-sm">{stat.label}</p>
            <p className={`text-2xl font-bold ${stat.color}`}>{stat.value}</p>
          </motion.div>
        ))}
      </div>

      {/* Progress Bar */}
      <div className="p-4 rounded-xl glass">
        <div className="flex justify-between text-sm mb-2">
          <span className="text-gray-400">Training Progress</span>
          <span className="text-white">48%</span>
        </div>
        <div className="h-3 bg-white/10 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: "48%" }}
            transition={{ duration: 1, delay: 0.5 }}
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
            {[0.2, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.72, 0.74, 0.76].map((h, i) => (
              <motion.div
                key={i}
                initial={{ height: 0 }}
                animate={{ height: `${h * 100}%` }}
                transition={{ delay: 0.5 + i * 0.05, duration: 0.3 }}
                className="flex-1 bg-gradient-to-t from-purple-500 to-blue-500 rounded-t"
              />
            ))}
          </div>
        </div>

        {/* Rewards */}
        <div className="p-4 rounded-xl glass">
          <h4 className="text-sm font-semibold mb-4">🎯 Rewards</h4>
          <div className="h-32 flex items-center space-x-1">
            {[0.1, -0.05, 0.15, 0.08, -0.02, 0.12, 0.05, 0.18, -0.03, 0.1, 0.07, 0.09].map((r, i) => (
              <motion.div
                key={i}
                initial={{ height: 0 }}
                animate={{ height: `${Math.abs(r) * 300}%` }}
                transition={{ delay: 0.5 + i * 0.05, duration: 0.3 }}
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
          {Array.from({ length: 15 }).map((_, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8 + i * 0.03 }}
              className="h-8 rounded"
              style={{
                backgroundColor: `rgba(147, 51, 234, ${0.1 + Math.random() * 0.6})`,
              }}
            />
          ))}
        </div>
        <div className="flex justify-between mt-2 text-xs text-gray-500">
          <span>Easy</span>
          <span>Medium</span>
          <span>Hard</span>
        </div>
      </div>

      {/* Start Button */}
      <motion.button
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1 }}
        className="w-full py-4 rounded-xl bg-gradient-to-r from-purple-500 to-blue-500 text-white font-semibold text-lg hover:opacity-90 transition-opacity"
      >
        ▶️ Start Training
      </motion.button>
    </div>
  );
}
