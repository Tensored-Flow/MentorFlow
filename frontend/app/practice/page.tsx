"use client";

import { motion } from "framer-motion";
import { Shuffle } from "lucide-react";
import SubjectGrid from "@/components/SubjectGrid";

export default function PracticePage() {
  return (
    <div className="min-h-screen pt-28 pb-20 px-6">
      <div className="max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            Choose a <span className="gradient-text">Subject</span>
          </h1>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto">
            Practice the same tasks our AI learns. Each subject contains multiple difficulty levels 
            with procedurally generated questions.
          </p>
        </motion.div>

        {/* Subject Grid */}
        <SubjectGrid />

        {/* Random Task */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="mt-12 text-center"
        >
          <p className="text-gray-500 mb-4">Or try something random</p>
          <button className="inline-flex items-center space-x-2 px-6 py-3 rounded-full border border-white/20 hover:bg-white/5 transition-colors">
            <Shuffle className="w-5 h-5" />
            <span>Generate Random Task</span>
          </button>
        </motion.div>

        {/* Stats Preview */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-16 p-6 rounded-2xl glass"
        >
          <h3 className="text-lg font-semibold mb-4">Your Progress</h3>
          <div className="grid grid-cols-4 gap-4 text-center">
            {[
              { label: "Tasks Completed", value: "0" },
              { label: "Accuracy", value: "0%" },
              { label: "Current Streak", value: "0" },
              { label: "Time Spent", value: "0m" },
            ].map((stat) => (
              <div key={stat.label}>
                <p className="text-2xl font-bold gradient-text">{stat.value}</p>
                <p className="text-gray-500 text-sm">{stat.label}</p>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
