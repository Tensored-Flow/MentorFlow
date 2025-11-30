"use client";

import { motion } from "framer-motion";
import { Shuffle } from "lucide-react";
import SubjectGrid from "@/components/SubjectGrid";

const statHighlights = [
  { label: "Streak", value: "12 days" },
  { label: "XP", value: "2,400" },
  { label: "Badges", value: "3 unlocked" },
  { label: "Masters", value: "5 topics" },
];

export default function PracticePage() {
  return (
    <div className="min-h-screen bg-[#020013] text-white pt-28 pb-20 px-6">
      <div className="max-w-5xl mx-auto space-y-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-4"
        >
          <p className="text-sm uppercase tracking-[0.4em] text-white/60">
            Practice the same RL curriculum
          </p>
          <h1 className="text-4xl md:text-5xl font-bold">
            Choose a <span className="gradient-text">Subject</span>
          </h1>
          <p className="text-gray-400 text-lg max-w-3xl mx-auto">
            Every subject mirrors the tasks the AI teacher uses to evaluate students. Dive into adaptive
            prompts and grow your streaks with personalized XP rewards.
          </p>
        </motion.div>

        <motion.div
          className="grid sm:grid-cols-2 gap-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          {statHighlights.map((stat) => (
            <div key={stat.label} className="glass rounded-3xl border border-white/10 p-5">
              <p className="text-sm uppercase tracking-[0.3em] text-white/40">{stat.label}</p>
              <p className="text-3xl font-semibold mt-2 text-white">{stat.value}</p>
            </div>
          ))}
        </motion.div>

        <motion.div
          className="glass rounded-3xl p-6 border border-white/10"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-sm text-white/60 uppercase tracking-[0.3em]">Continue</p>
              <p className="text-xl font-semibold">Continue where you left off</p>
            </div>
            <button className="px-5 py-2 rounded-full bg-gradient-to-r from-purple-500 to-blue-500 text-sm font-semibold">
              Continue Learning
            </button>
          </div>
          <div className="mt-4 text-sm text-white/70">
            <p>Current Task: Python fundamentals · Loop tracing</p>
            <p>Next recommended: JavaScript conditionals</p>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="space-y-6"
        >
          <SubjectGrid />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="text-center space-y-3"
        >
          <p className="text-white/70">Or try something random</p>
          <button className="inline-flex items-center space-x-2 px-6 py-3 rounded-full border border-white/30 hover:bg-white/10 transition-colors">
            <Shuffle className="w-5 h-5" />
            <span>Generate Random Task</span>
          </button>
        </motion.div>
      </div>
    </div>
  );
}
