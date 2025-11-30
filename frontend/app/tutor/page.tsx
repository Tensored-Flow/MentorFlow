"use client";

import { motion } from "framer-motion";
import { ChevronDown, Sparkles } from "lucide-react";
import Link from "next/link";

const weaknesses = [
  {
    title: "Loop Termination",
    detail: "You revisited loops often, but the inner condition occasionally skips termination checks, so the teacher flags a 2x slower exploration on loop-heavy sequences.",
  },
  {
    title: "Conditional Branching",
    detail: "Branch coverage dropped when nested `if` statements lacked fallback `else` logic, so the teacher suggests confirming both branches before finalizing answers.",
  },
  {
    title: "Variable Tracing",
    detail: "You spent extra steps deriving intermediate variable states, which indicates the RL agent should keep offering small trace puzzles to strengthen the intuition.",
  },
];

const explanations = [
  {
    title: "Why 2 × 3 + 4 equals 10",
    body: "Order of operations multiplies 2 and 3 first, then adds 4, so the correct total is 10. In the learner's attempt, addition happened first, creating an incorrect 8.",
  },
  {
    title: "Which index refers to the last list item",
    body: "Negative indices read from the end in Python. `arr[-1]` references the final value, guaranteeing you don't compute length each time.",
  },
];

const nextQuestion = {
  title: "Next Suggestion",
  prompt: "Trace the loop: `for i in range(4): if i % 2 == 0: total += i`",
  hint: "Inspect even numbers only; sum equals 0+2=2.",
};

export default function TutorModePage() {
  return (
    <div className="min-h-screen bg-[#020013] text-white pt-28 pb-20 px-6">
      <div className="max-w-5xl mx-auto space-y-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center space-y-3"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 border border-white/20 text-sm">
            <Sparkles className="w-4 h-4 text-purple-400" />
            AI Tutor Mode
          </div>
          <h1 className="text-4xl font-black">What the AI noticed after your last question</h1>
          <p className="text-white/70">Each attempt surfaces weaknesses, explanations, and the next question so you stay aligned with the RL teacher.</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass rounded-3xl border border-white/10 p-6 space-y-4"
        >
          <h2 className="text-lg font-semibold text-white/80">Weakness Summary</h2>
          <div className="space-y-3">
            {weaknesses.map((weakness) => (
              <div
                key={weakness.title}
                className="rounded-2xl bg-white/5 border border-white/10 p-4 text-white/80"
              >
                <p className="text-sm font-semibold text-white">{weakness.title}</p>
                <p className="text-sm text-white/60 mt-1">{weakness.detail}</p>
              </div>
            ))}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="space-y-3"
        >
          <h3 className="text-lg text-white/80">Explanation Vault</h3>
          <div className="space-y-3">
            {explanations.map((explanation) => (
              <details
                key={explanation.title}
                className="glass rounded-3xl border border-white/10 p-5"
              >
                <summary className="flex items-center justify-between cursor-pointer text-lg font-semibold">
                  <span>{explanation.title}</span>
                  <ChevronDown className="w-5 h-5 text-white/60" />
                </summary>
                <p className="text-sm text-white/70 mt-3 leading-relaxed">{explanation.body}</p>
              </details>
            ))}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="glass rounded-3xl border border-white/10 p-5 space-y-4"
        >
          <p className="text-sm uppercase tracking-[0.4em] text-white/50">Next Question</p>
          <div className="text-lg font-semibold text-white">{nextQuestion.prompt.trim()}</div>
          <p className="text-white/60 text-sm">Hint: {nextQuestion.hint}</p>
          <div className="flex flex-wrap gap-4">
            <Link
              href="/practice"
              className="px-6 py-3 rounded-full bg-gradient-to-r from-purple-500 to-blue-500 text-sm font-semibold"
            >
              Continue Learning
            </Link>
            <Link
              href="/practice"
              className="px-6 py-3 rounded-full border border-white/30 text-sm font-semibold"
            >
              Review last answer
            </Link>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
