"use client";

import { motion } from "framer-motion";
import SubjectGrid from "@/components/SubjectGrid";

export default function SubjectsPage() {
  return (
    <div className="min-h-screen bg-[#020013] text-white pt-28 pb-20 px-6">
      <div className="max-w-5xl mx-auto space-y-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center space-y-4"
        >
          <p className="text-sm uppercase tracking-[0.4em] text-white/60">Browse the curriculum</p>
          <h1 className="text-4xl md:text-5xl font-bold">
            All <span className="gradient-text">Subjects</span>
          </h1>
          <p className="text-gray-400 text-lg max-w-3xl mx-auto">
            Every subject mirrors the RL teacher's evaluation set. Pick one to dive into focused practice with
            dynamic prompts and adaptive explanations.
          </p>
        </motion.div>

        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
          <SubjectGrid />
        </motion.div>
      </div>
    </div>
  );
}
