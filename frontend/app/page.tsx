"use client";

import { motion } from "framer-motion";
import Link from "next/link";

const productCards = [
  {
    title: "Practice Hub",
    description: "Dynamic topics + instant feedback that mirror the AI teacher's curriculum.",
    accent: "from-purple-500 to-blue-500",
  },
  {
    title: "Training Studio",
    description: "Watch the RL teacher adjust exploration vs exploitation in real time.",
    accent: "from-indigo-500 to-purple-600",
  },
  {
    title: "Tutor Mode",
    description: "Step-by-step explanations, weakness tracking, and smart question suggestions.",
    accent: "from-emerald-500 to-cyan-500",
  },
];

const testimonials = [
  {
    quote: "MentorFlow feels like having a mentor review every answer instantly.",
    author: "Sophia, Algorithmic Learner",
  },
  {
    quote: "The RL teacher adapts so fast that I never feel stuck on a topic.",
    author: "Devon, Data Sciences",
  },
  {
    quote: "I love how explanations break down the problem before suggesting the next step.",
    author: "Maya, Competitive Coder",
  },
];

export default function Home() {
  return (
    <div className="bg-[#020013] text-white min-h-screen relative overflow-hidden">
      {/* animated gradient background */}
      <div className="absolute inset-0 -z-10">
        <motion.div
          className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(147,63,255,0.25),_transparent_55%)] animate-pulse"
          style={{ animationDuration: "8s" }}
        />
        <motion.div
          className="absolute inset-0 bg-[radial-gradient(circle_at_bottom,_rgba(59,130,246,0.2),_transparent_55%)]"
          animate={{ opacity: [0.8, 0.5, 0.8] }}
          transition={{ duration: 12, repeat: Infinity }}
        />
      </div>

      <main className="relative z-10">
        <section className="min-h-screen flex flex-col justify-center px-6 py-12">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="max-w-4xl mx-auto text-center space-y-6"
          >
            <p className="text-sm uppercase tracking-[0.5em] text-white/60">AI-first learning</p>
            <motion.h1
              initial={{ scale: 0.95 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.6 }}
              className="text-5xl md:text-7xl font-black leading-tight"
            >
              MentorFlow
            </motion.h1>

            <p className="text-xl md:text-2xl text-gradient font-semibold">
              Your Personal AI Tutor That Learns How to Teach.
            </p>

            <p className="text-base md:text-lg text-white/70 max-w-3xl mx-auto">
              Adaptive RL-powered curricula tailored to you, combining Teacher + Student agents with LLM
              insights to keep every question meaningful.
            </p>

            <div className="flex flex-wrap items-center justify-center gap-4">
              <Link
                href="/practice"
                className="relative inline-flex items-center justify-center px-8 py-4 rounded-full bg-gradient-to-r from-purple-500 to-blue-500 text-lg font-semibold shadow-[0_0_25px_rgba(99,102,241,0.45)] border border-transparent"
              >
                <span className="absolute inset-0 rounded-full border-2 border-white/30 blur-0 animate-pulse" />
                <span className="relative">Try Questions</span>
              </Link>

              <Link
                href="/training"
                className="relative inline-flex items-center justify-center px-8 py-4 rounded-full border border-white/30 text-lg font-semibold hover:border-purple-400/80"
              >
                <span className="absolute inset-0 rounded-full border border-purple-400/60 opacity-0 hover:opacity-100 transition" />
                <span className="relative">Watch the AI Learn</span>
              </Link>
            </div>

            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 border border-white/20 text-sm text-white/70">
              <span className="h-2 w-2 rounded-full bg-gradient-to-r from-purple-400 to-blue-400" />
              Powered by Reinforcement Learning + LLMs
            </div>
          </motion.div>
        </section>

        <section className="px-6 py-12">
          <div className="max-w-5xl mx-auto">
            <motion.div
              className="grid md:grid-cols-3 gap-6"
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.7 }}
            >
              {productCards.map((card) => (
                <div
                  key={card.title}
                  className={`p-6 rounded-3xl border border-white/10 bg-white/5 backdrop-blur hover:translate-y-[-4px] transition-all shadow-[0_20px_60px_rgba(15,23,42,0.5)] relative overflow-hidden`}
                >
                  <div className={`absolute inset-0 bg-gradient-to-br ${card.accent} opacity-30 blur-3xl`} />
                  <div className="relative space-y-3">
                    <p className="text-sm uppercase tracking-[0.35em] text-white/60">{card.title}</p>
                    <p className="text-lg font-semibold text-white">{card.description}</p>
                    <div className="inline-flex items-center space-x-2 text-sm text-blue-200">
                      <span>Explore</span>
                      <span className="text-blue-100">→</span>
                    </div>
                  </div>
                </div>
              ))}
            </motion.div>
          </div>
        </section>

        <section className="px-6 py-12">
          <div className="max-w-6xl mx-auto">
            <h3 className="text-sm uppercase text-white/60 tracking-[0.4em] mb-3">Testimonials</h3>
            <div className="grid md:grid-cols-3 gap-6">
              {testimonials.map((item) => (
                <motion.div
                  key={item.author}
                  className="glass rounded-3xl p-6 border border-white/10"
                  whileHover={{ y: -6 }}
                  transition={{ duration: 0.3 }}
                >
                  <p className="text-lg text-white/90 mb-4">“{item.quote}”</p>
                  <p className="text-sm text-white/60 font-semibold">{item.author}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        <section className="px-6 py-12">
          <div className="max-w-4xl mx-auto text-center space-y-4">
            <p className="text-sm uppercase text-white/60 tracking-[0.4em]">Why MentorFlow</p>
            <h3 className="text-3xl font-bold">Premium AI tutoring backed by live RL metrics</h3>
            <p className="text-white/70">Experience a system that monitors you, explains answers, and queues the optimal next question.</p>
          </div>
        </section>
      </main>
    </div>
  );
}
