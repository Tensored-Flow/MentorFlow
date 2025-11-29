"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, Sparkles } from "lucide-react";

export default function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 via-transparent to-blue-900/20" />
      
      {/* Animated orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/30 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/30 rounded-full blur-3xl animate-pulse" style={{ animationDelay: "1s" }} />

      <div className="relative z-10 max-w-5xl mx-auto px-6 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          {/* Badge */}
          <div className="inline-flex items-center space-x-2 px-4 py-2 rounded-full glass mb-8">
            <Sparkles className="w-4 h-4 text-purple-400" />
            <span className="text-sm text-gray-300">Powered by Reinforcement Learning</span>
          </div>

          {/* Headline */}
          <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
            <span className="gradient-text">MentorFlow</span>
          </h1>
          <h2 className="text-3xl md:text-5xl font-bold mb-6 text-white/90">
            The AI Teacher That Learns to Teach.
          </h2>

          {/* Subtext */}
          <p className="text-lg md:text-xl text-gray-400 max-w-3xl mx-auto mb-10 leading-relaxed">
            Experience the same adaptive curriculum system our RL agents use — 
            personalised coding & STEM microtasks powered by Teacher + Student AI.
          </p>

          {/* CTAs */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="/practice"
              className="group px-8 py-4 rounded-full bg-gradient-to-r from-purple-500 to-blue-500 text-white font-semibold text-lg hover:opacity-90 transition-all flex items-center space-x-2"
            >
              <span>Try Questions Now</span>
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              href="/training"
              className="px-8 py-4 rounded-full border border-white/20 text-white font-semibold text-lg hover:bg-white/5 transition-all"
            >
              Watch the AI Learn
            </Link>
          </div>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1, duration: 0.6 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
        >
          <div className="w-6 h-10 border-2 border-white/30 rounded-full flex justify-center pt-2">
            <motion.div
              animate={{ y: [0, 8, 0] }}
              transition={{ repeat: Infinity, duration: 1.5 }}
              className="w-1.5 h-1.5 bg-white rounded-full"
            />
          </div>
        </motion.div>
      </div>
    </section>
  );
}
