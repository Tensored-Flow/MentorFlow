"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";

export default function CTA() {
  return (
    <section className="py-24 px-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="max-w-4xl mx-auto text-center"
      >
        <div className="p-12 rounded-3xl glass relative overflow-hidden">
          {/* Background gradient */}
          <div className="absolute inset-0 bg-gradient-to-br from-purple-600/20 to-blue-600/20" />
          
          <div className="relative z-10">
            <h2 className="text-3xl md:text-5xl font-bold mb-4">
              Ready To Learn <span className="gradient-text">Smarter</span>?
            </h2>
            <p className="text-gray-400 text-lg mb-8 max-w-2xl mx-auto">
              Join thousands of learners using AI-powered adaptive curriculum to master coding and STEM.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link
                href="/practice"
                className="group px-8 py-4 rounded-full bg-gradient-to-r from-purple-500 to-blue-500 text-white font-semibold text-lg hover:opacity-90 transition-all flex items-center space-x-2"
              >
                <span>Get Started</span>
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link
                href="/dashboard"
                className="px-8 py-4 rounded-full border border-white/20 text-white font-semibold text-lg hover:bg-white/5 transition-all"
              >
                Pick Up Where You Left Off
              </Link>
            </div>
          </div>
        </div>
      </motion.div>
    </section>
  );
}
