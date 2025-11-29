"use client";

import { motion } from "framer-motion";
import { Bot, Brain, BookOpen, LineChart, Lightbulb, Zap } from "lucide-react";

const features = [
  {
    icon: Bot,
    title: "Adaptive AI Teacher",
    description: "UCB & Thompson Sampling bandit algorithms optimise your learning path in real-time.",
  },
  {
    icon: Brain,
    title: "PPO Student Learner",
    description: "Watch a neural network student learn coding tasks using Proximal Policy Optimisation.",
  },
  {
    icon: BookOpen,
    title: "Procedural Questions",
    description: "Infinite unique coding & STEM questions generated on-the-fly across multiple difficulty levels.",
  },
  {
    icon: LineChart,
    title: "Learning Analytics",
    description: "Visualise accuracy curves, RL rewards, and curriculum heatmaps as training progresses.",
  },
  {
    icon: Lightbulb,
    title: "Emergent Sequencing",
    description: "The teacher discovers optimal task orderings that accelerate student learning.",
  },
  {
    icon: Zap,
    title: "Instant Practice",
    description: "Jump straight into practising the same tasks our AI learns — no setup required.",
  },
];

export default function Features() {
  return (
    <section className="py-24 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Why <span className="gradient-text">MentorFlow</span>?
          </h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            A complete meta-curriculum learning system where the teacher learns to teach.
          </p>
        </motion.div>

        {/* Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="group p-6 rounded-2xl glass hover:bg-white/10 transition-all duration-300"
            >
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <feature.icon className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
              <p className="text-gray-400">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
